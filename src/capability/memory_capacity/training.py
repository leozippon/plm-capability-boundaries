"""GPU-resident AR and OA-ARDM MLM training loop with DDP, compile, and resume support."""
from __future__ import annotations

import contextlib
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from ..core.io import sha256_file
from ..memory_theory.io_utils import load_json_file, write_json_with_sha256
from .data import generate_iid_aa_dataset
from .evaluation import (
    DEFAULT_EVAL_MAX_SEQUENCES_MLM,
    DEFAULT_LOO_MAX_SEQUENCES_MLM,
    DEFAULT_LOO_PROBES_PER_SEQ,
    evaluate_mlm_loo_diagnostic,
    evaluate_run_checkpoint,
)
from .model import UniformVocabTransformer, count_model_parameters
from .spec import (
    C_GRID_PRIOR_BITS_PER_PARAM,
    LN2,
    SCALE_SPECS,
    SEQ_LEN,
    SPECIAL_TOKEN_ID,
    VOCAB_AA,
    RunSpec,
    assert_no_forbidden_infra_strings,
    choose_global_batch_size,
    dataset_entropy_bits,
    gibbs_memorizable_bits,
    null_ordered_entropy_bits,
    slugify_token,
)

RUN_RECEIPT_SCHEMA = "plm_parametric_capacity_run_v1"
CHECKPOINT_SCHEMA = "plm_parametric_capacity_ckpt_v1"


@dataclass(frozen=True)
class DistributedContext:
    """Active single-process or torchrun DDP execution context."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_distributed: bool


def setup_distributed(device: torch.device | str | None = None) -> DistributedContext:
    """Resolve single-GPU or initialize NCCL process group under torchrun."""
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(
            os.environ.get("LOCAL_RANK", str(rank % max(1, torch.cuda.device_count())))
        )
        dev = (
            torch.device("cuda", local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        return DistributedContext(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=dev,
            is_distributed=world_size > 1,
        )

    if env_world > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training with WORLD_SIZE > 1 requires CUDA")
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=dev)
        return DistributedContext(
            rank=rank,
            local_rank=local_rank,
            world_size=env_world,
            device=dev,
            is_distributed=True,
        )

    if device is not None:
        dev = torch.device(device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda", 0)
    else:
        dev = torch.device("cpu")

    if dev.type == "cuda" and dev.index is not None:
        torch.cuda.set_device(dev.index)

    return DistributedContext(
        rank=0,
        local_rank=dev.index or 0,
        world_size=1,
        device=dev,
        is_distributed=False,
    )


def compute_peak_lr(base_lr: float, global_batch_size: int) -> float:
    """Batch-size square-root scaled peak learning rate clamped to [0.5x, 2.0x] base_lr."""
    scale = min(2.0, max(0.5, math.sqrt(float(global_batch_size) / 256.0)))
    return float(base_lr) * scale


def compute_warmup_steps(total_steps: int) -> int:
    """Linear warmup steps: min(max(20, int(0.03 * total_steps)), 2000), clamped to total_steps."""
    raw = min(max(20, int(0.03 * float(total_steps))), 2000)
    return min(raw, max(1, int(total_steps)))


def compute_cosine_lr(
    step_1based: int,
    *,
    total_steps: int,
    peak_lr: float,
    warmup_steps: int,
) -> float:
    """Linear warmup to peak_lr followed by cosine decay to min_lr = 0.05 * peak_lr."""
    min_lr = 0.05 * float(peak_lr)
    if step_1based <= warmup_steps:
        return float(peak_lr) * (float(step_1based) / float(max(1, warmup_steps)))
    if total_steps <= warmup_steps:
        return min_lr
    progress = float(step_1based - warmup_steps) / float(total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (float(peak_lr) - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_optimizer(
    model: nn.Module,
    *,
    peak_lr: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.AdamW:
    """Create fused AdamW with weight_decay restricted strictly to >=2D weight matrices."""
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    use_fused = device.type == "cuda"
    return torch.optim.AdamW(
        param_groups,
        lr=float(peak_lr),
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused,
    )


def sample_mlm_stratified_batch(
    batch_seqs: torch.Tensor,
    *,
    generator: torch.Generator,
    pos_arange: torch.Tensor,
    special_token_id: int = SPECIAL_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stratified OA-ARDM / MDLM batch corruption returning (masked_input, targets, mask, n_mask)."""
    bsz, seq_len = batch_seqs.shape
    device = batch_seqs.device
    targets = batch_seqs.to(dtype=torch.long)

    perm = torch.randperm(bsz, device=device, generator=generator, dtype=torch.long).float()
    jitter = torch.rand(bsz, device=device, generator=generator, dtype=torch.float32)
    u = (perm + jitter) / float(bsz)
    n_mask = torch.clamp((u * float(seq_len)).long() + 1, 1, seq_len)

    noise = torch.rand((bsz, seq_len), device=device, generator=generator, dtype=torch.float32)
    order = torch.argsort(noise, dim=-1)
    mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=device)
    mask.scatter_(1, order, pos_arange.unsqueeze(0) < n_mask.unsqueeze(1))

    masked_input = torch.where(mask, int(special_token_id), targets)
    return masked_input, targets, mask, n_mask


def compute_ar_batch_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    vocab_size: int = VOCAB_AA,
) -> torch.Tensor:
    """Exact fp32 token-averaged autoregressive cross-entropy in nats."""
    return F.cross_entropy(
        logits.float().reshape(-1, int(vocab_size)),
        targets.reshape(-1),
        reduction="mean",
    )


def compute_mlm_batch_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    n_mask: torch.Tensor,
) -> torch.Tensor:
    """Exact unbiased OA-ARDM / MDLM joint-NELBO gradient estimator (in nats per token)."""
    tok_ce = F.cross_entropy(
        logits.float().transpose(1, 2),
        targets,
        reduction="none",
    )
    masked_ce = torch.where(mask, tok_ce, torch.zeros_like(tok_ce))
    per_row = masked_ce.sum(dim=-1) / n_mask.float()
    return per_row.mean()


def find_resume_checkpoint(
    checkpoint_dir: Path,
    run_spec: RunSpec,
    *,
    explicit_checkpoint: Path | None = None,
) -> Path | None:
    """Locate the highest-epoch valid checkpoint with epoch_done < max_epochs for resuming."""
    if explicit_checkpoint is not None:
        cand = Path(explicit_checkpoint)
        if not cand.is_file():
            raise FileNotFoundError(f"Explicit resume checkpoint not found: {cand}")
        return cand

    if not checkpoint_dir.is_dir():
        return None

    latest_ckpt = checkpoint_dir / f"{run_spec.run_id}_latest.pt"
    if latest_ckpt.is_file():
        return latest_ckpt

    prefix = (
        f"{run_spec.track}_{run_spec.paradigm}_{slugify_token(run_spec.scale_id)}"
        f"_r{slugify_token(f'{run_spec.multiplier:.2f}')}"
    )
    epoch_re = re.compile(r"_e(\d+)\.pt$")
    best_epoch = -1
    best_path: Path | None = None

    for pattern in (f"{run_spec.run_id}_e*.pt", f"{prefix}_e*.pt"):
        for path in checkpoint_dir.glob(pattern):
            match = epoch_re.search(path.name)
            if match is None:
                continue
            ep = int(match.group(1))
            if 0 < ep < run_spec.max_epochs and ep > best_epoch:
                best_epoch = ep
                best_path = path

    return best_path


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _verified_existing_completed_run(
    run_json_path: Path,
    run_spec: RunSpec,
    final_ckpt_path: Path | None,
) -> dict[str, Any] | None:
    sidecar = run_json_path.with_name(run_json_path.name + ".sha256")
    if not (run_json_path.is_file() and sidecar.is_file()):
        return None
    expected_line = sidecar.read_text(encoding="utf-8").strip().split()
    if not expected_line or expected_line[0] != sha256_file(run_json_path):
        return None
    payload = load_json_file(run_json_path)
    if (
        isinstance(payload, dict)
        and payload.get("schema") == RUN_RECEIPT_SCHEMA
        and payload.get("status") == "complete"
        and int(payload.get("epoch_done", -1)) == run_spec.max_epochs
    ):
        if (
            run_spec.save_checkpoint
            and final_ckpt_path is not None
            and not final_ckpt_path.is_file()
        ):
            return None
        return payload
    return None


def _build_run_payload(
    *,
    run_spec: RunSpec,
    scale_dict: dict[str, Any],
    param_counts: dict[str, int],
    seq_len: int,
    h_null_bits: float,
    h_data_bits: float,
    h_gibbs_bits: float,
    target_grid_bits: float,
    training_meta: dict[str, Any],
    epoch_done: int,
    elapsed_wall: float,
    init_eval: dict[str, Any] | None,
    exposures: list[dict[str, Any]],
    mlm_loo_diag: dict[str, Any] | None,
    is_complete: bool,
) -> dict[str, Any]:
    return {
        "schema": RUN_RECEIPT_SCHEMA,
        "status": "complete" if is_complete else "in_progress",
        "run_id": run_spec.run_id,
        "track": run_spec.track,
        "paradigm": run_spec.paradigm,
        "scale_id": run_spec.scale_id,
        "multiplier": float(run_spec.multiplier),
        "n_sequences": int(run_spec.n_sequences),
        "seq_len": int(seq_len),
        "vocab_size": VOCAB_AA,
        "n_params_non_embedding": int(param_counts["n_params_non_embedding"]),
        "n_params_embedding": int(param_counts["n_params_embedding"]),
        "n_params_total": int(param_counts["n_params_total"]),
        "weight_decay": float(run_spec.weight_decay),
        "precision": run_spec.precision,
        "batch_size_global": int(training_meta["global_batch_size"]),
        "peak_lr": float(training_meta["peak_lr"]),
        "min_lr": float(training_meta["min_lr"]),
        "warmup_steps": int(training_meta["warmup_steps"]),
        "total_steps": int(training_meta["total_steps"]),
        "max_epochs": int(run_spec.max_epochs),
        "seed": int(run_spec.seed),
        "data_seed": int(run_spec.data_seed),
        "world_size": int(training_meta["world_size"]),
        "H_null_ordered_bits": float(h_null_bits),
        "H_data_bits": float(h_data_bits),
        "H_gibbs_max_bits": float(h_gibbs_bits),
        "target_grid_bits": float(target_grid_bits),
        "run_spec": run_spec.to_dict(),
        "scale_spec": scale_dict,
        "parameter_counts": param_counts,
        "training_config": training_meta,
        "epoch_done": int(epoch_done),
        "wall_time_sec": float(elapsed_wall),
        "init_eval": init_eval,
        "exposures": exposures,
        "evaluations": exposures,
        "final_eval": exposures[-1] if exposures else None,
        "mlm_loo_diagnostic": mlm_loo_diag,
    }


def execute_run(
    run_spec: RunSpec,
    out_dir: Path | str,
    *,
    checkpoint_dir: Path | str | None = None,
    data_cache_dir: Path | str | None = None,
    device: torch.device | str | None = None,
    use_compile: bool = True,
    resume: bool = True,
    resume_from: Path | str | None = None,
    skip_completed: bool | None = None,
    run_init_eval: bool = True,
    seq_len: int = SEQ_LEN,
    batch_size_override: int | None = None,
    eval_max_sequences_mlm: int = DEFAULT_EVAL_MAX_SEQUENCES_MLM,
    loo_max_sequences_mlm: int = DEFAULT_LOO_MAX_SEQUENCES_MLM,
    loo_probes_per_seq: int = DEFAULT_LOO_PROBES_PER_SEQ,
) -> dict[str, Any]:
    """Execute one AR or MLM parametric memory-capacity run end-to-end."""
    ctx = setup_distributed(device=device)
    out_root = Path(out_dir)
    runs_dir = out_root / "runs"
    ckpt_root = Path(checkpoint_dir) if checkpoint_dir is not None else (out_root / "checkpoints")
    run_json_path = runs_dir / f"{run_spec.run_id}.json"
    final_ckpt_path = ckpt_root / f"{run_spec.run_id}_e{run_spec.max_epochs}.pt"
    latest_ckpt_path = ckpt_root / f"{run_spec.run_id}_latest.pt"

    eff_skip_completed = bool(resume) if skip_completed is None else bool(skip_completed)
    if eff_skip_completed:
        existing = _verified_existing_completed_run(run_json_path, run_spec, final_ckpt_path)
        if existing is not None:
            if ctx.is_distributed:
                dist.barrier()
            return existing

    scale_spec = SCALE_SPECS[run_spec.scale_id]
    if batch_size_override is not None:
        if batch_size_override < ctx.world_size or batch_size_override % ctx.world_size != 0:
            raise ValueError(
                f"batch_size_override ({batch_size_override}) must be a positive multiple of "
                f"world_size ({ctx.world_size})"
            )
        global_batch_size = int(batch_size_override)
    else:
        global_batch_size = choose_global_batch_size(
            run_spec.n_sequences,
            scale_spec.max_batch_size,
            world_size=ctx.world_size,
        )

    micro_batch_size = global_batch_size // ctx.world_size
    steps_per_epoch = math.ceil(float(run_spec.n_sequences) / float(global_batch_size))
    total_steps = steps_per_epoch * run_spec.max_epochs
    peak_lr = compute_peak_lr(scale_spec.base_lr, global_batch_size)
    min_lr = 0.05 * peak_lr
    warmup_steps = compute_warmup_steps(total_steps)
    eval_batch_size = max(micro_batch_size, min(scale_spec.max_batch_size, 512))

    prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
    if run_spec.precision == "bf16":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_autocast = ctx.device.type == "cuda"
    elif run_spec.precision == "fp32_strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        use_autocast = False
    else:
        raise ValueError(f"Unsupported precision {run_spec.precision!r}")

    try:
        # Stage deterministic uint8 dataset onto GPU memory with zero CPU-GPU DataLoader overhead
        if ctx.is_distributed and ctx.device.type == "cuda":
            if ctx.rank == 0:
                data_np = generate_iid_aa_dataset(
                    run_spec.n_sequences,
                    seq_len=seq_len,
                    vocab_size=VOCAB_AA,
                    data_seed=run_spec.data_seed,
                    data_cache_dir=data_cache_dir,
                )
                data_gpu = torch.from_numpy(data_np).to(device=ctx.device, dtype=torch.uint8)
                del data_np
            else:
                data_gpu = torch.empty(
                    (run_spec.n_sequences, seq_len),
                    dtype=torch.uint8,
                    device=ctx.device,
                )
            dist.broadcast(data_gpu, src=0)
        else:
            data_np = generate_iid_aa_dataset(
                run_spec.n_sequences,
                seq_len=seq_len,
                vocab_size=VOCAB_AA,
                data_seed=run_spec.data_seed,
                data_cache_dir=data_cache_dir,
            )
            data_gpu = torch.from_numpy(data_np).to(device=ctx.device, dtype=torch.uint8)
            del data_np

        torch.manual_seed(run_spec.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_spec.seed)

        raw_model = UniformVocabTransformer(
            scale_spec,
            run_spec.paradigm,
            seq_len=seq_len,
        ).to(ctx.device)
        param_counts = count_model_parameters(raw_model, scale_spec)
        optimizer = build_optimizer(
            raw_model,
            peak_lr=peak_lr,
            weight_decay=run_spec.weight_decay,
            device=ctx.device,
        )

        start_epoch = 1
        global_step = 0
        prior_wall_time = 0.0
        init_eval: dict[str, Any] | None = None
        exposures: list[dict[str, Any]] = []

        if resume or resume_from is not None:
            ckpt_to_load = find_resume_checkpoint(
                ckpt_root,
                run_spec,
                explicit_checkpoint=Path(resume_from) if resume_from is not None else None,
            )
            if ckpt_to_load is not None:
                ckpt = torch.load(ckpt_to_load, map_location=ctx.device, weights_only=False)
                ckpt_epoch = int(ckpt["epoch_done"])
                if ckpt_epoch < run_spec.max_epochs:
                    raw_model.load_state_dict(ckpt["model_state_dict"])
                    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    start_epoch = ckpt_epoch + 1
                    global_step = int(ckpt.get("global_step", ckpt_epoch * steps_per_epoch))
                    prior_wall_time = float(ckpt.get("elapsed_wall_time_sec", 0.0))
                    init_eval = ckpt.get("init_eval")
                    loaded_exps = ckpt.get("exposures", ckpt.get("evaluations", []))
                    exposures = [
                        rec for rec in loaded_exps if int(rec["epoch"]) <= ckpt_epoch
                    ]

        eval_model: nn.Module = torch.compile(raw_model) if use_compile else raw_model
        if ctx.is_distributed:
            train_model: nn.Module = DDP(
                eval_model,
                device_ids=[ctx.local_rank],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        else:
            train_model = eval_model

        t_run_start = time.perf_counter()

        if run_init_eval and init_eval is None and start_epoch == 1:
            init_metrics = evaluate_run_checkpoint(
                eval_model,
                data_gpu,
                paradigm=run_spec.paradigm,
                batch_size=eval_batch_size,
                n_params_non_embedding=param_counts["n_params_non_embedding"],
                n_params_total=param_counts["n_params_total"],
                precision=run_spec.precision,
                eval_max_sequences_mlm=eval_max_sequences_mlm,
                eval_seed=run_spec.data_seed + 17,
            )
            init_metrics["epoch"] = 0
            init_metrics["step"] = 0
            init_metrics["tokens_seen"] = 0
            init_eval = init_metrics

        eval_epoch_set = set(run_spec.eval_epochs)
        padded_epoch_items = steps_per_epoch * global_batch_size
        pad_needed = padded_epoch_items - run_spec.n_sequences
        bos_col = torch.full(
            (micro_batch_size, 1),
            SPECIAL_TOKEN_ID,
            dtype=torch.long,
            device=ctx.device,
        )
        pos_arange = torch.arange(seq_len, device=ctx.device, dtype=torch.int64)
        autocast_ctx: Any = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if use_autocast
            else contextlib.nullcontext()
        )

        h_null_bits = null_ordered_entropy_bits(
            run_spec.n_sequences, seq_len=seq_len, vocab_size=VOCAB_AA
        )
        h_data_bits = dataset_entropy_bits(
            run_spec.n_sequences, seq_len=seq_len, vocab_size=VOCAB_AA
        )
        h_gibbs_bits = gibbs_memorizable_bits(
            run_spec.n_sequences, seq_len=seq_len, vocab_size=VOCAB_AA
        )
        target_grid_bits = (
            float(run_spec.multiplier)
            * C_GRID_PRIOR_BITS_PER_PARAM
            * float(scale_spec.exact_non_embedding_params)
        )
        scale_dict = scale_spec.to_dict()
        training_meta = {
            "world_size": int(ctx.world_size),
            "global_batch_size": int(global_batch_size),
            "micro_batch_size": int(micro_batch_size),
            "steps_per_epoch": int(steps_per_epoch),
            "total_steps": int(total_steps),
            "warmup_steps": int(warmup_steps),
            "peak_lr": float(peak_lr),
            "min_lr": float(min_lr),
            "weight_decay": float(run_spec.weight_decay),
            "precision": run_spec.precision,
            "use_compile": bool(use_compile),
            "eval_max_sequences_mlm": int(eval_max_sequences_mlm),
        }

        mlm_loo_diag: dict[str, Any] | None = None
        last_lr = peak_lr

        for epoch in range(start_epoch, run_spec.max_epochs + 1):
            train_model.train()

            epoch_perm_gen = torch.Generator(device=ctx.device)
            epoch_perm_gen.manual_seed(int(run_spec.seed) * 1_000_003 + int(epoch))
            perm = torch.randperm(
                run_spec.n_sequences,
                generator=epoch_perm_gen,
                device=ctx.device,
            )
            if pad_needed > 0:
                repeats = (pad_needed + run_spec.n_sequences - 1) // run_spec.n_sequences
                extra = perm.repeat(repeats)[:pad_needed]
                perm = torch.cat((perm, extra), dim=0)
            perm_steps = perm.view(steps_per_epoch, global_batch_size)

            mask_gen = torch.Generator(device=ctx.device)
            mask_gen.manual_seed(
                int(run_spec.seed) * 10_000_019 + int(epoch) * 1_009 + int(ctx.rank)
            )

            epoch_loss_sum = torch.zeros((), dtype=torch.float64, device=ctx.device)

            for step_in_epoch in range(steps_per_epoch):
                global_step += 1
                last_lr = compute_cosine_lr(
                    global_step,
                    total_steps=total_steps,
                    peak_lr=peak_lr,
                    warmup_steps=warmup_steps,
                )
                for group in optimizer.param_groups:
                    group["lr"] = last_lr

                step_indices = perm_steps[step_in_epoch, ctx.rank :: ctx.world_size]
                batch_u8 = data_gpu.index_select(0, step_indices)

                optimizer.zero_grad(set_to_none=True)

                if run_spec.paradigm == "ar":
                    targets = batch_u8.to(dtype=torch.long)
                    input_ids = torch.cat((bos_col, targets[:, :-1]), dim=1)
                    with autocast_ctx:
                        logits = train_model(input_ids)
                    loss = compute_ar_batch_loss(logits, targets, vocab_size=VOCAB_AA)
                else:
                    masked_input, targets, mask, n_mask = sample_mlm_stratified_batch(
                        batch_u8,
                        generator=mask_gen,
                        pos_arange=pos_arange,
                        special_token_id=SPECIAL_TOKEN_ID,
                    )
                    with autocast_ctx:
                        logits = train_model(masked_input)
                    loss = compute_mlm_batch_loss(logits, targets, mask, n_mask)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss_sum += loss.detach().to(torch.float64)

            if epoch in eval_epoch_set or epoch == run_spec.max_epochs:
                if ctx.is_distributed:
                    dist.all_reduce(epoch_loss_sum, op=dist.ReduceOp.AVG)
                mean_train_loss_nats = float(epoch_loss_sum.item()) / float(steps_per_epoch)
                mean_train_loss_bits = mean_train_loss_nats / LN2

                eval_rec = evaluate_run_checkpoint(
                    eval_model,
                    data_gpu,
                    paradigm=run_spec.paradigm,
                    batch_size=eval_batch_size,
                    n_params_non_embedding=param_counts["n_params_non_embedding"],
                    n_params_total=param_counts["n_params_total"],
                    precision=run_spec.precision,
                    eval_max_sequences_mlm=eval_max_sequences_mlm,
                    eval_seed=run_spec.data_seed + 17,
                )

                if epoch == run_spec.max_epochs and run_spec.paradigm == "mlm":
                    mlm_loo_diag = evaluate_mlm_loo_diagnostic(
                        eval_model,
                        data_gpu,
                        batch_size=eval_batch_size,
                        n_params_non_embedding=param_counts["n_params_non_embedding"],
                        precision=run_spec.precision,
                        max_sequences_loo=loo_max_sequences_mlm,
                        probes_per_sequence=loo_probes_per_seq,
                        loo_seed=run_spec.data_seed + 31,
                    )
                    eval_rec.update(mlm_loo_diag)

                elapsed_wall = prior_wall_time + (time.perf_counter() - t_run_start)
                eval_rec["epoch"] = int(epoch)
                eval_rec["step"] = int(global_step)
                eval_rec["global_step"] = int(global_step)
                eval_rec["tokens_seen"] = int(epoch) * int(run_spec.n_sequences) * int(seq_len)
                eval_rec["lr"] = float(last_lr)
                eval_rec["elapsed_sec"] = float(elapsed_wall)
                eval_rec["train_loss_nats"] = float(mean_train_loss_nats)
                eval_rec["train_loss_last_epoch_bits_per_tok"] = float(mean_train_loss_bits)

                exposures.append(eval_rec)
                is_done = epoch == run_spec.max_epochs

                if ctx.rank == 0:
                    payload = _build_run_payload(
                        run_spec=run_spec,
                        scale_dict=scale_dict,
                        param_counts=param_counts,
                        seq_len=seq_len,
                        h_null_bits=h_null_bits,
                        h_data_bits=h_data_bits,
                        h_gibbs_bits=h_gibbs_bits,
                        target_grid_bits=target_grid_bits,
                        training_meta=training_meta,
                        epoch_done=epoch,
                        elapsed_wall=elapsed_wall,
                        init_eval=init_eval,
                        exposures=exposures,
                        mlm_loo_diag=mlm_loo_diag,
                        is_complete=is_done,
                    )
                    assert_no_forbidden_infra_strings(payload, label=run_spec.run_id)
                    write_json_with_sha256(run_json_path, payload)

                    if run_spec.save_checkpoint:
                        ckpt_payload = {
                            "schema": CHECKPOINT_SCHEMA,
                            "run_spec": run_spec.to_dict(),
                            "epoch_done": int(epoch),
                            "global_step": int(global_step),
                            "model_state_dict": raw_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "init_eval": init_eval,
                            "exposures": exposures,
                            "elapsed_wall_time_sec": float(elapsed_wall),
                        }
                        if is_done:
                            _atomic_torch_save(final_ckpt_path, ckpt_payload)
                            if latest_ckpt_path.exists():
                                latest_ckpt_path.unlink()
                        else:
                            _atomic_torch_save(latest_ckpt_path, ckpt_payload)

                if ctx.is_distributed:
                    dist.barrier()

        total_elapsed = prior_wall_time + (time.perf_counter() - t_run_start)
        return _build_run_payload(
            run_spec=run_spec,
            scale_dict=scale_dict,
            param_counts=param_counts,
            seq_len=seq_len,
            h_null_bits=h_null_bits,
            h_data_bits=h_data_bits,
            h_gibbs_bits=h_gibbs_bits,
            target_grid_bits=target_grid_bits,
            training_meta=training_meta,
            epoch_done=run_spec.max_epochs,
            elapsed_wall=total_elapsed,
            init_eval=init_eval,
            exposures=exposures,
            mlm_loo_diag=mlm_loo_diag,
            is_complete=True,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul
        torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn


train_memory_capacity_run = execute_run
run_training = execute_run
