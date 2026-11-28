"""Spot-safe controlled multimodal training engine (pure_seq, struct_cond, joint_to_seq)."""
from __future__ import annotations

import contextlib
import json
import math
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from ..memory_capacity.optimizers import build_named_optimizer, set_optimizer_lr
from ..supply_demand.corpus import atomic_write_json_with_sha256, sha256_file
from ..supply_demand.evaluation import (
    evaluate_proteingym_substitutions,
    evaluate_unseen_fold_contacts,
)
from ..supply_demand.objectives import compute_objective_loss, prepare_objective_batch
from ..supply_demand.training import (
    PREEMPT_EXIT_CODE,
    ActiveLease,
    SharedLeaseManager,
    _build_epoch_batches,
    atomic_torch_save,
    compute_lr,
    compute_peak_lr,
    compute_warmup_steps,
    setup_distributed,
)
from .data import Multimodal3DBundle, PackedMultimodalStore
from .evaluation import (
    evaluate_multimodal_megascale,
    evaluate_multimodal_three_tier,
    evaluate_sequence_to_structure_readout,
)
from .model import MultiTrackProteinTransformer, count_multimodal_parameters
from .spec import (
    CONTACT_DISTANCE_CUTOFF,
    LN2,
    SCALE_SPECS,
    STRUCT_ABSENT_TOKEN_ID,
    STRUCT_MASK_TOKEN_ID,
    STRUCT_PAD_TOKEN_ID,
    MultimodalRunSpec,
    objective_display_name,
    should_enable_gradient_checkpointing,
    vram_safe_micro_batch_cap,
)

MULTIMODAL_RECEIPT_SCHEMA = "multimodal_3d_run_v1"
MULTIMODAL_CKPT_SCHEMA = "multimodal_3d_ckpt_v1"


def is_multimodal_run_completed(runs_dir: Path, run_id: str) -> bool:
    """Verify completed multimodal receipt JSON + SHA-256 sidecar."""
    json_path = runs_dir / f"{run_id}.json"
    sha_path = json_path.with_name(json_path.name + ".sha256")
    if not (json_path.is_file() and sha_path.is_file()):
        return False
    parts = sha_path.read_text(encoding="utf-8").strip().split()
    if not parts or parts[0] != sha256_file(json_path):
        return False
    try:
        doc = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        isinstance(doc, dict)
        and doc.get("schema") == MULTIMODAL_RECEIPT_SCHEMA
        and doc.get("status") == "complete"
    )


def compute_multimodal_batch_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    run_spec: MultimodalRunSpec,
    generator: torch.Generator,
    autocast_ctx: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute exact controlled multimodal loss for pure_seq, struct_cond, or joint_to_seq."""
    tokens = batch["aa_tokens"]
    lengths = batch["lengths"]
    bsz, max_len = tokens.shape
    device = tokens.device

    prepared = prepare_objective_batch(
        run_spec.objective,
        tokens,
        lengths,
        mask_fraction=run_spec.effective_mask_rate,
        generator=generator,
    )
    valid = prepared.attention_mask

    if run_spec.mode == "pure_seq":
        with autocast_ctx:
            aa_logits = model(prepared.input_ids, valid, right_padded=True)
        l_seq = compute_objective_loss(aa_logits, prepared)
        return l_seq, {
            "loss_total": float(l_seq.detach().item()),
            "loss_seq": float(l_seq.detach().item()),
            "loss_struct": 0.0,
            "loss_contact": 0.0,
        }

    s_tokens = batch["struct_tokens"]
    geom_feats = batch["geom_feats"]
    cb_coords = batch["cb_coords"]
    cb_mask = batch["cb_mask"]

    if run_spec.mode == "struct_cond":
        s_present = torch.ones((bsz,), dtype=torch.bool, device=device)
        with autocast_ctx:
            aa_logits = model(
                prepared.input_ids,
                valid,
                struct_tokens=s_tokens,
                geom_feats=geom_feats,
                cb_coords=cb_coords,
                cb_mask=cb_mask,
                struct_present=s_present,
                right_padded=True,
            )
        l_seq = compute_objective_loss(aa_logits, prepared)
        return l_seq, {
            "loss_total": float(l_seq.detach().item()),
            "loss_seq": float(l_seq.detach().item()),
            "loss_struct": 0.0,
            "loss_contact": 0.0,
        }

    # Mode 3: joint_to_seq (ESM-3 paradigm!)
    # Independently drop the entire structure track per sequence with probability struct_drop_prob (default 50%)
    # so Z = empty is strictly in-distribution; for retained sequences, mask per-residue 3Di with struct_mask_prob.
    drop_draws = torch.rand((bsz,), device=device, generator=generator, dtype=torch.float32)
    s_present = drop_draws >= float(run_spec.struct_drop_prob)

    tok_mask_draws = torch.rand((bsz, max_len), device=device, generator=generator, dtype=torch.float32)
    per_res_struct_mask = (tok_mask_draws < float(run_spec.struct_mask_prob)) & valid
    # Dropped sequences have ALL valid 3Di positions supervised from pure AA sequence (Z = empty)!
    struct_loss_mask = ((~s_present).unsqueeze(1) | per_res_struct_mask) & valid

    s_input = torch.where(per_res_struct_mask, STRUCT_MASK_TOKEN_ID, s_tokens)
    s_input = torch.where(valid, s_input, STRUCT_PAD_TOKEN_ID)
    s_input = torch.where(s_present.unsqueeze(1), s_input, STRUCT_ABSENT_TOKEN_ID)

    # Zero out per-residue geom_feats at masked structure positions too so 3Di supervision isn't trivially leaked
    eff_geom = torch.where(
        (s_present.unsqueeze(1) & (~per_res_struct_mask) & valid).unsqueeze(-1),
        geom_feats,
        torch.zeros_like(geom_feats),
    )

    unwrapped = model.module if isinstance(model, DDP) else model
    # Call through DDP wrapper when distributed via forward_multimodal? DDP wraps __call__ -> forward.
    # For DDP compatibility we invoke a clean forward helper:
    with autocast_ctx:
        if isinstance(model, DDP):
            out = unwrapped.forward_multimodal(
                prepared.input_ids,
                valid,
                struct_tokens=s_input,
                geom_feats=eff_geom,
                cb_coords=cb_coords,
                cb_mask=cb_mask,
                struct_present=s_present,
                right_padded=True,
                return_struct_heads=True,
            )
        else:
            out = model.forward_multimodal(
                prepared.input_ids,
                valid,
                struct_tokens=s_input,
                geom_feats=eff_geom,
                cb_coords=cb_coords,
                cb_mask=cb_mask,
                struct_present=s_present,
                right_padded=True,
                return_struct_heads=True,
            )

    l_seq = compute_objective_loss(out.aa_logits, prepared)

    l_struct = torch.zeros((), dtype=torch.float32, device=device)
    if out.struct_3di_logits is not None and float(run_spec.lambda_struct) > 0.0:
        safe_s_tgt = torch.where(struct_loss_mask, s_tokens, torch.zeros_like(s_tokens))
        s_ce = F.cross_entropy(
            out.struct_3di_logits.float().transpose(1, 2),
            safe_s_tgt,
            reduction="none",
        )
        s_ce = torch.where(struct_loss_mask, s_ce, torch.zeros_like(s_ce))
        l_struct = s_ce.sum() / struct_loss_mask.sum().to(dtype=torch.float32).clamp(min=1.0)

    l_contact = torch.zeros((), dtype=torch.float32, device=device)
    if out.contact_logits is not None and float(run_spec.lambda_contact) > 0.0:
        # Supervise contact head strictly on sequences where structure was dropped (Z = empty)
        # OR on all sequences if none were dropped in this micro-batch, over |i-j| >= 6 pairs
        coords_f = cb_coords.float()
        diff = coords_f.unsqueeze(2) - coords_f.unsqueeze(1)
        dists = torch.sqrt((diff * diff).sum(dim=-1).clamp(min=0.0))
        c_targets = (dists < float(CONTACT_DISTANCE_CUTOFF)).float()

        idx_t = torch.arange(max_len, device=device)
        sep_mask = (idx_t.unsqueeze(0) - idx_t.unsqueeze(1)).abs() >= 6
        pair_mask = cb_mask.unsqueeze(2) & cb_mask.unsqueeze(1) & sep_mask.unsqueeze(0)
        dropped_seq = ~s_present
        if bool(dropped_seq.any().item()):
            pair_mask = pair_mask & dropped_seq.view(bsz, 1, 1)
        if bool(pair_mask.any().item()):
            c_bce = F.binary_cross_entropy_with_logits(
                out.contact_logits.float(), c_targets, reduction="none"
            )
            c_bce = torch.where(pair_mask, c_bce, torch.zeros_like(c_bce))
            l_contact = c_bce.sum() / pair_mask.sum().to(dtype=torch.float32).clamp(min=1.0)

    total_loss = (
        l_seq
        + float(run_spec.lambda_struct) * l_struct
        + float(run_spec.lambda_contact) * l_contact
    )
    return total_loss, {
        "loss_total": float(total_loss.detach().item()),
        "loss_seq": float(l_seq.detach().item()),
        "loss_struct": float(l_struct.detach().item()),
        "loss_contact": float(l_contact.detach().item()),
    }


def _sync_ddp_grads(raw_model: nn.Module, world_size: int) -> None:
    """Explicit gradient average across ranks for multi-track forward_multimodal DDP steps.

    Every trainable parameter participates in ``all_reduce`` in deterministic order (materializing
    a zero gradient whenever a rank skipped an optional branch such as structure dropout),
    preventing NCCL collective mismatches or rank weight divergence.
    """
    inv_ws = 1.0 / float(world_size)
    for p in raw_model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.mul_(inv_ws)


def execute_multimodal_run(
    run_spec: MultimodalRunSpec,
    bundle: Multimodal3DBundle | Path | str,
    out_dir: Path | str,
    *,
    device: torch.device | str | None = None,
    resume: bool = True,
    run_init_eval: bool = True,
    eval_batch_size: int | None = None,
    n_eval_draws: int = 16,
    loo_probes: int = 16,
    lease_manager: SharedLeaseManager | None = None,
    active_lease: ActiveLease | None = None,
    install_signal_handlers: bool = True,
    gradient_checkpointing: bool | None = None,
) -> dict[str, Any]:
    """Execute one controlled multimodal run (pure_seq, struct_cond, or joint_to_seq) end-to-end."""
    ctx = setup_distributed(device=device)
    dev = ctx.device
    b_obj = bundle if isinstance(bundle, Multimodal3DBundle) else Multimodal3DBundle(bundle)
    out_root = Path(out_dir)
    runs_dir = out_root / "runs"
    ckpt_dir = out_root / "checkpoints"
    if ctx.rank == 0:
        runs_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if ctx.is_distributed:
        dist.barrier()

    run_json_path = runs_dir / f"{run_spec.run_id}.json"
    latest_ckpt_path = ckpt_dir / f"{run_spec.run_id}_latest.pt"
    final_ckpt_path = ckpt_dir / f"{run_spec.run_id}_final.pt"

    if resume and is_multimodal_run_completed(runs_dir, run_spec.run_id):
        if ctx.is_distributed:
            dist.barrier()
        return json.loads(run_json_path.read_text(encoding="utf-8"))

    splits = b_obj.load_splits()
    train_store: PackedMultimodalStore = splits["train"]
    train_sub_store: PackedMultimodalStore = splits["eval_train_sub"]
    homolog_store: PackedMultimodalStore = splits["eval_unseen_homolog"]
    unseen_fold_store: PackedMultimodalStore = splits["eval_unseen_fold"]

    megascale_data = b_obj.load_megascale() if ctx.rank == 0 else None
    proteingym_data = b_obj.load_proteingym() if ctx.rank == 0 else None
    contact_domains = b_obj.load_contacts() if ctx.rank == 0 else []

    scale_spec = SCALE_SPECS[run_spec.scale_id]
    total_vram_bytes = (
        int(torch.cuda.get_device_properties(dev).total_memory) if dev.type == "cuda" else None
    )
    req_gc = gradient_checkpointing if gradient_checkpointing is not None else run_spec.gradient_checkpointing
    use_grad_ckpt = should_enable_gradient_checkpointing(scale_spec.scale_id, total_vram_bytes, requested=req_gc)
    vram_cap = vram_safe_micro_batch_cap(scale_spec.scale_id, total_vram_bytes, default_cap=int(run_spec.batch_size))
    requested_per_rank_bs = int(run_spec.batch_size)
    micro_bs = min(requested_per_rank_bs, vram_cap)
    split_factor = max(1, math.ceil(float(requested_per_rank_bs) / float(micro_bs)))
    grad_accum = int(run_spec.grad_accum_steps) * split_factor
    eff_bs = micro_bs * grad_accum * int(ctx.world_size)

    n_micro_total = math.ceil(float(train_store.n_sequences) / float(micro_bs))
    if ctx.is_distributed and (n_micro_total % ctx.world_size != 0):
        n_micro_total += ctx.world_size - (n_micro_total % ctx.world_size)
    n_micro_rank = n_micro_total // ctx.world_size
    steps_per_epoch = math.ceil(float(n_micro_rank) / float(grad_accum))
    total_steps = steps_per_epoch * int(run_spec.max_epochs)
    peak_lr = compute_peak_lr(scale_spec.base_lr, eff_bs, override=run_spec.peak_lr_override)
    warmup_steps = compute_warmup_steps(total_steps, override=run_spec.warmup_steps_override)
    eff_eval_bs = int(eval_batch_size) if eval_batch_size is not None else max(micro_bs, 64)

    prev_tf32_m = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_c = torch.backends.cudnn.allow_tf32
    if run_spec.precision == "bf16":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_autocast = dev.type == "cuda"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        use_autocast = False
    autocast_ctx: Any = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_autocast else contextlib.nullcontext()
    )

    preempt_state = {"requested": False, "signal": None}
    prev_sigterm = prev_sigint = None

    def _handle_preempt(signum: int, _frame: Any) -> None:
        preempt_state["requested"] = True
        preempt_state["signal"] = int(signum)

    if install_signal_handlers:
        prev_sigterm = signal.getsignal(signal.SIGTERM)
        prev_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGTERM, _handle_preempt)
        signal.signal(signal.SIGINT, _handle_preempt)

    try:
        torch.manual_seed(run_spec.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_spec.seed)

        raw_model = MultiTrackProteinTransformer(
            scale_spec,
            mode=run_spec.mode,
            objective=run_spec.objective,
            max_seq_len=run_spec.max_seq_len,
            use_pair_geom=run_spec.use_pair_geom,
            gradient_checkpointing=use_grad_ckpt,
        ).to(dev)
        param_counts = count_multimodal_parameters(raw_model, scale_spec)
        optimizer = build_named_optimizer(
            raw_model,
            optimizer_name=run_spec.optimizer,
            peak_lr=peak_lr,
            weight_decay=run_spec.weight_decay,
            device=dev,
            muon_lr=run_spec.muon_lr,
            muon_lr_scale=run_spec.muon_lr_scale,
        )

        cur_epoch = 1
        resume_step_in_epoch = 0
        global_step = 0
        tokens_seen = 0
        prior_wall_sec = 0.0
        init_eval: dict[str, Any] | None = None
        milestones: list[dict[str, Any]] = []
        epoch_loss_sum = 0.0
        epoch_loss_steps = 0

        if resume and latest_ckpt_path.is_file():
            ckpt = torch.load(latest_ckpt_path, map_location=dev, weights_only=False)
            saved_spec = ckpt.get("run_spec", {})
            for k in ("run_id", "mode", "objective", "scale_id", "seed"):
                if saved_spec.get(k) != getattr(run_spec, k):
                    raise ValueError(f"Multimodal checkpoint mismatch on {k!r}")
            raw_model.load_state_dict(ckpt["model_state_dict"])
            if ckpt.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            saved_ep = int(ckpt["epoch"])
            saved_st = int(ckpt["step_in_epoch"])
            global_step = int(ckpt["global_step"])
            tokens_seen = int(ckpt.get("tokens_seen", 0))
            prior_wall_sec = float(ckpt.get("elapsed_wall_sec", 0.0))
            init_eval = ckpt.get("init_eval")
            milestones = list(ckpt.get("milestones", []))
            if saved_st >= steps_per_epoch and bool(ckpt.get("eval_done", False)):
                cur_epoch = saved_ep + 1
                resume_step_in_epoch = 0
            else:
                cur_epoch = saved_ep
                resume_step_in_epoch = saved_st
                epoch_loss_sum = float(ckpt.get("epoch_loss_sum", 0.0))
                epoch_loss_steps = int(ckpt.get("epoch_loss_steps", 0))

        # Broadcast initial weights from rank 0 under DDP
        if ctx.is_distributed:
            for p in raw_model.parameters():
                dist.broadcast(p.data, src=0)

        t_run_start = time.perf_counter()
        last_ckpt_mono = time.monotonic()

        def _touch_lease() -> None:
            if ctx.rank == 0 and lease_manager is not None and active_lease is not None:
                lease_manager.heartbeat(active_lease)

        def _build_payload(*, status: str, epoch_done: int, step_in_epoch: int) -> dict[str, Any]:
            wall = prior_wall_sec + (time.perf_counter() - t_run_start)
            return {
                "schema": MULTIMODAL_RECEIPT_SCHEMA,
                "status": status,
                "run_id": run_spec.run_id,
                "mode": run_spec.mode,
                "objective": run_spec.objective,
                "paradigm_display": objective_display_name(run_spec.objective),
                "scale_id": run_spec.scale_id,
                "parameter_counts": param_counts,
                "scale_spec": scale_spec.to_dict(),
                "run_spec": run_spec.to_dict(),
                "corpus_stats": {
                    "n_train_sequences": train_store.n_sequences,
                    "n_train_unique_tokens": train_store.total_tokens,
                    "n_eval_train_sub": train_sub_store.n_sequences,
                    "n_eval_unseen_homolog": homolog_store.n_sequences,
                    "n_eval_unseen_fold": unseen_fold_store.n_sequences,
                },
                "training_config": {
                    "world_size": int(ctx.world_size),
                    "micro_batch_size": micro_bs,
                    "grad_accum_steps": grad_accum,
                    "effective_batch_size": eff_bs,
                    "steps_per_epoch": steps_per_epoch,
                    "total_steps": total_steps,
                    "peak_lr": peak_lr,
                    "warmup_steps": warmup_steps,
                    "optimizer": run_spec.optimizer,
                    "gradient_checkpointing": bool(use_grad_ckpt),
                    "precision": run_spec.precision,
                },
                "epoch_done": int(epoch_done),
                "step_in_epoch": int(step_in_epoch),
                "global_step": int(global_step),
                "tokens_seen": int(tokens_seen),
                "wall_time_sec": float(wall),
                "init_eval": init_eval,
                "milestones": milestones,
                "final_eval": milestones[-1] if milestones else init_eval,
            }

        def _save_ckpt(path: Path, *, epoch_num: int, step_in_ep: int, eval_done: bool) -> None:
            if ctx.rank == 0:
                atomic_torch_save(
                    path,
                    {
                        "schema": MULTIMODAL_CKPT_SCHEMA,
                        "run_spec": run_spec.to_dict(),
                        "epoch": int(epoch_num),
                        "step_in_epoch": int(step_in_ep),
                        "eval_done": bool(eval_done),
                        "global_step": int(global_step),
                        "tokens_seen": int(tokens_seen),
                        "elapsed_wall_sec": float(prior_wall_sec + (time.perf_counter() - t_run_start)),
                        "epoch_loss_sum": float(epoch_loss_sum),
                        "epoch_loss_steps": int(epoch_loss_steps),
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "init_eval": init_eval,
                        "milestones": milestones,
                    },
                )
            if ctx.is_distributed:
                dist.barrier()

        def _run_eval(
            *,
            epoch_num: int,
            is_final: bool,
            last_lr: float,
            mean_train_loss_nats: float | None,
            include_downstream: bool,
            include_contacts: bool,
        ) -> dict[str, Any] | None:
            rec: dict[str, Any] | None = None
            if ctx.rank == 0:
                _touch_lease()
                tt = evaluate_multimodal_three_tier(
                    raw_model,
                    train_sub_store=train_sub_store,
                    unseen_homolog_store=homolog_store,
                    unseen_fold_store=unseen_fold_store,
                    mode=run_spec.mode,
                    objective=run_spec.objective,
                    batch_size=eff_eval_bs,
                    precision=run_spec.precision,
                    n_draws=n_eval_draws,
                    loo_probes=loo_probes,
                    eval_seed=run_spec.seed + 7919,
                    mask_rate=run_spec.effective_mask_rate,
                    eval_temperature_calibration=bool(run_spec.eval_temperature_calibration),
                )
                rec = {
                    "epoch": int(epoch_num),
                    "global_step": int(global_step),
                    "tokens_seen": int(tokens_seen),
                    "lr": float(last_lr),
                    "elapsed_wall_sec": float(prior_wall_sec + (time.perf_counter() - t_run_start)),
                    "train_loss_nats": mean_train_loss_nats,
                    "train_loss_bits_per_res": (
                        None if mean_train_loss_nats is None else float(mean_train_loss_nats / LN2)
                    ),
                    "three_tier_bits": tt,
                }
                if run_spec.mode == "joint_to_seq":
                    rec["seq_to_struct_readout"] = evaluate_sequence_to_structure_readout(
                        raw_model,
                        unseen_fold_store,
                        batch_size=min(eff_eval_bs, 32),
                        precision=run_spec.precision,
                    )
                if include_downstream and megascale_data is not None:
                    _touch_lease()
                    rec["megascale"] = evaluate_multimodal_megascale(
                        raw_model,
                        megascale_data,
                        mode=run_spec.mode,
                        objective=run_spec.objective,
                        batch_size=eff_eval_bs,
                        precision=run_spec.precision,
                    )
                if include_downstream and proteingym_data is not None and run_spec.mode != "struct_cond":
                    _touch_lease()
                    rec["proteingym_milestone"] = evaluate_proteingym_substitutions(
                        raw_model,
                        proteingym_data,
                        objective=run_spec.objective,
                        milestone_only=True,
                        batch_size=eff_eval_bs,
                        precision=run_spec.precision,
                    )
                if include_contacts and contact_domains and run_spec.mode != "struct_cond":
                    _touch_lease()
                    rec["contacts"] = evaluate_unseen_fold_contacts(
                        raw_model,
                        contact_domains,
                        objective=run_spec.objective,
                        batch_size=eff_eval_bs,
                        precision=run_spec.precision,
                    )
                _touch_lease()
            if ctx.is_distributed:
                dist.barrier()
            return rec

        if run_init_eval and init_eval is None and cur_epoch == 1 and resume_step_in_epoch == 0:
            init_eval = _run_eval(
                epoch_num=0,
                is_final=False,
                last_lr=0.0,
                mean_train_loss_nats=None,
                include_downstream=False,
                include_contacts=False,
            )

        eval_epoch_set = set(run_spec.eval_epochs)
        last_lr = peak_lr

        for epoch in range(cur_epoch, run_spec.max_epochs + 1):
            raw_model.train()
            all_mbs = _build_epoch_batches(
                train_store.n_sequences, micro_bs, seed=run_spec.seed, epoch=epoch
            )
            if ctx.is_distributed:
                if len(all_mbs) < n_micro_total:
                    pad_c = n_micro_total - len(all_mbs)
                    all_mbs = all_mbs + [all_mbs[i % len(all_mbs)] for i in range(pad_c)]
                micro_batches = all_mbs[ctx.rank :: ctx.world_size]
            else:
                micro_batches = all_mbs

            step_start = resume_step_in_epoch if epoch == cur_epoch else 0
            if step_start == 0:
                epoch_loss_sum = 0.0
                epoch_loss_steps = 0

            for step_idx in range(step_start, steps_per_epoch):
                global_step += 1
                last_lr = compute_lr(
                    global_step,
                    total_steps=total_steps,
                    peak_lr=peak_lr,
                    warmup_steps=warmup_steps,
                    schedule=run_spec.lr_schedule,
                    cooldown_fraction=run_spec.cooldown_fraction,
                    min_lr_ratio=run_spec.min_lr_ratio,
                )
                set_optimizer_lr(optimizer, last_lr, peak_lr=peak_lr)

                mb_lo = step_idx * grad_accum
                mb_hi = min(len(micro_batches), mb_lo + grad_accum)
                actual_accum = max(1, mb_hi - mb_lo)

                optimizer.zero_grad(set_to_none=True)
                step_loss_val = 0.0
                step_tokens = 0

                for mb_off, mb_i in enumerate(range(mb_lo, mb_hi)):
                    batch = train_store.collate_indices(
                        micro_batches[mb_i], device=dev, pad_to_multiple=8
                    )
                    gen = torch.Generator(device=dev)
                    gen.manual_seed(
                        int(run_spec.seed) * 10_000_019
                        + int(epoch) * 100_003
                        + int(step_idx) * 97
                        + int(mb_off) * 17
                        + int(ctx.rank)
                    )
                    loss, _parts = compute_multimodal_batch_loss(
                        raw_model,
                        batch,
                        run_spec=run_spec,
                        generator=gen,
                        autocast_ctx=autocast_ctx,
                    )
                    (loss / float(actual_accum)).backward()
                    step_loss_val += float(loss.detach().item()) / float(actual_accum)
                    step_tokens += int(batch["lengths"].sum().item())

                if ctx.is_distributed:
                    _sync_ddp_grads(raw_model, ctx.world_size)

                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
                optimizer.step()

                if ctx.is_distributed:
                    st_t = torch.tensor([step_loss_val, float(step_tokens)], dtype=torch.float64, device=dev)
                    dist.all_reduce(st_t, op=dist.ReduceOp.SUM)
                    step_loss_val = float(st_t[0].item()) / float(ctx.world_size)
                    step_tokens = int(round(float(st_t[1].item())))

                tokens_seen += step_tokens
                epoch_loss_sum += step_loss_val
                epoch_loss_steps += 1
                completed_step = step_idx + 1

                should_check = (not ctx.is_distributed) or (
                    completed_step == steps_per_epoch or completed_step % 8 == 0
                )
                if should_check:
                    need_preempt = bool(preempt_state["requested"])
                    now_mono = time.monotonic()
                    need_periodic = (now_mono - last_ckpt_mono) >= run_spec.checkpoint_interval_sec
                    if ctx.is_distributed:
                        fl = torch.tensor(
                            [1 if need_preempt else 0, 1 if (ctx.rank == 0 and need_periodic) else 0],
                            dtype=torch.int64,
                            device=dev,
                        )
                        dist.all_reduce(fl, op=dist.ReduceOp.MAX)
                        need_preempt = bool(fl[0].item() > 0)
                        need_periodic = bool(fl[1].item() > 0)

                    if need_preempt:
                        _save_ckpt(latest_ckpt_path, epoch_num=epoch, step_in_ep=completed_step, eval_done=False)
                        payload = _build_payload(
                            status="preempted",
                            epoch_done=epoch - 1 if completed_step < steps_per_epoch else epoch,
                            step_in_epoch=completed_step,
                        )
                        payload["preempt_signal"] = preempt_state["signal"]
                        payload["exit_code"] = PREEMPT_EXIT_CODE
                        if ctx.rank == 0:
                            atomic_write_json_with_sha256(run_json_path, payload)
                            if lease_manager is not None and active_lease is not None:
                                lease_manager.release(active_lease)
                        if ctx.is_distributed:
                            dist.barrier()
                        return payload

                    if need_periodic:
                        _save_ckpt(latest_ckpt_path, epoch_num=epoch, step_in_ep=completed_step, eval_done=False)
                        _touch_lease()
                        last_ckpt_mono = now_mono

            is_final = epoch == run_spec.max_epochs
            if epoch in eval_epoch_set or is_final:
                mean_ep_loss = epoch_loss_sum / float(max(1, epoch_loss_steps))
                m_rec = _run_eval(
                    epoch_num=epoch,
                    is_final=is_final,
                    last_lr=last_lr,
                    mean_train_loss_nats=mean_ep_loss,
                    include_downstream=(is_final or run_spec.eval_downstream_at_milestones),
                    include_contacts=(is_final or run_spec.eval_contacts_at_milestones),
                )
                if ctx.rank == 0 and m_rec is not None:
                    milestones.append(m_rec)
                    payload = _build_payload(
                        status="complete" if is_final else "in_progress",
                        epoch_done=epoch,
                        step_in_epoch=steps_per_epoch,
                    )
                    atomic_write_json_with_sha256(run_json_path, payload)

                if is_final:
                    if run_spec.save_final_checkpoint:
                        _save_ckpt(final_ckpt_path, epoch_num=epoch, step_in_ep=steps_per_epoch, eval_done=True)
                    if ctx.rank == 0 and latest_ckpt_path.exists():
                        latest_ckpt_path.unlink(missing_ok=True)
                else:
                    _save_ckpt(latest_ckpt_path, epoch_num=epoch, step_in_ep=steps_per_epoch, eval_done=True)
                if ctx.is_distributed:
                    dist.barrier()
                last_ckpt_mono = time.monotonic()
                _touch_lease()

            resume_step_in_epoch = 0

        return _build_payload(status="complete", epoch_done=run_spec.max_epochs, step_in_epoch=steps_per_epoch)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_m
        torch.backends.cudnn.allow_tf32 = prev_tf32_c
        if install_signal_handlers:
            if prev_sigterm is not None:
                signal.signal(signal.SIGTERM, prev_sigterm)
            if prev_sigint is not None:
                signal.signal(signal.SIGINT, prev_sigint)
