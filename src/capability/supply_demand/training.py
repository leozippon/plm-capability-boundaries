"""Spot-safe WSD/cosine trainer, step-level checkpoint/resume, and shared-storage leases."""
from __future__ import annotations

import contextlib
import datetime
import json
import math
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..memory_capacity.optimizers import build_named_optimizer, set_optimizer_lr
from .corpus import (
    Stage2Bundle,
    atomic_write_json_with_sha256,
    sha256_file,
)
from .evaluation import (
    evaluate_megascale_ddg,
    evaluate_proteingym_substitutions,
    evaluate_three_tier_bits,
    evaluate_unseen_fold_contacts,
)
from .model import VariableLengthProteinTransformer, count_model_parameters
from .objectives import (
    PackedSequenceStore,
    compute_objective_loss,
    prepare_objective_batch,
)
from .spec import (
    LN2,
    SCALE_SPECS,
    Stage2RunSpec,
    assert_no_forbidden_infra_strings,
    objective_display_name,
    should_enable_gradient_checkpointing,
    vram_safe_micro_batch_cap,
)

RUN_RECEIPT_SCHEMA = "stage2_supply_demand_run"
CHECKPOINT_SCHEMA = "stage2_supply_demand_ckpt"
PREEMPT_EXIT_CODE = 143


@dataclass(frozen=True)
class DistributedContext:
    """Active single-GPU or torchrun DDP execution context for Stage-2 runs."""

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
            raise RuntimeError("Distributed Stage-2 training with WORLD_SIZE > 1 requires CUDA")
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=dev,
            timeout=datetime.timedelta(minutes=60),
        )
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


# --------------------------------------------------------------------------- #
# Learning Rate Schedules (WSD & Cosine) and Optimizer
# --------------------------------------------------------------------------- #


def compute_peak_lr(
    base_lr: float,
    effective_batch_size: int,
    *,
    override: float | None = None,
) -> float:
    if override is not None:
        return float(override)
    scale = min(2.0, max(0.5, math.sqrt(float(effective_batch_size) / 256.0)))
    return float(base_lr) * scale


def compute_warmup_steps(total_steps: int, *, override: int | None = None) -> int:
    if override is not None:
        return min(max(1, int(override)), max(1, int(total_steps)))
    raw = min(max(20, int(0.03 * float(total_steps))), 2000)
    return min(raw, max(1, int(total_steps)))


def compute_lr(
    step_1based: int,
    *,
    total_steps: int,
    peak_lr: float,
    warmup_steps: int,
    schedule: str = "wsd",
    cooldown_fraction: float = 0.10,
    min_lr_ratio: float = 0.05,
) -> float:
    """Compute step learning rate under WSD (Warmup-Stable-Decay) or Cosine schedule."""
    min_lr = float(min_lr_ratio) * float(peak_lr)
    step = max(1, int(step_1based))
    if step <= warmup_steps:
        return float(peak_lr) * (float(step) / float(max(1, warmup_steps)))
    if total_steps <= warmup_steps:
        return min_lr

    if schedule == "cosine":
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr + 0.5 * (float(peak_lr) - min_lr) * (1.0 + math.cos(math.pi * progress))

    if schedule == "wsd":
        cooldown_steps = int(round(float(cooldown_fraction) * float(total_steps)))
        cooldown_steps = min(cooldown_steps, max(0, total_steps - warmup_steps))
        cooldown_start = total_steps - cooldown_steps
        if cooldown_steps <= 0 or step <= cooldown_start:
            return float(peak_lr)
        progress = float(step - cooldown_start) / float(max(1, cooldown_steps))
        progress = min(max(progress, 0.0), 1.0)
        return min_lr + 0.5 * (float(peak_lr) - min_lr) * (1.0 + math.cos(math.pi * progress))

    raise ValueError(f"Unsupported lr_schedule {schedule!r}")


def build_optimizer(
    model: nn.Module,
    *,
    peak_lr: float,
    weight_decay: float,
    device: torch.device,
    optimizer_name: str = "adamw",
    muon_lr: float = 0.02,
    muon_lr_scale: float | None = None,
) -> torch.optim.Optimizer:
    """Create fused AdamW or hybrid Muon (Newton-Schulz 5) + AdamW optimizer."""
    return build_named_optimizer(
        model,
        optimizer_name=optimizer_name,
        peak_lr=peak_lr,
        weight_decay=weight_decay,
        device=device,
        muon_lr=muon_lr,
        muon_lr_scale=muon_lr_scale,
    )


# --------------------------------------------------------------------------- #
# Atomic Checkpointing, Receipt Verification, and Shared-Storage Lease Manager
# --------------------------------------------------------------------------- #


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist a PyTorch checkpoint via tmp file + fsync + POSIX os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}")
    try:
        with tmp.open("wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def is_run_completed(runs_dir: Path, run_id: str) -> bool:
    """Verify completed run receipt JSON and its .sha256 sidecar."""
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
        and doc.get("schema") == RUN_RECEIPT_SCHEMA
        and doc.get("status") == "complete"
    )


@dataclass
class ActiveLease:
    run_id: str
    path: Path
    owner_token: str
    claimed_epoch_sec: float = 0.0


class SharedLeaseManager:
    """Atomic filesystem lease manager for multi-worker Spot GPU concurrency on shared storage.

    Uses ``O_CREAT | O_EXCL`` for race-free claiming, heartbeat timestamps for liveness,
    owner-verified heartbeat/release so stolen leases are never clobbered, and stale-lease
    reclamation without persisting forbidden cluster/pod hostnames.
    """

    def __init__(
        self,
        lease_dir: Path | str,
        *,
        stale_after_sec: float = 900.0,
        worker_label: str = "spot_gpu",
    ) -> None:
        self.lease_dir = Path(lease_dir)
        self.lease_dir.mkdir(parents=True, exist_ok=True)
        self.stale_after_sec = float(stale_after_sec)
        assert_no_forbidden_infra_strings(worker_label, label="worker_label")
        self.worker_label = worker_label

    def lease_path(self, run_id: str) -> Path:
        return self.lease_dir / f"{run_id}.lease.json"

    def _is_stale(self, path: Path, now: float) -> bool:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            updated = float(doc.get("heartbeat_epoch_sec", 0.0))
            return (now - updated) > self.stale_after_sec
        except Exception:
            try:
                return (now - path.stat().st_mtime) > self.stale_after_sec
            except FileNotFoundError:
                return False

    def try_claim(self, run_id: str) -> ActiveLease | None:
        """Atomically claim `run_id`; returns ActiveLease if acquired, else None."""
        path = self.lease_path(run_id)
        now = time.time()
        owner_token = f"{self.worker_label}_{os.getpid()}_{time.monotonic_ns()}"
        payload = {
            "run_id": run_id,
            "owner_token": owner_token,
            "claimed_epoch_sec": now,
            "heartbeat_epoch_sec": now,
        }
        body = json.dumps(payload, sort_keys=True).encode("utf-8")

        for attempt in range(2):
            try:
                fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                try:
                    os.write(fd, body)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                return ActiveLease(
                    run_id=run_id,
                    path=path,
                    owner_token=owner_token,
                    claimed_epoch_sec=now,
                )
            except FileExistsError:
                if attempt == 0 and self._is_stale(path, now):
                    # Atomically evict stale lease via rename before retrying O_EXCL create
                    evicted = path.with_name(f".{path.name}.stale-{os.getpid()}-{time.monotonic_ns()}")
                    try:
                        os.replace(path, evicted)
                        evicted.unlink(missing_ok=True)
                    except FileNotFoundError:
                        pass
                    continue
                return None
        return None

    def heartbeat(self, lease: ActiveLease) -> bool:
        """Refresh lease heartbeat atomically ONLY if still owned by this worker.

        Returns True if refreshed, False if the lease was lost or stolen by another worker.
        """
        try:
            doc = json.loads(lease.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return False
        if doc.get("owner_token") != lease.owner_token:
            return False

        now = time.time()
        payload = {
            "run_id": lease.run_id,
            "owner_token": lease.owner_token,
            "claimed_epoch_sec": float(doc.get("claimed_epoch_sec", lease.claimed_epoch_sec)),
            "heartbeat_epoch_sec": now,
        }
        tmp = lease.path.with_name(f".{lease.path.name}.hb-{os.getpid()}-{time.monotonic_ns()}")
        try:
            with tmp.open("wb") as fh:
                fh.write(json.dumps(payload, sort_keys=True).encode("utf-8"))
                fh.flush()
                os.fsync(fh.fileno())
            # Re-verify ownership immediately prior to replace
            try:
                cur_doc = json.loads(lease.path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                return False
            if cur_doc.get("owner_token") != lease.owner_token:
                return False
            os.replace(tmp, lease.path)
            return True
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def release(self, lease: ActiveLease) -> bool:
        """Release lease ONLY if ownership is verified; never unlink another worker's lease."""
        try:
            doc = json.loads(lease.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return False
        if doc.get("owner_token") != lease.owner_token:
            return False
        try:
            lease.path.unlink()
            return True
        except FileNotFoundError:
            return False


# --------------------------------------------------------------------------- #
# Spot-Safe Training Engine
# --------------------------------------------------------------------------- #


def _build_epoch_batches(
    n_sequences: int,
    batch_size: int,
    *,
    seed: int,
    epoch: int,
) -> list[np.ndarray]:
    """Deterministic per-epoch mini-batch index partition."""
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(epoch))
    perm = rng.permutation(n_sequences)
    batches: list[np.ndarray] = []
    for start in range(0, n_sequences, batch_size):
        batches.append(perm[start : start + batch_size])
    return batches


def _validate_checkpoint_spec(ckpt: dict[str, Any], run_spec: Stage2RunSpec) -> None:
    """Fail fast if a resume checkpoint belongs to a different run configuration."""
    saved = ckpt.get("run_spec")
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint missing run_spec metadata")
    for key in ("run_id", "arm_id", "objective", "scale_id", "batch_size", "grad_accum_steps", "seed"):
        if saved.get(key) != getattr(run_spec, key):
            raise ValueError(
                f"Resume checkpoint mismatch on {key!r}: checkpoint has {saved.get(key)!r}, "
                f"requested {getattr(run_spec, key)!r}"
            )


def execute_stage2_run(
    run_spec: Stage2RunSpec,
    bundle: Stage2Bundle | Path | str,
    out_dir: Path | str,
    *,
    device: torch.device | str | None = None,
    use_compile: bool = False,
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
    """Execute one Stage-2 controlled sampling run with Spot-safe step-level checkpoint/resume and optional torchrun DDP."""
    ctx = setup_distributed(device=device)
    dev = ctx.device
    b_obj = bundle if isinstance(bundle, Stage2Bundle) else Stage2Bundle(bundle)
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

    if resume and is_run_completed(runs_dir, run_spec.run_id):
        if ctx.is_distributed:
            dist.barrier()
        return json.loads(run_json_path.read_text(encoding="utf-8"))

    splits = b_obj.load_arm_splits(run_spec.arm_id)
    train_store: PackedSequenceStore = splits["train"]
    train_sub_store: PackedSequenceStore = splits["eval_train_sub"]
    homolog_store: PackedSequenceStore = splits["eval_unseen_homolog"]
    unseen_fold_store: PackedSequenceStore = splits["eval_unseen_fold"]
    global_nat_store: PackedSequenceStore | None = splits.get("eval_global_natural")

    megascale_data = b_obj.load_megascale() if ctx.rank == 0 else None
    proteingym_data = b_obj.load_proteingym() if ctx.rank == 0 else None
    contact_domains = b_obj.load_contacts() if ctx.rank == 0 else []

    scale_spec = SCALE_SPECS[run_spec.scale_id]
    total_vram_bytes = (
        int(torch.cuda.get_device_properties(dev).total_memory)
        if dev.type == "cuda"
        else None
    )
    req_grad_ckpt = (
        gradient_checkpointing
        if gradient_checkpointing is not None
        else run_spec.gradient_checkpointing
    )
    use_grad_ckpt = should_enable_gradient_checkpointing(
        scale_spec.scale_id,
        total_vram_bytes,
        requested=req_grad_ckpt,
    )
    vram_cap = vram_safe_micro_batch_cap(
        scale_spec.scale_id,
        total_vram_bytes,
        default_cap=int(run_spec.batch_size),
    )
    # Preserve exact effective batch size while safely splitting micro-batches on smaller GPUs
    requested_per_rank_bs = int(run_spec.batch_size)
    micro_bs = min(requested_per_rank_bs, vram_cap)
    split_factor = max(1, math.ceil(float(requested_per_rank_bs) / float(micro_bs)))
    grad_accum = int(run_spec.grad_accum_steps) * split_factor
    eff_bs = micro_bs * grad_accum * int(ctx.world_size)
    n_micro_per_epoch_total = math.ceil(float(train_store.n_sequences) / float(micro_bs))
    # Under DDP, pad micro-batch count to a multiple of world_size so all ranks execute equal steps
    if ctx.is_distributed and (n_micro_per_epoch_total % ctx.world_size != 0):
        n_micro_per_epoch_total += ctx.world_size - (n_micro_per_epoch_total % ctx.world_size)
    n_micro_per_rank = n_micro_per_epoch_total // ctx.world_size
    steps_per_epoch = math.ceil(float(n_micro_per_rank) / float(grad_accum))
    total_steps = steps_per_epoch * int(run_spec.max_epochs)
    peak_lr = compute_peak_lr(scale_spec.base_lr, eff_bs, override=run_spec.peak_lr_override)
    warmup_steps = compute_warmup_steps(total_steps, override=run_spec.warmup_steps_override)
    eval_vram_cap = vram_safe_micro_batch_cap(
        scale_spec.scale_id,
        total_vram_bytes,
        default_cap=int(eval_batch_size) if eval_batch_size is not None else max(micro_bs, 128),
    )
    eff_eval_bs = min(
        int(eval_batch_size) if eval_batch_size is not None else max(micro_bs, 128),
        max(micro_bs, eval_vram_cap),
    )

    prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
    if run_spec.precision == "bf16":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_autocast = dev.type == "cuda"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        use_autocast = False

    autocast_ctx: Any = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_autocast
        else contextlib.nullcontext()
    )

    preempt_state = {"requested": False, "signal": None}
    prev_sigterm = None
    prev_sigint = None

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

        raw_model = VariableLengthProteinTransformer(
            scale_spec,
            run_spec.objective,
            max_seq_len=run_spec.max_seq_len,
            gradient_checkpointing=use_grad_ckpt,
        ).to(dev)
        param_counts = count_model_parameters(raw_model, scale_spec)
        optimizer = build_optimizer(
            raw_model,
            peak_lr=peak_lr,
            weight_decay=run_spec.weight_decay,
            device=dev,
            optimizer_name=run_spec.optimizer,
            muon_lr=run_spec.muon_lr,
            muon_lr_scale=run_spec.muon_lr_scale,
        )

        # Resume state defaults
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
            _validate_checkpoint_spec(ckpt, run_spec)
            raw_model.load_state_dict(ckpt["model_state_dict"])
            if ckpt.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            saved_epoch = int(ckpt["epoch"])
            saved_step_in_ep = int(ckpt["step_in_epoch"])
            global_step = int(ckpt["global_step"])
            tokens_seen = int(ckpt.get("tokens_seen", 0))
            prior_wall_sec = float(ckpt.get("elapsed_wall_sec", 0.0))
            init_eval = ckpt.get("init_eval")
            milestones = list(ckpt.get("milestones", []))
            saved_eval_done = bool(ckpt.get("eval_done", False))
            if saved_step_in_ep >= steps_per_epoch and saved_eval_done:
                cur_epoch = saved_epoch + 1
                resume_step_in_epoch = 0
                epoch_loss_sum = 0.0
                epoch_loss_steps = 0
            else:
                cur_epoch = saved_epoch
                resume_step_in_epoch = saved_step_in_ep
                epoch_loss_sum = float(ckpt.get("epoch_loss_sum", 0.0))
                epoch_loss_steps = int(ckpt.get("epoch_loss_steps", 0))

        compiled_model: nn.Module = torch.compile(raw_model) if use_compile else raw_model
        if ctx.is_distributed:
            train_model: nn.Module = DDP(
                compiled_model,
                device_ids=[ctx.local_rank],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
            eval_model: nn.Module = compiled_model
        else:
            train_model = compiled_model
            eval_model = compiled_model

        t_run_start = time.perf_counter()
        last_ckpt_mono = time.monotonic()

        def _build_payload(*, status: str, epoch_done: int, step_in_epoch: int) -> dict[str, Any]:
            wall = prior_wall_sec + (time.perf_counter() - t_run_start)
            return {
                "schema": RUN_RECEIPT_SCHEMA,
                "status": status,
                "run_id": run_spec.run_id,
                "arm_id": run_spec.arm_id,
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
                    "lr_schedule": run_spec.lr_schedule,
                    "optimizer": run_spec.optimizer,
                    "muon_lr": float(run_spec.muon_lr) if run_spec.optimizer == "muon" else None,
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

        def _save_checkpoint(
            path: Path,
            *,
            epoch_num: int,
            step_in_ep: int,
            eval_done: bool,
        ) -> None:
            if ctx.rank == 0:
                wall = prior_wall_sec + (time.perf_counter() - t_run_start)
                ckpt_payload = {
                    "schema": CHECKPOINT_SCHEMA,
                    "run_spec": run_spec.to_dict(),
                    "epoch": int(epoch_num),
                    "step_in_epoch": int(step_in_ep),
                    "eval_done": bool(eval_done),
                    "global_step": int(global_step),
                    "tokens_seen": int(tokens_seen),
                    "elapsed_wall_sec": float(wall),
                    "epoch_loss_sum": float(epoch_loss_sum),
                    "epoch_loss_steps": int(epoch_loss_steps),
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "init_eval": init_eval,
                    "milestones": milestones,
                }
                atomic_torch_save(path, ckpt_payload)
            if ctx.is_distributed:
                dist.barrier()

        def _touch_lease() -> None:
            if ctx.rank == 0 and lease_manager is not None and active_lease is not None:
                lease_manager.heartbeat(active_lease)

        def _run_milestone_eval(
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
                three_tier = evaluate_three_tier_bits(
                    eval_model,
                    train_sub_store=train_sub_store,
                    unseen_homolog_store=homolog_store,
                    unseen_fold_store=unseen_fold_store,
                    global_natural_store=global_nat_store,
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
                    "three_tier_bits": three_tier,
                }

                if include_downstream and megascale_data is not None:
                    _touch_lease()
                    rec["megascale"] = evaluate_megascale_ddg(
                        eval_model,
                        megascale_data,
                        objective=run_spec.objective,
                        batch_size=eff_eval_bs,
                        precision=run_spec.precision,
                    )
                if include_downstream and proteingym_data is not None:
                    _touch_lease()
                    rec["proteingym_milestone"] = evaluate_proteingym_substitutions(
                        eval_model,
                        proteingym_data,
                        objective=run_spec.objective,
                        milestone_only=True,
                        batch_size=eff_eval_bs,
                        precision=run_spec.precision,
                    )
                    if is_final and run_spec.eval_full_proteingym_at_final:
                        _touch_lease()
                        rec["proteingym_full"] = evaluate_proteingym_substitutions(
                            eval_model,
                            proteingym_data,
                            objective=run_spec.objective,
                            milestone_only=False,
                            batch_size=eff_eval_bs,
                            precision=run_spec.precision,
                        )
                if include_contacts and contact_domains:
                    _touch_lease()
                    rec["contacts"] = evaluate_unseen_fold_contacts(
                        eval_model,
                        contact_domains,
                        objective=run_spec.objective,
                        batch_size=eff_eval_bs,
                        precision=run_spec.precision,
                    )
                _touch_lease()
            if ctx.is_distributed:
                dist.barrier()
            return rec

        # Initial random-baseline evaluation (epoch 0)
        if run_init_eval and init_eval is None and cur_epoch == 1 and resume_step_in_epoch == 0:
            init_eval = _run_milestone_eval(
                epoch_num=0,
                is_final=False,
                last_lr=0.0,
                mean_train_loss_nats=None,
                include_downstream=False,
                include_contacts=False,
            )

        eval_epoch_set = set(run_spec.eval_epochs)
        last_lr = (
            compute_lr(
                global_step,
                total_steps=total_steps,
                peak_lr=peak_lr,
                warmup_steps=warmup_steps,
                schedule=run_spec.lr_schedule,
                cooldown_fraction=run_spec.cooldown_fraction,
                min_lr_ratio=run_spec.min_lr_ratio,
            )
            if global_step > 0
            else peak_lr
        )

        for epoch in range(cur_epoch, run_spec.max_epochs + 1):
            train_model.train()
            all_micro_batches = _build_epoch_batches(
                train_store.n_sequences,
                micro_bs,
                seed=run_spec.seed,
                epoch=epoch,
            )
            if ctx.is_distributed:
                if len(all_micro_batches) < n_micro_per_epoch_total:
                    pad_cnt = n_micro_per_epoch_total - len(all_micro_batches)
                    all_micro_batches = all_micro_batches + [
                        all_micro_batches[i % len(all_micro_batches)] for i in range(pad_cnt)
                    ]
                micro_batches = all_micro_batches[ctx.rank :: ctx.world_size]
            else:
                micro_batches = all_micro_batches

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

                for mb_offset, mb_i in enumerate(range(mb_lo, mb_hi)):
                    idx_arr = micro_batches[mb_i]
                    tokens, lengths = train_store.collate_indices(
                        idx_arr, device=dev, pad_to_multiple=8
                    )
                    mask_gen = torch.Generator(device=dev)
                    mask_gen.manual_seed(
                        int(run_spec.seed) * 10_000_019
                        + int(epoch) * 100_003
                        + int(step_idx) * 97
                        + int(mb_offset) * 17
                        + int(ctx.rank)
                    )
                    prepared = prepare_objective_batch(
                        run_spec.objective,
                        tokens,
                        lengths,
                        mask_fraction=run_spec.effective_mask_rate,
                        generator=mask_gen,
                    )
                    sync_ctx: Any = (
                        train_model.no_sync()
                        if (ctx.is_distributed and mb_offset < actual_accum - 1)
                        else contextlib.nullcontext()
                    )
                    with sync_ctx:
                        with autocast_ctx:
                            logits = train_model(
                                prepared.input_ids,
                                prepared.attention_mask,
                                right_padded=True,
                            )
                        loss = compute_objective_loss(logits, prepared)
                        scaled_loss = loss / float(actual_accum)
                        scaled_loss.backward()

                    step_loss_val += float(loss.detach().item()) / float(actual_accum)
                    step_tokens += int(lengths.sum().item())

                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
                optimizer.step()

                if ctx.is_distributed:
                    stats_t = torch.tensor(
                        [step_loss_val, float(step_tokens)], dtype=torch.float64, device=dev
                    )
                    dist.all_reduce(stats_t, op=dist.ReduceOp.SUM)
                    step_loss_val = float(stats_t[0].item()) / float(ctx.world_size)
                    step_tokens = int(round(float(stats_t[1].item())))

                tokens_seen += step_tokens
                epoch_loss_sum += step_loss_val
                epoch_loss_steps += 1
                completed_step_in_epoch = step_idx + 1

                # Synchronized Spot preemption / periodic checkpoint check
                should_check_sync = (not ctx.is_distributed) or (
                    completed_step_in_epoch == steps_per_epoch or completed_step_in_epoch % 8 == 0
                )
                if should_check_sync:
                    need_preempt = bool(preempt_state["requested"])
                    now_mono = time.monotonic()
                    need_periodic = (now_mono - last_ckpt_mono) >= run_spec.checkpoint_interval_sec
                    if ctx.is_distributed:
                        flags = torch.tensor(
                            [1 if need_preempt else 0, 1 if (ctx.rank == 0 and need_periodic) else 0],
                            dtype=torch.int64,
                            device=dev,
                        )
                        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
                        need_preempt = bool(flags[0].item() > 0)
                        need_periodic = bool(flags[1].item() > 0)

                    if need_preempt:
                        _save_checkpoint(
                            latest_ckpt_path,
                            epoch_num=epoch,
                            step_in_ep=completed_step_in_epoch,
                            eval_done=False,
                        )
                        payload = _build_payload(
                            status="preempted",
                            epoch_done=epoch - 1 if completed_step_in_epoch < steps_per_epoch else epoch,
                            step_in_epoch=completed_step_in_epoch,
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
                        _save_checkpoint(
                            latest_ckpt_path,
                            epoch_num=epoch,
                            step_in_ep=completed_step_in_epoch,
                            eval_done=False,
                        )
                        _touch_lease()
                        last_ckpt_mono = now_mono

            # End of epoch: run milestone evaluation if scheduled
            is_final = epoch == run_spec.max_epochs
            if epoch in eval_epoch_set or is_final:
                mean_ep_loss = epoch_loss_sum / float(max(1, epoch_loss_steps))
                m_rec = _run_milestone_eval(
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
                        _save_checkpoint(
                            final_ckpt_path,
                            epoch_num=epoch,
                            step_in_ep=steps_per_epoch,
                            eval_done=True,
                        )
                    if ctx.rank == 0 and latest_ckpt_path.exists():
                        latest_ckpt_path.unlink(missing_ok=True)
                else:
                    _save_checkpoint(
                        latest_ckpt_path,
                        epoch_num=epoch,
                        step_in_ep=steps_per_epoch,
                        eval_done=True,
                    )
                if ctx.is_distributed:
                    dist.barrier()
                last_ckpt_mono = time.monotonic()
                _touch_lease()

            # Reset resume offset after finishing resumed epoch
            resume_step_in_epoch = 0

        return _build_payload(
            status="complete",
            epoch_done=run_spec.max_epochs,
            step_in_epoch=steps_per_epoch,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul
        torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn
        if install_signal_handlers:
            if prev_sigterm is not None:
                signal.signal(signal.SIGTERM, prev_sigterm)
            if prev_sigint is not None:
                signal.signal(signal.SIGINT, prev_sigint)
