"""GPU-resident AR, MLM-15%, and OA-Uniform training engine with WSD, adaptive stop, and Spot resume."""
from __future__ import annotations

import contextlib
import math
import os
import random
import re
import signal
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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
    DEFAULT_MLM_FIXED_NUM_DRAWS,
    evaluate_mlm_fixed_rate_bits,
    evaluate_mlm_joint_bits_16pt,
    evaluate_mlm_loo_diagnostic,
    evaluate_run_checkpoint,
)
from .model import UniformVocabTransformer, count_model_parameters
from .spec import (
    C_GRID_PRIOR_BITS_PER_PARAM,
    LN2,
    MLM_15_MASK_RATE,
    SCALE_SPECS,
    SEQ_LEN,
    SPECIAL_TOKEN_ID,
    VOCAB_AA,
    RunSpec,
    assert_no_forbidden_infra_strings,
    canonicalize_paradigm,
    choose_global_batch_size,
    dataset_entropy_bits,
    gibbs_memorizable_bits,
    mlm_fixed_mask_count,
    null_ordered_entropy_bits,
    slugify_token,
    vram_safe_micro_batch_cap,
)

RUN_RECEIPT_SCHEMA = "plm_parametric_capacity_run_v1"
CHECKPOINT_SCHEMA = "plm_parametric_capacity_ckpt_v1"
PREEMPTION_EXIT_CODE: int = 143
DEFAULT_CHECKPOINT_INTERVAL_SEC: float = 300.0


@dataclass(frozen=True)
class DistributedContext:
    """Active single-process or torchrun DDP execution context."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_distributed: bool


@dataclass(frozen=True)
class AdaptiveEarlyStopConfig:
    """Two-phase adaptive early-stopping configuration under WSD."""

    enabled: bool = False
    takeoff_delta_bpt: float = 0.05
    takeoff_c_eff_bpp: float = 0.02
    takeoff_consecutive_evals: int = 2
    data_ceiling_frac_gibbs: float = 0.995
    ar_xent_ceiling_bpt: float = 0.025
    mlm_xent_ceiling_bpt: float = 0.04
    plateau_abs_gain_per_doubling: float = 0.02
    plateau_rel_gain_per_doubling: float = 0.015
    plateau_consecutive_intervals: int = 2
    plateau_min_exposure_span_ratio: float = 1.5
    plateau_min_takeoff_multiplier: float = 2.0
    cooldown_fraction: float = 0.1
    cooldown_min_epochs: int = 20
    cooldown_max_epochs: int = 200

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdaptiveEarlyStopConfig:
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


def compute_cooldown_epochs(
    trigger_epoch: int,
    *,
    fraction: float = 0.1,
    min_epochs: int = 20,
    max_epochs: int = 200,
) -> int:
    """Compute WSD cosine cooldown duration: clamp(round(fraction * trigger_epoch), min, max)."""
    raw = int(round(float(fraction) * float(max(1, int(trigger_epoch)))))
    return max(int(min_epochs), min(int(max_epochs), raw))


@dataclass
class AdaptiveEarlyStopTracker:
    """Stateful two-phase adaptive stopping + WSD cooldown controller."""

    config: AdaptiveEarlyStopConfig = field(default_factory=AdaptiveEarlyStopConfig)
    takeoff_streak: int = 0
    takeoff_detected: bool = False
    e_takeoff: int | None = None
    plateau_intervals: list[list[int]] = field(default_factory=list)
    cooldown_triggered: bool = False
    cooldown_trigger_epoch: int | None = None
    cooldown_trigger_reason: str | None = None
    cooldown_start_step: int | None = None
    cooldown_end_step: int | None = None
    stop_epoch: int | None = None
    prev_epoch: int | None = None
    prev_c_eff: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "takeoff_streak": int(self.takeoff_streak),
            "takeoff_detected": bool(self.takeoff_detected),
            "e_takeoff": int(self.e_takeoff) if self.e_takeoff is not None else None,
            "plateau_intervals": [[int(a), int(b)] for a, b in self.plateau_intervals],
            "cooldown_triggered": bool(self.cooldown_triggered),
            "cooldown_trigger_epoch": (
                int(self.cooldown_trigger_epoch)
                if self.cooldown_trigger_epoch is not None
                else None
            ),
            "cooldown_trigger_reason": self.cooldown_trigger_reason,
            "cooldown_start_step": (
                int(self.cooldown_start_step) if self.cooldown_start_step is not None else None
            ),
            "cooldown_end_step": (
                int(self.cooldown_end_step) if self.cooldown_end_step is not None else None
            ),
            "stop_epoch": int(self.stop_epoch) if self.stop_epoch is not None else None,
            "prev_epoch": int(self.prev_epoch) if self.prev_epoch is not None else None,
            "prev_c_eff": float(self.prev_c_eff) if self.prev_c_eff is not None else None,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        fallback_config: AdaptiveEarlyStopConfig | None = None,
    ) -> AdaptiveEarlyStopTracker:
        cfg_data = data.get("config")
        if isinstance(cfg_data, dict):
            cfg = AdaptiveEarlyStopConfig.from_dict(cfg_data)
        elif fallback_config is not None:
            cfg = fallback_config
        else:
            cfg = AdaptiveEarlyStopConfig()
        return cls(
            config=cfg,
            takeoff_streak=int(data.get("takeoff_streak", 0)),
            takeoff_detected=bool(data.get("takeoff_detected", False)),
            e_takeoff=int(data["e_takeoff"]) if data.get("e_takeoff") is not None else None,
            plateau_intervals=[
                [int(pair[0]), int(pair[1])] for pair in data.get("plateau_intervals", [])
            ],
            cooldown_triggered=bool(data.get("cooldown_triggered", False)),
            cooldown_trigger_epoch=(
                int(data["cooldown_trigger_epoch"])
                if data.get("cooldown_trigger_epoch") is not None
                else None
            ),
            cooldown_trigger_reason=data.get("cooldown_trigger_reason"),
            cooldown_start_step=(
                int(data["cooldown_start_step"])
                if data.get("cooldown_start_step") is not None
                else None
            ),
            cooldown_end_step=(
                int(data["cooldown_end_step"])
                if data.get("cooldown_end_step") is not None
                else None
            ),
            stop_epoch=int(data["stop_epoch"]) if data.get("stop_epoch") is not None else None,
            prev_epoch=int(data["prev_epoch"]) if data.get("prev_epoch") is not None else None,
            prev_c_eff=float(data["prev_c_eff"]) if data.get("prev_c_eff") is not None else None,
        )

    def activate_cooldown(
        self,
        *,
        trigger_epoch: int,
        reason: str,
        global_step: int,
        steps_per_epoch: int,
        max_epochs: int,
        cooldown_epochs_override: int | None = None,
    ) -> int:
        """Enter WSD cosine cooldown phase and return the target terminal epoch."""
        if cooldown_epochs_override is not None:
            cd_epochs = max(0, int(cooldown_epochs_override))
        else:
            cd_epochs = compute_cooldown_epochs(
                trigger_epoch,
                fraction=self.config.cooldown_fraction,
                min_epochs=self.config.cooldown_min_epochs,
                max_epochs=self.config.cooldown_max_epochs,
            )
        target_stop = min(int(max_epochs), int(trigger_epoch) + cd_epochs)
        eff_cd_epochs = max(0, target_stop - int(trigger_epoch))
        self.cooldown_triggered = True
        self.cooldown_trigger_epoch = int(trigger_epoch)
        self.cooldown_trigger_reason = str(reason)
        self.cooldown_start_step = int(global_step)
        self.cooldown_end_step = int(global_step) + eff_cd_epochs * int(steps_per_epoch)
        self.stop_epoch = int(target_stop)
        return target_stop

    def update(
        self,
        *,
        epoch: int,
        eval_rec: dict[str, Any],
        paradigm: str,
        global_step: int,
        steps_per_epoch: int,
        max_epochs: int,
        vocab_size: int = VOCAB_AA,
    ) -> str | None:
        """Process one evaluation snapshot; returns trigger reason if cooldown just triggered."""
        canon_paradigm = canonicalize_paradigm(paradigm)
        tok_xent = float(eval_rec.get("token_xent_bits", eval_rec["mean_tok_xent_bits"]))
        c_eff = float(eval_rec["bits_per_non_emb_param"])
        frac_gibbs = float(eval_rec["fraction_of_H_gibbs"])
        if not (math.isfinite(tok_xent) and math.isfinite(c_eff) and math.isfinite(frac_gibbs)):
            raise FloatingPointError(
                f"Non-finite eval metrics at epoch {epoch}: "
                f"tok_xent={tok_xent}, c_eff={c_eff}, frac_gibbs={frac_gibbs}"
            )

        delta_bpt = math.log2(float(vocab_size)) - tok_xent

        # Phase A: Track takeoff even when cooldown already inactive/disabled for provenance.
        # Gate on either delta_bpt >= takeoff_delta_bpt OR capacity-units c_eff >= takeoff_c_eff_bpp
        # (r-invariant so high-r runs such as r=8, 16 accurately record e_takeoff).
        above_takeoff = (
            delta_bpt >= self.config.takeoff_delta_bpt
            or c_eff >= self.config.takeoff_c_eff_bpp
        )
        if above_takeoff:
            self.takeoff_streak += 1
            if (
                not self.takeoff_detected
                and self.takeoff_streak >= self.config.takeoff_consecutive_evals
            ):
                self.takeoff_detected = True
                self.e_takeoff = int(epoch)
        else:
            self.takeoff_streak = 0

        if not self.config.enabled or self.cooldown_triggered:
            self.prev_epoch = int(epoch)
            self.prev_c_eff = c_eff
            return None

        # Phase A invariant: NEVER plateau-stop while pre-takeoff or below takeoff thresholds
        if not self.takeoff_detected or not above_takeoff:
            self.plateau_intervals.clear()
            self.prev_epoch = int(epoch)
            self.prev_c_eff = c_eff
            return None

        # Phase B (i): Data ceiling reached
        # For mlm_15, conditional pseudo-bits omit the log2(N) sequence-ID cost so
        # pseudo_bits / H_gibbs can approach/exceed 1.0 before h_15 reaches ceiling;
        # gate mlm_15 strictly on masked-token xent <= mlm_xent_ceiling_bpt (0.04 bpt).
        xent_ceiling = (
            self.config.ar_xent_ceiling_bpt
            if canon_paradigm == "ar"
            else self.config.mlm_xent_ceiling_bpt
        )
        hit_data_ceiling = (
            (tok_xent <= xent_ceiling)
            if canon_paradigm == "mlm_15"
            else (frac_gibbs >= self.config.data_ceiling_frac_gibbs or tok_xent <= xent_ceiling)
        )
        if hit_data_ceiling:
            self.prev_epoch = int(epoch)
            self.prev_c_eff = c_eff
            self.activate_cooldown(
                trigger_epoch=epoch,
                reason="data_ceiling",
                global_step=global_step,
                steps_per_epoch=steps_per_epoch,
                max_epochs=max_epochs,
            )
            return "data_ceiling"

        # Phase B (ii): Capacity plateau over consecutive eval intervals
        if self.prev_epoch is not None and int(epoch) > self.prev_epoch and self.prev_c_eff is not None:
            doublings = math.log2(float(epoch) / float(self.prev_epoch))
            if doublings > 0.0:
                gain_per_doubling = (c_eff - self.prev_c_eff) / doublings
                rel_gain = gain_per_doubling / max(c_eff, 0.1)
                if (
                    gain_per_doubling < self.config.plateau_abs_gain_per_doubling
                    and rel_gain < self.config.plateau_rel_gain_per_doubling
                ):
                    self.plateau_intervals.append([int(self.prev_epoch), int(epoch)])
                else:
                    self.plateau_intervals.clear()

        self.prev_epoch = int(epoch)
        self.prev_c_eff = c_eff

        if (
            self.e_takeoff is not None
            and len(self.plateau_intervals) >= self.config.plateau_consecutive_intervals
            and float(epoch) >= self.config.plateau_min_takeoff_multiplier * float(self.e_takeoff)
        ):
            span_start = self.plateau_intervals[0][0]
            span_end = self.plateau_intervals[-1][1]
            if span_start > 0 and (float(span_end) / float(span_start)) >= self.config.plateau_min_exposure_span_ratio:
                self.activate_cooldown(
                    trigger_epoch=epoch,
                    reason="capacity_plateau",
                    global_step=global_step,
                    steps_per_epoch=steps_per_epoch,
                    max_epochs=max_epochs,
                )
                return "capacity_plateau"

        return None


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
    min_lr_ratio: float = 0.05,
) -> float:
    """Linear warmup to peak_lr followed by cosine decay to min_lr = min_lr_ratio * peak_lr."""
    min_lr = float(min_lr_ratio) * float(peak_lr)
    if step_1based <= warmup_steps:
        return float(peak_lr) * (float(step_1based) / float(max(1, warmup_steps)))
    if total_steps <= warmup_steps:
        return min_lr
    progress = float(step_1based - warmup_steps) / float(total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (float(peak_lr) - min_lr) * (1.0 + math.cos(math.pi * progress))


def compute_wsd_lr(
    step_1based: int,
    *,
    peak_lr: float,
    warmup_steps: int,
    min_lr_ratio: float = 0.05,
    cooldown_start_step: int | None = None,
    cooldown_end_step: int | None = None,
) -> float:
    """Warmup-Stable-Decay LR: linear warmup -> constant peak_lr -> cosine cooldown."""
    min_lr = float(min_lr_ratio) * float(peak_lr)
    if cooldown_start_step is not None and step_1based > cooldown_start_step:
        if cooldown_end_step is None or cooldown_end_step <= cooldown_start_step:
            return min_lr
        if cooldown_start_step < warmup_steps:
            start_lr = max(
                min_lr,
                float(peak_lr) * (float(cooldown_start_step) / float(max(1, warmup_steps))),
            )
        else:
            start_lr = float(peak_lr)
        progress = float(step_1based - cooldown_start_step) / float(
            cooldown_end_step - cooldown_start_step
        )
        progress = min(max(progress, 0.0), 1.0)
        return min_lr + 0.5 * (start_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    if step_1based <= warmup_steps:
        return float(peak_lr) * (float(step_1based) / float(max(1, warmup_steps)))
    return float(peak_lr)


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


def sample_mlm_fixed_rate_batch(
    batch_seqs: torch.Tensor,
    *,
    mask_count: int,
    generator: torch.Generator,
    special_token_id: int = SPECIAL_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamic fixed-rate MLM corruption: fresh independent uniform mask of exact size k per row."""
    bsz, seq_len = batch_seqs.shape
    if not (1 <= int(mask_count) <= int(seq_len)):
        raise ValueError(f"mask_count must be in [1, {seq_len}], got {mask_count}")
    device = batch_seqs.device
    targets = batch_seqs.to(dtype=torch.long)

    noise = torch.rand((bsz, seq_len), device=device, generator=generator, dtype=torch.float32)
    positions = noise.topk(int(mask_count), dim=-1, largest=False).indices
    mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=device)
    mask.scatter_(1, positions, True)
    n_mask = torch.full((bsz,), int(mask_count), dtype=torch.long, device=device)

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
    """Exact unbiased OA-ARDM / fixed-rate MLM masked cross-entropy (in nats per token)."""
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


def verify_checkpoint_compatibility(
    ckpt: dict[str, Any],
    run_spec: RunSpec,
    *,
    steps_per_epoch: int | None = None,
) -> None:
    """Verify that a loaded checkpoint is compatible with ``run_spec`` before resuming."""
    ckpt_hash = ckpt.get("run_spec_hash")
    expected_hash = run_spec.spec_hash()
    if ckpt_hash is not None:
        if str(ckpt_hash) != expected_hash:
            ckpt_spec = ckpt.get("run_spec", {})
            raise ValueError(
                f"Checkpoint run_spec_hash mismatch for {run_spec.run_id}: "
                f"ckpt={ckpt_hash[:12]} vs spec={expected_hash[:12]} "
                f"(ckpt_spec={ckpt_spec!r}, target_spec={run_spec.to_dict()!r})"
            )
    elif isinstance(ckpt.get("run_spec"), dict):
        # Legacy checkpoint without run_spec_hash: verify core fields via RunSpec.from_dict
        legacy_spec = RunSpec.from_dict(ckpt["run_spec"])
        for field_name in ("paradigm", "scale_id", "n_sequences", "weight_decay", "precision", "seed", "data_seed"):
            if getattr(legacy_spec, field_name) != getattr(run_spec, field_name):
                raise ValueError(
                    f"Legacy checkpoint incompatibility on {field_name}: "
                    f"ckpt={getattr(legacy_spec, field_name)!r} vs spec={getattr(run_spec, field_name)!r}"
                )

    step_in_epoch = int(ckpt.get("step_in_epoch", 0))
    ckpt_spe = ckpt.get("steps_per_epoch")
    if step_in_epoch > 0 and steps_per_epoch is not None and ckpt_spe is not None:
        if int(ckpt_spe) != int(steps_per_epoch):
            raise ValueError(
                f"Cannot resume mid-epoch (step_in_epoch={step_in_epoch}) when "
                f"steps_per_epoch changed ({ckpt_spe} -> {steps_per_epoch})"
            )


def _capture_rng_states(
    mask_gen: torch.Generator | None = None,
    *,
    mask_gen_states_by_rank: list[torch.Tensor | None] | None = None,
) -> dict[str, Any]:
    local_mask_state = (
        mask_gen.get_state().detach().to(device="cpu", dtype=torch.uint8)
        if mask_gen is not None
        else None
    )
    states: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "mask_gen_state": local_mask_state,
        "mask_gen_states_by_rank": mask_gen_states_by_rank,
    }
    return states


def _restore_rng_states(rng_states: dict[str, Any] | None) -> None:
    if not isinstance(rng_states, dict):
        return
    if rng_states.get("python_random") is not None:
        random.setstate(rng_states["python_random"])
    if rng_states.get("numpy_random") is not None:
        np.random.set_state(rng_states["numpy_random"])
    if rng_states.get("torch_cpu") is not None:
        cpu_state = rng_states["torch_cpu"]
        if isinstance(cpu_state, torch.Tensor):
            torch.set_rng_state(cpu_state.to(device="cpu", dtype=torch.uint8))
    if torch.cuda.is_available() and rng_states.get("torch_cuda") is not None:
        cuda_states = rng_states["torch_cuda"]
        if isinstance(cuda_states, list) and len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(
                [s.to(device="cpu", dtype=torch.uint8) for s in cuda_states]
            )


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with tmp.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _verified_existing_completed_run(
    run_json_path: Path,
    run_spec: RunSpec,
    ckpt_root: Path,
) -> dict[str, Any] | None:
    sidecar = run_json_path.with_name(run_json_path.name + ".sha256")
    if not (run_json_path.is_file() and sidecar.is_file()):
        return None
    expected_line = sidecar.read_text(encoding="utf-8").strip().split()
    if not expected_line or expected_line[0] != sha256_file(run_json_path):
        return None
    payload = load_json_file(run_json_path)
    if not (
        isinstance(payload, dict)
        and payload.get("schema") == RUN_RECEIPT_SCHEMA
        and payload.get("status") == "complete"
    ):
        return None

    epoch_done = int(payload.get("epoch_done", -1))
    stopped_early = bool(payload.get("stopped_early", False))
    if not (epoch_done == run_spec.max_epochs or (stopped_early and 1 <= epoch_done <= run_spec.max_epochs)):
        return None

    if run_spec.save_checkpoint:
        final_ckpt = ckpt_root / f"{run_spec.run_id}_e{epoch_done}.pt"
        if not final_ckpt.is_file():
            return None
    return payload


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
    mlm_fixed_15pct_diag: dict[str, Any] | None,
    mlm_oa_16pt_diag: dict[str, Any] | None,
    early_stop: AdaptiveEarlyStopTracker,
    is_complete: bool,
) -> dict[str, Any]:
    stopped_early = bool(
        is_complete
        and early_stop.cooldown_triggered
        and early_stop.cooldown_trigger_reason not in (None, "max_epochs_horizon")
        and int(epoch_done) < int(run_spec.max_epochs)
    )
    return {
        "schema": RUN_RECEIPT_SCHEMA,
        "status": "complete" if is_complete else "in_progress",
        "run_id": run_spec.run_id,
        "run_spec_hash": run_spec.spec_hash(),
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
        "lr_schedule": run_spec.lr_schedule,
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
        "stopped_early": stopped_early,
        "stop_reason": early_stop.cooldown_trigger_reason if is_complete else None,
        "e_takeoff": early_stop.e_takeoff,
        "cooldown_trigger_epoch": early_stop.cooldown_trigger_epoch,
        "wsd_early_stop_state": early_stop.to_dict(),
        "wall_time_sec": float(elapsed_wall),
        "init_eval": init_eval,
        "exposures": exposures,
        "evaluations": exposures,
        "final_eval": exposures[-1] if exposures else None,
        "mlm_loo_diagnostic": mlm_loo_diag,
        "mlm_fixed_15pct_diagnostic": mlm_fixed_15pct_diag,
        "mlm_oa_16pt_diagnostic": mlm_oa_16pt_diag,
    }


def _compute_horizon_cooldown_epochs(
    max_epochs: int,
    cfg: AdaptiveEarlyStopConfig,
) -> int:
    if max_epochs <= 1:
        return 0
    if max_epochs <= cfg.cooldown_min_epochs:
        raw = int(round(float(cfg.cooldown_fraction) * float(max_epochs)))
        return max(1, min(max_epochs - 1, raw))
    cd = compute_cooldown_epochs(
        max_epochs,
        fraction=cfg.cooldown_fraction,
        min_epochs=cfg.cooldown_min_epochs,
        max_epochs=cfg.cooldown_max_epochs,
    )
    return min(cd, max_epochs - 1)


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
    mlm_fixed_num_draws: int = DEFAULT_MLM_FIXED_NUM_DRAWS,
    checkpoint_interval_sec: float = DEFAULT_CHECKPOINT_INTERVAL_SEC,
    adaptive_stop_config: AdaptiveEarlyStopConfig | None = None,
    cross_eval_epochs: Sequence[int] | None = None,
    delete_transient_ckpt_on_complete: bool | None = None,
    handle_signals: bool = True,
) -> dict[str, Any]:
    """Execute one AR, MLM-15%, or OA-Uniform parametric memory-capacity run end-to-end."""
    ctx = setup_distributed(device=device)
    out_root = Path(out_dir)
    runs_dir = out_root / "runs"
    ckpt_root = Path(checkpoint_dir) if checkpoint_dir is not None else (out_root / "checkpoints")
    run_json_path = runs_dir / f"{run_spec.run_id}.json"
    latest_ckpt_path = ckpt_root / f"{run_spec.run_id}_latest.pt"

    eff_skip_completed = bool(resume) if skip_completed is None else bool(skip_completed)
    if eff_skip_completed:
        existing = _verified_existing_completed_run(run_json_path, run_spec, ckpt_root)
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
            min_steps_per_epoch=run_spec.min_steps_per_epoch,
        )

    micro_batch_size = global_batch_size // ctx.world_size
    total_vram_bytes = (
        int(torch.cuda.get_device_properties(ctx.device).total_memory)
        if ctx.device.type == "cuda"
        else None
    )
    micro_step_cap = vram_safe_micro_batch_cap(
        scale_spec.scale_id,
        total_vram_bytes,
        default_cap=micro_batch_size,
    )
    eval_cap = vram_safe_micro_batch_cap(
        scale_spec.scale_id,
        total_vram_bytes,
        default_cap=min(scale_spec.max_batch_size, 512),
    )
    micro_step_batch_size = min(micro_batch_size, micro_step_cap)
    grad_accum_steps = (micro_batch_size + micro_step_cap - 1) // micro_step_cap
    steps_per_epoch = math.ceil(float(run_spec.n_sequences) / float(global_batch_size))
    total_steps = steps_per_epoch * run_spec.max_epochs
    peak_lr = compute_peak_lr(scale_spec.base_lr, global_batch_size)
    min_lr = float(run_spec.min_lr_ratio) * peak_lr
    warmup_steps = compute_warmup_steps(total_steps)
    eval_batch_size = min(max(micro_batch_size, min(scale_spec.max_batch_size, 512)), eval_cap)
    fixed_mask_k = mlm_fixed_mask_count(seq_len=seq_len, mask_rate=run_spec.mask_rate)

    es_cfg = (
        adaptive_stop_config
        if adaptive_stop_config is not None
        else AdaptiveEarlyStopConfig(enabled=bool(run_spec.adaptive_early_stop))
    )
    early_stop = AdaptiveEarlyStopTracker(config=es_cfg)

    eff_delete_transient = (
        bool(delete_transient_ckpt_on_complete)
        if delete_transient_ckpt_on_complete is not None
        else bool(run_spec.delete_transient_ckpt_on_complete or (not run_spec.save_checkpoint))
    )

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

    preempt_signal: int = 0
    prev_sigterm: Any = None
    prev_sigint: Any = None
    signals_installed = False

    if handle_signals:
        def _on_preempt_signal(signum: int, _frame: Any) -> None:
            nonlocal preempt_signal
            preempt_signal = int(signum)

        try:
            prev_sigterm = signal.getsignal(signal.SIGTERM)
            prev_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGTERM, _on_preempt_signal)
            signal.signal(signal.SIGINT, _on_preempt_signal)
            signals_installed = True
        except ValueError:
            # Running in a non-main thread (e.g. unit test thread)
            signals_installed = False

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

        resume_epoch = 1
        resume_step_in_epoch = 0
        resume_epoch_loss_sum = 0.0
        resume_mask_gen_state: torch.Tensor | None = None
        global_step = 0
        prior_wall_time = 0.0
        init_eval: dict[str, Any] | None = None
        exposures: list[dict[str, Any]] = []
        last_lr = peak_lr

        if resume or resume_from is not None:
            ckpt_to_load = find_resume_checkpoint(
                ckpt_root,
                run_spec,
                explicit_checkpoint=Path(resume_from) if resume_from is not None else None,
            )
            if ckpt_to_load is not None:
                ckpt = torch.load(ckpt_to_load, map_location=ctx.device, weights_only=False)
                verify_checkpoint_compatibility(ckpt, run_spec, steps_per_epoch=steps_per_epoch)
                ckpt_epoch_done = int(ckpt["epoch_done"])
                ckpt_step_in_epoch = int(ckpt.get("step_in_epoch", 0))
                ckpt_current_epoch = int(ckpt.get("current_epoch", ckpt_epoch_done + 1))

                if ckpt_epoch_done < run_spec.max_epochs:
                    raw_model.load_state_dict(ckpt["model_state_dict"])
                    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    resume_epoch = ckpt_current_epoch
                    resume_step_in_epoch = ckpt_step_in_epoch
                    resume_epoch_loss_sum = float(ckpt.get("epoch_loss_sum", 0.0))
                    global_step = int(
                        ckpt.get(
                            "global_step",
                            ckpt_epoch_done * steps_per_epoch + ckpt_step_in_epoch,
                        )
                    )
                    prior_wall_time = float(ckpt.get("elapsed_wall_time_sec", 0.0))
                    init_eval = ckpt.get("init_eval")
                    loaded_exps = ckpt.get("exposures", ckpt.get("evaluations", []))
                    exposures = [
                        rec for rec in loaded_exps if int(rec["epoch"]) <= ckpt_epoch_done
                    ]
                    if "last_lr" in ckpt:
                        last_lr = float(ckpt["last_lr"])
                    if "warmup_steps" in ckpt:
                        warmup_steps = int(ckpt["warmup_steps"])
                    if isinstance(ckpt.get("wsd_early_stop_state"), dict):
                        early_stop = AdaptiveEarlyStopTracker.from_dict(
                            ckpt["wsd_early_stop_state"],
                            fallback_config=es_cfg,
                        )
                    rng_states = ckpt.get("rng_states")
                    _restore_rng_states(rng_states)
                    if isinstance(rng_states, dict):
                        by_rank = rng_states.get("mask_gen_states_by_rank")
                        if (
                            isinstance(by_rank, list)
                            and ctx.rank < len(by_rank)
                            and by_rank[ctx.rank] is not None
                        ):
                            resume_mask_gen_state = by_rank[ctx.rank]
                        elif rng_states.get("mask_gen_state") is not None:
                            resume_mask_gen_state = rng_states["mask_gen_state"]

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
        last_ckpt_mono = time.monotonic()

        if run_init_eval and init_eval is None and resume_epoch == 1 and resume_step_in_epoch == 0:
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
                mask_rate=run_spec.mask_rate,
                mask_count=fixed_mask_k,
                mlm_fixed_num_draws=mlm_fixed_num_draws,
            )
            init_metrics["epoch"] = 0
            init_metrics["step"] = 0
            init_metrics["tokens_seen"] = 0
            init_eval = init_metrics

        eval_epoch_set = set(run_spec.eval_epochs)
        cross_eval_epoch_set = set(int(e) for e in (cross_eval_epochs or ()))
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
            "micro_step_batch_size": int(micro_step_batch_size),
            "grad_accum_steps": int(grad_accum_steps),
            "eval_batch_size": int(eval_batch_size),
            "steps_per_epoch": int(steps_per_epoch),
            "total_steps": int(total_steps),
            "warmup_steps": int(warmup_steps),
            "peak_lr": float(peak_lr),
            "min_lr": float(min_lr),
            "min_lr_ratio": float(run_spec.min_lr_ratio),
            "lr_schedule": run_spec.lr_schedule,
            "weight_decay": float(run_spec.weight_decay),
            "precision": run_spec.precision,
            "use_compile": bool(use_compile),
            "eval_max_sequences_mlm": int(eval_max_sequences_mlm),
            "mlm_fixed_mask_count": int(fixed_mask_k),
            "mlm_fixed_num_draws": int(mlm_fixed_num_draws),
            "adaptive_early_stop": es_cfg.to_dict(),
            "checkpoint_interval_sec": float(checkpoint_interval_sec),
        }

        mlm_loo_diag: dict[str, Any] | None = None
        mlm_fixed_15pct_diag: dict[str, Any] | None = None
        mlm_oa_16pt_diag: dict[str, Any] | None = None

        horizon_cd_epochs = (
            _compute_horizon_cooldown_epochs(run_spec.max_epochs, es_cfg)
            if run_spec.lr_schedule == "wsd"
            else 0
        )
        horizon_trigger_epoch = run_spec.max_epochs - horizon_cd_epochs

        def _save_latest_checkpoint(
            *,
            epoch_done_val: int,
            current_epoch_val: int,
            step_in_epoch_val: int,
            epoch_loss_sum_tensor: torch.Tensor,
            mask_gen_obj: torch.Generator | None,
            elapsed_val: float,
            target_path: Path,
        ) -> None:
            mask_states_by_rank: list[torch.Tensor | None] | None = None
            loss_for_ckpt = float(epoch_loss_sum_tensor.item())
            if ctx.is_distributed:
                avg_loss_t = epoch_loss_sum_tensor.detach().clone()
                dist.all_reduce(avg_loss_t, op=dist.ReduceOp.AVG)
                loss_for_ckpt = float(avg_loss_t.item())

                local_m = (
                    mask_gen_obj.get_state().detach().to(device="cpu", dtype=torch.uint8)
                    if mask_gen_obj is not None
                    else None
                )
                gathered: list[torch.Tensor | None] | None = (
                    [None for _ in range(ctx.world_size)] if ctx.rank == 0 else None
                )
                dist.gather_object(local_m, gathered, dst=0)
                if ctx.rank == 0:
                    mask_states_by_rank = gathered

            if ctx.rank == 0:
                ckpt_payload = {
                    "schema": CHECKPOINT_SCHEMA,
                    "run_spec": run_spec.to_dict(),
                    "run_spec_hash": run_spec.spec_hash(),
                    "epoch_done": int(epoch_done_val),
                    "current_epoch": int(current_epoch_val),
                    "step_in_epoch": int(step_in_epoch_val),
                    "global_step": int(global_step),
                    "steps_per_epoch": int(steps_per_epoch),
                    "global_batch_size": int(global_batch_size),
                    "warmup_steps": int(warmup_steps),
                    "last_lr": float(last_lr),
                    "epoch_loss_sum": float(loss_for_ckpt),
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rng_states": _capture_rng_states(
                        mask_gen_obj,
                        mask_gen_states_by_rank=mask_states_by_rank,
                    ),
                    "wsd_early_stop_state": early_stop.to_dict(),
                    "init_eval": init_eval,
                    "exposures": exposures,
                    "elapsed_wall_time_sec": float(elapsed_val),
                }
                _atomic_torch_save(target_path, ckpt_payload)
            if ctx.is_distributed:
                dist.barrier()

        final_epoch_done = min(
            run_spec.max_epochs,
            early_stop.stop_epoch if early_stop.stop_epoch is not None else run_spec.max_epochs,
        )

        for epoch in range(resume_epoch, run_spec.max_epochs + 1):
            # Activate horizon-triggered WSD cooldown if adaptive stopping did not fire earlier
            if (
                run_spec.lr_schedule == "wsd"
                and not early_stop.cooldown_triggered
                and horizon_cd_epochs > 0
                and epoch > horizon_trigger_epoch
            ):
                early_stop.activate_cooldown(
                    trigger_epoch=horizon_trigger_epoch,
                    reason="max_epochs_horizon",
                    global_step=horizon_trigger_epoch * steps_per_epoch,
                    steps_per_epoch=steps_per_epoch,
                    max_epochs=run_spec.max_epochs,
                    cooldown_epochs_override=horizon_cd_epochs,
                )

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
            if (
                epoch == resume_epoch
                and resume_step_in_epoch > 0
                and resume_mask_gen_state is not None
            ):
                # PyTorch Generator.set_state strictly requires a CPU ByteTensor even for CUDA generators.
                mask_gen.set_state(resume_mask_gen_state.to(device="cpu", dtype=torch.uint8))
                resume_mask_gen_state = None
            else:
                mask_gen.manual_seed(
                    int(run_spec.seed) * 10_000_019 + int(epoch) * 1_009 + int(ctx.rank)
                )

            start_step = resume_step_in_epoch if epoch == resume_epoch else 0
            epoch_loss_sum = torch.tensor(
                resume_epoch_loss_sum if (epoch == resume_epoch and start_step > 0) else 0.0,
                dtype=torch.float64,
                device=ctx.device,
            )
            if epoch == resume_epoch:
                resume_step_in_epoch = 0
                resume_epoch_loss_sum = 0.0

            for step_in_epoch in range(start_step, steps_per_epoch):
                global_step += 1
                if run_spec.lr_schedule == "wsd":
                    last_lr = compute_wsd_lr(
                        global_step,
                        peak_lr=peak_lr,
                        warmup_steps=warmup_steps,
                        min_lr_ratio=run_spec.min_lr_ratio,
                        cooldown_start_step=early_stop.cooldown_start_step,
                        cooldown_end_step=early_stop.cooldown_end_step,
                    )
                else:
                    last_lr = compute_cosine_lr(
                        global_step,
                        total_steps=total_steps,
                        peak_lr=peak_lr,
                        warmup_steps=warmup_steps,
                        min_lr_ratio=run_spec.min_lr_ratio,
                    )
                for group in optimizer.param_groups:
                    group["lr"] = last_lr

                step_indices = perm_steps[step_in_epoch, ctx.rank :: ctx.world_size]
                batch_u8 = data_gpu.index_select(0, step_indices)

                optimizer.zero_grad(set_to_none=True)

                if run_spec.paradigm == "mlm_15":
                    masked_input, targets, mask, n_mask = sample_mlm_fixed_rate_batch(
                        batch_u8,
                        mask_count=fixed_mask_k,
                        generator=mask_gen,
                        special_token_id=SPECIAL_TOKEN_ID,
                    )
                elif run_spec.paradigm != "ar":
                    masked_input, targets, mask, n_mask = sample_mlm_stratified_batch(
                        batch_u8,
                        generator=mask_gen,
                        pos_arange=pos_arange,
                        special_token_id=SPECIAL_TOKEN_ID,
                    )

                step_loss = torch.zeros((), dtype=torch.float64, device=ctx.device)
                for chunk_idx in range(grad_accum_steps):
                    c_start = chunk_idx * micro_step_cap
                    c_end = min(micro_batch_size, c_start + micro_step_cap)
                    c_len = c_end - c_start
                    c_weight = float(c_len) / float(micro_batch_size)
                    sync_ctx: Any = (
                        train_model.no_sync()
                        if (ctx.is_distributed and chunk_idx < grad_accum_steps - 1)
                        else contextlib.nullcontext()
                    )
                    with sync_ctx:
                        if run_spec.paradigm == "ar":
                            c_targets = batch_u8[c_start:c_end].to(dtype=torch.long)
                            c_input = torch.cat((bos_col[:c_len], c_targets[:, :-1]), dim=1)
                            with autocast_ctx:
                                logits = train_model(c_input)
                            chunk_loss = compute_ar_batch_loss(
                                logits, c_targets, vocab_size=VOCAB_AA
                            )
                        else:
                            with autocast_ctx:
                                logits = train_model(masked_input[c_start:c_end])
                            chunk_loss = compute_mlm_batch_loss(
                                logits,
                                targets[c_start:c_end],
                                mask[c_start:c_end],
                                n_mask[c_start:c_end],
                            )

                        if not bool(torch.isfinite(chunk_loss).item()):
                            raise FloatingPointError(
                                f"Non-finite training loss at epoch={epoch} step={global_step}: {chunk_loss.item()}"
                            )

                        scaled_loss = chunk_loss * c_weight if grad_accum_steps > 1 else chunk_loss
                        scaled_loss.backward()

                    step_loss += chunk_loss.detach().to(torch.float64) * c_weight
                    del logits, chunk_loss, scaled_loss

                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss_sum += step_loss

                # Spot-safe SIGTERM/SIGINT preemption & periodic time-based checkpoint check.
                # Under DDP, sync cadence MUST be strictly rank-invariant (never conditioned on
                # local signal delivery) so all ranks enter dist.all_reduce together without deadlock.
                next_step_in_epoch = step_in_epoch + 1
                if ctx.is_distributed:
                    should_check_sync = (next_step_in_epoch == steps_per_epoch) or (
                        next_step_in_epoch % 8 == 0
                    )
                else:
                    should_check_sync = True
                if should_check_sync:
                    need_preempt = preempt_signal != 0
                    need_periodic = (
                        checkpoint_interval_sec > 0.0
                        and (time.monotonic() - last_ckpt_mono) >= checkpoint_interval_sec
                    )
                    if ctx.is_distributed:
                        flags = torch.tensor(
                            [1 if need_preempt else 0, 1 if (ctx.rank == 0 and need_periodic) else 0],
                            dtype=torch.int64,
                            device=ctx.device,
                        )
                        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
                        need_preempt = bool(flags[0].item() > 0)
                        need_periodic = bool(flags[1].item() > 0)

                    if need_preempt or need_periodic:
                        elapsed_now = prior_wall_time + (time.perf_counter() - t_run_start)
                        _save_latest_checkpoint(
                            epoch_done_val=epoch - 1,
                            current_epoch_val=epoch,
                            step_in_epoch_val=next_step_in_epoch,
                            epoch_loss_sum_tensor=epoch_loss_sum,
                            mask_gen_obj=mask_gen,
                            elapsed_val=elapsed_now,
                            target_path=latest_ckpt_path,
                        )
                        last_ckpt_mono = time.monotonic()
                        if need_preempt:
                            raise SystemExit(PREEMPTION_EXIT_CODE)

            target_stop_epoch = (
                min(run_spec.max_epochs, early_stop.stop_epoch)
                if early_stop.stop_epoch is not None
                else run_spec.max_epochs
            )
            is_eval_epoch = (
                epoch in eval_epoch_set
                or epoch == run_spec.max_epochs
                or epoch == target_stop_epoch
            )

            if is_eval_epoch:
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
                    mask_rate=run_spec.mask_rate,
                    mask_count=fixed_mask_k,
                    mlm_fixed_num_draws=mlm_fixed_num_draws,
                )

                # Two-phase adaptive early-stopping update (rank 0 authoritative in DDP, broadcast to all)
                if run_spec.lr_schedule == "wsd":
                    if (not ctx.is_distributed) or ctx.rank == 0:
                        early_stop.update(
                            epoch=epoch,
                            eval_rec=eval_rec,
                            paradigm=run_spec.paradigm,
                            global_step=global_step,
                            steps_per_epoch=steps_per_epoch,
                            max_epochs=run_spec.max_epochs,
                            vocab_size=VOCAB_AA,
                        )
                    if ctx.is_distributed:
                        sync_state: list[Any] = [early_stop.to_dict() if ctx.rank == 0 else None]
                        dist.broadcast_object_list(sync_state, src=0)
                        if ctx.rank != 0:
                            early_stop = AdaptiveEarlyStopTracker.from_dict(
                                sync_state[0],
                                fallback_config=es_cfg,
                            )
                    target_stop_epoch = (
                        min(run_spec.max_epochs, early_stop.stop_epoch)
                        if early_stop.stop_epoch is not None
                        else run_spec.max_epochs
                    )

                is_done = epoch >= target_stop_epoch
                run_cross_diag = is_done or (epoch in cross_eval_epoch_set)

                if run_cross_diag and run_spec.paradigm in ("mlm_15", "oa_uniform"):
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

                    if run_spec.paradigm == "mlm_15":
                        mlm_oa_16pt_diag = evaluate_mlm_joint_bits_16pt(
                            eval_model,
                            data_gpu,
                            batch_size=eval_batch_size,
                            n_params_non_embedding=param_counts["n_params_non_embedding"],
                            n_params_total=param_counts["n_params_total"],
                            precision=run_spec.precision,
                            eval_max_sequences_mlm=eval_max_sequences_mlm,
                            eval_seed=run_spec.data_seed + 17,
                        )
                        eval_rec["mlm_oa_16pt_diagnostic"] = mlm_oa_16pt_diag
                    elif run_spec.paradigm == "oa_uniform":
                        mlm_fixed_15pct_diag = evaluate_mlm_fixed_rate_bits(
                            eval_model,
                            data_gpu,
                            batch_size=eval_batch_size,
                            n_params_non_embedding=param_counts["n_params_non_embedding"],
                            n_params_total=param_counts["n_params_total"],
                            mask_count=fixed_mask_k,
                            mask_rate=run_spec.mask_rate,
                            num_draws=mlm_fixed_num_draws,
                            precision=run_spec.precision,
                            eval_max_sequences_mlm=eval_max_sequences_mlm,
                            eval_seed=run_spec.data_seed + 47,
                        )
                        eval_rec["mlm_fixed_15pct_diagnostic"] = mlm_fixed_15pct_diag

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
                final_epoch_done = int(epoch)

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
                        mlm_fixed_15pct_diag=mlm_fixed_15pct_diag,
                        mlm_oa_16pt_diag=mlm_oa_16pt_diag,
                        early_stop=early_stop,
                        is_complete=is_done,
                    )
                    assert_no_forbidden_infra_strings(payload, label=run_spec.run_id)
                    write_json_with_sha256(run_json_path, payload)

                    if is_done:
                        if run_spec.save_checkpoint:
                            final_ckpt_path = ckpt_root / f"{run_spec.run_id}_e{epoch}.pt"
                            _atomic_torch_save(
                                final_ckpt_path,
                                {
                                    "schema": CHECKPOINT_SCHEMA,
                                    "run_spec": run_spec.to_dict(),
                                    "run_spec_hash": run_spec.spec_hash(),
                                    "epoch_done": int(epoch),
                                    "current_epoch": int(epoch + 1),
                                    "step_in_epoch": 0,
                                    "global_step": int(global_step),
                                    "steps_per_epoch": int(steps_per_epoch),
                                    "global_batch_size": int(global_batch_size),
                                    "warmup_steps": int(warmup_steps),
                                    "last_lr": float(last_lr),
                                    "model_state_dict": raw_model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "rng_states": _capture_rng_states(None),
                                    "wsd_early_stop_state": early_stop.to_dict(),
                                    "init_eval": init_eval,
                                    "exposures": exposures,
                                    "elapsed_wall_time_sec": float(elapsed_wall),
                                },
                            )
                        if eff_delete_transient and latest_ckpt_path.exists():
                            latest_ckpt_path.unlink()
                    elif run_spec.save_checkpoint:
                        _atomic_torch_save(
                            latest_ckpt_path,
                            {
                                "schema": CHECKPOINT_SCHEMA,
                                "run_spec": run_spec.to_dict(),
                                "run_spec_hash": run_spec.spec_hash(),
                                "epoch_done": int(epoch),
                                "current_epoch": int(epoch + 1),
                                "step_in_epoch": 0,
                                "global_step": int(global_step),
                                "steps_per_epoch": int(steps_per_epoch),
                                "global_batch_size": int(global_batch_size),
                                "warmup_steps": int(warmup_steps),
                                "last_lr": float(last_lr),
                                "model_state_dict": raw_model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "rng_states": _capture_rng_states(None),
                                "wsd_early_stop_state": early_stop.to_dict(),
                                "init_eval": init_eval,
                                "exposures": exposures,
                                "elapsed_wall_time_sec": float(elapsed_wall),
                            },
                        )
                        last_ckpt_mono = time.monotonic()

                if ctx.is_distributed:
                    dist.barrier()

                if is_done:
                    break

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
            epoch_done=final_epoch_done,
            elapsed_wall=total_elapsed,
            init_eval=init_eval,
            exposures=exposures,
            mlm_loo_diag=mlm_loo_diag,
            mlm_fixed_15pct_diag=mlm_fixed_15pct_diag,
            mlm_oa_16pt_diag=mlm_oa_16pt_diag,
            early_stop=early_stop,
            is_complete=True,
        )
    finally:
        if signals_installed:
            signal.signal(signal.SIGTERM, prev_sigterm)
            signal.signal(signal.SIGINT, prev_sigint)
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul
        torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn


train_memory_capacity_run = execute_run
run_training = execute_run
