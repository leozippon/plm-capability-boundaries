"""Stage-2 supply-demand specifications, matched model scales, and run contracts."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

AA20: str = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID: dict[str, int] = {aa: idx for idx, aa in enumerate(AA20)}
VOCAB_AA: int = 20

# Variable-length natural protein token layout:
# 0..19 = 20 canonical amino acids, 20 = <BOS>, 21 = <MASK>, 22 = <PAD>
BOS_TOKEN_ID: int = 20
MASK_TOKEN_ID: int = 21
PAD_TOKEN_ID: int = 22
INPUT_VOCAB_SIZE: int = 23

DEFAULT_MAX_SEQ_LEN: int = 384
DEFAULT_MIN_SEQ_LEN: int = 50
LOG2_20: float = math.log2(20.0)
LN2: float = math.log(2.0)

OBJECTIVES: tuple[str, ...] = ("ar", "mlm_15", "oa_uniform")
ALL_OBJECTIVES: tuple[str, ...] = ("ar", "mlm_15", "mlm_30", "mlm_50", "oa_uniform", "mdm_cosine")
FIXED_MASK_OBJECTIVE_RATES: dict[str, float] = {
    "mlm_15": 0.15,
    "mlm_30": 0.30,
    "mlm_50": 0.50,
}
OBJECTIVE_ALIASES: dict[str, str] = {
    "mlm": "oa_uniform",
    "mdm": "oa_uniform",
    "mdm_uniform": "oa_uniform",
    "oa_cosine": "mdm_cosine",
}
OBJECTIVE_DISPLAY_NAMES: dict[str, str] = {
    "ar": "AR",
    "mlm_15": "MLM-15%",
    "mlm_30": "MLM-30%",
    "mlm_50": "MLM-50%",
    "oa_uniform": "MDM",
    "mdm_cosine": "MDM-Cosine",
}
OPTIMIZERS: tuple[str, ...] = ("adamw", "muon")
PRECISIONS: tuple[str, ...] = ("bf16", "fp32_strict")
LR_SCHEDULES: tuple[str, ...] = ("wsd", "cosine")

DEFAULT_MILESTONE_EPOCHS: tuple[int, ...] = (1, 2, 5, 10, 20, 40, 80, 150, 300)

HIGH_VRAM_THRESHOLD_BYTES: int = 110 * (1024 ** 3)
MID_VRAM_THRESHOLD_BYTES: int = 56 * (1024 ** 3)
LOW_VRAM_THRESHOLD_BYTES: int = 32 * (1024 ** 3)


def canonicalize_objective(objective: str) -> str:
    """Canonicalize Stage-2 objective identifier, aliasing 'mdm'/'mlm' -> 'oa_uniform'."""
    raw = str(objective).strip().lower()
    canon = OBJECTIVE_ALIASES.get(raw, raw)
    if canon not in ALL_OBJECTIVES:
        raise ValueError(
            f"Unsupported objective {objective!r}; expected {ALL_OBJECTIVES} "
            f"(or aliases {tuple(OBJECTIVE_ALIASES)})"
        )
    return canon


def objective_display_name(objective: str) -> str:
    """Return canonical unified-theory display label ('AR', 'MLM-15%', 'MDM', etc.)."""
    canon = canonicalize_objective(objective)
    return OBJECTIVE_DISPLAY_NAMES.get(canon, canon.upper())


def is_fixed_mask_objective(objective: str) -> bool:
    return canonicalize_objective(objective) in FIXED_MASK_OBJECTIVE_RATES


def default_mask_rate_for_objective(objective: str) -> float:
    return float(FIXED_MASK_OBJECTIVE_RATES.get(canonicalize_objective(objective), 0.15))


def should_enable_gradient_checkpointing(
    scale_id: str,
    total_vram_bytes: int | None,
    requested: bool | None = None,
) -> bool:
    """Auto-enable activation checkpointing for 650m on <110 GiB GPUs unless overridden."""
    if requested is not None:
        return bool(requested)
    if scale_id == "650m" and total_vram_bytes is not None and total_vram_bytes < HIGH_VRAM_THRESHOLD_BYTES:
        return True
    return False


def vram_safe_micro_batch_cap(
    scale_id: str,
    total_vram_bytes: int | None,
    default_cap: int,
) -> int:
    """Return a VRAM-safe per-GPU micro-batch cap for Stage-2 variable-length runs (L<=384)."""
    cap = max(1, int(default_cap))
    if total_vram_bytes is None or total_vram_bytes <= 0:
        return cap
    if scale_id == "650m":
        if total_vram_bytes < LOW_VRAM_THRESHOLD_BYTES:
            return min(cap, 2)
        if total_vram_bytes < MID_VRAM_THRESHOLD_BYTES:
            return min(cap, 8)
        if total_vram_bytes < HIGH_VRAM_THRESHOLD_BYTES:
            return min(cap, 16)
        return min(cap, 32)
    if scale_id == "150m":
        if total_vram_bytes < LOW_VRAM_THRESHOLD_BYTES:
            return min(cap, 16)
        if total_vram_bytes < MID_VRAM_THRESHOLD_BYTES:
            return min(cap, 32)
        if total_vram_bytes < HIGH_VRAM_THRESHOLD_BYTES:
            return min(cap, 64)
    return cap

FORBIDDEN_INFRA_RE = re.compile(
    r"(?:\b(?:[a-z0-9_-]*master-\d+|[a-z0-9_-]*worker-\d+|worker-\d+\.k8s|node[-_]?\d*|10\.\d+\.\d+\.\d+)\b)",
    re.IGNORECASE,
)


def slugify_token(value: str | float | int) -> str:
    """Convert a scale or numeric token into a clean ASCII identifier slug."""
    raw = str(value).strip().lower().replace(".", "p").replace("-", "_")
    cleaned = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")
    if not cleaned:
        raise ValueError(f"Cannot slugify empty token from {value!r}")
    if FORBIDDEN_INFRA_RE.search(cleaned):
        raise ValueError(f"Slug {cleaned!r} matched forbidden infrastructure pattern")
    return cleaned


def assert_no_forbidden_infra_strings(obj: Any, *, label: str = "payload") -> None:
    """Recursively verify that no cluster/pod/node/IP identifier appears in persisted data."""
    if isinstance(obj, str):
        match = FORBIDDEN_INFRA_RE.search(obj)
        if match is not None:
            raise ValueError(
                f"{label} contains forbidden cluster/pod identifier: {match.group(0)!r}"
            )
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            assert_no_forbidden_infra_strings(k, label=f"{label}.key")
            assert_no_forbidden_infra_strings(v, label=f"{label}[{k!r}]")
    elif isinstance(obj, (list, tuple, set)):
        for idx, item in enumerate(obj):
            assert_no_forbidden_infra_strings(item, label=f"{label}[{idx}]")


@dataclass(frozen=True)
class ScaleSpec:
    """Matched Pre-LN Transformer scale specification with exact Stage-1 non-embedding count."""

    scale_id: str
    nominal_params: int
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    base_lr: float
    max_batch_size: int

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.d_model < 1 or self.n_heads < 1 or self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError(
                f"head_dim ({self.d_model // self.n_heads}) must be even for RoPE"
            )
        if self.d_ff < 1:
            raise ValueError(f"d_ff must be >= 1, got {self.d_ff}")
        if self.base_lr <= 0.0:
            raise ValueError(f"base_lr must be > 0, got {self.base_lr}")
        if self.max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {self.max_batch_size}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def exact_non_embedding_params(self) -> int:
        per_layer = (
            4 * (self.d_model ** 2)
            + 3 * self.d_model * self.d_ff
            + 2 * self.d_model
            + 2 * (self.d_model // self.n_heads)
        )
        return self.n_layers * per_layer + self.d_model

    @property
    def exact_embedding_params(self) -> int:
        return (INPUT_VOCAB_SIZE + VOCAB_AA) * self.d_model

    @property
    def exact_total_params(self) -> int:
        return self.exact_non_embedding_params + self.exact_embedding_params

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale_id": self.scale_id,
            "nominal_params": self.nominal_params,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "base_lr": self.base_lr,
            "max_batch_size": self.max_batch_size,
            "exact_non_embedding_params": self.exact_non_embedding_params,
            "exact_embedding_params": self.exact_embedding_params,
            "exact_total_params": self.exact_total_params,
        }


SCALE_ORDER: tuple[str, ...] = ("0.5m", "2m", "8m", "35m", "150m")
EXTENDED_SCALE_ORDER: tuple[str, ...] = ("0.5m", "2m", "8m", "35m", "150m", "650m")

# Exact layer/head/width geometry matching Stage-1 memory_capacity SCALE_SPECS:
# exact_non_embedding_params = 492,608 / 1,992,640 / 7,992,128 / 35,337,088 / 148,671,360 / 654,303,104
SCALE_SPECS: dict[str, ScaleSpec] = {
    "0.5m": ScaleSpec("0.5m", 500_000, 3, 128, 4, 256, 2.0e-3, 256),
    "2m": ScaleSpec("2m", 2_000_000, 4, 192, 6, 608, 1.5e-3, 256),
    "8m": ScaleSpec("8m", 8_000_000, 6, 320, 5, 960, 1.2e-3, 256),
    "35m": ScaleSpec("35m", 35_000_000, 11, 512, 8, 1408, 8.0e-4, 128),
    "150m": ScaleSpec("150m", 150_000_000, 21, 768, 12, 2048, 6.0e-4, 64),
    "650m": ScaleSpec("650m", 650_000_000, 33, 1280, 20, 3456, 3.0e-4, 32),
}


def choose_milestone_epochs(
    max_epochs: int,
    base_schedule: tuple[int, ...] = DEFAULT_MILESTONE_EPOCHS,
) -> tuple[int, ...]:
    """Return sorted log-spaced evaluation epochs up to and including max_epochs."""
    if max_epochs < 1:
        raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
    milestones = [int(e) for e in base_schedule if 1 <= int(e) <= max_epochs]
    if max_epochs > (base_schedule[-1] if base_schedule else 0):
        cur = base_schedule[-1] * 2 if base_schedule else 1
        while cur < max_epochs:
            milestones.append(cur)
            cur *= 2
    if not milestones or milestones[-1] != max_epochs:
        milestones.append(int(max_epochs))
    return tuple(sorted(set(milestones)))


@dataclass(frozen=True)
class Stage2RunSpec:
    """Immutable specification for one Stage-2 controlled sampling training run."""

    run_id: str
    arm_id: str
    objective: str
    scale_id: str
    max_epochs: int
    eval_epochs: tuple[int, ...] = ()
    lr_schedule: str = "wsd"
    peak_lr_override: float | None = None
    warmup_steps_override: int | None = None
    cooldown_fraction: float = 0.10
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.01
    precision: str = "bf16"
    batch_size: int = 128
    grad_accum_steps: int = 1
    seed: int = 20261128
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    checkpoint_interval_sec: float = 300.0
    save_final_checkpoint: bool = True
    eval_downstream_at_milestones: bool = True
    eval_contacts_at_milestones: bool = True
    eval_full_proteingym_at_final: bool = True
    eval_temperature_calibration: bool = True
    optimizer: str = "adamw"
    muon_lr: float = 0.02
    muon_lr_scale: float | None = None
    mask_rate: float | None = None
    gradient_checkpointing: bool | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        assert_no_forbidden_infra_strings(self.run_id, label="run_id")
        if not self.arm_id:
            raise ValueError("arm_id must be non-empty")
        assert_no_forbidden_infra_strings(self.arm_id, label="arm_id")
        canon_obj = canonicalize_objective(self.objective)
        object.__setattr__(self, "objective", canon_obj)
        if self.scale_id not in SCALE_SPECS:
            raise ValueError(
                f"Unknown scale_id {self.scale_id!r}; expected {tuple(SCALE_SPECS)}"
            )
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {self.max_epochs}")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"Unsupported lr_schedule {self.lr_schedule!r}; expected {LR_SCHEDULES}"
            )
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"Unsupported precision {self.precision!r}; expected {PRECISIONS}"
            )
        opt_norm = str(self.optimizer).strip().lower()
        if opt_norm not in OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer {self.optimizer!r}; expected {OPTIMIZERS}")
        object.__setattr__(self, "optimizer", opt_norm)
        if float(self.muon_lr) <= 0.0 or not math.isfinite(float(self.muon_lr)):
            raise ValueError(f"muon_lr must be positive finite, got {self.muon_lr!r}")
        if self.muon_lr_scale is not None and (
            float(self.muon_lr_scale) <= 0.0 or not math.isfinite(float(self.muon_lr_scale))
        ):
            raise ValueError(f"muon_lr_scale must be positive finite, got {self.muon_lr_scale!r}")
        if self.mask_rate is not None and (
            not (0.0 < float(self.mask_rate) <= 1.0) or not math.isfinite(float(self.mask_rate))
        ):
            raise ValueError(f"mask_rate must be in (0, 1], got {self.mask_rate!r}")
        if not (0.0 <= self.cooldown_fraction < 1.0):
            raise ValueError(f"cooldown_fraction must be in [0, 1), got {self.cooldown_fraction}")
        if not (0.0 < self.min_lr_ratio <= 1.0):
            raise ValueError(f"min_lr_ratio must be in (0, 1], got {self.min_lr_ratio}")
        if self.weight_decay < 0.0 or not math.isfinite(self.weight_decay):
            raise ValueError(f"Invalid weight_decay {self.weight_decay!r}")
        if self.batch_size < 1 or self.grad_accum_steps < 1:
            raise ValueError(
                f"batch_size ({self.batch_size}) and grad_accum_steps ({self.grad_accum_steps}) must be >= 1"
            )
        if self.max_seq_len < 8:
            raise ValueError(f"max_seq_len must be >= 8, got {self.max_seq_len}")
        if self.checkpoint_interval_sec <= 0.0:
            raise ValueError(
                f"checkpoint_interval_sec must be > 0, got {self.checkpoint_interval_sec}"
            )

        if self.eval_epochs:
            norm = tuple(
                sorted({int(e) for e in self.eval_epochs if 1 <= int(e) <= self.max_epochs})
            )
            if not norm or norm[-1] != self.max_epochs:
                norm = tuple(sorted({*norm, int(self.max_epochs)}))
        else:
            norm = choose_milestone_epochs(self.max_epochs)
        object.__setattr__(self, "eval_epochs", norm)

    @property
    def effective_mask_rate(self) -> float:
        if self.mask_rate is not None:
            return float(self.mask_rate)
        return default_mask_rate_for_objective(self.objective)

    @property
    def paradigm_display(self) -> str:
        return objective_display_name(self.objective)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "arm_id": self.arm_id,
            "objective": self.objective,
            "scale_id": self.scale_id,
            "max_epochs": int(self.max_epochs),
            "eval_epochs": [int(e) for e in self.eval_epochs],
            "lr_schedule": self.lr_schedule,
            "peak_lr_override": (
                None if self.peak_lr_override is None else float(self.peak_lr_override)
            ),
            "warmup_steps_override": (
                None if self.warmup_steps_override is None else int(self.warmup_steps_override)
            ),
            "cooldown_fraction": float(self.cooldown_fraction),
            "min_lr_ratio": float(self.min_lr_ratio),
            "weight_decay": float(self.weight_decay),
            "precision": self.precision,
            "batch_size": int(self.batch_size),
            "grad_accum_steps": int(self.grad_accum_steps),
            "seed": int(self.seed),
            "max_seq_len": int(self.max_seq_len),
            "checkpoint_interval_sec": float(self.checkpoint_interval_sec),
            "save_final_checkpoint": bool(self.save_final_checkpoint),
            "eval_downstream_at_milestones": bool(self.eval_downstream_at_milestones),
            "eval_contacts_at_milestones": bool(self.eval_contacts_at_milestones),
            "eval_full_proteingym_at_final": bool(self.eval_full_proteingym_at_final),
            "eval_temperature_calibration": bool(self.eval_temperature_calibration),
            "optimizer": self.optimizer,
            "muon_lr": float(self.muon_lr),
            "muon_lr_scale": None if self.muon_lr_scale is None else float(self.muon_lr_scale),
            "mask_rate": None if self.mask_rate is None else float(self.mask_rate),
            "gradient_checkpointing": self.gradient_checkpointing,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Stage2RunSpec:
        raw_gc = data.get("gradient_checkpointing", None)
        return cls(
            run_id=str(data["run_id"]),
            arm_id=str(data["arm_id"]),
            objective=canonicalize_objective(str(data["objective"])),
            scale_id=str(data["scale_id"]),
            max_epochs=int(data["max_epochs"]),
            eval_epochs=tuple(int(e) for e in data.get("eval_epochs", ())),
            lr_schedule=str(data.get("lr_schedule", "wsd")),
            peak_lr_override=(
                None
                if data.get("peak_lr_override") is None
                else float(data["peak_lr_override"])
            ),
            warmup_steps_override=(
                None
                if data.get("warmup_steps_override") is None
                else int(data["warmup_steps_override"])
            ),
            cooldown_fraction=float(data.get("cooldown_fraction", 0.10)),
            min_lr_ratio=float(data.get("min_lr_ratio", 0.05)),
            weight_decay=float(data.get("weight_decay", 0.01)),
            precision=str(data.get("precision", "bf16")),
            batch_size=int(data.get("batch_size", 128)),
            grad_accum_steps=int(data.get("grad_accum_steps", 1)),
            seed=int(data.get("seed", 20261128)),
            max_seq_len=int(data.get("max_seq_len", DEFAULT_MAX_SEQ_LEN)),
            checkpoint_interval_sec=float(data.get("checkpoint_interval_sec", 300.0)),
            save_final_checkpoint=bool(data.get("save_final_checkpoint", True)),
            eval_downstream_at_milestones=bool(
                data.get("eval_downstream_at_milestones", True)
            ),
            eval_contacts_at_milestones=bool(data.get("eval_contacts_at_milestones", True)),
            eval_full_proteingym_at_final=bool(
                data.get("eval_full_proteingym_at_final", True)
            ),
            eval_temperature_calibration=bool(
                data.get("eval_temperature_calibration", True)
            ),
            optimizer=str(data.get("optimizer", "adamw")),
            muon_lr=float(data.get("muon_lr", 0.02)),
            muon_lr_scale=(
                None if data.get("muon_lr_scale") is None else float(data["muon_lr_scale"])
            ),
            mask_rate=None if data.get("mask_rate") is None else float(data["mask_rate"]),
            gradient_checkpointing=None if raw_gc is None else bool(raw_gc),
        )
