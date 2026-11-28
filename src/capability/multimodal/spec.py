"""Specifications for the ESM-3-style controlled multimodal protein transformer suite.

Aligns strictly to ``docs/foundational-plm-theory.md`` notation:
  - ``E``, ``E_gen``, ``E_mem``, ``W = E_mem / E_gen``, ``Delta_cal``
  - ``MDM`` (uniform masked diffusion, aliased cleanly with ``"oa_uniform"``)
  - ``B_ext(Z)`` and per-residue ``b_ext(Z) = H(X) - H(X | Z)``
  - Exact matched transformer trunk ``n_params_non_embedding`` across ``0.5m..650m``
  - Three controlled training/inference modes:
      1. ``pure_seq``: pure sequence training & inference (B_ext(Z) = 0).
      2. ``struct_cond``: structure-conditioned training & inference (B_ext(Z) > 0, Eq. 4).
      3. ``joint_to_seq``: ESM-3-style joint (AA + 3Di + contact) co-training with 50%
         independent structure drop so Z = empty is in-distribution, evaluated strictly
         from pure AA sequence (Z = empty, B_ext = 0).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..supply_demand.spec import (
    AA20,
    AA_TO_ID,
    ALL_OBJECTIVES,
    BOS_TOKEN_ID,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_MILESTONE_EPOCHS,
    EXTENDED_SCALE_ORDER,
    INPUT_VOCAB_SIZE as AA_INPUT_VOCAB_SIZE,
    LN2,
    LOG2_20,
    LR_SCHEDULES,
    MASK_TOKEN_ID,
    OBJECTIVES,
    OPTIMIZERS,
    PAD_TOKEN_ID,
    PRECISIONS,
    SCALE_ORDER,
    SCALE_SPECS,
    VOCAB_AA,
    ScaleSpec,
    assert_no_forbidden_infra_strings,
    canonicalize_objective,
    choose_milestone_epochs,
    default_mask_rate_for_objective,
    is_fixed_mask_objective,
    objective_display_name,
    should_enable_gradient_checkpointing,
    slugify_token,
    vram_safe_micro_batch_cap,
)

MULTIMODAL_MODES: tuple[str, ...] = ("pure_seq", "struct_cond", "joint_to_seq")

# Foldseek 3Di alphabet uses the 20 standard uppercase letters A..Y
THREEDI_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWY"
THREEDI_TO_ID: dict[str, int] = {ch: idx for idx, ch in enumerate(THREEDI_ALPHABET)}
VOCAB_3DI: int = 20
LOG2_3DI: float = math.log2(float(VOCAB_3DI))

# Structure 3Di track token layout:
# 0..19 = 20 Foldseek 3Di tokens, 20 = <STRUCT_MASK>, 21 = <STRUCT_PAD>, 22 = <STRUCT_ABSENT> (Z = empty)
STRUCT_MASK_TOKEN_ID: int = 20
STRUCT_PAD_TOKEN_ID: int = 21
STRUCT_ABSENT_TOKEN_ID: int = 22
STRUCT_INPUT_VOCAB_SIZE: int = 23

# Rotation/translation-invariant per-residue local geometry feature dimension:
# 8 sequential CB inverse-distances (k in -4..-1, +1..+4) + 6 backbone dihedrals (sin/cos phi,psi,omega)
# + 3 non-local contact density features (8A, 12A, 16A) + 1 valid mask = 18
GEOM_FEAT_DIM: int = 18

# Pairwise C_beta distance-bin edges (Angstroms) for attention bias:
# bin 0 = absent / missing / masked structure; bins 1..14 = distance intervals; bin 15 = >22A
PAIR_DIST_BIN_EDGES: tuple[float, ...] = (
    3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0
)
PAIR_DIST_NUM_BINS: int = len(PAIR_DIST_BIN_EDGES) + 2  # 16 bins (0=absent, 1..15=distance bins)
CONTACT_DISTANCE_CUTOFF: float = 8.0


def canonicalize_mode(mode: str) -> str:
    raw = str(mode).strip().lower()
    aliases = {
        "seq_only": "pure_seq",
        "esm3_joint": "joint_to_seq",
        "joint": "joint_to_seq",
        "conditioned": "struct_cond",
    }
    canon = aliases.get(raw, raw)
    if canon not in MULTIMODAL_MODES:
        raise ValueError(
            f"Unsupported multimodal mode {mode!r}; expected {MULTIMODAL_MODES}"
        )
    return canon


@dataclass(frozen=True)
class MultimodalRunSpec:
    """Immutable specification for one controlled multimodal protein transformer run."""

    run_id: str
    mode: str
    objective: str
    scale_id: str
    max_epochs: int
    eval_epochs: tuple[int, ...] = ()
    lambda_struct: float = 1.0
    lambda_contact: float = 0.25
    struct_drop_prob: float = 0.50
    struct_mask_prob: float = 0.50
    use_pair_geom: bool = True
    lr_schedule: str = "wsd"
    peak_lr_override: float | None = None
    warmup_steps_override: int | None = None
    cooldown_fraction: float = 0.10
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.01
    precision: str = "bf16"
    optimizer: str = "adamw"
    muon_lr: float = 0.02
    muon_lr_scale: float | None = None
    mask_rate: float | None = None
    batch_size: int = 64
    grad_accum_steps: int = 1
    seed: int = 20261128
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    checkpoint_interval_sec: float = 300.0
    save_final_checkpoint: bool = True
    eval_downstream_at_milestones: bool = True
    eval_contacts_at_milestones: bool = True
    eval_temperature_calibration: bool = True
    gradient_checkpointing: bool | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        assert_no_forbidden_infra_strings(self.run_id, label="run_id")
        object.__setattr__(self, "mode", canonicalize_mode(self.mode))
        object.__setattr__(self, "objective", canonicalize_objective(self.objective))
        if self.scale_id not in SCALE_SPECS:
            raise ValueError(f"Unknown scale_id {self.scale_id!r}; expected {tuple(SCALE_SPECS)}")
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {self.max_epochs}")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"Unsupported lr_schedule {self.lr_schedule!r}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"Unsupported precision {self.precision!r}")
        opt_norm = str(self.optimizer).strip().lower()
        if opt_norm not in OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer {self.optimizer!r}; expected {OPTIMIZERS}")
        object.__setattr__(self, "optimizer", opt_norm)
        if not (0.0 <= float(self.struct_drop_prob) <= 1.0):
            raise ValueError(f"struct_drop_prob must be in [0, 1], got {self.struct_drop_prob}")
        if not (0.0 <= float(self.struct_mask_prob) <= 1.0):
            raise ValueError(f"struct_mask_prob must be in [0, 1], got {self.struct_mask_prob}")
        if float(self.lambda_struct) < 0.0 or float(self.lambda_contact) < 0.0:
            raise ValueError("lambda_struct and lambda_contact must be >= 0")
        if self.batch_size < 1 or self.grad_accum_steps < 1:
            raise ValueError("batch_size and grad_accum_steps must be >= 1")
        if self.eval_epochs:
            norm = tuple(sorted({int(e) for e in self.eval_epochs if 1 <= int(e) <= self.max_epochs}))
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
            "mode": self.mode,
            "objective": self.objective,
            "scale_id": self.scale_id,
            "max_epochs": int(self.max_epochs),
            "eval_epochs": [int(e) for e in self.eval_epochs],
            "lambda_struct": float(self.lambda_struct),
            "lambda_contact": float(self.lambda_contact),
            "struct_drop_prob": float(self.struct_drop_prob),
            "struct_mask_prob": float(self.struct_mask_prob),
            "use_pair_geom": bool(self.use_pair_geom),
            "lr_schedule": self.lr_schedule,
            "peak_lr_override": None if self.peak_lr_override is None else float(self.peak_lr_override),
            "warmup_steps_override": (
                None if self.warmup_steps_override is None else int(self.warmup_steps_override)
            ),
            "cooldown_fraction": float(self.cooldown_fraction),
            "min_lr_ratio": float(self.min_lr_ratio),
            "weight_decay": float(self.weight_decay),
            "precision": self.precision,
            "optimizer": self.optimizer,
            "muon_lr": float(self.muon_lr),
            "muon_lr_scale": None if self.muon_lr_scale is None else float(self.muon_lr_scale),
            "mask_rate": None if self.mask_rate is None else float(self.mask_rate),
            "batch_size": int(self.batch_size),
            "grad_accum_steps": int(self.grad_accum_steps),
            "seed": int(self.seed),
            "max_seq_len": int(self.max_seq_len),
            "checkpoint_interval_sec": float(self.checkpoint_interval_sec),
            "save_final_checkpoint": bool(self.save_final_checkpoint),
            "eval_downstream_at_milestones": bool(self.eval_downstream_at_milestones),
            "eval_contacts_at_milestones": bool(self.eval_contacts_at_milestones),
            "eval_temperature_calibration": bool(self.eval_temperature_calibration),
            "gradient_checkpointing": self.gradient_checkpointing,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MultimodalRunSpec:
        raw_gc = data.get("gradient_checkpointing", None)
        return cls(
            run_id=str(data["run_id"]),
            mode=str(data["mode"]),
            objective=str(data["objective"]),
            scale_id=str(data["scale_id"]),
            max_epochs=int(data["max_epochs"]),
            eval_epochs=tuple(int(e) for e in data.get("eval_epochs", ())),
            lambda_struct=float(data.get("lambda_struct", 1.0)),
            lambda_contact=float(data.get("lambda_contact", 0.25)),
            struct_drop_prob=float(data.get("struct_drop_prob", 0.50)),
            struct_mask_prob=float(data.get("struct_mask_prob", 0.50)),
            use_pair_geom=bool(data.get("use_pair_geom", True)),
            lr_schedule=str(data.get("lr_schedule", "wsd")),
            peak_lr_override=(
                None if data.get("peak_lr_override") is None else float(data["peak_lr_override"])
            ),
            warmup_steps_override=(
                None if data.get("warmup_steps_override") is None else int(data["warmup_steps_override"])
            ),
            cooldown_fraction=float(data.get("cooldown_fraction", 0.10)),
            min_lr_ratio=float(data.get("min_lr_ratio", 0.05)),
            weight_decay=float(data.get("weight_decay", 0.01)),
            precision=str(data.get("precision", "bf16")),
            optimizer=str(data.get("optimizer", "adamw")),
            muon_lr=float(data.get("muon_lr", 0.02)),
            muon_lr_scale=None if data.get("muon_lr_scale") is None else float(data["muon_lr_scale"]),
            mask_rate=None if data.get("mask_rate") is None else float(data["mask_rate"]),
            batch_size=int(data.get("batch_size", 64)),
            grad_accum_steps=int(data.get("grad_accum_steps", 1)),
            seed=int(data.get("seed", 20261128)),
            max_seq_len=int(data.get("max_seq_len", DEFAULT_MAX_SEQ_LEN)),
            checkpoint_interval_sec=float(data.get("checkpoint_interval_sec", 300.0)),
            save_final_checkpoint=bool(data.get("save_final_checkpoint", True)),
            eval_downstream_at_milestones=bool(data.get("eval_downstream_at_milestones", True)),
            eval_contacts_at_milestones=bool(data.get("eval_contacts_at_milestones", True)),
            eval_temperature_calibration=bool(data.get("eval_temperature_calibration", True)),
            gradient_checkpointing=None if raw_gc is None else bool(raw_gc),
        )


__all__ = [
    "AA20",
    "AA_INPUT_VOCAB_SIZE",
    "AA_TO_ID",
    "ALL_OBJECTIVES",
    "BOS_TOKEN_ID",
    "CONTACT_DISTANCE_CUTOFF",
    "DEFAULT_MAX_SEQ_LEN",
    "DEFAULT_MILESTONE_EPOCHS",
    "EXTENDED_SCALE_ORDER",
    "GEOM_FEAT_DIM",
    "LN2",
    "LOG2_20",
    "LOG2_3DI",
    "LR_SCHEDULES",
    "MASK_TOKEN_ID",
    "MULTIMODAL_MODES",
    "MultimodalRunSpec",
    "OBJECTIVES",
    "OPTIMIZERS",
    "PAD_TOKEN_ID",
    "PAIR_DIST_BIN_EDGES",
    "PAIR_DIST_NUM_BINS",
    "PRECISIONS",
    "SCALE_ORDER",
    "SCALE_SPECS",
    "STRUCT_ABSENT_TOKEN_ID",
    "STRUCT_INPUT_VOCAB_SIZE",
    "STRUCT_MASK_TOKEN_ID",
    "STRUCT_PAD_TOKEN_ID",
    "ScaleSpec",
    "THREEDI_ALPHABET",
    "THREEDI_TO_ID",
    "VOCAB_3DI",
    "VOCAB_AA",
    "assert_no_forbidden_infra_strings",
    "canonicalize_mode",
    "canonicalize_objective",
    "choose_milestone_epochs",
    "default_mask_rate_for_objective",
    "is_fixed_mask_objective",
    "objective_display_name",
    "should_enable_gradient_checkpointing",
    "slugify_token",
    "vram_safe_micro_batch_cap",
]
