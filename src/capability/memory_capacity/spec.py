"""Parametric memory-capacity specifications, scaling ladder, and campaign builder."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SEQ_LEN: int = 256
VOCAB_AA: int = 20
SPECIAL_TOKEN_ID: int = 20  # <BOS> for AR, <MASK> for MLM
INPUT_VOCAB_SIZE: int = 21
LOG2_20: float = math.log2(20.0)
LN2: float = math.log(2.0)
C_GRID_PRIOR_BITS_PER_PARAM: float = 3.5
MLM_15_MASK_RATE: float = 0.15

EVAL_EPOCHS: tuple[int, ...] = (
    1,
    2,
    5,
    10,
    20,
    35,
    50,
    75,
    100,
    150,
    200,
    300,
    400,
    600,
    800,
    1000,
    1500,
    2000,
    3000,
    4000,
    6000,
    8000,
    10000,
)
EXPOSURE_CHECKPOINTS: tuple[int, ...] = EVAL_EPOCHS
CAPACITY_MULTIPLIERS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

PARADIGMS: tuple[str, ...] = ("ar", "mlm_15", "oa_uniform")
LEGACY_PARADIGM_ALIASES: dict[str, str] = {"mlm": "oa_uniform"}
LR_SCHEDULES: tuple[str, ...] = ("wsd", "cosine")
CAMPAIGN_TRACKS: tuple[str, ...] = (
    "primary",
    "stage1_wsd",
    "ablation_wd",
    "ablation_fp32",
    "ablation_lr_horizon",
)
PRECISIONS: tuple[str, ...] = ("bf16", "fp32_strict")

FORBIDDEN_INFRA_RE = re.compile(
    r"(?:\b(?:[a-z0-9_-]*master-\d+|[a-z0-9_-]*worker-\d+|worker-\d+\.k8s|node[-_]?\d*|10\.\d+\.\d+\.\d+)\b)",
    re.IGNORECASE,
)


def canonicalize_paradigm(paradigm: str) -> str:
    """Canonicalize paradigm identifier, aliasing legacy 'mlm' -> 'oa_uniform'."""
    raw = str(paradigm).strip().lower()
    canon = LEGACY_PARADIGM_ALIASES.get(raw, raw)
    if canon not in PARADIGMS:
        raise ValueError(
            f"Unsupported paradigm {paradigm!r}; expected {PARADIGMS} "
            f"(or legacy alias {tuple(LEGACY_PARADIGM_ALIASES)})"
        )
    return canon


def is_mlm_paradigm(paradigm: str) -> bool:
    """Return True for bidirectional masked objectives ('mlm_15', 'oa_uniform', legacy 'mlm')."""
    return canonicalize_paradigm(paradigm) in ("mlm_15", "oa_uniform")


def mlm_fixed_mask_count(
    seq_len: int = SEQ_LEN,
    mask_rate: float = MLM_15_MASK_RATE,
) -> int:
    """Exact masked token count k = max(1, round(mask_rate * seq_len)), clamped to seq_len."""
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if not (0.0 < float(mask_rate) <= 1.0) or not math.isfinite(float(mask_rate)):
        raise ValueError(f"mask_rate must be in (0, 1], got {mask_rate!r}")
    return min(int(seq_len), max(1, int(round(float(mask_rate) * float(seq_len)))))


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
    """Architecture and optimization specification for one model scale rung."""

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
            "exact_total_params": self.exact_total_params,
        }


SCALE_ORDER: tuple[str, ...] = ("0.5m", "2m", "8m", "35m", "150m")

SCALE_SPECS: dict[str, ScaleSpec] = {
    "0.5m": ScaleSpec("0.5m", 500_000, 3, 128, 4, 256, 2.0e-3, 512),
    "2m": ScaleSpec("2m", 2_000_000, 4, 192, 6, 608, 1.5e-3, 1024),
    "8m": ScaleSpec("8m", 8_000_000, 6, 320, 5, 960, 1.2e-3, 1024),
    "35m": ScaleSpec("35m", 35_000_000, 11, 512, 8, 1408, 8.0e-4, 512),
    "150m": ScaleSpec("150m", 150_000_000, 21, 768, 12, 2048, 6.0e-4, 512),
}


def null_ordered_entropy_bits(
    n_sequences: int,
    seq_len: int = SEQ_LEN,
    vocab_size: int = VOCAB_AA,
) -> float:
    """Ordered tuple entropy N * L * log2(V) in bits."""
    if n_sequences < 0:
        raise ValueError(f"n_sequences must be >= 0, got {n_sequences}")
    if seq_len < 1 or vocab_size < 2:
        raise ValueError(f"Invalid seq_len={seq_len} or vocab_size={vocab_size}")
    if n_sequences == 0:
        return 0.0
    return float(n_sequences) * float(seq_len) * math.log2(float(vocab_size))


def dataset_entropy_bits(
    n_sequences: int,
    seq_len: int = SEQ_LEN,
    vocab_size: int = VOCAB_AA,
) -> float:
    """Unordered distinct-set entropy N * L * log2(V) - log2(N!) in bits."""
    if n_sequences < 0:
        raise ValueError(f"n_sequences must be >= 0, got {n_sequences}")
    if seq_len < 1 or vocab_size < 2:
        raise ValueError(f"Invalid seq_len={seq_len} or vocab_size={vocab_size}")
    if n_sequences == 0:
        return 0.0
    return (
        float(n_sequences) * float(seq_len) * math.log2(float(vocab_size))
        - math.lgamma(n_sequences + 1) / LN2
    )


def gibbs_memorizable_bits(
    n_sequences: int,
    seq_len: int = SEQ_LEN,
    vocab_size: int = VOCAB_AA,
) -> float:
    """Maximum KL reduction N * (L * log2(V) - log2(N)) achievable by q(x_i) = 1/N."""
    if seq_len < 1 or vocab_size < 2:
        raise ValueError(f"Invalid seq_len={seq_len} or vocab_size={vocab_size}")
    if n_sequences <= 0:
        return 0.0
    return float(n_sequences) * (
        float(seq_len) * math.log2(float(vocab_size)) - math.log2(float(n_sequences))
    )


def solve_sequence_count_for_entropy(
    target_bits: float,
    seq_len: int = SEQ_LEN,
    vocab_size: int = VOCAB_AA,
) -> int:
    """Exact binary search for smallest N >= 1 with dataset_entropy_bits(N) >= target_bits."""
    if not math.isfinite(target_bits):
        raise ValueError(f"target_bits must be finite, got {target_bits!r}")
    if seq_len < 1 or vocab_size < 2:
        raise ValueError(f"Invalid seq_len={seq_len} or vocab_size={vocab_size}")

    seq_null_bits = float(seq_len) * math.log2(float(vocab_size))
    if target_bits <= dataset_entropy_bits(1, seq_len=seq_len, vocab_size=vocab_size):
        return 1

    lo = 1
    hi = max(2, int(target_bits / max(1.0, seq_null_bits - 32.0)))
    while dataset_entropy_bits(hi, seq_len=seq_len, vocab_size=vocab_size) < target_bits:
        if math.log2(float(hi)) >= seq_null_bits:
            raise ValueError(
                f"target_bits={target_bits} exceeds maximum distinct-set entropy for "
                f"seq_len={seq_len}, vocab_size={vocab_size}"
            )
        lo = hi
        hi *= 2
        if math.log2(float(hi)) > seq_null_bits:
            hi = int(vocab_size ** seq_len)
            if dataset_entropy_bits(hi, seq_len=seq_len, vocab_size=vocab_size) < target_bits:
                raise ValueError(
                    f"target_bits={target_bits} exceeds maximum distinct-set entropy"
                )
            break

    while lo < hi:
        mid = (lo + hi) // 2
        if dataset_entropy_bits(mid, seq_len=seq_len, vocab_size=vocab_size) >= target_bits:
            hi = mid
        else:
            lo = mid + 1
    return lo


def choose_global_batch_size(
    n_sequences: int,
    max_batch_size: int,
    world_size: int = 1,
    *,
    min_steps_per_epoch: int = 16,
) -> int:
    """Choose global batch size given per-GPU max_batch_size, world_size, and min_steps_per_epoch.

    Guarantees at least ``min_steps_per_epoch`` optimizer steps per epoch whenever
    ``n_sequences >= world_size * min_steps_per_epoch`` by capping the power-of-2
    micro-batch size so tiny datasets (e.g. 0.5m/2m at r=0.25, 0.5) receive ample
    dynamic-mask optimization steps per epoch.
    """
    if n_sequences < 1:
        raise ValueError(f"n_sequences must be >= 1, got {n_sequences}")
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if n_sequences < world_size:
        raise ValueError(
            f"n_sequences ({n_sequences}) must be >= world_size ({world_size})"
        )
    if min_steps_per_epoch < 0:
        raise ValueError(f"min_steps_per_epoch must be >= 0, got {min_steps_per_epoch}")

    if world_size == 1:
        hard_cap = min(max_batch_size, n_sequences)
        if min_steps_per_epoch > 0:
            step_cap = min(hard_cap, max(1, n_sequences // int(min_steps_per_epoch)))
            pow2 = 1 << (step_cap.bit_length() - 1)
            return min(max(1, pow2), hard_cap)
        if hard_cap >= 32:
            step_cap = min(hard_cap, max(32, n_sequences // 16))
            pow2 = 1 << (step_cap.bit_length() - 1)
            return min(max(32, pow2), hard_cap)
        return 1 << (hard_cap.bit_length() - 1)

    if min_steps_per_epoch > 0:
        global_cap = min(
            world_size * max_batch_size,
            max(world_size, n_sequences // int(min_steps_per_epoch)),
            n_sequences,
        )
        micro_cap = max(1, min(max_batch_size, global_cap // world_size))
        micro_pow2 = 1 << (micro_cap.bit_length() - 1)
        micro_batch = min(max(1, micro_pow2), max_batch_size)
        return world_size * micro_batch

    if n_sequences < 32 * world_size or max_batch_size < 32:
        micro_fallback = min(world_size * max_batch_size, n_sequences) // world_size
        return world_size * max(1, micro_fallback)

    global_cap = min(
        world_size * max_batch_size,
        max(32 * world_size, n_sequences // 16),
        n_sequences,
    )
    micro_cap = global_cap // world_size
    micro_pow2 = 1 << (micro_cap.bit_length() - 1)
    micro_batch = min(max(32, micro_pow2), max_batch_size)
    return world_size * micro_batch


@dataclass(frozen=True)
class RunSpec:
    """Complete immutable specification of one parametric memory-capacity training run."""

    run_id: str
    track: str
    paradigm: str
    scale_id: str
    multiplier: float
    n_sequences: int
    max_epochs: int
    eval_epochs: tuple[int, ...]
    weight_decay: float
    precision: str
    seed: int
    data_seed: int
    save_checkpoint: bool
    ddp_preferred: bool
    estimated_tokens: int = 0
    lr_schedule: str = "wsd"
    min_lr_ratio: float = 0.05
    min_steps_per_epoch: int = 16
    adaptive_early_stop: bool = False
    mask_rate: float = MLM_15_MASK_RATE
    delete_transient_ckpt_on_complete: bool = False

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        assert_no_forbidden_infra_strings(self.run_id, label="run_id")
        if not self.track:
            raise ValueError("track must be non-empty")
        canon_paradigm = canonicalize_paradigm(self.paradigm)
        object.__setattr__(self, "paradigm", canon_paradigm)
        if self.scale_id not in SCALE_SPECS:
            raise ValueError(
                f"Unknown scale_id {self.scale_id!r}; expected {tuple(SCALE_SPECS)}"
            )
        if self.multiplier <= 0.0 or not math.isfinite(self.multiplier):
            raise ValueError(f"multiplier must be positive finite, got {self.multiplier!r}")
        if self.n_sequences < 1:
            raise ValueError(f"n_sequences must be >= 1, got {self.n_sequences}")
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {self.max_epochs}")
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"Unsupported precision {self.precision!r}; expected {PRECISIONS}"
            )
        if self.weight_decay < 0.0 or not math.isfinite(self.weight_decay):
            raise ValueError(f"Invalid weight_decay {self.weight_decay!r}")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"Unsupported lr_schedule {self.lr_schedule!r}; expected {LR_SCHEDULES}"
            )
        if not (0.0 <= float(self.min_lr_ratio) <= 1.0) or not math.isfinite(float(self.min_lr_ratio)):
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio!r}")
        if int(self.min_steps_per_epoch) < 1:
            raise ValueError(f"min_steps_per_epoch must be >= 1, got {self.min_steps_per_epoch}")
        if not (0.0 < float(self.mask_rate) <= 1.0) or not math.isfinite(float(self.mask_rate)):
            raise ValueError(f"mask_rate must be in (0, 1], got {self.mask_rate!r}")

        normalized_eval = tuple(sorted({int(e) for e in self.eval_epochs if 1 <= int(e) <= self.max_epochs}))
        object.__setattr__(self, "eval_epochs", normalized_eval)

        expected_tokens = int(self.n_sequences) * SEQ_LEN * int(self.max_epochs)
        if self.estimated_tokens <= 0:
            object.__setattr__(self, "estimated_tokens", expected_tokens)

    def compatibility_payload(self) -> dict[str, Any]:
        """Immutable training-identity fields that must match when resuming a checkpoint."""
        payload: dict[str, Any] = {
            "paradigm": self.paradigm,
            "scale_id": self.scale_id,
            "n_sequences": int(self.n_sequences),
            "weight_decay": float(self.weight_decay),
            "precision": self.precision,
            "seed": int(self.seed),
            "data_seed": int(self.data_seed),
            "lr_schedule": self.lr_schedule,
            "min_lr_ratio": float(self.min_lr_ratio),
            "min_steps_per_epoch": int(self.min_steps_per_epoch),
            "mask_rate": float(self.mask_rate),
        }
        if self.lr_schedule == "cosine":
            payload["max_epochs"] = int(self.max_epochs)
        return payload

    def spec_hash(self) -> str:
        """Deterministic SHA-256 digest of resume-compatibility fields."""
        encoded = json.dumps(self.compatibility_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "track": self.track,
            "paradigm": self.paradigm,
            "scale_id": self.scale_id,
            "multiplier": float(self.multiplier),
            "n_sequences": int(self.n_sequences),
            "max_epochs": int(self.max_epochs),
            "eval_epochs": [int(e) for e in self.eval_epochs],
            "weight_decay": float(self.weight_decay),
            "precision": self.precision,
            "seed": int(self.seed),
            "data_seed": int(self.data_seed),
            "save_checkpoint": bool(self.save_checkpoint),
            "ddp_preferred": bool(self.ddp_preferred),
            "estimated_tokens": int(self.estimated_tokens),
            "lr_schedule": self.lr_schedule,
            "min_lr_ratio": float(self.min_lr_ratio),
            "min_steps_per_epoch": int(self.min_steps_per_epoch),
            "adaptive_early_stop": bool(self.adaptive_early_stop),
            "mask_rate": float(self.mask_rate),
            "delete_transient_ckpt_on_complete": bool(self.delete_transient_ckpt_on_complete),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunSpec:
        # Legacy Batch-1 run specs omitted lr_schedule and used cosine decay.
        lr_schedule = str(data.get("lr_schedule", "cosine"))
        return cls(
            run_id=str(data["run_id"]),
            track=str(data["track"]),
            paradigm=canonicalize_paradigm(str(data["paradigm"])),
            scale_id=str(data["scale_id"]),
            multiplier=float(data["multiplier"]),
            n_sequences=int(data["n_sequences"]),
            max_epochs=int(data["max_epochs"]),
            eval_epochs=tuple(int(e) for e in data["eval_epochs"]),
            weight_decay=float(data["weight_decay"]),
            precision=str(data["precision"]),
            seed=int(data["seed"]),
            data_seed=int(data["data_seed"]),
            save_checkpoint=bool(data["save_checkpoint"]),
            ddp_preferred=bool(data["ddp_preferred"]),
            estimated_tokens=int(data.get("estimated_tokens", 0)),
            lr_schedule=lr_schedule,
            min_lr_ratio=float(data.get("min_lr_ratio", 0.05)),
            min_steps_per_epoch=int(data.get("min_steps_per_epoch", 16)),
            adaptive_early_stop=bool(data.get("adaptive_early_stop", False)),
            mask_rate=float(data.get("mask_rate", MLM_15_MASK_RATE)),
            delete_transient_ckpt_on_complete=bool(
                data.get("delete_transient_ckpt_on_complete", False)
            ),
        )


def _eval_schedule_for_horizon(max_epochs: int) -> tuple[int, ...]:
    checkpoints = tuple(e for e in EVAL_EPOCHS if e <= max_epochs)
    if not checkpoints or checkpoints[-1] != max_epochs:
        checkpoints = tuple(sorted({*checkpoints, max_epochs}))
    return checkpoints


def _make_run_spec(
    *,
    run_id: str,
    track: str,
    paradigm: str,
    scale_id: str,
    multiplier: float,
    max_epochs: int,
    weight_decay: float,
    precision: str,
    seed: int,
    data_seed: int,
    save_checkpoint: bool,
    ddp_preferred: bool,
    lr_schedule: str = "wsd",
    min_lr_ratio: float = 0.05,
    min_steps_per_epoch: int = 16,
    adaptive_early_stop: bool = False,
    mask_rate: float = MLM_15_MASK_RATE,
    delete_transient_ckpt_on_complete: bool = False,
) -> RunSpec:
    spec = SCALE_SPECS[scale_id]
    canon_paradigm = canonicalize_paradigm(paradigm)
    target_bits = multiplier * C_GRID_PRIOR_BITS_PER_PARAM * spec.exact_non_embedding_params
    n_sequences = solve_sequence_count_for_entropy(target_bits)
    eval_epochs = _eval_schedule_for_horizon(max_epochs)
    return RunSpec(
        run_id=run_id,
        track=track,
        paradigm=canon_paradigm,
        scale_id=scale_id,
        multiplier=multiplier,
        n_sequences=n_sequences,
        max_epochs=max_epochs,
        eval_epochs=eval_epochs,
        weight_decay=weight_decay,
        precision=precision,
        seed=seed,
        data_seed=data_seed,
        save_checkpoint=save_checkpoint,
        ddp_preferred=ddp_preferred,
        estimated_tokens=n_sequences * SEQ_LEN * max_epochs,
        lr_schedule=lr_schedule,
        min_lr_ratio=min_lr_ratio,
        min_steps_per_epoch=min_steps_per_epoch,
        adaptive_early_stop=adaptive_early_stop,
        mask_rate=mask_rate,
        delete_transient_ckpt_on_complete=delete_transient_ckpt_on_complete,
    )


def build_campaign_runs(
    *,
    include_ablations: bool = True,
    paradigms: Sequence[str] = PARADIGMS,
    lr_schedule: str = "wsd",
    adaptive_early_stop: bool = False,
    min_steps_per_epoch: int = 16,
    data_seed: int = 20261127,
    model_seed: int = 20261128,
) -> list[RunSpec]:
    """Build the parametric memory-capacity campaign specification across active paradigms."""
    canon_paradigms = tuple(dict.fromkeys(canonicalize_paradigm(p) for p in paradigms))
    runs: list[RunSpec] = []

    primary_35m_epochs: dict[float, int] = {
        0.25: 400,
        0.5: 400,
        1.0: 200,
        2.0: 200,
        4.0: 100,
        8.0: 50,
    }
    primary_150m_epochs: dict[float, int] = {
        0.25: 100,
        0.5: 50,
        1.0: 25,
        2.0: 25,
        4.0: 5,
        8.0: 2,
    }

    # 1. Primary scaling & capacity-multiplier grid
    for scale_id in SCALE_ORDER:
        for paradigm in canon_paradigms:
            for mult in CAPACITY_MULTIPLIERS:
                if scale_id in ("0.5m", "2m", "8m"):
                    max_epochs = 400
                    ddp_preferred = False
                    save_checkpoint = False
                elif scale_id == "35m":
                    max_epochs = primary_35m_epochs[mult]
                    ddp_preferred = True
                    save_checkpoint = max_epochs < 400
                else:
                    max_epochs = primary_150m_epochs[mult]
                    ddp_preferred = True
                    save_checkpoint = True

                run_id = (
                    f"primary_{paradigm}_{slugify_token(scale_id)}"
                    f"_r{slugify_token(f'{mult:.2f}')}"
                )
                runs.append(
                    _make_run_spec(
                        run_id=run_id,
                        track="primary",
                        paradigm=paradigm,
                        scale_id=scale_id,
                        multiplier=mult,
                        max_epochs=max_epochs,
                        weight_decay=0.0,
                        precision="bf16",
                        seed=model_seed,
                        data_seed=data_seed,
                        save_checkpoint=save_checkpoint,
                        ddp_preferred=ddp_preferred,
                        lr_schedule=lr_schedule,
                        min_steps_per_epoch=min_steps_per_epoch,
                        adaptive_early_stop=adaptive_early_stop,
                    )
                )

    if include_ablations:
        # 2a. Weight decay = 0.01 ablation
        wd_grid: tuple[tuple[str, tuple[float, ...], int], ...] = (
            ("2m", CAPACITY_MULTIPLIERS, 400),
            ("8m", (0.5, 1.0, 2.0, 4.0), 400),
        )
        for scale_id, mults, max_epochs in wd_grid:
            for paradigm in canon_paradigms:
                for mult in mults:
                    run_id = (
                        f"ablation_wd_{paradigm}_{slugify_token(scale_id)}"
                        f"_r{slugify_token(f'{mult:.2f}')}"
                    )
                    runs.append(
                        _make_run_spec(
                            run_id=run_id,
                            track="ablation_wd",
                            paradigm=paradigm,
                            scale_id=scale_id,
                            multiplier=mult,
                            max_epochs=max_epochs,
                            weight_decay=0.01,
                            precision="bf16",
                            seed=model_seed,
                            data_seed=data_seed,
                            save_checkpoint=False,
                            ddp_preferred=False,
                            lr_schedule=lr_schedule,
                            min_steps_per_epoch=min_steps_per_epoch,
                            adaptive_early_stop=adaptive_early_stop,
                        )
                    )

        # 2b. Strict FP32 (no TF32, no bf16 autocast) precision ablation
        fp32_grid: tuple[tuple[str, tuple[float, ...], int], ...] = (
            ("2m", (0.5, 1.0, 2.0, 4.0), 200),
            ("8m", (0.5, 1.0, 2.0), 100),
        )
        for scale_id, mults, max_epochs in fp32_grid:
            for paradigm in canon_paradigms:
                for mult in mults:
                    run_id = (
                        f"ablation_fp32_{paradigm}_{slugify_token(scale_id)}"
                        f"_r{slugify_token(f'{mult:.2f}')}"
                    )
                    runs.append(
                        _make_run_spec(
                            run_id=run_id,
                            track="ablation_fp32",
                            paradigm=paradigm,
                            scale_id=scale_id,
                            multiplier=mult,
                            max_epochs=max_epochs,
                            weight_decay=0.0,
                            precision="fp32_strict",
                            seed=model_seed,
                            data_seed=data_seed,
                            save_checkpoint=False,
                            ddp_preferred=False,
                            lr_schedule=lr_schedule,
                            min_steps_per_epoch=min_steps_per_epoch,
                            adaptive_early_stop=adaptive_early_stop,
                        )
                    )

        # 2c. Dedicated horizon vs intermediate-snapshot ablation
        for paradigm in canon_paradigms:
            for mult in (1.0, 2.0):
                for max_epochs in (10, 50):
                    run_id = (
                        f"ablation_lr_horizon_{paradigm}_2m"
                        f"_r{slugify_token(f'{mult:.2f}')}_e{max_epochs}"
                    )
                    runs.append(
                        _make_run_spec(
                            run_id=run_id,
                            track="ablation_lr_horizon",
                            paradigm=paradigm,
                            scale_id="2m",
                            multiplier=mult,
                            max_epochs=max_epochs,
                            weight_decay=0.0,
                            precision="bf16",
                            seed=model_seed,
                            data_seed=data_seed,
                            save_checkpoint=False,
                            ddp_preferred=False,
                            lr_schedule=lr_schedule,
                            min_steps_per_epoch=min_steps_per_epoch,
                            adaptive_early_stop=False,
                        )
                    )

    seen_ids: set[str] = set()
    for r in runs:
        if r.run_id in seen_ids:
            raise ValueError(f"Duplicate campaign run_id generated: {r.run_id!r}")
        seen_ids.add(r.run_id)
        assert_no_forbidden_infra_strings(r.to_dict(), label=r.run_id)

    return runs


def get_campaign_run_map(
    *,
    include_ablations: bool = True,
    paradigms: Sequence[str] = PARADIGMS,
    lr_schedule: str = "wsd",
    adaptive_early_stop: bool = False,
    min_steps_per_epoch: int = 16,
    data_seed: int = 20261127,
    model_seed: int = 20261128,
) -> dict[str, RunSpec]:
    """Return campaign runs indexed by run_id."""
    return {
        r.run_id: r
        for r in build_campaign_runs(
            include_ablations=include_ablations,
            paradigms=paradigms,
            lr_schedule=lr_schedule,
            adaptive_early_stop=adaptive_early_stop,
            min_steps_per_epoch=min_steps_per_epoch,
            data_seed=data_seed,
            model_seed=model_seed,
        )
    }


def build_stage1_wsd_calibration_runs(
    *,
    data_seed: int = 20261127,
    model_seed: int = 20261128,
    min_steps_per_epoch: int = 16,
) -> list[RunSpec]:
    """Build the 70-run Stage-1 WSD adaptive-stopping calibration grid in LPT priority order.

    Priority ordering:
      - Single-GPU workers claim fast 0.5m pilots (oa_uniform & mlm_15) first, then remaining
        single-GPU runs in descending LPT (estimated_tokens * n_params_non_embedding).
      - DDP workers claim 150m AR first (descending LPT), then 35m and remaining DDP runs in
        descending LPT (estimated_tokens * n_params_non_embedding) to prevent tail stragglers.
    """
    grid: list[tuple[str, str, tuple[float, ...], int, float | None]] = [
        # (paradigm, scale_id, multipliers, max_epochs, ddp_min_mult; None -> never DDP, 0.0 -> always DDP)
        ("ar", "0.5m", (0.5, 1.0, 2.0, 4.0, 8.0), 1500, None),
        ("ar", "2m", (0.5, 1.0, 2.0, 4.0, 8.0), 1500, None),
        ("ar", "8m", (0.5, 1.0, 2.0, 4.0, 8.0), 1500, None),
        ("ar", "35m", (1.0, 2.0, 4.0, 8.0), 800, 0.0),
        ("ar", "150m", (0.5, 1.0, 2.0, 4.0), 500, 0.0),
        ("mlm_15", "0.5m", (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0), 4000, None),
        ("mlm_15", "2m", (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0), 4000, None),
        ("mlm_15", "8m", (0.25, 0.5, 1.0, 2.0, 4.0, 8.0), 3000, 2.0),
        ("mlm_15", "35m", (0.25, 0.5, 1.0, 2.0, 4.0), 1500, 0.0),
        ("oa_uniform", "0.5m", (0.25, 0.5, 1.0, 2.0, 4.0, 8.0), 10000, None),
        ("oa_uniform", "2m", (0.25, 0.5, 1.0, 2.0, 4.0, 8.0), 8000, 4.0),
        ("oa_uniform", "8m", (0.25, 0.5, 1.0, 2.0, 4.0), 5000, 1.0),
        ("oa_uniform", "35m", (0.25, 0.5, 1.0, 2.0, 4.0), 3000, 0.0),
    ]

    runs: list[RunSpec] = []
    for paradigm, scale_id, mults, max_epochs, ddp_min_mult in grid:
        for mult in mults:
            ddp_preferred = (ddp_min_mult is not None) and (mult >= ddp_min_mult - 1e-12)
            run_id = (
                f"wsd_{paradigm}_{slugify_token(scale_id)}"
                f"_r{slugify_token(f'{mult:.2f}')}"
            )
            runs.append(
                _make_run_spec(
                    run_id=run_id,
                    track="stage1_wsd",
                    paradigm=paradigm,
                    scale_id=scale_id,
                    multiplier=mult,
                    max_epochs=max_epochs,
                    weight_decay=0.0,
                    precision="bf16",
                    seed=model_seed,
                    data_seed=data_seed,
                    save_checkpoint=False,
                    ddp_preferred=ddp_preferred,
                    lr_schedule="wsd",
                    min_lr_ratio=0.05,
                    min_steps_per_epoch=min_steps_per_epoch,
                    adaptive_early_stop=True,
                    delete_transient_ckpt_on_complete=True,
                )
            )

    def _compute_cost(r: RunSpec) -> int:
        return int(r.estimated_tokens) * int(SCALE_SPECS[r.scale_id].exact_non_embedding_params)

    def _priority_key(r: RunSpec) -> tuple[int, int, str]:
        # Tier 0: Fast 0.5m pilots (oa_uniform & mlm_15) -> immediate pilot curves on 1-GPU slots
        if (not r.ddp_preferred) and r.scale_id == "0.5m" and r.paradigm in ("oa_uniform", "mlm_15"):
            return (0, -_compute_cost(r), r.run_id)
        # Tier 1: 150m AR DDP priority -> start heaviest flagship AR rung immediately on DDP nodes
        if r.ddp_preferred and r.scale_id == "150m" and r.paradigm == "ar":
            return (1, -_compute_cost(r), r.run_id)
        # Tier 2: Remaining DDP runs (35m, 8m/2m high-r) in strict descending LPT order
        if r.ddp_preferred:
            return (2, -_compute_cost(r), r.run_id)
        # Tier 3: Remaining single-GPU runs in strict descending LPT order
        return (3, -_compute_cost(r), r.run_id)

    runs.sort(key=_priority_key)

    seen_ids: set[str] = set()
    for r in runs:
        if r.run_id in seen_ids:
            raise ValueError(f"Duplicate Stage-1 WSD run_id generated: {r.run_id!r}")
        seen_ids.add(r.run_id)
        assert_no_forbidden_infra_strings(r.to_dict(), label=r.run_id)

    return runs
