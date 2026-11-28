"""Parametric memory-capacity specifications, scaling ladder, and campaign builder."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SEQ_LEN: int = 256
VOCAB_AA: int = 20
SPECIAL_TOKEN_ID: int = 20  # <BOS> for AR, <MASK> for MLM
INPUT_VOCAB_SIZE: int = 21
LOG2_20: float = math.log2(20.0)
LN2: float = math.log(2.0)
C_GRID_PRIOR_BITS_PER_PARAM: float = 3.5
EXPOSURE_CHECKPOINTS: tuple[int, ...] = (1, 2, 5, 10, 25, 50, 100, 200, 400)
CAPACITY_MULTIPLIERS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

PARADIGMS: tuple[str, ...] = ("ar", "mlm")
CAMPAIGN_TRACKS: tuple[str, ...] = (
    "primary",
    "ablation_wd",
    "ablation_fp32",
    "ablation_lr_horizon",
)
PRECISIONS: tuple[str, ...] = ("bf16", "fp32_strict")

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
) -> int:
    """Choose global batch size given per-GPU max_batch_size and DDP world_size.

    - world_size == 1: largest power-of-2 in [32, max_batch_size] with
      batch_size <= max(32, n_sequences // 16), clamped to n_sequences.
    - world_size > 1: per-rank micro_batch is largest power-of-2 in [32, max_batch_size]
      such that world_size * micro_batch <= min(world_size * max_batch_size,
      max(32 * world_size, n_sequences // 16)), falling back cleanly when
      n_sequences < 32 * world_size.
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

    if world_size == 1:
        hard_cap = min(max_batch_size, n_sequences)
        if hard_cap >= 32:
            step_cap = min(hard_cap, max(32, n_sequences // 16))
            pow2 = 1 << (step_cap.bit_length() - 1)
            return min(max(32, pow2), hard_cap)
        return 1 << (hard_cap.bit_length() - 1)

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

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        assert_no_forbidden_infra_strings(self.run_id, label="run_id")
        if not self.track:
            raise ValueError("track must be non-empty")
        if self.paradigm not in PARADIGMS:
            raise ValueError(f"Unsupported paradigm {self.paradigm!r}; expected {PARADIGMS}")
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

        normalized_eval = tuple(sorted({int(e) for e in self.eval_epochs if 1 <= int(e) <= self.max_epochs}))
        object.__setattr__(self, "eval_epochs", normalized_eval)

        expected_tokens = int(self.n_sequences) * SEQ_LEN * int(self.max_epochs)
        if self.estimated_tokens <= 0:
            object.__setattr__(self, "estimated_tokens", expected_tokens)

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
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunSpec:
        return cls(
            run_id=str(data["run_id"]),
            track=str(data["track"]),
            paradigm=str(data["paradigm"]),
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
        )


def _eval_schedule_for_horizon(max_epochs: int) -> tuple[int, ...]:
    checkpoints = tuple(e for e in EXPOSURE_CHECKPOINTS if e <= max_epochs)
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
) -> RunSpec:
    spec = SCALE_SPECS[scale_id]
    target_bits = multiplier * C_GRID_PRIOR_BITS_PER_PARAM * spec.exact_non_embedding_params
    n_sequences = solve_sequence_count_for_entropy(target_bits)
    eval_epochs = _eval_schedule_for_horizon(max_epochs)
    return RunSpec(
        run_id=run_id,
        track=track,
        paradigm=paradigm,
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
    )


def build_campaign_runs(
    *,
    include_ablations: bool = True,
    data_seed: int = 20261127,
    model_seed: int = 20261128,
) -> list[RunSpec]:
    """Build the full Option A parametric memory-capacity campaign specification."""
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
        for paradigm in PARADIGMS:
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
                    )
                )

    if include_ablations:
        # 2a. Weight decay = 0.01 ablation
        wd_grid: tuple[tuple[str, tuple[float, ...], int], ...] = (
            ("2m", CAPACITY_MULTIPLIERS, 400),
            ("8m", (0.5, 1.0, 2.0, 4.0), 400),
        )
        for scale_id, mults, max_epochs in wd_grid:
            for paradigm in PARADIGMS:
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
                        )
                    )

        # 2b. Strict FP32 (no TF32, no bf16 autocast) precision ablation
        fp32_grid: tuple[tuple[str, tuple[float, ...], int], ...] = (
            ("2m", (0.5, 1.0, 2.0, 4.0), 200),
            ("8m", (0.5, 1.0, 2.0), 100),
        )
        for scale_id, mults, max_epochs in fp32_grid:
            for paradigm in PARADIGMS:
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
                        )
                    )

        # 2c. Dedicated cosine-horizon vs intermediate-snapshot ablation
        for paradigm in PARADIGMS:
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
    data_seed: int = 20261127,
    model_seed: int = 20261128,
) -> dict[str, RunSpec]:
    """Return campaign runs indexed by run_id."""
    return {
        r.run_id: r
        for r in build_campaign_runs(
            include_ablations=include_ablations,
            data_seed=data_seed,
            model_seed=model_seed,
        )
    }
