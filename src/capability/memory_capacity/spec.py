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
    12000,
    16000,
    20000,
    25000,
    30000,
    40000,
)
EXPOSURE_CHECKPOINTS: tuple[int, ...] = EVAL_EPOCHS
CAPACITY_MULTIPLIERS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

PARADIGMS: tuple[str, ...] = ("ar", "mlm_15", "oa_uniform")
ALL_PARADIGMS: tuple[str, ...] = ("ar", "mlm_15", "mlm_30", "mlm_50", "oa_uniform", "mdm_cosine")
FIXED_MASK_PARADIGM_RATES: dict[str, float] = {
    "mlm_15": 0.15,
    "mlm_30": 0.30,
    "mlm_50": 0.50,
}
LEGACY_PARADIGM_ALIASES: dict[str, str] = {
    "mlm": "oa_uniform",
    "mdm": "oa_uniform",
    "mdm_uniform": "oa_uniform",
    "oa_cosine": "mdm_cosine",
}
PARADIGM_DISPLAY_NAMES: dict[str, str] = {
    "ar": "AR",
    "mlm_15": "MLM-15%",
    "mlm_30": "MLM-30%",
    "mlm_50": "MLM-50%",
    "oa_uniform": "MDM",
    "mdm_cosine": "MDM-Cosine",
}
OPTIMIZERS: tuple[str, ...] = ("adamw", "muon")


@dataclass(frozen=True)
class VocabModeSpec:
    """Vocabulary-size ablation specification preserving exact ground-truth AA bit entropy."""

    mode_id: str
    vocab_size: int
    kmer_size: int
    default_seq_len: int
    aa_equivalent_len: int

    @property
    def special_token_id(self) -> int:
        return int(self.vocab_size)

    @property
    def input_vocab_size(self) -> int:
        return int(self.vocab_size) + 1

    @property
    def bits_per_token(self) -> float:
        return math.log2(float(self.vocab_size))


VOCAB_MODES: dict[str, VocabModeSpec] = {
    "aa20": VocabModeSpec("aa20", vocab_size=20, kmer_size=1, default_seq_len=256, aa_equivalent_len=256),
    "murphy10": VocabModeSpec("murphy10", vocab_size=10, kmer_size=1, default_seq_len=256, aa_equivalent_len=256),
    "dipeptide400": VocabModeSpec("dipeptide400", vocab_size=400, kmer_size=2, default_seq_len=128, aa_equivalent_len=256),
    "tripeptide8000": VocabModeSpec("tripeptide8000", vocab_size=8000, kmer_size=3, default_seq_len=85, aa_equivalent_len=255),
}
VOCAB_MODE_ORDER: tuple[str, ...] = ("aa20", "murphy10", "dipeptide400", "tripeptide8000")

LR_SCHEDULES: tuple[str, ...] = ("wsd", "cosine")
CAMPAIGN_TRACKS: tuple[str, ...] = (
    "primary",
    "stage1_wsd",
    "ablation_wd",
    "ablation_fp32",
    "ablation_lr_horizon",
    "ablation_muon",
    "ablation_vocab",
    "ablation_mask_rate",
)
PRECISIONS: tuple[str, ...] = ("bf16", "fp32_strict")

FORBIDDEN_INFRA_RE = re.compile(
    r"(?:\b(?:[a-z0-9_-]*master-\d+|[a-z0-9_-]*worker-\d+|worker-\d+\.k8s|node[-_]?\d*|10\.\d+\.\d+\.\d+)\b)",
    re.IGNORECASE,
)


def canonicalize_paradigm(paradigm: str) -> str:
    """Canonicalize paradigm identifier, aliasing 'mdm'/'mlm' -> 'oa_uniform' without breaking RunSpec hashes."""
    raw = str(paradigm).strip().lower()
    canon = LEGACY_PARADIGM_ALIASES.get(raw, raw)
    if canon not in ALL_PARADIGMS:
        raise ValueError(
            f"Unsupported paradigm {paradigm!r}; expected {ALL_PARADIGMS} "
            f"(or aliases {tuple(LEGACY_PARADIGM_ALIASES)})"
        )
    return canon


def paradigm_display_name(paradigm: str) -> str:
    """Return canonical unified-theory display label ('AR', 'MLM-15%', 'MDM', etc.)."""
    canon = canonicalize_paradigm(paradigm)
    return PARADIGM_DISPLAY_NAMES.get(canon, canon.upper())


def is_fixed_mask_paradigm(paradigm: str) -> bool:
    """Return True for fixed-rate masked objectives ('mlm_15', 'mlm_30', 'mlm_50')."""
    return canonicalize_paradigm(paradigm) in FIXED_MASK_PARADIGM_RATES


def default_mask_rate_for_paradigm(paradigm: str) -> float:
    """Return default fixed mask rate for a paradigm (0.15 for mlm_15, 0.30 for mlm_30, 0.50 for mlm_50)."""
    canon = canonicalize_paradigm(paradigm)
    return float(FIXED_MASK_PARADIGM_RATES.get(canon, MLM_15_MASK_RATE))


def get_vocab_mode_spec(vocab_mode: str = "aa20") -> VocabModeSpec:
    """Look up a validated VocabModeSpec by name."""
    key = str(vocab_mode).strip().lower()
    if key not in VOCAB_MODES:
        raise ValueError(f"Unsupported vocab_mode {vocab_mode!r}; expected {tuple(VOCAB_MODES)}")
    return VOCAB_MODES[key]


def is_mlm_paradigm(paradigm: str) -> bool:
    """Return True for bidirectional masked objectives (MLM-15/30/50%, MDM, MDM-Cosine)."""
    return canonicalize_paradigm(paradigm) != "ar"


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
EXTENDED_SCALE_ORDER: tuple[str, ...] = ("0.5m", "2m", "8m", "35m", "150m", "650m")

SCALE_SPECS: dict[str, ScaleSpec] = {
    "0.5m": ScaleSpec("0.5m", 500_000, 3, 128, 4, 256, 2.0e-3, 512),
    "2m": ScaleSpec("2m", 2_000_000, 4, 192, 6, 608, 1.5e-3, 1024),
    "8m": ScaleSpec("8m", 8_000_000, 6, 320, 5, 960, 1.2e-3, 1024),
    "35m": ScaleSpec("35m", 35_000_000, 11, 512, 8, 1408, 8.0e-4, 512),
    "150m": ScaleSpec("150m", 150_000_000, 21, 768, 12, 2048, 6.0e-4, 512),
    # ESM-2 650M matched geometry: 33L, d=1280, 20H (head_dim=64), d_ff=3456 -> 654,303,104 non-emb params
    "650m": ScaleSpec("650m", 650_000_000, 33, 1280, 20, 3456, 3.0e-4, 256),
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


HIGH_VRAM_THRESHOLD_BYTES: int = 110 * (1024 ** 3)
MID_VRAM_THRESHOLD_BYTES: int = 56 * (1024 ** 3)
LOW_VRAM_THRESHOLD_BYTES: int = 32 * (1024 ** 3)


def should_enable_gradient_checkpointing(
    scale_id: str,
    total_vram_bytes: int | None,
    requested: bool | None = None,
) -> bool:
    """Resolve activation checkpointing; auto-enabled for 650m on <110 GiB GPUs unless overridden."""
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
    """Return a VRAM-safe per-GPU micro-step batch cap without altering 0.5m-35m on 80 GB GPUs."""
    cap = max(1, int(default_cap))
    if total_vram_bytes is None or total_vram_bytes <= 0:
        return cap
    if scale_id == "650m":
        if total_vram_bytes < LOW_VRAM_THRESHOLD_BYTES:
            return min(cap, 4)
        if total_vram_bytes < MID_VRAM_THRESHOLD_BYTES:
            return min(cap, 16)
        if total_vram_bytes < HIGH_VRAM_THRESHOLD_BYTES:
            return min(cap, 32)
        return min(cap, 64)
    if scale_id == "150m":
        if total_vram_bytes < LOW_VRAM_THRESHOLD_BYTES:
            return min(cap, 64)
        if total_vram_bytes < HIGH_VRAM_THRESHOLD_BYTES:
            return min(cap, 128)
    elif scale_id == "35m" and total_vram_bytes < LOW_VRAM_THRESHOLD_BYTES:
        return min(cap, 256)
    return cap


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
    optimizer: str = "adamw"
    muon_lr: float = 0.02
    muon_lr_scale: float | None = None
    vocab_mode: str = "aa20"
    gradient_checkpointing: bool | None = None

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

        # Auto-align default mask_rate for mlm_30 / mlm_50 if caller left the default 0.15
        eff_mask_rate = float(self.mask_rate)
        if canon_paradigm in FIXED_MASK_PARADIGM_RATES and abs(eff_mask_rate - MLM_15_MASK_RATE) < 1e-12:
            eff_mask_rate = FIXED_MASK_PARADIGM_RATES[canon_paradigm]
            object.__setattr__(self, "mask_rate", eff_mask_rate)
        if not (0.0 < eff_mask_rate <= 1.0) or not math.isfinite(eff_mask_rate):
            raise ValueError(f"mask_rate must be in (0, 1], got {self.mask_rate!r}")

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

        vm_spec = get_vocab_mode_spec(self.vocab_mode)
        object.__setattr__(self, "vocab_mode", vm_spec.mode_id)

        normalized_eval = tuple(sorted({int(e) for e in self.eval_epochs if 1 <= int(e) <= self.max_epochs}))
        object.__setattr__(self, "eval_epochs", normalized_eval)

        expected_tokens = int(self.n_sequences) * vm_spec.default_seq_len * int(self.max_epochs)
        if self.estimated_tokens <= 0:
            object.__setattr__(self, "estimated_tokens", expected_tokens)

    def compatibility_payload(self) -> dict[str, Any]:
        """Immutable training-identity fields that must match when resuming a checkpoint.

        Non-default extensions (optimizer='muon', vocab_mode!='aa20') are added only when active
        so existing AdamW/aa20 saved checkpoints retain bit-identical spec_hash() digests.
        """
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
        if self.optimizer != "adamw":
            payload["optimizer"] = self.optimizer
            payload["muon_lr"] = float(self.muon_lr)
            if self.muon_lr_scale is not None:
                payload["muon_lr_scale"] = float(self.muon_lr_scale)
        if self.vocab_mode != "aa20":
            payload["vocab_mode"] = self.vocab_mode
        return payload

    def spec_hash(self) -> str:
        """Deterministic SHA-256 digest of resume-compatibility fields."""
        encoded = json.dumps(self.compatibility_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def paradigm_display(self) -> str:
        return paradigm_display_name(self.paradigm)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
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
            "optimizer": self.optimizer,
            "muon_lr": float(self.muon_lr),
            "muon_lr_scale": None if self.muon_lr_scale is None else float(self.muon_lr_scale),
            "vocab_mode": self.vocab_mode,
            "gradient_checkpointing": self.gradient_checkpointing,
        }
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunSpec:
        # Legacy Batch-1 run specs omitted lr_schedule and used cosine decay.
        lr_schedule = str(data.get("lr_schedule", "cosine"))
        raw_gc = data.get("gradient_checkpointing", None)
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
            mask_rate=float(data.get("mask_rate", default_mask_rate_for_paradigm(str(data["paradigm"])))),
            delete_transient_ckpt_on_complete=bool(
                data.get("delete_transient_ckpt_on_complete", False)
            ),
            optimizer=str(data.get("optimizer", "adamw")),
            muon_lr=float(data.get("muon_lr", 0.02)),
            muon_lr_scale=(
                None if data.get("muon_lr_scale") is None else float(data["muon_lr_scale"])
            ),
            vocab_mode=str(data.get("vocab_mode", "aa20")),
            gradient_checkpointing=None if raw_gc is None else bool(raw_gc),
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
    """Build the 76-run Stage-1 WSD adaptive-stopping calibration grid in milestone priority order.

    Includes the 70 base WSD calibration runs across (ar, mlm_15, oa_uniform) and 6 asymptotic
    0.5m probes (mlm_15 to 25k epochs, oa_uniform to 40k epochs at r in {0.25, 1.0, 2.0}).

    Priority ordering (optimized for 24-GPU Spot concurrency without 12h+ tail blocking):
      - DDP (`ddp_preferred=True`):
        (0) 150m AR r=(0.5, 1.0, 2.0)
        (1) 35m AR r=(1.0, 2.0)
        (2) 8m mlm_15 r=(2.0, 4.0) & 8m oa_uniform r=(1.0, 2.0)
        (3) 35m mlm_15 r=(0.5, 1.0, 2.0) & 35m oa_uniform r=(0.5, 1.0, 2.0)
        (4) remaining r=(0.25, 4.0, 8.0) DDP runs (descending LPT)
      - Single-GPU (`ddp_preferred=False`):
        (0) 0.5m initial pilots (mlm_15 & oa_uniform, <=10k epochs)
        (1) 6 asymptotic 0.5m probes (wsd_asymp_*, 25k/40k epochs)
        (2) 0.5m AR + 2m {ar, mlm_15, oa_uniform} r=(0.5, 1.0, 2.0)
            + 8m {ar, mlm_15} r=(0.5, 1.0) & 8m oa_uniform r=0.5
        (3) remaining r=(0.25, 4.0, 8.0, 16.0) single-GPU runs (descending LPT)
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

    asymp_probes: tuple[tuple[str, tuple[float, ...], int], ...] = (
        ("mlm_15", (0.25, 1.0, 2.0), 25000),
        ("oa_uniform", (0.25, 1.0, 2.0), 40000),
    )
    for paradigm, mults, max_epochs in asymp_probes:
        for mult in mults:
            run_id = f"wsd_asymp_{paradigm}_0p5m_r{slugify_token(f'{mult:.2f}')}"
            runs.append(
                _make_run_spec(
                    run_id=run_id,
                    track="stage1_wsd",
                    paradigm=paradigm,
                    scale_id="0.5m",
                    multiplier=mult,
                    max_epochs=max_epochs,
                    weight_decay=0.0,
                    precision="bf16",
                    seed=model_seed,
                    data_seed=data_seed,
                    save_checkpoint=False,
                    ddp_preferred=False,
                    lr_schedule="wsd",
                    min_lr_ratio=0.05,
                    min_steps_per_epoch=min_steps_per_epoch,
                    adaptive_early_stop=True,
                    delete_transient_ckpt_on_complete=True,
                )
            )

    def _compute_cost(r: RunSpec) -> int:
        return int(r.estimated_tokens) * int(SCALE_SPECS[r.scale_id].exact_non_embedding_params)

    def _in_mults(r: RunSpec, mults: tuple[float, ...]) -> bool:
        return any(abs(r.multiplier - m) < 1e-9 for m in mults)

    def _priority_key(r: RunSpec) -> tuple[int, int, int, str]:
        cost = _compute_cost(r)
        if r.ddp_preferred:
            if r.scale_id == "150m" and r.paradigm == "ar" and _in_mults(r, (0.5, 1.0, 2.0)):
                return (1, 0, cost, r.run_id)
            if r.scale_id == "35m" and r.paradigm == "ar" and _in_mults(r, (1.0, 2.0)):
                return (1, 1, cost, r.run_id)
            if (
                (r.scale_id == "8m" and r.paradigm == "mlm_15" and _in_mults(r, (2.0, 4.0)))
                or (r.scale_id == "8m" and r.paradigm == "oa_uniform" and _in_mults(r, (1.0, 2.0)))
            ):
                return (1, 2, cost, r.run_id)
            if (
                r.scale_id == "35m"
                and r.paradigm in ("mlm_15", "oa_uniform")
                and _in_mults(r, (0.5, 1.0, 2.0))
            ):
                return (1, 3, cost, r.run_id)
            return (1, 4, -cost, r.run_id)

        # Single-GPU milestone priority
        is_asymp = r.run_id.startswith("wsd_asymp_")
        if (not is_asymp) and r.scale_id == "0.5m" and r.paradigm in ("oa_uniform", "mlm_15"):
            return (0, 0, -cost, r.run_id)
        if is_asymp:
            return (0, 1, cost, r.run_id)
        if (
            (r.scale_id == "0.5m" and r.paradigm == "ar")
            or (r.scale_id == "2m" and _in_mults(r, (0.5, 1.0, 2.0)))
            or (r.scale_id == "8m" and r.paradigm in ("ar", "mlm_15") and _in_mults(r, (0.5, 1.0)))
            or (r.scale_id == "8m" and r.paradigm == "oa_uniform" and _in_mults(r, (0.5,)))
        ):
            return (0, 2, cost, r.run_id)
        return (0, 3, -cost, r.run_id)

    runs.sort(key=_priority_key)

    seen_ids: set[str] = set()
    for r in runs:
        if r.run_id in seen_ids:
            raise ValueError(f"Duplicate Stage-1 WSD run_id generated: {r.run_id!r}")
        seen_ids.add(r.run_id)
        assert_no_forbidden_infra_strings(r.to_dict(), label=r.run_id)

    return runs
