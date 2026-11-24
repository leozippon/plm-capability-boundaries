"""Canonical 65-arm cross-paradigm roster and precision-aware stage resolvers.

This module provides a single authoritative metadata registry and deterministic
artifact resolver spanning:

* The 33 pre-registered unconditional causal/joint/text arms
  (``src.capability.interactions.pairwise_epistasis.ROSTER``), kept strictly
  unmodified.
* The 32 cross-paradigm general-PLM arms evaluated in ``general_plm_20261120``
  (29 single-sequence arms + 3 multi-sequence MSA context arms).

Every stage resolver follows the exact ``torch.float32``-preferred
deduplication rule of ``scripts/capability/general_plm/fit_general_plm_summary.py``:
``float32/float64`` (rank 2) beats ``bfloat16`` (rank 1), breaking ties by the
newest run directory timestamp.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from src.capability.interactions.pairwise_epistasis import ROSTER as _PREREGISTERED_ROSTER

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Default output root for unified single-yardstick artifacts (created by runners, never on import).
UNIFIED_OUTPUT_DIR = Path("results/extensions/unified_yardstick_20261124")

#: Canonical display ordering of model paradigms for tables and figures.
PARADIGM_ORDER: tuple[str, ...] = (
    "causal_text",
    "causal_joint",
    "causal_adaptation",
    "causal_protein",
    "masked_lm",
    "structure_aware_mlm",
    "discrete_diffusion",
    "multiseq_msa",
)

_PARADIGM_RANK: Mapping[str, int] = MappingProxyType(
    {paradigm: idx for idx, paradigm in enumerate(PARADIGM_ORDER)}
)

VALID_MODALITIES = frozenset({"text", "joint", "protein", "protein_multimodal"})
VALID_VOCAB_TYPES = frozenset(
    {"bpe", "byte", "aa20", "aa20_bpe", "aa20_3di", "aa20_vqvae"}
)
VALID_ORIGIN_COHORTS = frozenset({"preregistered_33", "expanded_32"})


@dataclass(frozen=True)
class ArmMetadata:
    """Authoritative metadata for one arm in the unified 65-arm benchmark universe."""

    arm_id: str
    display_name: str
    family: str
    paradigm: str
    modality: str
    architecture: str
    params_m: float
    vocab_type: str
    is_single_sequence: bool
    has_r1_raw_readout_1024: bool
    origin_cohort: str

    def __post_init__(self) -> None:
        if self.paradigm not in _PARADIGM_RANK:
            raise ValueError(f"{self.arm_id}: unknown paradigm {self.paradigm!r}")
        if self.modality not in VALID_MODALITIES:
            raise ValueError(f"{self.arm_id}: unknown modality {self.modality!r}")
        if self.vocab_type not in VALID_VOCAB_TYPES:
            raise ValueError(f"{self.arm_id}: unknown vocab_type {self.vocab_type!r}")
        if self.origin_cohort not in VALID_ORIGIN_COHORTS:
            raise ValueError(f"{self.arm_id}: unknown origin_cohort {self.origin_cohort!r}")
        if not (self.params_m > 0.0):
            raise ValueError(f"{self.arm_id}: params_m must be positive, got {self.params_m!r}")


def _arm(
    arm_id: str,
    display_name: str,
    family: str,
    paradigm: str,
    modality: str,
    architecture: str,
    params_m: float,
    vocab_type: str,
    *,
    is_single_sequence: bool = True,
    has_r1_raw_readout_1024: bool = False,
    origin_cohort: str,
) -> ArmMetadata:
    return ArmMetadata(
        arm_id=arm_id,
        display_name=display_name,
        family=family,
        paradigm=paradigm,
        modality=modality,
        architecture=architecture,
        params_m=float(params_m),
        vocab_type=vocab_type,
        is_single_sequence=is_single_sequence,
        has_r1_raw_readout_1024=has_r1_raw_readout_1024,
        origin_cohort=origin_cohort,
    )


_ALL_METADATA_LIST: tuple[ArmMetadata, ...] = (
    # --------------------------------------------------------- causal_text (15)
    _arm("bygpt5-small-en", "ByGPT5-small", "ByGPT5", "causal_text", "text", "t5_decoder", 54.0, "byte", origin_cohort="preregistered_33"),
    _arm("bygpt5-base-en", "ByGPT5-base", "ByGPT5", "causal_text", "text", "t5_decoder", 117.0, "byte", origin_cohort="preregistered_33"),
    _arm("bygpt5-medium-en", "ByGPT5-medium", "ByGPT5", "causal_text", "text", "t5_decoder", 292.0, "byte", origin_cohort="preregistered_33"),
    _arm("dialogpt-small", "DialoGPT-small", "DialoGPT", "causal_text", "text", "gpt2", 124.0, "bpe", origin_cohort="preregistered_33"),
    _arm("gpt2", "GPT-2-small", "GPT2", "causal_text", "text", "gpt2", 124.0, "bpe", origin_cohort="preregistered_33"),
    _arm("gpt2-medium", "GPT-2-medium", "GPT2", "causal_text", "text", "gpt2", 355.0, "bpe", origin_cohort="preregistered_33"),
    _arm("gpt2-large", "GPT-2-large", "GPT2", "causal_text", "text", "gpt2", 774.0, "bpe", origin_cohort="preregistered_33"),
    _arm("gpt2-xl", "GPT-2-XL", "GPT2", "causal_text", "text", "gpt2", 1558.0, "bpe", origin_cohort="preregistered_33"),
    _arm("llama-3.2-3b", "Llama-3.2-3B", "Llama3", "causal_text", "text", "llama", 3210.0, "bpe", origin_cohort="preregistered_33"),
    _arm("llama-2-7b", "Llama-2-7B", "Llama2", "causal_text", "text", "llama", 6740.0, "bpe", origin_cohort="preregistered_33"),
    _arm("qwen2.5-0.5b", "Qwen2.5-0.5B", "Qwen2.5", "causal_text", "text", "qwen2", 494.0, "bpe", origin_cohort="preregistered_33"),
    _arm("qwen2.5-0.5b-instruct", "Qwen2.5-0.5B-Instruct", "Qwen2.5", "causal_text", "text", "qwen2", 494.0, "bpe", origin_cohort="preregistered_33"),
    _arm("qwen2.5-7b", "Qwen2.5-7B", "Qwen2.5", "causal_text", "text", "qwen2", 7620.0, "bpe", origin_cohort="preregistered_33"),
    _arm("qwen2.5-32b", "Qwen2.5-32B", "Qwen2.5", "causal_text", "text", "qwen2", 32800.0, "bpe", origin_cohort="preregistered_33"),
    _arm("qwen3-8b-base", "Qwen3-8B-Base", "Qwen3", "causal_text", "text", "qwen3", 8190.0, "bpe", origin_cohort="preregistered_33"),
    # --------------------------------------------------------- causal_joint (5)
    _arm("galactica-125m", "Galactica-125M", "Galactica", "causal_joint", "joint", "opt", 125.0, "bpe", origin_cohort="preregistered_33"),
    _arm("galactica-1.3b", "Galactica-1.3B", "Galactica", "causal_joint", "joint", "opt", 1300.0, "bpe", origin_cohort="preregistered_33"),
    _arm("galactica-6.7b", "Galactica-6.7B", "Galactica", "causal_joint", "joint", "opt", 6700.0, "bpe", origin_cohort="preregistered_33"),
    _arm("galactica-30b", "Galactica-30B", "Galactica", "causal_joint", "joint", "opt", 30000.0, "bpe", origin_cohort="preregistered_33"),
    _arm("instructprotein", "InstructProtein", "InstructProtein", "causal_joint", "joint", "opt", 1300.0, "bpe", origin_cohort="preregistered_33"),
    # ---------------------------------------------------- causal_adaptation (2)
    _arm("prollama-stage-1", "ProLLaMA-Stage-1", "ProLLaMA", "causal_adaptation", "joint", "llama", 7000.0, "aa20", origin_cohort="preregistered_33"),
    _arm("prollama", "ProLLaMA", "ProLLaMA", "causal_adaptation", "joint", "llama", 7000.0, "aa20", origin_cohort="preregistered_33"),
    # ------------------------------------------------------ causal_protein (18)
    _arm("progen2-small", "ProGen2-small", "ProGen2", "causal_protein", "protein", "progen", 151.0, "aa20", origin_cohort="preregistered_33"),
    _arm("progen2-base", "ProGen2-base", "ProGen2", "causal_protein", "protein", "progen", 764.0, "aa20", origin_cohort="preregistered_33"),
    _arm("progen2-medium", "ProGen2-medium", "ProGen2", "causal_protein", "protein", "progen", 764.0, "aa20", origin_cohort="preregistered_33"),
    _arm("progen2-large", "ProGen2-large", "ProGen2", "causal_protein", "protein", "progen", 2700.0, "aa20", origin_cohort="preregistered_33"),
    _arm("progen2-xlarge", "ProGen2-xlarge", "ProGen2", "causal_protein", "protein", "progen", 6400.0, "aa20", origin_cohort="preregistered_33"),
    _arm("progen3-112m", "ProGen3-112M", "ProGen3", "causal_protein", "protein", "progen3_moe", 112.0, "aa20", origin_cohort="preregistered_33"),
    _arm("progen3-219m", "ProGen3-219M", "ProGen3", "causal_protein", "protein", "progen3_moe", 219.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("progen3-339m", "ProGen3-339M", "ProGen3", "causal_protein", "protein", "progen3_moe", 339.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("progen3-762m", "ProGen3-762M", "ProGen3", "causal_protein", "protein", "progen3_moe", 762.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("progen3-1b", "ProGen3-1B", "ProGen3", "causal_protein", "protein", "progen3_moe", 1000.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("progen3-3b", "ProGen3-3B", "ProGen3", "causal_protein", "protein", "progen3_moe", 3000.0, "aa20", origin_cohort="preregistered_33"),
    _arm("protgpt2", "ProtGPT2", "ProtGPT2", "causal_protein", "protein", "gpt2", 738.0, "aa20_bpe", origin_cohort="preregistered_33"),
    _arm("protgpt3-112m", "ProtGPT3-112M", "ProtGPT3", "causal_protein", "protein", "mixtral_moe", 112.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("protgpt3-1.3b", "ProtGPT3-1.3B", "ProtGPT3", "causal_protein", "protein", "mixtral_moe", 1300.0, "aa20", origin_cohort="preregistered_33"),
    _arm("protgpt3-10b", "ProtGPT3-10B", "ProtGPT3", "causal_protein", "protein", "mixtral_moe", 10000.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("dayhoff-170m", "Dayhoff-170M", "Dayhoff", "causal_protein", "protein", "jamba_hybrid", 170.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("rita-xl", "RITA-XL", "RITA", "causal_protein", "protein", "rita", 1200.0, "aa20", origin_cohort="preregistered_33"),
    _arm("proteinglm-7b-clm", "ProteinGLM-7B-CLM", "ProteinGLM", "causal_protein", "protein", "glm", 7000.0, "aa20", origin_cohort="preregistered_33"),
    # ----------------------------------------------------------- masked_lm (15)
    _arm("esm2-8m", "ESM2-8M", "ESM2", "masked_lm", "protein", "esm2", 7.8, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("esm2-35m", "ESM2-35M", "ESM2", "masked_lm", "protein", "esm2", 34.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("esm2-150m", "ESM2-150M", "ESM2", "masked_lm", "protein", "esm2", 148.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("esm2-650m", "ESM2-650M", "ESM2", "masked_lm", "protein", "esm2", 651.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("esm2-3b", "ESM2-3B", "ESM2", "masked_lm", "protein", "esm2", 2840.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("esm2-15b", "ESM2-15B", "ESM2", "masked_lm", "protein", "esm2", 14900.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("esmc-300m", "ESM C-300M", "ESMC", "masked_lm", "protein", "esmc", 300.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("esmc-600m", "ESM C-600M", "ESMC", "masked_lm", "protein", "esmc", 600.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("esmc-6b", "ESM C-6B", "ESMC", "masked_lm", "protein", "esmc", 6000.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("amplify-120m", "AMPLIFY-120M", "AMPLIFY", "masked_lm", "protein", "amplify", 120.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("amplify-350m", "AMPLIFY-350M", "AMPLIFY", "masked_lm", "protein", "amplify", 350.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("proteinglm-1b-mlm", "ProteinGLM-1B-MLM", "ProteinGLM-MLM", "masked_lm", "protein", "proteinglm_mlm", 1000.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("proteinglm-3b-mlm", "ProteinGLM-3B-MLM", "ProteinGLM-MLM", "masked_lm", "protein", "proteinglm_mlm", 3000.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("proteinglm-10b-mlm", "ProteinGLM-10B-MLM", "ProteinGLM-MLM", "masked_lm", "protein", "proteinglm_mlm", 10000.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("prime-690m", "Prime-690M", "Prime", "masked_lm", "protein", "prime_mlm", 690.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    # -------------------------------------------------- structure_aware_mlm (3)
    _arm("saprot-35m", "SaProt-35M", "SaProt", "structure_aware_mlm", "protein_multimodal", "saprot", 35.0, "aa20_3di", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("saprot-650m", "SaProt-650M", "SaProt", "structure_aware_mlm", "protein_multimodal", "saprot", 650.0, "aa20_3di", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("saprot-1.3b", "SaProt-1.3B", "SaProt", "structure_aware_mlm", "protein_multimodal", "saprot", 1300.0, "aa20_3di", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    # --------------------------------------------------- discrete_diffusion (4)
    _arm("dplm-150m", "DPLM-150M", "DPLM", "discrete_diffusion", "protein", "esm2", 150.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("dplm-650m", "DPLM-650M", "DPLM", "discrete_diffusion", "protein", "esm2", 650.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("dplm-3b", "DPLM-3B", "DPLM", "discrete_diffusion", "protein", "esm2", 3000.0, "aa20", has_r1_raw_readout_1024=True, origin_cohort="expanded_32"),
    _arm("esm3-1.4b", "ESM3-1.4B", "ESM3", "discrete_diffusion", "protein_multimodal", "esm3_multimodal", 1400.0, "aa20_vqvae", has_r1_raw_readout_1024=False, origin_cohort="expanded_32"),
    # --------------------------------------------------------- multiseq_msa (3)
    _arm("msa-transformer-100m", "MSA-Transformer-100M", "MSA-Transformer", "multiseq_msa", "protein", "msa_transformer", 100.0, "aa20", is_single_sequence=False, has_r1_raw_readout_1024=False, origin_cohort="expanded_32"),
    _arm("protgpt3-msa", "ProtGPT3-MSA", "ProtGPT3-MSA", "multiseq_msa", "protein", "mixtral_moe", 1300.0, "aa20", is_single_sequence=False, has_r1_raw_readout_1024=False, origin_cohort="expanded_32"),
    _arm("dayhoff-3b-msa", "Dayhoff-3B-MSA", "Dayhoff-MSA", "multiseq_msa", "protein", "jamba_hybrid", 3000.0, "aa20", is_single_sequence=False, has_r1_raw_readout_1024=False, origin_cohort="expanded_32"),
)

#: Immutable arm-id -> ArmMetadata mapping over all 65 arms.
ARM_METADATA: Mapping[str, ArmMetadata] = MappingProxyType(
    {meta.arm_id: meta for meta in _ALL_METADATA_LIST}
)

if len(ARM_METADATA) != len(_ALL_METADATA_LIST):
    raise AssertionError("Duplicate arm_id in unified roster declaration")


def _paradigm_scale_key(arm_id: str) -> tuple[int, str, float, str]:
    meta = ARM_METADATA[arm_id]
    return (_PARADIGM_RANK[meta.paradigm], meta.family.lower(), meta.params_m, meta.arm_id)


#: Provenance cohorts (metadata only; disjoint by construction).
PREREGISTERED_33_ARMS: tuple[str, ...] = tuple(
    sorted(meta.arm_id for meta in _ALL_METADATA_LIST if meta.origin_cohort == "preregistered_33")
)
EXPANDED_32_ARMS: tuple[str, ...] = tuple(
    sorted(meta.arm_id for meta in _ALL_METADATA_LIST if meta.origin_cohort == "expanded_32")
)

#: All 65 unique arms ordered by (paradigm, family, parameter scale).
UNIFIED_65_ARMS: tuple[str, ...] = tuple(
    sorted(ARM_METADATA.keys(), key=_paradigm_scale_key)
)
#: All 62 unique single-sequence arms (33 pre-registered + 29 expanded).
UNIFIED_62_SINGLE_SEQUENCE_ARMS: tuple[str, ...] = tuple(
    arm_id for arm_id in UNIFIED_65_ARMS if ARM_METADATA[arm_id].is_single_sequence
)
#: The 3 multi-sequence MSA context arms (R4-only).
UNIFIED_3_MSA_CONTEXT_ARMS: tuple[str, ...] = (
    "msa-transformer-100m",
    "protgpt3-msa",
    "dayhoff-3b-msa",
)

#: Alphabetical ordering helpers for deterministic serialization & diffing.
UNIFIED_65_ARMS_ALPHA: tuple[str, ...] = tuple(sorted(UNIFIED_65_ARMS))
UNIFIED_62_SINGLE_SEQUENCE_ARMS_ALPHA: tuple[str, ...] = tuple(
    sorted(UNIFIED_62_SINGLE_SEQUENCE_ARMS)
)

# Structural invariants verified once at import time.
assert tuple(sorted(_PREREGISTERED_ROSTER)) == PREREGISTERED_33_ARMS, (
    "PREREGISTERED_33_ARMS drifted from pairwise_epistasis.ROSTER"
)
assert len(PREREGISTERED_33_ARMS) == 33
assert len(EXPANDED_32_ARMS) == 32
assert not (set(PREREGISTERED_33_ARMS) & set(EXPANDED_32_ARMS)), (
    "Pre-registered 33 and expanded 32 rosters must be strictly disjoint"
)
assert len(UNIFIED_65_ARMS) == 65
assert len(UNIFIED_62_SINGLE_SEQUENCE_ARMS) == 62
assert set(UNIFIED_62_SINGLE_SEQUENCE_ARMS) | set(UNIFIED_3_MSA_CONTEXT_ARMS) == set(
    UNIFIED_65_ARMS
)
assert not (set(UNIFIED_62_SINGLE_SEQUENCE_ARMS) & set(UNIFIED_3_MSA_CONTEXT_ARMS))


def get_arm_metadata(arm_id: str) -> ArmMetadata:
    """Return authoritative ``ArmMetadata`` for ``arm_id`` or raise KeyError."""
    if arm_id not in ARM_METADATA:
        raise KeyError(
            f"Unknown arm_id {arm_id!r}; expected one of the 65 unified arms."
        )
    return ARM_METADATA[arm_id]


def arms_by_paradigm(paradigm: str) -> tuple[str, ...]:
    """Return arms belonging to ``paradigm`` in canonical paradigm-scale order."""
    if paradigm not in _PARADIGM_RANK:
        raise ValueError(
            f"Unknown paradigm {paradigm!r}; expected one of {list(PARADIGM_ORDER)}"
        )
    return tuple(a for a in UNIFIED_65_ARMS if ARM_METADATA[a].paradigm == paradigm)


def sort_arms(arm_ids: Iterable[str], *, by: str = "paradigm_scale") -> tuple[str, ...]:
    """Sort an iterable of unified arm ids by ``'paradigm_scale'`` or ``'alphabetical'``."""
    ids = list(arm_ids)
    for arm_id in ids:
        get_arm_metadata(arm_id)
    if by == "paradigm_scale":
        return tuple(sorted(ids, key=_paradigm_scale_key))
    if by == "alphabetical":
        return tuple(sorted(ids))
    raise ValueError(f"Unsupported sort mode {by!r}; use 'paradigm_scale' or 'alphabetical'")


# --------------------------------------------------------------------------- #
# Deterministic precision-aware stage-file resolvers for expanded_32 arms
# --------------------------------------------------------------------------- #


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root)
    env_root = os.environ.get("TRANSFER_DATA_ROOT", "").strip()
    return Path(env_root) if env_root else REPO_ROOT


def dtype_rank(dtype_realised: Any) -> int:
    """Precision preference rank matching ``fit_general_plm_summary.py``:
    float32/float64 (2) > bfloat16/bf16 (1) > unknown/other (0).
    """
    val = str(dtype_realised or "").strip().lower()
    if "float32" in val or "float64" in val:
        return 2
    if "bfloat16" in val or "bf16" in val:
        return 1
    return 0


def default_scan_dirs(root: Path | str | None = None) -> tuple[Path, ...]:
    """Ordered external-baseline campaign directories (oldest -> newest) plus
    ``results/extensions/general_plm_20261120``.
    """
    base = _resolve_root(root)
    ext_base = base / "results/shared/external_baseline"
    dirs: list[Path] = []
    if ext_base.is_dir():
        dirs.extend(sorted(p for p in ext_base.glob("2026112*") if p.is_dir()))
    gen_dir = base / "results/extensions/general_plm_20261120"
    if gen_dir.is_dir():
        dirs.append(gen_dir)
    return tuple(dirs)


@dataclass(frozen=True)
class ResolvedArtifact:
    """One resolved stage artifact on disk with realised precision metadata."""

    arm_id: str
    stage: str
    path: Path
    dtype_realised: str | None = None
    sibling_json: Path | None = None
    shard_index: int | None = None
    num_shards: int | None = None


def _read_json_dtype(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    prov = doc.get("provenance")
    if isinstance(prov, dict) and prov.get("dtype_realised"):
        return str(prov["dtype_realised"])
    if doc.get("dtype_realised"):
        return str(doc["dtype_realised"])
    return None


def _prefer_candidate(
    current: ResolvedArtifact | None,
    candidate_dtype: str | None,
) -> bool:
    """Prefer higher dtype_rank (float32 > bfloat16); break ties by newer scan order."""
    if current is None:
        return True
    return dtype_rank(candidate_dtype) >= dtype_rank(current.dtype_realised)


def _find_esm3_seqonly_dir(scan_dirs: tuple[Path, ...]) -> Path:
    best: Path | None = None
    for scan_dir in scan_dirs:
        for meta_path in sorted(scan_dir.rglob("esm3_r123_seqonly.json")):
            best = meta_path.parent
    if best is None:
        raise FileNotFoundError(
            "esm3-1.4b: could not locate esm3_r123_seqonly.json in scan dirs "
            f"{[str(d) for d in scan_dirs]}"
        )
    return best


def _require_expanded_arm(arm_id: str, stage: str) -> ArmMetadata:
    meta = get_arm_metadata(arm_id)
    if meta.origin_cohort != "expanded_32":
        raise ValueError(
            f"{arm_id} belongs to {meta.origin_cohort!r}, not expanded_32; "
            f"GP stage resolver for {stage!r} applies to expanded_32 arms only."
        )
    return meta


def resolve_all_gp_r1_scores(
    *,
    root: Path | str | None = None,
    scan_dirs: Iterable[Path] | None = None,
) -> dict[str, ResolvedArtifact]:
    """Resolve all 29 single-sequence expanded_32 R1 score files in one scan.

    Returns 28 ``r1_ladder_scores_<arm>.npz`` files (deduplicated with
    float32-preferred precedence via sibling JSON) plus ``esm3-1.4b``'s
    ``r1_ladder_scores.jsonl``.
    """
    dirs = tuple(scan_dirs) if scan_dirs is not None else default_scan_dirs(root)
    resolved: dict[str, ResolvedArtifact] = {}
    for scan_dir in dirs:
        if not scan_dir.is_dir():
            continue
        for npz_path in sorted(scan_dir.rglob("r1_ladder_scores_*.npz")):
            arm_id = npz_path.stem.removeprefix("r1_ladder_scores_")
            if arm_id not in ARM_METADATA:
                continue
            sibling = npz_path.parent / f"r1_ladder_and_conditionals_{arm_id}.json"
            sibling_opt = sibling if sibling.is_file() else None
            cand_dtype = _read_json_dtype(sibling_opt)
            if _prefer_candidate(resolved.get(arm_id), cand_dtype):
                resolved[arm_id] = ResolvedArtifact(
                    arm_id=arm_id,
                    stage="r1_scores",
                    path=npz_path,
                    dtype_realised=cand_dtype,
                    sibling_json=sibling_opt,
                )
    try:
        esm3_dir = _find_esm3_seqonly_dir(dirs)
        esm3_scores = esm3_dir / "r1_ladder_scores.jsonl"
        if esm3_scores.is_file():
            meta_json = esm3_dir / "esm3_r123_seqonly.json"
            resolved["esm3-1.4b"] = ResolvedArtifact(
                arm_id="esm3-1.4b",
                stage="r1_scores",
                path=esm3_scores,
                dtype_realised=_read_json_dtype(meta_json) or "torch.float32",
                sibling_json=meta_json if meta_json.is_file() else None,
            )
    except FileNotFoundError:
        pass
    return resolved


def resolve_gp_r1_scores(
    arm_id: str,
    *,
    root: Path | str | None = None,
    scan_dirs: Iterable[Path] | None = None,
) -> ResolvedArtifact:
    """Resolve the canonical R1 per-variant score file for one expanded_32 arm."""
    meta = _require_expanded_arm(arm_id, "r1_scores")
    if not meta.is_single_sequence:
        raise ValueError(f"{arm_id} is a multi-sequence R4-only arm and has no R1 score file")
    found = resolve_all_gp_r1_scores(root=root, scan_dirs=scan_dirs).get(arm_id)
    if found is None:
        raise FileNotFoundError(f"{arm_id}: no R1 ladder score artifact found")
    return found


def resolve_all_gp_r1_readouts(
    *,
    root: Path | str | None = None,
    scan_dirs: Iterable[Path] | None = None,
) -> dict[str, ResolvedArtifact]:
    """Resolve all 28 cached ``r1_readout_blocks_<arm>.npz`` files in one scan."""
    dirs = tuple(scan_dirs) if scan_dirs is not None else default_scan_dirs(root)
    resolved: dict[str, ResolvedArtifact] = {}
    for scan_dir in dirs:
        if not scan_dir.is_dir():
            continue
        for npz_path in sorted(scan_dir.rglob("r1_readout_blocks_*.npz")):
            arm_id = npz_path.stem.removeprefix("r1_readout_blocks_")
            if arm_id not in ARM_METADATA:
                continue
            sibling = npz_path.parent / f"r1_ladder_and_conditionals_{arm_id}.json"
            sibling_opt = sibling if sibling.is_file() else None
            cand_dtype = _read_json_dtype(sibling_opt)
            if _prefer_candidate(resolved.get(arm_id), cand_dtype):
                resolved[arm_id] = ResolvedArtifact(
                    arm_id=arm_id,
                    stage="r1_readout",
                    path=npz_path,
                    dtype_realised=cand_dtype,
                    sibling_json=sibling_opt,
                )
    return resolved


def resolve_gp_r1_readout(
    arm_id: str,
    *,
    root: Path | str | None = None,
    scan_dirs: Iterable[Path] | None = None,
) -> ResolvedArtifact:
    """Resolve the cached 1024-d R1 readout ``.npz`` for one expanded_32 arm."""
    meta = _require_expanded_arm(arm_id, "r1_readout")
    if not meta.has_r1_raw_readout_1024:
        raise ValueError(
            f"{arm_id}: no per-variant 1024-d R1 readout archive exists for this arm"
        )
    found = resolve_all_gp_r1_readouts(root=root, scan_dirs=scan_dirs).get(arm_id)
    if found is None:
        raise FileNotFoundError(f"{arm_id}: no r1_readout_blocks_{arm_id}.npz found")
    return found


def resolve_gp_r1_json(
    arm_id: str,
    *,
    root: Path | str | None = None,
    scan_dirs: Iterable[Path] | None = None,
) -> ResolvedArtifact:
    """Resolve the canonical R1 ladder & conditionals JSON for one expanded_32 arm.

    For ``esm3-1.4b``, resolves ``r1_family_conditionals.jsonl`` with
    ``sibling_json`` pointing to ``esm3_1_4b_seqonly_eval.json`` (if present) or
    ``esm3_r123_seqonly.json``.
    """
    meta = _require_expanded_arm(arm_id, "r1_json")
    if not meta.is_single_sequence:
        raise ValueError(f"{arm_id} is an R4-only MSA arm and has no R1 JSON")
    dirs = tuple(scan_dirs) if scan_dirs is not None else default_scan_dirs(root)
    if arm_id == "esm3-1.4b":
        esm3_dir = _find_esm3_seqonly_dir(dirs)
        fc_jsonl = esm3_dir / "r1_family_conditionals.jsonl"
        if not fc_jsonl.is_file():
            raise FileNotFoundError(f"esm3-1.4b: missing {fc_jsonl}")
        eval_json = _resolve_root(root) / "results/extensions/general_plm_20261120/esm3_1_4b_seqonly_eval.json"
        meta_json = esm3_dir / "esm3_r123_seqonly.json"
        return ResolvedArtifact(
            arm_id=arm_id,
            stage="r1_json",
            path=fc_jsonl,
            dtype_realised=_read_json_dtype(meta_json) or "torch.float32",
            sibling_json=eval_json if eval_json.is_file() else (meta_json if meta_json.is_file() else None),
        )
    best: ResolvedArtifact | None = None
    filename = f"r1_ladder_and_conditionals_{arm_id}.json"
    for scan_dir in dirs:
        if not scan_dir.is_dir():
            continue
        for path in sorted(scan_dir.rglob(filename)):
            cand_dtype = _read_json_dtype(path)
            if _prefer_candidate(best, cand_dtype):
                best = ResolvedArtifact(
                    arm_id=arm_id,
                    stage="r1_json",
                    path=path,
                    dtype_realised=cand_dtype,
                    sibling_json=path,
                )
    if best is None:
        raise FileNotFoundError(f"{arm_id}: no {filename} found")
    return best


def resolve_gp_r2_arrays(
    arm_id: str,
    lane: str,
    *,
    root: Path | str | None = None,
    scan_dirs: Iterable[Path] | None = None,
) -> ResolvedArtifact:
    """Resolve raw R2 per-variant arrays for ``lane`` in ``('abundance', 'stability', 'phenotype')``."""
    meta = _require_expanded_arm(arm_id, f"r2_{lane}")
    if not meta.is_single_sequence:
        raise ValueError(f"{arm_id} is an R4-only MSA arm and has no R2 arrays")
    if lane not in ("abundance", "stability", "phenotype"):
        raise ValueError(f"Unknown R2 lane {lane!r}; expected 'abundance', 'stability', or 'phenotype'")
    dirs = tuple(scan_dirs) if scan_dirs is not None else default_scan_dirs(root)
    if arm_id == "esm3-1.4b":
        esm3_dir = _find_esm3_seqonly_dir(dirs)
        fname = {
            "abundance": "r2_abundance_scores.jsonl",
            "stability": "r2_stability_scores.jsonl",
            "phenotype": "r2_phenotypes_scores.jsonl",
        }[lane]
        target = esm3_dir / fname
        if not target.is_file():
            raise FileNotFoundError(f"esm3-1.4b: missing {target}")
        meta_json = esm3_dir / "esm3_r123_seqonly.json"
        return ResolvedArtifact(
            arm_id=arm_id,
            stage=f"r2_{lane}",
            path=target,
            dtype_realised=_read_json_dtype(meta_json) or "torch.float32",
            sibling_json=meta_json if meta_json.is_file() else None,
        )
    best: ResolvedArtifact | None = None
    filename = f"r2_{lane}_arrays_{arm_id}.npz"
    for scan_dir in dirs:
        if not scan_dir.is_dir():
            continue
        for npz_path in sorted(scan_dir.rglob(filename)):
            sibling = npz_path.parent / f"r2_phenotypes_and_readouts_{arm_id}.json"
            sibling_opt = sibling if sibling.is_file() else None
            cand_dtype = _read_json_dtype(sibling_opt)
            if _prefer_candidate(best, cand_dtype):
                best = ResolvedArtifact(
                    arm_id=arm_id,
                    stage=f"r2_{lane}",
                    path=npz_path,
                    dtype_realised=cand_dtype,
                    sibling_json=sibling_opt,
                )
    if best is None:
        raise FileNotFoundError(f"{arm_id}: no {filename} found")
    return best


def resolve_gp_r3_cycles(
    arm_id: str,
    *,
    root: Path | str | None = None,
    scan_dirs: Iterable[Path] | None = None,
) -> ResolvedArtifact:
    """Resolve raw R3 double-mutant cycle ``.npz`` (or ``esm3`` ``.jsonl``) with float32 precedence."""
    meta = _require_expanded_arm(arm_id, "r3_cycles")
    if not meta.is_single_sequence:
        raise ValueError(f"{arm_id} is an R4-only MSA arm and has no R3 cycle archive")
    dirs = tuple(scan_dirs) if scan_dirs is not None else default_scan_dirs(root)
    if arm_id == "esm3-1.4b":
        esm3_dir = _find_esm3_seqonly_dir(dirs)
        target = esm3_dir / "r3_cycle_epistasis.jsonl"
        if not target.is_file():
            raise FileNotFoundError(f"esm3-1.4b: missing {target}")
        meta_json = esm3_dir / "esm3_r123_seqonly.json"
        return ResolvedArtifact(
            arm_id=arm_id,
            stage="r3_cycles",
            path=target,
            dtype_realised=_read_json_dtype(meta_json) or "torch.float32",
            sibling_json=meta_json if meta_json.is_file() else None,
        )
    best: ResolvedArtifact | None = None
    filename = f"r3_pairwise_cycles_{arm_id}.npz"
    for scan_dir in dirs:
        if not scan_dir.is_dir():
            continue
        for npz_path in sorted(scan_dir.rglob(filename)):
            sibling = npz_path.parent / f"r3_pairwise_and_contacts_{arm_id}.json"
            sibling_opt = sibling if sibling.is_file() else None
            cand_dtype = _read_json_dtype(sibling_opt)
            if _prefer_candidate(best, cand_dtype):
                best = ResolvedArtifact(
                    arm_id=arm_id,
                    stage="r3_cycles",
                    path=npz_path,
                    dtype_realised=cand_dtype,
                    sibling_json=sibling_opt,
                )
    if best is None:
        raise FileNotFoundError(f"{arm_id}: no {filename} found")
    return best


def resolve_gp_r3_cycle_eval(
    arm_id: str,
    *,
    root: Path | str | None = None,
) -> ResolvedArtifact:
    """Resolve the cached CPU nested-CV R3 cycle evaluation JSON for one expanded_32 arm."""
    meta = _require_expanded_arm(arm_id, "r3_cycle_eval")
    if not meta.is_single_sequence:
        raise ValueError(f"{arm_id} is an R4-only MSA arm and has no R3 cycle evaluation")
    base = _resolve_root(root) / "results/extensions/general_plm_20261120"
    path = (
        base / "esm3_1_4b_seqonly_eval.json"
        if arm_id == "esm3-1.4b"
        else base / f"r3_cycle_eval_{arm_id}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"{arm_id}: missing cached R3 cycle evaluation at {path}")
    return ResolvedArtifact(
        arm_id=arm_id,
        stage="r3_cycle_eval",
        path=path,
        dtype_realised=_read_json_dtype(path),
        sibling_json=path,
    )


def resolve_gp_family_distribution_shards(
    arm_id: str,
    *,
    root: Path | str | None = None,
    scan_dirs: Iterable[Path] | None = None,
) -> tuple[ResolvedArtifact, ...]:
    """Resolve deduplicated ``family_distribution`` shard JSONs for an expanded_32 arm.

    Deduplicates shards by ``(shard_index, num_shards)`` using the shared
    float32-preferred / newest-run tie-breaker (e.g., yielding the exact 4
    unique shards ``0/4..3/4`` for ``esm2-15b``).
    """
    _require_expanded_arm(arm_id, "r1_family_distribution")
    dirs = tuple(scan_dirs) if scan_dirs is not None else default_scan_dirs(root)
    by_shard: dict[tuple[int, int], ResolvedArtifact] = {}
    prefix = f"r1_family_distribution_{arm_id}"
    for scan_dir in dirs:
        if not scan_dir.is_dir():
            continue
        for path in sorted(scan_dir.rglob(f"{prefix}*.json")):
            if not (path.stem == prefix or path.stem.startswith(f"{prefix}_")):
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(doc, dict) or doc.get("arm") != arm_id:
                continue
            if doc.get("schema") != "general_plm_r1_family_distribution_pll_v1":
                continue
            shard_idx = int(doc.get("shard_index", 0))
            num_shards = int(doc.get("num_shards", 1))
            key = (shard_idx, num_shards)
            cand_dtype = doc.get("provenance", {}).get("dtype_realised")
            if _prefer_candidate(by_shard.get(key), cand_dtype):
                by_shard[key] = ResolvedArtifact(
                    arm_id=arm_id,
                    stage="r1_family_distribution",
                    path=path,
                    dtype_realised=str(cand_dtype) if cand_dtype else None,
                    sibling_json=path,
                    shard_index=shard_idx,
                    num_shards=num_shards,
                )
    if not by_shard:
        raise FileNotFoundError(f"{arm_id}: no r1_family_distribution shards found")
    return tuple(by_shard[k] for k in sorted(by_shard))


def resolve_gp_r4_json(
    arm_id: str,
    *,
    budget_tokens: int = 1024,
    aligned_msa: bool = True,
    root: Path | str | None = None,
    scan_dirs: Iterable[Path] | None = None,
) -> ResolvedArtifact:
    """Resolve the canonical R4 context JSON for an expanded_32 arm (or ESM3 Route A)."""
    _require_expanded_arm(arm_id, "r4_context")
    dirs = tuple(scan_dirs) if scan_dirs is not None else default_scan_dirs(root)
    if arm_id == "esm3-1.4b":
        best_e3: ResolvedArtifact | None = None
        for scan_dir in dirs:
            if not scan_dir.is_dir():
                continue
            for path in sorted(scan_dir.rglob("esm3_r4_multimodal.json")):
                cand_dtype = _read_json_dtype(path)
                if _prefer_candidate(best_e3, cand_dtype):
                    best_e3 = ResolvedArtifact(
                        arm_id=arm_id,
                        stage="r4_route_a",
                        path=path,
                        dtype_realised=cand_dtype,
                        sibling_json=path,
                    )
        if best_e3 is None:
            raise FileNotFoundError("esm3-1.4b: no esm3_r4_multimodal.json found")
        return best_e3

    best: ResolvedArtifact | None = None
    filename = f"r4_multiseq_{arm_id}.json"
    for scan_dir in dirs:
        if not scan_dir.is_dir():
            continue
        for path in sorted(scan_dir.rglob(filename)):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(doc, dict) or doc.get("arm") != arm_id:
                continue
            if int(doc.get("budget_tokens", -1)) != int(budget_tokens):
                continue
            if bool(doc.get("aligned_msa", True)) != bool(aligned_msa):
                continue
            cand_dtype = doc.get("provenance", {}).get("dtype_realised")
            if _prefer_candidate(best, cand_dtype):
                best = ResolvedArtifact(
                    arm_id=arm_id,
                    stage=f"r4_b{budget_tokens}" + ("" if aligned_msa else "_unaligned"),
                    path=path,
                    dtype_realised=str(cand_dtype) if cand_dtype else None,
                    sibling_json=path,
                )
    if best is None:
        raise FileNotFoundError(
            f"{arm_id}: no R4 JSON matching budget_tokens={budget_tokens}, aligned_msa={aligned_msa}"
        )
    return best


def resolve_gp_r5_json(
    arm_id: str,
    *,
    diffusion_mode: str = "max_confidence",
    root: Path | str | None = None,
    scan_dirs: Iterable[Path] | None = None,
) -> ResolvedArtifact:
    """Resolve the canonical R5 generation JSON for an expanded_32 arm."""
    _require_expanded_arm(arm_id, "r5_generation")
    dirs = tuple(scan_dirs) if scan_dirs is not None else default_scan_dirs(root)
    if arm_id == "esm3-1.4b":
        best_e3: ResolvedArtifact | None = None
        for scan_dir in dirs:
            if not scan_dir.is_dir():
                continue
            for path in sorted(scan_dir.rglob("esm3_r5_generation.json")):
                cand_dtype = _read_json_dtype(path)
                if _prefer_candidate(best_e3, cand_dtype):
                    best_e3 = ResolvedArtifact(
                        arm_id=arm_id,
                        stage="r5_generation",
                        path=path,
                        dtype_realised=cand_dtype,
                        sibling_json=path,
                    )
        if best_e3 is None:
            raise FileNotFoundError("esm3-1.4b: no esm3_r5_generation.json found")
        return best_e3

    best: ResolvedArtifact | None = None
    filename = f"r5_infill_and_gen_{arm_id}.json"
    for scan_dir in dirs:
        if not scan_dir.is_dir():
            continue
        for path in sorted(scan_dir.rglob(filename)):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(doc, dict) or doc.get("arm") != arm_id:
                continue
            d_mode = (
                (doc.get("denovo_summary") or {}).get("diffusion_sampling_mode")
                or (doc.get("ladder_summary") or {}).get("diffusion_sampling_mode")
                or "max_confidence"
            )
            if d_mode != diffusion_mode:
                continue
            cand_dtype = doc.get("provenance", {}).get("dtype_realised")
            if _prefer_candidate(best, cand_dtype):
                best = ResolvedArtifact(
                    arm_id=arm_id,
                    stage=f"r5_{diffusion_mode}",
                    path=path,
                    dtype_realised=str(cand_dtype) if cand_dtype else None,
                    sibling_json=path,
                )
    if best is None:
        raise FileNotFoundError(
            f"{arm_id}: no R5 JSON matching diffusion_mode={diffusion_mode!r}"
        )
    return best
