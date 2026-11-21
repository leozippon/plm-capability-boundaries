"""Unified registry and fail-fast checkpoint resolver for general PLMs.

Every model family in the 2026-11-20 general-PLM extension is declared here
once: ESM2 (8M-15B), DPLM (150M-3B), ESM C HF (300M-6B), AMPLIFY (120M-350M),
SaProt (35M-1.3B), ProteinGLM-MLM (1B-10B), Prime-690M, multi-sequence R4
models (MSA Transformer, ProtGPT3-MSA, Dayhoff-3B-MSA), and newer causal PLMs
(ProtGPT3-112M/1.3B/10B, ProGen3-219M/339M/762M/1B, Dayhoff-170M/3B).

This registry is strictly separate from the frozen 33-arm causal ``ROSTER``
and ``PANEL``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


#: Ordered fallback search roots when no explicit ``--model-root`` overrides.
#: Covers both local Compute staging (``/home/lzp/models/general_plm_20261120``,
#: ``/Data/lzp/models/general_plm_20261120``, ``/Data/public/models_R2``,
#: ``/Data/public``) and H200 GPFS (``$MODEL_ROOT`` = ``/gpfs/.../models``).
DEFAULT_SEARCH_ROOTS: tuple[Path, ...] = (
    Path("/home/lzp/models/general_plm_20261120"),
    Path("/Data/lzp/models/general_plm_20261120"),
    Path("/Data/public/models_R2"),
    Path("/Data/public"),
)

_WEIGHT_SUFFIXES: tuple[str, ...] = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
)


@dataclass(frozen=True)
class GeneralPLMSpec:
    """Static declaration of one exploratory general-PLM checkpoint."""

    arm: str
    family: str
    paradigm: str
    loader_kind: str
    candidate_dirs: tuple[str, ...]
    hf_repo: str
    default_dtype: str = "float32"
    max_residues: int = 1024
    approx_params: int = 0
    supports_r1_r3: bool = True
    supports_r4_multiseq: bool = False
    supports_r5_infill: bool = False
    supports_r5_diffusion_full: bool = False
    supports_r5_causal_free: bool = False
    trust_remote_code: bool = False
    allow_fp16: bool = False
    notes: str = ""

    @property
    def is_causal(self) -> bool:
        return self.paradigm in ("causal_lm", "multiseq_causal")

    @property
    def is_masked(self) -> bool:
        return self.paradigm in (
            "masked_lm",
            "discrete_diffusion",
            "structure_aware_mlm",
            "msa_mlm",
        )


def _spec(
    arm: str,
    family: str,
    paradigm: str,
    loader_kind: str,
    candidate_dirs: tuple[str, ...],
    hf_repo: str,
    **kwargs,
) -> tuple[str, GeneralPLMSpec]:
    return (
        arm,
        GeneralPLMSpec(
            arm=arm,
            family=family,
            paradigm=paradigm,
            loader_kind=loader_kind,
            candidate_dirs=candidate_dirs,
            hf_repo=hf_repo,
            **kwargs,
        ),
    )


GENERAL_PLM_REGISTRY: dict[str, GeneralPLMSpec] = dict(
    [
        # ----------------------------------------------------- ESM2 ladder (6)
        _spec(
            "esm2-8m",
            "esm2",
            "masked_lm",
            "esm_mlm",
            ("esm2-8m", "esm2_t6_8M_UR50D", "esm2_8m"),
            "facebook/esm2_t6_8M_UR50D",
            approx_params=7_800_000,
            max_residues=1024,
            supports_r5_infill=True,
        ),
        _spec(
            "esm2-35m",
            "esm2",
            "masked_lm",
            "esm_mlm",
            ("esm2-35m", "esm2_t12_35M_UR50D", "esm2_35m"),
            "facebook/esm2_t12_35M_UR50D",
            approx_params=34_000_000,
            max_residues=1024,
            supports_r5_infill=True,
        ),
        _spec(
            "esm2-150m",
            "esm2",
            "masked_lm",
            "esm_mlm",
            ("esm2-150m", "esm2_t30_150M_UR50D", "esm2_150m"),
            "facebook/esm2_t30_150M_UR50D",
            approx_params=148_000_000,
            max_residues=1024,
            supports_r5_infill=True,
        ),
        _spec(
            "esm2-650m",
            "esm2",
            "masked_lm",
            "esm_mlm",
            ("esm2-650m", "esm2_t33_650M_UR50D", "esm2_650m"),
            "facebook/esm2_t33_650M_UR50D",
            approx_params=651_000_000,
            max_residues=1024,
            supports_r5_infill=True,
        ),
        _spec(
            "esm2-3b",
            "esm2",
            "masked_lm",
            "esm_mlm",
            ("esm2-3b", "esm2_t36_3B_UR50D", "esm2_3b"),
            "facebook/esm2_t36_3B_UR50D",
            approx_params=2_840_000_000,
            max_residues=1024,
            supports_r5_infill=True,
            notes="Avoid FP16 (activation overflow); use FP32 or declared BF16.",
        ),
        _spec(
            "esm2-15b",
            "esm2",
            "masked_lm",
            "esm_mlm",
            ("esm2-15b", "esm2_t48_15B_UR50D", "esm2_15b"),
            "facebook/esm2_t48_15B_UR50D",
            approx_params=14_900_000_000,
            max_residues=1024,
            supports_r5_infill=True,
            notes="Fits single H200 in FP32 (~60 GB); avoid FP16 (overflows).",
        ),
        # ----------------------------------------------------- DPLM ladder (3)
        _spec(
            "dplm-150m",
            "dplm",
            "discrete_diffusion",
            "esm_mlm",
            ("dplm-150m", "dplm_150m"),
            "airkingbd/dplm_150m",
            approx_params=150_000_000,
            max_residues=1024,
            supports_r5_infill=True,
            supports_r5_diffusion_full=True,
            notes="Absorbing-state discrete diffusion initialised from ESM2-150M.",
        ),
        _spec(
            "dplm-650m",
            "dplm",
            "discrete_diffusion",
            "esm_mlm",
            ("dplm-650m", "dplm_650m"),
            "airkingbd/dplm_650m",
            approx_params=650_000_000,
            max_residues=1024,
            supports_r5_infill=True,
            supports_r5_diffusion_full=True,
            notes="Absorbing-state discrete diffusion initialised from ESM2-650M.",
        ),
        _spec(
            "dplm-3b",
            "dplm",
            "discrete_diffusion",
            "esm_mlm",
            ("dplm-3b", "dplm_3b"),
            "airkingbd/dplm_3b",
            approx_params=3_000_000_000,
            max_residues=1024,
            supports_r5_infill=True,
            supports_r5_diffusion_full=True,
            notes="Absorbing-state discrete diffusion initialised from ESM2-3B.",
        ),
        # --------------------------------------------------- ESMC-hf ladder (3)
        _spec(
            "esmc-300m",
            "esmc",
            "masked_lm",
            "esmc_hf",
            ("esmc-300m-hf", "esmc-300m", "ESMC-300M-hf", "esmc_300m"),
            "biohub/ESMC-300M-hf",
            approx_params=300_000_000,
            max_residues=2048,
            supports_r5_infill=True,
            trust_remote_code=True,
            notes="ESM C HuggingFace port; may require transformers>=5 (e.g. runtimes/esmfold2) if native esmc class is used.",
        ),
        _spec(
            "esmc-600m",
            "esmc",
            "masked_lm",
            "esmc_hf",
            ("esmc-600m-hf", "esmc-600m", "ESMC-600M-hf", "esmc_600m"),
            "biohub/ESMC-600M-hf",
            approx_params=600_000_000,
            max_residues=2048,
            supports_r5_infill=True,
            trust_remote_code=True,
        ),
        _spec(
            "esmc-6b",
            "esmc",
            "masked_lm",
            "esmc_hf",
            ("esmc-6b-hf", "esmc-6b", "ESMC-6B-hf", "esmc_6b"),
            "biohub/ESMC-6B-hf",
            approx_params=6_000_000_000,
            max_residues=2048,
            supports_r5_infill=True,
            trust_remote_code=True,
        ),
        # -------------------------------------------------- AMPLIFY ladder (2)
        _spec(
            "amplify-120m",
            "amplify",
            "masked_lm",
            "amplify_mlm",
            ("amplify-120m", "AMPLIFY_120M", "AMPLIFY-120M"),
            "chandar-lab/AMPLIFY_120M",
            approx_params=120_000_000,
            max_residues=2048,
            supports_r5_infill=True,
            trust_remote_code=True,
        ),
        _spec(
            "amplify-350m",
            "amplify",
            "masked_lm",
            "amplify_mlm",
            ("amplify-350m", "AMPLIFY_350M", "AMPLIFY-350M"),
            "chandar-lab/AMPLIFY_350M",
            approx_params=350_000_000,
            max_residues=2048,
            supports_r5_infill=True,
            trust_remote_code=True,
        ),
        # --------------------------------------------------- SaProt ladder (3)
        _spec(
            "saprot-35m",
            "saprot",
            "structure_aware_mlm",
            "saprot_mlm",
            ("saprot-35m", "SaProt_35M_AF2", "saprot_35m"),
            "westlake-repl/SaProt_35M_AF2",
            approx_params=35_000_000,
            max_residues=1024,
            supports_r5_infill=True,
            notes="441-token AA+3Di alphabet; supports seq_only (#-masked marginalization) and foldseek_3di modes.",
        ),
        _spec(
            "saprot-650m",
            "saprot",
            "structure_aware_mlm",
            "saprot_mlm",
            ("saprot-650m", "SaProt_650M_AF2", "saprot_650m"),
            "westlake-repl/SaProt_650M_AF2",
            approx_params=650_000_000,
            max_residues=1024,
            supports_r5_infill=True,
            notes="441-token AA+3Di alphabet; supports seq_only (#-masked marginalization) and foldseek_3di modes.",
        ),
        _spec(
            "saprot-1.3b",
            "saprot",
            "structure_aware_mlm",
            "saprot_mlm",
            ("saprot-1.3b", "SaProt_1.3B_AF2", "saprot_1.3b"),
            "westlake-repl/SaProt_1.3B_AF2",
            approx_params=1_300_000_000,
            max_residues=1024,
            supports_r5_infill=True,
            notes="441-token AA+3Di alphabet; supports seq_only (#-masked marginalization) and foldseek_3di modes.",
        ),
        # ------------------------------------------- ProteinGLM-MLM ladder (3)
        _spec(
            "proteinglm-1b-mlm",
            "proteinglm_mlm",
            "masked_lm",
            "proteinglm_mlm",
            ("proteinglm-1b-mlm", "ProteinGLM-1B-MLM"),
            "proteinglm/proteinglm-1b-mlm",
            approx_params=1_000_000_000,
            max_residues=1024,
            supports_r5_infill=True,
            trust_remote_code=True,
        ),
        _spec(
            "proteinglm-3b-mlm",
            "proteinglm_mlm",
            "masked_lm",
            "proteinglm_mlm",
            ("proteinglm-3b-mlm", "ProteinGLM-3B-MLM"),
            "proteinglm/proteinglm-3b-mlm",
            approx_params=3_000_000_000,
            max_residues=1024,
            supports_r5_infill=True,
            trust_remote_code=True,
        ),
        _spec(
            "proteinglm-10b-mlm",
            "proteinglm_mlm",
            "masked_lm",
            "proteinglm_mlm",
            ("proteinglm-10b-mlm", "ProteinGLM-10B-MLM"),
            "proteinglm/proteinglm-10b-mlm",
            approx_params=10_000_000_000,
            max_residues=1024,
            supports_r5_infill=True,
            trust_remote_code=True,
        ),
        # ---------------------------------------------------------- Prime-690M
        _spec(
            "prime-690m",
            "prime",
            "masked_lm",
            "prime_mlm",
            ("Prime_690M", "prime-690m", "prime_690m"),
            "AI4Protein/Prime_690M",
            approx_params=690_000_000,
            max_residues=1024,
            supports_r5_infill=True,
            trust_remote_code=True,
            notes="Already staged on GPFS at /gpfs/jiaotongdamoxing/zhk_zip/models/Prime_690M.",
        ),
        # -------------------------------------------------- Multi-Seq R4 PLMs
        _spec(
            "msa-transformer-100m",
            "msa_transformer",
            "msa_mlm",
            "msa_transformer",
            (
                "esm-msa1b-100m",
                "msa-transformer-100m",
                "esm_msa1b_t12_100M_UR50S",
            ),
            "katielink/esm_msa1b_t12_100M_UR50S",
            approx_params=100_000_000,
            max_residues=1024,
            supports_r1_r3=False,
            supports_r4_multiseq=True,
            trust_remote_code=True,
            notes="2D axial-attention MSA Transformer for R4 homolog-aligned context scoring.",
        ),
        _spec(
            "protgpt3-msa",
            "protgpt3_msa",
            "multiseq_causal",
            "protgpt3_causal",
            ("protgpt3-msa", "ProtGPT3-MSA"),
            "AI4PD/ProtGPT3-MSA",
            approx_params=1_300_000_000,
            max_residues=4096,
            supports_r1_r3=True,
            supports_r4_multiseq=True,
            supports_r5_infill=True,
            supports_r5_causal_free=True,
            notes="Multi-sequence homolog-trained ProtGPT3 checkpoint.",
        ),
        _spec(
            "dayhoff-3b-msa",
            "dayhoff_msa",
            "multiseq_causal",
            "dayhoff_causal",
            (
                "dayhoff-3b-msa",
                "dayhoff-3b-GR-HM-c",
                "Dayhoff-3b-GR-HM-c",
                "Dayhoff-3b-GR-HM",
                "dayhoff-3b-gr-hm-c",
            ),
            "microsoft/Dayhoff-3b-GR-HM-c",
            approx_params=3_000_000_000,
            max_residues=8192,
            supports_r1_r3=True,
            supports_r4_multiseq=True,
            supports_r5_infill=True,
            supports_r5_causal_free=True,
            trust_remote_code=True,
            notes="Multi-sequence homolog-trained Dayhoff Jamba/hybrid checkpoint.",
        ),
        # --------------------------------------------------- Newer Causal PLMs
        _spec(
            "protgpt3-112m",
            "protgpt3",
            "causal_lm",
            "protgpt3_causal",
            ("ProtGPT3-112M", "protgpt3-112m"),
            "AI4PD/ProtGPT3-112M",
            approx_params=112_000_000,
            max_residues=1022,
            supports_r4_multiseq=True,
            supports_r5_infill=True,
            supports_r5_causal_free=True,
            notes="MixtralForCausalLM with bos_direction_seq packing (<|bos|> + '1' + AA20).",
        ),
        _spec(
            "protgpt3-1.3b",
            "protgpt3",
            "causal_lm",
            "protgpt3_causal",
            ("ProtGPT3-1.3B", "protgpt3-1.3b"),
            "AI4PD/ProtGPT3-1.3B",
            approx_params=1_300_000_000,
            max_residues=1022,
            supports_r4_multiseq=True,
            supports_r5_infill=True,
            supports_r5_causal_free=True,
            notes="Existing panel rung retained as reference comparator for 112M->1.3B->10B ladder.",
        ),
        _spec(
            "protgpt3-10b",
            "protgpt3",
            "causal_lm",
            "protgpt3_causal",
            ("ProtGPT3-10B", "protgpt3-10b"),
            "AI4PD/ProtGPT3-10B",
            approx_params=10_000_000_000,
            max_residues=1022,
            supports_r4_multiseq=True,
            supports_r5_infill=True,
            supports_r5_causal_free=True,
            notes="Completes ProtGPT3 112M -> 1.3B -> 10B causal scaling ladder.",
        ),
        _spec(
            "progen3-219m",
            "progen3",
            "causal_lm",
            "progen3_causal",
            ("progen3-219m", "ProGen3-219M"),
            "Profluent-Bio/progen3-219m",
            default_dtype="bfloat16",
            approx_params=219_000_000,
            max_residues=2048,
            supports_r4_multiseq=True,
            supports_r5_infill=True,
            supports_r5_causal_free=True,
            allow_fp16=True,
            notes="Loaded via src.capability.models.progen3 megablocks->eager strict conversion (BF16 only).",
        ),
        _spec(
            "progen3-339m",
            "progen3",
            "causal_lm",
            "progen3_causal",
            ("progen3-339m", "ProGen3-339M"),
            "Profluent-Bio/progen3-339m",
            default_dtype="bfloat16",
            approx_params=339_000_000,
            max_residues=2048,
            supports_r4_multiseq=True,
            supports_r5_infill=True,
            supports_r5_causal_free=True,
            allow_fp16=True,
        ),
        _spec(
            "progen3-762m",
            "progen3",
            "causal_lm",
            "progen3_causal",
            ("progen3-762m", "ProGen3-762M"),
            "Profluent-Bio/progen3-762m",
            default_dtype="bfloat16",
            approx_params=762_000_000,
            max_residues=2048,
            supports_r4_multiseq=True,
            supports_r5_infill=True,
            supports_r5_causal_free=True,
            allow_fp16=True,
        ),
        _spec(
            "progen3-1b",
            "progen3",
            "causal_lm",
            "progen3_causal",
            ("progen3-1b", "ProGen3-1B"),
            "Profluent-Bio/progen3-1b",
            default_dtype="bfloat16",
            approx_params=1_000_000_000,
            max_residues=2048,
            supports_r4_multiseq=True,
            supports_r5_infill=True,
            supports_r5_causal_free=True,
            allow_fp16=True,
        ),
        _spec(
            "dayhoff-170m",
            "dayhoff",
            "causal_lm",
            "dayhoff_causal",
            ("dayhoff-170m", "Dayhoff-170m-UR50", "Dayhoff-170M"),
            "microsoft/Dayhoff-170m-UR50",
            approx_params=170_000_000,
            max_residues=8192,
            supports_r4_multiseq=True,
            supports_r5_infill=True,
            supports_r5_causal_free=True,
            trust_remote_code=True,
        ),
        _spec(
            "dayhoff-3b",
            "dayhoff",
            "causal_lm",
            "dayhoff_causal",
            ("dayhoff-3b", "Dayhoff-3b-UR90", "Dayhoff-3B"),
            "microsoft/Dayhoff-3b-UR90",
            approx_params=3_000_000_000,
            max_residues=8192,
            supports_r4_multiseq=True,
            supports_r5_infill=True,
            supports_r5_causal_free=True,
            trust_remote_code=True,
        ),
    ]
)


def list_arms() -> tuple[str, ...]:
    """Return every declared exploratory arm name in registry order."""
    return tuple(GENERAL_PLM_REGISTRY.keys())


def get_spec(arm: str) -> GeneralPLMSpec:
    """Look up a declared general-PLM arm or fail fast with available names."""
    if arm not in GENERAL_PLM_REGISTRY:
        raise ValueError(
            f"Unknown general-PLM arm {arm!r}. Registered arms: {list(GENERAL_PLM_REGISTRY)}"
        )
    return GENERAL_PLM_REGISTRY[arm]


def arms_by_family(family: str) -> tuple[str, ...]:
    """Return all declared arm names belonging to ``family``."""
    return tuple(name for name, spec in GENERAL_PLM_REGISTRY.items() if spec.family == family)


def arms_for_module(module: str) -> tuple[str, ...]:
    """Return declared arms compatible with a scientific module (r1, r2, r3, r4, r5_infill, r5_diffusion, r5_causal)."""
    mod = module.lower()
    if mod in ("r1", "r2", "r3"):
        return tuple(k for k, s in GENERAL_PLM_REGISTRY.items() if s.supports_r1_r3)
    if mod == "r4":
        return tuple(k for k, s in GENERAL_PLM_REGISTRY.items() if s.supports_r4_multiseq)
    if mod == "r5_infill":
        return tuple(k for k, s in GENERAL_PLM_REGISTRY.items() if s.supports_r5_infill)
    if mod == "r5_diffusion":
        return tuple(k for k, s in GENERAL_PLM_REGISTRY.items() if s.supports_r5_diffusion_full)
    if mod == "r5_causal":
        return tuple(k for k, s in GENERAL_PLM_REGISTRY.items() if s.supports_r5_causal_free)
    raise ValueError(f"Unsupported module selector {module!r}")


def _has_checkpoint_payload(directory: Path) -> bool:
    """Require config.json (or fair-esm .pt) plus at least one weight file."""
    if not directory.is_dir():
        return False
    has_weights = any(
        item.is_file()
        and (
            item.suffix in _WEIGHT_SUFFIXES
            or item.name.endswith(".safetensors.index.json")
            or item.name.endswith(".bin.index.json")
        )
        for item in directory.iterdir()
    )
    if not has_weights:
        return False
    if (directory / "config.json").is_file():
        return True
    # Fair-ESM standalone .pt checkpoints (e.g. esm_msa1b_t12_100M_UR50S.pt)
    if any(item.suffix == ".pt" and item.is_file() for item in directory.iterdir()):
        return True
    return False


def _candidate_roots(
    model_root: Path | str | None = None,
    extra_roots: Iterable[Path | str] = (),
) -> list[Path]:
    """Assemble deduplicated search roots in deterministic priority order."""
    raw: list[Path] = []
    if model_root is not None and str(model_root).strip():
        raw.append(Path(model_root))
    for var in ("GENERAL_PLM_MODEL_ROOT", "MODEL_ROOT", "MODEL_BASE_DIR"):
        val = os.environ.get(var, "").strip()
        if val:
            raw.append(Path(val))
    for item in extra_roots:
        if item is not None and str(item).strip():
            raw.append(Path(item))
    raw.extend(DEFAULT_SEARCH_ROOTS)

    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in raw:
        expanded = path.expanduser()
        if expanded not in seen:
            seen.add(expanded)
            ordered.append(expanded)
    return ordered


def resolve_checkpoint_dir(
    arm: str,
    *,
    model_root: Path | str | None = None,
    extra_roots: Iterable[Path | str] = (),
) -> Path:
    """Locate a complete local checkpoint directory for ``arm`` or raise FileNotFoundError.

    When ``model_root`` is explicitly provided, ONLY ``model_root`` (as a direct
    checkpoint directory or parent holding one of ``spec.candidate_dirs``) is
    searched, so an explicit ``--model-root`` pointing to a missing or
    half-downloaded checkpoint fails fast rather than silently falling back to a
    different host directory.
    """
    spec = get_spec(arm)
    if model_root is not None and str(model_root).strip():
        direct = Path(model_root).expanduser()
        if _has_checkpoint_payload(direct):
            return direct.resolve()
        checked_explicit: list[str] = [str(direct)]
        for cand in spec.candidate_dirs:
            probe = direct / cand
            checked_explicit.append(str(probe))
            if _has_checkpoint_payload(probe):
                return probe.resolve()
        raise FileNotFoundError(
            f"{arm}: no complete local checkpoint found under explicit model_root={direct} "
            f"(needs config.json + weight files). Checked {checked_explicit}. "
            f"Expected HF repo: {spec.hf_repo}."
        )

    roots = _candidate_roots(model_root=None, extra_roots=extra_roots)
    checked: list[str] = []
    for root in roots:
        for cand in spec.candidate_dirs:
            probe = root / cand
            checked.append(str(probe))
            if _has_checkpoint_payload(probe):
                return probe.resolve()

    raise FileNotFoundError(
        f"{arm}: no complete local checkpoint found (needs config.json + weight files). "
        f"Searched {len(checked)} paths across roots {[str(r) for r in roots]} "
        f"for candidate directories {list(spec.candidate_dirs)}. "
        f"Expected HF repo: {spec.hf_repo}."
    )
