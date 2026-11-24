"""Unified Experiment 4.2: Eight-Domain Family Phase Space Across All Generative Paradigms.

Compares single-family Potts MCMC sampling (``P``) and held-out natural homologs (``H``)
against 22 protein language models spanning autoregressive protein/joint/hybrid models,
discrete absorbing diffusion (DPLM), and multimodal discrete diffusion (ESM3-1.4B) on the
eight sealed E20 evaluation domains (``1E0L, 1F0M, 1IFY, 1V1C, 2K1B, 2LGW, 2LVN, 3DKM``).

Design invariants
-----------------
1. **Bit-for-bit local reuse**: Reuses the cached & folded E20 confirmatory records
   (``P``, ``H``, ``progen2-medium``, ``protgpt2``) and ProGen3 robustness records
   (``progen3-112m``), verifying ``< 1e-12`` numerical reproduction against
   ``confirmatory.json`` and ``robustness.json``.
2. **Unified 5-arm phase-space grid** for all 19 newly evaluated PLMs (``pool_size=100``
   per ``(model, domain, arm)`` cell):
   - ``N``: native unguided fixed-length sampling, ``lambda = (0.0, 0.0)``
   - ``Pr``: profile guidance at ``<20%``-identity dose, ``lambda = (0.25, 0.0)``
   - ``Pr_strong``: profile frontier anchoring dose, ``lambda = (1.0, 0.0)``
   - ``C``: pairwise Potts couplings alone, ``lambda = (0.0, 4.0)``
   - ``Cx``: column-permuted coupling control, ``lambda = (0.0, 4.0)``, exact
     ``permutation_for(domain, L, 20261013)``.
3. **Exact bias accounting**: Every guided sampler (autoregressive, ProGen3 singleton MoE,
   DPLM ``uniform_random`` 1-by-1 diffusion, and ESM3 cosine 1-by-1 diffusion) checks that
   the step-accumulated applied bias agrees with ``sequence_component_sums`` within ``1e-6``.
4. **Dual-stratum reporting**: Reports BOTH the primary ``<20%`` WT-identity novel stratum
   (with the frozen 25-candidate floor) AND the unrestricted whole-pool phase-space
   trajectory, with 9,999-draw BCa domain bootstrap 95% confidence intervals.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from ..context.family_potts import FamilyModel, encode as encode_aa20, load_family_model
from ..core.amino_acids import AA20
from ..core.io import sha256_file
from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS, bootstrap_unit_floor
from ..decoding import guided_decoding as gd
from ..generation import fold_recapitulation as fr
from ..ladder import fold_geometry as fg
from .roster import ARM_METADATA, REPO_ROOT, UNIFIED_OUTPUT_DIR

SCHEMA_VERSION = "r6_experiment_4_2_phase_space_unified_v1"
GENERATED_RECORD_SCHEMA = "exp4_2_generated_sequence_v1"
FOLDED_RECORD_SCHEMA = "exp4_2_folded_sequence_v1"

SEALED_8_DOMAINS: tuple[str, ...] = (
    "1E0L",
    "1F0M",
    "1IFY",
    "1V1C",
    "2K1B",
    "2LGW",
    "2LVN",
    "3DKM",
)

PERMUTATION_SEED = 20261013
BOOTSTRAP_SEED = 20261013
N_BOOTSTRAP = 9999
CONFIDENCE = 0.95
ADMISSIBLE_WT_IDENTITY_PCT = 20.0
MIN_ADMISSIBLE_PER_CELL = 25
DEFAULT_POOL_SIZE = 100
DEFAULT_BATCH_SIZE = 50
DEFAULT_PROGEN3_BATCH_SIZE = 1
DEFAULT_BASE_SEED = 20261124
GUIDANCE_BIAS_TOL = 1e-6
EPS = 1e-12

DEFAULT_FAMILY_REF_DIR = Path("data/pairwise_assets/family_references_jointh_20261013")
DEFAULT_MSA_DIR = Path("data/pairwise_assets/same12_provenance_20260924/msa")
DEFAULT_WT_FASTA = Path("data/pairwise_assets/same12_provenance_20260924/query_12.faa")
DEFAULT_WT_REF_MANIFEST = Path(
    "results/R6/e20_wild_type_references_20261013/freeze/wild_type_references.json"
)
DEFAULT_E20_PER_SEQUENCE = Path(
    "results/R6/e20_confirmatory_20261013/e20conf_measure_cpu/per_sequence.jsonl"
)
DEFAULT_E20_CONFIRMATORY_JSON = Path(
    "results/R6/e20_confirmatory_analysis_20261014/confirmatory.json"
)
DEFAULT_PG3RB_PER_SEQUENCE = Path(
    "results/R6/progen3_robustness_20261015/pg3rb_bfloat16_measure_cpu/per_sequence.jsonl"
)
DEFAULT_PG3RB_ROBUSTNESS_JSON = Path(
    "results/R6/progen3_robustness_analysis_20261015/robustness.json"
)
DEFAULT_WORK_DIR = UNIFIED_OUTPUT_DIR / "exp4_2_phase_space"
DEFAULT_OUTPUT_JSON = UNIFIED_OUTPUT_DIR / "r6_experiment_4_2_phase_space_unified.json"

BCA_COVERAGE_NOTE = (
    "BCa over 8 domains: realised coverage of a nominal 95% interval is about 0.89 at eight "
    "units (src.capability.core.statistics); read bounds as optimistic rather than exact."
)

AA20_SET = frozenset(AA20)
AA20_INDEX: dict[str, int] = {aa: idx for idx, aa in enumerate(AA20)}


@dataclass(frozen=True)
class PhaseSpaceArmSpec:
    """One standardized guidance setting in the Experiment 4.2 phase-space protocol."""

    arm_key: str
    guidance_arm: str
    lambda_fields: float
    lambda_coupling: float
    description: str

    @property
    def strength(self) -> gd.GuidanceStrength:
        return gd.GuidanceStrength(
            fields=float(self.lambda_fields),
            coupling=float(self.lambda_coupling),
        )


PHASE_SPACE_ARMS: tuple[PhaseSpaceArmSpec, ...] = (
    PhaseSpaceArmSpec(
        arm_key="N",
        guidance_arm="native",
        lambda_fields=0.0,
        lambda_coupling=0.0,
        description="Unguided native fixed-length sampling",
    ),
    PhaseSpaceArmSpec(
        arm_key="Pr",
        guidance_arm="profile",
        lambda_fields=0.25,
        lambda_coupling=0.0,
        description="Independent-site profile guidance at <20%-identity dose",
    ),
    PhaseSpaceArmSpec(
        arm_key="Pr_strong",
        guidance_arm="profile",
        lambda_fields=1.0,
        lambda_coupling=0.0,
        description="Independent-site profile frontier anchoring dose",
    ),
    PhaseSpaceArmSpec(
        arm_key="C",
        guidance_arm="coupling",
        lambda_fields=0.0,
        lambda_coupling=4.0,
        description="Pairwise Potts coupling guidance alone",
    ),
    PhaseSpaceArmSpec(
        arm_key="Cx",
        guidance_arm="permuted_coupling",
        lambda_fields=0.0,
        lambda_coupling=4.0,
        description="Column-permuted Potts coupling magnitude-matched control",
    ),
)

PHASE_SPACE_ARM_MAP: dict[str, PhaseSpaceArmSpec] = {
    spec.arm_key: spec for spec in PHASE_SPACE_ARMS
}
PHASE_SPACE_ARM_KEYS: tuple[str, ...] = tuple(spec.arm_key for spec in PHASE_SPACE_ARMS)


@dataclass(frozen=True)
class PhaseSpaceModelSpec:
    """Authoritative specification for one PLM in the 22-PLM Experiment 4.2 panel."""

    model_id: str
    backend: str  # cached_e20 | cached_pg3rb | guided_decoding | general_plm_causal | progen3_causal | dplm_diffusion | esm3_diffusion
    default_dtype: str
    default_batch_size: int
    use_cache: bool = True
    estimated_cost: float = 1.0
    notes: str = ""


PHASE_SPACE_MODELS: tuple[PhaseSpaceModelSpec, ...] = (
    # ---- Cached local E20 / ProGen3 robustness PLMs (3; 0 GPU needed)
    PhaseSpaceModelSpec("progen2-medium", "cached_e20", "float32", 50, True, 0.0, "Cached E20 confirmatory"),
    PhaseSpaceModelSpec("protgpt2", "cached_e20", "float32", 50, True, 0.0, "Cached E20 confirmatory (BPE)"),
    PhaseSpaceModelSpec("progen3-112m", "cached_pg3rb", "bfloat16", 1, True, 0.0, "Cached ProGen3 robustness"),
    # ---- ProGen2 ladder (+4)
    PhaseSpaceModelSpec("progen2-small", "guided_decoding", "float32", 50, True, 1.0),
    PhaseSpaceModelSpec("progen2-base", "guided_decoding", "float32", 50, True, 2.0),
    PhaseSpaceModelSpec("progen2-large", "guided_decoding", "float32", 50, True, 4.5),
    PhaseSpaceModelSpec("progen2-xlarge", "guided_decoding", "float32", 50, True, 9.0),
    # ---- ProGen3 ladder (+5; MegaBlocks MoE singleton_only=True, BF16 required)
    PhaseSpaceModelSpec("progen3-219m", "progen3_causal", "bfloat16", 1, True, 5.0, "MegaBlocks MoE singleton"),
    PhaseSpaceModelSpec("progen3-339m", "progen3_causal", "bfloat16", 1, True, 6.0, "MegaBlocks MoE singleton"),
    PhaseSpaceModelSpec("progen3-762m", "progen3_causal", "bfloat16", 1, True, 8.0, "MegaBlocks MoE singleton"),
    PhaseSpaceModelSpec("progen3-1b", "progen3_causal", "bfloat16", 1, True, 9.5, "MegaBlocks MoE singleton"),
    PhaseSpaceModelSpec("progen3-3b", "progen3_causal", "bfloat16", 1, True, 14.0, "MegaBlocks MoE singleton"),
    # ---- ProtGPT3 ladder (+3; single-AA20 Mixtral)
    PhaseSpaceModelSpec("protgpt3-112m", "general_plm_causal", "float32", 50, True, 1.2),
    PhaseSpaceModelSpec("protgpt3-1.3b", "general_plm_causal", "float32", 50, True, 3.0),
    PhaseSpaceModelSpec("protgpt3-10b", "general_plm_causal", "float32", 50, True, 12.0),
    # ---- Additional causal / hybrid / joint (+4)
    PhaseSpaceModelSpec("proteinglm-7b-clm", "guided_decoding", "float32", 50, True, 10.0),
    PhaseSpaceModelSpec("rita-xl", "guided_decoding", "float32", 25, False, 4.5, "No KV cache per ADMITTED"),
    PhaseSpaceModelSpec("prollama", "guided_decoding", "bfloat16", 50, True, 8.0, "LLaMA-2 SentencePiece span bias"),
    PhaseSpaceModelSpec("dayhoff-170m", "general_plm_causal", "float32", 50, True, 2.5, "Jamba HybridMambaAttentionDynamicCache"),
    # ---- Discrete absorbing diffusion (+3)
    PhaseSpaceModelSpec("dplm-150m", "dplm_diffusion", "float32", 50, False, 1.5, "uniform_random 1-by-1 unmasking"),
    PhaseSpaceModelSpec("dplm-650m", "dplm_diffusion", "float32", 50, False, 3.0, "uniform_random 1-by-1 unmasking"),
    PhaseSpaceModelSpec("dplm-3b", "dplm_diffusion", "float32", 50, False, 8.5, "uniform_random 1-by-1 unmasking"),
    # ---- Multimodal discrete diffusion (+1)
    PhaseSpaceModelSpec("esm3-1.4b", "esm3_diffusion", "bfloat16", 50, False, 6.0, "cosine 1-by-1 unmasking"),
)

PHASE_SPACE_MODEL_MAP: dict[str, PhaseSpaceModelSpec] = {
    spec.model_id: spec for spec in PHASE_SPACE_MODELS
}
CACHED_LOCAL_PLM_MODELS: tuple[str, ...] = tuple(
    spec.model_id for spec in PHASE_SPACE_MODELS if spec.backend.startswith("cached_")
)
NEW_H200_MODELS: tuple[str, ...] = tuple(
    spec.model_id for spec in PHASE_SPACE_MODELS if not spec.backend.startswith("cached_")
)
REFERENCE_ARMS: Mapping[str, str] = {"P": "potts-joint", "H": "natural"}

# Validate every PLM id against the canonical unified roster at import time.
for _spec in PHASE_SPACE_MODELS:
    if _spec.model_id not in ARM_METADATA:
        raise AssertionError(f"{_spec.model_id!r} missing from unified ARM_METADATA")
assert len(PHASE_SPACE_MODELS) == 23
assert len(CACHED_LOCAL_PLM_MODELS) == 3
assert len(NEW_H200_MODELS) == 20


# --------------------------------------------------------------------------- #
# Deterministic sharding & seed derivation
# --------------------------------------------------------------------------- #


def cell_stream_seed(
    model_id: str,
    domain: str,
    arm_key: str,
    *,
    base_seed: int = DEFAULT_BASE_SEED,
) -> int:
    """Deterministic, cell-unique, layout-independent stream seed."""
    tag = f"{int(base_seed)}|{model_id}|{domain}|{arm_key}".encode("utf-8")
    digest = int(hashlib.sha256(tag).hexdigest()[:12], 16)
    # Keep comfortably within 32-bit signed range even after +pool_size offset.
    return int((digest % 1_900_000_000) + 10_000)


def plan_generation_shards(
    models: Sequence[str],
    num_shards: int,
) -> list[list[str]]:
    """Greedy LPT bin-packing of models across ``num_shards`` GPUs by estimated compute cost.

    Each model runs on exactly one shard so weights are loaded once per GPU.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    unique_models: list[str] = []
    for m in models:
        if m not in PHASE_SPACE_MODEL_MAP:
            raise ValueError(f"Unknown phase-space model {m!r}")
        if m not in unique_models:
            unique_models.append(m)
    ordered = sorted(
        unique_models,
        key=lambda m: (-PHASE_SPACE_MODEL_MAP[m].estimated_cost, m),
    )
    bins: list[list[str]] = [[] for _ in range(num_shards)]
    loads = [0.0] * num_shards
    for model_id in ordered:
        target = min(range(num_shards), key=lambda idx: (loads[idx], len(bins[idx]), idx))
        bins[target].append(model_id)
        loads[target] += PHASE_SPACE_MODEL_MAP[model_id].estimated_cost
    return bins


def plan_fold_shards(
    cells: Sequence[tuple[str, str, str]],
    num_shards: int,
) -> list[list[tuple[str, str, str]]]:
    """Deterministic round-robin sharding of ``(model, domain, arm)`` cells for ESMFold2."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    ordered = sorted(cells)
    bins: list[list[tuple[str, str, str]]] = [[] for _ in range(num_shards)]
    for idx, cell in enumerate(ordered):
        bins[idx % num_shards].append(cell)
    return bins


# --------------------------------------------------------------------------- #
# Per-candidate sequence & Potts metrics
# --------------------------------------------------------------------------- #


def compute_potts_energies(
    family_model: FamilyModel, sequence: str
) -> tuple[float, float, float]:
    """Return ``(full, fields, couplings)`` Potts energy in plmc sign convention (higher = better)."""
    codes = encode_aa20(sequence)
    if len(codes) != family_model.length:
        raise ValueError(
            f"{family_model.domain}: sequence length {len(codes)} != model length {family_model.length}"
        )
    idx = np.arange(family_model.length)
    fields = float(family_model.h[idx, codes].sum())
    pair = family_model.J[idx[:, None], idx[None, :], codes[:, None], codes[None, :]]
    couplings = float(np.triu(pair, k=1).sum())
    full = float(fields + couplings)
    return full, fields, couplings


def compute_shannon_entropy_bits(sequence: str) -> float:
    """Residue-composition Shannon entropy ``H`` in bits (``log2``) for one sequence."""
    if not sequence:
        raise ValueError("Cannot compute Shannon entropy of an empty sequence")
    counts = np.zeros(20, dtype=np.float64)
    for ch in sequence:
        idx = AA20_INDEX.get(ch)
        if idx is None:
            raise ValueError(f"Non-canonical residue {ch!r} in sequence")
        counts[idx] += 1.0
    probs = counts[counts > 0.0] / float(len(sequence))
    return float(-np.sum(probs * np.log2(probs)))


def load_frozen_wt_ca_traces(
    manifest_path: Path | str = DEFAULT_WT_REF_MANIFEST,
    *,
    domains: Sequence[str] = SEALED_8_DOMAINS,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    """Load verified frozen WT CA traces and sequences from ``wild_type_references.json``."""
    mpath = Path(manifest_path)
    manifest = fr.load_reference_manifest(mpath)
    traces: dict[str, np.ndarray] = {}
    sequences: dict[str, str] = {}
    for dom in domains:
        entry = manifest["domains"].get(dom)
        if entry is None:
            raise KeyError(f"Domain {dom!r} missing from WT reference manifest {mpath}")
        pdb_path = mpath.parent / entry["pdb"]
        ca = fg.parse_ca_trace(pdb_path.read_text(encoding="utf-8"))
        seq = str(entry["sequence"])
        if ca.shape != (len(seq), 3):
            raise ValueError(f"{dom}: WT CA shape {ca.shape} != ({len(seq)}, 3)")
        traces[dom] = ca
        sequences[dom] = seq
    return traces, sequences, manifest


def enrich_sequence_metrics(
    sequence: str,
    *,
    family_model: FamilyModel,
    msa_rows: Sequence[str],
) -> dict[str, float]:
    """Compute sequence-side identity, Potts energy, and composition metrics."""
    e_full, e_fields, e_couplings = compute_potts_energies(family_model, sequence)
    wt = family_model.wildtype
    return {
        "wt_identity_pct": float(fr.identity(sequence, wt)),
        "msa_max_identity_pct": float(fr.max_msa_identity(sequence, msa_rows)),
        "potts_energy_full": e_full,
        "potts_energy_fields": e_fields,
        "potts_energy_couplings": e_couplings,
        "shannon_H_bits": compute_shannon_entropy_bits(sequence),
    }


# --------------------------------------------------------------------------- #
# Guided Sampling Engines (Causal, ProGen3 Singleton, DPLM, ESM3)
# --------------------------------------------------------------------------- #


def build_phase_space_bias(
    family_model: FamilyModel,
    arm_spec: PhaseSpaceArmSpec,
    *,
    permutation_seed: int = PERMUTATION_SEED,
) -> gd.ArmBias:
    """Build the canonical ``guided_decoding.ArmBias`` for one ``(domain, arm_spec)``."""
    p_seed = int(permutation_seed) if arm_spec.guidance_arm == "permuted_coupling" else None
    return gd.build_arm_bias(arm_spec.guidance_arm, family_model, permutation_seed=p_seed)


def _verify_applied_bias(
    bias: gd.ArmBias,
    strength: gd.GuidanceStrength,
    sequence: str,
    applied_bias: float,
    *,
    label: str,
) -> tuple[float, float, float]:
    """Verify accumulated step bias equals per-residue ``sequence_component_sums`` within 1e-6."""
    fields_sum, coupling_sum = gd.sequence_component_sums(bias, sequence)
    expected = float(strength.fields * fields_sum + strength.coupling * coupling_sum)
    if abs(expected - float(applied_bias)) > GUIDANCE_BIAS_TOL:
        raise RuntimeError(
            f"{label}: accumulated step bias ({applied_bias:.9f}) differs from per-residue "
            f"recomputation ({expected:.9f}) by {abs(expected - applied_bias):.3e}"
        )
    return expected, float(fields_sum), float(coupling_sum)


def _make_generated_record(
    *,
    model_id: str,
    domain: str,
    arm_spec: PhaseSpaceArmSpec,
    seed: int,
    sequence_index: int,
    sequence: str,
    applied_bias: float,
    fields_score: float | None,
    coupling_score: float | None,
    mean_step_entropy: float,
    sampler: str,
    token_steps: int,
) -> dict[str, Any]:
    guided = arm_spec.arm_key != "N"
    guidance_score = (
        float((fields_score or 0.0) + (coupling_score or 0.0)) if guided else None
    )
    return {
        "schema_version": GENERATED_RECORD_SCHEMA,
        "id": f"{domain}_{model_id}_{arm_spec.arm_key}_idx{sequence_index:04d}_seed{int(seed)}",
        "model": model_id,
        "domain": domain,
        "arm": arm_spec.arm_key,
        "guidance_arm": arm_spec.guidance_arm,
        "lambda_fields": float(arm_spec.lambda_fields),
        "lambda_coupling": float(arm_spec.lambda_coupling),
        "seed": int(seed),
        "sequence_index": int(sequence_index),
        "sequence": sequence,
        "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
        "length": len(sequence),
        "guidance_score": guidance_score,
        "fields_score": fields_score,
        "coupling_score": coupling_score,
        "guidance_bias_applied": float(applied_bias),
        "mean_step_entropy": float(mean_step_entropy),
        "token_steps": int(token_steps),
        "sampler": sampler,
    }


def _entropy_from_logits_1d(scores: torch.Tensor) -> float:
    lp = F.log_softmax(scores, dim=-1)
    probs = lp.exp()
    finite = torch.isfinite(lp)
    return float(-(probs * torch.where(finite, lp, torch.zeros_like(lp))).sum().item())


def _build_layout_for_guided_arm(arm: Any, length: int) -> gd.NativeLayout:
    """Derive ``NativeLayout`` via ``gd.native_layout`` or direct AA20 slot layout for ProLLaMA."""
    if getattr(arm, "name", "") in ("prollama", "prollama-stage-1"):
        prompt = gd.generation_point(arm).prompt
        template = np.full(length, gd.RESIDUE_SLOT, dtype=np.int64)
        residues_before = np.arange(length + 1, dtype=np.int64)
        return gd.NativeLayout(
            arm=arm.name,
            length=length,
            prompt=prompt,
            template=template,
            literals={},
            residues_before=residues_before,
        )
    return gd.native_layout(arm, length)


class _InRangeVocabTokenizer:
    """Tokenizer view excluding non-residue special ids >= logit_width (e.g. RITA-XL id 26)."""

    def __init__(self, tokenizer: Any, logit_width: int) -> None:
        self._tok = tokenizer
        self._logit_width = int(logit_width)

    def get_vocab(self) -> dict[str, int]:
        return {
            k: int(v)
            for k, v in self._tok.get_vocab().items()
            if 0 <= int(v) < self._logit_width
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tok, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._tok(*args, **kwargs)


def _token_span_table_safe(
    arm: Any, layout: gd.NativeLayout, *, logit_width: int
) -> gd.TokenSpanTable:
    vocab = arm.tokenizer.get_vocab()
    if max(int(v) for v in vocab.values()) < int(logit_width):
        return gd.token_span_table(arm, layout, logit_width=int(logit_width))
    # Verify no canonical AA20 token sits outside logit_width before filtering non-residue specials.
    filtered_tok = _InRangeVocabTokenizer(arm.tokenizer, int(logit_width))
    proxy = type("_ArmView", (), {"name": arm.name, "tokenizer": filtered_tok})()
    return gd.token_span_table(proxy, layout, logit_width=int(logit_width))


@torch.no_grad()
def sample_guided_causal_decoding(
    *,
    arm: Any,
    model_id: str,
    family_model: FamilyModel,
    arm_spec: PhaseSpaceArmSpec,
    count: int,
    batch_size: int,
    stream_seed: int,
    use_cache: bool = True,
    logit_width: int | None = None,
    permutation_seed: int = PERMUTATION_SEED,
) -> list[dict[str, Any]]:
    """Guided fixed-length autoregressive sampling via ``guided_decoding.GuidedDecoder``.

    Supports both single-AA and multi-residue BPE/SentencePiece tokenizers (e.g. ProLLaMA),
    and both cached (``use_cache=True``) and uncached (``use_cache=False``, e.g. RITA-XL)
    forward loops.
    """
    bias = build_phase_space_bias(family_model, arm_spec, permutation_seed=permutation_seed)
    layout = _build_layout_for_guided_arm(arm, family_model.length)
    if logit_width is None:
        from ..core.arms import output_logit_width

        w_info = output_logit_width(arm)
        logit_width = int(w_info["size"] if isinstance(w_info, Mapping) else w_info)
    table = _token_span_table_safe(arm, layout, logit_width=int(logit_width))
    decoder = gd.GuidedDecoder(
        bias=bias, layout=layout, table=table, strength=arm_spec.strength
    )
    prompt_ids = gd._prompt_ids(arm, decoder)
    device = str(arm.device)
    records: list[dict[str, Any]] = []

    for start in range(0, count, max(1, int(batch_size))):
        chunk_n = min(int(batch_size), count - start)
        seeds = [int(stream_seed) + start + i for i in range(chunk_n)]
        gens = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
        offsets = [0] * chunk_n
        prefixes: list[list[int]] = [[] for _ in range(chunk_n)]
        token_ids: list[list[int]] = [[] for _ in range(chunk_n)]
        entropies: list[list[float]] = [[] for _ in range(chunk_n)]
        applied = [0.0] * chunk_n
        done = [False] * chunk_n

        prompt = torch.tensor([prompt_ids] * chunk_n, dtype=torch.long, device=device)
        cur_full = prompt
        past = None
        if use_cache:
            step_out = arm.model(input_ids=prompt, use_cache=True)
            past = step_out.past_key_values
            logits = step_out.logits[:, -1]
        else:
            step_out = arm.model(input_ids=cur_full, use_cache=False)
            logits = step_out.logits[:, -1]

        for _ in range(layout.characters):
            drawn = torch.zeros((chunk_n, 1), dtype=torch.long, device=device)
            for row in range(chunk_n):
                if done[row]:
                    drawn[row, 0] = token_ids[row][-1]
                    continue
                prefix_codes = np.asarray(prefixes[row], dtype=np.int64)
                scores, _mask, tok_bias = gd._row_scores(
                    decoder, logits[row], offsets[row], prefix_codes
                )
                probs = torch.softmax(scores, dim=-1).cpu()
                token = int(torch.multinomial(probs, 1, generator=gens[row]).item())
                pos = int(np.searchsorted(table.ids, token))
                if pos >= table.size or int(table.ids[pos]) != token:
                    raise RuntimeError(f"{model_id}: drew non-candidate token {token}")
                entropies[row].append(_entropy_from_logits_1d(scores))
                applied[row] += float(tok_bias[pos])
                token_ids[row].append(token)
                span = table.residues[pos, : int(table.residue_length[pos])]
                prefixes[row].extend(int(code) for code in span)
                offsets[row] += int(table.character_length[pos])
                drawn[row, 0] = token
                if offsets[row] == layout.characters:
                    done[row] = True
                elif offsets[row] > layout.characters:
                    raise RuntimeError(f"{model_id}: drawn token exceeded layout characters")
            if all(done):
                break
            if use_cache:
                step_out = arm.model(input_ids=drawn, past_key_values=past, use_cache=True)
                past = step_out.past_key_values
                logits = step_out.logits[:, -1]
            else:
                cur_full = torch.cat([cur_full, drawn], dim=1)
                step_out = arm.model(input_ids=cur_full, use_cache=False)
                logits = step_out.logits[:, -1]

        if not all(done):
            raise RuntimeError(
                f"{model_id}: {done.count(False)} sequences failed to reach length {family_model.length}"
            )
        for row in range(chunk_n):
            seq = "".join(AA20[code] for code in prefixes[row])
            expected, f_sum, c_sum = _verify_applied_bias(
                bias, arm_spec.strength, seq, applied[row], label=f"{model_id}/{arm_spec.arm_key}"
            )
            records.append(
                _make_generated_record(
                    model_id=model_id,
                    domain=family_model.domain,
                    arm_spec=arm_spec,
                    seed=seeds[row],
                    sequence_index=start + row,
                    sequence=seq,
                    applied_bias=expected,
                    fields_score=f_sum if bias.has_fields else None,
                    coupling_score=c_sum if bias.has_coupling else None,
                    mean_step_entropy=float(np.mean(entropies[row])),
                    sampler=f"guided_decoding_cache_{int(use_cache)}",
                    token_steps=len(token_ids[row]),
                )
            )
    return records


def _causal_aa20_step_bias(
    bias: gd.ArmBias,
    strength: gd.GuidanceStrength,
    pos: int,
    prefix_codes: np.ndarray,
) -> np.ndarray:
    """Exact 20-AA float64 logit bias at residue ``pos`` given ``prefix_codes`` of length ``pos``."""
    out = np.zeros(20, dtype=np.float64)
    if bias.u is not None and strength.fields != 0.0:
        out += float(strength.fields) * bias.u[pos]
    if bias.C is not None and strength.coupling != 0.0 and pos > 0:
        out += float(strength.coupling) * bias.C[pos, np.arange(pos), :, prefix_codes].sum(axis=0)
    return out


def _supports_kwarg(model: Any, kwarg: str) -> bool:
    try:
        return kwarg in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


@torch.no_grad()
def sample_guided_general_plm_causal(
    *,
    adapter: Any,
    model_id: str,
    family_model: FamilyModel,
    arm_spec: PhaseSpaceArmSpec,
    count: int,
    batch_size: int,
    stream_seed: int,
    permutation_seed: int = PERMUTATION_SEED,
) -> list[dict[str, Any]]:
    """Guided fixed-length AA20 causal sampling for ProtGPT3 and Dayhoff (Jamba) adapters."""
    bias = build_phase_space_bias(family_model, arm_spec, permutation_seed=permutation_seed)
    L = family_model.length
    device = str(adapter.device)
    prompt_ids = list(adapter.bos_prefix_ids)
    if not prompt_ids:
        raise ValueError(f"{model_id}: adapter.bos_prefix_ids is empty")
    aa20_ids = [int(tid) for tid in adapter.aa20_ids]
    if len(aa20_ids) != 20:
        raise ValueError(f"{model_id}: expected 20 AA token ids, got {len(aa20_ids)}")
    aa20_t = torch.tensor(aa20_ids, dtype=torch.long, device=device)
    supports_cpos = _supports_kwarg(adapter.model, "cache_position")
    records: list[dict[str, Any]] = []

    with adapter._precision_ctx():
        for start in range(0, count, max(1, int(batch_size))):
            chunk_n = min(int(batch_size), count - start)
            seeds = [int(stream_seed) + start + i for i in range(chunk_n)]
            gens = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
            cur = torch.tensor([prompt_ids] * chunk_n, dtype=torch.long, device=device)
            past = adapter._init_causal_cache(batch_size=chunk_n)
            prefilled = False
            seq_pos = 0
            prefixes = np.empty((chunk_n, L), dtype=np.int64)
            entropies = np.zeros((chunk_n, L), dtype=np.float64)
            applied = np.zeros(chunk_n, dtype=np.float64)

            for pos in range(L):
                inp = cur if not prefilled else cur[:, -1:]
                fwd_kwargs: dict[str, Any] = {
                    "input_ids": inp,
                    "past_key_values": past,
                    "use_cache": True,
                }
                if supports_cpos:
                    fwd_kwargs["cache_position"] = torch.arange(
                        seq_pos, seq_pos + inp.shape[1], dtype=torch.long, device=device
                    )
                out = adapter.model(**fwd_kwargs)
                past = out.past_key_values
                if past is not None:
                    prefilled = True
                    seq_pos += int(inp.shape[1])
                else:
                    prefilled = False
                    seq_pos = 0

                aa_logits = out.logits[:, -1, :].index_select(-1, aa20_t).float()  # (B, 20)
                bias_mat = np.zeros((chunk_n, 20), dtype=np.float64)
                for row in range(chunk_n):
                    bias_mat[row] = _causal_aa20_step_bias(
                        bias, arm_spec.strength, pos, prefixes[row, :pos]
                    )
                scores = aa_logits + torch.as_tensor(
                    bias_mat.astype(np.float32), dtype=torch.float32, device=device
                )
                probs = torch.softmax(scores, dim=-1).cpu()
                next_toks: list[int] = []
                for row in range(chunk_n):
                    aa_idx = int(torch.multinomial(probs[row], 1, generator=gens[row]).item())
                    prefixes[row, pos] = aa_idx
                    applied[row] += float(bias_mat[row, aa_idx])
                    entropies[row, pos] = _entropy_from_logits_1d(scores[row])
                    next_toks.append(aa20_ids[aa_idx])
                if pos + 1 < L:
                    nxt = torch.tensor(next_toks, dtype=torch.long, device=device).unsqueeze(1)
                    cur = torch.cat([cur, nxt], dim=1)

            for row in range(chunk_n):
                seq = "".join(AA20[int(c)] for c in prefixes[row])
                expected, f_sum, c_sum = _verify_applied_bias(
                    bias,
                    arm_spec.strength,
                    seq,
                    float(applied[row]),
                    label=f"{model_id}/{arm_spec.arm_key}",
                )
                records.append(
                    _make_generated_record(
                        model_id=model_id,
                        domain=family_model.domain,
                        arm_spec=arm_spec,
                        seed=seeds[row],
                        sequence_index=start + row,
                        sequence=seq,
                        applied_bias=expected,
                        fields_score=f_sum if bias.has_fields else None,
                        coupling_score=c_sum if bias.has_coupling else None,
                        mean_step_entropy=float(entropies[row].mean()),
                        sampler="general_plm_causal_guided",
                        token_steps=L,
                    )
                )
    return records


def _progen3_prompt_and_aa20_ids(pg_handle: Any) -> tuple[list[int], list[int]]:
    """Extract ProGen3 ``[<bos>, '1']`` prompt ids and AA20 token ids."""
    tok = pg_handle.tokenizer
    bos_id = int(pg_handle.config.bos_token_id)
    if hasattr(tok, "token_to_id"):
        dir_id = int(tok.token_to_id("1"))
        aa_ids = [int(tok.token_to_id(aa)) for aa in AA20]
    else:
        dir_id = int(tok.convert_tokens_to_ids("1"))
        aa_ids = [int(tok.convert_tokens_to_ids(aa)) for aa in AA20]
    if any(tid < 0 for tid in [bos_id, dir_id, *aa_ids]):
        raise ValueError("ProGen3 tokenizer missing <bos>, '1', or canonical AA20 token")
    return [bos_id, dir_id], aa_ids


@torch.no_grad()
def sample_guided_progen3(
    *,
    pg_handle: Any,
    model_id: str,
    family_model: FamilyModel,
    arm_spec: PhaseSpaceArmSpec,
    count: int,
    stream_seed: int,
    batch_size: int = DEFAULT_PROGEN3_BATCH_SIZE,
    permutation_seed: int = PERMUTATION_SEED,
) -> list[dict[str, Any]]:
    """Guided fixed-length AA20 sampling for ProGen3 (MegaBlocks MoE, BF16, KV-cached).

    Defaults to ``batch_size=1`` to honour the documented ProGen3 MegaBlocks MoE
    singleton-only invariance contract (``src/capability/decoding/progen3_decoding.py``).
    """
    bias = build_phase_space_bias(family_model, arm_spec, permutation_seed=permutation_seed)
    L = family_model.length
    device = str(pg_handle.device)
    prompt_ids, aa20_ids = _progen3_prompt_and_aa20_ids(pg_handle)
    aa20_t = torch.tensor(aa20_ids, dtype=torch.long, device=device)
    bsz = max(1, int(batch_size))
    records: list[dict[str, Any]] = []

    for start in range(0, count, bsz):
        chunk_n = min(bsz, count - start)
        seeds = [int(stream_seed) + start + i for i in range(chunk_n)]
        gens = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
        cur = torch.tensor([prompt_ids] * chunk_n, dtype=torch.long, device=device)
        past = None
        offset = 0
        prefixes = np.empty((chunk_n, L), dtype=np.int64)
        entropies = np.zeros((chunk_n, L), dtype=np.float64)
        applied = np.zeros(chunk_n, dtype=np.float64)

        for pos in range(L):
            inp = cur if past is None else cur[:, -1:]
            width = int(inp.shape[1])
            position_ids = (
                torch.arange(offset, offset + width, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(chunk_n, width)
            )
            sequence_ids = torch.zeros((chunk_n, width), dtype=torch.long, device=device)
            out = pg_handle.model(
                input_ids=inp,
                position_ids=position_ids,
                sequence_ids=sequence_ids,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            offset += width

            aa_logits = out.logits[:, -1, :].index_select(-1, aa20_t).float()
            bias_mat = np.zeros((chunk_n, 20), dtype=np.float64)
            for row in range(chunk_n):
                bias_mat[row] = _causal_aa20_step_bias(
                    bias, arm_spec.strength, pos, prefixes[row, :pos]
                )
            scores = aa_logits + torch.as_tensor(
                bias_mat.astype(np.float32), dtype=torch.float32, device=device
            )
            probs = torch.softmax(scores, dim=-1).cpu()
            next_toks: list[int] = []
            for row in range(chunk_n):
                aa_idx = int(torch.multinomial(probs[row], 1, generator=gens[row]).item())
                prefixes[row, pos] = aa_idx
                applied[row] += float(bias_mat[row, aa_idx])
                entropies[row, pos] = _entropy_from_logits_1d(scores[row])
                next_toks.append(aa20_ids[aa_idx])
            if pos + 1 < L:
                nxt = torch.tensor(next_toks, dtype=torch.long, device=device).unsqueeze(1)
                cur = torch.cat([cur, nxt], dim=1)

        for row in range(chunk_n):
            seq = "".join(AA20[int(c)] for c in prefixes[row])
            expected, f_sum, c_sum = _verify_applied_bias(
                bias,
                arm_spec.strength,
                seq,
                float(applied[row]),
                label=f"{model_id}/{arm_spec.arm_key}",
            )
            records.append(
                _make_generated_record(
                    model_id=model_id,
                    domain=family_model.domain,
                    arm_spec=arm_spec,
                    seed=seeds[row],
                    sequence_index=start + row,
                    sequence=seq,
                    applied_bias=expected,
                    fields_score=f_sum if bias.has_fields else None,
                    coupling_score=c_sum if bias.has_coupling else None,
                    mean_step_entropy=float(entropies[row].mean()),
                    sampler=f"progen3_causal_guided_bs{bsz}",
                    token_steps=L,
                )
            )
    return records


def _diffusion_position_bias(
    bias: gd.ArmBias,
    strength: gd.GuidanceStrength,
    pos: int,
    committed_positions: Sequence[int],
    committed_codes: Sequence[int],
) -> np.ndarray:
    """Exact 20-AA logit bias at pending site ``pos`` given committed sites ``s in committed``."""
    out = np.zeros(20, dtype=np.float64)
    if bias.u is not None and strength.fields != 0.0:
        out += float(strength.fields) * bias.u[pos]
    if bias.C is not None and strength.coupling != 0.0 and committed_positions:
        s_idx = np.asarray(committed_positions, dtype=np.int64)
        a_idx = np.asarray(committed_codes, dtype=np.int64)
        out += float(strength.coupling) * bias.C[pos, s_idx, :, a_idx].sum(axis=0)
    return out


@torch.no_grad()
def sample_guided_dplm(
    *,
    adapter: Any,
    model_id: str,
    family_model: FamilyModel,
    arm_spec: PhaseSpaceArmSpec,
    count: int,
    batch_size: int,
    stream_seed: int,
    permutation_seed: int = PERMUTATION_SEED,
) -> list[dict[str, Any]]:
    """Guided absorbing-state discrete diffusion (DPLM) with ``uniform_random`` 1-by-1 unmasking.

    Each sequence unmasks positions along an independent uniform random permutation of
    ``0..L-1``, adding ``lambda_fields * u[pos, aa] + lambda_coupling * sum_{s in committed}
    C[pos, s, aa, aa_s]`` to the AA20 logits before sampling. Because 1 position commits
    per step and ``C`` is symmetric, every unordered pair ``(i, j)`` enters exactly once,
    preserving the exact ``sequence_component_sums`` accounting invariant.
    """
    bias = build_phase_space_bias(family_model, arm_spec, permutation_seed=permutation_seed)
    L = family_model.length
    device = str(adapter.device)
    template_ids, offset = adapter._encode_sequence("A" * L)
    base_row = list(template_ids)
    for pos in range(L):
        base_row[offset + pos] = int(adapter.mask_id)
    aa20_ids = [int(tid) for tid in adapter.aa20_ids]
    aa20_t = torch.tensor(aa20_ids, dtype=torch.long, device=device)
    records: list[dict[str, Any]] = []

    with adapter._precision_ctx():
        for start in range(0, count, max(1, int(batch_size))):
            chunk_n = min(int(batch_size), count - start)
            seeds = [int(stream_seed) + start + i for i in range(chunk_n)]
            gens = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
            orders = [torch.randperm(L, generator=gens[row]).tolist() for row in range(chunk_n)]
            ids = torch.tensor([base_row] * chunk_n, dtype=torch.long, device=device)

            committed_pos: list[list[int]] = [[] for _ in range(chunk_n)]
            committed_aa: list[list[int]] = [[] for _ in range(chunk_n)]
            final_codes = np.full((chunk_n, L), -1, dtype=np.int64)
            entropies = np.zeros((chunk_n, L), dtype=np.float64)
            applied = np.zeros(chunk_n, dtype=np.float64)

            for step in range(L):
                logits = adapter._forward_logits(ids)  # (B, T, V)
                step_pos = [orders[row][step] for row in range(chunk_n)]
                row_idx = torch.arange(chunk_n, dtype=torch.long, device=device)
                tok_idx = torch.tensor([offset + p for p in step_pos], dtype=torch.long, device=device)
                aa_logits = logits[row_idx, tok_idx].index_select(-1, aa20_t).float()

                bias_mat = np.zeros((chunk_n, 20), dtype=np.float64)
                for row in range(chunk_n):
                    bias_mat[row] = _diffusion_position_bias(
                        bias,
                        arm_spec.strength,
                        step_pos[row],
                        committed_pos[row],
                        committed_aa[row],
                    )
                scores = aa_logits + torch.as_tensor(
                    bias_mat.astype(np.float32), dtype=torch.float32, device=device
                )
                probs = torch.softmax(scores, dim=-1).cpu()
                for row in range(chunk_n):
                    pos = step_pos[row]
                    aa_idx = int(torch.multinomial(probs[row], 1, generator=gens[row]).item())
                    ids[row, offset + pos] = aa20_ids[aa_idx]
                    committed_pos[row].append(pos)
                    committed_aa[row].append(aa_idx)
                    final_codes[row, pos] = aa_idx
                    applied[row] += float(bias_mat[row, aa_idx])
                    entropies[row, step] = _entropy_from_logits_1d(scores[row])

            for row in range(chunk_n):
                seq = "".join(AA20[int(c)] for c in final_codes[row])
                expected, f_sum, c_sum = _verify_applied_bias(
                    bias,
                    arm_spec.strength,
                    seq,
                    float(applied[row]),
                    label=f"{model_id}/{arm_spec.arm_key}",
                )
                records.append(
                    _make_generated_record(
                        model_id=model_id,
                        domain=family_model.domain,
                        arm_spec=arm_spec,
                        seed=seeds[row],
                        sequence_index=start + row,
                        sequence=seq,
                        applied_bias=expected,
                        fields_score=f_sum if bias.has_fields else None,
                        coupling_score=c_sum if bias.has_coupling else None,
                        mean_step_entropy=float(entropies[row].mean()),
                        sampler="dplm_uniform_random_guided",
                        token_steps=L,
                    )
                )
    return records


@torch.no_grad()
def sample_guided_esm3(
    *,
    handle: Any,
    model_id: str,
    family_model: FamilyModel,
    arm_spec: PhaseSpaceArmSpec,
    count: int,
    batch_size: int,
    stream_seed: int,
    num_steps: int | None = None,
    permutation_seed: int = PERMUTATION_SEED,
    forward_fn: Any = None,
) -> list[dict[str, Any]]:
    """Guided cosine-schedule unmasking for ESM3-1.4B on exact domain length ``L``.

    Defaults to ``num_steps = L`` (1-by-1 cosine max-confidence unmasking) so every
    pairwise coupling term ``C[pos, s, aa, aa_s]`` over committed sites ``s`` enters
    exactly once and obeys the ``sequence_component_sums`` invariant.
    """
    from ..esm3_ext import esm3_adapter as e3

    bias = build_phase_space_bias(family_model, arm_spec, permutation_seed=permutation_seed)
    L = family_model.length
    steps = L if num_steps is None else max(1, min(int(num_steps), L))
    step_counts = e3._cosine_unmask_steps(L, steps)
    exact_one_per_step = all(n == 1 for n in step_counts)
    device = handle.device
    aa20_ids = [int(tid) for tid in handle.aa20_token_ids]
    aa20_t = torch.tensor(aa20_ids, dtype=torch.long, device=device)
    fwd = e3.forward_tracks if forward_fn is None else forward_fn
    records: list[dict[str, Any]] = []

    for start in range(0, count, max(1, int(batch_size))):
        chunk_n = min(int(batch_size), count - start)
        seeds = [int(stream_seed) + start + i for i in range(chunk_n)]
        gens = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]

        cur = torch.full((chunk_n, L + 2), int(handle.mask_id), dtype=torch.long, device=device)
        cur[:, 0] = int(handle.bos_id)
        cur[:, -1] = int(handle.eos_id)

        pending: list[list[int]] = [list(range(L)) for _ in range(chunk_n)]
        committed_pos: list[list[int]] = [[] for _ in range(chunk_n)]
        committed_aa: list[list[int]] = [[] for _ in range(chunk_n)]
        final_codes = np.full((chunk_n, L), -1, dtype=np.int64)
        entropies: list[list[float]] = [[] for _ in range(chunk_n)]
        applied = np.zeros(chunk_n, dtype=np.float64)

        for n_commit in step_counts:
            if not pending[0]:
                break
            seq_logits = fwd(handle, sequence_tokens=cur).sequence_logits.float()  # (B, L+2, V)
            for row in range(chunk_n):
                pend = pending[row]
                if not pend:
                    continue
                pos_t = torch.tensor([p + 1 for p in pend], dtype=torch.long, device=device)
                aa_logits = seq_logits[row].index_select(0, pos_t).index_select(-1, aa20_t)  # (P, 20)
                bias_mat = np.zeros((len(pend), 20), dtype=np.float64)
                for idx_p, pos in enumerate(pend):
                    bias_mat[idx_p] = _diffusion_position_bias(
                        bias,
                        arm_spec.strength,
                        pos,
                        committed_pos[row],
                        committed_aa[row],
                    )
                scores = aa_logits + torch.as_tensor(
                    bias_mat.astype(np.float32), dtype=torch.float32, device=device
                )
                probs = torch.softmax(scores, dim=-1).cpu()
                sampled = torch.multinomial(probs, num_samples=1, generator=gens[row]).squeeze(-1)
                conf = probs.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
                k_take = min(int(n_commit), len(pend))
                top_local = torch.topk(conf, k=k_take, largest=True).indices.tolist()
                chosen = [(pend[i], int(sampled[i]), i) for i in top_local]
                for i in sorted(top_local, reverse=True):
                    pend.pop(i)
                for pos, aa_idx, local_i in chosen:
                    cur[row, pos + 1] = aa20_ids[aa_idx]
                    committed_pos[row].append(pos)
                    committed_aa[row].append(aa_idx)
                    final_codes[row, pos] = aa_idx
                    applied[row] += float(bias_mat[local_i, aa_idx])
                    entropies[row].append(_entropy_from_logits_1d(scores[local_i]))

        for row in range(chunk_n):
            seq = "".join(AA20[int(c)] for c in final_codes[row])
            fields_sum, coupling_sum = gd.sequence_component_sums(bias, seq)
            expected = float(
                arm_spec.strength.fields * fields_sum
                + arm_spec.strength.coupling * coupling_sum
            )
            if exact_one_per_step:
                _verify_applied_bias(
                    bias,
                    arm_spec.strength,
                    seq,
                    float(applied[row]),
                    label=f"{model_id}/{arm_spec.arm_key}",
                )
            records.append(
                _make_generated_record(
                    model_id=model_id,
                    domain=family_model.domain,
                    arm_spec=arm_spec,
                    seed=seeds[row],
                    sequence_index=start + row,
                    sequence=seq,
                    applied_bias=expected if exact_one_per_step else float(applied[row]),
                    fields_score=float(fields_sum) if bias.has_fields else None,
                    coupling_score=float(coupling_sum) if bias.has_coupling else None,
                    mean_step_entropy=float(np.mean(entropies[row])),
                    sampler="esm3_cosine_guided",
                    token_steps=len(step_counts),
                )
            )
    return records


# --------------------------------------------------------------------------- #
# Runtime Model Loaders (H200 GPU generation only)
# --------------------------------------------------------------------------- #


def _resolve_progen3_3b_dir(model_root: Path | str | None = None) -> Path:
    candidates: list[Path] = []
    if model_root is not None and str(model_root).strip():
        candidates.append(Path(model_root) / "progen3-3b")
        candidates.append(Path(model_root) / "ProGen3-3B")
    for var in ("MODEL_ROOT", "GENERAL_PLM_MODEL_ROOT", "MODEL_BASE_DIR"):
        v = os.environ.get(var, "").strip()
        if v:
            candidates.append(Path(v) / "progen3-3b")
    try:
        from ..core.arms import arm_spec

        candidates.append(Path(arm_spec("progen3-3b").path))
    except Exception:
        pass
    candidates.extend(
        [
            Path("/gpfs/jiaotongdamoxing/zhk_zip/models/progen3-3b"),
            Path("/Data/public/models_R2/progen3-3b"),
        ]
    )
    for cand in candidates:
        if cand.is_dir() and (cand / "config.json").is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"progen3-3b checkpoint not found; searched {[str(c) for c in candidates]}"
    )


def load_phase_space_model(
    model_id: str,
    *,
    device: str = "cuda:0",
    dtype_override: str | None = None,
    model_root: Path | str | None = None,
    esm3_model_dir: Path | str | None = None,
) -> tuple[str, Any, str]:
    """Load one of the 19 new H200 models, returning ``(backend, handle, dtype_used)``."""
    spec = PHASE_SPACE_MODEL_MAP[model_id]
    dtype = dtype_override or spec.default_dtype

    if spec.backend == "guided_decoding":
        import importlib.util

        stage46_path = REPO_ROOT / "scripts/capability/stages/context_homologue.py"
        sp = importlib.util.spec_from_file_location("stage46_loader", stage46_path)
        if sp is None or sp.loader is None:
            raise RuntimeError(f"Could not load stage46 helper from {stage46_path}")
        stage46 = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(stage46)
        arm = stage46.load_scorable_arm(model_id, device=device, dtype=dtype)
        return spec.backend, arm, dtype

    if spec.backend in ("general_plm_causal", "dplm_diffusion"):
        from ..general_plm.adapters import load_general_plm

        adapter = load_general_plm(
            model_id, device=device, dtype=dtype, model_root=model_root
        )
        return spec.backend, adapter, adapter.dtype

    if spec.backend == "progen3_causal":
        if dtype not in ("bfloat16", "float16"):
            dtype = "bfloat16"
        if model_id == "progen3-3b":
            from ..models.progen3 import load_progen3

            ckpt = _resolve_progen3_3b_dir(model_root=model_root)
            torch_dt = torch.bfloat16 if dtype == "bfloat16" else torch.float16
            pg_handle = load_progen3(ckpt, device=device, dtype=torch_dt)
            return spec.backend, pg_handle, dtype
        from ..general_plm.adapters import load_general_plm

        adapter = load_general_plm(
            model_id, device=device, dtype=dtype, model_root=model_root
        )
        return spec.backend, adapter.handle, dtype

    if spec.backend == "esm3_diffusion":
        from ..esm3_ext.esm3_adapter import load_esm3_offline

        resolved_dir = esm3_model_dir
        if resolved_dir is None and model_root is not None:
            cand = Path(model_root) / "esm3-sm-open-v1"
            if cand.is_dir():
                resolved_dir = cand
        handle = load_esm3_offline(model_dir=resolved_dir, device=device, dtype=dtype)
        return spec.backend, handle, dtype

    raise ValueError(f"{model_id}: backend {spec.backend!r} cannot be loaded for GPU generation")


def generate_cell_records(
    *,
    model_id: str,
    backend: str,
    loaded: Any,
    family_model: FamilyModel,
    arm_spec: PhaseSpaceArmSpec,
    count: int = DEFAULT_POOL_SIZE,
    batch_size: int | None = None,
    progen3_batch_size: int = DEFAULT_PROGEN3_BATCH_SIZE,
    base_seed: int = DEFAULT_BASE_SEED,
    permutation_seed: int = PERMUTATION_SEED,
) -> list[dict[str, Any]]:
    """Dispatch guided generation for one ``(model, domain, arm)`` cell."""
    spec = PHASE_SPACE_MODEL_MAP[model_id]
    s_seed = cell_stream_seed(model_id, family_model.domain, arm_spec.arm_key, base_seed=base_seed)
    eff_bs = spec.default_batch_size if batch_size is None else int(batch_size)

    if backend == "guided_decoding":
        return sample_guided_causal_decoding(
            arm=loaded,
            model_id=model_id,
            family_model=family_model,
            arm_spec=arm_spec,
            count=count,
            batch_size=min(eff_bs, spec.default_batch_size) if not spec.use_cache else eff_bs,
            stream_seed=s_seed,
            use_cache=spec.use_cache,
            permutation_seed=permutation_seed,
        )
    if backend == "general_plm_causal":
        return sample_guided_general_plm_causal(
            adapter=loaded,
            model_id=model_id,
            family_model=family_model,
            arm_spec=arm_spec,
            count=count,
            batch_size=eff_bs,
            stream_seed=s_seed,
            permutation_seed=permutation_seed,
        )
    if backend == "progen3_causal":
        return sample_guided_progen3(
            pg_handle=loaded,
            model_id=model_id,
            family_model=family_model,
            arm_spec=arm_spec,
            count=count,
            batch_size=int(progen3_batch_size),
            stream_seed=s_seed,
            permutation_seed=permutation_seed,
        )
    if backend == "dplm_diffusion":
        return sample_guided_dplm(
            adapter=loaded,
            model_id=model_id,
            family_model=family_model,
            arm_spec=arm_spec,
            count=count,
            batch_size=eff_bs,
            stream_seed=s_seed,
            permutation_seed=permutation_seed,
        )
    if backend == "esm3_diffusion":
        return sample_guided_esm3(
            handle=loaded,
            model_id=model_id,
            family_model=family_model,
            arm_spec=arm_spec,
            count=count,
            batch_size=eff_bs,
            stream_seed=s_seed,
            permutation_seed=permutation_seed,
        )
    raise ValueError(f"{model_id}: unsupported generation backend {backend!r}")


# --------------------------------------------------------------------------- #
# Cached E20 & ProGen3 Robustness Loaders + <1e-12 Reproduction Gates
# --------------------------------------------------------------------------- #


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _bca_interval(values: Sequence[float]) -> tuple[list[float], bool]:
    """95% BCa bootstrap CI over domains matching ``analyse_e20_confirmatory.bca_interval``."""
    v = np.asarray(values, dtype=np.float64)
    if float(np.ptp(v)) == 0.0:
        return [float(v[0]), float(v[0])], True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = stats.bootstrap(
            (v,),
            np.mean,
            method="BCa",
            confidence_level=CONFIDENCE,
            n_resamples=N_BOOTSTRAP,
            random_state=np.random.default_rng(BOOTSTRAP_SEED),
        )
    lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise RuntimeError("BCa bootstrap interval is non-finite")
    return [lo, hi], False


def _sign_flip_p(diffs: Sequence[float]) -> float:
    d = np.asarray(diffs, dtype=np.float64)
    signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * len(d))).reshape(len(d), -1).T
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= abs(float(d.mean())) - EPS))


def _estimate_domain_contrast(
    per_domain: Mapping[str, Mapping[str, float]],
    missing: Mapping[str, str],
) -> dict[str, Any]:
    """Paired domain-level contrast with 9,999-draw BCa CI and exact sign-flip p-value."""
    ordered = {d: dict(per_domain[d]) for d in sorted(per_domain)}
    diffs = np.asarray([v["diff"] for v in ordered.values()], dtype=np.float64)
    n = int(diffs.size)
    floor = bootstrap_unit_floor(n, minimum_units=MINIMUM_BOOTSTRAP_UNITS)
    out: dict[str, Any] = {
        "n_domains": n,
        "per_domain": ordered,
        "missing_domains": dict(sorted(missing.items())),
        "mean": float(diffs.mean()) if n else None,
        "mean_identity_diff": (
            float(np.mean([v["identity_diff"] for v in ordered.values()])) if n else None
        ),
        "mean_msa_identity_diff": (
            float(np.mean([v["msa_identity_diff"] for v in ordered.values()])) if n else None
        ),
        "favouring": {
            "positive": int((diffs > 0).sum()),
            "negative": int((diffs < 0).sum()),
            "tied": int((diffs == 0).sum()),
        },
        "interval95_bca": None,
        "interval_zero_variance": None,
        "interval_coverage_note": None,
        "p_ttest": None,
        "p_signflip": None,
        "status": "unestimable",
        "reason": None,
    }
    if floor["degenerate"]:
        reason = f"{n} domains have a value, below the {MINIMUM_BOOTSTRAP_UNITS}-domain floor"
        if missing:
            reason += "; missing: " + "; ".join(f"{d} ({why})" for d, why in sorted(missing.items()))
        out["reason"] = reason
        return out
    ci, zero_var = _bca_interval(diffs)
    sd = float(diffs.std(ddof=1))
    p_t = (
        float(stats.ttest_1samp(diffs, 0.0).pvalue)
        if sd > 0.0
        else (1.0 if float(diffs.mean()) == 0.0 else 0.0)
    )
    out.update(
        status="estimated",
        interval95_bca=ci,
        interval_zero_variance=zero_var,
        interval_coverage_note=BCA_COVERAGE_NOTE,
        p_ttest=p_t,
        p_signflip=_sign_flip_p(diffs),
    )
    return out


def verify_e20_confirmatory_reproduction(
    *,
    per_sequence_path: Path | str = DEFAULT_E20_PER_SEQUENCE,
    confirmatory_json_path: Path | str = DEFAULT_E20_CONFIRMATORY_JSON,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Verify ``< 1e-12`` exact reproduction of E20 ``confirmatory.json`` from ``per_sequence.jsonl``."""
    rows = _read_jsonl(Path(per_sequence_path))
    ref = json.loads(Path(confirmatory_json_path).read_text(encoding="utf-8"))

    by_cell: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cell[(r["model"], r["domain"], r["arm"], r["lambda_fields"], r["lambda_coupling"])].append(r)

    cell_summaries: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, grp in by_cell.items():
        ident = np.asarray([x["identity_to_wild_type"] for x in grp], dtype=np.float64)
        msa = np.asarray([x["max_msa_identity"] for x in grp], dtype=np.float64)
        tm = np.asarray([x["tm_score"] for x in grp], dtype=np.float64)
        adm = ident < ADMISSIBLE_WT_IDENTITY_PCT
        n_adm = int(adm.sum())
        cell_summaries[key] = {
            "n": len(grp),
            "n_admissible": n_adm,
            "feasibility": n_adm / len(grp),
            "tm": float(tm[adm].mean()) if n_adm >= MIN_ADMISSIBLE_PER_CELL else None,
            "identity_admissible": float(ident[adm].mean()) if n_adm else None,
            "msa_identity_admissible": float(msa[adm].mean()) if n_adm else None,
            "identity_pool": float(ident.mean()),
            "msa_identity_pool": float(msa.mean()),
            "tm_all_candidates": float(tm.mean()),
        }

    max_abs_diff = 0.0
    n_checked = 0

    def _cmp(a: float | None, b: float | None, label: str) -> None:
        nonlocal max_abs_diff, n_checked
        if a is None or b is None:
            if a is not b:
                raise AssertionError(f"E20 reproduction null mismatch at {label}: {a} vs {b}")
            return
        diff = abs(float(a) - float(b))
        n_checked += 1
        if diff > max_abs_diff:
            max_abs_diff = diff
        if diff > tolerance:
            raise AssertionError(
                f"E20 reproduction exceeded tolerance {tolerance} at {label}: "
                f"got {a!r}, ref {b!r}, diff={diff:.3e}"
            )

    # 1. Check all cells in confirmatory.json
    for ref_cell in ref["cells"]:
        key = (
            ref_cell["model"],
            ref_cell["domain"],
            ref_cell["arm"],
            ref_cell["lambda_fields"],
            ref_cell["lambda_coupling"],
        )
        got = cell_summaries[key]
        for field in (
            "feasibility",
            "tm",
            "identity_admissible",
            "msa_identity_admissible",
            "identity_pool",
            "msa_identity_pool",
            "tm_all_candidates",
        ):
            _cmp(got[field], ref_cell[field], f"cell:{key}:{field}")

    # 2. Check frozen-dose contrasts (primary_tm, coprimary_feasibility, reference_arms_descriptive)
    frozen_doses = ref["frozen"]["doses"]
    models = ("progen2-medium", "protgpt2")
    domains = list(ref["domains"])

    def _get_cell(model: str, dom: str, arm: str) -> dict[str, Any] | None:
        if arm in REFERENCE_ARMS:
            return cell_summaries.get((REFERENCE_ARMS[arm], dom, arm, None, None))
        lf, lc = frozen_doses[arm][model]
        return cell_summaries.get((model, dom, arm, lf, lc))

    def _check_contrast_block(
        ref_block: Mapping[str, Any],
        pairs: Sequence[tuple[str, str, str]],
        metric: str,
        *,
        wrapped: bool,
    ) -> None:
        id_k = "identity_admissible" if metric == "tm" else "identity_pool"
        msa_k = "msa_identity_admissible" if metric == "tm" else "msa_identity_pool"
        for cname, left, right in pairs:
            by_m: dict[str, tuple[dict[str, dict[str, float]], dict[str, str]]] = {}
            for m in models:
                per, miss = {}, {}
                for d in domains:
                    sa = _get_cell(m, d, left)
                    sb = _get_cell(m, d, right)
                    if sa is None or sb is None or sa[metric] is None or sb[metric] is None:
                        miss[d] = "missing_or_below_min_admissible"
                        continue
                    per[d] = {
                        "diff": sa[metric] - sb[metric],
                        "identity_diff": sa[id_k] - sb[id_k],
                        "msa_identity_diff": sa[msa_k] - sb[msa_k],
                    }
                by_m[m] = (per, miss)
                est = _estimate_domain_contrast(per, miss)
                ref_c = ref_block[m]["contrasts"][cname] if wrapped else ref_block[m][cname]
                _cmp(est["mean"], ref_c["mean"], f"{metric}:{m}:{cname}:mean")
                if ref_c["interval95_bca"] is not None:
                    _cmp(est["interval95_bca"][0], ref_c["interval95_bca"][0], f"{metric}:{m}:{cname}:ci_lo")
                    _cmp(est["interval95_bca"][1], ref_c["interval95_bca"][1], f"{metric}:{m}:{cname}:ci_hi")
            # Pooled
            p_per, p_miss = {}, {}
            for d in domains:
                if d in by_m[models[0]][0] and d in by_m[models[1]][0]:
                    p_per[d] = {
                        k: float(np.mean([by_m[m][0][d][k] for m in models]))
                        for k in ("diff", "identity_diff", "msa_identity_diff")
                    }
                else:
                    p_miss[d] = "missing_in_model"
            p_est = _estimate_domain_contrast(p_per, p_miss)
            ref_p = ref_block["pooled"]["contrasts"][cname] if wrapped else ref_block["pooled"][cname]
            _cmp(p_est["mean"], ref_p["mean"], f"{metric}:pooled:{cname}:mean")
            if ref_p["interval95_bca"] is not None:
                _cmp(p_est["interval95_bca"][0], ref_p["interval95_bca"][0], f"{metric}:pooled:{cname}:ci_lo")
                _cmp(p_est["interval95_bca"][1], ref_p["interval95_bca"][1], f"{metric}:pooled:{cname}:ci_hi")

    cell_pairs = (
        ("C-Pr", "C", "Pr"),
        ("F-M", "F", "M"),
        ("C-Cx", "C", "Cx"),
        ("C-Cd", "C", "Cd"),
        ("C-N", "C", "N"),
        ("Pr-N", "Pr", "N"),
    )
    ref_pairs = (("P-N", "P", "N"), ("H-N", "H", "N"))
    _check_contrast_block(ref["primary_tm"], cell_pairs, "tm", wrapped=True)
    _check_contrast_block(ref["coprimary_feasibility"], cell_pairs, "feasibility", wrapped=True)
    _check_contrast_block(ref["reference_arms_descriptive"], ref_pairs, "tm", wrapped=False)

    return {
        "source_per_sequence": str(per_sequence_path),
        "reference_json": str(confirmatory_json_path),
        "n_scalars_checked": n_checked,
        "max_abs_diff": max_abs_diff,
        "tolerance": tolerance,
        "passed": max_abs_diff <= tolerance,
    }


def verify_progen3_robustness_reproduction(
    *,
    per_sequence_path: Path | str = DEFAULT_PG3RB_PER_SEQUENCE,
    robustness_json_path: Path | str = DEFAULT_PG3RB_ROBUSTNESS_JSON,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Verify ``< 1e-12`` exact reproduction of ProGen3 ``robustness.json`` bfloat16 strata."""
    rows = _read_jsonl(Path(per_sequence_path))
    ref = json.loads(Path(robustness_json_path).read_text(encoding="utf-8"))

    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cell[(r["model"], r["domain"], r["arm"])].append(r)

    summaries: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, grp in by_cell.items():
        ident = np.asarray([x["identity_to_wild_type"] for x in grp], dtype=np.float64)
        msa = np.asarray([x["max_msa_identity"] for x in grp], dtype=np.float64)
        tm = np.asarray([x["tm_score"] for x in grp], dtype=np.float64)
        adm = ident < ADMISSIBLE_WT_IDENTITY_PCT
        n_adm = int(adm.sum())
        summaries[key] = {
            "feasibility": n_adm / len(grp),
            "tm": float(tm[adm].mean()) if n_adm >= MIN_ADMISSIBLE_PER_CELL else None,
            "identity_admissible": float(ident[adm].mean()) if n_adm else None,
            "msa_identity_admissible": float(msa[adm].mean()) if n_adm else None,
            "identity_pool": float(ident.mean()),
            "msa_identity_pool": float(msa.mean()),
        }

    max_abs_diff = 0.0
    n_checked = 0

    def _cmp(a: float | None, b: float | None, label: str) -> None:
        nonlocal max_abs_diff, n_checked
        if a is None or b is None:
            if a is not b:
                raise AssertionError(f"ProGen3 robustness null mismatch at {label}: {a} vs {b}")
            return
        diff = abs(float(a) - float(b))
        n_checked += 1
        if diff > max_abs_diff:
            max_abs_diff = diff
        if diff > tolerance:
            raise AssertionError(
                f"ProGen3 robustness reproduction exceeded {tolerance} at {label}: {a} vs {b}"
            )

    pairs = (("C-Pr", "C", "Pr"), ("C-M", "C", "M"), ("C-Cx", "C", "Cx"), ("Pr-N", "Pr", "N"), ("M-N", "M", "N"))
    rungs = ("progen3-112m-bfloat16", "progen3-3b-bfloat16")
    for ep_key, metric in (("primary_tm", "tm"), ("novelty_admissibility_rate", "feasibility")):
        id_k = "identity_admissible" if metric == "tm" else "identity_pool"
        msa_k = "msa_identity_admissible" if metric == "tm" else "msa_identity_pool"
        for cname, left, right in pairs:
            for rung in rungs:
                per, miss = {}, {}
                for d in SEALED_8_DOMAINS:
                    sa = summaries[(rung, d, left)]
                    sb = summaries[(rung, d, right)]
                    if sa[metric] is None or sb[metric] is None:
                        miss[d] = "below_min_admissible"
                        continue
                    per[d] = {
                        "diff": sa[metric] - sb[metric],
                        "identity_diff": sa[id_k] - sb[id_k],
                        "msa_identity_diff": sa[msa_k] - sb[msa_k],
                    }
                est = _estimate_domain_contrast(per, miss)
                ref_c = ref["bounds"][ep_key]["strata"][rung][cname]
                _cmp(est["mean"], ref_c["mean"], f"{ep_key}:{rung}:{cname}:mean")
                if ref_c["interval95_bca"] is not None:
                    _cmp(est["interval95_bca"][0], ref_c["interval95_bca"][0], f"{ep_key}:{rung}:{cname}:ci_lo")
                    _cmp(est["interval95_bca"][1], ref_c["interval95_bca"][1], f"{ep_key}:{rung}:{cname}:ci_hi")

    return {
        "source_per_sequence": str(per_sequence_path),
        "reference_json": str(robustness_json_path),
        "n_scalars_checked": n_checked,
        "max_abs_diff": max_abs_diff,
        "tolerance": tolerance,
        "passed": max_abs_diff <= tolerance,
    }


def _normalize_cached_row(
    raw: Mapping[str, Any],
    *,
    model_id: str,
    arm_key: str,
    family_model: FamilyModel,
    source_tag: str,
) -> dict[str, Any]:
    seq = str(raw["sequence"])
    e_full, e_fields, e_couplings = compute_potts_energies(family_model, seq)
    return {
        "model": model_id,
        "domain": str(raw["domain"]),
        "arm": arm_key,
        "source_arm": str(raw["arm"]),
        "lambda_fields": raw.get("lambda_fields"),
        "lambda_coupling": raw.get("lambda_coupling"),
        "seed": raw.get("seed"),
        "sequence": seq,
        "length": len(seq),
        "tm_score": float(raw["tm_score"]),
        "plddt_mean": float(raw["plddt_mean"]),
        "ptm": float(raw["ptm"]),
        "wt_identity_pct": float(raw["identity_to_wild_type"]),
        "msa_max_identity_pct": float(raw["max_msa_identity"]),
        "potts_energy_full": e_full,
        "potts_energy_fields": e_fields,
        "potts_energy_couplings": e_couplings,
        "shannon_H_bits": compute_shannon_entropy_bits(seq),
        "source": source_tag,
    }


def load_cached_local_cells(
    *,
    family_models: Mapping[str, FamilyModel],
    e20_per_sequence: Path | str = DEFAULT_E20_PER_SEQUENCE,
    pg3rb_per_sequence: Path | str = DEFAULT_PG3RB_PER_SEQUENCE,
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Load the 5 cached local arms (``P``, ``H``, ``progen2-medium``, ``protgpt2``, ``progen3-112m``)."""
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    # Map E20 (model, arm, lf, lc) -> canonical phase-space arm_key.
    e20_map: dict[tuple[str, str, float | None, float | None], tuple[str, str]] = {
        ("potts-joint", "P", None, None): ("potts-joint", "P"),
        ("natural", "H", None, None): ("natural", "H"),
        ("progen2-medium", "N", 0.0, 0.0): ("progen2-medium", "N"),
        ("progen2-medium", "Pr", 0.25, 0.0): ("progen2-medium", "Pr"),
        ("progen2-medium", "Pr", 1.0, 0.0): ("progen2-medium", "Pr_strong"),
        ("progen2-medium", "C", 0.0, 4.0): ("progen2-medium", "C"),
        ("progen2-medium", "Cx", 0.0, 4.0): ("progen2-medium", "Cx"),
        ("protgpt2", "N", 0.0, 0.0): ("protgpt2", "N"),
        ("protgpt2", "Pr", 0.25, 0.0): ("protgpt2", "Pr"),
        ("protgpt2", "Pr", 1.0, 0.0): ("protgpt2", "Pr_strong"),
        # Frozen E20 calibrated coupling dose pair for ProtGPT2 is (0.0, 8.0).
        ("protgpt2", "C", 0.0, 8.0): ("protgpt2", "C"),
        ("protgpt2", "Cx", 0.0, 8.0): ("protgpt2", "Cx"),
    }

    for raw in _read_jsonl(Path(e20_per_sequence)):
        dom = str(raw["domain"])
        if dom not in family_models:
            continue
        k = (str(raw["model"]), str(raw["arm"]), raw.get("lambda_fields"), raw.get("lambda_coupling"))
        mapped = e20_map.get(k)
        if mapped is None:
            continue
        model_id, arm_key = mapped
        cells[(model_id, dom, arm_key)].append(
            _normalize_cached_row(
                raw,
                model_id=model_id,
                arm_key=arm_key,
                family_model=family_models[dom],
                source_tag="cached_e20_confirmatory_20261013",
            )
        )

    # Cached ProGen3-112M from bfloat16 robustness campaign (calibrated doses: N, Pr, C, Cx).
    pg3_arm_keep = {"N": "N", "Pr": "Pr", "C": "C", "Cx": "Cx"}
    for raw in _read_jsonl(Path(pg3rb_per_sequence)):
        dom = str(raw["domain"])
        if dom not in family_models or str(raw["model"]) != "progen3-112m-bfloat16":
            continue
        arm_key = pg3_arm_keep.get(str(raw["arm"]))
        if arm_key is None:
            continue
        cells[("progen3-112m", dom, arm_key)].append(
            _normalize_cached_row(
                raw,
                model_id="progen3-112m",
                arm_key=arm_key,
                family_model=family_models[dom],
                source_tag="cached_progen3_robustness_20261015",
            )
        )

    return dict(cells)


def load_h200_folded_cells(
    folded_root: Path | str,
    *,
    domains: Sequence[str] = SEALED_8_DOMAINS,
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Load newly folded H200 JSONL cell summaries from ``<folded_root>/<model>/<domain>__<arm>.jsonl``."""
    root = Path(folded_root)
    if not root.is_dir():
        return {}
    domain_set = set(domains)
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model_id = model_dir.name
        if model_id not in PHASE_SPACE_MODEL_MAP:
            continue
        for jsonl_path in sorted(model_dir.glob("*.jsonl")):
            rows = _read_jsonl(jsonl_path)
            if not rows:
                continue
            dom = str(rows[0]["domain"])
            arm_key = str(rows[0]["arm"])
            if dom not in domain_set or arm_key not in PHASE_SPACE_ARM_MAP:
                continue
            cells[(model_id, dom, arm_key)] = rows
    return cells


# --------------------------------------------------------------------------- #
# Dual-Stratum Cell & Phase-Space Statistical Aggregation
# --------------------------------------------------------------------------- #


def summarise_phase_space_cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise one ``(model, domain, arm)`` cell across BOTH ``<20%`` novel and whole-pool strata."""
    if not rows:
        raise ValueError("Cannot summarise an empty cell")

    def _arr(key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in rows], dtype=np.float64)

    tm = _arr("tm_score")
    plddt = _arr("plddt_mean")
    ptm = _arr("ptm")
    wt_id = _arr("wt_identity_pct")
    msa_id = _arr("msa_max_identity_pct")
    e_full = _arr("potts_energy_full")
    e_fields = _arr("potts_energy_fields")
    e_couplings = _arr("potts_energy_couplings")
    shannon = _arr("shannon_H_bits")
    seqs = [str(r["sequence"]) for r in rows]

    adm = wt_id < ADMISSIBLE_WT_IDENTITY_PCT
    n_total = len(rows)
    n_adm = int(adm.sum())
    adm_seqs = [seqs[i] for i in range(n_total) if adm[i]]

    div_all = fr.diversity(seqs) if n_total >= 2 else {"mean_pairwise_identity": None, "near_duplicate_rate": None}
    div_adm = (
        fr.diversity(adm_seqs)
        if len(adm_seqs) >= 2
        else {"mean_pairwise_identity": None, "near_duplicate_rate": None}
    )

    novel_meets_floor = n_adm >= MIN_ADMISSIBLE_PER_CELL
    novel_block = {
        "n_admissible": n_adm,
        "feasibility_rate": float(n_adm / n_total),
        "meets_25_floor": novel_meets_floor,
        "tm_score": float(tm[adm].mean()) if novel_meets_floor else None,
        "tm_score_raw_any_n": float(tm[adm].mean()) if n_adm > 0 else None,
        "plddt_mean": float(plddt[adm].mean()) if novel_meets_floor else None,
        "ptm": float(ptm[adm].mean()) if novel_meets_floor else None,
        "wt_identity_pct": float(wt_id[adm].mean()) if n_adm > 0 else None,
        "msa_max_identity_pct": float(msa_id[adm].mean()) if n_adm > 0 else None,
        "potts_energy_full": float(e_full[adm].mean()) if n_adm > 0 else None,
        "potts_energy_fields": float(e_fields[adm].mean()) if n_adm > 0 else None,
        "potts_energy_couplings": float(e_couplings[adm].mean()) if n_adm > 0 else None,
        "shannon_H_bits": float(shannon[adm].mean()) if n_adm > 0 else None,
        "mean_pairwise_identity": div_adm["mean_pairwise_identity"],
        "near_duplicate_rate": div_adm["near_duplicate_rate"],
    }
    whole_pool_block = {
        "n": n_total,
        "tm_score": float(tm.mean()),
        "plddt_mean": float(plddt.mean()),
        "ptm": float(ptm.mean()),
        "wt_identity_pct": float(wt_id.mean()),
        "msa_max_identity_pct": float(msa_id.mean()),
        "potts_energy_full": float(e_full.mean()),
        "potts_energy_fields": float(e_fields.mean()),
        "potts_energy_couplings": float(e_couplings.mean()),
        "shannon_H_bits": float(shannon.mean()),
        "mean_pairwise_identity": div_all["mean_pairwise_identity"],
        "near_duplicate_rate": div_all["near_duplicate_rate"],
    }

    first = rows[0]
    return {
        "model": str(first["model"]),
        "domain": str(first["domain"]),
        "arm": str(first["arm"]),
        "lambda_fields": first.get("lambda_fields"),
        "lambda_coupling": first.get("lambda_coupling"),
        "source": str(first.get("source") or "h200_exp4_2"),
        "n": n_total,
        "novel_lt20": novel_block,
        "whole_pool": whole_pool_block,
    }


def _safe_mean(values: Sequence[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else None


def _aggregate_arm_across_domains(
    cell_map: Mapping[tuple[str, str, str], Mapping[str, Any]],
    model_id: str,
    arm_key: str,
    domains: Sequence[str],
) -> dict[str, Any] | None:
    present = [cell_map[(model_id, d, arm_key)] for d in domains if (model_id, d, arm_key) in cell_map]
    if not present:
        return None
    novels = [c["novel_lt20"] for c in present]
    wholes = [c["whole_pool"] for c in present]
    return {
        "n_domains_present": len(present),
        "n_domains_meeting_25_floor": sum(1 for n in novels if n["meets_25_floor"]),
        "lambda_fields": present[0].get("lambda_fields"),
        "lambda_coupling": present[0].get("lambda_coupling"),
        "novel_lt20": {
            "tm_score": _safe_mean([n["tm_score"] for n in novels]),
            "feasibility_rate": _safe_mean([n["feasibility_rate"] for n in novels]),
            "plddt_mean": _safe_mean([n["plddt_mean"] for n in novels]),
            "ptm": _safe_mean([n["ptm"] for n in novels]),
            "wt_identity_pct": _safe_mean([n["wt_identity_pct"] for n in novels]),
            "msa_max_identity_pct": _safe_mean([n["msa_max_identity_pct"] for n in novels]),
            "potts_energy_full": _safe_mean([n["potts_energy_full"] for n in novels]),
            "potts_energy_fields": _safe_mean([n["potts_energy_fields"] for n in novels]),
            "potts_energy_couplings": _safe_mean([n["potts_energy_couplings"] for n in novels]),
            "shannon_H_bits": _safe_mean([n["shannon_H_bits"] for n in novels]),
            "mean_pairwise_identity": _safe_mean([n["mean_pairwise_identity"] for n in novels]),
            "near_duplicate_rate": _safe_mean([n["near_duplicate_rate"] for n in novels]),
        },
        "whole_pool": {
            "tm_score": _safe_mean([w["tm_score"] for w in wholes]),
            "plddt_mean": _safe_mean([w["plddt_mean"] for w in wholes]),
            "ptm": _safe_mean([w["ptm"] for w in wholes]),
            "wt_identity_pct": _safe_mean([w["wt_identity_pct"] for w in wholes]),
            "msa_max_identity_pct": _safe_mean([w["msa_max_identity_pct"] for w in wholes]),
            "potts_energy_full": _safe_mean([w["potts_energy_full"] for w in wholes]),
            "potts_energy_fields": _safe_mean([w["potts_energy_fields"] for w in wholes]),
            "potts_energy_couplings": _safe_mean([w["potts_energy_couplings"] for w in wholes]),
            "shannon_H_bits": _safe_mean([w["shannon_H_bits"] for w in wholes]),
            "mean_pairwise_identity": _safe_mean([w["mean_pairwise_identity"] for w in wholes]),
            "near_duplicate_rate": _safe_mean([w["near_duplicate_rate"] for w in wholes]),
        },
    }


def _compute_model_contrasts(
    cell_map: Mapping[tuple[str, str, str], Mapping[str, Any]],
    model_id: str,
    domains: Sequence[str],
) -> dict[str, Any]:
    """Compute paired domain contrasts for one model across novel (<20%) and whole-pool strata."""
    contrast_specs: tuple[tuple[str, tuple[str, str], tuple[str, str]], ...] = (
        ("C-Pr", (model_id, "C"), (model_id, "Pr")),
        ("C-Cx", (model_id, "C"), (model_id, "Cx")),
        ("Pr-N", (model_id, "Pr"), (model_id, "N")),
        ("Pr_strong-N", (model_id, "Pr_strong"), (model_id, "N")),
        ("C-N", (model_id, "C"), (model_id, "N")),
        ("P-N", ("potts-joint", "P"), (model_id, "N")),
        ("H-N", ("natural", "H"), (model_id, "N")),
        ("P-Pr", ("potts-joint", "P"), (model_id, "Pr")),
        ("P-Pr_strong", ("potts-joint", "P"), (model_id, "Pr_strong")),
        ("P-C", ("potts-joint", "P"), (model_id, "C")),
    )
    out: dict[str, dict[str, Any]] = {
        "novel_lt20_tm": {},
        "novel_lt20_feasibility": {},
        "whole_pool_tm": {},
    }
    for cname, (m_a, arm_a), (m_b, arm_b) in contrast_specs:
        for block_name, stratum, metric in (
            ("novel_lt20_tm", "novel_lt20", "tm_score"),
            ("novel_lt20_feasibility", "novel_lt20", "feasibility_rate"),
            ("whole_pool_tm", "whole_pool", "tm_score"),
        ):
            per: dict[str, dict[str, float]] = {}
            miss: dict[str, str] = {}
            for d in domains:
                ca = cell_map.get((m_a, d, arm_a))
                cb = cell_map.get((m_b, d, arm_b))
                if ca is None or cb is None:
                    miss[d] = f"missing {(m_a, arm_a) if ca is None else (m_b, arm_b)}"
                    continue
                va = ca[stratum][metric]
                vb = cb[stratum][metric]
                if va is None or vb is None:
                    miss[d] = (
                        f"below {MIN_ADMISSIBLE_PER_CELL} admissible in "
                        f"{arm_a if va is None else arm_b}"
                    )
                    continue
                id_stratum = "novel_lt20" if block_name == "novel_lt20_tm" else "whole_pool"
                per[d] = {
                    "diff": float(va - vb),
                    "identity_diff": float(
                        ca[id_stratum]["wt_identity_pct"] - cb[id_stratum]["wt_identity_pct"]
                    ),
                    "msa_identity_diff": float(
                        ca[id_stratum]["msa_max_identity_pct"]
                        - cb[id_stratum]["msa_max_identity_pct"]
                    ),
                }
            out[block_name][cname] = _estimate_domain_contrast(per, miss)
    return out


def assemble_phase_space_summary(
    *,
    family_ref_dir: Path | str = DEFAULT_FAMILY_REF_DIR,
    e20_per_sequence: Path | str = DEFAULT_E20_PER_SEQUENCE,
    e20_confirmatory_json: Path | str = DEFAULT_E20_CONFIRMATORY_JSON,
    pg3rb_per_sequence: Path | str = DEFAULT_PG3RB_PER_SEQUENCE,
    pg3rb_robustness_json: Path | str = DEFAULT_PG3RB_ROBUSTNESS_JSON,
    h200_folded_dir: Path | str | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Assemble the complete unified Experiment 4.2 JSON artifact."""
    family_models = {
        d: load_family_model(d, Path(family_ref_dir)) for d in SEALED_8_DOMAINS
    }
    e20_gate = verify_e20_confirmatory_reproduction(
        per_sequence_path=e20_per_sequence,
        confirmatory_json_path=e20_confirmatory_json,
    )
    pg3_gate = verify_progen3_robustness_reproduction(
        per_sequence_path=pg3rb_per_sequence,
        robustness_json_path=pg3rb_robustness_json,
    )

    raw_cells = load_cached_local_cells(
        family_models=family_models,
        e20_per_sequence=e20_per_sequence,
        pg3rb_per_sequence=pg3rb_per_sequence,
    )
    if h200_folded_dir is not None:
        h200_cells = load_h200_folded_cells(h200_folded_dir, domains=SEALED_8_DOMAINS)
        raw_cells.update(h200_cells)

    cell_summaries: dict[tuple[str, str, str], dict[str, Any]] = {
        key: summarise_phase_space_cell(rows) for key, rows in sorted(raw_cells.items())
    }

    expected_new_cells = [
        (m, d, a)
        for m in NEW_H200_MODELS
        for d in SEALED_8_DOMAINS
        for a in PHASE_SPACE_ARM_KEYS
    ]
    missing_new_cells = [c for c in expected_new_cells if c not in cell_summaries]
    if missing_new_cells and not allow_partial:
        raise FileNotFoundError(
            f"Missing {len(missing_new_cells)}/{len(expected_new_cells)} H200 folded cells "
            f"(first 5: {missing_new_cells[:5]}). Pass --allow-partial for interim assembly."
        )

    # Reference arms (P, H)
    reference_arms_summary: dict[str, Any] = {}
    for arm_code, model_label in REFERENCE_ARMS.items():
        reference_arms_summary[arm_code] = {
            "model_label": model_label,
            "summary": _aggregate_arm_across_domains(
                cell_summaries, model_label, arm_code, SEALED_8_DOMAINS
            ),
            "per_domain": {
                d: cell_summaries[(model_label, d, arm_code)]
                for d in SEALED_8_DOMAINS
                if (model_label, d, arm_code) in cell_summaries
            },
        }

    # Per-PLM summaries & contrasts
    models_out: dict[str, Any] = {}
    paradigm_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    completed_plms: list[str] = []
    pending_plms: list[str] = []

    for spec in PHASE_SPACE_MODELS:
        mid = spec.model_id
        meta = ARM_METADATA[mid]
        arms_present: dict[str, Any] = {}
        for arm_key in PHASE_SPACE_ARM_KEYS:
            agg = _aggregate_arm_across_domains(cell_summaries, mid, arm_key, SEALED_8_DOMAINS)
            if agg is not None:
                arms_present[arm_key] = agg
        if not arms_present:
            pending_plms.append(mid)
            continue
        completed_plms.append(mid)
        contrasts = _compute_model_contrasts(cell_summaries, mid, SEALED_8_DOMAINS)
        entry = {
            "model_id": mid,
            "display_name": meta.display_name,
            "family": meta.family,
            "paradigm": meta.paradigm,
            "modality": meta.modality,
            "architecture": meta.architecture,
            "params_m": meta.params_m,
            "vocab_type": meta.vocab_type,
            "backend": spec.backend,
            "arms": arms_present,
            "contrasts": contrasts,
        }
        models_out[mid] = entry
        paradigm_buckets[meta.paradigm].append(entry)

    # Paradigm-level rollups across completed PLMs
    paradigm_summary: dict[str, Any] = {}
    for paradigm, entries in sorted(paradigm_buckets.items()):
        arm_rollups: dict[str, Any] = {}
        for arm_key in PHASE_SPACE_ARM_KEYS:
            sub = [e["arms"][arm_key] for e in entries if arm_key in e["arms"]]
            if not sub:
                continue
            arm_rollups[arm_key] = {
                "n_models": len(sub),
                "novel_lt20_tm_mean": _safe_mean([s["novel_lt20"]["tm_score"] for s in sub]),
                "novel_lt20_feasibility_mean": _safe_mean([s["novel_lt20"]["feasibility_rate"] for s in sub]),
                "whole_pool_tm_mean": _safe_mean([s["whole_pool"]["tm_score"] for s in sub]),
                "whole_pool_wt_identity_mean": _safe_mean([s["whole_pool"]["wt_identity_pct"] for s in sub]),
                "whole_pool_potts_full_mean": _safe_mean([s["whole_pool"]["potts_energy_full"] for s in sub]),
            }
        paradigm_summary[paradigm] = {
            "models": [e["model_id"] for e in entries],
            "n_models": len(entries),
            "arms": arm_rollups,
            "mean_C_minus_Pr_novel_lt20_tm": _safe_mean(
                [e["contrasts"]["novel_lt20_tm"]["C-Pr"]["mean"] for e in entries]
            ),
            "mean_C_minus_Cx_novel_lt20_tm": _safe_mean(
                [e["contrasts"]["novel_lt20_tm"]["C-Cx"]["mean"] for e in entries]
            ),
            "mean_C_minus_Pr_whole_pool_tm": _safe_mean(
                [e["contrasts"]["whole_pool_tm"]["C-Pr"]["mean"] for e in entries]
            ),
            "mean_C_minus_Cx_whole_pool_tm": _safe_mean(
                [e["contrasts"]["whole_pool_tm"]["C-Cx"]["mean"] for e in entries]
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domains": list(SEALED_8_DOMAINS),
        "domain_lengths": {d: family_models[d].length for d in SEALED_8_DOMAINS},
        "protocol": {
            "pool_size_new_models": DEFAULT_POOL_SIZE,
            "permutation_seed": PERMUTATION_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": N_BOOTSTRAP,
            "admissible_wt_identity_below_pct": ADMISSIBLE_WT_IDENTITY_PCT,
            "min_admissible_per_cell": MIN_ADMISSIBLE_PER_CELL,
            "arms": [
                {
                    "arm_key": a.arm_key,
                    "guidance_arm": a.guidance_arm,
                    "lambda_fields": a.lambda_fields,
                    "lambda_coupling": a.lambda_coupling,
                    "description": a.description,
                }
                for a in PHASE_SPACE_ARMS
            ],
            "interval_coverage_note": BCA_COVERAGE_NOTE,
        },
        "coverage": {
            "reference_arms_present": sorted(REFERENCE_ARMS.keys()),
            "total_panel_plms": len(PHASE_SPACE_MODELS),
            "completed_plms": completed_plms,
            "n_completed_plms": len(completed_plms),
            "pending_plms": pending_plms,
            "n_pending_plms": len(pending_plms),
            "expected_new_h200_cells": len(expected_new_cells),
            "missing_new_h200_cells": len(missing_new_cells),
            "is_complete_panel": len(missing_new_cells) == 0,
        },
        "reproduction_gates": {
            "e20_confirmatory_20261014": e20_gate,
            "progen3_robustness_20261015": pg3_gate,
        },
        "reference_arms": reference_arms_summary,
        "models": models_out,
        "paradigm_summary": paradigm_summary,
        "cells": [cell_summaries[k] for k in sorted(cell_summaries)],
    }


def write_unified_artifact(payload: Mapping[str, Any], out_path: Path | str) -> tuple[Path, Path, str]:
    """Atomically write JSON artifact and its ``.sha256`` sidecar."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, target)
    digest = sha256_file(target)
    sha_path = Path(str(target) + ".sha256")
    sha_tmp = Path(str(sha_path) + ".tmp")
    sha_tmp.write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    os.replace(sha_tmp, sha_path)
    return target, sha_path, digest
