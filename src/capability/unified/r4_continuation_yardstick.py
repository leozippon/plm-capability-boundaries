"""Unified Workstream R4-Continuation: Unaligned Homologous-Prefix Continuation Across 31 Models.

Expands historical Experiment E11 (``progen2-medium``) to all 31 generative checkpoints:
29 unconditional protein / joint / hybrid / discrete-diffusion / multimodal checkpoints
(matching Experiment 4.1) plus 2 text-initialized causal controls (``gpt2-large``, ``llama-2-7b``).

Design invariants
-----------------
1. **Historical E11 Reproduction Anchor (``< 1e-12``)**: Recomputes historical
   ``progen2-medium`` E11 ``yields``, ``yield_contrast`` (10,000-draw paired cluster
   bootstrap, ``seed=20261008``), and ``context_identity`` directly from the frozen
   ``products.jsonl`` and ``generation_annotation.json`` and asserts ``< 1e-12`` match to
   ``results/R6/e11_completed_20261011/comparison/conditioned_generation_comparison.json``.
2. **Matched 22-Target, 5-Condition Ladder (`budget = 1024`, `attempts_per_cell = 32`)**:
   - **Primary 12-Target / 12-Cluster E11 Panel**: ``no_context``, ``unrelated``,
     ``remote_homolog`` (``id_30_50``), ``close_homolog`` (``id_70_90``) (`48` cells).
   - **Companion 12-Target / 11-Cluster Twilight Panel**: ``no_context``, ``unrelated``,
     ``twilight_homolog`` (``id_lt_30``), reusing overlapping controls from Primary 12
     (`+32` cells -> `80` cells/model, `2,560` attempts/model).
3. **Single-Chain Diffusion & Multimodal Continuation**: ``dplm-{150m,650m,3b}``
   (strictly ``uniform_random``, never ``max_confidence``) and ``esm3-1.4b`` (cosine
   unmasking) pack ``[<cls/bos> item_1 <eos> ... <cls/bos> <mask>^{L_gen} <eos>]`` inside
   ``budget=1024``, holding all context residues fixed while unmasking the continuation span.
4. **Full Downstream Evaluation**: Exact 3-part E11 prompt-copying verdict
   (``max_lcs >= 30``, ``kmer_containment >= 0.50``, ``DIAMOND >= 50%``), Shannon entropy
   and ``distinct_5mer``, pinned Pfam-A ``hmmscan`` target-family recall, UniRef50 novelty,
   and ``ESMFold2-hf`` 3D folding + target-WT ``TM-score`` with 10,000-draw paired cluster
   bootstrap contrasts.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..context import context_homologue as ch
from ..context import homology_context as H
from ..core.amino_acids import AA20
from ..core.io import sha256_file
from ..core.protein_annotations import parse_hmmscan_table
from ..generation import unconditional_generation as ug
from ..generation.family_oracle import (
    HMMSCAN_RELATIVE,
    PFAM_RELATIVE,
    Oracle,
    scan_command,
)
from ..generation.generative_control import (
    COMPLETE_DOMAIN_COVERAGE,
    best_profile_coverage,
    parse_domain_table,
)
from ..ladder import fold_geometry as fg
from .roster import ARM_METADATA, REPO_ROOT, UNIFIED_OUTPUT_DIR

SCHEMA_VERSION = "r4_continuation_unified_31_v1"
ATTEMPT_SCHEMA = "r4_continuation_attempt_v1"
FOLDED_SCHEMA = "r4_continuation_folded_v1"

BUDGET_TOKENS = 1024
ATTEMPTS_PER_CELL = 32
DEFAULT_MAX_NEW_TOKENS = 384
RITA_MAX_NEW_TOKENS = 256
MIN_PRODUCT_RESIDUES = H.MIN_PRODUCT_RESIDUES  # 32
FOLD_CANDIDATES_PER_CELL = 2
DIFFUSION_MAX_STEPS = 64
GENERATION_SEED_BASE = 20261125
BOOTSTRAP_DRAWS = H.BOOTSTRAP_DRAWS  # 10000
BOOTSTRAP_SEED = H.BOOTSTRAP_SEED  # 20261008
NATURAL_ENTROPY_THRESHOLD = 3.0
FOLD_SELECT_ENTROPY_THRESHOLD = 2.5
TM_RECAP_THRESHOLD = 0.5

PRIMARY_CONDITIONS: tuple[str, ...] = (
    "no_context",
    "unrelated",
    "remote_homolog",
    "close_homolog",
)
TWILIGHT_CONDITIONS: tuple[str, ...] = (
    "no_context",
    "unrelated",
    "twilight_homolog",
)
ALL_CONDITIONS: tuple[str, ...] = (
    "no_context",
    "unrelated",
    "twilight_homolog",
    "remote_homolog",
    "close_homolog",
)
CONDITION_TO_E09_BIN: dict[str, str] = {
    "no_context": "no_context",
    "unrelated": "unrelated",
    "twilight_homolog": "id_lt_30",
    "remote_homolog": "id_30_50",
    "close_homolog": "id_70_90",
}

#: Exact historical E11 12-target / 12-cluster Primary panel.
PRIMARY_12_TARGETS: tuple[str, ...] = (
    "q00000",
    "q00003",
    "q00004",
    "q00005",
    "q00007",
    "q00009",
    "q00010",
    "q00012",
    "q00013",
    "q00020",
    "q00022",
    "q00024",
)
#: Companion 12-target / 11-cluster <30% Twilight panel (q00013, q00024 overlap Primary 12).
TWILIGHT_12_TARGETS: tuple[str, ...] = (
    "q00013",
    "q00024",
    "q00052",
    "q00062",
    "q00063",
    "q00084",
    "q00085",
    "q00091",
    "q00103",
    "q00115",
    "q00117",
    "q00133",
)
ALL_22_TARGETS: tuple[str, ...] = tuple(sorted(set(PRIMARY_12_TARGETS) | set(TWILIGHT_12_TARGETS)))
assert len(PRIMARY_12_TARGETS) == 12
assert len(TWILIGHT_12_TARGETS) == 12
assert len(ALL_22_TARGETS) == 22

DEFAULT_HOMOLOGS_JSON = Path(
    "results/extensions/homology_context_20261008/uniref90_2026_03/bin/homology_context.json"
)
DEFAULT_E11_COMPARISON_JSON = Path(
    "results/R6/e11_completed_20261011/comparison/conditioned_generation_comparison.json"
)
DEFAULT_E11_ANNOTATION_JSON = Path(
    "results/R6/e11_completed_20261011/annotation/generation_annotation.json"
)
DEFAULT_E11_PRODUCTS_JSONL = Path(
    "results/external_baseline/20261008104808_96e83c09ad71/homctx_e11_generate_progen2_medium_gpu0/products.jsonl"
)
DEFAULT_WORK_DIR = UNIFIED_OUTPUT_DIR / "r4_continuation_pools"
DEFAULT_OUTPUT_JSON = UNIFIED_OUTPUT_DIR / "r4_continuation_unified.json"
EXP4_1_CACHE_DIR = UNIFIED_OUTPUT_DIR / "exp4_1_cache"
ORACLE_ROOT_REL = Path("external/tools/r227-oracle")
DIAMOND_BIN_REL = Path("external/tools/diamond/diamond")
UNIREF50_DB_REL = Path("data/homology_db/uniref50_full.dmnd")

AA20_SET = frozenset(AA20)
AA20_INDEX: dict[str, int] = {aa: idx for idx, aa in enumerate(AA20)}
_AA20_RUN_RE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+")


@dataclass(frozen=True)
class ContinuationModelSpec:
    """Authoritative specification for one checkpoint/budget entry in the 35-entry R4-Continuation panel."""

    model_id: str
    backend: str  # stage46_causal | general_plm_causal | multiseq_icl_causal | progen3_causal | llama2_hf | dplm_diffusion | esm3_diffusion
    default_dtype: str
    default_batch_size: int
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    use_cache: bool = True
    is_text_control: bool = False
    estimated_cost: float = 1.0
    notes: str = ""
    base_model_id: str | None = None
    context_budget: int = BUDGET_TOKENS
    max_context_items: int = 4
    input_alignment_format: str = "unaligned_concat"
    context_training_regime: str = "single_chain_pretrained"

    @property
    def arm_id(self) -> str:
        return self.base_model_id or self.model_id


def model_file_slug(model_id: str) -> str:
    """Filesystem-safe slug for per-model JSONLs (maps '|' to '__')."""
    return str(model_id).replace("|", "__")


R4_CONTINUATION_MODELS: tuple[ContinuationModelSpec, ...] = (
    # ---- ProGen2 (5)
    ContinuationModelSpec("progen2-small", "stage46_causal", "float32", 32, 384, True, False, 1.0),
    ContinuationModelSpec("progen2-base", "stage46_causal", "bfloat16", 32, 384, True, False, 1.8),
    ContinuationModelSpec("progen2-medium", "stage46_causal", "float32", 32, 384, True, False, 1.8),
    ContinuationModelSpec("progen2-large", "stage46_causal", "bfloat16", 32, 384, True, False, 3.0),
    ContinuationModelSpec("progen2-xlarge", "stage46_causal", "bfloat16", 32, 384, True, False, 4.5),
    # ---- ProGen3 (6; BF16 MoE)
    ContinuationModelSpec("progen3-112m", "progen3_causal", "bfloat16", 32, 384, True, False, 1.5),
    ContinuationModelSpec("progen3-219m", "progen3_causal", "bfloat16", 32, 384, True, False, 2.0),
    ContinuationModelSpec("progen3-339m", "progen3_causal", "bfloat16", 32, 384, True, False, 2.5),
    ContinuationModelSpec("progen3-762m", "progen3_causal", "bfloat16", 32, 384, True, False, 3.0),
    ContinuationModelSpec("progen3-1b", "progen3_causal", "bfloat16", 32, 384, True, False, 3.5),
    ContinuationModelSpec("progen3-3b", "progen3_causal", "bfloat16", 32, 384, True, False, 5.5),
    # ---- ProtGPT2/3 (4)
    ContinuationModelSpec("protgpt2", "stage46_causal", "float32", 32, 384, True, False, 2.0),
    ContinuationModelSpec("protgpt3-112m", "general_plm_causal", "float32", 32, 384, True, False, 1.5),
    ContinuationModelSpec("protgpt3-1.3b", "general_plm_causal", "bfloat16", 32, 384, True, False, 2.5),
    ContinuationModelSpec("protgpt3-10b", "general_plm_causal", "bfloat16", 32, 384, True, False, 6.0),
    # ---- Hybrid SSM / Other Protein Causal (3)
    ContinuationModelSpec("dayhoff-170m", "general_plm_causal", "float32", 32, 384, True, False, 2.5, "Jamba hybrid cache"),
    ContinuationModelSpec("proteinglm-7b-clm", "stage46_causal", "float32", 32, 384, True, False, 5.5, "FP32 required by ProteinGLM RoPE SDPA"),
    ContinuationModelSpec("rita-xl", "stage46_causal", "float32", 16, RITA_MAX_NEW_TOKENS, False, False, 6.0, "Uncached; capped 256 tok"),
    # ---- Joint / Adaptation (7)
    ContinuationModelSpec("galactica-125m", "stage46_causal", "float32", 32, 384, True, False, 1.2),
    ContinuationModelSpec("galactica-1.3b", "stage46_causal", "bfloat16", 32, 384, True, False, 2.5),
    ContinuationModelSpec("galactica-6.7b", "stage46_causal", "bfloat16", 32, 384, True, False, 5.0),
    ContinuationModelSpec("galactica-30b", "stage46_causal", "bfloat16", 16, 384, True, False, 11.0, "BF16 for 30B headroom"),
    ContinuationModelSpec("instructprotein", "stage46_causal", "bfloat16", 32, 384, True, False, 2.5),
    ContinuationModelSpec("prollama-stage-1", "stage46_causal", "bfloat16", 32, 384, True, False, 4.5),
    ContinuationModelSpec("prollama", "stage46_causal", "bfloat16", 32, 384, True, False, 4.5),
    # ---- Single-Chain Discrete Diffusion & Multimodal (4)
    ContinuationModelSpec("dplm-150m", "dplm_diffusion", "float32", 32, 384, False, False, 1.5, "uniform_random"),
    ContinuationModelSpec("dplm-650m", "dplm_diffusion", "bfloat16", 32, 384, False, False, 3.0, "uniform_random"),
    ContinuationModelSpec("dplm-3b", "dplm_diffusion", "bfloat16", 32, 384, False, False, 5.0, "uniform_random"),
    ContinuationModelSpec("esm3-1.4b", "esm3_diffusion", "bfloat16", 32, 384, False, False, 4.0, "cosine unmasking"),
    # ---- EXP-R2-228 Text-Initialized Controls (2)
    ContinuationModelSpec("gpt2-large", "stage46_causal", "float32", 32, 384, True, True, 2.0),
    ContinuationModelSpec("llama-2-7b", "llama2_hf", "bfloat16", 32, 384, True, True, 4.5),
    # ---- Multi-Sequence ICL Unaligned Generators (4 entries: 2 models x matched b1024 + native b8192/b4096)
    ContinuationModelSpec(
        "protgpt3-msa",
        "multiseq_icl_causal",
        "bfloat16",
        32,
        384,
        True,
        False,
        3.0,
        "Unaligned <|bos|> 1 <no_gap> + <s> ICL at matched b1024",
        base_model_id="protgpt3-msa",
        context_budget=1024,
        max_context_items=4,
        input_alignment_format="unaligned_concat",
        context_training_regime="multiseq_icl_unaligned",
    ),
    ContinuationModelSpec(
        "protgpt3-msa|b8192",
        "multiseq_icl_causal",
        "bfloat16",
        16,
        384,
        True,
        False,
        5.0,
        "Unaligned <|bos|> 1 <no_gap> + <s> ICL at native b8192 (up to k=8 homologs)",
        base_model_id="protgpt3-msa",
        context_budget=8192,
        max_context_items=8,
        input_alignment_format="unaligned_concat",
        context_training_regime="multiseq_icl_unaligned",
    ),
    ContinuationModelSpec(
        "dayhoff-3b-msa",
        "multiseq_icl_causal",
        "bfloat16",
        32,
        384,
        True,
        False,
        4.5,
        "Unaligned @ + / Jamba ICL at matched b1024",
        base_model_id="dayhoff-3b-msa",
        context_budget=1024,
        max_context_items=4,
        input_alignment_format="unaligned_concat",
        context_training_regime="multiseq_icl_unaligned",
    ),
    ContinuationModelSpec(
        "dayhoff-3b-msa|b4096",
        "multiseq_icl_causal",
        "bfloat16",
        16,
        384,
        True,
        False,
        6.5,
        "Unaligned @ + / Jamba ICL at native b4096 (up to k=8 homologs)",
        base_model_id="dayhoff-3b-msa",
        context_budget=4096,
        max_context_items=8,
        input_alignment_format="unaligned_concat",
        context_training_regime="multiseq_icl_unaligned",
    ),
)

R4_MODEL_MAP: dict[str, ContinuationModelSpec] = {
    spec.model_id: spec for spec in R4_CONTINUATION_MODELS
}
R4_MODEL_IDS: tuple[str, ...] = tuple(spec.model_id for spec in R4_CONTINUATION_MODELS)
R4_SINGLE_CHAIN_MODEL_IDS: tuple[str, ...] = tuple(
    spec.model_id for spec in R4_CONTINUATION_MODELS if spec.context_training_regime == "single_chain_pretrained"
)
R4_MULTISEQ_ICL_MODEL_IDS: tuple[str, ...] = tuple(
    spec.model_id for spec in R4_CONTINUATION_MODELS if spec.context_training_regime == "multiseq_icl_unaligned"
)
for _m in R4_CONTINUATION_MODELS:
    if _m.arm_id not in ARM_METADATA:
        raise AssertionError(f"{_m.arm_id!r} missing from ARM_METADATA")
assert len(R4_SINGLE_CHAIN_MODEL_IDS) == 31
assert len(R4_MULTISEQ_ICL_MODEL_IDS) == 4
assert len(R4_CONTINUATION_MODELS) == 35


# --------------------------------------------------------------------------- #
# Target Panel, Cell Schedule & Sharding
# --------------------------------------------------------------------------- #


def required_target_condition_cells() -> list[tuple[str, str]]:
    """All 80 unique ``(target_id, condition)`` cells per model (48 primary + 32 extra twilight)."""
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for tid in PRIMARY_12_TARGETS:
        for cond in PRIMARY_CONDITIONS:
            pair = (tid, cond)
            if pair not in seen:
                seen.add(pair)
                ordered.append(pair)
    for tid in TWILIGHT_12_TARGETS:
        for cond in TWILIGHT_CONDITIONS:
            pair = (tid, cond)
            if pair not in seen:
                seen.add(pair)
                ordered.append(pair)
    assert len(ordered) == 80
    return ordered


def plan_model_shards(models: Sequence[str], num_shards: int) -> list[list[str]]:
    """Greedy LPT bin-packing of models across ``num_shards`` GPUs by estimated compute cost."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    unique: list[str] = []
    for m in models:
        if m not in R4_MODEL_MAP:
            raise ValueError(f"Unknown R4-Continuation model {m!r}")
        if m not in unique:
            unique.append(m)
    ordered = sorted(unique, key=lambda m: (-R4_MODEL_MAP[m].estimated_cost, m))
    bins: list[list[str]] = [[] for _ in range(num_shards)]
    loads = [0.0] * num_shards
    for mid in ordered:
        idx = min(range(num_shards), key=lambda i: (loads[i], len(bins[i]), i))
        bins[idx].append(mid)
        loads[idx] += R4_MODEL_MAP[mid].estimated_cost
    return bins


def cell_seed(model_id: str, target_id: str, condition: str, *, base: int = GENERATION_SEED_BASE) -> int:
    digest = int(
        hashlib.sha256(f"{base}|{model_id}|{target_id}|{condition}".encode("utf-8")).hexdigest()[:12],
        16,
    )
    return int((digest % 1_900_000_000) + 10_000)


def build_canonical_target_plans(
    contexts: Mapping[str, Any],
    *,
    budget: int = BUDGET_TOKENS,
    max_items: int = 4,
    target_ids: Sequence[str] = ALL_22_TARGETS,
) -> dict[str, dict[str, Any]]:
    """Build canonical 1-token-per-AA reference plans for the 22 targets.

    At ``budget == 1024`` and ``max_items == 4``, delegates verbatim to ``H.plan_conditions``
    for exact historical E11 parity. For native multi-sequence budgets (``b4096`` / ``b8192``,
    ``max_items == 8``), packs up to ``k = min(max_items, floor((budget - referent) / referent))``
    length-windowed homologs (and matched unrelated donors) per condition without any ``-`` gaps.
    """
    by_id = {str(t["target_id"]): t for t in contexts["targets"]}

    def _ref_tok_len(seq: str) -> int:
        # Canonical single-AA item length: 1 start + len(seq) + 1 close + 1 eos
        return len(seq) + 3

    plans: dict[str, dict[str, Any]] = {}
    for tid in target_ids:
        target = by_id[tid]
        wt = str(target["wildtype"])
        referent = _ref_tok_len(wt)
        room = budget - referent
        if int(budget) == BUDGET_TOKENS and int(max_items) <= H.CONTEXT_ITEMS_MAX:
            conditions, budget_rec = H.plan_conditions(
                wildtype=wt,
                bins=target["bins"],
                donors=target["unrelated_pool"],
                self_copy=target.get("self_copy"),
                token_length=_ref_tok_len,
                referent=referent,
                room=room,
            )
            items_by_cond: dict[str, list[dict[str, Any]]] = {}
            for cond_label, e09_name in CONDITION_TO_E09_BIN.items():
                plan_obj = conditions.get(e09_name)
                if plan_obj is not None and plan_obj.status == "present":
                    items_by_cond[cond_label] = [
                        {
                            "sequence": str(it["sequence"]),
                            "subject": str(it["subject"]),
                            "length": len(str(it["sequence"])),
                        }
                        for it in plan_obj.items
                    ]
        else:
            # Extended multi-sequence ICL budget (b4096 / b8192): pack up to max_items (k=8).
            low, high = H.item_token_window(referent)
            k_cap = min(int(max_items), max(1, (room - DEFAULT_MAX_NEW_TOKENS) // max(1, referent)))
            budget_rec = {
                "item_count_cap": int(k_cap),
                "item_token_referent": int(referent),
                "item_token_window": [int(low), int(high)],
                "room": int(room),
                "position_budget": int(budget),
            }

            def _select_extended(cands: Sequence[Mapping[str, Any]], *, limit: int) -> list[dict[str, Any]]:
                in_win = [c for c in cands if low <= _ref_tok_len(str(c["sequence"])) <= high]
                pool = in_win if in_win else list(cands)
                chosen: list[dict[str, Any]] = []
                used_toks = 0
                max_ctx_toks = max(MIN_PRODUCT_RESIDUES, int(budget) - DEFAULT_MAX_NEW_TOKENS - 16)
                for it in pool:
                    if len(chosen) >= limit:
                        break
                    seq = str(it["sequence"])
                    tlen = len(seq) + 1
                    if chosen and used_toks + tlen > max_ctx_toks:
                        break
                    chosen.append({"sequence": seq, "subject": str(it["subject"]), "length": len(seq)})
                    used_toks += tlen
                return chosen

            items_by_cond = {"no_context": []}
            for cond_label, e09_name in CONDITION_TO_E09_BIN.items():
                if cond_label == "no_context":
                    continue
                if cond_label == "unrelated":
                    cands = list(target.get("unrelated_pool") or ())
                else:
                    cands = list(target.get("bins", {}).get(e09_name) or ())
                if cands:
                    items_by_cond[cond_label] = _select_extended(cands, limit=k_cap)
        plans[tid] = {
            "target_id": tid,
            "cluster": str(target["cluster"]),
            "wildtype": wt,
            "wildtype_length": len(wt),
            "budget_record": budget_rec,
            "items": items_by_cond,
        }
    return plans


# --------------------------------------------------------------------------- #
# Sequence Complexity, Extraction & Structural TM-score
# --------------------------------------------------------------------------- #


def shannon_entropy_bits(sequence: str) -> float:
    if not sequence:
        return 0.0
    counts = np.zeros(20, dtype=np.float64)
    valid = 0
    for ch_aa in sequence:
        idx = AA20_INDEX.get(ch_aa)
        if idx is not None:
            counts[idx] += 1.0
            valid += 1
    if valid == 0:
        return 0.0
    probs = counts[counts > 0.0] / float(valid)
    return float(-np.sum(probs * np.log2(probs)))


def distinct_kmer_fraction(sequence: str, k: int = 5) -> float:
    n = len(sequence) - k + 1
    if n <= 0:
        return 0.0
    kmers = {sequence[i : i + k] for i in range(n)}
    return float(len(kmers) / n)


def extract_canonical_residues(model_id: str, raw_text: str) -> tuple[str, str]:
    """Extract canonical AA20 protein continuation and classify stop status."""
    spec = R4_MODEL_MAP[model_id]
    if spec.is_text_control:
        # Text controls: take the leading/longest canonical AA20 run before any EOT/EOS marker.
        cleaned = raw_text.split("<|endoftext|>", 1)[0].split("</s>", 1)[0]
        terminated = ("<|endoftext|>" in raw_text) or ("</s>" in raw_text)
        cleaned = re.sub(r"[\s\r\n]+", "", cleaned)
        runs = _AA20_RUN_RE.findall(cleaned)
        seq = max(runs, key=len) if runs else ""
        if not seq:
            return "", "empty_or_noncanonical"
        return seq, ("native_terminal" if terminated else "budget_censored")

    if model_id in ug.ADMITTED:
        seq = ug.extract_residues(model_id, raw_text)
        status = ug.stop_status(model_id, raw_text, seq)
        return seq, status

    # General PLM / Multi-seq ICL causal (progen3-*, protgpt3-*, protgpt3-msa, dayhoff-170m, dayhoff-3b-msa)
    close_markers = ("<s>", "<|eos|>", "<eos>", "2", "*", "/")
    terminated = False
    head = raw_text
    for marker in close_markers:
        if marker in head:
            head = head.split(marker, 1)[0]
            terminated = True
    seq = "".join(c for c in head.upper() if c in AA20_SET)
    if not seq:
        return "", "empty_or_noncanonical"
    return seq, ("native_terminal" if terminated else "budget_censored")


def _nw_structural_alignment(score_mat: np.ndarray, gap_penalty: float = -0.6) -> tuple[np.ndarray, np.ndarray]:
    """Fast Needleman-Wunsch DP on ``(L_m, L_r)`` TM-score similarity matrix using native Python lists."""
    lm, lr = score_mat.shape
    prev_dp = [gap_penalty * j for j in range(lr + 1)]
    tb = np.zeros((lm + 1, lr + 1), dtype=np.int8)  # 0=diag, 1=up(gap in ref), 2=left(gap in mob)
    tb[1:, 0] = 1
    tb[0, 1:] = 2
    s_rows = score_mat.tolist()

    for i in range(1, lm + 1):
        row_s = s_rows[i - 1]
        cur_dp = [0.0] * (lr + 1)
        cur_dp[0] = gap_penalty * i
        cur_tb = [0] * (lr + 1)
        cur_tb[0] = 1
        left = cur_dp[0]
        gp = gap_penalty
        for j in range(1, lr + 1):
            d_v = prev_dp[j - 1] + row_s[j - 1]
            u_v = prev_dp[j] + gp
            l_v = left + gp
            if d_v >= u_v and d_v >= l_v:
                left = d_v
                cur_tb[j] = 0
            elif u_v >= l_v:
                left = u_v
                cur_tb[j] = 1
            else:
                left = l_v
                cur_tb[j] = 2
            cur_dp[j] = left
        prev_dp = cur_dp
        tb[i] = cur_tb

    idx_m: list[int] = []
    idx_r: list[int] = []
    i, j = lm, lr
    while i > 0 and j > 0:
        move = int(tb[i, j])
        if move == 0:
            idx_m.append(i - 1)
            idx_r.append(j - 1)
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    idx_m.reverse()
    idx_r.reverse()
    return np.asarray(idx_m, dtype=np.int64), np.asarray(idx_r, dtype=np.int64)


def structural_tm_score(mobile: np.ndarray, reference: np.ndarray) -> float:
    """Sequence-independent TM-score normalized by ``len(reference)``, handling unequal lengths.

    Uses ``fold_geometry.tm_score`` directly when ``len(mobile) == len(reference)`` and also
    screens multi-seed gapless fragment superpositions followed by Needleman-Wunsch DP
    iterative Kabsch refinement on top seeds (built on ``fold_geometry._superpose``, ``_apply``,
    and ``_tm_d0``) so insertions, deletions, and length mismatches between free-length
    continuations and target WTs are scored faithfully and fast.
    """
    mob = np.asarray(mobile, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if mob.ndim != 2 or ref.ndim != 2 or mob.shape[1] != 3 or ref.shape[1] != 3:
        raise ValueError(f"Expected (L, 3) CA traces, got {mob.shape} and {ref.shape}")
    lm, lr = int(mob.shape[0]), int(ref.shape[0])
    if lm < 3 or lr < 3:
        return 0.0

    best_tm = 0.0
    if lm == lr:
        best_tm = float(fg.tm_score(mob, ref)["tm_score"])
        if best_tm >= 0.99:
            return best_tm

    d0 = max(0.5, 1.24 * float(lr - 15) ** (1.0 / 3.0) - 1.8) if lr > 21 else 0.5
    d0_sq = d0 * d0
    frag = min(lm, lr, 24)
    step_m = max(1, (lm - frag) // 3) if lm > frag else 1
    step_r = max(1, (lr - frag) // 3) if lr > frag else 1

    m_starts = sorted({0, max(0, (lm - frag) // 2), max(0, lm - frag), *range(0, lm - frag + 1, step_m)})[:5]
    r_starts = sorted({0, max(0, (lr - frag) // 2), max(0, lr - frag), *range(0, lr - frag + 1, step_r)})[:5]

    # Stage A: Fast vectorized gapless screening of all fragment seeds to pick top-4 candidates.
    ranked_seeds: list[tuple[float, np.ndarray, np.ndarray]] = []
    for ms in m_starts:
        for rs in r_starts:
            s_m = np.arange(ms, ms + frag, dtype=np.int64)
            s_r = np.arange(rs, rs + frag, dtype=np.int64)
            rot, trans = fg._superpose(mob[s_m], ref[s_r])
            aligned = fg._apply(rot, trans, mob)
            shift = rs - ms
            i0 = max(0, -shift)
            i1 = min(lm, lr - shift)
            if i1 - i0 < 3:
                continue
            d2_diag = np.sum((aligned[i0:i1] - ref[i0 + shift : i1 + shift]) ** 2, axis=1)
            coarse_tm = float(np.sum(1.0 / (1.0 + d2_diag / d0_sq)) / lr)
            if coarse_tm > best_tm:
                best_tm = coarse_tm
            ranked_seeds.append((coarse_tm, rot, trans))

    ranked_seeds.sort(key=lambda item: item[0], reverse=True)
    # Stage B: Iterative NW-DP + Kabsch refinement on the top 4 screened seeds.
    for _coarse, rot, trans in ranked_seeds[:4]:
        for _ in range(3):
            aligned = fg._apply(rot, trans, mob)
            diff = aligned[:, None, :] - ref[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            s_mat = 1.0 / (1.0 + d2 / d0_sq)
            im, ir = _nw_structural_alignment(s_mat, gap_penalty=-0.6)
            if im.size < 3:
                break
            pair_d2 = d2[im, ir]
            tm_val = float(np.sum(1.0 / (1.0 + pair_d2 / d0_sq)) / lr)
            if tm_val > best_tm:
                best_tm = tm_val
            close = pair_d2 <= (2.0 * d0) ** 2
            if int(close.sum()) >= 3:
                rot, trans = fg._superpose(mob[im[close]], ref[ir[close]])
            else:
                rot, trans = fg._superpose(mob[im], ref[ir])
    return float(min(1.0, max(0.0, best_tm)))


# --------------------------------------------------------------------------- #
# Prompt Construction & Batched Continuation Samplers (Causal, DPLM, ESM3)
# --------------------------------------------------------------------------- #


def _filtered_sampling_scores(
    logits: torch.Tensor, *, temperature: float = 1.0, top_k: int = 950, top_p: float = 0.95
) -> torch.Tensor:
    """Vectorized temperature + top-k + nucleus (top-p) filter matching E11 / ug.POLICY."""
    scores = logits.float() / float(temperature)
    vocab = scores.shape[-1]
    if 0 < int(top_k) < vocab:
        kth = torch.topk(scores, int(top_k), dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    if 0.0 < float(top_p) < 1.0:
        sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
        probs = torch.softmax(sorted_scores, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = (cum - probs) >= float(top_p)
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        out = torch.full_like(scores, float("-inf"))
        out.scatter_(-1, sorted_idx, sorted_scores)
        scores = out
    return scores


def _sample_next_token_ids(
    logits: torch.Tensor,
    *,
    generator: torch.Generator,
    temperature: float = 1.0,
    top_k: int = 950,
    top_p: float = 0.95,
) -> torch.Tensor:
    """Fast vectorized GPU/CPU top-k + nucleus (top-p) sampling returning ``(B,)`` token IDs.

    Applies ``torch.topk(..., sorted=True)`` *before* nucleus masking and softmax so large-vocab
    models (e.g., Galactica 50k, GPT-2 50k, LLaMA 32k) never sort or CPU-sample 50k vectors.
    """
    scores = logits.float() / float(temperature)
    vocab = int(scores.shape[-1])
    k = min(int(top_k), vocab) if int(top_k) > 0 else vocab
    top_vals, top_idx = torch.topk(scores, k, dim=-1, sorted=True)
    if 0.0 < float(top_p) < 1.0:
        probs_raw = torch.softmax(top_vals, dim=-1)
        cum = torch.cumsum(probs_raw, dim=-1)
        remove = (cum - probs_raw) >= float(top_p)
        remove[..., 0] = False
        top_vals = top_vals.masked_fill(remove, float("-inf"))
    probs = torch.softmax(top_vals, dim=-1)
    sampled_pos = torch.multinomial(probs, 1, generator=generator)
    return top_idx.gather(-1, sampled_pos).squeeze(-1)


def _supports_kwarg(model: Any, kwarg: str) -> bool:
    try:
        return kwarg in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _token_id_or_none(tokenizer: Any, token_str: str) -> int | None:
    if hasattr(tokenizer, "token_to_id"):
        val = tokenizer.token_to_id(token_str)
        return None if val is None or int(val) < 0 else int(val)
    val = tokenizer.convert_tokens_to_ids(token_str)
    if val is None or isinstance(val, list):
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    if int(val) < 0 or (unk is not None and int(val) == int(unk) and token_str != getattr(tokenizer, "unk_token", "")):
        return None
    return int(val)


def build_causal_prompt_and_stops(
    *,
    model_id: str,
    backend: str,
    loaded: Any,
    context_sequences: Sequence[str],
    budget: int = BUDGET_TOKENS,
    min_headroom: int = MIN_PRODUCT_RESIDUES,
) -> tuple[list[int], set[int], int]:
    """Pack unaligned context items + native start marker inside ``budget`` tokens.

    Trims trailing context items by 1 if needed so at least ``min_headroom`` positions remain.
    Returns ``(prompt_ids, stop_ids, items_used)``.
    """
    items = list(context_sequences)

    if backend == "progen3_causal":
        tok = loaded.tokenizer
        bos = int(loaded.config.bos_token_id)
        eos = int(loaded.config.eos_token_id)
        d1 = _token_id_or_none(tok, "1")
        d2 = _token_id_or_none(tok, "2")
        if d1 is None or d2 is None:
            raise ValueError(f"{model_id}: ProGen3 tokenizer missing direction markers '1'/'2'")

        def _enc_seq(s: str) -> list[int]:
            return [int(x) for x in tok.encode(s, add_special_tokens=False).ids]

        start_ids = [bos, d1]
        stops = {d2, eos}
        encoded_items = [[bos, d1] + _enc_seq(s) + [d2, eos] for s in items]
        prefix_ids: list[int] = []

    elif backend == "multiseq_icl_causal":
        # Native unaligned multi-sequence ICL grammars (NO '-' gap insertion!):
        # - protgpt3-msa: <|bos|> 1 <no_gap> + hom_1 <s> hom_2 <s> ... <s>
        # - dayhoff-3b-msa: @ + hom_1 / hom_2 / ... /
        tok = loaded.tokenizer
        prefix_ids = list(loaded.bos_prefix_ids)
        sep_ids = list(getattr(loaded, "separator_ids", ()))
        if not sep_ids:
            raise ValueError(f"{model_id}: multi-sequence ICL adapter missing separator_ids")
        eos_id = getattr(tok, "eos_token_id", None)
        if eos_id is None:
            eos_id = _token_id_or_none(tok, "<|eos|>") or _token_id_or_none(tok, "*")
        d2 = _token_id_or_none(tok, "2") if model_id.startswith("protgpt3-msa") else None
        start_ids = []
        stops = set(sep_ids) | {x for x in (eos_id, d2) if x is not None}
        encoded_items = [
            [loaded.aa_to_id[aa] for aa in s if aa in loaded.aa_to_id] + sep_ids for s in items
        ]

    elif backend == "general_plm_causal":
        # ProtGPT3 (112M/1.3B/10B) or Dayhoff-170M
        tok = loaded.tokenizer
        start_ids = list(loaded.bos_prefix_ids)
        eos_id = getattr(tok, "eos_token_id", None)
        if eos_id is None:
            eos_id = _token_id_or_none(tok, "<|eos|>") or _token_id_or_none(tok, "*")
        d2 = _token_id_or_none(tok, "2") if model_id.startswith("protgpt3-") else None
        sep_tail = [x for x in (d2, eos_id) if x is not None]
        stops = set(sep_tail) | set(getattr(loaded, "separator_ids", ()))
        encoded_items = [
            start_ids + [loaded.aa_to_id[aa] for aa in s if aa in loaded.aa_to_id] + sep_tail
            for s in items
        ]
        prefix_ids = []

    elif backend == "llama2_hf":
        tok = loaded.tokenizer
        bos = tok.bos_token_id if tok.bos_token_id is not None else 1
        eos = tok.eos_token_id if tok.eos_token_id is not None else 2
        start_ids = [int(bos)]
        stops = {int(eos)}
        encoded_items = [
            [int(bos)] + [int(x) for x in tok.encode(s, add_special_tokens=False)] + [int(eos)]
            for s in items
        ]
        prefix_ids = []

    elif backend == "stage46_causal":
        tok = loaded.tokenizer
        fmt = loaded.spec.input_format
        prefix_ids = list(ch.row_prefix_ids(loaded))
        if fmt == "n_to_c_control":
            # Exact historical E11 ProGen2 grammar: `1 SEQ 2 <|eos|>` + `1`
            d1 = _token_id_or_none(tok, "1")
            d2 = _token_id_or_none(tok, "2")
            doc_end = _token_id_or_none(tok, "<|eos|>")
            assert d1 is not None and d2 is not None and doc_end is not None
            start_ids = [d1]
            stops = {d2, doc_end}
            encoded_items = [
                list(ch.item_ids(loaded, s, modality="protein")) + [d2, doc_end] for s in items
            ]
        elif fmt == "fasta_wrapped":
            # ProtGPT2: `<|endoftext|>\nSEQ` items + `<|endoftext|>\n` start
            start_ids = [int(x) for x in tok("<|endoftext|>\n", add_special_tokens=False)["input_ids"]]
            eot = int(tok.eos_token_id)
            stops = {eot}
            encoded_items = [list(ch.item_ids(loaded, s, modality="protein")) for s in items]
        elif fmt == "gmask_sop_eos":
            # ProteinGLM-7B-CLM
            eos = int(tok.eos_token_id) if tok.eos_token_id is not None else _token_id_or_none(tok, "<eos>")
            assert eos is not None
            probe = list(ch.item_ids(loaded, "A", modality="protein"))
            start_ids = probe[:-1] if len(probe) >= 2 else probe
            stops = {int(eos)}
            encoded_items = [list(ch.item_ids(loaded, s, modality="protein")) + [int(eos)] for s in items]
        elif fmt == "eos_bounded_seq":
            # RITA-XL: `<EOS> SEQ <EOS>` + `<EOS>`
            eos_id = _token_id_or_none(tok, "<EOS>")
            assert eos_id is not None
            start_ids = [eos_id]
            stops = {eos_id}
            encoded_items = [
                list(ch.item_ids(loaded, s, modality="protein")) + [eos_id] for s in items
            ]
        elif model_id.startswith("galactica-"):
            start_ids = [int(x) for x in tok("[START_AMINO]", add_special_tokens=False)["input_ids"]]
            end_id = _token_id_or_none(tok, "[END_AMINO]")
            stops = {x for x in (end_id, tok.eos_token_id) if x is not None}
            encoded_items = [list(ch.item_ids(loaded, s, modality="protein")) for s in items]
        elif model_id == "instructprotein":
            start_ids = [int(x) for x in tok("</s><protein>", add_special_tokens=False)["input_ids"]]
            end_id = _token_id_or_none(tok, "</protein>")
            stops = {x for x in (end_id, tok.eos_token_id) if x is not None}
            encoded_items = [list(ch.item_ids(loaded, s, modality="protein")) for s in items]
        elif model_id in ("prollama", "prollama-stage-1"):
            start_ids = [int(x) for x in tok("Seq=<", add_special_tokens=False)["input_ids"]]
            gt_id = _token_id_or_none(tok, ">")
            stops = {x for x in (gt_id, tok.eos_token_id) if x is not None}
            encoded_items = [list(ch.item_ids(loaded, s, modality="protein")) for s in items]
        elif model_id == "gpt2-large":
            eot = int(tok.eos_token_id)
            start_ids = [eot]
            stops = {eot}
            encoded_items = [
                [int(x) for x in tok(s, add_special_tokens=False)["input_ids"]] + [eot] for s in items
            ]
        else:
            raise ValueError(f"{model_id}: unhandled stage46 prompt format {fmt!r}")
    else:
        raise ValueError(f"{model_id}: unsupported causal backend {backend!r}")

    # Trim trailing context items one by one if prompt leaves < min_headroom tokens.
    used_items = list(encoded_items)
    while True:
        flat_ctx = [tid for chunk in used_items for tid in chunk]
        prompt = prefix_ids + flat_ctx + start_ids
        if budget - len(prompt) >= min_headroom or not used_items:
            return prompt, stops, len(used_items)
        used_items.pop()


@torch.no_grad()
def sample_causal_continuations(
    *,
    model_id: str,
    backend: str,
    loaded: Any,
    prompt_ids: Sequence[int],
    stop_ids: set[int],
    count: int = ATTEMPTS_PER_CELL,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = 32,
    use_cache: bool = True,
    seed: int = GENERATION_SEED_BASE,
    temperature: float = 1.0,
    top_k: int = 950,
    top_p: float = 0.95,
) -> list[str]:
    """Batched continuation decoding for all causal backends (ProGen2/3, ProtGPT2/3, Dayhoff,
    ProteinGLM, RITA-XL, Galactica, InstructProtein, ProLLaMA, GPT-2, LLaMA-2).
    """
    model = loaded.model
    tok = loaded.tokenizer
    device = str(loaded.device)
    is_progen3 = backend == "progen3_causal"
    init_cache_fn = getattr(loaded, "_init_causal_cache", None)
    prec_ctx = loaded._precision_ctx if hasattr(loaded, "_precision_ctx") else None
    supports_cpos = (not is_progen3) and _supports_kwarg(model, "cache_position")

    raw_outputs: list[str] = []
    bsz_max = max(1, int(batch_size))

    if model_id == "rita-xl" and str(device).startswith("cuda"):
        ctx_mgr = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        ctx_mgr = prec_ctx() if prec_ctx is not None else torch.no_grad()
    gen_dev = torch.device(device)
    with ctx_mgr:
        for start in range(0, int(count), bsz_max):
            chunk_n = min(bsz_max, int(count) - start)
            batch_gen = torch.Generator(device=gen_dev).manual_seed(int(seed) + start)
            cur = torch.tensor([list(prompt_ids)] * chunk_n, dtype=torch.long, device=device)
            rows: list[list[int]] = [[] for _ in range(chunk_n)]
            live = [True] * chunk_n
            pad_tok = next(iter(stop_ids)) if stop_ids else int(prompt_ids[-1])

            past = init_cache_fn(batch_size=chunk_n) if (use_cache and init_cache_fn is not None) else None
            prefilled = False
            seq_pos = 0

            for _step in range(int(max_new_tokens)):
                inp = cur if (not use_cache or not prefilled) else cur[:, -1:]
                width = int(inp.shape[1])
                if is_progen3:
                    pos_ids = (
                        torch.arange(seq_pos, seq_pos + width, dtype=torch.long, device=device)
                        .unsqueeze(0)
                        .expand(chunk_n, width)
                    )
                    seq_ids = torch.zeros((chunk_n, width), dtype=torch.long, device=device)
                    out = model(
                        input_ids=inp,
                        position_ids=pos_ids,
                        sequence_ids=seq_ids,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    past = out.past_key_values
                    prefilled = past is not None
                    seq_pos += width
                elif use_cache:
                    fwd_kwargs: dict[str, Any] = {
                        "input_ids": inp,
                        "past_key_values": past,
                        "use_cache": True,
                    }
                    if supports_cpos:
                        fwd_kwargs["cache_position"] = torch.arange(
                            seq_pos, seq_pos + width, dtype=torch.long, device=device
                        )
                    out = model(**fwd_kwargs)
                    past = out.past_key_values
                    if past is not None:
                        prefilled = True
                        seq_pos += width
                    else:
                        prefilled = False
                        seq_pos = 0
                else:
                    out = model(input_ids=cur, use_cache=False)

                logits = out.logits[:, -1, :]
                sampled_ids = _sample_next_token_ids(
                    logits,
                    generator=batch_gen,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                ).tolist()
                next_ids: list[int] = []
                for row in range(chunk_n):
                    if not live[row]:
                        next_ids.append(pad_tok)
                        continue
                    nxt = int(sampled_ids[row])
                    rows[row].append(nxt)
                    next_ids.append(nxt)
                    if nxt in stop_ids:
                        live[row] = False
                if not any(live):
                    break
                cur = torch.cat(
                    [cur, torch.tensor(next_ids, dtype=torch.long, device=device).unsqueeze(1)],
                    dim=1,
                )

            for row_toks in rows:
                if hasattr(tok, "decode"):
                    try:
                        text = tok.decode(row_toks, skip_special_tokens=False)
                    except TypeError:
                        text = tok.decode(row_toks)
                else:
                    text = "".join(tok.id_to_token(t) or "" for t in row_toks)
                raw_outputs.append(str(text))

    return raw_outputs


def build_diffusion_prefix_and_span(
    *,
    cls_id: int,
    eos_id: int,
    mask_id: int,
    aa_to_id: Mapping[str, int],
    context_sequences: Sequence[str],
    target_length: int,
    budget: int = BUDGET_TOKENS,
    min_gen_length: int = MIN_PRODUCT_RESIDUES,
) -> tuple[list[int], int, int, int]:
    """Pack ``[<cls> item_1 <eos> ... <eos> <cls> <mask>^{L_gen} <eos>]`` inside ``budget``.

    Returns ``(template_ids, span_start, gen_length, items_used)``.
    """
    encoded_items = [[aa_to_id[aa] for aa in s if aa in aa_to_id] for s in context_sequences]
    used = list(encoded_items)

    def _prefix_tokens(item_list: list[list[int]]) -> list[int]:
        if not item_list:
            return [int(cls_id)]
        toks = [int(cls_id)]
        for it in item_list:
            toks.extend(it)
            toks.append(int(eos_id))
        toks.append(int(cls_id))
        return toks

    # Trim context items until full target_length fits (or at least min_gen_length fits).
    while used:
        pref = _prefix_tokens(used)
        if len(pref) + int(target_length) + 1 <= int(budget):
            break
        used.pop()

    pref = _prefix_tokens(used)
    available = int(budget) - len(pref) - 1
    gen_len = min(int(target_length), available)
    if gen_len < min_gen_length:
        raise ValueError(
            f"Diffusion continuation headroom {gen_len} < {min_gen_length} inside budget={budget}"
        )
    template = pref + [int(mask_id)] * gen_len + [int(eos_id)]
    return template, len(pref), gen_len, len(used)


@torch.no_grad()
def sample_dplm_continuations(
    *,
    adapter: Any,
    context_sequences: Sequence[str],
    target_length: int,
    count: int = ATTEMPTS_PER_CELL,
    batch_size: int = 32,
    budget: int = BUDGET_TOKENS,
    seed: int = GENERATION_SEED_BASE,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> tuple[list[str], int, int]:
    """Batched DPLM ``uniform_random`` unmasking of the target continuation span.

    Context residues remain unmasked and fixed throughout; only ``[span_start : span_start + L_gen)``
    is unmasked along a linear uniform-random schedule (``steps = min(64, L_gen)``).
    """
    tok = adapter.tokenizer
    cls_id = int(tok.cls_token_id if getattr(tok, "cls_token_id", None) is not None else 0)
    eos_id = int(tok.eos_token_id if getattr(tok, "eos_token_id", None) is not None else 2)
    template, span_start, gen_len, items_used = build_diffusion_prefix_and_span(
        cls_id=cls_id,
        eos_id=eos_id,
        mask_id=int(adapter.mask_id),
        aa_to_id=adapter.aa_to_id,
        context_sequences=context_sequences,
        target_length=target_length,
        budget=budget,
    )
    device = str(adapter.device)
    aa20_ids = [int(tid) for tid in adapter.aa20_ids]
    aa20_t = torch.tensor(aa20_ids, dtype=torch.long, device=device)
    total_steps = max(1, min(DIFFUSION_MAX_STEPS, gen_len))
    sequences: list[str] = []

    with adapter._precision_ctx():
        for start in range(0, int(count), max(1, int(batch_size))):
            chunk_n = min(int(batch_size), int(count) - start)
            seeds = [int(seed) + start + i for i in range(chunk_n)]
            gens = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
            ids = torch.tensor([template] * chunk_n, dtype=torch.long, device=device)
            orders = [torch.randperm(gen_len, generator=gens[r]).tolist() for r in range(chunk_n)]
            cursors = [0] * chunk_n
            chosen_codes = np.full((chunk_n, gen_len), -1, dtype=np.int64)

            for step in range(1, total_steps + 1):
                target_done = gen_len if step == total_steps else int(math.ceil(gen_len * step / total_steps))
                if all(c >= gen_len for c in cursors):
                    break
                logits = adapter._forward_logits(ids)  # (B, T, V)
                span_logits = logits[:, span_start : span_start + gen_len, :].index_select(-1, aa20_t).float()
                scores = _filtered_sampling_scores(
                    span_logits, temperature=temperature, top_k=0, top_p=top_p
                )
                probs = torch.softmax(scores, dim=-1).cpu()
                for row in range(chunk_n):
                    c0 = cursors[row]
                    c1 = min(gen_len, max(c0 + 1, target_done))
                    for pos in orders[row][c0:c1]:
                        aa_idx = int(torch.multinomial(probs[row, pos], 1, generator=gens[row]).item())
                        ids[row, span_start + pos] = aa20_ids[aa_idx]
                        chosen_codes[row, pos] = aa_idx
                    cursors[row] = c1

            # Invariant check: context prefix tokens were never modified.
            if span_start > 0:
                ref_prefix = torch.tensor(template[:span_start], dtype=torch.long, device=device)
                if not torch.equal(ids[:, :span_start], ref_prefix.unsqueeze(0).expand(chunk_n, -1)):
                    raise RuntimeError("DPLM continuation modified fixed context prefix tokens")
            for row in range(chunk_n):
                sequences.append("".join(AA20[int(c)] for c in chosen_codes[row]))

    return sequences, len(template) - gen_len - 1, items_used


@torch.no_grad()
def sample_esm3_continuations(
    *,
    handle: Any,
    context_sequences: Sequence[str],
    target_length: int,
    count: int = ATTEMPTS_PER_CELL,
    batch_size: int = 32,
    budget: int = BUDGET_TOKENS,
    seed: int = GENERATION_SEED_BASE,
    temperature: float = 1.0,
    top_p: float = 0.95,
    forward_fn: Any = None,
) -> tuple[list[str], int, int]:
    """Batched ESM3-1.4B sequence-only cosine-schedule unmasking of the target continuation span."""
    from ..esm3_ext import esm3_adapter as e3

    template, span_start, gen_len, items_used = build_diffusion_prefix_and_span(
        cls_id=int(handle.bos_id),
        eos_id=int(handle.eos_id),
        mask_id=int(handle.mask_id),
        aa_to_id=handle.aa_to_id,
        context_sequences=context_sequences,
        target_length=target_length,
        budget=budget,
    )
    device = handle.device
    aa20_ids = [int(tid) for tid in handle.aa20_token_ids]
    aa20_t = torch.tensor(aa20_ids, dtype=torch.long, device=device)
    total_steps = max(1, min(DIFFUSION_MAX_STEPS, gen_len))
    step_counts = e3._cosine_unmask_steps(gen_len, total_steps)
    fwd = e3.forward_tracks if forward_fn is None else forward_fn
    sequences: list[str] = []

    for start in range(0, int(count), max(1, int(batch_size))):
        chunk_n = min(int(batch_size), int(count) - start)
        seeds = [int(seed) + start + i for i in range(chunk_n)]
        gens = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
        ids = torch.tensor([template] * chunk_n, dtype=torch.long, device=device)
        pending: list[list[int]] = [list(range(gen_len)) for _ in range(chunk_n)]
        chosen_codes = np.full((chunk_n, gen_len), -1, dtype=np.int64)

        for n_commit in step_counts:
            if not pending[0]:
                break
            seq_logits = fwd(handle, sequence_tokens=ids).sequence_logits.float()
            span_logits = seq_logits[:, span_start : span_start + gen_len, :].index_select(-1, aa20_t)
            scores = _filtered_sampling_scores(
                span_logits, temperature=temperature, top_k=0, top_p=top_p
            )
            probs = torch.softmax(scores, dim=-1).cpu()
            for row in range(chunk_n):
                pend = pending[row]
                if not pend:
                    continue
                row_probs = probs[row, pend]  # (P, 20)
                sampled = torch.multinomial(row_probs, 1, generator=gens[row]).squeeze(-1)
                conf = row_probs.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
                k_take = min(int(n_commit), len(pend))
                top_local = torch.topk(conf, k=k_take, largest=True).indices.tolist()
                chosen = [(pend[i], int(sampled[i])) for i in top_local]
                for i in sorted(top_local, reverse=True):
                    pend.pop(i)
                for pos, aa_idx in chosen:
                    ids[row, span_start + pos] = aa20_ids[aa_idx]
                    chosen_codes[row, pos] = aa_idx

        if span_start > 0:
            ref_prefix = torch.tensor(template[:span_start], dtype=torch.long, device=device)
            if not torch.equal(ids[:, :span_start], ref_prefix.unsqueeze(0).expand(chunk_n, -1)):
                raise RuntimeError("ESM3 continuation modified fixed context prefix tokens")
        for row in range(chunk_n):
            sequences.append("".join(AA20[int(c)] for c in chosen_codes[row]))

    return sequences, len(template) - gen_len - 1, items_used


# --------------------------------------------------------------------------- #
# H200 Model Loaders
# --------------------------------------------------------------------------- #


def _resolve_model_subdir(dirname: str, model_root: Path | str | None = None) -> Path:
    candidates: list[Path] = []
    if model_root is not None and str(model_root).strip():
        candidates.append(Path(model_root) / dirname)
    for var in ("MODEL_ROOT", "GENERAL_PLM_MODEL_ROOT", "TEXT_MODEL_BASE_DIR", "MODEL_BASE_DIR"):
        v = os.environ.get(var, "").strip()
        if v:
            candidates.append(Path(v) / dirname)
    candidates.extend(
        [
            Path("/gpfs/jiaotongdamoxing/zhk_zip/models") / dirname,
            Path("/Data/public/models_R2") / dirname,
            Path("/Data/public") / dirname,
        ]
    )
    for cand in candidates:
        if cand.is_dir() and (cand / "config.json").is_file():
            return cand.resolve()
    raise FileNotFoundError(f"Could not resolve checkpoint {dirname!r}; checked {[str(c) for c in candidates]}")


def load_continuation_model(
    model_id: str,
    *,
    device: str = "cuda:0",
    dtype_override: str | None = None,
    model_root: Path | str | None = None,
    esm3_model_dir: Path | str | None = None,
) -> tuple[str, Any, str]:
    """Load one of the 31 R4-Continuation models offline for GPU generation."""
    spec = R4_MODEL_MAP[model_id]
    dtype = dtype_override or spec.default_dtype

    if spec.backend == "stage46_causal":
        _bf16_direct_dirs = {
            "galactica-1.3b": ("galactica-1.3b", False),
            "galactica-6.7b": ("galactica-6.7b", False),
            "galactica-30b": ("galactica-30b", False),
        }
        if model_id in _bf16_direct_dirs and dtype == "bfloat16":
            # Load heavy Galactica checkpoints directly in bfloat16 for H200 Tensor Core speed & headroom.
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dirname, trust_rc = _bf16_direct_dirs[model_id]
            ckpt = _resolve_model_subdir(dirname, model_root=model_root)
            tok = AutoTokenizer.from_pretrained(
                str(ckpt), local_files_only=True, trust_remote_code=trust_rc
            )
            model = (
                AutoModelForCausalLM.from_pretrained(
                    str(ckpt),
                    local_files_only=True,
                    trust_remote_code=trust_rc,
                    torch_dtype=torch.bfloat16,
                )
                .eval()
                .to(device)
            )
            handle = ch.tokenizer_arm(model_id)
            from ..core.arms import Arm

            return (
                spec.backend,
                Arm(spec=handle.spec, model=model, tokenizer=tok, device=device, dtype=dtype),
                dtype,
            )

        import importlib.util

        stage46_path = REPO_ROOT / "scripts/capability/stages/context_homologue.py"
        sp = importlib.util.spec_from_file_location("stage46_r4cont", stage46_path)
        if sp is None or sp.loader is None:
            raise RuntimeError(f"Could not load stage46 helper from {stage46_path}")
        stage46 = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(stage46)
        # Load via stage46 (which carries ProteinGLM deepspeed stub & ProGen2 trust_remote_code);
        # cast to bfloat16 after load if stage46 only accepts float32 for that arm.
        if model_id in ("proteinglm-7b-clm", "progen2-base") and dtype == "bfloat16":
            arm = stage46.load_scorable_arm(model_id, device=device, dtype="float32")
            arm.model.to(dtype=torch.bfloat16)
            from ..core.arms import Arm

            return (
                spec.backend,
                Arm(spec=arm.spec, model=arm.model, tokenizer=arm.tokenizer, device=device, dtype="bfloat16"),
                "bfloat16",
            )
        arm = stage46.load_scorable_arm(model_id, device=device, dtype=dtype)
        return spec.backend, arm, dtype

    if spec.backend in ("general_plm_causal", "multiseq_icl_causal", "dplm_diffusion"):
        from ..general_plm.adapters import load_general_plm

        adapter = load_general_plm(spec.arm_id, device=device, dtype=dtype, model_root=model_root)
        return spec.backend, adapter, adapter.dtype

    if spec.backend == "progen3_causal":
        torch_dt = torch.bfloat16
        if model_id in ("progen3-112m", "progen3-3b"):
            from ..models.progen3 import load_progen3

            ckpt = _resolve_model_subdir(model_id, model_root=model_root)
            return spec.backend, load_progen3(ckpt, device=device, dtype=torch_dt), "bfloat16"
        from ..general_plm.adapters import load_general_plm

        adapter = load_general_plm(model_id, device=device, dtype="bfloat16", model_root=model_root)
        return spec.backend, adapter.handle, "bfloat16"

    if spec.backend == "esm3_diffusion":
        from ..esm3_ext.esm3_adapter import load_esm3_offline

        resolved_dir = esm3_model_dir
        if resolved_dir is None and model_root is not None:
            cand = Path(model_root) / "esm3-sm-open-v1"
            if cand.is_dir():
                resolved_dir = cand
        return spec.backend, load_esm3_offline(model_dir=resolved_dir, device=device, dtype=dtype), dtype

    if spec.backend == "llama2_hf":
        from types import SimpleNamespace
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ckpt = _resolve_model_subdir("Llama-2-7b-hf", model_root=model_root)
        tok = AutoTokenizer.from_pretrained(str(ckpt), local_files_only=True)
        torch_dt = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        model = (
            AutoModelForCausalLM.from_pretrained(
                str(ckpt), local_files_only=True, torch_dtype=torch_dt
            )
            .eval()
            .to(device)
        )
        return spec.backend, SimpleNamespace(model=model, tokenizer=tok, device=device), dtype

    raise ValueError(f"{model_id}: unsupported backend {spec.backend!r}")


def generate_model_attempts(
    *,
    model_id: str,
    backend: str,
    loaded: Any,
    plans: Mapping[str, Mapping[str, Any]],
    attempts_per_cell: int = ATTEMPTS_PER_CELL,
    budget: int = BUDGET_TOKENS,
    seed_base: int = GENERATION_SEED_BASE,
) -> list[dict[str, Any]]:
    """Generate all 80 ``(target_id, condition)`` cells for one model."""
    spec = R4_MODEL_MAP[model_id]
    cells = required_target_condition_cells()
    records: list[dict[str, Any]] = []
    t_gen_start = time.monotonic()

    for cell_idx, (tid, cond) in enumerate(cells, start=1):
        plan = plans[tid]
        ctx_items = plan["items"].get(cond, [])
        ctx_seqs = [str(it["sequence"]) for it in ctx_items]
        ctx_subjects = [str(it["subject"]) for it in ctx_items]
        c_seed = cell_seed(model_id, tid, cond, base=seed_base)

        if backend == "dplm_diffusion":
            seqs, prompt_toks, items_used = sample_dplm_continuations(
                adapter=loaded,
                context_sequences=ctx_seqs,
                target_length=int(plan["wildtype_length"]),
                count=attempts_per_cell,
                batch_size=spec.default_batch_size,
                budget=budget,
                seed=c_seed,
            )
            used_ctx_seqs = ctx_seqs[:items_used]
            used_ctx_subjs = ctx_subjects[:items_used]
            for idx, seq in enumerate(seqs):
                copy_stats = H.copy_statistics(seq, used_ctx_seqs)
                records.append(
                    {
                        "schema_version": ATTEMPT_SCHEMA,
                        "attempt_id": f"{model_id}|{tid}|{cond}|{idx:04d}",
                        "model": model_id,
                        "target_id": tid,
                        "cluster": plan["cluster"],
                        "condition": cond,
                        "declared_condition": CONDITION_TO_E09_BIN[cond],
                        "attempt_index": idx,
                        "sequence": seq,
                        "sequence_sha256": hashlib.sha256(seq.encode("utf-8")).hexdigest(),
                        "residues": len(seq),
                        "stop_status": "native_terminal" if len(seq) >= MIN_PRODUCT_RESIDUES else "empty_or_noncanonical",
                        "prompt_tokens": int(prompt_toks),
                        "context_items_used": int(items_used),
                        "context_subjects": used_ctx_subjs,
                        "shannon_H_bits": shannon_entropy_bits(seq),
                        "distinct_5mer": distinct_kmer_fraction(seq, 5),
                        **copy_stats,
                    }
                )
        elif backend == "esm3_diffusion":
            seqs, prompt_toks, items_used = sample_esm3_continuations(
                handle=loaded,
                context_sequences=ctx_seqs,
                target_length=int(plan["wildtype_length"]),
                count=attempts_per_cell,
                batch_size=spec.default_batch_size,
                budget=budget,
                seed=c_seed,
            )
            used_ctx_seqs = ctx_seqs[:items_used]
            used_ctx_subjs = ctx_subjects[:items_used]
            for idx, seq in enumerate(seqs):
                copy_stats = H.copy_statistics(seq, used_ctx_seqs)
                records.append(
                    {
                        "schema_version": ATTEMPT_SCHEMA,
                        "attempt_id": f"{model_id}|{tid}|{cond}|{idx:04d}",
                        "model": model_id,
                        "target_id": tid,
                        "cluster": plan["cluster"],
                        "condition": cond,
                        "declared_condition": CONDITION_TO_E09_BIN[cond],
                        "attempt_index": idx,
                        "sequence": seq,
                        "sequence_sha256": hashlib.sha256(seq.encode("utf-8")).hexdigest(),
                        "residues": len(seq),
                        "stop_status": "native_terminal" if len(seq) >= MIN_PRODUCT_RESIDUES else "empty_or_noncanonical",
                        "prompt_tokens": int(prompt_toks),
                        "context_items_used": int(items_used),
                        "context_subjects": used_ctx_subjs,
                        "shannon_H_bits": shannon_entropy_bits(seq),
                        "distinct_5mer": distinct_kmer_fraction(seq, 5),
                        **copy_stats,
                    }
                )
        else:
            prompt_ids, stop_ids, items_used = build_causal_prompt_and_stops(
                model_id=model_id,
                backend=backend,
                loaded=loaded,
                context_sequences=ctx_seqs,
                budget=budget,
            )
            headroom = max(MIN_PRODUCT_RESIDUES, int(budget) - len(prompt_ids))
            max_new = min(int(spec.max_new_tokens), headroom)
            raws = sample_causal_continuations(
                model_id=model_id,
                backend=backend,
                loaded=loaded,
                prompt_ids=prompt_ids,
                stop_ids=stop_ids,
                count=attempts_per_cell,
                max_new_tokens=max_new,
                batch_size=spec.default_batch_size,
                use_cache=spec.use_cache,
                seed=c_seed,
            )
            used_ctx_seqs = ctx_seqs[:items_used]
            used_ctx_subjs = ctx_subjects[:items_used]
            for idx, raw in enumerate(raws):
                seq, stop_st = extract_canonical_residues(model_id, raw)
                copy_stats = H.copy_statistics(seq, used_ctx_seqs)
                records.append(
                    {
                        "schema_version": ATTEMPT_SCHEMA,
                        "attempt_id": f"{model_id}|{tid}|{cond}|{idx:04d}",
                        "model": model_id,
                        "target_id": tid,
                        "cluster": plan["cluster"],
                        "condition": cond,
                        "declared_condition": CONDITION_TO_E09_BIN[cond],
                        "attempt_index": idx,
                        "sequence": seq,
                        "sequence_sha256": hashlib.sha256(seq.encode("utf-8")).hexdigest() if seq else "",
                        "residues": len(seq),
                        "stop_status": stop_st,
                        "prompt_tokens": len(prompt_ids),
                        "max_new_tokens": int(max_new),
                        "context_items_used": int(items_used),
                        "context_subjects": used_ctx_subjs,
                        "shannon_H_bits": shannon_entropy_bits(seq),
                        "distinct_5mer": distinct_kmer_fraction(seq, 5),
                        **copy_stats,
                    }
                )
        if cell_idx % 16 == 0 or cell_idx == len(cells):
            elapsed = time.monotonic() - t_gen_start
            print(
                f"[r4_cont:progress] {model_id}: {cell_idx}/{len(cells)} cells "
                f"({elapsed:.1f}s elapsed, {elapsed / cell_idx:.2f}s/cell)",
                flush=True,
            )
    return records


def select_fold_candidates(
    attempts: Sequence[Mapping[str, Any]],
    *,
    max_per_cell: int = FOLD_CANDIDATES_PER_CELL,
) -> list[dict[str, Any]]:
    """Deterministically select up to ``max_per_cell`` valid candidates per ``(model, target, condition)``.

    Prefers non-copy (by LCS/k-mer), ``H >= 2.5`` bits, ``residues >= 32`` products; falls back
    to any non-copy ``residues >= 32``, then any ``residues >= 32`` product.
    """
    by_cell: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in attempts:
        if int(row.get("residues", 0)) < MIN_PRODUCT_RESIDUES:
            continue
        seq = str(row.get("sequence") or "")
        if not seq or set(seq) - AA20_SET:
            continue
        by_cell[(str(row["model"]), str(row["target_id"]), str(row["condition"]))].append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(by_cell):
        pool = sorted(by_cell[key], key=lambda r: int(r["attempt_index"]))
        tier1 = [
            r
            for r in pool
            if not H.copy_verdict(r)["is_copy"]
            and float(r.get("shannon_H_bits", 0.0)) >= FOLD_SELECT_ENTROPY_THRESHOLD
        ]
        tier2 = [r for r in pool if not H.copy_verdict(r)["is_copy"]]
        chosen = list(tier1[:max_per_cell])
        seen_ids = {r["attempt_id"] for r in chosen}
        for tier in (tier2, pool):
            for r in tier:
                if len(chosen) >= max_per_cell:
                    break
                if r["attempt_id"] not in seen_ids:
                    chosen.append(r)
                    seen_ids.add(r["attempt_id"])
        for r in chosen:
            selected.append(dict(r))
    return selected


# --------------------------------------------------------------------------- #
# Historical E11 <1e-12 Reproduction Gate
# --------------------------------------------------------------------------- #


def verify_historical_e11_reproduction(
    *,
    products_jsonl: Path | str = DEFAULT_E11_PRODUCTS_JSONL,
    annotation_json: Path | str = DEFAULT_E11_ANNOTATION_JSON,
    comparison_json: Path | str = DEFAULT_E11_COMPARISON_JSON,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Verify ``< 1e-12`` exact reproduction of historical E11 ``conditioned_generation_comparison.json``."""
    comp = json.loads(Path(comparison_json).read_text(encoding="utf-8"))
    ann = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    ann_by_id = {str(r["attempt_id"]): r for r in ann["annotations"]}

    merged: list[dict[str, Any]] = []
    with Path(products_jsonl).open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            a = ann_by_id.get(str(row["attempt_id"]), {})
            ident = a.get("context_alignment_identity")
            stats = row["copy_statistics"]
            verdict = H.copy_verdict(stats, identity_percent=ident)
            merged.append(
                {
                    **row,
                    **a,
                    "max_lcs_to_context": stats["max_lcs_to_context"],
                    "max_kmer_containment": stats["max_kmer_containment"],
                    "copy_verdict": verdict,
                }
            )

    max_abs_diff = 0.0
    n_checked = 0

    def _cmp(got: float | None, ref: float | None, label: str) -> None:
        nonlocal max_abs_diff, n_checked
        if got is None or ref is None:
            if got is not ref:
                raise AssertionError(f"E11 reproduction null mismatch at {label}: {got} vs {ref}")
            return
        d = abs(float(got) - float(ref))
        n_checked += 1
        if d > max_abs_diff:
            max_abs_diff = d
        if d > tolerance:
            raise AssertionError(
                f"E11 reproduction exceeded {tolerance} at {label}: got={got!r}, ref={ref!r}, diff={d:.3e}"
            )

    # 1. Recompute yields & context_identity per condition
    for cond in comp["conditions"]:
        rows = [r for r in merged if r["condition"] == cond]
        complete = [
            r
            for r in rows
            if r["stop_status"] == "native_terminal" and int(r["residues"]) >= MIN_PRODUCT_RESIDUES
        ]
        non_copy_complete = [r for r in complete if not r["copy_verdict"]["is_copy"]]
        ref_y = comp["yields"][cond]
        _cmp(len(rows), ref_y["attempts"], f"yields:{cond}:attempts")
        _cmp(len(complete), ref_y["complete_products"], f"yields:{cond}:complete_products")
        _cmp(len(complete) / len(rows), ref_y["complete_yield"], f"yields:{cond}:complete_yield")
        _cmp(
            len(non_copy_complete) / len(rows),
            ref_y["non_copy_complete_yield"],
            f"yields:{cond}:non_copy_complete_yield",
        )
        prods = [r for r in rows if int(r["residues"]) >= MIN_PRODUCT_RESIDUES]
        ref_ci = comp["context_identity"][cond]
        _cmp(len(prods), ref_ci["products"], f"ctx_id:{cond}:products")
        _cmp(
            float(np.mean([float(r["max_lcs_to_context"]) for r in prods])),
            ref_ci["max_lcs_to_context"]["mean"],
            f"ctx_id:{cond}:max_lcs_mean",
        )
        _cmp(
            float(np.mean([float(r["max_kmer_containment"]) for r in prods])),
            ref_ci["max_kmer_containment"]["mean"],
            f"ctx_id:{cond}:max_kmer_mean",
        )

    # 2. Recompute 10,000-draw paired cluster bootstrap yield_contrast via H.yield_contrast_panel
    recomputed_yc = H.yield_contrast_panel(
        merged,
        contrasts=H.STRUCTURE_CONTRASTS,
        group_key="target_id",
        draws=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
    )
    ref_cols = {c["contrast"]: c for c in comp["yield_contrast"]["columns"]}
    for col in recomputed_yc["columns"]:
        rc = ref_cols[col["contrast"]]
        _cmp(col["point"], rc["point"], f"yield_contrast:{col['contrast']}:point")
        _cmp(col["standard_error"], rc["standard_error"], f"yield_contrast:{col['contrast']}:se")
        _cmp(col["pointwise_interval"][0], rc["pointwise_interval"][0], f"yield_contrast:{col['contrast']}:pw_lo")
        _cmp(col["pointwise_interval"][1], rc["pointwise_interval"][1], f"yield_contrast:{col['contrast']}:pw_hi")
        _cmp(col["simultaneous_interval"][0], rc["simultaneous_interval"][0], f"yield_contrast:{col['contrast']}:sim_lo")
        _cmp(col["simultaneous_interval"][1], rc["simultaneous_interval"][1], f"yield_contrast:{col['contrast']}:sim_hi")

    return {
        "products_jsonl": str(products_jsonl),
        "annotation_json": str(annotation_json),
        "comparison_json": str(comparison_json),
        "n_scalars_checked": n_checked,
        "max_abs_diff": max_abs_diff,
        "tolerance": tolerance,
        "passed": max_abs_diff <= tolerance,
    }


# --------------------------------------------------------------------------- #
# CPU Annotation (Prompt-Copy DIAMOND, UniRef50 DIAMOND, Pinned Pfam hmmscan)
# --------------------------------------------------------------------------- #


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def run_prompt_context_diamond(
    attempts: Sequence[Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
    *,
    extra_plans: Sequence[Mapping[str, Mapping[str, Any]]] = (),
    repo_root: Path = REPO_ROOT,
    threads: int = 32,
) -> dict[tuple[str, str], float]:
    """Compute exact DIAMOND (e <= 0.001) alignment identity of valid products against their own prompt context items.

    Returns ``(product_sha256, subject_id) -> query_normalized_identity_pct``.
    """
    diamond_bin = repo_root / DIAMOND_BIN_REL
    if not diamond_bin.is_file():
        raise FileNotFoundError(f"DIAMOND binary missing at {diamond_bin}")

    # Collect unique context subjects across all plans (b1024 + b4096/b8192) & unique valid products.
    ctx_seqs: dict[str, str] = {}
    for plan_map in (plans, *extra_plans):
        for plan in plan_map.values():
            for items in plan["items"].values():
                for it in items:
                    ctx_seqs[str(it["subject"])] = str(it["sequence"])

    query_seqs: dict[str, str] = {}
    for row in attempts:
        if int(row.get("residues", 0)) < MIN_PRODUCT_RESIDUES:
            continue
        if not row.get("context_subjects"):
            continue
        seq = str(row["sequence"])
        sha = str(row.get("sequence_sha256") or hashlib.sha256(seq.encode("utf-8")).hexdigest())
        if sha not in query_seqs and not (set(seq) - AA20_SET):
            query_seqs[sha] = seq

    if not query_seqs or not ctx_seqs:
        return {}

    pair_ident: dict[tuple[str, str], float] = {}
    with tempfile.TemporaryDirectory(prefix="r4_cont_ctx_dmnd_") as tmpdir:
        tmp = Path(tmpdir)
        db_fa = tmp / "context_items.faa"
        db_path = tmp / "context_items.dmnd"
        q_fa = tmp / "queries.faa"
        out_tsv = tmp / "hits.tsv"

        with db_fa.open("w", encoding="utf-8") as fh:
            for sid, seq in sorted(ctx_seqs.items()):
                fh.write(f">{sid}\n{seq}\n")
        with q_fa.open("w", encoding="utf-8") as fh:
            for sha, seq in sorted(query_seqs.items()):
                fh.write(f">{sha}\n{seq}\n")

        subprocess.run(
            [str(diamond_bin), "makedb", "--in", str(db_fa), "-d", str(db_path), "--quiet"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                str(diamond_bin),
                "blastp",
                "-q",
                str(q_fa),
                "-d",
                str(db_path),
                "-o",
                str(out_tsv),
                "--outfmt",
                "6",
                "qseqid",
                "sseqid",
                "nident",
                "qlen",
                "evalue",
                "--evalue",
                "0.001",
                "--max-target-seqs",
                "25",
                "--more-sensitive",
                "--masking",
                "0",
                "--threads",
                str(int(threads)),
                "--quiet",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in out_tsv.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            qid, sid, nident_s, qlen_s, _ev = line.split("\t")
            pct = 100.0 * float(nident_s) / max(1.0, float(qlen_s))
            key = (qid, sid)
            if pct > pair_ident.get(key, 0.0):
                pair_ident[key] = pct
    return pair_ident


def ensure_uniref50_diamond_cache(
    unique_seqs: Mapping[str, str],
    *,
    cache_path: Path,
    seed_cache_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
    threads: int = 64,
) -> dict[str, dict[str, Any]]:
    """Run DIAMOND against ``uniref50_full.dmnd`` for uncached valid sequences."""
    cache: dict[str, dict[str, Any]] = {}
    if seed_cache_path is not None and seed_cache_path.is_file():
        cache.update(json.loads(seed_cache_path.read_text(encoding="utf-8")))
    if cache_path.is_file():
        cache.update(json.loads(cache_path.read_text(encoding="utf-8")))

    missing = sorted(
        (sha, seq)
        for sha, seq in unique_seqs.items()
        if sha not in cache and len(seq) >= MIN_PRODUCT_RESIDUES and not (set(seq) - AA20_SET)
    )
    if not missing:
        _atomic_write_json(cache_path, cache)
        return cache

    diamond_bin = repo_root / DIAMOND_BIN_REL
    db_path = repo_root / UNIREF50_DB_REL
    if not diamond_bin.is_file() or not db_path.is_file():
        raise FileNotFoundError(f"DIAMOND ({diamond_bin}) or UniRef50 DB ({db_path}) missing")

    sensitivity = "--very-sensitive" if len(missing) <= 12000 else "--more-sensitive"
    print(
        f"[r4_cont:diamond] querying {len(missing)} unique sequences ({sensitivity}, threads={threads})...",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="r4_cont_ur50_") as tmpdir:
        tmp = Path(tmpdir)
        q_fa = tmp / "queries.faa"
        out_tsv = tmp / "hits.tsv"
        with q_fa.open("w", encoding="utf-8") as fh:
            for sha, seq in missing:
                fh.write(f">{sha}\n{seq}\n")
        subprocess.run(
            [
                str(diamond_bin),
                "blastp",
                "-q",
                str(q_fa),
                "-d",
                str(db_path),
                "-o",
                str(out_tsv),
                "--outfmt",
                "6",
                "qseqid",
                "sseqid",
                "nident",
                "qlen",
                "evalue",
                "--max-target-seqs",
                "5",
                "--evalue",
                "1e-3",
                sensitivity,
                "--masking",
                "0",
                "--threads",
                str(int(threads)),
                "--quiet",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        best_hit: dict[str, tuple[float, str, float]] = {}
        for line in out_tsv.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            qid, sid, nident_s, qlen_s, ev_s = line.split("\t")
            pct = 100.0 * float(nident_s) / max(1.0, float(qlen_s))
            if qid not in best_hit or pct > best_hit[qid][0]:
                best_hit[qid] = (pct, sid, float(ev_s))
        for sha, _ in missing:
            if sha in best_hit:
                pct, sid, ev = best_hit[sha]
                band = (
                    "lt30_novel"
                    if pct < 30.0
                    else ("id30_70_remote" if pct < 70.0 else ("id70_95_close" if pct < 95.0 else "ge95_near_dup"))
                )
                cache[sha] = {
                    "identity_over_query": pct,
                    "subject": sid,
                    "evalue": ev,
                    "band": band,
                    "source": f"diamond_uniref50_{sensitivity.lstrip('-')}",
                }
            else:
                cache[sha] = {
                    "identity_over_query": 0.0,
                    "subject": None,
                    "evalue": None,
                    "band": "lt30_novel",
                    "source": f"diamond_uniref50_{sensitivity.lstrip('-')}",
                }

    _atomic_write_json(cache_path, cache)
    return cache


def ensure_pfam_hmmscan_cache(
    unique_seqs: Mapping[str, str],
    *,
    cache_path: Path,
    seed_cache_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
    workers: int = 32,
    threads_per_worker: int = 2,
) -> dict[str, dict[str, Any]]:
    """Run pinned Pfam-A ``hmmscan --cut_ga`` for uncached unique sequences."""
    cache: dict[str, dict[str, Any]] = {}
    if seed_cache_path is not None and seed_cache_path.is_file():
        cache.update(json.loads(seed_cache_path.read_text(encoding="utf-8")))
    if cache_path.is_file():
        cache.update(json.loads(cache_path.read_text(encoding="utf-8")))

    missing = sorted(
        (sha, seq)
        for sha, seq in unique_seqs.items()
        if sha not in cache and len(seq) >= 16 and not (set(seq) - AA20_SET)
    )
    if not missing:
        _atomic_write_json(cache_path, cache)
        return cache

    oracle_root = repo_root / ORACLE_ROOT_REL
    hmmscan_bin = oracle_root / HMMSCAN_RELATIVE
    pfam_db = oracle_root / PFAM_RELATIVE
    if not hmmscan_bin.is_file() or not pfam_db.is_file():
        raise FileNotFoundError(f"Pinned Pfam oracle missing under {oracle_root}")
    oracle = Oracle(hmmscan=hmmscan_bin, pfam=pfam_db, digests={}, root=oracle_root)

    n_shards = min(max(workers * 2, 1), len(missing))
    shards: list[list[tuple[str, str]]] = [[] for _ in range(n_shards)]
    for i, item in enumerate(missing):
        shards[i % n_shards].append(item)

    print(
        f"[r4_cont:hmmscan] scanning {len(missing)} unique sequences across {n_shards} shards "
        f"({workers} workers x {threads_per_worker} threads)...",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="r4_cont_hmmscan_") as tmpdir:
        tmp = Path(tmpdir)

        def _scan_one(idx: int) -> dict[str, dict[str, Any]]:
            items = shards[idx]
            fa = tmp / f"shard_{idx:03d}.faa"
            tbl = tmp / f"shard_{idx:03d}.tbl"
            domtbl = tmp / f"shard_{idx:03d}.domtbl"
            with fa.open("w", encoding="utf-8") as fh:
                for sha, seq in items:
                    fh.write(f">{sha}\n{seq}\n")
            cmd = scan_command(oracle, fa, tbl, domtbl, threads=threads_per_worker)
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"hmmscan shard {idx} failed: {res.stderr[-1500:]}")
            tbl_hits = parse_hmmscan_table(tbl)
            dom_hits = parse_domain_table(domtbl)
            local: dict[str, dict[str, Any]] = {}
            for sha, _ in items:
                fams = sorted({e["accession_unversioned"] for e in tbl_hits.get(sha, [])})
                dom_entries = dom_hits.get(sha, [])
                best_cov = float(best_profile_coverage(dom_entries) or 0.0) if dom_entries else 0.0
                any_fam = bool(fams)
                local[sha] = {
                    "any_family": any_fam,
                    "complete_domain": bool(any_fam and best_cov >= COMPLETE_DOMAIN_COVERAGE),
                    "best_profile_coverage": best_cov,
                    "families": fams,
                    "source": "hmmscan_pfam_a_cut_ga",
                }
            return local

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for shard_res in ex.map(_scan_one, range(len(shards))):
                cache.update(shard_res)

    _atomic_write_json(cache_path, cache)
    return cache


# --------------------------------------------------------------------------- #
# Unified 31-Model Statistical Aggregation
# --------------------------------------------------------------------------- #


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _paired_cluster_bootstrap(
    left_by_cluster: Mapping[str, float],
    right_by_cluster: Mapping[str, float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """10,000-draw paired cluster bootstrap for ``mean(left - right)`` over shared clusters."""
    shared = sorted(set(left_by_cluster) & set(right_by_cluster))
    n = len(shared)
    if n == 0:
        return {
            "n_clusters": 0,
            "point": None,
            "standard_error": None,
            "interval95": None,
            "status": "unestimable",
            "direction": "unestimable",
        }
    diffs = np.asarray([float(left_by_cluster[c]) - float(right_by_cluster[c]) for c in shared], dtype=np.float64)
    point = float(diffs.mean())
    if n < H.MINIMUM_BOOTSTRAP_UNITS:
        return {
            "n_clusters": n,
            "point": point,
            "standard_error": float(diffs.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "interval95": None,
            "status": "below_cluster_floor",
            "direction": "unresolved",
        }
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(draws), n))
    boot = diffs[idx].mean(axis=1)
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    direction = "above_zero" if lo > 0.0 else ("below_zero" if hi < 0.0 else "spans_zero")
    return {
        "n_clusters": n,
        "point": point,
        "standard_error": float(boot.std(ddof=1)),
        "interval95": [lo, hi],
        "status": "estimated",
        "direction": direction,
    }


def _safe_mean(vals: Sequence[float | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else None


def assemble_r4_continuation_summary(
    *,
    work_dir: Path | str = DEFAULT_WORK_DIR,
    homologs_json: Path | str = DEFAULT_HOMOLOGS_JSON,
    e11_products_jsonl: Path | str = DEFAULT_E11_PRODUCTS_JSONL,
    e11_annotation_json: Path | str = DEFAULT_E11_ANNOTATION_JSON,
    e11_comparison_json: Path | str = DEFAULT_E11_COMPARISON_JSON,
    repo_root: Path = REPO_ROOT,
    run_annotations: bool = True,
    allow_partial: bool = False,
    diamond_threads: int = 64,
    hmmscan_workers: int = 32,
) -> dict[str, Any]:
    """Assemble the complete unified 31-model R4-Continuation artifact."""
    e11_gate = verify_historical_e11_reproduction(
        products_jsonl=e11_products_jsonl,
        annotation_json=e11_annotation_json,
        comparison_json=e11_comparison_json,
    )
    contexts = json.loads(Path(homologs_json).read_text(encoding="utf-8"))
    plans = build_canonical_target_plans(contexts, budget=BUDGET_TOKENS)
    plans_b4096 = build_canonical_target_plans(contexts, budget=4096, max_items=8)
    plans_b8192 = build_canonical_target_plans(contexts, budget=8192, max_items=8)
    wt_len_by_tid = {tid: int(p["wildtype_length"]) for tid, p in plans.items()}

    wdir = Path(work_dir)
    gen_dir = wdir / "generated"
    fold_dir = wdir / "folded"
    cache_dir = wdir / "annotation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    attempts_by_model: dict[str, list[dict[str, Any]]] = {}
    folded_by_attempt_id: dict[str, dict[str, Any]] = {}
    if gen_dir.is_dir():
        for mid in R4_MODEL_IDS:
            p = gen_dir / f"{model_file_slug(mid)}.jsonl"
            if p.is_file():
                attempts_by_model[mid] = _read_jsonl(p)
    if fold_dir.is_dir():
        for p in sorted(fold_dir.glob("*.jsonl")):
            for row in _read_jsonl(p):
                folded_by_attempt_id[str(row["attempt_id"])] = row

    missing_models = [m for m in R4_MODEL_IDS if m not in attempts_by_model]
    if missing_models and not allow_partial:
        raise FileNotFoundError(
            f"Missing {len(missing_models)}/{len(R4_MODEL_IDS)} generated R4-Continuation models: "
            f"{missing_models[:8]}. Pass --allow-partial for interim assembly."
        )

    # Run / load CPU annotations (own-context DIAMOND, UniRef50 DIAMOND, Pfam hmmscan).
    all_attempts = [r for rows in attempts_by_model.values() for r in rows]
    ctx_pair_ident: dict[tuple[str, str], float] = {}
    ur50_cache: dict[str, dict[str, Any]] = {}
    pfam_cache: dict[str, dict[str, Any]] = {}

    # Target WT Pfam scan + valid non-copy product scan
    wt_sha_to_seq = {
        hashlib.sha256(str(p["wildtype"]).encode("utf-8")).hexdigest(): str(p["wildtype"])
        for p in plans.values()
    }
    wt_sha_by_tid = {
        tid: hashlib.sha256(str(p["wildtype"]).encode("utf-8")).hexdigest()
        for tid, p in plans.items()
    }

    if run_annotations and all_attempts:
        ctx_pair_ident = run_prompt_context_diamond(
            all_attempts,
            plans,
            extra_plans=(plans_b4096, plans_b8192),
            repo_root=repo_root,
            threads=min(32, diamond_threads),
        )
        valid_unique: dict[str, str] = {}
        non_copy_unique: dict[str, str] = dict(wt_sha_to_seq)
        for r in all_attempts:
            if int(r["residues"]) < MIN_PRODUCT_RESIDUES:
                continue
            seq = str(r["sequence"])
            sha = str(r.get("sequence_sha256") or hashlib.sha256(seq.encode("utf-8")).hexdigest())
            if set(seq) - AA20_SET:
                continue
            valid_unique[sha] = seq
            own_id = max(
                (ctx_pair_ident.get((sha, str(subj)), 0.0) for subj in r.get("context_subjects", ())),
                default=None,
            )
            if not H.copy_verdict(r, identity_percent=own_id)["is_copy"]:
                non_copy_unique[sha] = seq

        ur50_cache = ensure_uniref50_diamond_cache(
            valid_unique,
            cache_path=cache_dir / "diamond_uniref50_by_sha256.json",
            seed_cache_path=repo_root / EXP4_1_CACHE_DIR / "diamond_uniref50_by_sha256.json",
            repo_root=repo_root,
            threads=diamond_threads,
        )
        pfam_cache = ensure_pfam_hmmscan_cache(
            non_copy_unique,
            cache_path=cache_dir / "pfam_hmmscan_by_sha256.json",
            seed_cache_path=repo_root / EXP4_1_CACHE_DIR / "pfam_hmmscan_by_sha256.json",
            repo_root=repo_root,
            workers=hmmscan_workers,
        )
    elif run_annotations:
        pfam_cache = ensure_pfam_hmmscan_cache(
            wt_sha_to_seq,
            cache_path=cache_dir / "pfam_hmmscan_by_sha256.json",
            seed_cache_path=repo_root / EXP4_1_CACHE_DIR / "pfam_hmmscan_by_sha256.json",
            repo_root=repo_root,
            workers=min(8, hmmscan_workers),
        )

    target_wt_pfams: dict[str, set[str]] = {
        tid: set(pfam_cache.get(wt_sha_by_tid[tid], {}).get("families", ()))
        for tid in ALL_22_TARGETS
    }

    # Aggregate per model across both panels (Primary 12 and Twilight 12)
    models_out: dict[str, Any] = {}
    paradigm_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    regime_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def _fold_subset_summary(frows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "n_folded": len(frows),
            "mean_plddt": _safe_mean([x["plddt_mean"] for x in frows]),
            "mean_ptm": _safe_mean([x["ptm"] for x in frows]),
            "mean_pae": _safe_mean([x["mean_pae_angstrom"] for x in frows]),
            "mean_tm_score_to_target_wt": _safe_mean([x["tm_score_to_target_wt"] for x in frows]),
            "tm_ge_0_5_rate": float(np.mean([x["tm_ge_0_5"] for x in frows])) if frows else None,
        }

    for mid in R4_MODEL_IDS:
        if mid not in attempts_by_model:
            continue
        spec = R4_MODEL_MAP[mid]
        meta = ARM_METADATA[spec.arm_id]
        rows = attempts_by_model[mid]

        # Enrich every attempt with copy verdict, UniRef50, Pfam recall, and fold metrics
        enriched: list[dict[str, Any]] = []
        for r in rows:
            seq = str(r["sequence"])
            sha = str(r.get("sequence_sha256") or (hashlib.sha256(seq.encode("utf-8")).hexdigest() if seq else ""))
            subjs = r.get("context_subjects") or []
            own_id = (
                max((ctx_pair_ident.get((sha, str(s)), 0.0) for s in subjs), default=0.0)
                if subjs and ctx_pair_ident
                else (0.0 if subjs else None)
            )
            verdict = H.copy_verdict(r, identity_percent=own_id)
            is_copy = bool(verdict["is_copy"])
            valid_len = int(r["residues"]) >= MIN_PRODUCT_RESIDUES
            complete = valid_len and str(r["stop_status"]) == "native_terminal"
            non_copy_complete = complete and (not is_copy)
            nat_ent_non_copy = non_copy_complete and float(r["shannon_H_bits"]) >= NATURAL_ENTROPY_THRESHOLD

            pf_rec = pfam_cache.get(sha, {})
            prod_fams = set(pf_rec.get("families", ()))
            tgt_fams = target_wt_pfams.get(str(r["target_id"]), set())
            tgt_has_pfam = bool(tgt_fams)
            tgt_pfam_hit = bool(valid_len and (not is_copy) and tgt_has_pfam and (prod_fams & tgt_fams))

            ur_rec = ur50_cache.get(sha, {})
            ur_id = ur_rec.get("identity_over_query")
            ur_novel_lt30 = (float(ur_id) < 30.0) if ur_id is not None else None

            fold_rec = folded_by_attempt_id.get(str(r["attempt_id"]))
            plddt = float(fold_rec["plddt_mean"]) if fold_rec else None
            ptm = float(fold_rec["ptm"]) if fold_rec else None
            pae = float(fold_rec["mean_pae_angstrom"]) if fold_rec else None
            tm_wt = float(fold_rec["tm_score_to_target_wt"]) if fold_rec else None
            fold_high_qual = (
                bool(
                    (not is_copy)
                    and float(r["shannon_H_bits"]) >= NATURAL_ENTROPY_THRESHOLD
                    and plddt is not None
                    and plddt >= 70.0
                )
                if fold_rec
                else None
            )
            tm_recap = (tm_wt >= TM_RECAP_THRESHOLD) if tm_wt is not None else None

            enriched.append(
                {
                    **r,
                    "context_alignment_identity": own_id,
                    "is_copy": is_copy,
                    "complete": complete,
                    "non_copy_complete": non_copy_complete,
                    "natural_entropy_non_copy": nat_ent_non_copy,
                    "any_pfam": bool(pf_rec.get("any_family", False)) if valid_len and not is_copy else False,
                    "complete_pfam80": bool(pf_rec.get("complete_domain", False)) if valid_len and not is_copy else False,
                    "target_has_pfam": tgt_has_pfam,
                    "target_pfam_recalled": tgt_pfam_hit,
                    "uniref50_identity_pct": float(ur_id) if ur_id is not None else None,
                    "uniref50_novel_lt30": ur_novel_lt30,
                    "plddt_mean": plddt,
                    "ptm": ptm,
                    "mean_pae_angstrom": pae,
                    "tm_score_to_target_wt": tm_wt,
                    "folded_high_qual_non_copy": fold_high_qual,
                    "tm_ge_0_5": tm_recap,
                }
            )

        def _summarise_panel(panel_targets: Sequence[str], panel_conditions: Sequence[str]) -> dict[str, Any]:
            tset = set(panel_targets)
            sub = [x for x in enriched if str(x["target_id"]) in tset]
            per_cond: dict[str, Any] = {}
            by_cluster_metric: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

            for cond in panel_conditions:
                c_rows = [x for x in sub if x["condition"] == cond]
                n_att = len(c_rows)
                valid_rows = [x for x in c_rows if int(x["residues"]) >= MIN_PRODUCT_RESIDUES]
                pfam_elig = [x for x in c_rows if x["target_has_pfam"]]
                folded_rows = [x for x in c_rows if x["plddt_mean"] is not None]
                ur_rows = [x for x in valid_rows if x["uniref50_identity_pct"] is not None]

                matched_rows = [
                    x
                    for x in folded_rows
                    if 0.75 * wt_len_by_tid[str(x["target_id"])]
                    <= int(x["residues"])
                    <= 1.25 * wt_len_by_tid[str(x["target_id"])]
                ]
                unmatched_rows = [
                    x
                    for x in folded_rows
                    if not (
                        0.75 * wt_len_by_tid[str(x["target_id"])]
                        <= int(x["residues"])
                        <= 1.25 * wt_len_by_tid[str(x["target_id"])]
                    )
                ]
                band_16_128 = [x for x in folded_rows if 16 <= int(x["residues"]) <= 128]
                band_129_256 = [x for x in folded_rows if 129 <= int(x["residues"]) <= 256]
                band_257_512 = [x for x in folded_rows if 257 <= int(x["residues"]) <= 512]

                per_cond[cond] = {
                    "attempts": n_att,
                    "valid_ge32": len(valid_rows),
                    "complete_yield": float(np.mean([x["complete"] for x in c_rows])) if n_att else None,
                    "copy_rate_all": float(np.mean([x["is_copy"] for x in c_rows])) if n_att else None,
                    "copy_rate_valid": float(np.mean([x["is_copy"] for x in valid_rows])) if valid_rows else None,
                    "non_copy_complete_yield": float(np.mean([x["non_copy_complete"] for x in c_rows])) if n_att else None,
                    "natural_entropy_non_copy_yield": float(np.mean([x["natural_entropy_non_copy"] for x in c_rows])) if n_att else None,
                    "mean_shannon_H_bits": _safe_mean([x["shannon_H_bits"] for x in valid_rows]),
                    "mean_distinct_5mer": _safe_mean([x["distinct_5mer"] for x in valid_rows]),
                    "mean_max_lcs": _safe_mean([x["max_lcs_to_context"] for x in valid_rows]),
                    "target_pfam_recall_non_copy": float(np.mean([x["target_pfam_recalled"] for x in pfam_elig])) if pfam_elig else None,
                    "any_pfam_rate_all": float(np.mean([x["any_pfam"] for x in c_rows])) if n_att else None,
                    "complete_pfam80_rate_all": float(np.mean([x["complete_pfam80"] for x in c_rows])) if n_att else None,
                    "uniref50_novel_lt30_share": float(np.mean([x["uniref50_novel_lt30"] for x in ur_rows])) if ur_rows else None,
                    "uniref50_identity_median": float(np.median([x["uniref50_identity_pct"] for x in ur_rows])) if ur_rows else None,
                    "n_folded": len(folded_rows),
                    "mean_plddt": _safe_mean([x["plddt_mean"] for x in folded_rows]),
                    "mean_ptm": _safe_mean([x["ptm"] for x in folded_rows]),
                    "mean_pae": _safe_mean([x["mean_pae_angstrom"] for x in folded_rows]),
                    "folded_high_qual_non_copy_rate": float(np.mean([x["folded_high_qual_non_copy"] for x in folded_rows])) if folded_rows else None,
                    "mean_tm_score_to_target_wt": _safe_mean([x["tm_score_to_target_wt"] for x in folded_rows]),
                    "tm_ge_0_5_rate": float(np.mean([x["tm_ge_0_5"] for x in folded_rows])) if folded_rows else None,
                    "length_stratified_folding": {
                        "length_matched": _fold_subset_summary(matched_rows),
                        "unmatched": _fold_subset_summary(unmatched_rows),
                        "len_16_128": _fold_subset_summary(band_16_128),
                        "len_129_256": _fold_subset_summary(band_129_256),
                        "len_257_512": _fold_subset_summary(band_257_512),
                    },
                }

                # Cluster-level means for paired cluster bootstrap
                by_cl: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for x in c_rows:
                    by_cl[str(x["cluster"])].append(x)
                for cl, cl_rows in by_cl.items():
                    by_cluster_metric["non_copy_complete_yield"][cond][cl] = float(
                        np.mean([x["non_copy_complete"] for x in cl_rows])
                    )
                    by_cluster_metric["natural_entropy_non_copy_yield"][cond][cl] = float(
                        np.mean([x["natural_entropy_non_copy"] for x in cl_rows])
                    )
                    cl_pf = [x for x in cl_rows if x["target_has_pfam"]]
                    if cl_pf:
                        by_cluster_metric["target_pfam_recall_non_copy"][cond][cl] = float(
                            np.mean([x["target_pfam_recalled"] for x in cl_pf])
                        )
                    cl_fold = [x for x in cl_rows if x["plddt_mean"] is not None]
                    if cl_fold:
                        by_cluster_metric["mean_plddt"][cond][cl] = float(
                            np.mean([x["plddt_mean"] for x in cl_fold])
                        )
                        by_cluster_metric["mean_ptm"][cond][cl] = float(
                            np.mean([x["ptm"] for x in cl_fold])
                        )
                        by_cluster_metric["mean_tm_score_to_target_wt"][cond][cl] = float(
                            np.mean([x["tm_score_to_target_wt"] for x in cl_fold])
                        )
                        by_cluster_metric["tm_ge_0_5_rate"][cond][cl] = float(
                            np.mean([x["tm_ge_0_5"] for x in cl_fold])
                        )

            contrast_pairs = [
                (f"{c}_minus_unrelated", c, "unrelated")
                for c in panel_conditions
                if c not in ("no_context", "unrelated")
            ]
            contrasts: dict[str, dict[str, Any]] = {}
            for metric_name in (
                "non_copy_complete_yield",
                "natural_entropy_non_copy_yield",
                "target_pfam_recall_non_copy",
                "mean_plddt",
                "mean_ptm",
                "mean_tm_score_to_target_wt",
                "tm_ge_0_5_rate",
            ):
                contrasts[metric_name] = {}
                for cname, left_c, right_c in contrast_pairs:
                    contrasts[metric_name][cname] = _paired_cluster_bootstrap(
                        by_cluster_metric[metric_name].get(left_c, {}),
                        by_cluster_metric[metric_name].get(right_c, {}),
                    )
            return {
                "targets": list(panel_targets),
                "conditions": per_cond,
                "contrasts": contrasts,
            }

        entry = {
            "model_id": mid,
            "base_model_id": spec.arm_id,
            "display_name": (
                f"{meta.display_name} (b{spec.context_budget})"
                if spec.context_budget != BUDGET_TOKENS
                else meta.display_name
            ),
            "family": meta.family,
            "paradigm": meta.paradigm,
            "modality": meta.modality,
            "architecture": meta.architecture,
            "params_m": meta.params_m,
            "vocab_type": meta.vocab_type,
            "backend": spec.backend,
            "context_budget_tokens": int(spec.context_budget),
            "max_context_items": int(spec.max_context_items),
            "input_alignment_format": spec.input_alignment_format,
            "context_training_regime": spec.context_training_regime,
            "is_text_control": spec.is_text_control,
            "primary_12": _summarise_panel(PRIMARY_12_TARGETS, PRIMARY_CONDITIONS),
            "twilight_12": _summarise_panel(TWILIGHT_12_TARGETS, TWILIGHT_CONDITIONS),
        }
        models_out[mid] = entry
        paradigm_buckets[meta.paradigm].append(entry)
        regime_buckets[spec.context_training_regime].append(entry)

    def _summarize_bucket(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "n_models": len(entries),
            "models": [e["model_id"] for e in entries],
            "primary_close_minus_unrelated_non_copy_yield": _safe_mean(
                [e["primary_12"]["contrasts"]["non_copy_complete_yield"]["close_homolog_minus_unrelated"]["point"] for e in entries]
            ),
            "primary_close_minus_unrelated_pfam_recall": _safe_mean(
                [e["primary_12"]["contrasts"]["target_pfam_recall_non_copy"]["close_homolog_minus_unrelated"]["point"] for e in entries]
            ),
            "primary_remote_minus_unrelated_pfam_recall": _safe_mean(
                [e["primary_12"]["contrasts"]["target_pfam_recall_non_copy"]["remote_homolog_minus_unrelated"]["point"] for e in entries]
            ),
            "twilight_minus_unrelated_pfam_recall": _safe_mean(
                [e["twilight_12"]["contrasts"]["target_pfam_recall_non_copy"]["twilight_homolog_minus_unrelated"]["point"] for e in entries]
            ),
            "primary_close_minus_unrelated_tm": _safe_mean(
                [e["primary_12"]["contrasts"]["mean_tm_score_to_target_wt"]["close_homolog_minus_unrelated"]["point"] for e in entries]
            ),
            "primary_remote_minus_unrelated_tm": _safe_mean(
                [e["primary_12"]["contrasts"]["mean_tm_score_to_target_wt"]["remote_homolog_minus_unrelated"]["point"] for e in entries]
            ),
            "twilight_minus_unrelated_tm": _safe_mean(
                [e["twilight_12"]["contrasts"]["mean_tm_score_to_target_wt"]["twilight_homolog_minus_unrelated"]["point"] for e in entries]
            ),
        }

    paradigm_summary: dict[str, Any] = {
        par: _summarize_bucket(entries) for par, entries in sorted(paradigm_buckets.items())
    }
    regime_summary: dict[str, Any] = {
        reg: _summarize_bucket(entries) for reg, entries in sorted(regime_buckets.items())
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "budget_tokens": BUDGET_TOKENS,
            "extended_multiseq_budgets": {"dayhoff-3b-msa|b4096": 4096, "protgpt3-msa|b8192": 8192},
            "attempts_per_cell": ATTEMPTS_PER_CELL,
            "min_product_residues": MIN_PRODUCT_RESIDUES,
            "primary_12_targets": list(PRIMARY_12_TARGETS),
            "twilight_12_targets": list(TWILIGHT_12_TARGETS),
            "all_22_targets": list(ALL_22_TARGETS),
            "target_wt_pfam_families": {tid: sorted(fams) for tid, fams in sorted(target_wt_pfams.items())},
            "copy_rules": {
                "lcs_residues": H.COPY_LCS_RESIDUES,
                "kmer_containment": H.COPY_KMER_CONTAINMENT,
                "alignment_identity_percent": H.COPY_IDENTITY_PERCENT,
            },
            "length_stratification": {
                "length_matched_rule": "0.75 * L_wt <= L_prod <= 1.25 * L_wt",
                "unmatched_rule": "L_prod < 0.75 * L_wt or L_prod > 1.25 * L_wt",
                "length_bands_aa": {"len_16_128": [16, 128], "len_129_256": [129, 256], "len_257_512": [257, 512]},
            },
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "msa_transformer_applicability": {
            "model_id": "msa-transformer-100m",
            "input_alignment_format": "explicit_column_aligned",
            "status": "inapplicable_for_leak_free_free_continuation",
            "reason": (
                "MSA-Transformer-100M (aligned=True) constructs a 2D column-aligned MSA by running "
                "Needleman-Wunsch alignment of context homologs onto a known row-0 query sequence. "
                "In free de novo prefix continuation, the target sequence is unknown prior to generation; "
                "aligning homologs to the target wildtype would leak the target's exact length, indel "
                "pattern, and column correspondence. ProtGPT3-MSA and Dayhoff-3B-MSA therefore serve "
                "as the clean, leak-free multi-sequence ICL unaligned generative comparators."
            ),
        },
        "coverage": {
            "total_models": len(R4_CONTINUATION_MODELS),
            "n_single_chain_models": len(R4_SINGLE_CHAIN_MODEL_IDS),
            "n_multiseq_icl_entries": len(R4_MULTISEQ_ICL_MODEL_IDS),
            "completed_models": sorted(models_out.keys()),
            "n_completed_models": len(models_out),
            "pending_models": missing_models,
            "n_pending_models": len(missing_models),
            "n_folded_products": len(folded_by_attempt_id),
            "is_complete": len(missing_models) == 0,
        },
        "reproduction_gates": {
            "e11_historical_progen2_medium_20261011": e11_gate,
        },
        "models": models_out,
        "paradigm_summary": paradigm_summary,
        "regime_summary": regime_summary,
    }


def write_unified_artifact(payload: Mapping[str, Any], out_path: Path | str) -> tuple[Path, Path, str]:
    target = Path(out_path)
    _atomic_write_json(target, payload)
    digest = sha256_file(target)
    sha_path = Path(str(target) + ".sha256")
    tmp = Path(str(sha_path) + ".tmp")
    tmp.write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    os.replace(tmp, sha_path)
    return target, sha_path, digest
