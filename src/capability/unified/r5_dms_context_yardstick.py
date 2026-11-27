"""Unified Downstream-1 Yardstick: Homologous-Prefix DMS Mutation Ranking (Merged Result 5).

Unifies BOTH frozen DMS homolog-context sub-tests onto ONE cross-paradigm panel
spanning single-chain causal, single-chain bidirectional/masked/3Di/diffusion/3D,
multi-sequence unaligned ICL (ProtGPT3-MSA, Dayhoff-3B-MSA), 2D column-aligned vs
unaligned MSA-Transformer-100M, and causal text controls:

  - Subtest 1A (``rescue_30_80``): 30-80% homolog main contrast on frozen
    ``results/R5/context_mutation_rescue_20260923/cohort.json`` (217 assays x 128
    variants; 164 common-support assays / 134 family clusters), including
    ``Δρ(homolog - unrelated)``, ``Δρ(homolog - no_context)``, and
    ``profile_increment`` (``model + profile`` vs ``profile``).
  - Subtest 1B (``e09_five_band``): 5-band identity curve (``<30%``, ``30-50%``,
    ``50-70%``, ``70-90%``, ``90-100%``) + ``self_copy`` ceiling + ``unrelated`` +
    ``no_context`` on frozen ``results/extensions/homology_context_20261008/
    uniref90_2026_03/bin/homology_context.json`` (201 assays x 128 variants).

Every entry explicitly declares:
  - ``input_alignment_format``: ``"explicit_column_aligned"`` ONLY for
    ``msa-transformer-100m|aligned=True``; ``"unaligned_concat"`` for all other
    models (including ``protgpt3-msa``, ``dayhoff-3b-msa``,
    ``msa-transformer-100m|aligned=False``, and all single-chain models).
  - ``context_training_regime``: ``"single_chain_causal"``,
    ``"single_chain_bidirectional"``, ``"multiseq_icl_unaligned"``, or
    ``"msa_2d_axial"``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from src.capability.context import homology_context as H
from src.capability.context.profile_increment import (
    common_vector_support,
    context_rescue,
    profile_increment,
)
from src.capability.core.amino_acids import AA20
from src.capability.extensions.bidirectional import (
    counts as band_verdict_counts,
    paired_bands,
    verdict as band_verdicts,
)
from src.capability.extensions.family_distribution import _family_inference
from src.capability.memory_theory.io_utils import load_json_file, write_json_with_sha256
from src.capability.unified.master_summary import FORBIDDEN_INFRA_RE
from src.capability.unified.roster import ARM_METADATA, REPO_ROOT, UNIFIED_OUTPUT_DIR

SCHEMA_DMS_ARM_SCORE = "r5_dms_context_arm_score_v1"
SCHEMA_DMS_UNIFIED = "r5_dms_context_ranking_unified_v1"

RESCUE_COHORT_REL = Path("results/R5/context_mutation_rescue_20260923/cohort.json")
RESCUE_REF_PI_REL = Path(
    "results/R5/context_profile_increment_20260923/20260923075017_68b04c315654/"
    "context_profile_increment/profile_increment.json"
)
E09_COHORT_REL = Path(
    "results/extensions/homology_context_20261008/uniref90_2026_03/bin/homology_context.json"
)
E09_REF_CURVE_REL = Path(
    "results/extensions/homology_context_20261008/uniref90_2026_03/e09_curve/context_identity_curve.json"
)
DMS_SCORES_DIR_REL = Path(UNIFIED_OUTPUT_DIR) / "r5_dms_context_scores"

E09_BINS_ORDERED: tuple[str, ...] = (
    "id_lt_30",
    "id_30_50",
    "id_50_70",
    "id_70_90",
    "id_90_100",
    "self_copy",
)
RESCUE_CONDITIONS: tuple[str, ...] = ("no_context", "unrelated", "homolog")

CACHED_1A_PATHS: dict[str, str] = {
    "progen3-3b": "results/R5/context_profile_increment_20260923/20260923062631_dfd71cd7e171/rescue_progen3_3b/scores_progen3-3b.json",
    "prollama-stage-1": "results/R5/context_profile_increment_20260923/20260923062631_dfd71cd7e171/rescue_prollama_stage1/scores_prollama-stage-1.json",
    "prollama": "results/R5/context_profile_increment_20260923/20260923062631_dfd71cd7e171/rescue_prollama_stage2/scores_prollama.json",
    "proteinglm-7b-clm": "results/R5/context_profile_increment_20260923/20260923062631_dfd71cd7e171/rescue_proteinglm/scores_proteinglm-7b-clm.json",
}

CACHED_1B_PATHS: dict[str, str] = {
    "progen2-medium": "results/external_baseline/20261008104808_96e83c09ad71/homctx_e09_score_progen2_medium_gpu0/context_identity_scores.json",
    "progen2-xlarge": "results/external_baseline/20261008104808_96e83c09ad71/homctx_e09_score_progen2_xlarge_gpu5/context_identity_scores.json",
    "prollama-stage-1": "results/external_baseline/20261008104808_96e83c09ad71/homctx_e09_score_prollama_stage_1_gpu6/context_identity_scores.json",
    "protgpt2": "results/external_baseline/20261008104808_96e83c09ad71/homctx_e09_score_protgpt2_gpu1/context_identity_scores.json",
}


@dataclass(frozen=True)
class DmsContextSpec:
    """Declarative specification for one model/mode entry in Unified R5 DMS Context Ranking."""

    entry_key: str
    arm_id: str
    scorer_kind: str  # msa_transformer | gp_causal | progen3_causal | preregistered_causal | causal_text | bidirectional_mlm | esm3_mlm
    input_alignment_format: str  # "explicit_column_aligned" | "unaligned_concat"
    context_training_regime: str  # "msa_2d_axial" | "multiseq_icl_unaligned" | "single_chain_bidirectional" | "single_chain_causal"
    budget_tokens: int = 1024
    aligned_msa: bool | None = None
    priority_tier: int = 1  # 1 = 57 core entries (all protein/joint/bidir/MSA + 4 text); 2 = remaining 11 text arms
    shard_cost: float = 1.0

    @property
    def file_slug(self) -> str:
        return (
            self.entry_key.replace("|", "__")
            .replace("=", "_")
            .replace("/", "_")
        )


def _build_all_specs() -> tuple[DmsContextSpec, ...]:
    specs: list[DmsContextSpec] = [
        # 1. 2D Column-Alignment Switch (MSA-Transformer-100M)
        DmsContextSpec(
            "msa-transformer-100m|aligned=True",
            "msa-transformer-100m",
            "msa_transformer",
            "explicit_column_aligned",
            "msa_2d_axial",
            budget_tokens=1024,
            aligned_msa=True,
            priority_tier=1,
            shard_cost=1.5,
        ),
        DmsContextSpec(
            "msa-transformer-100m|aligned=False",
            "msa-transformer-100m",
            "msa_transformer",
            "unaligned_concat",
            "msa_2d_axial",
            budget_tokens=1024,
            aligned_msa=False,
            priority_tier=1,
            shard_cost=1.5,
        ),
        # 2. Multi-Sequence ICL Unaligned (ProtGPT3-MSA & Dayhoff-3B-MSA at b1024 + native long context)
        DmsContextSpec(
            "protgpt3-msa",
            "protgpt3-msa",
            "gp_causal",
            "unaligned_concat",
            "multiseq_icl_unaligned",
            budget_tokens=1024,
            priority_tier=1,
            shard_cost=2.5,
        ),
        DmsContextSpec(
            "protgpt3-msa|b8192",
            "protgpt3-msa",
            "gp_causal",
            "unaligned_concat",
            "multiseq_icl_unaligned",
            budget_tokens=8192,
            priority_tier=1,
            shard_cost=4.5,
        ),
        DmsContextSpec(
            "dayhoff-3b-msa",
            "dayhoff-3b-msa",
            "gp_causal",
            "unaligned_concat",
            "multiseq_icl_unaligned",
            budget_tokens=1024,
            priority_tier=1,
            shard_cost=3.5,
        ),
        DmsContextSpec(
            "dayhoff-3b-msa|b4096",
            "dayhoff-3b-msa",
            "gp_causal",
            "unaligned_concat",
            "multiseq_icl_unaligned",
            budget_tokens=4096,
            priority_tier=1,
            shard_cost=5.5,
        ),
    ]

    # 3. All 22 Single-Chain Bidirectional / Masked / 3Di / Diffusion / Multimodal (unaligned_concat)
    bidir_arms = [
        ("esm2-8m", "bidirectional_mlm", 0.4),
        ("esm2-35m", "bidirectional_mlm", 0.5),
        ("esm2-150m", "bidirectional_mlm", 0.7),
        ("esm2-650m", "bidirectional_mlm", 1.2),
        ("esm2-3b", "bidirectional_mlm", 2.2),
        ("esm2-15b", "bidirectional_mlm", 5.0),
        ("esmc-300m", "bidirectional_mlm", 1.0),
        ("esmc-600m", "bidirectional_mlm", 1.5),
        ("esmc-6b", "bidirectional_mlm", 3.5),
        ("proteinglm-1b-mlm", "bidirectional_mlm", 1.5),
        ("proteinglm-3b-mlm", "bidirectional_mlm", 2.5),
        ("proteinglm-10b-mlm", "bidirectional_mlm", 4.5),
        ("amplify-120m", "bidirectional_mlm", 0.7),
        ("amplify-350m", "bidirectional_mlm", 1.0),
        ("prime-690m", "bidirectional_mlm", 1.2),
        ("saprot-35m", "bidirectional_mlm", 0.5),
        ("saprot-650m", "bidirectional_mlm", 1.2),
        ("saprot-1.3b", "bidirectional_mlm", 1.8),
        ("dplm-150m", "bidirectional_mlm", 0.8),
        ("dplm-650m", "bidirectional_mlm", 1.3),
        ("dplm-3b", "bidirectional_mlm", 2.5),
        ("esm3-1.4b", "esm3_mlm", 2.0),
    ]
    for arm, skind, cost in bidir_arms:
        specs.append(
            DmsContextSpec(
                arm,
                arm,
                skind,
                "unaligned_concat",
                "single_chain_bidirectional",
                budget_tokens=1024,
                priority_tier=1,
                shard_cost=cost,
            )
        )

    # 4. All 25 Single-Chain Causal / Hybrid Protein & Joint/Adapted (unaligned_concat)
    causal_protein_joint = [
        ("progen2-small", "preregistered_causal", 1.0),
        ("progen2-base", "preregistered_causal", 1.8),
        ("progen2-medium", "preregistered_causal", 1.2),  # cached 1B
        ("progen2-large", "preregistered_causal", 3.0),
        ("progen2-xlarge", "preregistered_causal", 3.5),  # cached 1B
        ("progen3-112m", "progen3_causal", 1.5),
        ("progen3-219m", "progen3_causal", 2.0),
        ("progen3-339m", "progen3_causal", 2.5),
        ("progen3-762m", "progen3_causal", 3.5),
        ("progen3-1b", "progen3_causal", 4.0),
        ("progen3-3b", "progen3_causal", 4.5),  # cached 1A
        ("protgpt2", "preregistered_causal", 1.0),  # cached 1B
        ("protgpt3-112m", "gp_causal", 1.2),
        ("protgpt3-1.3b", "preregistered_causal", 2.2),
        ("protgpt3-10b", "gp_causal", 4.5),
        ("dayhoff-170m", "gp_causal", 1.8),
        ("proteinglm-7b-clm", "preregistered_causal", 3.0),  # cached 1A
        ("rita-xl", "preregistered_causal", 2.5),
        ("prollama-stage-1", "preregistered_causal", 0.1),  # cached BOTH 1A & 1B!
        ("prollama", "preregistered_causal", 3.0),  # cached 1A
        ("galactica-125m", "preregistered_causal", 1.0),
        ("galactica-1.3b", "preregistered_causal", 2.0),
        ("galactica-6.7b", "preregistered_causal", 4.0),
        ("galactica-30b", "preregistered_causal", 8.0),
        ("instructprotein", "preregistered_causal", 2.0),
    ]
    for arm, skind, cost in causal_protein_joint:
        specs.append(
            DmsContextSpec(
                arm,
                arm,
                skind,
                "unaligned_concat",
                "single_chain_causal",
                budget_tokens=1024,
                priority_tier=1,
                shard_cost=cost,
            )
        )

    # 5. Causal Text Controls (4 priority-tier 1 + 11 priority-tier 2 = all 15 text arms)
    text_primary = [
        ("gpt2-large", "causal_text", 1.5),
        ("llama-2-7b", "preregistered_causal", 3.5),
        ("qwen2.5-7b", "causal_text", 3.5),
        ("bygpt5-medium-en", "causal_text", 2.0),
    ]
    for arm, skind, cost in text_primary:
        specs.append(
            DmsContextSpec(
                arm,
                arm,
                skind,
                "unaligned_concat",
                "single_chain_causal",
                budget_tokens=1024,
                priority_tier=1,
                shard_cost=cost,
            )
        )

    text_extended = [
        ("gpt2", "causal_text", 0.8),
        ("gpt2-medium", "causal_text", 1.0),
        ("gpt2-xl", "causal_text", 2.0),
        ("dialogpt-small", "causal_text", 0.8),
        ("bygpt5-small-en", "causal_text", 1.2),
        ("bygpt5-base-en", "causal_text", 1.5),
        ("llama-3.2-3b", "causal_text", 2.2),
        ("qwen2.5-0.5b", "causal_text", 1.0),
        ("qwen2.5-0.5b-instruct", "causal_text", 1.0),
        ("qwen3-8b-base", "causal_text", 3.5),
        ("qwen2.5-32b", "causal_text", 8.0),
    ]
    for arm, skind, cost in text_extended:
        specs.append(
            DmsContextSpec(
                arm,
                arm,
                skind,
                "unaligned_concat",
                "single_chain_causal",
                budget_tokens=1024,
                priority_tier=2,
                shard_cost=cost,
            )
        )

    return tuple(specs)


DMS_CONTEXT_SPECS_ALL: tuple[DmsContextSpec, ...] = _build_all_specs()
DMS_CONTEXT_SPEC_BY_KEY: dict[str, DmsContextSpec] = {s.entry_key: s for s in DMS_CONTEXT_SPECS_ALL}
DMS_CONTEXT_CORE_KEYS: tuple[str, ...] = tuple(
    s.entry_key for s in DMS_CONTEXT_SPECS_ALL if s.priority_tier == 1
)


def assign_balanced_shards(
    specs: Sequence[DmsContextSpec], num_shards: int
) -> list[list[DmsContextSpec]]:
    """Deterministic greedy LPT (longest-processing-time) bin-packing across GPU shards."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    buckets: list[list[DmsContextSpec]] = [[] for _ in range(num_shards)]
    loads = [0.0] * num_shards
    for spec in sorted(specs, key=lambda s: (-s.shard_cost, s.entry_key)):
        idx = int(np.argmin(loads))
        buckets[idx].append(spec)
        loads[idx] += spec.shard_cost
    return buckets


# ---------------------------------------------------------------------------
# Pure-CPU Context Planning for Subtest 1A (rescue_30_80) & 1B (e09_five_band)
# ---------------------------------------------------------------------------


def fast_project_partner_to_query(
    query: str,
    partner: str,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_penalty: int = -2,
) -> str:
    """Bit-exact vectorized Needleman-Wunsch projection onto query coordinates.

    Replaces the inner Python loop of ``msa_utils.project_partner_to_query`` with
    exact ``np.maximum.accumulate`` over the gap-shifted prefix (~50x faster,
    100% bit-identical tie-breaking).
    """
    n = len(query)
    m = len(partner)
    if n == 0:
        return ""
    if m == 0:
        return "-" * n

    q_arr = np.frombuffer(query.encode("ascii"), dtype=np.uint8)
    p_arr = np.frombuffer(partner.encode("ascii"), dtype=np.uint8)
    j_idx = np.arange(m + 1, dtype=np.int32)
    j_shift = j_idx * gap_penalty  # shape (m + 1,)

    dp = np.empty((n + 1, m + 1), dtype=np.int32)
    trace = np.zeros((n + 1, m + 1), dtype=np.uint8)
    dp[:, 0] = np.arange(n + 1, dtype=np.int32) * gap_penalty
    dp[0, :] = j_shift
    trace[1:, 0] = 1
    trace[0, 1:] = 2

    shifted = np.empty(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        sub = np.where(p_arr == q_arr[i - 1], match_score, mismatch_score)
        diag = dp[i - 1, :-1] + sub
        up = dp[i - 1, 1:] + gap_penalty
        diag_ge_up = diag >= up
        best = np.where(diag_ge_up, diag, up)
        tr = np.where(diag_ge_up, 0, 1).astype(np.uint8)

        # Horizontal left-gap recurrence: dp[i, j] = max(best[j-1], dp[i, j-1] + gap_penalty)
        # Equivalently: dp[i, j] - j*g = max_prefix( [dp[i,0], best - (1..m)*g] )
        shifted[0] = dp[i, 0]
        shifted[1:] = best - j_shift[1:]
        acc = np.maximum.accumulate(shifted)
        dp[i, :] = acc + j_shift
        # Strict `>` matches `if cand_left > b:` tie-breaking verbatim
        trace[i, 1:] = np.where(acc[1:] > shifted[1:], 2, tr)

    projected = ["-"] * n
    i, j = n, m
    while i > 0 and j > 0:
        move = int(trace[i, j])
        if move == 0:
            projected[i - 1] = partner[j - 1]
            i -= 1
            j -= 1
        elif move == 1:
            projected[i - 1] = "-"
            i -= 1
        else:
            j -= 1
    return "".join(projected)


def fast_build_query_msa(
    target: str,
    partners: Sequence[str],
    *,
    max_depth: int = 16,
    aligned: bool = True,
) -> list[str]:
    """Bit-exact vectorized equivalent of ``msa_utils.build_query_msa``."""
    rows = [target]
    limit = max(0, int(max_depth) - 1)
    for partner in list(partners)[:limit]:
        if aligned:
            rows.append(fast_project_partner_to_query(target, partner))
        else:
            if len(partner) >= len(target):
                rows.append(partner[: len(target)])
            else:
                rows.append(partner + "-" * (len(target) - len(partner)))
    return rows


def has_common_substring_ge(a: str, b: str, k: int = 10) -> bool:
    """Exact O(len(a)+len(b)) test for whether ``a`` and ``b`` share any substring of length >= ``k``."""
    if len(a) < k or len(b) < k:
        return False
    kmers_a = {a[i : i + k] for i in range(len(a) - k + 1)}
    return any(b[j : j + k] in kmers_a for j in range(len(b) - k + 1))


def longest_common_substring_len(a: str, b: str) -> int:
    """Exact length of the longest common substring between two strings via C-accelerated SequenceMatcher."""
    import difflib

    if not a or not b:
        return 0
    return int(difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b)).size)


def _aa_comp(seq: str) -> np.ndarray:
    n = max(1, len(seq))
    return np.asarray([seq.count(aa) / n for aa in AA20], dtype=np.float64)


def plan_rescue_30_80_assay(
    assay_row: Mapping[str, Any],
    *,
    token_length_fn: Callable[[str], int],
    overhead_tokens: int,
    budget_tokens: int,
    multi_item_icl: bool = False,
    msa_mode: bool = False,
    max_msa_depth: int = 32,
) -> tuple[dict[str, Any] | None, str | None]:
    """Deterministic homolog vs matched-unrelated planner for one Subtest 1A assay.

    - For standard single-chain models (``budget=1024``, ``multi_item_icl=False``, ``msa_mode=False``):
      matches ``scripts/capability/context/context_mutation_rescue.py::plan_row`` verbatim
      (1 homolog + 1 exact-token-length-matched unrelated donor with ``max_lcs < 10``).
    - For ``msa_mode=True`` (MSA-Transformer-100M): selects up to ``max_msa_depth - 1`` homologs
      and the exact same count ``k`` of composition-matched unrelated donors (``max_lcs < 10``).
    - For ``multi_item_icl=True`` (ProtGPT3-MSA / Dayhoff-3B-MSA at ``b1024``/``b4096``/``b8192``):
      packs up to ``k`` homologs fitting inside ``budget_tokens`` and matches ``k`` unrelated donors
      (``max_lcs < 10``) with matched token count.
    """
    wt = str(assay_row["wildtype"])
    homs = list(assay_row.get("homolog_candidates") or [])
    unrs = list(assay_row.get("unrelated_candidates") or [])
    if not homs or not unrs:
        return None, "missing homolog or unrelated candidates"

    if msa_mode:
        # MSA-Transformer requires <= 1024 columns (WT length + 1 CLS)
        if len(wt) + 1 > 1024:
            return None, "wildtype exceeds MSA-Transformer 1024-column limit"
        k_max = max(1, int(max_msa_depth) - 1)
        picked_homs = homs[:k_max]
        # Filter unrelated donors by max_lcs < 10 in O(L) via 10-mer intersection and match count k
        valid_unrs = [u for u in unrs if not has_common_substring_ge(wt, str(u["sequence"]), 10)]
        if not valid_unrs:
            return None, "no unrelated donor passes max_lcs < 10"
        k = min(len(picked_homs), len(valid_unrs))
        picked_homs = picked_homs[:k]
        hcomp = np.mean([_aa_comp(str(h["sequence"])) for h in picked_homs], axis=0)
        valid_unrs.sort(key=lambda u: (float(np.square(_aa_comp(str(u["sequence"])) - hcomp).sum()), str(u["subject"])))
        picked_unrs = valid_unrs[:k]
        return {
            "homolog_items": [str(h["sequence"]) for h in picked_homs],
            "unrelated_items": [str(u["sequence"]) for u in picked_unrs],
            "context_tokens": sum(len(str(h["sequence"])) for h in picked_homs),
            "k_items": k,
            "homolog_identity": float(np.mean([float(h["identity"]) for h in picked_homs])),
            "max_lcs": int(max(longest_common_substring_len(wt, str(h["sequence"])) for h in picked_homs)),
        }, None

    target_lens = [token_length_fn(wt)] + [token_length_fn(str(s)) for s in assay_row["sequences"]]
    room = int(budget_tokens) - int(overhead_tokens) - max(target_lens)
    if room <= 0:
        return None, "full target exceeds context budget"

    if not multi_item_icl:
        # Exact match to context_mutation_rescue.py::plan_row
        donors = []
        for cand in unrs:
            tlen = token_length_fn(str(cand["sequence"]))
            if tlen <= room:
                donors.append((cand, tlen))
        for hom in homs:
            hlen = token_length_fn(str(hom["sequence"]))
            if hlen > room:
                continue
            eligible = [(d, dl) for d, dl in donors if dl == hlen]
            if not eligible:
                continue
            hcomp = _aa_comp(str(hom["sequence"]))
            eligible.sort(
                key=lambda pair: (
                    float(np.square(_aa_comp(str(pair[0]["sequence"])) - hcomp).sum()),
                    str(pair[0]["subject"]),
                )
            )
            for donor, _dl in eligible:
                if has_common_substring_ge(wt, str(donor["sequence"]), 10):
                    continue
                return {
                    "homolog_items": [str(hom["sequence"])],
                    "unrelated_items": [str(donor["sequence"])],
                    "context_tokens": int(hlen),
                    "k_items": 1,
                    "homolog_identity": float(hom["identity"]),
                    "max_lcs": int(longest_common_substring_len(wt, str(hom["sequence"]))),
                }, None
        return None, "no eligible homolog and exact-token-matched unrelated context within budget"

    # Multi-item ICL packing up to budget_tokens (matching k items & token budget)
    picked_homs = []
    used_toks = 0
    for hom in homs:
        hlen = token_length_fn(str(hom["sequence"]))
        if used_toks + hlen > room:
            continue
        picked_homs.append(hom)
        used_toks += hlen
    if not picked_homs:
        return None, "no homolog fits inside multi-item ICL budget"

    valid_unrs = [
        (d, token_length_fn(str(d["sequence"])))
        for d in unrs
        if not has_common_substring_ge(wt, str(d["sequence"]), 10)
    ]
    if not valid_unrs:
        return None, "no unrelated donor passes max_lcs < 10"
    hcomp = np.mean([_aa_comp(str(h["sequence"])) for h in picked_homs], axis=0)
    valid_unrs.sort(
        key=lambda pair: (
            abs(pair[1] - (used_toks / len(picked_homs))),
            float(np.square(_aa_comp(str(pair[0]["sequence"])) - hcomp).sum()),
            str(pair[0]["subject"]),
        )
    )
    picked_unrs: list[dict[str, Any]] = []
    u_toks = 0
    for donor, dlen in valid_unrs:
        if len(picked_unrs) >= len(picked_homs):
            break
        if u_toks + dlen <= room:
            picked_unrs.append(donor)
            u_toks += dlen
    if not picked_unrs:
        return None, "no unrelated donor fits inside multi-item ICL budget"
    # Match exact k between homolog & unrelated
    k = min(len(picked_homs), len(picked_unrs))
    picked_homs = picked_homs[:k]
    picked_unrs = picked_unrs[:k]
    return {
        "homolog_items": [str(h["sequence"]) for h in picked_homs],
        "unrelated_items": [str(u["sequence"]) for u in picked_unrs],
        "context_tokens": int(sum(token_length_fn(str(h["sequence"])) for h in picked_homs)),
        "k_items": k,
        "homolog_identity": float(np.mean([float(h["identity"]) for h in picked_homs])),
        "max_lcs": int(max(longest_common_substring_len(wt, str(h["sequence"])) for h in picked_homs)),
    }, None


def plan_e09_target(
    target_row: Mapping[str, Any],
    assay_rows: Sequence[Mapping[str, Any]],
    *,
    token_length_fn: Callable[[str], int],
    overhead_tokens: int,
    budget_tokens: int,
    msa_mode: bool = False,
    long_icl_mode: bool = False,
) -> dict[str, Any]:
    """Plan all E09 conditions (no_context, unrelated, 5 identity bins, self_copy) for one target.

    - Standard 1024-token models call frozen ``H.plan_conditions`` verbatim.
    - ``msa_mode=True`` (MSA-Transformer-100M) retains up to 16 bin items (matching exact count
      with unrelated donors) so 2D axial attention sees full bin MSA depth.
    - ``long_icl_mode=True`` (``b4096``/``b8192``) raises item cap up to 16 within ``budget_tokens``.
    """
    wt = str(target_row["wildtype"])
    if msa_mode:
        if len(wt) + 1 > 1024:
            return {"conditions": {}, "items": {}, "item_count": 0, "skipped_reason": "wt_gt_1023"}
        donors = [
            d for d in (target_row.get("unrelated_pool") or [])
            if not has_common_substring_ge(wt, str(d["sequence"]), H.UNRELATED_MAX_LCS)
        ]
        items_out: dict[str, list[dict[str, Any]]] = {"no_context": []}
        cond_meta: dict[str, dict[str, Any]] = {"no_context": {"status": "present", "k": 0}}
        # Choose k = min(16, max available bin count capped at len(donors))
        max_avail = max((len(target_row["bins"].get(b, [])) for b in H.BIN_NAMES), default=0)
        k_unr = min(16, max(1, max_avail), len(donors))
        if k_unr > 0:
            items_out["unrelated"] = [
                {"sequence": str(d["sequence"]), "subject": str(d.get("subject", "")), "identity": 0.0, "max_lcs": longest_common_substring_len(wt, str(d["sequence"]))}
                for d in donors[:k_unr]
            ]
            cond_meta["unrelated"] = {"status": "present", "k": k_unr}
        else:
            items_out["unrelated"] = []
            cond_meta["unrelated"] = {"status": "absent", "k": 0}
        for bname in H.BIN_NAMES:
            cands = list(target_row["bins"].get(bname) or [])[:16]
            if cands:
                items_out[bname] = [
                    {
                        "sequence": str(c["sequence"]),
                        "subject": str(c.get("subject", "")),
                        "identity": float(c.get("identity_over_query", c.get("identity", 0.0))),
                        "max_lcs": int(c.get("max_lcs", longest_common_substring_len(wt, str(c["sequence"])))),
                    }
                    for c in cands
                ]
                cond_meta[bname] = {"status": "present", "k": len(cands)}
            else:
                items_out[bname] = []
                cond_meta[bname] = {"status": "absent", "k": 0}
        sc = target_row.get("self_copy")
        if sc and sc.get("sequence"):
            items_out["self_copy"] = [
                {
                    "sequence": str(sc["sequence"]),
                    "subject": str(sc.get("subject", "self_copy")),
                    "identity": float(sc.get("identity_over_query", 100.0)),
                    "max_lcs": len(wt),
                }
            ]
            cond_meta["self_copy"] = {"status": "present", "k": 1}
        else:
            items_out["self_copy"] = []
            cond_meta["self_copy"] = {"status": "absent", "k": 0}
        return {"conditions": cond_meta, "items": items_out, "item_count": k_unr}

    # Rebuild mutant sequences to determine exact widest target token count
    from src.capability.models.fitness import parse_mutant

    referent = token_length_fn(wt)
    widest = referent
    for assay in assay_rows:
        for mut in assay["mutants"]:
            chars = list(wt)
            for _w, pos, m in parse_mutant(mut):
                chars[pos - 1] = m
            widest = max(widest, token_length_fn("".join(chars)))
    room = int(budget_tokens) - int(overhead_tokens) - widest

    if not long_icl_mode:
        conditions, budget_rec = H.plan_conditions(
            wildtype=wt,
            bins=target_row["bins"],
            donors=target_row["unrelated_pool"],
            self_copy=target_row.get("self_copy"),
            token_length=token_length_fn,
            referent=referent,
            room=room,
        )
        items_out = {}
        cond_meta = {}
        for cname, cplan in conditions.items():
            cond_meta[cname] = cplan.record()
            items_out[cname] = [
                {
                    "sequence": str(it["sequence"]),
                    "subject": str(it.get("subject", "")),
                    "identity": float(it.get("identity_over_query", it.get("identity", 0.0))),
                    "max_lcs": int(it.get("max_lcs", longest_common_substring_len(wt, str(it["sequence"])))),
                    "tokens": int(it.get("tokens", token_length_fn(str(it["sequence"])))),
                }
                for it in cplan.items
            ]
        return {
            "conditions": cond_meta,
            "items": items_out,
            "item_count": int(budget_rec["item_count"]),
        }

    # Long-budget multi-seq ICL (b4096/b8192): pack up to max_k <= 16 items fitting inside room
    if room <= 0:
        return {"conditions": {}, "items": {}, "item_count": 0, "skipped_reason": "target_exceeds_budget"}
    max_k = min(16, max(1, room // max(1, referent)))
    items_out = {"no_context": []}
    cond_meta = {"no_context": {"status": "present", "k": 0}}

    def _pack_list(cands: Sequence[Mapping[str, Any]], cap_k: int) -> list[dict[str, Any]]:
        chosen: list[dict[str, Any]] = []
        used = 0
        for c in cands:
            s = str(c["sequence"])
            t = token_length_fn(s)
            if used + t > room:
                continue
            chosen.append(
                {
                    "sequence": s,
                    "subject": str(c.get("subject", "")),
                    "identity": float(c.get("identity_over_query", c.get("identity", 0.0))),
                    "max_lcs": int(c.get("max_lcs", longest_common_substring_len(wt, s))),
                    "tokens": t,
                }
            )
            used += t
            if len(chosen) >= cap_k:
                break
        return chosen

    donors = [
        d for d in (target_row.get("unrelated_pool") or [])
        if longest_common_substring_len(wt, str(d["sequence"])) < H.UNRELATED_MAX_LCS
    ]
    unr_items = _pack_list(donors, max_k)
    items_out["unrelated"] = unr_items
    cond_meta["unrelated"] = {"status": "present" if unr_items else "absent", "k": len(unr_items)}

    for bname in H.BIN_NAMES:
        b_items = _pack_list(target_row["bins"].get(bname) or [], max_k)
        items_out[bname] = b_items
        cond_meta[bname] = {"status": "present" if b_items else "absent", "k": len(b_items)}

    sc = target_row.get("self_copy")
    if sc and sc.get("sequence") and token_length_fn(str(sc["sequence"])) <= room:
        sc_seq = str(sc["sequence"])
        items_out["self_copy"] = [
            {
                "sequence": sc_seq,
                "subject": str(sc.get("subject", "self_copy")),
                "identity": float(sc.get("identity_over_query", 100.0)),
                "max_lcs": len(wt),
                "tokens": token_length_fn(sc_seq),
            }
        ]
        cond_meta["self_copy"] = {"status": "present", "k": 1}
    else:
        items_out["self_copy"] = []
        cond_meta["self_copy"] = {"status": "absent", "k": 0}
    return {"conditions": cond_meta, "items": items_out, "item_count": len(unr_items)}


# ---------------------------------------------------------------------------
# Pure-CPU Aggregation, Reproduction Gates & Simultaneous Inference
# ---------------------------------------------------------------------------


def _spearman_safe(x: Sequence[float], y: Sequence[float]) -> float | None:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 3 or np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return None
    val = float(spearmanr(xa, ya).statistic)
    return val if math.isfinite(val) else None


def cluster_mean_bootstrap(
    values: Sequence[float],
    clusters: Sequence[int],
    *,
    draws: int = 10000,
    seed: int = 20261124,
) -> dict[str, Any]:
    """Equal-cluster mean + cluster-resampled percentile bootstrap 95% CI."""
    by_cluster: dict[int, list[float]] = defaultdict(list)
    for v, c in zip(values, clusters):
        if v is not None and math.isfinite(float(v)):
            by_cluster[int(c)].append(float(v))
    if not by_cluster:
        return {"n_assays": 0, "n_clusters": 0, "mean": None, "ci95": None, "se": None}
    c_ids = sorted(by_cluster.keys())
    c_means = np.asarray([float(np.mean(by_cluster[c])) for c in c_ids], dtype=np.float64)
    n_c = int(c_means.size)
    pt = float(np.mean(c_means))
    if n_c == 1:
        return {
            "n_assays": sum(len(v) for v in by_cluster.values()),
            "n_clusters": 1,
            "mean": pt,
            "ci95": [pt, pt],
            "se": 0.0,
        }
    se = float(np.std(c_means, ddof=1) / math.sqrt(n_c))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_c, size=(draws, n_c))
    boot = c_means[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "n_assays": sum(len(v) for v in by_cluster.values()),
        "n_clusters": n_c,
        "mean": pt,
        "ci95": [float(lo), float(hi)],
        "se": se,
    }


def verify_subtest_1a_reproduction(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Verify <1e-12 reproduction of frozen profile_increment.json on all 4 cached 1A arms."""
    repo_root = Path(repo_root)
    ref = load_json_file(repo_root / RESCUE_REF_PI_REL)
    sources = [load_json_file(repo_root / CACHED_1A_PATHS[arm]) for arm in sorted(CACHED_1A_PATHS)]
    subsets, support = common_vector_support(sources)
    ref_by_arm = {r["arm"]: r for r in ref["common_support"]["profile_increments"]}

    max_diff = 0.0
    checked = 0
    per_arm: dict[str, float] = {}
    for src, sub in zip(sources, subsets):
        arm = str(src["arm"])
        cr = context_rescue(sub, bootstrap=int(ref["bootstrap"]), seed=int(ref["seed"]))
        pi_hom = profile_increment(sub, condition="homolog", bootstrap=int(ref["bootstrap"]), seed=int(ref["seed"]))
        ref_arm = ref_by_arm[arm]

        d1 = abs(
            cr["strata"]["pooled"]["metrics"]["delta_spearman"]["point"]
            - ref_arm["context_rescue"]["strata"]["pooled"]["metrics"]["delta_spearman"]["point"]
        )
        d2 = abs(
            cr["strata"]["pooled"]["metrics"]["homolog_minus_no_context_spearman"]["point"]
            - ref_arm["context_rescue"]["strata"]["pooled"]["metrics"]["homolog_minus_no_context_spearman"]["point"]
        )
        d3 = abs(
            pi_hom["summaries"]["cv_delta_spearman"]["point"]
            - ref_arm["conditions"]["homolog"]["summaries"]["cv_delta_spearman"]["point"]
        )
        arm_max = max(d1, d2, d3)
        per_arm[arm] = arm_max
        max_diff = max(max_diff, arm_max)
        checked += 3

    return {
        "reference_path": str(RESCUE_REF_PI_REL),
        "n_common_assays": int(support["n_assays"]),
        "n_common_families": int(support["n_families"]),
        "scalars_checked": checked,
        "max_abs_diff": float(max_diff),
        "per_arm_max_diff": per_arm,
        "tolerance": 1e-12,
        "passed": bool(max_diff < 1e-12),
    }


def verify_subtest_1b_reproduction(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Verify <1e-12 reproduction of frozen context_identity_curve.json on all 4 cached 1B arms."""
    repo_root = Path(repo_root)
    ref = load_json_file(repo_root / E09_REF_CURVE_REL)
    ref_panels = {(p["panel"], p["referent"]): p for p in ref["panels"]}

    max_diff = 0.0
    checked = 0
    per_arm: dict[str, float] = {}
    for arm, rel_p in sorted(CACHED_1B_PATHS.items()):
        rec = load_json_file(repo_root / rel_p)
        arm_max = 0.0
        for ref_name in ("unrelated", "no_context"):
            panel = ref_panels[("per_bin_paired_support", ref_name)]
            ref_rows = {r["bin"]: r for r in panel["curve"][arm]}
            for bname, rrow in ref_rows.items():
                by_c: dict[int, list[float]] = defaultdict(list)
                for assay in rec["assays"]:
                    v_b = assay["spearman"].get(bname)
                    v_r = assay["spearman"].get(ref_name)
                    if v_b is not None and v_r is not None:
                        by_c[int(assay["cluster"])].append(float(v_b) - float(v_r))
                pt = float(np.mean([float(np.mean(vs)) for vs in by_c.values()]))
                diff = abs(pt - float(rrow["point"]))
                arm_max = max(arm_max, diff)
                max_diff = max(max_diff, diff)
                checked += 1
        per_arm[arm] = arm_max

    return {
        "reference_path": str(E09_REF_CURVE_REL),
        "scalars_checked": checked,
        "max_abs_diff": float(max_diff),
        "per_arm_max_diff": per_arm,
        "tolerance": 1e-12,
        "passed": bool(max_diff < 1e-12),
    }


def _resolve_subtest_score_doc(
    spec: DmsContextSpec,
    subtest: str,
    repo_root: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Load cached legacy score file or newly generated H200 score file for (spec, subtest)."""
    if subtest == "rescue_30_80" and spec.entry_key in CACHED_1A_PATHS:
        p = repo_root / CACHED_1A_PATHS[spec.entry_key]
        if p.is_file():
            return load_json_file(p), CACHED_1A_PATHS[spec.entry_key]
    if subtest == "e09_five_band" and spec.entry_key in CACHED_1B_PATHS:
        p = repo_root / CACHED_1B_PATHS[spec.entry_key]
        if p.is_file():
            return load_json_file(p), CACHED_1B_PATHS[spec.entry_key]

    new_p = repo_root / DMS_SCORES_DIR_REL / subtest / f"{spec.file_slug}.json"
    if new_p.is_file():
        doc = load_json_file(new_p)
        if doc.get("status") == "complete":
            return doc, str(new_p.relative_to(repo_root))
    return None, None


def _summarize_1a_doc(
    spec: DmsContextSpec,
    doc: Mapping[str, Any],
    source_path: str,
    common_assay_ids: set[str],
    *,
    draws: int = 10000,
    seed: int = 20261124,
) -> tuple[dict[str, Any], dict[int, float], dict[int, float]]:
    """Summarize Subtest 1A (rescue_30_80) for one model entry + return per-cluster contrast vectors."""
    assays = list(doc.get("assays") or [])
    by_cond_vals: dict[str, list[float]] = {c: [] for c in RESCUE_CONDITIONS}
    by_cond_cl: dict[str, list[int]] = {c: [] for c in RESCUE_CONDITIONS}
    d_hu_vals, d_hu_cl = [], []
    d_hn_vals, d_hn_cl = [], []
    d_un_vals, d_un_cl = [], []
    prof_vals, prof_cl = [], []

    cluster_d_hu: dict[int, list[float]] = defaultdict(list)
    cluster_d_hn: dict[int, list[float]] = defaultdict(list)

    for a in assays:
        cl = int(a["cluster"])
        sp = a.get("spearman") or {}
        if not sp and a.get("scores") and a.get("measured"):
            sp = {c: _spearman_safe(a["scores"][c], a["measured"]) for c in RESCUE_CONDITIONS if c in a["scores"]}
        for c in RESCUE_CONDITIONS:
            v = sp.get(c)
            if v is not None and math.isfinite(float(v)):
                by_cond_vals[c].append(float(v))
                by_cond_cl[c].append(cl)
        h, u, n = sp.get("homolog"), sp.get("unrelated"), sp.get("no_context")
        if h is not None and u is not None:
            diff = float(h) - float(u)
            d_hu_vals.append(diff)
            d_hu_cl.append(cl)
            cluster_d_hu[cl].append(diff)
        if h is not None and n is not None:
            diff = float(h) - float(n)
            d_hn_vals.append(diff)
            d_hn_cl.append(cl)
            cluster_d_hn[cl].append(diff)
        if u is not None and n is not None:
            d_un_vals.append(float(u) - float(n))
            d_un_cl.append(cl)
        lk = a.get("lookup_spearman")
        if lk is not None and math.isfinite(float(lk)):
            prof_vals.append(float(lk))
            prof_cl.append(cl)

    # Profile increment (model + profile vs profile) on all assays with per-variant scores
    pi_summary: dict[str, Any] = {}
    scored_with_vecs = [
        a for a in assays
        if a.get("scores") and a.get("measured") and a.get("profile_scores") and a.get("mutants")
    ]
    if len({int(a["cluster"]) for a in scored_with_vecs}) >= 3:
        for cond in ("no_context", "homolog"):
            sub_c = [a for a in scored_with_vecs if cond in a["scores"]]
            if len({int(a["cluster"]) for a in sub_c}) >= 3:
                pi_res = profile_increment(sub_c, condition=cond, bootstrap=min(draws, 2000), seed=seed)
                pi_summary[cond] = {
                    "n_assays": pi_res["n_assays"],
                    "n_families": pi_res["n_families"],
                    "cv_profile_spearman": pi_res["summaries"]["cv_profile_spearman"],
                    "cv_profile_model_spearman": pi_res["summaries"]["cv_profile_model_spearman"],
                    "cv_delta_spearman": pi_res["summaries"]["cv_delta_spearman"],
                    "partial_rank_correlation": pi_res["summaries"]["partial_rank_correlation"],
                }

    # Common-164 intersection contrast
    c164_hu_v, c164_hu_c = [], []
    c164_hn_v, c164_hn_c = [], []
    for a in assays:
        if str(a["assay"]) not in common_assay_ids:
            continue
        sp = a.get("spearman") or {}
        cl = int(a["cluster"])
        if sp.get("homolog") is not None and sp.get("unrelated") is not None:
            c164_hu_v.append(float(sp["homolog"]) - float(sp["unrelated"]))
            c164_hu_c.append(cl)
        if sp.get("homolog") is not None and sp.get("no_context") is not None:
            c164_hn_v.append(float(sp["homolog"]) - float(sp["no_context"]))
            c164_hn_c.append(cl)

    meta = asdict(ARM_METADATA[spec.arm_id]) if spec.arm_id in ARM_METADATA else {}
    rec = {
        "entry_key": spec.entry_key,
        "arm_id": spec.arm_id,
        "params_m": meta.get("params_m"),
        "family": meta.get("family"),
        "paradigm": meta.get("paradigm"),
        "input_alignment_format": spec.input_alignment_format,
        "context_training_regime": spec.context_training_regime,
        "budget_tokens": spec.budget_tokens,
        "aligned_msa": spec.aligned_msa,
        "source_path": source_path,
        "n_assays_scored": len(assays),
        "n_clusters_scored": len({int(a["cluster"]) for a in assays}),
        "spearman_by_condition": {
            c: cluster_mean_bootstrap(by_cond_vals[c], by_cond_cl[c], draws=draws, seed=seed + i)
            for i, c in enumerate(RESCUE_CONDITIONS)
        },
        "profile_lookup_spearman": cluster_mean_bootstrap(prof_vals, prof_cl, draws=draws, seed=seed + 10),
        "delta_homolog_minus_unrelated": cluster_mean_bootstrap(d_hu_vals, d_hu_cl, draws=draws, seed=seed + 20),
        "delta_homolog_minus_no_context": cluster_mean_bootstrap(d_hn_vals, d_hn_cl, draws=draws, seed=seed + 21),
        "delta_unrelated_minus_no_context": cluster_mean_bootstrap(d_un_vals, d_un_cl, draws=draws, seed=seed + 22),
        "common_164_delta_homolog_minus_unrelated": cluster_mean_bootstrap(
            c164_hu_v, c164_hu_c, draws=draws, seed=seed + 30
        ),
        "common_164_delta_homolog_minus_no_context": cluster_mean_bootstrap(
            c164_hn_v, c164_hn_c, draws=draws, seed=seed + 31
        ),
        "profile_increment": pi_summary,
    }
    cl_hu_means = {c: float(np.mean(vs)) for c, vs in cluster_d_hu.items()}
    cl_hn_means = {c: float(np.mean(vs)) for c, vs in cluster_d_hn.items()}
    return rec, cl_hu_means, cl_hn_means


def _summarize_1b_doc(
    spec: DmsContextSpec,
    doc: Mapping[str, Any],
    source_path: str,
    *,
    draws: int = 10000,
    seed: int = 20261124,
) -> dict[str, Any]:
    """Summarize Subtest 1B (e09_five_band) across all 5 identity bands + self_copy vs unrelated & no_context."""
    assays = list(doc.get("assays") or [])
    all_conds = ("no_context", "unrelated", *E09_BINS_ORDERED)
    abs_by_cond: dict[str, Any] = {}
    vs_unr: dict[str, Any] = {}
    vs_noctx: dict[str, Any] = {}
    vs_prof: dict[str, Any] = {}

    for idx, cond in enumerate(all_conds):
        vals, cls = [], []
        for a in assays:
            v = (a.get("spearman") or {}).get(cond)
            if v is not None and math.isfinite(float(v)):
                vals.append(float(v))
                cls.append(int(a["cluster"]))
        abs_by_cond[cond] = cluster_mean_bootstrap(vals, cls, draws=draws, seed=seed + idx)

    for idx, bname in enumerate((*E09_BINS_ORDERED, "no_context", "unrelated")):
        for ref_name, target_dict, s_off in (
            ("unrelated", vs_unr, 100),
            ("no_context", vs_noctx, 200),
            ("profile_lookup", vs_prof, 300),
        ):
            if bname == ref_name:
                continue
            vals, cls = [], []
            for a in assays:
                sp = a.get("spearman") or {}
                vb = sp.get(bname)
                vr = a.get("lookup_spearman") if ref_name == "profile_lookup" else sp.get(ref_name)
                if vb is not None and vr is not None and math.isfinite(float(vb)) and math.isfinite(float(vr)):
                    vals.append(float(vb) - float(vr))
                    cls.append(int(a["cluster"]))
            target_dict[bname] = cluster_mean_bootstrap(vals, cls, draws=draws, seed=seed + s_off + idx)

    # Direct verbatim-copy degradation diagnostics
    d_90 = (vs_noctx.get("id_90_100") or {}).get("mean")
    d_30_50 = (vs_noctx.get("id_30_50") or {}).get("mean")
    d_self = (vs_noctx.get("self_copy") or {}).get("mean")
    d_90_unr = (vs_unr.get("id_90_100") or {}).get("mean")
    d_self_unr = (vs_unr.get("self_copy") or {}).get("mean")

    meta = asdict(ARM_METADATA[spec.arm_id]) if spec.arm_id in ARM_METADATA else {}
    return {
        "entry_key": spec.entry_key,
        "arm_id": spec.arm_id,
        "params_m": meta.get("params_m"),
        "family": meta.get("family"),
        "paradigm": meta.get("paradigm"),
        "input_alignment_format": spec.input_alignment_format,
        "context_training_regime": spec.context_training_regime,
        "budget_tokens": spec.budget_tokens,
        "aligned_msa": spec.aligned_msa,
        "source_path": source_path,
        "n_assays_scored": len(assays),
        "n_clusters_scored": len({int(a["cluster"]) for a in assays}),
        "spearman_by_condition": abs_by_cond,
        "delta_vs_unrelated": vs_unr,
        "delta_vs_no_context": vs_noctx,
        "delta_vs_profile_lookup": vs_prof,
        "verbatim_copy_diagnostics": {
            "delta_id_90_100_vs_no_context": d_90,
            "delta_self_copy_vs_no_context": d_self,
            "delta_id_90_100_vs_unrelated": d_90_unr,
            "delta_self_copy_vs_unrelated": d_self_unr,
            "id_90_100_minus_id_30_50_vs_no_context": (
                float(d_90 - d_30_50) if d_90 is not None and d_30_50 is not None else None
            ),
            "exhibits_90_100_degradation_vs_no_context": bool(d_90 is not None and d_90 < 0.0),
            "exhibits_self_copy_degradation_vs_no_context": bool(d_self is not None and d_self < 0.0),
        },
    }


def _simultaneous_bands_from_cluster_maps(
    entry_keys: Sequence[str],
    cluster_maps: Mapping[str, Mapping[int, float]],
    *,
    draws: int = 10000,
    seed: int = 20261124,
) -> dict[str, Any]:
    """Compute both native ``paired_bands`` and studentized ``maxT`` on common complete clusters."""
    keys = [k for k in entry_keys if k in cluster_maps and len(cluster_maps[k]) >= 3]
    if not keys:
        return {"status": "empty", "n_entries": 0, "n_common_clusters": 0}
    common_clusters = sorted(set.intersection(*(set(cluster_maps[k].keys()) for k in keys)))
    if len(common_clusters) < 8:
        return {
            "status": "insufficient_common_clusters",
            "n_entries": len(keys),
            "n_common_clusters": len(common_clusters),
        }
    mat = np.column_stack(
        [np.asarray([cluster_maps[k][c] for c in common_clusters], dtype=np.float64) for k in keys]
    )
    native = paired_bands(mat, draws=draws, seed=seed, confidence=0.95)
    n_verdicts = band_verdicts(native["interval"])
    stud = _family_inference(mat, keys, draws=draws, seed=seed)
    s_intervals = [h["simultaneous"] for h in stud["primary"]["hypotheses"]]
    s_verdicts = band_verdicts(s_intervals)
    return {
        "status": "complete",
        "n_entries": len(keys),
        "n_common_clusters": len(common_clusters),
        "draws": int(draws),
        "seed": int(seed),
        "native_critical_value": float(native["critical_value"]),
        "native_verdict_counts": band_verdict_counts(n_verdicts),
        "studentized_critical_value": float(stud["primary"]["critical_value"]),
        "studentized_verdict_counts": band_verdict_counts(s_verdicts),
        "per_entry": {
            k: {
                "point": float(native["point"][i]),
                "native_ci95": [float(native["interval"][i][0]), float(native["interval"][i][1])],
                "native_verdict": n_verdicts[i],
                "studentized_ci95": [float(s_intervals[i][0]), float(s_intervals[i][1])],
                "studentized_verdict": s_verdicts[i],
            }
            for i, k in enumerate(keys)
        },
    }


def build_r5_dms_context_unified(
    repo_root: Path = REPO_ROOT,
    *,
    allow_pending: bool = False,
    priority_scope: str = "core",
    draws: int = 10000,
    seed: int = 20261124,
) -> dict[str, Any]:
    """Assemble the complete unified R5 Downstream-1 DMS homolog-context ranking JSON."""
    repo_root = Path(repo_root)
    repro_1a = verify_subtest_1a_reproduction(repo_root)
    repro_1b = verify_subtest_1b_reproduction(repo_root)
    if not repro_1a["passed"] or not repro_1b["passed"]:
        raise RuntimeError(f"Reproduction gate failed: 1A={repro_1a}, 1B={repro_1b}")

    ref_pi = load_json_file(repo_root / RESCUE_REF_PI_REL)
    common_164_ids = set(ref_pi["common_support"]["support"]["assay_ids"])

    target_specs = [
        s for s in DMS_CONTEXT_SPECS_ALL
        if (priority_scope == "all" or s.priority_tier == 1)
    ]

    entries_1a: dict[str, Any] = {}
    entries_1b: dict[str, Any] = {}
    pending_1a: list[str] = []
    pending_1b: list[str] = []
    cl_hu_maps: dict[str, dict[int, float]] = {}
    cl_hn_maps: dict[str, dict[int, float]] = {}

    for idx, spec in enumerate(target_specs):
        doc_1a, path_1a = _resolve_subtest_score_doc(spec, "rescue_30_80", repo_root)
        if doc_1a is not None and path_1a is not None:
            rec_1a, hu_m, hn_m = _summarize_1a_doc(
                spec, doc_1a, path_1a, common_164_ids, draws=draws, seed=seed + idx * 50
            )
            entries_1a[spec.entry_key] = rec_1a
            cl_hu_maps[spec.entry_key] = hu_m
            cl_hn_maps[spec.entry_key] = hn_m
        else:
            pending_1a.append(spec.entry_key)

        doc_1b, path_1b = _resolve_subtest_score_doc(spec, "e09_five_band", repo_root)
        if doc_1b is not None and path_1b is not None:
            entries_1b[spec.entry_key] = _summarize_1b_doc(
                spec, doc_1b, path_1b, draws=draws, seed=seed + 10000 + idx * 50
            )
        else:
            pending_1b.append(spec.entry_key)

    if not allow_pending and (pending_1a or pending_1b):
        raise RuntimeError(
            f"Incomplete R5 DMS context scores (strict mode): "
            f"pending_1a={len(pending_1a)} ({pending_1a[:5]}), "
            f"pending_1b={len(pending_1b)} ({pending_1b[:5]})"
        )

    # Simultaneous bands on Subtest 1A across all completed entries
    sim_1a_hu = _simultaneous_bands_from_cluster_maps(
        list(entries_1a.keys()), cl_hu_maps, draws=draws, seed=seed
    )
    sim_1a_hn = _simultaneous_bands_from_cluster_maps(
        list(entries_1a.keys()), cl_hn_maps, draws=draws, seed=seed + 1
    )

    # Paradigm rollups across completed entries
    def _rollup_1a() -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in entries_1a.values():
            key = f"{e['context_training_regime']}|{e['input_alignment_format']}"
            buckets[key].append(e)
        out: dict[str, Any] = {}
        for k, rows in sorted(buckets.items()):
            hu = [r["delta_homolog_minus_unrelated"]["mean"] for r in rows if r["delta_homolog_minus_unrelated"]["mean"] is not None]
            hn = [r["delta_homolog_minus_no_context"]["mean"] for r in rows if r["delta_homolog_minus_no_context"]["mean"] is not None]
            out[k] = {
                "n_entries": len(rows),
                "mean_delta_homolog_minus_unrelated": float(np.mean(hu)) if hu else None,
                "mean_delta_homolog_minus_no_context": float(np.mean(hn)) if hn else None,
                "entries": [r["entry_key"] for r in rows],
            }
        return out

    def _rollup_1b() -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in entries_1b.values():
            key = f"{e['context_training_regime']}|{e['input_alignment_format']}"
            buckets[key].append(e)
        out: dict[str, Any] = {}
        for k, rows in sorted(buckets.items()):
            band_means = {}
            for b in E09_BINS_ORDERED:
                vs = [r["delta_vs_no_context"][b]["mean"] for r in rows if r["delta_vs_no_context"].get(b, {}).get("mean") is not None]
                band_means[b] = float(np.mean(vs)) if vs else None
            n_deg90 = sum(1 for r in rows if r["verbatim_copy_diagnostics"]["exhibits_90_100_degradation_vs_no_context"])
            n_deg_sc = sum(1 for r in rows if r["verbatim_copy_diagnostics"]["exhibits_self_copy_degradation_vs_no_context"])
            out[k] = {
                "n_entries": len(rows),
                "mean_delta_vs_no_context_by_band": band_means,
                "n_exhibiting_id_90_100_degradation": n_deg90,
                "n_exhibiting_self_copy_degradation": n_deg_sc,
                "entries": [r["entry_key"] for r in rows],
            }
        return out

    # Explicit alignment-switch & ICL comparison slice
    key_comparators = [
        "msa-transformer-100m|aligned=True",
        "msa-transformer-100m|aligned=False",
        "protgpt3-msa",
        "protgpt3-msa|b8192",
        "dayhoff-3b-msa",
        "dayhoff-3b-msa|b4096",
    ]
    alignment_switch = {
        k: {
            "input_alignment_format": DMS_CONTEXT_SPEC_BY_KEY[k].input_alignment_format,
            "context_training_regime": DMS_CONTEXT_SPEC_BY_KEY[k].context_training_regime,
            "subtest_1a_rescue_30_80": entries_1a.get(k),
            "subtest_1b_e09_five_band": entries_1b.get(k),
        }
        for k in key_comparators
        if k in entries_1a or k in entries_1b
    }

    payload = {
        "schema": SCHEMA_DMS_UNIFIED,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if not (pending_1a or pending_1b) else "interim_partial",
        "protocol": {
            "priority_scope": priority_scope,
            "n_target_entries": len(target_specs),
            "n_completed_1a_rescue_30_80": len(entries_1a),
            "n_completed_1b_e09_five_band": len(entries_1b),
            "n_pending_1a": len(pending_1a),
            "n_pending_1b": len(pending_1b),
            "pending_1a": pending_1a,
            "pending_1b": pending_1b,
            "bootstrap_draws": int(draws),
            "bootstrap_seed": int(seed),
            "input_alignment_format_contract": {
                "explicit_column_aligned": "ONLY msa-transformer-100m|aligned=True (Needleman-Wunsch gap projection onto target columns)",
                "unaligned_concat": "protgpt3-msa (<no_gap>+<s>), dayhoff-3b-msa (@+/), msa-transformer-100m|aligned=False, and ALL single-chain models",
            },
        },
        "reproduction_gates": {
            "subtest_1a_profile_increment_20260923": repro_1a,
            "subtest_1b_e09_curve_20261008": repro_1b,
        },
        "subtest_1a_rescue_30_80": {
            "cohort_path": str(RESCUE_COHORT_REL),
            "common_support_164_reference": str(RESCUE_REF_PI_REL),
            "paradigm_rollup": _rollup_1a(),
            "simultaneous_bands_homolog_minus_unrelated": sim_1a_hu,
            "simultaneous_bands_homolog_minus_no_context": sim_1a_hn,
            "entries": entries_1a,
        },
        "subtest_1b_e09_five_band": {
            "cohort_path": str(E09_COHORT_REL),
            "curve_reference": str(E09_REF_CURVE_REL),
            "identity_bands_ordered": list(E09_BINS_ORDERED),
            "paradigm_rollup": _rollup_1b(),
            "entries": entries_1b,
        },
        "alignment_switch_and_icl_comparators": alignment_switch,
    }

    raw_txt = json.dumps(payload, sort_keys=True)
    if FORBIDDEN_INFRA_RE.search(raw_txt):
        raise ValueError("Refusing to emit output containing forbidden pod/node identifier")
    return payload


def write_r5_dms_context_unified(
    repo_root: Path = REPO_ROOT,
    out_dir: Path | None = None,
    *,
    allow_pending: bool = False,
    priority_scope: str = "core",
    draws: int = 10000,
    seed: int = 20261124,
) -> dict[str, Any]:
    """Assemble and atomically write ``r5_dms_context_ranking_unified.json`` + ``.sha256``."""
    repo_root = Path(repo_root)
    target_dir = Path(out_dir) if out_dir is not None else (repo_root / UNIFIED_OUTPUT_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    doc = build_r5_dms_context_unified(
        repo_root,
        allow_pending=allow_pending,
        priority_scope=priority_scope,
        draws=draws,
        seed=seed,
    )
    out_path = target_dir / "r5_dms_context_ranking_unified.json"
    digest = write_json_with_sha256(out_path, doc)
    return {"path": str(out_path), "sha256": digest, "doc": doc}


def sha256_file_hex(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
