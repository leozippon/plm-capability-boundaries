"""Unified 65-arm R4 unaligned homologous-prefix scoring yardstick (EXP-R2-228 cohort).

Covers 100% of the 65-arm repository universe on the frozen 800-unit retained protein
cohort (``results/R5/context_homologue_20260826/cohort.json``, 710 independent family
clusters across ``id_70_90``, ``id_50_70``, ``id_30_50``, ``id_lt_30``; ``budget_tokens = 1024``):

  - 19 cached early EXP-R2-228 protein causal arms (reused bit-for-bit)
  -  7 cached GP single-chain protein causal arms at budget 1024 (reused bit-for-bit)
  -  3 cached MSA comparator arms at budget 1024 (reused bit-for-bit)
  - 14 Group A causal text arms scored on the frozen protein cohort (13 previously
       unscored + ``gpt2-large`` protein-mode alongside its frozen text ``bm25_*`` control)
  - 22 Group B bidirectional / masked / structure-aware / discrete-diffusion / multimodal
       single-chain arms scored via paired ``pll_8fold`` (primary, exact 8-fold ~12.5%
       complementary masking) + ``full_target_masked`` (companion 100%-masked 1-pass CE)

Total: 19 + 7 + 14 + 22 = 62 single-sequence protein arms + 3 aligned MSA arms = 65/65.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.capability.extensions.bidirectional import (
    counts as band_verdict_counts,
    paired_bands,
    verdict as band_verdicts,
)
from src.capability.extensions.family_distribution import _family_inference
from src.capability.general_plm.eval_utils import cluster_bootstrap_mean, sha256_file
from src.capability.memory_theory.io_utils import (
    cluster_bootstrap_scaling_slope,
    load_json_file,
    write_json_with_sha256,
)
from src.capability.unified.master_summary import FORBIDDEN_INFRA_RE
from src.capability.unified.roster import (
    ARM_METADATA,
    PARADIGM_ORDER,
    REPO_ROOT,
    UNIFIED_3_MSA_CONTEXT_ARMS,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_65_ARMS,
    UNIFIED_OUTPUT_DIR,
    ResolvedArtifact,
    default_scan_dirs,
    dtype_rank,
    get_arm_metadata,
    resolve_gp_r4_json,
)

SCHEMA_ARM_RUNNER = "unified_r4_unaligned_scoring_arm_v1"
SCHEMA_MASTER = "unified_r4_scoring_65_v1"

DEFAULT_COHORT_REL = Path("results/R5/context_homologue_20260826/cohort.json")
DEFAULT_WORK_SUBDIR = UNIFIED_OUTPUT_DIR / "r4_unaligned_scoring"
DEFAULT_OUTPUT_REL = UNIFIED_OUTPUT_DIR / "r4_scoring_unified_65.json"

CONDITIONS: tuple[str, ...] = (
    "position_only",
    "mono_shuffled",
    "no_context",
    "unrelated",
    "homologue",
)
IDENTITY_BANDS: tuple[str, ...] = ("id_70_90", "id_50_70", "id_30_50", "id_lt_30")

# 19 cached early EXP-R2-228 protein arms (reused bit-for-bit).
CACHED_EARLY_PROTEIN_ARMS: tuple[str, ...] = (
    "llama-2-7b",
    "galactica-125m",
    "galactica-1.3b",
    "galactica-6.7b",
    "galactica-30b",
    "instructprotein",
    "prollama",
    "prollama-stage-1",
    "progen2-small",
    "progen2-base",
    "progen2-medium",
    "progen2-large",
    "progen2-xlarge",
    "progen3-112m",
    "progen3-3b",
    "proteinglm-7b-clm",
    "protgpt2",
    "protgpt3-1.3b",
    "rita-xl",
)

# 7 cached GP single-chain causal arms at budget_tokens=1024 (reused bit-for-bit).
CACHED_GP_SINGLE_ARMS: tuple[str, ...] = (
    "dayhoff-170m",
    "progen3-219m",
    "progen3-339m",
    "progen3-762m",
    "progen3-1b",
    "protgpt3-112m",
    "protgpt3-10b",
)

# 3 cached aligned MSA comparator arms at budget_tokens=1024 (reused bit-for-bit).
CACHED_MSA_ARMS: tuple[str, ...] = UNIFIED_3_MSA_CONTEXT_ARMS

# 14 Group A causal text arms to score on the frozen protein cohort
# (13 missing causal text arms + gpt2-large protein-mode; gpt2-large text bm25_* stays as companion).
GROUP_A_CAUSAL_TEXT_ARMS: tuple[str, ...] = (
    "bygpt5-small-en",
    "bygpt5-base-en",
    "bygpt5-medium-en",
    "dialogpt-small",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "llama-3.2-3b",
    "qwen2.5-0.5b",
    "qwen2.5-0.5b-instruct",
    "qwen2.5-7b",
    "qwen2.5-32b",
    "qwen3-8b-base",
)

# 22 Group B bidirectional / masked / structure-aware / diffusion / multimodal single-chain arms.
GROUP_B_BIDIRECTIONAL_ARMS: tuple[str, ...] = (
    "esm2-8m",
    "esm2-35m",
    "esm2-150m",
    "esm2-650m",
    "esm2-3b",
    "esm2-15b",
    "esmc-300m",
    "esmc-600m",
    "esmc-6b",
    "proteinglm-1b-mlm",
    "proteinglm-3b-mlm",
    "proteinglm-10b-mlm",
    "amplify-120m",
    "amplify-350m",
    "prime-690m",
    "saprot-35m",
    "saprot-650m",
    "saprot-1.3b",
    "dplm-150m",
    "dplm-650m",
    "dplm-3b",
    "esm3-1.4b",
)

NEW_SCORE_ARMS: tuple[str, ...] = GROUP_A_CAUSAL_TEXT_ARMS + GROUP_B_BIDIRECTIONAL_ARMS

# Scaling ladders covering all 13 multi-rung families across text, joint, causal, MLM, 3Di, and diffusion.
R4_SCALING_LADDERS: Mapping[str, tuple[str, ...]] = {
    "bygpt5": ("bygpt5-small-en", "bygpt5-base-en", "bygpt5-medium-en"),
    "gpt2": ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"),
    "qwen2.5": ("qwen2.5-0.5b", "qwen2.5-7b", "qwen2.5-32b"),
    "galactica": ("galactica-125m", "galactica-1.3b", "galactica-6.7b", "galactica-30b"),
    "progen2": ("progen2-small", "progen2-base", "progen2-medium", "progen2-large", "progen2-xlarge"),
    "progen3": ("progen3-112m", "progen3-219m", "progen3-339m", "progen3-762m", "progen3-1b", "progen3-3b"),
    "protgpt3": ("protgpt3-112m", "protgpt3-1.3b", "protgpt3-10b"),
    "esm2": ("esm2-8m", "esm2-35m", "esm2-150m", "esm2-650m", "esm2-3b", "esm2-15b"),
    "esmc": ("esmc-300m", "esmc-600m", "esmc-6b"),
    "proteinglm_mlm": ("proteinglm-1b-mlm", "proteinglm-3b-mlm", "proteinglm-10b-mlm"),
    "amplify": ("amplify-120m", "amplify-350m"),
    "saprot": ("saprot-35m", "saprot-650m", "saprot-1.3b"),
    "dplm": ("dplm-150m", "dplm-650m", "dplm-3b"),
}

# Structural invariants verified once at import time.
assert set(CACHED_EARLY_PROTEIN_ARMS) | set(CACHED_GP_SINGLE_ARMS) | set(NEW_SCORE_ARMS) == set(
    UNIFIED_62_SINGLE_SEQUENCE_ARMS
), "Single-sequence R4 partition does not match UNIFIED_62_SINGLE_SEQUENCE_ARMS"
assert len(CACHED_EARLY_PROTEIN_ARMS) + len(CACHED_GP_SINGLE_ARMS) + len(NEW_SCORE_ARMS) == 62
assert len(CACHED_EARLY_PROTEIN_ARMS) + len(CACHED_GP_SINGLE_ARMS) + len(NEW_SCORE_ARMS) + len(
    CACHED_MSA_ARMS
) == len(UNIFIED_65_ARMS) == 65


# ============================================================================
# Deterministic Cohort Input Construction & Mask Partitioning
# ============================================================================


def deterministic_shuffle(seq: str, seed_key: str) -> str:
    """Bit-for-bit identical deterministic monoresidue shuffle matching ``run_r4_multiseq_context.py``."""
    digest = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(digest)
    chars = list(seq)
    perm = rng.permutation(len(chars))
    return "".join(chars[int(i)] for i in perm)


def pick_unrelated_partners(
    unit: Mapping[str, Any],
    records: Sequence[str],
    related_map: Mapping[Any, Sequence[int]] | Sequence[Sequence[int]],
    n_needed: int,
    *,
    filler_idx: int,
) -> list[str]:
    """Deterministic matched-length unrelated records matching ``run_r4_multiseq_context.py`` bit-for-bit."""
    target_idx = int(unit["target"])
    if isinstance(related_map, Mapping):
        forbidden = set(related_map.get(str(target_idx), ())) | set(related_map.get(target_idx, ()))
    else:
        forbidden = set(related_map[target_idx]) if target_idx < len(related_map) else set()
    forbidden.add(target_idx)
    forbidden.add(filler_idx)
    forbidden.update(int(p) for p in unit["partners"])

    digest = int(hashlib.sha256(str(unit["key"]).encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(digest)
    n_rec = len(records)
    picked: list[str] = []
    used: set[int] = set()
    partner_lens = [len(records[int(p)]) for p in unit["partners"][:n_needed]]

    for want_len in partner_lens:
        best_idx = None
        best_gap = 10**9
        cands = rng.integers(0, n_rec, size=128)
        for cand in cands:
            ci = int(cand)
            if ci in forbidden or ci in used:
                continue
            gap = abs(len(records[ci]) - want_len)
            if gap < best_gap:
                best_gap = gap
                best_idx = ci
                if gap == 0:
                    break
        if best_idx is None:
            for ci in range(n_rec):
                if ci not in forbidden and ci not in used:
                    best_idx = ci
                    break
        if best_idx is not None:
            used.add(best_idx)
            picked.append(records[best_idx])
    return picked


def build_unit_condition_inputs(
    unit: Mapping[str, Any],
    records: Sequence[str],
    related_map: Mapping[Any, Sequence[int]] | Sequence[Sequence[int]],
    *,
    filler_idx: int,
    filler_seq: str,
) -> tuple[str, dict[str, list[str]]]:
    """Build deterministic ``(target_seq, {condition: context_items})`` for one R4 cohort unit."""
    target_seq = records[int(unit["target"])]
    homolog_seqs = [records[int(p)] for p in unit["partners"]]
    unrelated_seqs = pick_unrelated_partners(
        unit, records, related_map, len(homolog_seqs), filler_idx=filler_idx
    )
    shuffled_seqs = [
        deterministic_shuffle(seq, f"{unit['key']}|mono|{i}") for i, seq in enumerate(homolog_seqs)
    ]
    filler_seqs = [
        (filler_seq * ((len(seq) // len(filler_seq)) + 1))[: len(seq)] for seq in homolog_seqs
    ]
    return target_seq, {
        "position_only": filler_seqs,
        "mono_shuffled": shuffled_seqs,
        "no_context": [],
        "unrelated": unrelated_seqs,
        "homologue": homolog_seqs,
    }


def build_8fold_masks(unit_key: str, target_len: int, *, n_folds: int = 8) -> list[list[int]]:
    """Deterministically partition ``0..target_len-1`` into ``min(n_folds, target_len)`` disjoint folds.

    Every residue is masked and scored exactly once; the partition depends solely on
    ``(unit_key, target_len)`` so all 5 conditions of a unit share identical mask folds.
    """
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    k_folds = min(int(n_folds), int(target_len))
    digest = int(hashlib.sha256(f"{unit_key}|r4_8fold".encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(digest)
    perm = rng.permutation(int(target_len))
    splits = np.array_split(perm, k_folds)
    folds = [sorted(int(i) for i in chunk.tolist()) for chunk in splits if len(chunk) > 0]
    covered = sorted(idx for fold in folds for idx in fold)
    if covered != list(range(target_len)):
        raise AssertionError("8-fold partition failed exact disjoint coverage invariant")
    return folds


def pack_bidirectional_context(
    target_residue_ids: Sequence[int],
    context_residue_ids_list: Sequence[Sequence[int]],
    *,
    bos_prefix: Sequence[int],
    eos_suffix: Sequence[int],
    sep_ids: Sequence[int],
    budget_tokens: int = 1024,
) -> tuple[list[int], int, int, int]:
    """Pack ``[bos, item_1, sep, ..., item_k, sep, target, eos]`` strictly inside ``budget_tokens``.

    Returns ``(full_ids, target_start, target_end, k_items)`` where
    ``full_ids[target_start:target_end] == list(target_residue_ids)`` and
    ``len(full_ids) <= budget_tokens``.
    """
    bos = list(bos_prefix)
    eos = list(eos_suffix)
    sep = list(sep_ids)
    tgt = list(target_residue_ids)

    base_len = len(bos) + len(tgt) + len(eos)
    if base_len > budget_tokens:
        raise ValueError(
            f"Target sequence ({len(tgt)} residues; base_len={base_len}) exceeds budget_tokens={budget_tokens}"
        )
    remaining = budget_tokens - base_len

    prefix_body: list[int] = []
    used_k = 0
    for item_ids in context_residue_ids_list:
        chunk = list(item_ids) + sep
        if len(prefix_body) + len(chunk) > remaining:
            break
        prefix_body.extend(chunk)
        used_k += 1

    full_ids = bos + prefix_body + tgt + eos
    target_start = len(bos) + len(prefix_body)
    target_end = target_start + len(tgt)
    assert len(full_ids) <= budget_tokens
    assert full_ids[target_start:target_end] == tgt
    return full_ids, target_start, target_end, used_k


def pack_causal_text_context(
    target_ids: Sequence[int],
    context_ids_list: Sequence[Sequence[int]],
    *,
    mode: str,
    bos_id: int | None,
    eos_id: int | None,
    budget_tokens: int = 1024,
) -> tuple[list[int], int, int, int]:
    """Pack independently tokenized context items + target inside ``budget_tokens``.

    Modes:
      - ``sep_prefix`` (GPT-2, DialoGPT, Qwen2.5, Qwen3):
          ``[sep, *item_1, sep, ..., sep, *target]`` scoring all ``len(target_ids)`` tokens.
      - ``bos_eos_blocks`` (LLaMA-3.2):
          ``[bos, *item_1, eos, ..., bos, *item_k, eos, bos, *target]`` scoring all ``len(target_ids)`` tokens.
      - ``bare_eos_sep`` (ByGPT5):
          ``[*item_1, eos, ..., *item_k, eos, *target]`` scoring ``target[1:]`` (``len(target_ids) - 1`` tokens)
          identically across all 5 conditions including ``no_context`` without prepending an OOD prefix.

    Returns ``(full_ids, span_start, span_end, k_items)`` where logits at ``pos - 1`` predict
    ``full_ids[pos]`` for ``pos in range(span_start, span_end)``.
    """
    tgt = list(target_ids)
    if not tgt:
        raise ValueError("Empty target token sequence")

    if mode == "sep_prefix":
        if eos_id is None:
            raise ValueError("sep_prefix mode requires a non-None separator/eos token id")
        sep = [int(eos_id)]
        target_block = sep + tgt
        rel_start = 1
        rel_end = 1 + len(tgt)
        remaining = budget_tokens - len(target_block)
        if remaining < 0:
            raise ValueError(f"Target ({len(target_block)} toks) exceeds budget_tokens={budget_tokens}")
        prefix: list[int] = []
        used_k = 0
        for item_ids in context_ids_list:
            chunk = sep + list(item_ids)
            if len(prefix) + len(chunk) > remaining:
                break
            prefix.extend(chunk)
            used_k += 1
        full_ids = prefix + target_block
        return full_ids, len(prefix) + rel_start, len(prefix) + rel_end, used_k

    if mode == "bos_eos_blocks":
        if bos_id is None or eos_id is None:
            raise ValueError("bos_eos_blocks mode requires non-None bos_id and eos_id")
        bos = [int(bos_id)]
        eos = [int(eos_id)]
        target_block = bos + tgt
        rel_start = 1
        rel_end = 1 + len(tgt)
        remaining = budget_tokens - len(target_block)
        if remaining < 0:
            raise ValueError(f"Target ({len(target_block)} toks) exceeds budget_tokens={budget_tokens}")
        prefix = []
        used_k = 0
        for item_ids in context_ids_list:
            chunk = bos + list(item_ids) + eos
            if len(prefix) + len(chunk) > remaining:
                break
            prefix.extend(chunk)
            used_k += 1
        full_ids = prefix + target_block
        return full_ids, len(prefix) + rel_start, len(prefix) + rel_end, used_k

    if mode == "bare_eos_sep":
        if eos_id is None:
            raise ValueError("bare_eos_sep mode requires non-None eos_id")
        if len(tgt) < 2:
            raise ValueError("bare_eos_sep mode requires target length >= 2")
        eos = [int(eos_id)]
        remaining = budget_tokens - len(tgt)
        if remaining < 0:
            raise ValueError(f"Target ({len(tgt)} toks) exceeds budget_tokens={budget_tokens}")
        prefix = []
        used_k = 0
        for item_ids in context_ids_list:
            chunk = list(item_ids) + eos
            if len(prefix) + len(chunk) > remaining:
                break
            prefix.extend(chunk)
            used_k += 1
        full_ids = prefix + tgt
        # Score target[1:] across all 5 conditions (including no_context) for exact token-grid invariance.
        return full_ids, len(prefix) + 1, len(prefix) + len(tgt), used_k

    raise ValueError(f"Unsupported causal text packing mode: {mode!r}")


# ============================================================================
# Per-Arm Band Summarization & Simultaneous 65-Arm Aggregation
# ============================================================================


def summarize_unit_rows(unit_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute per-band and pooled cluster-bootstrap summaries from standardized unit rows."""
    bands_summary: dict[str, Any] = {}
    for band in IDENTITY_BANDS:
        sub = [r for r in unit_rows if r["band"] == band]
        if not sub:
            continue
        bands_summary[band] = _summarize_subset(sub)
    pooled = _summarize_subset(unit_rows) if unit_rows else {}
    return {"bands": bands_summary, "pooled": pooled}


def _summarize_subset(sub: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups = [int(r["group"]) for r in sub]
    hom_unr = [float(r["hom_beats_unrelated"]) for r in sub]
    hom_shuf = [float(r["hom_beats_mono_shuffled"]) for r in sub if r.get("hom_beats_mono_shuffled") is not None]
    hom_shuf_g = [int(r["group"]) for r in sub if r.get("hom_beats_mono_shuffled") is not None]
    frac_unr = [float(r["fractional_reduction_vs_unrelated"]) for r in sub if r.get("fractional_reduction_vs_unrelated") is not None]
    frac_unr_g = [int(r["group"]) for r in sub if r.get("fractional_reduction_vs_unrelated") is not None]
    frac_shuf = [float(r["fractional_reduction_vs_shuffled"]) for r in sub if r.get("fractional_reduction_vs_shuffled") is not None]
    frac_shuf_g = [int(r["group"]) for r in sub if r.get("fractional_reduction_vs_shuffled") is not None]
    k_vals = [float(r["k_homologue"]) for r in sub if r.get("k_homologue") is not None]
    return {
        "n_units": len(sub),
        "n_groups": len(set(groups)),
        "auroc_hom_vs_unrelated": cluster_bootstrap_mean(hom_unr, groups, seed=20261125),
        "auroc_hom_vs_mono_shuffled": (
            cluster_bootstrap_mean(hom_shuf, hom_shuf_g, seed=20261126) if hom_shuf else None
        ),
        "fractional_reduction_vs_unrelated": (
            cluster_bootstrap_mean(frac_unr, frac_unr_g, seed=20261127) if frac_unr else None
        ),
        "fractional_reduction_vs_shuffled": (
            cluster_bootstrap_mean(frac_shuf, frac_shuf_g, seed=20261128) if frac_shuf else None
        ),
        "mean_k_homologue": float(np.mean(k_vals)) if k_vals else None,
    }


def _resolve_gp_r4_any(
    arm_id: str,
    *,
    budget_tokens: int = 1024,
    aligned_msa: bool = True,
    repo_root: Path = REPO_ROOT,
) -> ResolvedArtifact | None:
    try:
        return resolve_gp_r4_json(
            arm_id, budget_tokens=budget_tokens, aligned_msa=aligned_msa, root=repo_root
        )
    except FileNotFoundError:
        pass
    best: ResolvedArtifact | None = None
    for scan_dir in default_scan_dirs(repo_root):
        if not scan_dir.is_dir():
            continue
        for path in sorted(scan_dir.rglob(f"r4_multiseq_{arm_id}*.json")):
            try:
                doc = load_json_file(path)
            except Exception:
                continue
            if not isinstance(doc, dict) or doc.get("arm") != arm_id:
                continue
            if int(doc.get("budget_tokens", -1)) != int(budget_tokens):
                continue
            if bool(doc.get("aligned_msa", True)) != bool(aligned_msa):
                continue
            cand_dtype = (doc.get("provenance") or {}).get("dtype_realised")
            if best is None or dtype_rank(cand_dtype) >= dtype_rank(best.dtype_realised):
                best = ResolvedArtifact(
                    arm_id=arm_id,
                    stage=f"r4_b{budget_tokens}" + ("" if aligned_msa else "_unaligned"),
                    path=path,
                    dtype_realised=str(cand_dtype) if cand_dtype else None,
                    sibling_json=path,
                )
    return best


def _load_early_protein_arm(
    arm_id: str,
    early_records: Mapping[str, tuple[Mapping[str, Any], str]],
) -> dict[str, Any]:
    """Load one frozen EXP-R2-228 protein arm and extract exact 800-unit retained rows + parity checks."""
    rec, rel_src = early_records[arm_id]
    if rec.get("modality") != "protein":
        raise ValueError(f"{arm_id}: expected protein modality, got {rec.get('modality')!r}")
    retained_rows: list[dict[str, Any]] = []
    for u in rec.get("per_unit") or []:
        if u.get("stratum") != "retained":
            continue
        nll_hom = float(u["homologue"])
        nll_unr = float(u["unrelated"])
        nll_shuf = float(u["mono_shuffled"])
        nll_pos = float(u["position_only"])
        retained_rows.append(
            {
                "key": str(u["key"]),
                "band": str(u["band"]),
                "stratum": "retained",
                "group": int(u["group"]),
                "k_homologue": int(u["k"]),
                "nll_homologue": nll_hom,
                "nll_unrelated": nll_unr,
                "nll_mono_shuffled": nll_shuf,
                "nll_position_only": nll_pos,
                "nll_no_context": float(u["no_context"]),
                "hom_beats_unrelated": float(u["auroc"]),
                "hom_beats_mono_shuffled": 1.0 if nll_hom < nll_shuf else (0.5 if nll_hom == nll_shuf else 0.0),
                "fractional_reduction_vs_unrelated": float(u["fractional_reduction"]),
                "fractional_reduction_vs_shuffled": (nll_shuf - nll_hom) / nll_pos if nll_pos > 0 else None,
            }
        )
    if len(retained_rows) != 800:
        raise ValueError(f"{arm_id}: expected 800 retained units, got {len(retained_rows)}")

    # Exact parity check against frozen per-band reported means
    max_diff = 0.0
    for band in IDENTITY_BANDS:
        frozen_ret = rec["bands"][band]["retained"]
        sub = [r for r in retained_rows if r["band"] == band]
        recomp_auroc = float(np.mean([r["hom_beats_unrelated"] for r in sub]))
        recomp_frac = float(np.mean([r["fractional_reduction_vs_unrelated"] for r in sub]))
        max_diff = max(
            max_diff,
            abs(recomp_auroc - float(frozen_ret["auroc"]["mean"])),
            abs(recomp_frac - float(frozen_ret["fractional_reduction"]["mean"])),
        )

    return {
        "arm_id": arm_id,
        "source_group": "cached_early_exp_r2_228",
        "context_paradigm": "single_chain_causal",
        "primary_estimand": "exact_causal_nll",
        "budget_tokens": 1024,
        "aligned_msa": None,
        "dtype_realised": "float32",
        "source_path": rel_src,
        "parity_max_abs_diff": max_diff,
        "units": retained_rows,
        "companion_full_target_masked_units": None,
        "exact_pll_diagnostic": None,
    }


def _load_gp_cached_arm(
    arm_id: str,
    repo_root: Path,
    *,
    context_paradigm: str,
) -> dict[str, Any]:
    art = _resolve_gp_r4_any(arm_id, budget_tokens=1024, aligned_msa=True, repo_root=repo_root)
    if art is None:
        raise FileNotFoundError(f"Missing cached GP R4 b1024 artifact for {arm_id}")
    doc = load_json_file(art.path)
    units_out: list[dict[str, Any]] = []
    for u in doc.get("units") or []:
        if u.get("stratum", "retained") != "retained":
            continue
        conds = u["conditions"]
        units_out.append(
            {
                "key": str(u["key"]),
                "band": str(u["band"]),
                "stratum": "retained",
                "group": int(u["group"]),
                "k_homologue": int(conds["homologue"]["k_items"]),
                "nll_homologue": float(conds["homologue"]["nll_per_residue"]),
                "nll_unrelated": float(conds["unrelated"]["nll_per_residue"]),
                "nll_mono_shuffled": float(conds["mono_shuffled"]["nll_per_residue"]),
                "nll_position_only": float(conds["position_only"]["nll_per_residue"]),
                "nll_no_context": float(conds["no_context"]["nll_per_residue"]),
                "hom_beats_unrelated": float(u["hom_beats_unrelated"]),
                "hom_beats_mono_shuffled": float(u["hom_beats_mono_shuffled"]),
                "fractional_reduction_vs_unrelated": (
                    float(u["fractional_reduction_vs_unrelated"])
                    if u.get("fractional_reduction_vs_unrelated") is not None
                    else None
                ),
                "fractional_reduction_vs_shuffled": (
                    float(u["fractional_reduction_vs_shuffled"])
                    if u.get("fractional_reduction_vs_shuffled") is not None
                    else None
                ),
            }
        )
    if len(units_out) != 800:
        raise ValueError(f"{arm_id}: expected 800 retained GP units, got {len(units_out)}")

    max_diff = 0.0
    for band in IDENTITY_BANDS:
        frozen_b = doc["bands"][band]
        sub = [r for r in units_out if r["band"] == band]
        groups_b = [r["group"] for r in sub]
        recomp_a = cluster_bootstrap_mean([r["hom_beats_unrelated"] for r in sub], groups_b, seed=20261125)["mean"]
        recomp_f = cluster_bootstrap_mean(
            [r["fractional_reduction_vs_unrelated"] for r in sub], groups_b, seed=20261127
        )["mean"]
        max_diff = max(
            max_diff,
            abs(float(recomp_a) - float(frozen_b["auroc_hom_vs_unrelated"]["mean"])),
            abs(float(recomp_f) - float(frozen_b["fractional_reduction_vs_unrelated"]["mean"])),
        )

    return {
        "arm_id": arm_id,
        "source_group": "cached_gp_r4",
        "context_paradigm": context_paradigm,
        "primary_estimand": "msa_masked_pll" if arm_id == "msa-transformer-100m" else "exact_causal_nll",
        "budget_tokens": 1024,
        "aligned_msa": bool(doc.get("aligned_msa", True)),
        "dtype_realised": art.dtype_realised or str((doc.get("provenance") or {}).get("dtype_realised", "unknown")),
        "source_path": str(art.path.relative_to(repo_root)),
        "parity_max_abs_diff": max_diff,
        "units": units_out,
        "companion_full_target_masked_units": None,
        "exact_pll_diagnostic": None,
    }


def _load_new_r4_arm(arm_id: str, work_dir: Path, repo_root: Path) -> dict[str, Any] | None:
    path = work_dir / f"r4_unaligned_{arm_id}.json"
    if not path.is_file():
        return None
    doc = load_json_file(path)
    if doc.get("schema") != SCHEMA_ARM_RUNNER:
        raise ValueError(f"{path}: unexpected schema {doc.get('schema')!r}")
    if doc.get("arm") != arm_id:
        raise ValueError(f"{path}: arm mismatch {doc.get('arm')!r} != {arm_id!r}")
    units_raw = doc.get("units") or []
    if len(units_raw) != 800:
        raise ValueError(f"{path}: expected 800 retained units, got {len(units_raw)}")

    is_group_b = arm_id in GROUP_B_BIDIRECTIONAL_ARMS
    primary_units: list[dict[str, Any]] = []
    full_mask_units: list[dict[str, Any]] | None = [] if is_group_b else None

    for u in units_raw:
        conds = u["conditions"]
        if is_group_b:
            p8 = u["pll_8fold"]
            primary_units.append(
                {
                    "key": str(u["key"]),
                    "band": str(u["band"]),
                    "stratum": "retained",
                    "group": int(u["group"]),
                    "k_homologue": int(conds["homologue"]["k_items"]),
                    "nll_homologue": float(conds["homologue"]["pll_8fold"]["nll_per_residue"]),
                    "nll_unrelated": float(conds["unrelated"]["pll_8fold"]["nll_per_residue"]),
                    "nll_mono_shuffled": float(conds["mono_shuffled"]["pll_8fold"]["nll_per_residue"]),
                    "nll_position_only": float(conds["position_only"]["pll_8fold"]["nll_per_residue"]),
                    "nll_no_context": float(conds["no_context"]["pll_8fold"]["nll_per_residue"]),
                    "hom_beats_unrelated": float(p8["hom_beats_unrelated"]),
                    "hom_beats_mono_shuffled": float(p8["hom_beats_mono_shuffled"]),
                    "fractional_reduction_vs_unrelated": p8.get("fractional_reduction_vs_unrelated"),
                    "fractional_reduction_vs_shuffled": p8.get("fractional_reduction_vs_shuffled"),
                }
            )
            fm = u["full_target_masked"]
            assert full_mask_units is not None
            full_mask_units.append(
                {
                    "key": str(u["key"]),
                    "band": str(u["band"]),
                    "stratum": "retained",
                    "group": int(u["group"]),
                    "k_homologue": int(conds["homologue"]["k_items"]),
                    "nll_homologue": float(conds["homologue"]["full_target_masked"]["nll_per_residue"]),
                    "nll_unrelated": float(conds["unrelated"]["full_target_masked"]["nll_per_residue"]),
                    "nll_mono_shuffled": float(conds["mono_shuffled"]["full_target_masked"]["nll_per_residue"]),
                    "nll_position_only": float(conds["position_only"]["full_target_masked"]["nll_per_residue"]),
                    "nll_no_context": float(conds["no_context"]["full_target_masked"]["nll_per_residue"]),
                    "hom_beats_unrelated": float(fm["hom_beats_unrelated"]),
                    "hom_beats_mono_shuffled": float(fm["hom_beats_mono_shuffled"]),
                    "fractional_reduction_vs_unrelated": fm.get("fractional_reduction_vs_unrelated"),
                    "fractional_reduction_vs_shuffled": fm.get("fractional_reduction_vs_shuffled"),
                }
            )
        else:
            primary_units.append(
                {
                    "key": str(u["key"]),
                    "band": str(u["band"]),
                    "stratum": "retained",
                    "group": int(u["group"]),
                    "k_homologue": int(conds["homologue"]["k_items"]),
                    "nll_homologue": float(conds["homologue"]["nll_per_residue"]),
                    "nll_unrelated": float(conds["unrelated"]["nll_per_residue"]),
                    "nll_mono_shuffled": float(conds["mono_shuffled"]["nll_per_residue"]),
                    "nll_position_only": float(conds["position_only"]["nll_per_residue"]),
                    "nll_no_context": float(conds["no_context"]["nll_per_residue"]),
                    "hom_beats_unrelated": float(u["hom_beats_unrelated"]),
                    "hom_beats_mono_shuffled": float(u["hom_beats_mono_shuffled"]),
                    "fractional_reduction_vs_unrelated": u.get("fractional_reduction_vs_unrelated"),
                    "fractional_reduction_vs_shuffled": u.get("fractional_reduction_vs_shuffled"),
                }
            )

    rel_src = str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path)
    return {
        "arm_id": arm_id,
        "source_group": "new_group_b_bidirectional" if is_group_b else "new_group_a_causal_text",
        "context_paradigm": "single_chain_bidirectional_unaligned" if is_group_b else "single_chain_causal",
        "primary_estimand": "pll_8fold" if is_group_b else "exact_causal_nll",
        "budget_tokens": int(doc.get("budget_tokens", 1024)),
        "aligned_msa": False,
        "dtype_realised": str((doc.get("provenance") or {}).get("dtype_realised", "unknown")),
        "source_path": rel_src,
        "source_sha256": sha256_file(path),
        "parity_max_abs_diff": None,
        "units": primary_units,
        "companion_full_target_masked_units": full_mask_units,
        "exact_pll_diagnostic": doc.get("exact_pll_diagnostic"),
    }


def _cluster_matrix(
    arm_ids: Sequence[str],
    loaded_arms: Mapping[str, Mapping[str, Any]],
    *,
    band: str | None,
    metric_key: str,
    null_shift: float = 0.0,
    use_companion_full_mask: bool = False,
) -> tuple[list[int], np.ndarray]:
    """Aggregate unit metrics to exact cluster-level means across all arms (shared cluster order)."""
    first_units = loaded_arms[arm_ids[0]][
        "companion_full_target_masked_units" if use_companion_full_mask else "units"
    ]
    groups = sorted({int(u["group"]) for u in first_units if band is None or u["band"] == band})
    mat = np.empty((len(groups), len(arm_ids)), dtype=np.float64)
    g_to_idx = {g: i for i, g in enumerate(groups)}

    for col_idx, arm_id in enumerate(arm_ids):
        units = loaded_arms[arm_id][
            "companion_full_target_masked_units" if use_companion_full_mask else "units"
        ]
        buckets: list[list[float]] = [[] for _ in groups]
        for u in units:
            if band is not None and u["band"] != band:
                continue
            val = u.get(metric_key)
            if val is None or not math.isfinite(float(val)):
                raise ValueError(f"{arm_id} ({band}, {metric_key}): non-finite unit value {val!r}")
            buckets[g_to_idx[int(u["group"])]].append(float(val) - null_shift)
        for row_idx, vals in enumerate(buckets):
            if not vals:
                raise ValueError(f"{arm_id}: missing cluster {groups[row_idx]} in band={band}")
            mat[row_idx, col_idx] = float(np.mean(vals))
    return groups, mat


def _simultaneous_family(
    arm_ids: Sequence[str],
    matrix: np.ndarray,
    *,
    draws: int = 10000,
    seed: int = 20261124,
    null_offset: float = 0.0,
) -> dict[str, Any]:
    """Run native ``paired_bands`` + studentized ``maxT`` across the full arm family."""
    bands = paired_bands(matrix, draws=draws, seed=seed, confidence=0.95)
    verdicts = band_verdicts(bands["interval"])
    tally = band_verdict_counts(verdicts)

    stud = _family_inference(matrix, list(arm_ids), draws=draws, seed=seed)
    stud_hyps = stud["primary"]["hypotheses"]
    stud_intervals = [h["simultaneous"] for h in stud_hyps]
    stud_verdicts = band_verdicts(stud_intervals)
    stud_tally = band_verdict_counts(stud_verdicts)
    pw_intervals = stud["auxiliary_pointwise_95"]["interval"]

    per_arm: dict[str, Any] = {}
    for idx, arm_id in enumerate(arm_ids):
        pt = float(bands["point"][idx])
        lo, hi = float(bands["interval"][idx][0]), float(bands["interval"][idx][1])
        slo, shi = float(stud_intervals[idx][0]), float(stud_intervals[idx][1])
        plo, phi = float(pw_intervals[idx][0]), float(pw_intervals[idx][1])
        per_arm[arm_id] = {
            "point": pt + null_offset,
            "excess_over_null": pt,
            "simultaneous_ci95": [lo + null_offset, hi + null_offset],
            "studentized_ci95": [slo + null_offset, shi + null_offset],
            "pointwise_ci95": [plo + null_offset, phi + null_offset],
            "verdict": verdicts[idx],
            "studentized_verdict": stud_verdicts[idx],
        }
    return {
        "n_groups": int(matrix.shape[0]),
        "family_size": int(matrix.shape[1]),
        "draws": int(draws),
        "seed": int(seed),
        "null_value": float(null_offset),
        "native_critical_value": float(bands["critical_value"]),
        "studentized_critical_value": float(stud["primary"]["critical_value"]),
        "verdict_counts": tally,
        "studentized_verdict_counts": stud_tally,
        "per_arm": per_arm,
    }


def classify_r4_context_axes(
    arm_id: str,
    *,
    aligned_msa: bool | None = None,
    is_route_a_3d: bool = False,
) -> tuple[str, str]:
    """Return explicit ``(input_alignment_format, context_training_regime)`` for any R4 entry.

    Only ``msa-transformer-100m`` with ``aligned_msa=True`` receives ``explicit_column_aligned``
    (Needleman-Wunsch ``-`` gap projection onto target columns). ``protgpt3-msa`` (``<no_gap> + <s>``),
    ``dayhoff-3b-msa`` (``@ + /``), ``msa-transformer-100m|unaligned_control``, and all 62
    single-chain models receive ``unaligned_concat``.
    """
    if is_route_a_3d:
        return "structural_3d_ss8_aligned_tracks", "multimodal_3d_tracks"
    if arm_id == "msa-transformer-100m":
        fmt = "explicit_column_aligned" if (aligned_msa is not False) else "unaligned_concat"
        return fmt, "msa_2d_axial"
    if arm_id in ("protgpt3-msa", "dayhoff-3b-msa"):
        return "unaligned_concat", "multiseq_icl_unaligned"
    if arm_id in GROUP_B_BIDIRECTIONAL_ARMS:
        return "unaligned_concat", "single_chain_bidirectional"
    return "unaligned_concat", "single_chain_causal"


def _collect_companions(repo_root: Path) -> dict[str, Any]:
    """Load companion long-context, unaligned MSA, ESM3 Route A 3D/SS8, and gpt2-large text bm25_* controls."""
    companions: dict[str, Any] = {}

    # 1. gpt2-large frozen text bm25_* reference from EXP-R2-228
    early_228 = repo_root / "results/R5/context_homologue_20260826/context_homologue.json"
    if early_228.is_file():
        doc = load_json_file(early_228)
        g2l = (doc.get("arms") or {}).get("gpt2-large")
        if isinstance(g2l, dict) and g2l.get("modality") == "text":
            bands_out = {}
            for b, entry in sorted((g2l.get("bands") or {}).items()):
                ret = entry.get("retained", entry)
                bands_out[b] = {
                    "auroc_hom_vs_unrelated": ret.get("auroc"),
                    "fractional_reduction": ret.get("fractional_reduction"),
                }
            fmt, reg = classify_r4_context_axes("gpt2-large")
            companions["gpt2-large|text_bm25_frozen"] = {
                "arm_id": "gpt2-large",
                "modality": "text",
                "input_alignment_format": fmt,
                "context_training_regime": reg,
                "budget_tokens": 1024,
                "aligned_msa": None,
                "bands": bands_out,
                "source_path": str(early_228.relative_to(repo_root)),
            }

    # 2. Long-context & unaligned MSA comparators
    for key, arm_id, budget, aligned, paradigm in (
        ("dayhoff-3b-msa|b4096", "dayhoff-3b-msa", 4096, True, "msa_aligned"),
        ("protgpt3-msa|b8192", "protgpt3-msa", 8192, True, "msa_aligned"),
        ("msa-transformer-100m|unaligned_control", "msa-transformer-100m", 1024, False, "msa_unaligned_control"),
    ):
        art = _resolve_gp_r4_any(arm_id, budget_tokens=budget, aligned_msa=aligned, repo_root=repo_root)
        if art is not None:
            doc = load_json_file(art.path)
            fmt, reg = classify_r4_context_axes(arm_id, aligned_msa=aligned)
            companions[key] = {
                "arm_id": arm_id,
                "context_paradigm": paradigm,
                "input_alignment_format": fmt,
                "context_training_regime": reg,
                "budget_tokens": budget,
                "aligned_msa": aligned,
                "bands": doc.get("bands"),
                "source_path": str(art.path.relative_to(repo_root)),
            }

    # 3. ESM3-1.4B Route A 3D/SS8 multimodal context comparator
    for cand in (
        repo_root / "results/extensions/esm3_20261120/r4_multimodal/r4_multimodal_summary.json",
        repo_root / "results/extensions/general_plm_20261120/r4_multimodal/r4_multimodal_summary.json",
    ):
        if cand.is_file():
            doc = load_json_file(cand)
            fmt, reg = classify_r4_context_axes("esm3-1.4b", is_route_a_3d=True)
            companions["esm3-1.4b|route_a_3d_ss8"] = {
                "arm_id": "esm3-1.4b",
                "context_paradigm": "multimodal_3d_ss8_template",
                "input_alignment_format": fmt,
                "context_training_regime": reg,
                "source_path": str(cand.relative_to(repo_root)),
                "summary": doc.get("summary") or doc.get("bands") or doc,
            }
            break

    return companions


def _extract_compact_band_cell(b_entry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(b_entry, Mapping):
        return None
    a_unr = b_entry.get("auroc_hom_vs_unrelated") or {}
    a_shuf = b_entry.get("auroc_hom_vs_mono_shuffled") or {}
    frac = b_entry.get("fractional_reduction_vs_unrelated") or b_entry.get("fractional_reduction") or {}

    def _ci(rec: Mapping[str, Any]) -> list[float] | None:
        if "ci95" in rec and rec["ci95"] is not None:
            return [float(rec["ci95"][0]), float(rec["ci95"][1])]
        if rec.get("ci_low") is not None and rec.get("ci_high") is not None:
            return [float(rec["ci_low"]), float(rec["ci_high"])]
        return None

    return {
        "auroc_hom_vs_unrelated": float(a_unr["mean"]) if a_unr.get("mean") is not None else None,
        "auroc_hom_vs_unrelated_ci95": _ci(a_unr),
        "auroc_hom_vs_mono_shuffled": float(a_shuf["mean"]) if a_shuf.get("mean") is not None else None,
        "auroc_hom_vs_mono_shuffled_ci95": _ci(a_shuf),
        "fractional_reduction_vs_unrelated": float(frac["mean"]) if frac.get("mean") is not None else None,
        "fractional_reduction_vs_unrelated_ci95": _ci(frac),
        "mean_k_homologue": float(b_entry["mean_k_homologue"]) if b_entry.get("mean_k_homologue") is not None else None,
    }


def _build_horizontal_benchmark_table(
    arms_present: Sequence[str],
    per_arm_summary: Mapping[str, Mapping[str, Any]],
    companions: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the unified horizontal 65+3 cross-paradigm comparison table & regime-alignment rollup."""
    regime_order = {
        "single_chain_causal": 0,
        "single_chain_bidirectional": 1,
        "multiseq_icl_unaligned": 2,
        "msa_2d_axial": 3,
    }
    rows: list[dict[str, Any]] = []
    for arm_id in arms_present:
        rec = per_arm_summary[arm_id]
        bands_compact = {
            b: _extract_compact_band_cell((rec.get("bands") or {}).get(b))
            for b in ("id_lt_30", "id_30_50", "id_50_70", "id_70_90")
        }
        bands_compact["pooled"] = _extract_compact_band_cell(rec.get("pooled"))
        rows.append(
            {
                "row_key": f"{arm_id}|b{rec['budget_tokens']}",
                "arm_id": arm_id,
                "display_name": rec["display_name"],
                "family": rec["family"],
                "paradigm": rec["paradigm"],
                "params_m": float(rec["params_m"]),
                "budget_tokens": int(rec["budget_tokens"]),
                "aligned_msa": rec["aligned_msa"],
                "input_alignment_format": rec["input_alignment_format"],
                "context_training_regime": rec["context_training_regime"],
                "primary_estimand": rec["primary_estimand"],
                "is_companion_control": False,
                "bands": bands_compact,
            }
        )

    # Append the 3 sequence-context companion rows (native long-budget ICL + unaligned MSA-Transformer)
    for comp_key in (
        "protgpt3-msa|b8192",
        "dayhoff-3b-msa|b4096",
        "msa-transformer-100m|unaligned_control",
    ):
        comp = companions.get(comp_key)
        if not comp:
            continue
        arm_id = str(comp["arm_id"])
        meta = get_arm_metadata(arm_id)
        comp_bands = comp.get("bands") or {}
        bands_compact = {
            b: _extract_compact_band_cell(comp_bands.get(b))
            for b in ("id_lt_30", "id_30_50", "id_50_70", "id_70_90")
        }
        rows.append(
            {
                "row_key": comp_key,
                "arm_id": arm_id,
                "display_name": meta.display_name,
                "family": meta.family,
                "paradigm": meta.paradigm,
                "params_m": float(meta.params_m),
                "budget_tokens": int(comp["budget_tokens"]),
                "aligned_msa": bool(comp["aligned_msa"]),
                "input_alignment_format": comp["input_alignment_format"],
                "context_training_regime": comp["context_training_regime"],
                "primary_estimand": "msa_masked_pll" if arm_id == "msa-transformer-100m" else "exact_causal_nll",
                "is_companion_control": True,
                "bands": bands_compact,
            }
        )

    rows.sort(
        key=lambda r: (
            regime_order.get(r["context_training_regime"], 9),
            r["input_alignment_format"],
            r["is_companion_control"],
            PARADIGM_ORDER.index(r["paradigm"]) if r["paradigm"] in PARADIGM_ORDER else 99,
            r["params_m"],
            r["budget_tokens"],
            r["arm_id"],
        )
    )

    # Rollup grouped by (context_training_regime, input_alignment_format, budget_tier)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        b_tier = "b1024" if int(r["budget_tokens"]) == 1024 else f"native_b{r['budget_tokens']}"
        gk = f"{r['context_training_regime']}|{r['input_alignment_format']}|{b_tier}"
        buckets.setdefault(gk, []).append(r)

    rollup: dict[str, Any] = {}
    for gk, grp in sorted(buckets.items()):
        by_band: dict[str, Any] = {}
        for b in ("id_lt_30", "id_30_50", "id_50_70", "id_70_90"):
            aurocs = [
                r["bands"][b]["auroc_hom_vs_unrelated"]
                for r in grp
                if r["bands"].get(b) and r["bands"][b].get("auroc_hom_vs_unrelated") is not None
            ]
            fracs = [
                r["bands"][b]["fractional_reduction_vs_unrelated"]
                for r in grp
                if r["bands"].get(b) and r["bands"][b].get("fractional_reduction_vs_unrelated") is not None
            ]
            by_band[b] = {
                "auroc_median": float(np.median(aurocs)) if aurocs else None,
                "auroc_min": float(np.min(aurocs)) if aurocs else None,
                "auroc_max": float(np.max(aurocs)) if aurocs else None,
                "frac_reduction_median": float(np.median(fracs)) if fracs else None,
                "frac_reduction_min": float(np.min(fracs)) if fracs else None,
                "frac_reduction_max": float(np.max(fracs)) if fracs else None,
            }
        rollup[gk] = {
            "n_rows": len(grp),
            "context_training_regime": grp[0]["context_training_regime"],
            "input_alignment_format": grp[0]["input_alignment_format"],
            "row_keys": [r["row_key"] for r in grp],
            "by_band": by_band,
        }
    return rows, rollup


def build_r4_scoring_unified_65(
    *,
    repo_root: Path = REPO_ROOT,
    work_dir: Path | None = None,
    out_path: Path | None = None,
    allow_pending: bool = False,
    draws: int = 10000,
    seed: int = 20261124,
) -> dict[str, Any]:
    """Assemble ``r4_scoring_unified_65.json`` (+ ``.sha256``) across all 65 repository arms."""
    wdir = Path(work_dir) if work_dir is not None else (repo_root / DEFAULT_WORK_SUBDIR)
    dest = Path(out_path) if out_path is not None else (repo_root / DEFAULT_OUTPUT_REL)

    # 1. Load early cached EXP-R2-228 sources (later dates override earlier for ProGen3).
    early_records: dict[str, tuple[Mapping[str, Any], str]] = {}
    for rel in (
        "results/R5/context_homologue_20260826/context_homologue.json",
        "results/R5/context_homologue_expansion_20260919/context_homologue.json",
        "results/R5/context_homologue_progen3_20260924/context_homologue.json",
    ):
        fpath = repo_root / rel
        if fpath.is_file():
            doc = load_json_file(fpath)
            for arm_id, rec in (doc.get("arms") or {}).items():
                if arm_id in CACHED_EARLY_PROTEIN_ARMS:
                    early_records[arm_id] = (rec, rel)

    loaded_arms: dict[str, dict[str, Any]] = {}
    pending_arms: list[str] = []

    for arm_id in CACHED_EARLY_PROTEIN_ARMS:
        if arm_id not in early_records:
            pending_arms.append(arm_id)
        else:
            loaded_arms[arm_id] = _load_early_protein_arm(arm_id, early_records)

    for arm_id in CACHED_GP_SINGLE_ARMS:
        try:
            loaded_arms[arm_id] = _load_gp_cached_arm(arm_id, repo_root, context_paradigm="single_chain_causal")
        except FileNotFoundError:
            pending_arms.append(arm_id)

    for arm_id in CACHED_MSA_ARMS:
        try:
            loaded_arms[arm_id] = _load_gp_cached_arm(arm_id, repo_root, context_paradigm="msa_aligned")
        except FileNotFoundError:
            pending_arms.append(arm_id)

    for arm_id in NEW_SCORE_ARMS:
        rec = _load_new_r4_arm(arm_id, wdir, repo_root)
        if rec is None:
            pending_arms.append(arm_id)
        else:
            loaded_arms[arm_id] = rec

    if pending_arms and not allow_pending:
        raise RuntimeError(
            f"Strict 65-arm R4 scoring build refused: {len(pending_arms)} arms pending "
            f"(pass --allow-pending for interim build): {pending_arms}"
        )

    arms_present = [a for a in UNIFIED_65_ARMS if a in loaded_arms]
    group_b_present = [a for a in GROUP_B_BIDIRECTIONAL_ARMS if a in loaded_arms]

    # 2. Reproduction parity gate over all cached arms (< 1e-12)
    cached_parity: dict[str, float] = {
        a: float(loaded_arms[a]["parity_max_abs_diff"])
        for a in arms_present
        if loaded_arms[a].get("parity_max_abs_diff") is not None
    }
    max_parity_diff = max(cached_parity.values()) if cached_parity else 0.0
    parity_gate = {
        "n_cached_arms_verified": len(cached_parity),
        "max_abs_diff": max_parity_diff,
        "tolerance": 1e-12,
        "passed": bool(len(cached_parity) == 29 and max_parity_diff < 1e-12),
        "per_arm_max_abs_diff": cached_parity,
    }

    # 3. Per-arm summaries (compact — no raw 800-unit dump in master JSON to keep file crisp & readable)
    per_arm_summary: dict[str, Any] = {}
    exact_pll_diagnostics: dict[str, Any] = {}
    for arm_id in arms_present:
        rec = loaded_arms[arm_id]
        prim = summarize_unit_rows(rec["units"])
        comp_fm = (
            summarize_unit_rows(rec["companion_full_target_masked_units"])
            if rec.get("companion_full_target_masked_units") is not None
            else None
        )
        if rec.get("exact_pll_diagnostic") is not None:
            exact_pll_diagnostics[arm_id] = rec["exact_pll_diagnostic"]
        fmt, reg = classify_r4_context_axes(arm_id, aligned_msa=rec["aligned_msa"])
        per_arm_summary[arm_id] = {
            **asdict(get_arm_metadata(arm_id)),
            "source_group": rec["source_group"],
            "context_paradigm": rec["context_paradigm"],
            "input_alignment_format": fmt,
            "context_training_regime": reg,
            "primary_estimand": rec["primary_estimand"],
            "budget_tokens": rec["budget_tokens"],
            "aligned_msa": rec["aligned_msa"],
            "dtype_realised": rec["dtype_realised"],
            "source_path": rec["source_path"],
            "bands": prim["bands"],
            "pooled": prim["pooled"],
            "companion_full_target_masked": comp_fm,
        }

    # 4. Simultaneous family inference across all present arms (cluster-aggregated)
    simultaneous_primary: dict[str, Any] = {}
    if arms_present:
        for b_label, band_arg in (
            ("id_70_90", "id_70_90"),
            ("id_50_70", "id_50_70"),
            ("id_30_50", "id_30_50"),
            ("id_lt_30", "id_lt_30"),
            ("pooled_800", None),
        ):
            _, mat_auroc = _cluster_matrix(
                arms_present, loaded_arms, band=band_arg, metric_key="hom_beats_unrelated", null_shift=0.5
            )
            _, mat_shuf = _cluster_matrix(
                arms_present, loaded_arms, band=band_arg, metric_key="hom_beats_mono_shuffled", null_shift=0.5
            )
            _, mat_frac = _cluster_matrix(
                arms_present,
                loaded_arms,
                band=band_arg,
                metric_key="fractional_reduction_vs_unrelated",
                null_shift=0.0,
            )
            simultaneous_primary[b_label] = {
                "auroc_hom_vs_unrelated": _simultaneous_family(
                    arms_present, mat_auroc, draws=draws, seed=seed, null_offset=0.5
                ),
                "auroc_hom_vs_mono_shuffled": _simultaneous_family(
                    arms_present, mat_shuf, draws=draws, seed=seed + 1, null_offset=0.5
                ),
                "fractional_reduction_vs_unrelated": _simultaneous_family(
                    arms_present, mat_frac, draws=draws, seed=seed + 2, null_offset=0.0
                ),
            }

    # Companion simultaneous inference for full_target_masked across all 22 Group B arms
    simultaneous_group_b_full_mask: dict[str, Any] = {}
    if group_b_present:
        for b_label, band_arg in (
            ("id_70_90", "id_70_90"),
            ("id_50_70", "id_50_70"),
            ("id_30_50", "id_30_50"),
            ("id_lt_30", "id_lt_30"),
            ("pooled_800", None),
        ):
            _, mat_auroc = _cluster_matrix(
                group_b_present,
                loaded_arms,
                band=band_arg,
                metric_key="hom_beats_unrelated",
                null_shift=0.5,
                use_companion_full_mask=True,
            )
            _, mat_frac = _cluster_matrix(
                group_b_present,
                loaded_arms,
                band=band_arg,
                metric_key="fractional_reduction_vs_unrelated",
                null_shift=0.0,
                use_companion_full_mask=True,
            )
            simultaneous_group_b_full_mask[b_label] = {
                "auroc_hom_vs_unrelated": _simultaneous_family(
                    group_b_present, mat_auroc, draws=draws, seed=seed + 10, null_offset=0.5
                ),
                "fractional_reduction_vs_unrelated": _simultaneous_family(
                    group_b_present, mat_frac, draws=draws, seed=seed + 11, null_offset=0.0
                ),
            }

    # 5. Paradigm rollups
    paradigm_rollup: dict[str, Any] = {}
    for p in PARADIGM_ORDER:
        p_arms = [a for a in arms_present if ARM_METADATA[a].paradigm == p]
        if not p_arms:
            continue
        band_stats: dict[str, Any] = {}
        for b in (*IDENTITY_BANDS, "pooled"):
            aurocs = [
                per_arm_summary[a]["pooled"]["auroc_hom_vs_unrelated"]["mean"]
                if b == "pooled"
                else per_arm_summary[a]["bands"][b]["auroc_hom_vs_unrelated"]["mean"]
                for a in p_arms
            ]
            fracs = [
                per_arm_summary[a]["pooled"]["fractional_reduction_vs_unrelated"]["mean"]
                if b == "pooled"
                else per_arm_summary[a]["bands"][b]["fractional_reduction_vs_unrelated"]["mean"]
                for a in p_arms
            ]
            band_stats[b] = {
                "auroc_median": float(np.median(aurocs)),
                "auroc_min": float(np.min(aurocs)),
                "auroc_max": float(np.max(aurocs)),
                "frac_reduction_median": float(np.median(fracs)),
                "frac_reduction_min": float(np.min(fracs)),
                "frac_reduction_max": float(np.max(fracs)),
            }
        paradigm_rollup[p] = {
            "n_arms": len(p_arms),
            "arms": p_arms,
            "by_band": band_stats,
        }

    # 6. Scaling slopes per decade of parameters across all 13 multi-rung ladders
    scaling_ladders: dict[str, Any] = {}
    for lname, rung_tuple in R4_SCALING_LADDERS.items():
        rungs = [r for r in rung_tuple if r in loaded_arms]
        if len(rungs) < 2:
            continue
        log10_n = np.asarray([math.log10(ARM_METADATA[r].params_m * 1e6) for r in rungs], dtype=np.float64)
        slopes_by_band: dict[str, Any] = {}
        for b_label, band_arg in (
            ("id_70_90", "id_70_90"),
            ("id_50_70", "id_50_70"),
            ("id_30_50", "id_30_50"),
            ("id_lt_30", "id_lt_30"),
            ("pooled_800", None),
        ):
            _, mat_a = _cluster_matrix(
                rungs, loaded_arms, band=band_arg, metric_key="hom_beats_unrelated", null_shift=0.0
            )
            _, mat_f = _cluster_matrix(
                rungs, loaded_arms, band=band_arg, metric_key="fractional_reduction_vs_unrelated", null_shift=0.0
            )
            # cluster_bootstrap_scaling_slope expects shape (n_rungs, n_clusters) and returns -d(Y)/d(log10 N);
            # pass -mat.T so positive slope means GAIN per decade of parameters.
            slopes_by_band[b_label] = {
                "auroc_gain_per_decade": cluster_bootstrap_scaling_slope(
                    log10_n, -mat_a.T, n_draws=min(2000, draws), seed=seed
                ),
                "frac_reduction_gain_per_decade": cluster_bootstrap_scaling_slope(
                    log10_n, -mat_f.T, n_draws=min(2000, draws), seed=seed + 1
                ),
            }
        scaling_ladders[lname] = {
            "rungs": rungs,
            "params_m": [ARM_METADATA[r].params_m for r in rungs],
            "span_ratio": float(ARM_METADATA[rungs[-1]].params_m / ARM_METADATA[rungs[0]].params_m),
            "slopes_by_band": slopes_by_band,
        }

    companions = _collect_companions(repo_root)
    horizontal_table, regime_alignment_rollup = _build_horizontal_benchmark_table(
        arms_present, per_arm_summary, companions
    )

    cohort_path = repo_root / DEFAULT_COHORT_REL
    payload: dict[str, Any] = {
        "schema": SCHEMA_MASTER,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "completion_status": "complete" if not pending_arms else "interim_pending",
        "is_complete_65": len(pending_arms) == 0 and len(arms_present) == 65,
        "cohort": {
            "path": str(DEFAULT_COHORT_REL),
            "sha256": sha256_file(cohort_path) if cohort_path.is_file() else None,
            "n_retained_units": 800,
            "n_clusters": 710,
            "budget_tokens": 1024,
            "bands": list(IDENTITY_BANDS),
            "conditions": list(CONDITIONS),
        },
        "coverage": {
            "n_total_roster_arms": len(UNIFIED_65_ARMS),
            "n_arms_present": len(arms_present),
            "n_single_sequence_present": sum(1 for a in arms_present if a in UNIFIED_62_SINGLE_SEQUENCE_ARMS),
            "n_msa_present": sum(1 for a in arms_present if a in UNIFIED_3_MSA_CONTEXT_ARMS),
            "n_group_b_bidirectional_present": len(group_b_present),
            "n_pending_arms": len(pending_arms),
            "pending_arms": pending_arms,
        },
        "reproduction_gate": parity_gate,
        "exact_pll_diagnostics": exact_pll_diagnostics,
        "arms_order": arms_present,
        "horizontal_benchmark_table": horizontal_table,
        "by_training_regime_and_alignment": regime_alignment_rollup,
        "per_arm": per_arm_summary,
        "simultaneous_primary_65": simultaneous_primary,
        "simultaneous_group_b_full_target_masked": simultaneous_group_b_full_mask,
        "paradigm_rollup": paradigm_rollup,
        "scaling_ladders": scaling_ladders,
        "companion_controls": companions,
    }

    serialized = json.dumps(payload, sort_keys=True)
    if FORBIDDEN_INFRA_RE.search(serialized):
        raise ValueError("Refusing to write r4_scoring_unified_65.json containing forbidden pod/node identifier")

    digest = write_json_with_sha256(dest, payload)
    return {
        "output_path": str(dest),
        "sha256": digest,
        "payload": payload,
    }


__all__ = [
    "CACHED_EARLY_PROTEIN_ARMS",
    "CACHED_GP_SINGLE_ARMS",
    "CACHED_MSA_ARMS",
    "CONDITIONS",
    "DEFAULT_COHORT_REL",
    "DEFAULT_OUTPUT_REL",
    "DEFAULT_WORK_SUBDIR",
    "FORBIDDEN_INFRA_RE",
    "GROUP_A_CAUSAL_TEXT_ARMS",
    "GROUP_B_BIDIRECTIONAL_ARMS",
    "IDENTITY_BANDS",
    "NEW_SCORE_ARMS",
    "R4_SCALING_LADDERS",
    "SCHEMA_ARM_RUNNER",
    "SCHEMA_MASTER",
    "build_8fold_masks",
    "build_r4_scoring_unified_65",
    "build_unit_condition_inputs",
    "classify_r4_context_axes",
    "deterministic_shuffle",
    "pack_bidirectional_context",
    "pack_causal_text_context",
    "pick_unrelated_partners",
    "summarize_unit_rows",
]
