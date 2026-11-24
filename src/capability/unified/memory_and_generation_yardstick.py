"""Unified single-yardstick evaluators for family memory (62/46 arms) and R4/R5/R6.

Implements Workstream C of the single cross-paradigm yardstick unification:
  1. ``build_family_memory_unified_62`` -> ``family_memory_unified_62.json`` (+ ``.sha256``):
     * 62-model ``family_conditionals`` (64 evaluation families) with exact baseline
       parity checks, 10,000-draw simultaneous ``paired_bands`` (seed=20261106) in
       nats/res and bits/res across overall, conserved vs non-conserved, and shallow
       vs deep MSA strata, plus Q1-Q4 conservation quartiles.
     * 46-model ``family_distribution`` (64 evaluation families) with 10,000-draw
       ``paired_bands`` on column-null and composition-null contrasts, plus 9-ladder
       scaling regressions.
  2. ``build_r4_r5_r6_unified_summary`` -> ``r4_r5_r6_unified_summary.json`` (+ ``.sha256``):
     * Unified R4 (30 sequence-context arms on frozen 800-unit cohort + companion
       long/unaligned controls + ESM3-1.4B Route A 3D/SS8 comparator).
     * Unified R5 master ledger (17 EXP-R2-246 causal + ladder_original + 13 GP/ESM3
       arms + all 16 ESMFold2 structure evaluations, reporting DPLM uniform_random
       as primary diffusion entry and max_confidence as homopolymer-collapse diagnostic).
     * Unified R6 post-training alignment & distinct-5-mer filter-control summary.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.capability.core.amino_acids import AA20
from src.capability.extensions.bidirectional import (
    counts as band_verdict_counts,
    paired_bands,
    verdict as band_verdicts,
)
from src.capability.extensions.family_conditionals import _attach_baselines
from src.capability.extensions.family_distribution import _family_inference
from src.capability.memory_theory.capacity_scaling import (
    classify_conservation_tier,
    load_msa_support,
    shannon_entropy_bits,
)
from src.capability.memory_theory.io_utils import (
    LN2,
    SCALING_LADDERS,
    cluster_bootstrap_scaling_slope,
    load_json_file,
    write_json_with_sha256,
)
from src.capability.unified.roster import (
    ARM_METADATA,
    PREREGISTERED_33_ARMS,
    REPO_ROOT,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_65_ARMS,
    UNIFIED_OUTPUT_DIR,
    ResolvedArtifact,
    default_scan_dirs,
    dtype_rank,
    get_arm_metadata,
    resolve_gp_family_distribution_shards,
    resolve_gp_r1_json,
    resolve_gp_r4_json,
    resolve_gp_r5_json,
)

AA_INDEX = {aa: idx for idx, aa in enumerate(AA20)}
UNIFORM_AA20_NATS = math.log(len(AA20))
UNIFORM_AA20_BITS = math.log2(len(AA20))

PROTEIN_9_LADDERS: tuple[str, ...] = (
    "esm2",
    "esmc",
    "dplm",
    "proteinglm_mlm",
    "amplify",
    "saprot",
    "progen2",
    "progen3",
    "protgpt3",
)
FD_4_LADDERS: tuple[str, ...] = ("esm2", "progen3", "progen2", "protgpt3")


def _arm_header(arm_id: str) -> dict[str, Any]:
    meta = get_arm_metadata(arm_id)
    rec = asdict(meta)
    return rec


def _run_simultaneous_family(
    arm_ids: Sequence[str],
    matrix_nats: np.ndarray,
    *,
    draws: int = 10000,
    seed: int = 20261106,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Run both native ``paired_bands`` and studentized ``maxT`` (``_family_inference``) in nats/res and bits/res."""
    mat_nats = np.asarray(matrix_nats, dtype=np.float64)
    if mat_nats.ndim != 2 or mat_nats.shape[1] != len(arm_ids):
        raise ValueError(
            f"matrix shape {mat_nats.shape} does not match {len(arm_ids)} arms"
        )
    bands_nats = paired_bands(mat_nats, draws=draws, seed=seed, confidence=confidence)
    verdicts = band_verdicts(bands_nats["interval"])
    tally = band_verdict_counts(verdicts)

    # Canonical studentized maxT matching family_conditionals_20261107 & family_distribution_20261106
    stud = _family_inference(mat_nats, arm_ids, draws=draws, seed=seed)
    stud_primary = stud["primary"]
    stud_hyps = stud_primary["hypotheses"]
    stud_intervals = [h["simultaneous"] for h in stud_hyps]
    stud_verdicts = band_verdicts(stud_intervals)
    stud_tally = band_verdict_counts(stud_verdicts)
    pw_intervals = stud["auxiliary_pointwise_95"]["interval"]

    crit_nats = float(bands_nats["critical_value"])
    crit_bits = crit_nats / LN2

    per_arm: dict[str, Any] = {}
    for idx, arm_id in enumerate(arm_ids):
        pt_n = float(bands_nats["point"][idx])
        lo_n, hi_n = float(bands_nats["interval"][idx][0]), float(bands_nats["interval"][idx][1])
        slo_n, shi_n = float(stud_intervals[idx][0]), float(stud_intervals[idx][1])
        plo_n, phi_n = float(pw_intervals[idx][0]), float(pw_intervals[idx][1])
        col_n = mat_nats[:, idx]
        se_n = float(np.std(col_n, ddof=1) / math.sqrt(len(col_n)))
        per_arm[arm_id] = {
            "point_nats": pt_n,
            "simultaneous_ci95_nats": [lo_n, hi_n],
            "point_bits": pt_n / LN2,
            "simultaneous_ci95_bits": [lo_n / LN2, hi_n / LN2],
            "studentized_ci95_nats": [slo_n, shi_n],
            "studentized_ci95_bits": [slo_n / LN2, shi_n / LN2],
            "pointwise_ci95_nats": [plo_n, phi_n],
            "pointwise_ci95_bits": [plo_n / LN2, phi_n / LN2],
            "se_nats": se_n,
            "se_bits": se_n / LN2,
            "verdict": verdicts[idx],
            "studentized_verdict": stud_verdicts[idx],
            "studentized_status": stud_hyps[idx]["status"],
        }

    return {
        "method": bands_nats["method"],
        "studentized_maxt_method": stud_primary["method"],
        "draws": int(bands_nats["draws"]),
        "seed": int(bands_nats["seed"]),
        "confidence": float(bands_nats["confidence"]),
        "groups": int(bands_nats["groups"]),
        "family_size": int(bands_nats["family_size"]),
        "critical_value_nats": crit_nats,
        "critical_value_bits": crit_bits,
        "studentized_critical_value": float(stud_primary["critical_value"]),
        "verdict_counts": tally,
        "studentized_verdict_counts": stud_tally,
        "studentized_degenerate_arms": list(stud["degenerate_arms"]),
        "per_arm": per_arm,
    }


def _load_fc_baselines_and_sites(repo_root: Path) -> dict[str, Any]:
    """Load frozen 64-family evaluation manifest, exact Profile/Markov baselines, and 512 sites."""
    fc_root = repo_root / "results/extensions/family_conditionals_20261107"
    manifest = load_json_file(fc_root / "conditional-inputs.json")
    eval_units = sorted(
        (u for u in manifest["units"] if u["partition"] == "evaluation"),
        key=lambda u: u["unit_id"],
    )
    for unit in eval_units:
        _attach_baselines(unit)

    unit_order: list[str] = []
    unit_to_wt: dict[str, str] = {}
    profile_logp_by_unit: dict[str, float] = {}
    markov_logp_by_unit: dict[str, float] = {}
    mean_h_prof_by_unit: dict[str, float] = {}
    site_records: list[dict[str, Any]] = []

    for unit in eval_units:
        uid = str(unit["unit_id"])
        wt_id = str(unit["wildtype_id"])
        unit_order.append(uid)
        unit_to_wt[uid] = wt_id
        u_prof_sites: list[float] = []
        u_mkv_sites: list[float] = []
        u_h_sites: list[float] = []
        for site in unit["sites"]:
            pos = str(site["query_position"])
            freq = np.asarray(unit["baselines"]["profile_frequencies"][pos], dtype=np.float64)
            h_prof = shannon_entropy_bits(freq)
            prof_lp = np.asarray(site["baseline"]["profile"], dtype=np.float64)
            markov_lps = site["baseline"]["markov"]
            obs_indices = [AA_INDEX[ctx["observed"]] for ctx in site["contexts"]]
            prof_ctx = [float(prof_lp[idx]) for idx in obs_indices]
            mkv_ctx = [float(mlp[idx]) for mlp, idx in zip(markov_lps, obs_indices)]
            s_prof = float(np.mean(prof_ctx))
            s_mkv = float(np.mean(mkv_ctx))
            u_prof_sites.append(s_prof)
            u_mkv_sites.append(s_mkv)
            u_h_sites.append(h_prof)
            site_records.append(
                {
                    "unit_id": uid,
                    "wildtype_id": wt_id,
                    "query_position": pos,
                    "n_contexts": len(site["contexts"]),
                    "row_ids": [str(ctx["row_id"]) for ctx in site["contexts"]],
                    "obs_indices": obs_indices,
                    "h_prof_bits": h_prof,
                    "tier": classify_conservation_tier(h_prof),
                    "profile_logp_nats": s_prof,
                    "markov_logp_nats": s_mkv,
                }
            )
        profile_logp_by_unit[uid] = float(np.mean(u_prof_sites))
        markov_logp_by_unit[uid] = float(np.mean(u_mkv_sites))
        mean_h_prof_by_unit[uid] = float(np.mean(u_h_sites))

    return {
        "unit_order": unit_order,
        "unit_to_wt": unit_to_wt,
        "profile_logp_by_unit": profile_logp_by_unit,
        "markov_logp_by_unit": markov_logp_by_unit,
        "mean_h_prof_by_unit": mean_h_prof_by_unit,
        "site_records": site_records,
    }


def _compute_site_quartiles_summary(
    site_records: Sequence[dict[str, Any]],
    site_logp_by_arm: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Compute exact 512-site conservation quartiles (Q1 variable -> Q4 conserved) + H_prof tiers."""
    h_vals = np.asarray([s["h_prof_bits"] for s in site_records], dtype=np.float64)
    prof_lp = np.asarray([s["profile_logp_nats"] for s in site_records], dtype=np.float64)
    # Sort descending by H_prof so Q1 = most variable, Q4 = most conserved
    order_desc = np.argsort(-h_vals, kind="mergesort")
    n_sites = len(site_records)
    splits = np.array_split(order_desc, 4)
    quartile_masks = {
        "Q1_most_variable": splits[0],
        "Q2_variable_mid": splits[1],
        "Q3_conserved_mid": splits[2],
        "Q4_most_conserved": splits[3],
    }
    tier_masks = {
        "conserved_lt_1_0b": np.asarray([i for i, s in enumerate(site_records) if s["tier"] == "conserved"], dtype=int),
        "moderate_1_0_to_2_5b": np.asarray([i for i, s in enumerate(site_records) if s["tier"] == "moderate"], dtype=int),
        "variable_ge_2_5b": np.asarray([i for i, s in enumerate(site_records) if s["tier"] == "variable"], dtype=int),
    }

    per_arm: dict[str, Any] = {}
    for arm_id in sorted(site_logp_by_arm.keys()):
        arr = site_logp_by_arm[arm_id]
        diff = arr - prof_lp  # positive = beats profile (higher logp)
        q_out = {}
        for qk, idx in quartile_masks.items():
            q_out[qk] = {
                "n_sites": int(idx.size),
                "model_logp_nats": float(np.mean(arr[idx])),
                "model_loss_bits": float(-np.mean(arr[idx]) / LN2),
                "vs_profile_nats": float(np.mean(diff[idx])),
                "vs_profile_bits": float(np.mean(diff[idx]) / LN2),
            }
        t_out = {}
        for tk, idx in tier_masks.items():
            t_out[tk] = {
                "n_sites": int(idx.size),
                "model_logp_nats": float(np.mean(arr[idx])),
                "model_loss_bits": float(-np.mean(arr[idx]) / LN2),
                "vs_profile_nats": float(np.mean(diff[idx])),
                "vs_profile_bits": float(np.mean(diff[idx]) / LN2),
            }
        per_arm[arm_id] = {"quartiles": q_out, "h_prof_tiers": t_out}

    return {
        "n_arms_with_site_level_scores": len(per_arm),
        "n_sites": n_sites,
        "quartile_h_prof_ranges_bits": {
            qk: [float(np.min(h_vals[idx])), float(np.max(h_vals[idx]))]
            for qk, idx in quartile_masks.items()
        },
        "profile_baseline_by_quartile_bits": {
            qk: float(-np.mean(prof_lp[idx]) / LN2) for qk, idx in quartile_masks.items()
        },
        "arms": per_arm,
    }


def _compute_family_quartiles_summary(
    unit_order: Sequence[str],
    profile_logp_by_unit: Mapping[str, float],
    family_logp_by_arm: Mapping[str, Mapping[str, float]],
    arm_order: Sequence[str],
) -> dict[str, Any]:
    """Compute exact 64-family conservation quartiles Q1..Q4 across all 62 arms.

    Matches the repository convention (Q1 = lowest profile_logp = most variable families,
    Q4 = highest profile_logp = most conserved families; 16 families per quartile).
    """
    prof_vec = np.asarray([profile_logp_by_unit[u] for u in unit_order], dtype=np.float64)
    sorted_idx = np.argsort(prof_vec, kind="mergesort")
    splits = np.array_split(sorted_idx, 4)
    q_specs = (
        ("Q1_most_variable", splits[0]),
        ("Q2_variable_mid", splits[1]),
        ("Q3_conserved_mid", splits[2]),
        ("Q4_most_conserved", splits[3]),
    )

    per_arm: dict[str, Any] = {}
    for arm_id in arm_order:
        u_map = family_logp_by_arm[arm_id]
        vec = np.asarray([u_map[u] for u in unit_order], dtype=np.float64)
        diff = vec - prof_vec
        per_arm[arm_id] = {
            qk: {
                "n_families": int(idx.size),
                "model_logp_nats": float(np.mean(vec[idx])),
                "model_loss_bits": float(-np.mean(vec[idx]) / LN2),
                "vs_profile_nats": float(np.mean(diff[idx])),
                "vs_profile_bits": float(np.mean(diff[idx]) / LN2),
            }
            for qk, idx in q_specs
        }

    return {
        "n_arms": len(arm_order),
        "ordering_note": "64 evaluation families sorted ascending by profile_logp_nats: Q1=most variable (16 fams) -> Q4=most conserved (16 fams).",
        "profile_baseline_by_quartile": {
            qk: {
                "n_families": int(idx.size),
                "profile_logp_nats": float(np.mean(prof_vec[idx])),
                "profile_loss_bits": float(-np.mean(prof_vec[idx]) / LN2),
                "unit_ids": [unit_order[int(i)] for i in idx],
            }
            for qk, idx in q_specs
        },
        "arms": per_arm,
    }


def _extract_early_fd_arms(repo_root: Path, unit_order: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Load exact per-family native, column-null, and composition-null nats/res for PREREGISTERED_33_ARMS."""
    fd_root = repo_root / "results/extensions/family_distribution_20261106"
    summary = load_json_file(fd_root / "analysis/summary.json")
    eval_u = summary["units"]["evaluation"]
    unit_to_wt = {uid: next(iter(eval_u[uid].keys())) for uid in unit_order}

    arms_out: dict[str, dict[str, Any]] = {}
    for arm_id in PREREGISTERED_33_ARMS:
        spath = fd_root / "scores" / arm_id / "evaluation.json"
        doc = load_json_file(spath)
        units_map = doc["units"]
        native_nats: dict[str, float] = {}
        col_delta_nats: dict[str, float] = {}
        comp_delta_nats: dict[str, float] = {}
        for uid in unit_order:
            wt_id = unit_to_wt[uid]
            wt_entry = units_map[uid]["wildtypes"][wt_id]
            tot_res = float(np.sum(np.asarray(wt_entry["lengths"], dtype=np.float64)))
            nat_sum = float(np.sum(np.asarray(wt_entry["native_nll"], dtype=np.float64)))
            col_arr = np.asarray(wt_entry["column_null_nll"], dtype=np.float64)
            comp_arr = np.asarray(wt_entry["composition_null_nll"], dtype=np.float64)
            col_sum = float(np.sum(col_arr)) if col_arr.ndim == 1 else float(np.mean(np.sum(col_arr, axis=1)))
            comp_sum = float(np.sum(comp_arr)) if comp_arr.ndim == 1 else float(np.mean(np.sum(comp_arr, axis=1)))
            native_nats[uid] = nat_sum / tot_res
            col_delta_nats[uid] = (col_sum - nat_sum) / tot_res
            comp_delta_nats[uid] = (comp_sum - nat_sum) / tot_res
        arms_out[arm_id] = {
            "native_nats_per_res": native_nats,
            "column_null_minus_native_nats_per_res": col_delta_nats,
            "comp_null_minus_native_nats_per_res": comp_delta_nats,
            "likelihood_objective": "exact_causal_nll",
            "dtype_realised": str(doc.get("dtype", "torch.float32")),
            "source_paths": [str(spath.relative_to(repo_root))],
        }
    return arms_out


def _extract_gp_fd_arms(repo_root: Path, unit_order: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Resolve and merge all expanded_32 arms with complete 64-unit family_distribution scores."""
    expected = set(unit_order)
    gp_single = [a for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS if a not in set(PREREGISTERED_33_ARMS)]
    out: dict[str, dict[str, Any]] = {}

    for arm_id in gp_single:
        try:
            shards = resolve_gp_family_distribution_shards(arm_id, root=repo_root)
        except FileNotFoundError:
            continue
        native_nats: dict[str, float] = {}
        col_delta_nats: dict[str, float] = {}
        comp_delta_nats: dict[str, float] = {}
        dtypes: set[str] = set()
        score_sems: set[str] = set()
        src_paths: list[str] = []
        for art in shards:
            doc = load_json_file(art.path)
            if art.dtype_realised:
                dtypes.add(art.dtype_realised)
            if doc.get("score_semantics"):
                score_sems.add(str(doc["score_semantics"]))
            src_paths.append(str(art.path.relative_to(repo_root)))
            for u in doc.get("units", []):
                if u.get("partition") != "evaluation":
                    continue
                uid = str(u["unit_id"])
                if uid not in expected:
                    continue
                col = u["column_contrast"]
                comp = u["composition_contrast"]
                tot_len = float(col["total_length"])
                native_nats[uid] = float(col["native_nats"]) / tot_len
                col_delta_nats[uid] = float(col["nats_per_residue"])
                comp_delta_nats[uid] = float(comp["nats_per_residue"])
        if set(native_nats.keys()) != expected:
            continue
        meta = get_arm_metadata(arm_id)
        is_masked = meta.paradigm in ("masked_lm", "structure_aware_mlm", "discrete_diffusion") or any(
            "masked" in s.lower() or "pll" in s.lower() for s in score_sems
        )
        out[arm_id] = {
            "native_nats_per_res": native_nats,
            "column_null_minus_native_nats_per_res": col_delta_nats,
            "comp_null_minus_native_nats_per_res": comp_delta_nats,
            "likelihood_objective": "masked_pll" if is_masked else "exact_causal_nll",
            "score_semantics": sorted(score_sems),
            "dtype_realised": sorted(dtypes)[0] if len(dtypes) == 1 else ",".join(sorted(dtypes)),
            "n_shards": len(shards),
            "source_paths": src_paths,
        }
    return out


def build_family_memory_unified_62(
    repo_root: Path = REPO_ROOT,
    *,
    draws: int = 10000,
    seed: int = 20261106,
    scaling_bootstrap_draws: int = 2000,
) -> dict[str, Any]:
    """Build unified 62-model family_conditionals + 46-model family_distribution report."""
    repo_root = Path(repo_root)
    fc_base = _load_fc_baselines_and_sites(repo_root)
    unit_order = fc_base["unit_order"]
    unit_to_wt = fc_base["unit_to_wt"]
    prof_by_unit = fc_base["profile_logp_by_unit"]
    mkv_by_unit = fc_base["markov_logp_by_unit"]
    site_records = fc_base["site_records"]

    msa_support = load_msa_support(
        repo_root / "results/extensions/statistical_ladder_20261106/inputs/support.json"
    )

    # ------------------------------------------------------------------
    # 1. Load all 62 arms for family_conditionals + assert baseline parity
    # ------------------------------------------------------------------
    fc_root = repo_root / "results/extensions/family_conditionals_20261107"
    family_logp_by_arm: dict[str, dict[str, float]] = {}
    site_logp_by_arm: dict[str, np.ndarray] = {}
    arm_fc_meta: dict[str, dict[str, Any]] = {}
    max_profile_diff = 0.0
    max_markov_diff = 0.0

    # 1a. 33 pre-registered arms
    for arm_id in PREREGISTERED_33_ARMS:
        spath = fc_root / "scores" / arm_id / "evaluation.json"
        doc = load_json_file(spath)
        units_map = doc["units"]
        site_vals = np.empty(len(site_records), dtype=np.float64)
        fam_buckets: dict[str, list[float]] = defaultdict(list)
        for s_idx, srec in enumerate(site_records):
            uid = srec["unit_id"]
            pos = srec["query_position"]
            lps = units_map[uid]["sites"][pos]["logprob"]
            obs_idx = srec["obs_indices"]
            s_val = float(np.mean([float(lp[idx]) for lp, idx in zip(lps, obs_idx)]))
            site_vals[s_idx] = s_val
            fam_buckets[uid].append(s_val)
        site_logp_by_arm[arm_id] = site_vals
        family_logp_by_arm[arm_id] = {uid: float(np.mean(fam_buckets[uid])) for uid in unit_order}
        arm_fc_meta[arm_id] = {
            **_arm_header(arm_id),
            "conditional_estimator": "causal_20string",
            "dtype_realised": str(doc.get("dtype", "torch.float32")),
            "source_path": str(spath.relative_to(repo_root)),
        }

    # 1b. 28 GP arms + esm3-1.4b via canonical roster resolver
    ctx_to_site: dict[tuple[str, str, str], int] = {}
    for s_idx, srec in enumerate(site_records):
        for rid in srec["row_ids"]:
            ctx_to_site[(srec["unit_id"], str(srec["query_position"]), str(rid))] = s_idx

    gp_single = [a for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS if a not in set(PREREGISTERED_33_ARMS)]
    for arm_id in gp_single:
        art = resolve_gp_r1_json(arm_id, root=repo_root)
        if arm_id == "esm3-1.4b":
            per_site: list[list[float]] = [[] for _ in range(len(site_records))]
            with art.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("partition") != "evaluation":
                        continue
                    key = (str(row["unit_id"]), str(row["query_position"]), str(row["row_id"]))
                    s_idx = ctx_to_site.get(key)
                    if s_idx is not None:
                        per_site[s_idx].append(float(row["observed_logp_nats"]))
            if not all(len(vs) == srec["n_contexts"] for vs, srec in zip(per_site, site_records)):
                raise RuntimeError("esm3-1.4b: incomplete evaluation contexts in r1_family_conditionals.jsonl")
            site_vals = np.asarray([float(np.mean(vs)) for vs in per_site], dtype=np.float64)
            site_logp_by_arm[arm_id] = site_vals
            fam_buckets = defaultdict(list)
            for srec, sval in zip(site_records, site_vals):
                fam_buckets[srec["unit_id"]].append(float(sval))
            family_logp_by_arm[arm_id] = {uid: float(np.mean(fam_buckets[uid])) for uid in unit_order}
            arm_fc_meta[arm_id] = {
                **_arm_header(arm_id),
                "conditional_estimator": "masked_1pass",
                "conditional_semantics": "esm3_seq_only_1pass_masked_conditional_over_aa20",
                "dtype_realised": art.dtype_realised or "torch.bfloat16",
                "source_path": str(art.path.relative_to(repo_root)),
            }
        else:
            doc = load_json_file(art.path)
            fc_sec = doc["family_conditionals"]
            eval_map: dict[str, float] = {}
            for u in fc_sec["units"]:
                if u.get("partition") != "evaluation":
                    continue
                uid = str(u["unit_id"])
                p_diff = abs(float(u["profile_logp"]) - prof_by_unit[uid])
                m_diff = abs(float(u["markov_logp"]) - mkv_by_unit[uid])
                max_profile_diff = max(max_profile_diff, p_diff)
                max_markov_diff = max(max_markov_diff, m_diff)
                if p_diff >= 1e-6 or m_diff >= 1e-6:
                    raise ValueError(
                        f"{arm_id}/{uid}: baseline mismatch profile_diff={p_diff:.2e}, markov_diff={m_diff:.2e}"
                    )
                eval_map[uid] = float(u["model_logp"])
            if set(eval_map.keys()) != set(unit_order):
                raise RuntimeError(f"{arm_id}: expected 64 eval units, got {len(eval_map)}")
            family_logp_by_arm[arm_id] = eval_map
            sem = str(fc_sec.get("conditional_semantics", ""))
            estimator = "masked_1pass" if "masked" in sem.lower() else "causal_20string"
            arm_fc_meta[arm_id] = {
                **_arm_header(arm_id),
                "conditional_estimator": estimator,
                "conditional_semantics": sem,
                "dtype_realised": art.dtype_realised or str(doc.get("provenance", {}).get("dtype_realised", "")),
                "source_path": str(art.path.relative_to(repo_root)),
            }

    assert set(family_logp_by_arm.keys()) == set(UNIFIED_62_SINGLE_SEQUENCE_ARMS)
    arms_62 = list(UNIFIED_62_SINGLE_SEQUENCE_ARMS)

    # Build (64, 62) matrices in nats/res
    prof_vec = np.asarray([prof_by_unit[u] for u in unit_order], dtype=np.float64)
    mkv_vec = np.asarray([mkv_by_unit[u] for u in unit_order], dtype=np.float64)
    prof_ce_bits_vec = -prof_vec / LN2
    depths_vec = np.asarray(
        [msa_support[unit_to_wt[u]]["effective_depth_per_site"] for u in unit_order],
        dtype=np.float64,
    )

    mat_model_logp = np.column_stack(
        [np.asarray([family_logp_by_arm[a][u] for u in unit_order], dtype=np.float64) for a in arms_62]
    )
    mat_vs_prof = mat_model_logp - prof_vec[:, None]  # positive = beats Profile
    mat_vs_mkv = mat_model_logp - mkv_vec[:, None]    # positive = beats Markov

    # Strata index arrays (strictly defined over the 64 evaluation families)
    strata_indices: dict[str, np.ndarray] = {
        "overall_64": np.arange(len(unit_order), dtype=int),
        "conserved_profile_lt_1_8_bits": np.flatnonzero(prof_ce_bits_vec < 1.8),
        "non_conserved_profile_ge_1_8_bits": np.flatnonzero(prof_ce_bits_vec >= 1.8),
        "shallow_neff_lt_5": np.flatnonzero(depths_vec < 5.0),
        "deep_neff_ge_5": np.flatnonzero(depths_vec >= 5.0),
    }

    fc_bands_by_stratum: dict[str, Any] = {}
    for sname, sidx in strata_indices.items():
        fc_bands_by_stratum[sname] = {
            "n_families": int(sidx.size),
            "unit_ids": [unit_order[int(i)] for i in sidx],
            "profile_baseline_nats": float(np.mean(prof_vec[sidx])),
            "profile_baseline_loss_bits": float(np.mean(prof_ce_bits_vec[sidx])),
            "markov_baseline_nats": float(np.mean(mkv_vec[sidx])),
            "markov_baseline_loss_bits": float(-np.mean(mkv_vec[sidx]) / LN2),
            "model_minus_profile": _run_simultaneous_family(
                arms_62, mat_vs_prof[sidx, :], draws=draws, seed=seed
            ),
            "model_minus_markov": _run_simultaneous_family(
                arms_62, mat_vs_mkv[sidx, :], draws=draws, seed=seed
            ),
            "model_absolute_logp": _run_simultaneous_family(
                arms_62, mat_model_logp[sidx, :], draws=draws, seed=seed
            ),
        }

    # Family-level Q1-Q4 (all 62 arms) & site-level Q1-Q4 (34 arms)
    family_quartiles = _compute_family_quartiles_summary(
        unit_order, prof_by_unit, family_logp_by_arm, arms_62
    )
    site_quartiles = _compute_site_quartiles_summary(site_records, site_logp_by_arm)

    # 9-ladder FC scaling regressions (-dH / d log10 N in bits/res per decade)
    fc_loss_bits_by_arm = {
        a: np.asarray([-family_logp_by_arm[a][u] / LN2 for u in unit_order], dtype=np.float64)
        for a in arms_62
    }
    fc_9_ladders: dict[str, Any] = {}
    for lname in PROTEIN_9_LADDERS:
        rungs = [r for r in SCALING_LADDERS[lname] if r in fc_loss_bits_by_arm]
        if len(rungs) < 2:
            continue
        log10_n = np.asarray([math.log10(ARM_METADATA[r].params_m * 1e6) for r in rungs], dtype=np.float64)
        by_stratum: dict[str, Any] = {}
        for sname, sidx in strata_indices.items():
            mat_sub = np.vstack([fc_loss_bits_by_arm[r][sidx] for r in rungs])
            by_stratum[sname] = cluster_bootstrap_scaling_slope(
                log10_n, mat_sub, n_draws=scaling_bootstrap_draws, seed=seed
            )
        fc_9_ladders[lname] = {
            "rungs": rungs,
            "params_m": [ARM_METADATA[r].params_m for r in rungs],
            "span_ratio": float(ARM_METADATA[rungs[-1]].params_m / ARM_METADATA[rungs[0]].params_m),
            "conditional_estimators": sorted({arm_fc_meta[r]["conditional_estimator"] for r in rungs}),
            "by_stratum": by_stratum,
        }

    # ------------------------------------------------------------------
    # 2. Load all 46 arms for family_distribution + 10k paired_bands + 4 FD ladders
    # ------------------------------------------------------------------
    fd_early = _extract_early_fd_arms(repo_root, unit_order)
    fd_gp = _extract_gp_fd_arms(repo_root, unit_order)
    fd_all = {**fd_early, **fd_gp}
    arms_46 = [a for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS if a in fd_all]
    assert len(arms_46) == 46, f"Expected 46 family_distribution arms, got {len(arms_46)}"

    mat_fd_native = np.column_stack(
        [np.asarray([fd_all[a]["native_nats_per_res"][u] for u in unit_order], dtype=np.float64) for a in arms_46]
    )
    mat_fd_col = np.column_stack(
        [
            np.asarray([fd_all[a]["column_null_minus_native_nats_per_res"][u] for u in unit_order], dtype=np.float64)
            for a in arms_46
        ]
    )
    mat_fd_comp = np.column_stack(
        [
            np.asarray([fd_all[a]["comp_null_minus_native_nats_per_res"][u] for u in unit_order], dtype=np.float64)
            for a in arms_46
        ]
    )

    fd_arm_meta = {
        a: {
            **_arm_header(a),
            "likelihood_objective": fd_all[a]["likelihood_objective"],
            "dtype_realised": fd_all[a]["dtype_realised"],
            "n_shards": fd_all[a].get("n_shards", 1),
            "source_paths": fd_all[a]["source_paths"],
        }
        for a in arms_46
    }

    fd_bands = {
        "column_null_minus_native": _run_simultaneous_family(
            arms_46, mat_fd_col, draws=draws, seed=seed
        ),
        "comp_null_minus_native": _run_simultaneous_family(
            arms_46, mat_fd_comp, draws=draws, seed=seed
        ),
        "native_nll_per_res": _run_simultaneous_family(
            arms_46, mat_fd_native, draws=draws, seed=seed
        ),
    }

    fd_ladders: dict[str, Any] = {}
    for lname in FD_4_LADDERS:
        rungs = [r for r in SCALING_LADDERS[lname] if r in fd_all]
        if len(rungs) < 2:
            continue
        log10_n = np.asarray([math.log10(ARM_METADATA[r].params_m * 1e6) for r in rungs], dtype=np.float64)
        nat_by_s: dict[str, Any] = {}
        col_by_s: dict[str, Any] = {}
        for sname in ("overall_64", "shallow_neff_lt_5", "deep_neff_ge_5"):
            sidx = strata_indices[sname]
            mat_nat_b = np.vstack(
                [
                    np.asarray([fd_all[r]["native_nats_per_res"][unit_order[int(i)]] / LN2 for i in sidx])
                    for r in rungs
                ]
            )
            # Negate column-null delta so slope is +d(Delta_col)/d(log10 N)
            mat_col_b = np.vstack(
                [
                    np.asarray(
                        [-fd_all[r]["column_null_minus_native_nats_per_res"][unit_order[int(i)]] / LN2 for i in sidx]
                    )
                    for r in rungs
                ]
            )
            nat_by_s[sname] = cluster_bootstrap_scaling_slope(
                log10_n, mat_nat_b, n_draws=scaling_bootstrap_draws, seed=seed
            )
            col_by_s[sname] = cluster_bootstrap_scaling_slope(
                log10_n, mat_col_b, n_draws=scaling_bootstrap_draws, seed=seed
            )
        fd_ladders[lname] = {
            "rungs": rungs,
            "params_m": [ARM_METADATA[r].params_m for r in rungs],
            "span_ratio": float(ARM_METADATA[rungs[-1]].params_m / ARM_METADATA[rungs[0]].params_m),
            "likelihood_objectives": sorted({fd_all[r]["likelihood_objective"] for r in rungs}),
            "native_bits_per_res_by_stratum": nat_by_s,
            "column_null_delta_gain_per_decade_by_stratum": col_by_s,
        }

    return {
        "schema": "unified_family_memory_62_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap": {
            "simultaneous_methods": {
                "native_paired_bands": "paired_bands (native max-abs centered bootstrap deviation)",
                "studentized_maxt": "shared family multinomial bootstrap; maximum absolute centered statistic/fixed SE (_family_inference, canonical primary estimator for heterogeneous-variance cross-paradigm families)",
            },
            "draws": int(draws),
            "seed": int(seed),
            "confidence": 0.95,
        },
        "family_conditionals_62": {
            "n_arms": len(arms_62),
            "n_evaluation_families": len(unit_order),
            "arms_order": arms_62,
            "uniform_aa20_nats": UNIFORM_AA20_NATS,
            "uniform_aa20_bits": UNIFORM_AA20_BITS,
            "baseline_parity_verification": {
                "max_abs_profile_logp_diff_nats": max_profile_diff,
                "max_abs_markov_logp_diff_nats": max_markov_diff,
                "tolerance_nats": 1e-6,
                "passed": bool(max_profile_diff < 1e-6 and max_markov_diff < 1e-6),
            },
            "sign_convention": "model_minus_profile > 0 means model achieves HIGHER conditional log-probability (LOWER cross-entropy) than training-only Profile.",
            "arm_metadata": arm_fc_meta,
            "simultaneous_bands_by_stratum": fc_bands_by_stratum,
            "family_conservation_quartiles_62": family_quartiles,
            "site_conservation_quartiles_34": site_quartiles,
            "scaling_ladders_9": fc_9_ladders,
        },
        "family_distribution_46": {
            "n_arms": len(arms_46),
            "n_evaluation_families": len(unit_order),
            "arms_order": arms_46,
            "sign_convention": "column_null_minus_native > 0 and comp_null_minus_native > 0 mean model assigns LOWER NLL (higher likelihood) to native sequences than to column- or composition-shuffled nulls.",
            "arm_metadata": fd_arm_meta,
            "simultaneous_bands_overall_64": fd_bands,
            "scaling_ladders_4": fd_ladders,
        },
    }


# ============================================================================
# Part 2: Unified R4 (30 + 1), R5 Master Ledger, and R6 Summary
# ============================================================================


def _resolve_gp_r4_any(
    arm_id: str,
    *,
    budget_tokens: int = 1024,
    aligned_msa: bool = True,
    repo_root: Path = REPO_ROOT,
) -> ResolvedArtifact | None:
    """Resolve GP R4 JSON via roster resolver first, falling back to ``_b<budget>`` MSA filenames."""
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
            cand_dtype = doc.get("provenance", {}).get("dtype_realised")
            if best is None or dtype_rank(cand_dtype) >= dtype_rank(best.dtype_realised):
                best = ResolvedArtifact(
                    arm_id=arm_id,
                    stage=f"r4_b{budget_tokens}" + ("" if aligned_msa else "_unaligned"),
                    path=path,
                    dtype_realised=str(cand_dtype) if cand_dtype else None,
                    sibling_json=path,
                )
    return best


def _standardize_early_r4_arm(
    arm_id: str,
    rec: Mapping[str, Any],
    source_rel: str,
) -> dict[str, Any]:
    """Normalize one early EXP-R2-228 arm record into the unified R4 schema."""
    modality = str(rec.get("modality", "protein"))
    bands_raw = rec.get("bands", {})
    bands_out: dict[str, Any] = {}
    for band_name, b_entry in sorted(bands_raw.items()):
        # Early schema nests under 'retained' (and sometimes 'high_local_overlap')
        ret = b_entry.get("retained") if isinstance(b_entry, dict) and "retained" in b_entry else b_entry
        if not isinstance(ret, dict):
            continue
        auroc = ret.get("auroc") or {}
        frac = ret.get("fractional_reduction") or {}
        bands_out[band_name] = {
            "auroc_hom_vs_unrelated": {
                "mean": auroc.get("mean"),
                "ci95": auroc.get("ci95"),
                "n_groups": auroc.get("n_groups"),
                "n_units": auroc.get("n_units") or auroc.get("n_rows"),
            },
            "auroc_hom_vs_mono_shuffled": None,
            "fractional_reduction": {
                "mean": frac.get("mean"),
                "ci95": frac.get("ci95"),
                "n_groups": frac.get("n_groups"),
            }
            if frac
            else None,
        }
    verdict = rec.get("verdict") or {}
    return {
        **_arm_header(arm_id),
        "r4_modality": modality,
        "context_paradigm": "single_chain_causal",
        "budget_tokens": 1024,
        "aligned_msa": None,
        "verdict_outcome": verdict.get("outcome"),
        "bands": bands_out,
        "source_path": source_rel,
    }


def _standardize_gp_r4_arm(
    art: ResolvedArtifact,
    repo_root: Path,
    *,
    context_paradigm: str,
) -> dict[str, Any]:
    doc = load_json_file(art.path)
    bands_out: dict[str, Any] = {}
    for band_name, b_entry in sorted((doc.get("bands") or {}).items()):
        a_unrel = b_entry.get("auroc_hom_vs_unrelated") or {}
        a_shuf = b_entry.get("auroc_hom_vs_mono_shuffled") or {}
        frac = b_entry.get("fractional_reduction_vs_unrelated") or b_entry.get("fractional_reduction") or {}
        bands_out[band_name] = {
            "auroc_hom_vs_unrelated": {
                "mean": a_unrel.get("mean"),
                "ci95": [a_unrel.get("ci_low"), a_unrel.get("ci_high")] if a_unrel else None,
                "n_groups": a_unrel.get("n_clusters"),
                "n_units": a_unrel.get("n_items"),
            },
            "auroc_hom_vs_mono_shuffled": {
                "mean": a_shuf.get("mean"),
                "ci95": [a_shuf.get("ci_low"), a_shuf.get("ci_high")] if a_shuf else None,
                "n_groups": a_shuf.get("n_clusters"),
                "n_units": a_shuf.get("n_items"),
            }
            if a_shuf
            else None,
            "fractional_reduction": {
                "mean": frac.get("mean"),
                "ci95": [frac.get("ci_low"), frac.get("ci_high")] if frac else None,
                "n_groups": frac.get("n_clusters"),
            }
            if frac
            else None,
        }
    return {
        **_arm_header(art.arm_id),
        "r4_modality": "protein",
        "context_paradigm": context_paradigm,
        "budget_tokens": int(doc.get("budget_tokens", 1024)),
        "aligned_msa": bool(doc.get("aligned_msa", True)),
        "dtype_realised": art.dtype_realised,
        "verdict_outcome": None,
        "bands": bands_out,
        "source_path": str(art.path.relative_to(repo_root)),
    }


def _build_unified_r4(repo_root: Path) -> dict[str, Any]:
    """Assemble 30 sequence-context arms on identical 800-unit cohort + companion controls + ESM3 Route A."""
    early_sources = [
        repo_root / "results/R5/context_homologue_20260826/context_homologue.json",
        repo_root / "results/R5/context_homologue_expansion_20260919/context_homologue.json",
        repo_root / "results/R5/context_homologue_progen3_20260924/context_homologue.json",
    ]
    primary_rows: dict[str, dict[str, Any]] = {}
    for src in early_sources:
        if not src.is_file():
            continue
        doc = load_json_file(src)
        rel = str(src.relative_to(repo_root))
        for arm_id, rec in (doc.get("arms") or {}).items():
            if arm_id in ARM_METADATA:
                primary_rows[arm_id] = _standardize_early_r4_arm(arm_id, rec, rel)

    # 10 GP R4 arms at matched budget_tokens=1024
    gp_single_r4 = [
        "dayhoff-170m",
        "progen3-219m",
        "progen3-339m",
        "progen3-762m",
        "progen3-1b",
        "protgpt3-112m",
        "protgpt3-10b",
    ]
    gp_msa_r4 = ["msa-transformer-100m", "protgpt3-msa", "dayhoff-3b-msa"]

    for arm_id in gp_single_r4:
        art = _resolve_gp_r4_any(arm_id, budget_tokens=1024, aligned_msa=True, repo_root=repo_root)
        if art is None:
            raise FileNotFoundError(f"Missing GP R4 b1024 artifact for {arm_id}")
        primary_rows[arm_id] = _standardize_gp_r4_arm(art, repo_root, context_paradigm="single_chain_causal")

    for arm_id in gp_msa_r4:
        art = _resolve_gp_r4_any(arm_id, budget_tokens=1024, aligned_msa=True, repo_root=repo_root)
        if art is None:
            raise FileNotFoundError(f"Missing GP MSA R4 b1024 artifact for {arm_id}")
        primary_rows[arm_id] = _standardize_gp_r4_arm(art, repo_root, context_paradigm="msa_aligned")

    # Companion long-context and unaligned ablation rows
    companion_controls: dict[str, dict[str, Any]] = {}
    for key, arm_id, budget, aligned, paradigm in (
        ("dayhoff-3b-msa|b4096", "dayhoff-3b-msa", 4096, True, "msa_aligned"),
        ("protgpt3-msa|b8192", "protgpt3-msa", 8192, True, "msa_aligned"),
        ("msa-transformer-100m|unaligned_control", "msa-transformer-100m", 1024, False, "msa_unaligned_control"),
    ):
        art = _resolve_gp_r4_any(arm_id, budget_tokens=budget, aligned_msa=aligned, repo_root=repo_root)
        if art is not None:
            companion_controls[key] = _standardize_gp_r4_arm(art, repo_root, context_paradigm=paradigm)

    # ESM3-1.4B Route A ground-truth 3D/SS8 comparator
    esm3_art = resolve_gp_r4_json("esm3-1.4b", root=repo_root)
    esm3_doc = load_json_file(esm3_art.path)
    esm3_bands: dict[str, Any] = {}
    for band, brec in sorted((esm3_doc.get("summary_by_band") or {}).items()):
        esm3_bands[band] = {
            "n_units": brec.get("n_units"),
            "n_groups": brec.get("n_groups"),
            "mean_nll_nats_loo_pll": brec.get("mean_nll_nats"),
            "full_masked_mean_nll_nats": brec.get("full_masked_mean_nll_nats"),
            "full_masked_fractional_reduction": brec.get("full_masked_fractional_reduction"),
            "full_masked_paired_auroc": brec.get("full_masked_paired_auroc"),
            "loo_pll_paired_auroc": brec.get("paired_auroc"),
        }

    esm3_comparator = {
        **_arm_header("esm3-1.4b"),
        "context_paradigm": "multimodal_3d_ss8",
        "route": esm3_doc.get("route", "ground_truth_3d"),
        "n_units_evaluated": esm3_doc.get("n_units_evaluated"),
        "note": "Route A evaluates ESM3-1.4B on ground-truth AFDBxCATH 3D/SS8 templates; full-masked NLL eliminates self-sequence leakage.",
        "bands": esm3_bands,
        "source_path": str(esm3_art.path.relative_to(repo_root)),
    }

    ordered_ids = [a for a in UNIFIED_65_ARMS if a in primary_rows]
    assert len(ordered_ids) == 30, f"Expected 30 primary R4 sequence-context arms, got {len(ordered_ids)}"

    return {
        "n_primary_sequence_context_arms": len(ordered_ids),
        "primary_cohort": "results/R5/context_homologue_20260826/cohort.json",
        "arms_order": ordered_ids,
        "primary_sequence_context_arms": {a: primary_rows[a] for a in ordered_ids},
        "companion_long_and_unaligned_controls": companion_controls,
        "multimodal_3d_ss8_comparator": esm3_comparator,
    }


def _extract_r5_exp_r2_246(repo_root: Path) -> dict[str, Any]:
    """Extract all 17 EXP-R2-246 causal unconditional generation arms + UniRef50 novelty."""
    base_dir = repo_root / "results/R6/generation_evidence_20260905/analysis/esmfold2_20260920012718"
    nov_path = repo_root / "results/R6/generation_novelty_20261008/generated_novelty.jsonl"

    novelty_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if nov_path.is_file():
        with nov_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("role") != "generated":
                    continue
                arm = str(row["arm"])
                st = str(row.get("identity_stratum", "unknown"))
                novelty_counts[arm]["total"] += 1
                novelty_counts[arm][st] += 1

    entries: dict[str, Any] = {}
    for arm_dir in sorted(base_dir.iterdir()):
        if not arm_dir.is_dir():
            continue
        arm_id = arm_dir.name
        gba_path = arm_dir / "generation_biology_analysis.json"
        if not gba_path.is_file() or arm_id not in ARM_METADATA:
            continue
        gba = load_json_file(gba_path)
        arm_delta = (gba.get("arms") or {}).get(arm_id) or {}
        sc_list = [c for c in (gba.get("structural_cells") or []) if c.get("arm") == arm_id]
        pl_list = [p for p in (gba.get("profile_ledger") or []) if p.get("arm") == arm_id]
        sc = sc_list[0] if sc_list else {}
        pl = pl_list[0] if pl_list else {}
        nov = novelty_counts.get(arm_id, {})
        tot_nov = nov.get("total", 0)
        entries[arm_id] = {
            **_arm_header(arm_id),
            "generation_regime": "free_length_causal",
            "campaign": "EXP-R2-246",
            "n_attempts": pl.get("n_attempts"),
            "n_supported": pl.get("n_supported"),
            "any_pfam_profile_rate": pl.get("any_profile_rate"),
            "n_distinct_pfam_families": pl.get("n_distinct_families"),
            "paired_mean_ca_plddt_delta_vs_comp_ref": arm_delta.get("mean"),
            "paired_mean_ca_plddt_delta_ci95": arm_delta.get("ci95"),
            "paired_fraction_plddt_ge70_delta_vs_comp_ref": sc.get("paired_fraction_delta"),
            "mean_ca_plddt_generated": sc.get("generated_mean_ca_plddt") or sc.get("mean_ca_plddt"),
            "fraction_plddt_ge70_generated": sc.get("generated_fraction_ge70") or sc.get("fraction_plddt_ge70"),
            "uniref50_lt30_fraction": (
                float(nov.get("lt30_no_detectable_homology", 0) / tot_nov) if tot_nov > 0 else None
            ),
            "uniref50_n_searched": tot_nov if tot_nov > 0 else None,
            "source_path": str(gba_path.relative_to(repo_root)),
        }
    return entries


def _extract_r5_ladder_original(repo_root: Path) -> dict[str, Any]:
    """Summarize the original 16-backbone modification-extent ladder from ladder_divergence.json."""
    ld_path = repo_root / "results/R6/ladder_scoped_20261011/primary/ladder_divergence.json"
    ld = load_json_file(ld_path)
    cells = ld.get("cells") or {}

    by_arm: dict[str, dict[str, Any]] = {}
    for cell_key, cell in sorted(cells.items()):
        product_arm = str(cell.get("product_arm", ""))
        scoring_arm = str(cell.get("scoring_arm", ""))
        condition = str(cell.get("condition", ""))
        # Retain each arm's self-scored native condition (plus one composition-shuffle reference)
        if product_arm == "composition-shuffle":
            if scoring_arm != "dplm-650m":
                continue
        elif product_arm != scoring_arm:
            continue

        rung = str(cell.get("rung", ""))
        dists = cell.get("distributions") or {}
        corrs = cell.get("correlations") or {}
        if product_arm not in by_arm:
            regime = (
                "composition_shuffle_null"
                if product_arm == "composition-shuffle"
                else (
                    "bidirectional_infill"
                    if condition == "bidirectional_infill"
                    else "free_length_causal"
                )
            )
            by_arm[product_arm] = {
                "arm_id": product_arm,
                "in_unified_65": product_arm in ARM_METADATA,
                "generation_regime": regime,
                "condition": condition,
                "rungs": {},
            }
        by_arm[product_arm]["rungs"][rung] = {
            "n_observations": cell.get("n_observations"),
            "n_backbones": cell.get("n_backbones"),
            "mean_ca_plddt": (dists.get("mean_ca_plddt") or {}).get("mean"),
            "mean_tm_score_to_parent": (dists.get("tm_score_to_parent") or {}).get("mean"),
            "mean_window_lddt_to_parent": (dists.get("window_lddt_to_parent") or {}).get("mean"),
            "spearman_tm_raw": ((corrs.get("tm_score_to_parent") or {}).get("raw") or {}).get("point"),
            "spearman_plddt_raw": ((corrs.get("mean_ca_plddt") or {}).get("raw") or {}).get("point"),
        }

    return {
        "n_entries": len(by_arm),
        "source_path": str(ld_path.relative_to(repo_root)),
        "arms": by_arm,
    }


def _build_unified_r5_and_r6(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build unified R5 generation ledger (early + 13 GP/ESM3 + 16 ESMFold2 evals) and R6 summary."""
    exp246 = _extract_r5_exp_r2_246(repo_root)
    ladder_orig = _extract_r5_ladder_original(repo_root)

    # Index all 16 ESMFold2 R5 structure evaluations from general_plm_summary.json
    gs_path = repo_root / "results/extensions/general_plm_20261120/general_plm_summary.json"
    gs = load_json_file(gs_path)
    comp = gs.get("companion_artifacts", {})

    struct_evals_16: dict[str, Any] = {}
    for key, cmeta in sorted(comp.items()):
        if not key.startswith("r5_structure_eval::"):
            continue
        label = key.split("::", 1)[1]
        src = Path(cmeta["source"])
        doc = load_json_file(src)
        # Extract compact per-rung k40 & full_length_matched + denovo + r5c_conditioned summaries
        ladder_compact = {}
        for lk, lv in sorted((doc.get("r5a_ladder") or {}).items()):
            rung = lk.split("|")[-1]
            ladder_compact[rung] = {
                "condition": lv.get("condition"),
                "mean_plddt": (lv.get("mean_plddt") or {}).get("mean"),
                "mean_ptm": (lv.get("mean_ptm") or {}).get("mean"),
                "mean_tm_score_to_parent": (lv.get("mean_tm_score") or lv.get("mean_tm_score_to_parent") or {}).get("mean"),
                "mean_scrmsd_angstrom": (lv.get("mean_scrmsd_angstrom") or {}).get("mean"),
                "n_items": (lv.get("mean_plddt") or {}).get("n_items"),
            }
        denovo_raw = doc.get("r5b_denovo") or {}
        denovo_compact = next(iter(denovo_raw.values())) if denovo_raw else None
        r5c_raw = doc.get("r5c_conditioned") or {}
        struct_evals_16[label] = {
            "label": label,
            "n_records_total": doc.get("n_records_total"),
            "n_unique_sequences_folded": doc.get("n_unique_sequences_folded"),
            "r5a_ladder": ladder_compact,
            "r5b_denovo": denovo_compact,
            "r5c_conditioned": r5c_raw if r5c_raw else None,
            "source_path": str(src.relative_to(repo_root)) if src.is_absolute() and repo_root in src.parents else str(src),
        }

    # Map expanded GP R5 arms (12 arms + esm3-1.4b) with uniform_random as primary DPLM entry
    gp_r5_primary: dict[str, Any] = {}
    dplm_maxconf_diagnostics: dict[str, Any] = {}

    gp_r5_specs = [
        ("dayhoff-170m", "max_confidence", "free_length_causal", "r5_struct_dayhoff_170m"),
        ("protgpt3-112m", "max_confidence", "free_length_causal", "r5_struct_protgpt3_112m"),
        ("protgpt3-10b", "max_confidence", "free_length_causal", "r5_struct_protgpt3_10b"),
        ("esm2-8m", "max_confidence", "bidirectional_infill", "r5_struct_esm2_8m_infill"),
        ("esm2-35m", "max_confidence", "bidirectional_infill", "r5_struct_esm2_35m_infill"),
        ("esm2-150m", "max_confidence", "bidirectional_infill", "r5_struct_esm2_150m_infill"),
        ("esm2-650m", "max_confidence", "bidirectional_infill", "r5_struct_esm2_650m_infill"),
        ("esm2-3b", "max_confidence", "bidirectional_infill", "r5_struct_esm2_3b_infill"),
        ("esm2-15b", "max_confidence", "bidirectional_infill", "r5_struct_esm2_15b_infill"),
        ("dplm-150m", "uniform_random", "length_matched_diffusion", "r5_struct_dplm_150m_uniform"),
        ("dplm-650m", "uniform_random", "length_matched_diffusion", "r5_struct_dplm_650m_uniform"),
        ("dplm-3b", "uniform_random", "length_matched_diffusion", "r5_struct_dplm_3b_uniform"),
        ("esm3-1.4b", "max_confidence", "length_matched_diffusion_and_conditioned", "r5_struct_esm3_1_4b"),
    ]

    for arm_id, dmode, regime, struct_label in gp_r5_specs:
        art = resolve_gp_r5_json(arm_id, diffusion_mode=dmode, root=repo_root)
        doc = load_json_file(art.path)
        entry: dict[str, Any] = {
            **_arm_header(arm_id),
            "generation_regime": regime,
            "diffusion_sampling_mode": dmode if arm_id.startswith("dplm-") else None,
            "dtype_realised": art.dtype_realised,
            "ladder_summary": doc.get("ladder_summary") or (doc.get("results") or {}).get("r5a_ladder"),
            "denovo_summary": doc.get("denovo_summary") or (doc.get("results") or {}).get("r5b_denovo"),
            "esmfold2_structure_eval": struct_evals_16.get(struct_label),
            "source_path": str(art.path.relative_to(repo_root)),
        }
        gp_r5_primary[arm_id] = entry

    # Attach DPLM max_confidence explicitly as homopolymer-collapse diagnostic
    for arm_id, struct_label in (
        ("dplm-150m", "r5_struct_dplm_150m_maxconf"),
        ("dplm-650m", "r5_struct_dplm_650m_maxconf"),
        ("dplm-3b", "r5_struct_dplm_3b_maxconf"),
    ):
        art = resolve_gp_r5_json(arm_id, diffusion_mode="max_confidence", root=repo_root)
        doc = load_json_file(art.path)
        dplm_maxconf_diagnostics[arm_id] = {
            **_arm_header(arm_id),
            "generation_regime": "length_matched_diffusion_maxconf_diagnostic",
            "diffusion_sampling_mode": "max_confidence",
            "diagnostic_note": "Greedy max-confidence unmasking induces severe poly-L/S homopolymer collapse; reported strictly as diagnostic beside primary uniform_random entry.",
            "denovo_summary": doc.get("denovo_summary"),
            "esmfold2_structure_eval": struct_evals_16.get(struct_label),
            "source_path": str(art.path.relative_to(repo_root)),
        }

    r5_summary = {
        "n_exp_r2_246_causal_arms": len(exp246),
        "n_gp_and_esm3_primary_arms": len(gp_r5_primary),
        "n_esmfold2_structure_evaluations": len(struct_evals_16),
        "exp_r2_246_causal_arms": exp246,
        "ladder_original_6_arms": ladder_orig,
        "general_plm_and_esm3_primary_arms": gp_r5_primary,
        "dplm_max_confidence_homopolymer_diagnostics": dplm_maxconf_diagnostics,
        "esmfold2_structure_evaluations_16": struct_evals_16,
    }

    # Unified R6: 5-pair post-training alignment & distinct-5-mer filter control
    r6_ep_path = repo_root / "results/extensions/objective_tuned_endpoints_20261112/objective_tuned_endpoints.json"
    r6_fm_path = repo_root / "results/extensions/objective_tuned_five_mer_rule_20261114/five_mer_rule.json"
    r6_ep = load_json_file(r6_ep_path)
    r6_fm = load_json_file(r6_fm_path)

    primary_bands = (r6_ep.get("primary") or {}).get("bands", {}).get("columns", {})
    secondary_bands = (r6_ep.get("secondary_pair") or {}).get("bands", {}).get("columns", {})
    all_band_cols = {**primary_bands, **secondary_bands}

    pairs_summary: dict[str, dict[str, Any]] = defaultdict(dict)
    for col_key, col_val in sorted(all_band_cols.items()):
        pair_name, stratum = col_key.split("|", 1)
        pairs_summary[pair_name][stratum] = {
            "base_mean": col_val.get("base_mean"),
            "tuned_mean": col_val.get("tuned_mean"),
            "diff_mean": col_val.get("diff_mean") if "diff_mean" in col_val else col_val.get("point"),
            "ci95": col_val.get("ci95"),
            "status": col_val.get("status"),
        }

    selector_raw = (r6_ep.get("selector_competence") or {}).get("per_arm") or {}
    selector_compact = {
        arm: {
            "n_pool": val.get("n_pool"),
            "pool_mean_plddt": val.get("pool_mean_plddt"),
            "oracle_gain": val.get("oracle_gain"),
            "selector_gain": (val.get("captured") or {}).get("selector_gain"),
            "captured_fraction": (val.get("captured") or {}).get("captured_fraction"),
        }
        for arm, val in sorted(selector_raw.items())
    }

    r6_summary = {
        "n_contrast_pairs_with_band_tables": len(pairs_summary),
        "primary_and_secondary_band_contrasts": dict(pairs_summary),
        "identity_shift": r6_ep.get("identity_shift"),
        "selector_competence_per_arm": selector_compact,
        "distinct_5mer_filter_control": r6_fm.get("length_matched_gain", []),
        "sources": [
            str(r6_ep_path.relative_to(repo_root)),
            str(r6_fm_path.relative_to(repo_root)),
        ],
    }
    return r5_summary, r6_summary


def build_r4_r5_r6_unified_summary(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Assemble the complete unified R4 (30+1), R5 master ledger, and R6 summary."""
    repo_root = Path(repo_root)
    r4 = _build_unified_r4(repo_root)
    r5, r6 = _build_unified_r5_and_r6(repo_root)
    return {
        "schema": "unified_r4_r5_r6_summary_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "r4_context_access": r4,
        "r5_generative_expression": r5,
        "r6_post_training_alignment": r6,
    }


def write_unified_memory_and_generation_outputs(
    repo_root: Path = REPO_ROOT,
    out_dir: Path | None = None,
    *,
    draws: int = 10000,
    seed: int = 20261106,
) -> dict[str, Any]:
    """Run both builders and write both atomic JSON artifacts + .sha256 sidecars."""
    repo_root = Path(repo_root)
    target_dir = Path(out_dir) if out_dir is not None else (repo_root / UNIFIED_OUTPUT_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    fam_doc = build_family_memory_unified_62(repo_root, draws=draws, seed=seed)
    fam_path = target_dir / "family_memory_unified_62.json"
    fam_sha = write_json_with_sha256(fam_path, fam_doc)

    r456_doc = build_r4_r5_r6_unified_summary(repo_root)
    r456_path = target_dir / "r4_r5_r6_unified_summary.json"
    r456_sha = write_json_with_sha256(r456_path, r456_doc)

    return {
        "family_memory_unified_62": {"path": str(fam_path), "sha256": fam_sha, "doc": fam_doc},
        "r4_r5_r6_unified_summary": {"path": str(r456_path), "sha256": r456_sha, "doc": r456_doc},
    }
