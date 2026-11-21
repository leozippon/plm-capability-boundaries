"""Track B1: Zero-GPU capacity-starvation, entropy-budget, and tokenization re-analysis.

Reads frozen repository artifacts (family_conditionals_20261107,
family_distribution_20261106, statistical_ladder_20261106,
readout_panel_summary_20260923, explicit_statistics_lanes_20261110, R4
context_homologue, R6 objective_tuned) and any optional extra score directories
from later waves (ESM2, DPLM, ProtGPT3, ProGen3, etc.), emitting a complete,
bootstrap-verified JSON report.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.amino_acids import AA20
from ..extensions.family_conditionals import _attach_baselines
from .io_utils import (
    LN2,
    MODEL_MODALITY,
    MODEL_PARAM_COUNTS,
    SCALING_LADDERS,
    cluster_bootstrap_mean,
    cluster_bootstrap_scaling_slope,
    load_json_file,
    paired_cluster_bootstrap_diff,
)

AA_INDEX = {aa: idx for idx, aa in enumerate(AA20)}
UNIFORM_AA20_BITS = math.log2(len(AA20))  # 4.321928094887363

CONSERVATION_TIERS = (
    ("conserved", 0.0, 1.0),
    ("moderate", 1.0, 2.5),
    ("variable", 2.5, float("inf")),
)


def shannon_entropy_bits(probs: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(probs, dtype=np.float64)
    arr = arr[arr > 0.0]
    if arr.size == 0:
        return 0.0
    arr = arr / np.sum(arr)
    return float(-np.sum(arr * np.log2(arr)))


def classify_conservation_tier(h_prof_bits: float) -> str:
    for name, lo, hi in CONSERVATION_TIERS:
        if lo <= h_prof_bits < hi:
            return name
    return "variable"


def _discover_score_docs(
    score_roots: Sequence[Path],
    filename: str = "evaluation.json",
) -> dict[str, Path]:
    """Discover per-arm score JSONs across primary and optional extra roots."""
    found: dict[str, Path] = {}
    for root in score_roots:
        if not root or not root.exists():
            continue
        for entry in sorted(root.iterdir()):
            if entry.is_dir():
                candidate = entry / filename
                if candidate.is_file():
                    found[entry.name] = candidate
            elif entry.is_file() and entry.suffix == ".json":
                found[entry.stem] = entry
    return found


def discover_general_plm_family_conditionals(
    search_roots: Sequence[Path],
    unit_order: Sequence[str],
    site_records: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, str]]:
    """Discover V2 general-PLM family_conditionals JSONs and ESM3 per-site JSONLs.

    Returns:
        family_bits_by_arm: {arm: {unit_id: bits_per_res}} for all 64 eval units.
        extra_site_bits_by_arm: {arm: np.ndarray(len(site_records))} when per-site
            contexts exist (e.g. esm3-1.4b r1_family_conditionals.jsonl).
        semantics_by_arm: {arm: conditional_semantics_label}.
    """
    expected_units = set(unit_order)
    family_bits_by_arm: dict[str, dict[str, float]] = {}
    extra_site_bits_by_arm: dict[str, np.ndarray] = {}
    semantics_by_arm: dict[str, str] = {}

    # Build (unit_id, query_position, row_id) -> site index lookup for per-site JSONL ingestion
    ctx_to_site: dict[tuple[str, str, str], int] = {}
    site_n_contexts = [int(s["n_contexts"]) for s in site_records]
    for s_idx, srec in enumerate(site_records):
        uid = srec["unit_id"]
        pos = str(srec["query_position"])
        for rid in srec.get("row_ids", []):
            ctx_to_site[(uid, pos, str(rid))] = s_idx

    for root in search_roots:
        if not root or not root.exists():
            continue
        for r1_path in sorted(root.rglob("r1_ladder_and_conditionals_*.json")):
            doc = load_json_file(r1_path)
            arm = str(doc.get("arm", "")).strip()
            fc = doc.get("family_conditionals")
            if not arm or not isinstance(fc, dict):
                continue
            eval_map: dict[str, float] = {}
            for u in fc.get("units", []):
                if u.get("partition") != "evaluation":
                    continue
                uid = str(u["unit_id"])
                if uid in expected_units and u.get("model_logp") is not None:
                    val = float(-float(u["model_logp"]) / LN2)
                    if math.isfinite(val):
                        eval_map[uid] = val
            if len(eval_map) == len(expected_units):
                family_bits_by_arm[arm] = eval_map
                semantics_by_arm[arm] = str(
                    fc.get("conditional_semantics", "general_plm_family_conditional")
                )

        # Ingest ESM3 per-site/per-context family_conditionals JSONL when present
        for jsonl_path in sorted(root.rglob("r1_family_conditionals.jsonl")):
            arm = "esm3-1.4b"
            per_site_vals: list[list[float]] = [[] for _ in range(len(site_records))]
            with jsonl_path.open("r", encoding="utf-8") as fh:
                import json

                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("partition") != "evaluation":
                        continue
                    key = (str(row["unit_id"]), str(row["query_position"]), str(row["row_id"]))
                    s_idx = ctx_to_site.get(key)
                    if s_idx is not None and row.get("observed_logp_nats") is not None:
                        per_site_vals[s_idx].append(float(-float(row["observed_logp_nats"]) / LN2))
            if all(len(vs) == n_exp and n_exp > 0 for vs, n_exp in zip(per_site_vals, site_n_contexts)):
                site_arr = np.asarray([float(np.mean(vs)) for vs in per_site_vals], dtype=np.float64)
                extra_site_bits_by_arm[arm] = site_arr
                uids, means, _ = _aggregate_sites_by_family(site_records, site_arr, tier_filter=None)
                family_bits_by_arm[arm] = dict(zip(uids, means.tolist()))
                semantics_by_arm[arm] = "esm3_seq_only_1pass_masked_conditional_over_aa20"

    return family_bits_by_arm, extra_site_bits_by_arm, semantics_by_arm


def discover_general_plm_family_distribution(
    search_roots: Sequence[Path],
    unit_order: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Discover and merge (including multi-shard) general-PLM family_distribution JSONs."""
    expected_units = set(unit_order)
    merged: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: {
            "native_bits": {},
            "column_null_delta_bits": {},
            "composition_null_delta_bits": {},
            "tokens_per_residue": {},
            "bits_per_token": {},
        }
    )

    for root in search_roots:
        if not root or not root.exists():
            continue
        for fd_path in sorted(root.rglob("r1_family_distribution_*.json")):
            doc = load_json_file(fd_path)
            arm = str(doc.get("arm", "")).strip()
            if not arm:
                continue
            for u in doc.get("units", []):
                if u.get("partition") != "evaluation":
                    continue
                uid = str(u["unit_id"])
                if uid not in expected_units:
                    continue
                col = u.get("column_contrast") or {}
                comp = u.get("composition_contrast") or {}
                tot_len = float(col.get("total_length") or u.get("total_residues") or 0.0)
                if tot_len <= 0.0:
                    continue
                nat_nats = float(col["native_nats"])
                col_delta_nats = float(col["nats_per_residue"])
                comp_delta_nats = float(comp.get("nats_per_residue", 0.0))
                merged[arm]["native_bits"][uid] = float((nat_nats / tot_len) / LN2)
                merged[arm]["column_null_delta_bits"][uid] = float(col_delta_nats / LN2)
                merged[arm]["composition_null_delta_bits"][uid] = float(comp_delta_nats / LN2)
                merged[arm]["tokens_per_residue"][uid] = 1.0
                merged[arm]["bits_per_token"][uid] = float((nat_nats / tot_len) / LN2)

    complete: dict[str, dict[str, Any]] = {}
    for arm, rec in sorted(merged.items()):
        if len(rec["native_bits"]) == len(expected_units):
            complete[arm] = rec
    return complete


def load_msa_support(
    support_path: Path,
    *,
    verify_weights_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load per-wildtype MSA depth (N_eff / L) and verify against .weights files."""
    raw = load_json_file(support_path)
    out: dict[str, dict[str, Any]] = {}
    for wt_id, rec in raw["wildtypes"].items():
        length = int(rec["length"])
        neff_30 = float(rec["neff_at_30_percent_identity"])
        weights_sum = float(rec["weights_sum"])
        eff_depth = float(rec["effective_depth_per_site"])
        verified_weights_sum: float | None = None
        if verify_weights_dir is not None:
            w_path = verify_weights_dir / f"{wt_id}.weights"
            if w_path.is_file():
                vals = np.loadtxt(w_path, dtype=np.float64)
                verified_weights_sum = float(np.sum(vals))
        out[str(wt_id)] = {
            "wildtype_id": str(wt_id),
            "length": length,
            "rows": int(rec["rows"]),
            "neff_30": neff_30,
            "weights_sum": weights_sum,
            "verified_weights_sum": verified_weights_sum,
            "effective_depth_per_site": eff_depth,
            "neff_per_len_weights": weights_sum / max(length, 1),
            "assays": list(rec.get("assays", [])),
        }
    return out


# ---------------------------------------------------------------------------
# (a) Site-Conservation Stratification & Family-Conditional Extraction
# ---------------------------------------------------------------------------


def extract_family_conditionals_records(
    fc_root: Path,
    *,
    extra_score_dirs: Sequence[Path] = (),
) -> dict[str, Any]:
    """Load evaluation units, per-site H_prof, baselines, and all scored models."""
    manifest = load_json_file(fc_root / "conditional-inputs.json")
    eval_units = [u for u in manifest["units"] if u["partition"] == "evaluation"]
    for unit in eval_units:
        _attach_baselines(unit)

    score_paths = _discover_score_docs(
        [fc_root / "scores", *extra_score_dirs],
        filename="evaluation.json",
    )

    # Pre-compute site metadata and baseline losses (in bits/res)
    site_records: list[dict[str, Any]] = []
    unit_order: list[str] = []
    unit_to_wt: dict[str, str] = {}
    for unit in sorted(eval_units, key=lambda u: u["unit_id"]):
        uid = str(unit["unit_id"])
        wt_id = str(unit["wildtype_id"])
        unit_order.append(uid)
        unit_to_wt[uid] = wt_id
        for site in unit["sites"]:
            pos = str(site["query_position"])
            freq = np.asarray(unit["baselines"]["profile_frequencies"][pos], dtype=np.float64)
            h_prof = shannon_entropy_bits(freq)
            tier = classify_conservation_tier(h_prof)
            eff_alpha = float(2.0 ** h_prof)
            top1_prob = float(np.max(freq))

            prof_lp = np.asarray(site["baseline"]["profile"], dtype=np.float64)
            markov_lps = site["baseline"]["markov"]
            prof_ctx_bits = []
            markov_ctx_bits = []
            obs_indices = []
            for ctx, mlp in zip(site["contexts"], markov_lps):
                obs_idx = AA_INDEX[ctx["observed"]]
                obs_indices.append(obs_idx)
                prof_ctx_bits.append(float(-prof_lp[obs_idx] / LN2))
                markov_ctx_bits.append(float(-mlp[obs_idx] / LN2))

            site_records.append(
                {
                    "unit_id": uid,
                    "wildtype_id": wt_id,
                    "query_position": pos,
                    "n_contexts": len(site["contexts"]),
                    "training_rows_with_site": int(site["training_rows_with_site"]),
                    "heldout_rows_with_site": int(site["heldout_rows_with_site"]),
                    "h_prof_bits": h_prof,
                    "effective_alphabet": eff_alpha,
                    "top1_prob": top1_prob,
                    "tier": tier,
                    "row_ids": [str(ctx["row_id"]) for ctx in site["contexts"]],
                    "obs_indices": obs_indices,
                    "profile_bits": float(np.mean(prof_ctx_bits)),
                    "markov_bits": float(np.mean(markov_ctx_bits)),
                }
            )

    # Load each arm's per-site loss (bits/res) aligned with site_records
    arm_site_bits: dict[str, np.ndarray] = {}
    arm_site_bits["profile"] = np.asarray([s["profile_bits"] for s in site_records], dtype=np.float64)
    arm_site_bits["markov"] = np.asarray([s["markov_bits"] for s in site_records], dtype=np.float64)

    for arm, spath in sorted(score_paths.items()):
        doc = load_json_file(spath)
        units_map = doc.get("units", {})
        if not all(uid in units_map for uid in unit_order):
            continue
        vals = np.empty(len(site_records), dtype=np.float64)
        valid = True
        for i, srec in enumerate(site_records):
            uid = srec["unit_id"]
            pos = srec["query_position"]
            s_entry = units_map[uid]["sites"].get(pos)
            if s_entry is None:
                valid = False
                break
            lps = s_entry["logprob"]
            obs_idx = srec["obs_indices"]
            if len(lps) != len(obs_idx):
                valid = False
                break
            ctx_bits = [-float(lp[idx]) / LN2 for lp, idx in zip(lps, obs_idx)]
            vals[i] = float(np.mean(ctx_bits))
        if valid and np.isfinite(vals).all():
            arm_site_bits[arm] = vals

    return {
        "unit_order": unit_order,
        "unit_to_wt": unit_to_wt,
        "site_records": site_records,
        "arm_site_bits": arm_site_bits,
        "scored_arms": sorted(a for a in arm_site_bits if a not in ("profile", "markov")),
    }


def _aggregate_sites_by_family(
    site_records: Sequence[dict[str, Any]],
    values: np.ndarray,
    tier_filter: str | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (unit_ids, family_means, site_counts) for sites matching tier_filter."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for rec, val in zip(site_records, values):
        if tier_filter is None or rec["tier"] == tier_filter:
            grouped[rec["unit_id"]].append(float(val))
    uids = sorted(grouped.keys())
    means = np.asarray([float(np.mean(grouped[u])) for u in uids], dtype=np.float64)
    counts = np.asarray([float(len(grouped[u])) for u in uids], dtype=np.float64)
    return uids, means, counts


def compute_site_conservation_stratification(
    fc_data: dict[str, Any],
    *,
    n_draws: int = 2000,
    seed: int = 20261120,
) -> dict[str, Any]:
    """(a) Site-conservation stratification with paired family-cluster bootstrap 95% CIs."""
    sites = fc_data["site_records"]
    arm_bits = fc_data["arm_site_bits"]
    prof_arr = arm_bits["profile"]
    markov_arr = arm_bits["markov"]

    h_vals = np.asarray([s["h_prof_bits"] for s in sites], dtype=np.float64)
    eff_vals = np.asarray([s["effective_alphabet"] for s in sites], dtype=np.float64)
    top1_vals = np.asarray([s["top1_prob"] for s in sites], dtype=np.float64)

    tier_counts = {}
    for tier_name, lo, hi in CONSERVATION_TIERS:
        tier_sites = [s for s in sites if s["tier"] == tier_name]
        tier_fams = len({s["unit_id"] for s in tier_sites})
        tier_counts[tier_name] = {
            "n_sites": len(tier_sites),
            "fraction_sites": float(len(tier_sites) / len(sites)),
            "n_families": tier_fams,
            "h_prof_range_bits": [lo, None if math.isinf(hi) else hi],
        }

    h_prof_summary = {
        "n_sites": len(sites),
        "n_families": len(fc_data["unit_order"]),
        "n_contexts": int(sum(s["n_contexts"] for s in sites)),
        "mean_bits": float(np.mean(h_vals)),
        "median_bits": float(np.median(h_vals)),
        "sd_bits": float(np.std(h_vals, ddof=1)),
        "quantiles_bits": {
            f"p{q}": float(np.quantile(h_vals, q / 100.0)) for q in (10, 25, 50, 75, 90)
        },
        "effective_alphabet_mean": float(np.mean(eff_vals)),
        "effective_alphabet_median": float(np.median(eff_vals)),
        "top1_prob_mean": float(np.mean(top1_vals)),
        "tiers": tier_counts,
    }

    tiers_to_eval = [None, "conserved", "moderate", "variable"]
    models_out: dict[str, Any] = {}
    all_entries = ["profile", "markov", *fc_data["scored_arms"]]

    for arm in all_entries:
        vals = arm_bits[arm]
        arm_tiers: dict[str, Any] = {}
        for tier in tiers_to_eval:
            tkey = "overall" if tier is None else tier
            uids, fam_m, counts = _aggregate_sites_by_family(sites, vals, tier_filter=tier)
            _, prof_m, _ = _aggregate_sites_by_family(sites, prof_arr, tier_filter=tier)
            _, mkv_m, _ = _aggregate_sites_by_family(sites, markov_arr, tier_filter=tier)

            # Site-weighted family-cluster bootstrap (matches pooled site mean while resampling families)
            site_weighted = cluster_bootstrap_mean(
                fam_m, weights=counts, n_draws=n_draws, seed=seed
            )
            equal_family = cluster_bootstrap_mean(
                fam_m, weights=None, n_draws=n_draws, seed=seed
            )
            vs_profile_sw = paired_cluster_bootstrap_diff(
                fam_m, prof_m, weights=counts, n_draws=n_draws, seed=seed
            )
            vs_profile_eq = paired_cluster_bootstrap_diff(
                fam_m, prof_m, weights=None, n_draws=n_draws, seed=seed
            )
            vs_markov_sw = paired_cluster_bootstrap_diff(
                fam_m, mkv_m, weights=counts, n_draws=n_draws, seed=seed
            )

            arm_tiers[tkey] = {
                "site_weighted_bits": site_weighted,
                "equal_family_bits": equal_family,
                "minus_profile_site_weighted_bits": vs_profile_sw,
                "minus_profile_equal_family_bits": vs_profile_eq,
                "minus_markov_site_weighted_bits": vs_markov_sw,
                "n_sites": int(np.sum(counts)),
                "n_families": len(uids),
            }

        models_out[arm] = {
            "params": MODEL_PARAM_COUNTS.get(arm),
            "modality": MODEL_MODALITY.get(arm, "baseline" if arm in ("profile", "markov") else "unknown"),
            "tiers": arm_tiers,
        }

    return {
        "site_entropy_summary": h_prof_summary,
        "models": models_out,
    }


# ---------------------------------------------------------------------------
# Family-Distribution Extraction (Native bits/res & Column-null Delta)
# ---------------------------------------------------------------------------


def extract_family_distribution_records(
    fd_root: Path,
    *,
    extra_score_dirs: Sequence[Path] = (),
) -> dict[str, Any]:
    """Extract per-family native cross-entropy (bits/res), column-null delta, and tokenization stats."""
    summary = load_json_file(fd_root / "analysis/summary.json")
    eval_units_summary = summary["units"]["evaluation"]
    unit_order = sorted(eval_units_summary.keys())

    # Baselines (profile & markov) per family from analysis summary
    unit_to_wt: dict[str, str] = {}
    baseline_native_bits: dict[str, dict[str, float]] = {"profile": {}, "markov": {}}
    baseline_col_delta_bits: dict[str, dict[str, float]] = {"profile": {}, "markov": {}}
    unit_total_residues: dict[str, float] = {}

    for uid in unit_order:
        wt_dict = eval_units_summary[uid]
        wt_id = next(iter(wt_dict.keys()))
        unit_to_wt[uid] = wt_id
        rec = wt_dict[wt_id]
        total_len = float(rec["profile"]["total_length"])
        unit_total_residues[uid] = total_len
        for bname in ("profile", "markov"):
            brec = rec[bname]
            baseline_native_bits[bname][uid] = float((brec["native_nats"] / total_len) / LN2)
            baseline_col_delta_bits[bname][uid] = float(brec["nats_per_residue"] / LN2)

    score_paths = _discover_score_docs(
        [fd_root / "scores", *extra_score_dirs],
        filename="evaluation.json",
    )

    arms_data: dict[str, dict[str, Any]] = {}
    for arm, spath in sorted(score_paths.items()):
        doc = load_json_file(spath)
        units_map = doc.get("units", {})
        if not all(uid in units_map for uid in unit_order):
            continue

        native_bits = {}
        col_delta_bits = {}
        comp_delta_bits = {}
        tok_per_res = {}
        bits_per_tok = {}
        valid = True

        for uid in unit_order:
            wt_id = unit_to_wt[uid]
            wt_entry = units_map[uid]["wildtypes"].get(wt_id)
            if wt_entry is None:
                valid = False
                break
            lengths = np.asarray(wt_entry["lengths"], dtype=np.float64)
            tot_res = float(np.sum(lengths))
            nat_nll = np.asarray(wt_entry["native_nll"], dtype=np.float64)
            col_nll = np.asarray(wt_entry["column_null_nll"], dtype=np.float64)  # (S, N)
            comp_nll = np.asarray(wt_entry["composition_null_nll"], dtype=np.float64)  # (S, N)
            toks = np.asarray(wt_entry["scored_tokens"]["native"], dtype=np.float64)

            tot_nat_nats = float(np.sum(nat_nll))
            mean_col_nats = (
                float(np.sum(col_nll))
                if col_nll.ndim == 1
                else float(np.mean(np.sum(col_nll, axis=1)))
            )
            mean_comp_nats = (
                float(np.sum(comp_nll))
                if comp_nll.ndim == 1
                else float(np.mean(np.sum(comp_nll, axis=1)))
            )
            tot_toks = float(np.sum(toks))

            native_bits[uid] = float((tot_nat_nats / tot_res) / LN2)
            # Positive delta = null has HIGHER NLL than native (model prefers native)
            col_delta_bits[uid] = float(((mean_col_nats - tot_nat_nats) / tot_res) / LN2)
            comp_delta_bits[uid] = float(((mean_comp_nats - tot_nat_nats) / tot_res) / LN2)
            tok_per_res[uid] = float(tot_toks / tot_res)
            bits_per_tok[uid] = float((tot_nat_nats / max(tot_toks, 1.0)) / LN2)

        if valid:
            arms_data[arm] = {
                "native_bits": native_bits,
                "column_null_delta_bits": col_delta_bits,
                "composition_null_delta_bits": comp_delta_bits,
                "tokens_per_residue": tok_per_res,
                "bits_per_token": bits_per_tok,
            }

    return {
        "unit_order": unit_order,
        "unit_to_wt": unit_to_wt,
        "unit_total_residues": unit_total_residues,
        "baselines": {
            "profile": {
                "native_bits": baseline_native_bits["profile"],
                "column_null_delta_bits": baseline_col_delta_bits["profile"],
            },
            "markov": {
                "native_bits": baseline_native_bits["markov"],
                "column_null_delta_bits": baseline_col_delta_bits["markov"],
            },
        },
        "arms": arms_data,
    }


# ---------------------------------------------------------------------------
# (b) Family Evolutionary Density & Conservation Scaling Slopes
# ---------------------------------------------------------------------------


def compute_density_and_conservation_scaling(
    fc_data: dict[str, Any],
    fd_data: dict[str, Any],
    msa_support: Mapping[str, dict[str, Any]],
    sl_root: Path,
    otp_result_path: Path | None = None,
    *,
    n_draws: int = 2000,
    seed: int = 20261120,
) -> dict[str, Any]:
    """(b) Scaling slopes -dH/d(log10 N) stratified by MSA depth (N_eff/L) & site conservation."""
    unit_order = fc_data["unit_order"]
    unit_to_wt = fc_data["unit_to_wt"]

    depths = np.asarray(
        [msa_support[unit_to_wt[u]]["effective_depth_per_site"] for u in unit_order],
        dtype=np.float64,
    )
    q25, q50, q75 = [float(x) for x in np.quantile(depths, [0.25, 0.50, 0.75])]

    # Define non-overlapping quartile and threshold strata of families
    family_strata: dict[str, list[str]] = {
        "all": list(unit_order),
        "neff_lt_5": [u for u, d in zip(unit_order, depths) if d < 5.0],
        "neff_ge_5": [u for u, d in zip(unit_order, depths) if d >= 5.0],
        "neff_below_median": [u for u, d in zip(unit_order, depths) if d < q50],
        "neff_ge_median": [u for u, d in zip(unit_order, depths) if d >= q50],
        "Q1_lowest": [u for u, d in zip(unit_order, depths) if d <= q25],
        "Q2": [u for u, d in zip(unit_order, depths) if q25 < d <= q50],
        "Q3": [u for u, d in zip(unit_order, depths) if q50 < d <= q75],
        "Q4_highest": [u for u, d in zip(unit_order, depths) if d > q75],
    }

    density_summary = {
        "n_families": len(unit_order),
        "metric": "effective_depth_per_site (N_eff_30 / L from statistical_ladder_20261106/inputs/support.json)",
        "min": float(np.min(depths)),
        "max": float(np.max(depths)),
        "mean": float(np.mean(depths)),
        "median": q50,
        "quartiles": {"q25": q25, "q50": q50, "q75": q75},
        "strata_counts": {k: len(v) for k, v in family_strata.items()},
    }

    # Build per-family FC matrix (equal-site within family)
    fc_family_by_arm: dict[str, dict[str, float]] = {}
    for arm, arr in fc_data["arm_site_bits"].items():
        uids, means, _ = _aggregate_sites_by_family(fc_data["site_records"], arr, tier_filter=None)
        fc_family_by_arm[arm] = dict(zip(uids, means.tolist()))

    # 1. Scaling slopes on family_conditionals by MSA density & by site conservation
    fc_ladders: dict[str, Any] = {}
    for ladder_name, rungs in SCALING_LADDERS.items():
        present = [r for r in rungs if r in fc_data["arm_site_bits"] and r in MODEL_PARAM_COUNTS]
        if len(present) < 2:
            continue
        log10_n = np.asarray([math.log10(MODEL_PARAM_COUNTS[r]) for r in present], dtype=np.float64)

        by_density: dict[str, Any] = {}
        for sname, suids in family_strata.items():
            if len(suids) < 2:
                by_density[sname] = {"n_clusters": len(suids), "status": "insufficient_families"}
                continue
            mat = np.asarray(
                [[fc_family_by_arm[r][u] for u in suids] for r in present],
                dtype=np.float64,
            )
            by_density[sname] = cluster_bootstrap_scaling_slope(
                log10_n, mat, n_draws=n_draws, seed=seed
            )

        # Contrast Q4 vs Q1 and neff_ge_5 vs neff_lt_5 (independent strata bootstrap)
        by_conservation: dict[str, Any] = {}
        for tier in ("overall", "conserved", "moderate", "variable"):
            t_arg = None if tier == "overall" else tier
            # Align units across all rungs
            uids_ref, _, counts_ref = _aggregate_sites_by_family(
                fc_data["site_records"], fc_data["arm_site_bits"][present[0]], tier_filter=t_arg
            )
            mat = np.zeros((len(present), len(uids_ref)), dtype=np.float64)
            for ri, r in enumerate(present):
                uids_r, means_r, _ = _aggregate_sites_by_family(
                    fc_data["site_records"], fc_data["arm_site_bits"][r], tier_filter=t_arg
                )
                assert uids_r == uids_ref
                mat[ri] = means_r
            by_conservation[tier] = cluster_bootstrap_scaling_slope(
                log10_n, mat, cluster_weights=counts_ref, n_draws=n_draws, seed=seed
            )

        # Paired slope difference: conserved minus variable within the same families
        c_uids, _, c_w = _aggregate_sites_by_family(
            fc_data["site_records"], fc_data["arm_site_bits"][present[0]], tier_filter="conserved"
        )
        v_uids, _, v_w = _aggregate_sites_by_family(
            fc_data["site_records"], fc_data["arm_site_bits"][present[0]], tier_filter="variable"
        )
        shared_cv = sorted(set(c_uids) & set(v_uids))
        cv_contrast = None
        if len(shared_cv) >= 4:
            c_map_by_r = [
                dict(zip(*_aggregate_sites_by_family(fc_data["site_records"], fc_data["arm_site_bits"][r], tier_filter="conserved")[:2]))
                for r in present
            ]
            v_map_by_r = [
                dict(zip(*_aggregate_sites_by_family(fc_data["site_records"], fc_data["arm_site_bits"][r], tier_filter="variable")[:2]))
                for r in present
            ]
            mat_diff = np.asarray(
                [[c_map_by_r[ri][u] - v_map_by_r[ri][u] for u in shared_cv] for ri in range(len(present))],
                dtype=np.float64,
            )
            cv_contrast = cluster_bootstrap_scaling_slope(
                log10_n, mat_diff, n_draws=n_draws, seed=seed
            )

        fc_ladders[ladder_name] = {
            "rungs": present,
            "params": [MODEL_PARAM_COUNTS[r] for r in present],
            "by_msa_density": by_density,
            "by_site_conservation": by_conservation,
            "conserved_minus_variable_slope_shared_families": cv_contrast,
        }

    # 2. Scaling slopes on family_distribution (native bits/res & column-null delta)
    fd_ladders: dict[str, Any] = {}
    for ladder_name, rungs in SCALING_LADDERS.items():
        present = [r for r in rungs if r in fd_data["arms"] and r in MODEL_PARAM_COUNTS]
        if len(present) < 2:
            continue
        log10_n = np.asarray([math.log10(MODEL_PARAM_COUNTS[r]) for r in present], dtype=np.float64)
        native_by_density: dict[str, Any] = {}
        col_delta_by_density: dict[str, Any] = {}
        for sname, suids in family_strata.items():
            if len(suids) < 2:
                native_by_density[sname] = {"n_clusters": len(suids), "status": "insufficient_families"}
                col_delta_by_density[sname] = {"n_clusters": len(suids), "status": "insufficient_families"}
                continue
            mat_nat = np.asarray(
                [[fd_data["arms"][r]["native_bits"][u] for u in suids] for r in present],
                dtype=np.float64,
            )
            # For column-null delta, higher is better, so negate to get +d(Delta)/d(log10 N)
            mat_col = np.asarray(
                [[-fd_data["arms"][r]["column_null_delta_bits"][u] for u in suids] for r in present],
                dtype=np.float64,
            )
            native_by_density[sname] = cluster_bootstrap_scaling_slope(
                log10_n, mat_nat, n_draws=n_draws, seed=seed
            )
            col_delta_by_density[sname] = cluster_bootstrap_scaling_slope(
                log10_n, mat_col, n_draws=n_draws, seed=seed
            )

        fd_ladders[ladder_name] = {
            "rungs": present,
            "params": [MODEL_PARAM_COUNTS[r] for r in present],
            "native_bits_by_msa_density": native_by_density,
            "column_null_delta_gain_per_decade_by_msa_density": col_delta_by_density,
        }

    # 3. R1 DMS Statistical Ladder scaling (including ProtGPT3 112M -> 1.3B -> 10B!)
    r1_ladder_scaling = _extract_r1_dms_ladder_scaling(
        sl_root=sl_root,
        msa_support=msa_support,
        otp_result_path=otp_result_path,
        n_draws=n_draws,
        seed=seed,
    )

    return {
        "family_msa_depth_summary": density_summary,
        "family_conditionals_ladders": fc_ladders,
        "family_distribution_ladders": fd_ladders,
        "r1_dms_ladder_scaling": r1_ladder_scaling,
    }


def _extract_r1_dms_ladder_scaling(
    *,
    sl_root: Path,
    msa_support: Mapping[str, dict[str, Any]],
    otp_result_path: Path | None,
    n_draws: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize R1 statistical ladder scaling across ProGen2, ProGen3, Galactica, GPT2, and ProtGPT3."""
    contrasts_doc = load_json_file(sl_root / "full/contrasts.json")
    hyps = contrasts_doc["results"]["spearman"]["hypotheses"]
    by_model_peg: dict[str, dict[str, Any]] = defaultdict(dict)
    for h in hyps:
        if h["arm"] == "P+E+G":
            by_model_peg[h["model"]][h["direction"]] = {
                "point": float(h["point"]),
                "pointwise_ci95": [float(h["pointwise"][0]), float(h["pointwise"][1])],
                "simultaneous_ci95": [float(h["simultaneous"][0]), float(h["simultaneous"][1])],
                "verdict": h["verdict"],
            }

    # Build per-family spearman table from assay-metrics.json (M, PEG, MPEG) + MSA depth
    assay_to_depth: dict[str, float] = {}
    for wt_rec in msa_support.values():
        for assay in wt_rec["assays"]:
            assay_to_depth[assay] = wt_rec["effective_depth_per_site"]

    assay_rows = load_json_file(sl_root / "full/assay-metrics.json")
    # Average seeds within (model, cluster, assay), then equal assays within (model, cluster)
    by_mca: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    cluster_depths: dict[str, list[float]] = defaultdict(list)
    seen_cluster_assay: set[tuple[str, str]] = set()
    for row in assay_rows:
        m = row["model"]
        cl = str(row["cluster"])
        assay = row["assay"]
        sp = row["spearman"]
        for key in ("M", "PEG", "MPEG"):
            if key in sp and sp[key] is not None:
                by_mca[(m, cl, assay)][key].append(float(sp[key]))
        if (cl, assay) not in seen_cluster_assay and assay in assay_to_depth:
            seen_cluster_assay.add((cl, assay))
            cluster_depths[cl].append(assay_to_depth[assay])

    cluster_depth_mean = {cl: float(np.mean(vs)) for cl, vs in cluster_depths.items() if vs}

    # Collapse to (model, cluster)
    by_mc: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for (m, cl, _assay), metric_lists in by_mca.items():
        for key, vals in metric_lists.items():
            by_mc[m][cl][key].append(float(np.mean(vals)))

    family_scores: dict[str, dict[str, dict[str, float]]] = {}
    for m, cl_map in by_mc.items():
        family_scores[m] = {
            cl: {k: float(np.mean(vs)) for k, vs in kmap.items()}
            for cl, kmap in cl_map.items()
        }

    # Stratify 163 families by MSA depth
    all_clusters = sorted(cluster_depth_mean.keys())
    c_depths = np.asarray([cluster_depth_mean[c] for c in all_clusters], dtype=np.float64)
    cq25, cq50, cq75 = [float(x) for x in np.quantile(c_depths, [0.25, 0.50, 0.75])]
    c_strata = {
        "all": all_clusters,
        "neff_lt_5": [c for c, d in zip(all_clusters, c_depths) if d < 5.0],
        "neff_ge_5": [c for c, d in zip(all_clusters, c_depths) if d >= 5.0],
        "Q1_lowest": [c for c, d in zip(all_clusters, c_depths) if d <= cq25],
        "Q4_highest": [c for c, d in zip(all_clusters, c_depths) if d > cq75],
    }

    ladders_out: dict[str, Any] = {}
    for lname in ("progen2", "progen3", "galactica", "gpt2"):
        rungs = [r for r in SCALING_LADDERS[lname] if r in family_scores]
        if len(rungs) < 2:
            continue
        log10_n = np.asarray([math.log10(MODEL_PARAM_COUNTS[r]) for r in rungs], dtype=np.float64)
        rung_summaries = {}
        for r in rungs:
            cls = sorted(family_scores[r].keys())
            m_vals = np.asarray([family_scores[r][c]["M"] for c in cls], dtype=np.float64)
            peg_vals = np.asarray([family_scores[r][c]["PEG"] for c in cls], dtype=np.float64)
            mpeg_vals = np.asarray([family_scores[r][c]["MPEG"] for c in cls], dtype=np.float64)
            rung_summaries[r] = {
                "params": MODEL_PARAM_COUNTS[r],
                "raw_M_spearman": cluster_bootstrap_mean(m_vals, n_draws=n_draws, seed=seed),
                "PEG_spearman": float(np.mean(peg_vals)),
                "forward_M_given_PEG": by_model_peg.get(r, {}).get("model_given_controls"),
                "reverse_PEG_given_M": by_model_peg.get(r, {}).get("controls_given_model"),
                "MPEG_spearman": float(np.mean(mpeg_vals)),
            }

        # Scaling slopes of raw M and forward increment by MSA depth
        strata_slopes: dict[str, Any] = {}
        for sname, scls in c_strata.items():
            valid_cls = [c for c in scls if all(c in family_scores[r] for r in rungs)]
            if len(valid_cls) < 2:
                continue
            # Negate Spearman so cluster_bootstrap_scaling_slope returns +d(Spearman)/d(log10 N)
            mat_m = np.asarray(
                [[-family_scores[r][c]["M"] for c in valid_cls] for r in rungs],
                dtype=np.float64,
            )
            mat_fwd = np.asarray(
                [[-(family_scores[r][c]["MPEG"] - family_scores[r][c]["PEG"]) for c in valid_cls] for r in rungs],
                dtype=np.float64,
            )
            strata_slopes[sname] = {
                "n_families": len(valid_cls),
                "raw_M_spearman_gain_per_decade": cluster_bootstrap_scaling_slope(
                    log10_n, mat_m, n_draws=n_draws, seed=seed
                ),
                "forward_increment_gain_per_decade": cluster_bootstrap_scaling_slope(
                    log10_n, mat_fwd, n_draws=n_draws, seed=seed
                ),
            }
        ladders_out[lname] = {
            "rungs": rungs,
            "rung_summaries": rung_summaries,
            "by_msa_density": strata_slopes,
        }

    # Include ProtGPT3 112M -> 1.3B -> 10B ladder from objective_tuned_prediction_20261110
    protgpt3_ladder = None
    if otp_result_path is not None and otp_result_path.is_file():
        otp = load_json_file(otp_result_path)
        dfm = otp.get("descriptive_family_means", {})
        p3_rungs = ["protgpt3-112m", "protgpt3-1.3b", "protgpt3-10b"]
        if all(r in dfm for r in p3_rungs):
            entries = {}
            log10_n = np.asarray([math.log10(MODEL_PARAM_COUNTS[r]) for r in p3_rungs], dtype=np.float64)
            m_arr = np.asarray([float(dfm[r]["M"]) for r in p3_rungs], dtype=np.float64)
            peg_arr = np.asarray([float(dfm[r]["PEG"]) for r in p3_rungs], dtype=np.float64)
            mpeg_arr = np.asarray([float(dfm[r]["MPEG"]) for r in p3_rungs], dtype=np.float64)
            fwd_arr = mpeg_arr - peg_arr
            rev_arr = mpeg_arr - m_arr
            for r, m_v, p_v, mp_v, fw_v, rv_v in zip(p3_rungs, m_arr, peg_arr, mpeg_arr, fwd_arr, rev_arr):
                entries[r] = {
                    "params": MODEL_PARAM_COUNTS[r],
                    "raw_M_spearman": float(m_v),
                    "PEG_spearman": float(p_v),
                    "MPEG_spearman": float(mp_v),
                    "forward_M_given_PEG": float(fw_v),
                    "forward_share_of_PEG": float(fw_v / p_v),
                    "reverse_PEG_given_M": float(rv_v),
                }
            xc = log10_n - np.mean(log10_n)
            denom = float(np.sum(xc ** 2))
            protgpt3_ladder = {
                "source": str(otp_result_path),
                "n_families": 163,
                "rungs": entries,
                "raw_M_slope_per_decade": float(np.sum(xc * (m_arr - np.mean(m_arr))) / denom),
                "forward_increment_slope_per_decade": float(np.sum(xc * (fwd_arr - np.mean(fwd_arr))) / denom),
                "reverse_increment_slope_per_decade": float(np.sum(xc * (rev_arr - np.mean(rev_arr))) / denom),
            }

    return {
        "cluster_msa_depth_summary": {
            "n_families": len(all_clusters),
            "quartiles": {"q25": cq25, "q50": cq50, "q75": cq75},
            "strata_counts": {k: len(v) for k, v in c_strata.items()},
        },
        "panel_ladders": ladders_out,
        "protgpt3_ladder": protgpt3_ladder,
    }


# ---------------------------------------------------------------------------
# (c) Complete Protein Entropy-Budget Decomposition
# ---------------------------------------------------------------------------


def compute_entropy_budget_decomposition(
    fc_data: dict[str, Any],
    fd_data: dict[str, Any],
    sl_root: Path,
    *,
    n_draws: int = 2000,
    seed: int = 20261120,
) -> dict[str, Any]:
    """(c) Uniform -> Composition -> 1st-order Markov -> Family Profile -> Couplings -> 7B PLM."""
    sites = fc_data["site_records"]

    # Empirical global AA composition entropy across all 4,075 observed heldout contexts
    obs_counts = np.zeros(len(AA20), dtype=np.float64)
    for s in sites:
        for idx in s["obs_indices"]:
            obs_counts[idx] += 1.0
    h_global_comp = shannon_entropy_bits(obs_counts)

    # Per-family 20-AA site-conditional cross-entropy (bits/res)
    def _fc_fam_vec(arm: str) -> np.ndarray:
        _, means, _ = _aggregate_sites_by_family(sites, fc_data["arm_site_bits"][arm])
        return means

    fc_profile = _fc_fam_vec("profile")
    fc_markov = _fc_fam_vec("markov")
    fc_glm7b = _fc_fam_vec("proteinglm-7b-clm")
    fc_pg2xl = _fc_fam_vec("progen2-xlarge")

    # Native sequence cross-entropy & column-null / composition-null deltas from family_distribution
    fd_u = fd_data["unit_order"]
    fd_prof_nat = np.asarray([fd_data["baselines"]["profile"]["native_bits"][u] for u in fd_u])
    fd_mkv_nat = np.asarray([fd_data["baselines"]["markov"]["native_bits"][u] for u in fd_u])
    fd_mkv_col = np.asarray([fd_data["baselines"]["markov"]["column_null_delta_bits"][u] for u in fd_u])

    fd_glm7b_nat = np.asarray([fd_data["arms"]["proteinglm-7b-clm"]["native_bits"][u] for u in fd_u])
    fd_glm7b_col = np.asarray([fd_data["arms"]["proteinglm-7b-clm"]["column_null_delta_bits"][u] for u in fd_u])
    fd_glm7b_comp = np.asarray([fd_data["arms"]["proteinglm-7b-clm"]["composition_null_delta_bits"][u] for u in fd_u])

    fd_pg2xl_nat = np.asarray([fd_data["arms"]["progen2-xlarge"]["native_bits"][u] for u in fd_u])
    fd_pg2xl_col = np.asarray([fd_data["arms"]["progen2-xlarge"]["column_null_delta_bits"][u] for u in fd_u])
    fd_pg2xl_comp = np.asarray([fd_data["arms"]["progen2-xlarge"]["composition_null_delta_bits"][u] for u in fd_u])

    # Baseline strength from statistical_ladder (Profile vs Potts/Couplings vs GEMME)
    bs_doc = load_json_file(sl_root / "full/baseline-strength.json")
    bs_sp = bs_doc["metrics"]["spearman"]

    # Budget ladder on family_conditionals (equal-family, exact apples-to-apples 20-AA bits/res)
    h_unif = UNIFORM_AA20_BITS
    h_mkv = float(np.mean(fc_markov))
    h_prof = float(np.mean(fc_profile))
    h_glm7b = float(np.mean(fc_glm7b))
    h_pg2xl = float(np.mean(fc_pg2xl))

    # Native sequence budget on family_distribution (64 evaluation families)
    nat_mkv = float(np.mean(fd_mkv_nat))
    nat_prof = float(np.mean(fd_prof_nat))
    nat_glm7b = float(np.mean(fd_glm7b_nat))
    nat_pg2xl = float(np.mean(fd_pg2xl_nat))

    # Decomposition shares of uniform 4.3219-bit ceiling (on 20-AA conditionals)
    gain_comp = h_unif - h_global_comp
    gain_markov_over_comp = h_global_comp - h_mkv
    gain_profile_over_markov = h_mkv - h_prof
    gain_profile_total = h_unif - h_prof
    gain_glm7b_over_profile = h_prof - h_glm7b
    gain_pg2xl_over_profile = h_prof - h_pg2xl

    return {
        "uniform_ceiling_bits": h_unif,
        "global_aa_composition_entropy_bits": h_global_comp,
        "family_conditionals_20aa_budget": {
            "levels_equal_family_bits": {
                "uniform": h_unif,
                "global_composition": h_global_comp,
                "markov_1st_order": cluster_bootstrap_mean(fc_markov, n_draws=n_draws, seed=seed),
                "family_profile": cluster_bootstrap_mean(fc_profile, n_draws=n_draws, seed=seed),
                "progen2_xlarge_6.4b": cluster_bootstrap_mean(fc_pg2xl, n_draws=n_draws, seed=seed),
                "proteinglm_7b_clm": cluster_bootstrap_mean(fc_glm7b, n_draws=n_draws, seed=seed),
            },
            "incremental_bits_and_shares_of_uniform": {
                "composition_over_uniform": {
                    "bits": gain_comp,
                    "pct_of_uniform": 100.0 * gain_comp / h_unif,
                },
                "markov_over_composition": {
                    "bits": gain_markov_over_comp,
                    "pct_of_uniform": 100.0 * gain_markov_over_comp / h_unif,
                },
                "profile_over_markov": {
                    "bits": gain_profile_over_markov,
                    "pct_of_uniform": 100.0 * gain_profile_over_markov / h_unif,
                },
                "total_phylogenetic_profile_lookup": {
                    "bits": gain_profile_total,
                    "pct_of_uniform": 100.0 * gain_profile_total / h_unif,
                },
                "proteinglm_7b_beyond_profile": {
                    "bits": gain_glm7b_over_profile,
                    "pct_of_uniform": 100.0 * gain_glm7b_over_profile / h_unif,
                    "paired_ci95_bits": paired_cluster_bootstrap_diff(
                        fc_profile, fc_glm7b, n_draws=n_draws, seed=seed
                    )["ci95"],
                },
                "progen2_xlarge_beyond_profile": {
                    "bits": gain_pg2xl_over_profile,
                    "pct_of_uniform": 100.0 * gain_pg2xl_over_profile / h_unif,
                    "paired_ci95_bits": paired_cluster_bootstrap_diff(
                        fc_profile, fc_pg2xl, n_draws=n_draws, seed=seed
                    )["ci95"],
                },
                "residual_neutral_drift_floor_at_proteinglm_7b": {
                    "bits": h_glm7b,
                    "pct_of_uniform": 100.0 * h_glm7b / h_unif,
                },
                "residual_neutral_drift_floor_at_profile": {
                    "bits": h_prof,
                    "pct_of_uniform": 100.0 * h_prof / h_unif,
                },
            },
        },
        "family_distribution_native_budget": {
            "levels_equal_family_bits": {
                "uniform": h_unif,
                "markov_1st_order_native": cluster_bootstrap_mean(fd_mkv_nat, n_draws=n_draws, seed=seed),
                "family_profile_native": cluster_bootstrap_mean(fd_prof_nat, n_draws=n_draws, seed=seed),
                "progen2_xlarge_native": cluster_bootstrap_mean(fd_pg2xl_nat, n_draws=n_draws, seed=seed),
                "proteinglm_7b_clm_native": cluster_bootstrap_mean(fd_glm7b_nat, n_draws=n_draws, seed=seed),
            },
            "higher_order_column_null_delta_bits": {
                "markov_1st_order": cluster_bootstrap_mean(fd_mkv_col, n_draws=n_draws, seed=seed),
                "progen2_xlarge": cluster_bootstrap_mean(fd_pg2xl_col, n_draws=n_draws, seed=seed),
                "proteinglm_7b_clm": cluster_bootstrap_mean(fd_glm7b_col, n_draws=n_draws, seed=seed),
            },
            "composition_null_delta_bits": {
                "progen2_xlarge": cluster_bootstrap_mean(fd_pg2xl_comp, n_draws=n_draws, seed=seed),
                "proteinglm_7b_clm": cluster_bootstrap_mean(fd_glm7b_comp, n_draws=n_draws, seed=seed),
            },
            "pct_of_uniform": {
                "markov_native_pct": 100.0 * nat_mkv / h_unif,
                "profile_native_pct": 100.0 * nat_prof / h_unif,
                "progen2_xlarge_native_pct": 100.0 * nat_pg2xl / h_unif,
                "proteinglm_7b_native_pct": 100.0 * nat_glm7b / h_unif,
                "proteinglm_7b_column_null_share_pct": 100.0 * float(np.mean(fd_glm7b_col)) / h_unif,
                "progen2_xlarge_column_null_share_pct": 100.0 * float(np.mean(fd_pg2xl_col)) / h_unif,
            },
        },
        "potts_and_gemme_dms_hierarchy": {
            "source": str(sl_root / "full/baseline-strength.json"),
            "spearman_baseline_strength": bs_sp,
        },
    }


# ---------------------------------------------------------------------------
# (d) Matched-Scale Tokenization Natural Experiment (ProtGPT2 vs ProGen2-medium)
#     + R4 Non-Parametric Memory & R6 Post-Training Environment Syntheses
# ---------------------------------------------------------------------------


def compute_tokenization_natural_experiment(
    fc_data: dict[str, Any],
    fd_data: dict[str, Any],
    sl_root: Path,
    readout_dir: Path,
    lanes_root: Path,
    *,
    n_draws: int = 2000,
    seed: int = 20261120,
) -> dict[str, Any]:
    """(d) Full paired bootstrap comparison of ProtGPT2 (738M BPE 50k) vs ProGen2-medium (764M AA 32)."""
    arm_bpe = "protgpt2"
    arm_aa = "progen2-medium"
    fd_u = fd_data["unit_order"]

    # 1. family_distribution (64 families)
    fd_metrics = {}
    for key in (
        "native_bits",
        "column_null_delta_bits",
        "composition_null_delta_bits",
        "tokens_per_residue",
        "bits_per_token",
    ):
        a = np.asarray([fd_data["arms"][arm_bpe][key][u] for u in fd_u], dtype=np.float64)
        b = np.asarray([fd_data["arms"][arm_aa][key][u] for u in fd_u], dtype=np.float64)
        fd_metrics[key] = paired_cluster_bootstrap_diff(a, b, n_draws=n_draws, seed=seed)

    # 2. family_conditionals by conservation tier (64 families)
    fc_metrics = {}
    for tier in ("overall", "conserved", "moderate", "variable"):
        t_arg = None if tier == "overall" else tier
        uids_a, fam_a, counts = _aggregate_sites_by_family(
            fc_data["site_records"], fc_data["arm_site_bits"][arm_bpe], tier_filter=t_arg
        )
        uids_b, fam_b, _ = _aggregate_sites_by_family(
            fc_data["site_records"], fc_data["arm_site_bits"][arm_aa], tier_filter=t_arg
        )
        assert uids_a == uids_b
        fc_metrics[tier] = {
            "site_weighted_diff_bpe_minus_aa": paired_cluster_bootstrap_diff(
                fam_a, fam_b, weights=counts, n_draws=n_draws, seed=seed
            ),
            "equal_family_diff_bpe_minus_aa": paired_cluster_bootstrap_diff(
                fam_a, fam_b, weights=None, n_draws=n_draws, seed=seed
            ),
        }

    # 3. R1 DMS ladder (163 families, paired bootstrap from assay-metrics.json)
    assay_rows = load_json_file(sl_root / "full/assay-metrics.json")
    by_mca: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in assay_rows:
        if row["model"] not in (arm_bpe, arm_aa):
            continue
        for key in ("M", "PEG", "MPEG"):
            by_mca[(row["model"], str(row["cluster"]), row["assay"])][key].append(float(row["spearman"][key]))

    fam_r1: dict[str, dict[str, dict[str, float]]] = {arm_bpe: {}, arm_aa: {}}
    by_mc_tmp: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (m, cl, _a), kmap in by_mca.items():
        for k, vs in kmap.items():
            by_mc_tmp[(m, cl)][k].append(float(np.mean(vs)))
    for (m, cl), kmap in by_mc_tmp.items():
        fam_r1[m][cl] = {k: float(np.mean(vs)) for k, vs in kmap.items()}

    shared_r1 = sorted(set(fam_r1[arm_bpe]) & set(fam_r1[arm_aa]))
    r1_m_diff = paired_cluster_bootstrap_diff(
        [fam_r1[arm_bpe][c]["M"] for c in shared_r1],
        [fam_r1[arm_aa][c]["M"] for c in shared_r1],
        n_draws=n_draws,
        seed=seed,
    )
    r1_fwd_diff = paired_cluster_bootstrap_diff(
        [fam_r1[arm_bpe][c]["MPEG"] - fam_r1[arm_bpe][c]["PEG"] for c in shared_r1],
        [fam_r1[arm_aa][c]["MPEG"] - fam_r1[arm_aa][c]["PEG"] for c in shared_r1],
        n_draws=n_draws,
        seed=seed,
    )
    r1_rev_diff = paired_cluster_bootstrap_diff(
        [fam_r1[arm_bpe][c]["MPEG"] - fam_r1[arm_bpe][c]["M"] for c in shared_r1],
        [fam_r1[arm_aa][c]["MPEG"] - fam_r1[arm_aa][c]["M"] for c in shared_r1],
        n_draws=n_draws,
        seed=seed,
    )

    # Readout panel summary (R vs raw M, B, B+R from panel_increments.tsv + tables.md)
    readout_comparison = _extract_readout_pair(readout_dir, arm_bpe, arm_aa)

    # 4. R2 abundance (ranking Spearman, higher better) & stability (calibrated error RMSE, lower better)
    r2_lanes: dict[str, Any] = {}
    lane_specs = [
        ("abundance_ranking_spearman", "abundance", "ranking", "higher_is_better"),
        ("stability_calibrated_error_rmse", "stability", "calibrated_error", "lower_is_better"),
    ]
    for label, lane, stratum, direction in lane_specs:
        lane_out: dict[str, Any] = {"direction": direction, "rungs": {}}
        for rung in ("nuisance", "nuisance+profile", "nuisance+profile+couplings+gemme"):
            p_bpe = lanes_root / lane / "cells" / stratum / "likelihood/complete_case" / rung / f"{arm_bpe}.json"
            p_aa = lanes_root / lane / "cells" / stratum / "likelihood/complete_case" / rung / f"{arm_aa}.json"
            if not (p_bpe.is_file() and p_aa.is_file()):
                continue
            c_bpe = load_json_file(p_bpe)
            c_aa = load_json_file(p_aa)
            if c_bpe["groups"] != c_aa["groups"]:
                raise ValueError(f"Mismatched groups between {arm_bpe} and {arm_aa} in {lane}/{rung}")
            m_bpe = np.asarray(c_bpe["values"]["M"], dtype=np.float64)
            m_aa = np.asarray(c_aa["values"]["M"], dtype=np.float64)
            uc_bpe = np.asarray(c_bpe["values"]["U_C"], dtype=np.float64)
            ucm_bpe = np.asarray(c_bpe["values"]["U_C_M"], dtype=np.float64)
            uc_aa = np.asarray(c_aa["values"]["U_C"], dtype=np.float64)
            ucm_aa = np.asarray(c_aa["values"]["U_C_M"], dtype=np.float64)

            if direction == "higher_is_better":
                inc_bpe = ucm_bpe - uc_bpe
                inc_aa = ucm_aa - uc_aa
            else:
                # For error (lower better), improvement is U_C - U_C_M
                inc_bpe = uc_bpe - ucm_bpe
                inc_aa = uc_aa - ucm_aa

            lane_out["rungs"][rung] = {
                "n_groups": len(c_bpe["groups"]),
                "raw_M_diff_bpe_minus_aa": paired_cluster_bootstrap_diff(
                    m_bpe, m_aa, n_draws=n_draws, seed=seed
                ),
                "forward_increment_diff_bpe_minus_aa": paired_cluster_bootstrap_diff(
                    inc_bpe, inc_aa, n_draws=n_draws, seed=seed
                ),
                "protgpt2_forward_increment": cluster_bootstrap_mean(
                    inc_bpe, n_draws=n_draws, seed=seed
                ),
                "progen2_medium_forward_increment": cluster_bootstrap_mean(
                    inc_aa, n_draws=n_draws, seed=seed
                ),
            }
        r2_lanes[label] = lane_out

    return {
        "arms": {
            "bpe": {"name": arm_bpe, "params": 738e6, "vocab_size": 50257, "tokenization": "protein_bpe"},
            "aa": {"name": arm_aa, "params": 764e6, "vocab_size": 32, "tokenization": "single_aa"},
            "vocab_ratio_bpe_over_aa": 50257.0 / 32.0,
            "param_ratio_bpe_over_aa": 738.0 / 764.0,
        },
        "family_distribution_paired_bpe_minus_aa": fd_metrics,
        "family_conditionals_paired_bpe_minus_aa": fc_metrics,
        "r1_dms_paired_bpe_minus_aa": {
            "n_families": len(shared_r1),
            "raw_M_spearman": r1_m_diff,
            "forward_M_given_PEG_spearman": r1_fwd_diff,
            "reverse_PEG_given_M_spearman": r1_rev_diff,
        },
        "readout_panel_summary": readout_comparison,
        "r2_abundance_and_stability": r2_lanes,
    }


def _extract_readout_pair(readout_dir: Path, arm_a: str, arm_b: str) -> dict[str, Any]:
    """Read anchor201 readout increments for two arms from panel_increments.tsv."""
    tsv_path = readout_dir / "panel_increments.tsv"
    out: dict[str, dict[str, Any]] = {arm_a: {}, arm_b: {}}
    if not tsv_path.is_file():
        return out
    with tsv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["panel"] != "anchor201" or row["seed"] != "20260923":
                continue
            if row["arm"] in (arm_a, arm_b):
                out[row["arm"]][row["metric"]] = {
                    "point": float(row["point"]),
                    "ci95": [float(row["lower"]), float(row["upper"])],
                    "interval_state": row["interval_state"],
                }
    return out


def extract_r4_and_r6_memory_anchors(
    r4_paths: Sequence[Path],
    r6_endpoints_path: Path,
    r6_five_mer_path: Path,
) -> dict[str, Any]:
    """Summarize R4 non-parametric attention context and R6 post-training environment SNR."""
    r4_arms: dict[str, Any] = {}
    for path in r4_paths:
        if not path.is_file():
            continue
        doc = load_json_file(path)
        for arm, rec in doc.get("arms", {}).items():
            verdict = rec.get("verdict") or {}
            bands = rec.get("bands", {})
            lt30 = bands.get("id_lt_30", {}).get("retained", {})
            id50 = bands.get("id_50_70", {}).get("retained", {})
            r4_arms[arm] = {
                "modality": rec.get("modality"),
                "outcome": verdict.get("outcome"),
                "auroc_id_lt_30": lt30.get("auroc", {}).get("mean"),
                "auroc_id_lt_30_ci95": lt30.get("auroc", {}).get("ci95"),
                "auroc_id_50_70": id50.get("auroc", {}).get("mean"),
                "auroc_id_50_70_ci95": id50.get("auroc", {}).get("ci95"),
            }

    r6_summary: dict[str, Any] = {}
    if r6_endpoints_path.is_file():
        r6 = load_json_file(r6_endpoints_path)
        band_cols = r6.get("primary", {}).get("bands", {}).get("columns", {})
        lt30_contrasts = {}
        for col_key, col_val in sorted(band_cols.items()):
            if "lt30" in col_key or "30_to_70" in col_key:
                lt30_contrasts[col_key] = {
                    "base_mean": col_val.get("base_mean"),
                    "tuned_mean": col_val.get("tuned_mean"),
                    "diff_mean": col_val.get("diff_mean") if "diff_mean" in col_val else col_val.get("point"),
                    "ci95": col_val.get("ci95"),
                    "status": col_val.get("status"),
                }
        r6_summary["primary_band_contrasts"] = lt30_contrasts

    if r6_five_mer_path.is_file():
        r6f = load_json_file(r6_five_mer_path)
        r6_summary["five_mer_filter_length_matched_gain"] = r6f.get("length_matched_gain", [])

    return {
        "r4_non_parametric_context_verdicts": r4_arms,
        "r6_post_training_environment_snr": r6_summary,
    }


# ---------------------------------------------------------------------------
# (e) Parametric Capacity Budget Model (Shannon Excess Compression)
# ---------------------------------------------------------------------------


def compute_parametric_capacity_budget(
    *,
    profile_gain_bits_per_res: float,
    mean_domain_length: float = 250.0,
) -> dict[str, Any]:
    """(e) Explicit Shannon excess-compression budget curve across 8M-100B parameters.

    Explicitly separates Shannon excess-compression capacity (2.0-3.6 bits/param,
    Allen-Zhu & Li 2024; Morris et al. 2025) from linear-algebraic outer-product /
    whitened scalar capacity (M = d_k, M = 2 d_k).
    """
    cluster_grid = [1e6, 3e6, 1e7, 3e7, 6e7]
    length_grid = [175.0, float(mean_domain_length), 300.0]
    param_grid = [
        ("8M", 8e6),
        ("35M", 35e6),
        ("112M", 112e6),
        ("151M", 151e6),
        ("350M", 350e6),
        ("650M", 650e6),
        ("764M", 764e6),
        ("1.3B", 1.3e9),
        ("1.4B", 1.4e9),
        ("2.7B", 2.7e9),
        ("3.0B", 3.0e9),
        ("6.4B", 6.4e9),
        ("7.0B", 7.0e9),
        ("10B", 10.0e9),
        ("15B", 15.0e9),
        ("30B", 30.0e9),
        ("100B", 100.0e9),
    ]
    bits_per_param_bounds = {"allen_zhu_factual_low": 2.0, "morris_ceiling_high": 3.6}

    lookup_budgets: list[dict[str, Any]] = []
    for k_clusters in cluster_grid:
        for avg_len in length_grid:
            total_bits = float(k_clusters * avg_len * profile_gain_bits_per_res)
            lookup_budgets.append(
                {
                    "n_clusters": int(k_clusters),
                    "mean_length_aa": float(avg_len),
                    "profile_gain_bits_per_res": float(profile_gain_bits_per_res),
                    "total_lookup_bits": total_bits,
                    "total_lookup_gbits": total_bits / 1e9,
                    "critical_params_at_2_0_bpp": total_bits / 2.0,
                    "critical_params_at_3_6_bpp": total_bits / 3.6,
                }
            )

    # Reference scenarios: conservative 10^7 sub-families, full UniRef50 (6x10^7), UniRef90 (1.9x10^8)
    ref_10m_bits = float(1e7 * mean_domain_length * profile_gain_bits_per_res)
    ref_60m_bits = float(6e7 * mean_domain_length * profile_gain_bits_per_res)
    ref_190m_bits = float(1.9e8 * mean_domain_length * profile_gain_bits_per_res)

    def _regime(cap_lo: float, cap_hi: float, target: float) -> str:
        if cap_hi < target:
            return "capacity_starved"
        if cap_lo < target:
            return "transition_band"
        return "capacity_surplus"

    model_curve: list[dict[str, Any]] = []
    for label, n_params in param_grid:
        cap_low_bits = float(n_params * 2.0)
        cap_high_bits = float(n_params * 3.6)
        model_curve.append(
            {
                "scale_label": label,
                "params": float(n_params),
                "capacity_gbits_2_0_bpp": cap_low_bits / 1e9,
                "capacity_gbits_3_6_bpp": cap_high_bits / 1e9,
                "coverage_10m_clusters_at_2_0_bpp": cap_low_bits / ref_10m_bits,
                "coverage_10m_clusters_at_3_6_bpp": cap_high_bits / ref_10m_bits,
                "coverage_60m_uniref50_at_2_0_bpp": cap_low_bits / ref_60m_bits,
                "coverage_60m_uniref50_at_3_6_bpp": cap_high_bits / ref_60m_bits,
                "coverage_190m_uniref90_at_2_0_bpp": cap_low_bits / ref_190m_bits,
                "coverage_190m_uniref90_at_3_6_bpp": cap_high_bits / ref_190m_bits,
                "regime_vs_10m_clusters": _regime(cap_low_bits, cap_high_bits, ref_10m_bits),
                "regime_vs_60m_uniref50": _regime(cap_low_bits, cap_high_bits, ref_60m_bits),
                "regime_vs_190m_uniref90": _regime(cap_low_bits, cap_high_bits, ref_190m_bits),
            }
        )

    return {
        "theoretical_caveats": {
            "shannon_vs_linear_algebraic": (
                "Linear-algebraic outer-product / whitened memory (M = d_k and M = 2 d_k in "
                "W in R^{d_v x d_k}) bounds real scalar associations at SNR=1, NOT Shannon "
                "bits/parameter. The parametric capacity budget below uses Shannon excess-compression "
                "bounds (2.0 bits/param factual capacity, Allen-Zhu & Li; 3.6 bits/param memorization "
                "ceiling, Morris et al.)."
            ),
            "key_value_alignment": (
                "Isotropic random-key interference neglects constructive vs destructive key-value "
                "alignment; generalization requires key similarity to track latent compositional "
                "equivalence rather than colliding on shared local secondary-structure motifs."
            ),
        },
        "empirical_profile_gain_bits_per_res": float(profile_gain_bits_per_res),
        "reference_scenarios_gbits": {
            "conservative_10m_x_250aa": ref_10m_bits / 1e9,
            "full_uniref50_60m_x_250aa": ref_60m_bits / 1e9,
            "uniref90_190m_x_250aa": ref_190m_bits / 1e9,
        },
        "bits_per_param_bounds": bits_per_param_bounds,
        "cluster_lookup_budgets": lookup_budgets,
        "model_scale_curve": model_curve,
    }


def compute_family_level_general_plm_scaling(
    fc_data: dict[str, Any],
    general_fc_by_arm: Mapping[str, Mapping[str, float]],
    semantics_by_arm: Mapping[str, str],
    msa_support: Mapping[str, dict[str, Any]],
    *,
    n_draws: int = 2000,
    seed: int = 20261120,
) -> dict[str, Any]:
    """Compute unified 64-family scaling slopes across ALL ladders (original causal + V2 general PLMs).

    Stratifies the 64 evaluation families by:
      1. Family-mean training-profile entropy H_prof (quartiles Q1_most_conserved .. Q4_most_variable
         and conserved-family < 1.8b vs moderate 1.8-2.6b vs variable-family >= 2.6b).
      2. Family MSA evolutionary depth N_eff/L (quartiles Q1_lowest .. Q4_highest, <5 vs >=5,
         and below vs above median).
    """
    unit_order = fc_data["unit_order"]
    unit_to_wt = fc_data["unit_to_wt"]
    sites = fc_data["site_records"]

    # Merge original 33 arms (+ profile/markov + esm3-1.4b) with V2 general-PLM family-level arms
    family_bits: dict[str, dict[str, float]] = {}
    arm_semantics: dict[str, str] = {}
    for arm, arr in fc_data["arm_site_bits"].items():
        uids, means, _ = _aggregate_sites_by_family(sites, arr, tier_filter=None)
        family_bits[arm] = dict(zip(uids, means.tolist()))
        arm_semantics[arm] = semantics_by_arm.get(
            arm,
            "baseline" if arm in ("profile", "markov") else "canonical_20_string_causal_conditional",
        )
    for arm, umap in general_fc_by_arm.items():
        if arm not in family_bits:
            family_bits[arm] = {u: float(umap[u]) for u in unit_order}
            arm_semantics[arm] = semantics_by_arm.get(arm, "general_plm_family_conditional")

    # Compute per-family mean H_prof and profile cross-entropy
    h_by_fam: dict[str, list[float]] = defaultdict(list)
    for s in sites:
        h_by_fam[s["unit_id"]].append(float(s["h_prof_bits"]))
    fam_h_prof = np.asarray([float(np.mean(h_by_fam[u])) for u in unit_order], dtype=np.float64)
    hq25, hq50, hq75 = [float(x) for x in np.quantile(fam_h_prof, [0.25, 0.50, 0.75])]

    depths = np.asarray(
        [msa_support[unit_to_wt[u]]["effective_depth_per_site"] for u in unit_order],
        dtype=np.float64,
    )
    dq25, dq50, dq75 = [float(x) for x in np.quantile(depths, [0.25, 0.50, 0.75])]

    conservation_family_strata: dict[str, list[str]] = {
        "all": list(unit_order),
        "Q1_most_conserved": [u for u, h in zip(unit_order, fam_h_prof) if h <= hq25],
        "Q2_conserved_mid": [u for u, h in zip(unit_order, fam_h_prof) if hq25 < h <= hq50],
        "Q3_variable_mid": [u for u, h in zip(unit_order, fam_h_prof) if hq50 < h <= hq75],
        "Q4_most_variable": [u for u, h in zip(unit_order, fam_h_prof) if h > hq75],
        "conserved_fam_lt_1_8b": [u for u, h in zip(unit_order, fam_h_prof) if h < 1.8],
        "moderate_fam_1_8_to_2_6b": [u for u, h in zip(unit_order, fam_h_prof) if 1.8 <= h < 2.6],
        "variable_fam_ge_2_6b": [u for u, h in zip(unit_order, fam_h_prof) if h >= 2.6],
    }
    density_family_strata: dict[str, list[str]] = {
        "all": list(unit_order),
        "neff_lt_5": [u for u, d in zip(unit_order, depths) if d < 5.0],
        "neff_ge_5": [u for u, d in zip(unit_order, depths) if d >= 5.0],
        "neff_below_median": [u for u, d in zip(unit_order, depths) if d < dq50],
        "neff_ge_median": [u for u, d in zip(unit_order, depths) if d >= dq50],
        "Q1_lowest": [u for u, d in zip(unit_order, depths) if d <= dq25],
        "Q2": [u for u, d in zip(unit_order, depths) if dq25 < d <= dq50],
        "Q3": [u for u, d in zip(unit_order, depths) if dq50 < d <= dq75],
        "Q4_highest": [u for u, d in zip(unit_order, depths) if d > dq75],
    }

    prof_map = family_bits["profile"]
    per_arm_levels: dict[str, Any] = {}
    for arm in sorted(family_bits.keys()):
        if arm in ("profile", "markov"):
            continue
        amap = family_bits[arm]
        cons_out: dict[str, Any] = {}
        for sname, suids in conservation_family_strata.items():
            a_v = [amap[u] for u in suids]
            p_v = [prof_map[u] for u in suids]
            cons_out[sname] = {
                "loss_bits": cluster_bootstrap_mean(a_v, n_draws=n_draws, seed=seed),
                "minus_profile_bits": paired_cluster_bootstrap_diff(
                    a_v, p_v, n_draws=n_draws, seed=seed
                ),
            }
        per_arm_levels[arm] = {
            "params": MODEL_PARAM_COUNTS.get(arm),
            "modality": MODEL_MODALITY.get(arm, "unknown"),
            "conditional_semantics": arm_semantics.get(arm),
            "by_family_conservation": cons_out,
        }

    ladders_out: dict[str, Any] = {}
    for lname, rungs in SCALING_LADDERS.items():
        present = [r for r in rungs if r in family_bits and r in MODEL_PARAM_COUNTS]
        if len(present) < 2:
            continue
        log10_n = np.asarray([math.log10(MODEL_PARAM_COUNTS[r]) for r in present], dtype=np.float64)

        by_cons: dict[str, Any] = {}
        for sname, suids in conservation_family_strata.items():
            if len(suids) < 2:
                continue
            mat = np.asarray([[family_bits[r][u] for u in suids] for r in present], dtype=np.float64)
            by_cons[sname] = cluster_bootstrap_scaling_slope(log10_n, mat, n_draws=n_draws, seed=seed)

        by_dens: dict[str, Any] = {}
        for sname, suids in density_family_strata.items():
            if len(suids) < 2:
                continue
            mat = np.asarray([[family_bits[r][u] for u in suids] for r in present], dtype=np.float64)
            by_dens[sname] = cluster_bootstrap_scaling_slope(log10_n, mat, n_draws=n_draws, seed=seed)

        ladders_out[lname] = {
            "rungs": present,
            "params": [MODEL_PARAM_COUNTS[r] for r in present],
            "span_ratio": float(MODEL_PARAM_COUNTS[present[-1]] / MODEL_PARAM_COUNTS[present[0]]),
            "conditional_semantics": sorted({arm_semantics[r] for r in present}),
            "by_family_conservation": by_cons,
            "by_msa_density": by_dens,
        }

    return {
        "n_arms_total": len(per_arm_levels),
        "family_conservation_summary": {
            "metric": "mean training-profile Shannon entropy H_prof (bits) across the 8 sites of each family",
            "quartiles_bits": {"q25": hq25, "q50": hq50, "q75": hq75},
            "strata_counts": {k: len(v) for k, v in conservation_family_strata.items()},
        },
        "family_msa_depth_summary": {
            "quartiles": {"q25": dq25, "q50": dq50, "q75": dq75},
            "strata_counts": {k: len(v) for k, v in density_family_strata.items()},
        },
        "per_arm_family_conservation": per_arm_levels,
        "ladders": ladders_out,
    }


def run_b1_reanalysis(
    repo_root: Path,
    *,
    extra_fc_score_dirs: Sequence[Path] = (),
    extra_fd_score_dirs: Sequence[Path] = (),
    general_plm_dirs: Sequence[Path] = (),
    n_draws: int = 2000,
    seed: int = 20261120,
) -> dict[str, Any]:
    """Execute the complete B1 zero-GPU re-analysis and return a JSON-serializable report."""
    fc_root = repo_root / "results/extensions/family_conditionals_20261107"
    fd_root = repo_root / "results/extensions/family_distribution_20261106"
    sl_root = repo_root / "results/extensions/statistical_ladder_20261106"
    readout_dir = repo_root / "results/R2/readout_panel_summary_20260923"
    lanes_root = repo_root / "results/extensions/explicit_statistics_lanes_20261110"
    otp_path = repo_root / "results/extensions/objective_tuned_prediction_20261110/record/result.json"

    msa_support = load_msa_support(
        sl_root / "inputs/support.json",
        verify_weights_dir=sl_root / "inputs/msa",
    )
    fc_data = extract_family_conditionals_records(fc_root, extra_score_dirs=extra_fc_score_dirs)
    fd_data = extract_family_distribution_records(fd_root, extra_score_dirs=extra_fd_score_dirs)

    # Auto-ingest nested V2 general-PLM R1 conditionals & family_distribution artifacts if provided
    general_fc_by_arm, extra_site_bits, semantics_by_arm = discover_general_plm_family_conditionals(
        general_plm_dirs, fc_data["unit_order"], fc_data["site_records"]
    )
    for arm, site_arr in extra_site_bits.items():
        if arm not in fc_data["arm_site_bits"]:
            fc_data["arm_site_bits"][arm] = site_arr
    fc_data["scored_arms"] = sorted(a for a in fc_data["arm_site_bits"] if a not in ("profile", "markov"))

    general_fd_by_arm = discover_general_plm_family_distribution(general_plm_dirs, fd_data["unit_order"])
    for arm, fd_rec in general_fd_by_arm.items():
        if arm not in fd_data["arms"]:
            fd_data["arms"][arm] = fd_rec

    part_a = compute_site_conservation_stratification(fc_data, n_draws=n_draws, seed=seed)
    part_b = compute_density_and_conservation_scaling(
        fc_data,
        fd_data,
        msa_support,
        sl_root,
        otp_result_path=otp_path if otp_path.is_file() else None,
        n_draws=n_draws,
        seed=seed,
    )
    part_b_general = compute_family_level_general_plm_scaling(
        fc_data,
        general_fc_by_arm,
        semantics_by_arm,
        msa_support,
        n_draws=n_draws,
        seed=seed,
    )
    part_c = compute_entropy_budget_decomposition(fc_data, fd_data, sl_root, n_draws=n_draws, seed=seed)
    part_d = compute_tokenization_natural_experiment(
        fc_data, fd_data, sl_root, readout_dir, lanes_root, n_draws=n_draws, seed=seed
    )
    r4_r6 = extract_r4_and_r6_memory_anchors(
        r4_paths=[
            repo_root / "results/R5/context_homologue_20260826/context_homologue.json",
            repo_root / "results/R5/context_homologue_expansion_20260919/context_homologue.json",
            repo_root / "results/R5/context_homologue_progen3_20260924/context_homologue.json",
        ],
        r6_endpoints_path=repo_root / "results/extensions/objective_tuned_endpoints_20261112/objective_tuned_endpoints.json",
        r6_five_mer_path=repo_root / "results/extensions/objective_tuned_five_mer_rule_20261114/five_mer_rule.json",
    )
    profile_gain = float(
        part_c["family_conditionals_20aa_budget"]["incremental_bits_and_shares_of_uniform"][
            "total_phylogenetic_profile_lookup"
        ]["bits"]
    )
    part_e = compute_parametric_capacity_budget(
        profile_gain_bits_per_res=profile_gain,
        mean_domain_length=250.0,
    )

    return {
        "schema": "memory_theory_b1_capacity_starvation_v1",
        "bootstrap": {"method": "paired_family_cluster_bootstrap", "n_draws": int(n_draws), "seed": int(seed)},
        "scored_arms_discovered": {
            "family_conditionals_site_level": fc_data["scored_arms"],
            "family_conditionals_family_level_all": sorted(part_b_general["per_arm_family_conservation"].keys()),
            "general_plm_family_conditionals": sorted(general_fc_by_arm.keys()),
            "family_distribution": sorted(fd_data["arms"].keys()),
        },
        "a_site_conservation_stratification": part_a,
        "b_family_evolutionary_density_stratification": part_b,
        "b_general_plm_family_level_scaling": part_b_general,
        "c_protein_entropy_budget_decomposition": part_c,
        "d_matched_scale_tokenization_experiment": part_d,
        "d_aux_r4_and_r6_memory_anchors": r4_r6,
        "e_parametric_capacity_budget_model": part_e,
    }
