"""Stage-2 supply-demand campaign analysis: peaks, bit allocation, and master-equation fit."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .spec import LOG2_20, assert_no_forbidden_infra_strings
from .training import RUN_RECEIPT_SCHEMA

ANALYSIS_SCHEMA: str = "plm_supply_demand_analysis_v1"

NATURAL_ARM_ORDER: tuple[str, ...] = (
    "M64",
    "M256",
    "M1024",
    "M1802_all_cath",
    "M4096",
    "M_max_singleton",
)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _log_linear_first_crossing(
    epochs: Sequence[float],
    values: Sequence[float],
    threshold: float,
) -> float | None:
    """First exposure e > 0 where values cross >= threshold via log-linear interpolation."""
    pts = [
        (float(e), float(v))
        for e, v in zip(epochs, values)
        if e is not None and v is not None and math.isfinite(float(e)) and math.isfinite(float(v)) and float(e) > 0.0
    ]
    if not pts:
        return None
    pts.sort(key=lambda kv: kv[0])
    thr = float(threshold)
    for idx, (e1, v1) in enumerate(pts):
        if v1 >= thr:
            if idx == 0:
                return e1
            e0, v0 = pts[idx - 1]
            if v1 <= v0 + 1e-15:
                return e1
            frac = min(max((thr - v0) / (v1 - v0), 0.0), 1.0)
            return float(math.exp(math.log(e0) + frac * (math.log(e1) - math.log(e0))))
    return None


def _safe_corr(xs: Sequence[float], ys: Sequence[float]) -> dict[str, Any]:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    n = len(pairs)
    if n < 3:
        return {"n": n, "spearman_r": None, "spearman_p": None, "pearson_r": None, "pearson_p": None}
    xa = np.asarray([p[0] for p in pairs], dtype=np.float64)
    ya = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(np.std(xa)) <= 1e-15 or float(np.std(ya)) <= 1e-15:
        return {"n": n, "spearman_r": None, "spearman_p": None, "pearson_r": None, "pearson_p": None}
    sr = spearmanr(xa, ya)
    pr = pearsonr(xa, ya)
    return {
        "n": n,
        "spearman_r": _safe_float(sr.statistic),
        "spearman_p": _safe_float(sr.pvalue),
        "pearson_r": _safe_float(pr.statistic),
        "pearson_p": _safe_float(pr.pvalue),
    }


def _canonical_objective(raw: str) -> str:
    s = str(raw).strip().lower()
    return "oa_uniform" if s == "mlm" else s


def load_stage2_run_receipts(
    runs_dir: Path | str,
    *,
    include_in_progress: bool = True,
) -> list[dict[str, Any]]:
    """Load all Stage-2 run JSON receipts (skipping .sha256 sidecars) in deterministic order."""
    root = Path(runs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Stage-2 runs directory does not exist: {root}")
    docs: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(doc, dict) or doc.get("schema") != RUN_RECEIPT_SCHEMA:
            continue
        if (not include_in_progress) and doc.get("status") != "complete":
            continue
        docs.append(doc)
    return docs


def load_stage1_capacity_lookup(stage1_path: Path | str | None) -> dict[str, dict[str, Any]]:
    """Load Stage-1 calibrated c_eff(O, e) & e_crit(O) from a summary JSON or runs/ directory."""
    if stage1_path is None:
        return {}
    p = Path(stage1_path)
    if not p.exists():
        raise FileNotFoundError(f"Stage-1 calibration path not found: {p}")

    out: dict[str, dict[str, Any]] = {}
    if p.is_file():
        doc = json.loads(p.read_text(encoding="utf-8"))
        curves = doc.get("exposure_discount_curves", {})
        for raw_obj, entry in curves.items():
            obj = _canonical_objective(raw_obj)
            pts: list[tuple[float, float]] = []
            for row in entry.get("pooled_by_exposure", []):
                e = _safe_float(row.get("epoch"))
                c = _safe_float(
                    row.get("c_eff_empirical_mean", row.get("c_eff_at_2x_mean"))
                )
                if e is not None and e > 0.0 and c is not None:
                    pts.append((e, c))
            pts.sort(key=lambda kv: kv[0])
            ec_np = entry.get("pooled_e_crit_nonparametric") or {}
            e_crit = (
                _safe_float(ec_np.get("e_crit_c_eff_0p02"))
                or _safe_float(ec_np.get("e_crit_c_eff_0p05"))
                or _safe_float(ec_np.get("e_crit_delta_bpt_0p05"))
            )
            out[obj] = {
                "source": "stage1_summary_json",
                "curve_points": pts,
                "e_crit": e_crit,
                "c_inf_ref": _safe_float(ec_np.get("c_inf_reference_empirical")),
            }
        return out

    # Directory of Stage-1 run JSONs: aggregate median c_eff(e) across capacity-limited runs (r >= 1.0)
    by_obj_ep: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rp in sorted(p.glob("*.json")):
        try:
            doc = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(doc, dict) or "exposures" not in doc:
            continue
        raw_par = doc.get("paradigm") or (doc.get("run_spec") or {}).get("paradigm")
        mult = _safe_float(doc.get("multiplier") or (doc.get("run_spec") or {}).get("multiplier"))
        if not raw_par or mult is None or mult < 1.0 - 1e-9:
            continue
        obj = _canonical_objective(str(raw_par))
        for rec in doc.get("exposures", []):
            ep = rec.get("epoch")
            c_val = _safe_float(
                rec.get("pseudo_bits_per_non_emb_param")
                if obj == "mlm_15"
                else rec.get("bits_per_non_emb_param")
            )
            if ep is not None and int(ep) > 0 and c_val is not None:
                by_obj_ep[obj][int(ep)].append(c_val)

    for obj, ep_map in by_obj_ep.items():
        pts = sorted((float(e), float(np.median(vals))) for e, vals in ep_map.items() if vals)
        e_arr = [pt[0] for pt in pts]
        c_arr = [pt[1] for pt in pts]
        out[obj] = {
            "source": "stage1_runs_dir",
            "curve_points": pts,
            "e_crit": _log_linear_first_crossing(e_arr, c_arr, 0.02),
            "c_inf_ref": max(c_arr) if c_arr else None,
        }
    return out


def interpolate_stage1_c_eff(
    stage1_lookup: Mapping[str, Mapping[str, Any]],
    objective: str,
    epoch: float,
) -> float | None:
    """Log-linearly interpolate Stage-1 c_eff(O, e) in bits/non-emb param."""
    entry = stage1_lookup.get(_canonical_objective(objective))
    if not entry:
        return None
    pts = entry.get("curve_points") or []
    if not pts or epoch <= 0.0:
        return None
    if epoch <= pts[0][0]:
        return float(pts[0][1])
    if epoch >= pts[-1][0]:
        return float(pts[-1][1])
    log_e = math.log(float(epoch))
    for i in range(1, len(pts)):
        e0, c0 = pts[i - 1]
        e1, c1 = pts[i]
        if e0 <= epoch <= e1:
            if e1 <= e0:
                return float(c1)
            frac = (log_e - math.log(e0)) / (math.log(e1) - math.log(e0))
            return float(c0 + frac * (c1 - c0))
    return float(pts[-1][1])


def _extract_milestone_point(m: Mapping[str, Any]) -> dict[str, Any]:
    tt_all = m.get("three_tier_bits") or {}
    tt = tt_all.get("native") or {}
    tt_loo = tt_all.get("loo") or {}
    ms = m.get("megascale") or {}
    ms_bs = ms.get("by_stratum") or {}
    pg = m.get("proteingym_milestone") or {}
    pg_full = m.get("proteingym_full") or {}
    ct = m.get("contacts") or {}
    ct_lr = ct.get("long_range_sep_ge_24") or {}
    ct_s6 = ct.get("sep_ge_6") or {}

    return {
        "epoch": int(m.get("epoch", 0)),
        "global_step": int(m.get("global_step", 0)),
        "tokens_seen": int(m.get("tokens_seen", 0)),
        "lr": _safe_float(m.get("lr")),
        "H_train": _safe_float(tt.get("H_train")),
        "H_unseen_homolog": _safe_float(tt.get("H_unseen_homolog")),
        "H_unseen_fold": _safe_float(tt.get("H_unseen_fold")),
        "delta_h_univ": _safe_float(tt.get("delta_h_univ")),
        "delta_h_spec": _safe_float(tt.get("delta_h_spec")),
        "delta_h_res": _safe_float(tt.get("delta_h_res")),
        "delta_h_total": _safe_float(tt.get("delta_h_total")),
        "loo_delta_h_univ": _safe_float(tt_loo.get("delta_h_univ")),
        "loo_delta_h_spec": _safe_float(tt_loo.get("delta_h_spec")),
        "loo_delta_h_res": _safe_float(tt_loo.get("delta_h_res")),
        "megascale_overall": _safe_float(ms.get("overall_spearman_mean")),
        "megascale_de_novo": _safe_float((ms_bs.get("de_novo_design") or {}).get("spearman_mean")),
        "megascale_unseen_fold": _safe_float((ms_bs.get("unseen_fold_natural") or {}).get("spearman_mean")),
        "megascale_seen_pool": _safe_float((ms_bs.get("seen_pool_natural") or {}).get("spearman_mean")),
        "proteingym_assay": _safe_float(pg.get("assay_mean_spearman")),
        "proteingym_cluster": _safe_float(pg.get("cluster_mean_spearman")),
        "proteingym_full_assay": _safe_float(pg_full.get("assay_mean_spearman")),
        "proteingym_full_cluster": _safe_float(pg_full.get("cluster_mean_spearman")),
        "contacts_lr_p_at_L": _safe_float(ct_lr.get("p_at_L")),
        "contacts_lr_p_at_L_over_5": _safe_float(ct_lr.get("p_at_L_over_5")),
        "contacts_lr_auroc": _safe_float(ct_lr.get("auroc")),
        "contacts_sep6_auroc": _safe_float(ct_s6.get("auroc")),
    }


def _summarize_metric_peak_and_drop(
    trajectory: Sequence[Mapping[str, Any]],
    metric_key: str,
    *,
    positive_epochs_only: bool = True,
) -> dict[str, Any]:
    pts = [
        pt
        for pt in trajectory
        if pt.get(metric_key) is not None and ((not positive_epochs_only) or int(pt["epoch"]) > 0)
    ]
    if not pts:
        return {
            "n_points": 0,
            "peak_epoch": None,
            "peak_value": None,
            "final_epoch": None,
            "final_value": None,
            "overfitting_drop": None,
            "max_post_peak_drop": None,
            "delta_h_res_at_peak": None,
        }
    peak_idx = max(range(len(pts)), key=lambda i: float(pts[i][metric_key]))
    peak_pt = pts[peak_idx]
    final_pt = pts[-1]
    peak_val = float(peak_pt[metric_key])
    final_val = float(final_pt[metric_key])
    post_vals = [float(p[metric_key]) for p in pts[peak_idx:]]
    min_post = min(post_vals)
    return {
        "n_points": len(pts),
        "peak_epoch": int(peak_pt["epoch"]),
        "peak_value": peak_val,
        "final_epoch": int(final_pt["epoch"]),
        "final_value": final_val,
        "overfitting_drop": float(final_val - peak_val),
        "max_post_peak_drop": float(min_post - peak_val),
        "delta_h_res_at_peak": _safe_float(peak_pt.get("delta_h_res")),
    }


TRACKED_METRICS: tuple[str, ...] = (
    "delta_h_univ",
    "delta_h_spec",
    "megascale_overall",
    "megascale_de_novo",
    "megascale_unseen_fold",
    "megascale_seen_pool",
    "proteingym_assay",
    "proteingym_cluster",
    "contacts_lr_p_at_L",
    "contacts_lr_p_at_L_over_5",
    "contacts_lr_auroc",
    "contacts_sep6_auroc",
)


def analyze_stage2_run(
    doc: Mapping[str, Any],
    *,
    stage1_lookup: Mapping[str, Mapping[str, Any]] | None = None,
    res_takeoff_threshold_bpr: float = 0.05,
) -> dict[str, Any]:
    """Compute peak/drop, marginal bit allocation, K_res takeoff, and supply-demand terms for one run."""
    raw_pts: list[Mapping[str, Any]] = []
    if isinstance(doc.get("init_eval"), Mapping):
        raw_pts.append(doc["init_eval"])
    raw_pts.extend(m for m in doc.get("milestones", []) if isinstance(m, Mapping))

    by_epoch: dict[int, dict[str, Any]] = {}
    for m in raw_pts:
        pt = _extract_milestone_point(m)
        by_epoch[pt["epoch"]] = pt
    trajectory = [by_epoch[e] for e in sorted(by_epoch.keys())]

    param_counts = doc.get("parameter_counts") or {}
    p_non_emb = int(param_counts.get("n_params_non_embedding") or 0)
    corpus_stats = doc.get("corpus_stats") or {}
    t_uniq = int(corpus_stats.get("n_train_unique_tokens") or 0)
    objective = _canonical_objective(str(doc.get("objective", "")))

    # Marginal bit allocation d(delta_h)/d(ln e) across consecutive positive milestones
    pos_traj = [pt for pt in trajectory if pt["epoch"] > 0]
    marginal_allocation: list[dict[str, Any]] = []
    for i in range(1, len(pos_traj)):
        p0, p1 = pos_traj[i - 1], pos_traj[i]
        d_ln_e = math.log(float(p1["epoch"]) / float(p0["epoch"]))
        if d_ln_e <= 0.0:
            continue
        def _slope(key: str) -> float | None:
            v0, v1 = p0.get(key), p1.get(key)
            if v0 is None or v1 is None:
                return None
            return float((v1 - v0) / d_ln_e)
        marginal_allocation.append(
            {
                "epoch_start": int(p0["epoch"]),
                "epoch_end": int(p1["epoch"]),
                "d_delta_h_univ_d_ln_e": _slope("delta_h_univ"),
                "d_delta_h_spec_d_ln_e": _slope("delta_h_spec"),
                "d_delta_h_res_d_ln_e": _slope("delta_h_res"),
            }
        )

    peaks_and_drops = {
        key: _summarize_metric_peak_and_drop(trajectory, key, positive_epochs_only=True)
        for key in TRACKED_METRICS
    }

    pos_epochs = [float(pt["epoch"]) for pt in pos_traj]
    pos_dh_res = [pt["delta_h_res"] for pt in pos_traj]
    res_takeoff_epoch = _log_linear_first_crossing(
        [e for e, v in zip(pos_epochs, pos_dh_res) if v is not None],
        [float(v) for v in pos_dh_res if v is not None],
        res_takeoff_threshold_bpr,
    )

    univ_peak_ep = peaks_and_drops["delta_h_univ"]["peak_epoch"]
    univ_peak_val = peaks_and_drops["delta_h_univ"]["peak_value"]
    pre_blowup_pts = [
        pt
        for pt in pos_traj
        if pt.get("delta_h_res") is not None and float(pt["delta_h_res"]) <= 1.0
    ]
    pre_blowup_spec_max = (
        max(
            (float(pt["delta_h_spec"]) for pt in pre_blowup_pts if pt.get("delta_h_spec") is not None),
            default=None,
        )
    )
    absorption_order_holds: bool | None = None
    if univ_peak_ep is not None and res_takeoff_epoch is not None:
        absorption_order_holds = bool(float(univ_peak_ep) <= float(res_takeoff_epoch) + 1e-9)
    elif univ_peak_ep is not None and pos_traj:
        # Never crossed K_res takeoff while delta_h_univ rose cleanly
        absorption_order_holds = bool((univ_peak_val or 0.0) > 0.0)

    # Empirical bit demand & Stage-1 supply-demand alignment
    k_univ_bits = (
        max(0.0, float(univ_peak_val)) * float(t_uniq)
        if (univ_peak_val is not None and t_uniq > 0)
        else None
    )
    k_spec_bits = (
        max(0.0, float(pre_blowup_spec_max)) * float(t_uniq)
        if (pre_blowup_spec_max is not None and t_uniq > 0)
        else None
    )
    k_useful_bits = (
        float(k_univ_bits + k_spec_bits)
        if (k_univ_bits is not None and k_spec_bits is not None)
        else None
    )

    stage1_entry = (stage1_lookup or {}).get(objective)
    stage1_ecrit = stage1_entry.get("e_crit") if stage1_entry else None
    res_to_stage1_ratio = (
        float(res_takeoff_epoch / stage1_ecrit)
        if (res_takeoff_epoch is not None and stage1_ecrit is not None and stage1_ecrit > 0.0)
        else None
    )

    supply_curve: list[dict[str, Any]] = []
    predicted_e_star_supply: float | None = None
    if stage1_lookup and p_non_emb > 0:
        for pt in pos_traj:
            c_eff = interpolate_stage1_c_eff(stage1_lookup, objective, float(pt["epoch"]))
            b_supply = float(c_eff * p_non_emb) if c_eff is not None else None
            ratio = (
                float(b_supply / k_useful_bits)
                if (b_supply is not None and k_useful_bits is not None and k_useful_bits > 0.0)
                else None
            )
            supply_curve.append(
                {
                    "epoch": int(pt["epoch"]),
                    "stage1_c_eff_bpp": c_eff,
                    "b_param_supply_bits": b_supply,
                    "supply_to_useful_demand_ratio": ratio,
                }
            )
        if stage1_entry and k_useful_bits is not None and k_useful_bits > 0.0:
            s1_pts = stage1_entry.get("curve_points") or []
            if s1_pts:
                predicted_e_star_supply = _log_linear_first_crossing(
                    [pt[0] for pt in s1_pts],
                    [pt[1] * float(p_non_emb) for pt in s1_pts],
                    k_useful_bits,
                )

    final_pt = pos_traj[-1] if pos_traj else (trajectory[-1] if trajectory else {})
    return {
        "run_id": str(doc["run_id"]),
        "arm_id": str(doc["arm_id"]),
        "objective": objective,
        "scale_id": str(doc["scale_id"]),
        "status": str(doc.get("status", "unknown")),
        "epoch_done": int(doc.get("epoch_done", 0)),
        "max_epochs": int((doc.get("run_spec") or {}).get("max_epochs", 0)),
        "n_params_non_embedding": p_non_emb,
        "n_train_sequences": int(corpus_stats.get("n_train_sequences") or 0),
        "n_train_unique_tokens": t_uniq,
        "final_three_tier": {
            "H_train": final_pt.get("H_train"),
            "H_unseen_homolog": final_pt.get("H_unseen_homolog"),
            "H_unseen_fold": final_pt.get("H_unseen_fold"),
            "delta_h_univ": final_pt.get("delta_h_univ"),
            "delta_h_spec": final_pt.get("delta_h_spec"),
            "delta_h_res": final_pt.get("delta_h_res"),
        },
        "res_takeoff_threshold_bpr": float(res_takeoff_threshold_bpr),
        "res_takeoff_epoch": res_takeoff_epoch,
        "stage1_e_crit": stage1_ecrit,
        "res_takeoff_to_stage1_ecrit_ratio": res_to_stage1_ratio,
        "absorption_order_holds": absorption_order_holds,
        "peaks_and_drops": peaks_and_drops,
        "marginal_allocation": marginal_allocation,
        "demand_and_supply": {
            "k_univ_empirical_bits": k_univ_bits,
            "k_spec_pre_blowup_bits": k_spec_bits,
            "k_useful_empirical_bits": k_useful_bits,
            "k_useful_bits_per_param": (
                float(k_useful_bits / p_non_emb)
                if (k_useful_bits is not None and p_non_emb > 0)
                else None
            ),
            "predicted_e_star_from_stage1_supply": predicted_e_star_supply,
            "supply_by_milestone": supply_curve,
        },
        "trajectory": trajectory,
    }


def summarize_supply_demand_campaign(
    runs_dir: Path | str,
    *,
    stage1_path: Path | str | None = None,
    include_in_progress: bool = True,
    res_takeoff_threshold_bpr: float = 0.05,
) -> dict[str, Any]:
    """Aggregate Stage-2 runs into absorption-order, inversion, and supply-demand master summaries."""
    docs = load_stage2_run_receipts(runs_dir, include_in_progress=include_in_progress)
    stage1_lookup = load_stage1_capacity_lookup(stage1_path) if stage1_path is not None else {}

    per_run = [
        analyze_stage2_run(
            doc,
            stage1_lookup=stage1_lookup,
            res_takeoff_threshold_bpr=res_takeoff_threshold_bpr,
        )
        for doc in docs
    ]

    # 1. Objective-level K_res takeoff summary (tests AR << MLM-15% << OA-Uniform)
    by_objective: dict[str, Any] = {}
    for obj in sorted({r["objective"] for r in per_run}):
        subset = [r for r in per_run if r["objective"] == obj]
        takeoffs = [float(r["res_takeoff_epoch"]) for r in subset if r["res_takeoff_epoch"] is not None]
        ms_peaks = [
            float(r["peaks_and_drops"]["megascale_overall"]["peak_value"])
            for r in subset
            if r["peaks_and_drops"]["megascale_overall"]["peak_value"] is not None
        ]
        ms_drops = [
            float(r["peaks_and_drops"]["megascale_overall"]["overfitting_drop"])
            for r in subset
            if r["peaks_and_drops"]["megascale_overall"]["overfitting_drop"] is not None
        ]
        by_objective[obj] = {
            "n_runs": len(subset),
            "n_crossed_res_takeoff": len(takeoffs),
            "res_takeoff_epoch_median": float(np.median(takeoffs)) if takeoffs else None,
            "res_takeoff_epoch_min": float(min(takeoffs)) if takeoffs else None,
            "res_takeoff_epoch_max": float(max(takeoffs)) if takeoffs else None,
            "stage1_e_crit": (stage1_lookup.get(obj) or {}).get("e_crit"),
            "megascale_peak_mean": float(np.mean(ms_peaks)) if ms_peaks else None,
            "megascale_overfitting_drop_mean": float(np.mean(ms_drops)) if ms_drops else None,
        }

    # 2. Family-breadth (arm_id) summary across natural arms
    arm_order_idx = {a: i for i, a in enumerate(NATURAL_ARM_ORDER)}
    all_arms = sorted({r["arm_id"] for r in per_run}, key=lambda a: (arm_order_idx.get(a, 999), a))
    by_arm: dict[str, Any] = {}
    for arm in all_arms:
        subset = [r for r in per_run if r["arm_id"] == arm]
        univ_peaks = [
            float(r["peaks_and_drops"]["delta_h_univ"]["peak_value"])
            for r in subset
            if r["peaks_and_drops"]["delta_h_univ"]["peak_value"] is not None
        ]
        k_useful_vals = [
            float(r["demand_and_supply"]["k_useful_empirical_bits"])
            for r in subset
            if r["demand_and_supply"]["k_useful_empirical_bits"] is not None
        ]
        by_arm[arm] = {
            "n_runs": len(subset),
            "objectives_present": sorted({r["objective"] for r in subset}),
            "scales_present": sorted({r["scale_id"] for r in subset}),
            "delta_h_univ_peak_mean": float(np.mean(univ_peaks)) if univ_peaks else None,
            "k_useful_empirical_bits_mean": float(np.mean(k_useful_vals)) if k_useful_vals else None,
            "runs": [
                {
                    "run_id": r["run_id"],
                    "objective": r["objective"],
                    "scale_id": r["scale_id"],
                    "epoch_done": r["epoch_done"],
                    "res_takeoff_epoch": r["res_takeoff_epoch"],
                    "delta_h_univ_peak": r["peaks_and_drops"]["delta_h_univ"]["peak_value"],
                    "megascale_peak_epoch": r["peaks_and_drops"]["megascale_overall"]["peak_epoch"],
                    "megascale_peak": r["peaks_and_drops"]["megascale_overall"]["peak_value"],
                    "megascale_drop": r["peaks_and_drops"]["megascale_overall"]["overfitting_drop"],
                    "proteingym_peak": r["peaks_and_drops"]["proteingym_assay"]["peak_value"],
                    "proteingym_drop": r["peaks_and_drops"]["proteingym_assay"]["overfitting_drop"],
                }
                for r in subset
            ],
        }

    # 3. Likelihood-stability inversion correlations vs delta_h_res across all positive milestones
    inversion_metrics = (
        "megascale_overall",
        "megascale_de_novo",
        "megascale_unseen_fold",
        "megascale_seen_pool",
        "proteingym_assay",
        "contacts_lr_auroc",
        "contacts_lr_p_at_L",
    )
    inversion_vs_delta_h_res: dict[str, Any] = {}
    post_peak_inversion_vs_delta_h_res: dict[str, Any] = {}
    for mkey in inversion_metrics:
        xs_all: list[float] = []
        ys_all: list[float] = []
        xs_post: list[float] = []
        ys_post: list[float] = []
        for r in per_run:
            pk_ep = r["peaks_and_drops"][mkey]["peak_epoch"]
            for pt in r["trajectory"]:
                if pt["epoch"] <= 0:
                    continue
                dh_res = pt.get("delta_h_res")
                mval = pt.get(mkey)
                if dh_res is not None and mval is not None:
                    xs_all.append(float(dh_res))
                    ys_all.append(float(mval))
                    if pk_ep is not None and int(pt["epoch"]) >= int(pk_ep):
                        xs_post.append(float(dh_res))
                        ys_post.append(float(mval))
        inversion_vs_delta_h_res[mkey] = _safe_corr(xs_all, ys_all)
        post_peak_inversion_vs_delta_h_res[mkey] = _safe_corr(xs_post, ys_post)

    # 4. Master-equation optimal sizing P*(M, O, e) & observed vs predicted e* alignment
    master_equation_points: list[dict[str, Any]] = []
    for r in per_run:
        ds = r["demand_and_supply"]
        k_use = ds["k_useful_empirical_bits"]
        ms_pk_ep = r["peaks_and_drops"]["megascale_overall"]["peak_epoch"]
        obs_e_star = ms_pk_ep or r["peaks_and_drops"]["delta_h_univ"]["peak_epoch"]
        c_eff_at_obs = (
            interpolate_stage1_c_eff(stage1_lookup, r["objective"], float(obs_e_star))
            if (stage1_lookup and obs_e_star is not None)
            else None
        )
        p_star_at_obs = (
            float(k_use / c_eff_at_obs)
            if (k_use is not None and c_eff_at_obs is not None and c_eff_at_obs > 1e-6)
            else None
        )
        master_equation_points.append(
            {
                "run_id": r["run_id"],
                "arm_id": r["arm_id"],
                "objective": r["objective"],
                "scale_id": r["scale_id"],
                "n_params_non_embedding": r["n_params_non_embedding"],
                "k_useful_empirical_bits": k_use,
                "observed_e_star": obs_e_star,
                "predicted_e_star_from_stage1_supply": ds["predicted_e_star_from_stage1_supply"],
                "stage1_c_eff_at_obs_e_star": c_eff_at_obs,
                "optimal_params_P_star_at_obs_e_star": p_star_at_obs,
            }
        )

    summary: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "n_runs_analyzed": len(per_run),
        "n_runs_complete": sum(1 for r in per_run if r["status"] == "complete"),
        "res_takeoff_threshold_bpr": float(res_takeoff_threshold_bpr),
        "stage1_objectives_loaded": sorted(stage1_lookup.keys()),
        "by_objective": by_objective,
        "by_arm": by_arm,
        "inversion_vs_delta_h_res_all_milestones": inversion_vs_delta_h_res,
        "inversion_vs_delta_h_res_post_peak": post_peak_inversion_vs_delta_h_res,
        "master_equation_alignment": master_equation_points,
        "runs": per_run,
    }
    assert_no_forbidden_infra_strings(summary, label="supply_demand_summary")
    return summary
