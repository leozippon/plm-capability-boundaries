"""Unified-notation analysis for controlled multimodal B_ext(Z) offloading & ESM-3 joint-to-seq transfer."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..supply_demand.analysis import _log_linear_first_crossing, _safe_float
from .spec import (
    LOG2_20,
    assert_no_forbidden_infra_strings,
    canonicalize_mode,
    canonicalize_objective,
    objective_display_name,
)
from .training import MULTIMODAL_RECEIPT_SCHEMA

MULTIMODAL_ANALYSIS_SCHEMA = "plm_multimodal_analysis_v1"


def analyze_multimodal_run(
    doc: Mapping[str, Any],
    *,
    res_takeoff_threshold_bpr: float = 0.05,
) -> dict[str, Any]:
    """Extract unified-notation trajectory (E_gen, E_mem, W_window, Delta_cal, b_ext(Z), downstream peaks)."""
    mode = canonicalize_mode(str(doc.get("mode", "pure_seq")))
    objective = canonicalize_objective(str(doc.get("objective", "oa_uniform")))
    milestones = [m for m in doc.get("milestones", []) if isinstance(m, Mapping)]
    milestones.sort(key=lambda m: int(m.get("epoch", 0)))

    epochs: list[float] = []
    pos_univ: list[float] = []
    dh_res_list: list[float] = []
    traj: list[dict[str, Any]] = []

    for m in milestones:
        ep = int(m.get("epoch", 0))
        tt = (m.get("three_tier_bits") or {}).get("native") or {}
        b_diag = (m.get("three_tier_bits") or {}).get("b_ext_diagnostics") or {}
        ms = m.get("megascale") or {}
        pg = m.get("proteingym_milestone") or {}
        ct = (m.get("contacts") or {}).get("long_range_sep_ge_24") or {}
        s2s = m.get("seq_to_struct_readout") or {}

        dh_u = _safe_float(tt.get("delta_h_univ"))
        dh_s = _safe_float(tt.get("delta_h_spec"))
        dh_r = _safe_float(tt.get("delta_h_res"))
        h_uf = _safe_float(tt.get("H_unseen_fold"))
        h_uf_cal = _safe_float(tt.get("H_unseen_fold_cal"))
        d_cal = _safe_float(tt.get("Delta_cal"))
        if d_cal is None and h_uf is not None:
            d_cal = max(0.0, h_uf - LOG2_20)

        if ep > 0 and dh_u is not None:
            epochs.append(float(ep))
            pos_univ.append(max(0.0, dh_u))
        if ep > 0 and dh_r is not None:
            dh_res_list.append(dh_r)

        traj.append(
            {
                "epoch": ep,
                "H_unseen_fold": h_uf,
                "H_unseen_fold_cal": h_uf_cal,
                "Delta_cal": d_cal,
                "delta_h_univ": dh_u,
                "delta_h_spec": dh_s,
                "delta_h_res": dh_r,
                "b_ext_vs_empty": _safe_float(b_diag.get("b_ext_vs_empty_bits_per_res")),
                "b_ext_vs_shuffled": _safe_float(b_diag.get("b_ext_vs_shuffled_bits_per_res")),
                "megascale_overall": _safe_float(ms.get("overall_spearman_mean")),
                "proteingym_assay": _safe_float(pg.get("assay_mean_spearman")),
                "contacts_lr_auroc": _safe_float(ct.get("auroc")),
                "three_di_accuracy": _safe_float(s2s.get("three_di_accuracy")),
                "contact_head_lr_auroc": _safe_float(
                    (s2s.get("contact_head_long_range_sep_ge_24") or {}).get("auroc")
                ),
            }
        )

    peak_univ = max(pos_univ, default=0.0)
    e_gen = (
        _log_linear_first_crossing(epochs, pos_univ, 0.80 * peak_univ, sustained=False)
        if peak_univ > 0.02
        else None
    )
    e_mem = _log_linear_first_crossing(
        [float(pt["epoch"]) for pt in traj if pt["epoch"] > 0 and pt["delta_h_res"] is not None],
        [float(pt["delta_h_res"]) for pt in traj if pt["epoch"] > 0 and pt["delta_h_res"] is not None],
        res_takeoff_threshold_bpr,
        sustained=True,
    )
    epoch_done = int(doc.get("epoch_done", 0))
    w_window = float(e_mem / e_gen) if (e_mem is not None and e_gen is not None and e_gen > 0.0) else None
    w_window_lb = (
        float(epoch_done / e_gen)
        if (e_mem is None and e_gen is not None and e_gen > 0.0 and epoch_done > 0)
        else w_window
    )
    final_pt = traj[-1] if traj else {}

    def _peak(key: str) -> float | None:
        vals = [float(pt[key]) for pt in traj if pt.get(key) is not None]
        return max(vals) if vals else None

    return {
        "run_id": str(doc["run_id"]),
        "mode": mode,
        "objective": objective,
        "paradigm_display": objective_display_name(objective),
        "scale_id": str(doc["scale_id"]),
        "status": str(doc.get("status", "unknown")),
        "epoch_done": epoch_done,
        "n_params_non_embedding": int((doc.get("parameter_counts") or {}).get("n_params_non_embedding") or 0),
        "E_gen": e_gen,
        "E_mem": e_mem,
        "W_window": w_window,
        "W_window_lower_bound": w_window_lb,
        "Delta_cal": final_pt.get("Delta_cal"),
        "H_unseen_fold_final": final_pt.get("H_unseen_fold"),
        "H_unseen_fold_cal_final": final_pt.get("H_unseen_fold_cal"),
        "delta_h_univ_peak": _peak("delta_h_univ"),
        "megascale_peak": _peak("megascale_overall"),
        "proteingym_peak": _peak("proteingym_assay"),
        "contacts_lr_auroc_peak": _peak("contacts_lr_auroc"),
        "three_di_accuracy_peak": _peak("three_di_accuracy"),
        "contact_head_lr_auroc_peak": _peak("contact_head_lr_auroc"),
        "b_ext_vs_shuffled_final": final_pt.get("b_ext_vs_shuffled"),
        "trajectory": traj,
    }


def summarize_multimodal_campaign(
    runs_dir: Path | str,
    *,
    include_in_progress: bool = True,
) -> dict[str, Any]:
    """Aggregate multimodal receipts into paired pure_seq vs struct_cond (Eq. 4) & joint_to_seq (ESM-3) contrasts."""
    root = Path(runs_dir)
    docs: list[dict[str, Any]] = []
    for p in sorted(root.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or d.get("schema") != MULTIMODAL_RECEIPT_SCHEMA:
            continue
        if (not include_in_progress) and d.get("status") != "complete":
            continue
        docs.append(d)

    per_run = [analyze_multimodal_run(d) for d in docs]
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {
        (r["mode"], r["objective"], r["scale_id"]): r for r in per_run
    }

    # Paired contrasts against matched pure_seq (same objective & scale_id)
    contrasts: list[dict[str, Any]] = []
    for (mode, obj, sc), r in sorted(by_key.items()):
        if mode == "pure_seq":
            continue
        base = by_key.get(("pure_seq", obj, sc))
        if base is None:
            continue
        h_base = base.get("H_unseen_fold_final")
        h_cur = r.get("H_unseen_fold_final")
        b_ext_matched = (
            float(h_base - h_cur)
            if (h_base is not None and h_cur is not None and math.isfinite(h_base) and math.isfinite(h_cur))
            else None
        )
        ms_gain = (
            float(r["megascale_peak"] - base["megascale_peak"])
            if (r.get("megascale_peak") is not None and base.get("megascale_peak") is not None)
            else None
        )
        contrasts.append(
            {
                "mode": mode,
                "objective": obj,
                "paradigm_display": objective_display_name(obj),
                "scale_id": sc,
                "pure_seq_run_id": base["run_id"],
                "target_run_id": r["run_id"],
                "b_ext_matched_bits_per_res": b_ext_matched,
                "megascale_peak_gain": ms_gain,
            }
        )

    summary = {
        "schema": MULTIMODAL_ANALYSIS_SCHEMA,
        "n_runs_analyzed": len(per_run),
        "n_runs_complete": sum(1 for r in per_run if r["status"] == "complete"),
        "paired_vs_pure_seq_contrasts": contrasts,
        "runs": per_run,
    }
    assert_no_forbidden_infra_strings(summary, label="multimodal_summary")
    return summary
