#!/usr/bin/env python3
"""Seal compact (<500 KB), infra-clean Quick Win A layerwise probing summary (43 endpoints + 80 trajectory milestones)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.memory_theory.io_utils import write_json_with_sha256
from src.capability.supply_demand.spec import assert_no_forbidden_infra_strings, canonicalize_objective

SCHEMA_VERSION = "layerwise_probing_phase2_summary_v1"


def _canon_obj(raw: str) -> str:
    s = str(raw).strip().lower()
    if s in ("mdm", "mdm_uniform", "mlm"):
        return "oa_uniform"
    try:
        return canonicalize_objective(s)
    except ValueError:
        return s


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _extract_layer_curve(doc: Mapping[str, Any], probe_key: str, metric_fn: Any) -> dict[str, Any]:
    probe = doc.get(probe_key) or {}
    layers_list = probe.get("layers") or []
    by_name = {e.get("layer_name"): e for e in layers_list if isinstance(e, Mapping)}
    ordered = list(doc.get("layer_names") or by_name.keys())

    emb_val: float | None = None
    non_emb: list[tuple[str, float]] = []
    for ln in ordered:
        entry = by_name.get(ln)
        if entry is None:
            continue
        val = _safe_float(metric_fn(entry))
        if val is None:
            continue
        if ln in ("emb", "embed"):
            emb_val = val
        else:
            non_emb.append((str(ln), val))

    if not non_emb:
        return {
            "emb": emb_val,
            "peak_layer": None,
            "peak_value": None,
            "peak_rel_depth": None,
            "final_layer": None,
            "final_value": None,
            "erasure": None,
            "retention_frac": None,
            "by_layer": {},
        }

    pk_idx = int(np.argmax([v for _, v in non_emb]))
    pk_layer, pk_val = non_emb[pk_idx]
    fn_layer, fn_val = non_emb[-1]
    rel_depth = float(pk_idx) / float(max(1, len(non_emb) - 1))
    retention = float(fn_val / pk_val) if abs(pk_val) > 1e-9 else None
    return {
        "emb": round(emb_val, 5) if emb_val is not None else None,
        "peak_layer": pk_layer,
        "peak_value": round(float(pk_val), 5),
        "peak_rel_depth": round(rel_depth, 4),
        "final_layer": fn_layer,
        "final_value": round(float(fn_val), 5),
        "erasure": round(float(fn_val - pk_val), 5),
        "retention_frac": round(retention, 4) if retention is not None else None,
        "by_layer": {ln: round(float(v), 5) for ln, v in non_emb},
    }


def _summarize_probe_doc(doc: Mapping[str, Any], *, source: str, include_layer_map: bool = False) -> dict[str, Any]:
    bur = _extract_layer_curve(doc, "probe_burial", lambda x: x.get("pooled_r2"))
    cnt = _extract_layer_curve(
        doc, "probe_contacts", lambda x: (x.get("long_range_sep_ge_24") or {}).get("auroc")
    )
    cnt_p = _extract_layer_curve(
        doc, "probe_contacts", lambda x: (x.get("long_range_sep_ge_24") or {}).get("p_at_L")
    )
    ms_uf = _extract_layer_curve(
        doc,
        "probe_megascale_ddg",
        lambda x: ((x.get("by_stratum") or {}).get("unseen_fold_natural") or {}).get("spearman_mean"),
    )
    ms_dn = _extract_layer_curve(
        doc,
        "probe_megascale_ddg",
        lambda x: ((x.get("by_stratum") or {}).get("de_novo_design") or {}).get("spearman_mean"),
    )
    if not include_layer_map:
        for d in (bur, cnt, cnt_p, ms_uf, ms_dn):
            d.pop("by_layer", None)
    arm = str(doc.get("arm_id", ""))
    return {
        "checkpoint_id": str(doc.get("checkpoint_id", "")),
        "run_id": str(doc.get("run_id", "")),
        "source": source,
        "arm_id": arm,
        "is_knob3": arm.startswith("T"),
        "scale_id": str(doc.get("scale_id", "")),
        "objective": _canon_obj(str(doc.get("objective", ""))),
        "epoch": int(doc.get("E", doc.get("epoch", 0))),
        "global_step": int(doc.get("global_step", 0)),
        "n_layers": int(doc.get("n_layers", 0)),
        "H_unseen_fold": round(float(doc["H_unseen_fold"]), 5) if _safe_float(doc.get("H_unseen_fold")) is not None else None,
        "H_cal": round(float(doc["H_cal"]), 5) if _safe_float(doc.get("H_cal")) is not None else None,
        "Delta_cal": round(float(doc["Delta_cal"]), 5) if _safe_float(doc.get("Delta_cal")) is not None else None,
        "T_opt": round(float(doc["T_opt"]), 5) if _safe_float(doc.get("T_opt")) is not None else None,
        "burial_r2": bur,
        "contacts_lr_auroc": cnt,
        "contacts_lr_p_at_L": cnt_p,
        "megascale_unseen_fold_spearman": ms_uf,
        "megascale_de_novo_spearman": ms_dn,
    }


def _aggregate_endpoints(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_obj: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_obj[str(r["objective"])].append(r)

    obj_table: dict[str, Any] = {}
    for obj in sorted(by_obj.keys()):
        grp = by_obj[obj]

        def _mean(fn: Any) -> float | None:
            vals = [fn(r) for r in grp if fn(r) is not None and math.isfinite(float(fn(r)))]
            return round(float(np.mean(vals)), 5) if vals else None

        obj_table[obj] = {
            "n_checkpoints": len(grp),
            "mean_Delta_cal": _mean(lambda r: r["Delta_cal"]),
            "burial_peak_r2": _mean(lambda r: r["burial_r2"]["peak_value"]),
            "burial_final_r2": _mean(lambda r: r["burial_r2"]["final_value"]),
            "burial_erasure": _mean(lambda r: r["burial_r2"]["erasure"]),
            "burial_peak_rel_depth": _mean(lambda r: r["burial_r2"]["peak_rel_depth"]),
            "contacts_lr_peak_auroc": _mean(lambda r: r["contacts_lr_auroc"]["peak_value"]),
            "contacts_lr_final_auroc": _mean(lambda r: r["contacts_lr_auroc"]["final_value"]),
            "contacts_lr_erasure": _mean(lambda r: r["contacts_lr_auroc"]["erasure"]),
            "contacts_lr_peak_rel_depth": _mean(lambda r: r["contacts_lr_auroc"]["peak_rel_depth"]),
            "megascale_uf_peak": _mean(lambda r: r["megascale_unseen_fold_spearman"]["peak_value"]),
            "megascale_uf_final": _mean(lambda r: r["megascale_unseen_fold_spearman"]["final_value"]),
            "megascale_uf_erasure": _mean(lambda r: r["megascale_unseen_fold_spearman"]["erasure"]),
            "megascale_uf_retention_frac": _mean(lambda r: r["megascale_unseen_fold_spearman"]["retention_frac"]),
            "megascale_dn_erasure": _mean(lambda r: r["megascale_de_novo_spearman"]["erasure"]),
        }

    corrs: dict[str, Any] = {}
    for name, fn in (
        ("burial_erasure", lambda r: r["burial_r2"]["erasure"]),
        ("contacts_lr_erasure", lambda r: r["contacts_lr_auroc"]["erasure"]),
        ("megascale_uf_erasure", lambda r: r["megascale_unseen_fold_spearman"]["erasure"]),
        ("burial_peak_rel_depth", lambda r: r["burial_r2"]["peak_rel_depth"]),
        ("contacts_lr_peak_rel_depth", lambda r: r["contacts_lr_auroc"]["peak_rel_depth"]),
    ):
        pairs = [
            (float(r["Delta_cal"]), float(fn(r)))
            for r in rows
            if r["Delta_cal"] is not None and fn(r) is not None
        ]
        if len(pairs) >= 4:
            xs = np.asarray([a for a, _ in pairs], dtype=np.float64)
            ys = np.asarray([b for _, b in pairs], dtype=np.float64)
            if float(np.std(xs)) > 1e-12 and float(np.std(ys)) > 1e-12:
                sr = spearmanr(xs, ys)
                corrs[name] = {
                    "n": len(pairs),
                    "spearman_r": round(float(sr.statistic), 5),
                    "spearman_p": float(f"{sr.pvalue:.3e}"),
                }

    return {"by_objective": obj_table, "spearman_with_Delta_cal": corrs}


def _summarize_trajectory_stages(pts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Quantify Stage 0 (physics acquisition), Stage I (E=10..40 H1 Delta_cal), Stage II (E=40..300 H2 locality bifurcation)."""
    by_ep = {int(p["epoch"]): p for p in pts}
    pos_pts = [p for p in pts if int(p["epoch"]) >= 1]
    if not pos_pts:
        return {}

    bur_peak_ep = max(pos_pts, key=lambda p: float(p["burial_r2"]["final_value"] or -1e9))
    cnt_peak_ep = max(pos_pts, key=lambda p: float(p["contacts_lr_auroc"]["final_value"] or -1e9))
    ms_peak_ep = max(pos_pts, key=lambda p: float(p["megascale_unseen_fold_spearman"]["final_value"] or -1e9))

    # First epoch where Delta_cal >= 0.25 (H1 takeoff)
    h1_onset_ep = next((int(p["epoch"]) for p in pos_pts if (p["Delta_cal"] or 0.0) >= 0.25), None)
    final_pt = max(pts, key=lambda p: int(p["epoch"]))
    p40 = by_ep.get(40)

    bur_final_peak = float(bur_peak_ep["burial_r2"]["final_value"])
    bur_end = float(final_pt["burial_r2"]["final_value"])
    cnt_final_peak = float(cnt_peak_ep["contacts_lr_auroc"]["final_value"])
    cnt_end = float(final_pt["contacts_lr_auroc"]["final_value"])
    ms_final_peak = float(ms_peak_ep["megascale_unseen_fold_spearman"]["final_value"])
    ms_end = float(final_pt["megascale_unseen_fold_spearman"]["final_value"])

    return {
        "n_unique_epochs": len(pts),
        "epochs": [int(p["epoch"]) for p in pts],
        "h1_delta_cal_onset_epoch_ge_0p25": h1_onset_ep,
        "burial_norm_f_peak_epoch": int(bur_peak_ep["epoch"]),
        "burial_norm_f_peak_r2": round(bur_final_peak, 5),
        "burial_norm_f_final_r2": round(bur_end, 5),
        "burial_norm_f_drop_from_temporal_peak": round(bur_end - bur_final_peak, 5),
        "burial_norm_f_retention_from_temporal_peak": round(bur_end / max(bur_final_peak, 1e-9), 4),
        "contacts_norm_f_peak_epoch": int(cnt_peak_ep["epoch"]),
        "contacts_norm_f_peak_auroc": round(cnt_final_peak, 5),
        "contacts_norm_f_final_auroc": round(cnt_end, 5),
        "contacts_norm_f_drop_from_temporal_peak": round(cnt_end - cnt_final_peak, 5),
        "megascale_uf_norm_f_peak_epoch": int(ms_peak_ep["epoch"]),
        "megascale_uf_norm_f_peak": round(ms_final_peak, 5),
        "megascale_uf_norm_f_final": round(ms_end, 5),
        "megascale_uf_norm_f_retention_from_temporal_peak": round(ms_end / max(ms_final_peak, 1e-9), 4),
        "megascale_uf_layer_retention_at_final_epoch": final_pt["megascale_unseen_fold_spearman"]["retention_frac"],
        "stage1_e40_Delta_cal": p40["Delta_cal"] if p40 else None,
        "stage1_e40_burial_erasure": p40["burial_r2"]["erasure"] if p40 else None,
        "stage1_e40_contacts_erasure": p40["contacts_lr_auroc"]["erasure"] if p40 else None,
        "stage2_final_epoch": int(final_pt["epoch"]),
        "stage2_final_Delta_cal": final_pt["Delta_cal"],
        "stage2_final_burial_erasure": final_pt["burial_r2"]["erasure"],
        "stage2_final_contacts_erasure": final_pt["contacts_lr_auroc"]["erasure"],
        "stage2_final_megascale_uf_erasure": final_pt["megascale_unseen_fold_spearman"]["erasure"],
    }


def build_summary(all43_dir: Path, traj_dir: Path) -> dict[str, Any]:
    all43_files = sorted((all43_dir / "checkpoints").glob("*.probe.json"))
    traj_files = sorted((traj_dir / "checkpoints").glob("*.probe.json"))
    if not all43_files:
        raise FileNotFoundError(f"No probe JSONs in {all43_dir / 'checkpoints'}")
    if not traj_files:
        raise FileNotFoundError(f"No probe JSONs in {traj_dir / 'checkpoints'}")

    all43_rows = [
        _summarize_probe_doc(json.loads(p.read_text(encoding="utf-8")), source="probing_all43")
        for p in all43_files
    ]
    raw_traj_rows = [
        _summarize_probe_doc(json.loads(p.read_text(encoding="utf-8")), source="probing_trajectories")
        for p in traj_files
    ]

    # Deduplicate _latest snapshots that duplicate an exact epoch milestone (e.g. e0150 == latest on 35m)
    traj_grouped: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in raw_traj_rows:
        key = (row["scale_id"], row["arm_id"], row["objective"])
        ep = int(row["epoch"])
        # Prefer explicit epoch snapshot over _latest duplicate
        if ep not in traj_grouped[key] or "latest" not in row["checkpoint_id"]:
            traj_grouped[key][ep] = row

    # Stitch E=300 endpoints from probing_all43 ONLY onto trajectories stopping strictly earlier (35m M64 E=150 -> E=300)
    stitched_endpoints: list[str] = []
    for row in all43_rows:
        key = (row["scale_id"], row["arm_id"], row["objective"])
        ep = int(row["epoch"])
        if key in traj_grouped and ep > max(traj_grouped[key].keys()):
            stitched_row = dict(row)
            stitched_row["source"] = "probing_all43_stitched_e300"
            traj_grouped[key][ep] = stitched_row
            stitched_endpoints.append(f"{key[0]}_{key[1]}_{key[2]}_E{ep}")

    trajectories_out: dict[str, Any] = {}
    stage_summary_table: list[dict[str, Any]] = []
    for key in sorted(traj_grouped.keys()):
        tname = f"{key[0]}_{key[1]}_{key[2]}"
        pts = [traj_grouped[key][ep] for ep in sorted(traj_grouped[key].keys())]
        stages = _summarize_trajectory_stages(pts)
        trajectories_out[tname] = {
            "scale_id": key[0],
            "arm_id": key[1],
            "objective": key[2],
            "two_stage_dynamics": stages,
            "milestones": pts,
        }
        stage_summary_table.append({"trajectory": tname, "scale_id": key[0], "arm_id": key[1], "objective": key[2], **stages})

    nat_rows = [r for r in all43_rows if not r["is_knob3"]]
    k3_rows = [r for r in all43_rows if r["is_knob3"]]

    report: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "n_endpoints_all43": len(all43_rows),
            "n_endpoints_natural": len(nat_rows),
            "n_endpoints_knob3": len(k3_rows),
            "n_trajectory_probe_files": len(raw_traj_rows),
            "n_total_probed_checkpoints": len(all43_rows) + len(raw_traj_rows),
            "n_trajectories": len(trajectories_out),
            "stitched_e300_endpoints": sorted(stitched_endpoints),
        },
        "two_stage_temporal_summary": stage_summary_table,
        "trajectories": trajectories_out,
        "endpoints_43_synthesis": {
            "all_43": _aggregate_endpoints(all43_rows),
            "natural_31": _aggregate_endpoints(nat_rows),
            "natural_small_0p5m_2m": _aggregate_endpoints([r for r in nat_rows if r["scale_id"] in ("0.5m", "2m")]),
            "natural_large_8m_35m": _aggregate_endpoints([r for r in nat_rows if r["scale_id"] in ("8m", "35m")]),
            "knob3_temperature_sweep_2m": _aggregate_endpoints(k3_rows),
            "per_checkpoint": all43_rows,
        },
    }
    assert_no_forbidden_infra_strings(report, label="layerwise_probing_summary")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all43-dir", type=Path, default=REPO_ROOT / "output/probing_all43")
    p.add_argument("--traj-dir", type=Path, default=REPO_ROOT / "output/probing_trajectories")
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results/extensions/supply_demand_phase2_20261128/layerwise_probing_summary.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = build_summary(args.all43_dir, args.traj_dir)
    write_json_with_sha256(args.out, report)
    size_kb = args.out.stat().st_size / 1024.0
    print(
        json.dumps(
            {
                "status": "ok",
                "out": str(args.out),
                "size_kb": round(size_kb, 2),
                "counts": report["counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
