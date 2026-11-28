#!/usr/bin/env python3
"""Analyze Stage-2 supply-demand inversion runs and emit summary JSON + .sha256 sidecar."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.memory_theory.io_utils import write_json_with_sha256
from src.capability.supply_demand.analysis import summarize_supply_demand_campaign


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Stage-2 supply-demand campaign runs.")
    p.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Directory containing Stage-2 run JSON receipts.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for aggregated analysis summary JSON (+ .sha256 sidecar).",
    )
    p.add_argument(
        "--stage1-path",
        type=Path,
        default=None,
        help="Optional Stage-1 summary JSON or runs/ directory for supply-demand alignment.",
    )
    p.add_argument(
        "--completed-only",
        action="store_true",
        help="Exclude in-progress runs and analyze only status=='complete' receipts.",
    )
    p.add_argument(
        "--res-takeoff-threshold",
        type=float,
        default=0.05,
        help="Threshold in bits/res for delta_h_res takeoff detection (default: 0.05).",
    )
    return p.parse_args()


def _fmt(val: float | None, digits: int = 3) -> str:
    if val is None:
        return "-"
    return f"{val:.{digits}f}"


def main() -> None:
    args = parse_args()
    summary = summarize_supply_demand_campaign(
        args.runs_dir,
        stage1_path=args.stage1_path,
        include_in_progress=not args.completed_only,
        res_takeoff_threshold_bpr=args.res_takeoff_threshold,
    )
    digest = write_json_with_sha256(args.out, summary)

    print(
        f"Analyzed {summary['n_runs_analyzed']} Stage-2 runs "
        f"({summary['n_runs_complete']} complete) -> {args.out} (sha256={digest[:16]}...)"
    )
    print(
        f"{'run_id':34s} {'ep':>4s} {'e_res':>6s} {'Gamma':>6s} {'clipR':>5s} {'calEx':>6s} "
        f"{'dhU_pk':>7s} {'MS_e*':>5s} {'MS_pk':>6s} {'MS_drp':>7s} {'PG_pk':>6s} {'PG_drp':>7s}"
    )
    for r in summary["runs"]:
        pk = r["peaks_and_drops"]
        ft = r["final_three_tier"]
        ms = pk["megascale_overall"]
        pg = pk["proteingym_assay"]
        du = pk["delta_h_univ"]
        gam = r["adiabatic_window_gamma"]
        gam_s = _fmt(gam, 1) if gam is not None else (f">{_fmt(r['adiabatic_window_gamma_lower_bound'], 0)}" if r["adiabatic_window_gamma_lower_bound"] is not None else "-")
        print(
            f"{r['run_id']:34s} {r['epoch_done']:4d} "
            f"{_fmt(r['res_takeoff_epoch'], 1):>6s} {gam_s:>6s} "
            f"{_fmt(ft.get('clipped_memorized_res_bits'), 2):>5s} "
            f"{_fmt(ft.get('ood_miscalibration_excess'), 2):>6s} "
            f"{_fmt(du['peak_value']):>7s} "
            f"{str(ms['peak_epoch'] or '-'):>5s} {_fmt(ms['peak_value']):>6s} "
            f"{_fmt(ms['overfitting_drop']):>7s} {_fmt(pg['peak_value']):>6s} "
            f"{_fmt(pg['overfitting_drop']):>7s}"
        )

    post_inv = summary["inversion_vs_delta_h_res_post_peak"]
    ms_inv = post_inv.get("megascale_overall") or {}
    pg_inv = post_inv.get("proteingym_assay") or {}
    print(
        "Post-peak delta_h_res inversion: "
        f"MegaScale Spearman={_fmt(ms_inv.get('spearman_r'))} (n={ms_inv.get('n')}), "
        f"ProteinGym Spearman={_fmt(pg_inv.get('spearman_r'))} (n={pg_inv.get('n')})"
    )


if __name__ == "__main__":
    main()
