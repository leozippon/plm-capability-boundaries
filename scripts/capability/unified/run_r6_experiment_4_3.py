#!/usr/bin/env python3
"""Run unified 30-model / 33-pool Experiment 4.3 (Native Self-Likelihood Ranking &
Adverse Selection Inside Novel <20% / <30% Candidate Pools).

Writes ``results/extensions/unified_yardstick_20261124/r6_experiment_4_3_native_likelihood_unified.json``
plus its verified ``.sha256`` sidecar.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

from src.capability.unified.r6_experiment_4_3 import build_unified_r6_experiment_4_3  # noqa: E402
from src.capability.unified.roster import UNIFIED_OUTPUT_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / UNIFIED_OUTPUT_DIR,
        help="Output directory (default: results/extensions/unified_yardstick_20261124)",
    )
    parser.add_argument("--diamond-threads", type=int, default=32)
    parser.add_argument("--pfam-workers", type=int, default=16)
    parser.add_argument("--eval-workers", type=int, default=12)
    parser.add_argument(
        "--annotate-only",
        action="store_true",
        help="Run E17 reproduction gate + DIAMOND/Pfam annotations only (before GPU scores land)",
    )
    args = parser.parse_args()

    res = build_unified_r6_experiment_4_3(
        root=ROOT,
        out_dir=args.out_dir,
        diamond_threads=args.diamond_threads,
        pfam_workers=args.pfam_workers,
        eval_workers=args.eval_workers,
        annotate_only=args.annotate_only,
    )
    if args.annotate_only:
        print(
            f"\n[exp4.3] ANNOTATE-ONLY COMPLETE: unique={res['n_unique_input_sequences']}, "
            f"diamond={res['n_diamond_cached']}, pfam={res['n_pfam_cached']}, "
            f"e17_gate={res['e17_reproduction_gate']['passed']} "
            f"(max_diff={res['e17_reproduction_gate']['max_abs_diff']:.2e})",
            flush=True,
        )
        return

    print("\n=== Experiment 4.3 Cross-Model Summary ===", flush=True)
    for reg_name, blk in res["cross_model_summary"].items():
        n_ev = blk["n_pools_evaluable"]
        sp_p = blk["spearman_counts"]["plddt"]
        sp_5 = blk["spearman_counts"]["distinct_5mer"]
        sel25 = blk["truncation_selection_counts"]["plddt"]["top_25pct"]
        print(
            f"  {reg_name:36s}: eval={n_ev:2d}/33 | pLDDT rho(pos={sp_p['ci_positive']}, neg={sp_p['ci_negative']}, pt<0={sp_p['point_negative']}) "
            f"| 5mer rho(neg={sp_5['ci_negative']}, pt<0={sp_5['point_negative']}) "
            f"| top25% pLDDT(adverse_ci={sel25['n_ci_adverse']}, pt_adv={sel25['n_point_adverse']}, pos_ci={sel25['n_ci_positive']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
