#!/usr/bin/env python3
"""Run the unified 62-model Merged New Experiment 2.1 (MegaScale + Domainome).

Writes ``results/extensions/unified_yardstick_20261124/r2_biophysical_unified_62.json``
plus its verified ``.sha256`` sidecar and per-arm caches under
``results/extensions/unified_yardstick_20261124/cells/r2_exp21/{megascale,domainome}/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Pin BLAS/OpenMP threads BEFORE importing numpy or torch to prevent oversubscription.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

from src.capability.unified.r2_experiment_2_1 import (  # noqa: E402
    DEFAULT_R_ARCHIVE_DIR,
    build_unified_r2_experiment_2_1,
)
from src.capability.unified.roster import UNIFIED_OUTPUT_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / UNIFIED_OUTPUT_DIR,
        help="Output directory (default: results/extensions/unified_yardstick_20261124)",
    )
    parser.add_argument(
        "--r-archive-dir",
        type=Path,
        default=ROOT / DEFAULT_R_ARCHIVE_DIR,
        help="Directory holding H200-extracted r2_{stability,abundance}_R_<arm>.npz files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum spawn worker processes (capped at 16)",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=2,
        help="Per-worker PyTorch/BLAS thread cap",
    )
    parser.add_argument(
        "--allow-partial-r",
        action="store_true",
        help="Allow running while some arms do not yet have H200-extracted R archives",
    )
    parser.add_argument(
        "--require-full-r",
        action="store_true",
        help="Fail fast unless all 62/62 arms have R on both MegaScale and Domainome",
    )
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Ignore cached per-arm JSONs under cells/r2_exp21/",
    )
    args = parser.parse_args()

    allow_partial = not args.require_full_r
    if args.allow_partial_r and args.require_full_r:
        parser.error("Cannot specify both --allow-partial-r and --require-full-r")

    payload = build_unified_r2_experiment_2_1(
        root=ROOT,
        out_dir=args.out_dir,
        r_archive_dir=args.r_archive_dir,
        max_workers=args.workers,
        blas_threads=args.blas_threads,
        allow_partial_r=allow_partial,
        force_refit=args.force_refit,
    )

    cov = payload["coverage_summary"]
    print("\n=== Coverage Summary ===", flush=True)
    print(f"MegaScale: {cov['megascale']['n_arms_M']}/62 M, {cov['megascale']['n_arms_R']}/62 R", flush=True)
    print(f"Domainome: {cov['domainome']['n_arms_M']}/62 M, {cov['domainome']['n_arms_R']}/62 R", flush=True)

    gates = payload["verification_gates"]
    print("\n=== Verification Gates ===", flush=True)
    print(
        f"MegaScale 101g repro: passed={gates['megascale_101g_reproduction']['passed']} "
        f"oof_diff={gates['megascale_101g_reproduction']['max_oof_abs_diff']:.2e}",
        flush=True,
    )
    print(
        f"ESMC-6B 79g C_evo parity: passed={gates['megascale_79g_esmc6b_evo_parity']['passed']} "
        f"mse_diff={gates['megascale_79g_esmc6b_evo_parity']['max_mse_abs_diff']:.2e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
