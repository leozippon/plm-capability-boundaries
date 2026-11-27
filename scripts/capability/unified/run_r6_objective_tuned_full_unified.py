#!/usr/bin/env python3
"""Run full-cohort Result 6 ESMFold2 evaluation (68/68 shards + 2,000/2,000 long_cap) on CPU.

Reads the completed structure TSVs and merged 68/68 Pfam annotations under
``results/extensions/unified_yardstick_20261124/r6_objective_tuned_full_cache/``,
passes Gate 1 reproduction checks (< 1e-6 vs frozen 18/68 and 1,748/2,000),
runs all six analytical components (a)-(f) with ZERO sampling and ZERO
departures, and writes:

  - results/extensions/unified_yardstick_20261124/r6_objective_tuned_full_esmfold2_unified.json (+ .sha256)
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.unified.r6_objective_tuned_full_esmfold2 import run_r6_full_unified  # noqa: E402
from src.capability.unified.roster import UNIFIED_OUTPUT_DIR  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / UNIFIED_OUTPUT_DIR,
        help=f"Output directory (default: {UNIFIED_OUTPUT_DIR}).",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory holding checkpoint_structures_68of68.tsv, long_cap_structures_2000of2000.tsv, and pfam_remainder/.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_r6_full_unified(root=REPO_ROOT, out_dir=args.out_dir, cache_dir=args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
