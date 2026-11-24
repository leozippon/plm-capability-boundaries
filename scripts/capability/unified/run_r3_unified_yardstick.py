#!/usr/bin/env python3
"""Run Workstream B: Unified 62-model R3 Epistasis + Contacts yardstick (CPU).

Executes the < 1e-12 reproduction gate against cached 20261110 interaction cells,
fits all 62/62 single-sequence arms across M_cycle, R_cycle, and MR_cycle on exact
shared cached folds (61 CC groups, 7,805 double-mutant thermodynamic cycles),
packages the mechanism-tagged R3 contact panels (22-arm E01 perturbation propagation
and 16-arm attention/APC map), and writes:

  - results/extensions/unified_yardstick_20261124/r3_epistasis_and_contacts_unified_62.json (+ .sha256)
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.unified.r3_yardstick import run_r3_unified_yardstick  # noqa: E402
from src.capability.unified.roster import UNIFIED_OUTPUT_DIR  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unified 62-model R3 epistasis and contacts yardstick (CPU)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Max CPU worker processes (clamped to [1, 4]; default: 4).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / UNIFIED_OUTPUT_DIR,
        help=f"Output directory (default: {UNIFIED_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Recompute cached per-arm cells under <out-dir>/cells/r2_r3/r3_epistasis/.",
    )
    parser.add_argument(
        "--skip-repro-gate",
        action="store_true",
        help="Skip the < 1e-12 reproduction gate (for debugging only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workers = max(1, min(int(args.workers), 4))
    t0 = time.monotonic()

    run_r3_unified_yardstick(
        root=REPO_ROOT,
        out_dir=args.out_dir,
        workers=workers,
        force_refit=args.force_refit,
        skip_repro_gate=args.skip_repro_gate,
    )

    print(
        f"[unified-r3] completed in {time.monotonic() - t0:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
