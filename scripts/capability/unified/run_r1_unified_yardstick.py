#!/usr/bin/env python3
"""CLI runner for Workstream A: Unified 62-Model R1 ProteinGym Yardstick.

Generates and SHA-256-verifies all three R1 unified single-yardstick outputs under
``results/extensions/unified_yardstick_20261124/``:

1. ``r1_proteingym_unified_62.json`` (+ ``.sha256``)
2. ``r1_supplementary_c0_ladder_62.json`` (+ ``.sha256``)
3. ``r1_adversarial_pairs_62.json`` (+ ``.sha256``)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.unified.r1_yardstick import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    run_r1_unified_yardstick,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to results/extensions/unified_yardstick_20261124).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Worker processes (capped at <= 4 per workstream).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="BLAS / PyTorch threads per worker process.",
    )
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()

    workers = max(1, min(int(args.workers), 4))
    threads = max(1, min(int(args.threads), 2))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)

    receipt = run_r1_unified_yardstick(
        root=args.root,
        out_dir=args.out_dir,
        workers=workers,
        threads=threads,
        draws=args.draws,
    )
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
