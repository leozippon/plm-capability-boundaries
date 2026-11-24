#!/usr/bin/env python3
"""CLI for Experiment 4.1: Dual-Axis Full-Panel Unconditional Generation & Anchored Controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.unified.r5_experiment_4_1 import (  # noqa: E402
    annotate_available_pools,
    write_experiment_4_1_master,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--diamond-threads", type=int, default=64)
    parser.add_argument("--hmmscan-workers", type=int, default=32)
    parser.add_argument("--hmmscan-threads-per-worker", type=int, default=2)
    parser.add_argument(
        "--annotate-only",
        action="store_true",
        help="Run incremental CPU DIAMOND + hmmscan on all currently available pools and update caches.",
    )
    args = parser.parse_args()

    if args.annotate_only:
        res = annotate_available_pools(
            repo_root=args.repo_root,
            diamond_threads=args.diamond_threads,
            hmmscan_workers=args.hmmscan_workers,
            hmmscan_threads_per_worker=args.hmmscan_threads_per_worker,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return

    res = write_experiment_4_1_master(
        repo_root=args.repo_root,
        out_dir=args.out_dir,
        draws=args.draws,
        seed=args.seed,
        diamond_threads=args.diamond_threads,
        hmmscan_workers=args.hmmscan_workers,
        hmmscan_threads_per_worker=args.hmmscan_threads_per_worker,
    )
    print(f"Wrote {res['path']} (sha256={res['sha256']})")


if __name__ == "__main__":
    main()
