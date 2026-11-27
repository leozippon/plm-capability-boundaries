#!/usr/bin/env python3
"""CPU aggregator for Unified Result 5 Downstream-1: Homologous-Prefix DMS Mutation Ranking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.unified.r5_dms_context_yardstick import (  # noqa: E402
    write_r5_dms_context_unified,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--priority-scope", default="core", choices=("core", "all"))
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Allow interim partial assembly while H200 scoring is still running.",
    )
    args = parser.parse_args()

    res = write_r5_dms_context_unified(
        repo_root=args.repo_root,
        out_dir=args.out_dir,
        allow_pending=args.allow_pending,
        priority_scope=args.priority_scope,
        draws=args.draws,
        seed=args.seed,
    )
    proto = res["doc"]["protocol"]
    print(
        f"Wrote {res['path']} (sha256={res['sha256']}) | status={res['doc']['status']} | "
        f"1A={proto['n_completed_1a_rescue_30_80']}/{proto['n_target_entries']} | "
        f"1B={proto['n_completed_1b_e09_five_band']}/{proto['n_target_entries']}"
    )


if __name__ == "__main__":
    main()
