#!/usr/bin/env python3
"""Run Workstream C unified yardstick: family_memory_unified_62 + r4_r5_r6_unified_summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.unified.memory_and_generation_yardstick import (
    write_unified_memory_and_generation_outputs,
)
from src.capability.unified.roster import UNIFIED_OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / UNIFIED_OUTPUT_DIR,
    )
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261106)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_unified_memory_and_generation_outputs(
        args.repo_root,
        args.out_dir,
        draws=args.draws,
        seed=args.seed,
    )
    for name, rec in outputs.items():
        print(f"Wrote {rec['path']} (sha256={rec['sha256']})")


if __name__ == "__main__":
    main()
