#!/usr/bin/env python3
"""Build leakage-free CATH 3D/Foldseek-3Di pair cohort and Natural Text/Horizon cohort."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.memory_theory.cohort_builders import (
    build_cath_3d_kv_cohort,
    build_text_kv_and_horizon_cohort,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results/extensions/memory_theory_20261120",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path("/tmp/memory_theory_tools"),
    )
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--skip-cath", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_cath:
        cath_out = args.out_dir / "cath_3d_kv_cohort.json"
        cath = build_cath_3d_kv_cohort(
            args.repo_root,
            cath_out,
            scratch_root=args.scratch_dir,
            threads=args.threads,
            seed=args.seed,
        )
        print(f"Wrote {cath_out} (sha256={cath['sha256']}, summary={cath['summary']})")
    if not args.skip_text:
        text_out = args.out_dir / "text_kv_and_horizon_cohort.json"
        txt = build_text_kv_and_horizon_cohort(
            args.repo_root,
            text_out,
            seed=args.seed,
        )
        print(f"Wrote {text_out} (sha256={txt['sha256']}, summary={txt['summary']})")


if __name__ == "__main__":
    main()
