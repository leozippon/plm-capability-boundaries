#!/usr/bin/env python3
"""Track B1 CLI: Zero-GPU capacity-starvation & entropy-budget re-analysis.

Re-runnable whenever new arm scores (ESM2, DPLM, ProtGPT3, ProGen3, SaProt, etc.)
arrive in family_conditionals / family_distribution score directories.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("TRANSFER_DATA_ROOT") or REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.memory_theory.capacity_scaling import run_b1_reanalysis
from src.capability.memory_theory.io_utils import write_json_with_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DATA_ROOT,
        help="Repository root containing results/extensions/ artifacts.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_ROOT / "results/extensions/memory_theory_20261120/b1_baseline_reanalysis.json",
        help="Output JSON path (a matching .sha256 sidecar is written atomically).",
    )
    parser.add_argument(
        "--extra-fc-scores-dir",
        type=Path,
        action="append",
        default=[],
        help="Optional extra directory containing per-arm family_conditionals evaluation.json files.",
    )
    parser.add_argument(
        "--extra-fd-scores-dir",
        type=Path,
        action="append",
        default=[],
        help="Optional extra directory containing per-arm family_distribution evaluation.json files.",
    )
    parser.add_argument(
        "--general-plm-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional external_baseline run root(s) containing nested "
            "r1_ladder_and_conditionals_*.json / r1_family_distribution_*.json files. "
            "Defaults to results/shared/external_baseline/20261120213246_766da841c1a8 when present."
        ),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    general_dirs = list(args.general_plm_dir)
    if not general_dirs:
        default_v2 = args.repo_root / "results/shared/external_baseline/20261120213246_766da841c1a8"
        if default_v2.is_dir():
            general_dirs.append(default_v2)

    report = run_b1_reanalysis(
        args.repo_root,
        extra_fc_score_dirs=args.extra_fc_scores_dir,
        extra_fd_score_dirs=args.extra_fd_scores_dir,
        general_plm_dirs=general_dirs,
        n_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    digest = write_json_with_sha256(args.out, report)
    print(f"Wrote {args.out} (sha256={digest})")


if __name__ == "__main__":
    main()
