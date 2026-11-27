#!/usr/bin/env python3
"""Pure-CPU assembler CLI for the expanded 77/80-model scoring benchmarks (DPLM-2 + ProteinMPNN).

NEVER modifies or overwrites any already-sealed ``*_62.json`` / ``*_65.json`` artifact;
emits only new expanded ``*_77.json`` / ``*_80.json`` files with verified ``.sha256`` sidecars.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.unified.yardstick_77_80_scoring import (
    DEFAULT_NEW_SCORING_DIR,
    assemble_e01_contacts_77,
    assemble_family_memory_77,
    assemble_r1_proteingym_77,
    assemble_r2_biophysical_77,
    assemble_r2_r3_mid_vs_last_77,
    assemble_r3_epistasis_77,
    assemble_r4_scoring_unified_80,
    assemble_r5_dms_context_unified_80,
)
from src.capability.unified.roster import UNIFIED_OUTPUT_DIR

ALL_BENCHMARKS: tuple[str, ...] = (
    "r1",
    "family_memory",
    "r2",
    "e01_contacts",
    "r3_epistasis",
    "mid_vs_last",
    "r4_main",
    "r5_dms",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        default="all",
        help=f"Comma-separated list from {list(ALL_BENCHMARKS)} or 'all'.",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scoring-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Allow assembling partial outputs while H200 scoring jobs are still running.",
    )
    args = parser.parse_args()

    selected = (
        list(ALL_BENCHMARKS)
        if args.benchmarks.strip().lower() == "all"
        else [s.strip() for s in args.benchmarks.split(",") if s.strip()]
    )
    unknown = sorted(set(selected) - set(ALL_BENCHMARKS))
    if unknown:
        raise SystemExit(f"Unknown benchmark(s): {unknown}; expected subset of {list(ALL_BENCHMARKS)}")

    scoring_dir = args.scoring_dir or (args.root / DEFAULT_NEW_SCORING_DIR)
    out_dir = args.out_dir or (args.root / UNIFIED_OUTPUT_DIR)

    results = []
    for name in selected:
        print(f"[assemble-77-80] Running benchmark={name} ...", flush=True)
        if name == "r1":
            res = assemble_r1_proteingym_77(
                root=args.root,
                scoring_dir=scoring_dir,
                out_dir=out_dir,
                workers=args.workers,
                threads=args.threads,
                draws=args.draws,
                allow_pending=args.allow_pending,
            )
        elif name == "family_memory":
            res = assemble_family_memory_77(
                root=args.root,
                scoring_dir=scoring_dir,
                out_dir=out_dir,
                draws=args.draws,
                allow_pending=args.allow_pending,
            )
        elif name == "r2":
            res = assemble_r2_biophysical_77(
                root=args.root,
                scoring_dir=scoring_dir,
                out_dir=out_dir,
                workers=args.workers,
                threads=args.threads,
                draws=args.draws,
                allow_pending=args.allow_pending,
            )
        elif name == "e01_contacts":
            res = assemble_e01_contacts_77(
                root=args.root,
                scoring_dir=scoring_dir,
                out_dir=out_dir,
                workers=args.workers,
                allow_pending=args.allow_pending,
            )
        elif name == "r3_epistasis":
            res = assemble_r3_epistasis_77(
                root=args.root,
                scoring_dir=scoring_dir,
                out_dir=out_dir,
                workers=args.workers,
                threads=args.threads,
                allow_pending=args.allow_pending,
            )
        elif name == "mid_vs_last":
            res = assemble_r2_r3_mid_vs_last_77(
                root=args.root,
                scoring_dir=scoring_dir,
                out_dir=out_dir,
                workers=args.workers,
                threads=args.threads,
                draws=args.draws,
                allow_pending=args.allow_pending,
            )
        elif name == "r4_main":
            res = assemble_r4_scoring_unified_80(
                root=args.root,
                scoring_dir=scoring_dir,
                out_dir=out_dir,
                draws=args.draws,
                allow_pending=args.allow_pending,
            )
        elif name == "r5_dms":
            res = assemble_r5_dms_context_unified_80(
                root=args.root,
                scoring_dir=scoring_dir,
                out_dir=out_dir,
                draws=args.draws,
                allow_pending=args.allow_pending,
            )
        else:
            raise AssertionError(f"Unhandled benchmark {name}")
        results.append(res)
        print(f"[assemble-77-80] Done {name}: status={res['completion_status']}", flush=True)

    print(json.dumps({"results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
