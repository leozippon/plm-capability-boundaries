#!/usr/bin/env python3
"""Assemble Merged Result 5 for the core 65-model and/or expanded 84-model (full) tiers.

Core 65-model tier (``r5_homologous_context_merged.json`` + ``.sha256``):
  - ``r4_scoring_unified_65.json``
  - ``r5_dms_context_ranking_unified.json``
  - ``r4_continuation_unified.json``

Expanded 84-model full tier (``r5_homologous_context_merged_full.json`` + ``.sha256``):
  - ``r4_scoring_unified_full.json``
  - ``r5_dms_context_ranking_unified_full.json``
  - ``r4_continuation_unified_full.json``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.unified.r5_homologous_context_merged import (  # noqa: E402
    build_r5_homologous_context_merged,
    build_r5_homologous_context_merged_full,
)
from src.capability.unified.roster import UNIFIED_OUTPUT_DIR  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / UNIFIED_OUTPUT_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--tier",
        choices=("65", "core_65", "full", "84", "80", "both"),
        default=None,
        help="Result 5 tier to assemble (default: 'both' when --out is omitted, else '65').",
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Allow interim merged build while any Result 5 component is still pending.",
    )
    args = parser.parse_args(argv)

    tier = args.tier if args.tier is not None else ("65" if args.out is not None else "both")
    if tier == "both" and args.out is not None:
        parser.error("--out cannot be combined with --tier both; pass --out-dir or select --tier 65|full.")

    tiers_to_run = ("65", "full") if tier == "both" else ("65" if tier in ("65", "core_65") else "full",)
    for t in tiers_to_run:
        builder = build_r5_homologous_context_merged if t == "65" else build_r5_homologous_context_merged_full
        res = builder(
            root=ROOT,
            out_dir=args.out_dir,
            out_path=args.out,
            allow_pending=args.allow_pending,
        )
        p = res["payload"]
        print(
            f"[r5-merged:{t}] status={p['completion_status']} "
            f"({p['n_components_complete']}/{p['n_components_total']} complete, "
            f"n_pending={p['n_pending']}, gates_passed={p['all_reproduction_gates_passed']})",
            flush=True,
        )
        print(f"[r5-merged:{t}] wrote {res['output_path']} (sha256={res['sha256'][:16]}...)", flush=True)


if __name__ == "__main__":
    main()
