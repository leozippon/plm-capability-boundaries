#!/usr/bin/env python3
"""Verify SHA-256 sidecars and build the unified master index & PROVENANCE.md.

Writes:
  - ``results/extensions/unified_yardstick_20261124/unified_master_summary.json`` (+ ``.sha256``)
  - ``results/extensions/unified_yardstick_20261124/PROVENANCE.md``

By default runs in strict 100%-completion mode (requiring all 17 canonical
artifacts present, sidecar-verified, and ``r6_experiment_4_2`` ``n_pending_plms == 0``).
Pass ``--allow-pending`` during development before all parallel tracks land.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.unified.master_summary import build_unified_master_summary  # noqa: E402
from src.capability.unified.roster import UNIFIED_OUTPUT_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / UNIFIED_OUTPUT_DIR,
        help="Unified results directory (default: results/extensions/unified_yardstick_20261124)",
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Allow interim build while Experiment 4.1 or 4.2 H200 tracks are still finishing",
    )
    args = parser.parse_args()

    res = build_unified_master_summary(
        root=ROOT,
        out_dir=args.out_dir,
        allow_pending=args.allow_pending,
    )
    p = res["payload"]
    print(
        f"[master-summary] status={p['completion_status']} "
        f"({p['n_artifacts_complete']}/{p['n_artifacts_total']} complete, "
        f"{p['n_sidecars_verified']}/{p['n_artifacts_present']} sidecars OK, "
        f"gates_passed={p['all_reproduction_gates_passed']})",
        flush=True,
    )
    print(f"[master-summary] wrote {res['summary_path']} (sha256={res['summary_sha256'][:16]}...)", flush=True)
    print(f"[master-summary] wrote {res['provenance_path']}", flush=True)
    if p["pending_artifacts"]:
        print("[master-summary] pending:", "; ".join(p["pending_artifacts"]), flush=True)


if __name__ == "__main__":
    main()
