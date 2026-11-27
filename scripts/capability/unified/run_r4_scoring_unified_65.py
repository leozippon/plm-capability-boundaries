#!/usr/bin/env python3
"""CPU aggregator for Workstream R4-Score (65/65 unified unaligned homologous-prefix scoring).

Reads:
  - 29 cached protein arms (19 early EXP-R2-228 + 7 GP single-chain + 3 MSA at b=1024)
  - 36 newly scored H200 single-chain arms under ``results/extensions/unified_yardstick_20261124/r4_unaligned_scoring/``
    (14 Group A causal text arms + 22 Group B bidirectional / masked / 3Di / diffusion / ESM3 arms)

Writes:
  - ``results/extensions/unified_yardstick_20261124/r4_scoring_unified_65.json`` (+ ``.sha256``)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.unified.r4_scoring_yardstick import (
    DEFAULT_OUTPUT_REL,
    DEFAULT_WORK_SUBDIR,
    build_r4_scoring_unified_65,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / DEFAULT_WORK_SUBDIR)
    parser.add_argument("--out", type=Path, default=ROOT / DEFAULT_OUTPUT_REL)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Allow interim build while some of the 36 H200 arms are still running",
    )
    args = parser.parse_args()

    res = build_r4_scoring_unified_65(
        repo_root=ROOT,
        work_dir=args.work_dir,
        out_path=args.out,
        allow_pending=args.allow_pending,
        draws=args.draws,
        seed=args.seed,
    )
    p = res["payload"]
    cov = p["coverage"]
    gate = p["reproduction_gate"]
    print(
        f"[r4-scoring-65] status={p['completion_status']} "
        f"({cov['n_arms_present']}/{cov['n_total_roster_arms']} arms, "
        f"single={cov['n_single_sequence_present']}/62, msa={cov['n_msa_present']}/3, "
        f"parity_gate={'PASSED' if gate['passed'] else 'FAIL'} max_diff={gate['max_abs_diff']:.2e})",
        flush=True,
    )
    print(f"[r4-scoring-65] wrote {res['output_path']} (sha256={res['sha256'][:16]}...)", flush=True)


if __name__ == "__main__":
    main()
