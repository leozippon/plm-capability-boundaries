#!/usr/bin/env python3
"""Run Workstream R2/R3-Mid-vs-Last across all 62 single-sequence models on CPU.

Evaluates paired middle-layer ``R_mid`` (cols 0:512) vs final-layer ``R_last`` (cols 512:1024)
vs joint ``R_joint`` (cols 0:1024) across:
  - MegaScale Stability (79g CC + 101g full; C_evo + C_full)
  - Human Domainome Abundance (94g CC + 96g full; C_evo + C_full)
  - Double-Mutant Epistasis (61g, 7,805 cycles; all 3 rungs + top-rung MR_cycle)

Verifies strict ``< 1e-10`` parity of ``R_joint`` against sealed
``r2_biophysical_unified_62.json`` and ``r3_epistasis_and_contacts_unified_62.json``
without modifying any sealed artifact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.unified.r2_r3_mid_vs_last_yardstick import (
    DEFAULT_CELL_CACHE_REL,
    DEFAULT_OUTPUT_REL,
    build_r2_r3_middle_vs_last_62,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / DEFAULT_OUTPUT_REL)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / DEFAULT_CELL_CACHE_REL)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--blas-threads", type=int, default=2)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--force-refit", action="store_true")
    args = parser.parse_args()

    res = build_r2_r3_middle_vs_last_62(
        root=ROOT,
        out_path=args.out,
        cache_dir=args.cache_dir,
        max_workers=args.workers,
        blas_threads=args.blas_threads,
        draws=args.draws,
        seed=args.seed,
        force_refit=args.force_refit,
        verify_sealed_parity=True,
    )
    p = res["payload"]
    gate = p["reproduction_gate"]
    print(
        f"[r2-r3-mid-last] DONE in {p['elapsed_seconds']:.1f}s | arms={p['coverage']['n_arms']}/62 | "
        f"parity_gate={'PASSED' if gate['passed'] else 'FAIL'} (max_abs_diff={gate['max_abs_diff']:.2e})",
        flush=True,
    )
    print(
        f"[r2-r3-mid-last] wrote {res['output_path']} (sha256={res['sha256']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
