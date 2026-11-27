#!/usr/bin/env python3
"""CPU annotation & unified 31-model aggregation for Workstream R4-Continuation.

Runs:
1. Historical E11 ``progen2-medium`` reproduction gate (``< 1e-12`` vs frozen comparison JSON).
2. Own-context DIAMOND prompt-copy alignment (``e <= 0.001``, ``>= 50%`` rule).
3. UniRef50 DIAMOND novelty (reusing ``exp4_1_cache`` SHA-256 cache).
4. Pinned Pfam-A ``hmmscan --cut_ga`` target-family recall (reusing ``exp4_1_cache``).
5. Paired 10,000-draw cluster bootstrap contrasts across yield, Pfam recall, pLDDT, pTM, and
   target-WT TM-score, writing ``r4_continuation_unified.json`` (+ ``.sha256``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.unified import r4_continuation_yardstick as r4c  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", type=Path, default=r4c.DEFAULT_WORK_DIR)
    ap.add_argument("--homologs-json", type=Path, default=r4c.DEFAULT_HOMOLOGS_JSON)
    ap.add_argument("--out", type=Path, default=r4c.DEFAULT_OUTPUT_JSON)
    ap.add_argument("--diamond-threads", type=int, default=64)
    ap.add_argument("--hmmscan-workers", type=int, default=32)
    ap.add_argument("--skip-annotations", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = r4c.assemble_r4_continuation_summary(
        work_dir=args.work_dir,
        homologs_json=args.homologs_json,
        run_annotations=not args.skip_annotations,
        allow_partial=args.allow_partial,
        diamond_threads=args.diamond_threads,
        hmmscan_workers=args.hmmscan_workers,
    )
    out_path, sha_path, digest = r4c.write_unified_artifact(payload, args.out)
    cov = payload["coverage"]
    gate = payload["reproduction_gates"]["e11_historical_progen2_medium_20261011"]
    print(
        f"[r4_cont:assemble] wrote {out_path} (sha256={digest[:16]}..., sidecar={sha_path.name})\n"
        f"  Models: {cov['n_completed_models']}/{cov['total_models']} complete "
        f"(pending={cov['n_pending_models']}, folded={cov['n_folded_products']})\n"
        f"  E11 reproduction gate: passed={gate['passed']} "
        f"max_abs_diff={gate['max_abs_diff']:.3e} ({gate['n_scalars_checked']} scalars)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
