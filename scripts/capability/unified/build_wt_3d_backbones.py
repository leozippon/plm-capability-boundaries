#!/usr/bin/env python3
"""Build or update the unified leakage-free 3D wild-type backbone cache (Results 1-5).

Inventories 100% of unique wild-type, template, parent, homolog, and unrelated
context sequences across R1, R2, R3, Result 4, and Result 5, resolves every
available backbone locally from sealed freeze PDBs, AlphaFold DB v6 (exact +
exact contiguous substring slicing), and experimental mmCIF archives into
content-addressed ``(L, 4, 3)`` ``[N, CA, C, O]`` ``.npz`` files, and emits a
prioritized ``pending_wt_esmfold2.jsonl`` (with overlapping 1,024-aa Kabsch
windows for ``L > 1024``) for H200 ``ESMFold2`` folding.

Also supports ``--ingest-folded <path>`` to ingest H200 ``ESMFold2`` folded
records (stitching ``>1024``-aa windows via exact proper-rotation Kabsch
superposition) and refresh ``manifest.json`` + ``.sha256`` sidecars.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.unified.wt_backbones_3d import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    build_wt_3d_backbone_cache,
    ingest_esmfold2_folded_records,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or update the unified leakage-free 3D WT backbone cache (CPU)."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / DEFAULT_CACHE_DIR,
        help=f"Output backbone cache directory (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel CPU workers for local PDB/mmCIF scanning (clamped [1, 16]; default: 8).",
    )
    parser.add_argument(
        "--tier1-only",
        action="store_true",
        help="Inventory only Tier-1 core WTs and primary 1-item contexts (skip extra budget-1024 items).",
    )
    parser.add_argument(
        "--ingest-folded",
        type=Path,
        action="append",
        default=None,
        help="Optional JSONL file or directory of H200 ESMFold2 folded backbone records to ingest & stitch.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.monotonic()

    if args.ingest_folded:
        summary = ingest_esmfold2_folded_records(
            args.ingest_folded,
            cache_dir=args.cache_dir,
            root=REPO_ROOT,
        )
        print(
            f"[wt-3d-backbones] ingested direct={summary['n_ingested_direct']} "
            f"stitched={summary['n_ingested_kabsch_stitched']} "
            f"max_rmsd={summary['max_stitch_overlap_rmsd_angstrom']:.3f}A "
            f"| totals={summary['totals']} in {time.monotonic() - t0:.1f}s",
            flush=True,
        )
        return 0

    manifest = build_wt_3d_backbone_cache(
        root=REPO_ROOT,
        cache_dir=args.cache_dir,
        workers=args.workers,
        include_r5_extra_b1024_contexts=not args.tier1_only,
    )
    tot = manifest["totals"]
    t1 = manifest["by_priority_tier"]["tier_1_r1_r4_wt"]
    t2 = manifest["by_priority_tier"]["tier_2_r5_targets_and_primary_ctx"]
    t3 = manifest["by_priority_tier"]["tier_3_r5_extra_contexts"]
    print(
        f"[wt-3d-backbones] completed in {time.monotonic() - t0:.1f}s | "
        f"total={tot['n_unique_sequences']} resolved={tot['n_resolved']} "
        f"pending_seqs={tot['n_pending_sequences']} pending_jobs={tot['n_pending_esmfold2_jobs']} "
        f"| tier1_r1_r4={t1['resolved']}/{t1['total']} (jobs={t1['pending_windows']}) "
        f"| tier2_r5_primary={t2['resolved']}/{t2['total']} (jobs={t2['pending_windows']}) "
        f"| tier3_r5_extra={t3['resolved']}/{t3['total']} (jobs={t3['pending_windows']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
