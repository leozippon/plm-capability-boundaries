#!/usr/bin/env python3
"""Build the portable (<50 MB) Temporal + Fold-Disjoint + Family-Bucketed Open-Model Audit Bundle."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.open_model_audit.cohort import build_open_audit_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Data/lzp/plm-capability-boundaries"),
        help="Repository root containing data/, results/, output/stage2_cache/, external/tools/.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output/open_audit_bundle",
        help="Output directory for the portable <50 MB audit bundle.",
    )
    p.add_argument("--n-per-depth-bucket", type=int, default=256)
    p.add_argument("--n-seen-cath-matched", type=int, default=256)
    p.add_argument("--n-unseen-homolog", type=int, default=512)
    p.add_argument("--n-orphan-cath-fold", type=int, default=512)
    p.add_argument("--n-panel-natural", type=int, default=64)
    p.add_argument("--n-panel-denovo", type=int, default=64)
    p.add_argument("--min-len", type=int, default=50)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--seed", type=int, default=20261128)
    p.add_argument("--skip-xml-dates", action="store_true")
    p.add_argument("--skip-esmfold", action="store_true")
    p.add_argument("--fold-device", type=str, default="cuda:2")
    p.add_argument(
        "--denovo-pdb-cache",
        type=Path,
        default=Path("/tmp/open_model_audit_denovo_pdbs"),
        help="Reusable local cache directory for ESMFold2-folded de novo PDBs.",
    )
    p.add_argument(
        "--max-uniref50-records",
        type=int,
        default=None,
        help="Optional cap on UniRef50 records scanned (for fast smoke builds; omit for full 60.3M census).",
    )
    p.add_argument("--max-uniref90-scan", type=int, default=1_500_000)
    return p.parse_args()


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()
    manifest = build_open_audit_bundle(
        data_root=args.data_root,
        out_dir=args.out_dir,
        n_per_depth_bucket=args.n_per_depth_bucket,
        n_seen_cath_matched=args.n_seen_cath_matched,
        n_unseen_homolog=args.n_unseen_homolog,
        n_orphan_cath_fold=args.n_orphan_cath_fold,
        n_panel_natural=args.n_panel_natural,
        n_panel_denovo=args.n_panel_denovo,
        min_len=args.min_len,
        max_len=args.max_len,
        seed=args.seed,
        parse_xml_dates=not args.skip_xml_dates,
        fold_denovo=not args.skip_esmfold,
        fold_device=args.fold_device,
        denovo_pdb_cache_dir=args.denovo_pdb_cache,
        max_uniref50_records=args.max_uniref50_records,
        max_uniref90_scan=args.max_uniref90_scan,
    )
    elapsed = time.perf_counter() - t0
    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(args.out_dir.resolve()),
                "wall_seconds": round(elapsed, 2),
                "bundle_size_mb": manifest["bundle_size_mb"],
                "uniref50_total_records_scanned": manifest["uniref50_total_records_scanned"],
                "split_counts": manifest["split_counts"],
                "disjointness_checks": manifest["disjointness_checks"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
