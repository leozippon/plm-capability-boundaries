#!/usr/bin/env python3
"""Run unified 62-model R3 Contact Discrimination & Bidirectional Ablation (CPU).

Discovers all canonical ``position_likelihood_v1`` E01 contact extraction
directories across legacy (``results/external_baseline/20261008*``,
``20261010*``) and newly pulled general-PLM / ESM3 extractions
(``results/extensions/unified_yardstick_20261124/e01_contact_extractions/``),
executes the exact 0-based contact discrimination and nested-CV bidirectional
ablation across All (N=28), Deep (Neff/L >= 5, N=13), and Shallow (Neff/L < 5,
N=15) strata, computes independent two-sample family-bootstrap Deep-vs-Shallow
MSA contrasts, and writes all 7 JSON artifacts + ``.sha256`` sidecars:

  - contact_discrimination_unified_62{,_deep,_shallow}.json (+ .sha256)
  - contact_bidirectional_unified_62{,_deep,_shallow}.json (+ .sha256)
  - contacts_unified_62.json (+ .sha256)
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

from src.capability.unified.contacts_yardstick import (  # noqa: E402
    DEFAULT_LEGACY_EXTRACTION_ROOTS,
    DEFAULT_NEW_EXTRACTION_ROOT,
    run_contacts_unified_62,
)
from src.capability.unified.roster import UNIFIED_OUTPUT_DIR  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unified 62-model R3 contact discrimination & bidirectional yardstick (CPU)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Max CPU worker threads (clamped to [1, 4]; default: 4).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / UNIFIED_OUTPUT_DIR,
        help=f"Output directory (default: {UNIFIED_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--extra-extraction-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Additional directory containing posresp_extract_* subdirs and optional "
            f"refused_moe_layout.json (default includes {DEFAULT_NEW_EXTRACTION_ROOT})."
        ),
    )
    parser.add_argument(
        "--legacy-only",
        action="store_true",
        help="Scan only legacy 20261008/20261010 E01 extraction directories (for parity/smoke checks).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workers = max(1, min(int(args.workers), 4))
    t0 = time.monotonic()

    if args.legacy_only:
        roots = [REPO_ROOT / rel for rel in DEFAULT_LEGACY_EXTRACTION_ROOTS]
    elif args.extra_extraction_root:
        roots = [
            *(REPO_ROOT / rel for rel in DEFAULT_LEGACY_EXTRACTION_ROOTS),
            REPO_ROOT / DEFAULT_NEW_EXTRACTION_ROOT,
            *args.extra_extraction_root,
        ]
    else:
        roots = None

    result = run_contacts_unified_62(
        root=REPO_ROOT,
        out_dir=args.out_dir,
        extraction_roots=roots,
        workers=workers,
        write_outputs=True,
    )
    acct = result["master"]["roster_accounting"]
    elapsed = time.monotonic() - t0
    print(
        f"[unified-contacts] completed in {elapsed:.1f}s | "
        f"62-arm accounting: full_alpha={acct['n_admitted_full_alphabet']}, "
        f"thin_bpe={acct['n_admitted_thin_bpe']}, "
        f"refused_<8refs={acct['n_refused_below_8_refs']}, "
        f"refused_moe={acct['n_refused_moe_prefix_layout']}, "
        f"not_yet={acct['n_not_yet_extracted']} (total={acct['n_accounted_total']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
