#!/usr/bin/env python3
"""Run fixed-20-AA-target SaProt (35M/650M/1.3B) 3Di structural correspondence causal control.

Evaluates ``aa_only``, ``correct_3di``, ``shuffled_3di`` (within-domain), and
``cross_domain_shuffled_3di`` on identical frozen SaProt weights, identical CATH-3D
domains, identical ``<mask>`` query positions, and the identical 20-AA marginal
cross-entropy loss (21 tokens/AA via ``logsumexp``; plus secondary ``20_3di_only``
sensitivity check), sealing ``r4_saprot_3di_aa_target_causal_control.json`` and
its ``.sha256`` sidecar.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.unified.saprot_3di_causal_control import (  # noqa: E402
    DEFAULT_COHORT_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SAPROT_ARMS,
    run_full_saprot_3di_causal_control,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        default=",".join(DEFAULT_SAPROT_ARMS),
        help="Comma-separated frozen SaProt arms (default: saprot-35m,saprot-650m,saprot-1.3b).",
    )
    parser.add_argument("--device", default="cuda:6", help="CUDA device (default: cuda:6).")
    parser.add_argument(
        "--cohort",
        type=Path,
        default=REPO_ROOT / DEFAULT_COHORT_PATH,
        help=f"CATH-3D cohort JSON (default: {DEFAULT_COHORT_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / DEFAULT_OUTPUT_PATH,
        help=f"Output artifact JSON path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--arm-cache-dir",
        type=Path,
        default=None,
        help="Optional directory for crash-safe per-arm JSON checkpoints (enables parallel multi-GPU runs).",
    )
    parser.add_argument(
        "--only-populate-cache",
        action="store_true",
        help="Only populate per-arm cache files without sealing the combined output JSON yet.",
    )
    parser.add_argument(
        "--max-domains",
        type=int,
        default=None,
        help="Optional cap on sorted CATH-3D domains (default: None = all 1,166 domains).",
    )
    parser.add_argument(
        "--secondary-max-domains",
        type=int,
        default=128,
        help="Domain count for secondary 20_3di_only sensitivity check (default: 128).",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--mask-stride", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-boot-snr", type=int, default=5000)
    parser.add_argument("--n-boot-spectral", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261120)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.monotonic()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    payload = run_full_saprot_3di_causal_control(
        root=REPO_ROOT,
        arms=arms,
        device=args.device,
        cohort_path=args.cohort,
        output_path=args.output,
        arm_cache_dir=args.arm_cache_dir,
        only_populate_cache=args.only_populate_cache,
        max_domains=args.max_domains,
        secondary_max_domains=args.secondary_max_domains,
        max_tokens=args.max_tokens,
        mask_stride=args.mask_stride,
        batch_size=args.batch_size,
        n_boot_snr=args.n_boot_snr,
        n_boot_spectral=args.n_boot_spectral,
        seed=args.seed,
    )
    gate = payload["verification_gate"]
    print(
        f"[saprot-3di-control] done in {time.monotonic() - t0:.1f}s | "
        f"status={payload['completion_status']} gate_passed={gate['passed']} "
        f"max_abs_diff={gate['max_abs_diff']:.2e}",
        flush=True,
    )
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
