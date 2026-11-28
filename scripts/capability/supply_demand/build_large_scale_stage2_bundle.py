#!/usr/bin/env python3
"""Build large-scale (25M / 50M unique AA token) Stage-2 bundles for 150M / 650M runs.

Streams directly from ``output/stage2_cache/uniref50_cath_l50_384.parquet`` (with configurable
``--target-tokens``, e.g. ``25000000`` or ``50000000``), raises per-superfamily reservoir caps
so deep arms stay well-supplied, and optionally reuses pre-packaged benchmark assets from an
existing Stage-2 bundle to avoid redundant I/O.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Data/lzp/plm-capability-boundaries"),
        help="Repository/data root containing data/ and output/stage2_cache/.",
    )
    p.add_argument(
        "--prejoined-parquet",
        type=Path,
        default=None,
        help="Path to uniref50_cath_l50_384.parquet (default: <data-root>/output/stage2_cache/uniref50_cath_l50_384.parquet).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination directory for the large-scale Stage-2 bundle.",
    )
    p.add_argument(
        "--target-tokens",
        "--target-unique-tokens",
        dest="target_tokens",
        type=int,
        default=25_000_000,
        help="Target unique AA tokens per arm (e.g. 25000000 for 25M or 50000000 for 50M).",
    )
    p.add_argument(
        "--arm-families",
        type=str,
        default="256,1024,4096",
        help="Comma-separated CATH H-superfamily breadth arms (default: 256,1024,4096).",
    )
    p.add_argument(
        "--max-seqs-per-superfamily",
        type=int,
        default=16_384,
        help="Per-superfamily reservoir cap (raised for 25M/50M token budgets; default: 16384).",
    )
    p.add_argument(
        "--singleton-pool-cap",
        type=int,
        default=400_000,
        help="Unannotated UniRef50 singleton pool cap (default: 400000 for <=50M tokens).",
    )
    p.add_argument(
        "--reuse-benchmarks-from",
        type=Path,
        default=None,
        help="Optional existing Stage-2 bundle dir whose benchmarks/ folder can be copied instead of rebuilt.",
    )
    p.add_argument("--seed", type=int, default=20261128)
    p.add_argument("--skip-benchmarks", action="store_true")
    return p.parse_args(argv)


def build_bundle_command(args: argparse.Namespace) -> list[str]:
    data_root = Path(args.data_root).resolve()
    parquet_path = (
        Path(args.prejoined_parquet).resolve()
        if args.prejoined_parquet is not None
        else (data_root / "output/stage2_cache/uniref50_cath_l50_384.parquet")
    )
    base_script = Path(__file__).resolve().with_name("build_stage2_bundle.py")
    cmd = [
        sys.executable,
        str(base_script),
        "--data-root",
        str(data_root),
        "--prejoined-parquet",
        str(parquet_path),
        "--out-dir",
        str(Path(args.out_dir).resolve()),
        "--target-tokens",
        str(int(args.target_tokens)),
        "--arm-families",
        str(args.arm_families),
        "--allow-family-cap",
        "--max-seqs-per-superfamily",
        str(int(args.max_seqs_per_superfamily)),
        "--singleton-pool-cap",
        str(int(args.singleton_pool_cap)),
        "--seed",
        str(int(args.seed)),
    ]
    if args.skip_benchmarks or args.reuse_benchmarks_from is not None:
        cmd.append("--skip-benchmarks")
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cmd = build_bundle_command(args)
    print(f"[stage2-large-bundle] Executing: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    if args.reuse_benchmarks_from is not None:
        src_bench = Path(args.reuse_benchmarks_from).resolve() / "benchmarks"
        dst_bench = Path(args.out_dir).resolve() / "benchmarks"
        if src_bench.is_dir():
            dst_bench.mkdir(parents=True, exist_ok=True)
            for item in src_bench.iterdir():
                if item.is_file():
                    shutil.copy2(item, dst_bench / item.name)
            print(f"[stage2-large-bundle] Copied benchmarks from {src_bench} -> {dst_bench}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
