#!/usr/bin/env python3
"""Pure-CPU assembler for the 15-model DPLM-2 (x4) + ProteinMPNN (x11) expanded
generation, folding, and R6 adaptation benchmark artifacts.

Writes NEW ``*_full.json`` (+ ``.sha256``) artifacts under
``results/extensions/unified_yardstick_20261124/`` without ever modifying any sealed
baseline JSON:

  - ``r5_experiment_4_1_unified_generation_full.json``
  - ``r5a_local_infilling_full.json``
  - ``r6_experiment_4_2_phase_space_full.json``
  - ``r6_experiment_4_3_native_likelihood_full.json``
  - ``r4_continuation_unified_full.json``
  - ``r6_dplm2_mpnn_adaptation_unified.json``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.memory_theory.io_utils import load_json_file  # noqa: E402
from src.capability.unified import yardstick_77_80_generation as ygen  # noqa: E402
from src.capability.unified.roster import UNIFIED_OUTPUT_DIR  # noqa: E402

ALL_BENCHMARKS = ("exp4_1", "r5a", "exp4_2", "exp4_3", "r4_cont", "r6_adapt")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--benchmarks",
        type=str,
        default=",".join(ALL_BENCHMARKS),
        help="Comma-separated subset of: exp4_1,r5a,exp4_2,exp4_3,r4_cont,r6_adapt",
    )
    ap.add_argument("--work-dir", type=Path, default=REPO_ROOT / ygen.WORK_DIR_REL)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / UNIFIED_OUTPUT_DIR)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--skip-external-oracles", action="store_true")
    ap.add_argument("--draws", type=int, default=ygen.DEFAULT_BOOTSTRAP_DRAWS)
    ap.add_argument("--fwl-draws", type=int, default=2000)
    ap.add_argument("--diamond-threads", type=int, default=32)
    ap.add_argument("--hmmscan-workers", type=int, default=16)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    unknown = set(selected) - set(ALL_BENCHMARKS)
    if unknown:
        raise ValueError(f"Unknown benchmarks: {sorted(unknown)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, str]] = {}

    if "exp4_1" in selected:
        doc = ygen.assemble_exp4_1_full(
            REPO_ROOT,
            args.work_dir,
            allow_partial=args.allow_partial,
            draws=args.draws,
            diamond_threads=args.diamond_threads,
            hmmscan_workers=args.hmmscan_workers,
            skip_external_oracles=args.skip_external_oracles,
        )
        docs["exp4_1"] = doc
        receipts["exp4_1"] = ygen.write_checked_artifact(
            out_dir / "r5_experiment_4_1_unified_generation_full.json", doc
        )
        cov = doc["coverage"]
        print(
            f"[assemble] exp4_1: sealed={cov['n_sealed_entries']} new={cov['n_new_loaded']}/{cov['n_new_expected']} "
            f"sha256={receipts['exp4_1']['sha256'][:12]}",
            flush=True,
        )

    if "r5a" in selected:
        doc = ygen.assemble_r5a_full(
            REPO_ROOT,
            args.work_dir,
            allow_partial=args.allow_partial,
            draws=args.draws,
        )
        docs["r5a"] = doc
        receipts["r5a"] = ygen.write_checked_artifact(
            out_dir / "r5a_local_infilling_full.json", doc
        )
        cov = doc["coverage"]
        print(
            f"[assemble] r5a: new={cov['n_new_loaded']}/{cov['n_new_expected']} "
            f"sha256={receipts['r5a']['sha256'][:12]}",
            flush=True,
        )

    if "exp4_2" in selected:
        doc = ygen.assemble_exp4_2_full(
            REPO_ROOT,
            args.work_dir,
            allow_partial=args.allow_partial,
        )
        docs["exp4_2"] = doc
        receipts["exp4_2"] = ygen.write_checked_artifact(
            out_dir / "r6_experiment_4_2_phase_space_full.json", doc
        )
        cov = doc["coverage"]
        print(
            f"[assemble] exp4_2: sealed={cov['n_sealed_plms']} new={cov['n_new_loaded_entries']}/{cov['n_new_expected_entries']} "
            f"missing_cells={cov['n_missing_cells']} sha256={receipts['exp4_2']['sha256'][:12]}",
            flush=True,
        )

    if "exp4_3" in selected:
        doc = ygen.assemble_exp4_3_full(
            REPO_ROOT,
            args.work_dir,
            allow_partial=args.allow_partial,
            n_draws=args.draws,
            fwl_draws=args.fwl_draws,
            diamond_threads=args.diamond_threads,
            hmmscan_workers=args.hmmscan_workers,
            skip_external_oracles=args.skip_external_oracles,
        )
        docs["exp4_3"] = doc
        receipts["exp4_3"] = ygen.write_checked_artifact(
            out_dir / "r6_experiment_4_3_native_likelihood_full.json", doc
        )
        cov = doc["coverage"]
        print(
            f"[assemble] exp4_3: sealed={cov['n_sealed_pools']} new={cov['n_new_scored_pools']} "
            f"sha256={receipts['exp4_3']['sha256'][:12]}",
            flush=True,
        )

    if "r4_cont" in selected:
        doc = ygen.assemble_r4_continuation_full(
            REPO_ROOT,
            args.work_dir,
            allow_partial=args.allow_partial,
            diamond_threads=args.diamond_threads,
            hmmscan_workers=args.hmmscan_workers,
            skip_external_oracles=args.skip_external_oracles,
        )
        docs["r4_cont"] = doc
        receipts["r4_cont"] = ygen.write_checked_artifact(
            out_dir / "r4_continuation_unified_full.json", doc
        )
        cov = doc["coverage"]
        print(
            f"[assemble] r4_cont: sealed={cov['n_sealed_entries']} new={cov['n_new_loaded']}/{cov['n_new_expected']} "
            f"sha256={receipts['r4_cont']['sha256'][:12]}",
            flush=True,
        )

    if "r6_adapt" in selected:
        def _get_or_load(key: str, filename: str) -> dict[str, Any] | None:
            if key in docs:
                return docs[key]
            p = out_dir / filename
            return load_json_file(p) if p.is_file() else None

        doc = ygen.assemble_r6_dplm2_mpnn_adaptation(
            REPO_ROOT,
            exp4_1_doc=_get_or_load("exp4_1", "r5_experiment_4_1_unified_generation_full.json"),
            r5a_doc=_get_or_load("r5a", "r5a_local_infilling_full.json"),
            exp4_2_doc=_get_or_load("exp4_2", "r6_experiment_4_2_phase_space_full.json"),
            exp4_3_doc=_get_or_load("exp4_3", "r6_experiment_4_3_native_likelihood_full.json"),
            r4_cont_doc=_get_or_load("r4_cont", "r4_continuation_unified_full.json"),
            draws=args.draws,
        )
        receipts["r6_adapt"] = ygen.write_checked_artifact(
            out_dir / "r6_dplm2_mpnn_adaptation_unified.json", doc
        )
        print(
            f"[assemble] r6_adapt: 5 adaptation axes written sha256={receipts['r6_adapt']['sha256'][:12]}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
