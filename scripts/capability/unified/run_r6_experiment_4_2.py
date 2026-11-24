#!/usr/bin/env python3
"""CPU assembly and statistical summary for Unified Experiment 4.2 (Eight-Domain Phase Space).

Reuses the cached local E20 confirmatory (``P``, ``H``, ``progen2-medium``, ``protgpt2``)
and ProGen3 robustness (``progen3-112m``) cells, verifies the ``< 1e-12`` reproduction gates
against ``confirmatory.json`` and ``robustness.json``, merges all newly folded H200 JSONL
summaries under ``<work-dir>/folded/``, computes dual-stratum phase-space trajectories and
9,999-draw BCa domain bootstrap contrasts, and writes
``results/extensions/unified_yardstick_20261124/r6_experiment_4_2_phase_space_unified.json``
plus its ``.sha256`` sidecar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.unified import r6_experiment_4_2 as exp42  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", type=Path, default=exp42.DEFAULT_WORK_DIR)
    ap.add_argument("--family-reference-dir", type=Path, default=exp42.DEFAULT_FAMILY_REF_DIR)
    ap.add_argument("--e20-per-sequence", type=Path, default=exp42.DEFAULT_E20_PER_SEQUENCE)
    ap.add_argument("--e20-confirmatory-json", type=Path, default=exp42.DEFAULT_E20_CONFIRMATORY_JSON)
    ap.add_argument("--pg3rb-per-sequence", type=Path, default=exp42.DEFAULT_PG3RB_PER_SEQUENCE)
    ap.add_argument("--pg3rb-robustness-json", type=Path, default=exp42.DEFAULT_PG3RB_ROBUSTNESS_JSON)
    ap.add_argument("--out", type=Path, default=exp42.DEFAULT_OUTPUT_JSON)
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow interim assembly before all 19 H200 models finish folding",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    folded_dir = args.work_dir / "folded"
    payload = exp42.assemble_phase_space_summary(
        family_ref_dir=args.family_reference_dir,
        e20_per_sequence=args.e20_per_sequence,
        e20_confirmatory_json=args.e20_confirmatory_json,
        pg3rb_per_sequence=args.pg3rb_per_sequence,
        pg3rb_robustness_json=args.pg3rb_robustness_json,
        h200_folded_dir=folded_dir if folded_dir.is_dir() else None,
        allow_partial=args.allow_partial,
    )
    out_path, sha_path, digest = exp42.write_unified_artifact(payload, args.out)
    cov = payload["coverage"]
    e20_gate = payload["reproduction_gates"]["e20_confirmatory_20261014"]
    pg3_gate = payload["reproduction_gates"]["progen3_robustness_20261015"]
    print(
        f"[exp4.2:assemble] wrote {out_path} (sha256={digest[:16]}..., sidecar={sha_path.name})\n"
        f"  PLMs: {cov['n_completed_plms']}/{cov['total_panel_plms']} complete "
        f"(pending={cov['n_pending_plms']}, missing_h200_cells={cov['missing_new_h200_cells']})\n"
        f"  E20 reproduction gate: passed={e20_gate['passed']} "
        f"max_abs_diff={e20_gate['max_abs_diff']:.3e} ({e20_gate['n_scalars_checked']} scalars)\n"
        f"  ProGen3 robustness gate: passed={pg3_gate['passed']} "
        f"max_abs_diff={pg3_gate['max_abs_diff']:.3e} ({pg3_gate['n_scalars_checked']} scalars)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
