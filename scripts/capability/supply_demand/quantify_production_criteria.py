#!/usr/bin/env python3
"""Quantify Section-IV production engineering criteria across Stage-1, Stage-2, Probing, and Open-PLM suites."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.memory_theory.io_utils import write_json_with_sha256
from src.capability.supply_demand.production_criteria import synthesize_production_criteria


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage1-summary",
        type=Path,
        default=REPO_ROOT / "output/stage1_wsd/stage1_combined_wave3_summary.json",
        help="Path to calibrated Stage-1 parametric capacity summary JSON.",
    )
    p.add_argument(
        "--stage2-summary",
        type=Path,
        default=REPO_ROOT / "output/stage2_inversion/stage2_wave3_analysis_summary.json",
        help="Optional Stage-2 supply-demand analysis summary JSON.",
    )
    p.add_argument(
        "--stage2-bundle-manifest",
        type=Path,
        default=REPO_ROOT / "output/stage2_bundle/bundle_manifest.json",
        help="Optional Stage-2 bundle manifest JSON for exact family counts M.",
    )
    p.add_argument(
        "--probing-dir",
        type=Path,
        default=REPO_ROOT / "output/probing_all43",
        help="Optional directory containing layerwise probing checkpoints/*.probe.json.",
    )
    p.add_argument(
        "--stage2-large-dir",
        type=Path,
        default=REPO_ROOT / "output/stage2_large_25m_results",
        help="Optional directory containing 25M Stage-2 large-scale run receipts.",
    )
    p.add_argument(
        "--open-model-audit-dir",
        type=Path,
        default=REPO_ROOT / "output/open_model_audit",
        help="Optional directory containing Open-PLM audit report JSONs.",
    )
    p.add_argument(
        "--multimodal-3d-dir",
        type=Path,
        default=REPO_ROOT / "output/multimodal_3d",
        help="Optional directory containing Multimodal 3D conditioning receipts.",
    )
    p.add_argument(
        "--supply-ablations-dir",
        type=Path,
        default=REPO_ROOT / "output/supply_stage2_muon",
        help="Optional directory containing supply-side Muon/vocab/mask ablation receipts.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "output/stage2_inversion/section4_production_criteria.json",
        help="Output JSON path (also writes .sha256 sidecar).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = synthesize_production_criteria(
        stage1_summary_path=args.stage1_summary,
        stage2_summary_path=args.stage2_summary if args.stage2_summary.is_file() else None,
        stage2_bundle_manifest_path=args.stage2_bundle_manifest if args.stage2_bundle_manifest.is_file() else None,
        probing_dir=args.probing_dir if args.probing_dir.is_dir() else None,
        stage2_large_dir=args.stage2_large_dir,
        open_model_audit_dir=args.open_model_audit_dir,
        multimodal_3d_dir=args.multimodal_3d_dir,
        supply_ablations_dir=args.supply_ablations_dir,
    )
    write_json_with_sha256(args.out, report)
    profiles = report["calibrated_supply_profiles"]
    probing = report["probing_erasure_synthesis"]
    print(
        json.dumps(
            {
                "status": "ok",
                "out": str(args.out),
                "objectives_calibrated": sorted(profiles.keys()),
                "n_scenarios": len(report["production_scenarios"]),
                "probing_checkpoints": probing.get("n_checkpoints", 0) if isinstance(probing, dict) else 0,
                "optional_suite_status": {k: v["status"] for k, v in report["optional_suites"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
