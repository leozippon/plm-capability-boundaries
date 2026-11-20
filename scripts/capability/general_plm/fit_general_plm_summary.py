#!/usr/bin/env python3
"""Aggregate completed general-PLM per-arm JSON artifacts alongside the frozen
33-arm causal baseline reference (read-only; never mutates the 33-arm
simultaneous band or any frozen panel artifact).

Also stitches multi-shard ``r1_family_distribution_<arm>_shard*of*.json``
outputs into one unified per-arm family-distribution summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("TRANSFER_DATA_ROOT") or ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.general_plm.eval_utils import atomic_write_json, cluster_bootstrap_mean
from src.capability.general_plm.registry import GENERAL_PLM_REGISTRY

SCHEMA = "general_plm_aggregated_summary_v1"


def _load_frozen_33_arm_reference(root: Path) -> dict[str, Any]:
    """Read-only pointer summary of frozen 33-arm causal baselines for side-by-side display."""
    ref: dict[str, Any] = {
        "note": "Read-only reference from frozen 33-arm causal panel; simultaneous bands untouched.",
        "sources": {},
    }
    candidates = {
        "r1_zero_shot": root / "results/extensions/statistical_ladder_20261106/full/zero-shot.json",
        "r1_family_distribution": root / "results/extensions/family_distribution_20261106/analysis/analysis.json",
        "r1_family_conditionals": root / "results/extensions/family_conditionals_20261107/analysis/analysis.json",
    }
    for label, path in candidates.items():
        if path.is_file():
            ref["sources"][label] = str(path.relative_to(root))
    return ref


def _merge_family_distribution_shards(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine 1 or more family-distribution shards for one arm into one deduplicated summary."""
    by_unit: dict[str, dict[str, Any]] = {}
    for doc in docs:
        for u in doc.get("units", ()):
            by_unit[u["unit_id"]] = u
    eval_units = [u for u in by_unit.values() if u.get("partition") == "evaluation"]
    col_vals = [float(u["column_contrast"]["nats_per_residue"]) for u in eval_units]
    comp_vals = [float(u["composition_contrast"]["nats_per_residue"]) for u in eval_units]
    ids = [u["unit_id"] for u in eval_units]
    return {
        "n_shards_merged": len(docs),
        "n_units_total": len(by_unit),
        "n_eval_units": len(eval_units),
        "score_semantics": docs[0].get("score_semantics"),
        "column_null_delta_nats_per_res": cluster_bootstrap_mean(col_vals, ids, seed=20261106),
        "composition_null_delta_nats_per_res": cluster_bootstrap_mean(comp_vals, ids, seed=20261107),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DATA_ROOT / "results/extensions/general_plm_20261120",
        help="Root directory scanned recursively for per-arm general-PLM JSON outputs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_ROOT / "results/extensions/general_plm_20261120",
    )
    parser.add_argument("--expect-file", default="general_plm_summary.json")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    arms_summary: dict[str, dict[str, Any]] = {}
    famdist_shards: dict[str, list[dict[str, Any]]] = {}
    scanned_files = 0

    if args.results_dir.is_dir():
        for path in sorted(args.results_dir.rglob("*.json")):
            if path.name == args.expect_file:
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            schema = doc.get("schema", "")
            arm = doc.get("arm")
            if not isinstance(schema, str) or not schema.startswith("general_plm_") or not arm:
                continue
            scanned_files += 1
            spec = GENERAL_PLM_REGISTRY.get(arm)
            entry = arms_summary.setdefault(
                arm,
                {
                    "arm": arm,
                    "family": spec.family if spec else "unknown",
                    "paradigm": spec.paradigm if spec else "unknown",
                    "approx_params": spec.approx_params if spec else 0,
                    "modules": {},
                },
            )
            if schema == "general_plm_r1_ladder_and_conditionals_v1":
                entry["modules"]["r1_ladder_and_conditionals"] = {
                    "source": str(path),
                    "dtype_realised": doc.get("provenance", {}).get("dtype_realised"),
                    "mutation_ladder": doc.get("mutation_ladder", {}).get("summary"),
                    "matched_discrimination": doc.get("matched_discrimination", {}).get("tiers"),
                    "family_conditionals": doc.get("family_conditionals", {}).get("evaluation_summary"),
                }
            elif schema == "general_plm_r1_family_distribution_pll_v1":
                famdist_shards.setdefault(arm, []).append(doc)
            elif schema == "general_plm_r2_phenotypes_and_readouts_v1":
                entry["modules"]["r2_phenotypes_and_readouts"] = {
                    "source": str(path),
                    "lanes": {
                        lane: {
                            "n_variants": ldoc.get("n_variants"),
                            "zero_shot_spearman": ldoc.get("zero_shot_spearman"),
                            "ridge_summary": (ldoc.get("bidirectional_ridge") or {}).get("summary"),
                        }
                        for lane, ldoc in doc.get("lanes", {}).items()
                    },
                }
            elif schema == "general_plm_r3_pairwise_pll_and_contacts_v1":
                entry["modules"]["r3_pairwise_and_contacts"] = {
                    "source": str(path),
                    "pairwise_cycles": {
                        k: v
                        for k, v in (doc.get("pairwise_cycles") or {}).items()
                        if k != "assays"
                    },
                    "structural_contacts": (doc.get("structural_contacts") or {}).get(
                        "summary_auc_minus_half"
                    ),
                }
            elif schema == "general_plm_r4_multiseq_context_v1":
                key = f"r4_b{doc.get('budget_tokens')}" + (
                    "_unaligned" if not doc.get("aligned_msa", True) else ""
                )
                entry["modules"][key] = {
                    "source": str(path),
                    "bands": doc.get("bands"),
                }
            elif schema == "general_plm_r5_infill_and_diffusion_gen_v1":
                entry["modules"]["r5_infill_and_gen"] = {
                    "source": str(path),
                    "ladder_summary": doc.get("ladder_summary"),
                    "denovo_summary": doc.get("denovo_summary"),
                }

    for arm, docs in famdist_shards.items():
        arms_summary[arm]["modules"]["r1_family_distribution_pll"] = _merge_family_distribution_shards(docs)

    ordered_arms = [
        arms_summary[a]
        for a in sorted(
            arms_summary,
            key=lambda x: (arms_summary[x]["family"], arms_summary[x]["approx_params"], x),
        )
    ]

    payload = {
        "schema": SCHEMA,
        "roster": "general_plm_20261120_exploratory",
        "frozen_33_arm_panel_mutated": False,
        "frozen_33_arm_reference": _load_frozen_33_arm_reference(DATA_ROOT),
        "scanned_files": scanned_files,
        "n_arms_present": len(ordered_arms),
        "arms": ordered_arms,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(args.out / args.expect_file, payload)
    print(
        f"[summary] aggregated {scanned_files} files across {len(ordered_arms)} arms -> "
        f"{args.out / args.expect_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
