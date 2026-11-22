#!/usr/bin/env python3
"""Aggregate completed general-PLM per-arm JSON artifacts alongside the frozen
33-arm causal baseline reference (read-only; never mutates the 33-arm
simultaneous band or any frozen panel artifact).

Features:
- Discovers and merges completed stages across BOTH V1
  (``results/shared/external_baseline/20261120133411_4956b7c48429``) and V2
  (``results/shared/external_baseline/20261120213246_766da841c1a8``) plus
  ``results/extensions/general_plm_20261120`` (or custom ``--results-dir`` roots).
- Stitches multi-shard ``r1_family_distribution_<arm>_shard*of*.json`` outputs
  into one unified per-arm family-distribution summary.
- Evaluates every completed R3 ``r3_pairwise_cycles_<arm>.npz`` on CPU for
  zero-shot Spearman vs measured epsilon AND exact nested-CV bidirectional
  cycle MSE / Spearman increments (``M_cycle``, ``R_cycle``, ``MR_cycle`` vs
  ``nuisance``, ``nuisance+profile``, ``nuisance+profile+couplings`` matching
  ``src/capability/interactions/pairwise_epistasis.py`` and
  ``results/extensions/explicit_statistics_lanes_20261110/interaction``).
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

from src.capability.general_plm.esm3_seqonly_eval import evaluate_esm3_seqonly_dir
from src.capability.general_plm.eval_utils import (
    atomic_write_json,
    cluster_bootstrap_mean,
    sha256_file,
)
from src.capability.general_plm.r3_cycle_eval import (
    PairwiseCycleContext,
    evaluate_r3_pairwise_cycles_npz,
    load_pairwise_cycle_context,
)
from src.capability.general_plm.registry import GENERAL_PLM_REGISTRY

SCHEMA = "general_plm_aggregated_summary_v2"

def _default_scan_dirs(root: Path) -> list[Path]:
    """Auto-discover all ``results/shared/external_baseline/2026112*`` campaigns + local extension dir."""
    ext_base = root / "results/shared/external_baseline"
    discovered = (
        sorted(p for p in ext_base.glob("2026112*") if p.is_dir())
        if ext_base.is_dir()
        else []
    )
    discovered.append(root / "results/extensions/general_plm_20261120")
    return discovered


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
        "r3_interaction_complete_case": (
            root
            / "results/extensions/explicit_statistics_lanes_20261110/interaction/panels/"
            "cycle_squared_error-likelihood-complete_case.json"
        ),
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


def _write_sha256_sidecar(path: Path) -> Path:
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return sidecar


def _evaluate_or_load_r3_cycle_eval(
    npz_path: Path,
    arm: str,
    *,
    out_dir: Path,
    ctx_box: list[PairwiseCycleContext | None],
    data_root: Path,
    force_refit: bool,
) -> dict[str, Any]:
    """Evaluate ``r3_pairwise_cycles_<arm>.npz`` on CPU, reusing a SHA256-matched cache if present."""
    cache_json = out_dir / f"r3_cycle_eval_{arm}.json"
    npz_sha = sha256_file(npz_path)
    if not force_refit and cache_json.is_file():
        try:
            cached = json.loads(cache_json.read_text(encoding="utf-8"))
            if cached.get("npz_sha256") == npz_sha and cached.get("arm") == arm:
                if not cache_json.with_name(cache_json.name + ".sha256").is_file():
                    _write_sha256_sidecar(cache_json)
                return cached
        except Exception:
            pass

    if ctx_box[0] is None:
        ctx_box[0] = load_pairwise_cycle_context(data_root)
    result = evaluate_r3_pairwise_cycles_npz(npz_path, arm, ctx=ctx_box[0])
    atomic_write_json(cache_json, result)
    _write_sha256_sidecar(cache_json)
    print(f"[summary:R3-eval] fitted {arm} from {npz_path} -> {cache_json}", flush=True)
    return result


def _evaluate_or_load_esm3_seqonly(
    esm3_dir: Path,
    *,
    out_dir: Path,
    ctx_box: list[PairwiseCycleContext | None],
    data_root: Path,
    force_refit: bool,
) -> dict[str, Any]:
    """Evaluate admitted ``esm3_1_4b_r123_b4`` R1/R2/R3 outputs on CPU with SHA256 caching."""
    cache_json = out_dir / "esm3_1_4b_seqonly_eval.json"
    meta_sha = sha256_file(esm3_dir / "esm3_r123_seqonly.json")
    if not force_refit and cache_json.is_file():
        try:
            cached = json.loads(cache_json.read_text(encoding="utf-8"))
            if cached.get("source_meta_sha256") == meta_sha:
                if not cache_json.with_name(cache_json.name + ".sha256").is_file():
                    _write_sha256_sidecar(cache_json)
                return cached
        except Exception:
            pass

    if ctx_box[0] is None:
        ctx_box[0] = load_pairwise_cycle_context(data_root)
    result = evaluate_esm3_seqonly_dir(esm3_dir, data_root, ctx=ctx_box[0])
    atomic_write_json(cache_json, result)
    _write_sha256_sidecar(cache_json)
    print(f"[summary:ESM3-eval] fitted esm3-1.4b from {esm3_dir} -> {cache_json}", flush=True)
    return result


def _rel_or_str(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _compact_r3_eval_summary(eval_doc: dict[str, Any]) -> dict[str, Any]:
    """Compact high-signal view of R3 nested-CV cycle evaluation embedded in the master summary."""
    channels = eval_doc.get("nested_cv_channels", {})
    compact_channels: dict[str, Any] = {}
    for ch_name in ("M_cycle", "R_cycle"):
        if ch_name not in channels:
            continue
        compact_channels[ch_name] = {}
        for rung_name, rdoc in channels[ch_name].items():
            compact_channels[ch_name][rung_name] = {
                "control_mse_kcal2": rdoc["control_mse_kcal2"],
                "model_only_mse_kcal2": rdoc["model_only_mse_kcal2"],
                "joint_mse_kcal2": rdoc["joint_mse_kcal2"],
                "control_spearman": rdoc["control_spearman"],
                "model_only_spearman": rdoc["model_only_spearman"],
                "joint_spearman": rdoc["joint_spearman"],
                "forward_model_given_controls": rdoc["forward_model_given_controls"],
                "reverse_controls_given_model": rdoc["reverse_controls_given_model"],
            }
    if "MR_cycle_over_nuisance_profile_couplings" in channels:
        compact_channels["MR_cycle_over_nuisance_profile_couplings"] = channels[
            "MR_cycle_over_nuisance_profile_couplings"
        ]
    return {
        "support": eval_doc.get("support"),
        "npz_sha256": eval_doc.get("npz_sha256"),
        "zero_shot": eval_doc.get("zero_shot"),
        "nested_cv_channels": compact_channels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Root directory to scan recursively for per-arm general-PLM outputs. "
            "May be passed multiple times. Defaults to V1 + V2 external_baseline dirs "
            "plus results/extensions/general_plm_20261120."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_ROOT / "results/extensions/general_plm_20261120",
    )
    parser.add_argument("--expect-file", default="general_plm_summary.json")
    parser.add_argument(
        "--skip-r3-cycle-fit",
        action="store_true",
        help="Skip CPU nested-CV fitting of R3 r3_pairwise_cycles_<arm>.npz files.",
    )
    parser.add_argument(
        "--force-refit-r3",
        action="store_true",
        help="Re-run CPU nested-CV fitting even when a SHA256-matched r3_cycle_eval_<arm>.json exists.",
    )
    args = parser.parse_args()

    scan_dirs = list(args.results_dir) if args.results_dir else _default_scan_dirs(DATA_ROOT)

    args.out.mkdir(parents=True, exist_ok=True)
    arms_summary: dict[str, dict[str, Any]] = {}
    famdist_shards: dict[str, dict[str, dict[str, Any]]] = {}
    r3_npz_by_arm: dict[str, Path] = {}
    esm3_seqonly_dirs: list[Path] = []
    companion_artifacts: dict[str, dict[str, Any]] = {}
    scanned_files = 0
    existing_scan_dirs: list[str] = []

    def _ensure_arm(arm: str) -> dict[str, Any]:
        spec = GENERAL_PLM_REGISTRY.get(arm)
        return arms_summary.setdefault(
            arm,
            {
                "arm": arm,
                "family": spec.family if spec else "unknown",
                "paradigm": spec.paradigm if spec else "unknown",
                "approx_params": spec.approx_params if spec else 0,
                "modules": {},
            },
        )

    for root_dir in scan_dirs:
        if not root_dir.is_dir():
            continue
        existing_scan_dirs.append(_rel_or_str(root_dir, DATA_ROOT))
        # Discover R3 cycle npz archives directly (in case a run has npz beside JSON or standalone).
        for npz_path in sorted(root_dir.rglob("r3_pairwise_cycles_*.npz")):
            arm_name = npz_path.stem.removeprefix("r3_pairwise_cycles_")
            if arm_name:
                r3_npz_by_arm[arm_name] = npz_path

        for path in sorted(root_dir.rglob("*.json")):
            if (
                path.name == args.expect_file
                or path.name.startswith("r3_cycle_eval_")
                or path.name == "esm3_1_4b_seqonly_eval.json"
            ):
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(doc, dict):
                continue

            schema = doc.get("schema", "")
            arm = doc.get("arm")
            rel_src = _rel_or_str(path, DATA_ROOT)

            if path.name == "esm3_r123_seqonly.json" and schema == "esm3_r123_seqonly_v1":
                esm3_seqonly_dirs.append(path.parent)
                continue

            # Index companion non-general-PLM V1/V2 artifacts (ESM3 R4/R5, R5 structure evals).
            if path.name == "esm3_r4_multimodal.json":
                companion_artifacts["esm3_1_4b_r4_route_a"] = {
                    "source": rel_src,
                    "schema": schema,
                    "n_units": doc.get("n_units") or doc.get("summary", {}).get("n_units"),
                }
                continue
            if path.name == "esm3_r5_generation.json":
                companion_artifacts["esm3_1_4b_r5_generation"] = {
                    "source": rel_src,
                    "schema": schema,
                }
                continue
            if isinstance(schema, str) and schema in (
                "general_plm_r5_structure_eval_v1",
                "r5_structure_evaluation_v1",
            ):
                inst = doc.get("esmfold2_instrument") or {}
                companion_artifacts[f"r5_structure_eval::{path.parent.name}"] = {
                    "source": rel_src,
                    "schema": schema,
                    "label": doc.get("label") or path.parent.name,
                    "n_unique_folded": inst.get("n_unique_folded") or doc.get("n_sequences_folded"),
                    "n_records_total": doc.get("n_records_total"),
                }
                continue

            if not isinstance(schema, str) or not schema.startswith("general_plm_") or not arm:
                continue

            scanned_files += 1
            entry = _ensure_arm(arm)

            if schema == "general_plm_r1_ladder_and_conditionals_v1":
                entry["modules"]["r1_ladder_and_conditionals"] = {
                    "source": rel_src,
                    "dtype_realised": doc.get("provenance", {}).get("dtype_realised"),
                    "mutation_ladder": doc.get("mutation_ladder", {}).get("summary"),
                    "matched_discrimination": doc.get("matched_discrimination", {}).get("tiers"),
                    "family_conditionals": doc.get("family_conditionals", {}).get("evaluation_summary"),
                }
            elif schema == "general_plm_r1_family_distribution_pll_v1":
                famdist_shards.setdefault(arm, {})[rel_src] = doc
            elif schema == "general_plm_r2_phenotypes_and_readouts_v1":
                entry["modules"]["r2_phenotypes_and_readouts"] = {
                    "source": rel_src,
                    "dtype_realised": doc.get("provenance", {}).get("dtype_realised"),
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
                sibling_npz = path.parent / f"r3_pairwise_cycles_{arm}.npz"
                if sibling_npz.is_file():
                    r3_npz_by_arm[arm] = sibling_npz
                entry["modules"]["r3_pairwise_and_contacts"] = {
                    "source": rel_src,
                    "dtype_realised": doc.get("provenance", {}).get("dtype_realised"),
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
                    "source": rel_src,
                    "dtype_realised": doc.get("provenance", {}).get("dtype_realised"),
                    "bands": doc.get("bands"),
                }
            elif schema == "general_plm_r5_infill_and_diffusion_gen_v1":
                d_mode = (doc.get("denovo_summary") or {}).get("diffusion_sampling_mode") or (
                    doc.get("ladder_summary") or {}
                ).get("diffusion_sampling_mode")
                r5_key = (
                    f"r5_infill_and_gen_{d_mode}"
                    if d_mode and d_mode != "max_confidence"
                    else "r5_infill_and_gen"
                )
                entry["modules"][r5_key] = {
                    "source": rel_src,
                    "dtype_realised": doc.get("provenance", {}).get("dtype_realised"),
                    "ladder_summary": doc.get("ladder_summary"),
                    "denovo_summary": doc.get("denovo_summary"),
                }

    for arm, doc_map in famdist_shards.items():
        arms_summary[arm]["modules"]["r1_family_distribution_pll"] = _merge_family_distribution_shards(
            list(doc_map.values())
        )

    # Evaluate completed R3 double-mutant cycle .npz archives on CPU.
    ctx_box: list[PairwiseCycleContext | None] = [None]
    r3_eval_arms: list[str] = []
    if not args.skip_r3_cycle_fit:
        for arm, npz_path in sorted(r3_npz_by_arm.items()):
            entry = _ensure_arm(arm)
            r3_mod = entry["modules"].setdefault(
                "r3_pairwise_and_contacts",
                {"source": _rel_or_str(npz_path, DATA_ROOT), "pairwise_cycles": {}},
            )
            eval_doc = _evaluate_or_load_r3_cycle_eval(
                npz_path,
                arm,
                out_dir=args.out,
                ctx_box=ctx_box,
                data_root=DATA_ROOT,
                force_refit=args.force_refit_r3,
            )
            r3_mod["pairwise_cycles"]["nested_cv_epistasis"] = _compact_r3_eval_summary(eval_doc)
            r3_mod["pairwise_cycles"]["zero_shot_spearman_vs_measured_epsilon"] = (
                eval_doc["zero_shot"]["all_64_groups_raw_epsilon"]["within_background_spearman"]
            )
            r3_mod["pairwise_cycles"]["eval_detail_file"] = _rel_or_str(
                args.out / f"r3_cycle_eval_{arm}.json", DATA_ROOT
            )
            r3_eval_arms.append(arm)

        # Evaluate admitted ESM3-1.4B sequence-only R1/R2/R3 outputs and include directly in arms.
        if esm3_seqonly_dirs:
            esm3_dir = esm3_seqonly_dirs[-1]
            esm3_eval = _evaluate_or_load_esm3_seqonly(
                esm3_dir,
                out_dir=args.out,
                ctx_box=ctx_box,
                data_root=DATA_ROOT,
                force_refit=args.force_refit_r3,
            )
            esm3_entry = arms_summary.setdefault(
                "esm3-1.4b",
                {
                    "arm": "esm3-1.4b",
                    "family": esm3_eval["family"],
                    "paradigm": esm3_eval["paradigm"],
                    "approx_params": esm3_eval["approx_params"],
                    "modules": {},
                },
            )
            r1_e = esm3_eval["r1_ladder_and_conditionals"]
            esm3_entry["modules"]["r1_ladder_and_conditionals"] = {
                "source": _rel_or_str(esm3_dir / "esm3_r123_seqonly.json", DATA_ROOT),
                "eval_detail_file": _rel_or_str(args.out / "esm3_1_4b_seqonly_eval.json", DATA_ROOT),
                "dtype_realised": esm3_eval.get("dtype_realised"),
                "mutation_ladder": r1_e["mutation_ladder"]["summary"],
                "mutation_ladder_ridge": r1_e["mutation_ladder"]["bidirectional_ridge"],
                "matched_discrimination": r1_e["matched_discrimination"]["tiers"],
                "family_conditionals": r1_e["family_conditionals"]["evaluation_summary"],
                "family_conditionals_quartiles": r1_e["family_conditionals"]["conservation_quartiles_var_to_cons"],
            }
            r2_e = esm3_eval["r2_phenotypes_and_readouts"]["lanes"]
            esm3_entry["modules"]["r2_phenotypes_and_readouts"] = {
                "source": _rel_or_str(esm3_dir / "esm3_r123_seqonly.json", DATA_ROOT),
                "eval_detail_file": _rel_or_str(args.out / "esm3_1_4b_seqonly_eval.json", DATA_ROOT),
                "dtype_realised": esm3_eval.get("dtype_realised"),
                "lanes": {
                    lane: {
                        "n_variants": ldoc.get("n_variants"),
                        "zero_shot_spearman": ldoc.get("zero_shot_spearman"),
                        "readouts_granularity": ldoc.get("readouts_granularity"),
                        "ridge_summary": (ldoc.get("bidirectional_ridge") or {}).get("summary"),
                        "paired_increments": (ldoc.get("bidirectional_ridge") or {}).get("paired_increments"),
                    }
                    for lane, ldoc in r2_e.items()
                },
            }
            r3_e = esm3_eval["r3_pairwise_and_contacts"]["pairwise_cycles"]
            esm3_entry["modules"]["r3_pairwise_and_contacts"] = {
                "source": _rel_or_str(esm3_dir / "r3_cycle_epistasis.jsonl", DATA_ROOT),
                "eval_detail_file": _rel_or_str(args.out / "esm3_1_4b_seqonly_eval.json", DATA_ROOT),
                "dtype_realised": esm3_eval.get("dtype_realised"),
                "pairwise_cycles": {
                    **{k: v for k, v in r3_e.items() if k != "nested_cv_epistasis"},
                    "nested_cv_epistasis": _compact_r3_eval_summary(r3_e["nested_cv_epistasis"]),
                },
                "structural_contacts": None,
            }
            r3_eval_arms.append("esm3-1.4b")

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
        "scanned_directories": existing_scan_dirs,
        "scanned_files": scanned_files,
        "r3_cycle_evaluated_arms": r3_eval_arms,
        "n_arms_present": len(ordered_arms),
        "arms": ordered_arms,
        "companion_artifacts": companion_artifacts,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path = args.out / args.expect_file
    atomic_write_json(out_path, payload)
    _write_sha256_sidecar(out_path)
    print(
        f"[summary] aggregated {scanned_files} files across {len(ordered_arms)} arms "
        f"(R3 cycle evals: {r3_eval_arms}) -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
