"""Unified Merged Result 5 Assembler: Homologous Context Access Horizontal Benchmark.

Merges the three unified Result 5 component artifacts under
``results/extensions/unified_yardstick_20261124/`` into canonical cross-paradigm
benchmark documents for BOTH the core 65-model tier and the expanded 80-model tier:

1. **Core 65-Model Tier (``r5_homologous_context_merged.json`` + ``.sha256``)**:
   - ``main_table_800_units`` from ``r4_scoring_unified_65.json``
   - ``downstream_1_dms_mutation_ranking`` from ``r5_dms_context_ranking_unified.json``
   - ``downstream_2_prefix_continuation_and_folding`` from ``r4_continuation_unified.json``

2. **Expanded 80-Model Tier (``r5_homologous_context_merged_80.json`` + ``.sha256``)**:
   - ``main_table_800_units`` from ``r4_scoring_unified_80.json`` (65 core + 4 DPLM-2
     seq-only + 4 DPLM-2|struct + 11 ProteinMPNN = 84 horizontal rows)
   - ``downstream_1_dms_mutation_ranking`` from ``r5_dms_context_ranking_unified_80.json``
     (68 core + 19 DPLM-2/ProteinMPNN entries across 30-80% main + 5-band E09)
   - ``downstream_2_prefix_continuation_and_folding`` from ``r4_continuation_unified_full.json``
     (35 sealed unaligned prefix + 19 DPLM-2/ProteinMPNN entries)

Explicitly preserves both ``conditioning_regime`` (``sequence_only`` vs
``backbone_3d_*``) and ``input_alignment_format`` (``explicit_column_aligned`` vs
``unaligned_concat`` vs ``two_chain_spatial_offset_3d`` vs ``structural_3d_ss8_aligned_tracks``).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.capability.core.io import sha256_file
from src.capability.unified.master_summary import FORBIDDEN_INFRA_RE
from src.capability.unified.roster import REPO_ROOT, UNIFIED_OUTPUT_DIR

SCHEMA = "unified_r5_homologous_context_merged_v1"
SCHEMA_FULL = "unified_r5_homologous_context_merged_full_v1"
SCHEMA_80 = SCHEMA_FULL
DEFAULT_OUTPUT_NAME = "r5_homologous_context_merged.json"
DEFAULT_OUTPUT_NAME_FULL = "r5_homologous_context_merged_full.json"
DEFAULT_OUTPUT_NAME_80 = DEFAULT_OUTPUT_NAME_FULL


@dataclass(frozen=True)
class Result5ComponentSpec:
    key: str
    filename: str
    section: str
    producer: str


RESULT_5_COMPONENTS: tuple[Result5ComponentSpec, ...] = (
    Result5ComponentSpec(
        "main_table_800_units",
        "r4_scoring_unified_65.json",
        "Main Table: 65-Model 800-Unit 4-Band Unaligned/Aligned Homologous-Prefix Target Scoring",
        "scripts/capability/unified/run_r4_scoring_unified_65.py",
    ),
    Result5ComponentSpec(
        "downstream_1_dms_mutation_ranking",
        "r5_dms_context_ranking_unified.json",
        "Downstream 1: Homologous-Prefix DMS Mutation Ranking (30-80% Main + 5-Band 90-100%/self_copy)",
        "scripts/capability/unified/run_r5_dms_context_unified.py",
    ),
    Result5ComponentSpec(
        "downstream_2_prefix_continuation_and_folding",
        "r4_continuation_unified.json",
        "Downstream 2: Homologous-Prefix Continuation, Pfam Recall & Length-Stratified ESMFold2 Folding",
        "scripts/capability/unified/run_r4_continuation_unified.py",
    ),
)

RESULT_5_COMPONENTS_FULL: tuple[Result5ComponentSpec, ...] = (
    Result5ComponentSpec(
        "main_table_800_units",
        "r4_scoring_unified_full.json",
        "Main Table (Full 84-Model Tier): 84-Checkpoint (+4 DPLM-2|struct) 800-Unit 4-Band Target Scoring",
        "scripts/capability/unified/assemble_77_80_scoring_benchmarks.py",
    ),
    Result5ComponentSpec(
        "downstream_1_dms_mutation_ranking",
        "r5_dms_context_ranking_unified_full.json",
        "Downstream 1 (Full 84-Model Tier): 91-Entry Cross-Paradigm DMS Homolog-Context Ranking (30-80% + 5-Band E09)",
        "scripts/capability/unified/assemble_77_80_scoring_benchmarks.py",
    ),
    Result5ComponentSpec(
        "downstream_2_prefix_continuation_and_folding",
        "r4_continuation_unified_full.json",
        "Downstream 2 (Full 84-Model Tier): Homologous-Prefix (incl. DiMA x4) & 3D-Backbone Suffix Continuation + ESMFold2 Folding",
        "scripts/capability/unified/assemble_77_80_generation_benchmarks.py",
    ),
)
RESULT_5_COMPONENTS_80 = RESULT_5_COMPONENTS_FULL

INPUT_ALIGNMENT_FORMAT_TAXONOMY: dict[str, str] = {
    "explicit_column_aligned": (
        "Explicit 2D column-aligned MSA with Needleman-Wunsch '-' gap projection onto target columns "
        "(ONLY MSA-Transformer aligned=True)."
    ),
    "unaligned_concat": (
        "Unaligned sequence concatenation without gap characters: ProtGPT3-MSA (<no_gap> + <s>), "
        "Dayhoff-3B-MSA (@ + /), MSA-Transformer unaligned control (aligned=False), and all single-chain "
        "causal / bidirectional / 3Di / discrete diffusion / continuous latent diffusion (DiMA x4) / "
        "multimodal sequence-only models (including DPLM-2 seq-only)."
    ),
    "two_chain_spatial_offset_3d": (
        "Two-chain 3D backbone coordinate graph conditioning (homolog + target backbone with inter-chain "
        "spatial offset, used by ProteinMPNN inverse-folding checkpoints)."
    ),
    "structural_3d_ss8_aligned_tracks": (
        "Multimodal 3D backbone / SS8 / discrete structure-token track conditioning "
        "(ESM3-1.4B Route A comparator and DPLM-2|struct backbone-conditioned companions)."
    ),
}

CONDITIONING_REGIME_TAXONOMY: dict[str, str] = {
    "sequence_only": (
        "No 3D wild-type or homolog backbone coordinates supplied at inference (all 65 core arms + "
        "4 DPLM-2 seq-only arms + 4 DiMA continuous latent diffusion arms)."
    ),
    "unaligned_homolog_prefix": (
        "Unaligned homolog amino-acid prefix continuation without 3D coordinates "
        "(35 sealed continuation entries + 4 DPLM-2 seq-only + 4 DiMA prefix entries)."
    ),
    "backbone_3d_conditioned_oracle": (
        "Wild-type or homolog 3D backbone coordinates/structure tokens supplied directly as input "
        "(4 DPLM-2|struct companions + 11 ProteinMPNN checkpoints; strictly separated from sequence-only claims)."
    ),
    "backbone_3d_wt_prefix_suffix": (
        "Wild-type 3D backbone + N-terminal prefix conditioned C-terminal suffix design "
        "(4 DPLM-2 struct_conditioned + 11 ProteinMPNN checkpoints in Downstream 2)."
    ),
}


def _verify_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file():
        return {"exists": False, "sidecar_exists": sidecar.is_file(), "verified": False, "sha256": None}
    actual = sha256_file(path)
    if not sidecar.is_file():
        return {"exists": True, "sidecar_exists": False, "verified": False, "sha256": actual}
    declared = sidecar.read_text(encoding="utf-8").strip().split()[0]
    return {
        "exists": True,
        "sidecar_exists": True,
        "verified": actual == declared,
        "sha256": actual,
        "declared_sha256": declared,
    }


def _component_is_complete(key: str, doc: Mapping[str, Any], *, tier: str = "65") -> tuple[bool, str | None]:
    if tier == "65":
        if key == "main_table_800_units":
            cov = doc.get("coverage") or {}
            n_pend = int(cov.get("n_pending_arms", 0))
            ok = bool(doc.get("is_complete_65")) and n_pend == 0 and doc.get("completion_status") == "complete"
            return ok, (None if ok else f"r4_scoring_unified_65 has n_pending_arms={n_pend}")

        status = str(doc.get("completion_status", "complete"))
        cov = doc.get("coverage") or {}
        n_pend = int(cov.get("n_pending", cov.get("n_pending_arms", cov.get("n_pending_models", 0))))
        if status != "complete" or n_pend > 0:
            return False, f"{key} incomplete (completion_status={status!r}, n_pending={n_pend})"
        return True, None

    # 80-model tier completeness contracts matching yardstick_77_80_scoring.py & yardstick_77_80_generation.py
    if key == "main_table_800_units":
        status = str(doc.get("completion_status", ""))
        pend_p = list(doc.get("pending_primary_arms") or [])
        pend_s = list(doc.get("pending_struct_companions") or [])
        ok = status == "complete" and not pend_p and not pend_s
        return ok, (
            None
            if ok
            else f"r4_scoring_unified_80 incomplete (status={status!r}, pending_primary={len(pend_p)}, pending_struct={len(pend_s)})"
        )
    if key == "downstream_1_dms_mutation_ranking":
        status = str(doc.get("completion_status", ""))
        pending_lists = [
            len(doc.get(k) or [])
            for k in (
                "pending_base_1a_entries",
                "pending_base_1b_entries",
                "pending_new_1a_primary",
                "pending_new_1b_primary",
            )
        ]
        ok = status == "complete" and sum(pending_lists) == 0
        return ok, (
            None
            if ok
            else f"r5_dms_context_ranking_unified_80 incomplete (status={status!r}, pending_total={sum(pending_lists)})"
        )
    if key == "downstream_2_prefix_continuation_and_folding":
        cov = doc.get("coverage") or {}
        missing = list(cov.get("missing_entries") or [])
        ok = bool(cov.get("is_complete")) and len(missing) == 0
        return ok, (
            None
            if ok
            else f"r4_continuation_unified_full incomplete (is_complete={cov.get('is_complete')}, missing={len(missing)})"
        )
    raise ValueError(f"Unknown component key {key!r}")


def _tally_regime_and_alignment(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_cond = Counter()
    by_align = Counter()
    by_pair = Counter()
    for rec in records:
        cond = str(rec.get("conditioning_regime") or "sequence_only")
        align = str(rec.get("input_alignment_format") or "unaligned_concat")
        by_cond[cond] += 1
        by_align[align] += 1
        by_pair[f"{cond}::{align}"] += 1
    return {
        "by_conditioning_regime": dict(sorted(by_cond.items())),
        "by_input_alignment_format": dict(sorted(by_align.items())),
        "by_conditioning_and_alignment": dict(sorted(by_pair.items())),
    }


def _summarize_main_table(doc: Mapping[str, Any] | None) -> dict[str, Any]:
    if doc is None:
        return {"status": "pending", "reason": "r4_scoring_unified_65.json not yet written"}
    ok, reason = _component_is_complete("main_table_800_units", doc, tier="65")
    rows = doc.get("horizontal_benchmark_table") or []
    return {
        "status": "complete" if ok else "interim_pending",
        "pending_reason": reason,
        "schema": doc.get("schema"),
        "cohort": doc.get("cohort"),
        "coverage": doc.get("coverage"),
        "reproduction_gate": doc.get("reproduction_gate"),
        "exact_pll_diagnostics": doc.get("exact_pll_diagnostics"),
        "n_horizontal_rows": len(rows),
        "horizontal_benchmark_table": rows,
        "by_training_regime_and_alignment": doc.get("by_training_regime_and_alignment"),
        "paradigm_rollup": doc.get("paradigm_rollup"),
        "scaling_ladders": doc.get("scaling_ladders"),
        "simultaneous_verdict_summary": {
            band: {
                metric: {
                    "verdict_counts": (m_rec or {}).get("verdict_counts"),
                    "studentized_verdict_counts": (m_rec or {}).get("studentized_verdict_counts"),
                }
                for metric, m_rec in (b_rec or {}).items()
            }
            for band, b_rec in (doc.get("simultaneous_primary_65") or {}).items()
        },
        "companion_controls": doc.get("companion_controls"),
    }


def _summarize_main_table_80(doc: Mapping[str, Any] | None) -> dict[str, Any]:
    if doc is None:
        return {"status": "pending", "reason": "r4_scoring_unified_80.json not yet written"}
    ok, reason = _component_is_complete("main_table_800_units", doc, tier="80")
    rows = doc.get("horizontal_table") or []
    metrics_by_band = doc.get("metrics_by_band") or {}
    compact_bands: dict[str, Any] = {}
    for band_label, band_rec in metrics_by_band.items():
        if not isinstance(band_rec, Mapping):
            continue
        compact_bands[band_label] = {}
        for mkey, mrec in band_rec.items():
            if not isinstance(mrec, Mapping):
                continue
            b = mrec.get("bands") or {}
            compact_bands[band_label][mkey] = {
                "n_clusters": mrec.get("n_clusters"),
                "null_shift": mrec.get("null_shift"),
                "counts": b.get("counts"),
                "stratified_summary": mrec.get("stratified_summary"),
            }
    return {
        "status": "complete" if ok else "interim_pending",
        "pending_reason": reason,
        "schema": doc.get("schema"),
        "n_arms_primary": doc.get("n_arms_primary"),
        "n_arms_total_rows": doc.get("n_arms_total_rows", len(rows)),
        "pending_primary_arms": doc.get("pending_primary_arms") or [],
        "pending_struct_companions": doc.get("pending_struct_companions") or [],
        "n_horizontal_rows": len(rows),
        "counts_by_conditioning_and_alignment": _tally_regime_and_alignment(rows),
        "horizontal_table": rows,
        "metrics_by_band_summary": compact_bands,
    }


def _summarize_downstream_1(doc: Mapping[str, Any] | None) -> dict[str, Any]:
    if doc is None:
        return {"status": "pending", "reason": "r5_dms_context_ranking_unified.json not yet written"}
    ok, reason = _component_is_complete("downstream_1_dms_mutation_ranking", doc, tier="65")
    summary: dict[str, Any] = {
        "status": "complete" if ok else "interim_pending",
        "pending_reason": reason,
        "schema": doc.get("schema") or doc.get("schema_version"),
        "coverage": doc.get("coverage"),
        "reproduction_gates": doc.get("reproduction_gates") or doc.get("reproduction_gate"),
    }
    for key in (
        "protocol",
        "subtest_1a_rescue_30_80",
        "rescue_30_80",
        "subtest_1b_e09_five_band",
        "e09_five_band",
        "horizontal_summary",
        "by_training_regime_and_alignment",
        "verbatim_copy_degradation_summary",
        "paradigm_rollup",
        "simultaneous_bands",
    ):
        if key in doc:
            summary[key] = doc[key]
    return summary


def _summarize_downstream_1_80(doc: Mapping[str, Any] | None) -> dict[str, Any]:
    if doc is None:
        return {"status": "pending", "reason": "r5_dms_context_ranking_unified_80.json not yet written"}
    ok, reason = _component_is_complete("downstream_1_dms_mutation_ranking", doc, tier="80")
    sub1a = doc.get("subtest_1a_rescue_30_80") or {}
    sub1b = doc.get("subtest_1b_e09_five_band") or {}
    meta_1a = list((sub1a.get("entry_metadata") or {}).values())
    meta_1b = list((sub1b.get("entry_metadata") or {}).values())
    return {
        "status": "complete" if ok else "interim_pending",
        "pending_reason": reason,
        "schema": doc.get("schema"),
        "pending_base_1a_entries": doc.get("pending_base_1a_entries") or [],
        "pending_base_1b_entries": doc.get("pending_base_1b_entries") or [],
        "pending_new_1a_primary": doc.get("pending_new_1a_primary") or [],
        "pending_new_1b_primary": doc.get("pending_new_1b_primary") or [],
        "counts_by_conditioning_and_alignment": {
            "subtest_1a_rescue_30_80": _tally_regime_and_alignment(meta_1a),
            "subtest_1b_e09_five_band": _tally_regime_and_alignment(meta_1b),
        },
        "subtest_1a_rescue_30_80": sub1a,
        "subtest_1b_e09_five_band": sub1b,
    }


def _summarize_downstream_2(doc: Mapping[str, Any] | None) -> dict[str, Any]:
    if doc is None:
        return {"status": "pending", "reason": "r4_continuation_unified.json not yet written"}
    ok, reason = _component_is_complete("downstream_2_prefix_continuation_and_folding", doc, tier="65")
    summary: dict[str, Any] = {
        "status": "complete" if ok else "interim_pending",
        "pending_reason": reason,
        "schema": doc.get("schema") or doc.get("schema_version"),
        "coverage": doc.get("coverage"),
        "protocol": doc.get("protocol"),
    }
    for key in (
        "primary_12_family_summary",
        "primary_panel",
        "twilight_lt30_companion",
        "twilight_panel",
        "per_model_summary",
        "horizontal_table",
        "by_training_regime_and_alignment",
        "length_band_folding_summary",
        "paradigm_rollup",
        "paired_contrasts",
        "reproduction_gates",
    ):
        if key in doc:
            summary[key] = doc[key]
    return summary


def _summarize_downstream_2_80(doc: Mapping[str, Any] | None) -> dict[str, Any]:
    if doc is None:
        return {"status": "pending", "reason": "r4_continuation_unified_full.json not yet written"}
    ok, reason = _component_is_complete("downstream_2_prefix_continuation_and_folding", doc, tier="80")
    models_all = doc.get("models_all") or {}
    models_new = doc.get("models_new_dplm2_mpnn") or {}
    normalized_meta: list[dict[str, Any]] = []
    for ek, rec in models_all.items():
        if not isinstance(rec, Mapping):
            continue
        cond = str(rec.get("conditioning_regime") or "unaligned_homolog_prefix")
        if rec.get("input_alignment_format"):
            align = str(rec["input_alignment_format"])
        elif cond == "backbone_3d_wt_prefix_suffix":
            align = (
                "two_chain_spatial_offset_3d"
                if str(rec.get("paradigm") or "") == "inverse_folding_gnn"
                else "structural_3d_ss8_aligned_tracks"
            )
        elif str(ek).startswith("msa-transformer") and "aligned=True" in str(ek):
            align = "explicit_column_aligned"
        else:
            align = "unaligned_concat"
        normalized_meta.append(
            {
                "entry_key": str(ek),
                "conditioning_regime": cond,
                "input_alignment_format": align,
            }
        )
    return {
        "status": "complete" if ok else "interim_pending",
        "pending_reason": reason,
        "schema": doc.get("schema_version") or doc.get("schema"),
        "sealed_baseline_source": doc.get("sealed_baseline_source"),
        "coverage": doc.get("coverage"),
        "e11_reproduction_gate": doc.get("e11_reproduction_gate"),
        "n_models_all": len(models_all),
        "n_models_new_dplm2_mpnn": len(models_new),
        "counts_by_conditioning_and_alignment": _tally_regime_and_alignment(normalized_meta),
        "models_new_dplm2_mpnn": models_new,
        "models_all": models_all,
    }


def _extract_gate_statuses(
    loaded: Mapping[str, Mapping[str, Any]],
    specs: Sequence[Result5ComponentSpec],
) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    fname_by_key = {s.key: s.filename for s in specs}
    if "main_table_800_units" in loaded:
        g = loaded["main_table_800_units"].get("reproduction_gate") or {}
        if g:
            gates["r4_scoring_cached_29arm_parity"] = {
                "artifact": fname_by_key["main_table_800_units"],
                "passed": bool(g.get("passed")),
                "max_abs_diff": g.get("max_abs_diff"),
            }
    for comp_key in (
        "downstream_1_dms_mutation_ranking",
        "downstream_2_prefix_continuation_and_folding",
    ):
        if comp_key not in loaded:
            continue
        fname = fname_by_key[comp_key]
        doc = loaded[comp_key]
        raw = doc.get("reproduction_gates") or doc.get("reproduction_gate") or doc.get("e11_reproduction_gate")
        if isinstance(raw, Mapping):
            if "passed" in raw:
                gates[f"{comp_key}_parity"] = {
                    "artifact": fname,
                    "passed": bool(raw.get("passed")),
                    "max_abs_diff": raw.get("max_abs_diff"),
                }
            else:
                for sub_k, sub_v in raw.items():
                    if isinstance(sub_v, Mapping):
                        gates[f"{comp_key}_{sub_k}"] = {
                            "artifact": fname,
                            "passed": bool(sub_v.get("passed")),
                            "max_abs_diff": sub_v.get("max_abs_diff"),
                        }
    return gates


def _assemble_tier(
    *,
    tier: str,
    schema: str,
    specs: Sequence[Result5ComponentSpec],
    default_name: str,
    root: Path,
    out_dir: Path | None,
    out_path: Path | None,
    allow_pending: bool,
) -> dict[str, Any]:
    target_dir = Path(out_dir) if out_dir is not None else (root / UNIFIED_OUTPUT_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = Path(out_path) if out_path is not None else (target_dir / default_name)

    manifest: list[dict[str, Any]] = []
    loaded: dict[str, dict[str, Any]] = {}
    pending_reasons: list[str] = []

    for spec in specs:
        fpath = target_dir / spec.filename
        chk = _verify_sidecar(fpath)
        status = "complete"
        detail: str | None = None
        doc: dict[str, Any] | None = None

        if not chk["exists"]:
            status = "missing"
            detail = f"{spec.filename} is absent"
            pending_reasons.append(detail)
        elif not chk["verified"]:
            raise RuntimeError(
                f"SHA-256 sidecar verification FAILED for {fpath}: "
                f"actual={chk['sha256']} vs declared={chk.get('declared_sha256')}"
            )
        else:
            doc = json.loads(fpath.read_text(encoding="utf-8"))
            loaded[spec.key] = doc
            ok, reason = _component_is_complete(spec.key, doc, tier=tier)
            if not ok:
                status = "interim_pending"
                detail = reason
                pending_reasons.append(reason or f"{spec.filename} incomplete")

        manifest.append(
            {
                "component_key": spec.key,
                "filename": spec.filename,
                "section": spec.section,
                "producer": spec.producer,
                "exists": chk["exists"],
                "sidecar_verified": chk["verified"],
                "sha256": chk["sha256"],
                "size_bytes": int(fpath.stat().st_size) if chk["exists"] else None,
                "schema": (doc.get("schema") or doc.get("schema_version")) if doc else None,
                "status": status,
                "detail": detail,
            }
        )

    if pending_reasons and not allow_pending:
        raise RuntimeError(
            f"Strict Merged Result 5 (tier={tier}) build refused due to incomplete/pending component artifacts "
            "(pass --allow-pending for interim build): " + "; ".join(pending_reasons)
        )

    gates = _extract_gate_statuses(loaded, specs)
    if tier == "65":
        main_blk = _summarize_main_table(loaded.get("main_table_800_units"))
        d1_blk = _summarize_downstream_1(loaded.get("downstream_1_dms_mutation_ranking"))
        d2_blk = _summarize_downstream_2(loaded.get("downstream_2_prefix_continuation_and_folding"))
    else:
        main_blk = _summarize_main_table_80(loaded.get("main_table_800_units"))
        d1_blk = _summarize_downstream_1_80(loaded.get("downstream_1_dms_mutation_ranking"))
        d2_blk = _summarize_downstream_2_80(loaded.get("downstream_2_prefix_continuation_and_folding"))

    payload: dict[str, Any] = {
        "schema": schema,
        "tier": tier,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "completion_status": "complete" if not pending_reasons else "interim_pending",
        "n_components_total": len(specs),
        "n_components_complete": sum(1 for m in manifest if m["status"] == "complete"),
        "n_pending": len(pending_reasons),
        "pending_components": pending_reasons,
        "input_alignment_format_taxonomy": dict(INPUT_ALIGNMENT_FORMAT_TAXONOMY),
        "conditioning_regime_taxonomy": dict(CONDITIONING_REGIME_TAXONOMY),
        "component_manifest": manifest,
        "reproduction_gates": gates,
        "all_reproduction_gates_passed": all(bool(g.get("passed")) for g in gates.values()) if gates else True,
        "main_table_800_units": main_blk,
        "downstream_1_dms_mutation_ranking": d1_blk,
        "downstream_2_prefix_continuation_and_folding": d2_blk,
    }

    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if FORBIDDEN_INFRA_RE.search(text):
        raise ValueError(f"Refusing to write {dest.name} containing forbidden pod/node identifier")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(dest)
    digest = sha256_file(dest)
    dest.with_name(dest.name + ".sha256").write_text(f"{digest}  {dest.name}\n", encoding="utf-8")

    return {
        "tier": tier,
        "output_path": str(dest),
        "sha256": digest,
        "payload": payload,
    }


def build_r5_homologous_context_merged(
    *,
    root: Path = REPO_ROOT,
    out_dir: Path | None = None,
    out_path: Path | None = None,
    allow_pending: bool = False,
) -> dict[str, Any]:
    """Assemble ``r5_homologous_context_merged.json`` (+ ``.sha256``) for the core 65-model tier."""
    return _assemble_tier(
        tier="65",
        schema=SCHEMA,
        specs=RESULT_5_COMPONENTS,
        default_name=DEFAULT_OUTPUT_NAME,
        root=root,
        out_dir=out_dir,
        out_path=out_path,
        allow_pending=allow_pending,
    )


def build_r5_homologous_context_merged_full(
    *,
    root: Path = REPO_ROOT,
    out_dir: Path | None = None,
    out_path: Path | None = None,
    allow_pending: bool = False,
) -> dict[str, Any]:
    """Assemble ``r5_homologous_context_merged_full.json`` (+ ``.sha256``) for the expanded 84-model tier."""
    return _assemble_tier(
        tier="full",
        schema=SCHEMA_FULL,
        specs=RESULT_5_COMPONENTS_FULL,
        default_name=DEFAULT_OUTPUT_NAME_FULL,
        root=root,
        out_dir=out_dir,
        out_path=out_path,
        allow_pending=allow_pending,
    )


build_r5_homologous_context_merged_80 = build_r5_homologous_context_merged_full


__all__ = [
    "CONDITIONING_REGIME_TAXONOMY",
    "DEFAULT_OUTPUT_NAME",
    "DEFAULT_OUTPUT_NAME_80",
    "DEFAULT_OUTPUT_NAME_FULL",
    "FORBIDDEN_INFRA_RE",
    "INPUT_ALIGNMENT_FORMAT_TAXONOMY",
    "RESULT_5_COMPONENTS",
    "RESULT_5_COMPONENTS_80",
    "RESULT_5_COMPONENTS_FULL",
    "Result5ComponentSpec",
    "SCHEMA",
    "SCHEMA_80",
    "SCHEMA_FULL",
    "build_r5_homologous_context_merged",
    "build_r5_homologous_context_merged_80",
    "build_r5_homologous_context_merged_full",
]
