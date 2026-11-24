"""Unified Master Index & Provenance Builder for ``results/extensions/unified_yardstick_20261124/``.

Verifies SHA-256 sidecars across all 17 canonical JSON artifacts spanning Results 1–4,
validates completion gates (including ``r5_experiment_4_1_unified_generation.json`` and
full 25-arm ``r6_experiment_4_2_phase_space_unified.json`` with ``n_pending_plms == 0``
in strict mode), and writes:

  - ``results/extensions/unified_yardstick_20261124/unified_master_summary.json`` (+ ``.sha256``)
  - ``results/extensions/unified_yardstick_20261124/PROVENANCE.md``

Strictly pure-Python / stdlib + lightweight JSON reading (zero torch imports, zero pod/node names).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.capability.core.io import sha256_file
from src.capability.unified.roster import (
    ARM_METADATA,
    EXPANDED_32_ARMS,
    PARADIGM_ORDER,
    PREREGISTERED_33_ARMS,
    REPO_ROOT,
    UNIFIED_3_MSA_CONTEXT_ARMS,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_65_ARMS,
    UNIFIED_OUTPUT_DIR,
    arms_by_paradigm,
)

SCHEMA = "unified_master_summary_v1"

# Guard against accidentally writing disposable pod or cluster node identifiers.
FORBIDDEN_INFRA_RE = re.compile(
    r"\b(?:[a-z0-9-]+-master-\d+|[a-z0-9-]+-worker-\d+|worker-\d+\.k8s|10\.\d+\.\d+\.\d+)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ArtifactSpec:
    filename: str
    section: str
    role: str
    producer: str
    required_complete: bool = True


CANONICAL_ARTIFACTS: tuple[ArtifactSpec, ...] = (
    # Result 1: ProteinGym DMS Yardstick & Adversarial Pairs
    ArtifactSpec(
        "r1_proteingym_unified_62.json",
        "result_1",
        "Unified 62-model ProteinGym DMS zero-shot, bidirectional P/PEG, 29-arm +R bridge & 5-category functional breakdown",
        "scripts/capability/unified/run_r1_unified_yardstick.py",
    ),
    ArtifactSpec(
        "r1_supplementary_c0_ladder_62.json",
        "result_1",
        "Unified 62-model ProteinGym C0/BPL supplementary ladder (744-col family vs 396-col historical band)",
        "scripts/capability/unified/run_r1_unified_yardstick.py",
    ),
    ArtifactSpec(
        "r1_adversarial_pairs_62.json",
        "result_1",
        "Unified 62-model ProteinGym adversarial pair contrasts & T-match breakdown",
        "scripts/capability/unified/run_r1_unified_yardstick.py",
    ),
    # Result 2: Merged Experiment 2.1 Biophysical Transfer, Calibration & Scaling
    ArtifactSpec(
        "r2_biophysical_unified_62.json",
        "result_2",
        "Unified 62/62 M vs R biophysical transfer, calibration MSE, and 13-ladder scaling divergence on MegaScale & Human Domainome",
        "scripts/capability/unified/run_r2_experiment_2_1.py",
    ),
    # Result 3: Double-Mutant Epistasis & E01 Contact Propagation
    ArtifactSpec(
        "r3_epistasis_and_contacts_unified_62.json",
        "result_3",
        "Unified 62/62 double-mutant thermodynamic cycle epistasis (M_cycle, R_cycle, MR_cycle) & contact summary",
        "scripts/capability/unified/run_r3_unified_yardstick.py",
    ),
    ArtifactSpec(
        "contacts_unified_62.json",
        "result_3",
        "Unified 62-model E01 contact propagation master summary across All (28), Deep (13), and Shallow (15) MSA strata",
        "scripts/capability/unified/run_contacts_unified_yardstick.py",
    ),
    ArtifactSpec(
        "contact_discrimination_unified_62.json",
        "result_3",
        "E01 3D contact discrimination across all 28 Potts-admitted references",
        "scripts/capability/unified/run_contacts_unified_yardstick.py",
    ),
    ArtifactSpec(
        "contact_discrimination_unified_62_deep.json",
        "result_3",
        "E01 3D contact discrimination across 13 Deep-MSA references",
        "scripts/capability/unified/run_contacts_unified_yardstick.py",
    ),
    ArtifactSpec(
        "contact_discrimination_unified_62_shallow.json",
        "result_3",
        "E01 3D contact discrimination across 15 Shallow-MSA references",
        "scripts/capability/unified/run_contacts_unified_yardstick.py",
    ),
    ArtifactSpec(
        "contact_bidirectional_unified_62.json",
        "result_3",
        "E01 bidirectional contact ablation vs explicit Potts couplings (all 28 references)",
        "scripts/capability/unified/run_contacts_unified_yardstick.py",
    ),
    ArtifactSpec(
        "contact_bidirectional_unified_62_deep.json",
        "result_3",
        "E01 bidirectional contact ablation vs explicit Potts couplings (13 Deep-MSA references)",
        "scripts/capability/unified/run_contacts_unified_yardstick.py",
    ),
    ArtifactSpec(
        "contact_bidirectional_unified_62_shallow.json",
        "result_3",
        "E01 bidirectional contact ablation vs explicit Potts couplings (15 Shallow-MSA references)",
        "scripts/capability/unified/run_contacts_unified_yardstick.py",
    ),
    # Result 4: Associative Memory, Context, Generation (Exp 4.1, 4.2, 4.3), & Alignment
    ArtifactSpec(
        "family_memory_unified_62.json",
        "result_4_memory",
        "Unified 62-arm family conditionals & 46-arm family distribution associative memory metrics",
        "scripts/capability/unified/run_memory_and_generation_yardstick.py",
    ),
    ArtifactSpec(
        "r4_r5_r6_unified_summary.json",
        "result_4_context_and_alignment",
        "Unified R4 sequence/3D/SS8 context access, R5 summary, and R6 post-training alignment contrasts",
        "scripts/capability/unified/run_memory_and_generation_yardstick.py",
    ),
    ArtifactSpec(
        "r5_experiment_4_1_unified_generation.json",
        "result_4_exp_4_1",
        "Experiment 4.1: 29-checkpoint (32 mode-entries) + 2-control unconditional de novo generation novelty, Pfam & ESMFold2 census",
        "scripts/capability/unified/run_r5_experiment_4_1.py",
    ),
    ArtifactSpec(
        "r6_experiment_4_2_phase_space_unified.json",
        "result_4_exp_4_2",
        "Experiment 4.2: 23-PLM + Potts MCMC (P) + Natural (H) 8-domain family phase-space guided decoding & TM-score evaluation",
        "scripts/capability/unified/run_r6_experiment_4_2.py",
    ),
    ArtifactSpec(
        "r6_experiment_4_3_native_likelihood_unified.json",
        "result_4_exp_4_3",
        "Experiment 4.3: 30-checkpoint (33-pool) native self-likelihood ranking, repetitiveness trap & adverse selection inside <20%/<30% novel pools",
        "scripts/capability/unified/run_r6_experiment_4_3.py",
    ),
)


def verify_sidecar(path: Path) -> dict[str, Any]:
    """Verify ``<path>.sha256`` matches the bytes on disk."""
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
        "verified": (actual == declared),
        "sha256": actual,
        "declared_sha256": declared,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if FORBIDDEN_INFRA_RE.search(text):
        raise ValueError("Refusing to write master summary containing forbidden pod/node identifier")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def _summarize_r1(docs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    r1 = docs["r1_proteingym_unified_62.json"]
    l1 = r1["layer1_zero_shot"]
    l2 = r1["layer2_core_bidirectional"]
    rb = r1["layer2_representation_bridge"]
    return {
        "n_models": int(r1["n_models"]),
        "n_expanded_arms_with_R": int(rb["n_expanded_arms_with_R"]),
        "includes_esm3_1_4b_R": bool(rb.get("includes_esm3_1_4b")),
        "functional_categories": list(l1.get("functional_categories") or []),
        "c0_ladder_n_models": int(docs["r1_supplementary_c0_ladder_62.json"]["n_models"]),
        "adversarial_pairs_n_models": int(docs["r1_adversarial_pairs_62.json"]["n_models"]),
        "reproduction_gate_max_diff": _extract_r1_gate(r1),
        "primary_scalar_baselines": list((l2.get("primary_scalar_baselines") or {}).keys()),
    }


def _extract_r1_gate(r1: dict[str, Any]) -> dict[str, Any]:
    rg = r1.get("reproduction_gate") or {}
    details = rg.get("details") or {}
    max_diff = float(rg["max_abs_diff"]) if isinstance(rg.get("max_abs_diff"), (int, float)) else 0.0
    for arm_rec in details.values():
        for sub_k in ("batched_solver_max_abs_diff", "direct_sl_fit_max_abs_diff"):
            sub = arm_rec.get(sub_k) or {}
            if isinstance(sub, dict):
                for v in sub.values():
                    if isinstance(v, (int, float)):
                        max_diff = max(max_diff, float(v))
    if "passed" in rg:
        passed = bool(rg["passed"])
    else:
        passed = rg.get("status") == "passed"
    return {
        "passed": passed,
        "arms_checked": sorted(details.keys()),
        "max_abs_diff": max_diff,
    }


def _summarize_r2(docs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    r2 = docs["r2_biophysical_unified_62.json"]
    cov = r2["coverage_summary"]
    ms_evo = r2["datasets"]["megascale"]["primary_evolutionary_control"]["cc_79_profile_potts_gemme"]
    dm_evo = r2["datasets"]["domainome"]["primary_evolutionary_control"]["cc_94_profile_potts_gemme"]
    ms_full = r2["datasets"]["megascale"]["full_nuisance_evolutionary_control"]["cc_79_top_rung_6_designs"]
    dm_full = r2["datasets"]["domainome"]["full_nuisance_evolutionary_control"]["cc_94_top_rung_6_designs"]
    s2 = r2["datasets"]["megascale"]["full_nuisance_evolutionary_control"]["full_101_S_to_S2_62_M"]
    return {
        "coverage": {
            "megascale_M": cov["megascale"]["n_arms_M"],
            "megascale_R": cov["megascale"]["n_arms_R"],
            "domainome_M": cov["domainome"]["n_arms_M"],
            "domainome_R": cov["domainome"]["n_arms_R"],
        },
        "dual_protease_anchor": {
            "all_backgrounds_pearson_r": r2["instrument_anchor"]["megascale_dual_protease_reliability"][
                "endpoint_qualification_all_backgrounds"
            ]["pearson_r"]["point"],
            "natural_backgrounds_pearson_r": r2["instrument_anchor"]["megascale_dual_protease_reliability"][
                "endpoint_qualification_natural_backgrounds"
            ]["pearson_r"]["point"],
            "cohort_101_pooled_pearson_r": r2["instrument_anchor"]["megascale_dual_protease_reliability"][
                "cohort_101_pooled_trypsin_vs_chymotrypsin"
            ]["pearson_r"],
        },
        "megascale_cc_79_C_evo_headline": {
            "baseline_C_levels": {k: ms_evo["baseline_C_levels"][k]["point"] for k in ("ranking", "calibrated_error", "transfer_error")},
            "plus_M_calibrated_nat_pos": ms_evo["dimension_1_quantitative_error"]["calibrated_error_62_M"]["plus_M_given_C"]["counts"]["native"]["positive"],
            "plus_M_transfer_nat_pos": ms_evo["dimension_1_quantitative_error"]["transfer_error_62_M"]["plus_M_given_C"]["counts"]["native"]["positive"],
            "plus_MR_calibrated_nat_pos": ms_evo["dimension_1_quantitative_error"]["representation_R"]["calibrated_error"]["plus_MR_given_C"]["counts"]["native"]["positive"],
            "plus_MR_transfer_nat_pos": ms_evo["dimension_1_quantitative_error"]["representation_R"]["transfer_error"]["plus_MR_given_C"]["counts"]["native"]["positive"],
            "esmc_6b_C_M_R": ms_evo["per_arm_summary"]["esmc-6b"]["C_M_R"],
        },
        "megascale_101g_S_to_S2_62_M_studentized_pos": {
            k: s2[k]["counts"]["studentized"]["positive"]
            for k in (
                "matched_ranking",
                "profile_ranking",
                "matched_calibrated_error",
                "profile_calibrated_error",
                "matched_transfer_error",
                "profile_transfer_error",
            )
        },
        "megascale_cc_79_C_full_transfer_nat_pos": {
            "plus_M": ms_full["dimension_1_quantitative_error"]["transfer_error_62_M"]["plus_M_given_C"]["counts"]["native"]["positive"],
            "plus_MR": ms_full["dimension_1_quantitative_error"]["representation_R"]["transfer_error"]["plus_MR_given_C"]["counts"]["native"]["positive"],
        },
        "domainome_cc_94_C_evo_headline": {
            "baseline_C_levels": {k: dm_evo["baseline_C_levels"][k]["point"] for k in ("ranking", "calibrated_error", "transfer_error")},
            "plus_M_calibrated_nat_pos": dm_evo["dimension_1_quantitative_error"]["calibrated_error_62_M"]["plus_M_given_C"]["counts"]["native"]["positive"],
            "plus_M_transfer_nat_pos": dm_evo["dimension_1_quantitative_error"]["transfer_error_62_M"]["plus_M_given_C"]["counts"]["native"]["positive"],
            "plus_MR_calibrated_nat_pos": dm_evo["dimension_1_quantitative_error"]["representation_R"]["calibrated_error"]["plus_MR_given_C"]["counts"]["native"]["positive"],
            "plus_MR_transfer_nat_pos": dm_evo["dimension_1_quantitative_error"]["representation_R"]["transfer_error"]["plus_MR_given_C"]["counts"]["native"]["positive"],
            "esmc_6b_C_M_R": dm_evo["per_arm_summary"]["esmc-6b"]["C_M_R"],
        },
        "domainome_cc_94_C_full_transfer_nat_pos": {
            "plus_M_calibrated": dm_full["dimension_1_quantitative_error"]["calibrated_error_62_M"]["plus_M_given_C"]["counts"]["native"]["positive"],
            "plus_M_transfer": dm_full["dimension_1_quantitative_error"]["transfer_error_62_M"]["plus_M_given_C"]["counts"]["native"]["positive"],
            "plus_MR_transfer": dm_full["dimension_1_quantitative_error"]["representation_R"]["transfer_error"]["plus_MR_given_C"]["counts"]["native"]["positive"],
        },
    }


def _summarize_r3(docs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    r3 = docs["r3_epistasis_and_contacts_unified_62.json"]
    ct = docs["contacts_unified_62.json"]
    ep = r3["epistasis"]
    mr = ep["MR_cycle_62_over_nuisance_profile_couplings"]
    mr_mse = (mr.get("mse") or {}).get("MR_given_controls") or {}
    ra = ct["roster_accounting"]
    return {
        "epistasis_62": {
            "n_arms": int(r3["roster"]["n_single_sequence_arms"]),
            "support": ep["support"],
            "MR_cycle_MSE_states_native": mr_mse.get("states_native"),
            "MR_cycle_MSE_states_studentized": mr_mse.get("states_studentized"),
            "MR_cycle_MSE_states_marginal": mr_mse.get("states_marginal"),
        },
        "contacts_e01_62": {
            "n_roster_total": int(ra["n_roster_total"]),
            "n_admitted_full_alphabet": int(ra["n_admitted_full_alphabet"]),
            "n_admitted_thin_bpe": int(ra["n_admitted_thin_bpe"]),
            "n_refused_below_8_refs": int(ra["n_refused_below_8_refs"]),
            "n_refused_moe_prefix_layout": int(ra["n_refused_moe_prefix_layout"]),
            "depth_strata": sorted(ct["depth_strata"].keys()),
        },
    }


def _summarize_exp4_1(doc: dict[str, Any] | None) -> dict[str, Any]:
    if doc is None:
        return {"status": "pending", "reason": "r5_experiment_4_1_unified_generation.json not yet written"}
    proto = doc.get("protocol") or {}
    stage3 = (doc.get("stage3_cross_paradigm_novel_foldability_ladder") or {}).get(
        "ranked_by_gated_novel_foldable_yield"
    ) or []
    top5 = [
        {
            "entry_key": r["entry_key"],
            "paradigm_group": r["paradigm_group"],
            "gated_yield_of_pool": (r.get("gated_lt30_h30_yield_of_pool") or {}).get("mean"),
            "gated_mean_plddt": (r.get("gated_lt30_h30_mean_plddt") or {}).get("mean"),
            "pfam_complete_domain_rate": r.get("pfam_complete_domain_rate_all"),
        }
        for r in stage3[:5]
    ]
    return {
        "status": "complete",
        "n_unconditional_checkpoints": proto.get("n_unconditional_checkpoints"),
        "n_mode_entries": proto.get("n_mode_entries_including_dplm_ablation"),
        "n_controls": proto.get("n_controls"),
        "n_total_rows": proto.get("n_total_rows"),
        "top5_by_gated_novel_foldable_yield": top5,
    }


def _summarize_exp4_2(doc: dict[str, Any] | None) -> dict[str, Any]:
    if doc is None:
        return {"status": "pending", "reason": "r6_experiment_4_2_phase_space_unified.json missing"}
    cov = doc.get("coverage") or {}
    is_complete = bool(cov.get("is_complete_panel")) and int(cov.get("n_pending_plms", 1)) == 0
    return {
        "status": "complete" if is_complete else "partial_pending_h200",
        "total_panel_plms": cov.get("total_panel_plms"),
        "n_completed_plms": cov.get("n_completed_plms"),
        "n_pending_plms": cov.get("n_pending_plms"),
        "reference_arms_present": cov.get("reference_arms_present"),
        "domains": doc.get("domains"),
        "paradigm_groups_present": sorted((doc.get("paradigm_summary") or {}).keys()),
    }


def _summarize_exp4_3(doc: dict[str, Any]) -> dict[str, Any]:
    cs = doc["cross_model_summary"]
    return {
        "status": "complete",
        "n_unique_checkpoints": int(doc["n_unique_checkpoints"]),
        "n_primary_pools": int(doc["n_primary_pools"]),
        "strictly_novel_lt20_natural_entropy": {
            "n_evaluable": cs["strictly_novel_lt20_natural_entropy"]["n_pools_evaluable"],
            "distinct_5mer_spearman": cs["strictly_novel_lt20_natural_entropy"]["spearman_counts"]["distinct_5mer"],
            "plddt_spearman": cs["strictly_novel_lt20_natural_entropy"]["spearman_counts"]["plddt"],
            "ptm_spearman": cs["strictly_novel_lt20_natural_entropy"]["spearman_counts"]["ptm"],
            "top_25pct_plddt": {
                k: cs["strictly_novel_lt20_natural_entropy"]["truncation_selection_counts"]["plddt"]["top_25pct"][k]
                for k in ("n_ci_adverse", "n_point_adverse", "n_ci_positive", "median_gain_over_random")
            },
            "top_25pct_ptm": {
                k: cs["strictly_novel_lt20_natural_entropy"]["truncation_selection_counts"]["ptm"]["top_25pct"][k]
                for k in ("n_ci_adverse", "n_point_adverse", "n_ci_positive", "median_gain_over_random")
            },
            "top_25pct_distinct_5mer": {
                k: cs["strictly_novel_lt20_natural_entropy"]["truncation_selection_counts"]["distinct_5mer"]["top_25pct"][k]
                for k in ("n_ci_adverse", "n_point_adverse", "n_ci_positive", "median_gain_over_random")
            },
        },
        "novel_lt20_unfiltered_distinct_5mer_top25": {
            k: cs["novel_lt20_unfiltered"]["truncation_selection_counts"]["distinct_5mer"]["top_25pct"][k]
            for k in ("n_evaluable", "n_ci_adverse", "n_point_adverse", "median_gain_over_random")
        },
        "whole_pool_top25_medians": {
            "plddt_gain": cs["whole_pool"]["truncation_selection_counts"]["plddt"]["top_25pct"]["median_gain_over_random"],
            "ptm_gain": cs["whole_pool"]["truncation_selection_counts"]["ptm"]["top_25pct"]["median_gain_over_random"],
            "distinct_5mer_gain": cs["whole_pool"]["truncation_selection_counts"]["distinct_5mer"]["top_25pct"]["median_gain_over_random"],
        },
        "objective_tuned_companion_status": doc["companion_verifications"][
            "objective_tuned_base_likelihood_20261114"
        ]["status"],
    }


def _render_provenance_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Unified Cross-Paradigm Yardstick Provenance (`unified_yardstick_20261124`)",
        "",
        f"Generated UTC: `{payload['generated_utc']}`  ",
        f"Completion status: `{payload['completion_status']}` (`{payload['n_artifacts_complete']}/{payload['n_artifacts_total']}` canonical artifacts complete, `{payload['n_sidecars_verified']}/{payload['n_artifacts_present']}` SHA-256 sidecars verified).",
        "",
        "## Benchmark Roster Universe",
        "",
        f"The canonical roster in `src/capability/unified/roster.py` covers **{payload['roster_universe']['n_total_arms']} unique arms** = **{payload['roster_universe']['n_single_sequence_arms']} single-sequence arms** (`33` frozen pre-registered `ROSTER` arms + `29` expanded single-sequence general PLMs) plus **{payload['roster_universe']['n_msa_context_arms']} multi-sequence MSA context arms** (`msa-transformer`, `dayhoff-170m-msa`, `protmamba`). Frozen pre-registered baselines and `src/capability/core/arms.py` remain untouched.",
        "",
        "## Canonical Artifact Manifest & SHA-256 Digests",
        "",
        "| Section | Artifact | Status | Sidecar | Size (MB) | SHA-256 (prefix) | Producer |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for item in payload["artifact_manifest"]:
        size_mb = f"{item['size_bytes'] / 1e6:.2f}" if item["size_bytes"] is not None else "—"
        sha_short = f"`{item['sha256'][:16]}`" if item["sha256"] else "—"
        sidecar_s = "OK" if item["sidecar_verified"] else ("MISSING" if not item["exists"] else "FAIL")
        lines.append(
            f"| `{item['section']}` | `{item['filename']}` | `{item['status']}` | {sidecar_s} | {size_mb} | {sha_short} | `{item['producer']}` |"
        )

    lines.extend(
        [
            "",
            "## Reproduction & Parity Verification Gates",
            "",
            "| Gate | Artifact | Status | Observed Max Abs Diff |",
            "| --- | --- | --- | --- |",
        ]
    )
    for gname, grec in payload["reproduction_gates"].items():
        diff_s = (
            f"`{grec['max_abs_diff']:.2e}`"
            if isinstance(grec.get("max_abs_diff"), (int, float))
            else "—"
        )
        lines.append(
            f"| `{gname}` | `{grec['artifact']}` | `{'PASSED' if grec.get('passed') else 'PENDING/FAILED'}` | {diff_s} |"
        )

    lines.extend(
        [
            "",
            "## Reproducibility & Hygiene Guarantees",
            "",
            "- All unified CPU analyses run with deterministic seeds (`SPLIT_SEEDS = (20260923, 20260924, 20260925)`, paired bootstrap seed `20261106` / `20261011`), `PYTHONDONTWRITEBYTECODE=1`, and spawn-safe multiprocessing.",
            "- Every canonical JSON is accompanied by a verified `.sha256` sidecar checked by `scripts/capability/unified/build_unified_master_summary.py`.",
            "- No disposable cluster pod names, internal IP addresses, or host-specific credentials are recorded in any repository artifact or provenance table.",
            "",
        ]
    )
    text = "\n".join(lines)
    if FORBIDDEN_INFRA_RE.search(text):
        raise ValueError("Refusing to write PROVENANCE.md containing forbidden pod/node identifier")
    return text


def build_unified_master_summary(
    *,
    root: Path = REPO_ROOT,
    out_dir: Path | None = None,
    allow_pending: bool = False,
) -> dict[str, Any]:
    """Verify all canonical unified artifacts and assemble ``unified_master_summary.json`` + ``PROVENANCE.md``."""
    target_dir = Path(out_dir) if out_dir is not None else (root / UNIFIED_OUTPUT_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    loaded_docs: dict[str, dict[str, Any]] = {}
    pending_reasons: list[str] = []

    for spec in CANONICAL_ARTIFACTS:
        fpath = target_dir / spec.filename
        check = verify_sidecar(fpath)
        doc: dict[str, Any] | None = None
        status = "complete"
        detail: str | None = None

        if not check["exists"]:
            status = "missing"
            detail = f"{spec.filename} is absent"
            pending_reasons.append(detail)
        elif not check["verified"]:
            raise RuntimeError(
                f"SHA-256 sidecar verification FAILED for {fpath}: "
                f"actual={check['sha256']} vs declared={check.get('declared_sha256')}"
            )
        else:
            doc = json.loads(fpath.read_text(encoding="utf-8"))
            loaded_docs[spec.filename] = doc
            if spec.filename == "r6_experiment_4_2_phase_space_unified.json":
                cov = doc.get("coverage") or {}
                n_pend = int(cov.get("n_pending_plms", 0))
                if n_pend > 0 or not cov.get("is_complete_panel", False):
                    status = "partial_pending_h200"
                    detail = f"r6_experiment_4_2 has n_pending_plms={n_pend} (is_complete_panel={cov.get('is_complete_panel')})"
                    pending_reasons.append(detail)

        manifest.append(
            {
                "filename": spec.filename,
                "section": spec.section,
                "role": spec.role,
                "producer": spec.producer,
                "exists": check["exists"],
                "sidecar_verified": check["verified"],
                "size_bytes": int(fpath.stat().st_size) if check["exists"] else None,
                "sha256": check["sha256"],
                "schema": (doc.get("schema") or doc.get("schema_version")) if doc else None,
                "status": status,
                "detail": detail,
            }
        )

    if pending_reasons and not allow_pending:
        raise RuntimeError(
            "Strict master summary build refused due to incomplete/pending artifacts "
            "(pass --allow-pending for interim build): " + "; ".join(pending_reasons)
        )

    # Assemble reproduction gates ledger across all completed artifacts.
    gates: dict[str, dict[str, Any]] = {}
    if "r1_proteingym_unified_62.json" in loaded_docs:
        g1 = _extract_r1_gate(loaded_docs["r1_proteingym_unified_62.json"])
        gates["r1_proteingym_33arm_parity"] = {"artifact": "r1_proteingym_unified_62.json", **g1}
    if "r2_biophysical_unified_62.json" in loaded_docs:
        vg = loaded_docs["r2_biophysical_unified_62.json"]["verification_gates"]
        gates["r2_megascale_101g_progen2_medium"] = {
            "artifact": "r2_biophysical_unified_62.json",
            "passed": bool(vg["megascale_101g_reproduction"]["passed"]),
            "max_abs_diff": float(vg["megascale_101g_reproduction"]["max_oof_abs_diff"]),
        }
        gates["r2_megascale_79g_esmc6b_evo_parity"] = {
            "artifact": "r2_biophysical_unified_62.json",
            "passed": bool(vg["megascale_79g_esmc6b_evo_parity"]["passed"]),
            "max_abs_diff": float(vg["megascale_79g_esmc6b_evo_parity"]["max_mse_abs_diff"]),
        }
        gates["r2_megascale_79g_full_nuisance_U_C"] = {
            "artifact": "r2_biophysical_unified_62.json",
            "passed": float(vg["megascale_79g_full_nuisance_U_C_max_diff"]) < 1e-10,
            "max_abs_diff": float(vg["megascale_79g_full_nuisance_U_C_max_diff"]),
        }
    if "r3_epistasis_and_contacts_unified_62.json" in loaded_docs:
        g3 = loaded_docs["r3_epistasis_and_contacts_unified_62.json"]["reproduction_gate"]
        gates["r3_epistasis_progen2_small"] = {
            "artifact": "r3_epistasis_and_contacts_unified_62.json",
            "passed": bool(g3["passed"]),
            "max_abs_diff": float(g3["max_abs_diff"]),
        }
    if "r6_experiment_4_2_phase_space_unified.json" in loaded_docs:
        rg42 = loaded_docs["r6_experiment_4_2_phase_space_unified.json"].get("reproduction_gates") or {}
        for k, v in rg42.items():
            gates[f"exp4_2_{k}"] = {
                "artifact": "r6_experiment_4_2_phase_space_unified.json",
                "passed": bool(v.get("passed")),
                "max_abs_diff": float(v.get("max_abs_diff", 0.0)),
            }
    if "r6_experiment_4_3_native_likelihood_unified.json" in loaded_docs:
        g43 = loaded_docs["r6_experiment_4_3_native_likelihood_unified.json"]["e17_reproduction_gate"]
        gates["exp4_3_e17_column_span_protgpt2_zymctrl"] = {
            "artifact": "r6_experiment_4_3_native_likelihood_unified.json",
            "passed": bool(g43["passed"]),
            "max_abs_diff": float(g43["max_abs_diff"]),
        }

    # Assemble structured results summary
    results_summary: dict[str, Any] = {}
    if all(
        k in loaded_docs
        for k in (
            "r1_proteingym_unified_62.json",
            "r1_supplementary_c0_ladder_62.json",
            "r1_adversarial_pairs_62.json",
        )
    ):
        results_summary["result_1_proteingym"] = _summarize_r1(loaded_docs)
    if "r2_biophysical_unified_62.json" in loaded_docs:
        results_summary["result_2_biophysical_transfer_and_scaling"] = _summarize_r2(loaded_docs)
    if (
        "r3_epistasis_and_contacts_unified_62.json" in loaded_docs
        and "contacts_unified_62.json" in loaded_docs
    ):
        results_summary["result_3_epistasis_and_contacts"] = _summarize_r3(loaded_docs)

    r4_block: dict[str, Any] = {
        "experiment_4_1_unconditional_generation": _summarize_exp4_1(
            loaded_docs.get("r5_experiment_4_1_unified_generation.json")
        ),
        "experiment_4_2_phase_space": _summarize_exp4_2(
            loaded_docs.get("r6_experiment_4_2_phase_space_unified.json")
        ),
    }
    if "r6_experiment_4_3_native_likelihood_unified.json" in loaded_docs:
        r4_block["experiment_4_3_native_likelihood_selection"] = _summarize_exp4_3(
            loaded_docs["r6_experiment_4_3_native_likelihood_unified.json"]
        )
    if "family_memory_unified_62.json" in loaded_docs:
        fm = loaded_docs["family_memory_unified_62.json"]
        r4_block["associative_memory"] = {
            "n_arms_family_conditionals": int(fm["family_conditionals_62"]["n_arms"]),
            "n_arms_family_distribution": int(fm["family_distribution_46"]["n_arms"]),
            "n_evaluation_families": int(fm["family_conditionals_62"]["n_evaluation_families"]),
        }
    if "r4_r5_r6_unified_summary.json" in loaded_docs:
        r456 = loaded_docs["r4_r5_r6_unified_summary.json"]
        r4_block["context_and_alignment_summary"] = {
            "n_r4_primary_sequence_context_arms": int(
                r456["r4_context_access"]["n_primary_sequence_context_arms"]
            ),
            "n_r5_esmfold2_evaluations": int(
                r456["r5_generative_expression"]["n_esmfold2_structure_evaluations"]
            ),
            "n_r6_contrast_pairs": int(
                r456["r6_post_training_alignment"]["n_contrast_pairs_with_band_tables"]
            ),
        }
    results_summary["result_4_generation_memory_and_alignment"] = r4_block

    n_present = sum(1 for m in manifest if m["exists"])
    n_complete = sum(1 for m in manifest if m["status"] == "complete")
    n_verified = sum(1 for m in manifest if m["sidecar_verified"])

    counts_by_paradigm = {p: len(arms_by_paradigm(p)) for p in PARADIGM_ORDER}
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "completion_status": "complete" if not pending_reasons else "interim_pending",
        "pending_artifacts": pending_reasons,
        "n_artifacts_total": len(manifest),
        "n_artifacts_present": n_present,
        "n_artifacts_complete": n_complete,
        "n_sidecars_verified": n_verified,
        "roster_universe": {
            "n_total_arms": len(UNIFIED_65_ARMS),
            "n_single_sequence_arms": len(UNIFIED_62_SINGLE_SEQUENCE_ARMS),
            "n_preregistered_arms": len(PREREGISTERED_33_ARMS),
            "n_expanded_arms": len(EXPANDED_32_ARMS),
            "n_msa_context_arms": len(UNIFIED_3_MSA_CONTEXT_ARMS),
            "counts_by_paradigm": counts_by_paradigm,
            "single_sequence_arms_62": list(UNIFIED_62_SINGLE_SEQUENCE_ARMS),
            "msa_context_arms_3": list(UNIFIED_3_MSA_CONTEXT_ARMS),
            "arm_metadata_count": len(ARM_METADATA),
        },
        "artifact_manifest": manifest,
        "reproduction_gates": gates,
        "all_reproduction_gates_passed": all(bool(g.get("passed")) for g in gates.values()) if gates else False,
        "results_summary": results_summary,
    }

    out_json = target_dir / "unified_master_summary.json"
    digest = _atomic_write_json(out_json, payload)
    prov_path = target_dir / "PROVENANCE.md"
    prov_path.write_text(_render_provenance_markdown(payload), encoding="utf-8")

    return {
        "summary_path": str(out_json),
        "summary_sha256": digest,
        "provenance_path": str(prov_path),
        "payload": payload,
    }
