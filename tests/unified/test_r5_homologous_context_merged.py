"""Unit tests for the Merged Result 5 unified homologous-context assembler (65 & full 84 tiers)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.capability.core.io import sha256_file
from src.capability.unified.r4_scoring_yardstick import classify_r4_context_axes
from src.capability.unified.r5_homologous_context_merged import (
    DEFAULT_OUTPUT_NAME,
    DEFAULT_OUTPUT_NAME_FULL,
    FORBIDDEN_INFRA_RE,
    RESULT_5_COMPONENTS,
    RESULT_5_COMPONENTS_FULL,
    SCHEMA,
    SCHEMA_FULL,
    build_r5_homologous_context_merged,
    build_r5_homologous_context_merged_full,
)
from src.capability.unified.roster import REPO_ROOT


def _write_with_sidecar(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def test_input_alignment_format_classification_is_unambiguous() -> None:
    assert classify_r4_context_axes("msa-transformer-100m", aligned_msa=True) == (
        "explicit_column_aligned",
        "msa_2d_axial",
    )
    assert classify_r4_context_axes("msa-transformer-100m", aligned_msa=False) == (
        "unaligned_concat",
        "msa_2d_axial",
    )
    assert classify_r4_context_axes("protgpt3-msa", aligned_msa=True) == (
        "unaligned_concat",
        "multiseq_icl_unaligned",
    )
    assert classify_r4_context_axes("dayhoff-3b-msa", aligned_msa=True) == (
        "unaligned_concat",
        "multiseq_icl_unaligned",
    )
    assert classify_r4_context_axes("esm2-650m") == (
        "unaligned_concat",
        "single_chain_bidirectional",
    )
    assert classify_r4_context_axes("progen3-3b") == (
        "unaligned_concat",
        "single_chain_causal",
    )
    assert classify_r4_context_axes("esm3-1.4b", is_route_a_3d=True) == (
        "structural_3d_ss8_aligned_tracks",
        "multimodal_3d_tracks",
    )


def _populate_synthetic_65_components(tmp_path: Path) -> None:
    _write_with_sidecar(
        tmp_path / "r4_scoring_unified_65.json",
        {
            "schema": "unified_r4_scoring_65_v1",
            "completion_status": "complete",
            "is_complete_65": True,
            "coverage": {"n_arms_present": 65, "n_pending_arms": 0},
            "reproduction_gate": {"passed": True, "max_abs_diff": 0.0},
            "horizontal_benchmark_table": [
                {
                    "row_key": "msa-transformer-100m|b1024",
                    "arm_id": "msa-transformer-100m",
                    "input_alignment_format": "explicit_column_aligned",
                    "context_training_regime": "msa_2d_axial",
                },
                {
                    "row_key": "protgpt3-msa|b8192",
                    "arm_id": "protgpt3-msa",
                    "input_alignment_format": "unaligned_concat",
                    "context_training_regime": "multiseq_icl_unaligned",
                },
            ],
            "by_training_regime_and_alignment": {},
        },
    )
    _write_with_sidecar(
        tmp_path / "r5_dms_context_ranking_unified.json",
        {
            "schema": "unified_r5_dms_context_ranking_v1",
            "completion_status": "complete",
            "coverage": {"n_models": 68, "n_pending": 0},
            "reproduction_gates": {"rescue_30_80": {"passed": True, "max_abs_diff": 1e-15}},
            "subtest_1a_rescue_30_80": {"n_assays": 164},
            "subtest_1b_e09_five_band": {"n_assays": 201},
        },
    )
    _write_with_sidecar(
        tmp_path / "r4_continuation_unified.json",
        {
            "schema": "unified_r4_continuation_v1",
            "completion_status": "complete",
            "coverage": {"n_models": 35, "n_pending": 0},
            "primary_12_family_summary": {"n_families": 12},
            "length_band_folding_summary": {"bands": ["len_16_128", "len_129_256", "len_257_512"]},
        },
    )


def _populate_synthetic_full_components(tmp_path: Path) -> None:
    _write_with_sidecar(
        tmp_path / "r4_scoring_unified_full.json",
        {
            "schema": "unified_r4_scoring_full_v1",
            "completion_status": "complete",
            "n_arms_primary": 84,
            "n_arms_total_rows": 88,
            "arms_order": [
                "msa-transformer-100m",
                "dplm2-650m",
                "dima-esm2-3b",
                "dplm2-650m|struct",
                "proteinmpnn-v-020",
            ],
            "pending_primary_arms": [],
            "pending_struct_companions": [],
            "horizontal_table": [
                {
                    "arm_id": "msa-transformer-100m",
                    "conditioning_regime": "sequence_only",
                    "input_alignment_format": "explicit_column_aligned",
                },
                {
                    "arm_id": "dplm2-650m",
                    "conditioning_regime": "sequence_only",
                    "input_alignment_format": "unaligned_concat",
                },
                {
                    "arm_id": "dima-esm2-3b",
                    "conditioning_regime": "sequence_only",
                    "input_alignment_format": "unaligned_concat",
                },
                {
                    "arm_id": "dplm2-650m|struct",
                    "conditioning_regime": "backbone_3d_conditioned_oracle",
                    "input_alignment_format": "structural_3d_ss8_aligned_tracks",
                },
                {
                    "arm_id": "proteinmpnn-v-020",
                    "conditioning_regime": "backbone_3d_conditioned_oracle",
                    "input_alignment_format": "two_chain_spatial_offset_3d",
                },
            ],
            "metrics_by_band": {
                "all_800": {
                    "hom_beats_unrelated": {
                        "n_clusters": 160,
                        "null_shift": 0.5,
                        "bands": {"counts": {"studentized": {"positive": 5}}},
                        "stratified_summary": {
                            "sequence_only": {"n": 3},
                            "backbone_3d_conditioned_oracle": {"n": 2},
                        },
                    }
                }
            },
        },
    )
    _write_with_sidecar(
        tmp_path / "r5_dms_context_ranking_unified_full.json",
        {
            "schema": "r5_dms_context_ranking_unified_full_v1",
            "completion_status": "complete",
            "pending_base_1a_entries": [],
            "pending_base_1b_entries": [],
            "pending_new_1a_primary": [],
            "pending_new_1b_primary": [],
            "subtest_1a_rescue_30_80": {
                "n_entries": 4,
                "n_common_clusters": 40,
                "entry_metadata": {
                    "msa-transformer-100m|aligned=True": {
                        "conditioning_regime": "sequence_only",
                        "input_alignment_format": "explicit_column_aligned",
                    },
                    "dplm2-650m": {
                        "conditioning_regime": "sequence_only",
                        "input_alignment_format": "unaligned_concat",
                    },
                    "dima-esm2-3b": {
                        "conditioning_regime": "sequence_only",
                        "input_alignment_format": "unaligned_concat",
                    },
                    "proteinmpnn-v-020": {
                        "conditioning_regime": "backbone_3d_conditioned_oracle",
                        "input_alignment_format": "two_chain_spatial_offset_3d",
                    },
                },
            },
            "subtest_1b_e09_five_band": {
                "n_entries": 3,
                "entry_metadata": {
                    "dplm2-650m": {
                        "conditioning_regime": "sequence_only",
                        "input_alignment_format": "unaligned_concat",
                    },
                    "dima-esm2-3b": {
                        "conditioning_regime": "sequence_only",
                        "input_alignment_format": "unaligned_concat",
                    },
                    "proteinmpnn-v-020": {
                        "conditioning_regime": "backbone_3d_conditioned_oracle",
                        "input_alignment_format": "two_chain_spatial_offset_3d",
                    },
                },
                "bands": {},
            },
        },
    )
    _write_with_sidecar(
        tmp_path / "r4_continuation_unified_full.json",
        {
            "schema_version": "r4_continuation_unified_full_v1",
            "sealed_baseline_source": {"path": "r4_continuation_unified.json", "n_sealed_entries": 35},
            "coverage": {
                "n_sealed_entries": 35,
                "n_new_expected": 23,
                "n_new_loaded": 23,
                "n_unaligned_prefix_total": 43,
                "n_backbone_3d_suffix_entries": 15,
                "missing_entries": [],
                "is_complete": True,
            },
            "e11_reproduction_gate": {"passed": True, "max_abs_diff": 0.0},
            "models_new_dplm2_mpnn": {
                "dplm2-650m|seq_only": {
                    "conditioning_regime": "unaligned_homolog_prefix",
                    "paradigm": "multimodal_diffusion",
                },
                "dima-esm2-3b|seq_only": {
                    "conditioning_regime": "unaligned_homolog_prefix",
                    "paradigm": "continuous_latent_diffusion",
                },
                "proteinmpnn-v-020|backbone_3d_suffix": {
                    "conditioning_regime": "backbone_3d_wt_prefix_suffix",
                    "paradigm": "inverse_folding_gnn",
                },
            },
            "models_all": {
                "protgpt3-msa|b8192": {
                    "conditioning_regime": "unaligned_homolog_prefix",
                    "input_alignment_format": "unaligned_concat",
                },
                "dplm2-650m|seq_only": {
                    "conditioning_regime": "unaligned_homolog_prefix",
                    "paradigm": "multimodal_diffusion",
                },
                "dima-esm2-3b|seq_only": {
                    "conditioning_regime": "unaligned_homolog_prefix",
                    "paradigm": "continuous_latent_diffusion",
                },
                "proteinmpnn-v-020|backbone_3d_suffix": {
                    "conditioning_regime": "backbone_3d_wt_prefix_suffix",
                    "paradigm": "inverse_folding_gnn",
                },
            },
        },
    )


def test_merged_r5_strict_refusal_sidecar_check_and_full_assembly(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Strict Merged Result 5"):
        build_r5_homologous_context_merged(root=REPO_ROOT, out_dir=tmp_path, allow_pending=False)

    interim = build_r5_homologous_context_merged(root=REPO_ROOT, out_dir=tmp_path, allow_pending=True)
    assert interim["payload"]["completion_status"] == "interim_pending"
    assert interim["payload"]["n_components_total"] == len(RESULT_5_COMPONENTS) == 3
    assert interim["payload"]["n_components_complete"] == 0
    assert interim["payload"]["n_pending"] == 3

    bad_p = tmp_path / "r4_scoring_unified_65.json"
    bad_p.write_text('{"schema": "unified_r4_scoring_65_v1"}\n', encoding="utf-8")
    bad_p.with_name(bad_p.name + ".sha256").write_text("0" * 64 + "  r4_scoring_unified_65.json\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 sidecar verification FAILED"):
        build_r5_homologous_context_merged(root=REPO_ROOT, out_dir=tmp_path, allow_pending=True)

    _populate_synthetic_65_components(tmp_path)
    full = build_r5_homologous_context_merged(root=REPO_ROOT, out_dir=tmp_path, allow_pending=False)
    p = full["payload"]
    assert p["completion_status"] == "complete"
    assert p["schema"] == SCHEMA
    assert p["tier"] == "65"
    assert p["n_components_complete"] == 3
    assert p["n_pending"] == 0
    assert p["all_reproduction_gates_passed"] is True
    assert p["main_table_800_units"]["n_horizontal_rows"] == 2
    assert p["downstream_1_dms_mutation_ranking"]["subtest_1a_rescue_30_80"]["n_assays"] == 164
    assert p["downstream_2_prefix_continuation_and_folding"]["primary_12_family_summary"]["n_families"] == 12
    out_txt = (tmp_path / DEFAULT_OUTPUT_NAME).read_text(encoding="utf-8")
    assert not FORBIDDEN_INFRA_RE.search(out_txt)
    assert (tmp_path / f"{DEFAULT_OUTPUT_NAME}.sha256").is_file()


def test_merged_r5_full_84_tier_preserves_conditioning_alignment_and_cli_both(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="tier=full"):
        build_r5_homologous_context_merged_full(root=REPO_ROOT, out_dir=tmp_path, allow_pending=False)

    interim_full = build_r5_homologous_context_merged_full(root=REPO_ROOT, out_dir=tmp_path, allow_pending=True)
    assert interim_full["payload"]["schema"] == SCHEMA_FULL
    assert interim_full["payload"]["n_components_total"] == len(RESULT_5_COMPONENTS_FULL) == 3
    assert interim_full["payload"]["n_components_complete"] == 0

    _populate_synthetic_65_components(tmp_path)
    _populate_synthetic_full_components(tmp_path)

    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(REPO_ROOT / "scripts/capability/unified/run_r5_homologous_context_merged.py"),
            "--out-dir",
            str(tmp_path),
            "--tier",
            "both",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"CLI --tier both failed: {proc.stderr}"
    assert (tmp_path / DEFAULT_OUTPUT_NAME).is_file()
    assert (tmp_path / f"{DEFAULT_OUTPUT_NAME}.sha256").is_file()
    assert (tmp_path / DEFAULT_OUTPUT_NAME_FULL).is_file()
    assert (tmp_path / f"{DEFAULT_OUTPUT_NAME_FULL}.sha256").is_file()

    doc_full = json.loads((tmp_path / DEFAULT_OUTPUT_NAME_FULL).read_text(encoding="utf-8"))
    assert doc_full["schema"] == SCHEMA_FULL
    assert doc_full["tier"] == "full"
    assert doc_full["completion_status"] == "complete"
    assert doc_full["n_components_complete"] == 3
    assert doc_full["all_reproduction_gates_passed"] is True

    mt_counts = doc_full["main_table_800_units"]["counts_by_conditioning_and_alignment"]
    assert mt_counts["by_conditioning_regime"] == {
        "backbone_3d_conditioned_oracle": 2,
        "sequence_only": 3,
    }
    assert mt_counts["by_input_alignment_format"] == {
        "explicit_column_aligned": 1,
        "structural_3d_ss8_aligned_tracks": 1,
        "two_chain_spatial_offset_3d": 1,
        "unaligned_concat": 2,
    }
    d1_counts = doc_full["downstream_1_dms_mutation_ranking"]["counts_by_conditioning_and_alignment"]
    assert d1_counts["subtest_1a_rescue_30_80"]["by_input_alignment_format"]["explicit_column_aligned"] == 1
    assert d1_counts["subtest_1a_rescue_30_80"]["by_input_alignment_format"]["two_chain_spatial_offset_3d"] == 1
    assert d1_counts["subtest_1a_rescue_30_80"]["by_input_alignment_format"]["unaligned_concat"] == 2
    d2_counts = doc_full["downstream_2_prefix_continuation_and_folding"]["counts_by_conditioning_and_alignment"]
    assert d2_counts["by_conditioning_regime"] == {
        "backbone_3d_wt_prefix_suffix": 1,
        "unaligned_homolog_prefix": 3,
    }
    assert not FORBIDDEN_INFRA_RE.search((tmp_path / DEFAULT_OUTPUT_NAME_FULL).read_text(encoding="utf-8"))
