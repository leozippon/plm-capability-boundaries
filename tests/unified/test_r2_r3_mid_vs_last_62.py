"""Tests for Workstream R2/R3-Mid-vs-Last (62-model middle vs final representation contrast)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.capability.core.io import sha256_file
from src.capability.readouts.readout_analysis import row_weights
from src.capability.unified import r2_experiment_2_1 as r2_mod
from src.capability.unified.r2_r3_mid_vs_last_yardstick import (
    DEFAULT_CELL_CACHE_REL,
    DEFAULT_OUTPUT_REL,
    FINE_PARADIGM_ORDER,
    FORBIDDEN_INFRA_RE,
    MACRO_PARADIGM_ORDER,
    MODEL_LAYERS_62,
    SCHEMA,
    _summarize_r3_epistasis,
    build_depth_sensitivity_diagnostic,
    build_layer_provenance_table,
    classify_fine_paradigm,
    classify_macro_paradigm,
    fit_designs_group_suffstats,
)
from src.capability.unified.roster import REPO_ROOT, UNIFIED_62_SINGLE_SEQUENCE_ARMS

OUT_JSON = REPO_ROOT / DEFAULT_OUTPUT_REL


def test_paradigm_partitions_and_layer_table_cover_all_62_arms() -> None:
    assert set(MODEL_LAYERS_62.keys()) == set(UNIFIED_62_SINGLE_SEQUENCE_ARMS)
    table = build_layer_provenance_table()
    assert len(table) == 62

    macro_counts = {k: 0 for k in MACRO_PARADIGM_ORDER}
    fine_counts = {k: 0 for k in FINE_PARADIGM_ORDER}
    for arm_id, rec in table.items():
        L = rec["n_layers_L"]
        assert L >= 4
        assert rec["mid_block_0based"] == (L - 1) // 2
        assert rec["final_block_0based"] == L - 1
        assert rec["r_mid_cols"] == [0, 512]
        assert rec["r_last_cols"] == [512, 1024]
        macro_counts[classify_macro_paradigm(arm_id)] += 1
        fine_counts[classify_fine_paradigm(arm_id)] += 1

    # Exact 40 + 16 + 3 + 3 = 62 partition
    assert macro_counts == {
        "causal_autoregressive_and_ssm": 40,
        "bidirectional_mlm_and_multimodal": 16,
        "discrete_diffusion": 3,
        "structure_vocabulary_mlm": 3,
    }
    assert sum(fine_counts.values()) == 62
    assert fine_counts["hybrid_ssm"] == 1
    assert fine_counts["multimodal_seq_only"] == 1


def test_depth_sensitivity_diagnostic_runs_cleanly() -> None:
    diag = build_depth_sensitivity_diagnostic(REPO_ROOT)
    assert "available" in diag
    if diag["available"]:
        assert diag["n_arms_checked"] >= 30
        adj = diag["adjacent_mid_layer_blocks_floor_vs_upper"]
        assert adj["r_spearman_mean_abs_diff"] is not None
        assert adj["r_spearman_mean_abs_diff"] < 0.05


@pytest.mark.skipif(
    not (REPO_ROOT / "results/extensions/unified_yardstick_20261124/r2_biophysical_unified_62.json").is_file(),
    reason="Sealed unified R2/R3 baseline artifacts absent",
)
def test_two_arm_end_to_end_parity_against_sealed_r2_and_r3() -> None:
    """Verify live <1e-10 R_joint parity from raw MegaScale 1024-d ``.npz`` readouts on 2 arms
    (causal ``progen2-medium`` + MLM ``esm2-650m``) against sealed R2, plus cached R3 cell parity.
    """
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(min(8, max(1, prev_threads)))
    try:
        arms = ("progen2-medium", "esm2-650m")
        ms = r2_mod.load_megascale_supports(root=REPO_ROOT)
        panel_79 = ms["panel_79"]
        groups_79 = ms["groups_79"]
        mask_79 = ms["mask_79"]
        ms_w_79 = row_weights(panel_79["site"], panel_79["group"])
        fc_79_evo = r2_mod.build_fold_cache(groups=panel_79["group"], weights=ms_w_79)

        def _eval_79(p: np.ndarray) -> dict[str, np.ndarray]:
            return r2_mod.evaluate_megascale_oof(p, panel_79["target"], panel_79["group"], panel_79["site"], groups_79)

        sealed_r2 = json.loads(
            (REPO_ROOT / "results/extensions/unified_yardstick_20261124/r2_biophysical_unified_62.json").read_text(
                encoding="utf-8"
            )
        )
        sealed_ms_79 = sealed_r2["datasets"]["megascale"]["primary_evolutionary_control"][
            "cc_79_profile_potts_gemme"
        ]["per_arm_summary"]

        r_dir = REPO_ROOT / r2_mod.DEFAULT_R_ARCHIVE_DIR
        max_diff = 0.0
        for arm_id in arms:
            m_101, r_101, _ = r2_mod.load_megascale_arm(arm_id, ms["cohort_101"], root=REPO_ROOT, r_archive_dir=r_dir)
            assert r_101 is not None and r_101.shape[1] == 1024
            m_79 = m_101[mask_79]
            r_79 = r_101[mask_79]
            blocks = {
                "P": ms["p_logodds_79"][:, None],
                "POTTS": ms["potts_raw_79"][:, None],
                "GEMME": ms["gemme_raw_79"][:, None],
                "M": m_79[:, None],
                "R_mid": np.ascontiguousarray(r_79[:, :512]),
                "R_last": np.ascontiguousarray(r_79[:, 512:]),
            }
            designs = {
                "R_mid": ("R_mid",),
                "R_last": ("R_last",),
                "C_M_R_joint": ("P", "POTTS", "GEMME", "M", "R_mid", "R_last"),
            }
            fit = r2_mod.collapse_oof(
                fit_designs_group_suffstats(
                    panel_79["target"],
                    panel_79["group"],
                    blocks,
                    designs,
                    fc_79_evo,
                    alphas=r2_mod.ALPHAS_17,
                    renorm_val_weights=True,
                    tie_largest_alpha=True,
                ),
                _eval_79,
            )
            for ep in r2_mod.ENDPOINTS:
                ours = float(np.mean(fit[ep]["C_M_R_joint"]))
                theirs = float(sealed_ms_79[arm_id]["C_M_R"][ep])
                max_diff = max(max_diff, abs(ours - theirs))
        assert max_diff < 1e-10, f"Live MegaScale raw-npz parity diff {max_diff:.3e} >= 1e-10"

        # Also verify cached R3 cells for the same 2 arms reproduce sealed R3 epistasis to < 1e-10.
        r3_cdir = REPO_ROOT / DEFAULT_CELL_CACHE_REL / "r3_epistasis"
        if all((r3_cdir / f"{a}.json").is_file() for a in arms):
            r3_recs = {a: json.loads((r3_cdir / f"{a}.json").read_text(encoding="utf-8")) for a in arms}
            r3_sum = _summarize_r3_epistasis(arms, r3_recs, draws=100, seed=20261106)
            sealed_r3 = json.loads(
                (REPO_ROOT / "results/extensions/unified_yardstick_20261124/r3_epistasis_and_contacts_unified_62.json").read_text(
                    encoding="utf-8"
                )
            )
            mr_blk = sealed_r3["epistasis"]["MR_cycle_62_over_nuisance_profile_couplings"]
            for metric in ("mse", "spearman"):
                for a in arms:
                    ours_r_c = float(r3_sum[metric]["contrasts"]["plus_R_joint_given_C"]["columns"][a]["point"])
                    seal_r_c = float(mr_blk[metric]["R_given_controls"]["arms"][a]["point"])
                    assert abs(ours_r_c - seal_r_c) < 1e-10
    finally:
        torch.set_num_threads(prev_threads)


@pytest.mark.skipif(not OUT_JSON.is_file(), reason="Full 62-arm mid-vs-last artifact not yet built")
def test_full_62_artifact_integrity_and_zero_infra_leaks() -> None:
    sidecar = OUT_JSON.with_name(OUT_JSON.name + ".sha256")
    assert sidecar.is_file()
    assert sha256_file(OUT_JSON) == sidecar.read_text(encoding="utf-8").strip().split()[0]
    raw = OUT_JSON.read_text(encoding="utf-8")
    assert not FORBIDDEN_INFRA_RE.search(raw)
    doc = json.loads(raw)
    assert doc["schema"] == SCHEMA
    assert doc["completion_status"] == "complete"
    assert doc["coverage"]["is_complete_62"] is True
    assert doc["reproduction_gate"]["passed"] is True
    assert doc["reproduction_gate"]["max_abs_diff"] < 1e-10
