"""Unit & integration tests for the unified 62-model R1 ProteinGym yardstick."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.capability.extensions import overlap
from src.capability.general_plm.eval_utils import sha256_file
from src.capability.readouts.readout_analysis import family_folds
from src.capability.unified import r1_yardstick as r1y
from src.capability.unified.roster import (
    PREREGISTERED_33_ARMS,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_OUTPUT_DIR,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_synthetic_signature(families: np.ndarray, seed: int = 20260923):
    outer = family_folds(families, 5, seed)
    all_fams = sorted(set(families.tolist()))
    sig = []
    for fold_idx, held in enumerate(outer):
        held_list = sorted(held)
        train_list = sorted(set(all_fams) - set(held_list))
        inner_mask = np.isin(families, train_list)
        inner = [sorted(v) for v in family_folds(families[inner_mask], 4, seed + fold_idx + 1)]
        sig.append((fold_idx, held_list, train_list, inner))
    return sig


def test_projected_predict_batch_matches_overlap_projected_predict_exact() -> None:
    """Batched covariance-slicing ridge solver matches ``overlap.projected_predict`` to < 1e-12."""
    rng = np.random.default_rng(12345)
    n_fams = 25
    n_per = 12
    n_rows = n_fams * n_per
    families = np.repeat(np.arange(n_fams), n_per)
    aids = np.array([f"assay_{f:02d}" for f in families])
    measured = rng.standard_normal(n_rows)
    x_full = rng.standard_normal((n_rows, 30))
    sig = _make_synthetic_signature(families)

    design_cols = {
        "scalar": [0],
        "small": [0, 1, 2, 3],
        "block": list(range(1, 15)),
        "wide": list(range(30)),
    }
    batch_preds, batch_recs = r1y.projected_predict_batch(
        x_full, design_cols, measured, aids, families, sig
    )
    for name, cols in design_cols.items():
        ref_pred, ref_rec = overlap.projected_predict(
            x_full[:, cols], measured, aids, families, sig
        )
        max_diff = float(np.max(np.abs(batch_preds[name] - ref_pred)))
        assert max_diff < 1e-12, f"{name}: max_diff={max_diff}"
        assert [r["alpha"] for r in batch_recs[name]] == [r["alpha"] for r in ref_rec]


def test_reproduction_gate_and_adversarial_pairs_match_frozen_33_arm_references() -> None:
    """Verify < 1e-12 reproduction of cached pre-registered R1 cells and 20261114 pair tiers."""
    ctx = r1y.load_r1_unified_context(REPO_ROOT)
    assert len(ctx.raw_model_scores) == 62
    assert set(ctx.raw_model_scores) == set(UNIFIED_62_SINGLE_SEQUENCE_ARMS)

    repro = r1y.verify_reproduction_gate(ctx, models=("progen2-small", "gpt2"), seed=20260923)
    assert repro["status"] == "passed"
    assert repro["max_abs_diff"] < 1e-12

    # Fast 200-draw pair verification confirming exact pooled-accuracy reproduction vs 20261114
    pairs = r1y.compute_adversarial_pairs_62(
        ctx, draws=200, tolerances=(0.20,), verify_ref=True
    )
    assert pairs["reproduction_vs_33_arm_reference"]["status"] == "passed"
    assert pairs["reproduction_vs_33_arm_reference"]["max_forward_pooled_diff_33_arms"] < 1e-12
    assert pairs["reproduction_vs_33_arm_reference"]["max_reverse_T_bal_M_pooled_diff_33_arms"] < 1e-12
    assert pairs["expanded_t_match_37_arms"]["n_arms"] == 37


def test_unified_r1_production_artifacts_contract_and_sha256() -> None:
    """Verify emitted unified 62-model R1 JSON artifacts and SHA-256 sidecars when present."""
    out_dir = REPO_ROOT / UNIFIED_OUTPUT_DIR
    p_main = out_dir / "r1_proteingym_unified_62.json"
    p_c0 = out_dir / "r1_supplementary_c0_ladder_62.json"
    p_pairs = out_dir / "r1_adversarial_pairs_62.json"
    if not (p_main.is_file() and p_c0.is_file() and p_pairs.is_file()):
        pytest.skip("Unified R1 production artifacts have not been generated yet")

    for path in (p_main, p_c0, p_pairs):
        sidecar = path.with_name(path.name + ".sha256")
        assert sidecar.is_file(), f"Missing sidecar {sidecar}"
        recorded = sidecar.read_text(encoding="utf-8").strip().split()[0]
        assert recorded == sha256_file(path)

    main = json.loads(p_main.read_text(encoding="utf-8"))
    assert main["n_models"] == 62
    assert tuple(main["models"]) == UNIFIED_62_SINGLE_SEQUENCE_ARMS
    assert main["reproduction_gate"]["status"] == "passed"
    assert main["reproduction_gate"]["max_abs_diff"] < 1e-12

    l1_163 = main["layer1_zero_shot"]["supports"]["clusters_163"]
    l1_156 = main["layer1_zero_shot"]["supports"]["clusters_156_gemme_nondegenerate"]
    assert len(l1_163["models"]) == 62
    assert abs(l1_163["baselines"]["profile"]["point"] - 0.36066263) < 1e-6
    assert abs(l1_163["baselines"]["potts"]["point"] - 0.42825182) < 1e-6
    assert l1_163["baselines"]["gemme"]["status"] == "not_estimable_on_refused_assays"
    assert l1_156["baselines"]["gemme"]["status"] == "complete"
    assert l1_156["baselines"]["gemme"]["point"] > 0.46

    l1_cats_163 = main["layer1_zero_shot"]["by_functional_category"]["clusters_163"]
    l1_cats_156 = main["layer1_zero_shot"]["by_functional_category"]["clusters_156_gemme_nondegenerate"]
    assert tuple(l1_cats_163.keys()) == r1y.FUNCTIONAL_CATEGORIES
    assert sum(c["n_assays"] for c in l1_cats_163.values()) == 201
    assert sum(c["n_assays"] for c in l1_cats_156.values()) == 194
    for cat in r1y.FUNCTIONAL_CATEGORIES:
        assert len(l1_cats_156[cat]["models"]) == 62
        assert l1_cats_156[cat]["baselines"]["gemme"]["status"] == "complete"

    l2 = main["layer2_core_bidirectional"]
    for variant in ("primary_scalar_baselines", "profile_block_baselines"):
        for sup_key in ("clusters_163", "clusters_156_gemme_nondegenerate"):
            for metric in ("spearman", "rank_mse"):
                block = l2[variant]["supports"][sup_key]["metrics"][metric]
                assert block["hypothesis_count"] == 248  # 62 x 2 x 2

    l2_cats_156 = l2["by_functional_category"]["clusters_156_gemme_nondegenerate"]
    assert tuple(l2_cats_156.keys()) == r1y.FUNCTIONAL_CATEGORIES
    for cat in r1y.FUNCTIONAL_CATEGORIES:
        assert l2_cats_156[cat]["n_models_M"] == 62
        assert l2_cats_156[cat]["n_models_R"] in (28, 29)
        sp_m = l2_cats_156[cat]["scalar_M_increments"]["spearman"]["contrasts"]
        sp_r = l2_cats_156[cat]["representation_R_increments"]["spearman"]["contrasts"]
        assert "M_given_P1" in sp_m and "M_given_PEG3" in sp_m
        assert "R_given_P1" in sp_r and "MR_given_PEG3" in sp_r

    rb = main["layer2_representation_bridge"]
    has_esm3_r = (out_dir / r1y.ESM3_R1_READOUT_REL).is_file()
    expected_r = 29 if has_esm3_r else 28
    assert rb["n_expanded_arms_with_R"] == expected_r
    assert rb["includes_esm3_1_4b"] is has_esm3_r
    assert rb["preregistered_33_arms_cached_progression"]["n_arms"] == 33

    c0 = json.loads(p_c0.read_text(encoding="utf-8"))
    assert c0["hypotheses_per_metric"] == 744  # 62 x 6 x 2
    assert c0["unified_62_arm_744_band"]["clusters_163"]["metrics"]["spearman"]["hypothesis_count"] == 744
    hist = c0["historical_preregistered_33_arm_396_band"]["payload"]
    assert hist["declared_hypotheses_per_metric"] == 396

    pairs = json.loads(p_pairs.read_text(encoding="utf-8"))
    assert pairs["n_models"] == 62
    assert pairs["reproduction_vs_33_arm_reference"]["status"] == "passed"
    assert pairs["forward_tiers"]["T_bal_PEG"]["family_62_arms"]["family_size"] == 62
    assert pairs["forward_tiers"]["T_bal_PEG"]["counts"]["pairs"] == 713
    assert set(pairs["reverse_tiers"]) == set(UNIFIED_62_SINGLE_SEQUENCE_ARMS)
    assert set(PREREGISTERED_33_ARMS) < set(pairs["reverse_tiers"])
    assert pairs["expanded_t_match_37_arms"]["n_arms"] == 37
