"""Tests for the unified 62-model Merged New Experiment 2.1 (MegaScale + Domainome)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.capability.core.io import sha256_file
from src.capability.interactions.pairwise_epistasis import ALPHAS
from src.capability.readouts.readout_analysis import ridge_predict, row_weights
from src.capability.unified.r2_experiment_2_1 import (
    ALPHAS_17,
    ENDPOINTS,
    FROZEN_MS_101_PANEL_DIR,
    REPRODUCTION_TOLERANCE,
    SCALING_LADDERS,
    SCHEMA,
    _solve_ridge_from_gram,
    build_fold_cache,
    load_megascale_supports,
    paired_bands_summary,
    run_esmc6b_evo_parity_gate,
    run_megascale_reproduction_gate,
)
from src.capability.unified.roster import (
    PREREGISTERED_33_ARMS,
    REPO_ROOT,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_OUTPUT_DIR,
)

OUTPUT_JSON = REPO_ROOT / UNIFIED_OUTPUT_DIR / "r2_biophysical_unified_62.json"
HAS_LOCAL_DATA = (REPO_ROOT / FROZEN_MS_101_PANEL_DIR / "panel.json").is_file()


def test_paired_bands_summary_basic_properties() -> None:
    rng = np.random.default_rng(7)
    n_groups = 79
    strong_pos = rng.normal(0.5, 0.1, size=n_groups)
    noise = rng.normal(0.0, 0.1, size=n_groups)
    strong_neg = rng.normal(-0.5, 0.1, size=n_groups)
    mat = np.column_stack([strong_pos, noise, strong_neg])

    res = paired_bands_summary(mat, ["pos", "zero", "neg"], draws=2000, seed=20261106)
    assert res["n_groups"] == n_groups
    assert res["family_size"] == 3
    assert res["columns"]["pos"]["verdict_native"] == "positive"
    assert res["columns"]["pos"]["verdict_studentized"] == "positive"
    assert res["columns"]["neg"]["verdict_native"] == "negative"
    assert res["columns"]["zero"]["verdict_native"] == "unresolved"
    for col in ("pos", "zero", "neg"):
        c = res["columns"][col]
        assert c["native_simultaneous_95"][0] <= c["marginal_95"][0] + 1e-12
        assert c["native_simultaneous_95"][1] >= c["marginal_95"][1] - 1e-12


@pytest.mark.parametrize("alpha_grid", [ALPHAS, ALPHAS_17])
def test_gram_cholesky_and_eigh_match_ridge_predict(alpha_grid) -> None:
    rng = np.random.default_rng(123)
    x_tr = rng.standard_normal((240, 40)) + 100.0
    x_tr[:, 7] = 5.0  # constant column on train
    x_te = rng.standard_normal((60, 40)) + 100.0
    x_te[:, 7] = 12.0  # non-zero shift on test/val
    y_tr = rng.standard_normal(240) + 10.0
    w_tr = rng.uniform(0.2, 3.0, size=240)

    ref = ridge_predict(x_tr, y_tr, w_tr, x_te, alpha_grid, device="cpu")

    xt = torch.as_tensor(x_tr, dtype=torch.float64)
    yt = torch.as_tensor(y_tr, dtype=torch.float64)
    wt = torch.as_tensor(w_tr, dtype=torch.float64)
    w_sum = wt.sum()
    mu0 = (wt[:, None] * xt).sum(0) / w_sum
    y0 = (wt * yt).sum() / w_sum
    xc = xt - mu0
    yc = yt - y0
    var = (wt[:, None] * xc.square()).sum(0) / w_sum
    w_xc = wt[:, None] * xc
    beta, mean_c, inv_scale, ym_c = _solve_ridge_from_gram(
        w_sum, torch.zeros_like(mu0), var, xc.T @ w_xc, (wt * yc).sum(), w_xc.T @ yc, alpha_grid
    )
    xc_te = torch.as_tensor(x_te, dtype=torch.float64) - mu0
    fast = (((xc_te - mean_c) * inv_scale) @ beta + (y0 + ym_c)).numpy()
    assert float(np.max(np.abs(ref - fast))) < 1e-12


def test_roster_and_ladders_integrity() -> None:
    assert len(PREREGISTERED_33_ARMS) == 33
    assert len(UNIFIED_62_SINGLE_SEQUENCE_ARMS) == 62
    arm_set = set(UNIFIED_62_SINGLE_SEQUENCE_ARMS)
    for ladder, members in SCALING_LADDERS.items():
        assert len(members) >= 2, ladder
        for arm in members:
            assert arm in arm_set, f"{ladder}: unknown arm {arm}"


@pytest.mark.skipif(not HAS_LOCAL_DATA, reason="frozen MegaScale inputs absent locally")
def test_megascale_verification_gates() -> None:
    from src.capability.stability import stability_gate as gate

    ms = load_megascale_supports(REPO_ROOT)
    assert len(ms["groups_101"]) == 101
    assert len(ms["groups_79"]) == 79
    w_101 = row_weights(ms["panel_101"]["site"], ms["panel_101"]["group"])
    w_79 = row_weights(ms["panel_79"]["site"], ms["panel_79"]["group"])
    fc_101 = build_fold_cache(
        groups=ms["panel_101"]["group"],
        weights=w_101,
        g_builder=lambda held, inner: gate.nuisance_response(
            ms["panel_101"]["states"], ms["panel_101"]["pair_states"], np.asarray(held), inner, device="cpu"
        )[0],
    )
    gate_101 = run_megascale_reproduction_gate(ms, fc_101, root=REPO_ROOT)
    assert gate_101["passed"] is True
    assert gate_101["max_oof_abs_diff"] < REPRODUCTION_TOLERANCE

    fc_79_evo = build_fold_cache(groups=ms["panel_79"]["group"], weights=w_79)
    gate_evo = run_esmc6b_evo_parity_gate(ms, fc_79_evo, root=REPO_ROOT)
    assert gate_evo["passed"] is True
    assert gate_evo["max_mse_abs_diff"] < 1e-9
    assert gate_evo["max_spearman_abs_diff"] < 1e-9


@pytest.mark.skipif(not OUTPUT_JSON.is_file(), reason="unified Experiment 2.1 output JSON not yet generated")
def test_generated_artifact_integrity_and_schema() -> None:
    sidecar = OUTPUT_JSON.with_name(OUTPUT_JSON.name + ".sha256")
    assert sidecar.is_file()
    expected_sha = sidecar.read_text(encoding="utf-8").strip().split()[0]
    assert sha256_file(OUTPUT_JSON) == expected_sha

    doc = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    assert doc["schema"] == SCHEMA
    assert len(doc["arms_62"]) == 62
    assert doc["verification_gates"]["megascale_101g_reproduction"]["passed"] is True
    assert doc["verification_gates"]["megascale_79g_esmc6b_evo_parity"]["passed"] is True
    assert doc["verification_gates"]["megascale_79g_full_nuisance_U_C_max_diff"] < REPRODUCTION_TOLERANCE

    cov = doc["coverage_summary"]
    assert cov["megascale"]["n_arms_M"] == 62
    assert cov["megascale"]["n_arms_R"] >= 28
    assert cov["domainome"]["n_arms_M"] == 62
    assert cov["domainome"]["n_arms_R"] >= 61

    for ds_name in ("megascale", "domainome"):
        ds = doc["datasets"][ds_name]
        assert "zero_shot_spearman" in ds
        assert "primary_evolutionary_control" in ds
        assert "full_nuisance_evolutionary_control" in ds
        for block in ds["primary_evolutionary_control"].values():
            assert "dimension_1_quantitative_error" in block
            assert "dimension_2_ranking" in block
            assert len(block["per_arm_summary"]) == 62
            for ep in ENDPOINTS:
                assert ep in block["baseline_C_levels"]

    curves = doc["scaling_divergence_curves"]
    assert "megascale_cc_79" in curves
    assert "domainome_cc_94" in curves
    for ds_curves in curves.values():
        for ladder in SCALING_LADDERS:
            assert ladder in ds_curves["ladders"]
