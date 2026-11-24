"""Tests for unified 62-model R3 epistasis and contacts yardstick."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from src.capability.core.io import sha256_file
from src.capability.readouts.readout_analysis import ALPHAS, ridge_predict
from src.capability.unified.r3_yardstick import (
    UNIFIED_62_WITH_R_ARMS,
    _build_gram_state,
    _predict_augmented_from_gram,
    paired_bands_complete,
    verify_r3_reproduction_gate,
)
from src.capability.unified.roster import (
    REPO_ROOT,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_OUTPUT_DIR,
)


def _assert_no_nan_or_inf(obj: object, path: str = "root") -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float at {path}: {obj}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_nan_or_inf(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_nan_or_inf(v, f"{path}[{i}]")


@pytest.mark.parametrize("k_extra", [1, 16])
def test_precomputed_gram_cholesky_matches_ridge_predict(k_extra: int) -> None:
    """_build_gram_state + _predict_augmented_from_gram must match ridge_predict to < 1e-12."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(20261124 + k_extra)
    n_tr, n_ev, p_base = 120, 45, 18
    x_base_tr = rng.normal(size=(n_tr, p_base))
    x_base_ev = rng.normal(size=(n_ev, p_base))
    x_base_tr[:, 3] = 5.0
    x_base_ev[:, 3] = 5.0

    x_extra_tr = rng.normal(size=(n_tr, k_extra))
    x_extra_ev = rng.normal(size=(n_ev, k_extra))
    if k_extra > 1:
        x_extra_tr[:, 2] = -1.5
        x_extra_ev[:, 2] = -1.5

    y_tr = rng.normal(size=n_tr)
    w_tr = rng.uniform(0.2, 3.0, size=n_tr)

    gram = _build_gram_state(x_base_tr, y_tr, w_tr, x_base_ev)
    ref_base = ridge_predict(x_base_tr, y_tr, w_tr, x_base_ev, ALPHAS)
    assert float(np.max(np.abs(gram.base_pred_alphas - ref_base))) < 1e-12

    pred_aug = _predict_augmented_from_gram(gram, x_extra_tr, x_extra_ev)
    ref_aug = ridge_predict(
        np.column_stack([x_base_tr, x_extra_tr]),
        y_tr,
        w_tr,
        np.column_stack([x_base_ev, x_extra_ev]),
        ALPHAS,
    )
    assert float(np.max(np.abs(pred_aug - ref_aug))) < 1e-12


def test_paired_bands_complete_structure_and_ordering() -> None:
    """Simultaneous bands must be at least as wide as marginal bands and resolve clear signal."""
    rng = np.random.default_rng(7)
    mat = np.column_stack(
        [
            rng.normal(loc=0.5, scale=0.1, size=40),
            rng.normal(loc=0.0, scale=0.1, size=40),
            rng.normal(loc=-0.5, scale=0.1, size=40),
            rng.normal(loc=0.2, scale=0.1, size=40),
        ]
    )
    res = paired_bands_complete(mat, draws=2000, seed=20261106)
    assert res["verdict"][0] == "positive"
    assert res["verdict"][2] == "negative"
    for i in range(4):
        lo_sim, hi_sim = res["interval"][i]
        lo_mrg, hi_mrg = res["marginal_interval"][i]
        lo_stu, hi_stu = res["studentized_interval"][i]
        assert lo_sim <= res["point"][i] <= hi_sim
        assert lo_mrg <= res["point"][i] <= hi_mrg
        assert lo_stu <= res["point"][i] <= hi_stu
        assert (hi_sim - lo_sim) >= (hi_mrg - lo_mrg) - 1e-9


def test_r3_reproduction_gate_exact() -> None:
    """R3 interaction Gram engine must reproduce cached 20261110 cells to < 1e-12."""
    gate = verify_r3_reproduction_gate(root=REPO_ROOT, arm="progen2-small", tolerance=1e-12)
    assert gate["passed"] is True
    assert gate["max_abs_diff"] < 1e-12


def test_r3_unified_artifact_integrity() -> None:
    """Validate 62/62 coverage, SHA256, readout tags, and finiteness of r3_epistasis_and_contacts_unified_62.json."""
    out_dir = REPO_ROOT / UNIFIED_OUTPUT_DIR
    path = out_dir / "r3_epistasis_and_contacts_unified_62.json"
    sha_path = out_dir / "r3_epistasis_and_contacts_unified_62.json.sha256"
    assert path.is_file(), f"Missing unified R3 artifact: {path}"

    expected_sha = sha_path.read_text(encoding="utf-8").strip().split()[0]
    assert sha256_file(path) == expected_sha

    doc = json.loads(path.read_text(encoding="utf-8"))
    _assert_no_nan_or_inf(doc)
    assert doc["reproduction_gate"]["passed"] is True
    assert doc["reproduction_gate"]["max_abs_diff"] < 1e-12
    assert doc["roster"]["n_single_sequence_arms"] == 62
    assert len(doc["roster"]["arms_62"]) == len(UNIFIED_62_SINGLE_SEQUENCE_ARMS) == 62
    assert len(doc["roster"]["arms_62_with_R_cycle"]) == len(UNIFIED_62_WITH_R_ARMS) == 62
    assert "esm3-1.4b" in doc["roster"]["arms_62_with_R_cycle"]

    ep = doc["epistasis"]
    assert ep["support"]["n_groups"] == 61
    assert ep["support"]["n_cycles"] == 7805
    assert ep["support"]["n_arms_M_cycle"] == 62
    assert ep["support"]["n_arms_R_and_MR_cycle"] == 62
    assert ep["support"]["max_cached_20261110_mse_diff_across_33_arms"] < 1e-12
    assert ep["M_cycle_62"]["cycle_squared_error"]["n_arms"] == 62
    assert ep["R_cycle_62"]["cycle_squared_error"]["n_arms"] == 62
    mr_mse = ep["MR_cycle_62_over_nuisance_profile_couplings"]["mse"]
    assert mr_mse["MR_given_controls"]["family_size"] == 62
    assert "esm3-1.4b" in mr_mse["MR_given_controls"]["arms"]
    assert "esm3-1.4b" in mr_mse["R_given_controls"]["arms"]

    contacts = doc["contacts"]
    e01 = contacts["perturbation_propagation_e01"]
    assert e01["readout_mechanism"] == "perturbation_propagation_e01"
    e01_all_cols = e01["discrimination"]["all"]["primary_thin_bpe_inclusive"]["columns"]
    assert sum(1 for c in e01_all_cols if c["kind"] == "model") == 22
    assert contacts["attention_apc_map"]["readout_mechanism"] == "attention_apc_map"
    assert contacts["attention_apc_map"]["n_arms"] == 16
