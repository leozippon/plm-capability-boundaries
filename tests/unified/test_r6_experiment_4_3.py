"""Tests for unified 30-model / 33-pool Experiment 4.3 (Native Self-Likelihood Ranking)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.capability.core.io import sha256_file
from src.capability.unified.r6_experiment_4_3 import (
    E17_FROZEN_RANKING_REL,
    E17_REPRODUCTION_TOLERANCE,
    POOL_METADATA,
    PRIMARY_33_POOLS,
    REGIMES,
    SCHEMA,
    _bootstrap_spearman_and_selection,
    distinct_kmer_fraction,
    load_e17_frozen_pools,
    run_e17_reproduction_gate,
    sequence_complexity_features,
)
from src.capability.unified.roster import REPO_ROOT, UNIFIED_OUTPUT_DIR

OUTPUT_JSON = REPO_ROOT / UNIFIED_OUTPUT_DIR / "r6_experiment_4_3_native_likelihood_unified.json"
HAS_E17 = (REPO_ROOT / E17_FROZEN_RANKING_REL).is_file()


def test_pool_roster_30_checkpoints_33_pools() -> None:
    assert len(PRIMARY_33_POOLS) == 33
    unique_ckpts = {POOL_METADATA[p]["arm_id"] for p in PRIMARY_33_POOLS}
    assert len(unique_ckpts) == 30
    uncond_ckpts = {POOL_METADATA[p]["arm_id"] for p in PRIMARY_33_POOLS if not POOL_METADATA[p]["conditioned"]}
    assert len(uncond_ckpts) == 29
    assert "zymctrl" in unique_ckpts


def test_sequence_complexity_and_adverse_selection_orientation() -> None:
    rep = sequence_complexity_features("A" * 50)
    assert rep["entropy_bits"] == pytest.approx(0.0, abs=1e-12)
    assert rep["homopolymer_frac"] == pytest.approx(1.0, abs=1e-12)
    assert rep["distinct_5mer"] == pytest.approx(1.0 / 46.0, rel=1e-9)

    diverse = sequence_complexity_features("ACDEFGHIKLMNPQRSTVWY" * 2)
    assert diverse["entropy_bits"] == pytest.approx(np.log2(20.0), rel=1e-9)
    assert diverse["longest_run"] == 1
    assert distinct_kmer_fraction("ACDEFGHIK", 5) == pytest.approx(1.0)

    # Synthetic adverse selection: lower NLL (higher likelihood) strictly predicts LOWER pLDDT
    rng = np.random.default_rng(42)
    n = 80
    nll = np.linspace(0.5, 3.0, n)  # lower NLL = more likely
    plddt = 40.0 + 15.0 * nll + rng.normal(0.0, 1.0, size=n)  # higher NLL = better pLDDT -> adverse!
    ptm = 0.2 + 0.15 * nll + rng.normal(0.0, 0.01, size=n)
    d5 = 0.3 + 0.2 * nll
    units = [f"u{i}" for i in range(n)]
    res = _bootstrap_spearman_and_selection(
        nll,
        {"plddt": plddt, "ptm": ptm, "distinct_5mer": d5},
        units,
        n_draws=2000,
        seed=20261011,
    )
    assert res["spearman"]["plddt"]["point"] < -0.9
    assert res["spearman"]["plddt"]["verdict"] == "negative"
    sel10 = res["truncation_selection"]["plddt"]["top_10pct"]
    assert sel10["gain_over_random"] < -10.0
    assert sel10["point_adverse"] is True
    assert sel10["adverse_class"] == "adverse_ci_negative"


@pytest.mark.skipif(not HAS_E17, reason="frozen E17 inputs absent locally")
def test_e17_reproduction_gate_exact() -> None:
    e17 = load_e17_frozen_pools(REPO_ROOT)
    gate = run_e17_reproduction_gate(e17, root=REPO_ROOT)
    assert gate["passed"] is True
    assert gate["max_abs_diff"] < E17_REPRODUCTION_TOLERANCE
    assert gate["arms"]["protgpt2"]["n_pool"] == 600
    assert gate["arms"]["zymctrl"]["n_pool"] == 600


@pytest.mark.skipif(not OUTPUT_JSON.is_file(), reason="Experiment 4.3 output JSON not yet generated")
def test_generated_artifact_integrity_and_schema() -> None:
    sidecar = OUTPUT_JSON.with_name(OUTPUT_JSON.name + ".sha256")
    assert sidecar.is_file()
    expected_sha = sidecar.read_text(encoding="utf-8").strip().split()[0]
    assert sha256_file(OUTPUT_JSON) == expected_sha

    doc = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    assert doc["schema"] == SCHEMA
    assert doc["e17_reproduction_gate"]["passed"] is True
    assert doc["e17_reproduction_gate"]["max_abs_diff"] < E17_REPRODUCTION_TOLERANCE
    assert doc["n_unique_checkpoints"] == 30
    assert doc["n_primary_pools"] == 33
    assert len(doc["pools"]) == 33
    for reg_name, _, _ in REGIMES:
        assert reg_name in doc["cross_model_summary"]
    assert doc["companion_verifications"]["objective_tuned_base_likelihood_20261114"]["status"] == "complete"
