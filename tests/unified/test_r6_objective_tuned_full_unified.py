"""Tests for full-cohort Result 6 ESMFold2 evaluation (68/68 shards + 2,000/2,000 long_cap)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.capability.core.io import sha256_file
from src.capability.unified.r6_objective_tuned_full_esmfold2 import (
    EXPECTED_SIGNATURE,
    PTM_MARGIN_CRITERION,
    PTM_MARGIN_FLOOR,
    PTM_MARGIN_FRACTION,
    SEVEN_FAMILIES,
    _nn_match_fast,
    ptm_margin_for,
    verify_r6_reproduction_gate,
)
from src.capability.unified.roster import REPO_ROOT, UNIFIED_OUTPUT_DIR


def _assert_no_nan_or_inf(obj: object, path: str = "root") -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float at {path}: {obj}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_nan_or_inf(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_nan_or_inf(v, f"{path}[{i}]")


def test_nn_match_fast_matches_reference_loop() -> None:
    """Vectorized caliper-5 nearest-length matcher must match reference loop to < 1e-12."""
    rng = np.random.default_rng(20261124)
    lb = rng.integers(50, 401, size=250).astype(float)
    yb = rng.normal(loc=60.0, scale=10.0, size=250)
    lt = rng.integers(50, 401, size=120).astype(float)
    yt = rng.normal(loc=65.0, scale=10.0, size=120)

    # Reference implementation from length_adjusted_contrasts_20261110/analyse.py
    order = np.argsort(lb)
    ls, ys = lb[order], yb[order]
    diffs = []
    for l_val, y_val in zip(lt, yt):
        i = int(np.searchsorted(ls, l_val))
        cand = [j for j in (i - 1, i) if 0 <= j < len(ls)]
        best = min(cand, key=lambda j: abs(ls[j] - l_val))
        lo = int(np.searchsorted(ls, ls[best], "left"))
        hi = int(np.searchsorted(ls, ls[best], "right"))
        if abs(ls[best] - l_val) <= 5:
            diffs.append(y_val - float(ys[lo:hi].mean()))

    fast_val, fast_n = _nn_match_fast(lb, yb, lt, yt, cal=5)
    assert fast_n == len(diffs)
    assert abs(fast_val - float(np.mean(diffs))) < 1e-12


def test_seven_finetuning_families_count() -> None:
    assert len(SEVEN_FAMILIES) == 7


def test_r6_reproduction_gate_when_full_tsvs_present() -> None:
    cache_dir = REPO_ROOT / UNIFIED_OUTPUT_DIR / "r6_objective_tuned_full_cache"
    main_tsv = cache_dir / "checkpoint_structures_68of68.tsv"
    lc_tsv = cache_dir / "long_cap_structures_2000of2000.tsv"
    if not (main_tsv.is_file() and lc_tsv.is_file()):
        pytest.skip("Full 68/68 + 2000/2000 TSVs not yet staged locally")
    gate = verify_r6_reproduction_gate(main_tsv, lc_tsv, root=REPO_ROOT, tolerance=1e-6)
    assert gate["passed"] is True
    assert gate["main_68_total_rows"] == 47778
    assert gate["main_18_subset_rows"] == 12640
    assert gate["long_cap_total_rows"] == 2000
    assert gate["long_cap_existing_rows"] == 1748
    assert gate["long_cap_newly_folded_rows"] == 252
    assert gate["single_evaluation_signature"] == EXPECTED_SIGNATURE


def test_ptm_margin_for_uses_0_1_scale_never_plddt_100_scale() -> None:
    """Regression guard: pTM margin must stay on [0, 1] scale (never ~17.4 from pLDDT 70.0)."""
    m_low = ptm_margin_for(0.44)
    assert m_low["criterion"] == PTM_MARGIN_CRITERION == 0.50
    assert m_low["floor"] == PTM_MARGIN_FLOOR == 0.02
    assert m_low["fraction"] == PTM_MARGIN_FRACTION == 0.25
    # headroom = 0.06 -> 0.25 * 0.06 = 0.015 < floor 0.02 -> 0.02
    assert abs(float(m_low["margin"]) - 0.02) < 1e-12

    m_very_low = ptm_margin_for(0.20)
    # headroom = 0.30 -> 0.25 * 0.30 = 0.075
    assert abs(float(m_very_low["margin"]) - 0.075) < 1e-12

    m_above = ptm_margin_for(0.75)
    assert abs(float(m_above["margin"]) - 0.02) < 1e-12

    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        ptm_margin_for(44.0)


def test_r6_full_unified_artifact_integrity() -> None:
    out_dir = REPO_ROOT / UNIFIED_OUTPUT_DIR
    path = out_dir / "r6_objective_tuned_full_esmfold2_unified.json"
    sha_path = out_dir / "r6_objective_tuned_full_esmfold2_unified.json.sha256"
    if not path.is_file():
        pytest.skip("r6_objective_tuned_full_esmfold2_unified.json not yet generated")

    expected_sha = sha_path.read_text(encoding="utf-8").strip().split()[0]
    assert sha256_file(path) == expected_sha

    doc = json.loads(path.read_text(encoding="utf-8"))
    _assert_no_nan_or_inf(doc)
    assert doc["reproduction_gate"]["passed"] is True
    cov = doc["coverage_declaration"]
    assert cov["main_cohort"]["attempts"] == 48000
    assert cov["main_cohort"]["non_empty"] == 47788
    assert cov["main_cohort"]["unique_sequences"] == 47778
    assert cov["main_cohort"]["fold_coverage"] == 1.0
    assert cov["pfam_recognition_68of68"]["n_merged_total"] == 47778
    assert cov["long_cap_companion"]["n_total"] == 2000
    assert cov["long_cap_companion"]["n_folded_ok"] == 2000

    # Verify all six sections (a)-(f) are populated
    sec_a = doc["a_novelty_stratified_contrasts"]
    assert len(sec_a["primary_plddt_3stream_rows"]) == 12
    assert len(sec_a["secondary_plddt_3stream_rows"]) == 4
    assert len(sec_a["primary_ptm_3stream_rows"]) == 12
    assert len(sec_a["secondary_ptm_3stream_rows"]) == 4
    assert sec_a["primary_ptm_full"]["units"] == "ESMFold2 pTM on the 0-1 scale"
    assert sec_a["secondary_ptm_full"]["units"] == "ESMFold2 pTM on the 0-1 scale"

    # Regression check: every resolved pTM margin is <= 0.125 (0-1 scale, NEVER ~17.4 from pLDDT!),
    # and large positive pTM gains (e.g. diff > +0.20 with band_lo > margin) are "supported".
    for row in (*sec_a["primary_ptm_3stream_rows"], *sec_a["secondary_ptm_3stream_rows"]):
        if row["status"] == "resolved":
            assert row["margin"] is not None
            assert 0.02 <= float(row["margin"]) <= 0.125
            if float(row["band_lo"]) > float(row["margin"]):
                assert row["finding"] == "supported"
                assert row["zero_verdict"] == "positive"

    assert len(doc["b_identity_shift_and_pfam_recognition"]["progen2_mix7_seven_family_attribution"]["out_of_seven_families_contrast"]) == 4
    assert set(doc["c_filter_controls_and_selector_competence"]["selection_curve_k2_k4_k8"].keys()) == {"k2", "k4", "k8"}
    assert doc["d_length_and_cap_confounding_audit"]["long_cap_768_companion_2000of2000"]["n_total"] == 2000
    assert len(doc["e_five_mer_ties_and_length_matched_shares"]["ties"]) == 36
    assert len(doc["f_sampled_18of68_vs_full_68of68_comparison"]["main_18of68_vs_68of68"]) == 16
