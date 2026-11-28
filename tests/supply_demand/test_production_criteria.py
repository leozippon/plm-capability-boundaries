"""Unit and integration tests for Section-IV production engineering criteria."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.capability.supply_demand.production_criteria import (
    DEFAULT_K_FAM_COUPLED_BITS,
    DEFAULT_K_UNIV_BITS,
    PRODUCTION_CRITERIA_SCHEMA,
    analyze_probing_erasure,
    compute_objective_phase_boundary,
    compute_optimal_model_size,
    compute_safe_stopping_horizon,
    compute_structural_parameter_equivalence,
    default_supply_profiles,
    estimate_k_useful_bits,
    load_calibrated_supply_profiles,
    synthesize_production_criteria,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_k_useful_scaling_and_narrow_m_diversity_warning() -> None:
    cross_narrow = estimate_k_useful_bits(64, "cross_fold_physics")
    assert cross_narrow["K_useful_bits"] == pytest.approx(DEFAULT_K_UNIV_BITS)
    assert cross_narrow["K_spec_useful_bits"] == 0.0
    assert cross_narrow["narrow_M_diversity_warning"] is True

    cross_wide = estimate_k_useful_bits(1024, "cross_fold_physics")
    assert cross_wide["K_useful_bits"] == pytest.approx(DEFAULT_K_UNIV_BITS)
    assert cross_wide["narrow_M_diversity_warning"] is False

    fit_small = estimate_k_useful_bits(1_000, "in_family_fitness", k_fam_bits=1_500.0)
    fit_large = estimate_k_useful_bits(20_000_000, "in_family_fitness", k_fam_bits=1_500.0)
    assert fit_small["K_spec_useful_bits"] == pytest.approx(1_500_000.0)
    assert fit_large["K_spec_useful_bits"] == pytest.approx(30_000_000_000.0)
    assert fit_large["K_useful_bits"] > fit_small["K_useful_bits"]

    with pytest.raises(ValueError, match="n_families"):
        estimate_k_useful_bits(0, "cross_fold_physics")
    with pytest.raises(ValueError, match="Unsupported task_type"):
        estimate_k_useful_bits(100, "unknown_task")


def test_optimal_sizing_stopping_and_external_conditioning_equivalence() -> None:
    profiles = default_supply_profiles()
    k_use = 15_000_000.0  # 15 Mbit universal physics

    # Asymptotic P* = K_net / c_inf: AR needs far fewer params than MDM at equal K_net
    p_ar = compute_optimal_model_size(k_use, "ar", 20.0, supply_profiles=profiles)
    p_mdm = compute_optimal_model_size(k_use, "mdm", 20.0, supply_profiles=profiles)
    assert p_ar["P_star_info_asymptotic"] == pytest.approx(k_use / profiles["ar"].c_inf_bpp)
    assert p_mdm["P_star_info_asymptotic"] == pytest.approx(k_use / profiles["oa_uniform"].c_inf_bpp)
    assert p_ar["P_star_info_asymptotic"] < p_mdm["P_star_info_asymptotic"]

    # Compute floor dominates when P_compute > P_info
    p_floor = compute_optimal_model_size(k_use, "ar", 20.0, supply_profiles=profiles, compute_floor_params=50_000_000)
    assert p_floor["P_star_recommended"] == pytest.approx(50_000_000.0)

    # External supply B_ext(Z) reduces net demand and saves Delta_P_saved = B_ext / c_eff
    b_ext = 10_000_000.0
    p_cond = compute_optimal_model_size(k_use, "ar", 20.0, supply_profiles=profiles, b_ext_bits=b_ext)
    equiv = compute_structural_parameter_equivalence(b_ext, "ar", supply_profiles=profiles)
    assert p_cond["K_net_bits"] == pytest.approx(5_000_000.0)
    assert p_ar["P_star_info_asymptotic"] - p_cond["P_star_info_asymptotic"] == pytest.approx(
        equiv["Delta_P_saved_asymptotic"]
    )

    # Under-capacity AR (2M params * 3.69 bpp = 7.38 Mbit < 15 Mbit) is safe from K_res overflow
    stop_under = compute_safe_stopping_horizon(2_000_000.0, k_use, "ar", supply_profiles=profiles)
    assert stop_under["stopping_regime"] == "under_capacity_safe"
    assert stop_under["E_max_safe_epochs"] is None

    # Over-capacity AR (35M params * 3.69 bpp = 129 Mbit >> 15 Mbit) requires strict early stopping;
    # MDM at the same 35M scale has a vastly longer safe horizon than AR!
    stop_over_ar = compute_safe_stopping_horizon(35_000_000.0, k_use, "ar", supply_profiles=profiles)
    stop_over_mdm = compute_safe_stopping_horizon(35_000_000.0, k_use, "mdm", supply_profiles=profiles)
    assert stop_over_ar["E_max_safe_epochs"] is not None
    assert stop_over_ar["E_max_safe_epochs"] < 30.0
    # 35M * 0.29 = 10.15 Mbit < 15 Mbit -> MDM is actually under-capacity safe at 35M!
    assert stop_over_mdm["stopping_regime"] == "under_capacity_safe"


def test_objective_phase_boundary_transitions() -> None:
    profiles = default_supply_profiles()
    k_use = 15_000_000.0

    # Under-capacity: P=2M -> eta_AR = 7.38 / 15 = 0.492 < 1.0 -> AR favorable
    pb_under = compute_objective_phase_boundary(2_000_000.0, k_use, supply_profiles=profiles, planned_epochs=50.0)
    assert pb_under["eta_AR"] < 1.0
    assert pb_under["regime"] == "under_capacity_ar_favorable"
    assert pb_under["recommended_objective"] == "ar"

    # Transition: P=10M -> eta_AR = 2.46 >= 1, eta_MLM = 0.53 < 1, E=50 <= E_mem(MLM) -> MLM-15% favorable
    pb_mid = compute_objective_phase_boundary(10_000_000.0, k_use, supply_profiles=profiles, planned_epochs=50.0)
    assert pb_mid["eta_AR"] >= 1.0
    assert pb_mid["eta_MLM"] < 1.0
    assert pb_mid["regime"] == "transition_mlm_favorable"
    assert pb_mid["recommended_objective"] == "mlm_15"

    # Over-capacity multi-epoch: P=100M -> eta_AR = 24.6, eta_MLM = 5.27 >= 1 -> MDM favorable
    pb_over = compute_objective_phase_boundary(100_000_000.0, k_use, supply_profiles=profiles, planned_epochs=50.0)
    assert pb_over["regime"] == "over_capacity_mdm_favorable"
    assert pb_over["recommended_objective"] == "oa_uniform"

    # Externally satisfied: B_ext >= K_useful
    pb_ext = compute_objective_phase_boundary(
        10_000_000.0, k_use, supply_profiles=profiles, b_ext_bits=20_000_000.0
    )
    assert pb_ext["regime"] == "externally_satisfied"
    assert pb_ext["K_net_bits"] == 0.0


def test_monotone_envelope_and_real_repo_synthesis_if_present() -> None:
    s1 = REPO_ROOT / "output/stage1_wsd/stage1_combined_wave3_summary.json"
    s2 = REPO_ROOT / "output/stage2_inversion/stage2_wave3_analysis_summary.json"
    bm = REPO_ROOT / "output/stage2_bundle/bundle_manifest.json"
    pr = REPO_ROOT / "output/probing_all43"
    if not s1.is_file():
        pytest.skip("Local Stage-1 summary not staged")

    profiles = load_calibrated_supply_profiles(s1, stage2_summary_path=s2 if s2.is_file() else None)
    assert set(profiles.keys()) >= {"ar", "mlm_15", "oa_uniform"}
    # Monotone envelope must be strictly non-decreasing (eliminating any single-epoch dip)
    for prof in profiles.values():
        cs = prof.c_mem_envelope_bpp
        assert all(cs[i] <= cs[i + 1] + 1e-12 for i in range(len(cs) - 1))
        assert prof.w_window > 1.0
    assert profiles["ar"].c_inf_bpp > profiles["mlm_15"].c_inf_bpp > profiles["oa_uniform"].c_inf_bpp
    assert profiles["oa_uniform"].e_mem_epochs > profiles["mlm_15"].e_mem_epochs > profiles["ar"].e_mem_epochs

    if pr.is_dir():
        probe_syn = analyze_probing_erasure(pr)
        assert probe_syn["n_checkpoints"] == 43
        nat = probe_syn["natural_arms"]["by_objective"]
        # Natural arms: MDM has near-zero late-layer erasure while AR suffers severe erasure
        assert abs(nat["oa_uniform"]["burial_erasure"]) < abs(nat["ar"]["burial_erasure"])
        assert abs(nat["oa_uniform"]["contacts_lr_erasure"]) < abs(nat["ar"]["contacts_lr_erasure"])
        assert probe_syn["natural_arms"]["spearman_with_Delta_cal"]["burial_erasure"]["spearman_r"] < -0.5

    report = synthesize_production_criteria(
        stage1_summary_path=s1,
        stage2_summary_path=s2 if s2.is_file() else None,
        stage2_bundle_manifest_path=bm if bm.is_file() else None,
        probing_dir=pr if pr.is_dir() else None,
    )
    assert report["schema"] == PRODUCTION_CRITERIA_SCHEMA
    assert len(report["production_scenarios"]) >= 5
    # Verify strict RFC-8259 JSON serializability (no NaN/Infinity)
    dumped = json.dumps(report, allow_nan=False)
    assert PRODUCTION_CRITERIA_SCHEMA in dumped
    assert DEFAULT_K_FAM_COUPLED_BITS == 50_000.0
