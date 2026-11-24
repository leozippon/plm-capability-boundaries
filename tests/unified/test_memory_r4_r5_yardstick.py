"""Unit & integration tests for Workstream C unified family memory and R4/R5/R6 yardstick."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.unified.memory_and_generation_yardstick import (
    _run_simultaneous_family,
    build_family_memory_unified_62,
    build_r4_r5_r6_unified_summary,
)
from src.capability.unified.roster import UNIFIED_62_SINGLE_SEQUENCE_ARMS


def test_simultaneous_family_nats_bits_scaling_and_verdicts() -> None:
    rng = np.random.default_rng(123)
    # 40 groups x 4 arms: arm0 clean positive, arm1 clean negative, arm2 zero-mean, arm3 huge-variance noisy
    mat = np.column_stack(
        [
            0.5 + rng.normal(0.0, 0.05, size=40),
            -0.5 + rng.normal(0.0, 0.05, size=40),
            rng.normal(0.0, 0.25, size=40),
            -4.0 + rng.normal(0.0, 5.0, size=40),
        ]
    )
    res = _run_simultaneous_family(["a0", "a1", "a2", "a3_noisy"], mat, draws=1000, seed=20261106)
    assert res["family_size"] == 4
    assert res["groups"] == 40
    # Under native unstandardized bands, a3_noisy widens critical_value so a0/a1 become unresolved,
    # whereas studentized maxT stays scale-invariant and resolves a0 positive and a1 negative.
    assert res["per_arm"]["a0"]["studentized_verdict"] == "positive"
    assert res["per_arm"]["a1"]["studentized_verdict"] == "negative"
    assert res["studentized_verdict_counts"]["positive"] >= 1
    assert res["studentized_verdict_counts"]["negative"] >= 1
    assert math.isclose(
        res["critical_value_bits"],
        res["critical_value_nats"] / math.log(2.0),
        rel_tol=1e-12,
    )
    assert math.isclose(
        res["per_arm"]["a0"]["point_bits"],
        res["per_arm"]["a0"]["point_nats"] / math.log(2.0),
        rel_tol=1e-12,
    )
    assert math.isclose(
        res["per_arm"]["a0"]["studentized_ci95_bits"][0],
        res["per_arm"]["a0"]["studentized_ci95_nats"][0] / math.log(2.0),
        rel_tol=1e-12,
    )
    assert math.isclose(
        res["per_arm"]["a0"]["pointwise_ci95_bits"][1],
        res["per_arm"]["a0"]["pointwise_ci95_nats"][1] / math.log(2.0),
        rel_tol=1e-12,
    )


def test_family_memory_unified_62_live_invariants() -> None:
    # Fast draws=200 smoke of the full live 62/46-arm builder on repository data
    doc = build_family_memory_unified_62(
        REPO_ROOT, draws=200, seed=20261106, scaling_bootstrap_draws=200
    )
    fc = doc["family_conditionals_62"]
    assert fc["n_arms"] == 62
    assert fc["arms_order"] == list(UNIFIED_62_SINGLE_SEQUENCE_ARMS)
    assert fc["baseline_parity_verification"]["passed"] is True
    assert fc["baseline_parity_verification"]["max_abs_profile_logp_diff_nats"] < 1e-6
    assert fc["baseline_parity_verification"]["max_abs_markov_logp_diff_nats"] < 1e-6
    assert fc["arm_metadata"]["esm2-15b"]["conditional_estimator"] == "masked_1pass"
    assert fc["arm_metadata"]["progen2-xlarge"]["conditional_estimator"] == "causal_20string"
    assert fc["arm_metadata"]["dayhoff-170m"]["dtype_realised"] == "torch.float32"
    assert fc["site_conservation_quartiles_34"]["n_arms_with_site_level_scores"] == 34
    assert len(fc["scaling_ladders_9"]) == 9
    overall_prof = fc["simultaneous_bands_by_stratum"]["overall_64"]["model_minus_profile"]
    assert overall_prof["per_arm"]["esmc-6b"]["studentized_verdict"] == "positive"
    assert overall_prof["studentized_verdict_counts"]["positive"] >= 1

    fd = doc["family_distribution_46"]
    assert fd["n_arms"] == 46
    assert fd["arm_metadata"]["esm2-15b"]["likelihood_objective"] == "masked_pll"
    assert fd["arm_metadata"]["esm2-15b"]["n_shards"] == 4
    assert fd["arm_metadata"]["progen2-xlarge"]["likelihood_objective"] == "exact_causal_nll"
    assert len(fd["scaling_ladders_4"]) == 4


def test_r4_r5_r6_unified_summary_live_invariants() -> None:
    doc = build_r4_r5_r6_unified_summary(REPO_ROOT)
    r4 = doc["r4_context_access"]
    assert r4["n_primary_sequence_context_arms"] == 30
    assert r4["primary_sequence_context_arms"]["msa-transformer-100m"]["context_paradigm"] == "msa_aligned"
    assert r4["primary_sequence_context_arms"]["progen3-1b"]["context_paradigm"] == "single_chain_causal"
    assert "dayhoff-3b-msa|b4096" in r4["companion_long_and_unaligned_controls"]
    assert "msa-transformer-100m|unaligned_control" in r4["companion_long_and_unaligned_controls"]
    assert r4["multimodal_3d_ss8_comparator"]["context_paradigm"] == "multimodal_3d_ss8"

    r5 = doc["r5_generative_expression"]
    assert r5["n_exp_r2_246_causal_arms"] == 17
    assert r5["n_gp_and_esm3_primary_arms"] == 13
    assert r5["n_esmfold2_structure_evaluations"] == 16
    assert r5["general_plm_and_esm3_primary_arms"]["dplm-3b"]["diffusion_sampling_mode"] == "uniform_random"
    assert r5["dplm_max_confidence_homopolymer_diagnostics"]["dplm-3b"]["diffusion_sampling_mode"] == "max_confidence"

    r6 = doc["r6_post_training_alignment"]
    assert r6["n_contrast_pairs_with_band_tables"] >= 4
    assert len(r6["selector_competence_per_arm"]) == 8
    assert len(r6["distinct_5mer_filter_control"]) > 0
