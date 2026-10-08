"""Conditions the small-domain stability predictor must always satisfy.

The point of this experiment is that a stability number must be earned, so the
cases here are mostly about refusal: that a structural confidence can never be
compared with a free energy, that the predictor is refused outside the length
band its training data covers, that an instrument which only matches a
composition baseline does not pass its gate, and that a gate which cannot be
resolved is a failure rather than a pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.capability.evaluation import domain_stability as ds


# ------------------------------------------------------- quantity separation


def test_a_confidence_is_never_comparable_with_a_free_energy():
    """The substitution this experiment exists to prevent, as a crash.

    pLDDT and pTM are the structure predictor's confidence in its own
    coordinates. An undocumented thermostability score is a third thing again.
    None of them may be compared with, converted into, or averaged with a folding
    free energy, and the guard raises rather than warning.
    """

    for other in ("esmfold2_mean_ca_plddt", "esmfold2_ptm", "prime_value_head"):
        with pytest.raises(ValueError, match="different physical quantities"):
            ds.require_comparable("predicted_delta_g", other)
        with pytest.raises(ValueError):
            ds.require_comparable(other, "measured_delta_g")


def test_a_prediction_is_comparable_with_the_measurement_of_the_same_quantity():
    """Predicted against measured is the validation and must be allowed.

    The axis that is enforced is the physical quantity, not whether a reading was
    predicted or measured. Enforcing the latter would forbid the regression this
    whole module is built to validate.
    """

    ds.require_comparable("predicted_delta_g", "measured_delta_g")
    ds.require_comparable("measured_delta_g", "predicted_delta_g")
    ds.require_comparable("esmfold2_mean_ca_plddt", "esmfold2_mean_ca_plddt")
    assert (
        ds.QUANTITIES["predicted_delta_g"]["kind"]
        == ds.QUANTITIES["measured_delta_g"]["kind"]
        == "folding_free_energy"
    )
    assert ds.QUANTITIES["predicted_delta_g"]["source"] == "predicted"
    assert ds.QUANTITIES["measured_delta_g"]["source"] == "measured"


def test_two_readings_of_one_quantity_in_different_units_are_refused():
    with pytest.raises(ValueError, match="would have"):
        ds.require_comparable("esmfold2_mean_ca_plddt", "esmfold2_ptm")


def test_an_undeclared_quantity_cannot_be_compared_at_all():
    with pytest.raises(KeyError):
        ds.require_comparable("predicted_delta_g", "some_new_score")


# --------------------------------------------------------------- length band


def test_the_predictor_is_refused_outside_its_training_band():
    ds.require_in_band([60, 70, 80], (60, 80), label="in band")
    with pytest.raises(ValueError, match=r"outside the licensed band"):
        ds.require_in_band([60, 400], (60, 80), label="a censored continuation")
    with pytest.raises(ValueError, match=r"outside the licensed band"):
        ds.require_in_band([45], (60, 80), label="below the band")
    with pytest.raises(ValueError, match="nothing to score"):
        ds.require_in_band([], (60, 80), label="empty")


def test_the_licensed_band_comes_from_the_training_lengths():
    assert ds.licensed_band([60, 69, 80, 61]) == (60, 80)
    with pytest.raises(ValueError):
        ds.licensed_band([])


# ------------------------------------------------------- the cheap baseline


def test_composition_features_are_fractions_plus_length():
    vector = ds.composition_features("AAAACCCCGGGGWWWW")
    assert vector.size == len(ds.BASELINE_FEATURES)
    assert ds.BASELINE_FEATURES[-2:] == ("length", "log_length")
    assert vector[: len(ds.AA20)].sum() == pytest.approx(1.0)
    assert vector[len(ds.AA20)] == 16.0
    with pytest.raises(ValueError, match="non-canonical"):
        ds.composition_features("AAAX")
    with pytest.raises(ValueError):
        ds.composition_features("")


# --------------------------------------------------------------- the ridge


def _linear_problem(n: int = 2000, p: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.normal(0.0, 3.0, (n, p))
    truth = rng.normal(0.0, 1.0, p)
    targets = features @ truth + 5.0 + rng.normal(0.0, 0.05, n)
    return features, targets, truth


def test_chunked_ridge_recovers_a_known_linear_relationship():
    features, targets, truth = _linear_problem()
    gram = ds.RidgeGram(features.shape[1])
    for start in range(0, features.shape[0], 137):
        gram.add(features[start : start + 137], targets[start : start + 137])
    gram.finalise()
    model = gram.solve(1e-6)
    predicted = ds.predict_with(model, features)
    assert np.corrcoef(predicted, targets)[0, 1] > 0.999
    # Coefficients are on the standardised design, so compare after rescaling.
    recovered = model["coefficients"] / model["scale"]
    assert np.allclose(recovered, truth, atol=0.02)


def test_chunking_does_not_change_the_ridge_solution():
    features, targets, _ = _linear_problem()
    solutions = []
    for chunk in (features.shape[0], 500, 71):
        gram = ds.RidgeGram(features.shape[1])
        for start in range(0, features.shape[0], chunk):
            gram.add(features[start : start + chunk], targets[start : start + chunk])
        gram.finalise()
        solutions.append(gram.solve(10.0)["coefficients"])
    for other in solutions[1:]:
        assert np.allclose(solutions[0], other, rtol=1e-8, atol=1e-10)


def test_a_constant_feature_does_not_divide_by_zero():
    features, targets, _ = _linear_problem(p=6)
    features = np.hstack([features, np.full((features.shape[0], 1), 7.0)])
    gram = ds.RidgeGram(features.shape[1])
    gram.add(features, targets)
    gram.finalise()
    assert gram.n_constant_features == 1
    model = gram.solve(1.0)
    assert np.isfinite(model["coefficients"]).all()


def test_the_ridge_refuses_a_design_it_cannot_identify():
    gram = ds.RidgeGram(40)
    gram.add(np.zeros((10, 40)) + np.eye(10, 40), np.arange(10.0))
    with pytest.raises(ValueError, match="cannot identify"):
        gram.finalise()


def test_the_ridge_refuses_non_finite_input_and_an_unpenalised_solve():
    gram = ds.RidgeGram(3)
    with pytest.raises(ValueError, match="non-finite"):
        gram.add(np.array([[1.0, 2.0, np.nan]]), np.array([1.0]))
    features, targets, _ = _linear_problem(p=3)
    gram.add(features, targets)
    gram.finalise()
    with pytest.raises(ValueError, match="penalty must be positive"):
        gram.solve(0.0)


# ------------------------------------------------------------ the validation


def test_error_in_kcal_per_mol_is_reported_only_when_the_units_allow_it():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    prediction = truth + 0.5
    free_energy = ds.regression_report(truth, prediction, unit="kcal_per_mol")
    assert free_energy["rmse"] == pytest.approx(0.5)
    assert free_energy["bias"] == pytest.approx(0.5)
    ordinal = ds.regression_report(truth, prediction, unit="undeclared_by_model_card")
    assert ordinal["rmse"] is None and ordinal["mae"] is None
    assert "no units" in ordinal["error_not_reported_because"]
    assert ordinal["spearman"] == pytest.approx(1.0)


def _gate_inputs(n_clusters: int = 60, per_cluster: int = 20, seed: int = 3):
    rng = np.random.default_rng(seed)
    groups, truth, baseline, good, noise = [], [], [], [], []
    for cluster in range(n_clusters):
        offset = rng.normal(0.0, 2.0)
        for _ in range(per_cluster):
            value = offset + rng.normal(0.0, 1.0)
            groups.append(f"c{cluster}")
            truth.append(value)
            baseline.append(value * 0.1 + rng.normal(0.0, 2.0))
            good.append(value + rng.normal(0.0, 0.4))
            noise.append(rng.normal(0.0, 1.0))
    return (
        np.asarray(truth),
        np.asarray(baseline),
        np.asarray(good),
        np.asarray(noise),
        groups,
    )


def test_an_instrument_that_transfers_and_beats_the_baseline_passes_the_gate():
    truth, baseline, good, _, groups = _gate_inputs()
    contrast = ds.transfer_contrast(truth, good, baseline, groups, seed=1, n_bootstrap=2000)
    gate = ds.evaluate_gate(contrast)
    assert contrast["resolved"] is True
    assert contrast["instrument_spearman"] > contrast["baseline_spearman"]
    assert gate["passed"] is True
    assert gate["G1_transfers"] and gate["G2_beats_the_cheap_baseline"]


def test_an_instrument_that_does_not_transfer_fails_the_first_condition():
    truth, baseline, _, noise, groups = _gate_inputs()
    gate = ds.evaluate_gate(
        ds.transfer_contrast(truth, noise, baseline, groups, seed=1, n_bootstrap=2000)
    )
    assert gate["passed"] is False
    assert gate["G1_transfers"] is False
    assert "G1 failed" in gate["reason"]


def test_an_instrument_that_only_matches_the_cheap_baseline_fails_the_second():
    """Correlating with the target is not enough; it has to add to a free feature."""

    truth, baseline, _, _, groups = _gate_inputs()
    gate = ds.evaluate_gate(
        ds.transfer_contrast(truth, baseline.copy(), baseline, groups, seed=1, n_bootstrap=2000)
    )
    assert gate["G2_beats_the_cheap_baseline"] is False
    assert gate["passed"] is False
    assert "adds nothing to a free feature" in gate["reason"]


def test_an_unresolvable_gate_is_a_failure_and_not_a_pass():
    truth, baseline, good, _, _ = _gate_inputs(n_clusters=3, per_cluster=5)
    contrast = ds.transfer_contrast(
        truth, good, baseline, ["a", "b", "c"] * 5, seed=1, n_bootstrap=2000
    )
    assert contrast["resolved"] is False
    gate = ds.evaluate_gate(contrast)
    assert gate["passed"] is False
    assert "not resolvable" in gate["reason"]


def test_the_gate_conditions_are_declared_and_choose_no_threshold():
    for key in ("G1_transfers", "G2_beats_the_cheap_baseline", "both_required", "unit_of_resampling"):
        assert ds.GATE_CONDITIONS[key].strip()
    # Both conditions are "an interval excludes zero", so there is no magic
    # number anywhere that could have been chosen after seeing the result.
    joined = " ".join(ds.GATE_CONDITIONS.values())
    assert "excluding zero" in joined


# ------------------------------------------------------- third-party loading


def test_the_compatibility_shim_rebinds_names_without_touching_the_download():
    """Patching somebody else's code is part of how a number was obtained.

    The shim rebinds moved names on the transformers module in this process. It
    must report what it rebound, must claim no file was edited, and must be safe
    to call twice.
    """

    import transformers.modeling_utils as modeling_utils

    first = ds.transformers_compatibility_shim()
    assert first["files_modified_on_disk"] == []
    assert first["source_module"] == "transformers.pytorch_utils"
    for name in first["rebound_into_transformers_modeling_utils"]:
        assert hasattr(modeling_utils, name)
    second = ds.transformers_compatibility_shim()
    assert second["rebound_into_transformers_modeling_utils"] == []
