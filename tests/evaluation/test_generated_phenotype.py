"""Conditions the generated-phenotype evaluation must always satisfy.

The cases here are the ones that decide whether E14 and E17 mean anything: that a
natively finished product is never pooled with a budget-censored fragment, that
the natural comparator is drawn without replacement at matched length, that a
contrast below the package's unit floor returns no interval, that every selection
method is compared at the same selected-set size, and that stability is never
reported as measured.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.capability.evaluation import generated_phenotype as gp


def attempt(**overrides):
    row = {
        "id": "ge_" + "0" * 24,
        "arm": "protgpt2",
        "campaign": "replicate_1",
        "condition": "unconditioned",
        "class_key": None,
        "length": 120,
        "sequence": "M" + "A" * 119,
        "decoder_stop": "eos",
        "valid_aa20": True,
    }
    row.update(overrides)
    return row


# --------------------------------------------------------------- termination


def test_outcome_classes_separate_finished_products_from_fragments():
    assert gp.classify_outcome(attempt(length=120)) == "native"
    # A stop token emitted immediately is a native termination that made nothing.
    assert gp.classify_outcome(attempt(length=0)) == "native_degenerate"
    assert gp.classify_outcome(attempt(length=1)) == "native_degenerate"
    assert gp.classify_outcome(attempt(length=400)) == "native_outside_band"
    assert gp.classify_outcome(attempt(decoder_stop="max_new_tokens", length=400)) == "censored"
    assert (
        gp.classify_outcome(attempt(decoder_stop="max_new_tokens", length=120))
        == "censored_outside_band"
    )
    assert gp.classify_outcome(attempt(valid_aa20=False)) == "non_canonical_residues"
    assert gp.classify_outcome(attempt(decoder_stop="")).startswith("unknown_decoder_stop")


def test_termination_rate_is_counted_on_the_stop_reason_not_on_eligibility():
    """A censored attempt carrying a non-canonical residue is still censored.

    Counting termination through the eligibility classes undercounts censoring,
    because a non-canonical product is classified by its composition. The census
    therefore reports the stop reason separately, and that is the termination
    fact.
    """

    rows = [
        attempt(id="a", decoder_stop="max_new_tokens", length=400),
        attempt(id="b", decoder_stop="max_new_tokens", length=400, valid_aa20=False),
        attempt(id="c", decoder_stop="eos", length=120),
        attempt(id="d", decoder_stop="eos", length=0),
    ]
    census = gp.termination_census(rows)
    assert census["decoder_stop"] == {"eos": 2, "max_new_tokens": 2}
    assert census["budget_censored_fraction"] == pytest.approx(0.5)
    assert census["outcome_classes_pooled"]["non_canonical_residues"] == 1
    assert census["outcome_classes_pooled"]["native_degenerate"] == 1
    assert census["budget_censored_per_cell_mean"] == pytest.approx(2.0)


def test_census_refuses_an_empty_attempt_set():
    with pytest.raises(ValueError):
        gp.termination_census([])


# -------------------------------------------------------- natural comparator


def natural(index: int, length: int) -> dict:
    return {"id": f"nat_{index}", "sequence": "A" * length, "length": length}


def test_natural_comparator_is_matched_on_length_and_drawn_without_replacement():
    targets = [{"id": f"ge_{i}", "length": 100} for i in range(3)]
    population = [natural(i, 100) for i in range(3)] + [natural(9, 300)]
    pairs, report = gp.match_natural_records(targets, population, tolerance=2, seed=7)
    assert len(pairs) == 3
    assert report["n_unmatched"] == 0
    assert report["max_absolute_length_difference"] == 0.0
    assert report["comparator_kind"] == "whole_natural_record"
    assert len({pair["natural"]["id"] for pair in pairs}) == 3


def test_an_unmatchable_length_band_is_reported_rather_than_filled():
    targets = [{"id": "ge_0", "length": 100}, {"id": "ge_1", "length": 900}]
    pairs, report = gp.match_natural_records(targets, [natural(0, 101)], tolerance=2, seed=7)
    assert [pair["pair_id"] for pair in pairs] == ["ge_0"]
    assert report["n_unmatched"] == 1
    assert report["unmatched"] == [{"id": "ge_1", "length": 900}]
    assert report["max_absolute_length_difference"] == 1.0


def test_matching_nothing_is_refused_rather_than_reported_against_nothing():
    with pytest.raises(ValueError):
        gp.match_natural_records(
            [{"id": "ge_0", "length": 100}], [natural(0, 500)], tolerance=2, seed=7
        )


# ---------------------------------------------------------- cheap descriptors


def test_composition_background_is_smoothed_over_every_canonical_residue():
    background = gp.residue_background(["AAAA"])
    assert set(background) == set(gp.AA20)
    assert min(background.values()) > 0.0
    assert background["A"] > background["W"]


def test_composition_score_prices_a_rare_residue_run_worse_than_natural_composition():
    """The score discriminates only against a non-uniform background.

    Fitted on a uniform composition it is constant by construction, which is the
    honest behaviour and the reason the background is fitted on whole Swiss-Prot
    records rather than on anything convenient.
    """

    # Leucine common, tryptophan rare, as in natural protein composition.
    background = gp.residue_background(["L" * 100 + "A" * 80 + gp.AA20 * 5 + "W"])
    natural_like = gp.composition_cross_entropy("LLLAAALAAL" * 6, background)
    rare_run = gp.composition_cross_entropy("W" * 60, background)
    assert rare_run > natural_like

    uniform = gp.residue_background([gp.AA20 * 20])
    assert gp.composition_cross_entropy("W" * 60, uniform) == pytest.approx(
        gp.composition_cross_entropy(gp.AA20 * 3, uniform)
    )


def test_composition_score_refuses_a_background_that_cannot_price_a_residue():
    with pytest.raises(ValueError):
        gp.composition_cross_entropy("AAA", {"W": 1.0})
    with pytest.raises(ValueError):
        gp.composition_cross_entropy("XXX", gp.residue_background(["AAAA"]))


def test_repertoire_separates_a_concentrated_set_from_an_even_one():
    even = gp.family_repertoire([["PF1"], ["PF2"], ["PF3"], ["PF4"]])
    concentrated = gp.family_repertoire([["PF1"], ["PF1"], ["PF1"], ["PF4"]])
    assert even["distinct_families"] == 4
    assert concentrated["distinct_families"] == 2
    assert even["effective_families"] > concentrated["effective_families"]
    assert gp.family_repertoire([[], []])["distinct_families"] == 0


def test_kmer_distance_falls_when_a_set_collapses():
    diverse = ["MKWVTFISLLLLFSSAYS", "GQPRTEEDNIQKVLDTVA", "ACDEFGHIKLMNPQRSTV"]
    collapsed = ["MKWVTFISLLLLFSSAYS", "MKWVTFISLLLLFSSAYS", "MKWVTFISLLLLFSSAYT"]
    assert gp.mean_pairwise_kmer_distance(collapsed) < gp.mean_pairwise_kmer_distance(diverse)
    assert gp.mean_pairwise_kmer_distance(["AAAAAA", "AAAAAA"]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        gp.mean_pairwise_kmer_distance(["AAAAAA"])


def test_kmer_distance_stays_a_cosine_distance_on_unequal_alphabets():
    """A distance outside [0, 1] means the dot product was not a dot product.

    Sequences of different k-mer richness take the asymmetric branch of the
    pairwise loop, which is where iterating the wrong mapping silently returns a
    squared norm and drives the cosine above one.
    """

    rich = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    poor = "AAAAAAAAAA"
    for pair in ([rich, poor], [poor, rich]):
        value = gp.mean_pairwise_kmer_distance(pair)
        assert 0.0 <= value <= 1.0
    # No 3-mer is shared, so the vectors are orthogonal and the distance is one.
    assert gp.mean_pairwise_kmer_distance([rich, poor]) == pytest.approx(1.0)
    assert gp.mean_pairwise_kmer_distance([rich, poor]) == pytest.approx(
        gp.mean_pairwise_kmer_distance([poor, rich])
    )


# ------------------------------------------------------------- the contrasts


def test_matched_contrast_recovers_a_known_offset():
    rng = np.random.default_rng(0)
    natural_values = rng.normal(90.0, 5.0, 200)
    generated_values = natural_values - 12.0 + rng.normal(0.0, 1.0, 200)
    result = gp.matched_contrast(
        generated_values, natural_values, [f"p{i}" for i in range(200)], seed=1, n_bootstrap=2000
    )
    assert result["resolved"] is True
    assert result["difference"] == pytest.approx(-12.0, abs=0.5)
    low, high = result["difference_ci95"]
    assert low < -12.0 < high
    assert result["excludes_zero"] is True


def test_a_contrast_below_the_unit_floor_returns_no_interval():
    """Three units must not come back with a narrow interval beside hundreds.

    A percentile interval over a handful of atoms can be narrower than one over
    hundreds, so the stratum is marked unresolved instead of being given a number
    a reader would compare.
    """

    result = gp.matched_contrast([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], ["a", "b", "c"], seed=1)
    assert result["resolved"] is False
    assert result["difference_ci95"] is None
    assert result["unit_floor"]["degenerate"] is True
    assert result["difference"] == pytest.approx(2.0)


def test_matched_contrast_refuses_a_non_finite_evaluator_value():
    with pytest.raises(ValueError):
        gp.matched_contrast([1.0, float("nan")], [0.0, 0.0], ["a", "b"], seed=1)


def test_stratum_gap_contrast_recovers_the_truncation_share():
    rng = np.random.default_rng(3)
    n = 150
    natural_native = rng.normal(90.0, 4.0, n)
    natural_censored = rng.normal(90.0, 4.0, n)
    # Native products sit 5 below their comparator, censored fragments 25 below.
    generated = np.concatenate(
        [natural_native - 5.0 + rng.normal(0, 1, n), natural_censored - 25.0 + rng.normal(0, 1, n)]
    )
    natural_values = np.concatenate([natural_native, natural_censored])
    strata = ["native"] * n + ["censored"] * n
    result = gp.stratum_gap_contrast(
        generated,
        natural_values,
        strata,
        [f"p{i}" for i in range(2 * n)],
        seed=5,
        n_bootstrap=2000,
    )
    assert result["resolved"] is True
    assert result["gap_native_minus_gap_censored"] == pytest.approx(20.0, abs=1.5)
    assert result["excludes_zero"] is True


def test_stratum_gap_contrast_is_unresolved_with_only_one_stratum():
    result = gp.stratum_gap_contrast(
        [1.0] * 20, [0.0] * 20, ["native"] * 20, [f"p{i}" for i in range(20)], seed=1
    )
    assert result["resolved"] is False
    assert "one stratum" in result["reason"]


def test_stratum_gap_contrast_refuses_an_unknown_stratum_label():
    with pytest.raises(ValueError):
        gp.stratum_gap_contrast(
            [1.0] * 20, [0.0] * 20, ["made_up"] * 20, [f"p{i}" for i in range(20)], seed=1
        )


# -------------------------------------------------------------- the selection


def test_every_method_selects_the_same_number_at_a_fraction():
    assert gp.selection_count(100, 0.05) == 5
    assert gp.selection_count(100, 1.0) == 100
    # A fraction that rounds below one still selects one, so no method is
    # silently compared against an empty set.
    assert gp.selection_count(10, 0.02) == 1
    with pytest.raises(ValueError):
        gp.selection_count(10, 0.0)
    with pytest.raises(ValueError):
        gp.selection_count(0, 0.5)


def test_selection_yield_takes_the_lowest_scores_and_is_oriented_once():
    evaluator = [0.0, 1.0, 2.0, 3.0]
    score = [3.0, 2.0, 1.0, 0.0]
    assert gp.selection_yield(evaluator, score, fraction=0.5) == pytest.approx(2.5)
    assert gp.selection_yield(evaluator, score, fraction=1.0) == pytest.approx(1.5)


def test_rank_average_combines_two_selectors_without_an_undeclared_exchange_rate():
    combined = gp.rank_average([0.0, 1.0, 2.0], [2.0, 1.0, 0.0])
    assert combined.tolist() == [1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        gp.rank_average([0.0, 1.0])


def test_an_informative_selector_beats_random_and_an_uninformative_one_does_not():
    rng = np.random.default_rng(11)
    n = 400
    evaluator = rng.normal(0.0, 1.0, n)
    informative = -evaluator  # lower score, higher evaluator value
    uninformative = rng.normal(0.0, 1.0, n)
    units = [f"u{i}" for i in range(n)]
    good = gp.selection_contrast(
        evaluator, informative, units, fraction=0.1, seed=2, n_bootstrap=2000
    )
    bad = gp.selection_contrast(
        evaluator, uninformative, units, fraction=0.1, seed=2, n_bootstrap=2000
    )
    assert good["n_selected"] == bad["n_selected"] == 40
    assert good["difference_vs_random"] > 0.0
    assert good["excludes_zero"] is True
    assert bad["excludes_zero"] is False


def test_selection_contrast_is_unresolved_below_the_unit_floor():
    result = gp.selection_contrast(
        [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], ["a", "a", "b", "b"], fraction=0.5, seed=1
    )
    assert result["resolved"] is False
    assert result["difference_vs_random_ci95"] is None


def test_random_baseline_is_a_distribution_centred_on_the_pool_mean():
    rng = np.random.default_rng(13)
    evaluator = rng.normal(5.0, 2.0, 500)
    baseline = gp.random_baseline(evaluator, fraction=0.2, seed=4)
    assert baseline["n_keys"] == gp.RANDOM_BASELINE_KEYS
    assert baseline["pool_mean"] == pytest.approx(float(evaluator.mean()))
    low, high = baseline["interval"]
    assert low < baseline["pool_mean"] < high


# ------------------------------------------------------------- the guarantees


def test_output_directory_policy_tolerates_the_queue_and_refuses_prior_work(tmp_path):
    """The queue creates the output directory before it launches the cell.

    Refusing on mere existence loses every cell of a slot. What must be refused
    is evidence of work: a completion record, or any other content.
    """

    fresh = tmp_path / "fresh"
    gp.require_fresh_out(fresh, "done.json")
    assert fresh.is_dir()

    empty = tmp_path / "empty"
    empty.mkdir()
    gp.require_fresh_out(empty, "done.json")

    completed = tmp_path / "completed"
    completed.mkdir()
    (completed / "done.json").write_text("{}")
    with pytest.raises(SystemExit):
        gp.require_fresh_out(completed, "done.json")

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "partial.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit):
        gp.require_fresh_out(occupied, "done.json")


def test_stability_is_declared_unavailable_and_never_substituted():
    """The one substitution this experiment must not make, asserted.

    Predicted confidence is not stability. The declaration has to say so, name
    reasons, and say what would lift the limitation, and no evaluator the
    aggregator computes may be labelled a stability quantity.
    """

    assert gp.STABILITY_UNAVAILABLE["status"] == "unavailable"
    for key in (
        "no_predictor_is_staged",
        "confidence_is_not_stability",
        "ddG_predictors_do_not_apply",
        "what_would_lift_it",
    ):
        assert gp.STABILITY_UNAVAILABLE[key].strip()
    from scripts.capability.evaluation.aggregate_generated_evaluation import EVALUATORS

    assert not [name for name in EVALUATORS if "stab" in name.lower()]
    assert set(EVALUATORS) == {
        "mean_ca_plddt",
        "ptm",
        "confident_fold",
        "neg_mean_pae_angstrom",
        "complete_domain",
        "any_family",
    }


def test_declared_independence_records_the_residual_dependence_it_cannot_remove():
    record = gp.declared_independence(["likelihood"], ["mean_ca_plddt"])
    assert record["selectors"] == ["likelihood"]
    assert "not claimed" in record["residual_dependence"]
    assert json.dumps(record)  # serialisable into the artefact
