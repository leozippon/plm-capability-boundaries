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


def test_stability_enters_only_as_a_separately_validated_quantity():
    """Confidence is still never relabelled as stability.

    E14 reported stability as unavailable and that declaration stands, with its
    reasons. A free-energy evaluator now exists, but it is a different quantity
    produced by a separately validated instrument, it is restricted to a
    sub-pool, and the guard in :mod:`domain_stability` refuses any comparison
    between it and a structural confidence.
    """

    assert gp.STABILITY_UNAVAILABLE["status"] == "unavailable"
    for key in (
        "no_predictor_is_staged",
        "confidence_is_not_stability",
        "ddG_predictors_do_not_apply",
        "what_would_lift_it",
        "why_it_would_still_not_answer_this_question",
    ):
        assert gp.STABILITY_UNAVAILABLE[key].strip()

    from src.capability.evaluation import domain_stability as ds
    from scripts.capability.evaluation.aggregate_generated_evaluation import (
        EVALUATOR_QUANTITY,
        EVALUATORS,
        SUB_POOL_EVALUATORS,
    )

    # No evaluator smuggles a confidence in under a stability name, and the one
    # free-energy evaluator is declared as such and as a sub-pool quantity.
    assert not [name for name in EVALUATORS if "stab" in name.lower()]
    assert set(EVALUATORS) == {
        "mean_ca_plddt",
        "ptm",
        "confident_fold",
        "neg_mean_pae_angstrom",
        "complete_domain",
        "any_family",
        "predicted_delta_g",
    }
    assert SUB_POOL_EVALUATORS == {"predicted_delta_g": SUB_POOL_EVALUATORS["predicted_delta_g"]}
    assert ds.QUANTITIES[EVALUATOR_QUANTITY["predicted_delta_g"]]["kind"] == "folding_free_energy"
    for confidence in ("mean_ca_plddt", "ptm"):
        quantity = EVALUATOR_QUANTITY[confidence]
        assert ds.QUANTITIES[quantity]["kind"] == "structure_confidence"
        with pytest.raises(ValueError):
            ds.require_comparable("predicted_delta_g", quantity)


def test_declared_independence_records_the_residual_dependence_it_cannot_remove():
    record = gp.declared_independence(["likelihood"], ["mean_ca_plddt"])
    assert record["selectors"] == ["likelihood"]
    assert "not claimed" in record["residual_dependence"]
    assert json.dumps(record)  # serialisable into the artefact


# ------------------------------------------- selected-set profile and collapse


def test_homopolymer_runs_are_measured_not_guessed():
    assert gp.longest_homopolymer_run("ACDE") == 1
    assert gp.longest_homopolymer_run("AAABBBBBBBBCC") == 8
    assert gp.longest_homopolymer_run("G" * 40) == 40
    with pytest.raises(ValueError):
        gp.longest_homopolymer_run("")


def test_a_selected_set_profile_carries_every_axis_a_gain_must_be_checked_against():
    """Repertoire, duplication, complexity and length travel together.

    The question is whether a yield gain came from concentrating on a few
    families or on repetitive sequences, and that cannot be answered from numbers
    read out of different places.
    """

    sequences = ["MKWVTFISLLLLFSSAYSRGV", "GQPRTEEDNIQKVLDTVAKYQ", "G" * 21, "ACDEFGHIKLMNPQRSTVWYA"]
    families = [["PF1"], ["PF2"], [], ["PF1", "PF3"]]
    profile = gp.selected_set_profile(sequences, families)
    assert profile["n_sequences"] == 4
    assert profile["n_distinct_sequences"] == 4
    assert profile["duplicate_fraction"] == pytest.approx(0.0)
    assert profile["distinct_families"] == 3
    assert profile["n_with_any_family"] == 3
    assert profile["fraction_with_homopolymer_run"] == pytest.approx(0.25)
    assert profile["longest_homopolymer_run_max"] == 21
    assert profile["min_composition_entropy_nats"] == pytest.approx(0.0)
    assert profile["mean_length"] == pytest.approx(21.0)
    assert 0.0 <= profile["mean_pairwise_kmer_distance"] <= 1.0

    repeated = gp.selected_set_profile(["G" * 21, "G" * 21], [[], []])
    assert repeated["duplicate_fraction"] == pytest.approx(0.5)
    assert repeated["n_distinct_sequences"] == 1
    with pytest.raises(ValueError):
        gp.selected_set_profile([], [])
    with pytest.raises(ValueError):
        gp.selected_set_profile(["AAA"], [])


def test_the_diversity_reference_is_size_matched_not_pool_matched():
    """Every repertoire measure falls as a set shrinks.

    Comparing a small selected set against the whole pool would report a collapse
    at every small fraction, so the reference is a random draw of the same size.
    """

    rng = np.random.default_rng(5)
    sequences = [
        "".join(rng.choice(list(gp.AA20), size=60)) for _ in range(120)
    ]
    families = [[f"PF{index % 30}"] for index in range(120)]
    small = gp.diversity_reference(sequences, families, fraction=0.1, seed=1, n_keys=8)
    large = gp.diversity_reference(sequences, families, fraction=1.0, seed=1, n_keys=8)
    assert small["distinct_families"]["mean"] < large["distinct_families"]["mean"]
    assert small["fraction"] == 0.1 and small["n_keys"] == 8
    low, high = small["distinct_families"]["interval"]
    assert low <= small["distinct_families"]["mean"] <= high


def test_a_yield_gain_bought_with_a_collapse_is_reported_as_not_a_gain():
    """The verdict the user asked for hardest, as a computed field."""

    rng = np.random.default_rng(7)
    sequences = ["".join(rng.choice(list(gp.AA20), size=60)) for _ in range(120)]
    families = [[f"PF{index % 30}"] for index in range(120)]
    reference = gp.diversity_reference(sequences, families, fraction=0.1, seed=1, n_keys=8)
    collapsed = gp.selected_set_profile(["G" * 60, "G" * 59 + "A"], [[], []])

    verdict = gp.collapse_check(collapsed, reference, yield_difference=0.25)
    assert verdict["collapsed"] is True
    assert verdict["gain_is_not_a_gain"] is True
    assert "narrower product" in verdict["verdict"]
    assert set(verdict["axes_flagged"]) <= set(gp.COLLAPSE_AXES)

    # A collapse without a gain is not relabelled as a gain, and a gain without a
    # collapse is left standing.
    assert gp.collapse_check(collapsed, reference, yield_difference=-0.1)["gain_is_not_a_gain"] is False
    healthy = gp.selected_set_profile(sequences[:12], families[:12])
    standing = gp.collapse_check(healthy, reference, yield_difference=0.25)
    assert standing["yield_gain"] is True
    assert standing["gain_is_not_a_gain"] is False


def test_the_collapse_axes_cover_both_failure_shapes_the_user_named():
    """Concentrating on few families, and concentrating on repetitive sequences."""

    assert gp.COLLAPSE_AXES["effective_families"] == "below"
    assert gp.COLLAPSE_AXES["fraction_with_homopolymer_run"] == "above"
    assert gp.COLLAPSE_AXES["mean_composition_entropy_nats"] == "below"
    assert gp.COLLAPSE_AXES["mean_pairwise_kmer_distance"] == "below"


def test_the_length_control_and_sub_pool_declarations_are_present():
    """A length selector is required, because every structural score rises with length."""

    from scripts.capability.evaluation.aggregate_generated_evaluation import (
        EVALUATOR_QUANTITY,
        EVALUATORS,
        RENDERING_RISK,
        SELECTOR_DECLARATION,
        SUB_POOL_EVALUATORS,
    )

    assert set(SELECTOR_DECLARATION) == {
        "random",
        "likelihood",
        "likelihood_per_residue",
        "composition",
        "length",
        "combined",
    }
    assert "length" in SELECTOR_DECLARATION["length"]
    # The convention check has to say which way round the declared selector runs,
    # because an inverted likelihood is a silent anti-selector.
    assert "lower is taken first" in SELECTOR_DECLARATION["likelihood"]
    assert set(EVALUATORS) <= set(EVALUATOR_QUANTITY)
    # Predicted free energy is a sub-pool evaluator and must say so, because its
    # licensed band would otherwise select the pool silently.
    assert "predicted_delta_g" in SUB_POOL_EVALUATORS
    assert "licensed" in SUB_POOL_EVALUATORS["predicted_delta_g"]
    assert EVALUATOR_QUANTITY["predicted_delta_g"] == "predicted_delta_g"
    assert EVALUATOR_QUANTITY["mean_ca_plddt"] == "esmfold2_mean_ca_plddt"
    # The two prior rendering measurements that bound how much a layout mistake
    # could have moved the likelihood selector.
    assert RENDERING_RISK["protgpt2_unwrapped_penalty_nats_per_token"] == 1.42
    assert RENDERING_RISK["zymctrl_tag_leak_nats"] == 1.73


# ------------------------------------------------- length-conditional contrast


def test_length_bins_are_equal_count_not_equal_width():
    lengths = [50] * 40 + [51, 52, 300, 310]
    binning = gp.length_bins(lengths, n_bins=4)
    assert binning["n_bins_requested"] == 4
    assert binning["bin_of_row"].shape == (44,)
    assert sum(binning["bin_counts"].values()) == 44
    with pytest.raises(ValueError):
        gp.length_bins([60, 61], n_bins=8)


def _length_confounded_pool(n: int = 600, seed: int = 0):
    """A pool whose evaluator is length-driven, plus three selectors.

    ``proxy`` is the realistic case: a noisy length proxy, like a model
    likelihood whose rank correlation with length is around a half rather than
    one. ``signal`` carries information orthogonal to length. The pure length
    vector is the degenerate case.
    """

    rng = np.random.default_rng(seed)
    lengths = rng.integers(50, 320, n)
    signal = rng.normal(0.0, 1.0, n)
    units = [f"u{index}" for index in range(n)]
    length_only = lengths / 100.0 + rng.normal(0.0, 0.05, n)
    with_signal = lengths / 100.0 + signal + rng.normal(0.0, 0.05, n)
    proxy = -(lengths.astype(float) / 100.0 + rng.normal(0.0, 1.6, n))
    return {
        "lengths": lengths,
        "signal": signal,
        "units": units,
        "length_only": length_only,
        "with_signal": with_signal,
        "proxy": proxy,
    }


def test_a_noisy_length_proxy_loses_its_whole_gain_to_a_length_matched_draw():
    """The contrast that separates protein knowledge from a length proxy.

    This is the realistic shape of the confound: a selector correlated with
    length but not identical to it beats an unrestricted random draw on any
    length-sensitive evaluator while knowing nothing about proteins. Matching the
    comparator on length composition has to remove that.
    """

    pool = _length_confounded_pool()
    unconditional = gp.selection_contrast(
        pool["length_only"], pool["proxy"], pool["units"], fraction=0.1, seed=1, n_bootstrap=2000
    )
    conditional = gp.length_conditional_contrast(
        pool["length_only"],
        pool["proxy"],
        pool["lengths"],
        pool["units"],
        fraction=0.1,
        seed=1,
        n_bootstrap=2000,
    )
    # Large and resolved against plain random ...
    assert unconditional["difference_vs_random"] > 0.25
    assert unconditional["excludes_zero"] is True
    # ... and nothing once length composition is held fixed.
    assert conditional["resolved"] is True
    assert conditional["beats_length"] is False
    low, high = conditional["difference_ci95"]
    assert low <= 0.0 <= high
    assert abs(conditional["gain_over_length_matched_random"]) < 0.1
    # The reported point averages the comparator over many draws, because one
    # matched draw at a tight budget is noisier than the effects measured here.
    assert conditional["matched_draw_repeats"] == gp.MATCHED_DRAW_REPEATS
    assert "single_matched_draw_point" in conditional


def test_information_beyond_length_does_survive_the_matched_draw():
    pool = _length_confounded_pool()
    conditional = gp.length_conditional_contrast(
        pool["with_signal"],
        -pool["signal"],
        pool["lengths"],
        pool["units"],
        fraction=0.1,
        seed=1,
        n_bootstrap=2000,
    )
    assert conditional["resolved"] is True
    assert conditional["beats_length"] is True
    assert conditional["difference_ci95"][0] > 0.0
    # The unrestricted random key run through the same metric is the null's scale.
    assert abs(conditional["random_selector_control"]) < 0.5


def test_a_pure_length_selector_has_no_length_matched_comparator_at_all():
    """Selecting the longest sequences *is* selecting a length stratum.

    There is no random set of the same size and the same length composition other
    than that stratum itself, so the honest verdict is that the contrast cannot be
    formed -- not that the selector fails it.
    """

    pool = _length_confounded_pool()
    conditional = gp.length_conditional_contrast(
        pool["length_only"],
        -pool["lengths"].astype(float),
        pool["lengths"],
        pool["units"],
        fraction=0.1,
        seed=1,
        n_bootstrap=2000,
    )
    assert conditional["resolved"] is False
    assert conditional["forced_match_share"] == pytest.approx(1.0)
    assert conditional["difference_ci95"] is None
    assert "unselected rows" in conditional["reason"]


def test_the_contrast_refuses_where_length_matching_has_no_room():
    """At a large fraction each bin holds about as many selected as unselected rows.

    The matched draw then has to redraw the selected rows and the contrast
    collapses toward zero. Reporting that as "does not beat length" would
    discredit a real effect for an arithmetic reason, so it is refused instead.
    """

    pool = _length_confounded_pool()
    forced = gp.length_conditional_contrast(
        pool["with_signal"],
        -pool["signal"],
        pool["lengths"],
        pool["units"],
        fraction=0.5,
        seed=1,
        n_bootstrap=2000,
    )
    assert forced["resolved"] is False
    assert forced["forced_match_share"] > gp.MAX_FORCED_MATCH_SHARE
    assert forced["difference_ci95"] is None

    whole = gp.length_conditional_contrast(
        pool["with_signal"],
        -pool["signal"],
        pool["lengths"],
        pool["units"],
        fraction=1.0,
        seed=1,
        n_bootstrap=2000,
    )
    assert whole["resolved"] is False
    assert "whole pool" in whole["reason"]


def test_the_contrast_refuses_misaligned_or_non_finite_input():
    pool = _length_confounded_pool(n=40)
    with pytest.raises(ValueError):
        gp.length_conditional_contrast(
            pool["with_signal"][:20],
            -pool["signal"],
            pool["lengths"],
            pool["units"],
            fraction=0.1,
            seed=1,
        )
    broken = pool["with_signal"].copy()
    broken[0] = np.nan
    with pytest.raises(ValueError):
        gp.length_conditional_contrast(
            broken, -pool["signal"], pool["lengths"], pool["units"], fraction=0.1, seed=1
        )


# --------------------------------------------------------------- novelty axis


def test_corpus_identity_enters_the_profile_and_the_collapse_axes():
    """Drifting toward the corpus is the other way a gain fails to be a gain."""

    assert gp.COLLAPSE_AXES["nearest_corpus_identity"] == "above"
    sequences = ["MKWVTFISLLLLFSSAYSRGV", "GQPRTEEDNIQKVLDTVAKYQ", "ACDEFGHIKLMNPQRSTVWYA"]
    families = [["PF1"], ["PF2"], ["PF3"]]
    bare = gp.selected_set_profile(sequences, families)
    assert bare["nearest_corpus_identity"] is None
    assert "no homology search" in bare["nearest_corpus_identity_note"]

    scored = gp.selected_set_profile(
        sequences, families, corpus_identity=[12.0, 40.0, 99.0]
    )
    assert scored["nearest_corpus_identity"] == pytest.approx(50.3333, abs=1e-3)
    assert scored["max_nearest_corpus_identity"] == pytest.approx(99.0)
    assert scored["fraction_near_duplicate_of_corpus"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        gp.selected_set_profile(sequences, families, corpus_identity=[1.0])


def test_a_selection_drifting_toward_the_corpus_is_flagged():
    rng = np.random.default_rng(11)
    sequences = ["".join(rng.choice(list(gp.AA20), size=60)) for _ in range(120)]
    families = [[f"PF{index % 30}"] for index in range(120)]
    identities = list(rng.uniform(5.0, 30.0, 120))
    reference = gp.diversity_reference(
        sequences, families, fraction=0.1, seed=1, n_keys=8, corpus_identity=identities
    )
    assert reference["nearest_corpus_identity"] is not None
    retrieved = gp.selected_set_profile(
        sequences[:12], families[:12], corpus_identity=[97.0] * 12
    )
    verdict = gp.collapse_check(retrieved, reference, yield_difference=0.2)
    assert "nearest_corpus_identity" in verdict["axes_flagged"]
    assert verdict["axes_flagged"]["nearest_corpus_identity"]["moved"] == "above"
    assert verdict["gain_is_not_a_gain"] is True


def test_the_novelty_stage_declares_the_masking_rationale_where_it_reports():
    from scripts.capability.evaluation.search_generated_novelty import MASKING_RATIONALE

    assert "--masking 0" in MASKING_RATIONALE
    assert "82.9" in MASKING_RATIONALE
    assert "flattering" in MASKING_RATIONALE


def test_concentrating_onto_near_duplicates_is_flagged_even_at_a_flat_mean():
    """The corpus-identity collapse has two shapes and both have to be caught.

    A set can drift toward the corpus by picking up remote homologues, which the
    mean identity catches, or by concentrating onto the handful of verbatim
    corpus members while most of its mass stays novel, which the mean does not.
    """

    assert gp.COLLAPSE_AXES["fraction_near_duplicate_of_corpus"] == "above"
    rng = np.random.default_rng(23)
    sequences = ["".join(rng.choice(list(gp.AA20), size=60)) for _ in range(200)]
    families = [[f"PF{index % 40}"] for index in range(200)]
    # A pool that is almost all novel with a few verbatim corpus members.
    identities = [99.0] * 10 + [8.0] * 190
    reference = gp.diversity_reference(
        sequences, families, fraction=0.05, seed=3, n_keys=16, corpus_identity=identities
    )
    assert reference["fraction_near_duplicate_of_corpus"] is not None

    # Ten selected rows, all of them the near-duplicates. The mean identity of a
    # size-matched random draw is dominated by the novel majority, so this is the
    # case the near-duplicate share exists to catch.
    concentrated = gp.selected_set_profile(
        sequences[:10], families[:10], corpus_identity=identities[:10]
    )
    verdict = gp.collapse_check(concentrated, reference, yield_difference=0.3)
    assert "fraction_near_duplicate_of_corpus" in verdict["axes_flagged"]
    assert verdict["axes_flagged"]["fraction_near_duplicate_of_corpus"]["moved"] == "above"
    assert verdict["gain_is_not_a_gain"] is True

    # A draw that is not enriched in near-duplicates is left standing on that axis.
    novel = gp.selected_set_profile(
        sequences[10:20], families[10:20], corpus_identity=identities[10:20]
    )
    standing = gp.collapse_check(novel, reference, yield_difference=0.3)
    assert "fraction_near_duplicate_of_corpus" not in standing["axes_flagged"]


# ------------------------------------------------------- the gap decomposition


def decomposition_pool(n=240, noise=0.0, seed=19):
    """A pool with a known evaluator, a known-skill selector and length bins."""

    rng = np.random.default_rng(seed)
    quality = rng.normal(size=n)
    lengths = rng.integers(60, 320, size=n).tolist()
    return {
        "quality": quality,
        "lengths": lengths,
        "units": [f"u{index}" for index in range(n)],
        "perfect": -quality,
        "noisy": -quality + rng.normal(scale=noise, size=n) if noise else -quality,
        "blind": rng.normal(size=n),
    }


def test_a_perfect_selector_reaches_the_whole_attainable_gap_and_a_blind_one_none():
    pool = decomposition_pool()
    record = gp.gap_decomposition(
        pool["quality"],
        {"perfect": pool["perfect"], "blind": pool["blind"]},
        pool["units"],
        fractions=(0.05, 0.25),
        seed=5,
        n_bootstrap=400,
    )
    assert record["resolved"] is True
    assert record["n_pool"] == 240 and record["n_units"] == 240
    for point in record["points"]:
        assert point["attainable_gap"] > 0.0
        assert point["pool_holds_selectable_quality"] is True
        # The oracle is a ceiling: no selector may exceed it.
        for block in point["selectors"].values():
            assert block["selected_yield"] <= point["oracle_yield"] + 1e-9
            assert block["shortfall_vs_oracle"] >= -1e-9
        perfect = point["selectors"]["perfect"]["share_of_attainable_gap"]
        blind = point["selectors"]["blind"]["share_of_attainable_gap"]
        assert perfect["value"] == pytest.approx(1.0)
        assert perfect["limiting_factor"]["code"] == "pool_content"
        assert blind["value"] < 0.5
        assert blind["limiting_factor"]["code"] == "selector_skill"


def test_a_pool_holding_nothing_selectable_is_not_read_as_a_selector_failure():
    """A constant evaluator has no attainable gap, so no share is a quantity."""

    record = gp.gap_decomposition(
        np.ones(64),
        {"anything": np.arange(64, dtype=np.float64)},
        [f"u{index}" for index in range(64)],
        fractions=(0.1,),
        seed=5,
        n_bootstrap=200,
    )
    point = record["points"][0]
    assert point["attainable_gap"] == pytest.approx(0.0)
    assert point["pool_holds_selectable_quality"] is False
    share = point["selectors"]["anything"]["share_of_attainable_gap"]
    assert share["resolved"] is False and share["value"] is None
    assert "for a share to be a quantity" in share["reason"]


def test_full_selection_is_an_identity_point_with_nothing_to_decompose():
    pool = decomposition_pool(n=64)
    record = gp.gap_decomposition(
        pool["quality"],
        {"perfect": pool["perfect"]},
        pool["units"],
        fractions=(1.0,),
        seed=5,
        n_bootstrap=200,
        lengths=pool["lengths"],
    )
    point = record["points"][0]
    assert point["identity_point"] is True
    assert point["n_selected"] == 64
    assert point["oracle_yield"] == pytest.approx(point["random_yield"])
    assert point["selectors"]["perfect"]["share_of_attainable_gap"]["resolved"] is False
    assert point["length_matched"]["resolved"] is False
    assert "whole pool" in point["length_matched"]["reason"]


def test_an_unresolved_share_reports_the_pool_size_that_would_resolve_it():
    """The answer the user asked for when the pool cannot separate the two causes."""

    rng = np.random.default_rng(31)
    quality = rng.normal(size=40)
    weak = -quality + rng.normal(scale=1.5, size=40)
    record = gp.gap_decomposition(
        quality,
        {"weak": weak},
        [f"u{index}" for index in range(40)],
        fractions=(0.25,),
        seed=5,
        n_bootstrap=600,
    )
    share = record["points"][0]["selectors"]["weak"]["share_of_attainable_gap"]
    assert share["resolved"] is True
    low, high = share["ci95"]
    assert low < high
    assert share["half_width"] == pytest.approx((high - low) / 2)
    assert share["reaches_target_half_width"] is False
    # Wider than the target, so more units are needed than the pool has, and the
    # scaling that produced the number is declared rather than implied.
    assert share["units_for_target_half_width"] > record["n_units"]
    assert "square root" in share["units_scaling_assumption"]


def test_a_length_only_ceiling_is_reported_as_one():
    """A pool whose quality is length must have no matched ceiling left over."""

    rng = np.random.default_rng(41)
    lengths = rng.integers(60, 320, size=240)
    quality = lengths.astype(np.float64) / 100.0
    record = gp.gap_decomposition(
        quality,
        {"by_length": -lengths.astype(np.float64), "blind": rng.normal(size=240)},
        [f"u{index}" for index in range(240)],
        fractions=(0.25,),
        seed=5,
        n_bootstrap=400,
        lengths=lengths.tolist(),
    )
    point = record["points"][0]
    # Unmatched, the pool looks full of selectable quality.
    assert point["attainable_gap"] > 0.0
    matched = point["length_matched"]
    # The oracle here *is* the length selector, so length matching has no room
    # and the block refuses a ceiling rather than reporting a zero one.
    assert matched["resolved"] is False
    assert matched["oracle_forced_match_share"] > gp.MAX_FORCED_MATCH_SHARE


def test_the_matched_comparator_point_is_averaged_not_one_draw():
    """One matched draw of a dozen rows is noisier than the effects measured here.

    Two independent single draws of the comparator disagree by more than the
    averaged estimate moves, which is why the reported point averages.
    """

    rng = np.random.default_rng(53)
    values = rng.normal(size=120)
    bins = np.repeat(np.arange(8), 15)
    chosen = np.arange(12)
    singles = [
        gp.length_matched_mean(values, bins, chosen, np.random.default_rng(seed))
        for seed in (1, 2, 3, 4)
    ]
    averaged = [
        gp.length_matched_mean(
            values, bins, chosen, np.random.default_rng(seed), repeats=gp.MATCHED_DRAW_REPEATS
        )
        for seed in (1, 2, 3, 4)
    ]
    assert np.ptp(averaged) < np.ptp(singles)
    with pytest.raises(ValueError):
        gp.length_matched_mean(values, bins, chosen, np.random.default_rng(1), repeats=0)


def test_a_real_matched_ceiling_separates_length_from_protein_quality():
    rng = np.random.default_rng(47)
    lengths = rng.integers(60, 320, size=240)
    # Quality rises with length *and* carries a length-independent component.
    beyond_length = rng.normal(size=240)
    quality = lengths.astype(np.float64) / 200.0 + beyond_length
    record = gp.gap_decomposition(
        quality,
        {
            "beyond_length": -beyond_length,
            "by_length": -lengths.astype(np.float64),
        },
        [f"u{index}" for index in range(240)],
        fractions=(0.1,),
        seed=5,
        n_bootstrap=400,
        lengths=lengths.tolist(),
    )
    matched = record["points"][0]["length_matched"]
    assert matched["resolved"] is True
    assert matched["matched_ceiling"] > 0.0
    low, high = matched["matched_ceiling_ci95"]
    assert low <= high
    beyond = matched["selectors"]["beyond_length"]
    assert beyond["resolved"] is True
    assert beyond["matched_gain"] > 0.0
    assert beyond["share_of_matched_ceiling"]["value"] > 0.0
    # The pure length selector has no matched comparator at all, by construction.
    assert matched["selectors"]["by_length"]["resolved"] is False
    assert matched["selectors"]["by_length"]["forced_match_share"] > gp.MAX_FORCED_MATCH_SHARE


def test_the_decomposition_refuses_misaligned_non_finite_or_sub_floor_input():
    pool = decomposition_pool(n=64)
    with pytest.raises(ValueError):
        gp.gap_decomposition(
            pool["quality"], {"short": np.ones(3)}, pool["units"], seed=1, n_bootstrap=100
        )
    broken = pool["quality"].copy()
    broken[0] = np.nan
    with pytest.raises(ValueError):
        gp.gap_decomposition(
            broken, {"perfect": pool["perfect"]}, pool["units"], seed=1, n_bootstrap=100
        )
    with pytest.raises(ValueError):
        gp.gap_decomposition(
            pool["quality"], {}, pool["units"], seed=1, n_bootstrap=100
        )
    # Below the package's unit floor nothing is estimated and the reason is named.
    tiny = gp.gap_decomposition(
        pool["quality"],
        {"perfect": pool["perfect"]},
        ["only_one_unit"] * 64,
        seed=1,
        n_bootstrap=100,
    )
    assert tiny["resolved"] is False and tiny["points"] == []
    assert tiny["unit_floor"]["degenerate"] is True


def test_the_oracle_key_is_declared_as_unusable_and_the_question_is_stated():
    from scripts.capability.evaluation.aggregate_generated_evaluation import (
        GAP_DECOMPOSITION_QUESTION,
    )

    assert "not a method" in GAP_DECOMPOSITION_QUESTION["what_the_oracle_is_not"]
    assert "too few promising candidates" in GAP_DECOMPOSITION_QUESTION["question"]
    assert "length ceiling" in GAP_DECOMPOSITION_QUESTION["length"]
    assert "independence units" in GAP_DECOMPOSITION_QUESTION["when_it_cannot_be_answered"]
    assert gp.GAP_SHARE_SPLIT == 0.5
    # The oracle key is the evaluator with the module's one orientation applied.
    assert np.allclose(gp.oracle_key([1.0, 3.0, 2.0]), [-1.0, -3.0, -2.0])
    # Oriented the one way the module orients everything: the lowest key first,
    # which on a negated evaluator is the best member.
    assert gp.selection_yield([1.0, 3.0, 2.0], gp.oracle_key([1.0, 3.0, 2.0]), fraction=0.2) == 3.0
    assert gp.selection_yield([1.0, 3.0, 2.0], gp.oracle_key([1.0, 3.0, 2.0]), fraction=0.5) == 2.5
