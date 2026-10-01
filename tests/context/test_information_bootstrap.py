"""What the information bootstrap must always do, including the parts that fail quietly.

The width comparisons are the load-bearing tests here. A bootstrap that
resamples the wrong thing still returns a plausible interval with a plausible
point estimate, so the only way to catch it is to assert the *ordering* of
widths against a deliberately broken variant of the same arithmetic:

* sharing one cohort draw between the two terms of ``I`` must be narrower than
  drawing the cohort twice, because the terms are paired by construction;
* resampling the reference set must be wider than holding it fixed, because the
  baseline is fitted on a finite corpus;
* a cross-arm contrast under common indices must be narrower than independent
  per-arm bootstrapping implies, because arms on one cohort are positively
  correlated.

Each is driven through the same production arithmetic with only the resampling
topology changed, so a passing test is evidence about the implementation rather
than about a second implementation written in the test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.capability.context.information_bootstrap import (
    FIELLER_MAXIMUM_G,
    ArmStatistics,
    CohortStatistics,
    ReferenceStatistics,
    SparseCounts,
    _group_multiplicities,
    _prepare_arm,
    _reference_displacement,
    _statistics_from_weights,
    bootstrap_arms,
    bootstrap_information,
    mixed_unigram_arm,
    ratio_interval,
    unigram_null_control,
    unpaired_contrast,
)
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS

DRAWS = 1000


# --------------------------------------------------------------------------- #
# Synthetic cohorts with a known information content
# --------------------------------------------------------------------------- #


def _sparse_from_dense(count_rows: list[np.ndarray]) -> SparseCounts:
    ids = [np.flatnonzero(row) for row in count_rows]
    counts = [row[index] for row, index in zip(count_rows, ids)]
    return SparseCounts.from_records(ids, counts)


def _multinomial_records(
    rng: np.random.Generator,
    probabilities: np.ndarray,
    tokens: list[int],
) -> list[np.ndarray]:
    return [rng.multinomial(count, probabilities) for count in tokens]


def build_arm(
    *,
    name: str = "arm",
    seed: int,
    vocab_size: int = 128,
    n_groups: int = 20,
    records_per_group: int = 3,
    tokens_per_record: int = 150,
    information: float = 0.5,
    group_sd: float = 0.05,
    cohort_kappa: float = 4000.0,
    reference_groups: int = 16,
    reference_records_per_group: int = 4,
    reference_tokens_per_record: int = 1500,
    reference_kappa: float = 4000.0,
    symbols_per_token: int = 3,
    smoothing: float = 1.0,
    group_token_scale: np.ndarray | None = None,
    delta_by_group: np.ndarray | None = None,
    reference_seed: int | None = None,
) -> ArmStatistics:
    """A cohort whose context information is ``information`` nats/token by construction.

    Tokens are drawn from a unigram ``p``; each group's own composition is a
    Dirichlet perturbation of ``p`` whose concentration ``kappa`` controls how
    heterogeneous the groups are. The per-record NLL sum is the record's
    cross-entropy under ``p`` less ``token_count * delta_g``, so the model term
    sits ``delta`` below the corpus baseline and ``I`` recovers ``mean(delta)``
    up to the reference set's own estimation error.

    ``reference_seed`` draws the reference block from its own stream while the
    unigram ``p``, and therefore the population, stays the one ``seed`` fixed.
    Two arms built at one ``seed`` and two ``reference_seed`` values differ in
    nothing but which independent sample of that population their reference is,
    which is what a null control needs and what sharing one stream cannot give.
    """

    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(vocab_size, 2.0))
    surprisal = -np.log(p)

    if delta_by_group is None:
        delta_by_group = rng.normal(information, group_sd, size=n_groups)
    scale = (
        np.ones(n_groups, dtype=float) if group_token_scale is None else group_token_scale
    )

    rows: list[np.ndarray] = []
    nll_sum: list[float] = []
    token_count: list[int] = []
    n_symbols: list[int] = []
    group_id: list[int] = []
    for group in range(n_groups):
        composition = rng.dirichlet(p * cohort_kappa)
        tokens = [int(round(tokens_per_record * scale[group]))] * records_per_group
        for row in _multinomial_records(rng, composition, tokens):
            total = int(row.sum())
            rows.append(row)
            nll_sum.append(float(row @ surprisal) - total * float(delta_by_group[group]))
            token_count.append(total)
            n_symbols.append(total * symbols_per_token)
            group_id.append(group)

    reference_rng = rng if reference_seed is None else np.random.default_rng(reference_seed)
    reference_rows: list[np.ndarray] = []
    reference_tokens: list[int] = []
    reference_groups_ids: list[int] = []
    for group in range(reference_groups):
        composition = reference_rng.dirichlet(p * reference_kappa)
        rows_here = _multinomial_records(
            reference_rng,
            composition,
            [reference_tokens_per_record] * reference_records_per_group,
        )
        for row in rows_here:
            reference_rows.append(row)
            reference_tokens.append(int(row.sum()))
            reference_groups_ids.append(group)

    return ArmStatistics(
        name=name,
        cohort=CohortStatistics(
            clean_nll_sum=np.asarray(nll_sum, dtype=np.float64),
            token_count=np.asarray(token_count, dtype=np.int64),
            n_symbols=np.asarray(n_symbols, dtype=np.int64),
            targets=_sparse_from_dense(rows),
            group_id=np.asarray(group_id, dtype=np.int64),
        ),
        reference=ReferenceStatistics(
            token_count=np.asarray(reference_tokens, dtype=np.int64),
            targets=_sparse_from_dense(reference_rows),
            group_id=np.asarray(reference_groups_ids, dtype=np.int64),
        ),
        vocab_size=vocab_size,
        smoothing=smoothing,
    )


def _width(draws: np.ndarray) -> float:
    return float(np.percentile(draws, 97.5) - np.percentile(draws, 2.5))


def _information(record: dict) -> dict:
    return record["statistics"]["information_nats_per_token"]


# --------------------------------------------------------------------------- #
# 1-2. The interval says what it should about a known effect
# --------------------------------------------------------------------------- #


def test_a_clearly_positive_information_is_bounded_away_from_zero() -> None:
    arm = build_arm(seed=1, information=0.5, group_sd=0.05)
    result = bootstrap_information(arm, seed=7, n_bootstrap=DRAWS)

    assert not result.refused
    block = _information(result.record)
    assert block["interval"][0] > 0.0
    assert block["point"] == pytest.approx(0.5, abs=0.05)
    assert block["interval"][0] < block["point"] < block["interval"][1]
    assert block["fraction_of_draws_positive"] == 1.0


def test_a_near_zero_information_gives_an_interval_that_straddles_zero() -> None:
    arm = build_arm(seed=2, information=0.0, group_sd=0.03)
    result = bootstrap_information(arm, seed=7, n_bootstrap=DRAWS)

    block = _information(result.record)
    assert block["point"] == pytest.approx(0.0, abs=0.02)
    assert block["interval"][0] < 0.0 < block["interval"][1]


# --------------------------------------------------------------------------- #
# 3-4. The resampling topology
# --------------------------------------------------------------------------- #


def test_resampling_the_reference_widens_the_interval() -> None:
    """The baseline is fitted on a finite corpus, so it carries its own uncertainty."""

    arm = build_arm(
        seed=3,
        information=0.4,
        cohort_kappa=300.0,
        reference_kappa=60.0,
        reference_tokens_per_record=400,
    )
    prepared = _prepare_arm(arm)
    rng = np.random.default_rng(101)
    cohort_weights = _group_multiplicities(rng, prepared.n_cohort_groups, DRAWS)
    reference_weights = _group_multiplicities(rng, prepared.n_reference_groups, DRAWS)
    fixed_reference = np.ones((DRAWS, prepared.n_reference_groups), dtype=np.int64)

    resampled = _statistics_from_weights(
        prepared, cohort_weights, cohort_weights, reference_weights
    )
    held_fixed = _statistics_from_weights(
        prepared, cohort_weights, cohort_weights, fixed_reference
    )

    # Same cohort draw in both, so the reference is the only thing that moved.
    # The reference draw is independent of the cohort draw, so the variance
    # inequality is exact and the width inequality follows it; both are
    # asserted, because a width is a noisier read of the same fact.
    for statistic in ("baseline_entropy_nats_per_token", "information_nats_per_token"):
        assert held_fixed[statistic].std(ddof=1) < resampled[statistic].std(ddof=1)
        assert _width(held_fixed[statistic]) < _width(resampled[statistic])


def test_a_small_reference_biases_the_draws_upward_and_the_record_says_so() -> None:
    """The interval may sit above the point estimate, and must be readable as such.

    ``-log`` is convex in the resampled reference counts, so resampling a small
    reference shifts the whole baseline draw distribution upward. Nothing here
    corrects for it -- there is no BCa in this package -- so the mandatory bias
    diagnostics are the only thing standing between a reader and an interval
    they would otherwise take at face value.
    """

    arm = build_arm(
        seed=20,
        vocab_size=256,
        information=0.4,
        reference_groups=10,
        reference_records_per_group=1,
        reference_tokens_per_record=200,
    )
    small = bootstrap_information(arm, seed=3, n_bootstrap=DRAWS).record["statistics"][
        "baseline_entropy_nats_per_token"
    ]
    assert small["bootstrap_bias"] > 0.01
    assert small["median_bias_z0"] < -1.0
    assert small["interval"][0] > small["point"]

    generous = bootstrap_information(
        build_arm(seed=20, vocab_size=256, information=0.4), seed=3, n_bootstrap=DRAWS
    ).record["statistics"]["baseline_entropy_nats_per_token"]
    assert abs(generous["bootstrap_bias"]) < small["bootstrap_bias"]
    assert generous["interval"][0] < generous["point"] < generous["interval"][1]


def test_the_displacement_correction_moves_the_interval_and_nothing_else() -> None:
    """A constant shift of the draws: position moves, width and point do not.

    The failure this guards against is a "correction" that also narrows the
    interval, which would buy a lower false-positive rate at a true zero by
    understating the uncertainty instead of by removing the displacement.
    """

    arm = build_arm(
        seed=20,
        vocab_size=256,
        information=0.4,
        reference_groups=10,
        reference_records_per_group=1,
        reference_tokens_per_record=200,
    )
    block = _information(bootstrap_information(arm, seed=3, n_bootstrap=DRAWS).record)

    displacement = block["reference_resampling_displacement"]
    assert displacement > 0.01
    low, high = block["interval"]
    corrected_low, corrected_high = block["displacement_corrected_interval"]
    assert corrected_low == pytest.approx(low - displacement, rel=1e-12)
    assert corrected_high == pytest.approx(high - displacement, rel=1e-12)
    assert (corrected_high - corrected_low) == pytest.approx(high - low, rel=1e-12)

    # On this deliberately small reference the uncorrected interval sits
    # entirely above the point it surrounds, which is the L34 condition; the
    # corrected one contains it.
    assert low > block["point"]
    assert corrected_low < block["point"] < corrected_high


def test_the_displacement_is_the_baseline_term_and_is_never_negative() -> None:
    """``-log`` is convex and the model term does not see the reference at all.

    Both halves are conditions on any cohort, not observations about this one:
    a negative displacement or a displaced model term would mean the second
    evaluation is not holding the reference fixed.
    """

    for seed, reference_tokens in ((21, 200), (22, 1500), (23, 4000)):
        arm = build_arm(
            seed=seed,
            vocab_size=256,
            information=0.4,
            reference_groups=12,
            reference_records_per_group=2,
            reference_tokens_per_record=reference_tokens,
        )
        statistics = bootstrap_information(
            arm, seed=5, n_bootstrap=DRAWS
        ).record["statistics"]
        baseline = statistics["baseline_entropy_nats_per_token"][
            "reference_resampling_displacement"
        ]
        information = statistics["information_nats_per_token"][
            "reference_resampling_displacement"
        ]
        assert baseline > 0.0
        assert statistics["model_entropy_nats_per_token"][
            "reference_resampling_displacement"
        ] == 0.0
        assert information == pytest.approx(baseline, rel=1e-10)


def test_a_reference_that_cannot_be_resampled_carries_no_displacement() -> None:
    """One reference group has one resample, so the correction must be exactly nothing.

    The degenerate case is where a correction estimated from the draws would
    quietly invent a shift out of Monte-Carlo noise.
    """

    arm = build_arm(
        seed=24,
        information=0.4,
        reference_groups=1,
        reference_records_per_group=8,
        reference_tokens_per_record=800,
    )
    for name, block in bootstrap_information(
        arm, seed=6, n_bootstrap=DRAWS
    ).record["statistics"].items():
        assert block["reference_resampling_displacement"] == 0.0, name
        assert block["displacement_corrected_interval"] == block["interval"], name
        assert block["reference_held_fixed_interval"] == block["interval"], name
        assert block["cohort_only_bootstrap_se"] == pytest.approx(
            block["bootstrap_se"], rel=1e-12
        ), name


def test_holding_the_reference_fixed_is_narrower_than_correcting_it() -> None:
    """The two must stay distinguishable in the artefact, because they differ in width.

    Dropping the reference resampling removes the displacement *and* the
    reference's share of the variance; the correction removes only the first.
    A reader who could not tell them apart could not tell a repaired interval
    from a shrunken one.
    """

    arm = build_arm(
        seed=25,
        vocab_size=256,
        information=0.4,
        reference_groups=10,
        reference_records_per_group=1,
        reference_tokens_per_record=200,
    )
    block = _information(bootstrap_information(arm, seed=8, n_bootstrap=DRAWS).record)

    assert block["cohort_only_bootstrap_se"] < block["bootstrap_se"]
    fixed_low, fixed_high = block["reference_held_fixed_interval"]
    corrected_low, corrected_high = block["displacement_corrected_interval"]
    assert (fixed_high - fixed_low) < (corrected_high - corrected_low)


def test_the_displacement_refuses_draws_it_cannot_pair() -> None:
    """Two draw counts cannot be differenced draw for draw, and must not be averaged."""

    with pytest.raises(ValueError, match="one reference-held-fixed draw"):
        _reference_displacement(np.zeros(10), np.zeros(9), 0.95)


def test_sharing_the_cohort_draw_narrows_the_interval() -> None:
    """The two terms of ``I`` are averages over one token multiset, so they pair.

    Drawing the cohort twice estimates the variance of a difference between two
    independent cohorts. That is a different, much larger quantity, and the
    interval it produces looks exactly like the right one.
    """

    arm = build_arm(seed=4, information=0.4, cohort_kappa=30.0)
    prepared = _prepare_arm(arm)
    rng = np.random.default_rng(202)
    cohort_weights = _group_multiplicities(rng, prepared.n_cohort_groups, DRAWS)
    second_cohort_draw = _group_multiplicities(rng, prepared.n_cohort_groups, DRAWS)
    reference_weights = _group_multiplicities(rng, prepared.n_reference_groups, DRAWS)

    paired = _statistics_from_weights(
        prepared, cohort_weights, cohort_weights, reference_weights
    )
    unpaired = _statistics_from_weights(
        prepared, cohort_weights, second_cohort_draw, reference_weights
    )

    assert _width(paired["information_nats_per_token"]) < _width(
        unpaired["information_nats_per_token"]
    )
    # Not a marginal difference on this cohort: its groups differ enough in
    # composition that the term the two halves share dominates the variation
    # across cohort draws, so breaking the pairing inflates the interval several
    # fold rather than by a few per cent.
    assert _width(unpaired["information_nats_per_token"]) > 2.0 * _width(
        paired["information_nats_per_token"]
    )


# --------------------------------------------------------------------------- #
# 5. The sufficient statistics reproduce the direct computation
# --------------------------------------------------------------------------- #


def _dense_counts(counts: SparseCounts, vocab_size: int) -> np.ndarray:
    dense = np.zeros(vocab_size, dtype=np.float64)
    np.add.at(dense, counts.unique_token_ids, counts.counts.astype(np.float64))
    return dense


def test_aggregation_from_sufficient_statistics_matches_the_direct_computation() -> None:
    arm = build_arm(seed=5, vocab_size=256, reference_tokens_per_record=200,
                    reference_records_per_group=1, reference_groups=10)
    result = bootstrap_information(arm, seed=7, n_bootstrap=DRAWS)
    statistics = result.record["statistics"]

    cohort = _dense_counts(arm.cohort.targets, arm.vocab_size)
    reference = _dense_counts(arm.reference.targets, arm.vocab_size)
    q = (reference + arm.smoothing) / (
        reference.sum() + arm.smoothing * arm.vocab_size
    )
    direct_baseline = float(-(cohort * np.log(q)).sum() / cohort.sum())
    direct_model = float(
        arm.cohort.clean_nll_sum.sum() / arm.cohort.token_count.sum()
    )

    assert statistics["model_entropy_nats_per_token"]["point"] == pytest.approx(
        direct_model, rel=1e-12
    )
    assert statistics["baseline_entropy_nats_per_token"]["point"] == pytest.approx(
        direct_baseline, rel=1e-12
    )
    assert statistics["information_nats_per_token"]["point"] == pytest.approx(
        direct_baseline - direct_model, rel=1e-12
    )
    assert statistics["relative_information"]["point"] == pytest.approx(
        (direct_baseline - direct_model) / direct_baseline, rel=1e-12
    )
    expansion = float(arm.cohort.n_symbols.sum() / arm.cohort.token_count.sum())
    assert statistics["information_bits_per_symbol"]["point"] == pytest.approx(
        (direct_baseline - direct_model) / (math.log(2.0) * expansion), rel=1e-12
    )

    # The smoothing blind spot is reported, and reported correctly: these tokens
    # keep a zero reference count in every resample, so nothing here can move
    # their contribution to the baseline.
    diagnostics = result.record["diagnostics"]
    unseen = float(cohort[reference <= 0].sum() / cohort.sum())
    assert diagnostics["cohort_token_share_unseen_in_reference"] == pytest.approx(
        unseen, rel=1e-12
    )
    assert diagnostics["cohort_token_share_reference_count_at_most_5"] == pytest.approx(
        float(cohort[reference <= 5].sum() / cohort.sum()), rel=1e-12
    )
    assert unseen > 0.0


# --------------------------------------------------------------------------- #
# 6. Seeds
# --------------------------------------------------------------------------- #


def test_a_seed_reproduces_the_result_and_a_different_seed_does_not() -> None:
    arm = build_arm(seed=6, information=0.3)
    first = bootstrap_information(arm, seed=11, n_bootstrap=DRAWS)
    again = bootstrap_information(arm, seed=11, n_bootstrap=DRAWS)
    other = bootstrap_information(arm, seed=12, n_bootstrap=DRAWS)

    assert first.record == again.record
    assert np.array_equal(
        first.draws["information_nats_per_token"],
        again.draws["information_nats_per_token"],
    )
    assert _information(other.record)["interval"] != _information(first.record)["interval"]
    # The point estimate is a property of the data, not of the resample.
    assert _information(other.record)["point"] == _information(first.record)["point"]
    # An arm measured alone and the same arm measured inside a panel agree.
    panel = bootstrap_arms([arm], seed=11, n_bootstrap=DRAWS)
    assert panel.arms[arm.name].record == first.record


# --------------------------------------------------------------------------- #
# 7. The effective-unit floor
# --------------------------------------------------------------------------- #


def test_a_low_effective_group_count_refuses_without_falling_back() -> None:
    """One dominant group makes forty records worth barely more than one."""

    scale = np.full(12, 0.02)
    scale[0] = 40.0
    arm = build_arm(seed=7, n_groups=12, records_per_group=3, group_token_scale=scale)
    result = bootstrap_information(arm, seed=7, n_bootstrap=DRAWS)

    assert result.refused
    assert result.draws == {}
    assert result.record["statistics"] is None
    assert result.cohort_multiplicities is None
    floor = result.record["unit_floor"]
    assert floor["degenerate"]
    assert floor["n_effective_groups"] < MINIMUM_BOOTSTRAP_UNITS
    # The raw counts clear every floor that is not the effective one, which is
    # exactly the case a group-count check would wave through.
    assert floor["n_groups"] > MINIMUM_BOOTSTRAP_UNITS
    assert result.record["diagnostics"]["n_records"] > MINIMUM_BOOTSTRAP_UNITS
    assert "record" in floor["degenerate_reason"]
    with pytest.raises(ValueError, match="refused"):
        _ = result.information


def test_an_unequal_but_adequate_cohort_is_not_refused() -> None:
    scale = np.linspace(0.6, 1.6, 20)
    arm = build_arm(seed=8, group_token_scale=scale)
    result = bootstrap_information(arm, seed=7, n_bootstrap=DRAWS)

    assert not result.refused
    assert result.record["unit_floor"]["n_effective_groups"] >= MINIMUM_BOOTSTRAP_UNITS
    assert result.record["unit_floor"]["n_effective_groups"] < result.record[
        "unit_floor"
    ]["n_groups"]


# --------------------------------------------------------------------------- #
# 8. Ratios and the Fieller precondition
# --------------------------------------------------------------------------- #


def _numerator_under_the_same_draws(
    arm: ArmStatistics, result, per_group_rate: np.ndarray
) -> tuple[np.ndarray, float]:
    """A token-weighted per-group rate, recomputed under the denominator's draws."""

    prepared = _prepare_arm(arm)
    assert np.array_equal(prepared.cohort_group_labels, result.cohort_group_labels)
    mass = prepared.cohort_group_tokens * per_group_rate
    weights = result.cohort_multiplicities.astype(np.float64)
    draws = (weights @ mass) / (weights @ prepared.cohort_group_tokens)
    point = float(mass.sum() / prepared.cohort_group_tokens.sum())
    return draws, point


def test_a_comfortable_denominator_publishes_a_jointly_formed_ratio() -> None:
    arm = build_arm(seed=9, information=0.5, group_sd=0.04)
    result = bootstrap_information(arm, seed=7, n_bootstrap=DRAWS)
    rate = np.random.default_rng(3).normal(0.2, 0.02, size=20)
    numerator_draws, numerator_point = _numerator_under_the_same_draws(
        arm, result, rate
    )

    record = ratio_interval(numerator_draws, numerator_point, result)

    assert record["published"] is True
    assert record["fieller_g"] < FIELLER_MAXIMUM_G
    assert record["ratio"]["point"] == pytest.approx(
        numerator_point / _information(result.record)["point"], rel=1e-12
    )
    assert record["ratio"]["interval"][0] < record["ratio"]["point"]
    assert record["ratio"]["interval"][1] > record["ratio"]["point"]
    assert record["n_draws_non_positive_denominator"] == 0
    # The jointly formed interval is not the interval a reader would get by
    # dividing the two published intervals endpoint by endpoint.
    naive = numerator_point / _information(result.record)["interval"][1]
    assert record["ratio"]["interval"][0] != pytest.approx(naive, rel=1e-6)


def test_a_denominator_at_zero_refuses_the_ratio_and_keeps_both_terms() -> None:
    arm = build_arm(seed=10, information=0.0, group_sd=0.03)
    result = bootstrap_information(arm, seed=7, n_bootstrap=DRAWS)
    rate = np.random.default_rng(4).normal(0.2, 0.02, size=20)
    numerator_draws, numerator_point = _numerator_under_the_same_draws(
        arm, result, rate
    )

    record = ratio_interval(numerator_draws, numerator_point, result)

    assert record["published"] is False
    assert record["refusal_reason"] == "denominator not identified away from zero"
    assert record["ratio"] is None
    assert record["fieller_g"] > FIELLER_MAXIMUM_G
    # The measurements that were made are still reported.
    assert record["numerator"]["point"] == pytest.approx(numerator_point)
    assert record["denominator"]["point"] == _information(result.record)["point"]
    # The non-positive-denominator count is a diagnostic beside the gate, not
    # the gate: it can sit at zero while the ratio is still unpublishable.
    assert isinstance(record["n_draws_non_positive_denominator"], int)


def test_a_ratio_against_a_refused_denominator_is_refused() -> None:
    scale = np.full(12, 0.02)
    scale[0] = 40.0
    arm = build_arm(seed=11, n_groups=12, group_token_scale=scale)
    result = bootstrap_information(arm, seed=7, n_bootstrap=DRAWS)

    record = ratio_interval(np.zeros(DRAWS), 0.0, result)
    assert record["published"] is False
    assert "refused" in record["refusal_reason"]


def test_a_numerator_that_does_not_align_with_the_draws_is_refused() -> None:
    arm = build_arm(seed=12, information=0.5)
    result = bootstrap_information(arm, seed=7, n_bootstrap=DRAWS)
    with pytest.raises(ValueError, match="one value per bootstrap draw"):
        ratio_interval(np.zeros(DRAWS - 1), 0.1, result)


# --------------------------------------------------------------------------- #
# 9. Cross-arm contrasts
# --------------------------------------------------------------------------- #


def _paired_arms() -> list[ArmStatistics]:
    """Two arms on one cohort, differing by 0.010 nats/token.

    Both are built from the same seed, so they score the same records with the
    same grouping and differ only in what the model committed on them -- the
    realistic shape of a matched panel. The cohort's groups are strongly
    heterogeneous in composition, which is what makes the two arms' intervals
    move together and their difference far better resolved than either.
    """

    rng = np.random.default_rng(99)
    shared = {"seed": 13, "cohort_kappa": 10.0}
    left = build_arm(
        name="left", delta_by_group=rng.normal(0.500, 0.004, size=20), **shared
    )
    right = build_arm(
        name="right", delta_by_group=rng.normal(0.490, 0.004, size=20), **shared
    )
    return [left, right]


def test_a_common_index_contrast_is_narrower_than_independent_arms_imply() -> None:
    """Non-overlap of two per-arm intervals is not a test of difference."""

    panel = bootstrap_arms(_paired_arms(), seed=21, n_bootstrap=DRAWS)
    contrast = panel.contrasts["left_minus_right"]
    assert not contrast["refused"]

    block = contrast["statistics"]["information_nats_per_token"]
    left = _information(panel.arms["left"].record)
    right = _information(panel.arms["right"].record)
    independent_se = math.hypot(left["bootstrap_se"], right["bootstrap_se"])

    # Positively correlated arms: Var(A - B) < Var(A) + Var(B).
    assert block["bootstrap_se"] < independent_se
    z = 1.959963984540054
    assert block["interval"][1] - block["interval"][0] < 2.0 * z * independent_se
    assert block["point"] == pytest.approx(left["point"] - right["point"], rel=1e-12)
    # The difference is resolved even though the per-arm intervals overlap.
    assert block["interval"][0] > 0.0
    assert left["interval"][0] < right["interval"][1]


def test_a_panel_refuses_a_contrast_whose_term_was_refused() -> None:
    scale = np.full(20, 1.0)
    scale[0] = 400.0
    good = build_arm(name="good", seed=14, information=0.5)
    bad = build_arm(name="bad", seed=14, information=0.5, group_token_scale=scale)
    panel = bootstrap_arms([good, bad], seed=21, n_bootstrap=DRAWS)

    assert not panel.arms["good"].refused
    assert panel.arms["bad"].refused
    contrast = panel.contrasts["good_minus_bad"]
    assert contrast["refused"]
    assert "bad" in contrast["refusal_reason"]
    assert contrast["statistics"] is None


def test_a_panel_refuses_arms_that_do_not_share_a_group_universe() -> None:
    left = build_arm(name="left", seed=15, n_groups=20)
    right = build_arm(name="right", seed=15, n_groups=19)
    with pytest.raises(ValueError, match="cohort groups differ"):
        bootstrap_arms([left, right], seed=21, n_bootstrap=DRAWS)


def test_arms_on_different_cohorts_are_contrasted_as_unpaired() -> None:
    """Two cohorts, no common unit: the interval carries the sum of the variances."""

    left = bootstrap_information(
        build_arm(name="left", seed=15, n_groups=20, information=0.5),
        seed=21,
        n_bootstrap=DRAWS,
    )
    right = bootstrap_information(
        build_arm(name="right", seed=16, n_groups=19, information=0.3),
        seed=22,
        n_bootstrap=DRAWS,
    )
    record = unpaired_contrast(left, right)

    assert record["paired"] is False
    assert record["common_resample_indices"] is False
    block = record["statistics"]["information_nats_per_token"]
    independent_se = math.hypot(
        _information(left.record)["bootstrap_se"],
        _information(right.record)["bootstrap_se"],
    )
    # Independent draws, so the contrast variance is the sum and nothing is
    # cancelled; the tolerance is Monte-Carlo noise at this draw count.
    assert block["bootstrap_se"] == pytest.approx(independent_se, rel=0.1)


def test_an_unpaired_contrast_refuses_two_bootstraps_from_one_seed() -> None:
    left = bootstrap_information(
        build_arm(name="left", seed=15, n_groups=20), seed=21, n_bootstrap=DRAWS
    )
    right = bootstrap_information(
        build_arm(name="right", seed=16, n_groups=19), seed=21, n_bootstrap=DRAWS
    )
    with pytest.raises(ValueError, match="one stream"):
        unpaired_contrast(left, right)


# --------------------------------------------------------------------------- #
# 10. The null control, where the true information is zero
# --------------------------------------------------------------------------- #


def test_a_control_fitted_on_the_baselines_own_reference_is_refused() -> None:
    """The degenerate case is refused rather than reported as a measurement.

    Fitted on the counts the baseline is fitted on, the control's ``q`` *is* the
    baseline's, ``I`` is zero at the point estimate by algebra rather than by
    evidence, and the interval around it describes only the noise of refitting
    one term. Both readings are indistinguishable from a real zero once written
    to an artefact, which is why this cannot be a warning.
    """

    arm = build_arm(seed=41)
    with pytest.raises(ValueError, match="identically rather than by measurement"):
        unigram_null_control(arm, arm.reference, name="arm::unigram-null")


def test_a_control_on_an_independent_reference_measures_a_known_zero() -> None:
    """Two independent references over one population put the true ``I`` at zero.

    The control's model term is the cross-entropy of the cohort under a smoothed
    unigram fitted on the *other* sample, so both terms of ``I`` estimate the
    same population and their difference has expectation zero. The measured
    value has to land near it, its interval has to cover it, and the operative
    0.30 nats/token floor has to refuse it -- while the real arm built on the
    same cohort reads its designed 0.5 nats.
    """

    arm = build_arm(seed=42, reference_seed=4201, information=0.5)
    other = build_arm(seed=42, reference_seed=4202, information=0.5)
    control = unigram_null_control(arm, other.reference, name="arm::unigram-null")

    # The cohort is untouched: only the model term is replaced, so the two arms
    # are averages over one token multiset and the contrast stays paired.
    assert np.array_equal(control.cohort.token_count, arm.cohort.token_count)
    assert np.array_equal(control.cohort.group_id, arm.cohort.group_id)
    assert np.array_equal(control.cohort.n_symbols, arm.cohort.n_symbols)
    assert control.reference is arm.reference

    # The per-record model term is the cross-entropy under the control's unigram,
    # computed here from the dense counts rather than the CSR layout.
    counts = _dense_counts(other.reference.targets, arm.vocab_size)
    q = (counts + arm.smoothing) / (counts.sum() + arm.smoothing * arm.vocab_size)
    targets = arm.cohort.targets
    expected = [
        float(-(targets.counts[a:b] * np.log(q[targets.unique_token_ids[a:b]])).sum())
        for a, b in zip(targets.record_offsets[:-1], targets.record_offsets[1:])
    ]
    assert control.cohort.clean_nll_sum == pytest.approx(expected)

    measured = _information(
        bootstrap_information(control, seed=9, n_bootstrap=DRAWS).record
    )
    real = _information(bootstrap_information(arm, seed=9, n_bootstrap=DRAWS).record)
    low, high = measured["interval"]
    assert low <= 0.0 <= high
    assert abs(measured["point"]) < 0.30
    assert real["point"] == pytest.approx(0.5, abs=0.05)

    # The departure from zero is of the order the smoothing constant works at,
    # not of the order the real effect works at.
    bound = math.log1p(arm.smoothing * arm.vocab_size / arm.reference.token_count.sum())
    assert abs(measured["point"]) < 10.0 * bound
    assert abs(measured["point"]) < 0.1 * real["point"]


def test_a_control_is_measured_by_the_same_estimator_as_the_arm_beside_it() -> None:
    """The control joins the panel; it does not get an estimator of its own.

    Adding it must leave every other arm's draws untouched -- ``bootstrap_arms``
    draws the multiplicities before it visits any arm -- and its own record must
    carry the same resampling unit, seed and draw count as the arm it sits with,
    so that a reader comparing the two is comparing measurements and not methods.
    """

    arm = build_arm(name="arm", seed=43, reference_seed=4301)
    other = build_arm(name="arm", seed=43, reference_seed=4302)
    control = unigram_null_control(arm, other.reference, name="arm::unigram-null")

    alone = bootstrap_arms([arm], seed=3, n_bootstrap=DRAWS, contrasts=())
    together = bootstrap_arms([arm, control], seed=3, n_bootstrap=DRAWS)

    assert (
        alone.arms["arm"].record["statistics"]
        == together.arms["arm"].record["statistics"]
    )
    real, null = (together.arms["arm"].record, together.arms["arm::unigram-null"].record)
    for field in ("seed", "n_bootstrap", "confidence", "resampling_unit", "refused"):
        assert real[field] == null[field], field
    assert set(real["statistics"]) == set(null["statistics"])
    assert real["diagnostics"]["n_groups"] == null["diagnostics"]["n_groups"]
    # Sharing the cohort draw is what makes the contrast against the control a
    # paired one rather than two intervals held up beside each other.
    contrast = together.contrasts["arm_minus_arm::unigram-null"]
    assert contrast["paired"] is True
    assert contrast["statistics"]["information_nats_per_token"]["interval"][0] > 0.0


# --------------------------------------------------------------------------- #
# The family of arms whose true information is a known multiple of an arm's
# --------------------------------------------------------------------------- #


def test_a_zero_weight_mixture_is_exactly_the_null_control() -> None:
    """The family's origin is the control itself, bit for bit and not merely near it.

    A calibration curve whose zero point is a *different* arm from the null
    control measures the criteria at two zeros and reports one; the reduction
    has to be exact, so it is asserted on the per-record statistic and on the
    published record rather than on the point estimate alone.
    """

    arm = build_arm(seed=51, reference_seed=5101, information=0.5)
    other = build_arm(seed=51, reference_seed=5102, information=0.5)

    control = unigram_null_control(arm, other.reference, name="arm::null")
    mixed = mixed_unigram_arm(arm, other.reference, name="arm::null", model_weight=0.0)

    assert np.array_equal(mixed.cohort.clean_nll_sum, control.cohort.clean_nll_sum)
    assert np.array_equal(mixed.cohort.token_count, control.cohort.token_count)
    assert np.array_equal(mixed.cohort.n_symbols, control.cohort.n_symbols)
    assert np.array_equal(mixed.cohort.group_id, control.cohort.group_id)
    assert mixed.reference is control.reference
    assert (
        bootstrap_information(mixed, seed=11, n_bootstrap=DRAWS).record
        == bootstrap_information(control, seed=11, n_bootstrap=DRAWS).record
    )


def test_the_mixed_arms_information_is_the_weighted_mixture_of_its_two_ends() -> None:
    """``I_mixed = (1 - w) I_control + w I_arm``, per draw and not only on average.

    Both terms of ``I`` are token-weighted averages over one multiset and only
    the model term moves, so the identity is algebra rather than approximation.
    It is what makes the family's true value known: the control's true ``I`` is
    zero, so the mixed arm's is ``w`` times the arm's, and a criterion can be
    watched at a chosen distance from zero. Asserted inside the bootstrap draws
    because that is where a decision rule reads it.
    """

    arm = build_arm(name="arm", seed=52, reference_seed=5201, information=0.5)
    other = build_arm(name="arm", seed=52, reference_seed=5202, information=0.5)
    weights = (0.0, 0.05, 0.2, 0.6, 1.0)
    panel = bootstrap_arms(
        [arm]
        + [
            mixed_unigram_arm(arm, other.reference, name=f"w{w}", model_weight=w)
            for w in weights
        ],
        seed=5,
        n_bootstrap=DRAWS,
        contrasts=(),
    )
    real = panel.arms["arm"].draws["information_nats_per_token"]
    control = panel.arms["w0.0"].draws["information_nats_per_token"]
    for w in weights:
        drawn = panel.arms[f"w{w}"].draws["information_nats_per_token"]
        assert drawn == pytest.approx((1.0 - w) * control + w * real, abs=1e-12)
        point = _information(panel.arms[f"w{w}"].record)["point"]
        # The true value is w * I_arm; what is measurable is that the estimate
        # tracks it to within the control's own departure from zero, which is
        # the error the pooled reading of I_arm is known up to.
        assert abs(point - w * 0.5) <= (1.0 - w) * abs(
            _information(panel.arms["w0.0"].record)["point"]
        ) + 0.05

    # At the top of the family the mixture is the arm: same records, same
    # statistic, and therefore the same interval under the same index.
    assert panel.arms["w1.0"].record["statistics"] == panel.arms["arm"].record[
        "statistics"
    ]


def test_a_mixed_arm_is_measured_by_the_same_estimator_and_reproduces_at_a_seed() -> None:
    """It joins the panel; it does not get an estimator, an index or a stream of its own."""

    arm = build_arm(name="arm", seed=53, reference_seed=5301, information=0.5)
    other = build_arm(name="arm", seed=53, reference_seed=5302, information=0.5)
    mixed = mixed_unigram_arm(arm, other.reference, name="arm::w05", model_weight=0.05)

    alone = bootstrap_arms([arm], seed=4, n_bootstrap=DRAWS, contrasts=())
    together = bootstrap_arms([arm, mixed], seed=4, n_bootstrap=DRAWS)
    assert (
        alone.arms["arm"].record["statistics"]
        == together.arms["arm"].record["statistics"]
    )
    real, synthetic = together.arms["arm"].record, together.arms["arm::w05"].record
    for field in ("seed", "n_bootstrap", "confidence", "resampling_unit", "refused"):
        assert real[field] == synthetic[field], field
    assert real["diagnostics"]["n_groups"] == synthetic["diagnostics"]["n_groups"]

    repeated = bootstrap_arms([arm, mixed], seed=4, n_bootstrap=DRAWS)
    assert repeated.record == together.record
    assert np.array_equal(
        repeated.arms["arm::w05"].draws["information_nats_per_token"],
        together.arms["arm::w05"].draws["information_nats_per_token"],
    )


def test_a_mixture_fitted_on_the_baselines_own_reference_is_refused_at_every_weight() -> None:
    """A family tautological at one weight calibrates nothing at any other."""

    arm = build_arm(seed=54)
    for weight in (0.0, 0.1, 1.0):
        with pytest.raises(ValueError, match="identically rather than by measurement"):
            mixed_unigram_arm(
                arm, arm.reference, name="arm::mixed", model_weight=weight
            )


def test_a_mixing_weight_outside_the_unit_interval_is_refused() -> None:
    arm = build_arm(seed=55, reference_seed=5501)
    other = build_arm(seed=55, reference_seed=5502)
    for weight in (-0.01, 1.5):
        with pytest.raises(ValueError, match="model_weight must lie"):
            mixed_unigram_arm(
                arm, other.reference, name="arm::mixed", model_weight=weight
            )


# --------------------------------------------------------------------------- #
# Input contract and draw request
# --------------------------------------------------------------------------- #


def test_sparse_counts_that_disagree_with_the_token_count_are_refused() -> None:
    arm = build_arm(seed=16)
    broken = arm.cohort.token_count.copy()
    broken[0] += 1
    with pytest.raises(ValueError, match="do not sum to token_count"):
        CohortStatistics(
            clean_nll_sum=arm.cohort.clean_nll_sum,
            token_count=broken,
            n_symbols=arm.cohort.n_symbols,
            targets=arm.cohort.targets,
            group_id=arm.cohort.group_id,
        )


def test_a_token_outside_the_declared_vocabulary_is_refused() -> None:
    arm = build_arm(seed=17, vocab_size=128)
    with pytest.raises(ValueError, match="outside the declared vocabulary"):
        ArmStatistics(
            name="arm",
            cohort=arm.cohort,
            reference=arm.reference,
            vocab_size=64,
            smoothing=1.0,
        )


def test_too_few_draws_for_the_requested_tail_are_refused() -> None:
    arm = build_arm(seed=18)
    with pytest.raises(ValueError, match="below the lower percentile"):
        bootstrap_information(arm, seed=7, n_bootstrap=100)


def test_every_published_result_carries_its_diagnostics() -> None:
    arm = build_arm(seed=19, information=0.4)
    result = bootstrap_information(arm, seed=7, n_bootstrap=DRAWS)
    diagnostics = result.record["diagnostics"]

    for key in (
        "n_groups",
        "n_effective_groups",
        "largest_group_token_share",
        "n_singleton_groups",
        "top10_record_token_share",
        "cohort_token_share_unseen_in_reference",
        "cohort_token_share_reference_count_at_most_5",
        "reference_n_effective_groups",
        "symbols_per_token",
    ):
        assert key in diagnostics
    assert result.record["seed"] == 7
    assert result.record["n_bootstrap"] == DRAWS
    for name, block in result.record["statistics"].items():
        assert set(block) == {
            "point",
            "interval",
            "confidence",
            "bootstrap_se",
            "bootstrap_bias",
            "median_bias_fraction_below_point",
            "median_bias_z0",
            "interval_mc_se",
            "fraction_of_draws_positive",
            "n_draws",
            "reference_resampling_displacement",
            "displacement_corrected_interval",
            "cohort_only_bootstrap_se",
            "reference_held_fixed_interval",
        }, name
        assert block["interval_mc_se"][0] > 0.0
        assert block["interval_mc_se"][1] > 0.0
        # Nothing is bias-corrected: the reported point is the estimate on the
        # data, so the interval and the bias are independent facts, and the
        # displacement-corrected interval is a second field beside them rather
        # than a replacement for the first.
        assert block["bootstrap_bias"] != 0.0

    # The model term is computed from the cohort alone, so resampling the
    # reference cannot move it and its displacement is exactly zero. Every other
    # statistic is a function of the fitted unigram and is displaced.
    statistics = result.record["statistics"]
    assert statistics["model_entropy_nats_per_token"][
        "reference_resampling_displacement"
    ] == 0.0
    assert (
        statistics["model_entropy_nats_per_token"]["displacement_corrected_interval"]
        == statistics["model_entropy_nats_per_token"]["interval"]
    )
    for name in (
        "baseline_entropy_nats_per_token",
        "information_nats_per_token",
        "relative_information",
        "information_bits_per_symbol",
    ):
        assert statistics[name]["reference_resampling_displacement"] != 0.0
        assert statistics[name]["interval"] != statistics[name][
            "displacement_corrected_interval"
        ]
