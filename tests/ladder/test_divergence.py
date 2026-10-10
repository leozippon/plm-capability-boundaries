"""The ladder's statistics: stratified correlation, intervals, and where it diverges."""

from __future__ import annotations

import numpy as np
import pytest

from src.capability.ladder import divergence as dv


def _statistic(values, strata):
    return dv.within_stratum_rank_correlation(values["x"], values["y"], strata)


# ------------------------------------------------- the stratified correlation


def test_a_within_stratum_correlation_survives_an_opposite_between_stratum_trend():
    """This is why the estimand is stratified and not pooled.

    Each backbone's own draws agree perfectly, while the backbones themselves run
    the other way -- exactly the shape a length effect across backbones of 150 to
    300 residues produces. A pooled correlation reads the between-backbone trend;
    the within-backbone one reads the question.
    """

    x: list[float] = []
    y: list[float] = []
    strata: list[str] = []
    for index in range(10):
        offset = 100.0 * index
        for draw in range(5):
            x.append(offset + draw)
            y.append(-offset + draw)
            strata.append(f"b{index}")
    assert dv.within_stratum_rank_correlation(x, y, strata) == pytest.approx(1.0)
    pooled = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]
    assert pooled < 0.0


def test_a_perfectly_reversed_within_stratum_relation_reads_minus_one():
    x = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    y = [3.0, 2.0, 1.0, 30.0, 20.0, 10.0]
    assert dv.within_stratum_rank_correlation(x, y, ["a", "a", "a", "b", "b", "b"]) == pytest.approx(-1.0)


def test_a_constant_vector_inside_every_stratum_yields_no_correlation():
    assert np.isnan(
        dv.within_stratum_rank_correlation([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "b"])
    )


def test_a_singleton_stratum_contributes_nothing_rather_than_being_dropped_silently():
    x = [1.0, 2.0, 3.0, 99.0]
    y = [1.0, 2.0, 3.0, -99.0]
    both = dv.within_stratum_rank_correlation(x, y, ["a", "a", "a", "b"])
    without = dv.within_stratum_rank_correlation(x[:3], y[:3], ["a", "a", "a"])
    assert both == pytest.approx(without)


def test_non_finite_input_is_refused_rather_than_silently_dropped():
    with pytest.raises(ValueError, match="non-finite"):
        dv.within_stratum_rank_correlation([1.0, float("nan")], [1.0, 2.0], ["a", "a"])


def test_misaligned_inputs_are_refused():
    with pytest.raises(ValueError):
        dv.within_stratum_rank_correlation([1.0, 2.0], [1.0], ["a", "a"])


# --------------------------------------------------- the composition adjustment


def test_a_correlation_carried_entirely_by_a_covariate_does_not_survive_adjustment():
    generator = np.random.default_rng(11)
    covariate, x, y, strata = [], [], [], []
    for index in range(12):
        for draw in range(8):
            z = generator.normal()
            covariate.append(z)
            x.append(z + 0.4 * generator.normal())
            y.append(z + 0.4 * generator.normal())
            strata.append(f"b{index}")
    raw = dv.within_stratum_rank_correlation(x, y, strata)
    adjusted = dv.partial_within_stratum_rank_correlation(x, y, covariate, strata)
    assert raw > 0.7
    assert abs(adjusted) < 0.2


def test_a_covariate_that_explains_everything_leaves_nothing_to_correlate():
    """Reported as undefined, not as a correlation of zero."""

    covariate = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    strata = ["a"] * 4 + ["b"] * 4
    assert np.isnan(
        dv.partial_within_stratum_rank_correlation(covariate, covariate, covariate, strata)
    )


def test_a_correlation_independent_of_the_covariate_survives_adjustment():
    generator = np.random.default_rng(12)
    covariate, x, y, strata = [], [], [], []
    for index in range(12):
        for draw in range(8):
            covariate.append(generator.normal())
            signal = generator.normal()
            x.append(signal)
            y.append(signal + 0.01 * generator.normal())
            strata.append(f"b{index}")
    assert dv.partial_within_stratum_rank_correlation(x, y, covariate, strata) > 0.9


# ------------------------------------------------------------ per-stratum view


def test_the_per_stratum_view_reports_one_correlation_per_backbone():
    x, y, strata = [], [], []
    for index in range(4):
        for draw in range(6):
            x.append(float(draw))
            y.append(float(draw) if index % 2 == 0 else float(-draw))
            strata.append(f"b{index}")
    record = dv.per_stratum_spearman(x, y, strata)
    assert record["n_strata_scored"] == 4
    assert record["mean"] == pytest.approx(0.0)
    assert sorted(record["per_stratum"].values()) == pytest.approx([-1.0, -1.0, 1.0, 1.0])


# --------------------------------------------------------------- the bootstrap


def _cell(effect, *, n_units=16, n_draws=8, seed=0, noise=1.0):
    generator = np.random.default_rng(seed)
    x, y, strata = [], [], []
    for index in range(n_units):
        for _ in range(n_draws):
            signal = generator.normal()
            x.append(signal + 50.0 * index)
            y.append(effect * signal + noise * generator.normal() + 50.0 * index)
            strata.append(f"b{index}")
    return {"x": x, "y": y}, strata, [f"b{index}" for index in range(n_units)]


def test_the_resamples_are_refused_below_the_unit_floor():
    with pytest.raises(ValueError, match="refused"):
        dv.stratum_resamples([f"b{i}" for i in range(4)], seed=1, draws=10)


def test_a_cluster_bootstrap_interval_brackets_a_strong_effect_and_excludes_zero():
    values, strata, units = _cell(3.0, seed=1)
    resamples = dv.stratum_resamples(units, seed=2, draws=400)
    cell = dv.cluster_bootstrap(values, strata, _statistic, units=units, resamples=resamples)
    assert cell["n_observations"] == 128 and cell["n_units"] == 16 and cell["n_panel_units"] == 16
    assert cell["point"] > 0.5
    assert cell["interval"][0] > 0.0
    assert cell["bootstrap_sd"] > 0.0
    assert cell["draws"].shape == (400,)


def test_a_cluster_bootstrap_interval_admits_zero_when_there_is_no_effect():
    values, strata, units = _cell(0.0, seed=5)
    resamples = dv.stratum_resamples(units, seed=6, draws=400)
    cell = dv.cluster_bootstrap(values, strata, _statistic, units=units, resamples=resamples)
    assert cell["interval"][0] < 0.0 < cell["interval"][1]


def test_resamples_that_do_not_match_the_unit_list_are_refused():
    values, strata, units = _cell(1.0)
    with pytest.raises(ValueError, match="over the supplied unit list"):
        dv.cluster_bootstrap(
            values, strata, _statistic, units=units, resamples=np.zeros((10, 3), dtype=int)
        )


def test_a_cell_missing_some_units_still_bootstraps_over_the_whole_panel():
    values, strata, units = _cell(2.0, seed=9)
    keep = [index for index, stratum in enumerate(strata) if stratum not in {"b0", "b1"}]
    partial = {name: [vector[index] for index in keep] for name, vector in values.items()}
    partial_strata = [strata[index] for index in keep]
    resamples = dv.stratum_resamples(units, seed=10, draws=200)
    cell = dv.cluster_bootstrap(
        partial, partial_strata, _statistic, units=units, resamples=resamples
    )
    assert cell["n_units"] == 14 and cell["n_panel_units"] == 16


# ------------------------------------------------------- the simultaneous band


def test_the_simultaneous_band_is_never_narrower_than_the_marginal_interval():
    units = [f"b{index}" for index in range(16)]
    resamples = dv.stratum_resamples(units, seed=3, draws=400)
    cells = {}
    for index, effect in enumerate((3.0, 1.0, 0.3, 0.0, -0.5)):
        values, strata, _ = _cell(effect, seed=20 + index)
        cells[f"c{index}"] = dv.cluster_bootstrap(
            values, strata, _statistic, units=units, resamples=resamples
        )
    band = dv.simultaneous_band(cells)
    assert band["quantile"] > 1.96
    for name, cell in cells.items():
        marginal = cell["interval"][1] - cell["interval"][0]
        joint = band["intervals"][name][1] - band["intervals"][name][0]
        assert joint >= marginal - 1e-9, name


def test_cells_recomputed_on_different_resamples_cannot_form_a_joint_statement():
    units = [f"b{index}" for index in range(16)]
    values, strata, _ = _cell(1.0, seed=31)
    short = dv.cluster_bootstrap(
        values, strata, _statistic, units=units, resamples=dv.stratum_resamples(units, seed=1, draws=50)
    )
    long = dv.cluster_bootstrap(
        values, strata, _statistic, units=units, resamples=dv.stratum_resamples(units, seed=1, draws=60)
    )
    with pytest.raises(ValueError, match="same resamples"):
        dv.simultaneous_band({"a": short, "b": long})


def test_a_cell_without_a_usable_spread_is_named_rather_than_given_a_band():
    band = dv.simultaneous_band({"a": {"point": 0.5, "bootstrap_sd": float("nan"), "draws": [0.5]}})
    assert band["intervals"] == {}
    assert band["excluded_cells"] == ["a"]


# --------------------------------------------------------- locating divergence


def _ladder(points, intervals):
    return [
        {"rung": f"k{extent}", "point": point, "interval": interval}
        for extent, point, interval in zip((1, 2, 5, 10, 20, 40), points, intervals)
    ]


def test_the_divergence_rung_is_the_first_one_that_admits_zero_and_stays_there():
    ladder = _ladder(
        [0.6, 0.5, 0.4, 0.1, 0.05, -0.02],
        [[0.4, 0.8], [0.3, 0.7], [0.2, 0.6], [-0.1, 0.3], [-0.2, 0.3], [-0.3, 0.2]],
    )
    record = dv.divergence_rung(ladder)
    assert record["loses_significance"] == "k10"
    assert record["undetermined_at_bottom"] is False


def test_a_single_unresolved_rung_inside_a_resolved_ladder_is_not_a_divergence():
    ladder = _ladder(
        [0.6, 0.5, 0.1, 0.4, 0.4, 0.4],
        [[0.4, 0.8], [0.3, 0.7], [-0.1, 0.3], [0.2, 0.6], [0.2, 0.6], [0.2, 0.6]],
    )
    assert dv.divergence_rung(ladder)["loses_significance"] is None


def test_a_sign_change_is_reported_separately_from_loss_of_significance():
    ladder = _ladder(
        [0.6, 0.5, 0.2, -0.3, -0.4, -0.5],
        [[0.4, 0.8], [0.3, 0.7], [0.05, 0.4], [-0.5, -0.1], [-0.6, -0.2], [-0.7, -0.3]],
    )
    record = dv.divergence_rung(ladder)
    assert record["changes_sign"] == "k10"
    assert record["loses_significance"] is None


def test_a_ladder_unresolved_at_the_bottom_has_no_divergence_point_to_find():
    ladder = _ladder(
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [[-0.2, 0.4]] * 6,
    )
    record = dv.divergence_rung(ladder)
    assert record["undetermined_at_bottom"] is True
    assert record["loses_significance"] is None and record["changes_sign"] is None


def test_the_simultaneous_reading_is_reported_beside_the_marginal_one():
    ladder = _ladder(
        [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [[0.4, 0.8], [0.3, 0.7], [0.2, 0.6], [0.1, 0.5], [0.05, 0.35], [-0.05, 0.25]],
    )
    simultaneous = {
        "k1": [0.3, 0.9],
        "k2": [0.2, 0.8],
        "k5": [-0.1, 0.9],
        "k10": [-0.2, 0.8],
        "k20": [-0.3, 0.7],
        "k40": [-0.4, 0.6],
    }
    record = dv.divergence_rung(ladder, simultaneous=simultaneous)
    assert record["loses_significance"] == "k40"
    assert record["loses_significance_simultaneous"] == "k5"


# ------------------------------------------------------- the continuous axis


def test_a_planted_crossing_is_recovered_inside_the_measured_range():
    extents = [1, 2, 5, 10, 20, 40]
    # A line in log2 k that reaches zero at k = 8.
    points = [0.3 * (3.0 - np.log2(extent)) for extent in extents]
    matrix = np.tile(points, (200, 1)) + np.random.default_rng(4).normal(0, 0.01, size=(200, 6))
    record = dv.continuous_crossing(extents, matrix, point_estimates=points)
    assert record["crossing_extent"] == pytest.approx(8.0, rel=1e-6)
    assert record["interval"][0] < 8.0 < record["interval"][1]
    assert record["fraction_without_crossing"] < 0.05


def test_a_ladder_that_never_reaches_zero_reports_no_crossing():
    extents = [1, 2, 5, 10, 20, 40]
    points = [0.6, 0.58, 0.56, 0.55, 0.54, 0.53]
    matrix = np.tile(points, (100, 1))
    record = dv.continuous_crossing(extents, matrix, point_estimates=points)
    assert record["crossing_extent"] is None
    assert record["fraction_without_crossing"] == pytest.approx(1.0)
    assert record["measured_range"] == [1.0, 40.0]


def test_a_crossing_needs_at_least_three_rungs():
    with pytest.raises(ValueError, match="at least three"):
        dv.continuous_crossing([1, 2], np.zeros((5, 2)), point_estimates=[0.1, 0.1])


# ------------------------------------------------------------- sample sizing


def test_the_required_unit_count_scales_as_the_inverse_square_of_the_effect():
    small = dv.required_units(bootstrap_sd=0.2, n_units=16, target_effect=0.2)
    smaller = dv.required_units(bootstrap_sd=0.2, n_units=16, target_effect=0.1)
    assert smaller["required_n_units"] > small["required_n_units"]
    assert smaller["required_n_units"] == pytest.approx(4 * small["required_n_units"], rel=0.02)
    assert small["power"] == 0.8


def test_a_zero_effect_cannot_be_resolved_at_any_sample_size():
    with pytest.raises(ValueError, match="cannot be resolved"):
        dv.required_units(bootstrap_sd=0.2, n_units=16, target_effect=0.0)


def test_a_scaling_statement_needs_an_admissible_interval():
    with pytest.raises(ValueError, match="admissible"):
        dv.required_units(bootstrap_sd=0.2, n_units=4, target_effect=0.2)
