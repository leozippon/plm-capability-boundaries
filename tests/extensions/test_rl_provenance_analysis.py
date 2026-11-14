"""The best-of-k control, the declared band, and the readings the declaration fixes.

The best-of-k estimator is the experiment's decisive quantity, so it is checked
against brute-force enumeration rather than against itself, and the two places it
could flatter the control -- a noisy reward and a tied selection score -- are
checked to be handled as declared.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.capability.extensions import rl_environment_provenance as rp
from src.capability.extensions import rl_provenance_analysis as ra


class TestBestOfKWeights:
    def test_at_k_one_the_weights_are_uniform(self):
        weights = ra.best_of_k_weights(5, 1)
        assert weights == pytest.approx([0.2] * 5)

    def test_the_weights_are_a_distribution_at_every_declared_k(self):
        for k in rp.K_GRID:
            weights = ra.best_of_k_weights(37, k)
            assert weights.sum() == pytest.approx(1.0)
            assert (weights >= 0).all()

    def test_mass_moves_onto_the_best_rank_as_k_grows(self):
        first = [ra.best_of_k_weights(20, k)[0] for k in (1, 2, 8, 64)]
        assert first == sorted(first)
        assert first[-1] > 0.95

    def test_an_empty_pool_or_a_zero_k_is_refused(self):
        with pytest.raises(ValueError, match="non-empty pool and a positive k"):
            ra.best_of_k_weights(0, 1)
        with pytest.raises(ValueError, match="non-empty pool and a positive k"):
            ra.best_of_k_weights(5, 0)


class TestBestOfK:
    def test_r_at_one_is_the_pool_mean(self):
        values = np.array([0.1, 0.4, 0.9, 0.2, 0.55])
        assert ra.best_of_k(values, values, [1])[0] == pytest.approx(values.mean())

    def test_it_matches_brute_force_enumeration(self):
        """E[max of k draws with replacement], enumerated over every k-tuple."""

        values = np.array([0.0, 1.0, 2.0, 3.0, 7.0])
        for k in (1, 2, 3):
            brute = np.mean([max(combination) for combination
                             in itertools.product(values, repeat=k)])
            assert ra.best_of_k(values, values, [k])[0] == pytest.approx(brute)

    def test_it_matches_brute_force_when_selection_and_evaluation_differ(self):
        """The cross-fit case: the argmax is taken on one score, read off the other."""

        selection = np.array([0.0, 1.0, 2.0, 3.0])
        evaluation = np.array([10.0, 30.0, 20.0, 5.0])
        for k in (1, 2, 3):
            total = 0.0
            for combination in itertools.product(range(len(selection)), repeat=k):
                best = max(combination, key=lambda index: selection[index])
                total += evaluation[best]
            brute = total / len(selection) ** k
            assert ra.best_of_k(selection, evaluation, [k])[0] == pytest.approx(brute)

    def test_it_rises_with_k_and_approaches_the_pool_maximum(self):
        values = np.array([0.1, 0.2, 0.35, 0.5, 0.9])
        curve = ra.best_of_k(values, values, rp.K_GRID)
        assert list(curve) == sorted(curve)
        assert curve[-1] == pytest.approx(values.max(), abs=1e-3)

    def test_a_tie_group_is_credited_with_its_mean_evaluation_value(self):
        """Which member a selector returns among equals is arbitrary, so the
        expectation under a uniform tie-break is the group's mean."""

        selection = np.array([1.0, 1.0, 0.0, 0.0])
        evaluation = np.array([10.0, 20.0, 0.0, 100.0])
        # with k=2, the two top-ranked members take the mass that is not the
        # chance of both draws landing in the bottom group
        bottom = (2 / 4) ** 2
        expected = (1 - bottom) * 15.0 + bottom * 50.0
        assert ra.best_of_k(selection, evaluation, [2])[0] == pytest.approx(expected)

    def test_a_tie_break_does_not_depend_on_the_order_of_the_pool(self):
        selection = np.array([1.0, 0.0, 1.0, 0.0])
        evaluation = np.array([10.0, 0.0, 20.0, 100.0])
        permuted = [2, 0, 3, 1]
        assert ra.best_of_k(selection, evaluation, [2])[0] == pytest.approx(
            ra.best_of_k(selection[permuted], evaluation[permuted], [2])[0]
        )

    def test_a_non_finite_score_is_refused(self):
        with pytest.raises(ValueError, match="finite score"):
            ra.best_of_k([0.1, float("inf")], [0.1, 0.2], [1])

    def test_the_cross_fit_removes_the_winners_curse_a_naive_control_would_carry(self):
        """Selecting and evaluating on the same noisy score inflates the maximum.

        The signal here is constant, so every pool member has the same true
        value and no selection can do better than that value. A naive control
        reads back its own noise and climbs; the cross-fitted control stays flat.
        That direction matters: the inflation would push k* downward, which is the
        reading this project prefers.
        """

        rng = np.random.default_rng(0)
        truth = np.full(4000, 0.5)
        realisation_a = truth + rng.normal(0.0, 0.05, truth.size)
        realisation_b = truth + rng.normal(0.0, 0.05, truth.size)
        naive = ra.best_of_k(realisation_a, realisation_a, [64])[0]
        crossfit = ra.best_of_k(realisation_a, realisation_b, [64])[0]
        assert naive > 0.58          # three or more noise sigmas above the truth
        assert crossfit == pytest.approx(0.5, abs=0.01)


class TestBestOfKDraws:
    def _streams(self, rng):
        return {index: rng.uniform(size=64) for index in range(rp.STREAMS)}

    def test_the_draws_centre_on_the_point_estimate(self):
        rng = np.random.default_rng(1)
        streams = self._streams(rng)
        family = rp.family_name(rp.FAMILY_BEST_OF_K, rp.ARM_ENVIRONMENT)
        draws = ra.best_of_k_draws(streams, streams, family=family, column=0,
                                   ks=[1, 8], draws=400)
        pooled = np.concatenate([streams[key] for key in sorted(streams)])
        point = ra.best_of_k(pooled, pooled, [1, 8])
        assert draws.shape == (400, 2)
        assert draws[:, 0].mean() == pytest.approx(point[0], abs=0.02)
        assert draws[:, 1].mean() == pytest.approx(point[1], abs=0.03)

    def test_one_index_draw_serves_both_realisations(self):
        """Identical realisations must give identical selection and evaluation."""

        rng = np.random.default_rng(2)
        streams = self._streams(rng)
        family = rp.family_name(rp.FAMILY_BEST_OF_K, rp.ARM_ENVIRONMENT)
        same = ra.best_of_k_draws(streams, streams, family=family, column=1,
                                  ks=[4], draws=200)
        again = ra.best_of_k_draws(streams, dict(streams), family=family, column=1,
                                   ks=[4], draws=200)
        assert np.array_equal(same, again)

    def test_mismatched_realisations_are_refused(self):
        rng = np.random.default_rng(3)
        streams = self._streams(rng)
        other = {index: values[:-1] for index, values in streams.items()}
        family = rp.family_name(rp.FAMILY_BEST_OF_K, rp.ARM_ENVIRONMENT)
        with pytest.raises(ValueError, match="same sequences in the same order"):
            ra.best_of_k_draws(streams, other, family=family, column=0, ks=[2], draws=100)
        with pytest.raises(ValueError, match="different streams"):
            ra.best_of_k_draws(streams, {9: streams[0]}, family=family, column=0,
                               ks=[2], draws=100)


class TestPools:
    def test_the_pool_mean_is_stream_equal_and_not_sequence_equal(self):
        streams = {0: [0.0, 0.0, 0.0, 0.0], 1: [1.0]}
        assert ra.pool_mean(streams) == pytest.approx(0.5)

    def test_the_shortfall_names_every_stream_below_the_declared_floor(self):
        floor = ra.MIN_UNITS_PER_POOL
        streams = {0: [0.1] * floor, 1: [0.1] * (floor - 1), 2: []}
        assert ra.pool_shortfall(streams) == ["1", "2"]
        assert ra.pool_shortfall({}) == ["<no stream>"]

    def test_pool_draws_centre_on_the_pool_mean(self):
        rng = np.random.default_rng(4)
        streams = {index: rng.normal(1.0, 0.2, 128) for index in range(rp.STREAMS)}
        family = rp.FAMILY_ENTROPY
        draws = ra.pool_draws(streams, family=family, column=0, side=0, draws=1000)
        assert draws.mean() == pytest.approx(ra.pool_mean(streams), abs=0.01)

    def test_a_non_finite_value_in_a_pool_is_refused(self):
        streams = {0: [0.1, float("nan")], 1: [0.2, 0.3]}
        with pytest.raises(ValueError, match="non-finite"):
            ra.pool_draws(streams, family=rp.FAMILY_ENTROPY, column=0, side=0, draws=10)


class TestContrastBands:
    def _family(self, rng, *, shift=0.0):
        columns = []
        for pool in rp.CONTRAST_POOLS:
            left = {index: rng.normal(0.5 + shift, 0.1, 64) for index in range(rp.STREAMS)}
            right = {index: rng.normal(0.5, 0.1, 64) for index in range(rp.STREAMS)}
            columns.append({"key": pool, "left": left, "right": right})
        return columns

    def test_the_realised_family_must_be_the_declared_one(self):
        rng = np.random.default_rng(5)
        columns = self._family(rng)
        with pytest.raises(ValueError, match="declares 4 columns and realises 3"):
            ra.contrast_bands(columns[:-1], family=rp.FAMILY_ENTROPY, draws=200)
        with pytest.raises(ValueError, match="repeats a column key"):
            ra.contrast_bands(columns[:-1] + [dict(columns[0])],
                              family=rp.FAMILY_ENTROPY, draws=200)
        with pytest.raises(KeyError, match="no declared size"):
            ra.contrast_bands(columns, family="invented_family", draws=200)

    def test_one_critical_value_serves_the_whole_resolved_family(self):
        rng = np.random.default_rng(6)
        record = ra.contrast_bands(self._family(rng), family=rp.FAMILY_ENTROPY, draws=600)
        assert record["resolved_family_size"] == len(rp.CONTRAST_POOLS)
        critical = record["critical_value"]
        assert critical > 0
        for column in record["columns"].values():
            low, high = column["interval"]
            assert high - low == pytest.approx(2 * critical)
            assert low == pytest.approx(column["point"] - critical)

    def test_a_column_below_the_resampling_floor_is_unresolved_with_its_counts(self):
        rng = np.random.default_rng(7)
        columns = self._family(rng)
        columns[0]["right"] = {0: [0.5, 0.4], 1: [0.5], 2: [0.6]}
        record = ra.contrast_bands(columns, family=rp.FAMILY_ENTROPY, draws=400)
        first = record["columns"][rp.CONTRAST_POOLS[0]]
        assert first["status"] == "unresolved"
        assert first["point"] is None
        assert "below the resampling floor" in first["reason"]
        assert record["resolved_family_size"] == len(rp.CONTRAST_POOLS) - 1

    def test_a_stated_skip_reason_is_not_reported_as_a_floor_failure(self):
        rng = np.random.default_rng(8)
        columns = self._family(rng)
        columns[1]["skip_reason"] = "the homology search over this pool has not run"
        record = ra.contrast_bands(columns, family=rp.FAMILY_ENTROPY, draws=400)
        entry = record["columns"][rp.CONTRAST_POOLS[1]]
        assert entry["status"] == "unresolved"
        assert entry["reason"] == "the homology search over this pool has not run"

    def test_a_family_with_no_resolved_column_carries_no_band(self):
        rng = np.random.default_rng(9)
        columns = self._family(rng)
        for column in columns:
            column["skip_reason"] = "not measured"
        record = ra.contrast_bands(columns, family=rp.FAMILY_ENTROPY, draws=200)
        assert record["critical_value"] is None
        assert "no column of this family clears" in record["reason"]

    def test_a_precomputed_draw_pair_enters_as_a_column(self):
        """How a best-of-k column enters: its left side is not a pool mean."""

        rng = np.random.default_rng(10)
        columns = []
        for pool in rp.CONTRAST_POOLS:
            columns.append({
                "key": pool,
                "left_point": 0.6, "right_point": 0.5,
                "left_draws": rng.normal(0.6, 0.02, 500),
                "right_draws": rng.normal(0.5, 0.02, 500),
            })
        record = ra.contrast_bands(columns, family=rp.FAMILY_ENTROPY, draws=500)
        assert record["resolved_family_size"] == len(rp.CONTRAST_POOLS)
        for column in record["columns"].values():
            assert column["point"] == pytest.approx(0.1)

    def test_the_studentised_form_is_carried_as_a_sensitivity_only(self):
        rng = np.random.default_rng(11)
        record = ra.contrast_bands(self._family(rng), family=rp.FAMILY_ENTROPY, draws=600)
        assert record["standardized"]["role"].startswith("sensitivity")
        for column in record["columns"].values():
            assert "standardized_sensitivity" in column


class TestReadings:
    def test_optimisation_needs_the_whole_interval_above_zero(self):
        assert ra.optimisation_finding([0.01, 0.2])["optimised"] is True
        assert ra.optimisation_finding([-0.01, 0.2])["optimised"] is False
        assert ra.optimisation_finding([-0.3, -0.1])["optimised"] is False
        assert ra.optimisation_finding(None)["optimised"] is None
        assert "wrong way" in ra.optimisation_finding([-0.3, -0.1])["reason"]
        assert "not evidence that the capability is absent" in \
            ra.optimisation_finding([-0.01, 0.2])["reason"]

    def _best_of_k_columns(self, reached_from: int | None):
        columns = {}
        for k in rp.K_GRID:
            interval = ([0.01, 0.1] if reached_from is not None and k >= reached_from
                        else [-0.2, -0.05])
            columns[f"memory|k={k}"] = {"status": "resolved", "interval": interval}
        return columns

    def test_k_star_is_the_first_declared_k_whose_interval_clears_zero(self):
        found = ra.k_star(self._best_of_k_columns(16), pool="memory")
        assert found["k_star"] == 16
        assert found["redistribution"] is True

    def test_a_k_star_above_the_declared_threshold_is_not_redistribution(self):
        found = ra.k_star(self._best_of_k_columns(256), pool="memory")
        assert found["k_star"] == 256
        assert found["redistribution"] is False

    def test_an_unreached_grid_names_its_ceiling_rather_than_an_infinity(self):
        found = ra.k_star(self._best_of_k_columns(None), pool="memory")
        assert found["k_star"] is None
        assert found["k_ceiling"] == rp.K_GRID[-1]
        assert found["redistribution"] is False
        assert "greater than the grid's ceiling" in found["reason"]

    def test_an_unresolved_grid_leaves_redistribution_undetermined(self):
        columns = self._best_of_k_columns(None)
        for key in columns:
            columns[key] = {"status": "unresolved", "interval": None}
        found = ra.k_star(columns, pool="memory")
        assert found["k_star"] is None
        assert found["redistribution"] is None
        assert len(found["k_values_unresolved"]) == len(rp.K_GRID)

    def test_the_threshold_is_the_declared_one(self):
        assert ra.k_star(self._best_of_k_columns(rp.K_REDISTRIBUTION),
                         pool="memory")["redistribution"] is True
        nxt = rp.K_GRID[rp.K_GRID.index(rp.K_REDISTRIBUTION) + 1]
        assert ra.k_star(self._best_of_k_columns(nxt), pool="memory")["redistribution"] is False

    def test_support_tail_share_is_read_against_its_own_null(self):
        base = np.linspace(0.0, 1.0, 1000)
        contained = ra.support_tail_share(base, base)
        assert contained["null_share"] == pytest.approx(1 - rp.SUPPORT_TAIL_QUANTILE)
        assert contained["share_above"] == pytest.approx(0.01, abs=0.002)
        outside = ra.support_tail_share(np.full(100, 5.0), base)
        assert outside["share_above"] == 1.0

    def test_support_tail_share_refuses_an_empty_or_non_finite_pool(self):
        with pytest.raises(ValueError, match="non-empty pool"):
            ra.support_tail_share([], [0.1, 0.2])
        with pytest.raises(ValueError, match="finite NLL"):
            ra.support_tail_share([float("nan")], [0.1, 0.2])

    def test_the_reward_range_share_states_its_ceiling(self):
        base = np.array([0.0, 0.5, 1.0])
        record = ra.reward_range_share(np.array([1.0, 0.9]), base)
        assert record["base_saturates_reward_bound"] is True
        assert record["share_above"] == 0.0
        assert "identically zero" in record["ceiling_note"]

    def _degeneracy(self, **columns):
        base = {"mean_ca_plddt": 50.0, "distinct_5mer_fraction_quantile": 0.35,
                "longest_single_residue_run_quantile": 12.0,
                "composition_entropy_bits": 3.3}
        summaries = {rp.FAMILY_PLDDT: {"interval": [48.0, 52.0]},
                     rp.FAMILY_DISTINCT_5MER: {"interval": [0.9, 1.0]},
                     rp.FAMILY_LONGEST_RUN: {"interval": [3.0, 5.0]},
                     rp.FAMILY_COMPOSITION_ENTROPY: {"interval": [3.2, 3.4]}}
        summaries.update(columns)
        return ra.degeneracy_breaches(summaries, base)

    def test_a_protein_like_pool_breaches_nothing(self):
        assert self._degeneracy()["breached"] is False

    def test_each_declared_bound_is_breached_by_the_thing_it_names(self):
        collapsed = self._degeneracy(**{rp.FAMILY_PLDDT: {"interval": [20.0, 30.0]}})
        assert rp.FAMILY_PLDDT in collapsed["breaches"]
        repetitive = self._degeneracy(**{rp.FAMILY_DISTINCT_5MER: {"interval": [0.1, 0.2]}})
        assert rp.FAMILY_DISTINCT_5MER in repetitive["breaches"]
        homopolymer = self._degeneracy(**{rp.FAMILY_LONGEST_RUN: {"interval": [40.0, 60.0]}})
        assert rp.FAMILY_LONGEST_RUN in homopolymer["breaches"]
        narrow = self._degeneracy(**{rp.FAMILY_COMPOSITION_ENTROPY: {"interval": [1.0, 1.5]}})
        assert rp.FAMILY_COMPOSITION_ENTROPY in narrow["breaches"]

    def test_an_unmeasured_bound_is_named_rather_than_passed(self):
        record = self._degeneracy(**{rp.FAMILY_PLDDT: {"interval": None}})
        assert rp.FAMILY_PLDDT in record["unmeasured_bounds"]
        assert rp.FAMILY_PLDDT not in record["breaches"]

    def _reading(self, *, optimised, k_star_value, share, breached):
        return ra.reading(
            optimised={"optimised": optimised, "reason": "declared test fixture"},
            k={"k_star": k_star_value,
               "redistribution": None if k_star_value is None and share is None
               else (k_star_value is not None and k_star_value <= rp.K_REDISTRIBUTION),
               "reason": "declared test fixture"},
            support={"share_above": share or 0.0, "null_share": 0.01},
            degeneracy={"breached": breached, "breaches": {"x": "a declared bound"}},
        )

    def test_a_failed_optimisation_is_read_as_one_and_nothing_further(self):
        got = self._reading(optimised=False, k_star_value=512, share=0.9, breached=False)
        assert got["reading"] == rp.READING_NO_OPTIMISATION

    def test_a_small_k_star_is_redistribution_whatever_the_support_shows(self):
        got = self._reading(optimised=True, k_star_value=16, share=0.9, breached=False)
        assert got["reading"] == rp.READING_REDISTRIBUTION

    def test_an_installed_bias_needs_all_three_conditions(self):
        on_manifold = self._reading(optimised=True, k_star_value=256, share=0.3,
                                    breached=False)
        assert on_manifold["reading"] == rp.READING_BIAS_ON_MANIFOLD
        off_manifold = self._reading(optimised=True, k_star_value=256, share=0.3,
                                     breached=True)
        assert off_manifold["reading"] == rp.READING_BIAS_OFF_MANIFOLD
        contained = self._reading(optimised=True, k_star_value=256, share=0.005,
                                  breached=False)
        assert contained["reading"] == rp.READING_UNRESOLVED

    def test_an_undetermined_k_star_is_unresolved_not_a_bias(self):
        got = ra.reading(
            optimised={"optimised": True, "reason": "fixture"},
            k={"k_star": None, "redistribution": None, "reason": "columns unresolved"},
            support={"share_above": 0.9, "null_share": 0.01},
            degeneracy={"breached": False, "breaches": {}},
        )
        assert got["reading"] == rp.READING_UNRESOLVED


class TestDeclaration:
    def test_every_declared_family_carries_a_distinct_seed_block(self):
        offsets = rp.FAMILY_SEED_OFFSETS
        assert set(offsets) == set(rp.FAMILY_SIZES)
        assert len(set(offsets.values())) == len(offsets)

    def test_a_pool_seed_is_a_function_of_its_family_position(self):
        family = rp.FAMILY_ENTROPY
        seeds = {rp.pool_draw_seed(family, column, side, stream)
                 for column in range(4) for side in (0, 1) for stream in range(3)}
        assert len(seeds) == 24

    def test_an_undeclared_family_or_an_out_of_range_position_is_refused(self):
        with pytest.raises(KeyError, match="no declared seed block"):
            rp.pool_draw_seed("invented", 0, 0, 0)
        with pytest.raises(ValueError, match="at most"):
            rp.pool_draw_seed(rp.FAMILY_ENTROPY, rp.MAX_FAMILY_COLUMNS, 0, 0)
        with pytest.raises(ValueError, match="two sides"):
            rp.pool_draw_seed(rp.FAMILY_ENTROPY, 0, 2, 0)

    def test_pool_seeds_do_not_overlap_between_pools(self):
        seeds = [seed for pool in rp.EVALUATED_POOLS for seed in rp.pool_seeds(pool)]
        assert len(set(seeds)) == len(seeds)

    def test_the_redistribution_threshold_is_a_point_of_the_declared_grid(self):
        assert rp.K_REDISTRIBUTION in rp.K_GRID

    def test_the_base_pool_is_deep_enough_for_the_deepest_declared_k(self):
        assert (rp.BASE_ATTEMPTS_PER_STREAM * rp.STREAMS
                >= rp.K_GRID[-1] * rp.DEEPEST_K_POOL_MULTIPLE)

    def test_only_the_base_pool_is_folded_twice(self):
        assert rp.DOUBLE_FOLDED_POOLS == (rp.POOL_BASE,)

    def test_the_contract_is_serialisable_and_names_its_declaration(self):
        import json

        contract = rp.contract()
        assert contract["declared_in"] == "docs/rl-environment-provenance.md"
        assert json.loads(json.dumps(contract))["campaign"] == rp.CAMPAIGN
        assert set(contract["families"]) == set(rp.FAMILY_SIZES)
