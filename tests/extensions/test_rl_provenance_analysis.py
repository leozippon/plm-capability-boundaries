"""The best-of-k control, the declared band, and the readings the declaration fixes.

The best-of-k estimator is the experiment's decisive quantity, so it is checked
against brute-force enumeration rather than against itself, and the two places it
could flatter the control -- a noisy reward and a tied selection score -- are
checked to be handled as declared.
"""

from __future__ import annotations

import itertools
from pathlib import Path

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
        realised = len(rp.CONTRAST_POOLS) - 1
        with pytest.raises(ValueError,
                           match=f"declares {len(rp.CONTRAST_POOLS)} columns and "
                                 f"realises {realised}"):
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

    #: The base pool's own levels, as the analyser derives them. The bounds are
    #: limits on the pool-minus-base contrast, so a pool that matches the base
    #: carries an interval around zero and must breach nothing.
    BASE_LEVELS = {"mean_ca_plddt": 50.0,
                   "distinct_5mer_fraction_mean": 0.9,
                   "distinct_5mer_fraction_quantile": 0.35,
                   "longest_single_residue_run_mean": 4.0,
                   "longest_single_residue_run_quantile": 12.0,
                   "composition_entropy_bits": 3.3}

    def _degeneracy(self, **columns):
        summaries = {rp.FAMILY_PLDDT: {"interval": [-2.0, 2.0]},
                     rp.FAMILY_DISTINCT_5MER: {"interval": [-0.02, 0.02]},
                     rp.FAMILY_LONGEST_RUN: {"interval": [-1.0, 1.0]},
                     rp.FAMILY_COMPOSITION_ENTROPY: {"interval": [-0.1, 0.1]}}
        summaries.update(columns)
        return ra.degeneracy_breaches(summaries, self.BASE_LEVELS)

    def test_the_bounds_are_limits_on_the_contrast_not_on_the_level(self):
        """The banded quantity is the contrast, so the limits have to be too.

        Reading a level bound against a contrast interval is the one way this
        could go wrong without looking wrong: every pool would breach every
        bound, because a contrast near zero sits far below the base's level.
        """

        offsets = ra.degeneracy_offsets(self.BASE_LEVELS)
        assert offsets[rp.FAMILY_PLDDT]["limit"] == -rp.PLDDT_DEGENERACY_MARGIN
        assert offsets[rp.FAMILY_DISTINCT_5MER]["limit"] == pytest.approx(0.35 - 0.9)
        assert offsets[rp.FAMILY_LONGEST_RUN]["limit"] == pytest.approx(2 * 12.0 - 4.0)
        assert offsets[rp.FAMILY_LONGEST_RUN]["side"] == "above"
        assert offsets[rp.FAMILY_COMPOSITION_ENTROPY]["limit"] == pytest.approx(-1.0)

    def test_a_protein_like_pool_breaches_nothing(self):
        record = self._degeneracy()
        assert record["breached"] is False
        assert record["unmeasured_bounds"] == []

    def test_each_declared_bound_is_breached_by_the_thing_it_names(self):
        collapsed = self._degeneracy(**{rp.FAMILY_PLDDT: {"interval": [-30.0, -20.0]}})
        assert rp.FAMILY_PLDDT in collapsed["breaches"]
        repetitive = self._degeneracy(**{rp.FAMILY_DISTINCT_5MER: {"interval": [-0.8, -0.7]}})
        assert rp.FAMILY_DISTINCT_5MER in repetitive["breaches"]
        homopolymer = self._degeneracy(**{rp.FAMILY_LONGEST_RUN: {"interval": [30.0, 60.0]}})
        assert rp.FAMILY_LONGEST_RUN in homopolymer["breaches"]
        narrow = self._degeneracy(**{rp.FAMILY_COMPOSITION_ENTROPY: {"interval": [-2.0, -1.5]}})
        assert rp.FAMILY_COMPOSITION_ENTROPY in narrow["breaches"]

    def test_a_point_past_the_limit_with_an_interval_over_it_is_not_a_breach(self):
        marginal = self._degeneracy(**{rp.FAMILY_PLDDT: {"interval": [-18.0, -2.0]}})
        assert rp.FAMILY_PLDDT not in marginal["breaches"]
        assert marginal["measured"][rp.FAMILY_PLDDT]["breached"] is False

    def test_an_unmeasured_bound_is_named_rather_than_passed(self):
        record = self._degeneracy(**{rp.FAMILY_PLDDT: {"interval": None}})
        assert rp.FAMILY_PLDDT in record["unmeasured_bounds"]
        assert rp.FAMILY_PLDDT not in record["breaches"]

    def test_a_base_reference_with_no_levels_measures_no_bound(self):
        record = ra.degeneracy_breaches({}, {})
        assert record["breached"] is False
        assert sorted(record["unmeasured_bounds"]) == sorted(rp.DEGENERACY_BOUNDS)

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


class TestDigestJoin:
    """The one join from distinct-sequence measurements back onto attempts."""

    def _rows(self):
        return [
            {"stream": 0, "length": 100, "sequence_sha256": "aa"},
            {"stream": 0, "length": 100, "sequence_sha256": "aa"},
            {"stream": 1, "length": 90, "sequence_sha256": "bb"},
            {"stream": 2, "length": 0, "sequence_sha256": "cc"},
        ]

    def test_a_shared_sequence_is_counted_once_per_attempt(self):
        """The attempt is the resampling unit, so a duplicate sequence is two units."""

        streams = ra.streams_by_digest(self._rows(), {"aa": 1.0, "bb": 2.0},
                                       pool="base", quantity="likelihood")
        assert streams == {0: [1.0, 1.0], 1: [2.0]}

    def test_an_empty_attempt_carries_no_measurement_and_is_skipped(self):
        streams = ra.streams_by_digest(self._rows(), {"aa": 1.0, "bb": 2.0},
                                       pool="base", quantity="likelihood")
        assert 2 not in streams

    def test_a_scored_table_missing_a_sequence_is_refused(self):
        with pytest.raises(ValueError, match="have no likelihood in the supplied table"):
            ra.streams_by_digest(self._rows(), {"aa": 1.0}, pool="base",
                                 quantity="likelihood")

    def test_a_pool_with_nothing_scorable_is_refused(self):
        with pytest.raises(ValueError, match="no attempt with a likelihood"):
            ra.streams_by_digest([{"stream": 0, "length": 0, "sequence_sha256": "cc"}],
                                 {}, pool="base", quantity="likelihood")

    def test_the_join_feeds_the_band_machinery_directly(self):
        rng = np.random.default_rng(42)
        rows = [{"stream": s, "length": 100, "sequence_sha256": f"{s}-{i}"}
                for s in range(rp.STREAMS) for i in range(32)]
        scores = {row["sequence_sha256"]: float(rng.normal(1.0, 0.2)) for row in rows}
        streams = ra.streams_by_digest(rows, scores, pool="base", quantity="likelihood")
        assert sorted(streams) == list(range(rp.STREAMS))
        assert ra.pool_mean(streams) == pytest.approx(1.0, abs=0.15)


class TestReachabilityLadder:
    """The base pool's own reachability at harder targets, declared before the pools."""

    def _statistics(self, n, *, charge_mean, length_mean, seed):
        from src.capability.extensions import sequence_reward as sr

        rng = np.random.default_rng(seed)
        return [sr.SequenceStatistics(
            sequence="", length=int(max(1, rng.normal(length_mean, 25))),
            net_charge=float(rng.normal(charge_mean, 10.0)),
            low_complexity_fraction=0.1, composition_entropy_bits=3.3,
            longest_single_residue_run=4, distinct_5mer_fraction=0.9,
            non_canonical_residues=0) for _ in range(n)]

    def test_the_ladder_is_strictly_harder_and_starts_at_the_declared_window(self):
        from src.capability.extensions import sequence_reward as sr

        assert rp.CHARGE_LADDER[0] == sr.CHARGE_WINDOW
        lows = [low for low, _ in rp.CHARGE_LADDER]
        assert lows == sorted(lows) and len(set(lows)) == len(lows)
        assert len({high - low for low, high in rp.CHARGE_LADDER}) == 1

    def test_the_conjunction_rate_falls_as_the_target_hardens(self):
        statistics = {stream: self._statistics(400, charge_mean=0.0, length_mean=95,
                                               seed=stream)
                      for stream in range(rp.STREAMS)}
        curve = ra.reachability_curve(statistics, ks=[1, 8, 64], draws=200)
        rates = [rung["conjunction_rate"] for rung in curve["rungs"]]
        assert all(rung["status"] == "resolved" for rung in curve["rungs"])
        assert rates[0] > rates[-1]

    def test_the_conjunction_curve_is_the_rejection_sampler_s_own_hit_probability(self):
        """R@k on a 0/1 indicator with base rate p is 1 - (1 - p)^k."""

        statistics = {stream: self._statistics(600, charge_mean=0.0, length_mean=95,
                                               seed=10 + stream)
                      for stream in range(rp.STREAMS)}
        curve = ra.reachability_curve(statistics, ks=[1, 2, 8, 64], draws=200)
        rung = curve["rungs"][0]
        p = rung["conjunction_rate"]
        assert p > 0.0
        for k in (1, 2, 8, 64):
            expected = 1.0 - (1.0 - p) ** k
            assert rung["conjunction_r_at_k"][str(k)]["point"] == pytest.approx(
                expected, abs=2e-3)

    def test_it_carries_percentile_intervals_and_says_they_are_not_the_families_band(self):
        statistics = {stream: self._statistics(200, charge_mean=0.0, length_mean=95,
                                               seed=20 + stream)
                      for stream in range(rp.STREAMS)}
        curve = ra.reachability_curve(statistics, ks=[1, 8], draws=200)
        assert "not a contrast" in curve["method"]
        rung = curve["rungs"][0]
        low, high = rung["r_at_k"]["8"]["interval"]
        assert low <= rung["r_at_k"]["8"]["point"] <= high

    def test_a_stream_below_the_resampling_floor_leaves_a_rung_unresolved(self):
        statistics = {0: self._statistics(200, charge_mean=0.0, length_mean=95, seed=1),
                      1: self._statistics(3, charge_mean=0.0, length_mean=95, seed=2),
                      2: self._statistics(200, charge_mean=0.0, length_mean=95, seed=3)}
        curve = ra.reachability_curve(statistics, ks=[1, 8], draws=100)
        assert all(rung["status"] == "unresolved" for rung in curve["rungs"])
        assert "resampling floor" in curve["rungs"][0]["reason"]

    def test_it_states_the_half_of_the_question_it_cannot_answer(self):
        assert "cannot answer the other half" in rp.LADDER_NOTE
        assert "trained against a higher rung" in rp.LADDER_NOTE


class TestHardRungChoice:
    """Choosing the positive control's target, and refusing to manufacture one."""

    def test_the_bound_is_exact_and_not_a_bootstrap_of_a_zero_count(self):
        """A target the base pool never hits is the case a bootstrap gets wrong.

        The nonparametric bootstrap of an all-zero indicator is a zero-width
        interval at zero: it would assert unreachability with no uncertainty, at
        exactly the point where unreachability is the thing being decided.
        """

        empty = {0: [0.0] * 64, 1: [0.0] * 64, 2: [0.0] * 64}
        drawn = ra.best_of_k_draws(empty, empty, family=rp.family_name(
            rp.FAMILY_BEST_OF_K, rp.ARM_ENVIRONMENT), column=3, ks=[512], draws=200)
        assert float(drawn.min()) == 0.0 and float(drawn.max()) == 0.0
        # the exact bound sees the rare event the sample missed
        assert ra.conjunction_rate_upper_bound(0, 192) > 0.0

    def test_the_upper_bound_is_monotone_and_bounded(self):
        bounds = [ra.conjunction_rate_upper_bound(hits, 3072) for hits in (0, 1, 2, 10)]
        assert bounds == sorted(bounds)
        assert all(0.0 < value < 1.0 for value in bounds)
        assert ra.conjunction_rate_upper_bound(7, 7) == 1.0

    def test_the_bound_tightens_with_the_sample_size(self):
        assert (ra.conjunction_rate_upper_bound(0, 3072)
                < ra.conjunction_rate_upper_bound(0, 512))

    def test_a_malformed_count_is_refused(self):
        with pytest.raises(ValueError, match="0 <= hits <= n"):
            ra.conjunction_rate_upper_bound(5, 3)
        with pytest.raises(ValueError, match="0 <= hits <= n"):
            ra.conjunction_rate_upper_bound(-1, 10)
        with pytest.raises(ValueError, match="positive budget"):
            ra.rejection_hit_probability(0.1, 0)

    def test_the_rejection_hit_probability_is_its_closed_form(self):
        assert ra.rejection_hit_probability(0.0, 512) == 0.0
        assert ra.rejection_hit_probability(1.0, 1) == 1.0
        assert ra.rejection_hit_probability(0.001, 512) == pytest.approx(
            1 - 0.999 ** 512)

    def _rungs(self, rates):
        return [{"status": "resolved", "window": [8.0 + 6 * index, 12.0 + 6 * index],
                 "conjunction_rate": rate} for index, rate in enumerate(rates)]

    def test_the_first_unreachable_rung_is_chosen(self):
        """One observed hit in 3072 is still reachable: its bound reaches 0.55.

        That is the bound doing its job rather than a threshold being generous.
        A single observation cannot rule out a rate a 512-sample sampler would
        find more often than not, and the rung chosen as a positive control has
        to be one the sampler misses.
        """

        choice = ra.choose_hard_rung(
            self._rungs([0.02, 0.004, 1 / 3072, 0.0, 0.0]), n=3072)
        assert choice["chosen_window"] == [26.0, 30.0]
        reached = [rung["unreachable"] for rung in choice["rungs_examined"]]
        assert reached == [False, False, False, True, True]
        one_hit = choice["rungs_examined"][2]
        assert one_hit["conjunction_hits"] == 1
        assert one_hit["hit_probability_upper_bound_at_k512"] > 0.5

    def test_the_declared_target_itself_may_be_the_easy_end(self):
        choice = ra.choose_hard_rung(self._rungs([0.0039, 0.0, 0.0, 0.0, 0.0]), n=3072)
        assert choice["rungs_examined"][0]["unreachable"] is False
        assert choice["chosen_window"] == [14.0, 18.0]

    def test_a_ladder_with_no_unreachable_rung_is_reported_rather_than_extended(self):
        choice = ra.choose_hard_rung(self._rungs([0.05, 0.04, 0.03, 0.02, 0.01]), n=3072)
        assert choice["chosen_window"] is None
        assert "The ladder is not extended" in choice["reason"] \
            or "ladder is not extended" in choice["reason"]
        assert "cannot support a positive control" in choice["reason"]

    def test_an_unresolved_rung_is_carried_rather_than_skipped_silently(self):
        rungs = self._rungs([0.02, 0.0, 0.0, 0.0, 0.0])
        rungs[1] = {"status": "unresolved", "window": [14.0, 18.0],
                    "reason": "below the resampling floor on stream(s) ['1']"}
        choice = ra.choose_hard_rung(rungs, n=3072)
        assert choice["rungs_examined"][1]["status"] == "unresolved"
        assert choice["chosen_window"] == [20.0, 24.0]

    def test_the_declared_constants_are_what_the_rule_names(self):
        assert rp.RUNG_BUDGET == rp.K_GRID[-1]
        assert 0.0 < rp.RUNG_UNREACHABLE_BOUND < 1.0
        assert "Clopper-Pearson" in rp.RUNG_RULE
        assert "not extended" in rp.RUNG_RULE


class TestHardArmDeclaration:
    """The fourth arm, and the refusals that keep its target from being chosen late."""

    def test_the_hard_arm_is_declared_and_needs_no_structure(self):
        from src.capability.extensions import sequence_reward as sr

        assert rp.ARM_ENVIRONMENT_HARD in rp.TRAINED_ARMS
        arm = rp.arm(rp.ARM_ENVIRONMENT_HARD)
        assert arm.needs_structure is False
        assert sr.REWARD_NEEDS_DECLARED_WINDOW[sr.REWARD_ENVIRONMENT_HARD] is True

    def test_every_family_is_the_contrast_pools_crossed_with_its_own_axis(self):
        """Five contrast pools, times whatever second axis the family declares."""

        assert len(rp.CONTRAST_POOLS) == 5
        assert rp.POOL_BASE not in rp.CONTRAST_POOLS
        for family, size in rp.FAMILY_SIZES.items():
            if family.startswith(rp.FAMILY_BEST_OF_K):
                width = len(rp.K_GRID)
            elif family.endswith("_by_length") or "_by_length__" in family:
                width = len(rp.LENGTH_STRATUM_NAMES)
            elif family == rp.FAMILY_CROSS_NLL:
                width = 2
            else:
                width = 1
            assert size == len(rp.CONTRAST_POOLS) * width, family

    def test_the_hard_reward_refuses_until_its_window_is_declared(self):
        from src.capability.extensions import sequence_reward as sr

        stats = sr.statistics("ACDEFGHIKLMNPQRSTVWY" * 4)
        if sr.hard_window_declared():
            pytest.skip("the window has been chosen; the refusal path is behind it")
        with pytest.raises(ValueError, match="has not been chosen"):
            sr.reward(sr.REWARD_ENVIRONMENT_HARD, stats)
        with pytest.raises(ValueError, match="has not been chosen"):
            sr.require_hard_window()
        # and it is omitted from the computable set rather than raising there
        assert sr.REWARD_ENVIRONMENT_HARD not in sr.rewards(stats, mean_ca_plddt=50.0)

    def test_the_hard_reward_is_the_environment_arithmetic_at_another_window(self):
        from src.capability.extensions import sequence_reward as sr

        stats = sr.statistics("K" * 20 + "ACDEFGHILMNPQSTVWY" * 4)
        declared = sr.environment_reward_at(stats, charge_window=sr.CHARGE_WINDOW)
        assert declared == pytest.approx(sr.reward(sr.REWARD_ENVIRONMENT, stats))
        harder = sr.environment_reward_at(stats, charge_window=(32.0, 36.0))
        assert 0.0 <= harder <= 1.0

    def test_the_hard_window_must_be_a_rung_of_the_declared_ladder(self):
        """Not enforced by the type system, so it is asserted where it is read."""

        from src.capability.extensions import sequence_reward as sr

        if sr.hard_window_declared():
            assert tuple(sr.HARD_CHARGE_WINDOW) in rp.CHARGE_LADDER


class TestNotMeasuredIsNotANull:
    """An absent pool must not be reported as a failed optimisation."""

    def _reading(self, optimised):
        return ra.reading(
            optimised=optimised,
            k={"k_star": None, "redistribution": False, "reason": "fixture"},
            support={"share_above": 0.5, "null_share": 0.01},
            degeneracy={"breached": False, "breaches": {}},
        )

    def test_an_unmeasured_gain_is_unresolved_and_says_why(self):
        got = self._reading(ra.optimisation_finding(None))
        assert got["reading"] == rp.READING_UNRESOLVED
        assert "not measured" in got["reason"]

    def test_a_measured_gain_that_did_not_move_is_a_failed_optimisation(self):
        got = self._reading(ra.optimisation_finding([-0.02, 0.05]))
        assert got["reading"] == rp.READING_NO_OPTIMISATION

    def test_a_measured_gain_that_moved_the_wrong_way_is_also_one(self):
        got = self._reading(ra.optimisation_finding([-0.3, -0.1]))
        assert got["reading"] == rp.READING_NO_OPTIMISATION
        assert "wrong way" in got["reason"]


class TestBestOfKUsesTheDeclaredWeighting:
    """R@1 must be the declared pool mean, or k = 1 is not the no-selection baseline."""

    def _streams(self):
        # unequal realised sizes, which unequal empty-continuation counts give
        return {0: [0.2] * 100, 1: [0.9] * 10, 2: [0.5] * 40}

    def test_stream_equal_probabilities_sum_to_one_and_equalise_the_streams(self):
        probability = ra.stream_equal_member_probability([100, 10, 40])
        assert float(probability.sum()) == pytest.approx(1.0, abs=1e-12)
        assert float(probability[:100].sum()) == pytest.approx(1 / 3, abs=1e-12)
        assert float(probability[100:110].sum()) == pytest.approx(1 / 3, abs=1e-12)

    def test_equal_stream_sizes_give_the_uniform_weighting_back(self):
        probability = ra.stream_equal_member_probability([32, 32, 32])
        assert np.allclose(probability, 1 / 96)

    def test_r_at_one_equals_the_declared_stream_equal_pool_mean(self):
        streams = self._streams()
        order = sorted(streams)
        pooled = np.concatenate([np.asarray(streams[s], dtype=float) for s in order])
        probability = ra.stream_equal_member_probability([len(streams[s]) for s in order])
        assert ra.best_of_k(pooled, pooled, [1], probability)[0] == pytest.approx(
            ra.pool_mean(streams), abs=1e-12)

    def test_the_unweighted_pooling_it_replaced_does_not_satisfy_that(self):
        """Recording the defect, so a future change back would fail here."""

        streams = self._streams()
        pooled = np.concatenate([np.asarray(streams[s], dtype=float)
                                 for s in sorted(streams)])
        assert ra.best_of_k(pooled, pooled, [1])[0] != pytest.approx(
            ra.pool_mean(streams), abs=1e-6)

    def test_the_weighted_curve_still_rises_to_the_pool_maximum(self):
        streams = self._streams()
        order = sorted(streams)
        pooled = np.concatenate([np.asarray(streams[s], dtype=float) for s in order])
        probability = ra.stream_equal_member_probability([len(streams[s]) for s in order])
        curve = ra.best_of_k(pooled, pooled, [1, 2, 8, 64, 512], probability)
        assert list(curve) == sorted(curve)
        assert curve[-1] == pytest.approx(float(pooled.max()), abs=1e-3)

    def test_the_weighted_formula_matches_brute_force(self):
        """Enumerated over every k-tuple, with the draw probabilities applied."""

        import itertools

        values = np.array([0.0, 1.0, 2.0, 3.0])
        probability = ra.stream_equal_member_probability([1, 3])
        for k in (1, 2, 3):
            total = 0.0
            for combination in itertools.product(range(4), repeat=k):
                weight = float(np.prod([probability[i] for i in combination]))
                total += weight * max(values[i] for i in combination)
            assert ra.best_of_k(values, values, [k], probability)[0] == pytest.approx(
                total, abs=1e-12)

    def test_a_malformed_probability_vector_is_refused(self):
        values = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="one draw probability per pool member"):
            ra.best_of_k(values, values, [1], np.array([0.5, 0.5]))
        with pytest.raises(ValueError, match="sum to one"):
            ra.best_of_k(values, values, [1], np.array([0.2, 0.2, 0.2]))
        with pytest.raises(ValueError, match="non-empty size per stream"):
            ra.stream_equal_member_probability([10, 0])

    def test_the_draws_carry_the_same_weighting_as_the_point(self):
        streams = self._streams()
        family = rp.family_name(rp.FAMILY_BEST_OF_K, rp.ARM_ENVIRONMENT)
        drawn = ra.best_of_k_draws(streams, streams, family=family, column=9,
                                   ks=[1], draws=500)
        assert float(drawn[:, 0].mean()) == pytest.approx(ra.pool_mean(streams), abs=0.01)


class TestLengthStratification:
    """The length confound, and the device that withholds a verdict it could fake.

    Mean CA-pLDDT falls with length across the retained structural support -- 69.2
    at 16-23 residues against 51.3 at 72-95, priced from 93,049 of this project's
    own readings -- so a policy that shortens gains reward without anything having
    been installed. None of the four degeneracy bounds reads length, so the
    stratified contrast is what stands between that and an installed-bias verdict.
    """

    def test_the_edges_are_declared_and_anchored_to_the_support_and_the_budget(self):
        from src.capability.extensions import sequence_reward as sr

        edges = rp.LENGTH_STRATUM_EDGES
        assert list(edges) == sorted(edges) and len(set(edges)) == len(edges)
        assert edges[0] == sr.STRUCTURE_SUPPORT[0]
        assert edges[-1] == rp.MAX_NEW_TOKENS + 1

    def test_the_strata_widen_from_the_short_end_and_bracket_the_minimum(self):
        """The dependence is U-shaped with a minimum near 72-100 residues.

        The edges are finer at the short end, where the fall from 16 to 32
        residues is about nine pLDDT points, and they bracket the turn at 64-95
        and 96-127. Stratification is agnostic to the sign of the dependence, so
        the edges do the work on the rising limb as well -- which is the point of
        the device for a non-monotone confound, since no linear control can
        remove a U.
        """

        widths = [high - low for low, high in zip(rp.LENGTH_STRATUM_EDGES[:-1],
                                                  rp.LENGTH_STRATUM_EDGES[1:])]
        assert widths == sorted(widths)
        assert widths[0] < widths[-1]
        assert "64-95" in rp.LENGTH_STRATUM_NAMES
        assert "96-127" in rp.LENGTH_STRATUM_NAMES

    def test_nothing_below_the_structural_support_reaches_a_stratum(self):
        from src.capability.extensions import sequence_reward as sr

        for length in (0, 1, 8, sr.STRUCTURE_SUPPORT[0] - 1):
            assert rp.length_stratum(length) is None
        assert rp.length_stratum(sr.STRUCTURE_SUPPORT[0]) == rp.LENGTH_STRATUM_NAMES[0]

    def test_a_completion_past_the_budget_reaches_none(self):
        assert rp.length_stratum(rp.MAX_NEW_TOKENS) == rp.LENGTH_STRATUM_NAMES[-1]
        assert rp.length_stratum(rp.MAX_NEW_TOKENS + 1) is None

    def test_the_confidence_rate_is_stratified_because_a_threshold_amplifies(self):
        """The rate is more length-driven than the mean it comes from, not less.

        The recorded mean at 16-23 residues is 69.2, essentially on the 70-point
        criterion, so about half of that stratum clears it from length alone; at
        72-95 the mean is 51.3 and almost nothing does. A threshold inside a
        location shift amplifies it. The rate is also the quotable form, and a
        caveat in a declaration does not travel the way a stratified number does.
        """

        family = rp.FAMILY_PLDDT_CRITERION_RATE_BY_LENGTH
        assert rp.FAMILY_SIZES[family] == (len(rp.CONTRAST_POOLS)
                                           * len(rp.LENGTH_STRATUM_NAMES))
        assert "amplifies" in rp.CRITERION_RATE_AMPLIFICATION
        assert "69.2" in rp.CRITERION_RATE_AMPLIFICATION
        assert "51.3" in rp.CRITERION_RATE_AMPLIFICATION

    def test_the_stratified_diagnostics_are_not_verdict_inputs(self):
        """They say what the length shift did; the gate reads the gain alone."""

        assert set(rp.STRATIFIED_DIAGNOSTIC_FAMILIES) == {
            rp.FAMILY_PLDDT_BY_LENGTH, rp.FAMILY_PLDDT_CRITERION_RATE_BY_LENGTH}
        gate = {rp.family_name(rp.FAMILY_REWARD_GAIN_BY_LENGTH, reward)
                for reward in rp.structural_rewards()}
        assert not gate & set(rp.STRATIFIED_DIAGNOSTIC_FAMILIES)

    def test_four_families_carry_the_stratification(self):
        stratified = [name for name in rp.FAMILY_SIZES
                      if name.endswith("_by_length") or "_by_length__" in name]
        assert len(stratified) == 4
        assert all(rp.FAMILY_SIZES[name] == len(rp.CONTRAST_POOLS)
                   * len(rp.LENGTH_STRATUM_NAMES) for name in stratified)

    def test_only_the_structural_rewards_are_stratified(self):
        from src.capability.extensions import sequence_reward as sr

        assert set(rp.structural_rewards()) == {
            name for name in sr.REWARD_NAMES if sr.REWARD_NEEDS_STRUCTURE[name]}
        assert sr.REWARD_ENVIRONMENT not in rp.structural_rewards()
        for reward in rp.structural_rewards():
            family = rp.family_name(rp.FAMILY_REWARD_GAIN_BY_LENGTH, reward)
            assert rp.FAMILY_SIZES[family] == (len(rp.CONTRAST_POOLS)
                                               * len(rp.LENGTH_STRATUM_NAMES))

    def _columns(self, **per_stratum):
        out = {}
        for stratum in rp.LENGTH_STRATUM_NAMES:
            key = f"memory|length={stratum}"
            entry = per_stratum.get(stratum)
            out[key] = (entry if entry is not None
                        else {"status": "unresolved", "interval": None,
                              "reason": "below the resampling floor"})
        return out

    def test_a_gain_that_holds_at_matched_length_survives(self):
        got = ra.stratified_gain(
            self._columns(**{"160-200": {"status": "resolved", "point": 0.08,
                                         "interval": [0.02, 0.14]}}), pool="memory")
        assert got["survives"] is True
        assert got["survived_in"] == ["160-200"]
        assert "bounded above zero at matched length" in got["reason"]

    def test_a_gain_that_vanishes_at_matched_length_does_not(self):
        got = ra.stratified_gain(
            self._columns(**{"160-200": {"status": "resolved", "point": 0.01,
                                         "interval": [-0.03, 0.05]}}), pool="memory")
        assert got["survives"] is False
        assert "does not survive being read at matched length" in got["reason"]

    def test_non_overlapping_length_distributions_say_so_rather_than_passing(self):
        """The case a policy that shortened out of the base's range produces."""

        got = ra.stratified_gain(self._columns(), pool="memory")
        assert got["survives"] is False
        assert got["unresolved_in"] == list(rp.LENGTH_STRATUM_NAMES)
        assert "do not overlap in length" in got["reason"]

    def test_a_negative_stratum_is_recorded_as_negative_not_as_unresolved(self):
        got = ra.stratified_gain(
            self._columns(**{"64-95": {"status": "resolved", "point": -0.1,
                                       "interval": [-0.2, -0.02]}}), pool="memory")
        assert got["negative_in"] == ["64-95"]
        assert got["survives"] is False


class TestStratifiedVerdictGate:
    def _reading(self, stratified):
        return ra.reading(
            optimised={"optimised": True, "reason": "fixture"},
            k={"k_star": 256, "redistribution": False, "reason": "fixture"},
            support={"share_above": 0.4, "null_share": 0.01},
            degeneracy={"breached": False, "breaches": {}},
            stratified=stratified,
        )

    def test_an_installed_bias_needs_the_gain_to_survive_stratification(self):
        survived = self._reading({"survives": True, "reason": "holds at 128-200"})
        assert survived["reading"] == rp.READING_BIAS_ON_MANIFOLD
        withheld = self._reading({"survives": False, "reason": "no stratum carries it"})
        assert withheld["reading"] == rp.READING_UNRESOLVED
        assert "does not survive length stratification" in withheld["reason"]

    def test_the_gate_runs_before_the_redistribution_reading_too(self):
        """Neither reading is taken on a gain that may be a length artifact."""

        got = ra.reading(
            optimised={"optimised": True, "reason": "fixture"},
            k={"k_star": 8, "redistribution": True, "reason": "fixture"},
            support={"share_above": 0.0, "null_share": 0.01},
            degeneracy={"breached": False, "breaches": {}},
            stratified={"survives": False, "reason": "no stratum carries it"},
        )
        assert got["reading"] == rp.READING_UNRESOLVED

    def test_the_closed_form_arms_pass_no_stratification_and_are_unaffected(self):
        got = ra.reading(
            optimised={"optimised": True, "reason": "fixture"},
            k={"k_star": 8, "redistribution": True, "reason": "fixture"},
            support={"share_above": 0.0, "null_share": 0.01},
            degeneracy={"breached": False, "breaches": {}},
        )
        assert got["reading"] == rp.READING_REDISTRIBUTION

    def test_a_failed_optimisation_is_still_read_before_stratification(self):
        got = ra.reading(
            optimised={"optimised": False, "reason": "interval contains zero"},
            k={"k_star": None, "redistribution": False, "reason": "fixture"},
            support={"share_above": 0.0, "null_share": 0.01},
            degeneracy={"breached": False, "breaches": {}},
            stratified={"survives": False, "reason": "fixture"},
        )
        assert got["reading"] == rp.READING_NO_OPTIMISATION


class TestTheSecondRungRule:
    """Two rules on the record: what the first selected, and why it was voided."""

    def test_the_first_rule_s_selection_is_kept(self):
        assert tuple(rp.RUNG_RULE_FIRST_SELECTION) == rp.CHARGE_LADDER[0]
        assert "degenerate by construction" in rp.RUNG_RULE_VOIDED
        assert "voidable because" in rp.RUNG_RULE_VOIDED

    def test_the_ground_for_voiding_is_the_duplicate_arm_and_not_the_number(self):
        """The only ground that costs the pre-registration nothing."""

        assert "duplicate" in rp.RUNG_RULE_VOIDED
        assert "answers no question" in rp.RUNG_RULE_VOIDED
        assert "without reference to any trained pool" in rp.RUNG_RULE_VOIDED

    def test_the_second_rule_is_posed_on_a_quantity_that_discriminates(self):
        assert rp.RUNG_SHAPED_BOUND == 0.70
        assert rp.RUNG_SHAPED_BUDGET == rp.K_GRID[-1]
        assert "shaped" in rp.RUNG_SHAPED_RULE
        assert "attainable at reward exactly 1.0" in rp.RUNG_SHAPED_RULE

    def _ladder(self, values):
        return [{"status": "resolved", "window": [8.0 + 6 * i, 12.0 + 6 * i],
                 "r_at_k": {"512": {"point": v}}} for i, v in enumerate(values)]

    def test_it_takes_the_first_rung_below_the_bound(self):
        got = ra.choose_shaped_rung(self._ladder([0.97, 0.86, 0.75, 0.74, 0.62]))
        assert got["chosen_window"] == [32.0, 36.0]

    def test_a_ladder_entirely_above_the_bound_is_reported_not_extended(self):
        got = ra.choose_shaped_rung(self._ladder([0.97, 0.96, 0.95, 0.94, 0.93]))
        assert got["chosen_window"] is None
        assert "the ladder is not extended" in got["reason"]

    def test_the_declared_window_is_what_the_rule_selects_from_the_measured_ladder(self):
        """The constant and the rule must not be able to disagree."""

        import json

        from src.capability.extensions import sequence_reward as sr

        path = (Path(__file__).resolve().parents[2]
                / "results/extensions/rl_environment_provenance_20261114/pools"
                / "base_reachability_ladder.json")
        if not path.is_file():
            pytest.skip("the base pool's ladder has not been written on this host")
        ladder = json.loads(path.read_text(encoding="utf-8"))
        chosen = ra.choose_shaped_rung(ladder["rungs"])["chosen_window"]
        assert chosen == list(sr.require_hard_window())

    def test_the_chosen_rung_is_attainable_at_reward_one(self):
        """Hard, not impossible: a witness constructed from the declaration itself."""

        from src.capability.extensions import sequence_reward as sr

        window = sr.require_hard_window()
        backbone = "ACDEFGHILMNPQRSTVWY"
        for length in (100, 110, 120):
            for count in range(20, 50):
                residues = [backbone[index % len(backbone)] for index in range(length)]
                for index in range(count):
                    residues[(index * length) // count] = "K"
                stats = sr.statistics("".join(residues))
                if sr.environment_reward_at(stats, charge_window=window) == 1.0:
                    assert stats.longest_single_residue_run <= 2
                    assert stats.low_complexity_fraction < sr.LOW_COMPLEXITY_LIMIT
                    return
        raise AssertionError("no witness attains the chosen rung at reward one")


class TestObjectiveAttainment:
    """A reward gain is not evidence that the objective was attained."""

    def test_every_reward_has_a_family_carrying_its_own_attainment_rate(self):
        from src.capability.extensions import sequence_reward as sr

        assert set(rp.OBJECTIVE_ATTAINMENT_FAMILY) == set(sr.REWARD_NAMES)
        for family in rp.OBJECTIVE_ATTAINMENT_FAMILY.values():
            assert family in rp.FAMILY_SIZES

    def test_a_rate_bounded_above_the_base_is_attainment(self):
        got = ra.objective_attainment({"status": "resolved", "interval": [0.1, 0.3]},
                                      {"status": "resolved", "interval": [0.02, 0.08]})
        assert got["state"] == ra.ATTAINED

    def test_a_rising_reward_with_a_flat_rate_is_approaching_without_attaining(self):
        got = ra.objective_attainment({"status": "resolved", "interval": [0.1, 0.3]},
                                      {"status": "resolved", "interval": [-0.01, 0.01]})
        assert got["state"] == ra.APPROACHED
        assert "approach its objective's constraints without meeting them" in got["reason"]

    def test_neither_moving_is_also_the_approached_state_with_its_own_reason(self):
        got = ra.objective_attainment({"status": "resolved", "interval": [-0.05, 0.05]},
                                      {"status": "resolved", "interval": [-0.01, 0.01]})
        assert got["state"] == ra.APPROACHED
        assert "neither the shaped reward nor the attainment rate" in got["reason"]

    def test_an_unresolved_rate_is_not_measured_rather_than_a_shortfall(self):
        got = ra.objective_attainment({"status": "resolved", "interval": [0.1, 0.3]},
                                      {"status": "unresolved", "interval": None})
        assert got["state"] == ra.ATTAINMENT_UNMEASURED
        assert got["state"] in ra.ATTAINMENT_STATES
        got = ra.objective_attainment(None, None)
        assert got["state"] == ra.ATTAINMENT_UNMEASURED

    def test_the_note_carries_the_base_pool_s_own_disagreement(self):
        assert "0.9963" in rp.ATTAINMENT_NOTE
        assert "3,072" in rp.ATTAINMENT_NOTE


class TestTheTopStratumSplit:
    def test_the_split_at_160_is_declared(self):
        assert 160 in rp.LENGTH_STRATUM_EDGES
        assert "128-159" in rp.LENGTH_STRATUM_NAMES
        assert "160-200" in rp.LENGTH_STRATUM_NAMES

    def test_no_stratum_now_spans_more_than_the_split_width(self):
        widths = [high - low for low, high in zip(rp.LENGTH_STRATUM_EDGES[:-1],
                                                  rp.LENGTH_STRATUM_EDGES[1:])]
        assert max(widths) <= 41

    def test_the_stratified_families_widened_with_it(self):
        for family, size in rp.FAMILY_SIZES.items():
            if family.endswith("_by_length") or "_by_length__" in family:
                assert size == len(rp.CONTRAST_POOLS) * len(rp.LENGTH_STRATUM_NAMES)
