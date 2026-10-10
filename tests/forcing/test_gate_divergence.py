"""The endpoint's null, and the properties that make it a null rather than a hope.

The design claims the double difference is exactly zero when forcing the foreign
residue does nothing position-specific. That claim is testable without a model:
build cells whose distributions have the stated structure and check the arithmetic
lands on zero. Three cases matter and all three are here.

*Nothing changes.* The two conditions give the same distribution everywhere, so
every divergence is zero and so is the difference.

*Everything changes equally.* Forcing the foreign residue shifts the completion's
whole composition. Both terms move by the same amount and cancel. This is the case
the second term exists for, and it is exact rather than approximate.

*Only the partner changes.* Then the endpoint is the divergence at the partner,
with nothing subtracted but the controls' zero.

Beyond the null: the sampling unit must be the backbone and never the draw,
near-identical completions must count once, censored draws must not be silently
imputed, and the verdict must distinguish an effect below the pre-registered
threshold from a bounded null from an unresolved interval.
"""

import numpy as np
import pytest

from src.capability.core.amino_acids import AA20
from src.capability.forcing import forcing_design as D
from src.capability.forcing import gate_divergence as G


def _distinct_completion(draw: int, *, length: int = 40) -> str:
    """A completion no other draw's shingles are contained in.

    Real completions of forty residues are essentially all distinct -- the
    admitted cohort's own sequences all score a distinct-5-mer fraction of 1.0 --
    so a fixture whose draws differ only in a trailing run of alanines would be
    collapsed to two groups by the near-duplicate screen and would exercise the
    deduplication path rather than the estimator. Deduplication has its own tests.
    """

    rng = np.random.default_rng(10_000 + draw)
    return "".join(AA20[index] for index in rng.integers(len(AA20), size=length))


def cell(
    *, native_residues, transplant_residues, unit_id="P1:40:70", accession="P1",
    partner=70, references=(71, 72, 73), native_rows=None, transplant_rows=None,
):
    """A cell whose realised residues at each read position are given directly.

    ``native_residues`` maps a position to the list of residues that condition's
    draws realised there. The per-draw distributions are one-hot on the realised
    residue, which makes the mixture estimator agree with the empirical one and
    lets one fixture exercise both.
    """

    positions = [partner, *references]
    column = {residue: index for index, residue in enumerate(AA20)}

    def side(table, rows):
        draws = len(next(iter(table.values())))
        distributions = np.zeros((draws, len(positions), len(AA20)), dtype=np.float32)
        records = []
        for draw in range(draws):
            realised = {str(position): table[position][draw] for position in positions}
            for slot, position in enumerate(positions):
                residue = table[position][draw]
                if residue is not None:
                    distributions[draw, slot, column[residue]] = 1.0
            records.append(
                {
                    "draw": draw,
                    "completion": _distinct_completion(draw),
                    "censored": any(value is None for value in realised.values()),
                    "realised": realised,
                    "hydrophobic_fraction": 0.1 + 0.8 * (draw % 2),
                    "repeat_flagged": False,
                }
            )
        if rows is not None:
            for record, override in zip(records, rows):
                record.update(override)
        return {"draws": records, "distributions": distributions}

    return {
        "unit_id": unit_id,
        "accession": accession,
        "length_band": "short",
        "partner": partner,
        "reference_positions": list(references),
        "native": side(native_residues, native_rows),
        "transplant": side(transplant_residues, transplant_rows),
    }


def uniform(positions, residues, draws):
    """The same residue cycle at every position, so no position is special."""

    return {
        position: [residues[index % len(residues)] for index in range(draws)]
        for position in positions
    }


class TestJensenShannon:
    def test_identical_distributions_have_zero_divergence(self):
        first = np.array([0.5, 0.3, 0.2])
        assert G.jensen_shannon(first, first) == 0.0

    def test_disjoint_supports_reach_the_upper_bound(self):
        assert G.jensen_shannon(
            np.array([1.0, 0.0]), np.array([0.0, 1.0])
        ) == pytest.approx(np.log(2.0))

    def test_the_divergence_is_symmetric(self):
        first, second = np.array([0.7, 0.2, 0.1]), np.array([0.1, 0.1, 0.8])
        assert G.jensen_shannon(first, second) == pytest.approx(
            G.jensen_shannon(second, first)
        )

    def test_a_zero_mass_residue_stays_finite(self):
        """Nucleus truncation guarantees zeros; a KL contrast would be infinite."""

        value = G.jensen_shannon(np.array([1.0, 0.0, 0.0]), np.array([0.4, 0.6, 0.0]))
        assert np.isfinite(value) and value > 0

    def test_a_row_that_is_not_a_distribution_is_refused(self):
        with pytest.raises(ValueError, match="sums to one"):
            G.jensen_shannon(np.array([0.5, 0.2]), np.array([0.5, 0.5]))
        with pytest.raises(ValueError, match="non-negative"):
            G.jensen_shannon(np.array([1.5, -0.5]), np.array([0.5, 0.5]))

    def test_stored_float32_rows_are_renormalised_rather_than_tolerated(self):
        rows = (np.ones((2, 20), dtype=np.float32) / 20.0)
        renormalised = G.renormalise(rows)
        assert np.allclose(renormalised.sum(axis=-1), 1.0, atol=1e-15)
        G.jensen_shannon(renormalised[0], renormalised[1])

    def test_an_empty_mass_row_is_refused(self):
        with pytest.raises(ValueError, match="no mass"):
            G.renormalise(np.zeros((1, 20)))


class TestTheNull:
    positions = (70, 71, 72, 73)

    def test_the_endpoint_is_exactly_zero_when_nothing_changes(self):
        table = uniform(self.positions, "ACDE", 24)
        record = G.cell_endpoint(
            cell(native_residues=table, transplant_residues=table), deduplicate=False,
        )
        assert record["partner_divergence_nats"] == 0.0
        assert record["endpoint_nats"] == 0.0

    def test_the_endpoint_is_exactly_zero_under_a_global_composition_shift(self):
        """The case the second term exists for: forcing the foreign residue moves
        the whole completion's composition, both terms move together, and the
        difference cancels exactly rather than approximately."""

        native = uniform(self.positions, "ACDE", 24)
        shifted = uniform(self.positions, "KLMN", 24)
        record = G.cell_endpoint(
            cell(native_residues=native, transplant_residues=shifted), deduplicate=False,
        )
        assert record["partner_divergence_nats"] == pytest.approx(np.log(2.0))
        assert record["reference_divergence_nats"] == pytest.approx(np.log(2.0))
        assert record["endpoint_nats"] == pytest.approx(0.0, abs=1e-12)

    def test_the_endpoint_isolates_a_change_at_the_partner_alone(self):
        native = uniform(self.positions, "ACDE", 24)
        transplant = dict(native)
        transplant[70] = ["K"] * 24
        record = G.cell_endpoint(
            cell(native_residues=native, transplant_residues=transplant), deduplicate=False,
        )
        assert record["reference_divergence_nats"] == 0.0
        assert record["endpoint_nats"] == pytest.approx(record["partner_divergence_nats"])
        assert record["endpoint_nats"] > 0

    def test_a_change_at_the_controls_alone_drives_the_endpoint_negative(self):
        """The endpoint is signed: a design that could only go up would not have a
        two-sided null."""

        native = uniform(self.positions, "ACDE", 24)
        transplant = dict(native)
        for position in (71, 72, 73):
            transplant[position] = ["K"] * 24
        record = G.cell_endpoint(
            cell(native_residues=native, transplant_residues=transplant), deduplicate=False,
        )
        assert record["endpoint_nats"] < 0

    def test_both_estimators_agree_on_one_hot_draws(self):
        """The mixture of one-hot per-draw rows is the empirical histogram, so the
        two estimators must coincide exactly on this fixture."""

        native = uniform(self.positions, "ACDE", 24)
        transplant = dict(native)
        transplant[70] = ["K"] * 24
        fixture = cell(native_residues=native, transplant_residues=transplant)
        empirical = G.cell_endpoint(fixture, estimator="empirical", deduplicate=False)
        mixture = G.cell_endpoint(fixture, estimator="mixture", deduplicate=False)
        assert mixture["endpoint_nats"] == pytest.approx(empirical["endpoint_nats"])

    def test_an_unknown_estimator_is_refused(self):
        table = uniform(self.positions, "ACDE", 24)
        with pytest.raises(ValueError, match="unknown estimator"):
            G.cell_endpoint(
                cell(native_residues=table, transplant_residues=table), estimator="vibes",
            )


class TestSupportAndCensoring:
    positions = (70, 71, 72, 73)

    def test_a_cell_below_the_draw_floor_has_no_value(self):
        table = uniform(self.positions, "ACDE", G.MIN_DRAWS_PER_CONDITION - 1)
        record = G.cell_endpoint(
            cell(native_residues=table, transplant_residues=table), deduplicate=False,
        )
        assert record["endpoint_nats"] is None
        assert "minimum draws" in record["undefined"]

    def test_a_censored_draw_contributes_no_residue_and_is_not_imputed(self):
        table = uniform(self.positions, "ACDE", 24)
        censored = {position: list(values) for position, values in table.items()}
        for position in self.positions:
            censored[position][:4] = [None] * 4
        record = G.cell_endpoint(
            cell(native_residues=table, transplant_residues=censored), deduplicate=False,
        )
        assert record["divergences"][0]["transplant_draws"] == 20
        assert record["divergences"][0]["native_draws"] == 24

    def test_completed_only_removes_the_censored_draws_from_both_sides(self):
        table = uniform(self.positions, "ACDE", 24)
        censored = {position: list(values) for position, values in table.items()}
        for position in self.positions:
            censored[position][:4] = [None] * 4
        fixture = cell(native_residues=table, transplant_residues=censored)
        selection = G.select_draws(
            fixture["native"]["draws"], fixture["transplant"]["draws"],
            completed_only=True, exclude_repeat_flagged=False, deduplicate=False,
        )
        assert len(selection["kept"][D.CONDITION_TRANSPLANT]) == 20
        assert selection["dropped"]["censored"] == 4

    def test_the_repeat_filter_drops_only_flagged_draws(self):
        table = uniform(self.positions, "ACDE", 24)
        fixture = cell(
            native_residues=table, transplant_residues=table,
            native_rows=[{"repeat_flagged": index < 6} for index in range(24)],
        )
        selection = G.select_draws(
            fixture["native"]["draws"], fixture["transplant"]["draws"],
            completed_only=False, exclude_repeat_flagged=True, deduplicate=False,
        )
        assert len(selection["kept"][D.CONDITION_NATIVE]) == 18
        assert len(selection["kept"][D.CONDITION_TRANSPLANT]) == 24


class TestDeduplication:
    positions = (70, 71, 72, 73)

    def test_near_identical_completions_count_once_per_condition(self):
        table = uniform(self.positions, "ACDE", 24)
        repeated = [{"completion": "ACDEACDEACDEACDEACDE"} for _ in range(24)]
        fixture = cell(
            native_residues=table, transplant_residues=table,
            native_rows=repeated, transplant_rows=repeated,
        )
        selection = G.select_draws(
            fixture["native"]["draws"], fixture["transplant"]["draws"],
            completed_only=False, exclude_repeat_flagged=False, deduplicate=True,
        )
        assert len(selection["kept"][D.CONDITION_NATIVE]) == 1
        assert len(selection["kept"][D.CONDITION_TRANSPLANT]) == 1
        assert selection["dropped"]["duplicate"] == 46

    def test_the_grouping_cannot_see_the_condition_label(self):
        """Grouping each condition separately would let one completion be a
        distinct unit on one side and a duplicate on the other."""

        distinct = [
            "MKTAYIAKQRQISFVKSHFSRQLE", "GGGSSSGGGSSSGGGSSSPPPWWW",
        ]
        groups = G.duplicate_groups(distinct + distinct)
        assert groups[0] == groups[2] and groups[1] == groups[3]
        assert groups[0] != groups[1]

    def test_an_empty_completion_list_yields_an_empty_grouping(self):
        assert G.duplicate_groups([]).size == 0


class TestAggregation:
    def test_a_backbone_with_several_pairs_contributes_one_value(self):
        records = [
            {"accession": "A", "endpoint_nats": 0.2},
            {"accession": "A", "endpoint_nats": 0.4},
            {"accession": "B", "endpoint_nats": 1.0},
        ]
        assert G.backbone_values(records) == {"A": pytest.approx(0.3), "B": 1.0}

    def test_an_unresolved_unit_is_dropped_rather_than_zero_imputed(self):
        records = [
            {"accession": "A", "endpoint_nats": None},
            {"accession": "A", "endpoint_nats": 0.4},
        ]
        assert G.backbone_values(records) == {"A": pytest.approx(0.4)}

    def test_the_interval_resamples_backbones_not_draws(self):
        """Twenty backbones and two hundred draws must not give the same interval
        width; treating draws as units is how a null becomes a finding."""

        values = {f"B{index}": 0.1 * ((-1) ** index) for index in range(20)}
        estimate = G.bootstrap_interval(values, draws=2000, seed=1)
        assert estimate["units"] == 20
        assert estimate["standard_error"] == pytest.approx(
            estimate["unit_sd"] / np.sqrt(20)
        )
        assert "the backbone" in estimate["unit"]

    def test_a_single_backbone_yields_no_interval(self):
        estimate = G.bootstrap_interval({"A": 0.5})
        assert estimate["interval"] is None and estimate["excludes_zero"] is None

    def test_the_interval_is_reproducible(self):
        values = {f"B{index}": float(index) / 20 for index in range(20)}
        first = G.bootstrap_interval(values, draws=500, seed=7)
        second = G.bootstrap_interval(values, draws=500, seed=7)
        assert first["interval"] == second["interval"]

    def test_strata_are_never_pooled(self):
        records = [
            {"accession": "A", "endpoint_nats": 1.0, "length_band": "short"},
            {"accession": "B", "endpoint_nats": 2.0, "length_band": "short"},
            {"accession": "C", "endpoint_nats": 9.0, "length_band": "long"},
            {"accession": "E", "endpoint_nats": 9.0, "length_band": "long"},
        ]
        result = G.by_stratum(records, stratum="length_band", draws=200, seed=1)
        assert set(result) == {"short", "long"}
        assert result["short"]["point"] == pytest.approx(1.5)
        assert result["long"]["point"] == pytest.approx(9.0)

    def test_a_record_without_the_stratum_label_is_left_out(self):
        records = [
            {"accession": "A", "endpoint_nats": 1.0, "coverage_band": None},
            {"accession": "B", "endpoint_nats": 2.0, "coverage_band": "remote"},
            {"accession": "C", "endpoint_nats": 3.0, "coverage_band": "remote"},
        ]
        result = G.by_stratum(records, stratum="coverage_band", draws=200, seed=1)
        assert set(result) == {"remote"} and result["remote"]["units"] == 2


class TestPermutationCalibration:
    positions = (70, 71, 72, 73)

    def _cells(self, count, *, shifted=False):
        cells = []
        for index in range(count):
            native = uniform(self.positions, "ACDEFGHI", 24)
            transplant = (
                uniform(self.positions, "KLMNPQRS", 24) if shifted else dict(native)
            )
            cells.append(
                cell(
                    native_residues=native, transplant_residues=transplant,
                    unit_id=f"B{index}:40:70", accession=f"B{index}",
                )
            )
        return cells

    def test_the_calibration_centres_on_the_estimator_s_residual_bias(self):
        """With no position-specific effect the permuted endpoint has to sit on
        zero, which is what makes the observed value interpretable."""

        cells = self._cells(6)
        null = G.permutation_calibration(cells, 0.0, draws=40, seed=3)
        assert null["draws"] == 40
        assert null["mean_nats"] == pytest.approx(0.0, abs=1e-9)

    def test_a_global_shift_still_permutes_to_zero(self):
        cells = self._cells(6, shifted=True)
        null = G.permutation_calibration(cells, 0.0, draws=40, seed=3)
        assert null["mean_nats"] == pytest.approx(0.0, abs=1e-9)

    def test_the_calibration_reports_its_own_absence(self):
        assert G.permutation_calibration(self._cells(2), None)["draws"] == 0
        thin = [
            cell(
                native_residues=uniform(self.positions, "AC", 4),
                transplant_residues=uniform(self.positions, "AC", 4),
            )
        ]
        assert G.permutation_calibration(thin, 0.1, draws=10)["draws"] == 0

    def test_the_calibration_is_reproducible(self):
        cells = self._cells(4)
        first = G.permutation_calibration(cells, 0.05, draws=20, seed=11)
        second = G.permutation_calibration(cells, 0.05, draws=20, seed=11)
        assert first["two_sided_p"] == second["two_sided_p"]


class TestTeacherForced:
    def _cell(self, native_rows, transplant_rows):
        return {
            "unit_id": "P1:40:70", "accession": "P1", "length_band": "short",
            "partner": 70, "reference_positions": [71, 72, 73],
            "native": {"distributions": np.asarray(native_rows, dtype=np.float32)},
            "transplant": {"distributions": np.asarray(transplant_rows, dtype=np.float32)},
        }

    def _one_hot(self, residues):
        column = {residue: index for index, residue in enumerate(AA20)}
        rows = np.zeros((len(residues), len(AA20)), dtype=np.float32)
        for index, residue in enumerate(residues):
            rows[index, column[residue]] = 1.0
        return rows

    def test_the_exact_endpoint_is_zero_under_a_global_shift(self):
        record = G.teacher_forced_endpoint(
            self._cell(self._one_hot("ACDE"), self._one_hot("KLMN"))
        )
        assert record["exact_nats"] == pytest.approx(0.0, abs=1e-12)

    def test_the_exact_endpoint_isolates_the_partner(self):
        record = G.teacher_forced_endpoint(
            self._cell(self._one_hot("ACDE"), self._one_hot("KCDE"))
        )
        assert record["exact_nats"] == pytest.approx(np.log(2.0))

    def test_the_resampled_companion_shares_the_sampled_estimator_s_support(self):
        """Comparing a bias-free exact value against a biased empirical one would
        make the decomposition an artefact of the estimators."""

        record = G.teacher_forced_endpoint(
            self._cell(self._one_hot("ACDE"), self._one_hot("ACDE")), resample_draws=24,
        )
        assert record["resample_draws"] == 24
        assert record["exact_nats"] == pytest.approx(0.0, abs=1e-12)
        assert record["resampled_nats"] == pytest.approx(0.0, abs=1e-12)

    def test_a_mismatched_conditional_shape_is_refused(self):
        with pytest.raises(ValueError, match="one row per read position"):
            G.teacher_forced_endpoint(
                self._cell(self._one_hot("AC"), self._one_hot("AC"))
            )


class TestSimultaneousBand:
    def _values(self, columns, backbones):
        return {
            column: {f"B{index}": 0.1 * (index % 5) + 0.01 * position
                     for index in range(backbones)}
            for position, column in enumerate(columns)
        }

    def test_the_band_is_wider_than_the_pointwise_interval(self):
        values = self._values(("a", "b", "c", "d"), 30)
        band = G.simultaneous_band(["a", "b", "c", "d"], values, draws=2000, seed=5)
        assert band["critical_value"] > 1.9
        for column in band["columns"]:
            low, high = column["simultaneous_interval_nats"]
            plow, phigh = column["pointwise_interval_nats"]
            assert high - low >= phigh - plow - 1e-9

    def test_the_band_carries_no_out_of_fold_caveat(self):
        values = self._values(("a", "b"), 10)
        band = G.simultaneous_band(["a", "b"], values, draws=500, seed=5)
        assert band["conditional_on_fitted_predictions"] is False

    def test_an_incomplete_table_is_refused_not_patched(self):
        """Arms that did not measure the same panel are not a panel."""

        values = self._values(("a", "b"), 10)
        del values["b"]["B3"]
        with pytest.raises(ValueError, match="did not measure one panel"):
            G.simultaneous_band(["a", "b"], values, draws=100, seed=5)

    def test_one_column_is_not_a_simultaneous_family(self):
        with pytest.raises(ValueError, match="at least two columns"):
            G.simultaneous_band(["a"], self._values(("a",), 10))


class TestVerdict:
    claimable = D.CLAIMABLE["sequence_double_difference"]

    def test_a_resolved_positive_needs_the_interval_the_size_and_the_p_value(self):
        verdict = G.gate_verdict(
            {"point": 0.12, "interval": [0.06, 0.18]}, {"two_sided_p": 0.001},
        )
        assert verdict["verdict"] == "resolved_positive"
        assert verdict["structure_channel_gated"] is True

    def test_an_effect_below_the_pre_registered_size_is_not_the_claim(self):
        verdict = G.gate_verdict(
            {"point": 0.02, "interval": [0.005, 0.035]}, {"two_sided_p": 0.01},
        )
        assert verdict["verdict"] == "resolved_below_claimable"
        assert verdict["structure_channel_gated"] is False

    def test_a_significant_interval_without_calibration_is_not_the_claim(self):
        verdict = G.gate_verdict({"point": 0.12, "interval": [0.06, 0.18]}, {})
        assert verdict["verdict"] == "resolved_below_claimable"

    def test_a_bounded_null_excludes_the_claimable_effect(self):
        verdict = G.gate_verdict(
            {"point": 0.004, "interval": [-0.02, 0.03]}, {"two_sided_p": 0.7},
        )
        assert verdict["verdict"] == "bounded_null"

    def test_a_wide_interval_is_unresolved_and_not_a_null(self):
        verdict = G.gate_verdict(
            {"point": 0.01, "interval": [-0.09, 0.11]}, {"two_sided_p": 0.8},
        )
        assert verdict["verdict"] == "unresolved"
        assert "not a null" in verdict["reason"]

    def test_a_missing_interval_is_unresolved(self):
        assert G.gate_verdict({"point": None, "interval": None}, {})["verdict"] == "unresolved"

    def test_every_verdict_carries_the_non_identifiable_list(self):
        verdict = G.gate_verdict({"point": 0.12, "interval": [0.06, 0.18]}, {"two_sided_p": 0.001})
        assert "stability_and_function" in verdict["non_identifiable"]


class TestHydrophobicBins:
    positions = (70, 71, 72, 73)

    def test_the_bins_split_on_the_cell_s_own_median(self):
        native = uniform(self.positions, "ACDE", 24)
        transplant = dict(native)
        transplant[70] = ["K"] * 24
        record = G.hydrophobic_bin_endpoint(
            cell(native_residues=native, transplant_residues=transplant), minimum=4,
        )
        assert record["resolved_bins"] == 2
        assert record["endpoint_nats"] == pytest.approx(np.log(2.0))

    def test_a_bin_below_the_floor_is_skipped_and_counted(self):
        native = uniform(self.positions, "ACDE", 24)
        record = G.hydrophobic_bin_endpoint(
            cell(native_residues=native, transplant_residues=dict(native)), minimum=50,
        )
        assert record["resolved_bins"] == 0 and record["endpoint_nats"] is None
