"""The frozen design: its sizing arithmetic, its bands, and the claims it asserts.

The pre-registration is the object a reader checks a result against, so the
numbers in it are derived here rather than transcribed, and the two guarantees the
whole design rests on -- that the reciprocal swap equalises the forced-residue
multiset, and that no quantity is pooled across length bands -- are tested as
properties rather than left as prose.
"""

import math

import pytest

from src.capability.forcing import forcing_design as D


class TestSizing:
    def test_reproduces_the_pre_registered_effect_sizes(self):
        sequence = D.minimum_detectable_effect(unit_sd=0.21, units=120)
        structure = D.minimum_detectable_effect(unit_sd=0.15, units=120)
        assert sequence["standard_error"] == pytest.approx(0.0192, abs=5e-5)
        assert sequence["minimum_detectable_effect"] == pytest.approx(0.054, abs=1e-3)
        assert structure["standard_error"] == pytest.approx(0.0137, abs=5e-5)
        assert structure["minimum_detectable_effect"] == pytest.approx(0.0384, abs=1e-3)

    def test_the_declared_claimable_effect_is_not_below_the_computed_one(self):
        """A pre-registered threshold below the MDE would promise power it lacks."""

        for channel, sd in D.ASSUMED_UNIT_SD.items():
            mde = D.minimum_detectable_effect(unit_sd=sd, units=120)
            assert D.CLAIMABLE[channel] >= mde["minimum_detectable_effect"] - 1e-9

    def test_the_standard_error_falls_as_the_root_of_the_unit_count(self):
        few = D.minimum_detectable_effect(unit_sd=0.2, units=25)
        many = D.minimum_detectable_effect(unit_sd=0.2, units=100)
        assert few["standard_error"] == pytest.approx(2.0 * many["standard_error"])

    def test_a_degenerate_request_is_refused(self):
        with pytest.raises(ValueError):
            D.minimum_detectable_effect(unit_sd=0.0, units=120)
        with pytest.raises(ValueError):
            D.minimum_detectable_effect(unit_sd=0.2, units=1)

    def test_the_normal_quantile_matches_the_textbook_values(self):
        assert D._normal_quantile(0.975) == pytest.approx(1.959964, abs=1e-5)
        assert D._normal_quantile(0.80) == pytest.approx(0.841621, abs=1e-5)
        assert D._normal_quantile(0.5) == pytest.approx(0.0, abs=1e-9)

    def test_the_pre_registration_reports_the_realised_unit_count(self):
        """A cohort that lands short must publish the effect size it has power for."""

        short = D.pre_registration(units=80)
        full = D.pre_registration(units=120)
        assert short["units"] == 80
        assert (
            short["sizing"]["sequence_double_difference"]["minimum_detectable_effect"]
            > full["sizing"]["sequence_double_difference"]["minimum_detectable_effect"]
        )


class TestBands:
    def test_length_bands_are_disjoint_and_closed(self):
        assert D.length_band(120) == "short"
        assert D.length_band(110) == "short"
        assert D.length_band(130) == "short"
        assert D.length_band(131) is None
        assert D.length_band(240) == "long"
        assert D.length_band(180) is None
        assert D.band_bounds("short") == (110, 130)
        with pytest.raises(KeyError):
            D.band_bounds("medium")

    def test_the_two_bands_cannot_overlap(self):
        low, high = D.band_bounds("short")
        other_low, _ = D.band_bounds("long")
        assert high < other_low

    def test_burial_bands_split_on_the_declared_boundary(self):
        assert D.rsa_band(0.0) == "buried"
        assert D.rsa_band(D.RSA_BOUNDARY - 1e-9) == "buried"
        assert D.rsa_band(D.RSA_BOUNDARY) == "exposed"
        assert D.rsa_band(None) is None
        with pytest.raises(ValueError):
            D.rsa_band(math.inf)

    def test_secondary_structure_classes_cover_the_assignment_range(self):
        assert [D.ss_class(index) for index in range(3)] == list(D.SS_CLASSES)
        with pytest.raises(ValueError):
            D.ss_class(3)

    def test_separation_strata_partition_the_supported_range(self):
        assert D.separation_stratum(D.MIN_SEPARATION) == "24-31"
        assert D.separation_stratum(31) == "24-31"
        assert D.separation_stratum(32) == "32-63"
        assert D.separation_stratum(128) == "128+"
        assert D.separation_stratum(10_000) == "128+"

    def test_a_separation_below_the_floor_is_refused_not_banded(self):
        """A pair below the floor is outside this design's population entirely."""

        with pytest.raises(ValueError):
            D.separation_stratum(D.MIN_SEPARATION - 1)


class TestDeclarations:
    def test_the_non_contact_floor_is_above_the_contact_cutoff(self):
        """A pair between the two is in neither class; collapsing them would
        dilute the control with near-contacts and bias the endpoint to zero."""

        assert D.NON_CONTACT_ANGSTROM > D.CONTACT_ANGSTROM

    def test_the_window_floor_is_stricter_than_the_whole_model_floor(self):
        assert D.WINDOW_PLDDT_FLOOR > D.MODEL_PLDDT_FLOOR
        assert D.RESIDUE_PLDDT_FLOOR >= D.WINDOW_PLDDT_FLOOR

    def test_the_declared_arms_are_disjoint_from_the_unselected_ones(self):
        assert not set(D.ARMS) & set(D.UNSELECTED_ARMS)

    def test_every_filter_the_census_may_record_is_declared(self):
        assert len(set(D.FILTER_ORDER)) == len(D.FILTER_ORDER)

    def test_the_cell_seed_separates_conditions_and_modes(self):
        """Two cells sharing a prompt must not share a sample."""

        base = {"arm": "progen2-medium", "unit": "P00000:40:70"}
        seeds = {
            D.cell_seed(**base, condition=condition, mode=mode)
            for condition in D.CONDITIONS for mode in D.MODES
        }
        assert len(seeds) == len(D.CONDITIONS) * len(D.MODES)

    def test_the_cell_seed_is_reproducible_and_bounded(self):
        first = D.cell_seed(arm="rita-xl", unit="P0:1:2", condition="native", mode="sampled")
        second = D.cell_seed(arm="rita-xl", unit="P0:1:2", condition="native", mode="sampled")
        assert first == second and 0 <= first < 2**31 - 1

    def test_the_cell_seed_refuses_an_undeclared_condition_or_mode(self):
        with pytest.raises(ValueError):
            D.cell_seed(arm="a", unit="b", condition="control", mode="sampled")
        with pytest.raises(ValueError):
            D.cell_seed(arm="a", unit="b", condition="native", mode="masked")

    def test_the_output_names_every_non_identifiable_contrast(self):
        registration = D.pre_registration()
        for key in (
            "partner_identity_matched_scoring_contrast",
            "structural_chemistry_versus_learned_coevolution",
            "mechanism_attribution_to_the_anchor",
            "stability_and_function",
        ):
            assert key in registration["non_identifiable"]
        assert "never thermodynamic stability" in registration["interpretation"]


class TestReciprocalSwapGuarantee:
    def test_a_closed_matching_equalises_the_forced_residue_multiset(self):
        units = [
            {"native_residue": "A", "transplant_residue": "C"},
            {"native_residue": "C", "transplant_residue": "A"},
            {"native_residue": "K", "transplant_residue": "W"},
            {"native_residue": "W", "transplant_residue": "K"},
        ]
        report = D.forced_residue_multisets_match(units)
        assert report["matches"]
        assert report["native"] == report["transplant"] == {"A": 1, "C": 1, "K": 1, "W": 1}

    def test_an_unclosed_matching_is_reported_as_a_mismatch(self):
        """Dropping one side of a matched pair breaks the composition match, and
        the design has to say so rather than average over it."""

        units = [
            {"native_residue": "A", "transplant_residue": "C"},
            {"native_residue": "K", "transplant_residue": "W"},
        ]
        assert not D.forced_residue_multisets_match(units)["matches"]
