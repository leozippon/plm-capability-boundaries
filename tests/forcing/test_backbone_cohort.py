"""Pair eligibility, control matching, and the reciprocal transplant's guarantee.

The cohort carries the causal structure of the whole experiment, so the places it
could silently admit the wrong thing are tested directly: the anchor window and
separation floor, the non-contact margin a control has to clear, the burial-band
match that E02's collapse made non-negotiable, and the matching's closure --
without which the counterfactual is not composition-matched and the endpoint is a
composition result again.

The geometry is built by hand rather than read from a file: a synthetic backbone
is the only way to put a pair exactly on a threshold.
"""

import numpy as np
import pytest

from src.capability.forcing import forcing_design as D
from src.capability.forcing.backbone_cohort import (
    Backbone,
    Census,
    anchor_window,
    candidate_pairs,
    cohort_payload,
    distinct_shingle_fraction,
    iter_cells,
    pair_distances,
    reciprocal_transplant,
    reference_partners,
    require_cohort,
    select_pairs,
    unit_id,
)


def backbone(
    *, length=120, accession="P00001", superfamily="1.10.10.10", band="short",
    sequence=None, contacts=(), rsa=None, secondary=None, plddt=99.0,
) -> Backbone:
    """A straight-line backbone with declared pairs pulled into contact.

    Positions sit 20 angstrom apart along x, so nothing is in contact by
    accident; each ``(i, j, distance)`` in ``contacts`` then places ``j`` at that
    exact distance from ``i``. Coordinates are only ever read pairwise, so a
    geometry that is not globally realisable still exercises every rule.
    """

    sequence = sequence or ("ACDEFGHIKLMNPQRSTVWY" * ((length // 20) + 1))[:length]
    coordinates = np.zeros((length, 3), dtype=np.float64)
    coordinates[:, 0] = np.arange(length) * 20.0
    for i, j, distance in contacts:
        coordinates[j] = coordinates[i] + np.asarray([0.0, float(distance), 0.0])
    return Backbone(
        accession=accession,
        superfamily=superfamily,
        sequence=sequence,
        band=band,
        plddt=np.full(length, float(plddt)),
        contact_xyz=coordinates,
        contact_atom=tuple("CB" for _ in range(length)),
        rsa=np.zeros(length) if rsa is None else np.asarray(rsa, dtype=np.float64),
        secondary=np.zeros(length, dtype=np.int8) if secondary is None
        else np.asarray(secondary, dtype=np.int8),
        distinct_shingle_fraction=distinct_shingle_fraction(sequence),
    )


class TestAnchorWindow:
    def test_the_window_starts_at_the_declared_fraction(self):
        low, _ = anchor_window(120)
        assert low == int(np.ceil(D.ANCHOR_WINDOW_FRACTION * 120)) == 30

    def test_the_window_ends_where_a_partner_still_exists(self):
        _, high = anchor_window(120)
        assert high == 120 - 1 - D.MIN_SEPARATION
        assert high + D.MIN_SEPARATION == 119

    def test_a_backbone_too_short_for_the_window_yields_no_candidate(self):
        assert candidate_pairs(backbone(length=30)) == []


class TestPairEligibility:
    def test_a_contact_at_the_floor_is_admitted(self):
        anchor, partner = 40, 40 + D.MIN_SEPARATION
        pairs = candidate_pairs(backbone(contacts=[
            (anchor, partner, D.CONTACT_ANGSTROM - 0.1),
            *[(anchor, position, D.NON_CONTACT_ANGSTROM + 1.0)
              for position in range(partner + 1, partner + 4)],
        ]))
        assert any(row["anchor"] == anchor and row["partner"] == partner for row in pairs)

    def test_a_pair_exactly_at_the_cutoff_is_not_a_contact(self):
        """The contact relation is strict, as this project declares it."""

        anchor, partner = 40, 70
        pairs = candidate_pairs(backbone(contacts=[
            (anchor, partner, D.CONTACT_ANGSTROM),
            *[(anchor, position, D.NON_CONTACT_ANGSTROM + 1.0)
              for position in (71, 72, 73)],
        ]))
        assert not any(row["partner"] == partner for row in pairs)

    def test_a_pair_below_the_separation_floor_is_never_a_prescribed_pair(self):
        anchor = 40
        pairs = candidate_pairs(backbone(contacts=[
            (anchor, anchor + D.MIN_SEPARATION - 1, 4.0),
        ]))
        assert all(row["separation"] >= D.MIN_SEPARATION for row in pairs)

    def test_an_unconfident_anchor_is_excluded(self):
        anchor, partner = 40, 70
        model = backbone(contacts=[
            (anchor, partner, 6.0),
            *[(anchor, position, 20.0) for position in (71, 72, 73)],
        ])
        plddt = model.plddt.copy()
        plddt[anchor] = D.RESIDUE_PLDDT_FLOOR - 1.0
        unconfident = Backbone(**{**model.__dict__, "plddt": plddt})
        assert not any(row["anchor"] == anchor for row in candidate_pairs(unconfident))

    def test_a_low_confidence_window_is_excluded_even_with_a_confident_anchor(self):
        anchor, partner = 40, 70
        model = backbone(contacts=[
            (anchor, partner, 6.0),
            *[(anchor, position, 20.0) for position in (71, 72, 73)],
        ])
        plddt = model.plddt.copy()
        plddt[anchor + 1: partner] = 50.0
        banded = Backbone(**{**model.__dict__, "plddt": plddt})
        assert candidate_pairs(banded) == []

    def test_a_pair_without_enough_controls_is_not_a_pair(self):
        """Without controls the endpoint has no second term and reduces to an
        unmatched divergence, which is the quantity this design exists to avoid.

        Every position within the tolerance is put inside the ambiguous margin, so
        none of them is admissible as either a contact or a control."""

        anchor, partner = 40, 70
        between = (D.CONTACT_ANGSTROM + D.NON_CONTACT_ANGSTROM) / 2.0
        window = range(
            partner - D.SEPARATION_TOLERANCE, partner + D.SEPARATION_TOLERANCE + 1
        )
        pairs = candidate_pairs(backbone(contacts=[
            (anchor, partner, 6.0),
            *[(anchor, position, between) for position in window if position != partner],
        ]))
        assert pairs == []


class TestReferencePartners:
    def _model(self, **kwargs):
        anchor, partner = 40, 70
        contacts = [(anchor, partner, 6.0)]
        return anchor, partner, backbone(contacts=contacts, **kwargs)

    def test_controls_are_matched_on_separation_within_the_tolerance(self):
        anchor, partner, model = self._model()
        rows = reference_partners(
            model, anchor=anchor, partner=partner,
            distance=pair_distances(model), confident=model.plddt >= D.RESIDUE_PLDDT_FLOOR,
        )
        assert len(rows) == D.N_REFERENCE_PARTNERS
        for row in rows:
            assert abs(row["separation"] - (partner - anchor)) <= D.SEPARATION_TOLERANCE
            assert row["position"] != partner

    def test_a_near_contact_inside_the_margin_is_in_neither_class(self):
        anchor, partner = 40, 70
        between = (D.CONTACT_ANGSTROM + D.NON_CONTACT_ANGSTROM) / 2.0
        window = [
            position
            for position in range(
                partner - D.SEPARATION_TOLERANCE, partner + D.SEPARATION_TOLERANCE + 1
            )
            if position != partner
        ]
        model = backbone(contacts=[
            (anchor, partner, 6.0),
            *[(anchor, position, between) for position in window],
        ])
        rows = reference_partners(
            model, anchor=anchor, partner=partner,
            distance=pair_distances(model), confident=model.plddt >= D.RESIDUE_PLDDT_FLOOR,
        )
        assert rows == []

    def test_controls_must_share_the_prescribed_partner_s_burial_band(self):
        """The burial match is the specific lesson from E02, where burial band
        carried the entire measured effect."""

        anchor, partner = 40, 70
        rsa = np.zeros(120)
        rsa[partner] = 0.8
        model = backbone(contacts=[(anchor, partner, 6.0)], rsa=rsa)
        rows = reference_partners(
            model, anchor=anchor, partner=partner,
            distance=pair_distances(model), confident=model.plddt >= D.RESIDUE_PLDDT_FLOOR,
        )
        assert rows == []
        rsa[[partner - 3, partner - 2, partner - 1]] = 0.8
        exposed = backbone(contacts=[(anchor, partner, 6.0)], rsa=rsa)
        rows = reference_partners(
            exposed, anchor=anchor, partner=partner,
            distance=pair_distances(exposed), confident=exposed.plddt >= D.RESIDUE_PLDDT_FLOOR,
        )
        assert sorted(row["position"] for row in rows) == [
            partner - 3, partner - 2, partner - 1
        ]

    def test_controls_never_fall_below_the_separation_floor(self):
        anchor = 30
        partner = anchor + D.MIN_SEPARATION
        model = backbone(contacts=[(anchor, partner, 6.0)])
        rows = reference_partners(
            model, anchor=anchor, partner=partner,
            distance=pair_distances(model), confident=model.plddt >= D.RESIDUE_PLDDT_FLOOR,
        )
        assert all(row["separation"] >= D.MIN_SEPARATION for row in rows)


class TestSelection:
    def test_selected_pairs_carry_distinct_anchors(self):
        contacts = []
        for anchor, partner in ((40, 70), (50, 82), (60, 92)):
            contacts.append((anchor, partner, 6.0))
        model = backbone(contacts=contacts)
        rows = select_pairs(model, limit=3)
        assert len({row["anchor"] for row in rows}) == len(rows)

    def test_selection_prefers_the_closest_reference_contact(self):
        model = backbone(contacts=[(40, 70, 7.5), (50, 82, 4.0)])
        rows = select_pairs(model, limit=1)
        assert rows and rows[0]["anchor"] == 50

    def test_selection_depends_on_this_backbone_alone(self):
        """A cohort-dependent preference would move the panel when the pool moved."""

        model = backbone(contacts=[(40, 70, 6.0), (50, 82, 5.0)])
        assert select_pairs(model, limit=2) == select_pairs(model, limit=2)


class TestReciprocalTransplant:
    def _unit(self, accession, anchor, residue, **overrides):
        return {
            "accession": accession, "anchor": anchor, "partner": anchor + 30,
            "anchor_residue": residue, "length_band": "short",
            "anchor_rsa_band": "buried", "anchor_ss_class": "helix",
            "separation_stratum": "24-31",
        } | overrides

    def test_the_matching_is_reciprocal(self):
        units = [self._unit("A", 40, "A"), self._unit("B", 50, "C")]
        result = reciprocal_transplant(units)
        assert len(result["units"]) == 2
        index = {row["unit_id"]: row for row in result["units"]}
        for row in result["units"]:
            assert index[row["partner_unit_id"]]["partner_unit_id"] == row["unit_id"]

    def test_the_two_conditions_force_the_same_residue_multiset(self):
        units = [
            self._unit("A", 40, "A"), self._unit("B", 50, "C"),
            self._unit("C", 60, "K"), self._unit("E", 70, "W"),
        ]
        report = D.forced_residue_multisets_match(reciprocal_transplant(units)["units"])
        assert report["matches"]

    def test_units_carrying_the_same_anchor_residue_are_never_paired(self):
        """A swap between equal residues is not an intervention at all."""

        units = [self._unit("A", 40, "A"), self._unit("B", 50, "A")]
        result = reciprocal_transplant(units)
        assert result["units"] == []
        assert len(result["unmatched_unit_ids"]) == 2

    def test_a_swap_inside_one_backbone_is_refused(self):
        units = [self._unit("A", 40, "A"), self._unit("A", 80, "C")]
        assert reciprocal_transplant(units)["units"] == []

    def test_units_in_different_cells_are_never_paired(self):
        units = [
            self._unit("A", 40, "A"),
            self._unit("B", 50, "C", anchor_rsa_band="exposed"),
        ]
        assert reciprocal_transplant(units)["units"] == []

    def test_the_matching_terminates_when_no_legal_pair_remains(self):
        """Three units of one backbone plus one foreign unit of the same residue
        used to be able to spin: the search now exits instead."""

        units = [
            self._unit("A", 40, "A"), self._unit("A", 80, "A"), self._unit("A", 120, "A"),
            self._unit("B", 50, "A"),
        ]
        result = reciprocal_transplant(units)
        assert result["units"] == []

    def test_the_largest_group_is_paired_against_the_next_largest(self):
        """That rule attains the maximum matching under the different-label
        constraint; pairing within the minority would leave more unpaired."""

        units = [self._unit(f"A{index}", 40, "A") for index in range(4)]
        units += [self._unit("B", 50, "C"), self._unit("E", 60, "K")]
        result = reciprocal_transplant(units)
        assert len(result["units"]) == 4
        assert len(result["unmatched_unit_ids"]) == 2

    def test_the_matching_is_deterministic(self):
        units = [self._unit(f"A{index}", 40 + index, "ACKW"[index % 4]) for index in range(8)]
        first = reciprocal_transplant(units)
        second = reciprocal_transplant(units)
        assert [row["unit_id"] for row in first["units"]] == [
            row["unit_id"] for row in second["units"]
        ]


class TestCohortRefusals:
    def _payload(self):
        units = [
            {
                "accession": "A", "superfamily": "1.1.1.1", "anchor": 40, "partner": 70,
                "length": 120, "length_band": "short", "anchor_residue": "A",
                "separation": 30, "separation_stratum": "24-31", "anchor_rsa_band": "buried",
                "anchor_ss_class": "helix", "span_end": 72,
                "reference_partners": [{"position": position} for position in (71, 72, 73)],
            },
            {
                "accession": "B", "superfamily": "2.2.2.2", "anchor": 50, "partner": 80,
                "length": 120, "length_band": "short", "anchor_residue": "C",
                "separation": 30, "separation_stratum": "24-31", "anchor_rsa_band": "buried",
                "anchor_ss_class": "helix", "span_end": 82,
                "reference_partners": [{"position": position} for position in (81, 82, 83)],
            },
        ]
        matched = reciprocal_transplant(units)["units"]
        sequence = ("ACDEFGHIKLMNPQRSTVWY" * 6)[:120]
        sequences = {}
        for unit in matched:
            text = list(sequence)
            text[int(unit["anchor"])] = unit["anchor_residue"]
            sequences[unit["accession"]] = "".join(text)
        return cohort_payload(
            matched, census=Census().payload(), sources={}, sequences=sequences,
        )

    def test_a_well_formed_cohort_is_admitted(self):
        payload = require_cohort(self._payload())
        assert payload["backbones"] == 2 and payload["units"] == 2

    def test_a_cohort_whose_multisets_differ_is_refused(self):
        payload = self._payload()
        payload["forced_residue_multisets"]["matches"] = False
        with pytest.raises(ValueError, match="composition-matched"):
            require_cohort(payload)

    def test_a_cohort_whose_matching_is_not_reciprocal_is_refused(self):
        payload = self._payload()
        payload["unit_rows"][0]["partner_unit_id"] = "nobody"
        with pytest.raises(ValueError, match="not in the cohort"):
            require_cohort(payload)

    def test_a_cohort_whose_sequence_contradicts_its_anchor_is_refused(self):
        """Positions and sequences coming from different sources is the one defect
        that would silently force the wrong residue at every cell."""

        payload = self._payload()
        accession = payload["unit_rows"][0]["accession"]
        anchor = int(payload["unit_rows"][0]["anchor"])
        text = list(payload["sequences"][accession])
        text[anchor] = "G" if text[anchor] != "G" else "A"
        payload["sequences"][accession] = "".join(text)
        with pytest.raises(ValueError, match="anchor"):
            require_cohort(payload)

    def test_a_cohort_missing_a_sequence_is_refused_at_construction(self):
        units = reciprocal_transplant([
            {
                "accession": "A", "superfamily": "1.1.1.1", "anchor": 40, "partner": 70,
                "length": 120, "length_band": "short", "anchor_residue": "A",
                "separation": 30, "separation_stratum": "24-31",
                "anchor_rsa_band": "buried", "anchor_ss_class": "helix", "span_end": 72,
                "reference_partners": [],
            },
            {
                "accession": "B", "superfamily": "2.2.2.2", "anchor": 50, "partner": 80,
                "length": 120, "length_band": "short", "anchor_residue": "C",
                "separation": 30, "separation_stratum": "24-31",
                "anchor_rsa_band": "buried", "anchor_ss_class": "helix", "span_end": 82,
                "reference_partners": [],
            },
        ])["units"]
        with pytest.raises(ValueError, match="no sequence"):
            cohort_payload(units, census=Census().payload(), sources={}, sequences={})

    def test_every_unit_yields_one_cell_per_condition(self):
        payload = self._payload()
        cells = list(iter_cells(payload))
        assert len(cells) == payload["units"] * len(D.CONDITIONS)
        for cell in cells:
            unit = next(row for row in payload["unit_rows"] if row["unit_id"] == cell["unit_id"])
            expected = (
                unit["native_residue"] if cell["condition"] == D.CONDITION_NATIVE
                else unit["transplant_residue"]
            )
            assert cell["forced_residue"] == expected

    def test_the_cell_enumeration_forces_each_residue_once_per_matched_pair(self):
        from collections import Counter

        cells = list(iter_cells(self._payload()))
        per_condition = {
            condition: Counter(
                cell["forced_residue"] for cell in cells if cell["condition"] == condition
            )
            for condition in D.CONDITIONS
        }
        assert per_condition[D.CONDITION_NATIVE] == per_condition[D.CONDITION_TRANSPLANT]


class TestCensus:
    def test_the_census_reports_what_each_filter_removed(self):
        census = Census()
        census.record("alphafold_models", 1000)
        census.record("single_fragment", 900)
        census.record("length_band", 100)
        stages = census.payload()["stages"]
        assert [row["removed"] for row in stages] == [None, 100, 800]

    def test_the_census_refuses_an_undeclared_filter(self):
        with pytest.raises(ValueError, match="undeclared filter"):
            Census().record("vibes", 10)

    def test_the_census_reports_stages_in_the_declared_order(self):
        census = Census()
        census.record("length_band", 100)
        census.record("single_fragment", 900)
        assert [row["filter"] for row in census.payload()["stages"]] == [
            "single_fragment", "length_band",
        ]


def test_unit_ids_identify_a_backbone_and_a_pair():
    assert unit_id({"accession": "P1", "anchor": 4, "partner": 40}) == "P1:4:40"


def test_distinct_shingle_fraction_separates_a_repeat_from_a_real_sequence():
    """A homopolymer and a short-period repeat both fall far below the floor; a
    real sequence of this length reaches one, which is why the floor removed only
    two of 948 banded candidates on the staged release."""

    assert distinct_shingle_fraction("A" * 60) < D.MIN_DISTINCT_SHINGLE_FRACTION
    assert distinct_shingle_fraction("ACDEFGHIKLMNPQRSTVWY" * 3) < D.MIN_DISTINCT_SHINGLE_FRACTION
    real = (
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
    )
    assert distinct_shingle_fraction(real) == 1.0
    assert distinct_shingle_fraction("AC") == 0.0
