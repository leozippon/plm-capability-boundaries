"""Contact definition edge cases, the causal zero, and the refusals that matter.

The contact relation carries the structural claim of E01 and E02, so the places
it could silently admit the wrong pair are tested directly: the cutoff boundary,
the sequence-separation floor, and a position whose coordinate was never mapped.
An assay without an admitted structure must fail by name rather than quietly
produce a non-structural number under a structural heading, which is the last
test group here.
"""

import hashlib

import numpy as np
import pytest

from src.capability.position.anticipation import (
    anchor_partner_design,
    composition,
    excess_logprob,
    pair_conditional,
)
from src.capability.position.contact_response import (
    CONTACT_ANGSTROM,
    MIN_SEQUENCE_SEPARATION,
    ProfileAccumulator,
    SiteGeometry,
    admitted_geometry,
    agrees_with_frozen,
    contact_pairs,
    decay_fit,
    receiver_census,
    require_geometry,
    separation_stratum,
    stratified_contact_contrast,
)
from src.capability.position.position_likelihood import CAUSAL, MASKED

WILDTYPE = "MKQLEDKVEELLSKNYHLENEVARLKKLVGER"
ALPHABET = tuple(sorted(set(WILDTYPE)))


def line(positions, *, spacing=1.0):
    """Residues on a straight line, so every distance is exactly computable."""

    return {
        position: SiteGeometry(
            j=position, residue=WILDTYPE[position],
            atom="CA" if WILDTYPE[position] == "G" else "CB",
            xyz=(spacing * position, 0.0, 0.0), rsa=0.25,
        )
        for position in positions
    }


# ------------------------------------------------------- the contact relation


def test_the_cutoff_is_strict_so_a_pair_exactly_at_it_is_not_a_contact():
    geometry = line([0, 8, 16], spacing=1.0)
    pairs = contact_pairs(geometry, cutoff=8.0, min_separation=6)
    assert pairs["distance"][(0, 8)] == 8.0
    assert (0, 8) not in pairs["contacts"]
    assert (8, 16) not in pairs["contacts"]
    closer = contact_pairs(line([0, 8], spacing=0.9), cutoff=8.0, min_separation=6)
    assert (0, 8) in closer["contacts"]


def test_the_separation_floor_removes_trivially_close_neighbours():
    geometry = line(range(12), spacing=0.5)
    floored = contact_pairs(geometry, min_separation=6)
    assert all(high - low >= 6 for low, high, _ in floored["pairs"])
    assert floored["min_separation"] == 6
    loose = contact_pairs(geometry, min_separation=1)
    assert loose["n_pairs"] > floored["n_pairs"]
    assert all(high - low >= 1 for low, high, _ in loose["pairs"])


def test_a_position_without_a_coordinate_is_in_neither_class():
    full = contact_pairs(line([0, 7, 14, 21]), min_separation=6)
    partial = contact_pairs(line([0, 14, 21]), min_separation=6)
    assert 7 in full["eligible_positions"]
    assert 7 not in partial["eligible_positions"]
    # A dropped position does not become a non-contact: it leaves the population.
    assert partial["n_pairs"] < full["n_pairs"]
    assert all(7 not in (low, high) for low, high, _ in partial["pairs"])


def test_the_relation_is_symmetric_carries_no_self_pair_and_declares_itself():
    pairs = contact_pairs(line([0, 7, 14]), min_separation=6)
    assert all(low < high for low, high, _ in pairs["pairs"])
    assert all(low != high for low, high, _ in pairs["pairs"])
    assert "8.0 angstrom" in pairs["definition"] and "at least 6" in pairs["definition"]
    assert pairs["cutoff_angstrom"] == CONTACT_ANGSTROM == 8.0
    assert MIN_SEQUENCE_SEPARATION == 6


def test_an_impossible_cutoff_or_floor_is_refused_rather_than_clamped():
    geometry = line([0, 7, 14])
    for cutoff in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="positive finite distance"):
            contact_pairs(geometry, cutoff=cutoff)
    with pytest.raises(ValueError, match="at least one residue"):
        contact_pairs(geometry, min_separation=0)


def test_a_non_finite_coordinate_cannot_enter_the_relation():
    geometry = dict(line([0, 7]))
    geometry[14] = SiteGeometry(j=14, residue="S", atom="CB",
                                xyz=(float("nan"), 0.0, 0.0), rsa=None)
    with pytest.raises(ValueError):
        contact_pairs(geometry, min_separation=6)


def test_the_separation_strata_reuse_the_frozen_edges():
    assert separation_stratum(6) == "3-8"
    assert separation_stratum(9) == "9-16"
    assert separation_stratum(33) == "33-64"
    assert separation_stratum(500) == "129+"


# ----------------------------------------------- missing structure: refusals


def structure_fixture(wildtype=WILDTYPE, *, admitted_positions=None, drop_position=None,
                      assay="TEST_ASSAY", cluster=7):
    """A minimal site table and the coverage receipt it must agree with."""

    digest = hashlib.sha256(wildtype.encode()).hexdigest()
    source_sha = "a" * 64
    source_path = "/structures/TEST.cif.gz"
    provenance = {
        "source_sha256": source_sha, "source_path": source_path, "entry_id": "TEST",
        "entity_id": "1", "chain": "A", "auth_chain": "A", "model": 1,
        "mapping_kind": "full_entity",
    }
    admitted = set(range(len(wildtype)) if admitted_positions is None else admitted_positions)
    rows = []
    for index, residue in enumerate(wildtype):
        if drop_position is not None and index == drop_position:
            continue
        present = index in admitted
        rows.append({
            "assay_id": assay, "cluster": cluster, "wt_position": index + 1,
            "residue": residue, "wildtype": wildtype, "wildtype_sha256": digest,
            "status": "admitted", "method": "SOLUTION NMR",
            "contact_atom": "CA" if residue == "G" else "CB",
            "contact_atom_present": present, "coordinate_present": present,
            "contact_coordinate": [float(index), 0.0, 0.0] if present else None,
            "rsa": 0.3 if present else None, "secondary_structure": None,
            **provenance,
        })
    coverage = {
        "assays": [{"assay": assay, "cluster": cluster, "status": "admitted",
                    "method": "SOLUTION NMR", **provenance}],
        "sources": [{"path": source_path, "sha256": source_sha, "status": "parsed",
                     "method": "SOLUTION NMR", "entry_id": "TEST"}],
    }
    return rows, coverage, assay, cluster


def test_a_valid_site_table_yields_only_the_mapped_coordinates():
    rows, coverage, assay, cluster = structure_fixture(admitted_positions=range(0, 32, 2))
    geometry = admitted_geometry(rows, assay=assay, family=cluster, wildtype=WILDTYPE,
                                 coverage=coverage)
    assert sorted(geometry) == list(range(0, 32, 2))
    assert geometry[0].residue == WILDTYPE[0]


def test_an_incomplete_site_table_fails_rather_than_mapping_what_it_has():
    rows, coverage, assay, cluster = structure_fixture(drop_position=5)
    with pytest.raises(ValueError, match="every WT site"):
        admitted_geometry(rows, assay=assay, family=cluster, wildtype=WILDTYPE,
                          coverage=coverage)


def test_a_site_table_for_another_family_fails_rather_than_being_reassigned():
    rows, coverage, assay, cluster = structure_fixture()
    with pytest.raises(ValueError, match="assay/family binding mismatch"):
        admitted_geometry(rows, assay=assay, family=cluster + 1, wildtype=WILDTYPE,
                          coverage=coverage)


def test_an_assay_absent_from_the_table_fails_by_name():
    rows, coverage, _assay, cluster = structure_fixture()
    with pytest.raises(ValueError):
        admitted_geometry(rows, assay="MISSING_ASSAY", family=cluster, wildtype=WILDTYPE,
                          coverage=coverage)


def test_an_unmappable_assay_is_refused_and_not_analysed_without_structure():
    with pytest.raises(ValueError, match="no structural support and is refused"):
        require_geometry({}, assay="TEST_ASSAY")
    with pytest.raises(ValueError, match="no structural support and is refused"):
        require_geometry(line([3]), assay="TEST_ASSAY")
    require_geometry(line([3, 20]), assay="TEST_ASSAY")


# --------------------------------------------------- the per-residue response


def archive(wild_terms, mutant_terms, *, counts=None, offset=0):
    """A minimal two-state payload in the layout the archive writes."""

    wild = np.asarray(wild_terms, dtype=np.float32)
    mutant = np.asarray(mutant_terms, dtype=np.float32)
    counts = np.ones(len(wild), dtype=np.int64) if counts is None else np.asarray(counts)
    return {
        "position_nats": np.concatenate([wild, mutant]),
        "position_offsets": np.asarray([0, len(wild), len(wild) + len(mutant)]),
        "position_residue_counts": np.concatenate([counts, counts]),
        "position_residue_offsets": np.asarray([offset, offset]),
        "state_id_offsets": np.asarray([0, len(wild) + 1, 2 * (len(wild) + 1)]),
        "state_ids": np.arange(2 * (len(wild) + 1)),
        "state_spans": np.asarray([[1, len(wild) + 1], [1, len(mutant) + 1]]),
    }


def test_the_response_before_a_substitution_is_required_to_be_exactly_zero():
    wild = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    moved = [1.0, 1.0, 1.0001, 1.0, 2.0, 2.0, 2.0, 2.0]
    with pytest.raises(ValueError, match="packing, alignment or forward defect"):
        receiver_census(archive(wild, moved), 0, site=4, paradigm=CAUSAL)


def test_a_clean_causal_response_is_one_sided_and_reports_its_own_census():
    wild = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    mutant = [1.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.5, 1.0]
    census = receiver_census(archive(wild, mutant), 0, site=4, paradigm=CAUSAL)
    assert census["upstream_max_abs_nats"] == 0.0
    assert census["upstream_receivers"] == 4
    assert census["downstream_receivers"] == 3
    assert census["site_response"] == pytest.approx(-2.0)
    assert all(row["response"] == 0.0 for row in census["receivers"]
               if row["direction"] == "upstream")
    assert census["total"] == pytest.approx(-3.5)


def test_a_masked_arm_may_respond_upstream_and_that_is_recorded_not_refused():
    wild = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    mutant = [1.2, 0.9, 1.0, 1.4, 3.0, 2.0, 1.5, 1.0]
    census = receiver_census(archive(wild, mutant), 0, site=4, paradigm=MASKED)
    assert census["upstream_max_abs_nats"] == pytest.approx(0.4, abs=1e-6)
    assert any(row["response"] != 0.0 for row in census["receivers"]
               if row["direction"] == "upstream")


def test_a_multiresidue_token_is_a_remainder_and_never_attributed_to_a_position():
    wild = [1.0, 1.0, 1.0, 1.0, 1.0]
    mutant = [1.0, 1.0, 1.0, 2.0, 3.0]
    counts = np.asarray([1, 1, 1, 2, 1], dtype=np.int64)
    census = receiver_census(archive(wild, mutant, counts=counts), 0, site=2,
                             paradigm=CAUSAL)
    assert {row["j"] for row in census["receivers"]} == {0, 1, 5}
    assert census["multiresidue_remainder"] == pytest.approx(-1.0)


def test_a_disagreement_with_the_frozen_reader_is_an_error_not_a_tolerance():
    wild = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    mutant = [1.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.5, 1.0]
    census = receiver_census(archive(wild, mutant), 0, site=4, paradigm=CAUSAL)
    frozen = {"receivers": [{"j": 5, "response": -1.0}, {"j": 6, "response": -0.5},
                            {"j": 7, "response": 0.0}]}
    assert agrees_with_frozen(census, frozen)["downstream_receivers"] == 3
    frozen["receivers"][0]["response"] = -1.25
    with pytest.raises(ValueError, match="differ from the frozen reader"):
        agrees_with_frozen(census, frozen)
    del frozen["receivers"][0]
    with pytest.raises(ValueError, match="receiver sets disagree"):
        agrees_with_frozen(census, frozen)


# -------------------------------------------------- the profile and contrast


def test_the_profile_streams_and_its_decay_describes_a_falling_response():
    accumulator = ProfileAccumulator(direction="downstream", min_support=1)
    for separation in range(1, 21):
        accumulator.add([{"direction": "downstream", "separation": separation,
                          "response": -float(np.exp(-0.1 * separation))}
                         for _ in range(3)])
    profile = accumulator.profile()
    assert profile["receivers"] == 60 and profile["max_separation"] == 20
    assert profile["decay"]["fitted"] is True
    assert profile["decay"]["slope_per_residue"] == pytest.approx(-0.1, abs=1e-6)
    assert profile["decay"]["half_distance_residues"] == pytest.approx(np.log(2) / 0.1, rel=1e-3)


def test_a_flat_or_rising_profile_reports_no_half_distance():
    flat = [{"separation": d, "n": 10, "mean_absolute_nats": 0.5, "sufficient": True}
            for d in range(1, 6)]
    assert decay_fit(flat)["fitted"] is False
    assert decay_fit(flat)["half_distance_residues"] is None
    assert decay_fit(flat)["half_distance_undefined_reason"]

    rising = [{"separation": d, "n": 10, "mean_absolute_nats": 0.1 * d, "sufficient": True}
              for d in range(1, 8)]
    fitted = decay_fit(rising)
    assert fitted["fitted"] is True and fitted["slope_per_residue"] > 0
    assert fitted["half_distance_residues"] is None

    noisy = [{"separation": d, "n": 10, "sufficient": True,
              "mean_absolute_nats": 0.5 + 0.3 * ((-1) ** d)} for d in range(1, 9)]
    assert decay_fit(noisy)["half_distance_residues"] is None

    # Too few supported separations is not a fit at all, and still has the key.
    short = decay_fit(flat[:2])
    assert short["fitted"] is False and short["half_distance_residues"] is None


def test_the_contrast_compares_only_inside_one_anchor_and_one_stratum():
    def row(contact, stratum, separation, response, mutation="A1G"):
        return {"assay": "A", "family": 1, "mutation": mutation, "i": 0, "j": separation,
                "separation": separation, "direction": "downstream",
                "response": response, "absolute_response": abs(response),
                "contact": contact, "structure_distance_angstrom": 5.0 if contact else 15.0,
                "rsa": 0.2, "stratum": stratum}

    unmatched = [row(True, "9-16", 10, -1.0), row(False, "33-64", 40, -0.1)]
    result = stratified_contact_contrast(unmatched)
    assert result["matched_cells"] == 0
    assert result["estimate_nats"]["point"] is None

    matched = []
    for family in range(1, 7):
        for contact, value in ((True, -1.0), (False, -0.4)):
            item = row(contact, "9-16", 10, value)
            item["family"] = family
            item["assay"] = f"A{family}"
            matched.append(item)
    result = stratified_contact_contrast(matched)
    assert result["matched_cells"] == 6
    assert result["estimate_nats"]["point"] == pytest.approx(0.6)
    assert result["separation_imbalance_residues"] == pytest.approx(0.0)


# --------------------------------------------------- the anticipation statistic


def test_the_anticipation_design_keeps_only_matched_anchor_strata():
    geometry = line(range(0, 32, 2), spacing=0.6)
    pairs = contact_pairs(geometry, cutoff=8.0, min_separation=6)
    rows = anchor_partner_design(
        geometry, pairs, anchors=sorted(geometry), wildtype=WILDTYPE, residues=ALPHABET,
    )
    assert rows
    assert all(row["j"] > row["i"] for row in rows)
    assert all(row["separation"] >= 6 for row in rows)
    for anchor in {row["i"] for row in rows}:
        for stratum in {row["stratum"] for row in rows if row["i"] == anchor}:
            cell = [row for row in rows if row["i"] == anchor and row["stratum"] == stratum]
            assert any(row["contact"] for row in cell)
            assert any(not row["contact"] for row in cell)


def test_the_excess_log_probability_is_renormalised_and_composition_centred():
    residues = ("A", "C", "D", "E")
    weights = composition("AACD", residues)
    assert weights == pytest.approx([0.5, 0.25, 0.25, 0.0])
    flat = np.log(np.full(4, 0.25))
    assert excess_logprob(flat, weights, 0) == pytest.approx(0.0)
    # An unnormalised row gives the same answer as its normalisation.
    peaked = np.asarray([0.0, -2.0, -2.0, -2.0])
    assert excess_logprob(peaked, weights, 0) == pytest.approx(
        excess_logprob(peaked - 3.1415, weights, 0)
    )
    assert excess_logprob(peaked, weights, 0) > 0.0
    assert excess_logprob(peaked, weights, 1) < 0.0
    with pytest.raises(ValueError, match="align and be finite"):
        excess_logprob(np.asarray([0.0, float("inf"), 0.0, 0.0]), weights, 0)


def test_the_covariation_control_never_sees_its_own_family():
    residues = ("A", "C", "D", "E")
    rows = [{"family": 1, "anchor_residue": "A", "partner_residue": "A", "contact": True},
            {"family": 1, "anchor_residue": "A", "partner_residue": "A", "contact": False},
            {"family": 2, "anchor_residue": "A", "partner_residue": "E", "contact": True},
            {"family": 2, "anchor_residue": "A", "partner_residue": "E", "contact": False}]
    held_one = pair_conditional(rows, residues, holdout=1)
    held_two = pair_conditional(rows, residues, holdout=2)
    assert held_one[0].argmax() == residues.index("E")
    assert held_two[0].argmax() == residues.index("A")
    with pytest.raises(ValueError, match="only family"):
        pair_conditional(rows[:2], residues, holdout=1)


# ------------------------------- the admitted upstream tolerance


def test_the_census_admits_only_the_tolerance_the_extraction_recorded():
    wild = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    drifted = [1.0, 1.0, 1.000004, 1.0, 2.0, 2.0, 2.0, 2.0]
    payload = archive(wild, drifted)
    with pytest.raises(ValueError, match="admitted tolerance of 0.0"):
        receiver_census(payload, 0, site=4, paradigm=CAUSAL)
    census = receiver_census(payload, 0, site=4, paradigm=CAUSAL, upstream_tolerance=1e-5)
    assert census["upstream_max_abs_nats"] == pytest.approx(4e-6, rel=5e-2)
    # A tolerance never turns a half-nat layout shift into a measurement.
    gross = archive(wild, [1.0, 1.0, 1.5, 1.0, 2.0, 2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="packing, alignment or forward defect"):
        receiver_census(gross, 0, site=4, paradigm=CAUSAL, upstream_tolerance=1e-4)
    for bad in (-1e-9, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite non-negative"):
            receiver_census(payload, 0, site=4, paradigm=CAUSAL, upstream_tolerance=bad)


# ------------------------------- the E02 attribution controls


def anticipation_rows(n_families=8, *, contact_shift=0.0):
    """Matched cells across several families, with a tunable contact signal."""

    rows = []
    for family in range(n_families):
        for anchor in (0, 10):
            for offset, contact in ((7, True), (8, False), (9, False)):
                rows.append({
                    "assay": f"A{family}", "family": family, "i": anchor,
                    "j": anchor + offset, "separation": offset, "stratum": "3-8",
                    "contact": contact, "structure_distance_angstrom": 5.0 if contact else 15.0,
                    "anchor_residue": "A", "partner_residue": "L" if contact else "D",
                    "anchor_rsa": 0.1 if anchor == 0 else 0.6,
                    "partner_rsa": 0.2 if contact else 0.5,
                })
    return rows


def test_the_within_protein_anchor_control_is_a_derangement_of_its_own_protein():
    from src.capability.position.anticipation import permuted_anchor_maps

    rows = anticipation_rows()
    within, across = permuted_anchor_maps(rows)
    assert within
    for (assay, anchor), (other_assay, other_anchor) in within.items():
        assert other_assay == assay            # same protein, same family, same fold
        assert other_anchor != anchor          # and never the anchor itself
    for (assay, _anchor), (other_assay, _other) in across.items():
        assert other_assay != assay            # a different protein in a different family


def test_a_single_anchor_protein_contributes_no_within_protein_control():
    from src.capability.position.anticipation import permuted_anchor_maps

    rows = [row for row in anticipation_rows() if row["i"] == 0]
    within, _across = permuted_anchor_maps(rows)
    assert within == {}


def test_the_contact_label_permutation_null_is_centred_on_zero():
    from src.capability.position.anticipation import permutation_null

    cells = {
        (family, f"A{family}", 0, "3-8"): {"contact": [1.0], "control": [0.0, 0.2]}
        for family in range(8)
    }
    null = permutation_null(cells, 0.9, draws=400, seed=7)
    assert null["draws"] == 400
    assert abs(null["mean_nats"]) < 0.2
    assert null["interval_nats"][0] < 0 < null["interval_nats"][1]
    assert null["two_sided_p"] < 0.05
    # An observed value inside the null is not significant.
    assert permutation_null(cells, 0.0, draws=400, seed=7)["two_sided_p"] > 0.2
    assert permutation_null({}, 0.9)["draws"] == 0


def test_the_attribution_pairs_the_measurement_against_each_control():
    from src.capability.position.anticipation import paired_attribution

    sources = {
        "model_conditional": {"per_family_contrast": {str(f): 0.20 for f in range(10)}},
        "anchor_permuted_within_protein": {
            "per_family_contrast": {str(f): 0.12 for f in range(10)}
        },
        "pair_covariation": {"per_family_contrast": {}},
    }
    result = paired_attribution(sources)
    fold = result["controls"]["anchor_permuted_within_protein"]
    assert fold["families"] == 10
    assert fold["control_nats"] == pytest.approx(0.12)
    assert fold["share_explained"] == pytest.approx(0.6)
    assert fold["residual_nats"]["point"] == pytest.approx(0.08)
    assert result["controls"]["pair_covariation"]["status"] == "no shared family"
    assert paired_attribution({})["status"]


def test_a_control_that_fully_explains_the_measurement_leaves_no_residual():
    from src.capability.position.anticipation import paired_attribution

    sources = {
        "model_conditional": {"per_family_contrast": {str(f): 0.2 + 0.01 * f for f in range(10)}},
        "anchor_permuted_within_protein": {
            "per_family_contrast": {str(f): 0.2 + 0.01 * f for f in range(10)}
        },
    }
    residual = paired_attribution(sources)["controls"]["anchor_permuted_within_protein"]
    assert residual["residual_nats"]["point"] == pytest.approx(0.0)
    assert residual["residual_nats"]["excludes_zero"] is False
    assert residual["share_explained"] == pytest.approx(1.0)
