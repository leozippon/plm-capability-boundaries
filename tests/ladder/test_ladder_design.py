"""The ladder's declarations, the window algebra, and the refusals around them."""

from __future__ import annotations

import pytest

from src.capability.ladder import design


# ------------------------------------------------------------------ the ladder


def test_the_rungs_are_the_declared_extents_plus_one_anchor():
    assert design.RUNGS == ("k1", "k2", "k5", "k10", "k20", "k40", "full")
    assert [design.rung_extent(rung) for rung in design.RUNGS[:-1]] == list(design.WINDOW_EXTENTS)
    assert design.rung_extent(design.FULL_GENERATION) is None


def test_an_unknown_rung_is_refused_rather_than_given_an_extent():
    with pytest.raises(KeyError):
        design.rung_extent("k7")


def test_every_arm_declares_a_model_class_and_a_condition_the_ladder_knows():
    for name, spec in design.ARMS.items():
        assert spec.condition in design.CONDITIONS, name
        assert spec.model_class
        assert spec.note


def test_the_two_local_conditions_are_carried_by_different_arms():
    causal = {name for name, spec in design.ARMS.items() if spec.condition == design.CONDITION_CAUSAL_PREFIX}
    infill = set(design.INFILL_ARMS)
    assert causal and infill
    assert not causal & infill, "an arm cannot be in both local conditions"


def test_the_masked_arms_do_not_claim_a_sequence_likelihood():
    for name in design.INFILL_ARMS:
        assert design.arm(name).likelihood_kind == design.LIKELIHOOD_MASKED_PLL
    assert "pll_is_not_a_likelihood" in design.CEILING


def test_the_declaration_carries_the_ceiling_and_the_length_rule():
    declared = design.declaration()
    assert declared["ceiling"]["length_is_the_first_confound"] == design.REPORT_LENGTH_WITH_CONFIDENCE
    assert declared["backbones"]["n_backbones"] == design.N_BACKBONES
    assert declared["structure_readouts"]["primary"] == design.PRIMARY_STRUCTURE_READOUT


# ------------------------------------------------------------- window algebra


@pytest.mark.parametrize("length", [150, 187, 223, 300])
def test_the_windows_are_interior_and_nested_across_extents(length):
    spans = {extent: design.window_span(length, extent) for extent in design.WINDOW_EXTENTS}
    for extent, (start, stop) in spans.items():
        assert 1 <= start < stop <= length - 1, extent
        assert stop - start == extent
        assert start == design.window_anchor(length), "every extent shares one start"
    ordered = sorted(spans)
    for smaller, larger in zip(ordered, ordered[1:]):
        inner, outer = spans[smaller], spans[larger]
        assert outer[0] == inner[0] and inner[1] <= outer[1], (smaller, larger)


@pytest.mark.parametrize("length", [150, 187, 223, 300])
def test_the_anchor_centres_the_widest_extent_and_leaves_both_flanks(length):
    anchor = design.window_anchor(length)
    widest = max(design.WINDOW_EXTENTS)
    assert anchor >= 1
    assert anchor + widest <= length - 1
    assert abs(anchor - (length - widest) / 2) <= 1.0


def test_a_realised_start_inside_the_alignment_radius_is_accepted():
    length = 200
    anchor = design.window_anchor(length)
    for offset in (-design.TOKEN_ALIGNMENT_RADIUS, 0, design.TOKEN_ALIGNMENT_RADIUS):
        start, stop = design.window_span(length, 40, start=anchor + offset)
        assert start == anchor + offset and stop == start + 40
        assert 1 <= start and stop <= length - 1


def test_a_realised_start_that_would_reach_the_terminus_is_refused():
    with pytest.raises(ValueError, match="strictly interior"):
        design.window_span(150, 40, start=120)


def test_a_chain_too_short_for_an_interior_window_is_refused():
    with pytest.raises(ValueError):
        design.window_span(10, 40)


def test_an_extent_below_one_residue_is_not_a_rung():
    with pytest.raises(ValueError):
        design.window_span(200, 0)


def test_splicing_preserves_length_exactly():
    parent = "A" * 50 + "C" * 50
    start, stop = design.window_span(100, 10)
    spliced = design.splice(parent, start, "W" * 10)
    assert len(spliced) == len(parent)
    assert spliced[start:stop] == "W" * 10
    assert spliced[:start] == parent[:start] and spliced[stop:] == parent[stop:]


def test_splicing_a_window_of_the_wrong_length_reaches_the_terminus_and_is_refused():
    parent = "A" * 20
    with pytest.raises(ValueError):
        design.splice(parent, 1, "C" * 20)


def test_splicing_a_noncanonical_window_is_refused():
    with pytest.raises(ValueError):
        design.splice("A" * 20, 5, "XX")


def test_realised_substitutions_counts_only_what_actually_changed():
    parent = "ACDEFGHIKL"
    assert design.realised_substitutions(parent, parent) == 0
    assert design.realised_substitutions(parent, "ACDEFGHIKM") == 1
    with pytest.raises(ValueError):
        design.realised_substitutions(parent, parent + "A")


# ---------------------------------------------------------- the prefix prompt


@pytest.mark.parametrize("cut", [1, 59, 60, 61, 119, 120, 121])
def test_a_fasta_wrapped_prefix_rendering_is_always_an_exact_prefix(cut):
    def wrap(sequence: str) -> str:
        return "<|endoftext|>\n" + "\n".join(
            sequence[index : index + 60] for index in range(0, len(sequence), 60)
        )

    parent_sequence = "ACDEFGHIKLMNPQRSTVWY" * 10
    prompt = design.causal_prefix_prompt(wrap(parent_sequence), wrap(parent_sequence[:cut]))
    assert parent_sequence[:cut] == "".join(
        character for character in prompt.removeprefix("<|endoftext|>\n") if character != "\n"
    )


def test_a_conditioned_prefix_prompt_drops_the_terminal_marker():
    parent = "1.1.1.1<sep><start>ACDEFG<end>"
    prefix = "1.1.1.1<sep><start>ACD<end>"
    prompt = design.causal_prefix_prompt(parent, prefix, terminal_marker="<end>")
    assert prompt == "1.1.1.1<sep><start>ACD"
    assert parent.startswith(prompt)


def test_a_prefix_rendering_that_is_not_a_prefix_is_refused_not_sampled_from():
    with pytest.raises(ValueError):
        design.causal_prefix_prompt("Seq=<ACDEFG>", "Other=<ACD>", terminal_marker=">")


# ------------------------------------------------------------------ the seeds


def test_a_cell_seed_depends_on_every_coordinate_of_the_cell():
    base = dict(arm_name="protgpt2", backbone_id="lb_P1", rung="k5", draw=0)
    seeds = {
        design.cell_seed(**base),
        design.cell_seed(**{**base, "draw": 1}),
        design.cell_seed(**{**base, "rung": "k10"}),
        design.cell_seed(**{**base, "backbone_id": "lb_P2"}),
        design.cell_seed(**{**base, "arm_name": "zymctrl"}),
    }
    assert len(seeds) == 5
    assert design.cell_seed(**base) == design.cell_seed(**base)


def test_a_variant_id_is_stable_and_cell_specific():
    one = design.variant_id(arm_name="protgpt2", backbone_id="lb_P1", rung="k5", draw=3)
    assert one == design.variant_id(arm_name="protgpt2", backbone_id="lb_P1", rung="k5", draw=3)
    assert one != design.variant_id(arm_name="protgpt2", backbone_id="lb_P1", rung="k5", draw=4)


# --------------------------------------------------------------- the controls


def test_composition_distance_is_zero_for_a_permutation_and_positive_otherwise():
    assert design.composition_distance("ACDE", "ECDA") == pytest.approx(0.0)
    assert design.composition_distance("AAAA", "CCCC") == pytest.approx(1.0)


def test_repeat_and_low_complexity_readouts_see_a_homopolymer():
    assert design.longest_homopolymer("ACAAAAG") == 4
    assert design.longest_homopolymer("") == 0
    assert design.composition_entropy_bits("AAAA") == pytest.approx(0.0)
    assert design.composition_entropy_bits("ACGT"[:4]) > 1.0


def test_sequence_descriptors_report_realised_extent_only_at_equal_length():
    parent = "ACDEFGHIKL"
    same = design.sequence_descriptors("ACDEFGHIKM", parent=parent)
    assert same["realised_substitutions"] == 1
    assert same["length"] == 10
    longer = design.sequence_descriptors("ACDEFGHIKLMN", parent=parent)
    assert "realised_substitutions" not in longer
    assert "composition_distance_to_parent" in longer


# ------------------------------------------------------- backbone admission


def _candidate(**overrides):
    row = {
        "id": "nat_P0",
        "accession": "P0",
        "length": 200,
        "sequence": "ACDEFGHIKLMNPQRSTVWY" * 10,
        "roles": ["natural"],
        "structure": {
            "status": "ok",
            "mean_ca_plddt": 95.0,
            "fraction_ca_plddt_ge70": 0.99,
            "ptm": 0.95,
            "sequence_sha256": "deadbeef",
            "object_directory": "objects/deadbeef",
            "mean_pae_angstrom": 3.0,
        },
    }
    row.update(overrides)
    return row


def test_an_admissible_candidate_passes_every_declared_condition():
    assert design.backbone_rejection(_candidate(), ec_labels={"P0": ["1.1.1.1"]}) is None


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"structure": {**_candidate()["structure"], "status": "error"}}, "fold_not_ok"),
        ({"length": 60, "sequence": "A" * 60}, "outside_length_band"),
        ({"sequence": "X" * 200}, "noncanonical_residues"),
        (
            {"structure": {**_candidate()["structure"], "mean_ca_plddt": 70.0}},
            "below_confidence_floor",
        ),
        ({"structure": {**_candidate()["structure"], "ptm": 0.4}}, "below_confidence_floor"),
    ],
)
def test_each_declared_refusal_reason_is_reachable(overrides, expected):
    assert design.backbone_rejection(_candidate(**overrides), ec_labels={"P0": ["1.1.1.1"]}) == expected


def test_a_candidate_without_exactly_one_ec_number_cannot_carry_the_conditioned_arm():
    assert design.backbone_rejection(_candidate(), ec_labels={}) == "no_single_ec_label"
    assert (
        design.backbone_rejection(_candidate(), ec_labels={"P0": ["1.1.1.1", "2.2.2.2"]})
        == "no_single_ec_label"
    )


def test_the_strata_tile_the_declared_band_without_gaps_or_overlap():
    strata = design.length_strata()
    low, high = design.BACKBONE_LENGTH_BAND
    assert strata[0][0] == low and strata[-1][1] == high
    for left, right in zip(strata, strata[1:]):
        assert right[0] == left[1] + 1
    for length in range(low, high + 1):
        assert 0 <= design.stratum_of(length) < design.BACKBONE_STRATA
    with pytest.raises(ValueError):
        design.stratum_of(low - 1)


# ------------------------------------------------------------ variant records


def _backbone():
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 10
    return {
        "backbone_id": "lb_P0",
        "accession": "P0",
        "ec_label": "1.1.1.1",
        "stratum": 1,
        "length": len(sequence),
        "sequence": sequence,
        "parent_sequence_sha256": "deadbeef",
        "window_spans": {
            f"k{extent}": list(design.window_span(len(sequence), extent))
            for extent in design.WINDOW_EXTENTS
        },
    }


def test_a_filled_variant_record_carries_its_window_and_preserves_length():
    backbone = _backbone()
    start, stop = backbone["window_spans"]["k5"]
    sequence = design.splice(backbone["sequence"], start, "WWWWW")
    row = design.variant_record(
        arm_name="protgpt2",
        condition=design.CONDITION_CAUSAL_PREFIX,
        backbone=backbone,
        rung="k5",
        draw=2,
        sequence=sequence,
        status="filled",
    )
    assert row["length_preserved"] is True
    assert row["window"] == "WWWWW"
    assert row["parent_window"] == backbone["sequence"][start:stop]
    assert row["realised_substitutions"] == design.realised_substitutions(backbone["sequence"], sequence)
    assert row["sequence_sha256"] == design.sequence_digest(sequence)
    assert row["nominal_extent"] == 5


def test_an_unfilled_variant_record_stays_in_the_denominator():
    row = design.variant_record(
        arm_name="protgpt2",
        condition=design.CONDITION_CAUSAL_PREFIX,
        backbone=_backbone(),
        rung="k40",
        draw=0,
        sequence="",
        status="terminated_short",
    )
    assert row["sequence"] == "" and row["sequence_sha256"] == ""
    assert row["length_preserved"] is False
    assert row["status"] == "terminated_short"
    assert row["window_span"] is not None, "the rung it failed to express is still recorded"


def test_an_anchor_record_has_no_window_and_no_parent_referenced_descriptor():
    row = design.variant_record(
        arm_name="protgpt2",
        condition=design.CONDITION_FULL_GENERATION,
        backbone=_backbone(),
        rung=design.FULL_GENERATION,
        draw=0,
        sequence="ACDEFGHIKL",
        status="filled",
    )
    assert row["window_span"] is None
    assert "window" not in row
    assert "composition_distance_to_parent" not in row
    assert row["nominal_extent"] is None
