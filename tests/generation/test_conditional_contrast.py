"""The requested-minus-mismatched endpoint: pairing balance, degeneracy, refusals."""

from __future__ import annotations

import hashlib
import json

import pytest

from src.capability.generation import conditional_contrast as cc
from src.capability.generation import conditioned_generation as cg


def _row(**overrides):
    sequence = overrides.pop("sequence", "MKV")
    row = {
        "id": overrides.pop("id", "ge_" + hashlib.sha256(sequence.encode()).hexdigest()[:24]),
        "arm": "zymctrl",
        "class_key": "c1",
        "condition": "requested",
        "sequence": sequence,
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "length": len(sequence),
        "target_profile_hit": False,
        "near_duplicate_group": None,
    }
    row.update(overrides)
    return row


def _cells(classes, *, requested_hits, mismatched_hits, pairing):
    rows = []
    for index, cls in enumerate(classes):
        for condition, hits in (("requested", requested_hits), ("mismatched", mismatched_hits)):
            for sample in range(10):
                sequence = f"MK{cls}{condition}{sample}"
                rows.append(
                    _row(
                        id=f"ge_{cls}_{condition}_{sample}",
                        class_key=cls,
                        condition=condition,
                        sequence=sequence,
                        target_profile_hit=sample < hits,
                        native_prompt_class=pairing[cls] if condition == "mismatched" else cls,
                    )
                )
    return rows


# ------------------------------------------------------------------ the pairing


def test_balanced_derangement_is_accepted():
    pairing = {"a": "b", "b": "c", "c": "a"}
    record = cc.verify_pairing_balance(pairing)
    assert record["balanced"] is True
    assert record["fixed_points"] == []
    assert record["donor_use_counts"] == {"a": 1, "b": 1, "c": 1}


def test_a_fixed_point_is_refused():
    with pytest.raises(ValueError, match="paired with themselves"):
        cc.verify_pairing_balance({"a": "a", "b": "c", "c": "b"})


def test_an_unbalanced_donor_set_is_refused():
    # "b" donates twice and "c" never; one easy donor would carry the negative side.
    with pytest.raises(ValueError, match="not used exactly once as a donor"):
        cc.verify_pairing_balance({"a": "b", "b": "a", "c": "b"})


def test_a_donor_outside_the_class_set_is_refused():
    # "z" is not a requested class, so the negative side would price "a" against a
    # class the requested side never covers.
    with pytest.raises(ValueError, match="outside the class set"):
        cc.verify_pairing_balance({"a": "z", "b": "c", "c": "b"})


def test_a_single_class_cannot_be_paired():
    with pytest.raises(ValueError, match="at least two classes"):
        cc.verify_pairing_balance({"a": "a"})


def test_the_frozen_queue_pairing_is_balanced_for_every_arm(tmp_path):
    queue = {
        "pre_registration": cg.PRE_REGISTRATION,
        "draw": {"classes_per_arm": 3},
        "arms": {
            "zymctrl": {
                "classes": [
                    {"key": "a", "label": "a", "n_corpus_records": 500, "mismatched_key": "b",
                     "mismatched_label": "b"},
                    {"key": "b", "label": "b", "n_corpus_records": 500, "mismatched_key": "c",
                     "mismatched_label": "c"},
                    {"key": "c", "label": "c", "n_corpus_records": 500, "mismatched_key": "a",
                     "mismatched_label": "a"},
                ]
            }
        },
    }
    queue["digest"] = cg.queue_digest(queue)
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(queue), encoding="utf-8")
    loaded = cg.load_queue(path)
    pairing = cc.pairing_from_queue(loaded, "zymctrl")
    assert cc.verify_pairing_balance(pairing)["balanced"] is True
    with pytest.raises(KeyError):
        cc.pairing_from_queue(loaded, "prollama")


# -------------------------------------------------------------- the per-cell rates


def test_cell_rates_report_both_estimators_and_the_degeneracy_census():
    rows = [_row(id=f"ge_{index}", sequence="MKV", target_profile_hit=True) for index in range(4)]
    rows += [_row(id="ge_x", sequence="MKA", target_profile_hit=False)]
    record = cc.cell_rates(rows)
    assert record["n_attempts"] == 5
    assert record["n_hits"] == 4
    assert record["attempt_rate"] == pytest.approx(0.8)
    # Four attempts share one sequence, so with no declared near-duplicate group the
    # digest is the group and the collapsed rate is one half, not four fifths.
    assert record["grouped_rate"] == pytest.approx(0.5)
    assert record["n_distinct_sequences"] == 2
    assert record["duplication_rate"] == pytest.approx(0.6)
    assert record["largest_exact_duplicate_share"] == pytest.approx(0.8)
    assert record["n_distinct_target_sequences"] == 1


def test_a_degenerate_cell_is_visible_in_the_summary():
    classes = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
    pairing = {cls: classes[(index + 1) % len(classes)] for index, cls in enumerate(classes)}
    rows = []
    for cls in classes:
        for sample in range(10):
            # Every requested attempt of every class is the SAME sequence and hits.
            rows.append(
                _row(id=f"ge_{cls}_req_{sample}", class_key=cls, condition="requested",
                     sequence="MKREPEAT", target_profile_hit=True)
            )
            rows.append(
                _row(id=f"ge_{cls}_mis_{sample}", class_key=cls, condition="mismatched",
                     sequence=f"MK{cls}{sample}", target_profile_hit=False)
            )
    contrast = cc.arm_contrast(rows, arm="zymctrl", pairing=pairing)
    assert contrast["primary"]["mean"] == pytest.approx(1.0)
    degeneracy = contrast["degeneracy"]["requested"]
    assert degeneracy["n_attempts"] == 80
    assert degeneracy["n_distinct_sequences"] == 8  # one per class, all identical
    assert degeneracy["exact_duplication_rate"] == pytest.approx(0.9)
    assert degeneracy["max_single_sequence_share_over_classes"] == pytest.approx(1.0)
    assert degeneracy["classes_with_single_sequence_majority"] == classes


def test_an_attempt_without_a_recognition_outcome_is_refused():
    rows = [_row(id="ge_1", target_profile_hit=None)]
    with pytest.raises(ValueError, match="no target_profile_hit"):
        cc.cell_rates(rows)


def test_a_class_measured_under_one_condition_only_is_refused():
    rows = [
        _row(id="ge_1", class_key="c1", condition="requested"),
        _row(id="ge_2", class_key="c2", condition="requested"),
        _row(id="ge_3", class_key="c2", condition="mismatched"),
    ]
    with pytest.raises(ValueError, match="only one of the two conditions"):
        cc.arm_contrast(rows, arm="zymctrl", pairing={"c1": "c2", "c2": "c1"})


def test_a_measured_class_absent_from_the_frozen_pairing_is_refused():
    classes = [f"c{index}" for index in range(8)]
    pairing = {cls: classes[(index + 1) % 8] for index, cls in enumerate(classes)}
    rows = _cells(classes, requested_hits=5, mismatched_hits=0, pairing=pairing)
    rows += _cells(["extra"], requested_hits=1, mismatched_hits=0, pairing={"extra": "c0"})
    with pytest.raises(ValueError, match="absent from the frozen pairing"):
        cc.arm_contrast(rows, arm="zymctrl", pairing=pairing)


# ------------------------------------------------------------------- the endpoint


def _silent_floor(classes):
    """A floor arm that never carries any class's referent: clause 2 passes freely."""

    return {
        cc.PRIMARY_FLOOR: {
            "n_attempts": 10,
            "any_family_rate": 0.0,
            "n_distinct_sequences": 10,
            "n_near_duplicate_groups": 10,
            "per_class": {
                cls: {"attempt_rate": 0.0, "grouped_rate": 0.0} for cls in classes
            },
        }
    }


def test_the_endpoint_is_a_difference_not_a_rate():
    classes = [f"c{index}" for index in range(8)]
    pairing = {cls: classes[(index + 1) % 8] for index, cls in enumerate(classes)}
    # Both conditions hit nine times in ten: a high rate, no conditional capability.
    rows = _cells(classes, requested_hits=9, mismatched_hits=9, pairing=pairing)
    contrast = cc.arm_contrast(rows, arm="zymctrl", pairing=pairing, resamples=200)
    requested = contrast["per_class"]["c0"]["conditions"]["requested"]["attempt_rate"]
    assert requested == pytest.approx(0.9)
    assert contrast["primary"]["mean"] == pytest.approx(0.0)
    assert contrast["primary"]["ci95"] == [pytest.approx(0.0), pytest.approx(0.0)]
    verdict = cc.arm_verdict(contrast, floor=_silent_floor(classes))
    assert verdict["clause_1_requested_minus_mismatched"] is False
    assert verdict["clause_2_requested_minus_floor"] is True
    assert verdict["outcome"] == "tag_moves_the_distribution_without_selecting_the_requested_class"


def test_fewer_classes_than_the_floor_report_no_interval():
    classes = ["c0", "c1", "c2"]
    pairing = {cls: classes[(index + 1) % 3] for index, cls in enumerate(classes)}
    rows = _cells(classes, requested_hits=8, mismatched_hits=1, pairing=pairing)
    contrast = cc.arm_contrast(rows, arm="zymctrl", pairing=pairing, resamples=200)
    panel = contrast["primary"]
    assert panel["degenerate"] is True
    assert panel["ci95"] is None
    assert cc.arm_verdict(contrast, floor=None)["outcome"] == "not_scored"


def test_the_admitted_support_follows_the_instrument_anchor():
    classes = [f"c{index}" for index in range(9)]
    pairing = {cls: classes[(index + 1) % 9] for index, cls in enumerate(classes)}
    rows = _cells(classes, requested_hits=7, mismatched_hits=0, pairing=pairing)
    referents = {
        cls: {"referent": ("PF00001",), "admitted": cls != "c8", "label": cls,
              "real_rate": 0.9, "random_rate": 0.0}
        for cls in classes
    }
    contrast = cc.arm_contrast(
        rows, arm="zymctrl", pairing=pairing, referents=referents, resamples=200
    )
    assert contrast["panels"]["all/attempt"]["n_classes"] == 9
    assert contrast["panels"]["admitted/attempt"]["n_classes"] == 8
    assert "c8" not in contrast["panels"]["admitted/attempt"]["classes"]
    assert contrast["per_class"]["c8"]["instrument_anchor"]["admitted"] is False


def test_the_floor_clause_uses_per_class_floor_rates_and_the_primary_floor():
    classes = [f"c{index}" for index in range(8)]
    pairing = {cls: classes[(index + 1) % 8] for index, cls in enumerate(classes)}
    rows = _cells(classes, requested_hits=8, mismatched_hits=0, pairing=pairing)
    referents = {cls: {"referent": (f"PF0000{index}",), "admitted": True, "label": cls}
                 for index, cls in enumerate(classes)}
    floor_rows = [
        {
            "id": f"ge_floor_{index}",
            "arm": cc.PRIMARY_FLOOR,
            "condition": "unconditioned_floor",
            "role": "unconditioned_floor",
            "sequence": f"MKFLOOR{index}",
            "sequence_sha256": hashlib.sha256(f"MKFLOOR{index}".encode()).hexdigest(),
            "length": 8,
            "near_duplicate_group": None,
            "any_profile_hit": True,
            # The floor arm only ever carries c0's referent, so the other seven
            # classes must not be charged for it.
            "pfam_families": ["PF00000"],
        }
        for index in range(10)
    ]
    floors = cc.floor_rates(floor_rows, referents=referents)
    assert floors[cc.PRIMARY_FLOOR]["per_class"]["c0"]["attempt_rate"] == pytest.approx(1.0)
    assert floors[cc.PRIMARY_FLOOR]["per_class"]["c1"]["attempt_rate"] == pytest.approx(0.0)
    contrast = cc.arm_contrast(
        rows, arm="zymctrl", pairing=pairing, referents=referents, resamples=200
    )
    verdict = cc.arm_verdict(contrast, floor=floors)
    assert verdict["primary_floor"] == cc.PRIMARY_FLOOR
    assert verdict["clause_2_requested_minus_floor"] is True
    # c0 is charged the floor's 1.0 and the other seven classes 0.0, so the
    # equal-weight mean is (0.8 - 1.0 + 7 * 0.8) / 8.
    assert verdict["requested_minus_floor"][cc.PRIMARY_FLOOR]["mean"] == pytest.approx(0.675)


def test_a_missing_primary_floor_leaves_the_arm_unscored():
    classes = [f"c{index}" for index in range(8)]
    pairing = {cls: classes[(index + 1) % 8] for index, cls in enumerate(classes)}
    rows = _cells(classes, requested_hits=8, mismatched_hits=0, pairing=pairing)
    contrast = cc.arm_contrast(rows, arm="zymctrl", pairing=pairing, resamples=200)
    verdict = cc.arm_verdict(contrast, floor={})
    assert verdict["clause_2_requested_minus_floor"] is None
    assert verdict["outcome"] == "not_scored"


def test_streams_are_summarised_at_the_stream_not_pooled():
    classes = [f"c{index}" for index in range(8)]
    pairing = {cls: classes[(index + 1) % 8] for index, cls in enumerate(classes)}
    streams = {
        name: cc.arm_contrast(
            _cells(classes, requested_hits=hits, mismatched_hits=0, pairing=pairing),
            arm="zymctrl",
            pairing=pairing,
            resamples=200,
        )
        for name, hits in (("s1", 6), ("s2", 8), ("s3", 7))
    }
    combined = cc.combine_streams(streams)
    assert combined["n_streams"] == 3
    assert combined["mean"] == pytest.approx(0.7)
    assert combined["direction_replicated"] is True
    assert combined["interval"][0] < combined["mean"] < combined["interval"][1]


def test_one_stream_reports_no_across_stream_interval():
    classes = [f"c{index}" for index in range(8)]
    pairing = {cls: classes[(index + 1) % 8] for index, cls in enumerate(classes)}
    streams = {
        "only": cc.arm_contrast(
            _cells(classes, requested_hits=6, mismatched_hits=0, pairing=pairing),
            arm="zymctrl",
            pairing=pairing,
            resamples=200,
        )
    }
    combined = cc.combine_streams(streams)
    assert combined["interval"] is None
    assert "single_stream_reason" in combined


def test_no_requested_or_mismatched_attempt_is_refused():
    with pytest.raises(ValueError, match="no requested or mismatched attempt"):
        cc.group_cells([_row(condition="unconditioned_floor")], arm="zymctrl")


def test_read_attempts_refuses_duplicate_identifiers(tmp_path):
    path = tmp_path / "attempts.jsonl"
    path.write_text(
        json.dumps(_row(id="ge_1")) + "\n" + json.dumps(_row(id="ge_1")) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate attempt identifiers"):
        cc.read_attempts(path)


# ------------------------------------------- the one-sided-stream defect (2026-10-08)


def test_a_one_sided_stream_is_refused_and_says_what_to_supply():
    """The defect that failed `condgen_conditional_endpoint_cpu`.

    Each fresh generation cell carries only the mismatched condition, so a stream
    label that collects one cell cannot form the paired difference. The refusal
    must name the conditions it did receive and say that a stream is the set of
    ledgers carrying both sides, or the next caller repeats the same wiring error.
    """

    classes = [f"c{index}" for index in range(8)]
    pairing = {cls: classes[(index + 1) % 8] for index, cls in enumerate(classes)}
    rows = [
        row
        for row in _cells(classes, requested_hits=0, mismatched_hits=3, pairing=pairing)
        if row["condition"] == "mismatched"
    ]
    with pytest.raises(ValueError) as error:
        cc.arm_contrast(rows, arm="zymctrl", pairing=pairing)
    message = str(error.value)
    assert "only one of the two conditions" in message
    assert "['mismatched']" in message
    assert "set of ledgers" in message


def test_the_two_sides_of_a_stream_may_arrive_in_separate_ledgers():
    classes = [f"c{index}" for index in range(8)]
    pairing = {cls: classes[(index + 1) % 8] for index, cls in enumerate(classes)}
    both = _cells(classes, requested_hits=7, mismatched_hits=1, pairing=pairing)
    requested = [row for row in both if row["condition"] == "requested"]
    mismatched = [row for row in both if row["condition"] == "mismatched"]
    contrast = cc.arm_contrast(requested + mismatched, arm="zymctrl",
                               pairing=pairing, resamples=200)
    assert contrast["primary"]["mean"] == pytest.approx(0.6)


# ------------------------------------------------- deriving the target hit (2026-10-08)


def _profile_row(class_key, families, **overrides):
    sequence = f"MK{class_key}{families}"
    row = {
        "id": f"ge_{class_key}_{families}",
        "arm": "zymctrl",
        "class_key": class_key,
        "condition": "requested",
        "sequence": sequence,
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "length": len(sequence),
        "near_duplicate_group": None,
        "profile": {"generated": {"families": list(families), "any_family": bool(families)}},
    }
    row.update(overrides)
    return row


def test_a_family_only_ledger_gets_its_target_hit_derived():
    referents = {
        "c0": {"referent": ("PF00001",), "admitted": True, "label": "c0"},
        "c1": {"referent": ("PF00002", "PF00003"), "admitted": True, "label": "c1"},
    }
    rows = [
        _profile_row("c0", ["PF00001"]),
        _profile_row("c0", ["PF09999"]),
        _profile_row("c1", ["PF00003.7"]),
    ]
    filled = cc.derive_target_hits(rows, referents)
    assert [row["target_profile_hit"] for row in filled] == [True, False, True]
    assert all(row["target_profile_hit_derived"] for row in filled)
    assert filled[0]["target_referent"] == ["PF00001"]


def test_an_existing_target_hit_is_never_overwritten():
    referents = {"c0": {"referent": ("PF00001",), "admitted": True, "label": "c0"}}
    row = _profile_row("c0", ["PF09999"], target_profile_hit=True)
    filled = cc.derive_target_hits([row], referents)
    assert filled[0]["target_profile_hit"] is True
    assert "target_profile_hit_derived" not in filled[0]


def test_an_empty_referent_derives_a_non_hit_and_is_marked():
    referents = {"c0": {"referent": (), "admitted": False, "label": "c0"}}
    filled = cc.derive_target_hits([_profile_row("c0", ["PF00001"])], referents)
    assert filled[0]["target_profile_hit"] is False
    assert filled[0]["target_referent_empty"] is True


def test_a_row_with_no_recognition_and_no_referent_is_left_alone():
    row = {"id": "ge_x", "arm": "zymctrl", "class_key": "zz", "condition": "requested",
           "sequence": "MKV", "sequence_sha256": "x", "length": 3}
    filled = cc.derive_target_hits([row], {})
    assert filled[0].get("target_profile_hit") is None


# ----------------------------------------------------- grouping provenance (2026-10-08)


def test_the_collapsed_estimator_records_which_grouping_it_used():
    declared = [
        _row(id=f"ge_{index}", sequence=f"MK{index}", near_duplicate_group="g0",
             target_profile_hit=True)
        for index in range(4)
    ]
    assert cc.cell_rates(declared)["grouping_source"] == "near_duplicate_group"
    assert cc.cell_rates(declared)["grouped_rate"] == pytest.approx(1.0)
    undeclared = [
        _row(id=f"ge_{index}", sequence=f"MK{index}", target_profile_hit=True)
        for index in range(4)
    ]
    assert cc.cell_rates(undeclared)["grouping_source"] == "exact_sequence_digest"
    mixed = declared[:2] + undeclared[2:]
    assert cc.cell_rates(mixed)["grouping_source"] == "mixed"
