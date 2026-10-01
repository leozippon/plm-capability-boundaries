"""Scientific accounting invariants; no model outputs or biological claims."""

import copy
import hashlib

import numpy as np
import pytest

from src.capability.generation.generation_biology_analysis import (
    _joint_bounds, analyze, class_interval, merge_reference_annotations, profile_ledger, sequence_cluster_interval, validate_predictions,
)


def row(key, sequence="ACDEFGHIKLMNPQRS", **kwargs):
    return dict(id=key, sequence=sequence, sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                length=len(sequence), arm="test", class_key="c", condition="requested",
                role="generation", primary_class=True, near_duplicate_group=key,
                target_profile_hit=True, any_profile_hit=True, profile_hit_classes=["c"],
                reference_identity=None, reference_coverage=None, reference_search_status="not_searched",
                phase="main", stratum="16_128", inclusion_probability=1., paired_id=None) | kwargs


def prediction(source, confidence=80., status="ok"):
    result = copy.deepcopy(source)
    result["structure"] = dict(status=status, sequence_sha256=source["sequence_sha256"], length=source["length"])
    if status == "ok":
        result["structure"].update(ca_plddt=[confidence] * source["length"], mean_ca_plddt=confidence,
                                   fraction_ca_plddt_ge70=float(confidence >= 70), ptm=.7)
    return result


def paired(source, confidence=80., shuffled=40., status="ok"):
    control = row(source["id"] + "_shuffle", source["sequence"][::-1], role="composition_shuffle",
                  phase=source["phase"], paired_id=source["id"], inclusion_probability=source["inclusion_probability"])
    return [source, control], [prediction(source, confidence, status), prediction(control, shuffled)]


def test_refuses_partial_nonterminal_or_relabelled_structures():
    source = row("x")
    for outputs in ([], [prediction(source, status="pending")]):
        with pytest.raises(ValueError):
            validate_predictions([source], outputs)
    bad = prediction(source)
    bad["structure"]["sequence_sha256"] = "changed"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_predictions([source], [bad])


def test_metrics_reconstruct_and_reject_scale_or_residue_mismatch():
    source = row("x")
    bad = prediction(source)
    bad["structure"]["mean_ca_plddt"] = .8
    with pytest.raises(ValueError, match="reconstructed"):
        validate_predictions([source], [bad])
    bad = prediction(source)
    bad["structure"]["ca_plddt"].pop()
    with pytest.raises(ValueError, match="coverage"):
        validate_predictions([source], [bad])


def test_unequal_sampling_weights_recover_population_not_sample_mean():
    small = [row(f"short{i}", target_profile_hit=False) for i in range(8)]
    large = [row(f"long{i}", "AC" * 100, stratum="129_256") for i in range(2)]
    selected = [small[0] | {"inclusion_probability": 1 / 8}, large[0] | {"inclusion_probability": 1 / 2}]
    subset, predictions = [], []
    for source, score in zip(selected, [80., 50.]):
        ss, pp = paired(source, score, 40.)
        subset += ss
        predictions += pp
    report = analyze(small + large, subset, predictions, phase="main", calibration={})
    assert report["structural_cells"][0]["paired_mean_ca_plddt_delta"] == pytest.approx(34.)
    assert report["structural_cells"][0]["confidence_event_all_attempt_lower_estimate"] == pytest.approx(.8)
    assert report["arms"]["test"]["interpretation"] == "uncalibrated_predictor"


def test_failure_is_unknown_not_zero_biology_and_denominator_retains_empty():
    source = row("x")
    empty = row("empty", "", stratum="below_minimum_length", target_profile_hit=False, any_profile_hit=False, profile_hit_classes=[])
    subset, predictions = paired(source, status="failed")
    report = analyze([source, empty], subset, predictions, phase="main", calibration={})
    cell = report["structural_cells"][0]
    assert cell["confidence_event_all_attempt_lower_estimate"] == 0
    assert cell["confidence_event_all_attempt_upper_estimate"] == 1
    assert cell["paired_mean_ca_plddt_delta"] is None
    assert report["profile_ledger"][0]["target_profile_rate"] == .5
    assert report["profile_ledger"][0]["n_empty"] == 1


def test_joint_target_structural_estimate_cannot_exceed_known_target_rate():
    population = [row(str(i), target_profile_hit=i == 0) for i in range(10)]
    selected = [population[0] | {"inclusion_probability": .1}]
    scores = validate_predictions(selected, [prediction(selected[0])])
    result = _joint_bounds(population, selected, scores)
    assert result["lower_estimate"] == pytest.approx(.1)
    assert result["upper_estimate"] == pytest.approx(.1)
    result = _joint_bounds(population, [population[1]], {population[1]["id"]: {"status": "failed"}})
    assert result["lower_estimate"] == 0
    assert result["upper_estimate"] == .1


def test_pilot_cannot_admit_generated_sequences_or_overlap_main_naturals():
    source = row("x", phase="pilot")
    subset, predictions = paired(source)
    with pytest.raises(ValueError, match="only natural"):
        analyze([source], subset, predictions, phase="pilot")
    natural = row("natural", role="natural_reference")
    subset, predictions = paired(natural)
    with pytest.raises(ValueError, match="overlap"):
        analyze([natural], subset, predictions, phase="main", calibration={"natural_control_sequence_sha256": [natural["sequence_sha256"]]})


def test_unknown_oracle_is_not_a_failed_oracle_and_class_interval_is_replayable():
    result = profile_ledger([row("natural", role="natural_reference", any_profile_hit=None,
                                 target_profile_hit=None, profile_hit_classes=None)])[0]
    assert result["any_profile_rate"] is None
    assert result["target_profile_rate"] is None
    assert result["confusion_counts_multilabel"]["__profile_unknown__"] == 1
    a = class_interval(list(np.arange(8)), resamples=100)
    assert a == class_interval(list(np.arange(8)), resamples=100)
    assert a["unit"] == "class"
    assert class_interval([1., 2.], resamples=100)["ci95"] is None


def test_single_native_task_uncertainty_keeps_duplicate_groups_and_survey_weights():
    records = [{"near_duplicate_group": str(i), "paired_delta": float(i), "inclusion_probability": .5}
               for i in range(8)]
    records.append({"near_duplicate_group": "0", "paired_delta": 0., "inclusion_probability": .25})
    result = sequence_cluster_interval(records, resamples=100)
    assert result["n_clusters"] == 8
    assert result["mean"] == pytest.approx(2.8)
    assert result["ci95"] is not None
    assert result == sequence_cluster_interval(records, resamples=100)


def test_reference_sidecar_preserves_frozen_identity_and_requires_sequence_match():
    source = row("x", reference_identity=60.)
    sidecar = {"id": "x", "sequence_sha256": source["sequence_sha256"],
               "reference_identity": 60., "reference_coverage": .8}
    updated = merge_reference_annotations([source], [sidecar])[0]
    assert updated["reference_coverage"] == .8 and source["reference_coverage"] is None
    for changed in (sidecar | {"reference_identity": 70.}, sidecar | {"sequence_sha256": "wrong"}):
        with pytest.raises(ValueError):
            merge_reference_annotations([source], [changed])
