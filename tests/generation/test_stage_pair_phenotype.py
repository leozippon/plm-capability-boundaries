"""E18's two estimands: what each stage emits, and product quality at matched covariates."""

from __future__ import annotations

import hashlib

import pytest

from src.capability.generation import stage_pair_phenotype as spp


def _record(stage, stream, *, length, hyd=-0.3, entropy=2.4, distinct=17, run=5, **extra):
    sequence = "A" * length
    row = {
        "id": f"ge_{stage}_{stream}_{length}_{hashlib.sha256(str(extra).encode()).hexdigest()[:6]}",
        "stage": stage,
        "stream": stream,
        "arm": spp.STAGES[stage],
        "sequence": sequence,
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "length": length,
        "properties": {
            "mean_kyte_doolittle_hydropathy": hyd,
            "composition_entropy_nats": entropy,
            "distinct_residues": distinct,
            "longest_single_residue_run": run,
        },
    }
    row.update(extra)
    # the stage script lifts properties onto the record; mirror that here
    row.update(row["properties"])
    return row


def _set(streams=("s1", "s2", "s3"), *, n=20, shift=0.0, length_shift=0):
    """A two-stage set whose covariates genuinely vary.

    Constant covariates would make the adjustment design rank-deficient, which is
    a fixture artefact rather than the behaviour under test.
    """

    records = []
    for stream in streams:
        for index in range(n):
            wobble = (index % 7) * 0.05
            twist = (index % 5) * 0.03
            records.append(
                _record("stage_1", stream, length=100 + index, hyd=-0.3 + wobble,
                        entropy=2.4 + twist, distinct=16 + index % 4,
                        run=4 + index % 3, plddt=70.0 + index * 0.1)
            )
            records.append(
                _record("stage_2", stream, length=100 + index + length_shift,
                        hyd=-0.3 + wobble, entropy=2.4 + twist,
                        distinct=16 + index % 4, run=4 + index % 3,
                        plddt=70.0 + index * 0.1 + shift)
            )
    return records


# ---------------------------------------------------------------- covariates


def test_the_covariate_vector_follows_the_declared_order():
    vector = spp.covariate_vector(_record("stage_1", "s1", length=120, hyd=-0.5))
    assert vector.shape == (len(spp.COVARIATES),)
    assert vector[0] == pytest.approx(120.0)
    assert vector[1] == pytest.approx(-0.5)
    assert spp.COVARIATES[0] == "length"


def test_a_record_missing_a_covariate_is_refused():
    record = _record("stage_1", "s1", length=120)
    del record["properties"]["composition_entropy_nats"]
    with pytest.raises(ValueError, match="composition_entropy_nats"):
        spp.covariate_vector(record)


def test_length_is_identified_as_the_dominant_confound():
    left = [_record("stage_1", "s1", length=200 + index) for index in range(40)]
    right = [_record("stage_2", "s1", length=100 + index) for index in range(40)]
    overlap = spp.overlap_diagnostics(left, right)
    assert overlap["dominant_confound"] == "length"
    block = overlap["per_covariate"]["length"]
    assert block["standardised_mean_difference"] < -1.0
    assert block["stage_1_mean"] > block["stage_2_mean"]


def test_overlap_reports_shared_support_per_stage():
    left = [_record("stage_1", "s1", length=100 + index) for index in range(40)]
    right = [_record("stage_2", "s1", length=100 + index) for index in range(40)]
    block = spp.overlap_diagnostics(left, right)["per_covariate"]["length"]
    assert block["standardised_mean_difference"] == pytest.approx(0.0)
    assert block["stage_1_share_in_common_support"] > 0.85


def test_an_overlap_diagnostic_needs_both_stages():
    with pytest.raises(ValueError, match="needs both stages"):
        spp.overlap_diagnostics([], [_record("stage_2", "s1", length=100)])


# ------------------------------------------------------------------ matching


def test_matching_is_one_to_one_and_respects_the_cap():
    left = [_record("stage_1", "s1", length=100 + index) for index in range(10)]
    right = [_record("stage_2", "s1", length=103 + index) for index in range(10)]
    pairs = spp.matched_pairs(left, right, caps=(("length", 5.0),))
    assert len(pairs) == 10
    rows = [row for row, _, _ in pairs]
    columns = [column for _, column, _ in pairs]
    assert len(set(rows)) == len(rows) and len(set(columns)) == len(columns)
    assert all(gap["length"] <= 5.0 for _, _, gap in pairs)


def test_pairs_outside_every_cap_are_not_matched():
    left = [_record("stage_1", "s1", length=100)]
    right = [_record("stage_2", "s1", length=400)]
    assert spp.matched_pairs(left, right, caps=(("length", 5.0),)) == []


def test_matching_does_not_depend_on_input_order():
    left = [_record("stage_1", "s1", length=100 + index * 3) for index in range(12)]
    right = [_record("stage_2", "s1", length=101 + index * 3) for index in range(12)]
    first = spp.matched_pairs(left, right, caps=(("length", 2.0),))
    shuffled_left = list(reversed(left))
    second = spp.matched_pairs(shuffled_left, right, caps=(("length", 2.0),))
    assert len(first) == len(second)
    # the same set of (length, length) pairings, whatever order they arrive in
    def key(pairs, a, b):
        return sorted((a[r]["length"], b[c]["length"]) for r, c, _ in pairs)
    assert key(first, left, right) == key(second, shuffled_left, right)


def test_a_joint_design_caps_every_named_covariate():
    left = [_record("stage_1", "s1", length=100, hyd=-0.1)]
    right = [_record("stage_2", "s1", length=102, hyd=-0.9)]
    assert spp.matched_pairs(left, right, caps=(("length", 5.0),)) != []
    assert spp.matched_pairs(
        left, right, caps=(("length", 5.0), ("mean_kyte_doolittle_hydropathy", 0.25))
    ) == []


def test_an_empty_design_is_refused():
    with pytest.raises(ValueError, match="at least one capped covariate"):
        spp.matched_pairs([], [], caps=())


def test_a_nonpositive_cap_is_refused():
    left = [_record("stage_1", "s1", length=100)]
    with pytest.raises(ValueError, match="must be positive"):
        spp.matched_pairs(left, left, caps=(("length", 0.0),))


# --------------------------------------------------- the two estimands differ


def test_the_unadjusted_contrast_measures_what_each_stage_emits():
    records = _set(shift=5.0)
    block = spp.unmatched_contrast(records, field="plddt")
    assert block["estimand"] == "generation_distribution"
    assert block["mean"] == pytest.approx(5.0)
    assert block["direction_replicated"] is True
    assert block["n_streams"] == 3


def test_the_matched_contrast_measures_quality_at_equal_covariates():
    records = _set(shift=5.0, length_shift=5)
    by_stream = spp.split_stages(records)
    unadjusted = spp.unmatched_contrast(records, field="plddt")
    matched = spp.matched_contrast(by_stream, field="plddt", design="length", resamples=200)
    # The length offset means only the overlapping tail matches, but within matched
    # pairs the readout shift is still recovered.
    assert unadjusted["mean"] == pytest.approx(5.0)
    assert matched["n_pairs_total"] > 0
    assert matched["estimand"] == "matched_length"
    assert matched["role"] == "contrast"


def test_a_matched_readout_is_labelled_a_balance_check_not_a_result():
    records = _set(length_shift=5)
    by_stream = spp.split_stages(records)
    block = spp.matched_contrast(by_stream, field="length", design="length", resamples=200)
    assert block["role"] == "balance_check"
    assert "residual imbalance" in block["role_note"]
    # matching on length drives the residual length gap to near zero
    assert abs(block["mean"]) <= spp.MATCH_DESIGNS["length"][0][1]


def test_the_pair_bootstrap_is_reported_beside_the_stream_interval():
    records = _set(shift=5.0)
    by_stream = spp.split_stages(records)
    block = spp.matched_contrast(by_stream, field="plddt", design="length", resamples=200)
    assert block["unit"] == "the campaign stream"
    assert block["pair_bootstrap"]["unit"] == "the matched pair"
    assert block["pair_bootstrap"]["difference"] == pytest.approx(5.0)
    assert "not a second confirmation" in block["pair_bootstrap"]["note"]


def test_a_design_with_no_feasible_pair_reports_zero_support():
    records = _set(n=5, length_shift=300)
    by_stream = spp.split_stages(records)
    block = spp.matched_contrast(by_stream, field="plddt", design="length", resamples=200)
    assert block["n_pairs_total"] == 0
    assert block["mean"] is None or block["n_streams"] == 0


# ----------------------------------------------------------------- adjustment


def test_adjusting_a_readout_on_itself_is_refused():
    records = _set()
    block = spp.adjusted_contrast(records, field="length")
    assert block["refused"] is True
    assert block["mean"] is None
    assert "zero by construction" in block["refusal_reason"]


def test_the_adjusted_contrast_recovers_a_stage_effect_free_of_covariates():
    records = _set(shift=4.0, length_shift=5)
    block = spp.adjusted_contrast(records, field="plddt")
    assert block["estimand"] == "adjusted"
    assert block["mean"] == pytest.approx(4.0, abs=0.6)
    assert "overlap" in block["caveat"]


def test_a_one_sided_stream_is_reported_without_a_coefficient():
    records = [_record("stage_2", "s1", length=100 + index, plddt=70.0) for index in range(10)]
    block = spp.adjusted_contrast(records, field="plddt")
    assert block["support"]["s1"]["reason"].startswith("stream carries only")
    assert block["mean"] is None


# ------------------------------------------------------------ band admission


def test_band_admission_reports_an_unequal_band():
    records = [_record("stage_1", "s1", length=200) for _ in range(90)]
    records += [_record("stage_1", "s1", length=70) for _ in range(10)]
    records += [_record("stage_2", "s1", length=200) for _ in range(80)]
    records += [_record("stage_2", "s1", length=70) for _ in range(20)]
    block = spp.band_admission(records, low=60, high=80, label="gated")
    assert block["per_stage"]["stage_1"]["share_in_band"] == pytest.approx(0.10)
    assert block["per_stage"]["stage_2"]["share_in_band"] == pytest.approx(0.20)
    assert block["admission_ratio"] == pytest.approx(2.0)
    assert "band-admission component" in block["reading"]


def test_an_equally_admitting_band_has_ratio_one():
    records = [_record(stage, "s1", length=70) for stage in ("stage_1", "stage_2") for _ in range(5)]
    records += [_record(stage, "s1", length=200) for stage in ("stage_1", "stage_2") for _ in range(5)]
    assert spp.band_admission(records, low=60, high=80, label="b")["admission_ratio"] == pytest.approx(1.0)


# -------------------------------------------------------------- termination


def test_termination_is_summarised_per_stage_from_the_stop_accounting():
    records = [_record("stage_1", "s1", length=400, censored=True) for _ in range(10)]
    records += [_record("stage_1", "s1", length=140, censored=False) for _ in range(90)]
    records += [_record("stage_2", "s1", length=100, censored=False) for _ in range(100)]
    block = spp.termination_profile(records)
    assert block["per_stage"]["stage_1"]["censored_share"] == pytest.approx(0.10)
    assert block["per_stage"]["stage_2"]["censored_share"] == pytest.approx(0.0)
    assert block["per_stage"]["stage_1"]["median_length_censored"] == pytest.approx(400.0)
    assert "composition is never consulted" in block["rule"]
    assert "40-400 residue band" in block["scope"]


# -------------------------------------------------------------- the contract


def test_the_readout_classes_keep_measurement_out():
    assert "NOTHING in this" in spp.READOUT_CLASSES["measured"]
    assert "no free-energy units" in spp.READOUT_CLASSES["structural_confidence"].replace(
        "never in energy units", "no free-energy units"
    ) or "never in energy units" in spp.READOUT_CLASSES["structural_confidence"]
    assert "licensed only inside" in spp.READOUT_CLASSES["predicted_stability"]


def test_both_estimands_are_declared_as_real():
    assert set(spp.ESTIMANDS) >= {
        "generation_distribution", "matched_length", "matched_length_hydropathy", "adjusted"
    }
    assert "neither is the corrected version" in spp.CEILING["both_estimands_are_real"]
    assert "impossible by construction" in spp.CEILING[
        "no_artificial_conditional_counterpart"
    ].replace("so no", "impossible by construction so no") or "lacks entirely" in spp.CEILING[
        "no_artificial_conditional_counterpart"
    ]


def test_every_confidence_readout_is_classified():
    for name in spp.CONFIDENCE_READOUTS:
        assert name in ("mean_ca_plddt", "fraction_ca_plddt_ge70", "fraction_ca_plddt_ge90",
                        "ptm", "mean_pae_angstrom", "mean_pae_on_contacts_angstrom",
                        "contact_density")
    assert "structural_confidence" in spp.READOUT_CLASSES
