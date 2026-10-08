"""Comparator adequacy: fragment against whole natural record, and the census."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.capability.generation import generation_controls as gcx

GATE = (
    Path(gcx.__file__).resolve().parents[3]
    / "results/R6/generative_control_20260924/gate_endpoints.json"
)


def _control(difference, ci95, *, control_rate=0.5):
    return {
        "control_rate": control_rate,
        "difference": difference,
        "ci95": list(ci95),
        "qualified": True,
        "unit": "frozen_near_duplicate_sequence_group_of_the_attempt_ledger",
    }


def _gate(cells):
    return {"cells": cells}


def _cell(name, *, fragment, natural, model_rate=0.3, endpoint="complete_domain"):
    return {
        "cell": name,
        "arm": name.split("__")[0],
        "condition": name.split("__")[1],
        "n_attempts": 800,
        "n_clusters": 700,
        "distinct_families": {"generated": 10, "fragment": 500, "natural": 510},
        "endpoints": {
            endpoint: {
                "model_rate": model_rate,
                "denominator_rule": "all attempts",
                "controls": {"fragment": fragment, "natural": natural},
            }
        },
    }


# -------------------------------------------------------------- the comparator table


def test_a_verdict_that_survives_the_substitution_is_reported_as_unchanged():
    gate = _gate([
        _cell("a__unconditioned", fragment=_control(0.2, (0.1, 0.3)),
              natural=_control(0.15, (0.05, 0.25))),
    ])
    table = gcx.comparator_table(gate)
    row = table["cells"][0]
    assert row["fragment"]["verdict"] == "positive"
    assert row["natural"]["verdict"] == "positive"
    assert row["verdict_changes"] is False
    assert table["n_cells_whose_verdict_changes"] == 0
    assert table["sign_changes"] == {"n_positive_lost": 0, "n_positive_gained": 0}


def test_a_verdict_that_changes_is_named():
    gate = _gate([
        _cell("a__unconditioned", fragment=_control(-0.02, (-0.05, 0.01)),
              natural=_control(-0.10, (-0.15, -0.05))),
    ])
    table = gcx.comparator_table(gate)
    assert table["cells"][0]["fragment"]["verdict"] == "unresolved"
    assert table["cells"][0]["natural"]["verdict"] == "negative"
    assert table["cells_whose_verdict_changes"] == ["a__unconditioned"]
    assert table["sign_changes"]["n_positive_lost"] == 0


def test_a_lost_positive_is_counted_separately_from_a_changed_verdict():
    gate = _gate([
        _cell("a__unconditioned", fragment=_control(0.2, (0.1, 0.3)),
              natural=_control(0.05, (-0.02, 0.12))),
    ])
    table = gcx.comparator_table(gate)
    assert table["sign_changes"]["n_positive_lost"] == 1
    assert table["sign_changes"]["n_positive_gained"] == 0


def test_the_direction_of_the_comparator_shift_is_declared():
    gate = _gate([
        _cell("a__unconditioned", fragment=_control(0.2, (0.1, 0.3)),
              natural=_control(0.1, (0.0, 0.2))),
        _cell("b__unconditioned", fragment=_control(-0.1, (-0.2, -0.05)),
              natural=_control(-0.2, (-0.3, -0.1))),
    ])
    table = gcx.comparator_table(gate)
    shift = table["fragment_minus_natural_difference"]
    assert shift["mean"] == pytest.approx(0.1)
    assert shift["n_cells_fragment_easier_to_beat"] == 2
    assert "more generous" in shift["reading"]


def test_an_unknown_endpoint_is_refused():
    with pytest.raises(ValueError, match="unknown gate endpoint"):
        gcx.comparator_table(_gate([]), endpoint="folding")


def test_a_gate_without_cells_is_refused():
    with pytest.raises(ValueError, match="carries no cell"):
        gcx.comparator_table(_gate([]))


@pytest.mark.skipif(not GATE.is_file(), reason="the retained gate artefact is not present")
def test_the_retained_gate_carries_both_comparators_on_every_cell():
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    for endpoint in gcx.ENDPOINTS:
        table = gcx.comparator_table(gate, endpoint=endpoint)
        assert table["n_cells"] == 20
        assert sum(table["verdict_counts"]["fragment"].values()) == 20
        assert sum(table["verdict_counts"]["natural"].values()) == 20


# ---------------------------------------------------------------------- the census


def _attempt(sequence, **overrides):
    row = {
        "sequence": sequence,
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "length": len(sequence),
        "decoder_stop": "eos",
        "native_delimiter_observed": True,
        "generated_tokens": 100,
        "effective_max_new_tokens": 400,
        "reference_search_status": "not_searched",
        "reference_identity": None,
    }
    row.update(overrides)
    return row


def test_duplication_is_measured_not_inferred():
    rows = [_attempt("MKV") for _ in range(8)] + [_attempt("MKA"), _attempt("MKC")]
    census = gcx.cell_census(rows)
    assert census["duplication"]["n_distinct_sequences"] == 3
    assert census["duplication"]["exact_duplication_rate"] == pytest.approx(0.7)
    assert census["duplication"]["largest_exact_duplicate_share"] == pytest.approx(0.8)
    assert census["duplication"]["n_sequences_seen_more_than_once"] == 1


def test_termination_separates_the_delimiter_from_the_budget():
    rows = [
        _attempt("MKV", decoder_stop="eos", native_delimiter_observed=True, generated_tokens=50),
        _attempt("MKA", decoder_stop="max_new_tokens", native_delimiter_observed=False,
                 generated_tokens=400),
        _attempt("MKC", decoder_stop="max_new_tokens", native_delimiter_observed=True,
                 generated_tokens=400),
    ]
    census = gcx.cell_census(rows)
    assert census["termination"]["decoder_stop_counts"] == {"eos": 1, "max_new_tokens": 2}
    assert census["termination"]["n_native_delimiter_observed"] == 2
    assert census["termination"]["n_at_token_budget"] == 2


def test_an_unsearched_cell_reports_novelty_as_not_run():
    census = gcx.cell_census([_attempt("MKV"), _attempt("MKA")])
    assert census["novelty_covariate"]["status"] == "not_run"
    assert census["novelty_covariate"]["max_identity_percent"] is None


def test_a_searched_cell_reports_the_identity_covariate():
    rows = [
        _attempt("MKV", reference_identity=42.5, reference_search_status="aligned"),
        _attempt("MKA", reference_identity=97.0, reference_search_status="aligned"),
    ]
    census = gcx.cell_census(rows)
    novelty = census["novelty_covariate"]
    assert novelty["status"] == "measured"
    assert novelty["n_searched"] == 2
    assert novelty["max_identity_percent"] == pytest.approx(97.0)
    assert novelty["n_at_or_above_95_percent"] == 1
    assert "not novelty" in novelty["note"]


def test_recognition_rates_are_read_for_both_cohorts_when_present():
    def profile(generated, fragment):
        return {
            "generated": {"any_family": generated, "complete_domain": generated},
            "fragment": {"any_family": fragment, "complete_domain": fragment},
        }

    rows = [
        _attempt("MKV", profile=profile(True, False)),
        _attempt("MKA", profile=profile(False, True)),
        _attempt("MKC", profile=profile(False, True)),
        _attempt("MKD", profile=profile(False, True)),
    ]
    census = gcx.cell_census(rows)
    assert census["recognition"]["generated"]["any_family_rate"] == pytest.approx(0.25)
    assert census["recognition"]["fragment"]["complete_domain_rate"] == pytest.approx(0.75)


def test_an_empty_cell_has_no_census():
    with pytest.raises(ValueError, match="at least one attempt"):
        gcx.cell_census([])


def test_the_across_cell_summary_names_the_duplicating_cells():
    ledgers = {
        "clean": [_attempt(f"MK{index}" + "A") for index in range(10)],
        "degenerate": [_attempt("MKV") for _ in range(9)] + [_attempt("MKA")],
    }
    # The clean cell's sequences must stay canonical, so re-letter them.
    ledgers["clean"] = [_attempt("MK" + "A" * (index + 1)) for index in range(10)]
    summary = gcx.census_over_cells(ledgers)
    assert summary["n_cells"] == 2
    assert summary["across_cells"]["exact_duplication_rate"]["cells_above_one_tenth"] == [
        "degenerate"
    ]


# ----------------------------------------------------------- the family-matched draw


def test_the_length_band_widens_by_the_declared_tolerance():
    rows = [{"length": 100}, {"length": 200}, {"length": 0}]
    low, high = gcx.length_band(rows, tolerance=0.10)
    assert (low, high) == (90, 220)


def test_a_cell_with_no_non_empty_attempt_has_no_band():
    with pytest.raises(ValueError, match="at least one non-empty attempt"):
        gcx.length_band([{"length": 0}])


def _records():
    return [
        ("sp|A00001|X_Y description", "M" + "A" * 150),
        ("sp|A00002|X_Y description", "M" + "C" * 150),
        ("sp|A00003|X_Y description", "M" + "D" * 600),   # outside the band
        ("sp|A00004|X_Y description", "M" + "E" * 150),   # carries no wanted family
        ("tr|A00005|X_Y description", "M" + "F" * 150),
        ("not a swissprot header", "M" + "G" * 150),
    ]


def test_the_draw_matches_on_family_and_length():
    classes = {"c1": {"referent": ("PF00001",), "admitted": True, "arm": "zymctrl"}}
    members = {"A00001": {"PF00001"}, "A00002": {"PF00001"}, "A00003": {"PF00001"},
               "A00005": {"PF00001"}}
    draw = gcx.family_matched_draw(
        classes=classes,
        length_bands={"c1": (100, 200)},
        records=_records(),
        members=members,
        per_class=10,
    )
    assert draw["per_class"]["c1"]["n_eligible"] == 3
    assert draw["per_class"]["c1"]["n_drawn"] == 3
    assert draw["per_class"]["c1"]["shortfall"] == 7
    assert "never widened" in draw["per_class"]["c1"]["shortfall_reason"]
    accessions = {record["accession"] for record in draw["records"]}
    assert accessions == {"A00001", "A00002", "A00005"}
    assert all(record["arm"] == "zymctrl" for record in draw["records"])
    assert all(record["role"] == "family_matched_natural" for record in draw["records"])


def test_a_class_without_a_referent_is_excluded_and_reported():
    classes = {
        "c1": {"referent": ("PF00001",), "admitted": True, "arm": "zymctrl"},
        "c2": {"referent": (), "admitted": False, "arm": "zymctrl"},
    }
    draw = gcx.family_matched_draw(
        classes=classes,
        length_bands={"c1": (100, 200), "c2": (100, 200)},
        records=_records(),
        members={"A00001": {"PF00001"}},
        per_class=1,
    )
    assert draw["classes_without_referent"] == ["c2"]
    assert "unmeasurable" in draw["classes_without_referent_reason"]
    assert set(draw["per_class"]) == {"c1"}


def test_no_class_with_a_referent_is_refused():
    with pytest.raises(ValueError, match="no supplied class carries a Pfam referent"):
        gcx.family_matched_draw(
            classes={"c1": {"referent": (), "admitted": False}},
            length_bands={"c1": (10, 20)},
            records=_records(),
            members={"A00001": {"PF00001"}},
        )


def test_an_empty_draw_is_refused_not_reported_as_a_zero_rate():
    classes = {"c1": {"referent": ("PF99999",), "admitted": True, "arm": "zymctrl"}}
    with pytest.raises(ValueError, match="refused rather than reported as a zero"):
        gcx.family_matched_draw(
            classes=classes,
            length_bands={"c1": (100, 200)},
            records=_records(),
            # The only carrier of the family is 601 residues long, outside the band.
            members={"A00003": {"PF99999"}},
            per_class=1,
        )


def test_a_class_without_a_length_band_is_refused():
    with pytest.raises(ValueError, match="no generated length band"):
        gcx.family_matched_draw(
            classes={"c1": {"referent": ("PF00001",), "admitted": True}},
            length_bands={},
            records=_records(),
            members={"A00001": {"PF00001"}},
        )


def test_the_draw_is_reproducible_from_its_seed():
    classes = {"c1": {"referent": ("PF00001",), "admitted": True, "arm": "zymctrl"}}
    members = {f"A0000{index}": {"PF00001"} for index in range(1, 6)}
    records = [(f"sp|A0000{index}|X_Y d", "M" + "A" * (100 + index)) for index in range(1, 6)]
    first = gcx.family_matched_draw(classes=classes, length_bands={"c1": (100, 200)},
                                    records=records, members=members, per_class=2, seed=11)
    second = gcx.family_matched_draw(classes=classes, length_bands={"c1": (100, 200)},
                                     records=list(reversed(records)), members=members,
                                     per_class=2, seed=11)
    assert [row["accession"] for row in first["records"]] == [
        row["accession"] for row in second["records"]
    ]


def test_swissprot_headers_are_parsed_and_others_rejected():
    assert gcx.swissprot_accession("sp|P12345|NAME_ORG x") == "P12345"
    assert gcx.swissprot_accession("tr|A0A000|NAME_ORG x") == "A0A000"
    assert gcx.swissprot_accession("gi|12345|ref") is None
    assert gcx.swissprot_accession("plain header") is None


def test_pfam_members_refuses_a_table_that_is_not_the_expected_one(tmp_path):
    path = tmp_path / "pfam.tsv"
    path.write_text("a\tb\tc\n1\t2\t3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not the expected Pfam residue table"):
        gcx.pfam_members(path, ["PF00001"])


def test_pfam_members_reads_only_the_wanted_families(tmp_path):
    path = tmp_path / "pfam.tsv"
    path.write_text(
        "uniprot\tstart\tend\tpfam_id\n"
        "A1\t1\t10\tPF00001.3\n"
        "A1\t20\t30\tPF00002\n"
        "A2\t1\t10\tPF00009\n",
        encoding="utf-8",
    )
    members = gcx.pfam_members(path, ["PF00001", "PF00002"])
    assert members == {"A1": {"PF00001", "PF00002"}}


def test_pfam_members_refuses_a_lookup_with_no_hit(tmp_path):
    path = tmp_path / "pfam.tsv"
    path.write_text("uniprot\tstart\tend\tpfam_id\nA1\t1\t10\tPF00001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="would be empty and is refused"):
        gcx.pfam_members(path, ["PF99999"])


# ------------------------------------------------------------- the family-matched rates


def test_the_family_matched_ceiling_is_reported_per_class():
    draw = {
        "per_class": {"c1": {"referent": ["PF00001"]}},
        "records": [
            {"id": "n1", "class_key": "c1", "length": 150},
            {"id": "n2", "class_key": "c1", "length": 160},
        ],
    }
    recognition = {
        "n1": {"families": ["PF00001"], "any_family": True, "complete_domain": True},
        "n2": {"families": ["PF00002"], "any_family": True, "complete_domain": False},
    }
    rates = gcx.natural_recognition_rates(draw, recognition)
    assert rates["per_class"]["c1"]["any_family_rate"] == pytest.approx(1.0)
    assert rates["per_class"]["c1"]["complete_domain_rate"] == pytest.approx(0.5)
    assert rates["per_class"]["c1"]["target_family_rate"] == pytest.approx(0.5)
    assert rates["n_unscored_records"] == 0
    assert "never a control the model is asked to beat" in rates["ceiling"]


def test_an_unscored_drawn_record_is_named():
    draw = {
        "per_class": {"c1": {"referent": ["PF00001"]}},
        "records": [{"id": "n1", "class_key": "c1", "length": 150},
                    {"id": "n2", "class_key": "c1", "length": 150}],
    }
    rates = gcx.natural_recognition_rates(
        draw, {"n1": {"families": ["PF00001"], "any_family": True, "complete_domain": True}}
    )
    assert rates["n_unscored_records"] == 1
    assert rates["unscored_records"] == ["n2"]


def test_the_ceiling_states_why_unconditional_cells_have_no_family_matched_arm():
    assert "select the comparator on the outcome" in gcx.CEILING[
        "family_matching_needs_a_request"
    ]
    assert "one decoding configuration" in gcx.CEILING["gate_support_is_one_stream"]


# ------------------------------ the family-matched join defect (2026-10-08)


def test_the_draw_completion_json_alone_cannot_price_the_ceiling():
    """The defect that failed `condgen_controls_table_cpu`.

    `run_draw` writes the accounting to the completion JSON and the sequences to a
    sibling JSONL, so a caller that reads only the JSON has no records. That must
    be a named refusal, not a KeyError and not a ceiling computed over nothing.
    """

    accounting_only = {"per_class": {"c1": {"referent": ["PF00001"]}}, "n_drawn": 3}
    with pytest.raises(ValueError, match="keeps only the accounting"):
        gcx.natural_recognition_rates(accounting_only, {"n1": {"families": ["PF00001"]}})


def test_records_may_be_supplied_separately_from_the_draw():
    draw = {"per_class": {"c1": {"referent": ["PF00001"]}}}
    records = [{"id": "n1", "class_key": "c1", "length": 150}]
    recognition = {"n1": {"pfam_families": ["PF00001"], "any_profile_hit": True,
                          "complete_domain": True}}
    rates = gcx.natural_recognition_rates(draw, recognition, records=records)
    assert rates["per_class"]["c1"]["complete_domain_rate"] == pytest.approx(1.0)


def test_both_recognition_spellings_are_accepted():
    """The oracle sidecar and the in-memory result spell the same fields differently.

    Reading one and being handed the other yields all-zero rates: a comparator
    that looks measured and is not.
    """

    sidecar = [{"id": "a", "pfam_families": ["PF1"], "any_profile_hit": True,
                "complete_domain": True, "best_profile_coverage": 0.9}]
    in_memory = [{"id": "a", "families": ["PF1"], "any_family": True,
                  "complete_domain": True, "best_profile_coverage": 0.9}]
    assert gcx.normalise_recognition(sidecar) == gcx.normalise_recognition(in_memory)
    assert gcx.normalise_recognition(sidecar)["a"]["any_family"] is True


def test_a_row_carrying_neither_spelling_is_refused():
    with pytest.raises(ValueError, match="not a recognition record"):
        gcx.normalise_recognition([{"id": "a", "complete_domain": True}])


def test_an_empty_recognition_set_is_refused():
    with pytest.raises(ValueError, match="no recognition record"):
        gcx.normalise_recognition([])


def test_a_sidecar_shaped_recognition_prices_the_ceiling_correctly():
    draw = {"per_class": {"c1": {"referent": ["PF00001"]}}}
    records = [{"id": f"n{index}", "class_key": "c1", "length": 150} for index in range(4)]
    recognition = {
        "n0": {"id": "n0", "pfam_families": ["PF00001"], "any_profile_hit": True,
               "complete_domain": True},
        "n1": {"id": "n1", "pfam_families": ["PF00001"], "any_profile_hit": True,
               "complete_domain": False},
        "n2": {"id": "n2", "pfam_families": ["PF00002"], "any_profile_hit": True,
               "complete_domain": True},
        "n3": {"id": "n3", "pfam_families": [], "any_profile_hit": False,
               "complete_domain": False},
    }
    rates = gcx.natural_recognition_rates(draw, recognition, records=records)
    block = rates["per_class"]["c1"]
    assert block["any_family_rate"] == pytest.approx(0.75)
    assert block["complete_domain_rate"] == pytest.approx(0.5)
    assert block["target_family_rate"] == pytest.approx(0.5)


# ------------------------------------- termination accounting (2026-10-08)


def test_censoring_is_classified_on_the_stop_accounting_not_on_composition():
    """A censored attempt carrying a non-canonical residue is still censored."""

    rows = [
        _attempt("MKV", generated_tokens=400, effective_max_new_tokens=400,
                 decoder_stop="max_new_tokens"),
        _attempt("MKV", generated_tokens=50, effective_max_new_tokens=400,
                 decoder_stop="eos"),
    ]
    block = gcx.cell_census(rows)["termination"]
    assert block["n_at_token_budget"] == 1
    assert block["n_natively_terminated"] == 1
    assert "never consulted" in block["classification_rule"]


def test_termination_falls_back_to_the_stop_reason_when_tokens_are_absent():
    rows = [
        _attempt("MKV", generated_tokens=None, decoder_stop="max_new_tokens"),
        _attempt("MKA", generated_tokens=None, decoder_stop="native_terminal"),
        _attempt("MKC", generated_tokens=None, decoder_stop=None),
    ]
    block = gcx.cell_census(rows)["termination"]
    assert block["n_at_token_budget"] == 1
    assert block["n_natively_terminated"] == 1
    assert block["n_termination_unknown"] == 1


def test_an_immediate_native_stop_is_not_counted_as_a_product():
    rows = [
        _attempt("M", generated_tokens=1, effective_max_new_tokens=400, decoder_stop="eos"),
        _attempt("MK" + "A" * 30, generated_tokens=40, effective_max_new_tokens=400,
                 decoder_stop="eos"),
        _attempt("MK" + "A" * 160, generated_tokens=200, effective_max_new_tokens=400,
                 decoder_stop="eos"),
    ]
    native = gcx.cell_census(rows)["termination"]["native_products"]
    assert native["n_immediate_stop"] == 1
    assert native["n_below_short_threshold"] == 2
    assert "refused to start" in native["note"]


def test_the_native_versus_censored_length_contrast_is_declared_unsupported():
    block = gcx.cell_census([_attempt("MKV")])["termination"]
    contrast = block["length_matched_native_versus_censored"]
    assert contrast["supported"] is False
    assert "barely overlap in length" in contrast["reason"]


def test_the_across_cell_census_reports_both_censoring_scopes():
    ledgers = {
        "small": [
            _attempt("MK" + "A" * 10, generated_tokens=400, effective_max_new_tokens=400,
                     decoder_stop="max_new_tokens")
            for _ in range(8)
        ]
        + [_attempt("MK" + "C" * 10, generated_tokens=5, effective_max_new_tokens=400,
                    decoder_stop="eos")
           for _ in range(2)],
        "large": [
            _attempt("MK" + "D" * (index + 1), generated_tokens=5,
                     effective_max_new_tokens=400, decoder_stop="eos")
            for index in range(20)
        ],
    }
    block = gcx.census_over_cells(ledgers)["across_cells"]["token_budget_censoring"]
    assert block["mean_attempts_per_cell"] == pytest.approx(4.0)
    assert block["mean_share_per_cell"] == pytest.approx(0.4)
    assert block["share_of_attempts_pooled"] == pytest.approx(8 / 30)
    assert "dominated by the conditional cells" in block["scope_note"]


def test_a_checkpoint_whose_native_stops_are_all_immediate_is_named():
    ledgers = {
        "degenerate": [
            _attempt("M", generated_tokens=1, effective_max_new_tokens=400, decoder_stop="eos")
            for _ in range(3)
        ],
        "healthy": [
            _attempt("MK" + "A" * (100 + index), generated_tokens=120,
                     effective_max_new_tokens=400, decoder_stop="eos")
            for index in range(3)
        ],
    }
    block = gcx.census_over_cells(ledgers)["across_cells"]["native_products"]
    assert block["cells_whose_native_terminations_are_all_immediate"] == ["degenerate"]
