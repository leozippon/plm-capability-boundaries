"""The ProLLaMA stage pair: decoding symmetry, outcome-blind selection, keying."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.capability.generation import stage_pair as sp

MANIFEST = Path(sp.__file__).resolve().parents[3] / "configs/generation_replication_manifest.json"


def _cell(**overrides):
    cell = {
        "prompt": "Seq=<",
        "condition": "unconditioned",
        "attempts": 800,
        "max_new_tokens": 400,
        "effective_max_new_tokens": 400,
        "temperature": 0.85,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "batch_size": 8,
        "dtype": "bfloat16",
        "use_cache": True,
        "add_special_tokens": True,
        "seed_rule": "campaign_seed + batch_index",
    }
    cell.update(overrides)
    return cell


def _rows(n, *, prefix="MKV", start=0, extra=None):
    rows = []
    for index in range(start, start + n):
        sequence = prefix + "A" * (40 + index % 60)
        row = {
            "id": f"ge_{prefix}_{index}",
            "sequence": sequence,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "length": len(sequence),
            "valid_aa20": True,
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


# -------------------------------------------------------------------- the symmetry


def test_identical_declarations_are_symmetric():
    record = sp.verify_symmetry({"stage_1": _cell(), "stage_2": _cell()})
    assert record["symmetric"] is True
    assert record["declared"]["prompt"] == "Seq=<"
    assert record["arms"] == {"stage_1": "prollama-stage-1", "stage_2": "prollama"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt", "[Generate by superfamily] Seq=<"),
        ("temperature", 1.0),
        ("top_p", 0.9),
        ("top_k", 40),
        ("attempts", 400),
        ("max_new_tokens", 200),
        ("seed_rule", "campaign_seed"),
        ("dtype", "float32"),
    ],
)
def test_any_decoding_asymmetry_is_refused(field, value):
    with pytest.raises(ValueError, match="not declared at identical decoding"):
        sp.verify_symmetry({"stage_1": _cell(), "stage_2": _cell(**{field: value})})


def test_a_cell_missing_a_declared_field_is_refused():
    stripped = _cell()
    del stripped["temperature"]
    with pytest.raises(ValueError, match="declares no 'temperature'"):
        sp.verify_symmetry({"stage_1": _cell(), "stage_2": stripped})


def test_a_one_sided_pair_is_refused():
    with pytest.raises(ValueError, match="no manifest cell was supplied"):
        sp.verify_symmetry({"stage_2": _cell()})


def test_the_project_manifest_declares_the_two_stages_symmetrically():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cells = {}
    for stage, arm in sp.STAGES.items():
        name = f"{arm}__unconditioned"
        cells[stage] = next(cell for cell in manifest["cells"] if cell["cell"] == name)
    record = sp.verify_symmetry(cells)
    assert record["symmetric"] is True
    assert record["declared"]["prompt"] == "Seq=<"
    assert record["declared"]["attempts"] == 800


# -------------------------------------------------------------------- the selection


def test_selection_reads_only_the_declared_fields():
    # Every record also carries a recognition outcome; the draw must be identical
    # whether that outcome is present or not.
    plain = _rows(40)
    decorated = _rows(40, extra={"profile": {"generated": {"complete_domain": True}},
                                 "target_profile_hit": True})
    first = sp.select_stream(plain, stage="stage_1", stream="s1", per_stream=10)
    second = sp.select_stream(decorated, stage="stage_1", stream="s1", per_stream=10)
    assert [row["id"] for row in first["records"]] == [row["id"] for row in second["records"]]


def test_the_selection_field_set_is_frozen():
    assert sp.SELECTION_FIELDS == frozenset(
        {"id", "sequence", "sequence_sha256", "length", "valid_aa20"}
    )
    assert "profile" not in sp.SELECTION_FIELDS
    assert "target_profile_hit" not in sp.SELECTION_FIELDS


def test_exact_duplicates_collapse_to_one_representative():
    rows = _rows(1) * 5 + _rows(1, start=1)
    # The duplicated record repeats its own identifier, so re-key it to keep the
    # ledger legal while the sequence stays the same.
    rows = [dict(row, id=f"ge_{index}") for index, row in enumerate(rows)]
    block = sp.select_stream(rows, stage="stage_2", stream="s1", per_stream=10)
    assert block["n_eligible"] == 2
    assert block["rejected"]["exact_duplicate_of_a_kept_sequence"] == 4


def test_non_canonical_and_out_of_band_attempts_are_rejected_with_reasons():
    rows = _rows(5)
    rows.append(dict(_rows(1, start=99)[0], id="ge_short", sequence="MKV", length=3,
                     sequence_sha256=hashlib.sha256(b"MKV").hexdigest()))
    rows.append(dict(_rows(1, start=98)[0], id="ge_bad", sequence="MKBZ", length=4,
                     valid_aa20=False,
                     sequence_sha256=hashlib.sha256(b"MKBZ").hexdigest()))
    block = sp.select_stream(rows, stage="stage_1", stream="s1", per_stream=10)
    assert block["rejected"]["outside_length_band"] == 1
    assert block["rejected"]["non_canonical_residue"] == 1
    assert block["n_selected"] == 5
    assert block["shortfall"] == 5
    assert "never widened afterwards" in block["shortfall_reason"]


def test_the_draw_is_reproducible_from_the_seed_alone():
    rows = _rows(50)
    first = sp.select_stream(rows, stage="stage_1", stream="s1", per_stream=12, seed=7)
    second = sp.select_stream(list(reversed(rows)), stage="stage_1", stream="s1",
                              per_stream=12, seed=7)
    assert [row["id"] for row in first["records"]] == [row["id"] for row in second["records"]]
    other = sp.select_stream(rows, stage="stage_1", stream="s2", per_stream=12, seed=7)
    assert [row["id"] for row in other["records"]] != [row["id"] for row in first["records"]]


# ---------------------------------------------------------------------- normalisation


def test_both_retained_ledger_shapes_normalise():
    direct = {"id": "ge_a", "sequence": "MKV", "length": 3, "valid_aa20": True}
    build_cell = {
        "attempt_id": "ge_b",
        "sequences": {"generated": "MKA", "fragment": "AAA", "natural": "CCC"},
    }
    rows = sp.normalise_ledger([direct, build_cell])
    assert [row["id"] for row in rows] == ["ge_a", "ge_b"]
    # Only the model's own product is read; the matched controls are not.
    assert rows[1]["sequence"] == "MKA"
    assert rows[1]["valid_aa20"] is True


def test_canonicality_is_measured_not_trusted():
    with pytest.raises(ValueError, match="says otherwise"):
        sp.normalise_ledger([{"id": "ge_a", "sequence": "MKXZ", "valid_aa20": True}])
    rows = sp.normalise_ledger([{"attempt_id": "ge_b", "sequences": {"generated": "MKXZ"}}])
    assert rows[0]["valid_aa20"] is False


def test_an_unrecognised_ledger_shape_is_refused():
    with pytest.raises(ValueError, match="neither id/sequence nor attempt_id/sequences"):
        sp.normalise_ledger([{"foo": 1}])


# ------------------------------------------------------------------ freezing and keys


def _blocks(per_stream=5, streams=("s1", "s2")):
    # Prefixes stay inside the canonical alphabet: a frozen set must not carry a
    # character no residue property is defined on.
    codes = {"s1": "W", "s2": "Y"}
    blocks = []
    for stream in streams:
        for stage, prefix in (("stage_1", "MKA"), ("stage_2", "MKC")):
            rows = _rows(20, prefix=f"{prefix}{codes[stream]}")
            blocks.append(
                sp.select_stream(rows, stage=stage, stream=stream, per_stream=per_stream)
            )
    return blocks


def test_the_frozen_set_is_keyed_on_the_attempt_identifier():
    frozen = sp.freeze(_blocks(), symmetry=sp.verify_symmetry({"stage_1": _cell(),
                                                              "stage_2": _cell()}))
    assert frozen["keying"]["primary_key"] == "id"
    assert frozen["n_sequences"] == 20
    assert set(frozen["per_stage"]) == {"stage_1", "stage_2"}
    identifiers = [record["id"] for record in frozen["records"]]
    assert len(set(identifiers)) == len(identifiers)
    for record in frozen["records"]:
        assert record["stage"] in sp.STAGES
        assert record["stream"] in {"s1", "s2"}
        assert record["sequence_sha256"] == hashlib.sha256(record["sequence"].encode()).hexdigest()
        assert record["properties"]["length"] == record["length"]


def test_an_unknown_stage_is_refused():
    block = dict(_blocks()[0], stage="stage_3")
    with pytest.raises(ValueError, match="unknown stage"):
        sp.freeze([block], symmetry={})


def test_a_one_sided_frozen_set_is_refused():
    blocks = [block for block in _blocks() if block["stage"] == "stage_2"]
    with pytest.raises(ValueError, match="empty for stages"):
        sp.freeze(blocks, symmetry={})


# --------------------------------------------------------------- sequence properties


def test_sequence_properties_are_finite_and_shares_sum_to_one():
    record = sp.sequence_properties("MKVAACDEFGHIKLMNPQRSTVWY")
    assert record["length"] == 24
    assert record["composition_entropy_nats"] > 0.0
    assert record["longest_single_residue_run"] == 2
    assert sum(record["residue_class_shares"].values()) == pytest.approx(1.0)
    assert record["distinct_residues"] == 21 - 1  # one residue repeats


def test_a_homopolymer_is_detected():
    record = sp.sequence_properties("A" * 30)
    assert record["longest_single_residue_run"] == 30
    assert record["composition_entropy_nats"] == pytest.approx(0.0)
    assert record["distinct_residues"] == 1


def test_an_empty_sequence_has_no_properties():
    with pytest.raises(ValueError, match="non-empty sequence"):
        sp.sequence_properties("")


# ----------------------------------------------------------------- paired contrasts


def test_the_paired_contrast_is_summarised_at_the_stream():
    records = []
    for stream, (one, two) in (("s1", (10.0, 12.0)), ("s2", (10.0, 14.0))):
        records += [{"stage": "stage_1", "stream": stream, "x": one} for _ in range(3)]
        records += [{"stage": "stage_2", "stream": stream, "x": two} for _ in range(3)]
    contrast = sp.paired_property_contrast(records, field="x")
    assert contrast["per_stream"]["s1"]["difference"] == pytest.approx(2.0)
    assert contrast["per_stream"]["s2"]["difference"] == pytest.approx(4.0)
    assert contrast["mean_difference"] == pytest.approx(3.0)
    assert contrast["direction_replicated"] is True
    assert contrast["unit"] == "the campaign stream"


def test_a_stream_with_one_stage_reports_no_difference():
    records = [{"stage": "stage_2", "stream": "s1", "x": 1.0}]
    contrast = sp.paired_property_contrast(records, field="x")
    assert contrast["per_stream"]["s1"]["difference"] is None
    assert contrast["mean_difference"] is None
    assert contrast["direction_replicated"] is None


def test_an_unreplicated_direction_is_reported_as_such():
    records = []
    for stream, (one, two) in (("s1", (10.0, 12.0)), ("s2", (10.0, 8.0))):
        records += [{"stage": "stage_1", "stream": stream, "x": one}]
        records += [{"stage": "stage_2", "stream": stream, "x": two}]
    contrast = sp.paired_property_contrast(records, field="x")
    assert contrast["direction_replicated"] is False


def test_the_impossible_conditional_arm_is_recorded():
    assert "cannot be constructed" in sp.CEILING["conditional_comparison_is_impossible"]
    assert "never reported as stability" in sp.CEILING["confidence_is_not_stability"]
