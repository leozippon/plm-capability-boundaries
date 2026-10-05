"""Retained-record invariants; no GPU, scan or model dependencies."""
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

MODULE = Path(__file__).resolve().parents[2] / "scripts/capability/reporting/export_generation_details.py"
spec = importlib.util.spec_from_file_location("generation_details", MODULE)
assert spec and spec.loader
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def raw(ident="a", sequence="ACDE", **kwargs):
    return dict(id=ident, sequence=sequence, arm="model", condition="requested", **kwargs)


def attempt(row):
    return export.make_attempt(row, "campaign", "model__requested", "source.jsonl#L1", "a" * 64)


def test_empty_format_failure_and_unknown_truncation_are_not_dropped():
    empty = attempt(raw(sequence="", format_failure=True, official_compilation_valid=False))
    assert empty["empty"] is True and empty["length"] == 0
    assert empty["format_failure"] is True
    assert empty["official_compilation_valid"] is False
    assert empty["truncated"] is None
    assert empty["pfam_status"] == "not_scanned_empty_or_noncanonical"
    assert attempt(raw(sequence="A" * 400))["truncated"] is None
    assert attempt(raw(decoder_stop="max_new_tokens"))["truncated"] is True
    assert attempt(raw(decoder_stop="eos", native_delimiter_observed=False))["truncated"] is False


def test_pfam_hit_without_domain_is_legitimate_and_empty_not_a_negative_scan():
    row = raw(profile={"generated": {"any_family": True, "best_profile_coverage": None, "families": ["PF00001"], "complete_domain": False}})
    pfam = export.profile(row, "annotation#L1", retained_only=False)
    assert pfam["pfam_any_hit"] is True
    assert pfam["pfam_max_profile_coverage"] is None
    assert pfam["pfam_status"] == "scanned_hit"
    pfam = export.profile(raw(sequence="", any_profile_hit=False), "source#L2")
    assert pfam["pfam_any_hit"] is None
    assert pfam["retained_any_profile_hit"] is False


def test_id_and_exact_sequence_mismatches_are_rejected():
    left = attempt(raw())
    export.exact_join(left, raw())
    with pytest.raises(ValueError, match="ID/sequence"):
        export.exact_join(left, raw(sequence="AAAA"))
    with pytest.raises(ValueError, match="ID/sequence"):
        export.exact_join(left, raw(ident="b"))
    with pytest.raises(ValueError, match="SHA mismatch"):
        attempt(raw(sequence_sha256="0" * 64))


def test_original_support_membership_not_first_n(tmp_path, monkeypatch):
    obj = export.Export(tmp_path)
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    obj.source(source)
    a = obj.add(raw("first"), "historical", "model__requested", "source.jsonl#L1")
    b = obj.add(raw("last"), "historical", "model__requested", "source.jsonl#L2")
    support = tmp_path / export.GATE / "cells/model__requested.jsonl"
    support.parent.mkdir(parents=True)
    support.write_text(json.dumps(dict(arm="model", condition="requested", attempt_id="last", sequences={"generated": "ACDE"})) + "\n")
    manifest = support.parent.parent / "build_manifest.json"
    manifest.write_text(json.dumps({"seed": 1, "cells": [{"cell": "model__requested", "campaign": "historical", "cell_sha256": export.sha(support.read_bytes()), "ledger": "source.jsonl", "ledger_sha256": export.sha(source.read_bytes())}]}))
    monkeypatch.setattr(obj, "gate_oracle", lambda _: None)
    obj.support()
    assert not a["main_comparison_member"]
    assert b["main_comparison_member"]
    assert len(obj.attempts) == 2


def test_reclassification_exact_join_does_not_add_attempt(tmp_path):
    obj = export.Export(tmp_path)
    p = tmp_path / "original.jsonl"
    p.write_text("{}\n")
    obj.source(p)
    original = obj.add(raw(), "historical", "model__requested", "original.jsonl#L1")
    reclassified = raw(format_failure=True, stop_reason="unknown_without_token_trace")
    target = obj.unique(reclassified["id"], "model")
    export.exact_join(target, reclassified)
    target["reclassification_json"] = {k: reclassified[k] for k in export.RAW_FIELDS if k in reclassified}
    assert len(obj.attempts) == 1
    assert original["reclassification_json"]["format_failure"] is True


def test_folding_failures_and_signatures_remain_distinct_and_bad_role_rejected(tmp_path):
    a = attempt(raw())
    index = {"a": [a]}
    folds = []
    for signature, status in [("sig1", "failed"), ("sig2", "ok")]:
        row = dict(attempt_id="a", arm="model", condition="requested", role="generation", sequence_sha256=a["sequence_sha256"], evaluation_signature=signature, status=status)
        export.join_folding(row, index, set())
        folds.append(row)
    assert len(folds) == 2 and folds[0]["status"] == "failed"
    assert {r["evaluation_signature"] for r in folds} == {"sig1", "sig2"}
    bad = dict(folds[0], role="shuffle")
    with pytest.raises(ValueError, match="Bad folding role"):
        export.join_folding(bad, index, set())
    bad = dict(folds[0], role="paired_real_fragment")
    with pytest.raises(ValueError, match="control role/sequence"):
        export.join_folding(bad, index, set())
    fragment = dict(folds[0], role="real_fragment", sequence_sha256="f" * 64)
    export.join_folding(fragment, index, {(a["attempt_key"], "f" * 64)})
    assert fragment["join_status"] == "exact_paired_control_not_generation"
    unmatched = dict(folds[0], attempt_id="outside_scope")
    export.join_folding(unmatched, index, set())
    assert unmatched["join_status"] == "unmatched_generation_retained"
    assert unmatched["attempt_key"] is None


def test_deterministic_gzip_explicit_booleans_null_and_whitelist(tmp_path):
    rows = [{"boolean": False, "unknown": None, "nested": [1, 2]}]
    for name in ["one.gz", "two.gz"]:
        export.csv_gzip(tmp_path / name, rows, ["boolean", "unknown", "nested"])
    assert (tmp_path / "one.gz").read_bytes() == (tmp_path / "two.gz").read_bytes()
    with gzip.open(tmp_path / "one.gz", "rt") as f:
        row = next(csv.DictReader(f))
    assert row == {"boolean": "false", "unknown": "", "nested": "[1,2]"}
    assert export.safe_config({"pod": "sensitive", "token_secret": "sensitive", "seed": 3, "checkpoint": {"files_sha256": {"config.json": "a" * 64}, "environment": "secret"}}) == {"seed": 3, "checkpoint": {"files_sha256": {"config.json": "a" * 64}}}


def test_help_reads_no_source_and_nonempty_output_refused(tmp_path):
    help_run = subprocess.run([sys.executable, str(MODULE), "--help"], capture_output=True, text=True)
    assert help_run.returncode == 0 and "--root" in help_run.stdout
    (tmp_path / "keep").write_text("preserve")
    run = subprocess.run([sys.executable, str(MODULE), "--root", str(tmp_path / "missing"), "--out", str(tmp_path)], capture_output=True, text=True)
    assert run.returncode == 2
    assert "NEW or EMPTY" in run.stderr
    assert (tmp_path / "keep").read_text() == "preserve"
