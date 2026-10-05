"""Small source-shaped checks: no model loading, fitting or resampling."""
from contextlib import ExitStack
import csv
import gzip
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/capability/reporting/export_prediction_details.py"
spec = importlib.util.spec_from_file_location("prediction_details", SCRIPT)
assert spec is not None and spec.loader is not None
details = importlib.util.module_from_spec(spec)
spec.loader.exec_module(details)


def test_exact_mutations_and_content_bound_ids():
    assert details.positions("A1G:D3E", "ACD") == "1:3"
    with pytest.raises(ValueError, match="disagrees"):
        details.positions("A2G", "ACD")
    with pytest.raises(ValueError, match="duplicate"):
        details.positions("A1G:A1C", "ACD")
    sid = details.sample_id("cohort-sha", "assay", "A1G:D3E")
    assert sid == details.sample_id("cohort-sha", "assay", "A1G:D3E")
    assert sid != details.sample_id("other-cohort-sha", "assay", "A1G:D3E")
    assert sid != details.sample_id("cohort-sha", "other-assay", "A1G:D3E")
    assert details.pointer("a/b", "x~y", 0) == "/a~1b/x~0y/0"


def test_vector_alignment_and_unexpected_design_rejection():
    cohort = {"mutants": ["A1G", "D3E"]}
    record = {"assay": "a", "mutants": ["A1G", "D3E"], "B": [0.2, -0.1]}
    details.aligned_predictions(record, cohort, ["B"])
    with pytest.raises(ValueError, match="exact mutation order"):
        details.aligned_predictions(record | {"mutants": ["D3E", "A1G"]}, cohort, ["B"])
    with pytest.raises(ValueError, match="unaligned"):
        details.aligned_predictions(record | {"B": [0.2]}, cohort, ["B"])
    with pytest.raises(ValueError, match="nonfinite"):
        details.aligned_predictions(record | {"B": [0.2, float("nan")]}, cohort, ["B"])
    with pytest.raises(ValueError, match="unexpected prediction design"):
        details.aligned_predictions(record | {"new_design": [0.0, 0.0]}, cohort, ["B"])


def test_saved_outer_and_inner_memberships():
    folds = [
        {"fold": 0, "held_families": [1], "training_families": [2, 3],
         "inner_folds": [{"validation_families": [2]}, {"validation_families": [3]}]},
        {"fold": 1, "held_families": [2, 3], "training_families": [1],
         "inner_folds": [{"validation_families": [1]}]},
    ]
    assert details.outer_mapping(folds, {1, 2, 3}, held="held_families", training="training_families") == {1: 0, 2: 1, 3: 1}
    with pytest.raises(ValueError, match="overlap|partition"):
        details.outer_mapping([{"fold": 0, "held_groups": [1, 2]},
                               {"fold": 1, "held_groups": [2, 3]}], {1, 2, 3})
    with pytest.raises(ValueError, match="inner folds"):
        bad = dict(folds[0], inner_folds=[{"validation_families": [1, 2]}])
        details.outer_mapping([bad, folds[1]], {1, 2, 3}, held="held_families", training="training_families")
    with pytest.raises(ValueError, match="held/training overlap"):
        details.outer_mapping([{"fold": 0, "held_groups": [1], "training_groups": [1, 2]}], {1, 2})


def test_target_and_design_semantics():
    import numpy as np
    from scipy.stats import rankdata

    values = [3.0, 1.0, 1.0, -7.0]
    expected = rankdata(values)
    expected = (expected - expected.mean()) / expected.std()
    assert np.allclose(details.standardized_ranks(values), expected, rtol=0, atol=1e-15)
    assert details.standardized_ranks([4.0, 4.0, 4.0]) == [0.0] * 3
    assert "B=profile PLUS likelihood PLUS sequence" in details.README
    assert "NOT nats" in details.README
    assert "positive is stabilizing" in details.README
    assert set(details.RAW).isdisjoint(details.FITTED)
    assert set(details.PERMUTED) <= set(details.DESIGNS)


def test_missing_values_not_negative_and_deterministic_gzip(tmp_path):
    for name in ("first.gz", "second.gz"):
        with details.gzip_text(tmp_path / name) as stream:
            table = details.Table(stream, ["prediction", "outcome", "member", "held_out"])
            table.row(prediction=None, outcome=None, member=True, held_out=False)
    assert (tmp_path / "first.gz").read_bytes() == (tmp_path / "second.gz").read_bytes()
    with gzip.open(tmp_path / "first.gz", "rt") as stream:
        assert list(csv.DictReader(stream)) == [
            {"prediction": "", "outcome": "", "member": "true", "held_out": "false"}]
    table = details.Table(io.StringIO(), ["prediction"])
    with pytest.raises(ValueError, match="nonfinite"):
        table.row(prediction=float("nan"))


def test_failures_exclusions_and_nulls_are_retained(tmp_path):
    with ExitStack() as stack:
        ex = details.Export(tmp_path, tmp_path, stack)
        ex.failures("source.json", {"excluded": [{"name": "bad", "reason": "censored", "target": None}],
                                   "nested": {"skipped": [{"assay": "long", "reason": "budget"}],
                                              "failure": None}})
        ex.availability_row("cohort", "sha", "model", 20260923, "S", "not_serialized",
                            "source.json", "/per_seed/20260923", samples=2)
    with gzip.open(tmp_path / "provenance.jsonl.gz", "rt") as stream:
        records = [json.loads(line) for line in stream]
    assert records[0]["data"][0] == {"name": "bad", "reason": "censored", "target": None}
    assert records[1]["data"][0]["assay"] == "long"
    assert records[2]["data"] is None
    with gzip.open(tmp_path / "availability.csv.gz", "rt") as stream:
        row = next(csv.DictReader(stream))
    assert row["prediction_status"] == "not_serialized"
    assert "prediction" not in row  # No fabricated zero or fit outcome.


def test_selector_uses_stored_scope_not_filename():
    record = {"status": "complete", "support": {"definition": "common"}, "n_assays": 201,
              "n_families": 163, "n_variants": 25728, "predictions": [{"assay": "x"}]}
    assert details.is_supporting_readout(record)
    assert not details.is_supporting_readout(record | {"n_assays": 40})
    assert not details.is_supporting_readout(record | {"support": {"definition": "native"}})
    assert not details.is_supporting_readout(record | {"predictions": []})


def test_help_does_not_read_root_and_nonempty_output_refused(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True, check=True)
    assert "--root" in result.stdout and "--out" in result.stdout
    (tmp_path / "keep.txt").write_text("do not overwrite")
    with pytest.raises(ValueError, match="new or empty"):
        details.export(tmp_path, tmp_path)
    assert (tmp_path / "keep.txt").read_text() == "do not overwrite"
