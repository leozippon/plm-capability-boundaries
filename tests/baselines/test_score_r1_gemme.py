"""The R1 GEMME rung's contract, without running GEMME.

The official runs themselves are exercised in the tool smoke and in the pilot
recorded under `results/extensions/statistical_ladder_20261106`. What is pinned
here is what a panel can silently get wrong: deduplicating a mutation must never
drop a sample, a wild type with one failed repeat must publish no mean at all,
and a resumed checkpoint must not be reused when any input to the score changed.
"""

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "capability" / "baselines"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name.split('.')[-1]}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gemme = _module("gemme_runtime")
rung = _module("score_r1_gemme")

QUERY = "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def export_root(tmp_path: Path, assay_mutations: list[tuple[str, str]] | None = None) -> Path:
    """A miniature export with the schema the real one writes."""

    source = write(tmp_path / "samples.json", json.dumps({"samples": len(assay_mutations or [])}))
    alignment = write(
        tmp_path / "inputs" / "msa" / "q00000.fa",
        f">focus\n{QUERY}\n" + "".join(f">h{index:06d}\n{QUERY}\n" for index in range(3)),
    )
    pairs = assay_mutations or [("assay_a", "A4V"), ("assay_b", "A4V"), ("assay_a", "K2A:Q9F")]
    samples = [
        {
            "sample_id": f"s{index:02d}",
            "assay": assay,
            "cluster": 0,
            "wildtype_id": "q00000",
            "mutation": mutation,
            "variant_index": index,
            "original_index": index,
            "substitutions": [],
            "n_substitutions": mutation.count(":") + 1,
        }
        for index, (assay, mutation) in enumerate(pairs)
    ]
    write(
        tmp_path / "inputs" / "manifest.json",
        json.dumps(
            {
                "schema": rung.B.SCHEMA_MANIFEST,
                "inputs": {"samples": {"path": str(source), "sha256": rung.sha256_file(source)}},
                "counts": {"samples": len(samples), "wildtypes": 1, "single_substitution": 2,
                           "multiple_substitution": 1, "assays": 2},
            }
        ),
    )
    write(
        tmp_path / "inputs" / "support.json",
        json.dumps(
            {
                "schema": rung.B.SCHEMA_SUPPORT,
                "protocol": {},
                "wildtypes": {
                    "q00000": {
                        "wildtype_id": "q00000",
                        "length": len(QUERY),
                        "assays": ["assay_a", "assay_b"],
                        "msa": str(alignment),
                        "msa_sha256": rung.sha256_file(alignment),
                    }
                },
            }
        ),
    )
    write(
        tmp_path / "inputs" / "variant_table.json",
        json.dumps({"schema": rung.SCHEMA_VARIANT_TABLE, "samples": samples}),
    )
    return tmp_path


def test_one_mutation_shared_by_two_assays_is_planned_once_per_sample():
    """Deduplication is of the computation: both sample_ids survive it."""

    entry = {
        "wildtype_id": "q00000",
        "length": len(QUERY),
        "msa": "inputs/msa/q00000.fa",
        "msa_sha256": "0" * 64,
    }
    rows = [
        {"sample_id": "s00", "assay": "assay_a", "mutation": "A4V", "n_substitutions": 1},
        {"sample_id": "s01", "assay": "assay_b", "mutation": "A4V", "n_substitutions": 1},
        {"sample_id": "s02", "assay": "assay_a", "mutation": "K2A:Q9F", "n_substitutions": 2},
    ]
    planned = rung.plan_wildtype(entry, rows)
    assert planned["mutations"] == ["A4V", "K2A:Q9F"]
    assert planned["shared_mutations"] == 1
    assert [sample["sample_id"] for sample in planned["samples_by_mutation"]["A4V"]] == ["s00", "s01"]
    assert planned["samples"] == 3


def test_export_is_verified_and_a_changed_source_stops_the_stage(tmp_path):
    root = export_root(tmp_path)
    assert rung.verify_export(root)["counts"]["samples"] == 3
    write(tmp_path / "samples.json", json.dumps({"samples": 4}))
    with pytest.raises(RuntimeError, match="changed since the export"):
        rung.verify_export(root)


def test_a_changed_alignment_stops_the_stage(tmp_path):
    root = export_root(tmp_path)
    export = rung.verify_export(root)
    write(tmp_path / "inputs" / "msa" / "q00000.fa", f">focus\n{QUERY}\n")
    with pytest.raises(RuntimeError, match="is not the alignment the export hashed"):
        rung.plan(root, export, None)


def test_a_mean_is_only_ever_over_three_complete_runs():
    entry = {
        "wildtype_id": "q00000",
        "length": len(QUERY),
        "mutations": ["A4V", "K2A"],
        "samples_by_mutation": {
            "A4V": [{"sample_id": "s00", "assay": "assay_a", "n_substitutions": 1}],
            "K2A": [{"sample_id": "s01", "assay": "assay_b", "n_substitutions": 1}],
        },
    }
    labels = {"A4V": "A4V", "K2A": "K2A"}
    complete = {"runs": [{"labels": labels, "scores": {"A4V": v, "K2A": 2 * v}} for v in (-1.0, -2.0, -3.0)]}
    aggregated = rung.aggregate(entry, complete)
    assert [variant["gemme_score"] for variant in aggregated["variants"]] == [-2.0, -4.0]
    assert [variant["gemme_sd"] for variant in aggregated["variants"]] == [1.0, 2.0]
    assert all(variant["n_runs"] == 3 for variant in aggregated["variants"])
    # the three raw values travel with the sample, so a mean can be re-derived
    assert aggregated["variants"][0]["gemme_run1"] == -1.0
    assert aggregated["variants"][0]["gemme_run3"] == -3.0

    short = {"runs": [{"labels": labels, "scores": {"A4V": -1.0, "K2A": -2.0}}] * 2}
    with pytest.raises(RuntimeError, match="do not carry 3 values"):
        rung.aggregate(entry, short)


def test_a_wildtype_whose_support_places_nothing_is_flagged_not_silently_zero():
    """Four homolog rows score 0.0 everywhere; that is a finding, not a column of scores."""

    entry = {
        "wildtype_id": "q00025",
        "length": 40,
        "mutations": ["A12Q", "A19H"],
        "samples_by_mutation": {
            "A12Q": [{"sample_id": "s00", "assay": "assay_a", "n_substitutions": 1}],
            "A19H": [{"sample_id": "s01", "assay": "assay_b", "n_substitutions": 1}],
        },
    }
    labels = {"A12Q": "A12Q", "A19H": "A19H"}
    zeros = {"runs": [{"labels": labels, "scores": {"A12Q": 0.0, "A19H": 0.0}} for _ in range(3)]}
    aggregated = rung.aggregate(entry, zeros)
    assert aggregated["degenerate"] is True
    assert aggregated["deficient_note"].startswith("every mutation scored 0.0")
    assert aggregated["sd_max"] == 0.0
    # A constant column has no ranking, so the agreement is undefined rather than 1.0.
    assert aggregated["run_rank_spearman"] == [None, None, None]
    assert aggregated["rank_agreement_undefined"] == 3
    assert aggregated["run_rank_spearman_min"] is None


def test_an_attempt_directory_is_never_reused(tmp_path):
    first = rung.next_attempt(tmp_path / "q00025" / "key")
    second = rung.next_attempt(tmp_path / "q00025" / "key")
    assert first.name == "attempt1" and second.name == "attempt2"
    assert first.is_dir() and second.is_dir()


def test_assembly_emits_no_sample_for_an_incomplete_wildtype(tmp_path):
    root = export_root(tmp_path)
    export = rung.verify_export(root)
    plans = rung.plan(root, export, None)
    args = Namespace(root=str(root), out=str(tmp_path / "gemme"), nseqs=20000, iterations=1)
    log = rung.Log(tmp_path / "logs" / "gemme.log")
    checkpoint = {
        "wildtype_id": "q00000",
        "run_key": "k" * 64,
        "status": "failed",
        "failure": "repeat 2 of 3 failed: GemmeError: docker exited 1",
        "runs": [
            {"index": 1, "status": "completed", "out_dir": "x", "scores": {"A4V": -1.0}},
            {"index": 2, "status": "failed", "out_dir": "y", "error": "GemmeError: docker exited 1"},
        ],
    }
    document = rung.assemble(
        args, log, export, {"image": "i", "image_id": "d"}, {"workers": 1}, plans, {"q00000": checkpoint})
    # every frozen sample keeps its row, so coverage is never faked by omission
    assert len(document["samples"]) == 3
    assert {row["status"] for row in document["samples"]} == {"wildtype_incomplete"}
    assert all(row["gemme_score"] is None for row in document["samples"])
    assert document["counts"]["samples_scored"] == 0
    assert document["counts"]["complete"] is False
    assert document["incomplete"][0]["wildtype_id"] == "q00000"
    assert document["wildtypes"]["q00000"]["status"] == "failed"
    assert "gemme_sd" not in json.dumps(document["wildtypes"])
    assert document["multi_mutant_note"].startswith("official GEMME semantics")
    assert "ADDITIVE" in document["multi_mutant_note"]


def test_a_complete_wildtype_publishes_one_numeric_column_and_a_matching_dump(tmp_path):
    """The ladder joins `gemme_score`; the dump must not disagree with the document."""

    import numpy as np

    root = export_root(tmp_path)
    export = rung.verify_export(root)
    plans = rung.plan(root, export, None)
    args = Namespace(root=str(root), out=str(tmp_path / "gemme"), nseqs=20000, iterations=1)
    log = rung.Log(tmp_path / "logs" / "gemme.log")
    checkpoint = {
        "wildtype_id": "q00000",
        "run_key": "k" * 64,
        "status": "complete",
        "runs": [
            {"index": index, "status": "completed", "out_dir": "x",
             "labels": {"A4V": "A4V", "K2A:Q9F": "K2A,Q9F"},
             "scores": {"A4V": -1.0 * index, "K2A:Q9F": -2.0 * index}}
            for index in (1, 2, 3)
        ],
    }
    document = rung.assemble(
        args, log, export, {"image": "i", "image_id": "d"}, {"workers": 1}, plans, {"q00000": checkpoint})
    assert document["counts"]["complete"] is True
    assert [row["sample_id"] for row in document["samples"]] == ["s00", "s01", "s02"]
    assert {row["status"] for row in document["samples"]} == {"scored"}
    assert [row["gemme_score"] for row in document["samples"]] == [-2.0, -2.0, -4.0]

    with np.load(tmp_path / "gemme" / "scores.npz", allow_pickle=False) as arrays:
        assert list(arrays["sample_id"]) == ["s00", "s01", "s02"]
        assert list(arrays["gemme_score"]) == [-2.0, -2.0, -4.0]
        assert all(arrays["n_runs"] == 3)


def test_run_key_covers_every_input_that_could_move_a_score():
    entry = {"wildtype_id": "q00000", "alignment_sha256": "a" * 64, "mutations": ["A4V", "K2A"]}
    base = rung.run_key(entry, image_digest="i", nseqs=20000, iterations=1)
    assert base == rung.run_key(entry, image_digest="i", nseqs=20000, iterations=1)
    assert base != rung.run_key(entry, image_digest="i2", nseqs=20000, iterations=1)
    assert base != rung.run_key(entry, image_digest="i", nseqs=5000, iterations=1)
    assert base != rung.run_key(entry, image_digest="i", nseqs=20000, iterations=10)
    moved = {**entry, "alignment_sha256": "b" * 64}
    assert base != rung.run_key(moved, image_digest="i", nseqs=20000, iterations=1)
    reordered = {**entry, "mutations": ["K2A", "A4V"]}
    assert base != rung.run_key(reordered, image_digest="i", nseqs=20000, iterations=1)


def test_more_than_four_concurrent_runs_is_refused(tmp_path):
    """Each run holds a fixed-size JVM, so the cap is a refusal rather than a clamp."""

    code = rung.main(["--root", str(tmp_path), "--workers", "5", "--wildtypes", "q00000"])
    assert code == 2