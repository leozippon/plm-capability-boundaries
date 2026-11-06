"""The GEMME baseline's contract, checked without running GEMME.

The official image is exercised for real in the smoke run under
`results/extensions/statistical_ladder_20261106/tool_smoke`, which is where the
numbers come from. What is checked here is the part a smoke cannot check
cheaply: the input contract the official driver depends on (query-first,
ungapped, ascending positions, comma-separated multi-mutants), the mapping from
its one-column-per-mutant output back to the mutants that were asked for, and
the promise that a run which cannot be produced leaves a record saying why and
never a score.

The last one is checked against the real docker CLI with an image that does not
exist, so the failure path is exercised rather than described.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "capability" / "baselines"


def _runtime():
    spec = importlib.util.spec_from_file_location("gemme_runtime", SCRIPTS / "gemme_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gemme = _runtime()

QUERY = "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"
ALIGNMENT = f">query\n{QUERY}\n>homolog_a\nMKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP\n>homolog_b\nMKTAYIAKQRQIAFVKSHFSRQDILDLWVYHTQGYFP\n"


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def staged(tmp_path: Path, alignment: str = ALIGNMENT, mutations: str = "A4V\n") -> dict:
    source = write(tmp_path / "input.fasta", alignment)
    mutants = write(tmp_path / "input.mutations", mutations)
    return gemme.stage_inputs(source, mutants, tmp_path / "work")


def test_multi_mutant_stays_one_entry_in_the_official_list(tmp_path):
    """A multi-mutant is one `-m` line, in position order, never split into singles."""

    result = staged(tmp_path, mutations="K2A:Q9F\nI6W:K2A\n")
    assert [mutant["gemme_mutant"] for mutant in result["mutants"]] == ["K2A,Q9F", "K2A,I6W"]
    assert [mutant["positions"] for mutant in result["mutants"]] == [[2, 9], [2, 6]]
    assert result["mutations"]["multi_mutants"] == 2
    assert result["mutation_path"].read_text() == "K2A,Q9F\nK2A,I6W\n"


def test_single_mutants_are_written_unchanged(tmp_path):
    result = staged(tmp_path, mutations="A4V\n")
    assert result["mutations"]["count"] == 1
    assert result["mutations"]["multi_mutants"] == 0
    assert result["mutation_path"].read_text() == "A4V\n"


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("V3A", "declares V at position 3"),
        ("T99V", "outside the"),
        ("T3T", "is not a substitution"),
        ("T3V:3F", "not like D136R"),
        ("T3V:T3F", "mutates the same position twice"),
        ("T3X", "non-standard residue"),
    ],
)
def test_mutation_contract_failures_are_explicit(tmp_path, mutation, message):
    with pytest.raises(gemme.GemmeError, match=message):
        staged(tmp_path, mutations=mutation)


def test_gapped_query_is_refused_because_positions_would_shift(tmp_path):
    gapped = f">query\nMKTAYIAKQR-QISFVKSHFSRQDILDLWIYHTQGYFP\n>homolog_a\n{QUERY}\n"
    with pytest.raises(gemme.GemmeError, match="contains gaps"):
        staged(tmp_path, alignment=gapped)


def test_staged_alignment_is_one_line_per_record_and_uppercase(tmp_path):
    mkta = "mktayiakqrqisfvkshfsrqdildlwiyhtqgyfp"
    wrapped = f">query\n{QUERY[:20]}\n{QUERY[20:]}\n>lower\n{mkta}\n"
    result = staged(tmp_path, alignment=wrapped)
    assert result["alignment_path"].read_text() == f">query\n{QUERY}\n>lower\n{mkta.upper()}\n"
    assert result["alignment"]["depth"] == 2
    assert result["alignment"]["query_length"] == len(QUERY)


def test_duplicate_headers_are_refused(tmp_path):
    doubled = f">query\n{QUERY}\n>query\n{QUERY}\n"
    with pytest.raises(gemme.GemmeError, match="duplicate FASTA headers"):
        staged(tmp_path, alignment=doubled)


def test_scores_are_mapped_by_order_and_the_label_is_checked(tmp_path):
    mutants = [
        {"mutant": "A4V", "gemme_mutant": "A4V", "positions": [4], "n_sites": 1},
        {"mutant": "K2A:Q9F", "gemme_mutant": "K2A,Q9F", "positions": [2, 9], "n_sites": 2},
    ]
    combined = write(tmp_path / "prot_normPred_evolCombi.txt", '"x"\n"A4V" -1.5\n"K2A,Q9F" -4.25\n')
    scores = gemme.parse_combined(combined, mutants)
    assert [score["gemme_score"] for score in scores] == [-1.5, -4.25]
    assert scores[1]["gemme_mutant"] == "K2A,Q9F"


def test_a_relabelled_row_is_drift_and_not_a_score(tmp_path):
    mutants = [{"mutant": "A4V", "gemme_mutant": "A4V", "positions": [4], "n_sites": 1}]
    combined = write(tmp_path / "prot_normPred_evolCombi.txt", '"x"\n"V4A" -1.5\n')
    with pytest.raises(gemme.GemmeError, match="labelled"):
        gemme.parse_combined(combined, mutants)


@pytest.mark.parametrize("value", ["NA", "NaN", "Inf"])
def test_a_value_that_is_not_a_number_is_refused(tmp_path, value):
    mutants = [{"mutant": "A4V", "gemme_mutant": "A4V", "positions": [4], "n_sites": 1}]
    combined = write(tmp_path / "prot_normPred_evolCombi.txt", f'"x"\n"A4V" {value}\n')
    with pytest.raises(gemme.GemmeError, match="finite value"):
        gemme.parse_combined(combined, mutants)


def test_row_count_must_match_the_mutant_list(tmp_path):
    mutants = [
        {"mutant": "A4V", "gemme_mutant": "A4V", "positions": [4], "n_sites": 1},
        {"mutant": "K2A", "gemme_mutant": "K2A", "positions": [2], "n_sites": 1},
    ]
    combined = write(tmp_path / "prot_normPred_evolCombi.txt", '"x"\n"A4V" -1.5\n')
    with pytest.raises(gemme.GemmeError, match="1 rows for 2 mutants"):
        gemme.parse_combined(combined, mutants)


def test_jet_sequence_selection_is_read_from_the_official_log(tmp_path):
    """The evidence depth is JET's own count, not the depth of the profile rows."""

    write(
        tmp_path / "prot.out",
        "Alignment file:/project/prot/prot_A.fasta\n"
        "Selected 14782 sequences before filtering\n"
        "Filtering sequences (Iteration 1):\n"
        "  -Identity: Removed 0 seqs\n"
        "  -Redundancy: Removed 1446 seqs\n"
        "Selected 13336 after filtering\n"
        "Filtering sequences (Iteration 2):\n"
        "  -Identity: Removed 2834 seqs\n"
        "  -CPU Load: Removed 8604 seqs\n"
        "Selected 1898 after filtering\n"
        "Nb alignments:43 Nb sequences:43\n",
    )
    selection = gemme.parse_jet_selection(tmp_path)
    assert selection["parsed"] is True
    assert (selection["submitted"], selection["retained"]) == (14782, 1898)
    assert selection["filter_iterations"][0]["removed"]["Redundancy"] == 1446
    assert selection["filter_iterations"][1]["removed"]["CPU Load"] == 8604
    assert (selection["alignments"], selection["sequences_in_alignments"]) == (43, 43)


def test_a_missing_jet_log_is_recorded_rather_than_guessed(tmp_path):
    selection = gemme.parse_jet_selection(tmp_path)
    assert selection["parsed"] is False
    assert "reason" in selection


def test_concurrent_runs_never_share_a_container_name():
    """Two wild types start together in one process; docker refuses a duplicate name."""

    names = [gemme.container_name() for _ in range(16)]
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"gemme-\d+-\d+-[0-9a-f]{8}", name) for name in names)


def test_a_non_empty_output_directory_is_refused_and_left_untouched(tmp_path):
    """A reused directory could hand this run another run's official outputs."""

    alignment = write(tmp_path / "alignment.fasta", ALIGNMENT)
    mutations = write(tmp_path / "mutations.txt", "A4V\n")
    out = tmp_path / "out"
    out.mkdir()
    # Another run's official output, sitting where find_combined would find it.
    write(out / "prot_normPred_evolCombi.txt", '"x"\n"A4V" -1.0\n')
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises(gemme.GemmeError, match="is not empty"):
        gemme.execute(alignment, mutations, out, "elodielaine/gemme:gemme", 20000, 1, 120)
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before
    assert not (out / gemme.RECORD).exists()


def test_a_new_output_directory_is_created_before_anything_runs(tmp_path):
    """The refusal above must not become a refusal to work in a fresh directory."""

    out = tmp_path / "nested" / "out"
    with pytest.raises(gemme.GemmeError, match="does not exist"):
        gemme.execute(tmp_path / "missing.fasta", tmp_path / "missing.txt", out, "elodielaine/gemme:gemme", 20000, 1, 120)
    assert out.is_dir()
    # What a run is comes from the image digest and the probed digests; the record's
    # own shape label carries no version tag.
    record = json.loads((out / gemme.RECORD).read_text())
    assert record["schema_version"] == gemme.SCHEMA_VERSION == "gemme_runtime"
    assert record["status"] == "failed"


@pytest.mark.skipif(shutil.which("docker") is None, reason="the official GEMME image needs docker")
def test_an_unavailable_image_fails_with_a_record_and_no_scores(tmp_path):
    """No image, no score: the failure is recorded where the caller will look."""

    alignment = write(tmp_path / "alignment.fasta", ALIGNMENT)
    mutations = write(tmp_path / "mutations.txt", "A4V\nK2A:Q9F\n")
    out = tmp_path / "out"
    with pytest.raises(gemme.GemmeError):
        gemme.execute(alignment, mutations, out, "elodielaine/gemme:missing", 20000, 1, 120)

    record = json.loads((out / gemme.RECORD).read_text())
    assert record["status"] == "failed"
    assert record["failed_at"] == "image"
    assert "missing" in record["error"]
    assert not (out / gemme.SCORES).exists()
    # The interpretation a reader needs is there even when the run is not, and
    # the alignment was already validated before the image was reached.
    assert "higher = less deleterious" in record["score_direction"]
    assert "1-based" in record["matrix_indexing"]
    assert "normalizePredWithNbSeqsPCSelMult" in record["multi_mutant_semantics"]
    assert record["distribution"]["license"].startswith("MIT")
    assert record["alignment"]["depth"] == 3
