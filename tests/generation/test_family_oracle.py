"""The restored family-recognition oracle: digests, the frozen call, and loud failure.

The failure these tests exist for: a broken or absent oracle returns an empty
recognition set, which is indistinguishable in the artefact from a model that
generated nothing recognisable. Every path that could produce that silently must
raise instead.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from src.capability.generation import family_oracle as fo

FROZEN_RUNNER = (
    Path(fo.__file__).resolve().parents[3]
    / "scripts/capability/generation/annotate_generation_replication.py"
)


def _oracle_tree(root: Path, *, contents: dict[str, bytes] | None = None) -> dict[str, str]:
    payload = contents or {name: name.encode() for name in fo.ORACLE_BASENAMES}
    (root / "hmmer/bin").mkdir(parents=True, exist_ok=True)
    (root / "pfam").mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name, blob in payload.items():
        path = (root / fo.HMMSCAN_RELATIVE) if name == "hmmscan" else (root / "pfam" / name)
        path.write_bytes(blob)
        digests[name] = hashlib.sha256(blob).hexdigest()
    return digests


def _manifest(path: Path, digests: dict[str, str]) -> Path:
    path.write_text(
        json.dumps({"oracle_inputs": {f"/authoring/host/{name}": value
                                      for name, value in digests.items()}}),
        encoding="utf-8",
    )
    return path


# ----------------------------------------------------------------- the pinned release


def test_pinned_digests_are_read_by_basename(tmp_path):
    digests = _oracle_tree(tmp_path / "oracle")
    pinned = fo.pinned_digests(_manifest(tmp_path / "m.json", digests))
    assert pinned == digests


def test_a_manifest_without_oracle_inputs_is_refused(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"cells": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="pins no oracle_inputs"):
        fo.pinned_digests(path)


def test_a_manifest_pinning_a_different_file_set_is_refused(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"oracle_inputs": {"/x/hmmscan": "a" * 64}}), encoding="utf-8")
    with pytest.raises(ValueError, match="not the same instrument"):
        fo.pinned_digests(path)


def test_the_project_manifest_pins_the_declared_file_set():
    manifest = Path(fo.__file__).resolve().parents[3] / "configs/generation_replication_manifest.json"
    assert set(fo.pinned_digests(manifest)) == set(fo.ORACLE_BASENAMES)


# ------------------------------------------------------------------ loading the oracle


def test_a_complete_matching_tree_loads(tmp_path):
    root = tmp_path / "oracle"
    digests = _oracle_tree(root)
    oracle = fo.load_oracle(root, pinned=digests)
    assert oracle.hmmscan == root / fo.HMMSCAN_RELATIVE
    record = oracle.record()
    assert record["threshold"] == "gathering"
    assert record["complete_domain_coverage_threshold"] == pytest.approx(0.80)


def test_an_absent_oracle_raises_and_names_the_restore_paths(tmp_path):
    root = tmp_path / "oracle"
    digests = _oracle_tree(root)
    (root / "pfam/Pfam-A.hmm.h3i").unlink()
    with pytest.raises(FileNotFoundError) as error:
        fo.load_oracle(root, pinned=digests)
    message = str(error.value)
    assert "Pfam-A.hmm.h3i" in message
    assert "oracle-inputs.tar.gz" in message
    assert "true-negative yield of zero" in message


def test_a_completely_missing_root_raises(tmp_path):
    digests = {name: "0" * 64 for name in fo.ORACLE_BASENAMES}
    with pytest.raises(FileNotFoundError, match="not restored"):
        fo.load_oracle(tmp_path / "absent", pinned=digests)


def test_a_drifted_pfam_release_is_refused(tmp_path):
    root = tmp_path / "oracle"
    digests = _oracle_tree(root)
    (root / "pfam/Pfam-A.hmm").write_bytes(b"a different release")
    with pytest.raises(ValueError, match="differ from the release the campaign pinned"):
        fo.load_oracle(root, pinned=digests)


# -------------------------------------------------------------------- the frozen call


def _frozen_command_tokens() -> list[str]:
    """The literal hmmscan argument list the frozen replication runner issues."""

    tree = ast.parse(FROZEN_RUNNER.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "scan"
    )
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            if any(
                isinstance(element, ast.Constant) and element.value == "--cut_ga"
                for element in node.value.elts
            ):
                return [
                    element.value if isinstance(element, ast.Constant) else "<expr>"
                    for element in node.value.elts
                ]
    raise AssertionError("the frozen runner's hmmscan command literal was not found")


def test_the_call_matches_the_frozen_replication_runner(tmp_path):
    root = tmp_path / "oracle"
    oracle = fo.load_oracle(root, pinned=_oracle_tree(root))
    command = fo.scan_command(
        oracle, tmp_path / "q.fasta", tmp_path / "q.tbl", tmp_path / "q.domtbl", threads=1
    )
    flags = [token for token in command if token.startswith("-")]
    frozen = [token for token in _frozen_command_tokens() if isinstance(token, str) and token.startswith("-")]
    assert flags == frozen, (
        "the recognition call diverged from the frozen replication runner's; a rate "
        "computed here would not be the same measurement"
    )
    assert "--cut_ga" in command and "--noali" in command
    assert "-E" not in command


def test_a_nonpositive_thread_count_is_refused(tmp_path):
    root = tmp_path / "oracle"
    oracle = fo.load_oracle(root, pinned=_oracle_tree(root))
    with pytest.raises(ValueError, match="positive thread count"):
        fo.scan_command(oracle, tmp_path / "q.fa", tmp_path / "a", tmp_path / "b", threads=0)


# ------------------------------------------------------------------- table collection


_TBL = """#  target name  accession  query name  accession
PF00578.24           PF00578.24 q0  -  1.2e-30  104.1
PF08534.13           PF08534.13 q0  -  3.1e-10   38.0
"""

_DOMTBL = (
    "# target  acc  tlen  query  qacc  qlen  E  score  bias  #  of  cE  iE  dscore  dbias  "
    "hmmfrom  hmmto  alifrom  alito  envfrom  envto  acc description\n"
    "AhpC-TSA PF00578.24 120 q0 - 160 1e-30 104.1 0.0 1 1 1e-33 1e-30 100.0 0.0 "
    "2 100 5 103 4 104 0.95 x\n"
)


def _tables(tmp_path: Path, *, tbl: str = _TBL, domtbl: str = _DOMTBL) -> Path:
    stem = tmp_path / "shard_00000"
    stem.with_suffix(".tbl").write_text(tbl, encoding="utf-8")
    stem.with_suffix(".domtbl").write_text(domtbl, encoding="utf-8")
    return stem


def test_collect_reads_families_and_the_best_profile_coverage(tmp_path):
    collected = fo._collect([_tables(tmp_path)])
    assert collected["q0"]["families"] == ["PF00578", "PF08534"]
    assert collected["q0"]["any_family"] is True
    # hmm 2..100 of a 120-residue model is 99/120 = 0.825, which clears 0.80.
    assert collected["q0"]["best_profile_coverage"] == pytest.approx(99 / 120)
    assert collected["q0"]["complete_domain"] is True


def test_a_family_below_the_coverage_threshold_is_not_a_complete_domain(tmp_path):
    shortened = _DOMTBL.replace(" 2 100 5 103", " 2 50 5 53")
    collected = fo._collect([_tables(tmp_path, domtbl=shortened)])
    assert collected["q0"]["any_family"] is True
    assert collected["q0"]["complete_domain"] is False


def test_a_domain_row_the_sequence_table_does_not_carry_is_refused(tmp_path):
    rogue = _DOMTBL.replace("PF00578.24", "PF99999.1")
    with pytest.raises(RuntimeError, match="not the same scan"):
        fo._collect([_tables(tmp_path, domtbl=rogue)])


# ----------------------------------------------------------------- the empty-set guard


def test_recognising_nothing_is_refused_not_reported_as_zero(tmp_path):
    root = tmp_path / "oracle"
    oracle = fo.load_oracle(root, pinned=_oracle_tree(root))
    with pytest.raises(ValueError, match="An empty recognition set is refused"):
        fo.recognise(oracle, {}, workspace=tmp_path / "w")
    with pytest.raises(ValueError, match="An empty recognition set is refused"):
        fo.recognise(oracle, {"q0": ""}, workspace=tmp_path / "w")


def test_an_invalid_shard_geometry_is_refused(tmp_path):
    root = tmp_path / "oracle"
    oracle = fo.load_oracle(root, pinned=_oracle_tree(root))
    with pytest.raises(ValueError, match="positive shard size"):
        fo.recognise(oracle, {"q0": "MKV"}, workspace=tmp_path / "w", shard_size=0)


# ------------------------------------------------------------- the recognition controls


def test_the_shipped_controls_declare_both_positives_and_negatives():
    controls = fo.load_positive_controls()
    assert any(entry["required_families"] for entry in controls)
    assert any(not entry["required_families"] for entry in controls)
    for entry in controls:
        assert set(entry["sequence"]) <= set("ACDEFGHIKLMNPQRSTVWY")


def test_a_positives_only_control_file_is_refused(tmp_path):
    path = tmp_path / "controls.json"
    path.write_text(
        json.dumps({"controls": [{"id": "a", "sequence": "MKV", "required_families": ["PF1"]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="both positives and negatives"):
        fo.load_positive_controls(path)


def _stub_recognise(monkeypatch, outcomes):
    def fake(oracle, sequences, **kwargs):
        hits = {
            name: {
                "families": sorted(outcomes.get(name, [])),
                "any_family": bool(outcomes.get(name)),
                "best_profile_coverage": 1.0 if outcomes.get(name) else None,
                "complete_domain": bool(outcomes.get(name)),
            }
            for name in sequences
            if outcomes.get(name)
        }
        return hits, {"n_requested": len(sequences), "n_searched": len(sequences)}

    monkeypatch.setattr(fo, "recognise", fake)


CONTROLS = [
    {"id": "p1", "sequence": "MKV", "required_families": ["PF00001"]},
    {"id": "p2", "sequence": "MKA", "required_families": ["PF00002", "PF00003"]},
    {"id": "n1", "sequence": "MKG", "required_families": []},
]


def test_a_faithful_oracle_passes_its_controls(tmp_path, monkeypatch):
    _stub_recognise(monkeypatch, {"p1": ["PF00001"], "p2": ["PF00002", "PF00003"]})
    result = fo.run_positive_control(object(), workspace=tmp_path, controls=CONTROLS)
    assert result["passed"] is True
    assert result["n_positive"] == 2 and result["n_negative"] == 1


def test_an_oracle_that_recognises_nothing_fails_loudly(tmp_path, monkeypatch):
    _stub_recognise(monkeypatch, {})
    with pytest.raises(RuntimeError) as error:
        fo.run_positive_control(object(), workspace=tmp_path, controls=CONTROLS)
    assert "failed its own controls" in str(error.value)
    assert "PF00001" in str(error.value)


def test_an_oracle_that_recognises_everything_fails_loudly(tmp_path, monkeypatch):
    _stub_recognise(
        monkeypatch,
        {"p1": ["PF00001"], "p2": ["PF00002", "PF00003"], "n1": ["PF00009"]},
    )
    with pytest.raises(RuntimeError, match="failed its own controls"):
        fo.run_positive_control(object(), workspace=tmp_path, controls=CONTROLS)


def test_a_partially_recovering_oracle_fails_loudly(tmp_path, monkeypatch):
    _stub_recognise(monkeypatch, {"p1": ["PF00001"], "p2": ["PF00002"]})
    with pytest.raises(RuntimeError, match="PF00003"):
        fo.run_positive_control(object(), workspace=tmp_path, controls=CONTROLS)


def test_duplicate_control_identifiers_are_refused(tmp_path, monkeypatch):
    _stub_recognise(monkeypatch, {"p1": ["PF00001"]})
    duplicated = [CONTROLS[0], dict(CONTROLS[0]), CONTROLS[2]]
    with pytest.raises(ValueError, match="duplicate identifiers"):
        fo.run_positive_control(object(), workspace=tmp_path, controls=duplicated)


# ------------------------------------------------------------------ the referent join


def test_target_hit_is_a_referent_intersection():
    assert fo.target_hit(["PF00578"], ["PF00578", "PF08534"]) is True
    assert fo.target_hit(["PF00578.24"], ["PF00578"]) is True
    assert fo.target_hit(["PF12345"], ["PF00578"]) is False
    assert fo.target_hit([], ["PF00578"]) is False


def test_an_empty_referent_cannot_score_a_class():
    with pytest.raises(ValueError, match="unmeasurable class"):
        fo.target_hit(["PF00578"], [])


def test_class_referents_prefer_the_full_class_block():
    anchors = {
        "arm": "zymctrl",
        "classes": {
            "1.1.1.1": {"referent": ["PF1"], "admitted": True, "label": "1.1.1.1",
                        "real_rate": 0.9, "random_rate": 0.0},
            "2.2.2.2": {"referent": [], "admitted": False, "label": "2.2.2.2"},
        },
        "instrument_anchors": {"1.1.1.1": {"referent": ["PF1"], "admitted": True}},
    }
    referents = fo.class_referents(anchors)
    assert set(referents) == {"1.1.1.1", "2.2.2.2"}
    assert referents["2.2.2.2"]["admitted"] is False


def test_class_referents_refuse_an_artefact_without_referents():
    with pytest.raises(ValueError, match="no class referent"):
        fo.class_referents({"arm": "zymctrl", "classes": {"a": {"label": "a"}}})
