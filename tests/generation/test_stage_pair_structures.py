"""Pairwise geometry derived from folded coordinates, and the refusals around it."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.capability.generation import stage_pair_structures as sps


def _atom(serial, name, residue, number, x, y, z, *, b=90.0, chain="A"):
    # Columns follow the PDB specification exactly: serial 7-11, name 13-16,
    # altLoc 17, resName 18-20, chainID 22, resSeq 23-26, coordinates 31-54,
    # occupancy 55-60, tempFactor 61-66 (one-based).
    return (
        f"ATOM  {serial:5d} {name:<4s} {residue:>3s} {chain}{number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           C  "
    )


def _pdb(positions, *, residue="ALA", chain="A", b=90.0):
    lines = []
    serial = 1
    for index, (x, y, z) in enumerate(positions, start=1):
        lines.append(_atom(serial, "CA", residue, index, x, y, z, b=b, chain=chain))
        serial += 1
        if residue != "GLY":
            lines.append(_atom(serial, "CB", residue, index, x + 0.1, y, z, b=b, chain=chain))
            serial += 1
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _line_positions(n, spacing=4.0):
    return [(index * spacing, 0.0, 0.0) for index in range(n)]


# --------------------------------------------------------------------- parsing


def test_cb_is_preferred_and_glycine_falls_back_to_ca():
    pdb = _pdb(_line_positions(3))
    backbone = sps.parse_backbone(pdb)
    assert backbone["residue_numbers"] == [1, 2, 3]
    assert backbone["coordinates"].shape == (3, 3)
    # CB sits 0.1 A along x from CA, so the parser took CB.
    assert backbone["coordinates"][0][0] == pytest.approx(0.1)
    assert backbone["n_glycine_using_ca"] == 0

    glycine = _pdb(_line_positions(3), residue="GLY")
    backbone = sps.parse_backbone(glycine)
    assert backbone["coordinates"][0][0] == pytest.approx(0.0)
    assert backbone["n_glycine_using_ca"] == 3


def test_only_the_first_chain_is_read():
    pdb = _pdb(_line_positions(3)) + _pdb(_line_positions(5), chain="B")
    backbone = sps.parse_backbone(pdb)
    assert backbone["coordinates"].shape == (3, 3)
    assert backbone["chain"] == "A"


def test_a_pdb_without_atoms_is_refused():
    with pytest.raises(ValueError, match="no ATOM record"):
        sps.parse_backbone("HEADER nothing\nEND\n")


def test_a_residue_with_no_backbone_atom_is_refused():
    pdb = _pdb(_line_positions(2))
    # Add a residue that carries only a sidechain atom.
    pdb += _atom(99, "CG", "LEU", 3, 0.0, 0.0, 0.0) + "\n"
    with pytest.raises(ValueError, match="neither CB nor CA"):
        sps.parse_backbone(pdb)


# ------------------------------------------------------------------- geometry


def test_distances_and_contacts_follow_the_declared_cutoff():
    geometry = sps.pairwise_geometry(_pdb(_line_positions(4, spacing=4.0)), cutoff=8.0)
    assert geometry["n_residues"] == 4
    distance = geometry["cb_distance_angstrom"]
    assert distance.shape == (4, 4)
    assert distance[0][0] == pytest.approx(0.0)
    assert distance[0][1] == pytest.approx(4.0)
    assert distance[0][3] == pytest.approx(12.0)
    contact = geometry["contact_map"]
    assert not contact.diagonal().any()
    # Neighbours at 4 and 8 A are contacts; 12 A is not.
    assert contact[0][1] and contact[0][2] and not contact[0][3]
    assert geometry["n_contacts"] == 5
    assert geometry["contact_cutoff_angstrom"] == pytest.approx(8.0)
    assert geometry["glycine_uses_ca"] is True


def test_a_tighter_cutoff_yields_fewer_contacts():
    loose = sps.pairwise_geometry(_pdb(_line_positions(5)), cutoff=8.0)
    tight = sps.pairwise_geometry(_pdb(_line_positions(5)), cutoff=5.0)
    assert tight["n_contacts"] < loose["n_contacts"]
    assert tight["contact_density"] < loose["contact_density"]


def test_a_nonpositive_cutoff_is_refused():
    with pytest.raises(ValueError, match="cutoff must be positive"):
        sps.pairwise_geometry(_pdb(_line_positions(3)), cutoff=0.0)


def test_b_factors_travel_with_the_geometry():
    geometry = sps.pairwise_geometry(_pdb(_line_positions(3), b=42.5))
    assert np.allclose(geometry["atom_b_factors"], 42.5)


# ---------------------------------------------------------- loading and joining


def _object(root, sha, *, length=4, status="ok", plddt=85.0, ptm=0.8):
    directory = root / "objects" / sha
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(
        json.dumps({"status": status, "sequence_sha256": sha, "length": length}), encoding="utf-8"
    )
    if status != "ok":
        return directory
    (directory / "prediction.pdb").write_text(_pdb(_line_positions(length)), encoding="utf-8")
    np.savez_compressed(
        directory / "prediction.npz",
        ca_plddt_0_100=np.full(length, plddt),
        predicted_aligned_error_angstrom=np.full((length, length), 3.0),
        ptm=np.asarray([ptm]),
        diffusion_sample_index=np.asarray([0]),
        length=np.asarray([length]),
    )
    return directory


def test_an_absent_object_returns_none(tmp_path):
    (tmp_path / "objects").mkdir()
    assert sps.load_structure(tmp_path, "a" * 64) is None


def test_a_not_evaluable_object_is_reported_not_raised(tmp_path):
    _object(tmp_path, "b" * 64, status="not_evaluable")
    found = sps.load_structure(tmp_path, "b" * 64)
    assert found["status"] == "not_evaluable"


def test_an_ok_object_missing_its_prediction_files_is_refused(tmp_path):
    directory = _object(tmp_path, "c" * 64)
    (directory / "prediction.pdb").unlink()
    with pytest.raises(FileNotFoundError, match="missing its prediction files"):
        sps.load_structure(tmp_path, "c" * 64)


def test_a_record_and_its_fold_are_joined(tmp_path):
    sha = "d" * 64
    _object(tmp_path, sha, length=5, plddt=88.0, ptm=0.77)
    record = {
        "id": "ge_1", "stage": "stage_2", "arm": "prollama", "stream": "s1",
        "sequence_sha256": sha, "length": 5, "properties": {"length": 5},
    }
    summary = sps.summarise_record(record, sps.load_structure(tmp_path, sha), cutoff=8.0)
    assert summary["mean_ca_plddt"] == pytest.approx(88.0)
    assert summary["fraction_ca_plddt_ge70"] == pytest.approx(1.0)
    assert summary["fraction_ca_plddt_ge90"] == pytest.approx(0.0)
    assert summary["ptm"] == pytest.approx(0.77)
    assert summary["mean_pae_angstrom"] == pytest.approx(3.0)
    assert summary["mean_pae_on_contacts_angstrom"] == pytest.approx(3.0)
    assert summary["n_contacts"] > 0
    assert summary["diffusion_sample_index"] == 0


def test_a_length_disagreement_between_fold_and_sequence_is_refused(tmp_path):
    sha = "e" * 64
    _object(tmp_path, sha, length=5)
    record = {
        "id": "ge_1", "stage": "stage_1", "arm": "prollama-stage-1", "stream": "s1",
        "sequence_sha256": sha, "length": 7, "properties": {},
    }
    with pytest.raises(ValueError, match="folded structure has 5 residues"):
        sps.summarise_record(record, sps.load_structure(tmp_path, sha), cutoff=8.0)


def test_the_pairwise_archive_carries_distances_pae_and_confidence(tmp_path):
    sha = "f" * 64
    _object(tmp_path, sha, length=6)
    record = {"id": "ge_x", "sequence_sha256": sha, "length": 6}
    written = sps.write_pairwise(
        tmp_path / "pairwise", record, sps.load_structure(tmp_path, sha), cutoff=8.0
    )
    with np.load(written["path"]) as handle:
        assert set(handle.files) == {
            "cb_distance_angstrom", "contact_map", "predicted_aligned_error_angstrom",
            "ca_plddt_0_100", "atom_b_factors", "contact_cutoff_angstrom", "length",
        }
        assert handle["cb_distance_angstrom"].shape == (6, 6)
        assert handle["predicted_aligned_error_angstrom"].shape == (6, 6)
        assert handle["contact_map"].dtype == bool


def test_structure_roots_without_objects_are_refused(tmp_path):
    with pytest.raises(ValueError, match="refused rather than reported as an absence"):
        sps.iter_structure_roots([tmp_path / "nothing"])


def test_a_root_with_objects_is_accepted(tmp_path):
    (tmp_path / "objects").mkdir()
    assert sps.iter_structure_roots([tmp_path, tmp_path / "absent"]) == [tmp_path]


# -------------------------------------------------------------------- summaries


def test_the_stage_summary_reports_per_stream_means():
    rows = [
        {"stage": "stage_1", "stream": "s1", "x": 1.0},
        {"stage": "stage_1", "stream": "s2", "x": 3.0},
        {"stage": "stage_2", "stream": "s1", "x": 5.0},
        {"stage": "stage_2", "stream": "s2", "x": 7.0},
    ]
    summary = sps.stage_summary(rows, field="x")
    assert summary["stage_1"]["mean"] == pytest.approx(2.0)
    assert summary["stage_2"]["per_stream_mean"] == {"s1": 5.0, "s2": 7.0}
    assert summary["stage_1"]["n"] == 2


def test_missing_values_are_skipped_not_imputed():
    rows = [
        {"stage": "stage_1", "stream": "s1", "x": None},
        {"stage": "stage_1", "stream": "s1", "x": 4.0},
    ]
    summary = sps.stage_summary(rows, field="x")
    assert summary["stage_1"]["n"] == 1
    assert summary["stage_1"]["mean"] == pytest.approx(4.0)


def test_the_ceiling_keeps_confidence_apart_from_stability():
    assert "No stability predictor" in sps.CEILING["confidence_is_not_stability"]
    assert "predicted coordinates" in sps.CEILING["contacts_are_predicted"]
    assert "never differenced against a retained ESMFold v1" in sps.CEILING[
        "within_instrument_only"
    ]
