"""Sequence-only selection, exact mapping and unchanged-WT rigid geometry."""
import copy
import gzip
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import scripts.capability.reporting.render_homolog_structures as homologs


def test_exact_domain_mapping_preserves_author_numbering():
    residues = [{"residue": aa, "coordinate_id": i + 244} for i, aa in enumerate("WWACDEWW")]
    mapping, missing = homologs.map_af("ACDE", residues)
    assert not missing
    assert [r["coordinate_id"] for r in mapping.values()] == [246, 247, 248, 249]
    for sequence in ("ACE", "WW"):
        with pytest.raises(ValueError, match="uniquely and exactly"):
            homologs.map_af(sequence, residues)


def test_global_affine_alignment_identity_and_bidirectional_coverage():
    exact = homologs.align_sequences("ACDE", "ACDE")
    assert exact["score"] == 4 + 9 + 6 + 5
    assert exact["identity"] == exact["query_coverage"] == exact["candidate_coverage"] == 1
    # A single length-2 terminal gap costs -10.5; terminal gaps are not free.
    gap = homologs.align_sequences("WWACDE", "ACDE")
    assert gap["aligned_candidate"] == "--ACDE"
    assert gap["score"] == exact["score"] - 10.5
    assert gap["pairs"] == [[3, 1], [4, 2], [5, 3], [6, 4]]
    assert gap["identity"] == 1
    assert gap["query_coverage"] == 4/6
    assert gap["candidate_coverage"] == 1
    mismatch = homologs.align_sequences("ACDE", "ACDF")
    assert mismatch["pair_count"] == 4
    assert mismatch["identity"] == .75
    assert np.array_equal(homologs.BLOSUM62, homologs.BLOSUM62.T)
    with pytest.raises(ValueError, match="canonical"):
        homologs.align_sequences("AX", "AC")


def test_selection_ignores_confidence_and_fails_without_required_bands():
    def row(name, identity, confidence=0, coverage: float = 1.):
        return {"id": name, "sequence": name, "identity": identity,
                "query_coverage": coverage, "candidate_coverage": 1,
                "confidence": confidence}
    rows = [row("query", 1), row("close_a", .84, 1), row("close_b", .84, 99),
            row("middle", .49), row("lower", .21), row("very_low", .1), row("short", .85, coverage=.79)]
    first, stats = homologs.select_candidates(copy.deepcopy(rows), "query")
    expected_close = min(("close_a", "close_b"), key=homologs.id_hash)
    assert [r["id"] for r in first] == [expected_close, "middle", "lower"]
    assert stats["bands"] == {"close": 2, "intermediate": 1, "lower": 2}
    changed = list(reversed(copy.deepcopy(rows)))
    for r in changed:
        r["confidence"] = 1000 - r["confidence"]
    second, _ = homologs.select_candidates(changed, "query")
    assert [r["id"] for r in second] == [r["id"] for r in first]
    for absent in ("close", "lower"):
        subset = [r for r in rows if not (r["identity"] >= .7 if absent == "close" else r["identity"] < .3)]
        with pytest.raises(ValueError, match="Required identity band absent"):
            homologs.select_candidates(subset, "query")
    duplicate = row("close_copy", .84)
    duplicate["sequence"] = "close_a"
    duplicate_rows = copy.deepcopy(rows) + [duplicate]
    homologs.select_candidates(duplicate_rows, "query")
    assert sum(r.get("exclusion") == "query or exact sequence duplicate" for r in duplicate_rows) == 2


def test_kabsch_is_proper_rigid_and_uses_all_pairs():
    reference = np.array([[0,0,0], [2,0,0], [0,3,0], [0,0,4], [2,3,4]], float)
    known_rotation = np.array([[0,-1,0], [1,0,0], [0,0,1]], float)
    mobile = reference @ known_rotation + [5,-3,8]
    rotation, translation, rmsd = homologs.kabsch(mobile, reference)
    assert rmsd < 1e-12
    assert np.allclose(mobile @ rotation + translation, reference)
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    reflected = reference * [-1,1,1]
    rotation, _, rmsd = homologs.kabsch(reflected, reference)
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert rmsd > .1  # reflections must not be fitted away
    with pytest.raises(ValueError, match="Degenerate"):
        homologs.kabsch(np.zeros((3,3)), np.zeros((3,3)))


def test_frozen_local_universe_and_rendered_provenance():
    if not homologs.COHORT.is_file() or not homologs.PFAM.is_file():
        pytest.skip("Local cached research inputs are not distributed")
    rows, maps, inputs = homologs.collect_candidates()
    query = next(r for r in rows if r["id"] == homologs.QUERY)
    selected, stats = homologs.select_candidates(rows, query["sequence"])
    assert inputs["retained_pfam_records"] == 26
    assert len(rows) == 26  # 25 cached cohort domains plus independently annotated DVL3
    assert stats["bands"] == {"close": 1, "intermediate": 12, "lower": 11}
    assert [r["id"] for r in selected] == ["Q92997_PF00595_250", "Q9Y6N9_PF00595_207", "Q13009_PF00595_843"]
    assert query["length"] == 93
    assert maps[homologs.QUERY][31]["coordinate_id"] == 276
    assert maps[homologs.QUERY][31]["ca_plddt"] == 70.75
    receipt = homologs.OUT / "selection.json"
    if receipt.exists():
        manifest = json.loads(receipt.read_text())
        assert manifest["selected_ids"] == [homologs.QUERY, *[r["id"] for r in selected]]
        for output in manifest["outputs"].values():
            assert homologs.digest(homologs.ROOT / output["path"]) == output["sha256"]
        assert np.allclose(manifest["camera"]["actual_pymol_view"], manifest["camera"]["pymol_view"], atol=1e-4)
        assert manifest["ce_method"]["parameters"] == homologs.CE_PARAMETERS
        assert manifest["ce_method"]["pymol_version"] == manifest["pymol_version"]
        assert not any((homologs.OUT / name).exists() for name in ("render.log", "tests.log", "resource-checks.json"))
        with gzip.open(homologs.ROOT / query["structure_source"]["path"], "rt") as handle:
            query_atoms = {int(line[6:11]): line for line in handle if line.startswith("ATOM")}
        for row in manifest["candidates"]:
            assert ("confidence_after_selection" in row) == (row["id"] in manifest["selected_ids"])
            if row["id"] not in manifest["selected_ids"]:
                continue
            with gzip.open(homologs.ROOT / row["structure_source"]["path"], "rt") as handle:
                original = [line.rstrip() for line in handle if line.startswith("ATOM") and line[21] == "A"
                            and row["coordinate_start"] <= int(line[22:26]) <= row["coordinate_end"]]
            exported = [line for line in (homologs.OUT / f"{row['id']}.pdb").read_text().splitlines()
                        if line.startswith("ATOM")]
            assert len(original) == len(exported)
            assert all(a[:30] == b[:30] and a[54:] == b[54:] for a, b in zip(original, exported))
            def xyz(lines):
                return np.array([[float(line[a:b]) for a, b in ((30,38), (38,46), (46,54))] for line in lines])
            transform = row["rigid_alignment"]
            assert transform["query_coverage"] == transform["pair_count"] / query["length"]
            assert transform["candidate_coverage"] == transform["pair_count"] / row["length"]
            if row["id"] != homologs.QUERY:
                mobile_atoms = {int(line[6:11]): line for line in original}
                assert transform["pair_count"] == len(transform["source_atom_pairs"]) == 80
                for pair, (i, j) in zip(transform["source_atom_pairs"], transform["pairs"]):
                    for obj, source_atoms, position, mapping in (("query", query_atoms, i, maps[homologs.QUERY]),
                                                               ("mobile", mobile_atoms, j, maps[row["id"]])):
                        atom = pair[obj]
                        source = source_atoms[atom["pdb_atom_id"]]
                        assert atom["pymol_index"] > 0
                        assert source[12:16].strip() == atom["atom_name"] == "CA"
                        assert source[17:20] == atom["residue_name"]
                        assert source[21] == atom["chain"] == "A"
                        assert int(source[22:26]) == atom["coordinate_id"] == mapping[position]["coordinate_id"]
                fitted = np.array([maps[row["id"]][j]["ca"] for i, j in transform["pairs"]]) @ np.array(transform["rotation"]) + transform["translation"]
                reference = np.array([maps[homologs.QUERY][i]["ca"] for i, j in transform["pairs"]])
                rmsd = float(np.sqrt(np.mean(np.sum((fitted - reference)**2, axis=1))))
                assert rmsd == pytest.approx(transform["rmsd_angstrom"])
                assert rmsd == pytest.approx(transform["ce_reported_rmsd_angstrom"], abs=.002)
            if row["id"] == "Q9Y6N9_PF00595_207":
                assert row["sequence_pair_kabsch_diagnostic"]["rmsd_angstrom"] == pytest.approx(9.8466547922)
                assert row["pair_count"] == 78  # global sequence pairs, not the 80 structural pairs
                assert transform["rmsd_angstrom"] == pytest.approx(2.030, abs=.005)
            expected = xyz(original) @ np.array(transform["rotation"]) + transform["translation"]
            assert np.max(np.abs(xyz(exported) - expected)) <= .000501
        labels = [node.text for node in ET.parse(homologs.OUT / "contact-sheet.svg").iter()
                  if node.tag.rsplit("}", 1)[-1] == "text"]
        assert homologs.CAPTION in labels
        assert all(identifier in labels for identifier in manifest["selected_ids"])
