"""Small synthetic receipts exercise the actual evidence verifier; no real folds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.capability.reporting import render_generated_structures as renderer
from src.capability.generation import structure_evidence as se


def pdb_text(sequence: str, confidence: np.ndarray) -> str:
    reverse = {v: k for k, v in renderer.AA3.items()}
    lines = ["HEADER    SYNTHETIC ESMFOLD2 TEST", "REMARK    RAW B FACTORS 0-1"]
    serial = 1
    for i, (aa, value) in enumerate(zip(sequence, confidence), 1):
        for name, offset, element in (("N", -.8, "N"), ("CA", 0., "C"), ("C", .8, "C"), ("O", 1.1, "O")):
            x, y, z = i * 3.4 + offset, (i % 2) * 1.2, (i % 3) * .5
            lines.append(f"ATOM  {serial:5d} {name:^4s} {reverse[aa]:3s} A{i:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}{1.:6.2f}{value:6.2f}          {element:>2s}")
            serial += 1
    return "\n".join(lines) + "\nTER\nEND\n"


@pytest.fixture
def tree(tmp_path, monkeypatch):
    assets, predictions = tmp_path / "assets", tmp_path / "predictions"
    assets.mkdir()
    predictions.mkdir()
    rows, selections = [], []
    for i, sequence in enumerate(("ACDE", "FGHI", "KLMN", "PQRS", "TVWY", "ARND")):
        pair = i // 2 + 1
        identifier = f"test-{pair}__" + ("generated" if i % 2 == 0 else "real-fragment")
        rows.append({"id": identifier, "pair_id": f"test-{pair}", "arm": f"arm-{pair}",
                     "role": "generation" if i % 2 == 0 else "real_fragment",
                     "sequence": sequence, "length": 4, "sequence_sha256": se.sequence_digest(sequence)})
        if i % 2 == 0:
            selections.append({"generated_id": identifier, "real_fragment_id": f"test-{pair}__real-fragment"})
    (assets / "cohort.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    renderer.write_json(assets / "source-provenance.json", {"generators": [
        {"cell": {"arm": f"arm-{i}"}, "snapshot": "synthetic", "torch": "test"} for i in range(1, 4)]})
    selection = {"selections": selections, "protocol": {"test": True},
                 "files_sha256": {"source-provenance.json": renderer.sha256_file(assets / "source-provenance.json")}}
    renderer.write_json(assets / "selection.json", selection)
    hashes = {name: renderer.sha256_file(assets / name) for name in renderer.FROZEN_HASHES}
    monkeypatch.setattr(renderer, "FROZEN_HASHES", hashes)
    plan = {"frozen_input": {"sha256": hashes["cohort.jsonl"], "selection_sha256": hashes["selection.json"], "sequence_records": 6},
            "local_checkpoint": {"model_id": se.PREDICTOR_ID, "model_files_sha256": {"config.json": "synthetic"},
                                 "num_loops": 3, "num_sampling_steps": 14, "num_diffusion_samples": 32},
            "settings": {**renderer.SETTINGS, "precision": renderer.PRECISION},
            "source_sha256": {"src/capability/generation/structure_evidence.py": "test-module",
                              "scripts/capability/generation/run_structure_evidence.py": "test-runner"},
            "local_runtime_import_verified": {"python": "test", "torch": "test", "transformers": "test"}}
    renderer.write_json(assets / "prediction-plan.json", plan)
    contract = {**renderer.SETTINGS, "schema_version": se.SCHEMA_VERSION, "model_id": se.PREDICTOR_ID,
                "predictor_kind": se.PREDICTOR_KIND, "never_truncate": True,
                "precision": renderer.PRECISION + "; single sequence; no MSA",
                "sample_selection": "highest pTM, matching infer_protein_as_pdb",
                "model_files_sha256": plan["local_checkpoint"]["model_files_sha256"],
                "code_sha256": {"module": "test-module", "runner": "test-runner"},
                "python": "test", "versions": {"torch": "test", "transformers": "test", "numpy": "test"}}
    signature = se.contract_signature(contract)
    worker = {"contract": contract, "evaluation_signature": signature, "cohort_sha256": hashes["cohort.jsonl"],
              "shard_index": 0, "num_shards": 1, "unique_sequences_assigned": 6, "finished_utc": "test", "gpu_name": "NVIDIA L20"}
    renderer.write_json(predictions / "worker-000-of-001.json", worker)
    for i, row in enumerate(rows):
        directory = predictions / "objects" / row["sequence_sha256"]
        ca = np.array([.0249, .3751, .701, .945]) * 100
        arrays = {"ca_plddt_0_100": ca, "predicted_aligned_error_angstrom": np.full((4, 4), 7.),
                  "ptm": np.array([.41]), "diffusion_sample_index": np.array([3]), "length": np.array([4])}
        result = {"evaluation_signature": signature, "sequence_sha256": row["sequence_sha256"],
                  "status": "ok", "length": 4, "evaluator_truncated": False, "predictor": se.PREDICTOR_ID,
                  "mean_ca_plddt": ca.mean(), "ca_plddt": ca.tolist(), "fraction_ca_plddt_ge70": .5,
                  "predicted_confidence_event": False, "mean_pae_angstrom": 7., "ptm": .41,
                  "diffusion_sample_index": 3, "diffusion_sample_splits": [16, 16] if i == 1 else [32],
                  "files_sha256": se.save_prediction(directory, arrays, pdb_text(row["sequence"], ca / 100))}
        renderer.write_json(directory / "result.json", result)
    refresh(rows, predictions)
    return assets, predictions, rows, plan


def refresh(rows, predictions):
    summary = se.write_index(rows, predictions, se.tree_signature(predictions), shard_index=0, num_shards=1)
    renderer.write_json(predictions / "structure_evidence.json", summary)


def test_six_rows_preserve_all_confidence_and_original_coordinates(tree, tmp_path):
    assets, predictions, rows, _ = tree
    metadata, entries = renderer.collect(assets, predictions)
    assert [e["id"] for e in entries] == [r["id"] for r in rows]
    assert len(entries) == 6
    assert entries[0]["confidence"]["min_ca_plddt"] == pytest.approx(2.49)
    assert metadata["pairs"][0]["unequal_within_pair_splits"] is True
    assert entries[1]["confidence"]["chunk_stream_seeds"] == [20260905, 20260906]
    stage = tmp_path / "stage"
    stage.mkdir()
    renderer.stage_pdbs(entries, stage)
    for e in entries:
        assert (stage / (e["stem"] + ".pdb")).read_bytes() == Path(e["source"]["prediction.pdb"]["path"]).read_bytes()
        assert len(e["confidence"]["ca_plddt_0_100"]) == e["length"]
    renderer.confidence_csv(entries, stage)
    assert len((stage / "confidence.csv").read_text().splitlines()) == 7


@pytest.mark.parametrize("name", ["cohort.jsonl", "selection.json"])
def test_changed_frozen_bytes_abort(tree, name):
    assets, predictions, _, _ = tree
    with (assets / name).open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="Changed frozen"):
        renderer.collect(assets, predictions)


@pytest.mark.parametrize("mutation", ["missing", "failed", "bad_signature", "bad_cohort", "settings", "checkpoint"])
def test_receipt_and_complete_rowset_rejection(tree, mutation):
    assets, predictions, rows, _ = tree
    directory = predictions / "objects" / rows[-1]["sequence_sha256"]
    if mutation == "missing":
        (directory / "result.json").unlink()
    elif mutation == "failed":
        result = json.loads((directory / "result.json").read_text())
        result["status"] = "failed"
        renderer.write_json(directory / "result.json", result)
        refresh(rows, predictions)  # A valid failure receipt must still not become a partial gallery.
    else:
        path = predictions / "worker-000-of-001.json"
        worker = json.loads(path.read_text())
        if mutation == "bad_signature":
            worker["evaluation_signature"] = "forged"
        elif mutation == "bad_cohort":
            worker["cohort_sha256"] = "changed"
        else:
            if mutation == "settings":
                worker["contract"]["num_diffusion_samples"] = 1
            else:
                worker["contract"]["model_files_sha256"] = {"config.json": "different"}
            # Keep receipt/object/index signatures internally consistent: reject against the plan.
            worker["evaluation_signature"] = se.contract_signature(worker["contract"])
            for row in rows:
                result_path = predictions / "objects" / row["sequence_sha256"] / "result.json"
                result = json.loads(result_path.read_text())
                result["evaluation_signature"] = worker["evaluation_signature"]
                renderer.write_json(result_path, result)
        renderer.write_json(path, worker)
        if mutation in {"settings", "checkpoint"}:
            refresh(rows, predictions)
    with pytest.raises(ValueError):
        renderer.collect(assets, predictions)
    assert not (assets / "prediction-results.json").exists()


@pytest.mark.parametrize("mutation,match", [("sequence", "sequence/length"), ("scale", "confidence scale"),
                                           ("mixed_residue_confidence", "per-residue confidence"), ("coordinates", "coordinates"),
                                           ("summary", "summary disagrees")])
def test_pdb_and_confidence_rejection(tree, mutation, match):
    assets, predictions, rows, _ = tree
    directory = predictions / "objects" / rows[0]["sequence_sha256"]
    path = directory / "prediction.pdb"
    text = path.read_text()
    if mutation == "sequence":
        text = text.replace("ALA", "VAL")
    elif mutation == "scale":
        text = text.replace("  0.38", " 38.00")
    elif mutation == "mixed_residue_confidence":
        text = text.replace("  0.38", "  0.39", 1)
    elif mutation == "coordinates":
        text = text.replace("   3.400", "     nan")
    path.write_text(text)
    result = json.loads((directory / "result.json").read_text())
    result["files_sha256"]["prediction.pdb"] = renderer.sha256_file(path)
    if mutation == "summary":
        result["mean_ca_plddt"] += 1
    renderer.write_json(directory / "result.json", result)
    refresh(rows, predictions)
    with pytest.raises(ValueError, match=match):
        renderer.collect(assets, predictions)


def test_pdb_residue_confidence_is_distinct_from_ca_metric(tree):
    """The real ESMFold2 writer uses output.plddt, not output.plddt_ca."""
    assets, predictions, rows, _ = tree
    directory = predictions / "objects" / rows[0]["sequence_sha256"]
    path = directory / "prediction.pdb"
    path.write_text(path.read_text().replace("  0.38", "  0.61"))
    result = json.loads((directory / "result.json").read_text())
    result["files_sha256"]["prediction.pdb"] = renderer.sha256_file(path)
    renderer.write_json(directory / "result.json", result)
    refresh(rows, predictions)
    _, entries = renderer.collect(assets, predictions)
    confidence = entries[0]["confidence"]
    assert confidence["pdb_residue_plddt_raw_0_1"][1] == .61
    assert confidence["ca_plddt_0_100"][1] == pytest.approx(37.51)


def test_independent_views_share_pair_scale_and_no_coordinate_transforms(tree):
    assets, predictions, _, _ = tree
    _, entries = renderer.collect(assets, predictions)
    points = np.array([[0., 0., 0.], [3., 0., 0.], [0., 2., 0.], [1., 1., 1.]])
    rotated = points @ np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]) + 10
    geometries = [{"ca": p, "atoms": p} for p in (points, rotated)]
    views = renderer.pair_views(geometries)
    assert views == renderer.pair_views(geometries)
    assert not np.allclose(views[0][:9], views[1][:9])
    assert views[0][9:12] == views[1][9:12]
    assert views[0][15:] == views[1][15:]
    for view, p in zip(views, (points, rotated)):
        np.testing.assert_allclose(view[12:15], p.mean(0))
        assert np.linalg.det(np.asarray(view[:9]).reshape(3, 3)) == pytest.approx(1.)
    pml = renderer.render_pml(entries)
    assert pml.count("scene ") == 7  # Six stored scenes and initial recall.
    assert pml.count("png ") == 6
    for forbidden in ("align ", "super ", "cealign", "remove ", "alter ", "transform_object", "save pair-"):
        assert forbidden not in pml


def test_cpu_pymol_synthetic_roundtrip(tree, tmp_path, monkeypatch):
    """Small CPU integration, never touches the real pending prediction tree."""
    pymol = "/home/lzp/miniconda3/envs/pymolviz/bin/pymol"
    if not Path(pymol).is_file():
        pytest.skip("CPU PyMOL unavailable")
    assets, predictions, _, _ = tree
    _, entries = renderer.collect(assets, predictions)
    frozen_before = {p.name: renderer.sha256_file(p) for p in assets.iterdir()}
    monkeypatch.setattr(renderer, "IMAGE_SIZE", 160)
    monkeypatch.setattr("sys.argv", ["renderer", "--assets", str(assets), "--predictions", str(predictions),
                                    "--runtime", str(tmp_path / "runtime"), "--pymol", pymol])
    renderer.main()
    metadata = json.loads((assets / "prediction-results.json").read_text())
    assert len(metadata["render"]["actual_views"]) == 6
    assert len(metadata["outputs_sha256"]) == 17  # Metadata does not hash itself.
    assert not list(assets.glob("*.npz"))
    for name, digest in frozen_before.items():
        assert renderer.sha256_file(assets / name) == digest
    svg = (assets / "contact-sheet.svg").read_text()
    assert "<text" in svg and "Unequal within-pair splitting: 1" in svg
    assert all(e["id"] in svg for e in entries)
    for e in entries:
        assert renderer.sha256_file(assets / (e["stem"] + ".pdb")) == e["source"]["prediction.pdb"]["sha256"]
    assert metadata["evidence_verification"]["status_counts"] == {"ok": 6}
