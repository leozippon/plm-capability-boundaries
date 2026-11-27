"""Tests for unified leakage-free 3D wild-type backbone cache (Results 1-5)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.capability.core.io import sha256_file
from src.capability.unified.roster import REPO_ROOT
from src.capability.unified.wt_backbones_3d import (
    DEFAULT_CACHE_DIR,
    DEFAULT_FREEZE_PDB_DIR,
    SequenceEntry,
    _match_targets_against_corpus,
    ingest_esmfold2_folded_records,
    kabsch_align,
    load_backbone_npz,
    parse_mmcif_chains,
    parse_pdb_backbone,
    plan_overlapping_windows,
    save_backbone_npz,
    seq_sha256,
    stitch_windowed_backbones,
)


def test_kabsch_align_and_window_stitching_recovers_exact_geometry() -> None:
    """Overlapping 1024-aa window stitching must recover rigid-transformed >1024-aa coordinates to <1e-5 A."""
    rng = np.random.default_rng(20261125)
    parent_len = 1500
    # Synthetic smooth helical backbone (N, CA, C, O)
    t = np.linspace(0.0, 50.0, parent_len)
    ca = np.column_stack([2.3 * np.cos(t), 2.3 * np.sin(t), 1.5 * t])
    true_coords = np.stack(
        [
            ca + np.array([-1.2, 0.4, -0.3]),  # N
            ca,                                 # CA
            ca + np.array([1.2, -0.3, 0.4]),   # C
            ca + np.array([1.8, -1.1, 0.9]),   # O
        ],
        axis=1,
    )
    true_plddt = np.linspace(70.0, 95.0, parent_len)

    windows = plan_overlapping_windows("A" * parent_len, max_len=1024, overlap=256)
    assert windows == [(0, 1024), (476, 1500)]

    # Apply a random proper rotation + translation to window 1
    q_mat, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q_mat) < 0:
        q_mat[:, -1] *= -1.0
    shift = np.array([15.0, -22.5, 9.3])
    w1_coords = np.einsum("lai,ij->laj", true_coords[476:1500], q_mat) + shift[None, None, :]

    stitched, stitched_plddt, rmsds = stitch_windowed_backbones(
        [
            {"window_start": 0, "window_end": 1024, "coords": true_coords[0:1024], "plddt": true_plddt[0:1024]},
            {"window_start": 476, "window_end": 1500, "coords": w1_coords, "plddt": true_plddt[476:1500]},
        ],
        parent_length=parent_len,
    )
    assert len(rmsds) == 1
    assert rmsds[0] < 1e-5
    assert float(np.max(np.abs(stitched - true_coords))) < 1e-4
    assert float(np.max(np.abs(stitched_plddt - true_plddt))) < 1e-4

    # Verify proper rotation guard (det(R) == +1 even if reflected input is tested)
    rot, _, _ = kabsch_align(w1_coords[:50, 1, :], true_coords[476:526, 1, :])
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-6)


def test_substring_matching_and_pdb_mmcif_parsers() -> None:
    """Exact + contiguous substring matching and real PDB/mmCIF parsing produce finite (L, 4, 3) coords."""
    freeze_pdb = REPO_ROOT / DEFAULT_FREEZE_PDB_DIR / "1E0L.pdb"
    seq, coords, plddt = parse_pdb_backbone(freeze_pdb)
    assert len(seq) == 37
    assert coords.shape == (37, 4, 3)
    assert np.isfinite(coords).all()
    assert np.isfinite(plddt).all()

    sub_seq = seq[5:28]
    e_exact = SequenceEntry(seq_sha256(seq), seq, {"r4_exp4_2_domains_12"}, ["1E0L"])
    e_sub = SequenceEntry(seq_sha256(sub_seq), sub_seq, {"r2_domainome_428"}, ["1E0L_sub"])
    hits = _match_targets_against_corpus(
        {e_exact.seq_sha256: e_exact, e_sub.seq_sha256: e_sub},
        [(str(freeze_pdb), "", seq)],
    )
    assert hits[e_exact.seq_sha256] == (str(freeze_pdb), "", 0, 37, True)
    assert hits[e_sub.seq_sha256] == (str(freeze_pdb), "", 5, 28, False)

    # Also verify real experimental mmCIF parsing
    exp_cif = REPO_ROOT / "archive/logs/shared/gate_structure_contact_20260924/structures/1ORC.cif.gz"
    chains = parse_mmcif_chains(exp_cif)
    assert len(chains) >= 1
    c_seq, c_coords = next(iter(chains.values()))
    assert len(c_seq) > 20
    assert c_coords.shape == (len(c_seq), 4, 3)
    assert np.isfinite(c_coords[:, :3, :]).all()


def test_npz_roundtrip_and_windowed_ingestion(tmp_path: Path) -> None:
    """Save/load roundtrip and JSONL ESMFold2 ingestion (direct + stitched) work cleanly."""
    seq_short = "ACDEFGHIKLMNPQRSTVWY" * 2
    coords_short = np.arange(len(seq_short) * 12, dtype=np.float32).reshape(len(seq_short), 4, 3)
    plddt_short = np.full(len(seq_short), 88.5, dtype=np.float32)

    p = tmp_path / "backbones" / seq_sha256(seq_short)[:2] / f"{seq_sha256(seq_short)}.npz"
    save_backbone_npz(
        p,
        sequence=seq_short,
        coords=coords_short,
        plddt=plddt_short,
        source_kind="afdb_v6_exact",
        source_path="dummy.pdb.gz",
    )
    loaded = load_backbone_npz(tmp_path, seq_short)
    assert loaded["sequence"] == seq_short
    assert np.array_equal(loaded["coords"], coords_short)
    assert loaded["res_mask"].all() and loaded["ca_mask"].all()
    assert loaded["source_kind"] == "afdb_v6_exact"

    # Test ingestion of a synthetic direct folded JSONL record
    seq_new = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH"
    sha_new = seq_sha256(seq_new)
    coords_new = np.ones((len(seq_new), 4, 3), dtype=np.float32)
    folded_jsonl = tmp_path / "folded.jsonl"
    folded_jsonl.write_text(
        json.dumps(
            {
                "parent_seq_sha256": sha_new,
                "parent_length": len(seq_new),
                "sequence": seq_new,
                "window_start": 0,
                "window_end": len(seq_new),
                "num_windows": 1,
                "coords": coords_new.tolist(),
                "plddt": [85.0] * len(seq_new),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ing = ingest_esmfold2_folded_records([folded_jsonl], cache_dir=tmp_path, root=REPO_ROOT)
    assert ing["n_ingested_direct"] == 1
    loaded_new = load_backbone_npz(tmp_path, sha_new)
    assert loaded_new["sequence"] == seq_new
    assert loaded_new["source_kind"] == "esmfold2_direct"


def test_production_wt_backbones_cache_integrity() -> None:
    """Validate built wt_backbones_3d/manifest.json, pending_wt_esmfold2.jsonl, and .sha256 sidecars."""
    cache_dir = REPO_ROOT / DEFAULT_CACHE_DIR
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("wt_backbones_3d/manifest.json not yet built")

    for fname in ("manifest.json", "pending_wt_esmfold2.jsonl"):
        fpath = cache_dir / fname
        spath = cache_dir / f"{fname}.sha256"
        assert fpath.is_file() and spath.is_file()
        assert sha256_file(fpath) == spath.read_text(encoding="utf-8").strip().split()[0]

    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert doc["atom_order"] == ["N", "CA", "C", "O"]
    tot = doc["totals"]
    assert tot["n_unique_sequences"] == tot["n_resolved"] + tot["n_pending_sequences"]
    assert tot["n_resolved"] > 500
    # Verify all 12 Exp 4.2 freeze domains and all 30 E01 contact WTs are resolved locally
    assert doc["by_benchmark"]["r4_exp4_2_domains_12"]["pending"] == 0
    assert doc["by_benchmark"]["r3_e01_contacts"]["pending"] == 0
    # Spot-check 5 resolved .npz files for exact shape & finiteness
    checked = 0
    for digest, rec in doc["entries"].items():
        if not rec["resolved"]:
            continue
        loaded = load_backbone_npz(cache_dir, digest)
        assert loaded["coords"].shape == (rec["length"], 4, 3)
        assert loaded["res_mask"].shape == (rec["length"],)
        assert math.isfinite(float(loaded["coords"][loaded["res_mask"], :3, :].mean()))
        checked += 1
        if checked >= 5:
            break
    assert checked == 5
