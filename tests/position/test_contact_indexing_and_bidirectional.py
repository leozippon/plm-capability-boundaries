"""Regression tests for 0-based contact indexing and bidirectional refitting.

Guards against the P0 off-by-one bug where 0-based pair keys from
``structure_sites()`` / ``contact_pairs()`` had ``1`` subtracted a second time
when indexing Potts coupling matrices and per-column MSA conservation vectors,
shifting every lookup by -1 residue and silently wrapping residue 1 (index 0)
to index -1 (residue L - 1).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from src.capability.position.contact_response import (
    admitted_geometry,
    contact_pairs,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "capability" / "position"


def _load_script(name: str):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


measure_mod = _load_script("measure_contact_discrimination")
bidir_mod = _load_script("fit_contact_bidirectional")


def _synthetic_structure_rows(assay: str, family: str, wildtype: str, coords: np.ndarray):
    """Build valid 1-based structure_sites rows and matching coverage receipt."""
    wt_sha = hashlib.sha256(wildtype.encode("utf-8")).hexdigest()
    src_sha = hashlib.sha256(f"mmcif:{assay}".encode("utf-8")).hexdigest()
    src_path = f"synthetic/{assay}.cif"
    rows = []
    for pos_1based, residue in enumerate(wildtype, start=1):
        atom = "CA" if residue == "G" else "CB"
        rows.append({
            "assay_id": assay,
            "cluster": family,
            "wt_position": pos_1based,
            "residue": residue,
            "wt_sequence": wildtype,
            "wt_sha256": wt_sha,
            "status": "admitted",
            "method": "X-RAY DIFFRACTION",
            "source_path": src_path,
            "source_sha256": src_sha,
            "entry_id": assay[-4:],
            "entity_id": "1",
            "chain": "A",
            "auth_chain": "A",
            "model": 1,
            "mapping_kind": "full_entity",
            "coordinate_present": True,
            "contact_atom": atom,
            "contact_atom_present": True,
            "contact_coordinates": [float(v) for v in coords[pos_1based - 1]],
            "rsa": 0.2,
        })
    coverage = {
        "assays": [{
            "assay": assay,
            "cluster": family,
            "status": "admitted",
            "method": "X-RAY DIFFRACTION",
            "source_path": src_path,
            "source_sha256": src_sha,
            "entry_id": assay[-4:],
            "entity_id": "1",
            "chain": "A",
            "auth_chain": "A",
            "model": 1,
            "mapping_kind": "full_entity",
        }],
        "sources": [{
            "path": src_path,
            "sha256": src_sha,
            "status": "parsed",
            "method": "X-RAY DIFFRACTION",
        }],
    }
    return rows, coverage


def _folded_coords(length: int, contact_set: set[tuple[int, int]]) -> np.ndarray:
    """Place non-contacting residues far apart on a 3D helix and pull contacts < 8 A."""
    coords = np.zeros((length, 3), dtype=np.float64)
    for idx in range(length):
        coords[idx] = np.array([20.0 * idx, 15.0 * (idx % 3), 0.0])
    # Pull designated contact pairs to 4.5 A apart by pairing the second partner
    # near the first partner's coordinate (only valid for disjoint pairs here).
    for i, j in sorted(contact_set):
        coords[j] = coords[i] + np.array([4.5, 0.0, 0.0])
    return coords


def test_zero_based_indexing_aligns_with_1based_structure_sites_and_never_wraps_residue_one():
    """Residue wt_position=1 (index 0) reads [0] / [0, j], never [-1] (index L - 1)."""
    wildtype = "MKQLEDKVEELLSKNY"  # L = 16
    length = len(wildtype)
    planted_contacts = {(0, 7), (1, 10), (2, 13)}
    coords = _folded_coords(length, planted_contacts)
    rows, coverage = _synthetic_structure_rows("SYN_1ABC", "fam_0", wildtype, coords)

    geometry = admitted_geometry(
        rows, assay="SYN_1ABC", family="fam_0", wildtype=wildtype, coverage=coverage
    )
    assert min(geometry) == 0
    assert max(geometry) == length - 1
    assert geometry[0].residue == wildtype[0] == "M"
    assert geometry[length - 1].residue == wildtype[-1] == "Y"

    pairs = contact_pairs(geometry, cutoff=8.0, min_separation=6)
    assert (0, 7) in pairs["contacts"]
    assert pairs["contacts"] == planted_contacts

    # Distinct per-site conservation with a poison sentinel at C-terminus L - 1.
    conservation = np.arange(10.0, 10.0 + length, dtype=np.float64)
    conservation[-1] = -9999.0

    # Unique encoding for every coupling entry, with poison sentinels on row/col L - 1.
    couplings = np.zeros((length, length), dtype=np.float64)
    for i in range(length):
        for j in range(i + 1, length):
            value = 100.0 * i + float(j)
            couplings[i, j] = value
            couplings[j, i] = value
    couplings[length - 1, :] = -8888.0
    couplings[:, length - 1] = -8888.0

    record = {
        "assay": "SYN_1ABC",
        "length": length,
        "pairs": pairs,
        "couplings": couplings,
    }

    keys_m, _, potts_m, _ = measure_mod.pair_coupling_scores(record)
    keys_b, built = bidir_mod.pair_blocks(record, conservation)
    assert keys_m == keys_b

    idx_0_7 = keys_m.index((0, 7))
    # Must read couplings[0, 7] == 7.0, NOT couplings[-1, 6] == -8888.0!
    assert potts_m[idx_0_7] == pytest.approx(7.0)
    assert built["couplings"][idx_0_7, 0] == pytest.approx(7.0)
    # Must read conservation[0] == 10.0 and conservation[7] == 17.0, NOT conservation[-1] == -9999.0!
    assert built["prof"][idx_0_7, 0] == pytest.approx(10.0)  # min(10, 17)
    assert built["prof"][idx_0_7, 1] == pytest.approx(17.0)  # max(10, 17)
    assert built["prof"][idx_0_7, 2] == pytest.approx(27.0)  # sum
    assert built["prof"][idx_0_7, 3] == pytest.approx(170.0)  # product

    # Every non-C-terminal pair (low < high < L - 1) must be free of sentinel values.
    for pos, (low, high) in enumerate(keys_m):
        if high < length - 1:
            assert potts_m[pos] == pytest.approx(100.0 * low + float(high))
            assert built["couplings"][pos, 0] == pytest.approx(100.0 * low + float(high))
            assert built["prof"][pos, 0] == pytest.approx(min(conservation[low], conservation[high]))
            assert built["prof"][pos, 1] == pytest.approx(max(conservation[low], conservation[high]))


def test_out_of_bounds_pair_keys_raise_explicitly():
    """Out-of-bounds or negative pair keys fail fast instead of wrapping silently."""
    length = 12
    couplings = np.zeros((length, length), dtype=np.float64)
    conservation = np.ones(length, dtype=np.float64)

    for bad_key in [(-1, 6), (0, length), (8, 8), (9, 2)]:
        record = {
            "assay": "BAD_1XYZ",
            "length": length,
            "pairs": {"distance": {bad_key: 5.0}, "cutoff_angstrom": 8.0},
            "couplings": couplings,
        }
        with pytest.raises(ValueError):
            measure_mod.pair_coupling_scores(record)
        with pytest.raises(ValueError):
            bidir_mod.pair_blocks(record, conservation)


def test_synthetic_planted_contacts_yield_perfect_auc_and_positive_pure_couplings_increment():
    """Planted contact couplings give AUC=1.0 and resolved positive couplings|(geom+prof+M)."""
    rng = np.random.default_rng(20261120)
    n_families = 10
    length = 24
    # Disjoint planted contacts including residue 0 (N-terminus) and residue L - 1 (C-terminus).
    base_contacts = [(0, 8), (1, 12), (2, 16), (3, 20), (4, 23), (5, 14), (6, 18)]

    blocks = {}
    model = {"arm_noisy": {}}
    support = {}

    for fam_idx in range(n_families):
        assay = f"SYN_{fam_idx:02d}_1SYN"
        family = f"fam_{fam_idx:02d}"
        wildtype = "M" + "ACDEFGHIKLMNPQRSTVWY"[fam_idx % 20] + "A" * (length - 2)
        # Rotate two contact indices slightly per family while always keeping (0, 8) and (4, 23).
        planted = set(base_contacts)
        coords = _folded_coords(length, planted)
        rows, coverage = _synthetic_structure_rows(assay, family, wildtype, coords)
        geometry = admitted_geometry(
            rows, assay=assay, family=family, wildtype=wildtype, coverage=coverage
        )
        pairs = contact_pairs(geometry, cutoff=8.0, min_separation=6)
        assert pairs["contacts"] == planted

        # Plant strong positive coupling values exactly at the 0-based contact coordinates.
        couplings = rng.normal(loc=0.0, scale=0.05, size=(length, length))
        couplings = 0.5 * (couplings + couplings.T)
        np.fill_diagonal(couplings, 0.0)
        for i, j in planted:
            couplings[i, j] = 2.0 + rng.uniform(0.0, 0.2)
            couplings[j, i] = couplings[i, j]

        record = {"assay": assay, "length": length, "pairs": pairs, "couplings": couplings}

        # 1. measure_contact_discrimination.py must recover exact AUC = 1.0 (auc_minus_half == +0.5).
        keys, label, potts, _ = measure_mod.pair_coupling_scores(record)
        assert measure_mod.auc_minus_half(potts, label) == pytest.approx(0.5)

        # Contrast with the old buggy (k-1) indexing, which must severely degrade AUC.
        buggy_potts = np.array([couplings[k[0] - 1, k[1] - 1] for k in keys], dtype=float)
        assert measure_mod.auc_minus_half(buggy_potts, label) < 0.2

        # 2. Build blocks for fit_contact_bidirectional.py with uninformative conservation & noisy M.
        conservation = rng.uniform(0.5, 2.5, size=length)
        keys_b, built = bidir_mod.pair_blocks(record, conservation)
        assert keys_b == keys
        blocks[assay] = {"keys": keys_b, **built}
        model["arm_noisy"][assay] = {
            "rows": np.arange(len(keys_b), dtype=np.int64),
            "column": rng.normal(0.0, 1.0, size=(len(keys_b), 1)),
        }
        support[assay] = {
            "family": family,
            "entry": "1SYN",
            "length": length,
            "pairs": len(keys_b),
            "contacts": int(built["label"].sum()),
            "effective_depth_per_site": 10.0,
            "arm_coverage": {"arm_noisy": {"pairs": len(keys_b)}},
        }

    panel_arms = [{"arm": "arm_noisy", "paradigm": "masked_marginal_pseudolikelihood",
                   "references": sorted(blocks)}]
    result = bidir_mod.evaluate_panel(panel_arms, blocks, model, support)

    for stratum in ("contact_ranking", "contact_squared_error", "contact_squared_error_clipped"):
        pure = result["strata"][stratum]["pure_couplings_increment"]
        col = pure["columns"][0]
        assert col["arm"] == "arm_noisy"
        assert col["contrast"] == "couplings_given_geom_prof_model"
        assert col["point"] > 0.0
        assert col["interval"][0] > 0.0
        assert col["marginal_interval"][0] > 0.0
        assert col["verdict"] == "positive"
        assert col["marginal_verdict"] == "positive"

    # Ranking AUC of geom+prof+couplings+M should approach 1.0.
    rank_col = result["strata"]["contact_ranking"]["pure_couplings_increment"]["columns"][0]
    assert rank_col["joint_performance"] > 0.99


def test_clipped_brier_prevents_overshoot_penalty():
    """Predictions overshooting [0, 1] are penalized less under clipped Brier than raw Brier."""
    label = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    # Single severe overshoot > 1.0 mimicking UBR5_HUMAN_1I2T raw-nats Ridge prediction (1.158).
    pred = np.array([1.35, -0.15, 0.05, 0.05, 0.90, 0.05])
    weights = np.ones_like(label)
    groups = np.array(["fam_a"] * len(label))

    stats = bidir_mod.group_statistics(pred, label, weights, groups)
    assert stats["contact_squared_error_clipped"]["fam_a"] < stats["contact_squared_error"]["fam_a"]
    # Clipped prediction matches exact manual formula.
    expected_clipped = float(np.mean((np.clip(pred, 0.0, 1.0) - label) ** 2))
    assert stats["contact_squared_error_clipped"]["fam_a"] == pytest.approx(expected_clipped)
