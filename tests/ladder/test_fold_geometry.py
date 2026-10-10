"""Parent-referenced structural comparison at a known residue correspondence."""

from __future__ import annotations

import numpy as np
import pytest

from src.capability.ladder import fold_geometry as fg


def _atom(serial, name, residue, number, x, y, z, *, b=90.0, chain="A"):
    # PDB columns exactly: serial 7-11, name 13-16, resName 18-20, chainID 22,
    # resSeq 23-26, coordinates 31-54, occupancy 55-60, tempFactor 61-66.
    return (
        f"ATOM  {serial:5d} {name:<4s} {residue:>3s} {chain}{number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           C  "
    )


def _pdb(coordinates, *, chain="A", with_ca=True):
    lines = []
    serial = 1
    for index, (x, y, z) in enumerate(coordinates, start=1):
        if with_ca:
            lines.append(_atom(serial, "CA", "ALA", index, x, y, z, chain=chain))
            serial += 1
        lines.append(_atom(serial, "CB", "ALA", index, x + 0.5, y, z, chain=chain))
        serial += 1
    return "\n".join(lines + ["TER", "END"]) + "\n"


def _helix(n, *, rise=1.5, radius=2.3, turn=100.0):
    """A plausible alpha-helical CA trace, so distances are protein-like."""

    angles = np.deg2rad(turn * np.arange(n))
    return np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), rise * np.arange(n)], axis=1
    )


def _rigid(coordinates, *, seed=0):
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(3, 3))
    rotation, _ = np.linalg.qr(matrix)
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1.0
    return coordinates @ rotation.T + generator.normal(size=3) * 10.0


# ------------------------------------------------------------------- parsing


def test_the_parser_reads_residue_ordered_ca_coordinates_of_one_chain():
    coordinates = _helix(12)
    parsed = fg.parse_ca_trace(_pdb(coordinates))
    assert parsed.shape == (12, 3)
    assert np.allclose(parsed, coordinates, atol=1e-3)


def test_a_second_chain_is_not_folded_into_the_first():
    text = _pdb(_helix(8)) + _pdb(_helix(5), chain="B")
    assert fg.parse_ca_trace(text).shape == (8, 3)


def test_a_residue_without_a_ca_atom_is_refused_not_silently_dropped():
    with pytest.raises(ValueError, match="no CA atom"):
        fg.parse_ca_trace(_pdb(_helix(6), with_ca=False))


def test_a_pdb_without_atom_records_is_refused():
    with pytest.raises(ValueError, match="no ATOM record"):
        fg.parse_ca_trace("HEADER something\nEND\n")


# ----------------------------------------------------------------- TM-score


def test_d0_follows_the_published_form_with_its_small_chain_floor():
    assert fg.tm_d0(10) == pytest.approx(0.5)
    assert fg.tm_d0(100) == pytest.approx(1.24 * 85 ** (1 / 3) - 1.8)
    with pytest.raises(ValueError):
        fg.tm_d0(0)


def test_an_identical_trace_scores_one():
    coordinates = _helix(60)
    result = fg.tm_score(coordinates, coordinates)
    assert result["tm_score"] == pytest.approx(1.0, abs=1e-9)
    assert result["identity_superposition_tm_score"] == pytest.approx(1.0, abs=1e-9)
    assert result["n_residues"] == 60


def test_the_score_is_invariant_under_a_rigid_body_transform():
    coordinates = _helix(60)
    moved = _rigid(coordinates, seed=3)
    assert fg.tm_score(moved, coordinates)["tm_score"] == pytest.approx(1.0, abs=1e-6)


def test_displacing_a_window_lowers_the_score_monotonically():
    coordinates = _helix(80)
    scores = []
    for shift in (0.0, 1.0, 4.0, 12.0):
        variant = coordinates.copy()
        variant[35:45] += np.asarray([shift, 0.0, 0.0])
        scores.append(fg.tm_score(variant, coordinates)["tm_score"])
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]


def test_the_search_never_reports_less_than_the_all_residue_superposition():
    coordinates = _helix(70)
    variant = coordinates.copy()
    variant[20:30] += np.asarray([0.0, 5.0, 0.0])
    result = fg.tm_score(variant, coordinates)
    assert result["tm_score"] >= result["identity_superposition_tm_score"] - 1e-12
    assert result["tm_score"] <= 1.0


def test_unequal_lengths_are_refused_rather_than_aligned():
    with pytest.raises(ValueError, match="one length"):
        fg.tm_score(_helix(40), _helix(41))


# --------------------------------------------------------------------- lDDT


def test_an_identical_trace_has_unit_lddt():
    coordinates = _helix(50)
    assert fg.lddt(coordinates, coordinates)["lddt"] == pytest.approx(1.0)


def test_lddt_is_invariant_under_a_rigid_body_transform():
    coordinates = _helix(50)
    assert fg.lddt(_rigid(coordinates, seed=7), coordinates)["lddt"] == pytest.approx(1.0, abs=1e-6)


def test_lddt_falls_where_local_distances_change():
    coordinates = _helix(50)
    variant = coordinates.copy()
    variant[20:30] *= 1.5
    whole = fg.lddt(variant, coordinates)["lddt"]
    window = fg.lddt(variant, coordinates, positions=range(20, 30))["lddt"]
    outside = fg.lddt(variant, coordinates, positions=range(0, 10))["lddt"]
    assert window < whole
    assert window < outside


def test_an_lddt_position_outside_the_trace_is_refused():
    coordinates = _helix(20)
    with pytest.raises(ValueError, match="outside the trace"):
        fg.lddt(coordinates, coordinates, positions=[25])


def test_a_pair_set_that_would_be_empty_is_refused_rather_than_scored():
    # Two residues 100 A apart: nothing falls inside the inclusion radius.
    far = np.asarray([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="inclusion radius"):
        fg.lddt(far, far, positions=[0])


# ---------------------------------------------- flank-superposed window RMSD


def test_an_identical_trace_leaves_the_window_where_it_was():
    coordinates = _helix(60)
    result = fg.window_rmsd_flank_superposed(coordinates, coordinates, start=25, stop=35)
    assert result["window_rmsd_flank_superposed_angstrom"] == pytest.approx(0.0, abs=1e-8)
    assert result["flank_rmsd_angstrom"] == pytest.approx(0.0, abs=1e-8)
    assert result["n_window_residues"] == 10 and result["n_flank_residues"] == 50


def test_a_displaced_window_is_measured_against_a_converged_flank_fit():
    coordinates = _helix(60)
    variant = coordinates.copy()
    variant[25:35] += np.asarray([3.0, 0.0, 0.0])
    result = fg.window_rmsd_flank_superposed(variant, coordinates, start=25, stop=35)
    assert result["flank_rmsd_angstrom"] == pytest.approx(0.0, abs=1e-6)
    assert result["window_rmsd_flank_superposed_angstrom"] == pytest.approx(3.0, abs=1e-6)


def test_a_window_that_leaves_no_flank_to_fit_on_is_refused():
    coordinates = _helix(10)
    with pytest.raises(ValueError, match="fewer than three"):
        fg.window_rmsd_flank_superposed(coordinates, coordinates, start=1, stop=9)


# ------------------------------------------------------- confidence readouts


def _arrays(length, *, plddt=None, pae=None):
    return {
        "ca_plddt_0_100": np.full(length, 90.0) if plddt is None else plddt,
        "predicted_aligned_error_angstrom": np.full((length, length), 2.0) if pae is None else pae,
        "ptm": np.asarray([0.9]),
    }


def test_confidence_readouts_keep_the_pairwise_field_pairwise():
    length = 20
    pae = np.full((length, length), 2.0)
    pae[np.ix_(range(8, 12), [index for index in range(length) if not 8 <= index < 12])] = 9.0
    readouts = fg.confidence_readouts(_arrays(length, pae=pae), start=8, stop=12)
    assert readouts["window_flank_mean_pae_angstrom"] == pytest.approx(9.0)
    assert readouts["mean_pae_angstrom"] < 9.0
    assert readouts["window_span"] == [8, 12]


def test_confidence_readouts_report_length_beside_every_confidence_number():
    readouts = fg.confidence_readouts(_arrays(30))
    assert readouts["length"] == 30
    assert set(readouts) >= {"mean_ca_plddt", "fraction_ca_plddt_ge70", "ptm", "mean_pae_angstrom"}


def test_a_stored_pae_that_does_not_match_the_stored_plddt_is_refused():
    arrays = _arrays(20)
    arrays["predicted_aligned_error_angstrom"] = np.full((19, 19), 2.0)
    with pytest.raises(ValueError, match="does not match"):
        fg.confidence_readouts(arrays)


def test_a_window_outside_the_prediction_is_refused():
    with pytest.raises(ValueError, match="not inside"):
        fg.confidence_readouts(_arrays(20), start=15, stop=25)


# --------------------------------------------------------- the joined readout


def test_compare_to_parent_returns_the_primary_local_and_flank_readouts():
    parent = _helix(60)
    variant = parent.copy()
    variant[25:35] += np.asarray([2.0, 0.0, 0.0])
    result = fg.compare_to_parent(_pdb(variant), _pdb(parent), start=25, stop=35)
    assert 0.0 < result["tm_score_to_parent"] < 1.0
    assert 0.0 < result["window_lddt_to_parent"] <= 1.0
    assert result["window_lddt_to_parent"] < result["global_lddt_to_parent"]
    assert result["window_rmsd_flank_superposed_angstrom"] == pytest.approx(2.0, abs=1e-5)
    assert result["schema_version"] == fg.SCHEMA_VERSION


def test_compare_to_parent_refuses_a_length_mismatch_instead_of_inferring_a_correspondence():
    with pytest.raises(ValueError, match="inferred residue correspondence"):
        fg.compare_to_parent(_pdb(_helix(40)), _pdb(_helix(45)))


def test_compare_to_parent_without_a_window_reports_only_the_global_readouts():
    parent = _helix(40)
    result = fg.compare_to_parent(_pdb(parent), _pdb(parent))
    assert result["tm_score_to_parent"] == pytest.approx(1.0, abs=1e-9)
    assert "window_lddt_to_parent" not in result
