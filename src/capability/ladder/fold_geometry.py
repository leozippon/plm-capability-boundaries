"""Structural comparison of a variant fold to its parent, at a known correspondence.

Why this is not a structural alignment problem
==============================================

Every window rung of the ladder preserves length exactly, so residue *i* of the
variant is residue *i* of the parent by construction. The residue correspondence
is therefore **known**, not inferred, and the comparison needs no sequence or
structure alignment and no external aligner. That is a real simplification and
it is also what makes the local measure meaningful: "the window moved" is only a
statement if we know which residues the window is.

Nothing here folds anything. The coordinates come from the project's one
structure instrument, ESMFold2 run through
``scripts/capability/generation/run_structure_evidence.py``, which persists per
folded sequence the selected diffusion sample's CA pLDDT, the full
predicted-aligned-error matrix, pTM and an all-atom PDB. This module reads those
objects and derives three things from them.

The three readouts
==================

``tm_score`` is the TM-score at the identity correspondence: the standard
length-normalised superposition score, maximised over rigid-body superpositions
by the standard iterative-extension heuristic. It is the global similarity and
it carries the main claim.

``window_lddt`` is the local distance difference test restricted to the modified
window: superposition-free, so a window that is locally correct but globally
displaced is not punished twice.

``window_rmsd_flank_superposed`` superimposes on the *unmodified* flanks alone
and reports the window's own RMSD. It is the one readout that asks "given that
the rest of the protein is where it was, did this window stay put".

A confidence number is never a stability number and never a function number.
These three are geometric comparisons of two predicted structures.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "ladder_fold_geometry_v1"

#: Inclusion radius of the local distance difference test, in angstrom, and its
#: four tolerance thresholds. The published lDDT definition; declared here rather
#: than passed in, because a local score computed at a different radius is a
#: different quantity and would silently become comparable to this one.
LDDT_INCLUSION_RADIUS = 15.0
LDDT_THRESHOLDS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)

#: Pairs closer than this in sequence are excluded from lDDT, as in the published
#: definition: adjacent CA distances are fixed by the backbone geometry and would
#: inflate every score by the same constant.
LDDT_MIN_SEQUENCE_SEPARATION = 1

#: Seed fragment lengths of the TM-score superposition search, as fractions of
#: the chain. The standard heuristic sweeps L, L/2, L/4, ... with every start;
#: the stride below subsamples the starts, which is a cost choice and is declared
#: because it makes the search a lower bound on the true maximum rather than the
#: maximum itself. On a parent and a k-residue variant of it the identity
#: superposition is already near-optimal, so the search refines rather than finds.
TM_SEED_DIVISORS: tuple[int, ...] = (1, 2, 4, 8, 16)
TM_MIN_SEED = 4
TM_MAX_REFINEMENTS = 20


def parse_ca_trace(pdb_text: str) -> np.ndarray:
    """Residue-ordered CA coordinates of the first chain of one PDB.

    Only ATOM records are read, in residue order, and a residue without a CA is
    refused: a missing coordinate would silently drop a row and a column from
    every distance matrix built on it, which is exactly the failure that turns a
    structural comparison into a comparison of two different residue sets.
    """

    order: list[int] = []
    coordinates: dict[int, tuple[float, float, float]] = {}
    chain: str | None = None
    seen: set[int] = set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        this_chain = line[21]
        if chain is None:
            chain = this_chain
        if this_chain != chain:
            continue
        number = int(line[22:26])
        if number not in seen:
            seen.add(number)
            order.append(number)
        if line[12:16].strip() != "CA":
            continue
        if number in coordinates:
            continue
        coordinates[number] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    if not order:
        raise ValueError("the PDB carries no ATOM record")
    missing = [number for number in order if number not in coordinates]
    if missing:
        raise ValueError(f"{len(missing)} residues of this PDB carry no CA atom: {missing[:5]}")
    return np.asarray([coordinates[number] for number in order], dtype=np.float64)


def _superpose(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rotation and translation that take ``mobile`` onto ``reference``.

    Kabsch, with the reflection correction. Returned rather than applied so one
    superposition can be scored on a different residue set than it was fitted on,
    which is what the flank-superposed window RMSD needs.
    """

    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("superposition needs two equally shaped (n, 3) coordinate sets")
    if mobile.shape[0] < 3:
        raise ValueError("a rigid-body superposition needs at least three points")
    mobile_centre = mobile.mean(axis=0)
    reference_centre = reference.mean(axis=0)
    covariance = (mobile - mobile_centre).T @ (reference - reference_centre)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, sign if sign != 0 else 1.0])
    rotation = vt.T @ correction @ u.T
    translation = reference_centre - rotation @ mobile_centre
    return rotation, translation


def _apply(rotation: np.ndarray, translation: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation


def tm_d0(length: int) -> float:
    """The TM-score length-normalising distance, in angstrom.

    The published form, with the small-chain floor the reference implementation
    applies. Below 16 residues the cube root is not defined for this formula at
    all, so the floor is not a tolerance but the definition's own boundary.
    """

    length = int(length)
    if length < 1:
        raise ValueError("d0 needs a positive length")
    if length <= 15:
        return 0.5
    return max(0.5, 1.24 * (length - 15.0) ** (1.0 / 3.0) - 1.8)


def _tm_from_superposition(
    mobile: np.ndarray, reference: np.ndarray, subset: np.ndarray, d0: float
) -> tuple[float, np.ndarray]:
    rotation, translation = _superpose(mobile[subset], reference[subset])
    distances = np.linalg.norm(_apply(rotation, translation, mobile) - reference, axis=1)
    score = float(np.mean(1.0 / (1.0 + (distances / d0) ** 2)))
    return score, distances


def tm_score(mobile: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """TM-score at the identity residue correspondence.

    ``identity_superposition_tm_score`` is the score of the single superposition
    fitted on every residue; ``tm_score`` is the best the iterative-extension
    search found. Both are returned because the gap between them says how much
    of the comparison rests on the search heuristic rather than on the data.
    """

    mobile = np.asarray(mobile, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if mobile.shape != reference.shape:
        raise ValueError(
            f"a known-correspondence TM-score needs two traces of one length; got "
            f"{mobile.shape[0]} and {reference.shape[0]}"
        )
    length = mobile.shape[0]
    if length < 3:
        raise ValueError("a TM-score needs at least three residues")
    d0 = tm_d0(length)
    everything = np.arange(length)
    best, _ = _tm_from_superposition(mobile, reference, everything, d0)
    identity = best
    seeds: list[np.ndarray] = []
    for divisor in TM_SEED_DIVISORS:
        fragment = max(TM_MIN_SEED, length // divisor)
        if fragment > length:
            continue
        stride = max(1, fragment // 2)
        for start in range(0, length - fragment + 1, stride):
            seeds.append(np.arange(start, start + fragment))
    for seed in seeds:
        subset = seed
        for _ in range(TM_MAX_REFINEMENTS):
            score, distances = _tm_from_superposition(mobile, reference, subset, d0)
            best = max(best, score)
            cutoff = max(d0, 3.0)
            grown = np.flatnonzero(distances < cutoff)
            while grown.size < 3:
                cutoff += 0.5
                grown = np.flatnonzero(distances < cutoff)
            if grown.size == subset.size and np.array_equal(grown, subset):
                break
            subset = grown
    return {
        "tm_score": float(min(1.0, best)),
        "identity_superposition_tm_score": float(min(1.0, identity)),
        "d0_angstrom": float(d0),
        "n_residues": int(length),
        "correspondence": "identity; the window rungs preserve length exactly",
    }


def lddt(
    mobile: np.ndarray,
    reference: np.ndarray,
    *,
    positions: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Local distance difference test over ``positions`` (all residues by default).

    Superposition-free: only distances enter it, so a correct local geometry that
    has been displaced as a rigid body still scores well. The reference structure
    supplies the inclusion set, which is why the parent and not the variant is
    the second argument.
    """

    mobile = np.asarray(mobile, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if mobile.shape != reference.shape:
        raise ValueError("lDDT needs two traces of one length")
    length = mobile.shape[0]
    chosen = np.arange(length) if positions is None else np.asarray(sorted(set(int(p) for p in positions)))
    if chosen.size == 0:
        raise ValueError("lDDT needs at least one position")
    if chosen.min() < 0 or chosen.max() >= length:
        raise ValueError("an lDDT position lies outside the trace")
    reference_distance = np.linalg.norm(reference[:, None, :] - reference[None, :, :], axis=-1)
    mobile_distance = np.linalg.norm(mobile[:, None, :] - mobile[None, :, :], axis=-1)
    separation = np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])
    included = (reference_distance < LDDT_INCLUSION_RADIUS) & (
        separation > LDDT_MIN_SEQUENCE_SEPARATION
    )
    rows = included[chosen]
    n_pairs = int(rows.sum())
    if n_pairs == 0:
        raise ValueError(
            "no residue pair falls inside the lDDT inclusion radius for these "
            "positions; a local score over an empty pair set is refused"
        )
    deviation = np.abs(mobile_distance[chosen] - reference_distance[chosen])[rows]
    preserved = [float(np.mean(deviation <= threshold)) for threshold in LDDT_THRESHOLDS]
    return {
        "lddt": float(np.mean(preserved)),
        "per_threshold": {str(t): value for t, value in zip(LDDT_THRESHOLDS, preserved)},
        "n_positions": int(chosen.size),
        "n_pairs": n_pairs,
        "inclusion_radius_angstrom": LDDT_INCLUSION_RADIUS,
    }


def window_rmsd_flank_superposed(
    mobile: np.ndarray, reference: np.ndarray, *, start: int, stop: int
) -> dict[str, Any]:
    """The window's RMSD after superposing on the unmodified flanks alone.

    The flanks are what the two structures genuinely share, so fitting on them
    and measuring the window is the one readout that separates "the window moved"
    from "the whole chain is in a different frame". Reported in angstrom beside
    the flank RMSD itself, because a window displacement read against a flank fit
    that did not converge is not a window displacement.
    """

    mobile = np.asarray(mobile, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if mobile.shape != reference.shape:
        raise ValueError("a flank-superposed window RMSD needs two traces of one length")
    length = mobile.shape[0]
    start, stop = int(start), int(stop)
    if not 0 <= start < stop <= length:
        raise ValueError(f"window [{start}, {stop}) is not inside a {length}-residue trace")
    flank = np.concatenate([np.arange(0, start), np.arange(stop, length)])
    if flank.size < 3:
        raise ValueError("the flanks carry fewer than three residues to superpose on")
    rotation, translation = _superpose(mobile[flank], reference[flank])
    moved = _apply(rotation, translation, mobile)
    window = np.arange(start, stop)
    return {
        "window_rmsd_flank_superposed_angstrom": float(
            np.sqrt(np.mean(np.sum((moved[window] - reference[window]) ** 2, axis=1)))
        ),
        "flank_rmsd_angstrom": float(
            np.sqrt(np.mean(np.sum((moved[flank] - reference[flank]) ** 2, axis=1)))
        ),
        "n_flank_residues": int(flank.size),
        "n_window_residues": int(window.size),
    }


def confidence_readouts(
    arrays: Mapping[str, Any], *, start: int | None = None, stop: int | None = None
) -> dict[str, Any]:
    """Confidence summaries from one stored prediction, window-resolved.

    The pairwise field is kept pairwise: ``window_flank_mean_pae_angstrom`` is
    the mean predicted aligned error of the window-against-flank block, which is
    the part of the matrix that says whether the predictor knows where the new
    window sits relative to the rest. Reducing PAE to one global mean would throw
    exactly that away.
    """

    plddt = np.asarray(arrays["ca_plddt_0_100"], dtype=np.float64).reshape(-1)
    pae = np.asarray(arrays["predicted_aligned_error_angstrom"], dtype=np.float64)
    length = plddt.size
    if pae.shape != (length, length):
        raise ValueError("the stored PAE matrix does not match the stored pLDDT length")
    off_diagonal = ~np.eye(length, dtype=bool)
    readouts: dict[str, Any] = {
        "length": int(length),
        "mean_ca_plddt": float(plddt.mean()),
        "fraction_ca_plddt_ge70": float((plddt >= 70.0).mean()),
        "fraction_ca_plddt_ge90": float((plddt >= 90.0).mean()),
        "mean_pae_angstrom": float(pae[off_diagonal].mean()),
        "ptm": float(np.asarray(arrays["ptm"]).reshape(-1)[0]),
    }
    if start is not None and stop is not None:
        start, stop = int(start), int(stop)
        if not 0 <= start < stop <= length:
            raise ValueError(f"window [{start}, {stop}) is not inside a {length}-residue prediction")
        window = np.arange(start, stop)
        flank = np.concatenate([np.arange(0, start), np.arange(stop, length)])
        readouts["window_mean_ca_plddt"] = float(plddt[window].mean())
        readouts["window_span"] = [start, stop]
        if flank.size:
            readouts["window_flank_mean_pae_angstrom"] = float(
                pae[np.ix_(window, flank)].mean()
            )
            readouts["flank_mean_ca_plddt"] = float(plddt[flank].mean())
    return readouts


def compare_to_parent(
    variant_pdb: str,
    parent_pdb: str,
    *,
    start: int | None = None,
    stop: int | None = None,
) -> dict[str, Any]:
    """Every parent-referenced geometric readout of one variant fold.

    Raises when the two traces differ in length. That is not a tolerance: a
    parent comparison at a differing length would need an inferred
    correspondence, and the whole reason this experiment's main claim sits on the
    window rungs is that it does not need one.
    """

    variant = parse_ca_trace(variant_pdb)
    parent = parse_ca_trace(parent_pdb)
    if variant.shape != parent.shape:
        raise ValueError(
            f"the variant trace has {variant.shape[0]} residues and the parent "
            f"{parent.shape[0]}; a parent comparison at unequal length is refused "
            "because it would require an inferred residue correspondence"
        )
    global_score = tm_score(variant, parent)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        # Named for what it is a comparison *to*, because these rows are joined
        # beside confidence readouts of the variant alone and a bare "tm_score"
        # column would not say which structure it was measured against.
        "tm_score_to_parent": global_score["tm_score"],
        "identity_superposition_tm_score_to_parent": global_score["identity_superposition_tm_score"],
        "tm_d0_angstrom": global_score["d0_angstrom"],
        "n_residues_compared": global_score["n_residues"],
        "correspondence": global_score["correspondence"],
        "global_lddt_to_parent": lddt(variant, parent)["lddt"],
    }
    if start is not None and stop is not None:
        window = lddt(variant, parent, positions=range(int(start), int(stop)))
        result["window_lddt_to_parent"] = window["lddt"]
        result["window_lddt_n_pairs"] = window["n_pairs"]
        result.update(window_rmsd_flank_superposed(variant, parent, start=start, stop=stop))
    return result
