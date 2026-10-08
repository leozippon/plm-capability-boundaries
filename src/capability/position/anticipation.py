"""Does a conditional formed before position i already favour its later contact partner?

E02 asks whether, while predicting an earlier amino acid, a generative model
already carries information about that residue's structural contacts with later
residues. Read literally as an intervention, the question is **not identifiable
for a causal model**, and saying so is part of the answer rather than a
concession. The conditional at position ``i`` is a deterministic function of the
prefix ``w_{<i}``; a residue ``w_j`` with ``j > i`` is not in that prefix, so no
manipulation of ``w_j`` can move ``p(. | w_{<i})`` by any amount. Such an
experiment would measure exactly zero on every arm and every pair, with no power
to discuss and nothing to detect.

The identifiable question in the same place is predictive, and it is the one this
module implements. The conditional at the anchor is a distribution over residues;
ask whether it places more mass on the residue that will actually come into
contact with the anchor than on the residue sitting at an equally distant
*non-contacting* position of the same protein. In nats, for an anchor ``i`` whose
conditional restricted to the residue alphabet is ``q_i``:

    z(i, j) = log q_i(w_j) - sum_a pbar(a) log q_i(a)

where ``pbar`` is the protein's own residue composition. ``z`` is the excess log
probability the conditional gives the residue that is really at ``j``, over the
residue it would expect there if ``j`` were a random position of this protein.
The endpoint is the difference of mean ``z`` between contacting and
separation-matched non-contacting partners **of the same anchor**. Its null is
zero and needs no estimator: nothing is fitted, so there is no fold, no
regularisation and no cross-family transfer to go wrong.

Why the matching is inside the anchor. ``z`` subtracts an anchor-specific
baseline, so the anchor's own entropy, the protein's composition and the depth of
the prefix are all held fixed; and because a contact and its control are drawn
from the same sequence-separation stratum, the comparison never confuses
structure with the decay of predictability along the chain. An anchor-and-stratum
cell that lacks either class contributes nothing.

Two things a positive value would still not establish, and one control for each.

*Pair covariation.* Residues in contact covary in natural sequences, so the
partner of a contacting position is partly predictable from the anchor residue
alone, with no model involved. The same statistic is therefore computed with
``q_i`` replaced by a label-blind empirical pair conditional ``f(a | w_i)``,
estimated from the matched pairs of **other** families so that no protein informs
its own control. A model whose contrast does not exceed this one has reproduced
sequence covariation rather than anticipated structure.

*Having already read the partner.* A masked arm's conditional at ``i`` is formed
with the rest of the sequence visible, so it has literally seen ``w_j``. Such an
arm is carried as an upper reference for the scale of the contrast, never as a
comparable measurement of anticipation.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..core.amino_acids import AA20
from ..interactions.pairwise_epistasis import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, interval
from .contact_response import SiteGeometry, separation_stratum

SCHEMA = "contact_anticipation_v1"

IDENTIFIABILITY = (
    "A causal model's conditional at position i is a function of the prefix w_{<i} "
    "alone, so intervening on a later partner w_j cannot change it: the literal "
    "prefix-intervention design measures exactly zero on every arm and every pair, by "
    "construction rather than by measurement. The identifiable quantity in the same "
    "place is predictive -- the excess log probability the position-i conditional "
    "gives the residue that really contacts it, over a non-contacting residue at "
    "matched sequence separation from the same anchor. A positive value says the "
    "prefix already constrains which residue can occupy a position the model has not "
    "reached; it does not say the model saw that residue."
)

#: The two statistics a record reports, in order.
SOURCES = ("model_conditional", "pair_covariation")

SOURCE_SEMANTICS = {
    "model_conditional": (
        "the arm's own log p(a | w_{<i}) at the anchor, renormalised over the resolved "
        "residue alphabet; for a masked arm this is log p(a | w_{-i}) and has seen the "
        "partner"
    ),
    "pair_covariation": (
        "a label-blind empirical conditional of the partner residue given the anchor "
        "residue, counted over the matched pairs of every other family, with Laplace "
        "smoothing; the model-free control for covariation between contacting residues "
        "in natural sequences"
    ),
}

#: Families a reported contrast needs. Five is the floor this project's own
#: response-bin analysis already applies to a family-grouped estimate.
MIN_FAMILIES = 5

#: Added to every cell of the pair-conditional count table before normalising, so a
#: residue pair unseen in the other families has a finite log probability rather
#: than being dropped from one side of the contrast.
PAIR_PSEUDOCOUNT = 1.0


def anchor_partner_design(
    geometry: Mapping[int, SiteGeometry], pairs: Mapping[str, Any],
    *, anchors: Sequence[int], wildtype: str, residues: Sequence[str],
    forward_only: bool = True,
) -> list[dict[str, Any]]:
    """Contacting partners and separation-matched non-contacting controls.

    ``anchors`` are the positions whose conditional the extraction resolved. A
    partner must carry an admitted coordinate, clear the separation floor, and --
    when ``forward_only`` -- lie after the anchor, because the question is about a
    residue the model has not yet produced. Only anchor-and-stratum cells holding
    both a contact and a non-contact survive, so the contrast is never taken
    across sequence separations or across anchors.
    """

    alphabet = set(residues)
    floor = int(pairs["min_separation"])
    cells: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for anchor in sorted({int(value) for value in anchors}):
        if anchor not in geometry or wildtype[anchor] not in alphabet:
            continue
        for partner in pairs["eligible_positions"]:
            partner = int(partner)
            if partner == anchor or (forward_only and partner <= anchor):
                continue
            separation = abs(partner - anchor)
            if separation < floor:
                continue
            distance = pairs["distance"].get((min(anchor, partner), max(anchor, partner)))
            if distance is None or wildtype[partner] not in alphabet:
                continue
            stratum = separation_stratum(separation)
            cells.setdefault((anchor, stratum), []).append(
                {
                    "i": anchor,
                    "j": partner,
                    "separation": separation,
                    "stratum": stratum,
                    "contact": bool(distance < pairs["cutoff_angstrom"]),
                    "structure_distance_angstrom": float(distance),
                    "anchor_residue": wildtype[anchor],
                    "partner_residue": wildtype[partner],
                }
            )
    rows: list[dict[str, Any]] = []
    for members in cells.values():
        if any(row["contact"] for row in members) and any(not row["contact"] for row in members):
            rows.extend(members)
    return rows


def composition(wildtype: str, residues: Sequence[str]) -> np.ndarray:
    """The protein's own residue frequencies over the resolved alphabet."""

    counts = np.asarray([wildtype.count(residue) for residue in residues], dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError("no resolved residue occurs in this wild type")
    return counts / total


def excess_logprob(logprobs: np.ndarray, weights: np.ndarray, column: int) -> float:
    """``log q(w_j)`` less the composition-weighted mean of ``log q``, in nats.

    ``logprobs`` is renormalised over the alphabet first, so the statistic is a
    comparison inside the residue alphabet and does not move when an arm places
    mass on tokens that are not residues at all.
    """

    row = np.asarray(logprobs, dtype=np.float64)
    if row.ndim != 1 or row.shape != weights.shape or not np.isfinite(row).all():
        raise ValueError("a conditional row and the composition must align and be finite")
    normalised = row - float(np.log(np.exp(row - row.max()).sum()) + row.max())
    return float(normalised[column] - float(weights @ normalised))


def pair_conditional(
    rows: Sequence[Mapping[str, Any]], residues: Sequence[str], *, holdout: Any
) -> np.ndarray:
    """Label-blind ``log f(partner | anchor)`` from every family but ``holdout``.

    Counted over contacts and controls together: a table built from contacts
    alone would carry the contact label this control exists to be free of.
    """

    index = {residue: position for position, residue in enumerate(residues)}
    table = np.full((len(residues), len(residues)), PAIR_PSEUDOCOUNT, dtype=np.float64)
    used = 0
    for row in rows:
        if row["family"] == holdout:
            continue
        table[index[row["anchor_residue"]], index[row["partner_residue"]]] += 1.0
        used += 1
    if used == 0:
        raise ValueError("the held-out family is the only family; no control table is estimable")
    return np.log(table / table.sum(axis=1, keepdims=True))


def _nested_mean(entries: Sequence[tuple[Any, float]]) -> float | None:
    grouped: dict[Any, list[float]] = {}
    for key, value in entries:
        grouped.setdefault(key, []).append(float(value))
    if not grouped:
        return None
    return float(np.mean([float(np.mean(values)) for values in grouped.values()]))


def _collapse(
    cells: Mapping[tuple[Any, Any, int, str], float]
) -> dict[Any, float]:
    """Average anchor-stratum cells up to one value per family, equally at each level."""

    per_anchor: dict[tuple[Any, Any, int], list[tuple[str, float]]] = {}
    for (family, assay, anchor, stratum), value in cells.items():
        per_anchor.setdefault((family, assay, anchor), []).append((stratum, value))
    per_assay: dict[tuple[Any, Any], list[tuple[int, float]]] = {}
    for (family, assay, anchor), entries in per_anchor.items():
        value = _nested_mean(entries)
        if value is not None:
            per_assay.setdefault((family, assay), []).append((anchor, value))
    per_family: dict[Any, list[tuple[Any, float]]] = {}
    for (family, assay), entries in per_assay.items():
        value = _nested_mean(entries)
        if value is not None:
            per_family.setdefault(family, []).append((assay, value))
    return {
        family: value
        for family, entries in per_family.items()
        if (value := _nested_mean(entries)) is not None
    }


def anticipation_contrast(
    rows: Sequence[Mapping[str, Any]], *, residues: Sequence[str],
    conditionals: Mapping[tuple[Any, int], np.ndarray],
    compositions: Mapping[Any, np.ndarray],
    draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """The E02 endpoint: contacting minus matched non-contacting excess log probability.

    One value per anchor and separation stratum, averaged equally up through
    anchor, assay and family, with families as the bootstrap units -- the nesting
    this project already applies to its structural pair census. The per-side means
    are reported beside the contrast so that a reader sees the levels and not only
    their difference.
    """

    column = {residue: position for position, residue in enumerate(residues)}
    families = sorted({row["family"] for row in rows})
    result: dict[str, Any] = {
        "identifiability": IDENTIFIABILITY,
        "contact_pairs": sum(1 for row in rows if row["contact"]),
        "control_pairs": sum(1 for row in rows if not row["contact"]),
        "anchors": len({(row["assay"], row["i"]) for row in rows}),
        "assays": sorted({str(row["assay"]) for row in rows}),
        "families": [str(family) for family in families],
        "residue_alphabet": list(residues),
        "strata": sorted({row["stratum"] for row in rows}),
        "weighting": (
            "equal separation strata within anchor, equal anchors within assay, equal "
            "assays within family, equal families; families are the bootstrap units"
        ),
        "sources": {},
    }
    if len(families) < MIN_FAMILIES:
        result["status"] = (
            f"{len(families)} families is below the {MIN_FAMILIES}-family floor for a "
            "family-grouped contrast"
        )
        return result
    result["status"] = "estimated"
    tables = {family: pair_conditional(rows, residues, holdout=family) for family in families}
    for source in SOURCES:
        cells: dict[tuple[Any, Any, int, str], dict[str, list[float]]] = {}
        for row in rows:
            weights = compositions[row["assay"]]
            target = column[row["partner_residue"]]
            if source == "model_conditional":
                vector = np.asarray(conditionals[(row["assay"], int(row["i"]))], dtype=np.float64)
            else:
                vector = tables[row["family"]][column[row["anchor_residue"]]]
            value = excess_logprob(vector, weights, target)
            key = (row["family"], row["assay"], int(row["i"]), row["stratum"])
            cells.setdefault(key, {"contact": [], "control": []})[
                "contact" if row["contact"] else "control"
            ].append(value)
        differences, contacts, controls = {}, {}, {}
        for key, sides in cells.items():
            if not sides["contact"] or not sides["control"]:
                continue
            contacts[key] = float(np.mean(sides["contact"]))
            controls[key] = float(np.mean(sides["control"]))
            differences[key] = contacts[key] - controls[key]
        per_family = _collapse(differences)
        result["sources"][source] = {
            "semantics": SOURCE_SEMANTICS[source],
            "matched_cells": len(differences),
            "families": sorted(str(family) for family in per_family),
            "contrast_nats": interval(
                [per_family[family] for family in sorted(per_family)], draws=draws, seed=seed
            ),
            "contact_excess_nats": interval(
                [value for _, value in sorted(_collapse(contacts).items())],
                draws=draws, seed=seed,
            ),
            "control_excess_nats": interval(
                [value for _, value in sorted(_collapse(controls).items())],
                draws=draws, seed=seed,
            ),
            "per_family_contrast": {
                str(family): per_family[family] for family in sorted(per_family)
            },
        }
    return result


def resolved_alphabet(residues: Sequence[str]) -> tuple[str, ...]:
    """The conditional's residue columns, in AA20 order, refusing anything else."""

    ordered = tuple(residue for residue in AA20 if residue in set(residues))
    if len(ordered) != len(set(residues)):
        raise ValueError("a conditional column is not an AA20 residue")
    return ordered
