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
SOURCES = (
    "model_conditional",
    "pair_covariation",
    "anchor_permuted_within_protein",
    "anchor_permuted_across_family",
)

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
    "anchor_permuted_within_protein": (
        "the arm's conditional taken from a different anchor of the SAME protein, with "
        "the family, the fold and the partner set unchanged; the fold control -- a "
        "contrast that survives here is protein-level composition rather than anything "
        "specific to the anchor"
    ),
    "anchor_permuted_across_family": (
        "the arm's conditional taken from an anchor of a DIFFERENT family; a contrast "
        "that survives here is generic amino-acid class statistics and carries no fold "
        "information at all"
    ),
}

SOURCE_READS_AS = {
    "model_conditional": "the measurement",
    "pair_covariation": "model-free covariation of contacting residue pairs",
    "anchor_permuted_within_protein": "protein- and fold-level composition",
    "anchor_permuted_across_family": "generic amino-acid class composition",
}

DEGENERATE_CONTROL = (
    "a control matched on the partner's residue identity is identically zero, because "
    "given the anchor, z(i, j) depends on j only through w_j. The statistic measures "
    "alignment between the conditional and the COMPOSITION of contacting partners, not "
    "position-specific prediction of which residue sits where"
)

#: Families a reported contrast needs. Five is the floor this project's own
#: response-bin analysis already applies to a family-grouped estimate.
MIN_FAMILIES = 5

#: Added to every cell of the pair-conditional count table before normalising, so a
#: residue pair unseen in the other families has a finite log probability rather
#: than being dropped from one side of the contrast.
PAIR_PSEUDOCOUNT = 1.0

#: Draws of the contact-label permutation null, and its seed. Both fixed here.
PERMUTATION_DRAWS = 2000
PERMUTATION_SEED = 20261008

#: Relative-accessibility bands are a median split of the admitted sites actually
#: analysed, which is the rule ``interactions.contact_enrichment`` already declares
#: for an RSA cell. Computed before any excess log probability is read.
RSA_BAND_RULE = "median split of the partner relative accessibility over the analysed matched pairs"


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
                    "anchor_rsa": geometry[anchor].rsa,
                    "partner_rsa": geometry[partner].rsa,
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


def _collapse(cells: Mapping[tuple, float]) -> dict[Any, float]:
    """Average cells up to one value per family, equally at each nesting level.

    A cell key is ``(family, assay, anchor, *extra)``; everything after the anchor
    is averaged first, then anchors within assay, assays within family.
    """

    per_anchor: dict[tuple[Any, Any, int], list[tuple[tuple, float]]] = {}
    for key, value in cells.items():
        family, assay, anchor = key[0], key[1], key[2]
        per_anchor.setdefault((family, assay, anchor), []).append((key[3:], value))
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


def _derangement(count: int) -> list[int]:
    """A fixed-point-free index map, deterministic and seedless.

    Rotating by ``max(1, n // 2)`` has no fixed point for any ``n >= 2`` and puts
    the greatest distance between an anchor and its substitute, which is what a
    control wants: the substituted conditional should be as unlike the anchor's own
    as the protein allows.
    """

    if count < 2:
        return []
    shift = max(1, count // 2)
    return [(index + shift) % count for index in range(count)]


def permuted_anchor_maps(
    rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[tuple[Any, int], tuple[Any, int]], dict[tuple[Any, int], tuple[Any, int]]]:
    """Where each anchor's substitute conditional comes from, for both controls.

    Within protein: a derangement of that protein's own anchors, so family, fold
    and partner set are identical and only the anchor moves. Across family: a
    derangement of the global anchor list ordered by family, which places most
    anchors in another family; an anchor whose substitute lands in its own family
    is left out of that control rather than counted as a cross-family draw.
    """

    by_assay: dict[Any, list[int]] = {}
    family_of: dict[Any, Any] = {}
    for row in rows:
        by_assay.setdefault(row["assay"], []).append(int(row["i"]))
        family_of[row["assay"]] = row["family"]
    within: dict[tuple[Any, int], tuple[Any, int]] = {}
    for assay, anchors in by_assay.items():
        ordered = sorted(set(anchors))
        for source, target in enumerate(_derangement(len(ordered))):
            within[(assay, ordered[source])] = (assay, ordered[target])
    allanchors = sorted(
        {(row["assay"], int(row["i"])) for row in rows},
        key=lambda item: (str(family_of[item[0]]), str(item[0]), item[1]),
    )
    across: dict[tuple[Any, int], tuple[Any, int]] = {}
    for source, target in enumerate(_derangement(len(allanchors))):
        origin, substitute = allanchors[source], allanchors[target]
        if family_of[origin[0]] != family_of[substitute[0]]:
            across[origin] = substitute
    return within, across


def _source_values(
    rows: Sequence[Mapping[str, Any]], source: str, *, residues: Sequence[str],
    conditionals: Mapping[tuple[Any, int], np.ndarray],
    compositions: Mapping[Any, np.ndarray],
    tables: Mapping[Any, np.ndarray],
    within: Mapping[tuple[Any, int], tuple[Any, int]],
    across: Mapping[tuple[Any, int], tuple[Any, int]],
) -> list[tuple[Mapping[str, Any], float]]:
    """``z`` for every row this source can be evaluated on, and the row beside it."""

    column = {residue: position for position, residue in enumerate(residues)}
    values = []
    for row in rows:
        anchor = (row["assay"], int(row["i"]))
        weights = compositions[row["assay"]]
        target = column[row["partner_residue"]]
        if source == "model_conditional":
            vector = conditionals.get(anchor)
        elif source == "pair_covariation":
            vector = tables[row["family"]][column[row["anchor_residue"]]]
        elif source == "anchor_permuted_within_protein":
            substitute = within.get(anchor)
            vector = None if substitute is None else conditionals.get(substitute)
        elif source == "anchor_permuted_across_family":
            substitute = across.get(anchor)
            vector = None if substitute is None else conditionals.get(substitute)
        else:
            raise ValueError(f"unknown source {source!r}")
        if vector is None:
            continue
        values.append((row, excess_logprob(np.asarray(vector, dtype=np.float64), weights, target)))
    return values


def _cells(
    values: Sequence[tuple[Mapping[str, Any], float]], *, rsa_matched: bool,
    rsa_median: float | None,
) -> tuple[dict[tuple, dict[str, list[float]]], int]:
    """Group ``z`` into anchor-and-stratum cells, optionally matched on partner RSA."""

    cells: dict[tuple, dict[str, list[float]]] = {}
    dropped = 0
    for row, value in values:
        key = [row["family"], row["assay"], int(row["i"]), row["stratum"]]
        if rsa_matched:
            rsa = row.get("partner_rsa")
            if rsa is None or rsa_median is None:
                dropped += 1
                continue
            key.append("buried" if float(rsa) <= rsa_median else "exposed")
        cells.setdefault(tuple(key), {"contact": [], "control": []})[
            "contact" if row["contact"] else "control"
        ].append(value)
    return cells, dropped


def _matched(cells: Mapping[tuple, Mapping[str, Sequence[float]]]) -> dict[tuple, dict[str, float]]:
    return {
        key: {
            "contact": float(np.mean(sides["contact"])),
            "control": float(np.mean(sides["control"])),
            "difference": float(np.mean(sides["contact"])) - float(np.mean(sides["control"])),
        }
        for key, sides in cells.items()
        if sides["contact"] and sides["control"]
    }


def _estimate(
    matched: Mapping[tuple, Mapping[str, float]], *, draws: int, seed: int
) -> dict[str, Any]:
    per_family = _collapse({key: value["difference"] for key, value in matched.items()})
    return {
        "matched_cells": len(matched),
        "families": sorted(str(family) for family in per_family),
        "contrast_nats": interval(
            [per_family[family] for family in sorted(per_family)], draws=draws, seed=seed
        ),
        "contact_excess_nats": interval(
            [
                value
                for _, value in sorted(
                    _collapse({k: v["contact"] for k, v in matched.items()}).items(),
                    key=lambda item: str(item[0]),
                )
            ],
            draws=draws, seed=seed,
        ) if matched else None,
        "control_excess_nats": interval(
            [
                value
                for _, value in sorted(
                    _collapse({k: v["control"] for k, v in matched.items()}).items(),
                    key=lambda item: str(item[0]),
                )
            ],
            draws=draws, seed=seed,
        ) if matched else None,
        "per_family_contrast": {str(family): per_family[family] for family in sorted(per_family)},
    }


def permutation_null(
    cells: Mapping[tuple, Mapping[str, Sequence[float]]], observed: float | None, *,
    draws: int = PERMUTATION_DRAWS, seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """The statistic's distribution when the contact label carries no information.

    Within every cell the member values are held exactly as measured and only the
    labels are permuted, so the cell's separation distribution, its composition and
    the nesting are all untouched. This calibrates the group bootstrap -- it is the
    answer to "could a contact assignment with these separations have produced this
    contrast by itself" -- rather than replacing it.
    """

    usable = {
        key: list(sides["contact"]) + list(sides["control"])
        for key, sides in cells.items()
        if sides["contact"] and sides["control"]
    }
    sizes = {
        key: len(cells[key]["contact"]) for key in usable
    }
    if not usable or observed is None:
        return {"draws": 0, "status": "no matched cell to permute"}
    generator = np.random.default_rng(seed)
    nulls = []
    for _ in range(int(draws)):
        drawn = {}
        for key, members in usable.items():
            values = np.asarray(members, dtype=np.float64)
            order = generator.permutation(values.size)
            take = sizes[key]
            drawn[key] = float(values[order[:take]].mean()) - float(values[order[take:]].mean())
        per_family = _collapse(drawn)
        if per_family:
            nulls.append(float(np.mean([per_family[f] for f in sorted(per_family)])))
    if not nulls:
        return {"draws": 0, "status": "the permutation produced no family value"}
    array = np.asarray(nulls, dtype=np.float64)
    extreme = int(np.sum(np.abs(array) >= abs(float(observed))))
    return {
        "draws": int(array.size),
        "seed": int(seed),
        "mean_nats": float(array.mean()),
        "interval_nats": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))],
        "observed_nats": float(observed),
        "two_sided_p": float((1 + extreme) / (array.size + 1)),
        "reads_as": (
            "the contrast when which members of a cell are labelled contacting is "
            "permuted, with every member value, the separation distribution and the "
            "nesting untouched"
        ),
    }


def anticipation_contrast(
    rows: Sequence[Mapping[str, Any]], *, residues: Sequence[str],
    conditionals: Mapping[tuple[Any, int], np.ndarray],
    compositions: Mapping[Any, np.ndarray],
    draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
    permutation_draws: int = PERMUTATION_DRAWS,
) -> dict[str, Any]:
    """The E02 endpoint and the controls that decide what it can be attributed to."""

    families = sorted({row["family"] for row in rows})
    rsa_values = [
        float(row["partner_rsa"]) for row in rows if row.get("partner_rsa") is not None
    ]
    rsa_median = float(np.median(rsa_values)) if rsa_values else None
    result: dict[str, Any] = {
        "identifiability": IDENTIFIABILITY,
        "degenerate_control": DEGENERATE_CONTROL,
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
        "rsa_band_rule": RSA_BAND_RULE,
        "rsa_median": rsa_median,
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
    within, across = permuted_anchor_maps(rows)
    for source in SOURCES:
        values = _source_values(
            rows, source, residues=residues, conditionals=conditionals,
            compositions=compositions, tables=tables, within=within, across=across,
        )
        cells, _dropped = _cells(values, rsa_matched=False, rsa_median=None)
        matched = _matched(cells)
        block: dict[str, Any] = {
            "semantics": SOURCE_SEMANTICS[source],
            "reads_as": SOURCE_READS_AS[source],
            "rows_evaluated": len(values),
            **_estimate(matched, draws=draws, seed=seed),
        }
        if source == "model_conditional":
            block["permutation_null"] = permutation_null(
                cells, block["contrast_nats"].get("point"),
                draws=permutation_draws, seed=PERMUTATION_SEED,
            )
            banded, dropped = _cells(values, rsa_matched=True, rsa_median=rsa_median)
            block["partner_rsa_matched"] = {
                "rule": RSA_BAND_RULE,
                "dropped_without_rsa": int(dropped),
                **_estimate(_matched(banded), draws=draws, seed=seed),
            }
            strata = {}
            for band in ("buried", "exposed"):
                selected = [
                    (row, value) for row, value in values
                    if row.get("anchor_rsa") is not None and rsa_median is not None
                    and (("buried" if float(row["anchor_rsa"]) <= rsa_median else "exposed") == band)
                ]
                subcells, _ = _cells(selected, rsa_matched=False, rsa_median=None)
                strata[band] = {
                    "rows_evaluated": len(selected),
                    **_estimate(_matched(subcells), draws=draws, seed=seed),
                }
            block["anchor_rsa_strata"] = strata
        result["sources"][source] = block
    result["attribution"] = paired_attribution(result["sources"], draws=draws, seed=seed)
    return result


def paired_attribution(
    sources: Mapping[str, Mapping[str, Any]], *, draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """How much of the measurement each control does not account for.

    The measurement and every control are computed on the same cells and the same
    families, so the quantity that decides attribution is the **paired** per-family
    difference, not the overlap of two intervals. ``anchor_permuted_within_protein``
    is the one that matters: its paired residual is the part of the contrast that a
    different anchor of the same protein -- same family, same fold, same partner set
    -- does not reproduce, which is the only part that can be called anticipation by
    this anchor rather than composition of this protein.
    """

    measurement = sources.get("model_conditional", {}).get("per_family_contrast") or {}
    if not measurement:
        return {"status": "the measurement carried no family value"}
    attribution: dict[str, Any] = {
        "rule": (
            "per-family difference between the measurement and the control, over the "
            "families both carry; a residual whose interval excludes zero is the part "
            "of the contrast that control does not explain"
        ),
        "controls": {},
    }
    for name, block in sources.items():
        if name == "model_conditional":
            continue
        control = block.get("per_family_contrast") or {}
        shared = sorted(set(measurement) & set(control))
        if not shared:
            attribution["controls"][name] = {"status": "no shared family"}
            continue
        differences = [measurement[family] - control[family] for family in shared]
        residual = interval(differences, draws=draws, seed=seed)
        point = measurement and float(
            np.mean([measurement[family] for family in shared])
        )
        attribution["controls"][name] = {
            "reads_as": SOURCE_READS_AS[name],
            "families": len(shared),
            "measurement_nats": point,
            "control_nats": float(np.mean([control[family] for family in shared])),
            "residual_nats": residual,
            "share_explained": (
                None if not point else float(
                    np.mean([control[family] for family in shared]) / point
                )
            ),
        }
    return attribution


def resolved_alphabet(residues: Sequence[str]) -> tuple[str, ...]:
    """The conditional's residue columns, in AA20 order, refusing anything else."""

    ordered = tuple(residue for residue in AA20 if residue in set(residues))
    if len(ordered) != len(set(residues)):
        raise ValueError("a conditional column is not an AA20 residue")
    return ordered
