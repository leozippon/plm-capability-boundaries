"""E10: distance to a *traceable* pretraining corpus, and what it can be read for.

The question
============

**Can a checkpoint's predictive advantage extend to proteins that have no
homologue in the corpus it was trained on?** Answering it needs two things the
prior remote-homology gate did not have together: a corpus that can be named, and
a corpus that is the one the checkpoint was actually trained on.

Traceability is the result, not a precondition
==============================================

The authoritative in-repo record of what each checkpoint was trained on is
:attr:`~..core.arms.ArmSpec.pretraining_corpus`. Read against it, the 33-arm panel
splits four ways, and the split is itself a finding about how much of this
question the panel can support:

``declared_and_searchable``
    the registry names a corpus and a corpus staged on this host can stand for it.
    The relation is recorded per arm, with the direction of its error.
``declared_not_searchable``
    the registry names something this host cannot search: a proprietary corpus, an
    unidentified mixture, a component that was never staged, or a text corpus to
    which protein homology is not defined.
``undeclared``
    the registry carries :data:`~..core.arms.PRETRAINING_UNDECLARED`. A sentinel,
    deliberately, because inventing a corpus would put a false fact into every
    artefact.
``no_arm_spec``
    the checkpoint is reached by its own loader and has no registry entry at all,
    so the repository holds no corpus declaration for it.

An arm outside the first tier is **refused**, with its tier and reason, rather
than stratified against a corpus that does not describe it. That refusal is the
honest form of this experiment: a defensible statement about a few checkpoints is
worth more than a broad one whose corpus labels are guesses.

A present-day release is a proxy, and the bias has a direction
==============================================================

No host holds the historical snapshot any checkpoint was trained on. Every
relation in :data:`CORPUS_RELATIONS` therefore records which way its error runs,
and the two that matter are opposite:

* A **later release of the same corpus** is approximately a superset of the
  snapshot, so measured identity is biased *upward*: a target this measurement
  calls remote was at least as remote in the training snapshot (conservative),
  while a target it calls close may not have been in it at all. "Approximately"
  because releases also delete and merge entries.
* A **subset or partial** relation -- searching UniRef90 for an arm trained on
  UniRef100, or searching only the UniRef component of a UniRef+BFD or
  UniRef+ColabFoldDB mixture -- biases identity *downward*: a target called remote
  may sit inside the unsearched part of the corpus, so a remote call is **not**
  conservative, and only a close call is safe.

The second case is exactly the one the repository already warns about for the
ProGen2 mixture (:data:`~..core.arms.UNIREF90_BFD30_INCOMPLETE_SEARCH`), and it is
why searching UniRef90 rather than UniRef50 for those arms is an upgrade and not a
fix: it removes the UniRef90 half of the gap and leaves the BFD30 half named.

The estimate
============

Nothing is refitted. The increment read per stratum is the frozen out-of-fold
contrast of the completed information-progression panel -- ``BMPL`` minus ``BPL``
within-assay Spearman, on the exact 25,728-row, 201-assay, 163-family anchor
support -- restricted to the rows a stratum holds. Loading, hash-verification and
the group bootstrap are
:mod:`~..extensions.phenotype_strata`'s, imported rather than restated, so this
module adds one thing only: the stratum a target belongs to.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .homology import STRATUM_EDGES, STRATUM_NAMES, assign_stratum
from .homology_context import GROUP_FLOOR, power_record

SCHEMA_VERSION = "corpus_distance_v1"

#: Tier names, in the order they are reported.
TIERS: tuple[str, ...] = (
    "declared_and_searchable",
    "declared_not_searchable",
    "undeclared",
    "no_arm_spec",
)

#: Why a declared corpus cannot be searched here.
REFUSAL_CLASSES: tuple[str, ...] = (
    "non_protein_corpus",
    "proprietary_or_unidentified_corpus",
    "component_not_staged",
)

#: How a corpus staged on this host relates to a corpus the registry declares.
#: ``bias`` states which way measured identity is wrong, and ``remote_is_conservative``
#: whether a remote call can be trusted under that relation.
CORPUS_RELATIONS: dict[str, dict[str, Any]] = {
    "uniref90_bfd30": {
        "searchable_with": "uniref90_2026_03",
        "relation": "later_release_of_one_component",
        "unsearched_components": ["bfd30"],
        "bias": (
            "upward for the UniRef90 component (a later release is approximately a "
            "superset of the training snapshot) and downward overall (BFD30 is not "
            "searched at all)"
        ),
        "remote_is_conservative": False,
        "note": (
            "the registry's own caveat for this mixture: the previously staged "
            "snapshot was UniRef50, which under-counts retrievable support. Searching "
            "UniRef90 removes that half of the gap; the BFD30 half remains"
        ),
    },
    "uniref50": {
        "searchable_with": "uniref50_local_snapshot",
        "relation": "same_corpus_release_unrecorded",
        "unsearched_components": [],
        "bias": (
            "unknown in size and direction, because the staged snapshot's release is "
            "not recorded; a later release would bias identity upward"
        ),
        "remote_is_conservative": None,
        "alternative": "uniref90_2026_03",
        "alternative_relation": (
            "superset: every UniRef50 representative is a UniRef90 representative, so "
            "identity to UniRef90 is an upper bound on identity to UniRef50"
        ),
        "note": "the only arm of the 33 whose declared corpus is UniRef50 is ProtGPT2",
    },
    "uniref100": {
        "searchable_with": "uniref90_2026_03",
        "relation": "subset_proxy",
        "unsearched_components": ["uniref100 members absent from uniref90 representatives"],
        "bias": "downward: UniRef90 representatives are a subset of UniRef100",
        "remote_is_conservative": False,
    },
    "uniref50s_uniref90_colabfolddb": {
        "searchable_with": "uniref90_2026_03",
        "relation": "later_release_of_one_component",
        "unsearched_components": ["colabfolddb"],
        "bias": "downward overall: ColabFoldDB is not staged on this host",
        "remote_is_conservative": False,
    },
    "uniprot_ec_annotated": {
        "searchable_with": "swissprot",
        "relation": "same_corpus_release_unrecorded",
        "unsearched_components": [],
        "bias": "unknown; the staged Swiss-Prot release is not recorded",
        "remote_is_conservative": None,
        "note": "ZymCTRL sits outside the 33-arm panel and is reported separately",
    },
}

#: Declared corpora this host cannot search, and the class of the refusal.
UNSEARCHABLE: dict[str, tuple[str, str]] = {
    "progen2_base_mixture": (
        "proprietary_or_unidentified_corpus",
        "an unidentified mixture; the registry records no searchable component",
    ),
    "profluent_protein_atlas_v1": (
        "proprietary_or_unidentified_corpus",
        "a proprietary corpus that is not published or staged",
    ),
    "webtext": ("non_protein_corpus", "a text corpus; protein homology to it is not defined"),
    "reddit_dialogue": (
        "non_protein_corpus",
        "a text corpus; protein homology to it is not defined",
    ),
    "qwen2.5_pretraining_mixture": (
        "non_protein_corpus",
        "a text mixture; protein homology to it is not defined",
    ),
    "llama3_web_corpus_with_llama3.1_logit_distillation": (
        "non_protein_corpus",
        "a text corpus; protein homology to it is not defined",
    ),
}

#: The relation that defines the primary panel: the arms whose declared corpus is
#: the corpus actually searched, at a later release of its main component.
PRIMARY_RELATION = "later_release_of_one_component"

#: Distance bands. The repository's already-frozen retrieval strata, imported and
#: not restated, so this experiment introduces no new boundary.
DISTANCE_BANDS: tuple[str, ...] = STRATUM_NAMES
DISTANCE_EDGES: tuple[float, ...] = STRATUM_EDGES


def distance_band(max_identity_over_query: float) -> str:
    """Band a target by its highest identity to the searched corpus."""

    return assign_stratum(max_identity_over_query)


def traceability(arms: Sequence[str]) -> dict[str, Any]:
    """Classify every arm by what the registry declares about its corpus.

    Reads :mod:`~..core.arms` for the declaration and nothing else: no corpus is
    inferred from a model name, a paper or a family resemblance.
    """

    from ..core import arms as registry

    rows = []
    for name in arms:
        try:
            declared = registry.arm_spec(name).pretraining_corpus
        except KeyError:
            rows.append(
                {
                    "arm": name,
                    "declared_corpus": None,
                    "tier": "no_arm_spec",
                    "reason": (
                        "this checkpoint is reached by its own loader and has no registry "
                        "entry, so the repository holds no corpus declaration for it"
                    ),
                }
            )
            continue
        if declared == registry.PRETRAINING_UNDECLARED:
            rows.append(
                {
                    "arm": name,
                    "declared_corpus": declared,
                    "tier": "undeclared",
                    "reason": "the registry carries the undeclared sentinel for this checkpoint",
                }
            )
        elif declared in CORPUS_RELATIONS:
            relation = CORPUS_RELATIONS[declared]
            rows.append(
                {
                    "arm": name,
                    "declared_corpus": declared,
                    "tier": "declared_and_searchable",
                    "searchable_with": relation["searchable_with"],
                    "relation": relation["relation"],
                    "bias": relation["bias"],
                    "remote_is_conservative": relation["remote_is_conservative"],
                    "unsearched_components": relation["unsearched_components"],
                    "note": relation.get("note"),
                }
            )
        elif declared in UNSEARCHABLE:
            reason_class, reason = UNSEARCHABLE[declared]
            rows.append(
                {
                    "arm": name,
                    "declared_corpus": declared,
                    "tier": "declared_not_searchable",
                    "reason_class": reason_class,
                    "reason": reason,
                }
            )
        else:
            rows.append(
                {
                    "arm": name,
                    "declared_corpus": declared,
                    "tier": "declared_not_searchable",
                    "reason_class": "component_not_staged",
                    "reason": (
                        "this module declares no relation between the named corpus and "
                        "anything staged here; a relation must be declared before an arm "
                        "can be stratified"
                    ),
                }
            )
    counts = {tier: sum(1 for row in rows if row["tier"] == tier) for tier in TIERS}
    primary = sorted(
        row["arm"]
        for row in rows
        if row["tier"] == "declared_and_searchable" and row.get("relation") == PRIMARY_RELATION
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "src/capability/core/arms.py ArmSpec.pretraining_corpus",
        "tiers": list(TIERS),
        "tier_counts": counts,
        "arms": sorted(rows, key=lambda row: (TIERS.index(row["tier"]), row["arm"])),
        "primary_relation": PRIMARY_RELATION,
        "primary_arms": primary,
        "searchable_arms": sorted(
            row["arm"] for row in rows if row["tier"] == "declared_and_searchable"
        ),
        "refused_arms": sorted(
            row["arm"] for row in rows if row["tier"] != "declared_and_searchable"
        ),
    }


def arms_for_corpus(record: Mapping[str, Any], corpus_id: str) -> dict[str, Any]:
    """Which traceable arms *this* search can stand for, and which it cannot.

    Being traceable is not enough: the corpus actually searched has to be the one
    the arm's relation names, either as its direct stand-in or as the declared
    alternative. An arm whose corpus relation points somewhere else is refused for
    this search rather than stratified against a corpus that does not describe it.
    """

    eligible: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for row in record["arms"]:
        if row["tier"] != "declared_and_searchable":
            refused.append({**row, "refusal": "not traceable to a searchable corpus"})
            continue
        relation = CORPUS_RELATIONS[row["declared_corpus"]]
        if relation["searchable_with"] == corpus_id:
            eligible.append({**row, "applied_relation": relation["relation"]})
        elif relation.get("alternative") == corpus_id:
            eligible.append(
                {
                    **row,
                    "applied_relation": "declared_alternative",
                    "applied_relation_detail": relation["alternative_relation"],
                }
            )
        else:
            refused.append(
                {
                    **row,
                    "refusal": (
                        f"the corpus searched here ({corpus_id}) does not stand for this "
                        f"arm's declared corpus ({row['declared_corpus']}); its relation "
                        f"names {relation['searchable_with']}"
                    ),
                }
            )
    return {
        "corpus_id": corpus_id,
        "eligible": sorted(eligible, key=lambda row: row["arm"]),
        "refused": sorted(refused, key=lambda row: row["arm"]),
        "eligible_arms": sorted(row["arm"] for row in eligible),
        "primary_arms": sorted(
            row["arm"] for row in eligible if row["applied_relation"] == PRIMARY_RELATION
        ),
    }


def assay_bands(
    targets: Sequence[Mapping[str, Any]], assays: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Band every assay by its own wild type's distance to the searched corpus."""

    by_target = {target["target_id"]: target for target in targets}
    bands: dict[str, str] = {}
    clusters: dict[str, set[int]] = {}
    for assay in assays:
        target = by_target.get(assay["target_id"])
        if target is None:
            raise ValueError(f"{assay['assay']}: no retrieval record for {assay['target_id']}")
        band = distance_band(target["max_identity_over_query"])
        bands[assay["assay"]] = band
        clusters.setdefault(band, set()).add(int(assay["cluster"]))
    support = {
        "bands": list(DISTANCE_BANDS),
        "edges": list(DISTANCE_EDGES),
        "assays_per_band": {
            band: sum(1 for value in bands.values() if value == band) for band in DISTANCE_BANDS
        },
        "clusters_per_band": {band: len(clusters.get(band, ())) for band in DISTANCE_BANDS},
        "group_floor": GROUP_FLOOR,
        "admitted_bands": [
            band for band in DISTANCE_BANDS if len(clusters.get(band, ())) >= GROUP_FLOOR
        ],
    }
    return bands, support


def stratum_matrix(
    scores: Sequence[Mapping[str, Any]],
    *,
    bands: Mapping[str, str],
    arms: Sequence[str],
    admitted: Sequence[str],
) -> tuple[list[int], list[tuple[str, str]], np.ndarray]:
    """Family-by-(arm, band) matrix of the frozen out-of-fold increment.

    Seeds are averaged within an assay, assays carry equal weight inside a family,
    and a family that supplies no assay to a cell is ``NaN``: structural absence,
    never a zero.
    """

    per_cell: dict[tuple[str, str], dict[int, dict[str, list[float]]]] = {}
    families: set[int] = set()
    for row in scores:
        arm = row["arm"]
        if arm not in set(arms):
            continue
        band = bands.get(row["assay"])
        if band is None or band not in set(admitted):
            continue
        families.add(int(row["cluster"]))
        cell = per_cell.setdefault((arm, band), {}).setdefault(int(row["cluster"]), {})
        cell.setdefault(row["assay"], []).append(float(row["contrast"]))
    order = sorted(families)
    columns = [(arm, band) for arm in arms for band in admitted]
    matrix = np.full((len(order), len(columns)), np.nan)
    for index, column in enumerate(columns):
        per_family = per_cell.get(column, {})
        for position, family in enumerate(order):
            assays = per_family.get(family)
            if assays:
                matrix[position, index] = float(
                    np.mean([float(np.mean(values)) for values in assays.values()])
                )
    return order, columns, matrix


def stratified_panel(
    scores: Sequence[Mapping[str, Any]],
    *,
    bands: Mapping[str, str],
    arms: Sequence[str],
    admitted: Sequence[str],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """The per-arm, per-band increment with its interval and its power."""

    from ..extensions.phenotype_strata import shared_bootstrap

    families, columns, matrix = stratum_matrix(
        scores, bands=bands, arms=arms, admitted=admitted
    )
    keep = [index for index in range(matrix.shape[1]) if np.isfinite(matrix[:, index]).sum() >= 2]
    dropped = [
        {"arm": columns[index][0], "band": columns[index][1], "reason": "fewer than two families"}
        for index in range(matrix.shape[1])
        if index not in keep
    ]
    if not keep:
        return {"status": "no estimable cell", "families": len(families), "dropped": dropped}
    statistics, _ = shared_bootstrap(matrix[:, keep], draws=draws, seed=seed)
    critical = float(statistics["critical_value"])
    cells = []
    for position, index in enumerate(keep):
        arm, band = columns[index]
        point = statistics["point"][position]
        error = statistics["se"][position]
        groups = statistics["supported_families"][position]
        interval = statistics["simultaneous_interval"][position]
        cells.append(
            {
                "arm": arm,
                "band": band,
                **power_record(
                    point=point,
                    standard_error=error,
                    groups=groups,
                    interval=interval,
                    critical=critical,
                ),
                "pointwise_interval": statistics["pointwise_interval"][position],
            }
        )
    return {
        "status": "estimated",
        "endpoint": (
            "within-assay Spearman of the frozen BMPL out-of-fold prediction minus the "
            "same quantity for BPL; positive means the likelihood block adds ranking "
            "information over the baseline blocks"
        ),
        "families": len(families),
        "cells": cells,
        "dropped": dropped,
        "bootstrap": {
            key: statistics[key]
            for key in (
                "draws",
                "seed",
                "confidence",
                "critical_value",
                "family_size",
                "original_families",
                "jointly_rejected_draws",
                "method",
                "conditional_on_fitted_predictions",
            )
        },
    }


def monotonicity(cells: Sequence[Mapping[str, Any]], *, order: Sequence[str]) -> dict[str, Any]:
    """Does the increment fall as distance to the corpus grows, per arm?

    Descriptive and declared as such. The question E10 asks is answered by the
    *resolved* cells, not by the sign of a difference between two unresolved ones,
    so the two are reported apart.
    """

    index = {band: position for position, band in enumerate(order)}
    out: dict[str, Any] = {}
    for arm in sorted({cell["arm"] for cell in cells}):
        rows = sorted(
            (cell for cell in cells if cell["arm"] == arm), key=lambda cell: index[cell["band"]]
        )
        points = [cell["point"] for cell in rows]
        resolved = [cell for cell in rows if cell["resolution"] == "resolved"]
        out[arm] = {
            "bands": [cell["band"] for cell in rows],
            "points": points,
            "resolved_bands": [cell["band"] for cell in resolved],
            "resolved_positive_bands": [
                cell["band"] for cell in resolved if cell["direction"] == "above_zero"
            ],
            # ``order`` runs from the nearest band to the most distant, so the
            # most distant resolved positive is the last one, not the first.
            "most_distant_resolved_positive_band": next(
                (
                    cell["band"]
                    for cell in reversed(rows)
                    if cell["resolution"] == "resolved" and cell["direction"] == "above_zero"
                ),
                None,
            ),
            "decreasing_with_distance": all(
                earlier >= later for earlier, later in zip(points, points[1:])
            )
            if len(points) > 1
            else None,
        }
    return out
