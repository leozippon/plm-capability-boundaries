"""How far a single substitution moves the likelihood, and whether structure explains it.

A position-resolved archive (:mod:`position_likelihood`) turns one mutation score
into a profile: at every residue ``j`` of the wild-type coordinate frame, the
signed change in that position's own log likelihood,
``log p(w_j | mutant context) - log p(w_j | wild-type context)``. This module
reads that profile and asks the two questions E01 is about -- how far along the
sequence the change reaches, and whether the residues that respond are the ones
in three-dimensional contact with the mutated site.

Three things about the profile have to be said before any number is read.

**The site itself is not a response.** At ``j = i`` the token differs between the
two states, so the two terms are the likelihood of *different* residues and their
difference is the mutation score's own leading term, not propagation. It is
reported separately and never enters a distance profile.

**For a causal arm, ``j < i`` is zero by construction.** The prefix that predicts
an upstream residue is untouched by a downstream substitution, so those terms are
the same numbers in both states. :func:`receiver_census` asserts that and
:mod:`position_likelihood` asserts it again at extraction time. The consequence
is not a defect to be worked around: a left-to-right model's likelihood response
is one-sided, and propagation can only be measured downstream. A masked arm,
whose conditional at ``j`` sees the whole remaining sequence, responds on both
sides, which is why one is carried beside the causal panel.

**Contact and sequence distance are confounded.** Residues in contact are on
average closer along the chain, and the likelihood response decays with sequence
distance for reasons that have nothing to do with structure. Every contact
reading here is therefore a contrast *inside* a sequence-separation stratum,
nested so that no large protein, no heavily sampled assay and no densely packed
mutation can carry the panel, and the residual within-stratum separation
imbalance is reported beside the estimate rather than assumed away.

The structural inputs are not re-derived. ``src.capability.extensions.structure``
already mapped the anchor cohort onto experimental mmCIF entries and wrote, for
every wild-type position of every admitted assay, the C-beta coordinate (C-alpha
for glycine), its relative accessibility and its provenance; and
``src.capability.extensions.responses.structure_sites`` already validates that
table against its paired coverage receipt. This module calls that validator and
then does the one thing it does not: form the pair geometry at a declared cutoff
and sequence-separation floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..extensions.responses import DISTANCE_EDGES, DISTANCE_NAMES, structure_sites
from ..interactions.contact_enrichment import CB_CONTACT_ANGSTROM
from ..interactions.pairwise_epistasis import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, interval
from .position_likelihood import CAUSAL, MASKED, state_counts, state_terms
from .position_terms import partition_masks, residue_bounds

SCHEMA = "position_response_v1"

#: Contact cutoff on the C-beta (glycine C-alpha) distance, in angstrom. Taken
#: from ``interactions.contact_enrichment``, which is where this project declared
#: it, so a contact here is the same relation a contact is in R3.
CONTACT_ANGSTROM = float(CB_CONTACT_ANGSTROM)

#: Sequence-separation floor, in residues. A pair closer than this along the
#: chain is in contact for trivial reasons -- consecutive residues and one turn of
#: a helix are always within 8 angstrom -- and carries no information about
#: tertiary structure. Six is the first separation at which an alpha helix no
#: longer guarantees contact.
MIN_SEQUENCE_SEPARATION = 6

CONTACT_DEFINITION = (
    "C-beta, or C-alpha for glycine, strictly below 8.0 angstrom, between two "
    "residues whose wild-type sequence separation is at least 6; a position "
    "without an admitted coordinate is in neither class and is excluded from the "
    "pair population rather than counted as a non-contact"
)


# ------------------------------------------------------------- the geometry


@dataclass(frozen=True)
class SiteGeometry:
    """One wild-type position with an admitted experimental coordinate."""

    j: int
    residue: str
    atom: str
    xyz: tuple[float, float, float]
    rsa: float | None


def admitted_geometry(
    site_rows: Sequence[Mapping[str, Any]], *, assay: str, family: Any, wildtype: str,
    coverage: Mapping[str, Any],
) -> dict[int, SiteGeometry]:
    """Admitted coordinates for one assay, through the existing validator.

    Delegates every provenance, identity and exactness check to
    :func:`responses.structure_sites`: the method must be experimental, the
    source digest must match the coverage receipt, the mapping must be a full
    entity or a full wild-type fragment, and every wild-type position including
    the excluded ones must be present exactly once. This function only indexes
    what that returns.
    """

    usable, _receipt = structure_sites(
        list(site_rows), assay=assay, family=family, wildtype=wildtype, coverage=coverage
    )
    geometry: dict[int, SiteGeometry] = {}
    for site in usable:
        coordinate = np.asarray(site["xyz"], dtype=np.float64)
        if coordinate.shape != (3,) or not np.isfinite(coordinate).all():
            raise ValueError(f"{assay}: position {site['j']} has no finite coordinate")
        geometry[int(site["j"])] = SiteGeometry(
            j=int(site["j"]),
            residue=str(site["identity"]),
            atom=str(site["atom"]),
            xyz=(float(coordinate[0]), float(coordinate[1]), float(coordinate[2])),
            rsa=None if site["rsa"] is None else float(site["rsa"]),
        )
    return geometry


def require_geometry(
    geometry: Mapping[int, SiteGeometry], *, assay: str, minimum: int = 2
) -> None:
    """Refuse an assay whose structure cannot support a pair, naming the assay.

    A missing or unmappable structure is a reason an assay is absent from a
    structural result, never a reason to report a weaker non-structural number
    under the same name.
    """

    if len(geometry) < minimum:
        raise ValueError(
            f"{assay}: {len(geometry)} admitted structural coordinate(s) is below the "
            f"{minimum} a contact pair needs; this assay has no structural support and "
            "is refused rather than analysed without it"
        )


def contact_pairs(
    geometry: Mapping[int, SiteGeometry], *, cutoff: float = CONTACT_ANGSTROM,
    min_separation: int = MIN_SEQUENCE_SEPARATION,
) -> dict[str, Any]:
    """Every eligible position pair of one protein, classified.

    A pair is eligible when both positions carry an admitted coordinate and their
    sequence separation is at least ``min_separation``. It is a contact when the
    distance is **strictly** below ``cutoff``. Nothing is clamped and nothing is
    imputed: a position whose coordinate is absent appears in neither class, and
    the count of such positions is returned so that a reader can see the support
    the geometry actually carries.
    """

    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("the contact cutoff is a positive finite distance")
    if min_separation < 1:
        raise ValueError("the sequence-separation floor is at least one residue")
    positions = sorted(int(key) for key in geometry)
    pairs: list[tuple[int, int, float]] = []
    for first in range(len(positions)):
        for second in range(first + 1, len(positions)):
            low, high = positions[first], positions[second]
            separation = high - low
            if separation < min_separation:
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(geometry[high].xyz) - np.asarray(geometry[low].xyz)
                )
            )
            if not np.isfinite(distance):
                # A non-finite distance compares false against the cutoff, so an
                # unchecked coordinate would silently become a *non-contact* and
                # enter the control side of every contrast. Refused here, where the
                # relation is formed, rather than trusted to the caller's validator.
                raise ValueError(
                    f"positions {low} and {high} have no finite C-beta distance; a "
                    "coordinate this pair cannot be formed from is a refusal, not a "
                    "non-contact"
                )
            pairs.append((low, high, distance))
    contacts = {(low, high) for low, high, distance in pairs if distance < cutoff}
    return {
        "pairs": pairs,
        "contacts": contacts,
        "distance": {(low, high): distance for low, high, distance in pairs},
        "cutoff_angstrom": float(cutoff),
        "min_separation": int(min_separation),
        "eligible_positions": positions,
        "n_pairs": len(pairs),
        "n_contacts": len(contacts),
        "definition": CONTACT_DEFINITION,
    }


def separation_stratum(separation: int) -> str:
    """The declared sequence-separation stratum, reusing the frozen edges."""

    return DISTANCE_NAMES[int(np.searchsorted(DISTANCE_EDGES, int(separation), side="left"))]


# --------------------------------------------------------- the response profile


def rebuild_states(payload: Mapping[str, Any], sequences: Sequence[str]) -> list[dict[str, Any]]:
    """The packing sidecar, read back out of the archive that already carries it.

    ``RetainedResponses`` takes the packing as a separate JSON because the
    archives it was written for did not retain packed ids. Ours do, so the
    sidecar is reconstructed rather than re-derived from a tokeniser: there is no
    second packing and therefore nothing for a second packing to disagree with.
    """

    offsets = np.asarray(payload["position_offsets"], dtype=np.int64)
    id_offsets = np.asarray(payload["state_id_offsets"], dtype=np.int64)
    spans = np.asarray(payload["state_spans"], dtype=np.int64)
    residue_offsets = np.asarray(payload["position_residue_offsets"], dtype=np.int64)
    if len(sequences) != len(residue_offsets):
        raise ValueError("the archive does not hold one state per supplied sequence")
    states = []
    for index, sequence in enumerate(sequences):
        low, high = int(id_offsets[index]), int(id_offsets[index + 1])
        states.append(
            {
                "sequence": sequence,
                "ids": [int(value) for value in payload["state_ids"][low:high]],
                "span": [int(spans[index][0]), int(spans[index][1])],
                "counts": [int(value) for value in state_counts(payload, index)],
                "offset": int(residue_offsets[index]),
            }
        )
        if offsets[index + 1] - offsets[index] != spans[index][1] - spans[index][0]:
            raise ValueError("the archive's term count disagrees with its own scored span")
    return states


def receiver_census(
    payload: Mapping[str, Any], index: int, *, site: int, paradigm: str,
) -> dict[str, Any]:
    """The signed per-residue response of one single substitution, both directions.

    The partition is the residue-axis one :mod:`position_terms` declares. A
    receiver is a scored token covering exactly one residue that is not the
    mutated residue's own token; tokens covering several residues and tokens
    covering none are summed into named remainders instead of being attributed to
    a position, because a byte-pair merge spanning three residues is not a
    measurement at any one of them.

    The returned ``upstream_max_abs_nats`` is the invariant: zero is required of a
    causal arm and recorded for a masked one.
    """

    if paradigm not in (CAUSAL, MASKED):
        raise ValueError(f"unknown paradigm {paradigm!r}")
    wild_terms = state_terms(payload, 0).astype(np.float64)
    mutant_terms = state_terms(payload, index + 1).astype(np.float64)
    counts = state_counts(payload, 0)
    offset = int(payload["position_residue_offsets"][0])
    if wild_terms.shape != mutant_terms.shape:
        raise ValueError(
            "the two states do not share a token grid; this variant is not aligned and "
            "has no per-position correspondence"
        )
    delta = wild_terms - mutant_terms
    starts, ends = residue_bounds(counts, offset)
    widths = ends - starts
    own = partition_masks(counts, offset, site)["own"]
    single = (widths == 1) & ~own
    upstream = single & (starts < site)
    downstream = single & (starts > site)
    rows = []
    for token in np.flatnonzero(single):
        j = int(starts[token])
        rows.append(
            {
                "j": j,
                "separation": j - int(site),
                "direction": "downstream" if j > site else "upstream",
                "response": float(delta[token]),
                "wt_nll": float(wild_terms[token]),
                "mutant_nll": float(mutant_terms[token]),
            }
        )
    worst_upstream = float(np.max(np.abs(delta[upstream]))) if upstream.any() else 0.0
    if paradigm == CAUSAL and worst_upstream != 0.0:
        raise ValueError(
            f"a causal arm's terms upstream of position {site} differ by "
            f"{worst_upstream} nats; the prefix is identical, so this is a packing, "
            "alignment or forward defect and not a measurement"
        )
    return {
        "i": int(site),
        "site_response": float(delta[own].sum()),
        "site_tokens": int(own.sum()),
        "receivers": rows,
        "upstream_receivers": int(upstream.sum()),
        "downstream_receivers": int(downstream.sum()),
        "upstream_max_abs_nats": worst_upstream,
        "multiresidue_remainder": float(delta[(widths > 1) & ~own].sum()),
        "unresidued_remainder": float(delta[(widths == 0) & ~own].sum()),
        "total": float(delta.sum()),
    }


def agrees_with_frozen(census: Mapping[str, Any], frozen: Mapping[str, Any]) -> dict[str, Any]:
    """Compare this census against ``responses.RetainedResponses.response``.

    The frozen reader is the project's prior authority on what a downstream
    receiver is, and it reports only downstream single-residue tokens. This census
    extends it upstream, so the two must agree exactly where they overlap; a
    disagreement means one of them is reading a different partition and neither
    number may be published.
    """

    mine = {row["j"]: row["response"] for row in census["receivers"] if row["separation"] > 0}
    theirs = {int(row["j"]): float(row["response"]) for row in frozen["receivers"]}
    if set(mine) != set(theirs):
        raise ValueError(
            "the downstream receiver sets disagree with the frozen reader: "
            f"{sorted(set(mine) ^ set(theirs))[:8]}"
        )
    worst = max((abs(mine[j] - theirs[j]) for j in mine), default=0.0)
    if worst != 0.0:
        raise ValueError(f"downstream receiver responses differ from the frozen reader by {worst}")
    return {"downstream_receivers": len(mine), "max_abs_difference_nats": worst}


class ProfileAccumulator:
    """Running per-separation statistics of the likelihood response.

    Streaming rather than row-storing, because a full-anchor panel produces a few
    million receivers per arm and the profile needs only three sums per
    separation. Both the signed and the absolute mean are kept, because they
    answer different questions: the signed mean says whether a substitution
    systematically raises or lowers its neighbours' likelihood, and the absolute
    mean says how far any perturbation at all reaches.
    """

    def __init__(self, *, direction: str, min_support: int = 20) -> None:
        if direction not in ("downstream", "upstream"):
            raise ValueError("direction is downstream or upstream")
        self.direction = direction
        self.min_support = int(min_support)
        self._count: dict[int, int] = {}
        self._total: dict[int, float] = {}
        self._absolute: dict[int, float] = {}
        self._mutations = 0

    def add(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            if row["direction"] != self.direction:
                continue
            separation = abs(int(row["separation"]))
            value = float(row["response"])
            self._count[separation] = self._count.get(separation, 0) + 1
            self._total[separation] = self._total.get(separation, 0.0) + value
            self._absolute[separation] = self._absolute.get(separation, 0.0) + abs(value)

    def count_mutation(self) -> None:
        self._mutations += 1

    def profile(self) -> dict[str, Any]:
        separations = [
            {
                "separation": int(separation),
                "n": int(self._count[separation]),
                "mean_signed_nats": self._total[separation] / self._count[separation],
                "mean_absolute_nats": self._absolute[separation] / self._count[separation],
                "sufficient": bool(self._count[separation] >= self.min_support),
            }
            for separation in sorted(self._count)
        ]
        return {
            "direction": self.direction,
            "mutations": int(self._mutations),
            "receivers": int(sum(self._count.values())),
            "max_separation": max(self._count, default=0),
            "separations": separations,
            "decay": decay_fit(separations),
            "min_support": self.min_support,
        }


def response_profile(
    rows: Sequence[Mapping[str, Any]], *, direction: str, min_support: int = 20,
) -> dict[str, Any]:
    """:class:`ProfileAccumulator` over one in-memory row set, for small supports."""

    accumulator = ProfileAccumulator(direction=direction, min_support=min_support)
    accumulator.add(rows)
    return accumulator.profile()


def decay_fit(profile: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Log-linear decay of the mean absolute response with sequence separation.

    A one-parameter description, not a mechanism: ``log mean|response| = a + b d``
    over the separations carrying enough receivers. ``half_distance_residues`` is
    ``ln 2 / -b``, reported only when the decay is actually resolved: the profile
    must vary with separation at all, and the slope must be negative by more than
    twice its own standard error. A perfectly flat profile still leaves a slope of
    order 1e-17 in the least-squares solve, and dividing into that yields a
    half-distance of 1e16 residues -- a number, but not a distance.
    """

    points = [row for row in profile if row["sufficient"] and row["mean_absolute_nats"] > 0]
    if len(points) < 3:
        return {
            "fitted": False,
            "reason": f"{len(points)} separations carry a positive mean with sufficient support",
            "separations_used": int(len(points)),
            "half_distance_residues": None,
            "half_distance_undefined_reason": "the profile was not fitted",
        }
    separation = np.asarray([row["separation"] for row in points], dtype=np.float64)
    response = np.log(np.asarray([row["mean_absolute_nats"] for row in points], dtype=np.float64))
    total = float(((response - response.mean()) ** 2).sum())
    if total == 0.0:
        return {
            "fitted": False,
            "reason": "the mean absolute response does not vary with separation",
            "separations_used": int(len(points)),
            "half_distance_residues": None,
            "half_distance_undefined_reason": "the profile was not fitted",
        }
    design = np.vstack([np.ones_like(separation), separation]).T
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    intercept, slope = float(coefficients[0]), float(coefficients[1])
    predicted = design @ coefficients
    residual = float(((response - predicted) ** 2).sum())
    variance = residual / max(len(points) - 2, 1)
    standard_error = float(np.sqrt(variance * np.linalg.inv(design.T @ design)[1, 1]))
    resolved = bool(slope < 0 and abs(slope) > 2.0 * standard_error)
    return {
        "fitted": True,
        "separations_used": int(len(points)),
        "intercept_log_nats": intercept,
        "slope_per_residue": slope,
        "slope_standard_error": standard_error,
        "r_squared": float(1 - residual / total),
        "half_distance_residues": float(np.log(2) / -slope) if resolved else None,
        "half_distance_undefined_reason": (
            None if resolved
            else "the fitted slope is not negative by more than twice its standard error"
        ),
    }


# ------------------------------------------------- the structural contrast


def structural_rows(
    census: Mapping[str, Any], geometry: Mapping[int, SiteGeometry], pairs: Mapping[str, Any],
    *, assay: str, family: Any, mutation: str,
) -> list[dict[str, Any]]:
    """Join one mutation's receivers to the pair geometry of its own protein.

    A receiver enters only when both it and the mutated site carry an admitted
    coordinate and their separation clears the floor; otherwise the pair is not in
    the eligible population and is dropped rather than called a non-contact.
    """

    site = int(census["i"])
    if site not in geometry:
        return []
    floor = int(pairs["min_separation"])
    rows = []
    for row in census["receivers"]:
        j = int(row["j"])
        if j not in geometry or abs(int(row["separation"])) < floor:
            continue
        key = (min(site, j), max(site, j))
        distance = pairs["distance"].get(key)
        if distance is None:
            continue
        rows.append(
            {
                "assay": assay,
                "family": family,
                "mutation": mutation,
                "i": site,
                "j": j,
                "separation": abs(int(row["separation"])),
                "direction": row["direction"],
                "response": float(row["response"]),
                "absolute_response": abs(float(row["response"])),
                "contact": bool(distance < pairs["cutoff_angstrom"]),
                "structure_distance_angstrom": float(distance),
                "rsa": geometry[j].rsa,
                "stratum": separation_stratum(abs(int(row["separation"]))),
            }
        )
    return rows


def _nested_mean(values: Sequence[tuple[Any, float]]) -> float | None:
    """Mean over the distinct keys, each key's own rows averaged first."""

    grouped: dict[Any, list[float]] = {}
    for key, value in values:
        grouped.setdefault(key, []).append(float(value))
    if not grouped:
        return None
    return float(np.mean([float(np.mean(rows)) for rows in grouped.values()]))


def stratified_contact_contrast(
    rows: Sequence[Mapping[str, Any]], *, outcome: str = "absolute_response",
    direction: str | None = None, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Contact minus non-contact response, matched inside separation strata.

    The nesting is the one this project already declared for the geometry-only
    pair census: a contrast is formed inside one mutation and one separation
    stratum, those are averaged equally within the mutation, mutations equally
    within the assay, assays equally within the family, and the family values are
    the bootstrap units. A stratum in which the mutation has only contacts or only
    non-contacts contributes nothing, so no comparison is ever made across
    separations.

    The residual within-stratum separation imbalance is reported on the same
    weighting. A contrast whose imbalance is large is a contrast about sequence
    distance wearing a structural label, and the reader is given both numbers.
    """

    if outcome not in ("absolute_response", "response"):
        raise ValueError("outcome is absolute_response or response")
    selected = [row for row in rows if direction is None or row["direction"] == direction]
    cells: dict[tuple[Any, Any, str, str], list[Mapping[str, Any]]] = {}
    for row in selected:
        key = (row["family"], row["assay"], row["mutation"], row["stratum"])
        cells.setdefault(key, []).append(row)
    per_mutation: dict[tuple[Any, Any, str], list[tuple[str, float]]] = {}
    per_mutation_imbalance: dict[tuple[Any, Any, str], list[tuple[str, float]]] = {}
    matched_cells = 0
    matched_receivers = 0
    for (family, assay, mutation, stratum), members in cells.items():
        contacts = [row for row in members if row["contact"]]
        others = [row for row in members if not row["contact"]]
        if not contacts or not others:
            continue
        matched_cells += 1
        matched_receivers += len(contacts) + len(others)
        difference = float(np.mean([row[outcome] for row in contacts])) - float(
            np.mean([row[outcome] for row in others])
        )
        imbalance = float(np.mean([row["separation"] for row in contacts])) - float(
            np.mean([row["separation"] for row in others])
        )
        per_mutation.setdefault((family, assay, mutation), []).append((stratum, difference))
        per_mutation_imbalance.setdefault((family, assay, mutation), []).append(
            (stratum, imbalance)
        )

    def _collapse(table: Mapping[tuple[Any, Any, str], list[tuple[str, float]]]) -> dict[Any, float]:
        per_assay: dict[tuple[Any, Any], list[tuple[str, float]]] = {}
        for (family, assay, mutation), entries in table.items():
            value = _nested_mean(entries)
            if value is not None:
                per_assay.setdefault((family, assay), []).append((mutation, value))
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

    family_values = _collapse(per_mutation)
    family_imbalance = _collapse(per_mutation_imbalance)
    estimate = interval(sorted(family_values.values()), draws=draws, seed=seed) if family_values else {
        "point": None,
        "interval": None,
        "groups": 0,
        "undefined": "no family carries a matched contact and non-contact receiver",
    }
    return {
        "outcome": outcome,
        "direction": direction or "both",
        "definition": CONTACT_DEFINITION,
        "weighting": (
            "equal separation strata within mutation, equal mutations within assay, "
            "equal assays within family, equal families; families are the bootstrap units"
        ),
        "estimate_nats": estimate,
        "families": sorted(family_values),
        "matched_cells": matched_cells,
        "matched_receivers": matched_receivers,
        "separation_imbalance_residues": (
            float(np.mean(sorted(family_imbalance.values()))) if family_imbalance else None
        ),
        "strata_present": sorted({row["stratum"] for row in selected}),
    }
