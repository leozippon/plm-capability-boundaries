"""The backbone panel, its prescribed pairs, and the reciprocal transplant matching.

Three things happen here, in this order, and each one reports how many candidates
it removed so the funnel is visible rather than summarised.

**The panel.** AlphaFold models, covered by exactly one fragment, in one of the
two declared length bands, cross-referenced to a reviewed Swiss-Prot entry whose
sequence the model reproduces exactly, canonical residues only, confident, not
repetitive, assigned exactly one CATH superfamily, one backbone per superfamily,
and pairwise below the identity ceiling. Nothing here is re-derived: the model
reader, the mmCIF reader, the accessibility implementation, the CA-trace
secondary-structure assignment, the CATH table and the shingle machinery are all
the declarations this repository already holds.

**The prescribed pairs.** For every admitted backbone, the eligible anchor and
partner pairs -- a reference contact at separation at least the floor, with the
anchor inside the declared window -- each carried with
:data:`~forcing_design.N_REFERENCE_PARTNERS` non-contacting reference partners
matched on sequence separation and on the partner's relative-accessibility band.
A pair whose controls cannot be filled is not a pair: without them the endpoint
has no second term and would reduce to an unmatched divergence.

**The reciprocal transplant.** Units are matched within cells of (length band,
anchor burial band, anchor secondary-structure class, separation stratum) to a
unit of a *different* backbone whose anchor residue differs. The matching is
reciprocal, so the multiset of forced residues is identical between the two
conditions by construction; :func:`~forcing_design.forced_residue_multisets_match`
checks that on the realised cohort.

The selection is blind to every model: the preference order over candidate pairs
is reference C-beta distance, then separation, then position. No likelihood, no
confidence and no generated sequence enters the cohort.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core.amino_acids import AA20
from ..generation.near_duplicates import RESIDUE_SHINGLE, shingles
from ..generation.structure_inputs import ca_secondary_structure, read_alphafold_model
from ..interactions.contact_enrichment import (
    load_structure,
    relative_accessibility,
    residue_accessibility,
)
from . import forcing_design as D

SCHEMA = "forcing_backbone_cohort_v1"


# -------------------------------------------------------------- one backbone


@dataclass(frozen=True)
class Backbone:
    """One reference backbone, with everything the design keys on.

    Positions are 0-based indices into :attr:`sequence` throughout this package.
    The reference files are read twice on purpose: the PDB model carries the
    pLDDT in its B-factor column and the CA trace the secondary-structure
    assignment is defined on, and the mmCIF carries the heavy atoms that the
    C-beta geometry and the accessibility need. The two must agree on the
    sequence, which :func:`read_backbone` refuses to assume.
    """

    accession: str
    superfamily: str
    sequence: str
    band: str
    plddt: np.ndarray
    contact_xyz: np.ndarray
    contact_atom: tuple[str, ...]
    rsa: np.ndarray
    secondary: np.ndarray
    distinct_shingle_fraction: float

    def __post_init__(self) -> None:
        length = len(self.sequence)
        for name, array in (
            ("plddt", self.plddt), ("rsa", self.rsa), ("secondary", self.secondary),
        ):
            if array.shape != (length,):
                raise ValueError(f"{self.accession}: {name} does not align with the sequence")
        if self.contact_xyz.shape != (length, 3):
            raise ValueError(f"{self.accession}: contact atoms do not align with the sequence")
        if len(self.contact_atom) != length:
            raise ValueError(f"{self.accession}: atom names do not align with the sequence")
        if not np.isfinite(self.contact_xyz).all() or not np.isfinite(self.plddt).all():
            raise ValueError(f"{self.accession}: non-finite coordinates or confidence")
        if not np.isfinite(self.rsa).all():
            raise ValueError(f"{self.accession}: non-finite relative accessibility")

    def __len__(self) -> int:
        return len(self.sequence)

    @property
    def mean_plddt(self) -> float:
        return float(self.plddt.mean())


def distinct_shingle_fraction(sequence: str, *, length: int = RESIDUE_SHINGLE) -> float:
    """Distinct residue shingles over the number of shingle positions.

    The per-draw repeat and low-complexity flag is the same quantity on the
    completion, so a backbone and a draw are screened by one definition.
    """

    positions = len(sequence) - int(length) + 1
    if positions < 1:
        return 0.0
    return len(shingles(sequence, unit="residues", length=length)) / positions


def read_backbone(
    *, pdb: Path, cif: Path, superfamily: str, band: str
) -> Backbone:
    """One backbone from its paired AlphaFold files, refusing any disagreement."""

    model = read_alphafold_model(Path(pdb))
    if model.n_non_canonical_residues:
        raise ValueError(
            f"{model.accession}: {model.n_non_canonical_residues} non-canonical residue(s); "
            "this design's alphabet is the canonical twenty and a model carrying more is "
            "refused rather than truncated"
        )
    structure = load_structure(Path(cif))
    chains = structure["chains"]
    if len(chains) != 1:
        raise ValueError(f"{model.accession}: mmCIF holds {len(chains)} chains, expected one")
    chain = chains[next(iter(chains))]
    order = sorted(chain.residues)
    sequence = "".join(chain.residues[position].residue for position in order)
    if sequence != model.sequence:
        raise ValueError(
            f"{model.accession}: the mmCIF sequence and the PDB sequence disagree; the two "
            "reference files do not describe one model"
        )
    coordinates = np.empty((len(order), 3), dtype=np.float64)
    atoms: list[str] = []
    for index, position in enumerate(order):
        residue = chain.residues[position]
        name = "CA" if residue.residue == "G" else "CB"
        coordinate = residue.named(name)
        if coordinate is None:
            raise ValueError(
                f"{model.accession}: residue {position} has no {name} atom; a position the "
                "contact relation cannot be formed from is a refusal, not a non-contact"
            )
        coordinates[index] = coordinate
        atoms.append(name)
    areas = residue_accessibility(chain)
    accessibility = np.asarray(
        [relative_accessibility(areas[position], chain.residues[position].residue)
         for position in order],
        dtype=np.float64,
    )
    return Backbone(
        accession=model.accession,
        superfamily=str(superfamily),
        sequence=sequence,
        band=str(band),
        plddt=model.plddt,
        contact_xyz=coordinates,
        contact_atom=tuple(atoms),
        rsa=accessibility,
        secondary=ca_secondary_structure(model.ca),
        distinct_shingle_fraction=distinct_shingle_fraction(sequence),
    )


# --------------------------------------------------------- the prescribed pairs


def anchor_window(length: int) -> tuple[int, int]:
    """Inclusive 0-based bounds on the anchor, as the design declares them.

    The lower bound is the declared fraction of the length, so the model is given
    a real prefix. The upper bound is the last position from which a partner at
    the separation floor still exists.
    """

    low = int(np.ceil(D.ANCHOR_WINDOW_FRACTION * int(length)))
    high = int(length) - 1 - D.MIN_SEPARATION
    return low, high


def pair_distances(backbone: Backbone) -> np.ndarray:
    """The full C-beta distance matrix of one backbone."""

    delta = backbone.contact_xyz[:, None, :] - backbone.contact_xyz[None, :, :]
    distance = np.sqrt((delta**2).sum(-1))
    if not np.isfinite(distance).all():
        raise ValueError(f"{backbone.accession}: non-finite C-beta distance")
    return distance


def candidate_pairs(backbone: Backbone) -> list[dict[str, Any]]:
    """Every eligible prescribed pair of one backbone, with its reference partners.

    A candidate needs an anchor inside the window, a forward partner at or beyond
    the separation floor whose reference C-beta distance is below the contact
    cutoff, confidence at or above the residue floor at the anchor and at every
    position used, and :data:`~forcing_design.N_REFERENCE_PARTNERS` non-contacting
    partners at matched separation and matched partner burial band. Candidates are
    returned in the declared preference order.
    """

    length = len(backbone)
    low, high = anchor_window(length)
    if high < low:
        return []
    distance = pair_distances(backbone)
    confident = backbone.plddt >= D.RESIDUE_PLDDT_FLOOR
    rows: list[dict[str, Any]] = []
    for anchor in range(low, high + 1):
        if not confident[anchor] or backbone.sequence[anchor] not in AA20:
            continue
        for partner in range(anchor + D.MIN_SEPARATION, length):
            if not confident[partner] or backbone.sequence[partner] not in AA20:
                continue
            if distance[anchor, partner] >= D.CONTACT_ANGSTROM:
                continue
            separation = partner - anchor
            references = reference_partners(
                backbone, anchor=anchor, partner=partner, distance=distance,
                confident=confident,
            )
            if len(references) < D.N_REFERENCE_PARTNERS:
                continue
            span_end = max([partner, *(row["position"] for row in references)])
            window = backbone.plddt[anchor: span_end + 1]
            if float(window.mean()) < D.WINDOW_PLDDT_FLOOR:
                continue
            rows.append(
                {
                    "accession": backbone.accession,
                    "superfamily": backbone.superfamily,
                    "length": length,
                    "length_band": backbone.band,
                    "anchor": anchor,
                    "partner": partner,
                    "separation": separation,
                    "separation_stratum": D.separation_stratum(separation),
                    "anchor_residue": backbone.sequence[anchor],
                    "partner_residue": backbone.sequence[partner],
                    "anchor_rsa": float(backbone.rsa[anchor]),
                    "anchor_rsa_band": D.rsa_band(float(backbone.rsa[anchor])),
                    "anchor_ss_class": D.ss_class(int(backbone.secondary[anchor])),
                    "partner_rsa": float(backbone.rsa[partner]),
                    "partner_rsa_band": D.rsa_band(float(backbone.rsa[partner])),
                    "reference_distance_angstrom": float(distance[anchor, partner]),
                    "anchor_plddt": float(backbone.plddt[anchor]),
                    "partner_plddt": float(backbone.plddt[partner]),
                    "reference_partners": references,
                    "span_end": span_end,
                    "window_mean_plddt": float(window.mean()),
                }
            )
    rows.sort(key=lambda row: (
        row["reference_distance_angstrom"], -row["separation"], row["anchor"], row["partner"]
    ))
    return rows


def reference_partners(
    backbone: Backbone, *, anchor: int, partner: int, distance: np.ndarray,
    confident: np.ndarray,
) -> list[dict[str, Any]]:
    """Non-contacting reference partners for one prescribed pair.

    Matched to the prescribed partner on sequence separation, within the declared
    tolerance, and on relative-accessibility band in the reference structure --
    the burial match is the specific lesson from E02, where burial band carried
    the whole of the measured effect. A position between the contact cutoff and
    the non-contact floor is in neither class: it is not admitted here and it is
    not relabelled a contact.

    Returned in order of decreasing reference distance, so the admitted controls
    are the least ambiguous non-contacts available at the matched separation.
    """

    target = partner - anchor
    band = D.rsa_band(float(backbone.rsa[partner]))
    rows: list[dict[str, Any]] = []
    for candidate in range(anchor + D.MIN_SEPARATION, len(backbone)):
        if candidate == partner or not confident[candidate]:
            continue
        if backbone.sequence[candidate] not in AA20:
            continue
        separation = candidate - anchor
        if abs(separation - target) > D.SEPARATION_TOLERANCE:
            continue
        if distance[anchor, candidate] < D.NON_CONTACT_ANGSTROM:
            continue
        if D.rsa_band(float(backbone.rsa[candidate])) != band:
            continue
        rows.append(
            {
                "position": candidate,
                "separation": separation,
                "residue": backbone.sequence[candidate],
                "rsa": float(backbone.rsa[candidate]),
                "rsa_band": D.rsa_band(float(backbone.rsa[candidate])),
                "reference_distance_angstrom": float(distance[anchor, candidate]),
                "plddt": float(backbone.plddt[candidate]),
            }
        )
    rows.sort(key=lambda row: (-row["reference_distance_angstrom"], row["position"]))
    return rows[: D.N_REFERENCE_PARTNERS]


def select_pairs(
    backbone: Backbone, *, limit: int = D.PAIRS_PER_BACKBONE
) -> list[dict[str, Any]]:
    """Up to ``limit`` prescribed pairs of one backbone, with distinct anchors.

    Distinct anchors rather than the ``limit`` best pairs of one anchor: three
    partners of one anchor are three readings of one intervention, and the design
    counts a backbone's pairs as repeated interventions on it. Selection depends
    on this backbone alone, so it does not move when the rest of the cohort does.
    """

    chosen: list[dict[str, Any]] = []
    used_anchors: set[int] = set()
    used_positions: set[int] = set()
    for row in candidate_pairs(backbone):
        if row["anchor"] in used_anchors or row["partner"] in used_positions:
            continue
        chosen.append(row)
        used_anchors.add(row["anchor"])
        used_positions.add(row["partner"])
        if len(chosen) == int(limit):
            break
    return chosen


# -------------------------------------------------------- the transplant match


def match_cell(unit: Mapping[str, Any]) -> tuple:
    return tuple(str(unit[key]) for key in D.TRANSPLANT_MATCH_KEYS)


def reciprocal_transplant(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pair units reciprocally inside their match cells, on differing anchor residues.

    Inside one cell the units are grouped by anchor residue and the two largest
    groups are repeatedly paired, which is the maximum matching of a set under
    "the two members must carry different labels": any matching leaves at least
    ``2 * largest - total`` members of the largest group unpaired, and this rule
    attains that bound. A pair is refused when both members come from one
    backbone, because a swap inside one protein would put both conditions on the
    same prefix population.

    Unmatched units are returned rather than discarded silently: they are the
    reason the realised pair count falls short of the design's target, and a
    reader is owed the number.
    """

    by_cell: dict[tuple, list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        if any(unit[key] is None for key in D.TRANSPLANT_MATCH_KEYS):
            continue
        by_cell[match_cell(unit)].append(index)
    pairs: list[tuple[int, int]] = []
    matched: set[int] = set()
    for cell in sorted(by_cell):
        members = sorted(
            by_cell[cell],
            key=lambda index: (str(units[index]["accession"]), int(units[index]["anchor"])),
        )
        groups: dict[str, list[int]] = {}
        for index in members:
            groups.setdefault(str(units[index]["anchor_residue"]), []).append(index)
        while True:
            order = sorted(
                (residue for residue, items in groups.items() if items),
                key=lambda residue: (-len(groups[residue]), residue),
            )
            if len(order) < 2:
                break
            # The largest residue group first, against the next largest it can
            # legally pair with. Taking the two largest is the maximum matching
            # of a set under "the two members carry different labels"; the inner
            # scan only moves off that choice when every candidate in a group
            # comes from the same backbone as the member being paired.
            choice: tuple[str, int, str, int] | None = None
            head = order[0]
            for other in order[1:]:
                for left in range(len(groups[head])):
                    for right in range(len(groups[other])):
                        if (units[groups[head][left]]["accession"]
                                != units[groups[other][right]]["accession"]):
                            choice = (head, left, other, right)
                            break
                    if choice is not None:
                        break
                if choice is not None:
                    break
            if choice is None:
                break
            head, left, other, right = choice
            first = groups[head].pop(left)
            second = groups[other].pop(right)
            pairs.append((first, second))
            matched.update((first, second))
    assignment: list[dict[str, Any]] = []
    for first, second in pairs:
        for own, other in ((first, second), (second, first)):
            assignment.append(
                dict(units[own])
                | {
                    "unit_id": unit_id(units[own]),
                    "partner_unit_id": unit_id(units[other]),
                    "partner_accession": units[other]["accession"],
                    "native_residue": units[own]["anchor_residue"],
                    "transplant_residue": units[other]["anchor_residue"],
                    "match_cell": dict(zip(D.TRANSPLANT_MATCH_KEYS, match_cell(units[own]))),
                }
            )
    assignment.sort(key=lambda unit: unit["unit_id"])
    unmatched = [unit_id(units[index]) for index in range(len(units)) if index not in matched]
    return {
        "units": assignment,
        "unmatched_unit_ids": sorted(unmatched),
        "cells": {"|".join(cell): len(members) for cell, members in sorted(by_cell.items())},
        "rule": (
            "inside a cell of (length band, anchor burial band, anchor "
            "secondary-structure class, separation stratum), repeatedly pair the two "
            "largest anchor-residue groups, refusing a pair drawn from one backbone"
        ),
    }


def unit_id(unit: Mapping[str, Any]) -> str:
    return f"{unit['accession']}:{int(unit['anchor'])}:{int(unit['partner'])}"


# ----------------------------------------------------------------- the census


class Census:
    """An ordered record of how many candidates each filter removed.

    A funnel rather than a pass/fail: a cohort that fails its quota has to say
    which filter took the candidates away, and a summary count cannot.
    """

    def __init__(self, order: Sequence[str] = D.FILTER_ORDER) -> None:
        self._order = tuple(order)
        self._counts: dict[str, int] = {}
        self._notes: dict[str, str] = {}

    def record(self, name: str, surviving: int, note: str | None = None) -> None:
        if name not in self._order:
            raise ValueError(f"undeclared filter {name!r}; declared: {self._order}")
        self._counts[name] = int(surviving)
        if note is not None:
            self._notes[name] = note

    def payload(self) -> dict[str, Any]:
        rows = []
        previous: int | None = None
        for name in self._order:
            if name not in self._counts:
                continue
            surviving = self._counts[name]
            rows.append(
                {
                    "filter": name,
                    "surviving": surviving,
                    "removed": None if previous is None else previous - surviving,
                    "note": self._notes.get(name),
                }
            )
            previous = surviving
        return {"order": list(self._order), "stages": rows}


# -------------------------------------------------------------- the cohort file


def cohort_payload(
    units: Sequence[Mapping[str, Any]], *, census: Mapping[str, Any],
    sources: Mapping[str, Any], sequences: Mapping[str, str],
    coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The frozen cohort artefact every later stage reads.

    The wild-type sequences travel with the cohort rather than being looked up
    again downstream. The generation stage runs in a pod that holds neither the
    CATH table nor the Swiss-Prot cross-reference this panel was selected with, so
    a stage that re-derived the sequence would be deriving it from a different
    source than the one the selection used.
    """

    accessions = sorted({str(unit["accession"]) for unit in units})
    missing = [accession for accession in accessions if accession not in sequences]
    if missing:
        raise ValueError(f"the cohort carries no sequence for {missing[:5]}")
    by_band = Counter(str(unit["length_band"]) for unit in units)
    multiset = D.forced_residue_multisets_match(units)
    return {
        "schema": SCHEMA,
        "pre_registration": D.pre_registration(units=len(accessions), pairs=len(units)),
        "backbones": len(accessions),
        "backbones_per_band": dict(
            Counter(
                str(unit["length_band"])
                for unit in {str(u["accession"]): u for u in units}.values()
            )
        ),
        "units": len(units),
        "units_per_band": dict(by_band),
        "pairs_per_backbone": dict(
            Counter(Counter(str(unit["accession"]) for unit in units).values())
        ),
        "forced_residue_multisets": multiset,
        "superfamilies": sorted({str(unit["superfamily"]) for unit in units}),
        "accessions": accessions,
        "sequences": {accession: str(sequences[accession]) for accession in accessions},
        "census": dict(census),
        "sources": dict(sources),
        "reference_database_coverage": dict(coverage) if coverage else None,
        "unit_rows": [dict(unit) for unit in units],
        "geometry": {
            "contact_angstrom": D.CONTACT_ANGSTROM,
            "non_contact_angstrom": D.NON_CONTACT_ANGSTROM,
            "min_separation": D.MIN_SEPARATION,
            "separation_tolerance": D.SEPARATION_TOLERANCE,
            "anchor_window_fraction": D.ANCHOR_WINDOW_FRACTION,
            "rsa_boundary": D.RSA_BOUNDARY,
            "rsa_method": D.RSA_METHOD,
            "reference_partners": D.N_REFERENCE_PARTNERS,
        },
    }


def require_cohort(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse a cohort whose own guarantees do not hold.

    The reciprocal swap's multiset equality is the design's central claim, so a
    cohort that does not satisfy it is not usable at any sample size and the
    refusal happens here rather than being discovered in the analysis.
    """

    if payload.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} cohort: {payload.get('schema')!r}")
    if not payload["forced_residue_multisets"]["matches"]:
        raise ValueError(
            "the forced-residue multisets of the two conditions differ; the reciprocal "
            "swap is not closed over these units and the counterfactual is not "
            "composition-matched"
        )
    ids = [unit["unit_id"] for unit in payload["unit_rows"]]
    if len(set(ids)) != len(ids):
        raise ValueError("the cohort repeats a unit id")
    index = {unit["unit_id"]: unit for unit in payload["unit_rows"]}
    for unit in payload["unit_rows"]:
        sequence = payload["sequences"].get(unit["accession"])
        if sequence is None or len(sequence) != int(unit["length"]):
            raise ValueError(f"{unit['unit_id']}: its backbone's sequence is absent or the wrong length")
        if sequence[int(unit["anchor"])] != unit["anchor_residue"]:
            raise ValueError(
                f"{unit['unit_id']}: the carried sequence does not hold the recorded anchor "
                "residue at the anchor; the cohort's positions and its sequences disagree"
            )
        partner = index.get(unit["partner_unit_id"])
        if partner is None:
            raise ValueError(f"{unit['unit_id']}: its transplant partner is not in the cohort")
        if partner["partner_unit_id"] != unit["unit_id"]:
            raise ValueError(f"{unit['unit_id']}: the transplant matching is not reciprocal")
        if partner["anchor_residue"] == unit["anchor_residue"]:
            raise ValueError(
                f"{unit['unit_id']}: its transplant partner carries the same anchor residue, "
                "so the two conditions would force the same residue and the intervention "
                "would be empty"
            )
        if partner["accession"] == unit["accession"]:
            raise ValueError(f"{unit['unit_id']}: transplanted within one backbone")
        if len(unit["reference_partners"]) != D.N_REFERENCE_PARTNERS:
            raise ValueError(f"{unit['unit_id']}: wrong number of reference partners")
    return dict(payload)


def load_cohort(path: Path) -> dict[str, Any]:
    return require_cohort(json.loads(Path(path).read_text()))


def iter_cells(payload: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Every (unit, condition) cell the generation stage has to produce.

    One place that enumerates the work, so the runner, the resume test and the
    sizing cannot disagree about what a complete arm is.
    """

    for unit in payload["unit_rows"]:
        for condition in D.CONDITIONS:
            residue = (
                unit["native_residue"] if condition == D.CONDITION_NATIVE
                else unit["transplant_residue"]
            )
            yield {
                "unit_id": unit["unit_id"],
                "condition": condition,
                "forced_residue": residue,
                "accession": unit["accession"],
                "anchor": int(unit["anchor"]),
                "partner": int(unit["partner"]),
                "span_end": int(unit["span_end"]),
                "reference_positions": [int(row["position"]) for row in unit["reference_partners"]],
                "length_band": unit["length_band"],
            }
