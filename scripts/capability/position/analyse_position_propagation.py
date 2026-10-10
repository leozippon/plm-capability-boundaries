#!/usr/bin/env python3
"""E01: how far a single substitution's likelihood response reaches, and whether structure explains it.

Reads the position-resolved archives ``extract_position_likelihood.py`` writes and
reports, per arm, three things.

**The propagation profile.** The signed and absolute mean response at every
sequence separation from the mutated site, with a log-linear decay fit and the
separation at which the absolute response halves. Reported separately downstream
and upstream, because for a causal arm the upstream side is exactly zero by
construction and the number that matters there is the assertion, not an estimate.

**The site term, apart.** At the mutated residue the two states hold the
likelihood of different residues, so that position is not a response. It is
summarised on its own and never enters a profile.

**The structural contrast.** Among residues that carry an admitted experimental
coordinate, the response of residues in C-beta contact with the mutated site
minus the response of non-contacting residues at matched sequence separation,
nested so that the bootstrap unit is the family. The within-stratum separation
imbalance is reported beside it, because a contact contrast that is really a
sequence-distance contrast is the failure this design exists to exclude.

Every archive is validated by the project's own reader
(``responses.RetainedResponses``) before a single number is taken from it: the
per-state closure, the native mutation closure and the biological exclusions are
that reader's, not this stage's. Where this stage's own receiver census overlaps
that reader's, the two are required to agree exactly.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.extensions.responses import RetainedResponses  # noqa: E402
from src.capability.position.contact_response import (  # noqa: E402
    CONTACT_ANGSTROM,
    CONTACT_DEFINITION,
    MIN_SEQUENCE_SEPARATION,
    SCHEMA,
    ProfileAccumulator,
    admitted_geometry,
    agrees_with_frozen,
    contact_pairs,
    panel_contact_simultaneous,
    receiver_census,
    rebuild_states,
    require_geometry,
    simultaneous_admits,
    stratified_contact_contrast,
    structural_rows,
)
from src.capability.position.position_likelihood import (  # noqa: E402
    CAUSAL,
    declared_refusals,
    read_archive,
)

#: The two outcomes a contact contrast is formed on. Each is its own simultaneous
#: family: the absolute response and the signed response are different
#: quantities, and one critical value spanning both would be a multiplicity
#: correction across units rather than across arms.
OUTCOMES = ("absolute_response", "response")

COMPLETION = "position_propagation.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)



def extraction_status(directory: Path) -> tuple[dict | None, str | None]:
    """The completion record of one extraction cell, or why there is not one.

    An arm whose extraction cell failed leaves either no directory or no
    completion record, and the campaign is designed so that this happens: a cell
    that refuses its own invariant must not take the rest of the panel with it.
    The arm is therefore recorded as absent with the reason it is absent and
    excluded from every estimate, which is not the same as being dropped -- a
    reader of the panel sees the arm, sees that it is missing, and sees why.
    """

    if not Path(directory).is_dir():
        return None, "no extraction directory; the cell did not run or was not pulled"
    record = Path(directory) / "position_likelihood.json"
    if not record.is_file():
        return None, (
            "no completion record; the cell ran and exited without admitting its own "
            "output, which for this stage means it refused an invariant"
        )
    try:
        payload = read_json(record)
    except (OSError, ValueError) as error:
        return None, f"completion record is unreadable: {error}"
    if payload.get("status") != "complete":
        return None, f"completion record status is {payload.get('status')!r}, not complete"
    if not payload.get("assays"):
        return None, "completion record carries no assay"
    return payload, None

def analyse_arm(completion, directory: Path, cohort_rows, geometry_source, *, min_support,
                max_separation, max_residues=None):
    """One arm's profile, site summary and structural rows."""

    identity = completion["identity"]
    tolerance = float(
        completion.get("prefix_invariant", {}).get("admitted_tolerance_nats", 0.0)
    )
    arm, paradigm = identity["arm"], identity["paradigm"]
    downstream = ProfileAccumulator(direction="downstream", min_support=min_support)
    upstream = ProfileAccumulator(direction="upstream", min_support=min_support)
    site_values: list[float] = []
    structural: list[dict] = []
    per_assay, agreement = [], {"assays": 0, "receivers": 0, "max_abs_difference_nats": 0.0}
    worst_upstream = 0.0
    skipped: list[str] = []
    missing: list[dict] = []
    for receipt in completion["assays"]:
        assay = receipt["assay"]
        row = cohort_rows[assay]
        if max_residues is not None and len(row["wildtype"]) > max_residues:
            skipped.append(assay)
            continue
        archive = directory / "archives" / receipt["file"]
        if not archive.is_file():
            missing.append({"assay": assay, "reason": "archive named by the receipt is absent"})
            continue
        payload = read_archive(archive)
        states = rebuild_states(payload, [row["wildtype"], *row["sequences"]])
        identity_block = {
            "wildtype": row["wildtype"], "mutants": row["mutants"], "sequences": row["sequences"],
        }
        with np.load(archive, allow_pickle=False) as data:
            retained = RetainedResponses(data, states, identity_block)
            frozen = {index: retained.response(index) for index in retained.selected_indices}
        geometry, pairs = geometry_source(assay, row)
        census_count = 0
        structural_before = len(structural)
        for index, frozen_response in frozen.items():
            site = int(frozen_response["i"])
            census = receiver_census(payload, index, site=site, paradigm=paradigm,
                                     upstream_tolerance=tolerance)
            check = agrees_with_frozen(census, frozen_response)
            agreement["receivers"] += check["downstream_receivers"]
            agreement["max_abs_difference_nats"] = max(
                agreement["max_abs_difference_nats"], check["max_abs_difference_nats"]
            )
            worst_upstream = max(worst_upstream, float(census["upstream_max_abs_nats"]))
            rows = census["receivers"]
            if max_separation is not None:
                rows = [r for r in rows if abs(int(r["separation"])) <= max_separation]
            downstream.add(rows)
            upstream.add(rows)
            downstream.count_mutation()
            upstream.count_mutation()
            site_values.append(float(census["site_response"]))
            census_count += 1
            if geometry is not None:
                structural.extend(
                    structural_rows(
                        census, geometry, pairs, assay=assay, family=row["cluster"],
                        mutation=row["mutants"][index],
                    )
                )
        agreement["assays"] += 1
        assay_structural = len(structural) - structural_before
        per_assay.append({
            "assay": assay,
            "family": row["cluster"],
            "residues": len(row["wildtype"]),
            "selected_single_substitutions": census_count,
            "excluded": retained.exclusions[:8],
            "excluded_total": len(retained.exclusions),
            "structural_receivers": assay_structural,
        })
    if paradigm == CAUSAL and worst_upstream > tolerance:
        raise SystemExit(
            f"{arm}: upstream terms differ by {worst_upstream} nats against the extraction's "
            f"admitted tolerance of {tolerance}"
        )
    sites = np.asarray(site_values, dtype=np.float64) if site_values else np.zeros(0)
    # A half-distance is a property of the scope it was fitted on, not of the arm
    # alone: the fit covers only the separations these assays offer, so widening
    # the length cap moves it. Carried inside each profile rather than left at
    # file level, so a half-distance cannot be quoted without its scope and two
    # scopes cannot be read as one column.
    scope = {
        "max_residues": None if max_residues is None else int(max_residues),
        "assays": len(per_assay),
        "longest_wildtype_residues": max((item["residues"] for item in per_assay), default=0),
        "note": (
            "half_distance_residues and its interval are fitted on these assays only; a "
            "different wild-type length cap offers different separations and gives a "
            "different half-distance for the same arm"
        ),
    }
    return {
        "arm": arm,
        "paradigm": paradigm,
        "identity": identity,
        "extraction": {"directory": str(directory),
                       "completion_sha256": sha256_file(directory / "position_likelihood.json")},
        "assays": per_assay,
        "assays_outside_length_cap": skipped,
        "assays_missing_archive": missing,
        "prefix_invariant": completion.get("prefix_invariant"),
        "upstream_admitted_tolerance_nats": tolerance,
        "frozen_reader_agreement": agreement,
        "upstream_max_abs_nats": worst_upstream,
        "upstream_is_exactly_zero": bool(worst_upstream == 0.0),
        "site_term": {
            "n": int(sites.size),
            "mean_nats": float(sites.mean()) if sites.size else None,
            "mean_absolute_nats": float(np.abs(sites).mean()) if sites.size else None,
            "note": (
                "the mutated residue's own term, where the two states hold the "
                "likelihood of different residues; not a response and never profiled"
            ),
        },
        "profile": {
            "downstream": {**downstream.profile(), "scope": scope},
            "upstream": {**upstream.profile(), "scope": scope},
        },
        "structural_receivers": len(structural),
    }, structural


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", type=Path, action="append", required=True,
                        help="an extract_position_likelihood.py output directory; repeatable")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True,
                        help="the structural coverage receipt paired with --structures")
    parser.add_argument("--contact-angstrom", type=float, default=CONTACT_ANGSTROM)
    parser.add_argument("--min-separation", type=int, default=MIN_SEQUENCE_SEPARATION)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--max-residues", type=int, default=0,
                        help="restrict every arm to assays at or below this wild-type length, so "
                             "that arms extracted on different scopes are compared on one assay "
                             "set; 0 uses whatever each arm carries")
    parser.add_argument("--max-separation", type=int, default=0,
                        help="0 keeps every separation the cohort offers")
    parser.add_argument("--device", default="cpu", help="accepted for queue injection; unused")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cohort = read_json(args.cohort)
    cohort_rows = {row["assay"]: row for row in cohort["assays"]}
    sites = read_json(args.structures)
    site_rows = sites["sites"] if isinstance(sites, dict) else sites
    coverage = read_json(args.coverage)
    admitted = {row["assay_id"] for row in site_rows if row["status"] == "admitted"}
    cache: dict[str, tuple] = {}

    def geometry_source(assay: str, row):
        if assay not in admitted:
            return None, None
        if assay not in cache:
            geometry = admitted_geometry(
                site_rows, assay=assay, family=row["cluster"], wildtype=row["wildtype"],
                coverage=coverage,
            )
            require_geometry(geometry, assay=assay)
            cache[assay] = (
                geometry,
                contact_pairs(
                    geometry, cutoff=args.contact_angstrom, min_separation=args.min_separation
                ),
            )
        return cache[assay]

    arms, contrasts, absent = [], [], []
    panel_columns: dict[str, list[dict]] = {outcome: [] for outcome in OUTCOMES}
    for directory in args.extraction:
        completion, reason = extraction_status(directory)
        if completion is None:
            absent.append({"extraction": str(directory), "arm": Path(directory).name,
                           "reason": reason})
            continue
        block, structural = analyse_arm(
            completion, directory, cohort_rows, geometry_source,
            min_support=args.min_support,
            max_separation=args.max_separation or None,
            max_residues=args.max_residues or None,
        )
        for direction in ("downstream", "upstream"):
            if not any(row["direction"] == direction for row in structural):
                continue
            for outcome in OUTCOMES:
                contrast = stratified_contact_contrast(
                    structural, outcome=outcome, direction=direction
                )
                contrast["arm"] = block["arm"]
                contrasts.append(contrast)
                if contrast["family_values"] and simultaneous_admits(
                    block["paradigm"], direction
                ):
                    panel_columns[outcome].append({
                        "arm": block["arm"],
                        "direction": direction,
                        "family_values": contrast["family_values"],
                    })
        arms.append(block)
        del structural
    if not arms:
        raise SystemExit(
            "no extraction directory carried an admitted completion record; there is "
            f"nothing to analyse. Absent: {absent}"
        )
    simultaneous = {
        outcome: panel_contact_simultaneous(panel_columns[outcome], outcome=outcome)
        for outcome in OUTCOMES
    }

    write_json(args.out / COMPLETION, {
        "schema": SCHEMA,
        "status": "complete",
        "created_utc": _now(),
        "experiment": "E01",
        "question": (
            "how far along the sequence a single substitution moves the position-wise "
            "log likelihood, and whether the residues that move are the ones in "
            "three-dimensional contact with the substituted site"
        ),
        "contact_definition": CONTACT_DEFINITION,
        "settings": {
            "contact_angstrom": float(args.contact_angstrom),
            "min_separation": int(args.min_separation),
            "stratum_labels": (
                "sequence-separation strata reuse the frozen edges of this project's own "
                "structural pair census, so the first label reads 3-8 while the separation "
                "floor truncates that stratum to [min_separation, 8]"
            ),
            "min_support": int(args.min_support),
            "max_separation": int(args.max_separation) or None,
            "max_residues": int(args.max_residues) or None,
        },
        "causal_asymmetry": (
            "for a left-to-right arm the response at every position before the "
            "substitution is identically zero, because the prefix that predicts it is "
            "unchanged; propagation is therefore measurable downstream only, and the "
            "upstream block records the assertion rather than an estimate. A masked arm "
            "responds on both sides and is the comparison that makes the one-sided "
            "causal profile readable"
        ),
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in (("cohort", args.cohort), ("structures", args.structures),
                               ("coverage", args.coverage))
        },
        "arms": arms,
        "absent_arms": absent,
        "declared_refusals": declared_refusals(),
        "panel": {
            "requested_extractions": len(args.extraction),
            "analysed_arms": len(arms),
            "absent_arms": len(absent),
            "declared_refusals": len(declared_refusals()),
            "policy": (
                "an arm whose extraction cell left no admitted completion record is "
                "recorded here with its reason and excluded from every estimate; it is "
                "neither silently dropped nor fatal to the rest of the panel. An arm the "
                "project refuses position-resolved work for never reaches this stage at "
                "all, so it is named from that declaration rather than inferred from a "
                "missing file: a refusal and a failure are different outcomes"
            ),
        },
        "contact_contrasts": contrasts,
        "panel_simultaneous": simultaneous,
        "panel_inference_policy": (
            "per-arm intervals under contact_contrasts are marginal and are not "
            "panel-wide statements. Any claim about the panel reads panel_simultaneous, "
            "whose bands hold jointly at 95 percent over every admitted (arm, direction) "
            "column of one outcome"
        ),
    })


if __name__ == "__main__":
    main()
