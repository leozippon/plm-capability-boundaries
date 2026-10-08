#!/usr/bin/env python3
"""E02: does a conditional formed before position i already constrain its later contact partner?

The literal reading of E02 -- intervene on a later residue and watch an earlier
position's conditional move -- is not identifiable for a left-to-right model, and
this stage does not pretend to run it. The conditional at position ``i`` is a
function of the prefix ``w_{<i}``; a residue after ``i`` is not in that prefix, so
every such intervention moves it by exactly zero on every arm and every pair. The
identifiable question in the same place, and the one measured here, is predictive:
treating the model's own distribution at ``i`` as a feature and the realised
residue at a contacting position ``j > i`` as a label, how many nats does a
family-held-out predictor save over the marginal of that label -- and does it save
more for contacting partners than for non-contacting partners at matched sequence
separation from the same anchor?

Two controls decide whether a positive number means anything. The
separation-matched non-contacting partner removes the predictability that comes
from composition and local propensity rather than from the fold; it is subtracted,
so the endpoint is a difference and never one side alone. The anchor-identity
feature -- a one-hot of the realised residue at ``i``, no model involved --
measures the covariation of contacting residue pairs in the sequences themselves,
and a model conditional that does not beat it has reproduced that covariation
rather than anticipated anything.

A masked arm is read as a positive reference and not as a competitor: its
conditional at ``i`` was formed with the rest of the sequence visible, so it has
already seen the partner, and its value calibrates the scale the causal arms are
being measured on.
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
from src.capability.position.anticipation import (  # noqa: E402
    IDENTIFIABILITY,
    MIN_FAMILIES,
    SCHEMA,
    anchor_partner_design,
    anticipation_contrast,
    composition,
    resolved_alphabet,
)
from src.capability.position.contact_response import (  # noqa: E402
    CONTACT_ANGSTROM,
    CONTACT_DEFINITION,
    MIN_SEQUENCE_SEPARATION,
    admitted_geometry,
    contact_pairs,
    require_geometry,
)
from src.capability.position.position_likelihood import read_archive  # noqa: E402

COMPLETION = "contact_anticipation.json"


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

def arm_design(completion, directory: Path, cohort_rows, geometry_source, *, forward_only):
    """One arm's matched anchor/partner rows and its conditional table."""

    identity = completion["identity"]
    blocks, per_assay = [], []
    for receipt in completion["assays"]:
        assay = receipt["assay"]
        row = cohort_rows[assay]
        geometry, pairs = geometry_source(assay, row)
        if geometry is None:
            continue
        archive = directory / "archives" / receipt["file"]
        if not archive.is_file():
            per_assay.append({"assay": assay, "status": "archive named by the receipt is absent"})
            continue
        payload = read_archive(archive)
        if "wt_conditional_logprobs" not in payload:
            per_assay.append({"assay": assay, "status": "no residue conditional retained"})
            continue
        residues = [str(value) for value in payload["wt_conditional_residues"]]
        positions = [int(value) for value in payload["wt_conditional_positions"]]
        logprobs = np.asarray(payload["wt_conditional_logprobs"], dtype=np.float64)
        if logprobs.shape != (len(positions), len(residues)):
            raise SystemExit(f"{assay}: the retained conditional block is not rectangular")
        rows = anchor_partner_design(
            geometry, pairs, anchors=positions, wildtype=row["wildtype"], residues=residues,
            forward_only=forward_only,
        )
        for item in rows:
            item["assay"] = assay
            item["family"] = row["cluster"]
        blocks.append({
            "assay": assay,
            "residues": residues,
            "positions": positions,
            "logprobs": logprobs,
            "rows": rows,
        })
        per_assay.append({
            "assay": assay,
            "status": "designed",
            "family": row["cluster"],
            "anchors": len({item["i"] for item in rows}),
            "contact_pairs": sum(1 for item in rows if item["contact"]),
            "control_pairs": sum(1 for item in rows if not item["contact"]),
            "resolved_residues": len(residues),
        })
    if not blocks:
        return identity, [], (), {}, {}, per_assay
    shared = resolved_alphabet(
        sorted(set.intersection(*[set(block["residues"]) for block in blocks]))
    )
    conditionals: dict[tuple[str, int], np.ndarray] = {}
    compositions: dict[str, np.ndarray] = {}
    rows = []
    for block in blocks:
        columns = [block["residues"].index(residue) for residue in shared]
        for order, position in enumerate(block["positions"]):
            conditionals[(block["assay"], int(position))] = block["logprobs"][order, columns]
        compositions[block["assay"]] = composition(
            cohort_rows[block["assay"]]["wildtype"], shared
        )
        rows.extend(
            item for item in block["rows"]
            if item["partner_residue"] in shared and item["anchor_residue"] in shared
        )
    return identity, rows, shared, conditionals, compositions, per_assay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", type=Path, action="append", required=True,
                        help="an extract_position_likelihood.py output directory; repeatable")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--contact-angstrom", type=float, default=CONTACT_ANGSTROM)
    parser.add_argument("--min-separation", type=int, default=MIN_SEQUENCE_SEPARATION)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--both-directions", action="store_true",
                        help="also admit partners before the anchor; off by default because the "
                             "question is about a residue the model has not yet produced")
    parser.add_argument("--device", default="cpu", help="accepted for queue injection; unused")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.draws < 1:
        parser.error("--draws is positive")
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

    arms, absent = [], []
    for directory in args.extraction:
        completion, reason = extraction_status(directory)
        if completion is None:
            absent.append({"extraction": str(directory), "arm": Path(directory).name,
                           "reason": reason})
            continue
        identity, rows, shared, conditionals, compositions, per_assay = arm_design(
            completion, directory, cohort_rows, geometry_source,
            forward_only=not args.both_directions,
        )
        block = {
            "arm": identity["arm"],
            "paradigm": identity["paradigm"],
            "reads_as": (
                "an upper reference: this conditional was formed with the partner visible"
                if identity["paradigm"] != "causal_next_token"
                else "anticipation: this conditional was formed from the prefix alone"
            ),
            "extraction": {
                "directory": str(directory),
                "completion_sha256": sha256_file(directory / "position_likelihood.json"),
            },
            "assays": per_assay,
        }
        if rows:
            block["result"] = anticipation_contrast(
                rows, residues=shared, conditionals=conditionals, compositions=compositions,
                draws=args.draws, seed=args.seed,
            )
        else:
            block["result"] = None
            block["undefined_reason"] = (
                "no anchor carried both a contacting and a separation-matched "
                "non-contacting partner with a retained residue conditional"
            )
        arms.append(block)
        del rows, conditionals, compositions
    if not arms:
        raise SystemExit(
            "no extraction directory carried an admitted completion record; there is "
            f"nothing to analyse. Absent: {absent}"
        )

    write_json(args.out / COMPLETION, {
        "schema": SCHEMA,
        "status": "complete",
        "created_utc": _now(),
        "experiment": "E02",
        "question": (
            "when predicting an earlier amino acid, does the model already contain "
            "information about that residue's structural contacts with later residues"
        ),
        "identifiability": IDENTIFIABILITY,
        "contact_definition": CONTACT_DEFINITION,
        "settings": {
            "contact_angstrom": float(args.contact_angstrom),
            "min_separation": int(args.min_separation),
            "stratum_labels": (
                "sequence-separation strata reuse the frozen edges of this project's own "
                "structural pair census, so the first label reads 3-8 while the separation "
                "floor truncates that stratum to [min_separation, 8]"
            ),
            "draws": int(args.draws),
            "seed": int(args.seed),
            "forward_only": not args.both_directions,
            "min_families": MIN_FAMILIES,
        },
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in (("cohort", args.cohort), ("structures", args.structures),
                               ("coverage", args.coverage))
        },
        "arms": arms,
        "absent_arms": absent,
        "panel": {
            "requested_extractions": len(args.extraction),
            "analysed_arms": len(arms),
            "absent_arms": len(absent),
            "policy": (
                "an arm whose extraction cell left no admitted completion record is "
                "recorded here with its reason and excluded from every estimate; it is "
                "neither silently dropped nor fatal to the rest of the panel"
            ),
        },
    })


if __name__ == "__main__":
    main()
