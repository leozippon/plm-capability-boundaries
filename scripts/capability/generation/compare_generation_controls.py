#!/usr/bin/env python3
"""Does the published natural-fragment comparator decide the generation conclusion?

Section 6 reports generated complete-domain recognition minus that of a
contiguous substring of a real protein cut to the attempt's exact length. A
fragment is an odd reference for a whole-protein generator, and the question is
whether the conclusion depends on it.

Two phases.

``table``   reads the generation-and-control gate, which measured a **whole
            natural record in the attempt's length band** beside the fragment on
            the same attempts, the same oracle and the same resampling unit, and
            reports per cell whether the published verdict survives the
            substitution. It also censuses duplication, termination behaviour and
            the novelty covariate on the annotated attempt ledgers, and, when a
            family-matched recognition run is supplied, adds the ceiling that
            real full-length proteins of the requested family reach.

``draw``    builds that family-matched natural comparator: for each conditional
            class, full-length Swiss-Prot proteins carrying the class's frozen
            Pfam referent, inside the length band the class's own generated
            attempts occupied. The family comes from the *request*, never from
            what the generations happened to hit, so the comparator is not
            selected on the outcome. For an unconditional cell no family is
            requested and no family-matched arm exists; that is stated rather
            than approximated.

Both phases are CPU-only. ``draw`` writes sequences for the oracle stage to
score; it never scores them itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import write_json  # noqa: E402
from src.capability.generation import family_oracle as fo  # noqa: E402
from src.capability.generation import generation_controls as gcx  # noqa: E402

DEFAULT_GATE = REPO_ROOT / "results/R6/generative_control_20260924/gate_endpoints.json"
DEFAULT_PROFILES = REPO_ROOT / "results/R6/generation_replication_20260927/profiles"
DEFAULT_SWISSPROT = REPO_ROOT / "data/swissprot/uniprot_sprot.fasta.gz"
DEFAULT_PFAM_RESIDUE = REPO_ROOT / "data/interpro/pfam_residue.tsv"

TABLE_COMPLETION = "expanded_controls.json"
DRAW_COMPLETION = "natural_family_matched_draw.json"
DRAW_RECORDS = "natural_family_matched.jsonl"


def _iter_fasta(path: Path):
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    header, chunks = None, []
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:].strip(), []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def _ledgers(root: Path, names: list[str] | None) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {}
    for directory in sorted(Path(root).iterdir()):
        path = directory / "annotated_attempts.jsonl"
        if not path.is_file():
            continue
        if names and directory.name not in names:
            continue
        found[directory.name] = gcx.read_jsonl(path)
    if not found:
        raise SystemExit(f"{root} holds no annotated_attempts.jsonl")
    return found


def run_table(args: argparse.Namespace) -> dict[str, Any]:
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    tables = {
        endpoint: gcx.comparator_table(gate, endpoint=endpoint) for endpoint in gcx.ENDPOINTS
    }
    census = gcx.census_over_cells(_ledgers(args.profiles, args.cells))

    family_matched: dict[str, Any] = {
        "status": "not_run",
        "reason": (
            "no --natural-recognition was supplied, so the family-matched full-length "
            "ceiling is reported as NOT RUN. It is never reported as an absent effect"
        ),
    }
    if args.natural_recognition is not None:
        if args.natural_draw is None:
            raise SystemExit("--natural-recognition needs the --natural-draw it was scored from")
        draw = json.loads(args.natural_draw.read_text(encoding="utf-8"))
        recognition = {
            str(row["id"]): row
            for row in gcx.read_jsonl(args.natural_recognition)
        }
        family_matched = gcx.natural_recognition_rates(draw, recognition)
        family_matched["draw"] = {
            "path": str(args.natural_draw),
            "sha256": hashlib.sha256(args.natural_draw.read_bytes()).hexdigest(),
            "n_drawn": draw["n_drawn"],
            "draw_seed": draw["draw_seed"],
        }

    verdict = {
        "question": (
            "does the published length-matched natural-fragment comparator decide the "
            "section-6 conclusion?"
        ),
        "per_endpoint": {
            endpoint: {
                "fragment_verdicts": table["verdict_counts"]["fragment"],
                "natural_verdicts": table["verdict_counts"]["natural"],
                "n_cells_whose_verdict_changes": table["n_cells_whose_verdict_changes"],
                "cells_whose_verdict_changes": table["cells_whose_verdict_changes"],
                "n_positive_lost": table["sign_changes"]["n_positive_lost"],
                "n_positive_gained": table["sign_changes"]["n_positive_gained"],
                "mean_fragment_minus_natural_difference": table[
                    "fragment_minus_natural_difference"
                ]["mean"],
            }
            for endpoint, table in sorted(tables.items())
        },
        "reading_rule": (
            "the conclusion is comparator-dependent only if a cell that clears zero "
            "against the fragment fails to against the whole natural record, or the "
            "reverse. A cell that moves between negative and unresolved changes its "
            "verdict without changing which checkpoints clear the bar"
        ),
    }
    record = {
        "schema_version": gcx.SCHEMA_VERSION,
        "status": "complete",
        "phase": "table",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate": str(args.gate),
        "gate_sha256": hashlib.sha256(args.gate.read_bytes()).hexdigest(),
        "profiles_root": str(args.profiles),
        "comparator_tables": tables,
        "census": census,
        "family_matched_natural": family_matched,
        "verdict": verdict,
        "ceiling": dict(gcx.CEILING),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / TABLE_COMPLETION, record)
    return record


def run_draw(args: argparse.Namespace) -> dict[str, Any]:
    if not args.anchors:
        raise SystemExit(
            "--anchors is required: the family a class requests comes from its frozen "
            "Pfam referent, never from what its generations happened to carry"
        )
    classes: dict[str, dict[str, Any]] = {}
    for path in args.anchors:
        payload = json.loads(path.read_text(encoding="utf-8"))
        arm = str(payload.get("arm") or path.stem)
        for class_key, entry in fo.class_referents(payload).items():
            classes[class_key] = {**entry, "arm": arm}

    ledgers = _ledgers(args.profiles, args.cells)
    bands: dict[str, tuple[int, int]] = {}
    for name, rows in ledgers.items():
        by_class: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = row.get("class_key")
            if key:
                by_class.setdefault(str(key), []).append(row)
        for class_key, group in by_class.items():
            if class_key not in classes:
                continue
            low, high = gcx.length_band(group, tolerance=args.length_tolerance)
            if class_key in bands:
                previous = bands[class_key]
                bands[class_key] = (min(previous[0], low), max(previous[1], high))
            else:
                bands[class_key] = (low, high)
    if not bands:
        raise SystemExit(
            "no conditional cell in the supplied ledgers carries a class the anchors "
            "declare; a family-matched draw has no length band to match against"
        )
    classes = {key: value for key, value in classes.items() if key in bands}

    families = sorted({family for entry in classes.values() for family in entry["referent"]})
    members = gcx.pfam_members(args.pfam_residue, families)
    draw = gcx.family_matched_draw(
        classes=classes,
        length_bands=bands,
        records=_iter_fasta(args.swissprot),
        members=members,
        per_class=args.per_class,
        seed=args.draw_seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    records = draw.pop("records")
    path = args.out / DRAW_RECORDS
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
    )
    record = {
        "status": "complete",
        "phase": "draw",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **draw,
        "classes": {key: {"arm": value["arm"], "referent": list(value["referent"])}
                    for key, value in sorted(classes.items())},
        "corpus": str(args.swissprot),
        "pfam_residue_table": str(args.pfam_residue),
        "records_file": str(path),
        "records_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "unconditional_cells_have_no_family_matched_arm": gcx.CEILING[
            "family_matching_needs_a_request"
        ],
        "next": "recognise_generated_families.py --sequences " + DRAW_RECORDS,
    }
    write_json(args.out / DRAW_COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("table", "draw"))
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES,
                        help="directory of <cell>/annotated_attempts.jsonl")
    parser.add_argument("--cells", nargs="*", default=None, help="restrict to these cell directories")
    parser.add_argument("--anchors", type=Path, nargs="*", default=[])
    parser.add_argument("--swissprot", type=Path, default=DEFAULT_SWISSPROT)
    parser.add_argument("--pfam-residue", type=Path, default=DEFAULT_PFAM_RESIDUE)
    parser.add_argument("--per-class", type=int, default=gcx.DRAW_PER_CLASS)
    parser.add_argument("--draw-seed", type=int, default=gcx.DRAW_SEED)
    parser.add_argument("--length-tolerance", type=float, default=gcx.LENGTH_MATCH_TOLERANCE)
    parser.add_argument("--natural-draw", type=Path, default=None)
    parser.add_argument("--natural-recognition", type=Path, default=None,
                        help="family_recognition.jsonl for the family-matched draw")
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = run_table(args) if args.phase == "table" else run_draw(args)
    print(json.dumps({key: record[key] for key in ("status", "phase")}
                     | ({"verdict": record["verdict"]["per_endpoint"]} if args.phase == "table"
                        else {"n_drawn": record["n_drawn"]}), sort_keys=True))


if __name__ == "__main__":
    main()
