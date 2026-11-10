#!/usr/bin/env python3
"""Freeze the folding cohort for the objective-tuned campaign: the whole pool.

Every non-empty continuation of every admitted cell enters the cohort, base arms
included and unsubsampled. That is a requirement of the design rather than a
default. The explicit-statistics control ranks the base pool by a statistical
rule and keeps its top quarter, and which quarter that is cannot be known before
the rule runs, so pLDDT has to exist for whatever it selects. The primary
contrast is a within-stratum mean over the whole base pool, which a subsample
would not supply either. Folding the lane's own 128-parent structure subset
instead would answer neither.

The cohort is keyed by sequence digest, so identical continuations -- within a
cell or across arms -- share one fold. That is deduplication of the *instrument*,
not of the sampling unit: the attempt ledgers remain the denominator, and a
downstream join is sequence to pLDDT, so a sequence two arms both produced
carries the same measured value in both pools, which is what it should.

Empty continuations are not folded, because there is nothing to fold. They stay
in each cell's ledger, and their count per cell is carried here so that a
non-empty rate is readable from this record without reopening the ledgers. A
continuation shorter than the instrument's declared support is kept in the cohort
and refused by the folder itself, which records the reason per sequence; that
refusal belongs to the instrument and is not this script's to anticipate.

``--verified`` is the record ``verify_objective_tuned_cells.py`` writes, and it is
required: a cohort assembled from unverified cells could carry a tuned pool drawn
from base weights, and no property of this script would notice.

Each row carries ``id``, ``sequence``, ``role`` and ``length``. The folder itself
reads only the first two; ``role`` and ``length`` are what the downstream novelty
search requires of every cohort row, and emitting them here keeps one cohort file
serving both instruments instead of having the analysis rebuild it. ``role`` is
``generation`` for every row, because this cohort is generated sequences and
nothing else -- no wild-type reference, no shuffle control. The join key stays the
recomputed sequence digest, which is also ``id``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation import structure_evidence as se  # noqa: E402
from src.capability.generation import unconditional_generation as ug  # noqa: E402

COHORT = "fold_cohort.jsonl"
RECORD = "objective_tuned_fold_cohort.json"
SCHEMA = "objective_tuned_fold_cohort_v1"

#: Every row of this cohort is a generated sequence. The field exists because the
#: novelty search requires it on each row and distinguishes generations from
#: references and controls; this cohort contains neither.
COHORT_ROLE = "generation"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verified", type=Path, required=True, help="objective_tuned_cells.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="accepted and unused; CPU-only")
    args = parser.parse_args(argv)

    verified = json.loads(args.verified.read_text(encoding="utf-8"))
    if verified.get("schema_version") != "objective_tuned_cells_v1":
        raise SystemExit(f"--verified is not a cell admission record: {args.verified}")
    if not verified.get("complete"):
        raise SystemExit(
            f"{args.verified}: {verified.get('cells_admitted')} of "
            f"{verified.get('cells_declared')} cells are admitted; refusing to freeze a "
            "cohort over part of the campaign"
        )

    sequences: dict[str, str] = {}
    per_cell = []
    for cell in verified["cells"]:
        ledger = args.run_dir / cell["label"] / ug.LEDGER_NAME
        digest = sha256_file(ledger)
        if digest != cell["attempts_sha256"]:
            raise SystemExit(
                f"{cell['label']}: {ug.LEDGER_NAME} changed since it was admitted"
            )
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != cell["attempts"]:
            raise SystemExit(f"{cell['label']}: {len(rows)} rows against {cell['attempts']}")
        unique_here = set()
        for row in rows:
            sequence = row["sequence"]
            if not sequence:
                continue
            key = se.sequence_digest(sequence)
            previous = sequences.setdefault(key, sequence)
            if previous != sequence:
                raise SystemExit(f"sequence digest collision on {key}")
            unique_here.add(key)
        per_cell.append(
            {
                "label": cell["label"],
                "arm": cell["arm"],
                "role": cell["role"],
                "stream": cell["stream"],
                "replicate": cell["replicate"],
                "attempts": cell["attempts"],
                "non_empty": cell["non_empty"],
                "unique_sequences": len(unique_here),
                "attempts_sha256": digest,
            }
        )

    if not sequences:
        raise SystemExit("no non-empty continuation in the whole campaign")

    args.out.mkdir(parents=True, exist_ok=True)
    cohort = args.out / COHORT
    cohort.write_text(
        "".join(
            json.dumps(
                {
                    "id": key,
                    "sequence": sequences[key],
                    "role": COHORT_ROLE,
                    "length": len(sequences[key]),
                }
            )
            + "\n"
            for key in sorted(sequences)
        ),
        encoding="utf-8",
    )
    by_arm: dict[str, dict[str, int]] = {}
    for entry in per_cell:
        arm = by_arm.setdefault(entry["arm"], {"cells": 0, "attempts": 0, "non_empty": 0})
        arm["cells"] += 1
        arm["attempts"] += entry["attempts"]
        arm["non_empty"] += entry["non_empty"]
    record = {
        "schema_version": SCHEMA,
        "campaign": verified["campaign"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verified_sha256": sha256_file(args.verified),
        "cohort_sha256": sha256_file(cohort),
        "cells": len(per_cell),
        "attempts_total": sum(entry["attempts"] for entry in per_cell),
        "non_empty_total": sum(entry["non_empty"] for entry in per_cell),
        "unique_sequences": len(sequences),
        "subsampled": False,
        "arms": by_arm,
        "per_cell": per_cell,
        "code_sha256": sha256_file(Path(__file__)),
    }
    write_json(args.out / RECORD, record)
    print(json.dumps({key: record[key] for key in ("cells", "attempts_total", "non_empty_total", "unique_sequences")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
