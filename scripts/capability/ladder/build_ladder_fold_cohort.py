#!/usr/bin/env python3
"""Union every arm's ladder variants into one cohort for the structure instrument.

The structure instrument is this project's existing resumable ESMFold2 runner,
``scripts/capability/generation/run_structure_evidence.py``, which reads a JSONL
cohort of ``id`` and ``sequence`` and keys its fold objects by the sequence
digest. Nothing about it is changed or reimplemented here: this stage only puts
every arm's rows into the one file it reads, so that two arms which happened to
produce the same string share one fold instead of paying for two.

The parents are deliberately **not** in this cohort. Every backbone was selected
from folds this project had already measured, so its parent fold exists on GPFS
and is referenced by literal path. Re-folding it would introduce a second
measurement of the same quantity for no gain.

Draws that did not express their rung carry an empty sequence and are left out of
the fold cohort, with their count recorded here: the folding instrument has
nothing to fold, and the analysis stage reads the full variant files, so the
denominator is never silently reduced.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.ladder import design  # noqa: E402

COMPLETION = "ladder_fold_cohort.json"
RECORDS = "ladder_fold_cohort.jsonl"
SCHEMA_VERSION = "ladder_fold_cohort_v1"


def run(args: argparse.Namespace) -> dict[str, Any]:
    design.require_fresh_out(args.out, COMPLETION)
    rows: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    skipped: Counter[str] = Counter()
    per_arm: Counter[str] = Counter()
    for path in args.variants:
        for row in design.read_jsonl(path):
            if row["status"] != "filled" or not row["sequence"]:
                skipped[f"{row['arm']}|{row['status']}"] += 1
                continue
            if row["id"] in identifiers:
                raise SystemExit(f"duplicate variant id {row['id']} across the supplied files")
            identifiers.add(row["id"])
            per_arm[str(row["arm"])] += 1
            rows.append(
                {
                    "id": row["id"],
                    "sequence": row["sequence"],
                    "sequence_sha256": row["sequence_sha256"],
                    "arm": row["arm"],
                    "condition": row["condition"],
                    "rung": row["rung"],
                    "backbone_id": row["backbone_id"],
                    "length": len(row["sequence"]),
                }
            )
    if not rows:
        raise SystemExit("no filled variant reached the fold cohort")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    digest = design.write_jsonl(out / RECORDS, rows)
    lengths = sorted(row["length"] for row in rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records": RECORDS,
        "records_sha256": digest,
        "n_rows": len(rows),
        "n_unique_sequences": len({row["sequence_sha256"] for row in rows}),
        "per_arm": dict(sorted(per_arm.items())),
        "skipped_unfilled": dict(sorted(skipped.items())),
        "length_summary": {
            "min": lengths[0],
            "max": lengths[-1],
            "median": lengths[len(lengths) // 2],
        },
        "sources": {str(path): sha256_file(path) for path in args.variants},
        "parents_excluded": (
            "the parent folds are reused from the staged tree the backbones were "
            "selected from and are referenced by literal path, not refolded"
        ),
    }
    write_json(out / COMPLETION, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="accepted and unused; this stage is CPU-only")
    parser.add_argument("--variants", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"n_rows": payload["n_rows"], "per_arm": payload["per_arm"]}))


if __name__ == "__main__":
    main()
