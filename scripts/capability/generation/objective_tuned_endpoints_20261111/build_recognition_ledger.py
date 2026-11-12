#!/usr/bin/env python3
"""The sequence ledger the Pfam/HMMER oracle is run on: one folded eighteen-shard structure.

``recognise_generated_families.py`` takes JSONL records carrying ``id`` and
``sequence``. The per-structure tables carry the digest and no sequence, and the
frozen fold cohort carries the sequence and no arm, so the ledger is their join
on the sequence digest -- recomputed from the bytes, not trusted from a column.
The id is the digest, which is unique by construction and is what the analysis
module joins recognition on.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/Data/lzp/plm-capability-boundaries")
sys.path.insert(0, str(REPO_ROOT))

from src.capability.generation.structure_evidence import sequence_digest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neighbour", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot", type=int, default=200,
                        help="also write a pilot ledger of this many rows, to size the scan")
    args = parser.parse_args()

    cohort = {}
    for line in args.cohort.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            cohort[str(record["id"])] = record
    with args.neighbour.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    out = []
    for row in sorted(rows, key=lambda row: (row["arm"], row["sequence_sha256"])):
        digest = row["sequence_sha256"]
        record = cohort.get(digest)
        if record is None:
            raise SystemExit(f"{digest} is not in the fold cohort")
        sequence = str(record["sequence"])
        if sequence_digest(sequence) != digest:
            raise SystemExit(f"{digest}: the cohort sequence digests to {sequence_digest(sequence)}")
        out.append({"id": digest, "sequence_sha256": digest, "sequence": sequence,
                    "length": int(record["length"]), "arm": row["arm"], "role": "generation"})
    if len({row["id"] for row in out}) != len(out):
        raise SystemExit("the ledger carries a duplicate id")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cohort_18of68.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in out), encoding="utf-8")
    (args.out / f"cohort_pilot{args.pilot}.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in out[:args.pilot]),
        encoding="utf-8")
    print(json.dumps({"rows": len(out), "arms": sorted({row["arm"] for row in out}),
                      "min_length": min(row["length"] for row in out),
                      "max_length": max(row["length"] for row in out)}, sort_keys=True))


if __name__ == "__main__":
    main()
