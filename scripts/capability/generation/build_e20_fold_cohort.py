#!/usr/bin/env python3
"""Freeze the folding cohort for the E20 fold-recapitulation measurement.

Either the twelve wild-type sequences (``--wild-type``: the reference structures)
or the unique sequences of a generator record file (``--records``, validated
against the e20_generated_sequence_v1 contract). The output ``fold_cohort.jsonl``
is folded by the unchanged ``run_structure_evidence.py``; both runs must use the
same instrument settings, which ``measure_fold_recapitulation.py`` enforces.
Flat parser: the campaign queue prepends ``--device`` and ``--out``.
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
from src.capability.generation import fold_recapitulation as fr  # noqa: E402
from src.capability.generation import structure_evidence as se  # noqa: E402

COMPLETION = "e20_fold_cohort.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="accepted and unused; CPU-only")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wild-type", action="store_true")
    source.add_argument("--records", type=Path)
    args = parser.parse_args()
    wild = fr.load_wild_types(REPO_ROOT)
    if args.wild_type:
        sequences = {f"{d}_wild_type": s for d, s in sorted(wild.items())}
        inputs = {"wild_type_fasta": sha256_file(REPO_ROOT / fr.WT_FASTA)}
    else:
        records = fr.load_records(args.records, wild)
        unique = sorted({r["sequence"] for r in records})
        sequences = {se.sequence_digest(s)[:16]: s for s in unique}
        inputs = {"records": sha256_file(args.records)}
    args.out.mkdir(parents=True, exist_ok=True)
    cohort = args.out / "fold_cohort.jsonl"
    cohort.write_text("".join(json.dumps({"id": i, "sequence": s}) + "\n" for i, s in sequences.items()))
    write_json(args.out / COMPLETION, {
        "schema_version": "e20_fold_cohort_v1", "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_sequences": len(sequences), "cohort_sha256": sha256_file(cohort), "inputs": inputs,
        "code_sha256": sha256_file(Path(__file__)), "contract": fr.SCHEMA_VERSION,
    })
    print(json.dumps({"n_sequences": len(sequences)}))


if __name__ == "__main__":
    main()
