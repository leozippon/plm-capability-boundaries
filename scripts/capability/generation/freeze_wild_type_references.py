#!/usr/bin/env python3
"""Freeze the twelve ESMFold2 wild-type structures as the E20 reference set.

Reads a verified ``run_structure_evidence.py`` tree folded from the
``--wild-type`` cohort, copies each domain's selected-sample PDB beside a hashed
manifest, and records the instrument contract (checkpoint digests, loops,
sampling steps, diffusion samples, seed, versions) and each wild type's pLDDT and
pTM. Produced once and reused for every contrast; never refolded per comparison.
Primary reference is predicted, so both sides of the TM-score share one
instrument. A wild type with mean pLDDT < 70 is flagged as a weak reference.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation import fold_recapitulation as fr  # noqa: E402
from src.capability.generation import structure_evidence as se  # noqa: E402

LOW_CONFIDENCE_PLDDT = 70.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="accepted and unused; CPU-only")
    parser.add_argument("--structure", type=Path, required=True, help="evidence tree of the wild-type cohort")
    parser.add_argument("--cohort", type=Path, required=True, help="the fold_cohort.jsonl that tree was folded from")
    args = parser.parse_args()
    wild = fr.load_wild_types(REPO_ROOT)
    summary = se.verify_evidence(se.load_cohort(args.cohort), args.structure, shard_index=0, num_shards=1)
    signature = se.tree_signature(args.structure)
    receipts = sorted(args.structure.glob("worker-*.json"))
    contract = json.loads(receipts[0].read_text())["contract"]
    args.out.mkdir(parents=True, exist_ok=True)
    domains = {}
    for domain, sequence in sorted(wild.items()):
        fold = fr.load_structure(args.structure, sequence)
        target = args.out / f"{domain}.pdb"
        shutil.copyfile(fold["pdb"], target)
        if sha256_file(target) != fold["pdb_sha256"]:
            raise ValueError(f"{domain}: copied PDB does not match the evidence tree digest")
        domains[domain] = {
            "sequence": sequence, "length": len(sequence), "pdb": target.name, "pdb_sha256": fold["pdb_sha256"],
            "plddt_mean": fold["plddt_mean"], "ptm": fold["ptm"], "source_object": fold["object"],
            "low_confidence_reference": fold["plddt_mean"] < LOW_CONFIDENCE_PLDDT,
        }
    write_json(args.out / "wild_type_references.json", {
        "schema_version": fr.REFERENCE_SCHEMA, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_signature": signature, "instrument_contract": contract,
        "cohort_sha256": sha256_file(args.cohort), "domains": domains,
        "verification": summary, "code_sha256": sha256_file(Path(__file__)),
        "interpretation": se.INTERPRETATION,
    })
    print(json.dumps({d: [round(e["plddt_mean"], 1), round(e["ptm"], 3)] for d, e in domains.items()}))


if __name__ == "__main__":
    main()
