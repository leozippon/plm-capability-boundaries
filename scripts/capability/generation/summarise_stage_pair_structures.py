#!/usr/bin/env python3
"""Join the frozen ProLLaMA stage-pair set to its ESMFold2 folds and derive pairwise geometry.

E18 asks whether Stage 2's better mutation-effect prediction comes with better
generation. This stage answers the structural half of that from folds produced by
the project's own instrument (``run_structure_evidence.py``, ESMFold2), and
writes the pairwise arrays the two later analyses will need.

Three products, all CPU:

* the per-stage predicted-confidence comparison -- mean CA pLDDT, the share of
  residues above 70, pTM and mean PAE -- summarised at the campaign stream as the
  unit, because seed streams resample from fixed checkpoints and are not
  independent lineages;
* per sequence, a ``pairwise/<id>.npz`` carrying the CB-CB distance matrix, the
  contact map at the declared cutoff, the instrument's full predicted-aligned-error
  matrix and the per-residue confidence. The ESM contact head is unusable on this
  checkpoint, so contacts come from coordinates; writing them once means the
  double-mutant analysis never has to re-fold;
* one index joining every frozen identifier to its structure, its geometry and its
  sequence properties.

Confidence is not stability. No stability predictor and no measurement is applied
to a generated sequence here, and a predicted structure cannot demonstrate
folding or function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import write_json  # noqa: E402
from src.capability.generation import stage_pair as sp  # noqa: E402
from src.capability.generation import stage_pair_structures as sps  # noqa: E402

COMPLETION = "stage_pair_structures.json"
READOUTS = (
    "mean_ca_plddt",
    "fraction_ca_plddt_ge70",
    "fraction_ca_plddt_ge90",
    "ptm",
    "mean_pae_angstrom",
    "mean_pae_on_contacts_angstrom",
    "contact_density",
    "length",
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = sps.read_jsonl(args.sequences)
    roots = sps.iter_structure_roots(args.structures)
    args.out.mkdir(parents=True, exist_ok=True)
    pairwise_dir = args.out / "pairwise"

    summarised: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    missing: list[str] = []
    unfolded: list[dict[str, Any]] = []
    for record in records:
        found = None
        for root in roots:
            found = sps.load_structure(root, record["sequence_sha256"])
            if found is not None:
                break
        if found is None:
            missing.append(record["id"])
            continue
        if found["status"] != "ok":
            unfolded.append(
                {"id": record["id"], "status": found["status"],
                 "reason": found["result"].get("reason")}
            )
            continue
        summarised.append(sps.summarise_record(record, found, cutoff=args.contact_cutoff))
        if not args.no_pairwise:
            pairwise.append(
                sps.write_pairwise(pairwise_dir, record, found, cutoff=args.contact_cutoff)
            )

    if not summarised:
        raise SystemExit(
            f"none of the {len(records)} frozen sequences has a completed fold under "
            f"{[str(root) for root in roots]}; a structural summary of an unfolded set "
            "is refused rather than reported as an absence of confidence"
        )
    stages = {row["stage"] for row in summarised}
    if stages != set(sp.STAGES):
        raise SystemExit(
            f"folds are present for {sorted(stages)} only; a one-sided set cannot "
            "answer a within-model stage comparison"
        )

    index = args.out / "stage_pair_structure_index.jsonl"
    index.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in summarised), encoding="utf-8"
    )

    per_stage = {field: sps.stage_summary(summarised, field=field) for field in READOUTS}
    contrasts = {
        field: sp.paired_property_contrast(
            [
                {"stage": row["stage"], "stream": row["stream"], field: row[field]}
                for row in summarised
                if row.get(field) is not None
            ],
            field=field,
        )
        for field in READOUTS
    }
    record = {
        "schema_version": sps.SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sequences": str(args.sequences),
        "sequences_sha256": hashlib.sha256(args.sequences.read_bytes()).hexdigest(),
        "structure_roots": [str(root) for root in roots],
        "instrument": "ESMFold2 via run_structure_evidence.py; this stage does not fold",
        "contact_cutoff_angstrom": float(args.contact_cutoff),
        "glycine_uses_ca": sps.GLYCINE_USES_CA,
        "n_frozen_sequences": len(records),
        "n_summarised": len(summarised),
        "n_without_a_fold_object": len(missing),
        "sequences_without_a_fold_object": missing[:20],
        "n_not_evaluable_or_failed": len(unfolded),
        "not_evaluable_or_failed": unfolded[:20],
        "per_stage_counts": dict(Counter(row["stage"] for row in summarised)),
        "per_stage": per_stage,
        "stage_2_minus_stage_1": contrasts,
        "pairwise_arrays": {
            "written": len(pairwise),
            "directory": None if args.no_pairwise else str(pairwise_dir),
            "key": "the frozen-set identifier, <id>.npz",
            "contents": [
                "cb_distance_angstrom",
                "contact_map",
                "predicted_aligned_error_angstrom",
                "ca_plddt_0_100",
                "atom_b_factors",
                "contact_cutoff_angstrom",
                "length",
            ],
            "note": (
                "written once so a later mutation or double-mutant analysis never has "
                "to re-fold; ESM's own contact head is unusable on this checkpoint"
            ),
        },
        "index_file": str(index),
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "ceiling": dict(sps.CEILING),
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True,
                        help="stage_pair_sequences.jsonl from freeze_stage_pair_sequences.py")
    parser.add_argument("--structures", type=Path, nargs="+", required=True,
                        help="one or more run_structure_evidence.py output roots")
    parser.add_argument("--contact-cutoff", type=float, default=sps.CONTACT_CUTOFF_ANGSTROM)
    parser.add_argument("--no-pairwise", action="store_true",
                        help="summarise without writing the per-sequence pairwise arrays")
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = run(args)
    print(json.dumps({key: record[key] for key in ("status", "n_summarised", "per_stage_counts",
                                                   "per_stage")}, sort_keys=True)[:4000])


if __name__ == "__main__":
    main()
