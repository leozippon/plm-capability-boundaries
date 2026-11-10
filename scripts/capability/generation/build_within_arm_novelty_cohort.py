#!/usr/bin/env python3
"""Freeze the within-arm novelty cohort of the EC-conditioned pair.

The 2026-11-10 amendment reads the ZymCTRL pair within-arm only: the
distribution of identity to the nearest natural sequence under native EC
conditioning, with the mismatched-label negative beside it. That reading needs
identity over the query against UniRef50, and no such search covered these
generations -- the existing ``results/R6/generation_novelty_20261008`` search is
of the *replication* campaign and overlaps these cells in sixteen sequences.
This script freezes the cohort that search needs, from artefacts that already
exist, so that no sequence is generated again.

Two sources, and both are checked against each other. The sequences come from
the frozen EXP-R2-227 attempt ledger, which carries an identifier, the bytes and
the condition per attempt. The *population* comes from the conditioned-generation
artefacts themselves, and every canonical generation in the ledger must appear in
them: a cohort assembled from the ledger alone could carry a row from some other
campaign and nothing here would notice.

The natural references in that ledger are excluded by role. They are oracle
reference draws rather than generations, and pooling them into a generated
identity distribution would move it toward the corpus by construction.

One row per distinct sequence, keyed by its digest, because identity over the
query is a property of the sequence and the downstream join is on that digest.
The attempt ledger remains the denominator; this cohort is the instrument's
input, not a sampling unit.

``reference_identity`` in the same ledger is a *different* instrument -- an
aligned-coverage-unavailable reference search, not identity over the query with
masking off -- and is never read here and never pooled with what this search
produces.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation import objective_tuned_analysis as ota  # noqa: E402
from src.capability.generation.structure_evidence import sequence_digest  # noqa: E402

COHORT = "within_arm_cohort.jsonl"
RECORD = "within_arm_novelty_cohort.json"
SCHEMA = "objective_tuned_within_arm_cohort"

#: Ledger roles that are generations of these arms. ``natural_reference`` is
#: excluded by name rather than by filtering on the condition string.
GENERATION_ROLES = frozenset({"generation", "unconditioned_floor"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence", type=Path, required=True,
                        help="the frozen EXP-R2-227 attempt ledger (attempts.jsonl)")
    parser.add_argument("--conditioned-root", type=Path, required=True,
                        help="the conditioned-generation results directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    arms = sorted({arm for pair in ota.WITHIN_ARM_PAIRS for arm in pair.arms})
    population: dict[str, set[tuple[str, str]]] = defaultdict(set)
    samples: Counter[tuple[str, str]] = Counter()
    for arm in arms:
        path = args.conditioned_root / f"generations_{arm}.json"
        if not path.is_file():
            raise SystemExit(f"{args.conditioned_root} holds no {path.name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for cell in (payload.get("cells") or {}).values():
            condition = str(cell["condition"])
            for sample in cell["samples"]:
                samples[(arm, condition)] += 1
                population[sequence_digest(str(sample))].add((arm, condition))
    if not population:
        raise SystemExit(f"{args.conditioned_root} carries no conditioned sample")

    rows: dict[str, dict[str, Any]] = {}
    seen = Counter()
    non_canonical = Counter()
    for line in args.evidence.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("arm") not in arms or row.get("role") not in GENERATION_ROLES:
            continue
        sequence = str(row["sequence"])
        digest = sequence_digest(sequence)
        if digest != row.get("sequence_sha256"):
            raise SystemExit(f"{row['id']}: the ledger digest disagrees with its bytes")
        if digest not in population:
            raise SystemExit(
                f"{row['id']}: a generation of {row['arm']} that is not in "
                f"{args.conditioned_root}; this ledger is not of that campaign")
        seen[(str(row["arm"]), str(row.get("condition")))] += 1
        if not sequence or set(sequence) - set(AA20):
            non_canonical[str(row["arm"])] += 1
            continue
        rows[digest] = {"id": digest, "sequence": sequence, "length": len(sequence),
                        "role": "generation"}
    if not rows:
        raise SystemExit(f"{args.evidence} carries no canonical generation of {arms}")
    uncovered = sorted(set(population) - set(rows))

    args.out.mkdir(parents=True, exist_ok=True)
    cohort = args.out / COHORT
    cohort.write_text("".join(json.dumps(rows[key], sort_keys=True) + "\n"
                              for key in sorted(rows)), encoding="utf-8")
    record = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": "docs/objective-alignment.md, amendment of 2026-11-10",
        "arms": arms,
        "evidence_ledger": str(args.evidence),
        "evidence_ledger_sha256": sha256_file(args.evidence),
        "conditioned_root": str(args.conditioned_root),
        "conditioned_samples": {f"{arm}|{condition}": count
                                for (arm, condition), count in sorted(samples.items())},
        "ledger_attempts_selected": {f"{arm}|{condition}": count
                                     for (arm, condition), count in sorted(seen.items())},
        "excluded_roles": "natural_reference; oracle reference draws are not generations",
        "non_canonical_attempts": dict(sorted(non_canonical.items())),
        "distinct_conditioned_sequences": len(population),
        "cohort_rows": len(rows),
        "conditioned_sequences_not_in_the_cohort": len(uncovered),
        "cohort": str(cohort),
        "cohort_sha256": sha256_file(cohort),
        "identity_instrument": (
            "identity over the query against UniRef50 with masking off, by "
            "scripts/capability/evaluation/search_generated_novelty.py. The "
            "reference_identity column of the same ledger is a different instrument and "
            "is never pooled with it"),
        "code_sha256": sha256_file(Path(__file__)),
    }
    write_json(args.out / RECORD, record)
    print(json.dumps({key: record[key] for key in
                      ("cohort_rows", "distinct_conditioned_sequences",
                       "conditioned_sequences_not_in_the_cohort", "non_canonical_attempts")},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
