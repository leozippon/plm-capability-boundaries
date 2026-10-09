#!/usr/bin/env python3
"""Nearest natural relative of every generated sequence, by homology search.

Why E17 needs this. The most plausible alternative explanation for a
likelihood-selection gain is that the model's likelihood is highest on the
sequences most like the natural proteins it was trained on, so selecting by
likelihood would be selecting by retrieval rather than by design quality. That
explanation is testable only against a corpus, which is what this stage
measures: for each sequence, the percent of the *query* that is identically
matched by its closest UniRef50 relative, and the identity band that falls in.

Identity is expressed over the query, not over the aligned region, because a
corpus entry that aligns perfectly to 60% of a sequence is not a stored copy of
it; ``pident`` would call that 100 and put it in the near-duplicate band.

``--masking 0`` is not a tuning choice and the reason belongs beside the number.
DIAMOND masks low-complexity and repetitive query regions by default, and
repetitive sequences are exactly what likelihood selection is suspected of
concentrating on. With masking on, the alignment stops at the repeat:
:mod:`src.capability.context.homology` records a cohort member that is
byte-identical to ``UniRef50_Q3E8Z8`` over all 732 residues being reported at
``pident`` 100 over 607 aligned residues, i.e. identity over query 82.9, which
moved a verbatim corpus member out of the near-duplicate band. For a
novelty-versus-selection analysis that error runs in the flattering direction:
it would make the most repetitive, most retrievable sequences look the most
novel. The search, the identity definition and the masking setting therefore all
come from that module rather than being spelled here.

The search is database-scan bound, so one pass covers the whole cohort and the
cost barely depends on how many sequences are queried. It loads no model and
needs no GPU; ``--device`` is accepted because the campaign queue injects it.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402

COMPLETION = "generated_novelty.json"
SCHEMA_VERSION = "d1_generated_novelty_v1"

#: Why the masking setting is part of the measurement rather than a flag.
MASKING_RATIONALE = (
    "DIAMOND masks low-complexity and repetitive query regions by default, which "
    "truncates the alignment of exactly the sequences this analysis is about. A "
    "cohort member byte-identical to UniRef50_Q3E8Z8 over all 732 residues was "
    "reported at pident 100 over 607 aligned residues, i.e. identity over query "
    "82.9, placing a verbatim corpus member outside the near-duplicate band. The "
    "error runs in the flattering direction here: it makes the most repetitive and "
    "most retrievable sequences look the most novel. The search therefore runs with "
    "--masking 0, which is a correctness requirement and not a tuning choice"
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    gp.require_fresh_out(args.out, COMPLETION)

    from src.capability.context import homology as hm
    from src.capability.core.arms import Cohort

    rows = [
        json.loads(line)
        for line in args.cohort.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wanted = set(args.roles)
    selected = [
        row
        for row in rows
        if wanted & set(row.get("roles", [row.get("role")]))
        and not (set(str(row["sequence"])) - set(gp.AA20))
    ]
    selected.sort(key=lambda row: str(row["id"]))
    if not selected:
        raise SystemExit(f"{args.cohort} carries no canonical record with a role in {sorted(wanted)}")
    identifiers = [str(row["id"]) for row in selected]
    if len(set(identifiers)) != len(identifiers):
        raise SystemExit("the selected records carry duplicate identifiers")

    # The scan workspace holds an extracted aligner and a hit table; neither
    # belongs in the artefact that gets staged, so it defaults beside --out and
    # can be pointed elsewhere to keep the pushable result small.
    work = args.work or (args.out / "work")
    work.mkdir(parents=True, exist_ok=True)
    tool = hm.prepare_diamond(args.diamond_tarball, args.diamond_checksum, work / "diamond")
    sequences, letters = hm.database_counts(tool, args.database)
    database = hm.DiamondDatabase(
        path=args.database,
        source_fasta=args.database_source,
        source_records=args.database_record_count,
        sequences=sequences,
        letters=letters,
        makedb_command=("adopted-index", "built-outside-this-stage"),
    )

    cohort = Cohort(
        name="generated_novelty",
        kind="protein",
        records=[str(row["sequence"]) for row in selected],
        min_symbols=min(int(row["length"]) for row in selected),
        max_symbols=max(int(row["length"]) for row in selected),
        metadata={"sampling": {"mode": "frozen_cohort", "source": str(args.cohort)}},
    )
    query_fasta = work / "query.faa"
    positional = hm.write_query_fasta(cohort, query_fasta)
    position_of = {name: index for index, name in enumerate(positional)}

    command, log = hm.run_diamond_blastp(
        tool,
        database,
        query_fasta,
        work / "hits.tsv",
        threads=args.threads,
        sensitivity=args.sensitivity,
        evalue=args.evalue,
        max_target_seqs=args.max_target_seqs,
    )
    hits = hm.parse_hits(work / "hits.tsv")

    best: dict[int, Any] = {}
    for hit in hits:
        index = position_of.get(hit.query)
        if index is None:
            raise SystemExit(f"the search returned an unsubmitted query {hit.query!r}")
        current = best.get(index)
        if current is None or hit.identity_over_query > current.identity_over_query:
            best[index] = hit

    annotations: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        hit = best.get(index)
        identity = float(hit.identity_over_query) if hit is not None else 0.0
        annotations.append(
            {
                "id": str(row["id"]),
                "role": row.get("role"),
                "roles": row.get("roles"),
                "arm": row.get("arm"),
                "stratum": row.get("stratum"),
                "length": int(row["length"]),
                "nearest_corpus_identity": identity,
                "nearest_corpus_subject": None if hit is None else hit.subject,
                "nearest_corpus_evalue": None if hit is None else float(hit.evalue),
                "identity_stratum": hm.assign_stratum(identity),
                "searched": True,
                "had_any_hit": hit is not None,
            }
        )
    sidecar = args.out / "generated_novelty.jsonl"
    sidecar.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in annotations), encoding="utf-8"
    )

    identities = np.asarray([row["nearest_corpus_identity"] for row in annotations])
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cohort": str(args.cohort),
        "cohort_sha256": sha256_file(args.cohort),
        "roles": sorted(wanted),
        "n_searched": len(annotations),
        "n_with_any_hit": int(sum(1 for row in annotations if row["had_any_hit"])),
        "identity_over_query": {
            "definition": (
                "100 * nident / qlen: the percent of the query identically matched by "
                "its best corpus hit. Not pident, which is identity inside the aligned "
                "region and would call a 60%-length fragment a near-duplicate"
            ),
            "min": float(identities.min()),
            "max": float(identities.max()),
            "mean": float(identities.mean()),
            "median": float(np.median(identities)),
        },
        "identity_strata": dict(
            sorted(collections.Counter(row["identity_stratum"] for row in annotations).items())
        ),
        "stratum_edges": list(hm.STRATUM_EDGES),
        "masking_rationale": MASKING_RATIONALE,
        "search": {
            "workspace": str(work),
            "command": list(command),
            "log_tail": log,
            "sensitivity": args.sensitivity,
            "evalue": args.evalue,
            "max_target_seqs": int(args.max_target_seqs),
            "threads": int(args.threads),
        },
        "diamond": tool.record(),
        "database": database.record(),
        "ceiling": {
            "not_a_pretraining_corpus": (
                "UniRef50 is a reference corpus, not any model's declared pretraining "
                "set. A low identity here means no close relative in UniRef50; it is "
                "not proof that a sequence was absent from training data"
            ),
            "novelty_is_sequence_level": (
                "identity over query is a sequence-level statement. A novel sequence by "
                "this measure may still adopt a known fold, and a high-identity one is "
                "not thereby functional"
            ),
        },
        "annotations_jsonl": str(sidecar),
        "annotations_sha256": sha256_file(sidecar),
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True, help="the frozen evaluation cohort JSONL")
    parser.add_argument("--roles", nargs="+", default=["pool", "generated"])
    parser.add_argument("--database", type=Path, required=True, help="the DIAMOND index to search")
    parser.add_argument("--database-source", type=Path, required=True, help="the FASTA it was built from")
    parser.add_argument(
        "--database-record-count",
        type=int,
        required=True,
        help="records in the source release, so index coverage can be stated rather than assumed",
    )
    parser.add_argument("--diamond-tarball", type=Path, required=True)
    parser.add_argument("--diamond-checksum", type=Path, required=True)
    parser.add_argument("--sensitivity", default="very-sensitive")
    parser.add_argument("--evalue", type=float, default=1e-3)
    parser.add_argument("--max-target-seqs", type=int, default=25)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--work",
        type=Path,
        default=None,
        help="scan workspace; defaults to <out>/work and is never a data directory",
    )
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.max_target_seqs < 1 or args.database_record_count < 1:
        parser.error("threads, max-target-seqs and database-record-count must be positive")
    record = run(args)
    print(
        json.dumps(
            {
                key: record[key]
                for key in ("status", "n_searched", "n_with_any_hit", "identity_strata")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
