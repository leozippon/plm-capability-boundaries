#!/usr/bin/env python3
"""The evolutionary-profile control for a generated-versus-natural comparison.

A natural protein has relatives in a sequence corpus; a generated product mostly
does not. That asymmetry is the first alternative explanation for any difference
in how a model's likelihood responds to mutating the two, and in this programme
the profile channel has already overturned or qualified results three times. This
stage turns "no evolutionary-profile control applies to a generated protein" from
an assertion into a measurement.

One DIAMOND ``--very-sensitive --masking 0`` search of every cohort wild type
against the staged UniRef50 database, then, per wild type, the project's own
position-specific profile and the LOOKUP score of each scanned substitution. No
new profile code: ``context.homology`` runs and parses the search,
``context.profiles.build_profile`` forms the weighted column frequencies and
``profile_scores`` evaluates the mutation-local evolutionary statistic. The
search settings are the call site's own defaults for the same reason they are
there -- ``--very-sensitive`` because the claim that matters is the *negative*
one that a product has no relative, and ``--masking 0`` because this cohort is
full of internal repeats and default masking would stop the alignment at the
repeat and report the opposite of the truth.

Three things the output supports, and they are different questions:

* the retrievability profile of each origin, which is a measurement in itself --
  how many generated products have a retrievable relative at all, and at what
  identity;
* whether the evolutionary channel itself differs between the two origins, as
  the LOOKUP score contrast;
* whether the likelihood contrast survives once the channel is controlled, which
  the analysis stage does two ways: restricted to pairs where *neither* member
  has a retrievable profile, and after residualising on the mutation-local
  LOOKUP score.

A wild type with no admissible hit has no profile. That is recorded with its
reason and its LOOKUP score is absent; it is never replaced by a background-only
score, because a background score is a composition statistic and not an
evolutionary one.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.context import homology, profiles  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.interactions import generated_mutation as gm  # noqa: E402

COMPLETION = "generated_profile_control.json"
TABLE = "profile-control.json"
SCHEMA = "generated_profile_control_v1"

#: Search settings. Not tuning: see the module docstring and the reasons recorded
#: at ``homology.run_diamond_blastp``.
SENSITIVITY = "very-sensitive"
EVALUE = 1e-3
MAX_TARGET_SEQS = 100


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def variants_of(labels, wildtype: str):
    """The mutation strings as ``profiles`` wants them: one-based positions."""

    out = []
    for label in labels:
        if ":" in label:
            raise SystemExit(f"{label}: the profile control covers single substitutions")
        position = int(label[1:-1])
        if wildtype[position - 1] != label[0]:
            raise SystemExit(f"{label}: disagrees with its own wild type")
        out.append([(label[0], position, label[-1])])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", type=Path, required=True, help="the singles cohort")
    parser.add_argument("--database", type=Path,
                        default=ROOT / "data/homology_db/uniref50_full.dmnd")
    parser.add_argument("--corpus-fasta", type=Path, default=ROOT / "data/uniref50/uniref50.fasta")
    parser.add_argument("--background", type=Path,
                        help="a previous run's corpus-scan record, to skip the corpus pass")
    parser.add_argument("--diamond-tarball", type=Path,
                        default=ROOT / "external/tools/diamond-linux64-v2.1.24.tar.gz")
    parser.add_argument("--diamond-dir", type=Path, default=ROOT / "external/tools/diamond")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--max-target-seqs", type=int, default=MAX_TARGET_SEQS)
    parser.add_argument("--max-sequences", type=int, default=MAX_TARGET_SEQS,
                        help="profile depth cap; it right-censors Neff and is recorded")
    parser.add_argument("--hits", type=Path,
                        help="a previous run's DIAMOND output, to skip the search")
    parser.add_argument("--device", default="cpu",
                        help="accepted because the campaign queue injects it; unused")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.max_target_seqs < 1 or args.max_sequences < 1:
        parser.error("--threads, --max-target-seqs and --max-sequences are positive")

    out = gm.prepare_output_directory(args.out, COMPLETION)
    cohort = json.loads(Path(args.cohort).read_text())
    if cohort.get("schema") != gm.COHORT_SCHEMA or cohort.get("mode") != "singles":
        raise SystemExit(f"{args.cohort}: not a singles cohort of {gm.COHORT_SCHEMA}")
    assays = {row["assay"]: row for row in cohort["assays"]}
    gm.require_unique_assays(cohort["assays"])

    query = out / "queries.faa"
    with query.open("w") as handle:
        for assay, row in sorted(assays.items()):
            handle.write(f">{assay}\n{row['wildtype']}\n")

    tool = homology.prepare_diamond(args.diamond_tarball,
                                    Path(str(args.diamond_tarball) + ".sha256"),
                                    args.diamond_dir)
    sequences, letters = homology.database_counts(tool, args.database)
    database = homology.DiamondDatabase(
        path=args.database,
        source_fasta=args.corpus_fasta,
        source_records=sequences,
        sequences=sequences,
        letters=letters,
        makedb_command=("pre-staged",),
    )
    search = out / "hits.tsv"
    if args.hits is not None:
        search = Path(args.hits)
        command, log = ["reused", str(search)], "search reused from a previous run"
    else:
        command, log = homology.run_diamond_blastp(
            tool, database, query, search,
            threads=args.threads, sensitivity=SENSITIVITY, evalue=EVALUE,
            max_target_seqs=args.max_target_seqs, fields=homology.ALIGNMENT_FIELDS,
        )
    hits = homology.parse_hits(search, fields=homology.ALIGNMENT_FIELDS)
    by_query: dict[str, list] = {}
    for hit in hits:
        by_query.setdefault(hit.query, []).append(hit)

    if args.background is not None:
        scan_record = json.loads(Path(args.background).read_text())
        background = np.asarray(scan_record["background_vector"], dtype=np.float64)
    else:
        scan = profiles.scan_corpus(args.corpus_fasta, targets=())
        background = scan.background_vector()
        scan_record = dict(scan.record(), background_vector=[float(v) for v in background])
        write_json(out / "corpus-scan.json", scan_record)

    rows, sequence_rows = [], []
    for assay, row in sorted(assays.items()):
        found = by_query.get(assay, [])
        best = max((homology.potential_identity_over_query(hit) for hit in found), default=0.0)
        record = {
            "sequence_id": assay,
            "origin": row["origin"],
            "group": row["group"],
            "length": int(row["length"]),
            "n_hits": len(found),
            "max_identity_over_query": best,
            "stratum": homology.assign_stratum(best) if found else None,
            "profile": None,
            "reason": None,
        }
        if not found:
            record["reason"] = "no admissible UniRef50 hit; no evolutionary profile exists"
            sequence_rows.append(record)
            for label in row["mutants"]:
                rows.append({"sequence_id": assay, "origin": row["origin"],
                             "group": row["group"], "mutation": label,
                             "lookup_score_nats": None})
            continue
        try:
            profile = profiles.build_profile(
                row["wildtype"], assay, found, max_sequences=args.max_sequences
            )
            scores = profiles.profile_scores(
                profile, background, variants_of(row["mutants"], row["wildtype"])
            )
        except ValueError as error:
            record["reason"] = f"profile refused: {error}"
            sequence_rows.append(record)
            for label in row["mutants"]:
                rows.append({"sequence_id": assay, "origin": row["origin"],
                             "group": row["group"], "mutation": label,
                             "lookup_score_nats": None})
            continue
        record["profile"] = profile.record()
        sequence_rows.append(record)
        for label, score in zip(row["mutants"], scores):
            rows.append({"sequence_id": assay, "origin": row["origin"], "group": row["group"],
                         "mutation": label, "lookup_score_nats": float(score)})

    def share(origin: str, predicate) -> dict:
        members = [row for row in sequence_rows if row["origin"] == origin]
        hitting = [row for row in members if predicate(row)]
        return {"sequences": len(members), "matching": len(hitting),
                "share": len(hitting) / len(members) if members else None}

    summary = {
        "with_any_hit": {o: share(o, lambda r: r["n_hits"] > 0) for o in gm.ORIGINS},
        "with_a_profile": {o: share(o, lambda r: r["profile"] is not None) for o in gm.ORIGINS},
        "with_a_close_relative_70pc": {
            o: share(o, lambda r: r["max_identity_over_query"] >= 70.0) for o in gm.ORIGINS
        },
        "median_log10_neff": {
            o: (
                float(np.median([r["profile"]["log10_neff"] for r in sequence_rows
                                 if r["origin"] == o and r["profile"] is not None]))
                if any(r["origin"] == o and r["profile"] is not None for r in sequence_rows)
                else None
            )
            for o in gm.ORIGINS
        },
    }
    write_json(out / TABLE, {
        "schema": SCHEMA, "cohort": str(args.cohort), "mutations": rows,
        "sequences": sequence_rows,
    })
    write_json(out / COMPLETION, {
        "status": "complete",
        "schema": SCHEMA,
        "created_utc": _now(),
        "question": (
            "does a retrievable evolutionary profile, and the mutation-local LOOKUP score "
            "it defines, differ between generated products and their matched natural partners?"
        ),
        "table": TABLE,
        "table_sha256": sha256_file(out / TABLE),
        "cohort": {"path": str(args.cohort), "sha256": sha256_file(args.cohort)},
        "search": {
            "sensitivity": SENSITIVITY, "evalue": EVALUE,
            "max_target_seqs": args.max_target_seqs, "threads": args.threads,
            "fields": list(homology.ALIGNMENT_FIELDS), "command": list(command),
            "log_tail": log, "hits": len(hits), "queries": len(assays),
            "output_sha256": sha256_file(search),
        },
        "tool": tool.record(),
        "database": database.record(),
        "profile": {"max_sequences": args.max_sequences,
                    "coverage_floor": profiles.PROFILE_COVERAGE_FLOOR,
                    "reweight_identity": profiles.REWEIGHT_IDENTITY_FLOOR,
                    "pseudocount_alpha": profiles.PSEUDOCOUNT_ALPHA},
        "summary": summary,
        "limitations": [
            "A product with no admissible hit has no profile and no LOOKUP score; the absence "
            "is the measurement and is never replaced by a background-only score.",
            "The profile depth is capped, which right-censors Neff at the top of the range.",
            "The searched corpus is the staged UniRef50 snapshot, whose release string the "
            "project records as unrecoverable, so this is a retrievability measurement against "
            "one reference and not against any model's own pretraining corpus.",
        ],
        "code_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "scripts/capability/interactions/measure_generated_profile_control.py",
                "src/capability/context/homology.py",
                "src/capability/context/profiles.py",
            )
        },
    })


if __name__ == "__main__":
    main()
