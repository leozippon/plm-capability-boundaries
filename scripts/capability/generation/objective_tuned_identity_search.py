#!/usr/bin/env python3
"""Search the objective-tuned cohort's not-yet-searched sequences for their nearest natural relative.

The campaign's primary family is read *within* an identity stratum, and a folded
structure carries no stratum until its sequence has been searched against the
reference corpus. At eighteen of sixty-eight shards that search was done in two
passes -- 8,390 queries at the twelve-shard checkpoint, then 4,550 covering the
delta -- whose union is exactly the 12,640 structures the analysis tables carry.
This generalises those two passes into one repeatable step, so the remaining
shards do not each need a hand-written merge.

**Identity does not depend on the fold.** Percent identity to the corpus is a
property of a sequence, so this does not have to follow the folder shard by
shard: it can search any sequence of the frozen cohort at any time, and the
analysis joins the annotation onto whichever structures have folded. Scoping the
earlier passes to folded shards kept their tables aligned; it was never a
dependency. The practical consequence is that the whole remainder can be
searched in one pass now rather than queued behind the fold -- which matters
because the corpus index is one of the three payloads the 2026-11-12 evacuation
deliberately left behind, so it exists only on this host.

**Why one pass rather than a batch per shard.** The search is database-scan
bound. The three recorded runs took 1,663 s for 2,000 queries, 1,580 s for 4,550
and 2,563 s for 8,390, so the corpus scan dominates and the marginal cost of a
query is a fraction of a second. Running a batch per shard would pay that scan
fifty times over for no gain. One pass over everything outstanding is both the
cheapest and the simplest thing that is correct.

**What makes it safely re-runnable.** The set to search is recomputed every time
as the cohort minus the union of every prior annotation sidecar passed in, so a
second invocation with nothing new to do exits without searching, and an
invocation after more sequences appear searches only those. Each run writes its
own output directory; nothing is overwritten.

**The equivalence gate, which is the project's own.** A batch must not change the
answer for a sequence that was already searched in a different batch, or the
union of several passes would not be one measurement. So every run re-searches a
deterministic sample of already-searched digests as controls and refuses unless
each reproduces its recorded identity and stratum exactly. That is the check
``merge_nearest_neighbour_hits.py`` applied to the eighteen-shard delta with 300
controls, hoisted out of that one-off script. The stratum rule, the identity
definition and the masking setting are imported from
:mod:`src.capability.context.homology` through the existing search entry point
rather than restated, so this cannot drift from them.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402

SEARCH_ENTRY_POINT = REPO_ROOT / "scripts" / "capability" / "evaluation" / "search_generated_novelty.py"
RECEIPT = "identity_search_increment.json"
UNION_SIDECAR = "identity_annotations_union.jsonl"
BATCH_COHORT = "batch_cohort.jsonl"

#: Tolerance on a control's reproduced identity. The quantity is 100*nident/qlen
#: computed from integers, so an exact match is the expectation and this guards
#: only float formatting.
IDENTITY_TOLERANCE = 1e-9


def load_cohort(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "sequence", "length"):
            if key not in row:
                raise SystemExit(f"{path} line {number}: missing {key!r}")
        if row["id"] in seen:
            raise SystemExit(f"{path} line {number}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path} carries no records")
    return rows


def load_prior(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Every digest already searched, with the identity and stratum recorded for it.

    A digest searched in more than one prior pass must agree with itself across
    them, or the union is not one measurement and there is nothing to extend.
    """

    prior: dict[str, dict[str, Any]] = {}
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["id"])
            entry = {
                "nearest_corpus_identity": float(row["nearest_corpus_identity"]),
                "identity_stratum": str(row["identity_stratum"]),
                "source": str(path),
            }
            established = prior.get(key)
            if established is None:
                prior[key] = entry
                continue
            same = (
                abs(established["nearest_corpus_identity"] - entry["nearest_corpus_identity"])
                <= IDENTITY_TOLERANCE
                and established["identity_stratum"] == entry["identity_stratum"]
            )
            if not same:
                raise SystemExit(
                    f"{path} line {number}: digest {key} was already recorded as "
                    f"{established['nearest_corpus_identity']} "
                    f"({established['identity_stratum']}) by {established['source']}, "
                    f"and this file records {entry['nearest_corpus_identity']} "
                    f"({entry['identity_stratum']}); the prior passes disagree"
                )
    return prior


def choose_controls(candidates: list[str], count: int, seed: int) -> list[str]:
    """A deterministic sample of already-searched digests, for the equivalence gate."""

    ordered = sorted(candidates)
    if count >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, count))


def check_controls(
    prior: dict[str, dict[str, Any]], produced: dict[str, dict[str, Any]], controls: list[str]
) -> dict[str, Any]:
    disagreements = []
    for digest in controls:
        was = prior[digest]
        now = produced.get(digest)
        if now is None:
            disagreements.append({"id": digest, "reason": "control absent from this run's output"})
            continue
        identity_moved = (
            abs(float(now["nearest_corpus_identity"]) - was["nearest_corpus_identity"])
            > IDENTITY_TOLERANCE
        )
        stratum_moved = str(now["identity_stratum"]) != was["identity_stratum"]
        if identity_moved or stratum_moved:
            disagreements.append(
                {
                    "id": digest,
                    "recorded_identity": was["nearest_corpus_identity"],
                    "reproduced_identity": float(now["nearest_corpus_identity"]),
                    "recorded_stratum": was["identity_stratum"],
                    "reproduced_stratum": str(now["identity_stratum"]),
                }
            )
    return {
        "controls": len(controls),
        "disagreements": len(disagreements),
        "detail": disagreements[:20],
        "identity_tolerance": IDENTITY_TOLERANCE,
        "passed": not disagreements,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", type=Path, required=True, help="the frozen fold cohort JSONL; id is the sequence digest")
    parser.add_argument(
        "--searched",
        type=Path,
        nargs="+",
        required=True,
        help="prior generated_novelty.jsonl sidecars; their union is what counts as already searched",
    )
    parser.add_argument("--out", type=Path, required=True, help="a new directory for this pass")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--database-source", type=Path, required=True)
    parser.add_argument("--database-record-count", type=int, required=True)
    parser.add_argument("--diamond-tarball", type=Path, required=True)
    parser.add_argument("--diamond-checksum", type=Path, required=True)
    parser.add_argument("--controls", type=int, default=300, help="already-searched digests re-searched as the equivalence gate")
    parser.add_argument("--control-seed", type=int, default=20261114)
    parser.add_argument(
        "--controls-only",
        action="store_true",
        help="search the controls and nothing else, to settle equivalence before committing a full pass",
    )
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--sensitivity", default="very-sensitive")
    parser.add_argument("--evalue", type=float, default=1e-3)
    parser.add_argument("--max-target-seqs", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true", help="report what would be searched and stop")
    args = parser.parse_args()
    if args.controls < 1:
        parser.error("--controls must be positive; the equivalence gate is not optional")

    rows = load_cohort(args.cohort)
    by_id = {str(row["id"]): row for row in rows}
    prior = load_prior(list(args.searched))
    outstanding = [row for row in rows if str(row["id"]) not in prior]
    control_candidates = [digest for digest in prior if digest in by_id]
    if not control_candidates:
        raise SystemExit(
            "no already-searched digest is present in this cohort, so the equivalence gate "
            "cannot be formed; check that --searched belongs to this cohort"
        )
    controls = choose_controls(control_candidates, args.controls, args.control_seed)

    summary = {
        "cohort_records": len(rows),
        "already_searched": len(prior),
        "outstanding": len(outstanding),
        "controls": len(controls),
        "controls_only": bool(args.controls_only),
    }
    if not outstanding and not args.controls_only:
        print(json.dumps({**summary, "status": "nothing_to_search"}, sort_keys=True))
        return 0
    if args.dry_run:
        print(json.dumps({**summary, "status": "dry_run"}, sort_keys=True))
        return 0

    batch_rows = [by_id[digest] for digest in controls]
    if not args.controls_only:
        batch_rows += outstanding
    # Sorting by id makes the query order, and therefore the query FASTA, a
    # function of the batch contents alone.
    batch_rows.sort(key=lambda row: str(row["id"]))
    args.out.mkdir(parents=True, exist_ok=True)
    batch_path = args.out / BATCH_COHORT
    batch_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in batch_rows), encoding="utf-8"
    )
    roles = sorted({str(row.get("role")) for row in batch_rows if row.get("role")})
    if not roles:
        raise SystemExit(f"{args.cohort} carries no role on its records; the search selects on roles")

    search_out = args.out / "search"
    command = [
        sys.executable,
        str(SEARCH_ENTRY_POINT),
        "--cohort", str(batch_path),
        "--roles", *roles,
        "--database", str(args.database),
        "--database-source", str(args.database_source),
        "--database-record-count", str(args.database_record_count),
        "--diamond-tarball", str(args.diamond_tarball),
        "--diamond-checksum", str(args.diamond_checksum),
        "--sensitivity", args.sensitivity,
        "--evalue", str(args.evalue),
        "--max-target-seqs", str(args.max_target_seqs),
        "--threads", str(args.threads),
        "--progress-log", str(args.out / "diamond_progress.log"),
        "--out", str(search_out),
    ]
    print(json.dumps({**summary, "status": "searching", "queries": len(batch_rows)}, sort_keys=True), flush=True)
    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    if completed.returncode != 0:
        write_json(args.out / RECEIPT, {
            "status": "search_failed",
            "search_returncode": completed.returncode,
            "command": command,
            **summary,
        })
        raise SystemExit(f"the search exited {completed.returncode}; nothing was promoted")

    sidecar = search_out / "generated_novelty.jsonl"
    produced = {}
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            produced[str(row["id"])] = row
    gate = check_controls(prior, produced, controls)

    receipt: dict[str, Any] = {
        "status": "complete" if gate["passed"] else "refused_controls_disagree",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cohort": str(args.cohort),
        "cohort_sha256": sha256_file(args.cohort),
        "searched_inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.searched
        ],
        "batch_cohort": str(batch_path),
        "batch_cohort_sha256": sha256_file(batch_path),
        "annotations_jsonl": str(sidecar),
        "annotations_sha256": sha256_file(sidecar),
        "queries": len(batch_rows),
        "newly_searched": 0 if args.controls_only else len(outstanding),
        "equivalence_gate": gate,
        "command": command,
        **summary,
    }
    if not gate["passed"]:
        write_json(args.out / RECEIPT, receipt)
        print(json.dumps({"status": receipt["status"], "equivalence_gate": gate}, sort_keys=True))
        raise SystemExit(
            "controls did not reproduce their recorded identity or stratum; the union is "
            "not one measurement and nothing was promoted. Stop and report."
        )

    # The union is written only behind a passing gate, so a file that exists is a
    # file whose batches agreed.
    union = {digest: {"id": digest, **{k: v for k, v in entry.items() if k != "source"}}
             for digest, entry in prior.items()}
    for digest, row in produced.items():
        union[digest] = {
            "id": digest,
            "nearest_corpus_identity": float(row["nearest_corpus_identity"]),
            "identity_stratum": str(row["identity_stratum"]),
        }
    union_path = args.out / UNION_SIDECAR
    union_path.write_text(
        "".join(json.dumps(union[key], sort_keys=True) + "\n" for key in sorted(union)),
        encoding="utf-8",
    )
    receipt["union_jsonl"] = str(union_path)
    receipt["union_records"] = len(union)
    receipt["union_sha256"] = sha256_file(union_path)
    write_json(args.out / RECEIPT, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "queries": receipt["queries"],
        "newly_searched": receipt["newly_searched"],
        "union_records": receipt["union_records"],
        "equivalence_gate": {k: gate[k] for k in ("controls", "disagreements", "passed")},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
