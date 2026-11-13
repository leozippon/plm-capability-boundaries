#!/usr/bin/env python3
"""Read each arm's own likelihood of its own products into the ``--selector-score`` shape.

What this supplies, and why it is a read rather than a scoring pass
------------------------------------------------------------------
``analyse_objective_tuned.selector_competence`` measures the pLDDT headroom an
oracle reaches at a budget of ten inside an arm's own pool, and the share of that
headroom a *supplied* selector captures. Without a selector it records
``not_measured``, because the free-length generation ledgers carry no
per-sequence likelihood.

That likelihood does exist. ``scripts/capability/generation/score_cohort_likelihood.py``
recomputed it on the H200 cluster for the whole objective-tuned cohort on
2026-11-11, writing one row per distinct (source arm, sequence) with the token
count, the summed and the mean negative log-likelihood. Its own-model cells --
``<arm>__on__<arm>.jsonl`` -- are exactly the quantity this endpoint wants: the
generating checkpoint's likelihood of the sequences it produced, under the arm's
own prompt contract and residue alphabet, at temperature 1 and in float32. So
this script joins those cells onto the folded sample rather than scoring a model
again, and refuses rather than degrades on any gap.

The selector, stated once
-------------------------
The score is ``-mean_nll``: mean log-likelihood per residue token, in nats, so a
higher score is a more likely sequence and ``captured_fraction``'s top-k by
descending score is the arm's own most-likely ten. The likelihood is comparable
within an arm, which is all a within-pool selection endpoint needs, and the
endpoint never compares two arms' nats.

What is checked before anything is written
------------------------------------------
* every folded attempt of every arm carries a score row, and no arm's score file
  holds a duplicate digest -- an uncovered digest would make the arm
  ``unresolved`` inside the analysis module rather than fail, so it is refused
  here, where it is visible;
* the scored token count equals the sequence's own length in the fold cohort,
  which is what ties a score to the bytes it was computed on (one residue token
  per residue, under the arm's declared alphabet contract);
* ``mean_nll`` is ``sum_nll / n_tokens`` and both are finite;
* with ``--replicate``, a second independent run of the same scorer is compared
  row by row: the largest absolute disagreement in mean NLL per arm, and whether
  the two runs select the same top ten. Two runs agreeing is the only
  reproduction evidence available without a GPU allocation, and it is recorded
  rather than assumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation import objective_tuned_analysis as ota  # noqa: E402
from src.capability.generation import objective_tuned_arms as ot  # noqa: E402
from src.capability.generation.structure_evidence import sequence_digest  # noqa: E402

#: The selector, in one place. ``captured_fraction`` ranks by descending score.
SELECTOR = (
    "the arm's own likelihood of its own product: -mean_nll, the mean "
    "log-likelihood per residue token in nats under the generating checkpoint, "
    "its own prompt contract and its own residue alphabet, at temperature 1 and "
    "in float32. Higher is more likely. Comparable within an arm only"
)

#: Printed-precision slack on the scorer's own ``mean_nll = sum_nll / n_tokens``.
MEAN_TOLERANCE = 1e-9


def fail(message: str) -> None:
    raise SystemExit(f"build_selector_scores: {message}")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_scores(directory: Path, arm: str) -> dict[str, dict[str, Any]]:
    """One arm's own-model score rows, keyed by sequence digest."""

    path = directory / f"{arm}__on__{arm}.jsonl"
    if not path.is_file():
        fail(f"{path} is missing; the own-model cell of {arm} is what this endpoint scores")
    rows: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        digest = row.get("sequence_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            fail(f"{path} line {number}: no sequence_sha256")
        if digest in rows:
            fail(f"{path}: {digest} appears twice; the scorer's unit is the distinct sequence")
        total, count = float(row["sum_nll"]), int(row["n_tokens"])
        mean = float(row["mean_nll"])
        if count < 1 or not math.isfinite(total) or not math.isfinite(mean):
            fail(f"{path} line {number}: {count} tokens, sum {total}, mean {mean}")
        if abs(mean - total / count) > MEAN_TOLERANCE:
            fail(f"{path} line {number}: mean_nll {mean} is not sum_nll/n_tokens {total / count}")
        rows[digest] = {"n_tokens": count, "mean_nll": mean}
    return rows


def read_folded(neighbour: Path, cohort: Path) -> dict[str, list[tuple[str, int]]]:
    """The folded sample as ``{arm: [(digest, length)]}``, lengths from the cohort bytes."""

    lengths: dict[str, int] = {}
    for line in cohort.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sequence = str(record["sequence"])
        digest = sequence_digest(sequence)
        if digest != str(record["id"]):
            fail(f"the fold cohort keys a sequence by {record['id']} but it digests to {digest}")
        lengths[digest] = len(sequence)
    folded: dict[str, list[tuple[str, int]]] = {}
    for row in read_tsv(neighbour):
        digest = row["sequence_sha256"]
        if digest not in lengths:
            fail(f"{digest} is folded but is not in the fold cohort, so its length is unknown")
        folded.setdefault(row["arm"], []).append((digest, lengths[digest]))
    declared = {arm for pair in ot.pairs_in_lane(ot.LANE_FREE_LENGTH) for arm in pair.arms}
    if set(folded) != declared:
        fail(f"the folded sample holds arms {sorted(folded)} against the declared free-length "
             f"arms {sorted(declared)}")
    return {arm: sorted(members) for arm, members in sorted(folded.items())}


def top_digests(scores: dict[str, dict[str, Any]], digests: list[str]) -> list[str]:
    """The selector's own budget-sized pick over one arm's folded pool."""

    ordered = sorted(digests, key=lambda digest: (scores[digest]["mean_nll"], digest))
    return ordered[: ota.SELECTION_BUDGET]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--likelihood", type=Path, required=True,
                        help="directory holding <arm>__on__<arm>.jsonl for the eight arms")
    parser.add_argument("--neighbour", type=Path, required=True,
                        help="the per-structure nearest-neighbour table: the folded sample")
    parser.add_argument("--cohort", type=Path, required=True, help="the frozen fold cohort")
    parser.add_argument("--replicate", type=Path, default=None,
                        help="a second independent run of the same scorer, compared row by row")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    folded = read_folded(args.neighbour, args.cohort)
    per_arm: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for arm, members in folded.items():
        scores = read_scores(args.likelihood, arm)
        missing = [digest for digest, _ in members if digest not in scores]
        if missing:
            fail(f"{arm}: {len(missing)} folded attempts carry no score (first {missing[:3]}). "
                 "An uncovered digest makes the arm unresolved inside the analysis module "
                 "instead of failing, so it is refused here")
        mismatched = [digest for digest, length in members
                      if scores[digest]["n_tokens"] != length]
        if mismatched:
            fail(f"{arm}: {len(mismatched)} scores were computed over a token count that is not "
                 f"the sequence's length (first {mismatched[:3]}); the score and the bytes "
                 "disagree")
        for digest, length in members:
            rows.append({"arm": arm, "sequence_sha256": digest,
                         "score": -scores[digest]["mean_nll"],
                         "mean_nll_per_residue_nats": scores[digest]["mean_nll"],
                         "n_tokens": length})
        entry: dict[str, Any] = {
            "n_folded_scored": len(members),
            "n_scored_in_source": len(scores),
            "mean_nll_over_folded": sum(scores[d]["mean_nll"] for d, _ in members) / len(members),
            "source": str(args.likelihood / f"{arm}__on__{arm}.jsonl"),
            "source_sha256": sha256_file(args.likelihood / f"{arm}__on__{arm}.jsonl"),
        }
        if args.replicate is not None:
            other = read_scores(args.replicate, arm)
            absent = [digest for digest, _ in members if digest not in other]
            if absent:
                fail(f"{arm}: the replicate run does not cover {len(absent)} folded attempts")
            worst = max(abs(scores[d]["mean_nll"] - other[d]["mean_nll"]) for d, _ in members)
            digests = [digest for digest, _ in members]
            entry["replicate"] = {
                "source": str(args.replicate / f"{arm}__on__{arm}.jsonl"),
                "source_sha256": sha256_file(args.replicate / f"{arm}__on__{arm}.jsonl"),
                "max_abs_diff_mean_nll": worst,
                "same_top_k": top_digests(scores, digests) == top_digests(other, digests),
                "k": ota.SELECTION_BUDGET,
            }
        per_arm[arm] = entry

    args.out.mkdir(parents=True, exist_ok=True)
    sidecar = args.out / "selector_score.jsonl"
    sidecar.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                       encoding="utf-8")
    record = {
        "status": "complete",
        "selector": SELECTOR,
        "join": "the sequence digest, under analyse_objective_tuned.digest_of: a row carrying "
                "sequence_sha256 and no sequence joins on the recorded digest",
        "producer_of_the_scores": "scripts/capability/generation/score_cohort_likelihood.py, "
                                 "own-model cells <arm>__on__<arm>.jsonl",
        "n_rows": len(rows),
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in
                   (("neighbour", args.neighbour), ("cohort", args.cohort))},
        "selector_score_jsonl": {"path": str(sidecar), "sha256": sha256_file(sidecar)},
        "per_arm": per_arm,
    }
    write_json(args.out / "selector_score.json", record)
    print(json.dumps({"rows": len(rows), "arms": len(per_arm),
                      "out": str(sidecar)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
