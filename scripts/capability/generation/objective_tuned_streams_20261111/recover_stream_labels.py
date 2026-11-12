"""Recover the per-digest generation-cell label for the objective-tuned campaign, in-pod, read-only.

The R6 decision rule needs a stream label per folded structure (docs/objective-alignment.md),
and the per-structure fold table carries none. The label exists only in the campaign's 60
generation cell directories, named ot_<arm>_s<stream>_r<replicate>, each holding the
attempts.jsonl ledger that is also the lane's denominator. This script reads those ledgers
where they live, on cluster storage, and emits the (digest -> cell) mapping on stdout.

It writes nothing anywhere and reconstructs nothing. Every emitted label is a line of a
ledger whose sha256 matches the admission record that froze the campaign; a digest absent
from all 60 ledgers gets no label.

Three integrity conditions are checked and each one is fatal:
  - the two campaign records agree with each other on the 60 cell labels and on the frozen
    per-cell ledger digests;
  - every ledger's sha256 still equals the digest the admission record pinned;
  - every ledger row's recorded sequence_sha256 equals sha256 of its own sequence field.

Output is restricted to the folded shard sample (int(digest,16) % --num-shards in
range(--shards-done)) because that is the cohort the contrast reads; the summary counts
are over the whole campaign so coverage can be reported against both.

  python3 recover_stream_labels.py --run-dir <generation run> \
      --cohort-record <ot_fold_cohort/objective_tuned_fold_cohort.json> \
      --cells-record <ot_cell_verification/objective_tuned_cells.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

LEDGER_NAME = "attempts.jsonl"
LABEL_RE = re.compile(r"^ot_(?P<arm>.+)_s(?P<stream>\d+)_r(?P<replicate>\d+)$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--cohort-record", type=Path, required=True)
    ap.add_argument("--cells-record", type=Path, required=True)
    ap.add_argument("--num-shards", type=int, default=68)
    ap.add_argument("--shards-done", type=int, default=18)
    args = ap.parse_args(argv)

    cohort = json.loads(args.cohort_record.read_text(encoding="utf-8"))
    cells = json.loads(args.cells_record.read_text(encoding="utf-8"))
    if cohort.get("schema_version") != "objective_tuned_fold_cohort_v1":
        raise SystemExit(f"not a cohort record: {args.cohort_record}")
    if cells.get("schema_version") != "objective_tuned_cells_v1":
        raise SystemExit(f"not a cell admission record: {args.cells_record}")
    if not cells.get("complete"):
        raise SystemExit("the admission record does not admit the whole campaign")

    # The two records must agree before either is trusted as the source of a label.
    by_label = {entry["label"]: entry for entry in cohort["per_cell"]}
    admitted = {entry["label"]: entry for entry in cells["cells"]}
    if set(by_label) != set(admitted):
        raise SystemExit("cohort record and admission record name different cells")
    for label, entry in by_label.items():
        other = admitted[label]
        for key in ("arm", "role", "stream", "replicate", "attempts", "attempts_sha256"):
            if entry[key] != other[key]:
                raise SystemExit(f"{label}: records disagree on {key}")

    done = set(range(args.shards_done))
    # digest -> {label: attempts in that cell}
    mapping: dict[str, dict[str, int]] = {}
    all_digests: set[str] = set()
    per_cell = []
    for label in sorted(by_label):
        entry = by_label[label]
        match = LABEL_RE.match(label)
        if not match:
            raise SystemExit(f"cell directory name is not ot_<arm>_s<stream>_r<replicate>: {label}")
        if match.group("arm") != entry["arm"]:
            raise SystemExit(f"{label}: directory name arm disagrees with the record")
        if int(match.group("stream")) != entry["stream"] or int(match.group("replicate")) != entry["replicate"]:
            raise SystemExit(f"{label}: directory name stream/replicate disagrees with the record")
        ledger = args.run_dir / label / LEDGER_NAME
        digest = sha256_file(ledger)
        if digest != entry["attempts_sha256"]:
            raise SystemExit(f"{label}: {LEDGER_NAME} changed since it was admitted")
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != entry["attempts"]:
            raise SystemExit(f"{label}: {len(rows)} rows against {entry['attempts']}")
        empty = 0
        here: set[str] = set()
        for row in rows:
            if row["arm"] != entry["arm"]:
                raise SystemExit(f"{label}: a row carries arm {row['arm']!r}")
            sequence = row["sequence"]
            if not sequence:
                empty += 1
                continue
            key = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
            if key != row["sequence_sha256"]:
                raise SystemExit(f"{label}: recorded sequence_sha256 does not hash its own sequence")
            here.add(key)
            all_digests.add(key)
            if int(key, 16) % args.num_shards in done:
                mapping.setdefault(key, {})
                mapping[key][label] = mapping[key].get(label, 0) + 1
        if entry["attempts"] - empty != entry["non_empty"]:
            raise SystemExit(f"{label}: {entry['attempts'] - empty} non-empty rows against {entry['non_empty']}")
        per_cell.append(
            {
                "label": label,
                "arm": entry["arm"],
                "role": entry["role"],
                "stream": entry["stream"],
                "replicate": entry["replicate"],
                "attempts": entry["attempts"],
                "non_empty": entry["non_empty"],
                "unique_sequences": len(here),
                "unique_in_shard_sample": sum(1 for k in here if int(k, 16) % args.num_shards in done),
                "attempts_sha256": digest,
            }
        )

    if len(all_digests) != cohort["unique_sequences"]:
        raise SystemExit(
            f"{len(all_digests)} unique digests over the ledgers against {cohort['unique_sequences']} in the cohort record"
        )

    summary = {
        "campaign": cohort["campaign"],
        "cells": len(per_cell),
        "attempts_total": sum(e["attempts"] for e in per_cell),
        "non_empty_total": sum(e["non_empty"] for e in per_cell),
        "unique_sequences_campaign": len(all_digests),
        "unique_sequences_in_shard_sample": len(mapping),
        "num_shards": args.num_shards,
        "shards_done": args.shards_done,
        "cohort_record_sha256": sha256_file(args.cohort_record),
        "cells_record_sha256": sha256_file(args.cells_record),
        "cohort_sha256_from_record": cohort["cohort_sha256"],
        "code_sha256": sha256_file(Path(__file__)),
        "per_cell": per_cell,
    }
    out = sys.stdout
    out.write("#SUMMARY " + json.dumps(summary, sort_keys=True) + "\n")
    out.write("sequence_sha256\tlabel\tattempts\n")
    for key in sorted(mapping):
        for label in sorted(mapping[key]):
            out.write(f"{key}\t{label}\t{mapping[key][label]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
