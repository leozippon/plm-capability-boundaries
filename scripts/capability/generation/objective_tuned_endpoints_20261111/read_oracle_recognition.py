#!/usr/bin/env python3
"""Turn the Pfam/HMMER oracle's own sidecar into the two things this run needs.

The oracle is ``scripts/capability/generation/recognise_generated_families.py``
over the eighteen-shard sequences: HMMER 3.4 ``hmmscan`` against the pinned
pressed Pfam-A at the release's gathering thresholds, with the declared positive
control passing before a single generated sequence is scored. This file reads
that run's ``family_recognition.jsonl`` and writes

* ``recognition_flags.jsonl`` -- the ``--recognition`` shape
  ``analyse_objective_tuned`` declares: sequence digest, ``any_family``,
  ``complete_domain`` at the gate's own 80% model-length coverage. A rename of
  the oracle's own columns; no threshold is applied here.
* ``recognition_by_arm_and_stratum.tsv`` -- descriptive counts per arm over the
  whole folded sample and over the stratum below 30% identity: the two
  recognition endpoints, and how much of each arm's recognised mass falls in the
  seven Pfam families ``progen2-small-mix7`` was fine-tuned on. These are
  counts, not a declared endpoint: the declared recognition family is pooled
  over strata and carries the interval, and nothing here is banded.
* ``natural_reference_recognition.tsv`` -- optional, the recognition rates of the
  frozen natural-fragment draw under the identical oracle release, which is the
  reference the declaration's prediction points at. It is reported beside the
  endpoint and never merged into its family.
* ``novel_stratum_plddt_by_recognition.tsv`` -- optional, needing the per-structure
  pLDDT: within the stratum below 30% identity, each arm's mean pLDDT split by
  whether the oracle placed the sequence in the seven fine-tuning families, in
  some other family, or in none. Descriptive group means with their counts and no
  interval: the declared endpoints are the banded ones above, and this split is
  neither declared nor banded. It exists because the campaign asks whether the
  ProGen2 pair's novel-stratum pLDDT difference sits inside the fine-tuning
  domain, and that is a question about which sequences carry it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

#: The seven families the fine-tune was trained on, parsed from the arm
#: declaration's own objective string so the list has one source.
import sys

REPO_ROOT = Path("/Data/lzp/plm-capability-boundaries")
sys.path.insert(0, str(REPO_ROOT))

from src.capability.generation import objective_tuned_arms as ot  # noqa: E402

SEVEN = tuple(sorted(set(re.findall(r"PF\d{5}", ot.OBJECTIVE_PFAM_SEVEN_FAMILIES))))
NOVEL_EDGE = 30.0


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="family_recognition.jsonl from the oracle run")
    parser.add_argument("--neighbour", type=Path, required=True,
                        help="per-structure nearest-neighbour table, for identity over the query")
    parser.add_argument("--composition", type=Path, default=None,
                        help="per-structure composition table, for the descriptive pLDDT split")
    parser.add_argument("--natural-reference", type=Path, default=None,
                        help="a family_recognition directory whose sidecar carries role=natural "
                             "rows, scanned under the same pinned oracle")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if len(SEVEN) != 7:
        raise SystemExit(f"the arm declaration names {SEVEN}, not seven families")
    rows = read_jsonl(args.annotations)
    with args.neighbour.open(encoding="utf-8") as handle:
        identity = {row["sequence_sha256"]: float(row["identity_over_query"])
                    for row in csv.DictReader(handle, delimiter="\t")}
    if {row["sequence_sha256"] for row in rows} != set(identity):
        raise SystemExit("the oracle run and the neighbour table cover different digests")
    unsearched = [row["id"] for row in rows if row["profile_search_status"] != "searched"]
    if unsearched:
        raise SystemExit(f"{len(unsearched)} sequences were not searched, e.g. {unsearched[:3]}")

    args.out.mkdir(parents=True, exist_ok=True)
    flags = args.out / "recognition_flags.jsonl"
    with flags.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda row: row["sequence_sha256"]):
            handle.write(json.dumps({"sequence_sha256": row["sequence_sha256"],
                                     "any_family": bool(row["any_profile_hit"]),
                                     "complete_domain": bool(row["complete_domain"])},
                                    sort_keys=True) + "\n")

    out_rows: list[dict[str, Any]] = []
    families: dict[str, Counter] = {}
    for arm in sorted({row["arm"] for row in rows}):
        for label, keep in (("all", lambda value: True),
                            ("below_30_identity", lambda value: value < NOVEL_EDGE)):
            block = [row for row in rows
                     if row["arm"] == arm and keep(identity[row["sequence_sha256"]])]
            inside = [row for row in block
                      if set(row["pfam_families"]) & set(SEVEN)]
            outside = [row for row in block
                       if row["pfam_families"] and not set(row["pfam_families"]) & set(SEVEN)]
            counter: Counter = Counter()
            for row in block:
                counter.update(row["pfam_families"])
            families[f"{arm}|{label}"] = counter
            out_rows.append({
                "arm": arm, "support": label, "n": len(block),
                "n_any_family": sum(1 for row in block if row["any_profile_hit"]),
                "rate_any_family": (sum(1 for row in block if row["any_profile_hit"]) / len(block)
                                    if block else None),
                "n_complete_domain": sum(1 for row in block if row["complete_domain"]),
                "rate_complete_domain": (sum(1 for row in block if row["complete_domain"])
                                         / len(block) if block else None),
                "n_in_seven_finetuning_families": len(inside),
                "rate_in_seven_finetuning_families": len(inside) / len(block) if block else None,
                "n_recognised_outside_the_seven": len(outside),
                "n_distinct_families": len(counter),
                "top_families": "; ".join(f"{name}:{count}" for name, count in
                                          counter.most_common(8)),
                "thin_under_30": len(block) < 30,
            })
    write_tsv(args.out / "recognition_by_arm_and_stratum.tsv", out_rows)

    if args.composition is not None:
        with args.composition.open(encoding="utf-8") as handle:
            plddt = {row["sequence_sha256"]: float(row["plddt"])
                     for row in csv.DictReader(handle, delimiter="\t")}
        if set(plddt) != set(identity):
            raise SystemExit("the composition table and the neighbour table cover different "
                             "digests")
        split: list[dict[str, Any]] = []
        for arm in sorted({row["arm"] for row in rows}):
            block = [row for row in rows if row["arm"] == arm
                     and identity[row["sequence_sha256"]] < NOVEL_EDGE]
            groups: dict[str, list[float]] = {"in_seven_finetuning_families": [],
                                              "recognised_outside_the_seven": [],
                                              "recognised_by_nothing": []}
            for row in block:
                found = set(row["pfam_families"])
                name = ("in_seven_finetuning_families" if found & set(SEVEN)
                        else "recognised_outside_the_seven" if found
                        else "recognised_by_nothing")
                groups[name].append(plddt[row["sequence_sha256"]])
            for name, values in groups.items():
                split.append({
                    "arm": arm, "group": name, "n": len(values),
                    "share_of_stratum": len(values) / len(block) if block else None,
                    "mean_plddt": (sum(values) / len(values)) if values else None,
                    "thin_under_30": len(values) < 30,
                })
        write_tsv(args.out / "novel_stratum_plddt_by_recognition.tsv", split)

    reference: dict[str, Any] | None = None
    if args.natural_reference is not None:
        record = json.loads((args.natural_reference / "family_recognition.json")
                            .read_text(encoding="utf-8"))
        theirs = record["oracle"]["digests"]
        mine = json.loads((args.annotations.parent / "family_recognition.json")
                          .read_text(encoding="utf-8"))["oracle"]["digests"]
        if theirs != mine:
            raise SystemExit("the natural reference was scanned under a different oracle release; "
                             "it is not the same instrument and is not reported beside this run")
        natural = [row for row in read_jsonl(args.natural_reference / "family_recognition.jsonl")
                   if row.get("role") == "natural"]
        if not natural:
            raise SystemExit(f"{args.natural_reference} carries no role=natural row")
        lengths = sorted(int(row["length"]) for row in natural)
        reference = {
            "arm": "natural_fragments", "support": "frozen reference draw", "n": len(natural),
            "n_any_family": sum(1 for row in natural if row["any_profile_hit"]),
            "rate_any_family": sum(1 for row in natural if row["any_profile_hit"]) / len(natural),
            "n_complete_domain": sum(1 for row in natural if row["complete_domain"]),
            "rate_complete_domain": sum(1 for row in natural if row["complete_domain"])
            / len(natural),
            "length_min": lengths[0], "length_median": lengths[len(lengths) // 2],
            "length_max": lengths[-1],
            "source": str(args.natural_reference),
            "note": "the same pinned oracle release, bytes checked. A different cohort and a "
                    "different length distribution from this campaign's, so it is a reference "
                    "level reported beside the endpoint and never a column of its family",
        }
        write_tsv(args.out / "natural_reference_recognition.tsv", [reference])

    print(json.dumps({"seven_finetuning_families": list(SEVEN),
                      "recognition_flags": str(flags),
                      "natural_reference": reference,
                      "rows": out_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
