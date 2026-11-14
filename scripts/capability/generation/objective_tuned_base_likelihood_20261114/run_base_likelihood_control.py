#!/usr/bin/env python3
"""The base-likelihood control of the 2026-11-14 amendment, over every pair the other controls cover.

Ranks each base pool by the base model's own likelihood of its own samples and
reads the share of the tuned-minus-base pLDDT gain the top quarter reproduces,
through the analysis module's own ``explicit_statistics_control``: the same
table builder, stream split, ranking scope, nested draws, floor and native band
as the stream-separated run of 2026-11-12, imported and called unchanged. Only
the ranking statistic and its two declared families differ.

The scores are read from the base arms' own-model cells of
``score_cohort_likelihood.py`` (nothing is scored here and no model is loaded).
Coverage is checked before any statistic: every folded base attempt must carry
a score, which ``score_pool`` enforces by refusing on the first gap. The
stored three-rule results are read for the comparison table and must agree with
this run's realised gains to printed precision, which ties the two runs to one
table.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/capability/generation"))

import analyse_objective_tuned as aot  # noqa: E402

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation import generative_control as gc  # noqa: E402
from src.capability.generation import objective_tuned_analysis as ota  # noqa: E402
from src.capability.generation import objective_tuned_arms as ot  # noqa: E402

_STREAMED = (REPO_ROOT / "scripts/capability/generation/objective_tuned_endpoints_20261112"
             / "run_stream_endpoints.py")


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


streamed = _load(_STREAMED, "objective_tuned_stream_endpoints")

#: Rule order in the comparison table: the three declared statistics, then the
#: declared likelihood rule, then its summed sensitivity.
COMPARED = (*ota.FILTER_RULE_NAMES, *ota.LIKELIHOOD_RULE_NAMES)


def cell_summary(column: dict[str, Any]) -> dict[str, Any]:
    interval = column.get("interval") or [None, None]
    finding = (column.get("statistics_reproduce_the_gain") or {}).get("finding")
    return {"status": column["status"], "share": column.get("point"),
            "lo": interval[0], "hi": interval[1], "finding": finding,
            "gain": (column.get("gains") or {}).get("tuned_minus_base_gain")}


def comparison_rows(stored: dict[str, dict[str, Any]], fresh: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (pair, stratum): the stored three rules beside the fresh two."""

    keys = list(next(iter(fresh["rules"].values()))["bands"]["columns"])
    rows = []
    for key in keys:
        pair, stratum = key.split("|")
        row: dict[str, Any] = {"pair": pair, "stratum": stratum}
        gains = []
        for name in COMPARED:
            source = stored[name] if name in stored else fresh["rules"][name]["bands"]["columns"]
            summary = cell_summary(source[key])
            if summary["gain"] is not None:
                gains.append(summary["gain"])
            for field in ("status", "share", "lo", "hi", "finding"):
                row[f"{name}_{field}"] = summary[field]
        if gains and max(gains) - min(gains) > 1e-6:
            raise SystemExit(f"{key}: the stored and fresh runs disagree on the tuned-minus-base "
                             f"gain ({gains}); they are not reading one table")
        row["tuned_minus_base_gain"] = gains[0] if gains else None
        rows.append(row)
    return rows


def stored_columns(record: dict[str, Any], block: str) -> dict[str, dict[str, Any]]:
    rules = record[block]["rules"]
    return {name: rules[name]["bands"]["columns"] for name in ota.FILTER_RULE_NAMES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neighbour", type=Path, required=True)
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--fold-structures", type=Path, default=None)
    parser.add_argument("--stream-labels", type=Path, required=True)
    parser.add_argument("--kmer-background", type=Path, required=True)
    parser.add_argument("--likelihood", type=Path, required=True,
                        help="directory holding <arm>__on__<arm>.jsonl for the four base arms")
    parser.add_argument("--stored-primary", type=Path, required=True,
                        help="objective_tuned_endpoints.json of the stream-separated run")
    parser.add_argument("--stored-secondary", type=Path, required=True,
                        help="progen2_attribution.json of the stream-separated run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base_arms = sorted({pair.base for pair in (*ota.PRIMARY_PAIRS, *ota.SECONDARY_PAIRS)})
    score_files = {arm: args.likelihood / f"{arm}__on__{arm}.jsonl" for arm in base_arms}
    for arm, path in score_files.items():
        if not path.is_file():
            raise SystemExit(f"{path} is missing; the own-model cell of {arm} is the base-likelihood input")
    likelihood = ota.merge_likelihood({arm: ota.read_own_likelihood(path)
                                       for arm, path in score_files.items()})

    one_stream, worst = streamed.pooled.build_table(args.neighbour, args.composition, args.cohort,
                                                    args.fold_structures)
    table = streamed.split_streams(one_stream, streamed.read_labels(args.stream_labels))
    counts = gc.load_kmer_counts(args.kmer_background, aot.KMER_ORDERS)

    shift = aot.identity_shift(table, ota.PRIMARY_PAIRS)
    primary = aot.stratified_contrast(table, ota.PRIMARY_PAIRS, ota.PRIMARY_FAMILY, shift=shift)
    secondary = aot.stratified_contrast(table, ota.SECONDARY_PAIRS, ota.SECONDARY_FAMILY)
    control_primary = aot.explicit_statistics_control(
        table, ota.PRIMARY_PAIRS, counts, primary, ota.LIKELIHOOD_CONTROL_FAMILIES, likelihood)
    control_secondary = aot.explicit_statistics_control(
        table, ota.SECONDARY_PAIRS, counts, secondary,
        ota.SECONDARY_LIKELIHOOD_CONTROL_FAMILIES, likelihood)

    coverage: dict[str, Any] = {}
    for arm in base_arms:
        digests = {row["digest"] for (a, _), pool in table["pools"].items() if a == arm
                   for row in pool["rows"]}
        coverage[arm] = {
            "n_scored_in_source": sum(1 for line in score_files[arm].read_text().splitlines()
                                      if line.strip()),
            "n_folded_attempts": sum(len(pool["rows"]) for (a, _), pool in table["pools"].items()
                                     if a == arm),
            "n_folded_distinct": len(digests),
            "n_folded_without_score": sum(1 for digest in digests if digest not in likelihood),
            "source_sha256": sha256_file(score_files[arm]),
        }
        if coverage[arm]["n_folded_without_score"]:
            raise SystemExit(f"{arm}: folded attempts without a base-model score: {coverage[arm]}")

    stored_p = json.loads(args.stored_primary.read_text(encoding="utf-8"))
    stored_s = json.loads(args.stored_secondary.read_text(encoding="utf-8"))
    rows = (comparison_rows(stored_columns(stored_p, "explicit_statistics_control"),
                            control_primary)
            + comparison_rows(stored_columns(stored_s, "explicit_statistics_control_secondary"),
                              control_secondary))

    record = {
        "schema": ota.SCHEMA, "status": "complete", "run": args.out.name,
        "registration": "docs/objective-alignment.md, nineteenth amendment of 2026-11-14",
        "campaign": ot.CAMPAIGN,
        "amendment": ota.LIKELIHOOD_CONTROL_AMENDMENT,
        "rule": ota.LIKELIHOOD_RULE_NAME, "sensitivity": ota.LIKELIHOOD_SENSITIVITY_NAME,
        "provenance": ota.LIKELIHOOD_PROVENANCE,
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in
                   (("neighbour", args.neighbour), ("composition", args.composition),
                    ("cohort", args.cohort), ("stream_labels", args.stream_labels),
                    ("stored_primary", args.stored_primary),
                    ("stored_secondary", args.stored_secondary),
                    *((("fold_structures", args.fold_structures),)
                      if args.fold_structures is not None else ()))},
        "cross_table_agreement": {"columns": worst},
        "coverage": coverage,
        "base_likelihood_control_primary": control_primary,
        "base_likelihood_control_secondary": control_secondary,
        "finding_states": ota.FINDING_STATES, "bounds": ota.BOUNDS,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "base_likelihood_control.json", record)
    tables = args.out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    streamed.write_tsv(tables / "base_likelihood_control.tsv",
                       streamed.control_rows(control_primary) + streamed.control_rows(control_secondary))
    streamed.write_tsv(tables / "comparison_with_statistical_filters.tsv", rows)
    print(json.dumps({"coverage": coverage,
                      "control_status": {family: {rule: block["bands"]["control_status"]
                                                  for rule, block in c["rules"].items()}
                                         for family, c in (("primary", control_primary),
                                                           ("secondary", control_secondary))}},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
