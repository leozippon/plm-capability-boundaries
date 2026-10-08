#!/usr/bin/env python3
"""Freeze the modelling tables for the small-domain stability predictor.

Three tables, built once and never rebuilt, so that training, validation and
application all read the same frozen rows:

``fit``
    the MGnify recalibrated absolute-dG release, restricted to its **own**
    declared ``train``/``validation``/``test`` folds. The dataset's split is used
    rather than a new one, so the held-out numbers are comparable to the published
    ones and no split is invented here. The 1.3 million rows the release leaves
    without a split assignment are deliberately unused, and counted.
``transfer``
    MegaScale ``dataset2``, an absolute-dG measurement over a different sequence
    population (natural and de novo designed domains rather than MGnify-derived
    ones), drawn stratified by wild-type cluster so that the cross-dataset check
    keeps its own grouping. Every wild-type row is kept; variant rows are capped
    per cluster. Sequences that also occur in the MGnify release are removed, so
    the check cannot be scored on anything the predictor could have seen.
``apply``
    the generated sequences of the frozen generation-evaluation cohort and their
    length-matched whole natural comparators. Nothing is drawn here: these are the
    rows E14 already froze, filtered to the length band, so the stability result
    sits on the same support as the structural one.

This stage reads two large local files and writes a compact table, which is what
travels to the cluster. It loads no model and needs no GPU; it accepts
``--device`` because the campaign queue injects it.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.evaluation import domain_stability as ds  # noqa: E402
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402

COMPLETION = "domain_stability_tables.json"
SCHEMA_VERSION = "d1_domain_stability_tables_v1"

#: The folds of the MGnify release this experiment uses, in the release's own
#: spelling. ``validation_online`` is a second validation stream and is kept for
#: alpha selection beside ``validation``; rows carrying no split are not used.
USED_SPLITS: tuple[str, ...] = ("train", "validation", "validation_online", "test")

#: The widest length band considered anywhere here. It is deliberately wider than
#: the predictor's licensed band, which the fit stage derives from the training
#: rows it actually sees, so the artefact records what was available as well as
#: what was used.
CANDIDATE_BAND: tuple[int, int] = (31, 90)

AA20 = set(gp.AA20)


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def iter_mgnify(path: Path) -> Iterator[dict[str, Any]]:
    csv.field_size_limit(10**7)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield row


def read_fit_table(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: collections.Counter = collections.Counter()
    seen: set[str] = set()
    for row in iter_mgnify(path):
        counts["rows_read"] += 1
        split = (row.get("split") or "").strip()
        if not split:
            counts["no_declared_split_unused"] += 1
            continue
        if split not in USED_SPLITS:
            counts[f"unknown_split_unused:{split}"] += 1
            continue
        sequence = (row.get("aa_seq") or "").strip().upper()
        if not sequence or set(sequence) - AA20:
            counts["non_canonical_dropped"] += 1
            continue
        if not CANDIDATE_BAND[0] <= len(sequence) <= CANDIDATE_BAND[1]:
            counts["outside_candidate_band_dropped"] += 1
            continue
        try:
            value = float(row["deltaG"])
        except (KeyError, TypeError, ValueError):
            counts["non_numeric_delta_g_dropped"] += 1
            continue
        if not np.isfinite(value):
            counts["non_finite_delta_g_dropped"] += 1
            continue
        if sequence in seen:
            counts["duplicate_sequence_dropped"] += 1
            continue
        seen.add(sequence)
        rows.append(
            {
                "id": "mg_" + _digest(sequence),
                "sequence": sequence,
                "length": len(sequence),
                "measured_delta_g": value,
                "split": split,
                "dataset": "mgnify_stability_cho2026",
                # Each row of this release is a distinct MGnify-derived domain
                # measured once, so the sequence is its own resampling unit.
                "group": "mg_" + _digest(sequence),
                "table": "fit",
            }
        )
        counts[f"kept:{split}"] += 1
    if not rows:
        raise SystemExit(f"{path} yielded no usable fit row")
    return rows, dict(sorted(counts.items()))


def read_transfer_table(
    directory: Path, *, exclude: set[str], cap_per_cluster: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise SystemExit(f"{directory} holds no parquet shard")
    table = pa.concat_tables(
        [
            pq.read_table(path, columns=["aa_seq", "deltaG", "mut_type", "WT_name", "WT_cluster"])
            for path in files
        ]
    )
    counts: collections.Counter = collections.Counter({"rows_read": table.num_rows})
    sequences = table.column("aa_seq").to_pylist()
    values = table.column("deltaG").to_pylist()
    mutations = table.column("mut_type").to_pylist()
    wild_types = table.column("WT_name").to_pylist()
    clusters = table.column("WT_cluster").to_pylist()
    by_cluster: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    seen: set[str] = set()
    for sequence, value, mutation, wild_type, cluster in zip(
        sequences, values, mutations, wild_types, clusters
    ):
        sequence = (sequence or "").strip().upper()
        if not sequence or set(sequence) - AA20:
            counts["non_canonical_dropped"] += 1
            continue
        if not CANDIDATE_BAND[0] <= len(sequence) <= CANDIDATE_BAND[1]:
            counts["outside_candidate_band_dropped"] += 1
            continue
        if value is None or not np.isfinite(value):
            counts["non_finite_delta_g_dropped"] += 1
            continue
        if sequence in seen:
            counts["duplicate_sequence_dropped"] += 1
            continue
        if sequence in exclude:
            # Present in the training release, so scoring it here would not be a
            # cross-dataset check at all.
            counts["shared_with_fit_release_dropped"] += 1
            continue
        seen.add(sequence)
        by_cluster[str(cluster)].append(
            {
                "id": "ms_" + _digest(sequence),
                "sequence": sequence,
                "length": len(sequence),
                "measured_delta_g": float(value),
                "split": "transfer",
                "dataset": "megascale_tsuboyama2023",
                "wild_type_name": str(wild_type),
                # Thousands of variants of one domain are not thousands of
                # independent units, so the wild-type cluster is the unit.
                "group": str(cluster),
                "is_wild_type": mutation == "wt",
                "table": "transfer",
            }
        )
    rows: list[dict[str, Any]] = []
    for index, (cluster, members) in enumerate(sorted(by_cluster.items())):
        wild = [row for row in members if row["is_wild_type"]]
        variants = [row for row in members if not row["is_wild_type"]]
        keep = list(wild)
        budget = max(0, cap_per_cluster - len(keep))
        if budget and variants:
            ordered = sorted(variants, key=lambda row: row["id"])
            if len(ordered) <= budget:
                keep.extend(ordered)
            else:
                rng = np.random.default_rng(seed + index)
                chosen = rng.choice(len(ordered), size=budget, replace=False)
                keep.extend(ordered[int(position)] for position in sorted(chosen))
        rows.extend(keep)
    if not rows:
        raise SystemExit(f"{directory} yielded no usable transfer row")
    counts["clusters"] = len(by_cluster)
    counts["kept"] = len(rows)
    counts["kept_wild_type"] = sum(1 for row in rows if row["is_wild_type"])
    return rows, dict(sorted(counts.items()))


def read_apply_table(cohort_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cohort_path = cohort_dir / "evaluation_cohort.jsonl"
    declaration = json.loads((cohort_dir / "e14_declaration.json").read_text(encoding="utf-8"))
    records = {
        json.loads(line)["id"]: json.loads(line)
        for line in cohort_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rows: list[dict[str, Any]] = []
    counts: collections.Counter = collections.Counter()
    for pair in declaration["pairs"]:
        for role, identifier in (
            ("generated", pair["generated_id"]),
            ("natural", pair["natural_id"]),
        ):
            record = records.get(identifier)
            if record is None:
                raise SystemExit(f"cohort {cohort_path} has no record for {identifier!r}")
            sequence = str(record["sequence"]).upper()
            if set(sequence) - AA20:
                counts["non_canonical_dropped"] += 1
                continue
            if not CANDIDATE_BAND[0] <= len(sequence) <= CANDIDATE_BAND[1]:
                counts["outside_candidate_band_dropped"] += 1
                continue
            rows.append(
                {
                    "id": identifier,
                    "sequence": sequence,
                    "length": len(sequence),
                    "measured_delta_g": None,
                    "split": "apply",
                    "dataset": "generation_evaluation_20261008",
                    "role": role,
                    "arm": pair["arm"],
                    "stratum": pair["stratum"],
                    "pair_id": pair["pair_id"],
                    "group": pair["pair_id"],
                    "table": "apply",
                }
            )
            counts[f"kept:{role}"] += 1
    if not rows:
        raise SystemExit(f"{cohort_dir} yielded no applicable row")
    return rows, dict(sorted(counts.items()))


def run(args: argparse.Namespace) -> dict[str, Any]:
    gp.require_fresh_out(args.out, COMPLETION)
    fit_rows, fit_counts = read_fit_table(args.mgnify)
    if args.max_train:
        # A seeded cap on the training fold only. The tuning and test folds are
        # never thinned, so a capped run is still validated on the release's whole
        # held-out support; what shrinks is the embedding cost. Recorded, because a
        # training-set size is part of how a predictor was obtained.
        train = [row for row in fit_rows if row["split"] == "train"]
        other = [row for row in fit_rows if row["split"] != "train"]
        if len(train) > args.max_train:
            rng = np.random.default_rng(args.seed + 17)
            chosen = rng.choice(len(train), size=args.max_train, replace=False)
            train = [train[int(position)] for position in sorted(chosen)]
            fit_counts["train_capped_to"] = args.max_train
        fit_rows = other + train
    fit_sequences = {row["sequence"] for row in fit_rows}
    transfer_rows, transfer_counts = read_transfer_table(
        args.megascale,
        exclude=fit_sequences,
        cap_per_cluster=args.transfer_cap_per_cluster,
        seed=args.seed,
    )
    apply_rows, apply_counts = read_apply_table(args.cohort_dir)

    identifiers: set[str] = set()
    rows: list[dict[str, Any]] = []
    for block in (fit_rows, transfer_rows, apply_rows):
        for row in block:
            if row["id"] in identifiers:
                raise SystemExit(f"identifier {row['id']!r} occurs in two tables")
            identifiers.add(row["id"])
            rows.append(row)
    rows.sort(key=lambda row: (row["table"], row["id"]))

    table_path = args.out / "domain_stability_rows.jsonl"
    table_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )

    by_split = collections.Counter(row["split"] for row in rows)
    lengths = {
        name: [row["length"] for row in rows if row["split"] == name] for name in sorted(by_split)
    }
    train_lengths = lengths.get("train", [])
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows_jsonl": str(table_path),
        "rows_sha256": sha256_file(table_path),
        "n_rows": len(rows),
        "n_by_split": dict(by_split),
        "length_by_split": {
            name: {
                "min": min(values),
                "max": max(values),
                "median": float(np.median(values)),
            }
            for name, values in lengths.items()
            if values
        },
        "band_the_training_fold_covers": list(ds.licensed_band(train_lengths))
        if train_lengths
        else None,
        "candidate_band": list(CANDIDATE_BAND),
        "sources": {
            "mgnify": {
                "path": str(args.mgnify),
                "accounting": fit_counts,
                "used_splits": list(USED_SPLITS),
                "note": (
                    "the release's own split is used. Rows carrying no split "
                    "assignment are counted and left unused rather than folded in "
                    "under a split invented here"
                ),
            },
            "megascale": {
                "path": str(args.megascale),
                "accounting": transfer_counts,
                "cap_per_cluster": int(args.transfer_cap_per_cluster),
                "resampling_unit": "WT_cluster",
            },
            "cohort": {"path": str(args.cohort_dir), "accounting": apply_counts},
        },
        "assay_independence": {
            "claim": (
                "the fit and transfer tables are different sequence populations, not "
                "different measurement technologies"
            ),
            "detail": (
                "both are cDNA-display proteolysis assays reporting a free energy "
                "derived from trypsin and chymotrypsin K50 values, and this project's "
                "own notes record that they share an author and a technology. "
                "Cross-dataset transfer therefore tests generalisation across sequence "
                "populations. Cross-technology corroboration has to come from a "
                "third-party instrument"
            ),
        },
        "quantities": dict(ds.QUANTITIES),
        "gate_conditions": dict(ds.GATE_CONDITIONS),
        "seed": int(args.seed),
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mgnify",
        type=Path,
        default=REPO_ROOT
        / "data/mgnify_stability_cho2026/230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv",
    )
    parser.add_argument(
        "--megascale",
        type=Path,
        default=REPO_ROOT / "data/megascale_tsuboyama2023/dataset2/data",
    )
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=REPO_ROOT / "results/R6/generation_evaluation_20261008",
    )
    parser.add_argument("--transfer-cap-per-cluster", type=int, default=200)
    parser.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="seeded cap on the training fold only; 0 keeps every training row",
    )
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.transfer_cap_per_cluster < 1:
        parser.error("--transfer-cap-per-cluster must be positive")
    if args.max_train < 0:
        parser.error("--max-train is non-negative; 0 means no cap")
    record = run(args)
    print(json.dumps({key: record[key] for key in ("status", "n_rows", "n_by_split")}, sort_keys=True))


if __name__ == "__main__":
    main()
