#!/usr/bin/env python3
"""Freeze the swept draws into one ledger and one folding cohort.

The generation cells write a ledger each, one per arm and configuration shard.
This stage joins them, checks that the sweep is *complete* -- every declared
arm, configuration, cluster and draw index present exactly once -- and emits the
two frozen artefacts everything downstream reads: the full ledger, which keeps
every attempt including the empty and out-of-band ones so the census
denominators stay honest, and the folding cohort, which carries only the
products inside the structural evaluation band.

Incompleteness is refused rather than reported. A sweep missing a cell would
otherwise be analysed as if the grid were smaller than it is, and a
configuration silently absent from a factorial is the one failure that looks
like a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.decoding import decoding_sweep as ds  # noqa: E402
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402

COMPLETION = "decoding_cohort.json"
DEFAULT_QUEUE = REPO_ROOT / "results/R6/conditioned_generation_queue_20260826/class_queue.json"
LEDGER = "sweep_ledger.jsonl"
FOLD_COHORT = "fold_cohort.jsonl"

#: The fields the folding and novelty stages read, plus the sweep coordinates
#: that let a folded row be attributed back to its configuration.
COHORT_FIELDS: tuple[str, ...] = (
    "id",
    "role",
    "arm",
    "config_key",
    "config_axis",
    "cluster",
    "draw_index",
    "sequence",
    "sequence_sha256",
    "length",
)


def expected_cells(arms: list[str], queue_path: Path) -> set[tuple[str, str, str, int]]:
    # The frozen queue is read only when a conditioned arm is present, so a sweep
    # over the unconditioned arm alone does not depend on it.
    queue = (
        ds.cg.load_queue(queue_path) if any(ds.arm(name).conditioned for name in arms) else None
    )
    wanted: set[tuple[str, str, str, int]] = set()
    for name in arms:
        clusters = ds.clusters_for(name, queue if ds.arm(name).conditioned else None)
        for setting in ds.GRID:
            for cluster, _ in clusters:
                for index in range(setting.draws_per_cluster):
                    wanted.add((name, setting.key, cluster, index))
    return wanted


def run(args: argparse.Namespace) -> dict[str, Any]:
    gp.require_fresh_out(args.out, COMPLETION)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in args.ledgers:
        block = ds.read_jsonl(path)
        sources.append({"path": str(path), "sha256": sha256_file(path), "rows": len(block)})
        rows.extend(block)
    identifiers = Counter(str(row["id"]) for row in rows)
    duplicated = sorted(key for key, count in identifiers.items() if count > 1)
    if duplicated:
        raise SystemExit(
            f"{len(duplicated)} candidate identifiers appear more than once across the "
            f"supplied ledgers, e.g. {duplicated[:5]}; two cells sampled the same cell"
        )
    arms = sorted({str(row["arm"]) for row in rows})
    unknown = sorted(set(arms) - set(ds.ARM_NAMES))
    if unknown:
        raise SystemExit(f"the ledgers carry arms this sweep does not declare: {unknown}")
    observed = {
        (str(row["arm"]), str(row["config_key"]), str(row["cluster"]), int(row["draw_index"]))
        for row in rows
    }
    wanted = expected_cells(arms, args.queue)
    missing = sorted(wanted - observed)
    extra = sorted(observed - wanted)
    if missing or extra:
        raise SystemExit(
            f"the sweep is not complete for arms {arms}: {len(missing)} declared draws are "
            f"missing (e.g. {missing[:3]}) and {len(extra)} undeclared draws are present "
            f"(e.g. {extra[:3]})"
        )

    rows.sort(key=lambda row: str(row["id"]))
    ledger_digest = ds.write_jsonl(args.out / LEDGER, rows)
    folded = [
        {field: row[field] for field in COHORT_FIELDS}
        for row in rows
        if row["in_band"]
    ]
    if not folded:
        raise SystemExit("no product of this sweep lies inside the structural evaluation band")
    cohort_digest = ds.write_jsonl(args.out / FOLD_COHORT, folded)

    census: dict[str, Any] = {}
    for name in arms:
        per_config = {}
        for setting in ds.GRID:
            cell = [
                row
                for row in rows
                if row["arm"] == name and row["config_key"] == setting.key
            ]
            per_config[setting.key] = ds.config_census(cell)
        census[name] = {
            "n_attempts": sum(1 for row in rows if row["arm"] == name),
            "n_in_band": sum(1 for row in rows if row["arm"] == name and row["in_band"]),
            "per_configuration": per_config,
        }

    record = {
        "schema_version": ds.SCHEMA_VERSION,
        "status": "complete",
        "campaign": ds.CAMPAIGN,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "arms": arms,
        "n_configurations": len(ds.GRID),
        "configurations": [setting.record() for setting in ds.GRID],
        "clusters_per_configuration": ds.CLUSTERS_PER_CONFIG,
        "evaluation_band": list(ds.EVALUATION_BAND),
        "n_attempts": len(rows),
        "n_in_band": len(folded),
        "n_distinct_in_band_sequences": len({row["sequence_sha256"] for row in folded}),
        "deduplicated_folds": (
            "exact duplicate sequences share one fold in the structure instrument and "
            "remain separate sampling units in every census and interval"
        ),
        "sources": sources,
        "sweep_ledger": str(args.out / LEDGER),
        "sweep_ledger_sha256": ledger_digest,
        "fold_cohort": str(args.out / FOLD_COHORT),
        "fold_cohort_sha256": cohort_digest,
        "census": census,
        "ceiling": dict(ds.CEILING),
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledgers", nargs="+", type=Path, required=True)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = run(args)
    print(
        json.dumps(
            {
                key: record[key]
                for key in ("status", "arms", "n_attempts", "n_in_band", "n_distinct_in_band_sequences")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
