#!/usr/bin/env python3
"""Adapt E18's frozen ProLLaMA stage pair into the pool shape the likelihood scorer reads.

E18 froze, per ProLLaMA stage, 900 generated sequences over three campaign
streams, and the two stages share one unconditional rendering family: they differ
only by further training. That is what makes a *cross-arm* likelihood meaningful
here -- the same frozen members can be scored by either checkpoint through the
same ``Seq=<...>`` block, and the difference is the checkpoint rather than the
rendering. This adapter writes the cohort that lets the scorer do it, so the
2x2 of {scorer stage 1, scorer stage 2} x {stage 1 pool, stage 2 pool} reads one
frozen pool per generating stage.

Nothing is resampled and nothing is filtered. Every declared frozen record is
emitted once, with its sequence bytes, identifier, stage, stream, sequence digest
and sequence properties preserved; only the ``roles`` marker and the declared
unconditional condition are added. Selection was already outcome-blind when E18
froze the set, and re-selecting here would make the pool a second, undocumented
draw.

The pool does not depend on the scorer. Each generating arm gets one digest, and
all four cells of the 2x2 record the same digest for a given generator, because a
scorer changes the likelihood values and never the members.

Only the shared unconditional interface is adapted. Stage 2 carries an
instruction interface that stage 1 does not have, so a conditioned stage-1
counterpart does not exist; the adapter refuses a freeze that declares anything
other than the unconditional pair rather than inventing the missing cell.

The count expected of the cohort is read from the freeze's own declared
per-stream and per-stage counts. Size is a property of the artefact, never a
constant here that a fixture could satisfy while the real set did not.
"""

from __future__ import annotations

import argparse
import hashlib
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
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402
from src.capability.generation import stage_pair as sp  # noqa: E402

COMPLETION = "stage_cross_scoring_cohort.json"
COHORT = "stage_cross_scoring_cohort.jsonl"
SCHEMA_VERSION = "d1_stage_cross_scoring_cohort_v1"

#: Every field a frozen record must carry to be reusable as a pool member. The
#: properties block is required rather than optional because it is part of what
#: the frozen set promises to later analyses and would otherwise be dropped here.
REQUIRED_FIELDS = (
    "id",
    "stage",
    "arm",
    "stream",
    "sequence",
    "sequence_sha256",
    "length",
    "properties",
)


def read_freeze(path: Path, sequences: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """The verified freeze and its declared decoding, or a refusal.

    The freeze's own ``records_sha256`` is the staleness check: a sequences file
    that is not byte-for-byte the one the freeze describes is a different pool,
    and adapting it would attach the freeze's provenance to records it never
    governed.
    """

    try:
        freeze = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{path}: not a readable JSON freeze ({error})") from error
    if freeze.get("status") != "complete":
        raise SystemExit(f"{path}: status is {freeze.get('status')!r}, not 'complete'")
    if freeze.get("schema_version") != sp.SCHEMA_VERSION:
        raise SystemExit(
            f"{path}: schema_version is {freeze.get('schema_version')!r}, not "
            f"{sp.SCHEMA_VERSION!r}; a different artefact shape is not adapted silently"
        )
    decoding = freeze.get("decoding") or {}
    declared = decoding.get("declared") or {}
    if declared.get("condition") != "unconditioned":
        raise SystemExit(
            f"{path}: the frozen pair declares condition {declared.get('condition')!r}. "
            "Only the shared unconditional stage pair has a rendering both stages can "
            "be scored under, and stage 1 has no conditioned form, so this freeze is "
            "refused rather than adapted"
        )
    if decoding.get("arms") != sp.STAGES:
        raise SystemExit(
            f"{path}: the frozen arms are {decoding.get('arms')!r}, not the declared "
            f"pair {sp.STAGES!r}. An arm whose stage is not the one it declares cannot "
            "be pooled under that stage"
        )
    if decoding.get("symmetric") is not True:
        raise SystemExit(
            f"{path}: decoding symmetry is {decoding.get('symmetric')!r}, not verified "
            "true; a difference between the stages would then be partly a decoding "
            "difference and the cross-arm score would not be what it is named"
        )
    observed = sha256_file(sequences)
    if freeze.get("records_sha256") != observed:
        raise SystemExit(
            f"{sequences}: sha256 {observed} does not match the digest "
            f"{freeze.get('records_sha256')!r} the freeze records. The frozen set was "
            "edited or a different file was supplied; a pool that is not the declared "
            "one is refused"
        )
    return freeze, declared, dict(decoding["arms"])


def declared_counts(freeze: dict[str, Any]) -> dict[tuple[str, str], int]:
    """The freeze's own per-(stage, stream) realised counts, cross-checked.

    ``n_selected`` is used rather than the per-stream target so a registered
    shortfall is reproduced rather than overridden, and the stage and total
    declarations must agree with the sum before any record is read.
    """

    blocks = freeze.get("per_stream")
    if not isinstance(blocks, list) or not blocks:
        raise SystemExit("the freeze declares no per-stream block; nothing to adapt")
    counts: dict[tuple[str, str], int] = {}
    for block in blocks:
        key = (str(block.get("stage")), str(block.get("stream")))
        if key in counts:
            raise SystemExit(f"the freeze declares the stream {key} twice")
        if key[0] not in sp.STAGES:
            raise SystemExit(f"the freeze declares an unknown stage {key[0]!r}")
        counts[key] = int(block["n_selected"])
    per_stage = freeze.get("per_stage") or {}
    if set(per_stage) != set(sp.STAGES):
        raise SystemExit(
            f"the freeze declares per-stage counts for {sorted(per_stage)}, not "
            f"{sorted(sp.STAGES)}"
        )
    for stage, block in sorted(per_stage.items()):
        observed = sum(count for (stream_stage, _), count in counts.items() if stream_stage == stage)
        if int(block.get("n_sequences", -1)) != observed:
            raise SystemExit(
                f"{stage}: the freeze declares {block.get('n_sequences')} sequences and "
                f"its streams sum to {observed}"
            )
    total = sum(counts.values())
    if int(freeze.get("n_sequences", -1)) != total:
        raise SystemExit(
            f"the freeze declares {freeze.get('n_sequences')} sequences and its streams "
            f"sum to {total}"
        )
    return counts


def read_pool_records(
    path: Path, counts: dict[tuple[str, str], int], arms: dict[str, str]
) -> list[dict[str, Any]]:
    """Read every declared record, refusing anything the freeze did not declare."""

    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SystemExit(f"{path}: not a readable sequences file ({error})") from error
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{number}: {error}") from error
    if not rows:
        raise SystemExit(f"{path} carries no record; an empty pool is refused")

    identifiers = [str(row.get("id")) for row in rows]
    duplicated = sorted(identifier for identifier, count in Counter(identifiers).items() if count > 1)
    if duplicated:
        raise SystemExit(
            f"{path} repeats {len(duplicated)} identifier(s) (first {duplicated[:3]}); a "
            "pool is keyed on the attempt identifier and a repeat is not a member"
        )

    for row in rows:
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise SystemExit(f"{path}: a record lacks {missing}")
        identifier = str(row["id"])
        stage = str(row["stage"])
        if stage not in sp.STAGES:
            raise SystemExit(f"{identifier}: unknown stage {stage!r}")
        if str(row["arm"]) != arms[stage]:
            raise SystemExit(
                f"{identifier}: stage {stage} declares arm {arms[stage]!r} and the record "
                f"carries {row['arm']!r}; stage and arm disagree"
            )
        sequence = row["sequence"]
        if not isinstance(sequence, str) or not sequence:
            raise SystemExit(f"{identifier}: the record carries no sequence")
        observed = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        if row["sequence_sha256"] != observed:
            raise SystemExit(
                f"{identifier}: sequence_sha256 {row['sequence_sha256']!r} does not match "
                f"the sequence it names ({observed}); the frozen bytes were edited"
            )
        if int(row["length"]) != len(sequence):
            raise SystemExit(
                f"{identifier}: declared length {row['length']} and sequence length "
                f"{len(sequence)} disagree"
            )
        if not isinstance(row["properties"], dict) or not row["properties"]:
            raise SystemExit(f"{identifier}: the record carries no sequence properties")
        # A conditioning label would mean this pool was not drawn under the shared
        # unconditional interface. It is refused rather than scored as if native.
        if row.get("class_key") not in (None, ""):
            raise SystemExit(
                f"{identifier}: the record carries class_key {row['class_key']!r}; a "
                "conditioned pool has no unconditional stage-1 counterpart"
            )
        if row.get("condition") not in (None, "unconditioned"):
            raise SystemExit(
                f"{identifier}: the record declares condition {row['condition']!r}, not "
                "the unconditional interface the stage pair shares"
            )

    observed_counts = dict(Counter((str(row["stage"]), str(row["stream"])) for row in rows))
    if observed_counts != counts:
        extra = sorted(set(observed_counts) - set(counts))
        absent = sorted(set(counts) - set(observed_counts))
        mismatched = sorted(
            key for key in set(observed_counts) & set(counts) if observed_counts[key] != counts[key]
        )
        raise SystemExit(
            f"{path} does not carry the declared frozen set: extra streams {extra}, "
            f"absent streams {absent}, and {mismatched} with a count other than the "
            "declared one. The pool is not resampled or topped up here"
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    gp.require_fresh_out(args.out, COMPLETION)
    freeze, declared, arms = read_freeze(args.freeze, args.sequences)
    counts = declared_counts(freeze)
    rows = read_pool_records(args.sequences, counts, arms)

    # The adaptation adds only what the pool shape needs. Everything else travels
    # exactly as the freeze wrote it.
    cohort = [{**row, "roles": ["pool"], "condition": "unconditioned"} for row in rows]

    pools: dict[str, Any] = {}
    for stage, arm in sorted(arms.items()):
        members = sorted((row for row in cohort if row["arm"] == arm), key=lambda row: str(row["id"]))
        by_stream = Counter(str(row["stream"]) for row in members)
        pools[arm] = {
            "stage": stage,
            "condition": "unconditioned",
            "n_members": len(members),
            "n_streams": len(by_stream),
            "n_by_stream": dict(sorted(by_stream.items())),
            "candidate_key": "id",
            "independence_unit": "stream",
            "independence_limitation": (
                "campaign streams are the primary replication unit, not individual "
                "products or independently trained model lineages; family and duplicate "
                "dependence require separate sensitivity checks"
            ),
            "pool_digest": gp.cohort_digest(members),
        }

    scorers = sorted(arms.values())
    cells = [
        {
            "generator_arm": arm,
            "scorer_arm": scorer,
            "cross_arm": scorer != arm,
            "pool_digest": pools[arm]["pool_digest"],
        }
        for arm in sorted(pools)
        for scorer in scorers
    ]

    path = args.out / COHORT
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in cohort), encoding="utf-8"
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cohort_jsonl": str(path),
        "cohort_sha256": sha256_file(path),
        "n_records": len(cohort),
        "n_by_stage": dict(Counter(str(row["stage"]) for row in cohort)),
        "generator_arms": dict(sorted(arms.items())),
        "pools": pools,
        "cells": cells,
        "cell_note": (
            "the pool is keyed by its generating arm only, so both scorers read "
            "byte-identical members and the same pool_digest; the scorer changes the "
            "likelihood values and never the pool"
        ),
        "provenance": {
            "freeze": {
                "path": str(args.freeze),
                "sha256": sha256_file(args.freeze),
                "schema_version": freeze.get("schema_version"),
                "created_utc": freeze.get("created_utc"),
                "manifest_sha256": (freeze.get("decoding") or {}).get("manifest_sha256"),
            },
            "sequences": {
                "path": str(args.sequences),
                "sha256": sha256_file(args.sequences),
                "records_file_declared": freeze.get("records_file"),
                "records_sha256_declared": freeze.get("records_sha256"),
            },
            "generation_inputs": freeze.get("inputs"),
            "decoding": declared,
            "selection": {
                "selection_seed": next(
                    (block.get("selection_seed") for block in freeze.get("per_stream") or []),
                    None,
                ),
                "per_stream_target": freeze.get("per_stream_target"),
                "length_band_residues": next(
                    (block.get("length_band_residues") for block in freeze.get("per_stream") or []),
                    None,
                ),
                "no_new_sampling": freeze.get("no_new_sampling"),
            },
            "ceiling": freeze.get("ceiling"),
        },
        "conservation": {
            "resampling": "none; every declared frozen record is emitted exactly once",
            "outcome_filtering": (
                "none; no property, stop reason or score was read to select a record"
            ),
            "sequence_bytes": (
                "preserved verbatim, with id, stage, stream, digest and properties; only "
                "the pool role and the declared condition are added"
            ),
        },
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True, help="stage_pair_freeze.json")
    parser.add_argument(
        "--sequences", type=Path, required=True, help="stage_pair_sequences.jsonl"
    )
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = run(args)
    print(
        json.dumps(
            {
                "status": record["status"],
                "n_records": record["n_records"],
                "pools": {
                    arm: pool["pool_digest"] for arm, pool in sorted(record["pools"].items())
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()