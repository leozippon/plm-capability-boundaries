#!/usr/bin/env python3
"""E18: the structural and computational phenotype of both ProLLaMA stages.

Does Stage 2's better mutation-effect prediction correspond to better generated
proteins? The frozen stage-pair set and its ESMFold2 folds already exist, so this
is analysis: nothing is generated and nothing is folded here.

The comparison rests on decoding symmetry, which
``freeze_stage_pair_sequences.py`` verified over fourteen declared manifest
fields -- prompt, condition, attempt count, token budget, temperature, top-p,
top-k, repetition penalty, batch size, dtype, cache use, special tokens, seed
rule -- and that verification travels into this artefact rather than being
asserted again.

Four contrasts, each labelled with what it estimates. The unadjusted contrast
measures what each stage emits; the two matched contrasts measure product quality
at equal length, and at equal length and hydrophobicity; the adjusted contrast
holds the measured covariates fixed. None is the corrected version of another,
and the overlap diagnostic says where the adjusted one would extrapolate.

Confidence, predicted stability and measurement are kept apart. Results from the
gated stability predictor, the interaction analysis and the homology-support
analysis are **integrated from a declaration file**, never recomputed, and the
band-admission diagnostic reports whether a licensed residue band samples the two
stages unequally -- because a stage difference inside a narrow band can be an
artefact of one stage being admitted to it more often.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import write_json  # noqa: E402
from src.capability.generation import stage_pair as sp  # noqa: E402
from src.capability.generation import stage_pair_phenotype as spp  # noqa: E402

COMPLETION = "stage_pair_phenotype.json"

#: Readouts taken from the sequence alone, available without any fold.
SEQUENCE_READOUTS: tuple[str, ...] = (
    "length",
    "mean_kyte_doolittle_hydropathy",
    "composition_entropy_nats",
    "distinct_residues",
    "longest_single_residue_run",
)


def _parse_ledger(value: str) -> tuple[str, str, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise SystemExit(f"--ledger takes stage:stream:path, got {value!r}")
    stage, stream, path = parts
    if stage not in sp.STAGES:
        raise SystemExit(f"unknown stage {stage!r}; declared: {sorted(sp.STAGES)}")
    return stage, stream, Path(path)


def _attach_termination(records: list[dict[str, Any]], ledgers: list[str]) -> dict[str, Any]:
    """Join each frozen record to its source attempt for the stop accounting."""

    source: dict[str, dict[str, Any]] = {}
    read: list[dict[str, Any]] = []
    for value in ledgers:
        _stage, _stream, path = _parse_ledger(value)
        rows = sp.read_jsonl(path)
        read.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "n_records": len(rows),
            }
        )
        for row in rows:
            identifier = row.get("id") or row.get("attempt_id")
            if identifier:
                source[str(identifier)] = row
    joined = 0
    for record in records:
        row = source.get(record["id"])
        if row is None:
            record["censored"] = None
            continue
        joined += 1
        tokens = row.get("generated_tokens")
        budget = row.get("effective_max_new_tokens")
        if tokens is not None and budget is not None:
            record["censored"] = int(tokens) >= int(budget)
        else:
            stop = row.get("decoder_stop") or row.get("stop_status")
            record["censored"] = (
                None if stop is None else str(stop) not in ("eos", "native_terminal", "stop_string")
            )
    return {"ledgers": read, "n_joined": joined, "n_records": len(records)}


def _flatten(record: dict[str, Any]) -> dict[str, Any]:
    """Lift the sequence properties onto the record so every readout is one key."""

    flat = dict(record)
    for name, value in (record.get("properties") or {}).items():
        if isinstance(value, (int, float)):
            flat.setdefault(name, value)
    return flat


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = [_flatten(row) for row in sp.read_jsonl(args.sequences)]
    freeze = json.loads(args.freeze.read_text(encoding="utf-8")) if args.freeze else None

    structural = {}
    structure_source: dict[str, Any] | None = None
    if args.structures_index is not None:
        rows = sp.read_jsonl(args.structures_index)
        structural = {str(row["id"]): row for row in rows}
        structure_source = {
            "path": str(args.structures_index),
            "sha256": hashlib.sha256(args.structures_index.read_bytes()).hexdigest(),
            "n_records": len(rows),
        }
        mismatched = [
            row["id"]
            for row in rows
            if row.get("sequence_sha256")
            and row["id"] in {record["id"] for record in records}
            and row["sequence_sha256"]
            != next(r["sequence_sha256"] for r in records if r["id"] == row["id"])
        ]
        if mismatched:
            raise SystemExit(
                f"{len(mismatched)} structure records describe a different sequence than "
                f"the frozen set, first {mismatched[:3]}; the two artefacts are not the "
                "same set"
            )
        for record in records:
            found = structural.get(record["id"])
            if found is None:
                continue
            for name in spp.CONFIDENCE_READOUTS:
                if found.get(name) is not None:
                    record[name] = found[name]

    termination_join = _attach_termination(records, args.ledger)

    by_stream = spp.split_stages(records)
    stage_1 = [record for record in records if record["stage"] == "stage_1"]
    stage_2 = [record for record in records if record["stage"] == "stage_2"]

    readouts = list(SEQUENCE_READOUTS)
    folded = [name for name in spp.CONFIDENCE_READOUTS if any(name in r for r in records)]
    readouts += folded

    contrasts: dict[str, Any] = {}
    for field in readouts:
        block: dict[str, Any] = {
            "readout_class": (
                "structural_confidence" if field in spp.CONFIDENCE_READOUTS else "sequence_property"
            ),
            "generation_distribution": spp.unmatched_contrast(records, field=field),
            "adjusted": spp.adjusted_contrast(records, field=field),
        }
        for design in sorted(spp.MATCH_DESIGNS):
            block[f"matched_{design}"] = spp.matched_contrast(
                by_stream, field=field, design=design,
                resamples=args.resamples, seed=args.bootstrap_seed,
            )
        contrasts[field] = block

    integration = None
    if args.integration is not None:
        integration = json.loads(args.integration.read_text(encoding="utf-8"))
        integration["declaration_sha256"] = hashlib.sha256(
            args.integration.read_bytes()
        ).hexdigest()
        integration["declaration_path"] = str(args.integration)
        bands = {}
        for name, block in (integration.get("licensed_bands") or {}).items():
            bands[name] = spp.band_admission(
                records, low=int(block["low"]), high=int(block["high"]), label=name
            )
        integration["band_admission"] = bands

    record = {
        "schema_version": spp.SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "question": (
            "does the improvement in mutation-effect prediction from Stage 1 to Stage 2 "
            "correspond to better biological properties in generated proteins?"
        ),
        "sequences": str(args.sequences),
        "sequences_sha256": hashlib.sha256(args.sequences.read_bytes()).hexdigest(),
        "n_sequences": len(records),
        "n_stage_1": len(stage_1),
        "n_stage_2": len(stage_2),
        "decoding_symmetry": (freeze or {}).get("decoding"),
        "decoding_symmetry_source": str(args.freeze) if args.freeze else None,
        "structure_source": structure_source,
        "n_with_structure": sum(1 for record in records if "mean_ca_plddt" in record),
        "folded_readouts": folded,
        "termination_join": termination_join,
        "termination": spp.termination_profile(records),
        "overlap": spp.overlap_diagnostics(stage_1, stage_2),
        "estimands": dict(spp.ESTIMANDS),
        "readout_classes": dict(spp.READOUT_CLASSES),
        "contrasts": contrasts,
        "integration": integration,
        "interfaces": {
            "compared": "the bare Seq=< prompt, which both checkpoints support",
            "not_compared": (
                "Stage 2's [Generate by superfamily] instruction interface, which Stage "
                "1 does not have; no Stage 1 conditional counterpart is constructed"
            ),
        },
        "ceiling": dict(spp.CEILING),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True,
                        help="stage_pair_sequences.jsonl of the frozen set")
    parser.add_argument("--freeze", type=Path, default=None,
                        help="stage_pair_freeze.json, whose verified decoding symmetry is cited")
    parser.add_argument("--structures-index", type=Path, default=None,
                        help="stage_pair_structure_index.jsonl from summarise_stage_pair_structures.py")
    parser.add_argument("--ledger", nargs="+", default=[],
                        help="stage:stream:path source ledgers, for the stop accounting")
    parser.add_argument("--integration", type=Path, default=None,
                        help="declaration file of results integrated from other experiments")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261008)
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = run(args)
    print(json.dumps({key: record[key] for key in ("status", "n_sequences", "n_with_structure",
                                                   "folded_readouts")}, sort_keys=True))


if __name__ == "__main__":
    main()
