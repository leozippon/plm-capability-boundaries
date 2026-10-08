#!/usr/bin/env python3
"""Freeze the ProLLaMA Stage 1 / Stage 2 generated sequence set, keyed for reuse.

Stage 2 is Stage 1 plus further training and is one of only two panel
checkpoints whose single-substitution stability increment resolves positive.
E18 asks whether that improvement in *prediction* comes with better *generation*.

The sets this stage freezes are the retained ones. The replication campaign
already declared both cells at identical prompt, sampling configuration, seed
rule and attempt count over the same campaign seeds, which is exactly the
symmetry the comparison depends on; the symmetry is verified here against the
manifest rather than assumed. Sampling a third stream would not be comparable
with the streams the manuscript's stage contrast reads, so nothing is generated.

Selection is outcome-blind by construction: identity, length and
canonical-residue validity only, under one declared seed, with exact duplicates
collapsed to one representative. No recognition rate, confidence value or
sequence property enters it.

The artefact is for three consumers. This experiment reads its sequence
properties and, after folding, its predicted-structure confidence; a separate
mutation-scanning analysis and a separate double-mutant interaction analysis read
the sequences. The key is the original generation attempt identifier, which is
unique across the whole programme, and every record carries its stage, arm,
stream and sequence digest so no join has to be reconstructed.
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

DEFAULT_MANIFEST = REPO_ROOT / "configs/generation_replication_manifest.json"
COMPLETION = "stage_pair_freeze.json"
RECORDS = "stage_pair_sequences.jsonl"


def _parse_ledger(value: str) -> tuple[str, str, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise SystemExit(
            f"--ledger takes stage:stream:path, got {value!r}; the stream label is "
            "explicit so a stream is never inferred from a path"
        )
    stage, stream, path = parts
    if stage not in sp.STAGES:
        raise SystemExit(f"unknown stage {stage!r}; declared: {sorted(sp.STAGES)}")
    return stage, stream, Path(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cells = {}
    for stage, arm in sp.STAGES.items():
        name = f"{arm}__unconditioned"
        cell = next((block for block in manifest["cells"] if block["cell"] == name), None)
        if cell is None:
            raise SystemExit(f"the manifest declares no cell {name!r}")
        cells[stage] = cell
    symmetry = sp.verify_symmetry(cells)
    symmetry["campaign_seeds"] = {
        block["id"]: int(block["seed"]) for block in manifest["campaigns"]
    }
    symmetry["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()

    ledgers = [_parse_ledger(value) for value in args.ledger]
    if not ledgers:
        raise SystemExit("at least one --ledger is required")
    seen_stages = {stage for stage, _, _ in ledgers}
    if seen_stages != set(sp.STAGES):
        raise SystemExit(
            f"ledgers were supplied for {sorted(seen_stages)} only; a one-sided set "
            "cannot answer a within-model stage comparison"
        )
    counts: dict[str, set[str]] = {}
    for stage, stream, _ in ledgers:
        counts.setdefault(stream, set()).add(stage)
    lopsided = sorted(stream for stream, stages in counts.items() if stages != set(sp.STAGES))
    if lopsided:
        raise SystemExit(
            f"these streams carry only one stage: {lopsided}. A stream measured for one "
            "stage only would weight the across-stream summary asymmetrically"
        )

    inputs: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for stage, stream, path in ledgers:
        rows = sp.normalise_ledger(sp.read_jsonl(path))
        inputs.append(
            {
                "stage": stage,
                "stream": stream,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "n_records": len(rows),
            }
        )
        blocks.append(
            sp.select_stream(
                rows,
                stage=stage,
                stream=stream,
                per_stream=args.per_stream,
                min_length=args.min_length,
                max_length=args.max_length,
                seed=args.selection_seed,
            )
        )
    frozen = sp.freeze(blocks, symmetry=symmetry, per_stream=args.per_stream)

    args.out.mkdir(parents=True, exist_ok=True)
    records = frozen.pop("records")
    path = args.out / RECORDS
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
    )
    contrasts = {
        field: sp.paired_property_contrast(
            [
                {
                    "stage": row["stage"],
                    "stream": row["stream"],
                    field: row["properties"][field],
                }
                for row in records
            ],
            field=field,
        )
        for field in (
            "length",
            "composition_entropy_nats",
            "longest_single_residue_run",
            "mean_kyte_doolittle_hydropathy",
        )
    }
    record = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **frozen,
        "inputs": inputs,
        "records_file": str(path),
        "records_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sequence_property_contrasts": contrasts,
        "no_new_sampling": (
            "the retained replication cells already declare identical decoding for both "
            "stages over the same campaign seeds; a new stream would not be comparable "
            "with the streams the manuscript's stage contrast reads, so none was sampled"
        ),
        "downstream": {
            "structures": "fold_generated_structures.py --sequences " + RECORDS,
            "recognition": "recognise_generated_families.py --sequences " + RECORDS,
            "mutation_scanning": (
                "not implemented here; consume " + RECORDS + " keyed on id, with stage "
                "and stream on every record"
            ),
        },
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", nargs="+", required=True,
                        help="stage:stream:path triples, e.g. stage_2:replicate_1:/path/attempts.jsonl")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--per-stream", type=int, default=sp.PER_STREAM)
    parser.add_argument("--min-length", type=int, default=sp.MIN_LENGTH)
    parser.add_argument("--max-length", type=int, default=sp.MAX_LENGTH)
    parser.add_argument("--selection-seed", type=int, default=sp.SELECTION_SEED)
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = run(args)
    print(json.dumps({"status": record["status"], "n_sequences": record["n_sequences"],
                      "per_stage": record["per_stage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
