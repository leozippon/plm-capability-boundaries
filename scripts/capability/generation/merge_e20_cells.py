#!/usr/bin/env python3
"""Join the per-cell outputs of an E20 sweep into the single inputs the later stages read.

The guided-decoding runner writes one directory per (model, domain, arm, strength) cell and
``run_structure_evidence.py`` one tree per fold shard, while ``build_e20_fold_cohort.py`` and
``measure_fold_recapitulation.py`` each take one records file and one structure tree. Two modes:

``--cells DIR...``
    Concatenate the cells' ``sequences.jsonl`` into ``records.jsonl``. Every cell must carry a
    complete ``guided_decoding_pool.json`` whose sequence digest matches, all cells must share one
    protocol digest and distinct stream seeds, and the joined records must satisfy the
    ``e20_generated_sequence_v1`` contract. With ``--protocol`` and ``--role`` a domain outside the
    named role is refused.
``--shards DIR...``
    Link every shard's ``objects/<sha>`` and copy its worker receipts into one tree. Every shard
    must be complete, no object may appear in two shards, and the tree must carry one signature.

Flat parser: the campaign queue prepends ``--device`` and ``--out``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation import fold_recapitulation as fr  # noqa: E402
from src.capability.generation import structure_evidence as se  # noqa: E402

RECORDS_COMPLETION = "e20_merged_records.json"
TREE_COMPLETION = "e20_merged_structures.json"


def merge_records(cells: list[Path], out: Path, wild_types: dict[str, str], roles: dict[str, str] | None, role: str | None) -> dict:
    lines: list[str] = []
    protocols, seeds, tally = set(), [], Counter()
    for cell in cells:
        completion = json.loads((cell / "guided_decoding_pool.json").read_text())
        if completion.get("status") != "complete":
            raise ValueError(f"{cell}: cell is not complete")
        sequences = cell / completion["artifacts"]["sequences"]["file"]
        if sha256_file(sequences) != completion["artifacts"]["sequences"]["sha256"]:
            raise ValueError(f"{cell}: sequences.jsonl differs from its completion record")
        protocols.add(completion["protocol"]["sha256"])
        info = completion["cell"]
        seeds.append(info["stream_seed"])
        tally[(info["model_arm"], info["domain"], info["arm_code"], info["lambda_fields"], info["lambda_coupling"])] += 1
        if roles is not None and roles[fr.domain_id(info["domain"])] != role:
            raise ValueError(f"{cell}: domain {info['domain']} is not a {role} domain")
        lines.extend(line + "\n" for line in sequences.read_text().splitlines() if line.strip())
    if len(protocols) != 1:
        raise ValueError(f"cells carry {len(protocols)} different protocol digests")
    if len(set(seeds)) != len(seeds):
        raise ValueError("two cells share a stream seed")
    duplicated = [key for key, count in tally.items() if count > 1]
    if duplicated:
        raise ValueError(f"cells repeated: {duplicated[:3]}")
    per_group = Counter((m, d) for m, d, *_ in tally)
    out.mkdir(parents=True, exist_ok=True)
    records = out / "records.jsonl"
    records.write_text("".join(lines))
    validated = fr.load_records(records, wild_types)
    write_json(out / RECORDS_COMPLETION, {
        "schema_version": "e20_merged_records_v1", "protocol_sha256": protocols.pop(),
        "cells": len(cells), "records": len(validated), "records_sha256": sha256_file(records),
        "settings_per_model_domain": {f"{m}|{d}": n for (m, d), n in sorted(per_group.items())},
    })
    return {"cells": len(cells), "records": len(validated)}


def merge_shards(shards: list[Path], out: Path) -> dict:
    objects = out / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    for shard in shards:
        summary = json.loads((shard / "structure_evidence.json").read_text())
        if summary["status_counts"].get("pending", 0):
            raise ValueError(f"{shard}: shard still has pending rows")
        for receipt in sorted(shard.glob("worker-*.json")):
            shutil.copy2(receipt, out / receipt.name)
        for source in sorted((shard / "objects").iterdir()):
            target = objects / source.name
            if target.exists() or target.is_symlink():
                raise ValueError(f"object {source.name} appears in two shards")
            target.symlink_to(source.resolve(), target_is_directory=True)
            counts[json.loads((source / "result.json").read_text())["status"]] += 1
    signature = se.tree_signature(out)
    write_json(out / TREE_COMPLETION, {
        "schema_version": "e20_merged_structures_v1", "evaluation_signature": signature,
        "shards": len(shards), "objects": sum(counts.values()), "status_counts": dict(counts),
    })
    return {"shards": len(shards), "objects": sum(counts.values()), "status_counts": dict(counts)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="accepted and unused; CPU-only")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cells", type=Path, nargs="+")
    mode.add_argument("--shards", type=Path, nargs="+")
    parser.add_argument("--wild-type-fasta", type=Path, help="query_12.faa; required with --cells")
    parser.add_argument("--protocol", type=Path, help="diagnostic_protocol.json; with --role")
    parser.add_argument("--role", choices=("development", "evaluation"))
    args = parser.parse_args(argv)
    if args.cells:
        if args.wild_type_fasta is None or bool(args.protocol) != bool(args.role):
            parser.error("--cells needs --wild-type-fasta, and --protocol together with --role")
        roles = fr.load_roles(args.protocol) if args.protocol else None
        summary = merge_records(args.cells, args.out, fr.load_wild_types(args.wild_type_fasta), roles, args.role)
    else:
        summary = merge_shards(args.shards, args.out)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
