#!/usr/bin/env python3
"""Join the per-cell outputs of an E20 sweep into the single inputs the later stages read.

The guided-decoding runner writes one directory per (model, domain, arm, strength) cell and
``run_structure_evidence.py`` one tree per fold shard, while ``build_e20_fold_cohort.py`` and
``measure_fold_recapitulation.py`` each take one records file and one structure tree. Two modes:

``--cells PATH...``
    Each PATH is a cell directory or a parent directory whose children are cells; a cell is a
    directory carrying ``guided_decoding_pool.json`` (other children are ignored). Concatenate the cells' ``sequences.jsonl`` into ``records.jsonl``. Every cell must carry a
    complete ``guided_decoding_pool.json`` whose sequence digest matches, all cells must share
    one sampling recipe, pool size and native contract (they may come from several protocols, whose
    digests are recorded) and carry distinct stream seeds, and the joined records must satisfy the
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

SCALE = "component_scale.json"
RECORDS_COMPLETION = "e20_merged_records.json"
TREE_COMPLETION = "e20_merged_structures.json"


def discover_cells(paths: list[Path]) -> list[Path]:
    """Expand parent directories into the cell directories beneath them (sorted, unique)."""

    cells: list[Path] = []
    for path in paths:
        if (path / "guided_decoding_pool.json").is_file():
            cells.append(path)
        elif path.is_dir():
            cells.extend(sorted(c for c in path.iterdir() if (c / "guided_decoding_pool.json").is_file()))
        else:
            raise FileNotFoundError(f"--cells path {path} is not a directory")
    if not cells:
        raise ValueError(f"no cell (directory with guided_decoding_pool.json) under {[str(p) for p in paths]}")
    return list(dict.fromkeys(cells))


def require_declared_depth(cell: Path, completion: dict, n_records: int, strengths) -> None:
    """The cell holds exactly the depth its own declaration names, not a depth shared with other cells.

    The declaration is the protocol's ``strengths`` entry for the cell's (model, arm, strength pair) and must
    carry a ``pool_size``. Arm H is limited by its admissible rows, so it must hold
    ``min(pool_size, admissible_rows)`` of the cell's own manifest.
    """

    info = completion["cell"]
    key = (info["model_arm"], info["arm_code"])
    pair = (info["lambda_fields"], info["lambda_coupling"])
    declaration = fr.parse_strength_declaration(strengths)
    if key not in declaration:
        raise ValueError(f"{cell}: (model, arm) {key} has no declaration in protocol.strengths")
    size = declaration[key][2].get(pair)
    if size is None:
        raise ValueError(f"{cell}: strength {pair} of {key} has no declared pool_size in protocol.strengths")
    expected = size
    if info["arm_code"] == "H":
        expected = min(size, completion["reference_arm"]["sampling"]["admissible_rows"])
    if n_records != expected or info["sequences_per_cell"] != n_records:
        raise ValueError(f"{cell}: holds {n_records} records (completion says {info['sequences_per_cell']}) but its "
                         f"declaration names a depth of {expected} for {key} at {pair}")


def merge_records(cells: list[Path], out: Path, wild_types: dict[str, str], roles: dict[str, str] | None, role: str | None,
                  expect_cells: int | None = None, strengths=None) -> dict:
    cells = discover_cells(cells)
    if expect_cells is not None and len(cells) != expect_cells:
        raise ValueError(f"found {len(cells)} complete cells, expected {expect_cells}; a failed cell is not discovered")
    lines: list[str] = []
    scale: list[dict] = []
    protocols, recipes, seeds, tally = set(), set(), [], Counter()
    for cell in cells:
        completion = json.loads((cell / "guided_decoding_pool.json").read_text())
        if completion.get("status") != "complete":
            raise ValueError(f"{cell}: cell is not complete")
        sequences = cell / completion["artifacts"]["sequences"]["file"]
        if sha256_file(sequences) != completion["artifacts"]["sequences"]["sha256"]:
            raise ValueError(f"{cell}: sequences.jsonl differs from its completion record")
        protocols.add(completion["protocol"]["sha256"])
        info = completion["cell"]
        if info["arm_code"] not in fr.MODEL_FREE_ARMS:  # reference arms have no sampling recipe to share
            # With a strengths declaration depth is checked per cell below; without one it must be uniform.
            depth = None if strengths is not None else info["sequences_per_cell"]
            recipes.add(json.dumps([completion["sampling"], completion["native_contract"], depth,
                                    info["attempt_budget"]], sort_keys=True))
        seeds.append(info["stream_seed"])
        tally[(info["model_arm"], info["domain"], info["arm_code"], info["lambda_fields"], info["lambda_coupling"])] += 1
        if roles is not None and roles[fr.domain_id(info["domain"])] != role:
            raise ValueError(f"{cell}: domain {info['domain']} is not a {role} domain")
        pair = {"lambda_fields": info["lambda_fields"], "lambda_coupling": info["lambda_coupling"]}
        scale.append({"model": info["model_arm"], "domain": fr.domain_id(info["domain"]), "arm": info["arm_code"],
                      **pair, "component_scale": completion["component_scale"]})
        cell_lines = [line for line in sequences.read_text().splitlines() if line.strip()]
        if strengths is not None:
            require_declared_depth(cell, completion, len(cell_lines), strengths)
        lines.extend(json.dumps({**json.loads(line), **pair}) + "\n" for line in cell_lines)
    if len(recipes) > 1:
        raise ValueError(f"cells carry {len(recipes)} different sampling recipes, pool sizes or native contracts")
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
    if role == "evaluation":
        if strengths is None:
            raise ValueError("--role evaluation needs the protocol's strengths declaration (--protocol)")
        fr.require_frozen_strength(validated, strengths)
    write_json(out / SCALE, scale)
    write_json(out / RECORDS_COMPLETION, {
        "schema_version": "e20_merged_records_v1", "protocol_sha256": sorted(protocols),
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
    mode.add_argument("--cells", type=Path, nargs="+", help="cell directories or a parent directory of cells")
    mode.add_argument("--shards", type=Path, nargs="+")
    parser.add_argument("--expect-cells", type=int, help="with --cells: refuse unless exactly this many cells are found")
    parser.add_argument("--wild-type-fasta", type=Path, help="query_12.faa; required with --cells")
    parser.add_argument("--protocol", type=Path, help="diagnostic_protocol.json; with --role")
    parser.add_argument("--role", choices=("development", "evaluation"))
    args = parser.parse_args(argv)
    if args.cells:
        if args.wild_type_fasta is None or bool(args.protocol) != bool(args.role):
            parser.error("--cells needs --wild-type-fasta, and --protocol together with --role")
        roles = fr.load_roles(args.protocol) if args.protocol else None
        strengths = fr.load_strengths(args.protocol) if args.role == "evaluation" else None
        summary = merge_records(args.cells, args.out, fr.load_wild_types(args.wild_type_fasta), roles, args.role,
                                args.expect_cells, strengths)
    else:
        summary = merge_shards(args.shards, args.out)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
