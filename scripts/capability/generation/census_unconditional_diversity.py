#!/usr/bin/env python3
"""CPU census of near-duplicate groups on frozen unconditioned 800-attempt ledgers.

Does not generate. Does not take a GPU. Does not invent a zero for empty
decodes or for zero Pfam hits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

EXPECT = "unconditional_diversity.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, action="append", default=[], help="annotated_attempts.jsonl")
    parser.add_argument("--root", type=Path, action="append", default=[], help="directory tree to search")
    parser.add_argument("--package", type=Path, default=REPO, help="repository or freeze that provides src.capability")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def discover(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        found.extend(sorted(root.rglob("annotated_attempts.jsonl")))
    return found


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    package = str(args.package.resolve())
    if package not in sys.path:
        sys.path.insert(0, package)
    # Imported here, after --package has been put on the path, so the option
    # selects which tree provides src.capability rather than being inert.
    from src.capability.generation.generation_annotations import (
        read_attempts, unconditional_diversity_census)
    from src.capability.core.io import write_json

    paths = list(args.ledger) + discover(args.root)
    if not paths:
        raise SystemExit("no annotated ledgers")
    checkpoints: dict[str, dict] = {}
    for path in paths:
        rows = read_attempts(path)
        census = unconditional_diversity_census(rows)
        name = census["arm"]
        if name in checkpoints:
            raise SystemExit(f"duplicate arm {name}: {path}")
        census["ledger"] = str(path)
        checkpoints[name] = census
        print(json.dumps({"arm": name, "n_groups_nonempty": census["n_groups_nonempty"], "n_any_profile_groups": census["n_any_profile_groups"]}, sort_keys=True))
    payload = {
        "campaign": "EXP-R2-246-diversity-census",
        "is_scientific_measurement": True,
        "not_biological_diversity": True,
        "not_function": True,
        "unit": "residues",
        "shingle_length": 5,
        "checkpoints": checkpoints,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / EXPECT, payload)


if __name__ == "__main__":
    main()
