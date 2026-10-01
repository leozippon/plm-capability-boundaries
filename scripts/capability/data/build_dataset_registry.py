#!/usr/bin/env python3
"""Measure the file inventory of every dataset directory the registry declares.

The registry's provenance fields are authored by hand; this stage only fills the
measured half, so it never writes a provenance value and never invents one. It
refuses a registry whose declared directories and the directories actually
present under ``data/`` disagree, because a silently added dataset is exactly
the unhashed external resource the registry exists to prevent.

An already measured dataset is left alone unless ``--remeasure`` names it, so
authoring provenance after a measurement does not force an 80 GB re-read.
"""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.capability.core.dataset_registry import (
    REGISTRY_RELPATH,
    inventory,
    structural_problems,
)
from src.capability.core.io import write_json

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPO_ROOT / REGISTRY_RELPATH,
        help="registry JSON to update in place",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="directory holding the declared dataset directories",
    )
    parser.add_argument(
        "--remeasure",
        action="append",
        default=[],
        metavar="DATASET",
        help="re-read a dataset that already carries an inventory; repeatable, ALL for every one",
    )
    parser.add_argument("--workers", type=int, default=16, help="hashing threads")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    declared = set(registry["datasets"])
    present = {path.name for path in args.data_root.iterdir() if path.is_dir()}

    undeclared = sorted(present - declared)
    if undeclared:
        raise SystemExit(
            "these dataset directories are present but not declared in the registry, "
            f"so they carry no provenance: {', '.join(undeclared)}"
        )
    missing = sorted(declared - present)
    if missing:
        raise SystemExit(f"these declared datasets are absent from disk: {', '.join(missing)}")

    remeasure = set(args.remeasure)
    total_bytes = 0
    total_files = 0
    for name in sorted(declared):
        entry = registry["datasets"][name]
        if entry.get("inventory") is not None and "ALL" not in remeasure and name not in remeasure:
            print(f"{name}: inventory retained, {entry['inventory']['file_count']} files")
            continue
        started = time.time()
        records = inventory(args.data_root / name, workers=args.workers)
        measured_bytes = sum(record.bytes for record in records)
        entry["inventory"] = {
            "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "file_count": len(records),
            "total_bytes": measured_bytes,
            "files": [record.as_json() for record in records],
        }
        elapsed = time.time() - started
        rate = measured_bytes / 1e9 / elapsed if elapsed > 0 else float("nan")
        print(
            f"{name}: {len(records)} files, {measured_bytes / 1e9:.3f} GB, "
            f"{elapsed:.1f} s, {rate:.2f} GB/s"
        )
        total_bytes += measured_bytes
        total_files += len(records)

    problems = structural_problems(registry)
    if problems:
        raise SystemExit("the updated registry is not self-consistent:\n  " + "\n  ".join(problems))

    write_json(args.registry, registry)
    print(f"measured {total_files} files and {total_bytes / 1e9:.3f} GB into {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
