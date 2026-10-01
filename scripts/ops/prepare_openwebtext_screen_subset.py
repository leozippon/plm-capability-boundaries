#!/usr/bin/env python3
"""Freeze a document-disjoint OpenWebText subset without duplicating text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    # `--shard-selection-seed` used to be required here. Nothing read it except
    # the manifest, which transcribed it into a `shard_selection` block naming a
    # sampling algorithm this script does not run: shard selection is
    # `sorted(source_dir.glob("*.parquet"))`, every shard present, no sample and
    # no seed. A required argument whose only effect is to assert a procedure
    # that did not happen is worse than no argument. The manifest now records
    # what the code does.
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--train-count", type=int, default=300_000)
    parser.add_argument("--validation-count", type=int, default=5_000)
    parser.add_argument("--test-count", type=int, default=5_000)
    return parser.parse_args()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    parquet_paths = sorted(args.source_dir.glob("*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"No parquet files found under {args.source_dir}")

    requested = args.train_count + args.validation_count + args.test_count
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    empty_count = 0
    duplicate_count = 0
    input_count = 0

    for path in parquet_paths:
        relative_source = path.relative_to(args.source_dir).as_posix()
        row_index = 0
        parquet_file = pq.ParquetFile(path)
        if "text" not in parquet_file.schema.names:
            raise SystemExit(f"Missing text column in {path}")
        for batch in parquet_file.iter_batches(columns=["text"], batch_size=4096):
            for text in batch.column(0).to_pylist():
                input_count += 1
                if not isinstance(text, str) or not text.strip():
                    empty_count += 1
                    row_index += 1
                    continue
                encoded = text.encode("utf-8")
                text_sha256 = hashlib.sha256(encoded).hexdigest()
                if text_sha256 in seen:
                    duplicate_count += 1
                    row_index += 1
                    continue
                seen.add(text_sha256)
                priority = hashlib.sha256(
                    f"{args.selection_seed}:{text_sha256}".encode("ascii")
                ).hexdigest()
                records.append(
                    {
                        "id": f"openwebtext:{text_sha256}",
                        "source": relative_source,
                        "row_index": row_index,
                        "sha256": text_sha256,
                        "characters": len(text),
                        "utf8_bytes": len(encoded),
                        "selection_priority": priority,
                    }
                )
                row_index += 1

    if len(records) < requested:
        raise SystemExit(
            f"Only {len(records)} unique nonempty documents for {requested} requested"
        )

    records.sort(key=lambda row: (row["selection_priority"], row["sha256"]))
    boundaries = (
        ("test", args.test_count),
        ("validation", args.validation_count),
        ("train", args.train_count),
    )
    selected: list[dict[str, Any]] = []
    offset = 0
    counts: dict[str, int] = {}
    for split, count in boundaries:
        split_rows = records[offset : offset + count]
        for row in split_rows:
            row["split"] = split
        selected.extend(split_rows)
        counts[split] = len(split_rows)
        offset += count

    split_order = {"train": 0, "validation": 1, "test": 2}
    selected.sort(
        key=lambda row: (
            split_order[row["split"]],
            row["selection_priority"],
            row["sha256"],
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "selection_manifest.jsonl"
    write_jsonl_atomic(manifest_path, selected)

    summary = {
        "schema_version": "openwebtext_screen_subset",
        "source_dataset": "Skylion007/openwebtext",
        "source_revision": args.source_revision,
        "source_directory": str(args.source_dir),
        "selected_shards": [path.name for path in parquet_paths],
        "shard_selection": {
            "shards_used": len(parquet_paths),
            "seed": None,
            "algorithm": 'sorted(source_dir.glob("*.parquet")) -- every shard present, unsampled',
            "note": (
                "shard selection is not randomised: the subset's randomness is "
                "entirely in document_selection below, over the union of these "
                "shards. Whichever shards source_dir holds is therefore part of "
                "this artefact's provenance, which is why they are listed in "
                "selected_shards."
            ),
        },
        "document_selection": {
            "seed": args.selection_seed,
            "text_identity": "sha256(exact UTF-8 text)",
            "priority": "sha256(f'{seed}:{text_sha256}')",
            "assignment_order": ["test", "validation", "train"],
            "counts": counts,
        },
        "observations": {
            "input_documents": input_count,
            "unique_nonempty_documents": len(records),
            "empty_documents": empty_count,
            "duplicate_documents": duplicate_count,
            "selected_documents": len(selected),
            "unselected_documents": len(records) - len(selected),
        },
        "selection_manifest": {
            "path": manifest_path.name,
        },
    }
    write_json_atomic(args.output_dir / "selection_summary.json", summary)

    print(json.dumps(summary["observations"], sort_keys=True))
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
