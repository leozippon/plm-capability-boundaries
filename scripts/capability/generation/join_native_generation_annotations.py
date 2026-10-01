#!/usr/bin/env python3
"""Create a derived first-annotation view without changing the frozen native ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.capability.generation.generation_biology_analysis import index_rows, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.attempts)
    originals = index_rows(rows)
    annotations = index_rows(read_jsonl(args.annotations))
    if originals.keys() != annotations.keys():
        raise ValueError("First annotations must cover every frozen attempt exactly")
    joined = []
    for original in rows:
        extra = annotations[original["id"]]
        if extra["sequence_sha256"] != original["sequence_sha256"]:
            raise ValueError("First annotation sequence mismatch")
        if original["class_key"] is not None or original["condition"] != "unconditioned":
            raise ValueError("This join is only for the frozen native unconditional task")
        if extra["target_profile_hit"] is not None or extra["profile_hit_classes"] is not None:
            raise ValueError("The unconditional task has no target-class oracle")
        if original.get("reference_identity") is not None or original.get("any_profile_hit") is not None:
            raise ValueError("Cannot overwrite a previously measured annotation")
        if not isinstance(extra["any_profile_hit"], bool):
            raise ValueError("Incomplete first profile annotation")
        allowed = {k: v for k, v in extra.items() if k.startswith("reference_") or k in
                   {"any_profile_hit", "pfam_families", "profile_hit_classes", "profile_search_status", "target_profile_hit"}}
        joined.append(original | allowed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in joined)
    args.out.write_text(content)
    receipt = {"derivation": "first native annotation joined on exact attempt ID and sequence SHA-256",
               "frozen_inputs_unchanged": True, "records": len(joined),
               "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (args.attempts, args.annotations)},
               "output_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
    args.out.with_suffix(".provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
