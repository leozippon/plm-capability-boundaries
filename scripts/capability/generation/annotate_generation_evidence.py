#!/usr/bin/env python3
"""Recover R227 reference coverage or annotate the new native ProGen3 batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.capability.generation import generation_annotations as ga  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recover = commands.add_parser("recover-r227")
    recover.add_argument("--attempts", type=Path, default=Path("results/R6/generation_evidence_20260905/attempts.jsonl"))
    # The R227 identity tables were written on the authoring host and are not
    # distributed with the repository, so they are named rather than defaulted:
    # a default that cannot resolve anywhere would fail later and less clearly.
    recover.add_argument("--table", type=Path, required=True,
                         help="R227 identity table (generations.tsv), from the authoring host")
    recover.add_argument("--query-fasta", type=Path, required=True,
                         help="R227 query FASTA (generations.fasta), from the authoring host")
    recover.add_argument("--report", type=Path, default=Path("results/R6/conditioned_generation_20260826/conditioned_generation.json"))
    recover.add_argument("--out", type=Path, default=Path("results/R6/generation_evidence_20260905/reference_annotations"))
    native = commands.add_parser("native-progen3")
    native_uncond = commands.add_parser("native")
    for parser_native in (native, native_uncond):
        for name in ("attempts", "out", "hmmscan", "pfam-hmm", "diamond", "diamond-db", "reference-metadata"):
            parser_native.add_argument(f"--{name}", type=Path, required=True)
        parser_native.add_argument("--threads", type=int, default=8)
        parser_native.add_argument("--shards", type=int, default=4)
    reference = commands.add_parser("reference-only")
    for name in ("attempts", "out", "diamond", "diamond-db", "reference-metadata"):
        reference.add_argument(f"--{name}", type=Path, required=True)
    reference.add_argument("--threads", type=int, default=8)
    args = vars(parser.parse_args())
    command = args.pop("command")
    args["output"] = args.pop("out")
    functions = {"recover-r227": ga.recover_r227, "native-progen3": ga.annotate_native,
                 "native": ga.annotate_native, "reference-only": ga.reference_only}
    result = functions[command](**args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
