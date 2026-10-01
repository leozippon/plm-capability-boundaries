#!/usr/bin/env python3
"""EXP-R2-246: one native unlabelled generation point per admitted arm.

``--stage interface-only`` writes the prompt contract and does not load weights
or write the 800-attempt ledger. ``generate`` writes the ledger. ``sample-structure``
draws the score-independent 128-parent subset after the ledger exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.capability.generation.unconditional_generation import (  # noqa: E402
    ADMITTED_ARMS,
    LEDGER_NAME,
    checkpoint_for_arm,
    generate,
    load_generator,
    read_attempts,
    require_admitted,
    write_interface_record,
    write_structure_subset,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("generate", "sample-structure", "interface-only"),
    )
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args(argv)


def resolve_checkpoint(arm: str, checkpoint: Path | None) -> Path:
    require_admitted(arm)
    if checkpoint is not None:
        return Path(checkpoint)
    return checkpoint_for_arm(arm)


def run_interface(args: argparse.Namespace) -> dict:
    checkpoint = resolve_checkpoint(args.arm, args.checkpoint)
    return write_interface_record(args.out, args.arm, checkpoint=checkpoint)


def run_generate(args: argparse.Namespace) -> dict:
    if args.arm == "galactica-30b" and args.device == "cpu":
        raise SystemExit("galactica-30b generate is not a CPU or test path")
    checkpoint = resolve_checkpoint(args.arm, args.checkpoint)
    model, tokenizer = load_generator(args.arm, device=args.device, checkpoint=checkpoint)
    rows = generate(
        args.arm,
        model=model,
        tokenizer=tokenizer,
        output_dir=args.out,
    )
    structure = write_structure_subset(args.out, rows, arm=args.arm)
    return {
        "arm": args.arm,
        "attempts": len(rows),
        "out": str(args.out),
        "structure_records_including_shuffles": structure["structure_records_including_shuffles"],
    }


def run_sample_structure(args: argparse.Namespace) -> dict:
    require_admitted(args.arm)
    ledger = Path(args.out) / LEDGER_NAME
    if not ledger.is_file():
        raise SystemExit(f"sample-structure needs {ledger}")
    return write_structure_subset(args.out, read_attempts(ledger), arm=args.arm)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.arm not in ADMITTED_ARMS:
        require_admitted(args.arm)
    if args.stage == "interface-only":
        result = run_interface(args)
    elif args.stage == "generate":
        result = run_generate(args)
    else:
        result = run_sample_structure(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
