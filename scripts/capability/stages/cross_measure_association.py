#!/usr/bin/env python3
"""EXP-R2-247: inventory or associate gate, scoring, generation, and size.

CPU only. ``inventory`` writes presence. ``associate`` computes a coefficient
only when a pair's frozen seats are all resolved; pending pairs stay pending.
Does not invent a missing cell. Does not take a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.capability.context.cross_measure_association import (  # noqa: E402
    EXPECT_ASSOCIATION,
    EXPECT_INVENTORY,
    associate,
    inventory,
)
from src.capability.core.io import write_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("inventory", "associate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.device != "cpu":
        raise SystemExit("cross-measure association is CPU-only")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "inventory":
        payload = inventory()
        write_json(args.out / EXPECT_INVENTORY, payload)
    else:
        payload = associate()
        write_json(args.out / EXPECT_ASSOCIATION, payload)
    print(json.dumps({"stage": args.stage, "out": str(args.out), "pending_pairs": payload.get("pending_pairs")}, sort_keys=True))


if __name__ == "__main__":
    main()
