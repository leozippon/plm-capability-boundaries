#!/usr/bin/env python3
"""Run the finite native ProGen3 generation extension on a root-assigned GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.capability.models.progen3 import PROGEN3_SOURCE  # noqa: E402
from src.capability.generation.progen3_generation import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=PROGEN3_SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--interface-only", action="store_true")
    args = parser.parse_args()
    result = run(args.checkpoint, args.out, source=args.source, device=args.device, interface_only=args.interface_only)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
