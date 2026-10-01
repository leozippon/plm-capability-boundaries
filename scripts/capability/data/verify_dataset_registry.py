#!/usr/bin/env python3
"""Re-read the staged bytes and refuse a registry that no longer describes them.

This is the operational entry point behind the rule that an unhashed external
resource is not usable evidence: it recomputes every recorded SHA-256, reports
each listed file that is missing, resized or altered, and each file present in a
dataset directory that the registry does not list. It exits non-zero on the
first kind of disagreement it finds rather than summarising a partial pass.
"""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.capability.core.dataset_registry import (
    REGISTRY_RELPATH,
    is_volatile,
    structural_problems,
    verify_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REPO_ROOT / REGISTRY_RELPATH)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="DATASET",
        help="verify only these datasets; repeatable, default every declared one",
    )
    parser.add_argument("--workers", type=int, default=16, help="hashing threads")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))

    problems = structural_problems(registry)
    if problems:
        print("the registry is not self-consistent:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    wanted = set(args.dataset) or set(registry["datasets"])
    unknown = sorted(wanted - set(registry["datasets"]))
    if unknown:
        raise SystemExit(f"not declared in the registry: {', '.join(unknown)}")

    failures: list[str] = []
    volatile_disagreements: dict[str, list[str]] = {}
    checked = 0
    read_bytes = 0
    started = time.time()
    for name in sorted(wanted):
        entry = registry["datasets"][name]
        dataset_failures, hashed, read = verify_dataset(
            args.data_root / name,
            entry["inventory"]["files"],
            workers=args.workers,
        )
        checked += hashed
        read_bytes += read
        if is_volatile(entry):
            # A directory another campaign is still writing into will disagree,
            # and that disagreement is expected rather than informative. It is
            # printed with its declared reason and kept out of the exit status,
            # so a non-zero exit still means something changed unexpectedly.
            volatile_disagreements[name] = dataset_failures
            print(
                f"{name}: {hashed} files rehashed, {read / 1e9:.3f} GB, "
                f"{len(dataset_failures)} disagreements, DECLARED VOLATILE"
            )
            continue
        failures.extend(dataset_failures)
        verdict = "ok" if not dataset_failures else f"{len(dataset_failures)} FAILURES"
        print(f"{name}: {hashed} files rehashed, {read / 1e9:.3f} GB, {verdict}")

    elapsed = time.time() - started
    print(
        f"rehashed {checked} files and {read_bytes / 1e9:.3f} GB in {elapsed:.1f} s "
        f"across {len(wanted)} datasets"
    )
    for name, disagreements in sorted(volatile_disagreements.items()):
        entry = registry["datasets"][name]
        print(f"{name} is declared volatile: {entry['volatile']['reason']}")
        for disagreement in disagreements:
            print(f"  {disagreement}")

    if failures:
        print(f"{len(failures)} disagreements:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(
        "every listed file of every non-volatile dataset is present at its "
        "recorded size and digest"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
