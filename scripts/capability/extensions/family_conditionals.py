#!/usr/bin/env python3
"""Conditional twenty-way log-score: prepare, score one arm, analyse.

``prepare`` is CPU-only and freezes sites, contexts and training-only baseline
parameters from the verified first-stage cohort. Reference log-probabilities are
derived only in memory after complete input validation. ``score``
measures one inherited roster arm on one declared partition. ``analyse`` forms
the family-equal observed log-scores, the two contrasts per arm and the declared
simultaneous family.

No stage retrieves homologs, reads a phenotype, reads a first-stage score, or
writes into another experiment's directory.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

from src.capability.extensions.family_conditionals import (  # noqa: E402
    analyse,
    prepare,
    read_config,
    score_partition,
)

CONFIG = ROOT / "configs/family_conditionals.json"
LOG = ROOT / "logs/operations/family-conditionals-20261107.log"


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=1, allow_nan=False) + "\n")
    os.replace(temporary, path)


def log(message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def resources() -> dict:
    return {
        "executable": sys.executable,
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "memory": [
            line for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemAvailable:", "MemTotal:"))
        ],
        "threads": {name: os.environ[name] for name in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    commands = parser.add_subparsers(dest="stage", required=True)
    commands.add_parser("prepare", help="freeze sites, contexts and baseline parameters on CPU")
    scorer = commands.add_parser("score", help="score one arm on one partition")
    scorer.add_argument("--arm", required=True)
    scorer.add_argument("--partition", choices=("pilot", "evaluation"), required=True)
    scorer.add_argument("--device", default="cuda:0")
    commands.add_parser("analyse", help="family-equal log-scores and the 66-column family")
    args = parser.parse_args()

    config = read_config(args.config)
    output = ROOT / config["output"]
    receipt = output / "receipts" / (
        "prepare.json" if args.stage == "prepare"
        else f"score-{args.arm}-{args.partition}.json" if args.stage == "score"
        else "analyse.json"
    )
    before = resources()
    write(receipt, {"status": "running", "stage": args.stage, "before": before})
    started = time.monotonic()
    try:
        if args.stage == "prepare":
            manifest = prepare(ROOT, config)
            result = manifest["counts"]
        elif args.stage == "score":
            document = score_partition(ROOT, config, arm=args.arm, partition=args.partition,
                                       device=args.device, log=log)
            result = {"arm": args.arm, "partition": args.partition,
                      "dtype": document.get("dtype"), "units": len(document["units"]),
                      "complete": document.get("complete")}
        else:
            summary = analyse(ROOT, config)
            write(output / "analysis" / "summary.json", summary)
            result = {partition: record["status"]
                      for partition, record in summary["partitions"].items()}
            result["declared"] = summary["declared"]
    except BaseException as error:
        write(receipt, {"status": "failed", "stage": args.stage, "before": before,
                        "after": resources(), "error": f"{type(error).__name__}: {error}"})
        log(f"{args.stage}: failed: {type(error).__name__}: {error}")
        raise
    write(receipt, {"status": "complete", "stage": args.stage, "before": before,
                    "after": resources(), "seconds": time.monotonic() - started,
                    "result": result})
    log(f"{args.stage}: complete {json.dumps(result, default=str)}")


if __name__ == "__main__":
    main()