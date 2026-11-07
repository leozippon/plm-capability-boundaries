#!/usr/bin/env python3
"""Native family-distribution contrast: prepare, score one arm, analyse.

``prepare`` is CPU-only and freezes the lossless native queue, the cluster splits
and the inference units from the retained corpus hits. ``score`` measures one
roster arm on one declared partition. ``analyse`` forms the family-equal
contrasts, the declared simultaneous family and the descriptive layer.

No stage re-retrieves homologs, reads a phenotype, changes the retained R1 data,
folds or likelihoods, or writes into another experiment's directory.
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

from src.capability.extensions.family_distribution import (  # noqa: E402
    analyse,
    load_json,
    prepare,
    read_config,
    score_partition,
    scientific_inputs,
)

CONFIG = ROOT / "configs/family_distribution.json"
LOG = ROOT / "logs/operations/family-distribution-20261106.log"


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
    commands.add_parser("prepare", help="freeze the native queue and units on CPU")
    scorer = commands.add_parser("score", help="score one arm on one partition")
    scorer.add_argument("--arm", required=True)
    scorer.add_argument("--partition", choices=("pilot", "evaluation"), required=True)
    scorer.add_argument("--device", default="cuda:0")
    scorer.add_argument("--batch-size", type=int, default=None)
    scorer.add_argument("--max-len", type=int, default=None)
    commands.add_parser("analyse", help="family-equal contrasts and inference")
    args = parser.parse_args()

    config = read_config(args.config)
    output = ROOT / config["output"]
    receipt = output / "receipts" / (
        "prepare.json" if args.stage == "prepare"
        else f"{args.stage}-{args.arm}-{args.partition}.json" if args.stage == "score"
        else "analyse.json"
    )
    before = resources()
    write(receipt, {"status": "running", "stage": args.stage, "before": before})
    started = time.monotonic()
    try:
        if args.stage == "prepare":
            queue, audit = prepare(ROOT, config)
            scientific_inputs(ROOT, config, queue, create=True)
            for name, value in (("queue.json", queue), ("audit.json", audit)):
                path = output / name
                if path.exists() and load_json(path) != value:
                    raise RuntimeError(
                        f"{path} already holds a different frozen queue; move the run directory "
                        "aside instead of re-preparing over a scored queue")
                write(path, value)
            result = audit["counts"]
        elif args.stage == "score":
            queue = load_json(output / "queue.json")
            batch_size = args.batch_size or int(config["scoring"]["batch_size_default"])
            max_len = args.max_len or int(config["scoring"]["max_len"])
            document = score_partition(
                ROOT, config, queue, arm=args.arm, partition=args.partition,
                device=args.device, batch_size=batch_size, max_len=max_len, log=log)
            result = {"arm": args.arm, "partition": args.partition,
                      "dtype": document.get("dtype"), "units": len(document["units"]),
                      "batch_size": batch_size, "max_len": max_len}
        else:
            summary = analyse(ROOT, config)
            write(output / "analysis" / "summary.json", summary)
            result = {partition: record["status"]
                      for partition, record in summary["partitions"].items()}
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