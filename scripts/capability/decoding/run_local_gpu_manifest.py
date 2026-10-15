#!/usr/bin/env python3
"""Execute a campaign manifest on this host's own GPUs, one cell per card.

``h200/h200_campaign_queue.sh`` is the cluster's executor and is host-local to the
cluster operations tree. A campaign that runs on the local cards needs the same
contract -- the manifest is the record of what ran -- without the cluster's SSH,
Kubernetes and GPFS layers, so this is that contract and nothing else.

The manifest is the tab-separated form the campaign builders already write:

    slot  key  gpu  stage  label  env  expect  args

Rows are grouped by ``slot`` and slots run in order; within a slot every row runs
concurrently, each owning one card. ``gpu`` is a card index or ``cpu``. A GPU row
runs with ``CUDA_VISIBLE_DEVICES`` set to its own index and is given
``--device cuda:0``, so a cell can only ever touch the card the manifest assigned
it: a card this campaign does not own cannot be reached by a cell that was
mislabelled. A ``cpu`` row is given ``--device cpu``.

``--device`` and ``--out`` are injected **before** the manifest's own arguments and
a manifest that spells either is refused, which is the contract the stage parsers
are written against. ``--out`` is ``<run>/<out-prefix>/<label>``, so a cell's
output path is a function of the manifest and the run directory alone.

``--dry-run`` resolves every row -- the stage path, the interpreter, the injected
arguments, the output directory -- and runs nothing. A manifest must dry-run clean
before it is executed. ``--resume`` skips a row whose ``expect`` file is already
present, because the stages refuse a non-empty output directory and a partial
campaign otherwise has to be discarded whole.

Writes ``<manifest>.status.tsv`` beside the manifest and one log per row under
``<run>/logs``. A row that exits non-zero fails its slot; later slots do not run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts/capability"))

from entrypoints import stage_path  # noqa: E402

COLUMNS = ("slot", "key", "gpu", "stage", "label", "env", "expect", "args")
INJECTED = ("--device", "--out")


class Row(NamedTuple):
    slot: int
    key: str
    gpu: str
    stage: str
    label: str
    env: dict[str, str]
    interpreter: Path
    expect: str
    args: list[str]
    line: int


def parse_env(field: str, *, line: int) -> tuple[dict[str, str], Path]:
    """``VAR=value`` assignments, of which ``TRANSFER_PYTHON`` names the interpreter."""

    env: dict[str, str] = {}
    for item in field.split():
        name, _, value = item.partition("=")
        if not name or not _:
            raise ValueError(f"line {line}: env field holds {item!r}, which is not VAR=value")
        env[name] = value
    interpreter = env.pop("TRANSFER_PYTHON", None)
    if interpreter is None:
        raise ValueError(f"line {line}: env field declares no TRANSFER_PYTHON interpreter")
    return env, Path(interpreter)


def read_manifest(path: Path) -> list[Row]:
    rows: list[Row] = []
    for number, text in enumerate(path.read_text().splitlines(), start=1):
        if not text.strip() or text.lstrip().startswith("#"):
            continue
        fields = text.split("\t")
        if len(fields) != len(COLUMNS):
            raise ValueError(
                f"line {number}: {len(fields)} tab-separated fields, expected {len(COLUMNS)} "
                f"({', '.join(COLUMNS)})")
        slot, key, gpu, stage, cell, env_field, expect, args = fields
        if gpu != "cpu" and not gpu.isdigit():
            raise ValueError(f"line {number}: gpu field {gpu!r} is neither a card index nor 'cpu'")
        arguments = args.split()
        spelled = [item for item in arguments if item in INJECTED]
        if spelled:
            raise ValueError(
                f"line {number}: the args column spells {spelled}, which this runner injects; "
                "a manifest that declares either would make the cell unlaunchable")
        env, interpreter = parse_env(env_field, line=number)
        rows.append(Row(int(slot), key, gpu, stage, cell, env, interpreter, expect,
                        arguments, number))
    if not rows:
        raise ValueError(f"{path} holds no manifest row")
    labels = [row.label for row in rows]
    if len(set(labels)) != len(labels):
        raise ValueError("two manifest rows carry the same label, so they would share one output")
    return rows


def resolve(row: Row, *, run: Path, out_prefix: str) -> dict[str, Any]:
    """Everything needed to launch one row, with nothing left to decide at launch time."""

    stage = stage_path(ROOT, row.stage)
    out = (run / out_prefix / row.label) if out_prefix not in ("", ".") else run / row.label
    device = "cpu" if row.gpu == "cpu" else "cuda:0"
    environment = dict(os.environ)
    environment.update(row.env)
    environment["CUDA_VISIBLE_DEVICES"] = "" if row.gpu == "cpu" else row.gpu
    command = [str(row.interpreter), str(stage), "--device", device, "--out", str(out), *row.args]
    return {"row": row, "stage": stage, "out": out, "command": command, "env": environment,
            "expect": out / row.expect}


def problems(plan: dict[str, Any], *, resume: bool) -> list[str]:
    row = plan["row"]
    found: list[str] = []
    if not plan["stage"].is_file():
        found.append(f"{row.label}: stage {plan['stage']} is not a file")
    if not row.interpreter.is_file():
        found.append(f"{row.label}: interpreter {row.interpreter} is not a file")
    out = plan["out"]
    if out.exists() and any(out.iterdir()) and not (resume and plan["expect"].is_file()):
        found.append(f"{row.label}: output directory {out} exists and is not empty")
    return found


def launch(plan: dict[str, Any], logs: Path) -> dict[str, Any]:
    row = plan["row"]
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{row.label}.log"
    handle = log.open("w")
    handle.write(" ".join(plan["command"]) + "\n")
    handle.flush()
    started = time.time()
    process = subprocess.Popen(plan["command"], stdout=handle, stderr=subprocess.STDOUT,
                               env=plan["env"], cwd=str(ROOT))
    return {**plan, "process": process, "handle": handle, "log": log, "started": started}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="run directory the outputs land in")
    parser.add_argument("--out-prefix", default=".",
                        help="subdirectory of --run each row's output sits in; '.' for the run root")
    parser.add_argument("--dry-run", action="store_true", help="resolve and validate; run nothing")
    parser.add_argument("--resume", action="store_true",
                        help="skip a row whose expected output file is already present")
    parser.add_argument("--only-slot", type=int, action="append", default=None,
                        help="run only these slots; repeatable")
    args = parser.parse_args(argv)

    rows = read_manifest(args.manifest)
    if args.only_slot:
        rows = [row for row in rows if row.slot in set(args.only_slot)]
        if not rows:
            raise SystemExit(f"no manifest row carries a slot in {sorted(set(args.only_slot))}")
    plans = [resolve(row, run=args.run, out_prefix=args.out_prefix) for row in rows]

    by_slot: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for plan in plans:
        by_slot[plan["row"].slot].append(plan)
    for slot, group in sorted(by_slot.items()):
        cards = [plan["row"].gpu for plan in group if plan["row"].gpu != "cpu"]
        if len(set(cards)) != len(cards):
            raise SystemExit(f"slot {slot} assigns one card to two rows")

    refusals = [message for plan in plans for message in problems(plan, resume=args.resume)]
    if refusals:
        for message in refusals:
            print(f"refused: {message}", file=sys.stderr)
        raise SystemExit(f"{len(refusals)} manifest rows cannot be launched as written")

    skipped = [plan for plan in plans if args.resume and plan["expect"].is_file()]
    runnable = [plan for plan in plans if plan not in skipped]
    print(f"{args.manifest.name}: {len(plans)} rows in {len(by_slot)} slots, "
          f"{len(skipped)} already complete, {len(runnable)} to run")
    if args.dry_run:
        for plan in plans:
            print(f"  slot {plan['row'].slot} gpu {plan['row'].gpu} -> {plan['out']}")
        return 0

    status = args.manifest.with_suffix(".status.tsv")
    lines = ["\t".join(("slot", "gpu", "label", "status", "returncode", "seconds", "log")) + "\n"]
    for plan in skipped:
        lines.append("\t".join((str(plan["row"].slot), plan["row"].gpu, plan["row"].label,
                                "skipped", "0", "0", "")) + "\n")
    failures = 0
    for slot, group in sorted(by_slot.items()):
        active = [launch(plan, args.run / "logs") for plan in group if plan not in skipped]
        if not active:
            continue
        print(f"slot {slot}: {len(active)} rows running", flush=True)
        for plan in active:
            code = plan["process"].wait()
            plan["handle"].close()
            seconds = time.time() - plan["started"]
            complete = code == 0 and plan["expect"].is_file()
            if not complete:
                failures += 1
            lines.append("\t".join((
                str(plan["row"].slot), plan["row"].gpu, plan["row"].label,
                "complete" if complete else "failed", str(code), f"{seconds:.1f}",
                str(plan["log"]))) + "\n")
            print(f"  {plan['row'].label}: {'complete' if complete else 'FAILED'} "
                  f"rc={code} {seconds:.1f}s", flush=True)
        status.write_text("".join(lines))
        if failures:
            print(f"{failures} rows failed in slot {slot}; later slots are not run",
                  file=sys.stderr)
            return 1
    status.write_text("".join(lines))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
