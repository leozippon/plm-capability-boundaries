#!/usr/bin/env python3
"""Admit the objective-tuned campaign's cells, or refuse the campaign.

This runs between generation and folding, and it exists because four properties
of a cell are invisible in the artefacts a reader would otherwise look at, and
each one silently destroys the paired contrast if it does not hold.

**The attempt count is the declared one.** ``is_scientific_measurement`` on a
ledger means "this cell is the policy's full 800 attempts", so a short cell must
not be pooled with full ones: a base pool is the concatenation of four cells and
a tuned pool of one, and the four-to-one budget ratio the declaration requires is
a count, not a label.

**The ledger names the checkpoint the loader opened.** Both sides of a pair share
an architecture, a tokenizer, a depth and a width, so a tuned cell that loaded
base weights would produce a plausible ledger, pass every shape check, and read
as "preference optimisation had no effect". The resolved path is compared against
the arm's own declaration, and a tuned cell's path is additionally required to
differ from its base's.

**The seed is the declared one.** Cells of one arm differ only by seed; two cells
drawing the same stream would make a pool a partial copy of itself and inflate
every interval computed over it.

**The denominator is whole.** The lane retains empty and budget-censored
continuations, so every attempt must still carry one of the three stop statuses
and the three must sum to the attempt count. A pool silently filtered to its
successes would turn the fraction of unusable draws -- a property of the
checkpoint -- into a property of this script.

The record it writes, ``objective_tuned_cells.json``, is what the fold cohort
builder requires before it will consume a cell. Any failure is an exit code and
a named cell, not a warning: a campaign with one mislabelled pool is not a
campaign with 59 good cells.

It also carries the **censoring rate** per cell, per stream and per arm, which is
a measured property of these checkpoints rather than bookkeeping. The lane's
400-token budget stays frozen, because the arms these are read beside were
generated under it, so a continuation that reaches the cap is retained as a
truncated fragment rather than a sequence the model chose to end. Preference
optimisation toward structural confidence has an obvious reason to move length
preference, so if a tuned arm is censored at a materially different rate than its
base, part of any pLDDT difference between them is a length difference rather
than a structural one. Aggregating it here rather than adding a field to each
ledger keeps one source -- ``attempts.jsonl``, which every cell writes and which
this reads uniformly -- and avoids a record that would exist on cells run after a
code change and not on cells run before it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.capability.generation.build_objective_tuned_campaign import label  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation import objective_tuned_arms as ot  # noqa: E402
from src.capability.generation import unconditional_generation as ug  # noqa: E402

RECORD = "objective_tuned_cells.json"
SCHEMA = "objective_tuned_cells_v1"

#: The three statuses the lane assigns, every one of which stays in the
#: denominator. ``stop_status`` is total by construction, so a row carrying
#: anything else means the ledger was written by other code than this lane.
STOP_STATUSES = ("native_terminal", "budget_censored", "empty_or_noncanonical")


def check_cell(cell: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Every admission check for one cell. Returns its row; raises on refusal."""

    name = label(cell)
    out_dir = run_dir / name
    summary_path = out_dir / ug.EXPECT_GENERATE
    ledger_path = out_dir / ug.LEDGER_NAME
    for path in (summary_path, ledger_path):
        if not path.is_file():
            raise SystemExit(f"{name}: missing {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    declared = int(cell["attempts"])
    if summary["arm"] != cell["arm"]:
        raise SystemExit(
            f"{name}: the ledger says arm {summary['arm']!r} where the plan declares "
            f"{cell['arm']!r}; the cell's identity and its directory disagree"
        )
    if int(summary["attempts"]) != declared:
        raise SystemExit(
            f"{name}: drew {summary['attempts']} attempts against {declared} declared"
        )
    if summary["is_scientific_measurement"] is not True:
        raise SystemExit(
            f"{name}: the ledger does not claim the policy's full attempt count"
        )
    if int(summary["seed"]) != int(cell["seed"]):
        raise SystemExit(
            f"{name}: drew seed {summary['seed']} against {cell['seed']} declared; "
            "two cells on one stream are not two streams"
        )
    if summary["attempts_sha256"] != sha256_file(ledger_path):
        raise SystemExit(f"{name}: {ug.LEDGER_NAME} does not match its recorded digest")

    expected_checkpoint = ot.spec_for(cell["arm"]).path
    recorded = summary.get("checkpoint")
    if recorded is None:
        raise SystemExit(
            f"{name}: the ledger names no checkpoint, so nothing says which sibling "
            "produced this pool"
        )
    if Path(recorded) != expected_checkpoint:
        raise SystemExit(
            f"{name}: the ledger names checkpoint {recorded} where arm {cell['arm']!r} "
            f"declares {expected_checkpoint}"
        )
    if cell["role"] == "tuned":
        base_path = ot.spec_for(ot.base_of(cell["arm"])).path
        if Path(recorded) == base_path:
            raise SystemExit(
                f"{name}: a tuned cell recorded its base's checkpoint {recorded}"
            )

    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != declared:
        raise SystemExit(f"{name}: {len(rows)} ledger rows against {declared} declared")
    statuses = Counter(row["stop_status"] for row in rows)
    unknown = sorted(set(statuses) - set(STOP_STATUSES))
    if unknown:
        raise SystemExit(f"{name}: unknown stop statuses {unknown}")
    if sum(statuses.values()) != declared:
        raise SystemExit(
            f"{name}: stop statuses cover {sum(statuses.values())} of {declared} attempts; "
            "the denominator is not whole"
        )
    non_empty = sum(1 for row in rows if row["sequence"])
    if non_empty != declared - statuses.get("empty_or_noncanonical", 0):
        raise SystemExit(
            f"{name}: {non_empty} non-empty sequences disagrees with the "
            f"{statuses.get('empty_or_noncanonical', 0)} attempts marked empty"
        )
    if not non_empty:
        raise SystemExit(f"{name}: every attempt extracted empty; that is an interface fault")

    lengths = sorted(len(row["sequence"]) for row in rows if row["sequence"])
    return {
        "label": name,
        "arm": cell["arm"],
        "pair": cell["pair"],
        "role": cell["role"],
        "stream": cell["stream"],
        "replicate": cell["replicate"],
        "seed": int(summary["seed"]),
        "attempts": declared,
        "checkpoint": recorded,
        "stop_status": {status: statuses.get(status, 0) for status in STOP_STATUSES},
        "non_empty": non_empty,
        "non_empty_rate": round(non_empty / declared, 4),
        "budget_censored_rate": round(statuses.get("budget_censored", 0) / declared, 4),
        "native_terminal_rate": round(statuses.get("native_terminal", 0) / declared, 4),
        "residues_min": lengths[0],
        "residues_median": lengths[len(lengths) // 2],
        "residues_max": lengths[-1],
        "attempts_sha256": summary["attempts_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True, help="the campaign's cell_plan.json")
    parser.add_argument("--run-dir", type=Path, required=True, help="the results run directory holding the cells")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="accepted and unused; CPU-only")
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    cells = plan["cells"]
    rows = [check_cell(cell, args.run_dir) for cell in cells]

    seeds = [row["seed"] for row in rows]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("two admitted cells realised the same seed")
    by_arm: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = by_arm.setdefault(
            row["arm"],
            {"cells": 0, "attempts": 0, "non_empty": 0, "checkpoints": set(), "stop_status": {}},
        )
        entry["cells"] += 1
        entry["attempts"] += row["attempts"]
        entry["non_empty"] += row["non_empty"]
        entry["checkpoints"].add(row["checkpoint"])
        for status in STOP_STATUSES:
            entry["stop_status"][status] = (
                entry["stop_status"].get(status, 0) + row["stop_status"][status]
            )
    for arm, entry in by_arm.items():
        if len(entry["checkpoints"]) != 1:
            raise SystemExit(f"{arm}: its cells name {sorted(entry['checkpoints'])}")
        entry["checkpoint"] = entry.pop("checkpoints").pop()
        entry["non_empty_rate"] = round(entry["non_empty"] / entry["attempts"], 4)
        entry["budget_censored_rate"] = round(
            entry["stop_status"]["budget_censored"] / entry["attempts"], 4
        )
        entry["native_terminal_rate"] = round(
            entry["stop_status"]["native_terminal"] / entry["attempts"], 4
        )

    # Per stream as well as per arm: the declaration treats the three streams as a
    # conjunction requirement, so a censoring rate that holds on average but not in
    # every stream is not the same finding.
    by_stream: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['arm']}_s{row['stream']}"
        entry = by_stream.setdefault(
            key,
            {"arm": row["arm"], "stream": row["stream"], "cells": 0, "attempts": 0,
             "non_empty": 0, "budget_censored": 0, "native_terminal": 0},
        )
        entry["cells"] += 1
        entry["attempts"] += row["attempts"]
        entry["non_empty"] += row["non_empty"]
        entry["budget_censored"] += row["stop_status"]["budget_censored"]
        entry["native_terminal"] += row["stop_status"]["native_terminal"]
    for entry in by_stream.values():
        entry["non_empty_rate"] = round(entry["non_empty"] / entry["attempts"], 4)
        entry["budget_censored_rate"] = round(entry["budget_censored"] / entry["attempts"], 4)
        entry["native_terminal_rate"] = round(entry["native_terminal"] / entry["attempts"], 4)

    # The paired reading the qualification exists for, computed once here rather
    # than left for each consumer to derive: a tuned-minus-base censoring
    # difference is how much of a pLDDT difference could be a length difference.
    pairs: dict[str, dict[str, Any]] = {}
    for cell in cells:
        if cell["role"] != "tuned" or cell["pair"] in pairs:
            continue
        tuned, base = cell["arm"], ot.base_of(cell["arm"])
        if tuned not in by_arm or base not in by_arm:
            continue
        pairs[cell["pair"]] = {
            "base": base,
            "tuned": tuned,
            "base_budget_censored_rate": by_arm[base]["budget_censored_rate"],
            "tuned_budget_censored_rate": by_arm[tuned]["budget_censored_rate"],
            "tuned_minus_base_censored_rate": round(
                by_arm[tuned]["budget_censored_rate"] - by_arm[base]["budget_censored_rate"], 4
            ),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": SCHEMA,
        "campaign": plan["campaign"],
        "declaration": plan["declaration"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": str(args.run_dir),
        "plan_sha256": sha256_file(args.plan),
        "cells_declared": len(cells),
        "cells_admitted": len(rows),
        "complete": len(rows) == len(cells),
        "attempts_total": sum(row["attempts"] for row in rows),
        "non_empty_total": sum(row["non_empty"] for row in rows),
        "budget_note": (
            "max_new_tokens stays at the lane's frozen 400; a budget_censored "
            "attempt is a truncated fragment retained in the denominator, not a "
            "sequence the model ended. Censoring rates are reported per cell, per "
            "stream and per arm so a tuned-minus-base pLDDT difference can be read "
            "against a tuned-minus-base length difference."
        ),
        "arms": by_arm,
        "streams": by_stream,
        "pair_censoring": pairs,
        "cells": rows,
        "code_sha256": sha256_file(Path(__file__)),
    }
    write_json(args.out / RECORD, record)
    print(json.dumps({key: record[key] for key in ("cells_admitted", "attempts_total", "non_empty_total", "complete")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
