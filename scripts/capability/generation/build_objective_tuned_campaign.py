#!/usr/bin/env python3
"""Freeze the objective-tuned free-length generation campaign and its manifests.

The design is ``docs/objective-alignment.md``; this entry point turns its
generation paragraph into cells and nothing else. It does not analyse, does not
fold, and does not decide anything the declaration left open.

The design, in one paragraph
============================

Three matched pairs reach the free-length lane: ProtGPT3-112M, ProtGPT3-1.3B and
ProtGPT3-10B against their DPO siblings, and the author's own ProGen2-small
mirror against ProGen2-small-mix7. ``objective_tuned_arms`` declares the pairing
and reads the filesystem for staging, so a pair whose weights are incomplete is
skipped with the reason written into the plan rather than quietly absent. The
ProtGPT2/ZymCTRL pair is declared into the conditioned lane and is not scheduled
here; the manifest header says so, because an unconditioned ZymCTRL prompt would
be a rendering that checkpoint never saw.

Budget and streams
==================

``unconditional_generation.POLICY`` is the operating point and nothing here moves
it: 800 attempts, temperature 0.85, top-p 0.95, top-k 50, 400 new tokens, and
empty or budget-censored continuations retained in the denominator. One *cell* is
therefore one 800-attempt draw, and the declaration's two budget requirements are
spelled in cells rather than in a new policy:

* three streams per arm, so three independent seeds per arm;
* the base arm at four times the tuned budget, so four cells per base stream
  against one per tuned stream.

That gives 3 cells and 2,400 attempts per tuned arm, 12 cells and 9,600 attempts
per base arm, 15 cells per pair. Spelling the ratio as repeated cells rather than
as a larger ``n`` keeps every cell at the policy's own attempt count, so
``is_scientific_measurement`` stays meaningful on each ledger and a base pool is
the concatenation of four cells rather than a differently-sized object.

A pair whose weights are incomplete is seeded and then dropped, so the manifest
has a gap where its cells would be and every scheduled cell keeps the seed it
would have had with the whole arm set staged.

Seeds
=====

The policy's batch seed rule is ``seed + batch_index``, and a cell runs
``800 / batch_size`` batches, so two cells whose seeds are closer than that
overlap their draws. :data:`SEED_STEP` is 1000 against a maximum of 800 batches,
and :data:`SEED_BASE` is clear of every seed family this repository has used: the
E20 development campaign occupies [20261013000, 20261268400], the confirmatory one
[20262013000, 20262236200] and the ProGen3 robustness check
[20263015000, 20263134200]. Forty-five cells from this base occupy
[20264011000, 20264055000].

Output
======

``<run>/queue/campaign_objective_tuned_20261109_generate.tsv`` is the H200 queue
manifest: ``slot, key, gpu, stage, label, env, expect, args``, with ``--device``
and ``--out`` injected by ``h200/h200_campaign_queue.sh``. That runner derives
each cell's output directory from the snapshot id and the cell label, so ``--run``
holds the manifest and the plan rather than the results.
``<run>/queue/cell_plan.json`` is the same plan as data, with the staging report
and the skipped pairs attached.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.generation import objective_tuned_arms as ot  # noqa: E402
from src.capability.generation import unconditional_generation as ug  # noqa: E402

CAMPAIGN = ot.CAMPAIGN
KEY = "ot"
STAGE = "unconditional_generation.py"
EXPECT = ug.EXPECT_GENERATE

#: Independent streams per arm, from the declaration. A stream is a conjunction
#: requirement for a supported reading, not three chances at one.
STREAMS = 3

#: Cells per stream, by role. The base arm's four is the selection headroom the
#: explicit-statistics control needs: the alignment chose its positives from
#: 80,000 of its own samples, so a statistical rule ranking the base pool must be
#: allowed to keep a top quarter of a pool four times the tuned one.
CELLS_PER_STREAM: dict[str, int] = {"base": 4, "tuned": 1}

SEED_BASE, SEED_STEP = 20264011000, 1000

PRELUDE = "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8"


def cell_plan(pairs: tuple[ot.ObjectiveTunedPair, ...]) -> list[dict[str, Any]]:
    """Every cell in manifest order: pair, then stream, then base cells before tuned.

    Base cells lead each stream so that a campaign stopped early holds whole
    streams of the pool the explicit-statistics control filters, rather than a
    tuned pool with no control to read it against.

    The cells are seeded over every pair this is given, **declared rather than
    staged**, and :func:`scheduled_cells` drops the unstaged ones afterwards. A
    seed therefore depends only on a cell's position in the declared pair order,
    so finishing a download and rebuilding does not move the draw of a cell that
    has already run. Seeding the staged subset instead would have shifted every
    ProGen2 cell by fifteen positions the moment the 10B pair arrived.
    """

    cells: list[dict[str, Any]] = []
    for pair in pairs:
        for stream in range(STREAMS):
            for role, arm in (("base", pair.base), ("tuned", pair.tuned)):
                for replicate in range(CELLS_PER_STREAM[role]):
                    cells.append(
                        {
                            "pair": pair.tuned,
                            "arm": arm,
                            "role": role,
                            "secondary": pair.secondary,
                            "stream": stream,
                            "replicate": replicate,
                            "attempts": ug.POLICY["attempts"],
                        }
                    )
    for index, cell in enumerate(cells):
        cell["cell_index"] = index
        cell["seed"] = SEED_BASE + SEED_STEP * index
    return cells


def scheduled_cells(
    cells: list[dict[str, Any]], pairs: tuple[ot.ObjectiveTunedPair, ...]
) -> list[dict[str, Any]]:
    """The seeded cells of the pairs a schedule may reach, in the same order."""

    reachable = {pair.tuned for pair in pairs}
    return [cell for cell in cells if cell["pair"] in reachable]


def label(cell: dict[str, Any]) -> str:
    """The cell's name, which the queue also uses as its output directory.

    The arm name is carried verbatim, dot included: ``protgpt3-1.3b`` is the
    checkpoint's identity and ``protgpt3-13b`` would name a rung that does not
    exist.
    """

    return f"{KEY}_{cell['arm']}_s{cell['stream']}_r{cell['replicate']}"


def require_seed_separation(cells: list[dict[str, Any]], batch_size_of: Any) -> None:
    """Refuse a plan whose cells could draw the same batch seeds.

    The policy seeds each batch at ``seed + batch_index``, so a cell consumes as
    many consecutive seeds as it runs batches. Two cells closer than that share
    draws, which would turn a stream into a partial copy of its neighbour.
    """

    for cell in cells:
        batches = -(-int(cell["attempts"]) // int(batch_size_of(cell["arm"])))
        if batches > SEED_STEP:
            raise SystemExit(
                f"{label(cell)}: {batches} batches at a seed step of {SEED_STEP} "
                "would overlap the next cell's draw"
            )
    seeds = [cell["seed"] for cell in cells]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("the cell plan repeats a seed")


def row(slot: int, gpu: int | str, cell_label: str, env: str, args: str) -> str:
    return "\t".join((str(slot), KEY, str(gpu), STAGE, cell_label, env, EXPECT, args)) + "\n"


def dispatch_order(
    cells: list[dict[str, Any]], deferred: tuple[str, ...]
) -> list[dict[str, Any]]:
    """The order the queue dispatches cells in, with named pairs scheduled last.

    Dispatch order is not a declared property of the experiment. A cell's draw is
    fixed by its own seed, which :func:`cell_plan` ties to its position in the
    *declared* pair order, so moving a pair to the end of the queue changes which
    slot runs when and nothing else. Seeds, labels and arguments are untouched,
    and ``cell_plan.json`` keeps the declared order so that the seed-to-cell
    mapping stays readable independently of how a run was scheduled.

    It exists because a pair's weights can finish staging after the campaign
    starts. Deferring that pair lets the rest of the campaign run while the
    transfer completes, instead of reaching cells whose checkpoint is not yet on
    the cluster and failing them into a re-dispatch.
    """

    unknown = sorted(set(deferred) - {cell["pair"] for cell in cells})
    if unknown:
        raise SystemExit(f"--defer names pairs this campaign does not schedule: {unknown}")
    first = [cell for cell in cells if cell["pair"] not in deferred]
    last = [cell for cell in cells if cell["pair"] in deferred]
    return first + last


def generate_manifest(
    cells: list[dict[str, Any]],
    gpus: tuple[int, ...],
    env: str,
    skipped: dict[str, str],
    deferred: tuple[str, ...] = (),
) -> str:
    out = [
        f"# {CAMPAIGN}_generate: {len(cells)} cells of {ug.POLICY['attempts']} attempts "
        f"({STREAMS} streams per arm; {CELLS_PER_STREAM['base']} base cells per "
        f"{CELLS_PER_STREAM['tuned']} tuned cell), "
        f"{len(gpus)} per slot on GPUs {','.join(str(gpu) for gpu in gpus)}. "
        "--device and --out are injected.\n",
    ]
    for pair in deferred:
        out.append(
            f"# dispatched last: {pair} -- its cells keep their declared seeds and "
            "labels; only the slot they run in moved, so the rest of the campaign "
            "runs while this pair's weights finish staging\n"
        )
    for pair, reason in sorted(skipped.items()):
        out.append(f"# not scheduled: {pair} -- {reason}\n")
    out.append("# slot\tkey\tgpu\tstage\tlabel\tenv\texpect\targs\n")
    for index, cell in enumerate(cells):
        args = f"--stage generate --arm {cell['arm']} --seed {cell['seed']}"
        out.append(
            row(index // len(gpus) + 1, gpus[index % len(gpus)], label(cell), env, args)
        )
    return "".join(out)


def skipped_pairs() -> dict[str, str]:
    """Declared pairs this campaign does not schedule, each with its reason."""

    report = ot.staging_report()
    out: dict[str, str] = {}
    for pair in ot.PAIRS:
        if pair.lane != ot.LANE_FREE_LENGTH:
            out[pair.tuned] = (
                f"declared into the {pair.lane} lane rather than this one: {pair.note}"
            )
            continue
        incomplete = report["pairs"][pair.tuned]["incomplete_sides"]
        if incomplete:
            out[pair.tuned] = (
                f"weights incomplete on this host for {incomplete}; the pair is "
                "declared and not scheduled"
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT / "results/R6" / CAMPAIGN,
        help="run directory the manifest addresses",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="cards one slot occupies; the H200 pod's free allocation",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/lzp/miniconda3/envs/ct/bin/python"),
        help="TRANSFER_PYTHON for the env column; the pod's staged runtime in production",
    )
    parser.add_argument(
        "--defer",
        action="append",
        default=None,
        metavar="TUNED_PAIR",
        help="repeatable; dispatch this pair's cells last, keeping their seeds and labels",
    )
    args = parser.parse_args(argv)

    pairs = ot.staged_pairs(ot.LANE_FREE_LENGTH)
    skipped = skipped_pairs()
    if not pairs:
        raise SystemExit(
            "no free-length pair is staged on this host, so there is nothing to "
            f"schedule; skipped: {json.dumps(skipped, indent=1)}"
        )
    cells = scheduled_cells(cell_plan(ot.pairs_in_lane(ot.LANE_FREE_LENGTH)), pairs)
    require_seed_separation(cells, ug.arm_batch_size)
    for cell in cells:
        ug.generation_point(cell["arm"])

    queue = args.run / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    env = f"TRANSFER_PYTHON={args.python} {PRELUDE}"
    manifest = queue / f"campaign_{CAMPAIGN}_generate.tsv"
    deferred = tuple(args.defer or ())
    manifest.write_text(
        generate_manifest(
            dispatch_order(cells, deferred), tuple(args.gpus), env, skipped, deferred
        )
    )
    plan = {
        "campaign": CAMPAIGN,
        "declaration": "docs/objective-alignment.md",
        "policy": ug.POLICY,
        "streams": STREAMS,
        "cells_per_stream": CELLS_PER_STREAM,
        "seed_base": SEED_BASE,
        "seed_step": SEED_STEP,
        "pairs": [
            {
                "base": pair.base,
                "tuned": pair.tuned,
                "objective": pair.objective,
                "secondary": pair.secondary,
            }
            for pair in pairs
        ],
        "skipped_pairs": skipped,
        "dispatched_last": list(deferred),
        "staging": ot.staging_report(),
        "cells": cells,
    }
    (queue / "cell_plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")

    attempts: dict[str, int] = {}
    for cell in cells:
        attempts[cell["arm"]] = attempts.get(cell["arm"], 0) + int(cell["attempts"])
    print(f"{manifest}")
    print(f"cells: {len(cells)}  seeds: {cells[0]['seed']}..{cells[-1]['seed']}")
    for arm, total in attempts.items():
        print(f"  {arm:26s} {total} attempts")
    for pair, reason in sorted(skipped.items()):
        print(f"  not scheduled: {pair} -- {reason[:96]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
