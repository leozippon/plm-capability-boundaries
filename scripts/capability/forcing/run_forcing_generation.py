#!/usr/bin/env python3
"""E19 generation: force one anchor residue and let the model finish to the partner.

One arm per invocation, one cell per (unit, condition, mode). The token grid is
measured before anything is generated and the arm is refused if a position-level
forcing intervention is not defined on it, so an inadmissible tokenisation is a
refusal at the start rather than a silently approximated measurement.

Generation stops at the furthest read position the unit needs, which is why the
gate is cheap: a draw emits a few tens of residues rather than a whole protein.
The teacher-forced mode costs one forward pass per cell and returns the exact
conditional at every read position, which is the design's free secondary
decomposition.

Cells are resumable individually. Each completed cell appends one record to
``cells.jsonl`` and writes its per-draw distributions to its own ``.npy``, so an
interrupted arm resumes where it stopped instead of regenerating hours of
sampling; the summary artefact is written only when every cell is present.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.arms import PANEL, arm_spec, load_arm_spec  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.forcing import forcing_design as D  # noqa: E402
from src.capability.forcing.backbone_cohort import iter_cells, load_cohort  # noqa: E402
from src.capability.forcing.forced_completion import (  # noqa: E402
    residue_grid,
    sample_cell,
    teacher_forced_cell,
)

COMPLETION = "forcing_generation.json"
CELLS = "cells.jsonl"
OBJECTS = "objects"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cell_key(cell: dict, mode: str) -> str:
    return f"{cell['unit_id'].replace(':', '_')}__{cell['condition']}__{mode}"


def load_completed(path: Path) -> dict[str, dict]:
    """Cell records already written, keyed by cell. A record without its array is not done."""

    if not path.is_file():
        return {}
    done: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        array = path.parent / OBJECTS / f"{record['cell']}.npy"
        if array.is_file():
            done[record["cell"]] = record
    return done


def resolve_arm(name: str, *, device: str, dtype: str):
    """Load a panel member or a declared staged checkpoint, through the declared door."""

    spec = PANEL[name] if name in PANEL else arm_spec(name)
    if spec.modality != "protein":
        raise SystemExit(f"{name} is a {spec.modality} arm; this design forces protein residues")
    return load_arm_spec(spec, device=device, dtype=dtype)


def run(args: argparse.Namespace) -> None:
    cohort = load_cohort(args.cohort)
    if args.arm not in D.ARMS and not args.allow_unlisted_arm:
        raise SystemExit(
            f"{args.arm} is not one of this experiment's declared arms {D.ARMS}. "
            f"Unselected arms and their reasons: {json.dumps(D.UNSELECTED_ARMS)}. "
            "Pass --allow-unlisted-arm to run a probe outside the declared panel; its "
            "output is then marked as such and is not part of the panel."
        )
    arm = resolve_arm(args.arm, device=args.device, dtype=args.dtype)
    grid = residue_grid(arm)

    cells = list(iter_cells(cohort))
    if args.num_shards > 1:
        cells = [cell for index, cell in enumerate(cells) if index % args.num_shards == args.shard]
    if args.limit:
        cells = cells[: args.limit]
    modes = D.MODES if args.mode == "both" else (args.mode,)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / OBJECTS).mkdir(exist_ok=True)
    journal = args.out / CELLS
    done = load_completed(journal)
    sequences = cohort["sequences"]

    with journal.open("a", encoding="utf-8") as handle:
        for cell in cells:
            for mode in modes:
                key = cell_key(cell, mode)
                if key in done:
                    continue
                wildtype = sequences[cell["accession"]]
                seed = D.cell_seed(
                    arm=args.arm, unit=cell["unit_id"], condition=cell["condition"], mode=mode,
                )
                positions = [cell["partner"], *cell["reference_positions"]]
                if mode == D.MODE_SAMPLED:
                    produced = sample_cell(
                        arm, grid, wildtype=wildtype, anchor=cell["anchor"],
                        forced_residue=cell["forced_residue"], span_end=cell["span_end"],
                        read_positions=positions, draws=args.draws, seed=seed,
                        batch_size=args.batch_size,
                    )
                else:
                    produced = teacher_forced_cell(
                        arm, grid, wildtype=wildtype, anchor=cell["anchor"],
                        forced_residue=cell["forced_residue"], span_end=cell["span_end"],
                        read_positions=positions,
                    )
                array = np.asarray(produced.pop("distributions"), dtype=np.float32)
                np.save(args.out / OBJECTS / f"{key}.npy", array)
                record = {
                    "cell": key,
                    "unit_id": cell["unit_id"],
                    "accession": cell["accession"],
                    "length_band": cell["length_band"],
                    "condition": cell["condition"],
                    "partner": cell["partner"],
                    "reference_positions": cell["reference_positions"],
                    **produced,
                }
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                done[key] = record

    expected = {cell_key(cell, mode) for cell in cells for mode in modes}
    missing = sorted(expected - set(done))
    if missing:
        raise SystemExit(f"{len(missing)} cell(s) did not complete, e.g. {missing[:3]}")
    sampled = [record for record in done.values() if record["mode"] == D.MODE_SAMPLED]
    write_json(args.out / COMPLETION, {
        "schema": "forcing_generation_v1",
        "status": "complete",
        "created_utc": _now(),
        "experiment": D.EXPERIMENT,
        "pre_registration": D.PRE_REGISTRATION,
        "arm": args.arm,
        "in_declared_panel": args.arm in D.ARMS,
        "token_grid": dict(grid.evidence),
        "device": args.device,
        "dtype": args.dtype,
        "modes": list(modes),
        "cells": len(done),
        "shard": {"index": int(args.shard), "count": int(args.num_shards)},
        "draws_per_cell": int(args.draws),
        "decoding": {
            "temperature": D.TEMPERATURE, "top_p": D.TOP_P, "top_k": D.TOP_K,
            "note": D.DECODING_POLICY_NOTE,
        },
        "sampling_seed": D.SAMPLING_SEED,
        "censoring": {
            "rule": (
                "the first generated token that is not one of the twenty canonical "
                "residues terminates the completion; read positions at or after it are "
                "censored and the terminating token is recorded"
            ),
            "censored_draws": sum(record.get("censored_draws", 0) for record in sampled),
            "total_draws": sum(len(record.get("draws", ())) for record in sampled),
            "cells_with_any_censoring": sum(
                1 for record in sampled if record.get("censored_draws", 0)
            ),
        },
        "cohort": {"path": str(args.cohort), "sha256": sha256_file(args.cohort)},
        "journal": CELLS,
        "objects": OBJECTS,
        "interpretation": D.INTERPRETATION,
    })
    print(json.dumps({
        "arm": args.arm, "cells": len(done),
        "censored_draws": sum(record.get("censored_draws", 0) for record in sampled),
        "artefact": str(args.out / COMPLETION),
    }))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--mode", choices=(*D.MODES, "both"), default="both")
    parser.add_argument("--dtype", default=D.DTYPE)
    parser.add_argument("--draws", type=int, default=D.DRAWS_PER_CELL)
    parser.add_argument("--batch-size", type=int, default=D.DRAWS_PER_CELL)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-unlisted-arm", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
