#!/usr/bin/env python3
"""The extent reference: windows no model wrote, drawn from the parent's composition.

Without this arm, a structural decline along the ladder says only that cutting
``k`` residues out of a protein and putting different ones back damages the fold,
which is true for any ``k`` and any replacement. This arm fixes that baseline. At
each extent the window is replaced by residues drawn with replacement from the
parent's **own** amino-acid frequencies, so composition is matched by
construction and only the identity and order of the window residues are
randomised.

It needs no GPU and no checkpoint. Its rows are scored by every causal arm later,
which turns it into a second question as well: does an arm's likelihood track
structure on sequences the arm did not produce? A likelihood that only ranks its
own samples is a different capability from one that ranks sequences.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.ladder import design  # noqa: E402

COMPLETION = "ladder_variants.json"
RECORDS = "ladder_variants.jsonl"
SCHEMA_VERSION = "ladder_variants_v1"
ARM = "composition-shuffle"


def draw_window(parent: str, extent: int, *, seed: int) -> str:
    """``extent`` residues drawn from the parent's own composition."""

    residues = sorted(set(parent))
    weights = np.asarray([parent.count(residue) for residue in residues], dtype=np.float64)
    weights /= weights.sum()
    generator = np.random.default_rng(int(seed))
    return "".join(generator.choice(residues, size=int(extent), p=weights))


def run(args: argparse.Namespace) -> dict[str, Any]:
    design.require_fresh_out(args.out, COMPLETION)
    spec = design.arm(ARM)
    backbones = design.read_jsonl(args.backbones)
    rows: list[dict[str, Any]] = []
    for backbone in backbones:
        for extent in design.WINDOW_EXTENTS:
            rung = f"k{extent}"
            start, _stop = backbone["window_spans"][rung]
            for draw in range(design.DRAWS_PER_CELL):
                seed = design.cell_seed(
                    arm_name=ARM, backbone_id=backbone["backbone_id"], rung=rung, draw=draw
                )
                window = draw_window(backbone["sequence"], extent, seed=seed)
                rows.append(
                    design.variant_record(
                        arm_name=ARM,
                        condition=spec.condition,
                        backbone=backbone,
                        rung=rung,
                        draw=draw,
                        sequence=design.splice(backbone["sequence"], start, window),
                        status="filled",
                        extra={
                            "window_source": "multinomial draw from the parent composition",
                            "composition_matched": True,
                        },
                    )
                )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    digest = design.write_jsonl(out / RECORDS, rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm": ARM,
        "model_class": spec.model_class,
        "condition": spec.condition,
        "rungs": [f"k{extent}" for extent in design.WINDOW_EXTENTS],
        "draws_per_cell": design.DRAWS_PER_CELL,
        "n_backbones": len(backbones),
        "n_rows": len(rows),
        "records": RECORDS,
        "records_sha256": digest,
        "backbones": {"path": str(args.backbones), "sha256": sha256_file(args.backbones)},
        "ceiling": dict(design.CEILING),
        "note": spec.note,
    }
    write_json(out / COMPLETION, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="accepted and unused; no checkpoint is loaded")
    parser.add_argument("--backbones", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"n_rows": payload["n_rows"]}))


if __name__ == "__main__":
    main()
