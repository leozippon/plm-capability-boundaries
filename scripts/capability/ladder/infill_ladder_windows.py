#!/usr/bin/env python3
"""True fixed-length infilling of the ladder's windows, with both flanks visible.

The condition a left-to-right decoder cannot express. The ``k`` window positions
of the backbone are masked and filled by iterative unmasking -- one position per
step, re-reading the whole partially filled chain each time, committing whichever
position the model was most confident about -- so every committed residue is
conditioned on both flanks and on every earlier commitment.

This is the counterpart to ``generate_ladder_windows.py`` and is never averaged
with it. The two conditions condition on different information, and the asymmetry
between them is one of the things the ladder is for.

The sampler is a declared member of the family, not a reference release sampler:
the reference ``byprot`` loader has never been executed in this project, so
nothing here is supported by a numerical A/B against it. The same sampler is used
for the masked and for the diffusion arm, so the pair differs by training and not
by decoder.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.ladder import design, runtime  # noqa: E402

COMPLETION = "ladder_variants.json"
RECORDS = "ladder_variants.jsonl"
SCHEMA_VERSION = "ladder_variants_v1"


def run(args: argparse.Namespace) -> dict[str, Any]:
    design.require_fresh_out(args.out, COMPLETION)
    spec = design.arm(args.arm)
    if spec.condition != design.CONDITION_BIDIRECTIONAL_INFILL:
        raise SystemExit(
            f"{args.arm} declares condition {spec.condition!r}; only an arm that sees "
            "both flanks may run the true-infilling condition"
        )
    backbones = design.read_jsonl(args.backbones)
    rungs = list(args.rungs) if args.rungs else [f"k{extent}" for extent in design.WINDOW_EXTENTS]
    unknown = [rung for rung in rungs if rung not in design.RUNGS or rung == design.FULL_GENERATION]
    if unknown:
        raise SystemExit(f"not window rungs of this ladder: {unknown}")

    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("infilling requires an assigned CUDA GPU")
    handle = runtime.load_masked_arm(args.arm, model_root=args.model_root, device=args.device)

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for backbone in backbones:
        for rung in rungs:
            extent = design.rung_extent(rung)
            start, _stop = backbone["window_spans"][rung]
            for draw in range(design.DRAWS_PER_CELL):
                seed = design.cell_seed(
                    arm_name=args.arm,
                    backbone_id=backbone["backbone_id"],
                    rung=rung,
                    draw=draw,
                )
                filled = runtime.infill_window(
                    handle,
                    backbone["sequence"],
                    start=start,
                    extent=extent,
                    seed=seed,
                )
                rows.append(
                    design.variant_record(
                        arm_name=args.arm,
                        condition=spec.condition,
                        backbone=backbone,
                        rung=rung,
                        draw=draw,
                        sequence=design.splice(backbone["sequence"], start, filled["window"]),
                        status=filled["status"],
                        extra={
                            "sampler": filled["sampler"],
                            "sampler_support": filled["support"],
                            "commit_order": filled["commit_order"],
                            "reference_loader_executed": False,
                        },
                    )
                )
            print(
                json.dumps(
                    {
                        "backbone": backbone["backbone_id"],
                        "rung": rung,
                        "rows": len(rows),
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    digest = design.write_jsonl(out / RECORDS, rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm": args.arm,
        "model_class": spec.model_class,
        "condition": spec.condition,
        "rungs": rungs,
        "draws_per_cell": design.DRAWS_PER_CELL,
        "n_backbones": len(backbones),
        "n_rows": len(rows),
        "records": RECORDS,
        "records_sha256": digest,
        "arm_facts": handle.facts,
        "sampling": {
            "temperature": design.TEMPERATURE,
            "top_p": design.TOP_P,
            "campaign_seed": design.CAMPAIGN_SEED,
            "member": "one_position_per_step_max_confidence_iterative_unmasking",
        },
        "backbones": {"path": str(args.backbones), "sha256": sha256_file(args.backbones)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": "float32",
        },
        "ceiling": dict(design.CEILING),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    write_json(out / COMPLETION, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbones", type=Path, required=True)
    parser.add_argument("--arm", required=True, choices=sorted(design.INFILL_ARMS))
    parser.add_argument("--model-root", type=Path, required=True, help="directory holding the staged checkpoints")
    parser.add_argument("--rungs", nargs="*", default=None)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"n_rows": payload["n_rows"]}))


if __name__ == "__main__":
    main()
