#!/usr/bin/env python3
"""Causal prefix regeneration of the ladder's windows, and the unmatched anchor.

One arm per cell. For every backbone and every declared extent the arm is
prompted with its own rendering of the backbone cut at the first residue of the
centred window, writes forward with no sight of what follows, and the first ``k``
residues it wrote are spliced back in place of the original ``k``. The original
suffix is then restored, so the length is the parent's to the residue and the
model's choice was made without the downstream context.

A draw that ended before writing ``k`` residues has not expressed the rung. It
is written out with its reason and an empty sequence, so the denominator of every
later rate is the number of draws attempted and not the number that happened to
work.

With ``--include-anchor`` the same cell also samples the full-generation rung:
whole sequences from the arm's bare native prompt. That rung is an anchor and not
a matched comparison -- its length is whatever the arm produces and it has no
parent fold -- and it is labelled as one in every row.
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
    if spec.condition != design.CONDITION_CAUSAL_PREFIX:
        raise SystemExit(
            f"{args.arm} declares condition {spec.condition!r}; this stage only runs the "
            f"causal prefix condition. A causal arm cannot express true infilling and an "
            f"infilling arm must not be prompted as if it could"
        )
    backbones = design.read_jsonl(args.backbones)
    rungs = list(args.rungs) if args.rungs else [f"k{extent}" for extent in design.WINDOW_EXTENTS]
    unknown = [rung for rung in rungs if rung not in design.RUNGS or rung == design.FULL_GENERATION]
    if unknown:
        raise SystemExit(f"not window rungs of this ladder: {unknown}")

    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("causal window generation requires an assigned CUDA GPU")
    handle = runtime.load_causal_arm(args.arm, device=args.device, dtype=args.dtype)

    rows: list[dict[str, Any]] = []
    alignment: dict[str, Any] = {}
    unalignable: list[str] = []
    started = time.monotonic()
    widest = max(design.rung_extent(rung) for rung in rungs)
    for backbone in backbones:
        ec_label = backbone["ec_label"] if spec.needs_ec_label else None
        # One aligned start per backbone, shared by every extent, so the windows
        # stay nested and the only thing a rung changes is how much is rewritten.
        aligned = runtime.aligned_prefix(
            handle,
            backbone["sequence"],
            design.window_anchor(backbone["length"]),
            ec_label=ec_label,
            max_extent=widest,
        )
        if aligned is None:
            unalignable.append(backbone["backbone_id"])
            print(
                json.dumps({"backbone": backbone["backbone_id"], "unalignable": True}),
                flush=True,
            )
            continue
        start = aligned["start"]
        alignment[backbone["backbone_id"]] = {
            key: value for key, value in aligned.items() if key != "prompt"
        }
        for rung in rungs:
            extent = design.rung_extent(rung)
            span = design.window_span(backbone["length"], extent, start=start)
            seed = design.cell_seed(
                arm_name=args.arm, backbone_id=backbone["backbone_id"], rung=rung, draw=0
            )
            draws = runtime.sample_window(
                handle,
                prompt=aligned["prompt"],
                extent=extent,
                draws=design.DRAWS_PER_CELL,
                seed=seed,
                batch_size=args.batch_size,
            )
            for entry in draws:
                sequence = (
                    design.splice(backbone["sequence"], start, entry["window"])
                    if entry["status"] == runtime.DRAW_FILLED
                    else ""
                )
                rows.append(
                    design.variant_record(
                        arm_name=args.arm,
                        condition=spec.condition,
                        backbone=backbone,
                        rung=rung,
                        draw=entry["draw"],
                        sequence=sequence,
                        status=entry["status"],
                        window_span=span,
                        extra={
                            "n_residues_written": entry["n_residues_written"],
                            "prompt_sha256": entry["prompt_sha256"],
                            "prompt_characters": entry["prompt_characters"],
                            "n_prompt_tokens": aligned["n_prompt_tokens"],
                            "window_start_residues_from_anchor": aligned["residues_from_anchor"],
                        },
                    )
                )
            print(
                json.dumps(
                    {
                        "backbone": backbone["backbone_id"],
                        "rung": rung,
                        "window_start": start,
                        "filled": sum(1 for e in draws if e["status"] == runtime.DRAW_FILLED),
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
    if not rows:
        raise SystemExit(
            f"{args.arm}: no backbone admitted an aligned prefix prompt inside the declared "
            f"{design.TOKEN_ALIGNMENT_RADIUS}-residue radius. An empty arm is reported as a "
            "refusal, not as a completed cell"
        )

    anchor_rows: list[dict[str, Any]] = []
    if args.include_anchor:
        for backbone in backbones:
            ec_label = backbone["ec_label"] if spec.needs_ec_label else None
            seed = design.cell_seed(
                arm_name=args.arm,
                backbone_id=backbone["backbone_id"],
                rung=design.FULL_GENERATION,
                draw=0,
            )
            draws = runtime.sample_full(
                handle,
                draws=design.DRAWS_PER_CELL,
                seed=seed,
                ec_label=ec_label,
                batch_size=args.batch_size,
            )
            for entry in draws:
                anchor_rows.append(
                    design.variant_record(
                        arm_name=args.arm,
                        condition=design.CONDITION_FULL_GENERATION,
                        backbone=backbone,
                        rung=design.FULL_GENERATION,
                        draw=entry["draw"],
                        sequence=entry["sequence"],
                        status=entry["status"],
                        extra={
                            "n_residues_written": entry["n_residues_written"],
                            "terminated": entry["terminated"],
                            "prompt_sha256": entry["prompt_sha256"],
                            "anchor_note": design.CEILING["the_anchor_is_not_matched"],
                        },
                    )
                )
            print(json.dumps({"anchor_for": backbone["backbone_id"], "n": len(draws)}), flush=True)
    rows.extend(anchor_rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    digest = design.write_jsonl(out / RECORDS, rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['rung']}|{row['status']}"
        status_counts[key] = status_counts.get(key, 0) + 1
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm": args.arm,
        "model_class": spec.model_class,
        "condition": spec.condition,
        "rungs": rungs + ([design.FULL_GENERATION] if args.include_anchor else []),
        "draws_per_cell": design.DRAWS_PER_CELL,
        "n_backbones": len(backbones),
        "n_backbones_with_aligned_prefix": len(alignment),
        "backbones_without_aligned_prefix": unalignable,
        "token_alignment_radius": design.TOKEN_ALIGNMENT_RADIUS,
        "window_alignment": alignment,
        "n_rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "records": RECORDS,
        "records_sha256": digest,
        "arm_facts": handle.facts,
        "sampling": {
            "temperature": design.TEMPERATURE,
            "top_p": design.TOP_P,
            "top_k": design.TOP_K,
            "campaign_seed": design.CAMPAIGN_SEED,
            "batch_size": args.batch_size,
        },
        "backbones": {
            "path": str(args.backbones),
            "sha256": sha256_file(args.backbones),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "device": args.device,
            "dtype": args.dtype,
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
    parser.add_argument("--backbones", type=Path, required=True, help="ladder_backbones.jsonl")
    parser.add_argument("--arm", required=True, choices=sorted(design.CAUSAL_ARMS))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rungs", nargs="*", default=None)
    parser.add_argument("--include-anchor", action="store_true")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"n_rows": payload["n_rows"], "status_counts": payload["status_counts"]}))


if __name__ == "__main__":
    main()
