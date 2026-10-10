#!/usr/bin/env python3
"""One arm's likelihood of the ladder's sequences, including its parents.

The scoring process never sees a structure and the folding process never sees a
likelihood. That independence is the whole reason the rank correlation between
them means anything, so the two are separate stages reading the same frozen
sequences and nothing more.

The scalar depends on the arm's paradigm and is labelled accordingly. A causal
arm gives a sequence negative log-likelihood under its own native rendering,
summed over the positions that rendering declares scorable. A masked or
absorbing-state arm has no sequence likelihood at all; its scalar is a sum of
masked marginals, one position masked at a time, and the artefact calls it a
pseudo-log-likelihood. Neither is comparable across arms and no cross-arm
magnitude is emitted.

Every parent is scored too, so each variant carries both its own likelihood and
the change relative to the backbone it came from. Within one backbone the two
induce the same ranking -- the parent is a constant there -- which is said here
rather than discovered by a reader who finds two identical correlations.
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

COMPLETION = "ladder_likelihood.json"
RECORDS = "ladder_likelihood.jsonl"
SCHEMA_VERSION = "ladder_likelihood_v1"


def collect(variant_paths: list[Path], backbones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every distinct scorable sequence, parents first.

    A sequence is scored once however many variant rows carry it: two draws that
    produced the same string are one likelihood and two sampling observations,
    and scoring it twice would only spend a forward pass on proving that.
    """

    jobs: dict[str, dict[str, Any]] = {}
    for backbone in backbones:
        jobs[backbone["parent_sequence_sha256"]] = {
            "sequence": backbone["sequence"],
            "sequence_sha256": backbone["parent_sequence_sha256"],
            "role": "parent",
            "ec_label": backbone["ec_label"],
            "backbone_ids": [backbone["backbone_id"]],
        }
    for path in variant_paths:
        for row in design.read_jsonl(path):
            if row["status"] != "filled" or not row["sequence"]:
                continue
            digest = row["sequence_sha256"]
            entry = jobs.get(digest)
            if entry is None:
                jobs[digest] = {
                    "sequence": row["sequence"],
                    "sequence_sha256": digest,
                    "role": "variant",
                    "ec_label": row["ec_label"],
                    "backbone_ids": [row["backbone_id"]],
                }
            elif row["backbone_id"] not in entry["backbone_ids"]:
                entry["backbone_ids"].append(row["backbone_id"])
    if not jobs:
        raise SystemExit("no scorable sequence was found in the supplied variant files")
    ordered = sorted(jobs.values(), key=lambda entry: (entry["role"] != "parent", entry["sequence_sha256"]))
    return ordered


def run(args: argparse.Namespace) -> dict[str, Any]:
    design.require_fresh_out(args.out, COMPLETION)
    spec = design.arm(args.arm)
    backbones = design.read_jsonl(args.backbones)
    jobs = collect(list(args.variants), backbones)

    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("likelihood scoring requires an assigned CUDA GPU")

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    if spec.likelihood_kind == design.LIKELIHOOD_CAUSAL_NLL:
        handle = runtime.load_causal_arm(args.arm, device=args.device, dtype=args.dtype)
        facts = handle.facts
        for start in range(0, len(jobs), args.batch_size):
            chunk = jobs[start : start + args.batch_size]
            scored = runtime.causal_sequence_nll(
                handle,
                [entry["sequence"] for entry in chunk],
                ec_labels=[entry["ec_label"] for entry in chunk],
                batch_size=args.batch_size,
                max_len=args.max_len,
            )
            for entry, result in zip(chunk, scored):
                rows.append({**entry, **result, "scored_by": args.arm,
                             "likelihood_kind": design.LIKELIHOOD_CAUSAL_NLL})
            print(
                json.dumps(
                    {"scored": len(rows), "total": len(jobs),
                     "elapsed_seconds": round(time.monotonic() - started, 1)}
                ),
                flush=True,
            )
    elif spec.likelihood_kind == design.LIKELIHOOD_MASKED_PLL:
        if args.model_root is None:
            raise SystemExit(f"{args.arm} is loaded from a staged directory and needs --model-root")
        handle = runtime.load_masked_arm(args.arm, model_root=args.model_root, device=args.device)
        facts = handle.facts
        for index, entry in enumerate(jobs, 1):
            result = runtime.masked_pseudo_log_likelihood(
                handle, entry["sequence"], batch_size=args.batch_size
            )
            rows.append({**entry, **result, "scored_by": args.arm})
            if index % 25 == 0 or index == len(jobs):
                print(
                    json.dumps(
                        {"scored": index, "total": len(jobs),
                         "elapsed_seconds": round(time.monotonic() - started, 1)}
                    ),
                    flush=True,
                )
    else:
        raise SystemExit(f"{args.arm} declares no likelihood readout of its own")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    digest = design.write_jsonl(out / RECORDS, rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arm": args.arm,
        "model_class": spec.model_class,
        "likelihood_kind": spec.likelihood_kind,
        "n_sequences": len(rows),
        "n_parents": sum(1 for row in rows if row["role"] == "parent"),
        "records": RECORDS,
        "records_sha256": digest,
        "arm_facts": facts,
        "variants": {str(path): sha256_file(path) for path in args.variants},
        "backbones": {"path": str(args.backbones), "sha256": sha256_file(args.backbones)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
        },
        "ceiling": {
            "within_arm_only": (
                "a magnitude here is comparable within this arm and within this "
                "rendering. It is not comparable to another arm's and no cross-arm "
                "number is emitted"
            ),
            "pll_is_not_a_likelihood": design.CEILING["pll_is_not_a_likelihood"],
            "independence": (
                "no structural quantity enters this stage. The likelihood and the "
                "structural evaluation of one sequence are computed by different "
                "checkpoints in different processes"
            ),
        },
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    write_json(out / COMPLETION, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbones", type=Path, required=True)
    parser.add_argument("--variants", type=Path, nargs="+", required=True)
    parser.add_argument("--arm", required=True, choices=sorted(set(design.CAUSAL_ARMS) | set(design.INFILL_ARMS)))
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=1024)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"n_sequences": payload["n_sequences"]}))


if __name__ == "__main__":
    main()
