#!/usr/bin/env python3
"""Recompute a generating checkpoint's own likelihood on its own frozen products.

E17 asks whether the predictive information a model carries can be used to pick
better products out of its own output. The quantity that asks that question is
the model's likelihood of the sequence it produced -- and it does not exist
anywhere in this project: the frozen generation ledgers retain token *ids* and a
stop reason, never a log-probability. So it is recomputed here, on the fixed pool
the cohort stage froze, under the arm's own native rendering.

Rendering is the whole correctness risk. A ProtGPT2 sequence scored as one
unwrapped line costs 1.42 nats/token more than the same sequence in the FASTA
layout it was pretrained on, and ZymCTRL's EC tag leaks 1.73 nats if it is scored
as content instead of as a conditioning prompt. Both are larger than any
selection effect this experiment could find, so the rendering, the target mask
and the excluded marker or boundary spans all come from the audited declarations
in :mod:`src.capability.core.arms` and :mod:`src.capability.core.scoring` rather
than being spelled again here.

The score is a mean negative log-likelihood per *scored token* in the arm's own
tokenisation. It is comparable within an arm, which is all selection needs, and
it is not comparable across arms, which is why no cross-arm likelihood number is
emitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402

COMPLETION = "generated_likelihood.json"
SCHEMA_VERSION = "d1_generated_likelihood_v1"


def load_pool(path: Path, arm: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("arm") != arm or "pool" not in record.get("roles", []):
                continue
            rows.append(record)
    if not rows:
        raise SystemExit(
            f"{path} carries no pool member for arm {arm!r}. Scoring an empty pool is "
            "refused rather than reported as a complete run"
        )
    identifiers = [row["id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise SystemExit("the pool carries duplicate identifiers")
    return sorted(rows, key=lambda row: str(row["id"]))


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from src.capability.core import scoring
    from src.capability.core.arms import (
        PANEL,
        Cohort,
        conditioning_boundary_ids,
        load_arm,
        rendering_marker_ids,
        tokenize_batch,
    )

    gp.require_fresh_out(args.out, COMPLETION)
    if args.arm not in PANEL:
        raise SystemExit(
            f"{args.arm!r} is not a panel member. A generating checkpoint whose native "
            f"rendering is not declared cannot be scored; panel is {sorted(PANEL)}"
        )
    rows = load_pool(args.cohort, args.arm)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("likelihood scoring requires an assigned CUDA GPU")

    arm = load_arm(args.arm, device=args.device, dtype=args.dtype)
    labels: list[str] | None = None
    if arm.spec.input_format == "ec_conditioned":
        labels = [str(row.get("class_key") or "") for row in rows]
        missing = [row["id"] for row, label in zip(rows, labels) if not label]
        if missing:
            raise SystemExit(
                f"{args.arm} renders an EC-conditioned prompt, but {len(missing)} pool "
                f"members carry no class_key (first {missing[:3]}). Scoring them without "
                "the tag they were generated under would measure a different quantity"
            )
    cohort = Cohort(
        name=f"e17_pool_{args.arm}",
        kind="protein",
        records=[str(row["sequence"]) for row in rows],
        min_symbols=min(int(row["length"]) for row in rows),
        max_symbols=max(int(row["length"]) for row in rows),
        metadata=(
            {"sampling": {"mode": "frozen_pool", "source": str(args.cohort)}}
            | ({"ec_labels": labels} if labels is not None else {})
        ),
    )
    rendered = cohort.input_strings(arm)
    rule = scoring.target_rule(arm.spec.input_format, ec_conditioning="native")
    start_id, end_id = conditioning_boundary_ids(arm, ec_conditioning="native")
    markers = () if arm.spec.input_format == "ec_conditioned" else rendering_marker_ids(arm)

    scored: list[dict[str, Any]] = []
    truncated: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            chunk = list(range(start, min(start + args.batch_size, len(rows))))
            ids, mask = tokenize_batch(arm, [rendered[index] for index in chunk], args.max_len)
            target_mask = scoring.sequence_target_mask(
                ids,
                mask,
                rule=rule,
                start_token_id=start_id,
                end_token_id=end_id,
                marker_token_ids=markers,
            )
            logits = arm.model(
                input_ids=ids.to(arm.device), attention_mask=mask.to(arm.device)
            ).logits.float().cpu()
            per_sequence = scoring.per_sequence_scores(logits, logits, ids, target_mask)
            for offset, index in enumerate(chunk):
                row = rows[index]
                entry = per_sequence[offset]
                tokens = int(entry["token_count"])
                # A rendering that hit --max-len lost residues from its own tail,
                # so its mean is over a prefix and is not the score of the product.
                if int(mask[offset].sum()) >= args.max_len:
                    truncated.append(str(row["id"]))
                scored.append(
                    {
                        "id": str(row["id"]),
                        "arm": args.arm,
                        "class_key": row.get("class_key"),
                        "length": int(row["length"]),
                        "scored_tokens": tokens,
                        "nll_sum_nats": float(entry["clean_nll_sum"]),
                        "mean_nll_per_token_nats": float(entry["clean_nll_sum"]) / tokens,
                        "mean_nll_per_residue_nats": float(entry["clean_nll_sum"]) / int(row["length"]),
                    }
                )
            print(
                json.dumps({"scored": len(scored), "total": len(rows)}),
                flush=True,
            )
    if truncated:
        raise SystemExit(
            f"{len(truncated)} pool members hit --max-len {args.max_len} (first "
            f"{truncated[:3]}): their likelihood is the likelihood of a prefix, not of "
            "the product, and a selection score built on it would rank truncations. "
            "Raise --max-len"
        )

    sidecar = args.out / "generated_likelihood.jsonl"
    sidecar.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored), encoding="utf-8"
    )
    values = [row["mean_nll_per_token_nats"] for row in scored]
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "arm": args.arm,
        "n_scored": len(scored),
        "cohort": str(args.cohort),
        "cohort_sha256": sha256_file(args.cohort),
        "pool_digest": gp.cohort_digest(rows),
        "rendering": {
            "input_format": arm.spec.input_format,
            "target_rule": rule,
            "conditioning_boundary_ids": [start_id, end_id],
            "marker_token_ids": list(markers),
            "tokenisation": arm.spec.tokenisation,
            "ec_conditioning": "native" if labels is not None else "not_applicable",
        },
        "device": args.device,
        "dtype": args.dtype,
        "max_len": int(args.max_len),
        "batch_size": int(args.batch_size),
        "mean_nll_per_token_nats": {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        },
        "annotations_jsonl": str(sidecar),
        "annotations_sha256": sha256_file(sidecar),
        "interpretation": (
            "mean negative log-likelihood per scored token under the arm's own native "
            "rendering and tokenisation. Comparable within this arm, which is what "
            "selection needs; not comparable to another arm's nats per token"
        ),
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True, help="the frozen evaluation cohort JSONL")
    parser.add_argument("--arm", required=True, help="the generating panel checkpoint to score with")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_len < 2:
        parser.error("--batch-size must be positive and --max-len at least two")
    record = run(args)
    print(json.dumps({key: record[key] for key in ("status", "arm", "n_scored")}, sort_keys=True))


if __name__ == "__main__":
    main()
