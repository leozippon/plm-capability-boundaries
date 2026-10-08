#!/usr/bin/env python3
"""Frozen ESM2 embeddings for every row of the small-domain stability tables.

One mean-pooled vector per sequence, taken over residue positions only. The
model is never fine-tuned and never sees a free energy: the whole predictor is a
linear head fitted later on these frozen vectors, which is what keeps the
training cheap enough to validate properly and keeps the backbone identical
between the training fold, the cross-dataset check and the generated cohort.

Two details that decide correctness. The pooling excludes the classification and
end-of-sequence tokens, because a mean that includes them mixes two learned
constants into every vector and does so unevenly across lengths. And the pooling
layer is not instantiated: ``EsmModel`` would otherwise build a randomly
initialised pooler, and this project has already been bitten once by reading a
randomly initialised head as a measurement.
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402

COMPLETION = "stability_embeddings.json"
SCHEMA_VERSION = "d1_stability_embeddings_v1"


def load_rows(path: Path, splits: list[str] | None) -> list[dict[str, Any]]:
    rows = []
    wanted = set(splits) if splits else None
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if wanted is not None and row.get("split") not in wanted:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path} yielded no row for splits {splits}")
    identifiers = [row["id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise SystemExit("the stability table carries duplicate identifiers")
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Before any weight is read: a dirty output directory is a cheap failure and
    # must not cost a checkpoint load first.
    gp.require_fresh_out(args.out, COMPLETION)

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    rows = load_rows(args.rows, args.splits)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("embedding extraction requires an assigned CUDA GPU")
    torch.cuda.set_device(torch.device(args.device))
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=getattr(torch, args.dtype),
        # Refuse the randomly initialised pooler rather than load and ignore it.
        add_pooling_layer=False,
    )
    model = model.eval().to(device)
    width = int(model.config.hidden_size)

    # Longest first, so the first batch is the worst case for memory and a run
    # that will not fit fails in its first seconds rather than near the end.
    order = sorted(range(len(rows)), key=lambda i: (-int(rows[i]["length"]), str(rows[i]["id"])))
    embeddings = np.zeros((len(rows), width), dtype=np.float16)
    started = time.monotonic()
    done = 0
    with torch.inference_mode():
        for start in range(0, len(order), args.batch_size):
            chunk = order[start : start + args.batch_size]
            batch = tokenizer(
                [str(rows[i]["sequence"]) for i in chunk], return_tensors="pt", padding=True
            )
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
            residues = mask.clone()
            residues[:, 0] = 0
            residues[torch.arange(ids.shape[0], device=device), mask.sum(1) - 1] = 0
            counts = residues.sum(1, keepdim=True)
            if int(counts.min()) < 1:
                raise RuntimeError("a batch row has no residue position left after masking")
            pooled = (hidden * residues.unsqueeze(-1)).sum(1) / counts
            block = pooled.float().cpu().numpy()
            if not np.isfinite(block).all():
                raise RuntimeError("a non-finite embedding was produced")
            for offset, index in enumerate(chunk):
                embeddings[index] = block[offset].astype(np.float16)
            done += len(chunk)
            if start % (args.batch_size * 50) == 0 or done == len(order):
                print(
                    json.dumps(
                        {
                            "embedded": done,
                            "total": len(order),
                            "seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    # A residue-length check on the pooling: the number of pooled positions must
    # equal the sequence length for every row, verified on a sample rather than
    # asserted, because an off-by-one here shifts every vector.
    sample = order[: min(32, len(order))]
    for index in sample:
        encoded = tokenizer(str(rows[index]["sequence"]))["input_ids"]
        if len(encoded) - 2 != int(rows[index]["length"]):
            raise RuntimeError(
                f"{rows[index]['id']}: tokenizer emits {len(encoded)} ids for a "
                f"{rows[index]['length']}-residue sequence; the two special tokens the "
                "pooling removes do not account for the difference"
            )

    array_path = args.out / "embeddings.npy"
    np.save(array_path, embeddings)
    index_path = args.out / "embedding_index.json"
    write_json(
        index_path,
        {
            "schema_version": SCHEMA_VERSION,
            "ids": [str(row["id"]) for row in rows],
            "splits": [str(row["split"]) for row in rows],
            "lengths": [int(row["length"]) for row in rows],
        },
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": str(args.rows),
        "rows_sha256": sha256_file(args.rows),
        "model": str(args.model),
        "model_id_note": "frozen protein language model; no parameter is updated here",
        "n_rows": len(rows),
        "width": width,
        "dtype_stored": "float16",
        "dtype_compute": args.dtype,
        "pooling": (
            "mean over residue positions of the last hidden state, excluding the "
            "classification, end-of-sequence and padding tokens"
        ),
        "pooler_instantiated": False,
        "batch_size": int(args.batch_size),
        "device": args.device,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "embeddings_npy": str(array_path),
        "embeddings_sha256": sha256_file(array_path),
        "index_json": str(index_path),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "python": platform.python_version(),
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="domain_stability_rows.jsonl")
    parser.add_argument("--model", type=Path, required=True, help="a staged ESM2 checkpoint")
    parser.add_argument("--splits", nargs="*", default=None, help="restrict to these splits")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    record = run(args)
    print(json.dumps({key: record[key] for key in ("status", "n_rows", "width")}, sort_keys=True))


if __name__ == "__main__":
    main()
