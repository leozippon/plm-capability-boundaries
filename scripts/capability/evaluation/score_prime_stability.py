#!/usr/bin/env python3
"""Score the stability tables with a third-party thermostability checkpoint.

Why a second instrument at all. The predictor this experiment fits is trained on
a cDNA-display proteolysis assay and checked against another cDNA-display
proteolysis assay, so the check establishes transfer across sequence populations
and not across measurement technology. A checkpoint trained by other people on
other data is the only cross-technology corroboration available.

What this checkpoint is, stated plainly because its own card does not. The model
card shipped with it is an unedited template: it declares no training data, no
intended use, no units and no license. Its scalar head is an attention-pooled
projection to one number, and on a four-sequence probe it scored a poly-glycine
chain *above* natural ubiquitin, which is not what a folding free energy does. So
its output is treated as an ordinal score of unknown units, is never converted to
kcal/mol, and -- this is the point -- it is put through the **same** validation
gate as the fitted head. An undocumented checkpoint does not get to corroborate
anything until it has shown that it ranks measured free energy.

The checkpoint's vendored modelling code targets an older transformers release.
The required names are rebound in-process by
:func:`src.capability.evaluation.domain_stability.transformers_compatibility_shim`,
nothing under the model directory is edited, and what was rebound is recorded in
this stage's artefact.
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
from src.capability.evaluation import domain_stability as ds  # noqa: E402
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402

COMPLETION = "prime_stability_scores.json"
SCHEMA_VERSION = "d1_prime_stability_scores_v1"

#: The splits this instrument is scored on. Not ``train``: the head is somebody
#: else's and is never fitted here, so the training fold would only cost time.
DEFAULT_SPLITS: tuple[str, ...] = ("validation", "validation_online", "test", "transfer", "apply")


def run(args: argparse.Namespace) -> dict[str, Any]:
    gp.require_fresh_out(args.out, COMPLETION)
    shim = ds.transformers_compatibility_shim()

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    wanted = set(args.splits)
    rows = []
    with args.rows.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") in wanted:
                rows.append(row)
    if not rows:
        raise SystemExit(f"{args.rows} yielded no row for splits {sorted(wanted)}")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("this checkpoint is scored on an assigned CUDA GPU")
    torch.cuda.set_device(torch.device(args.device))
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        dtype=getattr(torch, args.dtype),
    )
    model = model.eval().to(device)

    order = sorted(range(len(rows)), key=lambda i: (-int(rows[i]["length"]), str(rows[i]["id"])))
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(order), args.batch_size):
            chunk = order[start : start + args.batch_size]
            batch = tokenizer(
                [str(rows[i]["sequence"]) for i in chunk], return_tensors="pt", padding=True
            )
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            values = getattr(output, "predicted_values", None)
            if values is None:
                raise SystemExit(
                    "this checkpoint's forward pass returned no predicted_values field, "
                    "so it carries no scalar head and cannot serve as a second opinion"
                )
            block = values.float().cpu().numpy().reshape(-1)
            if block.size != len(chunk) or not np.isfinite(block).all():
                raise RuntimeError("the scalar head returned a non-finite or misshapen batch")
            for offset, index in enumerate(chunk):
                scores[index] = float(block[offset])
            if start % (args.batch_size * 20) == 0:
                print(
                    json.dumps({"scored": start + len(chunk), "total": len(order)}), flush=True
                )
    if not np.isfinite(scores).all():
        raise RuntimeError("a row was left unscored")

    sidecar = args.out / "prime_stability_scores.jsonl"
    sidecar.write_text(
        "".join(
            json.dumps(
                {
                    "id": str(row["id"]),
                    "split": row["split"],
                    "length": int(row["length"]),
                    "prime_value_head": float(score),
                },
                sort_keys=True,
            )
            + "\n"
            for row, score in zip(rows, scores)
        ),
        encoding="utf-8",
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": str(args.rows),
        "rows_sha256": sha256_file(args.rows),
        "model": str(args.model),
        "n_scored": len(rows),
        "splits": sorted(wanted),
        "quantity": "prime_value_head",
        "quantity_declaration": dict(ds.QUANTITIES["prime_value_head"]),
        "provenance_warning": (
            "the checkpoint's model card is an unedited template: it declares no "
            "training data, no intended use, no units and no license. Its scalar head "
            "is therefore used for rank agreement only, it is never converted to "
            "kcal/mol, and it must clear the same validation gate as the fitted head "
            "before its agreement is read as corroboration"
        ),
        "third_party_code": {
            "loaded_with_trust_remote_code": True,
            "compatibility_shim": shim,
        },
        "score_summary": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        },
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": int(args.batch_size),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "scores_jsonl": str(sidecar),
        "scores_sha256": sha256_file(sidecar),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "python": platform.python_version(),
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="domain_stability_rows.jsonl")
    parser.add_argument("--model", type=Path, required=True, help="the staged PRIME checkpoint")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--dtype", default="float32", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    record = run(args)
    print(json.dumps({key: record[key] for key in ("status", "n_scored", "score_summary")}, sort_keys=True))


if __name__ == "__main__":
    main()
