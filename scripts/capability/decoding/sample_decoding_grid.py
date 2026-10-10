#!/usr/bin/env python3
"""Draw one arm's decoding grid at frozen weights, and account for the tokens.

One cell samples a shard of :data:`src.capability.decoding.decoding_sweep.GRID`
for one arm. Nothing is trained, loaded with an adapter, or re-prompted beyond
the arm's own declared form: the conditioning is held at the frozen class queue
for a conditioned arm and absent for the unconditioned one, so the only thing
that differs between two rows of the ledger is how tokens were drawn.

What the ledger carries beyond the sequence. The realised new-token count, which
is the compute currency of a decoding strategy and the reason a fixed-compute
comparison is possible at all; whether the arm's own end delimiter closed the
product or the cap did; and the model's log-probability of the product under the
untempered, untruncated distribution, which is the selector a realistic
best-of-k strategy can afford. No structural evaluation happens here: folding is
a separate stage on a separate runtime, so a GPU is never held loading a 6.6 B
predictor beside a generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import write_json  # noqa: E402
from src.capability.decoding import decoding_sweep as ds  # noqa: E402
from src.capability.generation import conditioned_generation as cg  # noqa: E402

COMPLETION = "decoding_grid_cell.json"
LEDGER = "attempts.jsonl"
DEFAULT_QUEUE = REPO_ROOT / "results/R6/conditioned_generation_queue_20260826/class_queue.json"

#: Batch size per arm. A feasibility parameter, not a scientific one, but the
#: per-batch seed rule makes the realised sample depend on it, so it is declared
#: here and recorded with the run rather than chosen at the call site.
BATCH_SIZE: dict[str, int] = {"protgpt2": 32, "zymctrl": 32, "prollama": 16}


def _resources(tag: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "tag": tag,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        record["torch"] = torch.__version__
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            record["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "free_mib": free // (1 << 20),
                "total_mib": total // (1 << 20),
            }
    except Exception as error:  # pragma: no cover - a receipt must never fail a run
        record["torch_error"] = repr(error)
    return record


def _load(name: str, *, device: str, dtype: str) -> Any:
    """The declared door this arm's weights come through, and nothing else."""

    spec = cg.arm(name)
    if spec.loader == "panel":
        from src.capability.core.arms import load_arm

        return load_arm(spec.checkpoint, device=device, dtype=dtype)
    if spec.loader == "lineage":
        from src.capability.models.joint_lineage import load_rung

        return load_rung(name, device=device, dtype=dtype)
    raise ValueError(f"{name}: unknown loader {spec.loader!r}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    swept = ds.arm(args.arm)
    settings = ds.config_shard(args.config_shard, args.num_config_shards)
    if args.only_config:
        wanted = set(args.only_config)
        unknown = sorted(wanted - set(ds.CONFIG_KEYS))
        if unknown:
            raise SystemExit(f"unknown configuration keys: {unknown}")
        settings = tuple(row for row in settings if row.key in wanted)
        if not settings:
            raise SystemExit(
                f"--only-config selected nothing in shard {args.config_shard} of "
                f"{args.num_config_shards}"
            )
    queue = cg.load_queue(args.queue) if swept.conditioned else None
    clusters = ds.clusters_for(args.arm, queue)

    args.out.mkdir(parents=True, exist_ok=True)
    start = _resources("start")
    write_json(args.out / "runtime-start.json", start)

    spec = cg.arm(args.arm)
    handle = _load(args.arm, device=args.device, dtype=ds.DTYPE)
    stop_ids = ds.terminator_ids(handle, spec)
    batch_size = args.batch_size or BATCH_SIZE[args.arm]
    started = time.monotonic()

    rows: list[dict[str, Any]] = []
    per_cell: list[dict[str, Any]] = []
    for setting in settings:
        for cluster, label in clusters:
            prompt = cg.prompt_for(handle, spec, label)
            seed = ds.cell_seed(arm_name=args.arm, config_key=setting.key, cluster=cluster)
            draws = ds.sample_configuration(
                handle,
                spec,
                prompt,
                setting=setting,
                n=setting.draws_per_cluster,
                seed=seed,
                batch_size=min(batch_size, setting.draws_per_cluster),
                stop_ids=stop_ids,
            )
            delimiter = cg.end_delimiter_for(handle, spec)
            for index, draw in enumerate(draws):
                sequence = cg.extract_protein(
                    draw["raw_continuation"], end_delimiter=delimiter
                )
                rows.append(
                    {
                        "id": ds.candidate_id(
                            arm_name=args.arm,
                            config_key=setting.key,
                            cluster=cluster,
                            draw_index=index,
                        ),
                        "schema_version": ds.SCHEMA_VERSION,
                        "campaign": ds.CAMPAIGN,
                        "role": "generated",
                        "arm": args.arm,
                        "config_key": setting.key,
                        "config_axis": setting.axis,
                        "temperature": setting.temperature,
                        "top_p": setting.top_p,
                        "top_k": setting.top_k,
                        "max_new_tokens": setting.max_new_tokens,
                        "min_new_tokens": setting.min_new_tokens,
                        "allow_terminator": setting.allow_terminator,
                        "cluster": cluster,
                        "prompt_label": label,
                        "draw_index": index,
                        "cell_seed": seed,
                        "batch_index": draw["batch_index"],
                        "batch_size": min(batch_size, setting.draws_per_cluster),
                        "sequence": sequence,
                        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                        "length": len(sequence),
                        "n_new_tokens": draw["n_new_tokens"],
                        "terminated_natively": draw["terminated_natively"],
                        "stop_token_id": draw["stop_token_id"],
                        "model_logprob_total": draw["model_logprob_total"],
                        "model_logprob_per_token": draw["model_logprob_per_token"],
                        "in_band": ds.in_band(sequence),
                        "out_of_band_reason": ds.out_of_band_reason(sequence),
                        "raw_continuation_length_chars": len(draw["raw_continuation"]),
                    }
                )
        cell_rows = [row for row in rows if row["config_key"] == setting.key]
        per_cell.append({"config": setting.record(), "census": ds.config_census(cell_rows)})
        print(
            json.dumps(
                {
                    "config": setting.key,
                    "n_draws": len(cell_rows),
                    "n_in_band": sum(1 for row in cell_rows if row["in_band"]),
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    digest = ds.write_jsonl(args.out / LEDGER, rows)
    end = _resources("end")
    write_json(args.out / "runtime-end.json", end)

    record = {
        "schema_version": ds.SCHEMA_VERSION,
        "status": "complete",
        "campaign": ds.CAMPAIGN,
        "arm": args.arm,
        "arm_note": swept.note,
        "conditioned": swept.conditioned,
        "modality": swept.modality,
        "config_shard": [args.config_shard, args.num_config_shards],
        "configurations": [setting.key for setting in settings],
        "n_configurations": len(settings),
        "clusters": [cluster for cluster, _ in clusters],
        "n_clusters": len(clusters),
        "queue_digest": None if queue is None else queue["digest"],
        "weights_are_frozen": (
            "no optimiser, adapter, or prompt search is constructed anywhere in this "
            "stage; only the sampler's arguments vary across configurations"
        ),
        "terminator_token_ids": list(stop_ids),
        "end_delimiter": cg.end_delimiter_for(handle, spec),
        "batch_size": batch_size,
        "dtype": ds.DTYPE,
        "add_special_tokens": ds.ADD_SPECIAL_TOKENS,
        "repetition_penalty": ds.REPETITION_PENALTY,
        "sampling_seed": ds.SAMPLING_SEED,
        "seed_rule": "decoding_sweep.cell_seed(arm, config, cluster) + batch_index",
        "n_attempts": len(rows),
        "n_in_band": sum(1 for row in rows if row["in_band"]),
        "n_distinct_sequences": len({row["sequence_sha256"] for row in rows}),
        "generation_tokens_total": sum(int(row["n_new_tokens"]) for row in rows),
        "per_configuration": per_cell,
        "attempts_jsonl": str(args.out / LEDGER),
        "attempts_sha256": digest,
        "elapsed_seconds": time.monotonic() - started,
        "device": args.device,
        "serving_provenance": getattr(handle, "serving_provenance", None),
        "runtime_start": start,
        "runtime_end": end,
        "evaluation": (
            "no structural, novelty or family evaluation happens here; the folding and "
            "analysis stages read this ledger"
        ),
        "ceiling": dict(ds.CEILING),
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=list(ds.ARM_NAMES))
    parser.add_argument("--config-shard", type=int, default=0)
    parser.add_argument("--num-config-shards", type=int, default=1)
    parser.add_argument(
        "--only-config",
        nargs="+",
        default=None,
        help="restrict this cell to these configuration keys, intersected with the shard",
    )
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = run(args)
    print(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "status",
                    "arm",
                    "n_configurations",
                    "n_attempts",
                    "n_in_band",
                    "n_distinct_sequences",
                    "generation_tokens_total",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
