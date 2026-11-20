#!/usr/bin/env python3
"""R1 natural-fragment ``family_distribution`` exact O(L) leave-one-out PLL
(and exact causal NLL) runner with multi-GPU family sharding.

Reads the frozen 68-unit (4 pilot + 64 evaluation) queue at
``results/extensions/family_distribution_20261106/queue.json`` verbatim,
reconstructs the exact frozen column-permutation and composition-shuffle nulls
via ``src.capability.extensions.family_distribution``, and supports clean
``--shard-index`` / ``--num-shards`` splitting so heavy arms like ``esm2-15b``
can run across 4 H200 GPUs in parallel.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("TRANSFER_DATA_ROOT") or ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.extensions import family_distribution as fd_mod
from src.capability.general_plm import list_arms, load_general_plm
from src.capability.general_plm.eval_utils import (
    atomic_write_json,
    cluster_bootstrap_mean,
    sha256_file,
)

SCHEMA = "general_plm_r1_family_distribution_pll_v1"


def _score_sequences(adapter, sequences: list[str], *, batch_size: int) -> np.ndarray:
    """Return per-sequence total NLL (nats) via exact O(L) PLL or exact causal NLL."""
    nlls = np.empty(len(sequences), dtype=np.float64)
    for idx, seq in enumerate(sequences):
        nlls[idx] = float(adapter.exact_pll(seq, batch_size=batch_size).nll_sum)
    return nlls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", required=True, choices=list_arms())
    parser.add_argument("--dtype", default="auto", help="float32 (default) or explicit bf16 opt-in")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--saprot-mode", default="seq_only", choices=("seq_only", "foldseek_3di"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--partition", default="all", choices=("all", "pilot", "evaluation"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-units", type=int, default=0, help="0 = all units in shard")
    parser.add_argument(
        "--queue",
        type=Path,
        default=DATA_ROOT / "results/extensions/family_distribution_20261106/queue.json",
    )
    parser.add_argument("--expect-file", default="")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit(f"Invalid shard {args.shard_index}/{args.num_shards}")

    args.out.mkdir(parents=True, exist_ok=True)
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    if queue.get("schema") != fd_mod.SCHEMA_QUEUE:
        raise SystemExit(f"Unexpected queue schema: {queue.get('schema')!r}")

    protocol = queue["protocol"]
    col_seeds: list[int] = [int(s) for s in protocol["null"]["column_shuffle_seeds"]]
    queue_seed = int(protocol["queue"]["seed"])

    units = sorted(queue["units"], key=lambda u: u["unit_id"])
    if args.partition != "all":
        units = [u for u in units if u["partition"] == args.partition]
    shard_units = [u for idx, u in enumerate(units) if idx % args.num_shards == args.shard_index]
    if args.max_units > 0:
        shard_units = shard_units[: args.max_units]
    if not shard_units:
        raise SystemExit("Shard selected zero units; refusing to write empty artifact")

    adapter = load_general_plm(
        args.arm,
        device=args.device,
        dtype=args.dtype,
        model_root=args.model_root,
        saprot_mode=args.saprot_mode,
    )

    t0 = time.monotonic()
    unit_results: list[dict[str, Any]] = []
    eval_col_deltas: list[float] = []
    eval_comp_deltas: list[float] = []
    eval_ids: list[str] = []
    total_sequences = 0

    for u_idx, unit in enumerate(shard_units, 1):
        wt_entry = unit["wildtypes"][0]
        rows = fd_mod.evaluation_rows(wt_entry)
        natives = [r.native for r in rows]
        lengths = np.asarray([len(s) for s in natives], dtype=np.float64)

        # Build frozen column-permutation nulls and verify structural invariants.
        col_null_sets: list[list[str]] = []
        for seed in col_seeds:
            shuffled = fd_mod.column_null(rows, seed)
            fd_mod.verify_column_null(rows, shuffled)
            col_null_sets.append(shuffled)
        comp_nulls = [fd_mod.composition_null(row, queue_seed + i) for i, row in enumerate(rows)]

        native_nll = _score_sequences(adapter, natives, batch_size=args.batch_size)
        col_null_nlls = np.stack(
            [_score_sequences(adapter, sset, batch_size=args.batch_size) for sset in col_null_sets],
            axis=0,
        )
        comp_null_nll = _score_sequences(adapter, comp_nulls, batch_size=args.batch_size)
        total_sequences += len(natives) * (1 + len(col_null_sets) + 1)

        col_contrast = fd_mod.contrast(native_nll, col_null_nlls, lengths)
        comp_contrast = fd_mod.contrast(native_nll, comp_null_nll[None, :], lengths)

        rec = {
            "unit_id": unit["unit_id"],
            "wildtype_id": wt_entry["wildtype_id"],
            "partition": unit["partition"],
            "n_native": len(natives),
            "total_residues": int(lengths.sum()),
            "column_contrast": col_contrast,
            "composition_contrast": comp_contrast,
            "native_nll_nats": native_nll.tolist(),
            "column_null_nll_nats": col_null_nlls.tolist(),
            "composition_null_nll_nats": comp_null_nll.tolist(),
        }
        unit_results.append(rec)
        if unit["partition"] == "evaluation":
            eval_col_deltas.append(float(col_contrast["nats_per_residue"]))
            eval_comp_deltas.append(float(comp_contrast["nats_per_residue"]))
            eval_ids.append(unit["unit_id"])

        print(
            f"[R1-famdist:{adapter.arm} shard {args.shard_index}/{args.num_shards}] "
            f"{u_idx}/{len(shard_units)} {unit['unit_id']} "
            f"col_delta={col_contrast['nats_per_residue']:+.4f} "
            f"comp_delta={comp_contrast['nats_per_residue']:+.4f}",
            flush=True,
        )

    elapsed = time.monotonic() - t0
    suffix = (
        f"_shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
    )
    out_name = args.expect_file or f"r1_family_distribution_{args.arm}{suffix}.json"
    payload = {
        "schema": SCHEMA,
        "roster": "general_plm_20261120_exploratory",
        "arm": args.arm,
        "score_semantics": (
            "exact_O_L_leave_one_out_masked_pll_within_model_delta_only"
            if adapter.spec.is_masked
            else "exact_causal_sequence_nll"
        ),
        "queue_sha256": sha256_file(args.queue),
        "partition": args.partition,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_units": len(unit_results),
        "n_sequences_scored": int(total_sequences),
        "elapsed_seconds": elapsed,
        "provenance": adapter.provenance(),
        "evaluation_summary": {
            "column_null_delta_nats_per_res": cluster_bootstrap_mean(
                eval_col_deltas, eval_ids, seed=20261106
            ),
            "composition_null_delta_nats_per_res": cluster_bootstrap_mean(
                eval_comp_deltas, eval_ids, seed=20261107
            ),
        },
        "units": unit_results,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(args.out / out_name, payload)
    print(f"[R1-famdist:{args.arm}] wrote {args.out / out_name}", flush=True)


if __name__ == "__main__":
    main()
