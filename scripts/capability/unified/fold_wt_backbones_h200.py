#!/usr/bin/env python3
"""GPU driver to fold pending wild-type/context backbones with ESMFold2 on H200.

Reads ``pending_wt_esmfold2.jsonl`` produced by ``build_wt_3d_backbones.py``,
filters by ``--tiers`` and deterministic ``--shard-index / --num-shards``, folds
each unique sequence window with ``biohub/ESMFold2-hf`` (``bfloat16`` under the
``esmfold2`` runtime), parses exact ``[N, CA, C, O]`` backbone coordinates from
``output_to_pdb`` via ``parse_pdb_backbone_lines``, and appends JSONL records
directly consumable by ``build_wt_3d_backbones.py --ingest-folded``.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.generation import structure_evidence as se  # noqa: E402
from src.capability.unified.wt_backbones_3d import parse_pdb_backbone_lines  # noqa: E402

SEED_RULE = "sequence_digest"
DEFAULT_BASE_SEED = 20261125


def _load_done_job_ids(out_jsonl: Path) -> set[str]:
    done: set[str] = set()
    if not out_jsonl.is_file():
        return done
    with out_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                continue
            jid = rec.get("job_id")
            if jid:
                done.add(str(jid))
    return done


def _round_nested(arr: np.ndarray, ndigits: int) -> list[Any]:
    return np.round(arr.astype(np.float64), ndigits).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fold pending WT/context backbones via ESMFold2 on H200.")
    parser.add_argument("--pending-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--esmfold2-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tiers", type=str, default="tier_1_r1_r4_wt", help="Comma-separated priority_tier values or 'all'")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--diffusion-samples", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    args = parser.parse_args()

    from transformers import EsmFold2Model
    from transformers.models.esmfold2.protein_utils import output_to_pdb, prepare_protein_features

    tier_filter = None if args.tiers.strip().lower() == "all" else {t.strip() for t in args.tiers.split(",") if t.strip()}
    all_rows: list[dict[str, Any]] = []
    with args.pending_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            if tier_filter is not None and row.get("priority_tier") not in tier_filter:
                continue
            all_rows.append(row)

    # Interleaved modulo assignment across length-sorted jobs gives balanced shards
    shard_rows = [r for idx, r in enumerate(all_rows) if idx % args.num_shards == args.shard_index]
    done_ids = _load_done_job_ids(args.out_jsonl)
    todo = [r for r in shard_rows if str(r["job_id"]) not in done_ids]

    print(
        f"[wt-fold] tiers={args.tiers} shard={args.shard_index}/{args.num_shards} "
        f"total={len(shard_rows)} done={len(done_ids)} todo={len(todo)} on {args.device}",
        flush=True,
    )
    if not todo:
        print("[wt-fold] nothing to do; exiting cleanly.", flush=True)
        return

    esmfold2_dir = args.esmfold2_dir.resolve()
    se.require_esmfold2_checkpoint(esmfold2_dir)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    t0 = time.monotonic()
    model = (
        EsmFold2Model.from_pretrained(esmfold2_dir, local_files_only=True, dtype=torch.bfloat16)
        .eval()
        .to(dev)
    )
    print(f"[wt-fold] ESMFold2 loaded in {time.monotonic() - t0:.1f}s", flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with args.out_jsonl.open("a", encoding="utf-8") as out_fh:
        for idx, row in enumerate(todo, 1):
            seq = str(row["sequence"]).strip().upper()
            seed = se.fold_seed(seq, base_seed=int(args.base_seed), rule=SEED_RULE)
            torch.manual_seed(seed)
            if dev.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            ts = time.monotonic()
            feats = prepare_protein_features(seq, device=dev)
            with torch.inference_mode():
                output = model.fold(**feats, num_diffusion_samples=int(args.diffusion_samples))
            summ = se.summarize_esmfold2(output, sequence=seq)
            s_idx = int(summ["diffusion_sample_index"])
            pdb_text = output_to_pdb(output, feats, sample_idx=s_idx)
            if isinstance(pdb_text, list):
                pdb_text = pdb_text[0]
            parsed_seq, coords, _ = parse_pdb_backbone_lines(pdb_text.splitlines())
            if parsed_seq != seq or coords.shape != (len(seq), 4, 3):
                raise ValueError(
                    f"Backbone mismatch for {row['job_id']}: parsed_len={len(parsed_seq)} vs {len(seq)}, shape={coords.shape}"
                )
            plddt = np.asarray(summ["ca_plddt"], dtype=np.float32)
            if plddt.shape != (len(seq),):
                raise ValueError(f"pLDDT mismatch for {row['job_id']}: {plddt.shape} != ({len(seq)},)")

            rec = {
                "job_id": str(row["job_id"]),
                "parent_seq_sha256": str(row["parent_seq_sha256"]),
                "parent_length": int(row["parent_length"]),
                "sequence": seq,
                "length": len(seq),
                "window_index": int(row.get("window_index", 0)),
                "window_start": int(row["window_start"]),
                "window_end": int(row["window_end"]),
                "num_windows": int(row["num_windows"]),
                "priority_tier": str(row["priority_tier"]),
                "coords": _round_nested(coords, 3),
                "plddt": _round_nested(plddt, 2),
                "mean_ca_plddt": round(float(summ["mean_ca_plddt"]), 3),
                "ptm": round(float(summ["ptm"]), 4),
                "fold_seconds": round(time.monotonic() - ts, 3),
                "source_path": f"esmfold2_h200:s{int(args.diffusion_samples)}",
            }
            out_fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            out_fh.flush()
            n_ok += 1

            del output, feats
            if len(seq) >= 600 and dev.type == "cuda":
                torch.cuda.empty_cache()

            if idx % 25 == 0 or idx == len(todo):
                elapsed = time.monotonic() - t0
                print(
                    f"[wt-fold] shard={args.shard_index}/{args.num_shards} {idx}/{len(todo)} "
                    f"({elapsed:.1f}s, last_len={len(seq)}, plddt={rec['mean_ca_plddt']:.1f})",
                    flush=True,
                )

    gc.collect()
    print(f"[wt-fold] DONE shard={args.shard_index}/{args.num_shards} folded={n_ok} elapsed={time.monotonic() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
