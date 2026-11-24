#!/usr/bin/env python3
"""H200 GPU worker for Experiment 4.2: Eight-Domain Family Phase Space.

Supports two independently shardable phases across 6 H200 GPUs:

1. ``--phase generate`` (run under the ``ct`` environment, ``transformers 4.57.3``):
   - Shards the 19 new PLMs across ``--num-shards`` GPUs via deterministic cost-balanced
     bin-packing so each GPU loads its assigned models once and writes compact atomic JSONLs
     to ``<out-dir>/generated/<model>/<domain>__<arm>.jsonl``.
2. ``--phase fold`` (run under ``runtimes/esmfold2/bin/python``, ``transformers >= 5.16``):
   - Loads ``biohub/ESMFold2-hf`` once per GPU, verifies ``< 0.01`` TM parity on the 8
     sealed wild types against ``e20_wild_type_references_20261013/freeze/``, folds each
     unique candidate sequence once per shard, computes exact ``fold_geometry.tm_score``
     against the frozen WT CA trace plus ``plddt_mean``, ``ptm``, ``wt_identity_pct``,
     ``msa_max_identity_pct``, ``potts_energy_{full,fields,couplings}``, and
     ``shannon_H_bits``, and writes compact summaries to
     ``<out-dir>/folded/<model>/<domain>__<arm>.jsonl`` (zero bulk PDB files).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.context.family_potts import load_family_model  # noqa: E402
from src.capability.generation import fold_recapitulation as fr  # noqa: E402
from src.capability.generation import structure_evidence as se  # noqa: E402
from src.capability.ladder import fold_geometry as fg  # noqa: E402
from src.capability.unified import r6_experiment_4_2 as exp42  # noqa: E402

WT_PARITY_TM_THRESHOLD = 0.99  # < 0.01 TM deviation from 1.00 self-parity
FOLD_SEED_RULE = "sequence_digest"


def _data_root() -> Path:
    env = os.environ.get("TRANSFER_DATA_ROOT", "").strip()
    if env and Path(env).is_dir():
        return Path(env)
    return ROOT


def _resolve_path(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    alt = _data_root() / path
    if alt.exists():
        return alt.resolve()
    return path


def _resolve_esmfold2_dir(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    for var in ("ESMFOLD2_MODEL_DIR", "ESMFOLD2_DIR"):
        v = os.environ.get(var, "").strip()
        if v:
            candidates.append(Path(v))
    model_root = os.environ.get("MODEL_ROOT", "").strip()
    if model_root:
        candidates.append(Path(model_root) / "ESMFold2-hf")
    candidates.extend(
        [
            Path("/gpfs/jiaotongdamoxing/zhk_zip/models/ESMFold2-hf"),
            Path("/Data/lzp/models/ESMFold2-hf"),
            Path("/home/lzp/models/ESMFold2-hf"),
        ]
    )
    for cand in candidates:
        if cand.is_dir() and (cand / "config.json").is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"ESMFold2-hf checkpoint not found; searched {[str(c) for c in candidates]}"
    )


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _is_complete_jsonl(path: Path, expected_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            n = sum(1 for line in fh if line.strip())
        return n == expected_count
    except Exception:
        return False


def _extract_ca_coords(
    output: Any, features: dict[str, Any], sequence: str, sample_idx: int
) -> np.ndarray:
    from transformers.models.esmfold2.protein_utils import output_to_pdb

    pdb_text = output_to_pdb(output, features, sample_idx=sample_idx)
    ca = fg.parse_ca_trace(pdb_text)
    if ca.shape != (len(sequence), 3):
        raise ValueError(
            f"ESMFold2 CA shape {ca.shape} != expected ({len(sequence)}, 3)"
        )
    return ca


def run_generate_phase(args: argparse.Namespace) -> dict[str, Any]:
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else list(exp42.NEW_H200_MODELS)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in exp42.PHASE_SPACE_ARM_MAP:
            raise ValueError(f"Unknown arm {a!r}; expected subset of {exp42.PHASE_SPACE_ARM_KEYS}")

    shards = exp42.plan_generation_shards(models, args.num_shards)
    assigned_models = shards[args.shard_index]
    print(
        f"[exp4.2:generate shard {args.shard_index}/{args.num_shards}] "
        f"models={assigned_models} domains={len(domains)} arms={arms} pool={args.pool_size}",
        flush=True,
    )

    ref_dir = _resolve_path(args.family_reference_dir)
    family_models = {d: load_family_model(d, ref_dir) for d in domains}
    gen_root = args.out_dir / "generated"
    gen_root.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    cells_done = 0
    cells_skipped = 0

    for model_id in assigned_models:
        todo_cells = []
        for dom in domains:
            for arm_key in arms:
                out_jsonl = gen_root / model_id / f"{dom}__{arm_key}.jsonl"
                if _is_complete_jsonl(out_jsonl, args.pool_size) and not args.overwrite:
                    cells_skipped += 1
                else:
                    todo_cells.append((dom, arm_key, out_jsonl))
        if not todo_cells:
            print(f"[exp4.2:generate] {model_id}: all {len(domains) * len(arms)} cells cached, skipping load", flush=True)
            continue

        t_load = time.monotonic()
        backend, loaded, dtype_used = exp42.load_phase_space_model(
            model_id,
            device=args.device,
            dtype_override=args.dtype,
            model_root=args.model_root,
            esm3_model_dir=args.esm3_model_dir,
        )
        print(
            f"[exp4.2:generate] loaded {model_id} (backend={backend}, dtype={dtype_used}) "
            f"in {time.monotonic() - t_load:.1f}s; running {len(todo_cells)} cells",
            flush=True,
        )
        try:
            for dom, arm_key, out_jsonl in todo_cells:
                t_cell = time.monotonic()
                rows = exp42.generate_cell_records(
                    model_id=model_id,
                    backend=backend,
                    loaded=loaded,
                    family_model=family_models[dom],
                    arm_spec=exp42.PHASE_SPACE_ARM_MAP[arm_key],
                    count=args.pool_size,
                    batch_size=args.batch_size,
                    progen3_batch_size=args.progen3_batch_size,
                    base_seed=args.base_seed,
                    permutation_seed=exp42.PERMUTATION_SEED,
                )
                _atomic_write_jsonl(out_jsonl, rows)
                cells_done += 1
                print(
                    f"[exp4.2:generate] {model_id}/{dom}/{arm_key}: {len(rows)} seqs in "
                    f"{time.monotonic() - t_cell:.1f}s",
                    flush=True,
                )
        finally:
            del loaded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    receipt = {
        "phase": "generate",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "assigned_models": assigned_models,
        "cells_generated": cells_done,
        "cells_skipped_cached": cells_skipped,
        "elapsed_seconds": time.monotonic() - t_start,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = args.out_dir / f"receipt_generate_shard{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def run_fold_phase(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import EsmFold2Model
    from transformers.models.esmfold2.protein_utils import prepare_protein_features

    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else list(exp42.NEW_H200_MODELS)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    gen_root = args.out_dir / "generated"
    fold_root = args.out_dir / "folded"
    fold_root.mkdir(parents=True, exist_ok=True)

    all_cells = [(m, d, a) for m in models for d in domains for a in arms]
    assigned_cells = exp42.plan_fold_shards(all_cells, args.num_shards)[args.shard_index]

    todo_cells: list[tuple[str, str, str, Path, Path]] = []
    skipped = 0
    for m, d, a in assigned_cells:
        in_p = gen_root / m / f"{d}__{a}.jsonl"
        out_p = fold_root / m / f"{d}__{a}.jsonl"
        if not in_p.is_file():
            raise FileNotFoundError(
                f"Missing generated cell {in_p}; complete --phase generate before --phase fold"
            )
        if _is_complete_jsonl(out_p, args.pool_size) and not args.overwrite:
            skipped += 1
        else:
            todo_cells.append((m, d, a, in_p, out_p))

    print(
        f"[exp4.2:fold shard {args.shard_index}/{args.num_shards}] "
        f"assigned={len(assigned_cells)} todo={len(todo_cells)} cached={skipped}",
        flush=True,
    )

    ref_dir = _resolve_path(args.family_reference_dir)
    msa_dir = _resolve_path(args.msa_dir)
    wt_manifest_path = _resolve_path(args.wt_reference_manifest)

    family_models = {d: load_family_model(d, ref_dir) for d in domains}
    msas = {d: fr.load_msa(msa_dir, d, family_models[d].length) for d in domains}
    wt_ca_traces, wt_seqs, wt_manifest = exp42.load_frozen_wt_ca_traces(
        wt_manifest_path, domains=domains
    )
    wt_contract_seed = int(wt_manifest.get("instrument_contract", {}).get("seed", 20260905))

    t_start = time.monotonic()
    wt_parity_report: dict[str, dict[str, float]] = {}

    if todo_cells:
        esmfold2_dir = _resolve_esmfold2_dir(args.esmfold2_dir)
        se.require_esmfold2_checkpoint(esmfold2_dir)
        dev = torch.device(args.device)
        if dev.type == "cuda":
            torch.cuda.set_device(dev)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        t_load = time.monotonic()
        model = (
            EsmFold2Model.from_pretrained(esmfold2_dir, local_files_only=True, dtype=torch.bfloat16)
            .eval()
            .to(dev)
        )
        print(
            f"[exp4.2:fold] loaded ESMFold2-hf from {esmfold2_dir} in {time.monotonic() - t_load:.1f}s",
            flush=True,
        )

        def _fold_one(seq: str, *, seed_override: int | None = None) -> tuple[dict[str, float], np.ndarray]:
            seed = (
                int(seed_override)
                if seed_override is not None
                else se.fold_seed(seq, base_seed=args.base_seed, rule=FOLD_SEED_RULE)
            )
            torch.manual_seed(seed)
            if dev.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                features = prepare_protein_features(seq, device=dev)
                output = model.fold(**features, num_diffusion_samples=int(args.diffusion_samples))
            summary = se.summarize_esmfold2(output, sequence=seq)
            sample_idx = int(summary["diffusion_sample_index"])
            ca = _extract_ca_coords(output, features, seq, sample_idx)
            metrics = {
                "plddt_mean": float(summary["mean_ca_plddt"]),
                "ptm": float(summary["ptm"]),
                "mean_pae_angstrom": float(summary["mean_pae_angstrom"]),
                "diffusion_sample_index": sample_idx,
            }
            del output, features
            return metrics, ca

        # Verify < 0.01 TM instrument parity on WT sequences under the frozen WT seed contract.
        if not args.skip_wt_parity_check:
            for dom in domains:
                wt_metrics, wt_ca_refold = _fold_one(
                    wt_seqs[dom], seed_override=wt_contract_seed
                )
                tm_vs_frozen = float(fg.tm_score(wt_ca_refold, wt_ca_traces[dom])["tm_score"])
                wt_parity_report[dom] = {
                    "tm_vs_frozen_wt": tm_vs_frozen,
                    "plddt_mean": wt_metrics["plddt_mean"],
                    "ptm": wt_metrics["ptm"],
                }
                if tm_vs_frozen < WT_PARITY_TM_THRESHOLD:
                    raise RuntimeError(
                        f"WT instrument parity failed on {dom}: refolded vs frozen WT TM={tm_vs_frozen:.4f} "
                        f"< {WT_PARITY_TM_THRESHOLD}"
                    )
            print(
                "[exp4.2:fold] WT parity verified (<0.01 TM): min_tm="
                f"{min(v['tm_vs_frozen_wt'] for v in wt_parity_report.values()):.4f}",
                flush=True,
            )

        # Fold cache keyed by (domain, sequence_sha256) within this shard to deduplicate repeats.
        fold_cache: dict[tuple[str, str], dict[str, float]] = {}
        total_folded_unique = 0

        try:
            for idx_cell, (m, dom, arm_key, in_p, out_p) in enumerate(todo_cells, 1):
                t_c = time.monotonic()
                gen_rows = exp42._read_jsonl(in_p)
                fmodel = family_models[dom]
                msa_rows = msas[dom]
                wt_ca = wt_ca_traces[dom]
                folded_rows: list[dict[str, Any]] = []

                for rec in gen_rows:
                    seq = str(rec["sequence"])
                    sha = str(rec.get("sequence_sha256") or se.sequence_digest(seq))
                    ckey = (dom, sha)
                    struct_metrics = fold_cache.get(ckey)
                    if struct_metrics is None:
                        s_m, mobile_ca = _fold_one(seq)
                        tm_val = float(fg.tm_score(mobile_ca, wt_ca)["tm_score"])
                        struct_metrics = {
                            "tm_score": tm_val,
                            "plddt_mean": s_m["plddt_mean"],
                            "ptm": s_m["ptm"],
                            "mean_pae_angstrom": s_m["mean_pae_angstrom"],
                        }
                        fold_cache[ckey] = struct_metrics
                        total_folded_unique += 1

                    seq_metrics = exp42.enrich_sequence_metrics(
                        seq, family_model=fmodel, msa_rows=msa_rows
                    )
                    folded_rows.append(
                        {
                            "schema_version": exp42.FOLDED_RECORD_SCHEMA,
                            "id": rec["id"],
                            "model": m,
                            "domain": dom,
                            "arm": arm_key,
                            "guidance_arm": rec["guidance_arm"],
                            "lambda_fields": rec["lambda_fields"],
                            "lambda_coupling": rec["lambda_coupling"],
                            "seed": rec["seed"],
                            "sequence_index": rec["sequence_index"],
                            "sequence": seq,
                            "sequence_sha256": sha,
                            "length": len(seq),
                            "guidance_score": rec.get("guidance_score"),
                            "guidance_bias_applied": rec.get("guidance_bias_applied"),
                            "mean_step_entropy": rec.get("mean_step_entropy"),
                            "sampler": rec.get("sampler"),
                            "source": "h200_exp4_2_esmfold2",
                            **struct_metrics,
                            **seq_metrics,
                        }
                    )

                _atomic_write_jsonl(out_p, folded_rows)
                if idx_cell % 5 == 0 or idx_cell == len(todo_cells):
                    print(
                        f"[exp4.2:fold shard {args.shard_index}/{args.num_shards}] "
                        f"cell {idx_cell}/{len(todo_cells)} ({m}/{dom}/{arm_key}) "
                        f"in {time.monotonic() - t_c:.1f}s (unique_folded={total_folded_unique})",
                        flush=True,
                    )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    receipt = {
        "phase": "fold",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "cells_assigned": len(assigned_cells),
        "cells_folded": len(todo_cells),
        "cells_skipped_cached": skipped,
        "wt_parity": wt_parity_report,
        "elapsed_seconds": time.monotonic() - t_start,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = args.out_dir / f"receipt_fold_shard{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=("generate", "fold", "both"), required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out-dir", type=Path, default=exp42.DEFAULT_WORK_DIR)
    ap.add_argument("--family-reference-dir", type=Path, default=exp42.DEFAULT_FAMILY_REF_DIR)
    ap.add_argument("--msa-dir", type=Path, default=exp42.DEFAULT_MSA_DIR)
    ap.add_argument("--wt-reference-manifest", type=Path, default=exp42.DEFAULT_WT_REF_MANIFEST)
    ap.add_argument("--esmfold2-dir", type=Path, default=None)
    ap.add_argument("--model-root", type=Path, default=None)
    ap.add_argument("--esm3-model-dir", type=Path, default=None)
    ap.add_argument("--models", type=str, default=None, help="Optional comma-separated model subset")
    ap.add_argument("--domains", type=str, default=",".join(exp42.SEALED_8_DOMAINS))
    ap.add_argument("--arms", type=str, default=",".join(exp42.PHASE_SPACE_ARM_KEYS))
    ap.add_argument("--pool-size", type=int, default=exp42.DEFAULT_POOL_SIZE)
    ap.add_argument("--batch-size", type=int, default=exp42.DEFAULT_BATCH_SIZE)
    ap.add_argument("--progen3-batch-size", type=int, default=exp42.DEFAULT_PROGEN3_BATCH_SIZE)
    ap.add_argument("--dtype", type=str, default=None, help="Optional dtype override for generate")
    ap.add_argument("--diffusion-samples", type=int, default=32)
    ap.add_argument("--base-seed", type=int, default=exp42.DEFAULT_BASE_SEED)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-wt-parity-check", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"--shard-index ({args.shard_index}) must lie in [0, {args.num_shards})"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in ("generate", "both"):
        run_generate_phase(args)
    if args.phase in ("fold", "both"):
        run_fold_phase(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
