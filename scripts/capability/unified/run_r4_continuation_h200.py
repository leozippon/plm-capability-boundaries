#!/usr/bin/env python3
"""H200 GPU worker for Workstream R4-Continuation (31 Models × 22 Targets × 5 Conditions).

Phases:
1. ``--phase generate`` (run under ``ct`` Python; run ``esm3-1.4b`` under isolated ``ESM3_SITE``):
   Generates all 80 ``(target_id, condition)`` cells × 32 attempts = 2,560 attempts per model
   and writes compact atomic JSONLs to ``<out-dir>/generated/<model>.jsonl``.
2. ``--phase fold_wt`` (run once under ``runtimes/esmfold2/bin/python``):
   Folds the 22 target wildtypes with ``biohub/ESMFold2-hf`` and saves their CA coordinates
   to ``<out-dir>/wt_ca_traces.npz`` + ``<out-dir>/wt_folds.json``.
3. ``--phase fold`` (run under ``runtimes/esmfold2/bin/python``, shardable across GPUs):
   Selects up to 2 valid non-copy candidates per ``(model, target, condition)``, folds each
   unique sequence once per shard with ``ESMFold2-hf``, computes exact ``structural_tm_score``
   against the target WT CA trace plus ``plddt_mean``, ``ptm``, ``mean_pae_angstrom``, and
   writes compact summaries to ``<out-dir>/folded/<model>.jsonl`` (zero bulk PDB dumps).
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

from src.capability.generation import structure_evidence as se  # noqa: E402
from src.capability.ladder import fold_geometry as fg  # noqa: E402
from src.capability.unified import r4_continuation_yardstick as r4c  # noqa: E402

FOLD_SEED_RULE = "sequence_digest"


def _data_root() -> Path:
    env = os.environ.get("TRANSFER_DATA_ROOT", "").strip()
    return Path(env) if (env and Path(env).is_dir()) else ROOT


def _resolve_path(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    alt = _data_root() / path
    return alt.resolve() if alt.exists() else path


def _resolve_esmfold2_dir(explicit: Path | None) -> Path:
    cands: list[Path] = []
    if explicit is not None:
        cands.append(explicit)
    for var in ("ESMFOLD2_MODEL_DIR", "ESMFOLD2_DIR"):
        v = os.environ.get(var, "").strip()
        if v:
            cands.append(Path(v))
    mr = os.environ.get("MODEL_ROOT", "").strip()
    if mr:
        cands.append(Path(mr) / "ESMFold2-hf")
    cands.extend(
        [
            Path("/gpfs/jiaotongdamoxing/zhk_zip/models/ESMFold2-hf"),
            Path("/Data/lzp/models/ESMFold2-hf"),
            Path("/home/lzp/models/ESMFold2-hf"),
        ]
    )
    for c in cands:
        if c.is_dir() and (c / "config.json").is_file():
            return c.resolve()
    raise FileNotFoundError(f"ESMFold2-hf checkpoint not found; checked {[str(x) for x in cands]}")


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def run_generate_phase(args: argparse.Namespace) -> None:
    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else list(r4c.R4_MODEL_IDS)
    )
    assigned = r4c.plan_model_shards(models, args.num_shards)[args.shard_index]
    expected_rows = 80 * int(args.attempts_per_cell)
    print(
        f"[r4_cont:generate shard {args.shard_index}/{args.num_shards}] "
        f"models={assigned} attempts/cell={args.attempts_per_cell}",
        flush=True,
    )

    contexts = json.loads(_resolve_path(args.homologs_json).read_text(encoding="utf-8"))
    plans_cache: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    gen_dir = args.out_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    for mid in assigned:
        spec = r4c.R4_MODEL_MAP[mid]
        slug = r4c.model_file_slug(mid)
        out_jsonl = gen_dir / f"{slug}.jsonl"
        if _line_count(out_jsonl) == expected_rows and not args.overwrite:
            print(f"[r4_cont:generate] {mid}: cached ({expected_rows} rows), skipping", flush=True)
            continue
        eff_budget = int(spec.context_budget) if args.budget == r4c.BUDGET_TOKENS else int(args.budget)
        plan_key = (eff_budget, int(spec.max_context_items))
        if plan_key not in plans_cache:
            plans_cache[plan_key] = r4c.build_canonical_target_plans(
                contexts, budget=eff_budget, max_items=int(spec.max_context_items)
            )
        plans = plans_cache[plan_key]
        t0 = time.monotonic()
        backend, loaded, dt = r4c.load_continuation_model(
            mid,
            device=args.device,
            dtype_override=args.dtype,
            model_root=args.model_root,
            esm3_model_dir=args.esm3_model_dir,
        )
        t_load = time.monotonic() - t0
        print(
            f"[r4_cont:generate] loaded {mid} (backend={backend}, dtype={dt}, budget={eff_budget}) in {t_load:.1f}s",
            flush=True,
        )
        try:
            t1 = time.monotonic()
            rows = r4c.generate_model_attempts(
                model_id=mid,
                backend=backend,
                loaded=loaded,
                plans=plans,
                attempts_per_cell=args.attempts_per_cell,
                budget=eff_budget,
                seed_base=args.base_seed,
            )
            _atomic_write_jsonl(out_jsonl, rows)
            n_valid = sum(1 for r in rows if r["residues"] >= r4c.MIN_PRODUCT_RESIDUES)
            print(
                f"[r4_cont:generate] {mid}: {len(rows)} attempts ({n_valid} valid>=32aa) "
                f"in {time.monotonic() - t1:.1f}s",
                flush=True,
            )
        finally:
            del loaded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _load_esmfold2(esmfold2_dir: Path, device: str):
    from transformers import EsmFold2Model

    se.require_esmfold2_checkpoint(esmfold2_dir)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = (
        EsmFold2Model.from_pretrained(esmfold2_dir, local_files_only=True, dtype=torch.bfloat16)
        .eval()
        .to(dev)
    )
    return model, dev


def _fold_one_seq(
    model: Any, dev: torch.device, seq: str, *, base_seed: int, diffusion_samples: int
) -> tuple[dict[str, float], np.ndarray]:
    from transformers.models.esmfold2.protein_utils import output_to_pdb, prepare_protein_features

    seed = se.fold_seed(seq, base_seed=base_seed, rule=FOLD_SEED_RULE)
    samples_tried = [int(diffusion_samples)]
    if len(seq) > 700 and int(diffusion_samples) > 8:
        samples_tried.extend([8, 4])

    last_oom: Exception | None = None
    out = None
    feats = None
    used_samples = int(diffusion_samples)
    for n_samp in samples_tried:
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        try:
            with torch.inference_mode():
                feats = prepare_protein_features(seq, device=dev)
                out = model.fold(**feats, num_diffusion_samples=n_samp)
            used_samples = n_samp
            last_oom = None
            break
        except torch.OutOfMemoryError as exc:
            last_oom = exc
            out = None
            feats = None
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    if last_oom is not None or out is None or feats is None:
        raise last_oom or RuntimeError(f"Failed to fold sequence of length {len(seq)}")

    summ = se.summarize_esmfold2(out, sequence=seq)
    s_idx = int(summ["diffusion_sample_index"])
    pdb_text = output_to_pdb(out, feats, sample_idx=s_idx)
    ca = fg.parse_ca_trace(pdb_text)
    if ca.shape != (len(seq), 3):
        raise ValueError(f"CA trace shape {ca.shape} != ({len(seq)}, 3)")
    metrics = {
        "plddt_mean": float(summ["mean_ca_plddt"]),
        "ptm": float(summ["ptm"]),
        "mean_pae_angstrom": float(summ["mean_pae_angstrom"]),
        "diffusion_sample_index": s_idx,
        "diffusion_samples_used": used_samples,
    }
    del out, feats
    if len(seq) > 600 and dev.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, ca


def run_fold_wt_phase(args: argparse.Namespace) -> None:
    wt_npz = args.out_dir / "wt_ca_traces.npz"
    wt_json = args.out_dir / "wt_folds.json"
    if wt_npz.is_file() and wt_json.is_file() and not args.overwrite:
        print("[r4_cont:fold_wt] cached wt_ca_traces.npz found, skipping", flush=True)
        return
    contexts = json.loads(_resolve_path(args.homologs_json).read_text(encoding="utf-8"))
    plans = r4c.build_canonical_target_plans(contexts, budget=args.budget)
    esmfold2_dir = _resolve_esmfold2_dir(args.esmfold2_dir)
    model, dev = _load_esmfold2(esmfold2_dir, args.device)
    traces: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}
    try:
        for tid in r4c.ALL_22_TARGETS:
            wt = str(plans[tid]["wildtype"])
            m, ca = _fold_one_seq(
                model, dev, wt, base_seed=args.base_seed, diffusion_samples=args.diffusion_samples
            )
            traces[tid] = ca.astype(np.float32)
            meta[tid] = {"length": len(wt), **m}
            print(f"[r4_cont:fold_wt] {tid} (L={len(wt)}): pLDDT={m['plddt_mean']:.1f} pTM={m['ptm']:.3f}", flush=True)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(wt_npz, **traces)
    wt_json.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_fold_phase(args: argparse.Namespace) -> None:
    wt_npz = args.out_dir / "wt_ca_traces.npz"
    if not wt_npz.is_file():
        raise FileNotFoundError(f"Missing {wt_npz}; run --phase fold_wt first")
    wt_data = np.load(wt_npz, allow_pickle=False)
    wt_ca = {k: np.asarray(wt_data[k], dtype=np.float64) for k in wt_data.files}

    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else list(r4c.R4_MODEL_IDS)
    )
    ordered_models = sorted(models)
    assigned = [m for idx, m in enumerate(ordered_models) if idx % args.num_shards == args.shard_index]

    gen_dir = args.out_dir / "generated"
    fold_dir = args.out_dir / "folded"
    fold_dir.mkdir(parents=True, exist_ok=True)

    todo: list[tuple[str, list[dict[str, Any]], Path]] = []
    for mid in assigned:
        slug = r4c.model_file_slug(mid)
        in_p = gen_dir / f"{slug}.jsonl"
        out_p = fold_dir / f"{slug}.jsonl"
        if not in_p.is_file():
            raise FileNotFoundError(f"Missing generated file {in_p}")
        attempts = r4c._read_jsonl(in_p)
        selected = r4c.select_fold_candidates(attempts, max_per_cell=args.fold_per_cell)
        if _line_count(out_p) == len(selected) and not args.overwrite:
            print(f"[r4_cont:fold] {mid}: cached ({len(selected)} folded), skipping", flush=True)
            continue
        todo.append((mid, selected, out_p))

    print(
        f"[r4_cont:fold shard {args.shard_index}/{args.num_shards}] "
        f"assigned={len(assigned)} todo={len(todo)}",
        flush=True,
    )
    if not todo:
        return

    esmfold2_dir = _resolve_esmfold2_dir(args.esmfold2_dir)
    model, dev = _load_esmfold2(esmfold2_dir, args.device)
    fold_cache: dict[str, tuple[dict[str, float], np.ndarray]] = {}
    t_start = time.monotonic()

    try:
        for mid, selected, out_p in todo:
            t_m = time.monotonic()
            folded_rows: list[dict[str, Any]] = []
            for rec in selected:
                seq = str(rec["sequence"])
                sha = str(rec["sequence_sha256"])
                tid = str(rec["target_id"])
                cached = fold_cache.get(sha)
                if cached is None:
                    cached = _fold_one_seq(
                        model, dev, seq, base_seed=args.base_seed, diffusion_samples=args.diffusion_samples
                    )
                    fold_cache[sha] = cached
                s_metrics, mob_ca = cached
                tm_val = r4c.structural_tm_score(mob_ca, wt_ca[tid])
                folded_rows.append(
                    {
                        "schema_version": r4c.FOLDED_SCHEMA,
                        "attempt_id": rec["attempt_id"],
                        "model": mid,
                        "target_id": tid,
                        "cluster": rec["cluster"],
                        "condition": rec["condition"],
                        "sequence_sha256": sha,
                        "residues": len(seq),
                        "plddt_mean": s_metrics["plddt_mean"],
                        "ptm": s_metrics["ptm"],
                        "mean_pae_angstrom": s_metrics["mean_pae_angstrom"],
                        "tm_score_to_target_wt": float(tm_val),
                    }
                )
            _atomic_write_jsonl(out_p, folded_rows)
            print(
                f"[r4_cont:fold shard {args.shard_index}/{args.num_shards}] {mid}: "
                f"{len(folded_rows)} seqs in {time.monotonic() - t_m:.1f}s "
                f"(unique_cache={len(fold_cache)})",
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
        "models_folded": [m for m, _, _ in todo],
        "elapsed_seconds": time.monotonic() - t_start,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_dir / f"receipt_fold_s{args.shard_index:02d}_of_{args.num_shards:02d}.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=("generate", "fold_wt", "fold"), required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out-dir", type=Path, default=r4c.DEFAULT_WORK_DIR)
    ap.add_argument("--homologs-json", type=Path, default=r4c.DEFAULT_HOMOLOGS_JSON)
    ap.add_argument("--esmfold2-dir", type=Path, default=None)
    ap.add_argument("--model-root", type=Path, default=None)
    ap.add_argument("--esm3-model-dir", type=Path, default=None)
    ap.add_argument("--models", type=str, default=None)
    ap.add_argument("--attempts-per-cell", type=int, default=r4c.ATTEMPTS_PER_CELL)
    ap.add_argument("--fold-per-cell", type=int, default=r4c.FOLD_CANDIDATES_PER_CELL)
    ap.add_argument("--budget", type=int, default=r4c.BUDGET_TOKENS)
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument("--diffusion-samples", type=int, default=32)
    ap.add_argument("--base-seed", type=int, default=r4c.GENERATION_SEED_BASE)
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"--shard-index {args.shard_index} outside [0, {args.num_shards})")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "generate":
        run_generate_phase(args)
    elif args.phase == "fold_wt":
        run_fold_wt_phase(args)
    elif args.phase == "fold":
        run_fold_phase(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
