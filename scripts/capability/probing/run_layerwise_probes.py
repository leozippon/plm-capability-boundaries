#!/usr/bin/env python3
"""Batch layerwise hidden-state probe runner & aggregator for Stage-2 checkpoints.

Scans a directory of Stage-2 ``.pt`` checkpoints (``_final.pt``, ``_latest.pt``, or
milestone snapshots ``_e0000.pt .. _e0300.pt``), runs:
  - Eq. (3b) temperature calibration on ``eval_unseen_fold`` (``H_cal``, ``T_opt``, ``Delta_cal``)
  - Probe 1: MegaScale ddG numerical Ridge probe (train ``seen_pool_natural``, eval
    ``unseen_fold_natural`` & ``de_novo_design`` vs raw & temperature-calibrated logits)
  - Probe 2: Unseen-fold 3D contact pairwise 5-fold Ridge (``|i-j| >= 6`` & ``>= 24``)
    vs zero-shot Categorical Jacobian + APC
  - Probe 3: Unseen-fold per-residue 3D burial / coordination number 5-fold Ridge

Supports multi-GPU sharding via ``--gpus 0,1,2`` or explicit ``--worker-rank`` /
``--num-workers``, atomic JSON + SHA-256 receipts, resume-safe skipping, and
cross-epoch H1 vs H2 ``layer_degradation_profile`` aggregation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.probing.analysis import (
    PROBE_CHECKPOINT_SCHEMA,
    aggregate_probe_results,
    evaluate_checkpoint_probes,
)
from src.capability.probing.extractor import load_stage2_checkpoint
from src.capability.probing.probes import DEFAULT_RIDGE_LAMBDAS
from src.capability.supply_demand.corpus import (
    Stage2Bundle,
    atomic_write_json_with_sha256,
    sha256_file,
)
from src.capability.supply_demand.objectives import PackedSequenceStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run frozen layerwise hidden-state linear probes across Stage-2 checkpoints."
    )
    p.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help="Directory containing Stage-2 .pt checkpoints to scan.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Explicit checkpoint .pt path(s) to evaluate (repeatable).",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="*.pt",
        help="Glob pattern inside --checkpoints-dir (default: '*.pt').",
    )
    p.add_argument("--bundle-dir", type=Path, required=True, help="Path to Stage-2 bundle directory.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for probe JSONs.")
    p.add_argument(
        "--gpus",
        type=str,
        default="",
        help="Comma-separated GPU IDs for automatic multi-GPU sharding (e.g. '2,3').",
    )
    p.add_argument("--device", type=str, default="", help="Single-worker PyTorch device (e.g. 'cuda:0' or 'cpu').")
    p.add_argument("--worker-rank", type=int, default=0, help="0-based shard index for manual multi-worker runs.")
    p.add_argument("--num-workers", type=int, default=1, help="Total number of workers for manual sharding.")
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32_strict"])
    p.add_argument("--max-variants-per-wt", type=int, default=400, help="Max MegaScale variants per WT (0 = all).")
    p.add_argument("--megascale-batch-size", type=int, default=512)
    p.add_argument("--contact-batch-size", type=int, default=64)
    p.add_argument("--zeroshot-contact-batch-size", type=int, default=256)
    p.add_argument("--cal-batch-size", type=int, default=128)
    p.add_argument("--cal-n-draws", type=int, default=16)
    p.add_argument("--n-cv-folds", type=int, default=5)
    p.add_argument(
        "--lambdas",
        type=str,
        default=",".join(f"{x:g}" for x in DEFAULT_RIDGE_LAMBDAS),
        help="Comma-separated Ridge regularization grid.",
    )
    p.add_argument(
        "--skip-zeroshot-contacts",
        action="store_true",
        help="Skip zero-shot Categorical Jacobian contact map evaluation for faster sweeps.",
    )
    p.add_argument("--no-resume", action="store_true", help="Re-evaluate checkpoints even if JSON receipt exists.")
    p.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip writing probe_summary.json (used internally by spawned shard workers).",
    )
    return p.parse_args(argv)


def _discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for p in args.checkpoint:
        rp = p.resolve()
        if not rp.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        if rp not in seen:
            seen.add(rp)
            found.append(rp)
    if args.checkpoints_dir is not None:
        if not args.checkpoints_dir.is_dir():
            raise FileNotFoundError(f"Checkpoints directory not found: {args.checkpoints_dir}")
        for p in sorted(args.checkpoints_dir.glob(args.glob)):
            if not p.is_file() or p.name.startswith("."):
                continue
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                found.append(rp)
    return found


def _is_valid_probe_receipt(json_path: Path) -> bool:
    sha_path = json_path.with_name(json_path.name + ".sha256")
    if not (json_path.is_file() and sha_path.is_file()):
        return False
    parts = sha_path.read_text(encoding="utf-8").strip().split()
    if not parts or parts[0] != sha256_file(json_path):
        return False
    try:
        doc = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(doc, dict) and doc.get("schema") == PROBE_CHECKPOINT_SCHEMA


def _aggregate_directory(per_ckpt_dir: Path, summary_path: Path) -> dict[str, Any]:
    docs: list[dict[str, Any]] = []
    if per_ckpt_dir.is_dir():
        for jp in sorted(per_ckpt_dir.glob("*.probe.json")):
            try:
                doc = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(doc, dict) and doc.get("schema") == PROBE_CHECKPOINT_SCHEMA:
                docs.append(doc)
    summary = aggregate_probe_results(docs)
    atomic_write_json_with_sha256(summary_path, summary)
    return summary


def _launch_multi_gpu_workers(args: argparse.Namespace, gpu_ids: list[str]) -> int:
    """Spawn one shard subprocess per GPU ID and merge the resulting receipts."""
    n_workers = len(gpu_ids)
    procs: list[tuple[str, subprocess.Popen[Any]]] = []
    script_path = Path(__file__).resolve()

    base_cmd = [
        sys.executable,
        str(script_path),
        "--bundle-dir",
        str(args.bundle_dir),
        "--out-dir",
        str(args.out_dir),
        "--glob",
        str(args.glob),
        "--precision",
        str(args.precision),
        "--max-variants-per-wt",
        str(args.max_variants_per_wt),
        "--megascale-batch-size",
        str(args.megascale_batch_size),
        "--contact-batch-size",
        str(args.contact_batch_size),
        "--zeroshot-contact-batch-size",
        str(args.zeroshot_contact_batch_size),
        "--cal-batch-size",
        str(args.cal_batch_size),
        "--cal-n-draws",
        str(args.cal_n_draws),
        "--n-cv-folds",
        str(args.n_cv_folds),
        "--lambdas",
        str(args.lambdas),
        "--num-workers",
        str(n_workers),
        "--skip-aggregate",
    ]
    if args.checkpoints_dir is not None:
        base_cmd.extend(["--checkpoints-dir", str(args.checkpoints_dir)])
    for ck in args.checkpoint:
        base_cmd.extend(["--checkpoint", str(ck)])
    if args.skip_zeroshot_contacts:
        base_cmd.append("--skip-zeroshot-contacts")
    if args.no_resume:
        base_cmd.append("--no-resume")

    for rank, gpu_id in enumerate(gpu_ids):
        cmd = [*base_cmd, "--worker-rank", str(rank), "--device", f"cuda:{gpu_id}"]
        procs.append((gpu_id, subprocess.Popen(cmd)))

    exit_code = 0
    for gpu_id, proc in procs:
        rc = proc.wait()
        if rc != 0:
            print(f"[run_layerwise_probes] worker on cuda:{gpu_id} exited with code {rc}", file=sys.stderr)
            exit_code = rc
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.checkpoints_dir is None and not args.checkpoint:
        raise SystemExit("Must specify --checkpoints-dir or at least one --checkpoint")

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    out_dir = Path(args.out_dir)
    per_ckpt_dir = out_dir / "checkpoints"
    summary_path = out_dir / "probe_summary.json"
    per_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Multi-GPU orchestrator branch
    if len(gpu_ids) > 1:
        rc = _launch_multi_gpu_workers(args, gpu_ids)
        if rc != 0:
            return rc
        if not args.skip_aggregate:
            summary = _aggregate_directory(per_ckpt_dir, summary_path)
            print(
                f"[run_layerwise_probes] aggregated {summary['n_checkpoints']} checkpoints -> {summary_path}",
                flush=True,
            )
        return 0

    if args.device:
        device = torch.device(args.device)
    elif len(gpu_ids) == 1:
        device = torch.device(f"cuda:{gpu_ids[0]}")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    all_ckpts = _discover_checkpoints(args)
    if not all_ckpts:
        print("[run_layerwise_probes] no checkpoints matched.", file=sys.stderr)
        if not args.skip_aggregate:
            _aggregate_directory(per_ckpt_dir, summary_path)
        return 0

    num_workers = max(1, int(args.num_workers))
    worker_rank = int(args.worker_rank)
    if not (0 <= worker_rank < num_workers):
        raise ValueError(f"worker-rank {worker_rank} out of range [0, {num_workers})")

    shard_ckpts = [ck for i, ck in enumerate(all_ckpts) if i % num_workers == worker_rank]
    lambdas = tuple(float(tok.strip()) for tok in args.lambdas.split(",") if tok.strip())
    max_vars = None if int(args.max_variants_per_wt) <= 0 else int(args.max_variants_per_wt)

    bundle = Stage2Bundle(args.bundle_dir)
    megascale_data = bundle.load_megascale()
    contacts_data = bundle.load_contacts()
    unseen_cache: dict[str, PackedSequenceStore] = {}

    def _get_unseen_store(arm_id: str) -> PackedSequenceStore:
        if arm_id in unseen_cache:
            return unseen_cache[arm_id]
        arm_path = bundle.root / "splits" / f"eval_unseen_fold_{arm_id}.npz"
        split_name = f"eval_unseen_fold_{arm_id}" if arm_path.is_file() else "eval_unseen_fold"
        if split_name in unseen_cache:
            unseen_cache[arm_id] = unseen_cache[split_name]
            return unseen_cache[arm_id]
        store, _ = bundle.load_split(split_name)
        unseen_cache[split_name] = store
        unseen_cache[arm_id] = store
        return store

    t_all = time.perf_counter()
    for idx, ckpt_path in enumerate(shard_ckpts, start=1):
        out_json = per_ckpt_dir / f"{ckpt_path.stem}.probe.json"
        if (not args.no_resume) and _is_valid_probe_receipt(out_json):
            print(
                f"[rank {worker_rank}/{num_workers}] ({idx}/{len(shard_ckpts)}) skip verified {ckpt_path.name}",
                flush=True,
            )
            continue

        loaded = load_stage2_checkpoint(ckpt_path, device=device)
        unseen_store = _get_unseen_store(loaded.arm_id)
        doc = evaluate_checkpoint_probes(
            loaded,
            bundle,
            device=device,
            precision=args.precision,
            max_variants_per_wt=max_vars,
            megascale_batch_size=int(args.megascale_batch_size),
            contact_batch_size=int(args.contact_batch_size),
            zeroshot_contact_batch_size=int(args.zeroshot_contact_batch_size),
            cal_batch_size=int(args.cal_batch_size),
            cal_n_draws=int(args.cal_n_draws),
            lambdas=lambdas,
            n_cv_folds=int(args.n_cv_folds),
            compute_zeroshot_contacts=not bool(args.skip_zeroshot_contacts),
            preloaded_megascale=megascale_data,
            preloaded_contacts=contacts_data,
            preloaded_unseen_fold_store=unseen_store,
        )
        atomic_write_json_with_sha256(out_json, doc)

        hl = doc.get("headlines") or {}
        ms_unf = hl.get("megascale_unseen_fold_natural_spearman") or {}
        print(
            f"[rank {worker_rank}/{num_workers}] ({idx}/{len(shard_ckpts)}) {ckpt_path.name} "
            f"[{loaded.paradigm_display} {loaded.scale_id} E={loaded.epoch}] "
            f"Delta_cal={doc.get('Delta_cal', float('nan')):.3f} "
            f"MS_unf best={ms_unf.get('best_layer_score')} (L={ms_unf.get('best_layer_name')}) "
            f"vs ZS={ms_unf.get('zeroshot_score')} ({doc['total_wall_sec']:.2f}s)",
            flush=True,
        )

    if not args.skip_aggregate:
        summary = _aggregate_directory(per_ckpt_dir, summary_path)
        print(
            f"[run_layerwise_probes] finished {len(shard_ckpts)} shard checkpoints in "
            f"{time.perf_counter() - t_all:.2f}s; aggregated {summary['n_checkpoints']} -> {summary_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
