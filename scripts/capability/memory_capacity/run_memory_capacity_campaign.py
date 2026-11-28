#!/usr/bin/env python3
"""Two-phase multi-GPU campaign orchestrator for the 20-AA PLM parametric memory capacity study.

Execution architecture:
  - Worker mode (``--single-run-id <id>``): runs a single ``RunSpec`` via ``execute_run`` on either
    an isolated single GPU or inside a multi-GPU ``torchrun`` DDP group.
  - Orchestrator mode (default):
      Phase 1: Dynamic Longest-Processing-Time-First (LPT) single-GPU worker queue across all GPUs
               for runs with ``not run.ddp_preferred`` (or all runs when ``len(gpus) <= 1``).
      Phase 2: Sequential full-node ``torchrun --standalone`` DDP execution for ``ddp_preferred``
               runs (35M then 150M, ascending by scale then estimated tokens so fast runs seal early).
      Final:   Aggregates completed receipts into ``<out_dir>/parametric_capacity_summary.json``.

Strictly enforces zero hostname, pod-name, or cluster-IP persistence across all JSON artifacts and logs.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, IO

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.capability.core.io import sha256_file  # noqa: E402
from src.capability.memory_capacity.analysis import summarize_memory_capacity_campaign  # noqa: E402
from src.capability.memory_capacity.spec import (  # noqa: E402
    SCALE_ORDER,
    RunSpec,
    assert_no_forbidden_infra_strings,
    build_campaign_runs,
)
from src.capability.memory_capacity.training import execute_run  # noqa: E402
from src.capability.memory_theory.io_utils import load_json_file, write_json_with_sha256  # noqa: E402
from src.capability.unified.master_summary import FORBIDDEN_INFRA_RE  # noqa: E402

VALID_TRACKS: tuple[str, ...] = (
    "primary",
    "ablation_wd",
    "ablation_fp32",
    "ablation_lr_horizon",
)
MANIFEST_SCHEMA = "plm_parametric_capacity_campaign_manifest_v1"
COMPLETED_STATUSES = frozenset({"completed", "complete"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _redact_infra(text: str) -> str:
    return FORBIDDEN_INFRA_RE.sub("<redacted-infra>", text)


def _write_checked_json(path: Path, payload: dict[str, Any]) -> str:
    assert_no_forbidden_infra_strings(payload, label=path.name)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    match = FORBIDDEN_INFRA_RE.search(serialized)
    if match:
        raise ValueError(
            f"Refusing to write {path.name} containing forbidden infra identifier: {match.group(0)!r}"
        )
    return write_json_with_sha256(path, payload)


def _is_run_complete(run_json: Path) -> bool:
    if not run_json.is_file():
        return False
    sidecar = run_json.with_name(run_json.name + ".sha256")
    if sidecar.is_file():
        parts = sidecar.read_text(encoding="utf-8").strip().split()
        if not parts or parts[0] != sha256_file(run_json):
            return False
    try:
        doc = load_json_file(run_json)
    except Exception:
        return False
    return isinstance(doc, dict) and doc.get("status") in COMPLETED_STATUSES


def _read_final_c_eff(run_json: Path) -> str:
    if not run_json.is_file():
        return "n/a"
    try:
        doc = load_json_file(run_json)
        if isinstance(doc.get("final_eval"), dict):
            fe = doc["final_eval"]
            val = float(fe.get("memorized_bits_per_param", fe.get("bits_per_non_emb_param")))
            return f"{val:.4f}"
        exposures = doc.get("evaluations") or doc.get("exposures") or []
        if not exposures:
            return "n/a"
        last = exposures[-1]
        val = float(last.get("memorized_bits_per_param", last.get("bits_per_non_emb_param")))
        return f"{val:.4f}"
    except Exception:
        return "n/a"


def _parse_scales(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return tuple(SCALE_ORDER)
    items = tuple(tok.strip() for tok in raw.split(",") if tok.strip())
    if not items:
        raise ValueError("--scales produced an empty selection")
    unknown = [s for s in items if s not in SCALE_ORDER]
    if unknown:
        raise ValueError(f"Unknown scale(s) {unknown}; valid scales: {SCALE_ORDER}")
    return items


def _parse_tracks(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return VALID_TRACKS
    items = tuple(tok.strip() for tok in raw.split(",") if tok.strip())
    if not items:
        raise ValueError("--tracks produced an empty selection")
    unknown = [t for t in items if t not in VALID_TRACKS]
    if unknown:
        raise ValueError(f"Unknown track(s) {unknown}; valid tracks: {VALID_TRACKS}")
    return items


def _resolve_workers(gpus_arg: str | None) -> tuple[list[str], bool]:
    """Return (worker_slot_ids, use_cpu)."""
    if gpus_arg is not None:
        cleaned = gpus_arg.strip()
        if cleaned.lower() == "cpu":
            return ["cpu"], True
        ids = [tok.strip() for tok in cleaned.split(",") if tok.strip()]
        if not ids:
            raise ValueError("--gpus must specify at least one GPU index (e.g. '0,1,2,3,4,5')")
        for gid in ids:
            if not gid.isdigit():
                raise ValueError(f"Invalid GPU index {gid!r} in --gpus={gpus_arg!r}")
        return ids, False

    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count > 0:
        return [str(i) for i in range(count)], False
    return ["cpu"], True


def _resolve_torchrun_binary() -> str:
    sibling = Path(sys.executable).with_name("torchrun")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    found = shutil.which("torchrun")
    if found:
        return found
    raise FileNotFoundError(
        f"Could not locate 'torchrun' alongside {sys.executable} or on PATH"
    )


def _tail_log(log_path: Path, max_lines: int = 40) -> str:
    if not log_path.is_file():
        return "<no log file>"
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return _redact_infra("\n".join(lines[-max_lines:]))


def _run_single_worker(args: argparse.Namespace, skip_existing: bool) -> int:
    all_runs = build_campaign_runs(
        include_ablations=True,
        data_seed=args.data_seed,
        model_seed=args.model_seed,
    )
    by_id = {r.run_id: r for r in all_runs}
    if args.single_run_id not in by_id:
        raise ValueError(
            f"Unknown --single-run-id={args.single_run_id!r} "
            f"(available campaign runs: {len(by_id)})"
        )
    spec = by_id[args.single_run_id]

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if "LOCAL_RANK" in os.environ and world_size > 1 and torch.cuda.is_available():
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    else:
        device = args.device

    out_dir = Path(args.out_dir)
    data_cache_dir = (
        Path(args.data_cache_dir) if args.data_cache_dir is not None else out_dir / "data_cache"
    )
    checkpoint_dir = out_dir / "checkpoints"

    try:
        execute_run(
            spec,
            out_dir=out_dir,
            data_cache_dir=data_cache_dir,
            checkpoint_dir=checkpoint_dir,
            device=device,
            use_compile=not args.no_compile,
            eval_max_sequences_mlm=args.eval_max_mlm,
            resume=skip_existing,
            skip_completed=skip_existing,
        )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


def _build_worker_common_flags(args: argparse.Namespace, skip_existing: bool) -> list[str]:
    flags = [
        "--out-dir",
        str(args.out_dir),
        "--eval-max-mlm",
        str(args.eval_max_mlm),
        "--data-seed",
        str(args.data_seed),
        "--model-seed",
        str(args.model_seed),
    ]
    if args.data_cache_dir is not None:
        flags.extend(["--data-cache-dir", str(args.data_cache_dir)])
    if args.no_compile:
        flags.append("--no-compile")
    if not skip_existing:
        flags.append("--force")
    return flags


def _run_orchestrator(args: argparse.Namespace, skip_existing: bool) -> int:
    selected_scales = _parse_scales(args.scales)
    selected_tracks = _parse_tracks(args.tracks)
    include_ablations = any(t != "primary" for t in selected_tracks)

    campaign_runs = build_campaign_runs(
        include_ablations=include_ablations,
        data_seed=args.data_seed,
        model_seed=args.model_seed,
    )
    matched_runs = [
        r
        for r in campaign_runs
        if r.scale_id in selected_scales and r.track in selected_tracks
    ]

    worker_slots, use_cpu = _resolve_workers(args.gpus)
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"

    pending_runs: list[RunSpec] = []
    skipped_ids: list[str] = []
    for r in matched_runs:
        if skip_existing and _is_run_complete(runs_dir / f"{r.run_id}.json"):
            skipped_ids.append(r.run_id)
        else:
            pending_runs.append(r)

    multi_gpu_available = (not use_cpu) and (len(worker_slots) > 1)
    if multi_gpu_available:
        phase1_runs = [r for r in pending_runs if not r.ddp_preferred]
        phase2_runs = [r for r in pending_runs if r.ddp_preferred]
    else:
        phase1_runs = list(pending_runs)
        phase2_runs = []

    # Phase 1: Descending by estimated_tokens (Longest-Processing-Time-First)
    phase1_runs.sort(
        key=lambda r: (-int(r.estimated_tokens), -(int(r.n_sequences) * int(r.max_epochs)), r.run_id)
    )
    # Phase 2: Ascending by scale order (35m before 150m), then ascending by estimated_tokens
    phase2_runs.sort(
        key=lambda r: (
            SCALE_ORDER.index(r.scale_id) if r.scale_id in SCALE_ORDER else len(SCALE_ORDER),
            int(r.estimated_tokens),
            r.run_id,
        )
    )

    p1_tokens = sum(int(r.estimated_tokens) for r in phase1_runs)
    p2_tokens = sum(int(r.estimated_tokens) for r in phase2_runs)
    total_matched_tokens = sum(int(r.estimated_tokens) for r in matched_runs)

    if args.dry_run:
        worker_desc = "cpu (1 worker)" if use_cpu else f"{len(worker_slots)} GPU(s) [{','.join(worker_slots)}]"
        print("=== PLM Parametric Memory Capacity Campaign (DRY RUN) ===")
        print(f"Workers:              {worker_desc}")
        print(f"Scales:               {', '.join(selected_scales)}")
        print(f"Tracks:               {', '.join(selected_tracks)}")
        print(f"Matched jobs:         {len(matched_runs)} (total_tokens={total_matched_tokens:,})")
        print(f"Already complete:     {len(skipped_ids)}")
        print(f"Pending jobs:         {len(pending_runs)} (pending_tokens={p1_tokens + p2_tokens:,})")
        print(f"Phase 1 (1-GPU LPT):  {len(phase1_runs)} jobs (tokens={p1_tokens:,})")
        for idx, r in enumerate(phase1_runs, start=1):
            print(
                f"  [P1 {idx:03d}/{len(phase1_runs):03d}] {r.run_id} "
                f"scale={r.scale_id} paradigm={r.paradigm} mult={r.multiplier:g}x "
                f"epochs={r.max_epochs} est_tokens={int(r.estimated_tokens):,}"
            )
        print(f"Phase 2 (DDP {len(worker_slots)}-GPU): {len(phase2_runs)} jobs (tokens={p2_tokens:,})")
        for idx, r in enumerate(phase2_runs, start=1):
            print(
                f"  [P2 {idx:03d}/{len(phase2_runs):03d}] {r.run_id} "
                f"scale={r.scale_id} paradigm={r.paradigm} mult={r.multiplier:g}x "
                f"epochs={r.max_epochs} est_tokens={int(r.estimated_tokens):,}"
            )
        return 0

    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest_payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "created_utc": _utc_now(),
        "scales": list(selected_scales),
        "tracks": list(selected_tracks),
        "worker_mode": "cpu" if use_cpu else "cuda",
        "n_workers": len(worker_slots),
        "n_matched_runs": len(matched_runs),
        "n_skipped_existing": len(skipped_ids),
        "n_pending_runs": len(pending_runs),
        "phase1_n_runs": len(phase1_runs),
        "phase2_n_runs": len(phase2_runs),
        "phase1_estimated_tokens": p1_tokens,
        "phase2_estimated_tokens": p2_tokens,
        "total_matched_estimated_tokens": total_matched_tokens,
        "phase1_run_ids": [r.run_id for r in phase1_runs],
        "phase2_run_ids": [r.run_id for r in phase2_runs],
        "runs": [r.to_dict() for r in matched_runs],
    }
    _write_checked_json(out_dir / "campaign_manifest.json", manifest_payload)

    total_pending = len(pending_runs)
    completed_so_far = 0
    script_path = str(Path(__file__).resolve())
    common_flags = _build_worker_common_flags(args, skip_existing=skip_existing)

    active_procs: dict[str, tuple[RunSpec, subprocess.Popen[bytes], float, Path, IO[bytes]]] = {}

    def _terminate_active() -> None:
        for _, proc, _, _, log_fh in list(active_procs.values()):
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + 5.0
        for _, proc, _, _, log_fh in list(active_procs.values()):
            rem = max(0.1, deadline - time.monotonic())
            try:
                proc.wait(timeout=rem)
            except subprocess.TimeoutExpired:
                proc.kill()
            finally:
                log_fh.close()
        active_procs.clear()

    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _signal_handler(signum: int, _frame: Any) -> None:
        _terminate_active()
        raise KeyboardInterrupt(f"Campaign interrupted by signal {signum}")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        # Phase 1: Single-GPU Dynamic LPT Queue
        if phase1_runs:
            print(
                f"{_utc_now()} [PHASE 1 START] {len(phase1_runs)} single-worker jobs across "
                f"{len(worker_slots)} slot(s) (LPT order)",
                flush=True,
            )
            queue: deque[RunSpec] = deque(phase1_runs)
            while queue or active_procs:
                # Fill all idle slots immediately
                for slot in worker_slots:
                    if slot in active_procs or not queue:
                        continue
                    spec = queue.popleft()
                    log_path = logs_dir / f"{spec.run_id}.log"
                    log_fh = log_path.open("wb")
                    env = os.environ.copy()
                    env["OMP_NUM_THREADS"] = "4"
                    if use_cpu:
                        worker_dev = "cpu"
                    else:
                        env["CUDA_VISIBLE_DEVICES"] = slot
                        worker_dev = "cuda:0"
                    cmd = [
                        sys.executable,
                        script_path,
                        "--single-run-id",
                        spec.run_id,
                        "--device",
                        worker_dev,
                        *common_flags,
                    ]
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(ROOT),
                        env=env,
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                    )
                    active_procs[slot] = (spec, proc, time.monotonic(), log_path, log_fh)

                # Check finished slots
                any_finished = False
                for slot in list(active_procs.keys()):
                    spec, proc, t0, log_path, log_fh = active_procs[slot]
                    rc = proc.poll()
                    if rc is None:
                        continue
                    any_finished = True
                    log_fh.close()
                    del active_procs[slot]
                    elapsed = time.monotonic() - t0
                    if rc != 0:
                        _terminate_active()
                        tail = _tail_log(log_path)
                        raise RuntimeError(
                            f"Phase 1 worker failed for {spec.run_id} (rc={rc}):\n{tail}"
                        )
                    completed_so_far += 1
                    c_eff_str = _read_final_c_eff(runs_dir / f"{spec.run_id}.json")
                    print(
                        f"{_utc_now()} [DONE {completed_so_far}/{total_pending}] "
                        f"{spec.run_id} phase=1 elapsed={elapsed:.1f}s c_eff_final={c_eff_str}",
                        flush=True,
                    )

                if not any_finished and active_procs:
                    time.sleep(1.0)

        # Phase 2: Multi-GPU torchrun DDP Executor
        if phase2_runs:
            torchrun_bin = _resolve_torchrun_binary()
            nproc = len(worker_slots)
            vis_gpus = ",".join(worker_slots)
            print(
                f"{_utc_now()} [PHASE 2 START] {len(phase2_runs)} multi-GPU DDP jobs "
                f"(nproc_per_node={nproc})",
                flush=True,
            )
            for spec in phase2_runs:
                log_path = logs_dir / f"{spec.run_id}.log"
                log_fh = log_path.open("wb")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = vis_gpus
                env["OMP_NUM_THREADS"] = "4"
                cmd = [
                    torchrun_bin,
                    "--standalone",
                    "--nnodes=1",
                    f"--nproc_per_node={nproc}",
                    script_path,
                    "--single-run-id",
                    spec.run_id,
                    *common_flags,
                ]
                t0 = time.monotonic()
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    env=env,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                )
                active_procs["ddp"] = (spec, proc, t0, log_path, log_fh)
                rc = proc.wait()
                log_fh.close()
                active_procs.pop("ddp", None)
                elapsed = time.monotonic() - t0
                if rc != 0:
                    tail = _tail_log(log_path)
                    raise RuntimeError(
                        f"Phase 2 DDP job failed for {spec.run_id} (rc={rc}):\n{tail}"
                    )
                completed_so_far += 1
                c_eff_str = _read_final_c_eff(runs_dir / f"{spec.run_id}.json")
                print(
                    f"{_utc_now()} [DONE {completed_so_far}/{total_pending}] "
                    f"{spec.run_id} phase=2 elapsed={elapsed:.1f}s c_eff_final={c_eff_str}",
                    flush=True,
                )
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        if active_procs:
            _terminate_active()

    # Final aggregation if completed receipts exist
    completed_files = [p for p in runs_dir.glob("*.json") if _is_run_complete(p)]
    if completed_files:
        summary_path = out_dir / "parametric_capacity_summary.json"
        summarize_memory_capacity_campaign(runs_dir, out_path=summary_path)
        print(
            f"{_utc_now()} [SUMMARY] Aggregated {len(completed_files)} completed run(s) -> "
            f"{summary_path}",
            flush=True,
        )

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    default_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Campaign output directory (writes runs/, checkpoints/, campaign_manifest.json, summary)",
    )
    parser.add_argument(
        "--data-cache-dir",
        type=Path,
        default=None,
        help="Optional directory for cached synthetic sequence panels (default: <out-dir>/data_cache)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU indices (e.g. '0,1,2,3,4,5'), 'cpu', or auto-detect",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="all",
        help=f"Comma-separated subset of {','.join(SCALE_ORDER)} or 'all' (default: all)",
    )
    parser.add_argument(
        "--tracks",
        type=str,
        default="all",
        help=f"Comma-separated subset of {','.join(VALID_TRACKS)} or 'all' (default: all)",
    )
    parser.add_argument(
        "--single-run-id",
        "--run-id",
        dest="single_run_id",
        type=str,
        default=None,
        help="Worker mode: execute a single RunSpec by run_id and exit",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_dev,
        help="PyTorch device string for single-GPU worker mode (default: cuda:0 if available else cpu)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (useful for quick smoke tests or debugging)",
    )
    parser.add_argument(
        "--eval-max-mlm",
        type=int,
        default=65536,
        help="Maximum evaluation sequences subsampled for MLM 16-stratum quadrature (default: 65536)",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip runs with verified complete receipts on disk (default: True)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run and overwrite existing receipts (overrides --skip-existing)",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=20261127,
        help="Master dataset generation seed (default: 20261127)",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=20261128,
        help="Master model initialization seed (default: 20261128)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print Phase 1 / Phase 2 job counts, ordering, and token totals, then exit 0",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    skip_existing = False if args.force else bool(args.skip_existing)

    if args.single_run_id is not None:
        return _run_single_worker(args, skip_existing=skip_existing)
    return _run_orchestrator(args, skip_existing=skip_existing)


if __name__ == "__main__":
    raise SystemExit(main())
