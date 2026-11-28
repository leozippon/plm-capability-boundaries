#!/usr/bin/env python3
"""Spot-safe shared-filesystem lease worker for Stage-1 parametric memory capacity runs.

Reads a JSON manifest of ``RunSpec`` definitions and atomically claims unfinished tasks via
``SharedFSLeaseManager`` lease/heartbeat files on shared storage (expiring stale leases if a
Spot node dies without SIGTERM). Supports both parallel 1-GPU workers and single-node
``torchrun`` DDP workers, and forwards SIGTERM cleanly to flush checkpoints and release leases.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from src.capability.memory_capacity.leases import (  # noqa: E402
    DEFAULT_HEARTBEAT_INTERVAL_SEC,
    DEFAULT_LEASE_TTL_SEC,
    LeaseHeartbeat,
    LeaseRecord,
    SharedFSLeaseManager,
    generate_worker_token,
    load_run_manifest,
)
from src.capability.memory_capacity.spec import RunSpec  # noqa: E402
from src.capability.memory_capacity.training import (  # noqa: E402
    DEFAULT_CHECKPOINT_INTERVAL_SEC,
    PREEMPTION_EXIT_CODE,
    execute_run,
)
from src.capability.unified.master_summary import FORBIDDEN_INFRA_RE  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _redact_infra(text: str) -> str:
    return FORBIDDEN_INFRA_RE.sub("<redacted-infra>", text)


def _resolve_workers(gpus_arg: str | None) -> tuple[list[str], bool]:
    if gpus_arg is not None:
        cleaned = gpus_arg.strip()
        if cleaned.lower() == "cpu":
            return ["cpu"], True
        ids = [tok.strip() for tok in cleaned.split(",") if tok.strip()]
        if not ids:
            raise ValueError("--gpus must specify at least one GPU index or 'cpu'")
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


def _run_manifest_single(args: argparse.Namespace) -> int:
    specs = load_run_manifest(args.manifest)
    by_id = {s.run_id: s for s in specs}
    if args.single_run_id not in by_id:
        raise ValueError(
            f"Run id {args.single_run_id!r} not found in manifest {args.manifest}"
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
    try:
        execute_run(
            spec,
            out_dir=out_dir,
            checkpoint_dir=out_dir / "checkpoints",
            data_cache_dir=data_cache_dir,
            device=device,
            use_compile=not args.no_compile,
            resume=not args.force,
            skip_completed=not args.force,
            eval_max_sequences_mlm=args.eval_max_mlm,
            checkpoint_interval_sec=args.checkpoint_interval_sec,
            cross_eval_epochs=args.cross_eval_epochs,
        )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


def _build_child_flags(args: argparse.Namespace) -> list[str]:
    flags = [
        "--manifest",
        str(args.manifest),
        "--out-dir",
        str(args.out_dir),
        "--eval-max-mlm",
        str(args.eval_max_mlm),
        "--checkpoint-interval-sec",
        str(args.checkpoint_interval_sec),
    ]
    if args.data_cache_dir is not None:
        flags.extend(["--data-cache-dir", str(args.data_cache_dir)])
    if args.no_compile:
        flags.append("--no-compile")
    if args.force:
        flags.append("--force")
    if args.cross_eval_epochs:
        flags.extend(["--cross-eval-epochs", ",".join(str(e) for e in args.cross_eval_epochs)])
    return flags


def _run_spot_pool(args: argparse.Namespace) -> int:
    specs = load_run_manifest(args.manifest)
    worker_slots, use_cpu = _resolve_workers(args.gpus)
    out_dir = Path(args.out_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)

    manager = SharedFSLeaseManager(out_dir, lease_ttl_sec=args.lease_ttl_sec)
    script_path = str(Path(__file__).resolve())
    child_flags = _build_child_flags(args)

    stop_requested = False
    active: dict[
        str,
        tuple[RunSpec, LeaseRecord, LeaseHeartbeat, subprocess.Popen[bytes], float, IO[bytes]],
    ] = {}

    def _graceful_shutdown() -> None:
        for _, _, _, proc, _, _ in list(active.values()):
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + float(args.shutdown_grace_sec)
        for slot, (spec, lease, hb, proc, _, log_fh) in list(active.items()):
            rem = max(0.5, deadline - time.monotonic())
            try:
                proc.wait(timeout=rem)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
            finally:
                hb.stop()
                manager.release(lease)
                log_fh.close()
                active.pop(slot, None)

    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigint = signal.getsignal(signal.SIGINT)

    def _on_signal(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    multi_gpu = (not use_cpu) and (len(worker_slots) > 1)
    run_single_phase = args.mode in ("single_gpu", "auto")
    run_ddp_phase = args.mode in ("ddp", "auto") and multi_gpu
    single_predicate = (
        (lambda s: True)
        if (args.allow_ddp_on_single_gpu or not multi_gpu)
        else (lambda s: not s.ddp_preferred)
    )
    ddp_predicate = (
        (lambda s: True) if args.mode == "ddp" else (lambda s: s.ddp_preferred)
    )

    try:
        # Phase A: 1-GPU parallel lease workers
        if run_single_phase:
            while not stop_requested:
                # Claim tasks for idle slots
                for slot in worker_slots:
                    if slot in active or stop_requested:
                        continue
                    slot_token = generate_worker_token(f"slot{slot}")
                    claimed = manager.claim_next(
                        specs,
                        worker_id=slot_token,
                        mode="single_gpu",
                        predicate=single_predicate,
                    )
                    if claimed is None:
                        continue
                    spec, lease = claimed
                    hb = LeaseHeartbeat(
                        manager, lease, interval_sec=args.heartbeat_interval_sec
                    )
                    hb.start()

                    log_path = logs_dir / f"{spec.run_id}.log"
                    log_fh = log_path.open("ab")
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
                        *child_flags,
                    ]
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(ROOT),
                        env=env,
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                    )
                    active[slot] = (spec, lease, hb, proc, time.monotonic(), log_fh)
                    print(
                        f"{_utc_now()} [CLAIM single_gpu] slot={slot} run={spec.run_id}",
                        flush=True,
                    )

                if not active:
                    break

                any_finished = False
                for slot in list(active.keys()):
                    spec, lease, hb, proc, t0, log_fh = active[slot]
                    rc = proc.poll()
                    if rc is None:
                        continue
                    any_finished = True
                    hb.stop()
                    manager.release(lease)
                    log_fh.close()
                    del active[slot]
                    elapsed = time.monotonic() - t0
                    if rc == 0:
                        print(
                            f"{_utc_now()} [COMPLETE single_gpu] run={spec.run_id} elapsed={elapsed:.1f}s",
                            flush=True,
                        )
                    elif rc == PREEMPTION_EXIT_CODE:
                        print(
                            f"{_utc_now()} [PREEMPTED single_gpu] run={spec.run_id} elapsed={elapsed:.1f}s",
                            flush=True,
                        )
                    else:
                        print(
                            f"{_utc_now()} [FAILED single_gpu] run={spec.run_id} rc={rc} elapsed={elapsed:.1f}s",
                            file=sys.stderr,
                            flush=True,
                        )
                        if args.fail_fast:
                            _graceful_shutdown()
                            return rc

                if not any_finished and active and not stop_requested:
                    time.sleep(min(2.0, args.poll_interval_sec))

            if stop_requested:
                _graceful_shutdown()
                return PREEMPTION_EXIT_CODE

        # Phase B: Full-node torchrun DDP lease worker
        if run_ddp_phase and not stop_requested:
            torchrun_bin = _resolve_torchrun_binary()
            nproc = len(worker_slots)
            vis_gpus = ",".join(worker_slots)
            while not stop_requested:
                ddp_token = generate_worker_token("ddp")
                claimed = manager.claim_next(
                    specs,
                    worker_id=ddp_token,
                    mode="ddp",
                    predicate=ddp_predicate,
                )
                if claimed is None:
                    break
                spec, lease = claimed
                hb = LeaseHeartbeat(
                    manager, lease, interval_sec=args.heartbeat_interval_sec
                )
                hb.start()

                log_path = logs_dir / f"{spec.run_id}.log"
                log_fh = log_path.open("ab")
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
                    *child_flags,
                ]
                t0 = time.monotonic()
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    env=env,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                )
                active["ddp"] = (spec, lease, hb, proc, t0, log_fh)
                print(
                    f"{_utc_now()} [CLAIM ddp] nproc={nproc} run={spec.run_id}",
                    flush=True,
                )

                while proc.poll() is None and not stop_requested:
                    time.sleep(min(2.0, args.poll_interval_sec))

                if stop_requested:
                    _graceful_shutdown()
                    return PREEMPTION_EXIT_CODE

                rc = proc.wait()
                hb.stop()
                manager.release(lease)
                log_fh.close()
                active.pop("ddp", None)
                elapsed = time.monotonic() - t0
                if rc == 0:
                    print(
                        f"{_utc_now()} [COMPLETE ddp] run={spec.run_id} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                elif rc == PREEMPTION_EXIT_CODE:
                    print(
                        f"{_utc_now()} [PREEMPTED ddp] run={spec.run_id} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    return PREEMPTION_EXIT_CODE
                else:
                    print(
                        f"{_utc_now()} [FAILED ddp] run={spec.run_id} rc={rc} elapsed={elapsed:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    if args.fail_fast:
                        return rc
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)
        if active:
            _graceful_shutdown()

    return 0


def _parse_int_csv(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return ()
    return tuple(sorted({int(tok.strip()) for tok in raw.split(",") if tok.strip()}))


def build_arg_parser() -> argparse.ArgumentParser:
    default_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to JSON manifest listing Stage-1 RunSpec entries",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Shared output directory (runs/, checkpoints/, leases/, logs/)",
    )
    parser.add_argument(
        "--data-cache-dir",
        type=Path,
        default=None,
        help="Optional shared directory for cached .npy sequence datasets",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU indices (e.g. '0,1,2,3'), 'cpu', or auto-detect",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "single_gpu", "ddp"),
        default="auto",
        help="Execution mode: single_gpu parallel slots, full-node torchrun ddp, or auto",
    )
    parser.add_argument(
        "--allow-ddp-on-single-gpu",
        action="store_true",
        help="Allow single_gpu slots to claim ddp_preferred runs",
    )
    parser.add_argument(
        "--single-run-id",
        type=str,
        default=None,
        help="Internal worker mode: execute one run_id from --manifest and exit",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_dev,
        help="PyTorch device for --single-run-id mode",
    )
    parser.add_argument(
        "--lease-ttl-sec",
        type=float,
        default=DEFAULT_LEASE_TTL_SEC,
        help="Lease expiration TTL in seconds before stale leases are reclaimed (default: 600)",
    )
    parser.add_argument(
        "--heartbeat-interval-sec",
        type=float,
        default=DEFAULT_HEARTBEAT_INTERVAL_SEC,
        help="Lease heartbeat renewal interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--checkpoint-interval-sec",
        type=float,
        default=DEFAULT_CHECKPOINT_INTERVAL_SEC,
        help="Time-based atomic checkpoint interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--shutdown-grace-sec",
        type=float,
        default=60.0,
        help="Seconds to wait for child workers to flush SIGTERM checkpoints (default: 60)",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=float,
        default=2.0,
        help="Worker loop poll interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--eval-max-mlm",
        type=int,
        default=65536,
        help="Max sequences evaluated per MLM evaluation pass (default: 65536)",
    )
    parser.add_argument(
        "--cross-eval-epochs",
        type=str,
        default=None,
        help="Optional comma-separated milestone epochs for cross-objective MLM diagnostics",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run from scratch instead of skipping/resuming",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort worker pool immediately if any child exits nonzero (other than preemption 143)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.cross_eval_epochs = _parse_int_csv(args.cross_eval_epochs)
    if args.single_run_id is not None:
        return _run_manifest_single(args)
    return _run_spot_pool(args)


if __name__ == "__main__":
    raise SystemExit(main())
