#!/usr/bin/env python3
"""Spot-safe shared-storage lease worker for controlled multimodal (pure_seq, struct_cond, joint_to_seq) runs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.multimodal.data import Multimodal3DBundle  # noqa: E402
from src.capability.multimodal.spec import (  # noqa: E402
    ALL_OBJECTIVES,
    MULTIMODAL_MODES,
    OPTIMIZERS,
    SCALE_SPECS,
    MultimodalRunSpec,
    canonicalize_mode,
    canonicalize_objective,
    choose_milestone_epochs,
    slugify_token,
)
from src.capability.multimodal.training import (  # noqa: E402
    execute_multimodal_run,
    is_multimodal_run_completed,
)
from src.capability.supply_demand.training import (  # noqa: E402
    PREEMPT_EXIT_CODE,
    SharedLeaseManager,
    setup_distributed,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--queue-json", type=Path, default=None)
    p.add_argument("--modes", type=str, default="pure_seq,struct_cond,joint_to_seq")
    p.add_argument("--objectives", type=str, default="ar,mlm_15,mdm")
    p.add_argument("--scales", type=str, default="0.5m,2m,8m,35m")
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--lambda-struct", type=float, default=1.0)
    p.add_argument("--lambda-contact", type=float, default=0.25)
    p.add_argument("--struct-drop-prob", type=float, default=0.50)
    p.add_argument("--optimizer", type=str, default="adamw", choices=list(OPTIMIZERS))
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32_strict"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--checkpoint-interval-sec", type=float, default=300.0)
    p.add_argument("--stale-lease-sec", type=float, default=900.0)
    p.add_argument("--seed", type=int, default=20261128)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--gradient-checkpointing", action="store_true", default=None)
    p.add_argument("--max-runs", type=int, default=0)
    p.add_argument("--no-downstream-milestones", action="store_true")
    p.add_argument("--no-contacts-milestones", action="store_true")
    return p.parse_args(argv)


def _build_queue(args: argparse.Namespace) -> list[MultimodalRunSpec]:
    if args.queue_json is not None:
        raw = json.loads(args.queue_json.read_text(encoding="utf-8"))
        items = raw["runs"] if isinstance(raw, dict) and "runs" in raw else raw
        return [MultimodalRunSpec.from_dict(it) for it in items]

    modes = [canonicalize_mode(m.strip()) for m in args.modes.split(",") if m.strip()]
    objectives = [canonicalize_objective(o.strip()) for o in args.objectives.split(",") if o.strip()]
    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    for m in modes:
        if m not in MULTIMODAL_MODES:
            raise ValueError(f"Unknown mode {m!r}")
    for o in objectives:
        if o not in ALL_OBJECTIVES:
            raise ValueError(f"Unknown objective {o!r}")
    for sc in scales:
        if sc not in SCALE_SPECS:
            raise ValueError(f"Unknown scale {sc!r}")

    eval_epochs = choose_milestone_epochs(int(args.max_epochs))
    queue: list[MultimodalRunSpec] = []
    for sc in scales:
        for obj in objectives:
            for mode in modes:
                run_id = (
                    f"mm3d_{slugify_token(mode)}_{slugify_token(obj)}"
                    f"_{slugify_token(sc)}_e{int(args.max_epochs)}"
                )
                queue.append(
                    MultimodalRunSpec(
                        run_id=run_id,
                        mode=mode,
                        objective=obj,
                        scale_id=sc,
                        max_epochs=int(args.max_epochs),
                        eval_epochs=eval_epochs,
                        lambda_struct=float(args.lambda_struct),
                        lambda_contact=float(args.lambda_contact),
                        struct_drop_prob=float(args.struct_drop_prob),
                        optimizer=args.optimizer,
                        muon_lr=float(args.muon_lr),
                        precision=args.precision,
                        batch_size=int(args.batch_size),
                        grad_accum_steps=int(args.grad_accum_steps),
                        seed=int(args.seed),
                        checkpoint_interval_sec=float(args.checkpoint_interval_sec),
                        eval_downstream_at_milestones=not args.no_downstream_milestones,
                        eval_contacts_at_milestones=not args.no_contacts_milestones,
                        gradient_checkpointing=args.gradient_checkpointing,
                    )
                )
    return queue


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    is_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    ctx = setup_distributed(device=None if is_ddp else args.device)
    bundle = Multimodal3DBundle(args.bundle_dir)
    out_dir = args.out_dir.resolve()
    runs_dir = out_dir / "runs"
    leases_dir = out_dir / "leases"
    if ctx.rank == 0:
        runs_dir.mkdir(parents=True, exist_ok=True)
    if ctx.is_distributed:
        dist.barrier()

    lease_mgr = SharedLeaseManager(leases_dir, stale_after_sec=float(args.stale_lease_sec)) if ctx.rank == 0 else None
    queue = _build_queue(args)

    completed = 0
    try:
        for spec in queue:
            if args.max_runs > 0 and completed >= args.max_runs:
                break
            lease = None
            should_run = False
            if ctx.rank == 0:
                if not is_multimodal_run_completed(runs_dir, spec.run_id):
                    assert lease_mgr is not None
                    lease = lease_mgr.try_claim(spec.run_id)
                    should_run = lease is not None
            if ctx.is_distributed:
                flag = torch.tensor([1 if should_run else 0], dtype=torch.int64, device=ctx.device)
                dist.broadcast(flag, src=0)
                should_run = bool(flag.item() > 0)
            if not should_run:
                continue

            if ctx.rank == 0:
                print(
                    f"[mm3d-worker] Claimed {spec.run_id} "
                    f"(mode={spec.mode}, obj={spec.objective}, scale={spec.scale_id}, world_size={ctx.world_size})",
                    flush=True,
                )
            try:
                res: dict[str, Any] = execute_multimodal_run(
                    spec,
                    bundle,
                    out_dir,
                    device=ctx.device,
                    resume=True,
                    lease_manager=lease_mgr,
                    active_lease=lease,
                    install_signal_handlers=True,
                )
            finally:
                if ctx.rank == 0 and lease_mgr is not None and lease is not None:
                    lease_mgr.release(lease)

            if res.get("status") == "preempted":
                return PREEMPT_EXIT_CODE
            completed += 1
            if ctx.rank == 0:
                tt = (res.get("final_eval") or {}).get("three_tier_bits", {}).get("native", {})
                print(
                    f"[mm3d-worker] Completed {spec.run_id}: "
                    f"H_uf={tt.get('H_unseen_fold')}, Delta_cal={tt.get('Delta_cal')}",
                    flush=True,
                )
        return 0
    finally:
        if ctx.is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
