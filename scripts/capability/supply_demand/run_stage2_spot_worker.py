#!/usr/bin/env python3
"""Spot-safe Stage-2 shared-storage lease worker for Cuihu H100 / multi-GPU queues.

Claims unclaimed runs atomically from a shared run-queue directory via O_CREAT|O_EXCL
leases, runs Spot-safe training with periodic (300s) and immediate SIGTERM atomic
checkpointing, exits with code 143 upon preemption so the Spot supervisor restarts cleanly,
and automatically skips verified completed runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.supply_demand.corpus import Stage2Bundle  # noqa: E402
from src.capability.supply_demand.spec import (  # noqa: E402
    OBJECTIVES,
    SCALE_SPECS,
    Stage2RunSpec,
    choose_milestone_epochs,
    slugify_token,
)
from src.capability.supply_demand.training import (  # noqa: E402
    PREEMPT_EXIT_CODE,
    SharedLeaseManager,
    execute_stage2_run,
    is_run_completed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-2 Spot-safe shared-storage lease worker.")
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--queue-json",
        type=Path,
        default=None,
        help="Optional JSON file containing a list of Stage2RunSpec dicts. "
        "If omitted, builds the grid from --arms, --objectives, --scales.",
    )
    p.add_argument("--arms", type=str, default="", help="Comma-separated arm IDs (default: all bundle arms).")
    p.add_argument("--objectives", type=str, default="ar,mlm_15,oa_uniform")
    p.add_argument("--scales", type=str, default="0.5m,2m,8m,35m")
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--lr-schedule", type=str, default="wsd", choices=["wsd", "cosine"])
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32_strict"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--checkpoint-interval-sec", type=float, default=300.0)
    p.add_argument("--stale-lease-sec", type=float, default=900.0)
    p.add_argument("--seed", type=int, default=20261128)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--use-compile", action="store_true")
    p.add_argument("--max-runs", type=int, default=0, help="Stop after completing N runs (0 = drain queue).")
    p.add_argument("--no-downstream-milestones", action="store_true")
    p.add_argument("--no-contacts-milestones", action="store_true")
    return p.parse_args()


def _build_run_queue(args: argparse.Namespace, bundle: Stage2Bundle) -> list[Stage2RunSpec]:
    if args.queue_json is not None:
        raw = json.loads(args.queue_json.read_text(encoding="utf-8"))
        items = raw["runs"] if isinstance(raw, dict) and "runs" in raw else raw
        return [Stage2RunSpec.from_dict(item) for item in items]

    arms = (
        [a.strip() for a in args.arms.split(",") if a.strip()]
        if args.arms.strip()
        else list(bundle.arm_ids)
    )
    objectives = [o.strip() for o in args.objectives.split(",") if o.strip()]
    scales = [s.strip() for s in args.scales.split(",") if s.strip()]

    for obj in objectives:
        if obj not in OBJECTIVES:
            raise ValueError(f"Unknown objective {obj!r}")
    for sc in scales:
        if sc not in SCALE_SPECS:
            raise ValueError(f"Unknown scale {sc!r}")

    eval_epochs = choose_milestone_epochs(int(args.max_epochs))
    queue: list[Stage2RunSpec] = []
    for scale_id in scales:
        for obj in objectives:
            for arm_id in arms:
                run_id = (
                    f"stage2_{slugify_token(arm_id)}_{slugify_token(obj)}"
                    f"_{slugify_token(scale_id)}_e{int(args.max_epochs)}"
                )
                queue.append(
                    Stage2RunSpec(
                        run_id=run_id,
                        arm_id=arm_id,
                        objective=obj,
                        scale_id=scale_id,
                        max_epochs=int(args.max_epochs),
                        eval_epochs=eval_epochs,
                        lr_schedule=args.lr_schedule,
                        weight_decay=float(args.weight_decay),
                        precision=args.precision,
                        batch_size=int(args.batch_size),
                        grad_accum_steps=int(args.grad_accum_steps),
                        seed=int(args.seed),
                        checkpoint_interval_sec=float(args.checkpoint_interval_sec),
                        eval_downstream_at_milestones=not args.no_downstream_milestones,
                        eval_contacts_at_milestones=not args.no_contacts_milestones,
                    )
                )
    return queue


def main() -> None:
    args = parse_args()
    bundle = Stage2Bundle(args.bundle_dir)
    out_dir = args.out_dir.resolve()
    runs_dir = out_dir / "runs"
    leases_dir = out_dir / "leases"
    runs_dir.mkdir(parents=True, exist_ok=True)

    lease_mgr = SharedLeaseManager(leases_dir, stale_after_sec=float(args.stale_lease_sec))
    queue = _build_run_queue(args, bundle)

    completed_by_me = 0
    for spec in queue:
        if args.max_runs > 0 and completed_by_me >= args.max_runs:
            break
        if is_run_completed(runs_dir, spec.run_id):
            continue

        lease = lease_mgr.try_claim(spec.run_id)
        if lease is None:
            continue

        print(
            f"[stage2-worker] Claimed {spec.run_id} "
            f"(arm={spec.arm_id}, obj={spec.objective}, scale={spec.scale_id})",
            flush=True,
        )
        try:
            res: dict[str, Any] = execute_stage2_run(
                spec,
                bundle,
                out_dir,
                device=args.device,
                use_compile=bool(args.use_compile),
                resume=True,
                lease_manager=lease_mgr,
                active_lease=lease,
                install_signal_handlers=True,
            )
        finally:
            lease_mgr.release(lease)

        if res.get("status") == "preempted":
            print(
                f"[stage2-worker] Preempted cleanly on {spec.run_id} at "
                f"epoch={res.get('epoch_done')} step_in_epoch={res.get('step_in_epoch')}; exiting {PREEMPT_EXIT_CODE}",
                flush=True,
            )
            sys.exit(PREEMPT_EXIT_CODE)

        completed_by_me += 1
        final_eq4 = (res.get("final_eval") or {}).get("three_tier_bits", {}).get("native", {})
        print(
            f"[stage2-worker] Completed {spec.run_id}: "
            f"dh_univ={final_eq4.get('delta_h_univ')}, "
            f"dh_spec={final_eq4.get('delta_h_spec')}, "
            f"dh_res={final_eq4.get('delta_h_res')}",
            flush=True,
        )

    print(f"[stage2-worker] Queue pass complete (finished {completed_by_me} runs).", flush=True)


if __name__ == "__main__":
    main()
