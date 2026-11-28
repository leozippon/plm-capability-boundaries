#!/usr/bin/env python3
"""Train a Stage-2 arm while saving frozen milestone .pt snapshots for 2D (E, layer l) probing.

Uses the exact matched ``VariableLengthProteinTransformer``, objectives, AdamW optimizer,
WSD/cosine LR schedule, and deterministic batch/mask seeds from ``src.capability.supply_demand``,
saving lightweight frozen model snapshots at ``E in (0, 1, 2, 5, 10, 20, 40, 80, 150, 300)``
(including random initialization at ``E = 0``).
"""
from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.probing.analysis import aggregate_probe_results, evaluate_checkpoint_probes
from src.capability.probing.extractor import (
    canonical_objective,
    load_stage2_checkpoint,
    paradigm_display_name,
)
from src.capability.supply_demand.corpus import Stage2Bundle, atomic_write_json_with_sha256
from src.capability.supply_demand.model import (
    VariableLengthProteinTransformer,
    count_model_parameters,
)
from src.capability.supply_demand.objectives import (
    compute_objective_loss,
    prepare_objective_batch,
)
from src.capability.supply_demand.spec import (
    LN2,
    SCALE_SPECS,
    Stage2RunSpec,
    slugify_token,
)
from src.capability.supply_demand.training import (
    CHECKPOINT_SCHEMA,
    _build_epoch_batches,
    atomic_torch_save,
    build_optimizer,
    compute_lr,
    compute_peak_lr,
    compute_warmup_steps,
)

DEFAULT_SNAPSHOT_EPOCHS: tuple[int, ...] = (0, 1, 2, 5, 10, 20, 40, 80, 150, 300)
TRAJECTORY_MANIFEST_SCHEMA: str = "stage2_probe_trajectory_manifest_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a Stage-2 arm and save frozen milestone .pt snapshots for layerwise probing."
    )
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--arm-id", type=str, required=True)
    p.add_argument(
        "--objective",
        type=str,
        required=True,
        choices=["ar", "mlm_15", "oa_uniform", "mdm"],
        help="Training objective ('mdm' is canonical alias for 'oa_uniform').",
    )
    p.add_argument("--scale-id", type=str, default="8m", choices=list(SCALE_SPECS.keys()))
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument(
        "--snapshot-epochs",
        type=str,
        default=",".join(str(e) for e in DEFAULT_SNAPSHOT_EPOCHS),
        help="Comma-separated epochs E at which to save frozen .pt snapshots (0 = random init).",
    )
    p.add_argument("--lr-schedule", type=str, default="wsd", choices=["wsd", "cosine"])
    p.add_argument("--peak-lr", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32_strict"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=20261128)
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--use-compile", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume training from _latest.pt if present.")
    p.add_argument(
        "--run-probes-after",
        action="store_true",
        help="Immediately run the layerwise probe suite on all saved milestone snapshots.",
    )
    p.add_argument("--skip-zeroshot-contacts", action="store_true")
    p.add_argument("--max-variants-per-wt", type=int, default=400)
    return p.parse_args(argv)


def _parse_snapshot_epochs(raw: str, max_epochs: int) -> tuple[int, ...]:
    vals = sorted({int(tok.strip()) for tok in raw.split(",") if tok.strip()})
    kept = [e for e in vals if 0 <= e <= int(max_epochs)]
    if int(max_epochs) not in kept:
        kept.append(int(max_epochs))
    return tuple(sorted(set(kept)))


def train_probe_trajectory(
    *,
    bundle_dir: Path | str,
    out_dir: Path | str,
    arm_id: str,
    objective: str,
    scale_id: str,
    max_epochs: int = 300,
    snapshot_epochs: tuple[int, ...] = DEFAULT_SNAPSHOT_EPOCHS,
    lr_schedule: str = "wsd",
    peak_lr_override: float | None = None,
    warmup_steps_override: int | None = None,
    weight_decay: float = 0.01,
    precision: str = "bf16",
    batch_size: int = 128,
    grad_accum_steps: int = 1,
    seed: int = 20261128,
    run_id: str = "",
    device: torch.device | str | None = None,
    use_compile: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Train one Stage-2 arm and persist frozen milestone ``.pt`` snapshots."""
    obj = canonical_objective(objective)
    paradigm_disp = paradigm_display_name(obj)
    scale_spec = SCALE_SPECS[scale_id]

    snaps = tuple(sorted({e for e in snapshot_epochs if 0 <= int(e) <= int(max_epochs)}))
    if int(max_epochs) not in snaps:
        snaps = tuple(sorted({*snaps, int(max_epochs)}))
    eval_pos = tuple(e for e in snaps if e >= 1)

    if not run_id:
        run_id = (
            f"probe_traj_{slugify_token(arm_id)}_{slugify_token(obj)}"
            f"_{slugify_token(scale_id)}_e{int(max_epochs)}"
        )

    run_spec = Stage2RunSpec(
        run_id=run_id,
        arm_id=arm_id,
        objective=obj,
        scale_id=scale_id,
        max_epochs=int(max_epochs),
        eval_epochs=eval_pos,
        lr_schedule=lr_schedule,
        peak_lr_override=peak_lr_override,
        warmup_steps_override=warmup_steps_override,
        weight_decay=float(weight_decay),
        precision=precision,
        batch_size=int(batch_size),
        grad_accum_steps=int(grad_accum_steps),
        seed=int(seed),
    )

    if device is not None and str(device):
        dev = torch.device(device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")
    if dev.type == "cuda" and dev.index is not None:
        torch.cuda.set_device(dev.index)

    out_root = Path(out_dir)
    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / f"{run_id}_latest.pt"

    bundle = Stage2Bundle(bundle_dir)
    splits = bundle.load_arm_splits(arm_id)
    train_store = splits["train"]

    micro_bs = int(run_spec.batch_size)
    grad_accum = int(run_spec.grad_accum_steps)
    eff_bs = micro_bs * grad_accum
    n_micro_per_epoch = math.ceil(float(train_store.n_sequences) / float(micro_bs))
    steps_per_epoch = math.ceil(float(n_micro_per_epoch) / float(grad_accum))
    total_steps = steps_per_epoch * int(run_spec.max_epochs)
    peak_lr = compute_peak_lr(scale_spec.base_lr, eff_bs, override=run_spec.peak_lr_override)
    warmup_steps = compute_warmup_steps(total_steps, override=run_spec.warmup_steps_override)

    prev_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
    if run_spec.precision == "bf16":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_autocast = dev.type == "cuda"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        use_autocast = False
    autocast_ctx: Any = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_autocast
        else contextlib.nullcontext()
    )

    try:
        torch.manual_seed(run_spec.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_spec.seed)

        raw_model = VariableLengthProteinTransformer(
            scale_spec,
            run_spec.objective,
            max_seq_len=run_spec.max_seq_len,
        ).to(dev)
        param_counts = count_model_parameters(raw_model, scale_spec)
        optimizer = build_optimizer(
            raw_model,
            peak_lr=peak_lr,
            weight_decay=run_spec.weight_decay,
            device=dev,
        )

        start_epoch = 1
        global_step = 0
        tokens_seen = 0
        prior_wall = 0.0
        saved_snapshots: list[dict[str, Any]] = []

        if resume and latest_path.is_file():
            ckpt = torch.load(latest_path, map_location=dev, weights_only=False)
            raw_model.load_state_dict(ckpt["model_state_dict"])
            if ckpt.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = int(ckpt["epoch"]) + 1
            global_step = int(ckpt["global_step"])
            tokens_seen = int(ckpt.get("tokens_seen", 0))
            prior_wall = float(ckpt.get("elapsed_wall_sec", 0.0))
            saved_snapshots = list(ckpt.get("saved_snapshots", []))

        train_model: Any = torch.compile(raw_model) if use_compile else raw_model
        t0 = time.perf_counter()

        def _save_snapshot(
            epoch_num: int,
            *,
            last_lr: float,
            mean_train_loss_nats: float | None,
        ) -> Path:
            snap_path = ckpt_dir / f"{run_id}_e{epoch_num:04d}.pt"
            wall = prior_wall + (time.perf_counter() - t0)
            payload = {
                "schema": CHECKPOINT_SCHEMA,
                "run_spec": run_spec.to_dict(),
                "paradigm_display": paradigm_disp,
                "E": int(epoch_num),
                "epoch": int(epoch_num),
                "global_step": int(global_step),
                "tokens_seen": int(tokens_seen),
                "lr": float(last_lr),
                "train_loss_nats": mean_train_loss_nats,
                "train_loss_bits_per_res": (
                    None if mean_train_loss_nats is None else float(mean_train_loss_nats / LN2)
                ),
                "elapsed_wall_sec": float(wall),
                "model_state_dict": {k: v.detach().cpu() for k, v in raw_model.state_dict().items()},
            }
            atomic_torch_save(snap_path, payload)
            rec = {
                "E": int(epoch_num),
                "epoch": int(epoch_num),
                "checkpoint_file": snap_path.name,
                "global_step": int(global_step),
                "tokens_seen": int(tokens_seen),
                "train_loss_bits_per_res": payload["train_loss_bits_per_res"],
                "elapsed_wall_sec": float(wall),
            }
            # Replace if resuming over same epoch
            saved_snapshots[:] = [s for s in saved_snapshots if int(s["E"]) != int(epoch_num)]
            saved_snapshots.append(rec)
            saved_snapshots.sort(key=lambda s: int(s["E"]))
            return snap_path

        # Save E=0 random initialization snapshot before any gradient step
        if 0 in snaps and start_epoch == 1 and not (ckpt_dir / f"{run_id}_e0000.pt").is_file():
            _save_snapshot(0, last_lr=0.0, mean_train_loss_nats=None)
            print(f"[train_probe_trajectory] saved E=0 snapshot: {run_id}_e0000.pt", flush=True)

        snap_set = set(snaps)
        for epoch in range(start_epoch, int(run_spec.max_epochs) + 1):
            train_model.train()
            micro_batches = _build_epoch_batches(
                train_store.n_sequences,
                micro_bs,
                seed=run_spec.seed,
                epoch=epoch,
            )
            ep_loss_sum = 0.0
            ep_loss_steps = 0
            last_lr = peak_lr

            for step_idx in range(steps_per_epoch):
                global_step += 1
                last_lr = compute_lr(
                    global_step,
                    total_steps=total_steps,
                    peak_lr=peak_lr,
                    warmup_steps=warmup_steps,
                    schedule=run_spec.lr_schedule,
                    cooldown_fraction=run_spec.cooldown_fraction,
                    min_lr_ratio=run_spec.min_lr_ratio,
                )
                for group in optimizer.param_groups:
                    group["lr"] = last_lr

                mb_lo = step_idx * grad_accum
                mb_hi = min(len(micro_batches), mb_lo + grad_accum)
                actual_accum = max(1, mb_hi - mb_lo)

                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                for mb_offset, mb_i in enumerate(range(mb_lo, mb_hi)):
                    idx_arr = micro_batches[mb_i]
                    tokens, lengths = train_store.collate_indices(
                        idx_arr, device=dev, pad_to_multiple=8
                    )
                    mask_gen = torch.Generator(device=dev)
                    mask_gen.manual_seed(
                        int(run_spec.seed) * 10_000_019
                        + int(epoch) * 100_003
                        + int(step_idx) * 97
                        + int(mb_offset)
                    )
                    prepared = prepare_objective_batch(
                        run_spec.objective,
                        tokens,
                        lengths,
                        generator=mask_gen,
                    )
                    with autocast_ctx:
                        logits = train_model(
                            prepared.input_ids,
                            prepared.attention_mask,
                            right_padded=True,
                        )
                    loss = compute_objective_loss(logits, prepared)
                    (loss / float(actual_accum)).backward()
                    step_loss += float(loss.detach().item()) / float(actual_accum)
                    tokens_seen += int(lengths.sum().item())

                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
                optimizer.step()
                ep_loss_sum += step_loss
                ep_loss_steps += 1

            mean_loss = ep_loss_sum / float(max(1, ep_loss_steps))
            if epoch in snap_set:
                snap_p = _save_snapshot(epoch, last_lr=last_lr, mean_train_loss_nats=mean_loss)
                # Update resumable _latest.pt at milestone boundaries
                atomic_torch_save(
                    latest_path,
                    {
                        "schema": CHECKPOINT_SCHEMA,
                        "run_spec": run_spec.to_dict(),
                        "paradigm_display": paradigm_disp,
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        "tokens_seen": int(tokens_seen),
                        "elapsed_wall_sec": float(prior_wall + (time.perf_counter() - t0)),
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "saved_snapshots": saved_snapshots,
                    },
                )
                print(
                    f"[train_probe_trajectory] E={epoch}/{run_spec.max_epochs} "
                    f"loss={mean_loss / LN2:.3f} bpr lr={last_lr:.2e} -> {snap_p.name}",
                    flush=True,
                )

        # Clean up _latest.pt after clean finish so checkpoint scans only see clean milestone snapshots
        if latest_path.is_file():
            latest_path.unlink(missing_ok=True)

        manifest = {
            "schema": TRAJECTORY_MANIFEST_SCHEMA,
            "run_id": run_id,
            "arm_id": arm_id,
            "objective": obj,
            "paradigm_display": paradigm_disp,
            "scale_id": scale_id,
            "parameter_counts": param_counts,
            "run_spec": run_spec.to_dict(),
            "snapshot_epochs": [int(e) for e in snaps],
            "checkpoints_dir": str(ckpt_dir.name),
            "snapshots": saved_snapshots,
            "wall_time_sec": float(prior_wall + (time.perf_counter() - t0)),
        }
        manifest_path = out_root / f"{run_id}_trajectory_manifest.json"
        atomic_write_json_with_sha256(manifest_path, manifest)
        return manifest
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_mm
        torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    snaps = _parse_snapshot_epochs(args.snapshot_epochs, int(args.max_epochs))
    manifest = train_probe_trajectory(
        bundle_dir=args.bundle_dir,
        out_dir=args.out_dir,
        arm_id=args.arm_id,
        objective=args.objective,
        scale_id=args.scale_id,
        max_epochs=int(args.max_epochs),
        snapshot_epochs=snaps,
        lr_schedule=args.lr_schedule,
        peak_lr_override=args.peak_lr,
        warmup_steps_override=args.warmup_steps,
        weight_decay=float(args.weight_decay),
        precision=args.precision,
        batch_size=int(args.batch_size),
        grad_accum_steps=int(args.grad_accum_steps),
        seed=int(args.seed),
        run_id=args.run_id,
        device=args.device or None,
        use_compile=bool(args.use_compile),
        resume=bool(args.resume),
    )

    if args.run_probes_after:
        dev = torch.device(args.device) if args.device else (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        bundle = Stage2Bundle(args.bundle_dir)
        ms_data = bundle.load_megascale()
        ct_data = bundle.load_contacts()
        unseen_store = bundle.load_arm_splits(args.arm_id)["eval_unseen_fold"]
        ckpt_dir = Path(args.out_dir) / "checkpoints"
        probes_out = Path(args.out_dir) / "probes"
        per_ckpt_dir = probes_out / "checkpoints"
        per_ckpt_dir.mkdir(parents=True, exist_ok=True)
        max_vars = None if int(args.max_variants_per_wt) <= 0 else int(args.max_variants_per_wt)

        docs: list[dict[str, Any]] = []
        for snap in manifest["snapshots"]:
            cp = ckpt_dir / snap["checkpoint_file"]
            loaded = load_stage2_checkpoint(cp, device=dev)
            doc = evaluate_checkpoint_probes(
                loaded,
                bundle,
                device=dev,
                precision=args.precision,
                max_variants_per_wt=max_vars,
                compute_zeroshot_contacts=not bool(args.skip_zeroshot_contacts),
                preloaded_megascale=ms_data,
                preloaded_contacts=ct_data,
                preloaded_unseen_fold_store=unseen_store,
            )
            atomic_write_json_with_sha256(per_ckpt_dir / f"{cp.stem}.probe.json", doc)
            docs.append(doc)
        summary = aggregate_probe_results(docs)
        atomic_write_json_with_sha256(probes_out / "probe_summary.json", summary)
        print(
            f"[train_probe_trajectory] probed {summary['n_checkpoints']} snapshots -> "
            f"{probes_out / 'probe_summary.json'}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
