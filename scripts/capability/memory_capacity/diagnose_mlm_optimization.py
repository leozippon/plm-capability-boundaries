#!/usr/bin/env python3
"""Train one MLM arm on synthetic IID tokens and emit a JSON diagnostic to stdout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.utils import clip_grad_norm_  # noqa: E402

from src.capability.memory_capacity.data import generate_iid_aa_dataset  # noqa: E402
from src.capability.memory_capacity.evaluation import (  # noqa: E402
    DEFAULT_LOO_PROBES_PER_SEQ,
    MLM_NUM_STRATA,
    evaluate_mlm_joint_bits_16pt,
    evaluate_mlm_loo_diagnostic,
)
from src.capability.memory_capacity.model import (  # noqa: E402
    UniformVocabTransformer,
    count_model_parameters,
)
from src.capability.memory_capacity.spec import (  # noqa: E402
    LN2,
    SCALE_SPECS,
    SEQ_LEN,
    SPECIAL_TOKEN_ID,
    VOCAB_AA,
    assert_no_forbidden_infra_strings,
    build_campaign_runs,
    choose_global_batch_size,
)
from src.capability.memory_capacity.training import (  # noqa: E402
    build_optimizer,
    compute_cosine_lr,
    compute_mlm_batch_loss,
    compute_peak_lr,
    compute_warmup_steps,
    sample_mlm_stratified_batch,
)


def generator(device: torch.device | str, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def fixed_mask(shape: torch.Size, k: int, rng: torch.Generator) -> torch.Tensor:
    scores = torch.rand(shape, device=rng.device, generator=rng)
    positions = scores.topk(k, dim=-1, largest=False).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, positions, True)


@torch.no_grad()
def score_masks(model, data, masks, batch_size):
    totals = torch.zeros(3, device=data.device, dtype=torch.float64)
    for mask in masks:
        for targets, selected in zip(data.split(batch_size), mask.split(batch_size), strict=True):
            targets = targets.long()
            logits = model(targets.masked_fill(selected, SPECIAL_TOKEN_ID))
            nll = F.cross_entropy(logits.float().transpose(1, 2), targets, reduction="none")
            totals += torch.stack((
                (nll * selected).sum(dtype=torch.float64),
                ((logits.argmax(dim=-1) == targets) & selected).sum(),
                selected.sum(),
            ))
    nll, correct, count = totals.tolist()
    return {"bits_per_masked_token": nll / (LN2 * count),
            "accuracy": correct / count, "masked_tokens": int(count)}


def main() -> None:
    primary = next(r for r in build_campaign_runs(include_ablations=False)
                   if r.scale_id == "0.5m" and r.paradigm == "oa_uniform")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("oa", "dynamic", "repeated"), default="oa",
                        help="OA-ARDM, new exact 15%% masks each epoch, or freeze epoch-one masks")
    parser.add_argument("--epochs", type=int, default=primary.max_epochs,
                        help="Independent cosine horizon: 400 (default) or 2000; smaller for smoke")
    parser.add_argument("--lr-multiplier", type=float, default=1.0,
                        help="Multiply the production peak LR (1 baseline, 4 high)")
    parser.add_argument("--n-sequences", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--data-seed", type=int, default=primary.data_seed)
    parser.add_argument("--model-seed", type=int, default=primary.seed)
    parser.add_argument("--device", type=torch.device, default="cuda:0", help="cuda:INDEX or cpu")
    parser.add_argument("--compile", action="store_true", help="Compile only the training model")
    args = parser.parse_args()
    device = args.device
    if min(args.n_sequences, args.epochs) < 1 or args.seq_len < 2:
        parser.error("require positive --epochs and --n-sequences, and --seq-len >= 2")
    if not math.isfinite(args.lr_multiplier) or args.lr_multiplier <= 0:
        parser.error("--lr-multiplier must be positive and finite")
    if os.environ.get("WORLD_SIZE", "1") != "1" or device.type not in ("cuda", "cpu"):
        parser.error("use a single process on one CUDA GPU or CPU")
    cuda = device.type == "cuda"
    if cuda:
        if not torch.cuda.is_available() or (device.index is not None and
                                             device.index >= torch.cuda.device_count()):
            parser.error("requested CUDA device is unavailable; use --device cpu for a smoke test")
        torch.cuda.set_device(device.index if device.index is not None else 0)
        free, total = torch.cuda.mem_get_info(device)
        print(f"CUDA memory free/total: {free // 2**20}/{total // 2**20} MiB", file=sys.stderr)
    print(f"arm={args.arm} device={device} epochs={args.epochs} lr_multiplier={args.lr_multiplier:g}",
          file=sys.stderr, flush=True)

    start = time.perf_counter()
    scale = SCALE_SPECS[primary.scale_id]
    batch_size = choose_global_batch_size(args.n_sequences, scale.max_batch_size)
    steps_per_epoch = math.ceil(args.n_sequences / batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = compute_warmup_steps(total_steps)
    peak = compute_peak_lr(scale.base_lr, batch_size) * args.lr_multiplier
    torch.backends.cuda.matmul.allow_tf32 = cuda and primary.precision == "bf16"
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    amp = torch.autocast(device.type, dtype=torch.bfloat16,
                         enabled=cuda and primary.precision == "bf16")

    sequences = generate_iid_aa_dataset(args.n_sequences, seq_len=args.seq_len, data_seed=args.data_seed)
    data = torch.from_numpy(sequences).to(device)
    torch.manual_seed(args.model_seed)
    model = UniformVocabTransformer(scale, "mlm", seq_len=args.seq_len).to(device)
    counts = count_model_parameters(model)
    optimizer = build_optimizer(model, peak_lr=peak, weight_decay=primary.weight_decay, device=device)
    train_model = torch.compile(model) if args.compile else model
    positions = torch.arange(args.seq_len, device=device)
    k = min(args.seq_len - 1, max(1, round(0.15 * args.seq_len)))

    # Match production order/mask streams (rank 0). Dynamic and repeated start with identical masks.
    def mask_rng(epoch):
        return generator(device, args.model_seed * 10_000_019 + epoch * 1_009)

    repeated = fixed_mask(data.shape, k, mask_rng(1)) if args.arm == "repeated" else None
    # Separate evaluation streams; never used for training or choosing when to stop.
    oa_seed, loo_seed, fixed_seed = (args.data_seed + offset for offset in (17, 31, 47))
    probe_rng = generator("cpu", fixed_seed)
    probe_masks = [fixed_mask(data.shape, k, probe_rng).to(device) for _ in range(MLM_NUM_STRATA)]
    non_emb = counts["n_params_non_embedding"]

    @torch.no_grad()
    def evaluate(epoch, step, train_elapsed, **training):
        model.eval()
        with amp:
            right_context = data[:1].long().repeat(2, 1)
            right_context[:, 0] = SPECIAL_TOKEN_ID
            right_context[1, 1] = (right_context[1, 1] + 1) % VOCAB_AA
            logits = model(right_context)[:, 0].float()
            fixed = score_masks(model, data, probe_masks, batch_size)
            repeat_metrics = score_masks(model, data, [repeated], batch_size) if repeated is not None else None
        oa = evaluate_mlm_joint_bits_16pt(
            model, data, batch_size=batch_size, precision=primary.precision, eval_seed=oa_seed,
            n_params_non_embedding=non_emb, n_params_total=counts["n_params_total"],
            eval_max_sequences_mlm=args.n_sequences,
        )
        loo = evaluate_mlm_loo_diagnostic(
            model, data, batch_size=batch_size, precision=primary.precision, loo_seed=loo_seed,
            n_params_non_embedding=non_emb, max_sequences_loo=args.n_sequences,
        )
        result = dict(epoch=epoch, step=step, train_elapsed_sec=train_elapsed,
                      elapsed_sec=time.perf_counter() - start, fixed_15pct=fixed, oa=oa, loo=loo,
                      right_visible_token_logit_delta_max_abs=(logits[1] - logits[0]).abs().max().item(),
                      **training)
        if repeat_metrics is not None:
            result["repeated_train_mask"] = repeat_metrics
        print(f"epoch={epoch} step={step} elapsed={result['elapsed_sec']:.2f}s "
              f"fixed_mask_bits={fixed['bits_per_masked_token']:.4f}", file=sys.stderr, flush=True)
        return result

    snapshots = [evaluate(0, 0, 0.0)]
    check_params = {name: param for name, param in model.named_parameters() if name in {
        "lm_head.weight", "blocks.0.attn.W_qkv.weight", "blocks.0.attn.W_o.weight",
    }}
    first_step = {}
    step = 0
    lr = peak
    train_elapsed = 0.0
    train_start = time.perf_counter()
    padded_n = steps_per_epoch * batch_size
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = mask_rng(epoch)
        masks = fixed_mask(data.shape, k, rng) if args.arm == "dynamic" else repeated
        perm = torch.randperm(args.n_sequences, device=device,
                              generator=generator(device, args.model_seed * 1_000_003 + epoch))
        perm = perm.repeat(math.ceil(padded_n / args.n_sequences))[:padded_n]
        epoch_loss = torch.zeros((), device=device, dtype=torch.float64)
        for indices in perm.split(batch_size):
            step += 1
            lr = compute_cosine_lr(step, total_steps=total_steps, peak_lr=peak, warmup_steps=warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            batch = data.index_select(0, indices)
            if masks is None:
                inputs, targets, mask, n_mask = sample_mlm_stratified_batch(
                    batch, generator=rng, pos_arange=positions)
            else:
                targets, mask = batch.long(), masks.index_select(0, indices)
                inputs, n_mask = targets.masked_fill(mask, SPECIAL_TOKEN_ID), mask.sum(dim=-1)
            with amp:
                logits = train_model(inputs)
            loss = compute_mlm_batch_loss(logits, targets, mask, n_mask)
            if step == 1:
                logits.retain_grad()
            loss.backward()
            before = None
            if step == 1:
                if logits.grad is None:
                    raise RuntimeError("Training loss did not produce logit gradients")
                visible = logits.grad[~mask]
                first_step = dict(
                    masked_logit_grad_max_abs=logits.grad[mask].abs().max().item(),
                    visible_logit_grad_max_abs=visible.abs().max().item() if visible.numel() else None,
                    visible_positions=visible.shape[0], masked_positions=mask.sum().item(),
                )
                before = {name: (p.detach().clone(), p.grad.norm().item() if p.grad is not None else None)
                          for name, p in check_params.items()}
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if before is not None:
                first_step["grad_l2_before_clip"] = grad_norm.item()
                first_step["parameters"] = {
                    name: dict(grad_l2_before_clip=grad, delta_l2=(check_params[name].detach() - old).norm().item())
                    for name, (old, grad) in before.items()
                }
            epoch_loss += loss.detach().double()
        if epoch in (primary.max_epochs, args.epochs):
            loss_bits = epoch_loss.item() / (steps_per_epoch * LN2)  # Also synchronizes CUDA.
            train_elapsed += time.perf_counter() - train_start
            snapshots.append(evaluate(epoch, step, train_elapsed, lr=lr, train_loss_bits=loss_bits))
            train_start = time.perf_counter()

    payload = dict(
        arm=args.arm, n_sequences=args.n_sequences, seq_len=args.seq_len, data_seed=args.data_seed,
        model_seed=args.model_seed, data_sha256=hashlib.sha256(sequences.tobytes()).hexdigest(),
        scale=scale.scale_id, parameter_counts=counts, device=str(device), compiled=args.compile,
        precision=primary.precision if cuda else "fp32", weight_decay=primary.weight_decay,
        lr_multiplier=args.lr_multiplier, peak_lr=peak, warmup_steps=warmup, epochs=args.epochs,
        batch_size=batch_size, steps_per_epoch=steps_per_epoch, steps=step,
        train_elapsed_sec=train_elapsed, elapsed_sec=time.perf_counter() - start,
        probe=dict(sequences="in_sample", oa_seed=oa_seed, loo_seed=loo_seed, fixed_seed=fixed_seed,
                   fixed_masks_per_sequence=len(probe_masks), fixed_mask_tokens_per_sequence=k,
                   loo_probes_per_sequence=min(DEFAULT_LOO_PROBES_PER_SEQ, args.seq_len)),
        first_step=first_step, evaluations=snapshots,
    )
    assert_no_forbidden_infra_strings(payload, label="mlm_optimization_diagnostic")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
