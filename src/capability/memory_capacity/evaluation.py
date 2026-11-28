"""Exact AR joint-bit, 16-stratum OA-ARDM joint-NELBO, and fixed-rate MLM-15% evaluation engines."""
from __future__ import annotations

import contextlib
import math
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .spec import (
    LN2,
    MLM_15_MASK_RATE,
    SPECIAL_TOKEN_ID,
    VOCAB_AA,
    canonicalize_paradigm,
    dataset_entropy_bits,
    gibbs_memorizable_bits,
    is_fixed_mask_paradigm,
    mlm_fixed_mask_count,
    null_ordered_entropy_bits,
    paradigm_display_name,
    vram_safe_micro_batch_cap,
)

DEFAULT_EVAL_MAX_SEQUENCES_MLM: int = 65_536
DEFAULT_LOO_MAX_SEQUENCES_MLM: int = 2_048
DEFAULT_LOO_PROBES_PER_SEQ: int = 16
MLM_NUM_STRATA: int = 16
DEFAULT_MLM_FIXED_NUM_DRAWS: int = 16


def _u64_to_i64(val: int) -> int:
    val &= (1 << 64) - 1
    return val - (1 << 64) if val >= (1 << 63) else val


def _ushift_right_i64(z: torch.Tensor, k: int) -> torch.Tensor:
    return (z >> k) & ((1 << (64 - k)) - 1)


_SM_C0 = _u64_to_i64(0x9E3779B97F4A7C15)
_SM_C1 = _u64_to_i64(0xD1B54A32D192ED03)
_SM_C2 = _u64_to_i64(0x94D049BB133111EB)
_SM_M1 = _u64_to_i64(0xBF58476D1CE4E5B9)
_SM_M2 = _u64_to_i64(0x94D049BB133111EB)


def _splitmix64_priorities(
    global_indices: torch.Tensor,
    stratum: int,
    seq_len: int,
    seed: int,
) -> torch.Tensor:
    """Deterministic rank- and batch-invariant (B, seq_len) int64 priorities on device."""
    device = global_indices.device
    pos = torch.arange(seq_len, device=device, dtype=torch.int64).unsqueeze(0)
    idx = global_indices.to(dtype=torch.int64).unsqueeze(1)
    base_seed = _u64_to_i64(int(seed))
    s_term = _u64_to_i64(int(stratum) * 0x517CC1B727220A95)

    z = base_seed + idx * _SM_C0 + s_term + pos * _SM_C1 + _SM_C2
    z = (z ^ _ushift_right_i64(z, 30)) * _SM_M1
    z = (z ^ _ushift_right_i64(z, 27)) * _SM_M2
    z = z ^ _ushift_right_i64(z, 31)
    return z


def compute_mlm_stratum_mask_counts(
    global_indices: torch.Tensor,
    stratum: int,
    seq_len: int,
) -> torch.Tensor:
    """Compute exact 16-stratum OA-ARDM mask counts n_{i,s} in [1, seq_len]."""
    if not (0 <= stratum < MLM_NUM_STRATA):
        raise ValueError(f"stratum must be in [0, {MLM_NUM_STRATA - 1}], got {stratum}")
    idx_i64 = global_indices.to(dtype=torch.int64)
    offset_mod16 = torch.remainder(idx_i64 + 7 * int(stratum), MLM_NUM_STRATA)
    if seq_len == 256:
        return 16 * int(stratum) + offset_mod16 + 1

    u = (
        float(stratum) + (offset_mod16.to(dtype=torch.float64) + 0.5) / float(MLM_NUM_STRATA)
    ) / float(MLM_NUM_STRATA)
    n_mask = torch.floor(u * float(seq_len)).to(dtype=torch.int64) + 1
    return torch.clamp(n_mask, 1, int(seq_len))


def deterministic_mask_for_stratum(
    global_indices: torch.Tensor,
    stratum: int,
    seq_len: int,
    *,
    eval_seed: int = 20261129,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic boolean mask (B, L) and mask count n_mask (B,) for (i, s)."""
    n_mask = compute_mlm_stratum_mask_counts(global_indices, stratum=stratum, seq_len=seq_len)
    priorities = _splitmix64_priorities(
        global_indices,
        stratum=stratum,
        seq_len=seq_len,
        seed=eval_seed,
    )
    order = torch.argsort(priorities, dim=-1, stable=True)
    pos_idx = torch.arange(seq_len, device=global_indices.device, dtype=torch.int64).unsqueeze(0)
    mask = torch.zeros(
        (global_indices.shape[0], seq_len),
        dtype=torch.bool,
        device=global_indices.device,
    )
    mask.scatter_(1, order, pos_idx < n_mask.unsqueeze(1))
    return mask, n_mask


def deterministic_fixed_rate_mask(
    global_indices: torch.Tensor,
    draw_idx: int,
    seq_len: int,
    mask_count: int,
    *,
    eval_seed: int = 20261141,
) -> torch.Tensor:
    """Generate deterministic rank- and batch-invariant boolean mask (B, L) with exact size mask_count."""
    if not (1 <= int(mask_count) <= int(seq_len)):
        raise ValueError(f"mask_count must be in [1, {seq_len}], got {mask_count}")
    priorities = _splitmix64_priorities(
        global_indices,
        stratum=draw_idx,
        seq_len=seq_len,
        seed=eval_seed,
    )
    order = torch.argsort(priorities, dim=-1, stable=True)
    mask = torch.zeros(
        (global_indices.shape[0], seq_len),
        dtype=torch.bool,
        device=global_indices.device,
    )
    mask.scatter_(1, order[:, : int(mask_count)], True)
    return mask


def _eval_autocast_context(device: torch.device, precision: str) -> Any:
    if precision == "bf16" and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _resolve_eval_batch_size(
    model: nn.Module,
    requested_batch_size: int,
    device: torch.device,
) -> int:
    req = max(1, int(requested_batch_size))
    if device.type != "cuda":
        return req
    unwrapped = model
    while hasattr(unwrapped, "module") or hasattr(unwrapped, "_orig_mod"):
        unwrapped = getattr(unwrapped, "module", getattr(unwrapped, "_orig_mod", unwrapped))
    spec = getattr(unwrapped, "spec", None)
    scale_id = getattr(spec, "scale_id", None)
    if not isinstance(scale_id, str):
        return req
    total_vram_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    return vram_safe_micro_batch_cap(scale_id, total_vram_bytes, req)


def summarize_memorization_metrics(
    *,
    paradigm: str,
    n_sequences_total: int,
    n_sequences_evaluated: int,
    seq_len: int,
    mean_seq_xent_bits: float,
    seq_xent_std_bits: float,
    eval_mc_se_seq_bits: float,
    n_params_non_embedding: int,
    n_params_total: int | None = None,
    vocab_size: int = VOCAB_AA,
) -> dict[str, Any]:
    """Compute unified joint-bit memorization accounting from mean per-sequence NLL in bits."""
    canon_paradigm = canonicalize_paradigm(paradigm)
    mean_tok_xent_bits = float(mean_seq_xent_bits) / float(seq_len)
    total_model_xent_bits = float(n_sequences_total) * float(mean_seq_xent_bits)

    h_null = null_ordered_entropy_bits(n_sequences_total, seq_len=seq_len, vocab_size=vocab_size)
    h_dataset = dataset_entropy_bits(n_sequences_total, seq_len=seq_len, vocab_size=vocab_size)
    h_gibbs = gibbs_memorizable_bits(n_sequences_total, seq_len=seq_len, vocab_size=vocab_size)

    seq_null_bits = float(seq_len) * math.log2(float(vocab_size))
    memorized_bits = float(n_sequences_total) * (seq_null_bits - float(mean_seq_xent_bits))
    memorized_bits_se = float(n_sequences_total) * float(eval_mc_se_seq_bits)

    denom_total_params = (
        int(n_params_total)
        if n_params_total is not None and n_params_total > 0
        else int(n_params_non_embedding)
    )
    bpp_non_emb = float(memorized_bits) / float(n_params_non_embedding)
    bpp_total = float(memorized_bits) / float(denom_total_params)
    frac_data = (float(memorized_bits) / float(h_dataset)) if h_dataset > 0.0 else 0.0
    frac_gibbs = (float(memorized_bits) / float(h_gibbs)) if h_gibbs > 0.0 else 0.0
    frac_null = (float(memorized_bits) / float(h_null)) if h_null > 0.0 else 0.0

    return {
        "paradigm": canon_paradigm,
        "paradigm_display": paradigm_display_name(canon_paradigm),
        "vocab_size": int(vocab_size),
        "n_sequences_total": int(n_sequences_total),
        "eval_n_sequences": int(n_sequences_evaluated),
        "n_sequences_evaluated": int(n_sequences_evaluated),
        "seq_len": int(seq_len),
        "token_xent_bits": float(mean_tok_xent_bits),
        "mean_tok_xent_bits": float(mean_tok_xent_bits),
        "mean_seq_xent_bits": float(mean_seq_xent_bits),
        "seq_xent_std_bits": float(seq_xent_std_bits),
        "eval_mc_se_seq_bits": float(eval_mc_se_seq_bits),
        "memorized_bits_se": float(memorized_bits_se),
        "total_model_xent_bits": float(total_model_xent_bits),
        "H_null_ordered_bits": float(h_null),
        "H_data_bits": float(h_dataset),
        "H_gibbs_max_bits": float(h_gibbs),
        "memorized_bits": float(memorized_bits),
        "bits_per_non_emb_param": float(bpp_non_emb),
        "bits_per_total_param": float(bpp_total),
        "memorized_bits_per_param": float(bpp_non_emb),
        "fraction_of_H_data": float(frac_data),
        "fraction_of_H_gibbs": float(frac_gibbs),
        "fraction_of_H_null": float(frac_null),
        "mlm_quadrature_16pt_bits_per_tok": None,
        "mlm_loo_pseudo_tok_xent_bits": None,
        "mlm_loo_pseudo_memorized_bits": None,
    }


@torch.no_grad()
def evaluate_ar_joint_bits(
    model: nn.Module,
    data_gpu: torch.Tensor,
    *,
    batch_size: int,
    n_params_non_embedding: int,
    n_params_total: int | None = None,
    precision: str = "bf16",
    vocab_size: int = VOCAB_AA,
    special_token_id: int = SPECIAL_TOKEN_ID,
) -> dict[str, Any]:
    """Exact joint cross-entropy evaluation over 100% of N training sequences for AR."""
    t0 = time.perf_counter()
    was_training = model.training
    model.eval()
    try:
        n_total, seq_len = int(data_gpu.shape[0]), int(data_gpu.shape[1])
        device = next(model.parameters()).device

        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        local_indices = torch.arange(rank, n_total, world_size, device=device, dtype=torch.int64)
        n_local = int(local_indices.numel())
        safe_batch = _resolve_eval_batch_size(model, batch_size, device)
        eff_batch = max(1, min(safe_batch, max(1, (n_total + world_size - 1) // world_size)))

        sum_seq_bits = torch.zeros((), dtype=torch.float64, device=device)
        sum_sq_seq_bits = torch.zeros((), dtype=torch.float64, device=device)
        token_correct = torch.zeros((), dtype=torch.float64, device=device)
        seq_exact = torch.zeros((), dtype=torch.float64, device=device)
        count_done = torch.zeros((), dtype=torch.float64, device=device)

        bos_col = torch.full(
            (eff_batch, 1),
            int(special_token_id),
            dtype=torch.long,
            device=device,
        )
        autocast_ctx = _eval_autocast_context(device, precision)

        for start in range(0, n_local, eff_batch):
            valid_b = min(eff_batch, n_local - start)
            batch_idx = local_indices[start : start + valid_b]
            if valid_b < eff_batch:
                pad = local_indices[:1].expand(eff_batch - valid_b)
                batch_idx = torch.cat((batch_idx, pad), dim=0)

            targets = data_gpu.index_select(0, batch_idx).to(device=device, dtype=torch.long)
            input_ids = torch.cat((bos_col, targets[:, :-1]), dim=1)

            with autocast_ctx:
                logits = model(input_ids)

            tok_nll_nats = F.cross_entropy(
                logits.float().transpose(1, 2),
                targets,
                reduction="none",
            )
            seq_bits = (tok_nll_nats[:valid_b].to(torch.float64) / LN2).sum(dim=-1)

            preds = logits[:valid_b].argmax(dim=-1)
            matches = preds == targets[:valid_b]

            sum_seq_bits += seq_bits.sum()
            sum_sq_seq_bits += (seq_bits * seq_bits).sum()
            token_correct += matches.sum().to(torch.float64)
            seq_exact += matches.all(dim=-1).sum().to(torch.float64)
            count_done += float(valid_b)

        stats = torch.stack(
            (count_done, sum_seq_bits, sum_sq_seq_bits, token_correct, seq_exact)
        )
        if world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        n_eval = int(round(float(stats[0].item())))
        if n_eval != n_total:
            raise RuntimeError(f"AR evaluation count mismatch: got {n_eval}, expected {n_total}")

        total_bits_sum = float(stats[1].item())
        total_bits_sq = float(stats[2].item())
        mean_seq_bits = total_bits_sum / float(n_eval)
        if n_eval > 1:
            var_seq_bits = max(
                0.0,
                (total_bits_sq - float(n_eval) * (mean_seq_bits ** 2)) / float(n_eval - 1),
            )
        else:
            var_seq_bits = 0.0

        metrics = summarize_memorization_metrics(
            paradigm="ar",
            n_sequences_total=n_total,
            n_sequences_evaluated=n_eval,
            seq_len=seq_len,
            mean_seq_xent_bits=mean_seq_bits,
            seq_xent_std_bits=math.sqrt(var_seq_bits),
            eval_mc_se_seq_bits=0.0,
            n_params_non_embedding=n_params_non_embedding,
            n_params_total=n_params_total,
            vocab_size=vocab_size,
        )
        metrics["token_accuracy"] = float(stats[3].item()) / float(n_eval * seq_len)
        metrics["seq_exact_match_rate"] = float(stats[4].item()) / float(n_eval)
        metrics["eval_wall_time_sec"] = float(time.perf_counter() - t0)
        return metrics
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def evaluate_mlm_joint_bits_16pt(
    model: nn.Module,
    data_gpu: torch.Tensor,
    *,
    batch_size: int,
    n_params_non_embedding: int,
    n_params_total: int | None = None,
    precision: str = "bf16",
    eval_max_sequences_mlm: int = DEFAULT_EVAL_MAX_SEQUENCES_MLM,
    eval_seed: int = 20261129,
    vocab_size: int = VOCAB_AA,
    special_token_id: int = SPECIAL_TOKEN_ID,
    paradigm: str = "oa_uniform",
) -> dict[str, Any]:
    """16-stratum OA-ARDM discrete integral joint NELBO evaluation in bits for MLM."""
    t0 = time.perf_counter()
    was_training = model.training
    model.eval()
    try:
        n_total, seq_len = int(data_gpu.shape[0]), int(data_gpu.shape[1])
        n_eval_target = min(n_total, int(eval_max_sequences_mlm))
        device = next(model.parameters()).device

        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        local_indices = torch.arange(
            rank, n_eval_target, world_size, device=device, dtype=torch.int64
        )
        n_local = int(local_indices.numel())
        safe_batch = _resolve_eval_batch_size(model, batch_size, device)
        eff_batch = max(
            1,
            min(safe_batch, max(1, (n_eval_target + world_size - 1) // world_size)),
        )

        sum_seq_bits = torch.zeros((), dtype=torch.float64, device=device)
        sum_sq_seq_bits = torch.zeros((), dtype=torch.float64, device=device)
        sum_h_sq = torch.zeros((), dtype=torch.float64, device=device)
        masked_correct = torch.zeros((), dtype=torch.float64, device=device)
        masked_total = torch.zeros((), dtype=torch.float64, device=device)
        count_done = torch.zeros((), dtype=torch.float64, device=device)
        stratum_sums = torch.zeros((MLM_NUM_STRATA,), dtype=torch.float64, device=device)

        autocast_ctx = _eval_autocast_context(device, precision)

        for start in range(0, n_local, eff_batch):
            valid_b = min(eff_batch, n_local - start)
            batch_idx = local_indices[start : start + valid_b]
            if valid_b < eff_batch:
                pad = local_indices[:1].expand(eff_batch - valid_b)
                batch_idx = torch.cat((batch_idx, pad), dim=0)

            targets = data_gpu.index_select(0, batch_idx).to(device=device, dtype=torch.long)
            h_bits_batch = torch.empty(
                (valid_b, MLM_NUM_STRATA), dtype=torch.float64, device=device
            )

            for s in range(MLM_NUM_STRATA):
                mask, n_mask = deterministic_mask_for_stratum(
                    batch_idx,
                    stratum=s,
                    seq_len=seq_len,
                    eval_seed=eval_seed,
                )
                masked_input = torch.where(mask, int(special_token_id), targets)
                with autocast_ctx:
                    logits = model(masked_input)

                tok_nll_nats = F.cross_entropy(
                    logits.float().transpose(1, 2),
                    targets,
                    reduction="none",
                )
                valid_mask = mask[:valid_b]
                valid_nll_bits = tok_nll_nats[:valid_b].to(torch.float64) / LN2
                masked_bits = torch.where(valid_mask, valid_nll_bits, 0.0)
                h_s = masked_bits.sum(dim=-1) / n_mask[:valid_b].to(torch.float64)
                h_bits_batch[:, s] = h_s
                stratum_sums[s] += h_s.sum()
                sum_h_sq += (h_s * h_s).sum()

                preds = logits[:valid_b].argmax(dim=-1)
                masked_correct += (
                    ((preds == targets[:valid_b]) & valid_mask).sum().to(torch.float64)
                )
                masked_total += valid_mask.sum().to(torch.float64)

            seq_xent_bits = float(seq_len) * h_bits_batch.mean(dim=-1)
            sum_seq_bits += seq_xent_bits.sum()
            sum_sq_seq_bits += (seq_xent_bits * seq_xent_bits).sum()
            count_done += float(valid_b)

        packed = torch.cat(
            (
                torch.stack(
                    (
                        count_done,
                        sum_seq_bits,
                        sum_sq_seq_bits,
                        masked_correct,
                        masked_total,
                        sum_h_sq,
                    )
                ),
                stratum_sums,
            ),
            dim=0,
        )
        if world_size > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)

        n_eval = int(round(float(packed[0].item())))
        if n_eval != n_eval_target:
            raise RuntimeError(
                f"MLM evaluation count mismatch: got {n_eval}, expected {n_eval_target}"
            )

        total_bits_sum = float(packed[1].item())
        total_bits_sq = float(packed[2].item())
        total_h_sq = float(packed[5].item())
        mean_seq_bits = total_bits_sum / float(n_eval)
        if n_eval > 1:
            var_seq_bits = max(
                0.0,
                (total_bits_sq - float(n_eval) * (mean_seq_bits ** 2)) / float(n_eval - 1),
            )
        else:
            var_seq_bits = 0.0
        std_seq_bits = math.sqrt(var_seq_bits)

        quadrature_16pt = [
            float(packed[6 + s].item()) / float(n_eval) for s in range(MLM_NUM_STRATA)
        ]

        # Exact two-stage combined SE (Cochran Theorem 11.2):
        # Var(mean) = (1 - n_eval / N) * (s_seq^2 / n_eval) + within_seq_var / N
        # where within-sequence stratum sampling variance is estimated from the
        # two-way (sequence x stratum) interaction sum of squares so fpc=0 never
        # zeroes out within-sequence 16-stratum mask-sampling SE when n_eval == N.
        mean_h_bar = mean_seq_bits / float(seq_len)
        sum_seq_h_bar_sq = total_bits_sq / (float(seq_len) ** 2)
        sum_stratum_mean_sq = sum(val * val for val in quadrature_16pt)
        if n_eval > 1 and MLM_NUM_STRATA > 1:
            ss_interaction = (
                total_h_sq
                - float(MLM_NUM_STRATA) * sum_seq_h_bar_sq
                - float(n_eval) * sum_stratum_mean_sq
                + float(n_eval * MLM_NUM_STRATA) * (mean_h_bar ** 2)
            )
            sigma2_stratum = max(0.0, ss_interaction) / float(
                (n_eval - 1) * (MLM_NUM_STRATA - 1)
            )
        elif MLM_NUM_STRATA > 1:
            ss_within_single = total_h_sq - float(MLM_NUM_STRATA) * (mean_h_bar ** 2)
            sigma2_stratum = max(0.0, ss_within_single) / float(MLM_NUM_STRATA - 1)
        else:
            sigma2_stratum = 0.0

        within_var_seq_bits = (float(seq_len) ** 2) * (sigma2_stratum / float(MLM_NUM_STRATA))
        fpc_frac = max(0.0, 1.0 - float(n_eval) / float(n_total))
        between_var_mean = (fpc_frac * var_seq_bits / float(n_eval)) if n_eval > 1 else 0.0
        within_var_mean = within_var_seq_bits / float(n_total)
        mc_se_seq_bits = math.sqrt(max(0.0, between_var_mean + within_var_mean))

        metrics = summarize_memorization_metrics(
            paradigm=canonicalize_paradigm(paradigm),
            n_sequences_total=n_total,
            n_sequences_evaluated=n_eval,
            seq_len=seq_len,
            mean_seq_xent_bits=mean_seq_bits,
            seq_xent_std_bits=std_seq_bits,
            eval_mc_se_seq_bits=mc_se_seq_bits,
            n_params_non_embedding=n_params_non_embedding,
            n_params_total=n_params_total,
            vocab_size=vocab_size,
        )
        tot_masked = float(packed[4].item())
        metrics["token_accuracy"] = (
            float(packed[3].item()) / tot_masked if tot_masked > 0.0 else 0.0
        )
        metrics["eval_between_seq_se_seq_bits"] = math.sqrt(max(0.0, between_var_mean))
        metrics["eval_within_seq_se_seq_bits"] = math.sqrt(max(0.0, within_var_mean))
        metrics["mlm_quadrature_16pt_bits_per_tok"] = quadrature_16pt
        metrics["eval_wall_time_sec"] = float(time.perf_counter() - t0)
        return metrics
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def evaluate_mlm_fixed_rate_bits(
    model: nn.Module,
    data_gpu: torch.Tensor,
    *,
    batch_size: int,
    n_params_non_embedding: int,
    n_params_total: int | None = None,
    mask_count: int | None = None,
    mask_rate: float = MLM_15_MASK_RATE,
    num_draws: int = DEFAULT_MLM_FIXED_NUM_DRAWS,
    precision: str = "bf16",
    eval_max_sequences_mlm: int = DEFAULT_EVAL_MAX_SEQUENCES_MLM,
    eval_seed: int = 20261141,
    vocab_size: int = VOCAB_AA,
    special_token_id: int = SPECIAL_TOKEN_ID,
    paradigm: str = "mlm_15",
) -> dict[str, Any]:
    """Fixed-rate MLM evaluation (default k=38 for L=256) with exact Slepian-Wolf & pseudo bounds.

    Computes deterministic ``num_draws`` (default 16) independent uniform masks of exact size ``k``
    per sequence over up to ``eval_max_sequences_mlm`` sequences and reports:
      1. ``mean_masked_token_xent_bits`` (h_15) with exact two-stage combined SE
         (within-sequence mask-draw variance + finite-population variance).
      2. Exact Slepian-Wolf valid joint-distribution lower bound:
         ``sw_memorized_bits = N * k * (log2(V) - h_15)`` and ``sw_bits_per_non_emb_param``.
      3. Symmetric slot pseudo-joint capacity:
         ``pseudo_memorized_bits = N * L * (log2(V) - h_15)`` and ``pseudo_bits_per_non_emb_param``.
    """
    t0 = time.perf_counter()
    was_training = model.training
    model.eval()
    try:
        n_total, seq_len = int(data_gpu.shape[0]), int(data_gpu.shape[1])
        n_eval_target = min(n_total, int(eval_max_sequences_mlm))
        k = (
            int(mask_count)
            if mask_count is not None
            else mlm_fixed_mask_count(seq_len=seq_len, mask_rate=mask_rate)
        )
        if not (1 <= k <= seq_len):
            raise ValueError(f"mask_count must be in [1, {seq_len}], got {k}")
        n_draws = max(1, int(num_draws))
        device = next(model.parameters()).device

        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        local_indices = torch.arange(
            rank, n_eval_target, world_size, device=device, dtype=torch.int64
        )
        n_local = int(local_indices.numel())
        safe_batch = _resolve_eval_batch_size(model, batch_size, device)
        eff_batch = max(
            1,
            min(safe_batch, max(1, (n_eval_target + world_size - 1) // world_size)),
        )

        sum_h_bar = torch.zeros((), dtype=torch.float64, device=device)
        sum_sq_h_bar = torch.zeros((), dtype=torch.float64, device=device)
        sum_within_var_h_bar = torch.zeros((), dtype=torch.float64, device=device)
        masked_correct = torch.zeros((), dtype=torch.float64, device=device)
        masked_total = torch.zeros((), dtype=torch.float64, device=device)
        count_done = torch.zeros((), dtype=torch.float64, device=device)

        autocast_ctx = _eval_autocast_context(device, precision)

        for start in range(0, n_local, eff_batch):
            valid_b = min(eff_batch, n_local - start)
            batch_idx = local_indices[start : start + valid_b]
            if valid_b < eff_batch:
                pad = local_indices[:1].expand(eff_batch - valid_b)
                batch_idx = torch.cat((batch_idx, pad), dim=0)

            targets = data_gpu.index_select(0, batch_idx).to(device=device, dtype=torch.long)
            h_draws = torch.empty((valid_b, n_draws), dtype=torch.float64, device=device)

            for d in range(n_draws):
                mask = deterministic_fixed_rate_mask(
                    batch_idx,
                    draw_idx=d,
                    seq_len=seq_len,
                    mask_count=k,
                    eval_seed=eval_seed,
                )
                masked_input = torch.where(mask, int(special_token_id), targets)
                with autocast_ctx:
                    logits = model(masked_input)

                tok_nll_nats = F.cross_entropy(
                    logits.float().transpose(1, 2),
                    targets,
                    reduction="none",
                )
                valid_mask = mask[:valid_b]
                valid_nll_bits = tok_nll_nats[:valid_b].to(torch.float64) / LN2
                masked_bits = torch.where(valid_mask, valid_nll_bits, 0.0)
                h_draws[:, d] = masked_bits.sum(dim=-1) / float(k)

                preds = logits[:valid_b].argmax(dim=-1)
                masked_correct += (
                    ((preds == targets[:valid_b]) & valid_mask).sum().to(torch.float64)
                )
                masked_total += valid_mask.sum().to(torch.float64)

            h_bar = h_draws.mean(dim=-1)
            sum_h_bar += h_bar.sum()
            sum_sq_h_bar += (h_bar * h_bar).sum()
            if n_draws > 1:
                within_var_i = ((h_draws - h_bar.unsqueeze(1)) ** 2).sum(dim=-1) / float(
                    n_draws * (n_draws - 1)
                )
                sum_within_var_h_bar += within_var_i.sum()
            count_done += float(valid_b)

        stats = torch.stack(
            (
                count_done,
                sum_h_bar,
                sum_sq_h_bar,
                sum_within_var_h_bar,
                masked_correct,
                masked_total,
            )
        )
        if world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        n_eval = int(round(float(stats[0].item())))
        if n_eval != n_eval_target:
            raise RuntimeError(
                f"MLM fixed-rate evaluation count mismatch: got {n_eval}, expected {n_eval_target}"
            )

        h_15 = float(stats[1].item()) / float(n_eval)
        if n_eval > 1:
            var_h_bar = max(
                0.0,
                (float(stats[2].item()) - float(n_eval) * (h_15 ** 2)) / float(n_eval - 1),
            )
        else:
            var_h_bar = 0.0
        mean_within_var_h_bar = float(stats[3].item()) / float(n_eval)

        # Exact Cochran two-stage variance: (1 - n_eval/N) * s^2 / n_eval + mean_within_var / N
        fpc_frac = max(0.0, 1.0 - float(n_eval) / float(n_total))
        between_var_h15 = (fpc_frac * var_h_bar / float(n_eval)) if n_eval > 1 else 0.0
        within_var_h15 = mean_within_var_h_bar / float(n_total)
        se_h15 = math.sqrt(max(0.0, between_var_h15 + within_var_h15))

        log2_v = math.log2(float(vocab_size))
        denom_total_params = (
            int(n_params_total)
            if n_params_total is not None and n_params_total > 0
            else int(n_params_non_embedding)
        )
        h_dataset = dataset_entropy_bits(n_total, seq_len=seq_len, vocab_size=vocab_size)
        h_gibbs = gibbs_memorizable_bits(n_total, seq_len=seq_len, vocab_size=vocab_size)

        # Exact Slepian-Wolf valid joint-distribution lower bound (zero OOD masks)
        sw_memorized_bits = float(n_total) * float(k) * (log2_v - h_15)
        sw_memorized_bits_se = float(n_total) * float(k) * se_h15
        sw_bpp_non_emb = sw_memorized_bits / float(n_params_non_embedding)
        sw_bpp_total = sw_memorized_bits / float(denom_total_params)
        sw_seq_xent_bits = float(seq_len) * log2_v - float(k) * (log2_v - h_15)

        # Symmetric slot pseudo-joint capacity
        pseudo_seq_xent_bits = float(seq_len) * h_15
        pseudo_memorized_bits = float(n_total) * float(seq_len) * (log2_v - h_15)
        pseudo_memorized_bits_se = float(n_total) * float(seq_len) * se_h15
        pseudo_bpp_non_emb = pseudo_memorized_bits / float(n_params_non_embedding)
        pseudo_bpp_total = pseudo_memorized_bits / float(denom_total_params)

        metrics = summarize_memorization_metrics(
            paradigm=canonicalize_paradigm(paradigm),
            n_sequences_total=n_total,
            n_sequences_evaluated=n_eval,
            seq_len=seq_len,
            mean_seq_xent_bits=pseudo_seq_xent_bits,
            seq_xent_std_bits=float(seq_len) * math.sqrt(var_h_bar),
            eval_mc_se_seq_bits=float(seq_len) * se_h15,
            n_params_non_embedding=n_params_non_embedding,
            n_params_total=n_params_total,
            vocab_size=vocab_size,
        )
        tot_masked = float(stats[5].item())
        metrics.update(
            {
                "mean_masked_token_xent_bits": float(h_15),
                "mean_masked_token_xent_se_bits": float(se_h15),
                "eval_between_seq_se_tok_bits": math.sqrt(max(0.0, between_var_h15)),
                "eval_within_seq_se_tok_bits": math.sqrt(max(0.0, within_var_h15)),
                "eval_between_seq_se_seq_bits": float(seq_len) * math.sqrt(max(0.0, between_var_h15)),
                "eval_within_seq_se_seq_bits": float(seq_len) * math.sqrt(max(0.0, within_var_h15)),
                "mlm_fixed_mask_count": int(k),
                "mlm_fixed_mask_rate": float(k) / float(seq_len),
                "mlm_fixed_num_draws": int(n_draws),
                "sw_seq_xent_bits": float(sw_seq_xent_bits),
                "sw_memorized_bits": float(sw_memorized_bits),
                "sw_memorized_bits_se": float(sw_memorized_bits_se),
                "sw_bits_per_non_emb_param": float(sw_bpp_non_emb),
                "sw_bits_per_total_param": float(sw_bpp_total),
                "sw_fraction_of_H_data": (
                    float(sw_memorized_bits) / float(h_dataset) if h_dataset > 0.0 else 0.0
                ),
                "sw_fraction_of_H_gibbs": (
                    float(sw_memorized_bits) / float(h_gibbs) if h_gibbs > 0.0 else 0.0
                ),
                "pseudo_seq_xent_bits": float(pseudo_seq_xent_bits),
                "pseudo_memorized_bits": float(pseudo_memorized_bits),
                "pseudo_memorized_bits_se": float(pseudo_memorized_bits_se),
                "pseudo_bits_per_non_emb_param": float(pseudo_bpp_non_emb),
                "pseudo_bits_per_total_param": float(pseudo_bpp_total),
                "pseudo_fraction_of_H_data": (
                    float(pseudo_memorized_bits) / float(h_dataset) if h_dataset > 0.0 else 0.0
                ),
                "pseudo_fraction_of_H_gibbs": (
                    float(pseudo_memorized_bits) / float(h_gibbs) if h_gibbs > 0.0 else 0.0
                ),
                "token_accuracy": (
                    float(stats[4].item()) / tot_masked if tot_masked > 0.0 else 0.0
                ),
                "eval_wall_time_sec": float(time.perf_counter() - t0),
            }
        )
        return metrics
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def evaluate_mlm_loo_diagnostic(
    model: nn.Module,
    data_gpu: torch.Tensor,
    *,
    batch_size: int,
    n_params_non_embedding: int,
    precision: str = "bf16",
    max_sequences_loo: int = DEFAULT_LOO_MAX_SEQUENCES_MLM,
    probes_per_sequence: int = DEFAULT_LOO_PROBES_PER_SEQ,
    loo_seed: int = 20261130,
    vocab_size: int = VOCAB_AA,
    special_token_id: int = SPECIAL_TOKEN_ID,
) -> dict[str, Any]:
    """Leave-One-Out (n_mask = 1) pseudo-entropy diagnostic on min(N, 2048) sequences."""
    t0 = time.perf_counter()
    was_training = model.training
    model.eval()
    try:
        n_total, seq_len = int(data_gpu.shape[0]), int(data_gpu.shape[1])
        n_loo = min(n_total, int(max_sequences_loo))
        n_probes = max(1, min(int(probes_per_sequence), seq_len))
        device = next(model.parameters()).device

        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        local_indices = torch.arange(rank, n_loo, world_size, device=device, dtype=torch.int64)
        n_local = int(local_indices.numel())
        safe_batch = _resolve_eval_batch_size(model, batch_size, device)
        eff_batch = max(1, min(safe_batch, max(1, (n_loo + world_size - 1) // world_size)))

        sum_tok_bits = torch.zeros((), dtype=torch.float64, device=device)
        sum_correct = torch.zeros((), dtype=torch.float64, device=device)
        sum_probes = torch.zeros((), dtype=torch.float64, device=device)
        count_done = torch.zeros((), dtype=torch.float64, device=device)

        autocast_ctx = _eval_autocast_context(device, precision)
        row_arange = torch.arange(eff_batch, device=device, dtype=torch.long)

        for start in range(0, n_local, eff_batch):
            valid_b = min(eff_batch, n_local - start)
            batch_idx = local_indices[start : start + valid_b]
            if valid_b < eff_batch:
                pad = local_indices[:1].expand(eff_batch - valid_b)
                batch_idx = torch.cat((batch_idx, pad), dim=0)

            targets = data_gpu.index_select(0, batch_idx).to(device=device, dtype=torch.long)
            priorities = _splitmix64_priorities(
                batch_idx,
                stratum=0,
                seq_len=seq_len,
                seed=loo_seed,
            )
            order = torch.argsort(priorities, dim=-1, stable=True)

            for p_idx in range(n_probes):
                pos_col = order[:, p_idx]
                masked_input = targets.clone()
                masked_input[row_arange, pos_col] = int(special_token_id)

                with autocast_ctx:
                    logits = model(masked_input)

                probe_logits = logits[row_arange[:valid_b], pos_col[:valid_b]].float()
                probe_targets = targets[row_arange[:valid_b], pos_col[:valid_b]]
                nll_bits = (
                    F.cross_entropy(probe_logits, probe_targets, reduction="none").to(
                        torch.float64
                    )
                    / LN2
                )
                sum_tok_bits += nll_bits.sum()
                sum_correct += (
                    (probe_logits.argmax(dim=-1) == probe_targets).sum().to(torch.float64)
                )
                sum_probes += float(valid_b)

            count_done += float(valid_b)

        stats = torch.stack((count_done, sum_tok_bits, sum_correct, sum_probes))
        if world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        n_eval = int(round(float(stats[0].item())))
        tot_probes = float(stats[3].item())
        loo_tok_bits = float(stats[1].item()) / tot_probes if tot_probes > 0.0 else 0.0
        loo_seq_bits = float(seq_len) * loo_tok_bits

        seq_null_bits = float(seq_len) * math.log2(float(vocab_size))
        loo_pseudo_mem_bits = float(n_total) * (seq_null_bits - loo_seq_bits)
        h_gibbs = gibbs_memorizable_bits(n_total, seq_len=seq_len, vocab_size=vocab_size)

        return {
            "mlm_loo_n_sequences": int(n_eval),
            "mlm_loo_probes_per_sequence": int(n_probes),
            "mlm_loo_pseudo_tok_xent_bits": float(loo_tok_bits),
            "mlm_loo_pseudo_seq_xent_bits": float(loo_seq_bits),
            "mlm_loo_pseudo_memorized_bits": float(loo_pseudo_mem_bits),
            "mlm_loo_pseudo_memorized_bits_per_param": (
                float(loo_pseudo_mem_bits) / float(n_params_non_embedding)
            ),
            "mlm_loo_pseudo_fraction_of_H_gibbs": (
                float(loo_pseudo_mem_bits) / float(h_gibbs) if h_gibbs > 0.0 else 0.0
            ),
            "mlm_loo_token_accuracy": (
                float(stats[2].item()) / tot_probes if tot_probes > 0.0 else 0.0
            ),
            "mlm_loo_wall_time_sec": float(time.perf_counter() - t0),
        }
    finally:
        if was_training:
            model.train()


def evaluate_run_checkpoint(
    model: nn.Module,
    data_gpu: torch.Tensor,
    *,
    paradigm: str,
    batch_size: int,
    n_params_non_embedding: int,
    n_params_total: int | None = None,
    precision: str = "bf16",
    eval_max_sequences_mlm: int = DEFAULT_EVAL_MAX_SEQUENCES_MLM,
    eval_seed: int = 20261129,
    mask_rate: float = MLM_15_MASK_RATE,
    mask_count: int | None = None,
    mlm_fixed_num_draws: int = DEFAULT_MLM_FIXED_NUM_DRAWS,
    vocab_size: int = VOCAB_AA,
    special_token_id: int = SPECIAL_TOKEN_ID,
) -> dict[str, Any]:
    """Dispatch primary evaluation to AR, fixed-rate MLM (15/30/50%), or 16-pt MDM/MDM-Cosine evaluator."""
    canon_paradigm = canonicalize_paradigm(paradigm)
    if canon_paradigm == "ar":
        return evaluate_ar_joint_bits(
            model,
            data_gpu,
            batch_size=batch_size,
            n_params_non_embedding=n_params_non_embedding,
            n_params_total=n_params_total,
            precision=precision,
            vocab_size=vocab_size,
            special_token_id=special_token_id,
        )
    if is_fixed_mask_paradigm(canon_paradigm):
        return evaluate_mlm_fixed_rate_bits(
            model,
            data_gpu,
            batch_size=batch_size,
            n_params_non_embedding=n_params_non_embedding,
            n_params_total=n_params_total,
            mask_count=mask_count,
            mask_rate=mask_rate,
            num_draws=mlm_fixed_num_draws,
            precision=precision,
            eval_max_sequences_mlm=eval_max_sequences_mlm,
            eval_seed=eval_seed,
            vocab_size=vocab_size,
            special_token_id=special_token_id,
            paradigm=canon_paradigm,
        )
    if canon_paradigm in ("oa_uniform", "mdm_cosine"):
        return evaluate_mlm_joint_bits_16pt(
            model,
            data_gpu,
            batch_size=batch_size,
            n_params_non_embedding=n_params_non_embedding,
            n_params_total=n_params_total,
            precision=precision,
            eval_max_sequences_mlm=eval_max_sequences_mlm,
            eval_seed=eval_seed,
            vocab_size=vocab_size,
            special_token_id=special_token_id,
            paradigm=canon_paradigm,
        )
    raise ValueError(f"Unsupported paradigm {paradigm!r}")
