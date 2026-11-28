"""Three-tier Eq. (4) bit readout and fast GPU-batched downstream evaluators for Stage-2.

Implements:
  1. Exact Eq. (4) three-tier bit conservation across ``train``, ``unseen_homolog``, and
     ``unseen_fold`` splits under BOTH:
       (i) Native objective readout (``ar`` exact causal, ``oa_uniform`` 16-stratum per-seq
           OA-ARDM NELBO, ``mlm_15`` 16-draw 15%-mask CE), and
       (ii) Matched single-site conditional LOO (16 deterministic single-masked probes per
            sequence for bidirectional models; exact causal conditional for AR).
  2. Fast GPU-batched zero-shot downstream benchmarks:
       - MegaScale ddG (WT-averaged Spearman overall and per stratum: ``de_novo_design``,
         ``unseen_fold_natural``, ``seen_pool_natural``; ``AR`` exact Delta-log-p,
         ``MLM/OA`` ``masked_marginal`` + ``wt_marginal``).
       - ProteinGym substitutions (assay-mean & cluster-mean Spearman overall and by
         ``unseen_fold`` vs ``seen_fold``, supporting 36-assay milestone subset and full L<=384).
       - Unseen-fold 3D contact prediction via zero-shot Categorical Jacobian + APC
         (``P@L``, ``P@L/2``, ``P@L/5``, ``AUROC`` for ``|i-j| >= 6`` and ``|i-j| >= 24``,
         ``d_CB < 8.0 A``).
"""
from __future__ import annotations

import contextlib
import math
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr

from .objectives import PackedSequenceStore, encode_aa_sequence, prepare_ar_batch
from .spec import (
    AA_TO_ID,
    BOS_TOKEN_ID,
    LN2,
    LOG2_20,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    VOCAB_AA,
    canonicalize_objective,
    default_mask_rate_for_objective,
    is_fixed_mask_objective,
    objective_display_name,
)

NUM_EVAL_STRATA: int = 16
DEFAULT_LOO_PROBES: int = 16
CONTACT_DISTANCE_ANGSTROM: float = 8.0
_MUT_TOK_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


@torch.no_grad()
def compute_temperature_calibrated_bits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    weights: torch.Tensor | None = None,
    log_tau_bounds: tuple[float, float] = (-4.0, 6.0),
    n_iter: int = 32,
) -> dict[str, float]:
    """Compute raw vs scalar-temperature-calibrated cross-entropy min_{T>0} H(theta, T) in bits/tok.

    Implements Eq. (3b):
        H_unseen_fold(theta) = min_{T > 0} H_unseen_fold(theta, T) + Delta_cal(theta).
    Scaling logits z -> alpha * z inflates raw_bits_per_tok while leaving calibrated_bits_per_tok invariant.
    """
    if valid_mask is not None:
        flat_mask = valid_mask.reshape(-1).bool()
        flat_logits = logits.reshape(-1, logits.shape[-1])[flat_mask].float()
        flat_targets = targets.reshape(-1)[flat_mask].long()
        flat_weights = weights.reshape(-1)[flat_mask].float() if weights is not None else None
    else:
        flat_logits = logits.reshape(-1, logits.shape[-1]).float()
        flat_targets = targets.reshape(-1).long()
        flat_weights = weights.reshape(-1).float() if weights is not None else None

    if flat_targets.numel() == 0:
        return {
            "raw_bits_per_tok": float("nan"),
            "calibrated_bits_per_tok": float("nan"),
            "optimal_temperature": float("nan"),
            "calibration_excess_bits": float("nan"),
            "Delta_cal": float("nan"),
        }

    w_sum = float(flat_weights.sum().item()) if flat_weights is not None else float(flat_targets.numel())
    if w_sum <= 0.0:
        flat_weights = None
        w_sum = float(flat_targets.numel())

    def _bits_at_log_tau(log_tau: float) -> float:
        inv_tau = math.exp(-float(log_tau))
        if flat_weights is None:
            nll = F.cross_entropy(flat_logits * inv_tau, flat_targets, reduction="mean")
            return float(nll.item() / LN2)
        nll_vec = F.cross_entropy(flat_logits * inv_tau, flat_targets, reduction="none")
        return float(((nll_vec * flat_weights).sum().item() / w_sum) / LN2)

    raw_bits = _bits_at_log_tau(0.0)

    a, b = float(log_tau_bounds[0]), float(log_tau_bounds[1])
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc = _bits_at_log_tau(c)
    fd = _bits_at_log_tau(d)
    for _ in range(int(n_iter)):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = _bits_at_log_tau(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = _bits_at_log_tau(d)

    best_log_tau = 0.5 * (a + b)
    cal_bits = min(raw_bits, _bits_at_log_tau(best_log_tau))
    opt_tau = 1.0 if cal_bits == raw_bits else math.exp(best_log_tau)
    delta_cal = float(max(0.0, raw_bits - cal_bits))
    return {
        "raw_bits_per_tok": float(raw_bits),
        "calibrated_bits_per_tok": float(cal_bits),
        "optimal_temperature": float(opt_tau),
        "calibration_excess_bits": delta_cal,
        "Delta_cal": delta_cal,
    }


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
    draw_idx: int,
    max_len: int,
    seed: int,
) -> torch.Tensor:
    """Deterministic batch-invariant int64 priorities of shape (B, max_len)."""
    device = global_indices.device
    pos = torch.arange(max_len, device=device, dtype=torch.int64).unsqueeze(0)
    idx = global_indices.to(dtype=torch.int64).unsqueeze(1)
    base_seed = _u64_to_i64(int(seed))
    s_term = _u64_to_i64(int(draw_idx) * 0x517CC1B727220A95)

    z = base_seed + idx * _SM_C0 + s_term + pos * _SM_C1 + _SM_C2
    z = (z ^ _ushift_right_i64(z, 30)) * _SM_M1
    z = (z ^ _ushift_right_i64(z, 27)) * _SM_M2
    z = z ^ _ushift_right_i64(z, 31)
    return z


def _deterministic_valid_mask(
    global_indices: torch.Tensor,
    lengths: torch.Tensor,
    n_mask: torch.Tensor,
    max_len: int,
    *,
    draw_idx: int,
    eval_seed: int,
) -> torch.Tensor:
    """Deterministic mask selecting exact n_mask[i] positions strictly within [0, lengths[i])."""
    device = global_indices.device
    pos = torch.arange(max_len, device=device, dtype=torch.int64).unsqueeze(0)
    valid = pos < lengths.to(dtype=torch.int64).unsqueeze(1)

    prio = _splitmix64_priorities(global_indices, draw_idx=draw_idx, max_len=max_len, seed=eval_seed)
    # Map int64 priorities strictly below INT64_MAX for valid positions; set <PAD> to INT64_MAX
    # Use float64 normalized priorities so invalid positions strictly sort last (+2.0 vs [0, 1))
    norm_p = ((prio.to(dtype=torch.float64) + 9223372036854775808.0) / 18446744073709551616.0)
    masked_p = torch.where(valid, norm_p, torch.full_like(norm_p, 2.0))
    order = torch.argsort(masked_p, dim=-1, stable=True)

    mask = torch.zeros((global_indices.shape[0], max_len), dtype=torch.bool, device=device)
    mask.scatter_(1, order, pos < n_mask.to(dtype=torch.int64).unsqueeze(1))
    return mask & valid


def _eval_autocast_ctx(device: torch.device, precision: str) -> Any:
    if precision == "bf16" and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(mask.sum()) < 3:
        return float("nan")
    if float(np.std(xa[mask])) == 0.0 or float(np.std(ya[mask])) == 0.0:
        return 0.0
    val = float(spearmanr(xa[mask], ya[mask]).statistic)
    return val if math.isfinite(val) else 0.0


def _fast_binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    lbl = np.asarray(labels, dtype=np.bool_)
    n_pos = int(lbl.sum())
    n_neg = int(lbl.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=np.float64), method="average")
    return float((ranks[lbl].sum() - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg))


# --------------------------------------------------------------------------- #
# 1. Per-Split Entropy Evaluation & Eq. (4) Three-Tier Conservation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate_split_native(
    model: nn.Module,
    store: PackedSequenceStore,
    *,
    objective: str,
    batch_size: int = 128,
    precision: str = "bf16",
    n_draws: int = NUM_EVAL_STRATA,
    eval_seed: int = 20261129,
    mask_rate: float | None = None,
    eval_temperature_calibration: bool = False,
    cal_max_tokens: int = 262_144,
) -> dict[str, Any]:
    """Evaluate native per-residue cross-entropy (bits/res) on a variable-length split.

    - ``ar``: exact causal NLL over every real AA position.
    - ``oa_uniform`` (MDM) / ``mdm_cosine``: deterministic 16-stratum per-sequence OA-ARDM NELBO.
    - ``mlm_15`` / ``mlm_30`` / ``mlm_50``: deterministic 16-draw fixed-rate CE.
    When ``eval_temperature_calibration=True``, also computes exact Eq. (3b) scalar
    temperature calibration ``H_cal = min_{T>0} H(theta, T)`` and ``Delta_cal``.
    """
    canon_obj = canonicalize_objective(objective)
    if store.n_sequences == 0:
        raise ValueError("Cannot evaluate empty PackedSequenceStore")

    t0 = time.perf_counter()
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    autocast_ctx = _eval_autocast_ctx(device, precision)

    # Sort by length for minimal padding waste while preserving exact global index for determinism
    order_idx = np.argsort(store.lengths, kind="mergesort")
    n_seq = store.n_sequences

    total_weighted_bits = 0.0
    total_residues = 0
    seq_mean_bits_sum = 0.0
    total_correct = 0.0
    total_eval_tokens = 0.0
    stratum_sums = np.zeros(int(n_draws), dtype=np.float64)

    cal_logits_chunks: list[torch.Tensor] = []
    cal_targets_chunks: list[torch.Tensor] = []
    cal_weights_chunks: list[torch.Tensor] = []
    cal_tokens_collected = 0
    eff_mask_rate = float(mask_rate) if mask_rate is not None else default_mask_rate_for_objective(canon_obj)

    def _maybe_collect_cal(
        batch_logits: torch.Tensor,
        batch_targets: torch.Tensor,
        batch_mask: torch.Tensor,
        per_pos_weights: torch.Tensor,
    ) -> None:
        nonlocal cal_tokens_collected
        if not eval_temperature_calibration or cal_tokens_collected >= int(cal_max_tokens):
            return
        sel_logits = batch_logits[batch_mask].detach().to(device="cpu", dtype=torch.float16)
        sel_targets = batch_targets[batch_mask].detach().to(device="cpu", dtype=torch.uint8)
        sel_weights = per_pos_weights[batch_mask].detach().to(device="cpu", dtype=torch.float32)
        n_new = int(sel_targets.numel())
        if n_new == 0:
            return
        rem = int(cal_max_tokens) - cal_tokens_collected
        if n_new > rem:
            sel_logits = sel_logits[:rem]
            sel_targets = sel_targets[:rem]
            sel_weights = sel_weights[:rem]
            n_new = rem
        cal_logits_chunks.append(sel_logits)
        cal_targets_chunks.append(sel_targets)
        cal_weights_chunks.append(sel_weights)
        cal_tokens_collected += n_new

    try:
        for start in range(0, n_seq, max(1, int(batch_size))):
            batch_global_np = order_idx[start : start + int(batch_size)]
            tokens, lengths = store.collate_indices(batch_global_np, device=device, pad_to_multiple=8)
            bsz, max_len = tokens.shape
            global_idx_t = torch.from_numpy(batch_global_np).to(device=device, dtype=torch.int64)
            pos = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
            valid = pos < lengths.unsqueeze(1)

            if canon_obj == "ar":
                prep = prepare_ar_batch(tokens, lengths)
                with autocast_ctx:
                    logits = model(prep.input_ids, prep.attention_mask, right_padded=True)
                safe_t = torch.where(prep.loss_mask, prep.targets, torch.zeros_like(prep.targets))
                tok_bits = (
                    F.cross_entropy(logits.float().transpose(1, 2), safe_t, reduction="none").to(
                        torch.float64
                    )
                    / LN2
                )
                tok_bits = torch.where(prep.loss_mask, tok_bits, torch.zeros_like(tok_bits))
                seq_sum_bits = tok_bits.sum(dim=-1)
                seq_h = seq_sum_bits / lengths.to(dtype=torch.float64)

                preds = logits.argmax(dim=-1)
                corr = ((preds == safe_t) & prep.loss_mask).sum().to(torch.float64).item()

                total_weighted_bits += float(seq_sum_bits.sum().item())
                total_residues += int(lengths.sum().item())
                seq_mean_bits_sum += float(seq_h.sum().item())
                total_correct += corr
                total_eval_tokens += float(lengths.sum().item())

                if eval_temperature_calibration:
                    _maybe_collect_cal(
                        logits,
                        safe_t,
                        prep.loss_mask,
                        torch.ones_like(safe_t, dtype=torch.float32),
                    )

            elif canon_obj in ("oa_uniform", "mdm_cosine"):
                h_draws = torch.empty((bsz, int(n_draws)), dtype=torch.float64, device=device)
                for s in range(int(n_draws)):
                    off = torch.remainder(global_idx_t + 7 * s, int(n_draws))
                    u = (float(s) + (off.to(dtype=torch.float64) + 0.5) / float(n_draws)) / float(
                        n_draws
                    )
                    m_mask = torch.clamp(
                        torch.floor(u * lengths.to(dtype=torch.float64)).to(dtype=torch.int64) + 1,
                        min=1,
                    )
                    m_mask = torch.minimum(m_mask, lengths.to(dtype=torch.int64))
                    mask = _deterministic_valid_mask(
                        global_idx_t,
                        lengths,
                        m_mask,
                        max_len,
                        draw_idx=s,
                        eval_seed=eval_seed,
                    )
                    safe_targets = torch.where(valid, tokens, torch.zeros_like(tokens))
                    inp = torch.where(mask, MASK_TOKEN_ID, tokens)
                    with autocast_ctx:
                        logits = model(inp, valid)
                    tok_bits = (
                        F.cross_entropy(
                            logits.float().transpose(1, 2), safe_targets, reduction="none"
                        ).to(torch.float64)
                        / LN2
                    )
                    masked_bits = torch.where(mask, tok_bits, torch.zeros_like(tok_bits))
                    h_s = masked_bits.sum(dim=-1) / m_mask.to(dtype=torch.float64)
                    h_draws[:, s] = h_s
                    stratum_sums[s] += float(h_s.sum().item())

                    preds = logits.argmax(dim=-1)
                    total_correct += float(((preds == safe_targets) & mask).sum().item())
                    total_eval_tokens += float(mask.sum().item())

                    if eval_temperature_calibration:
                        # Exact residue-weighted NELBO weight per masked site: L_i / (m_i * n_draws)
                        w_row = (lengths.to(dtype=torch.float32) / (m_mask.to(dtype=torch.float32) * float(n_draws)))
                        _maybe_collect_cal(
                            logits,
                            safe_targets,
                            mask,
                            w_row.unsqueeze(1).expand_as(safe_targets),
                        )

                seq_h = h_draws.mean(dim=-1)  # exact unbiased OA-ARDM per-residue NELBO for each seq
                total_weighted_bits += float((seq_h * lengths.to(dtype=torch.float64)).sum().item())
                total_residues += int(lengths.sum().item())
                seq_mean_bits_sum += float(seq_h.sum().item())

            elif is_fixed_mask_objective(canon_obj):
                k_mask = torch.clamp(
                    torch.round(lengths.to(dtype=torch.float32) * eff_mask_rate).to(dtype=torch.int64),
                    min=1,
                )
                k_mask = torch.minimum(k_mask, lengths.to(dtype=torch.int64))
                h_draws = torch.empty((bsz, int(n_draws)), dtype=torch.float64, device=device)
                for d in range(int(n_draws)):
                    mask = _deterministic_valid_mask(
                        global_idx_t,
                        lengths,
                        k_mask,
                        max_len,
                        draw_idx=d,
                        eval_seed=eval_seed,
                    )
                    safe_targets = torch.where(valid, tokens, torch.zeros_like(tokens))
                    inp = torch.where(mask, MASK_TOKEN_ID, tokens)
                    with autocast_ctx:
                        logits = model(inp, valid)
                    tok_bits = (
                        F.cross_entropy(
                            logits.float().transpose(1, 2), safe_targets, reduction="none"
                        ).to(torch.float64)
                        / LN2
                    )
                    masked_bits = torch.where(mask, tok_bits, torch.zeros_like(tok_bits))
                    h_d = masked_bits.sum(dim=-1) / k_mask.to(dtype=torch.float64)
                    h_draws[:, d] = h_d
                    stratum_sums[d] += float(h_d.sum().item())

                    preds = logits.argmax(dim=-1)
                    total_correct += float(((preds == safe_targets) & mask).sum().item())
                    total_eval_tokens += float(mask.sum().item())

                    if eval_temperature_calibration:
                        w_row = (lengths.to(dtype=torch.float32) / (k_mask.to(dtype=torch.float32) * float(n_draws)))
                        _maybe_collect_cal(
                            logits,
                            safe_targets,
                            mask,
                            w_row.unsqueeze(1).expand_as(safe_targets),
                        )

                seq_h = h_draws.mean(dim=-1)
                total_weighted_bits += float((seq_h * lengths.to(dtype=torch.float64)).sum().item())
                total_residues += int(lengths.sum().item())
                seq_mean_bits_sum += float(seq_h.sum().item())
    finally:
        if was_training:
            model.train()

    out: dict[str, Any] = {
        "objective": canon_obj,
        "paradigm_display": objective_display_name(canon_obj),
        "n_sequences": int(n_seq),
        "total_residues": int(total_residues),
        "bits_per_res": float(total_weighted_bits / float(max(1, total_residues))),
        "seq_mean_bits_per_res": float(seq_mean_bits_sum / float(n_seq)),
        "token_accuracy": float(total_correct / float(max(1.0, total_eval_tokens))),
        "eval_wall_sec": float(time.perf_counter() - t0),
    }
    if canon_obj != "ar":
        out["draw_bits_per_res"] = [float(x / float(n_seq)) for x in stratum_sums]
    if eval_temperature_calibration and cal_logits_chunks:
        all_logits = torch.cat(cal_logits_chunks, dim=0)
        all_targets = torch.cat(cal_targets_chunks, dim=0)
        all_weights = torch.cat(cal_weights_chunks, dim=0)
        cal_res = compute_temperature_calibrated_bits(
            all_logits,
            all_targets,
            weights=all_weights,
        )
        cal_res["n_tokens_calibrated"] = int(all_targets.numel())
        out["temperature_calibration"] = cal_res
    return out


@torch.no_grad()
def evaluate_split_loo(
    model: nn.Module,
    store: PackedSequenceStore,
    *,
    objective: str,
    batch_size: int = 128,
    precision: str = "bf16",
    n_probes: int = DEFAULT_LOO_PROBES,
    eval_seed: int = 20261131,
) -> dict[str, Any]:
    """Matched single-site conditional LOO bits/res (16 random single-masked sites for MLM/OA, causal for AR)."""
    if objective == "ar":
        res = evaluate_split_native(
            model,
            store,
            objective="ar",
            batch_size=batch_size,
            precision=precision,
            eval_seed=eval_seed,
        )
        return {
            "mode": "ar_causal",
            "n_sequences": res["n_sequences"],
            "n_probes_per_seq": None,
            "bits_per_res": res["bits_per_res"],
            "seq_mean_bits_per_res": res["seq_mean_bits_per_res"],
            "token_accuracy": res["token_accuracy"],
            "eval_wall_sec": res["eval_wall_sec"],
        }

    t0 = time.perf_counter()
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    autocast_ctx = _eval_autocast_ctx(device, precision)

    order_idx = np.argsort(store.lengths, kind="mergesort")
    n_seq = store.n_sequences

    weighted_bits_sum = 0.0
    total_residues = 0
    seq_mean_sum = 0.0
    correct_sum = 0.0
    probe_total = 0.0

    try:
        for start in range(0, n_seq, max(1, int(batch_size))):
            batch_global_np = order_idx[start : start + int(batch_size)]
            tokens, lengths = store.collate_indices(batch_global_np, device=device, pad_to_multiple=8)
            bsz, max_len = tokens.shape
            global_idx_t = torch.from_numpy(batch_global_np).to(device=device, dtype=torch.int64)
            pos = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
            valid = pos < lengths.unsqueeze(1)

            prio = _splitmix64_priorities(global_idx_t, draw_idx=0, max_len=max_len, seed=eval_seed)
            norm_p = ((prio.to(dtype=torch.float64) + 9223372036854775808.0) / 18446744073709551616.0)
            masked_p = torch.where(valid, norm_p, torch.full_like(norm_p, 2.0))
            probe_order = torch.argsort(masked_p, dim=-1, stable=True)

            eff_probes = min(int(n_probes), int(lengths.min().item()))
            eff_probes = max(1, eff_probes)
            row_ar = torch.arange(bsz, device=device, dtype=torch.long)

            probe_bits = torch.empty((bsz, eff_probes), dtype=torch.float64, device=device)
            for p_i in range(eff_probes):
                pos_col = probe_order[:, p_i]
                inp = tokens.clone()
                inp[row_ar, pos_col] = MASK_TOKEN_ID
                with autocast_ctx:
                    logits = model(inp, valid)
                probe_logits = logits[row_ar, pos_col].float()
                probe_targets = tokens[row_ar, pos_col]
                nll_b = (
                    F.cross_entropy(probe_logits, probe_targets, reduction="none").to(torch.float64)
                    / LN2
                )
                probe_bits[:, p_i] = nll_b
                correct_sum += float((probe_logits.argmax(dim=-1) == probe_targets).sum().item())
                probe_total += float(bsz)

            seq_h = probe_bits.mean(dim=-1)
            weighted_bits_sum += float((seq_h * lengths.to(dtype=torch.float64)).sum().item())
            total_residues += int(lengths.sum().item())
            seq_mean_sum += float(seq_h.sum().item())
    finally:
        if was_training:
            model.train()

    return {
        "mode": "single_site_loo",
        "n_sequences": int(n_seq),
        "n_probes_per_seq": int(n_probes),
        "bits_per_res": float(weighted_bits_sum / float(max(1, total_residues))),
        "seq_mean_bits_per_res": float(seq_mean_sum / float(n_seq)),
        "token_accuracy": float(correct_sum / float(max(1.0, probe_total))),
        "eval_wall_sec": float(time.perf_counter() - t0),
    }


def compute_three_tier_decomposition(
    *,
    h_train: float,
    h_unseen_homolog: float,
    h_unseen_fold: float,
    h0: float = LOG2_20,
) -> dict[str, float]:
    """Compute Eq. (4) three-tier conserved bit decomposition (bits/res)."""
    delta_h_univ = float(h0) - float(h_unseen_fold)
    delta_h_spec = float(h_unseen_fold) - float(h_unseen_homolog)
    delta_h_res = float(h_unseen_homolog) - float(h_train)
    delta_h_total = float(h0) - float(h_train)
    sum_parts = delta_h_univ + delta_h_spec + delta_h_res
    return {
        "H_0": float(h0),
        "H_train": float(h_train),
        "H_unseen_homolog": float(h_unseen_homolog),
        "H_unseen_fold": float(h_unseen_fold),
        "delta_h_univ": float(delta_h_univ),
        "delta_h_spec": float(delta_h_spec),
        "delta_h_res": float(delta_h_res),
        "delta_h_total": float(delta_h_total),
        "conservation_error": float(abs(sum_parts - delta_h_total)),
    }


@torch.no_grad()
def evaluate_three_tier_bits(
    model: nn.Module,
    *,
    train_sub_store: PackedSequenceStore,
    unseen_homolog_store: PackedSequenceStore,
    unseen_fold_store: PackedSequenceStore,
    global_natural_store: PackedSequenceStore | None = None,
    objective: str,
    batch_size: int = 128,
    precision: str = "bf16",
    n_draws: int = NUM_EVAL_STRATA,
    loo_probes: int = DEFAULT_LOO_PROBES,
    eval_seed: int = 20261129,
    mask_rate: float | None = None,
    eval_temperature_calibration: bool = True,
) -> dict[str, Any]:
    """Evaluate Eq. (3a)-(3b) three-tier bit conservation and temperature calibration."""
    canon_obj = canonicalize_objective(objective)
    t0 = time.perf_counter()
    splits = {
        "train": train_sub_store,
        "unseen_homolog": unseen_homolog_store,
        "unseen_fold": unseen_fold_store,
    }
    if global_natural_store is not None:
        splits["global_natural"] = global_natural_store

    native_res: dict[str, dict[str, Any]] = {}
    loo_res: dict[str, dict[str, Any]] = {}

    for idx, (name, store) in enumerate(splits.items()):
        native_res[name] = evaluate_split_native(
            model,
            store,
            objective=canon_obj,
            batch_size=batch_size,
            precision=precision,
            n_draws=n_draws,
            eval_seed=eval_seed + idx * 101,
            mask_rate=mask_rate,
            eval_temperature_calibration=(eval_temperature_calibration and name == "unseen_fold"),
        )
        if canon_obj == "ar":
            # Avoid running AR forward pass twice; native causal == conditional LOO for AR
            loo_res[name] = {
                "mode": "ar_causal",
                "n_sequences": native_res[name]["n_sequences"],
                "n_probes_per_seq": None,
                "bits_per_res": native_res[name]["bits_per_res"],
                "seq_mean_bits_per_res": native_res[name]["seq_mean_bits_per_res"],
                "token_accuracy": native_res[name]["token_accuracy"],
                "eval_wall_sec": 0.0,
            }
        else:
            loo_res[name] = evaluate_split_loo(
                model,
                store,
                objective=canon_obj,
                batch_size=batch_size,
                precision=precision,
                n_probes=loo_probes,
                eval_seed=eval_seed + 10_000 + idx * 101,
            )

    native_eq4 = compute_three_tier_decomposition(
        h_train=native_res["train"]["bits_per_res"],
        h_unseen_homolog=native_res["unseen_homolog"]["bits_per_res"],
        h_unseen_fold=native_res["unseen_fold"]["bits_per_res"],
    )
    temp_cal = native_res["unseen_fold"].get("temperature_calibration")
    if isinstance(temp_cal, Mapping):
        native_eq4["H_unseen_fold_cal"] = float(temp_cal["calibrated_bits_per_tok"])
        native_eq4["Delta_cal"] = float(temp_cal["Delta_cal"])
        native_eq4["optimal_temperature_unseen_fold"] = float(temp_cal["optimal_temperature"])

    native_eq4_seq_mean = compute_three_tier_decomposition(
        h_train=native_res["train"]["seq_mean_bits_per_res"],
        h_unseen_homolog=native_res["unseen_homolog"]["seq_mean_bits_per_res"],
        h_unseen_fold=native_res["unseen_fold"]["seq_mean_bits_per_res"],
    )
    loo_eq4 = compute_three_tier_decomposition(
        h_train=loo_res["train"]["bits_per_res"],
        h_unseen_homolog=loo_res["unseen_homolog"]["bits_per_res"],
        h_unseen_fold=loo_res["unseen_fold"]["bits_per_res"],
    )

    return {
        "objective": canon_obj,
        "paradigm_display": objective_display_name(canon_obj),
        "native": native_eq4,
        "native_seq_mean": native_eq4_seq_mean,
        "loo": loo_eq4,
        "temperature_calibration": temp_cal,
        "splits_native": native_res,
        "splits_loo": loo_res,
        "eval_wall_sec": float(time.perf_counter() - t0),
    }


# --------------------------------------------------------------------------- #
# 2. Fast GPU-Batched Mutation Scoring (AR exact Delta-log-p, MLM/OA masked & WT)
# --------------------------------------------------------------------------- #


def _parse_mut_string(mut_str: str) -> list[tuple[int, int, int]]:
    """Parse 'A12V' or 'A12V:K15R' into list of (pos_0based, wt_id, mut_id)."""
    out: list[tuple[int, int, int]] = []
    for tok in mut_str.split(":"):
        m = _MUT_TOK_RE.match(tok.strip().upper())
        if m is None:
            raise ValueError(f"Invalid mutation token {tok!r} in {mut_str!r}")
        w_aa, pos1, m_aa = m.groups()
        out.append((int(pos1) - 1, AA_TO_ID[w_aa], AA_TO_ID[m_aa]))
    return out


@torch.no_grad()
def _score_ar_sequences_nll_sum(
    model: nn.Module,
    seqs_encoded: np.ndarray,
    *,
    batch_size: int,
    precision: str,
) -> np.ndarray:
    """Compute exact AR total NLL (nats, sum over all L positions) for same-length (N, L) uint8 sequences."""
    device = next(model.parameters()).device
    autocast_ctx = _eval_autocast_ctx(device, precision)
    n_seq, length = seqs_encoded.shape
    out = np.empty(n_seq, dtype=np.float64)

    for start in range(0, n_seq, max(1, int(batch_size))):
        chunk = seqs_encoded[start : start + int(batch_size)]
        bsz = chunk.shape[0]
        targets = torch.from_numpy(chunk).to(device=device, dtype=torch.long)
        bos = torch.full((bsz, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
        inp = torch.cat((bos, targets[:, :-1]), dim=1)
        with autocast_ctx:
            logits = model(inp, attention_mask=None, right_padded=True)
        tok_nll = F.cross_entropy(
            logits.float().transpose(1, 2), targets, reduction="none"
        ).to(torch.float64)
        out[start : start + bsz] = tok_nll.sum(dim=-1).cpu().numpy()
    return out


@torch.no_grad()
def _score_mlm_site_logprobs(
    model: nn.Module,
    wt_encoded: np.ndarray,
    positions: Sequence[int],
    *,
    batch_size: int,
    precision: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (wt_marginal_lp[L, 20], masked_marginal_lp_by_pos[len(positions), 20]) in nats."""
    device = next(model.parameters()).device
    autocast_ctx = _eval_autocast_ctx(device, precision)
    L = int(wt_encoded.size)
    wt_t = torch.from_numpy(wt_encoded).to(device=device, dtype=torch.long)
    valid_1 = torch.ones((1, L), dtype=torch.bool, device=device)

    with autocast_ctx:
        wt_logits = model(wt_t.unsqueeze(0), valid_1)[0].float()
    wt_lp = F.log_softmax(wt_logits, dim=-1).cpu().numpy()

    pos_arr = np.asarray(positions, dtype=np.int64)
    n_pos = int(pos_arr.size)
    masked_lp = np.empty((n_pos, VOCAB_AA), dtype=np.float32)

    for start in range(0, n_pos, max(1, int(batch_size))):
        p_chunk = pos_arr[start : start + int(batch_size)]
        cur = int(p_chunk.size)
        inp = wt_t.unsqueeze(0).expand(cur, -1).clone()
        row_ar = torch.arange(cur, device=device, dtype=torch.long)
        pos_t = torch.from_numpy(p_chunk).to(device=device, dtype=torch.long)
        inp[row_ar, pos_t] = MASK_TOKEN_ID
        valid_b = torch.ones((cur, L), dtype=torch.bool, device=device)
        with autocast_ctx:
            logits = model(inp, valid_b)
        site_logits = logits[row_ar, pos_t].float()
        masked_lp[start : start + cur] = F.log_softmax(site_logits, dim=-1).cpu().numpy()

    return wt_lp, masked_lp


@torch.no_grad()
def evaluate_megascale_ddg(
    model: nn.Module,
    megascale_payload: Mapping[str, Any],
    *,
    objective: str,
    batch_size: int = 256,
    precision: str = "bf16",
    max_variants_per_domain: int | None = None,
) -> dict[str, Any]:
    """Fast GPU-batched MegaScale ddG evaluation overall and per stratum."""
    t0 = time.perf_counter()
    was_training = model.training
    model.eval()

    domains = megascale_payload.get("domains", [])
    primary_rhos_by_stratum: dict[str, list[float]] = defaultdict(list)
    wt_marg_rhos_by_stratum: dict[str, list[float]] = defaultdict(list)
    all_primary_rhos: list[float] = []
    all_wt_marg_rhos: list[float] = []

    try:
        for dom in domains:
            wt_seq = str(dom["wildtype"])
            wt_enc = encode_aa_sequence(wt_seq)
            variants = dom["variants"]
            if max_variants_per_domain is not None and len(variants) > max_variants_per_domain:
                step = max(1, len(variants) // int(max_variants_per_domain))
                variants = variants[::step][: int(max_variants_per_domain)]
            if len(variants) < 3:
                continue

            pos_list = np.asarray([int(v[0]) for v in variants], dtype=np.int64)
            wt_ids = np.asarray([AA_TO_ID[str(v[1])] for v in variants], dtype=np.int64)
            mut_ids = np.asarray([AA_TO_ID[str(v[2])] for v in variants], dtype=np.int64)
            targets = np.asarray([float(v[3]) for v in variants], dtype=np.float64)
            stratum = str(dom.get("stratum", "unknown"))

            if objective == "ar":
                n_var = len(variants)
                all_seqs = np.repeat(wt_enc[None, :], n_var + 1, axis=0)
                all_seqs[1:, :][np.arange(n_var), pos_list] = mut_ids.astype(np.uint8)
                nlls = _score_ar_sequences_nll_sum(
                    model, all_seqs, batch_size=batch_size, precision=precision
                )
                scores = nlls[0] - nlls[1:]  # log p(mut) - log p(wt)
                rho = _safe_spearman(scores, targets)
                if math.isfinite(rho):
                    all_primary_rhos.append(rho)
                    primary_rhos_by_stratum[stratum].append(rho)
            else:
                uniq_pos = np.unique(pos_list)
                pos_to_idx = {int(p): i for i, p in enumerate(uniq_pos)}
                wt_lp, masked_lp = _score_mlm_site_logprobs(
                    model, wt_enc, uniq_pos, batch_size=batch_size, precision=precision
                )
                mapped = np.asarray([pos_to_idx[int(p)] for p in pos_list], dtype=np.int64)
                mm_scores = masked_lp[mapped, mut_ids] - masked_lp[mapped, wt_ids]
                wm_scores = wt_lp[pos_list, mut_ids] - wt_lp[pos_list, wt_ids]

                rho_mm = _safe_spearman(mm_scores, targets)
                rho_wm = _safe_spearman(wm_scores, targets)
                if math.isfinite(rho_mm):
                    all_primary_rhos.append(rho_mm)
                    primary_rhos_by_stratum[stratum].append(rho_mm)
                if math.isfinite(rho_wm):
                    all_wt_marg_rhos.append(rho_wm)
                    wt_marg_rhos_by_stratum[stratum].append(rho_wm)
    finally:
        if was_training:
            model.train()

    strata_summary: dict[str, Any] = {}
    for st in sorted(primary_rhos_by_stratum.keys()):
        vals = primary_rhos_by_stratum[st]
        entry: dict[str, Any] = {
            "n_domains": len(vals),
            "spearman_mean": float(np.mean(vals)) if vals else float("nan"),
        }
        if wt_marg_rhos_by_stratum.get(st):
            entry["wt_marginal_spearman_mean"] = float(np.mean(wt_marg_rhos_by_stratum[st]))
        strata_summary[st] = entry

    out: dict[str, Any] = {
        "scoring_method": "exact_delta_logp" if objective == "ar" else "masked_marginal",
        "n_domains": len(all_primary_rhos),
        "overall_spearman_mean": float(np.mean(all_primary_rhos)) if all_primary_rhos else float("nan"),
        "by_stratum": strata_summary,
        "eval_wall_sec": float(time.perf_counter() - t0),
    }
    if all_wt_marg_rhos:
        out["wt_marginal_overall_spearman_mean"] = float(np.mean(all_wt_marg_rhos))
    return out


@torch.no_grad()
def evaluate_proteingym_substitutions(
    model: nn.Module,
    proteingym_payload: Mapping[str, Any],
    *,
    objective: str,
    milestone_only: bool = True,
    batch_size: int = 128,
    precision: str = "bf16",
) -> dict[str, Any]:
    """Fast GPU-batched ProteinGym substitution evaluation (assay-mean & cluster-mean Spearman)."""
    t0 = time.perf_counter()
    was_training = model.training
    model.eval()

    assays = proteingym_payload.get("assays", [])
    if milestone_only:
        assays = [a for a in assays if a.get("in_milestone_subset", False)]

    assay_rhos: list[float] = []
    wt_marg_assay_rhos: list[float] = []
    cluster_rhos: dict[str, list[float]] = defaultdict(list)
    stratum_assay_rhos: dict[str, list[float]] = defaultdict(list)
    stratum_cluster_rhos: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    try:
        for assay in assays:
            wt_seq = str(assay["wildtype"])
            wt_enc = encode_aa_sequence(wt_seq)
            all_vars = assay["variants"]
            if milestone_only and "milestone_variant_indices" in assay:
                sel_vars = [all_vars[i] for i in assay["milestone_variant_indices"]]
            else:
                sel_vars = all_vars
            if len(sel_vars) < 3:
                continue

            parsed_muts = [_parse_mut_string(str(v[0])) for v in sel_vars]
            targets = np.asarray([float(v[1]) for v in sel_vars], dtype=np.float64)
            cluster = str(assay.get("cluster_id", assay["dms_id"]))
            stratum = str(assay.get("fold_stratum", "unannotated"))

            if objective == "ar":
                n_var = len(parsed_muts)
                seqs = np.repeat(wt_enc[None, :], n_var + 1, axis=0)
                for idx, mlist in enumerate(parsed_muts):
                    for pos0, _w_id, m_id in mlist:
                        seqs[idx + 1, pos0] = m_id
                nlls = _score_ar_sequences_nll_sum(
                    model, seqs, batch_size=batch_size, precision=precision
                )
                scores = nlls[0] - nlls[1:]
                rho = _safe_spearman(scores, targets)
                rho_wm = float("nan")
            else:
                uniq_pos = sorted({pos0 for mlist in parsed_muts for (pos0, _, _) in mlist})
                pos_to_idx = {p: i for i, p in enumerate(uniq_pos)}
                wt_lp, masked_lp = _score_mlm_site_logprobs(
                    model, wt_enc, uniq_pos, batch_size=batch_size, precision=precision
                )
                mm_scores = np.empty(len(parsed_muts), dtype=np.float64)
                wm_scores = np.empty(len(parsed_muts), dtype=np.float64)
                for idx, mlist in enumerate(parsed_muts):
                    s_mm = 0.0
                    s_wm = 0.0
                    for pos0, w_id, m_id in mlist:
                        pi = pos_to_idx[pos0]
                        s_mm += float(masked_lp[pi, m_id] - masked_lp[pi, w_id])
                        s_wm += float(wt_lp[pos0, m_id] - wt_lp[pos0, w_id])
                    mm_scores[idx] = s_mm
                    wm_scores[idx] = s_wm
                rho = _safe_spearman(mm_scores, targets)
                rho_wm = _safe_spearman(wm_scores, targets)

            if math.isfinite(rho):
                assay_rhos.append(rho)
                cluster_rhos[cluster].append(rho)
                stratum_assay_rhos[stratum].append(rho)
                stratum_cluster_rhos[stratum][cluster].append(rho)
            if math.isfinite(rho_wm):
                wt_marg_assay_rhos.append(rho_wm)
    finally:
        if was_training:
            model.train()

    by_stratum: dict[str, Any] = {}
    for st in sorted(stratum_assay_rhos.keys()):
        a_vals = stratum_assay_rhos[st]
        c_means = [float(np.mean(v)) for v in stratum_cluster_rhos[st].values() if v]
        by_stratum[st] = {
            "n_assays": len(a_vals),
            "n_clusters": len(c_means),
            "assay_mean_spearman": float(np.mean(a_vals)) if a_vals else float("nan"),
            "cluster_mean_spearman": float(np.mean(c_means)) if c_means else float("nan"),
        }

    cluster_means = [float(np.mean(v)) for v in cluster_rhos.values() if v]
    out: dict[str, Any] = {
        "scope": "milestone_subset" if milestone_only else "full_le384",
        "scoring_method": "exact_delta_logp" if objective == "ar" else "masked_marginal",
        "n_assays": len(assay_rhos),
        "n_clusters": len(cluster_means),
        "assay_mean_spearman": float(np.mean(assay_rhos)) if assay_rhos else float("nan"),
        "cluster_mean_spearman": float(np.mean(cluster_means)) if cluster_means else float("nan"),
        "by_stratum": by_stratum,
        "eval_wall_sec": float(time.perf_counter() - t0),
    }
    if wt_marg_assay_rhos:
        out["wt_marginal_assay_mean_spearman"] = float(np.mean(wt_marg_assay_rhos))
    return out


# --------------------------------------------------------------------------- #
# 3. Unseen-Fold 3D Contact Prediction via Zero-Shot Categorical Jacobian + APC
# --------------------------------------------------------------------------- #


def apply_apc(sym_matrix: np.ndarray) -> np.ndarray:
    """Apply Average Product Correction (APC) to a symmetric zero-diagonal (L, L) matrix."""
    mat = np.asarray(sym_matrix, dtype=np.float64).copy()
    L = mat.shape[0]
    np.fill_diagonal(mat, 0.0)
    if L <= 2:
        return mat
    row_mean = mat.sum(axis=1) / float(L - 1)
    grand_mean = float(mat.sum()) / float(L * (L - 1))
    if grand_mean > 1e-12:
        mat = mat - np.outer(row_mean, row_mean) / grand_mean
    np.fill_diagonal(mat, 0.0)
    return mat


@torch.no_grad()
def compute_categorical_jacobian_contact_map(
    model: nn.Module,
    sequence: str,
    *,
    objective: str,
    batch_size: int = 256,
    precision: str = "bf16",
) -> np.ndarray:
    """Fast GPU-batched Zero-Shot Categorical Jacobian + APC contact score matrix (L, L).

    Mutates each position ``i`` to all 20 canonical AAs:
      - Bidirectional (``mlm_15``, ``oa_uniform``): reads unmasked logits at all ``j != i``,
        double-centers across input AA ``a`` and output AA ``b``, computes Frobenius norm,
        symmetrizes ``0.5 * (raw + raw.T)``, and applies APC.
      - Causal (``ar``): reads causal downstream logits at ``j > i``, double-centers across
        ``(a, b)``, computes Frobenius norm, reflects upper triangle symmetrically
        (``triu + triu.T``), and applies APC.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    autocast_ctx = _eval_autocast_ctx(device, precision)

    wt_enc = encode_aa_sequence(sequence)
    L = int(wt_enc.size)
    wt_t = torch.from_numpy(wt_enc).to(device=device, dtype=torch.long)
    aa_arange = torch.arange(VOCAB_AA, device=device, dtype=torch.long)
    pos_per_batch = max(1, int(batch_size) // VOCAB_AA)

    raw = np.zeros((L, L), dtype=np.float64)

    try:
        for i_start in range(0, L, pos_per_batch):
            i_end = min(L, i_start + pos_per_batch)
            k = i_end - i_start
            bsz = k * VOCAB_AA

            # Construct (k, 20, L) mutated sequences
            muts = wt_t.view(1, 1, L).expand(k, VOCAB_AA, L).clone()
            for offset in range(k):
                muts[offset, :, i_start + offset] = aa_arange
            muts_flat = muts.reshape(bsz, L)

            if objective == "ar":
                bos = torch.full((bsz, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
                inp = torch.cat((bos, muts_flat[:, :-1]), dim=1)
                with autocast_ctx:
                    logits = model(inp, attention_mask=None, right_padded=True)
            else:
                valid_b = torch.ones((bsz, L), dtype=torch.bool, device=device)
                with autocast_ctx:
                    logits = model(muts_flat, valid_b)

            # Reshape to (k, 20_in, L, 20_out) in fp32 and double-center over (a, b)
            jac = logits.float().view(k, VOCAB_AA, L, VOCAB_AA)
            jac = (
                jac
                - jac.mean(dim=1, keepdim=True)
                - jac.mean(dim=3, keepdim=True)
                + jac.mean(dim=(1, 3), keepdim=True)
            )
            frob = torch.sqrt(jac.pow(2).sum(dim=(1, 3)) + 1e-12)  # (k, L)
            raw[i_start:i_end, :] = frob.cpu().numpy()
    finally:
        if was_training:
            model.train()

    np.fill_diagonal(raw, 0.0)
    if objective == "ar":
        # Strictly causal: only downstream responses j > i reflect sensitivity to residue i
        upper = np.triu(raw, k=1)
        sym = upper + upper.T
    else:
        sym = 0.5 * (raw + raw.T)
    return apply_apc(sym)


def compute_contact_metrics_for_domain(
    score_matrix: np.ndarray,
    cb_coords: np.ndarray,
    cb_mask: np.ndarray,
    *,
    min_sep: int = 6,
    contact_cutoff: float = CONTACT_DISTANCE_ANGSTROM,
) -> dict[str, float] | None:
    """Compute P@L, P@L/2, P@L/5, and AUROC for upper-triangle pairs with |i-j| >= min_sep."""
    L = score_matrix.shape[0]
    if L <= min_sep:
        return None
    ii, jj = np.triu_indices(L, k=min_sep)
    valid_pair = cb_mask[ii] & cb_mask[jj]
    ii = ii[valid_pair]
    jj = jj[valid_pair]
    if ii.size < 2:
        return None

    dists = np.linalg.norm(cb_coords[ii].astype(np.float64) - cb_coords[jj].astype(np.float64), axis=-1)
    labels = dists < float(contact_cutoff)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None

    scores = score_matrix[ii, jj].astype(np.float64)
    auroc = _fast_binary_auroc(scores, labels)

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]

    def _prec_at(k: int) -> float:
        kk = max(1, min(int(k), int(sorted_labels.size)))
        return float(sorted_labels[:kk].mean())

    return {
        "p_at_L": _prec_at(L),
        "p_at_L_over_2": _prec_at(max(1, L // 2)),
        "p_at_L_over_5": _prec_at(max(1, L // 5)),
        "auroc": float(auroc),
        "n_pairs": int(labels.size),
        "n_contacts": int(n_pos),
    }


@torch.no_grad()
def evaluate_unseen_fold_contacts(
    model: nn.Module,
    contact_domains: Sequence[Mapping[str, Any]],
    *,
    objective: str,
    batch_size: int = 256,
    precision: str = "bf16",
    max_domains: int | None = None,
) -> dict[str, Any]:
    """Evaluate zero-shot Categorical Jacobian + APC contacts across unseen-fold 3D domains."""
    t0 = time.perf_counter()
    domains = list(contact_domains[:max_domains]) if max_domains is not None else list(contact_domains)

    sep6_metrics: list[dict[str, float]] = []
    sep24_metrics: list[dict[str, float]] = []

    for dom in domains:
        seq = str(dom["sequence"])
        cb_coords = np.asarray(dom["cb_coords"], dtype=np.float32)
        cb_mask = np.asarray(dom["cb_mask"], dtype=np.bool_)

        apc_map = compute_categorical_jacobian_contact_map(
            model,
            seq,
            objective=objective,
            batch_size=batch_size,
            precision=precision,
        )
        m6 = compute_contact_metrics_for_domain(apc_map, cb_coords, cb_mask, min_sep=6)
        if m6 is not None:
            sep6_metrics.append(m6)
        m24 = compute_contact_metrics_for_domain(apc_map, cb_coords, cb_mask, min_sep=24)
        if m24 is not None:
            sep24_metrics.append(m24)

    def _agg(rows: list[dict[str, float]]) -> dict[str, Any]:
        if not rows:
            return {
                "n_domains": 0,
                "p_at_L": float("nan"),
                "p_at_L_over_2": float("nan"),
                "p_at_L_over_5": float("nan"),
                "auroc": float("nan"),
            }
        return {
            "n_domains": len(rows),
            "p_at_L": float(np.mean([r["p_at_L"] for r in rows])),
            "p_at_L_over_2": float(np.mean([r["p_at_L_over_2"] for r in rows])),
            "p_at_L_over_5": float(np.mean([r["p_at_L_over_5"] for r in rows])),
            "auroc": float(np.mean([r["auroc"] for r in rows])),
        }

    return {
        "method": "categorical_jacobian_apc",
        "objective": objective,
        "sep_ge_6": _agg(sep6_metrics),
        "long_range_sep_ge_24": _agg(sep24_metrics),
        "eval_wall_sec": float(time.perf_counter() - t0),
    }
