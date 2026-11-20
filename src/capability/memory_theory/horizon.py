"""Track B3: Prefix mutual information horizon Delta H(L_ctx) for text vs protein models."""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .io_utils import LN2, cluster_bootstrap_mean, paired_cluster_bootstrap_diff
from .model_hooks import LoadedModelBundle

DEFAULT_CONTEXT_GRID: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def compute_saturation_horizon(
    context_grid: Sequence[int],
    mean_h_by_ctx: Sequence[float],
    fraction: float = 0.90,
) -> float | None:
    """Interpolate context length L_f achieving `fraction` of total H(L_min) -> H(L_max) reduction."""
    xs = np.asarray(context_grid, dtype=np.float64)
    ys = np.asarray(mean_h_by_ctx, dtype=np.float64)
    total_gain = float(ys[0] - ys[-1])
    if total_gain <= 1e-9:
        return None
    target_h = float(ys[0] - fraction * total_gain)
    for i in range(1, len(xs)):
        if ys[i] <= target_h:
            y0, y1 = float(ys[i - 1]), float(ys[i])
            x0, x1 = math.log2(float(xs[i - 1])), math.log2(float(xs[i]))
            if abs(y0 - y1) < 1e-12:
                return float(xs[i])
            alpha = (y0 - target_h) / (y0 - y1)
            return float(2.0 ** (x0 + alpha * (x1 - x0)))
    return float(xs[-1])


@torch.no_grad()
def evaluate_prefix_horizon(
    bundle: LoadedModelBundle,
    sequences: Sequence[str],
    *,
    context_grid: Sequence[int] = DEFAULT_CONTEXT_GRID,
    anchor_offsets: Sequence[int] = (512,),
    n_draws: int = 1000,
    seed: int = 20261120,
) -> dict[str, Any]:
    r"""Evaluate fixed-anchor prefix cross-entropy H(L_ctx) across L_ctx in {1..512}.

    For causal models:
      - Primary endpoint (position-confound-free): fixes the exact same target token x_{t*}
        (e.g. t* = 512) and evaluates -log2 p(x_{t*} | x_{t*-L_ctx : t*-1}) for every L_ctx in
        context_grid. Every context length predicts the exact same target token!
      - Secondary endpoint (full-pass running curve): single teacher-forced pass over length-513
        prefix giving exact position-wise H(t) averaged in logarithmic bins around each L_ctx.
    For masked models:
      - Evaluates -log2 p(x_{t*} | x_{[t*-L_ctx/2 : t*+L_ctx/2] \ {t*}}) with x_{t*} masked.
    """
    model = bundle.model
    tok = bundle.tokenizer
    grid = tuple(sorted(int(x) for x in context_grid))
    max_ctx = max(grid)
    max_anchor = max(anchor_offsets)
    needed_len = max(max_ctx + 1, max_anchor + (max_ctx // 2 + 2 if bundle.is_masked else 2))

    # Fixed-anchor per-sequence losses: shape (N_valid, len(grid))
    fixed_anchor_rows: list[list[float]] = []
    # Single-pass causal position-wise losses: shape (N_valid, len(grid))
    running_pass_rows: list[list[float]] = []
    tokens_per_char: list[float] = []

    for text in sequences:
        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=needed_len + 8,
            add_special_tokens=bundle.is_masked,
        )
        ids = enc["input_ids"][0]
        seq_len = int(ids.shape[0])
        if bundle.is_masked:
            # Need room on both sides of anchor if possible; adapt anchor to seq_len // 2 if shorter
            if seq_len < max_ctx + 3:
                continue
            anchor = min(max_anchor, seq_len - (max_ctx // 2) - 2)
            anchor = max(anchor, max_ctx // 2 + 1)
            mask_id = tok.mask_token_id
            if mask_id is None:
                raise RuntimeError(f"Masked model {bundle.arm} lacks mask_token_id")
            row_bits: list[float] = []
            for l_ctx in grid:
                left = l_ctx // 2
                right = l_ctx - left
                lo = max(1, anchor - left)
                hi = min(seq_len - 1, anchor + right + 1)
                win = ids[lo:hi].clone()
                rel_anchor = anchor - lo
                target_id = int(win[rel_anchor].item())
                win[rel_anchor] = mask_id
                # Re-attach BOS/EOS tokens
                full = torch.cat([ids[:1], win, ids[-1:]], dim=0).unsqueeze(0).to(bundle.device)
                out = model(input_ids=full)
                logits = out.logits[0, rel_anchor + 1, :].float()
                nll = float(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_id], device=bundle.device)).item())
                row_bits.append(nll / LN2)
            fixed_anchor_rows.append(row_bits)
            tokens_per_char.append(float((seq_len - 2) / max(len(text), 1)))
        else:
            # Causal model: require at least max_ctx + 1 tokens
            if seq_len < max_ctx + 1:
                continue
            anchor = min(max_anchor, seq_len - 1)
            if anchor < max_ctx:
                anchor = max_ctx
            row_bits = []
            target_id = int(ids[anchor].item())
            for l_ctx in grid:
                # Context is exact l_ctx tokens immediately preceding anchor
                ctx_ids = ids[anchor - l_ctx : anchor].unsqueeze(0).to(bundle.device)
                out = model(input_ids=ctx_ids)
                logits = out.logits[0, -1, :].float()
                nll = float(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_id], device=bundle.device)).item())
                row_bits.append(nll / LN2)
            fixed_anchor_rows.append(row_bits)

            # Also run 1 full teacher-forced pass over ids[:max_ctx + 1] for running position curve
            full_ids = ids[: max_ctx + 1].unsqueeze(0).to(bundle.device)
            out_full = model(input_ids=full_ids)
            full_logits = out_full.logits[0, :-1, :].float()
            full_targets = full_ids[0, 1:]
            per_pos_bits = (
                F.cross_entropy(full_logits, full_targets, reduction="none").cpu().numpy() / LN2
            )
            # per_pos_bits[k] is prediction of token k+1 given k+1 context tokens (L_ctx = k+1)
            run_row = [float(per_pos_bits[l_ctx - 1]) for l_ctx in grid]
            running_pass_rows.append(run_row)
            tokens_per_char.append(float(seq_len / max(len(text), 1)))

    if not fixed_anchor_rows:
        raise RuntimeError(
            f"No sequences met minimum token length ({max_ctx + 1}) for arm={bundle.arm}"
        )

    mat = np.asarray(fixed_anchor_rows, dtype=np.float64)  # (N, G)
    n_seq = int(mat.shape[0])
    h_unif = math.log2(max(bundle.vocab_size, 2))
    h0_vec = mat[:, 0]  # L_ctx = 1 baseline

    curve_fixed: list[dict[str, Any]] = []
    mean_h_list: list[float] = []
    for gi, l_ctx in enumerate(grid):
        col = mat[:, gi]
        h_stat = cluster_bootstrap_mean(col, n_draws=n_draws, seed=seed + gi)
        mean_h_list.append(h_stat["point"])

        if gi == 0:
            marg = {"point": 0.0, "ci95": [0.0, 0.0], "se": 0.0, "n_clusters": n_seq, "n_draws": 0}
        else:
            marg = paired_cluster_bootstrap_diff(mat[:, gi - 1], col, n_draws=n_draws, seed=seed + 100 + gi)

        cum_gain = paired_cluster_bootstrap_diff(h0_vec, col, n_draws=n_draws, seed=seed + 200 + gi)
        frac_red_vs_l1 = float((float(np.mean(h0_vec)) - h_stat["point"]) / max(float(np.mean(h0_vec)), 1e-12))
        frac_red_vs_unif = float((h_unif - h_stat["point"]) / max(h_unif, 1e-12))

        curve_fixed.append(
            {
                "l_ctx": l_ctx,
                "h_bits_per_token": h_stat,
                "marginal_gain_bits": marg,
                "cumulative_gain_vs_l1_bits": cum_gain,
                "fractional_reduction_vs_l1": frac_red_vs_l1,
                "fractional_reduction_vs_uniform": frac_red_vs_unif,
            }
        )

    # Bootstrap L_90 and L_50 saturation horizons
    l90_point = compute_saturation_horizon(grid, mean_h_list, fraction=0.90)
    l50_point = compute_saturation_horizon(grid, mean_h_list, fraction=0.50)
    rng = np.random.default_rng(seed)
    l90_draws: list[float] = []
    l50_draws: list[float] = []
    if n_seq >= 2:
        for _ in range(n_draws):
            idx = rng.integers(0, n_seq, size=n_seq)
            m_draw = np.mean(mat[idx], axis=0)
            v90 = compute_saturation_horizon(grid, m_draw, fraction=0.90)
            v50 = compute_saturation_horizon(grid, m_draw, fraction=0.50)
            if v90 is not None:
                l90_draws.append(v90)
            if v50 is not None:
                l50_draws.append(v50)

    running_summary = None
    if running_pass_rows:
        rmat = np.asarray(running_pass_rows, dtype=np.float64)
        running_summary = [
            {
                "l_ctx": l_ctx,
                "h_bits_per_token": cluster_bootstrap_mean(rmat[:, gi], n_draws=n_draws, seed=seed + 300 + gi),
            }
            for gi, l_ctx in enumerate(grid)
        ]

    return {
        "arm": bundle.arm,
        "modality": bundle.modality,
        "is_masked": bundle.is_masked,
        "vocab_size": bundle.vocab_size,
        "uniform_ceiling_bits_per_token": h_unif,
        "n_sequences_evaluated": n_seq,
        "mean_tokens_per_char": float(np.mean(tokens_per_char)),
        "context_grid": list(grid),
        "fixed_anchor_curve": curve_fixed,
        "running_position_curve": running_summary,
        "saturation_horizon": {
            "L50_tokens": l50_point,
            "L50_ci95": [float(np.quantile(l50_draws, 0.025)), float(np.quantile(l50_draws, 0.975))] if len(l50_draws) >= 20 else None,
            "L90_tokens": l90_point,
            "L90_ci95": [float(np.quantile(l90_draws, 0.025)), float(np.quantile(l90_draws, 0.975))] if len(l90_draws) >= 20 else None,
            "total_gain_l1_to_lmax_bits": float(mean_h_list[0] - mean_h_list[-1]),
            "total_fractional_reduction_l1_to_lmax": float((mean_h_list[0] - mean_h_list[-1]) / max(mean_h_list[0], 1e-12)),
        },
    }
