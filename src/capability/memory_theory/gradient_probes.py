"""Track B2a: Dimensionless residual gradient effective rank & minibatch gradient SNR."""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .io_utils import LN2, cluster_bootstrap_mean
from .model_hooks import (
    LoadedModelBundle,
    get_input_embedding_module,
    get_transformer_blocks,
    select_depth_quantiles,
)


def compute_spectral_rank_metrics(gram: np.ndarray, d_model: int) -> dict[str, float]:
    """Compute dimensionless Roy-Vetterli & von Neumann effective rank ratios from Gram matrix G^T G."""
    sym = 0.5 * (gram + gram.T)
    eigvals = np.linalg.eigvalsh(sym)
    eigvals = np.maximum(eigvals, 0.0)
    total_var = float(np.sum(eigvals))
    if total_var <= 0.0:
        return {
            "vn_effective_rank": 0.0,
            "vn_effective_rank_ratio": 0.0,
            "rv_singular_effective_rank": 0.0,
            "rv_singular_effective_rank_ratio": 0.0,
            "numerical_rank_99": 0.0,
            "numerical_rank_99_ratio": 0.0,
            "top1_variance_share": 0.0,
            "top19_variance_share": 0.0,
        }

    # von Neumann entropy on eigenvalues lambda_i (variance spectrum of covariance Sigma_g)
    p = eigvals[eigvals > 1e-18 * total_var] / total_var
    h_vn = float(-np.sum(p * np.log(p)))
    vn_rank = float(math.exp(h_vn))

    # Roy-Vetterli (2007) entropy on singular values sigma_i = sqrt(lambda_i)
    sigmas = np.sqrt(eigvals[eigvals > 1e-18 * total_var])
    q = sigmas / float(np.sum(sigmas))
    h_rv = float(-np.sum(q * np.log(q)))
    rv_rank = float(math.exp(h_rv))

    sorted_eig = np.sort(eigvals)[::-1]
    cum = np.cumsum(sorted_eig) / total_var
    rank_99 = int(np.searchsorted(cum, 0.99) + 1)
    top1 = float(sorted_eig[0] / total_var)
    top19 = float(np.sum(sorted_eig[: min(19, sorted_eig.size)]) / total_var)

    return {
        "vn_effective_rank": vn_rank,
        "vn_effective_rank_ratio": float(vn_rank / d_model),
        "rv_singular_effective_rank": rv_rank,
        "rv_singular_effective_rank_ratio": float(rv_rank / d_model),
        "numerical_rank_99": float(rank_99),
        "numerical_rank_99_ratio": float(rank_99 / d_model),
        "top1_variance_share": top1,
        "top19_variance_share": top19,
    }


def compute_gradient_snr(grad_vectors: np.ndarray, *, n_draws: int = 1000, seed: int = 20261120) -> dict[str, Any]:
    """Compute dimensionless minibatch gradient SNR and pairwise cosine alignment with bootstrap CI.

    Args:
        grad_vectors: array of shape (B, D) where each row is one sample/unit gradient vector.
    """
    arr = np.asarray(grad_vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return {
            "n_samples": int(arr.shape[0]) if arr.ndim == 2 else 0,
            "snr_raw": 0.0,
            "snr_unbiased": 0.0,
            "mean_pairwise_cosine": 0.0,
            "status": "insufficient_samples",
        }

    # Filter any zero-norm rows safely
    norms = np.linalg.norm(arr, axis=1)
    keep = norms > 1e-12
    arr = arr[keep]
    norms = norms[keep]
    b = int(arr.shape[0])
    if b < 2:
        return {
            "n_samples": b,
            "snr_raw": 0.0,
            "snr_unbiased": 0.0,
            "mean_pairwise_cosine": 0.0,
            "status": "insufficient_nonzero_samples",
        }

    def _snr_stats(mat: np.ndarray) -> tuple[float, float, float]:
        n = mat.shape[0]
        mean_g = np.mean(mat, axis=0)
        sq_mean_norm = float(np.dot(mean_g, mean_g))
        centered = mat - mean_g
        var_trace = float(np.sum(centered * centered) / (n - 1))
        snr_raw = sq_mean_norm / max(var_trace, 1e-18)
        # Unbiased estimator of ||E[g]||^2 = ||mean_g||^2 - (1/n) * Var(g)
        unbiased_sig = max(0.0, sq_mean_norm - var_trace / n)
        snr_unbiased = unbiased_sig / max(var_trace, 1e-18)

        unit = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
        sum_u = np.sum(unit, axis=0)
        # Sum_{i != j} u_i . u_j = ||sum u_i||^2 - n
        mean_cos = float((np.dot(sum_u, sum_u) - n) / (n * (n - 1)))
        return snr_raw, snr_unbiased, mean_cos

    snr_raw, snr_unbiased, mean_cos = _snr_stats(arr)

    rng = np.random.default_rng(seed)
    boot_raw = np.empty(n_draws, dtype=np.float64)
    boot_unb = np.empty(n_draws, dtype=np.float64)
    boot_cos = np.empty(n_draws, dtype=np.float64)
    for d in range(n_draws):
        idx = rng.integers(0, b, size=b)
        r_v, u_v, c_v = _snr_stats(arr[idx])
        boot_raw[d] = r_v
        boot_unb[d] = u_v
        boot_cos[d] = c_v

    return {
        "n_samples": b,
        "snr_raw": float(snr_raw),
        "snr_raw_ci95": [float(np.quantile(boot_raw, 0.025)), float(np.quantile(boot_raw, 0.975))],
        "snr_unbiased": float(snr_unbiased),
        "snr_unbiased_ci95": [float(np.quantile(boot_unb, 0.025)), float(np.quantile(boot_unb, 0.975))],
        "mean_pairwise_cosine": float(mean_cos),
        "mean_pairwise_cosine_ci95": [float(np.quantile(boot_cos, 0.025)), float(np.quantile(boot_cos, 0.975))],
        "status": "ok",
    }


def _extract_block_output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise RuntimeError(f"Unsupported transformer block output type: {type(output)}")


def measure_residual_gradient_rank_and_snr(
    bundle: LoadedModelBundle,
    sequences: Sequence[str],
    *,
    max_tokens: int = 256,
    mask_stride: int = 7,
    target_positions: Sequence[int | None] | None = None,
    seed: int = 20261120,
) -> dict[str, Any]:
    """Measure residual-stream gradient effective rank ratio and sequence-level gradient SNR.

    Hooks block outputs at 5 depth quantiles {0, L/4, L/2, 3L/4, L-1}, runs forward + backward
    on exact per-sequence cross-entropy (causal next-token or masked-position MLM), and
    accumulates exact Gram matrices G^T G in float64 (constant O(d^2) memory).
    """
    model = bundle.model
    tok = bundle.tokenizer
    blocks = get_transformer_blocks(model)
    layer_indices = select_depth_quantiles(len(blocks))
    d_model = bundle.d_model

    # Freeze model weights so backward only flows to hooked activations cleanly without wasting
    # memory on parameter .grad buffers.
    prev_req = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    # Enable gradient flow through input embeddings (works even when custom models lack get_input_embeddings)
    emb = get_input_embedding_module(model)

    grams: dict[int, np.ndarray] = {li: np.zeros((d_model, d_model), dtype=np.float64) for li in layer_indices}
    token_counts: dict[int, int] = {li: 0 for li in layer_indices}
    seq_mean_grads: dict[int, list[np.ndarray]] = {li: [] for li in layer_indices}
    losses_bits: list[float] = []

    try:
        for seq_idx, text in enumerate(sequences):
            enc = tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
                add_special_tokens=bundle.is_masked,
            )
            input_ids = enc["input_ids"].to(bundle.device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(bundle.device)
            seq_len = int(input_ids.shape[1])
            if seq_len < 4:
                continue

            # Determine positions where loss is evaluated
            forced_pos = target_positions[seq_idx] if target_positions is not None else None
            if bundle.is_masked:
                mask_id = tok.mask_token_id
                if mask_id is None:
                    raise RuntimeError(f"Masked model {bundle.arm} has no mask_token_id")
                labels = input_ids.clone()
                if forced_pos is not None:
                    pos = int(forced_pos)
                    if not (1 <= pos < seq_len - 1):
                        continue
                    eval_pos = [pos]
                else:
                    eval_pos = list(range(1, seq_len - 1, mask_stride))
                    if not eval_pos:
                        continue
                masked_ids = input_ids.clone()
                for p_idx in eval_pos:
                    masked_ids[0, p_idx] = mask_id
                feed_ids = masked_ids
                grad_read_indices = eval_pos
            else:
                feed_ids = input_ids
                if forced_pos is not None:
                    # Causal prediction of token at forced_pos comes from hidden state at forced_pos - 1
                    pos = int(forced_pos)
                    if not (1 <= pos < seq_len):
                        continue
                    grad_read_indices = [pos - 1]
                else:
                    grad_read_indices = list(range(0, seq_len - 1))

            captured: dict[int, torch.Tensor] = {}
            handles = []

            def _emb_hook(_mod: Any, _inp: Any, out: torch.Tensor) -> torch.Tensor:
                if isinstance(out, torch.Tensor):
                    out = out.detach().requires_grad_(True)
                return out

            handles.append(emb.register_forward_hook(_emb_hook))

            for li in layer_indices:
                def _make_hook(layer_id: int):
                    def _hook(_mod: Any, _inp: Any, out: Any) -> None:
                        t = _extract_block_output_tensor(out)
                        if t.requires_grad:
                            t.retain_grad()
                            captured[layer_id] = t
                    return _hook
                handles.append(blocks[li].register_forward_hook(_make_hook(li)))

            try:
                model.zero_grad(set_to_none=True)
                kwargs: dict[str, Any] = {"input_ids": feed_ids}
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask
                out = model(**kwargs)
                logits = out.logits.float()  # (1, T, V)

                if bundle.is_masked:
                    pos_t = torch.tensor(grad_read_indices, device=bundle.device, dtype=torch.long)
                    sel_logits = logits[0, pos_t, :]
                    sel_targets = labels[0, pos_t]
                    per_tok_loss = F.cross_entropy(sel_logits, sel_targets, reduction="none")
                else:
                    if forced_pos is not None:
                        p_idx = grad_read_indices[0]
                        sel_logits = logits[0, p_idx : p_idx + 1, :]
                        sel_targets = feed_ids[0, p_idx + 1 : p_idx + 2]
                        per_tok_loss = F.cross_entropy(sel_logits, sel_targets, reduction="none")
                    else:
                        sel_logits = logits[0, :-1, :]
                        sel_targets = feed_ids[0, 1:]
                        per_tok_loss = F.cross_entropy(sel_logits, sel_targets, reduction="none")

                loss_sum = per_tok_loss.sum()
                losses_bits.append(float(per_tok_loss.mean().item() / LN2))
                loss_sum.backward()

                for li in layer_indices:
                    t = captured.get(li)
                    if t is None or t.grad is None:
                        continue
                    grad_t = t.grad
                    if bundle.seq_first_hidden and grad_t.ndim == 3 and grad_t.shape[0] == seq_len and grad_t.shape[1] == 1:
                        grad_t = grad_t.transpose(0, 1)
                    g = grad_t[0, grad_read_indices, :].detach().to(torch.float64).cpu().numpy()
                    if not np.isfinite(g).all():
                        continue
                    grams[li] += g.T @ g
                    token_counts[li] += int(g.shape[0])
                    seq_mean_grads[li].append(np.mean(g, axis=0))
            finally:
                for h in handles:
                    h.remove()
                captured.clear()
    finally:
        for p, req in zip(model.parameters(), prev_req):
            p.requires_grad_(req)

    # Effective vocabulary size (for AA20 PLMs, observed token output space is 20 -> rank <= 19)
    eff_vocab = min(bundle.vocab_size, 20) if (bundle.modality == "protein" and bundle.vocab_size <= 64) else bundle.vocab_size
    theo_rank_ratio = float(min(max(eff_vocab - 1, 1), d_model) / d_model)

    layers_out: dict[str, Any] = {}
    for li in layer_indices:
        n_tok = token_counts[li]
        rank_metrics = compute_spectral_rank_metrics(grams[li] / max(n_tok, 1), d_model)
        snr_metrics = compute_gradient_snr(
            np.asarray(seq_mean_grads[li], dtype=np.float64) if seq_mean_grads[li] else np.zeros((0, d_model)),
            n_draws=500,
            seed=seed + li,
        )
        layers_out[str(li)] = {
            "layer_index": li,
            "depth_fraction": float(li / max(len(blocks) - 1, 1)),
            "n_tokens": n_tok,
            "rank": rank_metrics,
            "sequence_gradient_snr": snr_metrics,
        }

    return {
        "arm": bundle.arm,
        "modality": bundle.modality,
        "is_masked": bundle.is_masked,
        "d_model": d_model,
        "n_layers": len(blocks),
        "vocab_size": bundle.vocab_size,
        "theoretical_final_layer_rank_ratio": theo_rank_ratio,
        "n_sequences_scored": len(losses_bits),
        "mean_loss_bits_per_token": cluster_bootstrap_mean(losses_bits, n_draws=500, seed=seed) if losses_bits else None,
        "layers": layers_out,
    }
