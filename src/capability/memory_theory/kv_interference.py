"""Track B2b: Attention & MLP key-value constructive vs destructive interference geometry."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .io_utils import cluster_bootstrap_mean
from .model_hooks import (
    BlockKVHookSpec,
    LoadedModelBundle,
    get_transformer_blocks,
    inspect_block_kv_modules,
    select_depth_quantiles,
    split_packed_kv,
)

ISOTROPIC_ABS_DOT_BASELINE = math.sqrt(2.0 / math.pi)  # E[|N(0,1)|] ~= 0.79788456


def compute_pair_kv_geometry(
    k_a: np.ndarray,
    v_a: np.ndarray,
    k_b: np.ndarray,
    v_b: np.ndarray,
) -> dict[str, float]:
    """Compute dimensionless key collision excess & constructive/destructive KV alignment.

    Args:
        k_a: (T_a, H, d_k), v_a: (T_a, H, d_v), k_b: (T_b, H, d_k), v_b: (T_b, H, d_v)
    """
    _, n_heads, d_k = k_a.shape
    kce_heads = []
    align_all_heads = []
    align_top10_heads = []
    tail_v_sim_heads = []
    corr_heads = []
    mean_k_cos_heads = []

    sqrt_dk = math.sqrt(float(d_k))
    for h in range(n_heads):
        ka = k_a[:, h, :].astype(np.float64)
        kb = k_b[:, h, :].astype(np.float64)
        va = v_a[:, h, :].astype(np.float64)
        vb = v_b[:, h, :].astype(np.float64)

        ka /= np.maximum(np.linalg.norm(ka, axis=1, keepdims=True), 1e-12)
        kb /= np.maximum(np.linalg.norm(kb, axis=1, keepdims=True), 1e-12)
        va /= np.maximum(np.linalg.norm(va, axis=1, keepdims=True), 1e-12)
        vb /= np.maximum(np.linalg.norm(vb, axis=1, keepdims=True), 1e-12)

        s_mat = ka @ kb.T  # (T_a, T_b)
        u_mat = va @ vb.T  # (T_a, T_b)

        abs_s = np.abs(s_mat)
        kce = float(np.mean(abs_s) * sqrt_dk)
        kce_heads.append(kce)
        mean_k_cos_heads.append(float(np.mean(s_mat)))

        prod = s_mat * u_mat
        align_all_heads.append(float(np.mean(prod)))

        tau90 = float(np.quantile(abs_s, 0.90))
        mask_tail = abs_s >= tau90
        if np.any(mask_tail):
            align_top10_heads.append(float(np.mean(prod[mask_tail])))
            # Signed high-positive-key value similarity
            pos_tau90 = float(np.quantile(s_mat, 0.90))
            pos_mask = s_mat >= pos_tau90
            tail_v_sim_heads.append(float(np.mean(u_mat[pos_mask])))
        else:
            align_top10_heads.append(0.0)
            tail_v_sim_heads.append(0.0)

        s_flat = s_mat.ravel()
        u_flat = u_mat.ravel()
        s_std = float(np.std(s_flat))
        u_std = float(np.std(u_flat))
        if s_std > 1e-12 and u_std > 1e-12:
            corr_heads.append(float(np.corrcoef(s_flat, u_flat)[0, 1]))
        else:
            corr_heads.append(0.0)

    kce_mean = float(np.mean(kce_heads))
    return {
        "key_collision_excess": kce_mean,
        "key_collision_iso_ratio": float(kce_mean / ISOTROPIC_ABS_DOT_BASELINE),
        "mean_key_cosine": float(np.mean(mean_k_cos_heads)),
        "kv_alignment_all": float(np.mean(align_all_heads)),
        "kv_alignment_top10": float(np.mean(align_top10_heads)),
        "tail_value_similarity": float(np.mean(tail_v_sim_heads)),
        "kv_similarity_corr": float(np.mean(corr_heads)),
    }


def compute_auroc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """Exact Mann-Whitney AUROC (0.5 = chance, 1.0 = perfect separation)."""
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0).astype(np.float64) + 0.5 * (diff == 0).astype(np.float64)))


def _subsample_positions(length: int, max_positions: int = 48, skip_ends: bool = False) -> np.ndarray:
    start = 1 if (skip_ends and length > 2) else 0
    end = length - 1 if (skip_ends and length > 2) else length
    avail = end - start
    if avail <= 0:
        return np.arange(0, length, dtype=np.int64)
    if avail <= max_positions:
        return np.arange(start, end, dtype=np.int64)
    return np.linspace(start, end - 1, num=max_positions, dtype=np.int64)


@torch.no_grad()
def extract_sequence_kv_features(
    bundle: LoadedModelBundle,
    sequences: Mapping[str, str],
    *,
    max_tokens: int = 256,
    max_sampled_positions: int = 48,
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Extract subsampled attention (K, V) and MLP (key_act, out_proxy) per layer for each sequence."""
    model = bundle.model
    tok = bundle.tokenizer
    blocks = get_transformer_blocks(model)
    layer_indices = select_depth_quantiles(len(blocks))
    specs: dict[int, BlockKVHookSpec] = {
        li: inspect_block_kv_modules(blocks[li], li) for li in layer_indices
    }

    features: dict[str, dict[int, dict[str, np.ndarray]]] = {}

    for seq_id, text in sequences.items():
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
        pos_idx = _subsample_positions(seq_len, max_positions=max_sampled_positions, skip_ends=bundle.is_masked)

        raw_k: dict[int, torch.Tensor] = {}
        raw_v: dict[int, torch.Tensor] = {}
        raw_mlp_k: dict[int, torch.Tensor] = {}
        raw_blk_out: dict[int, torch.Tensor] = {}
        handles = []

        def _to_batch_first(t: torch.Tensor) -> torch.Tensor:
            if bundle.seq_first_hidden and t.ndim == 3 and t.shape[0] == seq_len and t.shape[1] == 1:
                return t.transpose(0, 1).contiguous()
            return t

        for li, spec in specs.items():
            if spec.kv_mode == "separate":
                def _k_hook(layer_id: int):
                    def _fn(_m: Any, _i: Any, o: torch.Tensor) -> None:
                        raw_k[layer_id] = _to_batch_first(o.detach())
                    return _fn

                def _v_hook(layer_id: int):
                    def _fn(_m: Any, _i: Any, o: torch.Tensor) -> None:
                        raw_v[layer_id] = _to_batch_first(o.detach())
                    return _fn

                handles.append(spec.k_module.register_forward_hook(_k_hook(li)))
                handles.append(spec.v_module.register_forward_hook(_v_hook(li)))
            else:
                def _qkv_hook(layer_id: int, mode: str):
                    def _fn(_m: Any, _i: Any, o: torch.Tensor) -> None:
                        k_t, v_t = split_packed_kv(
                            _to_batch_first(o.detach()),
                            mode,
                            n_heads=bundle.n_heads,
                            head_dim=bundle.head_dim,
                        )
                        raw_k[layer_id] = k_t
                        raw_v[layer_id] = v_t
                    return _fn

                handles.append(spec.k_module.register_forward_hook(_qkv_hook(li, spec.kv_mode)))

            if spec.mlp_act_module is not None:
                def _mlp_hook(layer_id: int):
                    def _fn(_m: Any, _i: Any, o: torch.Tensor) -> None:
                        # Apply GELU nonlinearity so MLP key reflects active neuron pattern
                        raw_mlp_k[layer_id] = F.gelu(_to_batch_first(o.detach()).float())
                    return _fn

                handles.append(spec.mlp_act_module.register_forward_hook(_mlp_hook(li)))

            def _blk_hook(layer_id: int):
                def _fn(_m: Any, _i: Any, o: Any) -> None:
                    t = o[0] if isinstance(o, (tuple, list)) else o
                    raw_blk_out[layer_id] = _to_batch_first(t.detach())
                return _fn

            handles.append(blocks[li].register_forward_hook(_blk_hook(li)))

        try:
            kwargs: dict[str, Any] = {"input_ids": input_ids}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            model(**kwargs)

            per_layer: dict[int, dict[str, np.ndarray]] = {}
            pos_t = torch.tensor(pos_idx, device=bundle.device, dtype=torch.long)
            for li in layer_indices:
                k_t = raw_k[li][0, pos_t, :].float()  # (P, D_kv)
                v_t = raw_v[li][0, pos_t, :].float()  # (P, D_kv)
                # Handle GQA where num_kv_heads <= n_heads cleanly
                d_kv = int(k_t.shape[-1])
                head_dim = bundle.head_dim if (d_kv % bundle.head_dim == 0) else (d_kv // max(bundle.n_heads, 1))
                n_kv_heads = max(1, d_kv // max(head_dim, 1))
                k_np = k_t.view(-1, n_kv_heads, head_dim).cpu().numpy().astype(np.float32)
                v_np = v_t.view(-1, n_kv_heads, head_dim).cpu().numpy().astype(np.float32)

                entry: dict[str, np.ndarray] = {"attn_k": k_np, "attn_v": v_np}
                if li in raw_mlp_k and li in raw_blk_out:
                    mk = raw_mlp_k[li][0, pos_t, :].float().cpu().numpy().astype(np.float32)
                    mv = raw_blk_out[li][0, pos_t, :].float().cpu().numpy().astype(np.float32)
                    # Store as 1-head (P, 1, D) for identical geometry evaluator
                    entry["mlp_k"] = mk[:, None, :]
                    entry["mlp_v"] = mv[:, None, :]
                per_layer[li] = entry
            features[seq_id] = per_layer
        finally:
            for h in handles:
                h.remove()

    return features


def evaluate_kv_interference_pairs(
    bundle: LoadedModelBundle,
    sequences: Mapping[str, str],
    pairs_by_stratum: Mapping[str, Sequence[tuple[str, str]]],
    *,
    max_tokens: int = 256,
    max_sampled_positions: int = 48,
    n_draws: int = 1000,
    seed: int = 20261120,
) -> dict[str, Any]:
    """Evaluate attention & MLP KV collision excess and constructive/destructive alignment across strata."""
    needed_ids = {
        sid
        for plist in pairs_by_stratum.values()
        for a, b in plist
        for sid in (a, b)
        if sid in sequences
    }
    needed_seqs = {sid: sequences[sid] for sid in sorted(needed_ids)}
    feats = extract_sequence_kv_features(
        bundle,
        needed_seqs,
        max_tokens=max_tokens,
        max_sampled_positions=max_sampled_positions,
    )

    blocks = get_transformer_blocks(bundle.model)
    layer_indices = select_depth_quantiles(len(blocks))

    strata_results: dict[str, Any] = {}
    raw_per_stratum_layer: dict[str, dict[int, dict[str, list[float]]]] = {}

    for stratum, plist in pairs_by_stratum.items():
        layer_out: dict[str, Any] = {}
        raw_per_stratum_layer[stratum] = {}
        for li in layer_indices:
            attn_metrics_list = []
            mlp_metrics_list = []
            for a, b in plist:
                if a not in feats or b not in feats:
                    continue
                fa, fb = feats[a][li], feats[b][li]
                attn_metrics_list.append(
                    compute_pair_kv_geometry(fa["attn_k"], fa["attn_v"], fb["attn_k"], fb["attn_v"])
                )
                if "mlp_k" in fa and "mlp_k" in fb:
                    mlp_metrics_list.append(
                        compute_pair_kv_geometry(fa["mlp_k"], fa["mlp_v"], fb["mlp_k"], fb["mlp_v"])
                    )

            def _summarize(mlist: list[dict[str, float]]) -> tuple[dict[str, Any], dict[str, list[float]]]:
                if not mlist:
                    return {}, {}
                keys = list(mlist[0].keys())
                raw_map = {k: [m[k] for m in mlist] for k in keys}
                summ = {
                    k: cluster_bootstrap_mean(raw_map[k], n_draws=n_draws, seed=seed + li)
                    for k in keys
                }
                return summ, raw_map

            attn_summ, attn_raw = _summarize(attn_metrics_list)
            mlp_summ, _ = _summarize(mlp_metrics_list)
            raw_per_stratum_layer[stratum][li] = attn_raw
            layer_out[str(li)] = {
                "layer_index": li,
                "depth_fraction": float(li / max(len(blocks) - 1, 1)),
                "n_pairs": len(attn_metrics_list),
                "attention_kv": attn_summ,
                "mlp_kv": mlp_summ,
            }
        strata_results[stratum] = layer_out

    # Compute discrimination AUROCs between key strata (e.g. twilight vs unrelated, close vs unrelated, related vs random)
    contrasts: dict[str, Any] = {}
    candidate_pairs = [
        ("close_homolog", "unrelated_matched_ss"),
        ("twilight_remote_homolog", "unrelated_matched_ss"),
        ("close_homolog", "twilight_remote_homolog"),
        ("bm25_top10_related", "random_unrelated"),
    ]
    for pos_s, neg_s in candidate_pairs:
        if pos_s not in raw_per_stratum_layer or neg_s not in raw_per_stratum_layer:
            continue
        c_layers: dict[str, Any] = {}
        for li in layer_indices:
            p_map = raw_per_stratum_layer[pos_s].get(li, {})
            n_map = raw_per_stratum_layer[neg_s].get(li, {})
            if not p_map or not n_map:
                continue
            c_layers[str(li)] = {
                metric: {
                    "auroc": compute_auroc(p_map[metric], n_map[metric]),
                    "mean_diff": float(np.mean(p_map[metric]) - np.mean(n_map[metric])),
                }
                for metric in ("kv_alignment_top10", "tail_value_similarity", "kv_similarity_corr", "key_collision_excess")
                if metric in p_map and metric in n_map
            }
        contrasts[f"{pos_s}_vs_{neg_s}"] = c_layers

    return {
        "arm": bundle.arm,
        "modality": bundle.modality,
        "n_layers": len(blocks),
        "n_heads": bundle.n_heads,
        "head_dim": bundle.head_dim,
        "isotropic_abs_dot_baseline": ISOTROPIC_ABS_DOT_BASELINE,
        "strata": strata_results,
        "discrimination_contrasts": contrasts,
    }
