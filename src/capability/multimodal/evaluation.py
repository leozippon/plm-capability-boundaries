"""Evaluators for the controlled multimodal suite (Eq. 3a-3b, B_ext(Z), downstream, & structure readout)."""
from __future__ import annotations

import contextlib
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..supply_demand.evaluation import (
    DEFAULT_LOO_PROBES,
    NUM_EVAL_STRATA,
    _deterministic_valid_mask,
    _safe_spearman,
    compute_contact_metrics_for_domain,
    compute_temperature_calibrated_bits,
    compute_three_tier_decomposition,
    evaluate_megascale_ddg,
    evaluate_proteingym_substitutions,
    evaluate_split_loo,
    evaluate_unseen_fold_contacts,
)
from ..supply_demand.objectives import encode_aa_sequence, prepare_ar_batch
from .data import (
    PackedMultimodalStore,
    compute_backbone_geom_features,
    encode_3di_sequence,
)
from .model import MultiTrackProteinTransformer
from .spec import (
    AA_TO_ID,
    BOS_TOKEN_ID,
    CONTACT_DISTANCE_CUTOFF,
    LN2,
    MASK_TOKEN_ID,
    STRUCT_ABSENT_TOKEN_ID,
    VOCAB_AA,
    canonicalize_objective,
    default_mask_rate_for_objective,
    is_fixed_mask_objective,
    objective_display_name,
)


def _eval_autocast(device: torch.device, precision: str) -> Any:
    if precision == "bf16" and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _shuffle_struct_within_seq(
    struct_tokens: torch.Tensor,
    geom_feats: torch.Tensor,
    lengths: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically permute 3Di tokens and local geom features within each sequence."""
    bsz, max_len = struct_tokens.shape
    dev = struct_tokens.device
    s_out = struct_tokens.clone()
    g_out = geom_feats.clone()
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    for i in range(bsz):
        L = int(lengths[i].item())
        if L > 1:
            perm = torch.randperm(L, generator=gen).to(device=dev)
            s_out[i, :L] = struct_tokens[i, perm]
            g_out[i, :L] = geom_feats[i, perm]
    return s_out, g_out


@torch.no_grad()
def evaluate_multimodal_split_native(
    model: MultiTrackProteinTransformer,
    store: PackedMultimodalStore,
    *,
    objective: str,
    use_structure: bool,
    shuffle_structure: bool = False,
    batch_size: int = 64,
    precision: str = "bf16",
    n_draws: int = NUM_EVAL_STRATA,
    eval_seed: int = 20261129,
    mask_rate: float | None = None,
    eval_temperature_calibration: bool = False,
    cal_max_tokens: int = 262_144,
) -> dict[str, Any]:
    """Evaluate native AA cross-entropy (bits/res) with Z = Z_true, Z = Z_shuf, or Z = empty."""
    canon_obj = canonicalize_objective(objective)
    if store.n_sequences == 0:
        raise ValueError("Cannot evaluate empty PackedMultimodalStore")

    t0 = time.perf_counter()
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    autocast_ctx = _eval_autocast(device, precision)
    eff_mask_rate = float(mask_rate) if mask_rate is not None else default_mask_rate_for_objective(canon_obj)

    order_idx = np.argsort(store.lengths, kind="mergesort")
    n_seq = store.n_sequences

    total_weighted_bits = 0.0
    total_residues = 0
    seq_mean_bits_sum = 0.0
    total_correct = 0.0
    total_eval_tokens = 0.0

    cal_logits_chunks: list[torch.Tensor] = []
    cal_targets_chunks: list[torch.Tensor] = []
    cal_weights_chunks: list[torch.Tensor] = []
    cal_tokens_collected = 0

    def _collect_cal(
        logits_b: torch.Tensor,
        targets_b: torch.Tensor,
        mask_b: torch.Tensor,
        weights_b: torch.Tensor,
    ) -> None:
        nonlocal cal_tokens_collected
        if not eval_temperature_calibration or cal_tokens_collected >= int(cal_max_tokens):
            return
        sel_l = logits_b[mask_b].detach().to(device="cpu", dtype=torch.float16)
        sel_t = targets_b[mask_b].detach().to(device="cpu", dtype=torch.uint8)
        sel_w = weights_b[mask_b].detach().to(device="cpu", dtype=torch.float32)
        n_new = int(sel_t.numel())
        if n_new == 0:
            return
        rem = int(cal_max_tokens) - cal_tokens_collected
        if n_new > rem:
            sel_l, sel_t, sel_w = sel_l[:rem], sel_t[:rem], sel_w[:rem]
            n_new = rem
        cal_logits_chunks.append(sel_l)
        cal_targets_chunks.append(sel_t)
        cal_weights_chunks.append(sel_w)
        cal_tokens_collected += n_new

    try:
        for start in range(0, n_seq, max(1, int(batch_size))):
            batch_np = order_idx[start : start + int(batch_size)]
            batch = store.collate_indices(batch_np, device=device, pad_to_multiple=8)
            tokens = batch["aa_tokens"]
            lengths = batch["lengths"]
            bsz, max_len = tokens.shape
            global_idx_t = torch.from_numpy(batch_np).to(device=device, dtype=torch.int64)
            pos = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
            valid = pos < lengths.unsqueeze(1)

            if use_structure:
                s_tok = batch["struct_tokens"]
                g_feat = batch["geom_feats"]
                cb_c = batch["cb_coords"]
                cb_m = batch["cb_mask"]
                if shuffle_structure:
                    s_tok, g_feat = _shuffle_struct_within_seq(
                        s_tok, g_feat, lengths, seed=eval_seed + start
                    )
                    # Break 3D pairwise correspondence as well by setting cb_mask to False
                    cb_m = torch.zeros_like(cb_m)
                s_present = torch.ones((bsz,), dtype=torch.bool, device=device)
            else:
                s_tok = None
                g_feat = None
                cb_c = None
                cb_m = None
                s_present = torch.zeros((bsz,), dtype=torch.bool, device=device)

            if canon_obj == "ar":
                prep = prepare_ar_batch(tokens, lengths)
                with autocast_ctx:
                    logits = model(
                        prep.input_ids,
                        prep.attention_mask,
                        struct_tokens=s_tok,
                        geom_feats=g_feat,
                        cb_coords=cb_c,
                        cb_mask=cb_m,
                        struct_present=s_present,
                        right_padded=True,
                    )
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
                total_correct += float(((preds == safe_t) & prep.loss_mask).sum().item())
                total_eval_tokens += float(lengths.sum().item())
                total_weighted_bits += float(seq_sum_bits.sum().item())
                total_residues += int(lengths.sum().item())
                seq_mean_bits_sum += float(seq_h.sum().item())
                if eval_temperature_calibration:
                    _collect_cal(logits, safe_t, prep.loss_mask, torch.ones_like(safe_t, dtype=torch.float32))

            elif canon_obj in ("oa_uniform", "mdm_cosine"):
                h_draws = torch.empty((bsz, int(n_draws)), dtype=torch.float64, device=device)
                for s in range(int(n_draws)):
                    off = torch.remainder(global_idx_t + 7 * s, int(n_draws))
                    u = (float(s) + (off.to(dtype=torch.float64) + 0.5) / float(n_draws)) / float(n_draws)
                    m_mask = torch.clamp(
                        torch.floor(u * lengths.to(dtype=torch.float64)).to(dtype=torch.int64) + 1,
                        min=1,
                    )
                    m_mask = torch.minimum(m_mask, lengths.to(dtype=torch.int64))
                    mask = _deterministic_valid_mask(
                        global_idx_t, lengths, m_mask, max_len, draw_idx=s, eval_seed=eval_seed
                    )
                    safe_t = torch.where(valid, tokens, torch.zeros_like(tokens))
                    inp = torch.where(mask, MASK_TOKEN_ID, tokens)
                    with autocast_ctx:
                        logits = model(
                            inp,
                            valid,
                            struct_tokens=s_tok,
                            geom_feats=g_feat,
                            cb_coords=cb_c,
                            cb_mask=cb_m,
                            struct_present=s_present,
                        )
                    tok_bits = (
                        F.cross_entropy(logits.float().transpose(1, 2), safe_t, reduction="none").to(
                            torch.float64
                        )
                        / LN2
                    )
                    masked_bits = torch.where(mask, tok_bits, torch.zeros_like(tok_bits))
                    h_draws[:, s] = masked_bits.sum(dim=-1) / m_mask.to(dtype=torch.float64)
                    preds = logits.argmax(dim=-1)
                    total_correct += float(((preds == safe_t) & mask).sum().item())
                    total_eval_tokens += float(mask.sum().item())
                    if eval_temperature_calibration:
                        w_row = lengths.to(dtype=torch.float32) / (m_mask.to(dtype=torch.float32) * float(n_draws))
                        _collect_cal(logits, safe_t, mask, w_row.unsqueeze(1).expand_as(safe_t))

                seq_h = h_draws.mean(dim=-1)
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
                        global_idx_t, lengths, k_mask, max_len, draw_idx=d, eval_seed=eval_seed
                    )
                    safe_t = torch.where(valid, tokens, torch.zeros_like(tokens))
                    inp = torch.where(mask, MASK_TOKEN_ID, tokens)
                    with autocast_ctx:
                        logits = model(
                            inp,
                            valid,
                            struct_tokens=s_tok,
                            geom_feats=g_feat,
                            cb_coords=cb_c,
                            cb_mask=cb_m,
                            struct_present=s_present,
                        )
                    tok_bits = (
                        F.cross_entropy(logits.float().transpose(1, 2), safe_t, reduction="none").to(
                            torch.float64
                        )
                        / LN2
                    )
                    masked_bits = torch.where(mask, tok_bits, torch.zeros_like(tok_bits))
                    h_draws[:, d] = masked_bits.sum(dim=-1) / k_mask.to(dtype=torch.float64)
                    preds = logits.argmax(dim=-1)
                    total_correct += float(((preds == safe_t) & mask).sum().item())
                    total_eval_tokens += float(mask.sum().item())
                    if eval_temperature_calibration:
                        w_row = lengths.to(dtype=torch.float32) / (k_mask.to(dtype=torch.float32) * float(n_draws))
                        _collect_cal(logits, safe_t, mask, w_row.unsqueeze(1).expand_as(safe_t))

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
        "use_structure": bool(use_structure),
        "shuffle_structure": bool(shuffle_structure),
        "n_sequences": int(n_seq),
        "total_residues": int(total_residues),
        "bits_per_res": float(total_weighted_bits / float(max(1, total_residues))),
        "seq_mean_bits_per_res": float(seq_mean_bits_sum / float(n_seq)),
        "token_accuracy": float(total_correct / float(max(1.0, total_eval_tokens))),
        "eval_wall_sec": float(time.perf_counter() - t0),
    }
    if eval_temperature_calibration and cal_logits_chunks:
        cal_res = compute_temperature_calibrated_bits(
            torch.cat(cal_logits_chunks, dim=0),
            torch.cat(cal_targets_chunks, dim=0),
            weights=torch.cat(cal_weights_chunks, dim=0),
        )
        out["temperature_calibration"] = cal_res
    return out


@torch.no_grad()
def evaluate_multimodal_three_tier(
    model: MultiTrackProteinTransformer,
    *,
    train_sub_store: PackedMultimodalStore,
    unseen_homolog_store: PackedMultimodalStore,
    unseen_fold_store: PackedMultimodalStore,
    mode: str,
    objective: str,
    batch_size: int = 64,
    precision: str = "bf16",
    n_draws: int = NUM_EVAL_STRATA,
    loo_probes: int = DEFAULT_LOO_PROBES,
    eval_seed: int = 20261129,
    mask_rate: float | None = None,
    eval_temperature_calibration: bool = True,
) -> dict[str, Any]:
    """Evaluate Eq. (3a)-(3b) three-tier bits and B_ext(Z) external conditioning offload.

    - ``pure_seq``: Z = empty (B_ext = 0).
    - ``struct_cond``: primary evaluation at Z = Z_true (B_ext > 0), plus counterfactual
      Z = empty and Z = Z_shuf on unseen_fold to quantify exact b_ext(Z) = H(X) - H(X | Z).
    - ``joint_to_seq`` (ESM-3): primary evaluation STRICTLY at Z = empty (B_ext = 0),
      plus optional Z = Z_true diagnostic to measure in-model conditional capacity.
    """
    t0 = time.perf_counter()
    canon_obj = canonicalize_objective(objective)
    primary_use_struct = mode == "struct_cond"

    splits = {
        "train": train_sub_store,
        "unseen_homolog": unseen_homolog_store,
        "unseen_fold": unseen_fold_store,
    }
    splits_native: dict[str, dict[str, Any]] = {}
    for idx, (name, st) in enumerate(splits.items()):
        splits_native[name] = evaluate_multimodal_split_native(
            model,
            st,
            objective=canon_obj,
            use_structure=primary_use_struct,
            batch_size=batch_size,
            precision=precision,
            n_draws=n_draws,
            eval_seed=eval_seed + idx * 101,
            mask_rate=mask_rate,
            eval_temperature_calibration=(eval_temperature_calibration and name == "unseen_fold"),
        )

    native_eq3a = compute_three_tier_decomposition(
        h_train=splits_native["train"]["bits_per_res"],
        h_unseen_homolog=splits_native["unseen_homolog"]["bits_per_res"],
        h_unseen_fold=splits_native["unseen_fold"]["bits_per_res"],
    )
    tcal = splits_native["unseen_fold"].get("temperature_calibration")
    if isinstance(tcal, Mapping):
        native_eq3a["H_unseen_fold_cal"] = float(tcal["calibrated_bits_per_tok"])
        native_eq3a["Delta_cal"] = float(tcal["Delta_cal"])
        native_eq3a["optimal_temperature_unseen_fold"] = float(tcal["optimal_temperature"])

    # Compute pure-sequence LOO when Z = empty (for pure_seq & joint_to_seq)
    loo_eq3a: dict[str, Any] | None = None
    if not primary_use_struct:
        loo_splits = {}
        for idx, (name, st) in enumerate(splits.items()):
            loo_splits[name] = evaluate_split_loo(
                model,
                st.to_aa_sequence_store(),
                objective=canon_obj,
                batch_size=batch_size,
                precision=precision,
                n_probes=loo_probes,
                eval_seed=eval_seed + 10_000 + idx * 101,
            )
        loo_eq3a = compute_three_tier_decomposition(
            h_train=loo_splits["train"]["bits_per_res"],
            h_unseen_homolog=loo_splits["unseen_homolog"]["bits_per_res"],
            h_unseen_fold=loo_splits["unseen_fold"]["bits_per_res"],
        )

    # Exact external structural information b_ext(Z) = H(X) - H(X | Z) diagnostics
    b_ext_diag: dict[str, Any] = {
        "eval_condition": "Z_true" if primary_use_struct else "Z_empty",
        "B_ext_active_at_eval": bool(primary_use_struct),
    }
    if mode == "struct_cond":
        uf_absent = evaluate_multimodal_split_native(
            model,
            unseen_fold_store,
            objective=canon_obj,
            use_structure=False,
            batch_size=batch_size,
            precision=precision,
            n_draws=n_draws,
            eval_seed=eval_seed + 202,
            mask_rate=mask_rate,
        )
        uf_shuf = evaluate_multimodal_split_native(
            model,
            unseen_fold_store,
            objective=canon_obj,
            use_structure=True,
            shuffle_structure=True,
            batch_size=batch_size,
            precision=precision,
            n_draws=n_draws,
            eval_seed=eval_seed + 202,
            mask_rate=mask_rate,
        )
        h_true = float(splits_native["unseen_fold"]["bits_per_res"])
        h_abs = float(uf_absent["bits_per_res"])
        h_shuf = float(uf_shuf["bits_per_res"])
        n_res_uf = int(unseen_fold_store.total_tokens)
        b_ext_diag.update(
            {
                "H_unseen_fold_Z_true": h_true,
                "H_unseen_fold_Z_empty": h_abs,
                "H_unseen_fold_Z_shuffled": h_shuf,
                "b_ext_vs_empty_bits_per_res": float(h_abs - h_true),
                "b_ext_vs_shuffled_bits_per_res": float(h_shuf - h_true),
                "B_ext_unseen_fold_total_bits": float((h_shuf - h_true) * n_res_uf),
            }
        )
    elif mode == "joint_to_seq":
        # Measure both Z=empty (primary!) and Z=true (conditional upper diagnostic)
        uf_cond = evaluate_multimodal_split_native(
            model,
            unseen_fold_store,
            objective=canon_obj,
            use_structure=True,
            batch_size=batch_size,
            precision=precision,
            n_draws=n_draws,
            eval_seed=eval_seed + 202,
            mask_rate=mask_rate,
        )
        h_empty = float(splits_native["unseen_fold"]["bits_per_res"])
        h_cond = float(uf_cond["bits_per_res"])
        b_ext_diag.update(
            {
                "H_unseen_fold_Z_empty": h_empty,
                "H_unseen_fold_Z_true_diagnostic": h_cond,
                "b_ext_conditional_gain_bits_per_res": float(h_empty - h_cond),
            }
        )

    return {
        "mode": mode,
        "objective": canon_obj,
        "paradigm_display": objective_display_name(canon_obj),
        "native": native_eq3a,
        "loo": loo_eq3a,
        "temperature_calibration": tcal,
        "b_ext_diagnostics": b_ext_diag,
        "splits_native": splits_native,
        "eval_wall_sec": float(time.perf_counter() - t0),
    }


@torch.no_grad()
def evaluate_sequence_to_structure_readout(
    model: MultiTrackProteinTransformer,
    unseen_fold_store: PackedMultimodalStore,
    *,
    batch_size: int = 32,
    precision: str = "bf16",
    max_domains: int = 128,
) -> dict[str, Any]:
    """Direct sequence-to-structure readout at Z = empty (3Di CE/accuracy + contact head AUROC/P@L).

    Strictly passes ONLY pure AA sequence (Z = empty, struct_present = False) and evaluates the
    model's direct ``struct_3di_head`` and ``contact_head`` on held-out unseen-fold structures.
    """
    if not model.has_struct_heads:
        return {"available": False}

    t0 = time.perf_counter()
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    autocast_ctx = _eval_autocast(device, precision)

    n_eval = min(unseen_fold_store.n_sequences, int(max_domains))
    total_3di_bits = 0.0
    total_3di_correct = 0.0
    total_3di_tokens = 0.0
    sep6_rows: list[dict[str, float]] = []
    sep24_rows: list[dict[str, float]] = []

    try:
        for start in range(0, n_eval, max(1, int(batch_size))):
            idx_np = np.arange(start, min(n_eval, start + int(batch_size)), dtype=np.int64)
            batch = unseen_fold_store.collate_indices(idx_np, device=device, pad_to_multiple=8)
            tokens = batch["aa_tokens"]
            lengths = batch["lengths"]
            s_targets = batch["struct_tokens"]
            cb_coords = batch["cb_coords"].cpu().numpy()
            cb_mask = batch["cb_mask"].cpu().numpy()
            bsz, max_len = tokens.shape
            pos = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
            valid = pos < lengths.unsqueeze(1)
            struct_absent = torch.zeros((bsz,), dtype=torch.bool, device=device)

            with autocast_ctx:
                out = model.forward_multimodal(
                    tokens,
                    valid,
                    struct_tokens=None,
                    geom_feats=None,
                    cb_coords=None,
                    cb_mask=None,
                    struct_present=struct_absent,
                    right_padded=True,
                    return_struct_heads=True,
                )

            if out.struct_3di_logits is not None:
                safe_s = torch.where(valid, s_targets, torch.zeros_like(s_targets))
                ce_bits = (
                    F.cross_entropy(
                        out.struct_3di_logits.float().transpose(1, 2), safe_s, reduction="none"
                    ).to(torch.float64)
                    / LN2
                )
                total_3di_bits += float(torch.where(valid, ce_bits, torch.zeros_like(ce_bits)).sum().item())
                preds = out.struct_3di_logits.argmax(dim=-1)
                total_3di_correct += float(((preds == safe_s) & valid).sum().item())
                total_3di_tokens += float(valid.sum().item())

            if out.contact_logits is not None:
                c_np = out.contact_logits.float().cpu().numpy()
                for b_i in range(bsz):
                    L = int(lengths[b_i].item())
                    cmap = c_np[b_i, :L, :L]
                    m6 = compute_contact_metrics_for_domain(
                        cmap, cb_coords[b_i, :L], cb_mask[b_i, :L], min_sep=6, contact_cutoff=CONTACT_DISTANCE_CUTOFF
                    )
                    if m6 is not None:
                        sep6_rows.append(m6)
                    m24 = compute_contact_metrics_for_domain(
                        cmap, cb_coords[b_i, :L], cb_mask[b_i, :L], min_sep=24, contact_cutoff=CONTACT_DISTANCE_CUTOFF
                    )
                    if m24 is not None:
                        sep24_rows.append(m24)
    finally:
        if was_training:
            model.train()

    def _agg(rows: list[dict[str, float]]) -> dict[str, Any]:
        if not rows:
            return {"n_domains": 0, "p_at_L": float("nan"), "p_at_L_over_5": float("nan"), "auroc": float("nan")}
        return {
            "n_domains": len(rows),
            "p_at_L": float(np.mean([r["p_at_L"] for r in rows])),
            "p_at_L_over_2": float(np.mean([r["p_at_L_over_2"] for r in rows])),
            "p_at_L_over_5": float(np.mean([r["p_at_L_over_5"] for r in rows])),
            "auroc": float(np.mean([r["auroc"] for r in rows])),
        }

    return {
        "available": True,
        "eval_condition": "Z_empty_pure_sequence",
        "n_domains": int(n_eval),
        "three_di_xent_bits_per_res": float(total_3di_bits / max(1.0, total_3di_tokens)),
        "three_di_accuracy": float(total_3di_correct / max(1.0, total_3di_tokens)),
        "contact_head_sep_ge_6": _agg(sep6_rows),
        "contact_head_long_range_sep_ge_24": _agg(sep24_rows),
        "eval_wall_sec": float(time.perf_counter() - t0),
    }


@torch.no_grad()
def evaluate_multimodal_megascale(
    model: MultiTrackProteinTransformer,
    megascale_payload: Mapping[str, Any],
    *,
    mode: str,
    objective: str,
    batch_size: int = 128,
    precision: str = "bf16",
) -> dict[str, Any]:
    """Evaluate MegaScale ddG: strictly Z = empty for pure_seq/joint_to_seq, and Z = Z_WT for struct_cond."""
    canon_obj = canonicalize_objective(objective)
    if mode != "struct_cond":
        res = evaluate_megascale_ddg(
            model,
            megascale_payload,
            objective=canon_obj,
            batch_size=batch_size,
            precision=precision,
        )
        res["eval_condition"] = "Z_empty"
        return res

    # Structure-conditioned MegaScale scoring using attached WT 3Di + C_beta geometry
    t0 = time.perf_counter()
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    autocast_ctx = _eval_autocast(device, precision)

    primary_by_stratum: dict[str, list[float]] = defaultdict(list)
    all_rhos: list[float] = []
    n_with_struct = 0

    try:
        for dom in megascale_payload.get("domains", []):
            wt_seq = str(dom["wildtype"])
            wt_enc = encode_aa_sequence(wt_seq)
            L = int(wt_enc.size)
            variants = dom.get("variants", [])
            if len(variants) < 3:
                continue

            has_wt_struct = ("three_di" in dom) and ("cb_coords" in dom)
            if has_wt_struct:
                n_with_struct += 1
                s3_1 = torch.from_numpy(encode_3di_sequence(str(dom["three_di"]))).to(
                    device=device, dtype=torch.long
                ).unsqueeze(0)
                cb_np = np.asarray(dom["cb_coords"], dtype=np.float32)
                cbm_np = np.asarray(dom.get("cb_mask", np.ones(L, dtype=np.bool_)), dtype=np.bool_)
                g_np = (
                    np.asarray(dom["geom_feats"], dtype=np.float32)
                    if "geom_feats" in dom
                    else compute_backbone_geom_features(cb_np, cbm_np)
                )
                cb_1 = torch.from_numpy(cb_np).to(device=device, dtype=torch.float32).unsqueeze(0)
                cbm_1 = torch.from_numpy(cbm_np).to(device=device, dtype=torch.bool).unsqueeze(0)
                g_1 = torch.from_numpy(g_np).to(device=device, dtype=torch.float32).unsqueeze(0)
            else:
                s3_1 = cb_1 = cbm_1 = g_1 = None

            pos_list = np.asarray([int(v[0]) for v in variants], dtype=np.int64)
            wt_ids = np.asarray([AA_TO_ID[str(v[1])] for v in variants], dtype=np.int64)
            mut_ids = np.asarray([AA_TO_ID[str(v[2])] for v in variants], dtype=np.int64)
            targets = np.asarray([float(v[3]) for v in variants], dtype=np.float64)
            stratum = str(dom.get("stratum", "unknown"))

            if canon_obj == "ar":
                n_var = len(variants)
                all_seqs = np.repeat(wt_enc[None, :], n_var + 1, axis=0)
                all_seqs[1:, :][np.arange(n_var), pos_list] = mut_ids.astype(np.uint8)
                nlls = np.empty(n_var + 1, dtype=np.float64)
                for st_i in range(0, n_var + 1, max(1, int(batch_size))):
                    chunk = all_seqs[st_i : st_i + int(batch_size)]
                    cur = chunk.shape[0]
                    tgt = torch.from_numpy(chunk).to(device=device, dtype=torch.long)
                    bos = torch.full((cur, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
                    inp = torch.cat((bos, tgt[:, :-1]), dim=1)
                    sp = torch.full((cur,), bool(has_wt_struct), dtype=torch.bool, device=device)
                    with autocast_ctx:
                        logits = model(
                            inp,
                            struct_tokens=s3_1.expand(cur, -1) if s3_1 is not None else None,
                            geom_feats=g_1.expand(cur, -1, -1) if g_1 is not None else None,
                            cb_coords=cb_1.expand(cur, -1, -1) if cb_1 is not None else None,
                            cb_mask=cbm_1.expand(cur, -1) if cbm_1 is not None else None,
                            struct_present=sp,
                            right_padded=True,
                        )
                    tok_nll = F.cross_entropy(logits.float().transpose(1, 2), tgt, reduction="none").to(
                        torch.float64
                    )
                    nlls[st_i : st_i + cur] = tok_nll.sum(dim=-1).cpu().numpy()
                scores = nlls[0] - nlls[1:]
                rho = _safe_spearman(scores, targets)
            else:
                uniq_pos = np.unique(pos_list)
                pos_to_idx = {int(p): i for i, p in enumerate(uniq_pos)}
                wt_t = torch.from_numpy(wt_enc).to(device=device, dtype=torch.long)
                masked_lp = np.empty((len(uniq_pos), VOCAB_AA), dtype=np.float32)
                for st_i in range(0, len(uniq_pos), max(1, int(batch_size))):
                    p_chunk = uniq_pos[st_i : st_i + int(batch_size)]
                    cur = int(p_chunk.size)
                    inp = wt_t.unsqueeze(0).expand(cur, -1).clone()
                    row_ar = torch.arange(cur, device=device, dtype=torch.long)
                    pos_t = torch.from_numpy(p_chunk).to(device=device, dtype=torch.long)
                    inp[row_ar, pos_t] = MASK_TOKEN_ID
                    valid_b = torch.ones((cur, L), dtype=torch.bool, device=device)
                    sp = torch.full((cur,), bool(has_wt_struct), dtype=torch.bool, device=device)
                    with autocast_ctx:
                        logits = model(
                            inp,
                            valid_b,
                            struct_tokens=s3_1.expand(cur, -1) if s3_1 is not None else None,
                            geom_feats=g_1.expand(cur, -1, -1) if g_1 is not None else None,
                            cb_coords=cb_1.expand(cur, -1, -1) if cb_1 is not None else None,
                            cb_mask=cbm_1.expand(cur, -1) if cbm_1 is not None else None,
                            struct_present=sp,
                        )
                    masked_lp[st_i : st_i + cur] = F.log_softmax(logits[row_ar, pos_t].float(), dim=-1).cpu().numpy()
                mapped = np.asarray([pos_to_idx[int(p)] for p in pos_list], dtype=np.int64)
                mm_scores = masked_lp[mapped, mut_ids] - masked_lp[mapped, wt_ids]
                rho = _safe_spearman(mm_scores, targets)

            if math.isfinite(rho):
                all_rhos.append(rho)
                primary_by_stratum[stratum].append(rho)
    finally:
        if was_training:
            model.train()

    strata_summary = {
        st: {"n_domains": len(vals), "spearman_mean": float(np.mean(vals))}
        for st, vals in sorted(primary_by_stratum.items())
    }
    return {
        "eval_condition": "Z_WT_conditioned",
        "n_domains_with_wt_structure": int(n_with_struct),
        "scoring_method": "exact_delta_logp" if canon_obj == "ar" else "masked_marginal",
        "n_domains": len(all_rhos),
        "overall_spearman_mean": float(np.mean(all_rhos)) if all_rhos else float("nan"),
        "by_stratum": strata_summary,
        "eval_wall_sec": float(time.perf_counter() - t0),
    }


__all__ = [
    "evaluate_multimodal_megascale",
    "evaluate_multimodal_split_native",
    "evaluate_multimodal_three_tier",
    "evaluate_proteingym_substitutions",
    "evaluate_sequence_to_structure_readout",
    "evaluate_unseen_fold_contacts",
]
