"""Unified 62-Model Middle-Layer (R_mid) vs. Final-Layer (R_last) Representation Benchmark
across MegaScale Stability, Human Domainome Abundance, and Double-Mutant Epistasis.

Strictly pure-CPU reuse of the already-extracted 1024-d hidden-state representation archives
(zero sealed artifacts modified; zero GPU re-extraction required):

  - ``R_mid``   (cols ``0:512``):  canonical middle transformer layer ``blocks[(L - 1) // 2]``
                                   (1-based ``L // 2`` for even ``L``; 17/33 for odd ``L=33``),
                                   pooled ``[mean, last/cls]`` & projected (2 x 256 = 512d).
  - ``R_last``  (cols ``512:1024``): final transformer layer ``blocks[L - 1]``, pooled with the
                                   identical rule & projected (2 x 256 = 512d).
  - ``R_joint`` (cols ``0:1024``): full 1024-d ``[R_mid, R_last]`` evaluated as ``R`` in sealed
                                   ``r2_biophysical_unified_62.json`` &
                                   ``r3_epistasis_and_contacts_unified_62.json``, verified here
                                   against both sealed artifacts to ``< 1e-10`` across all 62 arms.

Sign convention (documented in output JSON):
  - All paired contrasts are oriented so **positive (> 0) means improvement / R_mid wins**:
      * MSE endpoints (``transfer_error``, ``calibrated_error``, ``r3_mse``):
          ``mid_minus_last_gain = MSE(last) - MSE(mid)`` (> 0 => ``R_mid`` has lower MSE).
          ``plus_R_given_CM     = MSE(C+M)  - MSE(C+M+R)`` (> 0 => adding ``R`` lowers MSE).
      * Spearman rho endpoints (``ranking``, ``r3_spearman``):
          ``mid_minus_last_gain = rho(mid) - rho(last)`` (> 0 => ``R_mid`` has higher rho).
          ``plus_R_given_CM     = rho(C+M+R) - rho(C+M)`` (> 0 => adding ``R`` raises rho).
"""

from __future__ import annotations

import gc
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.capability.core.io import sha256_file
from src.capability.general_plm.eval_utils import atomic_write_json
from src.capability.general_plm.r3_cycle_eval import load_pairwise_cycle_context
from src.capability.interactions import pairwise_epistasis as pw_mod
from src.capability.mutation import external_confirmation as abundance_mod
from src.capability.readouts.readout_analysis import row_weights
from src.capability.stability import stability_gate as gate
from src.capability.unified import r2_experiment_2_1 as r2_mod
from src.capability.unified import r3_yardstick as r3_mod
from src.capability.unified.master_summary import FORBIDDEN_INFRA_RE
from src.capability.unified.roster import (
    ARM_METADATA,
    REPO_ROOT,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_OUTPUT_DIR,
)

SCHEMA = "r2_r3_middle_vs_last_representation_62_v1"
CELL_CACHE_SCHEMA = "r2_r3_mid_vs_last_cell_v1"

DEFAULT_OUTPUT_REL = Path(UNIFIED_OUTPUT_DIR) / "r2_r3_middle_vs_last_representation_62.json"
DEFAULT_CELL_CACHE_REL = Path(UNIFIED_OUTPUT_DIR) / "cells" / "r2_r3_mid_vs_last"

BOOTSTRAP_DRAWS = r2_mod.BOOTSTRAP_DRAWS
BOOTSTRAP_SEED = r2_mod.BOOTSTRAP_SEED
PARITY_TOLERANCE = 1e-10

# Authoritative layer counts L for all 62 single-sequence models (verified from
# readout_depth_panel_20260925, core/arms.py, and staged checkpoint configs).
MODEL_LAYERS_62: Mapping[str, int] = {
    # causal_text (15)
    "bygpt5-small-en": 4,
    "bygpt5-base-en": 6,
    "bygpt5-medium-en": 12,
    "dialogpt-small": 12,
    "gpt2": 12,
    "gpt2-medium": 24,
    "gpt2-large": 36,
    "gpt2-xl": 48,
    "llama-2-7b": 32,
    "llama-3.2-3b": 28,
    "qwen2.5-0.5b": 24,
    "qwen2.5-0.5b-instruct": 24,
    "qwen2.5-7b": 28,
    "qwen2.5-32b": 64,
    "qwen3-8b-base": 36,
    # causal_joint (5)
    "galactica-125m": 12,
    "galactica-1.3b": 24,
    "galactica-6.7b": 32,
    "galactica-30b": 48,
    "instructprotein": 24,
    # causal_adaptation (2)
    "prollama-stage-1": 32,
    "prollama": 32,
    # causal_protein + hybrid_ssm (18)
    "dayhoff-170m": 24,
    "progen2-small": 12,
    "progen2-base": 27,
    "progen2-medium": 27,
    "progen2-large": 32,
    "progen2-xlarge": 32,
    "progen3-112m": 11,
    "progen3-219m": 11,
    "progen3-339m": 17,
    "progen3-762m": 17,
    "progen3-1b": 17,
    "progen3-3b": 24,
    "proteinglm-7b-clm": 36,
    "protgpt2": 36,
    "protgpt3-112m": 8,
    "protgpt3-1.3b": 17,
    "protgpt3-10b": 32,
    "rita-xl": 24,
    # masked_lm (15)
    "amplify-120m": 24,
    "amplify-350m": 32,
    "esm2-8m": 6,
    "esm2-35m": 12,
    "esm2-150m": 30,
    "esm2-650m": 33,
    "esm2-3b": 36,
    "esm2-15b": 48,
    "esmc-300m": 30,
    "esmc-600m": 36,
    "esmc-6b": 80,
    "prime-690m": 33,
    "proteinglm-1b-mlm": 24,
    "proteinglm-3b-mlm": 36,
    "proteinglm-10b-mlm": 40,
    # structure_aware_mlm (3)
    "saprot-35m": 12,
    "saprot-650m": 33,
    "saprot-1.3b": 66,
    # discrete_diffusion (3) + multimodal_seq_only (1)
    "dplm-150m": 30,
    "dplm-650m": 33,
    "dplm-3b": 36,
    "esm3-1.4b": 48,
}

FINE_PARADIGM_ORDER: tuple[str, ...] = (
    "causal_protein",
    "hybrid_ssm",
    "causal_text",
    "causal_joint",
    "causal_adapt",
    "masked_lm",
    "discrete_diffusion",
    "structure_aware_mlm",
    "multimodal_seq_only",
)

MACRO_PARADIGM_ORDER: tuple[str, ...] = (
    "causal_autoregressive_and_ssm",
    "bidirectional_mlm_and_multimodal",
    "discrete_diffusion",
    "structure_vocabulary_mlm",
)

assert set(MODEL_LAYERS_62.keys()) == set(UNIFIED_62_SINGLE_SEQUENCE_ARMS)


def classify_fine_paradigm(arm_id: str) -> str:
    """Return the 9-way fine-grained paradigm label for ``arm_id``."""
    if arm_id == "dayhoff-170m":
        return "hybrid_ssm"
    if arm_id == "esm3-1.4b":
        return "multimodal_seq_only"
    p = ARM_METADATA[arm_id].paradigm
    if p == "causal_adaptation":
        return "causal_adapt"
    return p


def classify_macro_paradigm(arm_id: str) -> str:
    """Return the 4-way macro-paradigm label (40 + 16 + 3 + 3 = 62 exact partition)."""
    fine = classify_fine_paradigm(arm_id)
    if fine in ("causal_protein", "hybrid_ssm", "causal_text", "causal_joint", "causal_adapt"):
        return "causal_autoregressive_and_ssm"
    if fine in ("masked_lm", "multimodal_seq_only"):
        return "bidirectional_mlm_and_multimodal"
    if fine == "discrete_diffusion":
        return "discrete_diffusion"
    if fine == "structure_aware_mlm":
        return "structure_vocabulary_mlm"
    raise ValueError(f"Unmapped fine paradigm {fine!r} for {arm_id}")


def build_layer_provenance_table() -> dict[str, dict[str, Any]]:
    """Record exact hooked layer indices ``(L, (L-1)//2, L-1)`` and pooling rule for all 62 arms."""
    table: dict[str, dict[str, Any]] = {}
    for arm_id in UNIFIED_62_SINGLE_SEQUENCE_ARMS:
        L = int(MODEL_LAYERS_62[arm_id])
        mid_0 = (L - 1) // 2
        fin_0 = L - 1
        meta = ARM_METADATA[arm_id]
        fine = classify_fine_paradigm(arm_id)
        macro = classify_macro_paradigm(arm_id)
        is_causal = macro == "causal_autoregressive_and_ssm"
        table[arm_id] = {
            "arm_id": arm_id,
            "display_name": meta.display_name,
            "family": meta.family,
            "params_m": float(meta.params_m),
            "roster_paradigm": meta.paradigm,
            "fine_paradigm": fine,
            "macro_paradigm": macro,
            "n_layers_L": L,
            "mid_block_0based": mid_0,
            "mid_layer_1based": mid_0 + 1,
            "final_block_0based": fin_0,
            "final_layer_1based": L,
            "r_mid_cols": [0, 512],
            "r_last_cols": [512, 1024],
            "r_joint_cols": [0, 1024],
            "pooling_rule": "mean_last" if is_causal else "mean_cls",
            "projection_dim_per_pool": 256,
        }
    return table


def build_depth_sensitivity_diagnostic(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Summarize adjacent middle-layer sensitivity ``blocks[(L-1)//2]`` vs ``blocks[L//2]``,
    middle vs final on ``anchor201`` (201 ProteinGym assays), and 3-seed JL invariance from
    ``results/R2/readout_depth_panel_20260925/readout_depth_panel.json`` when present.
    """
    rdp_path = root / "results/R2/readout_depth_panel_20260925/readout_depth_panel.json"
    if not rdp_path.is_file():
        return {
            "available": False,
            "note": (
                "Full-stack readout_depth_panel.json absent locally; all 62 arms use isotropic "
                "Gaussian JL projections N(0, 1/sqrt(256)) (PROJECTION_DIM=256, seeds 20260923..20260926)."
            ),
        }

    rdp = json.loads(rdp_path.read_text(encoding="utf-8"))
    per_arm_rows: list[dict[str, Any]] = []
    adj_diffs_r: list[float] = []
    adj_diffs_br: list[float] = []
    mid_minus_last_r: list[float] = []
    mid_minus_last_br: list[float] = []
    seed_spread_r: list[float] = []

    for cell in rdp.get("cells") or []:
        if cell.get("panel") != "anchor201":
            continue
        arm = str(cell["arm"])
        if arm not in ARM_METADATA:
            continue
        L = int(cell["depth"])
        mid_floor = (L - 1) // 2
        mid_upper = min(L - 1, L // 2)
        fin = L - 1

        seeds_map = cell.get("seeds") or {}
        per_seed_r_mid: list[float] = []
        per_seed_r_up: list[float] = []
        per_seed_r_fin: list[float] = []
        per_seed_br_mid: list[float] = []
        per_seed_br_up: list[float] = []
        per_seed_br_fin: list[float] = []

        for s_rec in seeds_map.values():
            axes = s_rec.get("axes") or {}
            depth_ax = axes.get("depth") or {}
            rows = depth_ax.get("profile") or depth_ax.get("rows") or []
            by_d = {int(r["depth"]): r for r in rows if "depth" in r}
            if mid_floor in by_d and mid_upper in by_d and fin in by_d:
                per_seed_r_mid.append(float(by_d[mid_floor]["r_spearman"]))
                per_seed_r_up.append(float(by_d[mid_upper]["r_spearman"]))
                per_seed_r_fin.append(float(by_d[fin]["r_spearman"]))
                per_seed_br_mid.append(float(by_d[mid_floor]["b_r_spearman"]))
                per_seed_br_up.append(float(by_d[mid_upper]["b_r_spearman"]))
                per_seed_br_fin.append(float(by_d[fin]["b_r_spearman"]))

        if not per_seed_r_mid:
            continue
        rm = float(np.mean(per_seed_r_mid))
        ru = float(np.mean(per_seed_r_up))
        rf = float(np.mean(per_seed_r_fin))
        brm = float(np.mean(per_seed_br_mid))
        bru = float(np.mean(per_seed_br_up))
        brf = float(np.mean(per_seed_br_fin))
        sd_spread = float(np.max(per_seed_r_mid) - np.min(per_seed_r_mid)) if len(per_seed_r_mid) >= 2 else 0.0

        adj_diffs_r.append(abs(rm - ru))
        adj_diffs_br.append(abs(brm - bru))
        mid_minus_last_r.append(rm - rf)
        mid_minus_last_br.append(brm - brf)
        seed_spread_r.append(sd_spread)
        per_arm_rows.append(
            {
                "arm_id": arm,
                "n_layers_L": L,
                "mid_floor_0based": mid_floor,
                "mid_upper_0based": mid_upper,
                "final_0based": fin,
                "n_seeds": len(per_seed_r_mid),
                "r_spearman_mid_floor": rm,
                "r_spearman_mid_upper": ru,
                "r_spearman_final": rf,
                "abs_diff_mid_floor_vs_upper_r": abs(rm - ru),
                "mid_minus_final_r": rm - rf,
                "b_r_spearman_mid_floor": brm,
                "b_r_spearman_mid_upper": bru,
                "b_r_spearman_final": brf,
                "abs_diff_mid_floor_vs_upper_br": abs(brm - bru),
                "mid_minus_final_br": brm - brf,
                "seed_range_r_mid": sd_spread,
            }
        )

    per_arm_rows.sort(key=lambda r: r["arm_id"])
    return {
        "available": True,
        "source_path": str(rdp_path.relative_to(root)),
        "source_sha256": sha256_file(rdp_path),
        "panel": "anchor201",
        "n_arms_checked": len(per_arm_rows),
        "adjacent_mid_layer_blocks_floor_vs_upper": {
            "description": "Comparison of blocks[(L-1)//2] vs blocks[L//2] across all pre-registered anchor201 arms.",
            "r_spearman_mean_abs_diff": float(np.mean(adj_diffs_r)) if adj_diffs_r else None,
            "r_spearman_median_abs_diff": float(np.median(adj_diffs_r)) if adj_diffs_r else None,
            "r_spearman_max_abs_diff": float(np.max(adj_diffs_r)) if adj_diffs_r else None,
            "b_r_spearman_mean_abs_diff": float(np.mean(adj_diffs_br)) if adj_diffs_br else None,
            "b_r_spearman_median_abs_diff": float(np.median(adj_diffs_br)) if adj_diffs_br else None,
            "b_r_spearman_max_abs_diff": float(np.max(adj_diffs_br)) if adj_diffs_br else None,
        },
        "anchor201_mid_minus_final_summary": {
            "r_spearman_median_mid_minus_final": float(np.median(mid_minus_last_r)) if mid_minus_last_r else None,
            "r_spearman_n_mid_gt_final": int(sum(1 for v in mid_minus_last_r if v > 0.0)),
            "b_r_spearman_median_mid_minus_final": float(np.median(mid_minus_last_br)) if mid_minus_last_br else None,
            "b_r_spearman_n_mid_gt_final": int(sum(1 for v in mid_minus_last_br if v > 0.0)),
        },
        "projection_seed_invariance": {
            "contract": "Isotropic Gaussian JL projection N(0, 1/sqrt(256)), PROJECTION_DIM=256, seeds 20260923..20260926.",
            "anchor201_3seed_mean_range_r_mid": float(np.mean(seed_spread_r)) if seed_spread_r else None,
            "anchor201_3seed_max_range_r_mid": float(np.max(seed_spread_r)) if seed_spread_r else None,
        },
        "per_arm": per_arm_rows,
    }


# ============================================================================
# Worker Pools: MegaScale, Domainome, and R3 Double-Mutant Epistasis
# ============================================================================

_MS_STATE: dict[str, Any] = {}
_DM_STATE: dict[str, Any] = {}
_R3_STATE: dict[str, Any] = {}


PARITY_GATE_ARMS: tuple[str, ...] = ("progen2-medium", "esm2-650m", "dplm-650m", "saprot-650m")


def fit_designs_group_suffstats(
    target: np.ndarray,
    groups: np.ndarray,
    blocks: dict[str, np.ndarray],
    designs: dict[str, tuple[str, ...]],
    fold_cache: dict[int, list[dict[str, Any]]],
    *,
    alphas: tuple[float, ...] = r2_mod.ALPHAS_5,
    renorm_val_weights: bool = False,
    tie_largest_alpha: bool = False,
) -> dict[int, dict[str, np.ndarray]]:
    """Exact float64 nested held-group Ridge via group-level centered sufficient statistics.

    Algebraically identical to ``r2_mod.fit_designs_with_cache`` (``< 1e-14`` float64 parity),
    reducing ``O(15 * N * p^2)`` outer-fold Gram passes and ``O(75 * N_val * p * A)`` validation
    matmuls to a single ``O(N * p^2)`` pass over the static blocks plus exact ``O(p^2 * A)``
    quadratic validation MSE evaluation. Supports both static designs and fold-varying ``G``.
    """
    y_np = np.asarray(target, dtype=np.float64)
    n_rows = len(y_np)
    needed: list[str] = []
    for cols in designs.values():
        for c in cols:
            if c not in needed:
                needed.append(c)

    has_g = any(f["G"] is not None for sf in fold_cache.values() for f in sf)
    use_g = "G" in needed and has_g
    static_keys = [k for k in needed if k != "G"]

    static_arrays: list[np.ndarray] = []
    block_slices: dict[str, tuple[int, int]] = {}
    off = 0
    for k in static_keys:
        a = np.asarray(blocks[k], dtype=np.float64)
        if a.ndim == 1:
            a = a[:, None]
        if not np.isfinite(a).all():
            raise ValueError(f"Block {k}: non-finite values")
        static_arrays.append(a)
        block_slices[k] = (off, off + a.shape[1])
        off += a.shape[1]
    p_static = off

    first_fold = next(iter(fold_cache.values()))[0]
    w_t = torch.as_tensor(first_fold["weights"], dtype=torch.float64)
    y_t = torch.as_tensor(y_np, dtype=torch.float64)
    w_all = w_t.sum()
    y_g0 = (w_t * y_t).sum() / w_all
    yc_all = y_t - y_g0

    if p_static > 0:
        x_static = torch.as_tensor(np.concatenate(static_arrays, axis=1), dtype=torch.float64)
        mu_g0 = (w_t[:, None] * x_static).sum(0) / w_all
        xc_static = x_static - mu_g0
    else:
        xc_static = torch.empty((n_rows, 0), dtype=torch.float64)

    uniq_groups, inv = np.unique(groups, return_inverse=True)
    n_g = len(uniq_groups)
    g_counts = torch.empty(n_g, dtype=torch.long)
    g_w = torch.empty(n_g, dtype=torch.float64)
    g_mu = torch.empty((n_g, p_static), dtype=torch.float64)
    g_var = torch.empty((n_g, p_static), dtype=torch.float64)
    g_wx = torch.empty((n_g, p_static), dtype=torch.float64)
    g_wxx = torch.empty((n_g, p_static, p_static), dtype=torch.float64)
    g_wy = torch.empty(n_g, dtype=torch.float64)
    g_wxy = torch.empty((n_g, p_static), dtype=torch.float64)
    g_wyy = torch.empty(n_g, dtype=torch.float64)

    for gi in range(n_g):
        idx = torch.as_tensor(np.flatnonzero(inv == gi), dtype=torch.long)
        g_counts[gi] = int(idx.numel())
        xg = xc_static[idx]
        yg = yc_all[idx]
        wg = w_t[idx]
        wgs = wg.sum()
        g_w[gi] = wgs
        g_wy[gi] = (wg * yg).sum()
        g_wyy[gi] = (wg * yg.square()).sum()
        if p_static > 0:
            w_xg = wg[:, None] * xg
            wx_sum = w_xg.sum(0)
            mu_g = wx_sum / wgs
            g_mu[gi] = mu_g
            g_var[gi] = (wg[:, None] * (xg - mu_g).square()).sum(0) / wgs
            g_wx[gi] = wx_sum
            g_wxx[gi] = xg.T @ w_xg
            g_wxy[gi] = w_xg.T @ yg

    oof_by_seed: dict[int, dict[str, np.ndarray]] = {}
    for seed in r2_mod.SPLIT_SEEDS:
        preds = {name: np.full(n_rows, np.nan, dtype=np.float64) for name in designs}
        for fold in fold_cache[int(seed)]:
            test = fold["test"]
            n_train_groups = fold["n_train_groups"]
            te_t = torch.as_tensor(test, dtype=torch.long)

            fold_slices = dict(block_slices)
            gc_t: torch.Tensor | None = None
            p_g = 0
            if use_g:
                g_arr = np.asarray(fold["G"], dtype=np.float64)
                if g_arr.ndim == 1:
                    g_arr = g_arr[:, None]
                if not np.isfinite(g_arr).all():
                    raise ValueError("Block G: non-finite values")
                p_g = int(g_arr.shape[1])
                g_t = torch.as_tensor(g_arr, dtype=torch.float64)
                mu_g_mat = (w_t[:, None] * g_t).sum(0) / w_all
                gc_t = g_t - mu_g_mat
                fold_slices["G"] = (p_static, p_static + p_g)
            p_tot = p_static + p_g

            chunk_stats: list[
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    int,
                ]
            ] = []
            total_val_rows = 0
            for _, val_idx, n_val in fold["inner_splits"]:
                total_val_rows += len(val_idx)
                vg_ids = np.unique(inv[val_idx])
                vg_t = torch.as_tensor(vg_ids, dtype=torch.long)
                if int(g_counts[vg_t].sum().item()) != len(val_idx):
                    raise ValueError("Inner split does not align with complete family groups")
                wv_g = g_w[vg_t]
                wv_sum = wv_g.sum()
                wy_v = g_wy[vg_t].sum()
                wyy_v = g_wyy[vg_t].sum()
                if p_static > 0:
                    mu_s = (wv_g[:, None] * g_mu[vg_t]).sum(0) / wv_sum
                    var_s = (wv_g[:, None] * (g_var[vg_t] + (g_mu[vg_t] - mu_s).square())).sum(0) / wv_sum
                    wx_s = g_wx[vg_t].sum(0)
                    wxx_s = g_wxx[vg_t].sum(0)
                    wxy_s = g_wxy[vg_t].sum(0)
                else:
                    mu_s = torch.empty(0, dtype=torch.float64)
                    var_s = torch.empty(0, dtype=torch.float64)
                    wx_s = torch.empty(0, dtype=torch.float64)
                    wxx_s = torch.empty((0, 0), dtype=torch.float64)
                    wxy_s = torch.empty(0, dtype=torch.float64)

                if use_g and gc_t is not None:
                    v_t = torch.as_tensor(val_idx, dtype=torch.long)
                    wv = w_t[v_t]
                    gv = gc_t[v_t]
                    w_gv = wv[:, None] * gv
                    wx_g = w_gv.sum(0)
                    mu_g = wx_g / wv_sum
                    var_g = (wv[:, None] * (gv - mu_g).square()).sum(0) / wv_sum
                    wgg = gv.T @ w_gv
                    wgy = w_gv.T @ yc_all[v_t]
                    mu_v = torch.cat([mu_s, mu_g], dim=0)
                    var_v = torch.cat([var_s, var_g], dim=0)
                    wx_v = torch.cat([wx_s, wx_g], dim=0)
                    wxy_v = torch.cat([wxy_s, wgy], dim=0)
                    wxx_v = torch.empty((p_tot, p_tot), dtype=torch.float64)
                    if p_static > 0:
                        wxg = xc_static[v_t].T @ w_gv
                        wxx_v[:p_static, :p_static] = wxx_s
                        wxx_v[:p_static, p_static:] = wxg
                        wxx_v[p_static:, :p_static] = wxg.T
                    wxx_v[p_static:, p_static:] = wgg
                else:
                    mu_v, var_v, wx_v, wxx_v, wxy_v = mu_s, var_s, wx_s, wxx_s, wxy_s

                chunk_stats.append((wv_sum, mu_v, var_v, wx_v, wxx_v, wy_v, wxy_v, wyy_v, int(n_val)))

            if total_val_rows != len(fold["train"]):
                raise ValueError("Inner splits do not partition outer training rows")

            n_c = len(chunk_stats)
            w_tot = torch.stack([c[0] for c in chunk_stats]).sum(0)
            mu_tot = torch.stack([c[0] * c[1] for c in chunk_stats]).sum(0) / w_tot
            var_tot = torch.stack([c[0] * (c[2] + (c[1] - mu_tot).square()) for c in chunk_stats]).sum(0) / w_tot
            wxx_tot = torch.stack([c[4] for c in chunk_stats]).sum(0)
            wy_tot = torch.stack([c[5] for c in chunk_stats]).sum(0)
            wxy_tot = torch.stack([c[6] for c in chunk_stats]).sum(0)

            inner_fit: list[
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ] = []
            for k in range(n_c):
                others = [j for j in range(n_c) if j != k]
                w_f = torch.stack([chunk_stats[j][0] for j in others]).sum(0)
                mu_f = torch.stack([chunk_stats[j][0] * chunk_stats[j][1] for j in others]).sum(0) / w_f
                var_f = (
                    torch.stack([chunk_stats[j][0] * (chunk_stats[j][2] + (chunk_stats[j][1] - mu_f).square()) for j in others]).sum(0)
                    / w_f
                )
                wxx_f = torch.stack([chunk_stats[j][4] for j in others]).sum(0)
                wy_f = torch.stack([chunk_stats[j][5] for j in others]).sum(0)
                wxy_f = torch.stack([chunk_stats[j][6] for j in others]).sum(0)
                inner_fit.append((w_f, mu_f, var_f, wxx_f, wy_f, wxy_f))

            xc_test = torch.cat([xc_static[te_t], gc_t[te_t]], dim=1) if (use_g and gc_t is not None) else xc_static[te_t]

            for name, cols in designs.items():
                col_indices = [i for c in cols for i in range(*fold_slices[c])]
                c_idx = torch.as_tensor(col_indices, dtype=torch.long)

                losses = np.zeros(len(alphas), dtype=np.float64)
                for k, (wv_sum, _, _, wx_v, wxx_v, wy_v, wxy_v, wyy_v, n_val) in enumerate(chunk_stats):
                    w_f, mu_f, var_f, wxx_f, wy_f, wxy_f = inner_fit[k]
                    beta, mean_c, inv_scale, ym_c = r2_mod._solve_ridge_from_gram(
                        w_f,
                        mu_f[c_idx],
                        var_f[c_idx],
                        wxx_f.index_select(0, c_idx).index_select(1, c_idx),
                        wy_f,
                        wxy_f[c_idx],
                        alphas,
                    )
                    b = beta * inv_scale[:, None]
                    c0 = ym_c - (mean_c @ b)
                    wxx_vk = wxx_v.index_select(0, c_idx).index_select(1, c_idx)
                    wx_vk = wx_v[c_idx]
                    wxy_vk = wxy_v[c_idx]
                    q = (b * (wxx_vk @ b)).sum(0)
                    sse = q + 2.0 * c0 * (wx_vk @ b) - 2.0 * (wxy_vk @ b) + (c0 * c0) * wv_sum - 2.0 * c0 * wy_v + wyy_v
                    scale_v = (1.0 / wv_sum) if renorm_val_weights else 1.0
                    losses += n_val * (sse * scale_v).numpy()

                alpha = alphas[r2_mod._select_alpha_index(losses / n_train_groups, tie_largest=tie_largest_alpha)]
                beta_te, mean_te, inv_scale_te, ym_te = r2_mod._solve_ridge_from_gram(
                    w_tot,
                    mu_tot[c_idx],
                    var_tot[c_idx],
                    wxx_tot.index_select(0, c_idx).index_select(1, c_idx),
                    wy_tot,
                    wxy_tot[c_idx],
                    [alpha],
                )
                pred_te = ((xc_test[:, c_idx] - mean_te) * inv_scale_te) @ beta_te + (y_g0 + ym_te)
                preds[name][test] = pred_te[:, 0].numpy()

        for name, arr in preds.items():
            if not np.isfinite(arr).all():
                raise ValueError(f"Seed {seed} {name}: incomplete held-out predictions")
        oof_by_seed[int(seed)] = preds
    return oof_by_seed


_DM_G_BUILD_PANEL: dict[str, Any] = {}


def _init_dm_g_worker(panel_94: dict[str, Any]) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    _DM_G_BUILD_PANEL.clear()
    _DM_G_BUILD_PANEL.update(panel_94)


def _compute_one_dm_g_fold(task: tuple[int, int, list[str], list[list[str]]]) -> tuple[int, int, np.ndarray]:
    seed, outer, held_list, inner_partition = task
    g_arr = abundance_mod.nuisance_response(
        _DM_G_BUILD_PANEL,
        r2_mod.DM_FIRST_STAGE,
        _DM_G_BUILD_PANEL["blocks"],
        np.asarray(held_list),
        inner_partition,
        device="cpu",
    )[0]
    return int(seed), int(outer), np.asarray(g_arr, dtype=np.float64)


def _build_or_load_dm_fc_94_full(
    dm: dict[str, Any],
    cache_path: Path,
    *,
    max_workers: int = 15,
) -> dict[int, list[dict[str, Any]]]:
    """Build (in parallel across 15 folds) or load cached Domainome 94g full-nuisance fold cache."""
    panel_94 = dm["panel_94"]
    h = hashlib.sha256()
    h.update(np.asarray(panel_94["target"], dtype=np.float64).tobytes())
    h.update(np.asarray(panel_94["weights"], dtype=np.float64).tobytes())
    h.update(",".join(dm["groups_94"]).encode("utf-8"))
    sig = h.hexdigest()

    fc = r2_mod.build_fold_cache(groups=panel_94["group"], weights=panel_94["weights"])
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as z:
                if str(z["sig"].item()) == sig:
                    for seed, folds in fc.items():
                        for fold in folds:
                            fold["G"] = np.asarray(z[f"G_{seed}_{fold['outer']}"], dtype=np.float64)
                    return fc
        except Exception:
            pass

    tasks: list[tuple[int, int, list[str], list[list[str]]]] = []
    for seed in r2_mod.SPLIT_SEEDS:
        for outer, held in enumerate(r2_mod.family_folds(panel_94["group"], r2_mod.OUTER_SPLITS, seed)):
            held_list = list(held)
            train = np.flatnonzero(~np.isin(panel_94["group"], held_list))
            inner_partition = [
                list(vg)
                for vg in r2_mod.family_folds(panel_94["group"][train], r2_mod.INNER_SPLITS, seed + 100 + outer)
            ]
            tasks.append((int(seed), int(outer), held_list, inner_partition))

    g_map: dict[tuple[int, int], np.ndarray] = {}
    gw = max(1, min(int(max_workers), len(tasks)))
    if gw == 1:
        _init_dm_g_worker(panel_94)
        for t in tasks:
            s, o, arr = _compute_one_dm_g_fold(t)
            g_map[(s, o)] = arr
    else:
        with ProcessPoolExecutor(
            max_workers=gw,
            mp_context=mp.get_context("spawn"),
            initializer=_init_dm_g_worker,
            initargs=(panel_94,),
        ) as pool:
            for s, o, arr in pool.map(_compute_one_dm_g_fold, tasks):
                g_map[(s, o)] = arr

    save_dict: dict[str, Any] = {"sig": np.asarray(sig)}
    for seed, folds in fc.items():
        for fold in folds:
            arr = g_map[(int(seed), int(fold["outer"]))]
            fold["G"] = arr
            save_dict[f"G_{seed}_{fold['outer']}"] = arr
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = cache_path.with_suffix(".tmp.npz")
    np.savez(tmp_npz, **save_dict)
    tmp_npz.replace(cache_path)
    return fc


def _init_ms_worker(shared: dict[str, Any]) -> None:
    bt = int(shared.get("blas_threads", 2))
    os.environ["OMP_NUM_THREADS"] = str(bt)
    os.environ["OPENBLAS_NUM_THREADS"] = str(bt)
    os.environ["MKL_NUM_THREADS"] = str(bt)
    _MS_STATE.clear()
    _MS_STATE.update(shared)
    torch.set_num_threads(bt)


def _load_r2_sealed_cell(root: Path, dataset: str, arm_id: str, m_sha: str, r_sha: str) -> dict[str, Any] | None:
    p = root / UNIFIED_OUTPUT_DIR / "cells" / "r2_exp21" / dataset / f"{arm_id}.json"
    if not p.is_file():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    if (
        doc.get("arm_id") == arm_id
        and doc.get("has_R") is True
        and doc.get("m_source_sha256") == m_sha
        and doc.get("r_source_sha256") == r_sha
    ):
        return doc
    return None


def _fit_one_ms_mid_last(arm_id: str) -> dict[str, Any]:
    root = Path(_MS_STATE["root"])
    cache_dir = Path(_MS_STATE["cache_dir"])
    r_archive_dir = Path(_MS_STATE["r_archive_dir"])
    force = bool(_MS_STATE.get("force", False))
    include_live_joint = arm_id in set(_MS_STATE.get("live_joint_arms") or ())
    cache_path = cache_dir / f"{arm_id}.json"

    m_101, r_101, meta = r2_mod.load_megascale_arm(
        arm_id, _MS_STATE["cohort_101"], root=root, r_archive_dir=r_archive_dir
    )
    if r_101 is None or r_101.shape[1] != 1024:
        raise RuntimeError(f"{arm_id}: missing 1024-d MegaScale R archive")

    if not force and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("schema") == CELL_CACHE_SCHEMA
                and cached.get("dataset") == "megascale"
                and cached.get("arm_id") == arm_id
                and cached.get("m_source_sha256") == meta["m_source_sha256"]
                and cached.get("r_source_sha256") == meta["r_source_sha256"]
                and (not include_live_joint or cached.get("live_joint_fitted") is True)
            ):
                return cached
        except Exception:
            pass

    t0 = time.monotonic()
    sealed_cell = _load_r2_sealed_cell(root, "megascale", arm_id, meta["m_source_sha256"], meta["r_source_sha256"])
    need_joint = include_live_joint or (sealed_cell is None)

    mask_79 = _MS_STATE["mask_79"]
    panel_101 = _MS_STATE["panel_101"]
    groups_101 = _MS_STATE["groups_101"]
    panel_79 = _MS_STATE["panel_79"]
    groups_79 = _MS_STATE["groups_79"]
    m_79 = m_101[mask_79]
    r_79 = r_101[mask_79]

    def _eval_79(p: np.ndarray) -> dict[str, np.ndarray]:
        return r2_mod.evaluate_megascale_oof(p, panel_79["target"], panel_79["group"], panel_79["site"], groups_79)

    def _eval_101(p: np.ndarray) -> dict[str, np.ndarray]:
        return r2_mod.evaluate_megascale_oof(
            p, panel_101["target"], panel_101["group"], panel_101["site"], groups_101
        )

    def _make_designs(ctrl: tuple[str, ...], *, fit_m: bool, fit_joint: bool) -> dict[str, tuple[str, ...]]:
        d: dict[str, tuple[str, ...]] = {
            "R_mid": ("R_mid",),
            "R_last": ("R_last",),
            "C_R_mid": (*ctrl, "R_mid"),
            "C_R_last": (*ctrl, "R_last"),
            "C_M_R_mid": (*ctrl, "M", "R_mid"),
            "C_M_R_last": (*ctrl, "M", "R_last"),
        }
        if fit_m:
            d["M"] = ("M",)
            d["C_M"] = (*ctrl, "M")
        if fit_joint:
            d["R_joint"] = ("R_mid", "R_last")
            d["C_R_joint"] = (*ctrl, "R_mid", "R_last")
            d["C_M_R_joint"] = (*ctrl, "M", "R_mid", "R_last")
        return d

    def _merge_sealed(
        fitted: dict[str, dict[str, np.ndarray]],
        tier_key: str,
        key_map: dict[str, str],
        pre33_fallback: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> dict[str, dict[str, list[float]]]:
        out: dict[str, dict[str, list[float]]] = {}
        for ep in r2_mod.ENDPOINTS:
            row = {d: v.tolist() for d, v in fitted[ep].items()}
            if sealed_cell is not None:
                src_ep = sealed_cell[tier_key][ep]
                for our_k, seal_k in key_map.items():
                    if our_k not in row and seal_k in src_ep:
                        row[our_k] = [float(x) for x in src_ep[seal_k]]
            if pre33_fallback is not None:
                for our_k, vec in pre33_fallback[ep].items():
                    if our_k not in row:
                        row[our_k] = [float(x) for x in vec]
            out[ep] = row
        return out

    # Pre-registered 33 M & C_M fallbacks for full_79 / full_101 (sealed cells only stored R/CR/CMR for pre-33)
    pre33_full79_fb = _MS_STATE["pre33_full79_m"].get(arm_id)
    pre33_full101_fb = _MS_STATE["pre33_full101_m"].get(arm_id)

    # 1. MegaScale 79g C_evo (ALPHAS_17)
    blocks_79_evo = {
        "P": _MS_STATE["p_logodds_79"][:, None],
        "POTTS": _MS_STATE["potts_raw_79"][:, None],
        "GEMME": _MS_STATE["gemme_raw_79"][:, None],
        "M": m_79[:, None],
        "R_mid": r_79[:, :512],
        "R_last": r_79[:, 512:],
    }
    evo_79_fit = r2_mod.collapse_oof(
        fit_designs_group_suffstats(
            panel_79["target"], panel_79["group"], blocks_79_evo,
            _make_designs(("P", "POTTS", "GEMME"), fit_m=sealed_cell is None, fit_joint=need_joint),
            _MS_STATE["fc_79_evo"], alphas=r2_mod.ALPHAS_17, renorm_val_weights=True, tie_largest_alpha=True,
        ),
        _eval_79,
    )
    evo_79 = _merge_sealed(
        evo_79_fit, "evo_79", {"M": "M", "C_M": "C_M", "R_joint": "R", "C_R_joint": "C_R", "C_M_R_joint": "C_M_R"}
    )

    # 2. MegaScale 101g C_evo (ALPHAS_17)
    blocks_101_evo = {
        "P": _MS_STATE["p_logodds_101"][:, None],
        "M": m_101[:, None],
        "R_mid": r_101[:, :512],
        "R_last": r_101[:, 512:],
    }
    evo_101_fit = r2_mod.collapse_oof(
        fit_designs_group_suffstats(
            panel_101["target"], panel_101["group"], blocks_101_evo,
            _make_designs(("P",), fit_m=sealed_cell is None, fit_joint=need_joint),
            _MS_STATE["fc_101_evo"], alphas=r2_mod.ALPHAS_17, renorm_val_weights=True, tie_largest_alpha=True,
        ),
        _eval_101,
    )
    evo_101 = _merge_sealed(
        evo_101_fit, "evo_101", {"M": "M", "C_M": "C_M", "R_joint": "R", "C_R_joint": "C_R", "C_M_R_joint": "C_M_R"}
    )

    # 3. MegaScale 79g C_full (ALPHAS_5 + cached G)
    ctrl_79_full = r2_mod.MS_PROGRESSION_RUNGS["nuisance+profile+couplings+gemme"]
    blocks_79_full = {
        **panel_79["blocks"], "M": m_79[:, None], "R_mid": r_79[:, :512], "R_last": r_79[:, 512:]
    }
    need_m_f79 = sealed_cell is None or (pre33_full79_fb is None and "M" not in sealed_cell["full_79"]["transfer_error"])
    full_79_fit = r2_mod.collapse_oof(
        fit_designs_group_suffstats(
            panel_79["target"], panel_79["group"], blocks_79_full,
            _make_designs(ctrl_79_full, fit_m=need_m_f79, fit_joint=need_joint),
            _MS_STATE["fc_79_full"], alphas=r2_mod.ALPHAS_5,
        ),
        _eval_79,
    )
    full_79 = _merge_sealed(
        full_79_fit,
        "full_79",
        {
            "M": "M",
            "C_M": "nuisance+profile+couplings+gemme|M",
            "R_joint": "R",
            "C_R_joint": "nuisance+profile+couplings+gemme|R",
            "C_M_R_joint": "nuisance+profile+couplings+gemme|MR",
        },
        pre33_fallback=pre33_full79_fb,
    )

    # 4. MegaScale 101g S2 full nuisance (ALPHAS_5 + cached G)
    ctrl_101_full = r2_mod.MS_DESIGN_S2
    blocks_101_full = {
        **panel_101["blocks"], "M": m_101[:, None], "R_mid": r_101[:, :512], "R_last": r_101[:, 512:]
    }
    has_sealed_m101 = sealed_cell is not None and "M" in sealed_cell["full_101"]["transfer_error"]
    need_cm_f101 = sealed_cell is None and pre33_full101_fb is None
    designs_101_full = _make_designs(ctrl_101_full, fit_m=need_cm_f101, fit_joint=need_joint)
    if not has_sealed_m101:
        designs_101_full["M"] = ("M",)
    full_101_fit = r2_mod.collapse_oof(
        fit_designs_group_suffstats(
            panel_101["target"], panel_101["group"], blocks_101_full,
            designs_101_full,
            _MS_STATE["fc_101_full"], alphas=r2_mod.ALPHAS_5,
        ),
        _eval_101,
    )
    full_101 = _merge_sealed(
        full_101_fit,
        "full_101",
        {"M": "M", "C_M": "S2_M", "R_joint": "R", "C_R_joint": "S2_R", "C_M_R_joint": "S2_MR"},
        pre33_fallback=pre33_full101_fb,
    )

    rec = {
        "schema": CELL_CACHE_SCHEMA,
        "dataset": "megascale",
        "arm_id": arm_id,
        "m_source_sha256": meta["m_source_sha256"],
        "r_source_sha256": meta["r_source_sha256"],
        "live_joint_fitted": need_joint,
        "elapsed_seconds": float(time.monotonic() - t0),
        "evo_79": evo_79,
        "evo_101": evo_101,
        "full_79": full_79,
        "full_101": full_101,
    }
    atomic_write_json(cache_path, rec)
    return rec


def _init_dm_worker(shared: dict[str, Any]) -> None:
    bt = int(shared.get("blas_threads", 2))
    os.environ["OMP_NUM_THREADS"] = str(bt)
    os.environ["OPENBLAS_NUM_THREADS"] = str(bt)
    os.environ["MKL_NUM_THREADS"] = str(bt)
    _DM_STATE.clear()
    _DM_STATE.update(shared)
    torch.set_num_threads(bt)


def _fit_one_dm_mid_last(arm_id: str) -> dict[str, Any]:
    root = Path(_DM_STATE["root"])
    cache_dir = Path(_DM_STATE["cache_dir"])
    r_archive_dir = Path(_DM_STATE["r_archive_dir"])
    force = bool(_DM_STATE.get("force", False))
    include_live_joint = arm_id in set(_DM_STATE.get("live_joint_arms") or ())
    cache_path = cache_dir / f"{arm_id}.json"

    m_96, r_96, meta = r2_mod.load_domainome_arm(
        arm_id, _DM_STATE["cohort_96"], root=root, r_archive_dir=r_archive_dir
    )
    if r_96 is None or r_96.shape[1] != 1024:
        raise RuntimeError(f"{arm_id}: missing 1024-d Domainome R archive")

    if not force and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("schema") == CELL_CACHE_SCHEMA
                and cached.get("dataset") == "domainome"
                and cached.get("arm_id") == arm_id
                and cached.get("m_source_sha256") == meta["m_source_sha256"]
                and cached.get("r_source_sha256") == meta["r_source_sha256"]
                and (not include_live_joint or cached.get("live_joint_fitted") is True)
            ):
                return cached
        except Exception:
            pass

    t0 = time.monotonic()
    sealed_cell = _load_r2_sealed_cell(root, "domainome", arm_id, meta["m_source_sha256"], meta["r_source_sha256"])
    need_joint = include_live_joint or (sealed_cell is None)

    mask_94 = _DM_STATE["mask_94"]
    panel_96 = _DM_STATE["panel_96"]
    groups_96 = _DM_STATE["groups_96"]
    panel_94 = _DM_STATE["panel_94"]
    groups_94 = _DM_STATE["groups_94"]

    m_94 = m_96[mask_94]
    r_94 = r_96[mask_94]

    def _eval_94_evo(p: np.ndarray) -> dict[str, np.ndarray]:
        return r2_mod.evaluate_domainome_oof(
            p, panel_94["target"], panel_94["group"], panel_94["domain"], panel_94["site"],
            groups_94, use_nested_weights=False,
        )

    def _eval_96_evo(p: np.ndarray) -> dict[str, np.ndarray]:
        return r2_mod.evaluate_domainome_oof(
            p, panel_96["target"], panel_96["group"], panel_96["domain"], panel_96["site"],
            groups_96, use_nested_weights=False,
        )

    def _eval_94_full(p: np.ndarray) -> dict[str, np.ndarray]:
        return r2_mod.evaluate_domainome_oof(
            p, panel_94["target"], panel_94["group"], panel_94["domain"], panel_94["site"],
            groups_94, use_nested_weights=True,
        )

    def _make_designs(ctrl: tuple[str, ...], *, fit_m: bool, fit_joint: bool) -> dict[str, tuple[str, ...]]:
        d: dict[str, tuple[str, ...]] = {
            "R_mid": ("R_mid",),
            "R_last": ("R_last",),
            "C_R_mid": (*ctrl, "R_mid"),
            "C_R_last": (*ctrl, "R_last"),
            "C_M_R_mid": (*ctrl, "M", "R_mid"),
            "C_M_R_last": (*ctrl, "M", "R_last"),
        }
        if fit_m:
            d["M"] = ("M",)
            d["C_M"] = (*ctrl, "M")
        if fit_joint:
            d["R_joint"] = ("R_mid", "R_last")
            d["C_R_joint"] = (*ctrl, "R_mid", "R_last")
            d["C_M_R_joint"] = (*ctrl, "M", "R_mid", "R_last")
        return d

    def _merge_sealed(
        fitted: dict[str, dict[str, np.ndarray]],
        tier_key: str,
        key_map: dict[str, str],
    ) -> dict[str, dict[str, list[float]]]:
        out: dict[str, dict[str, list[float]]] = {}
        for ep in r2_mod.ENDPOINTS:
            row = {d: v.tolist() for d, v in fitted[ep].items()}
            if sealed_cell is not None:
                src_ep = sealed_cell[tier_key][ep]
                for our_k, seal_k in key_map.items():
                    if our_k not in row and seal_k in src_ep:
                        row[our_k] = [float(x) for x in src_ep[seal_k]]
            out[ep] = row
        return out

    def _fit_tier_split_or_joint(
        target: np.ndarray,
        groups_arr: np.ndarray,
        base_blocks: dict[str, np.ndarray],
        r_mat: np.ndarray,
        ctrl: tuple[str, ...],
        fc: dict[int, list[dict[str, Any]]],
        eval_fn: Any,
        *,
        alphas: tuple[float, ...],
        renorm: bool = False,
        tie_largest: bool = False,
    ) -> dict[str, dict[str, np.ndarray]]:
        if need_joint:
            blocks = {**base_blocks, "R_mid": r_mat[:, :512], "R_last": r_mat[:, 512:]}
            return r2_mod.collapse_oof(
                fit_designs_group_suffstats(
                    target,
                    groups_arr,
                    blocks,
                    _make_designs(ctrl, fit_m=sealed_cell is None, fit_joint=True),
                    fc,
                    alphas=alphas,
                    renorm_val_weights=renorm,
                    tie_largest_alpha=tie_largest,
                ),
                eval_fn,
            )
        # Split R_mid (512d) and R_last (512d) into separate 512-col group-suffstats passes.
        d_mid: dict[str, tuple[str, ...]] = {
            "R_mid": ("R_mid",),
            "C_R_mid": (*ctrl, "R_mid"),
            "C_M_R_mid": (*ctrl, "M", "R_mid"),
        }
        if sealed_cell is None:
            d_mid["M"] = ("M",)
            d_mid["C_M"] = (*ctrl, "M")
        d_last: dict[str, tuple[str, ...]] = {
            "R_last": ("R_last",),
            "C_R_last": (*ctrl, "R_last"),
            "C_M_R_last": (*ctrl, "M", "R_last"),
        }
        res_mid = r2_mod.collapse_oof(
            fit_designs_group_suffstats(
                target,
                groups_arr,
                {**base_blocks, "R_mid": np.ascontiguousarray(r_mat[:, :512])},
                d_mid,
                fc,
                alphas=alphas,
                renorm_val_weights=renorm,
                tie_largest_alpha=tie_largest,
            ),
            eval_fn,
        )
        res_last = r2_mod.collapse_oof(
            fit_designs_group_suffstats(
                target,
                groups_arr,
                {**base_blocks, "R_last": np.ascontiguousarray(r_mat[:, 512:])},
                d_last,
                fc,
                alphas=alphas,
                renorm_val_weights=renorm,
                tie_largest_alpha=tie_largest,
            ),
            eval_fn,
        )
        merged: dict[str, dict[str, np.ndarray]] = {}
        for ep in r2_mod.ENDPOINTS:
            merged[ep] = {**res_mid[ep], **res_last[ep]}
        gc.collect()
        return merged

    # 1. Domainome 94g C_evo (ALPHAS_17)
    evo_94_fit = _fit_tier_split_or_joint(
        panel_94["target"],
        panel_94["group"],
        {
            "P": _DM_STATE["p_logodds_94"][:, None],
            "POTTS": _DM_STATE["potts_raw_94"][:, None],
            "GEMME": _DM_STATE["gemme_raw_94"][:, None],
            "M": m_94[:, None],
        },
        r_94,
        ("P", "POTTS", "GEMME"),
        _DM_STATE["fc_94_evo"],
        _eval_94_evo,
        alphas=r2_mod.ALPHAS_17,
        renorm=True,
        tie_largest=True,
    )
    evo_94 = _merge_sealed(
        evo_94_fit, "evo_94", {"M": "M", "C_M": "C_M", "R_joint": "R", "C_R_joint": "C_R", "C_M_R_joint": "C_M_R"}
    )

    # 2. Domainome 96g C_evo (ALPHAS_17)
    evo_96_fit = _fit_tier_split_or_joint(
        panel_96["target"],
        panel_96["group"],
        {"P": _DM_STATE["p_logodds_96"][:, None], "M": m_96[:, None]},
        r_96,
        ("P",),
        _DM_STATE["fc_96_evo"],
        _eval_96_evo,
        alphas=r2_mod.ALPHAS_17,
        renorm=True,
        tie_largest=True,
    )
    evo_96 = _merge_sealed(
        evo_96_fit, "evo_96", {"M": "M", "C_M": "C_M", "R_joint": "R", "C_R_joint": "C_R", "C_M_R_joint": "C_M_R"}
    )

    # 3. Domainome 94g C_full (ALPHAS_5 + cached G + nested_weights)
    ctrl_94_full = r2_mod.DM_PROGRESSION_RUNGS["nuisance+profile+couplings+gemme"]
    full_94_fit = _fit_tier_split_or_joint(
        panel_94["target"],
        panel_94["group"],
        {**panel_94["blocks"], "M": m_94[:, None]},
        r_94,
        ctrl_94_full,
        _DM_STATE["fc_94_full"],
        _eval_94_full,
        alphas=r2_mod.ALPHAS_5,
    )
    full_94 = _merge_sealed(
        full_94_fit,
        "full_94",
        {
            "M": "M",
            "C_M": "nuisance+profile+couplings+gemme|M",
            "R_joint": "R",
            "C_R_joint": "nuisance+profile+couplings+gemme|R",
            "C_M_R_joint": "nuisance+profile+couplings+gemme|MR",
        },
    )

    rec = {
        "schema": CELL_CACHE_SCHEMA,
        "dataset": "domainome",
        "arm_id": arm_id,
        "m_source_sha256": meta["m_source_sha256"],
        "r_source_sha256": meta["r_source_sha256"],
        "live_joint_fitted": need_joint,
        "elapsed_seconds": float(time.monotonic() - t0),
        "evo_94": evo_94,
        "evo_96": evo_96,
        "full_94": full_94,
    }
    atomic_write_json(cache_path, rec)
    return rec


def _init_r3_worker(
    root_str: str, cache_dir_str: str, blas_threads: int, force: bool, live_joint_arms: tuple[str, ...]
) -> None:
    bt = int(blas_threads)
    os.environ["OMP_NUM_THREADS"] = str(bt)
    os.environ["OPENBLAS_NUM_THREADS"] = str(bt)
    os.environ["MKL_NUM_THREADS"] = str(bt)
    torch.set_num_threads(bt)
    root = Path(root_str)
    ctx = load_pairwise_cycle_context(root)
    yfit = r3_mod._load_yfit_map(ctx)
    aa_caches = r3_mod._r3_build_rung_caches(ctx, ctx.panel["blocks"], rungs=r3_mod.R3_RUNGS)
    _R3_STATE.clear()
    _R3_STATE.update(
        {
            "root": root,
            "ctx": ctx,
            "yfit": yfit,
            "aa_caches": aa_caches,
            "cache_dir": Path(cache_dir_str),
            "force": bool(force),
            "live_joint_arms": set(live_joint_arms),
        }
    )


def _fit_one_r3_mid_last(arm_id: str) -> dict[str, Any]:
    ctx = _R3_STATE["ctx"]
    root = Path(_R3_STATE["root"])
    include_live_joint = arm_id in _R3_STATE["live_joint_arms"]
    cache_path = Path(_R3_STATE["cache_dir"]) / f"{arm_id}.json"
    if not _R3_STATE["force"] and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("schema") == CELL_CACHE_SCHEMA
                and cached.get("dataset") == "r3_epistasis"
                and cached.get("arm_id") == arm_id
                and cached.get("support_sha256") == ctx.support_sha256
                and (not include_live_joint or cached.get("live_joint_fitted") is True)
            ):
                return cached
        except Exception:
            pass

    t0 = time.monotonic()
    # Reuse verified sealed R3 cell for M_cycle & joint R_cycle/MR_cycle when present and not a live-joint gate arm
    sealed_r3_path = root / UNIFIED_OUTPUT_DIR / "cells" / "r2_r3" / "r3_epistasis" / f"{arm_id}.json"
    sealed_r3 = None
    if sealed_r3_path.is_file():
        cand = json.loads(sealed_r3_path.read_text(encoding="utf-8"))
        if cand.get("support_sha256") == ctx.support_sha256 and cand.get("MR_cycle_top_rung") is not None:
            sealed_r3 = cand
    need_joint = include_live_joint or (sealed_r3 is None)

    idx_cc = ctx.kept_global_indices
    is_pre33 = ARM_METADATA[arm_id].origin_cohort == "preregistered_33"
    if is_pre33:
        m_all, r_all, t_cc = r3_mod._load_preregistered_33_r3_arrays(arm_id, ctx)
        if np.array_equal(t_cc, ctx.panel["blocks"]["T"]):
            rung_caches, oof_targets, u_c_metrics = _R3_STATE["aa_caches"]
        else:
            blocks_arm = {**ctx.panel["blocks"], "T": t_cc}
            rung_caches, oof_targets, u_c_metrics = r3_mod._r3_build_rung_caches(
                ctx, blocks_arm, rungs=r3_mod.R3_RUNGS
            )
    else:
        m_all, r_all = r3_mod._load_expanded_r3_arrays(arm_id, ctx)
        rung_caches, oof_targets, u_c_metrics = _R3_STATE["aa_caches"]

    if r_all is None or r_all.shape[1] != 1024:
        raise RuntimeError(f"{arm_id}: missing 1024-d R3 cycle R array")

    m_cc = m_all[idx_cc][:, None]
    r_cc = r_all[idx_cc]

    panel = ctx.panel
    groups = panel["group"]
    site_pairs = panel["site_pair"]
    weights = ctx.weights
    n_rows = len(groups)
    any_rung = r3_mod.R3_RUNGS[0]
    top_rung = "nuisance+profile+couplings"

    def _score_seed_preds(preds_by_seed: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        mse_s, sp_s = [], []
        for seed in pw_mod.SPLIT_SEEDS:
            _, mse_v = pw_mod.group_errors(oof_targets[seed], preds_by_seed[seed], groups, site_pairs)
            _, sp_v = pw_mod.group_spearman(oof_targets[seed], preds_by_seed[seed], groups)
            mse_s.append(np.asarray(mse_v, dtype=np.float64))
            sp_s.append(sp_v)
        return np.mean(mse_s, axis=0), r3_mod._average_seed_spearman(sp_s, len(ctx.group_order))

    if sealed_r3 is not None and not need_joint:
        m_mse_l = [float(x) for x in sealed_r3["MR_cycle_top_rung"]["mse"]["M"]]
        m_sp_l = [float(x) for x in sealed_r3["MR_cycle_top_rung"]["spearman"]["M"]]
        cm_top_mse_l = [float(x) for x in sealed_r3["MR_cycle_top_rung"]["mse"]["U_C_M"]]
        cm_top_sp_l = [float(x) for x in sealed_r3["MR_cycle_top_rung"]["spearman"]["U_C_M"]]
    else:
        m_preds = {
            seed: r3_mod._r3_nested_predict_alone(
                rung_caches[any_rung][seed], m_cc, weights, n_rows, _R3_STATE["yfit"][seed]
            )
            for seed in pw_mod.SPLIT_SEEDS
        }
        m_mse, m_sp = _score_seed_preds(m_preds)
        cm_top_preds = {
            seed: r3_mod._r3_nested_predict_extra(rung_caches[top_rung][seed], m_cc, weights, n_rows)
            for seed in pw_mod.SPLIT_SEEDS
        }
        cm_top_mse, cm_top_sp = _score_seed_preds(cm_top_preds)
        m_mse_l, m_sp_l = m_mse.tolist(), m_sp.tolist()
        cm_top_mse_l, cm_top_sp_l = cm_top_mse.tolist(), cm_top_sp.tolist()

    variants: dict[str, Any] = {}
    slice_list: list[tuple[str, np.ndarray]] = [("mid", r_cc[:, :512]), ("last", r_cc[:, 512:])]
    if need_joint:
        slice_list.append(("joint", r_cc))

    for vname, r_slice in slice_list:
        r_preds, cr_preds_by_rung, cmr_preds = r3_mod._r3_fit_all_r_and_mr_shared(
            rung_caches, m_cc, r_slice, weights, n_rows
        )
        r_mse, r_sp = _score_seed_preds(r_preds)
        cmr_mse, cmr_sp = _score_seed_preds(cmr_preds)
        cr_mse_by_rung: dict[str, list[float]] = {}
        cr_sp_by_rung: dict[str, list[float]] = {}
        for rung in r3_mod.R3_RUNGS:
            crm, crs = _score_seed_preds(cr_preds_by_rung[rung])
            cr_mse_by_rung[rung] = crm.tolist()
            cr_sp_by_rung[rung] = crs.tolist()
        variants[vname] = {
            "R_alone_mse": r_mse.tolist(),
            "R_alone_spearman": r_sp.tolist(),
            "U_C_R_mse_by_rung": cr_mse_by_rung,
            "U_C_R_spearman_by_rung": cr_sp_by_rung,
            "U_C_MR_top_mse": cmr_mse.tolist(),
            "U_C_MR_top_spearman": cmr_sp.tolist(),
        }

    if "joint" not in variants:
        assert sealed_r3 is not None
        variants["joint"] = {
            "R_alone_mse": [float(x) for x in sealed_r3["MR_cycle_top_rung"]["mse"]["R"]],
            "R_alone_spearman": [float(x) for x in sealed_r3["MR_cycle_top_rung"]["spearman"]["R"]],
            "U_C_R_mse_by_rung": {
                r: [float(x) for x in sealed_r3["R_cycle"]["mse"][r]["U_C_M"]] for r in r3_mod.R3_RUNGS
            },
            "U_C_R_spearman_by_rung": {
                r: [float(x) for x in sealed_r3["R_cycle"]["spearman"][r]["U_C_M"]] for r in r3_mod.R3_RUNGS
            },
            "U_C_MR_top_mse": [float(x) for x in sealed_r3["MR_cycle_top_rung"]["mse"]["U_C_MR"]],
            "U_C_MR_top_spearman": [float(x) for x in sealed_r3["MR_cycle_top_rung"]["spearman"]["U_C_MR"]],
        }

    rec = {
        "schema": CELL_CACHE_SCHEMA,
        "dataset": "r3_epistasis",
        "arm_id": arm_id,
        "support_sha256": ctx.support_sha256,
        "groups": list(ctx.group_order),
        "live_joint_fitted": need_joint,
        "elapsed_seconds": float(time.monotonic() - t0),
        "U_C_mse_by_rung": {r: u_c_metrics[r]["mse"].tolist() for r in r3_mod.R3_RUNGS},
        "U_C_spearman_by_rung": {r: u_c_metrics[r]["spearman"].tolist() for r in r3_mod.R3_RUNGS},
        "M_alone_mse": m_mse_l,
        "M_alone_spearman": m_sp_l,
        "U_C_M_top_mse": cm_top_mse_l,
        "U_C_M_top_spearman": cm_top_sp_l,
        "variants": variants,
    }
    atomic_write_json(cache_path, rec)
    return rec


# ============================================================================
# Paired Banding, Paradigm Stratification, and Ladder Rollups
# ============================================================================


def _safe_paired_bands(
    mat: np.ndarray,
    columns: Sequence[str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run ``paired_bands_summary`` over groups finite across all columns, recording any dropped groups."""
    arr = np.asarray(mat, dtype=np.float64)
    finite_rows = np.isfinite(arr).all(axis=1)
    n_total = int(arr.shape[0])
    dropped = [int(i) for i in np.flatnonzero(~finite_rows).tolist()]
    res = r2_mod.paired_bands_summary(arr[finite_rows], list(columns), draws=draws, seed=seed)
    if dropped:
        res["total_groups_before_nan_filter"] = n_total
        res["dropped_nonfinite_group_indices"] = dropped
    return res


def _build_paradigm_and_ladder_rollups(
    arm_order: Sequence[str],
    contrasts_per_arm: Mapping[str, Mapping[str, float]],
    levels_per_arm: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Aggregate paired mid-vs-last contrasts across 9 fine paradigms, 4 macro paradigms, and 13 ladders."""
    contrast_keys = sorted(next(iter(contrasts_per_arm.values())).keys())
    level_keys = sorted(next(iter(levels_per_arm.values())).keys())

    def _agg_subset(arms: Sequence[str]) -> dict[str, Any]:
        c_stats: dict[str, Any] = {}
        for ck in contrast_keys:
            vals = np.asarray([contrasts_per_arm[a][ck] for a in arms], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            c_stats[ck] = {
                "mean": float(np.mean(finite)) if finite.size else None,
                "median": float(np.median(finite)) if finite.size else None,
                "min": float(np.min(finite)) if finite.size else None,
                "max": float(np.max(finite)) if finite.size else None,
                "n_positive": int(np.sum(finite > 0.0)),
                "n_negative": int(np.sum(finite < 0.0)),
                "n_arms": int(finite.size),
            }
        l_stats: dict[str, Any] = {}
        for lk in level_keys:
            vals = np.asarray([levels_per_arm[a][lk] for a in arms], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            l_stats[lk] = {
                "mean": float(np.mean(finite)) if finite.size else None,
                "median": float(np.median(finite)) if finite.size else None,
            }
        return {"n_arms": len(arms), "arms": list(arms), "levels": l_stats, "contrasts": c_stats}

    fine_map: dict[str, Any] = {}
    for fp in FINE_PARADIGM_ORDER:
        sub = [a for a in arm_order if classify_fine_paradigm(a) == fp]
        if sub:
            fine_map[fp] = _agg_subset(sub)

    macro_map: dict[str, Any] = {}
    for mp_name in MACRO_PARADIGM_ORDER:
        sub = [a for a in arm_order if classify_macro_paradigm(a) == mp_name]
        if sub:
            macro_map[mp_name] = _agg_subset(sub)

    ladders_map: dict[str, Any] = {}
    for lname, rungs in r2_mod.SCALING_LADDERS.items():
        present = [r for r in rungs if r in contrasts_per_arm]
        if not present:
            continue
        ladders_map[lname] = {
            "rungs": present,
            "params_m": [float(ARM_METADATA[r].params_m) for r in present],
            "per_rung_contrasts": {r: dict(contrasts_per_arm[r]) for r in present},
            "per_rung_levels": {r: dict(levels_per_arm[r]) for r in present},
        }

    return {
        "overall_62": _agg_subset(arm_order),
        "by_macro_paradigm_4": macro_map,
        "by_fine_paradigm_9": fine_map,
        "by_scaling_ladder_13": ladders_map,
    }


def _summarize_r2_panel(
    arm_order: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    panel_key: str,
    baseline_c: Mapping[str, np.ndarray],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Build complete mid-vs-last paired contrast tables for one MegaScale or Domainome panel."""
    out_by_ep: dict[str, Any] = {}
    for ep_idx, ep in enumerate(r2_mod.ENDPOINTS):
        is_mse = ep != "ranking"
        sign = 1.0 if is_mse else -1.0  # for MSE: (base - aug) > 0 is better; for rho: (aug - base) > 0 is better
        c_vec = np.asarray(baseline_c[ep], dtype=np.float64)

        def _get(arm_id: str, design: str) -> np.ndarray:
            if design == "C":
                return c_vec
            return np.asarray(records[arm_id][panel_key][ep][design], dtype=np.float64)

        # Build contrast matrices (n_groups, n_arms) — positive always means improvement / mid wins!
        contrast_defs: dict[str, tuple[str, str]] = {
            "R_mid_minus_R_last": ("R_last", "R_mid"),
            "C_R_mid_minus_C_R_last": ("C_R_last", "C_R_mid"),
            "C_M_R_mid_minus_C_M_R_last": ("C_M_R_last", "C_M_R_mid"),
            "plus_R_mid_given_CM": ("C_M", "C_M_R_mid"),
            "plus_R_last_given_CM": ("C_M", "C_M_R_last"),
            "plus_R_joint_given_CM": ("C_M", "C_M_R_joint"),
            "plus_R_mid_given_C": ("C", "C_R_mid"),
            "plus_R_last_given_C": ("C", "C_R_last"),
            "plus_R_joint_given_C": ("C", "C_R_joint"),
            "R_mid_minus_R_joint": ("R_joint", "R_mid"),
            "C_M_R_mid_minus_C_M_R_joint": ("C_M_R_joint", "C_M_R_mid"),
            "R_last_minus_R_joint": ("R_joint", "R_last"),
        }

        bands_out: dict[str, Any] = {}
        per_arm_contrasts: dict[str, dict[str, float]] = {a: {} for a in arm_order}
        for c_idx, (cname, (base_d, aug_d)) in enumerate(contrast_defs.items()):
            mat = np.column_stack(
                [sign * (_get(a, base_d) - _get(a, aug_d)) for a in arm_order]
            )
            b_res = _safe_paired_bands(mat, arm_order, draws=draws, seed=seed + ep_idx * 100 + c_idx)
            bands_out[cname] = b_res
            for a in arm_order:
                per_arm_contrasts[a][cname] = float(b_res["columns"][a]["point"])

        design_names = (
            "C", "M", "R_mid", "R_last", "R_joint",
            "C_M", "C_R_mid", "C_R_last", "C_R_joint",
            "C_M_R_mid", "C_M_R_last", "C_M_R_joint",
        )
        per_arm_levels: dict[str, dict[str, float]] = {}
        for a in arm_order:
            per_arm_levels[a] = {}
            for d in design_names:
                v = _get(a, d)
                fin = v[np.isfinite(v)]
                per_arm_levels[a][d] = float(np.mean(fin)) if fin.size else float("nan")

        rollups = _build_paradigm_and_ladder_rollups(arm_order, per_arm_contrasts, per_arm_levels)
        out_by_ep[ep] = {
            "sign_convention": (
                "Positive (> 0) means LOWER MSE (better performance / R_mid beats R_last)."
                if is_mse
                else "Positive (> 0) means HIGHER Spearman rho (better performance / R_mid beats R_last)."
            ),
            "baseline_C_mean": float(np.nanmean(c_vec)),
            "contrasts": bands_out,
            "per_arm_levels": per_arm_levels,
            "stratified_rollups": rollups,
        }
    return out_by_ep


def _summarize_r3_epistasis(
    arm_order: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Build complete mid-vs-last paired contrast tables for R3 double-mutant epistasis (61 groups)."""
    top_rung = "nuisance+profile+couplings"
    out_by_ep: dict[str, Any] = {}
    for ep_idx, ep in enumerate(("mse", "spearman")):
        is_mse = ep == "mse"
        sign = 1.0 if is_mse else -1.0

        def _vec(arm_id: str, key: str, rung: str = top_rung) -> np.ndarray:
            rec = records[arm_id]
            if key == "U_C":
                return np.asarray(rec[f"U_C_{ep}_by_rung"][rung], dtype=np.float64)
            if key == "M":
                return np.asarray(rec[f"M_alone_{ep}"], dtype=np.float64)
            if key == "U_C_M":
                return np.asarray(rec[f"U_C_M_top_{ep}"], dtype=np.float64)
            if key.startswith("R_"):
                vname = key.split("_", 1)[1]
                return np.asarray(rec["variants"][vname][f"R_alone_{ep}"], dtype=np.float64)
            if key.startswith("U_C_R_"):
                vname = key.split("U_C_R_", 1)[1]
                return np.asarray(rec["variants"][vname][f"U_C_R_{ep}_by_rung"][rung], dtype=np.float64)
            if key.startswith("U_C_MR_"):
                vname = key.split("U_C_MR_", 1)[1]
                return np.asarray(rec["variants"][vname][f"U_C_MR_top_{ep}"], dtype=np.float64)
            raise KeyError(key)

        top_contrast_defs: dict[str, tuple[str, str]] = {
            "R_cycle_mid_minus_last": ("R_last", "R_mid"),
            "U_C_R_mid_minus_last": ("U_C_R_last", "U_C_R_mid"),
            "MR_cycle_mid_minus_last": ("U_C_MR_last", "U_C_MR_mid"),
            "plus_R_mid_given_CM": ("U_C_M", "U_C_MR_mid"),
            "plus_R_last_given_CM": ("U_C_M", "U_C_MR_last"),
            "plus_R_joint_given_CM": ("U_C_M", "U_C_MR_joint"),
            "plus_R_mid_given_C": ("U_C", "U_C_R_mid"),
            "plus_R_last_given_C": ("U_C", "U_C_R_last"),
            "plus_R_joint_given_C": ("U_C", "U_C_R_joint"),
            "R_cycle_mid_minus_joint": ("R_joint", "R_mid"),
            "MR_cycle_mid_minus_joint": ("U_C_MR_joint", "U_C_MR_mid"),
        }

        bands_out: dict[str, Any] = {}
        per_arm_contrasts: dict[str, dict[str, float]] = {a: {} for a in arm_order}
        for c_idx, (cname, (base_k, aug_k)) in enumerate(top_contrast_defs.items()):
            mat = np.column_stack([sign * (_vec(a, base_k) - _vec(a, aug_k)) for a in arm_order])
            b_res = _safe_paired_bands(mat, arm_order, draws=draws, seed=seed + 500 + ep_idx * 100 + c_idx)
            bands_out[cname] = b_res
            for a in arm_order:
                per_arm_contrasts[a][cname] = float(b_res["columns"][a]["point"])

        # Also evaluate U_C + R_mid vs U_C + R_last across all 3 rungs
        by_rung_mid_vs_last: dict[str, Any] = {}
        for r_idx, rung in enumerate(r3_mod.R3_RUNGS):
            mat_r = np.column_stack(
                [sign * (_vec(a, "U_C_R_last", rung) - _vec(a, "U_C_R_mid", rung)) for a in arm_order]
            )
            by_rung_mid_vs_last[rung] = _safe_paired_bands(
                mat_r, arm_order, draws=draws, seed=seed + 800 + ep_idx * 20 + r_idx
            )

        level_keys = (
            "U_C", "M", "R_mid", "R_last", "R_joint",
            "U_C_M", "U_C_R_mid", "U_C_R_last", "U_C_R_joint",
            "U_C_MR_mid", "U_C_MR_last", "U_C_MR_joint",
        )
        per_arm_levels: dict[str, dict[str, float]] = {}
        for a in arm_order:
            per_arm_levels[a] = {}
            for lk in level_keys:
                v = _vec(a, lk)
                fin = v[np.isfinite(v)]
                per_arm_levels[a][lk] = float(np.mean(fin)) if fin.size else float("nan")

        rollups = _build_paradigm_and_ladder_rollups(arm_order, per_arm_contrasts, per_arm_levels)
        out_by_ep[ep] = {
            "sign_convention": (
                "Positive (> 0) means LOWER MSE (better performance / R_mid beats R_last)."
                if is_mse
                else "Positive (> 0) means HIGHER Spearman rho (better performance / R_mid beats R_last)."
            ),
            "top_rung": top_rung,
            "contrasts": bands_out,
            "all_rungs_U_C_R_mid_minus_last": by_rung_mid_vs_last,
            "per_arm_levels": per_arm_levels,
            "stratified_rollups": rollups,
        }
    return out_by_ep


# ============================================================================
# Strict Parity Gate (< 1e-10) Against Sealed R2 & R3 Unified Artifacts
# ============================================================================


def _verify_parity_against_sealed(
    root: Path,
    arm_order: Sequence[str],
    ms_summary: Mapping[str, Any],
    dm_summary: Mapping[str, Any],
    r3_summary: Mapping[str, Any],
    *,
    tolerance: float = PARITY_TOLERANCE,
) -> dict[str, Any]:
    """Verify ``R_joint`` (cols ``0:1024``) reproduces sealed ``r2_biophysical_unified_62.json`` and
    ``r3_epistasis_and_contacts_unified_62.json`` to ``< 1e-10`` across all 62 arms.
    """
    r2_path = root / UNIFIED_OUTPUT_DIR / "r2_biophysical_unified_62.json"
    r3_path = root / UNIFIED_OUTPUT_DIR / "r3_epistasis_and_contacts_unified_62.json"

    r2_doc = json.loads(r2_path.read_text(encoding="utf-8"))
    r3_doc = json.loads(r3_path.read_text(encoding="utf-8"))

    r2_diffs: dict[str, float] = {}
    r2_checks = [
        ("megascale_evo_79", ms_summary["evo_79"], r2_doc["datasets"]["megascale"]["primary_evolutionary_control"]["cc_79_profile_potts_gemme"]["per_arm_summary"]),
        ("megascale_evo_101", ms_summary["evo_101"], r2_doc["datasets"]["megascale"]["primary_evolutionary_control"]["full_101_profile"]["per_arm_summary"]),
        ("megascale_full_79", ms_summary["full_79"], r2_doc["datasets"]["megascale"]["full_nuisance_evolutionary_control"]["cc_79_top_rung_6_designs"]["per_arm_summary"]),
        ("domainome_evo_94", dm_summary["evo_94"], r2_doc["datasets"]["domainome"]["primary_evolutionary_control"]["cc_94_profile_potts_gemme"]["per_arm_summary"]),
        ("domainome_evo_96", dm_summary["evo_96"], r2_doc["datasets"]["domainome"]["primary_evolutionary_control"]["full_96_profile"]["per_arm_summary"]),
        ("domainome_full_94", dm_summary["full_94"], r2_doc["datasets"]["domainome"]["full_nuisance_evolutionary_control"]["cc_94_top_rung_6_designs"]["per_arm_summary"]),
    ]
    for label, ours_panel, sealed_per_arm in r2_checks:
        max_d = 0.0
        for a in arm_order:
            s_arm = sealed_per_arm[a]
            for our_d, sealed_d in (("R_joint", "R"), ("C_R_joint", "C_R"), ("C_M_R_joint", "C_M_R")):
                for ep in r2_mod.ENDPOINTS:
                    v_ours = float(ours_panel[ep]["per_arm_levels"][a][our_d])
                    v_seal = float(s_arm[sealed_d][ep])
                    if math.isfinite(v_ours) and math.isfinite(v_seal):
                        max_d = max(max_d, abs(v_ours - v_seal))
        r2_diffs[label] = max_d

    # R3 MR_cycle_62_over_nuisance_profile_couplings parity across all 62 arms (MSE & Spearman)
    sealed_mr = r3_doc["epistasis"]["MR_cycle_62_over_nuisance_profile_couplings"]
    r3_max_mse = 0.0
    r3_max_sp = 0.0
    for ep, is_mse in (("mse", True), ("spearman", False)):
        s_mr = sealed_mr[ep]["MR_given_controls"]["arms"]
        s_r = sealed_mr[ep]["R_given_controls"]["arms"]
        s_mr_over_m = sealed_mr[ep]["R_given_controls_and_M"]["arms"]
        for a in arm_order:
            # Compare +R_joint | C and +R_joint | (C+M) points directly
            our_r_c = float(r3_summary[ep]["contrasts"]["plus_R_joint_given_C"]["columns"][a]["point"])
            our_r_cm = float(r3_summary[ep]["contrasts"]["plus_R_joint_given_CM"]["columns"][a]["point"])
            # Sealed sign convention in r3_yardstick: MSE uses (reduced - joint), Spearman uses (joint - reduced)
            seal_r_c = float(s_r[a]["point"])
            seal_r_cm = float(s_mr_over_m[a]["point"])
            # Also check MR_given_controls = (U_C - U_C_MR) for MSE / (U_C_MR - U_C) for Spearman
            our_lv = r3_summary[ep]["per_arm_levels"][a]
            our_mr_c = (our_lv["U_C"] - our_lv["U_C_MR_joint"]) if is_mse else (our_lv["U_C_MR_joint"] - our_lv["U_C"])
            seal_mr_c = float(s_mr[a]["point"])
            d = max(abs(our_r_c - seal_r_c), abs(our_r_cm - seal_r_cm), abs(our_mr_c - seal_mr_c))
            if is_mse:
                r3_max_mse = max(r3_max_mse, d)
            else:
                r3_max_sp = max(r3_max_sp, d)

    all_max = max(max(r2_diffs.values()), r3_max_mse, r3_max_sp)
    passed = bool(all_max < tolerance)
    return {
        "passed": passed,
        "tolerance": float(tolerance),
        "max_abs_diff": float(all_max),
        "n_arms_checked": len(arm_order),
        "r2_sealed_sha256": sha256_file(r2_path),
        "r3_sealed_sha256": sha256_file(r3_path),
        "r2_max_abs_diff_by_panel": r2_diffs,
        "r3_max_abs_diff_mse": float(r3_max_mse),
        "r3_max_abs_diff_spearman": float(r3_max_sp),
    }


# ============================================================================
# Top-Level Orchestrator
# ============================================================================


def build_r2_r3_middle_vs_last_62(
    *,
    root: Path = REPO_ROOT,
    out_path: Path | None = None,
    cache_dir: Path | None = None,
    r_archive_dir: Path | None = None,
    arms: Sequence[str] | None = None,
    max_workers: int = 16,
    blas_threads: int = 2,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    force_refit: bool = False,
    verify_sealed_parity: bool = True,
) -> dict[str, Any]:
    """Run the full 62-model Middle-vs-Last representation contrast across MegaScale, Domainome, and R3."""
    t_start = time.monotonic()
    arm_order = tuple(arms) if arms is not None else UNIFIED_62_SINGLE_SEQUENCE_ARMS
    dest = Path(out_path) if out_path is not None else (root / DEFAULT_OUTPUT_REL)
    croot = Path(cache_dir) if cache_dir is not None else (root / DEFAULT_CELL_CACHE_REL)
    r_dir = Path(r_archive_dir) if r_archive_dir is not None else (root / r2_mod.DEFAULT_R_ARCHIVE_DIR)
    ms_cdir = croot / "megascale"
    dm_cdir = croot / "domainome"
    r3_cdir = croot / "r3_epistasis"
    for d in (ms_cdir, dm_cdir, r3_cdir):
        d.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    n_workers = max(1, min(int(max_workers), len(arm_order), 64))

    # -------------------- 1. MEGASCALE (79g & 101g) --------------------
    print(f"[r2-r3-mid-last] 1/3 MegaScale ({len(arm_order)} arms, workers={n_workers})...", flush=True)
    ms = r2_mod.load_megascale_supports(root=root)
    ms_w_101 = row_weights(ms["panel_101"]["site"], ms["panel_101"]["group"])
    ms_w_79 = row_weights(ms["panel_79"]["site"], ms["panel_79"]["group"])
    fc_ms_79_evo = r2_mod.build_fold_cache(groups=ms["panel_79"]["group"], weights=ms_w_79)
    fc_ms_101_evo = r2_mod.build_fold_cache(groups=ms["panel_101"]["group"], weights=ms_w_101)
    fc_ms_79_full = r2_mod.build_fold_cache(
        groups=ms["panel_79"]["group"],
        weights=ms_w_79,
        g_builder=lambda held, inner: gate.nuisance_response(
            ms["panel_79"]["states"], ms["panel_79"]["pair_states"], np.asarray(held), inner, device="cpu"
        )[0],
    )
    fc_ms_101_full = r2_mod.build_fold_cache(
        groups=ms["panel_101"]["group"],
        weights=ms_w_101,
        g_builder=lambda held, inner: gate.nuisance_response(
            ms["panel_101"]["states"], ms["panel_101"]["pair_states"], np.asarray(held), inner, device="cpu"
        )[0],
    )

    eval_ms_79 = lambda p: r2_mod.evaluate_megascale_oof(  # noqa: E731
        p, ms["panel_79"]["target"], ms["panel_79"]["group"], ms["panel_79"]["site"], ms["groups_79"]
    )
    eval_ms_101 = lambda p: r2_mod.evaluate_megascale_oof(  # noqa: E731
        p, ms["panel_101"]["target"], ms["panel_101"]["group"], ms["panel_101"]["site"], ms["groups_101"]
    )
    ms_base_79_evo = r2_mod.collapse_oof(
        r2_mod.fit_designs_with_cache(
            ms["panel_79"]["target"],
            {"P": ms["p_logodds_79"][:, None], "POTTS": ms["potts_raw_79"][:, None], "GEMME": ms["gemme_raw_79"][:, None]},
            {"C": ("P", "POTTS", "GEMME")},
            fc_ms_79_evo, alphas=r2_mod.ALPHAS_17, renorm_val_weights=True, tie_largest_alpha=True,
        ),
        eval_ms_79,
    )
    ms_base_101_evo = r2_mod.collapse_oof(
        r2_mod.fit_designs_with_cache(
            ms["panel_101"]["target"],
            {"P": ms["p_logodds_101"][:, None]},
            {"C": ("P",)},
            fc_ms_101_evo, alphas=r2_mod.ALPHAS_17, renorm_val_weights=True, tie_largest_alpha=True,
        ),
        eval_ms_101,
    )
    ms_base_79_full = r2_mod.collapse_oof(
        r2_mod.fit_designs_with_cache(
            ms["panel_79"]["target"],
            ms["panel_79"]["blocks"],
            {"C": r2_mod.MS_PROGRESSION_RUNGS["nuisance+profile+couplings+gemme"]},
            fc_ms_79_full, alphas=r2_mod.ALPHAS_5,
        ),
        eval_ms_79,
    )
    ms_base_101_full = r2_mod.collapse_oof(
        r2_mod.fit_designs_with_cache(
            ms["panel_101"]["target"],
            ms["panel_101"]["blocks"],
            {"C": r2_mod.MS_DESIGN_S2},
            fc_ms_101_full, alphas=r2_mod.ALPHAS_5,
        ),
        eval_ms_101,
    )

    pre33_ms_101 = r2_mod.load_preregistered_ms_101_readings(ms, root=root)
    pre33_ms_79 = r2_mod.load_preregistered_ms_79_cells(ms, root=root)
    pre33_full79_m = {
        a: {
            ep: {
                "M": pre33_ms_79[a]["nuisance+profile+couplings+gemme"][ep]["M"],
                "C_M": pre33_ms_79[a]["nuisance+profile+couplings+gemme"][ep]["U_C_M"],
            }
            for ep in r2_mod.ENDPOINTS
        }
        for a in pre33_ms_79
    }
    pre33_full101_m = {
        a: {
            ep: {
                "C_M": pre33_ms_101[a][ep]["S2_M"],
            }
            for ep in r2_mod.ENDPOINTS
        }
        for a in pre33_ms_101
    }
    live_joint_arms = tuple(a for a in arm_order if len(arm_order) <= 4 or a in PARITY_GATE_ARMS)

    ms_shared = {
        "root": str(root),
        "cache_dir": str(ms_cdir),
        "r_archive_dir": str(r_dir),
        "force": force_refit,
        "blas_threads": blas_threads,
        "live_joint_arms": live_joint_arms,
        "pre33_full79_m": pre33_full79_m,
        "pre33_full101_m": pre33_full101_m,
        "cohort_101": ms["cohort_101"],
        "panel_101": ms["panel_101"],
        "groups_101": ms["groups_101"],
        "panel_79": ms["panel_79"],
        "groups_79": ms["groups_79"],
        "mask_79": ms["mask_79"],
        "p_logodds_101": ms["p_logodds_101"],
        "p_logodds_79": ms["p_logodds_79"],
        "potts_raw_79": ms["potts_raw_79"],
        "gemme_raw_79": ms["gemme_raw_79"],
        "fc_79_evo": fc_ms_79_evo,
        "fc_101_evo": fc_ms_101_evo,
        "fc_79_full": fc_ms_79_full,
        "fc_101_full": fc_ms_101_full,
    }
    ms_records: dict[str, dict[str, Any]] = {}
    ms_todo: list[str] = []
    for arm_id in arm_order:
        cp = ms_cdir / f"{arm_id}.json"
        if not force_refit and cp.is_file():
            try:
                cd = json.loads(cp.read_text(encoding="utf-8"))
                if (
                    cd.get("schema") == CELL_CACHE_SCHEMA
                    and cd.get("dataset") == "megascale"
                    and cd.get("arm_id") == arm_id
                    and (arm_id not in live_joint_arms or cd.get("live_joint_fitted") is True)
                ):
                    ms_records[arm_id] = cd
                    continue
            except Exception:
                pass
        ms_todo.append(arm_id)
    if ms_todo:
        ms_w = max(1, min(n_workers, len(ms_todo), 16))
        if ms_w == 1:
            _init_ms_worker(ms_shared)
            for arm_id in ms_todo:
                ms_records[arm_id] = _fit_one_ms_mid_last(arm_id)
        else:
            with ProcessPoolExecutor(
                max_workers=ms_w, mp_context=mp.get_context("spawn"),
                initializer=_init_ms_worker, initargs=(ms_shared,),
            ) as pool:
                for idx, rec in enumerate(pool.map(_fit_one_ms_mid_last, ms_todo), 1):
                    ms_records[rec["arm_id"]] = rec
                    if idx % 5 == 0 or idx == len(ms_todo):
                        print(f"  [ms {idx}/{len(ms_todo)}] {rec['arm_id']} ({rec['elapsed_seconds']:.1f}s)", flush=True)
    else:
        print(f"  [ms] loaded all {len(ms_records)}/{len(arm_order)} cached MegaScale cells", flush=True)

    ms_summary = {
        "evo_79": _summarize_r2_panel(
            arm_order, ms_records, "evo_79", {ep: ms_base_79_evo[ep]["C"] for ep in r2_mod.ENDPOINTS},
            draws=draws, seed=seed,
        ),
        "evo_101": _summarize_r2_panel(
            arm_order, ms_records, "evo_101", {ep: ms_base_101_evo[ep]["C"] for ep in r2_mod.ENDPOINTS},
            draws=draws, seed=seed + 1000,
        ),
        "full_79": _summarize_r2_panel(
            arm_order, ms_records, "full_79", {ep: ms_base_79_full[ep]["C"] for ep in r2_mod.ENDPOINTS},
            draws=draws, seed=seed + 2000,
        ),
        "full_101": _summarize_r2_panel(
            arm_order, ms_records, "full_101", {ep: ms_base_101_full[ep]["C"] for ep in r2_mod.ENDPOINTS},
            draws=draws, seed=seed + 3000,
        ),
    }

    # -------------------- 2. DOMAINOME (94g & 96g) --------------------
    print(f"[r2-r3-mid-last] 2/3 Domainome ({len(arm_order)} arms, workers={n_workers})...", flush=True)
    dm = r2_mod.load_domainome_supports(root=root)
    dm_w_94_site = row_weights(dm["panel_94"]["site"], dm["panel_94"]["group"])
    dm_w_96_site = row_weights(dm["panel_96"]["site"], dm["panel_96"]["group"])
    fc_dm_94_evo = r2_mod.build_fold_cache(groups=dm["panel_94"]["group"], weights=dm_w_94_site)
    fc_dm_96_evo = r2_mod.build_fold_cache(groups=dm["panel_96"]["group"], weights=dm_w_96_site)
    dm_g_cache = root / DEFAULT_CELL_CACHE_REL / "_dm_fc_94_full_G_cache.npz"
    fc_dm_94_full = _build_or_load_dm_fc_94_full(dm, dm_g_cache, max_workers=min(15, max(1, int(max_workers))))

    eval_dm_94_evo = lambda p: r2_mod.evaluate_domainome_oof(  # noqa: E731
        p, dm["panel_94"]["target"], dm["panel_94"]["group"], dm["panel_94"]["domain"], dm["panel_94"]["site"],
        dm["groups_94"], use_nested_weights=False,
    )
    eval_dm_96_evo = lambda p: r2_mod.evaluate_domainome_oof(  # noqa: E731
        p, dm["panel_96"]["target"], dm["panel_96"]["group"], dm["panel_96"]["domain"], dm["panel_96"]["site"],
        dm["groups_96"], use_nested_weights=False,
    )
    eval_dm_94_full = lambda p: r2_mod.evaluate_domainome_oof(  # noqa: E731
        p, dm["panel_94"]["target"], dm["panel_94"]["group"], dm["panel_94"]["domain"], dm["panel_94"]["site"],
        dm["groups_94"], use_nested_weights=True,
    )
    dm_base_94_evo = r2_mod.collapse_oof(
        fit_designs_group_suffstats(
            dm["panel_94"]["target"],
            dm["panel_94"]["group"],
            {"P": dm["p_logodds_94"][:, None], "POTTS": dm["potts_raw_94"][:, None], "GEMME": dm["gemme_raw_94"][:, None]},
            {"C": ("P", "POTTS", "GEMME")},
            fc_dm_94_evo, alphas=r2_mod.ALPHAS_17, renorm_val_weights=True, tie_largest_alpha=True,
        ),
        eval_dm_94_evo,
    )
    dm_base_96_evo = r2_mod.collapse_oof(
        fit_designs_group_suffstats(
            dm["panel_96"]["target"],
            dm["panel_96"]["group"],
            {"P": dm["p_logodds_96"][:, None]},
            {"C": ("P",)},
            fc_dm_96_evo, alphas=r2_mod.ALPHAS_17, renorm_val_weights=True, tie_largest_alpha=True,
        ),
        eval_dm_96_evo,
    )
    dm_base_94_full = r2_mod.collapse_oof(
        fit_designs_group_suffstats(
            dm["panel_94"]["target"],
            dm["panel_94"]["group"],
            dm["panel_94"]["blocks"],
            {"C": r2_mod.DM_PROGRESSION_RUNGS["nuisance+profile+couplings+gemme"]},
            fc_dm_94_full, alphas=r2_mod.ALPHAS_5,
        ),
        eval_dm_94_full,
    )

    dm_shared = {
        "root": str(root),
        "cache_dir": str(dm_cdir),
        "r_archive_dir": str(r_dir),
        "force": force_refit,
        "blas_threads": blas_threads,
        "live_joint_arms": live_joint_arms,
        "cohort_96": dm["cohort_96"],
        "panel_96": dm["panel_96"],
        "groups_96": dm["groups_96"],
        "panel_94": dm["panel_94"],
        "groups_94": dm["groups_94"],
        "mask_94": dm["mask_94"],
        "p_logodds_96": dm["p_logodds_96"],
        "p_logodds_94": dm["p_logodds_94"],
        "potts_raw_94": dm["potts_raw_94"],
        "gemme_raw_94": dm["gemme_raw_94"],
        "fc_94_evo": fc_dm_94_evo,
        "fc_96_evo": fc_dm_96_evo,
        "fc_94_full": fc_dm_94_full,
    }
    dm_records: dict[str, dict[str, Any]] = {}
    dm_todo: list[str] = []
    for arm_id in arm_order:
        cp = dm_cdir / f"{arm_id}.json"
        if not force_refit and cp.is_file():
            try:
                cd = json.loads(cp.read_text(encoding="utf-8"))
                if (
                    cd.get("schema") == CELL_CACHE_SCHEMA
                    and cd.get("dataset") == "domainome"
                    and cd.get("arm_id") == arm_id
                    and (arm_id not in live_joint_arms or cd.get("live_joint_fitted") is True)
                ):
                    dm_records[arm_id] = cd
                    continue
            except Exception:
                pass
        dm_todo.append(arm_id)
    if dm_todo:
        # With group-level sufficient statistics + blas_threads=1, each worker uses ~1.5 GB RAM.
        dm_cap = 16 if int(blas_threads) == 1 else 8
        dm_w = max(1, min(n_workers, len(dm_todo), dm_cap))
        print(
            f"  [dm] {len(dm_records)} cached, fitting {len(dm_todo)} remaining arms (workers={dm_w}, blas={blas_threads})...",
            flush=True,
        )
        if dm_w == 1:
            _init_dm_worker(dm_shared)
            for idx, arm_id in enumerate(dm_todo, 1):
                rec = _fit_one_dm_mid_last(arm_id)
                dm_records[arm_id] = rec
                print(f"  [dm {idx}/{len(dm_todo)}] {rec['arm_id']} ({rec['elapsed_seconds']:.1f}s)", flush=True)
        else:
            with ProcessPoolExecutor(
                max_workers=dm_w, mp_context=mp.get_context("spawn"),
                initializer=_init_dm_worker, initargs=(dm_shared,),
            ) as pool:
                futs = {pool.submit(_fit_one_dm_mid_last, a): a for a in dm_todo}
                for idx, fut in enumerate(as_completed(futs), 1):
                    rec = fut.result()
                    dm_records[rec["arm_id"]] = rec
                    if idx % 5 == 0 or idx == len(dm_todo) or idx <= 3:
                        print(f"  [dm {idx}/{len(dm_todo)}] {rec['arm_id']} ({rec['elapsed_seconds']:.1f}s)", flush=True)
    else:
        print(f"  [dm] loaded all {len(dm_records)}/{len(arm_order)} cached Domainome cells", flush=True)

    dm_summary = {
        "evo_94": _summarize_r2_panel(
            arm_order, dm_records, "evo_94", {ep: dm_base_94_evo[ep]["C"] for ep in r2_mod.ENDPOINTS},
            draws=draws, seed=seed + 4000,
        ),
        "evo_96": _summarize_r2_panel(
            arm_order, dm_records, "evo_96", {ep: dm_base_96_evo[ep]["C"] for ep in r2_mod.ENDPOINTS},
            draws=draws, seed=seed + 5000,
        ),
        "full_94": _summarize_r2_panel(
            arm_order, dm_records, "full_94", {ep: dm_base_94_full[ep]["C"] for ep in r2_mod.ENDPOINTS},
            draws=draws, seed=seed + 6000,
        ),
    }

    # -------------------- 3. R3 DOUBLE-MUTANT EPISTASIS (61g, 7,805 cycles) --------------------
    r3_records: dict[str, dict[str, Any]] = {}
    r3_todo: list[str] = []
    for arm_id in arm_order:
        cp = r3_cdir / f"{arm_id}.json"
        if not force_refit and cp.is_file():
            try:
                cd = json.loads(cp.read_text(encoding="utf-8"))
                if (
                    cd.get("schema") == CELL_CACHE_SCHEMA
                    and cd.get("dataset") == "r3_epistasis"
                    and cd.get("arm_id") == arm_id
                    and (arm_id not in live_joint_arms or cd.get("live_joint_fitted") is True)
                ):
                    r3_records[arm_id] = cd
                    continue
            except Exception:
                pass
        r3_todo.append(arm_id)
    r3_workers = max(1, min(n_workers, len(r3_todo), 32)) if r3_todo else 1
    print(f"[r2-r3-mid-last] 3/3 R3 Epistasis ({len(arm_order)} arms, workers={r3_workers})...", flush=True)
    if r3_todo:
        if r3_workers == 1:
            _init_r3_worker(str(root), str(r3_cdir), blas_threads, force_refit, live_joint_arms)
            for arm_id in r3_todo:
                r3_records[arm_id] = _fit_one_r3_mid_last(arm_id)
        else:
            with ProcessPoolExecutor(
                max_workers=r3_workers, mp_context=mp.get_context("spawn"),
                initializer=_init_r3_worker, initargs=(str(root), str(r3_cdir), blas_threads, force_refit, live_joint_arms),
            ) as pool:
                futs = {pool.submit(_fit_one_r3_mid_last, a): a for a in r3_todo}
                for idx, fut in enumerate(as_completed(futs), 1):
                    rec = fut.result()
                    r3_records[rec["arm_id"]] = rec
                    if idx % 10 == 0 or idx == len(r3_todo):
                        print(f"  [r3 {idx}/{len(r3_todo)}] {rec['arm_id']} ({rec['elapsed_seconds']:.1f}s)", flush=True)
    else:
        print(f"  [r3] loaded all {len(r3_records)}/{len(arm_order)} cached R3 cells", flush=True)

    r3_summary = _summarize_r3_epistasis(arm_order, r3_records, draws=draws, seed=seed + 7000)

    # -------------------- 4. STRICT PARITY GATE AGAINST SEALED ARTIFACTS --------------------
    parity_gate: dict[str, Any]
    if verify_sealed_parity:
        parity_gate = _verify_parity_against_sealed(
            root, arm_order, ms_summary, dm_summary, r3_summary, tolerance=PARITY_TOLERANCE
        )
        if not parity_gate["passed"]:
            raise RuntimeError(
                f"Parity gate FAILED against sealed R2/R3 artifacts: max_abs_diff={parity_gate['max_abs_diff']:.3e} "
                f"(r2={parity_gate['r2_max_abs_diff_by_panel']}, r3_mse={parity_gate['r3_max_abs_diff_mse']:.3e}, "
                f"r3_sp={parity_gate['r3_max_abs_diff_spearman']:.3e})"
            )
    else:
        parity_gate = {"passed": True, "skipped": True, "max_abs_diff": 0.0}

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "completion_status": "complete" if len(arm_order) == 62 else "subset_test",
        "elapsed_seconds": float(time.monotonic() - t_start),
        "feature_contract": {
            "blocks_order": ["middle_mean", "middle_last_or_cls", "final_mean", "final_last_or_cls"],
            "projection_dim_per_block": 256,
            "projection_distribution": "N(0, 1/sqrt(256))",
            "projection_seeds": [20260923, 20260924, 20260925, 20260926],
            "R_mid_slice": "cols 0:512 (512d, canonical middle layer blocks[(L - 1) // 2])",
            "R_last_slice": "cols 512:1024 (512d, final layer blocks[L - 1])",
            "R_joint_slice": "cols 0:1024 (1024d, [R_mid, R_last]; exact match to sealed R)",
        },
        "coverage": {
            "n_arms": len(arm_order),
            "is_complete_62": len(arm_order) == 62,
            "arms_order": list(arm_order),
            "macro_paradigm_counts": {
                mp_k: sum(1 for a in arm_order if classify_macro_paradigm(a) == mp_k)
                for mp_k in MACRO_PARADIGM_ORDER
            },
            "fine_paradigm_counts": {
                fp_k: sum(1 for a in arm_order if classify_fine_paradigm(a) == fp_k)
                for fp_k in FINE_PARADIGM_ORDER
            },
        },
        "layer_provenance_62": {a: build_layer_provenance_table()[a] for a in arm_order},
        "depth_sensitivity_diagnostic": build_depth_sensitivity_diagnostic(root),
        "reproduction_gate": parity_gate,
        "bootstrap": {
            "draws": int(draws),
            "seed": int(seed),
            "confidence": 0.95,
            "methods": ["native_simultaneous_95", "studentized_maxT_95", "marginal_95"],
        },
        "datasets": {
            "megascale": {
                "support": {
                    "cc_79_groups": len(ms["groups_79"]),
                    "cc_79_variants": int(len(ms["panel_79"]["target"])),
                    "full_101_groups": len(ms["groups_101"]),
                    "full_101_variants": int(len(ms["panel_101"]["target"])),
                },
                "primary_evolutionary_control": {
                    "cc_79_profile_potts_gemme": ms_summary["evo_79"],
                    "full_101_profile": ms_summary["evo_101"],
                },
                "full_nuisance_evolutionary_control": {
                    "cc_79_top_rung": ms_summary["full_79"],
                    "full_101_S2": ms_summary["full_101"],
                },
            },
            "domainome": {
                "support": {
                    "cc_94_groups": len(dm["groups_94"]),
                    "cc_94_variants": int(len(dm["panel_94"]["target"])),
                    "full_96_groups": len(dm["groups_96"]),
                    "full_96_variants": int(len(dm["panel_96"]["target"])),
                },
                "primary_evolutionary_control": {
                    "cc_94_profile_potts_gemme": dm_summary["evo_94"],
                    "full_96_profile": dm_summary["evo_96"],
                },
                "full_nuisance_evolutionary_control": {
                    "cc_94_top_rung": dm_summary["full_94"],
                },
            },
            "r3_epistasis": {
                "support": {
                    "n_groups": 61,
                    "n_cycles": 7805,
                    "top_rung": "nuisance+profile+couplings",
                    "rungs": list(r3_mod.R3_RUNGS),
                },
                "results": r3_summary,
            },
        },
    }

    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if FORBIDDEN_INFRA_RE.search(text):
        raise ValueError("Refusing to write r2_r3_middle_vs_last_representation_62.json with forbidden infra id")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(dest)
    digest = sha256_file(dest)
    dest.with_name(dest.name + ".sha256").write_text(f"{digest}  {dest.name}\n", encoding="utf-8")

    return {
        "output_path": str(dest),
        "sha256": digest,
        "payload": payload,
    }


__all__ = [
    "DEFAULT_CELL_CACHE_REL",
    "DEFAULT_OUTPUT_REL",
    "FINE_PARADIGM_ORDER",
    "FORBIDDEN_INFRA_RE",
    "MACRO_PARADIGM_ORDER",
    "MODEL_LAYERS_62",
    "SCHEMA",
    "build_depth_sensitivity_diagnostic",
    "build_layer_provenance_table",
    "build_r2_r3_middle_vs_last_62",
    "classify_fine_paradigm",
    "classify_macro_paradigm",
]
