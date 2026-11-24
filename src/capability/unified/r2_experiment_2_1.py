"""Unified 62-model Merged New Experiment 2.1: Biophysical Transfer, Calibration,
and Scaling Divergence across MegaScale Stability and Human Domainome Abundance.

Implements the complete unified R2 biophysical yardstick strictly on CPU from cached
scores and 1024-d hidden-state readouts (with automatic incremental pickup of any
H200-extracted ``R`` matrices placed under
``results/extensions/unified_yardstick_20261124/r2_readout_archives/``):

1. **Datasets & Instrument Anchor**:
   - **MegaScale In Vitro Proteolysis ddG (kcal/mol)**: dual-protease trypsin vs.
     chymotrypsin reliability anchor from ``endpoint_qualification.json`` + cohort
     variants; evaluated on **79 complete-case family groups** (20,224 variants) and
     **101 full family groups** (25,856 variants).
   - **Human Domainome In-Cell Folding Abundance (aPCA)**: evaluated on **94
     complete-case family groups** (106,496 variants, 416 domains) and **96 full
     family groups** (109,568 variants, 428 domains).

2. **Full 62-Model ``M`` vs. ``R`` Coverage (Local + H200 Archive Ready)**:
   - Domainome: all 33 pre-registered arms load both ``M`` and 1024-d ``R`` directly
     from ``results/external_baseline/20260924085613_5f8d32cf0368/``; 28 expanded GP
     arms load from ``resolve_gp_r2_arrays``; ``esm3-1.4b`` loads ``M`` locally and
     ``R`` from ``r2_readout_archives/r2_abundance_R_esm3-1.4b.npz`` when present.
   - MegaScale: all 33 pre-registered arms load ``M`` from
     ``results/external_baseline/20261008022745_762304146fdd/`` and ``R`` from
     ``r2_readout_archives/r2_stability_R_<arm>.npz`` when present; 28 expanded GP
     arms load both ``M`` and ``R`` from ``resolve_gp_r2_arrays``; ``esm3-1.4b`` loads
     ``M`` locally and ``R`` from ``r2_readout_archives/r2_stability_R_esm3-1.4b.npz``.

3. **Two Control Hierarchies Across All Six Designs (``C, M, R, C+M, C+R, C+M+R``)**:
   - ``primary_evolutionary_control``: ``C_evo = [Profile, Potts, GEMME]`` on 79/94 CC
     groups and ``[Profile]`` on 101/96 full groups (17-point log-alpha grid,
     ``row_weights``, matching ``nested_group_ridge_eval``).
   - ``full_nuisance_evolutionary_control``: ``C_full = Nuisance + Profile + Potts +
     GEMME`` on 79/94 CC groups (exact 5-point ``ALPHAS`` + cached fold-wise ``G``
     matching ``StabilityAdapter`` / ``AbundanceAdapter``), plus the MegaScale 101-group
     ``S -> S2`` panel reproducing the frozen 33-arm counts ``12->4 / 12->4 / 3->0``.

4. **Two Complementary Analytical Dimensions**:
   - **Dimension 1 (Quantitative Error)**: within-family site-crossfitted affine
     ``calibrated_error`` MSE vs. uncalibrated cross-family ``transfer_error`` MSE on
     BOTH MegaScale and Domainome, with 10,000-draw paired group bootstraps
     (native simultaneous, studentized ``maxT``, marginal 95% CIs) for ``+M|C``,
     ``+R|C``, ``+(M+R)|C``, ``+R|(C+M)``, ``C|M``, ``C|R``, ``R - M``,
     ``calibration_advantage = delta_MSE_cal - delta_MSE_trans``, and absolute level
     gap ``MSE_trans - MSE_cal``.
   - **Dimension 2 (Spearman Ranking & Multi-Scale Divergence Curves)**: Zero-shot
     ``M`` vs. explicit baselines, CV Spearman ``rho`` across all 6 designs, and
     per-ladder log-linear scaling slopes & ultra-scale inverted-U diagnostics across
     all 13 model ladders + key singletons.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from src.capability.baselines.cohorts import SAMPLE_SEPARATOR
from src.capability.core.io import sha256_file
from src.capability.extensions import bidirectional as bidir
from src.capability.extensions import explicit_statistics_lanes as lanes
from src.capability.extensions.stability_followups import COHORT_SHA256, verify_scalar_archive
from src.capability.general_plm.eval_utils import atomic_write_json, safe_spearman
from src.capability.interactions.pairwise_epistasis import (
    ALPHAS,
    INNER_SPLITS,
    OUTER_SPLITS,
    SPLIT_SEEDS,
    _select_alpha,
    group_errors,
    group_spearman,
)
from src.capability.mutation import external_confirmation as abundance_mod
from src.capability.readouts.readout_analysis import family_folds, row_weights
from src.capability.stability import gate_inputs, ranking_error as metric, stability_gate as gate
from src.capability.unified.roster import (
    ARM_METADATA,
    PREREGISTERED_33_ARMS,
    REPO_ROOT,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_OUTPUT_DIR,
    resolve_gp_r2_arrays,
)

SCHEMA = "r2_biophysical_unified_62_v1"
ARM_CACHE_SCHEMA = "r2_exp21_arm_cache_v1"

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261106
# 10x tighter than fit_stability_ranking_panel.py's 1e-9 baseline gate, while
# accommodating ~5e-12 float64 BLAS thread-count accumulation differences.
REPRODUCTION_TOLERANCE = 1e-10

ALPHAS_17: tuple[float, ...] = tuple(float(a) for a in np.logspace(-2, 6, 17))
ALPHAS_5: tuple[float, ...] = tuple(float(a) for a in ALPHAS)

ENDPOINTS: tuple[str, ...] = ("ranking", "calibrated_error", "transfer_error")
SIX_DESIGNS: tuple[str, ...] = ("C", "M", "R", "C_M", "C_R", "C_M_R")

# Exact frozen MegaScale designs (column order matches stability_gate / explicit_statistics_lanes).
MS_DESIGN_S: tuple[str, ...] = ("ident", "geom", "chem", "G")
MS_DESIGN_S2: tuple[str, ...] = ("ident", "geom", "comp", "chem", "prof2", "G")
MS_PROGRESSION_RUNGS: dict[str, tuple[str, ...]] = {
    "nuisance": ("ident", "geom", "comp", "chem", "G"),
    "nuisance+profile": ("ident", "geom", "comp", "chem", "prof2", "G"),
    "nuisance+profile+couplings+gemme": (
        "ident",
        "geom",
        "comp",
        "chem",
        "prof2",
        "G",
        "POTTS",
        "GEMME",
    ),
}

# Exact Domainome full-nuisance progression (uniform across all 62 models, no arm-specific T).
DM_FIRST_STAGE: tuple[str, ...] = ("ident", "geom", "comp", "chem", "prof", "prof2")
DM_PROGRESSION_RUNGS: dict[str, tuple[str, ...]] = {
    "nuisance": ("ident", "geom", "comp", "chem", "G"),
    "nuisance+profile": ("ident", "geom", "comp", "chem", "prof", "prof2", "G"),
    "nuisance+profile+couplings+gemme": (
        "ident",
        "geom",
        "comp",
        "chem",
        "prof",
        "prof2",
        "G",
        "POTTS",
        "GEMME",
    ),
}

EXPANDED_29_ARMS: tuple[str, ...] = tuple(
    arm_id
    for arm_id in UNIFIED_62_SINGLE_SEQUENCE_ARMS
    if ARM_METADATA[arm_id].origin_cohort == "expanded_32"
)
PREREGISTERED_33_SET = frozenset(PREREGISTERED_33_ARMS)

# Relative paths to frozen immutable inputs.
FROZEN_MS_GATE_DIR = Path("results/R4/gate_stability_20260924")
FROZEN_MS_101_PANEL_DIR = Path(
    "results/external_baseline/20261008095614_7e8ae4b18a4d/stability_ranking_error_grid_cpu"
)
FROZEN_MS_SCALAR_DIR = Path("results/external_baseline/20261008022745_762304146fdd")
FROZEN_MS_79_CELLS_DIR = Path("results/extensions/explicit_statistics_lanes_20261110/stability/cells")
FROZEN_MS_POTTS = Path("results/extensions/explicit_statistics_pairwise_20261109/potts/scores.json")
FROZEN_MS_GEMME = Path("results/extensions/explicit_statistics_pairwise_20261109/gemme/scores.json")

FROZEN_DM_DIR = Path("results/R1/external_confirmation_20260924")
FROZEN_DM_EXTRACTION_DIR = Path("results/external_baseline/20260924085613_5f8d32cf0368")
FROZEN_DM_94_CELLS_DIR = Path("results/extensions/explicit_statistics_lanes_20261110/abundance/cells")
FROZEN_DM_POTTS = Path("results/extensions/explicit_statistics_abundance_20261109/potts/scores.json")
FROZEN_DM_GEMME = Path("results/extensions/explicit_statistics_abundance_20261109/gemme/scores.json")

DEFAULT_R_ARCHIVE_DIR = Path(UNIFIED_OUTPUT_DIR) / "r2_readout_archives"

# Multi-point scaling ladders + highlighted singletons for Dimension 2 divergence curves.
SCALING_LADDERS: dict[str, tuple[str, ...]] = {
    "ESM2": ("esm2-8m", "esm2-35m", "esm2-150m", "esm2-650m", "esm2-3b", "esm2-15b"),
    "ESMC": ("esmc-300m", "esmc-600m", "esmc-6b"),
    "ProteinGLM-MLM": ("proteinglm-1b-mlm", "proteinglm-3b-mlm", "proteinglm-10b-mlm"),
    "DPLM": ("dplm-150m", "dplm-650m", "dplm-3b"),
    "SaProt": ("saprot-35m", "saprot-650m", "saprot-1.3b"),
    "AMPLIFY": ("amplify-120m", "amplify-350m"),
    "ProGen2": ("progen2-small", "progen2-base", "progen2-medium", "progen2-large", "progen2-xlarge"),
    "ProGen3": ("progen3-112m", "progen3-219m", "progen3-339m", "progen3-762m", "progen3-1b", "progen3-3b"),
    "ProtGPT3": ("protgpt3-112m", "protgpt3-1.3b", "protgpt3-10b"),
    "Galactica": ("galactica-125m", "galactica-1.3b", "galactica-6.7b", "galactica-30b"),
    "Qwen2.5": ("qwen2.5-0.5b", "qwen2.5-7b", "qwen2.5-32b"),
    "GPT-2": ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"),
    "ByGPT5": ("bygpt5-small-en", "bygpt5-base-en", "bygpt5-medium-en"),
}
HIGHLIGHTED_SINGLETONS: tuple[str, ...] = ("prime-690m", "esm3-1.4b", "dayhoff-170m")


# --------------------------------------------------------------------------- #
# Shared statistical banding (native simultaneous, studentized maxT, marginal)
# --------------------------------------------------------------------------- #


def paired_bands_summary(
    matrix: np.ndarray,
    columns: list[str] | tuple[str, ...],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = bidir.CONFIDENCE,
) -> dict[str, Any]:
    """Compute native simultaneous, studentized maxT, and marginal 95% CIs on shared draws."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ValueError(f"Expected matrix shape (n_groups, {len(columns)}), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite entries in bootstrap contrast matrix")

    n_groups = int(values.shape[0])
    native = bidir.paired_bands(values, draws=draws, seed=seed, confidence=confidence)
    resampled = bidir.resample_means(values, draws=draws, seed=seed)
    points = values.mean(axis=0)

    alpha = 0.5 * (1.0 - confidence)
    marg_low = np.quantile(resampled, alpha, axis=0)
    marg_high = np.quantile(resampled, 1.0 - alpha, axis=0)
    marginal_interval = np.column_stack([marg_low, marg_high]).tolist()

    se = values.std(axis=0, ddof=1) / math.sqrt(n_groups) if n_groups > 1 else np.zeros(values.shape[1])
    positive_se = se > 1e-15
    if positive_se.any():
        t_dev = np.abs(resampled[:, positive_se] - points[positive_se]) / se[positive_se]
        t_crit = float(np.quantile(np.max(t_dev, axis=1), confidence))
    else:
        t_crit = 0.0
    stud_low = points - t_crit * se
    stud_high = points + t_crit * se
    studentized_interval = np.column_stack([stud_low, stud_high]).tolist()

    v_nat = bidir.verdict(native["interval"])
    v_stud = bidir.verdict(studentized_interval)
    v_marg = bidir.verdict(marginal_interval)

    per_col: dict[str, Any] = {}
    for idx, name in enumerate(columns):
        per_col[name] = {
            "point": float(points[idx]),
            "se": float(se[idx]),
            "native_simultaneous_95": [float(native["interval"][idx][0]), float(native["interval"][idx][1])],
            "studentized_maxT_95": [float(studentized_interval[idx][0]), float(studentized_interval[idx][1])],
            "marginal_95": [float(marginal_interval[idx][0]), float(marginal_interval[idx][1])],
            "verdict_native": v_nat[idx],
            "verdict_studentized": v_stud[idx],
            "verdict_marginal": v_marg[idx],
        }

    return {
        "n_groups": n_groups,
        "family_size": len(columns),
        "draws": draws,
        "seed": seed,
        "confidence": confidence,
        "native_critical_value": float(native["critical_value"]),
        "studentized_critical_t": float(t_crit),
        "counts": {
            "native": bidir.counts(v_nat),
            "studentized": bidir.counts(v_stud),
            "marginal": bidir.counts(v_marg),
        },
        "positive_columns": {
            "native": [c for c, v in zip(columns, v_nat) if v == "positive"],
            "studentized": [c for c, v in zip(columns, v_stud) if v == "positive"],
            "marginal": [c for c, v in zip(columns, v_marg) if v == "positive"],
        },
        "columns": per_col,
    }


def _stack_cols(by_col: dict[str, np.ndarray], order: tuple[str, ...] | list[str]) -> np.ndarray:
    return np.column_stack([np.asarray(by_col[k], dtype=np.float64) for k in order])


# --------------------------------------------------------------------------- #
# Exact Chan parallel-variance centered Gram-Cholesky / eigh Ridge solver
# --------------------------------------------------------------------------- #


def _solve_ridge_from_gram(
    w_sum: torch.Tensor,
    mean_c: torch.Tensor,
    var: torch.Tensor,
    wxx: torch.Tensor,
    wy: torch.Tensor,
    wxy: torch.Tensor,
    alphas: tuple[float, ...] | list[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact float64 weighted Ridge solve from centered sufficient statistics.

    Uses Chan's non-negative parallel variance ``var`` so columns that are
    constant on the inner training chunks (even when non-constant on the held-out
    validation chunk) yield exact ``scale = 0.0`` without subtraction cancellation.
    Uses Cholesky solves for small alpha grids (<= 6) and a single symmetric
    ``eigh`` for larger alpha grids (> 6), matching ``ridge_predict`` to < 1e-12.
    """
    ym_c = wy / w_sum
    sxx = wxx / w_sum - torch.outer(mean_c, mean_c)
    scale = var.clamp_min(0.0).sqrt()
    const_mask = scale < 1e-12
    safe_scale = torch.where(const_mask, torch.ones_like(scale), scale)
    inv_scale = 1.0 / safe_scale
    cov = sxx * torch.outer(inv_scale, inv_scale)
    rhs_vec = (wxy / w_sum - mean_c * ym_c) * inv_scale
    if bool(const_mask.any()):
        cov = cov.clone()
        cov[const_mask, :] = 0.0
        cov[:, const_mask] = 0.0
        rhs_vec = rhs_vec.clone()
        rhs_vec[const_mask] = 0.0
    rhs = rhs_vec[:, None]

    alpha_vals = [float(a) for a in alphas]
    if any(a <= 0.0 for a in alpha_vals):
        raise ValueError("ridge alpha must be positive")

    if len(alpha_vals) > 6:
        vals, vecs = torch.linalg.eigh(cov)
        alpha_t = torch.tensor(alpha_vals, device=cov.device, dtype=torch.float64)
        beta = vecs @ ((vecs.T @ rhs) / (vals.clamp_min(0.0)[:, None] + alpha_t[None, :]))
        return beta, mean_c, inv_scale, ym_c

    p = cov.shape[0]
    diag_idx = torch.arange(p, device=cov.device)
    betas: list[torch.Tensor] = []
    eigh_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    for a_f in alpha_vals:
        mat = cov.clone()
        mat[diag_idx, diag_idx] += a_f
        chol, info = torch.linalg.cholesky_ex(mat)
        if int(info.item()) == 0:
            betas.append(torch.cholesky_solve(rhs, chol)[:, 0])
        else:
            if eigh_cache is None:
                vals, vecs = torch.linalg.eigh(cov)
                eigh_cache = (vals.clamp_min(0.0), vecs)
            vals_pos, vecs = eigh_cache
            proj = vecs.T @ rhs
            betas.append((vecs @ (proj / (vals_pos[:, None] + a_f)))[:, 0])
    return torch.stack(betas, dim=1), mean_c, inv_scale, ym_c


def _select_alpha_index(losses: np.ndarray, *, tie_largest: bool) -> int:
    if tie_largest:
        # Matches nested_group_ridge_eval (largest alpha on exact tie).
        return len(losses) - 1 - int(np.argmin(losses[::-1]))
    return int(_select_alpha(losses))


def build_fold_cache(
    *,
    groups: np.ndarray,
    weights: np.ndarray,
    g_builder=None,
) -> dict[int, list[dict[str, Any]]]:
    """Precompute outer/inner held-group partitions and optional fold-wise ``G`` blocks."""
    cache: dict[int, list[dict[str, Any]]] = {}
    for seed in SPLIT_SEEDS:
        seed_folds: list[dict[str, Any]] = []
        for outer, held in enumerate(family_folds(groups, OUTER_SPLITS, seed)):
            held_list = list(held)
            train = np.flatnonzero(~np.isin(groups, held_list))
            test = np.flatnonzero(np.isin(groups, held_list))
            inner_partition = family_folds(groups[train], INNER_SPLITS, seed + 100 + outer)
            inner_splits: list[tuple[np.ndarray, np.ndarray, int]] = []
            for val_groups in inner_partition:
                is_val = np.isin(groups[train], val_groups)
                fit_idx = train[~is_val]
                val_idx = train[is_val]
                inner_splits.append((fit_idx, val_idx, len(val_groups)))
            g_block = g_builder(held_list, inner_partition) if g_builder is not None else None
            seed_folds.append(
                {
                    "outer": outer,
                    "held": held_list,
                    "train": train,
                    "test": test,
                    "n_train_groups": len(set(groups[train].tolist())),
                    "inner_splits": inner_splits,
                    "G": g_block,
                    "weights": weights,
                }
            )
        cache[int(seed)] = seed_folds
    return cache


def fit_designs_with_cache(
    target: np.ndarray,
    blocks: dict[str, np.ndarray],
    designs: dict[str, tuple[str, ...]],
    fold_cache: dict[int, list[dict[str, Any]]],
    *,
    alphas: tuple[float, ...] = ALPHAS_5,
    renorm_val_weights: bool = False,
    tie_largest_alpha: bool = False,
    device: str = "cpu",
) -> dict[int, dict[str, np.ndarray]]:
    """Run nested held-group Ridge over ``SPLIT_SEEDS`` using exact outer-fold
    centered Gram sufficient statistics + Chan parallel variance across all designs.
    """
    target_np = np.asarray(target, dtype=np.float64)
    n_rows = len(target_np)
    oof_by_seed: dict[int, dict[str, np.ndarray]] = {}

    needed_blocks: list[str] = []
    for cols in designs.values():
        for c in cols:
            if c not in needed_blocks:
                needed_blocks.append(c)

    for seed in SPLIT_SEEDS:
        preds = {name: np.full(n_rows, np.nan, dtype=np.float64) for name in designs}
        for fold in fold_cache[int(seed)]:
            train = fold["train"]
            test = fold["test"]
            weights_np = fold["weights"]
            n_train_groups = fold["n_train_groups"]
            fold_blocks = dict(blocks)
            if fold["G"] is not None:
                fold_blocks["G"] = fold["G"]

            block_arrays: list[np.ndarray] = []
            block_slices: dict[str, tuple[int, int]] = {}
            offset = 0
            for key in needed_blocks:
                arr = np.asarray(fold_blocks[key], dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr[:, None]
                if not np.isfinite(arr).all():
                    raise ValueError(f"Block {key}: non-finite values")
                block_arrays.append(arr)
                block_slices[key] = (offset, offset + arr.shape[1])
                offset += arr.shape[1]

            x_full = torch.as_tensor(np.concatenate(block_arrays, axis=1), device=device, dtype=torch.float64)
            y_full = torch.as_tensor(target_np, device=device, dtype=torch.float64)
            w_full = torch.as_tensor(weights_np, device=device, dtype=torch.float64)

            tr_t = torch.as_tensor(train, device=device, dtype=torch.long)
            te_t = torch.as_tensor(test, device=device, dtype=torch.long)
            w_tr = w_full[tr_t]
            w_tot = w_tr.sum()
            mu0 = (w_tr[:, None] * x_full[tr_t]).sum(0) / w_tot
            y0 = (w_tr * y_full[tr_t]).sum() / w_tot
            xc_test = x_full[te_t] - mu0

            val_chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
            w_c: list[torch.Tensor] = []
            mu_c: list[torch.Tensor] = []
            var_c: list[torch.Tensor] = []
            wxx_c: list[torch.Tensor] = []
            wy_c: list[torch.Tensor] = []
            wxy_c: list[torch.Tensor] = []
            total_val_rows = 0
            for _, val_idx, n_val in fold["inner_splits"]:
                total_val_rows += len(val_idx)
                v_t = torch.as_tensor(val_idx, device=device, dtype=torch.long)
                x_v = x_full[v_t]
                y_v = y_full[v_t]
                w_v = w_full[v_t]
                wv_sum = w_v.sum()
                mu_v = (w_v[:, None] * x_v).sum(0) / wv_sum
                var_v = (w_v[:, None] * (x_v - mu_v).square()).sum(0) / wv_sum
                xc_v = x_v - mu0
                yc_v = y_v - y0
                w_xc_v = w_v[:, None] * xc_v
                w_v_loss = (w_v / wv_sum) if renorm_val_weights else w_v
                val_chunks.append((xc_v, y_v, w_v_loss, int(n_val)))
                w_c.append(wv_sum)
                mu_c.append(mu_v)
                var_c.append(var_v)
                wxx_c.append(xc_v.T @ w_xc_v)
                wy_c.append((w_v * yc_v).sum())
                wxy_c.append(w_xc_v.T @ yc_v)
            if total_val_rows != len(train):
                raise ValueError("Inner splits do not partition outer training rows")

            n_chunks = len(val_chunks)
            inner_fit_stats: list[
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ] = []
            for k in range(n_chunks):
                others = [j for j in range(n_chunks) if j != k]
                w_f = torch.stack([w_c[j] for j in others]).sum(0)
                mu_f = torch.stack([w_c[j] * mu_c[j] for j in others]).sum(0) / w_f
                var_f = (
                    torch.stack([w_c[j] * (var_c[j] + (mu_c[j] - mu_f).square()) for j in others]).sum(0)
                    / w_f
                )
                wxx_f = torch.stack([wxx_c[j] for j in others]).sum(0)
                wy_f = torch.stack([wy_c[j] for j in others]).sum(0)
                wxy_f = torch.stack([wxy_c[j] for j in others]).sum(0)
                inner_fit_stats.append((w_f, mu_f - mu0, var_f, wxx_f, wy_f, wxy_f))

            w_sum_tot = torch.stack(w_c).sum(0)
            mu_tot = torch.stack([w_c[j] * mu_c[j] for j in range(n_chunks)]).sum(0) / w_sum_tot
            var_tot = (
                torch.stack([w_c[j] * (var_c[j] + (mu_c[j] - mu_tot).square()) for j in range(n_chunks)]).sum(0)
                / w_sum_tot
            )
            mean_c_tot = mu_tot - mu0
            wxx_tot = torch.stack(wxx_c).sum(0)
            wy_tot = torch.stack(wy_c).sum(0)
            wxy_tot = torch.stack(wxy_c).sum(0)

            for name, cols in designs.items():
                col_indices = [i for c in cols for i in range(*block_slices[c])]
                c_idx = torch.as_tensor(col_indices, device=device, dtype=torch.long)

                losses = np.zeros(len(alphas), dtype=np.float64)
                for k, (xc_v, y_v, w_v_loss, n_val) in enumerate(val_chunks):
                    w_f, mean_c_f, var_f, wxx_f, wy_f, wxy_f = inner_fit_stats[k]
                    beta, mean_c, inv_scale, ym_c = _solve_ridge_from_gram(
                        w_f,
                        mean_c_f[c_idx],
                        var_f[c_idx],
                        wxx_f.index_select(0, c_idx).index_select(1, c_idx),
                        wy_f,
                        wxy_f[c_idx],
                        alphas,
                    )
                    pred_v = ((xc_v[:, c_idx] - mean_c) * inv_scale) @ beta + (y0 + ym_c)
                    err2 = (pred_v - y_v[:, None]).square()
                    losses += n_val * (w_v_loss[:, None] * err2).sum(0).cpu().numpy()

                alpha = alphas[_select_alpha_index(losses / n_train_groups, tie_largest=tie_largest_alpha)]
                beta_te, mean_te, inv_scale_te, ym_te = _solve_ridge_from_gram(
                    w_sum_tot,
                    mean_c_tot[c_idx],
                    var_tot[c_idx],
                    wxx_tot.index_select(0, c_idx).index_select(1, c_idx),
                    wy_tot,
                    wxy_tot[c_idx],
                    [alpha],
                )
                pred_te = ((xc_test[:, c_idx] - mean_te) * inv_scale_te) @ beta_te + (y0 + ym_te)
                preds[name][test] = pred_te[:, 0].cpu().numpy()

        for name, arr in preds.items():
            if not np.isfinite(arr).all():
                raise ValueError(f"Seed {seed} {name}: incomplete held-out predictions")
        oof_by_seed[int(seed)] = preds
    return oof_by_seed


# --------------------------------------------------------------------------- #
# Endpoint evaluation (ranking, calibrated_error, transfer_error)
# --------------------------------------------------------------------------- #


def evaluate_megascale_oof(
    pred: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    sites: np.ndarray,
    group_order: list[str],
) -> dict[str, np.ndarray]:
    """Compute per-group ranking, site-crossfitted calibrated MSE, and transfer MSE on MegaScale."""
    r_labels, r_vals = group_spearman(target, pred, groups)
    if list(r_labels) != group_order:
        raise ValueError("MegaScale ranking group order mismatch")
    t_labels, t_vals = group_errors(target, pred, groups, sites)
    if list(t_labels) != group_order:
        raise ValueError("MegaScale transfer_error group order mismatch")
    cal_pred, _ = metric.family_calibrated(pred, target, groups, sites)
    c_labels, c_vals = group_errors(target, cal_pred, groups, sites)
    if list(c_labels) != group_order:
        raise ValueError("MegaScale calibrated_error group order mismatch")
    return {
        "ranking": np.asarray(r_vals, dtype=np.float64),
        "calibrated_error": np.asarray(c_vals, dtype=np.float64),
        "transfer_error": np.asarray(t_vals, dtype=np.float64),
    }


def evaluate_domainome_oof(
    pred: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    domains: np.ndarray,
    sites: np.ndarray,
    group_order: list[str],
    *,
    use_nested_weights: bool,
) -> dict[str, np.ndarray]:
    """Compute per-group ranking, within-domain site-crossfitted calibrated MSE, and transfer MSE
    on Domainome.

    - ``use_nested_weights=True`` (``C_full`` / AbundanceAdapter): 3-level nested
      group -> domain -> site -> variant MSE and within-domain Spearman averaged into groups.
    - ``use_nested_weights=False`` (``C_evo`` / ``nested_group_ridge_eval`` parity):
      2-level site-equal group MSE (``row_weights(site, group)``), within-domain
      site-crossfitted affine recalibrated MSE (same 2-level weights), plus BOTH
      within-domain-averaged Spearman (``ranking``) and pooled-within-group Spearman
      (``ranking_pooled_group``, matching ``nested_group_ridge_eval``).
    """
    # Within-domain site-crossfitted affine recalibration (each domain is its own aPCA assay).
    cal_pred, _ = metric.family_calibrated(pred, target, domains, sites)

    # Within-domain Spearman averaged over domains inside each group.
    d_labels, d_rhos = group_spearman(target, pred, domains)
    per_domain_rho = dict(zip(d_labels, d_rhos))
    domain_to_group = dict(zip(domains.tolist(), groups.tolist()))
    buckets: dict[str, list[float]] = {g: [] for g in group_order}
    for d_name, rho_val in per_domain_rho.items():
        if rho_val is not None and np.isfinite(rho_val):
            buckets[domain_to_group[d_name]].append(float(rho_val))
    ranking_dom = np.asarray(
        [float(np.mean(buckets[g])) if buckets[g] else np.nan for g in group_order],
        dtype=np.float64,
    )

    if use_nested_weights:
        t_labels, t_vals = abundance_mod.group_errors(target, pred, groups, domains, sites)
        c_labels, c_vals = abundance_mod.group_errors(target, cal_pred, groups, domains, sites)
        if list(t_labels) != group_order or list(c_labels) != group_order:
            raise ValueError("Domainome nested group_errors order mismatch")
        return {
            "ranking": ranking_dom,
            "calibrated_error": np.asarray(c_vals, dtype=np.float64),
            "transfer_error": np.asarray(t_vals, dtype=np.float64),
        }

    # 2-level row_weights(site, group) matching nested_group_ridge_eval.
    w_site = row_weights(sites, groups)
    t_vals_list: list[float] = []
    c_vals_list: list[float] = []
    r_pooled_list: list[float] = []
    for g in group_order:
        m = groups == g
        wg = w_site[m]
        wg_sum = float(np.sum(wg))
        t_vals_list.append(float(np.sum(wg * (target[m] - pred[m]) ** 2) / wg_sum))
        c_vals_list.append(float(np.sum(wg * (target[m] - cal_pred[m]) ** 2) / wg_sum))
        rho_p = safe_spearman(pred[m], target[m])
        r_pooled_list.append(np.nan if rho_p is None else float(rho_p))
    return {
        "ranking": ranking_dom,
        "ranking_pooled_group": np.asarray(r_pooled_list, dtype=np.float64),
        "calibrated_error": np.asarray(c_vals_list, dtype=np.float64),
        "transfer_error": np.asarray(t_vals_list, dtype=np.float64),
    }


def collapse_oof(
    oof_by_seed: dict[int, dict[str, np.ndarray]],
    eval_fn,
) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate every seed and average per-group vectors across SPLIT_SEEDS."""
    designs = list(next(iter(oof_by_seed.values())).keys())
    accum: dict[str, dict[str, list[np.ndarray]]] = {}
    for seed in SPLIT_SEEDS:
        for d in designs:
            res = eval_fn(oof_by_seed[int(seed)][d])
            for ep, vec in res.items():
                accum.setdefault(ep, {}).setdefault(d, []).append(vec)
    return {
        ep: {d: np.mean(np.stack(vs, axis=0), axis=0) for d, vs in by_d.items()}
        for ep, by_d in accum.items()
    }


# --------------------------------------------------------------------------- #
# Support loaders (MegaScale 101g/79g & Domainome 96g/94g) + Instrument Anchor
# --------------------------------------------------------------------------- #


def _load_potts_gemme_maps(potts_path: Path, gemme_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    potts_doc = json.loads(potts_path.read_text(encoding="utf-8"))
    gemme_doc = json.loads(gemme_path.read_text(encoding="utf-8"))
    potts_map = {
        row["sample_id"]: float(row["potts_delta"])
        for row in potts_doc["samples"]
        if row.get("status") == "fitted" and row.get("potts_delta") is not None
    }
    gemme_map = {
        row["sample_id"]: float(row["gemme_score"])
        for row in gemme_doc["samples"]
        if row.get("status") == "scored" and row.get("gemme_score") is not None
    }
    return potts_map, gemme_map


def load_megascale_supports(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Load MegaScale 101-group (full) and 79-group (CC) panels plus dual-protease anchor."""
    paths = gate_inputs.resolve(root / FROZEN_MS_GATE_DIR)
    if sha256_file(paths["cohort.json"]) != COHORT_SHA256:
        raise ValueError("Frozen MegaScale cohort SHA-256 mismatch")
    cohort_101 = json.loads(paths["cohort.json"].read_text(encoding="utf-8"))
    profiles, _ = gate.load_profiles(
        paths["profile_features.npz"],
        {row["name"]: row["wildtype"] for row in cohort_101["backgrounds"]},
    )
    panel_101 = gate.build_panel(cohort_101, profiles)
    groups_101 = sorted(set(panel_101["group"].tolist()))
    if (len(panel_101["target"]), len(groups_101)) != (25856, 101):
        raise ValueError(f"Unexpected MegaScale 101g shape: ({len(panel_101['target'])}, {len(groups_101)})")

    # Raw 1-column Profile log-odds (index 2 of gate.profile_block = log f_mut - log f_wt).
    p_logodds_101 = np.asarray(panel_101["blocks"]["prof"][:, 2], dtype=np.float64)

    export_keys: list[str] = []
    row_bg_names: list[str] = []
    subunits_101: list[str] = []
    ddg_trypsin: list[float] = []
    ddg_chymo: list[float] = []
    for bg in cohort_101["backgrounds"]:
        wt = bg["wildtype"]
        for v in bg["variants"]:
            pos0 = int(v["position"]) - 1
            mut = f"{wt[pos0]}{v['position']}{v['mutant']}"
            export_keys.append(f"{bg['name']}{SAMPLE_SEPARATOR}{mut}")
            row_bg_names.append(bg["name"])
            subunits_101.append(f"{bg['name']}:{pos0}")
            ddg_trypsin.append(float(v["ddg_trypsin"]))
            ddg_chymo.append(float(v["ddg_chymotrypsin"]))

    potts_map, gemme_map = _load_potts_gemme_maps(root / FROZEN_MS_POTTS, root / FROZEN_MS_GEMME)
    retained_keys = set(potts_map) & set(gemme_map)
    kept_bgs = {bg_name for bg_name, k in zip(row_bg_names, export_keys) if k in retained_keys}
    lost_bgs = {bg_name for bg_name, k in zip(row_bg_names, export_keys) if k not in retained_keys}
    if kept_bgs & lost_bgs:
        raise ValueError("Partial background loss in MegaScale complete-case support")

    mask_79 = np.asarray([bg_name in kept_bgs for bg_name in row_bg_names], dtype=bool)
    cohort_79 = dict(
        cohort_101,
        backgrounds=[row for row in cohort_101["backgrounds"] if row["name"] in kept_bgs],
    )
    panel_79 = gate.build_panel(cohort_79, profiles)
    groups_79 = sorted(set(panel_79["group"].tolist()))
    if (len(panel_79["target"]), len(groups_79)) != (20224, 79):
        raise ValueError(f"Unexpected MegaScale 79g shape: ({len(panel_79['target'])}, {len(groups_79)})")

    potts_raw_79 = np.asarray(
        [potts_map[k] for k, keep in zip(export_keys, mask_79) if keep], dtype=np.float64
    )
    gemme_raw_79 = np.asarray(
        [gemme_map[k] for k, keep in zip(export_keys, mask_79) if keep], dtype=np.float64
    )
    p_logodds_79 = p_logodds_101[mask_79]

    # Full-nuisance blocks use reranked POTTS & GEMME matching explicit_statistics_lanes.
    panel_79["blocks"]["POTTS"] = lanes.prepare_scalar(potts_raw_79, panel_79["group"])
    panel_79["blocks"]["GEMME"] = lanes.prepare_scalar(gemme_raw_79, panel_79["group"])

    # Dual-protease instrument anchor from frozen endpoint_qualification.json + cohort variants.
    eq_path = root / FROZEN_MS_GATE_DIR / "endpoint/endpoint_qualification.json"
    eq_doc = json.loads(eq_path.read_text(encoding="utf-8"))
    tr_arr = np.asarray(ddg_trypsin, dtype=np.float64)
    ch_arr = np.asarray(ddg_chymo, dtype=np.float64)
    instrument_anchor = {
        "source_path": str(eq_path.relative_to(root)),
        "source_sha256": sha256_file(eq_path),
        "endpoint_qualification_all_backgrounds": eq_doc["agreement"]["all"]["background"]["agreement"],
        "endpoint_qualification_natural_backgrounds": eq_doc["agreement"]["natural"]["background"]["agreement"],
        "cohort_101_pooled_trypsin_vs_chymotrypsin": {
            "n_variants": int(len(tr_arr)),
            "n_groups": 101,
            "pearson_r": float(pearsonr(tr_arr, ch_arr)[0]),
            "spearman_rho": float(spearmanr(tr_arr, ch_arr)[0]),
            "rms_difference_kcal_mol": float(np.sqrt(np.mean((tr_arr - ch_arr) ** 2))),
        },
        "cohort_79_pooled_trypsin_vs_chymotrypsin": {
            "n_variants": int(mask_79.sum()),
            "n_groups": 79,
            "pearson_r": float(pearsonr(tr_arr[mask_79], ch_arr[mask_79])[0]),
            "spearman_rho": float(spearmanr(tr_arr[mask_79], ch_arr[mask_79])[0]),
        },
    }

    return {
        "cohort_101": cohort_101,
        "panel_101": panel_101,
        "groups_101": [str(g) for g in groups_101],
        "subunits_101": np.asarray(subunits_101, dtype="U"),
        "p_logodds_101": p_logodds_101,
        "cohort_79": cohort_79,
        "panel_79": panel_79,
        "groups_79": [str(g) for g in groups_79],
        "subunits_79": np.asarray(subunits_101, dtype="U")[mask_79],
        "mask_79": mask_79,
        "p_logodds_79": p_logodds_79,
        "potts_raw_79": potts_raw_79,
        "gemme_raw_79": gemme_raw_79,
        "instrument_anchor": instrument_anchor,
    }


def load_domainome_supports(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Load Domainome 96-group (full, 109,568 vars) and 94-group (CC, 106,496 vars) panels."""
    cohort_path = root / FROZEN_DM_DIR / "cohort.json"
    prof_path = root / FROZEN_DM_DIR / "profile_features.npz"
    cohort_96 = json.loads(cohort_path.read_text(encoding="utf-8"))
    profiles, _ = gate.load_profiles(
        prof_path,
        {row["name"]: row["wildtype"] for row in cohort_96["domains"]},
    )
    panel_96 = abundance_mod.build_panel(cohort_96["domains"], profiles)
    groups_96 = sorted(set(panel_96["group"].tolist()))
    if (len(panel_96["target"]), len(groups_96)) != (109568, 96):
        raise ValueError(f"Unexpected Domainome 96g shape: ({len(panel_96['target'])}, {len(groups_96)})")

    p_logodds_96 = np.asarray(panel_96["blocks"]["prof"][:, 2], dtype=np.float64)

    export_keys: list[str] = []
    subunits_96: list[str] = []
    for dom in cohort_96["domains"]:
        wt = dom["wildtype"]
        for v in dom["variants"]:
            pos0 = int(v["position"]) - 1
            mut = f"{wt[pos0]}{v['position']}{v['mutant']}"
            export_keys.append(f"{dom['name']}{SAMPLE_SEPARATOR}{mut}")
            subunits_96.append(f"{dom['name']}:{pos0}")

    potts_map, gemme_map = _load_potts_gemme_maps(root / FROZEN_DM_POTTS, root / FROZEN_DM_GEMME)
    retained_keys = set(potts_map) & set(gemme_map)
    mask_94 = np.asarray([k in retained_keys for k in export_keys], dtype=bool)

    panel_94 = {
        key: val[mask_94]
        for key, val in panel_96.items()
        if key not in ("blocks", "weights")
    }
    panel_94["blocks"] = {k: v[mask_94] for k, v in panel_96["blocks"].items()}
    panel_94["weights"] = abundance_mod.nested_weights(
        panel_94["group"], panel_94["domain"], panel_94["site"]
    )
    groups_94 = sorted(set(panel_94["group"].tolist()))
    if (len(panel_94["target"]), len(groups_94)) != (106496, 94):
        raise ValueError(f"Unexpected Domainome 94g shape: ({len(panel_94['target'])}, {len(groups_94)})")

    potts_raw_94 = np.asarray(
        [potts_map[k] for k, keep in zip(export_keys, mask_94) if keep], dtype=np.float64
    )
    gemme_raw_94 = np.asarray(
        [gemme_map[k] for k, keep in zip(export_keys, mask_94) if keep], dtype=np.float64
    )
    p_logodds_94 = p_logodds_96[mask_94]

    panel_94["blocks"]["POTTS"] = lanes.prepare_scalar(potts_raw_94, panel_94["domain"])
    panel_94["blocks"]["GEMME"] = lanes.prepare_scalar(gemme_raw_94, panel_94["domain"])

    return {
        "cohort_96": cohort_96,
        "cohort_sha256": sha256_file(cohort_path),
        "panel_96": panel_96,
        "groups_96": [str(g) for g in groups_96],
        "subunits_96": np.asarray(subunits_96, dtype="U"),
        "p_logodds_96": p_logodds_96,
        "panel_94": panel_94,
        "groups_94": [str(g) for g in groups_94],
        "subunits_94": np.asarray(subunits_96, dtype="U")[mask_94],
        "mask_94": mask_94,
        "p_logodds_94": p_logodds_94,
        "potts_raw_94": potts_raw_94,
        "gemme_raw_94": gemme_raw_94,
    }


# --------------------------------------------------------------------------- #
# Full 62-model arm loaders (MegaScale & Domainome, local + H200 R archives)
# --------------------------------------------------------------------------- #


def _locate_ms_scalar_dir(arm_id: str, root: Path) -> Path:
    ext_root = root / FROZEN_MS_SCALAR_DIR
    matches = list(ext_root.rglob(f"manifest_{arm_id}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected 1 MegaScale scalar manifest for {arm_id}, found {len(matches)}")
    return matches[0].parent


def _load_external_R_archive(path: Path, expected_rows: int) -> tuple[np.ndarray, str] | tuple[None, None]:
    if not path.is_file():
        return None, None
    with np.load(path, allow_pickle=False) as z:
        r_key = "R" if "R" in z.files else ("representations" if "representations" in z.files else None)
        if r_key is None:
            raise ValueError(f"{path}: missing 'R' array (keys={z.files})")
        r_arr = np.asarray(z[r_key], dtype=np.float64)
    if r_arr.shape != (expected_rows, 1024):
        raise ValueError(f"{path}: expected R shape ({expected_rows}, 1024), got {r_arr.shape}")
    if not np.isfinite(r_arr).all():
        raise ValueError(f"{path}: non-finite entries in R")
    return r_arr, sha256_file(path)


def load_megascale_arm(
    arm_id: str,
    cohort_101: dict[str, Any],
    *,
    root: Path = REPO_ROOT,
    r_archive_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Load 25,856-row MegaScale ``M`` and (when available) ``(25856, 1024)`` ``R``."""
    r_dir = Path(r_archive_dir) if r_archive_dir is not None else (root / DEFAULT_R_ARCHIVE_DIR)
    expected_targets = np.asarray(
        [float(v["ddg"]) for bg in cohort_101["backgrounds"] for v in bg["variants"]],
        dtype=np.float64,
    )

    if arm_id in PREREGISTERED_33_SET:
        ext_dir = _locate_ms_scalar_dir(arm_id, root)
        scalar, _ = verify_scalar_archive(ext_dir, arm_id, cohort_101, COHORT_SHA256)
        m_arr = np.asarray(scalar, dtype=np.float64).reshape(-1)
        m_sha = sha256_file(ext_dir / f"manifest_{arm_id}.json")
        r_path = r_dir / f"r2_stability_R_{arm_id}.npz"
        r_arr, r_sha = _load_external_R_archive(r_path, len(expected_targets))
        return m_arr, r_arr, {
            "m_source": str((ext_dir / f"manifest_{arm_id}.json").relative_to(root)),
            "m_source_sha256": m_sha,
            "r_source": str(r_path.relative_to(root)) if r_arr is not None else None,
            "r_source_sha256": r_sha,
            "has_R": r_arr is not None,
        }

    art = resolve_gp_r2_arrays(arm_id, "stability", root=root)
    if arm_id == "esm3-1.4b":
        expected_keys = [
            (bg["name"], int(v["position"]) - 1, v["mutant"])
            for bg in cohort_101["backgrounds"]
            for v in bg["variants"]
        ]
        scores: list[float] = []
        obs_keys: list[tuple[str, int, str]] = []
        with art.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                obs_keys.append((row["group"], int(row["position_0based"]), row["mt_aa"]))
                scores.append(float(row["score_masked_marginal"]))
        if obs_keys != expected_keys:
            raise ValueError("esm3-1.4b MegaScale jsonl disagrees with frozen cohort.json")
        m_arr = np.asarray(scores, dtype=np.float64)
        m_sha = sha256_file(art.path)
        r_path = r_dir / "r2_stability_R_esm3-1.4b.npz"
        r_arr, r_sha = _load_external_R_archive(r_path, len(expected_targets))
        return m_arr, r_arr, {
            "m_source": str(art.path.relative_to(root)),
            "m_source_sha256": m_sha,
            "r_source": str(r_path.relative_to(root)) if r_arr is not None else None,
            "r_source_sha256": r_sha,
            "has_R": r_arr is not None,
        }

    with np.load(art.path, allow_pickle=False) as z:
        m_arr = np.asarray(z["score"], dtype=np.float64).reshape(-1)
        t_arr = np.asarray(z["target"], dtype=np.float64).reshape(-1)
        r_arr = np.asarray(z["R"], dtype=np.float64) if "R" in z.files else None
    if len(m_arr) != len(expected_targets) or not np.allclose(t_arr, expected_targets, rtol=0, atol=1e-9):
        raise ValueError(f"{arm_id}: MegaScale .npz rows do not align with frozen cohort.json")
    if r_arr is not None and r_arr.shape != (len(expected_targets), 1024):
        raise ValueError(f"{arm_id}: unexpected MegaScale R shape {r_arr.shape}")
    src_sha = sha256_file(art.path)
    return m_arr, r_arr, {
        "m_source": str(art.path.relative_to(root)),
        "m_source_sha256": src_sha,
        "r_source": str(art.path.relative_to(root)) if r_arr is not None else None,
        "r_source_sha256": src_sha if r_arr is not None else None,
        "has_R": r_arr is not None,
    }


def load_domainome_arm(
    arm_id: str,
    cohort_96: dict[str, Any],
    *,
    root: Path = REPO_ROOT,
    r_archive_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Load 109,568-row Domainome ``M`` and (when available) ``(109568, 1024)`` ``R``."""
    r_dir = Path(r_archive_dir) if r_archive_dir is not None else (root / DEFAULT_R_ARCHIVE_DIR)
    expected_targets = np.asarray(
        [float(v["target"]) for d in cohort_96["domains"] for v in d["variants"]],
        dtype=np.float64,
    )

    if arm_id in PREREGISTERED_33_SET:
        arm_dir = root / FROZEN_DM_EXTRACTION_DIR / f"ec-{arm_id}"
        man_path = arm_dir / f"manifest_{arm_id}.json"
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        by_bg = {row["background"]: row for row in manifest["backgrounds"]}
        if len(by_bg) != len(cohort_96["domains"]):
            raise ValueError(f"{arm_id}: Domainome manifest has {len(by_bg)} domains, expected {len(cohort_96['domains'])}")
        m_chunks: list[np.ndarray] = []
        r_chunks: list[np.ndarray] = []
        for dom in cohort_96["domains"]:
            entry = by_bg[dom["name"]]
            with np.load(arm_dir / entry["file"], allow_pickle=False) as z:
                states = np.asarray(z["variant_states"], dtype=np.int64)
                positions = np.asarray(z["variant_positions"], dtype=np.int64)
                expected_states = np.arange(1, len(dom["variants"]) + 1, dtype=np.int64)
                expected_positions = np.asarray([int(v["position"]) for v in dom["variants"]], dtype=np.int64)
                if not np.array_equal(states, expected_states) or not np.array_equal(positions, expected_positions):
                    raise ValueError(f"{arm_id}/{dom['name']}: Domainome variant_states/positions mismatch")
                lik = np.asarray(z["likelihood"], dtype=np.float64)
                m_chunks.append(lik[states] - float(lik[0]))
                proj = np.asarray(z["projected"], dtype=np.float64)
                r_dom = (proj[states] - proj[0:1]).reshape(len(states), 1024)
                r_chunks.append(r_dom)
        m_arr = np.concatenate(m_chunks, axis=0)
        r_arr = np.concatenate(r_chunks, axis=0)
        if m_arr.shape != (len(expected_targets),) or r_arr.shape != (len(expected_targets), 1024):
            raise ValueError(f"{arm_id}: unexpected Domainome shapes {m_arr.shape}, {r_arr.shape}")
        if not np.isfinite(m_arr).all() or not np.isfinite(r_arr).all():
            raise ValueError(f"{arm_id}: non-finite values in Domainome archive")
        man_sha = sha256_file(man_path)
        return m_arr, r_arr, {
            "m_source": str(man_path.relative_to(root)),
            "m_source_sha256": man_sha,
            "r_source": str(man_path.relative_to(root)),
            "r_source_sha256": man_sha,
            "has_R": True,
        }

    art = resolve_gp_r2_arrays(arm_id, "abundance", root=root)
    if arm_id == "esm3-1.4b":
        expected_keys = [
            (d["name"], int(v["position"]) - 1, v["mutant"])
            for d in cohort_96["domains"]
            for v in d["variants"]
        ]
        scores: list[float] = []
        obs_keys: list[tuple[str, int, str]] = []
        with art.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                obs_keys.append((row["group"], int(row["position_0based"]), row["mt_aa"]))
                scores.append(float(row["score_masked_marginal"]))
        if obs_keys != expected_keys:
            raise ValueError("esm3-1.4b Domainome jsonl disagrees with frozen cohort.json")
        m_arr = np.asarray(scores, dtype=np.float64)
        m_sha = sha256_file(art.path)
        r_path = r_dir / "r2_abundance_R_esm3-1.4b.npz"
        r_arr, r_sha = _load_external_R_archive(r_path, len(expected_targets))
        return m_arr, r_arr, {
            "m_source": str(art.path.relative_to(root)),
            "m_source_sha256": m_sha,
            "r_source": str(r_path.relative_to(root)) if r_arr is not None else None,
            "r_source_sha256": r_sha,
            "has_R": r_arr is not None,
        }

    with np.load(art.path, allow_pickle=False) as z:
        m_arr = np.asarray(z["score"], dtype=np.float64).reshape(-1)
        t_arr = np.asarray(z["target"], dtype=np.float64).reshape(-1)
        r_arr = np.asarray(z["R"], dtype=np.float64) if "R" in z.files else None
    if len(m_arr) != len(expected_targets) or not np.allclose(t_arr, expected_targets, rtol=0, atol=1e-9):
        raise ValueError(f"{arm_id}: Domainome .npz rows do not align with frozen cohort.json")
    if r_arr is not None and r_arr.shape != (len(expected_targets), 1024):
        raise ValueError(f"{arm_id}: unexpected Domainome R shape {r_arr.shape}")
    src_sha = sha256_file(art.path)
    return m_arr, r_arr, {
        "m_source": str(art.path.relative_to(root)),
        "m_source_sha256": src_sha,
        "r_source": str(art.path.relative_to(root)) if r_arr is not None else None,
        "r_source_sha256": src_sha if r_arr is not None else None,
        "has_R": r_arr is not None,
    }


# --------------------------------------------------------------------------- #
# Verification gates (MegaScale 101g reproduction + ESMC-6B 79g C_evo parity)
# --------------------------------------------------------------------------- #


def run_megascale_reproduction_gate(
    ms_supports: dict[str, Any],
    fc_101_full: dict[int, list[dict[str, Any]]],
    *,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Verify < 1e-10 reproduction of frozen 101g ``progen2-medium`` OOF & historical 33-arm counts."""
    arm = "progen2-medium"
    panel_101 = ms_supports["panel_101"]
    groups_101 = ms_supports["groups_101"]
    m_101, _, _ = load_megascale_arm(arm, ms_supports["cohort_101"], root=root)
    blocks = {**panel_101["blocks"], "M": m_101[:, None]}
    designs = {
        "S": MS_DESIGN_S,
        "S_M": (*MS_DESIGN_S, "M"),
        "S2": MS_DESIGN_S2,
        "S2_M": (*MS_DESIGN_S2, "M"),
    }
    oof = fit_designs_with_cache(
        panel_101["target"], blocks, designs, fc_101_full, alphas=ALPHAS_5
    )

    frozen_dir = root / FROZEN_MS_101_PANEL_DIR
    with np.load(frozen_dir / f"oof/oof_{arm}.npz", allow_pickle=False) as frozen_oof:
        max_oof_diff = 0.0
        for seed in SPLIT_SEEDS:
            for d_name in designs:
                diff = float(np.max(np.abs(oof[int(seed)][d_name] - frozen_oof[f"{seed}|{d_name}"])))
                max_oof_diff = max(max_oof_diff, diff)

    collapsed = collapse_oof(
        oof,
        lambda p: evaluate_megascale_oof(
            p, panel_101["target"], panel_101["group"], panel_101["site"], groups_101
        ),
    )
    fit_doc = json.loads((frozen_dir / f"fits/fit_{arm}.json").read_text(encoding="utf-8"))
    max_contrast_diff = 0.0
    for baseline, d_base, d_joint in (("matched", "S", "S_M"), ("profile", "S2", "S2_M")):
        for ep, sign in (("ranking", 1.0), ("calibrated_error", -1.0), ("transfer_error", -1.0)):
            repro_inc = sign * (collapsed[ep][d_joint] - collapsed[ep][d_base])
            frozen_inc = np.mean(
                np.stack(
                    [
                        np.asarray(
                            fit_doc["seeds"][str(s)]["contrasts"]["combined"][f"{baseline}_{ep}"],
                            dtype=np.float64,
                        )
                        for s in SPLIT_SEEDS
                    ],
                    axis=0,
                ),
                axis=0,
            )
            max_contrast_diff = max(max_contrast_diff, float(np.max(np.abs(repro_inc - frozen_inc))))

    panel_doc = json.loads((frozen_dir / "panel.json").read_text(encoding="utf-8"))
    hist_counts = {
        k: int(v["above_zero"])
        for k, v in panel_doc["ranking_versus_error"]["resolved_counts"].items()
    }
    expected_hist = {
        "matched_ranking": 12,
        "profile_ranking": 4,
        "matched_calibrated_error": 12,
        "profile_calibrated_error": 4,
        "matched_transfer_error": 3,
        "profile_transfer_error": 0,
    }
    if max_oof_diff >= REPRODUCTION_TOLERANCE or max_contrast_diff >= REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"MegaScale 101g reproduction gate FAILED: max_oof_diff={max_oof_diff:g}, "
            f"max_contrast_diff={max_contrast_diff:g}"
        )
    if hist_counts != expected_hist:
        raise RuntimeError(f"Historical 33-arm counts mismatch: {hist_counts} != {expected_hist}")
    return {
        "passed": True,
        "arm": arm,
        "tolerance": REPRODUCTION_TOLERANCE,
        "max_oof_abs_diff": max_oof_diff,
        "max_contrast_abs_diff": max_contrast_diff,
        "historical_33_arm_studentized_counts": hist_counts,
        "frozen_panel_sha256": sha256_file(frozen_dir / "panel.json"),
    }


def run_esmc6b_evo_parity_gate(
    ms_supports: dict[str, Any],
    fc_79_evo: dict[int, list[dict[str, Any]]],
    *,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Verify exact parity of our Gram-Cholesky/eigh solver with ``r2_phenotypes_and_readouts_esmc-6b.json``
    (``POTTS+GEMME``: C=0.8076, M=0.8130, R=0.5030, C+M+R=0.4857 MSE & 0.7017 Spearman) and record
    the 3-column ``[Profile, Potts, GEMME]`` delta side-by-side.
    """
    art = resolve_gp_r2_arrays("esmc-6b", "stability", root=root)
    if art.sibling_json is None or not art.sibling_json.is_file():
        return {"passed": False, "reason": "sibling_json missing"}
    ref_doc = json.loads(art.sibling_json.read_text(encoding="utf-8"))
    ref_summary = ref_doc["lanes"]["stability"]["bidirectional_ridge"]["summary"]

    m_101, r_101, _ = load_megascale_arm("esmc-6b", ms_supports["cohort_101"], root=root)
    mask = ms_supports["mask_79"]
    panel_79 = ms_supports["panel_79"]
    groups_79 = ms_supports["groups_79"]

    blocks = {
        "P": ms_supports["p_logodds_79"][:, None],
        "POTTS": ms_supports["potts_raw_79"][:, None],
        "GEMME": ms_supports["gemme_raw_79"][:, None],
        "M": m_101[mask][:, None],
        "R": r_101[mask],
    }
    designs = {
        "pg_controls": ("POTTS", "GEMME"),
        "M": ("M",),
        "R": ("R",),
        "pg_controls_M": ("POTTS", "GEMME", "M"),
        "pg_controls_R": ("POTTS", "GEMME", "R"),
        "pg_controls_M_R": ("POTTS", "GEMME", "M", "R"),
        "ppg_controls": ("P", "POTTS", "GEMME"),
        "ppg_controls_M_R": ("P", "POTTS", "GEMME", "M", "R"),
    }
    oof = fit_designs_with_cache(
        panel_79["target"],
        blocks,
        designs,
        fc_79_evo,
        alphas=ALPHAS_17,
        renorm_val_weights=True,
        tie_largest_alpha=True,
    )
    collapsed = collapse_oof(
        oof,
        lambda p: evaluate_megascale_oof(
            p, panel_79["target"], panel_79["group"], panel_79["site"], groups_79
        ),
    )

    key_map = {
        "controls": "pg_controls",
        "M": "M",
        "R": "R",
        "controls_M": "pg_controls_M",
        "controls_R": "pg_controls_R",
        "controls_M_R": "pg_controls_M_R",
    }
    max_mse_diff = 0.0
    max_rho_diff = 0.0
    comparison: dict[str, Any] = {}
    for ref_k, our_k in key_map.items():
        our_mse = float(collapsed["transfer_error"][our_k].mean())
        our_rho = float(collapsed["ranking"][our_k].mean())
        ref_mse = float(ref_summary[ref_k]["group_mse"]["mean"])
        ref_rho = float(ref_summary[ref_k]["group_spearman"]["mean"])
        max_mse_diff = max(max_mse_diff, abs(our_mse - ref_mse))
        max_rho_diff = max(max_rho_diff, abs(our_rho - ref_rho))
        comparison[ref_k] = {
            "ref_mse": ref_mse,
            "our_mse": our_mse,
            "ref_spearman": ref_rho,
            "our_spearman": our_rho,
        }

    if max_mse_diff >= 1e-9 or max_rho_diff >= 1e-9:
        raise RuntimeError(
            f"ESMC-6B 79g C_evo parity gate FAILED: max_mse_diff={max_mse_diff:g}, max_rho_diff={max_rho_diff:g}"
        )
    return {
        "passed": True,
        "arm": "esmc-6b",
        "max_mse_abs_diff": max_mse_diff,
        "max_spearman_abs_diff": max_rho_diff,
        "potts_gemme_parity": comparison,
        "profile_potts_gemme_comparison": {
            "C_ppg_transfer_mse": float(collapsed["transfer_error"]["ppg_controls"].mean()),
            "C_ppg_calibrated_mse": float(collapsed["calibrated_error"]["ppg_controls"].mean()),
            "C_ppg_ranking": float(collapsed["ranking"]["ppg_controls"].mean()),
            "C_ppg_M_R_transfer_mse": float(collapsed["transfer_error"]["ppg_controls_M_R"].mean()),
            "C_ppg_M_R_calibrated_mse": float(collapsed["calibrated_error"]["ppg_controls_M_R"].mean()),
            "C_ppg_M_R_ranking": float(collapsed["ranking"]["ppg_controls_M_R"].mean()),
        },
        "note": (
            "Earlier r2_phenotypes_and_readouts.py matched '|frequencies' incorrectly and used "
            "2-col [Potts, GEMME]; unified primary_evolutionary_control includes all 3 columns "
            "[Profile, Potts, GEMME] while confirming exact < 1e-9 parity with the 2-col numbers."
        ),
    }


def load_preregistered_ms_101_readings(
    ms_supports: dict[str, Any],
    *,
    root: Path = REPO_ROOT,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Load exact frozen 101-group (S, S_M, S2, S2_M) readings for all 33 pre-registered arms."""
    frozen_dir = root / FROZEN_MS_101_PANEL_DIR
    groups_101 = ms_supports["groups_101"]
    out: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for arm in PREREGISTERED_33_ARMS:
        fit_doc = json.loads((frozen_dir / f"fits/fit_{arm}.json").read_text(encoding="utf-8"))
        per_ep: dict[str, dict[str, list[np.ndarray]]] = {
            ep: {"S": [], "S_M": [], "S2": [], "S2_M": []} for ep in ENDPOINTS
        }
        for seed in SPLIT_SEEDS:
            comb = fit_doc["seeds"][str(seed)]["contrasts"]["combined"]
            if list(comb["groups"]) != groups_101:
                raise ValueError(f"{arm} seed {seed}: frozen 101g order mismatch")
            for ep, b_key, sign in (
                ("ranking", "baseline_within_family_ranking", 1.0),
                ("calibrated_error", "baseline_calibrated_error", -1.0),
                ("transfer_error", "baseline_transfer_error", -1.0),
            ):
                s_base = np.asarray(comb[b_key]["matched"], dtype=np.float64)
                s_inc = np.asarray(comb[f"matched_{ep}"], dtype=np.float64)
                s2_base = np.asarray(comb[b_key]["profile"], dtype=np.float64)
                s2_inc = np.asarray(comb[f"profile_{ep}"], dtype=np.float64)
                per_ep[ep]["S"].append(s_base)
                per_ep[ep]["S_M"].append(s_base + sign * s_inc)
                per_ep[ep]["S2"].append(s2_base)
                per_ep[ep]["S2_M"].append(s2_base + sign * s2_inc)
        out[arm] = {
            ep: {d: np.mean(np.stack(vs, axis=0), axis=0) for d, vs in by_d.items()}
            for ep, by_d in per_ep.items()
        }
    return out


def load_preregistered_ms_79_cells(
    ms_supports: dict[str, Any],
    *,
    root: Path = REPO_ROOT,
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """Load exact frozen 79-group cells (U_C, M, U_C_M) for all 33 pre-registered arms."""
    cells_root = root / FROZEN_MS_79_CELLS_DIR
    groups_79 = ms_supports["groups_79"]
    out: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    for arm in PREREGISTERED_33_ARMS:
        arm_rec: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for rung in MS_PROGRESSION_RUNGS:
            rung_rec: dict[str, dict[str, np.ndarray]] = {}
            for ep in ENDPOINTS:
                path = cells_root / ep / "likelihood" / "complete_case" / rung / f"{arm}.json"
                doc = json.loads(path.read_text(encoding="utf-8"))
                if list(doc["groups"]) != groups_79:
                    raise ValueError(f"{path}: 79-group order mismatch")
                rung_rec[ep] = {k: np.asarray(v, dtype=np.float64) for k, v in doc["values"].items()}
            arm_rec[rung] = rung_rec
        out[arm] = arm_rec
    return out


# --------------------------------------------------------------------------- #
# Spawn-safe workers with per-arm disk caching (MegaScale & Domainome)
# --------------------------------------------------------------------------- #

_MS_WORKER_STATE: dict[str, Any] = {}
_DM_WORKER_STATE: dict[str, Any] = {}


def _init_ms_worker(shared: dict[str, Any]) -> None:
    _MS_WORKER_STATE.clear()
    _MS_WORKER_STATE.update(shared)
    torch.set_num_threads(int(shared.get("blas_threads", 2)))


def _fit_one_megascale_arm(arm_id: str) -> dict[str, Any]:
    """Fit/load all MegaScale designs (79g/101g C_evo + C_full) for one arm."""
    root = Path(_MS_WORKER_STATE["root"])
    cache_dir = Path(_MS_WORKER_STATE["cache_dir"])
    r_archive_dir = Path(_MS_WORKER_STATE["r_archive_dir"])
    force = bool(_MS_WORKER_STATE.get("force", False))
    cache_path = cache_dir / f"{arm_id}.json"

    m_101, r_101, meta = load_megascale_arm(
        arm_id, _MS_WORKER_STATE["cohort_101"], root=root, r_archive_dir=r_archive_dir
    )

    if not force and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("schema") == ARM_CACHE_SCHEMA
                and cached.get("dataset") == "megascale"
                and cached.get("arm_id") == arm_id
                and cached.get("m_source_sha256") == meta["m_source_sha256"]
                and cached.get("has_R") == meta["has_R"]
                and cached.get("r_source_sha256") == meta["r_source_sha256"]
            ):
                return cached
        except Exception:
            pass

    t0 = time.monotonic()
    mask_79 = _MS_WORKER_STATE["mask_79"]
    panel_101 = _MS_WORKER_STATE["panel_101"]
    groups_101 = _MS_WORKER_STATE["groups_101"]
    panel_79 = _MS_WORKER_STATE["panel_79"]
    groups_79 = _MS_WORKER_STATE["groups_79"]

    m_79 = m_101[mask_79]
    r_79 = r_101[mask_79] if r_101 is not None else None

    # Zero-shot within-family Spearman on 101g and 79g.
    _, zs_101 = group_spearman(panel_101["target"], m_101, panel_101["group"])
    _, zs_79 = group_spearman(panel_79["target"], m_79, panel_79["group"])

    eval_101 = lambda p: evaluate_megascale_oof(  # noqa: E731
        p, panel_101["target"], panel_101["group"], panel_101["site"], groups_101
    )
    eval_79 = lambda p: evaluate_megascale_oof(  # noqa: E731
        p, panel_79["target"], panel_79["group"], panel_79["site"], groups_79
    )

    # 1. Primary evolutionary control — 79g CC (C = [P, POTTS, GEMME], ALPHAS_17).
    blocks_79_evo: dict[str, np.ndarray] = {
        "P": _MS_WORKER_STATE["p_logodds_79"][:, None],
        "POTTS": _MS_WORKER_STATE["potts_raw_79"][:, None],
        "GEMME": _MS_WORKER_STATE["gemme_raw_79"][:, None],
        "M": m_79[:, None],
    }
    designs_79_evo: dict[str, tuple[str, ...]] = {
        "M": ("M",),
        "C_M": ("P", "POTTS", "GEMME", "M"),
    }
    if r_79 is not None:
        blocks_79_evo["R"] = r_79
        designs_79_evo["R"] = ("R",)
        designs_79_evo["C_R"] = ("P", "POTTS", "GEMME", "R")
        designs_79_evo["C_M_R"] = ("P", "POTTS", "GEMME", "M", "R")
    oof_79_evo = fit_designs_with_cache(
        panel_79["target"],
        blocks_79_evo,
        designs_79_evo,
        _MS_WORKER_STATE["fc_79_evo"],
        alphas=ALPHAS_17,
        renorm_val_weights=True,
        tie_largest_alpha=True,
    )
    r_79_evo = collapse_oof(oof_79_evo, eval_79)

    # 2. Primary evolutionary control — 101g Full (C = [P], ALPHAS_17).
    blocks_101_evo: dict[str, np.ndarray] = {
        "P": _MS_WORKER_STATE["p_logodds_101"][:, None],
        "M": m_101[:, None],
    }
    designs_101_evo: dict[str, tuple[str, ...]] = {
        "M": ("M",),
        "C_M": ("P", "M"),
    }
    if r_101 is not None:
        blocks_101_evo["R"] = r_101
        designs_101_evo["R"] = ("R",)
        designs_101_evo["C_R"] = ("P", "R")
        designs_101_evo["C_M_R"] = ("P", "M", "R")
    oof_101_evo = fit_designs_with_cache(
        panel_101["target"],
        blocks_101_evo,
        designs_101_evo,
        _MS_WORKER_STATE["fc_101_evo"],
        alphas=ALPHAS_17,
        renorm_val_weights=True,
        tie_largest_alpha=True,
    )
    r_101_evo = collapse_oof(oof_101_evo, eval_101)

    # 3. Full nuisance+evolutionary control — 79g CC (ALPHAS_5 + cached G).
    # For pre-registered 33 arms, M-only rungs are loaded from exact frozen cells,
    # so we only need to fit R / C_R / C_M_R when R is present.
    r_79_full: dict[str, dict[str, np.ndarray]] = {ep: {} for ep in ENDPOINTS}
    blocks_79_full: dict[str, np.ndarray] = {**panel_79["blocks"], "M": m_79[:, None]}
    designs_79_full: dict[str, tuple[str, ...]] = {}
    if arm_id not in PREREGISTERED_33_SET:
        designs_79_full["M"] = ("M",)
        for rung_name, ctrl_cols in MS_PROGRESSION_RUNGS.items():
            designs_79_full[f"{rung_name}|M"] = (*ctrl_cols, "M")
    if r_79 is not None:
        blocks_79_full["R"] = r_79
        designs_79_full["R"] = ("R",)
        ctrl_top = MS_PROGRESSION_RUNGS["nuisance+profile+couplings+gemme"]
        designs_79_full["nuisance+profile+couplings+gemme|R"] = (*ctrl_top, "R")
        designs_79_full["nuisance+profile+couplings+gemme|MR"] = (*ctrl_top, "M", "R")
    if designs_79_full:
        oof_79_full = fit_designs_with_cache(
            panel_79["target"],
            blocks_79_full,
            designs_79_full,
            _MS_WORKER_STATE["fc_79_full"],
            alphas=ALPHAS_5,
        )
        r_79_full = collapse_oof(oof_79_full, eval_79)

    # 4. Full nuisance control — 101g S -> S2 panel (ALPHAS_5 + cached G).
    r_101_full: dict[str, dict[str, np.ndarray]] = {ep: {} for ep in ENDPOINTS}
    blocks_101_full: dict[str, np.ndarray] = {**panel_101["blocks"], "M": m_101[:, None]}
    designs_101_full: dict[str, tuple[str, ...]] = {}
    if arm_id not in PREREGISTERED_33_SET:
        designs_101_full["M"] = ("M",)
        designs_101_full["S_M"] = (*MS_DESIGN_S, "M")
        designs_101_full["S2_M"] = (*MS_DESIGN_S2, "M")
    if r_101 is not None:
        blocks_101_full["R"] = r_101
        designs_101_full["R"] = ("R",)
        designs_101_full["S2_R"] = (*MS_DESIGN_S2, "R")
        designs_101_full["S2_MR"] = (*MS_DESIGN_S2, "M", "R")
    if designs_101_full:
        oof_101_full = fit_designs_with_cache(
            panel_101["target"],
            blocks_101_full,
            designs_101_full,
            _MS_WORKER_STATE["fc_101_full"],
            alphas=ALPHAS_5,
        )
        r_101_full = collapse_oof(oof_101_full, eval_101)

    record = {
        "schema": ARM_CACHE_SCHEMA,
        "dataset": "megascale",
        "arm_id": arm_id,
        "m_source": meta["m_source"],
        "m_source_sha256": meta["m_source_sha256"],
        "r_source": meta["r_source"],
        "r_source_sha256": meta["r_source_sha256"],
        "has_R": meta["has_R"],
        "elapsed_seconds": float(time.monotonic() - t0),
        "zero_shot_101": [float(v) for v in zs_101],
        "zero_shot_79": [float(v) for v in zs_79],
        "evo_79": {ep: {d: v.tolist() for d, v in by_d.items()} for ep, by_d in r_79_evo.items()},
        "evo_101": {ep: {d: v.tolist() for d, v in by_d.items()} for ep, by_d in r_101_evo.items()},
        "full_79": {ep: {d: v.tolist() for d, v in by_d.items()} for ep, by_d in r_79_full.items()},
        "full_101": {ep: {d: v.tolist() for d, v in by_d.items()} for ep, by_d in r_101_full.items()},
    }
    atomic_write_json(cache_path, record)
    return record


def _init_dm_worker(shared: dict[str, Any]) -> None:
    _DM_WORKER_STATE.clear()
    _DM_WORKER_STATE.update(shared)
    torch.set_num_threads(int(shared.get("blas_threads", 2)))


def _fit_one_domainome_arm(arm_id: str) -> dict[str, Any]:
    """Fit/load all Domainome designs (94g/96g C_evo + 94g C_full) for one arm."""
    root = Path(_DM_WORKER_STATE["root"])
    cache_dir = Path(_DM_WORKER_STATE["cache_dir"])
    r_archive_dir = Path(_DM_WORKER_STATE["r_archive_dir"])
    force = bool(_DM_WORKER_STATE.get("force", False))
    cache_path = cache_dir / f"{arm_id}.json"

    m_96, r_96, meta = load_domainome_arm(
        arm_id, _DM_WORKER_STATE["cohort_96"], root=root, r_archive_dir=r_archive_dir
    )

    if not force and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("schema") == ARM_CACHE_SCHEMA
                and cached.get("dataset") == "domainome"
                and cached.get("arm_id") == arm_id
                and cached.get("m_source_sha256") == meta["m_source_sha256"]
                and cached.get("has_R") == meta["has_R"]
                and cached.get("r_source_sha256") == meta["r_source_sha256"]
            ):
                return cached
        except Exception:
            pass

    t0 = time.monotonic()
    mask_94 = _DM_WORKER_STATE["mask_94"]
    panel_96 = _DM_WORKER_STATE["panel_96"]
    groups_96 = _DM_WORKER_STATE["groups_96"]
    panel_94 = _DM_WORKER_STATE["panel_94"]
    groups_94 = _DM_WORKER_STATE["groups_94"]

    m_94 = m_96[mask_94]
    r_94 = r_96[mask_94] if r_96 is not None else None

    # Zero-shot within-domain Spearman averaged into groups on 96g and 94g.
    zs_96 = abundance_mod._domain_spearman(panel_96, m_96)
    zs_94 = abundance_mod._domain_spearman(panel_94, m_94)

    eval_94_evo = lambda p: evaluate_domainome_oof(  # noqa: E731
        p,
        panel_94["target"],
        panel_94["group"],
        panel_94["domain"],
        panel_94["site"],
        groups_94,
        use_nested_weights=False,
    )
    eval_96_evo = lambda p: evaluate_domainome_oof(  # noqa: E731
        p,
        panel_96["target"],
        panel_96["group"],
        panel_96["domain"],
        panel_96["site"],
        groups_96,
        use_nested_weights=False,
    )
    eval_94_full = lambda p: evaluate_domainome_oof(  # noqa: E731
        p,
        panel_94["target"],
        panel_94["group"],
        panel_94["domain"],
        panel_94["site"],
        groups_94,
        use_nested_weights=True,
    )

    # 1. Primary evolutionary control — 94g CC (C = [P, POTTS, GEMME], ALPHAS_17).
    blocks_94_evo: dict[str, np.ndarray] = {
        "P": _DM_WORKER_STATE["p_logodds_94"][:, None],
        "POTTS": _DM_WORKER_STATE["potts_raw_94"][:, None],
        "GEMME": _DM_WORKER_STATE["gemme_raw_94"][:, None],
        "M": m_94[:, None],
    }
    designs_94_evo: dict[str, tuple[str, ...]] = {
        "M": ("M",),
        "C_M": ("P", "POTTS", "GEMME", "M"),
    }
    if r_94 is not None:
        blocks_94_evo["R"] = r_94
        designs_94_evo["R"] = ("R",)
        designs_94_evo["C_R"] = ("P", "POTTS", "GEMME", "R")
        designs_94_evo["C_M_R"] = ("P", "POTTS", "GEMME", "M", "R")
    oof_94_evo = fit_designs_with_cache(
        panel_94["target"],
        blocks_94_evo,
        designs_94_evo,
        _DM_WORKER_STATE["fc_94_evo"],
        alphas=ALPHAS_17,
        renorm_val_weights=True,
        tie_largest_alpha=True,
    )
    r_94_evo = collapse_oof(oof_94_evo, eval_94_evo)

    # 2. Primary evolutionary control — 96g Full (C = [P], ALPHAS_17).
    blocks_96_evo: dict[str, np.ndarray] = {
        "P": _DM_WORKER_STATE["p_logodds_96"][:, None],
        "M": m_96[:, None],
    }
    designs_96_evo: dict[str, tuple[str, ...]] = {
        "M": ("M",),
        "C_M": ("P", "M"),
    }
    if r_96 is not None:
        blocks_96_evo["R"] = r_96
        designs_96_evo["R"] = ("R",)
        designs_96_evo["C_R"] = ("P", "R")
        designs_96_evo["C_M_R"] = ("P", "M", "R")
    oof_96_evo = fit_designs_with_cache(
        panel_96["target"],
        blocks_96_evo,
        designs_96_evo,
        _DM_WORKER_STATE["fc_96_evo"],
        alphas=ALPHAS_17,
        renorm_val_weights=True,
        tie_largest_alpha=True,
    )
    r_96_evo = collapse_oof(oof_96_evo, eval_96_evo)

    # 3. Full nuisance+evolutionary control — 94g CC (ALPHAS_5 + cached G + nested_weights).
    # Fit uniformly across all 62 arms (no arm-dependent T block) so calibrated_error,
    # transfer_error, and ranking are apples-to-apples across all 62 models.
    blocks_94_full: dict[str, np.ndarray] = {**panel_94["blocks"], "M": m_94[:, None]}
    designs_94_full: dict[str, tuple[str, ...]] = {"M": ("M",)}
    for rung_name, ctrl_cols in DM_PROGRESSION_RUNGS.items():
        designs_94_full[f"{rung_name}|M"] = (*ctrl_cols, "M")
    if r_94 is not None:
        blocks_94_full["R"] = r_94
        designs_94_full["R"] = ("R",)
        ctrl_top = DM_PROGRESSION_RUNGS["nuisance+profile+couplings+gemme"]
        designs_94_full["nuisance+profile+couplings+gemme|R"] = (*ctrl_top, "R")
        designs_94_full["nuisance+profile+couplings+gemme|MR"] = (*ctrl_top, "M", "R")
    oof_94_full = fit_designs_with_cache(
        panel_94["target"],
        blocks_94_full,
        designs_94_full,
        _DM_WORKER_STATE["fc_94_full"],
        alphas=ALPHAS_5,
    )
    r_94_full = collapse_oof(oof_94_full, eval_94_full)

    record = {
        "schema": ARM_CACHE_SCHEMA,
        "dataset": "domainome",
        "arm_id": arm_id,
        "m_source": meta["m_source"],
        "m_source_sha256": meta["m_source_sha256"],
        "r_source": meta["r_source"],
        "r_source_sha256": meta["r_source_sha256"],
        "has_R": meta["has_R"],
        "elapsed_seconds": float(time.monotonic() - t0),
        "zero_shot_96": [float(v) for v in zs_96],
        "zero_shot_94": [float(v) for v in zs_94],
        "evo_94": {ep: {d: v.tolist() for d, v in by_d.items()} for ep, by_d in r_94_evo.items()},
        "evo_96": {ep: {d: v.tolist() for d, v in by_d.items()} for ep, by_d in r_96_evo.items()},
        "full_94": {ep: {d: v.tolist() for d, v in by_d.items()} for ep, by_d in r_94_full.items()},
    }
    atomic_write_json(cache_path, record)
    return record


# --------------------------------------------------------------------------- #
# Aggregation helpers (Zero-shot, Dimension 1 Error, Dimension 2 Ranking/Scaling)
# --------------------------------------------------------------------------- #


def _build_zero_shot_block(
    arm_vecs: dict[str, np.ndarray],
    explicit_vecs: dict[str, np.ndarray],
    arms_order: tuple[str, ...],
) -> dict[str, Any]:
    arm_mat = _stack_cols(arm_vecs, arms_order)
    arm_bands = paired_bands_summary(arm_mat, arms_order)

    exp_summary: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for exp_name, exp_vec in explicit_vecs.items():
        vec = np.asarray(exp_vec, dtype=np.float64)
        valid = np.isfinite(vec)
        if valid.all():
            eb = paired_bands_summary(vec[:, None], [exp_name])
            exp_summary[exp_name] = eb["columns"][exp_name]
            diff = arm_mat - vec[:, None]
            contrasts[f"model_minus_{exp_name}"] = paired_bands_summary(diff, arms_order)
        else:
            # Handle rare groups where Profile is constant/absent on full support without dropping them silently.
            sub_vec = vec[valid]
            eb = paired_bands_summary(sub_vec[:, None], [exp_name])
            exp_summary[exp_name] = {
                **eb["columns"][exp_name],
                "defined_groups": int(valid.sum()),
                "total_groups": int(len(vec)),
            }
            diff = arm_mat[valid] - sub_vec[:, None]
            contrasts[f"model_minus_{exp_name}"] = {
                **paired_bands_summary(diff, arms_order),
                "defined_groups": int(valid.sum()),
                "total_groups": int(len(vec)),
            }

    return {
        "n_groups": int(arm_mat.shape[0]),
        "n_arms": len(arms_order),
        "explicit_baselines": exp_summary,
        "arms_62": arm_bands,
        "paired_contrasts_vs_explicit": contrasts,
    }


def _paired_bands_defined(
    matrix: np.ndarray,
    columns: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Run ``paired_bands_summary`` on finite rows, recording any undefined-Spearman groups explicitly."""
    arr = np.asarray(matrix, dtype=np.float64)
    valid = np.all(np.isfinite(arr), axis=1)
    if valid.all():
        return paired_bands_summary(arr, columns)
    res = paired_bands_summary(arr[valid], columns)
    res["defined_groups"] = int(valid.sum())
    res["total_groups"] = int(len(valid))
    res["undefined_group_indices"] = np.flatnonzero(~valid).tolist()
    return res


def _build_six_design_block(
    base_C: dict[str, np.ndarray],
    per_arm_designs: dict[str, dict[str, dict[str, np.ndarray]]],
    arms_62: tuple[str, ...],
    arms_with_R: tuple[str, ...],
    *,
    include_pooled_group_ranking: bool = False,
) -> dict[str, Any]:
    """Assemble Dimension 1 (calibrated_error vs. transfer_error) and Dimension 2
    (ranking Spearman) across all 6 designs (C, M, R, C_M, C_R, C_M_R).
    """
    # Baseline C absolute levels
    c_levels: dict[str, Any] = {}
    for ep in ENDPOINTS:
        cb = _paired_bands_defined(base_C[ep][:, None], ["C"])
        entry = dict(cb["columns"]["C"])
        if "defined_groups" in cb:
            entry["defined_groups"] = cb["defined_groups"]
            entry["total_groups"] = cb["total_groups"]
            entry["undefined_group_indices"] = cb["undefined_group_indices"]
        c_levels[ep] = entry
    if include_pooled_group_ranking and "ranking_pooled_group" in base_C:
        cb_p = _paired_bands_defined(base_C["ranking_pooled_group"][:, None], ["C"])
        entry_p = dict(cb_p["columns"]["C"])
        if "defined_groups" in cb_p:
            entry_p["defined_groups"] = cb_p["defined_groups"]
            entry_p["total_groups"] = cb_p["total_groups"]
            entry_p["undefined_group_indices"] = cb_p["undefined_group_indices"]
        c_levels["ranking_pooled_group"] = entry_p

    # 62-arm likelihood (M, C_M) contrasts across all 3 endpoints
    m_62: dict[str, Any] = {}
    fwd_62: dict[str, dict[str, np.ndarray]] = {}
    abs_m_62: dict[str, dict[str, np.ndarray]] = {}
    abs_cm_62: dict[str, dict[str, np.ndarray]] = {}
    for ep in ENDPOINTS:
        sign = 1.0 if ep == "ranking" else -1.0
        fwd = {a: sign * (per_arm_designs[a][ep]["C_M"] - base_C[ep]) for a in arms_62}
        rev = {a: sign * (per_arm_designs[a][ep]["C_M"] - per_arm_designs[a][ep]["M"]) for a in arms_62}
        fwd_62[ep] = fwd
        abs_m_62[ep] = {a: per_arm_designs[a][ep]["M"] for a in arms_62}
        abs_cm_62[ep] = {a: per_arm_designs[a][ep]["C_M"] for a in arms_62}
        m_62[ep] = {
            "plus_M_given_C": _paired_bands_defined(_stack_cols(fwd, arms_62), arms_62),
            "C_given_M": _paired_bands_defined(_stack_cols(rev, arms_62), arms_62),
            "M_levels": _paired_bands_defined(_stack_cols(abs_m_62[ep], arms_62), arms_62),
            "C_M_levels": _paired_bands_defined(_stack_cols(abs_cm_62[ep], arms_62), arms_62),
        }

    # Direct calibration vs. transfer contrasts for 62 arms (C, M, C_M)
    c_gap = base_C["transfer_error"] - base_C["calibrated_error"]
    cal_vs_trans_62 = {
        "C_transfer_minus_calibrated_mse": paired_bands_summary(c_gap[:, None], ["C"])["columns"]["C"],
        "plus_M_calibration_advantage": paired_bands_summary(
            _stack_cols(
                {a: fwd_62["calibrated_error"][a] - fwd_62["transfer_error"][a] for a in arms_62},
                arms_62,
            ),
            arms_62,
        ),
        "M_transfer_minus_calibrated_mse": paired_bands_summary(
            _stack_cols(
                {a: abs_m_62["transfer_error"][a] - abs_m_62["calibrated_error"][a] for a in arms_62},
                arms_62,
            ),
            arms_62,
        ),
        "C_M_transfer_minus_calibrated_mse": paired_bands_summary(
            _stack_cols(
                {a: abs_cm_62["transfer_error"][a] - abs_cm_62["calibrated_error"][a] for a in arms_62},
                arms_62,
            ),
            arms_62,
        ),
    }

    # Representation (R, C_R, C_M_R) contrasts across all N_R available arms
    r_block: dict[str, Any] = {"n_arms_with_R": len(arms_with_R), "arms_with_R": list(arms_with_R)}
    if arms_with_R:
        r_by_ep: dict[str, Any] = {}
        fwd_r: dict[str, dict[str, np.ndarray]] = {}
        fwd_mr: dict[str, dict[str, np.ndarray]] = {}
        abs_r: dict[str, dict[str, np.ndarray]] = {}
        abs_cr: dict[str, dict[str, np.ndarray]] = {}
        abs_cmr: dict[str, dict[str, np.ndarray]] = {}
        for ep in ENDPOINTS:
            sign = 1.0 if ep == "ranking" else -1.0
            f_r = {a: sign * (per_arm_designs[a][ep]["C_R"] - base_C[ep]) for a in arms_with_R}
            f_mr = {a: sign * (per_arm_designs[a][ep]["C_M_R"] - base_C[ep]) for a in arms_with_R}
            f_r_cm = {
                a: sign * (per_arm_designs[a][ep]["C_M_R"] - per_arm_designs[a][ep]["C_M"])
                for a in arms_with_R
            }
            rev_c_r = {
                a: sign * (per_arm_designs[a][ep]["C_R"] - per_arm_designs[a][ep]["R"])
                for a in arms_with_R
            }
            r_minus_m = {
                a: sign * (per_arm_designs[a][ep]["R"] - per_arm_designs[a][ep]["M"])
                for a in arms_with_R
            }
            fwd_r[ep] = f_r
            fwd_mr[ep] = f_mr
            abs_r[ep] = {a: per_arm_designs[a][ep]["R"] for a in arms_with_R}
            abs_cr[ep] = {a: per_arm_designs[a][ep]["C_R"] for a in arms_with_R}
            abs_cmr[ep] = {a: per_arm_designs[a][ep]["C_M_R"] for a in arms_with_R}
            r_by_ep[ep] = {
                "plus_R_given_C": _paired_bands_defined(_stack_cols(f_r, arms_with_R), arms_with_R),
                "plus_MR_given_C": _paired_bands_defined(_stack_cols(f_mr, arms_with_R), arms_with_R),
                "plus_R_given_CM": _paired_bands_defined(_stack_cols(f_r_cm, arms_with_R), arms_with_R),
                "C_given_R": _paired_bands_defined(_stack_cols(rev_c_r, arms_with_R), arms_with_R),
                "R_minus_M": _paired_bands_defined(_stack_cols(r_minus_m, arms_with_R), arms_with_R),
                "R_levels": _paired_bands_defined(_stack_cols(abs_r[ep], arms_with_R), arms_with_R),
                "C_R_levels": _paired_bands_defined(_stack_cols(abs_cr[ep], arms_with_R), arms_with_R),
                "C_M_R_levels": _paired_bands_defined(_stack_cols(abs_cmr[ep], arms_with_R), arms_with_R),
            }
        r_block["endpoints"] = r_by_ep
        r_block["calibration_vs_transfer"] = {
            "plus_R_calibration_advantage": paired_bands_summary(
                _stack_cols(
                    {a: fwd_r["calibrated_error"][a] - fwd_r["transfer_error"][a] for a in arms_with_R},
                    arms_with_R,
                ),
                arms_with_R,
            ),
            "plus_MR_calibration_advantage": paired_bands_summary(
                _stack_cols(
                    {a: fwd_mr["calibrated_error"][a] - fwd_mr["transfer_error"][a] for a in arms_with_R},
                    arms_with_R,
                ),
                arms_with_R,
            ),
            "R_transfer_minus_calibrated_mse": paired_bands_summary(
                _stack_cols(
                    {a: abs_r["transfer_error"][a] - abs_r["calibrated_error"][a] for a in arms_with_R},
                    arms_with_R,
                ),
                arms_with_R,
            ),
            "C_R_transfer_minus_calibrated_mse": paired_bands_summary(
                _stack_cols(
                    {a: abs_cr["transfer_error"][a] - abs_cr["calibrated_error"][a] for a in arms_with_R},
                    arms_with_R,
                ),
                arms_with_R,
            ),
            "C_M_R_transfer_minus_calibrated_mse": paired_bands_summary(
                _stack_cols(
                    {a: abs_cmr["transfer_error"][a] - abs_cmr["calibrated_error"][a] for a in arms_with_R},
                    arms_with_R,
                ),
                arms_with_R,
            ),
        }

    # Compact per-arm level table (all 62 arms, None where R absent)
    per_arm_table: dict[str, Any] = {}
    r_set = set(arms_with_R)
    for arm in arms_62:
        row: dict[str, Any] = {
            "has_R": arm in r_set,
            "M": {ep: float(per_arm_designs[arm][ep]["M"].mean()) for ep in ENDPOINTS},
            "C_M": {ep: float(per_arm_designs[arm][ep]["C_M"].mean()) for ep in ENDPOINTS},
        }
        if include_pooled_group_ranking and "ranking_pooled_group" in per_arm_designs[arm]:
            row["M"]["ranking_pooled_group"] = float(per_arm_designs[arm]["ranking_pooled_group"]["M"].mean())
            row["C_M"]["ranking_pooled_group"] = float(per_arm_designs[arm]["ranking_pooled_group"]["C_M"].mean())
        if arm in r_set:
            for d_key in ("R", "C_R", "C_M_R"):
                row[d_key] = {ep: float(per_arm_designs[arm][ep][d_key].mean()) for ep in ENDPOINTS}
                if include_pooled_group_ranking and "ranking_pooled_group" in per_arm_designs[arm]:
                    row[d_key]["ranking_pooled_group"] = float(
                        per_arm_designs[arm]["ranking_pooled_group"][d_key].mean()
                    )
        else:
            for d_key in ("R", "C_R", "C_M_R"):
                row[d_key] = None
        per_arm_table[arm] = row

    return {
        "baseline_C_levels": c_levels,
        "dimension_1_quantitative_error": {
            "calibrated_error_62_M": m_62["calibrated_error"],
            "transfer_error_62_M": m_62["transfer_error"],
            "calibration_vs_transfer_62_M": cal_vs_trans_62,
            "representation_R": {
                "n_arms_with_R": len(arms_with_R),
                "calibrated_error": r_block["endpoints"]["calibrated_error"] if arms_with_R else None,
                "transfer_error": r_block["endpoints"]["transfer_error"] if arms_with_R else None,
                "calibration_vs_transfer": r_block["calibration_vs_transfer"] if arms_with_R else None,
            },
        },
        "dimension_2_ranking": {
            "ranking_62_M": m_62["ranking"],
            "ranking_R": r_block["endpoints"]["ranking"] if arms_with_R else None,
        },
        "per_arm_summary": per_arm_table,
    }


def _summarize_ladder_series(
    arm_ids: tuple[str, ...],
    values_by_arm: dict[str, float | None],
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    xs: list[float] = []
    ys: list[float] = []
    for arm_id in arm_ids:
        val = values_by_arm.get(arm_id)
        pm = ARM_METADATA[arm_id].params_m
        points.append({"arm_id": arm_id, "params_m": pm, "value": None if val is None else float(val)})
        if val is not None and np.isfinite(val) and pm is not None and pm > 0:
            xs.append(math.log10(float(pm)))
            ys.append(float(val))
    n_obs = len(xs)
    if n_obs < 2:
        return {
            "n_observed": n_obs,
            "points": points,
            "slope_per_log10_params": None,
            "spearman_with_scale": None,
            "peak_arm": points[0]["arm_id"] if n_obs == 1 else None,
            "largest_arm": arm_ids[-1],
            "ultra_scale_drop_from_peak": None,
            "inverted_u_detected": False,
        }
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    slope = float(np.polyfit(x_arr, y_arr, 1)[0])
    rho_scale = safe_spearman(x_arr, y_arr)

    obs_pts = [p for p in points if p["value"] is not None]
    best_pt = max(obs_pts, key=lambda p: p["value"]) if higher_is_better else min(obs_pts, key=lambda p: p["value"])
    largest_pt = obs_pts[-1]
    # Positive ultra_scale_drop_from_peak means the largest model is WORSE than an intermediate peak.
    drop = (
        float(best_pt["value"] - largest_pt["value"])
        if higher_is_better
        else float(largest_pt["value"] - best_pt["value"])
    )
    inverted_u = bool(n_obs >= 3 and best_pt["arm_id"] != largest_pt["arm_id"] and drop > 0.0)
    return {
        "n_observed": n_obs,
        "points": points,
        "slope_per_log10_params": slope,
        "spearman_with_scale": None if rho_scale is None else float(rho_scale),
        "peak_arm": best_pt["arm_id"],
        "peak_value": float(best_pt["value"]),
        "largest_arm": largest_pt["arm_id"],
        "largest_value": float(largest_pt["value"]),
        "ultra_scale_drop_from_peak": drop,
        "inverted_u_detected": inverted_u,
    }


def build_scaling_divergence_curves(
    ms_79_summary: dict[str, Any],
    ms_101_zs: dict[str, Any],
    dm_94_summary: dict[str, Any],
    dm_96_zs: dict[str, Any],
) -> dict[str, Any]:
    """Compute per-ladder scaling trajectories, log-linear slopes, and M vs. R divergence."""
    datasets_data = {
        "megascale_cc_79": {
            "zero_shot_M_rho": {
                a: ms_101_zs["arms_62"]["columns"][a]["point"] for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
            "cv_M_rho": {a: ms_79_summary[a]["M"]["ranking"] for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS},
            "cv_R_rho": {
                a: (ms_79_summary[a]["R"]["ranking"] if ms_79_summary[a]["R"] else None)
                for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
            "cv_C_M_R_rho": {
                a: (ms_79_summary[a]["C_M_R"]["ranking"] if ms_79_summary[a]["C_M_R"] else None)
                for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
            "cv_M_transfer_mse": {
                a: ms_79_summary[a]["M"]["transfer_error"] for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
            "cv_R_transfer_mse": {
                a: (ms_79_summary[a]["R"]["transfer_error"] if ms_79_summary[a]["R"] else None)
                for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
            "cv_C_M_R_transfer_mse": {
                a: (ms_79_summary[a]["C_M_R"]["transfer_error"] if ms_79_summary[a]["C_M_R"] else None)
                for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
            "cv_M_calibrated_mse": {
                a: ms_79_summary[a]["M"]["calibrated_error"] for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
            "cv_R_calibrated_mse": {
                a: (ms_79_summary[a]["R"]["calibrated_error"] if ms_79_summary[a]["R"] else None)
                for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
            "cv_C_M_R_calibrated_mse": {
                a: (ms_79_summary[a]["C_M_R"]["calibrated_error"] if ms_79_summary[a]["C_M_R"] else None)
                for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
        },
        "domainome_cc_94": {
            "zero_shot_M_rho": {
                a: dm_96_zs["arms_62"]["columns"][a]["point"] for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
            "cv_M_rho": {a: dm_94_summary[a]["M"]["ranking"] for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS},
            "cv_R_rho": {
                a: (dm_94_summary[a]["R"]["ranking"] if dm_94_summary[a]["R"] else None)
                for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
            "cv_C_M_R_rho": {
                a: (dm_94_summary[a]["C_M_R"]["ranking"] if dm_94_summary[a]["C_M_R"] else None)
                for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
            "cv_M_transfer_mse": {
                a: dm_94_summary[a]["M"]["transfer_error"] for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
            "cv_R_transfer_mse": {
                a: (dm_94_summary[a]["R"]["transfer_error"] if dm_94_summary[a]["R"] else None)
                for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
            "cv_C_M_R_transfer_mse": {
                a: (dm_94_summary[a]["C_M_R"]["transfer_error"] if dm_94_summary[a]["C_M_R"] else None)
                for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
            "cv_M_calibrated_mse": {
                a: dm_94_summary[a]["M"]["calibrated_error"] for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
            "cv_R_calibrated_mse": {
                a: (dm_94_summary[a]["R"]["calibrated_error"] if dm_94_summary[a]["R"] else None)
                for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
            "cv_C_M_R_calibrated_mse": {
                a: (dm_94_summary[a]["C_M_R"]["calibrated_error"] if dm_94_summary[a]["C_M_R"] else None)
                for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
        },
    }

    out: dict[str, Any] = {}
    for ds_name, series_map in datasets_data.items():
        ladders_out: dict[str, Any] = {}
        for ladder_name, arm_ids in SCALING_LADDERS.items():
            ladders_out[ladder_name] = {
                metric_name: _summarize_ladder_series(
                    arm_ids,
                    vals,
                    higher_is_better=metric_name.endswith("_rho"),
                )
                for metric_name, vals in series_map.items()
            }
        singletons_out = {
            arm_id: {
                "params_m": ARM_METADATA[arm_id].params_m,
                "paradigm": ARM_METADATA[arm_id].paradigm,
                **{metric_name: vals.get(arm_id) for metric_name, vals in series_map.items()},
            }
            for arm_id in HIGHLIGHTED_SINGLETONS
        }
        out[ds_name] = {"ladders": ladders_out, "highlighted_singletons": singletons_out}
    return out


# --------------------------------------------------------------------------- #
# Main orchestrator
# --------------------------------------------------------------------------- #


def build_unified_r2_experiment_2_1(
    *,
    root: Path = REPO_ROOT,
    out_dir: Path | None = None,
    r_archive_dir: Path | None = None,
    max_workers: int = 4,
    blas_threads: int = 2,
    allow_partial_r: bool = True,
    force_refit: bool = False,
) -> dict[str, Any]:
    """Execute the full unified 62-model Merged New Experiment 2.1 across MegaScale & Domainome."""
    t_start = time.monotonic()
    target_dir = Path(out_dir) if out_dir is not None else (root / UNIFIED_OUTPUT_DIR)
    r_dir = Path(r_archive_dir) if r_archive_dir is not None else (root / DEFAULT_R_ARCHIVE_DIR)
    ms_cache_dir = target_dir / "cells" / "r2_exp21" / "megascale"
    dm_cache_dir = target_dir / "cells" / "r2_exp21" / "domainome"
    ms_cache_dir.mkdir(parents=True, exist_ok=True)
    dm_cache_dir.mkdir(parents=True, exist_ok=True)

    arms_62 = UNIFIED_62_SINGLE_SEQUENCE_ARMS
    n_workers = max(1, min(int(max_workers), 16))

    # ==================== 1. MEGASCALE ====================
    print("[r2-exp2.1] loading MegaScale 101g/79g supports and precomputing fold caches...", flush=True)
    ms = load_megascale_supports(root=root)
    ms_w_101 = row_weights(ms["panel_101"]["site"], ms["panel_101"]["group"])
    ms_w_79 = row_weights(ms["panel_79"]["site"], ms["panel_79"]["group"])

    fc_ms_79_evo = build_fold_cache(groups=ms["panel_79"]["group"], weights=ms_w_79)
    fc_ms_101_evo = build_fold_cache(groups=ms["panel_101"]["group"], weights=ms_w_101)
    fc_ms_79_full = build_fold_cache(
        groups=ms["panel_79"]["group"],
        weights=ms_w_79,
        g_builder=lambda held, inner: gate.nuisance_response(
            ms["panel_79"]["states"], ms["panel_79"]["pair_states"], np.asarray(held), inner, device="cpu"
        )[0],
    )
    fc_ms_101_full = build_fold_cache(
        groups=ms["panel_101"]["group"],
        weights=ms_w_101,
        g_builder=lambda held, inner: gate.nuisance_response(
            ms["panel_101"]["states"], ms["panel_101"]["pair_states"], np.asarray(held), inner, device="cpu"
        )[0],
    )

    # Verification gates on MegaScale
    print("[r2-exp2.1] running MegaScale verification gates (101g repro + ESMC-6B 79g parity)...", flush=True)
    gate_ms_101 = run_megascale_reproduction_gate(ms, fc_ms_101_full, root=root)
    gate_esmc6b = run_esmc6b_evo_parity_gate(ms, fc_ms_79_evo, root=root)
    print(
        f"[r2-exp2.1] MegaScale gates PASSED: 101g oof_diff={gate_ms_101['max_oof_abs_diff']:.2e}, "
        f"ESMC-6B mse_diff={gate_esmc6b['max_mse_abs_diff']:.2e}",
        flush=True,
    )

    # Model-independent MegaScale baselines (fit ONCE)
    eval_ms_79 = lambda p: evaluate_megascale_oof(  # noqa: E731
        p, ms["panel_79"]["target"], ms["panel_79"]["group"], ms["panel_79"]["site"], ms["groups_79"]
    )
    eval_ms_101 = lambda p: evaluate_megascale_oof(  # noqa: E731
        p, ms["panel_101"]["target"], ms["panel_101"]["group"], ms["panel_101"]["site"], ms["groups_101"]
    )

    ms_base_79_evo = collapse_oof(
        fit_designs_with_cache(
            ms["panel_79"]["target"],
            {
                "P": ms["p_logodds_79"][:, None],
                "POTTS": ms["potts_raw_79"][:, None],
                "GEMME": ms["gemme_raw_79"][:, None],
            },
            {"C": ("P", "POTTS", "GEMME")},
            fc_ms_79_evo,
            alphas=ALPHAS_17,
            renorm_val_weights=True,
            tie_largest_alpha=True,
        ),
        eval_ms_79,
    )
    ms_base_101_evo = collapse_oof(
        fit_designs_with_cache(
            ms["panel_101"]["target"],
            {"P": ms["p_logodds_101"][:, None]},
            {"C": ("P",)},
            fc_ms_101_evo,
            alphas=ALPHAS_17,
            renorm_val_weights=True,
            tie_largest_alpha=True,
        ),
        eval_ms_101,
    )
    ms_base_79_full = collapse_oof(
        fit_designs_with_cache(
            ms["panel_79"]["target"],
            ms["panel_79"]["blocks"],
            dict(MS_PROGRESSION_RUNGS),
            fc_ms_79_full,
            alphas=ALPHAS_5,
        ),
        eval_ms_79,
    )
    ms_base_101_full = collapse_oof(
        fit_designs_with_cache(
            ms["panel_101"]["target"],
            ms["panel_101"]["blocks"],
            {"S": MS_DESIGN_S, "S2": MS_DESIGN_S2},
            fc_ms_101_full,
            alphas=ALPHAS_5,
        ),
        eval_ms_101,
    )

    pre33_ms_101 = load_preregistered_ms_101_readings(ms, root=root)
    pre33_ms_79 = load_preregistered_ms_79_cells(ms, root=root)
    max_ms79_base_diff = max(
        float(np.max(np.abs(ms_base_79_full[ep][rung] - pre33_ms_79["progen2-medium"][rung][ep]["U_C"])))
        for rung in MS_PROGRESSION_RUNGS
        for ep in ENDPOINTS
    )
    if max_ms79_base_diff >= REPRODUCTION_TOLERANCE:
        raise RuntimeError(f"MegaScale 79g baseline U_C mismatch: {max_ms79_base_diff:g}")

    # Fit/load all 62 MegaScale arms
    ms_shared = {
        "root": str(root),
        "cache_dir": str(ms_cache_dir),
        "r_archive_dir": str(r_dir),
        "force": force_refit,
        "blas_threads": blas_threads,
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
    print(f"[r2-exp2.1] fitting/loading 62 MegaScale arms (workers={n_workers}, spawn)...", flush=True)
    if n_workers == 1:
        _init_ms_worker(ms_shared)
        for idx, arm_id in enumerate(arms_62, 1):
            rec = _fit_one_megascale_arm(arm_id)
            ms_records[arm_id] = rec
            if idx % 10 == 0 or idx == len(arms_62):
                print(f"  [ms {idx}/{len(arms_62)}] {arm_id} (has_R={rec['has_R']}, {rec['elapsed_seconds']:.1f}s)", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx, initializer=_init_ms_worker, initargs=(ms_shared,)
        ) as pool:
            for idx, rec in enumerate(pool.map(_fit_one_megascale_arm, arms_62), 1):
                ms_records[rec["arm_id"]] = rec
                if idx % 6 == 0 or idx == len(arms_62):
                    print(
                        f"  [ms {idx}/{len(arms_62)}] {rec['arm_id']} "
                        f"(has_R={rec['has_R']}, {rec['elapsed_seconds']:.1f}s)",
                        flush=True,
                    )
    ms_arms_with_R = tuple(a for a in arms_62 if ms_records[a]["has_R"])
    print(
        f"[r2-exp2.1] MegaScale complete: 62/62 M, {len(ms_arms_with_R)}/62 R "
        f"({time.monotonic() - t_start:.1f}s elapsed)",
        flush=True,
    )
    if not allow_partial_r and len(ms_arms_with_R) != 62:
        missing_r = [a for a in arms_62 if not ms_records[a]["has_R"]]
        raise RuntimeError(f"MegaScale missing R for {len(missing_r)} arms: {missing_r[:5]}...")

    # ==================== 2. DOMAINOME ====================
    t_dm = time.monotonic()
    print("[r2-exp2.1] loading Domainome 96g/94g supports and precomputing fold caches...", flush=True)
    dm = load_domainome_supports(root=root)
    dm_w_94_site = row_weights(dm["panel_94"]["site"], dm["panel_94"]["group"])
    dm_w_96_site = row_weights(dm["panel_96"]["site"], dm["panel_96"]["group"])

    fc_dm_94_evo = build_fold_cache(groups=dm["panel_94"]["group"], weights=dm_w_94_site)
    fc_dm_96_evo = build_fold_cache(groups=dm["panel_96"]["group"], weights=dm_w_96_site)
    fc_dm_94_full = build_fold_cache(
        groups=dm["panel_94"]["group"],
        weights=dm["panel_94"]["weights"],
        g_builder=lambda held, inner: abundance_mod.nuisance_response(
            dm["panel_94"],
            DM_FIRST_STAGE,
            dm["panel_94"]["blocks"],
            np.asarray(held),
            inner,
            device="cpu",
        )[0],
    )

    eval_dm_94_evo = lambda p: evaluate_domainome_oof(  # noqa: E731
        p,
        dm["panel_94"]["target"],
        dm["panel_94"]["group"],
        dm["panel_94"]["domain"],
        dm["panel_94"]["site"],
        dm["groups_94"],
        use_nested_weights=False,
    )
    eval_dm_96_evo = lambda p: evaluate_domainome_oof(  # noqa: E731
        p,
        dm["panel_96"]["target"],
        dm["panel_96"]["group"],
        dm["panel_96"]["domain"],
        dm["panel_96"]["site"],
        dm["groups_96"],
        use_nested_weights=False,
    )
    eval_dm_94_full = lambda p: evaluate_domainome_oof(  # noqa: E731
        p,
        dm["panel_94"]["target"],
        dm["panel_94"]["group"],
        dm["panel_94"]["domain"],
        dm["panel_94"]["site"],
        dm["groups_94"],
        use_nested_weights=True,
    )

    dm_base_94_evo = collapse_oof(
        fit_designs_with_cache(
            dm["panel_94"]["target"],
            {
                "P": dm["p_logodds_94"][:, None],
                "POTTS": dm["potts_raw_94"][:, None],
                "GEMME": dm["gemme_raw_94"][:, None],
            },
            {"C": ("P", "POTTS", "GEMME")},
            fc_dm_94_evo,
            alphas=ALPHAS_17,
            renorm_val_weights=True,
            tie_largest_alpha=True,
        ),
        eval_dm_94_evo,
    )
    dm_base_96_evo = collapse_oof(
        fit_designs_with_cache(
            dm["panel_96"]["target"],
            {"P": dm["p_logodds_96"][:, None]},
            {"C": ("P",)},
            fc_dm_96_evo,
            alphas=ALPHAS_17,
            renorm_val_weights=True,
            tie_largest_alpha=True,
        ),
        eval_dm_96_evo,
    )
    dm_base_94_full = collapse_oof(
        fit_designs_with_cache(
            dm["panel_94"]["target"],
            dm["panel_94"]["blocks"],
            dict(DM_PROGRESSION_RUNGS),
            fc_dm_94_full,
            alphas=ALPHAS_5,
        ),
        eval_dm_94_full,
    )

    dm_shared = {
        "root": str(root),
        "cache_dir": str(dm_cache_dir),
        "r_archive_dir": str(r_dir),
        "force": force_refit,
        "blas_threads": blas_threads,
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
    print(f"[r2-exp2.1] fitting/loading 62 Domainome arms (workers={n_workers}, spawn)...", flush=True)
    if n_workers == 1:
        _init_dm_worker(dm_shared)
        for idx, arm_id in enumerate(arms_62, 1):
            rec = _fit_one_domainome_arm(arm_id)
            dm_records[arm_id] = rec
            if idx % 10 == 0 or idx == len(arms_62):
                print(f"  [dm {idx}/{len(arms_62)}] {arm_id} (has_R={rec['has_R']}, {rec['elapsed_seconds']:.1f}s)", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx, initializer=_init_dm_worker, initargs=(dm_shared,)
        ) as pool:
            for idx, rec in enumerate(pool.map(_fit_one_domainome_arm, arms_62), 1):
                dm_records[rec["arm_id"]] = rec
                if idx % 6 == 0 or idx == len(arms_62):
                    print(
                        f"  [dm {idx}/{len(arms_62)}] {rec['arm_id']} "
                        f"(has_R={rec['has_R']}, {rec['elapsed_seconds']:.1f}s)",
                        flush=True,
                    )
    dm_arms_with_R = tuple(a for a in arms_62 if dm_records[a]["has_R"])
    print(
        f"[r2-exp2.1] Domainome complete: 62/62 M, {len(dm_arms_with_R)}/62 R "
        f"({time.monotonic() - t_dm:.1f}s for Domainome)",
        flush=True,
    )
    if not allow_partial_r and len(dm_arms_with_R) != 62:
        missing_r = [a for a in arms_62 if not dm_records[a]["has_R"]]
        raise RuntimeError(f"Domainome missing R for {len(missing_r)} arms: {missing_r[:5]}...")

    # ==================== 3. ASSEMBLE MEGASCALE SECTIONS ====================
    # Zero-shot explicit baselines on MegaScale
    _, ms_zs_p_101 = group_spearman(ms["panel_101"]["target"], ms["p_logodds_101"], ms["panel_101"]["group"])
    _, ms_zs_p_79 = group_spearman(ms["panel_79"]["target"], ms["p_logodds_79"], ms["panel_79"]["group"])
    _, ms_zs_e_79 = group_spearman(ms["panel_79"]["target"], ms["potts_raw_79"], ms["panel_79"]["group"])
    _, ms_zs_g_79 = group_spearman(ms["panel_79"]["target"], ms["gemme_raw_79"], ms["panel_79"]["group"])

    ms_zs_101 = _build_zero_shot_block(
        {a: np.asarray(ms_records[a]["zero_shot_101"], dtype=np.float64) for a in arms_62},
        {"profile": np.asarray([np.nan if v is None else float(v) for v in ms_zs_p_101], dtype=np.float64)},
        arms_62,
    )
    ms_zs_79 = _build_zero_shot_block(
        {a: np.asarray(ms_records[a]["zero_shot_79"], dtype=np.float64) for a in arms_62},
        {
            "profile": np.asarray([np.nan if v is None else float(v) for v in ms_zs_p_79], dtype=np.float64),
            "potts": np.asarray(ms_zs_e_79, dtype=np.float64),
            "gemme": np.asarray(ms_zs_g_79, dtype=np.float64),
        },
        arms_62,
    )

    # MegaScale primary_evolutionary_control (cc_79 & full_101)
    ms_evo_79_per_arm = {
        a: {ep: {d: np.asarray(v, dtype=np.float64) for d, v in by_d.items()} for ep, by_d in ms_records[a]["evo_79"].items()}
        for a in arms_62
    }
    ms_evo_101_per_arm = {
        a: {ep: {d: np.asarray(v, dtype=np.float64) for d, v in by_d.items()} for ep, by_d in ms_records[a]["evo_101"].items()}
        for a in arms_62
    }
    ms_evo_79_block = _build_six_design_block(
        {ep: ms_base_79_evo[ep]["C"] for ep in ENDPOINTS},
        ms_evo_79_per_arm,
        arms_62,
        ms_arms_with_R,
    )
    ms_evo_101_block = _build_six_design_block(
        {ep: ms_base_101_evo[ep]["C"] for ep in ENDPOINTS},
        ms_evo_101_per_arm,
        arms_62,
        ms_arms_with_R,
    )

    # MegaScale full_nuisance_evolutionary_control (cc_79 3-rung + 6-design top rung, & full_101 S->S2)
    ms_full_79_rungs: dict[str, Any] = {}
    for rung_name in MS_PROGRESSION_RUNGS:
        rung_ep: dict[str, Any] = {}
        for ep in ENDPOINTS:
            sign = 1.0 if ep == "ranking" else -1.0
            fwd_d: dict[str, np.ndarray] = {}
            rev_d: dict[str, np.ndarray] = {}
            for a in arms_62:
                if a in PREREGISTERED_33_SET:
                    cell = pre33_ms_79[a][rung_name][ep]
                    u_c, m_v, j_v = cell["U_C"], cell["M"], cell["U_C_M"]
                else:
                    u_c = ms_base_79_full[ep][rung_name]
                    m_v = np.asarray(ms_records[a]["full_79"][ep]["M"], dtype=np.float64)
                    j_v = np.asarray(ms_records[a]["full_79"][ep][f"{rung_name}|M"], dtype=np.float64)
                fwd_d[a] = sign * (j_v - u_c)
                rev_d[a] = sign * (j_v - m_v)
            rung_ep[ep] = {
                "forward_62": paired_bands_summary(_stack_cols(fwd_d, arms_62), arms_62),
                "reverse_62": paired_bands_summary(_stack_cols(rev_d, arms_62), arms_62),
            }
        ms_full_79_rungs[rung_name] = rung_ep

    ms_full_79_top_per_arm: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    top_rung = "nuisance+profile+couplings+gemme"
    for a in arms_62:
        arm_ep: dict[str, dict[str, np.ndarray]] = {}
        for ep in ENDPOINTS:
            if a in PREREGISTERED_33_SET:
                cell = pre33_ms_79[a][top_rung][ep]
                d_map = {"M": cell["M"], "C_M": cell["U_C_M"]}
            else:
                d_map = {
                    "M": np.asarray(ms_records[a]["full_79"][ep]["M"], dtype=np.float64),
                    "C_M": np.asarray(ms_records[a]["full_79"][ep][f"{top_rung}|M"], dtype=np.float64),
                }
            if ms_records[a]["has_R"]:
                d_map["R"] = np.asarray(ms_records[a]["full_79"][ep]["R"], dtype=np.float64)
                d_map["C_R"] = np.asarray(ms_records[a]["full_79"][ep][f"{top_rung}|R"], dtype=np.float64)
                d_map["C_M_R"] = np.asarray(ms_records[a]["full_79"][ep][f"{top_rung}|MR"], dtype=np.float64)
            arm_ep[ep] = d_map
        ms_full_79_top_per_arm[a] = arm_ep
    ms_full_79_top_block = _build_six_design_block(
        {ep: ms_base_79_full[ep][top_rung] for ep in ENDPOINTS},
        ms_full_79_top_per_arm,
        arms_62,
        ms_arms_with_R,
    )

    # MegaScale 101g S -> S2 panel
    ms_101_s_to_s2: dict[str, Any] = {}
    for ep in ENDPOINTS:
        sign = 1.0 if ep == "ranking" else -1.0
        for b_label, d_b, d_j in (("matched", "S", "S_M"), ("profile", "S2", "S2_M")):
            inc = {}
            for a in arms_62:
                if a in PREREGISTERED_33_SET:
                    inc[a] = sign * (pre33_ms_101[a][ep][d_j] - pre33_ms_101[a][ep][d_b])
                else:
                    inc[a] = sign * (
                        np.asarray(ms_records[a]["full_101"][ep][d_j], dtype=np.float64)
                        - ms_base_101_full[ep][d_b]
                    )
            ms_101_s_to_s2[f"{b_label}_{ep}"] = paired_bands_summary(_stack_cols(inc, arms_62), arms_62)

    ms_101_s2_top_per_arm: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for a in arms_62:
        arm_ep = {}
        for ep in ENDPOINTS:
            if a in PREREGISTERED_33_SET:
                # For 101g S2, M-only isn't in fit_<arm>.json, so use evo_101 M (identical scalar Ridge on 101g!)
                d_map = {
                    "M": np.asarray(ms_records[a]["evo_101"][ep]["M"], dtype=np.float64),
                    "C_M": pre33_ms_101[a][ep]["S2_M"],
                }
            else:
                d_map = {
                    "M": np.asarray(ms_records[a]["full_101"][ep]["M"], dtype=np.float64),
                    "C_M": np.asarray(ms_records[a]["full_101"][ep]["S2_M"], dtype=np.float64),
                }
            if ms_records[a]["has_R"]:
                d_map["R"] = np.asarray(ms_records[a]["full_101"][ep]["R"], dtype=np.float64)
                d_map["C_R"] = np.asarray(ms_records[a]["full_101"][ep]["S2_R"], dtype=np.float64)
                d_map["C_M_R"] = np.asarray(ms_records[a]["full_101"][ep]["S2_MR"], dtype=np.float64)
            arm_ep[ep] = d_map
        ms_101_s2_top_per_arm[a] = arm_ep
    ms_101_s2_six_block = _build_six_design_block(
        {ep: ms_base_101_full[ep]["S2"] for ep in ENDPOINTS},
        ms_101_s2_top_per_arm,
        arms_62,
        ms_arms_with_R,
    )

    # ==================== 4. ASSEMBLE DOMAINOME SECTIONS ====================
    dm_zs_p_96 = abundance_mod._domain_spearman(dm["panel_96"], dm["p_logodds_96"])
    dm_zs_p_94 = abundance_mod._domain_spearman(dm["panel_94"], dm["p_logodds_94"])
    dm_zs_e_94 = abundance_mod._domain_spearman(dm["panel_94"], dm["potts_raw_94"])
    dm_zs_g_94 = abundance_mod._domain_spearman(dm["panel_94"], dm["gemme_raw_94"])

    dm_zs_96 = _build_zero_shot_block(
        {a: np.asarray(dm_records[a]["zero_shot_96"], dtype=np.float64) for a in arms_62},
        {"profile": np.asarray([np.nan if v is None else float(v) for v in dm_zs_p_96], dtype=np.float64)},
        arms_62,
    )
    dm_zs_94 = _build_zero_shot_block(
        {a: np.asarray(dm_records[a]["zero_shot_94"], dtype=np.float64) for a in arms_62},
        {
            "profile": np.asarray([np.nan if v is None else float(v) for v in dm_zs_p_94], dtype=np.float64),
            "potts": np.asarray([np.nan if v is None else float(v) for v in dm_zs_e_94], dtype=np.float64),
            "gemme": np.asarray([np.nan if v is None else float(v) for v in dm_zs_g_94], dtype=np.float64),
        },
        arms_62,
    )

    dm_evo_94_per_arm = {
        a: {ep: {d: np.asarray(v, dtype=np.float64) for d, v in by_d.items()} for ep, by_d in dm_records[a]["evo_94"].items()}
        for a in arms_62
    }
    dm_evo_96_per_arm = {
        a: {ep: {d: np.asarray(v, dtype=np.float64) for d, v in by_d.items()} for ep, by_d in dm_records[a]["evo_96"].items()}
        for a in arms_62
    }
    dm_evo_94_block = _build_six_design_block(
        {ep: dm_base_94_evo[ep]["C"] for ep in dm_base_94_evo},
        dm_evo_94_per_arm,
        arms_62,
        dm_arms_with_R,
        include_pooled_group_ranking=True,
    )
    dm_evo_96_block = _build_six_design_block(
        {ep: dm_base_96_evo[ep]["C"] for ep in dm_base_96_evo},
        dm_evo_96_per_arm,
        arms_62,
        dm_arms_with_R,
        include_pooled_group_ranking=True,
    )

    dm_full_94_rungs: dict[str, Any] = {}
    for rung_name in DM_PROGRESSION_RUNGS:
        rung_ep = {}
        for ep in ENDPOINTS:
            sign = 1.0 if ep == "ranking" else -1.0
            u_c = dm_base_94_full[ep][rung_name]
            fwd_d = {
                a: sign * (np.asarray(dm_records[a]["full_94"][ep][f"{rung_name}|M"], dtype=np.float64) - u_c)
                for a in arms_62
            }
            rev_d = {
                a: sign
                * (
                    np.asarray(dm_records[a]["full_94"][ep][f"{rung_name}|M"], dtype=np.float64)
                    - np.asarray(dm_records[a]["full_94"][ep]["M"], dtype=np.float64)
                )
                for a in arms_62
            }
            rung_ep[ep] = {
                "forward_62": paired_bands_summary(_stack_cols(fwd_d, arms_62), arms_62),
                "reverse_62": paired_bands_summary(_stack_cols(rev_d, arms_62), arms_62),
            }
        dm_full_94_rungs[rung_name] = rung_ep

    dm_full_94_top_per_arm: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for a in arms_62:
        arm_ep = {}
        for ep in ENDPOINTS:
            d_map = {
                "M": np.asarray(dm_records[a]["full_94"][ep]["M"], dtype=np.float64),
                "C_M": np.asarray(dm_records[a]["full_94"][ep][f"{top_rung}|M"], dtype=np.float64),
            }
            if dm_records[a]["has_R"]:
                d_map["R"] = np.asarray(dm_records[a]["full_94"][ep]["R"], dtype=np.float64)
                d_map["C_R"] = np.asarray(dm_records[a]["full_94"][ep][f"{top_rung}|R"], dtype=np.float64)
                d_map["C_M_R"] = np.asarray(dm_records[a]["full_94"][ep][f"{top_rung}|MR"], dtype=np.float64)
            arm_ep[ep] = d_map
        dm_full_94_top_per_arm[a] = arm_ep
    dm_full_94_top_block = _build_six_design_block(
        {ep: dm_base_94_full[ep][top_rung] for ep in ENDPOINTS},
        dm_full_94_top_per_arm,
        arms_62,
        dm_arms_with_R,
    )

    # ==================== 5. SCALING DIVERGENCE CURVES ====================
    scaling_curves = build_scaling_divergence_curves(
        ms_evo_79_block["per_arm_summary"],
        ms_zs_101,
        dm_evo_94_block["per_arm_summary"],
        dm_zs_96,
    )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.monotonic() - t_start),
        "arms_62": list(arms_62),
        "coverage_summary": {
            "megascale": {
                "n_arms_M": 62,
                "n_arms_R": len(ms_arms_with_R),
                "arms_with_R": list(ms_arms_with_R),
                "arms_missing_R": [a for a in arms_62 if a not in set(ms_arms_with_R)],
            },
            "domainome": {
                "n_arms_M": 62,
                "n_arms_R": len(dm_arms_with_R),
                "arms_with_R": list(dm_arms_with_R),
                "arms_missing_R": [a for a in arms_62 if a not in set(dm_arms_with_R)],
            },
        },
        "arm_metadata": {
            arm: {
                "display_name": ARM_METADATA[arm].display_name,
                "paradigm": ARM_METADATA[arm].paradigm,
                "family": ARM_METADATA[arm].family,
                "params_m": ARM_METADATA[arm].params_m,
                "origin_cohort": ARM_METADATA[arm].origin_cohort,
                "has_R_megascale": ms_records[arm]["has_R"],
                "has_R_domainome": dm_records[arm]["has_R"],
            }
            for arm in arms_62
        },
        "verification_gates": {
            "megascale_101g_reproduction": gate_ms_101,
            "megascale_79g_esmc6b_evo_parity": gate_esmc6b,
            "megascale_79g_full_nuisance_U_C_max_diff": max_ms79_base_diff,
        },
        "instrument_anchor": {
            "megascale_dual_protease_reliability": ms["instrument_anchor"],
        },
        "datasets": {
            "megascale": {
                "supports": {
                    "full_101": {"n_rows": 25856, "n_groups": 101, "groups": ms["groups_101"]},
                    "cc_79": {"n_rows": 20224, "n_groups": 79, "groups": ms["groups_79"]},
                },
                "zero_shot_spearman": {
                    "full_101": ms_zs_101,
                    "cc_79": ms_zs_79,
                },
                "primary_evolutionary_control": {
                    "cc_79_profile_potts_gemme": ms_evo_79_block,
                    "full_101_profile": ms_evo_101_block,
                },
                "full_nuisance_evolutionary_control": {
                    "cc_79_progression_3_rungs_62_M": ms_full_79_rungs,
                    "cc_79_top_rung_6_designs": ms_full_79_top_block,
                    "full_101_S_to_S2_62_M": ms_101_s_to_s2,
                    "full_101_S2_6_designs": ms_101_s2_six_block,
                },
            },
            "domainome": {
                "supports": {
                    "full_96": {"n_rows": 109568, "n_groups": 96, "n_domains": 428, "groups": dm["groups_96"]},
                    "cc_94": {"n_rows": 106496, "n_groups": 94, "n_domains": 416, "groups": dm["groups_94"]},
                },
                "zero_shot_spearman": {
                    "full_96": dm_zs_96,
                    "cc_94": dm_zs_94,
                },
                "primary_evolutionary_control": {
                    "cc_94_profile_potts_gemme": dm_evo_94_block,
                    "full_96_profile": dm_evo_96_block,
                },
                "full_nuisance_evolutionary_control": {
                    "cc_94_progression_3_rungs_62_M": dm_full_94_rungs,
                    "cc_94_top_rung_6_designs": dm_full_94_top_block,
                },
            },
        },
        "scaling_divergence_curves": scaling_curves,
    }

    out_path = target_dir / "r2_biophysical_unified_62.json"
    atomic_write_json(out_path, payload)
    digest = sha256_file(out_path)
    sidecar = out_path.with_name(out_path.name + ".sha256")
    sidecar.write_text(f"{digest}  {out_path.name}\n", encoding="utf-8")
    print(
        f"[r2-exp2.1] WROTE {out_path} (sha256={digest[:16]}...) in {payload['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return payload
