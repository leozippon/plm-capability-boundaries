"""Fast closed-form SVD/Gram Ridge layerwise probes and Eq. (3b) temperature calibration.

Implements:
  0. Exact closed-form Ridge with inner group-disjoint CV across log-spaced
     ``lambda in {1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4}``, supporting both direct SVD
     (optimal when ``N <= p``) and streaming fold-sufficient-statistics eigendecomposition
     (exact to machine precision, O(p^2) inner CV, minimal RAM even on 650k+ contact pairs).
  1. Probe 1 — MegaScale ddG Numerical Ridge Probe:
     Features ``[h^(l)_i(mut) - h^(l)_i(wt), mean(h^(l)(mut)) - mean(h^(l)(wt))] in R^(2d)``,
     trained strictly on ``seen_pool_natural`` WTs and evaluated on ``unseen_fold_natural``
     and ``de_novo_design`` WTs, directly compared against raw AND temperature-calibrated
     zero-shot output-head logits.
  2. Probe 2 — Unseen-Fold 3D Contact Pairwise Probe (49 held-out C.A.T domains):
     Symmetric features ``[h^(l)_i * h^(l)_j, |h^(l)_i - h^(l)_j|] in R^(2d)`` for
     ``|i - j| >= 6`` and long-range ``|i - j| >= 24`` (``d_CB < 8 A``), 5-fold
     domain-disjoint CV reporting out-of-fold ``AUROC``, ``P@L``, ``P@L/2``, ``P@L/5``
     vs zero-shot Categorical Jacobian + APC.
  3. Probe 3 — Unseen-Fold Per-Residue 3D Burial / Coordination Number Ridge Probe:
     Predicts continuous C-beta 8 A contact count (``|i - j| >= 6``) from single-residue
     ``h^(l)_i in R^d`` via 5-fold domain-disjoint Ridge, reporting out-of-fold ``R^2``
     and Spearman ``rho`` per layer.
  4. Eq. (3b) Temperature Calibration on ``eval_unseen_fold``:
     Computes exact ``H_unseen_fold``, ``H_unseen_fold_cal = min_{T > 0} H(theta, T)``,
     ``T_opt``, and ``Delta_cal = H_unseen_fold - H_unseen_fold_cal`` matching native
     objective weighting for ``ar``, ``mlm_15``, and ``oa_uniform`` (``MDM``).
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from src.capability.supply_demand.evaluation import (
    CONTACT_DISTANCE_ANGSTROM,
    NUM_EVAL_STRATA,
    _deterministic_valid_mask,
    _eval_autocast_ctx,
    compute_contact_metrics_for_domain,
    evaluate_unseen_fold_contacts,
)
from src.capability.supply_demand.model import VariableLengthProteinTransformer
from src.capability.supply_demand.objectives import (
    PackedSequenceStore,
    encode_aa_sequence,
    prepare_ar_batch,
)
from src.capability.supply_demand.spec import (
    AA_TO_ID,
    LN2,
    MASK_TOKEN_ID,
    VOCAB_AA,
)

from .extractor import (
    canonical_objective,
    extract_layerwise_hidden_states,
    forward_layerwise_hidden_states,
    get_layer_names,
)

DEFAULT_RIDGE_LAMBDAS: tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)
_STD_EPS: float = 1e-8


# --------------------------------------------------------------------------- #
# Metric Helpers
# --------------------------------------------------------------------------- #


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(mask.sum()) < 3:
        return float("nan")
    if float(np.std(xa[mask])) <= 1e-15 or float(np.std(ya[mask])) <= 1e-15:
        return 0.0
    val = float(spearmanr(xa[mask], ya[mask]).statistic)
    return val if math.isfinite(val) else 0.0


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(mask.sum()) < 3:
        return float("nan")
    if float(np.std(xa[mask])) <= 1e-15 or float(np.std(ya[mask])) <= 1e-15:
        return 0.0
    val = float(pearsonr(xa[mask], ya[mask]).statistic)
    return val if math.isfinite(val) else 0.0


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if int(mask.sum()) < 2:
        return float("nan")
    yt = yt[mask]
    yp = yp[mask]
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 1e-15:
        return float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    val = 1.0 - ss_res / ss_tot
    return val if math.isfinite(val) else float("nan")


def _safe_mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


# --------------------------------------------------------------------------- #
# Closed-Form SVD / Sufficient-Statistics Ridge Solver with Inner Group CV
# --------------------------------------------------------------------------- #


@dataclass
class FoldSufficientStats:
    """Exact second-order sufficient statistics for one CV fold (in shifted coordinates)."""

    n: int
    sum_x: np.ndarray      # (p,) float64
    gram_x: np.ndarray     # (p, p) float64: X_shifted^T X_shifted
    sum_xy: np.ndarray     # (p,) float64: X_shifted^T y
    sum_y: float
    sum_y2: float

    @classmethod
    def empty(cls, dim: int) -> FoldSufficientStats:
        return cls(
            n=0,
            sum_x=np.zeros(dim, dtype=np.float64),
            gram_x=np.zeros((dim, dim), dtype=np.float64),
            sum_xy=np.zeros(dim, dtype=np.float64),
            sum_y=0.0,
            sum_y2=0.0,
        )

    def add_batch(self, x: np.ndarray, y: np.ndarray) -> None:
        xf = np.asarray(x, dtype=np.float64)
        yf = np.asarray(y, dtype=np.float64)
        if xf.shape[0] == 0:
            return
        self.n += int(xf.shape[0])
        self.sum_x += xf.sum(axis=0)
        self.gram_x += xf.T @ xf
        self.sum_xy += xf.T @ yf
        self.sum_y += float(yf.sum())
        self.sum_y2 += float(np.dot(yf, yf))

    @classmethod
    def from_array_fast(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        device: torch.device | None = None,
        chunk_size: int = 32_768,
    ) -> FoldSufficientStats:
        """Accumulate exact sufficient statistics using fast GPU FP32 GEMM -> FP64 sum when on CUDA."""
        n, p = x.shape
        if n == 0:
            return cls.empty(p)
        if device is None or device.type != "cuda":
            st = cls.empty(p)
            st.add_batch(x, y)
            return st
        sx = torch.zeros(p, dtype=torch.float64, device=device)
        gx = torch.zeros((p, p), dtype=torch.float64, device=device)
        sxy = torch.zeros(p, dtype=torch.float64, device=device)
        sy = 0.0
        sy2 = 0.0
        for s in range(0, n, max(1, int(chunk_size))):
            e = min(n, s + int(chunk_size))
            xb = torch.from_numpy(np.asarray(x[s:e], dtype=np.float32)).to(device=device)
            yb = torch.from_numpy(np.asarray(y[s:e], dtype=np.float32)).to(device=device)
            sx += xb.sum(dim=0, dtype=torch.float64)
            gx += (xb.T @ xb).to(dtype=torch.float64)
            sxy += (xb.T @ yb).to(dtype=torch.float64)
            sy += float(yb.sum(dtype=torch.float64).item())
            sy2 += float((yb.to(dtype=torch.float64) ** 2).sum().item())
        return cls(
            n=int(n),
            sum_x=sx.cpu().numpy(),
            gram_x=gx.cpu().numpy(),
            sum_xy=sxy.cpu().numpy(),
            sum_y=sy,
            sum_y2=sy2,
        )

    def copy(self) -> FoldSufficientStats:
        return FoldSufficientStats(
            n=int(self.n),
            sum_x=self.sum_x.copy(),
            gram_x=self.gram_x.copy(),
            sum_xy=self.sum_xy.copy(),
            sum_y=float(self.sum_y),
            sum_y2=float(self.sum_y2),
        )


def _sum_fold_stats(folds: Sequence[FoldSufficientStats]) -> FoldSufficientStats:
    dim = int(folds[0].sum_x.shape[0])
    tot = FoldSufficientStats.empty(dim)
    for f in folds:
        tot.n += f.n
        tot.sum_x += f.sum_x
        tot.gram_x += f.gram_x
        tot.sum_xy += f.sum_xy
        tot.sum_y += f.sum_y
        tot.sum_y2 += f.sum_y2
    return tot


def _subtract_fold_stats(
    total: FoldSufficientStats,
    held_out: FoldSufficientStats,
) -> FoldSufficientStats:
    return FoldSufficientStats(
        n=total.n - held_out.n,
        sum_x=total.sum_x - held_out.sum_x,
        gram_x=total.gram_x - held_out.gram_x,
        sum_xy=total.sum_xy - held_out.sum_xy,
        sum_y=total.sum_y - held_out.sum_y,
        sum_y2=total.sum_y2 - held_out.sum_y2,
    )


def _solve_ridge_path_from_stats(
    stats: FoldSufficientStats,
    lambdas: Sequence[float],
    *,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve standardized closed-form Ridge for all ``lambdas`` in a single eigendecomposition.

    Returns ``(W_orig, b_orig)`` with shapes ``(len(lambdas), p)`` and ``(len(lambdas),)``
    so predictions on unstandardized features are simply ``X @ W_orig[k] + b_orig[k]``.
    """
    n = max(1, int(stats.n))
    mu_x = stats.sum_x / float(n)
    mu_y = stats.sum_y / float(n)

    # Centered Gram matrix C = (X - mu_x)^T (X - mu_x) and RHS c = (X - mu_x)^T (y - mu_y)
    cov_sum = stats.gram_x - float(n) * np.outer(mu_x, mu_x)
    cov_sum = 0.5 * (cov_sum + cov_sum.T)
    rhs = stats.sum_xy - float(n) * mu_y * mu_x

    var_x = np.maximum(np.diag(cov_sum) / float(n), 0.0)
    std_x = np.sqrt(var_x)
    active = std_x > _STD_EPS
    inv_std = np.where(active, 1.0 / np.where(active, std_x, 1.0), 0.0)

    # Standardized Gram matrix G_std = D^{-1} C D^{-1}, RHS c_std = D^{-1} rhs
    g_std = (cov_sum * inv_std[:, None]) * inv_std[None, :]
    g_std = 0.5 * (g_std + g_std.T)
    c_std = rhs * inv_std

    if device is not None and device.type == "cuda" and g_std.shape[0] >= 64:
        # Fast GPU FP32 eigh (~35x faster than FP64 cuSOLVER on L20) with FP64 projection
        g_t = torch.from_numpy(g_std.astype(np.float32, copy=False)).to(device=device)
        evals_t, evecs_t = torch.linalg.eigh(g_t)
        evals = torch.clamp(evals_t, min=0.0).to(dtype=torch.float64).cpu().numpy()
        evecs = evecs_t.to(dtype=torch.float64).cpu().numpy()
        proj = evecs.T @ c_std
    else:
        evals, evecs = np.linalg.eigh(g_std)
        evals = np.maximum(evals, 0.0)
        proj = evecs.T @ c_std

    lam_arr = np.asarray(lambdas, dtype=np.float64)
    # W_std[k] = V @ ((V^T c_std) / (evals + lam_k))
    inv_filt = proj[None, :] / (evals[None, :] + lam_arr[:, None])  # (K, p)
    w_std = inv_filt @ evecs.T  # (K, p)

    # Map back to original (shifted) feature coordinates
    w_orig = w_std * inv_std[None, :]
    b_orig = mu_y - (w_orig @ mu_x)
    return w_orig, b_orig


def _exact_fold_sse_from_stats(
    val_stats: FoldSufficientStats,
    w_orig: np.ndarray,
    b_orig: np.ndarray,
) -> np.ndarray:
    """Compute exact validation SSE ``||X_val w + b - y_val||_2^2`` for all K lambdas in O(K p^2)."""
    # w_orig: (K, p), b_orig: (K,)
    gw = w_orig @ val_stats.gram_x  # (K, p)
    quad = np.einsum("kp,kp->k", w_orig, gw)
    sx_w = w_orig @ val_stats.sum_x
    c_w = w_orig @ val_stats.sum_xy
    sse = (
        quad
        + 2.0 * b_orig * sx_w
        - 2.0 * c_w
        + float(val_stats.n) * (b_orig ** 2)
        - 2.0 * b_orig * val_stats.sum_y
        + val_stats.sum_y2
    )
    return np.maximum(sse, 0.0)


def fit_ridge_from_stats_cv(
    fold_stats: Sequence[FoldSufficientStats],
    *,
    lambdas: Sequence[float] = DEFAULT_RIDGE_LAMBDAS,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Select optimal lambda via exact leave-one-fold-out CV on sufficient stats, then refit on all folds."""
    lam_list = [float(lam) for lam in lambdas]
    if not lam_list:
        raise ValueError("lambdas must be non-empty")
    valid_folds = [f for f in fold_stats if f.n > 0]
    if not valid_folds:
        raise ValueError("All CV folds are empty")

    total = _sum_fold_stats(valid_folds)
    n_folds = len(valid_folds)

    if n_folds >= 2:
        cv_sse = np.zeros(len(lam_list), dtype=np.float64)
        cv_n = 0
        for f_val in valid_folds:
            f_tr = _subtract_fold_stats(total, f_val)
            if f_tr.n <= 0:
                continue
            w_path, b_path = _solve_ridge_path_from_stats(f_tr, lam_list, device=device)
            cv_sse += _exact_fold_sse_from_stats(f_val, w_path, b_path)
            cv_n += f_val.n
        cv_mse = cv_sse / float(max(1, cv_n))
        best_idx = int(np.argmin(cv_mse))
    else:
        cv_mse = np.full(len(lam_list), float("nan"), dtype=np.float64)
        best_idx = len(lam_list) // 2

    best_lam = float(lam_list[best_idx])
    w_full, b_full = _solve_ridge_path_from_stats(total, [best_lam], device=device)
    return {
        "weights": w_full[0],
        "intercept": float(b_full[0]),
        "best_lambda": best_lam,
        "cv_mse_by_lambda": {f"{lam:g}": float(cv_mse[i]) for i, lam in enumerate(lam_list)},
    }


def _solve_ridge_path_direct_svd(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    lambdas: Sequence[float],
    *,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact thin-SVD closed-form Ridge path on standardized ``x_tr`` (optimal when N <= p)."""
    xf = np.asarray(x_tr, dtype=np.float64)
    yf = np.asarray(y_tr, dtype=np.float64)
    n, p = xf.shape
    mu_x = xf.mean(axis=0)
    mu_y = float(yf.mean())
    xc = xf - mu_x[None, :]
    yc = yf - mu_y
    std_x = np.sqrt(np.maximum(np.mean(xc * xc, axis=0), 0.0))
    active = std_x > _STD_EPS
    inv_std = np.where(active, 1.0 / np.where(active, std_x, 1.0), 0.0)
    x_std = xc * inv_std[None, :]

    if device is not None and device.type == "cuda" and min(n, p) >= 64:
        xs_t = torch.from_numpy(x_std).to(device=device, dtype=torch.float64)
        yc_t = torch.from_numpy(yc).to(device=device, dtype=torch.float64)
        u_t, s_t, vh_t = torch.linalg.svd(xs_t, full_matrices=False)
        uty = (u_t.T @ yc_t).cpu().numpy()
        s = s_t.cpu().numpy()
        vh = vh_t.cpu().numpy()
    else:
        u, s, vh = np.linalg.svd(x_std, full_matrices=False)
        uty = u.T @ yc

    lam_arr = np.asarray(lambdas, dtype=np.float64)
    filt = (s[None, :] / (s[None, :] ** 2 + lam_arr[:, None])) * uty[None, :]
    w_std = filt @ vh
    w_orig = w_std * inv_std[None, :]
    b_orig = mu_y - (w_orig @ mu_x)
    return w_orig, b_orig


def fit_ridge_svd_cv(
    x_train: np.ndarray,
    y_train: np.ndarray,
    group_ids: np.ndarray,
    *,
    lambdas: Sequence[float] = DEFAULT_RIDGE_LAMBDAS,
    n_cv_folds: int = 5,
    device: torch.device | None = None,
    return_oof_preds: bool = False,
) -> dict[str, Any]:
    """Closed-form SVD/Gram Ridge with leakage-free group-disjoint inner CV.

    Automatically uses thin SVD when ``N <= p`` and exact sufficient-statistics
    eigendecomposition when ``N > p`` (mathematically identical to SVD Ridge).
    """
    xf = np.asarray(x_train, dtype=np.float64)
    yf = np.asarray(y_train, dtype=np.float64)
    gids = np.asarray(group_ids)
    n, p = xf.shape
    lam_list = [float(lam) for lam in lambdas]

    uniq_groups = sorted(set(gids.tolist()))
    eff_folds = min(max(1, int(n_cv_folds)), len(uniq_groups))
    group_to_fold = {g: idx % eff_folds for idx, g in enumerate(uniq_groups)}
    fold_ids = np.asarray([group_to_fold[g] for g in gids.tolist()], dtype=np.int64)

    # Shift features by global train mean to eliminate Gram cancellation error
    shift_mu = xf.mean(axis=0)
    x_shifted = xf - shift_mu[None, :]

    oof_preds: np.ndarray | None = None
    if eff_folds >= 2 and n > p:
        folds = [
            FoldSufficientStats.from_array_fast(
                x_shifted[fold_ids == f_idx], yf[fold_ids == f_idx], device=device
            )
            for f_idx in range(eff_folds)
        ]
        res = fit_ridge_from_stats_cv(folds, lambdas=lam_list, device=device)
        best_lam = float(res["best_lambda"])
        if return_oof_preds:
            oof_preds = np.empty(n, dtype=np.float64)
            total = _sum_fold_stats(folds)
            for f_idx in range(eff_folds):
                val_m = fold_ids == f_idx
                if int(val_m.sum()) == 0:
                    continue
                f_tr = _subtract_fold_stats(total, folds[f_idx])
                if f_tr.n <= 0:
                    f_tr = total
                w_f, b_f = _solve_ridge_path_from_stats(f_tr, [best_lam], device=device)
                oof_preds[val_m] = x_shifted[val_m] @ w_f[0] + float(b_f[0])
        w = res["weights"]
        b = float(res["intercept"] - float(np.dot(w, shift_mu)))
        return {
            "weights": w,
            "intercept": b,
            "best_lambda": best_lam,
            "cv_mse_by_lambda": res["cv_mse_by_lambda"],
            "oof_predictions": oof_preds,
        }

    # Explicit fold loop (used when N <= p or when OOF predictions are requested)
    if eff_folds >= 2:
        cv_sse = np.zeros(len(lam_list), dtype=np.float64)
        oof_by_lam = np.empty((len(lam_list), n), dtype=np.float64) if return_oof_preds else None
        for f_idx in range(eff_folds):
            val_m = fold_ids == f_idx
            tr_m = ~val_m
            if int(tr_m.sum()) <= 0 or int(val_m.sum()) <= 0:
                continue
            if int(tr_m.sum()) <= p:
                w_path, b_path = _solve_ridge_path_direct_svd(
                    x_shifted[tr_m], yf[tr_m], lam_list, device=device
                )
            else:
                tr_stats = FoldSufficientStats.empty(p)
                tr_stats.add_batch(x_shifted[tr_m], yf[tr_m])
                w_path, b_path = _solve_ridge_path_from_stats(
                    tr_stats, lam_list, device=device
                )
            preds_val = x_shifted[val_m] @ w_path.T + b_path[None, :]  # (n_val, K)
            err = preds_val - yf[val_m, None]
            cv_sse += np.sum(err * err, axis=0)
            if oof_by_lam is not None:
                oof_by_lam[:, val_m] = preds_val.T
        cv_mse = cv_sse / float(max(1, n))
        best_idx = int(np.argmin(cv_mse))
        if oof_by_lam is not None:
            oof_preds = oof_by_lam[best_idx]
    else:
        cv_mse = np.full(len(lam_list), float("nan"), dtype=np.float64)
        best_idx = len(lam_list) // 2

    best_lam = float(lam_list[best_idx])
    if n <= p:
        w_full, b_full = _solve_ridge_path_direct_svd(
            x_shifted, yf, [best_lam], device=device
        )
    else:
        full_stats = FoldSufficientStats.empty(p)
        full_stats.add_batch(x_shifted, yf)
        w_full, b_full = _solve_ridge_path_from_stats(
            full_stats, [best_lam], device=device
        )
    w = w_full[0]
    b = float(b_full[0] - float(np.dot(w, shift_mu)))
    return {
        "weights": w,
        "intercept": b,
        "best_lambda": best_lam,
        "cv_mse_by_lambda": {f"{lam:g}": float(cv_mse[i]) for i, lam in enumerate(lam_list)},
        "oof_predictions": oof_preds,
    }


# --------------------------------------------------------------------------- #
# 4. Eq. (3b) Temperature Calibration on eval_unseen_fold
# --------------------------------------------------------------------------- #


@torch.no_grad()
def compute_weighted_temperature_calibrated_bits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    log_tau_bounds: tuple[float, float] = (-4.0, 6.0),
    n_iter: int = 32,
    chunk_size: int = 262_144,
) -> dict[str, float]:
    """Exact Eq. (3b) scalar temperature scaling ``min_{T>0} H(theta, T)`` with optional token weights.

    Minimizes weighted cross-entropy in bits/residue over ``T = exp(log_tau)``, returning both
    canonical theory symbols (``H_raw``, ``H_cal``, ``T_opt``, ``Delta_cal``) and legacy keys
    (``raw_bits_per_tok``, ``calibrated_bits_per_tok``, ``optimal_temperature``,
    ``calibration_excess_bits``).
    """
    n_tok = int(targets.numel())
    if n_tok == 0:
        return {
            "H_raw": float("nan"),
            "H_cal": float("nan"),
            "T_opt": float("nan"),
            "Delta_cal": float("nan"),
            "raw_bits_per_tok": float("nan"),
            "calibrated_bits_per_tok": float("nan"),
            "optimal_temperature": float("nan"),
            "calibration_excess_bits": float("nan"),
        }

    flat_logits = logits.reshape(n_tok, logits.shape[-1])
    flat_targets = targets.reshape(n_tok).long()
    flat_weights = None if weights is None else weights.reshape(n_tok).to(dtype=torch.float64)
    weight_denom = (
        float(n_tok)
        if flat_weights is None
        else float(flat_weights.sum().item())
    )
    if weight_denom <= 0.0:
        weight_denom = 1.0

    def _bits_at_log_tau(log_tau: float) -> float:
        inv_tau = math.exp(-float(log_tau))
        total_nats = 0.0
        for s in range(0, n_tok, max(1, int(chunk_size))):
            e = min(n_tok, s + int(chunk_size))
            lg = flat_logits[s:e].float() * inv_tau
            tg = flat_targets[s:e]
            if flat_weights is None:
                total_nats += float(F.cross_entropy(lg, tg, reduction="sum").item())
            else:
                ce = F.cross_entropy(lg, tg, reduction="none").to(dtype=torch.float64)
                total_nats += float((ce * flat_weights[s:e]).sum().item())
        return float(total_nats / (weight_denom * LN2))

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
    cand_bits = _bits_at_log_tau(best_log_tau)
    if cand_bits < raw_bits:
        cal_bits = float(cand_bits)
        opt_tau = float(math.exp(best_log_tau))
    else:
        cal_bits = float(raw_bits)
        opt_tau = 1.0
    delta_cal = float(max(0.0, raw_bits - cal_bits))
    return {
        "H_raw": float(raw_bits),
        "H_cal": float(cal_bits),
        "T_opt": float(opt_tau),
        "Delta_cal": float(delta_cal),
        "raw_bits_per_tok": float(raw_bits),
        "calibrated_bits_per_tok": float(cal_bits),
        "optimal_temperature": float(opt_tau),
        "calibration_excess_bits": float(delta_cal),
    }


@torch.no_grad()
def evaluate_unseen_fold_temperature_calibration(
    model: nn.Module,
    unseen_fold_store: PackedSequenceStore,
    *,
    objective: str,
    batch_size: int = 128,
    precision: str = "bf16",
    n_draws: int = NUM_EVAL_STRATA,
    eval_seed: int = 20261129,
) -> dict[str, Any]:
    """Compute exact Eq. (3b) ``H_unseen_fold``, ``H_cal``, ``T_opt``, and ``Delta_cal`` on ``eval_unseen_fold``."""
    obj = canonical_objective(objective)
    if unseen_fold_store.n_sequences == 0:
        raise ValueError("Cannot evaluate temperature calibration on empty PackedSequenceStore")

    t0 = time.perf_counter()
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    autocast_ctx = _eval_autocast_ctx(device, precision)

    order_idx = np.argsort(unseen_fold_store.lengths, kind="mergesort")
    n_seq = unseen_fold_store.n_sequences

    logits_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []

    try:
        for start in range(0, n_seq, max(1, int(batch_size))):
            batch_global_np = order_idx[start : start + int(batch_size)]
            tokens, lengths = unseen_fold_store.collate_indices(
                batch_global_np, device=device, pad_to_multiple=8
            )
            bsz, max_len = tokens.shape
            global_idx_t = torch.from_numpy(batch_global_np).to(device=device, dtype=torch.int64)
            pos = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
            valid = pos < lengths.unsqueeze(1)

            if obj == "ar":
                prep = prepare_ar_batch(tokens, lengths)
                with autocast_ctx:
                    logits = model(prep.input_ids, prep.attention_mask, right_padded=True)
                sel = prep.loss_mask
                logits_chunks.append(logits[sel].to(dtype=torch.float16).cpu())
                target_chunks.append(prep.targets[sel].to(dtype=torch.int16).cpu())
                weight_chunks.append(torch.ones(int(sel.sum().item()), dtype=torch.float32))

            elif obj == "oa_uniform":
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
                    inp = torch.where(mask, MASK_TOKEN_ID, tokens)
                    with autocast_ctx:
                        logits = model(inp, valid)
                    # Exact OA-ARDM residue-weighted contribution per masked token: L_i / (S * m_{i,s})
                    tok_w = (
                        lengths.to(dtype=torch.float32)
                        / (float(n_draws) * m_mask.to(dtype=torch.float32))
                    ).unsqueeze(1).expand(bsz, max_len)
                    logits_chunks.append(logits[mask].to(dtype=torch.float16).cpu())
                    target_chunks.append(tokens[mask].to(dtype=torch.int16).cpu())
                    weight_chunks.append(tok_w[mask].cpu())

            elif obj == "mlm_15":
                k_mask = torch.clamp(
                    torch.round(lengths.to(dtype=torch.float32) * 0.15).to(dtype=torch.int64),
                    min=1,
                )
                k_mask = torch.minimum(k_mask, lengths.to(dtype=torch.int64))
                for d in range(int(n_draws)):
                    mask = _deterministic_valid_mask(
                        global_idx_t,
                        lengths,
                        k_mask,
                        max_len,
                        draw_idx=d,
                        eval_seed=eval_seed,
                    )
                    inp = torch.where(mask, MASK_TOKEN_ID, tokens)
                    with autocast_ctx:
                        logits = model(inp, valid)
                    tok_w = (
                        lengths.to(dtype=torch.float32)
                        / (float(n_draws) * k_mask.to(dtype=torch.float32))
                    ).unsqueeze(1).expand(bsz, max_len)
                    logits_chunks.append(logits[mask].to(dtype=torch.float16).cpu())
                    target_chunks.append(tokens[mask].to(dtype=torch.int16).cpu())
                    weight_chunks.append(tok_w[mask].cpu())
    finally:
        if was_training:
            model.train()

    all_logits = torch.cat(logits_chunks, dim=0)
    all_targets = torch.cat(target_chunks, dim=0).long()
    all_weights = torch.cat(weight_chunks, dim=0)

    cal_dev = device if device.type == "cuda" and all_logits.numel() <= 100_000_000 else torch.device("cpu")
    cal = compute_weighted_temperature_calibrated_bits(
        all_logits.to(cal_dev),
        all_targets.to(cal_dev),
        all_weights.to(cal_dev),
    )
    return {
        "objective": obj,
        "n_sequences": int(n_seq),
        "total_residues": int(unseen_fold_store.total_tokens),
        "n_eval_tokens": int(all_targets.numel()),
        "H_unseen_fold": float(cal["H_raw"]),
        "H_unseen_fold_cal": float(cal["H_cal"]),
        "H_cal": float(cal["H_cal"]),
        "T_opt": float(cal["T_opt"]),
        "Delta_cal": float(cal["Delta_cal"]),
        **cal,
        "eval_wall_sec": float(time.perf_counter() - t0),
    }


# --------------------------------------------------------------------------- #
# 1. Probe 1 — MegaScale ddG Numerical Ridge Probe
# --------------------------------------------------------------------------- #


def _select_megascale_variants(
    variants: Sequence[Any],
    max_variants_per_wt: int | None,
) -> list[Any]:
    if max_variants_per_wt is None or len(variants) <= int(max_variants_per_wt):
        return list(variants)
    # Deterministic uniform stride matching evaluate_megascale_ddg
    step = max(1, len(variants) // int(max_variants_per_wt))
    return list(variants[::step][: int(max_variants_per_wt)])


def _summarize_megascale_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    wt_indices: np.ndarray,
    wt_strata: Sequence[str],
    *,
    affine_calibrate_from_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute per-stratum WT-averaged Spearman, Pearson, and R^2 (plus pooled metrics)."""
    by_stratum_wt: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_stratum_idx: dict[str, list[int]] = defaultdict(list)

    # Optional 1D linear calibration fitted strictly on train WTs for raw zero-shot logit R^2
    preds_for_r2 = preds
    if affine_calibrate_from_mask is not None and int(affine_calibrate_from_mask.sum()) >= 3:
        px = preds[affine_calibrate_from_mask].astype(np.float64)
        py = targets[affine_calibrate_from_mask].astype(np.float64)
        var_x = float(np.var(px))
        if var_x > 1e-15:
            slope = float(np.cov(px, py, ddof=0)[0, 1] / var_x)
            intercept = float(np.mean(py) - slope * np.mean(px))
            preds_for_r2 = slope * preds + intercept

    # Group variant indices by WT in a single O(N) pass
    wt_to_positions: dict[int, list[int]] = defaultdict(list)
    for pos_i, wt_i in enumerate(wt_indices.tolist()):
        wt_to_positions[int(wt_i)].append(pos_i)

    by_stratum_pos: dict[str, list[np.ndarray]] = defaultdict(list)
    for wt_i, pos_list in sorted(wt_to_positions.items()):
        if len(pos_list) < 3:
            continue
        idx_arr = np.asarray(pos_list, dtype=np.int64)
        st = str(wt_strata[wt_i])
        by_stratum_idx[st].append(wt_i)
        by_stratum_pos[st].append(idx_arr)
        by_stratum_wt[st].append(
            {
                "spearman": _safe_spearman(preds[idx_arr], targets[idx_arr]),
                "pearson": _safe_pearson(preds[idx_arr], targets[idx_arr]),
                "r2": _safe_r2(targets[idx_arr], preds_for_r2[idx_arr]),
            }
        )

    out_strata: dict[str, dict[str, Any]] = {}
    for st in sorted(by_stratum_wt.keys()):
        rows = by_stratum_wt[st]
        st_idx = np.concatenate(by_stratum_pos[st])
        out_strata[st] = {
            "n_domains": len(rows),
            "n_variants": int(st_idx.size),
            "spearman_mean": _safe_mean([r["spearman"] for r in rows]),
            "pearson_mean": _safe_mean([r["pearson"] for r in rows]),
            "r2_mean": _safe_mean([r["r2"] for r in rows]),
            "pooled_spearman": _safe_spearman(preds[st_idx], targets[st_idx]),
            "pooled_pearson": _safe_pearson(preds[st_idx], targets[st_idx]),
            "pooled_r2": _safe_r2(targets[st_idx], preds_for_r2[st_idx]),
        }

    # Combined OOD natural + de_novo summary
    ood_rows = by_stratum_wt.get("unseen_fold_natural", []) + by_stratum_wt.get("de_novo_design", [])
    if ood_rows:
        ood_pos_parts = by_stratum_pos.get("unseen_fold_natural", []) + by_stratum_pos.get("de_novo_design", [])
        ood_idx = np.concatenate(ood_pos_parts)
        out_strata["ood_unseen_and_de_novo"] = {
            "n_domains": len(ood_rows),
            "n_variants": int(ood_idx.size),
            "spearman_mean": _safe_mean([r["spearman"] for r in ood_rows]),
            "pearson_mean": _safe_mean([r["pearson"] for r in ood_rows]),
            "r2_mean": _safe_mean([r["r2"] for r in ood_rows]),
            "pooled_spearman": _safe_spearman(preds[ood_idx], targets[ood_idx]),
            "pooled_pearson": _safe_pearson(preds[ood_idx], targets[ood_idx]),
            "pooled_r2": _safe_r2(targets[ood_idx], preds_for_r2[ood_idx]),
        }
    return out_strata


@torch.no_grad()
def _score_mlm_masked_marginals_with_temp(
    model: VariableLengthProteinTransformer,
    wt_enc: np.ndarray,
    uniq_pos: np.ndarray,
    *,
    batch_size: int,
    precision: str,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(raw_masked_lp, cal_masked_lp)`` of shape ``(len(uniq_pos), 20)``."""
    device = next(model.parameters()).device
    autocast_ctx = _eval_autocast_ctx(device, precision)
    L = int(wt_enc.size)
    wt_t = torch.from_numpy(wt_enc).to(device=device, dtype=torch.long)
    n_pos = int(uniq_pos.size)
    raw_lp = np.empty((n_pos, VOCAB_AA), dtype=np.float32)
    cal_lp = np.empty((n_pos, VOCAB_AA), dtype=np.float32)
    inv_t = 1.0 / max(1e-6, float(temperature))

    for start in range(0, n_pos, max(1, int(batch_size))):
        p_chunk = uniq_pos[start : start + int(batch_size)]
        cur = int(p_chunk.size)
        inp = wt_t.unsqueeze(0).expand(cur, -1).clone()
        row_ar = torch.arange(cur, device=device, dtype=torch.long)
        pos_t = torch.from_numpy(p_chunk).to(device=device, dtype=torch.long)
        inp[row_ar, pos_t] = MASK_TOKEN_ID
        valid_b = torch.ones((cur, L), dtype=torch.bool, device=device)
        with autocast_ctx:
            logits = model(inp, valid_b)
        site_logits = logits[row_ar, pos_t].float()
        raw_lp[start : start + cur] = F.log_softmax(site_logits, dim=-1).cpu().numpy()
        cal_lp[start : start + cur] = F.log_softmax(site_logits * inv_t, dim=-1).cpu().numpy()
    return raw_lp, cal_lp


@torch.no_grad()
def evaluate_megascale_layerwise_probe(
    model: VariableLengthProteinTransformer,
    megascale_payload: Mapping[str, Any],
    *,
    objective: str | None = None,
    optimal_temperature: float = 1.0,
    train_stratum: str = "seen_pool_natural",
    eval_strata: Sequence[str] = ("unseen_fold_natural", "de_novo_design", "design_with_uniref_hit"),
    max_variants_per_wt: int | None = 400,
    batch_size: int = 512,
    precision: str = "bf16",
    lambdas: Sequence[float] = DEFAULT_RIDGE_LAMBDAS,
    n_cv_folds: int = 5,
    feature_storage_dtype: np.dtype[Any] | type[np.floating[Any]] = np.float16,
) -> dict[str, Any]:
    """Run Probe 1 (MegaScale ddG layerwise Ridge) + raw & temperature-calibrated zero-shot baselines.

    Extracts ``[h^(l)_i(mut) - h^(l)_i(wt), mean(h^(l)(mut)) - mean(h^(l)(wt))] in R^(2d)``
    in a single GPU pass per WT, fits inner-CV SVD/Gram Ridge strictly on ``train_stratum`` WTs,
    and evaluates out-of-fold on ``unseen_fold_natural`` and ``de_novo_design`` WTs.
    """
    t0 = time.perf_counter()
    obj = canonical_objective(objective if objective is not None else model.objective)
    n_taps = len(model.blocks) + 2
    layer_names = get_layer_names(len(model.blocks))
    d_model = int(model.spec.d_model)
    feat_dim = 2 * d_model
    device = next(model.parameters()).device
    inv_t = 1.0 / max(1e-6, float(optimal_temperature))

    allowed_strata = {str(train_stratum), *(str(s) for s in eval_strata)}
    raw_domains = [
        d for d in megascale_payload.get("domains", [])
        if str(d.get("stratum", "unknown")) in allowed_strata
    ]

    # Pre-scan valid WTs and variants to preallocate contiguous compact feature buffers
    selected_domains: list[dict[str, Any]] = []
    total_variants = 0
    for dom in raw_domains:
        sel_vars = _select_megascale_variants(dom["variants"], max_variants_per_wt)
        if len(sel_vars) < 3:
            continue
        selected_domains.append(
            {
                "wt_name": str(dom.get("wt_name", f"wt_{len(selected_domains)}")),
                "stratum": str(dom.get("stratum", "unknown")),
                "wildtype": str(dom["wildtype"]),
                "variants": sel_vars,
            }
        )
        total_variants += len(sel_vars)

    if total_variants == 0:
        raise ValueError("No valid MegaScale variants found for selected strata")

    feat_buffers = [
        np.empty((total_variants, feat_dim), dtype=feature_storage_dtype)
        for _ in range(n_taps)
    ]
    targets_all = np.empty(total_variants, dtype=np.float64)
    wt_idx_all = np.empty(total_variants, dtype=np.int64)
    wt_strata = [d["stratum"] for d in selected_domains]

    zs_primary_raw = np.empty(total_variants, dtype=np.float64)
    zs_primary_cal = np.empty(total_variants, dtype=np.float64)
    zs_wtmarg_raw = np.empty(total_variants, dtype=np.float64) if obj != "ar" else None
    zs_wtmarg_cal = np.empty(total_variants, dtype=np.float64) if obj != "ar" else None

    was_training = model.training
    model.eval()
    cursor = 0
    try:
        for wt_i, dom in enumerate(selected_domains):
            wt_enc = encode_aa_sequence(dom["wildtype"])
            L = int(wt_enc.size)
            variants = dom["variants"]
            n_var = len(variants)

            pos_arr = np.asarray([int(v[0]) for v in variants], dtype=np.int64)
            wt_ids = np.asarray([AA_TO_ID[str(v[1])] for v in variants], dtype=np.int64)
            mut_ids = np.asarray([AA_TO_ID[str(v[2])] for v in variants], dtype=np.int64)
            y_arr = np.asarray([float(v[3]) for v in variants], dtype=np.float64)

            # 1) Forward pass on WT sequence
            wt_tok_t = torch.from_numpy(wt_enc).unsqueeze(0).to(device=device, dtype=torch.long)
            wt_len_t = torch.tensor([L], device=device, dtype=torch.long)
            wt_layers, wt_logits = forward_layerwise_hidden_states(
                model,
                wt_tok_t,
                wt_len_t,
                objective=obj,
                precision=precision,
                return_logits=True,
                output_dtype=torch.float32,
            )
            assert wt_logits is not None
            wt_h = [t[0] for t in wt_layers]            # list of (L, d) on GPU
            wt_mean = [t[0].mean(dim=0) for t in wt_layers]  # list of (d,) on GPU

            if obj == "ar":
                wt_nll_raw = float(
                    F.cross_entropy(
                        wt_logits[0].float(), wt_tok_t[0], reduction="sum"
                    ).item()
                )
                wt_nll_cal = float(
                    F.cross_entropy(
                        wt_logits[0].float() * inv_t, wt_tok_t[0], reduction="sum"
                    ).item()
                )
            else:
                wt_lp_raw = F.log_softmax(wt_logits[0].float(), dim=-1).cpu().numpy()
                wt_lp_cal = F.log_softmax(wt_logits[0].float() * inv_t, dim=-1).cpu().numpy()
                assert zs_wtmarg_raw is not None and zs_wtmarg_cal is not None
                zs_wtmarg_raw[cursor : cursor + n_var] = (
                    wt_lp_raw[pos_arr, mut_ids] - wt_lp_raw[pos_arr, wt_ids]
                )
                zs_wtmarg_cal[cursor : cursor + n_var] = (
                    wt_lp_cal[pos_arr, mut_ids] - wt_lp_cal[pos_arr, wt_ids]
                )
                uniq_pos = np.unique(pos_arr)
                pos_to_idx = {int(p): i for i, p in enumerate(uniq_pos)}
                mm_raw_lp, mm_cal_lp = _score_mlm_masked_marginals_with_temp(
                    model,
                    wt_enc,
                    uniq_pos,
                    batch_size=batch_size,
                    precision=precision,
                    temperature=optimal_temperature,
                )
                mapped = np.asarray([pos_to_idx[int(p)] for p in pos_arr], dtype=np.int64)
                zs_primary_raw[cursor : cursor + n_var] = (
                    mm_raw_lp[mapped, mut_ids] - mm_raw_lp[mapped, wt_ids]
                )
                zs_primary_cal[cursor : cursor + n_var] = (
                    mm_cal_lp[mapped, mut_ids] - mm_cal_lp[mapped, wt_ids]
                )

            # 2) Forward pass on all single-mutant sequences of this WT directly on GPU (0 pad!)
            pos_all_t = torch.from_numpy(pos_arr).to(device=device, dtype=torch.long)
            mut_ids_t = torch.from_numpy(mut_ids).to(device=device, dtype=torch.long)

            for s in range(0, n_var, max(1, int(batch_size))):
                e = min(n_var, s + int(batch_size))
                cur = e - s
                pos_b = pos_all_t[s:e]
                row_ar = torch.arange(cur, device=device, dtype=torch.long)
                tok_b = wt_tok_t.expand(cur, L).clone()
                tok_b[row_ar, pos_b] = mut_ids_t[s:e]
                len_b = torch.full((cur,), L, device=device, dtype=torch.long)

                mut_layers, mut_logits = forward_layerwise_hidden_states(
                    model,
                    tok_b,
                    len_b,
                    objective=obj,
                    precision=precision,
                    return_logits=(obj == "ar"),
                    output_dtype=torch.float32,
                )
                stacked_feats = torch.empty(
                    (n_taps, cur, feat_dim),
                    device=device,
                    dtype=torch.float16 if feature_storage_dtype == np.float16 else torch.float32,
                )
                for tap in range(n_taps):
                    hm = mut_layers[tap]  # (cur, L, d)
                    d_site = hm[row_ar, pos_b] - wt_h[tap][pos_b]
                    d_mean = hm.mean(dim=1) - wt_mean[tap].unsqueeze(0)
                    stacked_feats[tap, :, :d_model] = d_site
                    stacked_feats[tap, :, d_model:] = d_mean

                feats_np = stacked_feats.cpu().numpy()
                for tap in range(n_taps):
                    feat_buffers[tap][cursor + s : cursor + e] = feats_np[tap]

                if obj == "ar":
                    assert mut_logits is not None
                    nll_raw = F.cross_entropy(
                        mut_logits.float().transpose(1, 2), tok_b, reduction="none"
                    ).to(torch.float64).sum(dim=-1)
                    nll_cal = F.cross_entropy(
                        (mut_logits.float() * inv_t).transpose(1, 2), tok_b, reduction="none"
                    ).to(torch.float64).sum(dim=-1)
                    zs_primary_raw[cursor + s : cursor + e] = wt_nll_raw - nll_raw.cpu().numpy()
                    zs_primary_cal[cursor + s : cursor + e] = wt_nll_cal - nll_cal.cpu().numpy()

            targets_all[cursor : cursor + n_var] = y_arr
            wt_idx_all[cursor : cursor + n_var] = wt_i
            cursor += n_var
    finally:
        if was_training:
            model.train()

    t_extract = time.perf_counter() - t0

    # Partition strictly into train_stratum vs held-out strata
    train_wt_ids = [i for i, st in enumerate(wt_strata) if st == str(train_stratum)]
    if not train_wt_ids:
        raise ValueError(f"No MegaScale WTs found in train_stratum={train_stratum!r}")
    train_mask = np.isin(wt_idx_all, train_wt_ids)
    y_train = targets_all[train_mask]
    g_train = wt_idx_all[train_mask]

    # Zero-shot raw & temperature-calibrated baselines
    zeroshot_raw_strata = _summarize_megascale_predictions(
        zs_primary_raw,
        targets_all,
        wt_idx_all,
        wt_strata,
        affine_calibrate_from_mask=train_mask,
    )
    zeroshot_cal_strata = _summarize_megascale_predictions(
        zs_primary_cal,
        targets_all,
        wt_idx_all,
        wt_strata,
        affine_calibrate_from_mask=train_mask,
    )
    zeroshot_block: dict[str, Any] = {
        "scoring_method": "exact_delta_logp" if obj == "ar" else "masked_marginal",
        "optimal_temperature_used": float(optimal_temperature),
        "raw": zeroshot_raw_strata,
        "temperature_calibrated": zeroshot_cal_strata,
    }
    if zs_wtmarg_raw is not None and zs_wtmarg_cal is not None:
        zeroshot_block["wt_marginal_raw"] = _summarize_megascale_predictions(
            zs_wtmarg_raw,
            targets_all,
            wt_idx_all,
            wt_strata,
            affine_calibrate_from_mask=train_mask,
        )
        zeroshot_block["wt_marginal_temperature_calibrated"] = _summarize_megascale_predictions(
            zs_wtmarg_cal,
            targets_all,
            wt_idx_all,
            wt_strata,
            affine_calibrate_from_mask=train_mask,
        )

    # Fit closed-form Ridge probe per layer strictly on train_stratum WTs
    t_fit0 = time.perf_counter()
    per_layer: list[dict[str, Any]] = []
    uniq_train_wts = sorted(train_wt_ids)
    eff_cv_folds = min(max(1, int(n_cv_folds)), len(uniq_train_wts))
    wt_to_fold = {w: idx % eff_cv_folds for idx, w in enumerate(uniq_train_wts)}
    train_fold_ids = np.asarray([wt_to_fold[int(w)] for w in g_train.tolist()], dtype=np.int64)
    use_gpu_fit = device.type == "cuda"

    for tap in range(n_taps):
        x_buf = feat_buffers[tap]
        if use_gpu_fit and int(train_mask.sum()) > feat_dim and eff_cv_folds >= 2:
            # Fast GPU FP32->FP64 fold sufficient-statistics Ridge (zero float64 CPU bloat)
            x_tr_f32 = x_buf[train_mask].astype(np.float32, copy=False)
            shift_mu = x_tr_f32.mean(axis=0, dtype=np.float64).astype(np.float32)
            folds = []
            for f_idx in range(eff_cv_folds):
                fm = train_fold_ids == f_idx
                folds.append(
                    FoldSufficientStats.from_array_fast(
                        x_tr_f32[fm] - shift_mu[None, :], y_train[fm], device=device
                    )
                )
            fit_res = fit_ridge_from_stats_cv(folds, lambdas=lambdas, device=device)
            best_lam = float(fit_res["best_lambda"])
            w_shift = fit_res["weights"]
            b_shift = float(fit_res["intercept"])
            w_full = w_shift
            b_full = float(b_shift - float(np.dot(w_shift, shift_mu.astype(np.float64))))

            # Compute predictions on GPU in chunks
            preds_eval = np.empty(total_variants, dtype=np.float64)
            w_t = torch.from_numpy(w_full.astype(np.float32)).to(device=device)
            for cs in range(0, total_variants, 65_536):
                ce = min(total_variants, cs + 65_536)
                xb = torch.from_numpy(x_buf[cs:ce].astype(np.float32, copy=False)).to(device=device)
                preds_eval[cs:ce] = (xb @ w_t).cpu().numpy().astype(np.float64) + b_full

            # Leakage-free inner-CV OOF predictions for seen_pool_natural
            total_st = _sum_fold_stats(folds)
            oof_tr = np.empty(int(train_mask.sum()), dtype=np.float64)
            for f_idx in range(eff_cv_folds):
                fm = train_fold_ids == f_idx
                f_tr = _subtract_fold_stats(total_st, folds[f_idx])
                w_f, b_f = _solve_ridge_path_from_stats(f_tr, [best_lam], device=device)
                wf_t = torch.from_numpy(w_f[0].astype(np.float32)).to(device=device)
                xb = torch.from_numpy((x_tr_f32[fm] - shift_mu[None, :])).to(device=device)
                oof_tr[fm] = (xb @ wf_t).cpu().numpy().astype(np.float64) + float(b_f[0])
            preds_eval[train_mask] = oof_tr
            cv_mse_map = fit_res["cv_mse_by_lambda"]
        else:
            x_all = x_buf.astype(np.float64, copy=False)
            x_tr = x_all[train_mask]
            fit = fit_ridge_svd_cv(
                x_tr,
                y_train,
                g_train,
                lambdas=lambdas,
                n_cv_folds=n_cv_folds,
                device=device,
                return_oof_preds=True,
            )
            preds_all = x_all @ fit["weights"] + float(fit["intercept"])
            if fit["oof_predictions"] is not None:
                preds_eval = preds_all.copy()
                preds_eval[train_mask] = fit["oof_predictions"]
            else:
                preds_eval = preds_all
            best_lam = float(fit["best_lambda"])
            cv_mse_map = fit["cv_mse_by_lambda"]

        strata_metrics = _summarize_megascale_predictions(
            preds_eval, targets_all, wt_idx_all, wt_strata
        )
        per_layer.append(
            {
                "layer_idx": int(tap),
                "layer_name": layer_names[tap],
                "rel_depth": float(tap / max(1, n_taps - 1)),
                "best_lambda": best_lam,
                "cv_mse_by_lambda": cv_mse_map,
                "by_stratum": strata_metrics,
            }
        )

    return {
        "probe": "megascale_ddg_ridge",
        "objective": obj,
        "feature_dim": int(feat_dim),
        "train_stratum": str(train_stratum),
        "n_train_domains": len(train_wt_ids),
        "n_train_variants": int(train_mask.sum()),
        "n_total_domains": len(selected_domains),
        "n_total_variants": int(total_variants),
        "max_variants_per_wt": max_variants_per_wt,
        "lambdas": [float(lam) for lam in lambdas],
        "zeroshot": zeroshot_block,
        "layers": per_layer,
        "extract_wall_sec": float(t_extract),
        "probe_fit_wall_sec": float(time.perf_counter() - t_fit0),
        "total_wall_sec": float(time.perf_counter() - t0),
    }


# --------------------------------------------------------------------------- #
# Shared Domain-Disjoint 5-Fold Assignment for 3D Unseen-Fold Domains
# --------------------------------------------------------------------------- #


def _assign_domain_folds(
    contact_domains: Sequence[Mapping[str, Any]],
    n_folds: int = 5,
) -> list[int]:
    """Deterministic balanced domain-disjoint fold assignment (sorted by length then domain_id)."""
    n_dom = len(contact_domains)
    eff_folds = min(max(1, int(n_folds)), max(1, n_dom))
    order = sorted(
        range(n_dom),
        key=lambda i: (-len(str(contact_domains[i]["sequence"])), str(contact_domains[i].get("domain_id", i))),
    )
    fold_of = [0] * n_dom
    for rank, dom_idx in enumerate(order):
        # Snake ordering balances both domain count and total pair count across folds
        block = rank // eff_folds
        pos = rank % eff_folds
        fold_of[dom_idx] = pos if (block % 2 == 0) else (eff_folds - 1 - pos)
    return fold_of


# --------------------------------------------------------------------------- #
# 2. Probe 2 — Unseen-Fold 3D Contact Pairwise Probe
# --------------------------------------------------------------------------- #


def _build_pair_features_gpu_or_np(
    h_dom: torch.Tensor,
    ii: np.ndarray,
    jj: np.ndarray,
) -> torch.Tensor:
    """Build symmetric pairwise features ``[h_i * h_j, |h_i - h_j|]`` in float32 on h_dom's device."""
    ii_t = torch.from_numpy(ii).to(device=h_dom.device, dtype=torch.long)
    jj_t = torch.from_numpy(jj).to(device=h_dom.device, dtype=torch.long)
    hi = h_dom[ii_t]
    hj = h_dom[jj_t]
    return torch.cat((hi * hj, torch.abs(hi - hj)), dim=-1)


@torch.no_grad()
def evaluate_contact_layerwise_probe(
    model: VariableLengthProteinTransformer,
    contact_domains: Sequence[Mapping[str, Any]],
    *,
    objective: str | None = None,
    preextracted_hidden: list[list[np.ndarray]] | None = None,
    batch_size: int = 64,
    zeroshot_batch_size: int = 256,
    precision: str = "bf16",
    n_folds: int = 5,
    lambdas: Sequence[float] = DEFAULT_RIDGE_LAMBDAS,
    min_sep: int = 6,
    contact_cutoff: float = CONTACT_DISTANCE_ANGSTROM,
    compute_zeroshot: bool = True,
    precomputed_zeroshot: Mapping[str, Any] | None = None,
    pair_chunk_size: int = 16_384,
) -> dict[str, Any]:
    """Run Probe 2 (Unseen-Fold 3D Contact Pairwise 5-fold domain-disjoint Ridge) across all layers."""
    t0 = time.perf_counter()
    if not contact_domains:
        raise ValueError("contact_domains must be non-empty")

    obj = canonical_objective(objective if objective is not None else model.objective)
    device = next(model.parameters()).device
    n_taps = len(model.blocks) + 2
    layer_names = get_layer_names(len(model.blocks))
    d_model = int(model.spec.d_model)
    feat_dim = 2 * d_model

    sequences = [str(d["sequence"]) for d in contact_domains]
    if preextracted_hidden is None:
        hidden_by_dom = extract_layerwise_hidden_states(
            model,
            sequences,
            objective=obj,
            batch_size=batch_size,
            precision=precision,
            output_dtype=np.float32,
        )
    else:
        hidden_by_dom = preextracted_hidden
    t_extract = time.perf_counter() - t0

    # Precompute valid upper-triangle pairs (|i-j| >= min_sep) and binary contact labels per domain
    dom_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    total_pairs = 0
    total_contacts = 0
    for dom in contact_domains:
        L = len(str(dom["sequence"]))
        cb_coords = np.asarray(dom["cb_coords"], dtype=np.float64)
        cb_mask = np.asarray(dom["cb_mask"], dtype=np.bool_)
        if L <= min_sep:
            dom_pairs.append((np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)))
            continue
        ii, jj = np.triu_indices(L, k=min_sep)
        valid_pair = cb_mask[ii] & cb_mask[jj]
        ii = ii[valid_pair]
        jj = jj[valid_pair]
        dists = np.linalg.norm(cb_coords[ii] - cb_coords[jj], axis=-1)
        labels = (dists < float(contact_cutoff)).astype(np.float64)
        dom_pairs.append((ii, jj, labels))
        total_pairs += int(labels.size)
        total_contacts += int(labels.sum())

    fold_of = _assign_domain_folds(contact_domains, n_folds=n_folds)
    eff_folds = max(fold_of) + 1

    t_fit0 = time.perf_counter()
    per_layer: list[dict[str, Any]] = []
    use_gpu_acc = device.type == "cuda"

    for tap in range(n_taps):
        # Accumulate exact per-fold sufficient statistics streaming over domains (zero large RAM copy)
        if use_gpu_acc:
            fold_n = [0] * eff_folds
            fold_sx = [torch.zeros(feat_dim, dtype=torch.float64, device=device) for _ in range(eff_folds)]
            fold_g = [torch.zeros((feat_dim, feat_dim), dtype=torch.float64, device=device) for _ in range(eff_folds)]
            fold_c = [torch.zeros(feat_dim, dtype=torch.float64, device=device) for _ in range(eff_folds)]
            fold_sy = [0.0] * eff_folds
            fold_sy2 = [0.0] * eff_folds
        else:
            folds_cpu = [FoldSufficientStats.empty(feat_dim) for _ in range(eff_folds)]

        for dom_i, (ii, jj, labels) in enumerate(dom_pairs):
            n_p = int(labels.size)
            if n_p == 0:
                continue
            f_idx = fold_of[dom_i]
            h_t = torch.from_numpy(hidden_by_dom[dom_i][tap]).to(
                device=device if use_gpu_acc else torch.device("cpu"),
                dtype=torch.float32,
            )
            for s in range(0, n_p, max(1, int(pair_chunk_size))):
                e = min(n_p, s + int(pair_chunk_size))
                phi = _build_pair_features_gpu_or_np(h_t, ii[s:e], jj[s:e])
                if use_gpu_acc:
                    # Fast FP32 GPU GEMM accumulated in FP64 (avoids 1/64th FP64 bottleneck on L20)
                    y32 = torch.from_numpy(labels[s:e].astype(np.float32, copy=False)).to(device=device)
                    fold_n[f_idx] += int(e - s)
                    fold_sx[f_idx] += phi.sum(dim=0, dtype=torch.float64)
                    fold_g[f_idx] += (phi.T @ phi).to(dtype=torch.float64)
                    fold_c[f_idx] += (phi.T @ y32).to(dtype=torch.float64)
                    fold_sy[f_idx] += float(y32.sum(dtype=torch.float64).item())
                    fold_sy2[f_idx] += float((y32.to(dtype=torch.float64) ** 2).sum().item())
                else:
                    folds_cpu[f_idx].add_batch(phi.numpy(), labels[s:e])

        if use_gpu_acc:
            folds_stats = [
                FoldSufficientStats(
                    n=fold_n[f],
                    sum_x=fold_sx[f].cpu().numpy(),
                    gram_x=fold_g[f].cpu().numpy(),
                    sum_xy=fold_c[f].cpu().numpy(),
                    sum_y=fold_sy[f],
                    sum_y2=fold_sy2[f],
                )
                for f in range(eff_folds)
            ]
        else:
            folds_stats = folds_cpu

        # Outer 5-fold domain-disjoint CV with exact inner CV over remaining 4 folds
        fold_models: list[tuple[np.ndarray, float, float]] = []
        for f_out in range(eff_folds):
            train_folds = [folds_stats[f] for f in range(eff_folds) if f != f_out and folds_stats[f].n > 0]
            if not train_folds:
                train_folds = [folds_stats[f_out]]
            fit = fit_ridge_from_stats_cv(train_folds, lambdas=lambdas, device=device)
            fold_models.append((fit["weights"], float(fit["intercept"]), float(fit["best_lambda"])))

        # Score each held-out domain using its out-of-fold linear classifier
        sep6_rows: list[dict[str, float]] = []
        sep24_rows: list[dict[str, float]] = []
        for dom_i, (ii, jj, labels) in enumerate(dom_pairs):
            if labels.size == 0:
                continue
            dom = contact_domains[dom_i]
            L = len(str(dom["sequence"]))
            w_f, b_f, _ = fold_models[fold_of[dom_i]]
            w_t = torch.from_numpy(w_f).to(
                device=device if use_gpu_acc else torch.device("cpu"),
                dtype=torch.float32,
            )
            h_t = torch.from_numpy(hidden_by_dom[dom_i][tap]).to(
                device=device if use_gpu_acc else torch.device("cpu"),
                dtype=torch.float32,
            )
            scores_flat = np.empty(int(labels.size), dtype=np.float64)
            for s in range(0, int(labels.size), max(1, int(pair_chunk_size))):
                e = min(int(labels.size), s + int(pair_chunk_size))
                phi = _build_pair_features_gpu_or_np(h_t, ii[s:e], jj[s:e])
                sc = (phi @ w_t + float(b_f)).cpu().numpy()
                scores_flat[s:e] = sc

            score_mat = np.zeros((L, L), dtype=np.float64)
            score_mat[ii, jj] = scores_flat
            score_mat[jj, ii] = scores_flat

            cb_coords = np.asarray(dom["cb_coords"], dtype=np.float32)
            cb_mask = np.asarray(dom["cb_mask"], dtype=np.bool_)
            m6 = compute_contact_metrics_for_domain(
                score_mat, cb_coords, cb_mask, min_sep=6, contact_cutoff=contact_cutoff
            )
            if m6 is not None:
                sep6_rows.append(m6)
            m24 = compute_contact_metrics_for_domain(
                score_mat, cb_coords, cb_mask, min_sep=24, contact_cutoff=contact_cutoff
            )
            if m24 is not None:
                sep24_rows.append(m24)

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
                "p_at_L": _safe_mean([r["p_at_L"] for r in rows]),
                "p_at_L_over_2": _safe_mean([r["p_at_L_over_2"] for r in rows]),
                "p_at_L_over_5": _safe_mean([r["p_at_L_over_5"] for r in rows]),
                "auroc": _safe_mean([r["auroc"] for r in rows]),
            }

        per_layer.append(
            {
                "layer_idx": int(tap),
                "layer_name": layer_names[tap],
                "rel_depth": float(tap / max(1, n_taps - 1)),
                "selected_lambdas_by_fold": [float(m[2]) for m in fold_models],
                "sep_ge_6": _agg(sep6_rows),
                "long_range_sep_ge_24": _agg(sep24_rows),
            }
        )

    t_fit = time.perf_counter() - t_fit0

    zeroshot_contacts: dict[str, Any] | None = None
    if precomputed_zeroshot is not None:
        zeroshot_contacts = dict(precomputed_zeroshot)
    elif compute_zeroshot:
        zeroshot_contacts = evaluate_unseen_fold_contacts(
            model,
            contact_domains,
            objective=obj,
            batch_size=zeroshot_batch_size,
            precision=precision,
        )

    return {
        "probe": "unseen_fold_contact_pairwise_ridge",
        "objective": obj,
        "feature_dim": int(feat_dim),
        "n_domains": len(contact_domains),
        "n_folds": int(eff_folds),
        "n_pairs_sep_ge_6": int(total_pairs),
        "n_contacts_sep_ge_6": int(total_contacts),
        "contact_rate_sep_ge_6": float(total_contacts / max(1, total_pairs)),
        "lambdas": [float(lam) for lam in lambdas],
        "zeroshot_categorical_jacobian": zeroshot_contacts,
        "layers": per_layer,
        "extract_wall_sec": float(t_extract),
        "probe_fit_wall_sec": float(t_fit),
        "total_wall_sec": float(time.perf_counter() - t0),
    }


# --------------------------------------------------------------------------- #
# 3. Probe 3 — Unseen-Fold Per-Residue 3D Burial / Coordination Number Probe
# --------------------------------------------------------------------------- #


def compute_residue_burial_targets(
    cb_coords: np.ndarray,
    cb_mask: np.ndarray,
    *,
    min_sep: int = 6,
    contact_cutoff: float = CONTACT_DISTANCE_ANGSTROM,
) -> np.ndarray:
    """Compute per-residue C-beta contact count (coordination number) for ``|i-j| >= min_sep``."""
    coords = np.asarray(cb_coords, dtype=np.float64)
    mask = np.asarray(cb_mask, dtype=np.bool_)
    L = int(coords.shape[0])
    counts = np.zeros(L, dtype=np.float64)
    if L <= min_sep or int(mask.sum()) == 0:
        return counts
    safe_coords = np.where(mask[:, None], coords, 0.0)
    dmat = np.linalg.norm(safe_coords[:, None, :] - safe_coords[None, :, :], axis=-1)
    idx = np.arange(L)
    sep_ok = np.abs(idx[:, None] - idx[None, :]) >= int(min_sep)
    valid_pair = np.outer(mask, mask) & sep_ok
    contacts = (dmat < float(contact_cutoff)) & valid_pair
    return contacts.sum(axis=1).astype(np.float64)


@torch.no_grad()
def evaluate_burial_layerwise_probe(
    model: VariableLengthProteinTransformer,
    contact_domains: Sequence[Mapping[str, Any]],
    *,
    objective: str | None = None,
    preextracted_hidden: list[list[np.ndarray]] | None = None,
    batch_size: int = 64,
    precision: str = "bf16",
    n_folds: int = 5,
    lambdas: Sequence[float] = DEFAULT_RIDGE_LAMBDAS,
    min_sep: int = 6,
    contact_cutoff: float = CONTACT_DISTANCE_ANGSTROM,
) -> dict[str, Any]:
    """Run Probe 3 (Per-Residue 3D Burial / Coordination Number 5-fold domain-disjoint Ridge)."""
    t0 = time.perf_counter()
    if not contact_domains:
        raise ValueError("contact_domains must be non-empty")

    obj = canonical_objective(objective if objective is not None else model.objective)
    device = next(model.parameters()).device
    n_taps = len(model.blocks) + 2
    layer_names = get_layer_names(len(model.blocks))
    d_model = int(model.spec.d_model)

    sequences = [str(d["sequence"]) for d in contact_domains]
    if preextracted_hidden is None:
        hidden_by_dom = extract_layerwise_hidden_states(
            model,
            sequences,
            objective=obj,
            batch_size=batch_size,
            precision=precision,
            output_dtype=np.float32,
        )
    else:
        hidden_by_dom = preextracted_hidden
    t_extract = time.perf_counter() - t0

    fold_of = _assign_domain_folds(contact_domains, n_folds=n_folds)
    eff_folds = max(fold_of) + 1

    # Gather valid residue targets, domain IDs, and fold assignments
    y_parts: list[np.ndarray] = []
    dom_parts: list[np.ndarray] = []
    fold_parts: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []

    for dom_i, dom in enumerate(contact_domains):
        cb_coords = np.asarray(dom["cb_coords"], dtype=np.float64)
        cb_mask = np.asarray(dom["cb_mask"], dtype=np.bool_)
        counts = compute_residue_burial_targets(
            cb_coords, cb_mask, min_sep=min_sep, contact_cutoff=contact_cutoff
        )
        valid_masks.append(cb_mask)
        n_val = int(cb_mask.sum())
        if n_val == 0:
            continue
        y_parts.append(counts[cb_mask])
        dom_parts.append(np.full(n_val, dom_i, dtype=np.int64))
        fold_parts.append(np.full(n_val, fold_of[dom_i], dtype=np.int64))

    y_all = np.concatenate(y_parts)
    dom_all = np.concatenate(dom_parts)
    fold_all = np.concatenate(fold_parts)
    n_res = int(y_all.size)

    t_fit0 = time.perf_counter()
    per_layer: list[dict[str, Any]] = []

    for tap in range(n_taps):
        x_all_f32 = np.concatenate(
            [
                hidden_by_dom[dom_i][tap][valid_masks[dom_i]]
                for dom_i in range(len(contact_domains))
                if int(valid_masks[dom_i].sum()) > 0
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        oof_preds = np.empty(n_res, dtype=np.float64)
        sel_lambdas: list[float] = []

        if eff_folds >= 2 and n_res > d_model:
            # Compute fold sufficient statistics ONCE per layer (O(1) pass over data)
            shift_mu = x_all_f32.mean(axis=0, dtype=np.float64).astype(np.float32)
            x_shift = x_all_f32 - shift_mu[None, :]
            folds_stats = [
                FoldSufficientStats.from_array_fast(
                    x_shift[fold_all == f], y_all[fold_all == f], device=device
                )
                for f in range(eff_folds)
            ]
            for f_out in range(eff_folds):
                val_m = fold_all == f_out
                if int(val_m.sum()) == 0:
                    continue
                train_folds = [
                    folds_stats[f]
                    for f in range(eff_folds)
                    if f != f_out and folds_stats[f].n > 0
                ]
                if not train_folds:
                    train_folds = [folds_stats[f_out]]
                fit = fit_ridge_from_stats_cv(train_folds, lambdas=lambdas, device=device)
                oof_preds[val_m] = x_shift[val_m].astype(np.float64) @ fit["weights"] + float(
                    fit["intercept"]
                )
                sel_lambdas.append(float(fit["best_lambda"]))
        else:
            x_all = x_all_f32.astype(np.float64, copy=False)
            for f_out in range(eff_folds):
                val_m = fold_all == f_out
                tr_m = ~val_m
                if int(val_m.sum()) == 0:
                    continue
                if int(tr_m.sum()) == 0:
                    tr_m = val_m
                fit = fit_ridge_svd_cv(
                    x_all[tr_m],
                    y_all[tr_m],
                    fold_all[tr_m],
                    lambdas=lambdas,
                    n_cv_folds=max(1, eff_folds - 1),
                    device=device,
                    return_oof_preds=False,
                )
                oof_preds[val_m] = x_all[val_m] @ fit["weights"] + float(fit["intercept"])
                sel_lambdas.append(float(fit["best_lambda"]))

        dom_r2s: list[float] = []
        dom_rhos: list[float] = []
        dom_pearsons: list[float] = []
        for dom_i in np.unique(dom_all):
            m = dom_all == dom_i
            if int(m.sum()) < 3:
                continue
            dom_r2s.append(_safe_r2(y_all[m], oof_preds[m]))
            dom_rhos.append(_safe_spearman(oof_preds[m], y_all[m]))
            dom_pearsons.append(_safe_pearson(oof_preds[m], y_all[m]))

        rmse = float(np.sqrt(np.mean((y_all - oof_preds) ** 2)))
        per_layer.append(
            {
                "layer_idx": int(tap),
                "layer_name": layer_names[tap],
                "rel_depth": float(tap / max(1, n_taps - 1)),
                "selected_lambdas_by_fold": sel_lambdas,
                "pooled_r2": _safe_r2(y_all, oof_preds),
                "pooled_spearman": _safe_spearman(oof_preds, y_all),
                "pooled_pearson": _safe_pearson(oof_preds, y_all),
                "pooled_rmse": rmse,
                "domain_mean_r2": _safe_mean(dom_r2s),
                "domain_mean_spearman": _safe_mean(dom_rhos),
                "domain_mean_pearson": _safe_mean(dom_pearsons),
            }
        )

    return {
        "probe": "unseen_fold_burial_coordination_ridge",
        "objective": obj,
        "feature_dim": int(d_model),
        "n_domains": len(contact_domains),
        "n_folds": int(eff_folds),
        "n_residues": int(n_res),
        "target_mean_contacts": float(np.mean(y_all)),
        "target_std_contacts": float(np.std(y_all)),
        "lambdas": [float(lam) for lam in lambdas],
        "layers": per_layer,
        "extract_wall_sec": float(t_extract),
        "probe_fit_wall_sec": float(time.perf_counter() - t_fit0),
        "total_wall_sec": float(time.perf_counter() - t0),
    }
