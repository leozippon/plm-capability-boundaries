"""Shared statistical, projection, and Ridge readout helpers for general-PLM scripts.

Reuses the repository's frozen Gaussian projection convention (seed 20260923,
4 blocks x 256 dims = 1024 dims) and nested held-group Ridge protocol
(``family_folds``, ``ridge_predict``, ``row_weights``) without modifying any
frozen panel file.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

from src.capability.interactions.pairwise_epistasis import (
    FEATURE_BLOCKS,
    PROJECTION_DIM,
    PROJECTION_SEED,
    SPLIT_SEEDS,
    projection_matrices,
)
from src.capability.readouts.readout_analysis import (
    family_folds,
    ridge_predict,
    row_weights,
)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write valid JSON atomically (refusing NaN/Infinity)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def project_four_blocks(
    raw_blocks: np.ndarray,
    *,
    dim: int = PROJECTION_DIM,
    seed: int = PROJECTION_SEED,
) -> np.ndarray:
    """Project ``(N, 4, hidden_dim)`` hidden summaries to ``(N, 4 * dim)`` float32.

    Uses the exact same Gaussian projection seeds (20260923..20260926) and
    scaling ``1/sqrt(dim)`` declared in ``pairwise_epistasis.py``.
    """
    arr = np.asarray(raw_blocks, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != len(FEATURE_BLOCKS):
        raise ValueError(
            f"Expected raw_blocks shape (N, {len(FEATURE_BLOCKS)}, H), got {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite values in raw hidden-state blocks")
    mats = projection_matrices(arr.shape[2], dim=dim, seed=seed)
    projected = [arr[:, idx, :] @ mat for idx, mat in enumerate(mats)]
    out = np.concatenate(projected, axis=1).astype(np.float32, copy=False)
    if not np.isfinite(out).all():
        raise ValueError("Non-finite values after Gaussian projection")
    return out


def safe_spearman(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float | None:
    """Finite Spearman rank correlation, or None if degenerate."""
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    if xv.size < 3 or xv.shape != yv.shape:
        return None
    if not (np.isfinite(xv).all() and np.isfinite(yv).all()):
        return None
    if np.std(xv) == 0.0 or np.std(yv) == 0.0:
        return None
    val = float(spearmanr(xv, yv).statistic)
    return val if np.isfinite(val) else None


def standardized_rank(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance mid-rank transform within one assay/group."""
    v = np.asarray(values, dtype=np.float64)
    r = rankdata(v, method="average")
    r = r - r.mean()
    sd = float(np.sqrt(np.mean(r * r)))
    return r / sd if sd > 0.0 else np.zeros_like(r)


def cluster_bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    clusters: Sequence[Any],
    *,
    draws: int = 2000,
    seed: int = 20261120,
) -> dict[str, Any]:
    """Equal-cluster-weight mean and 95% percentile CI via cluster bootstrap."""
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0 or len(clusters) != vals.size:
        return {"mean": None, "ci_low": None, "ci_high": None, "se": None, "n_clusters": 0, "n_items": 0}
    grouped: dict[Any, list[float]] = {}
    for v, c in zip(vals, clusters):
        if np.isfinite(v):
            grouped.setdefault(c, []).append(float(v))
    if not grouped:
        return {"mean": None, "ci_low": None, "ci_high": None, "se": None, "n_clusters": 0, "n_items": 0}
    keys = sorted(grouped.keys(), key=str)
    cluster_means = np.asarray([float(np.mean(grouped[k])) for k in keys], dtype=np.float64)
    point = float(cluster_means.mean())
    n_c = cluster_means.size
    if n_c < 2:
        return {
            "mean": point,
            "ci_low": point,
            "ci_high": point,
            "se": 0.0,
            "n_clusters": int(n_c),
            "n_items": int(sum(len(v) for v in grouped.values())),
        }
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n_c, size=(int(draws), n_c))
    boot = cluster_means[idx].mean(axis=1)
    return {
        "mean": point,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "se": float(boot.std(ddof=1)),
        "n_clusters": int(n_c),
        "n_items": int(sum(len(v) for v in grouped.values())),
    }


def nested_group_ridge_eval(
    target: np.ndarray,
    groups: np.ndarray,
    subunits: np.ndarray,
    feature_blocks: Mapping[str, np.ndarray],
    designs: Mapping[str, Sequence[str]],
    *,
    seeds: Sequence[int] = SPLIT_SEEDS,
    n_outer: int = 5,
    n_inner: int = 4,
    alphas: Sequence[float] = tuple(np.logspace(-2, 6, 17)),
) -> dict[str, Any]:
    """Nested held-group Ridge evaluation matching the repository's bidirectional control hierarchy.

    Returns per-seed and seed-averaged group-weighted MSE and within-group
    Spearman for every named design, plus paired increments between designs.
    """
    y = np.asarray(target, dtype=np.float64)
    g = np.asarray(groups)
    s = np.asarray(subunits)
    n = y.size
    if g.size != n or s.size != n:
        raise ValueError("target, groups, and subunits must share length")

    unique_groups = sorted(set(g.tolist()), key=str)
    n_groups = len(unique_groups)
    outer_k = min(int(n_outer), n_groups)
    if outer_k < 2:
        raise ValueError(f"Need at least 2 groups for held-group CV; got {n_groups}")

    weights = row_weights(s, g)
    alpha_grid = tuple(float(a) for a in alphas)

    def _concat(block_names: Sequence[str]) -> np.ndarray:
        parts = []
        for name in block_names:
            arr = np.asarray(feature_blocks[name], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[:, None]
            parts.append(arr)
        return np.concatenate(parts, axis=1) if parts else np.zeros((n, 0), dtype=np.float64)

    seed_results: dict[str, Any] = {}
    group_mse_by_design: dict[str, list[np.ndarray]] = {d: [] for d in designs}
    group_rho_by_design: dict[str, list[np.ndarray]] = {d: [] for d in designs}

    for seed in seeds:
        outer_folds = family_folds(g, outer_k, int(seed))
        preds_by_design: dict[str, np.ndarray] = {
            d: np.full(n, np.nan, dtype=np.float64) for d in designs
        }
        chosen_alphas: dict[str, list[float]] = {d: [] for d in designs}

        for d_name, block_names in designs.items():
            X = _concat(block_names)
            for fold_idx, held_groups in enumerate(outer_folds):
                test_mask = np.isin(g, held_groups)
                train_mask = ~test_mask
                if X.shape[1] == 0:
                    w_tr = row_weights(s[train_mask], g[train_mask])
                    pred_val = float(np.sum(w_tr * y[train_mask]) / np.sum(w_tr))
                    preds_by_design[d_name][test_mask] = pred_val
                    chosen_alphas[d_name].append(0.0)
                    continue

                train_idx = np.flatnonzero(train_mask)
                test_idx = np.flatnonzero(test_mask)
                train_groups = g[train_idx]
                inner_k = min(int(n_inner), len(set(train_groups.tolist())))
                if inner_k < 2:
                    best_alpha = float(alpha_grid[len(alpha_grid) // 2])
                else:
                    losses = np.zeros(len(alpha_grid), dtype=np.float64)
                    for val_groups in family_folds(train_groups, inner_k, int(seed) + 100 + fold_idx):
                        is_val = np.isin(train_groups, val_groups)
                        fit_idx = train_idx[~is_val]
                        val_idx = train_idx[is_val]
                        pred_grid = ridge_predict(
                            X[fit_idx],
                            y[fit_idx],
                            row_weights(s[fit_idx], g[fit_idx]),
                            X[val_idx],
                            alphas=alpha_grid,
                        )
                        w_val = row_weights(s[val_idx], g[val_idx])
                        fold_loss = (w_val[:, None] * (pred_grid - y[val_idx, None]) ** 2).sum(axis=0)
                        losses += len(val_groups) * fold_loss
                    losses /= max(1, len(set(train_groups.tolist())))
                    best_idx = len(losses) - 1 - int(np.argmin(losses[::-1]))
                    best_alpha = float(alpha_grid[best_idx])

                chosen_alphas[d_name].append(float(best_alpha))
                preds_by_design[d_name][test_idx] = ridge_predict(
                    X[train_idx],
                    y[train_idx],
                    row_weights(s[train_idx], g[train_idx]),
                    X[test_idx],
                    alphas=[best_alpha],
                )[:, 0]

        per_seed_designs: dict[str, Any] = {}
        for d_name, pred in preds_by_design.items():
            g_mses = []
            g_rhos = []
            for grp in unique_groups:
                mask = g == grp
                w_g = weights[mask]
                mse_g = float(np.sum(w_g * (y[mask] - pred[mask]) ** 2) / np.sum(w_g))
                g_mses.append(mse_g)
                rho = safe_spearman(pred[mask], y[mask])
                g_rhos.append(np.nan if rho is None else rho)
            g_mse_arr = np.asarray(g_mses, dtype=np.float64)
            g_rho_arr = np.asarray(g_rhos, dtype=np.float64)
            group_mse_by_design[d_name].append(g_mse_arr)
            group_rho_by_design[d_name].append(g_rho_arr)
            finite_rho = g_rho_arr[np.isfinite(g_rho_arr)]
            per_seed_designs[d_name] = {
                "mean_group_mse": float(g_mse_arr.mean()),
                "mean_group_spearman": float(finite_rho.mean()) if finite_rho.size else None,
                "alphas": chosen_alphas[d_name],
            }
        seed_results[str(seed)] = per_seed_designs

    # Average per-group errors across seeds for bootstrap summaries.
    summary: dict[str, Any] = {}
    mean_group_mse: dict[str, np.ndarray] = {}
    mean_group_rho: dict[str, np.ndarray] = {}
    for d_name in designs:
        mean_group_mse[d_name] = np.mean(np.stack(group_mse_by_design[d_name], axis=0), axis=0)
        rho_stack = np.stack(group_rho_by_design[d_name], axis=0)
        with np.errstate(invalid="ignore"):
            mean_group_rho[d_name] = np.nanmean(rho_stack, axis=0)
        summary[d_name] = {
            "group_mse": cluster_bootstrap_mean(mean_group_mse[d_name], unique_groups, seed=20261120),
            "group_spearman": cluster_bootstrap_mean(mean_group_rho[d_name], unique_groups, seed=20261121),
        }

    return {
        "n_rows": int(n),
        "n_groups": int(n_groups),
        "seeds": [int(sd) for sd in seeds],
        "per_seed": seed_results,
        "summary": summary,
        "mean_group_mse_vector": {k: v.tolist() for k, v in mean_group_mse.items()},
        "mean_group_spearman_vector": {
            k: [None if not np.isfinite(x) else float(x) for x in v]
            for k, v in mean_group_rho.items()
        },
        "group_order": [str(x) for x in unique_groups],
    }
