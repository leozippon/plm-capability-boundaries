"""Unified 62-model R3 epistasis and contacts yardstick.

Builds ``r3_epistasis_and_contacts_unified_62.json`` combining:

1. **62-Model Double-Mutant Epistasis** (MegaScale indel-excluded complete-case
   Potts support: 61 groups, 7,805 cycles, exact ``G+T+prof+POTTS`` nested-CV
   protocol):
   - ``M_cycle`` (62/62 single-sequence arms): combines the 33 pre-registered
     arms (reusing cached ``cycle_squared_error`` cells verified to ``< 1e-12``,
     plus exact per-group Spearman and zero-shot ``ZS_raw`` / ``ZS_res``) with
     all 29 expanded single-sequence arms (28 GP ``.npz`` + ``esm3-1.4b``).
   - ``R_cycle`` and ``MR_cycle`` (62/62 arms — 100% full single-sequence panel):
     loads raw per-state ``projected`` (1024-d) from
     ``results/pairwise_epistasis_20260924/extraction/<arm>/`` for all 33
     pre-registered arms, ``R_cycle`` from the 28 GP ``.npz`` archives, and
     ``r2_readout_archives/r3_cycles_R_esm3-1.4b.npz`` for ``esm3-1.4b``,
     fitting all 62 arms through the EXACT same cached-fold nested-CV Ridge
     protocol and running unified 62-model ``paired_bands`` (``ΔMSE`` and ``Δρ``
     over ``G+T+prof+POTTS``).
2. **Unified R3 Contacts** (explicitly tagged by ``readout_mechanism``):
   - ``perturbation_propagation_e01``: 22-arm E01 perturbation-propagation
     discrimination & bidirectional panel (``contact_discrimination_20261120`` +
     ``contact_bidirectional_20261120``: All / Deep / Shallow, Primary &
     Full-Alphabet, ``couplings_given_geom_prof_model``, raw & clipped Brier).
   - ``attention_apc_map``: 16-arm attention/APC contact-map panel from
     ``general_plm_summary.json`` and per-arm ``r3_pairwise_and_contacts_*.json``.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.capability.interactions import fit_pairwise_epistasis as fit_pw
from src.capability.core.io import sha256_file
from src.capability.extensions import bidirectional as bidir
from src.capability.general_plm.eval_utils import atomic_write_json
from src.capability.general_plm.r3_cycle_eval import (
    INTERACTION_RUNGS,
    PairwiseCycleContext,
    load_pairwise_cycle_context,
)
from src.capability.interactions import pairwise_epistasis as pw_mod
from src.capability.readouts.readout_analysis import ALPHAS, family_folds
from src.capability.unified.roster import (
    ARM_METADATA,
    PREREGISTERED_33_ARMS,
    REPO_ROOT,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_OUTPUT_DIR,
    resolve_gp_r3_cycles,
)

SCHEMA = "r3_epistasis_and_contacts_unified_62_v1"

R3_RUNGS: tuple[str, ...] = (
    "nuisance",
    "nuisance+profile",
    "nuisance+profile+couplings",
)
UNIFIED_62_WITH_R_ARMS: tuple[str, ...] = UNIFIED_62_SINGLE_SEQUENCE_ARMS


# --------------------------------------------------------------------------- #
# Self-contained statistical banding & exact Cholesky Gram Ridge helpers
# --------------------------------------------------------------------------- #


def paired_bands_complete(
    matrix: np.ndarray,
    *,
    draws: int = bidir.DRAWS,
    seed: int = bidir.SEED,
    confidence: float = bidir.CONFIDENCE,
) -> dict[str, Any]:
    """Paired group bootstrap with native simultaneous, marginal 95%, and studentized maxT bands."""
    values = np.asarray(matrix, dtype=np.float64)
    native = bidir.paired_bands(values, draws=draws, seed=seed, confidence=confidence)
    resampled = bidir.resample_means(values, draws=draws, seed=seed)
    points = values.mean(axis=0)
    alpha = 0.5 * (1.0 - confidence)
    marg_low = np.quantile(resampled, alpha, axis=0)
    marg_high = np.quantile(resampled, 1.0 - alpha, axis=0)
    marginal_interval = np.column_stack([marg_low, marg_high]).tolist()

    n_groups = values.shape[0]
    se = values.std(axis=0, ddof=1) / math.sqrt(n_groups)
    positive_se = se > 1e-15
    if positive_se.any():
        t_dev = np.abs(resampled[:, positive_se] - points[positive_se]) / se[positive_se]
        t_crit = float(np.quantile(np.max(t_dev, axis=1), confidence))
    else:
        t_crit = 0.0
    stud_low = points - t_crit * se
    stud_high = points + t_crit * se
    studentized_interval = np.column_stack([stud_low, stud_high]).tolist()

    return {
        **native,
        "verdict": bidir.verdict(native["interval"]),
        "marginal_interval": marginal_interval,
        "marginal_verdict": bidir.verdict(marginal_interval),
        "studentized_critical_t": t_crit,
        "studentized_se": se.tolist(),
        "studentized_interval": studentized_interval,
        "studentized_verdict": bidir.verdict(studentized_interval),
    }


def build_zero_shot_section(
    per_group_arms: dict[str, np.ndarray],
    per_group_explicit: dict[str, np.ndarray],
    arm_order: tuple[str, ...],
    group_order: list[str],
    *,
    draws: int = bidir.DRAWS,
    seed: int = bidir.SEED,
) -> dict[str, Any]:
    """Zero-shot summary across all arms + explicit baselines with 10k-draw paired bands."""
    explicit_names = [k for k in ("profile", "potts", "gemme") if k in per_group_explicit]
    all_keys = [*arm_order, *explicit_names]
    mat = np.column_stack(
        [
            np.asarray(
                per_group_arms[k] if k in per_group_arms else per_group_explicit[k],
                dtype=np.float64,
            )
            for k in all_keys
        ]
    )
    if mat.shape[0] != len(group_order):
        raise ValueError(f"Zero-shot matrix rows {mat.shape[0]} != {len(group_order)} groups")
    arm_mat = mat[:, : len(arm_order)]
    arm_bands = paired_bands_complete(arm_mat, draws=draws, seed=seed)
    exp_bands = (
        paired_bands_complete(mat[:, len(arm_order) :], draws=draws, seed=seed)
        if explicit_names
        else None
    )

    arms_out: dict[str, Any] = {}
    for idx, arm_id in enumerate(arm_order):
        meta = ARM_METADATA[arm_id]
        arms_out[arm_id] = {
            "arm_id": arm_id,
            "display_name": meta.display_name,
            "paradigm": meta.paradigm,
            "family": meta.family,
            "params_m": meta.params_m,
            "origin_cohort": meta.origin_cohort,
            "point": float(arm_bands["point"][idx]),
            "marginal_interval_95": arm_bands["marginal_interval"][idx],
            "simultaneous_interval_95": arm_bands["interval"][idx],
            "studentized_interval_95": arm_bands["studentized_interval"][idx],
            "verdict": arm_bands["verdict"][idx],
            "marginal_verdict": arm_bands["marginal_verdict"][idx],
            "studentized_verdict": arm_bands["studentized_verdict"][idx],
        }

    explicit_out: dict[str, Any] = {}
    if exp_bands is not None:
        for idx, name in enumerate(explicit_names):
            explicit_out[name] = {
                "point": float(exp_bands["point"][idx]),
                "marginal_interval_95": exp_bands["marginal_interval"][idx],
                "simultaneous_interval_95": exp_bands["interval"][idx],
                "verdict": exp_bands["verdict"][idx],
                "marginal_verdict": exp_bands["marginal_verdict"][idx],
            }

    contrasts_out: dict[str, Any] = {}
    for exp_name in explicit_names:
        exp_vec = np.asarray(per_group_explicit[exp_name], dtype=np.float64)[:, None]
        diff_mat = arm_mat - exp_vec
        cb = paired_bands_complete(diff_mat, draws=draws, seed=seed)
        per_arm_c: dict[str, Any] = {}
        for idx, arm_id in enumerate(arm_order):
            per_arm_c[arm_id] = {
                "point": float(cb["point"][idx]),
                "marginal_interval_95": cb["marginal_interval"][idx],
                "simultaneous_interval_95": cb["interval"][idx],
                "studentized_interval_95": cb["studentized_interval"][idx],
                "verdict": cb["verdict"][idx],
                "marginal_verdict": cb["marginal_verdict"][idx],
                "studentized_verdict": cb["studentized_verdict"][idx],
            }
        contrasts_out[f"model_minus_{exp_name}"] = {
            "family_size": len(arm_order),
            "groups": len(group_order),
            "draws": draws,
            "seed": seed,
            "native_critical_value": float(cb["critical_value"]),
            "studentized_critical_t": float(cb["studentized_critical_t"]),
            "states_native": bidir.counts(cb["verdict"]),
            "states_marginal": bidir.counts(cb["marginal_verdict"]),
            "states_studentized": bidir.counts(cb["studentized_verdict"]),
            "arms": per_arm_c,
        }

    return {
        "n_groups": len(group_order),
        "n_arms": len(arm_order),
        "draws": draws,
        "seed": seed,
        "arm_family_native_critical_value": float(arm_bands["critical_value"]),
        "arm_family_studentized_critical_t": float(arm_bands["studentized_critical_t"]),
        "explicit_baselines": explicit_out,
        "arms": arms_out,
        "paired_contrasts_vs_explicit": contrasts_out,
    }


def band_bidirectional_family(
    per_arm_rung_values: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    arm_order: tuple[str, ...],
    rungs: tuple[str, ...],
    to_loss: float,
    draws: int = bidir.DRAWS,
    seed: int = bidir.SEED,
) -> dict[str, Any]:
    """Band a (rungs x arms x 2 directions) family with exact paired_bands + marginal + studentized."""
    columns_meta: list[dict[str, str]] = []
    cols: list[np.ndarray] = []
    for rung in rungs:
        for arm_id in arm_order:
            cell = per_arm_rung_values[arm_id][rung]
            u_c = np.asarray(cell["U_C"], dtype=np.float64)
            m = np.asarray(cell["M"], dtype=np.float64)
            u_cm = np.asarray(cell["U_C_M"], dtype=np.float64)
            cols.append(to_loss * (u_c - u_cm))
            columns_meta.append(
                {"rung": rung, "arm": arm_id, "direction": "model_given_controls"}
            )
            cols.append(to_loss * (m - u_cm))
            columns_meta.append(
                {"rung": rung, "arm": arm_id, "direction": "controls_given_model"}
            )
    mat = np.column_stack(cols)
    bands = paired_bands_complete(mat, draws=draws, seed=seed)

    records: list[dict[str, Any]] = []
    for idx, meta in enumerate(columns_meta):
        cell = per_arm_rung_values[meta["arm"]][meta["rung"]]
        records.append(
            {
                **meta,
                "paradigm": ARM_METADATA[meta["arm"]].paradigm,
                "point": float(bands["point"][idx]),
                "interval": bands["interval"][idx],
                "verdict": bands["verdict"][idx],
                "marginal_interval": bands["marginal_interval"][idx],
                "marginal_verdict": bands["marginal_verdict"][idx],
                "studentized_interval": bands["studentized_interval"][idx],
                "studentized_verdict": bands["studentized_verdict"][idx],
                "control_mean": float(np.mean(cell["U_C"])),
                "model_mean": float(np.mean(cell["M"])),
                "joint_mean": float(np.mean(cell["U_C_M"])),
            }
        )

    by_rung_dir: dict[str, dict[str, Any]] = {}
    for rung in rungs:
        by_rung_dir[rung] = {}
        for direction in bidir.DIRECTIONS:
            sub = [r for r in records if r["rung"] == rung and r["direction"] == direction]
            pts = [r["point"] for r in sub]
            by_rung_dir[rung][direction] = {
                "states_native": bidir.counts([r["verdict"] for r in sub]),
                "states_marginal": bidir.counts([r["marginal_verdict"] for r in sub]),
                "states_studentized": bidir.counts([r["studentized_verdict"] for r in sub]),
                "max_point": float(max(pts)),
                "min_point": float(min(pts)),
                "max_arm": max(sub, key=lambda r: r["point"])["arm"],
                "max_simultaneous_upper": float(max(r["interval"][1] for r in sub)),
            }

    return {
        "family_size": int(mat.shape[1]),
        "groups": int(mat.shape[0]),
        "rungs": list(rungs),
        "n_arms": len(arm_order),
        "to_loss": float(to_loss),
        "draws": draws,
        "seed": seed,
        "native_critical_value": float(bands["critical_value"]),
        "studentized_critical_t": float(bands["studentized_critical_t"]),
        "states_native": bidir.counts(bands["verdict"]),
        "states_marginal": bidir.counts(bands["marginal_verdict"]),
        "states_studentized": bidir.counts(bands["studentized_verdict"]),
        "summary_by_rung_direction": by_rung_dir,
        "columns": records,
    }


def _select_alpha(losses: np.ndarray) -> int:
    """Largest penalty among the minima (frozen tie rule)."""
    return len(losses) - 1 - int(np.argmin(losses[::-1]))


@dataclass
class _GramState:
    """Precomputed weighted standardized base design matrix and Gram block on one split."""

    w_norm: torch.Tensor
    ym: torch.Tensor
    w_yc: torch.Tensor
    w_z_base: torch.Tensor
    cov_bb: torch.Tensor
    rhs_b: torch.Tensor
    z_eval_base: torch.Tensor
    base_pred_alphas: np.ndarray


def _solve_ridge_gram(
    cov: torch.Tensor,
    rhs: torch.Tensor,
    alphas: tuple[float, ...] = ALPHAS,
) -> torch.Tensor:
    """Solve ``(cov + alpha * I) beta = rhs`` for strictly-positive ``alphas`` via Cholesky."""
    n = cov.shape[0]
    cov_work = cov.clone()
    diag_idx = torch.arange(n, device=cov.device)
    diag0 = cov.diagonal().clone()
    rhs_col = rhs[:, None]
    cols = []
    for alpha in alphas:
        cov_work[diag_idx, diag_idx] = diag0 + float(alpha)
        chol, info = torch.linalg.cholesky_ex(cov_work)
        if int(info.item()) == 0:
            cols.append(torch.cholesky_solve(rhs_col, chol)[:, 0])
        else:
            vals, vecs = torch.linalg.eigh(cov)
            at = torch.tensor(alphas, dtype=torch.float64, device=cov.device)
            return vecs @ ((vecs.T @ rhs)[:, None] / (vals.clamp_min(0)[:, None] + at))
    return torch.stack(cols, dim=1)


def _build_gram_state(
    x_base_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    x_base_ev: np.ndarray,
    alphas: tuple[float, ...] = ALPHAS,
) -> _GramState:
    x = torch.as_tensor(x_base_tr, dtype=torch.float64)
    y = torch.as_tensor(y_tr, dtype=torch.float64)
    w = torch.as_tensor(w_tr, dtype=torch.float64)
    w_norm = w / w.sum()
    mean = (w_norm[:, None] * x).sum(0)
    xc = x - mean
    scale = (w_norm[:, None] * xc.square()).sum(0).sqrt()
    scale = torch.where(scale < 1e-12, torch.ones_like(scale), scale)
    z_base = xc / scale
    w_z_base = w_norm[:, None] * z_base
    cov_bb = z_base.T @ w_z_base
    ym = (w_norm * y).sum()
    w_yc = w_norm * (y - ym)
    rhs_b = z_base.T @ w_yc

    x_ev = torch.as_tensor(x_base_ev, dtype=torch.float64)
    z_eval_base = (x_ev - mean) / scale

    beta = _solve_ridge_gram(cov_bb, rhs_b, alphas)
    pred = (z_eval_base @ beta + ym).numpy()
    return _GramState(
        w_norm=w_norm,
        ym=ym,
        w_yc=w_yc,
        w_z_base=w_z_base,
        cov_bb=cov_bb,
        rhs_b=rhs_b,
        z_eval_base=z_eval_base,
        base_pred_alphas=pred,
    )


def _predict_augmented_from_gram(
    gs: _GramState,
    x_extra_tr: np.ndarray,
    x_extra_ev: np.ndarray,
    alphas: tuple[float, ...] = ALPHAS,
) -> np.ndarray:
    """Exact ``ridge_predict([x_base, x_extra], ...)`` reusing precomputed ``cov_bb``."""
    xe = torch.as_tensor(x_extra_tr, dtype=torch.float64)
    if xe.ndim == 1:
        xe = xe[:, None]
    xe_ev = torch.as_tensor(x_extra_ev, dtype=torch.float64)
    if xe_ev.ndim == 1:
        xe_ev = xe_ev[:, None]

    w_norm = gs.w_norm
    mean_e = (w_norm[:, None] * xe).sum(0)
    xec = xe - mean_e
    scale_e = (w_norm[:, None] * xec.square()).sum(0).sqrt()
    scale_e = torch.where(scale_e < 1e-12, torch.ones_like(scale_e), scale_e)
    z_e = xec / scale_e
    z_ev_e = (xe_ev - mean_e) / scale_e

    p = gs.cov_bb.shape[0]
    k = z_e.shape[1]
    cov_be = gs.w_z_base.T @ z_e
    cov_ee = z_e.T @ (w_norm[:, None] * z_e)

    cov = torch.empty((p + k, p + k), dtype=torch.float64)
    cov[:p, :p] = gs.cov_bb
    cov[:p, p:] = cov_be
    cov[p:, :p] = cov_be.T
    cov[p:, p:] = cov_ee

    rhs_e = z_e.T @ gs.w_yc
    rhs = torch.cat([gs.rhs_b, rhs_e], dim=0)

    beta = _solve_ridge_gram(cov, rhs, alphas)
    pred = gs.z_eval_base @ beta[:p] + z_ev_e @ beta[p:] + gs.ym
    return pred.numpy()


# --------------------------------------------------------------------------- #
# R3 cached-fold Gram builder and per-arm evaluator
# --------------------------------------------------------------------------- #


def _r3_build_rung_caches(
    ctx: PairwiseCycleContext,
    blocks_base: dict[str, np.ndarray],
    rungs: tuple[str, ...] = R3_RUNGS,
) -> tuple[dict[str, dict[int, list[dict[str, Any]]]], dict[int, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Precompute exact cached-fold Gram states and U_C metrics for one ``T`` block."""
    panel = ctx.panel
    groups = panel["group"]
    site_pairs = panel["site_pair"]
    weights = ctx.weights

    # Load cached fold targets once per seed.
    oof_targets: dict[int, np.ndarray] = {}
    fold_raw: dict[int, list[dict[str, Any]]] = {}
    for seed in pw_mod.SPLIT_SEEDS:
        target = np.full(len(groups), np.nan, dtype=np.float64)
        flist: list[dict[str, Any]] = []
        for outer, _held in enumerate(family_folds(groups, pw_mod.OUTER_SPLITS, seed)):
            entry = ctx.fold_entries[(seed, outer)]
            with np.load(entry["path"], allow_pickle=False) as cached:
                train = cached["train"].copy()
                test = cached["test"].copy()
                ytrain = cached["ytrain"].copy()
                ytest = cached["ytest"].copy()
                g_outer = cached["G"].copy()
                target[test] = ytest
                inner_items = []
                for idx, inner_meta in enumerate(entry["meta"]["inner"]):
                    inner_items.append(
                        {
                            "fit": cached[f"fit_{idx}"].copy(),
                            "valid": cached[f"valid_{idx}"].copy(),
                            "yfit": cached[f"yfit_{idx}"].copy(),
                            "yvalid": cached[f"yvalid_{idx}"].copy(),
                            "G": cached[f"G_{idx}"].copy(),
                            "n_val_groups": len(inner_meta["validation_groups"]),
                        }
                    )
            flist.append(
                {
                    "train": train,
                    "test": test,
                    "ytrain": ytrain,
                    "ytest": ytest,
                    "G": g_outer,
                    "n_train_groups": len(set(groups[train].tolist())),
                    "inner": inner_items,
                }
            )
        oof_targets[seed] = target
        fold_raw[seed] = flist

    rung_caches: dict[str, dict[int, list[dict[str, Any]]]] = {}
    u_c_metrics: dict[str, dict[str, np.ndarray]] = {}
    for rung in rungs:
        cols = INTERACTION_RUNGS[rung]
        per_seed: dict[int, list[dict[str, Any]]] = {}
        mse_seeds: list[np.ndarray] = []
        sp_seeds: list[list[float | None]] = []
        for seed in pw_mod.SPLIT_SEEDS:
            seed_cache: list[dict[str, Any]] = []
            pred_uc = np.full(len(groups), np.nan, dtype=np.float64)
            for f in fold_raw[seed]:
                tr, te = f["train"], f["test"]
                outer_blocks = {**blocks_base, "G": f["G"]}
                x_tr = np.column_stack([outer_blocks[k][tr] for k in cols])
                x_te = np.column_stack([outer_blocks[k][te] for k in cols])
                outer_gram = _build_gram_state(x_tr, f["ytrain"], weights[tr], x_te, ALPHAS)

                losses = np.zeros(len(ALPHAS), dtype=np.float64)
                inner_list = []
                for inn in f["inner"]:
                    in_blocks = {**blocks_base, "G": inn["G"]}
                    x_fit = np.column_stack([in_blocks[k][inn["fit"]] for k in cols])
                    x_val = np.column_stack([in_blocks[k][inn["valid"]] for k in cols])
                    ig = _build_gram_state(x_fit, inn["yfit"], weights[inn["fit"]], x_val, ALPHAS)
                    losses += inn["n_val_groups"] * (
                        weights[inn["valid"]][:, None] * (ig.base_pred_alphas - inn["yvalid"][:, None]) ** 2
                    ).sum(0)
                    inner_list.append(
                        {
                            "fit": inn["fit"],
                            "valid": inn["valid"],
                            "yvalid": inn["yvalid"],
                            "n_val_groups": inn["n_val_groups"],
                            "gram": ig,
                        }
                    )
                alpha_idx = len(losses) - 1 - int(np.argmin(losses[::-1]))
                pred_uc[te] = outer_gram.base_pred_alphas[:, alpha_idx]
                seed_cache.append(
                    {
                        "train": tr,
                        "test": te,
                        "ytrain": f["ytrain"],
                        "n_train_groups": f["n_train_groups"],
                        "outer_gram": outer_gram,
                        "inner": inner_list,
                    }
                )
            per_seed[seed] = seed_cache
            labels, mse_vals = pw_mod.group_errors(oof_targets[seed], pred_uc, groups, site_pairs)
            if list(labels) != ctx.group_order:
                raise RuntimeError("R3 group order mismatch on U_C")
            mse_seeds.append(np.asarray(mse_vals, dtype=np.float64))
            _, sp_vals = pw_mod.group_spearman(oof_targets[seed], pred_uc, groups)
            sp_seeds.append(sp_vals)

        rung_caches[rung] = per_seed
        u_c_metrics[rung] = {
            "mse": np.mean(mse_seeds, axis=0),
            "spearman": _average_seed_spearman(sp_seeds, len(ctx.group_order)),
        }
    return rung_caches, oof_targets, u_c_metrics


def _average_seed_spearman(seed_lists: list[list[float | None]], n_groups: int) -> np.ndarray:
    out = np.full(n_groups, np.nan, dtype=np.float64)
    for g_idx in range(n_groups):
        vals = [s[g_idx] for s in seed_lists if s[g_idx] is not None and np.isfinite(s[g_idx])]
        if len(vals) == len(seed_lists):
            out[g_idx] = float(np.mean(vals))
    return out


def _r3_nested_predict_extra(
    seed_cache: list[dict[str, Any]],
    x_extra: np.ndarray,
    weights: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    if x_extra.ndim == 1:
        x_extra = x_extra[:, None]
    pred = np.full(n_rows, np.nan, dtype=np.float64)
    for f in seed_cache:
        losses = np.zeros(len(ALPHAS), dtype=np.float64)
        for inn in f["inner"]:
            p_val = _predict_augmented_from_gram(
                inn["gram"], x_extra[inn["fit"]], x_extra[inn["valid"]], ALPHAS
            )
            losses += inn["n_val_groups"] * (
                weights[inn["valid"]][:, None] * (p_val - inn["yvalid"][:, None]) ** 2
            ).sum(0)
        alpha = ALPHAS[_select_alpha(losses)]
        tr, te = f["train"], f["test"]
        pred[te] = _predict_augmented_from_gram(
            f["outer_gram"], x_extra[tr], x_extra[te], (alpha,)
        )[:, 0]
    if not np.isfinite(pred).all():
        raise RuntimeError("Incomplete R3 out-of-fold prediction")
    return pred


def _r3_ridge_predict_chol(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    x_ev: np.ndarray,
    alphas: tuple[float, ...],
) -> np.ndarray:
    """Exact Cholesky-based ridge_predict (< 1e-14 vs eigh, 3-13x faster)."""
    x = torch.as_tensor(x_tr, dtype=torch.float64)
    y = torch.as_tensor(y_tr, dtype=torch.float64)
    w = torch.as_tensor(w_tr, dtype=torch.float64)
    w_norm = w / w.sum()
    mean = (w_norm[:, None] * x).sum(0)
    xc = x - mean
    scale = (w_norm[:, None] * xc.square()).sum(0).sqrt()
    scale = torch.where(scale < 1e-12, torch.ones_like(scale), scale)
    z = xc / scale
    ym = (w_norm * y).sum()
    cov = z.T @ (w_norm[:, None] * z)
    rhs = z.T @ (w_norm * (y - ym))
    beta = _solve_ridge_gram(cov, rhs, alphas)
    z_ev = (torch.as_tensor(x_ev, dtype=torch.float64) - mean) / scale
    return (z_ev @ beta + ym).numpy()


def _r3_nested_predict_alone(
    seed_cache: list[dict[str, Any]],
    x_model: np.ndarray,
    weights: np.ndarray,
    n_rows: int,
    fold_raw_yfit: dict[int, list[np.ndarray]],
) -> np.ndarray:
    """Exact model-only (M or R alone) prediction on R3 cached folds (with per-fold residual target)."""
    if x_model.ndim == 1:
        x_model = x_model[:, None]
    pred = np.full(n_rows, np.nan, dtype=np.float64)
    for outer_idx, f in enumerate(seed_cache):
        losses = np.zeros(len(ALPHAS), dtype=np.float64)
        for in_idx, inn in enumerate(f["inner"]):
            yfit = fold_raw_yfit[outer_idx][in_idx]
            p_val = _r3_ridge_predict_chol(
                x_model[inn["fit"]], yfit, weights[inn["fit"]], x_model[inn["valid"]], ALPHAS
            )
            losses += inn["n_val_groups"] * (
                weights[inn["valid"]][:, None] * (p_val - inn["yvalid"][:, None]) ** 2
            ).sum(0)
        alpha = ALPHAS[_select_alpha(losses)]
        tr, te = f["train"], f["test"]
        pred[te] = _r3_ridge_predict_chol(
            x_model[tr], f["ytrain"], weights[tr], x_model[te], (alpha,)
        )[:, 0]
    if not np.isfinite(pred).all():
        raise RuntimeError("Incomplete R3 model-only prediction")
    return pred


def _r3_fit_all_r_and_mr_shared(
    rung_caches: dict[str, dict[int, list[dict[str, Any]]]],
    m_cc: np.ndarray,
    r_cc: np.ndarray,
    weights: np.ndarray,
    n_rows: int,
) -> tuple[
    dict[int, np.ndarray],
    dict[str, dict[int, np.ndarray]],
    dict[int, np.ndarray],
]:
    """Fit ``R`` alone, all 3 rungs of ``U_C + R``, and top-rung ``U_C + M + R`` sharing ``cov_rr`` ONCE per split."""
    top_rung = "nuisance+profile+couplings"
    r_preds: dict[int, np.ndarray] = {}
    cr_preds: dict[str, dict[int, np.ndarray]] = {r: {} for r in R3_RUNGS}
    cmr_preds: dict[int, np.ndarray] = {}

    for seed in pw_mod.SPLIT_SEEDS:
        pr_seed = np.full(n_rows, np.nan, dtype=np.float64)
        pcr_seed = {r: np.full(n_rows, np.nan, dtype=np.float64) for r in R3_RUNGS}
        pcmr_seed = np.full(n_rows, np.nan, dtype=np.float64)
        n_outer = len(rung_caches[top_rung][seed])

        for o_idx in range(n_outer):
            f_ref = rung_caches[top_rung][seed][o_idx]
            n_inner = len(f_ref["inner"])
            l_r = np.zeros(len(ALPHAS), dtype=np.float64)
            l_cr = {r: np.zeros(len(ALPHAS), dtype=np.float64) for r in R3_RUNGS}
            l_cmr = np.zeros(len(ALPHAS), dtype=np.float64)

            def _prep_rm_split(
                gs_ref: Any, fit_idx: np.ndarray, ev_idx: np.ndarray
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                w_norm = gs_ref.w_norm
                xr = torch.as_tensor(r_cc[fit_idx], dtype=torch.float64)
                xr_ev = torch.as_tensor(r_cc[ev_idx], dtype=torch.float64)
                mean_r = (w_norm[:, None] * xr).sum(0)
                xrc = xr - mean_r
                scale_r = (w_norm[:, None] * xrc.square()).sum(0).sqrt()
                scale_r = torch.where(scale_r < 1e-12, torch.ones_like(scale_r), scale_r)
                z_r = xrc / scale_r
                z_ev_r = (xr_ev - mean_r) / scale_r
                w_z_r = w_norm[:, None] * z_r
                cov_rr = z_r.T @ w_z_r
                rhs_r = z_r.T @ gs_ref.w_yc

                xm = torch.as_tensor(m_cc[fit_idx], dtype=torch.float64)
                xm_ev = torch.as_tensor(m_cc[ev_idx], dtype=torch.float64)
                mean_m = (w_norm[:, None] * xm).sum(0)
                xmc = xm - mean_m
                scale_m = (w_norm[:, None] * xmc.square()).sum(0).sqrt()
                scale_m = torch.where(scale_m < 1e-12, torch.ones_like(scale_m), scale_m)
                z_m = xmc / scale_m
                z_ev_m = (xm_ev - mean_m) / scale_m
                cov_mm = z_m.T @ (w_norm[:, None] * z_m)
                cov_rm = w_z_r.T @ z_m
                rhs_m = z_m.T @ gs_ref.w_yc
                return z_r, z_ev_r, cov_rr, rhs_r, z_m, z_ev_m, cov_mm, cov_rm, rhs_m, gs_ref.ym

            for i_idx in range(n_inner):
                inn_ref = f_ref["inner"][i_idx]
                fit_idx = inn_ref["fit"]
                val_idx = inn_ref["valid"]
                w_val = weights[val_idx][:, None]
                y_val = inn_ref["yvalid"][:, None]
                ng = inn_ref["n_val_groups"]

                z_r, z_ev_r, cov_rr, rhs_r, z_m, z_ev_m, cov_mm, cov_rm, rhs_m, ym = _prep_rm_split(
                    inn_ref["gram"], fit_idx, val_idx
                )
                kr = cov_rr.shape[0]
                km = cov_mm.shape[0]

                # 1. R alone
                beta_r = _solve_ridge_gram(cov_rr, rhs_r, ALPHAS)
                p_r_val = (z_ev_r @ beta_r + ym).numpy()
                l_r += ng * (w_val * (p_r_val - y_val) ** 2).sum(0)

                # 2. U_C + R across all 3 rungs (reusing cov_rr and rhs_r!)
                for rung in R3_RUNGS:
                    gs = rung_caches[rung][seed][o_idx]["inner"][i_idx]["gram"]
                    p = gs.cov_bb.shape[0]
                    cov_br = gs.w_z_base.T @ z_r
                    cov_cr = torch.empty((p + kr, p + kr), dtype=torch.float64)
                    cov_cr[:p, :p] = gs.cov_bb
                    cov_cr[:p, p:] = cov_br
                    cov_cr[p:, :p] = cov_br.T
                    cov_cr[p:, p:] = cov_rr
                    rhs_cr = torch.cat([gs.rhs_b, rhs_r], dim=0)
                    beta_cr = _solve_ridge_gram(cov_cr, rhs_cr, ALPHAS)
                    p_cr_val = (gs.z_eval_base @ beta_cr[:p] + z_ev_r @ beta_cr[p:] + ym).numpy()
                    l_cr[rung] += ng * (w_val * (p_cr_val - y_val) ** 2).sum(0)

                    # 3. U_C + M + R on top rung (reusing cov_rr, cov_br, rhs_r!)
                    if rung == top_rung:
                        cov_bm = gs.w_z_base.T @ z_m
                        cov_cmr = torch.empty((p + km + kr, p + km + kr), dtype=torch.float64)
                        cov_cmr[:p, :p] = gs.cov_bb
                        cov_cmr[:p, p : p + km] = cov_bm
                        cov_cmr[:p, p + km :] = cov_br
                        cov_cmr[p : p + km, :p] = cov_bm.T
                        cov_cmr[p : p + km, p : p + km] = cov_mm
                        cov_cmr[p : p + km, p + km :] = cov_rm.T
                        cov_cmr[p + km :, :p] = cov_br.T
                        cov_cmr[p + km :, p : p + km] = cov_rm
                        cov_cmr[p + km :, p + km :] = cov_rr
                        rhs_cmr = torch.cat([gs.rhs_b, rhs_m, rhs_r], dim=0)
                        beta_cmr = _solve_ridge_gram(cov_cmr, rhs_cmr, ALPHAS)
                        p_cmr_val = (
                            gs.z_eval_base @ beta_cmr[:p]
                            + z_ev_m @ beta_cmr[p : p + km]
                            + z_ev_r @ beta_cmr[p + km :]
                            + ym
                        ).numpy()
                        l_cmr += ng * (w_val * (p_cmr_val - y_val) ** 2).sum(0)

            # Outer fold prediction (1 alpha per design -> 13x faster Cholesky!).
            tr_idx = f_ref["train"]
            te_idx = f_ref["test"]
            z_r, z_ev_r, cov_rr, rhs_r, z_m, z_ev_m, cov_mm, cov_rm, rhs_m, ym = _prep_rm_split(
                f_ref["outer_gram"], tr_idx, te_idx
            )
            kr = cov_rr.shape[0]
            km = cov_mm.shape[0]

            a_r = ALPHAS[_select_alpha(l_r)]
            beta_r = _solve_ridge_gram(cov_rr, rhs_r, (a_r,))
            pr_seed[te_idx] = (z_ev_r @ beta_r + ym).numpy()[:, 0]

            for rung in R3_RUNGS:
                gs = rung_caches[rung][seed][o_idx]["outer_gram"]
                p = gs.cov_bb.shape[0]
                cov_br = gs.w_z_base.T @ z_r
                cov_cr = torch.empty((p + kr, p + kr), dtype=torch.float64)
                cov_cr[:p, :p] = gs.cov_bb
                cov_cr[:p, p:] = cov_br
                cov_cr[p:, :p] = cov_br.T
                cov_cr[p:, p:] = cov_rr
                rhs_cr = torch.cat([gs.rhs_b, rhs_r], dim=0)
                a_cr = ALPHAS[_select_alpha(l_cr[rung])]
                beta_cr = _solve_ridge_gram(cov_cr, rhs_cr, (a_cr,))
                pcr_seed[rung][te_idx] = (gs.z_eval_base @ beta_cr[:p] + z_ev_r @ beta_cr[p:] + ym).numpy()[:, 0]

                if rung == top_rung:
                    cov_bm = gs.w_z_base.T @ z_m
                    cov_cmr = torch.empty((p + km + kr, p + km + kr), dtype=torch.float64)
                    cov_cmr[:p, :p] = gs.cov_bb
                    cov_cmr[:p, p : p + km] = cov_bm
                    cov_cmr[:p, p + km :] = cov_br
                    cov_cmr[p : p + km, :p] = cov_bm.T
                    cov_cmr[p : p + km, p : p + km] = cov_mm
                    cov_cmr[p : p + km, p + km :] = cov_rm.T
                    cov_cmr[p + km :, :p] = cov_br.T
                    cov_cmr[p + km :, p : p + km] = cov_rm
                    cov_cmr[p + km :, p + km :] = cov_rr
                    rhs_cmr = torch.cat([gs.rhs_b, rhs_m, rhs_r], dim=0)
                    a_cmr = ALPHAS[_select_alpha(l_cmr)]
                    beta_cmr = _solve_ridge_gram(cov_cmr, rhs_cmr, (a_cmr,))
                    pcmr_seed[te_idx] = (
                        gs.z_eval_base @ beta_cmr[:p]
                        + z_ev_m @ beta_cmr[p : p + km]
                        + z_ev_r @ beta_cmr[p + km :]
                        + ym
                    ).numpy()[:, 0]

        if not np.isfinite(pr_seed).all() or not np.isfinite(pcmr_seed).all():
            raise RuntimeError("Incomplete R3 shared R/MR prediction")
        r_preds[seed] = pr_seed
        cmr_preds[seed] = pcmr_seed
        for rung in R3_RUNGS:
            if not np.isfinite(pcr_seed[rung]).all():
                raise RuntimeError(f"Incomplete R3 U_C+R prediction on {rung}")
            cr_preds[rung][seed] = pcr_seed[rung]

    return r_preds, cr_preds, cmr_preds


def _load_yfit_map(ctx: PairwiseCycleContext) -> dict[int, dict[int, list[np.ndarray]]]:
    yfit_map: dict[int, dict[int, list[np.ndarray]]] = {}
    groups = ctx.panel["group"]
    for seed in pw_mod.SPLIT_SEEDS:
        yfit_map[seed] = {}
        for outer, _ in enumerate(family_folds(groups, pw_mod.OUTER_SPLITS, seed)):
            entry = ctx.fold_entries[(seed, outer)]
            with np.load(entry["path"], allow_pickle=False) as cached:
                yfit_map[seed][outer] = [
                    cached[f"yfit_{idx}"].copy() for idx in range(len(entry["meta"]["inner"]))
                ]
    return yfit_map


def _r3_fit_arm_with_caches(
    ctx: PairwiseCycleContext,
    rung_caches: dict[str, dict[int, list[dict[str, Any]]]],
    oof_targets: dict[int, np.ndarray],
    u_c_metrics: dict[str, dict[str, np.ndarray]],
    yfit_map: dict[int, dict[int, list[np.ndarray]]],
    m_cc: np.ndarray,
    r_cc: np.ndarray | None,
) -> dict[str, Any]:
    """Fit M_cycle (3 rungs) and optionally R_cycle (3 rungs) + MR_cycle (top rung) for one arm."""
    panel = ctx.panel
    groups = panel["group"]
    site_pairs = panel["site_pair"]
    weights = ctx.weights
    n_rows = len(groups)
    any_rung = R3_RUNGS[0]

    def _score_seed_preds(preds_by_seed: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        mse_s, sp_s = [], []
        for seed in pw_mod.SPLIT_SEEDS:
            _, mse_v = pw_mod.group_errors(oof_targets[seed], preds_by_seed[seed], groups, site_pairs)
            _, sp_v = pw_mod.group_spearman(oof_targets[seed], preds_by_seed[seed], groups)
            mse_s.append(np.asarray(mse_v, dtype=np.float64))
            sp_s.append(sp_v)
        return np.mean(mse_s, axis=0), _average_seed_spearman(sp_s, len(ctx.group_order))

    # M alone
    m_preds = {
        seed: _r3_nested_predict_alone(rung_caches[any_rung][seed], m_cc, weights, n_rows, yfit_map[seed])
        for seed in pw_mod.SPLIT_SEEDS
    }
    m_mse, m_sp = _score_seed_preds(m_preds)

    m_channel: dict[str, dict[str, dict[str, list[float]]]] = {"mse": {}, "spearman": {}}
    for rung in R3_RUNGS:
        cm_preds = {
            seed: _r3_nested_predict_extra(rung_caches[rung][seed], m_cc, weights, n_rows)
            for seed in pw_mod.SPLIT_SEEDS
        }
        cm_mse, cm_sp = _score_seed_preds(cm_preds)
        m_channel["mse"][rung] = {
            "U_C": u_c_metrics[rung]["mse"].tolist(),
            "M": m_mse.tolist(),
            "U_C_M": cm_mse.tolist(),
        }
        m_channel["spearman"][rung] = {
            "U_C": u_c_metrics[rung]["spearman"].tolist(),
            "M": m_sp.tolist(),
            "U_C_M": cm_sp.tolist(),
        }

    r_channel: dict[str, dict[str, dict[str, list[float]]]] | None = None
    mr_top: dict[str, Any] | None = None
    if r_cc is not None:
        r_preds, cr_preds_by_rung, cmr_preds = _r3_fit_all_r_and_mr_shared(
            rung_caches, m_cc, r_cc, weights, n_rows
        )
        r_mse, r_sp = _score_seed_preds(r_preds)
        r_channel = {"mse": {}, "spearman": {}}
        for rung in R3_RUNGS:
            cr_mse, cr_sp = _score_seed_preds(cr_preds_by_rung[rung])
            r_channel["mse"][rung] = {
                "U_C": u_c_metrics[rung]["mse"].tolist(),
                "M": r_mse.tolist(),
                "U_C_M": cr_mse.tolist(),
            }
            r_channel["spearman"][rung] = {
                "U_C": u_c_metrics[rung]["spearman"].tolist(),
                "M": r_sp.tolist(),
                "U_C_M": cr_sp.tolist(),
            }

        top_rung = "nuisance+profile+couplings"
        cmr_mse, cmr_sp = _score_seed_preds(cmr_preds)
        mr_top = {
            "rung": top_rung,
            "mse": {
                "U_C": u_c_metrics[top_rung]["mse"].tolist(),
                "M": m_mse.tolist(),
                "R": r_mse.tolist(),
                "U_C_M": m_channel["mse"][top_rung]["U_C_M"],
                "U_C_R": r_channel["mse"][top_rung]["U_C_M"],
                "U_C_MR": cmr_mse.tolist(),
            },
            "spearman": {
                "U_C": u_c_metrics[top_rung]["spearman"].tolist(),
                "M": m_sp.tolist(),
                "R": r_sp.tolist(),
                "U_C_M": m_channel["spearman"][top_rung]["U_C_M"],
                "U_C_R": r_channel["spearman"][top_rung]["U_C_M"],
                "U_C_MR": cmr_sp.tolist(),
            },
        }

    return {"M_cycle": m_channel, "R_cycle": r_channel, "MR_cycle_top_rung": mr_top}


def _load_preregistered_33_r3_arrays(
    arm_id: str,
    ctx: PairwiseCycleContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load exact 8,192-row ``M_cycle``, 1024-d ``R_cycle``, and per-arm ``T`` (CC rows) for a 33-arm model."""
    ext_dir = ctx.root / "results/pairwise_epistasis_20260924/extraction" / arm_id
    arm_data, _ = fit_pw.load_arm(ext_dir, arm_id, ctx.plan)
    m_all: list[float] = []
    r_chunks: list[np.ndarray] = []
    t_all: list[np.ndarray] = []

    for bg in ctx.plan["backgrounds"]:
        name = bg["name"]
        wt = bg["wildtype"]
        adata = arm_data[name]
        local_states = np.asarray([c["states"] for c in bg["cycles"]], dtype=np.int64)
        likelihood = adata["likelihood"]
        m_all.extend(pw_mod.cycle_contrast(likelihood, local_states).tolist())
        projected = adata["projected"].reshape(len(likelihood), -1)
        r_chunks.append(pw_mod.cycle_contrast(projected, local_states))
        pooled, ids = adata["pooled_token_counts"], adata["token_ids"]
        for cyc in bg["cycles"]:
            t_all.append(pw_mod.tokenisation_features(pooled, ids, tuple(cyc["states"]), len(wt)))

    m_arr = np.asarray(m_all, dtype=np.float64)
    r_arr = np.concatenate(r_chunks, axis=0).astype(np.float64)
    t_arr = np.asarray(t_all, dtype=np.float64)
    if m_arr.shape != ctx.raw_all_epsilon.shape:
        raise ValueError(f"{arm_id}: R3 M shape {m_arr.shape} != {ctx.raw_all_epsilon.shape}")
    return m_arr, r_arr, t_arr[ctx.kept_global_indices]


def _load_expanded_r3_arrays(
    arm_id: str,
    ctx: PairwiseCycleContext,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load exact 8,192-row ``M_cycle`` (and optional 1024-d ``R_cycle``) for one expanded_32 arm."""
    art = resolve_gp_r3_cycles(arm_id, root=ctx.root)
    if arm_id == "esm3-1.4b":
        rows = [json.loads(line) for line in art.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected_keys = [
            (bg["name"], cyc["positions"], cyc["separation"])
            for bg in ctx.plan["backgrounds"]
            for cyc in bg["cycles"]
        ]
        observed_keys = [(r["background"], r["positions"], r["separation"]) for r in rows]
        if observed_keys != expected_keys:
            raise ValueError(f"esm3-1.4b: R3 cycle keys mismatch ({len(observed_keys)} vs {len(expected_keys)})")
        eps_all = np.asarray([float(r["epsilon_pll_nats"]) for r in rows], dtype=np.float64)

        # Load esm3-1.4b 1024-d R_cycle archive staged by ops-staging-esm3 (wait briefly if unpacking).
        r_path = ctx.root / UNIFIED_OUTPUT_DIR / "r2_readout_archives/r3_cycles_R_esm3-1.4b.npz"
        deadline = time.monotonic() + 300.0
        r_all: np.ndarray | None = None
        last_err: Exception | None = None
        while time.monotonic() <= deadline:
            if r_path.is_file() and r_path.stat().st_size > 0:
                try:
                    with np.load(r_path, allow_pickle=False) as rdata:
                        key = "R_cycle" if "R_cycle" in rdata else "R"
                        arr = np.asarray(rdata[key], dtype=np.float64)
                    if arr.shape != (len(eps_all), 1024):
                        raise ValueError(
                            f"esm3-1.4b: r3_cycles_R shape {arr.shape} != {(len(eps_all), 1024)}"
                        )
                    if not np.isfinite(arr).all():
                        raise ValueError("esm3-1.4b: non-finite values in r3_cycles_R_esm3-1.4b.npz")
                    r_all = arr
                    break
                except Exception as exc:  # file may still be landing/unpacking
                    last_err = exc
            time.sleep(5.0)
        if r_all is None:
            raise FileNotFoundError(
                f"esm3-1.4b: missing or unreadable {r_path} after waiting 300s (last_err={last_err})"
            )
        return eps_all, r_all

    with np.load(art.path, allow_pickle=False) as data:
        eps_all = np.asarray(data["cycle_eps_pll"], dtype=np.float64)
        r_all = np.asarray(data["R_cycle"], dtype=np.float64) if "R_cycle" in data else None
    if eps_all.shape != ctx.raw_all_epsilon.shape:
        raise ValueError(f"{arm_id}: R3 cycle_eps_pll shape {eps_all.shape} != {ctx.raw_all_epsilon.shape}")
    return eps_all, r_all


def _zero_shot_per_group_spearman(
    scores: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    expected_groups: list[Any],
) -> np.ndarray:
    labels, vals = pw_mod.group_spearman(target, scores, groups)
    if list(labels) != expected_groups:
        raise RuntimeError("R3 zero-shot group order mismatch")
    return np.asarray([np.nan if v is None else float(v) for v in vals], dtype=np.float64)


# --------------------------------------------------------------------------- #
# R3 Reproduction Gate (< 1e-12 vs cached 20261110 interaction cell)
# --------------------------------------------------------------------------- #


def verify_r3_reproduction_gate(
    root: Path = REPO_ROOT,
    arm: str = "progen2-small",
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Verify fast R3 cached-fold engine reproduces cached 20261110 interaction cell to < 1e-12."""
    torch.set_num_threads(1)
    ctx = load_pairwise_cycle_context(root)
    m_all, _, t_cc = _load_preregistered_33_r3_arrays(arm, ctx)
    blocks_arm = {**ctx.panel["blocks"], "T": t_cc}
    top_rung = "nuisance+profile+couplings"
    rung_caches, oof_targets, u_c_metrics = _r3_build_rung_caches(ctx, blocks_arm, rungs=(top_rung,))
    yfit_map = _load_yfit_map(ctx)
    m_cc = m_all[ctx.kept_global_indices][:, None]

    # Fit M alone + U_C_M on top rung
    groups, site_pairs, weights = ctx.panel["group"], ctx.panel["site_pair"], ctx.weights
    m_mse_s, cm_mse_s = [], []
    for seed in pw_mod.SPLIT_SEEDS:
        pm = _r3_nested_predict_alone(rung_caches[top_rung][seed], m_cc, weights, len(groups), yfit_map[seed])
        pcm = _r3_nested_predict_extra(rung_caches[top_rung][seed], m_cc, weights, len(groups))
        _, em = pw_mod.group_errors(oof_targets[seed], pm, groups, site_pairs)
        _, ecm = pw_mod.group_errors(oof_targets[seed], pcm, groups, site_pairs)
        m_mse_s.append(np.asarray(em, dtype=np.float64))
        cm_mse_s.append(np.asarray(ecm, dtype=np.float64))

    computed = {
        "U_C": u_c_metrics[top_rung]["mse"],
        "M": np.mean(m_mse_s, axis=0),
        "U_C_M": np.mean(cm_mse_s, axis=0),
    }
    cached_path = (
        root
        / "results/extensions/explicit_statistics_lanes_20261110/interaction/cells"
        / "cycle_squared_error/likelihood/complete_case"
        / top_rung
        / f"{arm}.json"
    )
    cached = json.loads(cached_path.read_text(encoding="utf-8"))
    max_diff = max(
        float(np.max(np.abs(computed[k] - np.asarray(cached["values"][k], dtype=np.float64))))
        for k in ("U_C", "M", "U_C_M")
    )
    if max_diff >= tolerance:
        raise AssertionError(f"R3 reproduction gate failed on {arm}: max_abs_diff={max_diff:.3e}")
    return {
        "arm": arm,
        "rung": top_rung,
        "tolerance": tolerance,
        "max_abs_diff": max_diff,
        "passed": True,
    }


# --------------------------------------------------------------------------- #
# Parallel CPU Worker for R3 Epistasis (<= 4 workers)
# --------------------------------------------------------------------------- #

_WORKER_CTX: PairwiseCycleContext | None = None
_WORKER_YFIT: dict[int, dict[int, list[np.ndarray]]] | None = None
_WORKER_AA_CACHES: tuple[Any, Any, Any] | None = None
_WORKER_CACHE_DIR: Path | None = None


def _init_r3_worker(root_str: str, cache_dir_str: str) -> None:
    global _WORKER_CTX, _WORKER_YFIT, _WORKER_AA_CACHES, _WORKER_CACHE_DIR
    torch.set_num_threads(4)
    root = Path(root_str)
    _WORKER_CTX = load_pairwise_cycle_context(root)
    _WORKER_YFIT = _load_yfit_map(_WORKER_CTX)
    _WORKER_AA_CACHES = _r3_build_rung_caches(_WORKER_CTX, _WORKER_CTX.panel["blocks"], rungs=R3_RUNGS)
    _WORKER_CACHE_DIR = Path(cache_dir_str)


def _r3_worker_fit_arm(arm_id: str) -> dict[str, Any]:
    assert _WORKER_CTX is not None and _WORKER_YFIT is not None and _WORKER_AA_CACHES is not None
    assert _WORKER_CACHE_DIR is not None
    ctx = _WORKER_CTX
    t0 = time.monotonic()
    idx_cc = ctx.kept_global_indices
    is_pre33 = ARM_METADATA[arm_id].origin_cohort == "preregistered_33"

    if is_pre33:
        m_all, r_all, t_cc = _load_preregistered_33_r3_arrays(arm_id, ctx)
        # Reuse AA20 Gram cache when arm's T block is identical to ctx.panel['blocks']['T'].
        if np.array_equal(t_cc, ctx.panel["blocks"]["T"]):
            rung_caches, oof_targets, u_c_metrics = _WORKER_AA_CACHES
        else:
            blocks_arm = {**ctx.panel["blocks"], "T": t_cc}
            rung_caches, oof_targets, u_c_metrics = _r3_build_rung_caches(ctx, blocks_arm, rungs=R3_RUNGS)
    else:
        m_all, r_all = _load_expanded_r3_arrays(arm_id, ctx)
        rung_caches, oof_targets, u_c_metrics = _WORKER_AA_CACHES

    m_cc = m_all[idx_cc][:, None]
    r_cc = None if r_all is None else r_all[idx_cc]
    fit_res = _r3_fit_arm_with_caches(
        ctx, rung_caches, oof_targets, u_c_metrics, _WORKER_YFIT, m_cc, r_cc
    )

    mean_res_target = np.mean([oof_targets[s] for s in pw_mod.SPLIT_SEEDS], axis=0)
    zs_cc_raw = _zero_shot_per_group_spearman(
        m_all[idx_cc], ctx.panel["epsilon"], ctx.panel["group"], ctx.group_order
    )
    zs_cc_res = _zero_shot_per_group_spearman(
        m_all[idx_cc], mean_res_target, ctx.panel["group"], ctx.group_order
    )
    all_groups_order = sorted(set(ctx.raw_all_groups.tolist()))
    zs_all_raw = _zero_shot_per_group_spearman(
        m_all, ctx.raw_all_epsilon, ctx.raw_all_groups, all_groups_order
    )

    # Verify cached 20261110 M_cycle MSE for preregistered_33 arms (< 1e-10) and keep exact vectors.
    max_cached_diff = None
    if is_pre33:
        diffs = []
        croot = (
            ctx.root
            / "results/extensions/explicit_statistics_lanes_20261110/interaction/cells"
            / "cycle_squared_error/likelihood/complete_case"
        )
        for rung in R3_RUNGS:
            cdoc = json.loads((croot / rung / f"{arm_id}.json").read_text(encoding="utf-8"))
            for key in ("U_C", "M", "U_C_M"):
                diffs.append(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(fit_res["M_cycle"]["mse"][rung][key])
                                - np.asarray(cdoc["values"][key])
                            )
                        )
                    )
                )
        max_cached_diff = max(diffs)

    doc = {
        "arm_id": arm_id,
        "support_sha256": ctx.support_sha256,
        "groups": ctx.group_order,
        "all_64_groups": all_groups_order,
        "zs_cc_raw_spearman": zs_cc_raw.tolist(),
        "zs_cc_res_spearman": zs_cc_res.tolist(),
        "zs_all_64_raw_spearman": zs_all_raw.tolist(),
        "M_cycle": fit_res["M_cycle"],
        "R_cycle": fit_res["R_cycle"],
        "MR_cycle_top_rung": fit_res["MR_cycle_top_rung"],
        "max_cached_20261110_mse_diff": max_cached_diff,
        "seconds": time.monotonic() - t0,
    }
    atomic_write_json(_WORKER_CACHE_DIR / f"{arm_id}.json", doc)
    print(f"[R3-epistasis] finished {arm_id} ({doc['seconds']:.1f}s, cached_diff={max_cached_diff})", flush=True)
    return doc


def _build_r3_epistasis_section(
    root: Path,
    cache_dir: Path,
    *,
    workers: int = 4,
    force_refit: bool = False,
) -> dict[str, Any]:
    t0 = time.monotonic()
    ctx = load_pairwise_cycle_context(root)
    ep_cache_dir = cache_dir / "r3_epistasis"
    ep_cache_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, dict[str, Any]] = {}
    need_fit: list[str] = []
    for arm_id in UNIFIED_62_SINGLE_SEQUENCE_ARMS:
        cfile = ep_cache_dir / f"{arm_id}.json"
        if not force_refit and cfile.is_file():
            doc = json.loads(cfile.read_text(encoding="utf-8"))
            if (
                doc.get("support_sha256") == ctx.support_sha256
                and doc.get("R_cycle") is not None
                and doc.get("MR_cycle_top_rung") is not None
            ):
                loaded[arm_id] = doc
                continue
        need_fit.append(arm_id)
    # Schedule esm3-1.4b last so its freshly staged R_cycle archive has maximum time to land.
    if "esm3-1.4b" in need_fit:
        need_fit = [a for a in need_fit if a != "esm3-1.4b"] + ["esm3-1.4b"]

    if need_fit:
        n_workers = max(1, min(workers, 4, len(need_fit)))
        print(f"[R3-epistasis] fitting {len(need_fit)} arms (workers={n_workers})...", flush=True)
        if n_workers == 1:
            _init_r3_worker(str(root), str(ep_cache_dir))
            for arm_id in need_fit:
                loaded[arm_id] = _r3_worker_fit_arm(arm_id)
        else:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_r3_worker,
                initargs=(str(root), str(ep_cache_dir)),
            ) as pool:
                for doc in pool.map(_r3_worker_fit_arm, need_fit):
                    loaded[doc["arm_id"]] = doc

    # Explicit Potts zero-shot reference on 61 CC groups (raw epsilon and residualized epsilon).
    _, oof_targets, _ = _r3_build_rung_caches(ctx, ctx.panel["blocks"], rungs=("nuisance",))
    mean_res_target = np.mean([oof_targets[s] for s in pw_mod.SPLIT_SEEDS], axis=0)
    potts_zs_raw = _zero_shot_per_group_spearman(
        ctx.potts_cycle_raw, ctx.panel["epsilon"], ctx.panel["group"], ctx.group_order
    )
    potts_zs_res = _zero_shot_per_group_spearman(
        ctx.potts_cycle_raw, mean_res_target, ctx.panel["group"], ctx.group_order
    )

    zs_raw_by_arm = {a: np.asarray(loaded[a]["zs_cc_raw_spearman"], dtype=np.float64) for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS}
    zs_res_by_arm = {a: np.asarray(loaded[a]["zs_cc_res_spearman"], dtype=np.float64) for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS}

    # Check if any group has NaN Spearman (e.g. constant zero M_cycle predictions); handle gracefully.
    def _clean_zs_dict(d: dict[str, np.ndarray], ref: np.ndarray) -> tuple[dict[str, np.ndarray], list[str], dict[str, int]]:
        nan_counts = {a: int(np.isnan(v).sum()) for a, v in d.items() if np.isnan(v).any()}
        # Only drop groups if ALL or some arms have NaN? Better: report exact non-NaN count per arm,
        # and for paired_bands restrict to groups finite across all arms (or replace only if none).
        finite_mask = np.isfinite(ref)
        for v in d.values():
            finite_mask &= np.isfinite(v)
        kept_groups = [g for g, ok in zip(ctx.group_order, finite_mask) if ok]
        cleaned = {a: v[finite_mask] for a, v in d.items()}
        return cleaned, kept_groups, nan_counts

    zs_raw_clean, zs_raw_groups, zs_raw_nans = _clean_zs_dict(zs_raw_by_arm, potts_zs_raw)
    zs_res_clean, zs_res_groups, zs_res_nans = _clean_zs_dict(zs_res_by_arm, potts_zs_res)
    potts_raw_mask = np.isin(ctx.group_order, zs_raw_groups)
    potts_res_mask = np.isin(ctx.group_order, zs_res_groups)

    # Assemble M_cycle (62 models) and R_cycle / MR_cycle (61 models).
    m_mse_cells: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    m_sp_cells: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for arm_id in UNIFIED_62_SINGLE_SEQUENCE_ARMS:
        m_mse_cells[arm_id] = {
            rung: {k: np.asarray(v, dtype=np.float64) for k, v in loaded[arm_id]["M_cycle"]["mse"][rung].items()}
            for rung in R3_RUNGS
        }
        m_sp_cells[arm_id] = {
            rung: {k: np.asarray(v, dtype=np.float64) for k, v in loaded[arm_id]["M_cycle"]["spearman"][rung].items()}
            for rung in R3_RUNGS
        }

    r_mse_cells: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    r_sp_cells: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    mr_by_arm: dict[str, dict[str, Any]] = {}
    for arm_id in UNIFIED_62_WITH_R_ARMS:
        doc = loaded[arm_id]
        if doc["R_cycle"] is None or doc["MR_cycle_top_rung"] is None:
            raise RuntimeError(f"{arm_id}: missing R_cycle / MR_cycle_top_rung")
        r_mse_cells[arm_id] = {
            rung: {k: np.asarray(v, dtype=np.float64) for k, v in doc["R_cycle"]["mse"][rung].items()}
            for rung in R3_RUNGS
        }
        r_sp_cells[arm_id] = {
            rung: {k: np.asarray(v, dtype=np.float64) for k, v in doc["R_cycle"]["spearman"][rung].items()}
            for rung in R3_RUNGS
        }
        mr_by_arm[arm_id] = doc["MR_cycle_top_rung"]

    # Helper to band Spearman when some noisy 1D M_alone predictions may be constant on 1-2 groups.
    def _band_spearman_safe(
        cells: dict[str, dict[str, dict[str, np.ndarray]]],
        arm_order: tuple[str, ...],
        rungs: tuple[str, ...],
    ) -> dict[str, Any]:
        # Identify groups finite across all arms/rungs/designs (forward & reverse).
        n_g = len(ctx.group_order)
        fwd_ok = np.ones(n_g, dtype=bool)
        all_ok = np.ones(n_g, dtype=bool)
        for a in arm_order:
            for rung in rungs:
                c = cells[a][rung]
                fwd_ok &= np.isfinite(c["U_C"]) & np.isfinite(c["U_C_M"])
                all_ok &= np.isfinite(c["U_C"]) & np.isfinite(c["U_C_M"]) & np.isfinite(c["M"])
        # Always band forward+reverse on all_ok groups; if all_ok < n_g, also report forward-only on fwd_ok.
        sub_all = {
            a: {r: {k: v[all_ok] for k, v in cells[a][r].items()} for r in rungs}
            for a in arm_order
        }
        out_sp: dict[str, Any] = {
            "groups_finite_both_directions": int(all_ok.sum()),
            "groups_finite_forward": int(fwd_ok.sum()),
            "groups_dropped_nonfinite_m_alone": [g for g, ok in zip(ctx.group_order, all_ok) if not ok],
            "bidirectional": band_bidirectional_family(
                sub_all, arm_order=arm_order, rungs=rungs, to_loss=-1.0
            ),
        }
        if int(fwd_ok.sum()) > int(all_ok.sum()):
            # Forward-only over all finite-forward groups (usually all 61/61).
            cols = []
            meta_list = []
            for rung in rungs:
                for a in arm_order:
                    c = cells[a][rung]
                    cols.append((c["U_C_M"] - c["U_C"])[fwd_ok])
                    meta_list.append({"rung": rung, "arm": a, "direction": "model_given_controls"})
            mat = np.column_stack(cols)
            fb = paired_bands_complete(mat)
            out_sp["forward_only_full_finite_groups"] = {
                "groups": int(fwd_ok.sum()),
                "family_size": int(mat.shape[1]),
                "native_critical_value": float(fb["critical_value"]),
                "studentized_critical_t": float(fb["studentized_critical_t"]),
                "states_native": bidir.counts(fb["verdict"]),
                "states_marginal": bidir.counts(fb["marginal_verdict"]),
                "states_studentized": bidir.counts(fb["studentized_verdict"]),
                "columns": [
                    {
                        **meta_list[i],
                        "point": float(fb["point"][i]),
                        "interval": fb["interval"][i],
                        "verdict": fb["verdict"][i],
                        "marginal_interval": fb["marginal_interval"][i],
                        "marginal_verdict": fb["marginal_verdict"][i],
                    }
                    for i in range(len(meta_list))
                ],
            }
        return out_sp

    # Band MR_cycle (62 arms) over nuisance+profile+couplings on both MSE and Spearman.
    mr_summary: dict[str, Any] = {}
    for metric_key, to_loss in (("mse", 1.0), ("spearman", -1.0)):
        mr_contrasts: dict[str, Any] = {}
        for label, red_k, joint_k in (
            ("MR_given_controls", "U_C", "U_C_MR"),
            ("R_given_controls", "U_C", "U_C_R"),
            ("M_given_controls", "U_C", "U_C_M"),
            ("R_given_controls_and_M", "U_C_M", "U_C_MR"),
            ("M_given_controls_and_R", "U_C_R", "U_C_MR"),
            ("controls_given_R", "R", "U_C_R"),
        ):
            finite_mask = np.ones(len(ctx.group_order), dtype=bool)
            for a in UNIFIED_62_WITH_R_ARMS:
                d = mr_by_arm[a][metric_key]
                finite_mask &= np.isfinite(d[red_k]) & np.isfinite(d[joint_k])
            mat = np.column_stack(
                [
                    to_loss
                    * (
                        np.asarray(mr_by_arm[a][metric_key][red_k], dtype=np.float64)[finite_mask]
                        - np.asarray(mr_by_arm[a][metric_key][joint_k], dtype=np.float64)[finite_mask]
                    )
                    for a in UNIFIED_62_WITH_R_ARMS
                ]
            )
            cb = paired_bands_complete(mat)
            arms_c: dict[str, Any] = {}
            for idx, a in enumerate(UNIFIED_62_WITH_R_ARMS):
                d = mr_by_arm[a][metric_key]
                arms_c[a] = {
                    "paradigm": ARM_METADATA[a].paradigm,
                    "reduced_mean": float(np.mean(np.asarray(d[red_k], dtype=np.float64)[finite_mask])),
                    "joint_mean": float(np.mean(np.asarray(d[joint_k], dtype=np.float64)[finite_mask])),
                    "point": float(cb["point"][idx]),
                    "interval": cb["interval"][idx],
                    "verdict": cb["verdict"][idx],
                    "marginal_interval": cb["marginal_interval"][idx],
                    "marginal_verdict": cb["marginal_verdict"][idx],
                    "studentized_interval": cb["studentized_interval"][idx],
                    "studentized_verdict": cb["studentized_verdict"][idx],
                }
            mr_contrasts[label] = {
                "reduced": red_k,
                "joint": joint_k,
                "groups": int(finite_mask.sum()),
                "family_size": len(UNIFIED_62_WITH_R_ARMS),
                "native_critical_value": float(cb["critical_value"]),
                "studentized_critical_t": float(cb["studentized_critical_t"]),
                "states_native": bidir.counts(cb["verdict"]),
                "states_marginal": bidir.counts(cb["marginal_verdict"]),
                "states_studentized": bidir.counts(cb["studentized_verdict"]),
                "arms": arms_c,
            }
        mr_summary[metric_key] = mr_contrasts

    max_pre33_diff = max(
        (
            float(loaded[a]["max_cached_20261110_mse_diff"])
            for a in PREREGISTERED_33_ARMS
            if loaded[a].get("max_cached_20261110_mse_diff") is not None
        ),
        default=0.0,
    )

    return {
        "support": {
            "name": "complete_case_indel_excluded",
            "support_sha256": ctx.support_sha256,
            "n_groups": len(ctx.group_order),
            "n_cycles": int(len(ctx.panel["epsilon"])),
            "n_site_pairs": int(len(set(ctx.panel["site_pair"].tolist()))),
            "n_arms_M_cycle": len(UNIFIED_62_SINGLE_SEQUENCE_ARMS),
            "n_arms_R_and_MR_cycle": len(UNIFIED_62_WITH_R_ARMS),
            "r_cycle_coverage": "62/62 (100.0%: 33 preregistered + 28 GP28 + esm3-1.4b)",
            "max_cached_20261110_mse_diff_across_33_arms": max_pre33_diff,
            "rungs": {k: list(INTERACTION_RUNGS[k]) for k in R3_RUNGS},
        },
        "zero_shot_epistasis_62": {
            "ZS_raw_cc_61": {
                "nan_group_counts_by_arm": zs_raw_nans,
                **build_zero_shot_section(
                    zs_raw_clean,
                    {"potts": potts_zs_raw[potts_raw_mask]},
                    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
                    zs_raw_groups,
                ),
            },
            "ZS_res_cc_61": {
                "nan_group_counts_by_arm": zs_res_nans,
                **build_zero_shot_section(
                    zs_res_clean,
                    {"potts": potts_zs_res[potts_res_mask]},
                    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
                    zs_res_groups,
                ),
            },
        },
        "M_cycle_62": {
            "cycle_squared_error": band_bidirectional_family(
                m_mse_cells,
                arm_order=UNIFIED_62_SINGLE_SEQUENCE_ARMS,
                rungs=R3_RUNGS,
                to_loss=1.0,
            ),
            "cycle_spearman": _band_spearman_safe(
                m_sp_cells, UNIFIED_62_SINGLE_SEQUENCE_ARMS, R3_RUNGS
            ),
        },
        "R_cycle_62": {
            "cycle_squared_error": band_bidirectional_family(
                r_mse_cells,
                arm_order=UNIFIED_62_WITH_R_ARMS,
                rungs=R3_RUNGS,
                to_loss=1.0,
            ),
            "cycle_spearman": _band_spearman_safe(
                r_sp_cells, UNIFIED_62_WITH_R_ARMS, R3_RUNGS
            ),
        },
        "MR_cycle_62_over_nuisance_profile_couplings": mr_summary,
        "elapsed_seconds": time.monotonic() - t0,
    }


# --------------------------------------------------------------------------- #
# Unified R3 Contacts (22-arm E01 propagation + 16-arm attention/APC map)
# --------------------------------------------------------------------------- #


def _short_e01_arm(name: str) -> str:
    val = name
    for prefix in (
        "propagation:",
        "posresp_extract_",
        "posrespg_extract_",
        "posrespx_extract_",
        "posrespe01b_extract_",
    ):
        if val.startswith(prefix):
            val = val[len(prefix) :]
    for suffix in ("_gpu0", "_gpu1"):
        if val.endswith(suffix):
            val = val[: -len(suffix)]
    mapping = {
        "dplm_650m": "dplm-650m",
        "dplm_3b": "dplm-3b",
        "esm2_650m": "esm2-650m",
        "esm2_3b": "esm2-3b",
        "galactica_125m": "galactica-125m",
        "galactica_1_3b": "galactica-1.3b",
        "galactica_6_7b": "galactica-6.7b",
        "gpt2_large": "gpt2-large",
        "gpt2_xl": "gpt2-xl",
        "instructprotein": "instructprotein",
        "llama_2_7b": "llama-2-7b",
        "progen2_small": "progen2-small",
        "progen2_base": "progen2-base",
        "progen2_medium": "progen2-medium",
        "progen2_large": "progen2-large",
        "progen2_xlarge": "progen2-xlarge",
        "progen3_112m": "progen3-112m",
        "prollama": "prollama",
        "prollama_stage_1": "prollama-stage-1",
        "proteinglm_7b_clm": "proteinglm-7b-clm",
        "protgpt2": "protgpt2",
        "qwen2_5_7b": "qwen2.5-7b",
        "rita_xl": "rita-xl",
    }
    return mapping.get(val, val)


def _build_r3_contacts_section(root: Path) -> dict[str, Any]:
    """Package 22-arm E01 perturbation-propagation panel + 16-arm attention/APC map panel."""
    disc_dir = root / "results/extensions/contact_discrimination_20261120"
    bidir_dir = root / "results/extensions/contact_bidirectional_20261120"

    def _summarize_e01_disc(path: Path) -> dict[str, Any]:
        doc = json.loads(path.read_text(encoding="utf-8"))
        def _pack_cols(cols: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out = []
            for c in cols:
                canonical = _short_e01_arm(c["readout"]) if c["kind"] == "model" else c["readout"]
                meta = ARM_METADATA.get(canonical)
                out.append(
                    {
                        "readout": c["readout"],
                        "arm_id": canonical if c["kind"] == "model" else None,
                        "kind": c["kind"],
                        "paradigm": meta.paradigm if meta else c.get("paradigm"),
                        "readout_mechanism": (
                            "perturbation_propagation_e01" if c["kind"] == "model" else c["kind"]
                        ),
                        "point_auc_minus_half": c["point"],
                        "simultaneous_interval_95": c["interval"],
                        "verdict": c["verdict"],
                        "marginal_interval_95": c["marginal_interval"],
                        "marginal_verdict": c["marginal_verdict"],
                    }
                )
            return out

        fa = doc["full_alphabet_secondary"]
        return {
            "source_path": str(path.relative_to(root)),
            "source_sha256": sha256_file(path),
            "readout_mechanism": "perturbation_propagation_e01",
            "references_admitted": doc["references_admitted"],
            "primary_thin_bpe_inclusive": {
                "n_references": len(doc["references_complete"]),
                "critical_value": doc["native"].get("critical_value"),
                "columns": _pack_cols(doc["columns"]),
            },
            "full_alphabet_secondary": {
                "n_references": len(fa["references_complete"]),
                "critical_value": fa["native"].get("critical_value"),
                "thin_bpe_arms_excluded": fa["thin_bpe_arms_excluded"],
                "columns": _pack_cols(fa["columns"]),
            },
        }

    def _summarize_e01_bidir(path: Path) -> dict[str, Any]:
        doc = json.loads(path.read_text(encoding="utf-8"))

        def _pack_panel(strata: dict[str, Any], arms: list[str], n_refs: int) -> dict[str, Any]:
            out_s: dict[str, Any] = {"n_references": n_refs, "n_arms": len(arms), "strata": {}}
            for st_name in ("contact_ranking", "contact_squared_error", "contact_squared_error_clipped"):
                st = strata[st_name]
                by_key = {(c["rung"], c["arm"], c["direction"]): c for c in st["columns"]}
                pure_by_arm = {c["arm"]: c for c in st["pure_couplings_increment"]["columns"]}
                arm_rows = {}
                for raw_arm in arms:
                    arm_id = _short_e01_arm(raw_arm)
                    top = "nuisance+profile+couplings"
                    fwd = by_key[(top, raw_arm, "model_given_controls")]
                    rev = by_key[(top, raw_arm, "controls_given_model")]
                    pc = pure_by_arm[raw_arm]
                    arm_rows[arm_id] = {
                        "arm_id": arm_id,
                        "raw_arm": raw_arm,
                        "paradigm": ARM_METADATA[arm_id].paradigm if arm_id in ARM_METADATA else None,
                        "readout_mechanism": "perturbation_propagation_e01",
                        "U_geom": st["control_performance"]["nuisance"][raw_arm],
                        "U_geom_prof": st["control_performance"]["nuisance+profile"][raw_arm],
                        "U_geom_prof_couplings": st["control_performance"][top][raw_arm],
                        "M_alone": st["model_alone_performance"][raw_arm],
                        "J_geom_prof_couplings_M": st["joint_performance"][top][raw_arm],
                        "forward_M_given_gpc": {
                            "point": fwd["point"],
                            "interval": fwd["interval"],
                            "verdict": fwd["verdict"],
                            "marginal_interval": fwd["marginal_interval"],
                            "marginal_verdict": fwd["marginal_verdict"],
                        },
                        "reverse_gpc_given_M": {
                            "point": rev["point"],
                            "interval": rev["interval"],
                            "verdict": rev["verdict"],
                            "marginal_interval": rev["marginal_interval"],
                            "marginal_verdict": rev["marginal_verdict"],
                        },
                        "pure_couplings_given_gpM": {
                            "point": pc["point"],
                            "interval": pc["interval"],
                            "verdict": pc["verdict"],
                            "marginal_interval": pc["marginal_interval"],
                            "marginal_verdict": pc["marginal_verdict"],
                        },
                    }
                out_s["strata"][st_name] = {
                    "critical_value": st["native"]["critical_value"],
                    "states_native": st["states"],
                    "states_marginal": st["marginal_states"],
                    "pure_couplings_critical_value": st["pure_couplings_increment"]["native"]["critical_value"],
                    "pure_couplings_states_native": st["pure_couplings_increment"]["states"],
                    "pure_couplings_states_marginal": st["pure_couplings_increment"]["marginal_states"],
                    "arms": arm_rows,
                }
            return out_s

        fa = doc["full_alphabet_secondary"]
        return {
            "source_path": str(path.relative_to(root)),
            "source_sha256": sha256_file(path),
            "readout_mechanism": "perturbation_propagation_e01",
            "primary_thin_bpe_inclusive": _pack_panel(
                doc["strata"], doc["family"]["arms"], len(doc["support"]["banded_families"])
            ),
            "full_alphabet_secondary": _pack_panel(
                fa["strata"], fa["family"]["arms"], len(fa["banded_families"])
            ),
        }

    e01_section = {
        "readout_mechanism": "perturbation_propagation_e01",
        "description": (
            "Corrected 0-based E01 single-substitution perturbation propagation (mean |Delta NLL|) "
            "evaluated against zero-sum APC-Frobenius Potts couplings across 22 admitted arms "
            "(16 full-alphabet + 6 thin-BPE; protgpt2 refused at 6/28 refs)."
        ),
        "discrimination": {
            "all": _summarize_e01_disc(disc_dir / "contact_discrimination.json"),
            "deep_neff_ge_5": _summarize_e01_disc(disc_dir / "contact_discrimination_deep.json"),
            "shallow_neff_lt_5": _summarize_e01_disc(disc_dir / "contact_discrimination_shallow.json"),
        },
        "bidirectional": {
            "all": _summarize_e01_bidir(bidir_dir / "contact_bidirectional.json"),
            "deep_neff_ge_5": _summarize_e01_bidir(bidir_dir / "contact_bidirectional_deep.json"),
            "shallow_neff_lt_5": _summarize_e01_bidir(bidir_dir / "contact_bidirectional_shallow.json"),
        },
    }

    # 16-arm attention/APC contact-map panel from general_plm_summary.json + sibling per-protein AUCs.
    gps_path = root / "results/extensions/general_plm_20261120/general_plm_summary.json"
    gps = json.loads(gps_path.read_text(encoding="utf-8"))
    attn_arms_out: dict[str, Any] = {}
    per_assay_vectors: dict[str, dict[str, float]] = {}
    for row in gps["arms"]:
        arm_id = row["arm"]
        r3_mod = (row.get("modules") or {}).get("r3_pairwise_and_contacts") or {}
        sc = r3_mod.get("structural_contacts")
        if not sc:
            continue
        meta = ARM_METADATA.get(arm_id)
        entry: dict[str, Any] = {
            "arm_id": arm_id,
            "display_name": meta.display_name if meta else arm_id,
            "paradigm": meta.paradigm if meta else row.get("paradigm"),
            "family": meta.family if meta else row.get("family"),
            "params_m": meta.params_m if meta else None,
            "readout_mechanism": "attention_apc_map",
            "dtype_realised": r3_mod.get("dtype_realised"),
            "source": r3_mod.get("source"),
            "point_auc_minus_half": float(sc["mean"]),
            "auc": float(sc["mean"]) + 0.5,
            "marginal_interval_95": [float(sc["ci_low"]), float(sc["ci_high"])],
            "se": float(sc["se"]),
            "n_items": int(sc["n_items"]),
            "n_clusters": int(sc["n_clusters"]),
        }
        # Load per-assay AUC-0.5 vector from source JSON for simultaneous 16-arm paired_bands.
        src_rel = r3_mod.get("source")
        if src_rel:
            src_path = root / src_rel
            if src_path.is_file():
                sdoc = json.loads(src_path.read_text(encoding="utf-8"))
                per_assay = (sdoc.get("structural_contacts") or {}).get("per_assay") or []
                if per_assay:
                    per_assay_vectors[arm_id] = {
                        str(item["assay"]): float(item["auc_minus_half"])
                        for item in per_assay
                        if item.get("auc_minus_half") is not None
                    }
        attn_arms_out[arm_id] = entry

    attn_order = tuple(sorted(attn_arms_out.keys(), key=lambda a: (
        list(UNIFIED_62_SINGLE_SEQUENCE_ARMS).index(a) if a in UNIFIED_62_SINGLE_SEQUENCE_ARMS else 999, a
    )))
    attn_paired = None
    if attn_order and all(a in per_assay_vectors for a in attn_order):
        shared_assays = sorted(set.intersection(*(set(per_assay_vectors[a]) for a in attn_order)))
        if len(shared_assays) >= 8:
            mat = np.column_stack(
                [np.asarray([per_assay_vectors[a][k] for k in shared_assays], dtype=np.float64) for a in attn_order]
            )
            ab = paired_bands_complete(mat)
            for idx, a in enumerate(attn_order):
                attn_arms_out[a]["paired_bands_point_auc_minus_half"] = float(ab["point"][idx])
                attn_arms_out[a]["simultaneous_interval_95"] = ab["interval"][idx]
                attn_arms_out[a]["verdict"] = ab["verdict"][idx]
                attn_arms_out[a]["studentized_interval_95"] = ab["studentized_interval"][idx]
            attn_paired = {
                "n_shared_assays": len(shared_assays),
                "family_size": len(attn_order),
                "native_critical_value": float(ab["critical_value"]),
                "studentized_critical_t": float(ab["studentized_critical_t"]),
                "states_native": bidir.counts(ab["verdict"]),
                "states_marginal": bidir.counts(ab["marginal_verdict"]),
            }

    attn_section = {
        "readout_mechanism": "attention_apc_map",
        "description": (
            "Unsupervised symmetrized APC-corrected multi-head attention contact map AUC-0.5 "
            "across 30 experimental domain structures (distinct readout mechanism from E01 "
            "perturbation propagation; never pooled into one family)."
        ),
        "source_path": str(gps_path.relative_to(root)),
        "source_sha256": sha256_file(gps_path),
        "n_arms": len(attn_order),
        "arms_order": list(attn_order),
        "paired_simultaneous_band": attn_paired,
        "arms": {a: attn_arms_out[a] for a in attn_order},
    }

    return {
        "mechanisms_note": (
            "Two distinct structural-contact readout mechanisms are kept strictly separated "
            "and explicitly tagged: 'perturbation_propagation_e01' (22 arms, mutational "
            "Delta-NLL propagation against Potts couplings on 28/24 domains) and "
            "'attention_apc_map' (16 arms, direct symmetrized APC attention map on 30 domains)."
        ),
        "perturbation_propagation_e01": e01_section,
        "attention_apc_map": attn_section,
    }


def run_r3_unified_yardstick(
    *,
    root: Path = REPO_ROOT,
    out_dir: Path | None = None,
    workers: int = 4,
    force_refit: bool = False,
    skip_repro_gate: bool = False,
) -> dict[str, Any]:
    """Build and write ``r3_epistasis_and_contacts_unified_62.json`` (+ ``.sha256``)."""
    t0 = time.monotonic()
    out_root = Path(out_dir) if out_dir is not None else (root / UNIFIED_OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = out_root / "cells/r2_r3"
    cache_dir.mkdir(parents=True, exist_ok=True)

    repro_gate = (
        {"skipped": True}
        if skip_repro_gate
        else verify_r3_reproduction_gate(root=root, arm="progen2-small")
    )
    print(f"[R3] reproduction gate: {repro_gate}", flush=True)

    epistasis_sec = _build_r3_epistasis_section(
        root, cache_dir, workers=workers, force_refit=force_refit
    )
    contacts_sec = _build_r3_contacts_section(root)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "roster": {
            "n_single_sequence_arms": len(UNIFIED_62_SINGLE_SEQUENCE_ARMS),
            "arms_62": list(UNIFIED_62_SINGLE_SEQUENCE_ARMS),
            "arms_62_with_R_cycle": list(UNIFIED_62_WITH_R_ARMS),
            "metadata": {
                arm_id: {
                    "display_name": ARM_METADATA[arm_id].display_name,
                    "family": ARM_METADATA[arm_id].family,
                    "paradigm": ARM_METADATA[arm_id].paradigm,
                    "modality": ARM_METADATA[arm_id].modality,
                    "params_m": ARM_METADATA[arm_id].params_m,
                    "vocab_type": ARM_METADATA[arm_id].vocab_type,
                    "origin_cohort": ARM_METADATA[arm_id].origin_cohort,
                }
                for arm_id in UNIFIED_62_SINGLE_SEQUENCE_ARMS
            },
        },
        "reproduction_gate": repro_gate,
        "epistasis": epistasis_sec,
        "contacts": contacts_sec,
        "elapsed_seconds": time.monotonic() - t0,
    }

    out_path = out_root / "r3_epistasis_and_contacts_unified_62.json"
    atomic_write_json(out_path, payload)
    digest = sha256_file(out_path)
    (out_root / "r3_epistasis_and_contacts_unified_62.json.sha256").write_text(
        f"{digest}  {out_path.name}\n", encoding="utf-8"
    )
    print(f"[R3] wrote {out_path} (sha256={digest})", flush=True)
    return payload
