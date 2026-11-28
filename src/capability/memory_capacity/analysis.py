"""Statistical estimation and curve-fitting engine for 20-AA PLM parametric memory capacity.

Aggregates completed per-run JSON receipts (``plm_parametric_capacity_run_v1``) to estimate:
  1. Per-scale, per-exposure empirical plateaus and smooth 2-regime CES saturation curves.
  2. Native maximum storage capacity ``c_max`` (bits / non-embedding param and bits / total param)
     via direct ``e = 400`` linear/log-log scaling and exposure-anchored 5-scale (0.5M -> 150M)
     calibration, benchmarked against NLP reference constants.
  3. Exposure-discount curves ``c_eff(e)`` and Hill / Michaelis-Menten half-capacity horizon ``e_50``.
  4. Cell-paired AR vs. MLM storage efficiency, 16-stratum masking-rate decomposition, and
     leave-one-out (LOO) pseudo-entropy inflation audit.
  5. Methodological ablations (weight decay, strict FP32 vs BF16, single cosine vs dedicated horizon).

Pure Python + NumPy + SciPy (zero GPU / PyTorch runtime dependency) with SHA-256 sidecar verification
and strict ``FORBIDDEN_INFRA_RE`` infrastructure-hygiene enforcement.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from src.capability.core.io import sha256_file
from src.capability.memory_theory.io_utils import load_json_file, write_json_with_sha256
from src.capability.unified.master_summary import FORBIDDEN_INFRA_RE

from .spec import (
    EXTENDED_SCALE_ORDER,
    LOG2_20,
    SCALE_ORDER,
    assert_no_forbidden_infra_strings,
    canonicalize_paradigm,
    paradigm_display_name,
)

SUMMARY_SCHEMA = "plm_parametric_capacity_summary_v1"
ACCEPTED_RUN_SCHEMAS = frozenset(
    {"memory_capacity_run_v1", "plm_parametric_capacity_run_v1"}
)
COMPLETED_STATUSES = frozenset({"completed", "complete"})
PRIMARY_TRACKS = frozenset({"primary", "stage1_wsd"})

NLP_REFERENCE_CONSTANTS_BITS_PER_PARAM: dict[str, dict[str, Any]] = {
    "morris_et_al_2025_gpt": {
        "bits_per_param": 3.6,
        "citation": "Morris et al. (2025), uniform token-string capacity ~3.6 bits/param",
    },
    "allen_zhu_li_2024": {
        "bits_per_param": 2.0,
        "citation": "Allen-Zhu & Li (2024), Physics of Language Models ~2.0 bits/param",
    },
}


def _scale_sort_key(scale_id: str) -> tuple[int, str]:
    if scale_id in EXTENDED_SCALE_ORDER:
        return (EXTENDED_SCALE_ORDER.index(scale_id), scale_id)
    return (len(EXTENDED_SCALE_ORDER), scale_id)


def _mult_key(multiplier: float) -> str:
    return f"{float(multiplier):g}x"


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-30:
        return 1.0 if ss_res <= 1e-30 else 0.0
    val = 1.0 - ss_res / ss_tot
    return float(val) if math.isfinite(val) else 0.0


def _verify_sidecar_if_present(path: Path) -> bool:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        return False
    tokens = sidecar.read_text(encoding="utf-8").strip().split()
    if not tokens:
        raise ValueError(f"Empty SHA-256 sidecar file: {sidecar}")
    declared = tokens[0]
    actual = sha256_file(path)
    if actual != declared:
        raise ValueError(
            f"SHA-256 sidecar mismatch for {path}: declared {declared}, computed {actual}"
        )
    return True


def _bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_draws: int,
    seed: int,
) -> dict[str, Any] | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return None
    point = float(np.mean(arr))
    if n == 1 or n_draws <= 0:
        return {
            "point": point,
            "se": 0.0,
            "ci95": [point, point],
            "n": n,
            "n_draws": 0,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_draws, n))
    draws = arr[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "point": point,
        "se": float(np.std(draws, ddof=1)),
        "ci95": [float(lo), float(hi)],
        "n": n,
        "n_draws": int(n_draws),
    }


def _bootstrap_ratio_of_sums(
    numerator: Sequence[float] | np.ndarray,
    denominator: Sequence[float] | np.ndarray,
    *,
    n_draws: int,
    seed: int,
) -> dict[str, Any] | None:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    if num.size == 0 or num.shape != den.shape:
        return None
    den_sum = float(np.sum(den))
    if not math.isfinite(den_sum) or abs(den_sum) <= 1e-12:
        return None
    point = float(np.sum(num) / den_sum)
    n = int(num.size)
    if n == 1 or n_draws <= 0:
        return {
            "point": point,
            "se": 0.0,
            "ci95": [point, point],
            "n": n,
            "n_draws": 0,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_draws, n))
    b_num = num[idx].sum(axis=1)
    b_den = den[idx].sum(axis=1)
    valid = np.abs(b_den) > 1e-12
    if not np.any(valid):
        return {
            "point": point,
            "se": 0.0,
            "ci95": [point, point],
            "n": n,
            "n_draws": 0,
        }
    ratios = b_num[valid] / b_den[valid]
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    se = float(np.std(ratios, ddof=1)) if ratios.size > 1 else 0.0
    return {
        "point": point,
        "se": se,
        "ci95": [float(lo), float(hi)],
        "n": n,
        "n_draws": int(ratios.size),
    }


def _ces_memorized_norm(h_norm: np.ndarray, c_ces: float, gamma: float) -> np.ndarray:
    """Numerically stable normalized CES saturation curve:
    m_hat(h) = c / (1 + (c / h)^gamma)^(1/gamma) = exp(log(c) - logaddexp(0, gamma*(log(c)-log(h)))/gamma).
    """
    log_c = math.log(max(c_ces, 1e-15))
    log_h = np.log(np.maximum(h_norm, 1e-15))
    denom_log = np.logaddexp(0.0, gamma * (log_c - log_h)) / gamma
    return np.exp(log_c - denom_log)


def fit_ces_saturation_curve(
    h_bits: Sequence[float] | np.ndarray,
    m_bits: Sequence[float] | np.ndarray,
    *,
    n_params_non_emb: int,
) -> dict[str, Any] | None:
    """Fit M_hat(H) = C / (1 + (C / H)^gamma)^(1/gamma) with C > 0 and gamma in [0.5, 20.0]."""
    h_arr = np.asarray(h_bits, dtype=np.float64)
    m_arr = np.asarray(m_bits, dtype=np.float64)
    if h_arr.size < 4 or h_arr.shape != m_arr.shape or n_params_non_emb <= 0:
        return None
    if np.any(h_arr <= 0.0) or not (np.all(np.isfinite(h_arr)) and np.all(np.isfinite(m_arr))):
        return None
    if float(np.max(m_arr)) <= 0.0:
        return None

    scale = float(n_params_non_emb)
    h_norm = h_arr / scale
    m_norm = m_arr / scale

    c_upper = max(50.0, float(np.max(h_norm)) * 4.0, float(np.max(m_norm)) * 4.0)
    c_init = float(np.clip(np.max(m_norm), 1e-4, c_upper * 0.5))

    best_res = None
    for gamma_init in (1.0, 2.5, 8.0):
        res = least_squares(
            fun=lambda theta: _ces_memorized_norm(h_norm, float(theta[0]), float(theta[1])) - m_norm,
            x0=np.array([c_init, gamma_init], dtype=np.float64),
            bounds=(np.array([1e-6, 0.5]), np.array([c_upper, 20.0])),
            method="trf",
            max_nfev=2000,
        )
        if best_res is None or res.cost < best_res.cost:
            best_res = res

    assert best_res is not None
    c_ces = float(best_res.x[0])
    gamma = float(best_res.x[1])
    c_bits = c_ces * scale
    m_pred = _ces_memorized_norm(h_norm, c_ces, gamma) * scale
    rmse_bits = float(np.sqrt(np.mean((m_arr - m_pred) ** 2)))
    return {
        "C_asymptote_bits": c_bits,
        "c_eff_ces_bits_per_non_emb_param": c_ces,
        "gamma_sharpness": gamma,
        "r2": _safe_r2(m_arr, m_pred),
        "rmse_bits": rmse_bits,
        "n_multipliers": int(h_arr.size),
        "converged": bool(best_res.success),
    }


def fit_hill_exposure_curve(
    exposures: Sequence[float] | np.ndarray,
    c_eff_values: Sequence[float] | np.ndarray,
) -> dict[str, Any] | None:
    """Fit c_eff(e) = c_inf * (e^alpha / (e^alpha + e_50^alpha)) for e > 0."""
    e_arr = np.asarray(exposures, dtype=np.float64)
    y_arr = np.asarray(c_eff_values, dtype=np.float64)
    if e_arr.size < 4 or e_arr.shape != y_arr.shape:
        return None
    if np.any(e_arr <= 0.0) or not (np.all(np.isfinite(e_arr)) and np.all(np.isfinite(y_arr))):
        return None
    y_max = float(np.max(y_arr))
    if y_max <= 0.0:
        return None

    log_e = np.log(e_arr)

    def _pred(c_inf: float, e_50: float, alpha: float) -> np.ndarray:
        return c_inf * expit(alpha * (log_e - math.log(max(e_50, 1e-9))))

    c_upper = max(20.0, y_max * 5.0)
    e_upper = max(5000.0, float(np.max(e_arr)) * 10.0)
    c_init = float(np.clip(y_max * 1.05, 1e-5, c_upper * 0.8))

    best_res = None
    for e50_init in (5.0, 20.0, 80.0):
        for alpha_init in (0.7, 1.2):
            res = least_squares(
                fun=lambda th: _pred(float(th[0]), float(th[1]), float(th[2])) - y_arr,
                x0=np.array([c_init, min(e50_init, e_upper * 0.5), alpha_init], dtype=np.float64),
                bounds=(np.array([1e-6, 1e-3, 0.1]), np.array([c_upper, e_upper, 10.0])),
                method="trf",
                max_nfev=3000,
            )
            if best_res is None or res.cost < best_res.cost:
                best_res = res

    assert best_res is not None
    c_inf = float(best_res.x[0])
    e_50 = float(best_res.x[1])
    alpha = float(best_res.x[2])
    e_90 = float(e_50 * (9.0 ** (1.0 / alpha)))
    e_95 = float(e_50 * (19.0 ** (1.0 / alpha)))
    y_pred = _pred(c_inf, e_50, alpha)
    return {
        "c_inf_bits_per_param": c_inf,
        "e_50_epochs": e_50,
        "e_90_epochs": e_90,
        "e_95_epochs": e_95,
        "alpha_hill": alpha,
        "r2": _safe_r2(y_arr, y_pred),
        "rmse_bits_per_param": float(np.sqrt(np.mean((y_arr - y_pred) ** 2))),
        "n_exposures": int(e_arr.size),
        "converged": bool(best_res.success),
    }


def estimate_ecrit_nonparametric(
    exposures: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    threshold: float,
) -> float | None:
    """Estimate first threshold crossing e_crit via log-linear interpolation in exposure space."""
    e_arr = np.asarray(exposures, dtype=np.float64)
    y_arr = np.asarray(values, dtype=np.float64)
    if e_arr.size == 0 or e_arr.shape != y_arr.shape or not math.isfinite(float(threshold)):
        return None
    valid = np.isfinite(e_arr) & np.isfinite(y_arr) & (e_arr > 0.0)
    if not np.any(valid):
        return None
    e_arr = e_arr[valid]
    y_arr = y_arr[valid]
    order = np.argsort(e_arr)
    e_arr = e_arr[order]
    y_arr = y_arr[order]

    thresh = float(threshold)
    above = np.nonzero(y_arr >= thresh)[0]
    if above.size == 0:
        return None
    idx = int(above[0])
    if idx == 0:
        return float(e_arr[0])

    e0, e1 = float(e_arr[idx - 1]), float(e_arr[idx])
    y0, y1 = float(y_arr[idx - 1]), float(y_arr[idx])
    if y1 <= y0 + 1e-15:
        return float(e1)
    frac = min(max((thresh - y0) / (y1 - y0), 0.0), 1.0)
    log_e = math.log(e0) + frac * (math.log(e1) - math.log(e0))
    return float(math.exp(log_e))


def fit_delayed_hill_exposure_curve(
    exposures: Sequence[float] | np.ndarray,
    c_eff_values: Sequence[float] | np.ndarray,
    *,
    min_signal_bpp: float = 0.02,
    min_alpha: float = 0.6,
    max_c_inf_ratio: float = 1.35,
    max_c_inf_physical: float = 3.8,
) -> dict[str, Any] | None:
    """Fit delayed-Hill write dynamics:
    c_eff(e) = c_inf * ((e - e_crit)^alpha / ((e - e_crit)^alpha + tau^alpha)) for e > e_crit,
    else 0, where e_50 = e_crit + tau and e_90 = e_crit + tau * 9^(1/alpha).
    Returns None when the trajectory never reaches ``min_signal_bpp`` (pre-takeoff noise).
    Bounds ``alpha >= min_alpha`` and ``c_inf <= min(max_c_inf_physical, y_max * max_c_inf_ratio)``
    (unless ``y_max`` already exceeds ``max_c_inf_physical``) to prevent unphysical runaway
    extrapolation from terminal WSD cooldown jumps.
    """
    e_arr = np.asarray(exposures, dtype=np.float64)
    y_arr = np.asarray(c_eff_values, dtype=np.float64)
    if e_arr.size < 4 or e_arr.shape != y_arr.shape:
        return None
    if np.any(e_arr <= 0.0) or not (np.all(np.isfinite(e_arr)) and np.all(np.isfinite(y_arr))):
        return None
    order = np.argsort(e_arr)
    e_arr = e_arr[order]
    y_arr = y_arr[order]
    y_max = float(np.max(y_arr))
    if y_max < float(min_signal_bpp):
        return None

    def _pred(c_inf: float, e_crit: float, tau: float, alpha: float) -> np.ndarray:
        shifted = e_arr - e_crit
        out = np.zeros_like(e_arr, dtype=np.float64)
        pos = shifted > 0.0
        if np.any(pos):
            out[pos] = c_inf * expit(alpha * (np.log(shifted[pos]) - math.log(max(tau, 1e-9))))
        return out

    e_max = float(np.max(e_arr))
    phys_cap = max(float(max_c_inf_physical), y_max * 1.05)
    c_upper = min(phys_cap, max(y_max * float(max_c_inf_ratio), y_max + 0.05))
    e_crit_upper = max(1e-6, e_max * 0.95)
    tau_upper = max(5000.0, e_max * 10.0)
    c_init = float(np.clip(y_max * 1.05, 1e-5, c_upper * 0.95))

    emp_ecrit = estimate_ecrit_nonparametric(e_arr, y_arr, 0.05 * y_max)
    emp_e50 = estimate_ecrit_nonparametric(e_arr, y_arr, 0.50 * y_max)

    ecrit_candidates: list[float] = [0.0]
    if emp_ecrit is not None and emp_ecrit > 0.0:
        ecrit_candidates.extend([min(e_crit_upper * 0.9, emp_ecrit * 0.5), min(e_crit_upper * 0.9, emp_ecrit * 0.9)])
    ecrit_candidates = sorted({round(v, 6) for v in ecrit_candidates})

    tau_candidates: list[float] = [5.0, 30.0, 150.0]
    if emp_e50 is not None:
        base_ec = emp_ecrit if emp_ecrit is not None else 0.0
        tau_candidates.append(max(1.0, emp_e50 - base_ec))

    alpha_lb = float(min_alpha)
    best_res = None
    for ec_init in ecrit_candidates:
        for tau_init in tau_candidates:
            for alpha_init in (max(0.8, alpha_lb + 0.05), 1.5, 3.0):
                x0 = np.array(
                    [
                        c_init,
                        min(max(ec_init, 0.0), e_crit_upper * 0.99),
                        min(max(tau_init, 1e-2), tau_upper * 0.5),
                        alpha_init,
                    ],
                    dtype=np.float64,
                )
                res = least_squares(
                    fun=lambda th: _pred(float(th[0]), float(th[1]), float(th[2]), float(th[3])) - y_arr,
                    x0=x0,
                    bounds=(
                        np.array([1e-6, 0.0, 1e-3, alpha_lb], dtype=np.float64),
                        np.array([c_upper, e_crit_upper, tau_upper, 12.0], dtype=np.float64),
                    ),
                    method="trf",
                    max_nfev=3000,
                )
                if best_res is None or res.cost < best_res.cost:
                    best_res = res

    assert best_res is not None
    c_inf = float(best_res.x[0])
    e_crit = float(best_res.x[1])
    tau = float(best_res.x[2])
    alpha = float(best_res.x[3])
    e_05 = float(e_crit + tau * ((1.0 / 19.0) ** (1.0 / alpha)))
    e_50 = float(e_crit + tau)
    e_90 = float(e_crit + tau * (9.0 ** (1.0 / alpha)))
    e_95 = float(e_crit + tau * (19.0 ** (1.0 / alpha)))
    y_pred = _pred(c_inf, e_crit, tau, alpha)
    return {
        "c_inf_bits_per_param": c_inf,
        "e_crit_epochs": e_crit,
        "tau_half_rise_epochs": tau,
        "e_05_epochs": e_05,
        "e_50_epochs": e_50,
        "e_90_epochs": e_90,
        "e_95_epochs": e_95,
        "alpha_hill": alpha,
        "r2": _safe_r2(y_arr, y_pred),
        "rmse_bits_per_param": float(np.sqrt(np.mean((y_arr - y_pred) ** 2))),
        "n_exposures": int(e_arr.size),
        "converged": bool(best_res.success),
    }


def _extract_cell_summary(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all multiplier observations at a fixed (paradigm, scale_id, epoch)."""
    ordered = sorted(obs_list, key=lambda r: float(r["multiplier"]))
    p_non_emb = int(ordered[0]["n_params_non_embedding"])
    p_total = int(ordered[0]["n_params_total"])

    ge2 = [r for r in ordered if float(r["multiplier"]) >= 2.0 - 1e-9]
    ge1 = [r for r in ordered if float(r["multiplier"]) >= 1.0 - 1e-9]
    sat_ge2 = ge2 if ge2 else (ge1 if ge1 else ordered)
    sat_ge1 = ge1 if ge1 else ordered

    max_ge2_row = max(sat_ge2, key=lambda r: float(r["memorized_bits"]))
    max_ge2_bits = float(max_ge2_row["memorized_bits"])
    mean_ge2_bits = float(np.mean([float(r["memorized_bits"]) for r in sat_ge2]))
    plateau_max_ge1_bits = float(max(float(r["memorized_bits"]) for r in sat_ge1))

    at_2x_candidates = [r for r in ordered if abs(float(r["multiplier"]) - 2.0) <= 1e-6]
    at_2x_row = at_2x_candidates[0] if at_2x_candidates else max_ge2_row
    at_2x_bits = float(at_2x_row["memorized_bits"])

    mean_tok_xent_emp = float(max_ge2_row["mean_tok_xent_bits"])
    mean_tok_xent_2x = float(at_2x_row["mean_tok_xent_bits"])
    vocab_sz = int(ordered[0].get("vocab_size", 20) or 20)
    log2_v = math.log2(float(max(2, vocab_sz)))
    delta_bpt_emp = float(log2_v - mean_tok_xent_emp)
    delta_bpt_2x = float(log2_v - mean_tok_xent_2x)

    # Dual Slepian-Wolf / symmetric-slot pseudo capacity metrics (populated for mlm_15)
    has_sw = all(r.get("sw_memorized_bits") is not None for r in sat_ge2)
    sw_max_ge2_bits = (
        float(max(float(r["sw_memorized_bits"]) for r in sat_ge2)) if has_sw else None
    )
    sw_at_2x_bits = (
        float(at_2x_row["sw_memorized_bits"])
        if at_2x_row.get("sw_memorized_bits") is not None
        else sw_max_ge2_bits
    )
    has_pseudo = all(r.get("pseudo_memorized_bits") is not None for r in sat_ge2)
    pseudo_max_ge2_bits = (
        float(max(float(r["pseudo_memorized_bits"]) for r in sat_ge2)) if has_pseudo else None
    )
    pseudo_at_2x_bits = (
        float(at_2x_row["pseudo_memorized_bits"])
        if at_2x_row.get("pseudo_memorized_bits") is not None
        else pseudo_max_ge2_bits
    )

    c_eff_at_mult = {
        _mult_key(float(r["multiplier"])): float(r["bits_per_non_emb_param"]) for r in ordered
    }
    points = [
        {
            "multiplier": float(r["multiplier"]),
            "n_sequences": int(r["n_sequences"]),
            "H_data_bits": float(r["H_data_bits"]),
            "H_gibbs_max_bits": float(r["H_gibbs_max_bits"]),
            "mean_tok_xent_bits": float(r["mean_tok_xent_bits"]),
            "delta_bpt_vs_null": float(log2_v - float(r["mean_tok_xent_bits"])),
            "memorized_bits": float(r["memorized_bits"]),
            "memorized_bits_se": float(r["memorized_bits_se"]),
            "bits_per_non_emb_param": float(r["bits_per_non_emb_param"]),
            "bits_per_total_param": float(r["bits_per_total_param"]),
            "sw_bits_per_non_emb_param": (
                float(r["sw_bits_per_non_emb_param"])
                if r.get("sw_bits_per_non_emb_param") is not None
                else None
            ),
            "pseudo_bits_per_non_emb_param": (
                float(r["pseudo_bits_per_non_emb_param"])
                if r.get("pseudo_bits_per_non_emb_param") is not None
                else None
            ),
            "fraction_of_H_data": float(r["fraction_of_H_data"]),
            "fraction_of_H_gibbs": float(r["fraction_of_H_gibbs"]),
        }
        for r in ordered
    ]

    ces_fit = None
    if len(ordered) >= 4:
        ces_fit = fit_ces_saturation_curve(
            [float(r["H_data_bits"]) for r in ordered],
            [float(r["memorized_bits"]) for r in ordered],
            n_params_non_emb=p_non_emb,
        )

    return {
        "n_multipliers": len(ordered),
        "multipliers": [float(r["multiplier"]) for r in ordered],
        "n_params_non_embedding": p_non_emb,
        "n_params_total": p_total,
        "primary_estimator": "max_over_ge2x" if ge2 else "max_available_fallback",
        "has_ge2x_multiplier": bool(ge2),
        "has_exact_2x_multiplier": bool(at_2x_candidates),
        "mean_tok_xent_bits_empirical": mean_tok_xent_emp,
        "mean_tok_xent_bits_at_2x": mean_tok_xent_2x,
        "delta_bpt_empirical": delta_bpt_emp,
        "delta_bpt_at_2x": delta_bpt_2x,
        "plateau_empirical_bits": max_ge2_bits,
        "c_eff_empirical": max_ge2_bits / p_non_emb,
        "c_eff_total_empirical": max_ge2_bits / p_total,
        "sw_memorized_empirical_bits": sw_max_ge2_bits,
        "sw_bits_per_non_emb_param": (
            sw_max_ge2_bits / p_non_emb if sw_max_ge2_bits is not None else None
        ),
        "sw_bits_per_non_emb_param_at_2x": (
            sw_at_2x_bits / p_non_emb if sw_at_2x_bits is not None else None
        ),
        "pseudo_memorized_empirical_bits": pseudo_max_ge2_bits,
        "pseudo_bits_per_non_emb_param": (
            pseudo_max_ge2_bits / p_non_emb if pseudo_max_ge2_bits is not None else None
        ),
        "pseudo_bits_per_non_emb_param_at_2x": (
            pseudo_at_2x_bits / p_non_emb if pseudo_at_2x_bits is not None else None
        ),
        "max_over_ge2x_bits": max_ge2_bits,
        "c_eff_max_over_ge2x": max_ge2_bits / p_non_emb,
        "mean_over_ge2x_bits": mean_ge2_bits,
        "c_eff_mean_over_ge2x": mean_ge2_bits / p_non_emb,
        "mean_ge2x_bits": mean_ge2_bits,
        "c_eff_mean_ge2x": mean_ge2_bits / p_non_emb,
        "plateau_max_bits": plateau_max_ge1_bits,
        "c_eff_plateau_max": plateau_max_ge1_bits / p_non_emb,
        "c_eff_total_plateau_max": plateau_max_ge1_bits / p_total,
        "memorized_at_2x_bits": at_2x_bits,
        "c_eff_at_2x": at_2x_bits / p_non_emb,
        "c_eff_total_at_2x": at_2x_bits / p_total,
        "c_eff_at_multiplier": c_eff_at_mult,
        "ces_saturation_fit": ces_fit,
        "points": points,
    }


def _fit_linear_capacity_scaling(
    scale_ids: Sequence[str],
    params: Sequence[float],
    memorized_bits: Sequence[float],
    *,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any] | None:
    p_arr = np.asarray(params, dtype=np.float64)
    m_arr = np.asarray(memorized_bits, dtype=np.float64)
    n = int(p_arr.size)
    if n == 0 or p_arr.shape != m_arr.shape or np.any(p_arr <= 0.0):
        return None

    ratio_boot = _bootstrap_ratio_of_sums(m_arr, p_arr, n_draws=bootstrap_draws, seed=seed)
    assert ratio_boot is not None
    c_ratio = float(ratio_boot["point"])
    r2_zero_int = _safe_r2(m_arr, c_ratio * p_arr)

    per_scale_c = m_arr / p_arr
    mean_c_boot = _bootstrap_mean_ci(per_scale_c, n_draws=bootstrap_draws, seed=seed + 11)

    ols_block: dict[str, Any] | None = None
    loglog_block: dict[str, Any] | None = None

    if n >= 2 and float(np.ptp(p_arr)) > 0.0:
        p_mean = float(np.mean(p_arr))
        m_mean = float(np.mean(m_arr))
        dp = p_arr - p_mean
        dm = m_arr - m_mean
        slope_ols = float(np.sum(dp * dm) / np.sum(dp**2))
        intercept_ols = float(m_mean - slope_ols * p_mean)
        r2_ols = _safe_r2(m_arr, slope_ols * p_arr + intercept_ols)

        rng = np.random.default_rng(seed + 23)
        ols_slopes: list[float] = []
        ols_intercepts: list[float] = []
        if bootstrap_draws > 0:
            idx = rng.integers(0, n, size=(bootstrap_draws, n))
            for row in idx:
                pb = p_arr[row]
                if float(np.ptp(pb)) <= 0.0:
                    continue
                mb = m_arr[row]
                dpb = pb - np.mean(pb)
                dmb = mb - np.mean(mb)
                sb = float(np.sum(dpb * dmb) / np.sum(dpb**2))
                ib = float(np.mean(mb) - sb * np.mean(pb))
                ols_slopes.append(sb)
                ols_intercepts.append(ib)

        if len(ols_slopes) >= 2:
            s_lo, s_hi = np.percentile(ols_slopes, [2.5, 97.5])
            s_se = float(np.std(ols_slopes, ddof=1))
            i_lo, i_hi = np.percentile(ols_intercepts, [2.5, 97.5])
        else:
            s_lo = s_hi = slope_ols
            s_se = 0.0
            i_lo = i_hi = intercept_ols

        ols_block = {
            "c_max_slope": slope_ols,
            "c_max_se": s_se,
            "c_max_ci95": [float(s_lo), float(s_hi)],
            "intercept_bits": intercept_ols,
            "intercept_ci95": [float(i_lo), float(i_hi)],
            "r2": r2_ols,
            "n_valid_bootstrap_draws": len(ols_slopes),
        }

        if np.all(m_arr > 0.0):
            lp = np.log10(p_arr)
            lm = np.log10(m_arr)
            dlp = lp - np.mean(lp)
            dlm = lm - np.mean(lm)
            alpha = float(np.sum(dlp * dlm) / np.sum(dlp**2))
            beta = float(np.mean(lm) - alpha * np.mean(lp))
            r2_log = _safe_r2(lm, alpha * lp + beta)

            rng_log = np.random.default_rng(seed + 37)
            alphas: list[float] = []
            if bootstrap_draws > 0:
                idx_log = rng_log.integers(0, n, size=(bootstrap_draws, n))
                for row in idx_log:
                    lpb = lp[row]
                    if float(np.ptp(lpb)) <= 0.0:
                        continue
                    lmb = lm[row]
                    dlpb = lpb - np.mean(lpb)
                    dlmb = lmb - np.mean(lmb)
                    alphas.append(float(np.sum(dlpb * dlmb) / np.sum(dlpb**2)))

            if len(alphas) >= 2:
                a_lo, a_hi = np.percentile(alphas, [2.5, 97.5])
                a_se = float(np.std(alphas, ddof=1))
            else:
                a_lo = a_hi = alpha
                a_se = 0.0

            loglog_block = {
                "alpha_exponent": alpha,
                "alpha_se": a_se,
                "alpha_ci95": [float(a_lo), float(a_hi)],
                "beta_log10_intercept": beta,
                "deviation_from_unit_linearity": float(alpha - 1.0),
                "r2_log10": r2_log,
                "n_valid_bootstrap_draws": len(alphas),
            }

    return {
        "n_scales": n,
        "scale_ids": list(scale_ids),
        "scale_span_ratio": float(np.max(p_arr) / np.min(p_arr)),
        "zero_intercept_ratio": {
            "c_max": c_ratio,
            "se": float(ratio_boot["se"]),
            "ci95": list(ratio_boot["ci95"]),
            "r2": r2_zero_int,
        },
        "mean_across_scales": mean_c_boot,
        "per_scale_bits_per_param": {
            str(s): float(c) for s, c in zip(scale_ids, per_scale_c)
        },
        "ols": ols_block,
        "log_log_scaling": loglog_block,
    }


def _nlp_comparison_for_cmax(c_max: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ref_key, spec in NLP_REFERENCE_CONSTANTS_BITS_PER_PARAM.items():
        ref_val = float(spec["bits_per_param"])
        out[ref_key] = {
            "reference_bits_per_param": ref_val,
            "citation": spec["citation"],
            "plm_c_max": float(c_max),
            "ratio_plm_over_nlp": float(c_max / ref_val),
            "diff_plm_minus_nlp": float(c_max - ref_val),
        }
    return out


def _parse_quadrature_profile(raw: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if raw is None:
        return None
    if isinstance(raw, list) and len(raw) > 0:
        if isinstance(raw[0], (int, float)):
            vals = np.asarray([float(x) for x in raw], dtype=np.float64)
            q = vals.size
            rates = (np.arange(q, dtype=np.float64) + 0.5) / float(q)
            return rates, vals
        if isinstance(raw[0], dict):
            pairs: list[tuple[float, float]] = []
            for idx, item in enumerate(raw):
                rate = float(item.get("mask_rate", (idx + 0.5) / len(raw)))
                bpt = item.get(
                    "bits_per_tok",
                    item.get("mean_tok_xent_bits", item.get("xent_bits_per_tok")),
                )
                if bpt is None:
                    return None
                pairs.append((rate, float(bpt)))
            pairs.sort(key=lambda t: t[0])
            return (
                np.asarray([p[0] for p in pairs], dtype=np.float64),
                np.asarray([p[1] for p in pairs], dtype=np.float64),
            )
    if isinstance(raw, dict) and len(raw) > 0:
        pairs = []
        for k, v in raw.items():
            try:
                rate = float(k)
            except ValueError:
                continue
            pairs.append((rate, float(v)))
        if pairs:
            pairs.sort(key=lambda t: t[0])
            return (
                np.asarray([p[0] for p in pairs], dtype=np.float64),
                np.asarray([p[1] for p in pairs], dtype=np.float64),
            )
    return None


def _normalize_run_receipt(doc: dict[str, Any], source_label: str) -> dict[str, Any]:
    """Normalize both nested ('memory_capacity_run_v1') and flat ('plm_parametric_capacity_run_v1') receipts."""
    assert_no_forbidden_infra_strings(doc, label=source_label)
    raw_text = json.dumps(doc, sort_keys=True)
    m = FORBIDDEN_INFRA_RE.search(raw_text)
    if m is not None:
        raise ValueError(
            f"{source_label} contains forbidden infra identifier: {m.group(0)!r}"
        )

    if "exposures" in doc and "H_data_bits" in doc:
        norm = dict(doc)
        if "paradigm" in norm:
            norm["paradigm"] = canonicalize_paradigm(str(norm["paradigm"]))
        return norm

    if "run_spec" in doc and "dataset_info" in doc:
        rs = doc["run_spec"]
        pc = doc["parameter_counts"]
        di = doc["dataset_info"]
        canon_p = canonicalize_paradigm(str(rs["paradigm"]))
        raw_evals = doc.get("evaluations") or []
        norm_evals: list[dict[str, Any]] = []
        for ev in raw_evals:
            norm_evals.append(
                {
                    "epoch": int(ev["epoch"]),
                    "step": int(ev.get("global_step", ev.get("step", 0))),
                    "tokens_seen": int(ev.get("tokens_seen", 0)),
                    "lr": float(ev.get("lr", 0.0)),
                    "elapsed_sec": float(ev.get("eval_wall_time_sec", ev.get("elapsed_sec", 0.0))),
                    "train_loss_last_epoch_bits_per_tok": float(
                        ev.get("train_loss_bits", ev.get("train_loss_last_epoch_bits_per_tok", 0.0))
                    ),
                    "eval_n_sequences": int(
                        ev.get("n_sequences_evaluated", ev.get("eval_n_sequences", rs["n_sequences"]))
                    ),
                    "mean_seq_xent_bits": float(ev["mean_seq_xent_bits"]),
                    "mean_tok_xent_bits": float(ev["mean_tok_xent_bits"]),
                    "eval_mc_se_seq_bits": float(ev.get("eval_mc_se_seq_bits", 0.0)),
                    "total_model_xent_bits": float(
                        ev.get("total_dataset_xent_bits", ev.get("total_model_xent_bits", 0.0))
                    ),
                    "memorized_bits": float(ev["memorized_bits"]),
                    "memorized_bits_se": float(
                        ev.get("eval_mc_se_total_bits", ev.get("memorized_bits_se", 0.0))
                    ),
                    "bits_per_non_emb_param": float(
                        ev.get("memorized_bits_per_param", ev.get("bits_per_non_emb_param"))
                    ),
                    "bits_per_total_param": float(
                        ev.get("memorized_bits_per_total_param", ev.get("bits_per_total_param"))
                    ),
                    "sw_memorized_bits": ev.get("sw_memorized_bits"),
                    "sw_memorized_bits_se": ev.get("sw_memorized_bits_se"),
                    "sw_bits_per_non_emb_param": ev.get("sw_bits_per_non_emb_param"),
                    "sw_bits_per_total_param": ev.get("sw_bits_per_total_param"),
                    "pseudo_memorized_bits": ev.get("pseudo_memorized_bits"),
                    "pseudo_memorized_bits_se": ev.get("pseudo_memorized_bits_se"),
                    "pseudo_bits_per_non_emb_param": ev.get("pseudo_bits_per_non_emb_param"),
                    "pseudo_bits_per_total_param": ev.get("pseudo_bits_per_total_param"),
                    "mean_masked_token_xent_bits": ev.get("mean_masked_token_xent_bits"),
                    "fraction_of_H_data": float(
                        ev.get("fraction_of_H_dataset", ev.get("fraction_of_H_data", 0.0))
                    ),
                    "fraction_of_H_gibbs": float(ev.get("fraction_of_H_gibbs", 0.0)),
                    "mlm_quadrature_16pt_bits_per_tok": ev.get(
                        "mlm_quadrature_16pt_bits_per_tok"
                    ),
                    "mlm_loo_pseudo_tok_xent_bits": ev.get("mlm_loo_pseudo_tok_xent_bits"),
                    "mlm_loo_pseudo_memorized_bits": ev.get("mlm_loo_pseudo_memorized_bits"),
                    "mlm_fixed_15pct_diagnostic": ev.get("mlm_fixed_15pct_diagnostic"),
                    "mlm_oa_16pt_diagnostic": ev.get("mlm_oa_16pt_diagnostic"),
                }
            )
        return {
            "run_id": str(rs["run_id"]),
            "track": str(rs["track"]),
            "paradigm": canon_p,
            "scale_id": str(rs["scale_id"]),
            "multiplier": float(rs["multiplier"]),
            "n_sequences": int(rs["n_sequences"]),
            "seq_len": int(di["seq_len"]),
            "vocab_size": int(di["vocab_size"]),
            "n_params_non_embedding": int(pc["n_params_non_embedding"]),
            "n_params_embedding": int(pc["n_params_embedding"]),
            "n_params_total": int(pc["n_params_total"]),
            "weight_decay": float(rs["weight_decay"]),
            "precision": str(rs["precision"]),
            "max_epochs": int(rs["max_epochs"]),
            "seed": int(rs["seed"]),
            "data_seed": int(rs["data_seed"]),
            "H_null_ordered_bits": float(di["null_ordered_entropy_bits"]),
            "H_data_bits": float(di["dataset_entropy_bits"]),
            "H_gibbs_max_bits": float(di["gibbs_memorizable_bits"]),
            "exposures": norm_evals,
        }

    norm_fallback = dict(doc)
    if "paradigm" in norm_fallback:
        norm_fallback["paradigm"] = canonicalize_paradigm(str(norm_fallback["paradigm"]))
    return norm_fallback


def _load_completed_runs(runs_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = Path(runs_dir)
    if not base.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {base}")

    candidate_dir = base
    direct_jsons = sorted(base.glob("*.json"))
    if not direct_jsons and (base / "runs").is_dir():
        candidate_dir = base / "runs"
        direct_jsons = sorted(candidate_dir.glob("*.json"))

    completed: list[dict[str, Any]] = []
    n_sidecars_verified = 0
    n_non_complete = 0
    n_ignored_other_schema = 0

    def _scan_files(files: Sequence[Path]) -> None:
        nonlocal n_sidecars_verified, n_non_complete, n_ignored_other_schema
        for jpath in files:
            sidecar_ok = _verify_sidecar_if_present(jpath)
            doc = load_json_file(jpath)
            if not isinstance(doc, dict) or doc.get("schema") not in ACCEPTED_RUN_SCHEMAS:
                n_ignored_other_schema += 1
                continue
            if sidecar_ok:
                n_sidecars_verified += 1
            if doc.get("status") not in COMPLETED_STATUSES:
                n_non_complete += 1
                continue
            completed.append(_normalize_run_receipt(doc, jpath.name))

    _scan_files(direct_jsons)
    if not completed and (base / "runs").is_dir() and candidate_dir == base:
        candidate_dir = base / "runs"
        _scan_files(sorted(candidate_dir.glob("*.json")))

    provenance = {
        "resolved_runs_dir": candidate_dir.name,
        "n_completed_runs": len(completed),
        "n_sidecars_verified": n_sidecars_verified,
        "n_incomplete_runs_skipped": n_non_complete,
        "n_non_run_files_ignored": n_ignored_other_schema,
    }
    return completed, provenance


def _flatten_observations(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        exposures = run.get("exposures") or []
        for exp in exposures:
            rows.append(
                {
                    "run_id": str(run["run_id"]),
                    "track": str(run["track"]),
                    "paradigm": str(run["paradigm"]),
                    "scale_id": str(run["scale_id"]),
                    "multiplier": float(run["multiplier"]),
                    "n_sequences": int(run["n_sequences"]),
                    "seq_len": int(run["seq_len"]),
                    "vocab_size": int(run["vocab_size"]),
                    "n_params_non_embedding": int(run["n_params_non_embedding"]),
                    "n_params_embedding": int(run["n_params_embedding"]),
                    "n_params_total": int(run["n_params_total"]),
                    "weight_decay": float(run["weight_decay"]),
                    "precision": str(run["precision"]),
                    "max_epochs": int(run["max_epochs"]),
                    "seed": int(run["seed"]),
                    "data_seed": int(run["data_seed"]),
                    "H_null_ordered_bits": float(run["H_null_ordered_bits"]),
                    "H_data_bits": float(run["H_data_bits"]),
                    "H_gibbs_max_bits": float(run["H_gibbs_max_bits"]),
                    "epoch": int(exp["epoch"]),
                    "step": int(exp["step"]),
                    "tokens_seen": int(exp["tokens_seen"]),
                    "lr": float(exp["lr"]),
                    "elapsed_sec": float(exp["elapsed_sec"]),
                    "train_loss_last_epoch_bits_per_tok": float(
                        exp["train_loss_last_epoch_bits_per_tok"]
                    ),
                    "eval_n_sequences": int(exp["eval_n_sequences"]),
                    "mean_seq_xent_bits": float(exp["mean_seq_xent_bits"]),
                    "mean_tok_xent_bits": float(exp["mean_tok_xent_bits"]),
                    "eval_mc_se_seq_bits": float(exp["eval_mc_se_seq_bits"]),
                    "total_model_xent_bits": float(exp["total_model_xent_bits"]),
                    "memorized_bits": float(exp["memorized_bits"]),
                    "memorized_bits_se": float(exp["memorized_bits_se"]),
                    "bits_per_non_emb_param": float(exp["bits_per_non_emb_param"]),
                    "bits_per_total_param": float(exp["bits_per_total_param"]),
                    "sw_memorized_bits": exp.get("sw_memorized_bits"),
                    "sw_memorized_bits_se": exp.get("sw_memorized_bits_se"),
                    "sw_bits_per_non_emb_param": exp.get("sw_bits_per_non_emb_param"),
                    "sw_bits_per_total_param": exp.get("sw_bits_per_total_param"),
                    "pseudo_memorized_bits": exp.get("pseudo_memorized_bits"),
                    "pseudo_memorized_bits_se": exp.get("pseudo_memorized_bits_se"),
                    "pseudo_bits_per_non_emb_param": exp.get("pseudo_bits_per_non_emb_param"),
                    "pseudo_bits_per_total_param": exp.get("pseudo_bits_per_total_param"),
                    "mean_masked_token_xent_bits": exp.get("mean_masked_token_xent_bits"),
                    "fraction_of_H_data": float(exp["fraction_of_H_data"]),
                    "fraction_of_H_gibbs": float(exp["fraction_of_H_gibbs"]),
                    "mlm_quadrature_16pt_bits_per_tok": exp.get(
                        "mlm_quadrature_16pt_bits_per_tok"
                    ),
                    "mlm_loo_pseudo_tok_xent_bits": exp.get("mlm_loo_pseudo_tok_xent_bits"),
                    "mlm_loo_pseudo_memorized_bits": exp.get("mlm_loo_pseudo_memorized_bits"),
                    "mlm_fixed_15pct_diagnostic": exp.get("mlm_fixed_15pct_diagnostic"),
                    "mlm_oa_16pt_diagnostic": exp.get("mlm_oa_16pt_diagnostic"),
                }
            )
    return rows


def _build_saturation_and_cells(
    primary_obs: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, str, int], dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in primary_obs:
        key = (row["paradigm"], row["scale_id"], int(row["epoch"]))
        grouped.setdefault(key, []).append(row)

    all_cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    saturation_curves: dict[str, dict[str, dict[str, Any]]] = {}

    sorted_keys = sorted(
        grouped.keys(), key=lambda k: (k[0], _scale_sort_key(k[1]), k[2])
    )
    for paradigm, scale_id, epoch in sorted_keys:
        cell = _extract_cell_summary(grouped[(paradigm, scale_id, epoch)])
        all_cells[(paradigm, scale_id, epoch)] = cell
        if cell["n_multipliers"] >= 3:
            saturation_curves.setdefault(paradigm, {}).setdefault(scale_id, {})[
                str(epoch)
            ] = cell

    return saturation_curves, all_cells


def _saturated_epochs_by_scale(
    all_cells: dict[tuple[str, str, int], dict[str, Any]],
    paradigm: str,
) -> dict[str, list[int]]:
    """Return sorted saturated epochs per scale (requiring multiplier >= 2.0 when any >= 2.0 run exists)."""
    raw_by_scale: dict[str, list[int]] = {}
    ge2_by_scale: dict[str, list[int]] = {}
    for (p, s, e), cell in all_cells.items():
        if p != paradigm:
            continue
        raw_by_scale.setdefault(s, []).append(e)
        if cell["has_ge2x_multiplier"]:
            ge2_by_scale.setdefault(s, []).append(e)

    out: dict[str, list[int]] = {}
    for s, all_es in raw_by_scale.items():
        chosen = ge2_by_scale.get(s)
        out[s] = sorted(chosen if chosen else all_es)
    return out


def _select_anchor_epoch(
    sat_epochs_by_scale: dict[str, list[int]],
    *,
    paradigm: str | None = None,
    all_cells: Mapping[tuple[str, str, int], Mapping[str, Any]] | None = None,
    min_signal_bpp: float = 0.02,
) -> int | None:
    """Select the highest saturated common epoch across scales with post-takeoff signal.

    Filters out dead pre-takeoff epochs (c_eff < min_signal_bpp across all scales) whenever
    any post-takeoff epoch exists, so mixing Batch-1 truncated MLM/OA runs (e=400, c_eff=0)
    with Stage-1 WSD runs (e=4000/10000) anchors on the active regime.
    """
    if not sat_epochs_by_scale:
        return None
    counts: dict[int, int] = {}
    active_counts: dict[int, int] = {}
    for s, es in sat_epochs_by_scale.items():
        for e in es:
            counts[e] = counts.get(e, 0) + 1
            if all_cells is not None and paradigm is not None:
                cell = all_cells.get((paradigm, s, e))
                if cell is not None and float(cell.get("c_eff_empirical", 0.0)) >= min_signal_bpp:
                    active_counts[e] = active_counts.get(e, 0) + 1
    use_counts = active_counts if active_counts else counts
    multi_scale_epochs = [e for e, cnt in use_counts.items() if cnt >= 2]
    if multi_scale_epochs:
        return max(multi_scale_epochs)
    return max(use_counts.keys())


def _build_c_max_calibration_and_exposure_laws(
    all_cells: dict[tuple[str, str, int], dict[str, Any]],
    *,
    bootstrap_draws: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paradigms = sorted({k[0] for k in all_cells})
    c_max_calibration: dict[str, Any] = {
        "nlp_reference_constants": NLP_REFERENCE_CONSTANTS_BITS_PER_PARAM,
        "by_paradigm": {},
    }
    exposure_discount_curves: dict[str, Any] = {}

    for p_idx, paradigm in enumerate(paradigms):
        p_seed = seed + p_idx * 500
        sat_epochs_map = _saturated_epochs_by_scale(all_cells, paradigm)
        anchor_epoch = _select_anchor_epoch(
            sat_epochs_map, paradigm=paradigm, all_cells=all_cells
        )
        if anchor_epoch is None:
            continue

        scales = sorted(sat_epochs_map.keys(), key=_scale_sort_key)
        all_epochs = sorted({e for es in sat_epochs_map.values() for e in es})
        anchor_scales = [
            s for s in scales if anchor_epoch in sat_epochs_map[s]
        ]

        # 1. Per-scale exposure curves and completion ratios eta_s(e) = c_eff(s, e) / c_eff(s, anchor)
        by_scale_exp: dict[str, Any] = {}
        eta_2x_by_epoch: dict[int, list[float]] = {}
        eta_plat_by_epoch: dict[int, list[float]] = {}

        for s in scales:
            s_epochs = sat_epochs_map[s]
            anchor_cell = (
                all_cells[(paradigm, s, anchor_epoch)]
                if s in anchor_scales
                else None
            )
            c_anchor_2x = (
                float(anchor_cell["c_eff_at_2x"])
                if anchor_cell is not None and float(anchor_cell["c_eff_at_2x"]) > 1e-9
                else None
            )
            c_anchor_plat = (
                float(anchor_cell["c_eff_empirical"])
                if anchor_cell is not None and float(anchor_cell["c_eff_empirical"]) > 1e-9
                else None
            )

            traj: list[dict[str, Any]] = []
            prev_e: int | None = None
            prev_c_2x: float | None = None
            prev_c_plat: float | None = None

            for e in s_epochs:
                cell = all_cells[(paradigm, s, e)]
                c_2x = float(cell["c_eff_at_2x"])
                c_plat = float(cell["c_eff_empirical"])
                c_max_ge1 = float(cell["c_eff_plateau_max"])

                eta_2x = (c_2x / c_anchor_2x) if c_anchor_2x is not None else None
                eta_plat = (c_plat / c_anchor_plat) if c_anchor_plat is not None else None
                if eta_2x is not None and s in anchor_scales:
                    eta_2x_by_epoch.setdefault(e, []).append(eta_2x)
                if eta_plat is not None and s in anchor_scales:
                    eta_plat_by_epoch.setdefault(e, []).append(eta_plat)

                marg_2x: float | None = None
                marg_plat: float | None = None
                if prev_e is not None and e > prev_e and prev_e > 0:
                    doublings = math.log2(float(e) / float(prev_e))
                    if doublings > 0.0:
                        assert prev_c_2x is not None and prev_c_plat is not None
                        marg_2x = (c_2x - prev_c_2x) / doublings
                        marg_plat = (c_plat - prev_c_plat) / doublings

                traj.append(
                    {
                        "epoch": e,
                        "c_eff_at_2x": c_2x,
                        "c_eff_empirical_ge2x": c_plat,
                        "c_eff_plateau_max_ge1x": c_max_ge1,
                        "delta_bpt_empirical": float(cell["delta_bpt_empirical"]),
                        "delta_bpt_at_2x": float(cell["delta_bpt_at_2x"]),
                        "sw_bits_per_non_emb_param": cell.get("sw_bits_per_non_emb_param"),
                        "pseudo_bits_per_non_emb_param": cell.get("pseudo_bits_per_non_emb_param"),
                        "discount_ratio_to_anchor_at_2x": eta_2x,
                        "discount_ratio_to_anchor_empirical": eta_plat,
                        "marginal_bits_per_param_per_doubling_at_2x": marg_2x,
                        "marginal_bits_per_param_per_doubling_empirical": marg_plat,
                    }
                )
                prev_e, prev_c_2x, prev_c_plat = e, c_2x, c_plat

            traj_epochs = [pt["epoch"] for pt in traj]
            traj_c_2x = [pt["c_eff_at_2x"] for pt in traj]
            traj_c_emp = [pt["c_eff_empirical_ge2x"] for pt in traj]
            traj_dbpt_emp = [pt["delta_bpt_empirical"] for pt in traj]
            traj_dbpt_2x = [pt["delta_bpt_at_2x"] for pt in traj]

            e_crit_c_eff_0p02_emp = estimate_ecrit_nonparametric(traj_epochs, traj_c_emp, 0.02)
            e_crit_c_eff_0p05_emp = estimate_ecrit_nonparametric(traj_epochs, traj_c_emp, 0.05)
            e_crit_c_eff_0p02_2x = estimate_ecrit_nonparametric(traj_epochs, traj_c_2x, 0.02)
            e_crit_c_eff_0p05_2x = estimate_ecrit_nonparametric(traj_epochs, traj_c_2x, 0.05)
            e_crit_dbpt_emp = estimate_ecrit_nonparametric(traj_epochs, traj_dbpt_emp, 0.05)
            e_crit_dbpt_2x = estimate_ecrit_nonparametric(traj_epochs, traj_dbpt_2x, 0.05)
            has_takeoff_emp = e_crit_c_eff_0p02_emp is not None
            has_takeoff_2x = e_crit_c_eff_0p02_2x is not None

            hill_2x = fit_hill_exposure_curve(traj_epochs, traj_c_2x)
            hill_plat = fit_hill_exposure_curve(traj_epochs, traj_c_emp)
            delayed_hill_2x = (
                fit_delayed_hill_exposure_curve(traj_epochs, traj_c_2x) if has_takeoff_2x else None
            )
            delayed_hill_plat = (
                fit_delayed_hill_exposure_curve(traj_epochs, traj_c_emp) if has_takeoff_emp else None
            )

            c_inf_emp = (
                float(delayed_hill_plat["c_inf_bits_per_param"])
                if delayed_hill_plat is not None
                else (
                    float(hill_plat["c_inf_bits_per_param"])
                    if (has_takeoff_emp and hill_plat is not None)
                    else (max(traj_c_emp) if (has_takeoff_emp and traj_c_emp) else 0.0)
                )
            )
            c_inf_2x = (
                float(delayed_hill_2x["c_inf_bits_per_param"])
                if delayed_hill_2x is not None
                else (
                    float(hill_2x["c_inf_bits_per_param"])
                    if (has_takeoff_2x and hill_2x is not None)
                    else (max(traj_c_2x) if (has_takeoff_2x and traj_c_2x) else 0.0)
                )
            )
            e_crit_np = {
                "e_crit_c_eff_0p02": e_crit_c_eff_0p02_emp,
                "e_crit_c_eff_0p05": e_crit_c_eff_0p05_emp,
                "e_crit_c_eff_0p02_at_2x": e_crit_c_eff_0p02_2x,
                "e_crit_c_eff_0p05_at_2x": e_crit_c_eff_0p05_2x,
                "e_crit_delta_bpt_0p05": e_crit_dbpt_emp,
                "e_crit_delta_bpt_0p05_at_2x": e_crit_dbpt_2x,
                "e_crit_frac_c_inf_0p05": (
                    estimate_ecrit_nonparametric(traj_epochs, traj_c_emp, 0.05 * c_inf_emp)
                    if (has_takeoff_emp and c_inf_emp > 0.0)
                    else None
                ),
                "e_crit_frac_c_inf_0p05_at_2x": (
                    estimate_ecrit_nonparametric(traj_epochs, traj_c_2x, 0.05 * c_inf_2x)
                    if (has_takeoff_2x and c_inf_2x > 0.0)
                    else None
                ),
                "c_inf_reference_empirical": c_inf_emp if (has_takeoff_emp and c_inf_emp > 0.0) else None,
            }
            by_scale_exp[s] = {
                "n_exposures": len(traj),
                "has_anchor_epoch": anchor_cell is not None,
                "trajectory": traj,
                "hill_fit_at_2x": hill_2x,
                "hill_fit_empirical": hill_plat,
                "delayed_hill_fit_at_2x": delayed_hill_2x,
                "delayed_hill_fit_empirical": delayed_hill_plat,
                "e_crit_nonparametric": e_crit_np,
            }

        # 2. Pooled exposure-discount curve across anchor scales & across all available scales
        eta_invariance_by_epoch: dict[str, Any] = {}
        eta_lookup_emp: dict[int, float] = {}
        eta_lookup_2x: dict[int, float] = {}
        for e in all_epochs:
            vals_emp = eta_plat_by_epoch.get(e, [])
            vals_2x = eta_2x_by_epoch.get(e, [])
            if vals_emp:
                arr_emp = np.asarray(vals_emp, dtype=np.float64)
                mean_emp = float(np.mean(arr_emp))
                std_emp = float(np.std(arr_emp, ddof=0))
                eta_lookup_emp[e] = mean_emp
            else:
                mean_emp = None
                std_emp = None
            if vals_2x:
                arr_2x = np.asarray(vals_2x, dtype=np.float64)
                mean_2x = float(np.mean(arr_2x))
                std_2x = float(np.std(arr_2x, ddof=0))
                eta_lookup_2x[e] = mean_2x
            else:
                mean_2x = None
                std_2x = None

            if mean_emp is not None or mean_2x is not None:
                eta_invariance_by_epoch[str(e)] = {
                    "n_anchor_scales": len(vals_emp),
                    "eta_empirical_mean": mean_emp,
                    "eta_empirical_std": std_emp,
                    "eta_empirical_max_spread": (
                        float(np.max(vals_emp) - np.min(vals_emp)) if vals_emp else None
                    ),
                    "eta_at_2x_mean": mean_2x,
                    "eta_at_2x_std": std_2x,
                    "eta_at_2x_max_spread": (
                        float(np.max(vals_2x) - np.min(vals_2x)) if vals_2x else None
                    ),
                }

        pooled_anchor_traj: list[dict[str, Any]] = []
        prev_pe: int | None = None
        prev_pc_2x: float | None = None
        prev_pc_emp: float | None = None

        for e in all_epochs:
            pool_scales = [
                s for s in (anchor_scales if anchor_scales else scales)
                if e in sat_epochs_map[s]
            ]
            all_avail_scales = [s for s in scales if e in sat_epochs_map[s]]
            if not all_avail_scales:
                continue

            c2x_anchor_vals = [
                float(all_cells[(paradigm, s, e)]["c_eff_at_2x"]) for s in pool_scales
            ]
            cemp_anchor_vals = [
                float(all_cells[(paradigm, s, e)]["c_eff_empirical"]) for s in pool_scales
            ]
            cmax_anchor_vals = [
                float(all_cells[(paradigm, s, e)]["c_eff_plateau_max"]) for s in pool_scales
            ]
            cemp_all_vals = [
                float(all_cells[(paradigm, s, e)]["c_eff_empirical"]) for s in all_avail_scales
            ]
            dbpt_anchor_vals = [
                float(all_cells[(paradigm, s, e)]["delta_bpt_empirical"]) for s in pool_scales
            ]
            dbpt_all_vals = [
                float(all_cells[(paradigm, s, e)]["delta_bpt_empirical"]) for s in all_avail_scales
            ]
            sw_anchor_vals = [
                float(all_cells[(paradigm, s, e)]["sw_bits_per_non_emb_param"])
                for s in pool_scales
                if all_cells[(paradigm, s, e)].get("sw_bits_per_non_emb_param") is not None
            ]
            pseudo_anchor_vals = [
                float(all_cells[(paradigm, s, e)]["pseudo_bits_per_non_emb_param"])
                for s in pool_scales
                if all_cells[(paradigm, s, e)].get("pseudo_bits_per_non_emb_param") is not None
            ]

            mean_2x = float(np.mean(c2x_anchor_vals)) if c2x_anchor_vals else float(np.mean(cemp_all_vals))
            mean_emp = float(np.mean(cemp_anchor_vals)) if cemp_anchor_vals else float(np.mean(cemp_all_vals))
            mean_pmax = float(np.mean(cmax_anchor_vals)) if cmax_anchor_vals else mean_emp
            mean_dbpt = float(np.mean(dbpt_anchor_vals)) if dbpt_anchor_vals else float(np.mean(dbpt_all_vals))

            marg_2x_p: float | None = None
            marg_emp_p: float | None = None
            if prev_pe is not None and e > prev_pe and prev_pe > 0 and cemp_anchor_vals:
                doublings = math.log2(float(e) / float(prev_pe))
                if doublings > 0.0 and prev_pc_2x is not None and prev_pc_emp is not None:
                    marg_2x_p = (mean_2x - prev_pc_2x) / doublings
                    marg_emp_p = (mean_emp - prev_pc_emp) / doublings

            pooled_anchor_traj.append(
                {
                    "epoch": e,
                    "n_anchor_scales_present": len(pool_scales),
                    "n_total_scales_present": len(all_avail_scales),
                    "c_eff_at_2x_mean": mean_2x,
                    "c_eff_empirical_mean": mean_emp,
                    "c_eff_plateau_max_mean": mean_pmax,
                    "c_eff_all_scales_mean": float(np.mean(cemp_all_vals)),
                    "delta_bpt_empirical_mean": mean_dbpt,
                    "sw_bits_per_non_emb_param_mean": (
                        float(np.mean(sw_anchor_vals)) if sw_anchor_vals else None
                    ),
                    "pseudo_bits_per_non_emb_param_mean": (
                        float(np.mean(pseudo_anchor_vals)) if pseudo_anchor_vals else None
                    ),
                    "c_eff_empirical_ci95": _bootstrap_mean_ci(
                        cemp_anchor_vals if cemp_anchor_vals else cemp_all_vals,
                        n_draws=bootstrap_draws,
                        seed=p_seed + e,
                    ),
                    "discount_ratio_eta_at_2x": eta_lookup_2x.get(e),
                    "discount_ratio_eta_empirical": eta_lookup_emp.get(e),
                    "marginal_bits_per_param_per_doubling_at_2x": marg_2x_p,
                    "marginal_bits_per_param_per_doubling_empirical": marg_emp_p,
                }
            )
            if cemp_anchor_vals:
                prev_pe, prev_pc_2x, prev_pc_emp = e, mean_2x, mean_emp

        anchor_only_pts = [pt for pt in pooled_anchor_traj if pt["n_anchor_scales_present"] > 0]
        anchor_epochs_list = [pt["epoch"] for pt in anchor_only_pts]
        anchor_c2x_list = [pt["c_eff_at_2x_mean"] for pt in anchor_only_pts]
        anchor_cemp_list = [pt["c_eff_empirical_mean"] for pt in anchor_only_pts]
        anchor_pmax_list = [pt["c_eff_plateau_max_mean"] for pt in anchor_only_pts]
        anchor_dbpt_list = [pt["delta_bpt_empirical_mean"] for pt in anchor_only_pts]

        pooled_e_crit_c_eff_0p02 = estimate_ecrit_nonparametric(
            anchor_epochs_list, anchor_cemp_list, 0.02
        )
        pooled_e_crit_c_eff_0p05 = estimate_ecrit_nonparametric(
            anchor_epochs_list, anchor_cemp_list, 0.05
        )
        pooled_e_crit_dbpt = estimate_ecrit_nonparametric(anchor_epochs_list, anchor_dbpt_list, 0.05)
        pooled_has_takeoff = pooled_e_crit_c_eff_0p02 is not None

        pooled_hill_2x = fit_hill_exposure_curve(anchor_epochs_list, anchor_c2x_list)
        pooled_hill_emp = fit_hill_exposure_curve(anchor_epochs_list, anchor_cemp_list)
        pooled_hill_pmax = fit_hill_exposure_curve(anchor_epochs_list, anchor_pmax_list)
        pooled_delayed_2x = (
            fit_delayed_hill_exposure_curve(anchor_epochs_list, anchor_c2x_list)
            if pooled_has_takeoff
            else None
        )
        pooled_delayed_emp = (
            fit_delayed_hill_exposure_curve(anchor_epochs_list, anchor_cemp_list)
            if pooled_has_takeoff
            else None
        )
        pooled_delayed_pmax = (
            fit_delayed_hill_exposure_curve(anchor_epochs_list, anchor_pmax_list)
            if pooled_has_takeoff
            else None
        )

        pooled_c_inf_ref = (
            float(pooled_delayed_emp["c_inf_bits_per_param"])
            if pooled_delayed_emp is not None
            else (
                float(pooled_hill_emp["c_inf_bits_per_param"])
                if (pooled_has_takeoff and pooled_hill_emp is not None)
                else (max(anchor_cemp_list) if (pooled_has_takeoff and anchor_cemp_list) else 0.0)
            )
        )
        pooled_e_crit_np = {
            "e_crit_c_eff_0p02": pooled_e_crit_c_eff_0p02,
            "e_crit_c_eff_0p05": pooled_e_crit_c_eff_0p05,
            "e_crit_delta_bpt_0p05": pooled_e_crit_dbpt,
            "e_crit_frac_c_inf_0p05": (
                estimate_ecrit_nonparametric(
                    anchor_epochs_list, anchor_cemp_list, 0.05 * pooled_c_inf_ref
                )
                if (pooled_has_takeoff and pooled_c_inf_ref > 0.0)
                else None
            ),
            "c_inf_reference_empirical": (
                pooled_c_inf_ref if (pooled_has_takeoff and pooled_c_inf_ref > 0.0) else None
            ),
        }

        p_display = paradigm_display_name(paradigm)
        e_mem_canonical = (
            float(pooled_delayed_emp["e_crit_epochs"])
            if (pooled_delayed_emp is not None and pooled_delayed_emp.get("e_crit_epochs") is not None)
            else pooled_e_crit_c_eff_0p02
        )
        c_inf_canonical = (
            float(pooled_delayed_emp["c_inf_bits_per_param"])
            if pooled_delayed_emp is not None
            else (
                float(pooled_hill_emp["c_inf_bits_per_param"])
                if pooled_hill_emp is not None
                else None
            )
        )
        exposure_discount_curves[paradigm] = {
            "paradigm": paradigm,
            "paradigm_display": p_display,
            "E_gen": None,
            "E_mem": e_mem_canonical,
            "W_window": None,
            "Delta_cal": 0.0,
            "anchor_epoch": anchor_epoch,
            "anchor_scale_ids": anchor_scales,
            "pooled_by_exposure": pooled_anchor_traj,
            "pooled_hill_fit_at_2x": pooled_hill_2x,
            "pooled_hill_fit_empirical": pooled_hill_emp,
            "pooled_hill_fit_plateau_max": pooled_hill_pmax,
            "pooled_delayed_hill_fit_at_2x": pooled_delayed_2x,
            "pooled_delayed_hill_fit_empirical": pooled_delayed_emp,
            "pooled_delayed_hill_fit_plateau_max": pooled_delayed_pmax,
            "pooled_e_crit_nonparametric": pooled_e_crit_np,
            "eta_scale_invariance": eta_invariance_by_epoch,
            "by_scale": by_scale_exp,
        }

        # 3. Direct anchor-epoch (e = 400) strictly unadjusted c_max calibration
        direct_non_emb_p = [
            float(all_cells[(paradigm, s, anchor_epoch)]["n_params_non_embedding"])
            for s in anchor_scales
        ]
        direct_tot_p = [
            float(all_cells[(paradigm, s, anchor_epoch)]["n_params_total"])
            for s in anchor_scales
        ]
        direct_m_plat = [
            float(all_cells[(paradigm, s, anchor_epoch)]["plateau_empirical_bits"])
            for s in anchor_scales
        ]

        direct_non_emb_fit = _fit_linear_capacity_scaling(
            anchor_scales,
            direct_non_emb_p,
            direct_m_plat,
            bootstrap_draws=bootstrap_draws,
            seed=p_seed + 101,
        )
        direct_tot_fit = _fit_linear_capacity_scaling(
            anchor_scales,
            direct_tot_p,
            direct_m_plat,
            bootstrap_draws=bootstrap_draws,
            seed=p_seed + 151,
        )

        # 4. Secondary exposure-anchored multi-scale (0.5m -> 150m) calibration
        anchored_scales: list[str] = []
        anchored_non_emb_p: list[float] = []
        anchored_tot_p: list[float] = []
        anchored_m_bits: list[float] = []
        anchored_records: list[dict[str, Any]] = []

        for s in scales:
            if s in anchor_scales:
                cell = all_cells[(paradigm, s, anchor_epoch)]
                m_raw = float(cell["plateau_empirical_bits"])
                eta_val = 1.0
                used_epoch = anchor_epoch
                extrapolated = False
                eta_source = "direct_anchor"
            else:
                s_epochs = sat_epochs_map[s]
                if not s_epochs:
                    continue
                used_epoch = s_epochs[-1]
                cell = all_cells[(paradigm, s, used_epoch)]
                # Prefer matching 2.0x eta when target scale's saturated max_epoch lacks 4x/8x
                has_4x_or_higher = any(m >= 4.0 - 1e-9 for m in cell["multipliers"])
                if cell["has_exact_2x_multiplier"] and not has_4x_or_higher and used_epoch in eta_lookup_2x:
                    m_raw = float(cell["memorized_at_2x_bits"])
                    eta_val = float(eta_lookup_2x[used_epoch])
                    eta_source = f"anchor_mean_eta_at_2x_e{used_epoch}"
                elif used_epoch in eta_lookup_emp:
                    m_raw = float(cell["plateau_empirical_bits"])
                    eta_val = float(eta_lookup_emp[used_epoch])
                    eta_source = f"anchor_mean_eta_empirical_e{used_epoch}"
                elif pooled_hill_emp is not None and anchor_epoch > 0:
                    m_raw = float(cell["plateau_empirical_bits"])
                    e50 = float(pooled_hill_emp["e_50_epochs"])
                    alp = float(pooled_hill_emp["alpha_hill"])
                    num_h = float(expit(alp * (math.log(used_epoch) - math.log(e50))))
                    den_h = float(expit(alp * (math.log(anchor_epoch) - math.log(e50))))
                    eta_val = num_h / max(den_h, 1e-9)
                    eta_source = f"hill_interpolated_eta_e{used_epoch}"
                else:
                    continue

                if eta_val <= 1e-6:
                    continue
                extrapolated = True

            m_adj = m_raw / eta_val
            p_ne = float(cell["n_params_non_embedding"])
            p_tot = float(cell["n_params_total"])
            anchored_scales.append(s)
            anchored_non_emb_p.append(p_ne)
            anchored_tot_p.append(p_tot)
            anchored_m_bits.append(m_adj)
            anchored_records.append(
                {
                    "scale_id": s,
                    "observed_max_epoch": used_epoch,
                    "is_exposure_extrapolated": extrapolated,
                    "eta_completion_factor": eta_val,
                    "eta_source": eta_source,
                    "n_params_non_embedding": int(p_ne),
                    "n_params_total": int(p_tot),
                    "observed_plateau_bits": m_raw,
                    "anchored_e400_equivalent_bits": m_adj,
                    "observed_bits_per_non_emb_param": m_raw / p_ne,
                    "anchored_bits_per_non_emb_param": m_adj / p_ne,
                    "anchored_bits_per_total_param": m_adj / p_tot,
                    "sw_bits_per_non_emb_param": cell.get("sw_bits_per_non_emb_param"),
                    "pseudo_bits_per_non_emb_param": cell.get("pseudo_bits_per_non_emb_param"),
                }
            )

        anchored_non_emb_fit = _fit_linear_capacity_scaling(
            anchored_scales,
            anchored_non_emb_p,
            anchored_m_bits,
            bootstrap_draws=bootstrap_draws,
            seed=p_seed + 201,
        )
        anchored_tot_fit = _fit_linear_capacity_scaling(
            anchored_scales,
            anchored_tot_p,
            anchored_m_bits,
            bootstrap_draws=bootstrap_draws,
            seed=p_seed + 251,
        )

        # Dual Slepian-Wolf vs symmetric-slot pseudo capacity calibration for mlm_15
        mlm_15_dual_cal: dict[str, Any] | None = None
        if anchor_scales and all(
            all_cells[(paradigm, s, anchor_epoch)].get("sw_memorized_empirical_bits") is not None
            for s in anchor_scales
        ):
            direct_sw_bits = [
                float(all_cells[(paradigm, s, anchor_epoch)]["sw_memorized_empirical_bits"])
                for s in anchor_scales
            ]
            direct_pseudo_bits = [
                float(all_cells[(paradigm, s, anchor_epoch)]["pseudo_memorized_empirical_bits"])
                for s in anchor_scales
            ]
            mlm_15_dual_cal = {
                "sw_bits_per_non_emb_param": _fit_linear_capacity_scaling(
                    anchor_scales,
                    direct_non_emb_p,
                    direct_sw_bits,
                    bootstrap_draws=bootstrap_draws,
                    seed=p_seed + 301,
                ),
                "pseudo_bits_per_non_emb_param": _fit_linear_capacity_scaling(
                    anchor_scales,
                    direct_non_emb_p,
                    direct_pseudo_bits,
                    bootstrap_draws=bootstrap_draws,
                    seed=p_seed + 331,
                ),
            }

        primary_c_max_non_emb = (
            float(direct_non_emb_fit["zero_intercept_ratio"]["c_max"])
            if direct_non_emb_fit is not None
            else None
        )
        primary_c_max_total = (
            float(direct_tot_fit["zero_intercept_ratio"]["c_max"])
            if direct_tot_fit is not None
            else None
        )

        metric_semantics_map = {
            "ar": "ar_joint_exact_bits_per_non_emb_param",
            "oa_uniform": "oa_uniform_joint_nelbo_16pt_bits_per_non_emb_param",
            "mlm_15": "mlm_15_dual_sw_and_symmetric_pseudo_bits_per_non_emb_param",
        }
        max_obs_by_scale: dict[str, dict[str, Any]] = {}
        delayed_c_inf_by_scale: dict[str, float | None] = {}
        for s in scales:
            s_epochs = sat_epochs_map.get(s, [])
            if not s_epochs:
                continue
            max_e = s_epochs[-1]
            max_cell = all_cells[(paradigm, s, max_e)]
            s_dh = by_scale_exp[s].get("delayed_hill_fit_empirical")
            s_c_inf = float(s_dh["c_inf_bits_per_param"]) if s_dh is not None else None
            delayed_c_inf_by_scale[s] = s_c_inf
            entry_s: dict[str, Any] = {
                "max_observed_epoch": int(max_e),
                "c_eff_empirical_bpp": float(max_cell["c_eff_empirical"]),
                "c_eff_at_2x_bpp": float(max_cell["c_eff_at_2x"]),
                "delayed_hill_c_inf_bpp": s_c_inf,
            }
            if paradigm == "mlm_15":
                entry_s["sw_bits_per_non_emb_param"] = max_cell.get("sw_bits_per_non_emb_param")
                entry_s["pseudo_bits_per_non_emb_param"] = max_cell.get("pseudo_bits_per_non_emb_param")
            max_obs_by_scale[s] = entry_s

        pooled_delayed_const_lr = (
            fit_delayed_hill_exposure_curve(anchor_epochs_list[:-1], anchor_cemp_list[:-1])
            if (pooled_has_takeoff and len(anchor_epochs_list) >= 5)
            else pooled_delayed_emp
        )
        c_max_post_cooldown_bpp = (
            max((v["c_eff_empirical_bpp"] for v in max_obs_by_scale.values()), default=None)
        )
        asymptotic_summary: dict[str, Any] = {
            "metric_semantics": metric_semantics_map.get(paradigm, paradigm),
            "anchor_epoch": int(anchor_epoch),
            "max_observed_epoch": max((v["max_observed_epoch"] for v in max_obs_by_scale.values()), default=None),
            "c_max_post_cooldown_bpp": c_max_post_cooldown_bpp,
            "c_inf_constant_lr_bpp": (
                float(pooled_delayed_const_lr["c_inf_bits_per_param"])
                if pooled_delayed_const_lr is not None
                else None
            ),
            "delayed_hill_c_inf_pooled_bpp": (
                float(pooled_delayed_emp["c_inf_bits_per_param"])
                if pooled_delayed_emp is not None
                else None
            ),
            "delayed_hill_c_inf_by_scale": delayed_c_inf_by_scale,
            "max_observed_by_scale": max_obs_by_scale,
        }

        c_max_calibration["by_paradigm"][paradigm] = {
            "paradigm": paradigm,
            "paradigm_display": p_display,
            "E_gen": None,
            "E_mem": e_mem_canonical,
            "W_window": None,
            "Delta_cal": 0.0,
            "canonical_notation": {
                "paradigm_display": p_display,
                "c_inf": c_inf_canonical,
                "E_gen": None,
                "E_mem": e_mem_canonical,
                "W_window": None,
                "Delta_cal": 0.0,
            },
            "anchor_epoch": anchor_epoch,
            "is_canonical_e400_anchor": anchor_epoch == 400,
            "asymptotic_and_metric_summary": asymptotic_summary,
            "direct_anchor_unadjusted": {
                "non_embedding_params": direct_non_emb_fit,
                "total_params": direct_tot_fit,
                "per_scale_measurements": [
                    r for r in anchored_records if not r["is_exposure_extrapolated"]
                ],
            },
            "exposure_anchored_multi_scale": {
                "note": (
                    "Secondary multi-scale check applying scale-invariant completion factor "
                    "eta(e) = c_eff(e)/c_eff(anchor) to scales capped below anchor_epoch."
                ),
                "non_embedding_params": anchored_non_emb_fit,
                "total_params": anchored_tot_fit,
                "per_scale_measurements": anchored_records,
            },
            "mlm_15_dual_capacity": mlm_15_dual_cal,
            "nlp_reference_comparison": {
                "non_embedding_direct": (
                    _nlp_comparison_for_cmax(primary_c_max_non_emb)
                    if primary_c_max_non_emb is not None
                    else None
                ),
                "total_params_direct": (
                    _nlp_comparison_for_cmax(primary_c_max_total)
                    if primary_c_max_total is not None
                    else None
                ),
            },
        }

    return c_max_calibration, exposure_discount_curves


def _build_ar_vs_mlm_and_loo_audit(
    all_obs: Sequence[dict[str, Any]],
    *,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    primary_ar: dict[tuple[str, float, int], dict[str, Any]] = {}
    primary_oa: dict[tuple[str, float, int], dict[str, Any]] = {}
    primary_mlm15: dict[tuple[str, float, int], dict[str, Any]] = {}
    for row in all_obs:
        if row["track"] not in PRIMARY_TRACKS:
            continue
        key = (row["scale_id"], round(float(row["multiplier"]), 6), int(row["epoch"]))
        if row["paradigm"] == "ar":
            primary_ar[key] = row
        elif row["paradigm"] == "oa_uniform":
            primary_oa[key] = row
        elif row["paradigm"] == "mlm_15":
            primary_mlm15[key] = row

    shared_keys = sorted(
        set(primary_ar.keys()) & set(primary_oa.keys()),
        key=lambda k: (_scale_sort_key(k[0]), k[1], k[2]),
    )

    paired_cells: list[dict[str, Any]] = []
    quad_accum_rates: np.ndarray | None = None
    quad_bpt_rows: list[np.ndarray] = []
    quad_ar_bpt: list[float] = []
    quad_e400_bpt_rows: list[np.ndarray] = []
    quad_e400_ar_bpt: list[float] = []

    for sc, mult, ep in shared_keys:
        ar = primary_ar[(sc, mult, ep)]
        mlm = primary_oa[(sc, mult, ep)]
        m_ar = float(ar["memorized_bits"])
        m_mlm = float(mlm["memorized_bits"])
        c_ar = float(ar["bits_per_non_emb_param"])
        c_mlm = float(mlm["bits_per_non_emb_param"])
        ratio = (m_mlm / m_ar) if m_ar > 1e-6 else None

        paired_cells.append(
            {
                "scale_id": sc,
                "multiplier": float(ar["multiplier"]),
                "epoch": ep,
                "n_sequences": int(ar["n_sequences"]),
                "n_params_non_embedding": int(ar["n_params_non_embedding"]),
                "ar_memorized_bits": m_ar,
                "mlm_memorized_bits": m_mlm,
                "ar_bits_per_non_emb_param": c_ar,
                "mlm_bits_per_non_emb_param": c_mlm,
                "ar_minus_mlm_bits_per_param": c_ar - c_mlm,
                "mlm_over_ar_ratio": ratio,
            }
        )

        parsed_q = _parse_quadrature_profile(mlm.get("mlm_quadrature_16pt_bits_per_tok"))
        if parsed_q is not None:
            rates, bpt = parsed_q
            if quad_accum_rates is None:
                quad_accum_rates = rates
            if rates.shape == quad_accum_rates.shape:
                quad_bpt_rows.append(bpt)
                quad_ar_bpt.append(float(ar["mean_tok_xent_bits"]))
                if ep == 400 and float(ar["multiplier"]) >= 2.0 - 1e-9:
                    quad_e400_bpt_rows.append(bpt)
                    quad_e400_ar_bpt.append(float(ar["mean_tok_xent_bits"]))

    # Paired AR vs MLM-15% (reporting both exact Slepian-Wolf and symmetric slot pseudo bounds)
    shared_15_keys = sorted(
        set(primary_ar.keys()) & set(primary_mlm15.keys()),
        key=lambda k: (_scale_sort_key(k[0]), k[1], k[2]),
    )
    paired_mlm15_cells: list[dict[str, Any]] = []
    for sc, mult, ep in shared_15_keys:
        ar = primary_ar[(sc, mult, ep)]
        m15 = primary_mlm15[(sc, mult, ep)]
        c_ar = float(ar["bits_per_non_emb_param"])
        c_pseudo = float(
            m15["pseudo_bits_per_non_emb_param"]
            if m15.get("pseudo_bits_per_non_emb_param") is not None
            else m15["bits_per_non_emb_param"]
        )
        c_sw = (
            float(m15["sw_bits_per_non_emb_param"])
            if m15.get("sw_bits_per_non_emb_param") is not None
            else None
        )
        paired_mlm15_cells.append(
            {
                "scale_id": sc,
                "multiplier": float(ar["multiplier"]),
                "epoch": ep,
                "n_sequences": int(ar["n_sequences"]),
                "n_params_non_embedding": int(ar["n_params_non_embedding"]),
                "ar_bits_per_non_emb_param": c_ar,
                "mlm_15_sw_bits_per_non_emb_param": c_sw,
                "mlm_15_pseudo_bits_per_non_emb_param": c_pseudo,
                "ar_minus_mlm_15_pseudo_bpp": c_ar - c_pseudo,
                "ar_minus_mlm_15_sw_bpp": (c_ar - c_sw) if c_sw is not None else None,
            }
        )

    def _summarize_subset(cells: Sequence[dict[str, Any]], sub_seed: int) -> dict[str, Any] | None:
        if not cells:
            return None
        diffs = [float(c["ar_minus_mlm_bits_per_param"]) for c in cells]
        ar_bits = [float(c["ar_memorized_bits"]) for c in cells]
        mlm_bits = [float(c["mlm_memorized_bits"]) for c in cells]
        cell_ratios = [
            float(c["mlm_over_ar_ratio"])
            for c in cells
            if c["mlm_over_ar_ratio"] is not None and math.isfinite(float(c["mlm_over_ar_ratio"]))
        ]
        return {
            "n_paired_cells": len(cells),
            "ar_minus_mlm_bits_per_param": _bootstrap_mean_ci(
                diffs, n_draws=bootstrap_draws, seed=sub_seed
            ),
            "mlm_over_ar_total_bits_ratio": _bootstrap_ratio_of_sums(
                mlm_bits, ar_bits, n_draws=bootstrap_draws, seed=sub_seed + 7
            ),
            "mlm_over_ar_cell_ratio_mean": _bootstrap_mean_ci(
                cell_ratios, n_draws=bootstrap_draws, seed=sub_seed + 13
            ),
        }

    by_exposure: dict[str, Any] = {}
    for ep in sorted({c["epoch"] for c in paired_cells}):
        ep_cells = [c for c in paired_cells if c["epoch"] == ep]
        ep_sat = [c for c in ep_cells if c["multiplier"] >= 2.0 - 1e-9]
        by_exposure[str(ep)] = {
            "all_multipliers": _summarize_subset(ep_cells, seed + ep * 3),
            "saturated_ge2x": _summarize_subset(
                ep_sat if ep_sat else ep_cells, seed + ep * 3 + 1
            ),
        }

    e400_cells = [c for c in paired_cells if c["epoch"] == 400]
    e400_sat = [c for c in e400_cells if c["multiplier"] >= 2.0 - 1e-9]

    # Masking-rate profile decomposition
    masking_profile: dict[str, Any] | None = None
    active_q_rows = quad_e400_bpt_rows if quad_e400_bpt_rows else quad_bpt_rows
    active_ar_bpt = quad_e400_ar_bpt if quad_e400_ar_bpt else quad_ar_bpt
    if quad_accum_rates is not None and active_q_rows:
        mat = np.vstack(active_q_rows)
        mean_bpt_by_stratum = np.mean(mat, axis=0)
        ar_mean_bpt = float(np.mean(active_ar_bpt))
        saved_vs_null = LOG2_20 - mean_bpt_by_stratum
        excess_vs_ar = mean_bpt_by_stratum - ar_mean_bpt

        low_mask = quad_accum_rates < 0.25
        mid_mask = (quad_accum_rates >= 0.25) & (quad_accum_rates < 0.75)
        high_mask = quad_accum_rates >= 0.75
        upper_half = quad_accum_rates >= 0.50

        total_saved_sum = float(np.sum(np.maximum(saved_vs_null, 0.0)))
        total_excess_pos = float(np.sum(np.maximum(excess_vs_ar, 0.0)))

        masking_profile = {
            "subset_used": (
                "e400_saturated_ge2x" if quad_e400_bpt_rows else "all_paired_primary_cells"
            ),
            "n_cells_averaged": len(active_q_rows),
            "null_bits_per_tok": float(LOG2_20),
            "paired_ar_mean_tok_xent_bits": ar_mean_bpt,
            "paired_mlm_mean_tok_xent_bits": float(np.mean(mean_bpt_by_stratum)),
            "strata": [
                {
                    "stratum_idx": int(i),
                    "mask_rate": float(quad_accum_rates[i]),
                    "mlm_bits_per_tok": float(mean_bpt_by_stratum[i]),
                    "bits_saved_vs_null_per_tok": float(saved_vs_null[i]),
                    "mlm_minus_ar_bits_per_tok": float(excess_vs_ar[i]),
                }
                for i in range(quad_accum_rates.size)
            ],
            "regime_summary": {
                "low_mask_lt_0p25_mean_saved_bpt": (
                    float(np.mean(saved_vs_null[low_mask])) if np.any(low_mask) else None
                ),
                "mid_mask_0p25_to_0p75_mean_saved_bpt": (
                    float(np.mean(saved_vs_null[mid_mask])) if np.any(mid_mask) else None
                ),
                "high_mask_ge_0p75_mean_saved_bpt": (
                    float(np.mean(saved_vs_null[high_mask])) if np.any(high_mask) else None
                ),
                "high_mask_ge_0p75_share_of_memorized_bits": (
                    float(np.sum(np.maximum(saved_vs_null[high_mask], 0.0)) / total_saved_sum)
                    if total_saved_sum > 1e-12 and np.any(high_mask)
                    else None
                ),
                "upper_half_mask_ge_0p50_share_of_ar_deficit": (
                    float(np.sum(np.maximum(excess_vs_ar[upper_half], 0.0)) / total_excess_pos)
                    if total_excess_pos > 1e-12 and np.any(upper_half)
                    else None
                ),
            },
        }

    # Leave-One-Out (LOO) pseudo-entropy inflation audit across all MLM runs (oa_uniform & mlm_15)
    loo_records: list[dict[str, Any]] = []
    for row in all_obs:
        if row["paradigm"] not in ("oa_uniform", "mlm_15"):
            continue
        loo_mem = row.get("mlm_loo_pseudo_memorized_bits")
        if loo_mem is None:
            continue
        exact_mem = float(row["memorized_bits"])
        loo_val = float(loo_mem)
        p_ne = float(row["n_params_non_embedding"])
        inf_ratio = (loo_val / exact_mem) if exact_mem > 1e-6 else None
        loo_records.append(
            {
                "run_id": row["run_id"],
                "paradigm": row["paradigm"],
                "track": row["track"],
                "scale_id": row["scale_id"],
                "multiplier": float(row["multiplier"]),
                "epoch": int(row["epoch"]),
                "exact_16pt_memorized_bits": exact_mem,
                "loo_pseudo_memorized_bits": loo_val,
                "excess_loo_bits": loo_val - exact_mem,
                "exact_bits_per_non_emb_param": exact_mem / p_ne,
                "loo_pseudo_bits_per_non_emb_param": loo_val / p_ne,
                "excess_loo_bits_per_non_emb_param": (loo_val - exact_mem) / p_ne,
                "inflation_ratio_loo_over_exact": inf_ratio,
            }
        )

    def _summarize_loo(recs: Sequence[dict[str, Any]], lseed: int) -> dict[str, Any] | None:
        if not recs:
            return None
        loo_b = [float(r["loo_pseudo_memorized_bits"]) for r in recs]
        ex_b = [float(r["exact_16pt_memorized_bits"]) for r in recs]
        ex_bpp = [float(r["excess_loo_bits_per_non_emb_param"]) for r in recs]
        ratios = [
            float(r["inflation_ratio_loo_over_exact"])
            for r in recs
            if r["inflation_ratio_loo_over_exact"] is not None
            and math.isfinite(float(r["inflation_ratio_loo_over_exact"]))
        ]
        return {
            "n_mlm_observations": len(recs),
            "aggregate_inflation_ratio_sum_loo_over_sum_exact": _bootstrap_ratio_of_sums(
                loo_b, ex_b, n_draws=bootstrap_draws, seed=lseed
            ),
            "cell_inflation_ratio_mean": _bootstrap_mean_ci(
                ratios, n_draws=bootstrap_draws, seed=lseed + 5
            ),
            "excess_loo_bits_per_non_emb_param": _bootstrap_mean_ci(
                ex_bpp, n_draws=bootstrap_draws, seed=lseed + 9
            ),
        }

    primary_loo = [r for r in loo_records if r["track"] in PRIMARY_TRACKS]
    primary_loo_e400_sat = [
        r for r in primary_loo if r["epoch"] == 400 and r["multiplier"] >= 2.0 - 1e-9
    ]
    loo_by_epoch = {
        str(ep): _summarize_loo(
            [r for r in primary_loo if r["epoch"] == ep], seed + 700 + ep
        )
        for ep in sorted({r["epoch"] for r in primary_loo})
    }

    loo_by_paradigm = {
        par: _summarize_loo(
            [r for r in loo_records if r["paradigm"] == par], seed + 750 + idx
        )
        for idx, par in enumerate(sorted({r["paradigm"] for r in loo_records}))
    }

    return {
        "n_total_paired_cells": len(paired_cells),
        "overall_all_cells": _summarize_subset(paired_cells, seed + 401),
        "overall_saturated_ge2x": _summarize_subset(
            [c for c in paired_cells if c["multiplier"] >= 2.0 - 1e-9], seed + 419
        ),
        "asymptotic_e400_all_multipliers": _summarize_subset(e400_cells, seed + 433),
        "asymptotic_e400_saturated_ge2x": _summarize_subset(
            e400_sat if e400_sat else e400_cells, seed + 449
        ),
        "by_exposure": by_exposure,
        "masking_rate_profile": masking_profile,
        "ar_vs_mlm_15": {
            "n_total_paired_cells": len(paired_mlm15_cells),
            "paired_cells": paired_mlm15_cells,
            "overall_ar_minus_pseudo_bpp": _bootstrap_mean_ci(
                [c["ar_minus_mlm_15_pseudo_bpp"] for c in paired_mlm15_cells],
                n_draws=bootstrap_draws,
                seed=seed + 551,
            ),
            "overall_ar_minus_sw_bpp": _bootstrap_mean_ci(
                [
                    c["ar_minus_mlm_15_sw_bpp"]
                    for c in paired_mlm15_cells
                    if c["ar_minus_mlm_15_sw_bpp"] is not None
                ],
                n_draws=bootstrap_draws,
                seed=seed + 557,
            ),
        },
        "loo_inflation_audit": {
            "all_mlm_observations": _summarize_loo(loo_records, seed + 601),
            "primary_mlm_observations": _summarize_loo(primary_loo, seed + 613),
            "primary_e400_saturated_ge2x": _summarize_loo(
                primary_loo_e400_sat, seed + 631
            ),
            "by_paradigm": loo_by_paradigm,
            "by_exposure": loo_by_epoch,
        },
        "paired_cells": paired_cells,
    }


def _build_ablations(
    all_obs: Sequence[dict[str, Any]],
    *,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    primary_map: dict[tuple[str, str, float, int], dict[str, Any]] = {}
    for row in all_obs:
        if row["track"] in PRIMARY_TRACKS:
            key = (
                row["paradigm"],
                row["scale_id"],
                round(float(row["multiplier"]), 6),
                int(row["epoch"]),
            )
            primary_map[key] = row

    def _pair_standard_track(track_name: str, a_seed: int) -> dict[str, Any]:
        pairs: list[dict[str, Any]] = []
        for row in all_obs:
            if row["track"] != track_name:
                continue
            key = (
                row["paradigm"],
                row["scale_id"],
                round(float(row["multiplier"]), 6),
                int(row["epoch"]),
            )
            base = primary_map.get(key)
            if base is None:
                continue
            c_abl = float(row["bits_per_non_emb_param"])
            c_base = float(base["bits_per_non_emb_param"])
            m_abl = float(row["memorized_bits"])
            m_base = float(base["memorized_bits"])
            pairs.append(
                {
                    "paradigm": row["paradigm"],
                    "scale_id": row["scale_id"],
                    "multiplier": float(row["multiplier"]),
                    "epoch": int(row["epoch"]),
                    "ablation_run_id": row["run_id"],
                    "baseline_run_id": base["run_id"],
                    "ablation_weight_decay": float(row["weight_decay"]),
                    "baseline_weight_decay": float(base["weight_decay"]),
                    "ablation_precision": row["precision"],
                    "baseline_precision": base["precision"],
                    "ablation_bits_per_non_emb_param": c_abl,
                    "baseline_bits_per_non_emb_param": c_base,
                    "delta_ablation_minus_baseline_bpp": c_abl - c_base,
                    "abs_diff_bpp": abs(c_abl - c_base),
                    "ratio_ablation_over_baseline": (
                        m_abl / m_base if m_base > 1e-6 else None
                    ),
                }
            )

        pairs.sort(
            key=lambda p: (p["paradigm"], _scale_sort_key(p["scale_id"]), p["multiplier"], p["epoch"])
        )
        deltas = [p["delta_ablation_minus_baseline_bpp"] for p in pairs]
        by_scale: dict[str, Any] = {}
        for sc in sorted({p["scale_id"] for p in pairs}, key=_scale_sort_key):
            sc_d = [p["delta_ablation_minus_baseline_bpp"] for p in pairs if p["scale_id"] == sc]
            by_scale[sc] = _bootstrap_mean_ci(sc_d, n_draws=bootstrap_draws, seed=a_seed + 17)

        by_paradigm: dict[str, Any] = {}
        for par in sorted({p["paradigm"] for p in pairs}):
            par_d = [p["delta_ablation_minus_baseline_bpp"] for p in pairs if p["paradigm"] == par]
            by_paradigm[par] = _bootstrap_mean_ci(par_d, n_draws=bootstrap_draws, seed=a_seed + 29)

        return {
            "n_paired_observations": len(pairs),
            "overall_delta_bits_per_param": _bootstrap_mean_ci(
                deltas, n_draws=bootstrap_draws, seed=a_seed
            ),
            "max_abs_diff_bits_per_param": (
                float(max(p["abs_diff_bpp"] for p in pairs)) if pairs else None
            ),
            "by_scale": by_scale,
            "by_paradigm": by_paradigm,
            "pairs": pairs,
        }

    # Horizon ablation: compare dedicated run at its terminal max_epochs (e.g. 10 or 50)
    # against the single long-horizon primary run evaluated at that same intermediate epoch.
    horizon_pairs: list[dict[str, Any]] = []
    for row in all_obs:
        if row["track"] != "ablation_lr_horizon":
            continue
        if int(row["epoch"]) != int(row["max_epochs"]):
            continue
        key = (
            row["paradigm"],
            row["scale_id"],
            round(float(row["multiplier"]), 6),
            int(row["epoch"]),
        )
        base = primary_map.get(key)
        if base is None:
            continue
        c_ded = float(row["bits_per_non_emb_param"])
        c_traj = float(base["bits_per_non_emb_param"])
        m_ded = float(row["memorized_bits"])
        m_traj = float(base["memorized_bits"])
        horizon_pairs.append(
            {
                "paradigm": row["paradigm"],
                "scale_id": row["scale_id"],
                "multiplier": float(row["multiplier"]),
                "horizon_epoch": int(row["epoch"]),
                "dedicated_max_epochs": int(row["max_epochs"]),
                "single_trajectory_max_epochs": int(base["max_epochs"]),
                "dedicated_run_id": row["run_id"],
                "trajectory_run_id": base["run_id"],
                "dedicated_bits_per_non_emb_param": c_ded,
                "trajectory_checkpoint_bits_per_non_emb_param": c_traj,
                "dedicated_minus_trajectory_bpp": c_ded - c_traj,
                "ratio_dedicated_over_trajectory": (
                    m_ded / m_traj if m_traj > 1e-6 else None
                ),
            }
        )

    horizon_pairs.sort(
        key=lambda p: (
            p["paradigm"],
            _scale_sort_key(p["scale_id"]),
            p["horizon_epoch"],
            p["multiplier"],
        )
    )
    by_horizon: dict[str, Any] = {}
    for he in sorted({p["horizon_epoch"] for p in horizon_pairs}):
        hd = [
            p["dedicated_minus_trajectory_bpp"]
            for p in horizon_pairs
            if p["horizon_epoch"] == he
        ]
        by_horizon[str(he)] = _bootstrap_mean_ci(
            hd, n_draws=bootstrap_draws, seed=seed + 950 + he
        )

    return {
        "weight_decay_0_vs_0p01": _pair_standard_track("ablation_wd", seed + 801),
        "precision_bf16_vs_fp32_strict": _pair_standard_track("ablation_fp32", seed + 851),
        "cosine_single_run_vs_dedicated_horizon": {
            "n_paired_horizons": len(horizon_pairs),
            "overall_dedicated_minus_trajectory_bpp": _bootstrap_mean_ci(
                [p["dedicated_minus_trajectory_bpp"] for p in horizon_pairs],
                n_draws=bootstrap_draws,
                seed=seed + 901,
            ),
            "by_horizon_epoch": by_horizon,
            "pairs": horizon_pairs,
        },
    }


def _deduplicate_primary_runs(
    runs: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate overlapping primary/WSD runs per (paradigm, scale_id, multiplier).

    Prefers stage1_wsd over legacy Batch-1 cosine runs, and longer max_epochs
    (e.g. 25k asymptotic probes over 4k base pilots) so each cell uses the highest-horizon
    authoritative trajectory. Non-primary ablation tracks are preserved verbatim.
    """
    best_primary: dict[tuple[str, str, float], dict[str, Any]] = {}
    non_primary: list[dict[str, Any]] = []
    n_superseded = 0

    def _rank(run: dict[str, Any]) -> tuple[int, int, str]:
        return (
            1 if str(run.get("track")) == "stage1_wsd" else 0,
            int(run.get("max_epochs", 0)),
            str(run.get("run_id", "")),
        )

    for run in runs:
        if run.get("track") not in PRIMARY_TRACKS:
            non_primary.append(run)
            continue
        key = (
            str(run["paradigm"]),
            str(run["scale_id"]),
            round(float(run["multiplier"]), 6),
        )
        prev = best_primary.get(key)
        if prev is None:
            best_primary[key] = run
        elif _rank(run) > _rank(prev):
            best_primary[key] = run
            n_superseded += 1
        else:
            n_superseded += 1

    merged = sorted(
        best_primary.values(),
        key=lambda r: (str(r["paradigm"]), _scale_sort_key(str(r["scale_id"])), float(r["multiplier"])),
    ) + non_primary
    return merged, n_superseded


def summarize_memory_capacity_campaign(
    runs_dir: Path,
    *,
    extra_runs_dirs: Sequence[Path] | None = None,
    out_path: Path | None = None,
    bootstrap_draws: int = 2000,
    seed: int = 20261127,
) -> dict[str, Any]:
    """Aggregate completed run receipts into the canonical parametric capacity summary."""
    all_dirs = [Path(runs_dir)] + [Path(d) for d in (extra_runs_dirs or ())]
    raw_completed: list[dict[str, Any]] = []
    n_sidecars = 0
    n_incomplete = 0
    n_ignored = 0
    resolved_names: list[str] = []
    for d in all_dirs:
        runs_i, prov_i = _load_completed_runs(d)
        raw_completed.extend(runs_i)
        n_sidecars += int(prov_i["n_sidecars_verified"])
        n_incomplete += int(prov_i["n_incomplete_runs_skipped"])
        n_ignored += int(prov_i["n_non_run_files_ignored"])
        resolved_names.append(str(prov_i["resolved_runs_dir"]))

    if not raw_completed:
        raise ValueError(
            f"No completed {sorted(ACCEPTED_RUN_SCHEMAS)} receipts found in {all_dirs}"
        )

    completed_runs, n_superseded = _deduplicate_primary_runs(raw_completed)
    provenance = {
        "resolved_runs_dir": resolved_names[0] if len(resolved_names) == 1 else ",".join(resolved_names),
        "n_source_dirs": len(all_dirs),
        "n_completed_runs_raw": len(raw_completed),
        "n_superseded_primary_runs": n_superseded,
        "n_completed_runs": len(completed_runs),
        "n_sidecars_verified": n_sidecars,
        "n_incomplete_runs_skipped": n_incomplete,
        "n_non_run_files_ignored": n_ignored,
    }

    all_obs = _flatten_observations(completed_runs)
    primary_obs = [r for r in all_obs if r["track"] in PRIMARY_TRACKS]

    saturation_curves, all_cells = _build_saturation_and_cells(primary_obs)
    c_max_calibration, exposure_discount_curves = _build_c_max_calibration_and_exposure_laws(
        all_cells,
        bootstrap_draws=bootstrap_draws,
        seed=seed,
    )
    ar_vs_mlm_comparison = _build_ar_vs_mlm_and_loo_audit(
        all_obs,
        bootstrap_draws=bootstrap_draws,
        seed=seed,
    )
    ablations = _build_ablations(
        all_obs,
        bootstrap_draws=bootstrap_draws,
        seed=seed,
    )

    tracks_counts: dict[str, int] = {}
    paradigm_counts: dict[str, int] = {}
    scale_counts: dict[str, int] = {}
    for run in completed_runs:
        tracks_counts[run["track"]] = tracks_counts.get(run["track"], 0) + 1
        paradigm_counts[run["paradigm"]] = paradigm_counts.get(run["paradigm"], 0) + 1
        scale_counts[run["scale_id"]] = scale_counts.get(run["scale_id"], 0) + 1

    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "status": "complete",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_draws": int(bootstrap_draws),
        "seed": int(seed),
        "provenance": {
            **provenance,
            "runs_by_track": tracks_counts,
            "runs_by_paradigm": paradigm_counts,
            "runs_by_scale": {
                k: scale_counts[k] for k in sorted(scale_counts, key=_scale_sort_key)
            },
            "completed_run_ids": sorted(str(r["run_id"]) for r in completed_runs),
        },
        "saturation_curves_by_scale_exposure": saturation_curves,
        "c_max_calibration": c_max_calibration,
        "exposure_discount_curves": exposure_discount_curves,
        "ar_vs_mlm_comparison": ar_vs_mlm_comparison,
        "ablations": ablations,
    }

    assert_no_forbidden_infra_strings(summary, label="parametric_capacity_summary")
    serialized = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    match = FORBIDDEN_INFRA_RE.search(serialized)
    if match:
        raise ValueError(
            f"Refusing to emit parametric capacity summary containing forbidden infra token: "
            f"{match.group(0)!r}"
        )

    if out_path is not None:
        write_json_with_sha256(Path(out_path), summary)

    return summary
