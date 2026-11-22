"""Shared IO, metadata tables, and paired cluster-bootstrap utilities."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.io import sha256_file, write_json

LN2 = math.log(2.0)

# Approximate parameter counts (in total parameters) for all 33 panel arms
# plus planned Track A/B extension checkpoints so B1 re-runs automatically as
# new scores land.
MODEL_PARAM_COUNTS: dict[str, float] = {
    # ProGen2 ladder
    "progen2-small": 151e6,
    "progen2-medium": 764e6,
    "progen2-base": 764e6,
    "progen2-large": 2.7e9,
    "progen2-xlarge": 6.4e9,
    # ProGen3 ladder
    "progen3-112m": 112e6,
    "progen3-219m": 219e6,
    "progen3-339m": 339e6,
    "progen3-350m": 350e6,
    "progen3-762m": 762e6,
    "progen3-1b": 1.0e9,
    "progen3-3b": 3.0e9,
    "progen3-11b": 11.0e9,
    # ProtGPT3 ladder
    "protgpt3-112m": 112e6,
    "protgpt3-112m-dpo": 112e6,
    "protgpt3-1.3b": 1.3e9,
    "protgpt3-1.3b-dpo": 1.3e9,
    "protgpt3-10b": 10.0e9,
    "protgpt3-10b-dpo": 10.0e9,
    # Other protein decoders
    "protgpt2": 738e6,
    "proteinglm-7b-clm": 7.0e9,
    "rita-xl": 1.2e9,
    "instructprotein": 1.3e9,
    "prollama": 7.0e9,
    "prollama-stage-1": 7.0e9,
    # Galactica ladder
    "galactica-125m": 125e6,
    "galactica-1.3b": 1.3e9,
    "galactica-6.7b": 6.7e9,
    "galactica-30b": 30.0e9,
    # GPT-2 ladder
    "gpt2": 124e6,
    "gpt2-medium": 355e6,
    "gpt2-large": 774e6,
    "gpt2-xl": 1.558e9,
    "dialogpt-small": 124e6,
    # ByGPT5 ladder
    "bygpt5-small-en": 54e6,
    "bygpt5-base-en": 117e6,
    "bygpt5-medium-en": 292e6,
    # Modern text LLMs
    "qwen2.5-0.5b": 494e6,
    "qwen2.5-0.5b-instruct": 494e6,
    "qwen2.5-3b": 3.09e9,
    "qwen2.5-7b": 7.62e9,
    "qwen2.5-32b": 32.8e9,
    "qwen3-8b-base": 8.19e9,
    "llama-3.2-3b": 3.21e9,
    "llama-2-7b": 6.74e9,
    # Masked / Diffusion / Structure-aware extension ladders
    "esm2-8m": 8e6,
    "esm2-35m": 35e6,
    "esm2-150m": 150e6,
    "esm2-650m": 650e6,
    "esm2-3b": 3.0e9,
    "esm2-15b": 15.0e9,
    "dplm-150m": 150e6,
    "dplm-650m": 650e6,
    "dplm-3b": 3.0e9,
    "esmc-300m": 300e6,
    "esmc-600m": 600e6,
    "esmc-6b": 6.0e9,
    "amplify-120m": 120e6,
    "amplify-350m": 350e6,
    "prime-690m": 690e6,
    "proteinglm-1b-mlm": 1.0e9,
    "proteinglm-3b-mlm": 3.0e9,
    "proteinglm-10b-mlm": 10.0e9,
    "saprot-35m": 35e6,
    "saprot-650m": 650e6,
    "saprot-1.3b": 1.3e9,
    "esm3-1.4b": 1.4e9,
    "dayhoff-170m": 170e6,
    "dayhoff-3b": 3.0e9,
}

# Canonical scaling ladders evaluated automatically whenever >= 2 rungs exist.
SCALING_LADDERS: dict[str, list[str]] = {
    "progen2": ["progen2-small", "progen2-medium", "progen2-large", "progen2-xlarge"],
    "progen3": [
        "progen3-112m",
        "progen3-219m",
        "progen3-339m",
        "progen3-350m",
        "progen3-762m",
        "progen3-1b",
        "progen3-3b",
        "progen3-11b",
    ],
    "protgpt3": ["protgpt3-112m", "protgpt3-1.3b", "protgpt3-10b"],
    "galactica": ["galactica-125m", "galactica-1.3b", "galactica-6.7b", "galactica-30b"],
    "gpt2": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    "bygpt5": ["bygpt5-small-en", "bygpt5-base-en", "bygpt5-medium-en"],
    "qwen2.5": ["qwen2.5-0.5b", "qwen2.5-7b", "qwen2.5-32b"],
    "esm2": ["esm2-8m", "esm2-35m", "esm2-150m", "esm2-650m", "esm2-3b", "esm2-15b"],
    "dplm": ["dplm-150m", "dplm-650m", "dplm-3b"],
    "esmc": ["esmc-300m", "esmc-600m", "esmc-6b"],
    "amplify": ["amplify-120m", "amplify-350m"],
    "proteinglm_mlm": ["proteinglm-1b-mlm", "proteinglm-3b-mlm", "proteinglm-10b-mlm"],
    "saprot": ["saprot-35m", "saprot-650m", "saprot-1.3b"],
    "dayhoff": ["dayhoff-170m", "dayhoff-3b"],
}

MODEL_MODALITY: dict[str, str] = {
    "progen2-small": "protein_aa",
    "progen2-medium": "protein_aa",
    "progen2-base": "protein_aa",
    "progen2-large": "protein_aa",
    "progen2-xlarge": "protein_aa",
    "progen3-112m": "protein_aa",
    "progen3-219m": "protein_aa",
    "progen3-339m": "protein_aa",
    "progen3-762m": "protein_aa",
    "progen3-1b": "protein_aa",
    "progen3-3b": "protein_aa",
    "protgpt2": "protein_bpe",
    "protgpt3-112m": "protein_aa",
    "protgpt3-1.3b": "protein_aa",
    "protgpt3-10b": "protein_aa",
    "proteinglm-7b-clm": "protein_aa",
    "rita-xl": "protein_aa",
    "instructprotein": "joint_protein_text",
    "prollama": "joint_protein_text",
    "prollama-stage-1": "joint_protein_text",
    "galactica-125m": "joint_scientific",
    "galactica-1.3b": "joint_scientific",
    "galactica-6.7b": "joint_scientific",
    "galactica-30b": "joint_scientific",
    "gpt2": "text_bpe",
    "gpt2-medium": "text_bpe",
    "gpt2-large": "text_bpe",
    "gpt2-xl": "text_bpe",
    "dialogpt-small": "text_bpe",
    "bygpt5-small-en": "text_byte",
    "bygpt5-base-en": "text_byte",
    "bygpt5-medium-en": "text_byte",
    "qwen2.5-0.5b": "text_bpe",
    "qwen2.5-0.5b-instruct": "text_bpe",
    "qwen2.5-3b": "text_bpe",
    "qwen2.5-7b": "text_bpe",
    "qwen2.5-32b": "text_bpe",
    "qwen3-8b-base": "text_bpe",
    "llama-3.2-3b": "text_bpe",
    "llama-2-7b": "text_bpe",
    "esm2-8m": "protein_mlm_aa",
    "esm2-35m": "protein_mlm_aa",
    "esm2-150m": "protein_mlm_aa",
    "esm2-650m": "protein_mlm_aa",
    "esm2-3b": "protein_mlm_aa",
    "esm2-15b": "protein_mlm_aa",
    "dplm-150m": "protein_diffusion_aa",
    "dplm-650m": "protein_diffusion_aa",
    "dplm-3b": "protein_diffusion_aa",
    "esmc-300m": "protein_mlm_aa",
    "esmc-600m": "protein_mlm_aa",
    "esmc-6b": "protein_mlm_aa",
    "amplify-120m": "protein_mlm_aa",
    "amplify-350m": "protein_mlm_aa",
    "prime-690m": "protein_mlm_aa",
    "proteinglm-1b-mlm": "protein_mlm_aa",
    "proteinglm-3b-mlm": "protein_mlm_aa",
    "proteinglm-10b-mlm": "protein_mlm_aa",
    "saprot-35m": "protein_3di_aa",
    "saprot-650m": "protein_3di_aa",
    "saprot-1.3b": "protein_3di_aa",
    "esm3-1.4b": "protein_multimodal",
}


def write_json_with_sha256(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically write JSON (rejecting NaN/Inf) and an adjacent .sha256 file."""
    target = Path(path)
    write_json(target, payload)
    digest = sha256_file(target)
    sidecar = target.with_name(target.name + ".sha256")
    tmp = sidecar.with_name(f".{sidecar.name}.tmp")
    tmp.write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    tmp.replace(sidecar)
    return digest


def load_json_file(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize_finite(value: float, label: str = "value") -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return out


def cluster_bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    n_draws: int = 2000,
    seed: int = 20261120,
) -> dict[str, Any]:
    """Bootstrap mean and percentile 95% CI over independent clusters."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("cluster_bootstrap_mean requires a non-empty 1D array")
    if not np.isfinite(arr).all():
        raise ValueError("cluster_bootstrap_mean received non-finite values")

    n = int(arr.size)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != arr.shape or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("Invalid cluster bootstrap weights")
        point = float(np.sum(arr * w) / np.sum(w))
    else:
        w = None
        point = float(np.mean(arr))

    if n < 2:
        return {
            "point": point,
            "ci95": [point, point],
            "se": 0.0,
            "n_clusters": n,
            "n_draws": 0,
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_draws, n))
    sampled = arr[idx]
    if w is not None:
        w_s = w[idx]
        draws = np.sum(sampled * w_s, axis=1) / np.sum(w_s, axis=1)
    else:
        draws = np.mean(sampled, axis=1)

    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "point": float(point),
        "ci95": [float(lo), float(hi)],
        "se": float(np.std(draws, ddof=1)),
        "n_clusters": n,
        "n_draws": int(n_draws),
    }


def paired_cluster_bootstrap_diff(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    n_draws: int = 2000,
    seed: int = 20261120,
) -> dict[str, Any]:
    """Paired cluster bootstrap for (a - b) with exact aligned resamples."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape or arr_a.ndim != 1 or arr_a.size == 0:
        raise ValueError("paired_cluster_bootstrap_diff requires matching non-empty 1D arrays")
    res = cluster_bootstrap_mean(arr_a - arr_b, weights=weights, n_draws=n_draws, seed=seed)
    res["mean_a"] = float(
        np.average(arr_a, weights=weights) if weights is not None else np.mean(arr_a)
    )
    res["mean_b"] = float(
        np.average(arr_b, weights=weights) if weights is not None else np.mean(arr_b)
    )
    return res


def cluster_bootstrap_scaling_slope(
    log10_params: Sequence[float] | np.ndarray,
    loss_by_rung_and_cluster: np.ndarray,
    *,
    cluster_weights: np.ndarray | None = None,
    n_draws: int = 2000,
    seed: int = 20261120,
) -> dict[str, Any]:
    """Compute scaling slope -dH / d(log10 N) with family-cluster bootstrap 95% CI.

    Args:
        log10_params: shape (R,) log10 parameter counts for R >= 2 rungs.
        loss_by_rung_and_cluster: shape (R, C) cross-entropy (bits/res) per rung & cluster.
        cluster_weights: optional shape (C,) non-negative weights per cluster.
    """
    x = np.asarray(log10_params, dtype=np.float64)
    y = np.asarray(loss_by_rung_and_cluster, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] != x.size or x.size < 2 or y.shape[1] < 2:
        raise ValueError("Scaling slope requires R >= 2 rungs and C >= 2 clusters")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Non-finite inputs to cluster_bootstrap_scaling_slope")

    x_c = x - np.mean(x)
    denom = float(np.sum(x_c ** 2))
    if denom <= 0.0:
        raise ValueError("All scaling rungs have identical parameter counts")

    n_rungs, n_clusters = y.shape
    if cluster_weights is not None:
        w = np.asarray(cluster_weights, dtype=np.float64)
        w_norm = w / np.sum(w)
        y_mean = y @ w_norm
    else:
        w = None
        y_mean = np.mean(y, axis=1)

    # Positive slope means loss DECREASES with scale: -dH / d(log10 N)
    slope_point = float(-np.sum(x_c * (y_mean - np.mean(y_mean))) / denom)
    # R^2 of linear fit on mean curve
    y_hat = np.mean(y_mean) - slope_point * x_c
    ss_tot = float(np.sum((y_mean - np.mean(y_mean)) ** 2))
    ss_res = float(np.sum((y_mean - y_hat) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(n_draws, n_clusters))
    if w is not None:
        w_s = w[idx]  # (D, C)
        w_s = w_s / np.sum(w_s, axis=1, keepdims=True)
        # y[:, idx] has shape (R, D, C)
        y_draws = np.einsum("rdc,dc->dr", y[:, idx], w_s)  # (D, R)
    else:
        y_draws = np.mean(y[:, idx], axis=2).T  # (D, R)

    y_draws_c = y_draws - np.mean(y_draws, axis=1, keepdims=True)
    slopes = -(y_draws_c @ x_c) / denom
    lo, hi = np.quantile(slopes, [0.025, 0.975])

    return {
        "slope_bits_per_decade": slope_point,
        "ci95": [float(lo), float(hi)],
        "se": float(np.std(slopes, ddof=1)),
        "r2": r2,
        "n_rungs": int(n_rungs),
        "n_clusters": int(n_clusters),
        "n_draws": int(n_draws),
        "mean_loss_by_rung": [float(v) for v in y_mean],
        "delta_endpoints_bits": float(y_mean[0] - y_mean[-1]),
    }
