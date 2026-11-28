"""Information-theoretic metrics, strict AA20 renormalization, and deterministic masking.

All notation follows ``docs/foundational-plm-theory.md``:
  - $H_0 = \\log_2 20 = 4.321928094887363$ bits/res (strict 20 canonical amino acids)
  - Eq. (3a) three-tier conservation:
      $\\Delta h_{\\mathrm{univ}} = H_0 - H_{\\mathrm{unseen\\_fold}}$
      $\\Delta h_{\\mathrm{spec}} = H_{\\mathrm{unseen\\_fold}} - H_{\\mathrm{unseen\\_homolog}}$
      $\\Delta h_{\\mathrm{res}}  = H_{\\mathrm{unseen\\_homolog}} - H_{\\mathrm{seen}}$
  - Eq. (3b) temperature scaling:
      $H_{\\mathrm{cal}}(\\theta) = \\min_{T > 0} H(\\theta, T)$
      $\\Delta_{\\mathrm{cal}}(\\theta) = H(\\theta, 1) - H_{\\mathrm{cal}}(\\theta) \\ge 0$
  - Horvitz-Thompson corpus integral over 5 family/cluster-depth buckets:
      $\\hat{B}_{\\mathrm{total}}(\\theta) = \\sum_b T_b \\, \\Delta \\hat{h}_b(\\theta)$
      $\\hat{c}_{\\mathrm{app}}(\\theta) = \\hat{B}_{\\mathrm{total}}(\\theta) / P_{\\mathrm{non\\_emb}}$
  - External conditioning supply $B_{\\mathrm{ext}}(Z)$ (per-residue $b_{\\mathrm{ext}}(Z)$):
      $b_{\\mathrm{ext}}(Z) = H(X_{\\mathrm{seq}} \\mid Z=\\varnothing) - H(X_{\\mathrm{seq}} \\mid Z_{\\mathrm{3D}})$
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.capability.core.amino_acids import AA20

AA20_INDEX: dict[str, int] = {aa: idx for idx, aa in enumerate(AA20)}
AA20_SET: frozenset[str] = frozenset(AA20)
LN2: float = math.log(2.0)
H0_BITS: float = math.log2(20.0)  # 4.321928094887363 bits/res

#: Canonical 5 family/cluster-depth buckets over UniRef50 cluster member count `n=d`.
DEPTH_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("singleton", 1, 1),
    ("small", 2, 16),
    ("medium", 17, 128),
    ("deep", 129, 1024),
    ("ultra_deep", 1025, None),
)
DEPTH_BUCKET_ORDER: tuple[str, ...] = tuple(name for name, _, _ in DEPTH_BUCKETS)

DEFAULT_MLM15_DRAWS: int = 8
DEFAULT_LOO_PROBES: int = 16
DEFAULT_MDM_STRATA: int = 16


def assign_depth_bucket(cluster_size: int) -> str:
    """Map UniRef50 cluster depth ``d = n`` into one of the 5 canonical buckets."""
    d = int(cluster_size)
    if d < 1:
        raise ValueError(f"cluster_size must be >= 1, got {cluster_size}")
    for name, lo, hi in DEPTH_BUCKETS:
        if d >= lo and (hi is None or d <= hi):
            return name
    raise RuntimeError(f"Unreachable cluster_size={d}")


# --------------------------------------------------------------------------- #
# 1. Strict 20-AA Logit Renormalization & Temperature Scaling Eq. (3b)
# --------------------------------------------------------------------------- #


def aa20_restricted_logits(
    logits: torch.Tensor,
    aa20_ids: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Slice arbitrary-vocabulary logits strictly onto the 20 canonical AA token IDs.

    Strips `<eos>`, `<pad>`, `<bos>`, `X`, `B`, `Z`, `U`, `O`, and any non-AA tokens so
    every tokenizer is compared on the exact same 20-class simplex ($H_0 = \\log_2 20$).
    """
    if isinstance(aa20_ids, torch.Tensor):
        idx = aa20_ids.to(device=logits.device, dtype=torch.long)
    else:
        idx = torch.as_tensor(list(aa20_ids), device=logits.device, dtype=torch.long)
    if idx.shape != (20,):
        raise ValueError(f"aa20_ids must have shape (20,), got {tuple(idx.shape)}")
    return logits.index_select(-1, idx).float()


def aa20_log_probs(
    logits: torch.Tensor,
    aa20_ids: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Return normalized log-probabilities strictly over the 20 canonical amino acids."""
    return F.log_softmax(aa20_restricted_logits(logits, aa20_ids), dim=-1)


@torch.no_grad()
def compute_temperature_calibrated_bits(
    logits_20: torch.Tensor | np.ndarray,
    targets_20: torch.Tensor | np.ndarray,
    weights: torch.Tensor | np.ndarray | None = None,
    *,
    log_t_bounds: tuple[float, float] = (-4.0, 6.0),
    n_iter: int = 36,
) -> dict[str, float]:
    """Compute Eq. (3b) raw vs scalar-temperature-calibrated cross-entropy in ``bits/res``.

    Solves $H_{\\mathrm{cal}}(\\theta) = \\min_{T > 0} H(\\theta, T)$ via exact 1D golden-section
    search on the strictly convex objective $\\log T \\mapsto \\mathrm{CE}(z / T, y) / \\ln 2$.
    Optional non-negative ``weights`` (e.g. OA/MDM stratum weights $1 / m_{i,s}$) yield
    exact weighted NELBO temperature calibration.
    """
    z = (
        logits_20.detach().float()
        if isinstance(logits_20, torch.Tensor)
        else torch.as_tensor(logits_20, dtype=torch.float32)
    )
    y = (
        targets_20.detach().long()
        if isinstance(targets_20, torch.Tensor)
        else torch.as_tensor(targets_20, dtype=torch.long)
    )
    z = z.reshape(-1, 20)
    y = y.reshape(-1)
    if z.shape[0] != y.shape[0]:
        raise ValueError(f"logits_20 rows {z.shape[0]} != targets_20 len {y.shape[0]}")

    if y.numel() == 0:
        return {
            "raw_bits_per_res": float("nan"),
            "calibrated_bits_per_res": float("nan"),
            "optimal_temperature": float("nan"),
            "delta_cal_bits": float("nan"),
            "n_scored_positions": 0,
        }

    if weights is not None:
        w = (
            weights.detach().to(device=z.device, dtype=torch.float64)
            if isinstance(weights, torch.Tensor)
            else torch.as_tensor(weights, device=z.device, dtype=torch.float64)
        ).reshape(-1)
        if w.shape[0] != y.shape[0]:
            raise ValueError(f"weights len {w.shape[0]} != targets len {y.shape[0]}")
        w_sum = float(w.sum().item())
        if w_sum <= 0.0:
            raise ValueError("Sum of weights must be positive")
        w = w / w_sum
    else:
        w = None

    # Center logits per row for numerical stability across extreme temperatures.
    z = z - z.max(dim=-1, keepdim=True).values
    y = y.to(device=z.device)

    def _bits_at_log_t(log_t: float) -> float:
        inv_t = math.exp(-float(log_t))
        ce = F.cross_entropy(z * inv_t, y, reduction="none").to(torch.float64)
        val = float((ce * w).sum().item()) if w is not None else float(ce.mean().item())
        return val / LN2

    raw_bits = _bits_at_log_t(0.0)

    a, b = float(log_t_bounds[0]), float(log_t_bounds[1])
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc = _bits_at_log_t(c)
    fd = _bits_at_log_t(d)
    for _ in range(int(n_iter)):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = _bits_at_log_t(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = _bits_at_log_t(d)

    best_log_t = 0.5 * (a + b)
    cand_bits = _bits_at_log_t(best_log_t)
    if cand_bits < raw_bits:
        cal_bits = cand_bits
        opt_t = math.exp(best_log_t)
    else:
        cal_bits = raw_bits
        opt_t = 1.0

    return {
        "raw_bits_per_res": float(raw_bits),
        "calibrated_bits_per_res": float(cal_bits),
        "optimal_temperature": float(opt_t),
        "delta_cal_bits": float(max(0.0, raw_bits - cal_bits)),
        "n_scored_positions": int(y.numel()),
    }


# --------------------------------------------------------------------------- #
# 2. Deterministic SplitMix64 Masking (Batch-Invariant, Matching Stage-1/2)
# --------------------------------------------------------------------------- #


def _u64_to_i64(val: int) -> int:
    val &= (1 << 64) - 1
    return val - (1 << 64) if val >= (1 << 63) else val


_SM_C0 = _u64_to_i64(0x9E3779B97F4A7C15)
_SM_C1 = _u64_to_i64(0xD1B54A32D192ED03)
_SM_C2 = _u64_to_i64(0x94D049BB133111EB)
_SM_M1 = _u64_to_i64(0xBF58476D1CE4E5B9)
_SM_M2 = _u64_to_i64(0x94D049BB133111EB)


def _splitmix64_priorities_np(seq_idx: int, draw_idx: int, seq_len: int, seed: int) -> np.ndarray:
    """Deterministic uint64 priorities of shape ``(seq_len,)`` for ``(seq_idx, draw_idx)``."""
    mask64 = (1 << 64) - 1
    base = (
        (int(seed) & mask64)
        + ((int(seq_idx) & mask64) * 0x9E3779B97F4A7C15)
        + ((int(draw_idx) & mask64) * 0x517CC1B727220A95)
        + 0x94D049BB133111EB
    ) & mask64
    pos = np.arange(int(seq_len), dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(base) + pos * np.uint64(0xD1B54A32D192ED03)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def deterministic_fixed_rate_mask(
    seq_idx: int,
    seq_len: int,
    *,
    draw_idx: int,
    mask_rate: float = 0.15,
    eval_seed: int = 20261141,
) -> np.ndarray:
    """Deterministic boolean mask of shape ``(seq_len,)`` masking ``round(mask_rate * L)`` sites."""
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    k_mask = max(1, min(int(seq_len), int(round(float(seq_len) * float(mask_rate)))))
    prio = _splitmix64_priorities_np(seq_idx, draw_idx, seq_len, eval_seed)
    order = np.argsort(prio, kind="mergesort")
    mask = np.zeros(seq_len, dtype=np.bool_)
    mask[order[:k_mask]] = True
    return mask


def deterministic_mdm_stratum_mask(
    seq_idx: int,
    seq_len: int,
    *,
    stratum: int,
    n_strata: int = DEFAULT_MDM_STRATA,
    eval_seed: int = 20261129,
) -> tuple[np.ndarray, int, float]:
    """Deterministic OA-ARDM / MDM stratum mask ``(mask, n_mask, u_stratum)``.

    Exact 16-stratum estimator matching ``src/capability/supply_demand/evaluation.py``:
      $u_{i,s} = (s + ((i + 7 s) \\bmod S + 0.5) / S) / S$,
      $m_{i,s} = \\mathrm{clip}(\\lfloor u_{i,s} L \\rfloor + 1, 1, L)$.
    Averaging $\\frac{1}{m_{i,s}} \\sum_{j \\in M_{i,s}} -\\log_2 p_\\theta(x_j \\mid x_{\\setminus M_{i,s}})$
    over $s = 0, \\dots, S-1$ yields an unbiased per-residue joint NELBO upper bound on
    $-\\frac{1}{L} \\log_2 p_\\theta(x)$.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if not (0 <= stratum < n_strata):
        raise ValueError(f"stratum must be in [0, {n_strata - 1}], got {stratum}")
    off = (int(seq_idx) + 7 * int(stratum)) % int(n_strata)
    u = (float(stratum) + (float(off) + 0.5) / float(n_strata)) / float(n_strata)
    m_mask = max(1, min(int(seq_len), int(math.floor(u * float(seq_len))) + 1))
    prio = _splitmix64_priorities_np(seq_idx, stratum, seq_len, eval_seed)
    order = np.argsort(prio, kind="mergesort")
    mask = np.zeros(seq_len, dtype=np.bool_)
    mask[order[:m_mask]] = True
    return mask, m_mask, u


def deterministic_loo_positions(
    seq_idx: int,
    seq_len: int,
    *,
    n_probes: int = DEFAULT_LOO_PROBES,
    eval_seed: int = 20261153,
) -> np.ndarray:
    """Select ``min(n_probes, seq_len)`` deterministic unique 0-based probe positions."""
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    k = max(1, min(int(seq_len), int(n_probes)))
    prio = _splitmix64_priorities_np(seq_idx, 0, seq_len, eval_seed)
    order = np.argsort(prio, kind="mergesort")
    return np.sort(order[:k]).astype(np.int64)


# --------------------------------------------------------------------------- #
# 3. Non-Embedding Parameter Accounting
# --------------------------------------------------------------------------- #


def count_model_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, embedding, and non-embedding parameters (deduplicating tied weights).

    Follows the standard Kaplan/Chinchilla/foundational-PLM-theory convention:
    ``P_non_emb`` excludes lookup embedding tables (``nn.Embedding``) and untied final
    vocabulary projection heads whose output dim equals a token vocabulary table,
    while reporting ``P_total``, ``P_emb``, and ``P_non_emb`` transparently.
    """
    seen_ptrs: set[int] = set()
    emb_ptrs: set[int] = set()

    for mod in model.modules():
        if isinstance(mod, nn.Embedding):
            for p in mod.parameters(recurse=False):
                emb_ptrs.add(int(p.data_ptr()))

    total = 0
    emb = 0
    for name, p in model.named_parameters():
        ptr = int(p.data_ptr())
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        n = int(p.numel())
        total += n
        low_name = name.lower()
        is_emb_name = (
            ptr in emb_ptrs
            or "embed_tokens" in low_name
            or "word_embeddings" in low_name
            or "position_embeddings" in low_name
            or "wte." in low_name
            or "wpe." in low_name
            or low_name.endswith(".w_s.weight")
        )
        if is_emb_name:
            emb += n

    non_emb = max(1, total - emb)
    return {
        "n_params_total": int(total),
        "n_params_emb": int(emb),
        "n_params_non_emb": int(non_emb),
    }


# --------------------------------------------------------------------------- #
# 4. Eq. (3a) Three-Tier Conservation, Horvitz-Thompson Corpus Integral & b_ext
# --------------------------------------------------------------------------- #


def compute_three_tier_decomposition(
    h_seen: float,
    h_unseen_homolog: float,
    h_unseen_fold: float,
    *,
    h_seen_cal: float | None = None,
    h_unseen_homolog_cal: float | None = None,
    h_unseen_fold_cal: float | None = None,
    h0_bits: float = H0_BITS,
) -> dict[str, Any]:
    """Compute exact Eq. (3a) RHS 3-tier absorbed bits and Eq. (3b) calibrated version.

    Identity:
      ``delta_h_univ + delta_h_spec + delta_h_res == H_0 - H_seen``
    """
    h_s = float(h_seen)
    h_uh = float(h_unseen_homolog)
    h_uf = float(h_unseen_fold)

    dh_univ = float(h0_bits - h_uf)
    dh_spec = float(h_uf - h_uh)
    dh_res = float(h_uh - h_s)
    dh_total = float(h0_bits - h_s)
    conservation_err = abs((dh_univ + dh_spec + dh_res) - dh_total)

    out: dict[str, Any] = {
        "H_0_bits": float(h0_bits),
        "H_seen_bits": h_s,
        "H_unseen_homolog_bits": h_uh,
        "H_unseen_fold_bits": h_uf,
        "delta_h_univ": dh_univ,
        "delta_h_spec": dh_spec,
        "delta_h_res": dh_res,
        "delta_h_total": dh_total,
        "conservation_abs_error": float(conservation_err),
    }
    if h_seen_cal is not None and h_unseen_homolog_cal is not None and h_unseen_fold_cal is not None:
        hs_c = float(h_seen_cal)
        huh_c = float(h_unseen_homolog_cal)
        huf_c = float(h_unseen_fold_cal)
        out["calibrated"] = {
            "H_seen_cal_bits": hs_c,
            "H_unseen_homolog_cal_bits": huh_c,
            "H_unseen_fold_cal_bits": huf_c,
            "delta_h_univ_cal": float(h0_bits - huf_c),
            "delta_h_spec_cal": float(huf_c - huh_c),
            "delta_h_res_cal": float(huh_c - hs_c),
            "delta_h_total_cal": float(h0_bits - hs_c),
            "delta_cal_seen": float(max(0.0, h_s - hs_c)),
            "delta_cal_unseen_homolog": float(max(0.0, h_uh - huh_c)),
            "delta_cal_unseen_fold": float(max(0.0, h_uf - huf_c)),
        }
    return out


def compute_horvitz_thompson_corpus_estimate(
    bucket_eval_stats: Mapping[str, Mapping[str, Any]],
    population_bucket_stats: Mapping[str, Mapping[str, Any]],
    *,
    n_params_non_emb: int,
    n_params_total: int | None = None,
    h0_bits: float = H0_BITS,
    population_key: str = "total_residues",
) -> dict[str, Any]:
    """Compute Horvitz-Thompson corpus bits explained and apparent ``bits/param``.

    For each family/cluster-depth bucket $b \\in \\{\\text{singleton}, \\text{small},
    \\text{medium}, \\text{deep}, \\text{ultra\\_deep}\\}$ with full-UniRef50 population
    residue count $T_b$ and sampled bucket mean entropy $\\hat{h}_b(\\theta)$ (residue-weighted):

      $\\Delta \\hat{h}_b(\\theta) = H_0 - \\hat{h}_b(\\theta)$
      $\\hat{B}_{\\mathrm{total}}(\\theta) = \\sum_b T_b \\, \\Delta \\hat{h}_b(\\theta)$
      $\\mathrm{SE}(\\hat{B}_{\\mathrm{total}}) = \\sqrt{\\sum_b T_b^2 \\, \\mathrm{Var}(\\hat{h}_b)}$
      $\\hat{c}_{\\mathrm{app}}(\\theta) = \\hat{B}_{\\mathrm{total}}(\\theta) / P_{\\mathrm{non\\_emb}}$
    """
    if n_params_non_emb <= 0:
        raise ValueError(f"n_params_non_emb must be > 0, got {n_params_non_emb}")
    p_tot = int(n_params_total) if (n_params_total is not None and n_params_total > 0) else int(n_params_non_emb)

    per_bucket: dict[str, dict[str, Any]] = {}
    b_total = 0.0
    b_var = 0.0
    t_total = 0
    b_total_cal = 0.0
    has_cal = True

    for b_name in DEPTH_BUCKET_ORDER:
        if b_name not in bucket_eval_stats or b_name not in population_bucket_stats:
            continue
        ev = bucket_eval_stats[b_name]
        pop = population_bucket_stats[b_name]
        t_b = int(pop.get(population_key, pop.get("total_residues", 0)))
        n_pop = int(pop.get("n_clusters", 0))
        h_b = float(ev["bits_per_res"])
        dh_b = float(h0_bits - h_b)
        se_h_b = float(ev.get("se_bits_per_res", 0.0))
        contrib_bits = float(t_b) * dh_b
        contrib_var = (float(t_b) * se_h_b) ** 2

        b_total += contrib_bits
        b_var += contrib_var
        t_total += t_b

        h_b_cal = ev.get("calibrated_bits_per_res")
        if h_b_cal is not None and math.isfinite(float(h_b_cal)):
            dh_b_cal = float(h0_bits - float(h_b_cal))
            b_total_cal += float(t_b) * dh_b_cal
        else:
            has_cal = False
            dh_b_cal = None

        per_bucket[b_name] = {
            "population_clusters_N_b": n_pop,
            "population_residues_T_b": t_b,
            "sample_n_sequences": int(ev.get("n_sequences", 0)),
            "sample_n_residues": int(ev.get("n_residues", 0)),
            "H_b_bits_per_res": h_b,
            "delta_h_b_bits_per_res": dh_b,
            "se_H_b_bits_per_res": se_h_b,
            "delta_h_b_cal_bits_per_res": dh_b_cal,
            "B_b_bits": contrib_bits,
            "B_b_gbits": contrib_bits / 1e9,
        }

    b_se = math.sqrt(max(0.0, b_var))
    c_app = b_total / float(n_params_non_emb)
    c_app_se = b_se / float(n_params_non_emb)
    c_app_total = b_total / float(p_tot)
    corpus_mean_dh = (b_total / float(t_total)) if t_total > 0 else float("nan")
    corpus_mean_h = (float(h0_bits) - corpus_mean_dh) if math.isfinite(corpus_mean_dh) else float("nan")

    result: dict[str, Any] = {
        "population_key": population_key,
        "population_total_residues_T": int(t_total),
        "n_params_non_emb": int(n_params_non_emb),
        "n_params_total": int(p_tot),
        "corpus_weighted_H_bits_per_res": float(corpus_mean_h),
        "corpus_weighted_delta_h_bits_per_res": float(corpus_mean_dh),
        "B_total_bits": float(b_total),
        "B_total_gbits": float(b_total / 1e9),
        "B_total_se_bits": float(b_se),
        "c_app_bits_per_non_emb_param": float(c_app),
        "c_app_se_bits_per_non_emb_param": float(c_app_se),
        "c_app_bits_per_total_param": float(c_app_total),
        "per_bucket": per_bucket,
    }
    if has_cal and per_bucket:
        result["B_total_cal_bits"] = float(b_total_cal)
        result["B_total_cal_gbits"] = float(b_total_cal / 1e9)
        result["c_app_cal_bits_per_non_emb_param"] = float(b_total_cal / float(n_params_non_emb))
    return result


def compute_multitrack_b_ext(
    h_conditioned: float,
    *,
    h_unconditioned: float | None = None,
    h_seq_reference: float | None = None,
    h0_bits: float = H0_BITS,
) -> dict[str, Any]:
    """Compute multi-track / structure-conditioned external supply $b_{\\mathrm{ext}}(Z)$ (bits/res).

    Chain-rule decomposition:
      - When the model supports internal ablation $Z = \\varnothing$ (SaProt, DPLM-2, ESM-3):
          $b_{\\mathrm{ext}}(Z) = H(X_{\\mathrm{seq}} \\mid Z=\\varnothing) - H(X_{\\mathrm{seq}} \\mid Z_{\\mathrm{3D}})$
          $H_0 - H(X_{\\mathrm{seq}} \\mid Z_{\\mathrm{3D}}) = [H_0 - H(X_{\\mathrm{seq}} \\mid Z=\\varnothing)] + b_{\\mathrm{ext}}(Z)$
      - For inverse-folding-only models (ProteinMPNN, no $Z=\\varnothing$ mode):
          Reports total bits explained given 3D backbone $H_0 - H(X_{\\mathrm{seq}} \\mid Z_{\\mathrm{3D}})$
          and optional reference-anchored $b_{\\mathrm{ext}}^{\\mathrm{ref}}(Z) = H_{\\mathrm{ref}}(X) - H(X \\mid Z_{\\mathrm{3D}})$.
    """
    h_cond = float(h_conditioned)
    total_explained = float(h0_bits - h_cond)
    out: dict[str, Any] = {
        "H_0_bits": float(h0_bits),
        "H_seq_given_Z_bits": h_cond,
        "total_bits_explained_given_Z": total_explained,
    }
    if h_unconditioned is not None and math.isfinite(float(h_unconditioned)):
        h_uncond = float(h_unconditioned)
        seq_only_explained = float(h0_bits - h_uncond)
        b_ext = float(h_uncond - h_cond)
        out.update(
            {
                "H_seq_given_null_Z_bits": h_uncond,
                "seq_only_bits_explained": seq_only_explained,
                "b_ext_bits_per_res": b_ext,
                "chain_rule_abs_error": float(abs((seq_only_explained + b_ext) - total_explained)),
                "b_ext_share_of_total_explained": (
                    float(b_ext / total_explained) if abs(total_explained) > 1e-12 else float("nan")
                ),
            }
        )
    if h_seq_reference is not None and math.isfinite(float(h_seq_reference)):
        h_ref = float(h_seq_reference)
        out["H_seq_reference_bits"] = h_ref
        out["b_ext_vs_seq_reference_bits_per_res"] = float(h_ref - h_cond)
    return out
