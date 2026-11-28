"""Open-Source Foundation PLM Quantitative Physical Exam (`open_model_audit`).

Implements Temporal + Fold-Disjoint + Family-Bucketed evaluation of open-source
protein language models across four architectural families (Causal AR, Masked LM,
Discrete Diffusion / MDM, and Structure-Conditioned / Multi-Track PLMs), aligned
to the information supply-demand framework in ``docs/foundational-plm-theory.md``:

- Strict 20-canonical-AA logit renormalization ($H_0 = \\log_2 20 \\approx 4.321928$ bits/res)
- Horvitz-Thompson corpus bits explained $\\hat{B}_{\\mathrm{total}}(\\theta)$ and
  apparent capacity $\\hat{c}_{\\mathrm{app}}(\\theta) = \\hat{B}_{\\mathrm{total}} / P_{\\mathrm{non\\_emb}}$
- RHS three-tier absorbed bit decomposition Eq. (3a):
  $\\Delta h_{\\mathrm{univ}}, \\Delta h_{\\mathrm{spec}}, \\Delta h_{\\mathrm{res}}$
- Temperature-scaled intrinsic entropy & calibration error Eq. (3b):
  $H_{\\mathrm{cal}} = \\min_{T > 0} H(\\theta, T)$ and $\\Delta_{\\mathrm{cal}}(\\theta)$
- Multi-track / structure-conditioned external supply decomposition:
  $b_{\\mathrm{ext}}(Z) = H(X_{\\mathrm{seq}} \\mid Z=\\varnothing) - H(X_{\\mathrm{seq}} \\mid Z_{\\mathrm{3D}})$
"""

from __future__ import annotations

from .metrics import (
    AA20,
    AA20_INDEX,
    AA20_SET,
    DEPTH_BUCKETS,
    H0_BITS,
    LN2,
    assign_depth_bucket,
    aa20_log_probs,
    aa20_restricted_logits,
    compute_horvitz_thompson_corpus_estimate,
    compute_multitrack_b_ext,
    compute_temperature_calibrated_bits,
    compute_three_tier_decomposition,
    count_model_parameters,
    deterministic_fixed_rate_mask,
    deterministic_loo_positions,
    deterministic_mdm_stratum_mask,
)

__all__ = [
    "AA20",
    "AA20_INDEX",
    "AA20_SET",
    "DEPTH_BUCKETS",
    "H0_BITS",
    "LN2",
    "assign_depth_bucket",
    "aa20_log_probs",
    "aa20_restricted_logits",
    "compute_horvitz_thompson_corpus_estimate",
    "compute_multitrack_b_ext",
    "compute_temperature_calibrated_bits",
    "compute_three_tier_decomposition",
    "count_model_parameters",
    "deterministic_fixed_rate_mask",
    "deterministic_loo_positions",
    "deterministic_mdm_stratum_mask",
]
