"""Section-IV Production Engineering Quantifier for Foundational PLM Information Supply-Demand Theory.

Implements quantitative production design rules aligned strictly to
``docs/foundational-plm-theory.md``:

- Supply side (Eq. 1, 2a, 2b):
  ``c_eff(A, O, E) <= c_inf(A, O)``, ``c_mem(A, O, E) = c_inf(A, O) * phi_mem(A, O, E)``,
  generalization window ``W(A, O) = E_mem(A, O) / E_gen(A, O)``, and external supply
  ``B_ext(Z)``.
- Demand side (Eq. 1, 3a, 3b):
  ``K_univ`` (O(1)), ``K_spec(D)`` (O(M)), ``K_res(D)`` (O(N)), task-specific useful demand
  ``K_useful(D, T)``, calibrated intrinsic entropy ``H_cal``, and calibration error ``Delta_cal``.
- Section-IV Production Criteria (Eq. 4):
  1. Empirical ``K_useful(D, T)`` scaling formula in bits as a function of family count ``M``
     and task type ``T`` (with Mechanism-2 narrow-``M`` diversity guard).
  2. Closed-form Optimal Sizing ``P*(M, O, E, Z)`` (with Section V.2 compute floor).
  3. Safe Stopping Horizon ``E_max(P, M, O, Z)`` preventing ``Delta_cal`` blowup and
     upper-layer structural overwriting.
  4. Objective Phase-Boundary Criterion ``eta_crit = c_inf(AR) * P / [K_useful - B_ext(Z)]_+``.
  5. Parameter-Equivalence of Structural Conditioning
     ``Delta P_saved(Z) = B_ext(Z) / c_eff(A, O, E)``.
- Cross-Experiment Synthesis:
  Ingests Stage-1, Stage-2 (5M & optional 25M), Layerwise Probing (43 checkpoints & optional
  trajectories), and optional Open-PLM Audit / Multimodal-3D / Supply-Ablation outputs.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .analysis import _canonical_objective, _safe_float
from .spec import assert_no_forbidden_infra_strings, objective_display_name

PRODUCTION_CRITERIA_SCHEMA: str = "plm_section4_production_criteria_v1"

#: Section III.1 reference bounds from ``docs/foundational-plm-theory.md`` (in bits).
DEFAULT_K_UNIV_BITS: float = 15_000_000.0  # 15 Mbit (midpoint of 5-30 Mbit universal physics)
DEFAULT_K_UNIV_RANGE_BITS: tuple[float, float] = (5_000_000.0, 30_000_000.0)
DEFAULT_K_FAM_PROFILE_BITS: float = 1_500.0  # 1st-order profile (~30-90 Gbit across 20M-60M clusters)
DEFAULT_K_FAM_COUPLED_BITS: float = 50_000.0  # With pairwise couplings (~1-3 Tbit across 20M-60M clusters)
DEFAULT_MIN_DIVERSITY_FAMILIES: int = 256  # Mechanism-2 narrow-M floor from Knob-2 & Knob-3
DEFAULT_ETA_CRIT: float = 1.0  # Exact Eq. (4) balance c_inf(AR) * P = [K_useful - B_ext(Z)]_+
DEFAULT_MAX_SAFE_DELTA_CAL_BITS: float = 0.5  # Onset of severe OOD miscalibration & late-layer erasure


@dataclass(frozen=True)
class ObjectiveSupplyProfile:
    """Calibrated supply-side parameters for one training objective ``O``."""

    objective: str
    display_name: str
    c_inf_bpp: float
    c_max_post_cooldown_bpp: float
    e_mem_epochs: float
    e_gen_epochs: float
    w_window: float
    exposure_epochs: tuple[float, ...]
    c_mem_envelope_bpp: tuple[float, ...]
    source: str = "stage1_stage2_calibrated"

    def __post_init__(self) -> None:
        if self.c_inf_bpp <= 0.0:
            raise ValueError(f"{self.objective}: c_inf_bpp must be > 0, got {self.c_inf_bpp}")
        if self.e_mem_epochs <= 0.0 or self.e_gen_epochs <= 0.0:
            raise ValueError(
                f"{self.objective}: e_mem_epochs and e_gen_epochs must be > 0, "
                f"got ({self.e_mem_epochs}, {self.e_gen_epochs})"
            )
        if len(self.exposure_epochs) != len(self.c_mem_envelope_bpp):
            raise ValueError(f"{self.objective}: exposure_epochs and c_mem_envelope_bpp lengths differ")

    def c_mem_at_epoch(self, epochs: float) -> float:
        """Evaluate monotone residual-memorization supply ``c_mem(O, E)`` in bits/param (Eq. 2a)."""
        e = float(epochs)
        if e <= 0.0:
            return 0.0
        if not self.exposure_epochs:
            # Smooth delayed-Hill fallback when no discrete curve is attached
            if e <= self.e_mem_epochs:
                return 0.0
            ratio = (e - self.e_mem_epochs) / max(self.e_mem_epochs, 1.0)
            return float(self.c_inf_bpp * (ratio / (1.0 + ratio)))

        es = self.exposure_epochs
        cs = self.c_mem_envelope_bpp
        if e <= es[0]:
            return float(cs[0] * (e / es[0]))
        if e >= es[-1]:
            return float(max(cs[-1], 0.0))
        log_e = math.log(e)
        for i in range(1, len(es)):
            e0, e1 = es[i - 1], es[i]
            if e0 <= e <= e1:
                if e1 <= e0:
                    return float(cs[i])
                frac = (log_e - math.log(e0)) / (math.log(e1) - math.log(e0))
                return float(cs[i - 1] + frac * (cs[i] - cs[i - 1]))
        return float(cs[-1])

    def c_eff_at_epoch(self, epochs: float, *, include_useful_floor: bool = False) -> float:
        """Evaluate effective capacity ``c_eff(O, E)`` in bits/param at epoch ``E``.

        For memorization-capacity queries (default), returns ``c_mem(O, E)`` capped at
        ``c_max_post_cooldown_bpp``. When ``include_useful_floor=True`` (for sizing on
        useful shared structure written over ``1..E_gen`` epochs), smoothly ramps toward
        ``c_inf_bpp`` across ``E_gen`` and returns ``max(c_mem(O, E), c_useful(O, E))``.
        """
        e = float(epochs)
        if e <= 0.0:
            return 0.0
        c_mem = self.c_mem_at_epoch(e)
        if not include_useful_floor:
            return c_mem
        # Shared physics/family regularities write rapidly over E_gen epochs (Section II.1)
        phi_gen = 1.0 - math.exp(-e / max(self.e_gen_epochs, 1e-6))
        c_shared = self.c_inf_bpp * phi_gen
        return float(min(max(c_mem, c_shared), max(self.c_inf_bpp, self.c_max_post_cooldown_bpp)))

    def invert_epoch_for_c_mem(self, target_c_bpp: float) -> float | None:
        """Return smallest exposure ``E > 0`` where monotone ``c_mem(O, E) >= target_c_bpp``.

        Returns ``None`` when ``target_c_bpp`` exceeds the objective's reachable asymptotic
        memorization ceiling (meaning the model is capacity-bounded and never crosses ``target_c_bpp``).
        """
        target = float(target_c_bpp)
        if target <= 0.0:
            return 0.0
        ceiling = max(
            self.c_inf_bpp,
            self.c_max_post_cooldown_bpp,
            max(self.c_mem_envelope_bpp) if self.c_mem_envelope_bpp else 0.0,
        )
        if target > ceiling:
            return None
        if not self.exposure_epochs:
            if target >= self.c_inf_bpp:
                return None
            ratio = target / max(self.c_inf_bpp - target, 1e-12)
            return float(self.e_mem_epochs * (1.0 + ratio))

        es = self.exposure_epochs
        cs = self.c_mem_envelope_bpp
        if target > cs[-1]:
            return None
        if target <= cs[0]:
            if cs[0] <= 0.0:
                return float(es[0])
            return float(es[0] * (target / cs[0]))
        for i in range(1, len(es)):
            if cs[i] >= target:
                c0, c1 = cs[i - 1], cs[i]
                e0, e1 = es[i - 1], es[i]
                if c1 <= c0 + 1e-15:
                    return float(e1)
                frac = min(max((target - c0) / (c1 - c0), 0.0), 1.0)
                return float(math.exp(math.log(e0) + frac * (math.log(e1) - math.log(e0))))
        return float(es[-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "display_name": self.display_name,
            "c_inf_bpp": float(self.c_inf_bpp),
            "c_max_post_cooldown_bpp": float(self.c_max_post_cooldown_bpp),
            "E_mem_epochs": float(self.e_mem_epochs),
            "E_gen_epochs": float(self.e_gen_epochs),
            "W_window": float(self.w_window),
            "n_curve_points": len(self.exposure_epochs),
            "curve_epoch_range": [float(self.exposure_epochs[0]), float(self.exposure_epochs[-1])]
            if self.exposure_epochs
            else None,
            "source": self.source,
        }


def default_supply_profiles() -> dict[str, ObjectiveSupplyProfile]:
    """Theory-table reference profiles from ``docs/foundational-plm-theory.md`` Section VI."""
    return {
        "ar": ObjectiveSupplyProfile(
            objective="ar",
            display_name="AR",
            c_inf_bpp=3.69,
            c_max_post_cooldown_bpp=3.69,
            e_mem_epochs=13.0,
            e_gen_epochs=1.0,
            w_window=13.0,
            exposure_epochs=(1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0, 1000.0, 1500.0),
            c_mem_envelope_bpp=(0.0001, 0.0007, 0.0037, 0.054, 0.427, 1.041, 1.736, 2.349, 2.798, 3.361),
            source="theory_doc_section6_reference",
        ),
        "mlm_15": ObjectiveSupplyProfile(
            objective="mlm_15",
            display_name="MLM-15%",
            c_inf_bpp=0.79,
            c_max_post_cooldown_bpp=0.79,
            e_mem_epochs=436.0,
            e_gen_epochs=1.9,
            w_window=229.5,
            exposure_epochs=(1.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 5000.0, 12000.0, 25000.0),
            c_mem_envelope_bpp=(0.0, 0.0, 0.0006, 0.0032, 0.010, 0.055, 0.19, 0.55, 0.678, 0.762),
            source="theory_doc_section6_reference",
        ),
        "oa_uniform": ObjectiveSupplyProfile(
            objective="oa_uniform",
            display_name="MDM",
            c_inf_bpp=0.29,
            c_max_post_cooldown_bpp=0.29,
            e_mem_epochs=1933.0,
            e_gen_epochs=2.0,
            w_window=966.5,
            exposure_epochs=(1.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 40000.0),
            c_mem_envelope_bpp=(0.0, 0.0001, 0.001, 0.004, 0.018, 0.085, 0.155, 0.224, 0.273),
            source="theory_doc_section6_reference",
        ),
    }


def _monotone_envelope(points: Sequence[tuple[float, float]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Build a strictly non-decreasing, non-negative cumulative-max envelope over sorted epochs."""
    cleaned = sorted(
        (float(e), max(0.0, float(c)))
        for e, c in points
        if math.isfinite(float(e)) and math.isfinite(float(c)) and float(e) > 0.0
    )
    if not cleaned:
        return (), ()
    es: list[float] = []
    cs: list[float] = []
    running_max = 0.0
    for e, c in cleaned:
        running_max = max(running_max, c)
        if es and abs(e - es[-1]) <= 1e-12:
            cs[-1] = max(cs[-1], running_max)
        else:
            es.append(e)
            cs.append(running_max)
    return tuple(es), tuple(cs)


def load_calibrated_supply_profiles(
    stage1_summary_path: Path | str,
    *,
    stage2_summary_path: Path | str | None = None,
) -> dict[str, ObjectiveSupplyProfile]:
    """Load empirically calibrated ``ObjectiveSupplyProfile``s from Stage-1 (+ optional Stage-2) summaries."""
    s1_path = Path(stage1_summary_path)
    if not s1_path.is_file():
        raise FileNotFoundError(f"Stage-1 summary JSON not found: {s1_path}")
    s1 = json.loads(s1_path.read_text(encoding="utf-8"))
    c_cal = s1.get("c_max_calibration", {}).get("by_paradigm", {})
    exp_curves = s1.get("exposure_discount_curves", {})
    if not exp_curves:
        raise ValueError(f"Stage-1 summary missing 'exposure_discount_curves': {s1_path}")

    # Optional Stage-2 E_gen & natural-corpus W medians
    e_gen_by_obj: dict[str, list[float]] = defaultdict(list)
    if stage2_summary_path is not None:
        s2_path = Path(stage2_summary_path)
        if not s2_path.is_file():
            raise FileNotFoundError(f"Stage-2 summary JSON not found: {s2_path}")
        s2 = json.loads(s2_path.read_text(encoding="utf-8"))
        for r in s2.get("runs", []):
            if str(r.get("arm_id", "")).startswith("T"):
                continue
            obj = _canonical_objective(str(r.get("objective", "")))
            ec = _safe_float(r.get("e_conv_univ"))
            if ec is not None and ec > 0.0:
                e_gen_by_obj[obj].append(ec)

    default_e_gen = {"ar": 1.0, "mlm_15": 1.9, "oa_uniform": 2.0}
    profiles: dict[str, ObjectiveSupplyProfile] = {}

    for raw_obj, curve_entry in exp_curves.items():
        obj = _canonical_objective(raw_obj)
        cal_entry = c_cal.get(raw_obj, {}) or c_cal.get(obj, {})
        asym = cal_entry.get("asymptotic_and_metric_summary") or {}
        ec_np = curve_entry.get("pooled_e_crit_nonparametric") or {}

        raw_pts: list[tuple[float, float]] = []
        for row in curve_entry.get("pooled_by_exposure", []):
            ep = _safe_float(row.get("epoch"))
            cv = _safe_float(row.get("c_eff_empirical_mean", row.get("c_eff_at_2x_mean")))
            if ep is not None and ep > 0.0 and cv is not None:
                raw_pts.append((ep, cv))
        es, cs = _monotone_envelope(raw_pts)

        c_inf = (
            _safe_float(ec_np.get("c_inf_reference_empirical"))
            or _safe_float(asym.get("c_max_post_cooldown_bpp"))
            or _safe_float(asym.get("c_inf_constant_lr_bpp"))
            or (cs[-1] if cs else None)
        )
        c_max_cd = _safe_float(asym.get("c_max_post_cooldown_bpp")) or c_inf
        e_mem = (
            _safe_float(ec_np.get("e_crit_frac_c_inf_0p05"))
            or _safe_float(ec_np.get("e_crit_c_eff_0p02"))
            or _safe_float(ec_np.get("e_crit_delta_bpt_0p05"))
        )
        if c_inf is None or c_inf <= 0.0 or e_mem is None or e_mem <= 0.0:
            raise ValueError(f"{obj}: incomplete Stage-1 calibration (c_inf={c_inf}, e_mem={e_mem})")

        e_gen = float(np.median(e_gen_by_obj[obj])) if e_gen_by_obj.get(obj) else default_e_gen.get(obj, 1.5)
        w_window = float(e_mem / max(e_gen, 1e-6))

        profiles[obj] = ObjectiveSupplyProfile(
            objective=obj,
            display_name=objective_display_name(obj),
            c_inf_bpp=float(c_inf),
            c_max_post_cooldown_bpp=float(c_max_cd),
            e_mem_epochs=float(e_mem),
            e_gen_epochs=float(e_gen),
            w_window=w_window,
            exposure_epochs=es,
            c_mem_envelope_bpp=cs,
            source=f"{s1_path.name}" + (f"+{Path(stage2_summary_path).name}" if stage2_summary_path else ""),
        )

    return profiles


# --------------------------------------------------------------------------- #
# 1. Section-IV Closed-Form Production Design Rules
# --------------------------------------------------------------------------- #


def estimate_k_useful_bits(
    n_families: int,
    task_type: str,
    *,
    k_univ_bits: float = DEFAULT_K_UNIV_BITS,
    k_fam_bits: float = DEFAULT_K_FAM_PROFILE_BITS,
    coupling_multiplier: float = 1.0,
    min_diversity_families: int = DEFAULT_MIN_DIVERSITY_FAMILIES,
) -> dict[str, Any]:
    """Compute empirical useful information demand ``K_useful(D, T)`` in bits.

    - ``task_type="cross_fold_physics"`` (unseen-fold stability, de novo design, unseen-fold 3D
      contacts): ``K_useful = K_univ`` (O(1)), with a Mechanism-2 diversity guard when
      ``n_families < min_diversity_families``.
    - ``task_type="in_family_fitness"`` (seen-family DMS / homolog scoring):
      ``K_useful = K_univ + M * k_fam_bits * coupling_multiplier`` (O(1) + O(M)).
    """
    m = int(n_families)
    if m < 1:
        raise ValueError(f"n_families must be >= 1, got {m}")
    if k_univ_bits < 0.0 or k_fam_bits < 0.0 or coupling_multiplier <= 0.0:
        raise ValueError("k_univ_bits, k_fam_bits must be >= 0 and coupling_multiplier > 0")

    tt = str(task_type).strip().lower()
    if tt in ("cross_fold_physics", "cross_fold", "unseen_fold", "universal", "de_novo"):
        norm_task = "cross_fold_physics"
        k_spec_useful = 0.0
        k_useful = float(k_univ_bits)
    elif tt in ("in_family_fitness", "in_family", "homolog", "dms", "proteingym"):
        norm_task = "in_family_fitness"
        k_spec_useful = float(m) * float(k_fam_bits) * float(coupling_multiplier)
        k_useful = float(k_univ_bits) + k_spec_useful
    else:
        raise ValueError(
            f"Unsupported task_type {task_type!r}; use 'cross_fold_physics' or 'in_family_fitness'"
        )

    diversity_warning = m < int(min_diversity_families)
    return {
        "n_families_M": m,
        "task_type": norm_task,
        "K_univ_bits": float(k_univ_bits),
        "K_spec_useful_bits": float(k_spec_useful),
        "K_useful_bits": float(k_useful),
        "K_useful_mbits": float(k_useful / 1e6),
        "K_useful_gbits": float(k_useful / 1e9),
        "k_fam_effective_bits": float(k_fam_bits * coupling_multiplier) if norm_task == "in_family_fitness" else 0.0,
        "min_diversity_families": int(min_diversity_families),
        "narrow_M_diversity_warning": bool(diversity_warning),
        "diversity_note": (
            f"WARNING (Mechanism 2): M={m} < M_min={min_diversity_families} induces family-specific "
            "K_spec negative transfer on unseen folds even under zero K_res noise or high-E_mem MDM."
            if diversity_warning
            else f"OK: M={m} >= M_min={min_diversity_families} avoids narrow-M K_spec fold bias."
        ),
    }


def compute_optimal_model_size(
    k_useful_bits: float,
    objective: str,
    epochs: float,
    *,
    supply_profiles: Mapping[str, ObjectiveSupplyProfile],
    b_ext_bits: float = 0.0,
    compute_floor_params: int = 0,
) -> dict[str, Any]:
    """Closed-form Eq. (4) Optimal Sizing ``P*(M, O, E, Z)`` (with Section V.2 compute floor).

    Evaluates:
      ``P*_info = [K_useful(D, T) - B_ext(Z)]_+ / c_eff(A, O, E)``
      ``P*_recommended = max(P*_info, P*_compute)``
    returning both the asymptotic sizing lower bound (using ``c_inf(O)``) and the
    finite-horizon sizing at ``epochs`` (using ``c_eff(O, E)`` with shared-regularity ramp).
    """
    obj = _canonical_objective(objective)
    if obj not in supply_profiles:
        raise KeyError(f"Objective {objective!r} (canonical {obj!r}) not in supply_profiles")
    if k_useful_bits < 0.0 or b_ext_bits < 0.0 or epochs <= 0.0:
        raise ValueError("k_useful_bits, b_ext_bits must be >= 0 and epochs > 0")

    prof = supply_profiles[obj]
    k_net = max(0.0, float(k_useful_bits) - float(b_ext_bits))
    c_inf = prof.c_inf_bpp
    c_eff_useful = prof.c_eff_at_epoch(epochs, include_useful_floor=True)
    c_mem_e = prof.c_mem_at_epoch(epochs)

    p_star_asymptotic = k_net / c_inf if k_net > 0.0 else 0.0
    p_star_at_epoch = k_net / c_eff_useful if (k_net > 0.0 and c_eff_useful > 0.0) else 0.0
    p_rec = max(p_star_at_epoch, float(max(0, int(compute_floor_params))))

    return {
        "objective": obj,
        "display_name": prof.display_name,
        "epochs": float(epochs),
        "K_useful_bits": float(k_useful_bits),
        "B_ext_bits": float(b_ext_bits),
        "K_net_bits": float(k_net),
        "c_inf_bpp": float(c_inf),
        "c_eff_at_epoch_bpp": float(c_eff_useful),
        "c_mem_at_epoch_bpp": float(c_mem_e),
        "P_star_info_asymptotic": float(p_star_asymptotic),
        "P_star_info_at_epoch": float(p_star_at_epoch),
        "P_compute_floor": int(max(0, int(compute_floor_params))),
        "P_star_recommended": float(p_rec),
        "externally_satisfied": bool(k_net <= 0.0),
    }


def compute_safe_stopping_horizon(
    n_params_non_emb: float,
    k_useful_bits: float,
    objective: str,
    *,
    supply_profiles: Mapping[str, ObjectiveSupplyProfile],
    b_ext_bits: float = 0.0,
    safety_fraction: float = 1.0,
) -> dict[str, Any]:
    """Closed-form Safe Stopping Horizon ``E_max(P, M, O, Z)`` preventing ``Delta_cal`` & erasure.

    Solves Eq. (4) in exposure space:
      ``c_mem(O, E*_eq4) * P = safety_fraction * [K_useful(D, T) - B_ext(Z)]_+``
    and brackets against the memorization onset ``E_mem(O)`` and universal convergence ``E_gen(O)``.
    When ``c_inf(O) * P <= safety_fraction * K_net`` (under-capacity regime), residual memorization
    capacity never exceeds useful demand, so ``E_max_safe`` is unbounded by ``K_res`` memorization
    (returns ``None`` for ``E_eq4_capacity_match`` and ``regime="under_capacity_safe"``).
    """
    obj = _canonical_objective(objective)
    if obj not in supply_profiles:
        raise KeyError(f"Objective {objective!r} (canonical {obj!r}) not in supply_profiles")
    p = float(n_params_non_emb)
    if p <= 0.0 or k_useful_bits < 0.0 or b_ext_bits < 0.0 or safety_fraction <= 0.0:
        raise ValueError("n_params_non_emb > 0, k_useful_bits >= 0, b_ext_bits >= 0, safety_fraction > 0 required")

    prof = supply_profiles[obj]
    k_net = max(0.0, float(k_useful_bits) - float(b_ext_bits))
    max_supply_bits = prof.c_inf_bpp * p
    target_c_bpp = (float(safety_fraction) * k_net) / p

    e_eq4 = prof.invert_epoch_for_c_mem(target_c_bpp) if k_net > 0.0 else prof.e_gen_epochs
    if k_net <= 0.0:
        regime = "externally_satisfied"
        e_max_safe: float | None = prof.e_gen_epochs
    elif e_eq4 is None:
        regime = "under_capacity_safe"
        e_max_safe = None
    else:
        # In over-capacity regime, stop at or before Eq. (4) balance, floored at E_gen so physics converges
        e_max_safe = max(prof.e_gen_epochs, float(e_eq4))
        regime = "over_capacity_early_stop_required" if e_max_safe <= prof.e_mem_epochs * 2.0 else "window_buffered"

    return {
        "objective": obj,
        "display_name": prof.display_name,
        "n_params_non_emb": float(p),
        "K_useful_bits": float(k_useful_bits),
        "B_ext_bits": float(b_ext_bits),
        "K_net_bits": float(k_net),
        "max_param_supply_bits": float(max_supply_bits),
        "supply_to_net_demand_ratio": float(max_supply_bits / k_net) if k_net > 0.0 else None,
        "target_c_mem_bpp": float(target_c_bpp),
        "E_gen_min_epochs": float(prof.e_gen_epochs),
        "E_mem_onset_epochs": float(prof.e_mem_epochs),
        "W_window": float(prof.w_window),
        "E_eq4_capacity_match_epochs": float(e_eq4) if e_eq4 is not None else None,
        "E_max_safe_epochs": float(e_max_safe) if e_max_safe is not None else None,
        "stopping_regime": regime,
    }


def compute_objective_phase_boundary(
    n_params_non_emb: float,
    k_useful_bits: float,
    *,
    supply_profiles: Mapping[str, ObjectiveSupplyProfile],
    b_ext_bits: float = 0.0,
    planned_epochs: float = 30.0,
    eta_crit: float = DEFAULT_ETA_CRIT,
) -> dict[str, Any]:
    """Objective Phase-Boundary Criterion separating AR-favorable vs MLM/MDM-favorable regimes.

    Computes the critical supply-to-demand ratio:
      ``eta_AR = c_inf(AR) * P / [K_useful(D, T) - B_ext(Z)]_+``
    compared against ``eta_crit`` (default ``1.0`` from exact Eq. 4 balance):
    - ``eta_AR < eta_crit`` (under-capacity): ``AR`` wins via highest ``c_inf`` with zero spare
      capacity to memorize ``K_res``.
    - ``eta_AR >= eta_crit`` (sufficient/over-capacity):
      - if ``planned_epochs <= E_mem(AR)``: ``AR`` remains safe only with strict early stopping;
      - if ``E_mem(AR) < planned_epochs <= E_mem(MLM-15%)`` and ``eta_MLM < eta_crit``: ``MLM-15%``
        is favorable for in-family & general representation learning;
      - otherwise (``eta_AR >= eta_crit`` under sustained multi-epoch training or cross-fold
        zero-``Delta_cal`` requirement): ``MDM`` (highest ``W``) prevents both ``Delta_cal`` blowup
        and upper-layer structural erasure.
    """
    p = float(n_params_non_emb)
    if p <= 0.0 or k_useful_bits < 0.0 or b_ext_bits < 0.0 or planned_epochs <= 0.0 or eta_crit <= 0.0:
        raise ValueError("Invalid non-positive parameters for objective phase boundary")

    for req in ("ar", "mlm_15", "oa_uniform"):
        if req not in supply_profiles:
            raise KeyError(f"supply_profiles missing required objective {req!r}")

    prof_ar = supply_profiles["ar"]
    prof_mlm = supply_profiles["mlm_15"]
    prof_mdm = supply_profiles["oa_uniform"]

    k_net = max(0.0, float(k_useful_bits) - float(b_ext_bits))
    if k_net <= 0.0:
        return {
            "n_params_non_emb": p,
            "K_useful_bits": float(k_useful_bits),
            "B_ext_bits": float(b_ext_bits),
            "K_net_bits": 0.0,
            "planned_epochs": float(planned_epochs),
            "eta_crit": float(eta_crit),
            "eta_AR": None,
            "eta_MLM": None,
            "eta_MDM": None,
            "P_crit_AR": 0.0,
            "P_crit_MLM": 0.0,
            "P_crit_MDM": 0.0,
            "regime": "externally_satisfied",
            "recommended_objective": "structure_conditioned_decoder",
            "rationale": "External supply B_ext(Z) >= K_useful(D, T); a lightweight conditioned model suffices.",
        }

    eta_ar = (prof_ar.c_inf_bpp * p) / k_net
    eta_mlm = (prof_mlm.c_inf_bpp * p) / k_net
    eta_mdm = (prof_mdm.c_inf_bpp * p) / k_net

    p_crit_ar = (eta_crit * k_net) / prof_ar.c_inf_bpp
    p_crit_mlm = (eta_crit * k_net) / prof_mlm.c_inf_bpp
    p_crit_mdm = (eta_crit * k_net) / prof_mdm.c_inf_bpp

    if eta_ar < eta_crit:
        regime = "under_capacity_ar_favorable"
        rec_obj = "ar"
        rationale = (
            f"eta_AR={eta_ar:.3f} < eta_crit={eta_crit:.2f}: model is under-capacity relative to net useful "
            f"demand ({k_net / 1e6:.2f} Mbit); AR maximizes useful bits/param ({prof_ar.c_inf_bpp:.2f} vs "
            f"MLM {prof_mlm.c_inf_bpp:.2f}, MDM {prof_mdm.c_inf_bpp:.2f}) without spare capacity for K_res."
        )
    elif planned_epochs <= prof_ar.e_mem_epochs:
        regime = "over_capacity_single_or_low_epoch_ar_early_stop"
        rec_obj = "ar"
        rationale = (
            f"eta_AR={eta_ar:.3f} >= eta_crit={eta_crit:.2f} but planned E={planned_epochs:.1f} <= "
            f"E_mem(AR)={prof_ar.e_mem_epochs:.1f}: AR stays inside its generalization window if stopped strictly "
            "before E_mem; switch to MLM/MDM if training longer."
        )
    elif eta_mlm < eta_crit and planned_epochs <= prof_mlm.e_mem_epochs:
        regime = "transition_mlm_favorable"
        rec_obj = "mlm_15"
        rationale = (
            f"eta_AR={eta_ar:.3f} >= eta_crit while eta_MLM={eta_mlm:.3f} < eta_crit and "
            f"E={planned_epochs:.1f} <= E_mem(MLM)={prof_mlm.e_mem_epochs:.1f}: AR would memorize K_res "
            "and suffer Delta_cal/late-layer erasure, whereas MLM-15% captures useful structure safely."
        )
    else:
        regime = "over_capacity_mdm_favorable"
        rec_obj = "oa_uniform"
        rationale = (
            f"eta_AR={eta_ar:.3f} >= eta_crit (eta_MLM={eta_mlm:.3f}) at E={planned_epochs:.1f} > "
            f"E_mem(AR)={prof_ar.e_mem_epochs:.1f}: MDM's wide generalization window (W={prof_mdm.w_window:.1f}) "
            "suppresses K_res memorization, keeps Delta_cal~0, and preserves deep 3D representations to norm_f."
        )

    return {
        "n_params_non_emb": p,
        "K_useful_bits": float(k_useful_bits),
        "B_ext_bits": float(b_ext_bits),
        "K_net_bits": float(k_net),
        "planned_epochs": float(planned_epochs),
        "eta_crit": float(eta_crit),
        "eta_AR": float(eta_ar),
        "eta_MLM": float(eta_mlm),
        "eta_MDM": float(eta_mdm),
        "P_crit_AR": float(p_crit_ar),
        "P_crit_MLM": float(p_crit_mlm),
        "P_crit_MDM": float(p_crit_mdm),
        "regime": regime,
        "recommended_objective": rec_obj,
        "rationale": rationale,
    }


def compute_structural_parameter_equivalence(
    b_ext_bits: float,
    objective: str,
    *,
    supply_profiles: Mapping[str, ObjectiveSupplyProfile],
    epochs: float | None = None,
) -> dict[str, Any]:
    """Parameter-Equivalence of Structural Conditioning: ``Delta P_saved(Z) = B_ext(Z) / c_eff(A, O, E)``."""
    obj = _canonical_objective(objective)
    if obj not in supply_profiles:
        raise KeyError(f"Objective {objective!r} (canonical {obj!r}) not in supply_profiles")
    if b_ext_bits < 0.0:
        raise ValueError(f"b_ext_bits must be >= 0, got {b_ext_bits}")

    prof = supply_profiles[obj]
    c_inf = prof.c_inf_bpp
    p_saved_asymptotic = float(b_ext_bits) / c_inf

    c_eff = None
    p_saved_at_epoch = None
    if epochs is not None:
        if epochs <= 0.0:
            raise ValueError(f"epochs must be > 0, got {epochs}")
        c_eff = prof.c_eff_at_epoch(epochs, include_useful_floor=True)
        p_saved_at_epoch = float(b_ext_bits) / c_eff if c_eff > 0.0 else None

    return {
        "objective": obj,
        "display_name": prof.display_name,
        "B_ext_bits": float(b_ext_bits),
        "B_ext_gbits": float(b_ext_bits / 1e9),
        "c_inf_bpp": float(c_inf),
        "Delta_P_saved_asymptotic": float(p_saved_asymptotic),
        "epochs": float(epochs) if epochs is not None else None,
        "c_eff_at_epoch_bpp": float(c_eff) if c_eff is not None else None,
        "Delta_P_saved_at_epoch": float(p_saved_at_epoch) if p_saved_at_epoch is not None else None,
    }


# --------------------------------------------------------------------------- #
# 2. Layerwise Probing Synthesis (43-Checkpoint + Knob-3 Structural Erasure)
# --------------------------------------------------------------------------- #


def _extract_layer_profile(doc: Mapping[str, Any], probe_key: str, metric_fn: Any) -> dict[str, Any]:
    probe = doc.get(probe_key) or {}
    layers_list = probe.get("layers") or []
    by_name = {entry.get("layer_name"): entry for entry in layers_list if isinstance(entry, Mapping)}
    ordered_names = list(doc.get("layer_names") or by_name.keys())

    emb_val: float | None = None
    non_emb: list[tuple[str, float]] = []
    for ln in ordered_names:
        entry = by_name.get(ln)
        if entry is None:
            continue
        val = _safe_float(metric_fn(entry))
        if val is None:
            continue
        if ln in ("emb", "embed"):
            emb_val = val
        else:
            non_emb.append((str(ln), val))

    if not non_emb:
        return {
            "emb": emb_val,
            "peak_layer": None,
            "peak_value": None,
            "peak_rel_depth": None,
            "final_layer": None,
            "final_value": None,
            "erasure": None,
        }

    peak_idx = int(np.argmax([v for _, v in non_emb]))
    peak_layer, peak_val = non_emb[peak_idx]
    final_layer, final_val = non_emb[-1]
    rel_depth = float(peak_idx) / float(max(1, len(non_emb) - 1))
    return {
        "emb": emb_val,
        "peak_layer": peak_layer,
        "peak_value": float(peak_val),
        "peak_rel_depth": float(rel_depth),
        "final_layer": final_layer,
        "final_value": float(final_val),
        "erasure": float(final_val - peak_val),
    }


def analyze_probing_erasure(probing_dir: Path | str) -> dict[str, Any]:
    """Synthesize layerwise structural erasure & ``Delta_cal`` alignment across probing checkpoints."""
    root = Path(probing_dir)
    ckpt_dir = root / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Probing checkpoints directory not found: {ckpt_dir}")
    files = sorted(ckpt_dir.glob("*.probe.json"))
    if not files:
        raise FileNotFoundError(f"No *.probe.json files found in {ckpt_dir}")

    rows: list[dict[str, Any]] = []
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        bur = _extract_layer_profile(doc, "probe_burial", lambda x: x.get("pooled_r2"))
        cnt = _extract_layer_profile(
            doc, "probe_contacts", lambda x: (x.get("long_range_sep_ge_24") or {}).get("auroc")
        )
        ms_uf = _extract_layer_profile(
            doc,
            "probe_megascale_ddg",
            lambda x: ((x.get("by_stratum") or {}).get("unseen_fold_natural") or {}).get("spearman_mean"),
        )
        ms_dn = _extract_layer_profile(
            doc,
            "probe_megascale_ddg",
            lambda x: ((x.get("by_stratum") or {}).get("de_novo_design") or {}).get("spearman_mean"),
        )
        arm = str(doc.get("arm_id", ""))
        rows.append(
            {
                "checkpoint_id": doc.get("checkpoint_id"),
                "run_id": doc.get("run_id"),
                "arm_id": arm,
                "is_knob3": arm.startswith("T"),
                "scale_id": doc.get("scale_id"),
                "objective": _canonical_objective(str(doc.get("objective", ""))),
                "E": int(doc.get("E", doc.get("epoch", 0))),
                "n_layers": int(doc.get("n_layers", 0)),
                "H_unseen_fold": _safe_float(doc.get("H_unseen_fold")),
                "H_cal": _safe_float(doc.get("H_cal")),
                "Delta_cal": _safe_float(doc.get("Delta_cal")),
                "burial": bur,
                "contacts_lr": cnt,
                "megascale_unseen_fold": ms_uf,
                "megascale_de_novo": ms_dn,
            }
        )

    def _agg_subset(sub: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        by_obj: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for r in sub:
            by_obj[str(r["objective"])].append(r)
        out_obj: dict[str, Any] = {}
        for obj in sorted(by_obj.keys()):
            grp = by_obj[obj]
            def _m(fn: Any) -> float | None:
                vals = [fn(r) for r in grp if fn(r) is not None and math.isfinite(float(fn(r)))]
                return float(np.mean(vals)) if vals else None
            out_obj[obj] = {
                "n_checkpoints": len(grp),
                "mean_Delta_cal": _m(lambda r: r["Delta_cal"]),
                "burial_peak_r2": _m(lambda r: r["burial"]["peak_value"]),
                "burial_final_r2": _m(lambda r: r["burial"]["final_value"]),
                "burial_erasure": _m(lambda r: r["burial"]["erasure"]),
                "burial_peak_rel_depth": _m(lambda r: r["burial"]["peak_rel_depth"]),
                "contacts_lr_peak_auroc": _m(lambda r: r["contacts_lr"]["peak_value"]),
                "contacts_lr_final_auroc": _m(lambda r: r["contacts_lr"]["final_value"]),
                "contacts_lr_erasure": _m(lambda r: r["contacts_lr"]["erasure"]),
                "contacts_lr_peak_rel_depth": _m(lambda r: r["contacts_lr"]["peak_rel_depth"]),
                "megascale_uf_erasure": _m(lambda r: r["megascale_unseen_fold"]["erasure"]),
                "megascale_dn_erasure": _m(lambda r: r["megascale_de_novo"]["erasure"]),
            }

        # Correlations between Delta_cal and erasure across the subset
        corrs: dict[str, Any] = {}
        for cname, fn in (
            ("burial_erasure", lambda r: r["burial"]["erasure"]),
            ("contacts_lr_erasure", lambda r: r["contacts_lr"]["erasure"]),
            ("megascale_uf_erasure", lambda r: r["megascale_unseen_fold"]["erasure"]),
            ("burial_peak_rel_depth", lambda r: r["burial"]["peak_rel_depth"]),
            ("contacts_lr_peak_rel_depth", lambda r: r["contacts_lr"]["peak_rel_depth"]),
        ):
            pairs = [
                (float(r["Delta_cal"]), float(fn(r)))
                for r in sub
                if r["Delta_cal"] is not None and fn(r) is not None
            ]
            if len(pairs) >= 4:
                xs = np.asarray([a for a, _ in pairs], dtype=np.float64)
                ys = np.asarray([b for _, b in pairs], dtype=np.float64)
                if float(np.std(xs)) > 1e-12 and float(np.std(ys)) > 1e-12:
                    sr = spearmanr(xs, ys)
                    corrs[cname] = {
                        "n": len(pairs),
                        "spearman_r": _safe_float(sr.statistic),
                        "spearman_p": _safe_float(sr.pvalue),
                    }
        return {"by_objective": out_obj, "spearman_with_Delta_cal": corrs}

    nat_rows = [r for r in rows if not r["is_knob3"]]
    k3_rows = [r for r in rows if r["is_knob3"]]

    # Knob-3 temperature table (ordered T0p1 -> T1p2)
    k3_table = []
    for arm in ("T0p1", "T0p3", "T0p7", "T1p2"):
        for obj in ("ar", "mlm_15", "oa_uniform"):
            hits = [r for r in k3_rows if r["arm_id"] == arm and r["objective"] == obj]
            if hits:
                h = hits[0]
                k3_table.append(
                    {
                        "arm_id": arm,
                        "objective": obj,
                        "Delta_cal": h["Delta_cal"],
                        "H_cal": h["H_cal"],
                        "burial_peak_r2": h["burial"]["peak_value"],
                        "burial_final_r2": h["burial"]["final_value"],
                        "burial_erasure": h["burial"]["erasure"],
                        "contacts_lr_peak_auroc": h["contacts_lr"]["peak_value"],
                        "contacts_lr_final_auroc": h["contacts_lr"]["final_value"],
                        "contacts_lr_erasure": h["contacts_lr"]["erasure"],
                        "megascale_uf_erasure": h["megascale_unseen_fold"]["erasure"],
                        "megascale_dn_erasure": h["megascale_de_novo"]["erasure"],
                    }
                )

    return {
        "n_checkpoints": len(rows),
        "n_natural": len(nat_rows),
        "n_knob3": len(k3_rows),
        "all_checkpoints": _agg_subset(rows),
        "natural_arms": _agg_subset(nat_rows),
        "natural_small_0p5m_2m": _agg_subset([r for r in nat_rows if r["scale_id"] in ("0.5m", "2m")]),
        "natural_large_8m_35m": _agg_subset([r for r in nat_rows if r["scale_id"] in ("8m", "35m")]),
        "knob3_temperature_sweep_2m": _agg_subset(k3_rows),
        "knob3_by_temperature_objective": k3_table,
    }


# --------------------------------------------------------------------------- #
# 3. Cross-Experiment Orchestrator (Stage-1 + Stage-2 + Probing + Optional Suites)
# --------------------------------------------------------------------------- #


def _extract_stage2_empirical_demand(
    stage2_summary_path: Path | str,
    bundle_manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    s2 = json.loads(Path(stage2_summary_path).read_text(encoding="utf-8"))
    arm_families: dict[str, int] = {}
    if bundle_manifest_path is not None and Path(bundle_manifest_path).is_file():
        bm = json.loads(Path(bundle_manifest_path).read_text(encoding="utf-8"))
        for ak, av in (bm.get("arms") or {}).items():
            nf = av.get("realized_families") or av.get("n_train_backbones")
            if nf is not None:
                arm_families[str(ak)] = int(nf)

    natural_arms: list[dict[str, Any]] = []
    for r in s2.get("runs", []):
        arm = str(r.get("arm_id", ""))
        if arm.startswith("T"):
            continue
        ds = r.get("demand_and_supply") or {}
        natural_arms.append(
            {
                "run_id": r.get("run_id"),
                "arm_id": arm,
                "n_families_M": arm_families.get(arm),
                "scale_id": r.get("scale_id"),
                "objective": _canonical_objective(str(r.get("objective", ""))),
                "n_params_non_embedding": r.get("n_params_non_embedding"),
                "n_train_unique_tokens": r.get("n_train_unique_tokens"),
                "epoch_done": r.get("epoch_done"),
                "K_univ_empirical_bits": _safe_float(ds.get("k_univ_empirical_bits")),
                "K_spec_empirical_bits": _safe_float(ds.get("k_spec_pre_blowup_bits")),
                "K_useful_empirical_bits": _safe_float(ds.get("k_useful_empirical_bits")),
                "e_conv_univ": _safe_float(r.get("e_conv_univ")),
                "e_crit_res": _safe_float(r.get("e_crit_res")),
                "predicted_e_star": _safe_float(ds.get("predicted_e_star_from_stage1_supply")),
            }
        )

    return {
        "n_runs_analyzed": int(s2.get("n_runs_analyzed", 0)),
        "n_runs_complete": int(s2.get("n_runs_complete", 0)),
        "natural_runs": natural_arms,
        "master_equation_alignment": s2.get("master_equation_alignment", []),
    }


def _ingest_optional_directory(
    path: Path | str | None,
    *,
    kind: str,
) -> dict[str, Any]:
    """Ingest optional experiment suites (open_model_audit, multimodal_3d, stage2_large_25m, ablations)."""
    if path is None:
        return {"status": "not_provided", "n_files": 0}
    p = Path(path)
    if not p.exists():
        return {"status": "pending_sync", "path": str(p), "n_files": 0}

    json_files = [p] if p.is_file() else sorted(f for f in p.glob("*.json") if not f.name.endswith(".sha256"))
    if not json_files:
        return {"status": "pending_sync", "path": str(p), "n_files": 0}

    summaries: list[dict[str, Any]] = []
    for jf in json_files:
        try:
            doc = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(doc, Mapping):
            continue
        if kind == "open_model_audit" and doc.get("schema_version") == "open_model_audit_report_v1":
            mdl = doc.get("model") or {}
            prim = str(mdl.get("primary_readout", ""))
            ht_all = (doc.get("horvitz_thompson_all_uniref50") or {}).get(prim) or {}
            panel = (doc.get("paired_3d_panel") or {}).get("overall") or {}
            summaries.append(
                {
                    "model_id": mdl.get("model_id"),
                    "family_kind": mdl.get("family_kind"),
                    "primary_readout": prim,
                    "n_params_non_emb": mdl.get("n_params_non_emb"),
                    "B_total_bits_all": ht_all.get("B_total_bits"),
                    "c_app_bpp_all": ht_all.get("c_app_bits_per_non_emb_param"),
                    "has_3d_panel": bool(panel),
                }
            )
        elif kind == "multimodal_3d" and ("b_ext" in json.dumps(doc) or doc.get("suite") == "multimodal_3d" or "run_id" in doc):
            summaries.append(
                {
                    "file": jf.name,
                    "run_id": doc.get("run_id"),
                    "status": doc.get("status"),
                    "arm_id": doc.get("arm_id") or (doc.get("run_spec") or {}).get("arm_id"),
                }
            )
        else:
            summaries.append(
                {
                    "file": jf.name,
                    "run_id": doc.get("run_id"),
                    "status": doc.get("status"),
                }
            )

    return {
        "status": "ingested" if summaries else "pending_sync",
        "path": str(p),
        "n_files": len(summaries),
        "records": summaries,
    }


def synthesize_production_criteria(
    *,
    stage1_summary_path: Path | str,
    stage2_summary_path: Path | str | None = None,
    stage2_bundle_manifest_path: Path | str | None = None,
    probing_dir: Path | str | None = None,
    stage2_large_dir: Path | str | None = None,
    open_model_audit_dir: Path | str | None = None,
    multimodal_3d_dir: Path | str | None = None,
    supply_ablations_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Assemble the complete Section-IV quantitative production-engineering specification."""
    profiles = load_calibrated_supply_profiles(
        stage1_summary_path, stage2_summary_path=stage2_summary_path
    )

    # Canonical engineering scenarios demonstrating Eq. (4) sizing, stopping, phase boundary & B_ext(Z)
    scenarios: list[dict[str, Any]] = []
    reference_cases = [
        ("cross_fold_physics_stage2_5m_ar8m", 1024, "cross_fold_physics", 1_500_000.0, 8_000_000.0, 30.0, 0.0),
        ("cross_fold_physics_proteome_35m", 20_000, "cross_fold_physics", DEFAULT_K_UNIV_BITS, 35_000_000.0, 50.0, 0.0),
        ("cross_fold_physics_proteome_650m", 20_000, "cross_fold_physics", DEFAULT_K_UNIV_BITS, 650_000_000.0, 50.0, 0.0),
        (
            "uniref50_profile_fitness_650m",
            20_000_000,
            "in_family_fitness",
            estimate_k_useful_bits(20_000_000, "in_family_fitness")["K_useful_bits"],
            650_000_000.0,
            10.0,
            0.0,
        ),
        (
            "uniref50_profile_fitness_15b",
            20_000_000,
            "in_family_fitness",
            estimate_k_useful_bits(20_000_000, "in_family_fitness")["K_useful_bits"],
            15_000_000_000.0,
            50.0,
            0.0,
        ),
        (
            "uniref50_coupled_fitness_15b",
            20_000_000,
            "in_family_fitness",
            estimate_k_useful_bits(20_000_000, "in_family_fitness", k_fam_bits=DEFAULT_K_FAM_COUPLED_BITS)["K_useful_bits"],
            15_000_000_000.0,
            10.0,
            0.0,
        ),
    ]
    for name, m_fam, task, k_use, p_val, ep_plan, b_ext in reference_cases:
        pb = compute_objective_phase_boundary(
            p_val, k_use, supply_profiles=profiles, b_ext_bits=b_ext, planned_epochs=ep_plan
        )
        sizing = {
            obj: compute_optimal_model_size(k_use, obj, ep_plan, supply_profiles=profiles, b_ext_bits=b_ext)
            for obj in ("ar", "mlm_15", "oa_uniform")
        }
        stopping = {
            obj: compute_safe_stopping_horizon(p_val, k_use, obj, supply_profiles=profiles, b_ext_bits=b_ext)
            for obj in ("ar", "mlm_15", "oa_uniform")
        }
        scenarios.append(
            {
                "scenario": name,
                "n_families_M": m_fam,
                "task_type": task,
                "K_useful_bits": k_use,
                "n_params_non_emb": p_val,
                "planned_epochs": ep_plan,
                "phase_boundary": pb,
                "optimal_sizing_by_objective": sizing,
                "safe_stopping_by_objective": stopping,
            }
        )

    # Structural conditioning parameter-equivalence reference table (b_ext = 0.5, 1.0, 1.5, 2.0 bits/res
    # across UniRef50 eligible 5.38B residues and full 17.28B residues)
    b_ext_equiv: list[dict[str, Any]] = []
    for corpus_label, n_res in (("uniref50_eligible_5p38B", 5_378_061_546), ("uniref50_all_17p28B", 17_282_055_792)):
        for b_res in (0.5, 1.0, 1.5, 2.0):
            total_b_ext = float(b_res) * float(n_res)
            b_ext_equiv.append(
                {
                    "corpus": corpus_label,
                    "n_residues": n_res,
                    "b_ext_bits_per_res": b_res,
                    "B_ext_total_bits": total_b_ext,
                    "by_objective": {
                        obj: compute_structural_parameter_equivalence(total_b_ext, obj, supply_profiles=profiles)
                        for obj in ("ar", "mlm_15", "oa_uniform")
                    },
                }
            )

    report: dict[str, Any] = {
        "schema": PRODUCTION_CRITERIA_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "theory_alignment": "docs/foundational-plm-theory.md (Eq. 1-4, Sections I-VI)",
        "calibrated_supply_profiles": {k: v.to_dict() for k, v in sorted(profiles.items())},
        "k_useful_scaling_laws": {
            "cross_fold_physics_default": estimate_k_useful_bits(20_000, "cross_fold_physics"),
            "in_family_profile_20m_clusters": estimate_k_useful_bits(20_000_000, "in_family_fitness"),
            "in_family_coupled_20m_clusters": estimate_k_useful_bits(
                20_000_000, "in_family_fitness", k_fam_bits=DEFAULT_K_FAM_COUPLED_BITS
            ),
            "narrow_M_warning_example_M64": estimate_k_useful_bits(64, "cross_fold_physics"),
        },
        "production_scenarios": scenarios,
        "structural_parameter_equivalence_table": b_ext_equiv,
        "stage2_empirical_demand": (
            _extract_stage2_empirical_demand(stage2_summary_path, stage2_bundle_manifest_path)
            if stage2_summary_path is not None and Path(stage2_summary_path).is_file()
            else {"status": "not_provided"}
        ),
        "probing_erasure_synthesis": (
            analyze_probing_erasure(probing_dir)
            if probing_dir is not None and Path(probing_dir).is_dir()
            else {"status": "not_provided"}
        ),
        "optional_suites": {
            "stage2_large_25m": _ingest_optional_directory(stage2_large_dir, kind="stage2_large_25m"),
            "open_model_audit": _ingest_optional_directory(open_model_audit_dir, kind="open_model_audit"),
            "multimodal_3d": _ingest_optional_directory(multimodal_3d_dir, kind="multimodal_3d"),
            "supply_ablations": _ingest_optional_directory(supply_ablations_dir, kind="supply_ablations"),
        },
    }

    assert_no_forbidden_infra_strings(report, label="production_criteria_report")
    return report
