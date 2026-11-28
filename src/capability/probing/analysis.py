"""Orchestration and H1 vs H2 diagnostic aggregation for the layerwise probing suite.

Computes per-checkpoint probe evaluation (Eq. 3b temperature calibration + Probes 1, 2, 3)
and cross-checkpoint / cross-epoch trajectory aggregation:
  - ``best_layer_idx``, ``best_layer_name``, ``best_layer_score``
  - ``head_retention_ratio`` (zero-shot / best-layer probe score)
  - ``final_norm_retention_ratio`` (final ``norm_f`` probe / best-layer probe score)
  - ``best_layer_vs_zeroshot_gain`` (and vs temperature-calibrated zero-shot)
  - ``layer_degradation_profile`` from ``E = E_peak`` to ``E = E_final`` (e.g. ``E=300``)
    distinguishing H1 (Output-Head / Top-Layer Pollution) vs H2 (Deep Representation Overwriting).
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from src.capability.supply_demand.corpus import Stage2Bundle
from src.capability.supply_demand.objectives import PackedSequenceStore

from .extractor import (
    LoadedCheckpoint,
    extract_layerwise_hidden_states,
    load_stage2_checkpoint,
)
from .probes import (
    DEFAULT_RIDGE_LAMBDAS,
    evaluate_burial_layerwise_probe,
    evaluate_contact_layerwise_probe,
    evaluate_megascale_layerwise_probe,
    evaluate_unseen_fold_temperature_calibration,
)

PROBE_CHECKPOINT_SCHEMA: str = "stage2_layerwise_probe_checkpoint_v1"
PROBE_SUMMARY_SCHEMA: str = "stage2_layerwise_probe_summary_v1"


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def compute_layer_retention_metrics(
    layer_scores: Sequence[float | None],
    layer_names: Sequence[str],
    *,
    zeroshot_score: float | None = None,
    calibrated_zeroshot_score: float | None = None,
) -> dict[str, Any]:
    """Summarize best internal layer, output-head retention ratio, and probe-vs-zeroshot gain."""
    valid = [
        (idx, float(sc))
        for idx, sc in enumerate(layer_scores)
        if sc is not None and math.isfinite(float(sc))
    ]
    if not valid:
        return {
            "best_layer_idx": None,
            "best_layer_name": None,
            "best_layer_rel_depth": None,
            "best_layer_score": None,
            "final_norm_score": None,
            "zeroshot_score": _safe_float(zeroshot_score),
            "calibrated_zeroshot_score": _safe_float(calibrated_zeroshot_score),
            "head_retention_ratio": None,
            "calibrated_head_retention_ratio": None,
            "final_norm_retention_ratio": None,
            "best_layer_vs_zeroshot_gain": None,
            "best_layer_vs_calibrated_zeroshot_gain": None,
        }

    best_idx, best_score = max(valid, key=lambda kv: kv[1])
    n_taps = len(layer_names)
    final_score = _safe_float(layer_scores[-1]) if layer_scores else None
    zs = _safe_float(zeroshot_score)
    zs_cal = _safe_float(calibrated_zeroshot_score)

    def _ratio(num: float | None, den: float) -> float | None:
        if num is None or den <= 1e-6:
            return None
        return float(num / den)

    return {
        "best_layer_idx": int(best_idx),
        "best_layer_name": str(layer_names[best_idx]),
        "best_layer_rel_depth": float(best_idx / max(1, n_taps - 1)),
        "best_layer_score": float(best_score),
        "final_norm_score": final_score,
        "zeroshot_score": zs,
        "calibrated_zeroshot_score": zs_cal,
        "head_retention_ratio": _ratio(zs, best_score),
        "calibrated_head_retention_ratio": _ratio(zs_cal, best_score),
        "final_norm_retention_ratio": _ratio(final_score, best_score),
        "best_layer_vs_zeroshot_gain": None if zs is None else float(best_score - zs),
        "best_layer_vs_calibrated_zeroshot_gain": (
            None if zs_cal is None else float(best_score - zs_cal)
        ),
    }


def _extract_checkpoint_headlines(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Compute concise H1/H2 diagnostic headline metrics for one checkpoint."""
    layer_names: list[str] = list(doc.get("layer_names") or [])
    out: dict[str, Any] = {}

    ms = doc.get("probe_megascale_ddg") or {}
    ms_layers = ms.get("layers") or []
    ms_zs = ms.get("zeroshot") or {}
    for stratum in ("unseen_fold_natural", "de_novo_design", "ood_unseen_and_de_novo"):
        scores = [
            _safe_float(((lay.get("by_stratum") or {}).get(stratum) or {}).get("spearman_mean"))
            for lay in ms_layers
        ]
        zs_raw = _safe_float(((ms_zs.get("raw") or {}).get(stratum) or {}).get("spearman_mean"))
        zs_cal = _safe_float(
            ((ms_zs.get("temperature_calibrated") or {}).get(stratum) or {}).get("spearman_mean")
        )
        if any(s is not None for s in scores):
            out[f"megascale_{stratum}_spearman"] = compute_layer_retention_metrics(
                scores,
                layer_names,
                zeroshot_score=zs_raw,
                calibrated_zeroshot_score=zs_cal,
            )

    ct = doc.get("probe_contacts") or {}
    ct_layers = ct.get("layers") or []
    ct_zs = ct.get("zeroshot_categorical_jacobian") or {}
    for range_key, metric_key in (
        ("sep_ge_6", "auroc"),
        ("sep_ge_6", "p_at_L"),
        ("long_range_sep_ge_24", "auroc"),
        ("long_range_sep_ge_24", "p_at_L"),
        ("long_range_sep_ge_24", "p_at_L_over_5"),
    ):
        scores = [
            _safe_float(((lay.get(range_key) or {}).get(metric_key)))
            for lay in ct_layers
        ]
        zs_val = _safe_float(((ct_zs.get(range_key) or {}).get(metric_key)))
        if any(s is not None for s in scores):
            out[f"contacts_{range_key}_{metric_key}"] = compute_layer_retention_metrics(
                scores,
                layer_names,
                zeroshot_score=zs_val,
            )

    bur = doc.get("probe_burial") or {}
    bur_layers = bur.get("layers") or []
    for metric_key in ("pooled_r2", "pooled_spearman", "domain_mean_spearman"):
        scores = [_safe_float(lay.get(metric_key)) for lay in bur_layers]
        if any(s is not None for s in scores):
            out[f"burial_{metric_key}"] = compute_layer_retention_metrics(scores, layer_names)

    return out


@torch.no_grad()
def evaluate_checkpoint_probes(
    checkpoint: LoadedCheckpoint | Path | str,
    bundle: Stage2Bundle | Path | str,
    *,
    device: torch.device | str = "cpu",
    precision: str = "bf16",
    max_variants_per_wt: int | None = 400,
    megascale_batch_size: int = 512,
    contact_batch_size: int = 64,
    zeroshot_contact_batch_size: int = 256,
    cal_batch_size: int = 128,
    cal_n_draws: int = 16,
    lambdas: Sequence[float] = DEFAULT_RIDGE_LAMBDAS,
    n_cv_folds: int = 5,
    compute_zeroshot_contacts: bool = True,
    preloaded_megascale: Mapping[str, Any] | None = None,
    preloaded_contacts: Sequence[Mapping[str, Any]] | None = None,
    preloaded_unseen_fold_store: PackedSequenceStore | None = None,
) -> dict[str, Any]:
    """Run Eq. (3b) temperature calibration + Probes 1, 2, 3 on one frozen Stage-2 checkpoint."""
    t0 = time.perf_counter()
    loaded = (
        checkpoint
        if isinstance(checkpoint, LoadedCheckpoint)
        else load_stage2_checkpoint(checkpoint, device=device)
    )
    b_obj = bundle if isinstance(bundle, Stage2Bundle) else Stage2Bundle(bundle)

    # Load eval_unseen_fold for Eq. (3b) temperature calibration
    if preloaded_unseen_fold_store is not None:
        unseen_store = preloaded_unseen_fold_store
    else:
        arm_unseen = b_obj.root / "splits" / f"eval_unseen_fold_{loaded.arm_id}.npz"
        if arm_unseen.is_file():
            unseen_store, _ = b_obj.load_split(f"eval_unseen_fold_{loaded.arm_id}")
        else:
            unseen_store, _ = b_obj.load_split("eval_unseen_fold")

    eq3b_cal = evaluate_unseen_fold_temperature_calibration(
        loaded.model,
        unseen_store,
        objective=loaded.objective,
        batch_size=cal_batch_size,
        precision=precision,
        n_draws=cal_n_draws,
    )
    opt_temp = float(eq3b_cal["T_opt"])

    # Probe 1: MegaScale ddG
    ms_data = preloaded_megascale if preloaded_megascale is not None else b_obj.load_megascale()
    probe_ms: dict[str, Any] | None = None
    if ms_data is not None:
        probe_ms = evaluate_megascale_layerwise_probe(
            loaded.model,
            ms_data,
            objective=loaded.objective,
            optimal_temperature=opt_temp,
            max_variants_per_wt=max_variants_per_wt,
            batch_size=megascale_batch_size,
            precision=precision,
            lambdas=lambdas,
            n_cv_folds=n_cv_folds,
        )

    # Shared hidden-state extraction for Probes 2 & 3 on the 49 unseen-fold 3D domains
    contacts_data = (
        list(preloaded_contacts) if preloaded_contacts is not None else b_obj.load_contacts()
    )
    probe_contacts: dict[str, Any] | None = None
    probe_burial: dict[str, Any] | None = None
    if contacts_data:
        contact_hidden = extract_layerwise_hidden_states(
            loaded.model,
            [str(d["sequence"]) for d in contacts_data],
            objective=loaded.objective,
            batch_size=contact_batch_size,
            precision=precision,
        )
        probe_contacts = evaluate_contact_layerwise_probe(
            loaded.model,
            contacts_data,
            objective=loaded.objective,
            preextracted_hidden=contact_hidden,
            batch_size=contact_batch_size,
            zeroshot_batch_size=zeroshot_contact_batch_size,
            precision=precision,
            n_folds=n_cv_folds,
            lambdas=lambdas,
            compute_zeroshot=compute_zeroshot_contacts,
        )
        probe_burial = evaluate_burial_layerwise_probe(
            loaded.model,
            contacts_data,
            objective=loaded.objective,
            preextracted_hidden=contact_hidden,
            batch_size=contact_batch_size,
            precision=precision,
            n_folds=n_cv_folds,
            lambdas=lambdas,
        )

    result: dict[str, Any] = {
        "schema": PROBE_CHECKPOINT_SCHEMA,
        "checkpoint_id": loaded.checkpoint_id,
        "run_id": loaded.run_id,
        "arm_id": loaded.arm_id,
        "objective": loaded.objective,
        "paradigm_display": loaded.paradigm_display,
        "scale_id": loaded.scale_id,
        "E": int(loaded.epoch),
        "epoch": int(loaded.epoch),
        "global_step": int(loaded.global_step),
        "tokens_seen": int(loaded.tokens_seen),
        "n_layers": int(loaded.n_layers),
        "d_model": int(loaded.d_model),
        "layer_names": list(loaded.layer_names),
        "temperature_calibration_unseen_fold": eq3b_cal,
        "H_unseen_fold": _safe_float(eq3b_cal.get("H_unseen_fold")),
        "H_unseen_fold_cal": _safe_float(eq3b_cal.get("H_unseen_fold_cal")),
        "H_cal": _safe_float(eq3b_cal.get("H_cal")),
        "T_opt": _safe_float(eq3b_cal.get("T_opt")),
        "Delta_cal": _safe_float(eq3b_cal.get("Delta_cal")),
        "probe_megascale_ddg": probe_ms,
        "probe_contacts": probe_contacts,
        "probe_burial": probe_burial,
        "total_wall_sec": float(time.perf_counter() - t0),
    }
    result["headlines"] = _extract_checkpoint_headlines(result)
    return result


# --------------------------------------------------------------------------- #
# Cross-Checkpoint / Trajectory Degradation Profile Aggregation
# --------------------------------------------------------------------------- #


def _get_layer_series(doc: Mapping[str, Any], metric_key: str) -> tuple[list[float | None], float | None]:
    """Extract per-layer probe scores and zero-shot score for a canonical metric key."""
    if metric_key.startswith("megascale_"):
        st = metric_key[len("megascale_") :].removesuffix("_spearman")
        ms = doc.get("probe_megascale_ddg") or {}
        layers = ms.get("layers") or []
        scores = [
            _safe_float(((lay.get("by_stratum") or {}).get(st) or {}).get("spearman_mean"))
            for lay in layers
        ]
        zs = _safe_float((((ms.get("zeroshot") or {}).get("raw") or {}).get(st) or {}).get("spearman_mean"))
        return scores, zs
    if metric_key == "contacts_long_range_sep_ge_24_auroc":
        ct = doc.get("probe_contacts") or {}
        layers = ct.get("layers") or []
        scores = [_safe_float((lay.get("long_range_sep_ge_24") or {}).get("auroc")) for lay in layers]
        zs = _safe_float(((ct.get("zeroshot_categorical_jacobian") or {}).get("long_range_sep_ge_24") or {}).get("auroc"))
        return scores, zs
    if metric_key == "contacts_long_range_sep_ge_24_p_at_L":
        ct = doc.get("probe_contacts") or {}
        layers = ct.get("layers") or []
        scores = [_safe_float((lay.get("long_range_sep_ge_24") or {}).get("p_at_L")) for lay in layers]
        zs = _safe_float(((ct.get("zeroshot_categorical_jacobian") or {}).get("long_range_sep_ge_24") or {}).get("p_at_L"))
        return scores, zs
    if metric_key == "burial_pooled_r2":
        bur = doc.get("probe_burial") or {}
        layers = bur.get("layers") or []
        scores = [_safe_float(lay.get("pooled_r2")) for lay in layers]
        return scores, None
    return [], None


def _classify_h1_vs_h2(
    *,
    zs_peak: float | None,
    zs_final: float | None,
    best_mid_peak: float | None,
    best_mid_final: float | None,
) -> str:
    """Classify trajectory collapse pattern into H1 (head pollution) vs H2 (deep overwriting)."""
    if zs_peak is None or zs_final is None or best_mid_peak is None or best_mid_final is None:
        return "insufficient_data"
    zs_drop = zs_peak - zs_final
    if zs_drop < 0.02:
        return "no_zeroshot_collapse"
    mid_retention = best_mid_final / max(best_mid_peak, 1e-6) if best_mid_peak > 1e-6 else 0.0
    zs_retention = zs_final / max(zs_peak, 1e-6) if zs_peak > 1e-6 else 0.0
    if mid_retention >= 0.75 and (mid_retention - zs_retention) >= 0.25:
        return "H1_output_head_pollution"
    if mid_retention <= 0.50:
        return "H2_deep_representation_overwriting"
    return "mixed_head_and_internal_degradation"


def aggregate_probe_results(
    checkpoint_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-checkpoint probe JSONs into a summary with ``layer_degradation_profile``."""
    per_checkpoint_rows: list[dict[str, Any]] = []
    trajectories_raw: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)

    for doc in checkpoint_results:
        arm_id = str(doc.get("arm_id", "unknown"))
        objective = str(doc.get("objective", "unknown"))
        scale_id = str(doc.get("scale_id", "unknown"))
        trajectories_raw[(arm_id, objective, scale_id)].append(doc)

        hl = doc.get("headlines") or _extract_checkpoint_headlines(doc)
        ms_unf = hl.get("megascale_unseen_fold_natural_spearman") or {}
        ms_dn = hl.get("megascale_de_novo_design_spearman") or {}
        ct_lr = hl.get("contacts_long_range_sep_ge_24_auroc") or {}
        bur = hl.get("burial_pooled_r2") or {}

        per_checkpoint_rows.append(
            {
                "checkpoint_id": str(doc.get("checkpoint_id", "")),
                "run_id": str(doc.get("run_id", "")),
                "arm_id": arm_id,
                "objective": objective,
                "paradigm_display": str(doc.get("paradigm_display", objective)),
                "scale_id": scale_id,
                "E": int(doc.get("E", doc.get("epoch", 0))),
                "H_unseen_fold": _safe_float(doc.get("H_unseen_fold")),
                "H_cal": _safe_float(doc.get("H_cal")),
                "T_opt": _safe_float(doc.get("T_opt")),
                "Delta_cal": _safe_float(doc.get("Delta_cal")),
                "megascale_unseen_fold": {
                    "best_layer_idx": ms_unf.get("best_layer_idx"),
                    "best_layer_name": ms_unf.get("best_layer_name"),
                    "best_layer_score": ms_unf.get("best_layer_score"),
                    "zeroshot_score": ms_unf.get("zeroshot_score"),
                    "calibrated_zeroshot_score": ms_unf.get("calibrated_zeroshot_score"),
                    "head_retention_ratio": ms_unf.get("head_retention_ratio"),
                    "best_layer_vs_zeroshot_gain": ms_unf.get("best_layer_vs_zeroshot_gain"),
                },
                "megascale_de_novo": {
                    "best_layer_idx": ms_dn.get("best_layer_idx"),
                    "best_layer_name": ms_dn.get("best_layer_name"),
                    "best_layer_score": ms_dn.get("best_layer_score"),
                    "zeroshot_score": ms_dn.get("zeroshot_score"),
                    "calibrated_zeroshot_score": ms_dn.get("calibrated_zeroshot_score"),
                    "head_retention_ratio": ms_dn.get("head_retention_ratio"),
                    "best_layer_vs_zeroshot_gain": ms_dn.get("best_layer_vs_zeroshot_gain"),
                },
                "contacts_lr_auroc": {
                    "best_layer_idx": ct_lr.get("best_layer_idx"),
                    "best_layer_score": ct_lr.get("best_layer_score"),
                    "zeroshot_score": ct_lr.get("zeroshot_score"),
                    "head_retention_ratio": ct_lr.get("head_retention_ratio"),
                    "best_layer_vs_zeroshot_gain": ct_lr.get("best_layer_vs_zeroshot_gain"),
                },
                "burial_pooled_r2": {
                    "best_layer_idx": bur.get("best_layer_idx"),
                    "best_layer_score": bur.get("best_layer_score"),
                    "final_norm_retention_ratio": bur.get("final_norm_retention_ratio"),
                },
            }
        )

    per_checkpoint_rows.sort(
        key=lambda r: (r["arm_id"], r["scale_id"], r["objective"], r["E"], r["checkpoint_id"])
    )

    trajectories_summary: dict[str, Any] = {}
    tracked_metrics = (
        "megascale_unseen_fold_natural_spearman",
        "megascale_de_novo_design_spearman",
        "contacts_long_range_sep_ge_24_auroc",
        "contacts_long_range_sep_ge_24_p_at_L",
        "burial_pooled_r2",
    )

    for (arm_id, objective, scale_id), docs in sorted(trajectories_raw.items()):
        docs_sorted = sorted(docs, key=lambda d: int(d.get("E", d.get("epoch", 0))))
        key = f"{arm_id}__{objective}__{scale_id}"
        layer_names = list(docs_sorted[-1].get("layer_names") or [])
        epochs = [int(d.get("E", d.get("epoch", 0))) for d in docs_sorted]
        pos_docs = [d for d in docs_sorted if int(d.get("E", d.get("epoch", 0))) > 0]
        final_doc = docs_sorted[-1]
        final_epoch = int(final_doc.get("E", final_doc.get("epoch", 0)))

        degradation_by_metric: dict[str, Any] = {}
        for mkey in tracked_metrics:
            # Choose E_peak over positive epochs maximizing zero-shot score (fallback: best-layer score)
            candidates = pos_docs if pos_docs else docs_sorted
            best_doc = None
            best_key_val = -float("inf")
            for cand in candidates:
                scores_c, zs_c = _get_layer_series(cand, mkey)
                finite_sc = [s for s in scores_c if s is not None]
                ref = zs_c if zs_c is not None else (max(finite_sc) if finite_sc else None)
                if ref is not None and ref > best_key_val:
                    best_key_val = ref
                    best_doc = cand

            if best_doc is None:
                continue

            peak_epoch = int(best_doc.get("E", best_doc.get("epoch", 0)))
            peak_scores, peak_zs = _get_layer_series(best_doc, mkey)
            final_scores, final_zs = _get_layer_series(final_doc, mkey)

            layer_profile: list[dict[str, Any]] = []
            for l_idx, l_name in enumerate(layer_names):
                p_val = peak_scores[l_idx] if l_idx < len(peak_scores) else None
                f_val = final_scores[l_idx] if l_idx < len(final_scores) else None
                delta = (
                    None if (p_val is None or f_val is None) else float(f_val - p_val)
                )
                ret = (
                    None
                    if (p_val is None or f_val is None or p_val <= 1e-6)
                    else float(f_val / p_val)
                )
                layer_profile.append(
                    {
                        "layer_idx": int(l_idx),
                        "layer_name": l_name,
                        "rel_depth": float(l_idx / max(1, len(layer_names) - 1)),
                        "score_at_E_peak": p_val,
                        "score_at_E_final": f_val,
                        "delta_peak_to_final": delta,
                        "retention_ratio_peak_to_final": ret,
                    }
                )

            # Intermediate layers (exclude l=0 embedding and final norm_f if n_taps > 2)
            mid_slice = slice(1, max(2, len(layer_names) - 1))
            mid_peak_vals = [s for s in peak_scores[mid_slice] if s is not None]
            mid_final_vals = [s for s in final_scores[mid_slice] if s is not None]
            best_mid_peak = max(mid_peak_vals) if mid_peak_vals else None
            best_mid_final = max(mid_final_vals) if mid_final_vals else None

            degradation_by_metric[mkey] = {
                "E_peak": peak_epoch,
                "E_final": final_epoch,
                "zeroshot_at_E_peak": peak_zs,
                "zeroshot_at_E_final": final_zs,
                "zeroshot_delta_peak_to_final": (
                    None if (peak_zs is None or final_zs is None) else float(final_zs - peak_zs)
                ),
                "best_intermediate_at_E_peak": best_mid_peak,
                "best_intermediate_at_E_final": best_mid_final,
                "hypothesis_verdict": _classify_h1_vs_h2(
                    zs_peak=peak_zs,
                    zs_final=final_zs,
                    best_mid_peak=best_mid_peak,
                    best_mid_final=best_mid_final,
                ),
                "per_layer": layer_profile,
            }

        trajectories_summary[key] = {
            "arm_id": arm_id,
            "objective": objective,
            "paradigm_display": str(final_doc.get("paradigm_display", objective)),
            "scale_id": scale_id,
            "epochs": epochs,
            "n_checkpoints": len(docs_sorted),
            "layer_names": layer_names,
            "layer_degradation_profile": degradation_by_metric,
        }

    return {
        "schema": PROBE_SUMMARY_SCHEMA,
        "n_checkpoints": len(per_checkpoint_rows),
        "checkpoints": per_checkpoint_rows,
        "trajectories": trajectories_summary,
    }
