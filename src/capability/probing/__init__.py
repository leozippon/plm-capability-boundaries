"""Frozen layerwise hidden-state linear probe suite for Stage-2 PLM checkpoints.

Tests H1 (Output-Head / Top-Layer Pollution) vs H2 (Deep Representation Overwriting)
across matched AR, MLM-15%, and MDM (``oa_uniform``) checkpoints using:
  - Exact variable-length layerwise hidden-state extraction (``extractor.py``)
  - Closed-form SVD/Gram Ridge probes for MegaScale ddG, unseen-fold 3D contacts,
    and unseen-fold per-residue 3D burial / coordination number (``probes.py``)
  - Eq. (3b) temperature calibration and cross-epoch degradation analysis (``analysis.py``)
"""
from __future__ import annotations

from .analysis import (
    PROBE_CHECKPOINT_SCHEMA,
    PROBE_SUMMARY_SCHEMA,
    aggregate_probe_results,
    compute_layer_retention_metrics,
    evaluate_checkpoint_probes,
)
from .extractor import (
    LoadedCheckpoint,
    canonical_objective,
    extract_layerwise_hidden_states,
    get_layer_names,
    load_stage2_checkpoint,
    paradigm_display_name,
)
from .probes import (
    DEFAULT_RIDGE_LAMBDAS,
    FoldSufficientStats,
    evaluate_burial_layerwise_probe,
    evaluate_contact_layerwise_probe,
    evaluate_megascale_layerwise_probe,
    evaluate_unseen_fold_temperature_calibration,
    fit_ridge_from_stats_cv,
    fit_ridge_svd_cv,
)

__all__ = [
    "DEFAULT_RIDGE_LAMBDAS",
    "FoldSufficientStats",
    "LoadedCheckpoint",
    "PROBE_CHECKPOINT_SCHEMA",
    "PROBE_SUMMARY_SCHEMA",
    "aggregate_probe_results",
    "canonical_objective",
    "compute_layer_retention_metrics",
    "evaluate_burial_layerwise_probe",
    "evaluate_checkpoint_probes",
    "evaluate_contact_layerwise_probe",
    "evaluate_megascale_layerwise_probe",
    "evaluate_unseen_fold_temperature_calibration",
    "extract_layerwise_hidden_states",
    "fit_ridge_from_stats_cv",
    "fit_ridge_svd_cv",
    "get_layer_names",
    "load_stage2_checkpoint",
    "paradigm_display_name",
]
