"""Reusable external benchmark interface for protein language and generative models.

Allows any third-party or first-party model to be validated (including strict
anti-leakage checks), evaluated across Results 1–6 (smoke or full cohorts), and
ranked directly against the sealed 62–84 model unified reference leaderboards.
"""

from __future__ import annotations

from src.capability.benchmark.adapters import (
    HuggingFaceCausalPLMAdapter,
    HuggingFaceMaskedPLMAdapter,
    load_builtin_adapter,
    load_custom_adapter,
)
from src.capability.benchmark.protocol import (
    VALID_CONDITIONING_REGIMES,
    VALID_INPUT_ALIGNMENT_FORMATS,
    AdapterValidationError,
    BaseProteinBenchmarkAdapter,
    CapabilityFlags,
    CapabilityNotSupportedError,
    ModelMetadata,
    ProteinBenchmarkAdapter,
    pack_mid_and_last_hidden_states,
    validate_adapter,
)
from src.capability.benchmark.runner import (
    ALL_SUITES,
    ReferenceLeaderboard,
    compute_rank_and_percentile,
    load_reference_leaderboards,
    run_benchmark,
)

__all__ = [
    "ALL_SUITES",
    "AdapterValidationError",
    "BaseProteinBenchmarkAdapter",
    "CapabilityFlags",
    "CapabilityNotSupportedError",
    "HuggingFaceCausalPLMAdapter",
    "HuggingFaceMaskedPLMAdapter",
    "ModelMetadata",
    "ProteinBenchmarkAdapter",
    "ReferenceLeaderboard",
    "VALID_CONDITIONING_REGIMES",
    "VALID_INPUT_ALIGNMENT_FORMATS",
    "compute_rank_and_percentile",
    "load_builtin_adapter",
    "load_custom_adapter",
    "load_reference_leaderboards",
    "pack_mid_and_last_hidden_states",
    "run_benchmark",
    "validate_adapter",
]
