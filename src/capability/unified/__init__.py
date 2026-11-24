"""Unified cross-paradigm roster, deterministic resolvers, and yardstick package.

Single authoritative source for the 65-arm cross-paradigm benchmark universe
(62 single-sequence arms + 3 multi-sequence MSA context arms), keeping the
frozen 33-arm ``ROSTER`` and ``PANEL`` completely untouched while exposing one
consistent yardstick interface across Results 1–4.

Submodules (``r1_yardstick``, ``r2_experiment_2_1``, ``r3_yardstick``,
``contacts_yardstick``, ``memory_and_generation_yardstick``,
``r5_experiment_4_1``, ``r6_experiment_4_2``, ``r6_experiment_4_3``,
``master_summary``) are exposed via PEP 562 lazy attribute loading so importing
``src.capability.unified`` never eagerly loads PyTorch or heavy evaluation code.
"""

from __future__ import annotations

import importlib
from typing import Any

LAZY_SUBMODULES: tuple[str, ...] = (
    "roster",
    "r1_yardstick",
    "r2_experiment_2_1",
    "r3_yardstick",
    "contacts_yardstick",
    "memory_and_generation_yardstick",
    "r5_experiment_4_1",
    "r6_experiment_4_2",
    "r6_experiment_4_3",
    "master_summary",
)

ROSTER_EXPORTS: tuple[str, ...] = (
    "ARM_METADATA",
    "EXPANDED_32_ARMS",
    "PARADIGM_ORDER",
    "PREREGISTERED_33_ARMS",
    "UNIFIED_3_MSA_CONTEXT_ARMS",
    "UNIFIED_62_SINGLE_SEQUENCE_ARMS",
    "UNIFIED_62_SINGLE_SEQUENCE_ARMS_ALPHA",
    "UNIFIED_65_ARMS",
    "UNIFIED_65_ARMS_ALPHA",
    "UNIFIED_OUTPUT_DIR",
    "ArmMetadata",
    "ResolvedArtifact",
    "arms_by_paradigm",
    "default_scan_dirs",
    "dtype_rank",
    "get_arm_metadata",
    "resolve_all_gp_r1_readouts",
    "resolve_all_gp_r1_scores",
    "resolve_gp_family_distribution_shards",
    "resolve_gp_r1_json",
    "resolve_gp_r1_readout",
    "resolve_gp_r1_scores",
    "resolve_gp_r2_arrays",
    "resolve_gp_r3_cycle_eval",
    "resolve_gp_r3_cycles",
    "resolve_gp_r4_json",
    "resolve_gp_r5_json",
    "sort_arms",
)

_ROSTER_EXPORT_SET = frozenset(ROSTER_EXPORTS)
_LAZY_SUBMODULE_SET = frozenset(LAZY_SUBMODULES)


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULE_SET:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name in _ROSTER_EXPORT_SET:
        roster_mod = importlib.import_module(".roster", __name__)
        val = getattr(roster_mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULE_SET | _ROSTER_EXPORT_SET)


__all__ = sorted({"LAZY_SUBMODULES", "ROSTER_EXPORTS"} | _LAZY_SUBMODULE_SET | _ROSTER_EXPORT_SET)
