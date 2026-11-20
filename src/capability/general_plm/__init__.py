"""General-PLM exploratory model registry, adapters, and scoring utilities.

All models in this package belong strictly to the exploratory general-PLM
extension (``general_plm_20261120``). Nothing here mutates or widens the
frozen 33-arm causal ``ROSTER`` or ``PANEL``.
"""

from __future__ import annotations

from .registry import (
    DEFAULT_SEARCH_ROOTS,
    GENERAL_PLM_REGISTRY,
    GeneralPLMSpec,
    arms_by_family,
    arms_for_module,
    get_spec,
    list_arms,
    resolve_checkpoint_dir,
)
from .adapters import (
    GeneralPLMAdapter,
    PositionVector,
    load_general_plm,
    parse_mutation,
    apply_mutation,
)

__all__ = [
    "DEFAULT_SEARCH_ROOTS",
    "GENERAL_PLM_REGISTRY",
    "GeneralPLMSpec",
    "GeneralPLMAdapter",
    "PositionVector",
    "arms_by_family",
    "arms_for_module",
    "get_spec",
    "list_arms",
    "resolve_checkpoint_dir",
    "load_general_plm",
    "parse_mutation",
    "apply_mutation",
]
