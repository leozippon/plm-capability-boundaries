"""Two-factor group interval for the two-factor crossing.

A substituted-symbol-only interval resamples only the substituted symbol. That
prices pair composition and not the Swiss-Prot cohort draw. This interval
therefore resamples two declared factors:

* near-duplicate **sequence groups** of the scored cohort;
* **substituted-symbol groups** of the measured pairs.

The same sequence-group draw reweights the arm and the matching fragment
ceiling, so the difference stays paired. There is no record, token or position
fallback: too few units is a refusal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .information_bootstrap import effective_unit_floor, kish_effective_units
from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS, MINIMUM_FINITE_DRAW_FRACTION


def _pair_means(
    sums: np.ndarray, counts: np.ndarray, record_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Token-weighted pair means under a per-record multiplicity."""

    weight = np.asarray(record_weight, dtype=np.float64)
    numerator = sums @ weight
    denominator = counts @ weight
    return numerator, denominator


def _quadrant_delta(codes: np.ndarray, values: np.ndarray) -> float:
    high = values[codes > 0]
    low = values[codes < 0]
    if high.size == 0 or low.size == 0:
        return float("nan")
    return float(high.mean() - low.mean())


def crossed_group_interval(
    *,
    codes: np.ndarray,
    symbol_groups: np.ndarray,
    sequence_groups: np.ndarray,
    arm_sum: np.ndarray,
    arm_count: np.ndarray,
    ceiling_sum: np.ndarray,
    ceiling_count: np.ndarray,
    seed: int,
    n_draws: int = 2000,
) -> dict[str, Any]:
    """Paired arm-minus-ceiling interval under a crossed group draw.

    ``arm_sum[p, r]`` / ``arm_count[p, r]`` are the sufficient statistics of pair
    ``p`` on record ``r``; the ceiling matrices are the same shape. Pair damage
    is the token-weighted mean, which is the estimand here.
    """

    codes = np.asarray(codes, dtype=np.int64)
    symbol_groups = np.asarray(symbol_groups, dtype=np.int64)
    sequence_groups = np.asarray(sequence_groups, dtype=np.int64)
    arm_sum = np.asarray(arm_sum, dtype=np.float64)
    arm_count = np.asarray(arm_count, dtype=np.float64)
    ceiling_sum = np.asarray(ceiling_sum, dtype=np.float64)
    ceiling_count = np.asarray(ceiling_count, dtype=np.float64)
    if codes.ndim != 1:
        raise ValueError("codes must be one value per pair")
    n_pairs = int(codes.size)
    n_records = int(sequence_groups.size)
    for name, matrix in (
        ("arm_sum", arm_sum),
        ("arm_count", arm_count),
        ("ceiling_sum", ceiling_sum),
        ("ceiling_count", ceiling_count),
    ):
        if matrix.shape != (n_pairs, n_records):
            raise ValueError(
                f"{name} has shape {matrix.shape}, not {(n_pairs, n_records)}"
            )
    if symbol_groups.shape != codes.shape:
        raise ValueError("symbol groups must align with pairs")
    if n_draws < 1:
        raise ValueError("n_draws must be positive")

    ones = np.ones(n_records, dtype=np.float64)
    arm_num, arm_den = _pair_means(arm_sum, arm_count, ones)
    ceil_num, ceil_den = _pair_means(ceiling_sum, ceiling_count, ones)
    defined = (arm_den > 0.0) & (ceil_den > 0.0)
    if not bool(defined.any()):
        raise ValueError("no pair has scored tokens on both the arm and the ceiling")
    if not bool(((codes[defined] > 0).any() and (codes[defined] < 0).any())):
        raise ValueError("the defined pairs do not populate both contrast classes")

    arm_point = np.divide(arm_num, arm_den, out=np.full(n_pairs, np.nan), where=defined)
    ceil_point = np.divide(ceil_num, ceil_den, out=np.full(n_pairs, np.nan), where=defined)
    arm_delta = _quadrant_delta(codes[defined], arm_point[defined])
    ceiling_delta = _quadrant_delta(codes[defined], ceil_point[defined])
    difference = float(arm_delta - ceiling_delta)

    unique_sequences = np.unique(sequence_groups)
    unique_symbols = np.unique(symbol_groups)
    sequence_tokens = np.array(
        [
            float(arm_count[:, sequence_groups == group].sum())
            for group in unique_sequences
        ],
        dtype=np.float64,
    )
    symbol_tokens = np.array(
        [
            float(arm_count[symbol_groups == group].sum())
            for group in unique_symbols
        ],
        dtype=np.float64,
    )
    n_effective_sequences = kish_effective_units(sequence_tokens)
    n_effective_symbols = kish_effective_units(symbol_tokens)
    sequence_floor = effective_unit_floor(n_effective_sequences, int(unique_sequences.size))
    symbol_floor = effective_unit_floor(n_effective_symbols, int(unique_symbols.size))
    raw_sequence_short = int(unique_sequences.size) < MINIMUM_BOOTSTRAP_UNITS
    raw_symbol_short = int(unique_symbols.size) < MINIMUM_BOOTSTRAP_UNITS
    refused = bool(
        sequence_floor["degenerate"]
        or symbol_floor["degenerate"]
        or raw_sequence_short
        or raw_symbol_short
    )
    units = {
        "n_sequence_groups": int(unique_sequences.size),
        "n_effective_sequence_groups": n_effective_sequences,
        "n_symbol_groups": int(unique_symbols.size),
        "n_effective_symbol_groups": n_effective_symbols,
        "sequence_floor": sequence_floor,
        "symbol_floor": symbol_floor,
        "minimum_units": int(MINIMUM_BOOTSTRAP_UNITS),
        "n_draws": int(n_draws),
    }
    block = {
        "delta": float(arm_delta),
        "reference_delta": float(ceiling_delta),
        "reference_name": "matching_fragment_conditional",
        "difference": difference,
        "n_pairs": int(defined.sum()),
        "n_pairs_positive_class": int((codes[defined] > 0).sum()),
        "n_pairs_negative_class": int((codes[defined] < 0).sum()),
        "resampling_unit": (
            "crossed near-duplicate sequence groups and substituted-symbol groups"
        ),
        "units": units,
        "refused": refused,
    }
    if refused:
        reasons = []
        if raw_sequence_short or sequence_floor["degenerate"]:
            reasons.append(sequence_floor["degenerate_reason"] or (
                f"{unique_sequences.size} sequence groups is below the "
                f"{MINIMUM_BOOTSTRAP_UNITS}-unit floor"
            ))
        if raw_symbol_short or symbol_floor["degenerate"]:
            reasons.append(symbol_floor["degenerate_reason"] or (
                f"{unique_symbols.size} symbol groups is below the "
                f"{MINIMUM_BOOTSTRAP_UNITS}-unit floor"
            ))
        block["delta_ci95"] = None
        block["reference_delta_ci95"] = None
        block["difference_ci95"] = None
        block["refusal"] = "; ".join(reasons)
        return block

    rng = np.random.default_rng(seed)
    n_seq = int(unique_sequences.size)
    n_sym = int(unique_symbols.size)
    seq_index = {int(group): i for i, group in enumerate(unique_sequences)}
    record_group = np.asarray([seq_index[int(group)] for group in sequence_groups])
    arm_draws: list[float] = []
    ceiling_draws: list[float] = []
    differences: list[float] = []
    for _ in range(n_draws):
        seq_draw = rng.choice(n_seq, size=n_seq, replace=True)
        seq_weight = np.bincount(seq_draw, minlength=n_seq).astype(np.float64)
        record_weight = seq_weight[record_group]
        arm_n, arm_d = _pair_means(arm_sum, arm_count, record_weight)
        ceil_n, ceil_d = _pair_means(ceiling_sum, ceiling_count, record_weight)
        live = (arm_d > 0.0) & (ceil_d > 0.0)
        if not bool(live.any()):
            continue
        arm_means = np.divide(arm_n, arm_d, out=np.full(n_pairs, np.nan), where=live)
        ceil_means = np.divide(ceil_n, ceil_d, out=np.full(n_pairs, np.nan), where=live)
        sym_draw = rng.choice(n_sym, size=n_sym, replace=True)
        selected: list[int] = []
        for group in unique_symbols[sym_draw]:
            selected.extend(np.flatnonzero(symbol_groups == group).tolist())
        chosen = np.asarray(selected, dtype=np.int64)
        chosen = chosen[live[chosen]]
        if chosen.size == 0:
            continue
        drawn_codes = codes[chosen]
        if not bool(((drawn_codes > 0).any() and (drawn_codes < 0).any())):
            continue
        arm_draw = _quadrant_delta(drawn_codes, arm_means[chosen])
        ceil_draw = _quadrant_delta(drawn_codes, ceil_means[chosen])
        if not (np.isfinite(arm_draw) and np.isfinite(ceil_draw)):
            continue
        arm_draws.append(float(arm_draw))
        ceiling_draws.append(float(ceil_draw))
        differences.append(float(arm_draw - ceil_draw))

    if not differences:
        raise RuntimeError("no finite crossed-group draws were produced")
    minimum_draws = int(np.ceil(MINIMUM_FINITE_DRAW_FRACTION * n_draws))
    if len(differences) < minimum_draws:
        raise RuntimeError(
            f"only {len(differences)} of {n_draws} crossed-group draws were finite, "
            f"below the {MINIMUM_FINITE_DRAW_FRACTION:.0%} floor"
        )
    block["delta_ci95"] = [
        float(np.percentile(arm_draws, 2.5)),
        float(np.percentile(arm_draws, 97.5)),
    ]
    block["reference_delta_ci95"] = [
        float(np.percentile(ceiling_draws, 2.5)),
        float(np.percentile(ceiling_draws, 97.5)),
    ]
    block["difference_ci95"] = [
        float(np.percentile(differences, 2.5)),
        float(np.percentile(differences, 97.5)),
    ]
    block["n_finite_draws"] = len(differences)
    block["n_non_finite_draws"] = int(n_draws) - len(differences)
    block["n_draws_requested"] = int(n_draws)
    return block
