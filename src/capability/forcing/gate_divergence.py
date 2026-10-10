"""The gate endpoint: a pair-resolved double difference whose null is exactly zero.

For one unit -- one backbone and one prescribed pair -- the two conditions of the
reciprocal swap give two realised residue distributions at every read position.
``D`` is the Jensen-Shannon divergence between them, in nats. The endpoint is

    D(prescribed partner) - mean over the matched non-contacting reference partners

and it is zero whenever forcing the foreign residue changes the completion in a
way that is not specific to the prescribed partner's position. That is the whole
point of the second term: a shift in composition, in entropy, in quality or in
termination behaviour moves ``D`` at every position of the span and cancels, and a
divergence estimator's own finite-sample positive bias cancels with it as long as
the two terms are estimated from the same number of draws at matched separation
and matched reference burial -- which is what the cohort's matching rule
guarantees. :func:`permutation_calibration` measures the residual rather than
trusting the argument.

Two estimators of the realised distribution, both under the declared decoding
policy and both reported.

``empirical``
    the histogram of the residues the draws actually realised at the position.
    This is the literal reading of "the realised residue distribution" and it is
    the pre-registered primary.
``mixture``
    the mean over draws of the sampling distribution the model used at that
    position, which is the same marginal estimated with the per-draw conditional
    rather than a single sample from it. It has the same expectation and far less
    variance, and it costs nothing because the forward pass already produced the
    row. It is reported beside the primary, never instead of it.

The sampling unit is the backbone, never the draw: a unit's value is the mean
over the prescribed pairs of that backbone, and the interval resamples backbones.
Because one backbone per CATH superfamily is admitted, that is also a
superfamily-level resample.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..core.amino_acids import AA20
from ..generation.near_duplicates import near_duplicate_groups
from . import forcing_design as D

SCHEMA = "forcing_gate_divergence_v1"

#: Fewest usable draws a condition must contribute at a position for the
#: divergence there to be estimated. Below it the histogram is noise and the cell
#: contributes nothing rather than a value with an unstated support.
MIN_DRAWS_PER_CONDITION = 8

#: Fewest usable draws per condition inside a hydrophobic-fraction bin. Lower
#: than the cell floor because a bin is half a cell by construction; a bin below
#: it is skipped and the number of skipped bins is reported.
MIN_DRAWS_PER_BIN = 6

#: The reported variants of the primary endpoint, each a declared filter setting
#: rather than a search. ``primary`` is the pre-registered one.
VARIANTS: Mapping[str, Mapping[str, Any]] = {
    "primary": {"completed_only": False, "exclude_repeat_flagged": False, "deduplicate": True},
    "completed_only": {"completed_only": True, "exclude_repeat_flagged": False, "deduplicate": True},
    "repeat_excluded": {"completed_only": False, "exclude_repeat_flagged": True, "deduplicate": True},
    "without_deduplication": {
        "completed_only": False, "exclude_repeat_flagged": False, "deduplicate": False,
    },
}

ESTIMATORS = ("empirical", "mixture")


# ------------------------------------------------------------- the divergence


def jensen_shannon(first: np.ndarray, second: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats, bounded above by ``ln 2``.

    Symmetric and finite even where one distribution puts zero mass, which the
    declared nucleus truncation guarantees it sometimes does; a KL-based contrast
    would be infinite there and a clamp would be an undeclared smoothing.
    """

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("two distributions over one alphabet are compared")
    for array in (left, right):
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError("a distribution is finite and non-negative")
        if not np.isclose(array.sum(), 1.0, atol=1e-9):
            raise ValueError(f"a distribution sums to one, got {array.sum()}")
    middle = 0.5 * (left + right)
    return float(0.5 * _kl(left, middle) + 0.5 * _kl(right, middle))


def _kl(from_: np.ndarray, to: np.ndarray) -> float:
    support = from_ > 0
    return float((from_[support] * np.log(from_[support] / to[support])).sum())


def renormalise(array: Any) -> np.ndarray:
    """Rows of a stored distribution array, renormalised in float64.

    The generation stage stores distributions as float32 so a campaign's arrays
    stay small, which leaves the rows summing to one only to float32 precision.
    Renormalising at the point of use rather than loosening the divergence's own
    normalisation check keeps that check able to catch a row that is genuinely not
    a distribution.
    """

    rows = np.asarray(array, dtype=np.float64)
    totals = rows.sum(axis=-1, keepdims=True)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("a stored distribution row carries no mass")
    return rows / totals


def histogram(residues: Sequence[str]) -> np.ndarray | None:
    """The residue histogram of a set of realised draws, or ``None`` if empty."""

    counts = np.zeros(len(AA20), dtype=np.float64)
    column = {residue: index for index, residue in enumerate(AA20)}
    used = 0
    for residue in residues:
        if residue is None:
            continue
        if residue not in column:
            raise ValueError(f"realised residue {residue!r} is not canonical")
        counts[column[residue]] += 1.0
        used += 1
    if used == 0:
        return None
    return counts / counts.sum()


# ----------------------------------------------------------- the draw selection


def duplicate_groups(completions: Sequence[str]) -> np.ndarray:
    """Union-find near-duplicate groups over the pooled completions of one cell.

    Pooled across both conditions on purpose: a completion that appears under
    both forced anchors is one sequence, and grouping the conditions separately
    would let it be a distinct unit on one side and a duplicate on the other.
    """

    if not completions:
        return np.zeros(0, dtype=np.int64)
    groups, _summary = near_duplicate_groups(list(completions), unit="residues")
    return groups


def select_draws(
    native: Sequence[Mapping[str, Any]], transplant: Sequence[Mapping[str, Any]], *,
    completed_only: bool, exclude_repeat_flagged: bool, deduplicate: bool,
) -> dict[str, Any]:
    """Which draws of one cell enter the realised distributions, and why.

    One selection for both conditions, applied to the pooled draw list so the
    duplicate grouping cannot see the condition label. A kept draw's index into
    its own condition's list is returned, so a caller reads the same draws from
    the per-draw distributions.
    """

    pooled = [(D.CONDITION_NATIVE, index, row) for index, row in enumerate(native)]
    pooled += [(D.CONDITION_TRANSPLANT, index, row) for index, row in enumerate(transplant)]
    groups = duplicate_groups([str(row["completion"]) for _, _, row in pooled])
    kept: dict[str, list[int]] = {condition: [] for condition in D.CONDITIONS}
    seen: set[tuple[str, int]] = set()
    dropped = {"censored": 0, "repeat_flagged": 0, "duplicate": 0}
    for (condition, index, row), group in zip(pooled, groups):
        if completed_only and row["censored"]:
            dropped["censored"] += 1
            continue
        if exclude_repeat_flagged and row["repeat_flagged"]:
            dropped["repeat_flagged"] += 1
            continue
        if deduplicate:
            key = (condition, int(group))
            if key in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(key)
        kept[condition].append(index)
    return {
        "kept": {condition: sorted(values) for condition, values in kept.items()},
        "dropped": dropped,
        "duplicate_groups": int(len(set(groups.tolist()))),
        "pooled_draws": len(pooled),
    }


# ------------------------------------------------------------- the cell endpoint


def _distribution(
    rows: Sequence[Mapping[str, Any]], indices: Sequence[int], *, position: int,
    estimator: str, distributions: np.ndarray | None, slot: int,
) -> tuple[np.ndarray | None, int]:
    """One condition's realised distribution at one position, and its support."""

    if estimator == "empirical":
        realised = [rows[index]["realised"].get(str(position)) for index in indices]
        usable = [residue for residue in realised if residue is not None]
        return histogram(usable), len(usable)
    if estimator == "mixture":
        if distributions is None:
            raise ValueError("the mixture estimator needs the per-draw distributions")
        usable = [
            index for index in indices
            if rows[index]["realised"].get(str(position)) is not None
        ]
        if not usable:
            return None, 0
        mass = np.asarray(distributions[usable, slot], dtype=np.float64).mean(axis=0)
        total = mass.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("a mixture of sampling distributions carries no mass")
        return mass / total, len(usable)
    raise ValueError(f"unknown estimator {estimator!r}; declared: {ESTIMATORS}")


def cell_endpoint(
    cell: Mapping[str, Any], *, estimator: str = "empirical",
    completed_only: bool = False, exclude_repeat_flagged: bool = False,
    deduplicate: bool = True, minimum: int = MIN_DRAWS_PER_CONDITION,
    draw_indices: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """``D_j - mean(D_j')`` for one unit, or the reason it has no value.

    ``draw_indices`` overrides the selection, which is how
    :func:`permutation_calibration` reuses this function with the condition label
    permuted: the permutation has to change which draws are called native, not
    which draws are usable.
    """

    native, transplant = cell["native"], cell["transplant"]
    selection = (
        {"kept": {key: list(value) for key, value in draw_indices.items()},
         "dropped": None, "duplicate_groups": None, "pooled_draws": None}
        if draw_indices is not None
        else select_draws(
            native["draws"], transplant["draws"], completed_only=completed_only,
            exclude_repeat_flagged=exclude_repeat_flagged, deduplicate=deduplicate,
        )
    )
    positions = [int(cell["partner"]), *[int(value) for value in cell["reference_positions"]]]
    divergences: list[dict[str, Any]] = []
    for slot, position in enumerate(positions):
        left, left_n = _distribution(
            native["draws"], selection["kept"][D.CONDITION_NATIVE], position=position,
            estimator=estimator, distributions=native.get("distributions"), slot=slot,
        )
        right, right_n = _distribution(
            transplant["draws"], selection["kept"][D.CONDITION_TRANSPLANT], position=position,
            estimator=estimator, distributions=transplant.get("distributions"), slot=slot,
        )
        resolved = (
            left is not None and right is not None
            and left_n >= int(minimum) and right_n >= int(minimum)
        )
        divergences.append(
            {
                "position": position,
                "role": "prescribed_partner" if slot == 0 else "reference_partner",
                "divergence_nats": jensen_shannon(left, right) if resolved else None,
                "native_draws": left_n,
                "transplant_draws": right_n,
                "native_entropy_nats": _entropy(left) if left is not None else None,
                "transplant_entropy_nats": _entropy(right) if right is not None else None,
            }
        )
    partner = divergences[0]["divergence_nats"]
    references = [row["divergence_nats"] for row in divergences[1:] if row["divergence_nats"] is not None]
    endpoint = (
        None if partner is None or not references
        else float(partner - float(np.mean(references)))
    )
    return {
        "unit_id": cell["unit_id"],
        "accession": cell["accession"],
        "length_band": cell["length_band"],
        "estimator": estimator,
        "endpoint_nats": endpoint,
        "partner_divergence_nats": partner,
        "reference_divergence_nats": float(np.mean(references)) if references else None,
        "references_resolved": len(references),
        "divergences": divergences,
        "selection": selection,
        "undefined": None if endpoint is not None else (
            "the prescribed partner or every reference partner lacks the minimum draws "
            f"per condition ({minimum})"
        ),
    }


def _entropy(distribution: np.ndarray) -> float:
    support = distribution > 0
    return float(-(distribution[support] * np.log(distribution[support])).sum())


# --------------------------------------------------- the hydrophobic-fraction bins


def hydrophobic_bin_endpoint(
    cell: Mapping[str, Any], *, estimator: str = "empirical",
    minimum: int = MIN_DRAWS_PER_BIN,
) -> dict[str, Any]:
    """The endpoint inside bins of the completion's own hydrophobic fraction.

    A median split of the cell's pooled draws rather than fixed edges, so both
    bins are populated in every cell; two bins rather than three because a cell
    holds 48 draws and a three-way split leaves a divergence estimated from eight
    samples. The bin value is the equal-weighted mean over the resolved bins, so a
    cell whose effect lives in one composition regime is not averaged away by the
    other's noise.
    """

    native, transplant = cell["native"], cell["transplant"]
    base = select_draws(
        native["draws"], transplant["draws"], completed_only=False,
        exclude_repeat_flagged=False, deduplicate=True,
    )
    fractions = [
        row["hydrophobic_fraction"]
        for condition, rows in ((D.CONDITION_NATIVE, native["draws"]),
                                (D.CONDITION_TRANSPLANT, transplant["draws"]))
        for index, row in enumerate(rows)
        if index in base["kept"][condition] and row["hydrophobic_fraction"] is not None
    ]
    if not fractions:
        return {"unit_id": cell["unit_id"], "endpoint_nats": None,
                "undefined": "no kept draw carries a hydrophobic fraction", "bins": []}
    boundary = float(np.median(fractions))
    bins: list[dict[str, Any]] = []
    for name, predicate in (
        ("low", lambda value: value <= boundary),
        ("high", lambda value: value > boundary),
    ):
        indices = {
            condition: [
                index for index in base["kept"][condition]
                if (rows[index]["hydrophobic_fraction"] is not None
                    and predicate(rows[index]["hydrophobic_fraction"]))
            ]
            for condition, rows in ((D.CONDITION_NATIVE, native["draws"]),
                                    (D.CONDITION_TRANSPLANT, transplant["draws"]))
        }
        record = cell_endpoint(
            cell, estimator=estimator, minimum=int(minimum), draw_indices=indices,
        )
        bins.append({"bin": name, "boundary": boundary,
                     "draws": {key: len(value) for key, value in indices.items()},
                     "endpoint_nats": record["endpoint_nats"]})
    resolved = [row["endpoint_nats"] for row in bins if row["endpoint_nats"] is not None]
    return {
        "unit_id": cell["unit_id"],
        "accession": cell["accession"],
        "length_band": cell["length_band"],
        "boundary": boundary,
        "bins": bins,
        "resolved_bins": len(resolved),
        "endpoint_nats": float(np.mean(resolved)) if resolved else None,
        "rule": (
            "median split of the cell's own kept draws by completion hydrophobic "
            "fraction; the cell value is the equal-weighted mean over resolved bins"
        ),
    }


# -------------------------------------------------- teacher-forced decomposition


def teacher_forced_endpoint(
    cell: Mapping[str, Any], *, resample_draws: int = D.DRAWS_PER_CELL,
    seed: int = D.SAMPLING_SEED,
) -> dict[str, Any]:
    """The endpoint from the exact conditionals, and its matched-support companion.

    Two numbers, because one alone would mislead. ``exact_nats`` is the double
    difference between the exact conditionals, which carries no sampling bias at
    all. ``resampled_nats`` draws the same number of samples per condition the
    sampled mode uses and recomputes the endpoint with the *same* empirical
    estimator, so the sampled-minus-teacher-forced difference -- the quantity that
    isolates the model's own intervening commitments -- is a difference of like
    with like rather than of a biased estimator with an unbiased one.
    """

    native = renormalise(cell["native"]["distributions"])
    transplant = renormalise(cell["transplant"]["distributions"])
    if native.shape != transplant.shape or native.ndim != 2 or native.shape[1] != len(AA20):
        raise ValueError("teacher-forced conditionals are (positions, 20) and share a shape")
    positions = [int(cell["partner"]), *[int(value) for value in cell["reference_positions"]]]
    if native.shape[0] != len(positions):
        raise ValueError("the conditionals do not carry one row per read position")
    exact = [jensen_shannon(native[slot], transplant[slot]) for slot in range(len(positions))]
    generator = np.random.default_rng(int(seed))
    resampled = []
    for slot in range(len(positions)):
        left = histogram(
            [AA20[index] for index in generator.choice(
                len(AA20), size=int(resample_draws), p=native[slot])]
        )
        right = histogram(
            [AA20[index] for index in generator.choice(
                len(AA20), size=int(resample_draws), p=transplant[slot])]
        )
        resampled.append(jensen_shannon(left, right))
    return {
        "unit_id": cell["unit_id"],
        "accession": cell["accession"],
        "length_band": cell["length_band"],
        "exact_nats": float(exact[0] - float(np.mean(exact[1:]))),
        "resampled_nats": float(resampled[0] - float(np.mean(resampled[1:]))),
        "resample_draws": int(resample_draws),
        "exact_divergences": [
            {"position": position, "divergence_nats": value}
            for position, value in zip(positions, exact)
        ],
    }


# --------------------------------------------------------------- the aggregation


def backbone_values(records: Sequence[Mapping[str, Any]], *, key: str = "endpoint_nats") -> dict[str, float]:
    """One value per backbone: the equal-weighted mean over its resolved units."""

    grouped: dict[str, list[float]] = {}
    for record in records:
        value = record.get(key)
        if value is None or not np.isfinite(float(value)):
            continue
        grouped.setdefault(str(record["accession"]), []).append(float(value))
    return {
        accession: float(np.mean(values)) for accession, values in sorted(grouped.items())
    }


def bootstrap_interval(
    values: Mapping[str, float], *, draws: int = D.BOOTSTRAP_DRAWS, seed: int = D.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Percentile interval on the mean over backbones, resampling backbones.

    The unit is the backbone and nothing else. Resampling draws would treat 24
    samples of one prompt as 24 independent observations of a protein, which is
    how a null becomes a finding.
    """

    ordered = [float(values[key]) for key in sorted(values)]
    if len(ordered) < 2:
        return {
            "point": float(ordered[0]) if ordered else None,
            "interval": None, "units": len(ordered), "excludes_zero": None,
            "undefined": "an interval over backbones needs at least two backbones",
        }
    array = np.asarray(ordered, dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    means = array[generator.integers(array.size, size=(int(draws), array.size))].mean(axis=1)
    low, high = (float(value) for value in np.percentile(means, [2.5, 97.5]))
    return {
        "point": float(array.mean()),
        "interval": [low, high],
        "units": int(array.size),
        "unit_sd": float(array.std(ddof=1)),
        "standard_error": float(array.std(ddof=1) / np.sqrt(array.size)),
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "draws": int(draws),
        "seed": int(seed),
        "unit": D.SAMPLING_UNIT,
    }


def by_stratum(
    records: Sequence[Mapping[str, Any]], *, stratum: str,
    draws: int = D.BOOTSTRAP_DRAWS, seed: int = D.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """The endpoint inside each level of one declared stratum, never pooled across them."""

    levels: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        level = record.get(stratum)
        if level is None:
            continue
        levels.setdefault(str(level), []).append(record)
    return {
        level: bootstrap_interval(backbone_values(rows), draws=draws, seed=seed)
        | {"units_in_level": len(rows)}
        for level, rows in sorted(levels.items())
    }


# ----------------------------------------------------- the permutation calibration


def permutation_calibration(
    cells: Sequence[Mapping[str, Any]], observed: float | None, *,
    estimator: str = "empirical", draws: int = D.PERMUTATION_DRAWS,
    seed: int = D.PERMUTATION_SEED,
) -> dict[str, Any]:
    """The endpoint's distribution when the forced-anchor label carries no information.

    Within a cell the draws are held exactly as generated and only their condition
    labels are permuted, as whole draws, so every completion's internal
    correlation across positions is preserved and the two groups keep their sizes.
    This calibrates the divergence estimator's finite-sample positive bias
    directly: under the permutation the two groups come from one distribution at
    every position, so the double difference's null is realised rather than
    argued. It calibrates the bootstrap; it does not replace it.
    """

    if observed is None:
        return {"draws": 0, "status": "no observed endpoint to calibrate"}
    prepared = []
    for cell in cells:
        selection = select_draws(
            cell["native"]["draws"], cell["transplant"]["draws"],
            completed_only=False, exclude_repeat_flagged=False, deduplicate=True,
        )
        sizes = {condition: len(selection["kept"][condition]) for condition in D.CONDITIONS}
        if min(sizes.values()) < MIN_DRAWS_PER_CONDITION:
            continue
        prepared.append((cell, selection, sizes))
    if not prepared:
        return {"draws": 0, "status": "no cell carries the minimum draws per condition"}
    generator = np.random.default_rng(int(seed))
    nulls: list[float] = []
    for _ in range(int(draws)):
        records = []
        for cell, selection, sizes in prepared:
            pooled = (
                [(D.CONDITION_NATIVE, index) for index in selection["kept"][D.CONDITION_NATIVE]]
                + [(D.CONDITION_TRANSPLANT, index)
                   for index in selection["kept"][D.CONDITION_TRANSPLANT]]
            )
            order = generator.permutation(len(pooled))
            assigned = {D.CONDITION_NATIVE: [], D.CONDITION_TRANSPLANT: []}
            for rank, slot in enumerate(order):
                condition = (
                    D.CONDITION_NATIVE if rank < sizes[D.CONDITION_NATIVE]
                    else D.CONDITION_TRANSPLANT
                )
                assigned[condition].append(pooled[slot])
            records.append(_permuted_cell_endpoint(cell, assigned, estimator=estimator))
        values = backbone_values([row for row in records if row["endpoint_nats"] is not None])
        if values:
            nulls.append(float(np.mean(list(values.values()))))
    if not nulls:
        return {"draws": 0, "status": "the permutation produced no backbone value"}
    array = np.asarray(nulls, dtype=np.float64)
    extreme = int(np.sum(np.abs(array) >= abs(float(observed))))
    return {
        "draws": int(array.size),
        "seed": int(seed),
        "estimator": estimator,
        "mean_nats": float(array.mean()),
        "interval_nats": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))],
        "observed_nats": float(observed),
        "two_sided_p": float((1 + extreme) / (array.size + 1)),
        "reads_as": (
            "the double difference when which draws of a cell are called native is "
            "permuted, with every completion, its length, its composition and the "
            "group sizes untouched; its mean is the estimator's residual bias"
        ),
    }


def _permuted_cell_endpoint(
    cell: Mapping[str, Any], assigned: Mapping[str, Sequence[tuple[str, int]]], *, estimator: str,
) -> dict[str, Any]:
    """One cell's endpoint under a permuted condition label.

    A permuted group mixes draws from both original conditions, so the realised
    distribution has to be assembled from both lists; the convenience path in
    :func:`cell_endpoint` cannot express that and is deliberately not reused here.
    """

    positions = [int(cell["partner"]), *[int(value) for value in cell["reference_positions"]]]
    source = {
        D.CONDITION_NATIVE: cell["native"],
        D.CONDITION_TRANSPLANT: cell["transplant"],
    }
    divergences: list[float | None] = []
    for slot, position in enumerate(positions):
        sides = []
        for group in D.CONDITIONS:
            members = assigned[group]
            if estimator == "empirical":
                realised = [
                    source[origin]["draws"][index]["realised"].get(str(position))
                    for origin, index in members
                ]
                sides.append(histogram([value for value in realised if value is not None]))
            elif estimator == "mixture":
                rows = [
                    source[origin]["distributions"][index, slot]
                    for origin, index in members
                    if source[origin]["draws"][index]["realised"].get(str(position)) is not None
                ]
                if not rows:
                    sides.append(None)
                    continue
                mass = np.asarray(rows, dtype=np.float64).mean(axis=0)
                sides.append(mass / mass.sum())
            else:
                raise ValueError(f"unknown estimator {estimator!r}")
        divergences.append(
            None if sides[0] is None or sides[1] is None
            else jensen_shannon(sides[0], sides[1])
        )
    references = [value for value in divergences[1:] if value is not None]
    endpoint = (
        None if divergences[0] is None or not references
        else float(divergences[0] - float(np.mean(references)))
    )
    return {
        "unit_id": cell["unit_id"], "accession": cell["accession"],
        "length_band": cell["length_band"], "endpoint_nats": endpoint,
    }


# ------------------------------------------------------- the panel-wide statement


def simultaneous_band(
    columns: Sequence[str], values: Mapping[str, Mapping[str, float]], *,
    draws: int = D.BOOTSTRAP_DRAWS, seed: int = D.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """A studentised-maximum band over the arms, on shared backbone draws.

    Every arm runs every cell of one cohort, so the backbone universe is
    identical across columns by construction and the table is complete. A column
    missing a backbone means the arms did not measure the same panel, which is
    refused here rather than patched by rejecting draws: the simpler estimator is
    correct precisely because the design removed the missingness the general one
    exists to handle.

    One set of backbone draws is shared by every column, so the dependence that
    makes the arms correlated is preserved; the critical value is the 95th
    percentile of the maximum over columns of the absolute centred mean divided
    by that column's fixed standard error. Nothing is fitted anywhere in this
    experiment, so these bands carry no out-of-fold caveat.
    """

    names = [str(name) for name in columns]
    if len(names) < 2:
        raise ValueError("a simultaneous band needs at least two columns")
    universe = sorted(set(values[names[0]]))
    for name in names:
        if sorted(set(values[name])) != universe:
            missing = sorted(set(universe) ^ set(values[name]))
            raise ValueError(
                f"column {name!r} does not carry the same backbone universe "
                f"({len(missing)} differing, e.g. {missing[:5]}); the arms did not "
                "measure one panel and a simultaneous band over them is refused"
            )
    if len(universe) < 2:
        raise ValueError("a simultaneous band needs at least two backbones")
    table = np.asarray(
        [[float(values[name][backbone]) for name in names] for backbone in universe],
        dtype=np.float64,
    )
    point = table.mean(axis=0)
    standard_error = table.std(axis=0, ddof=1) / np.sqrt(table.shape[0])
    generator = np.random.default_rng(int(seed))
    index = generator.integers(table.shape[0], size=(int(draws), table.shape[0]))
    boot = table[index].mean(axis=1)
    varying = standard_error > 0
    maxima = (
        np.max(np.abs((boot[:, varying] - point[varying]) / standard_error[varying]), axis=1)
        if varying.any() else np.zeros(int(draws))
    )
    critical = float(np.quantile(maxima, 0.95))
    return {
        "method": (
            "shared backbone bootstrap over a complete table; maximum absolute centred "
            "mean over fixed per-column standard errors"
        ),
        "confidence": 0.95,
        "critical_value": critical,
        "draws": int(draws),
        "seed": int(seed),
        "backbones": len(universe),
        "conditional_on_fitted_predictions": False,
        "columns": [
            {
                "column": name,
                "point_nats": float(point[position]),
                "standard_error_nats": float(standard_error[position]),
                "pointwise_interval_nats": [
                    float(value) for value in np.percentile(boot[:, position], [2.5, 97.5])
                ],
                "simultaneous_interval_nats": [
                    float(point[position] - critical * standard_error[position]),
                    float(point[position] + critical * standard_error[position]),
                ],
                "simultaneously_excludes_zero": bool(
                    point[position] - critical * standard_error[position] > 0.0
                    or point[position] + critical * standard_error[position] < 0.0
                ),
            }
            for position, name in enumerate(names)
        ],
    }


# ------------------------------------------------------------------- the verdict


def gate_verdict(
    estimate: Mapping[str, Any], calibration: Mapping[str, Any], *,
    claimable: float = D.CLAIMABLE["sequence_double_difference"],
) -> dict[str, Any]:
    """Whether the gate resolved, against the pre-registered claimable effect.

    Four outcomes and no fifth, because the three that are usually collapsed say
    different things:

    ``resolved_positive``
        the interval excludes zero, the point estimate reaches the pre-registered
        claimable effect, and the permutation calibration rejects at alpha. Only
        this outcome opens the structure channel.
    ``resolved_below_claimable``
        the interval excludes zero but the point estimate is smaller than the
        design pre-registered as claimable. An effect is present and the claim is
        still not made: the threshold was fixed before the run and is not revised
        to meet the result.
    ``bounded_null``
        the interval contains zero and lies entirely inside the claimable effect.
        Something is being asserted here -- that an effect of claimable size is
        not there -- which is why it is not the same as unresolved.
    ``unresolved``
        the interval neither excludes zero nor excludes the claimable effect. This
        is not a null, and it is the one outcome that more data would change.
    """

    point, interval = estimate.get("point"), estimate.get("interval")
    if point is None or interval is None:
        return {"verdict": "unresolved", "reason": "the endpoint has no interval"}
    low, high = float(interval[0]), float(interval[1])
    p_value = calibration.get("two_sided_p")
    excludes_zero = low > 0.0 or high < 0.0
    threshold = float(claimable)
    if excludes_zero and abs(float(point)) >= threshold and (
        p_value is not None and float(p_value) < D.ALPHA
    ):
        verdict = "resolved_positive"
        reason = (
            f"the interval [{low:.4f}, {high:.4f}] excludes zero, the point estimate "
            f"{float(point):.4f} reaches the pre-registered {threshold} nats, and the "
            f"permutation p-value is {p_value:.4g}"
        )
    elif excludes_zero:
        verdict = "resolved_below_claimable"
        reason = (
            f"the interval [{low:.4f}, {high:.4f}] excludes zero but the point estimate "
            f"{float(point):.4f} is below the pre-registered claimable effect of "
            f"{threshold} nats"
            + ("" if p_value is None else f"; permutation p-value {p_value:.4g}")
        )
    elif high < threshold and low > -threshold:
        verdict = "bounded_null"
        reason = (
            f"the interval [{low:.4f}, {high:.4f}] contains zero and lies inside the "
            f"pre-registered claimable effect of {threshold} nats; an effect of that size "
            "is excluded and a smaller one is not distinguishable by this design"
        )
    else:
        verdict = "unresolved"
        reason = (
            f"the interval [{low:.4f}, {high:.4f}] neither excludes zero nor excludes the "
            f"pre-registered claimable effect of {threshold} nats; this is not a null"
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "pre_registered_claimable_nats": float(claimable),
        "structure_channel_gated": verdict == "resolved_positive",
        "structure_channel": D.STRUCTURE_CHANNEL_CONTRACT,
        "non_identifiable": dict(D.NON_IDENTIFIABLE),
        "interpretation": D.INTERPRETATION,
    }
