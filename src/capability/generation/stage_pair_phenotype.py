"""Does Stage 2's better mutation-effect prediction show up in its generated proteins?

Stage 2 predicts mutation effects better than Stage 1 -- it is one of only two
panel checkpoints whose single-substitution stability increment resolves positive.
E18 asks whether that improvement in *prediction* corresponds to better products.

Two different questions, both legitimate
========================================

Stage 2's products are 47 residues shorter than Stage 1's, slightly more
hydrophilic and marginally less compositionally diverse. Those are the covariates
that drive a structure predictor's confidence, so a raw comparison of folded
confidence between the stages is partly a comparison of lengths. The response is
**not** to call the raw contrast wrong. There are two estimands and this module
reports both, labelled:

``generation_distribution``
    the difference between what each stage actually emits, over its own product
    distribution. This is what a user of the checkpoint receives, and no
    adjustment belongs in it.

``matched`` / ``adjusted``
    the difference at equal length, or at equal length and hydrophobicity, or
    holding the measured covariates fixed. This is how good a product is *given*
    its size and composition.

Neither is the corrected version of the other. A stage that emits shorter
products is a real property of the stage; so is a stage whose products fold
better at equal size. :data:`ESTIMANDS` carries that distinction into the
artefact so a reader cannot collapse the two.

Three kinds of readout, kept apart
==================================

:data:`READOUT_CLASSES` separates them, because they license different language.

``structural_confidence``
    pLDDT, pTM, PAE. The predictor's uncertainty about its own coordinates. No
    free-energy units, no folding claim, not a measurement.
``predicted_stability``
    a gated predictor's output in kcal/mol, usable only inside the band its own
    licence declares, and still a prediction about a sequence.
``measured``
    an experimental observation. **Nothing in E18 is of this class**, and the
    artefact says so rather than leaving the reader to assume otherwise.

What is deliberately not constructed
====================================

Stage 2 carries a superfamily instruction interface that Stage 1 does not have.
No artificial Stage 1 conditional counterpart is built: the conditional
comparison is impossible by construction, and only interfaces both checkpoints
support enter a direct stage comparison.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS, mean_interval, paired_group_bootstrap
from .stage_pair import STAGES

SCHEMA_VERSION = "d1_stage_pair_phenotype_v1"

#: The covariates a structural comparison between the two stages is confounded
#: by. Length first because it dominates: it is the only one whose standardised
#: mean difference between the stages exceeds a quarter of a pooled standard
#: deviation.
COVARIATES: tuple[str, ...] = (
    "length",
    "mean_kyte_doolittle_hydropathy",
    "composition_entropy_nats",
    "distinct_residues",
    "longest_single_residue_run",
)

#: Caps for the two declared matched designs, as ``(covariate, cap)`` pairs. A
#: pair outside any cap is not matched; the realised counts are reported and the
#: caps are never widened after a readout has been seen.
MATCH_DESIGNS: dict[str, tuple[tuple[str, float], ...]] = {
    "length": (("length", 5.0),),
    "length_hydropathy": (
        ("length", 5.0),
        ("mean_kyte_doolittle_hydropathy", 0.25),
    ),
}

ESTIMANDS: dict[str, str] = {
    "generation_distribution": (
        "the difference between what each stage emits over its own product "
        "distribution, unadjusted. This is the quantity a user of the checkpoint "
        "experiences, and length and composition differences are part of it rather "
        "than nuisance"
    ),
    "matched_length": (
        "the difference at equal product length, within stream, by optimal one-to-one "
        "assignment under a declared length cap. This estimates how good a product is "
        "given its size, and it does not remove the residual composition shift"
    ),
    "matched_length_hydropathy": (
        "the difference at equal product length and equal mean hydropathy. This "
        "estimates how good a product is given its size and hydrophobicity"
    ),
    "adjusted": (
        "the stage coefficient of a within-stream linear fit of the readout on the "
        "stage indicator and the measured covariates. This estimates the difference "
        "holding those covariates fixed, and is interpretable only where the two "
        "stages share covariate support, which the overlap diagnostic reports"
    ),
}

READOUT_CLASSES: dict[str, str] = {
    "structural_confidence": (
        "the structure predictor's uncertainty about its own coordinates: pLDDT, pTM "
        "and predicted aligned error. Dimensionless or in angstrom, never in energy "
        "units, and not a folding, function or stability claim"
    ),
    "sequence_property": (
        "a property computable from the sequence alone, with no model and no "
        "structure: length, composition, complexity"
    ),
    "predicted_stability": (
        "a gated stability predictor's output in kcal/mol, licensed only inside the "
        "band its own qualification declares. Still a prediction about a sequence and "
        "never a measurement"
    ),
    "measured": (
        "an experimental observation of a generated protein. NOTHING in this "
        "experiment is of this class: no generated sequence here has been expressed, "
        "folded in vitro or assayed"
    ),
}

#: Which class each structural readout belongs to.
CONFIDENCE_READOUTS: tuple[str, ...] = (
    "mean_ca_plddt",
    "fraction_ca_plddt_ge70",
    "fraction_ca_plddt_ge90",
    "ptm",
    "mean_pae_angstrom",
    "mean_pae_on_contacts_angstrom",
    "contact_density",
)

CEILING: dict[str, str] = {
    "both_estimands_are_real": (
        "the unadjusted and the matched contrasts answer different questions and "
        "neither is the corrected version of the other"
    ),
    "confidence_is_not_stability": (
        "pLDDT and pTM are predicted confidence, carry no free-energy units and are "
        "not stability"
    ),
    "nothing_here_is_measured": (
        "no generated sequence in this experiment has been expressed or assayed; every "
        "readout is a prediction or a sequence property"
    ),
    "no_artificial_conditional_counterpart": (
        "Stage 2 has a superfamily instruction interface Stage 1 lacks entirely, so no "
        "conditioned Stage 1 counterpart exists or is constructed; the direct stage "
        "comparison is restricted to the bare-prompt interface both support"
    ),
    "streams_are_not_lineages": (
        "campaign seed streams resample from fixed checkpoints and are not independent "
        "training lineages; the stream is the resampling unit for every interval here"
    ),
    "adjustment_needs_overlap": (
        "a covariate-adjusted contrast extrapolates wherever the two stages do not "
        "share covariate support, so the overlap diagnostic is reported beside it and "
        "is part of the result"
    ),
}


def covariate_vector(record: Mapping[str, Any]) -> np.ndarray:
    """The declared covariates of one frozen-set record."""

    properties = record.get("properties") or {}
    values = []
    for name in COVARIATES:
        value = record.get(name) if name == "length" else properties.get(name)
        if value is None:
            raise ValueError(f"{record.get('id')!r} carries no covariate {name!r}")
        values.append(float(value))
    return np.asarray(values, dtype=float)


def overlap_diagnostics(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Per covariate, the standardised mean difference and the shared support.

    Reported because it decides whether the adjusted contrast interpolates or
    extrapolates. The shared-support share is the fraction of each stage lying
    inside the other's central 90% range: a covariate on which one stage mostly
    sits outside the other's range cannot be adjusted for, only matched on within
    the overlap or declared unmatchable.
    """

    a = np.asarray([covariate_vector(record) for record in left], dtype=float)
    b = np.asarray([covariate_vector(record) for record in right], dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("an overlap diagnostic needs both stages")
    per_covariate: dict[str, Any] = {}
    for index, name in enumerate(COVARIATES):
        x, y = a[:, index], b[:, index]
        pooled = float(np.sqrt((x.var() + y.var()) / 2.0))
        low = max(float(np.quantile(x, 0.05)), float(np.quantile(y, 0.05)))
        high = min(float(np.quantile(x, 0.95)), float(np.quantile(y, 0.95)))
        per_covariate[name] = {
            "stage_1_mean": float(x.mean()),
            "stage_1_sd": float(x.std()),
            "stage_2_mean": float(y.mean()),
            "stage_2_sd": float(y.std()),
            "standardised_mean_difference": (
                float((y.mean() - x.mean()) / pooled) if pooled > 0 else None
            ),
            "common_support_range": [low, high],
            "stage_1_share_in_common_support": float(((x >= low) & (x <= high)).mean()),
            "stage_2_share_in_common_support": float(((y >= low) & (y <= high)).mean()),
        }
    dominant = max(
        per_covariate,
        key=lambda name: abs(per_covariate[name]["standardised_mean_difference"] or 0.0),
    )
    return {
        "per_covariate": per_covariate,
        "dominant_confound": dominant,
        "n_stage_1": len(left),
        "n_stage_2": len(right),
        "rule": (
            "the standardised mean difference is the stage shift in pooled standard "
            "deviations; the common-support share is the fraction of each stage inside "
            "the other's central 90% range"
        ),
    }


def matched_pairs(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    caps: Sequence[tuple[str, float]],
) -> list[tuple[int, int, dict[str, float]]]:
    """Optimal one-to-one matching under hard per-covariate caps.

    An optimal assignment rather than a greedy sweep, so the matched set does not
    depend on the order the records happen to arrive in, and deterministic, so it
    needs no seed. A pair violating any cap is never matched; the realised count
    is what it is and the caps are not widened afterwards.
    """

    from scipy.optimize import linear_sum_assignment

    if not caps:
        raise ValueError("a matched design needs at least one capped covariate")
    indices = {name: COVARIATES.index(name) for name, _ in caps}
    a = np.asarray([covariate_vector(record) for record in left], dtype=float)
    b = np.asarray([covariate_vector(record) for record in right], dtype=float)
    if a.size == 0 or b.size == 0:
        return []
    feasible = np.ones((len(left), len(right)), dtype=bool)
    cost = np.zeros((len(left), len(right)), dtype=float)
    distances: dict[str, np.ndarray] = {}
    for name, cap in caps:
        if cap <= 0:
            raise ValueError(f"the cap on {name!r} must be positive")
        column = indices[name]
        gap = np.abs(a[:, column][:, None] - b[:, column][None, :])
        distances[name] = gap
        feasible &= gap <= cap
        cost += gap / cap
    blocked = np.where(feasible, cost, np.float64(1e9))
    rows, columns = linear_sum_assignment(blocked)
    pairs: list[tuple[int, int, dict[str, float]]] = []
    for row, column in zip(rows, columns):
        if not feasible[row, column]:
            continue
        pairs.append(
            (
                int(row),
                int(column),
                {name: float(gap[row, column]) for name, gap in distances.items()},
            )
        )
    return pairs


def _stream_interval(points: Mapping[str, float], *, label: str) -> dict[str, Any]:
    """Equal weight per stream, Student-t over the stream values."""

    values = [float(points[key]) for key in sorted(points)]
    record: dict[str, Any] = {
        "per_stream": {key: float(points[key]) for key in sorted(points)},
        "n_streams": len(values),
        "unit": "the campaign stream",
        "estimand": label,
    }
    if len(values) < 2:
        record.update(
            {
                "mean": values[0] if values else None,
                "interval": None,
                "single_stream_reason": (
                    "one stream carries no across-stream interval; seed streams are the "
                    "declared unit and a single one cannot bound itself"
                ),
            }
        )
        return record
    record.update(mean_interval(values))
    record["direction_replicated"] = bool(
        all(value > 0 for value in values) or all(value < 0 for value in values)
    )
    return record


def unmatched_contrast(
    records: Sequence[Mapping[str, Any]], *, field: str
) -> dict[str, Any]:
    """Stage 2 minus Stage 1 over each stage's own product distribution."""

    per_stream: dict[str, dict[str, list[float]]] = {}
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        per_stream.setdefault(str(record["stream"]), {}).setdefault(
            str(record["stage"]), []
        ).append(float(value))
    points: dict[str, float] = {}
    support: dict[str, Any] = {}
    for stream, block in sorted(per_stream.items()):
        if set(block) != set(STAGES):
            support[stream] = {"reason": f"stream carries only {sorted(block)}"}
            continue
        means = {stage: float(np.mean(block[stage])) for stage in sorted(block)}
        points[stream] = means["stage_2"] - means["stage_1"]
        support[stream] = {
            "means": means,
            "n": {stage: len(block[stage]) for stage in sorted(block)},
        }
    return {
        "field": field,
        "support": support,
        **_stream_interval(points, label="generation_distribution"),
    }


def matched_contrast(
    by_stream: Mapping[str, tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]],
    *,
    field: str,
    design: str,
    resamples: int = 2000,
    seed: int = 20261008,
) -> dict[str, Any]:
    """Stage 2 minus Stage 1 within matched pairs, per stream and across streams.

    The across-stream Student-t interval is primary, because the stream is this
    campaign's declared unit. A paired bootstrap over the matched pairs of all
    streams is reported beside it as a precision statement at the pair as the
    unit; it is not a second confirmation and the two must not be read as
    agreeing confirmations of one claim.
    """

    caps = MATCH_DESIGNS[design]
    points: dict[str, float] = {}
    support: dict[str, Any] = {}
    all_left: list[float] = []
    all_right: list[float] = []
    groups: list[str] = []
    for stream, (left, right) in sorted(by_stream.items()):
        usable_left = [record for record in left if record.get(field) is not None]
        usable_right = [record for record in right if record.get(field) is not None]
        pairs = matched_pairs(usable_left, usable_right, caps=caps)
        if not pairs:
            support[stream] = {"n_pairs": 0, "reason": "no pair satisfied the declared caps"}
            continue
        differences = [
            float(usable_right[column][field]) - float(usable_left[row][field])
            for row, column, _ in pairs
        ]
        points[stream] = float(np.mean(differences))
        gaps = {
            name: float(np.mean([gap[name] for _, _, gap in pairs])) for name, _ in caps
        }
        support[stream] = {
            "n_pairs": len(pairs),
            "n_candidates": {"stage_1": len(usable_left), "stage_2": len(usable_right)},
            "mean_absolute_gap": gaps,
        }
        for row, column, _ in pairs:
            all_left.append(float(usable_left[row][field]))
            all_right.append(float(usable_right[column][field]))
            groups.append(f"{stream}|{len(groups)}")
    capped = {name for name, _ in caps}
    record = {
        "field": field,
        "design": design,
        "caps": {name: cap for name, cap in caps},
        # A readout that the design matched on is not a result: it shows whether the
        # matching achieved the balance it claims. Labelled so it is read that way.
        "role": "balance_check" if field in capped else "contrast",
        "role_note": (
            "this readout is one of the matched covariates, so the value measures "
            "residual imbalance after matching, not a stage effect"
            if field in capped
            else "this readout is not matched on, so the value is the stage contrast "
            "at matched covariates"
        ),
        "support": support,
        "n_pairs_total": sum(
            block.get("n_pairs", 0) for block in support.values()
        ),
        **_stream_interval(points, label=f"matched_{design}"),
    }
    if len(all_left) >= MINIMUM_BOOTSTRAP_UNITS:
        truth = np.zeros(len(all_left))
        bootstrap = paired_group_bootstrap(
            truth,
            np.asarray(all_right),
            np.asarray(all_left),
            np.asarray(groups),
            lambda _t, predicted: float(np.mean(predicted)),
            seed=seed,
            n_bootstrap=resamples,
        )
        record["pair_bootstrap"] = {
            "difference": bootstrap["difference"],
            "ci95": bootstrap["difference_ci95"],
            "n_pairs": len(all_left),
            "unit": "the matched pair",
            "note": (
                "a precision statement at the pair as the unit, reported beside the "
                "stream-unit interval and not a second confirmation of it"
            ),
        }
    return record


def adjusted_contrast(
    records: Sequence[Mapping[str, Any]], *, field: str
) -> dict[str, Any]:
    """The stage coefficient of a within-stream linear fit on the declared covariates.

    Covariates are standardised within the stream so the coefficients are
    comparable, and the stage indicator is the quantity of interest. Fitted per
    stream and summarised across streams, so the unit stays the stream.
    """

    if field in COVARIATES:
        # Regressing a covariate on itself leaves the stage indicator no variance to
        # explain, so the coefficient is zero by construction. Refused rather than
        # reported, because a published "adjusted difference of zero" would read as a
        # finding that the stage shift vanishes under adjustment.
        return {
            "field": field,
            "estimand": "adjusted",
            "mean": None,
            "interval": None,
            "n_streams": 0,
            "unit": "the campaign stream",
            "refused": True,
            "refusal_reason": (
                f"{field!r} is itself one of the adjustment covariates, so the stage "
                "coefficient is zero by construction and carries no information"
            ),
        }

    per_stream: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if record.get(field) is None:
            continue
        per_stream.setdefault(str(record["stream"]), []).append(record)
    points: dict[str, float] = {}
    support: dict[str, Any] = {}
    for stream, block in sorted(per_stream.items()):
        stages = {str(record["stage"]) for record in block}
        if stages != set(STAGES):
            support[stream] = {"reason": f"stream carries only {sorted(stages)}"}
            continue
        covariates = np.asarray([covariate_vector(record) for record in block], dtype=float)
        spread = covariates.std(axis=0)
        spread[spread == 0] = 1.0
        standardised = (covariates - covariates.mean(axis=0)) / spread
        indicator = np.asarray(
            [1.0 if str(record["stage"]) == "stage_2" else 0.0 for record in block]
        )
        target = np.asarray([float(record[field]) for record in block])
        design = np.column_stack([np.ones_like(indicator), indicator, standardised])
        solution, _residuals, rank, _singular = np.linalg.lstsq(design, target, rcond=None)
        if rank < design.shape[1]:
            support[stream] = {
                "reason": (
                    f"the design matrix is rank {rank} of {design.shape[1]}; a "
                    "collinear covariate makes the stage coefficient unidentified"
                )
            }
            continue
        points[stream] = float(solution[1])
        support[stream] = {
            "n": len(block),
            "stage_coefficient": float(solution[1]),
            "covariate_coefficients": {
                name: float(value) for name, value in zip(COVARIATES, solution[2:])
            },
            "covariates_standardised_within_stream": True,
        }
    return {
        "field": field,
        "support": support,
        "covariates": list(COVARIATES),
        "caveat": CEILING["adjustment_needs_overlap"],
        **_stream_interval(points, label="adjusted"),
    }


def band_admission(
    records: Sequence[Mapping[str, Any]], *, low: int, high: int, label: str
) -> dict[str, Any]:
    """How unequally a licensed residue band admits the two stages.

    A gated predictor that is licensed only inside a residue band does not sample
    the two stages equally when their length distributions differ. If one stage
    contributes far more products to the band, a stage difference measured inside
    it is partly a difference in how often each stage is admitted at all, and that
    has to be reported next to the estimate rather than inferred from it.
    """

    per_stage: dict[str, Any] = {}
    for stage in sorted(STAGES):
        block = [record for record in records if str(record["stage"]) == stage]
        inside = [record for record in block if low <= int(record["length"]) <= high]
        per_stage[stage] = {
            "n_total": len(block),
            "n_in_band": len(inside),
            "share_in_band": len(inside) / len(block) if block else None,
        }
    shares = [
        per_stage[stage]["share_in_band"]
        for stage in sorted(STAGES)
        if per_stage[stage]["share_in_band"]
    ]
    ratio = max(shares) / min(shares) if len(shares) == 2 and min(shares) > 0 else None
    return {
        "band": label,
        "band_residues": [int(low), int(high)],
        "per_stage": per_stage,
        "admission_ratio": ratio,
        "reading": (
            "the ratio of the two stages' shares inside the band. A ratio far from one "
            "means the band samples the stages unequally, so a stage difference "
            "estimated inside it carries a band-admission component"
        ),
    }


def termination_profile(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Censoring inside the frozen set, per stage, on the stop accounting alone."""

    per_stage: dict[str, Any] = {}
    for stage in sorted(STAGES):
        block = [record for record in records if str(record["stage"]) == stage]
        flags = [record.get("censored") for record in block]
        censored = [record for record, flag in zip(block, flags) if flag is True]
        native = [record for record, flag in zip(block, flags) if flag is False]
        per_stage[stage] = {
            "n_total": len(block),
            "n_censored": len(censored),
            "censored_share": len(censored) / len(block) if block else None,
            "n_natively_terminated": len(native),
            "n_unknown": sum(1 for flag in flags if flag is None),
            "median_length_native": (
                float(np.median([record["length"] for record in native])) if native else None
            ),
            "median_length_censored": (
                float(np.median([record["length"] for record in censored]))
                if censored
                else None
            ),
        }
    return {
        "per_stage": per_stage,
        "rule": (
            "censoring is read from the generated token count against the effective "
            "budget, else from the stop reason; composition is never consulted"
        ),
        "scope": (
            "this is the frozen set, which was selected inside a 40-400 residue band; "
            "the censoring rate of the full ledgers is higher and is reported by the "
            "expanded-controls census, not here"
        ),
    }


def split_stages(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """``stream -> (stage_1 records, stage_2 records)``."""

    streams: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        streams.setdefault(str(record["stream"]), {}).setdefault(
            str(record["stage"]), []
        ).append(dict(record))
    return {
        stream: (block.get("stage_1", []), block.get("stage_2", []))
        for stream, block in sorted(streams.items())
    }
