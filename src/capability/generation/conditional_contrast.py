"""Requested minus mismatched: the aggregated conditional-generation endpoint.

The question
============

Can a model generate proteins that meet a stated functional requirement? A high
absolute recognition rate does not answer it. A conditioned decoder that emits
the same family whatever label it is given would score a near-perfect *rate* and
carry no conditional capability at all, and the leak measurement that motivated
this campaign -- ZymCTRL's EC tag priced at 1.73 nats -- cannot tell the two
apart. The quantity that can is a **difference**: the rate at which generations
produced under the request for class *c* are assigned to *c*, minus the rate at
which generations produced under a request for a **different** class are
assigned to *c*.

What this module adds
=====================

The per-cell rates have existed since 2026-09-05 and the per-arm aggregate under
the pre-registered convention since 2026-08-26
(``results/R6/conditioned_generation_20260826/conditioned_generation.json``).
What never existed is the class-resampled paired difference on the support the
manuscript's conditional panel quotes --
``conditional/panel/requested_minus_mismatched`` in
``results/shared/manuscript_evidence_20261004/evidence-gaps.json``, which reads
all sixteen drawn classes at the attempt-level rate rather than the admitted
classes at the near-duplicate-collapsed rate. Both conventions are computed
here, side by side, because they differ in **support** (16 classes versus the
14/15 the instrument gate admitted) and in **estimator** (one attempt per unit
versus one near-duplicate group per unit) and neither is a correction of the
other.

What is held fixed
==================

* The resampling unit is the **class**, and the interval comes from
  :func:`src.capability.generation.conditioned_generation.class_clustered_mean`,
  which is this package's one declared class-clustered resampler over
  :func:`src.capability.core.statistics.paired_group_bootstrap`. No bootstrap is
  written here.
* The mismatched pairing is the frozen fixed-point-free permutation of the class
  queue, so every class is requested exactly once and donates exactly once.
  :func:`verify_pairing_balance` refuses anything else: an unbalanced pairing
  would let one easy donor set the whole negative side.
* ZymCTRL and ProLLaMA are asked for different kinds of class through different
  oracles. Their differences are reported side by side and **never** pooled or
  differenced, which is the frozen ceiling of the campaign that produced them.
* Degeneracy is reported with every rate, not instead of it. A cell that emits
  one sequence two hundred times can score an arbitrarily high yield, so the
  distinct-sequence count, the near-duplicate group count and the count of
  distinct groups that carry the target family travel with the rate.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core.statistics import mean_interval
from .conditioned_generation import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    MINIMUM_CLASSES,
    class_clustered_mean,
    compound_verdict,
    grouped_rate,
)

SCHEMA_VERSION = "d1_conditional_contrast_v1"

#: The two conditions this endpoint differences. The unconditioned floor is a
#: different reference with its own clause in the campaign's compound verdict and
#: is read from the floor arms, not from a condition of a conditioned arm.
CONDITIONS: tuple[str, str] = ("requested", "mismatched")

#: The two declared estimators of one class's rate.
#:
#: ``attempt``  one unit per attempt. This is the estimator the manuscript's
#:              per-cell ``target_profile_rate`` numbers use.
#: ``grouped``  one unit per near-duplicate group, scored as the within-group hit
#:              fraction, so a burst of near-identical samples contributes once
#:              however many times it was drawn. This is the estimator the
#:              2026-08-26 campaign report used.
ESTIMATORS: tuple[str, str] = ("attempt", "grouped")

#: The two declared supports. ``all`` reads every drawn class; ``admitted`` reads
#: only the classes whose per-class instrument anchor cleared the pre-declared
#: real-exemplar floor and random-protein ceiling. A class the instrument cannot
#: price is an unmeasurable class, not a failing one.
SUPPORTS: tuple[str, str] = ("all", "admitted")

PRIMARY_ESTIMATOR = "attempt"
PRIMARY_SUPPORT = "all"

CEILING: dict[str, str] = {
    "a_difference_not_a_rate": (
        "the endpoint is requested minus mismatched. A high requested rate alone is "
        "equally consistent with a label that selects the requested class and with a "
        "decoder that emits one family whatever it is asked for"
    ),
    "cross_arm_rates_are_descriptive": (
        "ZymCTRL and ProLLaMA are asked for different kinds of class through "
        "different oracles; their differences are reported side by side and never "
        "pooled or differenced"
    ),
    "degeneracy_travels_with_the_rate": (
        "a cell that emits one sequence repeatedly can score a high yield, so the "
        "distinct-sequence count, the near-duplicate group count and the distinct "
        "target-carrying group count are reported for every cell"
    ),
    "the_cohort_is_not_independent_of_the_arm": (
        "the classes are drawn from the arm's own labelled corpus, so a positive is "
        "about these classes on this cohort and is not a statement about the label "
        "space as a whole"
    ),
    "the_oracle_bounds_the_reading": (
        "a Pfam assignment is a sequence-level homology statement at the release's "
        "gathering thresholds. It is not folding, not function and not novelty"
    ),
    "streams_are_not_training_lineages": (
        "campaign seed streams replicate sampling from fixed checkpoints; they are "
        "not independent training lineages and carry no causal reading"
    ),
}


# --------------------------------------------------------------- the pairing


def pairing_from_queue(queue: Mapping[str, Any], arm: str) -> dict[str, str]:
    """``class -> donor class`` for one arm of the frozen class queue.

    The donor is the class whose label the mismatched cell is *prompted* with
    while the oracle scores against the cell's own class, which is the convention
    the retained ledger records in ``native_prompt_class``.
    """

    arms = queue.get("arms")
    if not isinstance(arms, Mapping) or arm not in arms:
        raise KeyError(f"the frozen queue carries no arm {arm!r}; it carries {sorted(arms or ())}")
    entries = arms[arm]["classes"]
    return {str(entry["key"]): str(entry["mismatched_key"]) for entry in entries}


def verify_pairing_balance(pairing: Mapping[str, str]) -> dict[str, Any]:
    """Refuse a mismatched pairing that is not a balanced fixed-point-free permutation.

    Three conditions, each with its own failure mode. A fixed point would make a
    class its own negative. A donor used twice while another is never used would
    let one easy or one hard donor carry the negative side. A donor outside the
    class set would price the negative against a class the requested side never
    covers.
    """

    classes = sorted(pairing)
    donors = [pairing[key] for key in classes]
    fixed = sorted(key for key in classes if pairing[key] == key)
    outside = sorted(set(donors) - set(classes))
    counts = Counter(donors)
    unbalanced = sorted(key for key in classes if counts.get(key, 0) != 1)
    reasons: list[str] = []
    if len(classes) < 2:
        reasons.append("a mismatched pairing needs at least two classes")
    if fixed:
        reasons.append(f"classes paired with themselves: {fixed}")
    if outside:
        reasons.append(f"donors outside the class set: {outside}")
    if unbalanced:
        reasons.append(
            f"classes not used exactly once as a donor: {unbalanced} "
            f"(counts {{{', '.join(f'{k}: {counts.get(k, 0)}' for k in unbalanced)}}})"
        )
    record = {
        "n_classes": len(classes),
        "classes": classes,
        "donors": donors,
        "fixed_points": fixed,
        "donors_outside_class_set": outside,
        "donor_use_counts": dict(sorted(counts.items())),
        "balanced": not reasons,
        "failure_reasons": reasons,
        "rule": (
            "a fixed-point-free permutation of the drawn classes: every class is "
            "requested exactly once and donates to exactly one other class"
        ),
    }
    if reasons:
        raise ValueError(
            "the mismatched pairing is not a balanced fixed-point-free permutation: "
            + "; ".join(reasons)
        )
    return record


# ------------------------------------------------------------------- the cells


def _hit_vector(rows: Sequence[Mapping[str, Any]], field: str) -> list[bool]:
    values: list[bool] = []
    for row in rows:
        value = row.get(field)
        if value is None:
            raise ValueError(
                f"attempt {row.get('id')!r} carries no {field}; an attempt with no "
                "recognition outcome must not enter a rate"
            )
        values.append(bool(value))
    return values


def cell_rates(
    rows: Sequence[Mapping[str, Any]], *, hit_field: str = "target_profile_hit"
) -> dict[str, Any]:
    """Both declared estimators of one cell's rate, with its degeneracy census.

    The denominator is every attempt of the cell. An attempt that decodes to no
    canonical residue is a genuine failure of the interface and is counted as a
    non-hit; it is never dropped, because a denominator selected on the outcome is
    not the denominator the campaign declared.
    """

    if not rows:
        raise ValueError("a rate needs at least one attempt")
    hits = _hit_vector(rows, hit_field)
    groups = [row.get("near_duplicate_group") or row["sequence_sha256"] for row in rows]
    digests = [row["sequence_sha256"] for row in rows]
    hit_digests = {digest for digest, hit in zip(digests, hits) if hit}
    hit_groups = {group for group, hit in zip(groups, hits) if hit}
    counts = Counter(digests)
    return {
        "n_attempts": len(rows),
        "n_hits": int(sum(hits)),
        "attempt_rate": float(np.mean([1.0 if hit else 0.0 for hit in hits])),
        "grouped_rate": grouped_rate(hits, np.asarray(groups)),
        "n_distinct_sequences": len(counts),
        "n_near_duplicate_groups": len(set(groups)),
        "n_distinct_target_sequences": len(hit_digests),
        "n_distinct_target_groups": len(hit_groups),
        "duplication_rate": 1.0 - len(counts) / len(rows),
        "largest_exact_duplicate_share": max(counts.values()) / len(rows),
        "n_empty_or_noncanonical": sum(1 for row in rows if not row.get("sequence")),
        "near_duplicate_groups_declared": sum(
            1 for row in rows if row.get("near_duplicate_group")
        ),
    }


def group_cells(
    rows: Iterable[Mapping[str, Any]], *, arm: str
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """One arm's attempts, bucketed by ``(class_key, condition)``."""

    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("arm") != arm or row.get("condition") not in CONDITIONS:
            continue
        class_key = row.get("class_key")
        if not class_key:
            raise ValueError(
                f"attempt {row.get('id')!r} is a conditioned attempt with no class_key"
            )
        cells[(str(class_key), str(row["condition"]))].append(dict(row))
    if not cells:
        raise ValueError(f"no requested or mismatched attempt of {arm!r} was supplied")
    return dict(cells)


def _complete_cells(
    cells: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]
) -> list[str]:
    classes = sorted({key for key, _ in cells})
    complete = [cls for cls in classes if all((cls, cond) in cells for cond in CONDITIONS)]
    incomplete = sorted(set(classes) - set(complete))
    if incomplete:
        raise ValueError(
            f"these classes carry only one of the two conditions: {incomplete}. The "
            "endpoint is a within-class paired difference and a class measured under "
            "one condition only cannot enter it"
        )
    return complete


# ------------------------------------------------------------------ the endpoint


def arm_contrast(
    rows: Iterable[Mapping[str, Any]],
    *,
    arm: str,
    pairing: Mapping[str, str],
    referents: Mapping[str, Mapping[str, Any]] | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    hit_field: str = "target_profile_hit",
) -> dict[str, Any]:
    """One arm's requested-minus-mismatched endpoint under every declared convention.

    ``referents`` supplies the ``admitted`` support: a class whose per-class
    instrument anchor failed is reported, and excluded from that support, but is
    never silently dropped from the ``all`` support.
    """

    balance = verify_pairing_balance(dict(pairing))
    cells = group_cells(rows, arm=arm)
    classes = _complete_cells(cells)
    unpaired = sorted(set(classes) - set(pairing))
    if unpaired:
        raise ValueError(
            f"{arm}: these measured classes are absent from the frozen pairing: "
            f"{unpaired}. A mismatched rate whose donor is not the frozen one is a "
            "different measurement"
        )
    per_class: dict[str, dict[str, Any]] = {}
    for cls in classes:
        block: dict[str, Any] = {
            "donor_class": pairing[cls],
            "conditions": {
                cond: cell_rates(cells[(cls, cond)], hit_field=hit_field)
                for cond in CONDITIONS
            },
        }
        for estimator in ESTIMATORS:
            field = f"{estimator}_rate"
            block[f"difference_{estimator}"] = (
                block["conditions"]["requested"][field]
                - block["conditions"]["mismatched"][field]
            )
        if referents is not None:
            record = referents.get(cls)
            if record is None:
                raise ValueError(
                    f"{arm}/{cls}: no frozen instrument anchor; the class cannot be "
                    "placed in or out of the admitted support"
                )
            block["instrument_anchor"] = {
                "admitted": bool(record["admitted"]),
                "real_rate": record.get("real_rate"),
                "random_rate": record.get("random_rate"),
                "n_referent_families": len(record["referent"]),
            }
        per_class[cls] = block

    supports = {"all": classes}
    if referents is not None:
        supports["admitted"] = [
            cls for cls in classes if per_class[cls]["instrument_anchor"]["admitted"]
        ]

    panels: dict[str, dict[str, Any]] = {}
    for support, keys in supports.items():
        for estimator in ESTIMATORS:
            differences = {cls: per_class[cls][f"difference_{estimator}"] for cls in keys}
            panels[f"{support}/{estimator}"] = {
                "support": support,
                "estimator": estimator,
                "classes": sorted(keys),
                **class_clustered_mean(differences, resamples=resamples, seed=seed),
            }

    primary_key = f"{PRIMARY_SUPPORT}/{PRIMARY_ESTIMATOR}"
    primary = panels[primary_key]
    primary_differences = {
        cls: per_class[cls][f"difference_{PRIMARY_ESTIMATOR}"]
        for cls in supports[PRIMARY_SUPPORT]
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "arm": arm,
        "hit_field": hit_field,
        "pairing_balance": balance,
        "n_classes_measured": len(classes),
        "per_class": per_class,
        "panels": panels,
        "primary": {"key": primary_key, **primary},
        "degeneracy": degeneracy_summary(per_class),
        "class_positive_fraction": {
            "n_classes": len(primary_differences),
            "n_positive": int(sum(1 for value in primary_differences.values() if value > 0.0)),
            "minimum_classes": MINIMUM_CLASSES,
        },
        "resamples": int(resamples),
        "bootstrap_seed": int(seed),
        "resampling_unit": "the class",
        "ceiling": dict(CEILING),
    }


def degeneracy_summary(per_class: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The failure mode a naive yield hides, per condition and over the arm.

    Reported rather than gated: a conditioned decoder whose requested cells
    collapse onto few sequences is a real and interesting outcome, but a rate
    quoted without it is not interpretable.
    """

    summary: dict[str, Any] = {
        "rule": (
            "a rate is reported together with how many distinct sequences and "
            "near-duplicate groups produced it; one sequence repeated can reach any "
            "yield"
        )
    }
    for cond in CONDITIONS:
        attempts = sum(block["conditions"][cond]["n_attempts"] for block in per_class.values())
        distinct = sum(
            block["conditions"][cond]["n_distinct_sequences"] for block in per_class.values()
        )
        groups = sum(
            block["conditions"][cond]["n_near_duplicate_groups"] for block in per_class.values()
        )
        hit_groups = sum(
            block["conditions"][cond]["n_distinct_target_groups"] for block in per_class.values()
        )
        shares = [
            block["conditions"][cond]["largest_exact_duplicate_share"]
            for block in per_class.values()
        ]
        summary[cond] = {
            "n_attempts": attempts,
            "n_distinct_sequences": distinct,
            "n_near_duplicate_groups": groups,
            "n_distinct_target_groups": hit_groups,
            "exact_duplication_rate": 1.0 - distinct / attempts if attempts else None,
            "max_single_sequence_share_over_classes": max(shares) if shares else None,
            "classes_with_single_sequence_majority": sorted(
                cls
                for cls, block in per_class.items()
                if block["conditions"][cond]["largest_exact_duplicate_share"] > 0.5
            ),
        }
    return summary


def floor_rates(
    rows: Iterable[Mapping[str, Any]],
    *,
    referents: Mapping[str, Mapping[str, Any]] | None = None,
    hit_field: str = "any_profile_hit",
) -> dict[str, Any]:
    """The unconditioned floor arms' rates, for the campaign's clause 2.

    A floor arm generates under **no** class request at all, so its contribution
    to a class's reference is not its any-family rate but the rate at which its
    unprompted generations land in *that* class: an arm that recognisably
    produces one family at random should not be charged for the fifteen classes
    it was never asked about. ``referents`` supplies the frozen class-to-Pfam map
    and the per-class rates are computed against it, which is the convention the
    2026-08-26 campaign report used. The any-family rate is reported alongside as
    a descriptive attainability number and is never the clause.
    """

    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("condition") == "unconditioned_floor" or row.get("role") == "unconditioned_floor":
            by_arm[str(row["arm"])].append(dict(row))
    record: dict[str, Any] = {}
    for arm, cell in sorted(by_arm.items()):
        groups = [row.get("near_duplicate_group") or row["sequence_sha256"] for row in cell]
        block: dict[str, Any] = {
            "n_attempts": len(cell),
            "any_family_rate": float(
                np.mean([1.0 if bool(row.get(hit_field)) else 0.0 for row in cell])
            ),
            "n_distinct_sequences": len({row["sequence_sha256"] for row in cell}),
            "n_near_duplicate_groups": len(set(groups)),
            "per_class": {},
        }
        if referents:
            for class_key, entry in referents.items():
                wanted = {str(value).split(".", 1)[0] for value in entry["referent"]}
                hits = [
                    any(
                        str(value).split(".", 1)[0] in wanted
                        for value in (row.get("pfam_families") or ())
                    )
                    for row in cell
                ]
                block["per_class"][class_key] = {
                    "attempt_rate": float(np.mean([1.0 if hit else 0.0 for hit in hits])),
                    "grouped_rate": grouped_rate(hits, np.asarray(groups)),
                }
        record[arm] = block
    return record


def combine_streams(
    streams: Mapping[str, Mapping[str, Any]], *, key: str | None = None
) -> dict[str, Any]:
    """Across-stream summary of one arm's difference, at the stream as the unit.

    Seed streams resample from a fixed checkpoint, so the across-stream interval
    is a Student-t interval over the stream means, matching the replication
    campaign's own declared uncertainty rule. It is not a second bootstrap and it
    does not pool attempts across streams as if they were one draw.
    """

    chosen = key or f"{PRIMARY_SUPPORT}/{PRIMARY_ESTIMATOR}"
    points: dict[str, float] = {}
    for name, payload in sorted(streams.items()):
        panel = payload["panels"][chosen]
        if panel.get("mean") is None:
            raise ValueError(
                f"stream {name!r} reports no mean for {chosen!r}: {panel.get('degenerate_reason')}"
            )
        points[name] = float(panel["mean"])
    record = {
        "panel_key": chosen,
        "streams": points,
        "n_streams": len(points),
        "rule": (
            "equal weight per campaign stream; Student-t interval over the stream "
            "means, as the replication campaign declares. Streams are not independent "
            "training lineages"
        ),
    }
    if len(points) < 2:
        record["interval"] = None
        record["mean"] = next(iter(points.values())) if points else None
        record["single_stream_reason"] = (
            "one stream carries no across-stream interval; the within-stream "
            "class-clustered interval is the only uncertainty reported"
        )
        return record
    record.update(mean_interval(list(points.values())))
    record["direction_replicated"] = bool(
        all(value > 0.0 for value in points.values()) or all(value < 0.0 for value in points.values())
    )
    return record


#: The declared primary unconditioned floor for both conditioned protein arms:
#: an unconditioned protein decoder at comparable scale. The second floor is
#: reported beside it and is never the gate, which is the campaign's own rule.
PRIMARY_FLOOR = "progen2-medium"


def requested_minus_floor(
    contrast: Mapping[str, Any], floor: Mapping[str, Any], *, arm: str
) -> dict[str, Any]:
    """Per floor arm, the requested rate minus that arm's rate on the same classes."""

    estimator = contrast["primary"]["estimator"]
    keys = contrast["panels"][contrast["primary"]["key"]]["classes"]
    field = f"{estimator}_rate"
    blocks: dict[str, Any] = {}
    for name, block in sorted(floor.items()):
        per_class = block.get("per_class") or {}
        absent = sorted(cls for cls in keys if cls not in per_class)
        if absent:
            blocks[name] = {
                "ci95": None,
                "mean": None,
                "degenerate": True,
                "degenerate_reason": (
                    f"the floor arm {name} carries no per-class rate for {absent[:3]}; a "
                    "floor measured on a different class set is not this reference"
                ),
            }
            continue
        differences = {
            cls: contrast["per_class"][cls]["conditions"]["requested"][field]
            - per_class[cls][field]
            for cls in keys
        }
        blocks[name] = {
            **class_clustered_mean(
                differences, resamples=contrast["resamples"], seed=contrast["bootstrap_seed"]
            ),
            "floor_arm": name,
            "floor_any_family_rate": block["any_family_rate"],
            "estimator": estimator,
        }
    return blocks


def arm_verdict(contrast: Mapping[str, Any], *, floor: Mapping[str, Any] | None) -> dict[str, Any]:
    """The campaign's compound verdict, rebuilt from this contrast.

    Clause 2 reads the **primary** floor arm only, as the campaign declares;
    the secondary floor is reported and never gated on. Without a floor measured
    on the same classes the clause is unavailable rather than assumed, which
    keeps the outcome ``not_scored`` instead of promoting a two-clause pass.
    """

    estimator = contrast["primary"]["estimator"]
    support = contrast["primary"]["support"]
    keys = contrast["panels"][contrast["primary"]["key"]]["classes"]
    per_class_contrast = {
        cls: contrast["per_class"][cls][f"difference_{estimator}"] for cls in keys
    }
    blocks = requested_minus_floor(contrast, floor or {}, arm=contrast["arm"])
    primary_floor = blocks.get(PRIMARY_FLOOR) or {
        "ci95": None,
        "mean": None,
        "degenerate": True,
        "degenerate_reason": (
            f"the primary floor arm {PRIMARY_FLOOR} was not measured in this run, so "
            "clause 2 is unavailable and no arm-level verdict is issued"
        ),
    }
    verdict = compound_verdict(
        against_mismatch=contrast["panels"][contrast["primary"]["key"]],
        against_floor=primary_floor,
        per_class_contrast=per_class_contrast,
    )
    verdict["support"] = support
    verdict["estimator"] = estimator
    verdict["primary_floor"] = PRIMARY_FLOOR
    verdict["requested_minus_floor"] = blocks
    verdict["secondary_floors_are_reported_not_gated"] = sorted(
        name for name in blocks if name != PRIMARY_FLOOR
    )
    return verdict


def read_attempts(path: Path) -> list[dict[str, Any]]:
    """One attempt ledger, refusing duplicate identifiers."""

    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"{path} carries no attempt")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} carries duplicate attempt identifiers")
    return rows
