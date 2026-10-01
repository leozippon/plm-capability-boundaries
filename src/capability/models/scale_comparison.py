"""The arithmetic a descriptive scale ladder is read with, declared once.

Two campaigns now compare checkpoints rung by rung on the same frozen queues:
EXP-R2-224's ProGen2 medium -> large -> xlarge first round, and EXP-R2-225's
independent second stage over larger public checkpoints. They ask the same
question of different rungs, so the *arithmetic* -- how a paired difference is
resampled over its original units, how a cohort is aligned across rungs, how a
stage-41 identification record is qualified -- must be one implementation, while
each campaign's **freeze** stays where it was written.

That split is deliberate and is what this module exists for. A freeze is a
promise about particular rungs, particular censuses and particular seeds; it
belongs in the stage that made it, next to the refusals that enforce it. The
arithmetic underneath is the same operation whatever it is pointed at, and a
second copy of it would be free to drift from the first exactly where the two
campaigns are supposed to be commensurable. Nothing here declares a rung, a
census, a seed or a resample count: every one of those arrives as an argument.

Two consequences follow, and both are load-bearing.

Rung order is checked here but never *chosen* here. :func:`require_rungs` takes
the tuple a caller froze and refuses anything else, so a stage that hands over
its own frozen order gets the same refusal it had when the check was local.

A census is optional and, when absent, is not checked rather than defaulted.
:class:`DmsCensus` and :class:`MegascaleCensus` are the fixed counts a campaign
declared; passing ``None`` means a caller is aligning payloads whose census is
not the frozen one -- a test fixture, a shape check -- and the structural
refusals (a rung-specific skip, a skip that is not a shared-context overflow, a
disagreeing digest) still all apply. There is no default census, because a
default would be a fourth place a frozen number could live.

The one class of *name* declared here is the scoring-strata vocabulary, and it
is not a campaign freeze. A rule that N-to-C, bidirectional and masked /
pseudo-likelihood scores are never pooled is only enforceable if the three have
spellings a scoring stage writes and a comparison stage reads; three string
literals in three files would be the drift this module exists to prevent, on the
quantity that decides whether a paired difference is a difference of
checkpoints or of estimands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..mutation import designed_referent as D

__all__ = [
    "SCORING_STRATA",
    "STRATUM_BIDIRECTIONAL",
    "STRATUM_MASKED_PSEUDO_LIKELIHOOD",
    "STRATUM_N_TO_C",
    "DmsCensus",
    "MegascaleCensus",
    "align_dms",
    "align_megascale",
    "compound_verdict",
    "context_excluded_assays",
    "endpoint",
    "lower_bound_positive",
    "paired_unit_delta",
    "qualify_per_rung_stage41",
    "qualify_stage41",
    "require_rungs",
    "require_uniform_dtype",
    "require_uniform_stratum",
    "side_keys",
    "unit_mean_record",
]


@dataclass(frozen=True)
class DmsCensus:
    """The declared and analysed DMS cohort one ladder is frozen at.

    Two counts, because they are not the same set. ``declared_*`` is the
    benchmark queue a campaign froze; ``analysis_*`` is the subset every rung of
    that ladder can actually score, once the assays whose rendered variants
    exceed the ladder's shared context are removed. Carrying both is what lets
    an artefact say which cohort was declared and which was analysed instead of
    implying they are the same.

    ``context_excluded_assays`` is that removed set, named rather than counted,
    so a ladder cannot swap one unscorable assay for another and still report
    the frozen analysis count. ``label`` names the freeze the numbers come from
    and appears in the refusal, so a reader of the traceback learns which
    campaign's declaration was violated.
    """

    label: str
    declared_assays: int
    declared_clusters: int
    analysis_assays: int
    analysis_clusters: int
    context_excluded_assays: tuple[str, ...]


@dataclass(frozen=True)
class MegascaleCensus:
    """The fixed MegaScale design and natural-control counts of one ladder."""

    design_wildtypes: int
    design_series: int
    natural_wildtypes: int
    natural_clusters: int


def require_rungs(names: Sequence[str], rungs: Sequence[str]) -> None:
    """Refuse any comparison whose rung list is not the frozen one."""

    got = list(names)
    if got != list(rungs):
        raise ValueError(
            f"the descriptive comparison is fixed as {list(rungs)}; got {got}"
        )


#: The scoring conventions EXP-R2-225 declares as separate strata, spelled once.
#:
#: "N-to-C, bidirectional, and masked / pseudo-likelihood are separate strata. Do
#: not pool them, rank them together, or convert one into another to complete a
#: ladder." A rule about not mixing three things is only enforceable if the three
#: have names that a stage writes and another stage reads, so the names are here
#: rather than as three string literals in whichever file needed one first.
#:
#: These label the *convention a model score was computed under*, not a model and
#: not a benchmark. Two rungs of one ladder may be compared only when they carry
#: the same one; two ladders under different ones are two readings, reported side
#: by side and never pooled.
STRATUM_N_TO_C = "n_to_c_summed_log_likelihood"
STRATUM_BIDIRECTIONAL = "bidirectional_mean_of_directional_sums"
STRATUM_MASKED_PSEUDO_LIKELIHOOD = "masked_pseudo_likelihood"
SCORING_STRATA = (
    STRATUM_N_TO_C,
    STRATUM_BIDIRECTIONAL,
    STRATUM_MASKED_PSEUDO_LIKELIHOOD,
)


def require_uniform_stratum(
    models: dict[str, dict[str, Any]], *, rungs: Sequence[str], label: str
) -> str:
    """One scoring stratum across the rungs of one endpoint.

    :func:`require_uniform_dtype`'s shape, on the quantity that separates a
    paired difference of checkpoints from a paired difference of *estimands*. A
    rung with no recorded stratum is refused rather than assumed to share the
    others': the field was introduced with this campaign, so its absence means
    the payload predates the rule and cannot vouch for what it was scored under.
    """

    seen: dict[str, str] = {}
    for name in rungs:
        settings = models[name].get("settings")
        stratum = settings.get("scoring_stratum") if isinstance(settings, Mapping) else None
        if not stratum:
            raise ValueError(
                f"{label}: {name} records no scoring_stratum, so it cannot be "
                "paired with a rung whose scoring convention is known"
            )
        if str(stratum) not in SCORING_STRATA:
            raise ValueError(
                f"{label}: {name} records scoring stratum {stratum!r}, which is "
                f"not one of the declared {list(SCORING_STRATA)}"
            )
        seen[name] = str(stratum)
    if len(set(seen.values())) != 1:
        raise ValueError(
            f"{label}: the rungs were scored under mixed strata: {seen}. N-to-C, "
            "bidirectional and masked / pseudo-likelihood are separate estimands "
            "and are never pooled to complete a ladder"
        )
    return next(iter(seen.values()))


def require_uniform_dtype(
    models: dict[str, dict[str, Any]], *, rungs: Sequence[str], label: str
) -> str:
    """One scoring precision across the rungs of one endpoint.

    A rung scored at a different precision from the rung it is paired against
    makes the paired difference partly a difference of arithmetic rather than of
    checkpoints. Nothing downstream can detect that from the numbers, so it is
    refused here, where every rung's own record of what it was scored at is in
    hand.

    Here rather than in either stage because both descriptive ladders form the
    same paired difference and neither may be free to stop checking. The rungs
    arrive as an argument for the same reason every other quantity in this
    module does: the campaigns declare different ones.
    """

    seen: dict[str, str] = {}
    for name in rungs:
        settings = models[name].get("settings")
        dtype = settings.get("dtype") if isinstance(settings, Mapping) else None
        if not dtype:
            raise ValueError(f"{label}: {name} records no scoring dtype")
        seen[name] = str(dtype)
    if len(set(seen.values())) != 1:
        raise ValueError(f"{label}: the rungs were scored at mixed precision: {seen}")
    return next(iter(seen.values()))


def lower_bound_positive(record: dict[str, Any]) -> bool | None:
    """True/False when an interval exists; None when it cannot be read."""

    if record.get("degenerate") or record.get("interval") is None:
        return None
    return bool(record["interval"][0] > 0.0)


def compound_verdict(conditions: dict[str, bool | None]) -> bool | str:
    """True only when every named condition is True; unresolved if any is missing."""

    if any(value is None for value in conditions.values()):
        return "unresolved"
    return all(bool(value) for value in conditions.values())


def _aligned_keys(left: dict[str, Any], right: dict[str, Any], *, label: str) -> list[str]:
    if set(left) != set(right):
        missing = sorted(set(left) ^ set(right))
        raise ValueError(f"{label} keys disagree: {missing}")
    return sorted(left)


def paired_unit_delta(
    smaller: dict[str, float],
    larger: dict[str, float],
    units: dict[str, str],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired Δρ over the original unit labels, resampled as those units."""

    names = _aligned_keys(smaller, larger, label="paired Δρ")
    if not names:
        raise ValueError("paired Δρ has no shared units")
    missing = [name for name in names if name not in units]
    if missing:
        raise ValueError(f"paired Δρ is missing unit labels for {missing}")
    extra = sorted(set(units) - set(names))
    if extra:
        raise ValueError(f"paired Δρ has unit labels with no paired values: {extra}")
    unit_names = [units[name] for name in names]
    if any(not label for label in unit_names):
        raise ValueError("paired Δρ has an empty unit label")
    values = [larger[name] - smaller[name] for name in names]
    return D.unit_bootstrap(values, unit_names, resamples=resamples, seed=seed)


def unit_mean_record(
    values: dict[str, float],
    units: dict[str, str],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    names = sorted(values)
    missing = [name for name in names if name not in units]
    if missing:
        raise ValueError(f"unit labels missing for {missing}")
    return D.unit_bootstrap(
        [values[name] for name in names],
        [units[name] for name in names],
        resamples=resamples,
        seed=seed,
    )


def endpoint(
    per_rung: dict[str, dict[str, float]],
    units: dict[str, str],
    *,
    rungs: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """One endpoint: a per-rung interval each, and a paired Δρ on each pair.

    The seed offsets are positional -- rung ``i`` draws at ``seed + i`` and pair
    ``j`` at ``seed + 10 + j`` -- so two endpoints of the same ladder given
    different base seeds never share a resampling draw, and the same endpoint
    re-run at the same base seed reproduces exactly.
    """

    rung_records = {
        name: unit_mean_record(per_rung[name], units, resamples=resamples, seed=seed + index)
        for index, name in enumerate(rungs)
    }
    adjacent = {}
    for offset, (smaller, larger) in enumerate(pairs):
        delta = paired_unit_delta(
            per_rung[smaller],
            per_rung[larger],
            units,
            resamples=resamples,
            seed=seed + 10 + offset,
        )
        adjacent[f"{smaller}__{larger}"] = delta
    return {"per_rung": rung_records, "adjacent_delta_rho": adjacent}


def _assay_names(payload: dict[str, Any], *, label: str) -> list[str]:
    names = [row["assay"] for row in payload["assays"]]
    if len(names) != len(set(names)):
        raise ValueError(f"{label} repeats an assay")
    return names


def context_excluded_assays(
    models: dict[str, dict[str, Any]],
    lookup_set: set[str],
    *,
    rungs: Sequence[str],
    context: int,
    exclusion_reason: str,
) -> set[str]:
    """The assays no rung can score, refused unless every rung skips the same set.

    A rendered variant longer than the context every rung of a ladder shares is
    unscorable on all of them, so skipping it defines the cohort. A skip one
    rung takes and another does not would break the pairing the comparison rests
    on, and a skip taken for any other reason is a scoring failure rather than a
    cohort definition. Both are refusals.
    """

    per_rung: dict[str, set[str]] = {}
    for name in rungs:
        skipped = models[name].get("skipped")
        if skipped is None:
            raise ValueError(
                f"{name} carries no skipped block; stage 20 writes one on every "
                "scored arm and its absence cannot be read as an empty skip"
            )
        names: set[str] = set()
        for row in skipped:
            assay = str(row["assay"])
            if assay in names:
                raise ValueError(f"{name} repeats skipped assay {assay}")
            reason = str(row.get("reason", ""))
            if exclusion_reason not in reason:
                raise ValueError(
                    f"{name} skipped {assay} for {reason!r}; only a rendered "
                    "variant exceeding this ladder's shared context defines the "
                    "cohort, every other skip is a scoring failure"
                )
            if int(row["context"]) != context:
                raise ValueError(
                    f"{name} skipped {assay} against context {row['context']}, "
                    f"not this ladder's shared {context}"
                )
            if int(row["max_tokens"]) <= context:
                raise ValueError(
                    f"{name} skipped {assay} at {row['max_tokens']} tokens, "
                    f"which fits the {context}-position context"
                )
            names.add(assay)
        per_rung[name] = names
    first = rungs[0]
    reference = per_rung[first]
    for name, names in per_rung.items():
        if names != reference:
            raise ValueError(
                f"{name} and {first} disagree on whether "
                f"{sorted(names ^ reference)} can be scored; a paired "
                "comparison needs one common support across the rungs"
            )
    outside = sorted(reference - lookup_set)
    if outside:
        raise ValueError(f"skipped assays {outside} are not in the LOOKUP cohort")
    return reference


def align_dms(
    models: dict[str, dict[str, Any]],
    lookup: dict[str, Any],
    *,
    rungs: Sequence[str],
    context: int,
    exclusion_reason: str,
    census: DmsCensus | None = None,
) -> dict[str, Any]:
    """Align the DMS cohort: every rung's assay set equals the jointly scored subset.

    LOOKUP declares the benchmark's whole substitution queue. The rungs score
    the assays of it that fit the context they share. The returned record
    carries both, so the artefact can say which cohort was declared and which
    was analysed instead of implying they are the same set.
    """

    require_rungs(list(models), rungs)
    lookup_assays = _assay_names(lookup, label="LOOKUP")
    lookup_set = set(lookup_assays)
    excluded = context_excluded_assays(
        models,
        lookup_set,
        rungs=rungs,
        context=context,
        exclusion_reason=exclusion_reason,
    )
    analysis_assays = [name for name in lookup_assays if name not in excluded]
    analysis_set = set(analysis_assays)
    for name in rungs:
        model_set = set(_assay_names(models[name], label=name))
        if model_set != analysis_set:
            raise ValueError(
                f"{name} assay set disagrees with the jointly scored LOOKUP "
                f"subset: {sorted(model_set ^ analysis_set)}"
            )
    by_assay = {row["assay"]: row for row in lookup["assays"]}
    units: dict[str, str] = {}
    raw: dict[str, dict[str, float]] = {name: {} for name in rungs}
    contrasts: dict[str, dict[str, dict[str, float]]] = {
        "model_minus_lookup": {name: {} for name in rungs},
        "model_minus_blosum62": {name: {} for name in rungs},
    }
    for assay in analysis_assays:
        lookup_row = by_assay[assay]
        expected_digest = lookup_row["mutant_digest"]
        expected_wildtype = lookup_row.get("wildtype_id")
        expected_n = lookup_row.get("n_variants")
        for rung in rungs:
            rows = {row["assay"]: row for row in models[rung]["assays"]}
            entry = rows[assay]
            if entry["mutant_digest"] != expected_digest:
                raise ValueError(
                    f"{assay}: mutant_digest disagrees between {rung} and LOOKUP"
                )
            if expected_wildtype is not None and entry.get("wildtype_id") not in (
                None,
                expected_wildtype,
            ):
                raise ValueError(
                    f"{assay}: wildtype_id disagrees between {rung} and LOOKUP"
                )
            if expected_n is not None and entry.get("n_variants") not in (
                None,
                expected_n,
            ):
                raise ValueError(
                    f"{assay}: n_variants disagrees between {rung} and LOOKUP"
                )
            rho = float(entry["spearman"])
            raw[rung][assay] = rho
            contrasts["model_minus_lookup"][rung][assay] = (
                rho - float(lookup_row["spearman"]["lookup"])
            )
            contrasts["model_minus_blosum62"][rung][assay] = (
                rho - float(lookup_row["spearman"]["blosum62"])
            )
        units[assay] = str(lookup_row["cluster"])
    declared_clusters = {str(row["cluster"]) for row in lookup["assays"]}
    if census is not None:
        if len(lookup_assays) != census.declared_assays:
            raise ValueError(
                f"DMS cohort is {len(lookup_assays)} assays, not the fixed "
                f"{census.declared_assays}-assay declared substitution census"
            )
        if len(declared_clusters) != census.declared_clusters:
            raise ValueError(
                f"LOOKUP has {len(declared_clusters)} clusters, not the fixed "
                f"{census.declared_clusters}-family census"
            )
        if sorted(excluded) != sorted(census.context_excluded_assays):
            raise ValueError(
                f"the context-excluded assays are not the frozen {census.label} "
                f"set: {sorted(set(excluded) ^ set(census.context_excluded_assays))}"
            )
        if len(analysis_assays) != census.analysis_assays:
            raise ValueError(
                f"the jointly scored DMS set is {len(analysis_assays)} assays, "
                f"not the fixed {census.analysis_assays}"
            )
        if len(set(units.values())) != census.analysis_clusters:
            raise ValueError(
                f"the jointly scored DMS set spans {len(set(units.values()))} "
                f"families, not the fixed {census.analysis_clusters}"
            )
    return {
        "declared_assays": lookup_assays,
        "declared_clusters": len(declared_clusters),
        "context_excluded_assays": sorted(excluded),
        "analysis_assays": analysis_assays,
        "units": units,
        "raw": raw,
        "contrasts": contrasts,
    }


def side_keys(
    payload: dict[str, Any],
    *,
    side: str,
    spearman_of: Callable[[dict[str, Any]], Any],
    admissible: frozenset[str] | None = None,
) -> set[str]:
    """The wild types of one MegaScale side that carry a usable Spearman."""

    keys = set()
    for name, entry in payload["wildtypes"].items():
        if entry.get("kind") != side:
            continue
        if admissible is not None and name not in admissible:
            continue
        if spearman_of(entry) is None:
            continue
        keys.add(name)
    return keys


def align_megascale(
    models: dict[str, dict[str, Any]],
    baselines: dict[str, Any],
    *,
    rungs: Sequence[str],
    side: str,
    baseline_name: str,
    admissible: frozenset[str] | None = None,
    census: MegascaleCensus | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Align one MegaScale side: non-null keys equal across rungs and the baseline.

    ``admissible`` restricts the side to a declared census. The design side is
    restricted to the certified zero-hit designs; on the natural control, where
    hitting the corpus is the expected state, it has no meaning and is ``None``.
    """

    require_rungs(list(models), rungs)
    digests = {name: payload["cohort_sha256"] for name, payload in models.items()}
    if len(set(digests.values())) != 1:
        raise ValueError(f"MegaScale cohort_sha256 disagrees across rungs: {digests}")
    model_keys = {
        name: side_keys(
            payload,
            side=side,
            spearman_of=lambda entry: entry.get("spearman"),
            admissible=admissible,
        )
        for name, payload in models.items()
    }
    first = rungs[0]
    first_keys = model_keys[first]
    for name, keys in model_keys.items():
        if keys != first_keys:
            raise ValueError(
                f"{name} {side} non-null Spearman keys disagree with "
                f"{first}: {sorted(keys ^ first_keys)}"
            )
    if admissible is not None and first_keys != set(admissible):
        raise ValueError(
            f"the declared {side} census is not scored on every rung: "
            f"{sorted(first_keys ^ set(admissible))}"
        )
    baseline_keys = side_keys(
        baselines,
        side=side,
        spearman_of=lambda entry: (entry.get("spearman") or {}).get(baseline_name),
        admissible=admissible,
    )
    if baseline_keys != first_keys:
        raise ValueError(
            f"baseline {side} keys for {baseline_name} disagree with the models: "
            f"{sorted(baseline_keys ^ first_keys)}"
        )
    if side == "design":
        expected_n = None if census is None else census.design_wildtypes
        expected_units = None if census is None else census.design_series
        unit_label = "design series"
    else:
        expected_n = None if census is None else census.natural_wildtypes
        expected_units = None if census is None else census.natural_clusters
        unit_label = "WT_cluster"
    if expected_n is not None and len(first_keys) != expected_n:
        raise ValueError(
            f"MegaScale {side} has {len(first_keys)} wild types, not the fixed "
            f"{expected_n}"
        )
    units: dict[str, str] = {}
    raw: dict[str, dict[str, float]] = {name: {} for name in rungs}
    contrast: dict[str, dict[str, float]] = {name: {} for name in rungs}
    for wildtype in sorted(first_keys):
        baseline_entry = baselines["wildtypes"][wildtype]
        expected_kind = side
        expected_unit = str(baseline_entry["unit"])
        if baseline_entry.get("kind") != expected_kind:
            raise ValueError(f"{wildtype}: baseline kind is not {expected_kind}")
        for rung, payload in models.items():
            entry = payload["wildtypes"][wildtype]
            if entry.get("kind") != expected_kind:
                raise ValueError(f"{wildtype}: {rung} kind disagrees with the baseline")
            if str(entry["unit"]) != expected_unit:
                raise ValueError(f"{wildtype}: {rung} unit disagrees with the baseline")
            rho = float(entry["spearman"])
            raw[rung][wildtype] = rho
            contrast[rung][wildtype] = rho - float(baseline_entry["spearman"][baseline_name])
            if entry.get("n_variants") not in (None, baseline_entry.get("n_variants")):
                raise ValueError(
                    f"{wildtype}: {rung} n_variants disagrees with the baseline"
                )
        units[wildtype] = expected_unit
    n_units = len(set(units.values()))
    if expected_units is not None and n_units != expected_units:
        raise ValueError(
            f"MegaScale {side} has {n_units} {unit_label} units, not the fixed "
            f"{expected_units}"
        )
    return units, raw, contrast


def _positive_finite_interval(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        interval = [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(end) for end in interval) or interval[0] <= 0.0:
        return None
    return interval


def _arm_results(report: dict[str, Any], *, label: str) -> list[dict[str, Any]]:
    if "arm_results" not in report:
        raise ValueError(
            f"{label} needs a full stage-41 report with arm_results; a "
            "summary-only artefact is not an identification record"
        )
    return list(report["arm_results"])


def qualify_per_rung_stage41(
    reports: dict[str, dict[str, Any]], *, rungs: Sequence[str], row_arm: str
) -> dict[str, Any]:
    """Qualify a ladder whose rungs each wrote their **own** stage-41 report.

    A joint checkpoint's identification record is not shaped like a panel arm's.
    ``joint_mode_qualification.py`` writes one sidecar per checkpoint per
    mode, and ``context_information_bootstrap.py`` turns each into its own
    report whose ``arm`` field names the *condition* -- ``protein_declared``
    against ``protein_reversed`` -- rather than the checkpoint. A ladder over
    three such checkpoints is therefore three reports and not three rows of one,
    and :func:`qualify_stage41` cannot read it: it selects rows by rung name and
    would find none.

    ``row_arm`` names the condition the ladder is read on, so the declared and
    the reversed conditions of the same checkpoint can never be paired against
    each other by accident. Everything after the selection is the same
    qualification :func:`qualify_stage41` applies: identical block sets, every
    row ``PASS`` with a finite displacement-corrected interval strictly above
    zero, and one cohort digest per block across the rungs.
    """

    rungs = tuple(rungs)
    missing_reports = [name for name in rungs if name not in reports]
    if missing_reports:
        raise ValueError(f"no stage-41 report for rungs {missing_reports}")
    by_rung: dict[str, list[dict[str, Any]]] = {}
    for name in rungs:
        rows = [
            row
            for row in _arm_results(reports[name], label=f"{name}'s qualification")
            if row.get("arm") == row_arm and not row.get("is_unigram_null_control")
        ]
        if not rows:
            raise ValueError(
                f"{name}'s stage-41 report carries no {row_arm!r} rows; the "
                "condition this ladder is read on is not in it"
            )
        by_rung[name] = rows
    record = _qualify_rung_rows(by_rung, rungs=rungs)
    record["source"] = "stage41_arm_results_per_rung"
    record["row_arm"] = row_arm
    return record


def qualify_stage41(report: dict[str, Any], *, rungs: Sequence[str]) -> dict[str, Any]:
    """Qualify a ladder's rungs from a full stage-41 report's ``arm_results``.

    Summary-only artefacts, missing rungs or blocks, mixed identification
    verdicts, and disagreed cohort digests are refusals. Every retained row
    must be ``PASS``.
    """

    rungs = tuple(rungs)
    rows = [
        row
        for row in _arm_results(report, label="a scale comparison")
        if row.get("arm") in rungs and not row.get("is_unigram_null_control")
    ]
    by_rung: dict[str, list[dict[str, Any]]] = {name: [] for name in rungs}
    for row in rows:
        by_rung[row["arm"]].append(row)
    missing = [name for name in rungs if not by_rung[name]]
    if missing:
        raise ValueError(f"stage-41 arm_results is missing rungs {missing}")
    record = _qualify_rung_rows(by_rung, rungs=rungs)
    record["source"] = "stage41_arm_results"
    return record


def _qualify_rung_rows(
    by_rung: dict[str, list[dict[str, Any]]], *, rungs: Sequence[str]
) -> dict[str, Any]:
    """The qualification itself, once, whatever shape the rows were selected from."""

    rungs = tuple(rungs)
    block_sets = {
        name: frozenset(row["block_id"] for row in by_rung[name]) for name in rungs
    }
    for name in rungs:
        if len(block_sets[name]) != len(by_rung[name]):
            raise ValueError(f"stage-41 arm_results repeats a block for {name}")
    reference_blocks = block_sets[rungs[0]]
    if not reference_blocks:
        raise ValueError("stage-41 arm_results carries no blocks for the scale rungs")
    for name, blocks in block_sets.items():
        if blocks != reference_blocks:
            raise ValueError(
                f"{name} covers blocks {sorted(blocks)}, not {sorted(reference_blocks)}"
            )
    record: dict[str, Any] = {
        "passed": True,
        "blocks": sorted(reference_blocks),
        "rungs": {},
    }
    statuses: list[str] = []
    for name in rungs:
        per_block: dict[str, Any] = {}
        for row in by_rung[name]:
            status = row.get("per_arm_identification_status")
            statuses.append(str(status))
            interval = _positive_finite_interval(
                row.get("displacement_corrected_ci_95")
            )
            if status != "PASS" or interval is None:
                raise ValueError(
                    f"{name} block {row.get('block_id')} identification is "
                    f"{status!r}, not PASS with a finite displacement-corrected "
                    "interval strictly above zero"
                )
            per_block[row["block_id"]] = {
                "per_arm_identification_status": status,
                "displacement_corrected_ci_95": list(interval),
                "cohort_digest": row["cohort_digest"],
                "cohort_name": row.get("cohort_name"),
            }
        record["rungs"][name] = {"blocks": per_block}
    mixed = sorted(set(statuses))
    if mixed != ["PASS"]:
        raise ValueError(
            f"stage-41 identification is not uniformly PASS across rungs: {mixed}"
        )
    for block_id in reference_blocks:
        digests = {
            name: record["rungs"][name]["blocks"][block_id]["cohort_digest"]
            for name in rungs
        }
        if len(set(digests.values())) != 1:
            raise ValueError(
                f"block {block_id} cohort_digest disagrees across rungs: {digests}"
            )
    return record
