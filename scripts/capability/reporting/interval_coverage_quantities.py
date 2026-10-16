"""The coverage a nominal 95% interval actually realises at the resampling floor.

The guidance figure's caption and the Methods both read the sealed-eight
bias-corrected accelerated bounds as optimistic rather than exact, and both quote
one number for why: the realised coverage of a nominal 95% interval at eight
resampling units is about 0.89. Until now the table carried no row for it, so the
manuscript's own checker traced the number only by a coincidental value match to
an unrelated rate. This module gives it a row.

Where the number comes from matters, and it is not a fitted or resampled result
of either generation campaign. It is a simulation that sits beside the
eight-unit floor in ``src.capability.core.statistics``, where the floor's
derivation is recorded: on 1,200 normal samples per point, the realised coverage
of a nominal 95% *percentile* interval over the cluster mean is 0.50 at two
units, rising to 0.89 at eight and 0.94 at four hundred, and eight is where the
shortfall stops being of the same order as the confidence level itself. What was
simulated is therefore a percentile interval, while the campaigns that cite the
figure form bias-corrected accelerated intervals. The campaigns say so in their
own note and read their bounds against the figure as a reference; the row says so
in its own claim, so it cannot be quoted as a measured coverage of a BCa
interval. That distinction is the one thing a row for this number has to carry.

Nothing in the repository records the figure as a machine-readable scalar. The
module holds it in the declaration comment that justifies the floor, and the two
retained campaign analyses hold it in the prose of their interval-coverage note.
It is therefore bound by an anchor, following the pipeline-record precedent: the
anchor has to match exactly once in each text, every text has to yield the same
value, and anything else produces a gap and no row rather than one of several
readings. The absence of a machine-readable record is itself registered as a gap,
because a number a reader can only find by reading a comment is not retained in
the sense the rest of this table means.

The eight-unit floor is not re-emitted here. It is a declared design constant the
resampling modules already share, its numeral collides with numbers the
manuscript writes for other things, and the campaign records carry it on their
support descriptions; a row for it could only dilute the attribution of a number
the manuscript does quote.
"""
from __future__ import annotations

import re

from src.capability.core.evidence_ledger import LedgerError, resolve
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS
# The emit contract --- how a row carries its locator, digest and pointer --- is
# the follow-up module's.
from scripts.capability.reporting.followup_quantities import _emitter

#: Where the coverage was measured, beside the floor it justifies.
STATISTICS = "src/capability/core/statistics.py"
#: The retained analyses that reproduce the figure in their own interval note and
#: read their bounds against it, with the pointer to the note and to the floor
#: each one declares. The first is the frozen confirmatory analysis that owns the
#: interval convention for the sealed eight; the second reproduces it.
CAMPAIGNS = (
    ("results/R6/e20_confirmatory_analysis_20261014/confirmatory.json",
     ("frozen", "interval_coverage_note"), ("minimum_bootstrap_units",)),
    ("results/R6/progen3_robustness_analysis_20261015/robustness.json",
     ("discipline", "interval_coverage_note"), ("discipline", "minimum_bootstrap_units")),
)

#: How the campaigns state the figure, and how the resampling module records the
#: coverage point the floor is set at. Each has to match exactly once in its own
#: text, because a number bound to an anchor that matches twice is bound to
#: neither occurrence.
NOTE_ANCHOR = re.compile(
    r"realised coverage of a nominal 95% percentile interval is about (\d\.\d+) at eight units")
MODULE_ANCHOR = re.compile(r"(\d\.\d\d) at eight,")
#: The simulation the module's coverage table was measured on. It travels on the
#: support description, which is where the population behind a figure belongs.
SAMPLE_ANCHOR = re.compile(r"(\d+) normal samples per point")

COVERAGE = "dimensionless probability"


def add_interval_coverage_quantities(ledger) -> None:
    emit = _emitter(ledger)
    _realised_coverage(ledger, emit)


def _anchored(anchor: re.Pattern, text: str) -> str | None:
    """One value bound to one occurrence of an anchor, or nothing.

    A number bound to an anchor that matches twice is bound to neither
    occurrence, so a text that does not pin it down yields nothing and the caller
    records a gap.
    """
    matches = anchor.findall(text)
    return matches[0] if len(matches) == 1 else None


def _floor(ledger, path: str, pointer: tuple[str, ...]) -> int:
    """The unit floor a campaign publishes above, which has to be the declared one."""
    declared = int(resolve(ledger.artifacts.json(path), pointer))
    if declared != MINIMUM_BOOTSTRAP_UNITS:
        raise LedgerError(
            f"{path}: the campaign publishes intervals above a {declared}-unit floor and the "
            f"resampling module declares {MINIMUM_BOOTSTRAP_UNITS}, so \"at eight units\" is not "
            "one statement about one floor and the coverage cannot be read against it")
    return declared


def _realised_coverage(ledger, emit) -> None:
    module = ledger.artifacts.text(STATISTICS)
    readings = {STATISTICS: _anchored(MODULE_ANCHOR, module)}
    notes = {}
    for path, note, floor in CAMPAIGNS:
        _floor(ledger, path, floor)
        notes[path] = note
        readings[path] = _anchored(NOTE_ANCHOR,
                                   str(resolve(ledger.artifacts.json(path), note)))
    samples = _anchored(SAMPLE_ANCHOR, module)
    unbound = sorted(path for path, value in readings.items() if value is None)
    if samples is None:
        unbound = sorted({*unbound, STATISTICS})
    values = {value for value in readings.values() if value is not None}
    if unbound:
        reason = ("the declared anchor does not match exactly once in " + ", ".join(unbound) +
                  ", so the number cannot be bound to one occurrence of it")
    elif len(values) != 1:
        reason = (f"the texts that state the figure give {sorted(values)}, so there is no one "
                  "coverage to quote and the campaigns no longer cite the module they name")
    else:
        reason = None
    if reason is not None:
        ledger.gap(
            id="interval_coverage/percentile_95/realised_at_the_eight_unit_floor",
            claim="the realised coverage of a nominal 95% interval at the eight-unit resampling "
                  "floor, which the guidance figure's caption and the Methods both read their "
                  "bounds against",
            family="interval_coverage", reason=reason,
            looked_in=f"{STATISTICS} and " + ", ".join(path for path, _, _ in CAMPAIGNS),
            reachability="not_recorded",
            searched="the resampling module's floor declaration and each campaign's interval "
                     "coverage note")
        return
    coverage = values.pop()
    path, note = CAMPAIGNS[0][0], notes[CAMPAIGNS[0][0]]
    digests = "; ".join(f"{source} (sha256 {ledger.artifacts.sha256(source)})"
                        for source in (STATISTICS, *(entry[0] for entry in CAMPAIGNS)))
    support = ledger.declare_support(
        "interval_coverage/eight_unit_floor",
        f"A coverage simulation and not a reading of any campaign: {int(samples):,} normal samples "
        f"per point, the realised coverage of a nominal 95% percentile interval over the cluster "
        f"mean, measured in the resampling module beside the {MINIMUM_BOOTSTRAP_UNITS}-unit floor "
        "it justifies, which is the smallest number of resampling units at which a percentile "
        "interval may be published. What was simulated is a percentile interval, so the figure is "
        "the reference coverage the sealed-eight generation campaigns cite for their "
        "bias-corrected accelerated bounds and not a measured coverage of those bounds; both "
        f"campaigns publish above the same {MINIMUM_BOOTSTRAP_UNITS}-unit floor and state the "
        f"figure identically. The figure is prose in every text that carries it, so it is bound by "
        f"an anchor matching exactly once in each: {digests}. Nothing is resampled here")
    emit(path, note + ("<anchor>", NOTE_ANCHOR.pattern),
         identifier="interval_coverage/percentile_95/realised_at_the_eight_unit_floor",
         claim="realised coverage of a nominal 95% percentile interval over the cluster mean at the "
               f"declared {MINIMUM_BOOTSTRAP_UNITS}-unit resampling floor: the reference coverage "
               "the sealed-eight generation campaigns read their bias-corrected accelerated bounds "
               "against, which is why those bounds are read as optimistic rather than exact. It is "
               "a simulated coverage of a percentile interval and not a measured coverage of a BCa "
               "interval, and no campaign resamples it",
         family="interval_coverage", support=support, unit=COVERAGE, kind="artifact_leaf",
         value=float(coverage), level="measurement")
    ledger.gap(
        id="interval_coverage/percentile_95/report",
        claim="the realised coverage of a nominal 95% percentile interval, per resampling-unit "
              "count, as a machine-readable record rather than as prose",
        family="interval_coverage",
        reason="no retained artifact and no module constant records the coverage table. The "
               f"resampling module holds it in the declaration comment that justifies the "
               f"{MINIMUM_BOOTSTRAP_UNITS}-unit floor and each campaign holds the one point it "
               "cites in the prose of its interval-coverage note, so the quoted figure is "
               "anchor-bound rather than read from a field, and the coverage at every other unit "
               "count is not quotable at all",
        looked_in=f"{STATISTICS} and " + ", ".join(path for path, _, _ in CAMPAIGNS),
        reachability="not_recorded",
        searched="every numeric field of both campaign records and every declared constant of the "
                 "resampling module")
