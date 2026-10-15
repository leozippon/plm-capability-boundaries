"""Source mappings for the ProGen3 cross-model readings the guidance section closes on.

The guidance section reports the pre-registered E20 confirmatory analysis on two
checkpoints, and then asks whether the same readings hold on a third family. The
ProGen3 robustness campaign answers that: the same frozen analysis module, the
same eight sealed domains and the same admissibility rule, run on a 3B and a 112M
ProGen3 checkpoint at doses matched to the confirmatory intervention measured as
per-residue divergence, with a five-arm design that has no fields-plus-couplings
cell. Three readings of it are quoted --- what first-order guidance is worth over
unguided decoding, the bound on what couplings add over the profile, and how far
a float16 kernel moves the primary endpoint --- and every row here is one of
those, read out of the one retained analysis.

The campaign issues no verdict and no pass/fail flag, and reproduces E20's
declared margin only as a reference scale so that a bound can be read against
something. No row here carries a verdict, each bound says in its own claim that
nothing is tested against that margin, and the margin itself is not re-emitted:
the confirmatory family already carries the one declared margin, and a second row
for it on this campaign would read as a decision rule this campaign does not
apply. A report that began issuing a verdict would make those claims false, so it
is a refusal.

Precision is part of the native contract, so the campaign's measure lines are
precisions and no stratum pools across one. The strata are therefore the three
rungs plus the pooled bfloat16 stratum, and every row is keyed to the stratum it
is read from: the three quoted readings are the pooled bfloat16 ones and the
precision control's, never a rung's. ``RUNGS``, ``POOLED_RUNGS`` and ``ARMS``
carry what the artifact lists but does not declare, and the pooled stratum has to
pool exactly the two bfloat16 rungs, because "across 3B and 112M checkpoints" is
that and nothing else.

The precision control is its own section of the report rather than a fourth
stratum of the primary one. It re-runs one checkpoint in float16 at the doses
solved for it in bfloat16, so the kernel precision is the only thing that moves
between the two rungs; the row is a maximum over the five arms of that contrast's
achieved upper bound, which is what "shifts the primary endpoint by at most"
states. Identical doses on the two rungs is the condition that makes it a
precision control at all, so it is read rather than assumed.

The campaign's own procedural constants are not emitted: the bootstrap draw count
and seed, the permutation seed, the candidates per cell, the record counts and
the per-arm doses. No claim quotes them as a numeral and each collides with
numbers the manuscript writes for other things, so a row for any of them could
only dilute the attribution of a number it does quote. They travel on the support
descriptions instead. The novelty-admissibility and selection readings of this
campaign are not emitted either: the section states their direction and quotes no
value for either, and the paragraph's "in all eight domains" does not name which
of the two guided-against-unguided contrasts it is over, so a row keyed to one of
them would assert a reading the sentence does not pin down.
"""
from __future__ import annotations

from src.capability.core.evidence_ledger import LedgerError
# The emit contract --- how a row carries its locator, digest and pointer --- is
# the follow-up module's. The interval convention is E20's: this report is
# produced by the frozen confirmatory module, so its interval statement, interval
# kind, resampling unit and endpoint scale are read from the one place that
# already owns them rather than restated here.
from scripts.capability.reporting.followup_quantities import _emitter
from scripts.capability.reporting.coupling_guidance_quantities import (
    BCA, DOMAIN_UNIT, INTERVAL_STATEMENT, TM)

ROBUSTNESS = "results/R6/progen3_robustness_analysis_20261015/robustness.json"
SCHEMA = "progen3_robustness_bounds_v1"

#: The five arms this design runs. Fields and couplings together is absent, which
#: is why E20's F-M contrast cannot be formed here, and the arm roster is what the
#: precision control's maximum is taken over.
ARMS = ("C", "Cx", "M", "N", "Pr")
#: The three rungs the campaign measures: a checkpoint at a precision. No artifact
#: field declares the roster, and a rung missing from it would make the strata a
#: different set.
RUNGS = ("progen3-112m-bfloat16", "progen3-112m-float16", "progen3-3b-bfloat16")
#: The pooled stratum, and the two rungs it pools. "Across 3B and 112M
#: checkpoints" is exactly this pair on one measure line.
POOLED = "pooled_bfloat16"
POOLED_RUNGS = ("progen3-112m-bfloat16", "progen3-3b-bfloat16")
MEASURE_LINE = "bfloat16"
#: The one checkpoint the precision control covers. float16 yields no finite
#: forward on the larger one, which the section reports as a scope limit.
PRECISION_CHECKPOINT = "progen3-112m"

PRIMARY = "primary_tm"
#: The endpoints the report bounds. Both carry strata, so both are checked
#: against the declared roster even though only the primary one is quoted.
ENDPOINTS = ("novelty_admissibility_rate", "primary_tm")


def add_progen3_robustness_quantities(ledger) -> None:
    emit = _emitter(ledger)
    payload = _robustness(ledger)
    _profile_over_unguided(ledger, emit, payload)
    _coupling_over_profile(ledger, emit, payload)
    _precision_control(ledger, emit, payload)


# --------------------------------------------------------------------------- #
# The campaign, and the population every row sits on.
# --------------------------------------------------------------------------- #

def _robustness(ledger) -> dict:
    """The retained analysis, with the design and the strata it declares."""
    payload = ledger.artifacts.json(ROBUSTNESS)
    if payload["schema"] != SCHEMA:
        raise LedgerError(f"{ROBUSTNESS}: not a {SCHEMA} report")
    if payload["issues_a_verdict"] or payload["issues_a_pass_fail_flag"]:
        raise LedgerError(
            f"{ROBUSTNESS}: the report now issues a verdict or a pass/fail flag, so these rows "
            "would carry a decision the manuscript reports this campaign as not making")
    design = payload["design"]
    if not design["matches_sealed_eight"]:
        raise LedgerError(f"{ROBUSTNESS}: the analysed domains are not the sealed evaluation eight, "
                          "so a reduction over them is a reduction over another set")
    if tuple(sorted(design["arms"])) != tuple(sorted(ARMS)):
        raise LedgerError(f"{ROBUSTNESS}: the design runs arms {sorted(design['arms'])} and not the "
                          f"declared {sorted(ARMS)}")
    if tuple(sorted(design["rungs"])) != tuple(sorted(RUNGS)):
        raise LedgerError(f"{ROBUSTNESS}: the design runs rungs {sorted(design['rungs'])} and not "
                          f"the declared {sorted(RUNGS)}")
    declared = set(RUNGS) | {POOLED}
    for endpoint in ENDPOINTS:
        strata = payload["bounds"][endpoint]["strata"]
        if set(strata) != declared:
            raise LedgerError(f"{ROBUSTNESS}: {endpoint} carries strata {sorted(strata)} and not "
                              f"the declared {sorted(declared)}")
    line = design["measure_lines"][MEASURE_LINE]
    if line["pooled_stratum"] != POOLED or tuple(sorted(line["rungs"])) != \
            tuple(sorted(POOLED_RUNGS)):
        raise LedgerError(
            f"{ROBUSTNESS}: the {MEASURE_LINE} measure line pools {sorted(line['rungs'])} into "
            f"{line['pooled_stratum']!r} and not the declared {sorted(POOLED_RUNGS)} into "
            f"{POOLED!r}, so the pooled reading is not the one across both checkpoints")
    return payload


def _resampling(payload: dict) -> tuple[int, int]:
    """The draw count and seed every copied interval was computed under."""
    statement = payload["discipline"]["interval"]
    match = INTERVAL_STATEMENT.match(statement)
    if not match:
        raise LedgerError(f"{ROBUSTNESS}: the interval statement {statement!r} is not the BCa "
                          "bootstrap over domains these rows copy, so its draw count and seed "
                          "cannot be carried")
    return int(match.group(1)), int(match.group(2))


def _population(payload: dict, cells: str, rungs: tuple[str, ...]) -> str:
    """The shared half of every support description: domains, cells, admission.

    The candidate count and the dose rule are read rather than restated, and each
    has to be one value across the rungs a row rests on: a pool size that differed
    between two rungs, or two calibrations deciding the dose on different
    statistics, would make one support description cover two populations.
    """
    design, discipline = payload["design"], payload["discipline"]
    counts = {count for rung in rungs for count in design["rungs"][rung]["candidates_per_cell"]}
    if len(counts) != 1:
        raise LedgerError(f"{ROBUSTNESS}: {sorted(rungs)} are run at candidate counts "
                          f"{sorted(counts)}, so there is no one pool size behind these rows")
    statistics = {entry["decision_statistic"] for entry
                  in payload["declared_context_dose_floor"]["calibrations"].values()}
    if len(statistics) != 1:
        raise LedgerError(f"{ROBUSTNESS}: the calibrations decide the dose on {sorted(statistics)}, "
                          "so there is no one dose rule these rows are matched under")
    return (f"{len(design['domains'])} sealed evaluation domains held out from all development "
            f"({', '.join(design['domains'])}), {cells} at {counts.pop()} candidates per cell; "
            f"{discipline['admissibility']}; the independence unit is the "
            f"{discipline['independence_unit']}, a contrast below "
            f"{discipline['minimum_bootstrap_units']} domains carries no interval, the interval is "
            f"{discipline['interval']} and {discipline['interval_coverage_note']} "
            f"Doses are matched to the confirmatory intervention by the {statistics.pop()}")


def _primary_support(ledger, payload: dict, stratum: str, rungs: tuple[str, ...]) -> str:
    cells = f"each domain's value the mean over the cells of {' and '.join(sorted(rungs))}"
    return ledger.declare_support(
        f"progen3_robustness/primary_tm/{stratum}",
        f"{_population(payload, cells, rungs)}; the endpoint is "
        f"{payload['bounds'][PRIMARY]['definition']}. No verdict and no pass/fail flag is issued on "
        "this campaign")


def _covered(payload: dict, stratum: str, contrast: str) -> dict:
    """One contrast estimated over every sealed domain, or a refusal."""
    record = payload["bounds"][PRIMARY]["strata"][stratum][contrast]
    if record["status"] != "estimated" or record["interval95_bca"] is None:
        raise LedgerError(f"{ROBUSTNESS}: {PRIMARY} {stratum} {contrast} is {record['status']} "
                          f"({record['reason']}), so it carries no quotable estimate")
    expected = len(payload["design"]["domains"])
    if record["n_domains"] != expected or record["missing_domains"]:
        raise LedgerError(
            f"{ROBUSTNESS}: {PRIMARY} {stratum} {contrast} covers {record['n_domains']} of the "
            f"{expected} sealed domains (without a value: {sorted(record['missing_domains'])}), so "
            "it is not a contrast over the sealed eight")
    return record


# --------------------------------------------------------------------------- #
# What first-order guidance is worth on a third model family.
# --------------------------------------------------------------------------- #

def _profile_over_unguided(ledger, emit, payload: dict) -> None:
    support = _primary_support(ledger, payload, POOLED, POOLED_RUNGS)
    record = _covered(payload, POOLED, "Pr-N")
    draws, seed = _resampling(payload)
    base = ("bounds", PRIMARY, "strata", POOLED, "Pr-N")
    emit(ROBUSTNESS, base + ("mean",),
         identifier=f"progen3_robustness/primary_tm/{POOLED}/profile_over_unguided",
         claim="mean TM-score of independent-site profile guidance minus unguided decoding "
               "(Pr minus N), pooled over the two bfloat16 ProGen3 rungs at doses matched to the "
               "confirmatory intervention: what first-order guidance is worth on recapitulation on "
               "a third model family",
         family="progen3_robustness", support=support, unit=TM, kind="estimate",
         interval=record["interval95_bca"], interval_kind=BCA, resampling_unit=DOMAIN_UNIT,
         draws=draws, seed=(seed,), level="generation")
    favouring = record["favouring"]
    if sum(favouring.values()) != record["n_domains"]:
        raise LedgerError(f"{ROBUSTNESS}: the pooled Pr-N sign counts {favouring} do not add to "
                          f"the {record['n_domains']} domains the contrast covers")
    emit(ROBUSTNESS, base + ("favouring", "positive"),
         identifier=f"progen3_robustness/primary_tm/{POOLED}/profile_over_unguided/"
                    "domains_favouring_profile",
         claim="sealed evaluation domains in which profile guidance raises the pooled ProGen3 mean "
               "TM-score above unguided decoding, out of the eight the contrast covers",
         family="progen3_robustness", support=support, unit="sealed evaluation domains",
         kind="count", level="generation")


# --------------------------------------------------------------------------- #
# The bound on what couplings add over the profile, against no margin.
# --------------------------------------------------------------------------- #

def _agreeing_bound(payload: dict, stratum: str, contrast: str) -> tuple[tuple, float]:
    """One achieved upper bound, read in both places the artifact records it.

    The headline block and the contrast's own interval are one reading only while
    they agree, so a disagreement refuses rather than picking one. The headline
    block is the authoritative locator, because that is what the report presents
    as the bound.
    """
    record = _covered(payload, stratum, contrast)
    headline = payload["achieved_upper_bounds"][PRIMARY][contrast][stratum]
    own = record["interval95_bca"][1]
    if headline != own:
        raise LedgerError(
            f"{ROBUSTNESS}: {stratum} {contrast} records different achieved upper bounds "
            f"({headline}, {own}), so there is no one bound to quote")
    return ("achieved_upper_bounds", PRIMARY, contrast, stratum), float(headline)


def _coupling_over_profile(ledger, emit, payload: dict) -> None:
    locator, bound = _agreeing_bound(payload, POOLED, "C-Pr")
    emit(ROBUSTNESS, locator,
         identifier=f"progen3_robustness/primary_tm/{POOLED}/C-Pr/achieved_upper_bound",
         claim="achieved upper 95% bound on what coupling guidance is worth over the first-order "
               "profile, in mean TM-score (C minus Pr), pooled over the two bfloat16 ProGen3 rungs; "
               "it is the upper end of that contrast's own interval and not a point estimate, and "
               f"it is read against no margin: this campaign issues no verdict, and the "
               f"{payload['reference_scale']['margin_tm']} TM margin the confirmatory campaign "
               "declared is reproduced in its record as a reference scale only, with nothing "
               "tested against it",
         family="progen3_robustness", support=_primary_support(ledger, payload, POOLED,
                                                               POOLED_RUNGS),
         unit=TM, kind="range_bound", value=bound, level="generation")


# --------------------------------------------------------------------------- #
# The precision control: what the float16 kernel moves.
# --------------------------------------------------------------------------- #

def _precision_block(payload: dict) -> dict:
    """The one precision contrast, and everything that makes it one.

    A precision control is only a precision control while the two rungs are the
    same checkpoint on two measure lines at identical doses: that is what leaves
    the kernel precision as the only thing that moves. Each of those is read, and
    the arm roster the maximum is taken over has to be the declared five.
    """
    checkpoints = payload["precision_control"]["checkpoints"]
    if set(checkpoints) != {PRECISION_CHECKPOINT}:
        raise LedgerError(
            f"{ROBUSTNESS}: the precision control covers {sorted(checkpoints)} and not the one "
            f"declared checkpoint {PRECISION_CHECKPOINT!r}")
    block = checkpoints[PRECISION_CHECKPOINT]
    rungs, design = (block["float16_rung"], block["bfloat16_rung"]), payload["design"]["rungs"]
    for rung in rungs:
        if rung not in design or design[rung]["base_checkpoint"] != PRECISION_CHECKPOINT:
            raise LedgerError(f"{ROBUSTNESS}: the precision control reads {rung!r}, which is not a "
                              f"rung of {PRECISION_CHECKPOINT}")
    if block["contrast"] != f"{rungs[0]} - {rungs[1]}":
        raise LedgerError(f"{ROBUSTNESS}: the precision contrast is {block['contrast']!r} and not "
                          f"the float16 rung minus the bfloat16 rung")
    if design[rungs[0]]["measure_line"] == design[rungs[1]]["measure_line"]:
        raise LedgerError(f"{ROBUSTNESS}: both precision rungs sit on the "
                          f"{design[rungs[0]]['measure_line']!r} measure line, so the contrast is "
                          "not between two precisions")
    doses = [design[rung]["doses"] for rung in rungs] + [block["doses"]]
    if any(dose != doses[0] for dose in doses[1:]):
        raise LedgerError(
            f"{ROBUSTNESS}: the two precision rungs are run at different doses ({doses}), so the "
            "kernel precision is not the only thing that moves between them and the contrast is "
            "not a precision control")
    endpoint = block["endpoints"][PRIMARY]
    if tuple(sorted(endpoint["achieved_upper_bound"])) != tuple(sorted(ARMS)):
        raise LedgerError(
            f"{ROBUSTNESS}: the precision control bounds arms "
            f"{sorted(endpoint['achieved_upper_bound'])} and not the declared {sorted(ARMS)}; a "
            "maximum over a different roster is a different claim")
    expected = len(payload["design"]["domains"])
    for arm in ARMS:
        record = endpoint["by_arm"][arm]
        if record["status"] != "estimated" or record["interval95_bca"] is None:
            raise LedgerError(f"{ROBUSTNESS}: the precision control's {arm} arm is "
                              f"{record['status']} ({record['reason']}), so a maximum over the "
                              "roster would be a maximum over the rest of it")
        if record["n_domains"] != expected or record["missing_domains"]:
            raise LedgerError(
                f"{ROBUSTNESS}: the precision control's {arm} arm covers {record['n_domains']} of "
                f"the {expected} sealed domains (without a value: "
                f"{sorted(record['missing_domains'])})")
        if endpoint["achieved_upper_bound"][arm] != record["interval95_bca"][1]:
            raise LedgerError(
                f"{ROBUSTNESS}: the precision control's {arm} arm records different achieved upper "
                f"bounds ({endpoint['achieved_upper_bound'][arm]}, {record['interval95_bca'][1]}), "
                "so there is no one bound to reduce over")
    return block


def _precision_control(ledger, emit, payload: dict) -> None:
    block = _precision_block(payload)
    bounds = block["endpoints"][PRIMARY]["achieved_upper_bound"]
    widest = max(ARMS, key=lambda arm: bounds[arm])
    rungs = (block["float16_rung"], block["bfloat16_rung"])
    cells = ("each domain's value the paired difference between the "
             f"{rungs[0]} and {rungs[1]} cells")
    support = ledger.declare_support(
        f"progen3_robustness/precision_control/{PRECISION_CHECKPOINT}",
        f"{_population(payload, cells, rungs)}; the contrast is "
        f"{block['contrast']}, per arm, over the same domains and at the same doses, so the kernel "
        f"precision is the only thing that moves between the two rungs; the endpoint is "
        f"{payload['bounds'][PRIMARY]['definition']}. No verdict and no pass/fail flag is issued on "
        "this campaign")
    emit(ROBUSTNESS,
         ("precision_control", "checkpoints", PRECISION_CHECKPOINT, "endpoints", PRIMARY,
          "achieved_upper_bound", "<largest over the arms>"),
         identifier=f"progen3_robustness/precision_control/{PRECISION_CHECKPOINT}/primary_tm/"
                    "largest_endpoint_shift",
         claim="largest achieved upper 95% bound, over the five arms this design runs, on the "
               "change a float16 kernel makes to the primary TM endpoint: the 112M checkpoint "
               "re-run in float16 at the doses solved for it in bfloat16, so precision is the only "
               f"thing that moves ({widest} is the widest arm). It bounds how far reduced-precision "
               "arithmetic could move the result, is a maximum over an arm roster rather than a "
               "point estimate, and carries no verdict",
         family="progen3_robustness", support=support, unit=TM, kind="range_bound",
         value=bounds[widest], level="generation")
