"""The guidance and selection tables' cells, as ledger rows rather than as print.

R6 prints three tables of guidance and selection results --- the confirmatory
contrasts, the ProGen3 robustness contrasts and the per-selector captured
fractions --- and a handful of prose numbers beside them. Every printed value is
reproduced exactly from a retained analysis artifact, but only ten of the
fifty-eight estimate cells had a ledger row carrying the (point, interval) pair
the cell prints. The manuscript's own checker therefore could not trace the other
forty-eight: it matched each printed value against whatever unrelated row
happened to share the number and reported the quoted interval as belonging to
none of them. The defect was in the table of derived numbers, not in the
manuscript, and this module closes it.

Nothing here is a second reading of a quantity another module already binds. A
cell whose (artifact, pointer) pair the ledger already carries is left alone, and
the artifact is re-read to check that the carried row still states what the
artifact states: one number under two identifiers is one number neither of them
pins down, and a carried row that has drifted from its own source is a refusal
rather than a silent second row. That rule is what makes a rebuild additive ---
every row this module writes is a pointer no row read before.

The two campaigns are declared, not hard-coded. A campaign is an artifact, a
pointer template for one contrast record, the endpoints it carries with the unit
each is measured in, the strata and contrasts the tables print, and the support
and resampling statement its own module already owns. The record shape --- which
field holds the point, the interval, the status and the reason --- is declared
once and shared, because both campaigns write the same frozen shape. Registering
a further experiment is adding one such declaration and listing it in
``CAMPAIGNS``; nothing else in this module changes, and every refusal below then
applies to it.

Three rosters carry what the artifacts do not name. ``STRATA`` and ``CONTRASTS``
per campaign are the cells the tables print; the artifacts carry more --- the
cross-domain coupling control on both endpoints of the confirmatory campaign, and
the fields-only arm's two contrasts on the robustness campaign --- and those are
deliberately not emitted, because a row for a number the manuscript never writes
can only take an attribution away from one it does. ``ARMS`` and ``SELECTORS``
are the selection table's pool bases and scorer roster.

The selection table reads ``mean_captured_fraction`` and nothing else. The
per-selector blocks carry no ``captured_fraction`` field at all --- that name
belongs to the per-domain records one level down --- so a block that grows one is
a refusal here: a reduction read from a per-domain field would be a per-domain
number presented as a mean over the sealed eight.

Only a cell the artifact records as estimated over the whole domain set is
emitted. A cell the frozen rule reports as unestimable prints an em dash in every
table, so there is no printed number to trace, and the modules that own those
contrasts already record the bound's absence as a gap. A cell that is neither of
those two things is a refusal rather than a quieter row.

One condition these rows legitimately trip is declared here with them. A
contrast row prints one contrast on the two co-primary endpoints, which the
ledger holds on two supports, so once both cells resolve the per-row
one-support-per-comparison condition fires on every such row. The table's own
spanning column headings already name the two apart, and the condition does not
read column groups, so the policy exception that covers those rows is declared
beside the rows it covers and its anchors are checked against the committed
tables whenever the rows are written. This is narrow by construction: the
anchors are the contrast labels, so a row the tables grow under a new label is
not covered until that label is declared.

Four readings outside the three tables are bound here because the same gap hid
them. The declared bias-corrected accelerated draw count is quoted as a numeral
in four table captions and once in the body, and existed in the ledger only as
the ``resampling_draws`` field of other rows, which the traceability condition
resolves numbers against quantities and so can never reach; it is given the shape
every other declared draw count in this table has, read from the interval
statement of both campaigns and refusing if they disagree. The pooled
fields-plus-couplings contrast's two Holm-adjusted p-values are quoted in the
Supplementary text and are recorded scalars with no interval beside them. The
unconditioned ProtGPT2 selection sentence quotes a composition entropy and its
size-matched random reference. And the generation-stream sensitivity quotes what
replacing the fragment reference with a whole natural record is worth, which is a
mean over the gate's twenty conditions and is derived here from the two control
rates each condition records.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import re

from src.capability.core.evidence_ledger import LedgerError, resolve
# The emit contract --- how a row carries its locator, digest and pointer --- is
# the follow-up module's. A second copy here would be a second place for it to
# drift from the ledger's requirements.
from scripts.capability.reporting.followup_quantities import _emitter
# The two campaigns' own modules own their schema checks, their population and
# endpoint prose, their support descriptions and their interval statements. This
# module reads the cells those modules do not bind, so it borrows all of that
# rather than restating it: a second copy of a support description is a second
# place for the population behind a row to drift.
from scripts.capability.reporting import coupling_guidance_quantities as e20
from scripts.capability.reporting import progen3_robustness_quantities as progen3


# --------------------------------------------------------------------------- #
# The record shape both campaigns write, and the declaration of a campaign.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RecordShape:
    """Which field of one record holds what.

    Declared once because both R6 campaigns write the same frozen shape. An
    experiment that writes another shape declares its own instead of this one.
    """

    point: str = "mean"
    interval: str = "interval95_bca"
    status: str = "status"
    reason: str = "reason"
    domains: str = "n_domains"
    missing: str = "missing_domains"
    estimated: str = "estimated"
    unestimable: str = "unestimable"


FROZEN = RecordShape()


@dataclass(frozen=True)
class Campaign:
    """One contrast table: where its cells live, and what each one means.

    ``record`` is the pointer to one contrast record, written with the
    placeholders ``{endpoint}``, ``{stratum}`` and ``{contrast}``. ``endpoints``
    maps each endpoint block the tables print onto the unit it is measured in and
    the name the tables print it under. What the endpoint measures is stated in
    full on the support and deliberately not repeated in every claim: a claim
    carrying the population's generic vocabulary wins attributions it has no
    business winning, and one such collision was observed before these claims
    were narrowed to the contrast and its endpoint's name --- a measured
    contrast of 0.0007 took the search E-value threshold the Methods writes as
    ten to the minus three, because both round onto 0.001 and the claim shared
    the word "reference" with that sentence. ``strata`` and ``contrasts`` are
    the cells the tables print, which no artifact field declares. ``read``,
    ``resampling``, ``support`` and ``domains`` are the campaign module's own
    checked payload, interval statement, support declaration and domain roster,
    so this module adds no second statement of any of them.
    """

    name: str
    path: str
    prefix: str
    family: str
    record: tuple[str, ...]
    endpoints: Mapping[str, tuple[str, str]]
    strata: tuple[str, ...]
    contrasts: tuple[str, ...]
    read: Callable[[object], dict]
    resampling: Callable[[dict], tuple[int, int]]
    support: Callable[[object, dict, str, str], str]
    domains: Callable[[dict], Sequence[str]]
    shape: RecordShape = field(default=FROZEN)

    def locator(self, endpoint: str, stratum: str, contrast: str) -> tuple[str, ...]:
        return tuple(segment.format(endpoint=endpoint, stratum=stratum, contrast=contrast)
                     for segment in self.record)


#: What each printed contrast differences, as ``(what it is worth, the arms)``.
#: The three the equivalence readings already quote come from the confirmatory
#: module, so the same contrast cannot be described two ways; the two the tables
#: print and that module does not quote are declared here.
CONTRASTS = {
    **e20.CONTRAST_PROSE,
    "Pr-N": ("independent-site profile guidance is worth over unguided decoding", "Pr minus N"),
    "C-N": ("Potts-coupling guidance is worth over unguided decoding", "C minus N"),
}

TM = e20.TM
FRACTION = e20.FRACTION
BCA = e20.BCA
DOMAIN_UNIT = e20.DOMAIN_UNIT
DOMAINS = "sealed evaluation domains"
PROBABILITY = "dimensionless probability"
DRAWS = "bootstrap draws"
NATS = "nats"
RATE_DIFFERENCE = "dimensionless rate difference"


def _confirmatory_support(ledger, payload: dict, endpoint: str, stratum: str) -> str:
    if endpoint == "primary_tm":
        return e20._primary_support(ledger, payload, stratum)
    return e20._feasibility_support(ledger, payload, stratum)


def _robustness_support(ledger, payload: dict, endpoint: str, stratum: str) -> str:
    rungs = progen3._stratum_rungs(stratum)
    if endpoint == progen3.PRIMARY:
        return progen3._primary_support(ledger, payload, stratum, rungs)
    # The novelty-admissibility endpoint carries the same population as the
    # primary one and a different endpoint, and the report states both the
    # definition and the warning that it is not a structural-confidence rate.
    # Both are read rather than restated, so the support cannot describe an
    # endpoint the report has stopped describing that way.
    block = payload["bounds"][endpoint]
    return ledger.declare_support(
        f"progen3_robustness/{endpoint}/{stratum}",
        f"{progen3._population(payload, progen3._stratum_cells(rungs), rungs)}; the endpoint is "
        f"{block['definition']} {block['note']} No verdict and no pass/fail flag is issued on this "
        "campaign")


#: The two campaigns whose contrast tables R6 prints.
CAMPAIGNS = (
    Campaign(
        name="confirmatory campaign",
        path=e20.CONFIRMATORY,
        prefix="coupling_guidance",
        family="coupling_guidance",
        record=("{endpoint}", "{stratum}", "contrasts", "{contrast}"),
        endpoints={"primary_tm": (TM, "mean TM-score"),
                   "coprimary_feasibility": (FRACTION, "novelty-admissibility rate")},
        strata=(e20.POOLED, *e20.MODELS),
        contrasts=("Pr-N", "C-N", "C-Pr", "F-M", "C-Cx"),
        read=lambda ledger: e20._confirmatory(ledger),
        resampling=e20._resampling,
        support=_confirmatory_support,
        domains=lambda payload: payload["domains"],
    ),
    Campaign(
        name="ProGen3 robustness campaign",
        path=progen3.ROBUSTNESS,
        prefix="progen3_robustness",
        family="progen3_robustness",
        record=("bounds", "{endpoint}", "strata", "{stratum}", "{contrast}"),
        endpoints={"primary_tm": (TM, "mean TM-score"),
                   "novelty_admissibility_rate": (FRACTION, "novelty-admissibility rate")},
        strata=(progen3.POOLED, *progen3.RUNGS),
        contrasts=("Pr-N", "C-Pr", "C-Cx"),
        read=lambda ledger: progen3._robustness(ledger),
        resampling=progen3._resampling,
        support=_robustness_support,
        domains=lambda payload: payload["design"]["domains"],
    ),
)


# --------------------------------------------------------------------------- #
# The selection table.
# --------------------------------------------------------------------------- #

#: The pool bases the selection table prints, with the prose name and the
#: identifier segment each one carries. The coupling-guided pool is printed as
#: five em dashes: it loses a domain to the admissible floor, so every captured
#: fraction in it is unestimable and no cell carries a number.
ARMS = (("N", "unguided", "unguided_pool"), ("Pr", "profile-guided", "profile_guided_pool"))
#: The admissible scorer roster, with the prose name each row is printed under.
#: The TM oracle is deliberately outside it: it ranks by the measured endpoint,
#: which is prohibited for any method.
SELECTORS = (
    ("potts_full", "a full Potts score, fields and couplings together"),
    ("potts_fields", "the fields block of a Potts score alone"),
    ("potts_coupling", "the coupling block of a Potts score alone"),
    ("coupling_permuted", "a coupling block with its columns permuted, the incorrect-constraint "
                          "scorer control"),
    ("native_loglik", "native likelihood, which is the model's own likelihood of the "
                      "candidate"),
)
CAPTURE = "captured_fraction_of_oracle_headroom"
#: The field the per-selector mean lives in. The per-domain records one level
#: down carry a ``captured_fraction``, and a mean read from that name would be a
#: per-domain number reported as a reduction over the sealed eight.
CAPTURED = "mean_captured_fraction"
PER_DOMAIN_ONLY = "captured_fraction"


# --------------------------------------------------------------------------- #
# The readings outside the three tables.
# --------------------------------------------------------------------------- #

#: How each campaign records the interval every copied bound was computed under,
#: and the anchor that binds the draw count inside it. The statement is prose, so
#: the count is bound by an anchor matching exactly once, following the
#: interval-coverage precedent; both campaigns state it and a disagreement is a
#: refusal rather than a choice between two readings.
INTERVAL_STATEMENTS = (
    (e20.CONFIRMATORY, ("frozen", "interval")),
    (progen3.ROBUSTNESS, ("discipline", "interval")),
)
DRAWS_ANCHOR = re.compile(r"95% BCa bootstrap over domains, (\d+) resamples")

#: The contrast whose two Holm-adjusted p-values the Supplementary text quotes,
#: and the fields they are recorded in.
HOLM_CONTRAST = ("primary_tm", e20.POOLED, "F-M")
HOLM_TESTS = (
    ("p_holm_signflip", "exact two-sided sign-flip",
     "the exact two-sided sign-flip test over domains"),
    ("p_holm_ttest", "two-sided paired t-test", "the two-sided paired t-test over domains"),
)

#: The unconditioned ProtGPT2 pool the Supplementary selection sentence reads,
#: and the one curve point it quotes: the twelve candidates the model's own
#: likelihood ranks highest.
E17_GAP = e20.E17_GAP
E17_CURVE = ("e17", "arms", e20.E17_ARM, "evaluators", e20.E17_EVALUATOR, "curves", "likelihood")
ENTROPY = "mean_composition_entropy_nats"

#: The generation-control gate, and the two references whose complete-domain
#: rates the generation-stream sensitivity differences across every condition.
GATE = "results/R6/generative_control_20260924/gate_endpoints.json"
GATE_SCHEMA = "gate_generative_control"
GATE_SUPPORT = "generator_gate_cells"
GATE_ENDPOINT = "complete_domain"
GATE_REFERENCES = ("natural", "fragment")


def add_r6_guidance_table_quantities(ledger) -> None:
    emit = _emitter(ledger)
    carried = _carried(ledger)
    _cross_support_anchors(ledger)
    for campaign in CAMPAIGNS:
        _contrast_cells(ledger, emit, carried, campaign)
    _selection_cells(ledger, emit, carried)
    _resampling_draws(ledger, emit)
    _holm_p_values(ledger, emit)
    _selected_set_entropy(ledger, emit)
    _reference_rate_sensitivity(ledger, emit)


# --------------------------------------------------------------------------- #
# What the ledger already carries, and why that is checked rather than assumed.
# --------------------------------------------------------------------------- #

def _carried(ledger) -> dict[tuple[str, tuple[str, ...]], object]:
    """Every (artifact, pointer) pair a row already reads, keyed for lookup.

    A cell another module binds is left alone: a second row for one number is a
    number neither row pins down, and it would also make a rebuild of this table
    change rows this module does not own. The map is built once, before anything
    is emitted, so a pair this module writes cannot later look carried.
    """
    return {(quantity.source_path, quantity.source_pointer): quantity
            for quantity in ledger.quantities}


def _agrees_with_source(quantity, value: float, interval) -> None:
    """A carried row still states what its own artifact states, or a refusal.

    Skipping a carried pointer is only safe while the carried row is the same
    reading. A row whose value or interval has drifted from the artifact it
    points at is a defect in the row, and emitting a second row beside it would
    bury that rather than report it.
    """
    stored = None if quantity.interval is None else tuple(float(end) for end in quantity.interval)
    quoted = None if interval is None else tuple(float(end) for end in interval)
    if float(quantity.value) == float(value) and stored == quoted:
        return
    raise LedgerError(
        f"{quantity.id}: the row carries {quantity.value} on {stored} and "
        f"{quantity.source_path}/{'/'.join(quantity.source_pointer)} now states {value} on "
        f"{quoted}, so the carried row is no longer a reading of its own source and skipping it "
        "would hide that")


def _required_support(ledger, support_id: str, owner: str) -> str:
    """A support another module declares, used rather than restated.

    A row of this module that sits on the same measured population as a row of
    another must carry the same support identifier, or the two would read as a
    comparison across supports in the table row that prints them side by side.
    The description therefore has one owner and this module refuses if that owner
    has not run.
    """
    if support_id not in ledger.supports:
        raise LedgerError(
            f"support {support_id!r} is declared by {owner} and is not in the ledger, so this "
            "module is running before it; the rows it adds sit on populations that module owns")
    return support_id


# --------------------------------------------------------------------------- #
# The contrast tables.
# --------------------------------------------------------------------------- #

def _record(campaign: Campaign, payload: dict, endpoint: str, stratum: str,
            contrast: str) -> tuple[dict, bool]:
    """One declared cell, and whether it carries a quotable estimate.

    Both outcomes the frozen rule issues are admissible and nothing else is. An
    estimated cell has to cover the whole domain set, because a reduction over a
    set is only statable when the artifact covers the set. An unestimable one has
    to carry the reason it has none and no interval beside it, because a record
    holding both would leave the rule's own outcome ambiguous; it prints an em
    dash and is not emitted.
    """
    shape = campaign.shape
    locator = campaign.locator(endpoint, stratum, contrast)
    try:
        record = resolve(payload, locator)
    except KeyError:
        raise LedgerError(
            f"{campaign.path}: the declared cell {'/'.join(locator)} does not resolve, so the "
            f"{campaign.name}'s declared strata and contrasts are not the ones it records")
    status = record[shape.status]
    if status == shape.unestimable:
        if record[shape.interval] is not None or not record[shape.reason]:
            raise LedgerError(
                f"{campaign.path}: {endpoint} {stratum} {contrast} reports itself {status!r} and "
                f"carries interval {record[shape.interval]} with reason {record[shape.reason]!r}, "
                "so the frozen rule's own outcome for it is ambiguous")
        return record, False
    if status != shape.estimated or record[shape.interval] is None:
        raise LedgerError(
            f"{campaign.path}: {endpoint} {stratum} {contrast} is {status!r} with interval "
            f"{record[shape.interval]}, which is neither estimated nor unestimable")
    expected = len(campaign.domains(payload))
    if record[shape.domains] != expected or record[shape.missing]:
        raise LedgerError(
            f"{campaign.path}: {endpoint} {stratum} {contrast} covers {record[shape.domains]} of "
            f"the {expected} sealed domains (without a value: {sorted(record[shape.missing])}), so "
            "it is not a contrast over the sealed set")
    return record, True


def _contrast_cells(ledger, emit, carried: dict, campaign: Campaign) -> None:
    payload = campaign.read(ledger)
    draws, seed = campaign.resampling(payload)
    shape = campaign.shape
    for endpoint, (unit, endpoint_name) in sorted(campaign.endpoints.items()):
        for stratum in campaign.strata:
            support = campaign.support(ledger, payload, endpoint, stratum)
            for contrast in campaign.contrasts:
                record, estimated = _record(campaign, payload, endpoint, stratum, contrast)
                locator = campaign.locator(endpoint, stratum, contrast) + (shape.point,)
                existing = carried.get((campaign.path, locator))
                if existing is not None:
                    _agrees_with_source(existing, record[shape.point],
                                        record[shape.interval] if estimated else None)
                    continue
                if not estimated:
                    continue
                worth, arms = CONTRASTS[contrast]
                emit(campaign.path, locator,
                     identifier=f"{campaign.prefix}/{endpoint}/{stratum}/{contrast}/difference",
                     claim=f"difference in {endpoint_name}: what {worth} ({arms}) on {stratum} "
                           f"in the {campaign.name}",
                     family=campaign.family, support=support, unit=unit, kind="estimate",
                     interval=record[shape.interval], interval_kind=BCA,
                     resampling_unit=DOMAIN_UNIT, draws=draws, seed=(seed,), level="generation")


# --------------------------------------------------------------------------- #
# The selection table: what each scorer reaches of the oracle's headroom.
# --------------------------------------------------------------------------- #

def _selector(payload: dict, arm: str, selector: str) -> dict:
    """One scorer's captured fraction in one pool, read from the right field."""
    block = e20._capture_arm(payload, arm)["selectors"][selector]
    if PER_DOMAIN_ONLY in block:
        raise LedgerError(
            f"{e20.CONFIRMATORY}: the {arm} pool's {selector} block now carries a "
            f"{PER_DOMAIN_ONLY!r} field beside {CAPTURED!r}; that name belongs to the per-domain "
            "records, and a mean read from it would be one domain's fraction reported as the mean "
            "over the sealed eight")
    if block["status"] != "estimated" or block["interval95_bca"] is None:
        raise LedgerError(
            f"{e20.CONFIRMATORY}: the {arm} pool's {selector} captured fraction is "
            f"{block['status']!r} with interval {block['interval95_bca']}, so it carries no "
            "quotable estimate; the table prints an em dash for such a cell")
    return block


def _selection_cells(ledger, emit, carried: dict) -> None:
    payload = e20._confirmatory(ledger)
    draws, seed = e20._resampling(payload)
    budget = payload[CAPTURE]["k"]
    support = _required_support(ledger, "e20_selection_headroom",
                                "coupling_guidance_quantities")
    declared = {name for name, _ in SELECTORS}
    if declared != set(e20.SELECTORS):
        raise LedgerError(
            f"the selection table's scorer roster {sorted(declared)} is not the admissible roster "
            f"{sorted(e20.SELECTORS)} the confirmatory module reads, so the two modules describe "
            "different sets of rows")
    for arm, pool, segment in ARMS:
        headroom = e20._capture_arm(payload, arm)["mean_oracle_gain"]
        for selector, scorer in SELECTORS:
            block = _selector(payload, arm, selector)
            locator = (CAPTURE, "arms", arm, "selectors", selector, CAPTURED)
            existing = carried.get((e20.CONFIRMATORY, locator))
            if existing is not None:
                _agrees_with_source(existing, block[CAPTURED], block["interval95_bca"])
                continue
            loss = (" It is negative, so this scorer ranks the pool worse than not ranking it at "
                    "all." if block[CAPTURED] < 0.0 else "")
            emit(e20.CONFIRMATORY, locator,
                 identifier=f"coupling_selection/{segment}/{selector}/captured_fraction",
                 claim=f"fraction of the selection oracle's headroom that ranking by {scorer} "
                       f"captures in {pool} pools, at a budget of {budget} candidates: the mean "
                       f"endpoint of the candidates it ranks highest inside a cell's admissible "
                       f"pool, minus that pool's own mean, over the headroom of {headroom} the "
                       f"oracle has.{loss} The oracle ranks by the measured endpoint, which no "
                       "method may read, so it is not one of the scorers",
                 family="coupling_selection", support=support, unit=FRACTION, kind="estimate",
                 interval=block["interval95_bca"], interval_kind=BCA, resampling_unit=DOMAIN_UNIT,
                 draws=draws, seed=(seed,), level="generation")


# --------------------------------------------------------------------------- #
# The declared draw count the captions quote.
# --------------------------------------------------------------------------- #

def _resampling_draws(ledger, emit) -> None:
    """The bias-corrected accelerated draw count, as a quantity rather than a field.

    Both campaigns record it only inside the prose of their interval statement,
    so it is bound by an anchor matching exactly once in each, and both have to
    give the same count: a constant recorded twice is one declaration only while
    the recordings agree. The row has the shape every other declared draw count
    in this table has, which is what makes the numeral in a caption resolvable at
    all --- the traceability condition resolves numbers against quantities, so a
    count carried only as another row's ``resampling_draws`` can never trace.
    """
    readings = {}
    for path, pointer in INTERVAL_STATEMENTS:
        statement = str(resolve(ledger.artifacts.json(path), pointer))
        found = DRAWS_ANCHOR.findall(statement)
        if len(found) != 1:
            raise LedgerError(
                f"{path}: the declared anchor does not match exactly once in the interval "
                f"statement {statement!r}, so the draw count cannot be bound to one occurrence "
                "of it")
        readings[path] = int(found[0])
    if len(set(readings.values())) != 1:
        raise LedgerError(
            "the R6 campaigns state different bias-corrected accelerated draw counts "
            f"({readings}), so there is no one declared count the captions quote")
    path, pointer = INTERVAL_STATEMENTS[0]
    others = "; ".join(f"{other} (sha256 {ledger.artifacts.sha256(other)})"
                       for other, _ in INTERVAL_STATEMENTS[1:])
    emit(path, pointer + ("<anchor>", DRAWS_ANCHOR.pattern),
         identifier="coupling_guidance/bca_resampling_draws",
         claim="declared resampling draw count: every bias-corrected accelerated interval of both "
               "R6 generation campaigns is computed from this many bootstrap draws over the sealed "
               "evaluation domains, which is the figure the guidance and selection table captions "
               f"quote. Both campaigns state it in their own interval statement and agree; {others} "
               "carries the second statement",
         family="coupling_guidance",
         support=_required_support(ledger, f"e20_primary_tm/{e20.POOLED}",
                                   "coupling_guidance_quantities"),
         unit=DRAWS, kind="constant", value=readings[path], level="generation")


# --------------------------------------------------------------------------- #
# The two Holm-adjusted p-values the Supplementary text quotes.
# --------------------------------------------------------------------------- #

def _holm_p_values(ledger, emit) -> None:
    payload = e20._confirmatory(ledger)
    endpoint, stratum, contrast = HOLM_CONTRAST
    record = payload[endpoint][stratum]["contrasts"][contrast]
    worth, arms = CONTRASTS[contrast]
    size = payload[endpoint][stratum]["holm_family_size"]
    support = _required_support(ledger, f"e20_primary_tm/{stratum}",
                                "coupling_guidance_quantities")
    for field_name, test, population in HOLM_TESTS:
        value = record[field_name]
        if value is None:
            raise LedgerError(
                f"{e20.CONFIRMATORY}: {endpoint} {stratum} {contrast} carries no {field_name}, so "
                "the adjusted p-value the Supplementary text quotes for it is not recorded")
        emit(e20.CONFIRMATORY, (endpoint, stratum, "contrasts", contrast, field_name),
             identifier=f"coupling_guidance/{endpoint}/{stratum}/{contrast}/{field_name}",
             claim=f"Holm-adjusted two-sided p-value from {population}, for what {worth} "
                   f"({arms}) on {stratum} in mean TM-score; adjusted within the frozen family of "
                   f"{size} contrasts, in which an unestimable contrast counts as p = 1 so that "
                   "the family size is fixed. The declared decision rests on the interval and the "
                   f"margin and not on this test, and the {test} is reported beside the other "
                   "rather than in place of it",
             family="coupling_guidance", support=support, unit=PROBABILITY, kind="artifact_leaf",
             reason="a recorded adjusted p-value; the campaign resamples the contrast and not the "
                    "p-value, so no interval sits beside it",
             level="generation")


# --------------------------------------------------------------------------- #
# The unconditioned pool's composition entropy, and its random reference.
# --------------------------------------------------------------------------- #

def _curve_point(ledger) -> tuple[int, dict]:
    """The one likelihood-curve point whose budget is the twelve R6 quotes."""
    curve = resolve(ledger.artifacts.json(E17_GAP), E17_CURVE)
    found = [(index, point) for index, point in enumerate(curve)
             if point["n_selected"] == e20.E17_BUDGET]
    if len(found) != 1:
        raise LedgerError(f"{E17_GAP}: {len(found)} points of the likelihood curve select "
                          f"{e20.E17_BUDGET} candidates, so the quoted budget is not one reading")
    return found[0]


def _selected_set_entropy(ledger, emit) -> None:
    index, point = _curve_point(ledger)
    reference = point["size_matched_random_reference"][ENTROPY]
    keys = point["size_matched_random_reference"]["n_keys"]
    base = E17_CURVE + (f"[{index}]",)
    support = ledger.declare_support(
        "e17_protgpt2_likelihood_selected_twelve",
        f"The {point['n_selected']} products of the ProtGPT2 unconditioned pool that the model's "
        f"own likelihood ranks highest, the {point['fraction']} selection fraction of its "
        f"{point['n_pool']} retained products, against a reference of {keys} random draws of the "
        "same size from the same pool. Size-matched, because every repertoire measure falls as a "
        "set shrinks, so a selected set read against the whole pool would show a collapse at every "
        "small fraction whatever the selector did. The endpoint is the mean over a set's sequences "
        "of the Shannon entropy of each sequence's amino-acid composition, in nats: it describes "
        "the diversity of residues within one sequence and not the diversity between candidates")
    emit(E17_GAP, base + ("selected_set", ENTROPY),
         identifier=f"likelihood_anti_selection/{e20.E17_ARM}/composition_entropy/"
                    "top_twelve_by_likelihood",
         claim=f"mean amino-acid composition entropy of the {point['n_selected']} "
               f"{e20.E17_ARM} candidates its own likelihood ranks highest, in nats per sequence; "
               "a diversity axis read beside the predicted-confidence yield, and a within-sequence "
               "measure rather than a measure of diversity between candidates",
         family="likelihood_anti_selection", support=support, unit=NATS, kind="artifact_leaf",
         reason="a recorded profile of one selected set; the reference draws are resampled and the "
                "selected set's own profile is not",
         level="generation")
    emit(E17_GAP, base + ("size_matched_random_reference", ENTROPY, "mean"),
         identifier=f"likelihood_anti_selection/{e20.E17_ARM}/composition_entropy/"
                    "size_matched_random_reference",
         claim=f"mean amino-acid composition entropy of size-matched random draws of "
               f"{point['n_selected']} candidates from the same {e20.E17_ARM} unconditioned pool, "
               "in nats per sequence: the diversity a set of that size has without any selector, "
               "which is what the selected set's entropy is read against",
         family="likelihood_anti_selection", support=support, unit=NATS, kind="estimate",
         interval=reference["interval"], interval_kind="95% Student-t",
         resampling_unit="size-matched random draw", draws=keys, level="generation")


# --------------------------------------------------------------------------- #
# What a whole natural record is worth over a fragment reference.
# --------------------------------------------------------------------------- #

def _reference_rate_sensitivity(ledger, emit) -> None:
    """The mean the generation-stream sensitivity quotes, over the gate's conditions.

    The gate records each reference's own complete-domain rate per condition and
    no mean over conditions, so the sentence's number is a reduction this module
    performs rather than a field it reads. Every condition has to carry both
    references, because a mean over the conditions that happen to carry one is a
    mean over another set.
    """
    payload = ledger.artifacts.json(GATE)
    if payload["schema"] != GATE_SCHEMA:
        raise LedgerError(f"{GATE}: not a {GATE_SCHEMA} record")
    cells = payload["cells"]
    rates: dict[str, list[float]] = {name: [] for name in GATE_REFERENCES}
    for cell in cells:
        controls = cell["endpoints"][GATE_ENDPOINT]["controls"]
        missing = [name for name in GATE_REFERENCES if name not in controls]
        if missing:
            raise LedgerError(
                f"{GATE}: the {cell['cell']} condition carries no {', '.join(missing)} reference on "
                f"the {GATE_ENDPOINT} endpoint, so a mean over the conditions that carry both "
                "would be a mean over another set")
        for name in GATE_REFERENCES:
            rates[name].append(float(controls[name]["control_rate"]))
    whole, fragment = (sum(rates[name]) / len(cells) for name in GATE_REFERENCES)
    emit(GATE, ("<mean over the conditions>", "cells", "<each>", "endpoints", GATE_ENDPOINT,
                "controls", "<%s minus %s>" % GATE_REFERENCES, "control_rate"),
         identifier="generative_control/complete_domain/whole_record_over_fragment_reference",
         claim=f"what replacing the fragment reference with a whole natural record from the "
               f"corresponding length band is worth on the complete-domain reference rate, as the "
               f"mean over the {len(cells)} generation conditions the gate measures: the whole "
               f"natural record reaches {whole} of its attempts and the fragment reference "
               f"{fragment}. It is a property of the references the generators are matched "
               "against and not of any generator, and it is a descriptive sensitivity that pools "
               "no evidence with the generation-stream analysis",
         family="generative_control",
         support=_required_support(ledger, GATE_SUPPORT, "build_derived_numbers"),
         unit=RATE_DIFFERENCE, kind="estimate", value=whole - fragment,
         reason="the gate resamples each condition's paired model-minus-reference difference and "
                "records no interval for a reference rate or for a mean over the conditions, so "
                "this difference of means carries none",
         level="generation")


# --------------------------------------------------------------------------- #
# The one condition these rows legitimately trip, and the exception that covers it.
# --------------------------------------------------------------------------- #

#: The two committed tables whose contrast rows quote both co-primary endpoints.
GUIDANCE_TABLES = ("manuscript/submission/tables/table-1-guidance-selection.tex",
                   "manuscript/submission/tables/supplementary-guidance-selection.tex")
#: The row labels the cross-support exception is anchored on, one per contrast the
#: tables print. Binding the exception to the labels rather than to a table or a
#: file is what keeps it narrow: it covers exactly the rows that quote one
#: contrast on the two endpoints, and a row the tables grow under a new label is
#: not covered until that label is declared here.
CROSS_SUPPORT_ANCHORS = ("Profile $-$ unguided &", "Couplings $-$ unguided &",
                         "Couplings $-$ profile &", "Fields + couplings $-$ fields &",
                         "Couplings $-$ permuted &")
CROSS_SUPPORT_REASON = (
    "the two cells of a guidance-contrast row are one contrast on the two co-primary endpoints, "
    "and the table's own spanning column headings name them apart: 'Difference in mean TM-score' "
    "against 'Difference in novelty-admissibility rate'. The condition is evaluated per row and "
    "does not read those column groups, so the comparison is declared by the table and the row is "
    "exempt; it is not a licence for any other pair of supports, and the anchors are checked "
    "against the committed tables where the rows are emitted")


def cross_support_exceptions() -> list[dict]:
    """The checker-policy entries that cover these rows, declared once here.

    The policy file is what the checker reads, and this is where the anchors are
    decided; a test holds the two equal so the committed policy cannot widen or
    lose an exception without the declaration moving with it.
    """
    return [{"anchor": anchor, "reason": CROSS_SUPPORT_REASON}
            for anchor in CROSS_SUPPORT_ANCHORS]


def _cross_support_anchors(ledger) -> None:
    """Every declared anchor still binds exactly one row of each guidance table.

    The rows this module emits are what make those table cells resolvable, and
    so they are also what makes the per-row support condition fire on a table
    whose column groups already declare the comparison. The exception is
    therefore checked where the rows are written: an anchor that binds no row has
    outlived the label it was written for, and one that binds a row twice, or a
    row that binds two anchors, no longer names one row and would exempt more
    than the sentence it was written for.
    """
    for path in GUIDANCE_TABLES:
        text = ledger.artifacts.text(path)
        rows = [row for row in text.split("\\\\") if "&" in row]
        for anchor in CROSS_SUPPORT_ANCHORS:
            holding = [row for row in rows if anchor in row]
            if not holding:
                raise LedgerError(
                    f"{path}: the declared cross-support anchor {anchor!r} binds no row of this "
                    "table, so the exception has outlived the contrast label it was written for")
            if text.count(anchor) != len(holding):
                raise LedgerError(
                    f"{path}: the cross-support anchor {anchor!r} occurs {text.count(anchor)} "
                    f"times across {len(holding)} rows, so it binds a row more than once and no "
                    "longer names one row")
        for row in rows:
            held = [anchor for anchor in CROSS_SUPPORT_ANCHORS if anchor in row]
            if len(held) > 1:
                raise LedgerError(
                    f"{path}: one row binds the cross-support anchors {held}, so neither names one "
                    "row and the exception would cover more than the row it was written for")
