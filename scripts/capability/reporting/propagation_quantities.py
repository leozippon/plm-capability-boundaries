"""Source mappings for the position-propagation wave the rewritten R2 rests on.

Every row here is read out of a retained artifact of the E01 propagation cells,
the E02 contact-anticipation summary or the E19 forcing gate. Nothing is fitted
and no interval is resampled: an interval is copied with the resampling unit and
draw count the artifact records beside it, and the one derived quantity --- the
effect the simultaneous band has 80% power against --- is the band's own
standard error and critical value under this package's declared power
convention, with the arithmetic stated in the claim.

Two declared rosters carry what the artifacts do not. ``PROTEIN_ONLY`` names the
checkpoints trained on protein sequence alone, which no artifact field records
and which the manuscript's "in the protein-only arms" selects; an artifact that
does not cover it refuses rather than reporting a narrower panel as the whole
one. ``UNREADABLE_DECAY`` names the three arms whose log-linear decay fit the
project record does not read a half-distance from, and the refusal that travels
with it is the one that matters: every admitted arm's fit at the shared scope
has to explain more of its profile's variance than every arm left out, so the
roster is the well-fitted end of the ordering and not a hand-picked list. That
roster is fixed at the shared scope and carried to the wider one, because the
two windows are quoted against each other and a set that changed between them
would make the comparison a different one.

The procedural constants these cells record --- draw counts, seeds, the family
floor, the permutation draw count --- are not emitted. No claim quotes them, and
each collides with numbers the manuscript writes for other things, so a row for
any of them could only dilute the attribution of a number it does quote. The
band's critical value is emitted because the supplement quotes it and the
detectable-effect reduction is computed from it.
"""
from __future__ import annotations

from statistics import NormalDist

from src.capability.core.evidence_ledger import LedgerError
from src.capability.forcing.forcing_design import POWER
# The emit contract --- how a row carries its locator, digest and pointer --- is
# the follow-up module's. A second copy here would be a second place for it to
# drift from the ledger's requirements.
from scripts.capability.reporting.followup_quantities import _emitter

CAUSAL512 = ("results/audit_20261012/structure-recovery/final/e01_final/"
             "posrespf_propagation_causal_cpu/position_propagation.json")
SHARED128 = ("results/audit_20261012/structure-recovery/final/e01_bidir4/"
             "posresph_propagation_shared128_cpu/position_propagation.json")
ANTICIPATION = ("results/audit_20261012/structure-recovery/final/e01_bidir4/"
                "posresph_anticipation_cpu/contact_anticipation.json")
FORCING = "results/R6/forcing_balanced_20261011/forcing_gate.json"
RECEIVER_RSA = "results/R6/e01_receiver_rsa_20261012/position_propagation.json"

#: The checkpoints trained on protein sequence alone. No artifact field carries
#: this: a paradigm is ``causal_next_token`` for a protein generator and for a
#: general-text model alike, and ProLLaMA and InstructProtein are protein-
#: adapted text models rather than protein-only ones. The manuscript's
#: "protein-only arms" is this set, so it is declared once and every reduction
#: over it refuses if the artifact does not cover it.
PROTEIN_ONLY = ("progen2-base", "progen2-large", "progen2-medium", "progen2-small",
                "progen2-xlarge", "progen3-112m", "proteinglm-7b-clm", "protgpt2", "rita-xl")

#: The arms whose distance decay is measured but not read as a half-distance:
#: ProGen3-112M and Galactica-125M rest on poor log-linear fits, and ProtGPT2's
#: segmentation resolves single-residue tokens so rarely that its fit interval
#: spans [13.9, 254.9] residues. :func:`_readable_decay_arms` refuses unless
#: every admitted arm fits better than every arm named here, so this is the
#: ordering's well-fitted end rather than an assertion.
UNREADABLE_DECAY = ("galactica-125m", "progen3-112m", "protgpt2")

#: The separations the downstream profile is quoted at: immediately after the
#: substitution, and far enough away to show the decay.
QUOTED_SEPARATIONS = (1, 20)

#: The anchor burial strata the anticipation contrast is split by.
BURIAL_STRATA = ("buried", "exposed")


def add_propagation_quantities(ledger) -> None:
    emit = _emitter(ledger)
    _causal_identifiability(ledger, emit)
    _downstream_profile(ledger, emit)
    _contact_band(ledger, emit)
    _structural_support(ledger, emit)
    _receiver_burial_band(ledger, emit)
    _contact_anticipation(ledger, emit)
    _forcing_gate(ledger, emit)


# --------------------------------------------------------------------------- #
# Shared readers over the two propagation cells.
# --------------------------------------------------------------------------- #

def _arms(ledger, path: str) -> dict[str, dict]:
    payload = ledger.artifacts.json(path)
    arms = {arm["arm"]: arm for arm in payload["arms"]}
    if len(arms) != len(payload["arms"]):
        raise LedgerError(f"{path}: the arm roster repeats a checkpoint")
    missing = [arm for arm in PROTEIN_ONLY if arm not in arms]
    if missing:
        raise LedgerError(f"{path}: the declared protein-only roster is not covered; "
                          f"missing {missing}")
    return arms


def _cell_support(ledger, path: str, support_id: str, window: str) -> str:
    """Declare one propagation cell's support from the cell's own counts."""
    payload = ledger.artifacts.json(path)
    arms = payload["arms"]
    assays = sorted({len(arm["assays"]) for arm in arms})
    if len(assays) != 1:
        raise LedgerError(f"{path}: the cell's arms do not share one assay roster: {assays}")
    substitutions = sum(entry["selected_single_substitutions"]
                        for arm in arms for entry in arm["assays"])
    return ledger.declare_support(
        support_id,
        f"{len(arms)} position-resolved checkpoints over {assays[0]} ProteinGym assays each, "
        f"{window}; {len(arms) * assays[0]} arm--assay rows and {substitutions} grid-aligned "
        "single substitutions; a receiver is one (substitution, position) pair and the two cells "
        "do not nest")


def _readable_decay_arms(ledger) -> tuple[str, ...]:
    """The protein-only arms whose decay fit is read as a half-distance.

    The roster is fixed at the shared scope, where the two windows are compared,
    and the refusal is what makes it a reading rather than a choice: the worst
    admitted fit has to explain more of its profile than the best excluded one.
    """
    arms = _arms(ledger, SHARED128)
    admitted = tuple(arm for arm in PROTEIN_ONLY if arm not in UNREADABLE_DECAY)
    excluded = tuple(arm for arm in PROTEIN_ONLY if arm in UNREADABLE_DECAY)
    if not admitted or not excluded:
        raise LedgerError(f"{SHARED128}: the decay roster needs both an admitted and an "
                          "excluded protein-only arm to be a separation at all")

    def fit(arm: str) -> float:
        decay = arms[arm]["profile"]["downstream"]["decay"]
        if not decay["fitted"] or decay["half_distance_undefined_reason"]:
            raise LedgerError(f"{SHARED128}: {arm} carries no fitted downstream decay")
        return float(decay["r_squared"])

    worst, best = min(fit(arm) for arm in admitted), max(fit(arm) for arm in excluded)
    if not worst > best:
        raise LedgerError(
            f"{SHARED128}: the declared decay roster is no longer the well-fitted arms: the worst "
            f"admitted fit explains {worst:.4f} of its profile and the best excluded one "
            f"{best:.4f}; the half-distance range would be a hand-picked subset")
    return admitted


def _span(emit, path: str, pointer: tuple[str, ...], *, prefix: str, claim: str, family: str,
          support: str, unit: str, values: dict[str, float], level: str = "likelihood") -> None:
    """One recorded span: the smallest and largest member of a named set."""
    for bound, chosen in (("min", min(values, key=values.get)),
                          ("max", max(values, key=values.get))):
        emit(path, pointer + (f"<{bound}>",), identifier=f"{prefix}/{bound}",
             claim=f"{claim} ({bound}, {chosen})", family=family, support=support, unit=unit,
             kind="range_bound", value=values[chosen], level=level)


# --------------------------------------------------------------------------- #
# E01 --- the one-sided causal profile, measured rather than assumed.
# --------------------------------------------------------------------------- #

def _causal_identifiability(ledger, emit) -> None:
    arms = _arms(ledger, CAUSAL512)
    support = _cell_support(ledger, CAUSAL512, "e01_causal512", "no length cap")
    paradigms = sorted({arm["paradigm"] for arm in arms.values()})
    if paradigms != ["causal_next_token"]:
        raise LedgerError(f"{CAUSAL512}: this cell is the causal panel and carries {paradigms}")
    exact = [name for name, arm in sorted(arms.items()) if arm["upstream_is_exactly_zero"]]
    other = [name for name, arm in sorted(arms.items()) if not arm["upstream_is_exactly_zero"]]
    if len(other) != 1:
        raise LedgerError(
            f"{CAUSAL512}: the quoted claim is one exception among the causal arms, and this cell "
            f"carries {len(other)}: {other}")
    emit(CAUSAL512, ("arms", "<count>"), identifier="position_propagation/causal512/causal_arms",
         claim="causal checkpoints the unconstrained-length propagation cell analyses",
         family="position_propagation", support=support, unit="checkpoints",
         kind="support_count", value=len(arms))
    emit(CAUSAL512, ("arms", "<exactly zero upstream>"),
         identifier="position_propagation/causal512/arms_with_exactly_zero_upstream",
         claim="causal checkpoints whose response at every position before the substitution is "
               "exactly zero, which causal masking makes a property of the model class rather "
               "than a null result",
         family="position_propagation", support=support, unit="checkpoints", kind="count",
         value=len(exact))
    emit(CAUSAL512, ("arms", "<upstream receivers on the exactly-zero arms>"),
         identifier="position_propagation/causal512/exactly_zero_upstream_receivers",
         claim="receiving positions over which the upstream contribution is exactly zero, summed "
               "over the causal arms that carry an exact zero",
         family="position_propagation", support=support, unit="receiving positions", kind="census",
         value=sum(arms[name]["profile"]["upstream"]["receivers"] for name in exact))
    exception = other[0]
    emit(CAUSAL512, ("arms", f"<{exception}>", "upstream_max_abs_nats"),
         identifier=f"position_propagation/causal512/{exception}/upstream_max_abs_nats",
         claim=f"largest absolute upstream contribution on {exception}, the one causal arm whose "
               "upstream response is not exactly zero; its own repeat maximum, which is admitted "
               "arithmetic rather than a response",
         family="position_propagation", support=support, unit="nats", kind="artifact_leaf",
         value=arms[exception]["upstream_max_abs_nats"], level="likelihood",
         reason="a recorded worst-case magnitude over receivers, not a resampled estimate")


def _downstream_profile(ledger, emit) -> None:
    """The response's magnitude near the substitution, and its half-distance."""
    readable = _readable_decay_arms(ledger)
    cells = (("causal512", CAUSAL512, "no length cap", "the 512-residue scope"),
             ("shared128", SHARED128, "wild types capped at 128 residues", "the shared "
              "128-residue scope"))
    for key, path, window, scope in cells:
        arms = _arms(ledger, path)
        support = _cell_support(ledger, path, f"e01_{key}", window)
        halves = {}
        for arm in readable:
            decay = arms[arm]["profile"]["downstream"]["decay"]
            if not decay["fitted"] or decay["half_distance_undefined_reason"]:
                raise LedgerError(f"{path}: {arm} carries no fitted downstream half-distance")
            halves[arm] = float(decay["half_distance_residues"])
        _span(emit, path, ("arms", "<protein-only readable fits>", "profile", "downstream",
                           "decay", "half_distance_residues"),
              prefix=f"position_propagation/{key}/protein_only/half_distance",
              claim=f"fitted downstream half-distance across the {len(readable)} protein-only arms "
                    f"whose log-linear fit is read at {scope}",
              family="position_propagation", support=support, unit="residues", values=halves)
        if key != "causal512":
            continue
        for separation in QUOTED_SEPARATIONS:
            responses = {}
            for arm in readable:
                rows = {row["separation"]: row
                        for row in arms[arm]["profile"]["downstream"]["separations"]}
                if separation not in rows or not rows[separation]["sufficient"]:
                    raise LedgerError(f"{path}: {arm} carries no sufficient downstream response at "
                                      f"separation {separation}")
                responses[arm] = float(rows[separation]["mean_absolute_nats"])
            _span(emit, path, ("arms", "<protein-only readable fits>", "profile", "downstream",
                               "separations", f"<separation={separation}>", "mean_absolute_nats"),
                  prefix=f"position_propagation/{key}/protein_only/"
                         f"response_at_separation_{separation}",
                  claim=f"mean absolute downstream response {separation} residue"
                        f"{'' if separation == 1 else 's'} after the substitution, across the "
                        f"{len(readable)} protein-only arms whose decay is read",
                  family="position_propagation", support=support, unit="nats", values=responses)


# --------------------------------------------------------------------------- #
# E01 --- the structural contrast, simultaneously across the panel.
# --------------------------------------------------------------------------- #

def _band(ledger, path: str, outcome: str = "absolute_response") -> dict:
    block = ledger.artifacts.json(path)["panel_simultaneous"][outcome]
    if block["n_columns"] != len(block["columns"]):
        raise LedgerError(f"{path}: the {outcome} family states {block['n_columns']} columns and "
                          f"carries {len(block['columns'])}")
    return block


def _half_width(column: dict) -> float:
    low, high = column["simultaneous_interval_nats"]
    return (high - low) / 2.0


def _contact_band(ledger, emit) -> None:
    arms = _arms(ledger, SHARED128)
    support = _cell_support(ledger, SHARED128, "e01_shared128",
                            "wild types capped at 128 residues")
    block = _band(ledger, SHARED128)
    critical = float(block["critical_value"])
    named = {entry["arm"] for entry in block["excluded_columns"]}
    causal = [column for column in block["columns"]
              if arms[column["arm"]]["paradigm"] == "causal_next_token"]
    protein_only = [column for column in causal if column["arm"] in PROTEIN_ONLY]
    uncovered = [arm for arm in PROTEIN_ONLY
                 if arm not in named and arm not in {column["arm"] for column in protein_only}]
    if uncovered:
        raise LedgerError(f"{SHARED128}: protein-only arms {uncovered} carry neither a column nor "
                          "a named exclusion, so a reduction over the family would be a reduction "
                          "over an unstated subset")
    bidirectional = [column for column in block["columns"]
                     if arms[column["arm"]]["paradigm"] != "causal_next_token"
                     and column["direction"] == "downstream"]
    if not bidirectional:
        raise LedgerError(f"{SHARED128}: the band carries no bidirectional downstream column")

    emit(SHARED128, ("panel_simultaneous", "absolute_response", "<causal columns>"),
         identifier="position_propagation/shared128/contact/causal_columns",
         claim="causal checkpoints carrying an excess-response-at-contacts column in the "
               "simultaneous family; a causal arm's upstream response is zero by construction, so "
               "it contributes one downstream column and never an upstream one",
         family="position_propagation", support=support, unit="checkpoints", kind="support_count",
         value=len(causal))
    emit(SHARED128, ("panel_simultaneous", "absolute_response", "<causal columns excluding zero>"),
         identifier="position_propagation/shared128/contact/causal_columns_resolving",
         claim="causal checkpoints whose excess response at three-dimensional contacts resolves "
               "under the simultaneous band",
         family="position_propagation", support=support, unit="checkpoints", kind="count",
         value=sum(column["simultaneously_excludes_zero"] for column in causal))
    emit(SHARED128, ("panel_simultaneous", "absolute_response", "critical_value"),
         identifier="position_propagation/shared128/contact/critical_value",
         claim="studentised-maximum critical value of the absolute-response family",
         family="position_propagation", support=support, unit="dimensionless", kind="constant",
         value=critical)
    _span(emit, SHARED128, ("panel_simultaneous", "absolute_response", "columns",
                            "<protein-only half-width>"),
          prefix="position_propagation/shared128/contact/protein_only/half_width",
          claim=f"simultaneous half-width of the excess response at contacts across the "
                f"{len(protein_only)} protein-only columns",
          family="position_propagation", support=support, unit="nats",
          values={column["arm"]: _half_width(column) for column in protein_only})
    # The effect the band has 80% power against: the critical value that scales
    # its half-width plus the power quantile, times the column's own standard
    # error. Derived, because a half-width states what the band excludes and not
    # what the design could have found.
    quantile = NormalDist().inv_cdf(POWER)
    _span(emit, SHARED128, ("panel_simultaneous", "absolute_response", "columns",
                            "<protein-only detectable effect>"),
          prefix="position_propagation/shared128/contact/protein_only/detectable_at_80_power",
          claim=f"excess response at contacts detectable at {POWER:.0%} power across the "
                f"{len(protein_only)} protein-only columns, as the critical value plus the power "
                "quantile times each column's own standard error",
          family="position_propagation", support=support, unit="nats",
          values={column["arm"]: (critical + quantile) * column["standard_error_nats"]
                  for column in protein_only})
    _span(emit, SHARED128, ("panel_simultaneous", "absolute_response", "columns",
                            "<bidirectional downstream>"),
          prefix="position_propagation/shared128/contact/bidirectional_downstream",
          claim=f"excess response at contacts downstream across the {len(bidirectional)} "
                "bidirectional arms under sequence-separation matching",
          family="position_propagation", support=support, unit="nats",
          values={column["arm"]: float(column["point_nats"]) for column in bidirectional})


def _structural_support(ledger, emit) -> None:
    """The domains the structural contrast is measured on, and what bounds it."""
    arms = _arms(ledger, SHARED128)
    support = _cell_support(ledger, SHARED128, "e01_shared128",
                            "wild types capped at 128 residues")
    structural = {}
    for arm in arms.values():
        for entry in arm["assays"]:
            if entry["structural_receivers"]:
                structural[entry["assay"]] = entry
    families = {entry["family"] for entry in structural.values()}
    universe = _band(ledger, SHARED128)["family_universe"]
    if len(families) != universe or len(structural) != universe:
        raise LedgerError(
            f"{SHARED128}: the structural assays resolve to {len(structural)} assays in "
            f"{len(families)} families against the band's {universe}-family universe; a length "
            "range over one population and a band over another are not the same support")
    lengths = {assay: entry["residues"] for assay, entry in structural.items()}
    _span(emit, SHARED128, ("arms", "<any>", "assays", "<structure-carrying>", "residues"),
          prefix="position_propagation/shared128/structural_support/domain_residues",
          claim=f"wild-type length across the {universe} domains that carry an experimental "
                "structure", family="position_propagation", support=support, unit="residues",
          values=lengths, level="measurement")
    emit(SHARED128, ("arms", "<any>", "assays", "<structure-carrying>", "<max residues minus 1>"),
         identifier="position_propagation/shared128/structural_support/"
                    "largest_possible_contact_separation",
         claim="largest sequence separation a contact in this support can have, which is one less "
               "than the longest wild type; it is why the contrast says nothing about long-range "
               "contacts in large proteins",
         family="position_propagation", support=support, unit="residues", kind="range_bound",
         value=float(max(lengths.values()) - 1), level="measurement")


def _receiver_burial_band(ledger, emit) -> None:
    """How far the retrospectively declared burial matching moves the causal arms."""
    arms = _arms(ledger, RECEIVER_RSA)
    support = _cell_support(ledger, RECEIVER_RSA, "e01_receiver_rsa_matched",
                            "wild types capped at 128 residues, with the receiver burial band at "
                            "the fixed accessibility boundary appended to the matching key")
    payload = ledger.artifacts.json(RECEIVER_RSA)
    frozen = ledger.artifacts.json(SHARED128)["panel_simultaneous"]
    shifts: dict[str, float] = {}
    for outcome, matched in sorted(payload["panel_simultaneous_receiver_rsa_matched"].items()):
        primary = {(column["arm"], column["direction"]): column
                   for column in payload["panel_simultaneous"][outcome]["columns"]}
        reference = {(column["arm"], column["direction"]): column
                     for column in frozen[outcome]["columns"]}
        if {key: round(column["point_nats"], 12) for key, column in primary.items()} != \
                {key: round(column["point_nats"], 12) for key, column in reference.items()}:
            raise LedgerError(
                f"{RECEIVER_RSA}: the sensitivity's own {outcome} primary block is not the frozen "
                f"panel's, so a matched-minus-primary shift would compare two different panels")
        for column in matched["columns"]:
            key = (column["arm"], column["direction"])
            if column["arm"] not in PROTEIN_ONLY or \
                    arms[column["arm"]]["paradigm"] != "causal_next_token":
                continue
            if key not in primary:
                raise LedgerError(f"{RECEIVER_RSA}: {key} is matched but has no primary column")
            shifts[f"{outcome}/{column['arm']}/{column['direction']}"] = \
                abs(column["point_nats"] - primary[key]["point_nats"])
    if not shifts:
        raise LedgerError(f"{RECEIVER_RSA}: no protein-only causal column is matched")
    largest = max(shifts, key=shifts.get)
    emit(RECEIVER_RSA, ("panel_simultaneous_receiver_rsa_matched", "<largest protein-only shift>"),
         identifier="position_propagation/receiver_rsa/protein_only/largest_estimate_shift",
         claim=f"largest absolute move the receiver-burial-band matching makes in a protein-only "
               f"causal estimate, over both response outcomes ({largest})",
         family="position_propagation", support=support, unit="nats", kind="range_bound",
         value=shifts[largest], level="likelihood")


# --------------------------------------------------------------------------- #
# E02 --- what the conditional already contains about a buried contact.
# --------------------------------------------------------------------------- #

def _contact_anticipation(ledger, emit) -> None:
    payload = ledger.artifacts.json(ANTICIPATION)
    draws = int(payload["settings"]["draws"])
    arm = "progen2-small"
    records = {entry["arm"]: entry["result"] for entry in payload["arms"]}
    if arm not in records or not records[arm]:
        raise LedgerError(f"{ANTICIPATION}: {arm} carries no anticipation result")
    result = records[arm]
    strata = result["sources"]["model_conditional"]["anchor_rsa_strata"]
    missing = [band for band in BURIAL_STRATA if band not in strata]
    if missing:
        raise LedgerError(f"{ANTICIPATION}: {arm} carries no {missing} anchor burial stratum, so "
                          "the contrast cannot be split by burial at all")
    support = ledger.declare_support(
        "e02_anticipation_anchors",
        f"{len(result['assays'])} Tsuboyama domains in {len(result['families'])} families with an "
        f"admitted experimental structure; {result['anchors']} anchors, {result['contact_pairs']} "
        f"contacting and {result['control_pairs']} separation-matched non-contacting pairs; the "
        f"anchor is split at the median partner accessibility and {result['weighting']}")
    for band in BURIAL_STRATA:
        block = strata[band]["contrast_nats"]
        emit(ANTICIPATION, ("arms", f"<{arm}>", "result", "sources", "model_conditional",
                            "anchor_rsa_strata", band, "contrast_nats"),
             identifier=f"contact_anticipation/{arm}/{band}_anchors",
             claim=f"{arm} excess log probability its conditional at an earlier position gives the "
                   f"residue that will contact it, over a non-contacting residue at matched "
                   f"sequence separation, at {band} anchors",
             family="contact_anticipation", support=support, unit="nats", kind="estimate",
             value=block["point"], interval=block["interval"], draws=draws,
             resampling_unit="wild-type family at 50% identity", level="likelihood")


# --------------------------------------------------------------------------- #
# E19 --- the sequence endpoint the forcing gate bounds.
# --------------------------------------------------------------------------- #

def _forcing_gate(ledger, emit) -> None:
    payload = ledger.artifacts.json(FORCING)
    declared = tuple(payload["pre_registration"]["arms"])
    band = payload["panel"]["simultaneous"]
    columns = {column["column"]: column for column in band["columns"]}
    missing = [arm for arm in declared if arm not in columns]
    if missing:
        raise LedgerError(f"{FORCING}: the pre-registered arms {missing} carry no simultaneous "
                          "column, so a panel-wide bound would be a bound over fewer arms")
    thresholds = {arm["verdict"]["pre_registered_claimable_nats"] for arm in payload["arms"]}
    if len(thresholds) != 1:
        raise LedgerError(f"{FORCING}: the arms disagree on the pre-registered claimable effect: "
                          f"{sorted(thresholds)}")
    support = ledger.declare_support(
        "e19_forcing_shared_backbones",
        f"{band['backbones']} backbones shared by all {len(declared)} pre-registered arms, one per "
        f"CATH superfamily, in two length bands weighted equally; {band['estimand']}")
    largest = max(declared, key=lambda arm: abs(columns[arm]["point_nats"]))
    emit(FORCING, ("panel", "simultaneous", "columns", "<largest absolute point>"),
         identifier="forcing_gate/largest_partner_specific_shift",
         claim="largest double difference between what the model emits at a prescribed contacting "
               f"position and at matched non-contacting positions when an early residue is forced "
               f"({largest}); no column resolves above the pre-registered effect",
         family="forcing_gate", support=support, unit="nats", kind="range_bound",
         value=abs(columns[largest]["point_nats"]), level="generation")
    emit(FORCING, ("arms", "<any>", "verdict", "pre_registered_claimable_nats"),
         identifier="forcing_gate/pre_registered_claimable_effect",
         claim="pre-registered smallest claimable partner-specific shift; a measured effect below "
               "it is reported as bounded rather than as a positive",
         family="forcing_gate", support=support, unit="nats", kind="constant",
         value=thresholds.pop(), level="generation")
