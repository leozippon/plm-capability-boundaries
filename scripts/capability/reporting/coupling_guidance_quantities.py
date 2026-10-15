"""Source mappings for the guided-decoding and candidate-selection readings R6 rests on.

R6 reports the pre-registered E20 confirmatory analysis --- the achieved upper
bounds on the two pairwise-specific contrasts and on the permuted-column
control, the first-order gain that does resolve, the co-primary cost of guidance
and how much of the selection oracle's headroom any admissible scorer reaches
--- together with the one E17 reading the section cites as the same
anti-selection on a different endpoint. Every row here is read out of one of
those two retained artifacts. Nothing is fitted, no interval is resampled and no
contrast is recomputed: an interval is copied with the resampling unit, draw
count and seed the artifact's own interval statement records, and the two
derived rows carry the arithmetic their claims state --- the largest captured
fraction over a selector roster, and a count of the domains whose per-domain
difference is positive.

Five readings in this section are bounds rather than estimates, and are emitted
as such. An achieved upper 95% bound is the end of an interval, so it is a
``range_bound`` and not a point estimate the manuscript owes an interval for;
the same holds for the best admissible scorer's captured fraction, which is a
maximum over a roster rather than one resampled quantity. Each achieved bound is
read in every place the artifact records it and refuses on disagreement, because
one number recorded more than once is one reading only while the recordings
agree. The two contrasts the declared equivalence rule decides on are recorded
three times --- the headline block, the stratum's own ``upper_bound`` block and
the primary contrast's interval --- and the permuted-column contrast, which that
rule does not cover, is recorded once in its own interval; no second copy of it
is invented to agree with, and which contrasts the rule decides on is read off
the headline block rather than declared here.

Four rosters carry what the artifacts do not name. ``MODELS`` is the two
checkpoints the design runs; ``QUOTED_BOUNDS`` and ``UNESTIMABLE_BOUNDS`` split
the model-level and pooled equivalence readings into the four the section quotes
as a bound and the three it reports as unestimable; ``SELECTORS`` is the
admissible scorer roster the "best admissible scorer" is a maximum over; and
``QUOTED_POOLS`` names the two pool bases whose headroom the section quotes. A
contrast that does not cover all eight sealed domains, a capture arm with a
missing domain, and a selector roster the artifact does not carry are refusals
rather than quieter numbers: a reduction over a set is only statable when the
artifact covers the set, and "the best admissible scorer" over four of five
selectors is a different claim.

The resampling constants these campaigns record are not emitted: the bootstrap
draw count and seed, the permutation seed and the eight-domain bootstrap floor.
No claim quotes any of them as a numeral, and the floor's value is one the
manuscript does write for a window radius, so a row for any of them could only
dilute the attribution of a number it does quote. They travel on the support
descriptions instead, which is where the population they define belongs.

The three constants the endpoints are defined by are emitted: the identity
restriction, the admissible-candidate floor and the selection budget. The
Methods quotes each as a numeral, and each says what the endpoint counts rather
than how it was resampled --- which candidate enters it, how many a cell needs
before it carries a value at all, and how many a scorer's reading is taken over.
The floor and the budget were suppressed while the Methods wrote them as words,
and the collision that justified suppressing them was checked before they were
emitted: the floor's numeral appears nowhere else in the manuscript, and every
other number the budget's numeral collides with keeps the attribution it had, so
neither row takes a number away from a result that quotes it.
"""
from __future__ import annotations

import re

from src.capability.core.evidence_ledger import LedgerError
# The emit contract --- how a row carries its locator, digest and pointer --- is
# the follow-up module's. A second copy here would be a second place for it to
# drift from the ledger's requirements.
from scripts.capability.reporting.followup_quantities import _emitter

CONFIRMATORY = "results/R6/e20_confirmatory_analysis_20261014/confirmatory.json"
E17_GAP = "results/audit_20261011_h200/e17_gap/generation_evaluation.json"

#: The two checkpoints the confirmatory design runs, and the stratum that
#: averages their cells per domain. No artifact field lists them apart from the
#: stratum keys themselves, and the verdict rule reads each model separately.
MODELS = ("progen2-medium", "protgpt2")
POOLED = "pooled"

#: The achieved upper bounds the section quotes, as (stratum, contrast). Both
#: pairwise-specific contrasts of the pre-registered family are here --- coupling
#: guidance against the first-order profile, and couplings added to fields ---
#: together with the permuted-column control the coupling arm is read against,
#: which the declared equivalence rule does not decide on.
QUOTED_BOUNDS = (("protgpt2", "C-Pr"), ("progen2-medium", "F-M"), ("protgpt2", "F-M"),
                 ("protgpt2", "C-Cx"))
#: The readings the section reports as unestimable rather than as a bound, each
#: resting on seven of the eight domains. Each carries its own refusal: the
#: artifact has to say it is unestimable for exactly one domain, and that
#: domain's reason has to be the admissible-candidate floor, or the sentence
#: describes another failure.
UNESTIMABLE_BOUNDS = (("progen2-medium", "C-Pr"), (POOLED, "C-Cx"),
                      ("progen2-medium", "C-Cx"))

#: The admissible scorer roster. The TM oracle is deliberately outside it --- it
#: ranks by the measured endpoint, which is prohibited for any method --- so
#: "the best admissible scorer" is a maximum over exactly these five.
SELECTORS = ("coupling_permuted", "native_loglik", "potts_coupling", "potts_fields", "potts_full")
#: The pool bases whose oracle headroom the section quotes, with the prose name
#: and the identifier segment each one carries. The coupling-guided pool is
#: measured too and is not quoted: it loses a domain to the admissible floor, so
#: every captured fraction in it sits below the bootstrap floor.
QUOTED_POOLS = (("N", "unguided", "unguided_pool"),
                ("Pr", "profile-guided", "profile_guided_pool"))

#: How the analysis states its interval, read rather than assumed: the draw
#: count and seed exist nowhere else in the artifact, and a statement this
#: pattern cannot read is a statement about another interval.
INTERVAL_STATEMENT = re.compile(r"^95% BCa bootstrap over domains, (\d+) resamples, seed (\d+)$")
BCA = "95% BCa"
DOMAIN_UNIT = "sealed evaluation domain"

TM = "dimensionless TM-score"
FRACTION = "dimensionless fraction"
#: What the floor and the budget count. Both are candidates that already clear
#: the identity restriction: the floor is how many a cell needs to carry a value
#: and the budget is how many a scorer's reading is taken over.
ADMISSIBLE_CANDIDATES = "admissible candidates"
#: Mean per-residue CA pLDDT on the 0--100 scale E17 reports its yields in. It
#: is the folding model's confidence in its own coordinates, so it is neither a
#: percentage nor a measured property, and the unit says so rather than
#: borrowing ``percent``.
PLDDT = "pLDDT points"

#: The one E17 reading R6 cites, and everything about it that has to hold.
E17_ARM = "protgpt2"
E17_EVALUATOR = "mean_ca_plddt"
E17_QUANTITY = "esmfold2_mean_ca_plddt"
E17_SELECTOR = "likelihood"
E17_BUDGET = 12


def add_coupling_guidance_quantities(ledger) -> None:
    emit = _emitter(ledger)
    payload = _confirmatory(ledger)
    _identity_restriction(ledger, emit, payload)
    _admissible_floor(ledger, emit, payload)
    _equivalence_bounds(ledger, emit, payload)
    _primary_tm(ledger, emit, payload)
    _coprimary_feasibility(ledger, emit, payload)
    _selection_budget(ledger, emit, payload)
    _selection_headroom(ledger, emit, payload)
    _likelihood_anti_selection(ledger, emit)


# --------------------------------------------------------------------------- #
# The confirmatory analysis, and the population every row sits on.
# --------------------------------------------------------------------------- #

def _confirmatory(ledger) -> dict:
    """The frozen analysis, with the strata and the domain set it declares."""
    payload = ledger.artifacts.json(CONFIRMATORY)
    if payload["schema"] != "e20_confirmatory_v1":
        raise LedgerError(f"{CONFIRMATORY}: not an e20_confirmatory_v1 analysis")
    if not payload["matches_sealed_eight"]:
        raise LedgerError(f"{CONFIRMATORY}: the analysed domains are not the sealed evaluation "
                          "eight, so a reduction over them is a reduction over another set")
    declared = set(MODELS) | {POOLED}
    for block in ("primary_tm", "coprimary_feasibility"):
        if set(payload[block]) != declared:
            raise LedgerError(f"{CONFIRMATORY}: {block} carries strata {sorted(payload[block])} "
                              f"and not the declared {sorted(declared)}")
    return payload


def _resampling(payload: dict) -> tuple[int, int]:
    """The draw count and seed every copied interval was computed under."""
    statement = payload["frozen"]["interval"]
    match = INTERVAL_STATEMENT.match(statement)
    if not match:
        raise LedgerError(f"{CONFIRMATORY}: the interval statement {statement!r} is not the BCa "
                          "bootstrap over domains these rows copy, so its draw count and seed "
                          "cannot be carried")
    return int(match.group(1)), int(match.group(2))


def _population(payload: dict, stratum: str) -> str:
    """The shared half of every support description: domains, cells, admission."""
    frozen = payload["frozen"]
    cells = ("each domain's value the mean over both models' cells" if stratum == POOLED
             else f"{stratum} cells only")
    return (f"{len(payload['domains'])} sealed evaluation domains held out from all development "
            f"({', '.join(payload['domains'])}), {cells}; a candidate is admissible only strictly "
            f"below {frozen['admissible_identity_below']} percent identity to the wild-type "
            f"reference and a cell needs {frozen['minimum_admissible']} admissible candidates to "
            "carry a value, which is never imputed; the independence unit is the domain, a "
            f"contrast below {payload['minimum_bootstrap_units']} domains carries no interval, and "
            f"the interval is {frozen['interval']}")


def _primary_support(ledger, payload: dict, stratum: str) -> str:
    return ledger.declare_support(
        f"e20_primary_tm/{stratum}",
        f"{_population(payload, stratum)}; the endpoint is the mean TM-score to the wild-type "
        "reference over a cell's admissible candidates")


def _feasibility_support(ledger, payload: dict, stratum: str) -> str:
    return ledger.declare_support(
        f"e20_coprimary_feasibility/{stratum}",
        f"{_population(payload, stratum)}; the endpoint is the admissible fraction of a cell's "
        "pool, which is the share of its candidates below the identity restriction and not a "
        "predicted-confidence rate")


def _selection_support(ledger, payload: dict) -> str:
    capture = payload["captured_fraction_of_oracle_headroom"]
    return ledger.declare_support(
        "e20_selection_headroom",
        f"{_population(payload, POOLED)}; the endpoint is the mean TM-score of the {capture['k']} "
        "candidates a scorer ranks highest inside a cell's admissible pool, minus that pool's own "
        "mean; the oracle ranks by the realised TM-score, which no method may read, so it measures "
        "the ceiling and is not a selector")


# --------------------------------------------------------------------------- #
# The declared constants the endpoints are defined by.
# --------------------------------------------------------------------------- #

def _identity_restriction(ledger, emit, payload: dict) -> None:
    emit(CONFIRMATORY, ("frozen", "admissible_identity_below"),
         identifier="coupling_guidance/admissible_identity_below",
         claim="declared identity restriction: a candidate enters the primary endpoint only "
               "strictly below this percent identity to the wild-type reference, because at "
               "unrestricted identity recapitulation is confounded with sequence recovery and "
               "ranking by identity alone reproduces most of the apparent benefit of guidance",
         family="coupling_guidance", support=_primary_support(ledger, payload, POOLED),
         unit="percent", kind="constant", level="generation")


def _admissible_floor(ledger, emit, payload: dict) -> None:
    emit(CONFIRMATORY, ("frozen", "minimum_admissible"),
         identifier="coupling_guidance/minimum_admissible",
         claim="declared admissible-candidate floor: a cell carries a value on the co-primary "
               "endpoints only with at least this many candidates below the identity restriction, "
               "and a cell short of the floor is reported without a value rather than imputed or "
               "relaxed after the fact; it is the floor one coupling cell misses, which is why "
               "every contrast over that cell rests on seven domains and carries no interval",
         family="coupling_guidance", support=_primary_support(ledger, payload, POOLED),
         unit=ADMISSIBLE_CANDIDATES, kind="constant", level="generation")


def _selection_budget(ledger, emit, payload: dict) -> None:
    """The declared top-k budget, which the analysis records in two places."""
    frozen = payload["frozen"]["selection_k"]
    capture = payload["captured_fraction_of_oracle_headroom"]["k"]
    if frozen != capture:
        raise LedgerError(f"{CONFIRMATORY}: the selection budget is recorded twice and the two "
                          f"disagree ({frozen}, {capture}), so there is no one budget the captured "
                          "fractions and the oracle headroom were both measured at")
    emit(CONFIRMATORY, ("frozen", "selection_k"),
         identifier="coupling_selection/selection_budget",
         claim="declared selection budget: a scorer's reading is the mean endpoint of the top this "
               "many candidates it ranks inside a cell's admissible pool, and the oracle's "
               "headroom is measured at the same budget, so every captured fraction is a fraction "
               "of it",
         family="coupling_selection", support=_selection_support(ledger, payload),
         unit=ADMISSIBLE_CANDIDATES, kind="constant", level="generation")


# --------------------------------------------------------------------------- #
# The headline: achieved upper bounds against the declared margin.
# --------------------------------------------------------------------------- #

def _recorded_bounds(payload: dict, stratum: str,
                     contrast: str) -> list[tuple[tuple, float | None]]:
    """Every place the artifact records this contrast's achieved upper bound.

    The contrast's own interval always carries it. A contrast the declared
    equivalence rule decides on carries it twice more --- in the headline block
    and in the stratum's own ``upper_bound`` block --- because the rule reads
    those; which contrasts those are is the headline block's own keys rather
    than a roster typed in here. The authoritative locator comes first, so a row
    points at the headline block where there is one and at the interval
    otherwise.
    """
    equivalence = payload["equivalence"]
    interval = payload["primary_tm"][stratum]["contrasts"][contrast]["interval95_bca"]
    own = (("primary_tm", stratum, "contrasts", contrast, "interval95_bca", "[1]"),
           None if interval is None else interval[1])
    if contrast not in equivalence["achieved_upper_bound"]:
        return [own]
    return [(("equivalence", "achieved_upper_bound", contrast, stratum),
             equivalence["achieved_upper_bound"][contrast][stratum]),
            (("equivalence", "strata", stratum, "upper_bound", contrast),
             equivalence["strata"][stratum]["upper_bound"][contrast]),
            own]


def _agreeing_bound(payload: dict, stratum: str, contrast: str) -> tuple[tuple, float]:
    """One achieved upper bound, read in every place the artifact records it.

    Several recordings are one reading only while they agree, so a disagreement
    refuses rather than picking one. The contrast has to cover the sealed eight
    for its interval to have an end at all, and a contrast the equivalence rule
    decides on also has to lie below the declared margin; a contrast outside
    that rule is read against no margin, because the margin statement does not
    cover it.
    """
    _covered(payload, "primary_tm", stratum, contrast)
    recorded = _recorded_bounds(payload, stratum, contrast)
    locator, bound = recorded[0]
    if len({value for _, value in recorded}) != 1:
        raise LedgerError(
            f"{CONFIRMATORY}: {stratum} {contrast} records different achieved upper bounds "
            f"({', '.join(str(value) for _, value in recorded)}), so there is no one bound "
            "to quote")
    equivalence = payload["equivalence"]
    if contrast in equivalence["achieved_upper_bound"] and \
            not equivalence["strata"][stratum]["below_margin"][contrast]:
        raise LedgerError(f"{CONFIRMATORY}: {stratum} {contrast} does not lie below the declared "
                          f"margin {equivalence['margin_tm']}, so it is not a bound the margin "
                          "statement covers")
    return locator, float(bound)


def _unestimable_bound(payload: dict, stratum: str, contrast: str) -> None:
    """One reading the section reports as unestimable, and why it has to be.

    No place the artifact records a bound may carry one, and the reason has to
    be the one the prose gives: exactly one domain supplying too few admissible
    candidates.
    """
    if any(value is not None for _, value in _recorded_bounds(payload, stratum, contrast)):
        raise LedgerError(f"{CONFIRMATORY}: {stratum} {contrast} now carries an achieved upper "
                          "bound, so reporting it as unestimable states the wrong result")
    missing = payload["primary_tm"][stratum]["contrasts"][contrast]["missing_domains"]
    if len(missing) != 1 or "admissible" not in next(iter(missing.values())):
        raise LedgerError(
            f"{CONFIRMATORY}: {stratum} {contrast} is unestimable for {sorted(missing)} with "
            f"reasons {sorted(missing.values())}; the quoted claim is one domain supplying too few "
            "admissible candidates, which is a different statement")


def _equivalence_bounds(ledger, emit, payload: dict) -> None:
    prose = {"C-Pr": ("coupling guidance is worth over the first-order profile", "C minus Pr"),
             "F-M": ("adding couplings to the fields of a Potts model is worth", "F minus M"),
             "C-Cx": ("decoding under coupling guidance is worth over decoding under the same "
                      "guidance with its coupling columns permuted", "C minus Cx")}
    for stratum, contrast in QUOTED_BOUNDS:
        locator, bound = _agreeing_bound(payload, stratum, contrast)
        difference, arms = prose[contrast]
        below_zero = ", whose whole interval lies below zero" if bound < 0.0 else ""
        outside_rule = ("" if contrast in payload["equivalence"]["achieved_upper_bound"] else
                        " the contrast is an incorrect-constraint control the declared margin rule "
                        "does not decide on, and")
        emit(CONFIRMATORY, locator,
             identifier=f"coupling_guidance/equivalence/{stratum}/{contrast}/achieved_upper_bound",
             claim=f"achieved upper 95% bound on what {difference}, in mean TM-score "
                   f"({arms}) on {stratum}{below_zero};{outside_rule} it is the "
                   "upper end of that contrast's own interval and not a point estimate",
             family="coupling_guidance", support=_primary_support(ledger, payload, stratum),
             unit=TM, kind="range_bound", value=bound, level="generation")

    for stratum, contrast in UNESTIMABLE_BOUNDS:
        _unestimable_bound(payload, stratum, contrast)

    emit(CONFIRMATORY, ("equivalence", "margin_tm"),
         identifier="coupling_guidance/equivalence/declared_margin",
         claim="declared equivalence margin on the primary TM endpoint: the achieved upper bound "
               "has to lie below this margin in each model separately, and the margin was declared "
               "before any evaluation domain was read",
         family="coupling_guidance", support=_primary_support(ledger, payload, POOLED), unit=TM,
         kind="constant", level="generation")


# --------------------------------------------------------------------------- #
# The primary endpoint: what first-order guidance is worth, and the permuted
# control the coupling scorer is read against.
# --------------------------------------------------------------------------- #

def _covered(payload: dict, block: str, stratum: str, contrast: str) -> dict:
    """One contrast estimated over every sealed domain, or a refusal."""
    record = payload[block][stratum]["contrasts"][contrast]
    if record["status"] != "estimated" or record["interval95_bca"] is None:
        raise LedgerError(f"{CONFIRMATORY}: {block} {stratum} {contrast} is {record['status']} "
                          f"({record['reason']}), so it carries no quotable estimate")
    if record["n_domains"] != len(payload["domains"]) or record["missing_domains"]:
        raise LedgerError(
            f"{CONFIRMATORY}: {block} {stratum} {contrast} covers {record['n_domains']} of the "
            f"{len(payload['domains'])} sealed domains (without a value: "
            f"{sorted(record['missing_domains'])}), so it is not a contrast over the sealed eight")
    return record


def _contrast_row(emit, payload: dict, block: str, stratum: str, contrast: str, *,
                  identifier: str, claim: str, support: str, unit: str) -> dict:
    record = _covered(payload, block, stratum, contrast)
    draws, seed = _resampling(payload)
    emit(CONFIRMATORY, (block, stratum, "contrasts", contrast, "mean"), identifier=identifier,
         claim=claim, family="coupling_guidance", support=support, unit=unit, kind="estimate",
         interval=record["interval95_bca"], interval_kind=BCA, resampling_unit=DOMAIN_UNIT,
         draws=draws, seed=(seed,), level="generation")
    return record


def _primary_tm(ledger, emit, payload: dict) -> None:
    support = _primary_support(ledger, payload, POOLED)
    profile = _contrast_row(
        emit, payload, "primary_tm", POOLED, "Pr-N",
        identifier="coupling_guidance/primary_tm/pooled/profile_over_unguided",
        claim="mean TM-score of independent-site profile guidance minus unguided decoding "
              "(Pr minus N), pooled over the two models: what first-order guidance is worth on "
              "recapitulation",
        support=support, unit=TM)
    favouring = profile["favouring"]
    if sum(favouring.values()) != profile["n_domains"]:
        raise LedgerError(f"{CONFIRMATORY}: the Pr-N sign counts {favouring} do not add to the "
                          f"{profile['n_domains']} domains the contrast covers")
    emit(CONFIRMATORY, ("primary_tm", POOLED, "contrasts", "Pr-N", "favouring", "positive"),
         identifier="coupling_guidance/primary_tm/pooled/profile_over_unguided/"
                    "domains_favouring_profile",
         claim="sealed evaluation domains in which profile guidance raises the mean TM-score above "
               "unguided decoding, out of the eight the contrast covers",
         family="coupling_guidance", support=support, unit="sealed evaluation domains",
         kind="count", level="generation")

    _contrast_row(
        emit, payload, "primary_tm", POOLED, "potts_coupling-coupling_permuted",
        identifier="coupling_guidance/primary_tm/pooled/coupling_selector_over_permuted",
        # The claim says "checkpoints" where its siblings say "models", and
        # keeps the bare word "permuted" rather than only the compounds: this
        # value rounds onto the ``0.004`` a text-and-Galactica Spearman bound is
        # quoted to, and a claim sharing that sentence's generic words wins an
        # attribution it has no business winning. Vocabulary that is specific to
        # the pool and the scorer is what keeps the two apart.
        claim="mean TM-score of the candidates a Potts-coupling scorer ranks highest inside the "
              "unguided pool, minus the candidates its permuted counterpart ranks highest, pooled "
              "over the two checkpoints; a scorer contrast inside one pool rather than the "
              "coupling decoding arm against its permuted-column control (C minus Cx), which is "
              "unestimable pooled",
        support=support, unit=TM)


# --------------------------------------------------------------------------- #
# The co-primary endpoint: the admissible fraction guidance costs.
# --------------------------------------------------------------------------- #

#: The two co-primary reductions the section quotes, each against unguided
#: decoding, with the prose name of the guided arm and the identifier segment.
FEASIBILITY_CONTRASTS = (
    ("Pr-N", "independent-site profile guidance", "Pr minus N", "profile_minus_unguided"),
    ("C-N", "Potts-coupling guidance", "C minus N", "couplings_minus_unguided"),
)


def _coprimary_feasibility(ledger, emit, payload: dict) -> None:
    support = _feasibility_support(ledger, payload, POOLED)
    for contrast, guidance, difference, name in FEASIBILITY_CONTRASTS:
        _contrast_row(
            emit, payload, "coprimary_feasibility", POOLED, contrast,
            identifier=f"coupling_guidance/feasibility/pooled/{name}",
            claim=f"admissible fraction of a candidate pool under {guidance} minus unguided "
                  f"decoding ({difference}), pooled over the two models; the co-primary endpoint "
                  "is the share of candidates below the identity restriction, so the stored "
                  "difference is negative and its magnitude is the admissible fraction guidance "
                  "costs",
            support=support, unit=FRACTION)


# --------------------------------------------------------------------------- #
# What any admissible scorer reaches of the oracle's headroom.
# --------------------------------------------------------------------------- #

def _capture_arm(payload: dict, arm: str) -> dict:
    """One pool basis of the capture block, covering every sealed domain."""
    capture = payload["captured_fraction_of_oracle_headroom"]
    if arm not in capture["arms"]:
        raise LedgerError(f"{CONFIRMATORY}: the capture block carries no {arm} pool")
    block = capture["arms"][arm]
    if block["n_domains"] != len(payload["domains"]) or block["missing_domains"]:
        raise LedgerError(
            f"{CONFIRMATORY}: the {arm} pool covers {block['n_domains']} of the "
            f"{len(payload['domains'])} sealed domains (without a value: "
            f"{sorted(block['missing_domains'])}), so its headroom is not the sealed eight's")
    if set(block["selectors"]) != set(SELECTORS):
        raise LedgerError(
            f"{CONFIRMATORY}: the {arm} pool carries selectors {sorted(block['selectors'])} and "
            f"not the declared {sorted(SELECTORS)}; a maximum over a different roster is a "
            "different claim")
    return block


def _selection_headroom(ledger, emit, payload: dict) -> None:
    support = _selection_support(ledger, payload)
    capture = payload["captured_fraction_of_oracle_headroom"]
    draws, seed = _resampling(payload)
    for arm, prose, key in QUOTED_POOLS:
        _capture_arm(payload, arm)
        emit(CONFIRMATORY,
             ("captured_fraction_of_oracle_headroom", "arms", arm, "mean_oracle_gain"),
             identifier=f"coupling_selection/{key}/oracle_headroom",
             claim="headroom an oracle ranking by the measured endpoint has over the pool mean in "
                   f"{prose} pools, at a budget of {capture['k']} candidates: what the pool "
                   "already contains and no admissible scorer may read",
             family="coupling_selection", support=support, unit=TM, kind="estimate",
             reason="the analysis resamples the captured fraction over the domains and records the "
                    "oracle headroom only as the mean over them, so no interval sits beside it",
             level="generation")

    arm, prose, key = QUOTED_POOLS[-1]
    selectors = _capture_arm(payload, arm)["selectors"]
    unestimable = sorted(name for name, entry in selectors.items()
                         if entry["status"] != "estimated")
    if unestimable:
        raise LedgerError(
            f"{CONFIRMATORY}: the {arm} pool's {unestimable} captured fractions are unestimable, "
            "so a maximum over the admissible roster would be a maximum over the rest of it")
    best = max(SELECTORS, key=lambda name: selectors[name]["mean_captured_fraction"])
    emit(CONFIRMATORY, ("captured_fraction_of_oracle_headroom", "arms", arm, "selectors",
                        "<largest mean captured fraction>"),
         identifier=f"coupling_selection/{key}/best_admissible_selector_captured_fraction",
         claim="largest fraction of the oracle headroom any admissible scorer captures in "
               f"{prose} pools ({best}), over the {len(SELECTORS)} selectors the registry admits; "
               "the oracle itself ranks by the measured endpoint and is not one of them",
         family="coupling_selection", support=support, unit=FRACTION, kind="range_bound",
         value=selectors[best]["mean_captured_fraction"], level="generation")

    native = selectors["native_loglik"]
    if native["mean_captured_fraction"] >= 0.0:
        raise LedgerError(f"{CONFIRMATORY}: native likelihood captures "
                          f"{native['mean_captured_fraction']} of the {arm} pool's headroom, so it "
                          "is no longer the loss the prose quotes as one")
    emit(CONFIRMATORY, ("captured_fraction_of_oracle_headroom", "arms", arm, "selectors",
                        "native_loglik", "mean_captured_fraction"),
         identifier=f"coupling_selection/{key}/native_likelihood_captured_fraction",
         claim=f"fraction of the oracle headroom native-likelihood ranking captures in {prose} "
               "pools; it is negative, so the stored difference is a loss whose magnitude the "
               "prose quotes --- ranking the pool by the model's own likelihood is worse than not "
               "ranking at all",
         family="coupling_selection", support=support, unit=FRACTION, kind="estimate",
         interval=native["interval95_bca"], interval_kind=BCA, resampling_unit=DOMAIN_UNIT,
         draws=draws, seed=(seed,), level="generation")


# --------------------------------------------------------------------------- #
# E17 --- the same anti-selection on a predicted-confidence endpoint.
# --------------------------------------------------------------------------- #

def _e17_budget_point(evaluator: dict) -> tuple[int, dict]:
    """The one decomposition point whose budget is the twelve R6 quotes."""
    gap = evaluator["gap_decomposition"]
    if gap["pool_mean"] != evaluator["pool_mean"]:
        raise LedgerError(f"{E17_GAP}: the {E17_ARM} pool mean is recorded twice and the two "
                          f"disagree ({evaluator['pool_mean']}, {gap['pool_mean']})")
    found = [(index, point) for index, point in enumerate(gap["points"])
             if point["n_selected"] == E17_BUDGET]
    if len(found) != 1:
        raise LedgerError(f"{E17_GAP}: {len(found)} decomposition points select {E17_BUDGET} "
                          "candidates, so the quoted budget is not one reading")
    index, point = found[0]
    if E17_SELECTOR not in point["selectors"]:
        raise LedgerError(f"{E17_GAP}: the {E17_BUDGET}-candidate point carries selectors "
                          f"{sorted(point['selectors'])} and no {E17_SELECTOR} selector")
    return index, point


def _likelihood_anti_selection(ledger, emit) -> None:
    payload = ledger.artifacts.json(E17_GAP)
    arms = payload["e17"]["arms"]
    if E17_ARM not in arms:
        raise LedgerError(f"{E17_GAP}: the evaluation carries no {E17_ARM} arm")
    evaluator = arms[E17_ARM]["evaluators"][E17_EVALUATOR]
    if evaluator["quantity"] != E17_QUANTITY or evaluator["status"] != "measured":
        raise LedgerError(f"{E17_GAP}: the {E17_ARM} {E17_EVALUATOR} evaluator reads "
                          f"{evaluator['quantity']!r} at status {evaluator['status']!r}, not the "
                          "measured predicted-confidence endpoint these rows quote")
    if evaluator["n_dropped"]:
        raise LedgerError(f"{E17_GAP}: {evaluator['n_dropped']} {E17_ARM} products are dropped "
                          "from the pool, so its pool mean is not the generated set's")
    index, point = _e17_budget_point(evaluator)
    gap = evaluator["gap_decomposition"]
    support = ledger.declare_support(
        "e17_protgpt2_unconditioned_pool",
        f"ProtGPT2 unconditioned generation pool: {evaluator['n_pool']} retained products in "
        f"{gap['n_units']} independence units with {evaluator['n_dropped']} dropped; the endpoint "
        "is ESMFold2 mean per-residue CA pLDDT on the 0 to 100 scale, which is the folding model's "
        "confidence in its own coordinates and neither a measured stability nor a measured "
        f"function; the quoted budget is {point['n_selected']} candidates, the "
        f"{point['fraction']} selection fraction of the pool")
    base = ("e17", "arms", E17_ARM, "evaluators", E17_EVALUATOR)
    at_budget = base + ("gap_decomposition", "points", f"[{index}]")
    emit(E17_GAP, at_budget + ("selectors", E17_SELECTOR, "selected_yield"),
         identifier=f"likelihood_anti_selection/{E17_ARM}/mean_ca_plddt/top_twelve_by_likelihood",
         claim=f"mean predicted confidence of the {point['n_selected']} {E17_ARM} candidates its "
               "own likelihood ranks highest, on the 0 to 100 mean CA pLDDT scale",
         family="likelihood_anti_selection", support=support, unit=PLDDT, kind="artifact_leaf",
         reason="a recorded yield over the selected candidates; the decomposition resamples the "
                "gap against the oracle and the pool, not the yield itself",
         level="generation")
    emit(E17_GAP, base + ("pool_mean",),
         identifier=f"likelihood_anti_selection/{E17_ARM}/mean_ca_plddt/pool_mean",
         claim=f"mean predicted confidence over the whole {E17_ARM} unconditioned pool, which is "
               "random selection's expectation at every budget",
         family="likelihood_anti_selection", support=support, unit=PLDDT, kind="artifact_leaf",
         reason="a recorded pool mean over every retained product, not a resampled estimate",
         level="generation")
    emit(E17_GAP, at_budget + ("oracle_yield",),
         identifier=f"likelihood_anti_selection/{E17_ARM}/mean_ca_plddt/top_twelve_by_oracle",
         claim=f"mean predicted confidence of the {point['n_selected']} {E17_ARM} candidates an "
               "oracle ranking by the evaluator itself selects; circular by construction and "
               "unusable as a selector, it measures what the pool contains",
         family="likelihood_anti_selection", support=support, unit=PLDDT, kind="artifact_leaf",
         reason="a recorded ceiling over the selected candidates, not a resampled estimate",
         level="generation")
