"""The guided-decoding and selection rows R6 rests on, and the refusals that guard them.

R6 quotes three kinds of reading the artifacts do not hand over as single
leaves. An achieved upper bound of the pre-registered equivalence family is
recorded in three places at once, so a disagreement between them means there is
no one bound to quote; the permuted-column control the same rule does not decide
on is recorded once, in its own interval. A contrast is only the sealed eight's
while it covers all eight domains, so an unestimable contrast and one missing a
domain are refusals rather than quieter numbers. And "the best admissible
scorer" is a maximum over a declared selector roster, so a roster the artifact
does not carry, or one whose members are not all estimable, makes the maximum a
maximum over something else.

Two refusals carry the prose directly: every equivalence reading the section
reports as unestimable has to be unestimable for exactly one domain and for the
admissible-candidate floor, and native likelihood's captured fraction has to
still be negative for the section to quote it as a loss.

The last test runs on the retained artifacts and asserts the numbers the
manuscript quotes, at the precision it quotes them to.
"""
import json

import pytest

from scripts.capability.reporting import coupling_guidance_quantities as guidance
from src.capability.core.evidence_ledger import Artifacts, Ledger, LedgerError

DOMAINS = ("1E0L", "1F0M", "1IFY", "1V1C", "2K1B", "2LGW", "2LVN", "3DKM")
INTERVAL = "95% BCa bootstrap over domains, 9999 resamples, seed 20261013"
#: Every contrast the confirmatory family carries, cell contrasts and the two
#: scorer contrasts alike: the emitter reads four of them and the roster check
#: has to see the family it was computed inside.
CONTRASTS = ("C-Cd", "C-Cx", "C-N", "C-Pr", "F-M", "Pr-N",
             "potts_coupling-coupling_permuted", "potts_full-potts_fields")
#: The arms the design runs, as the cell list carries them. The analysis declares
#: no contrast definitions, so the permuted-column control is only that control
#: while both of the arms its key names are in this roster.
ARMS = ("C", "Cd", "Cx", "F", "H", "M", "N", "P", "Pr")


def cells(arms: tuple[str, ...] = ARMS) -> list[dict]:
    return [{"arm": arm, "model": "protgpt2", "domain": DOMAINS[0]} for arm in arms]


def contrast(mean: float, *, interval: tuple[float, float] | None = None,
             domains: tuple[str, ...] = DOMAINS, missing: dict | None = None,
             positive: int | None = None) -> dict:
    """One contrast record, estimated unless its interval is withheld."""
    estimated = interval is not None
    positive = len(domains) if positive is None else positive
    return {
        "n_domains": len(domains), "mean": mean, "interval95_bca": list(interval) if estimated
        else None,
        "favouring": {"positive": positive, "negative": len(domains) - positive, "tied": 0},
        "missing_domains": missing or {},
        "per_domain": {domain: {"diff": mean} for domain in domains},
        "status": "estimated" if estimated else "unestimable",
        "reason": None if estimated else f"{len(domains)} domains have a value",
    }


#: The permuted-column contrast as the retained analysis carries it in the one
#: stratum that resolves: ProtGPT2 covers the sealed eight, while the pooled and
#: ProGen2-medium readings lose the 2LGW coupling cell to the admissible floor,
#: which is the default :func:`primary` stages.
RESOLVED_PERMUTED = contrast(0.0064, interval=(-0.0010, 0.0168), positive=5)


def primary(**overrides) -> dict:
    """The primary TM family, one record per contrast and stratum."""
    base = {
        "C-Pr": contrast(-0.0138, interval=(-0.0397, -0.0040), positive=1),
        "F-M": contrast(0.0183, interval=(0.0033, 0.0422), positive=6),
        "C-N": contrast(-0.09, interval=(-0.2, -0.01), positive=0),
        "C-Cx": contrast(0.0056, domains=DOMAINS[:7],
                         missing={"2LGW": "C: 20 admissible < 25"}),
        "C-Cd": contrast(-0.0060, interval=(-0.0146, 0.0025), positive=3),
        "Pr-N": contrast(0.0126, interval=(0.0083, 0.0195)),
        "potts_coupling-coupling_permuted": contrast(0.0042, interval=(-0.0061, 0.0098),
                                                     positive=6),
        "potts_full-potts_fields": contrast(-0.0027, interval=(-0.0071, 0.0028), positive=3),
    }
    base.update(overrides)
    if set(base) != set(CONTRASTS):
        raise AssertionError("the staged family is not the frozen contrast family")
    return {"contrasts": base, "holm_family_size": len(base)}


def feasibility(**overrides) -> dict:
    base = {name: contrast(-0.1, interval=(-0.2, -0.05), positive=0) for name in CONTRASTS[:6]}
    base["Pr-N"] = contrast(-0.0622, interval=(-0.0991, -0.0359), positive=0)
    base["C-N"] = contrast(-0.1722, interval=(-0.2953, -0.0844), positive=0)
    base.update(overrides)
    return {"contrasts": base, "holm_family_size": len(base)}


def equivalence(primary_family: dict, *, margin: float = 0.05,
                bounds: dict | None = None) -> dict:
    """The headline block, read off the primary family it summarises."""
    strata = {}
    for stratum, family in primary_family.items():
        upper = {}
        for name in ("C-Pr", "F-M"):
            record = family["contrasts"][name]
            upper[name] = (record["interval95_bca"][1] if record["status"] == "estimated"
                           else None)
        if bounds and stratum in bounds:
            upper.update(bounds[stratum])
        strata[stratum] = {
            "mean": {name: family["contrasts"][name]["mean"] for name in ("C-Pr", "F-M")},
            "upper_bound": upper,
            "below_margin": {name: None if bound is None else bound < margin
                             for name, bound in upper.items()},
        }
    return {
        "margin_tm": margin, "rule": "conclude only if both bounds lie below the margin",
        "strata": strata,
        "achieved_upper_bound": {name: {stratum: strata[stratum]["upper_bound"][name]
                                        for stratum in strata}
                                 for name in ("C-Pr", "F-M")},
        "verdict": "unestimable", "verdict_is_secondary_flag": True, "verdict_reason": "staged",
    }


def selectors(*, estimated: tuple[str, ...] = guidance.SELECTORS,
              fractions: dict | None = None, roster: tuple[str, ...] = guidance.SELECTORS) -> dict:
    values = {"coupling_permuted": 0.0117, "native_loglik": -0.1413, "potts_coupling": 0.0530,
              "potts_fields": 0.1980, "potts_full": 0.2253}
    values.update(fractions or {})
    block = {}
    for name in roster:
        entry = {"mean_gain": 0.01, "mean_captured_fraction": values.get(name, 0.0)}
        if name in estimated:
            entry.update(status="estimated", interval95_bca=[values.get(name, 0.0) - 0.08,
                                                             values.get(name, 0.0) + 0.08])
        else:
            entry.update(status="unestimable", interval95_bca=None, reason="below the floor")
        block[name] = entry
    return block


def capture(*, arms: dict | None = None) -> dict:
    if arms is None:
        arms = {
            "N": {"mean_oracle_gain": 0.0734, "n_domains": len(DOMAINS), "missing_domains": {},
                  "per_domain": {}, "selectors": selectors(fractions={"potts_full": 0.1430})},
            "Pr": {"mean_oracle_gain": 0.0930, "n_domains": len(DOMAINS), "missing_domains": {},
                   "per_domain": {}, "selectors": selectors()},
        }
    return {"label": "captured_fraction_of_oracle_headroom", "k": 10, "arms": arms,
            "outside_primary_holm_family": True, "influences_primary_verdict": False}


def confirmatory_payload(*, strata: tuple[str, ...] = (*guidance.MODELS, guidance.POOLED),
                         pooled: dict | None = None, per_model: dict | None = None,
                         feasibility_pooled: dict | None = None,
                         interval: str = INTERVAL, capture_block: dict | None = None,
                         sealed: bool = True, margin: float = 0.05,
                         bounds: dict | None = None, arms: tuple[str, ...] = ARMS) -> dict:
    """One confirmatory analysis, with the pooled stratum carrying the quoted rows."""
    primary_family = {}
    for stratum in strata:
        if stratum == guidance.POOLED:
            primary_family[stratum] = pooled if pooled is not None else primary()
        elif stratum == "progen2-medium":
            primary_family[stratum] = per_model if per_model is not None else primary(
                **{"C-Pr": contrast(0.0015, domains=DOMAINS[:7],
                                    missing={"2LGW": "C: 20 admissible < 25"}),
                   "F-M": contrast(0.0038, interval=(-0.0002, 0.0119), positive=5)})
        else:
            # ProtGPT2 is the one stratum whose permuted-column contrast resolves,
            # and the section quotes its bound.
            primary_family[stratum] = primary(**{"C-Cx": RESOLVED_PERMUTED})
    feasibility_family = {stratum: (feasibility_pooled if stratum == guidance.POOLED
                                    and feasibility_pooled is not None else feasibility())
                          for stratum in strata}
    return {
        "schema": "e20_confirmatory_v1", "matches_sealed_eight": sealed,
        "domains": list(DOMAINS), "minimum_bootstrap_units": 8, "cells": cells(arms),
        "frozen": {"admissible_identity_below": 20.0, "minimum_admissible": 25,
                   "margin_tm": margin, "interval": interval, "selection_k": 10,
                   "selection_pool_arm": "N", "permutation_seed": 20261013},
        "primary_tm": primary_family,
        "coprimary_feasibility": feasibility_family,
        "equivalence": equivalence(primary_family, margin=margin, bounds=bounds),
        "captured_fraction_of_oracle_headroom": capture_block if capture_block is not None
        else capture(),
        "statement": "pre-specified analysis",
    }


def e17_payload(*, budgets: tuple[int, ...] = (12, 30, 600), pool_mean: float = 74.737,
                gap_pool_mean: float | None = None, quantity: str = guidance.E17_QUANTITY,
                status: str = "measured", dropped: int = 0,
                selector_names: tuple[str, ...] = ("likelihood", "random")) -> dict:
    points = [{"n_selected": budget, "fraction": budget / 600,
               "oracle_yield": 98.035 if budget == 12 else 90.0,
               "random_yield": pool_mean,
               "selectors": {name: {"selected_yield": 53.381 if name == "likelihood" else 81.5}
                             for name in selector_names}}
              for budget in budgets]
    evaluator = {
        "quantity": quantity, "status": status, "n_pool": 600, "n_dropped": dropped,
        "pool_mean": pool_mean,
        "gap_decomposition": {"n_units": 600, "n_pool": 600, "points": points,
                              "pool_mean": pool_mean if gap_pool_mean is None else gap_pool_mean},
    }
    return {"e17": {"arms": {"protgpt2": {"evaluators": {"mean_ca_plddt": evaluator}},
                             "zymctrl": {"evaluators": {"mean_ca_plddt": dict(evaluator)}}}}}


def stage(tmp_path, *, confirmatory: dict | None = None, e17: dict | None = None) -> Ledger:
    payloads = {
        guidance.CONFIRMATORY: confirmatory if confirmatory is not None
        else confirmatory_payload(),
        guidance.E17_GAP: e17 if e17 is not None else e17_payload(),
    }
    for relative, payload in payloads.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    ledger = Ledger(Artifacts(tmp_path))
    guidance.add_coupling_guidance_quantities(ledger)
    return ledger


def rows(ledger: Ledger) -> dict:
    return {quantity.id: quantity for quantity in ledger.quantities}


def test_every_quoted_reading_is_emitted_once_with_its_own_kind(tmp_path):
    emitted = rows(stage(tmp_path))
    bounds = [identifier for identifier in emitted if identifier.endswith("achieved_upper_bound")]
    # Every bound the section quotes: the four of the equivalence family and the
    # selector contrast inside the unguided pool. The permuted-column control's
    # upper end is not re-emitted beside the equivalence row that already carries
    # it, because one number under two identifiers is one number neither pins down.
    assert len(bounds) == len(guidance.QUOTED_BOUNDS) + 1
    assert "coupling_guidance/primary_tm/pooled/coupling_selector_over_permuted/" \
        "achieved_upper_bound" in bounds
    assert "coupling_guidance/primary_tm/protgpt2/C-Cx/achieved_upper_bound" not in emitted
    # An achieved upper bound is the end of an interval, so it is a bound and
    # never an estimate the manuscript owes a second interval for.
    for identifier in bounds:
        assert emitted[identifier].kind == "range_bound"
        assert emitted[identifier].interval is None
    best = emitted["coupling_selection/profile_guided_pool/"
                   "best_admissible_selector_captured_fraction"]
    assert best.kind == "range_bound" and "potts_full" in best.claim
    # The oracle headroom is a mean the analysis records without resampling it.
    for pool in ("unguided_pool", "profile_guided_pool"):
        headroom = emitted[f"coupling_selection/{pool}/oracle_headroom"]
        assert headroom.kind == "estimate" and headroom.interval is None
        assert headroom.no_interval_reason
    # Every copied interval carries the unit, draw count and seed the artifact's
    # own interval statement states, rather than a convention typed in here.
    for quantity in emitted.values():
        if quantity.interval is None:
            continue
        assert quantity.resampling_unit == guidance.DOMAIN_UNIT
        assert quantity.resampling_draws == 9999 and quantity.seed_set == (20261013,)
        assert quantity.interval_kind == guidance.BCA


def test_the_declared_constants_are_emitted_and_the_resampling_ones_are_not(tmp_path):
    emitted = rows(stage(tmp_path))
    values = {quantity.value for quantity in emitted.values()}
    # Draw count, and the bootstrap and permutation seeds.
    assert not values & {9999.0, 20261013.0}
    # The bootstrap floor is eight domains and so are the domain and sign counts
    # the section quotes, so the floor is excluded by every row at that value
    # being a count of domains rather than a declared constant.
    at_eight = {quantity.id: quantity for quantity in emitted.values()
                if quantity.value == 8.0}
    assert set(at_eight) == {
        "coupling_guidance/primary_tm/pooled/profile_over_unguided/domains_favouring_profile",
        "coupling_guidance/primary_tm/protgpt2/C-Cx/domains"}
    assert all(quantity.kind in ("count", "support_count")
               and quantity.unit == "sealed evaluation domains"
               for quantity in at_eight.values())
    # The three constants the endpoints are defined by are emitted, because the
    # Methods quotes each as a numeral: the identity restriction, the
    # admissible-candidate floor and the selection budget.
    assert emitted["coupling_guidance/admissible_identity_below"].value == 20.0
    assert emitted["coupling_guidance/minimum_admissible"].value == 25.0
    assert emitted["coupling_selection/selection_budget"].value == 10.0
    for identifier in ("coupling_guidance/minimum_admissible",
                       "coupling_selection/selection_budget"):
        assert emitted[identifier].kind == "constant"
        assert emitted[identifier].unit == guidance.ADMISSIBLE_CANDIDATES


def test_a_budget_recorded_twice_has_to_agree(tmp_path):
    payload = confirmatory_payload()
    payload["captured_fraction_of_oracle_headroom"]["k"] = 20
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=payload)
    assert "selection budget is recorded twice and the two disagree" in str(refusal.value)


def test_a_bound_recorded_three_times_has_to_agree_in_all_three(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(
            bounds={"protgpt2": {"F-M": 0.0431}}))
    assert "different achieved upper bounds" in str(refusal.value)


def test_the_permuted_column_bound_is_read_from_the_one_place_that_records_it(tmp_path):
    # It is outside the pre-registered family, so the artifact records it only
    # in its own interval and the row points there.
    emitted = rows(stage(tmp_path))
    bound = emitted["coupling_guidance/equivalence/protgpt2/C-Cx/achieved_upper_bound"]
    assert bound.kind == "range_bound" and round(bound.value, 4) == 0.0168
    assert bound.source_pointer == ("primary_tm", "protgpt2", "contrasts", "C-Cx",
                                    "interval95_bca", "[1]")
    assert "the declared margin rule does not decide on" in bound.claim
    # A stratum whose permuted-column contrast resolved would no longer be the
    # unestimable reading the section reports.
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(
            pooled=primary(**{"C-Cx": RESOLVED_PERMUTED})))
    assert "now carries an achieved upper bound" in str(refusal.value)


def test_a_bound_at_or_above_the_declared_margin_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(margin=0.01))
    assert "does not lie below the declared margin" in str(refusal.value)


def test_a_contrast_short_of_the_sealed_eight_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(pooled=primary(
            **{"Pr-N": contrast(0.0126, interval=(0.0083, 0.0195), domains=DOMAINS[:7],
                                missing={"2LGW": "C: 20 admissible < 25"})})))
    assert "is not a contrast over the sealed eight" in str(refusal.value)


def test_an_unestimable_quoted_contrast_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(feasibility_pooled=feasibility(
            **{"C-N": contrast(-0.1722, domains=DOMAINS[:6])})))
    assert "carries no quotable estimate" in str(refusal.value)


def test_the_unestimable_equivalence_reading_has_to_be_the_one_the_prose_describes(tmp_path):
    # Two domains short is not "one domain supplied too few admissible candidates".
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(per_model=primary(
            **{"C-Pr": contrast(0.0015, domains=DOMAINS[:6],
                                missing={"2LGW": "C: 20 admissible < 25",
                                         "2LVN": "C: 19 admissible < 25"}),
               "F-M": contrast(0.0038, interval=(-0.0002, 0.0119), positive=5)})))
    assert "one domain supplying too few admissible candidates" in str(refusal.value)
    # Nor is a domain withheld for another reason.
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(per_model=primary(
            **{"C-Pr": contrast(0.0015, domains=DOMAINS[:7],
                                missing={"2LGW": "no C cell at the frozen dose"}),
               "F-M": contrast(0.0038, interval=(-0.0002, 0.0119), positive=5)})))
    assert "one domain supplying too few admissible candidates" in str(refusal.value)


def test_a_reading_that_became_estimable_cannot_be_reported_as_unestimable(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(per_model=primary(
            **{"F-M": contrast(0.0038, interval=(-0.0002, 0.0119), positive=5)})))
    assert "now carries an achieved upper bound" in str(refusal.value)


def test_a_capture_arm_short_of_a_domain_or_a_selector_refuses(tmp_path):
    short = capture(arms={
        "N": {"mean_oracle_gain": 0.0767, "n_domains": 7,
              "missing_domains": {"2LGW": "no C pool"}, "per_domain": {}, "selectors": selectors()},
        "Pr": {"mean_oracle_gain": 0.0930, "n_domains": len(DOMAINS), "missing_domains": {},
               "per_domain": {}, "selectors": selectors()}})
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(capture_block=short))
    assert "its headroom is not the sealed eight's" in str(refusal.value)

    thinned = capture(arms={
        "N": {"mean_oracle_gain": 0.0734, "n_domains": len(DOMAINS), "missing_domains": {},
              "per_domain": {}, "selectors": selectors()},
        "Pr": {"mean_oracle_gain": 0.0930, "n_domains": len(DOMAINS), "missing_domains": {},
               "per_domain": {}, "selectors": selectors(roster=guidance.SELECTORS[:4])}})
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(capture_block=thinned))
    assert "not the declared" in str(refusal.value)


def test_a_best_over_a_partly_unestimable_roster_refuses(tmp_path):
    partial = capture(arms={
        "N": {"mean_oracle_gain": 0.0734, "n_domains": len(DOMAINS), "missing_domains": {},
              "per_domain": {}, "selectors": selectors()},
        "Pr": {"mean_oracle_gain": 0.0930, "n_domains": len(DOMAINS), "missing_domains": {},
               "per_domain": {},
               "selectors": selectors(estimated=guidance.SELECTORS[:4])}})
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(capture_block=partial))
    assert "a maximum over the rest of it" in str(refusal.value)


def test_native_likelihood_capturing_a_positive_share_refuses(tmp_path):
    positive = capture(arms={
        "N": {"mean_oracle_gain": 0.0734, "n_domains": len(DOMAINS), "missing_domains": {},
              "per_domain": {}, "selectors": selectors()},
        "Pr": {"mean_oracle_gain": 0.0930, "n_domains": len(DOMAINS), "missing_domains": {},
               "per_domain": {}, "selectors": selectors(fractions={"native_loglik": 0.04})}})
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(capture_block=positive))
    assert "no longer the loss the prose quotes" in str(refusal.value)


def test_an_interval_statement_the_rows_cannot_read_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(
            interval="95% percentile bootstrap over candidates, 2000 resamples"))
    assert "is not the BCa bootstrap over domains these rows copy" in str(refusal.value)


def test_another_domain_set_or_another_stratum_roster_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(sealed=False))
    assert "not the sealed evaluation" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(
            strata=("progen2-medium", guidance.POOLED)))
    assert "not the declared" in str(refusal.value)


def test_the_e17_budget_and_endpoint_have_to_be_the_ones_the_section_quotes(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, e17=e17_payload(budgets=(30, 60, 600)))
    assert "decomposition points select 12 candidates" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, e17=e17_payload(selector_names=("random",)))
    assert "no likelihood selector" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, e17=e17_payload(quantity="esmfold2_ptm"))
    assert "not the measured predicted-confidence endpoint" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, e17=e17_payload(dropped=3))
    assert "products are dropped" in str(refusal.value)


def test_a_pool_mean_recorded_twice_has_to_agree(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, e17=e17_payload(gap_pool_mean=74.734))
    assert "pool mean is recorded twice and the two disagree" in str(refusal.value)


def test_the_permuted_column_control_is_read_in_every_stratum(tmp_path):
    """Both sides of the comparison Supplementary Note 14 draws have to be sourced.

    The note reads this campaign's permuted-column control against the ProGen3
    campaign's, which departs from it. The ProGen3 side carries the contrast in
    every stratum, so this side does too: the stratum that resolves carries its
    mean and the lower end of its interval, and the two that do not carry the
    mean over the domains that have a value with the artifact's own reason for
    having no interval.
    """
    emitted = rows(stage(tmp_path))
    for stratum in guidance.PERMUTED_STRATA:
        prefix = f"coupling_guidance/primary_tm/{stratum}/{guidance.PERMUTED}"
        mean = emitted[f"{prefix}/true_over_permuted"]
        assert mean.kind == "estimate" and mean.unit == guidance.TM
        signs = [emitted[f"{prefix}/{name}"].value for name in
                 ("domains_favouring_true_columns", "domains_favouring_permuted_columns",
                  "domains_tied")]
        assert sum(signs) == emitted[f"{prefix}/domains"].value
        assert emitted[f"{prefix}/domains"].kind == "support_count"
    # ProtGPT2 resolves over the sealed eight, so its interval is copied with the
    # unit, draw count and seed it was computed under.
    resolved = emitted[f"coupling_guidance/primary_tm/protgpt2/{guidance.PERMUTED}"
                       "/true_over_permuted"]
    assert resolved.interval == (-0.0010, 0.0168)
    assert resolved.interval_kind == guidance.BCA
    assert resolved.resampling_unit == guidance.DOMAIN_UNIT
    assert resolved.resampling_draws == 9999 and resolved.seed_set == (20261013,)
    assert resolved.no_interval_reason is None
    lower = emitted[f"coupling_guidance/primary_tm/protgpt2/{guidance.PERMUTED}"
                    "/achieved_lower_bound"]
    assert lower.kind == "range_bound" and lower.interval is None and lower.value == -0.0010
    assert emitted[f"coupling_guidance/primary_tm/protgpt2/{guidance.PERMUTED}/domains"].value \
        == len(DOMAINS)


def test_an_unestimable_stratum_carries_its_reason_and_no_interval(tmp_path):
    ledger = stage(tmp_path)
    emitted = rows(ledger)
    for stratum in (guidance.POOLED, "progen2-medium"):
        prefix = f"coupling_guidance/primary_tm/{stratum}/{guidance.PERMUTED}"
        mean = emitted[f"{prefix}/true_over_permuted"]
        # The mean over seven domains exists and is recorded; what does not exist
        # is an interval, and the row carries the artifact's own reason for that
        # rather than a silence a reader could mistake for a resolved difference.
        assert mean.interval is None and mean.no_interval_reason
        assert "below the 8-domain floor" in mean.no_interval_reason or \
            "domains have a value" in mean.no_interval_reason
        assert "may not be quoted as one" in mean.claim
        assert emitted[f"{prefix}/domains"].value == len(DOMAINS) - 1
        # No bound row is written where the analysis issues no bound.
        assert f"{prefix}/achieved_lower_bound" not in emitted
        assert f"{prefix}/achieved_upper_bound" not in emitted


def test_every_unestimable_bound_is_recorded_as_a_gap(tmp_path):
    gaps = {gap.id: gap for gap in stage(tmp_path).gaps}
    assert set(gaps) == {
        f"coupling_guidance/equivalence/{stratum}/{contrast}/achieved_upper_bound"
        for stratum, contrast in guidance.UNESTIMABLE_BOUNDS}
    for gap in gaps.values():
        # The absence is the frozen rule's own outcome, so it is recorded as one
        # and never worked around by relaxing the floor.
        assert gap.reachability == "not_measured"
        assert "unestimable" in gap.reason and "not relaxed after the fact" in gap.reason
        assert gap.searched and gap.family == "coupling_guidance"


def test_a_permuted_control_whose_arms_the_design_does_not_run_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(
            arms=tuple(arm for arm in ARMS if arm != "Cx")))
    assert "it is not the permuted-column control" in str(refusal.value)


def test_a_stratum_that_is_neither_estimated_nor_unestimable_refuses(tmp_path):
    # A status outside the two the frozen rule issues is neither outcome, so no
    # row is written off it.
    other = contrast(0.0056, domains=DOMAINS[:7], missing={"2LGW": "C: 20 admissible < 25"})
    other["status"] = "withheld"
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(pooled=primary(**{"C-Cx": other})))
    assert "neither estimated nor unestimable" in str(refusal.value)
    # An unestimable record with no reason gives the mean-with-reason row nothing
    # to carry.
    silent = contrast(0.0056, domains=DOMAINS[:7], missing={"2LGW": "C: 20 admissible < 25"})
    silent["reason"] = None
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(pooled=primary(**{"C-Cx": silent})))
    assert "ambiguous" in str(refusal.value)
    # An unestimable record that also carries an interval is caught by the guard
    # that owns the section's unestimable readings, before any row is built.
    both = contrast(0.0056, domains=DOMAINS[:7], missing={"2LGW": "C: 20 admissible < 25"})
    both["interval95_bca"] = [-0.001, 0.012]
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, confirmatory=confirmatory_payload(pooled=primary(**{"C-Cx": both})))
    assert "now carries an achieved upper bound" in str(refusal.value)


def test_the_selector_contrast_bound_is_its_own_row(tmp_path):
    emitted = rows(stage(tmp_path))
    bound = emitted["coupling_guidance/primary_tm/pooled/coupling_selector_over_permuted/"
                    "achieved_upper_bound"]
    estimate = emitted["coupling_guidance/primary_tm/pooled/coupling_selector_over_permuted"]
    assert bound.kind == "range_bound" and bound.interval is None
    assert bound.value == estimate.interval[1]
    # The section quotes this reading only as a bound, and its claim has to keep
    # the scorer contrast apart from the decoding arm's permuted-column control.
    assert "scorer contrast inside one pool" in bound.claim
    assert guidance.CONTRAST_PROSE[guidance.PERMUTED][1] in bound.claim


def test_the_retained_artifacts_carry_the_numbers_the_manuscript_quotes():
    artifacts = Artifacts()
    for relative in (guidance.CONFIRMATORY, guidance.E17_GAP):
        if not artifacts.exists(relative):
            pytest.skip("the E20 and E17 artifacts are not staged in this checkout")
    ledger = Ledger(artifacts)
    guidance.add_coupling_guidance_quantities(ledger)
    emitted = rows(ledger)

    def quoted(identifier: str, places: int) -> float:
        return round(emitted[identifier].value, places)

    def bounds(identifier: str, places: int) -> tuple[float, float]:
        return tuple(round(bound, places) for bound in emitted[identifier].interval)

    assert quoted("coupling_guidance/admissible_identity_below", 0) == 20.0
    assert quoted("coupling_guidance/minimum_admissible", 0) == 25.0
    assert quoted("coupling_selection/selection_budget", 0) == 10.0
    assert quoted("coupling_guidance/equivalence/declared_margin", 2) == 0.05
    assert quoted("coupling_guidance/equivalence/protgpt2/C-Pr/achieved_upper_bound", 4) == -0.0040
    assert quoted("coupling_guidance/equivalence/progen2-medium/F-M/achieved_upper_bound", 4) \
        == 0.0119
    assert quoted("coupling_guidance/equivalence/protgpt2/F-M/achieved_upper_bound", 4) == 0.0422
    assert quoted("coupling_guidance/equivalence/protgpt2/C-Cx/achieved_upper_bound", 4) == 0.0168

    assert quoted("coupling_guidance/primary_tm/pooled/profile_over_unguided", 4) == 0.0126
    assert emitted["coupling_guidance/primary_tm/pooled/profile_over_unguided/"
                   "domains_favouring_profile"].value == 8.0
    coupling = emitted["coupling_guidance/primary_tm/pooled/coupling_selector_over_permuted"]
    assert round(coupling.value, 4) == 0.0042
    assert coupling.interval[0] < 0.0 < coupling.interval[1]
    # "at an achieved upper bound of $+0.0098$ TM", which the section quotes as a
    # bound and never as a point estimate.
    assert quoted("coupling_guidance/primary_tm/pooled/coupling_selector_over_permuted/"
                  "achieved_upper_bound", 4) == 0.0098

    # The co-primary reductions are stored as negative differences and quoted as
    # the magnitudes lost, so the point and both bounds have to flip together.
    assert quoted("coupling_guidance/feasibility/pooled/profile_minus_unguided", 4) == -0.0622
    assert bounds("coupling_guidance/feasibility/pooled/profile_minus_unguided", 4) \
        == (-0.0991, -0.0359)
    assert quoted("coupling_guidance/feasibility/pooled/couplings_minus_unguided", 4) == -0.1722
    assert bounds("coupling_guidance/feasibility/pooled/couplings_minus_unguided", 4) \
        == (-0.2953, -0.0844)

    assert quoted("coupling_selection/unguided_pool/oracle_headroom", 4) == 0.0734
    assert quoted("coupling_selection/profile_guided_pool/oracle_headroom", 4) == 0.0930
    assert quoted("coupling_selection/profile_guided_pool/"
                  "best_admissible_selector_captured_fraction", 3) == 0.225
    assert quoted("coupling_selection/profile_guided_pool/"
                  "native_likelihood_captured_fraction", 3) == -0.141
    assert bounds("coupling_selection/profile_guided_pool/"
                  "native_likelihood_captured_fraction", 3) == (-0.316, -0.045)

    assert quoted("likelihood_anti_selection/protgpt2/mean_ca_plddt/"
                  "top_twelve_by_likelihood", 1) == 53.4
    assert quoted("likelihood_anti_selection/protgpt2/mean_ca_plddt/pool_mean", 1) == 74.7
    assert quoted("likelihood_anti_selection/protgpt2/mean_ca_plddt/top_twelve_by_oracle", 1) \
        == 98.0


def test_the_retained_analysis_carries_the_permuted_column_control_in_every_stratum():
    """The confirmatory side of the comparison Supplementary Note 14 draws.

    The note's "there, decoding with permuted coupling columns was not
    distinguishable from decoding with the true ones" rests on these three
    readings, and only ProtGPT2's resolves. The two that do not are unestimable
    on seven of eight domains, which is the frozen consequence of one coupling
    cell supplying twenty admissible candidates against a floor of twenty-five,
    and they are recorded that way rather than relaxed.
    """
    artifacts = Artifacts()
    if not artifacts.exists(guidance.CONFIRMATORY):
        pytest.skip("the E20 confirmatory analysis is not staged in this checkout")
    ledger = Ledger(artifacts)
    guidance.add_coupling_guidance_quantities(ledger)
    emitted = rows(ledger)
    resolved = "coupling_guidance/primary_tm/protgpt2/C-Cx"
    mean = emitted[f"{resolved}/true_over_permuted"]
    assert round(mean.value, 6) == 0.006436
    assert tuple(round(bound, 6) for bound in mean.interval) == (-0.001041, 0.016840)
    assert round(emitted[f"{resolved}/achieved_lower_bound"].value, 6) == -0.001041
    # The upper end is the bound the main text quotes, and it has exactly one row.
    assert round(emitted["coupling_guidance/equivalence/protgpt2/C-Cx/"
                         "achieved_upper_bound"].value, 6) == 0.016840
    assert f"{resolved}/achieved_upper_bound" not in emitted
    assert emitted[f"{resolved}/domains"].value == 8.0
    assert [emitted[f"{resolved}/{name}"].value for name in
            ("domains_favouring_true_columns", "domains_favouring_permuted_columns",
             "domains_tied")] == [4.0, 4.0, 0.0]
    # The two unestimable strata: a mean over seven domains, no interval, the
    # artifact's own reason, and no bound row anywhere.
    for stratum, signs in (("pooled", [6.0, 1.0, 0.0]), ("progen2-medium", [3.0, 4.0, 0.0])):
        prefix = f"coupling_guidance/primary_tm/{stratum}/C-Cx"
        unestimable = emitted[f"{prefix}/true_over_permuted"]
        assert unestimable.interval is None
        assert "below the 8-domain floor" in unestimable.no_interval_reason
        assert "2LGW" in unestimable.no_interval_reason
        assert emitted[f"{prefix}/domains"].value == 7.0
        assert [emitted[f"{prefix}/{name}"].value for name in
                ("domains_favouring_true_columns", "domains_favouring_permuted_columns",
                 "domains_tied")] == signs
        assert f"{prefix}/achieved_upper_bound" not in emitted
    gaps = {gap.id: gap for gap in ledger.gaps}
    for stratum in ("pooled", "progen2-medium"):
        gap = gaps[f"coupling_guidance/equivalence/{stratum}/C-Cx/achieved_upper_bound"]
        assert "20 admissible < 25" in gap.reason and gap.reachability == "not_measured"
