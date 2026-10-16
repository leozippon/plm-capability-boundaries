"""The ProGen3 cross-model readings the guidance section closes on, and their refusals.

The campaign reports bounds and issues no verdict, so three things have to hold
for the quoted readings to mean what the section says. The report has to still
issue no verdict and no pass/fail flag, because every bound here says in its own
claim that nothing is tested against the confirmatory campaign's margin. The
pooled stratum has to pool exactly the two bfloat16 rungs, because "across 3B and
112M checkpoints" is that and nothing else, and no stratum may pool across a
measure line. And the precision control has to be two precisions of one
checkpoint at identical doses, because otherwise the kernel precision is not the
only thing moving and the contrast is not a precision control.

A reduction over a set is only statable when the artifact covers the set: a
contrast short of the sealed eight, a strata or arm roster the report does not
carry, and an unestimable arm inside the roster the precision maximum is taken
over are refusals rather than quieter numbers.

The last test runs on the retained analysis and asserts the numbers the section
quotes, at the precision it quotes them to.
"""
import json

import pytest

from scripts.capability.reporting import progen3_robustness_quantities as robustness
from src.capability.core.evidence_ledger import Artifacts, Ledger, LedgerError

DOMAINS = ("1E0L", "1F0M", "1IFY", "1V1C", "2K1B", "2LGW", "2LVN", "3DKM")
INTERVAL = "95% BCa bootstrap over domains, 9999 resamples, seed 20261013"
CONTRASTS = ("C-Cx", "C-M", "C-Pr", "M-N", "Pr-N")
DOSES = {"C": [0.0, 1.562], "Cx": [0.0, 2.652], "M": [0.1794, 0.0], "N": [0.0, 0.0],
         "Pr": [0.1741, 0.0]}
#: The precision contrast's achieved upper bound per arm, with C the widest.
PRECISION_BOUNDS = {"C": 0.004249827578879945, "Cx": 0.0012831380940152362,
                    "M": 0.001708722484243752, "N": -0.00011324981149631368,
                    "Pr": 0.003377873479761957}


def per_domain(domains: tuple[str, ...] = DOMAINS, positive: int | None = None, *,
               overrides: dict | None = None) -> dict:
    """One per-domain block whose signs match ``positive`` and whose superlatives differ.

    The endpoint difference is largest in magnitude at the last domain that
    favours the true columns, identity to the reference rises most in that same
    domain, and identity against the family alignment rises most in the first
    domain --- which is the shape the retained analysis has, and the one that
    makes a claim read off the wrong superlative visible.
    """
    positive = len(domains) if positive is None else positive
    block = {}
    for index, domain in enumerate(domains):
        favours = index < positive
        block[domain] = {
            "diff": (0.001 * (index + 1)) if favours else (-0.0005 * (index + 1)),
            "identity_diff": (1.0 * (index + 1)) if favours else (-0.1 * (index + 1)),
            "msa_identity_diff": 1.0 * (len(domains) - index),
        }
    block.update(overrides or {})
    return block


def contrast(mean: float, *, interval: tuple[float, float] | None = None,
             domains: tuple[str, ...] = DOMAINS, missing: dict | None = None,
             positive: int | None = None, domain_block: dict | None = None,
             identity: bool = True) -> dict:
    estimated = interval is not None
    positive = len(domains) if positive is None else positive
    block = per_domain(domains, positive) if domain_block is None else domain_block
    record = {
        "n_domains": len(domains), "mean": mean,
        "interval95_bca": list(interval) if estimated else None,
        "favouring": {"positive": positive, "negative": len(domains) - positive, "tied": 0},
        "missing_domains": missing or {}, "status": "estimated" if estimated else "unestimable",
        "reason": None if estimated else f"{len(domains)} domains have a value",
        "per_domain": block,
    }
    if identity:
        record["mean_identity_diff"] = sum(
            row["identity_diff"] for row in block.values()) / max(len(block), 1)
        record["mean_msa_identity_diff"] = sum(
            row["msa_identity_diff"] for row in block.values()) / max(len(block), 1)
    return record


def primary(**overrides) -> dict:
    base = {
        "C-Cx": contrast(0.0038, interval=(-0.0009, 0.0152), positive=5),
        "C-M": contrast(0.0029, interval=(-0.0042, 0.0209), positive=3),
        "C-Pr": contrast(-0.0003, interval=(-0.0043, 0.006611162597412526), positive=3),
        "M-N": contrast(0.0039, interval=(0.0010, 0.0072), positive=7),
        "Pr-N": contrast(0.007162862315468529,
                         interval=(0.0035030281095949373, 0.013585081527265841)),
    }
    base.update(overrides)
    if set(base) != set(CONTRASTS):
        raise AssertionError("the staged family is not the campaign's contrast family")
    return base


def strata(*, pooled: dict | None = None,
           names: tuple[str, ...] = (*robustness.RUNGS, robustness.POOLED)) -> dict:
    return {name: (pooled if name == robustness.POOLED and pooled is not None else primary())
            for name in names}


def precision(*, bounds: dict | None = None, arms: tuple[str, ...] = robustness.ARMS,
              unestimable: tuple[str, ...] = (), float16_doses: dict | None = None,
              contrast_label: str | None = None) -> dict:
    bounds = PRECISION_BOUNDS if bounds is None else bounds
    by_arm = {}
    for arm in arms:
        if arm in unestimable:
            by_arm[arm] = contrast(0.0, domains=DOMAINS[:7], missing={"2LGW": "no float16 cell"})
            continue
        by_arm[arm] = contrast(bounds[arm] / 2, interval=(-0.001, bounds[arm]))
    float16_rung, bfloat16_rung = "progen3-112m-float16", "progen3-112m-bfloat16"
    return {
        "checkpoints": {
            robustness.PRECISION_CHECKPOINT: {
                "bfloat16_rung": bfloat16_rung, "float16_rung": float16_rung,
                "contrast": contrast_label if contrast_label is not None
                else f"{float16_rung} - {bfloat16_rung}",
                "doses": float16_doses if float16_doses is not None else DOSES,
                "endpoints": {robustness.PRIMARY: {
                    "achieved_upper_bound": {arm: bounds[arm] for arm in arms},
                    "by_arm": by_arm,
                }},
            },
        },
    }


def robustness_payload(*, schema: str = robustness.SCHEMA, verdict: bool = False,
                       sealed: bool = True, arms: tuple[str, ...] = robustness.ARMS,
                       rungs: tuple[str, ...] = robustness.RUNGS,
                       pooled_rungs: tuple[str, ...] = robustness.POOLED_RUNGS,
                       pooled_stratum: str = robustness.POOLED,
                       stratum_names: tuple[str, ...] | None = None,
                       pooled: dict | None = None, interval: str = INTERVAL,
                       headline: dict | None = None, candidates: dict | None = None,
                       precision_block: dict | None = None,
                       dose_statistics: dict | None = None,
                       permuted_arms: tuple[str, str] = ("C", "Cx")) -> dict:
    names = stratum_names if stratum_names is not None else (*rungs, pooled_stratum)
    families = strata(pooled=pooled, names=names)
    bounds = {
        endpoint: {"definition": f"the {endpoint} endpoint", "strata": families}
        for endpoint in robustness.ENDPOINTS
    }
    achieved = {robustness.PRIMARY: {
        name: {stratum: (family[name]["interval95_bca"][1]
                         if family[name]["status"] == "estimated" else None)
               for stratum, family in families.items()}
        for name in CONTRASTS}}
    for stratum, moved in (headline or {}).items():
        for name, value in moved.items():
            achieved[robustness.PRIMARY][name][stratum] = value
    design_rungs = {
        rung: {"base_checkpoint": rung.rsplit("-", 1)[0],
               "measure_line": rung.rsplit("-", 1)[1],
               "candidates_per_cell": (candidates or {}).get(rung, [200]),
               "doses": DOSES}
        for rung in rungs
    }
    return {
        "schema": schema, "issues_a_verdict": verdict, "issues_a_pass_fail_flag": False,
        "statement": "descriptive robustness report",
        "design": {
            "matches_sealed_eight": sealed, "arms": list(arms), "domains": list(DOMAINS),
            "rungs": design_rungs,
            "contrast_definitions": {
                name: {"left": name.split("-")[0], "right": name.split("-")[1],
                       "reads": f"the {name} contrast"}
                for name in CONTRASTS
            } | {robustness.PERMUTED: {
                "left": permuted_arms[0], "right": permuted_arms[1],
                "reads": "coupling against its column-permuted control"}},
            "measure_lines": {
                "bfloat16": {"pooled_stratum": pooled_stratum, "rungs": list(pooled_rungs)},
                "float16": {"pooled_stratum": None, "rungs": ["progen3-112m-float16"]},
            },
        },
        "discipline": {
            "interval": interval, "admissibility": "identity strictly below 20%",
            "independence_unit": "domain", "minimum_bootstrap_units": 8,
            "interval_coverage_note": "BCa over 8 domains reads optimistic.",
        },
        "bounds": bounds, "achieved_upper_bounds": achieved,
        "precision_control": precision_block if precision_block is not None else precision(),
        "reference_scale": {"margin_tm": 0.05, "role": "reference scale only"},
        "declared_context_dose_floor": {"calibrations": dose_statistics if dose_statistics
                                        is not None else
                                        {rung: {"decision_statistic": "median per-residue KL"}
                                         for rung in robustness.POOLED_RUNGS}},
    }


def stage(tmp_path, payload: dict | None = None) -> Ledger:
    path = tmp_path / robustness.ROBUSTNESS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload if payload is not None else robustness_payload()))
    ledger = Ledger(Artifacts(tmp_path))
    robustness.add_progen3_robustness_quantities(ledger)
    return ledger


def rows(ledger: Ledger) -> dict:
    return {quantity.id: quantity for quantity in ledger.quantities}


def test_each_reading_carries_its_kind_its_stratum_and_no_verdict(tmp_path):
    emitted = rows(stage(tmp_path))
    gain = emitted[f"progen3_robustness/primary_tm/{robustness.POOLED}/profile_over_unguided"]
    assert gain.kind == "estimate" and gain.interval is not None
    assert gain.resampling_unit and gain.resampling_draws == 9999 and gain.seed_set == (20261013,)
    for identifier in (f"progen3_robustness/primary_tm/{robustness.POOLED}/C-Pr/"
                       "achieved_upper_bound",
                       f"progen3_robustness/precision_control/"
                       f"{robustness.PRECISION_CHECKPOINT}/primary_tm/largest_endpoint_shift"):
        # An achieved upper bound is the end of an interval, never a point
        # estimate the manuscript owes a second interval for.
        assert emitted[identifier].kind == "range_bound"
        assert emitted[identifier].interval is None
    # The campaign issues no verdict, so no row may carry one, and the margin is
    # named only as the reference scale it is.
    for quantity in emitted.values():
        assert quantity.verdict is None and quantity.level == "generation"
    assert "reference scale only" in emitted[
        f"progen3_robustness/primary_tm/{robustness.POOLED}/C-Pr/achieved_upper_bound"].claim
    # Every quoted reading is keyed to the stratum it is read from, and the three
    # readings the section closes on are the pooled ones and the precision
    # control's. Only the permuted-column control is read in every stratum,
    # because quoting it in the one stratum that resolves would be a selection.
    declared = (*robustness.RUNGS, robustness.POOLED)
    for identifier in emitted:
        assert "precision_control" in identifier or any(name in identifier for name in declared)
        if robustness.PERMUTED in identifier:
            continue
        assert robustness.POOLED in identifier or "precision_control" in identifier
    assert {identifier for identifier in emitted if robustness.PERMUTED in identifier
            and any(name in identifier for name in robustness.RUNGS)}


def test_the_margin_and_the_campaign_constants_are_not_emitted(tmp_path):
    emitted = rows(stage(tmp_path))
    values = {quantity.value for quantity in emitted.values()}
    # The declared margin, the draw count and seed, and the candidates per cell.
    assert not values & {0.05, 9999.0, 20261013.0, 200.0}
    # The counts emitted are the sign count the section quotes in words and the
    # permuted-column control's three sign counts in each stratum. Nothing else
    # is a count.
    expected = {f"progen3_robustness/primary_tm/{robustness.POOLED}/profile_over_unguided/"
                "domains_favouring_profile"}
    for stratum in (*robustness.RUNGS, robustness.POOLED):
        base = f"progen3_robustness/primary_tm/{stratum}/{robustness.PERMUTED}"
        expected |= {f"{base}/domains_favouring_true_columns",
                     f"{base}/domains_favouring_permuted_columns", f"{base}/domains_tied"}
    assert {quantity.id for quantity in emitted.values()
            if quantity.kind == "count"} == expected


def test_a_report_that_began_issuing_a_verdict_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(verdict=True))
    assert "now issues a verdict" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(schema="progen3_robustness_bounds_v2"))
    assert f"not a {robustness.SCHEMA} report" in str(refusal.value)


def test_another_roster_of_arms_rungs_strata_or_domains_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(sealed=False))
    assert "not the sealed evaluation eight" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(arms=robustness.ARMS[:4]))
    assert "and not the declared" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(rungs=robustness.RUNGS[:2]))
    assert "and not the declared" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(
            stratum_names=(*robustness.RUNGS, robustness.POOLED, "pooled_all")))
    assert "carries strata" in str(refusal.value)


def test_a_pooled_stratum_that_is_not_across_both_checkpoints_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(pooled_rungs=("progen3-112m-bfloat16",)))
    assert "not the one across both checkpoints" in str(refusal.value)


def test_a_contrast_short_of_the_sealed_eight_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(pooled=primary(
            **{"Pr-N": contrast(0.0072, interval=(0.0035, 0.0136), domains=DOMAINS[:7],
                                missing={"2LGW": "20 admissible < 25"})})))
    assert "is not a contrast over the sealed eight" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(pooled=primary(
            **{"C-Pr": contrast(-0.0003, domains=DOMAINS[:7],
                                missing={"2LGW": "20 admissible < 25"})})))
    assert "carries no quotable estimate" in str(refusal.value)


def test_a_bound_recorded_twice_has_to_agree(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(
            headline={robustness.POOLED: {"C-Pr": 0.0067}}))
    assert "records different achieved upper bounds" in str(refusal.value)


def test_a_precision_control_that_moves_more_than_the_precision_refuses(tmp_path):
    shifted = dict(DOSES, C=[0.0, 1.7])
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(precision_block=precision(float16_doses=shifted)))
    assert "the kernel precision is not the only thing that moves" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(
            precision_block=precision(contrast_label="progen3-112m-bfloat16 - "
                                                     "progen3-112m-float16")))
    assert "not the float16 rung minus the bfloat16 rung" in str(refusal.value)


def test_a_precision_maximum_over_part_of_the_arm_roster_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(
            precision_block=precision(arms=robustness.ARMS[:4])))
    assert "a maximum over a different roster is a different claim" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(precision_block=precision(unestimable=("Cx",))))
    assert "a maximum over the rest of it" in str(refusal.value)


def test_one_population_per_support(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(candidates={"progen3-3b-bfloat16": [100]}))
    assert "so there is no one pool size behind these rows" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(dose_statistics={
            "progen3-112m-bfloat16": {"decision_statistic": "median per-residue KL"},
            "progen3-3b-bfloat16": {"decision_statistic": "mean per-residue KL"}}))
    assert "no one dose rule" in str(refusal.value)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(
            interval="95% percentile bootstrap over candidates, 2000 resamples"))
    assert "is not the BCa bootstrap over domains these rows copy" in str(refusal.value)


def test_the_permuted_column_contrast_is_read_in_every_stratum(tmp_path):
    emitted = rows(stage(tmp_path))
    for stratum in (*robustness.RUNGS, robustness.POOLED):
        prefix = f"progen3_robustness/primary_tm/{stratum}/{robustness.PERMUTED}"
        mean = emitted[f"{prefix}/true_over_permuted"]
        # The interval is copied with the unit, draw count and seed it was
        # computed under, and nothing is resampled here.
        assert mean.kind == "estimate" and mean.interval == (-0.0009, 0.0152)
        assert mean.interval_kind == robustness.BCA
        assert mean.resampling_unit == robustness.DOMAIN_UNIT
        assert mean.resampling_draws == 9999 and mean.seed_set == (20261013,)
        # Each end of that interval is a bound, never a second point estimate.
        for end, value in (("achieved_lower_bound", -0.0009), ("achieved_upper_bound", 0.0152)):
            bound = emitted[f"{prefix}/{end}"]
            assert bound.kind == "range_bound" and bound.interval is None
            assert bound.value == value
        assert emitted[f"{prefix}/domains"].value == len(DOMAINS)
        signs = [emitted[f"{prefix}/{name}"].value for name in
                 ("domains_favouring_true_columns", "domains_favouring_permuted_columns",
                  "domains_tied")]
        assert sum(signs) == len(DOMAINS) and signs == [5.0, 3.0, 0.0]
        shift = f"progen3_robustness/identity_shift/{stratum}/{robustness.PERMUTED}"
        for name in ("to_reference", "to_family_alignment"):
            assert emitted[f"{shift}/{name}"].unit == robustness.IDENTITY
            assert emitted[f"{shift}/{name}"].kind == "artifact_leaf"
        # The domain contributing most is read, not declared: the fixture puts it
        # at the last domain favouring the true columns.
        largest = DOMAINS[4]
        assert emitted[f"{prefix}/{largest}/diff"].value == 0.005
        assert emitted[f"{shift}/{largest}/to_reference"].value == 5.0
    # No row of this contrast carries a verdict, in any stratum.
    assert all(quantity.verdict is None for quantity in emitted.values())


def test_each_identity_superlative_is_named_where_it_belongs(tmp_path):
    emitted = rows(stage(tmp_path))
    stratum = robustness.POOLED
    shift = f"progen3_robustness/identity_shift/{stratum}/{robustness.PERMUTED}"
    largest, alignment = DOMAINS[4], DOMAINS[0]
    # The fixture's alignment identity rises most somewhere other than the domain
    # contributing most of the endpoint difference, so the alignment row names its
    # own domain and says it is not that one.
    claim = emitted[f"{shift}/to_family_alignment"].claim
    assert f"rise against the alignment in this stratum is in {alignment}" in claim
    assert f"endpoint difference ({largest} is)" in claim
    # The reference-identity row says, read off the artifact, that its own
    # superlative does fall on the contributing domain.
    assert "which is also the domain whose identity to the reference rises most" in \
        emitted[f"{shift}/{largest}/to_reference"].claim


def test_a_permuted_column_contrast_between_other_arms_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(permuted_arms=("Cx", "C")))
    assert "is not the permuted-column control" in str(refusal.value)


def test_a_contrast_record_without_its_identity_companions_records_a_gap(tmp_path):
    ledger = stage(tmp_path, robustness_payload(pooled=primary(
        **{robustness.PERMUTED: contrast(0.0038, interval=(-0.0009, 0.0152), positive=5,
                                         domain_block={}, identity=False)})))
    emitted = rows(ledger)
    prefix = f"progen3_robustness/primary_tm/{robustness.POOLED}/{robustness.PERMUTED}"
    # The contrast itself is still read; only the companions it no longer carries
    # become a gap, and no row is written off an absent field.
    assert f"{prefix}/true_over_permuted" in emitted
    assert not [identifier for identifier in emitted
                if identifier.startswith(f"progen3_robustness/identity_shift/"
                                         f"{robustness.POOLED}/")]
    gaps = {gap.id: gap for gap in ledger.gaps}
    gap = gaps[f"progen3_robustness/identity_shift/{robustness.POOLED}/{robustness.PERMUTED}"]
    assert "mean_identity_diff" in gap.reason and "per_domain" in gap.reason
    assert gap.reachability == "not_recorded" and gap.searched
    # The other three strata still carry theirs.
    for stratum in robustness.RUNGS:
        assert f"progen3_robustness/identity_shift/{stratum}/{robustness.PERMUTED}/to_reference" \
            in emitted


def test_two_domains_at_the_same_magnitude_leave_no_largest_contribution(tmp_path):
    tied = per_domain(DOMAINS, 5, overrides={
        DOMAINS[0]: {"diff": 0.005, "identity_diff": 1.0, "msa_identity_diff": 1.0}})
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, robustness_payload(pooled=primary(
            **{robustness.PERMUTED: contrast(0.0038, interval=(-0.0009, 0.0152), positive=5,
                                             domain_block=tied)})))
    assert "no domain is the largest single contribution" in str(refusal.value)


def test_the_retained_analysis_carries_the_numbers_the_section_quotes():
    artifacts = Artifacts()
    if not artifacts.exists(robustness.ROBUSTNESS):
        pytest.skip("the ProGen3 robustness analysis is not staged in this checkout")
    ledger = Ledger(artifacts)
    robustness.add_progen3_robustness_quantities(ledger)
    emitted = rows(ledger)
    pooled = f"progen3_robustness/primary_tm/{robustness.POOLED}"
    # "by 0.0072 (95\% CI, 0.0035--0.0136) TM with all eight domains positive".
    gain = emitted[f"{pooled}/profile_over_unguided"]
    assert round(gain.value, 4) == 0.0072
    assert tuple(round(bound, 4) for bound in gain.interval) == (0.0035, 0.0136)
    assert emitted[f"{pooled}/profile_over_unguided/domains_favouring_profile"].value == 8.0
    # "bounded above by $+0.0066$ TM".
    assert round(emitted[f"{pooled}/C-Pr/achieved_upper_bound"].value, 4) == 0.0066
    # "shifts the primary endpoint by at most 0.0042 TM".
    assert round(emitted[f"progen3_robustness/precision_control/"
                         f"{robustness.PRECISION_CHECKPOINT}/primary_tm/"
                         "largest_endpoint_shift"].value, 4) == 0.0042


def test_the_retained_analysis_carries_the_permuted_column_contrast_the_note_quotes():
    """Supplementary Note 14's figures, at the precision the note needs them.

    The 3B stratum is the one that separates true from permuted columns; the
    other three do not, and each of their intervals has to include zero, or the
    note's "neither 112M stratum nor the stratum pooled across checkpoints
    separates" is not what the artifact says.
    """
    artifacts = Artifacts()
    if not artifacts.exists(robustness.ROBUSTNESS):
        pytest.skip("the ProGen3 robustness analysis is not staged in this checkout")
    ledger = Ledger(artifacts)
    robustness.add_progen3_robustness_quantities(ledger)
    emitted = rows(ledger)
    separating = "progen3-3b-bfloat16"
    prefix = f"progen3_robustness/primary_tm/{separating}/{robustness.PERMUTED}"
    shift = f"progen3_robustness/identity_shift/{separating}/{robustness.PERMUTED}"
    mean = emitted[f"{prefix}/true_over_permuted"]
    assert round(mean.value, 6) == 0.005033
    assert tuple(round(bound, 6) for bound in mean.interval) == (0.000535, 0.015153)
    assert round(emitted[f"{prefix}/achieved_lower_bound"].value, 6) == 0.000535
    assert round(emitted[f"{prefix}/achieved_upper_bound"].value, 6) == 0.015153
    assert emitted[f"{prefix}/domains"].value == 8.0
    assert [emitted[f"{prefix}/{name}"].value for name in
            ("domains_favouring_true_columns", "domains_favouring_permuted_columns",
             "domains_tied")] == [5.0, 3.0, 0.0]
    assert round(emitted[f"{shift}/to_reference"].value, 3) == 1.874
    assert round(emitted[f"{shift}/to_family_alignment"].value, 3) == 4.222
    # 2LGW contributes most of the endpoint difference and carries the largest
    # rise in identity to the reference; the alignment identity rises most at
    # 2LVN, which is why the two superlatives are kept apart.
    assert round(emitted[f"{prefix}/2LGW/diff"].value, 6) == 0.028398
    assert round(emitted[f"{shift}/2LGW/to_reference"].value, 3) == 5.994
    assert "largest single-domain rise against the alignment in this stratum is in 2LVN" in \
        emitted[f"{shift}/to_family_alignment"].claim
    assert "is not the domain contributing most of the endpoint difference (2LGW is)" in \
        emitted[f"{shift}/to_family_alignment"].claim
    # The three strata the note reports as not separating each include zero.
    for stratum in ("progen3-112m-bfloat16", "progen3-112m-float16", robustness.POOLED):
        interval = emitted[f"progen3_robustness/primary_tm/{stratum}/"
                           f"{robustness.PERMUTED}/true_over_permuted"].interval
        assert interval[0] < 0.0 < interval[1]
