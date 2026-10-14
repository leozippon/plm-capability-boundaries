"""The phenotype rows R3 rests on, and the refusals that guard them.

R3 quotes four kinds of reduction the artifacts do not state as a single field:
how many models a phenotype cell resolves and what bounds it from above, the
range the resolved models span, the within-family calibrated reduction as a
share of the error that remains and as a kcal/mol difference, and the activity
band's half-width as a multiple of a beyond-profile stability gain. Each is a
statement about a set, so each refuses unless the artifacts cover that set: a
roll-up and a panel that disagree, a declared model roster that is not the
panel's columns, a stated resolution the bands do not give, a calibrated
baseline the per-arm fits do not share, and a quantitative label whose unit is
no longer the traced one are refusals rather than quieter numbers.

Two refusals carry the section's own distinctions. The calibrated share pairs a
numerator from the panel with a denominator from the fits, so each fit has to
reproduce its own published point or the share would pair two readings. And the
calibrated reading has to resolve for exactly the models the ranking reading
resolves for, because "the same 12 models" is the claim.

The last test runs on the retained artifacts and asserts the numbers the
manuscript quotes, at the precision it quotes them to.
"""
import json

import pytest

from scripts.capability.reporting import phenotype_quantities as phenotype
from src.capability.core.evidence_ledger import Artifacts, Ledger, LedgerError

ARMS = ("galactica-1.3b", "gpt2-large", "gpt2-xl", "instructprotein", "progen2-medium",
        "progen2-small", "progen2-xlarge", "prollama", "prollama-stage-1", "protgpt2")
SEEDS = (20260923, 20260924, 20260925)
#: Two cells that resolve nothing and one that resolves three models, which is
#: the shape the retained campaign has.
CELLS = {
    ("activity_venus", "ranking"): (89, ()),
    ("binding_skempi", "ranking"): (126, ()),
    ("binding_skempi", "quantitative"): (126, ()),
    ("cellular_fitness_sge", "ranking"): (9, ("instructprotein", "progen2-medium",
                                              "progen2-xlarge")),
}
GRID_ARMS = phenotype.PROTEIN_SPECIALIZED + phenotype.TEXT_AND_GALACTICA
#: The grid's resolved rosters, small enough to read and shaped like the panel:
#: the ranking and calibrated readings resolve on the same models.
RESOLVING = ("progen3-3b", "proteinglm-7b-clm", "protgpt2")
BASELINE = 0.5
#: Per-family calibrated increments, chosen so the group-equal mean of each
#: arm's list is its published point.
POINTS = {"progen3-3b": 0.02, "proteinglm-7b-clm": 0.015, "protgpt2": 0.005}
#: The beyond-profile stability survivors, two of which the activity cell also
#: scores: the ratio the manuscript quotes is over exactly that intersection.
BEYOND_PROFILE = ("progen2-xlarge", "protgpt2")


def arm_result(arm: str, point: float, half: float, *, resolved: bool = False) -> dict:
    return {"arm": arm, "point": point, "pointwise": [point - half / 3, point + half / 3],
            "pointwise_positive": resolved, "resolved_negative": False,
            "resolved_positive": resolved,
            "simultaneous": [point - half, point + half]}


def panel_payload(cohort: str, endpoint: str, groups: int, resolved: tuple[str, ...],
                  *, half: float = 0.024, baseline: float = 5.8) -> dict:
    results = []
    for index, arm in enumerate(ARMS):
        point = (0.03 if arm in resolved else 0.002 * index)
        results.append(arm_result(arm, point, half, resolved=arm in resolved))
    metric = "spearman" if endpoint == "ranking" else "error"
    return {
        "arms": list(ARMS), "arm_results": results, "cohort": cohort, "endpoint": endpoint,
        "metric": metric, "groups": [f"group-{index}" for index in range(groups)],
        "resolved_positive": len(resolved), "resolved_negative": 0, "status": "admitted",
        "phenotype": f"{cohort} phenotype", "endpoint_definition": f"the {cohort} label",
        "aggregation": "equal groups", "qualified_controls": ["ident", "geom"],
        "split_seeds": list(SEEDS),
        "support": {"independent_groups": groups, "backgrounds": groups, "units": groups,
                    "rows": 100 * groups, "source_rows": 200 * groups},
        "inference": {"method": "95th percentile maximum absolute centered deviation",
                      "draws": 10000, "groups": groups},
        "control_qualification": {
            "profile_blocks_qualified": ["prof", "prof2"] if endpoint == "ranking" else [],
            "target": phenotype.QUANTITATIVE_TARGET,
            "qualified_error_by_seed": {str(seed): baseline for seed in SEEDS},
        },
    }


def breadth_payload(panels: dict[tuple[str, str], dict]) -> dict:
    return {
        "schema": "phenotype_breadth_v1", "status": "complete", "arms": list(ARMS),
        "results": [{"cohort": cohort, "endpoint": endpoint, "status": panel["status"],
                     "groups": panel["support"]["independent_groups"],
                     "resolved_positive": panel["resolved_positive"],
                     "arm_results": panel["arm_results"]}
                    for (cohort, endpoint), panel in panels.items()],
    }


def matched_payload(*, resolved: bool = False) -> dict:
    contrasts = ["increment_mean", "increment_difference", "shared_residual_change"]
    return {
        "schema": "matched_phenotype_v1", "status": "complete", "arms": list(ARMS),
        "within_protein_partition": "mutated-position-held-out folds inside each protein",
        "matched_pair_counts": {"pooled": {"assay_pairs": 15, "independent_groups": 13,
                                           "matched_substitution_rows": 60002}},
        "pooled_inference": {
            "aggregation": "assay pairs averaged within protein, equal proteins",
            "contrast_family": contrasts,
            "results": [{"arm": arm, "contrast": contrast, "point": 0.001,
                         "resolved_positive": resolved and contrast == contrasts[0],
                         "resolved_negative": False}
                        for arm in ARMS for contrast in contrasts],
        },
    }


def grid_payload(*, resolving: tuple[str, ...] = RESOLVING,
                 profile: tuple[str, ...] = BEYOND_PROFILE,
                 calibrated: tuple[str, ...] | None = None,
                 text_resolves: str | None = None, groups: int = 3) -> dict:
    def family(resolved: tuple[str, ...]) -> dict:
        point, interval, verdicts = [], [], {}
        for arm in GRID_ARMS:
            above = arm in resolved
            value = POINTS.get(arm, 0.02) if above else 0.0005
            half = value / 2 if above else 0.01
            point.append(value)
            interval.append([value - half, value + half])
            verdicts[arm] = phenotype.RESOLVED_ABOVE if above else "unresolved"
        return {"status": "complete", "columns": list(GRID_ARMS), "undefined_families": {},
                "bands": {"point": point, "interval": interval, "family_size": len(GRID_ARMS),
                          "groups": groups, "draws": 10000},
                "verdicts": verdicts,
                "resolved_above_zero": sorted(resolved)}

    ranking = tuple(resolving) + ((text_resolves,) if text_resolves else ())
    families = {
        "matched_ranking": family(ranking),
        "profile_ranking": family(profile),
        "matched_calibrated_error": family(tuple(calibrated) if calibrated is not None
                                           else ranking),
        "profile_calibrated_error": family(profile),
        "matched_transfer_error": family(resolving[:1]),
        "profile_transfer_error": family(()),
    }
    return {"schema": "stability_ranking_error_panel_v2", "arms": list(GRID_ARMS),
            "families": families, "endpoint": "ddG in kcal/mol",
            "multiplicity": "one 33-column maximum-statistic family per contrast",
            "support": {"groups": groups, "sites": 5664, "rows": 25856,
                        "split_seeds": list(SEEDS)}}


def fit_payload(arm: str, point: float, *, baseline: float = BASELINE,
                groups: int = 3) -> dict:
    """One arm's fit, whose group-equal calibrated mean is its published point."""
    return {
        "arm": arm,
        "seeds": {str(seed): {"contrasts": {"combined": {
            "groups": [f"nat-{index}" for index in range(groups)],
            "baseline_calibrated_error": {"matched": [baseline] * groups,
                                          "profile": [baseline - 0.05] * groups},
            "matched_calibrated_error": [point] * groups,
        }}} for seed in SEEDS},
    }


def abundance_payload(*, resolved: tuple[str, ...] = ("progen2-xlarge", "instructprotein"),
                      inconsistent: str | None = None) -> dict:
    arms = {}
    for arm in GRID_ARMS:
        above = arm in resolved
        point = 0.005 if above else 0.0001
        excludes = above or arm == inconsistent
        arms[arm] = {"cells": {"primary_likelihood_spearman": {
            "licensed_for_a_verdict": True,
            "verdict": "above zero" if above else "unresolved",
            "three_seed_mean": point,
            "per_seed": {str(seed): {"point": point, "excludes_zero": excludes}
                         for seed in SEEDS},
        }}}
    return {"schema": "nested_gate_panel_v1", "roster_complete": True,
            "arms_fitted": len(GRID_ARMS), "arms": arms,
            "qualified_control_set": ["ident", "geom", "prof2"],
            "verdict_rule": "above zero only when all three per-seed intervals exclude zero",
            "effective_units": {"variants": 109568, "domains": 428, "groups": 96}}


def stage(tmp_path, *, panels: dict | None = None, breadth: dict | None = None,
          matched: dict | None = None, grid: dict | None = None, fits: dict | None = None,
          abundance: dict | None = None) -> Ledger:
    panels = panels if panels is not None else {
        key: panel_payload(key[0], key[1], groups, resolved)
        for key, (groups, resolved) in CELLS.items()}
    payloads = {phenotype.BREADTH: breadth if breadth is not None else breadth_payload(panels),
                phenotype.MATCHED: matched if matched is not None else matched_payload(),
                phenotype.GRID: grid if grid is not None else grid_payload(),
                phenotype.ABUNDANCE: abundance if abundance is not None
                else abundance_payload()}
    for key, panel in panels.items():
        payloads[phenotype.PANELS[key]] = panel
    published = payloads[phenotype.GRID]["families"]["matched_calibrated_error"]
    if fits is None:
        fits = {arm: fit_payload(arm, published["bands"]["point"][index],
                                 groups=payloads[phenotype.GRID]["support"]["groups"])
                for index, arm in enumerate(published["columns"])}
    for arm, fit in fits.items():
        payloads[f"{phenotype.GRID_FITS}/fit_{arm}.json"] = fit
    for relative, payload in payloads.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    ledger = Ledger(Artifacts(tmp_path))
    phenotype.add_phenotype_quantities(ledger)
    return ledger


def rows(ledger: Ledger) -> dict:
    return {quantity.id: quantity for quantity in ledger.quantities}


def test_each_cell_states_either_what_resolved_or_what_bounds_it(tmp_path):
    emitted = rows(stage(tmp_path))
    bounded = "phenotype_breadth/activity_venus/ranking"
    assert emitted[f"{bounded}/independent_groups"].value == 89.0
    assert emitted[f"{bounded}/scored_models"].value == float(len(ARMS))
    assert emitted[f"{bounded}/models_resolving_above_zero"].value == 0.0
    assert emitted[f"{bounded}/models_resolving_above_zero"].verdict == "unresolved"
    assert emitted[f"{bounded}/simultaneous_upper_bound"].kind == "range_bound"
    assert f"{bounded}/resolved_improvement/min" not in emitted
    resolving = "phenotype_breadth/cellular_fitness_sge/ranking"
    assert emitted[f"{resolving}/models_resolving_above_zero"].value == 3.0
    assert emitted[f"{resolving}/resolved_improvement/min"].kind == "range_bound"
    # A cell that resolves states the range it resolved over, not a bound its
    # own resolved models contradict.
    assert f"{resolving}/simultaneous_upper_bound" not in emitted
    assert emitted["phenotype_breadth/matched_proteingym/headline_contrasts"].value == 30.0
    assert emitted["phenotype_breadth/matched_proteingym/contrasts_resolving"].value == 0.0


def test_the_procedural_constants_these_panels_record_are_not_emitted(tmp_path):
    emitted = rows(stage(tmp_path))
    assert not [identifier for identifier in emitted
                if identifier.endswith(("draws", "seed", "split_seeds", "critical_value",
                                        "family_size", "minimum_units"))]
    # Nor the baseline levels the derived shares are taken over: a share of one
    # is quoted and the level itself is not, so it stays in the claim.
    assert not [identifier for identifier in emitted
                if "baseline_level" in identifier or identifier.endswith("qualified_error")]


def test_a_panel_that_is_not_the_roll_ups_reading_refuses(tmp_path):
    panels = {key: panel_payload(key[0], key[1], groups, resolved)
              for key, (groups, resolved) in CELLS.items()}
    breadth = breadth_payload(panels)
    moved = json.loads(json.dumps(panels[("activity_venus", "ranking")]))
    moved["arm_results"][0]["point"] += 0.01
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, panels={**panels, ("activity_venus", "ranking"): moved},
              breadth=breadth)
    assert "two readings of this cell" in str(refusal.value)


def test_a_cell_roster_this_module_does_not_read_refuses(tmp_path):
    panels = {key: panel_payload(key[0], key[1], groups, resolved)
              for key, (groups, resolved) in CELLS.items()}
    breadth = breadth_payload(panels)
    breadth["results"].append(dict(breadth["results"][0], cohort="another_cohort"))
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, panels=panels, breadth=breadth)
    assert "a bound over one roster is not a bound over the other" in str(refusal.value)


def test_a_stated_resolution_the_columns_do_not_give_refuses(tmp_path):
    panels = {key: panel_payload(key[0], key[1], groups, resolved)
              for key, (groups, resolved) in CELLS.items()}
    panels[("activity_venus", "ranking")]["resolved_positive"] = 1
    breadth = breadth_payload(panels)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, panels=panels, breadth=breadth)
    assert "resolutions and the stated count disagree" in str(refusal.value)


def test_a_quantitative_label_outside_its_traced_unit_refuses(tmp_path):
    panels = {key: panel_payload(key[0], key[1], groups, resolved)
              for key, (groups, resolved) in CELLS.items()}
    panels[("binding_skempi", "quantitative")]["control_qualification"]["target"] = \
        "an untraced label"
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, panels=panels, breadth=breadth_payload(panels))
    assert "has no stated unit" in str(refusal.value)


def test_a_cell_whose_half_width_differs_across_columns_states_no_ratio(tmp_path):
    panels = {key: panel_payload(key[0], key[1], groups, resolved)
              for key, (groups, resolved) in CELLS.items()}
    activity = panels[("activity_venus", "ranking")]
    activity["arm_results"][0]["simultaneous"] = [-0.05, 0.05]
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, panels=panels, breadth=breadth_payload(panels))
    assert "no one half-width to quote" in str(refusal.value)


def test_a_grid_whose_columns_are_not_the_declared_rosters_refuses(tmp_path):
    grid = grid_payload()
    grid["arms"] = grid["arms"][:-1]
    for family in grid["families"].values():
        family["columns"] = grid["arms"]
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, grid=grid)
    assert "are not the same set" in str(refusal.value)


def test_a_stated_resolved_roster_the_bands_do_not_give_refuses(tmp_path):
    grid = grid_payload()
    grid["families"]["matched_ranking"]["resolved_above_zero"] = ["gpt2"]
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, grid=grid)
    assert "as resolved above zero and its bands give" in str(refusal.value)


def test_a_text_or_galactica_arm_that_resolves_refuses_the_bounded_negative(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, grid=grid_payload(text_resolves="galactica-6.7b"))
    assert "no longer a bounded negative" in str(refusal.value)


def test_two_readings_resolving_on_different_models_refuse_the_same_models_claim(tmp_path):
    grid = grid_payload(calibrated=RESOLVING[:2])
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, grid=grid)
    assert "they are not the same models" in str(refusal.value)


def test_a_calibrated_baseline_the_arms_do_not_share_refuses(tmp_path):
    grid = grid_payload()
    published = grid["families"]["matched_calibrated_error"]
    fits = {arm: fit_payload(arm, published["bands"]["point"][index],
                             baseline=BASELINE + (0.1 if arm == "gpt2" else 0.0))
            for index, arm in enumerate(GRID_ARMS)}
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, grid=grid, fits=fits)
    assert "is not one level across the arms" in str(refusal.value)


def test_a_fit_that_does_not_reproduce_its_published_point_refuses(tmp_path):
    grid = grid_payload()
    published = grid["families"]["matched_calibrated_error"]
    fits = {arm: fit_payload(arm, published["bands"]["point"][index] +
                             (0.001 if arm == "progen3-3b" else 0.0))
            for index, arm in enumerate(GRID_ARMS)}
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, grid=grid, fits=fits)
    assert "the share would pair two readings" in str(refusal.value)


def test_an_abundance_verdict_its_own_splits_contradict_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, abundance=abundance_payload(inconsistent="gpt2"))
    assert "resolved splits disagree" in str(refusal.value)


def test_no_model_carrying_both_readings_states_no_precision_ratio(tmp_path):
    panels = {key: panel_payload(key[0], key[1], groups, resolved)
              for key, (groups, resolved) in CELLS.items()}
    # Every beyond-profile stability survivor is outside the activity cell's
    # scored roster, so the activity bound cannot be read against a gain.
    grid = grid_payload(profile=("progen2-base", "progen2-large"))
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, panels=panels, breadth=breadth_payload(panels), grid=grid)
    assert "cannot be read against one" in str(refusal.value)


def test_the_derived_shares_follow_the_arithmetic_their_claims_state(tmp_path):
    emitted = rows(stage(tmp_path))
    share = emitted["stability_grid/matched_calibrated_error/resolved_share_of_remaining_error/max"]
    assert share.unit == "percent"
    assert round(share.value, 6) == round(100.0 * POINTS["progen3-3b"] / BASELINE, 6)
    drop = emitted["stability_grid/matched_calibrated_error/best_model_error_drop"]
    assert drop.unit == "kcal/mol"
    assert round(drop.value, 9) == round(BASELINE ** 0.5 - (BASELINE - POINTS["progen3-3b"]) ** 0.5,
                                         9)
    ratio = emitted["phenotype_breadth/activity_venus/ranking/half_width_over_stability_gain/max"]
    assert ratio.unit == "dimensionless ratio"
    assert ratio.kind == "range_bound"
    bound = emitted["phenotype_breadth/binding_skempi/quantitative/"
                    "upper_bound_share_of_baseline_error"]
    assert bound.kind == "ratio" and bound.unit == "percent"
    tracks = emitted["stability_grid/calibrated_error_tracks_ranking"]
    assert tracks.kind == "estimate" and tracks.interval is None
    assert tracks.no_interval_reason


def test_the_retained_artifacts_carry_the_numbers_the_manuscript_quotes():
    artifacts = Artifacts()
    needed = [phenotype.BREADTH, phenotype.MATCHED, phenotype.GRID, phenotype.ABUNDANCE,
              *phenotype.PANELS.values()]
    for relative in needed:
        if not artifacts.exists(relative):
            pytest.skip("the phenotype campaign artifacts are not staged in this checkout")
    ledger = Ledger(artifacts)
    phenotype.add_phenotype_quantities(ledger)
    emitted = rows(ledger)

    def quoted(identifier: str, places: int) -> float:
        return round(emitted[identifier].value, places)

    activity = "phenotype_breadth/activity_venus/ranking"
    assert emitted[f"{activity}/scored_models"].value == 10.0
    assert emitted[f"{activity}/independent_groups"].value == 89.0
    assert emitted[f"{activity}/models_resolving_above_zero"].value == 0.0
    assert quoted(f"{activity}/simultaneous_upper_bound", 3) == 0.046
    assert (quoted(f"{activity}/half_width_over_stability_gain/min", 0),
            quoted(f"{activity}/half_width_over_stability_gain/max", 0)) == (7.0, 12.0)

    binding = "phenotype_breadth/binding_skempi"
    assert emitted[f"{binding}/ranking/independent_groups"].value == 126.0
    assert emitted[f"{binding}/ranking/models_resolving_above_zero"].value == 0.0
    assert quoted(f"{binding}/ranking/simultaneous_upper_bound", 3) == 0.051
    assert emitted[f"{binding}/quantitative/models_resolving_above_zero"].value == 0.0
    assert quoted(f"{binding}/quantitative/simultaneous_upper_bound", 3) == 0.016
    assert quoted(f"{binding}/quantitative/upper_bound_share_of_baseline_error", 2) == 0.28

    fitness = "phenotype_breadth/cellular_fitness_sge/ranking"
    assert emitted[f"{fitness}/models_resolving_above_zero"].value == 3.0
    assert emitted[f"{fitness}/independent_groups"].value == 9.0
    assert (quoted(f"{fitness}/resolved_improvement/min", 3),
            quoted(f"{fitness}/resolved_improvement/max", 3)) == (0.021, 0.035)
    assert emitted["phenotype_breadth/matched_proteingym/headline_contrasts"].value == 30.0
    assert emitted["phenotype_breadth/matched_proteingym/contrasts_resolving"].value == 0.0

    assert emitted["stability_grid/arms"].value == 33.0
    assert emitted["stability_grid/protein_specialized_or_adapted_arms"].value == 14.0
    assert emitted["stability_grid/matched_ranking/models_resolving_above_zero"].value == 12.0
    assert (quoted("stability_grid/matched_ranking/resolved_improvement/min", 3),
            quoted("stability_grid/matched_ranking/resolved_improvement/max", 3)) == (0.005, 0.017)
    assert emitted["stability_grid/profile_ranking/models_resolving_above_zero"].value == 4.0
    assert (quoted("stability_grid/profile_ranking/resolved_improvement/min", 3),
            quoted("stability_grid/profile_ranking/resolved_improvement/max", 3)) == (0.002, 0.007)
    assert emitted["stability_grid/matched_calibrated_error/"
                   "models_resolving_above_zero"].value == 12.0
    assert (quoted("stability_grid/matched_calibrated_error/"
                   "resolved_share_of_remaining_error/min", 1),
            quoted("stability_grid/matched_calibrated_error/"
                   "resolved_share_of_remaining_error/max", 1)) == (1.0, 3.5)
    assert quoted("stability_grid/matched_calibrated_error/best_model_error_drop", 3) == 0.013
    assert quoted("stability_grid/calibrated_error_tracks_ranking", 3) == 0.994
    assert emitted["stability_grid/matched_transfer_error/models_resolving_above_zero"].value == 3.0
    assert emitted["stability_grid/profile_transfer_error/"
                   "models_resolving_above_zero"].value == 0.0
    assert quoted("stability_grid/profile_transfer_error/largest_half_width", 3) == 0.078
    assert quoted("stability_grid/matched_ranking/text_and_galactica_upper_bound", 4) == 0.0021

    assert (quoted("domainome/primary_ranking/resolved_improvement/min", 3),
            quoted("domainome/primary_ranking/resolved_improvement/max", 3)) == (0.001, 0.008)
