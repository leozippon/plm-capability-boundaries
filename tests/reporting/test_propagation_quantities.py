"""The position-propagation rows R2 rests on, and the refusals that guard them.

Three of R2's claims are reductions over a set of checkpoints the artifacts do
not name: the exact-zero upstream contribution over the causal arms, the decay
half-distance over the protein-only arms whose fit is read, and the simultaneous
half-width over the protein-only columns. A reduction over a set is only
statable when the artifact covers the set, so a missing arm, a column that is
neither measured nor named as excluded, and a second arm breaking the exact zero
are refusals rather than quieter numbers. The declared decay roster carries one
refusal of its own: it has to be the well-fitted end of the artifact's own fit
ordering, or the half-distance range is a hand-picked subset.

The last test runs on the retained artifacts and asserts the numbers the
manuscript quotes, at the precision it quotes them to.
"""
import json

import pytest

from scripts.capability.reporting import propagation_quantities as propagation
from src.capability.core.evidence_ledger import Artifacts, Ledger, LedgerError

CAUSAL = ("progen2-base", "progen2-large", "progen2-medium", "progen2-small", "progen2-xlarge",
          "progen3-112m", "proteinglm-7b-clm", "protgpt2", "rita-xl", "gpt2-xl")
BIDIRECTIONAL = ("esm2-650m", "dplm-650m")
STRUCTURES = 6


def assays(structural: int = STRUCTURES) -> list[dict]:
    rows = []
    for index in range(8):
        rows.append({"assay": f"assay-{index}", "family": index, "residues": 37 + index * 7,
                     "selected_single_substitutions": 11,
                     "structural_receivers": 400 if index < structural else 0})
    return rows


def profile(half: float, r_squared: float, *, response: float = 0.6) -> dict:
    return {
        "upstream": {"direction": "upstream", "receivers": 1000, "max_separation": 90,
                     "decay": {"fitted": False, "half_distance_residues": None,
                               "half_distance_undefined_reason": "the profile was not fitted",
                               "half_distance_interval_residues": None}},
        "downstream": {
            "direction": "downstream", "receivers": 2000, "max_separation": 120,
            "separations": [{"separation": separation, "n": 500, "sufficient": True,
                             "mean_absolute_nats": response / separation}
                            for separation in (1, 2, 20)],
            "decay": {"fitted": True, "r_squared": r_squared, "half_distance_residues": half,
                      "half_distance_undefined_reason": None,
                      "half_distance_interval_residues": [half - 1.0, half + 1.0]},
        },
    }


def arm(name: str, *, half: float, r_squared: float, exact: bool = True,
        paradigm: str = "causal_next_token", structural: int = STRUCTURES) -> dict:
    return {"arm": name, "paradigm": paradigm, "assays": assays(structural),
            "upstream_is_exactly_zero": exact,
            "upstream_max_abs_nats": 0.0 if exact else 3.1e-5,
            "profile": profile(half, r_squared)}


#: Fit quality by arm: the three arms the roster leaves out fit worst, which is
#: the separation the emitter refuses without.
FITS = {"progen2-base": 0.96, "progen2-large": 0.97, "progen2-medium": 0.98,
        "progen2-small": 0.95, "progen2-xlarge": 0.89, "proteinglm-7b-clm": 0.91,
        "rita-xl": 0.94, "progen3-112m": 0.57, "protgpt2": 0.27, "gpt2-xl": 0.79}
HALVES = {"progen2-base": 18.5, "progen2-large": 20.9, "progen2-medium": 19.2,
          "progen2-small": 28.8, "progen2-xlarge": 20.2, "proteinglm-7b-clm": 21.3,
          "rita-xl": 19.3, "progen3-112m": 83.2, "protgpt2": 26.3, "gpt2-xl": 35.1}


def column(name: str, *, direction: str = "downstream", point: float = 0.02,
           error: float = 0.02, resolves: bool = False) -> dict:
    return {"arm": name, "direction": direction, "families": 6, "point_nats": point,
            "pointwise_interval_nats": [point - 2 * error, point + 2 * error],
            "simultaneous_interval_nats": [point - 3 * error, point + 3 * error],
            "simultaneously_excludes_zero": resolves, "standard_error_nats": error}


def propagation_payload(*, arms: list[dict] | None = None, columns: list[dict] | None = None,
                        excluded: list[dict] | None = None, universe: int = STRUCTURES,
                        causal_only: bool = False) -> dict:
    """One propagation cell. The unconstrained cell holds no masked arm."""
    if arms is None:
        # One causal arm's upstream response is not exactly zero, as RITA-XL's is
        # not in the retained cell; every other causal arm's is.
        arms = [arm(name, half=HALVES[name], r_squared=FITS[name], exact=name != "rita-xl")
                for name in CAUSAL]
        if not causal_only:
            arms += [arm(name, half=45.0, r_squared=0.88, exact=False,
                         paradigm="masked_marginal_pseudolikelihood") for name in BIDIRECTIONAL]
    if columns is None:
        columns = [column(name) for name in CAUSAL if name != "protgpt2"]
        if not causal_only:
            columns += [column(name, point=0.10, resolves=True) for name in BIDIRECTIONAL] + \
                       [column(name, direction="upstream", point=0.05) for name in BIDIRECTIONAL]
    if excluded is None:
        excluded = [{"arm": "protgpt2", "direction": "downstream", "families": 1,
                     "reason": "below the five-family floor"}]
    family = {"critical_value": 3.13, "n_columns": len(columns), "columns": columns,
              "excluded_columns": excluded, "family_universe": universe, "draws": 10000,
              "confidence": 0.95, "outcome": "absolute_response"}
    return {"experiment": "E01", "status": "complete", "arms": arms,
            "settings": {"max_residues": 128},
            "panel": {"analysed_arms": len(arms)},
            "panel_simultaneous": {"absolute_response": family,
                                   "response": dict(family, critical_value=3.11,
                                                     outcome="response")}}


def anticipation_payload() -> dict:
    def band(point: float, low: float, high: float) -> dict:
        return {"point": point, "interval": [low, high], "groups": 6, "undefined_groups": 0,
                "excludes_zero": low > 0 or high < 0}
    strata = {"buried": {"contrast_nats": band(0.264, 0.109, 0.431)},
              "exposed": {"contrast_nats": band(-0.001, -0.087, 0.099)}}
    result = {"anchors": 100, "contact_pairs": 300, "control_pairs": 1200,
              "assays": [f"assay-{index}" for index in range(STRUCTURES)],
              "families": [str(index) for index in range(STRUCTURES)],
              "weighting": "families are the bootstrap units",
              "sources": {"model_conditional": {"anchor_rsa_strata": strata}}}
    return {"experiment": "E02", "settings": {"draws": 2000},
            "arms": [{"arm": "progen2-small", "paradigm": "causal_next_token", "result": result},
                     {"arm": "protgpt2", "paradigm": "causal_next_token", "result": None}]}


def forcing_payload(*, thresholds: tuple[float, ...] = (0.054, 0.054),
                    columns: tuple[str, ...] = ("progen2-large", "rita-xl")) -> dict:
    return {
        "pre_registration": {"arms": ["progen2-large", "rita-xl"]},
        "arms": [{"arm": name, "verdict": {"pre_registered_claimable_nats": threshold}}
                 for name, threshold in zip(("progen2-large", "rita-xl"), thresholds)],
        "panel": {"simultaneous": {
            "backbones": 32, "estimand": "equal-weight mean of within-band backbone means",
            "columns": [{"column": name, "point_nats": 0.0123 if index == 0 else -0.0111,
                         "simultaneous_interval_nats": [-0.01, 0.034],
                         "standard_error_nats": 0.009, "simultaneously_excludes_zero": False}
                        for index, name in enumerate(columns)]}},
    }


def stage(tmp_path, *, causal: dict | None = None, shared: dict | None = None,
          anticipation: dict | None = None, forcing: dict | None = None,
          receiver: dict | None = None) -> Ledger:
    payloads = {
        propagation.CAUSAL512: causal if causal is not None
        else propagation_payload(causal_only=True),
        propagation.SHARED128: shared if shared is not None else propagation_payload(),
        propagation.ANTICIPATION: anticipation if anticipation is not None
        else anticipation_payload(),
        propagation.FORCING: forcing if forcing is not None else forcing_payload(),
    }
    frozen = payloads[propagation.SHARED128]
    if receiver is None:
        receiver = dict(frozen, panel_simultaneous_receiver_rsa_matched={
            outcome: dict(block, columns=[dict(entry,
                                               point_nats=entry["point_nats"] + 0.015)
                                          for entry in block["columns"]])
            for outcome, block in frozen["panel_simultaneous"].items()})
    payloads[propagation.RECEIVER_RSA] = receiver
    for relative, payload in payloads.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    ledger = Ledger(Artifacts(tmp_path))
    propagation.add_propagation_quantities(ledger)
    return ledger


def rows(ledger: Ledger) -> dict:
    return {quantity.id: quantity for quantity in ledger.quantities}


def test_the_declared_reductions_come_from_the_sets_the_artifacts_cover(tmp_path):
    emitted = rows(stage(tmp_path))
    assert emitted["position_propagation/causal512/causal_arms"].value == float(len(CAUSAL))
    assert emitted["position_propagation/causal512/arms_with_exactly_zero_upstream"].value == \
        float(len(CAUSAL) - 1)
    # One receiver block per causal arm, summed over the arms that carry an exact zero.
    assert emitted["position_propagation/causal512/exactly_zero_upstream_receivers"].value == \
        float(1000 * (len(CAUSAL) - 1))
    assert "position_propagation/causal512/rita-xl/upstream_max_abs_nats" in emitted
    readable = [name for name in propagation.PROTEIN_ONLY
                if name not in propagation.UNREADABLE_DECAY]
    assert emitted["position_propagation/shared128/protein_only/half_distance/min"].value == \
        min(HALVES[name] for name in readable)
    assert emitted["position_propagation/shared128/protein_only/half_distance/max"].value == \
        max(HALVES[name] for name in readable)
    # ProtGPT2 is named as excluded from the band, so the half-width range is
    # over the eight protein-only columns that exist.
    assert emitted["position_propagation/shared128/contact/causal_columns"].value == \
        float(len(CAUSAL) - 1)
    assert emitted["position_propagation/shared128/contact/causal_columns_resolving"].value == 0.0


def test_the_procedural_constants_these_cells_record_are_not_emitted(tmp_path):
    emitted = rows(stage(tmp_path))
    assert not [identifier for identifier in emitted
                if identifier.endswith(("draws", "seed", "confidence", "family_universe",
                                        "minimum_families", "permutation_draws"))]
    # The band's critical value is emitted: the supplement quotes it and the
    # detectable-effect reduction is computed from it.
    assert "position_propagation/shared128/contact/critical_value" in emitted


def test_an_artifact_that_does_not_cover_the_protein_only_roster_refuses(tmp_path):
    payload = propagation_payload()
    payload["arms"] = [entry for entry in payload["arms"] if entry["arm"] != "rita-xl"]
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, causal=payload)
    assert "protein-only roster is not covered" in str(refusal.value)


def test_a_decay_roster_that_is_not_the_well_fitted_arms_refuses(tmp_path):
    # ProGen3-112M now fits better than an admitted arm, so the declared roster
    # no longer reads off the artifact's own fit ordering.
    fits = dict(FITS)
    fits["progen3-112m"] = 0.93
    payload = propagation_payload(arms=[
        arm(name, half=HALVES[name], r_squared=fits[name], exact=name != "rita-xl")
        for name in CAUSAL])
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, shared=payload, receiver=payload)
    assert "well-fitted arms" in str(refusal.value)


def test_a_second_arm_breaking_the_exact_zero_refuses(tmp_path):
    payload = propagation_payload(arms=[
        arm(name, half=HALVES[name], r_squared=FITS[name],
            exact=name not in ("rita-xl", "gpt2-xl"))
        for name in CAUSAL])
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, causal=payload)
    assert "one exception among the causal arms" in str(refusal.value)


def test_a_protein_only_arm_neither_measured_nor_named_refuses(tmp_path):
    payload = propagation_payload(excluded=[])
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, shared=payload, receiver=payload)
    assert "neither a column nor a named exclusion" in str(refusal.value)


def test_a_structural_population_that_is_not_the_bands_universe_refuses(tmp_path):
    payload = propagation_payload(universe=STRUCTURES + 1)
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, shared=payload, receiver=payload)
    assert "not the same support" in str(refusal.value)


def test_a_sensitivity_whose_primary_block_moved_refuses(tmp_path):
    frozen = propagation_payload()
    drifted = json.loads(json.dumps(frozen))
    drifted["panel_simultaneous"]["absolute_response"]["columns"][0]["point_nats"] += 0.004
    drifted["panel_simultaneous_receiver_rsa_matched"] = \
        drifted["panel_simultaneous"]
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, shared=frozen, receiver=drifted)
    assert "not the frozen panel's" in str(refusal.value)


def test_a_forcing_panel_missing_a_pre_registered_arm_refuses(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, forcing=forcing_payload(columns=("progen2-large", "progen2-medium")))
    assert "carry no simultaneous column" in str(refusal.value)


def test_forcing_arms_disagreeing_on_the_threshold_refuse(tmp_path):
    with pytest.raises(LedgerError) as refusal:
        stage(tmp_path, forcing=forcing_payload(thresholds=(0.054, 0.05)))
    assert "disagree on the pre-registered claimable effect" in str(refusal.value)


def test_the_retained_artifacts_carry_the_numbers_the_manuscript_quotes():
    artifacts = Artifacts()
    for relative in (propagation.CAUSAL512, propagation.SHARED128, propagation.ANTICIPATION,
                     propagation.FORCING, propagation.RECEIVER_RSA):
        if not artifacts.exists(relative):
            pytest.skip("the position-propagation artifacts are not staged in this checkout")
    ledger = Ledger(artifacts)
    propagation.add_propagation_quantities(ledger)
    emitted = rows(ledger)

    def quoted(identifier: str, places: int) -> float:
        return round(emitted[identifier].value, places)

    assert emitted["position_propagation/causal512/causal_arms"].value == 19.0
    assert emitted["position_propagation/causal512/arms_with_exactly_zero_upstream"].value == 18.0
    assert emitted["position_propagation/causal512/exactly_zero_upstream_receivers"].value == \
        19590844.0
    assert (quoted("position_propagation/causal512/protein_only/response_at_separation_1/min", 1),
            quoted("position_propagation/causal512/protein_only/response_at_separation_1/max", 1)) \
        == (0.5, 0.9)
    assert quoted("position_propagation/causal512/protein_only/response_at_separation_20/max", 1) \
        == 0.2
    assert (quoted("position_propagation/shared128/protein_only/half_distance/min", 1),
            quoted("position_propagation/shared128/protein_only/half_distance/max", 1)) \
        == (18.5, 28.8)
    assert (quoted("position_propagation/causal512/protein_only/half_distance/min", 0),
            quoted("position_propagation/causal512/protein_only/half_distance/max", 0)) \
        == (100.0, 124.0)
    assert emitted["position_propagation/shared128/contact/causal_columns"].value == 18.0
    assert emitted["position_propagation/shared128/contact/causal_columns_resolving"].value == 0.0
    assert quoted("position_propagation/shared128/contact/critical_value", 3) == 3.132
    assert (quoted("position_propagation/shared128/contact/protein_only/half_width/min", 3),
            quoted("position_propagation/shared128/contact/protein_only/half_width/max", 3)) \
        == (0.055, 0.1)
    assert (quoted("position_propagation/shared128/contact/protein_only/"
                   "detectable_at_80_power/min", 3),
            quoted("position_propagation/shared128/contact/protein_only/"
                   "detectable_at_80_power/max", 3)) == (0.07, 0.127)
    assert (quoted("position_propagation/shared128/contact/bidirectional_downstream/min", 2),
            quoted("position_propagation/shared128/contact/bidirectional_downstream/max", 2)) \
        == (0.06, 0.13)
    assert (emitted["position_propagation/shared128/structural_support/domain_residues/min"].value,
            emitted["position_propagation/shared128/structural_support/domain_residues/max"].value) \
        == (37.0, 72.0)
    assert emitted["position_propagation/shared128/structural_support/"
                   "largest_possible_contact_separation"].value == 71.0
    assert quoted("position_propagation/receiver_rsa/protein_only/largest_estimate_shift", 2) == 0.02

    buried = emitted["contact_anticipation/progen2-small/buried_anchors"]
    assert round(buried.value, 3) == 0.264
    assert tuple(round(bound, 3) for bound in buried.interval) == (0.109, 0.431)
    assert buried.resampling_draws == 2000 and buried.resampling_unit
    exposed = emitted["contact_anticipation/progen2-small/exposed_anchors"]
    assert round(exposed.value, 3) == -0.001
    assert exposed.interval[0] < 0.0 < exposed.interval[1]

    assert quoted("forcing_gate/largest_partner_specific_shift", 3) == 0.012
    assert emitted["forcing_gate/pre_registered_claimable_effect"].value == 0.054
