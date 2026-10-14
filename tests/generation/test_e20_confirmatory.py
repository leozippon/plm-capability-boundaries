"""E20 confirmatory analysis: the floor on real input, planted effects, frozen verdicts and the selector blind spot."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/capability/generation/analyse_e20_confirmatory.py"
spec = importlib.util.spec_from_file_location("analyse_e20_confirmatory", SCRIPT)
ana = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ana
spec.loader.exec_module(ana)

DEV_RECORDS = ROOT / "results/R6/e20_development_20261013/e20devc_measure_cpu/per_sequence.jsonl"
DEV_REFERENCES = ROOT / "data/pairwise_assets/family_references_jointh_20261013"
AA = "ACDEFGHIKLMNPQRSTVWY"
OFFSETS = np.linspace(-0.01, 0.01, 8)  # sum to zero, so the mean of a planted per-domain effect is the effect


def make_cell(model, domain, arm, target_tm, *, n_admissible=60, length=12, seed=0):
    lf, lc = ana.FROZEN_DOSES.get(arm, {}).get(model, (0.0, 0.0))
    rng = np.random.default_rng(abs(hash((model, domain, arm, seed))) % 2**32)
    rows = []
    for i in range(100):
        admissible = i < n_admissible
        rows.append({
            "model": model, "domain": domain, "arm": arm, "lambda_fields": lf, "lambda_coupling": lc,
            "sequence": "".join(rng.choice(list(AA), size=length)),
            "identity_to_wild_type": 10.0 if admissible else 40.0, "max_msa_identity": 15.0,
            "tm_score": target_tm if admissible else 0.9, "plddt_mean": 70.0, "ptm": 0.5,
            "native_loglik": float(rng.normal()), "composition_tv_to_wild_type": 0.3,
            "composition_tv_to_family": 0.2})
    return rows


def synthetic(effects, *, domains=ana.SEALED_DOMAINS, base=0.4, arms=("N", "Pr", "M", "C", "F", "Cx", "Cd"),
              model_effects=None):
    """Per-domain TM = base + offset (+ planted effect for the arm); ``effects`` maps arm -> planted shift.

    ``model_effects`` maps model -> effects and overrides ``effects`` for that model.
    """

    rows = []
    for model in ana.MODELS:
        planted = (model_effects or {}).get(model, effects)
        for k, domain in enumerate(domains):
            for arm in arms:
                rows += make_cell(model, domain, arm, base + OFFSETS[k % 8] + planted.get(arm, 0.0))
    return rows


def add_reference_arms(rows, domains=ana.SEALED_DOMAINS, *, p_effect=0.1, h_effect=0.2, p_model="potts-joint"):
    for k, domain in enumerate(domains):
        for arm, label, effect in (("P", p_model, p_effect), ("H", "natural", h_effect)):
            for r in make_cell(label, domain, arm, 0.4 + OFFSETS[k] + effect):
                r.update(lambda_fields=None, lambda_coupling=None, guidance_strength=None, native_loglik=None)
                rows.append(r)
    return rows


def run(rows, family_models=None):
    return ana.analyse(rows, family_models or {}, 7)


def test_development_records_are_refused_on_every_primary_contrast(tmp_path):
    out = tmp_path / "dev.json"
    assert ana.main(["--records", str(DEV_RECORDS), "--references", str(DEV_REFERENCES), "--out", str(out),
                     "--permutation-seed", "20261013"]) == 0
    result = json.loads(out.read_text())
    assert len(result["domains"]) == 4 < ana.MINIMUM_BOOTSTRAP_UNITS and not result["matches_sealed_eight"]
    for family in ("primary_tm", "coprimary_feasibility"):
        for stratum, block in result[family].items():
            for name, c in block["contrasts"].items():
                assert c["status"] == "unestimable", (stratum, name)
                assert c["interval95_bca"] is None and c["p_holm_ttest"] is None and c["p_holm_signflip"] is None, (stratum, name)
                assert c["n_domains"] <= 4
    pooled = result["primary_tm"]["pooled"]["contrasts"]
    assert pooled["C-Pr"]["n_domains"] == 4 and "4 domains" in pooled["C-Pr"]["reason"]
    assert len(pooled["C-Pr"]["per_domain"]) == 4  # the raw pattern stays visible
    assert pooled["C-Cx"]["n_domains"] == 0 and "no Cx cell" in pooled["C-Cx"]["reason"]
    assert result["equivalence"]["verdict"] == ana.VERDICT_UNESTIMABLE


def test_planted_effect_is_recovered_and_covered():
    result = run(synthetic({"C": 0.03, "Cx": 0.0, "Cd": 0.0, "Pr": 0.0, "N": -0.02, "F": 0.03, "M": 0.0}))
    c = result["primary_tm"]["pooled"]["contrasts"]["C-Pr"]
    assert c["status"] == "estimated" and c["n_domains"] == 8
    assert c["mean"] == pytest.approx(0.03)
    lo, hi = c["interval95_bca"]
    assert lo <= 0.03 <= hi
    assert c["favouring"] == {"positive": 8, "negative": 0, "tied": 0}
    assert c["mean_identity_diff"] == pytest.approx(0.0) and 0.0 <= c["p_holm_ttest"] <= 1.0
    cn = result["primary_tm"]["pooled"]["contrasts"]["C-N"]
    assert cn["mean"] == pytest.approx(0.05)
    assert result["primary_tm"]["pooled"]["holm_family_size"] == 8
    # the selection contrasts have no family model here: absent, never imputed
    assert result["primary_tm"]["pooled"]["contrasts"]["potts_full-potts_fields"]["status"] == "unestimable"


@pytest.mark.parametrize("effect, verdict", [(0.0, ana.VERDICT_CONCLUDE), (0.05, ana.VERDICT_INCONCLUSIVE)])
def test_verdict_at_the_margin(effect, verdict):
    result = run(synthetic({"C": effect, "F": effect}))
    eq = result["equivalence"]
    assert eq["verdict"] == verdict
    for model in ana.MODELS:
        upper = eq["strata"][model]["upper_bound"]
        assert (upper["C-Pr"] < ana.MARGIN) == (verdict == ana.VERDICT_CONCLUDE)
    assert eq["achieved_upper_bound"]["C-Pr"]["protgpt2"] == eq["strata"]["protgpt2"]["upper_bound"]["C-Pr"]
    assert verdict not in ("positive", "equivalent")


def test_zero_variance_effect_exactly_at_the_margin_is_not_below_it():
    rows = [r for r in synthetic({"C": 0.05, "F": 0.05}) if True]
    for r in rows:  # remove the per-domain spread so every difference is exactly 0.05
        if r["tm_score"] != 0.9:
            r["tm_score"] = 0.4 + (0.05 if r["arm"] in ("C", "F") else 0.0)
    result = run(rows)
    c = result["primary_tm"]["pooled"]["contrasts"]["C-Pr"]
    assert c["interval_zero_variance"] and c["mean"] == pytest.approx(0.05)
    assert result["equivalence"]["verdict"] == ana.VERDICT_INCONCLUSIVE


def test_one_bound_over_the_margin_is_enough_to_withhold():
    result = run(synthetic({"C": 0.0, "F": 0.05}))
    for stratum in ("pooled", *ana.MODELS):
        assert result["equivalence"]["strata"][stratum]["below_margin"] == {"C-Pr": True, "F-M": False}
    assert result["equivalence"]["verdict"] == ana.VERDICT_INCONCLUSIVE


def test_verdict_needs_both_models_and_pooled_cannot_mask_one():
    # protgpt2 carries the whole effect; the pooled mean is half of it and its bound clears the margin
    result = run(synthetic({}, model_effects={"progen2-medium": {"C": 0.0, "F": 0.0}, "protgpt2": {"C": 0.05, "F": 0.05}}))
    eq = result["equivalence"]
    assert eq["strata"]["pooled"]["below_margin"] == {"C-Pr": True, "F-M": True}
    assert eq["strata"]["progen2-medium"]["upper_bound"]["C-Pr"] < ana.MARGIN <= eq["strata"]["protgpt2"]["upper_bound"]["C-Pr"]
    assert eq["verdict"] == ana.VERDICT_INCONCLUSIVE
    assert set(eq["achieved_upper_bound"]["F-M"]) == {*ana.MODELS, "pooled"}


def test_unestimable_model_makes_the_verdict_unestimable():
    rows = [r for r in synthetic({}) if not (r["model"] == "protgpt2" and r["domain"] == "1E0L" and r["arm"] == "F")]
    assert run(rows)["equivalence"]["verdict"] == ana.VERDICT_UNESTIMABLE


def test_both_p_values_reported_and_sign_flip_resolution():
    result = run(synthetic({"C": 0.03, "F": 0.03, "N": -0.02}))
    block = result["primary_tm"]["progen2-medium"]["contrasts"]
    for c in block.values():
        if c["status"] == "estimated":
            assert c["p_holm_signflip"] >= 8 * 2 / 2**8 - 1e-12  # 2/256 raw floor, times the Holm multiplier 8
            assert c["interval_coverage_note"] and "0.89" in c["interval_coverage_note"]
    assert "0.0625" in result["frozen"]["p_value"] and "not on either p-value" in result["frozen"]["p_value"]
    assert ana.sign_flip_p([0.1] * 8) == pytest.approx(2 / 256)
    assert ana.sign_flip_p([0.1, -0.1, 0.1, -0.1]) == pytest.approx(1.0)
    assert ana.sign_flip_p([0.3, 0.1, -0.05]) == pytest.approx(4 / 8)  # +++ and ++- and their mirrors reach 0.1167


def test_reference_arms_keyed_on_labels_and_refused_otherwise():
    result = run(add_reference_arms(synthetic({})))
    ref = result["reference_arms_descriptive"]["pooled"]
    assert ref["P-N"]["status"] == "estimated" and ref["P-N"]["mean"] == pytest.approx(0.1)
    assert ref["H-N"]["mean"] == pytest.approx(0.2) and "p_holm_ttest" in ref["H-N"]
    assert ref["H-N"]["p_holm_ttest"] is None  # outside the Holm family
    with pytest.raises(ValueError, match="potts-joint"):
        run(add_reference_arms(synthetic({}), p_model="progen2-medium"))
    # H with fewer than 25 admissible rows in a domain drops that domain, which here leaves 7
    rows = add_reference_arms(synthetic({}))
    for r in rows:
        if r["arm"] == "H" and r["domain"] == "1IFY":
            r["identity_to_wild_type"] = 40.0
    assert run(rows)["reference_arms_descriptive"]["pooled"]["H-N"]["status"] == "unestimable"


def test_200_sequence_primary_cells_are_summarised_over_their_own_pool():
    rows = make_cell("protgpt2", "1E0L", "C", 0.4, n_admissible=60) + make_cell("protgpt2", "1E0L", "C", 0.4, n_admissible=40, seed=1)
    s = ana.summarise_cell(rows)
    assert s["n"] == 200 and s["feasibility"] == pytest.approx(0.5) and s["tm"] == pytest.approx(0.4)


def test_cell_with_24_admissible_is_missing_not_included():
    assert ana.summarise_cell(make_cell("protgpt2", "1E0L", "Pr", 0.4, n_admissible=24))["tm"] is None
    assert ana.summarise_cell(make_cell("protgpt2", "1E0L", "Pr", 0.4, n_admissible=25))["tm"] == pytest.approx(0.4)
    rows = [r for r in synthetic({}) if not (r["domain"] == "1IFY" and r["arm"] == "Pr")]
    rows += make_cell("progen2-medium", "1IFY", "Pr", 0.4, n_admissible=24) \
        + make_cell("protgpt2", "1IFY", "Pr", 0.4, n_admissible=24)
    result = run(rows)
    c = result["primary_tm"]["pooled"]["contrasts"]["C-Pr"]
    assert c["status"] == "unestimable" and c["n_domains"] == 7 and "1IFY" in c["missing_domains"]
    assert "24 admissible" in c["missing_domains"]["1IFY"]
    # feasibility is defined for the same cell, so the co-primary does not drop it
    assert result["coprimary_feasibility"]["pooled"]["contrasts"]["C-Pr"]["n_domains"] == 8
    sens = result["sensitivity_leave_out_1IFY_descriptive"]["pooled"]["C-Pr"]
    assert sens["status"] == "descriptive" and sens["n_domains"] == 7 and sens["p_holm_ttest"] is None
    assert sens["interval95_bca"] is not None


def test_pooled_needs_both_models():
    rows = [r for r in synthetic({}) if not (r["domain"] == "3DKM" and r["arm"] == "C" and r["model"] == "protgpt2")]
    result = run(rows)
    assert result["primary_tm"]["progen2-medium"]["contrasts"]["C-Pr"]["n_domains"] == 8
    pooled = result["primary_tm"]["pooled"]["contrasts"]["C-Pr"]
    assert pooled["n_domains"] == 7 and "3DKM" in pooled["missing_domains"]


@pytest.mark.parametrize("name", sorted(ana.FORBIDDEN_SELECTOR_FIELDS) + ["not_a_selector"])
def test_structural_fields_cannot_be_selectors(name):
    view = ana.selector_view(make_cell("protgpt2", "1E0L", "N", 0.4))
    with pytest.raises(ValueError):
        ana.selector_scores([name], view, None, 1)


def test_selector_view_carries_no_structural_field():
    view = ana.selector_view(make_cell("protgpt2", "1E0L", "N", 0.4))
    assert set(vars(view)) == {"domain", "sequences", "native_loglik"}


def make_model(domain, length, seed):
    rng = np.random.default_rng(seed)
    J = rng.normal(size=(length, length, 20, 20))
    J = J + J.transpose(1, 0, 3, 2)
    for i in range(length):
        J[i, i] = 0.0
    return SimpleNamespace(domain=domain, length=length, h=rng.normal(size=(length, 20)), J=J,
                           log_profile=np.log(rng.dirichlet(np.ones(20), size=length)))


def test_selection_ignores_structural_values_and_matches_hand_selection():
    domains = ana.SEALED_DOMAINS
    models = {d: make_model(d, 12, i) for i, d in enumerate(domains)}
    rows = synthetic({})
    base = run(rows, models)
    # structure-derived values are rewritten; the choice of candidates must not move
    shuffled = [{**r, "tm_score": r["tm_score"] + 0.01 * (i % 7), "plddt_mean": 99.0 - i % 5, "ptm": 0.1 * (i % 9)}
                for i, r in enumerate(rows)]
    other = run(shuffled, models)
    for stratum in ana.STRATA:
        for name in ("potts_full-potts_fields", "potts_coupling-coupling_permuted"):
            a, b = (x["primary_tm"][stratum]["contrasts"][name] for x in (base, other))
            assert a["status"] == "estimated" and a["n_domains"] == 8
            for d in domains:
                assert a["per_domain"][d]["identity_diff"] == b["per_domain"][d]["identity_diff"]
    # hand selection for one domain: top 10 admissible by potts_full
    domain, model = "1E0L", "protgpt2"
    pool = [r for r in rows if (r["model"], r["domain"], r["arm"]) == (model, domain, "N")
            and r["identity_to_wild_type"] < 20]
    view = ana.selector_view(pool)
    scores = ana.selector_scores(["potts_full", "potts_fields"], view, models[domain], 7)
    picked = {n: np.argsort(-s, kind="stable")[:10] for n, s in scores.items()}
    expected = np.mean([pool[i]["tm_score"] for i in picked["potts_full"]]) \
        - np.mean([pool[i]["tm_score"] for i in picked["potts_fields"]])
    got = base["primary_tm"][model]["contrasts"]["potts_full-potts_fields"]["per_domain"][domain]["diff"]
    assert got == pytest.approx(expected)


def test_holm_hand_computed():
    # sorted: .005 x4 = .02, .01 x3 = .03, .03 x2 = .06, .04 x1 = .04 -> carried up to .06
    assert ana.holm_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert ana.holm_adjust([0.9, 0.5]) == pytest.approx([1.0, 1.0])  # 0.5 x 2 = 1.0, then 0.9 is lifted to it
    # unestimable counts as p = 1 within a family of fixed size 4: multipliers 4, 3, then the rest
    assert ana.holm_adjust([0.01, None, 0.02], family_size=4) == pytest.approx([0.04, None, 0.06])


def test_missing_asset_names_the_option(tmp_path):
    run_ = subprocess.run([sys.executable, str(SCRIPT), "--records", str(tmp_path / "r.jsonl"),
                           "--out", str(tmp_path / "o.json"), "--permutation-seed", "1"],
                          capture_output=True, text=True)
    assert run_.returncode == 2 and "--references" in run_.stderr
    with pytest.raises(FileNotFoundError, match="--references"):
        ana.main(["--records", str(DEV_RECORDS), "--references", str(tmp_path / "nope"),
                  "--out", str(tmp_path / "o.json"), "--permutation-seed", "1"])
    with pytest.raises(FileNotFoundError, match="--records"):
        ana.main(["--records", str(tmp_path / "nope.jsonl"), "--references", str(DEV_REFERENCES),
                  "--out", str(tmp_path / "o.json"), "--permutation-seed", "1"])
