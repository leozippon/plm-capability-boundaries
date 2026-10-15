"""ProGen3 robustness bounds: the contrast arithmetic, the eight-domain floor, every refusal, the selector blind spot.

The fixtures carry the campaign's real shape -- three rungs over two measure lines, five arms, the sealed eight
domains, one dose per (rung, arm) -- at a smaller pool depth, and they use the real family references so the
selection machinery is exercised against real Potts fits rather than a mock.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/capability/generation/analyse_progen3_robustness.py"
spec = importlib.util.spec_from_file_location("analyse_progen3_robustness", SCRIPT)
ana = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ana
spec.loader.exec_module(ana)
e20 = ana.e20

REFERENCES = ROOT / "data/pairwise_assets/family_references_jointh_20261013"
DOMAINS = e20.SEALED_DOMAINS
ARMS = ana.ARMS
AA = "ACDEFGHIKLMNPQRSTVWY"
POOL = 40
ADMISSIBLE = 30
DEVELOPMENT = ("1K1V", "1UFM", "2KWH", "3L1X")

#: ``rung -> arm -> (lambda_fields, lambda_coupling)``, in the campaign's shape: the float16 control runs at
#: the doses solved for its own bfloat16 twin.
DOSES = {
    "progen3-3b-bfloat16": {"N": (0.0, 0.0), "Pr": (0.1874, 0.0), "M": (0.2035, 0.0),
                            "C": (0.0, 2.491), "Cx": (0.0, 3.299)},
    "progen3-112m-bfloat16": {"N": (0.0, 0.0), "Pr": (0.1741, 0.0), "M": (0.1794, 0.0),
                              "C": (0.0, 1.562), "Cx": (0.0, 2.652)},
    "progen3-112m-float16": {"N": (0.0, 0.0), "Pr": (0.1741, 0.0), "M": (0.1794, 0.0),
                             "C": (0.0, 1.562), "Cx": (0.0, 2.652)},
}
LINES = {"bfloat16": ("progen3-3b-bfloat16", "progen3-112m-bfloat16"),
         "float16": ("progen3-112m-float16",)}
#: Per-domain offsets that sum to zero, so a planted per-arm effect is exactly the mean over domains however
#: the arms' offsets are rolled against each other. Rolling per arm makes the per-domain difference vary, so
#: the BCa interval is exercised rather than collapsing onto a point.
OFFSETS = np.linspace(-0.01, 0.01, len(DOMAINS))
OFFSETS = OFFSETS - OFFSETS.mean()
ARM_OFFSETS = {arm: np.roll(OFFSETS, i) for i, arm in enumerate(ARMS)}
#: Admissible candidates per cell: a base per arm plus a per-arm, per-domain shift, so the second endpoint
#: varies by arm and by domain and never falls below the 25-candidate floor that would drop the cell.
ARM_ADMISSIBLE = {"N": 33, "Pr": 31, "M": 32, "C": 29, "Cx": 30}
#: Added to the float16 rung's TM in every arm, so the precision difference varies by domain and still has the
#: planted shift as its mean. Within-rung arm contrasts cancel it.
PRECISION_JITTER = np.roll(OFFSETS, 3) * 0.5


def columns() -> dict[str, int]:
    return {domain: json.loads((REFERENCES / f"{domain}.json").read_text())["length"] for domain in DOMAINS}


COLUMNS = columns()


# ----------------------------------------------------------------- fixtures


def make_cell(rung, domain, arm, tm, *, n_admissible=ADMISSIBLE, pool=POOL, seed=0, dose=None):
    lf, lc = DOSES[rung][arm] if dose is None else dose
    rng = np.random.default_rng(abs(hash((rung, domain, arm, seed))) % 2**32)
    rows = []
    for i in range(pool):
        admissible = i < n_admissible
        rows.append({
            "id": f"{domain}_{rung}_{arm}_{i}", "model": rung, "domain": domain, "arm": arm,
            "lambda_fields": lf, "lambda_coupling": lc,
            "sequence": "".join(rng.choice(list(AA), size=COLUMNS[domain])),
            "identity_to_wild_type": 10.0 if admissible else 40.0, "max_msa_identity": 15.0,
            "tm_score": tm if admissible else 0.9, "plddt_mean": 70.0, "ptm": 0.5,
            "native_loglik": float(rng.normal()),
            "composition_tv_to_wild_type": 0.3, "composition_tv_to_family": 0.2})
    return rows


def records_for(rung, effects, *, base=0.4, domains=DOMAINS, arms=ARMS, pool=POOL, jitter=None, flat=False):
    """One rung's 40 cells. The planted per-arm effect is exactly the mean over domains of every contrast.

    ``flat`` drops the per-arm offsets and holds the admissible count fixed, which makes every per-domain
    difference exactly constant and is how the zero-variance interval path is reached -- a varying count alone
    perturbs the cell mean in its last bit, which is enough to leave that path. ``jitter`` is a zero-sum
    per-domain shift applied to every arm, so it cancels inside a rung and varies the difference between rungs.
    """

    rows = []
    for k, domain in enumerate(domains):
        for arm in arms:
            tm = base + effects.get(arm, 0.0) + (0.0 if flat else ARM_OFFSETS[arm][k])
            admissible = (ADMISSIBLE if flat
                          else ARM_ADMISSIBLE[arm] + int(round(ARM_OFFSETS[arm][k] * 300)))
            rows += make_cell(rung, domain, arm, tm + (0.0 if jitter is None else jitter[k]), pool=pool,
                              n_admissible=admissible, seed=k)
    return rows


def cell_endpoint(rows, rung, domain, arm):
    """The two endpoints of one cell, recomputed from the rows independently of the script."""

    pool = [r for r in rows if (r["model"], r["domain"], r["arm"]) == (rung, domain, arm)]
    admissible = [r for r in pool if r["identity_to_wild_type"] < e20.ADMISSIBLE_IDENTITY]
    tm = (float(np.mean([r["tm_score"] for r in admissible]))
          if len(admissible) >= e20.MIN_ADMISSIBLE else None)
    return {"tm": tm, "feasibility": len(admissible) / len(pool)}


def expected_contrast(rows, rung, name, metric):
    """The per-domain differences and their mean, recomputed from the rows."""

    left, right = next((a, b) for n, a, b, _ in ana.CONTRASTS if n == name)
    per = {d: cell_endpoint(rows, rung, d, left)[metric] - cell_endpoint(rows, rung, d, right)[metric]
           for d in DOMAINS}
    return per, float(np.mean(list(per.values())))


def strengths():
    return [{"model": rung, "arm": arm,
             "frozen_strength": {"lambda_fields": lf, "lambda_coupling": lc, "pool_size": POOL}}
            for rung, arms in DOSES.items() for arm, (lf, lc) in arms.items()]


def write_protocol(tmp_path, precision, *, declared=None, columns_override=None):
    path = tmp_path / f"protocol_{precision}.json"
    lengths = {**COLUMNS, **(columns_override or {})}
    path.write_text(json.dumps({
        "schema": "coupling_guided_generation_v1", "phase": "robustness",
        "amendment": "progen3-native-contract-amendment-20261015",
        "native_contract": {"dtype": precision, "batch_size": 1, "budget": 1024, "score": "native"},
        "guidance_arms": ["native", "profile", "potts_fields", "coupling", "permuted_coupling"],
        "model_arms": ["progen3-112m"], "permutation_seed": 20261013, "sequences_per_cell": POOL,
        "domains": {d: {"L": lengths[d], "wt_sha256": "0" * 64} for d in DOMAINS},
        "roles": {"development": [f"{d}.pdb" for d in DEVELOPMENT],
                  "evaluation": [f"{d}.pdb" for d in DOMAINS]},
        "strengths": strengths() if declared is None else declared,
    }))
    return path


def write_records(tmp_path, name, rows):
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def write_calibration(tmp_path, checkpoint, precision, *, margins=None, status="solved", refusals=(),
                      dose_override=None):
    """One rung's dose calibration, carrying only the fields the declared-context block reads."""

    arms = {}
    for arm in ("Pr", "M", "C", "Cx"):
        lf, lc = DOSES[f"{checkpoint}-{precision}"][arm]
        component = "fields" if lf else "coupling"
        dose = dose_override[arm] if dose_override and arm in dose_override else (lf or lc)
        per_domain = {d: (margins or {}).get((arm, d), 20.0) for d in DOMAINS}
        arms[arm] = {
            "arm": arm, "component": component, "dose": dose, "noise_floor_nats": 0.036,
            "floor_margin_required": 5.0, "floor_margin_achieved": float(np.median(list(per_domain.values()))),
            "per_domain_floor_margin": per_domain,
            "achieved": {"per_domain": {d: 0.4 for d in DOMAINS}, "median": 0.4},
        }
    path = tmp_path / f"calibration_{checkpoint}_{precision}.json"
    path.write_text(json.dumps({
        "schema": "progen3_guidance_dose_calibration_v1", "status": status, "refusals": list(refusals),
        "basis": "each domain's own wild-type token path", "comparable_statistic": "kl_per_residue",
        "decision_statistic": "median over the evaluation domains", "permutation_seed": 20261013,
        "target": {"model": checkpoint, "dtype": precision}, "solved": arms,
    }))
    return path


def campaign(tmp_path, *, effects=None, float16_effects=None, rung_rows=None, flat=False):
    """Both measure lines written out, with per-rung planted effects."""

    effects = effects or {}
    rows = {rung: records_for(rung, effects, flat=flat) for rung in LINES["bfloat16"]}
    rows["progen3-112m-float16"] = records_for("progen3-112m-float16",
                                               {**effects, **(float16_effects or {})},
                                               jitter=None if flat else PRECISION_JITTER, flat=flat)
    rows.update(rung_rows or {})
    bf = [r for rung in LINES["bfloat16"] for r in rows[rung]]
    return {
        "records": [write_records(tmp_path, "bf16", bf),
                    write_records(tmp_path, "fp16", rows["progen3-112m-float16"])],
        "protocol": [write_protocol(tmp_path, "bfloat16"), write_protocol(tmp_path, "float16")],
        "calibration": [write_calibration(tmp_path, "progen3-3b", "bfloat16"),
                        write_calibration(tmp_path, "progen3-112m", "bfloat16")],
        "rows": rows,
    }


def argv(bundle, out=None, *, dry_run=False):
    args = []
    for path in bundle["records"]:
        args += ["--records", str(path)]
    for path in bundle["protocol"]:
        args += ["--protocol", str(path)]
    for path in bundle["calibration"]:
        args += ["--calibration", str(path)]
    args += ["--references", str(REFERENCES), "--permutation-seed", "20261013"]
    if out is not None:
        args += ["--out", str(out)]
    if dry_run:
        args.append("--dry-run")
    return args


def run(tmp_path, bundle, out_name="robustness.json"):
    out = tmp_path / out_name
    assert ana.main(argv(bundle, out)) == 0
    return json.loads(out.read_text())


# ----------------------------------------------------------------- contrast arithmetic


def test_contrast_arithmetic_recovers_planted_effects_per_rung(tmp_path):
    planted = {"C": 0.03, "Pr": 0.01, "M": 0.005, "Cx": 0.002}
    bundle = campaign(tmp_path, effects=planted)
    result = run(tmp_path, bundle)
    strata = result["bounds"]["primary_tm"]["strata"]
    effects = {"C-Pr": 0.02, "C-M": 0.025, "C-Cx": 0.028, "Pr-N": 0.01, "M-N": 0.005}
    for rung, rows in bundle["rows"].items():
        for name, planted_mean in effects.items():
            record = strata[rung][name]
            assert record["status"] == "estimated" and record["n_domains"] == 8
            per, mean = expected_contrast(rows, rung, name, "tm")
            for domain, diff in per.items():
                assert record["per_domain"][domain]["diff"] == pytest.approx(diff)
            assert record["mean"] == pytest.approx(mean) == pytest.approx(planted_mean, abs=1e-12)
            low, high = record["interval95_bca"]
            assert record["interval_zero_variance"] is False and low < mean < high
            assert result["achieved_upper_bounds"]["primary_tm"][name][rung] == pytest.approx(high)
            assert record["favouring"]["positive"] + record["favouring"]["negative"] == 8


def test_pooled_stratum_is_the_mean_over_its_own_measure_lines_rungs(tmp_path):
    bundle = campaign(tmp_path, effects={"C": 0.03})
    # the 3B rung alone is shifted, so pooled must land halfway between the two bfloat16 rungs
    bundle["rows"]["progen3-3b-bfloat16"] = records_for("progen3-3b-bfloat16", {"C": 0.05})
    bundle["records"][0] = write_records(
        tmp_path, "bf16", [r for rung in LINES["bfloat16"] for r in bundle["rows"][rung]])
    strata = run(tmp_path, bundle)["bounds"]["primary_tm"]["strata"]
    per_rung = [strata[rung]["C-Pr"]["mean"] for rung in LINES["bfloat16"]]
    assert strata["pooled_bfloat16"]["C-Pr"]["mean"] == pytest.approx(float(np.mean(per_rung)))
    for domain in DOMAINS:
        parts = [strata[rung]["C-Pr"]["per_domain"][domain]["diff"] for rung in LINES["bfloat16"]]
        assert strata["pooled_bfloat16"]["C-Pr"]["per_domain"][domain]["diff"] == pytest.approx(np.mean(parts))


def test_measure_lines_never_pool_together(tmp_path):
    result = run(tmp_path, campaign(tmp_path))
    design = result["design"]
    assert design["strata_by_measure_line"] == {
        "bfloat16": [*LINES["bfloat16"], "pooled_bfloat16"], "float16": [*LINES["float16"]]}
    assert design["measure_lines"]["float16"]["pooled_stratum"] is None
    assert "one rung" in design["measure_lines"]["float16"]["pooled_withheld_reason"]
    assert set(result["bounds"]["primary_tm"]["strata"]) == {
        *LINES["bfloat16"], "pooled_bfloat16", *LINES["float16"]}


def test_both_endpoints_are_reported_and_the_second_is_named_for_what_it_measures(tmp_path):
    bundle = campaign(tmp_path)
    result = run(tmp_path, bundle)
    assert set(result["bounds"]) == {"primary_tm", "novelty_admissibility_rate"}
    block = result["bounds"]["novelty_admissibility_rate"]
    assert "no predicted confidence" in block["definition"]
    assert "novelty-admissibility rate" in block["note"]
    for name, *_rest in ana.CONTRASTS:
        record = block["strata"]["progen3-3b-bfloat16"][name]
        per, mean = expected_contrast(bundle["rows"]["progen3-3b-bfloat16"], "progen3-3b-bfloat16",
                                      name, "feasibility")
        assert record["mean"] == pytest.approx(mean)
        for domain, diff in per.items():
            assert record["per_domain"][domain]["diff"] == pytest.approx(diff)


# ----------------------------------------------------------------- the ruling: no F arm, no verdict


def test_c_minus_m_is_its_own_contrast_and_f_minus_m_is_absent(tmp_path):
    result = run(tmp_path, campaign(tmp_path))
    definitions = result["design"]["contrast_definitions"]
    assert set(definitions) == {"C-Pr", "C-M", "C-Cx", "Pr-N", "M-N"}
    assert "F-M" not in definitions and "F" not in result["design"]["arms"]
    assert "not a substitute for it" in definitions["C-M"]["reads"]
    assert definitions["C-M"]["left"] == "C" and definitions["C-M"]["right"] == "M"
    assert "F-M contrast cannot be formed" in result["design"]["arms_absent"]["F"]


def keys_of(value, found=None):
    found = set() if found is None else found
    if isinstance(value, dict):
        found |= set(value)
        for item in value.values():
            keys_of(item, found)
    elif isinstance(value, list):
        for item in value:
            keys_of(item, found)
    return found


def test_no_verdict_no_flag_and_no_p_value_anywhere(tmp_path):
    result = run(tmp_path, campaign(tmp_path))
    assert result["issues_a_verdict"] is False and result["issues_a_pass_fail_flag"] is False
    assert "NO VERDICT" in result["statement"] and "NO PASS/FAIL FLAG" in result["statement"]
    assert result["reference_scale"]["margin_tm"] == e20.MARGIN
    assert "no bound is tested against it" in result["reference_scale"]["role"]
    # no verdict field, no margin flag field, and none of E20's verdict values anywhere in the artifact
    assert {"verdict", "verdict_reason", "below_margin", "verdict_is_secondary_flag", "equivalence"} \
        & keys_of(result) == set()
    text = json.dumps(result)
    for forbidden in (e20.VERDICT_CONCLUDE, e20.VERDICT_INCONCLUSIVE):
        assert forbidden not in text
    # "unestimable" survives only as the status the frozen estimate gives a contrast below the unit floor
    assert e20.VERDICT_UNESTIMABLE == "unestimable"
    assert "p_holm" in text  # the frozen record shape is kept, with the fields null
    for stratum in result["bounds"]["primary_tm"]["strata"].values():
        for record in stratum.values():
            assert record["p_holm_ttest"] is None and record["p_holm_signflip"] is None
            assert "_p" not in record and "_ps" not in record


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # a constant sample makes the frozen t-test degenerate
def test_a_constant_per_domain_difference_is_flagged_rather_than_given_a_fake_interval(tmp_path):
    strata = run(tmp_path, campaign(tmp_path, effects={"C": 0.03}, flat=True))["bounds"]["primary_tm"]["strata"]
    record = strata["progen3-3b-bfloat16"]["C-Pr"]
    assert record["status"] == "estimated" and record["interval_zero_variance"] is True
    assert record["interval95_bca"] == [pytest.approx(0.03), pytest.approx(0.03)]
    assert ana.upper_bound(record) == pytest.approx(0.03)


def test_bca_coverage_note_is_carried_forward_unchanged(tmp_path):
    result = run(tmp_path, campaign(tmp_path))
    assert result["discipline"]["interval_coverage_note"] == e20.BCA_NOTE
    assert "0.89" in e20.BCA_NOTE
    assert result["discipline"]["minimum_bootstrap_units"] == 8


# ----------------------------------------------------------------- the eight-domain floor


def test_a_contrast_on_seven_domains_carries_no_interval(tmp_path):
    bundle = campaign(tmp_path)
    rows = [r for r in bundle["rows"]["progen3-112m-float16"]
            if not (r["domain"] == "1IFY" and r["arm"] == "C")]
    rows += make_cell("progen3-112m-float16", "1IFY", "C", 0.4, n_admissible=e20.MIN_ADMISSIBLE - 1)
    bundle["records"][1] = write_records(tmp_path, "fp16", rows)
    strata = run(tmp_path, bundle)["bounds"]["primary_tm"]["strata"]
    for name in ("C-Pr", "C-M", "C-Cx"):
        record = strata["progen3-112m-float16"][name]
        assert record["status"] == "unestimable" and record["n_domains"] == 7
        assert record["interval95_bca"] is None
        assert "below the 8-domain floor" in record["reason"] and "1IFY" in record["reason"]
        assert record["mean"] is not None  # the mean is still shown; only the interval is withheld
        assert run_upper(strata, "progen3-112m-float16", name) is None
    # a contrast that does not touch C keeps all eight domains
    assert strata["progen3-112m-float16"]["Pr-N"]["n_domains"] == 8


def run_upper(strata, stratum, name):
    return ana.upper_bound(strata[stratum][name])


# ----------------------------------------------------------------- precision control


def test_precision_control_differences_the_same_arm_across_precisions(tmp_path):
    shift = {"C": 0.004, "N": 0.001}  # float16 minus bfloat16, planted per arm
    result = run(tmp_path, campaign(tmp_path, effects={"C": 0.03}, float16_effects={"C": 0.034, "N": 0.001}))
    control = result["precision_control"]["checkpoints"]["progen3-112m"]
    assert control["float16_rung"] == "progen3-112m-float16"
    assert control["bfloat16_rung"] == "progen3-112m-bfloat16"
    by_arm = control["endpoints"]["primary_tm"]["by_arm"]
    for arm in ARMS:
        assert by_arm[arm]["n_domains"] == 8
        assert by_arm[arm]["mean"] == pytest.approx(shift.get(arm, 0.0))
    # the 3B rung has no float16 twin, so it is not a precision pair
    assert set(result["precision_control"]["checkpoints"]) == {"progen3-112m"}
    assert "only thing that moves" in result["precision_control"]["statement"]


def test_precision_control_is_void_when_the_two_precisions_sit_at_different_doses(tmp_path):
    """Moving the float16 dose coherently -- protocol, records and its own calibration -- must void the control."""

    moved = {rung: dict(arms) for rung, arms in DOSES.items()}
    moved["progen3-112m-float16"]["C"] = (0.0, 9.9)
    bundle = campaign(tmp_path)
    declared = [{"model": rung, "arm": arm,
                 "frozen_strength": {"lambda_fields": lf, "lambda_coupling": lc, "pool_size": POOL}}
                for rung, arms in moved.items() for arm, (lf, lc) in arms.items()]
    rows = [r if not (r["model"] == "progen3-112m-float16" and r["arm"] == "C")
            else {**r, "lambda_coupling": 9.9} for r in bundle["rows"]["progen3-112m-float16"]]
    bundle["records"][1] = write_records(tmp_path, "fp16", rows)
    bundle["protocol"] = [write_protocol(tmp_path, "bfloat16", declared=declared),
                          write_protocol(tmp_path, "float16", declared=declared)]
    bundle["calibration"] = [write_calibration(tmp_path, "progen3-3b", "bfloat16"),
                             write_calibration(tmp_path, "progen3-112m", "bfloat16"),
                             write_calibration(tmp_path, "progen3-112m", "float16",
                                               dose_override={"C": 9.9})]
    with pytest.raises(ValueError, match="not at the same doses"):
        ana.main(argv(bundle, tmp_path / "out.json"))


# ----------------------------------------------------------------- declared context


def test_declared_context_lists_the_cells_below_the_required_floor_margin(tmp_path):
    margins = {("M", "1V1C"): 0.6, ("M", "2LGW"): 0.5, ("M", "2LVN"): 4.6, ("C", "1IFY"): 3.4,
               ("Cx", "1IFY"): 4.6}
    bundle = campaign(tmp_path)
    bundle["calibration"] = [write_calibration(tmp_path, "progen3-3b", "bfloat16", margins=margins),
                             write_calibration(tmp_path, "progen3-112m", "bfloat16", margins=margins)]
    floor = run(tmp_path, bundle)["declared_context_dose_floor"]
    assert "not a finding" in floor["statement"] and "ProGen2-medium showed the same weakness" in floor["statement"]
    flagged = {(f["rung"], f["arm"], f["domain"]) for f in floor["below_declared_floor_margin"]}
    for rung in DOSES:
        for arm, domain in margins:
            assert (rung, arm, domain) in flagged
    assert all(f["floor_margin"] < f["floor_margin_required"] for f in floor["below_declared_floor_margin"])
    assert floor["rungs"]["progen3-3b-bfloat16"]["M"]["domains_below_required_margin"] == ["1V1C", "2LGW", "2LVN"]
    assert "N" not in floor["rungs"]["progen3-3b-bfloat16"]  # the native arm carries no dose
    assert "guidance_strength" in floor["intervention_basis"]


def test_the_float16_rung_carries_its_declared_context_over_and_says_why(tmp_path):
    floor = run(tmp_path, campaign(tmp_path))["declared_context_dose_floor"]
    carried = floor["rungs"]["progen3-112m-float16"]["C"]["carried_over"]
    assert carried["from"] == "progen3-112m-bfloat16"
    assert "no calibration of its own" in carried["reason"] and "dose-for-dose" in carried["reason"]
    assert floor["rungs"]["progen3-112m-bfloat16"]["C"]["carried_over"] is None


def test_a_calibration_describing_another_dose_is_refused(tmp_path):
    bundle = campaign(tmp_path)
    bundle["calibration"] = [write_calibration(tmp_path, "progen3-3b", "bfloat16"),
                             write_calibration(tmp_path, "progen3-112m", "bfloat16",
                                               dose_override={"C": 1.999})]
    with pytest.raises(ValueError, match="does not match the calibrated"):
        ana.main(argv(bundle, tmp_path / "out.json"))


@pytest.mark.parametrize("kwargs", [{"status": "unsolved"}, {"refusals": ["a domain did not bracket"]}])
def test_a_calibration_that_was_not_solved_is_refused(tmp_path, kwargs):
    bundle = campaign(tmp_path)
    bundle["calibration"] = [write_calibration(tmp_path, "progen3-3b", "bfloat16"),
                             write_calibration(tmp_path, "progen3-112m", "bfloat16", **kwargs)]
    with pytest.raises(ValueError, match="has no declared context"):
        ana.main(argv(bundle, tmp_path / "out.json"))


# ----------------------------------------------------------------- refusals on the records and the protocol


def test_an_unparsable_model_label_is_refused(tmp_path):
    bundle = campaign(tmp_path)
    rows = [{**r, "model": "progen3-112m"} for r in bundle["rows"]["progen3-112m-float16"]]
    bundle["records"][1] = write_records(tmp_path, "fp16", rows)
    with pytest.raises(ValueError, match="does not parse as"):
        ana.main(argv(bundle, tmp_path / "out.json"))


def test_two_precisions_in_one_measure_line_are_refused(tmp_path):
    """A float16 cell written into the bfloat16 line must be refused, not merged: a line is a precision."""

    bundle = campaign(tmp_path)
    strays = [r for r in bundle["rows"]["progen3-112m-float16"] if r["arm"] == "N"]
    mixed = [r for rung in LINES["bfloat16"] for r in bundle["rows"][rung]] + strays
    bundle["records"][0] = write_records(tmp_path, "bf16", mixed)
    with pytest.raises(ValueError, match="must be measured on their own line"):
        ana.main(argv(bundle, tmp_path / "out.json"))


def test_a_missing_rung_is_refused(tmp_path):
    bundle = campaign(tmp_path)
    bundle["records"][0] = write_records(tmp_path, "bf16", bundle["rows"]["progen3-112m-bfloat16"])
    with pytest.raises(ValueError, match="the protocol declares"):
        ana.main(argv(bundle, tmp_path / "out.json"))


def test_a_missing_arm_is_refused(tmp_path):
    bundle = campaign(tmp_path)
    rows = [r for r in bundle["rows"]["progen3-112m-float16"] if r["arm"] != "Cx"]
    bundle["records"][1] = write_records(tmp_path, "fp16", rows)
    with pytest.raises(ValueError, match="no Cx cell for progen3-112m-float16"):
        ana.main(argv(bundle, tmp_path / "out.json"))


def test_a_domain_missing_from_one_rung_is_refused_cell_by_cell(tmp_path):
    bundle = campaign(tmp_path)
    rows = [r for r in bundle["rows"]["progen3-112m-float16"] if r["domain"] != "2K1B"]
    bundle["records"][1] = write_records(tmp_path, "fp16", rows)
    with pytest.raises(ValueError, match=r"progen3-112m-float16/2K1B/\w+: no \w+ cell"):
        ana.main(argv(bundle, tmp_path / "out.json"))


def test_a_domain_missing_from_the_whole_campaign_is_refused(tmp_path):
    bundle = campaign(tmp_path)
    for index, rungs in ((0, LINES["bfloat16"]), (1, LINES["float16"])):
        rows = [r for rung in rungs for r in bundle["rows"][rung] if r["domain"] != "2LVN"]
        bundle["records"][index] = write_records(tmp_path, f"line{index}", rows)
    with pytest.raises(ValueError, match="not the sealed eight"):
        ana.main(argv(bundle, tmp_path / "out.json"))


def test_a_short_cell_is_refused_as_an_incomplete_campaign(tmp_path):
    bundle = campaign(tmp_path)
    rows = [r for r in bundle["rows"]["progen3-112m-float16"]
            if not (r["domain"] == "3DKM" and r["arm"] == "M")]
    rows += make_cell("progen3-112m-float16", "3DKM", "M", 0.4, pool=POOL - 1)
    bundle["records"][1] = write_records(tmp_path, "fp16", rows)
    with pytest.raises(ValueError, match="the campaign is incomplete"):
        ana.main(argv(bundle, tmp_path / "out.json"))


def test_a_dose_the_protocol_did_not_declare_is_refused(tmp_path):
    bundle = campaign(tmp_path)
    rows = [r if r["arm"] != "C" else {**r, "lambda_coupling": 1.7}
            for r in bundle["rows"]["progen3-112m-float16"]]
    bundle["records"][1] = write_records(tmp_path, "fp16", rows)
    with pytest.raises(Exception, match="was not frozen"):
        ana.main(argv(bundle, tmp_path / "out.json"))


def test_a_sequence_of_the_wrong_length_is_refused(tmp_path):
    bundle = campaign(tmp_path)
    rows = [{**r, "sequence": r["sequence"][:-1]} if r["domain"] == "1E0L" and r["arm"] == "N" else r
            for r in bundle["rows"]["progen3-112m-float16"]]
    bundle["records"][1] = write_records(tmp_path, "fp16", rows)
    with pytest.raises(ValueError, match="columns the protocol declares"):
        ana.main(argv(bundle, tmp_path / "out.json"))


def test_a_candidate_shared_between_measure_lines_is_refused(tmp_path):
    bundle = campaign(tmp_path)
    overlap = bundle["rows"]["progen3-112m-float16"] + bundle["rows"]["progen3-112m-float16"][:1]
    bundle["records"][1] = write_records(tmp_path, "fp16", overlap)
    with pytest.raises(ValueError, match="more than one measure line"):
        ana.main(argv(bundle, tmp_path / "out.json"))


def test_protocols_that_disagree_on_the_doses_are_refused(tmp_path):
    """The float16 protocol moves a rung it holds no records for, so only the cross-protocol check can catch it."""

    bundle = campaign(tmp_path)
    moved = [{**entry, "frozen_strength": {**entry["frozen_strength"], "lambda_coupling": 9.0}}
             if (entry["model"], entry["arm"]) == ("progen3-3b-bfloat16", "Cx") else entry
             for entry in strengths()]
    bundle["protocol"][1] = write_protocol(tmp_path, "float16", declared=moved)
    with pytest.raises(ValueError, match="no longer matched"):
        ana.main(argv(bundle, tmp_path / "out.json"))


def test_unpaired_records_and_protocols_are_refused(tmp_path):
    bundle = campaign(tmp_path)
    bundle["protocol"] = bundle["protocol"][:1]
    with pytest.raises(ValueError, match="one protocol per records file"):
        ana.main(argv(bundle, tmp_path / "out.json"))


def test_a_missing_asset_names_the_option(tmp_path):
    bundle = campaign(tmp_path)
    bundle["records"][1] = tmp_path / "absent.jsonl"
    with pytest.raises(FileNotFoundError, match="--records"):
        ana.main(argv(bundle, tmp_path / "out.json"))
    bundle = campaign(tmp_path)
    args = argv(bundle, tmp_path / "out.json")
    args[args.index("--references") + 1] = str(tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="--references"):
        ana.main(args)


# ----------------------------------------------------------------- the selector blind spot


@pytest.mark.parametrize("name", sorted(e20.FORBIDDEN_SELECTOR_FIELDS) + ["tm_oracle", "not_a_selector"])
def test_structural_fields_cannot_be_selectors(name):
    view = e20.selector_view(make_cell("progen3-3b-bfloat16", "1E0L", "N", 0.4))
    with pytest.raises(ValueError):
        e20.selector_scores([name], view, None, 1)


def test_selector_view_carries_no_structural_field():
    view = e20.selector_view(make_cell("progen3-3b-bfloat16", "1E0L", "N", 0.4))
    assert set(vars(view)) == {"domain", "sequences", "native_loglik"}


def test_selection_is_blind_to_structure_derived_values(tmp_path):
    """Rewriting tm_score, plddt_mean and ptm must not move which candidates a selector picks."""

    bundle = campaign(tmp_path)
    base = run(tmp_path, bundle, "base.json")
    rows = {rung: [{**r, "plddt_mean": 99.0 - i % 5, "ptm": 0.1 * (i % 9)}
                   for i, r in enumerate(cell)] for rung, cell in bundle["rows"].items()}
    bundle["records"] = [write_records(tmp_path, "bf16b", [r for rung in LINES["bfloat16"] for r in rows[rung]]),
                         write_records(tmp_path, "fp16b", rows["progen3-112m-float16"])]
    other = run(tmp_path, bundle, "other.json")
    label = e20.CAPTURE_LABEL
    for stratum, block in base[label]["by_stratum"].items():
        for arm, arm_block in block["arms"].items():
            for domain, per in arm_block["per_domain"].items():
                moved = other[label]["by_stratum"][stratum]["arms"][arm]["per_domain"][domain]
                for selector, entry in per["selectors"].items():
                    assert moved["selectors"][selector]["gain"] == pytest.approx(entry["gain"])


# ----------------------------------------------------------------- captured fraction and diversity


def test_captured_fraction_reuses_the_frozen_definitions_and_no_selector_beats_the_ceiling(tmp_path):
    bundle = campaign(tmp_path)
    rows = {}
    for rung, cell in bundle["rows"].items():
        spread = []
        for i, r in enumerate(cell):
            spread.append({**r, "tm_score": 0.2 + 0.005 * (i % ADMISSIBLE)
                           if r["identity_to_wild_type"] < 20 else 0.9})
        rows[rung] = spread
    bundle["records"] = [write_records(tmp_path, "bf16", [r for rung in LINES["bfloat16"] for r in rows[rung]]),
                         write_records(tmp_path, "fp16", rows["progen3-112m-float16"])]
    block = run(tmp_path, bundle)[e20.CAPTURE_LABEL]["by_stratum"]
    assert set(block) == {*LINES["bfloat16"], "pooled_bfloat16", *LINES["float16"]}
    for stratum, strat_block in block.items():
        assert strat_block["k"] == e20.SELECTION_K == 10
        assert strat_block["expectation"] == e20.CAPTURE_EXPECTATION
        assert strat_block["oracle_statement"] == e20.CAPTURE_ORACLE_STATEMENT
        assert "200 candidates" in strat_block["depth_note"]
        assert set(strat_block["arms"]) == set(e20.CAPTURE_ARMS)
        for arm_block in strat_block["arms"].values():
            assert arm_block["n_domains"] == 8
            for per in arm_block["per_domain"].values():
                assert all(s["gain"] <= per["oracle_gain"] + 1e-12 for s in per["selectors"].values())
            for entry in arm_block["selectors"].values():
                assert entry["status"] == "estimated" and entry["interval95_bca"] is not None
    # the oracle is the top-k realised TM minus the pool mean of the same admissible pool
    pool = [r for r in rows["progen3-112m-float16"]
            if (r["domain"], r["arm"]) == ("1E0L", "N") and r["identity_to_wild_type"] < 20]
    tm = sorted((r["tm_score"] for r in pool), reverse=True)
    expected = float(np.mean(tm[:10]) - np.mean(tm))
    got = block["progen3-112m-float16"]["arms"]["N"]["per_domain"]["1E0L"]["oracle_gain"]
    assert got == pytest.approx(expected)


def test_captured_fraction_is_unestimable_when_a_domain_has_no_usable_pool(tmp_path):
    bundle = campaign(tmp_path)
    rows = [r for r in bundle["rows"]["progen3-112m-float16"]
            if not (r["domain"] == "1F0M" and r["arm"] == "N")]
    rows += make_cell("progen3-112m-float16", "1F0M", "N", 0.4, n_admissible=e20.MIN_ADMISSIBLE - 1)
    bundle["records"][1] = write_records(tmp_path, "fp16", rows)
    arm_block = run(tmp_path, bundle)[e20.CAPTURE_LABEL]["by_stratum"]["progen3-112m-float16"]["arms"]["N"]
    assert arm_block["n_domains"] == 7 and "1F0M" in arm_block["missing_domains"]
    assert all(entry["status"] == "unestimable" and entry["interval95_bca"] is None
               for entry in arm_block["selectors"].values())


def test_pool_diversity_is_reported_per_rung_and_arm(tmp_path):
    result = run(tmp_path, campaign(tmp_path))
    diversity = result["pool_diversity"]
    assert "mean pairwise identity" in diversity["definition"] and "90%-or-closer" in diversity["definition"]
    assert set(diversity["rungs"]) == set(DOSES)  # the artifact is written with sorted keys
    for arms in diversity["rungs"].values():
        assert set(arms) == set(ARMS)
        for block in arms.values():
            assert block["n_cells"] == 8 and len(block["per_domain"]) == 8
            assert 0.0 <= block["mean_pairwise_identity"] <= 100.0
            assert 0.0 <= block["near_duplicate_rate"] <= 1.0
    # the same numbers reach the per-cell table the frozen summary produced them in
    cell = next(c for c in result["cells"]
                if (c["model"], c["domain"], c["arm"]) == ("progen3-3b-bfloat16", "1E0L", "N"))
    assert cell["mean_pairwise_identity"] == pytest.approx(
        diversity["rungs"]["progen3-3b-bfloat16"]["N"]["per_domain"]["1E0L"]["mean_pairwise_identity"])
    assert cell["kl_per_residue"] is None  # these records carry no per-candidate guidance_strength


# ----------------------------------------------------------------- end to end


def test_dry_run_reports_shapes_and_counts_and_computes_no_contrast(tmp_path, capsys):
    bundle = campaign(tmp_path)
    out = tmp_path / "inputs.json"
    assert ana.main(argv(bundle, out, dry_run=True)) == 0
    report = json.loads(out.read_text())
    assert report["schema"] == ana.INPUTS_SCHEMA
    assert "no contrast, bound, interval or endpoint value was computed" in report["statement"]
    for key in ("bounds", "achieved_upper_bounds", "precision_control", "pool_diversity", e20.CAPTURE_LABEL):
        assert key not in report
    assert report["n_records_total"] == len(DOSES) * len(ARMS) * len(DOMAINS) * POOL
    assert set(report["measure_lines"]) == {"bfloat16", "float16"}
    assert report["measure_lines"]["bfloat16"]["rungs"] == list(LINES["bfloat16"])
    assert report["measure_lines"]["float16"]["n_records"] == len(ARMS) * len(DOMAINS) * POOL
    for rung, shape in report["rungs"].items():
        assert shape["arms"] == sorted(ARMS) and shape["n_domains"] == 8 and shape["n_cells"] == 40
        assert shape["candidates_per_cell"] == [POOL]
        assert shape["doses"] == {arm: list(DOSES[rung][arm]) for arm in sorted(ARMS)}
    for domain, block in report["columns_by_domain"].items():
        assert block["declared_by_protocol"] == block["family_reference"] == COLUMNS[domain]
        assert block["sequence_lengths_in_records"] == [COLUMNS[domain]]
    # the calibration is validated too, and only counts are reported
    calibration = report["calibration"]
    assert calibration["files"] == ["progen3-112m-bfloat16", "progen3-3b-bfloat16"]
    assert calibration["carried_over_rungs"] == ["progen3-112m-float16"]
    assert calibration["n_dose_cells_below_required_floor_margin"] == 0
    assert calibration["source_by_rung"]["progen3-112m-float16"] == ["progen3-112m-bfloat16"]
    assert set(calibration) == {"files", "source_by_rung", "carried_over_rungs",
                                "n_dose_cells_below_required_floor_margin"}
    assert "per_domain_floor_margin" not in json.dumps(calibration)
    assert "Dry run" in capsys.readouterr().out


def test_a_dry_run_refuses_a_calibration_the_full_run_would_refuse(tmp_path):
    """A clean dry run must mean the full run will not refuse on its inputs."""

    bundle = campaign(tmp_path)
    bundle["calibration"] = [write_calibration(tmp_path, "progen3-3b", "bfloat16"),
                             write_calibration(tmp_path, "progen3-112m", "bfloat16",
                                               dose_override={"M": 0.9})]
    with pytest.raises(ValueError, match="does not match the calibrated"):
        ana.main(argv(bundle, dry_run=True))


def test_dry_run_without_an_out_path_still_validates(tmp_path):
    assert ana.main(argv(campaign(tmp_path), dry_run=True)) == 0


def test_a_full_run_needs_an_out_path(tmp_path):
    with pytest.raises(SystemExit):
        ana.main(argv(campaign(tmp_path)))


def test_artifact_records_its_provenance_and_renders(tmp_path, capsys):
    result = run(tmp_path, campaign(tmp_path))
    assert result["schema"] == ana.SCHEMA
    provenance = result["provenance"]
    assert provenance["frozen_module"] == "scripts/capability/generation/analyse_e20_confirmatory.py"
    assert len(provenance["frozen_module_sha256"]) == 64
    assert provenance["inputs"]["permutation_seed"] == 20261013
    assert len(provenance["inputs"]["records"]) == 2 and len(provenance["inputs"]["protocol"]) == 2
    printed = capsys.readouterr().out
    assert "ACHIEVED UPPER 95% BCa BOUND" in printed and "PRECISION CONTROL" in printed
    assert "DECLARED CONTEXT" in printed


def test_the_frozen_module_is_not_modified_by_running_this_one(tmp_path):
    """The frozen tables must read exactly as E20 left them after this script has run."""

    before = (e20.MODELS, e20.STRATA, {a: dict(v) for a, v in e20.FROZEN_DOSES.items()}, e20.CELL_CONTRASTS)
    run(tmp_path, campaign(tmp_path))
    assert e20.MODELS == before[0] == ("progen2-medium", "protgpt2")
    assert e20.STRATA == before[1]
    assert {a: dict(v) for a, v in e20.FROZEN_DOSES.items()} == before[2]
    assert e20.CELL_CONTRASTS == before[3]
