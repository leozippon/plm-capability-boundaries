"""Selection-factorial analysis: invariances, component identity and loud failures."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/capability/generation/analyse_selection_factorial.py"
spec = importlib.util.spec_from_file_location("analyse_selection_factorial", SCRIPT)
sel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sel)


def make_model(domain: str, length: int, seed: int):
    rng = np.random.default_rng(seed)
    J = rng.normal(size=(length, length, 20, 20))
    J = J + J.transpose(1, 0, 3, 2)
    for i in range(length):
        J[i, i] = 0.0
    return SimpleNamespace(domain=domain, length=length, h=rng.normal(size=(length, 20)), J=J,
                           log_profile=np.log(rng.dirichlet(np.ones(20), size=length)))


def make_codes(length: int, n: int = 40):
    return np.random.default_rng(7).integers(0, 20, size=(n, length))


def shifted(model, seed=3):
    """Add a constant to every column of h and log_profile and to every pair block of J."""
    rng = np.random.default_rng(seed)
    L = model.length
    c = rng.normal(size=(L, L))
    c = c + c.T
    np.fill_diagonal(c, 0.0)
    return SimpleNamespace(domain=model.domain, length=L, h=model.h + rng.normal(size=(L, 1)),
                           J=model.J + c[:, :, None, None],
                           log_profile=model.log_profile + rng.normal(size=(L, 1)))


def test_every_score_shifts_by_a_constant_under_constant_additions():
    model, donor = make_model("T", 12, 1), make_model("D", 15, 2)
    codes = make_codes(12)
    base = sel.selector_scores(model, donor, 5, codes)
    moved = sel.selector_scores(shifted(model), shifted(donor, 4), 5, codes)
    assert set(base) == set(moved)
    for name in base:
        assert np.ptp(moved[name] - base[name]) < 1e-9, name


def test_full_is_fields_plus_coupling():
    scores = sel.selector_scores(make_model("T", 12, 1), None, 5, make_codes(12))
    assert np.allclose(scores["potts_full"], scores["potts_fields"] + scores["potts_coupling"])
    assert "coupling_cross_domain" not in scores  # no donor: absent, never silently substituted


def test_identity_permutation_is_refused():
    with pytest.raises(ValueError, match="identity"):
        sel.selector_scores(make_model("T", 1, 1), None, 5, make_codes(1))


def test_non_standard_residue_fails():
    with pytest.raises(ValueError, match="canonical"):
        sel.encode_pool(["ACDX"], 4, "T")
    with pytest.raises(ValueError, match="length"):
        sel.encode_pool(["ACD"], 4, "T")


def test_missing_asset_option_exits_naming_it(tmp_path):
    run = subprocess.run([sys.executable, str(SCRIPT), "--records", str(tmp_path / "r.jsonl"),
                          "--out", str(tmp_path / "o.json"), "--permutation-seed", "1"],
                         capture_output=True, text=True)
    assert run.returncode == 2 and "--references" in run.stderr


def pool_records(n_admissible: int, n: int = 60, length: int = 10, arm: str = "N"):
    rng = np.random.default_rng(11)
    ident = np.where(np.arange(n) < n_admissible, rng.uniform(2, 19, n), rng.uniform(25, 90, n))
    return [{"model": "m", "domain": "T", "arm": arm, "lambda_fields": 0.0, "lambda_coupling": 0.0,
             "sequence": "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=length)),
             "native_loglik": float(rng.normal()), "tm_score": float(0.2 + 0.005 * ident[i]),
             "identity_to_wild_type": float(ident[i]), "max_msa_identity": 15.0} for i in range(n)]


def one_cell(records):
    model = make_model("T", 10, 1)
    model.qualified = True
    cells, _ = sel.analyse(records, {"T": model}, 5)
    assert len(cells) == 1
    return cells[0]


def test_identity_oracle_picks_the_most_native_candidates_and_is_labelled_an_oracle():
    records = pool_records(40)
    cell = one_cell(records)
    top5 = sorted(r["identity_to_wild_type"] for r in records)[-5:]
    assert cell["selectors"]["identity_oracle"]["5"]["identity_to_wild_type"] == pytest.approx(np.mean(top5))
    assert "identity_oracle" in sel.ORACLES and "not a usable or proposed selector" in sel.ORACLE_STATEMENT


def test_restricted_stratum_selects_within_admissible_candidates_only():
    records = pool_records(40)
    cell = one_cell(records)
    admissible = [r for r in records if r["identity_to_wild_type"] < 20]
    assert cell["n_admissible"] == 40
    got = cell["restricted"]["selectors"]["identity_oracle"]["5"]["tm_score"]
    assert got == pytest.approx(np.mean(sorted(r["tm_score"] for r in admissible)[-5:]))
    assert cell["restricted"]["pool_mean"]["tm_score"] == pytest.approx(np.mean([r["tm_score"] for r in admissible]))
    assert cell["selectors"]["identity_oracle"]["5"]["tm_score"] > got  # the unrestricted numbers are kept beside


def test_cell_below_the_floor_has_no_restricted_value():
    assert one_cell(pool_records(24))["restricted"] is None
    assert one_cell(pool_records(25))["restricted"] is not None


def test_shrinkage_is_one_minus_the_gain_ratio():
    def block(gain):
        return {"k": {"5": {"selectors": {"s": {"n_domains": 4, "mean_gain": {"gain_tm": gain}}}}}}
    out = sel.shrinkage(block(0.10), block(0.05), block(0.02))["5"]["s"]
    assert out["shrinkage_matched"] == pytest.approx(0.6) and out["shrinkage_vs_all_cells"] == pytest.approx(0.8)
    assert sel.shrinkage(block(0.0), block(0.0), block(0.02))["5"]["s"]["shrinkage_matched"] is None  # no base, no ratio


def test_shrinkage_is_flagged_unreliable_where_the_base_gain_is_small():
    def block(gain):
        return {"k": {"5": {"selectors": {"s": {"n_domains": 4, "mean_gain": {"gain_tm": gain}}}}}}
    out = sel.shrinkage(block(0.005), block(0.05), block(0.02))["5"]["s"]
    assert out["shrinkage_vs_all_cells_unreliable"] is True and out["shrinkage_matched_unreliable"] is False
    assert "unreliable" in sel.STRATA_STATEMENT and "0.01" in sel.STRATA_STATEMENT
