"""Selection-budget analysis: nested depths, the TM ceiling, the determinism bound and the labels."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/capability/generation/analyse_selection_budget.py"
spec = importlib.util.spec_from_file_location("analyse_selection_budget", SCRIPT)
bud = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bud
spec.loader.exec_module(bud)


def make_model(domain, length, seed):
    rng = np.random.default_rng(seed)
    J = rng.normal(size=(length, length, 20, 20))
    J = J + J.transpose(1, 0, 3, 2)
    for i in range(length):
        J[i, i] = 0.0
    return SimpleNamespace(domain=domain, length=length, h=rng.normal(size=(length, 20)), J=J, qualified=True,
                           log_profile=np.log(rng.dirichlet(np.ones(20), size=length)))


def records(n, start_seed, tag, length=10, domain="T"):
    rng = np.random.default_rng(start_seed)
    out = []
    for i in range(n):
        admissible = i % 2 == 0
        out.append({"id": f"{tag}{i}", "seed": start_seed + i, "model": "m", "domain": domain, "arm": "N",
                    "lambda_fields": 0.0, "lambda_coupling": 0.0,
                    "sequence": "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=length)),
                    "native_loglik": float(rng.normal()), "tm_score": float(rng.uniform(0.1, 0.6)),
                    "identity_to_wild_type": float(rng.uniform(2, 19) if admissible else rng.uniform(30, 80)),
                    "max_msa_identity": 15.0})
    return out


def test_oracle_is_the_ceiling_depths_nest_and_labels_are_present():
    dev, pool = records(100, 1000, "d"), records(400, 5000, "p")
    result = bud.analyse(dev, pool, {"T": make_model("T", 10, 1)}, 5)
    cell = result["cells"][0]
    assert cell["n_candidates"] == 500 and set(cell["depths"]) == {"100", "200", "300", "400", "500"}
    assert [cell["depths"][d]["n_pool"] for d in ("100", "500")] == [100, 500]
    for entry in cell["depths"].values():
        r = entry["restricted"]
        assert r is not None and entry["n_admissible"] >= 25
        for k in ("1", "10", "25"):
            og = r["k"][k]["tm_oracle"]["gain"]
            assert og >= 0 and all(s["gain"] <= og + 1e-12 for s in r["k"][k]["selectors"].values())
    # hand check of the oracle at depth 200: the 200 lowest-seed candidates, admissible only
    rows = sorted(dev + pool, key=lambda r: r["seed"])[:200]
    tm = sorted((r["tm_score"] for r in rows if r["identity_to_wild_type"] < 20), reverse=True)
    assert cell["depths"]["200"]["restricted"]["k"]["10"]["tm_oracle"]["gain"] == pytest.approx(np.mean(tm[:10]) - np.mean(tm))
    assert cell["depths"]["200"]["restricted"]["pool_max_tm"] == pytest.approx(tm[0])
    assert "not a selector" in result["oracle_statement"] and "prohibited" in result["oracle_statement"]
    assert "development-only" in result["development_only"] and "cannot be tested" in result["development_only"]
    assert "tm_oracle" not in result["selectors"]  # the ceiling is never listed among the selectors


def test_overlapping_candidates_are_refused():
    dev = records(100, 1000, "d")
    with pytest.raises(ValueError, match="both"):
        bud.analyse(dev, dev[:30] + records(400, 5000, "p"), {"T": make_model("T", 10, 1)}, 5)


def test_determinism_bound_is_computed():
    base = records(4, 1, "a")
    twin = [{**base[0], "id": "x", "tm_score": base[0]["tm_score"] + 0.004}, {**base[1], "id": "y"}]
    out = bud.determinism(base + twin)
    assert out["n_records"] == 6 and out["n_distinct_domain_sequence_pairs"] == 4
    assert out["n_pairs_with_more_than_one_record"] == 2 and out["n_pairs_with_two_distinct_tm_values"] == 1
    assert out["max_abs_tm_difference"] == pytest.approx(0.004)
