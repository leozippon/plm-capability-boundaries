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
