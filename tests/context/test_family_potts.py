"""Conditions the external Potts family reference must always satisfy.

The algebra (bias decomposition, empty prefix, sampler determinism, loader refusals) is
held on a synthetic model written through the real exporter. The sign convention and
column correspondence are held on the exported domains themselves and skip when those
data-side artifacts are absent from a checkout.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.context.family_potts import (  # noqa: E402
    DEFAULT_DIRECTORY, guidance_biases, load_family_model, potts_score, sample_potts, save_family_model)
from src.capability.core.amino_acids import AA20  # noqa: E402

L = 7
DOMAINS = ["1E0L", "1F0M", "1IFY", "1K1V", "1UFM", "1V1C", "2K1B", "2KWH", "2LGW", "2LVN", "3DKM", "3L1X"]


@pytest.fixture()
def exported(tmp_path):
    rng = np.random.default_rng(0)
    wildtype = "".join(rng.choice(list(AA20), size=L))
    freq = rng.dirichlet(np.ones(20), size=L)
    save_family_model(tmp_path, "SYN", wildtype=wildtype, h=rng.normal(size=(L, 20)),
                      pairs=rng.normal(size=(L * (L - 1) // 2, 20, 20)), profile_freq=freq,
                      profile_column_weight=rng.uniform(5, 50, size=L), qualified=True,
                      qualification_failures=[], extra={})
    return tmp_path


def test_full_bias_is_fields_plus_coupling_only(exported):
    model = load_family_model("SYN", exported)
    prefix = model.wildtype[:L - 1]
    for t in range(L):
        b = guidance_biases(model, prefix[:t])
        np.testing.assert_allclose(b["full"], model.h[t] + b["coupling"], rtol=0, atol=1e-12)
        assert b["marginal"].shape == b["coupling"].shape == b["full"].shape == (20,)


def test_empty_prefix_has_exactly_zero_coupling_bias(exported):
    b = guidance_biases(load_family_model("SYN", exported), "")
    assert np.array_equal(b["coupling"], np.zeros(20))


def test_marginal_is_a_normalised_pseudocounted_log_profile(exported):
    model = load_family_model("SYN", exported)
    b = guidance_biases(model, "")
    assert np.isclose(np.exp(b["marginal"]).sum(), 1.0)
    count = model.profile_freq[0] * model.profile_column_weight[0]
    expected = (count + 0.5) / (model.profile_column_weight[0] + 10.0)
    np.testing.assert_allclose(np.exp(b["marginal"]), expected, rtol=1e-6)


def test_bias_matches_score_differences_at_the_last_position(exported):
    """Orientation of J: at t = L-1 no later coupling exists, so bias gaps equal score gaps."""
    model = load_family_model("SYN", exported)
    prefix = model.wildtype[:L - 1]
    full = guidance_biases(model, prefix)["full"]
    scores = np.array([potts_score(model, prefix + a) for a in AA20])
    np.testing.assert_allclose(scores - scores[0], full - full[0], atol=1e-9)


def test_potts_score_matches_explicit_double_loop(exported):
    model = load_family_model("SYN", exported)
    codes = [AA20.index(a) for a in model.wildtype]
    expected = sum(model.h[i, c] for i, c in enumerate(codes)) + sum(
        model.J[i, j, codes[i], codes[j]] for i in range(L) for j in range(i + 1, L))
    assert potts_score(model, model.wildtype) == pytest.approx(expected, abs=1e-9)


def test_sampler_is_seed_deterministic_canonical_and_full_length(exported):
    model = load_family_model("SYN", exported)
    first = sample_potts(model, np.random.default_rng(7))
    assert first == sample_potts(model, np.random.default_rng(7))
    draws = {sample_potts(model, np.random.default_rng(s)) for s in range(20)}
    assert len(draws) > 1
    assert all(len(d) == L and set(d) <= set(AA20) for d in draws | {first})


def test_guidance_refuses_a_complete_or_noncanonical_prefix(exported):
    model = load_family_model("SYN", exported)
    with pytest.raises(ValueError):
        guidance_biases(model, model.wildtype)
    with pytest.raises(ValueError):
        guidance_biases(model, "AX")


def test_loader_refuses_unqualified_unless_explicitly_allowed(tmp_path):
    rng = np.random.default_rng(3)
    save_family_model(tmp_path, "BAD", wildtype="ACDEF", h=rng.normal(size=(5, 20)), pairs=rng.normal(size=(10, 20, 20)),
                      profile_freq=rng.dirichlet(np.ones(20), size=5), profile_column_weight=np.full(5, 10.0),
                      qualified=False, qualification_failures=["gain <= 0"], extra={})
    with pytest.raises(ValueError, match="not qualified"):
        load_family_model("BAD", tmp_path)
    model = load_family_model("BAD", tmp_path, allow_unqualified=True)
    assert not model.qualified and model.qualification_failures == ("gain <= 0",)


def test_loader_refuses_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_family_model("SYN", tmp_path)


def test_loader_refuses_digest_mismatch(exported):
    side = exported / "SYN.json"
    data = json.loads(side.read_text())
    data["npz_sha256"] = "0" * 64
    side.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest"):
        load_family_model("SYN", exported)


def test_loader_refuses_shape_inconsistent_parameters_even_with_a_valid_digest(exported):
    from src.capability.context.family_potts import sha256_file
    npz = exported / "SYN.npz"
    with np.load(npz) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["J"] = arrays["J"][:-1]
    np.savez_compressed(npz, **arrays)
    side = exported / "SYN.json"
    data = json.loads(side.read_text())
    data["npz_sha256"] = sha256_file(npz)
    side.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="shape"):
        load_family_model("SYN", exported)


def test_exporter_refuses_a_column_count_different_from_the_wildtype(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        save_family_model(tmp_path, "BAD", wildtype="ACDEF", h=np.zeros((4, 20)), pairs=np.zeros((10, 20, 20)),
                          profile_freq=np.zeros((5, 20)), profile_column_weight=np.zeros(5),
                          qualified=False, qualification_failures=["x"], extra={})


EXPORTED = [d for d in DOMAINS if (DEFAULT_DIRECTORY / f"{d}.npz").exists()]
needs_artifacts = pytest.mark.skipif(not EXPORTED, reason="exported family references absent from this checkout")


@needs_artifacts
@pytest.mark.parametrize("domain", EXPORTED)
def test_exported_columns_are_the_wildtype_residues(domain):
    model = load_family_model(domain, allow_unqualified=True)
    query = {}
    name = None
    for line in (REPO_ROOT / "data/pairwise_assets/same12_provenance_20260924/query_12.faa").read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].strip().removesuffix(".pdb")
        else:
            query[name] = query.get(name, "") + line.strip()
    assert model.wildtype == query[domain]
    assert model.h.shape == (len(model.wildtype), 20) == model.log_profile.shape
    assert model.J.shape == (model.length, model.length, 20, 20)
    assert list(model.column_to_wt) == list(range(len(model.wildtype)))


@needs_artifacts
@pytest.mark.parametrize("domain", EXPORTED)
def test_wildtype_outscores_composition_matched_shuffles(domain):
    model = load_family_model(domain, allow_unqualified=True)
    rng = np.random.default_rng(1)
    wt = potts_score(model, model.wildtype)
    shuffles = [potts_score(model, "".join(rng.permutation(list(model.wildtype)))) for _ in range(100)]
    assert wt > max(shuffles)
