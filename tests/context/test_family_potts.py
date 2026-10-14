"""Conditions the external Potts family reference must always satisfy.

The algebra (bias decomposition, empty prefix, sampler determinism, loader refusals) is
held on a synthetic model written through the real exporter. The sign convention and
column correspondence are held on the exported domains themselves and skip when those
data-side artifacts are absent from a checkout.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path
import json
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.context.family_potts import (  # noqa: E402
    MixingError, validate_gibbs_mixing,
    DEFAULT_DIRECTORY, guidance_biases, load_family_model, potts_score, sample_potts_joint,
    sample_prefix_chain, save_family_model)
from src.capability.context.potts_gauge import to_zero_sum  # noqa: E402
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


def test_prefix_chain_is_seed_deterministic_canonical_and_full_length(exported):
    model = load_family_model("SYN", exported)
    first = sample_prefix_chain(model, np.random.default_rng(7))
    assert first == sample_prefix_chain(model, np.random.default_rng(7))
    draws = {sample_prefix_chain(model, np.random.default_rng(s)) for s in range(20)}
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


# ----------------------------------------------- joint versus prefix-chain sampling, and the gauge


def _tiny_model(tmp_path, length=3, seed=5, scale=0.8):
    rng = np.random.default_rng(seed)
    save_family_model(tmp_path, "TINY", wildtype="".join(rng.choice(list(AA20), size=length)),
                      h=rng.normal(scale=scale, size=(length, 20)),
                      pairs=rng.normal(scale=scale, size=(length * (length - 1) // 2, 20, 20)),
                      profile_freq=rng.dirichlet(np.ones(20), size=length),
                      profile_column_weight=np.full(length, 20.0), qualified=True,
                      qualification_failures=[], extra={})
    return load_family_model("TINY", tmp_path)


def _exact_joint(model):
    """Exact ``P_L`` over all 20^L sequences, as a tensor indexed by residue codes."""
    codes = np.array(list(product(range(20), repeat=model.length)))
    energy = np.array([potts_score(model, "".join(AA20[c] for c in row)) for row in codes])
    p = np.exp(energy - energy.max())
    return (p / p.sum()).reshape((20,) * model.length), energy


def _codes(sequences):
    return np.array([[AA20.index(a) for a in s] for s in sequences])


def _tv(a, b):
    return 0.5 * np.abs(a - b).sum()


def test_gibbs_sampler_matches_the_exact_joint_marginals(tmp_path):
    """Ground truth by enumeration: single-site and one pairwise marginal of the Gibbs draws."""
    model = _tiny_model(tmp_path)
    exact, _ = _exact_joint(model)
    n = 20000
    draws = sample_potts_joint(model, n, np.random.default_rng(1), sweeps=40, mixing_tolerance=0.05)
    X = _codes(draws.sequences)
    assert (draws.sweeps, draws.check_sweeps) == (40, 20)
    for i in range(model.length):
        empirical = np.bincount(X[:, i], minlength=20) / n
        exact_i = exact.sum(axis=tuple(k for k in range(model.length) if k != i))
        assert _tv(empirical, exact_i) < 0.03
    pair = np.zeros((20, 20))
    np.add.at(pair, (X[:, 0], X[:, 2]), 1.0)
    assert _tv(pair / n, exact.sum(axis=1)) < 0.08


def test_prefix_chain_is_not_the_potts_joint(tmp_path):
    """The renamed chain samples Q, not P_L: exact Q differs from P_L and sits at lower energy."""
    model = _tiny_model(tmp_path)
    exact, energy = _exact_joint(model)
    # exact chain distribution Q(x) = prod_t softmax_t(h_t + sum_{s<t} J_ts(., x_s))[x_t]
    q = np.ones((20,) * model.length)
    for idx in np.ndindex(*q.shape):
        for t in range(model.length):
            logits = guidance_biases(model, "".join(AA20[c] for c in idx[:t]))["full"]
            q[idx] *= np.exp(logits - logits.max())[idx[t]] / np.exp(logits - logits.max()).sum()
    assert q.sum() == pytest.approx(1.0)
    assert _tv(q.ravel(), exact.ravel()) > 0.01
    assert (q.ravel() * energy).sum() < (exact.ravel() * energy).sum()
    # and the sampler really draws from Q
    X = _codes([sample_prefix_chain(model, np.random.default_rng(s)) for s in range(3000)])
    assert np.abs(np.bincount(X[:, 0], minlength=20) / 3000 - q.sum(axis=(1, 2))).max() < 0.03


def test_gibbs_sampler_refuses_chains_that_have_not_mixed(exported):
    model = load_family_model("SYN", exported)
    with pytest.raises(RuntimeError, match="not mixed"):
        sample_potts_joint(model, 500, np.random.default_rng(0), sweeps=2, mixing_tolerance=1e-4)
    with pytest.raises(ValueError):
        sample_potts_joint(model, 500, np.random.default_rng(0), sweeps=1, mixing_tolerance=0.1)


def test_zero_sum_gauge_shifts_energy_by_a_constant_but_not_the_per_step_biases(exported):
    """Documents the gauge dependence recorded in the module docstring; it asserts no invariance of C."""
    model = load_family_model("SYN", exported)
    h2, J2 = to_zero_sum(model.h, model.J)
    moved = replace(model, h=h2, J=J2)
    rng = np.random.default_rng(2)
    seqs = ["".join(rng.choice(list(AA20), size=L)) for _ in range(30)]
    shifts = np.array([potts_score(moved, x) - potts_score(model, x) for x in seqs])
    np.testing.assert_allclose(shifts, shifts[0], atol=1e-9)  # E (hence accumulated F) invariant up to a constant
    centred = lambda v: v - v.mean()  # noqa: E731  biases are defined up to a constant in a
    for x in seqs:
        for t in range(L):
            b0, b1 = guidance_biases(model, x[:t]), guidance_biases(moved, x[:t])
            past = model.J[t, :t].mean(axis=2).sum(axis=0)       # sum_{s<t} mean_b J_ts(a, b)
            future = model.J[t, t + 1:].mean(axis=2).sum(axis=0)  # sum_{j>t} mean_b J_tj(a, b)
            np.testing.assert_allclose(centred(b1["coupling"] - b0["coupling"]), centred(-past), atol=1e-9)
            np.testing.assert_allclose(centred(b1["full"] - b0["full"]), centred(future), atol=1e-9)
    assert np.abs(centred(past)).max() > 0  # the C shift is a real, nonzero first-order term


def test_guidance_bias_is_the_prefix_truncated_conditional_and_not_the_full_joint_conditional(tmp_path):
    """``guidance_biases(...)["full"]`` is ``log P_t(x_t=a | x_<t)`` of the model restricted to columns 0..t.

    Checked by brute-force enumeration of a synthetic L=4 model (20^4 sequences): after removing one constant over
    the twenty residues the residual is at machine precision. It is not ``log P_L(x_t=a | x_<t)``, which also
    sums out the later columns. A synthetic model suffices because the claim is arithmetic and holds for any h, J;
    a real reference has L >= 37, so only a few columns of one could be enumerated and that is the same model.
    Not asserted here: that the realised step distribution is the product of the language model and this
    conditional, which needs a loaded language model (see tests/decoding).
    """
    model = _tiny_model(tmp_path, length=4, seed=11, scale=0.8)
    idx = np.arange(4)
    all_codes = np.array(list(product(range(20), repeat=4)))
    energy = (model.h[idx, all_codes].sum(axis=1)
              + sum(model.J[i, j, all_codes[:, i], all_codes[:, j]] for i in range(4) for j in range(i + 1, 4)))
    rng = np.random.default_rng(0)
    for t in (1, 2, 3):
        prefix = rng.integers(0, 20, size=t)
        match = (all_codes[:, :t] == prefix).all(axis=1)
        full = np.zeros(20)  # log P_L(x_t = a | x_<t): marginalise columns t+1..L-1 over all 20^(L-1-t)
        for a in range(20):
            full[a] = np.logaddexp.reduce(energy[match & (all_codes[:, t] == a)])
        # restricted model: drop every column after t, enumerate columns 0..t
        sub = all_codes[:, :t + 1]
        sub = sub[np.unique(sub, axis=0, return_index=True)[1]]
        e_sub = (model.h[np.arange(t + 1), sub].sum(axis=1)
                 + sum(model.J[i, j, sub[:, i], sub[:, j]] for i in range(t + 1) for j in range(i + 1, t + 1)))
        in_prefix = (sub[:, :t] == prefix).all(axis=1)
        truncated = np.array([np.logaddexp.reduce(e_sub[in_prefix & (sub[:, t] == a)]) for a in range(20)])
        truncated -= np.logaddexp.reduce(truncated)
        bias = guidance_biases(model, "".join(AA20[c] for c in prefix))["full"]
        residual = (bias - truncated) - (bias - truncated).mean()
        assert np.abs(residual).max() < 1e-12
        full -= np.logaddexp.reduce(full)
        spread = (bias - full) - (bias - full).mean()
        if t < 3:  # the full-joint conditional differs, so the distinction is real ...
            assert np.abs(spread).max() > 1e-3
        else:  # ... except at the last column, where no later column is left to sum out
            assert np.abs(spread).max() < 1e-12


def test_validation_passes_for_a_mixing_sampler_and_flags_a_stuck_one(tmp_path):
    model = _tiny_model(tmp_path, length=3, seed=5, scale=0.8)
    ok = validate_gibbs_mixing(model, np.random.default_rng(0), n=2000, sweeps=40, factor=2.0)
    assert ok["passed"] and ok["between_init_tv"] <= 2.0 * ok["noise_floor"]
    # one sweep leaves the chains near their starts, A uniform and B at the wild type: initialisation dependence must be seen
    with pytest.raises(MixingError) as stuck:
        validate_gibbs_mixing(model, np.random.default_rng(0), n=2000, sweeps=1, factor=2.0)
    assert not stuck.value.numbers["passed"] and stuck.value.numbers["ratio"] > 2.0
    with pytest.raises(ValueError):
        validate_gibbs_mixing(model, np.random.default_rng(0), n=3, sweeps=40, factor=2.0)
