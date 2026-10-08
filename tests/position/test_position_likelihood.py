"""The two invariants a per-position likelihood array has to satisfy to be one.

A position-resolved array is only a finer view of a published scalar if it
re-reduces to that scalar exactly, and it is only a measurement of propagation if
the positions a causal model could not have seen differently are bit-identical
between the two states. Both are asserted here, on a synthetic arm whose
arithmetic is fully controlled and, where the checkpoint resolves on this host,
on a real one.
"""

from importlib import util as import_util
from pathlib import Path

import numpy as np
import pytest
import torch

from src.capability.position.position_likelihood import (
    CAUSAL,
    MASKED,
    MASKED_ARMS,
    PackedState,
    PositionArchive,
    PositionScore,
    available_arms,
    default_dtype,
    paradigm_of,
    read_archive,
    state_terms,
    upstream_invariance,
)
from src.capability.position.position_terms import retention_residual

ROOT = Path(__file__).resolve().parents[2]
WILDTYPE = "MKQLEDKVEE"


def packed(sequence: str, *, first: int = 101) -> PackedState:
    """One token per residue behind a single marker token."""

    ids = (first,) + tuple(200 + ord(residue) for residue in sequence)
    return PackedState(
        sequence=sequence,
        ids=ids,
        span=(1, len(ids)),
        counts=np.ones(len(sequence), dtype=np.int64),
        offset=0,
    )


def scored(terms) -> PositionScore:
    """A score whose scalar is the device float32 reduction of its own vector."""

    array = np.asarray(terms, dtype=np.float32)
    total = float(torch.as_tensor(array, dtype=torch.float32).sum())
    return PositionScore(
        terms=array,
        nll_sum=total,
        retention_residual_nats=float(retention_residual(array, total, "cpu")),
    )


def test_retained_vector_reduces_to_its_own_scalar_identically():
    score = scored([0.5, 1.25, 2.0, 0.125, 3.5, 0.75, 1.5, 0.25, 2.25, 1.0])
    assert score.retention_residual_nats == 0.0


def test_archive_sum_reproduces_the_scalar_and_the_signed_difference(tmp_path):
    wild = scored([0.5, 1.25, 2.0, 0.125, 3.5, 0.75, 1.5, 0.25, 2.25, 1.0])
    mutant = scored([0.5, 1.25, 2.0, 0.125, 4.0, 1.25, 1.5, 0.25, 2.25, 1.0])
    archive = PositionArchive(assay="synthetic", paradigm=CAUSAL)
    archive.add_wildtype(packed(WILDTYPE), wild)
    archive.add_mutant("E5K", packed("MKQLKDKVEE"), mutant)
    path = tmp_path / "synthetic.npz"
    archive.write(path, metadata={"assay": "synthetic"})
    payload = read_archive(path)

    # The frozen contract: the wild-type log likelihood is minus the summed NLL,
    # and each mutant's entry is logP(mutant) - logP(wild type).
    assert float(payload["wt_likelihood"]) == -wild.nll_sum
    assert float(payload["likelihood"][0]) == wild.nll_sum - mutant.nll_sum
    for index, score in enumerate((wild, mutant)):
        reduced = float(torch.as_tensor(state_terms(payload, index)).sum())
        assert reduced == score.nll_sum
    assert float(payload["position_sum_check_nats"]) == 0.0
    assert payload["position_nats"].dtype == np.float32


def test_archive_refuses_a_causal_retention_residual_it_cannot_explain():
    wild = scored([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    drifted = PositionScore(
        terms=np.ones(10, dtype=np.float32),
        nll_sum=10.5,
        retention_residual_nats=-0.5,
    )
    archive = PositionArchive(assay="synthetic", paradigm=CAUSAL)
    archive.add_wildtype(packed(WILDTYPE), wild)
    archive.add_mutant("E5K", packed("MKQLKDKVEE"), drifted)
    with pytest.raises(ValueError, match="not identically zero"):
        archive.arrays()


def test_a_causal_arms_terms_before_the_substitution_are_bit_identical():
    wild = scored([0.5, 1.25, 2.0, 0.125, 3.5, 0.75, 1.5, 0.25, 2.25, 1.0])
    mutant = scored([0.5, 1.25, 2.0, 0.125, 4.0, 1.25, 9.5, 0.25, 2.25, 1.0])
    assert upstream_invariance(
        packed(WILDTYPE), packed("MKQLKDKVEE"), wild, mutant, 4
    ) == 0.0


def test_a_disturbed_prefix_term_is_reported_rather_than_tolerated():
    wild = scored([0.5, 1.25, 2.0, 0.125, 3.5, 0.75, 1.5, 0.25, 2.25, 1.0])
    mutant = scored([0.5, 1.25, 2.0, 0.126, 4.0, 1.25, 1.5, 0.25, 2.25, 1.0])
    worst = upstream_invariance(packed(WILDTYPE), packed("MKQLKDKVEE"), wild, mutant, 4)
    assert worst > 0.0
    assert worst == pytest.approx(0.001, abs=1e-5)


def test_an_empty_archive_and_a_shadowed_contract_key_are_refused():
    archive = PositionArchive(assay="synthetic", paradigm=CAUSAL)
    archive.add_wildtype(packed(WILDTYPE), scored(np.ones(10)))
    with pytest.raises(ValueError, match="wild type and one mutant"):
        archive.arrays()
    archive.add_mutant("E5K", packed("MKQLKDKVEE"), scored(np.ones(10)))
    with pytest.raises(ValueError, match="shadow the frozen contract"):
        archive.arrays(extras={"likelihood": np.zeros(1)})
    with pytest.raises(ValueError, match="duplicate mutant"):
        archive.add_mutant("E5K", packed("MKQLKDKVEE"), scored(np.ones(10)))


def test_a_packing_that_does_not_cover_its_own_sequence_is_refused():
    with pytest.raises(ValueError, match="not inside"):
        PackedState(sequence="MK", ids=(1, 2, 3), span=(0, 3),
                    counts=np.ones(3, dtype=np.int64), offset=0)
    with pytest.raises(ValueError, match="one residue count per scored token"):
        PackedState(sequence="MK", ids=(1, 2, 3), span=(1, 3),
                    counts=np.ones(3, dtype=np.int64), offset=0)
    with pytest.raises(ValueError, match="cover the sequence suffix"):
        PackedState(sequence="MKQ", ids=(1, 2, 3), span=(1, 3),
                    counts=np.ones(2, dtype=np.int64), offset=0)


def test_a_negative_likelihood_term_is_not_a_negative_log_likelihood():
    with pytest.raises(ValueError, match="non-negative"):
        PositionScore(terms=np.asarray([-0.5, 1.0], dtype=np.float32), nll_sum=0.5,
                      retention_residual_nats=0.0)
    with pytest.raises(ValueError, match="float32"):
        PositionScore(terms=np.asarray([0.5, 1.0]), nll_sum=1.5,
                      retention_residual_nats=0.0)


def test_the_conditioned_arm_and_an_unknown_name_are_refused_by_name():
    with pytest.raises(ValueError, match="EC conditioning tag"):
        paradigm_of("zymctrl")
    with pytest.raises(ValueError, match="no position-likelihood door"):
        paradigm_of("not-an-arm")
    assert paradigm_of("progen2-small") == CAUSAL
    assert paradigm_of("esm2-650m") == MASKED
    assert default_dtype("progen3-3b") == "bfloat16"
    assert default_dtype("galactica-1.3b") == "float32"
    assert "progen2-xlarge" in available_arms() and "prollama" in available_arms()


def test_the_bidirectional_checkpoint_directories_match_their_prior_declaration():
    """A wrong directory is a different model, not a slower run."""

    path = ROOT / "scripts/capability/interactions/exploratory_denoising_arms.py"
    spec = import_util.spec_from_file_location("exploratory_roster_source", path)
    assert spec is not None and spec.loader is not None
    module = import_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prior = {name: entry["directory"] for name, entry in module.EXPLORATORY_ROSTER.items()}
    assert MASKED_ARMS == prior


@pytest.mark.parametrize("arm", ["gpt2"])
def test_a_real_arm_reduces_its_retained_vector_to_its_published_scalar(arm):
    """The invariant against a loaded checkpoint, on CPU.

    Skipped rather than failed when the weights are not on this host: the
    environment resolves checkpoint locations from ``.env.local`` and a public
    checkout has none. The synthetic tests above cover the arithmetic; this one
    covers the integration with the arm's own packing and reduction.
    """

    from src.capability.core import arms as arms_module
    from src.capability.position.position_likelihood import load_position_scorer

    try:
        directory = Path(arms_module.arm_spec(arm).path)
    except Exception as error:  # pragma: no cover - host configuration
        pytest.skip(f"{arm}: checkpoint location unresolved ({error})")
    if not directory.is_dir():  # pragma: no cover - host configuration
        pytest.skip(f"{arm}: no checkpoint at {directory}")
    scorer = load_position_scorer(arm, device="cpu", dtype="float32")
    state = scorer.pack("MKQLEDKVEELLSKNYHLENEVARLKKLV")
    score = scorer.score([state])[0]
    assert score.retention_residual_nats == 0.0
    assert score.terms.shape == (state.scored_tokens,)
    assert float(torch.as_tensor(score.terms).sum()) == score.nll_sum
    assert int(state.counts.sum()) + state.offset == len(state.sequence)
