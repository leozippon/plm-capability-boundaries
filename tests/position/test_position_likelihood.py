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
    assert default_dtype("progen3-112m") == "bfloat16"
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


# ------------------------------- the tiered prefix invariant


def test_the_prefix_tolerance_agrees_with_the_lanes_own_declaration():
    """A second tolerance that drifted from the first would be a second standard."""

    from src.capability.position.position_likelihood import (
        TIER2_NATS, TIER2_REPEAT_MULTIPLE,
    )

    path = ROOT / "scripts/capability/position/analyse_position_terms.py"
    spec = import_util.spec_from_file_location("position_terms_analysis_source", path)
    assert spec is not None and spec.loader is not None
    module = import_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert TIER2_NATS == module.TIER2_NATS
    assert TIER2_REPEAT_MULTIPLE == module.TIER2_REPEAT_MULTIPLE


def test_a_deterministic_arm_has_no_room_above_exact_zero():
    """Repeat maximum zero means tier 2 is unreachable, so the invariant stays exact."""

    from src.capability.position.position_likelihood import residual_tier

    assert residual_tier(0.0, 0.0) == 1
    # The batch-composition drift this work measured on progen2-small, whose forward
    # reproduces bit-for-bit: it must stay detectable.
    assert residual_tier(3.24e-5, 0.0) == 3
    assert residual_tier(1e-12, 0.0) == 3


def test_a_nondeterministic_arm_is_admitted_only_within_its_own_reproducibility():
    """The rita-xl case, and why the repeat probe has to span the scored cohort.

    Measured end to end on ``rita-xl`` over the thirty structure-mapped assays, the
    arm-level worst upstream residual and the arm-level worst repeat difference are
    the *same number*, 8.702e-06 nats: 27 assays are exactly zero and the three that
    are not have ratios of 0.76, 1.00 and 1.00. That equality is what makes a
    three-fold multiple an adequate margin -- but only when the repeat maximum is a
    maximum over the same assays as the residual. A repeat estimate taken from a
    smaller or shorter population does not license a residual drawn from a larger
    one, and the rule refuses rather than guesses.
    """

    from src.capability.position.position_likelihood import TIER2_NATS, residual_tier

    # Measured on the same population: the residual and the repeat coincide.
    assert residual_tier(8.702e-06, 8.702e-06) == 2
    assert residual_tier(1.8e-5, 1.8e-5) == 2
    assert residual_tier(1.8e-5, 6.1e-6) == 2
    # An under-powered repeat estimate does not admit a larger residual.
    assert residual_tier(1.8e-5, 5.0e-6) == 3
    assert residual_tier(2.0e-5, 5.0e-6) == 3
    # The hard cap binds however large the measured repeat is.
    assert residual_tier(TIER2_NATS, 1.0) == 2
    assert residual_tier(TIER2_NATS * 1.001, 1.0) == 3
    # The ProGen3 layout shift is refused at any plausible reproducibility.
    assert residual_tier(0.5664, 5.0e-6) == 3
    assert residual_tier(0.5664, 0.01) == 3


def test_the_repeat_residual_measures_two_forwards_of_one_row():
    first = scored([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0])
    same = scored([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0])
    jittered = scored([1.0, 2.0, 3.0, 1.0, 2.000004, 3.0, 1.0, 2.0, 3.0, 1.0])
    from src.capability.position.position_likelihood import repeat_residual

    assert repeat_residual(first, same) == 0.0
    assert repeat_residual(first, jittered) == pytest.approx(4e-6, rel=5e-2)
    with pytest.raises(ValueError, match="same row"):
        repeat_residual(first, scored([1.0, 2.0]))


def test_the_layout_sensitive_mixture_arm_is_refused_by_name_with_its_measurement():
    from src.capability.position.position_likelihood import REFUSED_ARMS

    assert "progen3-3b" in REFUSED_ARMS
    reason = REFUSED_ARMS["progen3-3b"]
    assert "0.5317" in reason and "layout_assessment.json" in reason
    with pytest.raises(ValueError, match="expert mixture"):
        paradigm_of("progen3-3b")
    # progen3-112m is not refused: its prefix residual was exactly zero on the whole
    # scored cohort, which is a measurement on that checkpoint rather than a guarantee
    # from the architecture, and the invariant is what would catch it if it changed.
    assert "progen3-112m" not in REFUSED_ARMS
    assert paradigm_of("progen3-112m") == CAUSAL
