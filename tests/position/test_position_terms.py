"""Conditions the per-position retention must always satisfy.

The retention exists to make a published scalar checkable at finer grain, so the
tests here are about the properties that make it the same quantity: the vector
reduces to the scalar the unmodified expression produces, the residue attribution
reconstructs the sequence it claims to cover or refuses, and the residue-axis
partition is a partition. The failure paths are exercised deliberately, because
every one of them would otherwise produce a well-formed decomposition of the
wrong thing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.position import position_terms as pt


class _Tokenizer:
    """Decodes ids through a table, the way a real tokenizer decodes pieces."""

    def __init__(self, table: dict[int, str]) -> None:
        self.table = table

    def decode(self, ids, clean_up_tokenization_spaces=True):
        return "".join(self.table[int(i)] for i in ids)


class _Arm:
    def __init__(self, table, name="fixture"):
        self.tokenizer = _Tokenizer(table)
        self.name = name


def test_terms_reduce_to_the_unmodified_expression_bit_exactly():
    """``target_nll_terms`` is ``_target_nll`` before its collapse, not beside it."""

    torch.manual_seed(20260926)
    logits = torch.randn(1, 24, 61)
    ids = torch.randint(0, 61, (1, 24))
    start, end = 3, 20
    terms = pt.target_nll_terms(logits, ids, start, end)
    logprobs = torch.nn.functional.log_softmax(logits[0, start - 1 : end - 1].float(), dim=-1)
    expected = -logprobs.gather(-1, ids[0, start:end].unsqueeze(-1)).squeeze(-1)
    assert torch.equal(terms, expected)
    assert terms.dtype == torch.float32
    assert float(terms.sum()) == float(expected.sum())


def test_retention_residual_is_zero_on_the_written_array_and_signed_otherwise():
    torch.manual_seed(1)
    terms = torch.randn(97).float()
    written = terms.numpy()
    assert pt.retention_residual(written, float(terms.sum()), "cpu") == 0.0
    assert pt.retention_residual(written, float(terms.sum()) + 1.0, "cpu") == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        pt.retention_residual(written.astype(np.float64), 0.0, "cpu")


def test_first_scored_position_needs_a_predicting_column():
    with pytest.raises(ValueError):
        pt.target_nll_terms(torch.zeros(1, 4, 5), torch.zeros(1, 4, dtype=torch.long), 0, 3)


def test_coverage_counts_residues_and_recovers_a_suffix_offset():
    """A packing that leaves residue 1 unscored is read by subtraction, not by parsing."""

    table = {0: "[X]", 1: "M", 2: "A", 3: "K", 4: "L"}
    arm = _Arm(table)
    coverage = pt.ResidueCoverage(arm)
    counts, offset = coverage.counts([0, 1, 2, 3, 4], (2, 5), "MAKL")
    assert offset == 1
    assert counts.tolist() == [1, 1, 1]
    assert counts.dtype == np.int16


def test_coverage_handles_merged_and_formatting_tokens():
    table = {0: "MA", 1: "\n", 2: "KLY"}
    coverage = pt.ResidueCoverage(_Arm(table))
    counts, offset = coverage.counts([0, 1, 2], (0, 3), "MAKLY")
    assert offset == 0
    assert counts.tolist() == [2, 0, 3]


def test_coverage_refuses_a_decode_that_does_not_reconstruct_the_sequence():
    coverage = pt.ResidueCoverage(_Arm({0: "MA", 1: "KL"}))
    with pytest.raises(ValueError, match="reconstruct"):
        coverage.counts([0, 1], (0, 2), "MAKY")


def test_coverage_refuses_a_span_claiming_more_residues_than_the_sequence():
    coverage = pt.ResidueCoverage(_Arm({0: "MAKL"}))
    with pytest.raises(ValueError, match="more than"):
        coverage.counts([0], (0, 1), "MA")


def test_partition_is_a_partition_and_places_formatting_tokens_by_position():
    counts = np.asarray([2, 0, 3], dtype=np.int16)
    masks = pt.partition_masks(counts, 1, 3)
    assert masks["upstream"].tolist() == [True, True, False]
    assert masks["own"].tolist() == [False, False, True]
    assert masks["downstream"].tolist() == [False, False, False]
    masks = pt.partition_masks(counts, 1, 1)
    assert masks["own"].tolist() == [True, False, False]
    assert masks["downstream"].tolist() == [False, True, True]


def test_partition_covers_every_token_exactly_once_over_a_position_sweep():
    counts = np.asarray([1, 2, 0, 1, 3, 1], dtype=np.int16)
    for offset in (0, 1, 4):
        total = offset + int(counts.sum())
        for position in range(total):
            masks = pt.partition_masks(counts, offset, position)
            stacked = np.stack([masks[name] for name in pt.PARTITION])
            assert stacked.sum(0).tolist() == [1] * len(counts)


def test_an_unscored_mutated_residue_leaves_no_own_token():
    """ProteinGLM does not score residue 1 and the text renderings drop a prefix."""

    counts = np.asarray([1, 1, 1], dtype=np.int16)
    masks = pt.partition_masks(counts, 1, 0)
    assert not masks["own"].any()
    assert not masks["upstream"].any()
    assert masks["downstream"].all()


def test_multiple_substitutions_place_between_tokens_downstream():
    counts = np.asarray([1, 1, 1, 1, 1], dtype=np.int16)
    masks = pt.partition_masks(counts, 0, [1, 4])
    assert masks["own"].tolist() == [False, True, False, False, True]
    assert masks["upstream"].tolist() == [True, False, False, False, False]
    assert masks["downstream"].tolist() == [False, False, True, True, False]


def test_state_parts_and_mutation_parts_close_exactly():
    rng = np.random.default_rng(20260926)
    counts = np.asarray([1] * 40, dtype=np.int16)
    wild = pt.state_parts(rng.normal(size=40).astype(np.float32), counts, 0, 17)
    mutant = pt.state_parts(rng.normal(size=40).astype(np.float32), counts, 0, 17)
    exact = wild["total"] - mutant["total"]
    record = pt.mutation_parts(wild, mutant, exact)
    assert record["closure_nats"] == pytest.approx(0.0, abs=1e-12)
    assert record["full"] == exact
    assert record["own_scored"] is True
    # The published scalar is what ``full`` reports, so a scalar that differs from
    # the float64 re-summation shows up as a residual instead of being absorbed.
    shifted = pt.mutation_parts(wild, mutant, exact + 1e-5)
    assert shifted["full"] == exact + 1e-5
    assert shifted["closure_nats"] == pytest.approx(1e-5, rel=1e-6)
    assert shifted["closure_bound_nats"] > 0.0


def test_blas_pinning_refuses_an_undeclared_pool(monkeypatch):
    for name in pt.BLAS_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit):
        pt.blas_pinning()
    for name in pt.BLAS_VARIABLES:
        monkeypatch.setenv(name, str(torch.get_num_threads()))
    resolved = pt.blas_pinning(torch.get_num_threads())
    assert resolved["OMP_NUM_THREADS"] == torch.get_num_threads()
    with pytest.raises(SystemExit):
        pt.blas_pinning(torch.get_num_threads() + 1)
    monkeypatch.setenv(pt.BLAS_VARIABLES[0], str(torch.get_num_threads() + 1))
    with pytest.raises(SystemExit):
        pt.blas_pinning()


def _state(ids, span, counts, offset):
    return {'ids': list(ids), 'span': tuple(span),
            'counts': np.asarray(counts, dtype=np.int16), 'offset': int(offset)}


def test_alignment_accepts_a_substitution_confined_to_its_own_token():
    wild = _state([9, 1, 2, 3, 4], (1, 5), [1, 1, 1, 1], 0)
    mutant = _state([9, 1, 7, 3, 4], (1, 5), [1, 1, 1, 1], 0)
    record = pt.alignment(wild, mutant, 1)
    assert record['aligned'] is True
    assert record['differing_tokens'] == 1
    assert record['differing_outside_own'] == 0
    assert record['mismatch_extent_residues'] is None


def test_alignment_refuses_a_retokenisation_that_changes_length():
    wild = _state([9, 1, 2, 3], (1, 4), [1, 2, 1], 0)
    mutant = _state([9, 1, 5, 6, 3], (1, 5), [1, 1, 1, 1], 0)
    record = pt.alignment(wild, mutant, 1)
    assert record['aligned'] is False
    assert record['packed_length_delta'] == 1
    assert 'packed length' in record['reason']


def test_alignment_refuses_a_boundary_move_far_from_the_substitution():
    """Equal length, equal span, but a token differs six residues downstream."""

    wild = _state([9, 1, 2, 3, 4, 5, 6, 7], (1, 8), [1, 1, 1, 1, 1, 1, 1], 0)
    mutant = _state([9, 1, 20, 3, 4, 5, 6, 30], (1, 8), [1, 1, 1, 1, 1, 1, 1], 0)
    record = pt.alignment(wild, mutant, 1)
    assert record['aligned'] is False
    assert record['differing_outside_own'] == 1
    assert record['mismatch_extent_residues'] == 5


def test_alignment_refuses_a_residue_count_change_outside_the_own_token():
    wild = _state([9, 1, 2, 3], (1, 4), [1, 2, 1], 0)
    mutant = _state([9, 5, 2, 8], (1, 4), [1, 2, 2], 0)
    record = pt.alignment(wild, mutant, 0)
    assert record['aligned'] is False


def test_alignment_refuses_a_state_identical_to_its_wild_type():
    wild = _state([9, 1, 2], (1, 3), [1, 1], 0)
    record = pt.alignment(wild, dict(wild), 0)
    assert record['aligned'] is False
    assert 'no token differs' in record['reason']


def test_an_excluded_first_cycle_does_not_consume_the_single_mutant_state():
    """The first cycle that contains a state may be an indel cycle.

    That cycle is not a measurement, and it must not mark the state seen.
    A later kept cycle is the observation the contrast uses. A second kept
    cycle does not add a second copy.
    """

    import importlib.util

    path = ROOT / "scripts/capability/position/analyse_position_terms.py"
    spec = importlib.util.spec_from_file_location("analyse_position_terms_keep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen: set[int] = set()
    assert module.keep_first_unexcluded(1, seen, excluded=True) is False
    assert seen == set()
    assert module.keep_first_unexcluded(1, seen, excluded=False) is True
    assert module.keep_first_unexcluded(1, seen, excluded=False) is False
    assert module.keep_first_unexcluded(2, seen, excluded=True) is False
    assert module.keep_first_unexcluded(3, seen, excluded=False) is True
    assert seen == {1, 3}


def test_assay_shards_partition_the_sorted_list_and_merge(tmp_path):
    """Two shards cover every assay once, and merging restores assay order."""

    import importlib.util

    import numpy as np

    path = ROOT / "scripts/capability/position/analyse_position_terms.py"
    spec = importlib.util.spec_from_file_location("analyse_position_terms_shards", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = [f"a{i:03d}" for i in range(201)]
    shards = [module.assays_for_shard(names, shard, 2) for shard in (0, 1)]
    assert set(shards[0]).isdisjoint(shards[1])
    assert sorted(shards[0] + shards[1]) == names

    def payload(assay, value):
        entries = [{"own": value}]
        return {
            "rows": [{"assay": assay, "cluster": "c", "entries": entries}],
            "t1": [0.0], "t2": np.asarray([value, value + 1.0]),
            "closure": [(0.0, 0.0)], "coverage": [{"own_scored": True}],
            "repeat_max": 0.0, "wave_manifest": {"torch_version": "t"},
            "archive_manifest": {"torch_version": "a"}, "unit": "u", "endpoint": "e",
            "assays": [assay],
        }

    left = tmp_path / "shard0.pkl"
    right = tmp_path / "shard1.pkl"
    module.write_anchor_partial(left, payload("b", 2.0))
    module.write_anchor_partial(right, payload("a", 1.0))
    merged = module.merge_anchor_partials([left, right])
    assert [row["assay"] for row in merged["rows"]] == ["a", "b"]
    assert list(merged["t2"]) == [1.0, 2.0, 2.0, 3.0]


def test_archived_manifest_paths_resolve_from_the_manifest_directory(tmp_path):
    """A merged readout records shard files as paths relative to its manifest."""

    import importlib.util

    path = ROOT / "scripts/capability/position/analyse_position_terms.py"
    spec = importlib.util.spec_from_file_location("analyse_position_terms_paths", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    archive = tmp_path / "merged"
    archive.mkdir()
    bare = archive / "arm_bare.npz"
    bare.write_bytes(b"bare")
    shard = tmp_path / "shard_2"
    shard.mkdir()
    sibling = shard / "arm_sibling.npz"
    sibling.write_bytes(b"sibling")
    extra = tmp_path / "extra"
    extra.mkdir()
    named = extra / "arm_named.npz"
    named.write_bytes(b"named")
    index = module.index_files([archive, extra])

    assert module.locate_archived_npz(archive, "arm_bare.npz", index).resolve() == bare.resolve()
    assert module.locate_archived_npz(
        archive, "../shard_2/arm_sibling.npz", index).resolve() == sibling.resolve()
    assert module.locate_archived_npz(archive, "arm_named.npz", index).resolve() == named.resolve()
    try:
        module.locate_archived_npz(archive, "../shard_2/missing.npz", index)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a missing relative path must fail")


def test_alignment_separates_an_unscored_substitution_from_a_moved_boundary():
    """ProteinGLM does not score residue 1, so a substitution there has no own token."""

    wild = _state([9, 8, 1, 2, 3], (3, 5), [1, 1], 1)
    mutant = _state([9, 8, 5, 2, 3], (3, 5), [1, 1], 1)
    record = pt.alignment(wild, mutant, 0)
    assert record['aligned'] is False
    assert record['differing_outside_scored_span'] == 1
    assert record['mismatch_extent_residues'] is None
    assert 'outside the scored span' in record['reason']


def test_alignment_refuses_boundary_shift_inside_union_of_own_tokens():
    # Both changed tokens lie in the union of own masks, but partitioning these
    # different grids moves an unchanged residue from own to upstream.
    wild = _state([9, 1, 2, 3], (1, 4), [2, 1, 1], 0)
    mutant = _state([9, 7, 8, 3], (1, 4), [1, 2, 1], 0)
    assert not pt.alignment(wild, mutant, 1)['aligned']


def test_scalar_identity_is_not_replaced_by_float64_resummation():
    # Float32 sums tie here although their float64 retained-term sums differ.
    wild = pt.state_parts(np.array([2**24, 1.0], dtype=np.float32), np.array([1,1]), 0, 1)
    mutant = pt.state_parts(np.array([2**24, 0.5], dtype=np.float32), np.array([1,1]), 0, 1)
    record = pt.mutation_parts(wild, mutant, 0.0)
    assert record['full'] == 0.0
    assert record['float64_full'] == 0.5
    assert record['closure_nats'] == -0.5


def test_retokenized_span_recovers_shifted_boundary_and_closes():
    wild = _state([9, 1, 2, 3], (1, 4), [2, 1, 1], 0)
    mutant = _state([9, 7, 8, 3], (1, 4), [1, 2, 1], 0)
    masks=pt.retokenized_span_masks(wild,mutant,1)
    assert masks[0]['own'].tolist()==[True,True,False]
    assert masks[1]['own'].tolist()==[True,True,False]
    w=pt.masked_state_parts(np.array([1.,2.,3.]),wild['counts'],masks[0])
    m=pt.masked_state_parts(np.array([4.,5.,6.]),mutant['counts'],masks[1])
    parts=pt.mutation_parts(w,m,-9.)
    assert parts['upstream']==0 and parts['own']==-6 and parts['downstream']==-3
    assert parts['closure_nats']==0


def test_retokenized_span_handles_different_token_counts_with_same_rejoined_suffix():
    wild=_state([9,1,2,3],(1,4),[1,2,1],0)
    mutant=_state([9,1,7,8,3],(1,5),[1,1,1,1],0)
    masks=pt.retokenized_span_masks(wild,mutant,1)
    assert masks[0]['upstream'].tolist()==[True,False,False]
    assert masks[0]['own'].tolist()==[False,True,False]
    assert masks[1]['own'].tolist()==[False,True,True,False]
    assert masks[0]['downstream'].sum()==masks[1]['downstream'].sum()==1


def test_retokenized_span_refuses_changed_unscored_conditioning_and_unscored_mutation():
    wild=_state([9,1,2],(1,3),[1,1],1)
    mutant=_state([8,3,2],(1,3),[1,1],1)
    with pytest.raises(ValueError,match='conditioning'):
        pt.retokenized_span_masks(wild,mutant,1)
    with pytest.raises(ValueError,match='outside'):
        pt.retokenized_span_masks(wild,mutant,0)
