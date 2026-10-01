import numpy as np
import pytest

from src.capability.core.family_grouping import (
    Union, batch_align, encode, parent_background, reference_alignment)

GAP_OPEN, GAP_EXTEND = 11.0, 1.0


def statistics(left, right):
    codes, lengths = encode([left, right])
    out = batch_align(codes, lengths, np.array([[0, 1]]),
                      gap_open=GAP_OPEN, gap_extend=GAP_EXTEND)
    return (float(out.score[0]), int(out.columns[0]), int(out.identical[0]),
            float(out.coverage_a[0]), float(out.coverage_b[0]))


def test_batched_alignment_matches_an_independent_traceback():
    # Identity, an internal substitution, an indel that forces a gap column, a
    # local match inside longer flanks, and a pair with no positive alignment.
    pairs = [
        ("ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY"),
        ("ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWA"),
        ("ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLNPQRSTVWY"),
        ("WWWWACDEFGHIKLMNPQRSTVWYWWWW", "ACDEFGHIKLMNPQRSTVWY"),
        ("PPPPPPPPPP", "WWWWWWWWWW"),
    ]
    for left, right in pairs:
        assert statistics(left, right) == pytest.approx(
            reference_alignment(left, right, gap_open=GAP_OPEN, gap_extend=GAP_EXTEND))
    # A local match inside flanks must not be credited with covering the flanks.
    score, columns, identical, coverage_a, coverage_b = statistics(*pairs[3])
    assert (columns, identical, coverage_b) == (20, 20, 100.0)
    assert coverage_a == pytest.approx(100.0 * 20 / 28)
    assert statistics(*pairs[4]) == (0.0, 0, 0, 0.0, 0.0)


def test_encode_refuses_a_symbol_outside_the_alphabet():
    with pytest.raises(ValueError, match="outside the canonical alphabet"):
        encode(["ACDX"])


def test_parent_background_keeps_engineered_variants_with_their_parent():
    assert parent_background("1BK2.pdb_L5S") == "1BK2.pdb"
    assert parent_background("1BK2.pdb") == "1BK2.pdb"
    assert parent_background("EA|run2_0290_0001.pdb") == "EA|run2_0290_0001.pdb"


def test_union_records_every_edge_and_its_source():
    union = Union(["a", "b", "c"])
    assert union.join("a", "b", "R1") is True
    assert union.join("a", "b", "R2") is False  # already connected, still recorded
    assert union.components() == {"a": ["a", "b"], "c": ["c"]}
    assert union.edges == [("a", "b", "R1"), ("a", "b", "R2")]
    with pytest.raises(KeyError, match="undeclared label"):
        union.join("a", "d", "R1")
