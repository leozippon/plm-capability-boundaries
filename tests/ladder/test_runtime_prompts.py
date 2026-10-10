"""Token alignment of the causal prefix prompt, and why it is not optional.

Measured on the real checkpoint before this was written: ProtGPT2 prompted with
a FASTA prefix whose last token was the dangling single residue ``V`` emitted
the end-of-text token on 8 of 8 draws, while the same backbone cut two residues
earlier filled every draw. A prompt cut inside a byte-pair token is a string the
arm's own renderer never produces, and a ladder built on such cuts measures where
the merges fell rather than modification extent.

These tests exercise the predicate that fixes it -- the prompt's own tokenisation
must be a prefix of the parent's -- against a stub tokenizer whose merges are
known, so the search's behaviour is checkable without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.capability.ladder import design, runtime


class _ChunkTokenizer:
    """A deliberately coarse tokenizer: fixed-width chunks, greedy from the left.

    It stands in for a multi-residue byte-pair vocabulary. The only property the
    aligner depends on is the one this reproduces: re-tokenising a prefix that
    ends inside a chunk does *not* give a prefix of the whole string's ids.
    """

    def __init__(self, width: int = 4) -> None:
        self.width = width
        self.vocabulary: dict[str, int] = {}

    def _identifier(self, piece: str) -> int:
        return self.vocabulary.setdefault(piece, len(self.vocabulary) + 1)

    def __call__(self, text: str):
        pieces = [text[index : index + self.width] for index in range(0, len(text), self.width)]
        return {"input_ids": [self._identifier(piece) for piece in pieces]}


def _handle(tokenizer, *, terminal_marker=None, prefix=""):
    """A stub arm: the three attributes :func:`aligned_prefix` actually reads."""

    def render(sequence: str, *, ec_label: str | None = None) -> str:
        return prefix + sequence + (terminal_marker or "")

    return SimpleNamespace(render=render, tokenizer=tokenizer, terminal_marker=terminal_marker)


PARENT = "ACDEFGHIKLMNPQRSTVWY" * 10


def test_an_aligned_start_is_found_at_the_nearest_token_boundary():
    handle = _handle(_ChunkTokenizer(width=4))
    found = runtime.aligned_prefix(handle, PARENT, 83, ec_label=None, max_extent=40)
    assert found is not None
    # 84 is the nearest multiple of four, one residue above the anchor.
    assert found["start"] == 84
    assert found["residues_from_anchor"] == 1
    assert found["prompt"] == PARENT[:84]
    assert found["n_prompt_tokens"] == 21


def test_an_anchor_already_on_a_boundary_is_not_moved():
    handle = _handle(_ChunkTokenizer(width=4))
    found = runtime.aligned_prefix(handle, PARENT, 80, ec_label=None, max_extent=40)
    assert found["start"] == 80 and found["residues_from_anchor"] == 0


def test_the_search_prefers_the_lower_boundary_when_both_are_equally_near():
    handle = _handle(_ChunkTokenizer(width=4))
    found = runtime.aligned_prefix(handle, PARENT, 82, ec_label=None, max_extent=40)
    assert found["start"] == 80


def test_a_terminal_marker_is_stripped_before_the_alignment_test():
    handle = _handle(_ChunkTokenizer(width=4), terminal_marker="<end>")
    found = runtime.aligned_prefix(handle, PARENT, 80, ec_label=None, max_extent=40)
    assert found is not None
    assert not found["prompt"].endswith("<end>")
    assert found["prompt"] == PARENT[:80]


def test_a_residue_level_tokenizer_never_has_to_move_the_start():
    handle = _handle(_ChunkTokenizer(width=1))
    for anchor in (55, 80, 101):
        found = runtime.aligned_prefix(handle, PARENT, anchor, ec_label=None, max_extent=40)
        assert found["start"] == anchor and found["residues_from_anchor"] == 0


def test_no_boundary_inside_the_radius_is_reported_rather_than_forced():
    # One chunk of 150 residues: the only boundaries are 0 and 150, and neither
    # lies inside the radius around the anchor.
    handle = _handle(_ChunkTokenizer(width=150))
    assert design.TOKEN_ALIGNMENT_RADIUS < 60
    assert runtime.aligned_prefix(handle, PARENT, 83, ec_label=None, max_extent=40) is None


def test_the_search_will_not_return_a_start_whose_window_leaves_the_chain():
    handle = _handle(_ChunkTokenizer(width=1))
    # Only starts at or below 159 keep a 40-residue window strictly interior.
    found = runtime.aligned_prefix(handle, PARENT, 175, ec_label=None, max_extent=40)
    assert found is not None
    assert found["start"] + 40 <= len(PARENT) - 1


def test_the_outward_walk_visits_the_nearest_offsets_first():
    assert list(runtime._outward(3)) == [0, -1, 1, -2, 2, -3, 3]


@pytest.mark.parametrize(
    "continuation,residues,extent,expected",
    [
        ("ACDEF", "ACDEF", 5, runtime.DRAW_FILLED),
        ("ACD<end>", "ACD", 5, runtime.DRAW_TERMINATED_SHORT),
        ("AC!DE", "AC", 5, runtime.DRAW_OFF_ALPHABET),
        ("ACD", "ACD", 5, runtime.DRAW_BUDGET_EXHAUSTED),
    ],
)
def test_every_declared_draw_status_is_reachable(continuation, residues, extent, expected):
    assert runtime.classify_draw(continuation, residues, extent, end_delimiter="<end>") == expected
    assert expected in runtime.DRAW_STATUSES
