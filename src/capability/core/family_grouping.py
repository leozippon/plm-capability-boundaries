"""Held-group construction for the measured double-mutant stability cohort.

The grouping contract is frozen in a JSON file before any group is computed and
is passed in rather than restated here, so the scoring rule that produced a
group map is recoverable from the map's own provenance. Nothing in this module
reads a phenotype: every edge is a function of sequence, name or annotation.

The alignment is exact Smith-Waterman with affine gaps, batched over pairs.
:func:`reference_alignment` is the independent per-pair traceback the batched
path is checked against; the check is part of the contract, not a convenience.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np

from .amino_acids import AA20, BLOSUM62_ORDER, BLOSUM62_ROWS

#: Padding code for positions past a sequence's end. Substitutions against it
#: are unreachable because the recurrences mask cells outside both lengths.
PAD_CODE = len(AA20)

_NEG = np.float32(-1.0e9)


def blosum62() -> np.ndarray:
    """BLOSUM62 half-bit scores in :data:`AA20` order, with a padding margin."""

    published = np.array(BLOSUM62_ROWS, dtype=np.int32)
    order = [BLOSUM62_ORDER.index(residue) for residue in AA20]
    table = np.full((PAD_CODE + 1, PAD_CODE + 1), _NEG, dtype=np.float32)
    table[:PAD_CODE, :PAD_CODE] = published[np.ix_(order, order)].astype(np.float32)
    return table


def encode(sequences: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad ``sequences`` into a code matrix and return it with the lengths."""

    index = {residue: code for code, residue in enumerate(AA20)}
    lengths = np.array([len(s) for s in sequences], dtype=np.int32)
    width = int(lengths.max())
    codes = np.full((len(sequences), width), PAD_CODE, dtype=np.int64)
    for row, sequence in enumerate(sequences):
        try:
            codes[row, : len(sequence)] = [index[residue] for residue in sequence]
        except KeyError as exc:
            raise ValueError(
                f"sequence {row} carries {exc.args[0]!r}, outside the canonical alphabet"
            ) from None
    return codes, lengths


@dataclass(frozen=True)
class AlignmentStatistics:
    """Per-pair local-alignment statistics the edge rule consumes."""

    score: np.ndarray
    columns: np.ndarray
    identical: np.ndarray
    coverage_a: np.ndarray
    coverage_b: np.ndarray

    @property
    def percent_identity(self) -> np.ndarray:
        out = np.zeros(len(self.columns), dtype=np.float64)
        positive = self.columns > 0
        out[positive] = 100.0 * self.identical[positive] / self.columns[positive]
        return out

    def edges(self, *, identity_floor: float, coverage_floor: float) -> np.ndarray:
        return (
            (self.columns > 0)
            & (self.percent_identity >= identity_floor)
            & (self.coverage_a >= coverage_floor)
            & (self.coverage_b >= coverage_floor)
        )


def batch_align(
    codes: np.ndarray,
    lengths: np.ndarray,
    pairs: np.ndarray,
    *,
    gap_open: float,
    gap_extend: float,
    matrix: np.ndarray | None = None,
    block: int = 20000,
) -> AlignmentStatistics:
    """Exact Smith-Waterman statistics for every row of ``pairs``.

    The optimum is the highest-scoring match-state cell, ties resolved by the
    first such cell in row-major order, and the statistics of the optimal path
    are carried through the recurrence instead of recovered by traceback. A gap
    of length ``k`` costs ``gap_open + k * gap_extend``, so the first gap column
    costs ``gap_open + gap_extend``.
    """

    if gap_open < 0 or gap_extend <= 0 or block < 1:
        raise ValueError("gap costs must be nonnegative with a positive extension")
    table = blosum62() if matrix is None else matrix
    width = codes.shape[1]
    if width >= 128:
        raise ValueError("the packed start/end encoding assumes sequences under 128 residues")
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be an (n, 2) index array")
    open_total = np.float32(gap_open + gap_extend)
    extend = np.float32(gap_extend)
    columns = np.arange(1, width + 1, dtype=np.int64)

    out = [np.zeros(len(pairs), dtype=dtype) for dtype in
           (np.float32, np.int64, np.int64, np.float64, np.float64)]
    for start in range(0, len(pairs), block):
        stop = min(start + block, len(pairs))
        left, right = pairs[start:stop, 0], pairs[start:stop, 1]
        a_codes, b_codes = codes[left], codes[right]
        a_len, b_len = lengths[left].astype(np.int64), lengths[right].astype(np.int64)
        size = stop - start
        row = np.arange(size)
        in_b = columns[None, :] <= b_len[:, None]

        shape = (size, width + 1)
        m_score = np.full(shape, _NEG, dtype=np.float32)
        x_score = np.full(shape, _NEG, dtype=np.float32)
        y_score = np.full(shape, _NEG, dtype=np.float32)
        m_ident = np.zeros(shape, dtype=np.int64)
        x_ident = np.zeros(shape, dtype=np.int64)
        y_ident = np.zeros(shape, dtype=np.int64)
        m_cols = np.zeros(shape, dtype=np.int64)
        x_cols = np.zeros(shape, dtype=np.int64)
        y_cols = np.zeros(shape, dtype=np.int64)
        m_from = np.zeros(shape, dtype=np.int64)
        x_from = np.zeros(shape, dtype=np.int64)
        y_from = np.zeros(shape, dtype=np.int64)

        best = np.full(size, _NEG, dtype=np.float32)
        best_ident = np.zeros(size, dtype=np.int64)
        best_cols = np.zeros(size, dtype=np.int64)
        best_from = np.zeros(size, dtype=np.int64)
        best_end = np.zeros(size, dtype=np.int64)

        for i in range(1, width + 1):
            live = (i <= a_len)[:, None] & in_b
            same = (a_codes[:, i - 1][:, None] == b_codes) & (b_codes != PAD_CODE)
            substitution = table[a_codes[:, i - 1][:, None], b_codes]

            # Match state: the diagonal predecessor, or a fresh local start.
            base = m_score[:, :width].copy()
            ident = m_ident[:, :width].copy()
            cols = m_cols[:, :width].copy()
            origin = m_from[:, :width].copy()
            for score_source, ident_source, cols_source, from_source in (
                (x_score, x_ident, x_cols, x_from),
                (y_score, y_ident, y_cols, y_from),
            ):
                take = score_source[:, :width] > base
                base = np.where(take, score_source[:, :width], base)
                ident = np.where(take, ident_source[:, :width], ident)
                cols = np.where(take, cols_source[:, :width], cols)
                origin = np.where(take, from_source[:, :width], origin)
            fresh = base <= 0.0
            new_m = np.where(fresh, np.float32(0.0), base) + substitution
            new_m_ident = np.where(fresh, 0, ident) + same
            new_m_cols = np.where(fresh, 0, cols) + 1
            new_m_from = np.where(fresh, i * 128 + columns[None, :], origin)
            new_m = np.where(live, new_m, _NEG)

            # Gap in the second sequence: the vertical predecessor.
            opened = m_score[:, 1:] - open_total
            extended = x_score[:, 1:] - extend
            take = extended > opened
            new_x = np.where(take, extended, opened)
            new_x_ident = np.where(take, x_ident[:, 1:], m_ident[:, 1:])
            new_x_cols = np.where(take, x_cols[:, 1:], m_cols[:, 1:]) + 1
            new_x_from = np.where(take, x_from[:, 1:], m_from[:, 1:])
            new_x = np.where(live, new_x, _NEG)

            m_score[:, 1:], m_ident[:, 1:] = new_m, new_m_ident
            m_cols[:, 1:], m_from[:, 1:] = new_m_cols, new_m_from
            x_score[:, 1:], x_ident[:, 1:] = new_x, new_x_ident
            x_cols[:, 1:], x_from[:, 1:] = new_x_cols, new_x_from

            # Gap in the first sequence: the horizontal predecessor, which is in
            # the row just written, so this one column runs sequentially.
            y_score[:, 0] = _NEG
            for j in range(1, width + 1):
                opened_j = m_score[:, j - 1] - open_total
                extended_j = y_score[:, j - 1] - extend
                take_j = extended_j > opened_j
                value = np.where(take_j, extended_j, opened_j)
                y_score[:, j] = np.where(live[:, j - 1], value, _NEG)
                y_ident[:, j] = np.where(take_j, y_ident[:, j - 1], m_ident[:, j - 1])
                y_cols[:, j] = np.where(take_j, y_cols[:, j - 1], m_cols[:, j - 1]) + 1
                y_from[:, j] = np.where(take_j, y_from[:, j - 1], m_from[:, j - 1])

            argmax = m_score[:, 1:].argmax(axis=1)
            peak = m_score[row, argmax + 1]
            take = peak > best
            best = np.where(take, peak, best)
            best_ident = np.where(take, m_ident[row, argmax + 1], best_ident)
            best_cols = np.where(take, m_cols[row, argmax + 1], best_cols)
            best_from = np.where(take, m_from[row, argmax + 1], best_from)
            best_end = np.where(take, i * 128 + argmax + 1, best_end)

        empty = best <= 0.0
        span_a = best_end // 128 - best_from // 128 + 1
        span_b = best_end % 128 - best_from % 128 + 1
        out[0][start:stop] = np.where(empty, 0.0, best)
        out[1][start:stop] = np.where(empty, 0, best_cols)
        out[2][start:stop] = np.where(empty, 0, best_ident)
        out[3][start:stop] = np.where(empty, 0.0, 100.0 * span_a / a_len)
        out[4][start:stop] = np.where(empty, 0.0, 100.0 * span_b / b_len)
    return AlignmentStatistics(*out)


def reference_alignment(
    a: str, b: str, *, gap_open: float, gap_extend: float
) -> tuple[float, int, int, float, float]:
    """Independent per-pair Smith-Waterman with explicit traceback.

    Deliberately written as the textbook three-matrix recurrence with pointers so
    that agreement with :func:`batch_align` is evidence about the batched carry,
    not a restatement of it.
    """

    table = blosum62()
    index = {residue: code for code, residue in enumerate(AA20)}
    ca = [index[r] for r in a]
    cb = [index[r] for r in b]
    n, m = len(ca), len(cb)
    neg = float(_NEG)
    states = {name: [[neg] * (m + 1) for _ in range(n + 1)] for name in "MXY"}
    pointer: dict[tuple[str, int, int], tuple[str, int, int] | None] = {}
    M, X, Y = states["M"], states["X"], states["Y"]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = [(M[i - 1][j - 1], ("M", i - 1, j - 1)),
                        (X[i - 1][j - 1], ("X", i - 1, j - 1)),
                        (Y[i - 1][j - 1], ("Y", i - 1, j - 1))]
            value, source = max(diagonal, key=lambda option: option[0])
            if value <= 0.0:
                value, source = 0.0, None
            M[i][j] = value + float(table[ca[i - 1], cb[j - 1]])
            pointer[("M", i, j)] = source

            opened = M[i - 1][j] - gap_open - gap_extend
            extended = X[i - 1][j] - gap_extend
            if extended > opened:
                X[i][j], pointer[("X", i, j)] = extended, ("X", i - 1, j)
            else:
                X[i][j], pointer[("X", i, j)] = opened, ("M", i - 1, j)

            opened = M[i][j - 1] - gap_open - gap_extend
            extended = Y[i][j - 1] - gap_extend
            if extended > opened:
                Y[i][j], pointer[("Y", i, j)] = extended, ("Y", i, j - 1)
            else:
                Y[i][j], pointer[("Y", i, j)] = opened, ("M", i, j - 1)

    best, end = neg, None
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if M[i][j] > best:
                best, end = M[i][j], ("M", i, j)
    if best <= 0.0 or end is None:
        return 0.0, 0, 0, 0.0, 0.0

    columns = identical = 0
    state, i, j = end
    while True:
        first_i, first_j = i, j
        columns += 1
        if state == "M":
            identical += int(ca[i - 1] == cb[j - 1])
        source = pointer[(state, i, j)]
        if source is None:
            break
        state, i, j = source
    return (best, columns, identical,
            100.0 * (end[1] - first_i + 1) / len(a),
            100.0 * (end[2] - first_j + 1) / len(b))


class Union:
    """Union-find over labels, with the merge sources retained per edge."""

    def __init__(self, labels: Iterable[str]) -> None:
        self._parent = {label: label for label in labels}
        self.edges: list[tuple[str, str, str]] = []

    def find(self, label: str) -> str:
        root = label
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[label] != root:
            self._parent[label], label = root, self._parent[label]
        return root

    def join(self, left: str, right: str, source: str) -> bool:
        if left not in self._parent or right not in self._parent:
            raise KeyError("union over an undeclared label")
        self.edges.append((left, right, source))
        a, b = self.find(left), self.find(right)
        if a == b:
            return False
        self._parent[max(a, b)] = min(a, b)
        return True

    def components(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for label in sorted(self._parent):
            out.setdefault(self.find(label), []).append(label)
        return out


def parent_background(name: str) -> str:
    """The documented parent of an engineered MegaScale background.

    ``1BK2.pdb_L5S`` is a point variant of the ``1BK2.pdb`` background and must
    not be held out separately from it. Names without the marker are their own
    parent, which keeps a design name such as ``EA|run2_0290_0001.pdb`` whole.
    """

    marker = ".pdb_"
    return name.split(marker)[0] + ".pdb" if marker in name else name
