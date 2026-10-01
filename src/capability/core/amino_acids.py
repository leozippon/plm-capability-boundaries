"""Canonical twenty-residue alphabet and NCBI BLOSUM62 transcription.

This module is the single source for the alphabetical amino-acid string used as
``AA20``, ``ALPHABET`` and ``COMPOSITION_ALPHABET``, and for the BLOSUM62
half-bit matrix in published NCBI row order. It imports nothing from the rest of
the package and nothing that needs a GPU, so a string consumer can take the
alphabet without loading ``torch`` or :mod:`.arms`.

The integer table is the scientific object. :func:`blosum62_ncbi_rows` is the
lossless adapter onto the width-3 string transcription ``circuits`` historically
stored. Fitness scoring keeps a pair-dict public type built from this table.
"""

from __future__ import annotations

#: Alphabetical twenty-residue alphabet. Column order for profiles, k-mer
#: indexing and composition matches; not the published BLOSUM62 row order.
AA20 = "ACDEFGHIKLMNPQRSTVWY"

#: Published NCBI row and column order of the BLOSUM62 half-bit matrix.
BLOSUM62_ORDER = "ARNDCQEGHILKMFPSTWYV"

#: BLOSUM62 as distributed by NCBI, restricted to the twenty standard residues,
#: in :data:`BLOSUM62_ORDER`. Transcribed rather than loaded from a library so
#: the substitution scores cannot depend on which optional package is installed.
BLOSUM62_ROWS: tuple[tuple[int, ...], ...] = (
    (4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0),
    (-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3),
    (-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3),
    (-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3),
    (0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1),
    (-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2),
    (-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2),
    (0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3),
    (-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3),
    (-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3),
    (-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1),
    (-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2),
    (-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1),
    (-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1),
    (-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2),
    (1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2),
    (0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0),
    (-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3),
    (-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1),
    (0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4),
)


def blosum62_ncbi_rows() -> tuple[str, ...]:
    """Lossless NCBI width-3 integer transcription of :data:`BLOSUM62_ROWS`.

    ``circuits`` historically stored this presentation and parsed it with
    ``str.split``. The numeric values are identical; only the public type differs.
    """

    return tuple("".join(f"{value:3d}" for value in row) for row in BLOSUM62_ROWS)
