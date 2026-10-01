"""Invariants of the shared amino-acid alphabet and BLOSUM62 transcription.

Structure and published entries catch transposition; lossless adapters preserve
the matrix and public aliases; a subprocess import proves the constants module does
not pull in torch or arms.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.capability.core.amino_acids import (  # noqa: E402
    AA20,
    BLOSUM62_ORDER,
    BLOSUM62_ROWS,
    blosum62_ncbi_rows,
)

def test_alphabet_and_order() -> None:
    assert len(AA20) == 20
    assert len(set(AA20)) == 20
    assert AA20 == "".join(sorted(AA20))
    assert sorted(BLOSUM62_ORDER) == sorted(AA20)


def test_blosum62_symmetry_and_published_entries() -> None:
    assert len(BLOSUM62_ROWS) == 20
    assert all(len(row) == 20 for row in BLOSUM62_ROWS)
    assert all(type(value) is int for row in BLOSUM62_ROWS for value in row)
    assert BLOSUM62_ROWS == tuple(
        tuple(BLOSUM62_ROWS[column][row] for column in range(20)) for row in range(20)
    )

    def score(left: str, right: str) -> int:
        return BLOSUM62_ROWS[BLOSUM62_ORDER.index(left)][BLOSUM62_ORDER.index(right)]

    assert score("A", "A") == 4
    assert score("W", "W") == 11
    assert score("C", "C") == 9
    assert score("P", "W") == -4


def test_ncbi_adapter_is_lossless() -> None:
    rows = blosum62_ncbi_rows()
    assert type(rows) is tuple
    assert type(rows[0]) is str
    parsed = tuple(tuple(int(value) for value in row.split()) for row in rows)
    assert parsed == BLOSUM62_ROWS


def test_legacy_alphabet_aliases_keep_public_names() -> None:
    from src.capability.core.arms import AA20 as arms_aa20
    from src.capability.context.context_homologue import COMPOSITION_ALPHABET
    from src.capability.context.kmer_background import ALPHABET
    from src.capability.context.profiles import AA20 as profiles_aa20

    assert arms_aa20 is AA20
    assert profiles_aa20 is AA20
    assert ALPHABET is AA20
    assert COMPOSITION_ALPHABET is AA20
    assert type(ALPHABET) is str


def test_legacy_blosum_public_names_keep_types() -> None:
    from src.capability.models.fitness import BLOSUM62 as fitness_blosum

    assert len(fitness_blosum) == 400
    assert all(
        fitness_blosum[(left, right)] == BLOSUM62_ROWS[i][j]
        for i, left in enumerate(BLOSUM62_ORDER)
        for j, right in enumerate(BLOSUM62_ORDER)
    )


def test_constants_module_does_not_import_torch_or_arms() -> None:
    script = (
        "import sys\n"
        "from src.capability.core.amino_acids import AA20, BLOSUM62_ROWS\n"
        "assert AA20 == 'ACDEFGHIKLMNPQRSTVWY'\n"
        "assert len(BLOSUM62_ROWS) == 20\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'src.capability.core.arms' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
