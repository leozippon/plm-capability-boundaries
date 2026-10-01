"""Shared sequence alignment support for main measurements."""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from ..context.homology import Hit
from ..context.profiles import Profile, AA20, NEFF_IDENTITY_FLOOR, sequence_weights, GAP_CODE, REWEIGHT_IDENTITY_FLOOR, PROFILE_COVERAGE_FLOOR, _encode

def alignment_rows(
    wildtype: str,
    query_id: str,
    hits: Sequence[Hit],
    *,
    max_sequences: int,
    coverage_floor: float = PROFILE_COVERAGE_FLOOR,
    reweight_identity: float = REWEIGHT_IDENTITY_FLOOR,
) -> tuple[np.ndarray, np.ndarray, float]:
    """The corpus alignment of one wild type as ``(sequences, columns)`` residue codes.

    The column mapping is :func:`build_profile`'s, restated here because that
    function returns frequencies and a coupling estimate needs the rows. The
    restatement is checked rather than trusted -- see
    :func:`verify_rows_against_profile`, which every caller is expected to run.
    """

    length = len(wildtype)
    best: dict[str, Hit] = {}
    for hit in hits:
        if hit.qlen != length:
            raise ValueError(
                f"{query_id}: DIAMOND reports qlen {hit.qlen} for a "
                f"{length}-residue wild type"
            )
        if hit.qseq_gapped is None or hit.sseq_gapped is None:
            raise ValueError(
                f"{query_id}: hit against {hit.subject} carries no aligned sequences; "
                "the search must request homology.ALIGNMENT_FIELDS"
            )
        if 100.0 * (hit.qend - hit.qstart + 1) / length < coverage_floor:
            continue
        previous = best.get(hit.subject)
        if previous is None or hit.bitscore > previous.bitscore:
            best[hit.subject] = hit
    ordered = sorted(best.values(), key=lambda hit: -hit.bitscore)[:max_sequences]

    rows = np.full((len(ordered), length), GAP_CODE, dtype=np.int8)
    for index, hit in enumerate(ordered):
        if len(hit.qseq_gapped) != len(hit.sseq_gapped):
            raise ValueError(
                f"{query_id}: gapped query and subject differ in length against "
                f"{hit.subject}; the search must request qseq_gapped/sseq_gapped"
            )
        position = hit.qstart - 1
        codes = _encode(hit.sseq_gapped)
        for column, residue in enumerate(hit.qseq_gapped):
            if residue == "-":
                continue
            if position >= length:
                raise ValueError(
                    f"{query_id}: alignment against {hit.subject} runs past the "
                    "wild type's last residue"
                )
            rows[index, position] = codes[column]
            position += 1
        if position != hit.qend:
            raise ValueError(
                f"{query_id}: alignment against {hit.subject} covers query residues "
                f"{hit.qstart}-{position} where DIAMOND reports {hit.qstart}-{hit.qend}"
            )
    weights = (
        sequence_weights(rows, identity=reweight_identity)
        if len(ordered)
        else np.zeros(0, dtype=np.float64)
    )
    neff = float(
        sum(
            weight
            for weight, hit in zip(weights, ordered)
            if hit.identity_over_query >= NEFF_IDENTITY_FLOOR
        )
    )
    return rows, weights, neff


def verify_rows_against_profile(
    rows: np.ndarray, weights: np.ndarray, profile: Profile, *, tolerance: float = 1e-9
) -> None:
    """Require the rebuilt rows to reduce to the profile they were rebuilt beside.

    Appendix B rule 12 asks for one declaration of a shared decision. Two readers
    of one alignment are unavoidable here, because a coupling needs rows and the
    existing declaration returns columns; what is avoidable is their disagreeing
    silently. This raises instead.
    """

    frequencies = np.zeros_like(profile.frequencies, dtype=np.float64)
    for code in range(len(AA20)):
        frequencies[:, code] = (weights[:, None] * (rows == code)).sum(axis=0)
    column_weight = frequencies.sum(axis=1)
    positive = column_weight > 0
    frequencies[positive] /= column_weight[positive, None]
    difference = float(np.abs(frequencies - profile.frequencies).max())
    if difference > tolerance:
        raise RuntimeError(
            f"{profile.query_id}: the rebuilt alignment rows do not reproduce "
            f"build_profile's frequencies (max |difference| {difference:.3e} > "
            f"{tolerance:.0e}); the two readers of this alignment have diverged"
        )
