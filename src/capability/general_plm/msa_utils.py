"""Deterministic pairwise alignment and MSA projection utilities for R4 multi-seq PLMs.

Pure NumPy / Python (zero external bioinformatics binary dependencies) so it
works identically in ``ct`` on local Compute and inside air-gapped H200 pods.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def project_partner_to_query(
    query: str,
    partner: str,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_penalty: int = -2,
) -> str:
    """Global Needleman-Wunsch alignment projected onto query coordinates.

    Returns a string of length ``len(query)`` containing the aligned partner
    residue at each query position, or ``'-'`` where the partner has a gap.
    Subject insertions relative to the query are omitted so every row in an
    MSA shares the exact query coordinate frame.
    """
    n = len(query)
    m = len(partner)
    if n == 0:
        return ""
    if m == 0:
        return "-" * n

    # Fast banded / full DP table in int32 for short R4 chains (L <= 250).
    q_arr = np.frombuffer(query.encode("ascii"), dtype=np.uint8)
    p_arr = np.frombuffer(partner.encode("ascii"), dtype=np.uint8)

    dp = np.empty((n + 1, m + 1), dtype=np.int32)
    trace = np.zeros((n + 1, m + 1), dtype=np.uint8)  # 0=diag, 1=up(gap in p), 2=left(gap in q)

    dp[:, 0] = np.arange(n + 1, dtype=np.int32) * gap_penalty
    dp[0, :] = np.arange(m + 1, dtype=np.int32) * gap_penalty
    trace[1:, 0] = 1
    trace[0, 1:] = 2

    # Vectorize match score row-by-row for speed without C extensions.
    for i in range(1, n + 1):
        qi = q_arr[i - 1]
        sub = np.where(p_arr == qi, match_score, mismatch_score)
        diag = dp[i - 1, :-1] + sub
        up = dp[i - 1, 1:] + gap_penalty
        # First take max(diag, up), then sweep horizontal left gaps.
        best = np.where(diag >= up, diag, up)
        tr = np.where(diag >= up, 0, 1).astype(np.uint8)

        row_dp = dp[i]
        row_tr = trace[i]
        left_val = int(row_dp[0])
        for j in range(1, m + 1):
            cand_left = left_val + gap_penalty
            b = int(best[j - 1])
            if cand_left > b:
                left_val = cand_left
                row_dp[j] = cand_left
                row_tr[j] = 2
            else:
                left_val = b
                row_dp[j] = b
                row_tr[j] = tr[j - 1]

    # Traceback into query-coordinate projection.
    projected = ["-"] * n
    i, j = n, m
    while i > 0 and j > 0:
        move = int(trace[i, j])
        if move == 0:
            projected[i - 1] = partner[j - 1]
            i -= 1
            j -= 1
        elif move == 1:
            projected[i - 1] = "-"
            i -= 1
        else:
            j -= 1
    return "".join(projected)


def build_query_msa(
    target: str,
    partners: Sequence[str],
    *,
    max_depth: int = 32,
    aligned: bool = True,
) -> list[str]:
    """Construct a query-coordinate MSA list ``[target, partner_1, ...]``.

    When ``aligned=True``, each partner is globally aligned and projected onto
    ``target``'s exact length. When ``aligned=False`` (unaligned-stack control),
    each partner is truncated or right-padded with ``'-'`` to ``len(target)``.
    """
    rows = [target]
    limit = max(0, int(max_depth) - 1)
    for partner in list(partners)[:limit]:
        if aligned:
            rows.append(project_partner_to_query(target, partner))
        else:
            if len(partner) >= len(target):
                rows.append(partner[: len(target)])
            else:
                rows.append(partner + "-" * (len(target) - len(partner)))
    return rows
