"""The zero-sum gauge of a Potts model, in one place.

Potts parameters are identifiable only up to a gauge. The convention adopted for
E20 is the zero-sum one: ``J`` centred in both residue indices, with the
off-diagonal row-mean sum moved into ``h``. Used by the gauge diagnostic and by
the guidance-strength measurement, so both reproduce one convention.
"""

from __future__ import annotations

import numpy as np


def to_zero_sum(h: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the zero-sum-gauge representative of ``(h, J)``.

    ``J`` is centred in both residue indices; ``h`` absorbs the full
    off-diagonal row-mean sum, which is what keeps every full-sequence
    energy fixed up to an additive constant. The per-step coupling-only bias
    is not preserved: it shifts, up to a constant in the candidate residue
    ``a``, by ``-sum_{s<t} mean_b J[t, s, a, b]``, a first-order term in ``a``
    (see the gauge note in ``family_potts`` for the full statement).
    """
    L = h.shape[0]
    ra = J.mean(axis=3, keepdims=True)
    rb = J.mean(axis=2, keepdims=True)
    rab = J.mean(axis=(2, 3), keepdims=True)
    J_zero = J - ra - rb + rab
    off = ~np.eye(L, dtype=bool)
    induced = (J.mean(axis=3) * off[..., None]).sum(axis=1)
    return h + induced, J_zero
