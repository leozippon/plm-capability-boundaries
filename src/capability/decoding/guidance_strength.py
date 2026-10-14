"""How far a logit bias moved the sampling distribution: the effective guidance strength.

The nominal dose (``lambda``) and the raw entropy of the model's step distribution
are both poor measures of an intervention's strength. The entropy of ProtGPT2 is
taken over a ~50k-subword vocabulary and is comparable neither to a 20-residue
distribution nor to ProGen2's one-residue tokens. The quantity measured here is
the movement of the distribution actually sampled from.

At step ``t`` the model yields native logits ``z_t``; the arm adds a bias ``b_t``
and inadmissible tokens are removed. Over the admissible set ``A_t``

    p_native = softmax(z_t | A_t)        p_guided = softmax(z_t + b_t | A_t)

    KL_t = KL(p_guided || p_native)      TV_t = 0.5 * sum_{A_t} |p_guided - p_native|

Per residue, not per step
=========================

Models take different numbers of steps for the same sequence: ProGen2 emits one
residue per token, ProtGPT2 several per BPE subword. A per-step statistic is
therefore not comparable across models. The comparable statistic is
``kl_per_residue = sum_t KL_t / L`` for a sequence of ``L`` residues; the per-step
statistics are reported beside it for within-model reading only.

This module is pure: it knows no model, tokenizer, file or device. Two drivers
feed it, the sampling loop (exact, as it decodes) and the teacher-forced replay of
a finished sequence (:mod:`src.capability.decoding.guided_decoding`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

RECORD_KEYS = (
    "steps", "residues", "kl_total", "kl_per_residue", "kl_mean_step", "kl_median_step",
    "kl_p90_step", "tv_mean_step", "tv_max_step",
)


def _admissible_log_softmax(scores: np.ndarray, admissible: np.ndarray) -> np.ndarray:
    """Log-softmax over the admissible entries only, shifted by their maximum."""

    kept = scores[admissible]
    if not np.isfinite(kept).all():
        raise ValueError("an admissible logit is not finite")
    shifted = kept - kept.max()
    return shifted - np.log(np.exp(shifted).sum())


def step_divergence(
    native_logits: np.ndarray, bias: np.ndarray, admissible: np.ndarray
) -> tuple[float, float]:
    """``(KL(p_guided || p_native), TV)`` of one step over the admissible set.

    All arithmetic is float64 and in log space; masked entries are never read, so
    they contribute exactly zero. A zero bias gives exactly ``(0.0, 0.0)``, and a
    constant added to the bias or to the logits over the admissible set changes
    nothing, since a softmax cannot see it.
    """

    native_logits = np.asarray(native_logits, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    admissible = np.asarray(admissible, dtype=bool)
    if native_logits.ndim != 1 or bias.shape != native_logits.shape or admissible.shape != native_logits.shape:
        raise ValueError("logits, bias and admissibility mask must be vectors of one length")
    if not admissible.any():
        raise ValueError("a step with no admissible token has no distribution")
    log_native = _admissible_log_softmax(native_logits, admissible)
    log_guided = _admissible_log_softmax(native_logits + bias, admissible)
    guided = np.exp(log_guided)
    kl = float((guided * (log_guided - log_native)).sum())
    tv = float(0.5 * np.abs(guided - np.exp(log_native)).sum())
    return max(kl, 0.0), tv


@dataclass
class StrengthMeter:
    """Aggregates the steps of one sequence into its strength record."""

    kl: list[float] = field(default_factory=list)
    tv: list[float] = field(default_factory=list)
    residues: int = 0

    def add_step(
        self, native_logits: np.ndarray, bias: np.ndarray, admissible: np.ndarray, residues: int
    ) -> None:
        """Observe one step that emitted ``residues`` residues.

        A multi-residue token emits several; a purely structural token (a line
        break) emits none but is still a step whose distribution was moved.
        """

        if residues < 0:
            raise ValueError("a step cannot emit a negative number of residues")
        kl, tv = step_divergence(native_logits, bias, admissible)
        self.kl.append(kl)
        self.tv.append(tv)
        self.residues += int(residues)

    def record(self) -> dict[str, Any]:
        if not self.kl:
            raise ValueError("no step was observed")
        if self.residues < 1:
            raise ValueError("no residue was emitted; the per-residue statistic is undefined")
        kl, tv = np.asarray(self.kl), np.asarray(self.tv)
        return {
            "steps": int(kl.size),
            "residues": int(self.residues),
            "kl_total": float(kl.sum()),
            "kl_per_residue": float(kl.sum() / self.residues),
            "kl_mean_step": float(kl.mean()),
            "kl_median_step": float(np.median(kl)),
            "kl_p90_step": float(np.percentile(kl, 90)),
            "tv_mean_step": float(tv.mean()),
            "tv_max_step": float(tv.max()),
        }


def summarise_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool the strength records of one cell: the across-sequence mean, median and p90 of each statistic."""

    if not records:
        raise ValueError("no strength record to summarise")
    out: dict[str, Any] = {"sequences": len(records)}
    for key in RECORD_KEYS:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        out[key] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "max": float(values.max()),
        }
    return out
