"""Stage-1 substrate measurement: is an evaluation cohort measurable at all?

Every normalised interpretability score - loss recovered, KL recovered, a
mutual-information gate - is a ratio whose denominator is the information the
model actually commits on the cohort relative to a context-free baseline. If
that denominator is small the ratio is noise; if it is negative the arm is
off-distribution and the ratio is meaningless.

The motivating history is worth stating precisely, because the first version of
this docstring quoted figures that were themselves measurement artefacts and
have since been retracted. What actually happened: a production qualification on
a 64-246 residue EC-labelled cohort appeared to show ProGen2-medium starved at
0.099 nats/token and ProtGPT2 off-distribution at -1.73 nats/token. Neither
survived. The ProGen2 figure came from comparing against a *different* cohort
(plain rather than EC-labelled Swiss-Prot); its true value there is 1.24-1.52.
The ProtGPT2 figure came from rendering it as one unwrapped line when it was
pretrained on FASTA-formatted UniRef50, worth 1.42 nats/token; measured across
four cohorts under the correct rendering it is +0.21 to +3.89 and has never
reproduced as negative.

The lesson is not that the power check is unnecessary - it is that the power
check must be applied to a cohort the arm is actually native to, and with an
unbiased baseline. A plug-in unigram estimate computed on the scored tokens
understates entropy by 0.75 nats for a 50k-vocabulary text arm and 1.65 nats for
a 50k-vocabulary protein arm while leaving residue-level arms untouched, which
inflates every share computed from it by up to 74 per cent.

This module measures that power figure *before* any scientific gate is applied
so that a starved arm is reported as unmeasurable on the cohort rather than as
a failed scientific hypothesis.

Identifying an arm and admitting its reading as a denominator are two criteria,
and this module declares both. They were one undeclared 0.30-nat constant until
EXP-R2-218 calibrated each against a known-zero null family and a known-signal
mixing family, and they turned out to have different answers by more than an
order of magnitude. **Neither is a constant now.** Identification is
:func:`context_identification` -- the arm's displacement-corrected 95% interval
must lie strictly above zero -- and a bounded ratio needs the reading to sit
:data:`FIELLER_DENOMINATOR_MULTIPLE` of its own standard errors above zero, which
is 0.146 to 0.966 nats depending on the arm. Both read the arm's own precision,
so they are nested with the ratio criterion strictly the stronger, and the pair
of readings that a magnitude rule and a precision rule ordered oppositely
(EXP-R2-220) cannot recur.

Two constants survive as reporting columns. :data:`MIN_CONTEXT_INFORMATION_NATS`
decides nothing anywhere. :data:`SCREENING_CONTEXT_INFORMATION_NATS` decides
nothing where an interval exists and remains the *pre-interval screen* upstream
of the bootstrap, where no interval does -- :func:`arm_power` among them, since
it is the function that produces the statistics the interval is later computed
from. A verdict taken there is a screen and says so.

Per-token quantities are tokenizer-dependent and are therefore not comparable
across arms: ProtGPT2 uses multi-residue BPE while ZymCTRL and ProGen2-medium
are residue-level. The held-out residue Markov ladder is the only
tokenizer-independent axis on which the protein arms can be compared to each
other, so it is reported alongside every per-token figure.
"""

from __future__ import annotations

import inspect
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from scipy import stats

from .arms import (
    AA20,
    Arm,
    Cohort,
    conditioning_boundary_ids,
    output_logit_width,
    rendering_marker_ids,
    require_scoring_target_ids,
    scoring_target_alphabet,
    symbols_per_token,
    tokenize_batch,
)
from ..context.information_bootstrap import FIELLER_MAXIMUM_G
from .io import sha256_file
from .scoring import sequence_target_mask, target_rule
from .statistics import mean_interval

LN2 = math.log(2.0)

#: Default visible-context lengths for the truncation curve. Powers of two span
#: the range over which a protein decoder's local statistics saturate.
DEFAULT_CONTEXT_LENGTHS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)

#: Context information an arm must read before a stage that has no interval will
#: score it at all.
#:
#: **This stopped being the identification criterion at EXP-R2-221.** It is a
#: *pre-interval screen*: a magnitude comparison that a stage upstream of any
#: bootstrap can make, and the only thing such a stage can make. Identification
#: itself is :func:`context_identification`, which reads the arm's own
#: displacement-corrected interval and needs no constant at all. Where an
#: interval exists this constant decides nothing and is reported for
#: comparability; where none exists the verdict taken from it is a screen and the
#: artefact says so rather than calling it identification.
#:
#: **What the demotion cost the constant, measured.** As an identification rule a
#: fixed magnitude orders two readings by size while their own standard errors
#: order them the other way, and EXP-R2-220 found the first real pair on which
#: that happens: ``galactica-1.3b``'s protein mode reads +0.047678 at
#: SE 0.004571 -- 10.43 standard errors from zero and admissible as a ratio
#: denominator -- and this floor refuses it by 0.0023 nats, while
#: ``Llama-2-7b-hf``'s protein mode passes at +0.084287 and is refused by the
#: Fieller condition at 8.28. One of those refusals is a false negative and the
#: other is not, and no value of a constant separates them (EXP-R2-221, §5.10).
#:
#: The derivation below is what it was and is retained: it is why this remains an
#: adequate *screen* even though it is not the identification criterion.
#:
#: Derived at EXP-R2-218 against a known-zero null family (the independent-block
#: unigram control) and a known-signal lambda family, 1920 readings over 15 arms
#: and 24 cohort draws. On the point rule ``I_hat >= tau`` the false-positive
#: rate first reaches 0.05 at ``tau = 0.010`` and is exactly zero from
#: ``tau = 0.015`` upward; the largest null point departure anywhere was 0.0132
#: nats. 0.05 is adopted rather than 0.010 or 0.015 for three reasons: it keeps
#: a 3.8x margin over that largest observed null departure, it reaches 80% power
#: from a true ``I`` of about 0.05 on every arm of the panel, and it sits clear
#: of the smoothing and Jensen noise band -- the bootstrap displaces the null
#: interval upward by a mean of +0.0095 and a maximum of +0.0347 nats -- that
#: makes any threshold inside 0.010-0.020 indefensible in practice.
#:
#: **Cross-fitted at EXP-R2-219 and it holds; the 0.010 does not.** Re-deriving
#: the threshold on each calibration fold and scoring it on units the derivation
#: never saw -- 24 cohort draws, and separately seven tokenisation classes, the
#: 120 null cells being only 56 distinct measurements -- this floor admits 0 of
#: 120 held-out null readings under every scheme and no fold asks for more than
#: 0.015 nats. The calibrated 0.010 reaches a held-out rate of 0.0667 against its
#: own 0.05 target, because it is set by the single noisiest cohort draw. A rate
#: of zero is not proof: over 56 distinct measurements the one-sided 95% limit on
#: this floor's false-positive rate is 0.052, so it is validated as not too lax at
#: the resolution the panel supports and not as exact.
#:
#: **It changes no verdict on the current panel.** Every arm reads at or above
#: +0.90 nats (progen2-small, the lowest positive) or at or below -3.98 nats
#: (dialogpt-small), so the 15 arms fall on the same side of 0.05 as they fell of
#: the retired 0.30, and no held-out fold's threshold moves one either.
#:
#: The *uncorrected* interval variant ``lower bound >= tau`` was and remains the
#: weaker statistic near zero: 56 of 112 null lower bounds sit above zero before
#: any signal exists. What changed at EXP-R2-221 is that the displacement those
#: bounds were reading is now measured and removed, and the corrected variant at
#: ``tau = 0`` admits none of them -- which is why identification moved to
#: :func:`context_identification` and this constant did not follow it there.
SCREENING_CONTEXT_INFORMATION_NATS = 0.05

#: Carried into every artefact whose verdict came from the screen rather than
#: from the criterion, so the two cannot be read as the same thing.
SCREENING_FLOOR_NOTE = (
    f"{SCREENING_CONTEXT_INFORMATION_NATS} nats/token is a PRE-INTERVAL SCREEN "
    "and not the identification criterion. It is applied where no bootstrap "
    "interval exists yet -- upstream of the stage that computes one -- and it "
    "answers whether an arm reads enough on this cohort to be worth scoring. "
    "Identification is budget.context_identification, which asks whether the "
    "arm's own displacement-corrected 95% interval excludes zero and has no "
    "constant in it; a verdict taken from this screen is not an identification "
    "verdict and does not become one by being reported beside it"
)

#: The identification criterion's name, carried in every artefact that takes it.
IDENTIFICATION_CRITERION = "displacement_corrected_interval_excludes_zero"

IDENTIFICATION_NOTE = (
    "identification is a detection question and is decided on the arm's own "
    "precision: the displacement-corrected 95% interval for I must lie strictly "
    "above zero, which is approximately I > 1.96*SE(I). The correction is the "
    "one thing that makes a lower-bound rule usable here -- uncorrected, the "
    "Jensen displacement of L34/L42 lifts 64 of 120 null intervals above zero "
    "before any signal exists, and corrected it lifts none (EXP-R2-221, cross-"
    "fitted over 24 cohort draws and seven tokenisation classes, 0 of 120 "
    "held-out null readings and 0 of 56 distinct measurements under every "
    "scheme). Zero is also the one threshold that is the same criterion in all "
    "three units, which no positive constant in nats per token is. It says the "
    "arm read above no-context and NOT that its reading may be divided by; that "
    "is budget.ratio_denominator_admissibility, which is strictly stronger, so "
    "the two are nested and cannot cross"
)

#: Standard errors the denominator of a ratio must sit above before the ratio has
#: a bounded confidence set: ``z_{0.975} / sqrt(FIELLER_MAXIMUM_G)``.
#:
#: :data:`src.capability.context.information_bootstrap.FIELLER_MAXIMUM_G` gates every ratio
#: with ``I`` in the denominator on Fieller's ``g = (z * SE(I) / I)^2 < 0.05``.
#: That condition inverts exactly: ``g < g_max`` iff
#: ``I > (z / sqrt(g_max)) * SE(I)``, which at the 95% level is
#: ``I > 8.765225 * SE(I)``. The multiple is imported from that declaration
#: rather than restated, so the two forms of one condition cannot drift.
#:
#: **This is a precision reference, not a magnitude.** Its value in nats is a
#: different number on every arm and every block, because ``SE(I)`` depends on
#: the cohort size, the near-duplicate grouping, the reference size and the
#: vocabulary. Measured over the panel at EXP-R2-218 it runs from 0.1456 nats
#: (bygpt5-small) to 0.9664 (dialogpt-small), a spread of 6.6x, with 12 of the 15
#: arms above 0.30 -- which is why no constant can serve this purpose and why the
#: retired floor was simultaneously far too strict for identification and up to
#: 3.2x too lax here.
FIELLER_DENOMINATOR_MULTIPLE = float(stats.norm.ppf(0.975)) / math.sqrt(FIELLER_MAXIMUM_G)

#: **Legacy reporting column. This constant decides nothing.**
#:
#: It was the single "measurability floor" this programme applied from its first
#: stage until EXP-R2-218, in two incompatible jobs at once: screening an arm in
#: as measurable, and admitting its context information as the denominator of a
#: share. It was never derived. Its only defence was an observed distribution --
#: that no record in ``results/`` landed between 0 and 0.30 -- which is agreement
#: in a region where nothing was at stake.
#:
#: Calibration measured both jobs and found it wrong in opposite directions. For
#: identification it is 20-30x stricter than the false-positive rate requires
#: (calibrated tau 0.010 at FPR 0.05, 0.015 at FPR 0). For ratio boundedness it
#: is up to 3.2x too lax: 12 of 15 arms need more than 0.30 standard-error-widths
#: of denominator, and a single constant sufficient for every arm would have to
#: be at least 0.97. In the other two units it is not one criterion at all --
#: rho 0.092 on byte arms, 0.040 on GPT-2 arms, 0.104 at residue level, 0.034 on
#: protgpt2, a 3.1x spread.
#:
#: It is kept, and reported beside every verdict, so that results recorded under
#: it stay comparable to results recorded under the criteria that replaced it.
#: It is not deleted, because a reader of an older artefact needs to know which
#: number produced it.
MIN_CONTEXT_INFORMATION_NATS = 0.30

#: Carried into every artefact that reports the legacy column, so the reason it
#: is present and the reason it is inert cannot drift apart.
LEGACY_FLOOR_NOTE = (
    f"{MIN_CONTEXT_INFORMATION_NATS} nats/token was the single undeclared "
    "measurability floor applied before EXP-R2-218 and is reported here for "
    "comparability with results recorded under it. It decides nothing: "
    "identification is decided by SCREENING_CONTEXT_INFORMATION_NATS and "
    "denominator admissibility by ratio_denominator_admissibility, which are "
    "different criteria with different answers"
)

MEASURABLE = "measurable"
UNMEASURABLE = "unmeasurable_on_this_cohort"

#: The same two verdicts, qualified by the estimator that produced them.
#:
#: :func:`arm_power` decides measurability against a *plug-in* unigram baseline
#: unless the caller supplies a held-out one, and ``pathways.UNIGRAM_ESTIMATORS``
#: declares the plug-in to be "an explicit opt-in diagnostic" whose bias "grows
#: with vocabulary against sample size" -- roughly +0.003 nats at 32 symbols
#: against +1.65 at 50257 pieces on a cohort of this size. That differential is
#: more than thirty times :data:`SCREENING_CONTEXT_INFORMATION_NATS`, so which
#: arms pass a plug-in verdict is set mostly by vocabulary size.
#:
#: A campaign run asks for the held-out estimator, so no published verdict came
#: from the plug-in. The defect is that nothing stopped a *caller* inheriting
#: one: the estimator was recorded in a different field from the verdict, and a
#: reader who took ``measurability`` alone could not tell which they had. The
#: verdict now carries its estimator in its own value when that estimator is the
#: diagnostic one, which is the only form a reader cannot separate them in.
MEASURABLE_PLUG_IN = "measurable_against_plug_in_baseline"
UNMEASURABLE_PLUG_IN = "unmeasurable_on_this_cohort_against_plug_in_baseline"


@dataclass(frozen=True)
class ScoredTokens:
    """Next-token targets that belong to the cohort's own content.

    Padding, the first token of every sequence, and - for EC-conditioned
    ZymCTRL inputs - the conditioning prefix and the terminator are excluded,
    so the unigram baseline and the model cross-entropy are estimated on
    exactly the same token multiset.
    """

    target_ids: np.ndarray
    nll_nats: np.ndarray
    sequence_index: np.ndarray

    def __post_init__(self) -> None:
        if self.target_ids.ndim != 1:
            raise ValueError("scored-token arrays must be one-dimensional")
        if self.nll_nats.shape != self.target_ids.shape:
            raise ValueError("scored-token NLL does not align with targets")
        if self.sequence_index.shape != self.target_ids.shape:
            raise ValueError("scored-token sequence index does not align with targets")
        if self.target_ids.size == 0:
            raise ValueError("no scored tokens were produced")

    def __len__(self) -> int:
        return int(self.target_ids.size)


@torch.no_grad()
def _scored_tokens_rita(
    arm: Arm, input_strings: Sequence[str], *, max_len: int
) -> ScoredTokens:
    """NLL on the document rendering's scored targets, one record at a time.

    The targets are the arm's declared ones -- :func:`target_rule` and
    :func:`rendering_marker_ids` resolved once below, as every other arm resolves
    them -- so the document boundary this rendering prefixes and the terminal
    ``<EOS>`` its tokenizer appends are context rather than scored content.

    Does not assign ``tokenizer.pad_token`` and does not pass an attention mask.
    Variable-length records are not stacked: padding would require an attention
    mask the author forward does not take as a scoring contract.
    """

    from ..models.rita_fitness import native_encode_for_budget

    rule = target_rule(arm.spec.input_format)
    markers = rendering_marker_ids(arm)
    target_blocks: list[np.ndarray] = []
    nll_blocks: list[np.ndarray] = []
    index_blocks: list[np.ndarray] = []
    for index, text in enumerate(input_strings):
        ids = native_encode_for_budget(arm.tokenizer, text)
        if len(ids) > max_len:
            raise ValueError(
                f"{arm.name}: native encoding is {len(ids)} tokens and max_len is {max_len}"
            )
        if len(ids) < 2:
            raise ValueError(f"{arm.name}: native encoding produced no scored target")
        tensor = torch.tensor([ids], dtype=torch.long, device=arm.device)
        logits = arm.model(input_ids=tensor).logits
        logprobs = F.log_softmax(logits[:, :-1].float(), dim=-1)
        target = tensor[:, 1:]
        nll = -logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)[0]
        keep = sequence_target_mask(
            tensor,
            torch.ones_like(tensor),
            rule=rule,
            marker_token_ids=markers,
        )[0]
        if not bool(torch.isfinite(nll[keep]).all()):
            raise FloatingPointError(f"{arm.name}: non-finite clean NLL")
        target_blocks.append(target[0][keep].detach().cpu().numpy().astype(np.int64))
        nll_blocks.append(nll[keep].detach().cpu().numpy().astype(np.float64))
        index_blocks.append(np.full(int(keep.sum()), index, dtype=np.int64))
    return ScoredTokens(
        target_ids=np.concatenate(target_blocks),
        nll_nats=np.concatenate(nll_blocks),
        sequence_index=np.concatenate(index_blocks),
    )


@torch.no_grad()
def scored_tokens(
    arm: Arm,
    input_strings: Sequence[str],
    *,
    max_len: int,
    batch_size: int,
) -> ScoredTokens:
    """Per-token clean negative log-likelihood over the cohort's scored targets."""

    if not input_strings:
        raise ValueError(f"{arm.name}: empty cohort")
    if max_len < 2:
        raise ValueError("max_len must admit at least one next-token target")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if arm.spec.architecture == "progen3":
        from ..models.progen3 import n_to_c_scored_arrays, require_progen3_handle

        pg = require_progen3_handle(arm)
        target_ids, nll_nats, sequence_index = n_to_c_scored_arrays(
            pg, list(input_strings), max_len=max_len, batch_size=batch_size
        )
        return ScoredTokens(
            target_ids=target_ids, nll_nats=nll_nats, sequence_index=sequence_index
        )
    if arm.spec.architecture == "rita":
        return _scored_tokens_rita(arm, input_strings, max_len=max_len)

    rule = target_rule(arm.spec.input_format)
    start_id, end_id = conditioning_boundary_ids(arm)
    conditioned = start_id is not None
    # A conditioned arm's span is its <start>/<end> pair; every other rendering's
    # is whatever that rendering declared as its marker prefix. Resolved here from
    # the same two declarations the rendering is built from, so the scored span and
    # the rendered string cannot come to disagree about which positions are
    # content.
    markers = () if conditioned else rendering_marker_ids(arm)

    targets: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for offset in range(0, len(input_strings), batch_size):
        chunk = list(input_strings[offset : offset + batch_size])
        ids, mask = tokenize_batch(arm, chunk, max_len)
        if conditioned:
            complete = (ids == end_id).sum(dim=1).eq(1) & (ids == start_id).sum(dim=1).eq(1)
            if not bool(complete.all()):
                raise ValueError(
                    f"{arm.name}: max_len={max_len} truncates the EC-conditioned prompt "
                    "before its <end> boundary; the scored window would be undefined"
                )
        ids = ids.to(arm.device)
        mask = mask.to(arm.device)
        logits = arm.model(input_ids=ids, attention_mask=mask).logits
        logprobs = F.log_softmax(logits[:, :-1].float(), dim=-1)
        target = ids[:, 1:]
        nll = -logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        keep = sequence_target_mask(
            ids,
            mask,
            rule=rule,
            start_token_id=start_id,
            end_token_id=end_id,
            marker_token_ids=markers,
        )
        selected = nll[keep]
        if not bool(torch.isfinite(selected).all()):
            raise FloatingPointError(f"{arm.name}: non-finite clean NLL")
        row = torch.arange(ids.shape[0], device=arm.device).unsqueeze(1).expand_as(target)
        targets.append(target[keep].cpu().numpy().astype(np.int64))
        losses.append(selected.cpu().numpy().astype(np.float64))
        indices.append((row[keep] + offset).cpu().numpy().astype(np.int64))
    return ScoredTokens(
        target_ids=np.concatenate(targets),
        nll_nats=np.concatenate(losses),
        sequence_index=np.concatenate(indices),
    )


def decoded_symbols(arm: Arm, target_ids: Sequence[int] | np.ndarray) -> int:
    """Alphabet symbols the arm's tokenizer decodes ``target_ids`` into.

    Declared once because two quantities count them: the cohort-level expansion
    below, which converts a per-token cross-entropy into bits per symbol, and
    the per-record counts :func:`record_statistics` persists beside it. A
    per-record count that applied a different rule would not sum to the figure
    it sits next to.
    """

    decoded = arm.tokenizer.decode([int(value) for value in target_ids])
    return (
        sum(1 for character in decoded if character in AA20)
        if arm.modality == "protein"
        else len(decoded)
    )


def scored_symbols_per_token(arm: Arm, scored: ScoredTokens) -> float:
    """Tokenizer expansion over exactly the tokens the cross-entropy was taken on.

    ``arms.symbols_per_token`` measures the expansion of the *whole rendered
    string*: every token the tokenizer produced, against every alphabet symbol
    it decodes to. That is the right quantity for describing a rendering and the
    wrong one for converting a per-token cross-entropy into bits per symbol,
    because ``clean_ce`` and the unigram baseline are taken over
    :class:`ScoredTokens` -- which for an EC-conditioned arm excludes the EC
    tag, ``<sep>``, ``<start>`` and ``<end>``.

    Those prompt tokens raise the token count and contribute no AA20 symbols, so
    the whole-string expansion is *understated* for ZymCTRL alone, and a
    per-symbol figure divides by it. Every ``*_bits_per_symbol`` field was
    therefore inflated for exactly one arm, on the axis this module declares to
    be the cross-arm comparable one. ProtGPT2 is unaffected: its newlines are
    scored like any other token, so its scored window and its rendered string
    are the same multiset apart from the first token.

    Computed here rather than in ``arms`` because ``arms.symbols_per_token`` has
    a legitimate second caller and a legitimate second meaning; this is a third
    quantity, not a correction to that one, and both are reported.
    """

    symbols = decoded_symbols(arm, scored.target_ids)
    if symbols < 1:
        raise RuntimeError(
            f"{arm.name}: the scored targets decode to no alphabet symbols, so a "
            "per-symbol conversion is undefined"
        )
    return symbols / len(scored)


#: Identity of the per-record sufficient-statistics sidecar written beside a
#: power report. A reader that finds a different version must rebuild rather
#: than reinterpret: the fields below are the whole contract.
POWER_RECORDS_SCHEMA_VERSION = "r2_transfer_cohort_power_records_v1"


@dataclass(frozen=True)
class SparseCounts:
    """Per-record token counts over one vocabulary, in compressed-row form.

    ``offsets[i]:offsets[i + 1]`` selects record ``i``'s entries of
    ``token_ids`` and ``counts``; a record that contributed no scored target is
    an empty span rather than an absent row, so the rows stay aligned with the
    records that produced them. Dense storage would be ``records x vocabulary``
    int64 -- 80 MB for 200 records of a 50257-piece arm -- for rows that hold a
    few hundred non-zeros each.
    """

    offsets: np.ndarray
    token_ids: np.ndarray
    counts: np.ndarray

    def __post_init__(self) -> None:
        for name in ("offsets", "token_ids", "counts"):
            array = getattr(self, name)
            if array.ndim != 1 or array.dtype != np.int64:
                raise ValueError(f"{name} must be a one-dimensional int64 array")
        if self.offsets.size < 1 or int(self.offsets[0]) != 0:
            raise ValueError("offsets must start at zero")
        if np.any(np.diff(self.offsets) < 0):
            raise ValueError("offsets must be non-decreasing")
        if self.token_ids.shape != self.counts.shape:
            raise ValueError("token ids and counts do not align")
        if int(self.offsets[-1]) != int(self.token_ids.size):
            raise ValueError("offsets do not close over the stored entries")
        if self.counts.size and int(self.counts.min()) < 1:
            raise ValueError("a stored count must be positive")

    @classmethod
    def from_records(cls, records: Sequence[np.ndarray]) -> SparseCounts:
        """Compress one token-id array per record."""

        lengths: list[int] = []
        ids: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        for record in records:
            array = np.asarray(record, dtype=np.int64).reshape(-1)
            unique, tally = np.unique(array, return_counts=True)
            lengths.append(int(unique.size))
            ids.append(unique.astype(np.int64))
            counts.append(tally.astype(np.int64))
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:], dtype=np.int64)
        return cls(
            offsets=offsets,
            token_ids=(
                np.concatenate(ids).astype(np.int64) if ids else np.zeros(0, dtype=np.int64)
            ),
            counts=(
                np.concatenate(counts).astype(np.int64)
                if counts
                else np.zeros(0, dtype=np.int64)
            ),
        )

    @property
    def n_records(self) -> int:
        return int(self.offsets.size - 1)

    def record_totals(self) -> np.ndarray:
        """Tokens counted in each record.

        ``np.add.reduceat`` is not used: on an empty span it returns the element
        at the offset rather than zero, which would silently attribute another
        record's tokens to a record that contributed none.
        """

        cumulative = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(self.counts, dtype=np.int64)]
        )
        return cumulative[self.offsets[1:]] - cumulative[self.offsets[:-1]]

    def vocabulary_totals(self, vocab_size: int) -> np.ndarray:
        """The dense count vector these records sum to."""

        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.token_ids.size and int(self.token_ids.max()) >= vocab_size:
            raise ValueError("a stored token id falls outside the declared vocabulary")
        return np.bincount(self.token_ids, weights=self.counts, minlength=vocab_size).astype(
            np.int64
        )


@dataclass(frozen=True)
class RecordStatistics:
    """Everything a re-analysis of one arm's power figure needs, per record.

    The published figures depend on the cohort through exactly two things. The
    model term is a mean of per-token clean NLL, so per-record ``clean_nll_sum``
    and ``token_count`` reconstruct it and any resampling of it. The context-free
    baseline is a unigram model, so it depends on the cohort only through the
    per-record target-token counts and on the reference corpus only through the
    same counts there. Persisting these makes every uncertainty re-analysis --
    a sequence-clustered bootstrap above all -- a CPU job over a small array
    file instead of a second GPU sweep. That bill has already been paid twice in
    this programme.

    ``record_index`` indexes the cohort's own records, so a record that produced
    no scored target is absent rather than silently renumbered.
    ``reference_counts`` is attached by the caller that built the held-out
    corpus, because the reference is a property of the measurement's
    configuration rather than of the forward pass.
    """

    arm: str
    vocab_size: int
    record_index: np.ndarray
    clean_nll_sum: np.ndarray
    token_count: np.ndarray
    n_symbols: np.ndarray
    target_counts: SparseCounts
    reference_counts: SparseCounts | None = None

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        shape = self.record_index.shape
        for name, dtype in (
            ("record_index", np.int64),
            ("clean_nll_sum", np.float64),
            ("token_count", np.int64),
            ("n_symbols", np.int64),
        ):
            array = getattr(self, name)
            if array.ndim != 1 or array.dtype != dtype:
                raise ValueError(f"{name} must be a one-dimensional {dtype.__name__} array")
            if array.shape != shape:
                raise ValueError(f"{name} does not align with the record index")
        if self.target_counts.n_records != int(shape[0]):
            raise ValueError("target counts do not align with the record index")
        if not np.array_equal(self.target_counts.record_totals(), self.token_count):
            raise ValueError("per-record token counts disagree with the stored target counts")


def record_statistics(arm: Arm, scored: ScoredTokens) -> RecordStatistics:
    """Reduce a scored forward pass to its per-record sufficient statistics.

    Requires the ``budget`` capability for the reason :func:`arm_power` states:
    the vocabulary recorded here is the scoring-target alphabet, resolved by
    :func:`src.capability.core.arms.scoring_target_alphabet` rather than by reading
    ``config.vocab_size`` directly. RITA-xl is the exception: it declares the
    alphabet and is scored through native encoding, but its capability set stays
    empty so lens/pathway grants are not implied.
    """

    if arm.spec.architecture != "rita":
        arm.require("budget")
    alphabet = scoring_target_alphabet(arm.spec, getattr(arm.model, "config", None))
    require_scoring_target_ids(scored.target_ids, alphabet, arm=arm.name)
    order = np.argsort(scored.sequence_index, kind="mergesort")
    ordered_index = scored.sequence_index[order]
    starts = np.unique(ordered_index, return_index=True)[1][1:]
    nll_blocks = np.split(scored.nll_nats[order], starts)
    id_blocks = np.split(scored.target_ids[order], starts)
    return RecordStatistics(
        arm=arm.name,
        vocab_size=int(alphabet["size"]),
        record_index=np.unique(ordered_index).astype(np.int64),
        clean_nll_sum=np.array([block.sum() for block in nll_blocks], dtype=np.float64),
        token_count=np.array([block.size for block in id_blocks], dtype=np.int64),
        n_symbols=np.array(
            [decoded_symbols(arm, block) for block in id_blocks], dtype=np.int64
        ),
        target_counts=SparseCounts.from_records(id_blocks),
    )


def unigram_entropy_nats(target_ids: np.ndarray, vocab_size: int) -> float:
    """Plug-in entropy of the cohort's own scored-token distribution.

    This is the context-free baseline an arm has to beat. Estimating it on the
    cohort itself rather than on a held-out corpus is deliberate: it is the
    tightest baseline available to that arm on that cohort, so the resulting
    context-information figure is a conservative lower bound.
    """

    if target_ids.ndim != 1 or target_ids.size == 0:
        raise ValueError("target_ids must be a non-empty one-dimensional array")
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    if int(target_ids.min()) < 0 or int(target_ids.max()) >= vocab_size:
        raise ValueError("target_ids fall outside the declared vocabulary")
    counts = np.bincount(target_ids, minlength=vocab_size).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def miller_madow_entropy_nats(target_ids: np.ndarray, vocab_size: int) -> float:
    """Miller-Madow corrected entropy, in nats.

    The plug-in estimator is biased downwards, severely so for the 50k-token
    arms on a cohort of this size. The correction is reported as a diagnostic
    because it moves the baseline, and therefore the headline figure, upwards.
    """

    plugin = unigram_entropy_nats(target_ids, vocab_size)
    observed = int(np.unique(target_ids).size)
    return plugin + (observed - 1) / (2.0 * target_ids.size)


def markov_cross_entropy_bits(
    train_sequences: Sequence[str],
    test_sequences: Sequence[str],
    *,
    order: int,
    alphabet: str = AA20,
) -> float:
    """Held-out order-``k`` Markov cross-entropy in bits per symbol.

    Tokenizer-independent by construction, which is the only way a
    multi-residue-BPE arm and a residue-level arm can be placed on one axis.
    Train and test sets must be disjoint or the number is a training fit.
    """

    if order < 0:
        raise ValueError("Markov order must be non-negative")
    if len(set(alphabet)) != len(alphabet) or not alphabet:
        raise ValueError("alphabet must be non-empty and free of duplicates")
    if not train_sequences or not test_sequences:
        raise ValueError("both a training and a test sequence set are required")
    shared = set(train_sequences) & set(test_sequences)
    if shared:
        raise ValueError(
            f"Markov train and test sets share {len(shared)} sequences; "
            "held-out cross-entropy requires disjoint sets"
        )

    index = {symbol: position for position, symbol in enumerate(alphabet)}
    size = len(alphabet)
    counts = np.ones((size,) * order + (size,), dtype=np.float64)
    for sequence in train_sequences:
        encoded = [index[symbol] for symbol in sequence if symbol in index]
        for position in range(order, len(encoded)):
            counts[tuple(encoded[position - order : position]) + (encoded[position],)] += 1.0
    probabilities = counts / counts.sum(axis=-1, keepdims=True)

    total = 0.0
    scored = 0
    for sequence in test_sequences:
        encoded = [index[symbol] for symbol in sequence if symbol in index]
        for position in range(order, len(encoded)):
            probability = probabilities[
                tuple(encoded[position - order : position]) + (encoded[position],)
            ]
            total -= math.log2(probability)
            scored += 1
    if scored == 0:
        raise ValueError("the Markov test set contains no scorable symbols")
    return total / scored


#: Substring length at which a shared exact match between a training and a test
#: protein is evidence of family membership rather than of chance. Over a
#: twenty-letter alphabet a specific 30-mer has probability 20**-30; a corpus of
#: 10**10 residues expects none by chance, so every shared 30-mer is homology.
NEAR_DUPLICATE_KMER = 30


def near_duplicate_fraction(
    train_sequences: Sequence[str], test_sequences: Sequence[str], *, k: int = NEAR_DUPLICATE_KMER
) -> dict[str, Any]:
    """Share of test sequences that have a ``k``-residue exact match in the train set.

    :func:`markov_cross_entropy_bits` enforces "held out" by exact string
    identity, which catches byte-identical records and nothing else -- while the
    hazard this programme has actually declared for these corpora is *near-clonal
    family grouping*. Two sequences differing at one residue share essentially
    every order-1 and order-2 statistic, so an order-2 Markov model fitted on one
    is not held out with respect to the other in any sense that matters, and this
    ladder is described as "the only tokenizer-independent axis on which the
    protein arms can be compared".

    Exact set disjointness cannot be strengthened into clustering here without
    importing an alignment tool this module has no business owning, so the
    remaining leak is *measured* rather than removed: a shared 30-mer is a fact a
    reader can weigh, and a ladder computed on a train/test split where most test
    sequences have one is a ladder to distrust. One pass and one set, so it costs
    nothing next to the ladder it annotates.
    """

    if k < 2:
        raise ValueError("the near-duplicate substring must be at least two residues")
    seen: set[str] = set()
    for sequence in train_sequences:
        seen.update(sequence[i : i + k] for i in range(len(sequence) - k + 1))
    scorable = [sequence for sequence in test_sequences if len(sequence) >= k]
    matched = sum(
        1
        for sequence in scorable
        if any(sequence[i : i + k] in seen for i in range(len(sequence) - k + 1))
    )
    return {
        "kmer": int(k),
        "n_test_sequences_scorable": len(scorable),
        "n_test_sequences_with_shared_kmer": matched,
        "fraction_test_sequences_with_shared_kmer": (
            matched / len(scorable) if scorable else None
        ),
        "interpretation": (
            "disjointness is enforced by exact string identity only; a shared "
            f"{k}-mer over a twenty-letter alphabet is homology rather than chance, "
            "so this fraction bounds how much of the 'held-out' ladder is fitted on "
            "the test set's own protein families"
        ),
    }


def markov_baselines(
    train_sequences: Sequence[str],
    test_sequences: Sequence[str],
    *,
    orders: Sequence[int] = (0, 1, 2),
    alphabet: str = AA20,
) -> dict[str, Any]:
    """The tokenizer-independent residue ladder for a protein cohort."""

    if not orders:
        raise ValueError("at least one Markov order is required")
    return {
        "alphabet_size": len(alphabet),
        "n_train_sequences": len(train_sequences),
        "n_test_sequences": len(test_sequences),
        "cross_entropy_bits_per_residue": {
            f"order{order}": markov_cross_entropy_bits(
                train_sequences, test_sequences, order=order, alphabet=alphabet
            )
            for order in orders
        },
        "maximum_bits_per_residue": math.log2(len(alphabet)),
        # Travels with the ladder, because a reader who is handed only the
        # cross-entropies cannot tell a held-out corpus from a near-clonal one.
        "held_out_strictness": near_duplicate_fraction(train_sequences, test_sequences),
    }


def _supports_trimmed_logits(arm: Arm) -> bool:
    """Whether this build of ``transformers`` lets the arm trim its logit head.

    The answer is a property of the installed library, not of the measurement:
    transformers 4.57.3 gives ``GPT2LMHeadModel.forward`` an explicit
    ``logits_to_keep`` parameter and 4.52.4 does not, and the ProGen2 remote code
    has never had one. It is read from the signature rather than tried, because
    both signatures also accept ``**kwargs``, so an unsupported build would take
    the argument and silently return the full tensor.

    Trimming is not numerically inert. It moves the unembedding matmul from
    ``(batch, tokens, d)`` to ``(batch, 1, d)``, which selects a different cuBLAS
    kernel; measured on gpt2-large in bfloat16 the last-position logits differ by
    up to 0.25 and the resulting per-token NLL by up to 0.12 nats, with a mean
    shift of order 1e-3 nats. Which path ran is therefore recorded by
    :func:`truncation_curve` rather than left to the host.
    """

    return "logits_to_keep" in inspect.signature(arm.model.forward).parameters


@torch.no_grad()
def truncation_curve(
    arm: Arm,
    input_strings: Sequence[str],
    *,
    max_len: int,
    context_lengths: Sequence[int] = DEFAULT_CONTEXT_LENGTHS,
    queries_per_sequence: int = 6,
    batch_size: int = 64,
    seed: int,
    min_windows: int = 200,
) -> dict[str, Any]:
    """Clean NLL at sampled query positions as a function of visible context.

    A cohort can be starved either because the model is off-distribution or
    because the content is locally predictable and nothing beyond a short window
    matters. The two look identical in a single cross-entropy number and are
    told apart by how the NLL moves as context is added.
    """

    lengths = sorted({int(value) for value in context_lengths})
    if not lengths or lengths[0] < 1:
        raise ValueError("context lengths must be positive integers")
    if queries_per_sequence < 1 or batch_size < 1 or min_windows < 1:
        raise ValueError("queries_per_sequence, batch_size and min_windows must be positive")
    longest = lengths[-1]
    if max_len <= longest + 1:
        raise ValueError("max_len must exceed the longest requested context by at least two")

    logit_width = output_logit_width(arm)
    if not _supports_trimmed_logits(arm) and int(logit_width["size"]) > 1024:
        raise RuntimeError(
            f"{arm.name}: live logit width of {logit_width['size']} "
            f"(source {logit_width['source']}) without logits_to_keep support; "
            "the full logit tensor would dominate memory. "
            f"transformers {transformers.__version__} does not expose "
            "logits_to_keep on this architecture's forward; 4.57.3 does for the "
            "GPT-2 family. Relaxing the guard is not a free port: the untrimmed "
            "path is a different unembedding kernel and would not reproduce a "
            "curve measured on the trimmed one. This width is the model interface, "
            "not the scoring-target alphabet, and is not cropped"
        )

    generator = np.random.default_rng(seed)
    windows: list[tuple[list[int], int]] = []
    for text in input_strings:
        ids = arm.tokenizer(text, return_tensors=None)["input_ids"][:max_len]
        first, last = longest, len(ids) - 1
        if last <= first:
            continue
        picks = generator.choice(
            np.arange(first, last), size=min(queries_per_sequence, last - first), replace=False
        )
        windows.extend((ids, int(query)) for query in picks)
    if len(windows) < min_windows:
        raise RuntimeError(
            f"{arm.name}: only {len(windows)} truncation windows survive a "
            f"{longest}-token context requirement; need {min_windows}"
        )

    trimmed = _supports_trimmed_logits(arm)
    curve: dict[int, float] = {}
    for context in lengths:
        batches: list[np.ndarray] = []
        for offset in range(0, len(windows), batch_size):
            chunk = windows[offset : offset + batch_size]
            block = torch.tensor(
                [ids[query - context : query + 1] for ids, query in chunk], dtype=torch.long
            ).to(arm.device)
            extra = {"logits_to_keep": 1} if trimmed else {}
            logits = arm.model(input_ids=block[:, :-1], **extra).logits
            logprobs = F.log_softmax(logits[:, -1].float(), dim=-1)
            nll = -logprobs.gather(-1, block[:, -1:]).squeeze(-1)
            if not bool(torch.isfinite(nll).all()):
                raise FloatingPointError(f"{arm.name}: non-finite truncated NLL")
            batches.append(nll.cpu().numpy().astype(np.float64))
        curve[context] = float(np.concatenate(batches).mean())

    shortest = lengths[0]
    span = curve[shortest] - curve[longest]
    # ``span > 1e-9`` collapsed two different answers into one ``None``. A
    # negative span -- NLL *rising* as context is added -- is a finding: it says
    # the arm is off-distribution on this cohort in a way a single cross-entropy
    # cannot show, and it is one of the two readings this curve exists to
    # separate. A span at zero says the curve is flat and the fraction is
    # genuinely undefined. Reported apart, so a reader does not have to guess
    # which of the two produced the null.
    span_status = (
        "negative_nll_rises_with_context"
        if span < -1e-9
        else "flat_no_context_effect"
        if abs(span) <= 1e-9
        else "positive"
    )
    return {
        "n_windows": len(windows),
        "seed": int(seed),
        # Provenance, not a result: the trimmed and untrimmed unembedding paths
        # are the same quantity through different kernels, and a curve compared
        # across hosts has to say which one produced it.
        "logits_to_keep_used": bool(trimmed),
        "transformers_version": transformers.__version__,
        "context_lengths": lengths,
        "nll_nats_by_context": {str(context): value for context, value in curve.items()},
        "nll_reduction_shortest_to_longest_nats": span,
        "nll_reduction_status": span_status,
        "fraction_of_reduction_beyond_context_8": (
            (curve[8] - curve[longest]) / span if 8 in curve and span > 1e-9 else None
        ),
        "fraction_of_reduction_beyond_context_32": (
            (curve[32] - curve[longest]) / span if 32 in curve and span > 1e-9 else None
        ),
        "fraction_undefined_because": (
            None
            if span > 1e-9
            else (
                "the context span is not positive, so there is no reduction to take "
                "a fraction of; see nll_reduction_status for which case this is"
            )
        ),
    }


def threshold_in_units(
    threshold_nats: float,
    *,
    baseline_entropy_nats: float | None = None,
    symbols_per_token: float | None = None,
) -> dict[str, Any]:
    """One threshold in the three units a threshold has to be published in.

    Nats per token, ``rho = threshold / H_baseline``, and bits per symbol. A
    threshold expressed only in nats is not a cross-arm threshold: one value in
    nats is a different criterion on every arm, and on this panel 0.30 nats is
    rho 0.092 at byte level, 0.040 on the GPT-2 arms, 0.104 at residue level and
    0.034 on ProtGPT2 -- a 3.1x spread under nothing but a per-arm rescaling.

    A unit whose conversion factor the caller does not have is reported as
    ``None`` beside the reason it is missing, never inferred and never silently
    omitted: a reader who sees the key and a null learns that the conversion was
    unavailable here, and a reader who sees no key at all learns nothing.
    """

    if not math.isfinite(threshold_nats):
        raise ValueError("a threshold must be finite")
    rho: float | None = None
    rho_missing: str | None = "no context-free baseline entropy was supplied"
    if baseline_entropy_nats is not None:
        if not math.isfinite(baseline_entropy_nats) or baseline_entropy_nats <= 0.0:
            raise ValueError("the baseline entropy must be finite and positive")
        rho = threshold_nats / baseline_entropy_nats
        rho_missing = None
    bits: float | None = None
    bits_missing: str | None = "no scored symbols-per-token expansion was supplied"
    if symbols_per_token is not None:
        if not math.isfinite(symbols_per_token) or symbols_per_token <= 0.0:
            raise ValueError("the symbols-per-token expansion must be finite and positive")
        bits = threshold_nats / LN2 / symbols_per_token
        bits_missing = None
    return {
        "nats_per_token": float(threshold_nats),
        "relative_to_baseline": rho,
        "relative_to_baseline_undefined_because": rho_missing,
        "bits_per_symbol": bits,
        "bits_per_symbol_undefined_because": bits_missing,
        "baseline_entropy_nats": (
            None if baseline_entropy_nats is None else float(baseline_entropy_nats)
        ),
        "symbols_per_token": (
            None if symbols_per_token is None else float(symbols_per_token)
        ),
    }


def context_identification(
    context_information_nats: float,
    displacement_corrected_lower_bound_nats: float | None,
) -> dict[str, Any]:
    """Whether an arm's context information is distinguishable from zero.

    **This is the identification criterion.** The reading is identified when its
    displacement-corrected 95% interval lies strictly above zero, which is the
    arm's own precision asking the question rather than a constant asking it.
    It answers "did this arm extract information from context, or is the reading
    consistent with zero?", and nothing else: whether the same reading is precise
    enough to divide by is :func:`ratio_denominator_admissibility`, which is
    strictly stronger and therefore nested inside this one rather than crossing
    it.

    **The correction is not optional and the bound must come from it.** An
    uncorrected percentile lower bound reads the Jensen displacement L34 and L42
    catalogue rather than the measurement: at a true zero it sits above zero on
    56 of 112 readings, and the rule it supports has a false-positive rate of
    0.500 at a threshold of 0.005 nats where the point rule has 0.080. With the
    displacement measured and removed the same rule admits **0 of 120** held-out
    null readings and **0 of 56** distinct measurements under all three
    cross-fitting schemes (EXP-R2-221, §5.10). Supply
    ``statistics["information_nats_per_token"]["displacement_corrected_interval"][0]``
    from :mod:`src.capability.context.information_bootstrap`; a raw ``interval[0]`` is the
    rule that was measured failing.

    **There is no fallback when no interval exists.** A stage upstream of the
    bootstrap cannot answer this question, and substituting a magnitude constant
    for the interval is the defect EXP-R2-221 removed -- so a missing or
    non-finite bound raises. Such a stage reports
    :data:`SCREENING_CONTEXT_INFORMATION_NATS` as the pre-interval screen it is,
    under :data:`SCREENING_FLOOR_NOTE`, and names this function as the criterion
    its verdict is not.
    """

    if not math.isfinite(context_information_nats):
        raise ValueError("context information must be finite")
    if displacement_corrected_lower_bound_nats is None:
        raise ValueError(
            "identification needs the lower bound of the displacement-corrected "
            "interval for I and has no fallback: a magnitude constant "
            "substituted for a missing interval is exactly the rule EXP-R2-221 "
            "replaced, and it refused a reading 10.43 standard errors from zero. "
            "A stage with no bootstrap reports the pre-interval screen under "
            "SCREENING_FLOOR_NOTE instead of calling this"
        )
    bound = float(displacement_corrected_lower_bound_nats)
    if not math.isfinite(bound):
        raise ValueError(
            "the displacement-corrected lower bound must be finite; got "
            f"{displacement_corrected_lower_bound_nats!r}. A non-finite endpoint "
            "is not an interval and this criterion cannot be evaluated against it"
        )
    identified = bool(bound > 0.0)
    return {
        "identified": identified,
        "criterion": IDENTIFICATION_CRITERION,
        "verdict": "PASS" if identified else "FAIL",
        "measurability": MEASURABLE if identified else UNMEASURABLE,
        "context_information_nats": float(context_information_nats),
        "displacement_corrected_lower_bound_nats": bound,
        "note": IDENTIFICATION_NOTE,
        # The two demoted constants, reported so that a verdict here stays
        # readable against every verdict recorded under either of them. Neither
        # decides anything above.
        "legacy_screening_floor_nats": SCREENING_CONTEXT_INFORMATION_NATS,
        "clears_legacy_screening_floor": bool(
            context_information_nats >= SCREENING_CONTEXT_INFORMATION_NATS
        ),
        "legacy_screening_floor_note": SCREENING_FLOOR_NOTE,
        "legacy_minimum_context_information_nats": MIN_CONTEXT_INFORMATION_NATS,
        "clears_legacy_floor": bool(
            context_information_nats >= MIN_CONTEXT_INFORMATION_NATS
        ),
        "legacy_floor_note": LEGACY_FLOOR_NOTE,
    }


def ratio_denominator_admissibility(
    context_information_nats: float,
    context_information_se_nats: float | None,
    *,
    baseline_entropy_nats: float | None = None,
    symbols_per_token: float | None = None,
) -> dict[str, Any]:
    """Whether ``I`` is identified far enough from zero to divide by.

    The criterion is Fieller's precondition in its inverted form,
    ``I_hat > FIELLER_DENOMINATOR_MULTIPLE * SE(I_hat)``, evaluated from the
    denominator's own bootstrap standard error. It is a property of one arm on
    one cohort, not a constant: the admissible minimum runs 0.146 to 0.966 nats
    across the panel (EXP-R2-218).

    **It is strictly stronger than the sign test it replaces**, which is the
    whole reason a magnitude guard was introduced. Gating on ``I > 0`` alone once
    admitted a denominator of a hundredth of a nat and published a share with a
    median of 3.57 and a 97.5th percentile of 85.4 -- finite, well formed, and a
    statement about nothing. Since the multiple is positive and ``SE`` must be
    positive, anything this admits the sign test admits too, and the converse
    fails on exactly those denominators.

    **There is no fallback when the standard error is missing.** A magnitude
    constant substituted for an unavailable ``SE`` is precisely the defect
    EXP-R2-218 measured, so an absent, non-finite or non-positive ``SE`` raises.
    A zero ``SE`` raises with the rest: a bootstrap that produced no spread has
    not estimated the denominator's error, and treating it as zero would collapse
    this criterion back onto the sign test.
    """

    if not math.isfinite(context_information_nats):
        raise ValueError("context information must be finite")
    if context_information_se_nats is None:
        raise ValueError(
            "denominator admissibility needs the bootstrap standard error of the "
            "context information and has no fallback: the retired "
            f"{MIN_CONTEXT_INFORMATION_NATS}-nat constant was 20-30x stricter than "
            "identification requires and up to 3.2x too lax for this criterion, so "
            "substituting it for a missing SE would reinstate the defect EXP-R2-218 "
            "measured. Supply SE(I) from the same bootstrap that produced I"
        )
    if not math.isfinite(context_information_se_nats) or context_information_se_nats <= 0.0:
        raise ValueError(
            "the context-information standard error must be finite and strictly "
            f"positive; got {context_information_se_nats!r}. A zero or non-finite SE "
            "is not an estimate of the denominator's error, and this criterion "
            "cannot be evaluated without one"
        )
    minimum = FIELLER_DENOMINATOR_MULTIPLE * float(context_information_se_nats)
    z = FIELLER_DENOMINATOR_MULTIPLE * math.sqrt(FIELLER_MAXIMUM_G)
    g = (
        math.inf
        if context_information_nats == 0.0
        else (z * context_information_se_nats / context_information_nats) ** 2
    )
    return {
        "admissible": bool(context_information_nats > minimum),
        "criterion": "fieller_precondition_on_the_denominator",
        "context_information_nats": float(context_information_nats),
        "context_information_se_nats": float(context_information_se_nats),
        "fieller_denominator_multiple": FIELLER_DENOMINATOR_MULTIPLE,
        "fieller_g": float(g),
        "fieller_maximum_g": float(FIELLER_MAXIMUM_G),
        "minimum_admissible_context_information": threshold_in_units(
            minimum,
            baseline_entropy_nats=baseline_entropy_nats,
            symbols_per_token=symbols_per_token,
        ),
        "legacy_minimum_context_information_nats": MIN_CONTEXT_INFORMATION_NATS,
        "clears_legacy_floor": bool(
            context_information_nats >= MIN_CONTEXT_INFORMATION_NATS
        ),
        "legacy_floor_note": LEGACY_FLOOR_NOTE,
    }


def power_status(context_information_nats: float, threshold_nats: float) -> tuple[str, str]:
    """Map a measured power figure onto a verdict and a measurability status.

    A FAIL here is a statement about the cohort, not about the model or the
    interpretability method: below the threshold the arm's reading is not
    distinguishable from no context signal at all, so the arm must be excluded
    rather than reported as a negative result.

    **This is the pre-interval screen and not the identification criterion.** It
    compares a point estimate against :data:`SCREENING_CONTEXT_INFORMATION_NATS`,
    which is the only comparison a caller upstream of any bootstrap can make.
    Identification is :func:`context_identification`, evaluated on the arm's own
    displacement-corrected interval wherever one exists, and whether the reading
    may serve as the denominator of a share is
    :func:`ratio_denominator_admissibility`, stricter again. A caller that has an
    interval must not take its verdict from here.
    """

    if not math.isfinite(context_information_nats):
        raise ValueError("context information must be finite")
    if not math.isfinite(threshold_nats) or threshold_nats <= 0.0:
        raise ValueError("threshold must be finite and positive")
    if context_information_nats >= threshold_nats:
        return "PASS", MEASURABLE
    return "FAIL", UNMEASURABLE


def arm_power_with_records(
    arm: Arm,
    cohort: Cohort,
    *,
    max_len: int,
    batch_size: int,
    minimum_context_information_nats: float = SCREENING_CONTEXT_INFORMATION_NATS,
    unigram_estimator: str = "plugin",
    reference_token_counts: np.ndarray | None = None,
    reference: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], RecordStatistics]:
    """The headline power figure for one arm, and the records behind it.

    **The baseline is chosen here, where the scored targets are.** The estimator
    is named at the call: ``"plugin"`` fits the context-free baseline on the
    scored cohort itself, ``"disjoint"`` evaluates a unigram model fitted on a
    held-out reference corpus, whose token-count vector and provenance the
    caller supplies. Both go through
    :func:`src.capability.pathways.unigram_baseline`, so this function has no
    estimator dispatch of its own and there is no fallback: asking for the
    held-out estimator without a reference raises, because a silent downgrade
    would move every context-information figure without changing anything a
    reader can see.

    This used to be split across two files. ``arm_power`` computed everything
    against the plug-in and ``cohort_power.py`` recomputed *some* of it
    afterwards from a held-out cross-entropy of its own, which left the rest of
    the record normalised against the estimator that had been replaced. The
    published consequence: ZymCTRL carried
    ``context_information_miller_madow_nats`` 2.027 beside
    ``context_information_nats`` 2.029, two "context information" figures taken
    against different baselines with nothing in either name to say so.

    **How a field says which baseline it came from.** A name carrying
    ``plug_in``, ``on_cohort`` or ``miller_madow`` belongs to the in-cohort
    plug-in family and is computed against the plug-in whichever estimator was
    asked for. Every other baseline-derived field is computed against the
    estimator named in ``unigram_estimator``, published as a number in
    ``unigram_entropy_used_for_verdict_nats`` and with its full provenance in
    ``unigram_baseline``. The Miller-Madow variants are named
    ``*_plug_in_miller_madow_*`` rather than left bare because the correction
    ``(observed - 1) / 2N`` is a bias correction *for the plug-in entropy
    estimator*: it has no meaning applied to a held-out cross-entropy, so the
    honest form of the field is the plug-in one, named as such.

    The ``budget`` capability is required here rather than only at the stages
    that call this, because the scoring-target alphabet is the support an
    entropy is taken over. Panel members still read that support from
    ``config.vocab_size``. Staged ProGen2 rungs declare 32 on the ArmSpec so
    that progen2-large's 51200-column head and progen2-xlarge's missing
    ``vocab_size`` cannot become the unigram support. The live logit width is
    not cropped. Tokenizer length is not a fallback.

    The second return value is the per-record reduction of the same forward
    pass; see :class:`RecordStatistics` for why it is worth persisting.
    """

    # pathways declares the estimators and owns the held-out cross-entropy, and
    # it imports this module's measurability floor, so the import is deferred to
    # the call rather than duplicating either declaration.
    from .measurement_support import UNIGRAM_ESTIMATORS, unigram_baseline

    if arm.spec.architecture != "rita":
        arm.require("budget")
    if unigram_estimator not in UNIGRAM_ESTIMATORS:
        raise ValueError(
            f"unknown unigram estimator {unigram_estimator!r}; known {UNIGRAM_ESTIMATORS}"
        )
    supplied = reference_token_counts is not None or reference is not None
    if unigram_estimator == "disjoint" and not (
        reference_token_counts is not None and reference is not None
    ):
        raise ValueError(
            f"{arm.name}: the disjoint unigram estimator needs a held-out reference "
            "corpus -- both its token-count vector and its provenance record -- and "
            "there is no fallback to the plug-in. Supply one, or ask for the plug-in "
            "estimator explicitly."
        )
    if unigram_estimator == "plugin" and supplied:
        raise ValueError(
            f"{arm.name}: a held-out reference was supplied to the plug-in estimator. "
            "It would enter no published figure, and a record that names a reference "
            "it did not use is worse than one that names none."
        )

    inputs = cohort.input_strings(arm)
    scored = scored_tokens(arm, inputs, max_len=max_len, batch_size=batch_size)
    alphabet = scoring_target_alphabet(arm.spec, getattr(arm.model, "config", None))
    require_scoring_target_ids(scored.target_ids, alphabet, arm=arm.name)
    vocab = int(alphabet["size"])
    target_counts = np.bincount(scored.target_ids, minlength=vocab).astype(np.int64)
    if target_counts.size != vocab:
        raise ValueError(
            f"{arm.name}: scored target ids fall outside the scoring-target alphabet of {vocab}"
        )
    baseline_record = unigram_baseline(
        arm,
        estimator=unigram_estimator,
        target_counts=target_counts,
        reference_counts=reference_token_counts,
        reference=reference,
    )
    held_out = unigram_estimator == "disjoint"
    plug_in = float(baseline_record["cohort_plug_in_entropy_nats"])
    baseline = float(baseline_record["nats"])
    baseline_mm = miller_madow_entropy_nats(scored.target_ids, vocab)
    clean_ce = float(scored.nll_nats.mean())
    context_information = baseline - clean_ce
    verdict, status = power_status(context_information, minimum_context_information_nats)
    # Always the plug-in verdict, under a name that says so. It is what
    # ``power_verdict`` is when the plug-in estimator was asked for, and it stays
    # true beside a held-out verdict, so it can never go stale the way an
    # "estimator" label beside a recomputed verdict would.
    plug_in_verdict, plug_in_status = power_status(
        plug_in - clean_ce, minimum_context_information_nats
    )
    if not held_out:
        status = MEASURABLE_PLUG_IN if status == MEASURABLE else UNMEASURABLE_PLUG_IN

    records = record_statistics(arm, scored)
    # One reduction, used twice: the interval below and the sidecar are the same
    # per-record quantities, so they cannot disagree.
    per_sequence_ce = [
        float(value) for value in records.clean_nll_sum / records.token_count
    ]
    rendered_expansion = symbols_per_token(arm, inputs, max_len)
    # The per-symbol conversion divides a per-*scored-token* quantity, so it has
    # to divide by the expansion of the scored window and not of the rendered
    # string. Both are reported: the rendered figure describes the rendering and
    # several artefacts quote it, the scored figure is the one the bits-per-
    # symbol axis is built on.
    expansion = scored_symbols_per_token(arm, scored)

    report = {
        "arm": arm.name,
        "modality": arm.modality,
        "input_format": arm.spec.input_format,
        "tokenisation": arm.spec.tokenisation,
        "vocab_size": vocab,
        "scoring_target_alphabet_size": vocab,
        "scoring_target_alphabet_source": alphabet["source"],
        "max_len": int(max_len),
        "n_sequences": len(inputs),
        "n_scored_tokens": len(scored),
        "n_distinct_scored_tokens": int(np.unique(scored.target_ids).size),
        "symbols_per_token": expansion,
        "symbols_per_token_rendered_string": rendered_expansion,
        "symbols_per_token_basis": (
            "scored next-token targets only; for an EC-conditioned arm the "
            "rendered-string figure counts prompt tokens that carry no alphabet "
            "symbol and so understates the expansion, which inflates every "
            "bits-per-symbol figure for that arm alone"
        ),
        # The estimator, its smoothing, its sweep and its reference corpus, in
        # one block produced by the function that computed the number. Nothing
        # downstream recomputes any of it, so nothing can go stale beside it.
        "unigram_baseline": baseline_record,
        "unigram_estimator": baseline_record["estimator"],
        "baseline_naming": (
            "a field whose name carries plug_in, on_cohort or miller_madow is "
            "computed against the in-cohort plug-in baseline; every other "
            "baseline-derived field is computed against unigram_estimator, whose "
            "value is unigram_entropy_used_for_verdict_nats"
        ),
        "unigram_estimator_bias": (
            "held-out unigram cross-entropy on a disjoint reference corpus; "
            "carries an upward bias from its additive smoothing that grows with "
            "vocabulary size against reference size, measured in "
            "unigram_baseline.smoothing_diagnostics"
            if held_out
            else "plug-in on the scored targets; biased downwards by an amount that "
            "grows with vocabulary size, so context information is understated "
            "and understated unequally across the panel"
        ),
        "cross_arm_comparable": held_out,
        "unigram_entropy_on_cohort_nats": plug_in,
        "unigram_entropy_used_for_verdict_nats": baseline,
        "unigram_entropy_plug_in_miller_madow_nats": baseline_mm,
        "clean_ce_nats": clean_ce,
        "context_information_nats": context_information,
        "context_information_plug_in_nats": plug_in - clean_ce,
        "context_information_plug_in_miller_madow_nats": baseline_mm - clean_ce,
        "context_information_bits_per_symbol": context_information / LN2 / expansion,
        "clean_ce_bits_per_symbol": clean_ce / LN2 / expansion,
        "unigram_entropy_bits_per_symbol": baseline / LN2 / expansion,
        # ``clean_ce_nats`` above is token-weighted: a 2000-residue sequence
        # contributes thirty times what a 64-residue one does, which is the
        # convention ``scoring.per_sequence_scores`` states and defends. This
        # interval is over an unweighted mean of per-sequence cross-entropies.
        # On a 64-2000 residue cohort those are not the same estimand, so the
        # interval is not an interval *for* the number printed beside it, and
        # ``mean_interval``'s own ``mean`` key is the only place the difference
        # was visible. The sequence-weighted point estimate is now published so
        # that the pair reads as a point estimate and its interval.
        "clean_ce_nats_sequence_weighted": float(np.mean(per_sequence_ce)),
        "clean_ce_weighting": "token",
        "per_sequence_clean_ce_interval": {
            **mean_interval(per_sequence_ce),
            "estimand": "unweighted mean over sequences of the per-sequence clean CE",
            "is_an_interval_for_clean_ce_nats": False,
        },
        # The same shift applied to every endpoint: this interval is the one
        # above translated by a constant baseline, so it carries the spread of
        # the cross-entropy and none of the uncertainty in the baseline it is
        # measured against, despite the baseline being an estimate too.
        "per_sequence_context_information_interval": {
            **mean_interval([baseline - value for value in per_sequence_ce]),
            "estimand": (
                "unweighted mean over sequences of (fixed baseline - per-sequence "
                "clean CE)"
            ),
            "is_an_interval_for_context_information_nats": False,
            "baseline_uncertainty_included": False,
        },
        "minimum_context_information_nats": float(minimum_context_information_nats),
        # The threshold that produced the verdict, in all three units, because a
        # figure in nats alone is a different criterion on every arm.
        "minimum_context_information": threshold_in_units(
            minimum_context_information_nats,
            baseline_entropy_nats=baseline,
            symbols_per_token=expansion,
        ),
        "measurability_criterion": (
            "PRE-INTERVAL SCREEN: the point estimate against "
            "budget.SCREENING_CONTEXT_INFORMATION_NATS. This function runs "
            "upstream of any bootstrap -- it is what produces the per-record "
            "statistics an interval is computed from -- so it cannot evaluate "
            "the identification criterion and does not claim to"
        ),
        # Named rather than left out: a report that carried a verdict and no
        # statement about the criterion would read as an identification verdict.
        "identification_criterion": IDENTIFICATION_CRITERION,
        "identification_evaluable_here": False,
        "identification_not_evaluable_reason": (
            "budget.context_identification needs the lower bound of the "
            "displacement-corrected bootstrap interval for I, and this stage "
            "computes no bootstrap. Persist the per-record sufficient statistics "
            "and take the verdict from context_information_bootstrap.py"
        ),
        "screening_floor_note": SCREENING_FLOOR_NOTE,
        # Legacy column: what the retired single floor would have said here.
        "legacy_minimum_context_information_nats": MIN_CONTEXT_INFORMATION_NATS,
        "clears_legacy_floor": bool(
            context_information >= MIN_CONTEXT_INFORMATION_NATS
        ),
        "legacy_floor_note": LEGACY_FLOOR_NOTE,
        "power_verdict": verdict,
        "measurability": status,
        # Estimator-qualified and never overwritten, so an artefact always holds
        # one verdict whose provenance is unambiguous.
        "power_verdict_plug_in": plug_in_verdict,
        "measurability_plug_in": plug_in_status,
    }
    if held_out:
        report["unigram_entropy_held_out_nats"] = baseline
        report["plug_in_bias_nats"] = baseline - plug_in
    return report, records


def arm_power(
    arm: Arm,
    cohort: Cohort,
    *,
    max_len: int,
    batch_size: int,
    minimum_context_information_nats: float = SCREENING_CONTEXT_INFORMATION_NATS,
    unigram_estimator: str = "plugin",
    reference_token_counts: np.ndarray | None = None,
    reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The headline power figure for one arm on one frozen cohort.

    The report half of :func:`arm_power_with_records`, which documents the
    contract. Callers that persist the per-record sufficient statistics take the
    pair; callers that only publish the report take this.
    """

    report, _ = arm_power_with_records(
        arm,
        cohort,
        max_len=max_len,
        batch_size=batch_size,
        minimum_context_information_nats=minimum_context_information_nats,
        unigram_estimator=unigram_estimator,
        reference_token_counts=reference_token_counts,
        reference=reference,
    )
    return report


def write_power_records(
    path: Path,
    statistics: Mapping[str, RecordStatistics],
    *,
    cohort_digest: str,
    reference_digest: str | None,
    smoothing: float | None,
    seeds: Mapping[str, int],
    max_len: int,
) -> dict[str, Any]:
    """Persist per-record sufficient statistics beside a power report.

    One array file for the whole stage, because one stage run is one cohort and
    one set of seeds; per-arm arrays are prefixed ``"<arm>::"`` since the
    vocabulary, the tokenisation and therefore every count differ by arm. The
    returned block is the artefact's own record of the file -- schema, name and
    digest -- and belongs in the report JSON, so that a reader learns from the
    report whether the sidecar exists and whether it is the one that was
    written. A companion JSON would be a second file free to drift from the
    first.

    Global entries: ``schema_version``, ``arms``, ``cohort_digest``,
    ``reference_digest`` (empty when the run used no reference), ``smoothing``
    (NaN when no reference, in which case no smoothing entered any figure),
    ``max_len``, ``seed_names`` and ``seed_values``. Per arm: ``vocab_size``,
    ``record_index``, ``clean_nll_sum``, ``token_count``, ``n_symbols``,
    ``counts_offsets``, ``unique_token_ids``, ``counts`` and -- when a reference
    was used -- ``reference_token_count``, ``reference_counts_offsets``,
    ``reference_unique_token_ids``, ``reference_counts``.

    No sequence text: the cohort JSON beside this file already holds every
    record, and a second copy under a different digest is a second source.
    """

    if not statistics:
        raise ValueError("a sufficient-statistics sidecar needs at least one arm")
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(POWER_RECORDS_SCHEMA_VERSION),
        "arms": np.asarray(sorted(statistics), dtype=np.str_),
        "cohort_digest": np.asarray(str(cohort_digest)),
        "reference_digest": np.asarray("" if reference_digest is None else reference_digest),
        "smoothing": np.asarray(math.nan if smoothing is None else float(smoothing)),
        "max_len": np.asarray(int(max_len), dtype=np.int64),
        "seed_names": np.asarray(sorted(seeds), dtype=np.str_),
        "seed_values": np.asarray([int(seeds[name]) for name in sorted(seeds)], dtype=np.int64),
    }
    for name, record in sorted(statistics.items()):
        if "::" in name:
            raise ValueError(f"arm name {name!r} collides with the sidecar's key separator")
        payload[f"{name}::vocab_size"] = np.asarray(int(record.vocab_size), dtype=np.int64)
        payload[f"{name}::record_index"] = record.record_index
        payload[f"{name}::clean_nll_sum"] = record.clean_nll_sum
        payload[f"{name}::token_count"] = record.token_count
        payload[f"{name}::n_symbols"] = record.n_symbols
        payload[f"{name}::counts_offsets"] = record.target_counts.offsets
        payload[f"{name}::unique_token_ids"] = record.target_counts.token_ids
        payload[f"{name}::counts"] = record.target_counts.counts
        if record.reference_counts is None:
            continue
        payload[f"{name}::reference_token_count"] = record.reference_counts.record_totals()
        payload[f"{name}::reference_counts_offsets"] = record.reference_counts.offsets
        payload[f"{name}::reference_unique_token_ids"] = record.reference_counts.token_ids
        payload[f"{name}::reference_counts"] = record.reference_counts.counts

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    # The three steps src.capability.core.io._atomic_write performs and the reasons it
    # states, over a payload numpy writes rather than one this process can hand
    # over as bytes: commit the contents, rename, then commit the directory
    # entry that names them.
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "schema_version": POWER_RECORDS_SCHEMA_VERSION,
        "path": destination.name,
        "sha256": sha256_file(destination),
        "arms": sorted(statistics),
        "n_scored_records": {
            name: int(record.record_index.size) for name, record in sorted(statistics.items())
        },
        "n_reference_records": {
            name: (None if record.reference_counts is None else record.reference_counts.n_records)
            for name, record in sorted(statistics.items())
        },
        "declared_by": "src.capability.core.budget.write_power_records",
    }
