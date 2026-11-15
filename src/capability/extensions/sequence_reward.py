"""Closed-form sequence statistics, and the rewards whose provenance differs.

The experiment this serves is declared in ``docs/rl-environment-provenance.md``.
Its one manipulated variable is where a reward *comes from*. Protein design has
no simulated environment, so every reward a protein policy is usually optimised
against is a learned proxy trained on the same evolutionary corpus as the policy.
This module holds the half of the reward surface that is **not** such a proxy:
net charge at a declared pH, sequence length, and a residue-level low-complexity
mask, each computed in closed form from the residue string alone, with every
constant declared here and nowhere else.

The learned half -- mean CA-pLDDT from ESMFold2 -- is not computed here. It
arrives as a number the caller measured through the project's own folding
instrument (``src.capability.generation.structure_evidence``), and
:func:`reward` combines it with the closed-form terms. That split is deliberate:
this module imports no model, touches no GPU, and is therefore testable against
hand-computed values, which is what a reward function has to be before anything
is trained against it.

The terms, and the one shape they share
---------------------------------------
Every term is a :func:`plateau` score in ``[0, 1]``: one inside a declared
window, decaying linearly to zero over a declared half-width outside it. The
shape is the same for all three so that none of them can quietly carry a
different notion of "satisfied", and the decay exists because a conjunction of
hard windows gives a group of completions that all miss the window an identical
reward, and a group-relative advantage over identical rewards is exactly zero.
A binary conjunction would therefore not be a trainable objective at all, and
declaring it as one would be declaring an experiment that cannot move.

The reward is one at the conjunction and only there, which is the declared
objective; off it, the score says how far off.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from ..core.amino_acids import AA20
from ..forcing.backbone_cohort import distinct_shingle_fraction

# ------------------------------------------------------------- net charge
#
# The EMBOSS pKa set, which is also the set Biopython's ``IsoelectricPoint``
# carries. Transcribed here rather than imported so that the charge a reward
# is paid for does not depend on which optional package a host has, which is
# the same reason ``core.amino_acids`` transcribes BLOSUM62.

#: pH at which the declared net charge is evaluated.
REFERENCE_PH = 7.0

#: Terminal ionizable groups: one amino terminus and one carboxy terminus per
#: chain, whatever the sequence.
PKA_N_TERMINUS = 8.6
PKA_C_TERMINUS = 3.6

#: Side chains protonated to a positive charge below their pKa.
PKA_POSITIVE: Mapping[str, float] = {"K": 10.8, "R": 12.5, "H": 6.5}

#: Side chains deprotonated to a negative charge above their pKa.
PKA_NEGATIVE: Mapping[str, float] = {"D": 3.9, "E": 4.1, "C": 8.5, "Y": 10.1}

PKA_PROVENANCE = (
    "EMBOSS pKa set (Nterm 8.6, Cterm 3.6, K 10.8, R 12.5, H 6.5, D 3.9, "
    "E 4.1, C 8.5, Y 10.1), Henderson-Hasselbalch per ionizable group, "
    "evaluated at pH 7.0"
)


def _protonated(ph: float, pka: float) -> float:
    """Fraction of a group that is protonated at ``ph``: 1 / (1 + 10^(pH - pKa))."""

    return 1.0 / (1.0 + 10.0 ** (ph - pka))


def net_charge(sequence: str, *, ph: float = REFERENCE_PH) -> float:
    """Net charge of one residue string at ``ph``, in units of the elementary charge.

    Each positive group contributes its protonated fraction and each negative
    group its deprotonated fraction, so the quantity is continuous in ``ph`` and
    exactly computable from the sequence. The two chain termini are counted once
    each; a non-canonical character carries no ionizable side chain and
    contributes nothing but is still a residue of the chain.

    An empty sequence has no chain and therefore no termini, and is zero rather
    than carrying a spurious +0.976 from an amino terminus that does not exist.
    """

    if not sequence:
        return 0.0
    counts = Counter(sequence)
    charge = _protonated(ph, PKA_N_TERMINUS) - (1.0 - _protonated(ph, PKA_C_TERMINUS))
    for residue, pka in PKA_POSITIVE.items():
        charge += counts.get(residue, 0) * _protonated(ph, pka)
    for residue, pka in PKA_NEGATIVE.items():
        charge -= counts.get(residue, 0) * (1.0 - _protonated(ph, pka))
    return float(charge)


# ------------------------------------------------- low-complexity residues
#
# The published preference objective the memory-derived arm takes its pLDDT
# statistic from also reads "fewer than 25% low-complexity residues". It does not
# say which mask produced that fraction, and this repository holds no
# residue-level low-complexity mask at all: its existing screen is the distinct
# 5-mer fraction, which is a ranking over whole sequences rather than a
# per-residue label and whose direction is the opposite of what its name suggests
# -- higher means more diverse.
#
# So there are three distinct statistics here and not three versions of one. The
# published objective's mask is unknown. The objective-alignment campaign's
# ``objective_tuned_analysis._low_complexity`` is a distinct-5-mer fraction with
# no window and no entropy in it. And the mask below is **this experiment's own**,
# declared here with its window, its entropy threshold and its union rule. The
# only thing it shares with the published objective is the 0.25 limit on the share
# of residues; how a residue earns the label is ours, and nothing here reproduces
# that pipeline's mask or is interchangeable with the campaign's screen.

#: Window length, in residues, over which local composition entropy is read.
LOW_COMPLEXITY_WINDOW = 12

#: A window at or below this Shannon entropy, in bits over the residues it
#: contains, is low-complexity. A window of 12 distinct residues carries
#: log2(12) = 3.585 bits; four residues in equal proportion carry 2.0 and six
#: carry 2.585, so 2.2 bits labels a window that is effectively using about
#: four or five residues.
LOW_COMPLEXITY_ENTROPY_BITS = 2.2

#: The share of labelled residues at which the declared constraint is violated,
#: taken from the published objective.
LOW_COMPLEXITY_LIMIT = 0.25

LOW_COMPLEXITY_PROVENANCE = (
    f"union of sliding windows of {LOW_COMPLEXITY_WINDOW} residues whose "
    f"composition entropy is at most {LOW_COMPLEXITY_ENTROPY_BITS} bits; a "
    "residue is low-complexity if any window containing it is labelled. This is "
    "this experiment's own declared mask. It is not the published DPO pipeline's, "
    "which does not state which mask produced its fraction, and it is not the "
    "objective-alignment campaign's distinct-5-mer screen, which has no window "
    "and no entropy in it and runs in the opposite direction to its own name. "
    f"The only thing shared with the published objective is the "
    f"{LOW_COMPLEXITY_LIMIT:g} limit on the share of residues"
)


def composition_entropy_bits(sequence: str) -> float:
    """Shannon entropy of the residue composition of one string, in bits.

    Zero for an empty string and for a homopolymer, which are the two cases a
    degeneracy diagnostic has to separate from a protein-like composition.
    """

    if not sequence:
        return 0.0
    total = len(sequence)
    entropy = 0.0
    for count in Counter(sequence).values():
        share = count / total
        entropy -= share * math.log2(share)
    return float(entropy)


def low_complexity_fraction(
    sequence: str,
    *,
    window: int = LOW_COMPLEXITY_WINDOW,
    entropy_bits: float = LOW_COMPLEXITY_ENTROPY_BITS,
) -> float:
    """Share of residues covered by at least one low-entropy window.

    A sequence shorter than one window is read as a single window, so that a
    short homopolymer is labelled rather than escaping the mask for being too
    short to slide over. An empty sequence has no residues and is zero.
    """

    if window < 1:
        raise ValueError("a low-complexity window is at least one residue")
    length = len(sequence)
    if length == 0:
        return 0.0
    size = min(int(window), length)
    labelled = bytearray(length)
    for start in range(length - size + 1):
        if composition_entropy_bits(sequence[start : start + size]) <= entropy_bits:
            for index in range(start, start + size):
                labelled[index] = 1
    return float(sum(labelled) / length)


def longest_single_residue_run(sequence: str) -> int:
    """Longest run of one repeated residue, the plainest degeneracy diagnostic."""

    best = run = 0
    previous = ""
    for residue in sequence:
        run = run + 1 if residue == previous else 1
        previous = residue
        best = max(best, run)
    return int(best)


# ------------------------------------------------------------- term shape

def plateau(value: float, *, low: float, high: float, half_width: float) -> float:
    """One inside ``[low, high]``, decaying linearly to zero over ``half_width``.

    The shape every declared term takes. ``half_width`` is the distance outside
    the window at which the score reaches zero; it is strictly positive, because
    a zero half-width is a hard window and a group of completions that all miss
    a hard window carries no group-relative advantage at all.
    """

    if not math.isfinite(value):
        raise ValueError("a term score needs a finite value")
    if high < low:
        raise ValueError(f"a window is ordered: got [{low}, {high}]")
    if not half_width > 0:
        raise ValueError(
            "a term's decay half-width is strictly positive; a hard window gives "
            "a degenerate group and therefore no gradient"
        )
    if low <= value <= high:
        return 1.0
    distance = low - value if value < low else value - high
    return float(max(0.0, 1.0 - distance / float(half_width)))


# ------------------------------------------------------- declared windows
#
# Set from the base policy's own measured distributions before any training ran,
# and recorded with that provenance in the declaration: a target the base model
# already meets measures nothing, and a target the base model cannot approach at
# all gives a flat reward surface. The calibration record is named in the
# declaration; the values are frozen here.

#: Net charge at pH 7.0 the environment reward asks for.
CHARGE_WINDOW: tuple[float, float] = (8.0, 12.0)
#: Charge units beyond the window at which the charge term reaches zero.
CHARGE_HALF_WIDTH = 16.0

#: Residue count the environment reward asks for.
LENGTH_WINDOW: tuple[int, int] = (60, 120)
#: Residues beyond the window at which the length term reaches zero.
LENGTH_HALF_WIDTH = 120.0

#: The positive control's charge window: a target a rejection sampler of the
#: declared budget cannot reach from the base pool. ``None`` until it is chosen,
#: and chosen by one rule applied once to the base pool's measured conjunction
#: rates -- see ``rl_environment_provenance.RUNG_RULE``. Everything that would
#: pay or score this reward refuses while it is ``None``, because a positive
#: control whose target was picked after its arm's result is not a control.
HARD_CHARGE_WINDOW: tuple[float, float] | None = (32.0, 36.0)

HARD_CHARGE_PROVENANCE = (
    "the first rung of the declared charge ladder whose expected best-of-512 "
    "shaped environment reward over the base pool falls below 0.70, read off the "
    "base pool's own measured ladder exactly once and written here before the arm "
    "trained. Its R@512 is 0.6231 against the next easier rung's 0.7394.\n"
    "This is the ladder's *second* rule. The first was posed on the conjunction "
    "indicator and selected the declared target itself, because the base pool "
    "attains the conjunction nowhere in 3,072 draws and the exact bound therefore "
    "cleared its threshold at the first rung. It was voided on the narrow ground "
    "that its answer was degenerate by construction -- a rung equal to the "
    "declared target makes this arm a duplicate of arm E, which answers no "
    "question -- and not on the ground that the answer was unwelcome. See "
    "rl_environment_provenance.RUNG_RULE_VOIDED.\n"
    "The rung is hard and not impossible: it is attainable at reward exactly 1.0 "
    "by a lysine-rich, non-repeating, full-diversity sequence, at a K share of "
    "0.28 in the degeneracy sweep's own witness and 0.350, 0.318 and 0.325 at "
    "lengths 100, 110 and 120 in an independently constructed one"
)

#: CA-pLDDT scale the memory-derived term is divided by, so every term is a
#: share of its own attainable maximum and the arms' rewards are commensurable.
PLDDT_SCALE = 100.0

#: Residue counts on which a structural statement is defined at all. This is the
#: project's own declared structural support, ``unconditional_generation.POLICY``
#: ``["structure_support"]``, adopted rather than reinvented; a test asserts the
#: two have not drifted.
#:
#: It is a reward rule and not an engineering guard, and the measurement that
#: forces it is this: the folding instrument returns a mean CA-pLDDT of 85.3 on a
#: two-residue sequence and 87.8 on a four-residue one, against about 47 for the
#: base pool's own samples. A predicted-confidence number on a tetrapeptide is not
#: a structural statement, it is the instrument's behaviour on an input outside
#: what it describes -- which is why the project does not fold below sixteen
#: residues anywhere else either. Paying a structural reward there would open a
#: channel on which the policy could score above every natural-length sequence by
#: emitting peptides, and a gain earned that way would be an instrument artifact
#: wearing the shape of an installed bias.
#:
#: A completion outside the support is therefore paid the empty-completion reward,
#: which is a true statement under the objective rather than a missing
#: measurement: the policy emitted something the objective is not about. The
#: refusal in :func:`reward` is kept for the case it was written for, an
#: in-support completion whose fold is absent.
STRUCTURE_SUPPORT: tuple[int, int] = (16, 1024)

STRUCTURE_SUPPORT_PROVENANCE = (
    "unconditional_generation.POLICY['structure_support'], the residue range this "
    "project folds in every lane. Outside it a structural reward is paid the "
    "empty-completion value, because the instrument's output there describes the "
    "input rather than a structure: measured mean CA-pLDDT is 85.3 at two residues "
    "and 87.8 at four, against about 47 on the base pool's own samples, and it "
    "raises an IndexError at one residue"
)


def within_structure_support(stats: SequenceStatistics) -> bool:
    """Whether a structural statement is defined for this completion at all."""

    low, high = STRUCTURE_SUPPORT
    return low <= stats.length <= high


@dataclass(frozen=True)
class SequenceStatistics:
    """Every closed-form quantity one completion carries, measured once.

    The rewards, the degeneracy diagnostics and the declared constraint all read
    these, so a completion is measured once and every downstream number refers to
    the same measurement.
    """

    sequence: str
    length: int
    net_charge: float
    low_complexity_fraction: float
    composition_entropy_bits: float
    longest_single_residue_run: int
    distinct_5mer_fraction: float
    non_canonical_residues: int

    def as_record(self) -> dict[str, Any]:
        return {
            "length": self.length,
            "net_charge_ph7": self.net_charge,
            "low_complexity_fraction": self.low_complexity_fraction,
            "composition_entropy_bits": self.composition_entropy_bits,
            "longest_single_residue_run": self.longest_single_residue_run,
            "distinct_5mer_fraction": self.distinct_5mer_fraction,
            "non_canonical_residues": self.non_canonical_residues,
        }


def statistics(sequence: str) -> SequenceStatistics:
    """Measure one completion's closed-form statistics."""

    canonical = set(AA20)
    return SequenceStatistics(
        sequence=sequence,
        length=len(sequence),
        net_charge=net_charge(sequence),
        low_complexity_fraction=low_complexity_fraction(sequence),
        composition_entropy_bits=composition_entropy_bits(sequence),
        longest_single_residue_run=longest_single_residue_run(sequence),
        distinct_5mer_fraction=float(distinct_shingle_fraction(sequence)),
        non_canonical_residues=sum(1 for residue in sequence if residue not in canonical),
    )


# ------------------------------------------------------------- the rewards

#: The memory-derived reward: the published preference objective's own *score* --
#: mean CA-pLDDT from a predictor trained on the same evolutionary corpus as the
#: policy -- under this experiment's own low-complexity constraint at that
#: objective's 0.25 limit. The pLDDT statistic is shared; the mask is not, and
#: :data:`LOW_COMPLEXITY_PROVENANCE` says so.
REWARD_MEMORY = "memory"
#: The environment-supplied reward: charge and length windows under the same
#: constraint, every term exactly computable and no learned model anywhere.
REWARD_ENVIRONMENT = "environment"
#: Both, in equal weight, to ask whether an orthogonal bias can be installed
#: without destroying the memory-derived score.
REWARD_BOTH = "memory_and_environment"

#: The same environment objective at a harder charge window: the positive
#: control. It exists because a small ``k*`` on the declared target admits two
#: readings that predict the same number -- the method installs nothing, or the
#: target was base-reachable all along -- and only a target selection cannot
#: reach separates them.
REWARD_ENVIRONMENT_HARD = "environment_hard"

REWARD_NAMES: tuple[str, ...] = (REWARD_MEMORY, REWARD_ENVIRONMENT,
                                 REWARD_BOTH, REWARD_ENVIRONMENT_HARD)

#: Which rewards need a folded structure. The others are computable from the
#: residue string alone, which is what makes the environment arm a miniature
#: simulated environment rather than a second proxy.
REWARD_NEEDS_STRUCTURE: Mapping[str, bool] = {
    REWARD_MEMORY: True,
    REWARD_ENVIRONMENT: False,
    REWARD_BOTH: True,
    REWARD_ENVIRONMENT_HARD: False,
}

#: Rewards that cannot be computed until a declared constant is chosen. Named
#: here so that a caller can tell "not yet declared" from "not applicable".
REWARD_NEEDS_DECLARED_WINDOW: Mapping[str, bool] = {
    REWARD_MEMORY: False,
    REWARD_ENVIRONMENT: False,
    REWARD_BOTH: False,
    REWARD_ENVIRONMENT_HARD: True,
}


def hard_window_declared() -> bool:
    """Whether the positive control's target has been chosen yet."""

    return HARD_CHARGE_WINDOW is not None


def require_hard_window() -> tuple[float, float]:
    """The positive control's window, refusing while it is undeclared."""

    if HARD_CHARGE_WINDOW is None:
        raise ValueError(
            "the positive control's charge window has not been chosen. It is read "
            "off the base pool's measured conjunction rates by one declared rule "
            "and written into sequence_reward before the arm trains; refusing to "
            "pay or score a reward whose target does not exist yet"
        )
    return HARD_CHARGE_WINDOW

REWARD_PROVENANCE: Mapping[str, str] = {
    REWARD_MEMORY: (
        "ESMFold2 mean CA-pLDDT, a learned predictor trained on the same "
        "evolutionary record as the policy, under the closed-form "
        "low-complexity constraint declared here at the 0.25 limit the published "
        "objective also uses"
    ),
    REWARD_ENVIRONMENT: (
        "net charge at pH 7.0 and residue count, each inside a declared window, "
        "under the same low-complexity constraint. Exactly computable; no "
        "learned model enters it at any point"
    ),
    REWARD_BOTH: (
        "the equal-weight mean of the memory-derived and environment-supplied "
        "objectives, under the one shared constraint"
    ),
    REWARD_ENVIRONMENT_HARD: (
        "the same arithmetic as the environment reward at a charge window a "
        "rejection sampler of the declared budget cannot reach from the base "
        "pool. Identical in provenance to that reward and harder only in where "
        "its window sits, so a difference between the two arms is a difference in "
        "target difficulty and not in instrument"
    ),
}


def complexity_term(stats: SequenceStatistics) -> float:
    """The declared constraint, as a term: one below the limit, decaying above it.

    Shared by every reward, because a constraint at this limit is part of the
    published objective and because it is what forbids the degenerate solutions
    -- poly-lysine reaches any charge window -- that the environment reward would
    otherwise admit. Being shared, it cannot explain a difference between the
    arms, which is the only property of it this experiment depends on; the mask
    itself is this project's own and is not interchangeable with the published
    pipeline's or with the objective-alignment campaign's screen.
    """

    return plateau(
        stats.low_complexity_fraction,
        low=0.0,
        high=LOW_COMPLEXITY_LIMIT,
        half_width=1.0 - LOW_COMPLEXITY_LIMIT,
    )


def charge_term(stats: SequenceStatistics,
                window: tuple[float, float] = CHARGE_WINDOW) -> float:
    """The charge term, at the declared window or at another one.

    The window is a parameter because the base pool's *reachability* at harder
    targets is a question the closed-form statistics can answer for nothing: the
    same measured net charge, read against a window further into the base's tail,
    says how much selection pressure a harder target would need. The default is
    the declared window and is the only one any reward is paid at.
    """

    low, high = window
    return plateau(stats.net_charge, low=low, high=high, half_width=CHARGE_HALF_WIDTH)


def length_term(stats: SequenceStatistics) -> float:
    low, high = LENGTH_WINDOW
    return plateau(float(stats.length), low=float(low), high=float(high),
                   half_width=LENGTH_HALF_WIDTH)


def plddt_term(mean_ca_plddt: float) -> float:
    """Mean CA-pLDDT on the 0-100 scale, as a share of that scale."""

    if not math.isfinite(mean_ca_plddt):
        raise ValueError("a pLDDT term needs a finite mean CA-pLDDT")
    if not 0.0 <= mean_ca_plddt <= PLDDT_SCALE:
        raise ValueError(
            f"mean CA-pLDDT {mean_ca_plddt} is outside the 0-{PLDDT_SCALE:g} scale "
            "this term divides by"
        )
    return float(mean_ca_plddt / PLDDT_SCALE)


def terms(stats: SequenceStatistics, *, mean_ca_plddt: float | None) -> dict[str, float]:
    """Every term of every reward for one completion, measured once.

    ``mean_ca_plddt`` is ``None`` only for a completion no structure was measured
    for, and then the memory-derived term is absent rather than zero: a missing
    measurement is not a low score, and :func:`reward` refuses to pay one.
    """

    out = {
        "complexity": complexity_term(stats),
        "charge": charge_term(stats),
        "length": length_term(stats),
    }
    if mean_ca_plddt is not None:
        out["plddt"] = plddt_term(mean_ca_plddt)
    return out


#: An empty completion is a measured outcome of the sampler, not a missing
#: measurement: the policy emitted a stop before any residue. It is retained in
#: the denominator at this reward, which is the lane's existing rule for an
#: empty continuation.
EMPTY_COMPLETION_REWARD = 0.0


def environment_objective(stats: SequenceStatistics, *,
                          charge_window: tuple[float, float] = CHARGE_WINDOW) -> float:
    """The environment arm's objective term: the mean of its charge and length terms."""

    return 0.5 * (charge_term(stats, charge_window) + length_term(stats))


def environment_reward_at(stats: SequenceStatistics, *,
                          charge_window: tuple[float, float]) -> float:
    """The environment reward at an arbitrary charge window, constraint included.

    Used only by the declared reachability ladder, which asks what the *base* pool
    scores at harder targets. No policy is ever paid at a window other than the
    declared one, and a reward read at another window is labelled as a diagnostic
    wherever it appears.
    """

    if stats.length == 0:
        return EMPTY_COMPLETION_REWARD
    return float(complexity_term(stats)
                 * environment_objective(stats, charge_window=charge_window))


def reward(name: str, stats: SequenceStatistics, *, mean_ca_plddt: float | None = None) -> float:
    """One of the three declared rewards, in ``[0, 1]``.

    Every reward is the shared constraint term times the mean of that reward's
    own objective terms, so all three reach one exactly at their conjunction and
    none of them can exceed one. A reward that needs a structure and was given
    none is refused rather than scored: a policy paid zero for an unmeasured
    completion learns to produce unmeasurable completions.
    """

    if name not in REWARD_NAMES:
        raise KeyError(f"unknown reward {name!r}; declared: {list(REWARD_NAMES)}")
    if stats.length == 0:
        return EMPTY_COMPLETION_REWARD
    if REWARD_NEEDS_STRUCTURE[name] and not within_structure_support(stats):
        # Outside the declared structural support there is no structure to be
        # confident about, so this is a measured outcome of the sampler and not a
        # missing measurement. See STRUCTURE_SUPPORT.
        return EMPTY_COMPLETION_REWARD
    if REWARD_NEEDS_STRUCTURE[name] and mean_ca_plddt is None:
        raise ValueError(
            f"reward {name!r} needs a mean CA-pLDDT and was given none for a "
            f"{stats.length}-residue completion, which is inside the declared "
            f"structural support {STRUCTURE_SUPPORT}; refusing to score it as zero"
        )
    if name == REWARD_ENVIRONMENT_HARD:
        return environment_reward_at(stats, charge_window=require_hard_window())
    scored = terms(stats, mean_ca_plddt=mean_ca_plddt)
    environment = 0.5 * (scored["charge"] + scored["length"])
    if name == REWARD_ENVIRONMENT:
        objective = environment
    elif name == REWARD_MEMORY:
        objective = scored["plddt"]
    else:
        objective = 0.5 * (scored["plddt"] + environment)
    return float(scored["complexity"] * objective)


def rewards(stats: SequenceStatistics, *, mean_ca_plddt: float | None = None) -> dict[str, float]:
    """Every declared reward for one completion, where each can be computed.

    Every pool is read under every reward, not only the one it was trained
    against: that is how the experiment asks whether an environment-supplied
    bias cost the memory-derived score, and it costs nothing once the terms are
    measured.
    """

    out: dict[str, float] = {}
    for name in REWARD_NAMES:
        if (REWARD_NEEDS_STRUCTURE[name] and mean_ca_plddt is None
                and stats.length > 0 and within_structure_support(stats)):
            continue
        if REWARD_NEEDS_DECLARED_WINDOW[name] and not hard_window_declared():
            continue
        out[name] = reward(name, stats, mean_ca_plddt=mean_ca_plddt)
    return out


def meets_declared_objective(stats: SequenceStatistics, *, mean_ca_plddt: float | None = None,
                             plddt_criterion: float = 70.0) -> dict[str, bool]:
    """The declared windows as the hits a rate can be taken over.

    Reported beside the shaped rewards because a mean reward cannot say whether
    anything reached the target, and the experiment's degeneracy diagnostics ask
    for exactly these two rates: the charge-window hit rate for the arm trained
    on structure, and the predicted-confidence rate for the arm trained on
    charge.
    """

    low, high = CHARGE_WINDOW
    length_low, length_high = LENGTH_WINDOW
    out = {
        "charge_window": low <= stats.net_charge <= high,
        "length_window": length_low <= stats.length <= length_high,
        "low_complexity_ok": stats.low_complexity_fraction < LOW_COMPLEXITY_LIMIT,
    }
    out["environment_conjunction"] = all(out.values())
    if hard_window_declared():
        hard_low, hard_high = require_hard_window()
        out["charge_window_hard"] = hard_low <= stats.net_charge <= hard_high
        out["environment_hard_conjunction"] = (out["charge_window_hard"]
                                               and out["length_window"]
                                               and out["low_complexity_ok"])
    if mean_ca_plddt is not None:
        out["plddt_criterion"] = mean_ca_plddt >= plddt_criterion
        out["memory_conjunction"] = out["plddt_criterion"] and out["low_complexity_ok"]
    return out
