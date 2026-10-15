"""Guided autoregressive decoding with an external family Potts model, decomposed.

What this module is for
=======================

E20 asks whether an autoregressive protein decoder is missing *pairwise coupling*
information specifically. The intervention adds a per-residue bias to the model's
own next-token logits at every step, and the whole experiment turns on two
properties of that bias: the decomposition into a fields part and a coupling part
must be exact, and the arms must be identical apart from the bias term. So there
is exactly one bias hook here, exactly one decoding loop, and every arm --
including the native one -- goes through both.

One biasing function, two strengths
===================================

The bias at position ``t`` with committed prefix ``a_{<t}`` is

    B(t, a_{<t})[a] = lambda_fields * u[t, a]
                      + lambda_coupling * sum_{s<t} C[t, s, a, a_s]

with ``u`` of shape ``(L, 20)`` and ``C`` of shape ``(L, L, 20, 20)`` or absent.
An arm is a named choice of where ``u`` and ``C`` come from, and a dose is a
:class:`GuidanceStrength` pair; there is no per-arm code path.

===========================  ====================  =======================
arm (code)                   ``u``                 ``C``
===========================  ====================  =======================
``native`` (N)               absent                absent
``potts_fields`` (M)         ``h``                 absent
``coupling`` (C)             absent                ``J``
``full`` (F)                 ``h``                 ``J``
``profile`` (Pr)             ``log_profile``       absent
``permuted_coupling`` (Cx)   absent                ``J`` column-permuted
``cross_domain_coupling``    absent                donor ``J[:L, :L]``
(Cd)
===========================  ====================  =======================

**The factorial lives entirely inside one model's own decomposition.** ``N``,
``M``, ``C`` and ``F`` are the four corners of the Potts model's own
fields-by-couplings factorial: ``M`` carries Potts ``h`` and *not* the profile, so
``F - M`` and ``C - N`` are both the coupling component of the declared gauge's
decomposition and ``F - C`` and ``M - N`` are both its field component; these are
identities of additivity, and what each component *means* is conditional on the
gauge (see the gauge note in ``family_potts``). An ``M`` that carried ``log f`` would have made
``F - M`` the effect of adding couplings *and* swapping profile fields for Potts
fields -- two changes at once, and no isolated coupling component. At unit
strengths the ``F`` *bias* equals the prefix-truncated Potts conditional
``log P_t(x_t = a | x_<t)`` up to a per-position additive constant, where ``P_t``
is the Potts model restricted to the first ``t`` positions; ``P_t`` is not the
full-joint conditional ``P_L(x_t | x_<t)``, which carries an extra
residue-dependent future free-energy term. The bias enters as a logit bias, so
the *arm* samples the product of experts ``p_LM(a | .) * P_t(a | x_<t)``, not the
prefix-truncated conditional itself.

``Pr`` is the conventional profile-guidance baseline, ``lambda_fields * log
f_t(a)`` from the frozen pseudocount-0.5 profile with no couplings. It sits
**outside** the factorial and is read as a pairwise comparison, because "does this
beat established practice" is a different question from "which information
component is the model missing". ``P`` (prefix-chain or Gibbs Potts sampling) and ``H``
(held-out natural rows) involve no language model and are produced elsewhere.

**Why two strengths and not one.** The fields terms are order one to three nats,
while the coupling sum runs over up to ``L - 1`` prefix positions and is smaller
per unit strength: the ratio of fields to coupling spread, as measured in
``results/R6/e20_development_20261013/e20devc_records_cpu/component_scale.json``, is
3.1-7.1 on the wild-type basis and 3.1-32.9 (median 10.6) across sampled pools. Under one shared scalar the fields-on and
couplings-on cells would be interventions of very different effective strength,
so a difference between them would partly measure scale rather than information
content. The two components therefore carry independent scalars, and the arms are
corners and edges of one two-dimensional dose grid.
:func:`component_scale` measures that dynamic-range asymmetry directly, so it is
visible in the artefact rather than inferred from the tuned strengths. A strength
whose component an arm does not carry is refused rather than ignored: two grid
points that differ only in an inert scalar would otherwise produce byte-identical
pools under different labels.

**Why the coupling sum is not normalised by prefix length.** The raw sum is
the coupling component of the prefix-truncated Potts log-conditional (it omits
``h_t`` and the normaliser), and ``full`` adds ``h_t`` to it; this is what makes
``full`` the conventional DCA-guidance baseline rather than a variant of this
project's own invention. Dividing by ``t`` would break that correspondence. The consequence --
coupling influence grows along the sequence while the fields term stays flat -- is
a genuine property of left-to-right Potts conditioning, applies identically to
``coupling`` and ``full``, and does not disturb the factorial. Do not "fix" it.

Three facts the experiment rests on are structural rather than asserted: ``full``
equals the ``potts_fields`` bias plus the ``coupling`` bias, because the three arms
share one ``u`` and one ``C``;
``coupling`` at ``t = 0`` is exactly zero because the sum is empty; and with both
strengths at zero every arm adds a literal ``0.0`` to the same logits, so all six
draw the identical sequence from one seed. That the shared sequence is the one
*native sampling* draws, rather than merely a sequence all six agree on, is
checked against :func:`native_reference_draw`, which reimplements native sampling
without any of this module's layout or bias machinery. An arm that lacks a component
contributes exactly zero from it, not a small number, because the component is
absent rather than scaled down. :func:`build_arm_bias` additionally checks the
three target-model arms against
:func:`~src.capability.context.family_potts.guidance_biases` -- the family
module's own declaration of the same three vectors -- so a gauge, transposition
or alphabet drift cannot pass silently.

``permuted_coupling`` permutes the *column assignment*:
``C[t, s] = J[p(t), p(s)]`` for a permutation ``p`` fixed by
:func:`permutation_for` as a function of (domain, length, seed). That keeps the
multiset of coupling values and the coupling graph exactly, and breaks only the
pairing, so it is magnitude-matched by construction rather than by rescaling.

Column alignment, and why the layout is derived rather than written down
=======================================================================

Generated residue ``i`` must be column ``i`` of the family model and residue
``i`` of the wild type, so decoding emits exactly ``L`` residues and stops. The
hard part is that a model's native stream is not a bare residue string:
ProtGPT2's is FASTA hard-wrapped at 60 columns, so its token stream carries
newline characters that are not residues, while ProGen2's is a direction marker
followed by one token per residue.

Rather than branch on the model, :func:`native_layout` *derives* the character
layout from the arm's own declared rendering
(:meth:`~src.capability.core.arms.Cohort.input_strings`, through
:func:`~src.capability.context.context_homologue.render_records`) by rendering
two probe sequences that differ at every residue: characters that differ between
the two renderings are residue slots, characters that agree are structural
literals. A single-residue tokeniser is then the special case of a layout with no
literals and spans of length one; nothing in this module knows which model it is
running.

A candidate token is admissible at character offset ``c`` only if its decoded
text matches the layout there -- residues in residue slots, the declared literal
in a literal slot -- and does not run past the end. That one rule delivers every
suppression the protocol asks for: end-of-sequence and non-canonical tokens never
match a residue slot, tokens whose residues would overshoot the budget cannot
fit, and the structural newline stays sampleable exactly where the native
rendering puts it. It is applied identically in every arm, so it is part of the
shared protocol and not part of the treatment. An offset at which no token is
admissible is an explicit refusal (:class:`EmptyCandidateSet`), never a truncated
token and never a silently shortened sequence.

The token-span bias
===================

The bias is defined on residues and ProtGPT2 samples multi-residue tokens, so the
bias on a candidate token is the sum of the per-residue biases of the residues it
would emit, each evaluated against the committed prefix *plus the residues
emitted earlier inside the same token*. Written out for a token whose residues
are ``r_0 .. r_{k-1}`` at residue index ``n``, before the strengths are applied:

    fields   = sum_j u[n+j, r_j]
    coupling = sum_j [ sum_{s<n} C[n+j, s, r_j, a_s]
                       + sum_{i<j} C[n+j, n+i, r_j, r_i] ]

The prefix terms are evaluated once per step as ``(K, 20)`` matrices gathered by
each token's residue codes; the within-token term depends only on ``(n, token)``
and is cached per ``n``. :func:`sequence_component_sums` recomputes both
components straight from the finished sequence, and :func:`generate_cell` refuses
any sequence whose accumulated applied bias disagrees with them, so the span
arithmetic is checked on every draw rather than only in tests.

Records
=======

:func:`generate_cell` emits one flat diagnostics record per sequence, carrying
everything this stage knows: both strengths, both component scores, the token ids
and the per-step entropies. It is deliberately *not* the measurement side's input
format -- ``e20_generated_sequence_v1``
(:mod:`src.capability.generation.fold_recapitulation`) admits exactly twelve keys
and refuses an unknown one -- so the runner projects these records onto that
contract and writes the two files side by side, joined on ``id``. The names the
contract shares are spelled its way (``id, domain, model, seed, sequence,
native_loglik, guidance_score, mean_step_entropy, attempt_budget``) and the two
it cannot hold unabbreviated are ``arm_code`` and ``lambda_nominal``, under
:data:`ARM_CODES` and :func:`nominal_lambda`.

The native score is the caller's, injected as ``score_fn``: this module declares
no second scoring contract. ``guidance_score`` is the arm's bias accumulated
along the sequence at *unit* strengths, so it is the same biological quantity in
every arm and at every dose and is what a rerank-only comparison needs;
``guidance_bias_applied`` is the dose-weighted sum that actually entered the
logits. For ``native`` the three component scores are ``None`` rather than zero:
the arm has no guidance quantity, and the contract admits a null there.

Numerics and seeds
==================

Sampling runs on the caller's loaded handle, which this programme loads in
float32: bf16 logits tie heavily and the tie-breaking disagreement is large
enough to move a draw. Bias arithmetic is float64 numpy, applied to float32
logits. Temperature is 1.0 over the full vocabulary with no top-k and no top-p;
the recipe is :data:`SAMPLING_RECIPE` and a caller cannot vary it.

Each sequence draws from its own ``torch.Generator`` seeded ``stream_seed +
sequence_index``, and one ``multinomial`` call is made per row per step, so a
sequence is a function of its seed and the step distributions alone and does not
depend on which batch it was drawn in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
import hashlib
import math

import numpy as np
import torch

from ..context.context_homologue import render_records
from ..context.family_potts import FamilyModel, encode, guidance_biases
from ..core.amino_acids import AA20
from ..generation.unconditional_generation import require_admitted
from .guidance_strength import StrengthMeter

__all__ = [
    "ARM_CODES",
    "ARM_DECLARATIONS",
    "ATTEMPTS_PER_SEQUENCE",
    "ArmBias",
    "CellIdentity",
    "EmptyCandidateSet",
    "GUIDANCE_ARMS",
    "GuidanceStrength",
    "GuidedDecoder",
    "NATIVE_REFERENCE_DECLARATION",
    "NativeLayout",
    "RESIDUE_SLOT",
    "SAMPLING_RECIPE",
    "SEQUENCE_SCHEMA",
    "TokenSpanTable",
    "ARM_COMPONENTS",
    "bias_vector",
    "build_arm_bias",
    "component_bias_vectors",
    "component_scale",
    "generate_cell",
    "generation_point",
    "native_layout",
    "native_reference_draw",
    "nominal_lambda",
    "permutation_for",
    "select_cross_domain_donor",
    "sequence_bias_sum",
    "sequence_component_sums",
    "token_span_table",
]

SEQUENCE_SCHEMA = "coupling_guided_sequence_v1"

#: The seven language-model arms, in report order. ``native``, ``potts_fields``,
#: ``coupling`` and ``full`` are the four corners of the Potts model's own
#: factorial and ``coupling`` is the method under test; ``profile`` is the
#: conventional practice baseline outside it; ``permuted_coupling`` is the primary
#: matched incorrect-constraint control and ``cross_domain_coupling`` the
#: secondary wrong-biology variant.
GUIDANCE_ARMS: tuple[str, ...] = (
    "native",
    "potts_fields",
    "coupling",
    "full",
    "profile",
    "permuted_coupling",
    "cross_domain_coupling",
)

#: The four corners of the factorial, in the order the contrasts read them. The
#: two coupling main effects are ``full - potts_fields`` and ``coupling - native``,
#: the two field main effects ``full - coupling`` and ``potts_fields - native``.
FACTORIAL_ARMS: tuple[str, ...] = ("native", "potts_fields", "coupling", "full")

#: The short arm code every E20 record carries, bound to the measurement side's
#: ``e20_generated_sequence_v1`` contract.
#:
#: The two wrong-coupling controls keep distinct codes: ``Cx`` is the
#: permuted-column control, which is magnitude-matched by construction and is the
#: primary one, and ``Cd`` is the secondary cross-domain variant. The E20 design
#: document's table once wrote ``Cx`` for the cross-domain arm; one code cannot
#: mean two interventions, so they are two codes rather than one arm dropped.
#: ``P`` and ``H`` need no language-model generation and are produced elsewhere.
#: Every record also carries the unabbreviated ``guidance_arm``, so a reader never
#: has to trust the abbreviation.
ARM_CODES: dict[str, str] = {
    "native": "N",
    "potts_fields": "M",
    "coupling": "C",
    "full": "F",
    "profile": "Pr",
    "permuted_coupling": "Cx",
    "cross_domain_coupling": "Cd",
}

#: Attempts made per emitted sequence. The layout rule makes a length-``L``
#: emission certain, so there is no rejection loop and no attempt budget to match
#: across arms; the contract's ``attempt_budget`` is therefore exactly one.
ATTEMPTS_PER_SEQUENCE = 1

#: Where each arm's two components come from, as the artefacts record it.
#: ``fields`` is one of ``none``, ``log_profile``, ``potts_fields``; ``coupling``
#: is one of ``none``, ``target``, ``permuted_columns``, ``donor_domain``.
ARM_COMPONENTS: dict[str, dict[str, str]] = {
    "native": {"fields": "none", "coupling": "none"},
    "potts_fields": {"fields": "potts_fields", "coupling": "none"},
    "coupling": {"fields": "none", "coupling": "target"},
    "full": {"fields": "potts_fields", "coupling": "target"},
    "profile": {"fields": "log_profile", "coupling": "none"},
    "permuted_coupling": {"fields": "none", "coupling": "permuted_columns"},
    "cross_domain_coupling": {"fields": "none", "coupling": "donor_domain"},
}

#: What each arm supplies, as the artefacts record it.
ARM_DECLARATIONS: dict[str, str] = {
    "native": "no bias; the model's own next-token distribution under the shared layout rule",
    "potts_fields": (
        "lambda_fields * h[t,a]; the Potts model's own fields only, which is what makes the "
        "fields-by-couplings factorial internally consistent"
    ),
    "coupling": (
        "lambda_coupling * sum_{s<t} J[t,s,a,a_s]; couplings only, no fields (the method under test)"
    ),
    "full": (
        "lambda_fields * h[t,a] + lambda_coupling * sum_{s<t} J[t,s,a,a_s]; the conventional "
        "DCA-guidance baseline; at unit strengths its bias equals the prefix-truncated Potts "
        "conditional up to a per-position constant, and the arm samples that times the LM (a product of experts)"
    ),
    "profile": (
        "lambda_fields * log f_t(a) from the frozen pseudocount-0.5 profile; conventional "
        "profile guidance, read as a pairwise comparison outside the factorial"
    ),
    "permuted_coupling": (
        "lambda_coupling * coupling bias of the target's own J under a seeded column permutation; "
        "identical coupling-value multiset and coupling graph, wrong pairing"
    ),
    "cross_domain_coupling": (
        "lambda_coupling * coupling bias of another qualified domain's J restricted to its first "
        "L columns"
    ),
}

#: The one declared recipe. Full vocabulary, no truncation, unit temperature.
SAMPLING_RECIPE: dict[str, Any] = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}

#: Layout code for a character position that holds one canonical residue.
RESIDUE_SLOT = -1
_PAD = -2

_RESIDUE_CODE = {residue: index for index, residue in enumerate(AA20)}


class EmptyCandidateSet(RuntimeError):
    """No vocabulary token matches the native layout at this offset."""


@dataclass(frozen=True)
class GuidanceStrength:
    """The dose: one scalar per bias component, in natural-log units."""

    fields: float
    coupling: float

    def __post_init__(self) -> None:
        for name in ("fields", "coupling"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"lambda_{name} must be a number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"lambda_{name} must be finite and non-negative, got {value!r}")
        object.__setattr__(self, "fields", float(self.fields))
        object.__setattr__(self, "coupling", float(self.coupling))

    @property
    def is_zero(self) -> bool:
        return self.fields == 0.0 and self.coupling == 0.0


# --------------------------------------------------------------------- the bias


@dataclass(frozen=True)
class ArmBias:
    """One arm's two unscaled bias components, plus where they came from.

    ``u`` is ``(L, 20)`` or ``None`` and ``C`` is ``(L, L, 20, 20)`` or ``None``.
    Both are float64 in natural-log units and in :data:`AA20` column order, which
    is the order the family reference declares and the only order this module
    indexes. A component that an arm does not carry is ``None`` rather than an
    array of zeros, so it contributes exactly zero and no dose can revive it.
    """

    arm: str
    domain: str
    length: int
    u: np.ndarray | None
    C: np.ndarray | None
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if self.arm not in GUIDANCE_ARMS:
            raise ValueError(f"unknown guidance arm {self.arm!r}; arms are {list(GUIDANCE_ARMS)}")
        if self.u is not None and self.u.shape != (self.length, 20):
            raise ValueError(f"{self.arm}: u has shape {self.u.shape}, expected {(self.length, 20)}")
        if self.C is not None and self.C.shape != (self.length, self.length, 20, 20):
            raise ValueError(
                f"{self.arm}: C has shape {self.C.shape}, expected "
                f"{(self.length, self.length, 20, 20)}"
            )
        for name, array in (("u", self.u), ("C", self.C)):
            if array is not None and not np.isfinite(array).all():
                raise ValueError(f"{self.arm}: {name} is not finite")

    @property
    def has_fields(self) -> bool:
        return self.u is not None

    @property
    def has_coupling(self) -> bool:
        return self.C is not None

    def require_strength(self, strength: GuidanceStrength) -> None:
        """Refuse a dose on a component this arm does not carry.

        Ignoring it would let two grid points that differ only in an inert scalar
        produce byte-identical pools under different labels.
        """

        if strength.fields and not self.has_fields:
            raise ValueError(
                f"{self.arm} carries no fields component, so lambda_fields must be 0, got "
                f"{strength.fields!r}"
            )
        if strength.coupling and not self.has_coupling:
            raise ValueError(
                f"{self.arm} carries no coupling component, so lambda_coupling must be 0, got "
                f"{strength.coupling!r}"
            )


def nominal_lambda(bias: ArmBias, strength: GuidanceStrength) -> float:
    """The single ``lambda`` the measurement contract carries, as a declared rule.

    The dose of the arm's defining component: the coupling strength for every arm
    that carries couplings, the fields strength for an arm that carries only
    fields, and zero for ``native``. The pair is always recorded beside it, so
    nothing is lost by the abbreviation.
    """

    if bias.has_coupling:
        return strength.coupling
    if bias.has_fields:
        return strength.fields
    return 0.0


def bias_vector(bias: ArmBias, position: int, prefix_codes: Sequence[int]) -> np.ndarray:
    """``B(position, prefix)`` at unit strengths: the per-residue reference form.

    The sampling path evaluates whole token spans at once and applies the dose;
    this is the same quantity written one residue at a time with both strengths
    set to one, and is what the span arithmetic is checked against.
    """

    fields, coupling = component_bias_vectors(bias, position, prefix_codes)
    return fields + coupling


def component_bias_vectors(
    bias: ArmBias, position: int, prefix_codes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """The fields and coupling bias vectors at ``position``, separately and unscaled."""

    if not 0 <= position < bias.length:
        raise ValueError(f"position {position} is outside 0..{bias.length - 1}")
    codes = np.asarray(prefix_codes, dtype=np.int64)
    if codes.size != position:
        raise ValueError(f"prefix of {codes.size} residues cannot precede position {position}")
    fields = np.zeros(20, dtype=np.float64) if bias.u is None else bias.u[position].astype(np.float64)
    coupling = np.zeros(20, dtype=np.float64)
    if bias.C is not None and position:
        coupling = bias.C[position][np.arange(position), :, codes].sum(axis=0)
    return fields, coupling


def sequence_component_sums(bias: ArmBias, sequence: str) -> tuple[float, float]:
    """The two unscaled components accumulated along a finished sequence.

    ``(sum_t u[t, a_t], sum_t sum_{s<t} C[t, s, a_t, a_s])``. For the ``full`` arm
    their sum is the family model's own ``potts_score``.
    """

    codes = encode(sequence)
    if codes.size != bias.length:
        raise ValueError(f"sequence of {codes.size} residues is not length {bias.length}")
    index = np.arange(bias.length)
    fields = 0.0 if bias.u is None else float(bias.u[index, codes].sum())
    coupling = 0.0
    if bias.C is not None:
        pairs = bias.C[index[:, None], index[None, :], codes[:, None], codes[None, :]]
        coupling = float(np.tril(pairs, k=-1).sum())
    return fields, coupling


def sequence_bias_sum(bias: ArmBias, sequence: str) -> float:
    """The arm's unit-strength bias accumulated along a finished sequence.

    This is the per-arm reranking score the protocol calls the guidance score: the
    arm's own biological quantity, independent of the dose it was sampled at.
    """

    fields, coupling = sequence_component_sums(bias, sequence)
    return fields + coupling


#: What :func:`component_scale` measures, as the artefact records it.
COMPONENT_SCALE_DECLARATION = (
    "Standard deviation across the twenty canonical residues of one bias component's "
    "per-position vector, summarised over positions and sequences. It is the component's "
    "dynamic range in natural-log units at the scale a logit bias competes with, so the "
    "fields/coupling asymmetry is readable from the artefact rather than inferred from the "
    "tuned strengths, and a selected strength that cannot move a draw is visible as such. "
    "potts_fields and profile do not depend on the prefix, so their numbers are a property of "
    "the family reference alone and are identical on any sequence basis; coupling does, and it "
    "grows along the sequence because the prefix sum is deliberately not length-normalised."
)

#: The components :func:`component_scale` reports, and whether each depends on the prefix.
COMPONENT_SCALE_PREFIX_DEPENDENT: dict[str, bool] = {
    "potts_fields": False, "profile": False, "coupling": True,
}


def _coupling_bias_rows(couplings: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """``(L, 20)`` of ``sum_{s<t} C[t, s, a, a_s]`` for one full-length sequence."""

    length = int(codes.size)
    index = np.arange(length)
    blocks = couplings[index[:, None], index[None, :], :, codes[None, :]]  # (L, L, 20)
    below = np.tril(np.ones((length, length), dtype=np.float64), -1)[:, :, None]
    return (blocks * below).sum(axis=1)


def _spread(values: np.ndarray) -> dict[str, Any]:
    spread = values.std(axis=1, ddof=0)
    return {
        "steps": int(spread.size),
        "mean": float(spread.mean()),
        "median": float(np.median(spread)),
        "p90": float(np.percentile(spread, 90)),
        "max": float(spread.max()),
        "zero_steps": int((spread == 0.0).sum()),
    }


def component_scale(model: FamilyModel, sequences: Sequence[str]) -> dict[str, Any]:
    """The per-step dynamic range of each bias component on one set of sequences.

    Reported for the target reference's own ``h``, ``J`` and ``log f`` -- not for a
    substituted or permuted coupling tensor -- so the numbers describe the family
    reference and the sequence basis, and are comparable across arms of one
    domain. ``per_position_mean`` is the same statistic resolved by column, which
    is where the coupling term's growth along the sequence shows.
    """

    if not sequences:
        raise ValueError("a component-scale summary needs at least one sequence")
    stacked: dict[str, list[np.ndarray]] = {name: [] for name in COMPONENT_SCALE_PREFIX_DEPENDENT}
    for sequence in sequences:
        codes = encode(sequence)
        if codes.size != model.length:
            raise ValueError(
                f"{model.domain}: a component-scale sequence has {codes.size} residues, not "
                f"{model.length}"
            )
        stacked["potts_fields"].append(model.h.astype(np.float64))
        stacked["profile"].append(model.log_profile.astype(np.float64))
        stacked["coupling"].append(_coupling_bias_rows(model.J.astype(np.float64), codes))
    components = {}
    for name, rows in stacked.items():
        matrix = np.concatenate(rows, axis=0)
        by_position = np.stack([row.std(axis=1, ddof=0) for row in rows], axis=0).mean(axis=0)
        components[name] = {
            "prefix_dependent": COMPONENT_SCALE_PREFIX_DEPENDENT[name],
            **_spread(matrix),
            "per_position_mean": [float(value) for value in by_position],
        }
    return {
        "declaration": COMPONENT_SCALE_DECLARATION,
        "domain": model.domain,
        "alphabet": AA20,
        "units": "natural log",
        "sequences": len(sequences),
        "components": components,
    }


def permutation_for(domain: str, length: int, seed: int) -> np.ndarray:
    """The declared column permutation of ``permuted_coupling``.

    A deterministic function of ``(domain, length, seed)`` and of nothing else, so
    the control is reproducible from the recorded seed alone. The identity is
    refused rather than used: it would silently turn the control into the
    treatment.
    """

    digest = hashlib.sha256(f"permuted_coupling|{domain}|{length}|{seed}".encode()).digest()
    permutation = np.random.default_rng(int.from_bytes(digest[:8], "big")).permutation(length)
    if np.array_equal(permutation, np.arange(length)):
        raise ValueError(
            f"{domain}: the column permutation drawn from seed {seed} is the identity, which "
            "would make permuted_coupling identical to coupling"
        )
    return permutation


def select_cross_domain_donor(target: str, lengths: Mapping[str, int]) -> str:
    """The declared donor: smallest ``L' - L`` with ``L' >= L``, ties by ascending domain id.

    ``lengths`` must already be restricted to *qualified* domains. No admissible
    donor is a refusal, not a fallback onto a shorter donor and not a padded one.
    """

    if target not in lengths:
        raise ValueError(f"donor selection needs the target {target!r} among the qualified domains")
    length = int(lengths[target])
    admissible = sorted(
        (int(other) - length, name) for name, other in lengths.items()
        if name != target and int(other) >= length
    )
    if not admissible:
        raise ValueError(
            f"{target}: cross_domain_coupling has no admissible donor; no other qualified domain "
            f"has at least {length} columns (qualified lengths {dict(sorted(lengths.items()))})"
        )
    return admissible[0][1]


def _require_target_alphabet(model: FamilyModel) -> None:
    if model.alphabet != AA20:
        raise ValueError(
            f"{model.domain}: family reference declares alphabet {model.alphabet!r}, but every "
            f"bias vector here is indexed in {AA20!r} order; a permuted ordering would permute "
            "the whole bias"
        )


def _require_qualified(model: FamilyModel, *, role: str) -> None:
    if not model.qualified:
        raise ValueError(
            f"{model.domain} is not a qualified family reference "
            f"({list(model.qualification_failures)}) and cannot be this cell's {role}; E20 does "
            "not run on unqualified references"
        )


def build_arm_bias(
    arm: str,
    model: FamilyModel,
    *,
    permutation_seed: int | None = None,
    donor: FamilyModel | None = None,
) -> ArmBias:
    """The two unscaled bias components of one arm on one target domain.

    ``model`` must be the target domain's own qualified reference.
    ``permutation_seed`` is required by ``permuted_coupling`` and ``donor`` by
    ``cross_domain_coupling``; supplying either to an arm that does not use it is
    refused, so an artefact that records a seed or a donor is an artefact whose
    arm used it.
    """

    if arm not in GUIDANCE_ARMS:
        raise ValueError(f"unknown guidance arm {arm!r}; arms are {list(GUIDANCE_ARMS)}")
    _require_target_alphabet(model)
    _require_qualified(model, role="target")
    length = model.length
    provenance: dict[str, Any] = {
        "arm": arm,
        "code": ARM_CODES[arm],
        "declaration": ARM_DECLARATIONS[arm],
        "components": dict(ARM_COMPONENTS[arm]),
        "domain": model.domain,
        "length": length,
        "alphabet": AA20,
        "units": "natural log",
        "gauge": model.sidecar.get("gauge_convention"),
        "coupling_normalisation": (
            "none; the raw prefix sum is the coupling component of the prefix-truncated Potts "
            "log-conditional (it omits h_t and the normaliser), which is what makes the full arm "
            "the conventional DCA-guidance baseline"
        ),
    }
    if arm == "permuted_coupling":
        if permutation_seed is None:
            raise ValueError("permuted_coupling requires a declared permutation seed")
    elif permutation_seed is not None:
        raise ValueError(f"{arm} takes no permutation seed")
    if arm == "cross_domain_coupling":
        if donor is None:
            raise ValueError("cross_domain_coupling requires a donor family reference")
    elif donor is not None:
        raise ValueError(f"{arm} takes no donor")

    fields: np.ndarray | None = None
    couplings: np.ndarray | None = None
    if arm == "profile":
        fields = model.log_profile.astype(np.float64)
    elif arm in ("potts_fields", "full"):
        fields = model.h.astype(np.float64)
    if arm in ("coupling", "full"):
        couplings = model.J.astype(np.float64)
    elif arm == "permuted_coupling":
        permutation = permutation_for(model.domain, length, int(permutation_seed))
        couplings = model.J.astype(np.float64)[np.ix_(permutation, permutation)]
        provenance.update(
            permutation_seed=int(permutation_seed),
            column_permutation=[int(value) for value in permutation],
            permutation_sha256=hashlib.sha256(
                ",".join(str(int(value)) for value in permutation).encode()
            ).hexdigest(),
        )
    elif arm == "cross_domain_coupling":
        _require_target_alphabet(donor)
        _require_qualified(donor, role="donor")
        if donor.domain == model.domain:
            raise ValueError("cross_domain_coupling needs a donor other than the target domain")
        if donor.length < length:
            raise ValueError(
                f"donor {donor.domain} has {donor.length} columns, fewer than the target's {length}"
            )
        couplings = np.ascontiguousarray(donor.J.astype(np.float64)[:length, :length])
        provenance.update(
            donor_domain=donor.domain,
            donor_length=donor.length,
            donor_columns_used=length,
            donor_selection_rule=(
                "smallest donor_length - target_length among qualified domains with "
                "donor_length >= target_length, ties by ascending domain id; first L columns used"
            ),
        )

    bias = ArmBias(arm, model.domain, length, fields, couplings, provenance)
    _check_reference_agreement(bias, model)
    return bias


#: How each target-model arm's unit-strength bias is expressed in the family
#: module's own three declared vectors. ``potts_fields`` has no vector of its own
#: there, and ``full`` minus ``coupling`` is exactly ``h`` -- which is also the
#: statement that makes the factorial exact, so it is checked rather than assumed.
_REFERENCE_EXPRESSION: dict[str, Any] = {
    "potts_fields": lambda b: b["full"] - b["coupling"],
    "coupling": lambda b: b["coupling"],
    "full": lambda b: b["full"],
    "profile": lambda b: b["marginal"],
}


def _check_reference_agreement(bias: ArmBias, model: FamilyModel) -> None:
    """Bind the four target-model arms to the family module's own bias declaration.

    Checked on the wild-type prefix at the first, a middle and the last position.
    The permuted and cross-domain arms have no reference by construction, and
    ``native`` is checked to carry no component at all.
    """

    if bias.arm == "native":
        if bias.has_fields or bias.has_coupling:
            raise RuntimeError("the native arm must carry neither component")
        return
    expression = _REFERENCE_EXPRESSION.get(bias.arm)
    if expression is None:
        return
    for position in sorted({0, bias.length // 2, bias.length - 1}):
        prefix = model.wildtype[:position]
        reference = expression(guidance_biases(model, prefix))
        value = bias_vector(bias, position, encode(prefix))
        if not np.allclose(value, reference, rtol=0.0, atol=1e-9):
            raise RuntimeError(
                f"{bias.arm}: unit-strength bias at position {position} differs from its "
                f"family_potts.guidance_biases expression by "
                f"{float(np.abs(value - reference).max()):.3e}"
            )


# ------------------------------------------------------------------ the layout


@dataclass(frozen=True)
class NativeLayout:
    """The character layout of one length-``L`` native rendering.

    ``template`` holds :data:`RESIDUE_SLOT` at each residue-bearing character and
    the character's extended-alphabet code at each structural literal.
    ``literals`` maps each literal's code back to its character, so an artefact
    records which non-residue characters this arm's stream carries.
    """

    arm: str
    length: int
    prompt: str
    template: np.ndarray
    literals: dict[int, str]
    residues_before: np.ndarray

    @property
    def characters(self) -> int:
        return int(self.template.size)

    def expected_text(self, sequence: str) -> str:
        """The exact native character stream of one finished sequence, prompt excluded."""

        if len(sequence) != self.length:
            raise ValueError(f"sequence of {len(sequence)} residues is not length {self.length}")
        out, position = [], 0
        for code in self.template.tolist():
            if code == RESIDUE_SLOT:
                out.append(sequence[position])
                position += 1
            else:
                out.append(self.literals[code])
        return "".join(out)


def generation_point(arm: Any) -> Any:
    """Where this arm's native stream starts, and how its prompt tokenises.

    ``unconditional_generation.ADMITTED`` is the right source for every arm that
    campaign admits, and it stays the default. It is not, however, a declaration
    about *this* experiment: it refuses ``progen3-3b`` because EXP-R2-233 is
    complete, which says nothing about whether a later experiment may sample the
    checkpoint, and it reaches some arms through a released generator rather than
    through a forward loop. An arm that carries its own ``generation_point`` --
    :mod:`src.capability.decoding.progen3_decoding` derives one from the
    checkpoint's own batch preparer -- is read from there, so E20's roster is not
    pinned to another campaign's exclusions. The declaration is the same type
    either way, so a generation point has one shape.
    """

    declared = getattr(arm, "generation_point", None)
    if declared is not None:
        return declared
    return require_admitted(arm.name)


def _probe(length: int, offset: int) -> str:
    return "".join(AA20[(index + offset) % 20] for index in range(length))


def native_layout(arm: Any, length: int) -> NativeLayout:
    """Derive the arm's residue/literal character layout from its own rendering.

    Two probes that differ at every residue are rendered through the arm's
    declared input format. Characters that differ between the renderings are
    residue slots and characters that agree are structural literals, which
    identifies the layout without this module knowing anything about the model.
    """

    if length < 1:
        raise ValueError("a native layout needs at least one residue")
    prompt = generation_point(arm).prompt
    first, second = _probe(length, 0), _probe(length, 1)
    rendered = render_records(arm, [first, second], modality="protein")
    if len(rendered) != 2 or len(rendered[0]) != len(rendered[1]):
        raise ValueError(f"{arm.name}: two length-{length} renderings differ in character count")
    for text in rendered:
        if not text.startswith(prompt):
            raise ValueError(
                f"{arm.name}: the native rendering does not open with the declared generation "
                f"prompt {prompt!r}"
            )
    body = (rendered[0][len(prompt):], rendered[1][len(prompt):])
    codes, literals, slots = [], {}, []
    for index, (left, right) in enumerate(zip(body[0], body[1], strict=True)):
        if left != right:
            codes.append(RESIDUE_SLOT)
            slots.append(index)
        else:
            if left in _RESIDUE_CODE:
                raise ValueError(
                    f"{arm.name}: character {left!r} at rendered offset {index} is a canonical "
                    "residue in a structural position; the layout cannot be identified"
                )
            code = literals.setdefault(left, 20 + len(literals))
            codes.append(code)
    if len(slots) != length:
        raise ValueError(
            f"{arm.name}: the rendering exposes {len(slots)} residue slots for {length} residues"
        )
    template = np.asarray(codes, dtype=np.int64)
    layout = NativeLayout(
        arm=arm.name,
        length=length,
        prompt=prompt,
        template=template,
        literals={code: character for character, code in literals.items()},
        residues_before=np.concatenate([[0], np.cumsum(template == RESIDUE_SLOT)]).astype(np.int64),
    )
    for probe, text in zip((first, second), body, strict=True):
        if layout.expected_text(probe) != text:
            raise ValueError(f"{arm.name}: the derived layout does not reproduce its own rendering")
    return layout


# ------------------------------------------------------------- the token table


@dataclass(frozen=True)
class TokenSpanTable:
    """Every vocabulary token that could ever appear in this arm's residue stream.

    A candidate token is one whose decoded text is non-empty and built only from
    canonical residues and this layout's structural literals; everything else --
    end-of-sequence, padding, non-canonical residues, punctuation, partial UTF-8 --
    is outside the table and can never be drawn. ``characters`` and ``residues``
    are right-padded code matrices, so admissibility and the span bias are both
    evaluated over the whole candidate set at once.
    """

    ids: np.ndarray
    characters: np.ndarray
    residues: np.ndarray
    character_length: np.ndarray
    residue_length: np.ndarray
    texts: tuple[str, ...]
    logit_width: int

    @property
    def size(self) -> int:
        return int(self.ids.size)

    @property
    def max_residues(self) -> int:
        return int(self.residue_length.max())

    def admissible(self, layout: NativeLayout, offset: int) -> np.ndarray:
        """Which candidates match ``layout`` starting at character ``offset``."""

        total = layout.characters
        if not 0 <= offset < total:
            raise ValueError(f"character offset {offset} is outside 0..{total - 1}")
        ok = self.character_length <= total - offset
        for position in range(self.characters.shape[1]):
            active = ok & (self.character_length > position)
            if not active.any():
                break
            slot = int(layout.template[offset + position])
            column = self.characters[:, position]
            if slot == RESIDUE_SLOT:
                ok &= ~(active & ~((column >= 0) & (column < 20)))
            else:
                ok &= ~(active & (column != slot))
        return ok


def token_span_table(arm: Any, layout: NativeLayout, *, logit_width: int) -> TokenSpanTable:
    """Build the candidate table for one arm and layout.

    ``logit_width`` is the model's output width. Ids the tokeniser does not
    declare are simply absent from the table and therefore suppressed, which is
    the only safe reading of a logit column with no token behind it.
    """

    vocabulary = arm.tokenizer.get_vocab()
    highest = max(int(value) for value in vocabulary.values())
    if highest >= logit_width:
        raise ValueError(
            f"{arm.name}: tokenizer declares id {highest} beyond the {logit_width}-wide logits"
        )
    ids = sorted(int(value) for value in vocabulary.values())
    texts = arm.tokenizer.batch_decode(
        [[value] for value in ids], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    allowed = dict(_RESIDUE_CODE)
    for code, character in layout.literals.items():
        allowed[character] = code
    kept, spans, kept_texts = [], [], []
    for token_id, text in zip(ids, texts, strict=True):
        if not text or any(character not in allowed for character in text):
            continue
        kept.append(token_id)
        kept_texts.append(text)
        spans.append([allowed[character] for character in text])
    if not kept:
        raise ValueError(f"{arm.name}: no vocabulary token is built from residues and literals")
    width = max(len(span) for span in spans)
    characters = np.full((len(kept), width), _PAD, dtype=np.int64)
    character_length = np.asarray([len(span) for span in spans], dtype=np.int64)
    residue_spans = [[code for code in span if code < 20] for span in spans]
    residue_width = max(1, max(len(span) for span in residue_spans))
    residues = np.zeros((len(kept), residue_width), dtype=np.int64)
    residue_length = np.asarray([len(span) for span in residue_spans], dtype=np.int64)
    for row, (span, residue_span) in enumerate(zip(spans, residue_spans, strict=True)):
        characters[row, : len(span)] = span
        residues[row, : len(residue_span)] = residue_span
    return TokenSpanTable(
        ids=np.asarray(kept, dtype=np.int64),
        characters=characters,
        residues=residues,
        character_length=character_length,
        residue_length=residue_length,
        texts=tuple(kept_texts),
        logit_width=int(logit_width),
    )


# ---------------------------------------------------------------- the decoder


@dataclass(frozen=True)
class CellIdentity:
    """Which cell of the protocol one pool of sequences belongs to."""

    model_arm: str
    domain: str
    guidance_arm: str
    strength: GuidanceStrength
    strength_index: int
    stream_index: int
    stream_seed: int


class GuidedDecoder:
    """Fixed-length, column-aligned guided sampling for one (arm, bias, dose) cell."""

    def __init__(
        self, *, bias: ArmBias, layout: NativeLayout, table: TokenSpanTable,
        strength: GuidanceStrength,
    ):
        if layout.length != bias.length:
            raise ValueError(
                f"the family model has {bias.length} columns and the layout {layout.length} "
                "residues; a column-aligned comparison needs them equal"
            )
        bias.require_strength(strength)
        self.bias = bias
        self.layout = layout
        self.table = table
        self.strength = strength
        self._admissible: dict[int, np.ndarray] = {}
        self._within: dict[int, np.ndarray] = {}
        self._fields: dict[int, np.ndarray] = {}
        # Candidate rows carrying more than ``p`` residues, ``p`` ascending and so
        # shrinking fast: four residues per token is ProtGPT2's mode and 52 its
        # maximum, so a loop over span positions that touched every candidate at
        # every position would do two orders of magnitude more work than the spans
        # contain. Measured on ProtGPT2: the within-token coupling cache drops from
        # ~75 s per cell to under a second.
        self._rows = [np.flatnonzero(table.residue_length > position)
                      for position in range(table.residues.shape[1])]

    # -- the caches, all of them prefix-independent --------------------------

    def admissible(self, offset: int) -> np.ndarray:
        mask = self._admissible.get(offset)
        if mask is None:
            mask = self.table.admissible(self.layout, offset)
            if not mask.any():
                raise EmptyCandidateSet(
                    f"{self.layout.arm}: no vocabulary token matches the native layout at "
                    f"character offset {offset} of {self.layout.characters} without exceeding "
                    f"the remaining residue budget"
                )
            self._admissible[offset] = mask
        return mask

    def _window(self, residues: int) -> int:
        return int(min(self.table.max_residues, self.layout.length - residues))

    def _gather(self, matrix: np.ndarray) -> np.ndarray:
        """``sum_j matrix[j, r_j]`` over each candidate token's residue span."""

        total = np.zeros(self.table.size, dtype=np.float64)
        for position in range(matrix.shape[0]):
            rows = self._rows[position]
            if rows.size == 0:
                break
            total[rows] += matrix[position, self.table.residues[rows, position]]
        return total

    def fields_terms(self, residues: int) -> np.ndarray:
        """``sum_j u[n+j, r_j]`` per candidate token. Exactly zero without a fields component."""

        cached = self._fields.get(residues)
        if cached is None:
            if self.bias.u is None:
                cached = np.zeros(self.table.size, dtype=np.float64)
            else:
                cached = self._gather(self.bias.u[residues : residues + self._window(residues)])
            self._fields[residues] = cached
        return cached

    def within_token_terms(self, residues: int) -> np.ndarray:
        """``sum_{i<j} C[n+j, n+i, r_j, r_i]`` per candidate token: the within-token coupling."""

        cached = self._within.get(residues)
        if cached is not None:
            return cached
        total = np.zeros(self.table.size, dtype=np.float64)
        if self.bias.C is not None:
            for later in range(1, self._window(residues)):
                rows = self._rows[later]
                if rows.size == 0:
                    break
                codes_later = self.table.residues[rows, later]
                accumulated = np.zeros(rows.size, dtype=np.float64)
                for earlier in range(later):
                    block = self.bias.C[residues + later, residues + earlier]
                    accumulated += block[codes_later, self.table.residues[rows, earlier]]
                total[rows] += accumulated
        self._within[residues] = total
        return total

    # -- the per-step, per-row piece -----------------------------------------

    def coupling_terms(self, residues: int, prefix_codes: np.ndarray) -> np.ndarray:
        """The whole coupling span bias per candidate: prefix terms plus within-token terms."""

        total = self.within_token_terms(residues)
        if self.bias.C is None:
            return total
        window = self._window(residues)
        matrix = np.zeros((window, 20), dtype=np.float64)
        if residues:
            block = self.bias.C[residues : residues + window]
            matrix += block[:, np.arange(residues), :, prefix_codes].sum(axis=0)
        return total + self._gather(matrix)

    def token_bias(self, residues: int, prefix_codes: np.ndarray) -> np.ndarray:
        """The dose-weighted span bias of every candidate token at residue index ``residues``."""

        return (self.strength.fields * self.fields_terms(residues)
                + self.strength.coupling * self.coupling_terms(residues, prefix_codes))


# ----------------------------------------------------------------- the loop


def _row_scores(
    decoder: GuidedDecoder, logits: torch.Tensor, offset: int, prefix_codes: np.ndarray
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Biased, layout-masked scores over the full logit width for one row."""

    mask = decoder.admissible(offset)
    residues = int(decoder.layout.residues_before[offset])
    bias = decoder.token_bias(residues, prefix_codes)
    selected = decoder.table.ids[mask]
    scores = torch.full(
        (decoder.table.logit_width,), -math.inf, dtype=torch.float32, device=logits.device)
    index = torch.as_tensor(selected, dtype=torch.long, device=logits.device)
    shift = torch.as_tensor(
        bias[mask].astype(np.float32), dtype=torch.float32, device=logits.device)
    scores[index] = logits[index].to(torch.float32) + shift
    return scores, mask, bias


def _entropy_nats(scores: torch.Tensor) -> float:
    logprobs = torch.log_softmax(scores, dim=-1)
    probabilities = logprobs.exp()
    finite = torch.isfinite(logprobs)
    return float(-(probabilities * torch.where(finite, logprobs, torch.zeros_like(logprobs))).sum())


def _observe_step(
    meter: StrengthMeter, decoder: GuidedDecoder, logits: torch.Tensor, mask: np.ndarray,
    bias: np.ndarray, residues: int,
) -> None:
    """Feed one step to the strength meter; the single call both the sampler and the replay make.

    The bias is rounded to float32 first, exactly as :func:`_row_scores` adds it, so
    the measured guided distribution is the one that was sampled from.
    """

    ids = torch.as_tensor(decoder.table.ids, dtype=torch.long, device=logits.device)
    native = logits.index_select(0, ids).to(torch.float32).double().cpu().numpy()
    meter.add_step(native, bias.astype(np.float32), mask, residues)


def _prompt_ids(arm: Any, decoder: GuidedDecoder) -> list[int]:
    prompt_ids = arm.tokenizer(
        decoder.layout.prompt,
        add_special_tokens=generation_point(arm).add_special_tokens,
        return_tensors=None,
    )["input_ids"]
    if not prompt_ids:
        raise ValueError(f"{arm.name}: the native generation prompt tokenised to nothing")
    return prompt_ids


def _draw_batch(
    *, arm: Any, decoder: GuidedDecoder, seeds: Sequence[int], device: str
) -> list[dict[str, Any]]:
    """One batch of sequences, each from its own generator, through forward plus past.

    ``generate`` is not used: ProGen2's released remote code does not declare what
    Transformers' cache builder reads, so this programme either disables the cache
    or owns the loop. Owning it is exact and keeps the operating point explicit.
    """

    size = len(seeds)
    prompt_ids = _prompt_ids(arm, decoder)
    generators = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        generators.append(generator)
    offsets = [0] * size
    prefixes: list[list[int]] = [[] for _ in range(size)]
    token_ids: list[list[int]] = [[] for _ in range(size)]
    entropies: list[list[float]] = [[] for _ in range(size)]
    meters = [StrengthMeter() for _ in range(size)]
    applied = [0.0] * size
    done = [False] * size

    prompt = torch.tensor([prompt_ids] * size, dtype=torch.long, device=device)
    with torch.no_grad():
        step = arm.model(input_ids=prompt, use_cache=True)
        past = step.past_key_values
        logits = step.logits[:, -1]
        for _ in range(decoder.layout.characters):
            drawn = torch.zeros((size, 1), dtype=torch.long, device=device)
            for row in range(size):
                if done[row]:
                    drawn[row, 0] = token_ids[row][-1]
                    continue
                prefix_codes = np.asarray(prefixes[row], dtype=np.int64)
                scores, mask, bias = _row_scores(decoder, logits[row], offsets[row], prefix_codes)
                probabilities = torch.softmax(scores, dim=-1)
                token = int(torch.multinomial(probabilities, 1, generator=generators[row]).item())
                position = int(np.searchsorted(decoder.table.ids, token))
                if position >= decoder.table.size or int(decoder.table.ids[position]) != token:
                    raise RuntimeError(f"{arm.name}: drew token {token}, which is not a candidate")
                entropies[row].append(_entropy_nats(scores))
                _observe_step(meters[row], decoder, logits[row], mask, bias,
                              int(decoder.table.residue_length[position]))
                applied[row] += float(bias[position])
                token_ids[row].append(token)
                span = decoder.table.residues[position, : decoder.table.residue_length[position]]
                prefixes[row].extend(int(code) for code in span)
                offsets[row] += int(decoder.table.character_length[position])
                drawn[row, 0] = token
                if offsets[row] == decoder.layout.characters:
                    done[row] = True
                elif offsets[row] > decoder.layout.characters:
                    raise RuntimeError(f"{arm.name}: a drawn token ran past the native layout")
            if all(done):
                break
            step = arm.model(input_ids=drawn, past_key_values=past, use_cache=True)
            past = step.past_key_values
            logits = step.logits[:, -1]
    if not all(done):
        raise RuntimeError(
            f"{arm.name}: {done.count(False)} rows did not reach {decoder.layout.length} residues "
            f"within the {decoder.layout.characters}-character layout"
        )
    draws = []
    for row in range(size):
        sequence = "".join(AA20[code] for code in prefixes[row])
        if len(sequence) != decoder.layout.length:
            raise RuntimeError(
                f"{arm.name}: emitted {len(sequence)} residues, not {decoder.layout.length}")
        realised = arm.tokenizer.decode(
            token_ids[row], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        expected = decoder.layout.expected_text(sequence)
        if realised != expected:
            raise RuntimeError(
                f"{arm.name}: the drawn tokens decode to {realised!r}, not the native layout "
                f"{expected!r}"
            )
        draws.append(
            {
                "seed": int(seeds[row]),
                "sequence": sequence,
                "token_ids": token_ids[row],
                "applied_bias": applied[row],
                "step_entropy_nats": entropies[row],
                "guidance_strength": meters[row].record(),
            }
        )
    return draws


#: What :func:`native_reference_draw` establishes, as a test or an artefact records it.
NATIVE_REFERENCE_DECLARATION = (
    "a second, independent implementation of native sampling for a one-residue-per-token "
    "arm, used only to check that guidance at lambda = 0 reproduces native sampling token "
    "for token from one seed. It shares the declared sampling recipe, because that is the "
    "protocol rather than the thing under test, and shares no layout, candidate table or "
    "bias machinery. Comparing the native arm against itself would establish only that the "
    "module is deterministic."
)


def native_reference_draw(
    *, arm: Any, length: int, seed: int, device: str | None = None
) -> dict[str, Any]:
    """One native draw of ``length`` residues, written independently of the biased path.

    The zero-strength invariant -- every arm at ``lambda = 0`` draws the sequence
    native sampling draws from the same seed -- is what makes the arms differ in
    the bias term and in nothing else, and it cannot be checked against the native
    arm itself. :data:`NATIVE_REFERENCE_DECLARATION` says what this shares and what
    it does not.

    Defined for an arm whose native stream is one canonical residue per token, so
    that the candidate set at every step is "the twenty canonical residues" and no
    layout is needed to find it. A tokeniser carrying a multi-residue token is
    refused rather than approximated: its stream cannot be reproduced without the
    layout rule, and an approximation would turn the gate into a near-miss
    comparison.

    The score vector's width is read from the model's own output rather than
    declared, because one ``multinomial`` over a differently shaped vector is a
    different draw, so the comparison only holds against a candidate table built
    on that same width.
    """

    if length < 1:
        raise ValueError("a native reference draw needs at least one residue")
    vocabulary = arm.tokenizer.get_vocab()
    ids = sorted(int(value) for value in vocabulary.values())
    texts = arm.tokenizer.batch_decode(
        [[value] for value in ids], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    residue_ids: list[int] = []
    residue_of: dict[int, str] = {}
    spanning: list[str] = []
    for token_id, text in zip(ids, texts, strict=True):
        if not text or any(character not in _RESIDUE_CODE for character in text):
            continue
        if len(text) == 1:
            residue_ids.append(token_id)
            residue_of[token_id] = text
        else:
            spanning.append(text)
    if spanning:
        raise ValueError(
            f"{arm.name}: the vocabulary carries multi-residue tokens "
            f"{sorted(spanning)[:5]}, so its native stream is not one residue per token "
            "and cannot be reproduced without the layout rule"
        )
    if len(residue_ids) != 20:
        raise ValueError(
            f"{arm.name}: {len(residue_ids)} of the twenty canonical residues have a "
            "single-token spelling; a native reference draw would suppress the rest"
        )
    point = generation_point(arm)
    prompt_ids = arm.tokenizer(
        point.prompt, add_special_tokens=point.add_special_tokens, return_tensors=None
    )["input_ids"]
    if not prompt_ids:
        raise ValueError(f"{arm.name}: the native generation prompt tokenised to nothing")
    target = arm.device if device is None else device
    generator = torch.Generator(device=target)
    generator.manual_seed(int(seed))
    token_ids: list[int] = []
    residues: list[str] = []
    with torch.no_grad():
        step = arm.model(
            input_ids=torch.tensor([prompt_ids], dtype=torch.long, device=target), use_cache=True
        )
        past, logits = step.past_key_values, step.logits[:, -1]
        index = torch.as_tensor(residue_ids, dtype=torch.long, device=logits.device)
        for position in range(length):
            row = logits[0]
            scores = torch.full(
                (row.shape[-1],), -math.inf, dtype=torch.float32, device=row.device)
            scores[index] = row[index].to(torch.float32)
            token = int(
                torch.multinomial(torch.softmax(scores, dim=-1), 1, generator=generator).item())
            if token not in residue_of:
                raise RuntimeError(f"{arm.name}: drew token {token}, which is not a residue")
            token_ids.append(token)
            residues.append(residue_of[token])
            if position + 1 == length:
                break
            step = arm.model(
                input_ids=torch.tensor([[token]], dtype=torch.long, device=target),
                past_key_values=past,
                use_cache=True,
            )
            past, logits = step.past_key_values, step.logits[:, -1]
    return {
        "declaration": NATIVE_REFERENCE_DECLARATION,
        "seed": int(seed),
        "sequence": "".join(residues),
        "token_ids": token_ids,
        "prompt_ids": [int(value) for value in prompt_ids],
    }


#: Agreement required between the dose-weighted bias accumulated over token spans
#: during sampling and the same quantity recomputed per residue from the finished
#: sequence. Both are float64 sums of the identical terms in a different order, so
#: the slack is accumulation order alone.
GUIDANCE_SCORE_TOLERANCE = 1e-6


def generate_cell(
    *,
    arm: Any,
    decoder: GuidedDecoder,
    identity: CellIdentity,
    count: int,
    batch_size: int,
    score_fn: Callable[[str], float],
) -> list[dict[str, Any]]:
    """One cell's pool of sequences, with the per-sequence record of each.

    ``score_fn`` is the caller's native full-sequence scorer, injected so this
    module holds no second scoring contract. Every record's accumulated applied
    bias is checked against the per-residue recomputation from the finished
    sequence; a disagreement is a defect in the token-span arithmetic and refuses
    the cell.
    """

    if count < 1 or batch_size < 1:
        raise ValueError("a cell needs at least one sequence and a positive batch size")
    if identity.guidance_arm != decoder.bias.arm or identity.domain != decoder.bias.domain:
        raise ValueError("the cell identity does not match the decoder's bias")
    if identity.strength != decoder.strength:
        raise ValueError("the cell identity does not match the decoder's strength")
    nominal = nominal_lambda(decoder.bias, decoder.strength)
    code = ARM_CODES[identity.guidance_arm]
    records: list[dict[str, Any]] = []
    for start in range(0, count, batch_size):
        seeds = [identity.stream_seed + index
                 for index in range(start, min(start + batch_size, count))]
        draws = _draw_batch(arm=arm, decoder=decoder, seeds=seeds, device=arm.device)
        for offset, draw in enumerate(draws):
            fields, coupling = sequence_component_sums(decoder.bias, draw["sequence"])
            expected = decoder.strength.fields * fields + decoder.strength.coupling * coupling
            if abs(expected - draw["applied_bias"]) > GUIDANCE_SCORE_TOLERANCE:
                raise RuntimeError(
                    f"{identity.guidance_arm}: the bias accumulated over token spans "
                    f"({draw['applied_bias']:.9f}) differs from the same sum recomputed per "
                    f"residue ({expected:.9f})"
                )
            likelihood = float(score_fn(draw["sequence"]))
            if not math.isfinite(likelihood):
                raise RuntimeError(f"{identity.guidance_arm}: non-finite native log-likelihood")
            steps = len(draw["token_ids"])
            guided = decoder.bias.has_fields or decoder.bias.has_coupling
            records.append(
                {
                    "schema": SEQUENCE_SCHEMA,
                    "id": (
                        f"{identity.domain}_{identity.model_arm}_{code}"
                        f"_g{identity.strength_index}_seed{draw['seed']}"
                    ),
                    "domain": identity.domain,
                    "model": identity.model_arm,
                    "arm_code": code,
                    "guidance_arm": identity.guidance_arm,
                    "lambda_nominal": nominal,
                    "lambda_fields": decoder.strength.fields,
                    "lambda_coupling": decoder.strength.coupling,
                    "strength_index": identity.strength_index,
                    "seed": draw["seed"],
                    "sequence": draw["sequence"],
                    "native_loglik": likelihood,
                    "guidance_score": (fields + coupling) if guided else None,
                    "fields_score": fields if decoder.bias.has_fields else None,
                    "coupling_score": coupling if decoder.bias.has_coupling else None,
                    "guidance_bias_applied": expected,
                    "mean_step_entropy": float(np.mean(draw["step_entropy_nats"])),
                    "attempt_budget": ATTEMPTS_PER_SEQUENCE,
                    "stream_index": identity.stream_index,
                    "stream_seed": identity.stream_seed,
                    "sequence_index": start + offset,
                    "sequence_sha256": hashlib.sha256(draw["sequence"].encode()).hexdigest(),
                    "length": len(draw["sequence"]),
                    "token_steps": steps,
                    "residues_per_token": len(draw["sequence"]) / steps,
                    "token_ids": draw["token_ids"],
                    "step_entropy_nats": draw["step_entropy_nats"],
                    "guidance_strength": draw["guidance_strength"],
                }
            )
    return records


# --------------------------------------------------------------- the replay


def canonical_token_ids(arm: Any, decoder: GuidedDecoder, sequence: str) -> tuple[list[int], str]:
    """The tokeniser's canonical segmentation of a finished sequence, and how far to trust it.

    A stored record holds the residue string, not the drawn token ids. When every
    candidate token is one character the segmentation is unique, so the replay is
    ``"exact"``. Under BPE the canonical encoding may differ from the segmentation
    that was sampled, so the replay is ``"canonical_reapproximated"`` and measures
    the strength along an equivalent path, not the sampled one.
    """

    text = decoder.layout.expected_text(sequence)
    ids = [int(value) for value in arm.tokenizer(text, add_special_tokens=False)["input_ids"]]
    if arm.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
        raise ValueError(f"{arm.name}: the canonical encoding does not decode back to the native layout")
    exact = int(decoder.table.character_length.max()) == 1
    return ids, "exact" if exact else "canonical_reapproximated"


def replay_strength(*, arm: Any, decoder: GuidedDecoder, token_ids: Sequence[int]) -> dict[str, Any]:
    """The strength record of a given token path, teacher-forced through the model.

    One forward pass over the prompt and the path yields every step's native
    logits; admissibility, bias and divergence then go through the same
    :func:`_observe_step` the sampler uses. A token that is not admissible where it
    stands, or a path that does not fill the layout exactly, is refused.

    **This assumes the model's full-prefix logits are the logits the sampler saw**,
    which holds for a float32 arm whose forward is prefix-exact and does *not* hold
    for ProGen3. Its expert mixture reduces over the flattened token axis, so the
    incremental path the sampler runs and the single teacher-forced pass here are
    different computations: measured on ProGen3-3B over four 52-residue draws, the
    per-step twenty-residue log-probabilities disagree by up to 0.289 nats (mean
    0.031-0.056 per step) and the drawn token's own log-probability by 0.28-1.00
    nats summed along a sequence -- the same order as the smallest coupling doses.
    A ProGen3 strength measurement therefore has to replay through the sampler's
    own incremental cache rather than through this function; the sampled pool
    itself is unaffected, because the forward is bit-reproducible at a fixed batch
    extent and cache discipline.
    """

    if not len(token_ids):
        raise ValueError("a replay needs at least one token")
    prompt_ids = _prompt_ids(arm, decoder)
    forced = torch.tensor([prompt_ids + [int(value) for value in token_ids[:-1]]],
                          dtype=torch.long, device=arm.device)
    with torch.no_grad():
        logits = arm.model(input_ids=forced, use_cache=False).logits[0]
    meter, prefix, offset = StrengthMeter(), [], 0
    for step, token in enumerate(token_ids):
        position = int(np.searchsorted(decoder.table.ids, int(token)))
        if position >= decoder.table.size or int(decoder.table.ids[position]) != int(token):
            raise ValueError(f"{arm.name}: token {token} at step {step} is not a candidate")
        mask = decoder.admissible(offset)
        if not mask[position]:
            raise ValueError(f"{arm.name}: token {token} at step {step} does not fit the native layout")
        bias = decoder.token_bias(int(decoder.layout.residues_before[offset]),
                                  np.asarray(prefix, dtype=np.int64))
        _observe_step(meter, decoder, logits[len(prompt_ids) - 1 + step], mask, bias,
                      int(decoder.table.residue_length[position]))
        prefix.extend(int(code) for code in
                      decoder.table.residues[position, : decoder.table.residue_length[position]])
        offset += int(decoder.table.character_length[position])
    if offset != decoder.layout.characters:
        raise ValueError(f"{arm.name}: the path covers {offset} of {decoder.layout.characters} characters")
    return meter.record()
