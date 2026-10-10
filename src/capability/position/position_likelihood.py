"""The one place a position-resolved log likelihood is produced in this project.

Every scalar mutation-effect number this programme reports is a difference of two
*summed* next-token log likelihoods, and every stage that formed one collapsed the
sum in the same expression that built it. :mod:`position_terms` already retains
that vector for one arm family; this module makes the retention general, so that
the per-position array is produced once, by one convention, and reread by every
experiment that needs it.

What "one convention" means here is concrete, and it is not a new convention.
Each checkpoint family in this panel writes a protein differently -- ProGen2
prefixes a direction marker, ProGen3 brackets the residues between ``1`` and
``2``, ProteinGLM scores residues 2..L after a three-token prefix, RITA streams
documents separated by its native EOS, the Galactica and ProLLaMA lineages wrap
the residues in a declared joint rendering, and a text arm scores the literal
amino-acid string -- and all of that is already declared once, in
``src.capability.context.context_homologue`` and the per-family doors under
``src.capability.models``. :class:`CausalScorer` therefore packs through
:func:`src.capability.readouts.readout_extraction.pack_sequence` and reduces
through the stage's own ``_target_nll``, with :func:`position_terms.target_nll_terms`
supplying the same expression minus its ``.sum()``. The consequence is the
property this module exists to guarant: **the sum of the retained array is the
project's own scalar for that (arm, sequence), bit for bit**, so a
position-resolved reading is a finer view of a published number rather than a
second measurement of it. :func:`retention_residual` is evaluated for every
state and a nonzero value aborts the run.

Two paradigms, and the asymmetry between them is the point
----------------------------------------------------------
``causal_next_token``
    term ``j`` is ``-log p(w_j | w_{<j})``. Because the prefix of a mutation at
    site ``i`` is untouched for every ``j < i``, the terms upstream of a
    substitution are *identical by construction*. That is asserted, not hoped
    for: :func:`upstream_invariance` returns the worst upstream discrepancy and a
    nonzero one means the forward, the packing or the alignment is wrong. It also
    means a causal arm can only exhibit propagation downstream of the mutation,
    which is a property of the model class and not a defect of the measurement.

``masked_marginal_pseudolikelihood``
    term ``j`` is ``-log p(w_j | w_{-j})``, read from a forward in which position
    ``j`` alone is masked. The sum is a pseudo-log-likelihood; a masked language
    model has no sequence log likelihood and this module does not pretend
    otherwise. The member is the one the exploratory denoising roster already
    qualified (``scripts/capability/interactions/exploratory_denoising_arms.py``):
    corruption level ``1/L`` with one position masked. Its ``j = i`` term
    reproduces that roster's archived mutation score exactly, because both states
    are read against the same masked background, and
    :func:`masked_site_response` is what a caller checks that against. Unlike a
    causal arm, such a model responds upstream of the substitution, which is why
    one is carried: it is the comparison that makes a structural reading of the
    downstream-only causal profile interpretable.

The archive this module writes is not a new format. It is the one
``src.capability.extensions.responses.RetainedResponses`` and
``scripts.capability.position.analyse_position_terms.Retained`` already validate
and consume, declared in
``results/extensions/mutation_structure_20261006/responses/source-recovery-requirements.json``
before any producer for it existed. :class:`PositionArchive` emits exactly those
keys, plus the packed ids and the wild-type residue conditionals that the
contact-anticipation question needs, so a reader never has to re-derive packing
from a token count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import util as _import_util
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..core.amino_acids import AA20
from .position_terms import (
    ResidueCoverage,
    partition_masks,
    residue_bounds,
    retention_residual,
    target_nll_terms,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

SCHEMA = "position_likelihood_v1"

#: The two score families this module produces, spelled once.
CAUSAL = "causal_next_token"
MASKED = "masked_marginal_pseudolikelihood"

TERM_SEMANTICS = {
    CAUSAL: (
        "per-scored-token next-token negative log likelihood in nats, float32, in "
        "packed token order over the target span [start, end); the retained vector "
        "whose device float32 sum is the arm's own scalar for this state"
    ),
    MASKED: (
        "per-residue negative log likelihood in nats, float32, read at corruption "
        "level 1/L with that residue alone masked; the sum is a "
        "pseudo-log-likelihood and not a sequence log likelihood"
    ),
}

#: Bidirectional checkpoints, by arm name and checkpoint directory. These are not
#: panel arms: the admitted panel stays at 33 and nothing here enters it. The
#: directory names must agree with ``EXPLORATORY_ROSTER`` in
#: ``scripts/capability/interactions/exploratory_denoising_arms.py``, which is the
#: prior declaration of where these checkpoints live and of the score member they
#: are read under; ``tests/position`` enforces that agreement, because a wrong
#: directory is a different model rather than a slower run.
MASKED_ARMS: dict[str, str] = {
    "dplm-650m": "dplm_650m",
    "dplm-3b": "dplm_3b",
    "esm2-650m": "esm2_650m",
    "esm2-3b": "esm2_t36_3B_UR50D",
}

#: Declared residue capacity shared by the ESM-2 architecture of every masked arm:
#: 1,026 configured positions less the two terminal tokens.
MASKED_RESIDUE_CAPACITY = 1024

#: Arms refused by name, with the reason, rather than quietly absent.
REFUSED_ARMS: dict[str, str] = {
    "zymctrl": (
        "its rendering wraps the residues in an EC conditioning tag, so a "
        "position-resolved array would carry an enzyme-class prompt that no other "
        "arm in the panel carries and the per-position comparison would not be "
        "between the same quantities"
    ),
    "progen3-3b": (
        "its expert mixture reduces over the flattened token axis, so a single "
        "substitution changes the expert gather for the whole sequence including the "
        "tokens before it: the shared prefix is not the same computation in the two "
        "states and the per-position terms shift by about half a nat. This is the "
        "layout sensitivity the project already measured and already acted on -- "
        "results/R1/position_terms_20260926/receipts/layout/layout_assessment.json "
        "records a panel-mean wild-type-state shift of 0.5317 nats over 191 assays "
        "(median 0.4000, max 2.7940) and a downstream-term shift of 0.5149 nats, with "
        "bit-identical repeats of the same layout, which is why this arm is outside "
        "the position-term panel. It is excluded here for the same measured reason "
        "rather than admitted under a tolerance two orders of magnitude above every "
        "quantity a position-resolved analysis reports"
    ),
}

#: Pre-registered admission of a per-position residual, reused verbatim from this
#: lane's own declaration in ``scripts/capability/position/analyse_position_terms.py``
#: (``TIER2_NATS``, ``TIER2_REPEAT_MULTIPLE``), which fixed both before any residual
#: had been seen. ``tests/position`` asserts the two declarations agree, because a
#: second tolerance that drifted from the first would be a second standard.
TIER2_NATS = 1.0e-4
TIER2_REPEAT_MULTIPLE = 3.0

PREFIX_INVARIANT_RULE = (
    "the terms upstream of a substitution must be bit-identical between the two "
    "states (tier 1). A nonzero residual is admitted only as tier 2: at or below "
    "1.0e-4 nats AND at or below three times the arm's own measured repeat maximum "
    "on this same cohort, where the repeat maximum is the largest difference between "
    "two identical forwards of a wild-type row. An arm whose forward reproduces "
    "bit-for-bit has a repeat maximum of zero and therefore no tier-2 room at all, "
    "so the exact-zero property -- and with it the invariant's ability to catch a "
    "packing, alignment or batch-composition defect -- is preserved by construction "
    "rather than by an arm allow-list"
)


def declared_refusals() -> list[dict[str, str]]:
    """Every arm this project declines position-resolved work for, with the reason.

    A panel record names these from here rather than inferring them from a
    missing product. An arm that is absent because the project decided not to
    produce it is a different outcome from an arm that is absent because its cell
    broke, and only this declaration distinguishes the two; a reader of a
    downstream analysis therefore sees the refusal as a refusal even though the
    refused arm never reaches that analysis at all.
    """

    return [{"arm": arm, "reason": reason} for arm, reason in sorted(REFUSED_ARMS.items())]


def residual_tier(worst: float, repeat_max: float) -> int:
    """Which admission tier a prefix residual falls in: 1 exact, 2 admitted, 3 refused."""

    if worst == 0.0:
        return 1
    if worst <= TIER2_NATS and worst <= TIER2_REPEAT_MULTIPLE * float(repeat_max):
        return 2
    return 3


def available_arms() -> tuple[str, ...]:
    """Every arm this module can extract, causal arms first."""

    from ..context import context_homologue as ch
    from ..readouts.readout_extraction import text_readout_names

    causal = tuple(sorted(set(ch.ARMS) | set(text_readout_names())))
    return causal + tuple(sorted(MASKED_ARMS))


def paradigm_of(name: str) -> str:
    """Which score family ``name`` is read under."""

    if name in REFUSED_ARMS:
        raise ValueError(f"{name} is refused here: {REFUSED_ARMS[name]}")
    if name in MASKED_ARMS:
        return MASKED
    if name in available_arms():
        return CAUSAL
    raise ValueError(
        f"{name!r} has no position-likelihood door; arms are {list(available_arms())}"
    )


def default_dtype(name: str) -> str:
    """The precision this arm's own door admits, not a preference.

    ProGen3 has no kernel above float16 and is extracted in bfloat16 exactly as
    the admitted Readout production did; Galactica, ProteinGLM, RITA and every
    literal-amino-acid text interface refuse anything but float32 at their own
    doors. Everything else is float32, which is what a per-position array wants:
    a bfloat16 logit row costs about 0.01 nats of term-level noise for nothing.
    """

    paradigm_of(name)
    if name.startswith("progen3-"):
        return "bfloat16"
    return "float32"


def singleton_only(name: str) -> str | None:
    """Why this arm must be scored one row at a time, or ``None``.

    Read from the score stage's own declaration rather than restated: ProGen3's
    expert mixture reduces over the whole flattened batch, so a batched row is a
    different quantity and not a faster one.
    """

    if paradigm_of(name) == MASKED:
        return None
    stage = load_score_stage()
    from ..context import context_homologue as ch

    return stage.SINGLETON_ONLY_PACKINGS.get(ch.packing_of(name)) if name in ch.ARMS else None


def load_score_stage() -> Any:
    """The score stage whose ``_target_nll`` every causal scalar in this project is.

    Loaded by path because it is an executable stage rather than a package module;
    this is the same resolution ``analyse_position_terms.py`` and
    ``extract_paired_position_terms.py`` already use, so all three reduce through
    one expression.
    """

    path = REPO_ROOT / "scripts/capability/stages/context_homologue.py"
    spec = _import_util.spec_from_file_location("position_score_stage", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"the score stage is unavailable at {path}")
    module = _import_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ one state


@dataclass(frozen=True)
class PackedState:
    """One sequence as the arm's own rendering packs it.

    ``span`` is the half-open packed-token range that is scored, ``counts`` the
    number of residues each scored token covers, and ``offset`` the residue index
    the scored span starts at. The four together are what lets a token-axis array
    be read on the residue axis, and they are the sidecar
    ``RetainedResponses`` validates.
    """

    sequence: str
    ids: tuple[int, ...]
    span: tuple[int, int]
    counts: np.ndarray
    offset: int

    def __post_init__(self) -> None:
        start, end = self.span
        if not 1 <= start < end <= len(self.ids):
            raise ValueError(f"scored span {self.span} is not inside a {len(self.ids)}-token row")
        if self.counts.ndim != 1 or len(self.counts) != end - start:
            raise ValueError("one residue count per scored token is required")
        if self.offset < 0 or self.offset + int(self.counts.sum()) != len(self.sequence):
            raise ValueError("residue counts and offset do not cover the sequence suffix")

    @property
    def scored_tokens(self) -> int:
        return self.span[1] - self.span[0]

    def sidecar(self) -> dict[str, Any]:
        """The packing record ``RetainedResponses`` reads, in its own spelling."""

        return {
            "sequence": self.sequence,
            "ids": [int(value) for value in self.ids],
            "span": [int(self.span[0]), int(self.span[1])],
            "counts": [int(value) for value in self.counts],
            "offset": int(self.offset),
        }


@dataclass(frozen=True)
class PositionScore:
    """One state's retained vector, its own scalar, and optional gathered rows."""

    terms: np.ndarray
    nll_sum: float
    retention_residual_nats: float
    gathered: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.terms.dtype != np.float32 or self.terms.ndim != 1:
            raise ValueError("a retained position vector is a one-dimensional float32 array")
        if not np.isfinite(self.terms).all() or np.any(self.terms < 0):
            raise ValueError("a negative log likelihood is finite and non-negative")


# ---------------------------------------------------------------- the scorers


class CausalScorer:
    """Per-position next-token log likelihood under one arm's own rendering.

    Nothing about the packing, the scored span or the reduction is decided here.
    The arm is opened through the readout door that already routes every family
    (:func:`load_readout_arm`), packed by :func:`pack_sequence`, forwarded by the
    stage's own ``_forward_rows``, and reduced by the stage's own ``_target_nll``
    with :func:`target_nll_terms` supplying the unreduced vector. A caller that
    wanted a different convention would have to change those, which is the
    intent.
    """

    paradigm = CAUSAL

    def __init__(self, name: str, *, device: str, dtype: str | None = None) -> None:
        from ..readouts.readout_extraction import load_readout_arm

        if paradigm_of(name) != CAUSAL:
            raise ValueError(f"{name} is not a causal arm")
        self.name = name
        self.dtype = default_dtype(name) if dtype is None else dtype
        self._stage = load_score_stage()
        self.arm = load_readout_arm(name, self._stage, device=device, dtype=self.dtype)
        self.device = self.arm.device
        self._coverage = ResidueCoverage(self.arm)
        self.singleton_only = singleton_only(name)
        self.provenance = {
            "arm": name,
            "paradigm": CAUSAL,
            "dtype": self.dtype,
            "packing": self._packing_name(),
            "scalar_reference": "scripts/capability/stages/context_homologue.py::_target_nll",
            "term_expression": "src/capability/position/position_terms.py::target_nll_terms",
            "singleton_only": self.singleton_only,
        }

    def _packing_name(self) -> str:
        from ..context import context_homologue as ch
        from ..readouts.readout_extraction import text_boundary

        if text_boundary(self.arm) is not None:
            return "literal_amino_acid_text"
        return ch.packing_of(self.name)

    def pack(self, sequence: str) -> PackedState:
        from ..readouts.readout_extraction import pack_sequence

        ids, span, _ = pack_sequence(self.arm, sequence)
        counts, offset = self._coverage.counts(ids, span, sequence)
        return PackedState(
            sequence=sequence,
            ids=tuple(int(value) for value in ids),
            span=(int(span[0]), int(span[1])),
            counts=np.asarray(counts, dtype=np.int64),
            offset=int(offset),
        )

    def score(
        self, states: Sequence[PackedState], *, gather_ids: Sequence[int] | None = None,
    ) -> list[PositionScore]:
        """Retained vectors for these states, **one row per forward**.

        Not a performance oversight. A wild type and one of its single mutants
        share every token before the substituted residue, so their upstream terms
        must be the same numbers -- and that only holds if the two rows were
        computed by the same kernel. Measured on ``progen2-small`` in float32 over
        a 39-residue Tsuboyama domain, scoring the wild type alone and its mutants
        in batches of eight moves the shared-prefix terms by up to
        3.24e-5 nats: cuBLAS selects a different reduction over the hidden
        dimension at a different batch extent, so the arithmetic, not the model,
        changes. At one row per forward the two states are the same computation on
        the same shape and the invariant is exact rather than approximate, which
        is what lets a nonzero upstream difference be read as a defect.

        This is the extraction protocol the pairwise and stability panels of this
        project already run under (``PRODUCTION_BATCH_SIZE = 1``); ProGen3 would
        require it anyway, because its expert mixture reduces over the whole
        flattened batch.
        """

        ids_tensor = None
        if gather_ids is not None:
            ids_tensor = torch.as_tensor(
                [int(value) for value in gather_ids], dtype=torch.long, device=self.device
            )
        from ..readouts.readout_extraction import forward_readout_rows

        results: list[PositionScore] = []
        for state in states:
            # Through the readout door, not the score stage directly: a literal
            # amino-acid text arm is served under an fp32 matmul policy and a
            # float32 logit assertion that the protein path does not apply, and
            # its arm name is not even one the protein packing table knows.
            logits, packed = forward_readout_rows(self.arm, [list(state.ids)], self._stage)
            terms = target_nll_terms(logits, packed, *state.span)
            scalar = self._stage._target_nll(logits, packed, *state.span)["nll_sum"]
            array = terms.detach().cpu().numpy().astype(np.float32, copy=False)
            residual = retention_residual(array, scalar, self.device)
            gathered = None
            if ids_tensor is not None:
                begin, end = state.span
                logprobs = F.log_softmax(logits[0, begin - 1 : end - 1].float(), dim=-1)
                gathered = (
                    logprobs.index_select(-1, ids_tensor)
                    .detach().cpu().numpy().astype(np.float32, copy=False)
                )
            results.append(
                PositionScore(
                    terms=array,
                    nll_sum=float(scalar),
                    retention_residual_nats=float(residual),
                    gathered=gathered,
                )
            )
            del logits, packed
        return results

    def residue_token_slots(self, states: Sequence[PackedState]) -> dict[str, int]:
        """Residues whose scored token is one unambiguous id across these states.

        Measured on the packing rather than asserted from the tokeniser. A residue
        qualifies when every scored token that covers it alone carries the same
        token id, and when no two residues share an id. That is exactly the
        condition under which a vocabulary column is "the model's probability of
        residue ``a`` at this position", and a byte-pair interface in which a
        residue's segmentation depends on its neighbours fails it and is refused
        rather than reported.
        """

        observed: dict[str, set[int]] = {}
        for state in states:
            starts, ends = residue_bounds(state.counts, state.offset)
            for local, (begin, end) in enumerate(zip(starts, ends)):
                if end - begin != 1:
                    continue
                residue = state.sequence[int(begin)]
                observed.setdefault(residue, set()).add(int(state.ids[state.span[0] + local]))
        slots = {
            residue: next(iter(found))
            for residue, found in observed.items()
            if len(found) == 1 and residue in AA20
        }
        seen: dict[int, str] = {}
        for residue, token in sorted(slots.items()):
            if token in seen:
                return {}
            seen[token] = residue
        return slots


class MaskedScorer:
    """Per-residue masked-marginal log likelihood for a bidirectional checkpoint.

    One forward per masked position, at corruption level ``1/L``. The arm is
    opened the way the exploratory denoising roster opens it -- stock
    ``EsmForMaskedLM``, float32, zero missing or mismatched keys -- so that the
    ``j = i`` term of a wild-type/mutant pair is the score that roster already
    archived, and that identity is what a caller verifies rather than assumes.
    The reference loader of the DPLM release was never executed in this project
    and no numerical A/B supports equivalence to it; that limitation travels with
    every number these arms produce.
    """

    paradigm = MASKED

    def __init__(
        self, name: str, *, device: str, dtype: str | None = None, model_root: Path | None = None
    ) -> None:
        from transformers import AutoTokenizer, EsmForMaskedLM

        if paradigm_of(name) != MASKED:
            raise ValueError(f"{name} is not a bidirectional arm")
        if dtype not in (None, "float32"):
            raise ValueError(f"{name}: the masked-marginal member is float32-only; got {dtype!r}")
        root = Path(model_root) if model_root is not None else None
        if root is None:
            raise ValueError("a masked arm needs an explicit checkpoint root")
        directory = root / MASKED_ARMS[name]
        if not directory.is_dir():
            raise FileNotFoundError(f"{name}: no checkpoint directory at {directory}")
        self.name = name
        self.dtype = "float32"
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        model, info = EsmForMaskedLM.from_pretrained(
            directory, torch_dtype=torch.float32, output_loading_info=True,
            local_files_only=True,
        )
        unmatched = {key: info[key] for key in ("missing_keys", "mismatched_keys") if info[key]}
        if unmatched:
            raise RuntimeError(f"{name}: checkpoint does not fit the graph: {unmatched}")
        self.model = model.eval().to(device)
        self.device = device
        self.singleton_only = None
        self.provenance = {
            "arm": name,
            "paradigm": MASKED,
            "dtype": "float32",
            "packing": "residue_tokens_between_cls_and_eos",
            "member": "corruption level 1/L, one masked position, no sampling",
            "scalar_reference": (
                "the device float32 sum of the retained per-residue terms; a masked "
                "language model has no sequence log likelihood and none is claimed"
            ),
            "site_term_reference": (
                "scripts/capability/interactions/exploratory_denoising_arms.py::contrasts"
            ),
            "reference_loader_executed": False,
            "unexpected_keys": list(info["unexpected_keys"]),
            **{key: value for key, value in MASKED_ARMS.items() if key == name},
        }

    def pack(self, sequence: str) -> PackedState:
        if not sequence or set(sequence) - set(AA20):
            raise ValueError(f"{self.name}: a masked arm scores AA20 residues only")
        if len(sequence) > MASKED_RESIDUE_CAPACITY:
            raise ValueError(
                f"{self.name}: {len(sequence)} residues exceeds the declared "
                f"{MASKED_RESIDUE_CAPACITY}-position capacity of this architecture"
            )
        ids = [int(value) for value in self.tokenizer(sequence)["input_ids"]]
        span = (1, 1 + len(sequence))
        pieces = self.tokenizer.convert_ids_to_tokens(ids[span[0] : span[1]])
        if list(pieces) != list(sequence):
            raise ValueError(
                f"{self.name}: residue tokens do not reconstruct the sequence; the "
                "masked-marginal member requires a one-token-per-residue packing"
            )
        return PackedState(
            sequence=sequence,
            ids=tuple(ids),
            span=span,
            counts=np.ones(len(sequence), dtype=np.int64),
            offset=0,
        )

    @torch.no_grad()
    def score(
        self,
        states: Sequence[PackedState],
        *,
        batch_size: int = 16,
        gather_ids: Sequence[int] | None = None,
    ) -> list[PositionScore]:
        if batch_size < 1:
            raise ValueError("batch size is at least one")
        mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
            raise ValueError(f"{self.name}: the tokenizer declares no mask token")
        ids_tensor = None
        if gather_ids is not None:
            ids_tensor = torch.as_tensor(
                [int(value) for value in gather_ids], dtype=torch.long, device=self.device
            )
        results: list[PositionScore] = []
        for state in states:
            begin, end = state.span
            row = torch.as_tensor(list(state.ids), dtype=torch.long, device=self.device)
            terms = torch.empty(end - begin, dtype=torch.float32, device=self.device)
            gathered = (
                torch.empty(end - begin, len(ids_tensor), dtype=torch.float32, device=self.device)
                if ids_tensor is not None
                else None
            )
            for start in range(begin, end, batch_size):
                stop = min(start + batch_size, end)
                batch = row.unsqueeze(0).repeat(stop - start, 1)
                for offset, position in enumerate(range(start, stop)):
                    batch[offset, position] = mask_id
                logits = self.model(input_ids=batch).logits.float()
                for offset, position in enumerate(range(start, stop)):
                    logprobs = F.log_softmax(logits[offset, position], dim=-1)
                    terms[position - begin] = -logprobs[row[position]]
                    if gathered is not None and ids_tensor is not None:
                        gathered[position - begin] = logprobs.index_select(0, ids_tensor)
                del logits
            scalar = float(terms.sum())
            array = terms.detach().cpu().numpy().astype(np.float32, copy=False)
            results.append(
                PositionScore(
                    terms=array,
                    nll_sum=scalar,
                    retention_residual_nats=float(retention_residual(array, scalar, self.device)),
                    gathered=(
                        gathered.detach().cpu().numpy().astype(np.float32, copy=False)
                        if gathered is not None
                        else None
                    ),
                )
            )
        return results

    def residue_token_slots(self, states: Sequence[PackedState]) -> dict[str, int]:
        """Every AA20 residue, because this packing is one token per residue."""

        slots = {}
        for residue in AA20:
            token = self.tokenizer.convert_tokens_to_ids(residue)
            if token is None or int(token) == self.tokenizer.unk_token_id:
                return {}
            slots[residue] = int(token)
        if len(set(slots.values())) != len(slots):
            return {}
        return slots


def load_position_scorer(
    name: str, *, device: str, dtype: str | None = None, model_root: Path | None = None
) -> CausalScorer | MaskedScorer:
    """Open the one scorer this arm is read under."""

    if paradigm_of(name) == MASKED:
        return MaskedScorer(name, device=device, dtype=dtype, model_root=model_root)
    return CausalScorer(name, device=device, dtype=dtype)


# ------------------------------------------------------ invariants on a pair


def upstream_invariance(
    wild: PackedState, mutant: PackedState, score_wild: PositionScore,
    score_mutant: PositionScore, site: int,
) -> float:
    """Worst absolute disagreement upstream of a substitution, in nats.

    For a causal arm this must be exactly zero: the tokens whose last covered
    residue precedes ``site`` are predicted from a prefix the substitution does
    not touch, so their terms are the same numbers in both states. A nonzero
    result is a packing, alignment or forward defect and never a property of the
    model. For a masked arm it is a measurement -- the quantity that makes the
    bidirectional comparison worth carrying -- and is recorded rather than gated.
    """

    mask = partition_masks(wild.counts, wild.offset, site)["upstream"]
    other = partition_masks(mutant.counts, mutant.offset, site)["upstream"]
    if mask.shape != other.shape or not np.array_equal(mask, other):
        raise ValueError("the two states do not share an upstream token partition")
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(score_wild.terms[mask] - score_mutant.terms[mask])))


def repeat_residual(first: PositionScore, second: PositionScore) -> float:
    """Largest term difference between two identical forwards of the same row.

    This is the arm's own nondeterminism, measured on the cohort being scored
    rather than assumed. Some serving paths in this panel are not
    bit-reproducible: ``rita-xl`` in float32 under eager attention differs between
    two identical forwards by 2.5e-6 nats on a 37-residue row, 2.7e-6 on 41 and
    5.0e-6 on 87, while ``progen2-small`` differs by exactly zero on the same rows
    including one of 505 residues. Measuring it is what lets the prefix invariant
    distinguish "this arm's arithmetic wobbles" from "this packing is wrong":
    on ``rita-xl`` the wild-type-versus-mutant prefix difference equalled the
    repeat difference to the last digit on every row tested, a ratio of 1.00,
    which is what nondeterminism looks like and is not what a layout defect looks
    like.
    """

    if first.terms.shape != second.terms.shape:
        raise ValueError("a repeat check compares two forwards of the same row")
    return float(
        np.max(np.abs(first.terms.astype(np.float64) - second.terms.astype(np.float64)))
    )


def masked_site_response(
    wild: PackedState, score_wild: PositionScore, score_mutant: PositionScore, site: int
) -> float:
    """The ``j = i`` response, which for a masked arm is the archived roster score.

    Both states are read with position ``i`` masked, so their backgrounds are the
    same sequence and the contrast is exactly
    ``log p(mutant | background) - log p(wild | background)``: the member
    ``exploratory_denoising_arms.contrasts`` computes. Checking a run against that
    archive is the only external numerical anchor a pseudo-likelihood array has.
    """

    mask = partition_masks(wild.counts, wild.offset, site)["own"]
    if int(mask.sum()) != 1:
        raise ValueError("the site response is defined where one scored token covers the site")
    return float(score_wild.terms[mask][0] - score_mutant.terms[mask][0])


# ------------------------------------------------------------- the archive


class PositionArchive:
    """One assay's states, in the ragged layout this project already validates.

    State zero is the wild type and the rest follow the cohort's own mutant order
    exactly, because that ordering is what ``RetainedResponses`` binds an archive
    to its identity by. ``likelihood`` is the native signed scalar difference
    ``logP(mutant) - logP(wild type)`` formed from each state's own reduced
    scalar, not a float64 re-summation of the retained vector: the published
    quantity stays the published quantity and the difference between the two
    reductions is reported as a closure residual downstream.
    """

    def __init__(self, *, assay: str, paradigm: str) -> None:
        self.assay = assay
        self.paradigm = paradigm
        self._states: list[PackedState] = []
        self._scores: list[PositionScore] = []
        self._mutants: list[str] = []

    def add_wildtype(self, state: PackedState, score: PositionScore) -> None:
        if self._states:
            raise ValueError("the wild-type state is added once, first")
        self._states.append(state)
        self._scores.append(score)

    def add_mutant(self, mutant: str, state: PackedState, score: PositionScore) -> None:
        if not self._states:
            raise ValueError("the wild-type state is added before any mutant")
        if mutant in self._mutants:
            raise ValueError(f"{self.assay}: duplicate mutant identity {mutant!r}")
        self._mutants.append(mutant)
        self._states.append(state)
        self._scores.append(score)

    @property
    def states(self) -> list[PackedState]:
        return list(self._states)

    @property
    def mutants(self) -> list[str]:
        return list(self._mutants)

    @property
    def retention_max_abs_nats(self) -> float:
        return max((abs(score.retention_residual_nats) for score in self._scores), default=0.0)

    def arrays(self, *, extras: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The npz payload: the frozen contract, the packing, and declared extras."""

        if len(self._states) < 2:
            raise ValueError(f"{self.assay}: an archive carries the wild type and one mutant")
        residual = self.retention_max_abs_nats
        if self.paradigm == CAUSAL and residual != 0.0:
            raise ValueError(
                f"{self.assay}: retention residual {residual!r} is not identically zero; the "
                "retained vector is not the one the published scalar reduces"
            )
        terms = np.concatenate([score.terms for score in self._scores])
        counts = np.concatenate([state.counts for state in self._states]).astype(np.int64)
        offsets = np.zeros(len(self._states) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([state.scored_tokens for state in self._states])
        packed = np.concatenate(
            [np.asarray(state.ids, dtype=np.int64) for state in self._states]
        )
        id_offsets = np.zeros(len(self._states) + 1, dtype=np.int64)
        id_offsets[1:] = np.cumsum([len(state.ids) for state in self._states])
        wild_nll = self._scores[0].nll_sum
        payload: dict[str, Any] = {
            "position_nats": terms,
            "position_offsets": offsets,
            "position_residue_counts": counts,
            "position_residue_offsets": np.asarray(
                [state.offset for state in self._states], dtype=np.int64
            ),
            "position_sum_check_nats": np.float64(residual),
            "mutants": np.asarray(self._mutants, dtype=object).astype("U"),
            "likelihood": np.asarray(
                [wild_nll - score.nll_sum for score in self._scores[1:]], dtype=np.float64
            ),
            "wt_likelihood": np.float64(-wild_nll),
            "state_ids": packed,
            "state_id_offsets": id_offsets,
            "state_spans": np.asarray(
                [[state.span[0], state.span[1]] for state in self._states], dtype=np.int64
            ),
        }
        for key, value in (extras or {}).items():
            if key in payload:
                raise ValueError(f"{self.assay}: extra {key!r} would shadow the frozen contract")
            payload[key] = value
        return payload

    def write(self, path: Path, *, metadata: Mapping[str, Any],
              extras: Mapping[str, Any] | None = None) -> None:
        """Write the archive atomically, metadata last inside the same file."""

        payload = self.arrays(extras=extras)
        payload["metadata"] = json.dumps(dict(metadata), sort_keys=True, allow_nan=False)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(destination)


def read_archive(path: Path) -> dict[str, Any]:
    """Reopen one archive as plain arrays, with its metadata parsed.

    The ragged layout is checked by ``analyse_position_terms.Retained`` and
    ``responses.RetainedResponses``, which are the validators this project
    already has; this reader deliberately adds no third set of checks and only
    materialises what a position-resolved analysis indexes.
    """

    with np.load(Path(path), allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files if key != "metadata"}
        payload["metadata"] = json.loads(str(data["metadata"]))
    return payload


def state_terms(payload: Mapping[str, Any], index: int) -> np.ndarray:
    low, high = int(payload["position_offsets"][index]), int(payload["position_offsets"][index + 1])
    return np.asarray(payload["position_nats"][low:high])


def state_counts(payload: Mapping[str, Any], index: int) -> np.ndarray:
    low, high = int(payload["position_offsets"][index]), int(payload["position_offsets"][index + 1])
    return np.asarray(payload["position_residue_counts"][low:high], dtype=np.int64)
