"""Per-position retention of the autoregressive mutation score, before its collapse.

An autoregressive mutation score is the difference of two summed next-token log
likelihoods. The sum is formed and collapsed in one expression, so only the
scalar survives. This module retains the vector that expression reduces, and
nothing else: :func:`target_nll_terms` is
``scripts/capability/stages/context_homologue.py::_target_nll`` with the final
``.sum()`` removed, on the identical slice, the identical ``float()`` cast and
the identical ``log_softmax`` axis. Retention is therefore the same quantity
observed before its reduction, which is what makes the decomposition below a
finer view of a published number rather than a new measurement.

Two further things are needed to read the vector, and both are properties of the
rendering rather than of the model.

*Residue coverage.* A scored token covers a contiguous run of residues: exactly
one for a residue-level interface, zero for a pure formatting token such as a
ProtGPT2 FASTA newline, and several for a BPE merge. :class:`ResidueCoverage`
recovers that run per token and validates it twice against the sequence itself,
so an attribution is never assumed from a token count.

*The scored suffix.* Several interfaces do not score the whole sequence: the
ProteinGLM continuation scores residues 2..L and the literal-text renderings
treat the first token as context. The scored span is a residue suffix in every
admitted packing, so the offset of that suffix is recovered by subtraction
rather than by parsing a prefix, which would otherwise have to know that
``[START_AMINO]`` carries capital letters that are not residues.

The partition itself is defined on the residue axis and not on the token axis,
because a substitution changes BPE segmentation: the two states of a variant
share a residue index but need not share a token grid. Every scored token of one
state falls in exactly one class relative to the mutated residue ``m`` --
``pre`` when its last covered residue precedes ``m``, ``at`` when it covers
``m``, ``post`` when its first covered residue follows ``m`` -- so the three
sums reconstruct the state's own retained scalar exactly, and their signed
differences between states reconstruct the mutation score exactly.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..core.amino_acids import AA20

_AA20 = frozenset(AA20)

#: Classes of the residue-axis partition, in the order the record reports them.
PARTITION = ('upstream', 'own', 'downstream')

SEMANTICS = (
    'per-scored-token next-token negative log likelihood in nats, float32, in '
    'packed token order over the target span [start, end); the retained vector '
    'of the expression whose sum is the archived scalar'
)

PARTITION_SEMANTICS = (
    'residue-axis partition relative to the mutated residue: a scored token is '
    'upstream when its last covered residue precedes the mutated residue, own '
    'when it covers the mutated residue, downstream when its first covered '
    'residue follows it; a formatting token covering no residue is upstream '
    'when it precedes the mutated residue and downstream otherwise'
)


#: Thread-pool variables a position-term extraction must declare. Float32
#: reduction order depends on the pool, so an unpinned process is a different
#: quantity rather than a slower one, and is refused rather than recorded.
BLAS_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')


def blas_pinning(expected: int | None = None) -> dict[str, Any]:
    """Resolved thread pool, refusing a process whose pool is not declared."""

    import os

    resolved: dict[str, Any] = {'torch_num_threads': int(torch.get_num_threads()),
                                'torch_num_interop_threads': int(torch.get_num_interop_threads())}
    for name in BLAS_VARIABLES:
        value = os.environ.get(name)
        if value is None or not value.strip().isdigit() or int(value) < 1:
            raise SystemExit(
                f'{name} is not a declared positive thread count; a position-term '
                'extraction is refused rather than run with an undeclared BLAS pool'
            )
        resolved[name] = int(value)
    if len({resolved[name] for name in BLAS_VARIABLES}) != 1:
        raise SystemExit('the declared BLAS thread counts disagree with each other')
    declared = resolved[BLAS_VARIABLES[0]]
    if expected is not None and declared != int(expected):
        raise SystemExit(f'declared BLAS pool {declared} differs from the campaign\'s {expected}')
    if resolved['torch_num_threads'] != declared:
        raise SystemExit(
            f'torch reports {resolved["torch_num_threads"]} threads against the declared '
            f'{declared}; the pool is not the one the record would claim'
        )
    return resolved


def target_nll_terms(
    logits: torch.Tensor, ids: torch.Tensor, start: int, end: int
) -> torch.Tensor:
    """The per-token vector that ``_target_nll`` reduces, on the identical slice.

    Returned in float32 on the logits' own device, so that re-reducing it there
    with ``.sum()`` is the identical reduction over the identical values and the
    retention check is bit-exact rather than tolerant.
    """

    if start < 1:
        raise ValueError("the first target token has no preceding position to predict it")
    logprobs = F.log_softmax(logits[0, start - 1 : end - 1].float(), dim=-1)
    targets = ids[0, start:end]
    return -logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def retention_residual(terms: np.ndarray, nll_sum: float, device: Any) -> float:
    """Re-reduce the array that will be written and compare it to its own scalar.

    The comparison is against the value ``_target_nll`` produced in the same
    forward, and the reduction is the same ``.sum()`` on the same device, so a
    nonzero result is a retention defect -- a wrong slice, a lost element, a
    dtype demotion -- and never a property of the arithmetic.
    """

    if terms.dtype != np.float32:
        raise ValueError("retained position terms must be float32, as the expression forms them")
    reduced = float(torch.as_tensor(terms, device=device, dtype=torch.float32).sum())
    return reduced - float(nll_sum)


class ResidueCoverage:
    """How many residues each scored token of one arm's rendering covers.

    One decode per distinct token id, cached, because a cohort re-decodes the
    same few thousand ids millions of times. Both validations below run per
    state rather than per arm: a per-arm spot check cannot see a sequence whose
    own segmentation is the pathological one.
    """

    def __init__(self, arm: Any) -> None:
        self._tokenizer = arm.tokenizer
        self._name = getattr(arm, "name", None) or getattr(arm.spec, "name", "arm")
        self._cache: dict[int, str] = {}

    def _piece(self, token_id: int) -> str:
        piece = self._cache.get(token_id)
        if piece is None:
            try:
                piece = self._tokenizer.decode(
                    [token_id], clean_up_tokenization_spaces=False
                )
            except TypeError:
                piece = self._tokenizer.decode([token_id])
            if not isinstance(piece, str):
                raise ValueError(f"{self._name}: tokenizer decode did not return a string")
            self._cache[token_id] = piece
        return piece

    def counts(
        self, ids: Sequence[int], span: tuple[int, int], sequence: str
    ) -> tuple[np.ndarray, int]:
        """Per-scored-token residue counts, and the residue offset of the span.

        Raises rather than guesses whenever the counts do not reconstruct the
        sequence suffix they claim to cover, which is the only way a BPE or
        byte-level attribution can be trusted without asserting a tokeniser
        property this repository does not own.
        """

        start, end = int(span[0]), int(span[1])
        if not 0 <= start < end <= len(ids):
            raise ValueError(f"{self._name}: scored span {span} is not inside the packed row")
        pieces = [self._piece(int(token)) for token in ids[start:end]]
        counts = np.asarray(
            [sum(1 for character in piece if character in _AA20) for piece in pieces],
            dtype=np.int16,
        )
        total = int(counts.sum())
        if total > len(sequence):
            raise ValueError(
                f"{self._name}: scored span covers {total} residues, more than the "
                f"{len(sequence)} of its own sequence"
            )
        offset = len(sequence) - total
        expected = sequence[offset:]
        joined = "".join(character for character in "".join(pieces) if character in _AA20)
        if joined != expected:
            raise ValueError(
                f"{self._name}: per-token decode does not reconstruct the scored "
                "residue suffix"
            )
        try:
            whole = self._tokenizer.decode(
                list(int(token) for token in ids[start:end]),
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            whole = self._tokenizer.decode(list(int(token) for token in ids[start:end]))
        if "".join(character for character in whole if character in _AA20) != expected:
            raise ValueError(
                f"{self._name}: whole-span decode does not reconstruct the scored "
                "residue suffix"
            )
        return counts, offset


def residue_bounds(counts: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Half-open residue span of every scored token, in sequence coordinates."""

    widths = np.asarray(counts, dtype=np.int64)
    starts = offset + np.concatenate(([0], np.cumsum(widths)[:-1])) if len(widths) else np.zeros(0, dtype=np.int64)
    return starts, starts + widths


def partition_masks(
    counts: np.ndarray, offset: int, positions: int | Sequence[int]
) -> dict[str, np.ndarray]:
    """Classify every scored token relative to the mutated residues.

    With one mutated residue this is the three-way split the decomposition is
    read on. With several -- ProteinGym draws variants carrying up to 43
    substitutions -- ``own`` is every token covering any mutated residue,
    ``upstream`` is every token whose last covered residue precedes the first of
    them, and ``downstream`` is the rest: a token between two mutated residues is
    downstream of a mutation and its likelihood is context-mediated, which is the
    side of the contrast it belongs on. For a single substitution the two rules
    coincide exactly.
    """

    sites = np.atleast_1d(np.asarray(positions, dtype=np.int64))
    if sites.size < 1:
        raise ValueError("a mutated state carries at least one mutated residue")
    starts, ends = residue_bounds(counts, offset)
    own = np.zeros(len(starts), dtype=bool)
    for site in sites:
        own |= (starts <= site) & (site < ends)
    first = int(sites.min())
    upstream = (ends <= first) & ~own
    downstream = ~(own | upstream)
    if int(upstream.sum() + own.sum() + downstream.sum()) != len(starts):
        raise ValueError("the residue-axis partition is not disjoint")
    return {"upstream": upstream, "own": own, "downstream": downstream}


#: What makes a variant's decomposition interpretable, declared as a rule rather
#: than inferred from the tokeniser family.
ALIGNMENT = (
    'a variant is aligned when its packed token sequence differs from its wild '
    'type only at the scored tokens covering a substituted residue: equal packed '
    'length, equal scored span, equal residue offset, identical token ids '
    'everywhere outside the union of the two states own-token sets, and equal '
    'per-token residue counts everywhere (hence identical token residue boundaries). Where alignment fails there is '
    'no unique correspondence between a mutant position and a wild-type '
    'position, so the site-against-downstream split is undefined for that '
    'variant rather than merely noisy'
)


def alignment(
    wild: dict, mutant: dict, positions: int | Sequence[int]
) -> dict[str, object]:
    """Does a substitution change the token grid only where the residue changed?

    Both arguments carry ``ids``, ``span``, ``counts`` and ``offset`` for one
    state. The verdict is a measurement on the two token sequences, never an
    assumption from the interface: a byte-pair merge can move a boundary far from
    the substituted residue, and a residue-level interface can only fail this if
    its packing is length-dependent.
    """

    sites = np.atleast_1d(np.asarray(positions, dtype=np.int64))
    record: dict[str, object] = {
        'packed_length_delta': int(len(mutant['ids']) - len(wild['ids'])),
        'span_delta': (int(mutant['span'][0] - wild['span'][0]),
                       int(mutant['span'][1] - wild['span'][1])),
        'residue_offset_delta': int(mutant['offset'] - wild['offset']),
    }
    if (record['packed_length_delta'] or any(record['span_delta'])
            or record['residue_offset_delta']):
        record.update(aligned=False, differing_tokens=None, differing_outside_own=None,
                      differing_outside_scored_span=None, mismatch_extent_residues=None,
                      reason='packed length, scored span or residue offset differs')
        return record
    start, end = int(wild['span'][0]), int(wild['span'][1])
    ids_wild = np.asarray(wild['ids'], dtype=np.int64)
    ids_mutant = np.asarray(mutant['ids'], dtype=np.int64)
    differing = np.flatnonzero(ids_wild != ids_mutant)
    own = (partition_masks(wild['counts'], wild['offset'], sites)['own']
           | partition_masks(mutant['counts'], mutant['offset'], sites)['own'])
    own_indices = set((np.flatnonzero(own) + start).tolist())
    outside = [int(j) for j in differing if int(j) not in own_indices]
    counts_equal = np.array_equal(
        np.asarray(wild['counts']), np.asarray(mutant['counts']))
    outside_span = [index for index in outside if not start <= index < end]
    in_span = [index for index in outside if start <= index < end]
    extent = None
    if in_span:
        starts, ends = residue_bounds(mutant['counts'], mutant['offset'])
        distances = []
        for index in in_span:
            low, high = int(starts[index - start]), int(ends[index - start])
            distances.append(min(int(min(abs(low - site), abs(high - 1 - site)))
                                 for site in sites))
        extent = int(max(distances))
    aligned_flag = bool(not outside and counts_equal and differing.size > 0)
    if aligned_flag:
        reason = 'aligned'
    elif outside_span and not in_span and counts_equal:
        # The substituted residue is not scored at all: ProteinGLM scores residues
        # 2..L and a no-conditioning literal-text rendering treats the first token
        # as context. There is no own term to attribute, so this is a different
        # failure from a moved merge boundary and is counted apart from it. No
        # residue distance is reported, because the differing token lies outside
        # the scored span and has no scored residue span to measure from.
        reason = 'the substituted residue lies outside the scored span'
    elif outside or not counts_equal:
        reason = 'token ids or residue counts differ away from the substituted residue'
    else:
        reason = 'no token differs from the wild type'
    record.update(
        aligned=aligned_flag,
        differing_tokens=int(differing.size),
        differing_outside_own=len(outside),
        differing_outside_scored_span=len(outside_span),
        mismatch_extent_residues=extent,
        reason=reason)
    return record


def state_parts(
    terms: np.ndarray, counts: np.ndarray, offset: int, positions: int | Sequence[int]
) -> dict[str, float]:
    """The three class sums of one state's retained vector, in nats.

    Accumulated in float64 so that the three parts add back to the state's own
    float64-accumulated total exactly, which is what lets the mutation score's
    decomposition be checked independently of the float32 retention check.
    """

    masks = partition_masks(counts, offset, positions)
    parts = {name: float(np.asarray(terms, dtype=np.float64)[mask].sum()) for name, mask in masks.items()}
    parts["total"] = float(np.asarray(terms, dtype=np.float64).sum())
    parts["own_tokens"] = int(masks["own"].sum())
    parts["own_residue_width"] = int(np.asarray(counts, dtype=np.int64)[masks["own"]].sum())
    return parts


#: Relative bound on the partition residual, from the arithmetic rather than from
#: the data: the retained scalar is a float32 reduction of at most 1024 terms and
#: the class sums are a float64 reduction of the same terms, so the two totals may
#: differ by about ``log2(n) * 2**-24`` of each state's own summed magnitude, and
#: two states contribute.
PARTITION_RESIDUAL_RELATIVE = 2.0 * 10.0 * 2.0 ** -24


def mutation_parts(
    wild: dict[str, float], mutant: dict[str, float], scalar: float
) -> dict[str, float]:
    """Signed per-class contributions to ``logP(mutant) - logP(wild type)``.

    The sign convention is the archived one: the retained scalar is minus the
    summed negative log likelihood, so a positive contribution is likelihood the
    mutant state places above the wild type.

    ``scalar`` is the retained mutation score itself, and it is what ``full``
    reports, so the published quantity is the published quantity and not a
    float64 re-summation of it. The class sums are accumulated in float64, which
    is the better estimator of each part but not the same reduction, so the two
    totals differ by a float32 rounding residual. That residual is reported as
    ``closure_nats`` against an arithmetic bound rather than folded away.
    """

    record = {name: float(wild[name] - mutant[name]) for name in PARTITION}
    record["full"] = float(scalar)
    record["float64_full"] = float(wild["total"] - mutant["total"])
    record["closure_nats"] = float(
        record["full"] - (record["upstream"] + record["own"] + record["downstream"])
    )
    record["closure_bound_nats"] = float(
        PARTITION_RESIDUAL_RELATIVE * (abs(wild["total"]) + abs(mutant["total"]))
    )
    record["own_tokens_wild"] = int(wild["own_tokens"])
    record["own_tokens_mutant"] = int(mutant["own_tokens"])
    record["own_residue_width_mutant"] = int(mutant["own_residue_width"])
    record["own_scored"] = bool(mutant["own_tokens"] > 0 and wild["own_tokens"] > 0)
    return record


SPAN_ALIGNMENT = (
    'single substitution scored in both states with identical unscored conditioning; '
    'unchanged common-prefix tokens and rejoined suffix tokens match both ids and '
    'residue boundaries, and the remaining contiguous token span covers the mutation '
    'in each state even if its token counts differ'
)


def retokenized_span_masks(wild: dict, mutant: dict, position: int):
    """Exact local-span partition when a substitution changes token boundaries.

    The unchanged prefix and rejoined suffix must match both token identity and
    absolute residue boundaries. The remaining block is a mutation-associated
    token span, never a single-token attribution. Unscored conditioning changes
    and mutations outside either state's scored span are not admitted.
    """
    states = (wild, mutant)
    starts_ends = [residue_bounds(s['counts'], s['offset']) for s in states]
    own_masks = [partition_masks(s['counts'], s['offset'], position)['own'] for s in states]
    if not all(mask.any() for mask in own_masks):
        raise ValueError('mutation lies outside a scored span')
    if (wild['offset'] != mutant['offset']
            or list(wild['ids'][:wild['span'][0]]) != list(mutant['ids'][:mutant['span'][0]])):
        raise ValueError('unscored conditioning prefix differs')
    tokens = [list(s['ids'][s['span'][0]:s['span'][1]]) for s in states]
    widths = list(map(len, tokens))
    left = 0
    while left < min(widths):
        a, b = (bounds[0][left] for bounds in starts_ends)
        c, d = (bounds[1][left] for bounds in starts_ends)
        if not (tokens[0][left] == tokens[1][left] and a == b and c == d and c <= position):
            break
        left += 1
    right = 0
    while right < min(n-left for n in widths):
        wi, mi = widths[0]-right-1, widths[1]-right-1
        ws, we = starts_ends[0][0][wi], starts_ends[0][1][wi]
        ms, me = starts_ends[1][0][mi], starts_ends[1][1][mi]
        if not (tokens[0][wi] == tokens[1][mi] and ws == ms and we == me and ws > position):
            break
        right += 1
    masks = []
    for n, own in zip(widths, own_masks):
        index = np.arange(n)
        current = dict(upstream=index<left, own=(index>=left)&(index<n-right), downstream=index>=n-right)
        if not np.all(current['own'][own]):
            raise ValueError('mutation-associated span does not cover the mutation')
        masks.append(current)
    return tuple(masks)


def masked_state_parts(terms: np.ndarray, counts: np.ndarray, masks: dict) -> dict:
    """Sum a declared exhaustive per-state partition, retaining its native units."""
    values = np.asarray(terms, dtype=np.float64)
    stacked = np.stack([np.asarray(masks[k], dtype=bool) for k in PARTITION])
    if stacked.shape != (3,len(values)) or not np.all(stacked.sum(0)==1):
        raise ValueError('state partition must assign every scored term exactly once')
    parts = {k:float(values[masks[k]].sum()) for k in PARTITION}
    parts.update(total=float(values.sum()), own_tokens=int(masks['own'].sum()),
                 own_residue_width=int(np.asarray(counts)[masks['own']].sum()))
    return parts
