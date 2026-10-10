"""Is the decoding procedure, rather than the model, the limiting factor?

The question
============

This programme's arms predict mutation effects well and do not reliably
generate better proteins. Before any interesting explanation of that is
believed, the mundane one has to be tested: the weights may know more than the
sampler draws out of them. So **nothing here trains anything**. Every weight is
frozen; the only thing that varies is how tokens are drawn from the same fixed
conditional distributions.

What is swept, and why each axis is here
========================================

``temperature``
    Over a range wide enough to contain both over-sharpened sampling (a
    temperature low enough to collapse onto repeats and near-copies) and
    under-sharpened sampling (one high enough to lose the model's structure).

``top_p`` and ``top_k``
    The truncations these arms' sampler actually supports, named explicitly.
    ``top_p = 1.00`` is nucleus truncation switched off and ``top_k = 0`` is
    k-truncation switched off, so the grid contains the untruncated corner
    rather than implying it.

generation budget
    Drawing many candidates and keeping the best is itself a decoding strategy,
    and it is the one most likely to work, so it is measured rather than
    assumed: :data:`DEEP_BUDGET_KEYS` draw 32 candidates per cluster instead of
    8, and :func:`selection_curve` reads the whole best-of-k curve off those
    draws.

termination
    Premature termination and run-on both show up as apparent quality
    differences, so the token cap moves (:data:`TERMINATION_VARIANTS`), a
    minimum-length gate suppresses the terminator until a floor is reached, and
    one variant suppresses the terminator entirely. The arm's terminator is its
    own declared end delimiter -- ``<end>`` for ZymCTRL, ``>`` for ProLLaMA --
    not whatever the tokenizer happens to call ``eos``.

The three things that decide whether the answer means anything
==============================================================

**Fixed compute.** A lower temperature with ten times the candidates is not the
same experiment as a higher temperature at equal candidate count. Every
selection point therefore carries the generation tokens and the folds it spent
(:func:`selection_curve`), and :func:`matched_compute_table` re-reads the sweep
at equal total compute. Without that, "sample more and keep the best" wins by
construction.

**Length and termination are confounds, not achievements.** Predicted
confidence rises steeply with length: a 64-residue fragment of a real protein
folds to mean 0.477 and global 0.170 while the same protein at 320 residues
reaches 0.970 and 0.990. A configuration that merely produces longer sequences
looks better. So every configuration reports its full length distribution
(:func:`config_census`), every structural contrast is against a *whole* natural
record of near-identical length (:func:`natural_band_contrast`), and the
configuration-versus-configuration comparison has a length-standardised
counterpart (:func:`length_standardised`).

**A reference band.** The natural sequences and the existing generation results
were already folded by this project's ESMFold2 instrument at one evaluation
signature; those folds are read by literal path and never recomputed. A
configuration is only ever reported as reaching, or not reaching, that band.

Degenerate wins are measured, not hoped against
===============================================

Low-temperature sampling fails by collapsing onto repeats or onto near-copies of
natural sequences, and one conditioned arm in this programme already emits
members at 100.00% identity to natural entries. :func:`set_profile` therefore
reports duplication, composition entropy, homopolymer content, k-mer repertoire
distance and corpus identity for every configuration, beside a size-matched
natural reference, so that a "quality" gain which is actually memorisation or
collapse is visible as such.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core.statistics import bootstrap_unit_floor, mean_interval
from ..evaluation import generated_phenotype as gp
from ..generation import conditioned_generation as cg

SCHEMA_VERSION = "d1_decoding_sweep_v1"
CAMPAIGN = "D1-DECODE"

#: What this experiment may and may not be read as saying.
CEILING: dict[str, str] = {
    "structural_confidence_is_not_stability": (
        "ESMFold2 CA-pLDDT, pTM and PAE are the predictor's own confidence about "
        "coordinates. They are not thermodynamic stability, not activity and not "
        "function. No stability or ddG predictor is applied here and none is "
        "implied; nothing in this sweep is experimentally verified"
    ),
    "weights_are_frozen": (
        "no training, fine-tuning, adapter or prompt search happens anywhere in "
        "this experiment. A difference between two configurations is a difference "
        "of sampling procedure at identical weights and identical conditioning"
    ),
    "oracle_selection_is_circular": (
        "the esmfold2_confidence_oracle selector chooses the candidate with the "
        "highest confidence and is then scored by confidence. It is an attainable "
        "upper bound on best-of-k, not a decoding strategy: it costs one fold per "
        "candidate drawn, which the compute accounting charges it"
    ),
    "length_is_not_an_outcome": (
        "a configuration that produces longer products scores higher on every "
        "structural confidence measure for that reason alone. Only the "
        "length-matched and length-standardised readings license a quality claim"
    ),
    "novelty_is_not_pretraining_exposure": (
        "identity to UniRef50 measures reference-corpus coverage. UniRef50 is not "
        "any of these arms' declared pretraining set, so a low identity is not "
        "evidence that a sequence was absent from training data"
    ),
    "no_family_oracle_here": (
        "family recognition is not run in this sweep. Structural confidence, "
        "novelty, diversity and length are the measures; a configuration is never "
        "described as producing functional or family-correct proteins"
    ),
}

#: The residue band in which a product is structurally evaluated. It is the band
#: the project's existing natural comparator pool and its ESMFold2 fold contract
#: already cover, so a swept configuration is read against folds that exist
#: rather than against an extrapolation. A product outside the band stays in the
#: census denominator and out of every structural contrast, and the share
#: excluded is reported per configuration: for the run-on variants that share is
#: the finding.
EVALUATION_BAND: tuple[int, int] = (50, 400)

#: Clusters per configuration, and draws per cluster. The cluster is the unit a
#: configuration-level interval is taken over: a frozen class for a conditioned
#: arm, an independent seed block for an unconditioned one. Sixteen is twice the
#: package's eight-unit percentile floor, so a configuration can lose several
#: clusters to out-of-band products and still be reportable.
CLUSTERS_PER_CONFIG: int = 16
DRAWS_PER_CLUSTER: int = 8
DRAWS_PER_CLUSTER_DEEP: int = 32

#: The sampling seed every cell seed is derived from.
SAMPLING_SEED: int = 20261010

#: Fixed across the whole sweep, so that the sweep is a sweep of the sampler and
#: not of anything else. The value is the one both historical generation
#: campaigns used.
REPETITION_PENALTY: float = 1.0
ADD_SPECIAL_TOKENS: bool = True
DTYPE: str = "bfloat16"


@dataclass(frozen=True)
class DecodingConfig:
    """One operating point of the sampler, and nothing else.

    ``allow_terminator`` false suppresses the arm's declared end delimiter for
    the whole generation, which is the run-on arm of the termination axis.
    ``min_new_tokens`` suppresses it until that many tokens have been produced,
    which is the premature-termination arm read from the other side.
    """

    key: str
    axis: str
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    min_new_tokens: int
    allow_terminator: bool
    draws_per_cluster: int
    note: str

    @property
    def draws(self) -> int:
        return CLUSTERS_PER_CONFIG * self.draws_per_cluster

    def record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "axis": self.axis,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "allow_terminator": self.allow_terminator,
            "draws_per_cluster": self.draws_per_cluster,
            "draws": self.draws,
            "repetition_penalty": REPETITION_PENALTY,
            "note": self.note,
        }


CORE_TEMPERATURES: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2)
CORE_TRUNCATIONS: tuple[tuple[float, int], ...] = ((0.90, 0), (0.95, 0), (1.00, 0))
#: Deliberately outside the core grid's span on both sides: 0.3 is sharp enough
#: to collapse and 1.5 is flat enough to lose the model.
EXTREME_TEMPERATURES: tuple[float, ...] = (0.3, 1.5)
REFERENCE_MAX_NEW_TOKENS: int = 400

#: ``(suffix, max_new_tokens, min_new_tokens, allow_terminator, note)``.
TERMINATION_VARIANTS: tuple[tuple[str, int, int, bool, str], ...] = (
    ("eos128", 128, 0, True, "a cap low enough that the budget, not the model, usually stops it"),
    ("eos600", 600, 0, True, "a cap high enough that the model almost always stops first"),
    (
        "minnew64",
        REFERENCE_MAX_NEW_TOKENS,
        64,
        True,
        "the terminator is suppressed for 64 tokens, so an immediate stop cannot happen",
    ),
    (
        "noterm400",
        REFERENCE_MAX_NEW_TOKENS,
        0,
        False,
        "the terminator is suppressed throughout: every product is a run-on to the cap",
    ),
)


def _key(temperature: float, top_p: float, top_k: int, suffix: str) -> str:
    return f"t{temperature:.2f}_p{top_p:.2f}_k{top_k}_{suffix}"


REFERENCE_KEY: str = _key(1.0, 0.95, 0, f"eos{REFERENCE_MAX_NEW_TOKENS}")
#: The historical unconditioned operating point, carried so the sweep contains
#: the point the existing unconditional generation results were produced at.
HISTORICAL_UNCONDITIONED_KEY: str = _key(0.85, 0.95, 50, f"eos{REFERENCE_MAX_NEW_TOKENS}")

#: Which configurations get the deep candidate budget. The reference, because
#: the budget curve has to start where the programme already stands, and the
#: low-temperature point, because "lower the temperature and draw many more" is
#: the specific strategy the fixed-compute comparison exists to price.
DEEP_BUDGET_KEYS: tuple[str, ...] = (
    _key(0.6, 0.95, 0, f"eos{REFERENCE_MAX_NEW_TOKENS}"),
    REFERENCE_KEY,
)


def _build_grid() -> tuple[DecodingConfig, ...]:
    suffix = f"eos{REFERENCE_MAX_NEW_TOKENS}"
    rows: list[DecodingConfig] = []

    def add(axis, temperature, top_p, top_k, variant, note):
        name, max_new, min_new, allow, _ = variant
        key = _key(temperature, top_p, top_k, name)
        rows.append(
            DecodingConfig(
                key=key,
                axis=axis,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new,
                min_new_tokens=min_new,
                allow_terminator=allow,
                draws_per_cluster=(
                    DRAWS_PER_CLUSTER_DEEP if key in DEEP_BUDGET_KEYS else DRAWS_PER_CLUSTER
                ),
                note=note,
            )
        )

    native = (suffix, REFERENCE_MAX_NEW_TOKENS, 0, True, "")
    for temperature in CORE_TEMPERATURES:
        for top_p, top_k in CORE_TRUNCATIONS:
            add(
                "temperature_x_truncation",
                temperature,
                top_p,
                top_k,
                native,
                "the factorial core: temperature crossed with nucleus truncation",
            )
    for temperature in EXTREME_TEMPERATURES:
        add(
            "temperature_extreme",
            temperature,
            0.95,
            0,
            native,
            "outside the core span, to bracket over- and under-sharpened sampling",
        )
    add(
        "truncation_top_k",
        1.0,
        1.00,
        50,
        native,
        "k-truncation alone, with nucleus truncation off, as the other truncation this sampler supports",
    )
    for variant in TERMINATION_VARIANTS:
        add("termination", 1.0, 0.95, 0, variant, variant[4])
    add(
        "historical_operating_point",
        0.85,
        0.95,
        50,
        native,
        "the point the existing unconditional generation results were drawn at",
    )
    keys = [row.key for row in rows]
    if len(set(keys)) != len(keys):
        duplicated = sorted({key for key in keys if keys.count(key) > 1})
        raise AssertionError(f"the decoding grid declares {duplicated} more than once")
    return tuple(rows)


GRID: tuple[DecodingConfig, ...] = _build_grid()
CONFIG_KEYS: tuple[str, ...] = tuple(row.key for row in GRID)


def config(key: str) -> DecodingConfig:
    for row in GRID:
        if row.key == key:
            return row
    raise KeyError(f"unknown decoding configuration {key!r}; declared: {list(CONFIG_KEYS)}")


def config_shard(index: int, num_shards: int) -> tuple[DecodingConfig, ...]:
    """The configurations one generation cell samples.

    Round-robin over the grid sorted by descending draw count, so the deep
    budget configurations cannot land on the same card and leave the other idle.
    """

    if num_shards < 1 or not 0 <= index < num_shards:
        raise ValueError(f"shard {index} of {num_shards} is not a shard")
    order = sorted(GRID, key=lambda row: (-row.draws, row.key))
    return tuple(row for position, row in enumerate(order) if position % num_shards == index)


# ------------------------------------------------------------------- the arms


@dataclass(frozen=True)
class SweptArm:
    """One arm of the sweep, with its conditioning held fixed.

    ``conditioned`` arms take their sixteen clusters from the frozen class queue
    the rest of the programme is read against, so the sweep varies decoding and
    not the request. The unconditioned arm has no class, so its clusters are
    sixteen independent seed blocks -- the same number of units, so a
    configuration-level interval means the same thing in every arm.
    """

    name: str
    modality: str
    note: str

    @property
    def conditioned(self) -> bool:
        return cg.arm(self.name).conditioned


ARMS: dict[str, SweptArm] = {
    "protgpt2": SweptArm(
        name="protgpt2",
        modality="protein",
        note=(
            "the unconditioned protein decoder of the generation experiments. "
            "Prompted with the end-of-text token and a newline, which is the FASTA "
            "rendering its BPE merges were learned over, and with no class request"
        ),
    ),
    "zymctrl": SweptArm(
        name="zymctrl",
        modality="protein",
        note=(
            "the conditioned protein decoder of the generation experiments, held at "
            "the sixteen frozen EC classes of the class queue for every configuration"
        ),
    ),
    "prollama": SweptArm(
        name="prollama",
        modality="protein+text",
        note=(
            "the joint text-and-protein arm: ProLLaMA stage 2, instruction tuned on "
            "top of a protein-adapted Llama-2-7B, prompted through its own "
            "'[Generate by superfamily]' instruction form at the sixteen frozen "
            "superfamilies of the class queue"
        ),
    ),
}
ARM_NAMES: tuple[str, ...] = tuple(ARMS)


def arm(name: str) -> SweptArm:
    if name not in ARMS:
        raise KeyError(f"unknown swept arm {name!r}; declared: {list(ARM_NAMES)}")
    return ARMS[name]


def clusters_for(name: str, queue: Mapping[str, Any] | None) -> tuple[tuple[str, str | None], ...]:
    """This arm's sixteen clusters as ``(cluster_key, prompt_label)`` pairs.

    A conditioned arm's clusters are the frozen queue's classes, in the queue's
    own order, and a queue that does not carry sixteen of them is refused rather
    than padded: the cluster count is what every interval's unit floor is read
    against.
    """

    spec = arm(name)
    if not spec.conditioned:
        return tuple((f"seedblock_{index:02d}", None) for index in range(CLUSTERS_PER_CONFIG))
    if queue is None:
        raise ValueError(f"{name} is conditioned; it needs the frozen class queue")
    entries = cg.queue_entries(queue, name)
    if len(entries) != CLUSTERS_PER_CONFIG:
        raise ValueError(
            f"the frozen queue carries {len(entries)} classes for {name}, not "
            f"{CLUSTERS_PER_CONFIG}; the cluster count is part of this sweep's identity"
        )
    return tuple((entry.key, entry.label) for entry in entries)


def cell_seed(*, arm_name: str, config_key: str, cluster: str) -> int:
    """A per-cell sampling seed derived from :data:`SAMPLING_SEED` alone.

    Derived rather than drawn, so the sweep is reproducible, and per cell rather
    than per run, so two configurations that happen to share a prompt do not
    share a sample and become statistically dependent in a way the cluster
    bootstrap does not model.
    """

    material = f"{CAMPAIGN}|{arm_name}|{config_key}|{cluster}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
    return (SAMPLING_SEED + offset) % (2**31 - 1)


def candidate_id(*, arm_name: str, config_key: str, cluster: str, draw_index: int) -> str:
    material = f"{CAMPAIGN}|{arm_name}|{config_key}|{cluster}|{draw_index}"
    return "dc_" + hashlib.blake2b(material.encode(), digest_size=12).hexdigest()


def in_band(sequence: str) -> bool:
    low, high = EVALUATION_BAND
    return bool(sequence) and low <= len(sequence) <= high and not set(sequence) - set(gp.AA20)


def out_of_band_reason(sequence: str) -> str | None:
    low, high = EVALUATION_BAND
    if not sequence:
        return "empty_product"
    if set(sequence) - set(gp.AA20):
        return "noncanonical_residues"
    if len(sequence) < low:
        return "below_band"
    if len(sequence) > high:
        return "above_band"
    return None


# ------------------------------------------------------------------- sampling


def terminator_ids(handle: Any, spec: cg.GenerationArm) -> tuple[int, ...]:
    """The token ids that close a product for this arm.

    The arm's own declared end delimiter first -- ``<end>`` for ZymCTRL, ``>``
    for ProLLaMA -- and the tokenizer's end-of-sequence id beside it where they
    differ, because a checkpoint may emit either and a termination axis that
    could not see one of them would mis-call its own census. An end delimiter
    with no single id is refused: suppressing it, stopping on it and counting it
    all depend on there being one.
    """

    delimiter = cg.end_delimiter_for(handle, spec)
    identifier = handle.tokenizer.convert_tokens_to_ids(delimiter)
    if identifier is None or int(identifier) < 0:
        raise ValueError(
            f"{spec.name}: end delimiter {delimiter!r} has no single token id, so it "
            "can be neither suppressed nor counted"
        )
    ids = [int(identifier)]
    eos = handle.tokenizer.eos_token_id
    if eos is not None and int(eos) not in ids:
        ids.append(int(eos))
    return tuple(ids)


def sample_configuration(
    handle: Any,
    spec: cg.GenerationArm,
    prompt: str,
    *,
    setting: DecodingConfig,
    n: int,
    seed: int,
    batch_size: int,
    stop_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """``n`` draws of one prompt at one operating point, with the token account.

    This is deliberately not :func:`src.capability.generation.conditioned_generation.sample_continuations`.
    That function is the frozen operating point of the generation campaigns and
    exposes neither a minimum length, nor terminator suppression, nor the
    realised token count a fixed-compute comparison is impossible without.
    Everything it does declare -- the prompt, the end delimiter, the extractor,
    the per-batch seed rule, keeping specials in the decode because two of these
    arms' delimiters *are* special tokens -- is imported from it rather than
    restated.
    """

    import torch

    if n < 1 or batch_size < 1:
        raise ValueError("generation needs a positive count and batch size")
    model = cg.ensure_generate(handle.model)
    tokenizer = handle.tokenizer
    device = getattr(model, "device", None)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=ADD_SPECIAL_TOKENS)
    ids = encoded["input_ids"]
    if ids.shape[1] == 0:
        raise ValueError(f"{spec.name}: the declared prompt encodes to no token")
    ids = ids.to(device)
    prompt_length = int(ids.shape[1])
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError(f"{spec.name}: no pad or eos token to pad a batch with")
    stop = [int(value) for value in stop_ids]
    rows: list[dict[str, Any]] = []
    index = 0
    with torch.no_grad():
        while len(rows) < n:
            size = min(batch_size, n - len(rows))
            torch.manual_seed(seed + index)
            generated = model.generate(
                input_ids=ids.repeat(size, 1),
                attention_mask=torch.ones(
                    (size, prompt_length), dtype=torch.long, device=ids.device
                ),
                do_sample=True,
                temperature=setting.temperature,
                top_p=setting.top_p,
                top_k=setting.top_k,
                repetition_penalty=REPETITION_PENALTY,
                max_new_tokens=setting.max_new_tokens,
                min_new_tokens=setting.min_new_tokens or None,
                eos_token_id=None if not setting.allow_terminator else stop,
                suppress_tokens=stop if not setting.allow_terminator else None,
                pad_token_id=int(pad),
                use_cache=spec.kv_cache,
            )
            tails = generated[:, prompt_length:]
            scores = sequence_logprobs(
                model, generated, prompt_length=prompt_length, stop_ids=stop, pad_id=int(pad)
            )
            for position in range(tails.shape[0]):
                tail = tails[position].tolist()
                stopped_at = next(
                    (offset for offset, token in enumerate(tail) if token in stop), None
                )
                n_new = len(tail) if stopped_at is None else stopped_at + 1
                text = tokenizer.decode(tail[:n_new], skip_special_tokens=False)
                rows.append(
                    {
                        "raw_continuation": text,
                        "n_new_tokens": int(n_new),
                        "terminated_natively": stopped_at is not None,
                        "stop_token_id": None if stopped_at is None else int(tail[stopped_at]),
                        "batch_index": index,
                        "model_logprob_total": float(scores["total"][position]),
                        "model_logprob_per_token": float(scores["per_token"][position]),
                    }
                )
            index += 1
    return rows[:n]


def sequence_logprobs(
    model: Any,
    generated: Any,
    *,
    prompt_length: int,
    stop_ids: Sequence[int],
    pad_id: int,
    chunk: int = 8,
) -> dict[str, list[float]]:
    """The model's own log-probability of what it produced, at temperature one.

    The selector a realistic best-of-k strategy can afford has to be free at
    generation time and independent of the sampler, otherwise comparing budgets
    across temperatures compares selectors as well. This is one teacher-forced
    pass over the already-generated tokens under the **untempered, untruncated**
    distribution, so the same product scores the same whatever operating point
    drew it. Per-token and total are both returned because they rank differently
    on length and neither is the obvious choice.
    """

    import torch

    stop = set(int(value) for value in stop_ids)
    totals: list[float] = []
    per_token: list[float] = []
    with torch.no_grad():
        for start in range(0, int(generated.shape[0]), chunk):
            block = generated[start : start + chunk]
            logits = model(input_ids=block).logits.float()
            logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            targets = block[:, 1:]
            picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            for row in range(block.shape[0]):
                tail = block[row, prompt_length:].tolist()
                stopped = next((offset for offset, token in enumerate(tail) if token in stop), None)
                n_new = len(tail) if stopped is None else stopped + 1
                if n_new < 1:
                    raise ValueError("a draw produced no token to score")
                # Position ``prompt_length + j`` of the sequence is predicted by
                # column ``prompt_length + j - 1`` of the shifted logprobs.
                span = picked[row, prompt_length - 1 : prompt_length - 1 + n_new]
                total = float(span.sum())
                if not math.isfinite(total):
                    raise ValueError("a non-finite model log-probability was produced")
                totals.append(total)
                per_token.append(total / n_new)
    return {"total": totals, "per_token": per_token}


# ------------------------------------------------------- configuration census


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    points = (5, 25, 50, 75, 95)
    return {f"p{point}": float(np.percentile(array, point)) for point in points}


def length_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The full length distribution of a configuration's products.

    Reported for every configuration without exception, because the single most
    likely way this sweep produces a false positive is a configuration that
    merely generates longer sequences.
    """

    lengths = [int(row["length"]) for row in rows]
    if not lengths:
        raise ValueError("a length distribution needs at least one product")
    array = np.asarray(lengths, dtype=np.float64)
    low, high = EVALUATION_BAND
    return {
        "n": len(lengths),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if len(lengths) > 1 else 0.0,
        "min": int(array.min()),
        "max": int(array.max()),
        **_quantiles(lengths),
        "fraction_below_band": float(np.mean(array < low)),
        "fraction_above_band": float(np.mean(array > high)),
        "evaluation_band": [low, high],
    }


def config_census(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts, termination behaviour and token spend of one configuration.

    The termination table is a primary result and not a diagnostic: if most
    attempts never finished, most of the configuration's "products" are
    truncated continuations and their folding is a measurement of truncation.
    """

    if not rows:
        raise ValueError("a configuration census needs at least one attempt")
    tokens = np.asarray([int(row["n_new_tokens"]) for row in rows], dtype=np.float64)
    reasons = Counter(out_of_band_reason(str(row["sequence"])) or "in_band" for row in rows)
    return {
        "n_draws": len(rows),
        "n_clusters": len({str(row["cluster"]) for row in rows}),
        "n_in_band": int(sum(1 for row in rows if in_band(str(row["sequence"])))),
        "band_eligibility": {str(key): int(value) for key, value in sorted(reasons.items())},
        "fraction_in_band": float(
            sum(1 for row in rows if in_band(str(row["sequence"]))) / len(rows)
        ),
        "termination": {
            "n_terminated_natively": int(sum(1 for row in rows if row["terminated_natively"])),
            "fraction_terminated_natively": float(
                sum(1 for row in rows if row["terminated_natively"]) / len(rows)
            ),
            "n_budget_censored": int(
                sum(1 for row in rows if not row["terminated_natively"])
            ),
            "meaning": (
                "a natively terminated attempt emitted the arm's own end delimiter; a "
                "censored attempt was stopped by the token cap and its product is a "
                "truncated continuation"
            ),
        },
        "generation_tokens": {
            "total": int(tokens.sum()),
            "mean_per_draw": float(tokens.mean()),
            "definition": (
                "new tokens the model actually produced, counting the terminator and "
                "excluding padding. This is the compute currency of a decoding strategy"
            ),
        },
        "length_distribution_all_products": length_distribution(rows),
        "length_distribution_in_band": (
            length_distribution([row for row in rows if in_band(str(row["sequence"]))])
            if any(in_band(str(row["sequence"])) for row in rows)
            else None
        ),
    }


# ----------------------------------------------------------- degeneracy checks


def set_profile(
    sequences: Sequence[str], *, corpus_identity: Sequence[float] | None = None
) -> dict[str, Any]:
    """Duplication, complexity, repertoire and corpus identity of one set.

    The axes and their thresholds come from the project's own selected-set
    profile rather than being re-chosen here, so a collapse reads the same way
    in this sweep as in the selection experiments. The family repertoire is
    absent because no family oracle is run here; see
    :data:`CEILING` ``no_family_oracle_here``.
    """

    usable = [sequence for sequence in sequences if sequence]
    if not usable:
        raise ValueError("an empty set has no profile")
    lengths = np.asarray([len(sequence) for sequence in usable], dtype=np.float64)
    entropies = np.asarray(
        [
            -sum(
                share * math.log(share)
                for share in (
                    count / len(sequence) for count in Counter(sequence).values()
                )
                if share > 0.0
            )
            for sequence in usable
        ],
        dtype=np.float64,
    )
    runs = np.asarray([gp.longest_homopolymer_run(sequence) for sequence in usable])
    profile: dict[str, Any] = {
        "n_sequences": len(usable),
        "n_distinct_sequences": len(set(usable)),
        "duplicate_fraction": 1.0 - len(set(usable)) / len(usable),
        "mean_length": float(lengths.mean()),
        "mean_composition_entropy_nats": float(entropies.mean()),
        "min_composition_entropy_nats": float(entropies.min()),
        "longest_homopolymer_run_max": int(runs.max()),
        "homopolymer_run_threshold": int(gp.HOMOPOLYMER_RUN_THRESHOLD),
        "fraction_with_homopolymer_run": float(np.mean(runs >= gp.HOMOPOLYMER_RUN_THRESHOLD)),
        "mean_pairwise_kmer_distance": (
            gp.mean_pairwise_kmer_distance(usable, gp.DIVERSITY_KMER)
            if len(usable) >= 2
            else None
        ),
        "kmer": int(gp.DIVERSITY_KMER),
    }
    if corpus_identity is None:
        profile["nearest_corpus_identity"] = None
        profile["max_nearest_corpus_identity"] = None
        profile["fraction_near_duplicate_of_corpus"] = None
        profile["corpus_identity_note"] = "no homology search was supplied for this set"
    else:
        identity = np.asarray(corpus_identity, dtype=np.float64)
        if identity.size != len(usable):
            raise ValueError("the corpus identities must align with the sequences")
        profile["nearest_corpus_identity"] = float(identity.mean())
        profile["max_nearest_corpus_identity"] = float(identity.max())
        profile["fraction_near_duplicate_of_corpus"] = float(
            np.mean(identity >= gp.NEAR_DUPLICATE_IDENTITY)
        )
        profile["near_duplicate_identity_threshold"] = float(gp.NEAR_DUPLICATE_IDENTITY)
    return profile


def size_matched_reference(
    sequences: Sequence[str],
    *,
    size: int,
    seed: int,
    n_keys: int = 16,
    corpus_identity: Sequence[float] | None = None,
) -> dict[str, Any]:
    """The profile a random set of the same size has, as the honest reference.

    Every repertoire measure falls as a set shrinks, so a configuration's
    duplication and diversity can only be read against a draw of its own size.
    """

    if size < 2 or size > len(sequences):
        raise ValueError(
            f"a size-matched reference needs 2 <= size <= {len(sequences)}, got {size}"
        )
    generator = np.random.default_rng(seed)
    profiles = []
    for _ in range(n_keys):
        chosen = generator.choice(len(sequences), size=size, replace=False)
        profiles.append(
            set_profile(
                [sequences[index] for index in chosen],
                corpus_identity=(
                    None
                    if corpus_identity is None
                    else [corpus_identity[index] for index in chosen]
                ),
            )
        )
    summary: dict[str, Any] = {
        "n_keys": int(n_keys),
        "size": int(size),
        "band": (
            "``span`` is the range a random draw of this size actually realised "
            "across the keys, and it is what a configuration's single realised "
            "value is flagged against. ``interval`` is a confidence interval for "
            "the random draw's *mean*: it is far narrower, it shrinks as keys are "
            "added, and flagging against it would report ordinary sampling noise "
            "between two configurations as a collapse"
        ),
    }
    for field in (
        "duplicate_fraction",
        "mean_composition_entropy_nats",
        "fraction_with_homopolymer_run",
        "mean_pairwise_kmer_distance",
        "mean_length",
        "nearest_corpus_identity",
        "fraction_near_duplicate_of_corpus",
    ):
        values = [profile[field] for profile in profiles if profile[field] is not None]
        summary[field] = (
            {
                "mean": float(np.mean(values)),
                "span": [float(np.min(values)), float(np.max(values))],
                "interval": mean_interval(values)["interval"],
            }
            if len(values) >= 2
            else None
        )
    return summary


#: The axes a configuration is checked for degeneracy on, and the direction that
#: counts as degenerate. The same table the selection experiments use, so a
#: collapse means the same thing in both places.
DEGENERACY_AXES: dict[str, str] = {
    "duplicate_fraction": "above",
    "mean_pairwise_kmer_distance": "below",
    "mean_composition_entropy_nats": "below",
    "fraction_with_homopolymer_run": "above",
    "nearest_corpus_identity": "above",
    "fraction_near_duplicate_of_corpus": "above",
}


def degeneracy_verdict(
    profile: Mapping[str, Any], reference: Mapping[str, Any], *, gain: float | None
) -> dict[str, Any]:
    """Whether a configuration's apparent gain was bought with a collapse.

    An axis is flagged when the configuration falls outside the range a
    size-matched random draw actually realised, in the degenerate direction. The
    realised range and not a confidence interval for its mean: the comparison is
    with one configuration's single realised value, and a confidence interval
    would flag ordinary sampling noise between two healthy configurations. A
    positive gain with any axis flagged is reported as ``gain_is_not_a_gain``:
    better-folding candidates drawn from a narrower, more repetitive or more
    retrieved repertoire are a different product, not a better one.
    """

    flagged: dict[str, Any] = {}
    for axis, direction in DEGENERACY_AXES.items():
        observed = profile.get(axis)
        band = reference.get(axis)
        if observed is None or band is None:
            continue
        low, high = band["span"]
        if direction == "below" and observed < low:
            flagged[axis] = {"observed": observed, "reference_span": [low, high], "moved": "below"}
        elif direction == "above" and observed > high:
            flagged[axis] = {"observed": observed, "reference_span": [low, high], "moved": "above"}
    improved = gain is not None and gain > 0.0
    return {
        "axes_checked": sorted(DEGENERACY_AXES),
        "axes_flagged": flagged,
        "degenerate": bool(flagged),
        "improved": bool(improved),
        "gain_is_not_a_gain": bool(improved and flagged),
    }


# ----------------------------------------------------------- the natural band


#: What the structure instrument returns, and the direction each reads in. Every
#: one is the predictor's confidence about coordinates; none is stability or
#: function. ``neg_mean_pae_angstrom`` is negated so that larger is better on
#: every axis and a single comparison direction is correct everywhere.
EVALUATORS: dict[str, str] = {
    "mean_ca_plddt": "mean CA pLDDT on the 0-100 scale; higher is more confident",
    "fraction_ca_plddt_ge70": "share of residues at CA pLDDT 70 or above",
    "ptm": "predicted TM-score of the retained diffusion sample",
    "neg_mean_pae_angstrom": (
        "negated mean predicted aligned error in angstrom, read off the retained "
        "sample's full PAE matrix, which is kept on disk rather than reduced away"
    ),
    "predicted_confidence_event": (
        "the project's operational event: mean CA pLDDT >= 70 and at least 80% of "
        "residues at 70 or above. An operational prediction, not a folded protein"
    ),
}
PRIMARY_EVALUATOR = "mean_ca_plddt"

#: Width of the length strata the natural band is standardised over, and the
#: fewest natural records a stratum needs before a configuration's mass may be
#: compared inside it. Twenty residues is narrow enough that confidence varies
#: little inside a stratum and wide enough that the 1,822 folded natural records
#: populate it.
NATURAL_BIN_WIDTH: int = 20
MIN_NATURAL_PER_BIN: int = 8
#: The most length mass a configuration may have in strata without natural
#: support before its standardised gap is reported as unresolved.
MAX_UNSUPPORTED_MASS: float = 0.05

#: The paired whole-natural comparator draws without replacement, so it is formed
#: on a cluster-stratified subsample of this size rather than on a deep
#: configuration's full 512 draws, which the 1,822 folded natural records cannot
#: match one-to-one inside a narrow length window.
PAIRED_CONTRAST_CAP: int = 64
PAIRED_TOLERANCE_RESIDUES: int = 6
MIN_PAIRED_MATCH_RATE: float = 0.80


def k_values(draws_per_cluster: int) -> tuple[int, ...]:
    """The best-of-k budgets that divide a cluster's draws into whole blocks."""

    values = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= draws_per_cluster and draws_per_cluster % k == 0]
    if not values:
        raise ValueError(f"{draws_per_cluster} draws per cluster admit no block structure")
    return tuple(values)


def lift_structure(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """One folded row flattened onto the evaluator axes, or None if not folded.

    The pairwise fields are not reduced away: ``mean_pae_angstrom`` is a summary
    of a PAE matrix the structure instrument keeps on disk beside the per-residue
    pLDDT, and this function only lifts the summaries an interval is taken over.
    """

    structure = row.get("structure")
    if not isinstance(structure, Mapping) or structure.get("status") != "ok":
        return None
    lifted = {
        "mean_ca_plddt": float(structure["mean_ca_plddt"]),
        "fraction_ca_plddt_ge70": float(structure["fraction_ca_plddt_ge70"]),
        "ptm": float(structure["ptm"]),
        "neg_mean_pae_angstrom": -float(structure["mean_pae_angstrom"]),
        "predicted_confidence_event": float(bool(structure["predicted_confidence_event"])),
        "structure_object_directory": structure.get("object_directory"),
        "diffusion_sample_index": structure.get("diffusion_sample_index"),
        "evaluation_signature": structure.get("evaluation_signature"),
    }
    return lifted


def capped_by_cluster(
    rows: Sequence[Mapping[str, Any]], *, cap: int, seed: int
) -> list[Mapping[str, Any]]:
    """At most ``cap`` rows, spread evenly over the clusters present.

    Stratified rather than uniform, so a subsample taken for the paired
    comparator keeps the cluster structure the interval's unit is defined on.
    """

    if cap < 1:
        raise ValueError("a cap is positive")
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row["cluster"]), []).append(row)
    if not buckets:
        return []
    generator = np.random.default_rng(seed)
    per_cluster = max(1, cap // len(buckets))
    chosen: list[Mapping[str, Any]] = []
    for key in sorted(buckets):
        items = sorted(buckets[key], key=lambda row: str(row["id"]))
        take = min(per_cluster, len(items))
        picked = generator.choice(len(items), size=take, replace=False)
        chosen.extend(items[int(index)] for index in sorted(picked))
    return chosen[:cap]


def _length_stratum(length: int) -> int:
    low, _ = EVALUATION_BAND
    return int((int(length) - low) // NATURAL_BIN_WIDTH)


def natural_standardised_band(
    candidates: Sequence[Mapping[str, Any]],
    natural: Sequence[Mapping[str, Any]],
    *,
    field: str,
    seed: int,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """A configuration minus the natural band standardised to its own lengths.

    Direct standardisation, not pairing: the natural mean is computed inside each
    length stratum and re-weighted by the configuration's own mass in that
    stratum. It uses the whole folded natural pool instead of consuming it one
    record at a time, so a configuration whose products concentrate in a narrow
    length window still gets a comparator -- which the without-replacement paired
    comparator cannot give it.

    A stratum the configuration occupies but the natural pool barely populates is
    dropped and its mass reported: both sides are then renormalised over the same
    supported strata, so the gap is a contrast at matched length and not a
    contrast between two different length mixtures.
    """

    if not candidates:
        raise ValueError("a standardised band needs at least one in-band candidate")
    natural_by_bin: dict[int, list[float]] = {}
    for row in natural:
        natural_by_bin.setdefault(_length_stratum(int(row["length"])), []).append(float(row[field]))
    config_by_bin: dict[int, list[tuple[str, float]]] = {}
    for row in candidates:
        config_by_bin.setdefault(_length_stratum(int(row["length"])), []).append(
            (str(row["cluster"]), float(row[field]))
        )
    total = len(candidates)
    supported = sorted(
        key
        for key in config_by_bin
        if len(natural_by_bin.get(key, ())) >= MIN_NATURAL_PER_BIN
    )
    unsupported = sorted(set(config_by_bin) - set(supported))
    unsupported_mass = sum(len(config_by_bin[key]) for key in unsupported) / total
    report: dict[str, Any] = {
        "field": field,
        "bin_width_residues": NATURAL_BIN_WIDTH,
        "min_natural_per_bin": MIN_NATURAL_PER_BIN,
        "n_candidates": total,
        "n_natural": len(natural),
        "strata_supported": [int(key) for key in supported],
        "strata_unsupported": [int(key) for key in unsupported],
        "unsupported_length_mass": float(unsupported_mass),
        "estimand": (
            "the configuration's mean minus the folded natural pool's mean, both "
            "re-weighted to the configuration's own length distribution over the "
            "supported strata. Direct standardisation, so no natural record is "
            "consumed and a concentrated configuration still has a comparator"
        ),
    }
    if not supported or unsupported_mass > MAX_UNSUPPORTED_MASS:
        report.update(
            {
                "resolved": False,
                "unresolved_reason": (
                    f"{unsupported_mass:.3f} of this configuration's length mass falls in "
                    f"strata with fewer than {MIN_NATURAL_PER_BIN} folded natural records, "
                    f"above the declared {MAX_UNSUPPORTED_MASS:.2f} ceiling. The contrast "
                    "at matched length is not identifiable here and is not forced"
                ),
                "configuration_mean": None,
                "natural_band": None,
                "gap": None,
                "gap_interval": None,
            }
        )
        return report

    weights = {key: len(config_by_bin[key]) for key in supported}
    mass = sum(weights.values())
    clusters = sorted({cluster for key in supported for cluster, _ in config_by_bin[key]})
    floor = bootstrap_unit_floor(len(clusters))

    def evaluate(
        config_sample: Mapping[int, Sequence[float]],
        natural_sample: Mapping[int, Sequence[float]],
        bin_mass: Mapping[int, float],
    ) -> tuple[float, float]:
        """Both sides re-weighted over the strata this sample actually occupies.

        A stratum a resample happens to leave empty carries zero weight and is
        dropped from both sides together, rather than invalidating the draw. The
        alternative -- discarding any resample with an empty stratum -- throws
        away most draws whenever the strata are fine relative to the cluster
        count, and conditions the interval on the resamples that happened to be
        easy.
        """

        left = right = 0.0
        denominator = 0.0
        for key, weight in bin_mass.items():
            values = config_sample.get(key, ())
            reference = natural_sample.get(key, ())
            if weight <= 0 or not len(values) or not len(reference):
                continue
            left += weight * float(np.mean(values))
            right += weight * float(np.mean(reference))
            denominator += weight
        if denominator <= 0:
            return float("nan"), float("nan")
        return left / denominator, right / denominator

    point_config, point_natural = evaluate(
        {key: [value for _, value in config_by_bin[key]] for key in supported},
        {key: natural_by_bin[key] for key in supported},
        weights,
    )
    generator = np.random.default_rng(seed)
    gaps: list[float] = []
    for _ in range(n_bootstrap):
        picked = set(generator.choice(clusters, size=len(clusters), replace=True).tolist())
        config_sample = {
            key: [value for cluster, value in config_by_bin[key] if cluster in picked]
            for key in supported
        }
        bin_mass = {key: float(len(config_sample[key])) for key in supported}
        natural_sample = {
            key: list(
                np.asarray(natural_by_bin[key])[
                    generator.integers(0, len(natural_by_bin[key]), size=len(natural_by_bin[key]))
                ]
            )
            for key in supported
        }
        left, right = evaluate(config_sample, natural_sample, bin_mass)
        if math.isfinite(left) and math.isfinite(right):
            gaps.append(left - right)
    resolved = not floor["degenerate"] and len(gaps) >= n_bootstrap * 0.9
    report.update(
        {
            "unresolved_reason": (
                None
                if resolved
                else (
                    floor["degenerate_reason"]
                    if floor["degenerate"]
                    else (
                        f"only {len(gaps)} of {n_bootstrap} resamples left every supported "
                        "stratum populated on both sides, so the gap has no interval here"
                    )
                )
            ),
            "n_clusters": len(clusters),
            "unit": "cluster on the configuration side, record on the natural side",
            "unit_floor": floor,
            "supported_length_mass": float(mass / total),
            "configuration_mean": float(point_config),
            "natural_band": float(point_natural),
            "gap": float(point_config - point_natural),
            "gap_interval": (
                [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]
                if resolved
                else None
            ),
            "n_finite_draws": len(gaps),
            "resolved": bool(resolved),
            "reaches_natural_band": (
                bool(np.percentile(gaps, 2.5) > 0.0) if resolved else None
            ),
        }
    )
    return report


def natural_band_contrast(
    candidates: Sequence[Mapping[str, Any]],
    natural: Sequence[Mapping[str, Any]],
    *,
    field: str,
    seed: int,
    tolerance: int = PAIRED_TOLERANCE_RESIDUES,
    cap: int = PAIRED_CONTRAST_CAP,
    n_bootstrap: int = 10000,
) -> dict[str, Any]:
    """A configuration minus its paired whole-natural comparator, by length.

    This is the project's own comparator: whole Swiss-Prot entries, never
    fragments of them, already folded by this project's ESMFold2 instrument at
    one evaluation signature, drawn **without replacement** and matched on
    length. Without replacement is what makes it conservative and also what
    limits it: 1,822 folded natural records cannot one-to-one match a deep
    configuration's 512 draws inside a narrow length window, so the contrast is
    formed on a cluster-stratified subsample of ``cap`` candidates and the
    realised match rate is reported. Below :data:`MIN_PAIRED_MATCH_RATE` the
    contrast is declared unresolved rather than read off whichever candidates
    happened to be matchable, which would be a contrast on the configuration's
    least typical lengths. :func:`natural_standardised_band` is the estimator
    that has no such ceiling.

    A natural record may price a candidate in more than one configuration, which
    makes the configurations' gaps correlated; this interval is marginal.
    """

    sample = capped_by_cluster(candidates, cap=cap, seed=seed)
    pairs, report = gp.match_natural_records(
        list(sample), list(natural), tolerance=tolerance, seed=seed
    )
    rate = report["n_matched"] / report["n_targets"]
    record: dict[str, Any] = {
        "field": field,
        "cap": int(cap),
        "n_candidates_available": len(candidates),
        "match_rate": float(rate),
        "minimum_match_rate": float(MIN_PAIRED_MATCH_RATE),
        "matching": report,
        "interval_is_marginal": (
            "a marginal 95% interval for this configuration alone. Simultaneous "
            "statements across configurations come from the joint cluster bootstrap"
        ),
    }
    if rate < MIN_PAIRED_MATCH_RATE:
        record.update(
            {
                "resolved": False,
                "unresolved_reason": (
                    f"only {report['n_matched']} of {report['n_targets']} candidates found a "
                    f"whole natural record within {tolerance} residues. The paired contrast "
                    "is not identifiable at this configuration's length concentration"
                ),
                "contrast": None,
            }
        )
        return record
    left = [float(pair["target"][field]) for pair in pairs]
    right = [float(pair["natural"][field]) for pair in pairs]
    groups = [str(pair["target"]["cluster"]) for pair in pairs]
    contrast = gp.matched_contrast(left, right, groups, seed=seed, n_bootstrap=n_bootstrap)
    record.update({"resolved": bool(contrast["resolved"]), "contrast": contrast})
    return record


def cluster_means(
    rows: Sequence[Mapping[str, Any]], *, field: str
) -> dict[str, float]:
    """One value per cluster: the unit every configuration-level interval uses."""

    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(str(row["cluster"]), []).append(float(row[field]))
    return {key: float(np.mean(values)) for key, values in sorted(buckets.items())}


def cluster_level_mean(rows: Sequence[Mapping[str, Any]], *, field: str) -> dict[str, Any]:
    """A configuration's mean with its interval taken over clusters.

    The cluster -- a frozen class, or an independent seed block -- is the unit
    because draws inside one cluster share a prompt and are exchangeable only
    given it. The candidate-level interval is reported beside it and is the
    narrower, less conservative reading; both carry their own ``n``.
    """

    if not rows:
        raise ValueError("a configuration-level mean needs at least one row")
    per_cluster = cluster_means(rows, field=field)
    floor = bootstrap_unit_floor(len(per_cluster))
    values = [float(row[field]) for row in rows]
    record: dict[str, Any] = {
        "field": field,
        "n_draws": len(values),
        "n_clusters": len(per_cluster),
        "unit": "cluster",
        "unit_floor": floor,
        "mean": float(np.mean(list(per_cluster.values()))),
        "candidate_level": mean_interval(values) if len(values) >= 2 else None,
    }
    record["cluster_level"] = (
        mean_interval(list(per_cluster.values())) if len(per_cluster) >= 2 else None
    )
    record["resolved"] = not floor["degenerate"]
    return record


def joint_cluster_bootstrap(
    per_config: Mapping[str, Mapping[str, float]],
    *,
    seed: int,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Marginal and simultaneous statements over one arm's configurations.

    Every configuration of an arm is measured on the *same* sixteen clusters, so
    one resample of the cluster set re-scores all of them at once. That makes a
    panel-wide claim -- "no configuration reaches the natural band" -- available
    as a statement about the maximum across configurations rather than as a pile
    of marginal intervals, which are not a simultaneous statement.

    ``per_config`` maps a configuration key to its per-cluster values. A
    configuration with fewer than the package's unit floor of clusters is
    reported as unresolved and excluded from the maximum, which is recorded.
    """

    clusters = sorted({key for values in per_config.values() for key in values})
    if len(clusters) < 2:
        raise ValueError("a cluster bootstrap needs at least two clusters")
    eligible = {
        name: values
        for name, values in per_config.items()
        if not bootstrap_unit_floor(len(values))["degenerate"]
    }
    excluded = sorted(set(per_config) - set(eligible))
    generator = np.random.default_rng(seed)
    index = {name: index for index, name in enumerate(clusters)}
    matrix = np.full((len(per_config), len(clusters)), np.nan)
    order = sorted(per_config)
    for row, name in enumerate(order):
        for cluster, value in per_config[name].items():
            matrix[row, index[cluster]] = value
    draws = np.full((n_bootstrap, len(order)), np.nan)
    with warnings.catch_warnings():
        # A resample can miss every cluster a configuration has data in, which
        # is an all-NaN slice and a legitimate undefined draw, not a defect.
        warnings.simplefilter("ignore", RuntimeWarning)
        for draw in range(n_bootstrap):
            picked = generator.integers(0, len(clusters), size=len(clusters))
            draws[draw] = np.nanmean(matrix[:, picked], axis=1)
    tail = (1.0 - confidence) / 2.0
    marginal: dict[str, Any] = {}
    for row, name in enumerate(order):
        column = draws[:, row]
        finite = column[np.isfinite(column)]
        resolved = name in eligible and finite.size >= n_bootstrap * 0.95
        marginal[name] = {
            "n_clusters": len(per_config[name]),
            "mean": float(np.nanmean(matrix[row])),
            "interval": (
                [float(np.percentile(finite, 100 * tail)), float(np.percentile(finite, 100 * (1 - tail)))]
                if resolved
                else None
            ),
            "n_finite_draws": int(finite.size),
            "resolved": bool(resolved),
        }
    rows_eligible = [row for row, name in enumerate(order) if name in eligible]
    simultaneous: dict[str, Any] = {
        "n_configurations": len(rows_eligible),
        "excluded_configurations": excluded,
        "confidence": float(confidence),
    }
    if rows_eligible:
        maxima = np.nanmax(draws[:, rows_eligible], axis=1)
        finite = maxima[np.isfinite(maxima)]
        simultaneous["max_over_configurations"] = {
            "point": float(np.nanmax([np.nanmean(matrix[row]) for row in rows_eligible])),
            "upper_bound": float(np.percentile(finite, 100 * (1 - tail))),
            "n_finite_draws": int(finite.size),
            "meaning": (
                "the upper end of a one-sided 97.5% bound on the largest value any "
                "eligible configuration attains. A panel-wide claim is read off this, "
                "not off the marginal intervals"
            ),
        }
    return {
        "n_clusters": len(clusters),
        "n_bootstrap": int(n_bootstrap),
        "marginal": marginal,
        "simultaneous": simultaneous,
    }


# ----------------------------------------------- length-standardised contrast


def length_standardised(
    per_config_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    field: str,
    n_bins: int = gp.LENGTH_MATCH_BINS,
    seed: int,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """Each configuration's mean re-weighted to one shared length distribution.

    The length-matched counterpart of the configuration-versus-configuration
    comparison. Bins are equal-count over the pooled in-band products of the arm,
    so every configuration is read at the same length mixture and a configuration
    that wins only by being longer cannot win here. A configuration that leaves a
    bin empty has no standardised mean at all, which is reported rather than
    filled by extrapolation.
    """

    pooled: list[int] = []
    for rows in per_config_rows.values():
        pooled.extend(int(row["length"]) for row in rows)
    if len(pooled) < n_bins:
        raise ValueError(f"length standardisation needs at least {n_bins} pooled products")
    bins = gp.length_bins(pooled, n_bins=n_bins)
    edges = np.asarray(bins["edges"], dtype=np.float64)
    assignment = np.asarray(bins["bin_of_row"])
    weights = {
        int(key): count / len(pooled) for key, count in Counter(assignment.tolist()).items()
    }
    generator = np.random.default_rng(seed)
    report: dict[str, Any] = {
        "field": field,
        "n_bins_requested": int(n_bins),
        "n_bins_realised": int(bins["n_bins_realised"]),
        "bin_edges": [float(edge) for edge in edges],
        "bin_weights": {str(key): float(value) for key, value in sorted(weights.items())},
        "weighting": "equal-count length bins over the pooled in-band products of this arm",
        "configurations": {},
    }
    for name, rows in sorted(per_config_rows.items()):
        lengths = np.asarray([int(row["length"]) for row in rows], dtype=np.float64)
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        clusters = np.asarray([str(row["cluster"]) for row in rows])
        row_bin = np.searchsorted(edges, lengths, side="right")
        empty = sorted(key for key in weights if not np.any(row_bin == key))

        def standardise(mask: np.ndarray) -> float:
            total = 0.0
            for key, weight in weights.items():
                selected = values[mask & (row_bin == key)]
                if selected.size == 0:
                    return float("nan")
                total += weight * float(selected.mean())
            return total

        point = standardise(np.ones(values.size, dtype=bool))
        unique = sorted(set(clusters.tolist()))
        floor = bootstrap_unit_floor(len(unique))
        interval = None
        if not floor["degenerate"] and math.isfinite(point):
            samples = []
            for _ in range(n_bootstrap):
                picked = generator.choice(unique, size=len(unique), replace=True)
                mask = np.isin(clusters, picked)
                value = standardise(mask)
                if math.isfinite(value):
                    samples.append(value)
            if len(samples) >= n_bootstrap * 0.95:
                interval = [
                    float(np.percentile(samples, 2.5)),
                    float(np.percentile(samples, 97.5)),
                ]
        report["configurations"][name] = {
            "n_in_band": int(values.size),
            "n_clusters": len(unique),
            "raw_mean": float(values.mean()) if values.size else None,
            "standardised_mean": None if not math.isfinite(point) else float(point),
            "standardised_interval": interval,
            "empty_bins": [str(key) for key in empty],
            "unit_floor": floor,
            "resolved": bool(interval is not None),
        }
    return report


# --------------------------------------------------------- the budget axis


#: The selectors a best-of-k strategy may use here, and what each costs.
SELECTORS: dict[str, dict[str, str]] = {
    "model_logprob_per_token": {
        "kind": "realistic",
        "cost": "free at generation time; one fold for the selected candidate only",
        "note": (
            "the model's own mean per-token log-probability of the product under the "
            "untempered, untruncated distribution, so the score does not change with "
            "the operating point that drew the product"
        ),
    },
    "model_logprob_total": {
        "kind": "realistic",
        "cost": "free at generation time; one fold for the selected candidate only",
        "note": "the same quantity summed rather than averaged, which ranks long products higher",
    },
    "esmfold2_confidence_oracle": {
        "kind": "upper_bound",
        "cost": "one fold per candidate drawn",
        "note": (
            "selects on the evaluator itself. Reported as an attainable upper bound on "
            "best-of-k and never as a decoding strategy"
        ),
    },
}


def selection_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    selector: str,
    field: str,
    k_values: Sequence[int],
    draws_per_cluster: int,
) -> list[dict[str, Any]]:
    """Best-of-k within a cluster, with the compute each point spent.

    Blocks are consecutive draws *inside* a cluster, so a conditioned arm's
    best-of-k never trades a hard class for an easy one -- which pooled blocks
    would let it do, and which would read as a decoding gain. A block whose k
    draws contain no in-band product selects nothing: the block still spent its
    tokens, so it is counted in ``n_blocks`` and in the token cost and reported
    as a yield rather than dropped.

    ``rows`` must be *all* draws of the configuration, in-band or not, each
    carrying ``cluster``, ``draw_index``, ``n_new_tokens`` and the selector
    field; only in-band rows need the evaluator field.
    """

    if selector not in SELECTORS:
        raise KeyError(f"unknown selector {selector!r}; declared: {sorted(SELECTORS)}")
    by_cluster: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_cluster.setdefault(str(row["cluster"]), []).append(row)
    for cluster, items in by_cluster.items():
        items.sort(key=lambda row: int(row["draw_index"]))
        if len(items) != draws_per_cluster:
            raise ValueError(
                f"cluster {cluster} holds {len(items)} draws, not the declared "
                f"{draws_per_cluster}; a best-of-k block structure cannot be formed"
            )
    curve: list[dict[str, Any]] = []
    for k in k_values:
        if k < 1 or draws_per_cluster % k:
            raise ValueError(
                f"k={k} does not divide the {draws_per_cluster} draws per cluster into "
                "whole blocks"
            )
        selected: list[dict[str, Any]] = []
        n_blocks = 0
        empty_blocks = 0
        tokens: list[int] = []
        folds: list[int] = []
        for items in by_cluster.values():
            for start in range(0, len(items), k):
                block = items[start : start + k]
                n_blocks += 1
                tokens.append(sum(int(row["n_new_tokens"]) for row in block))
                eligible = [row for row in block if in_band(str(row["sequence"]))]
                if selector == "esmfold2_confidence_oracle":
                    folds.append(len(eligible))
                else:
                    folds.append(1 if eligible else 0)
                if not eligible:
                    empty_blocks += 1
                    continue
                score = field if selector == "esmfold2_confidence_oracle" else selector
                winner = max(eligible, key=lambda row: (float(row[score]), str(row["id"])))
                selected.append(dict(winner))
        values = [float(row[field]) for row in selected]
        floor = bootstrap_unit_floor(len(values))
        curve.append(
            {
                "k": int(k),
                "selector": selector,
                "field": field,
                "n_blocks": n_blocks,
                "n_blocks_with_candidate": len(values),
                "block_yield": float(len(values) / n_blocks) if n_blocks else 0.0,
                "n_blocks_empty": empty_blocks,
                "unit": "selection block",
                "unit_floor": floor,
                "mean": float(np.mean(values)) if values else None,
                "interval": (
                    mean_interval(values)["interval"]
                    if len(values) >= 2 and not floor["degenerate"]
                    else None
                ),
                "resolved": bool(len(values) >= 2 and not floor["degenerate"]),
                "generation_tokens_per_block": float(np.mean(tokens)) if tokens else None,
                "folds_per_block": float(np.mean(folds)) if folds else None,
                "selected_length_mean": (
                    float(np.mean([int(row["length"]) for row in selected])) if selected else None
                ),
                "selected_ids": [str(row["id"]) for row in selected],
            }
        )
    return curve


def matched_compute_table(
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    budgets: Sequence[float],
    tolerance: float = 0.35,
) -> list[dict[str, Any]]:
    """The sweep re-read at equal total generation compute.

    A configuration enters a budget at the k whose realised token cost is nearest
    that budget, and only when the realised cost is within ``tolerance`` of it: a
    configuration whose grid of k cannot reach a budget is listed as absent at
    that budget rather than compared at a different cost. Without this table the
    comparison is at equal candidate count, where drawing more candidates wins by
    construction.
    """

    table: list[dict[str, Any]] = []
    for budget in budgets:
        entries: list[dict[str, Any]] = []
        absent: list[str] = []
        for name, curve in sorted(curves.items()):
            feasible = [
                point
                for point in curve
                if point["generation_tokens_per_block"] is not None
                and abs(point["generation_tokens_per_block"] - budget) <= tolerance * budget
            ]
            if not feasible:
                absent.append(name)
                continue
            best = min(feasible, key=lambda point: abs(point["generation_tokens_per_block"] - budget))
            entries.append(
                {
                    "configuration": name,
                    "k": best["k"],
                    "generation_tokens_per_block": best["generation_tokens_per_block"],
                    "realised_over_budget": float(
                        best["generation_tokens_per_block"] / budget
                    ),
                    "mean": best["mean"],
                    "interval": best["interval"],
                    "n_blocks_with_candidate": best["n_blocks_with_candidate"],
                    "block_yield": best["block_yield"],
                    "folds_per_block": best["folds_per_block"],
                    "selected_length_mean": best["selected_length_mean"],
                    "resolved": best["resolved"],
                }
            )
        ranked = [entry for entry in entries if entry["mean"] is not None]
        ranked.sort(key=lambda entry: -float(entry["mean"]))
        table.append(
            {
                "generation_tokens_per_selected_candidate": float(budget),
                "tolerance_fraction": float(tolerance),
                "entries": entries,
                "configurations_absent": absent,
                "best_configuration": ranked[0]["configuration"] if ranked else None,
                "estimand": (
                    "the mean evaluator value of the candidate a strategy keeps when it "
                    "is allowed this many generated tokens per kept candidate. Compare "
                    "rows within one budget; across budgets the compute differs"
                ),
            }
        )
    return table


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path} line {number} is not a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} carries no record")
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    Path(path).write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()
