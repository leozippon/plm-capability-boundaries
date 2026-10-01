#!/usr/bin/env python3
"""Is one joint language-protein checkpoint measurable in BOTH of its modes?

This stage asks which differences remain when text and protein modes share one
checkpoint, and it admits nothing until **one dense joint decoder is qualified in
both modes**. This stage is that qualification, and it is an external baseline
rather than a panel measurement: ``--checkpoint`` names a directory, not an arm,
because a checkpoint that has not passed this stage must not be in ``arms.py`` at
all.

**Estimand, one per mode.** Context information: the held-out unigram
cross-entropy on the scored symbols minus the model's own clean cross-entropy on
the same symbols. It is stage 01's estimand deliberately -- a joint checkpoint's
qualification has to be commensurable with the panel's -- and it carries stage
01's reading with it: a mode below the floor is **unmeasurable on that cohort**,
not failing. That distinction is the whole verdict. It says the evaluation
interface cannot resolve anything on this cohort, and it says nothing about what
the checkpoint encodes.

**The published verdict is a pre-interval screen, not the identification
criterion.** The floor is
:data:`src.capability.core.budget.SCREENING_CONTEXT_INFORMATION_NATS` -- the point rule
EXP-R2-218 calibrated against a known-zero null family at FPR <= 0.05 -- and it
answers "does this reading justify an interval". Identification itself is
:func:`src.capability.core.budget.context_identification`, which reads the mode's
displacement-corrected bootstrap interval; this stage computes no bootstrap, so
it persists the per-record sufficient statistics into ``records/`` and
``context_information_bootstrap.py`` takes the verdict from them. The
difference is not hypothetical: on the EXP-R2-220 cells the screen refuses
``galactica-1.3b``'s protein mode at +0.047678 and the criterion identifies it,
its corrected interval reaching down only to +0.038694 (EXP-R2-221, §5.10). The 0.30 nats this
stage used to gate on was never derived; it is now an inert column
(``legacy_qualification_floor_nats``, ``clears_legacy_qualification_floor``)
that admits and refuses nothing, declared once as
:data:`src.capability.core.budget.MIN_CONTEXT_INFORMATION_NATS` and no longer restated
here. Old and new artefacts are not confusable: an artefact written before the
change carries ``modes[*].verdict`` and ``verdicts`` decided at 0.30; one
written after carries ``modes[*].identification_verdict`` and
``identification_verdicts`` decided at 0.05, and no key means two things
across the pair.

**What "this mode is not worth a behavioural read" rests on is the reversal
cost, and it is not a threshold.** Every protein record already carries
``controls.reversed.cost_nats_per_residue``, and that -- not a context-information
floor -- is the substantive evidence: ``Llama-2-7b-hf`` pays **-0.0013**
nats/residue to have a sequence reversed, which is indifference to reading
direction, against **+0.1442** for the adapted stage.
:data:`src.capability.concept_alignment.PROTEIN_MODE_BEHAVIOURAL_STATUS` already
states the refusal in exactly those terms. The cost is **reported and never
gated**, because pairing every published protein reading with its own reversal
cost shows the two orderings cross: the refused chimera families
``s1Body_baseVocab`` and ``s1Body_baseHead`` read -0.061 to -0.085 nats of
context information at reversal costs of +0.119 to +0.138, overlapping the
+0.136 to +0.167 of the admitted ProLLaMA-lineage cells -- ``s1Body_baseHead``
on the second draw pays +0.1379 and is refused while ``s1Body_baseEmbed`` pays
+0.1358 and is admitted -- while ``galactica-125m`` reads -0.139 at +0.088. A
reversal-cost criterion would be a **different partition**, not a tightening of
this one, and would change admissions in both directions. Both quantities are
published; neither is turned into a second gate.

**Per-record sufficient statistics are persisted, so uncertainty is a CPU job.**
Beside the report, each measured mode writes ``records/power_<cohort>_<digest>``
``.records.npz`` in ``budget.write_power_records``'s format, with the frozen
scored cohort and the frozen held-out reference beside it. It holds the
per-record clean-NLL sums, token counts, symbol counts and sparse target counts
of the declared **and** the reversed condition, which is everything a
group-clustered bootstrap of this stage's own point estimates needs. This stage
publishes point estimates with no interval, and that is the defect the sidecar
exists to let a later analysis close without a second GPU sweep.

**What one scored symbol is, is the family's declaration.** Galactica and
InstructProtein reach a per-residue alphabet, so a protein symbol is a residue and
the magnitude is in nats per residue. ProLLaMA does not -- ``Seq=<...>`` over the
unmodified LLaMA-2 vocabulary merges residue runs, and that merged form is what
the lineage was trained on -- so its protein symbol is a token. The estimand
follows the declared unit rather than the other way round: the unigram support,
the clean cross-entropy's denominator and the reported unit all move together.

That mobility is exactly why every protein record carries its **measured residues
per scored token** and an explicit cross-arm comparability verdict. A magnitude in
nats per token is not commensurable with one in nats per residue, and it is not
commensurable with another arm's nats per token either when the arms differ in
residues per token (Appendix B rule 26, limitation L23). The artefact says so in
its own field rather than leaving a reader to notice.

**The unigram is held out and its support is declared.** It is fitted on a draw
that is disjoint from the scored one by construction (a later window of the same
seeded permutation) and then deduplicated by content, because Swiss-Prot carries
the same sequence under several accessions (Appendix B rule 3). Its support is
always the set of ids a scored target of that mode can take -- and that set is
read from ``joint_modes``, so it follows the declared symbol unit: the twenty
residue ids for a residue-unit family, every residue-spelling token (463 on the
staged LLaMA-2 vocabulary) for a token-unit one, and the model's whole vocabulary
for text. Laplace smoothing inflates a cross-entropy by roughly ``log(1 + s*V/N)``,
and on the 41682-target ProLLaMA reference this stage's defaults draw that is
0.0005 nats over twenty symbols, 0.011 over 463 and 0.570 over the full 32000 --
so fitting a protein reference over the whole vocabulary would put more smoothing
bias into the baseline than the effect being measured.

**Controls, because the headline number alone cannot be read.**

``reversed``      the same sequences scored backwards under the same rendering. A
                  decoder carrying real directional sequence structure cannot be
                  indifferent to reversal, so a small reversal cost is what
                  distinguishes "this cohort is hard" from "this mode is not
                  reading the sequence at all". It is a within-arm difference over
                  an identical residue multiset, which is why it survives the unit
                  problem above and why the reading of a protein mode rests on it
                  rather than on the floor: the cost is read **per residue** in
                  both units. It is reported, never gated -- see above. For a token-unit
                  family that is also the only readable form -- reversal permutes
                  the residues but not the tokenisation, so the reversed condition
                  is a different token population and neither a per-token cost nor
                  the forward reference applies to it.
``naive``         the same sequences under the rendering with the per-residue
                  escape removed -- what an unaided ``AutoTokenizer`` call
                  produces. It prices the rendering (Appendix B rule 4) in the
                  estimand's own per-residue units, and it is the one measurement
                  in this stage that is deliberately not verified against the
                  per-residue alphabet. **Withheld with its reason** for a family
                  that declares no escape: there "the escape removed" is the
                  declared rendering itself, the price is zero by construction,
                  and no substitute control is invented.
``residue_mass``  the model's probability mass on its residue tokens at every
                  scored TEXT position, where the family declares a residue
                  subspace disjoint from text. This is what identifies a
                  *collapsed* text mode -- a model treating arbitrary prose as
                  the prefix of a protein -- rather than merely a bad number.
                  Withheld with its reason where the residues are ordinary
                  letters of the text vocabulary, because there the statistic
                  would be a fact about twenty capital letters.

The rendering itself is declared in ``src.capability.models.joint_modes`` and nowhere else
(Appendix B rule 12); this file names a family and reads the rest from there.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.models import joint_modes  # noqa: E402
from src.capability.core.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    REPO,
    Cohort,
    protein_cohort,
    require_input_path,
    text_cohort,
)
from src.capability.core.budget import (  # noqa: E402
    IDENTIFICATION_CRITERION,
    LEGACY_FLOOR_NOTE,
    MIN_CONTEXT_INFORMATION_NATS,
    SCREENING_CONTEXT_INFORMATION_NATS,
    SCREENING_FLOOR_NOTE,
    RecordStatistics,
    SparseCounts,
    power_status,
    write_power_records,
)
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.core.measurement_support import LAPLACE_SMOOTHING, assert_disjoint, disjoint_unigram_cross_entropy_nats, held_out_cohort

#: ``modes[*].verdict`` decided at an underived 0.30-nat floor became
#: ``modes[*].identification_verdict`` decided at the calibrated
#: :data:`~src.capability.core.budget.SCREENING_CONTEXT_INFORMATION_NATS`. The 0.30
#: comparison is an inert column, and every mode carries a sufficient-statistics
#: sidecar. An older record is not this record with fields missing: readers key
#: on those field names.
SCHEMA_VERSION = "r2_transfer_joint_mode_qualification_v2"
DEFAULT_OUT = REPO / "results/R1/joint_mode_qualification"

#: Where the per-record sufficient statistics and the frozen record lists go.
#: A subdirectory rather than ``--out`` itself, because
#: ``run_external_baseline_h200.sh`` reads "any .json appeared in the output
#: directory" as completion, and a frozen cohort written before the report would
#: be read as the run having finished.
RECORDS_SUBDIRECTORY = "records"

#: The two values ``ModeStatistics.symbol_definition`` may take, and the strings
#: the sidecar publishes as ``n_symbols_is``. Declared once, because the label
#: and the array it describes are checked against each other rather than kept in
#: step by hand: the protein mode counts residues in both symbol units, so a
#: token label on it would name a unit the numbers are not in.
RESIDUE_SYMBOL = "a scored residue"
TOKEN_SYMBOL = "a scored token"

#: Modules whose content decides this stage's numbers, hashed into the artefact.
#: The rendering module is first because it is the one that has been worth 2.9
#: nats/token when wrong.
PROVENANCE_MODULES = (
    "src/capability/models/joint_modes.py",
    "src/capability/core/arms.py",
    "src/capability/core/budget.py",
    "src/capability/core/measurement_support.py",
    "src/capability/core/io.py",
)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

#: How the screening floor reads *here*, carried into every artefact.
#:
#: The floor itself is :data:`src.capability.core.budget.SCREENING_CONTEXT_INFORMATION_NATS`
#: and is declared there, once. This stage adopts it unchanged, and since
#: EXP-R2-221 it is a **pre-interval screen** rather than the identification
#: criterion: identification is
#: :func:`src.capability.core.budget.context_identification`, which reads a
#: displacement-corrected bootstrap interval. This stage scores one cohort draw
#: per mode and publishes no bootstrap, so neither that criterion nor
#: :func:`src.capability.core.budget.ratio_denominator_admissibility` can be evaluated
#: here. The sidecar this stage writes is what a later CPU re-analysis needs to
#: supply both.
IDENTIFICATION_FLOOR_STATUS = (
    "PRE-INTERVAL SCREEN. budget.SCREENING_CONTEXT_INFORMATION_NATS "
    f"({SCREENING_CONTEXT_INFORMATION_NATS} nats), the point rule EXP-R2-218 "
    "fitted against a known-zero null family at a false-positive rate of 0.05. "
    "It says the reading justifies an interval. It is NOT the identification "
    "criterion, which since EXP-R2-221 is budget.context_identification and "
    "reads the displacement-corrected interval, and it is NOT a licence to "
    "divide by the reading, which is "
    "budget.ratio_denominator_admissibility. Both need the reading's own "
    "bootstrap and therefore cannot be evaluated from this stage's single "
    "cohort draw. Read them from the sufficient-statistics sidecar beside this "
    "artefact instead"
)

#: What the retired 0.30-nat magnitude is reported as, and why it decides nothing.
#:
#: It used to be this stage's gate, declared locally and recorded as UNDERIVED on
#: the argument that a pass here admits a mode to a behavioural read and so needs
#: a stronger criterion than identification. The argument does not survive: the
#: gate refused nothing -- ``main`` always scored both modes and always wrote the
#: artefact -- so the magnitude only ever selected a verdict string, and the
#: evidence that actually separates a readable protein mode from an unreadable
#: one is the reversal cost (see :data:`REVERSAL_COST_EVIDENCE_NOTE`). The
#: magnitude is :data:`src.capability.core.budget.MIN_CONTEXT_INFORMATION_NATS`,
#: imported rather than declared a second time.
LEGACY_QUALIFICATION_FLOOR_NOTE = (
    f"{MIN_CONTEXT_INFORMATION_NATS} nats was this stage's own gate until it was "
    "retired: it was never derived, and it refused nothing -- both modes were "
    "always scored and the artefact was always written, so the magnitude only "
    "ever chose a verdict string. It is reported for comparability with the "
    "artefacts recorded under it and decides nothing here. " + LEGACY_FLOOR_NOTE
)

#: The reversal cost, and the reason it is reported rather than gated.
#:
#: Verified over all 66 published mode readings in ``results/`` on 2026-08-22.
#: Twenty were refused at the retired floor. Lowering it to the calibrated one
#: turns **eight** of those into admissions, all in the +0.0719 to +0.0918 band:
#: two stage-21 qualifications, four stage-24 cells -- two of them identity
#: references measuring the unmodified base and two chimeras carrying a
#: ``ProLLaMA_Stage_1`` embedding -- and two stage-38 re-reads of the stage-21
#: value. **Twelve refusals stand**: eleven read negative, and ``galactica-1.3b``
#: 's protein mode at +0.0481 sits 0.0019 nats below the 0.05 floor.
#:
#: ``Llama-2-7b-hf``'s protein mode itself is read four times on the unmodified
#: checkpoint -- the two stage-21 qualifications and the two stage-24 identity
#: references -- at +0.0719, +0.0843 (twice) and +0.0918.
REVERSAL_COST_EVIDENCE_NOTE = (
    "the substantive evidence that a protein mode is not worth a behavioural "
    "read is this reversal cost and NOT the context-information floor: a decoder "
    "carrying directional sequence structure cannot be indifferent to reading a "
    "sequence backwards. Llama-2-7b-hf pays -0.0013 nats/residue, which is "
    "indifference, against +0.1442 for the adapted stage (EXP-R2-152, "
    "re-measured at EXP-R2-174); src.capability.concept_alignment."
    "PROTEIN_MODE_BEHAVIOURAL_STATUS refuses it in exactly those terms. It is "
    "reported and NOT gated, because it does not reduce to a threshold: paired "
    "with their own context information, the refused chimera families "
    "s1Body_baseVocab and s1Body_baseHead read -0.061 to -0.085 nats at reversal "
    "costs of +0.119 to +0.138, overlapping the +0.136 to +0.167 of the admitted "
    "ProLLaMA-lineage cells -- s1Body_baseHead on the second draw pays +0.1379 "
    "and is refused where s1Body_baseEmbed pays +0.1358 and is admitted -- while "
    "galactica-125m reads -0.139 at +0.088. A reversal-cost criterion is a "
    "DIFFERENT PARTITION of the published cells, not a tightening of this one, "
    "and would change admissions in both directions. Both quantities are "
    "published; neither is a second gate"
)

IDENTIFICATION_VERDICT_NOTE = (
    "a mode below --identification-floor-nats is reported UNMEASURABLE ON THIS "
    "COHORT, not failing. It is a statement about this cohort and this evaluation "
    "interface; downstream analyses must exclude the mode rather than report a "
    "negative result from it (cohort_power.py's rule). The floor is the "
    "PRE-INTERVAL SCREEN and not the identification criterion, which since "
    "EXP-R2-221 is budget.context_identification and needs a bootstrap interval "
    "this stage does not compute -- see identification_criterion_not_evaluable_"
    "reason. And this verdict answers identification alone: it does NOT say the "
    "mode carries context-derived signal an ablation could destroy. Read "
    "controls.reversed.cost_nats_per_residue beside it for that, and "
    "reversal_cost_evidence for why that quantity is reported rather than gated"
)


# ------------------------------------------------------------------- checkpoint


def load_tokenizer(path: Path) -> tuple[Path, Any]:
    """The tokenizer alone, so a wrong checkpoint/family pair fails before the weights.

    Resolving the rendering needs nothing but the tokenizer, and it is the check
    most likely to refuse: a checkpoint whose delimiters or per-residue alphabet
    are not the declared family's is a configuration error, and paying a
    multi-gigabyte load to discover it is the shape stage 01 already moved its own
    refusals ahead of.
    """

    resolved = require_input_path(Path(path).resolve(), "--checkpoint")
    return resolved, AutoTokenizer.from_pretrained(str(resolved))


def load_model(
    resolved: Path, tokenizer: Any, *, device: str, dtype: str
) -> tuple[Any, dict[str, Any]]:
    """Load the weights and read the shape back from what was built.

    Every fact recorded here is read off the loaded object rather than echoed
    from the request, including the dtype: a build that ignored the requested
    dtype would otherwise be recorded as honouring it.
    """

    if dtype not in _DTYPES:
        raise ValueError(f"unsupported inference dtype {dtype!r}; known: {sorted(_DTYPES)}")
    model = AutoModelForCausalLM.from_pretrained(
        str(resolved),
        # ``torch_dtype`` rather than ``dtype`` for the reason
        # ``src.capability.core.arms.load_arm`` records: it is the only spelling both the
        # workstation's transformers and the pod's honour, and the observed-dtype
        # check below is what actually enforces the outcome.
        torch_dtype=_DTYPES[dtype],
        device_map={"": device},
    )
    model.eval()
    config = model.config
    observed = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
    )
    if observed != [dtype]:
        raise ValueError(f"requested dtype {dtype}, observed {observed}")

    def fact(*names: str) -> int:
        for name in names:
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
        raise ValueError(
            f"this checkpoint's config declares none of {names}, so its shape cannot "
            "be recorded and the artefact could not say what was measured"
        )

    facts = {
        "resolved_path": str(resolved),
        "model_type": str(getattr(config, "model_type", "undeclared")),
        "architectures": list(getattr(config, "architectures", []) or []),
        "n_layers": fact("num_hidden_layers", "n_layer"),
        "d_model": fact("hidden_size", "n_embd", "d_model"),
        "n_heads": fact("num_attention_heads", "n_head"),
        "vocab_size": fact("vocab_size"),
        "max_position_embeddings": fact("max_position_embeddings", "n_positions"),
        "dtype_requested": dtype,
        "dtype_observed": observed,
        "device": device,
        "facts_source": "read back from the loaded model's config and parameters, "
        "not echoed from the request",
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": int(len(tokenizer)),
    }
    return model, facts


# ---------------------------------------------------------------------- scoring


@torch.no_grad()
def score_positions(
    model: Any,
    token_ids: Sequence[int],
    positions: Sequence[int],
    *,
    device: str,
    subspace: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-position NLL in nats, and optionally the mass on a token subspace.

    The log-softmax is taken in float32 whatever the parameters are stored in,
    which is the convention EXP-R2-151 measured under; a bfloat16 log-softmax
    quantises at a step comparable with the effects this stage reports.
    """

    if not positions:
        raise ValueError("a scored record must have at least one scored position")
    ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    index = torch.tensor(list(positions), dtype=torch.long, device=device)
    logits = model(ids).logits[0].float()
    log_probabilities = torch.log_softmax(logits[index - 1], dim=-1)
    targets = ids[0, index]
    nll = -log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    mass = None
    if subspace is not None:
        columns = torch.tensor(list(subspace), dtype=torch.long, device=device)
        mass = log_probabilities[:, columns].exp().sum(1).double().cpu().numpy()
    return nll.double().cpu().numpy(), mass


def text_token_ids(tokenizer: Any, document: str, *, max_tokens: int) -> list[int]:
    """A text document's ids, truncated to the scored window."""

    return joint_modes.encode(tokenizer, document)[:max_tokens]


# --------------------------------------------------------------------- cohorts


def draw_cohort(args: argparse.Namespace, *, mode: str, n: int, skip: int, name: str) -> Cohort:
    """One draw from the corpus this mode is scored on.

    Both draws go through ``src.capability.core.arms``: the seeded permutation, the
    eligibility filter and the sampling record are the panel's, not this stage's
    (Appendix B rules 1 and 12). ``--cohort-draw-seed 0`` still selects the
    historical file-order draw, and the mode reaches the artefact through
    ``Cohort.sampling`` either way.
    """

    seed = args.cohort_draw_seed or None
    if mode == "protein":
        return protein_cohort(
            n,
            args.protein_min_len,
            args.protein_max_len,
            skip=skip,
            name=name,
            seed=seed,
        )
    return text_cohort(n, args.text_min_chars, skip=skip, name=name, seed=seed)


def mode_cohorts(args: argparse.Namespace, mode: str) -> tuple[Cohort, Cohort, dict[str, int]]:
    """The scored cohort and a held-out reference that cannot overlap it.

    Disjointness is enforced twice because the two failures are different. The
    reference is drawn ``--sequences`` records further into the *same* seeded
    permutation, which makes the two windows disjoint by index; it is then
    deduplicated by content, because an index offset is not disjointness when the
    corpus carries the same sequence under several accessions. The final
    assertion is what turns a silent leak into a stopped run.
    """

    scored = draw_cohort(args, mode=mode, n=args.sequences, skip=0, name=f"{mode}_scored")
    candidate = draw_cohort(
        args,
        mode=mode,
        n=args.unigram_sequences,
        skip=args.sequences,
        name=f"{mode}_unigram_reference",
    )
    reference, overlap = held_out_cohort(candidate, scored)
    assert_disjoint(scored, reference)
    return scored, reference, overlap


def cohort_record(cohort: Cohort, *, band_unit: str) -> dict[str, Any]:
    return {
        "name": cohort.name,
        "kind": cohort.kind,
        "n_records": len(cohort),
        "digest": cohort.digest,
        "provenance_digest": cohort.provenance_digest,
        "sampling_record": cohort.sampling,
        "band": [cohort.min_symbols, cohort.max_symbols],
        "band_unit": band_unit,
    }


# --------------------------------------------------------------- unigram support


def scored_target_records(
    tokenisation: joint_modes.JointTokenisation,
    records: Sequence[str],
    *,
    context: str | None,
    variant: str = joint_modes.DECLARED,
) -> tuple[np.ndarray, SparseCounts]:
    """Protein-target counts over exactly the multiset the model is scored on.

    Counted through the same rendering the model sees and over the support the
    rendering declares, so the reference and the scored sample are counted over
    the same kind of symbol whichever symbol unit the family declares. The
    verification inside ``render`` is what guarantees every counted target is one
    the support can represent, so the lookup below cannot silently drop a target.

    Returns the dense count vector the unigram estimator consumes and the
    per-record counts it sums from, in one pass -- the shape
    :func:`src.capability.prediction_addressed.scored_target_records` established,
    and for its reason: a caller that persists the per-record statistics would
    otherwise render a four-hundred-record reference corpus a second time.

    **The id space is the declared support, not the vocabulary.** An index here
    is a position in ``tokenisation.scored_target_ids``, which the artefact
    publishes under ``rendering.scored_target_token_ids``. That is the support
    the held-out unigram is fitted over, and a re-analysis that refitted it over
    the whole vocabulary instead would be applying a different estimator -- worth
    0.570 nats of smoothing bias on the staged LLaMA-2 vocabulary against 0.011
    over the 463 residue-spelling tokens.
    """

    order = {value: index for index, value in enumerate(tokenisation.scored_target_ids)}
    blocks: list[np.ndarray] = []
    for sequence in records:
        rendered = tokenisation.render(sequence, context=context, variant=variant)
        blocks.append(
            np.asarray(
                [order[rendered.token_ids[position]] for position in rendered.scored_positions],
                dtype=np.int64,
            )
        )
    per_record = SparseCounts.from_records(blocks)
    counts = per_record.vocabulary_totals(len(order))
    if counts.sum() < 1:
        raise RuntimeError("the protein records yielded no scored targets")
    return counts, per_record


def scored_target_counts(
    tokenisation: joint_modes.JointTokenisation,
    records: Sequence[str],
    *,
    context: str | None,
) -> np.ndarray:
    """The dense half of :func:`scored_target_records`."""

    return scored_target_records(tokenisation, records, context=context)[0]


def text_target_records(
    tokenizer: Any, records: Sequence[str], *, vocab_size: int, max_tokens: int
) -> tuple[np.ndarray, SparseCounts]:
    """Next-token-target counts over the same window the model is scored on.

    Dense and per-record, for the reason :func:`scored_target_records` gives. The
    id space here IS the checkpoint's vocabulary, because that is the support a
    scored text target can take.
    """

    blocks: list[np.ndarray] = []
    for document in records:
        targets = text_token_ids(tokenizer, document, max_tokens=max_tokens)[1:]
        array = np.asarray(targets, dtype=np.int64)
        if array.size and (array.min() < 0 or array.max() >= vocab_size):
            raise ValueError("a token id fell outside the checkpoint's declared vocabulary")
        blocks.append(array)
    per_record = SparseCounts.from_records(blocks)
    counts = per_record.vocabulary_totals(vocab_size)
    if counts.sum() < 1:
        raise RuntimeError("the text records yielded no scored targets")
    return counts, per_record


def text_target_counts(
    tokenizer: Any, records: Sequence[str], *, vocab_size: int, max_tokens: int
) -> np.ndarray:
    """The dense half of :func:`text_target_records`."""

    return text_target_records(
        tokenizer, records, vocab_size=vocab_size, max_tokens=max_tokens
    )[0]


def unigram_record(
    reference_counts: np.ndarray,
    scored_counts: np.ndarray,
    *,
    support: str,
    reference: Cohort,
    overlap: dict[str, int],
) -> dict[str, Any]:
    """The held-out context-free baseline, with the support it was fitted over."""

    cross_entropy = disjoint_unigram_cross_entropy_nats(reference_counts, scored_counts)
    total = float(reference_counts.sum())
    size = int(reference_counts.size)
    return {
        "estimator": "disjoint_held_out",
        "support": support,
        "support_size": size,
        "smoothing": float(LAPLACE_SMOOTHING),
        "smoothing_mass_fraction": float(
            LAPLACE_SMOOTHING * size / (total + LAPLACE_SMOOTHING * size)
        ),
        "reference_tokens": int(total),
        "reference_cohort": cohort_record(reference, band_unit=(
            "residues" if reference.kind == "protein" else "characters"
        )),
        "reference_overlap_removed": overlap,
        "cross_entropy_nats": float(cross_entropy),
        "note": (
            "fitted on a draw disjoint from the scored one and deduplicated by "
            "content; never on the scored sample (Appendix B rule 3). The support "
            "is the set of ids a scored target of this mode can take, so the "
            "smoothing bias -- which scales with support size against reference "
            "size -- stays far below the effect being measured"
        ),
    }


def verdict_record(context_information: float, threshold: float) -> dict[str, Any]:
    """The pre-interval screening reading, in the vocabulary stage 01 declares.

    One verdict, decided against the pre-interval screening floor, plus the
    retired 0.30-nat comparison beside it as an inert column. The two are named
    apart so that neither can be read as the other. An older record, whose
    ``verdict`` was decided at 0.30, cannot be mistaken for this one:
    ``verdict`` does not appear here at all.

    **This stage does not evaluate the identification criterion and says so.**
    Since EXP-R2-221 identification is ``budget.context_identification``, which
    reads the mode's displacement-corrected bootstrap interval; this stage
    publishes one cohort draw and no bootstrap, so the criterion is not
    evaluable here. What it does instead is persist the per-record sufficient
    statistics into ``records/`` so that ``context_information_bootstrap.py``
    can take the verdict on CPU from the same numbers. The field names are kept
    as they were because artefacts and consumers already read them; the added
    fields are what stop the screen being read as the criterion.
    """

    _, status = power_status(context_information, threshold)
    return {
        "identification_verdict": status,
        "identification_floor_nats": float(threshold),
        "identification_verdict_is_the_criterion": False,
        "identification_criterion": IDENTIFICATION_CRITERION,
        "identification_criterion_not_evaluable_reason": (
            "budget.context_identification needs the lower bound of the "
            "displacement-corrected bootstrap interval for I, and this stage "
            "publishes one cohort draw and no bootstrap. Run "
            "context_information_bootstrap.py over the records/ sidecar "
            "beside this report to take the verdict; on the EXP-R2-220 cells "
            "the two agree everywhere except galactica-1.3b's protein mode, "
            "which this screen refuses at +0.047678 and the criterion identifies "
            "on a corrected interval whose lower end is +0.038694"
        ),
        "pre_interval_screen_criterion": (
            "the point estimate against budget.SCREENING_CONTEXT_INFORMATION_NATS. "
            "It says the mode reads enough on this cohort to be worth an "
            "interval, NOT that its reading may be divided by and NOT that it "
            "carries signal an ablation could destroy"
        ),
        "screening_floor_note": SCREENING_FLOOR_NOTE,
        "identification_verdict_note": IDENTIFICATION_VERDICT_NOTE,
        # Legacy column: what the retired floor this stage used to gate on would
        # have said here. It admits and refuses nothing.
        "legacy_qualification_floor_nats": MIN_CONTEXT_INFORMATION_NATS,
        "clears_legacy_qualification_floor": bool(
            context_information >= MIN_CONTEXT_INFORMATION_NATS
        ),
        "legacy_qualification_floor_note": LEGACY_QUALIFICATION_FLOOR_NOTE,
    }


# ------------------------------------------- per-record statistics, either mode


def record_statistics(
    name: str,
    *,
    support_size: int,
    clean_nll_sum: Sequence[float],
    token_count: Sequence[int],
    n_symbols: Sequence[int],
    targets: SparseCounts,
) -> RecordStatistics:
    """One condition's per-record sufficient statistics, refused if a record is empty.

    ``budget.RecordStatistics`` is the repository's carrier and is built directly
    rather than through :func:`src.capability.core.budget.record_statistics`, which
    requires an ``arms.Arm``: this stage measures a checkpoint reached by path,
    which is not one. The empty-record refusal is this stage's own. Every scored
    record here has at least one scored position -- :func:`score_positions`
    refuses otherwise -- so a zero token count means the two halves of the
    estimand were built over different record lists, and a bootstrap over it
    would divide by zero inside an iteration rather than here.
    """

    tokens = np.asarray(token_count, dtype=np.int64)
    if tokens.size and int(tokens.min()) < 1:
        raise ValueError(
            f"{name}: record {int(np.argmin(tokens))} carries no scored token, so it "
            "contributes to no term of the estimand and cannot be a resampling unit"
        )
    return RecordStatistics(
        arm=name,
        vocab_size=int(support_size),
        record_index=np.arange(tokens.size, dtype=np.int64),
        clean_nll_sum=np.asarray(clean_nll_sum, dtype=np.float64),
        token_count=tokens,
        n_symbols=np.asarray(n_symbols, dtype=np.int64),
        target_counts=targets,
    )


# ----------------------------------------------------------------- protein mode


def score_protein_records(
    model: Any,
    tokenisation: joint_modes.JointTokenisation,
    records: Sequence[str],
    *,
    device: str,
    context: str | None,
    variant: str,
    max_tokens: int,
    condition: str | None,
) -> tuple[dict[str, Any], RecordStatistics | None]:
    """One protein condition: the report, and the per-record statistics behind it.

    The aggregates are taken from the per-record arrays rather than accumulated
    beside them, so that re-aggregating the sidecar reproduces the published
    figure bit for bit instead of to within a summation order.

    ``condition=None`` returns no statistics, and the naive control is the only
    caller that passes it. That control deliberately scores merged multi-residue
    pieces, which are not ids the declared support can represent at all: counting
    them into this mode's id space would put targets the held-out unigram was
    never fitted over into the very file a re-analysis refits it from. There is
    no second id space invented for them.
    """

    nll_sums: list[float] = []
    tokens: list[int] = []
    residues: list[int] = []
    blocks: list[np.ndarray] = []
    order = (
        {}
        if condition is None
        else {value: index for index, value in enumerate(tokenisation.scored_target_ids)}
    )
    for sequence in records:
        rendered = tokenisation.render(sequence, context=context, variant=variant)
        if len(rendered.token_ids) > max_tokens:
            raise ValueError(
                f"a rendered protein needs {len(rendered.token_ids)} tokens and "
                f"--max-tokens is {max_tokens}; truncating it would drop the closing "
                "delimiter and silently change the scored span. Raise --max-tokens or "
                "lower --protein-max-len"
            )
        nll, _ = score_positions(
            model, rendered.token_ids, rendered.scored_positions, device=device
        )
        nll_sums.append(float(nll.sum()))
        tokens.append(rendered.n_scored_tokens)
        residues.append(rendered.n_residues)
        if condition is not None:
            blocks.append(
                np.asarray(
                    [
                        order[rendered.token_ids[position]]
                        for position in rendered.scored_positions
                    ],
                    dtype=np.int64,
                )
            )
    statistics = (
        None
        if condition is None
        else record_statistics(
            condition,
            support_size=len(order),
            clean_nll_sum=nll_sums,
            token_count=tokens,
            n_symbols=residues,
            targets=SparseCounts.from_records(blocks),
        )
    )
    total_nll = float(np.asarray(nll_sums, dtype=np.float64).sum())
    scored_tokens = int(np.asarray(tokens, dtype=np.int64).sum())
    n_residues = int(np.asarray(residues, dtype=np.int64).sum())
    report = {
        "variant": variant,
        # The name this condition's per-record statistics are persisted under in
        # the sidecar, or null where none are persisted.
        "persisted_condition": condition,
        "document_context": context,
        "n_records": len(records),
        "n_scored_tokens": scored_tokens,
        "n_scored_residues": n_residues,
        "residues_per_scored_token": n_residues / scored_tokens,
        "clean_nll_nats_per_scored_token": total_nll / scored_tokens,
        "clean_nll_nats_per_residue": total_nll / n_residues,
        "symbol_unit": tokenisation.declaration.symbol_unit,
        "verified_against_declared_symbol_unit": variant == joint_modes.DECLARED,
    }
    return report, statistics


def comparability_record(
    declaration: joint_modes.JointRendering, residues_per_scored_token: float
) -> dict[str, Any]:
    """Whether this arm's magnitude may be placed beside another arm's, and why not.

    Appendix B rule 26 and limitation L23: an estimand whose unit is the token is
    not a cross-arm estimand when arms differ in symbols per token. The verdict is
    a field rather than a caveat in prose because the number it qualifies is a
    finite, plausible-looking magnitude that reads exactly like a comparable one.
    """

    if declaration.symbol_unit == joint_modes.RESIDUE_UNIT:
        return {
            "verdict": "COMPARABLE_ACROSS_ARMS",
            "symbol_unit": declaration.symbol_unit,
            "measured_residues_per_scored_token": float(residues_per_scored_token),
            "note": (
                "the scored unit is the residue and the declared rendering was "
                "verified at exactly one token per residue, so this magnitude is in "
                "the same units as every other per-residue arm's"
            ),
        }
    return {
        "verdict": "NOT_COMPARABLE_ACROSS_ARMS",
        "symbol_unit": declaration.symbol_unit,
        "measured_residues_per_scored_token": float(residues_per_scored_token),
        "note": (
            f"{declaration.name} declares the TOKEN as its protein symbol unit, and "
            f"one scored token carried {residues_per_scored_token:.3f} residues on "
            "this cohort. Nats per token is therefore not the unit any per-residue "
            "arm reports, and it is not this arm's unit on another cohort either, "
            "since residues per token is a property of the corpus and the tokenizer "
            "together. Do NOT place this magnitude beside a per-residue arm's or "
            "convert it by dividing: the held-out reference is fitted over token "
            "targets, so the quotient is not a per-residue cross-entropy (Appendix B "
            "rule 26, limitation L23). The within-arm controls below -- reversal "
            "above all -- are what this family's verdict rests on"
        ),
    }


def protein_mode(
    args: argparse.Namespace,
    model: Any,
    tokenisation: joint_modes.JointTokenisation,
) -> tuple[dict[str, Any], "ModeStatistics"]:
    """Context information on the declared protein symbol, with its controls beside it.

    Returns the record and the per-record statistics behind it, the pair
    :func:`src.capability.core.budget.arm_power_with_records` established. The caller
    persists the second half; nothing here writes a file.
    """

    declaration = tokenisation.declaration
    per_residue = declaration.symbol_unit == joint_modes.RESIDUE_UNIT
    unit = "nats per scored residue" if per_residue else "nats per scored token"
    clean_key = (
        "clean_nll_nats_per_residue" if per_residue else "clean_nll_nats_per_scored_token"
    )

    scored, reference, overlap = mode_cohorts(args, "protein")
    print(f"[protein] {len(scored)} scored records, {len(reference)} held-out reference records")
    reference_counts, reference_per_record = scored_target_records(
        tokenisation, reference.records, context=args.protein_context
    )
    unigram = unigram_record(
        reference_counts,
        scored_target_counts(tokenisation, scored.records, context=args.protein_context),
        support=(
            "the declared residue token ids, the only ids a scored protein target "
            "can take under this rendering"
            if per_residue
            else "every token of this vocabulary spelled purely of canonical "
            "residues, which is the set of ids a scored protein target can take "
            "under a rendering whose symbol unit is the token"
        ),
        reference=reference,
        overlap=overlap,
    )
    declared, declared_statistics = score_protein_records(
        model,
        tokenisation,
        scored.records,
        device=args.device,
        context=args.protein_context,
        variant=joint_modes.DECLARED,
        max_tokens=args.max_tokens,
        condition="protein_declared",
    )
    context_information = unigram["cross_entropy_nats"] - declared[clean_key]
    print(
        f"  declared rendering  {declared[clean_key]:.4f} {unit} "
        f"against unigram {unigram['cross_entropy_nats']:.4f} "
        f"at {declared['residues_per_scored_token']:.3f} residues/token"
    )

    reversed_records = [sequence[::-1] for sequence in scored.records]
    reversed_score, reversed_statistics = score_protein_records(
        model,
        tokenisation,
        reversed_records,
        device=args.device,
        context=args.protein_context,
        variant=joint_modes.DECLARED,
        max_tokens=args.max_tokens,
        condition="protein_reversed",
    )
    reversed_score["cost_nats_per_residue"] = (
        reversed_score["clean_nll_nats_per_residue"] - declared["clean_nll_nats_per_residue"]
    )
    reversed_score["cost_unit"] = "nats per residue"
    reversed_score["evidence"] = REVERSAL_COST_EVIDENCE_NOTE
    reversed_score["note"] = (
        "the same sequences read C-to-N, and the control this stage's reading of a "
        "protein mode rests on, because it is a within-arm difference over an "
        "identical residue multiset -- it needs no cross-arm unit to be readable. A "
        "decoder carrying real directional sequence structure cannot be indifferent "
        "to it. "
        + (
            "Reversal preserves the residue multiset exactly, so the held-out "
            "reference above applies to this condition unchanged and the cost is a "
            "pure directional-information reading"
            if per_residue
            else "Reversal permutes the residues but NOT the tokenisation, so this "
            "condition is a different token population: the held-out token reference "
            "above does not apply to it, no context information is reported for it, "
            "and the cost is read per residue -- the denominator the two conditions "
            "do share (Appendix B rules 26, 27). Compare the two conditions' "
            "residues_per_scored_token to see how far the token populations differ"
        )
    )

    if declaration.naive_control_available:
        naive, _ = score_protein_records(
            model,
            tokenisation,
            scored.records,
            device=args.device,
            context=args.protein_context,
            variant=joint_modes.NAIVE,
            max_tokens=args.max_tokens,
            condition=None,
        )
        naive["price_nats_per_residue"] = (
            naive["clean_nll_nats_per_residue"] - declared["clean_nll_nats_per_residue"]
        )
        naive["price_nats_per_scored_token"] = (
            naive["clean_nll_nats_per_scored_token"]
            - declared["clean_nll_nats_per_scored_token"]
        )
        naive["context_information_nats_per_residue"] = (
            unigram["cross_entropy_nats"] - naive["clean_nll_nats_per_residue"]
        )
        naive["note"] = (
            "the same sequences with the per-residue escape removed -- what an unaided "
            "AutoTokenizer call produces. Deliberately NOT verified against the "
            "per-residue alphabet; it exists to be wrong. Read the per-residue price: "
            "the per-token one compares two different token populations, since this "
            "condition scores merged multi-residue pieces (Appendix B rules 26, 27). "
            "No per-record sufficient statistics are persisted for it: those pieces "
            "are not ids the declared support can represent, and counting them into "
            "this mode's id space would put targets the held-out unigram was never "
            "fitted over into the file a re-analysis refits it from"
        )
        print(
            f"  naive rendering     {naive['clean_nll_nats_per_scored_token']:.4f} nats/token "
            f"at {naive['residues_per_scored_token']:.3f} residues/token"
        )
    else:
        naive = {
            "verdict": "WITHHELD",
            "reason": (
                f"{declaration.name} declares no per-residue escape. Its trained "
                "protein format writes residues straight into the checkpoint's own "
                "vocabulary, so 'the same block with the escape removed' IS the "
                "declared rendering: the price would be exactly 0.0 by construction "
                "and would report a control that measured nothing. The rendering cost "
                "this control exists to price does not exist for this family, and no "
                "substitute control is invented in its place (Appendix B rule 4)"
            ),
        }
        print(f"  naive rendering     WITHHELD ({declaration.name} has no escape to remove)")
    print(f"  reversed            cost {reversed_score['cost_nats_per_residue']:+.4f} nats/residue")

    comparability = comparability_record(declaration, declared["residues_per_scored_token"])
    record: dict[str, Any] = {
        "cohort": cohort_record(scored, band_unit="residues"),
        "unigram_reference": unigram,
        "declared_rendering": declared,
        "symbol_unit": declaration.symbol_unit,
        "context_information_nats": float(context_information),
        "context_information_unit": unit,
        "measured_residues_per_scored_token": float(declared["residues_per_scored_token"]),
        "cross_arm_comparability": comparability,
        "context_information_definition": (
            f"held-out unigram cross-entropy on the scored protein targets minus the "
            f"model's clean cross-entropy on the same targets, in {unit}. One scored "
            f"symbol is {joint_modes.SYMBOL_UNIT_DEFINITIONS[declaration.symbol_unit]}"
        ),
        "reversal_cost_nats_per_residue": float(reversed_score["cost_nats_per_residue"]),
        "reversal_cost_evidence": REVERSAL_COST_EVIDENCE_NOTE,
        "controls": {"reversed": reversed_score, "naive_rendering": naive},
    }
    record.update(verdict_record(context_information, args.identification_floor_nats))

    # The held-out reference is fitted over the forward token population. It
    # applies to the reversed condition only where reversal preserves that
    # population exactly, which is the per-residue case; attaching it to a
    # token-unit reversal would hand a re-analysis a baseline for a multiset the
    # condition never scores. The absence is what a reader sees there.
    conditions = {
        "protein_declared": replace(declared_statistics, reference_counts=reference_per_record),
        "protein_reversed": (
            replace(reversed_statistics, reference_counts=reference_per_record)
            if per_residue
            else reversed_statistics
        ),
    }
    return record, ModeStatistics(
        mode="protein",
        scored=scored,
        reference=reference,
        support=unigram["support"],
        support_size=int(unigram["support_size"]),
        id_space=(
            "an index into rendering.scored_target_token_ids, which is the support "
            "the held-out unigram is fitted over. NOT a vocabulary id"
        ),
        # The protein mode's persisted n_symbols array is the RESIDUE count in
        # both units -- score_protein_records fills it from rendered.n_residues
        # unconditionally, because the reversal control is only a within-arm
        # difference over an identical residue multiset in that unit. Labelling
        # a token-unit family's array "a scored token" would hand a re-analysis
        # bits per residue under a name that says token; ModeStatistics now
        # refuses that combination outright.
        symbol_definition=RESIDUE_SYMBOL,
        conditions=conditions,
        reference_applies_to_reversed=bool(per_residue),
    )


# -------------------------------------------------------------------- text mode


def text_mode(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    tokenisation: joint_modes.JointTokenisation,
    *,
    vocab_size: int,
) -> tuple[dict[str, Any], "ModeStatistics"]:
    """Context information on text tokens, with the residue-mass control beside it.

    The record and the per-record statistics behind it, as :func:`protein_mode`
    returns them. There is no reversed condition here: reversing a document is
    not the within-modality control reversing a sequence is.
    """

    scored, reference, overlap = mode_cohorts(args, "text")
    print(f"[text] {len(scored)} scored documents, {len(reference)} held-out reference documents")
    reference_counts, reference_per_record = text_target_records(
        tokenizer, reference.records, vocab_size=vocab_size, max_tokens=args.max_tokens
    )
    unigram = unigram_record(
        reference_counts,
        text_target_counts(
            tokenizer, scored.records, vocab_size=vocab_size, max_tokens=args.max_tokens
        ),
        support="the checkpoint's full vocabulary, which is the set of ids a "
        "scored text target can take",
        reference=reference,
        overlap=overlap,
    )

    declaration = tokenisation.declaration
    disjoint = declaration.residue_subspace_disjoint_from_text
    subspace = (
        sorted(int(value) for value in tokenisation.residue_ids.values()) if disjoint else None
    )
    nll_sums: list[float] = []
    tokens: list[int] = []
    blocks: list[np.ndarray] = []
    mass_total = 0.0
    dominated = 0
    for document in scored.records:
        token_ids = text_token_ids(tokenizer, document, max_tokens=args.max_tokens)
        if len(token_ids) < 2:
            raise ValueError("a text record tokenised to fewer than two tokens")
        positions = list(range(1, len(token_ids)))
        nll, mass = score_positions(
            model, token_ids, positions, device=args.device, subspace=subspace
        )
        nll_sums.append(float(nll.sum()))
        tokens.append(len(positions))
        blocks.append(np.asarray(token_ids[1:], dtype=np.int64))
        if mass is not None:
            mass_total += float(mass.sum())
            dominated += int((mass > 0.5).sum())
    statistics = record_statistics(
        "text_declared",
        support_size=vocab_size,
        clean_nll_sum=nll_sums,
        token_count=tokens,
        # The declared text symbol IS the token, so the symbol count and the
        # token count are one array. Recorded rather than omitted because the
        # sidecar's consumers read a per-symbol rate off this field.
        n_symbols=tokens,
        targets=SparseCounts.from_records(blocks),
    )
    scored_tokens = int(statistics.token_count.sum())
    clean = float(statistics.clean_nll_sum.sum()) / scored_tokens
    context_information = unigram["cross_entropy_nats"] - clean
    print(
        f"  clean {clean:.4f} nats/token against unigram "
        f"{unigram['cross_entropy_nats']:.4f}"
    )

    if disjoint:
        control: dict[str, Any] = {
            "residue_subspace_ids": subspace,
            "mean_residue_probability_mass_per_position": mass_total / scored_tokens,
            "fraction_of_positions_with_residue_mass_above_half": dominated / scored_tokens,
            "n_scored_positions": scored_tokens,
            "uniform_reference_nats": math.log(vocab_size),
            "note": (
                "the mass this checkpoint places on its own residue tokens at every "
                "scored TEXT position. A high value identifies a COLLAPSED text mode "
                "-- the model treating arbitrary prose as the prefix of a protein -- "
                "which a cross-entropy alone cannot distinguish from a merely poor "
                "one. Compare the clean cross-entropy against uniform_reference_nats: "
                "worse than uniform is a broken interface until this control says "
                "otherwise"
            ),
        }
    else:
        control = {
            "verdict": "WITHHELD",
            "reason": (
                f"{declaration.name} declares no residue subspace disjoint from text: "
                "its residues are ordinary single-letter pieces of the text "
                "vocabulary, so a 'residue probability mass' here would be a "
                "statistic about twenty capital letters and would identify nothing"
            ),
        }

    record: dict[str, Any] = {
        "cohort": cohort_record(scored, band_unit="characters"),
        "unigram_reference": unigram,
        "clean_nll_nats_per_scored_token": clean,
        "n_scored_tokens": scored_tokens,
        "max_tokens": int(args.max_tokens),
        "context_information_nats": float(context_information),
        "context_information_unit": "nats per scored token",
        "context_information_definition": (
            "held-out unigram cross-entropy on the scored next-token targets minus "
            "the model's clean cross-entropy on the same targets"
        ),
        "controls": {"residue_subspace_mass": control},
    }
    record.update(verdict_record(context_information, args.identification_floor_nats))
    return record, ModeStatistics(
        mode="text",
        scored=scored,
        reference=reference,
        support=unigram["support"],
        support_size=int(vocab_size),
        id_space="a token id of the checkpoint's own vocabulary",
        symbol_definition=TOKEN_SYMBOL,
        conditions={
            "text_declared": replace(statistics, reference_counts=reference_per_record)
        },
        reference_applies_to_reversed=None,
    )


# ------------------------------------------------- per-record sufficient statistics



@dataclass(frozen=True)
class ModeStatistics:
    """One mode's scored cohort, its held-out reference and the statistics behind both.

    Assembled by :func:`protein_mode` and :func:`text_mode` and persisted by
    :func:`write_mode_records`. One object per mode because a mode is one cohort
    draw against one reference: the protein and text modes share neither, so
    their statistics cannot share a sidecar either.

    ``conditions`` is keyed by condition name -- ``protein_declared``,
    ``protein_reversed``, ``text_declared`` -- because the reversed condition is
    a second reading of the same records and not a second cohort. That is what
    makes a paired reversal-cost re-analysis possible from the file alone.
    """

    mode: str
    scored: Cohort
    reference: Cohort
    support: str
    support_size: int
    id_space: str
    symbol_definition: str
    conditions: dict[str, RecordStatistics]
    reference_applies_to_reversed: bool | None

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError(f"{self.mode}: a mode persists at least one condition")
        if self.symbol_definition not in (RESIDUE_SYMBOL, TOKEN_SYMBOL):
            raise ValueError(
                f"{self.mode}: {self.symbol_definition!r} is not one of the two "
                "declared symbol definitions, and the sidecar's n_symbols_is "
                "field is read as one of them"
            )
        for name, record in self.conditions.items():
            if record.vocab_size != self.support_size:
                raise ValueError(
                    f"{name}: statistics over {record.vocab_size} ids against a "
                    f"declared support of {self.support_size}; the counts and the "
                    "baseline would then be taken over different inventories"
                )
            # The label travels into the sidecar as n_symbols_is and is what a
            # re-analysis divides by to get a per-symbol rate. It must therefore
            # agree with what the array counts. Only the token claim is
            # checkable: a residue array equals the token array exactly when the
            # rendering is one token per residue, and then both names are true
            # of it, but a token claim over an array that is not the token count
            # is false however the rendering came out.
            if self.symbol_definition == TOKEN_SYMBOL and not np.array_equal(
                record.n_symbols, record.token_count
            ):
                raise ValueError(
                    f"{name}: n_symbols is declared to count {TOKEN_SYMBOL!r} but "
                    f"holds {int(record.n_symbols.sum())} against "
                    f"{int(record.token_count.sum())} scored tokens, so a "
                    "per-symbol rate read off this sidecar would be in a unit the "
                    "label denies"
                )
            if int(record.record_index.size) != len(self.scored):
                raise ValueError(
                    f"{name}: {int(record.record_index.size)} scored rows against "
                    f"{len(self.scored)} cohort records"
                )


def write_mode_records(
    out: Path, statistics: ModeStatistics, *, seeds: dict[str, int], max_tokens: int
) -> dict[str, Any]:
    """Persist one mode's sufficient statistics, its cohort and its reference.

    Three files under ``records/``, named the way ``cohort_power.py`` names
    its own so that ``context_information_bootstrap.py`` reads them without
    being told where to look: ``power_<cohort>_<digest>.records.npz`` beside
    ``cohort_<cohort>_<digest>.json`` and
    ``reference_<cohort>_<reference digest>.json``. The sidecar carries no
    sequence text by design, and the two record lists are what a re-analysis
    needs to group by near-duplicate content at all -- a singleton grouping is
    narrowest exactly where the group structure matters.

    Returned rather than written into the report here, so that the report can
    carry the digest of the file that was actually produced.
    """

    directory = Path(out) / RECORDS_SUBDIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    scored, reference = statistics.scored, statistics.reference
    stem = f"{scored.name}_{scored.digest[:12]}"
    sidecar_path = directory / f"power_{stem}.records.npz"
    block = write_power_records(
        sidecar_path,
        statistics.conditions,
        cohort_digest=scored.digest,
        reference_digest=reference.digest,
        smoothing=float(LAPLACE_SMOOTHING),
        seeds=seeds,
        max_len=int(max_tokens),
    )
    cohort_path = directory / f"cohort_{stem}.json"
    write_json(
        cohort_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact": "frozen_cohort",
            "cohort_digest": scored.digest,
            "cohort_name": scored.name,
            "cohort_kind": scored.kind,
            "min_symbols": scored.min_symbols,
            "max_symbols": scored.max_symbols,
            "n_records": len(scored),
            "records": scored.records,
            "metadata": scored.metadata,
        },
    )
    reference_path = directory / f"reference_{scored.name}_{reference.digest[:12]}.json"
    write_json(
        reference_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact": "held_out_unigram_reference",
            "reference_digest": reference.digest,
            "reference_name": reference.name,
            "reference_kind": reference.kind,
            # The scored cohort this block was held out against, which is what
            # makes the pair a leakage screen rather than two record lists.
            "cohort_digest": scored.digest,
            "min_symbols": reference.min_symbols,
            "max_symbols": reference.max_symbols,
            "n_records": len(reference),
            "records": reference.records,
            # Carries the sampling record held_out_cohort travels forward,
            # including how many records the content deduplication removed.
            "metadata": reference.metadata,
        },
    )
    return {
        **block,
        # Every path in this block is relative to --out, including the sidecar's:
        # write_power_records returns its own basename, which is the whole name
        # only for a stage that writes flat.
        "path": f"{RECORDS_SUBDIRECTORY}/{sidecar_path.name}",
        "directory": RECORDS_SUBDIRECTORY,
        "cohort_records_path": f"{RECORDS_SUBDIRECTORY}/{cohort_path.name}",
        "reference_records_path": f"{RECORDS_SUBDIRECTORY}/{reference_path.name}",
        "id_space": statistics.id_space,
        "support": statistics.support,
        "support_size": statistics.support_size,
        "n_symbols_is": statistics.symbol_definition,
        "reference_applies_to_reversed": statistics.reference_applies_to_reversed,
        "note": (
            "per-record clean-NLL sums, token counts, symbol counts and sparse "
            "target counts for every condition of this mode, in "
            "budget.write_power_records's format -- so the entries listed under "
            "'arms' above are this stage's CONDITION names, not panel arms: a "
            "checkpoint reached by path is not an arm, and the writer's key is "
            "spelled arms. Re-aggregating them reproduces "
            "this artefact's own point estimates exactly; grouping the frozen "
            "records beside them by near-duplicate content gives the resampling "
            "unit a bootstrap standard error needs. A condition with no "
            "reference_* arrays had no held-out baseline that applies to it and "
            "none is substituted"
        ),
    }


# ----------------------------------------------------------------------- driver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="directory of the joint checkpoint to qualify. Required and not an "
        "arm name: a checkpoint that has not passed this stage must not be in the "
        "panel, so there is nothing for a default to point at",
    )
    parser.add_argument(
        "--rendering",
        required=True,
        choices=joint_modes.RENDERING_NAMES,
        help="which declared family's input format this checkpoint takes. The set "
        "is composed by src.capability.models.joint_modes, which is the single place either "
        "mode's format is decided (Appendix B rule 12)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=tuple(_DTYPES), help="inference dtype; "
        "the log-softmax is taken in float32 regardless"
    )
    parser.add_argument(
        "--modes",
        default="both",
        choices=("text", "protein", "both"),
        help="which modes to qualify. 'both' is the default because the "
        "admission rule needs both and a checkpoint measured in one is not "
        "qualified",
    )
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument(
        "--unigram-sequences",
        type=int,
        default=400,
        help="records the held-out context-free baseline is fitted on, drawn past "
        "the scored cohort in the same permutation and then deduplicated by content",
    )
    parser.add_argument("--protein-min-len", type=int, default=64)
    parser.add_argument("--protein-max-len", type=int, default=246)
    parser.add_argument(
        "--text-min-chars",
        type=int,
        default=800,
        help="floor of the text cohort, in characters. src.capability.core.arms.text_cohort's "
        "own default, so the population is the one every other text measurement in "
        "this repository uses",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="scored window for text. A protein rendering is never truncated: the "
        "stage raises instead, because dropping the closing delimiter would silently "
        "change the scored span",
    )
    parser.add_argument(
        "--protein-context",
        default=None,
        help="optional document context the protein block is embedded in, filled "
        "into the family's declared template. Omitted means the bare block, and "
        "whichever was used reaches the artefact",
    )
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="seed for the permutation both draws are windows of; 0 selects the "
        "historical file-order draw, which is a declared choice and not a default "
        "(Appendix B rule 1)",
    )
    parser.add_argument(
        "--identification-floor-nats",
        type=float,
        default=SCREENING_CONTEXT_INFORMATION_NATS,
        help="the floor below which a mode is reported unmeasurable on this "
        "cohort. budget.SCREENING_CONTEXT_INFORMATION_NATS, the calibrated "
        "identification floor, which is the only criterion this stage can apply: "
        "it publishes no bootstrap, so no mode reading carries the standard error "
        "budget.ratio_denominator_admissibility needs. The option is spelled "
        "differently from the one the retired 0.30-nat gate used, so that an old "
        "command line fails rather than reinstating that gate by inertia",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    modes = ("protein", "text") if args.modes == "both" else (args.modes,)

    declaration = joint_modes.rendering(args.rendering)
    print(f"[load] {args.checkpoint} as {declaration.name} on {args.device}")
    # Resolving the rendering against this tokenizer is what refuses a
    # checkpoint/family pair. It happens before the weights are read, and long
    # before any cross-entropy exists to be misread.
    resolved, tokenizer = load_tokenizer(args.checkpoint)
    tokenisation = joint_modes.resolve(tokenizer, declaration)
    model, checkpoint_facts = load_model(
        resolved, tokenizer, device=args.device, dtype=args.dtype
    )
    checkpoint_facts["requested_path"] = str(args.checkpoint)
    print(
        f"  {checkpoint_facts['n_layers']}L x {checkpoint_facts['d_model']}d x "
        f"{checkpoint_facts['n_heads']}h, vocab {checkpoint_facts['vocab_size']}; "
        f"rendering resolved to delimiters {tokenisation.start_id}/{tokenisation.end_id}"
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "provenance": {
            "runner": {
                "path": "scripts/capability/stages/joint_mode_qualification.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "modules": {
                name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES
            },
        },
        "checkpoint": checkpoint_facts,
        "rendering": tokenisation.facts(),
        "seeds": {"cohort_draw": int(args.cohort_draw_seed)},
        "thresholds": {
            "identification_floor_nats": float(args.identification_floor_nats),
            "identification_floor_status": (
                IDENTIFICATION_FLOOR_STATUS
                if args.identification_floor_nats == SCREENING_CONTEXT_INFORMATION_NATS
                else "declared on the command line, overriding the calibrated "
                f"{SCREENING_CONTEXT_INFORMATION_NATS}-nat identification floor"
            ),
            "legacy_qualification_floor_nats": MIN_CONTEXT_INFORMATION_NATS,
            "legacy_qualification_floor_note": LEGACY_QUALIFICATION_FLOOR_NOTE,
        },
        "estimand": (
            "context information per mode: held-out unigram cross-entropy on the "
            "scored symbols minus the model's clean cross-entropy on the same "
            "symbols. A text symbol is a token; a protein symbol is whatever the "
            "family declares as its symbol unit in src.capability.models.joint_modes, so "
            "each mode's record names its own unit, its own support, and -- for "
            "protein -- its measured residues per scored token and whether the "
            "magnitude may be placed beside another arm's at all"
        ),
    }

    modes_record: dict[str, Any] = {}
    statistics: dict[str, ModeStatistics] = {}
    if "protein" in modes:
        modes_record["protein"], statistics["protein"] = protein_mode(
            args, model, tokenisation
        )
    if "text" in modes:
        modes_record["text"], statistics["text"] = text_mode(
            args,
            model,
            tokenizer,
            tokenisation,
            vocab_size=int(checkpoint_facts["vocab_size"]),
        )
    # Written before the report, for cohort_power.py's reason: the report then
    # carries the digest of the sidecar that was actually produced, so a reader
    # learns from the report whether the file on disk is the one this run wrote.
    for name, mode_statistics in statistics.items():
        modes_record[name]["sufficient_statistics"] = write_mode_records(
            args.out,
            mode_statistics,
            seeds={"cohort_draw": int(args.cohort_draw_seed)},
            max_tokens=int(args.max_tokens),
        )
    payload["modes"] = modes_record
    payload["identification_verdicts"] = {
        name: record["identification_verdict"] for name, record in modes_record.items()
    }
    payload["identification_verdict_note"] = IDENTIFICATION_VERDICT_NOTE
    payload["modes_measured"] = list(modes)

    destination = args.out / "joint_mode_qualification.json"
    write_json(destination, payload)
    print()
    for name, record in modes_record.items():
        line = (
            f"[{name}] context information {record['context_information_nats']:+.4f} "
            f"({record['context_information_unit']})  "
            f"{record['identification_verdict']}"
        )
        comparability = record.get("cross_arm_comparability")
        if comparability is not None:
            line += (
                f"  {comparability['verdict']} at "
                f"{comparability['measured_residues_per_scored_token']:.3f} residues/token"
                f"  reversal {record['reversal_cost_nats_per_residue']:+.4f} nats/residue"
            )
        print(line)
        print(f"  records {record['sufficient_statistics']['path']}")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
