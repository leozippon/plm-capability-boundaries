#!/usr/bin/env python3
"""Does a protein decoder predict phenotype on proteins its corpus does not contain?

**What is new here, and it is the only thing that is.** The retrieval bound could never exclude
retrieval, only estimate it: over 187 ProteinGym wild types the least homologous
is still 55.5% identical to a UniRef50 cluster the model trained on. EXP-R2-190
staged a referent where retrieval is excluded *by construction* -- 132 of 148 de
novo designs from Tsuboyama et al. 2023 return no DIAMOND hit against the
60.3 M-cluster staged UniRef50 at the retrieval bound's own gate, against a within-cohort natural
control from the same file, the same assay and matched length that hits at
328 of 330. Designs and naturals therefore differ in whether the corpus contains
their homologues and in nothing else that this file could confound.

**The estimand** is the retrieval bound's, unchanged: per wild type, the Spearman correlation
between an arm's summed log-likelihood of a variant and that variant's measured
folding stability. ``ddG_ML`` is the phenotype, which is the column ProteinGym
itself publishes as ``DMS_score`` for its 64 Tsuboyama assays. It is a
sequence-likelihood estimand, so L31's tokenisation constraint does not bind it.

**Pre-registered before any score existed** (EXP-R2-191, and
``designed_referent.arm_verdict`` is the executable form):

``positive``      the arm's series-clustered interval on MODEL - BASELINE lies
                  wholly above zero for EVERY free and fragment-level baseline on
                  the corpus-disjoint designs, AND the identical conjunction
                  holds on the natural control.
``negative``      the design-side conjunction fails while the control passes.
                  This is a clean negative: the instrument clears its bar where
                  retrieval is available and the model does not clear it where
                  retrieval is excluded.
``uninterpretable_instrument_bound``
                  the control fails. Nothing may then be read from the design
                  side, because the gate was never shown attainable.

The baselines are the free family -- BLOSUM62 and ``profiles.free_baselines``'
position index, hydropathy pair and corpus composition -- plus two fragment-level
retrieval baselines built as proper conditional sequence models over the
**corrected** record-local corpus k-mer background. The fragment baselines are
what replaces the profile LOOKUP, which is empty by construction on a referent
with no homologues: a homologue-free referent is not an information-free one.

**Unit of analysis.** The design series -- topology x round, run-family x run, or
hallucination round -- as EXP-R2-190 enumerated them, resampled under
``profiles.cluster_bootstrap`` at the package's declared 8-unit floor. The
natural control's unit is the dataset's own ``WT_cluster``. Two further readings
are pre-registered rather than chosen afterwards: a strict one over the design
series that are entirely zero-hit, and a topology-level one that is expected to
fall below the floor and is reported with the refusal rather than as a number.

Stages are independent and resumable. ``cohort``, ``baseline`` and ``analyse``
are CPU-only; ``score`` is the only stage that needs a GPU, and it reads the
cohort artefact rather than the staged parquet so that the pod needs no dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.mutation import designed_referent as D  # noqa: E402
from src.capability.core.arms import (  # noqa: E402
    INPUT_FORMAT_GMASK_SOP_EOS,
    PANEL,
    PROGEN3_ARMS,
    REPO,
    STAGED_CANDIDATE_ARMS,
    STAGED_SCALE_ARMS,
    STAGED_SECOND_STAGE_ARMS,
    Cohort,
    arm_spec,
    config_context_length,
    load_arm,
    load_arm_spec,
    rendering_marker_ids,
    tokenize_batch,
)
from src.capability.models import galactica_fitness as G  # noqa: E402
from src.capability.models import instructprotein_fitness as IP  # noqa: E402
from src.capability.models import joint_lineage as L  # noqa: E402
from src.capability.models import progen3 as PG  # noqa: E402
from src.capability.models import proteinglm as PGLM  # noqa: E402
from src.capability.models import rita_fitness as RITA  # noqa: E402
from src.capability.core.scoring import sequence_target_mask, target_rule  # noqa: E402
from src.capability.models.scale_comparison import (  # noqa: E402
    SCORING_STRATA,
    STRATUM_N_TO_C,
)
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.context.kmer_background import count_kmers  # noqa: E402
from src.capability.context.kmer_background import load as load_kmer_background  # noqa: E402
from src.capability.context.kmer_background import save as kmer_background_save  # noqa: E402
from src.capability.core.sequence_metadata import PROTEINGYM_ROOT

from scripts.capability.entrypoints import stage_path

def _load_stage(filename: str) -> Any:
    """One sibling stage module, loaded by path.

    ``train_transcoder.py``'s device for the same problem: a stage whose name
    begins with a digit is not importable, and the alternative -- restating
    EXP-R2-226's qualification contract here -- would be a second declaration of
    when a rung of that lineage may be scored.
    """

    import importlib.util

    path = stage_path(REPO_ROOT, filename)
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


QUALIFICATION = _load_stage("adaptation_stage_qualification.py")

SCHEMA_VERSION = D.SCHEMA_VERSION

#: The declared protein doors stage 29 admits, each a campaign's own name for a
#: set of staged checkpoints: EXP-R2-224's scale rungs, EXP-R2-225's second-stage
#: checkpoints, the budget-only candidate checkpoints and the packed-MoE ProGen3
#: rungs, beside the first-round panel. Listed one by one rather than folded into
#: a predicate, so a door added later is admitted by a decision recorded here and
#: not by matching a shape, and narrower than "any arm in ``arms.py``": a text arm
#: is refused by :func:`~src.capability.core.arms.arm_spec`'s modality and a joint
#: checkpoint reached by path is in none of these tuples.
#:
#: The ProGen3 door is a fourth tuple and not a widening of the others for the
#: reason ``arms._check_progen3_arms`` gives: those rungs are packed MoE and their
#: weights are reached through :func:`src.capability.models.progen3.load_progen3` rather
#: than through an AutoModel call, which loads random experts and still runs.
#: Admitting them here is the decision recorded at
#: :data:`~src.capability.mutation.designed_referent.EXCLUDED_ARMS`: this measurement scores
#: them on the single N-to-C direction and labels the payload accordingly.
ADMITTED_PROTEIN_DOORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("a protein panel arm", tuple(PANEL)),
    ("a staged scale rung", tuple(STAGED_SCALE_ARMS)),
    ("a staged second-stage arm", tuple(STAGED_SECOND_STAGE_ARMS)),
    ("a staged candidate arm", tuple(STAGED_CANDIDATE_ARMS)),
    ("a ProGen3 rung", tuple(PROGEN3_ARMS)),
)

#: The inference precisions this stage may be asked for. Declared once because
#: two of the arms it admits require float32 by their own checkpoint's terms --
#: ``rita_fitness`` refuses any other precision and ProteinGLM's derived serving
#: package does too -- while every other arm is read at the panel's bfloat16.
#: The precision each payload was taken at travels in its ``settings`` block.
DTYPES: tuple[str, ...] = ("bfloat16", "float16", "float32")

STAGES = (
    "cohort",
    "baseline",
    "score",
    "analyse",
    "interaction",
    "background",
    "fragment_order",
)
DEFAULT_OUT = REPO / "results/shared/designed_referent_20260813"
DEFAULT_ARMS = ("protgpt2", "progen2-small", "progen2-base", "progen2-medium")

#: The corpus counted at every order the dense representation can hold. k = 8
#: would need a 204 GB count vector, so 7 is not a preference but the ceiling of
#: this representation, and it is recorded as such.
HIGH_ORDER_BACKGROUND_DIR = REPO / "data/kmer_background/uniref50_high_order"
HOLDOUT_ROOT = REPO / "data/kmer_background/uniref50_holdout"

#: Two disjoint held-out draws rather than one. They are this stage's answer to
#: the standing requirement that a corpus sample report a sensitivity: the two
#: draws are independent samples of the same corpus, so a cross-entropy that
#: moves between them is telling the reader the estimate is not settled.
HOLDOUT_DRAWS: tuple[str, ...] = ("a", "b")


def _admitted_door(name: str) -> str | None:
    """Which declared protein door admits ``name``, or ``None`` if none does."""

    for label, members in ADMITTED_PROTEIN_DOORS:
        if name in members:
            return label
    return None


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist; run the stage that writes it")
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ stage: cohort


def stage_cohort(args: argparse.Namespace) -> dict[str, Any]:
    referent = D.build_referent(
        megascale_dir=args.megascale_dir,
        certificate_dir=args.certificate_dir,
        min_variants=min(D.VARIANT_FLOOR_SWEEP + (args.min_variants,)),
    )
    sweep = _floor_sweep(referent)
    referent = D.Referent(
        wildtypes=tuple(
            wt for wt in referent.wildtypes if len(wt.mutants) >= args.min_variants
        ),
        provenance={
            **referent.provenance,
            "min_variants": int(args.min_variants),
            "counts": D.cohort_counts(referent.wildtypes, min_variants=args.min_variants),
        },
    )
    path = args.out / "cohort.json"
    D.save_referent(referent, path)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "cohort",
        "created_utc": _timestamp(),
        "cohort_sha256": sha256_file(path),
        "provenance": referent.provenance,
        "variant_floor_sweep": sweep,
    }
    write_json(args.out / "cohort_summary.json", summary)
    counts = referent.provenance["counts"]
    print(json.dumps(counts, indent=2, sort_keys=True))
    return summary


def _floor_sweep(referent: D.Referent) -> list[dict[str, Any]]:
    """The cohort at every floor in the declared sweep.

    A constant that changes nothing is worth more than an argued one: the
    MegaScale variant count per wild type is bimodal, so this sweep is expected to
    return the identical cohort at every value, and if it ever does not the
    headline floor is doing work that has to be declared. Counted by filtering the
    one built cohort rather than by rebuilding it, so the sweep cannot disagree
    with the headline about anything except the floor.
    """

    return [
        {"min_variants": floor, **D.cohort_counts(referent.wildtypes, min_variants=floor)}
        for floor in D.VARIANT_FLOOR_SWEEP
    ]


# -------------------------------------------------------------- stage: background
#
# The surviving reading -- ProtGPT2 beats the corpus's own fragment statistics
# on sequences the corpus provably does not contain -- rests on 3-mer and 4-mer
# conditionals, which are a weak model of corpus statistics. Testing it against
# the strongest fragment channel the corpus supports needs two things this stage
# produces: counts at higher k, and a *held-out* natural sample the counts do not
# contain, because the order past which the conditional stops being estimated and
# starts being a lookup has to be measured rather than guessed.


def _record_starts(buffer: bytes) -> np.ndarray:
    """Byte offsets of every FASTA header in a buffer that begins with one.

    A ``>`` inside a header's description is not a record start, so a candidate
    counts only at offset zero or immediately after a newline.
    """

    array = np.frombuffer(buffer, dtype=np.uint8)
    candidates = np.flatnonzero(array == ord(">"))
    if candidates.size == 0:
        return candidates
    keep = (candidates == 0) | (array[np.maximum(candidates - 1, 0)] == ord("\n"))
    return candidates[keep]


def _write_holdout(
    fasta: Path,
    targets: dict[str, Path],
    *,
    seed: int,
    fraction: float,
    chunk_bytes: int,
) -> dict[str, Any]:
    """Split records out of the corpus under a seeded mark, never by file order.

    Each record draws one uniform mark from a seeded stream consumed in file
    order, and the draws partition the unit interval, so the two held-out sets are
    disjoint samples spread over the whole corpus. Taking a prefix instead would
    be the failure this programme has manufactured an effect with three times: a
    FASTA is sorted, and its first records are not its typical ones.

    Windows never span a record, so the counts are additive over records and the
    training background is the corpus background minus these -- exactly, with no
    second pass over 24 GB.
    """

    names = sorted(targets)
    if not names:
        raise ValueError("no held-out draws requested")
    if not 0.0 < fraction * len(names) < 1.0:
        raise ValueError(f"{len(names)} draws of {fraction} do not fit in the corpus")
    rng = np.random.default_rng(seed)
    handles = {name: targets[name].open("wb") for name in names}
    counts = {name: 0 for name in names}
    records = 0
    carry = b""
    started = time.time()
    try:
        with fasta.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                buffer = carry + chunk
                starts = _record_starts(buffer)
                if starts.size < 2:
                    carry = buffer
                    continue
                carry = buffer[int(starts[-1]) :]
                bounds = starts[:-1]
                marks = rng.random(bounds.size)
                records += bounds.size
                for index, name in enumerate(names):
                    chosen = np.flatnonzero(
                        (marks >= index * fraction) & (marks < (index + 1) * fraction)
                    )
                    counts[name] += int(chosen.size)
                    for position in chosen:
                        handles[name].write(
                            buffer[int(starts[position]) : int(starts[position + 1])]
                        )
        if carry:
            records += 1
            mark = float(rng.random(1)[0])
            for index, name in enumerate(names):
                if index * fraction <= mark < (index + 1) * fraction:
                    counts[name] += 1
                    handles[name].write(carry)
    finally:
        for handle in handles.values():
            handle.close()
    return {
        "seed": int(seed),
        "fraction_per_draw": float(fraction),
        "records_scanned": records,
        "records_held_out": counts,
        "wall_seconds": round(time.time() - started, 1),
    }


def _load_or_count(
    directory: Path, fasta: Path, ks: tuple[int, ...], *, chunk_bytes: int
) -> dict[str, Any]:
    """Count ``fasta`` into ``directory``, or reuse a background that already covers ``ks``.

    Reuse is announced and is decided on the manifest's declared ``k``, never on
    the directory existing: a partial background is a fault, not a cache hit.
    """

    manifest = directory / "manifest.json"
    if manifest.is_file():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        have = set(int(k) for k in record["k"])
        if set(ks) <= have:
            print(f"[background] reusing {directory} (k = {sorted(have)})")
            return record
        raise RuntimeError(
            f"{directory} carries k = {sorted(have)} and {sorted(ks)} was asked for; "
            "move it aside rather than mixing two passes"
        )
    print(f"[background] counting {fasta} at k = {list(ks)}", flush=True)
    background = count_kmers(fasta, ks=ks, chunk_bytes=chunk_bytes)
    return kmer_background_save(background, directory)


def stage_background(args: argparse.Namespace) -> dict[str, Any]:
    ks = tuple(sorted(set(int(k) for k in args.background_ks)))
    corpus_dir = args.high_order_background_dir
    holdout_root = args.holdout_dir
    holdout_root.mkdir(parents=True, exist_ok=True)
    print(f"[background] corpus fasta {args.corpus_fasta}")
    print(f"[background] corpus counts {corpus_dir}")
    print(f"[background] held-out root {holdout_root}")
    corpus = _load_or_count(
        corpus_dir, args.corpus_fasta, ks, chunk_bytes=args.background_chunk_bytes
    )
    targets = {name: holdout_root / f"holdout_{name}.fasta" for name in HOLDOUT_DRAWS}
    split_path = holdout_root / "split.json"
    if all(path.is_file() for path in targets.values()) and split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        print(f"[background] reusing held-out draws {split['records_held_out']}")
    else:
        split = _write_holdout(
            args.corpus_fasta,
            targets,
            seed=args.holdout_seed,
            fraction=args.holdout_fraction,
            chunk_bytes=args.background_chunk_bytes,
        )
        split["source"] = str(args.corpus_fasta)
        write_json(split_path, split)
        print(f"[background] held out {split['records_held_out']}")
    holdouts = {
        name: _load_or_count(
            holdout_root / name, targets[name], ks, chunk_bytes=args.background_chunk_bytes
        )
        for name in HOLDOUT_DRAWS
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "background",
        "created_utc": _timestamp(),
        "k": list(ks),
        "corpus": corpus,
        "split": split,
        "holdouts": holdouts,
    }
    write_json(holdout_root / "background_summary.json", payload)
    return payload


# ---------------------------------------------------------------- stage: baseline


def stage_baseline(args: argparse.Namespace) -> dict[str, Any]:
    referent = D.load_referent(args.cohort or args.out / "cohort.json")
    background = load_kmer_background(args.kmer_background_dir)
    background_record = D.require_corrected_background(background)
    print(f"[baseline] k-mer background {background_record['source']}")
    print(f"[baseline] totals {background_record['totals']}")
    log_conditional = {
        k: D.conditional_log_probabilities(background, k) for k in (3, 4)
    }
    residue_background = D.corpus_residue_background(args.retrieval_bound_dir)

    rows: dict[str, dict[str, Any]] = {}
    for index, wildtype in enumerate(referent.wildtypes):
        scores = D.baseline_scores(
            wildtype,
            residue_background=residue_background,
            log_conditional=log_conditional,
        )
        rows[wildtype.name] = {
            "kind": wildtype.kind,
            "unit": wildtype.unit,
            "n_variants": len(wildtype.mutants),
            "spearman": {
                name: D.spearman(value, wildtype.phenotype)
                for name, value in scores.items()
            },
        }
        if index % 50 == 0:
            print(f"[baseline] {index + 1}/{len(referent.wildtypes)} {wildtype.name}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "baseline",
        "created_utc": _timestamp(),
        "cohort_sha256": sha256_file(args.cohort or args.out / "cohort.json"),
        "kmer_background": background_record,
        "residue_background_source": str(args.retrieval_bound_dir),
        "baselines": list(D.BASELINES),
        "wildtypes": rows,
    }
    write_json(args.out / "baselines.json", payload)
    return payload


# ------------------------------------------------------------------- stage: score


class _ArmLikelihood:
    """Summed log-likelihood of a sequence under one arm, in its own rendering.

    The convention is stage 20's: render through ``Cohort.input_strings`` so the
    panel decides the input format, then sum the token log-probabilities over the
    positions the arm's own rendering declares as content. ProtGPT2's FASTA
    wrapping is worth 1.42 nats/token and is not this stage's decision to make,
    and neither is the scored span: a rendering that prefixes markers puts them in
    front of the content, and only the first token of a batch is context by
    construction, so a second marker -- ProtGPT3's direction token -- would
    otherwise be scored as content. What was computed travels in
    :attr:`score_description` and under which convention the sum was taken in
    :attr:`scoring_stratum`.
    """

    score_description = (
        "summed log-likelihood of the rendered variant's scored targets; the ids "
        "this arm's rendering declares as markers are context, not targets"
    )
    scoring_stratum = STRATUM_N_TO_C

    def __init__(self, name: str, *, device: str, dtype: str, batch_size: int) -> None:
        import torch

        self.torch = torch
        self.name = name
        self.batch_size = batch_size
        spec = arm_spec(name)
        if spec.input_format == "ec_conditioned":
            raise ValueError(
                f"{name}: EC-conditioned scoring is not this door's; the estimand "
                "is undefined on a de novo referent that carries no EC number"
            )
        if spec.input_format == INPUT_FORMAT_GMASK_SOP_EOS:
            raise ValueError(
                f"{name}: ProteinGLM scoring is residues 2..L under its native "
                "prefix; _ArmLikelihood would score the prefix and residue 1"
            )
        if spec.architecture == "rita":
            raise ValueError(
                f"{name}: RITA scoring is its own native encoder, because its "
                "tokenizer declares no pad token and tokenize_batch refuses every "
                "batch; _ArmLikelihood cannot render it"
            )
        if spec.architecture == "progen3":
            raise ValueError(
                f"{name}: ProGen3 is scored bidirectionally through its own "
                "loader; tokenize_batch would drop sequence_ids and AutoModel "
                "would score random experts"
            )
        self.arm = (
            load_arm(name, device=device, dtype=dtype)
            if name in PANEL
            else load_arm_spec(spec, device=device, dtype=dtype)
        )
        self.context = config_context_length(self.arm.model.config)
        # Resolved at construction rather than inside log_likelihood, so an arm
        # whose rendering declares no resolvable marker refuses before a wild type
        # is scored instead of hours into the cell.
        self.target_rule = target_rule(spec.input_format)
        self.markers = rendering_marker_ids(self.arm)

    def render(self, sequences: list[str]) -> list[str]:
        cohort = Cohort(
            name="designed_referent",
            kind="protein",
            records=list(sequences),
            min_symbols=min(len(s) for s in sequences),
            max_symbols=max(len(s) for s in sequences),
            metadata={},
        )
        return cohort.input_strings(self.arm)

    def token_lengths(self, texts: list[str]) -> list[int]:
        encoded = self.arm.tokenizer(list(texts), return_tensors=None)["input_ids"]
        return [len(row) for row in encoded]

    def log_likelihood(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        totals = np.empty(len(texts), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                chunk = texts[start : start + self.batch_size]
                ids, mask = tokenize_batch(self.arm, chunk, self.context)
                ids = ids.to(self.arm.device)
                mask = mask.to(self.arm.device)
                logits = self.arm.model(input_ids=ids, attention_mask=mask).logits
                logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                targets = ids[:, 1:]
                token = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                keep = sequence_target_mask(
                    ids,
                    mask,
                    rule=self.target_rule,
                    marker_token_ids=self.markers,
                )
                totals[start : start + len(chunk)] = (
                    (token * keep).sum(1).double().cpu().numpy()
                )
        return totals

    def release(self) -> None:
        del self.arm
        self.torch.cuda.empty_cache()


class _RitaLikelihood:
    """RITA-xl under its own native document rendering, at float32.

    A thin adapter over :class:`src.capability.models.rita_fitness.RitaFitnessScorer`,
    which owns the encoding, the span and the arithmetic. It takes raw residue
    strings rather than rendered text because its rendering is not a string
    transformation this stage may re-do: the boundary is a declared token and the
    terminal ``<EOS>`` comes from its own post-processor, and
    ``rita_fitness._encode_native`` verifies the ids of both. Its span is the
    sequence's residues -- the boundary is the first token and is context by
    construction, and the terminator carries the boundary's id and is the marker
    the rendering declares -- so a full-protein sum would fold the terminator's own
    next-token term into a stability reading.

    Batches are the arm's own equal-length substitution batches: the scorer
    refuses a variable-length batch rather than padding, because the encoding
    assigns no pad token and the forward pass takes no attention mask.
    """

    def __init__(self, name: str, *, device: str, dtype: str, batch_size: int) -> None:
        self.name = name
        self.scorer = RITA.RitaFitnessScorer(
            checkpoint=arm_spec(name).path,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        self.context = self.scorer.context

    @property
    def score_description(self) -> str:
        return self.scorer.score_description

    @property
    def scoring_stratum(self) -> str:
        return self.scorer.scoring_stratum

    def render(self, sequences: list[str]) -> list[str]:
        """The records :meth:`log_likelihood` consumes: raw residue strings."""

        return list(sequences)

    def token_lengths(self, sequences: list[str]) -> list[int]:
        return self.scorer.token_lengths(sequences)

    def log_likelihood(self, sequences: list[str]) -> np.ndarray:
        return self.scorer.log_likelihood(sequences)

    def release(self) -> None:
        self.scorer.release()


class _ProteinGLMLikelihood:
    """ProteinGLM continuation: native prefix, residues 2..L, float32.

    A different scoring functional from a full-protein left-to-right sum, and
    labelled as what it is rather than described as one: the native
    ``<gmask><sop><eos>`` prefix and residue 1 are not scored. Stage 20's own
    door, reproduced here because the estimand and the precision are properties of
    the checkpoint rather than of the queue it is read on.

    ``render`` returns the records :meth:`log_likelihood` consumes, which for this
    arm are the raw residue strings: :meth:`token_lengths` must be total -- it
    measures, it does not refuse -- so that a wild type whose rendering exceeds
    the serving window becomes a recorded skip rather than a failed cell.
    """

    score_description = (
        "summed log-likelihood of residues 2..L under the native "
        "<gmask><sop><eos> prefix; residue 1 and the prefix are not scored"
    )
    scoring_stratum = STRATUM_N_TO_C

    def __init__(self, name: str, *, device: str, dtype: str, batch_size: int) -> None:
        import torch

        if name != "proteinglm-7b-clm":
            raise ValueError(f"ProteinGLM scoring is proteinglm-7b-clm, not {name!r}")
        if dtype != "float32":
            raise ValueError(
                f"ProteinGLM scoring is float32-only; got {dtype!r}. The "
                "checkpoint declares float32 and its derived serving package "
                "refuses another precision"
            )
        spec = arm_spec(name)
        if spec.input_format != INPUT_FORMAT_GMASK_SOP_EOS:
            raise ValueError(
                f"{name}: expected {INPUT_FORMAT_GMASK_SOP_EOS!r}, got "
                f"{spec.input_format!r}"
            )
        self.torch = torch
        self.name = name
        self.batch_size = batch_size
        self.arm = load_arm_spec(spec, device=device, dtype=dtype)
        self.context = PGLM.CONTEXT_LENGTH
        declared = config_context_length(self.arm.model.config)
        if declared != self.context:
            raise ValueError(
                f"{name}: serving window is {self.context}, config context is "
                f"{declared}"
            )

    def render(self, sequences: list[str]) -> list[str]:
        return list(sequences)

    def token_lengths(self, sequences: list[str]) -> list[int]:
        return [PGLM.budget_token_length(sequence) for sequence in sequences]

    def log_likelihood(self, sequences: list[str]) -> np.ndarray:
        torch = self.torch
        texts = [PGLM.render_budget_sequence(sequence) for sequence in sequences]
        totals = np.empty(len(texts), dtype=np.float64)
        rule = target_rule(INPUT_FORMAT_GMASK_SOP_EOS)
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                chunk = texts[start : start + self.batch_size]
                ids, mask = tokenize_batch(self.arm, chunk, self.context)
                ids = ids.to(self.arm.device)
                mask = mask.to(self.arm.device)
                keep = sequence_target_mask(ids, mask, rule=rule)
                logits = self.arm.model(input_ids=ids, attention_mask=mask).logits
                logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                token = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                totals[start : start + len(chunk)] = (
                    (token * keep).sum(1).double().cpu().numpy()
                )
        return totals

    def release(self) -> None:
        del self.arm
        self.torch.cuda.empty_cache()


class _ProGen3Likelihood:
    """One ProGen3 rung, scored left-to-right on the single N-to-C direction.

    **This measurement's decision, and a declared deviation.** The lineage's own
    published convention is the mean of the N-to-C and C-to-N summed
    log-likelihoods, which ``retrieval_bound.py`` scores. The estimand here is
    one functional compared across many arms, and a bidirectional mean is a
    functional no other arm in this stage carries, so ProGen3 is read on the
    direction every other arm here is read on and the payload records
    :data:`~src.capability.models.scale_comparison.STRATUM_N_TO_C`. That is what makes the
    number readable beside the rest of this measurement and **not** readable
    beside the lineage's bidirectional ProteinGym reading: the two are separate
    functionals and are never pooled, ranked together or converted into one
    another. The deviation is declared here, in the payload and in
    :data:`~src.capability.mutation.designed_referent.ARM_IDENTIFICATION`, and is not
    inferred from an arm name.

    The rendering is the lineage's own: ``prepare_clm`` builds
    ``"1" + sequence + "2"`` and wraps it in ``<bos>`` and ``<eos>``, so with
    ``reverse=False`` the direction marker is the lineage's N-to-C marker rather
    than one this stage invents. **The scored targets are the sequence's residues
    1..L and nothing else.** ``progen3.scored_logits`` applies
    ``residue_target_mask``, which excludes ``"1"``, ``"2"``, ``<eos>``,
    ``<bos>``, ``<pad>`` and ``<mask>`` from the target set; before that repair
    the three marker positions of every record were inside the sum, and no other
    arm in this stage scores a marker, so the magnitude is not the one a
    marker-inclusive reading produces.

    Weights are reached through :func:`src.capability.models.progen3.load_progen3` and not
    through AutoModel, because the Hugging Face eager MoE path loads random
    experts and still runs, and :func:`src.capability.models.progen3.self_check` is run
    before a wild type is scored. ``context`` is the checkpoint's own declared
    ceiling -- 65536 on both rungs, against a rendering of L + 4 tokens and a
    cohort whose longest wild type is 72 residues -- so the over-context skip is
    reachable and does not fire here.
    """

    score_description = (
        "summed left-to-right N-to-C log-likelihood of the sequence's residues "
        "1..L under the lineage's own <bos>\"1\"<sequence>\"2\"<eos> rendering; "
        "the \"1\", \"2\", <bos> and <eos> markers are context, not targets"
    )
    scoring_stratum = STRATUM_N_TO_C

    def __init__(self, name: str, *, device: str, dtype: str, batch_size: int) -> None:
        import torch

        from src.capability.models.progen3 import load_progen3, self_check

        torch_dtype = PG.PROGEN3_DTYPES.get(dtype)
        if torch_dtype is None:
            raise ValueError(
                f"{name}: ProGen3's attention is pinned to the flash SDPA backend, "
                f"which has no kernel for {dtype!r}; supported: "
                f"{sorted(PG.PROGEN3_DTYPES)}"
            )
        self.torch = torch
        self.name = name
        self.batch_size = batch_size
        self.pg = load_progen3(arm_spec(name).path, device=device, dtype=torch_dtype)
        # Resolved from the loaded config's own fingerprint, so a rung gated
        # against another rung's band is a KeyError here rather than a number.
        self.check = self_check(self.pg)
        self.context = config_context_length(self.pg.config)

    def render(self, sequences: list[str]) -> list[str]:
        """The records :meth:`log_likelihood` consumes: raw residue strings."""

        return list(sequences)

    def token_lengths(self, sequences: list[str]) -> list[int]:
        """The rendered width, measured from the lineage's own preparer.

        Measured rather than computed as L + 4: the preparer decides how many ids
        a residue becomes, and a probe that disagreed with the encoding it
        measures would make the over-context skip fire on the wrong records.
        """

        return [
            int(self.pg.batch([sequence], reverse=False)["input_ids"].shape[-1])
            for sequence in sequences
        ]

    def log_likelihood(self, sequences: list[str]) -> np.ndarray:
        from src.capability.models.progen3 import scored_logits, token_nll

        torch = self.torch
        totals = np.empty(len(sequences), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                chunk = list(sequences[start : start + self.batch_size])
                batch = self.pg.batch(chunk, reverse=False)
                logits, targets, mask = scored_logits(self.pg, batch)
                nll = token_nll(logits, targets)
                totals[start : start + len(chunk)] = (
                    -(nll * mask).sum(1).double().cpu().numpy()
                )
        if not np.isfinite(totals).all():
            raise FloatingPointError(f"{self.name}: a non-finite N-to-C total")
        return totals

    def release(self) -> None:
        del self.pg
        self.torch.cuda.empty_cache()


class _GalacticaLikelihood:
    """One Galactica rung under the declared protein rendering, at float32.

    A thin adapter over :class:`src.capability.models.galactica_fitness.GalacticaFitnessScorer`,
    which is the ProteinGym door. It takes raw residue strings rather than
    rendered text because that scorer owns the ``[START_AMINO]...[END_AMINO]``
    rendering and the protein-content span. Float32 is required so the arithmetic
    is the one ProteinGym recorded for the same checkpoint, not a faster one.
    """

    def __init__(self, name: str, *, device: str, dtype: str, batch_size: int) -> None:
        if dtype != "float32":
            raise ValueError(
                f"Galactica MegaScale scoring is float32-only; got {dtype!r}. "
                "The ProteinGym door scores these rungs at float32"
            )
        loaded = G.load_galactica(name, device=device, dtype=dtype)
        self.name = name
        self.scorer = G.GalacticaFitnessScorer(loaded, batch_size=batch_size)
        self.context = self.scorer.context

    @property
    def score_description(self) -> str:
        return self.scorer.score_description

    @property
    def scoring_stratum(self) -> str:
        return self.scorer.scoring_stratum

    def render(self, sequences: list[str]) -> list[str]:
        """The records :meth:`log_likelihood` consumes: raw residue strings."""

        return list(sequences)

    def token_lengths(self, sequences: list[str]) -> list[int]:
        return self.scorer.token_lengths(sequences)

    def log_likelihood(self, sequences: list[str]) -> np.ndarray:
        return self.scorer.log_likelihood(sequences)

    def residue_accounting(self) -> dict[str, Any]:
        return self.scorer.residue_accounting()

    def release(self) -> None:
        self.scorer.release()


class _InstructProteinLikelihood:
    """InstructProtein protein mode under the declared residue-token rendering.

    A thin adapter over
    :class:`src.capability.models.instructprotein_fitness.InstructProteinFitnessScorer`,
    which is the ProteinGym door. It takes raw residue strings because that
    scorer owns the ``<protein>ƤA..ƤY</protein>`` rendering and the
    protein-content span. Text mode is not a scoring door.
    """

    def __init__(self, name: str, *, device: str, dtype: str, batch_size: int) -> None:
        loaded = IP.load_instructprotein(name, device=device, dtype=dtype)
        self.name = name
        self.scorer = IP.InstructProteinFitnessScorer(loaded, batch_size=batch_size)
        self.context = self.scorer.context

    @property
    def score_description(self) -> str:
        return self.scorer.score_description

    @property
    def scoring_stratum(self) -> str:
        return self.scorer.scoring_stratum

    def render(self, sequences: list[str]) -> list[str]:
        """The records :meth:`log_likelihood` consumes: raw residue strings."""

        return list(sequences)

    def token_lengths(self, sequences: list[str]) -> list[int]:
        return self.scorer.token_lengths(sequences)

    def log_likelihood(self, sequences: list[str]) -> np.ndarray:
        return self.scorer.log_likelihood(sequences)

    def residue_accounting(self) -> dict[str, Any]:
        return self.scorer.residue_accounting()

    def release(self) -> None:
        self.scorer.release()


class _JointRungLikelihood:
    """One rung of the ProLLaMA lineage, under the bare ``Seq=<...>`` block.

    The same three-method shape :class:`_ArmLikelihood` has -- render, then
    measure the renderings -- over
    :class:`src.capability.models.joint_lineage.BareBlockScorer`, which owns the
    rendering, the scored span and the arithmetic. What it renders to is a
    record rather than a string, which is the whole difference: the sum is taken
    over the residue-spelling token run the rendering declares and over none of
    the four wrapper tokens.

    **The estimand is not the panel arms'.** These tokens are merged
    multi-residue SentencePiece pieces, not single residues, so a magnitude from
    this rung is in nats per token and is not commensurable with a residue-unit
    family's (Appendix B rule 26, limitation L23). The measured rate travels in
    the payload.
    """

    def __init__(self, name: str, *, device: str, dtype: str, batch_size: int) -> None:
        loaded = L.load_rung(name, device=device, dtype=dtype)
        self.scorer = L.BareBlockScorer(loaded, batch_size=batch_size)
        self.name = name
        self.context = loaded.context
        self.facts = loaded.facts

    def render(self, sequences: list[str]) -> list[Any]:
        return self.scorer.render(sequences)

    def token_lengths(self, rendered: list[Any]) -> list[int]:
        return self.scorer.token_lengths(rendered)

    def log_likelihood(self, rendered: list[Any]) -> np.ndarray:
        return self.scorer.log_likelihood(rendered)

    def residue_accounting(self) -> dict[str, Any]:
        return self.scorer.residue_accounting()

    def release(self) -> None:
        self.scorer.release()


def _require_declared_fp32_policy(address: str) -> dict[str, Any]:
    """The live fp32 matmul policy, refused unless it is the declared one.

    Two of the arms this door admits are scored at float32, and at float32 a
    TF32 matmul is a different arithmetic rather than a faster one, so a reading
    taken under it is not commensurable with the one stage 20 recorded for the
    same checkpoint. The policy is declared once, in ``precision_policy``, and
    this is a check against that declaration and not a second copy of it.

    The two fields that govern a matmul -- ``cuda.matmul.allow_tf32`` and
    ``get_float32_matmul_precision`` -- are compared against the declaration.
    ``cudnn.allow_tf32`` is reported rather than required: it governs padded
    convolution, which neither of these decoders calls, and it is ``True`` by
    torch default. All three observed fields travel in the payload, so the
    policy the numbers were taken under is read rather than assumed.
    """

    import torch

    from src.capability.core import precision_policy as PP

    observed = PP.snapshot_matmul_policy(torch)
    requested = PP.requested_fp32_matmul_policy()
    for field in ("cuda_matmul_allow_tf32", "float32_matmul_precision"):
        if observed[field] != requested[field]:
            raise RuntimeError(
                f"{address}: scored at float32, whose arithmetic depends on the "
                f"matmul policy; {field} is {observed[field]!r} rather than the "
                f"declared {requested[field]!r}"
            )
    return {"requested": requested, "observed": observed}


def _joint_rendering_settings(scorer: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Settings block shared by the two ProteinGym rendering doors."""

    settings = {
        "checkpoint": scorer.scorer.facts["checkpoint"],
        "input_format": (
            f"{scorer.scorer.loaded.tokenisation.declaration.name} declared "
            "protein rendering, context=None"
        ),
        "context": scorer.context,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "score": scorer.score_description,
        "scoring_stratum": scorer.scoring_stratum,
        "checkpoint_facts": scorer.scorer.facts,
        "scientific_role": scorer.scorer.facts["scientific_role"],
        "door": "a ProteinGym protein-mode rendering door",
    }
    if args.dtype == "float32":
        settings["precision_policy"] = _require_declared_fp32_policy(scorer.name)
    return settings


def _open_scorer(name: str, args: argparse.Namespace) -> tuple[Any, dict[str, Any], dict[str, str]]:
    """One arm, one EXP-R2-226 rung, or one ProteinGym rendering door.

    The three doors are separate all the way down, deliberately. A panel arm is
    resolved through ``arms.arm_spec`` and carries an ``ARM_IDENTIFICATION``
    entry; a ProLLaMA rung is not in ``arms.py`` and carries a
    ``JOINT_LINEAGE_IDENTIFICATION`` entry; a Galactica or InstructProtein name
    is reached the same way ProteinGym reaches it and carries a
    ``JOINT_RENDERING_IDENTIFICATION`` entry. No table is indexed with a name
    another door admitted.
    """

    if name in G.GALACTICA_RUNGS:
        scorer = _GalacticaLikelihood(
            name, device=args.device, dtype=args.dtype, batch_size=args.batch_size
        )
        return scorer, _joint_rendering_settings(scorer, args), dict(
            D.JOINT_RENDERING_IDENTIFICATION[name]
        )
    if name in IP.CHECKPOINTS:
        scorer = _InstructProteinLikelihood(
            name, device=args.device, dtype=args.dtype, batch_size=args.batch_size
        )
        return scorer, _joint_rendering_settings(scorer, args), dict(
            D.JOINT_RENDERING_IDENTIFICATION[name]
        )
    if name in L.RUNGS:
        qualification = QUALIFICATION.read_verdict(
            args.joint_qualification_dir, name, dtype=args.dtype
        )
        scorer = _JointRungLikelihood(
            name, device=args.device, dtype=args.dtype, batch_size=args.batch_size
        )
        settings = {
            "checkpoint": scorer.facts["checkpoint"],
            "input_format": f"{L.RENDERING_FAMILY}:{L.PROTEIN_MODE} bare block",
            "context": scorer.context,
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "score": scorer.scorer.score_description,
            "scoring_stratum": scorer.scorer.scoring_stratum,
            "checkpoint_facts": scorer.facts,
            "qualification": {
                "verdict": qualification["verdict"],
                "nll_self_check": qualification["nll_self_check"]["verdict"],
                "directional_reversal": qualification["directional_reversal"]["verdict"],
                "directional_reversal_cost_nats_per_scored_token": (
                    qualification["directional_reversal"].get(
                        "cost_nats_per_scored_token"
                    )
                ),
            },
        }
        return scorer, settings, dict(D.JOINT_LINEAGE_IDENTIFICATION[name])
    if name in D.EXCLUDED_ARMS:
        raise KeyError(f"{name} is excluded: {D.EXCLUDED_ARMS[name]}")
    spec = arm_spec(name)
    if spec.modality != "protein":
        raise KeyError(f"{name} is not a protein arm")
    door = _admitted_door(name)
    if door is None:
        raise KeyError(
            f"{name} is not "
            + ", ".join(label for label, _ in ADMITTED_PROTEIN_DOORS[:-1])
            + f" or {ADMITTED_PROTEIN_DOORS[-1][0]}"
        )
    if spec.input_format == INPUT_FORMAT_GMASK_SOP_EOS:
        scorer: Any = _ProteinGLMLikelihood(
            name, device=args.device, dtype=args.dtype, batch_size=args.batch_size
        )
    elif spec.architecture == "rita":
        scorer = _RitaLikelihood(
            name, device=args.device, dtype=args.dtype, batch_size=args.batch_size
        )
    elif spec.architecture == "progen3":
        scorer = _ProGen3Likelihood(
            name, device=args.device, dtype=args.dtype, batch_size=args.batch_size
        )
    else:
        scorer = _ArmLikelihood(
            name, device=args.device, dtype=args.dtype, batch_size=args.batch_size
        )
    settings = {
        "checkpoint": str(spec.path),
        "input_format": spec.input_format,
        "context": scorer.context,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "score": scorer.score_description,
        "scoring_stratum": scorer.scoring_stratum,
        "door": door,
    }
    if spec.architecture == "progen3":
        settings["self_check"] = scorer.check
    if args.dtype == "float32":
        settings["precision_policy"] = _require_declared_fp32_policy(name)
    return scorer, settings, dict(D.ARM_IDENTIFICATION[name])


def stage_score(args: argparse.Namespace) -> dict[str, Any]:
    cohort_path = args.cohort or args.out / "cohort.json"
    referent = D.load_referent(cohort_path)
    digest = sha256_file(cohort_path)
    results: dict[str, Any] = {}
    for name in args.arms:
        print(f"[score] {name} over {len(referent.wildtypes)} wild types", flush=True)
        scorer, settings, identification = _open_scorer(name, args)
        rows: dict[str, Any] = {}
        skipped: list[dict[str, Any]] = []
        blocks: list[np.ndarray] = []
        order: list[str] = []
        for index, wildtype in enumerate(referent.wildtypes):
            texts = scorer.render(wildtype.sequences())
            lengths = scorer.token_lengths(texts)
            if max(lengths) > scorer.context:
                skipped.append(
                    {
                        "wildtype": wildtype.name,
                        "max_tokens": int(max(lengths)),
                        "context": scorer.context,
                        "reason": "the rendered variant exceeds this arm's context",
                    }
                )
                continue
            scores = scorer.log_likelihood(texts)
            rows[wildtype.name] = {
                "kind": wildtype.kind,
                "unit": wildtype.unit,
                "n_variants": len(wildtype.mutants),
                "max_tokens": int(max(lengths)),
                "spearman": D.spearman(scores, wildtype.phenotype),
            }
            blocks.append(scores)
            order.append(wildtype.name)
            if index % 25 == 0:
                print(
                    f"[score] {name} {index + 1}/{len(referent.wildtypes)} "
                    f"{wildtype.name} {rows[wildtype.name]['spearman']}",
                    flush=True,
                )
        if not blocks:
            raise RuntimeError(f"{name}: no wild type was scored")
        np.savez_compressed(
            args.out / f"model_{name}.npz",
            scores=np.concatenate(blocks),
            offsets=np.cumsum([0] + [block.size for block in blocks]),
            wildtypes=np.array(order),
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "stage": "score",
            "arm": name,
            "created_utc": _timestamp(),
            "cohort_sha256": digest,
            "identification": identification,
            "settings": {
                **settings,
                **(
                    {"symbol_unit_accounting": scorer.residue_accounting()}
                    if hasattr(scorer, "residue_accounting")
                    else {}
                ),
            },
            "wildtypes": rows,
            "skipped": skipped,
        }
        # Written last: the driver treats a JSON in the output directory as
        # completion, so the array must already be on disk when it appears.
        write_json(args.out / f"model_{name}.json", payload)
        results[name] = payload
        scorer.release()
    return results


# ----------------------------------------------------------------- stage: analyse


def _side_report(
    model: dict[str, float],
    baselines: dict[str, dict[str, float]],
    units: dict[str, str],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    names = sorted(model)
    report: dict[str, Any] = {
        "n_wildtypes": len(names),
        "n_units": len({units[name] for name in names}),
        "model": D.unit_bootstrap(
            [model[name] for name in names],
            [units[name] for name in names],
            resamples=resamples,
            seed=seed,
        ),
        "baselines": {},
        "contrasts": {},
    }
    gates: dict[str, bool] = {}
    for name in D.BASELINES:
        values = baselines[name]
        shared = sorted(set(values) & set(model))
        report["baselines"][name] = D.unit_bootstrap(
            [values[key] for key in shared],
            [units[key] for key in shared],
            resamples=resamples,
            seed=seed,
        )
        contrast = D.channel_comparison(
            model, values, units, resamples=resamples, seed=seed
        )
        report["contrasts"][name] = contrast
        gates[name] = bool(contrast["beats_baseline"])
    report["gates"] = gates
    return report


def _post_hoc(
    referent: D.Referent,
    baseline_payload: dict[str, Any],
    models: dict[str, dict[str, Any]],
    *,
    proteingym_dir: Path,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Two controls computed AFTER the pre-registered reading, and labelled so.

    Neither can change the verdict, which is decided by the frozen rule on the
    headline reading. They exist because the two questions a reader asks of a
    design-side null are "is the instrument the same one the retrieval bound used" and "is this
    about designed sequence or about short sequence", and answering them from
    the artefacts is better than answering them in prose.
    """

    by = {wt.name: wt for wt in referent.wildtypes}
    units = {name: by[name].unit for name in by}
    naturals = [wt for wt in referent.side("natural")]
    designs = [wt for wt in referent.side("design")]

    # 1. The same estimand on the 64 ProteinGym Tsuboyama wild types, which is
    #    the cohort EXP-R2-189 section 6 read the retrieval-bound artefacts over.
    assays = sorted(Path(proteingym_dir).glob("*Tsuboyama*.csv"))
    identifiers = {path.stem.split("_")[-1].upper() for path in assays}
    matched = sorted(
        wt.name
        for wt in naturals
        if "_" not in wt.name and wt.name.split(".")[0].upper() in identifiers
    )
    cross_check: dict[str, Any] = {
        "n_proteingym_tsuboyama_assays": len(assays),
        "n_matched_in_this_cohort": len(matched),
        "reference": dict(D.PROTEINGYM_TSUBOYAMA_REFERENCE),
        "reference_source": "EXP-R2-189 section 6, read off the retrieval-bound artefacts",
        "unweighted_mean_spearman": {},
    }
    for arm, payload in models.items():
        values = [
            payload["wildtypes"][name]["spearman"]
            for name in matched
            if payload["wildtypes"].get(name, {}).get("spearman") is not None
        ]
        cross_check["unweighted_mean_spearman"][arm] = float(np.mean(values))

    # 2. The natural control restricted to the designs' own length bands.
    bands: list[dict[str, Any]] = []
    for low, high in D.design_length_bands(designs):
        names = {wt.name for wt in naturals if low <= len(wt.sequence) <= high}
        band: dict[str, Any] = {
            "band": [low, high],
            "n_naturals": len(names),
            "n_clusters": len({by[name].cluster for name in names}),
            "arms": {},
        }
        for arm, payload in models.items():
            model = {
                name: float(payload["wildtypes"][name]["spearman"])
                for name in names
                if payload["wildtypes"].get(name, {}).get("spearman") is not None
            }
            entry: dict[str, Any] = {
                "model": D.unit_bootstrap(
                    [model[name] for name in sorted(model)],
                    [units[name] for name in sorted(model)],
                    resamples=resamples,
                    seed=seed,
                ),
                "contrasts": {},
            }
            for channel in D.BASELINES:
                values = {
                    name: float(baseline_payload["wildtypes"][name]["spearman"][channel])
                    for name in names
                    if baseline_payload["wildtypes"][name]["spearman"][channel] is not None
                }
                entry["contrasts"][channel] = D.channel_comparison(
                    model, values, units, resamples=resamples, seed=seed
                )
            band["arms"][arm] = entry
        bands.append(band)

    return {
        "preregistered": False,
        "note": (
            "computed after the pre-registered reading and unable to change it; "
            "the verdict is decided by arm_verdict on the headline reading alone"
        ),
        "instrument_cross_check": cross_check,
        "length_matched_control": {
            "design_length_summary": {
                "min": min(len(wt.sequence) for wt in designs),
                "median": int(
                    np.median([len(wt.sequence) for wt in designs])
                ),
                "max": max(len(wt.sequence) for wt in designs),
            },
            "natural_length_summary": {
                "min": min(len(wt.sequence) for wt in naturals),
                "median": int(
                    np.median([len(wt.sequence) for wt in naturals])
                ),
                "max": max(len(wt.sequence) for wt in naturals),
            },
            "bands": bands,
        },
    }


def stage_analyse(args: argparse.Namespace) -> dict[str, Any]:
    cohort_path = args.cohort or args.out / "cohort.json"
    referent = D.load_referent(cohort_path)
    digest = sha256_file(cohort_path)
    baseline_payload = _read(args.out / "baselines.json")
    if baseline_payload["cohort_sha256"] != digest:
        raise RuntimeError("the baselines were computed on a different cohort")

    by_name = {wt.name: wt for wt in referent.wildtypes}
    zero_hit = {wt.name for wt in referent.side("design")}
    clean_series = {wt.name for wt in referent.side("design") if wt.series_zero_hit}
    naturals = {wt.name for wt in referent.side("natural")}

    def baseline_channel(names: set[str], key: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in names:
            entry = baseline_payload["wildtypes"].get(name)
            if entry is None:
                continue
            value = entry["spearman"][key]
            if value is not None:
                out[name] = float(value)
        return out

    arms: dict[str, Any] = {}
    models: dict[str, dict[str, Any]] = {}
    for arm in args.arms:
        path = args.out / f"model_{arm}.json"
        if not path.is_file():
            print(f"[analyse] no model scores for {arm}; skipping")
            continue
        model_payload = _read(path)
        models[arm] = model_payload
        if model_payload["cohort_sha256"] != digest:
            raise RuntimeError(f"{arm} was scored on a different cohort")
        # A payload that records no stratum predates the field and is left as it
        # is -- it cannot vouch for what it was scored under, and neither can
        # this stage. A payload that records one must record a declared one, so a
        # stratum spelled fresh beside an arm cannot travel into a report.
        recorded_stratum = model_payload["settings"].get("scoring_stratum")
        if recorded_stratum is not None and recorded_stratum not in SCORING_STRATA:
            raise RuntimeError(
                f"{arm} records scoring stratum {recorded_stratum!r}, which is "
                f"not one of the declared {list(SCORING_STRATA)}"
            )
        model_all = {
            name: float(entry["spearman"])
            for name, entry in model_payload["wildtypes"].items()
            if entry["spearman"] is not None
        }

        def side(names: set[str], units: dict[str, str], label: str) -> dict[str, Any]:
            model = {name: value for name, value in model_all.items() if name in names}
            baselines = {key: baseline_channel(names, key) for key in D.BASELINES}
            report = _side_report(
                model, baselines, units, resamples=args.bootstrap, seed=args.seed
            )
            report["label"] = label
            return report

        series_units = {name: by_name[name].unit for name in by_name}
        topology_units = {
            name: (
                f"topology:{by_name[name].cluster}"
                if by_name[name].kind == "design"
                else by_name[name].unit
            )
            for name in by_name
        }
        designs = side(zero_hit, series_units, "a: certified zero-hit designs, series units")
        control = side(naturals, series_units, "b: natural-domain control, WT_cluster units")
        arms[arm] = {
            "identification": model_payload["identification"],
            "scoring_stratum": model_payload["settings"].get("scoring_stratum"),
            "designs": designs,
            "control": control,
            "verdict": D.arm_verdict(designs["gates"], control["gates"]),
            "preregistered_companions": {
                "designs_in_entirely_zero_hit_series": side(
                    clean_series, series_units, "a-strict: entirely zero-hit series only"
                ),
                "designs_at_topology_units": side(
                    zero_hit, topology_units, "a-topology: WT_cluster topology units"
                ),
            },
        }
        verdict = arms[arm]["verdict"]
        print(
            f"[analyse] {arm}: {verdict['verdict']} "
            f"(designs {designs['model']['point']:+.4f}, "
            f"control {control['model']['point']:+.4f})"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "analyse",
        "created_utc": _timestamp(),
        "cohort_sha256": digest,
        "cohort": referent.provenance,
        "kmer_background": baseline_payload["kmer_background"],
        "settings": {
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "baselines": list(D.BASELINES),
            "scoring_strata_declared": list(SCORING_STRATA),
        },
        "excluded_arms": dict(D.EXCLUDED_ARMS),
        "admitted_protein_doors": {
            label: list(members) for label, members in ADMITTED_PROTEIN_DOORS
        },
        "scoring_strata": {
            arm: record["scoring_stratum"] for arm, record in arms.items()
        },
        "arms": arms,
        "post_hoc": _post_hoc(
            referent,
            baseline_payload,
            models,
            proteingym_dir=args.proteingym_dir,
            resamples=args.bootstrap,
            seed=args.seed,
        ),
    }
    write_json(args.out / "designed_referent.json", payload)
    return payload


# -------------------------------------------------------------- stage: interaction


class _Rows:
    """Per-wild-type covariates and contrasts, assembled once from the artefacts.

    Everything this stage needs already exists: the per-wild-type Spearman values
    from EXP-R2-192's pinned scoring pass and its baseline pass. No model is
    loaded and no GPU is touched.
    """

    def __init__(self, referent: D.Referent, baselines: dict[str, Any]) -> None:
        from src.capability.context.profiles import AA20, KYTE_DOOLITTLE

        self.aa20 = list(AA20)
        kd = np.array([KYTE_DOOLITTLE[residue] for residue in AA20], dtype=np.float64)
        self.designs = referent.side("design")
        self.naturals = referent.side("natural")
        self.by_name = {wt.name: wt for wt in referent.wildtypes}
        self.unit = {wt.name: wt.unit for wt in referent.wildtypes}
        self.length = {wt.name: len(wt.sequence) for wt in referent.wildtypes}
        self.family = {wt.name: D.design_family(wt.series) for wt in self.designs}
        self.n_variants = {wt.name: len(wt.mutants) for wt in referent.wildtypes}
        self.phenotype_sd = {
            wt.name: float(np.std(wt.phenotype)) for wt in referent.wildtypes
        }
        self.phenotype_range = {
            wt.name: float(np.ptp(wt.phenotype)) for wt in referent.wildtypes
        }
        self.composition = {
            wt.name: self._composition(wt.sequence) for wt in referent.wildtypes
        }
        self.mean_hydropathy = {
            name: float(vector @ kd) for name, vector in self.composition.items()
        }
        self.baseline = {
            name: entry["spearman"] for name, entry in baselines["wildtypes"].items()
        }

    def _composition(self, sequence: str) -> np.ndarray:
        counts = np.array([sequence.count(residue) for residue in self.aa20], dtype=np.float64)
        return counts / counts.sum()

    def contrast(self, model: dict[str, float], channel: str, names: list[str]) -> tuple[list[float], list[str], list[str]]:
        values: list[float] = []
        units: list[str] = []
        kept: list[str] = []
        for name in names:
            base = self.baseline[name][channel]
            if name not in model or base is None:
                continue
            values.append(model[name] - float(base))
            units.append(self.unit[name])
            kept.append(name)
        return values, units, kept

    def unit_mean(self, model: dict[str, float], names: list[str]) -> float:
        return D.unit_mean_average(
            [model[name] for name in names], [self.unit[name] for name in names]
        )

    def unit_mean_channel(self, channel: str, names: list[str]) -> float:
        return D.unit_mean_average(
            [float(self.baseline[name][channel]) for name in names],
            [self.unit[name] for name in names],
        )

    def natural_names_in_band(self, low: int, high: int) -> list[str]:
        return sorted(
            wt.name for wt in self.naturals if low <= self.length[wt.name] <= high
        )

    def design_names(self, family: str | None) -> list[str]:
        return sorted(
            wt.name
            for wt in self.designs
            if family is None or self.family[wt.name] == family
        )


def _band(rows: _Rows, names: list[str], pad: int) -> tuple[int, int]:
    lengths = [rows.length[name] for name in names]
    return min(lengths) - pad, max(lengths) + pad


def _axis(
    rows: _Rows,
    model: dict[str, float],
    design_names: list[str],
    natural_names: list[str],
    *,
    channel: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    left, left_units, left_names = rows.contrast(model, channel, design_names)
    right, right_units, right_names = rows.contrast(model, channel, natural_names)
    if not left or not right:
        return {"degenerate": True, "degenerate_reason": "a side carries no readable wild type", "interval": None, "point": None, "outcome": "unresolved"}
    record = D.interaction_bootstrap(
        left, left_units, right, right_units, resamples=resamples, seed=seed
    )
    record["outcome"] = D.interaction_outcome(record)
    # The four halves the interaction is a difference of differences of. Carried
    # because a sign reversal can be the model falling, the channel rising, or
    # both, and a reader who is shown only the contrast cannot tell which.
    record["halves"] = {
        "model_designs": rows.unit_mean(model, left_names),
        "model_naturals": rows.unit_mean(model, right_names),
        "channel_designs": rows.unit_mean_channel(channel, left_names),
        "channel_naturals": rows.unit_mean_channel(channel, right_names),
        "median_length_designs": float(np.median([rows.length[n] for n in left_names])),
        "median_length_naturals": float(np.median([rows.length[n] for n in right_names])),
    }
    return record


def _subset_axis(
    rows: _Rows,
    model: dict[str, float],
    family: str | None,
    *,
    channel: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    design_names = rows.design_names(family)
    pads: dict[str, Any] = {}
    for pad in D.LENGTH_PADS:
        low, high = _band(rows, design_names, pad)
        natural_names = rows.natural_names_in_band(low, high)
        record = _axis(
            rows,
            model,
            design_names,
            natural_names,
            channel=channel,
            seed=seed,
            resamples=resamples,
        )
        record["band"] = [low, high]
        record["pad"] = pad
        pads[str(pad)] = record
    live = [pad for pad in D.LENGTH_PADS if not pads[str(pad)]["degenerate"]]
    primary = live[0] if live else None
    signs = {
        np.sign(pads[str(pad)]["point"]) for pad in live if pads[str(pad)]["point"] is not None
    }
    return {
        "n_design_series": len({rows.by_name[name].series for name in design_names}),
        "design_length_span": list(_band(rows, design_names, 0)),
        "pads": pads,
        "primary_pad": primary,
        "outcome": pads[str(primary)]["outcome"] if primary is not None else "unresolved",
        "sign_invariant_across_pads": bool(len(signs) <= 1),
    }


def _placebo(rows: _Rows, model: dict[str, float], *, channel: str, seed: int, resamples: int, splits: int) -> dict[str, Any]:
    """Two halves of the length-matched natural side, which differ in nothing.

    The attainability check for this statistic: a difference between two sides
    that are the same thing must return zero. At 95% about one split in twenty
    should exclude zero by chance, and materially more than that would mean the
    interval is not what it says.
    """

    low, high = _band(rows, rows.design_names(None), 0)
    names = rows.natural_names_in_band(low, high)
    clusters = sorted({rows.unit[name] for name in names})
    outcomes: list[dict[str, Any]] = []
    for index in range(splits):
        rng = np.random.default_rng(seed + index)
        order = rng.permutation(len(clusters))
        first = {clusters[position] for position in order[: len(clusters) // 2]}
        left_names = [name for name in names if rows.unit[name] in first]
        right_names = [name for name in names if rows.unit[name] not in first]
        left, left_units, _ = rows.contrast(model, channel, left_names)
        right, right_units, _ = rows.contrast(model, channel, right_names)
        record = D.interaction_bootstrap(
            left, left_units, right, right_units, resamples=resamples, seed=seed + index
        )
        outcomes.append(
            {
                "split_seed": seed + index,
                "point": record["point"],
                "interval": record["interval"],
                "excludes_zero": record["excludes_zero"],
            }
        )
    excluding = sum(1 for entry in outcomes if entry["excludes_zero"])
    return {
        "band": [low, high],
        "n_clusters": len(clusters),
        "splits": splits,
        "n_excluding_zero": excluding,
        "calibrated": bool(excluding <= 3),
        "note": (
            "more than three of twenty voids every axis in this entry, as "
            "pre-registered"
        ),
        "splits_detail": outcomes,
    }


def _composition_control(
    rows: _Rows,
    models: dict[str, dict[str, float]],
    *,
    channel: str,
    length_band: tuple[int, int],
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    design_names = rows.design_names(None)
    natural_names = sorted(wt.name for wt in rows.naturals)
    names = design_names + natural_names
    labels = np.array([1] * len(design_names) + [0] * len(natural_names))
    composition = np.array([rows.composition[name] for name in names])
    lengths = np.array([rows.length[name] for name in names], dtype=np.float64)
    features = np.hstack([composition, (lengths / 100.0)[:, None]])

    scores = np.zeros(len(names), dtype=np.float64)
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    for train, test in folds.split(features, labels):
        fitted = LogisticRegression(max_iter=5000, C=1.0).fit(features[train], labels[train])
        scores[test] = fitted.predict_proba(features[test])[:, 1]
    propensity = dict(zip(names, scores.tolist()))
    design_scores = scores[labels == 1]
    natural_scores = scores[labels == 0]
    support = (
        float(max(design_scores.min(), natural_scores.min())),
        float(min(design_scores.max(), natural_scores.max())),
    )

    covariate_names = ["length", "mean_hydropathy"] + [f"freq_{a}" for a in rows.aa20]

    def covariates(subset: list[str]) -> np.ndarray:
        return np.array(
            [
                [rows.length[name], rows.mean_hydropathy[name], *rows.composition[name]]
                for name in subset
            ],
            dtype=np.float64,
        )

    windows = {
        "common_support": support,
        "0.2_0.8": (0.2, 0.8),
        "0.3_0.7": (0.3, 0.7),
    }
    report: dict[str, Any] = {
        "propensity": {
            "features": "20 residue frequencies + length/100",
            "estimator": "5-fold stratified out-of-fold logistic regression, C=1, seed 0",
            "common_support": list(support),
        },
        "balance_before": D.standardised_mean_differences(
            covariates(design_names), covariates(natural_names), covariate_names
        ),
        "windows": {},
    }
    for label, (low, high) in windows.items():
        kept_designs = [name for name in design_names if low <= propensity[name] <= high]
        kept_naturals = [name for name in natural_names if low <= propensity[name] <= high]
        entry: dict[str, Any] = {
            "window": [float(low), float(high)],
            "n_designs": len(kept_designs),
            "n_design_series": len({rows.by_name[name].series for name in kept_designs}),
            "n_naturals": len(kept_naturals),
            "n_natural_clusters": len({rows.unit[name] for name in kept_naturals}),
            "balance_after": D.standardised_mean_differences(
                covariates(kept_designs), covariates(kept_naturals), covariate_names
            )
            if kept_designs and kept_naturals
            else None,
            "arms": {},
        }
        for arm, model in models.items():
            entry["arms"][arm] = _axis(
                rows,
                model,
                kept_designs,
                kept_naturals,
                channel=channel,
                seed=seed,
                resamples=resamples,
            )
        report["windows"][label] = entry

    # The propensity restriction is expected to leave the two sides unbalanced,
    # because designedness in this cohort is nearly a function of glutamate
    # content. What a restriction cannot do, an explicit match on the separating
    # covariate can -- on a smaller subcohort. The windows below are chosen to
    # minimise glutamate imbalance and are chosen on COVARIATES ALONE: no
    # interaction, correlation or model score was consulted in selecting them,
    # which is what keeps a matched subcohort a control rather than a search.
    band_naturals = set(rows.natural_names_in_band(*length_band))
    glutamate = {
        name: rows.composition[name][rows.aa20.index("E")]
        for name in design_names + natural_names
    }
    matched: dict[str, Any] = {
        "preregistered": False,
        "note": (
            "an explicit match on the single covariate that separates the two "
            "sides, inside the pooled length band. Windows selected on covariate "
            "balance only, with no outcome consulted; added after the reading and "
            "unable to change the verdict."
        ),
        "covariate": "glutamate frequency",
        "windows": {},
    }
    cells = (
        (0.110, 0.130, length_band),
        (0.110, 0.140, length_band),
        (0.105, 0.145, length_band),
        # The one cell in which glutamate AND length are both balanced at the
        # declared 0.25 threshold. It costs units -- ten natural clusters against
        # the pooled seventy-nine -- and is the price of matching two covariates
        # that a de novo design campaign fixed jointly.
        (0.080, 0.180, (40, 47)),
    )
    for low, high, band in cells:
        inside = set(rows.natural_names_in_band(*band)) & band_naturals
        kept_designs = [
            name
            for name in design_names
            if low <= glutamate[name] <= high and band[0] <= rows.length[name] <= band[1]
        ]
        kept_naturals = [
            name
            for name in natural_names
            if name in inside and low <= glutamate[name] <= high
        ]
        entry = {
            "window": [low, high],
            "length_band": list(band),
            "n_designs": len(kept_designs),
            "n_design_series": len({rows.by_name[name].series for name in kept_designs}),
            "n_naturals": len(kept_naturals),
            "n_natural_clusters": len({rows.unit[name] for name in kept_naturals}),
            "balance_after": D.standardised_mean_differences(
                covariates(kept_designs), covariates(kept_naturals), covariate_names
            ),
            "arms": {
                arm: _axis(
                    rows,
                    model,
                    kept_designs,
                    kept_naturals,
                    channel=channel,
                    seed=seed,
                    resamples=resamples,
                )
                for arm, model in models.items()
            },
        }
        matched["windows"][f"E{low}_{high}_L{band[0]}_{band[1]}"] = entry
    report["glutamate_matched"] = matched
    return report


def _adjusted(
    rows: _Rows,
    models: dict[str, dict[str, float]],
    *,
    channel: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    design_names = rows.design_names(None)
    natural_names = sorted(wt.name for wt in rows.naturals)
    names = design_names + natural_names
    composition = np.array([rows.composition[name] for name in names])
    centred = composition - composition.mean(axis=0)
    left, singular, _ = np.linalg.svd(centred, full_matrices=False)
    components = left[:, :5] * singular[:5]
    column = {
        "length": np.array([rows.length[name] for name in names], dtype=np.float64),
        "mean_hydropathy": np.array(
            [rows.mean_hydropathy[name] for name in names], dtype=np.float64
        ),
        "log_n_variants": np.log(
            np.array([rows.n_variants[name] for name in names], dtype=np.float64)
        ),
        "phenotype_sd": np.array(
            [rows.phenotype_sd[name] for name in names], dtype=np.float64
        ),
    }
    sets = {
        "S1_length": ["length"],
        "S2_length_hydropathy": ["length", "mean_hydropathy"],
        "S3_length_hydropathy_composition_pcs": ["length", "mean_hydropathy"],
        "S4_length_variants_range": ["length", "log_n_variants", "phenotype_sd"],
    }
    report: dict[str, Any] = {}
    for arm, model in models.items():
        readable = [
            name
            for name in names
            if name in model and rows.baseline[name][channel] is not None
        ]
        mask = np.array([name in set(readable) for name in names])
        values = [model[name] - float(rows.baseline[name][channel]) for name in readable]
        units = [rows.unit[name] for name in readable]
        designed = [name in set(design_names) for name in readable]
        report[arm] = {}
        for label, keys in sets.items():
            block = [column[key][mask] for key in keys]
            if label.endswith("composition_pcs"):
                block.append(components[mask])
            matrix = np.column_stack([np.asarray(part).reshape(len(readable), -1) for part in block])
            record = D.adjusted_interaction_bootstrap(
                values, units, designed, matrix, resamples=resamples, seed=seed
            )
            record["covariates"] = keys + (["composition_pc1..5"] if label.endswith("composition_pcs") else [])
            record["outcome"] = D.interaction_outcome(record) if record["point"] is not None else "unresolved"
            report[arm][label] = record
    return report


def stage_interaction(args: argparse.Namespace) -> dict[str, Any]:
    cohort_path = args.cohort or args.out / "cohort.json"
    referent = D.load_referent(cohort_path)
    digest = sha256_file(cohort_path)
    baseline_payload = _read(args.out / "baselines.json")
    if baseline_payload["cohort_sha256"] != digest:
        raise RuntimeError("the baselines were computed on a different cohort")

    rows = _Rows(referent, baseline_payload)
    models: dict[str, dict[str, float]] = {}
    identification: dict[str, Any] = {}
    for arm in args.arms:
        path = args.out / f"model_{arm}.json"
        if not path.is_file():
            print(f"[interaction] no model scores for {arm}; skipping")
            continue
        payload = _read(path)
        if payload["cohort_sha256"] != digest:
            raise RuntimeError(f"{arm} was scored on a different cohort")
        models[arm] = {
            name: float(entry["spearman"])
            for name, entry in payload["wildtypes"].items()
            if entry["spearman"] is not None
        }
        identification[arm] = dict(payload["identification"])
    if D.ORIGIN_ARM not in models:
        raise RuntimeError(f"the origin arm {D.ORIGIN_ARM} carries no scores")

    channel = D.INTERACTION_CHANNEL
    axes: dict[str, Any] = {}
    for arm, model in models.items():
        subsets = {"all": _subset_axis(rows, model, None, channel=channel, seed=args.seed, resamples=args.bootstrap)}
        for family in D.DESIGN_FAMILIES:
            subsets[family] = _subset_axis(
                rows, model, family, channel=channel, seed=args.seed, resamples=args.bootstrap
            )
        specificity = {}
        for other in D.BASELINES:
            record = _subset_axis(
                rows, model, None, channel=other, seed=args.seed, resamples=args.bootstrap
            )
            specificity[other] = (
                record["pads"][str(record["primary_pad"])]
                if record["primary_pad"] is not None
                else record
            )
        axes[arm] = {"subsets": subsets, "specificity": specificity}
        for label, entry in subsets.items():
            primary = entry["pads"].get(str(entry["primary_pad"]))
            point = "refused" if primary is None else f"{primary['point']:+.4f}"
            print(f"[interaction] {arm:16s} {label:14s} {entry['outcome']:11s} {point}")

    low, high = _band(rows, rows.design_names(None), 0)
    in_band = rows.natural_names_in_band(low, high)
    out_band = sorted(
        wt.name for wt in rows.naturals if not low <= rows.length[wt.name] <= high
    )
    design_variants = [rows.n_variants[name] for name in rows.design_names(None)]
    design_sd = [rows.phenotype_sd[name] for name in rows.design_names(None)]
    restricted = [
        name
        for name in in_band
        if min(design_variants) <= rows.n_variants[name] <= max(design_variants)
        and min(design_sd) <= rows.phenotype_sd[name] <= max(design_sd)
    ]

    controls = {
        "placebo_natural_half_splits": _placebo(
            rows,
            models[D.ORIGIN_ARM],
            channel=channel,
            seed=args.seed,
            resamples=args.bootstrap,
            splits=args.placebo_splits,
        ),
        "length_within_naturals": {
            "note": (
                "naturals inside the pooled design band against naturals outside "
                "it; how far the contrast moves with length when nothing is "
                "designed. Descriptive, not a gate."
            ),
            "preregistered": True,
            "band": [low, high],
            "arms": {
                arm: _axis(
                    rows, model, in_band, out_band, channel=channel, seed=args.seed, resamples=args.bootstrap
                )
                for arm, model in models.items()
            },
        },
        "length_within_band_split": {
            "note": (
                "short against long naturals INSIDE the pooled band. The "
                "pre-registered length placebo compares a 56-to-67 median shift; "
                "the design-versus-natural gap is 43 to 56, so this split is the "
                "closer length analogue. Added after the reading and labelled, "
                "and it cannot change the verdict."
            ),
            "preregistered": False,
            "split_at": args.length_split,
            "arms": {
                arm: _axis(
                    rows,
                    model,
                    [name for name in in_band if rows.length[name] < args.length_split],
                    [name for name in in_band if rows.length[name] >= args.length_split],
                    channel=channel,
                    seed=args.seed,
                    resamples=args.bootstrap,
                )
                for arm, model in models.items()
            },
        },
        "composition": _composition_control(
            rows,
            models,
            channel=channel,
            length_band=(low, high),
            seed=args.seed,
            resamples=args.bootstrap,
        ),
        "adjusted": _adjusted(
            rows, models, channel=channel, seed=args.seed, resamples=args.bootstrap
        ),
        "variant_and_range_matched": {
            "note": (
                "naturals inside the design band whose variant count and phenotype "
                "spread also fall inside the designs' own ranges"
            ),
            "n_naturals": len(restricted),
            "n_natural_clusters": len({rows.unit[name] for name in restricted}),
            "arms": {
                arm: _axis(
                    rows,
                    model,
                    rows.design_names(None),
                    restricted,
                    channel=channel,
                    seed=args.seed,
                    resamples=args.bootstrap,
                )
                for arm, model in models.items()
            },
        },
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "interaction",
        "created_utc": _timestamp(),
        "cohort_sha256": digest,
        "preregistration": "EXP-R2-193",
        "settings": {
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "channel": channel,
            "confirming_magnitude": D.INTERACTION_MAGNITUDE,
            "length_pads": list(D.LENGTH_PADS),
            "families": list(D.DESIGN_FAMILIES),
            "origin_arm": D.ORIGIN_ARM,
        },
        "identification": identification,
        "cohort_asymmetries": _asymmetries(rows, in_band),
        "axes": axes,
        "controls": controls,
        "verdict": _verdict(axes, controls),
    }
    write_json(args.out / "interaction.json", payload)
    print(json.dumps(payload["verdict"], indent=2, sort_keys=True))
    return payload


def _asymmetries(rows: _Rows, in_band: list[str]) -> dict[str, Any]:
    def summary(names: list[str]) -> dict[str, Any]:
        return {
            "n_wildtypes": len(names),
            "n_variants_total": int(sum(rows.n_variants[name] for name in names)),
            "n_variants_median": float(np.median([rows.n_variants[name] for name in names])),
            "length_median": float(np.median([rows.length[name] for name in names])),
            "phenotype_range_median": float(
                np.median([rows.phenotype_range[name] for name in names])
            ),
            "phenotype_sd_median": float(
                np.median([rows.phenotype_sd[name] for name in names])
            ),
            "mean_hydropathy_mean": float(
                np.mean([rows.mean_hydropathy[name] for name in names])
            ),
        }

    return {
        "designs": summary(rows.design_names(None)),
        "naturals_all": summary(sorted(wt.name for wt in rows.naturals)),
        "naturals_in_pooled_band": summary(in_band),
    }


def _verdict(axes: dict[str, Any], controls: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered overall rule, applied to the axes it was declared on."""

    origin = axes[D.ORIGIN_ARM]["subsets"]
    families = {family: origin[family]["outcome"] for family in D.DESIGN_FAMILIES}
    replicas = {
        arm: axes[arm]["subsets"]["all"]["outcome"]
        for arm in sorted(axes)
        if arm != D.ORIGIN_ARM
    }
    pooled = origin["all"]["outcome"]
    calibrated = bool(controls["placebo_natural_half_splits"]["calibrated"])
    confirming = [
        entry
        for entry in [origin["all"], *(origin[f] for f in D.DESIGN_FAMILIES)]
        if entry["outcome"] == "confirms"
    ] + [
        axes[arm]["subsets"]["all"]
        for arm in replicas
        if axes[arm]["subsets"]["all"]["outcome"] == "confirms"
    ]
    sign_invariant = all(entry["sign_invariant_across_pads"] for entry in confirming)

    family_confirms = sum(1 for value in families.values() if value == "confirms")
    family_refutes = sum(1 for value in families.values() if value == "refutes")
    replica_confirms = sum(1 for value in replicas.values() if value == "confirms")
    replica_refutes = sum(1 for value in replicas.values() if value == "refutes")

    confirmed = (
        calibrated
        and pooled == "confirms"
        and family_confirms >= 2
        and family_refutes == 0
        and replica_confirms >= 2
        and replica_refutes == 0
        and sign_invariant
    )
    refuted = calibrated and (
        pooled == "refutes" or family_refutes >= 2 or replica_refutes >= 2
    )
    verdict = "confirmed" if confirmed else ("refuted" if refuted else "underpowered")
    return {
        "verdict": verdict,
        "placebo_calibrated": calibrated,
        "pooled_origin_arm": pooled,
        "families": families,
        "replication_arms": replicas,
        "sign_invariant_on_confirming_axes": sign_invariant,
        "composition_control_bounds_interpretation": True,
    }


# --------------------------------------------------- stage: fragment_order
#
# EXP-R2-192's fragment channels are maximum-likelihood conditionals at k = 3 and
# k = 4, and the surviving claim is the margin over them. This stage rebuilds
# that channel as strongly as the corpus allows and re-runs the identical
# contrast, on both sides, at every order the corpus can still estimate.


def _fasta_runs(path: Path) -> list[str]:
    """Canonical-residue runs of every record, which is what the counter counted.

    A window containing ``X``, ``B``, ``Z``, ``U`` or ``O`` was never counted, so
    a held-out record has to be evaluated as its canonical runs rather than as one
    string with the non-canonical residues deleted: deleting them would join two
    fragments that are not adjacent in any protein, which is the same defect at
    residue scale that the record-boundary rule exists to prevent.
    """

    canonical = set(D.ALPHABET)
    runs: list[str] = []

    def emit(sequence: str) -> None:
        current: list[str] = []
        for residue in sequence:
            if residue in canonical:
                current.append(residue)
            elif current:
                runs.append("".join(current))
                current = []
        if current:
            runs.append("".join(current))

    parts: list[str] = []
    with path.open("r", encoding="ascii", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                if parts:
                    emit("".join(parts))
                    parts = []
            else:
                parts.append(line.strip())
    if parts:
        emit("".join(parts))
    return runs


def _training_counts(
    corpus: Any, holdouts: dict[str, Any], ks: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """The corpus counts with the held-out records removed, exactly.

    Subtraction rather than a second pass: a k-window never spans a record, so
    counts are additive over records and the difference *is* the count of the
    complement. Checked for negativity, because a subtraction that goes negative
    would mean the two passes did not see the same records and would otherwise
    surface as a silently distorted conditional.
    """

    counts: dict[int, np.ndarray] = {}
    for k in ks:
        vector = corpus.counts[k].copy()
        for background in holdouts.values():
            vector -= background.counts[k]
        if (vector < 0).any():
            raise RuntimeError(
                f"k = {k}: the held-out counts are not a subset of the corpus counts"
            )
        counts[k] = vector
    return counts


def _held_out_curves(
    counts: dict[int, np.ndarray],
    samples: dict[str, list[str]],
    *,
    max_order: int,
) -> dict[str, Any]:
    """Held-out cross-entropy on natural sequence, per scheme, per draw, per order.

    The estimator is held out and not plug-in: the counts here are the corpus
    minus these very records. That is the whole point -- a plug-in perplexity
    falls monotonically with k by construction and would license any order at all.
    """

    curves: dict[str, Any] = {}
    for scheme in D.FRAGMENT_SMOOTHING:
        model = D.InterpolatedFragmentModel(counts, max_order, scheme)
        curves[scheme] = {
            draw: {
                str(order): model.cross_entropy(sequences, order)
                for order in range(1, max_order + 1)
            }
            for draw, sequences in sorted(samples.items())
        }
        del model
    return curves


def _supported_orders(curves: dict[str, Any], max_order: int) -> dict[str, Any]:
    """The order past which the corpus stops supporting the estimate.

    Declared before any contrast is computed: order ``k`` is supported when
    held-out cross-entropy at ``k`` is strictly below its value at ``k - 1``, on
    **both** held-out draws and under **both** smoothing schemes. The highest
    supported order is the largest ``k`` for which that holds at every step up to
    ``k``. This is threshold-free -- it reads a turning point, not a cut -- and it
    is the only thing that decides admissibility; the sparsity diagnostics are
    reported beside every result and gate nothing.
    """

    supported: dict[int, bool] = {1: True}
    for order in range(2, max_order + 1):
        supported[order] = all(
            curves[scheme][draw][str(order)]["cross_entropy_nats"]
            < curves[scheme][draw][str(order - 1)]["cross_entropy_nats"]
            for scheme in curves
            for draw in curves[scheme]
        )
    highest = 1
    for order in range(2, max_order + 1):
        if not supported[order]:
            break
        highest = order
    return {
        "supported_step": {str(k): bool(v) for k, v in supported.items()},
        "highest_supported_order": highest,
        "turned_at": None if highest == max_order else highest + 1,
        "rule": (
            "order k is supported when held-out cross-entropy falls from k-1 to k "
            "on both draws under both schemes; admissibility is decided by this "
            "turning point alone and the sparsity diagnostics gate nothing"
        ),
    }


def _channel_readings(
    model: Any,
    wildtypes: Sequence[Any],
    sequences: dict[str, list[str]],
    order: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Per-wild-type Spearman of the channel, and the support it was read on."""

    spearman: dict[str, float] = {}
    totals = {
        "positions": 0,
        "positions_at_full_order": 0,
        "unseen_context_positions": 0,
        "unseen_kmer_positions": 0,
    }
    for wildtype in wildtypes:
        record = model.evaluate(sequences[wildtype.name], order)
        value = D.spearman(record["log_likelihood"], wildtype.phenotype)
        if value is not None:
            spearman[wildtype.name] = float(value)
        for key in totals:
            totals[key] += record[key]
    at_order = totals["positions_at_full_order"]
    support = {
        **totals,
        "unseen_context_fraction": (
            totals["unseen_context_positions"] / at_order if at_order else None
        ),
        "unseen_kmer_fraction": (
            totals["unseen_kmer_positions"] / at_order if at_order else None
        ),
        "n_wildtypes_read": len(spearman),
        "n_wildtypes_constant": len(wildtypes) - len(spearman),
    }
    return spearman, support


def _fragment_verdict(
    arm_rows: dict[str, Any], *, arm: str, admissible: list[int]
) -> dict[str, Any]:
    """The surviving claim, re-read against the strongest supported channel.

    Frozen before any contrast existed (EXP-R2-196). The half under test is
    "ProtGPT2's margin over corpus fragment statistics survives the exclusion of
    retrieval", so the verdict is taken on the identified arm, on the
    corpus-disjoint side, over the admissible orders only.
    """

    design = arm_rows[arm]["designs"]
    control = arm_rows[arm]["control"]
    beaten: list[str] = []
    undemonstrated: list[str] = []
    held: list[str] = []
    for order in admissible:
        for scheme in D.FRAGMENT_SMOOTHING:
            key = D.fragment_channel_name(order, scheme)
            interval = design[key]["interval"]
            label = f"{key}"
            if interval is None:
                undemonstrated.append(label)
            elif interval[1] < 0.0:
                beaten.append(label)
            elif interval[0] > 0.0:
                held.append(label)
            else:
                undemonstrated.append(label)
    if beaten:
        also_on_control = [
            name
            for name in beaten
            if control[name]["interval"] is not None and control[name]["interval"][1] < 0.0
        ]
        verdict = "overturned"
        reading = (
            "the channel beats the model on the natural control as well, so this is "
            "a statement about the channel"
            if len(also_on_control) == len(beaten)
            else "the channel beats the model on the designs and not on the natural "
            "control, so this is a statement about the referent"
        )
    elif undemonstrated:
        verdict = "weakened"
        reading = (
            "the margin over the strongest supported fragment channel is no longer "
            "demonstrable at every admissible order"
        )
    elif not admissible:
        verdict = "unresolved"
        reading = "the corpus supports no order beyond the two already used"
    else:
        verdict = "stands"
        reading = (
            "the margin holds at every admissible order under both declared "
            "smoothing schemes"
        )
    return {
        "arm": arm,
        "admissible_orders": admissible,
        "stronger_order_was_available": bool(max(admissible, default=0) > 4),
        "verdict": verdict,
        "reading": reading,
        "channels_the_model_beats": held,
        "channels_that_beat_the_model": beaten,
        "channels_without_a_demonstrated_margin": undemonstrated,
    }


def stage_fragment_order(args: argparse.Namespace) -> dict[str, Any]:
    cohort_path = args.cohort or args.out / "cohort.json"
    referent = D.load_referent(cohort_path)
    digest = sha256_file(cohort_path)
    baseline_payload = _read(args.out / "baselines.json")
    if baseline_payload["cohort_sha256"] != digest:
        raise RuntimeError("the baselines were computed on a different cohort")

    corpus = load_kmer_background(args.high_order_background_dir)
    background_record = D.require_corrected_background(corpus)
    print(f"[fragment_order] corpus {background_record['source']}")
    print(f"[fragment_order] totals {background_record['totals']}")
    new_manifest = json.loads(
        (args.high_order_background_dir / "manifest.json").read_text(encoding="utf-8")
    )
    pinned_manifest = json.loads(
        (D.KMER_BACKGROUND_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    identical = {
        str(k): new_manifest["sha256"][str(k)] == pinned_manifest["sha256"][str(k)]
        for k in (3, 4)
    }
    if not all(identical.values()):
        raise RuntimeError(
            "the higher-order background's k = 3/4 vectors are not byte-identical to "
            f"the pinned EXP-R2-192 background: {identical}"
        )
    print("[fragment_order] k=3/4 vectors byte-identical to the pinned background")

    max_order = min(int(args.max_order), max(corpus.ks))
    ks = tuple(range(1, max_order + 1))
    holdouts = {
        name: load_kmer_background(args.holdout_dir / name) for name in HOLDOUT_DRAWS
    }
    samples = {
        name: _fasta_runs(args.holdout_dir / f"holdout_{name}.fasta")
        for name in HOLDOUT_DRAWS
    }
    for name, runs in samples.items():
        print(
            f"[fragment_order] held-out draw {name}: {len(runs)} canonical runs, "
            f"{sum(len(run) for run in runs)} residues"
        )
    held_out_totals = {
        name: {"records": background.records, "residues": background.residues}
        for name, background in holdouts.items()
    }
    training = _training_counts(corpus, holdouts, ks)
    del holdouts
    curves = _held_out_curves(training, samples, max_order=max_order)
    del training
    admissibility = _supported_orders(curves, max_order)
    highest = int(admissibility["highest_supported_order"])
    admissible = [k for k in range(3, highest + 1)]
    print(
        f"[fragment_order] highest supported order {highest}; "
        f"admissible scoring orders {admissible}"
    )

    designs = referent.side("design")
    naturals = referent.side("natural")
    units = {wt.name: wt.unit for wt in referent.wildtypes}
    sides = {"designs": designs, "control": naturals}
    # Built once: WildType.sequences re-parses every mutation string, and ten
    # channels over half a million variants would re-parse it ten times.
    sequences = {wt.name: wt.sequences() for wt in referent.wildtypes}

    channels: dict[str, dict[str, dict[str, float]]] = {"designs": {}, "control": {}}
    support: dict[str, dict[str, Any]] = {"designs": {}, "control": {}}
    for scheme in D.FRAGMENT_SMOOTHING:
        model = D.InterpolatedFragmentModel(corpus.counts, max_order, scheme)
        for order in range(3, max_order + 1):
            key = D.fragment_channel_name(order, scheme)
            for label, wildtypes in sides.items():
                values, diagnostics = _channel_readings(
                    model, wildtypes, sequences, order
                )
                channels[label][key] = values
                support[label][key] = diagnostics
            print(
                f"[fragment_order] {key}: unseen k-mer fraction "
                f"designs {support['designs'][key]['unseen_kmer_fraction']}, "
                f"control {support['control'][key]['unseen_kmer_fraction']}",
                flush=True,
            )
        del model

    arms: dict[str, Any] = {}
    for arm in args.arms:
        path = args.out / f"model_{arm}.json"
        if not path.is_file():
            print(f"[fragment_order] no model scores for {arm}; skipping")
            continue
        payload = _read(path)
        if payload["cohort_sha256"] != digest:
            raise RuntimeError(f"{arm} was scored on a different cohort")
        model_all = {
            name: float(entry["spearman"])
            for name, entry in payload["wildtypes"].items()
            if entry["spearman"] is not None
        }
        entry: dict[str, Any] = {"identification": payload["identification"]}
        for label, wildtypes in sides.items():
            names = {wt.name for wt in wildtypes}
            model = {n: v for n, v in model_all.items() if n in names}
            contrasts: dict[str, Any] = {}
            for key, values in channels[label].items():
                contrasts[key] = D.channel_comparison(
                    model, values, units, resamples=args.bootstrap, seed=args.seed
                )
            for legacy in D.FRAGMENT_BASELINES:
                values = {
                    name: float(baseline_payload["wildtypes"][name]["spearman"][legacy])
                    for name in names
                    if baseline_payload["wildtypes"][name]["spearman"][legacy] is not None
                }
                contrasts[legacy] = D.channel_comparison(
                    model, values, units, resamples=args.bootstrap, seed=args.seed
                )
            entry[label] = contrasts
        arms[arm] = entry
        for order in admissible:
            for scheme in D.FRAGMENT_SMOOTHING:
                key = D.fragment_channel_name(order, scheme)
                point = entry["designs"][key]["point"]
                interval = entry["designs"][key]["interval"]
                print(f"[fragment_order] {arm} designs {key}: {point:+.4f} {interval}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "fragment_order",
        "created_utc": _timestamp(),
        "cohort_sha256": digest,
        "kmer_background": background_record,
        "kmer_background_matches_pinned_k3_k4": identical,
        "held_out": {
            "split": json.loads(
                (args.holdout_dir / "split.json").read_text(encoding="utf-8")
            ),
            "counted": held_out_totals,
            "canonical_runs": {
                name: {
                    "runs": len(runs),
                    "residues": sum(len(run) for run in runs),
                }
                for name, runs in samples.items()
            },
            "curves": curves,
        },
        "admissibility": admissibility,
        "settings": {
            "max_order": max_order,
            "schemes": list(D.FRAGMENT_SMOOTHING),
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "admissible_orders": admissible,
        },
        "corpus_coverage": {
            str(k): {
                "observed": corpus.coverage(k)[0],
                "possible": corpus.coverage(k)[1],
                "total_windows": int(corpus.counts[k].sum()),
            }
            for k in corpus.ks
        },
        "support": support,
        "arms": arms,
        "verdict": _fragment_verdict(arms, arm=D.ORIGIN_ARM, admissible=admissible)
        if D.ORIGIN_ARM in arms
        else None,
    }
    write_json(args.out / "fragment_order.json", payload)
    if payload["verdict"] is not None:
        print(f"[fragment_order] verdict {payload['verdict']['verdict']}")
    return payload


# ------------------------------------------------------------------------ main


def _require_joint_qualification_dir(args: argparse.Namespace) -> None:
    """An EXP-R2-226 rung needs its qualification, and nothing else may name one."""

    joint = sorted(set(args.arms) & set(L.RUNGS))
    if joint and args.joint_qualification_dir is None:
        raise ValueError(
            f"--arms names the EXP-R2-226 rungs {joint} and no "
            "--joint-qualification-dir. That campaign scores a rung only after "
            "every qualification clause holds, so an unqualified rung is refused "
            "here rather than discovered in an artefact"
        )
    if not joint and args.joint_qualification_dir is not None:
        raise ValueError(
            "--joint-qualification-dir belongs to the EXP-R2-226 rungs and no arm "
            "in --arms is one of them"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=None,
        help="the frozen cohort artefact; defaults to cohort.json under --out",
    )
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument(
        "--joint-qualification-dir",
        type=Path,
        default=None,
        help="directory holding adaptation_stage_qualification_<rung>.json for "
        "every EXP-R2-226 rung named in --arms. Required with one and refused "
        "without one: that campaign scores a rung only after every qualification "
        "clause holds, and an unqualified rung is not scored",
    )
    parser.add_argument("--megascale-dir", type=Path, default=None)
    parser.add_argument("--certificate-dir", type=Path, default=None)
    parser.add_argument(
        "--kmer-background-dir", type=Path, default=D.KMER_BACKGROUND_DIR
    )
    parser.add_argument("--retrieval-bound-dir", type=Path, default=None)
    parser.add_argument("--proteingym-dir", type=Path, default=PROTEINGYM_ROOT)
    parser.add_argument("--min-variants", type=int, default=D.MIN_VARIANTS)
    parser.add_argument("--bootstrap", type=int, default=D.BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--placebo-splits", type=int, default=20)
    parser.add_argument("--length-split", type=int, default=50)
    parser.add_argument(
        "--high-order-background-dir", type=Path, default=HIGH_ORDER_BACKGROUND_DIR
    )
    parser.add_argument("--holdout-dir", type=Path, default=HOLDOUT_ROOT)
    parser.add_argument(
        "--corpus-fasta", type=Path, default=REPO / "data/uniref50/uniref50.fasta"
    )
    parser.add_argument(
        "--background-ks", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7]
    )
    parser.add_argument(
        "--background-chunk-bytes", type=int, default=1024 * 1024 * 1024
    )
    parser.add_argument("--holdout-seed", type=int, default=20260813)
    parser.add_argument("--holdout-fraction", type=float, default=0.0005)
    parser.add_argument("--max-order", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=DTYPES)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    _require_joint_qualification_dir(args)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[paths] out={args.out}")
    print(f"[paths] megascale={args.megascale_dir or D.MEGASCALE_DIR}")
    print(f"[paths] certificate={args.certificate_dir or D.CERTIFICATE_DIR}")
    print(f"[paths] kmer_background={args.kmer_background_dir}")
    print(f"[paths] high_order_background={args.high_order_background_dir}")
    print(f"[paths] holdout={args.holdout_dir}")
    print(f"[paths] corpus_fasta={args.corpus_fasta}")
    print(f"[paths] cohort={args.cohort or args.out / 'cohort.json'}")
    print(f"[paths] arms={args.arms}")

    runners = {
        "cohort": stage_cohort,
        "baseline": stage_baseline,
        "score": stage_score,
        "analyse": stage_analyse,
        "interaction": stage_interaction,
        "background": stage_background,
        "fragment_order": stage_fragment_order,
    }
    for stage in args.stages:
        print(f"=== {stage}")
        runners[stage](args)


if __name__ == "__main__":
    main()
