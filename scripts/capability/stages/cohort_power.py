#!/usr/bin/env python3
"""Qualify an evaluation cohort before anything scientific is measured on it.

The cohort is built once, frozen by content hash, and then scored arm by arm for
the information the model commits relative to its own context-free baseline. An
arm below the threshold is reported as unmeasurable *on this cohort*: that is a
property of the evaluation set, not evidence about the model or about any
interpretability method, and downstream analyses must exclude the arm rather
than report a negative result from it.

Three artefacts, and the third is what makes this stage re-analysable. Beside
the frozen cohort and the report, ``--record-statistics`` (on by default) writes
``power_<cohort>_<digest>.records.npz``: the per-record clean-NLL sums, token
counts, symbol counts and target-token counts the report's figures are computed
from. Every published figure here is a function of those and of the reference
corpus's token counts, so a bootstrap, a re-weighting or a change of estimator
is a CPU job over that file rather than a second sweep of the panel. The report
carries the file's name and digest under ``sufficient_statistics``; a reader
that predates it finds ``null`` there and is otherwise unaffected.

A held-out run writes a fourth, ``reference_<cohort>_<reference digest>.json``:
the records of the block the context-free baseline was fitted on, frozen exactly
the way the scored cohort is. The sidecar deliberately carries no sequence text,
so without this file a re-analysis can neither group the reference by
near-duplicate content nor decide whether a reference record shares a k-mer with
a scored one -- the two things ``context_information_bootstrap.py`` needs and
reports as explicitly unavailable when they are missing. A plug-in run fits no
reference and writes no such file.

``--token-shuffle-control-seed`` turns the run into E3's negative control: each
record's scored target tokens are permuted within that record before the model
sees them, which destroys sequential context and leaves the unigram baseline
exactly unchanged. It is off unless it is asked for, and when it is asked for the
report is written under its own filename, declares itself under a different
``artifact`` name, and carries a ``negative_control`` block at the top and inside
every arm's record. Read
:class:`src.capability.core.scoring.TargetTokenShuffle` before quoting a number from
one: the control bounds what a context-free predictor achieves on shuffled
input, which is not the same as an arm whose true context information is zero.
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.capability.panel_contract import CAMPAIGN_PANEL, arm_can_run, stage_contract_record  # noqa: E402
from src.capability.core.io import write_json  # noqa: E402
from src.capability.core.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    PANEL,
    PROTEIN_SCALE_LADDER,
    PROGEN3_ARMS,
    REPO,
    STAGED_ARMS,
    STAGED_CANDIDATE_ARMS,
    STAGED_PROGEN3_ARMS,
    STAGED_SCALE_ARMS,
    STAGED_SECOND_STAGE_ARMS,
    Cohort,
    arm_spec,
    load_arm,
    load_arm_spec,
    protein_cohort,
    scoring_target_alphabet,
    target_shuffle_for,
    text_cohort,
)
from src.capability.core.budget import (  # noqa: E402
    DEFAULT_CONTEXT_LENGTHS,
    RecordStatistics,
    SCREENING_CONTEXT_INFORMATION_NATS,
    arm_power_with_records,
    markov_baselines,
    truncation_curve,
    write_power_records,
)
from src.capability.core.measurement_support import UNIGRAM_ESTIMATORS, held_out_cohort, subsample_cohort
from src.capability.core.measurement_support import scored_target_records
from src.capability.core.scoring import TOKEN_SHUFFLE_CONTROL  # noqa: E402

SCHEMA_VERSION = "r2_transfer_cohort_power_v1"
DEFAULT_OUT = REPO / "results/shared/cohort_power"


def default_arms(kind: str, with_ec: bool) -> list[str]:
    """Every campaign arm of this modality that this stage can qualify.

    It used to be three hard-coded names -- ``["gpt2-large"]`` for text, and
    ProtGPT2 plus ProGen2-medium for protein -- which meant a bare
    ``cohort_power.py --kind text`` qualified a *one-arm* text side of an
    eleven-arm panel and said nothing about the other six. That is L18's shape
    exactly: the run is well formed, every number in it is right, and the panel
    it covers is not the panel a reader assumes. The selection is now derived,
    and whatever it excludes is written into the artefact by
    ``stage_contract_record``.

    ``with_ec`` still gates the EC-conditioned arms: without the EC-labelled
    corpus their conditioning tag does not exist, and ``validate_arms`` refuses
    the pair rather than scoring them unconditioned.
    """

    names = [
        name
        for name in CAMPAIGN_PANEL
        if PANEL[name].modality == kind and arm_can_run("cohort_power", name).can_run
    ]
    if kind != "text" and not with_ec:
        names = [name for name in names if PANEL[name].input_format != "ec_conditioned"]
    if not names:
        raise ValueError(
            f"no campaign arm of modality {kind!r} can enter cohort_power"
            + ("" if with_ec else "; the EC-conditioned arms need --with-ec")
        )
    return names


def draw_records(args: argparse.Namespace, size: int, skip: int, name: str) -> Cohort:
    """``size`` records from the corpus, starting ``skip`` records in.

    ``--cohort-draw-seed`` reaches the corpus constructors, so ``size`` records
    are a window of a seeded permutation of the **whole** corpus rather than a
    prefix of the file. That distinction is the open qualification on EXP-R2-060:
    the campaign's "seeded" draw was seeded *within a head-of-file pool of 4000
    records*, so the 0.16-0.60 nat cohort-block sensitivity it reported bounds
    within-pool selection uncertainty and not corpus-wide selection uncertainty,
    and the audit records that the true figure is plausibly larger. It also makes
    ``--cohort-skip`` mean what it claims: at one seed, two skips index disjoint
    windows of the same permutation, so a skip-offset sensitivity is a
    sensitivity rather than two overlapping prefixes.

    ``--cohort-draw-seed 0`` still reproduces the historical file-order draw, and
    the mode reaches every artefact through ``Cohort.sampling``.
    """

    seed = args.cohort_draw_seed or None
    if args.kind == "text":
        return text_cohort(size, min_chars=args.min_chars, skip=skip, name=name, seed=seed)
    return protein_cohort(
        size,
        args.res_min,
        args.res_max,
        skip=skip,
        name=name,
        with_ec=args.with_ec,
        seed=seed,
    )


def build_cohort(args: argparse.Namespace) -> tuple[Cohort, dict[str, Any]]:
    """The scored cohort and the record of how it was drawn.

    Taking the first ``n_seq`` eligible records in corpus file order has
    manufactured an effect three times in this programme, most recently
    ProGen2-medium's 0.099-nat context information, which moved by +1.01 nats
    simply by reading past the first 48 Swiss-Prot records. File order is not a
    random order: Swiss-Prot's leading block is atypical in length, organism and
    annotation depth. So a seeded permutation is the default here, and the
    file-order draw remains reachable only by explicitly asking for it with
    ``--cohort-draw-seed 0``, which is a declared choice rather than a default
    that nobody notices. ``--cohort-skip`` moves the draw's origin so that the
    same measurement can be repeated on a disjoint block of the corpus, which is
    the sensitivity this hazard is detected by.

    **The seed reaches the corpus, not only the pool.** Until 2026-07-30 the pool
    itself was a file-order prefix of ``--cohort-pool-size`` and only the
    ``n_seq`` draw from it was seeded, so a "seeded" cohort was a seeded sample of
    a head-of-file block. The pool is now a window of a permutation of the whole
    corpus (see :func:`draw_records`), which is what makes the recorded
    cohort-block sensitivity a corpus-wide quantity.
    """

    if args.cohort_draw_seed == 0:
        cohort = draw_records(args, args.n_seq, args.cohort_skip, args.cohort_name)
        return cohort, {
            "sampling": "corpus_file_order",
            "cohort_skip": int(args.cohort_skip),
            "records_consumed": int(args.cohort_skip + args.n_seq),
        }
    if args.cohort_pool_size < args.n_seq:
        raise ValueError(
            f"--cohort-pool-size {args.cohort_pool_size} is smaller than --n-seq {args.n_seq}"
        )
    pool = draw_records(
        args, args.cohort_pool_size, args.cohort_skip, f"{args.cohort_name}_pool"
    )
    drawn = subsample_cohort(pool, args.n_seq, args.cohort_draw_seed)
    # subsample_cohort names the result after the pool and the draw; the output
    # filenames and every downstream reference key off --cohort-name, so the
    # caller's name is restored and the draw is recorded in the provenance block
    # instead.
    cohort = Cohort(
        name=args.cohort_name,
        kind=drawn.kind,
        records=drawn.records,
        min_symbols=drawn.min_symbols,
        max_symbols=drawn.max_symbols,
        metadata=drawn.metadata,
    )
    return cohort, {
        "sampling": "seeded_permutation_over_pool",
        "cohort_skip": int(args.cohort_skip),
        "pool_size": int(args.cohort_pool_size),
        "draw_seed": int(args.cohort_draw_seed),
        "pool_digest": pool.digest,
        "records_consumed": int(args.cohort_skip + args.cohort_pool_size),
    }


def write_reference_records(
    out: Path, cohort: Cohort, reference: Cohort | None
) -> Path | None:
    """Freeze the held-out reference block, or write nothing and say so by ``None``.

    The same kind of object as the frozen cohort beside it -- a record list under
    the digest of its own contents -- for the block that decides every
    context-information figure in the report. It exists because the
    sufficient-statistics sidecar stores order-free token counts by design, and
    two questions a re-analysis has to answer cannot be answered from counts:
    whether two reference records are near-duplicates of each other, and whether
    a reference record shares a k-mer with a scored one. Both need the sequence
    text, so a hash-per-record form would be the same dead end the sidecar
    already is; ``context_information_bootstrap.py`` reads this file through
    ``--reference-json`` and reports both as unavailable without it.

    Only a held-out run has a reference at all, so ``reference is None`` -- the
    plug-in estimator -- writes nothing rather than an empty artefact that a
    reader would have to interpret.

    The cost is the corpus's own: about 0.6 MB for a 4000-record Swiss-Prot
    reference and about 20 MB for a 4000-record OpenWebText one, whose documents
    average some five thousand characters. That is the smallest representation
    that supports the two questions.
    """

    if reference is None:
        return None
    destination = out / f"reference_{cohort.name}_{reference.digest[:12]}.json"
    write_json(
        destination,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact": "held_out_unigram_reference",
            # The quantity the sidecar names ``reference_digest`` and the report
            # names ``unigram_baseline.reference_digest``, under that name here
            # too: a consumer holding both can check it is reading the block this
            # run fitted its baseline on rather than another draw's.
            "reference_digest": reference.digest,
            "reference_name": reference.name,
            "reference_kind": reference.kind,
            # The scored cohort the block was held out against, which is what
            # makes the pair a leakage screen rather than two record lists.
            "cohort_digest": cohort.digest,
            "min_symbols": reference.min_symbols,
            "max_symbols": reference.max_symbols,
            "n_records": len(reference),
            "records": reference.records,
            # Carries the sampling record held_out_cohort travels forward,
            # including how many records the content deduplication removed.
            "metadata": reference.metadata,
        },
    )
    return destination


def validate_arms(names: list[str], args: argparse.Namespace) -> None:
    """Refuse an arm/cohort combination before the corpus is read.

    Every one of these three checks reads only ``--arms``, ``--kind`` and
    ``--with-ec``, yet all three used to run *after* ``build_cohort`` had scanned
    the corpus for a 4000-record pool, a 4000-record unigram reference and a
    2000-record Markov block. Nothing about the diagnosis needs the cohort: the
    modality comes from ``ArmSpec.modality``, and whether the cohort will carry EC
    labels is decided by ``--with-ec`` alone.

    The EC check is stated against ``ArmSpec.input_format`` rather than the
    literal name ``"zymctrl"``: a second EC-conditioned arm would otherwise be
    scored on an unconditioned prompt, which is separately measured at 1.73 nats
    of conditioning leak (EXP-R2-034).

    Staged checkpoints are not panel members, and reach this stage through three
    separate doors that never admit each other's arms. Without
    ``--allow-staged-scale-arms`` the EXP-R2-224 ProGen2 rungs
    (:data:`STAGED_SCALE_ARMS`) are unknown arms, exactly as before; without
    ``--allow-second-stage-arms`` the EXP-R2-225 checkpoints
    (:data:`STAGED_SECOND_STAGE_ARMS`) are too; without ``--allow-candidate-arms``
    the budget-only candidates (:data:`STAGED_CANDIDATE_ARMS`) are too. Each flag
    admits only its own tuple, and all three resolve through
    :func:`src.capability.core.arms.arm_spec` rather than through
    :func:`panel_contract.arm_can_run`.
    """

    allow_staged = bool(getattr(args, "allow_staged_scale_arms", False))
    allow_second_stage = bool(getattr(args, "allow_second_stage_arms", False))
    allow_candidate = bool(getattr(args, "allow_candidate_arms", False))
    allow_progen3 = bool(getattr(args, "allow_progen3_arms", False))
    unknown = []
    staged = []
    second_stage = []
    candidate = []
    progen3 = []
    panel_names = []
    for name in names:
        if name in PANEL:
            panel_names.append(name)
            continue
        if allow_staged and name in STAGED_SCALE_ARMS:
            staged.append(name)
            continue
        if allow_second_stage and name in STAGED_SECOND_STAGE_ARMS:
            second_stage.append(name)
            continue
        if allow_candidate and name in STAGED_CANDIDATE_ARMS:
            candidate.append(name)
            continue
        if allow_progen3 and name in STAGED_PROGEN3_ARMS:
            progen3.append(name)
            continue
        unknown.append(name)
    if unknown:
        raise ValueError(f"unknown arms {unknown}; panel is {sorted(PANEL)}")
    specs = {name: arm_spec(name) for name in names}
    # A staged opt-in checkpoint that cannot honour ``budget`` cannot produce this
    # stage's estimand at all, and the refusal is answerable from its declaration
    # alone. Raising here rather than inside the scoring loop is what keeps a
    # multi-billion-parameter load off the card for a run that was never going to
    # return a number.
    without_budget = [
        name
        for name in (*second_stage, *candidate, *progen3)
        if "budget" not in specs[name].capabilities and name != "rita-xl"
    ]
    if without_budget:
        raise ValueError(
            f"opt-in staged arms {without_budget} declare no 'budget' capability, "
            "so context information is not defined on them. Their declaration in "
            "src.capability.core.arms carries the reason; it is not discharged by naming "
            "them here"
        )
    mismatched = [name for name in names if specs[name].modality != args.kind]
    if mismatched:
        raise ValueError(
            f"arms {mismatched} do not match a {args.kind!r} cohort; "
            "a cross-modality cohort is not a measurement"
        )
    if progen3 and not bool(getattr(args, "skip_truncation", False)):
        raise ValueError(
            f"ProGen3 arms {progen3} need --skip-truncation: truncation_curve "
            "goes through tokenize_batch, which cannot carry sequence_ids"
        )
    rita = [name for name in names if name == "rita-xl"]
    if rita and not bool(getattr(args, "skip_truncation", False)):
        raise ValueError(
            "rita-xl needs --skip-truncation: native encoding does not go through "
            "tokenize_batch, and truncation_curve would"
        )
    if rita and str(getattr(args, "dtype", "")) != "float32":
        raise ValueError("rita-xl scoring is float32 only; pass --dtype float32")
    refused = {
        name: verdict.reason
        for name in panel_names
        if not (verdict := arm_can_run("cohort_power", name)).can_run
    }
    if refused:
        raise ValueError(f"cohort_power.py cannot qualify {sorted(refused)}: {refused}")
    conditioned = [name for name in names if specs[name].input_format == "ec_conditioned"]
    if conditioned and not args.with_ec:
        raise ValueError(
            f"arms {conditioned} are EC-conditioned and need an EC-labelled cohort; "
            "rebuild with --with-ec"
        )


def _arm_spec_record(name: str) -> dict[str, Any]:
    spec = arm_spec(name)
    record = {
        "path": str(spec.path),
        "modality": spec.modality,
        "n_layer": spec.n_layer,
        "d_model": spec.d_model,
        "tokenisation": spec.tokenisation,
        "input_format": spec.input_format,
        "source": spec.source,
    }
    if name in STAGED_ARMS or name in PROGEN3_ARMS:
        alphabet = scoring_target_alphabet(spec)
        record["not_panel_admission"] = True
        record["scoring_target_alphabet_size"] = alphabet["size"]
        record["scoring_target_alphabet_source"] = alphabet["source"]
    return record


def _cohort_power_stage_contract(names: list[str]) -> dict[str, Any]:
    """Panel contract for panel arms; a staged-only run must not impersonate one."""

    panel_names = [name for name in names if name in PANEL]
    staged = [
        name
        for name in names
        if name in STAGED_SCALE_ARMS
        or name in STAGED_SECOND_STAGE_ARMS
        or name in STAGED_CANDIDATE_ARMS
        or name in STAGED_PROGEN3_ARMS
    ]
    if not panel_names and staged:
        return {
            "stage": "cohort_power",
            "not_panel_admission": True,
            "measured": [],
            "measured_staged_arms": staged,
            "note": (
                "staged opt-in qualification; panel_contract was not asked to "
                "admit these arms and this is not an empty campaign-panel run"
            ),
        }
    return stage_contract_record("cohort_power", panel_names)


def _second_stage_record(names: list[str], allow_second_stage: bool) -> dict[str, Any] | None:
    """What an EXP-R2-225 second-stage run declares about the arms it measured.

    Separate from :func:`_staged_scale_record` because the two campaigns are
    separate: this one names no ladder, because its checkpoints are not one
    lineage, and it records each arm's own scoring-target alphabet rather than
    one shared 32.
    """

    second_stage = [name for name in names if name in STAGED_SECOND_STAGE_ARMS]
    if not second_stage:
        return None
    return {
        "not_panel_admission": True,
        "allow_second_stage_arms": bool(allow_second_stage),
        "scope": "exp_r2_225_second_stage_checkpoints",
        "allowed_second_stage_arms": list(STAGED_SECOND_STAGE_ARMS),
        "measured_second_stage_arms": second_stage,
        "scoring_target_alphabet": {
            name: scoring_target_alphabet(arm_spec(name)) for name in second_stage
        },
        "reason": (
            "these checkpoints remain outside PANEL and CAMPAIGN_PANEL; this "
            "artefact is an opt-in second-stage qualification, not panel "
            "admission and not a capability claim"
        ),
    }


def _candidate_record(names: list[str], allow_candidate: bool) -> dict[str, Any] | None:
    """What a budget-only candidate run declares about the arms it measured.

    Separate from the two older staged doors: this is not EXP-R2-224, not
    EXP-R2-225, not panel admission, and not an experiment ADMITTED digest.
    An implemented interface is not a passed 8-block identification interval.
    """

    candidate = [name for name in names if name in STAGED_CANDIDATE_ARMS]
    if not candidate:
        return None
    return {
        "not_panel_admission": True,
        "not_experiment_admitted": True,
        "allow_candidate_arms": bool(allow_candidate),
        "scope": "budget_loglikelihood_candidate_checkpoints",
        "allowed_candidate_arms": list(STAGED_CANDIDATE_ARMS),
        "measured_candidate_arms": candidate,
        "scoring_target_alphabet": {
            name: scoring_target_alphabet(arm_spec(name)) for name in candidate
        },
        "reason": (
            "these checkpoints remain outside PANEL and CAMPAIGN_PANEL; this "
            "artefact is an opt-in candidate budget qualification, not panel "
            "admission, not a capability beyond budget, and not an experiment "
            "ADMITTED digest"
        ),
    }


def _progen3_record(names: list[str], allow_progen3: bool) -> dict[str, Any] | None:
    """Opt-in ProGen3 N-to-C context information; not bidirectional DMS."""

    progen3 = [name for name in names if name in STAGED_PROGEN3_ARMS]
    if not progen3:
        return None
    return {
        "not_panel_admission": True,
        "allow_progen3_arms": bool(allow_progen3),
        "scope": "progen3_n_to_c_context_information",
        "allowed_progen3_arms": list(STAGED_PROGEN3_ARMS),
        "measured_progen3_arms": progen3,
        "scoring_directions": ["n_to_c"],
        "not_bidirectional": (
            "left-to-right next-token NLL matching stage 01; the bidirectional "
            "mean used by stage 20 is a different estimand"
        ),
        "scoring_target_alphabet": {
            name: scoring_target_alphabet(arm_spec(name)) for name in progen3
        },
        "reason": (
            "these checkpoints remain outside PANEL and CAMPAIGN_PANEL; packed "
            "megablocks weights are converted by load_progen3. This artefact is "
            "not panel admission"
        ),
    }


def _staged_scale_record(names: list[str], allow_staged: bool) -> dict[str, Any] | None:
    staged = [name for name in names if name in STAGED_SCALE_ARMS]
    if not staged:
        return None
    return {
        "not_panel_admission": True,
        "allow_staged_scale_arms": bool(allow_staged),
        "scope": "progen2_training_lineage_medium_to_xlarge",
        "ladder": list(PROTEIN_SCALE_LADDER),
        "allowed_staged_arms": list(STAGED_SCALE_ARMS),
        "measured_staged_arms": staged,
        "scoring_target_alphabet": {
            name: scoring_target_alphabet(arm_spec(name))
            for name in staged
        },
        "reason": (
            "these rungs remain outside PANEL and CAMPAIGN_PANEL; this artefact "
            "is an opt-in scale qualification, not panel admission"
        ),
    }


def artifact_names(cohort_name: str, digest: str, control_seed: int | None) -> tuple[str, str]:
    """The report's ``artifact`` name and its basename, given the run's kind.

    A control run must not be able to overwrite or be mistaken for a
    measurement, and the cohort digest cannot separate them: the shuffle happens
    in token space, after the cohort is frozen, so the two runs share a cohort
    byte for byte. The seed is in the basename because two control runs at
    different seeds are two different controls.
    """

    if control_seed is None:
        return "cohort_power_report", f"power_{cohort_name}_{digest[:12]}"
    return (
        "cohort_power_token_shuffle_control_report",
        f"power_{cohort_name}_{TOKEN_SHUFFLE_CONTROL}_seed{control_seed}_{digest[:12]}",
    )


def negative_control_record(control_seed: int | None) -> dict[str, Any] | None:
    """The run-level declaration a control artefact carries, or ``None``.

    ``None`` on a measurement run, so the key is present in every report and a
    reader never has to know whether the field predates the control.
    """

    if control_seed is None:
        return None
    return {
        "control": TOKEN_SHUFFLE_CONTROL,
        "seed": int(control_seed),
        "declared_by": "src.capability.core.scoring.TargetTokenShuffle",
        "this_is_not_a_measurement": (
            "every model-side figure in this report was computed on records whose "
            "scored target tokens were permuted within the record. The context "
            "information here is what the estimator returns for an arm that has no "
            "usable sequential context; it is not a measurement of the arm on this "
            "cohort and must never be quoted as one"
        ),
        "what_the_shuffle_enters": (
            "the model's forward pass. The unigram baseline, the scored-token count "
            "and the cohort digest are invariant under the permutation by "
            "construction, which is what makes the contrast with a measurement run "
            "readable. Two things are not. The per-symbol conversion moves by a few "
            "parts in a thousand on a byte-level BPE arm, because reordering tokens "
            "can split a multi-byte character and the decoder then emits replacement "
            "characters, so every *_bits_per_symbol field here carries that much "
            "noise; no nats-per-token figure does. And the residue Markov ladder is "
            "computed from the unshuffled cohort, so at order one and above it does "
            "not describe what this run scored"
        ),
    }


def validate_negative_control(args: argparse.Namespace) -> None:
    """Refuse the one combination a control run cannot produce honestly.

    The truncation curve tokenises the cohort itself rather than going through
    the scored pass, so the shuffle never reaches it. Under the control it would
    therefore be measured on the unshuffled rendering and would sit in the
    artefact as a context curve for a run whose context was destroyed -- the one
    figure in the report a reader is most likely to read as the control's own.
    """

    if args.token_shuffle_control_seed is None or args.skip_truncation:
        return
    raise ValueError(
        "--token-shuffle-control-seed and the truncation curve do not belong in one "
        "artefact: the curve is tokenised separately from the scored pass, so it "
        "would describe the unshuffled rendering. Pass --skip-truncation"
    )


def validate_truncation(args: argparse.Namespace) -> None:
    """Refuse a ``--max-len`` / ``--truncation-contexts`` pair before any model load.

    ``budget.truncation_curve`` requires ``max_len`` to exceed the longest
    requested context by at least two, and raises when it does not -- correctly,
    since a window that does not fit carries no truncation information. But it is
    called after ``load_arm`` and after the whole ``arm_power`` scoring pass, so
    a combination that was never going to work costs a checkpoint load and a full
    forward sweep first. Both values are command-line arguments.
    """

    if args.skip_truncation:
        return
    if not args.truncation_contexts:
        raise ValueError("--truncation-contexts is empty; pass --skip-truncation instead")
    longest = max(int(length) for length in args.truncation_contexts)
    if min(int(length) for length in args.truncation_contexts) < 1:
        raise ValueError("--truncation-contexts must all be at least one token")
    if args.max_len <= longest + 1:
        raise ValueError(
            f"--max-len {args.max_len} must exceed the longest --truncation-contexts "
            f"entry ({longest}) by at least two; raise --max-len, drop the long "
            "contexts, or pass --skip-truncation"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("protein", "text"), default="protein")
    parser.add_argument("--cohort-name", default=None)
    parser.add_argument("--n-seq", type=int, default=200)
    parser.add_argument("--res-min", type=int, default=64)
    parser.add_argument("--res-max", type=int, default=246)
    parser.add_argument("--min-chars", type=int, default=800, help="text cohort only")
    parser.add_argument("--with-ec", action="store_true", help="draw EC-labelled records")
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument(
        "--allow-staged-scale-arms",
        action="store_true",
        help="opt in to the staged ProGen2 large/xlarge rungs of "
        "PROTEIN_SCALE_LADDER. They stay outside the panel; the artefact "
        "declares not_panel_admission rather than asking panel_contract to "
        "admit them",
    )
    parser.add_argument(
        "--allow-second-stage-arms",
        action="store_true",
        help="opt in to the EXP-R2-225 second-stage checkpoints "
        "(STAGED_SECOND_STAGE_ARMS). A separate door from "
        "--allow-staged-scale-arms and not a widening of it: neither flag admits "
        "the other's arms, and neither is panel admission. rita-xl is scored "
        "through native encoding despite an empty capability set; other arms "
        "without 'budget' are still refused here rather than after they load",
    )
    parser.add_argument(
        "--allow-candidate-arms",
        action="store_true",
        help="opt in to the budget-only candidate checkpoints "
        "(STAGED_CANDIDATE_ARMS: qwen3-8b-base, protgpt3-1.3b). A third door, "
        "not --allow-second-stage-arms and not --allow-staged-scale-arms: each "
        "flag admits only its own tuple. Not panel admission and not an "
        "experiment ADMITTED digest. An arm whose declaration carries no "
        "'budget' capability is refused here rather than after it is loaded",
    )
    parser.add_argument(
        "--allow-progen3-arms",
        action="store_true",
        help="opt in to ProGen3-112M/3B (STAGED_PROGEN3_ARMS). Packed megablocks "
        "weights; load_progen3 converts them and self-checks. Not PANEL, not "
        "STAGED_ARMS, and not a widening of the other doors. Requires "
        "--skip-truncation. Scoring is N-to-C only, matching this stage's "
        "estimand, not the bidirectional DMS convention",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--threshold-nats",
        type=float,
        default=SCREENING_CONTEXT_INFORMATION_NATS,
        help="the identification floor a point estimate of I is screened against. "
        "It decides whether the arm read above no-context on this cohort and "
        "nothing else; whether that reading may be divided by is "
        "budget.ratio_denominator_admissibility, which is per-arm and is not a "
        "threshold this flag can set",
    )
    parser.add_argument("--markov-train", type=int, default=2000)
    parser.add_argument("--skip-markov", action="store_true")
    parser.add_argument("--skip-truncation", action="store_true")
    parser.add_argument(
        "--cohort-skip",
        type=int,
        default=0,
        help="records to pass over before the pool starts; use to repeat a "
        "measurement on a disjoint block and report a skip-offset sensitivity",
    )
    parser.add_argument(
        "--cohort-pool-size",
        type=int,
        default=4000,
        help="pool the seeded draw samples from; ignored with --cohort-draw-seed 0",
    )
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="seed for the permutation that draws --n-seq records from the pool; "
        "0 selects the corpus file order instead, which is a declared choice "
        "and not a default (see build_cohort)",
    )
    parser.add_argument(
        "--unigram-estimator",
        default="disjoint",
        choices=list(UNIGRAM_ESTIMATORS),
        help="context-free baseline: 'disjoint' fits it on a held-out block of "
        "the same corpus, 'plugin' on the scored cohort itself. The plug-in is "
        "biased down by up to 1.29 nats on a 50k-vocabulary arm and 0.01 on a "
        "residue-level one, so it distorts exactly the cross-arm comparison "
        "this stage exists to support",
    )
    parser.add_argument("--unigram-reference-size", type=int, default=4000)
    parser.add_argument(
        "--record-statistics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write the per-record sufficient statistics beside the report. They "
        "are what every uncertainty re-analysis needs -- per-record clean NLL "
        "sums, token counts and target-token counts -- and without them a "
        "bootstrap over this stage costs a second GPU sweep of the whole panel",
    )
    parser.add_argument(
        "--truncation-contexts",
        type=int,
        nargs="*",
        default=list(DEFAULT_CONTEXT_LENGTHS),
        help=(
            "visible-context ladder in tokens; it is arm-relative because "
            "tokenizer expansion differs, and a multi-residue-BPE arm on a short "
            "cohort needs a shorter ladder than a residue-level arm"
        ),
    )
    parser.add_argument("--truncation-queries", type=int, default=4)
    parser.add_argument("--truncation-batch", type=int, default=64)
    parser.add_argument("--truncation-min-windows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--token-shuffle-control-seed",
        type=int,
        default=None,
        help="run E3's negative control instead of a measurement: permute each "
        "record's scored target tokens within that record, under this seed, "
        "before the model sees them. Destroys sequential context and leaves the "
        "unigram baseline exactly unchanged, so context information collapses to "
        "whatever an arm with no usable context reads against that baseline -- "
        "gpt2 on 24 OpenWebText records falls from +4.64 to -0.29 nats/token, "
        "near the boundary the 0.30-nat floor and the sign criterion have never "
        "been exercised at, but on the negative side and not at zero. Off unless "
        "given; a run under it writes a differently named, self-identifying "
        "artefact and requires --skip-truncation",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.cohort_name is None:
        args.cohort_name = "swissprot" if args.kind == "protein" else "openwebtext"
    names = args.arms if args.arms else default_arms(args.kind, args.with_ec)

    # Everything that can be refused from the command line alone is refused here,
    # before the corpus is scanned and long before a checkpoint is loaded.
    validate_arms(names, args)
    validate_negative_control(args)
    validate_truncation(args)

    cohort, sampling = build_cohort(args)
    digest = cohort.digest

    # The held-out block starts after everything the scored draw consumed, and
    # is then deduplicated against the cohort by content: Swiss-Prot carries the
    # same sequence under several accessions, so a later file offset is not by
    # itself disjoint. There is no fallback to the plug-in -- asking for the
    # held-out estimator without a reference is a configuration error.
    reference: Cohort | None = None
    reference_overlap: dict[str, int] = {}
    if args.unigram_estimator == "disjoint":
        candidate = draw_records(
            args,
            args.unigram_reference_size,
            int(sampling["records_consumed"]),
            f"{args.cohort_name}_unigram_reference",
        )
        reference, reference_overlap = held_out_cohort(candidate, cohort)
    cohort_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "frozen_cohort",
        "cohort_digest": digest,
        "cohort_name": cohort.name,
        "cohort_kind": cohort.kind,
        "min_symbols": cohort.min_symbols,
        "max_symbols": cohort.max_symbols,
        "n_records": len(cohort),
        "records": cohort.records,
        "metadata": cohort.metadata,
    }
    write_json(args.out / f"cohort_{cohort.name}_{digest[:12]}.json", cohort_payload)
    reference_records_path = write_reference_records(args.out, cohort, reference)

    cohort_level: dict[str, Any] = {}
    if cohort.kind == "protein" and not args.skip_markov:
        # Offset past both the scored draw and the unigram reference block, so
        # that the residue ladder, the context-free baseline and the scored
        # cohort are three disjoint slices of the corpus rather than two.
        markov_skip = int(sampling["records_consumed"]) + (
            args.unigram_reference_size if reference is not None else 0
        )
        held_out = protein_cohort(
            args.markov_train,
            args.res_min,
            args.res_max,
            skip=markov_skip,
            name=f"{cohort.name}_markov_train",
            with_ec=args.with_ec,
            seed=args.cohort_draw_seed or None,
        )
        # A record-count offset is not disjointness. Swiss-Prot carries the same
        # sequence under several accessions and the EC-labelled corpus carries
        # it under several EC tags, so a later block in file order can still
        # repeat the cohort's content; markov_cross_entropy_bits refuses such a
        # pair outright, which is what caught this. Deduplicating by content is
        # the same remedy the unigram reference uses, and the number removed is
        # recorded rather than absorbed.
        held_out, markov_overlap = held_out_cohort(held_out, cohort)
        cohort_level["residue_markov_baselines"] = markov_baselines(
            held_out.records, cohort.records
        )
        cohort_level["residue_markov_train_digest"] = held_out.digest
        cohort_level["residue_markov_train_skip"] = markov_skip
        cohort_level["residue_markov_train_overlap_removed"] = markov_overlap

    control_seed = args.token_shuffle_control_seed
    arm_reports: dict[str, Any] = {}
    arm_records: dict[str, RecordStatistics] = {}
    for name in names:
        spec = arm_spec(name)
        if name in PANEL:
            arm = load_arm(name, device=args.device, dtype=args.dtype)
        elif name in PROGEN3_ARMS:
            from src.capability.models.progen3 import load_progen3_as_arm

            arm = load_progen3_as_arm(name, device=args.device, dtype=args.dtype)
        else:
            arm = load_arm_spec(spec, device=args.device, dtype=args.dtype)
        reference_counts = None
        reference_record = None
        reference_per_record = None
        if reference is not None:
            # Tokenizer only, over exactly the multiset arm_power scores, so the
            # held-out baseline costs no second forward pass.
            reference_counts, reference_per_record = scored_target_records(
                arm, reference.input_strings(arm), max_len=args.max_len
            )
            reference_record = {
                "cohort": reference.name,
                "digest": reference.digest,
                **reference_overlap,
            }
        # The reference is counted on the *unshuffled* arm, and the shuffled arm
        # is a second view of the same loaded checkpoint rather than a mutation of
        # it. The reference counts would in fact be identical either way -- the
        # permutation preserves each record's target multiset exactly, which is
        # the property the whole control rests on -- but a held-out corpus that
        # was described as shuffled would say something untrue about the baseline.
        shuffle = None if control_seed is None else target_shuffle_for(arm, seed=control_seed)
        scoring_arm = arm if shuffle is None else replace(arm, target_token_shuffle=shuffle)
        # The baseline is chosen inside arm_power, where the scored targets are,
        # so every field of the record -- the verdict, the per-symbol views, the
        # per-sequence interval -- is derived from the estimator the record
        # names. Recomputing a subset of them here is what left ZymCTRL's
        # Miller-Madow context information beside a held-out one under two names
        # that did not distinguish them.
        report, records = arm_power_with_records(
            scoring_arm,
            cohort,
            max_len=args.max_len,
            batch_size=args.batch_size,
            minimum_context_information_nats=args.threshold_nats,
            unigram_estimator=args.unigram_estimator,
            reference_token_counts=reference_counts,
            reference=reference_record,
        )
        if reference_per_record is not None:
            records = replace(records, reference_counts=reference_per_record)
        arm_records[name] = records
        if not args.skip_truncation:
            report["truncation_curve"] = truncation_curve(
                arm,
                cohort.input_strings(arm),
                max_len=args.max_len,
                context_lengths=args.truncation_contexts,
                queries_per_sequence=args.truncation_queries,
                batch_size=args.truncation_batch,
                seed=args.seed,
                min_windows=args.truncation_min_windows,
            )
        # Inside the arm's own record as well as at the top of the report: an arm
        # record is routinely lifted out of the artefact on its own, and one that
        # did not carry the control would read exactly like a measurement.
        if shuffle is not None:
            report["negative_control"] = shuffle.record()
        arm_reports[name] = report
        baseline_nats = report["unigram_entropy_used_for_verdict_nats"]
        print(
            f"{name:16s} unigram({args.unigram_estimator}) {baseline_nats:7.4f}  "
            f"clean_ce {report['clean_ce_nats']:7.4f}  "
            f"context_info {report['context_information_nats']:+8.4f} nats/token  "
            f"{report['power_verdict']} ({report['measurability']})"
            + ("" if shuffle is None else f"  [{TOKEN_SHUFFLE_CONTROL} NEGATIVE CONTROL]")
        )
        # Both names, because under the control ``scoring_arm`` is a second view
        # of the same checkpoint and would otherwise hold it alive across the next
        # ``load_arm``.
        del arm, scoring_arm
        gc.collect()
        torch.cuda.empty_cache()

    artifact, stem = artifact_names(cohort.name, digest, control_seed)
    destination = args.out / f"{stem}.json"
    # Written before the report, so that the report can carry the sidecar's
    # digest: a reader learns from the report whether the sidecar exists and
    # whether the one on disk is the one this run produced.
    sidecar_seeds = {
        "cohort_draw": int(args.cohort_draw_seed),
        "truncation_curve": int(args.seed),
    }
    if control_seed is not None:
        sidecar_seeds["token_shuffle_control"] = int(control_seed)
    sufficient_statistics = None
    if args.record_statistics:
        sufficient_statistics = write_power_records(
            destination.with_suffix(".records.npz"),
            arm_records,
            cohort_digest=digest,
            reference_digest=None if reference is None else reference.digest,
            smoothing=next(iter(arm_reports.values()))["unigram_baseline"]["smoothing"],
            seeds=sidecar_seeds,
            max_len=int(args.max_len),
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact": artifact,
        # ``None`` on a measurement run, a self-describing block on a control run.
        "negative_control": negative_control_record(control_seed),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_digest": digest,
        "cohort": {
            "name": cohort.name,
            "kind": cohort.kind,
            "n_records": len(cohort),
            "min_symbols": cohort.min_symbols,
            "max_symbols": cohort.max_symbols,
            "with_ec_labels": bool(cohort.metadata.get("ec_labels")),
        },
        "arm_specs": {name: _arm_spec_record(name) for name in names},
        "seeds": {
            "truncation_curve": int(args.seed),
            "cohort_draw": int(args.cohort_draw_seed),
            # Always present, and ``None`` on a measurement run, so this block
            # says which kind of run produced the artefact on its own.
            "token_shuffle_control": None if control_seed is None else int(control_seed),
        },
        "cohort_sampling": sampling,
        "stage_contract": _cohort_power_stage_contract(names),
        "unigram_baseline": {
            "estimator": args.unigram_estimator,
            "reference_size_requested": int(args.unigram_reference_size),
            "reference_sequences_used": None if reference is None else len(reference),
            "reference_digest": None if reference is None else reference.digest,
            "reference_overlap_removed": reference_overlap,
            # The scored cohort is a seeded draw from a file-order pool; the
            # held-out reference block is the next contiguous block of the corpus
            # in file order, with no permutation of its own. That matters because
            # this block *is* the context-free baseline every context-information
            # figure is measured against, and Appendix B rule 1 applies to it as
            # much as to the scored draw. Carried here rather than in a document:
            # arms.protein_cohort/text_cohort record their own sampling mode and
            # hazard, and this surfaces it beside the number it produced.
            "reference_sampling": (
                None if reference is None else reference.metadata.get("sampling")
            ),
        },
        "truncation_context_lengths": sorted({int(c) for c in args.truncation_contexts}),
        "thresholds": {
            "minimum_context_information_nats": float(args.threshold_nats),
            "minimum_context_information_provenance": (
                "src.capability.core.budget.SCREENING_CONTEXT_INFORMATION_NATS: the "
                "identification floor, calibrated at EXP-R2-218. It is not a "
                "licence to divide by the reading -- that is "
                "budget.ratio_denominator_admissibility, a per-arm bound on this "
                "arm's own standard error"
            ),
        },
        "settings": {
            "device": args.device,
            "dtype": args.dtype,
            "max_len": int(args.max_len),
            "batch_size": int(args.batch_size),
            "markov_train_sequences": int(args.markov_train),
            "truncation_queries_per_sequence": int(args.truncation_queries),
            "truncation_evaluated": not args.skip_truncation,
        },
        "cohort_level": cohort_level,
        # Absent when --no-record-statistics; every reader of this artefact
        # predates the sidecar and none of them require it.
        "sufficient_statistics": sufficient_statistics,
        "arms": arm_reports,
        "measurable_arms": sorted(
            name for name, report in arm_reports.items() if report["power_verdict"] == "PASS"
        ),
        "unmeasurable_arms": sorted(
            name for name, report in arm_reports.items() if report["power_verdict"] == "FAIL"
        ),
    }
    staged_scale = _staged_scale_record(names, bool(args.allow_staged_scale_arms))
    if staged_scale is not None:
        payload["not_panel_admission"] = True
        payload["staged_scale"] = staged_scale
    second_stage = _second_stage_record(names, bool(args.allow_second_stage_arms))
    if second_stage is not None:
        payload["not_panel_admission"] = True
        payload["second_stage"] = second_stage
    candidate = _candidate_record(names, bool(args.allow_candidate_arms))
    if candidate is not None:
        payload["not_panel_admission"] = True
        payload["not_experiment_admitted"] = True
        payload["candidate"] = candidate
    progen3 = _progen3_record(names, bool(args.allow_progen3_arms))
    if progen3 is not None:
        payload["not_panel_admission"] = True
        payload["progen3"] = progen3
    write_json(destination, payload)
    print(f"wrote {destination}")
    if sufficient_statistics is not None:
        print(f"wrote {args.out / sufficient_statistics['path']}")
    if reference_records_path is not None:
        # Named rather than left to be reconstructed: the frozen cohort beside it
        # is derived from the sidecar's name and never has to be typed, whereas
        # this path is an argument an operator passes to
        # context_information_bootstrap.py by hand.
        print(f"wrote {reference_records_path}  (pass to 41 as --reference-json)")
    if cohort_level.get("residue_markov_baselines"):
        ladder = cohort_level["residue_markov_baselines"]["cross_entropy_bits_per_residue"]
        print("residue Markov (bits/residue): " + "  ".join(
            f"{key} {value:.4f}" for key, value in sorted(ladder.items())
        ))


if __name__ == "__main__":
    main()
