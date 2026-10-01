#!/usr/bin/env python3
"""Does a conditioned arm generate what it is asked for? The native interface.

EXP-R2-227. **The first measurement in this programme that samples
from a model rather than scoring one.** Every prior reading is a likelihood, a
context information or a rank correlation; nothing had ever asked a generative
model to generate.

Two panel-adjacent checkpoints carry a conditioning interface that was trained
into them and has never been exercised: ZymCTRL's EC tag, which is priced at
1.73 nats -- so it demonstrably moves the distribution, and **nobody has asked
whether it moves it in the requested direction** -- and ProLLaMA Stage 2's
`[Generate by superfamily]` instruction, declared in `joint_modes` and never run.

**This is not the retired steering line.** Audit 9.1 retires internal-feature EC
steering, which returned a measured 0/8 across three attempts. Nothing here
injects a feature, a direction or a coefficient: the intervention is the prompt
the model was trained to receive. `src.capability.generation.conditioned_generation.CEILING`
carries that distinction and the rest of the binding ceiling, and every artefact
this stage writes carries it too.

Five sub-stages, in the order the registration froze and for the reason it gives
-- generation is cheap relative to the oracles, so nothing about the ordering may
be relaxed to save time:

``--stage instruments``   build HMMER, verify Pfam-A against its published
                          digest, press it, and record CLEAN, ESMFold and
                          Foldseek availability BEFORE any generation exists.
``--stage queue``         the frozen class queue, drawn under seed 20260826 from
                          each arm's own admissible class list and pinned by a
                          content digest every later sub-stage verifies.
``--stage anchors``       the per-class instrument price -- real exemplars
                          against length-matched random UniRef50 -- and the
                          surviving cohort. A class that fails is removed here,
                          before any generation is scored.
``--stage generate``      the only GPU sub-stage: 200 samples per
                          (arm x class x condition) at one frozen sampling
                          configuration.
``--stage score``         the oracle, the near-duplicate grouping, the
                          class-clustered bootstrap and the compound.

CPU for everything except ``generate``. The HMMER/Pfam oracle, the InterPro
release and the UniRef50 corpus are staged on the workstation and not in the
pod, so the split is where the instruments are.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.capability.generation import conditioned_generation as cg  # noqa: E402
from src.capability.core import protein_annotations as ci  # noqa: E402
from src.capability.core.arms import MODEL_BASE_DIR, REPO  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402

DEFAULT_OUT = REPO / "results/R6/conditioned_generation_20260826"
DEFAULT_WORK = REPO.parent / "work/conditioned_generation"

#: The frozen queue lives in tracked evidence rather than under ignored results:
#: it is built before any generation exists, it is what every later sub-stage is
#: pinned to, and a cohort that only exists on one host is not a frozen cohort.
FROZEN_QUEUE = REPO_ROOT / "results/R6/conditioned_generation_queue_20260826/class_queue.json"

#: Reference (b). ProGen2-medium at 764.8M is the declared floor for ZymCTRL;
#: ProtGPT2 at about 774M is the same architecture family as ZymCTRL and is
#: reported beside it as a second unconditioned floor, never as the gate.
PRIMARY_FLOOR = "progen2-medium"

HMMER_TARBALL = REPO / "external/tools/hmmer-3.4.tar.gz"
PFAM_ARCHIVE = REPO / "external/references/pfam/Pfam-A.hmm.gz"
PFAM_CHECKSUM = REPO / "external/references/pfam/Pfam-A.hmm.gz.sha256"
#: The staged CLEAN checkout is nested one level under the resource directory;
#: naming the outer directory would report the source as absent when what is
#: actually missing is the weights, which is a different failure.
CLEAN_ROOT = REPO / "external/code/clean/CLEAN"
FOLDSEEK_TARBALL = REPO / "external/tools/foldseek-linux-avx2.tar.gz"
ESMFOLD_DIR = MODEL_BASE_DIR / "esmfold_v1"
INTERPRO_ENTRY_LIST = REPO / "data/interpro/entry.list"
CATH_TABLE = REPO / "data/interpro/cath_superfamily.tsv"


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _preamble(artifact: str) -> dict[str, Any]:
    return {
        "schema_version": cg.SCHEMA_VERSION,
        "artifact": artifact,
        "campaign": cg.PRE_REGISTRATION,
        "created_utc": _timestamp(),
        "ceiling": dict(cg.CEILING),
        "not_the_retired_steering_line": cg.CEILING["behavioural_not_mechanistic"],
    }


# ---------------------------------------------------------------- instruments


def run_instruments(args: argparse.Namespace) -> dict[str, Any]:
    """Build the oracle, and record what cannot be run before anything is generated."""

    tool = ci.prepare_hmmer(HMMER_TARBALL, args.work / "hmmer")
    database = ci.prepare_pfam(PFAM_ARCHIVE, PFAM_CHECKSUM, args.work / "pfam", tool=tool)
    payload = _preamble("conditioned_generation_instruments")
    payload.update(
        {
            "primary_oracle": cg.PRIMARY_ORACLE,
            "primary_oracle_note": (
                "a profile HMM is a fixed, inspectable statistical model with no learned "
                "distributed representation, so the verdict is not routed through another "
                "neural model (L9)"
            ),
            "hmmer": tool.record(),
            "pfam": database.record(),
            "pfam_threshold": cg.PFAM_THRESHOLD,
            "clean_ec": cg.clean_availability(CLEAN_ROOT),
            "structural_covariates": cg.structural_covariate_availability(
                esmfold=args.esmfold, foldseek_tarball=FOLDSEEK_TARBALL
            ),
            "oracle_channels": list(cg.ORACLE_CHANNELS),
        }
    )
    return payload


# ---------------------------------------------------------------- the queue


def _ec_queue(seed: int, n_classes: int) -> dict[str, Any]:
    counts = cg.ec_class_census()
    eligible = sorted(
        (label for label, count in counts.items() if count >= cg.MIN_CLASS_RECORDS),
        key=cg.canonical_ec_order,
    )
    drawn = cg.seeded_draw(eligible, n=n_classes, seed=seed)
    entries = cg.build_queue([(label, label, counts[label]) for label in drawn], seed=seed)
    top_level: dict[str, int] = {}
    for label, count in counts.items():
        top_level[label.split(".")[0]] = top_level.get(label.split(".")[0], 0) + count
    return {
        "label_kind": "ec_number",
        "classes": [entry.record() for entry in entries],
        "census": {
            "corpus": "ec_labelled_swissprot",
            "records": int(sum(counts.values())),
            "distinct_full_ec_numbers": len(counts),
            "records_per_top_level_class": {key: int(top_level[key]) for key in sorted(top_level)},
            "classes_with_at_least_200_records": len(eligible),
            "eligible_note": (
                "the eligible cut is at 200 records because a class needs a "
                "100-record referent draw and a DISJOINT 100-record anchor draw; a "
                "referent fitted on the sequences the anchor prices would return a "
                "real-side rate of one whatever the oracle does"
            ),
        },
        "not_reusing_the_retired_eight": (
            "the eight EC classes of EXP-R2-003/004 are deliberately not reused: they "
            "were selected inside the retired steering and atlas line for a different "
            "purpose, they are not a frozen queue, and inheriting them would inherit a "
            "selection this campaign cannot audit"
        ),
    }


def _superfamily_queue(seed: int, n_classes: int) -> dict[str, Any]:
    digest = sha256_file(cg.PROLLAMA_SUPERFAMILIES)
    if digest != cg.PROLLAMA_SUPERFAMILIES_SHA256:
        raise RuntimeError(
            f"{cg.PROLLAMA_SUPERFAMILIES} hashes to {digest}, not the staged "
            f"{cg.PROLLAMA_SUPERFAMILIES_SHA256}; the prompt's label space is part of "
            "this measurement's identity"
        )
    labels = cg.single_superfamily_labels()
    resolved, census = cg.superfamily_exemplars(
        labels,
        entry_list=INTERPRO_ENTRY_LIST,
        interpro_xml=cg.INTERPRO_XML,
        cath_table=CATH_TABLE,
    )
    admissible = sorted(
        label
        for label, block in resolved.items()
        if len(block["accessions"]) >= cg.MIN_CLASS_RECORDS
    )
    census["with_at_least_200_swissprot_exemplars"] = len(admissible)
    census["superfamilies_file_sha256"] = digest
    realised = min(n_classes, len(admissible))
    block: dict[str, Any] = {
        "label_kind": "interpro_homologous_superfamily",
        "census": census,
        "admissible_classes": len(admissible),
        "requested_classes": n_classes,
        "realised_classes": realised,
    }
    if len(admissible) < cg.MINIMUM_CLASSES:
        block["classes"] = []
        block["not_drawn_reason"] = (
            f"{len(admissible)} admissible superfamilies is below the "
            f"{cg.MINIMUM_CLASSES}-class floor; this arm is not scored and the "
            "shortfall is reported. The cohort is never topped up"
        )
        return block
    drawn = cg.seeded_draw(admissible, n=realised, seed=seed)
    entries = cg.build_queue(
        [(resolved[label]["interpro"], label, len(resolved[label]["accessions"])) for label in drawn],
        seed=seed,
    )
    block["classes"] = [entry.record() for entry in entries]
    block["exemplars"] = {
        resolved[label]["interpro"]: {
            "label": label,
            "cath_superfamilies": resolved[label]["cath_superfamilies"],
            "n_accessions": len(resolved[label]["accessions"]),
            "accessions": resolved[label]["accessions"],
        }
        for label in drawn
    }
    block["not_drawn_reason"] = None
    return block


def run_queue(args: argparse.Namespace) -> dict[str, Any]:
    payload = _preamble("conditioned_generation_class_queue")
    payload["pre_registration"] = cg.PRE_REGISTRATION
    payload["draw"] = {
        "seed": int(args.draw_seed),
        "classes_per_arm": int(args.classes),
        "mode": "seeded_permutation_of_the_canonically_ordered_admissible_list",
        "mismatched_pairing": "fixed_point_free_permutation_under_the_same_seed",
        "rule": (
            "never a prefix of a frequency table and never the head of a file "
            "(Appendix B rule 1). The bootstrap unit is the CLASS, so the class count "
            "is the sample size, and sixteen rather than the eight-unit floor is a "
            "deliberate choice: eight is where a percentile interval becomes defined, "
            "not where it becomes informative"
        ),
    }
    payload["arms"] = {
        "zymctrl": _ec_queue(args.draw_seed, args.classes),
        "prollama": _superfamily_queue(args.draw_seed, args.classes),
    }
    payload["digest"] = cg.queue_digest(payload)
    return payload


# ----------------------------------------------------------------- anchors


def _anchor_pools(arm_name: str, entries: Sequence[cg.ClassEntry], queue: Mapping[str, Any]) -> dict[str, list[str]]:
    if arm_name == "zymctrl":
        return cg.ec_class_records([entry.key for entry in entries])
    exemplars = queue["arms"]["prollama"]["exemplars"]
    wanted = {
        accession
        for entry in entries
        for accession in exemplars[entry.key]["accessions"]
    }
    sequences = cg.swissprot_sequences(wanted)
    return {
        entry.key: [
            sequences[accession]
            for accession in exemplars[entry.key]["accessions"]
            if accession in sequences
        ]
        for entry in entries
    }


def run_anchors(args: argparse.Namespace) -> dict[str, Any]:
    queue = cg.load_queue(args.queue)
    spec = cg.arm(args.arm)
    entries = cg.queue_entries(queue, args.arm)
    if not entries:
        payload = _preamble("conditioned_generation_anchors")
        payload.update(
            {
                "arm": args.arm,
                "queue_digest": queue["digest"],
                "classes": {},
                "admitted_classes": [],
                "not_run_reason": queue["arms"][args.arm].get("not_drawn_reason"),
            }
        )
        return payload

    instruments = _read(args.instruments)
    tool = ci.prepare_hmmer(HMMER_TARBALL, args.work / "hmmer")
    database = ci.prepare_pfam(PFAM_ARCHIVE, PFAM_CHECKSUM, args.work / "pfam", tool=tool)

    pools = _anchor_pools(args.arm, entries, queue)
    draws: dict[str, dict[str, list[str]]] = {}
    for entry in entries:
        pool = pools.get(entry.key, [])
        if len(pool) < cg.MIN_CLASS_RECORDS:
            draws[entry.key] = {"referent": [], "real": [], "shortfall": len(pool)}
            continue
        referent, real = cg.split_draw(pool, seed=args.draw_seed)
        draws[entry.key] = {"referent": referent, "real": real}

    lengths = [len(sequence) for block in draws.values() for sequence in block.get("real", ())]
    if not lengths:
        raise RuntimeError(f"{args.arm}: no class supplied an anchor draw")
    # The reservoir is cached under the band it was drawn in, and the band is
    # verified on reuse. A reservoir drawn for one arm's length distribution and
    # reused for another's would silently supply length matches that are not
    # matches, which is the one property the random side of the anchor has.
    band = (
        max(1, int(min(lengths) * (1.0 - cg.LENGTH_MATCH_TOLERANCE))),
        int(max(lengths) * (1.0 + cg.LENGTH_MATCH_TOLERANCE)) + 1,
    )
    pool_path = args.work / f"uniref50_reservoir_{band[0]}_{band[1]}_{args.reservoir}.json"
    if pool_path.is_file():
        cached = json.loads(pool_path.read_text(encoding="utf-8"))
        if tuple(cached["band"]) != band or int(cached["seed"]) != int(args.draw_seed):
            raise RuntimeError(
                f"{pool_path} was drawn at band {cached['band']} under seed "
                f"{cached['seed']}, not at {list(band)} under {args.draw_seed}"
            )
        reservoir = cached["sequences"]
    else:
        reservoir = cg.uniref50_pool(
            size=args.reservoir, seed=args.draw_seed, min_len=band[0], max_len=band[1]
        )
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            pool_path,
            {"sequences": reservoir, "seed": int(args.draw_seed), "band": list(band)},
        )

    named: dict[str, str] = {}
    for entry in entries:
        block = draws[entry.key]
        if not block.get("real"):
            continue
        block["random"] = cg.length_matched(block["real"], reservoir, seed=args.draw_seed)
        safe = entry.key.replace(".", "_").replace("/", "_")
        for kind in ("referent", "real", "random"):
            for index, sequence in enumerate(block[kind]):
                named[f"{safe}|{kind}|{index}"] = sequence

    hits, scan = cg.annotate(
        named,
        tool=tool,
        database=database,
        workspace=args.work / f"anchors_{args.arm}",
        threads=args.hmmscan_threads,
        shards=args.hmmscan_shards,
        label=f"anchors_{args.arm}",
    )

    classes: dict[str, Any] = {}
    for entry in entries:
        block = draws[entry.key]
        safe = entry.key.replace(".", "_").replace("/", "_")
        if not block.get("real"):
            classes[entry.key] = {
                "label": entry.label,
                "admitted": False,
                "unmeasurable_reasons": [
                    f"the corpus supplied {block.get('shortfall', 0)} eligible records, "
                    f"below the {cg.MIN_CLASS_RECORDS} a disjoint referent and anchor draw need"
                ],
                "referent": [],
            }
            continue
        referent_names = [f"{safe}|referent|{index}" for index in range(len(block["referent"]))]
        real_names = [f"{safe}|real|{index}" for index in range(len(block["real"]))]
        random_names = [f"{safe}|random|{index}" for index in range(len(block["random"]))]
        referent = cg.referent_from_draw(hits, referent_names)
        record = {"label": entry.label}
        if referent:
            record.update(
                cg.anchor_record(
                    real=cg.assigned(hits, real_names, referent),
                    random=cg.assigned(hits, random_names, referent),
                    referent=referent,
                )
            )
        else:
            record.update(cg.anchor_record(real=[], random=[], referent=()))
        classes[entry.key] = record

    admitted = sorted(key for key, block in classes.items() if block.get("admitted"))
    payload = _preamble("conditioned_generation_anchors")
    payload.update(
        {
            "arm": args.arm,
            "channel": spec.channel,
            "queue_digest": queue["digest"],
            "instruments_digest": instruments.get("pfam", {}).get("source_sha256"),
            "referent_share": cg.REFERENT_FAMILY_SHARE,
            "referent_note": (
                "the class-to-profile map is derived from a disjoint referent draw of "
                "the class's own real records, never from the anchor draw and never "
                "from a generation"
            ),
            "draw_seed": int(args.draw_seed),
            "random_side": {
                "corpus": "uniref50",
                "reservoir_size": int(args.reservoir),
                "length_band": list(band),
                "tolerance": cg.LENGTH_MATCH_TOLERANCE,
                "note": (
                    "real length-matched UniRef50 proteins, which is what EXP-R2-015's "
                    "control was; a synthetic residue string would fail every profile "
                    "for reasons that have nothing to do with class specificity"
                ),
            },
            "hmmscan": scan,
            "classes": classes,
            "admitted_classes": admitted,
            "n_admitted": len(admitted),
            "minimum_classes": cg.MINIMUM_CLASSES,
            "arm_scorable": len(admitted) >= cg.MINIMUM_CLASSES,
            "shortfall_note": (
                None
                if len(admitted) >= cg.MINIMUM_CLASSES
                else "fewer than eight classes survived the instrument anchor, so this "
                "arm is not scored and the shortfall is reported. The cohort is not "
                "topped up with classes drawn after an anchor was seen"
            ),
            "not_run_reason": None,
        }
    )
    return payload


# ---------------------------------------------------------------- generation


def _load(spec: cg.GenerationArm, *, device: str, dtype: str) -> Any:
    """One arm on one card, through the door its family already declares."""

    if spec.loader == "panel":
        from src.capability.core.arms import load_arm

        return load_arm(spec.checkpoint, device=device, dtype=dtype)
    if spec.loader == "lineage":
        from src.capability.models import joint_lineage

        return joint_lineage.load_rung("prollama", device=device, dtype=dtype)
    raise ValueError(f"{spec.name}: no loader is declared")


def _cells(spec: cg.GenerationArm, entries: Sequence[cg.ClassEntry]) -> list[dict[str, Any]]:
    """Every (class, condition) cell this arm owes, and nothing else."""

    cells: list[dict[str, Any]] = []
    if spec.conditioned:
        for entry in entries:
            cells.append(
                {
                    "class_key": entry.key,
                    "condition": "requested",
                    "label": entry.label,
                    "requested_class": entry.key,
                }
            )
            cells.append(
                {
                    "class_key": entry.key,
                    "condition": "mismatched",
                    "label": entry.mismatched_label,
                    "requested_class": entry.mismatched_key,
                }
            )
    if spec.name in cg.FLOORS.get(spec.name, ()) or spec.role == "floor":
        cells.append(
            {
                "class_key": "__unconditioned__",
                "condition": "unconditioned_floor",
                "label": None,
                "requested_class": None,
            }
        )
    return cells


#: What a self-check samples. Deliberately tiny and deliberately NOT the frozen
#: configuration: it establishes that an arm loads, renders its own prompt and
#: returns a sequence at all, which is the interface clause the registration makes
#: a stopping condition. It is never a measurement and never writes a campaign
#: artefact.
SELF_CHECK_SAMPLES = 2
SELF_CHECK_TOKENS = 24


def run_self_check(args: argparse.Namespace) -> dict[str, Any]:
    """Can this arm be prompted and does it return a sequence? Interface only.

    "An arm fails a load, self-check or rendering check -> that arm stops. Do not
    substitute another conditioned arm; there are only two." This is that check,
    and it is separated from the campaign so that the frozen sampling
    configuration cannot be moved in order to run one.
    """

    spec = cg.arm(args.arm)
    if spec.conditioned and spec.modality == "protein":
        entries = cg.queue_entries(cg.load_queue(args.queue), args.arm)
    elif spec.conditioned:
        entries = cg.script_classes()
    else:
        entries = ()
    handle = _load(spec, device=args.device, dtype=args.dtype)
    end = cg.end_delimiter_for(handle, spec) if spec.modality == "protein" else None
    probes: dict[str, Any] = {}
    for cell in _cells(spec, entries)[:2]:
        prompt = cg.prompt_for(handle, spec, cell["label"])
        raw = cg.sample_continuations(
            handle.model,
            handle.tokenizer,
            prompt,
            n=SELF_CHECK_SAMPLES,
            seed=cg.SAMPLING_SEED,
            batch_size=SELF_CHECK_SAMPLES,
            max_new_tokens=SELF_CHECK_TOKENS,
            use_cache=spec.kv_cache,
        )
        samples = (
            [cg.extract_protein(text, end_delimiter=end) for text in raw]
            if spec.modality == "protein"
            else raw
        )
        probes[f"{cell['class_key']}|{cell['condition']}"] = {
            "prompt": prompt,
            "raw": raw,
            "extracted": samples,
            "nonempty": [bool(sample) for sample in samples],
        }
    passed = bool(probes) and any(
        any(block["nonempty"]) for block in probes.values()
    )
    payload = _preamble("conditioned_generation_self_check")
    payload.update(
        {
            "arm": args.arm,
            "is_not_a_measurement": (
                f"{SELF_CHECK_SAMPLES} samples of {SELF_CHECK_TOKENS} tokens at a "
                "configuration this campaign does not measure at. It establishes only "
                "that the arm loads, renders its own prompt and returns a sequence"
            ),
            "end_delimiter": end,
            "kv_cache": spec.kv_cache,
            "probes": probes,
            "passed": passed,
            "failure_branch": (
                None
                if passed
                else "this arm stops. Another conditioned arm is not substituted; there "
                "are only two"
            ),
            "checkpoint_facts": getattr(handle, "facts", None),
        }
    )
    return payload


def require_self_check(args: argparse.Namespace) -> dict[str, Any]:
    """No arm is sampled for the campaign before its interface check has passed.

    "An arm fails a load, self-check or rendering check -> that arm stops." That
    is executable here rather than a sentence in a log entry: a missing or failed
    self-check artefact refuses the cell, and no other conditioned arm is
    substituted for it.
    """

    path = args.self_check_dir / f"generation_self_check_{args.arm}.json"
    if not path.is_file():
        raise RuntimeError(
            f"{path} is absent: {args.arm} has not passed its interface check, and an "
            "arm is sampled only after it has. No other arm is substituted"
        )
    record = _read(path)
    if record.get("arm") != args.arm or not record.get("passed"):
        raise RuntimeError(
            f"{path} does not record a passed self-check for {args.arm}; this arm stops"
        )
    return record


def run_generate(args: argparse.Namespace) -> dict[str, Any]:
    spec = cg.arm(args.arm)
    self_check = require_self_check(args)
    entries: tuple[cg.ClassEntry, ...] = ()
    queue_digest = None
    if spec.conditioned and spec.modality == "protein":
        queue = cg.load_queue(args.queue)
        queue_digest = queue["digest"]
        entries = cg.queue_entries(queue, args.arm)
        if not entries:
            payload = _preamble("conditioned_generation_samples")
            payload.update(
                {
                    "arm": args.arm,
                    "queue_digest": queue_digest,
                    "cells": {},
                    "not_run_reason": queue["arms"][args.arm].get("not_drawn_reason"),
                }
            )
            return payload
    elif spec.conditioned:
        entries = cg.script_classes()

    handle = _load(spec, device=args.device, dtype=args.dtype)
    end = cg.end_delimiter_for(handle, spec) if spec.modality == "protein" else None
    cells: dict[str, Any] = {}
    for cell in _cells(spec, entries):
        prompt = cg.prompt_for(handle, spec, cell["label"])
        seed = cg.cell_seed(
            seed=args.sampling_seed,
            arm_name=spec.name,
            class_key=cell["class_key"],
            condition=cell["condition"],
        )
        raw = cg.sample_continuations(
            handle.model,
            handle.tokenizer,
            prompt,
            n=args.generations,
            seed=seed,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_cache=spec.kv_cache,
        )
        if spec.modality == "protein":
            samples = [cg.extract_protein(text, end_delimiter=end) for text in raw]
            statistics = cg.composition(samples)
            # A generation that ran to the token budget without ever reaching its
            # arm's end delimiter is a truncated fragment, not a finished
            # sequence, and the two are different objects. The count is reported
            # per cell rather than left to be inferred from a length histogram;
            # it is a covariate and is never gated on.
            statistics["n_terminated"] = int(sum(1 for text in raw if end in text))
            statistics["termination_rate"] = statistics["n_terminated"] / len(raw)
            statistics["termination_note"] = (
                "the frozen budget is 400 new tokens, which on a residue-tokenised arm "
                "is 400 residues; a cell whose termination rate is low is reporting on "
                "domain-length fragments and the per-class anchors were measured on "
                "whole real proteins"
            )
        else:
            samples = raw
            statistics = cg.text_statistics(samples)
        key = f"{cell['class_key']}|{cell['condition']}"
        cells[key] = {
            **cell,
            "prompt": prompt,
            "seed": seed,
            "samples": samples,
            "statistics": statistics,
        }
        print(
            f"{spec.name} {key}: {len(samples)} samples, {statistics['n_empty']} empty, "
            f"terminated {statistics.get('n_terminated', '-')}",
            flush=True,
        )

    payload = _preamble("conditioned_generation_samples")
    payload.update(
        {
            "arm": args.arm,
            "role": spec.role,
            "channel": spec.channel,
            "modality": spec.modality,
            "arm_note": spec.note,
            "queue_digest": queue_digest,
            "end_delimiter": end,
            "sampling": {
                "top_p": args.top_p,
                "temperature": args.temperature,
                "top_k": cg.TOP_K,
                "repetition_penalty": cg.REPETITION_PENALTY,
                "max_new_tokens": args.max_new_tokens,
                "campaign_seed": int(args.sampling_seed),
                "generations_per_cell": int(args.generations),
                "batch_size": int(args.batch_size),
                "kv_cache": spec.kv_cache,
                "kv_cache_note": (
                    "whether this checkpoint's modeling code honours Transformers' "
                    "current Cache interface; an arm declared False samples with the "
                    "cache off, which recomputes the prefix every step and is correct"
                ),
                "per_cell_seed_rule": (
                    "sha256 of the campaign, arm, class and condition, added to the "
                    "campaign seed, so a class and its mismatched partner never share a "
                    "sample and the run is still reproducible from the campaign seed"
                ),
                "batch_size_note": (
                    "a feasibility parameter, not a scientific one, but the per-batch "
                    "seed makes the sample depend on it, so it is recorded with the run"
                ),
                "post_selection_filter": cg.POST_SELECTION_FILTER,
                "post_selection_note": cg.POST_SELECTION_NOTE,
            },
            "checkpoint_facts": getattr(handle, "facts", None),
            "self_check": {
                "passed": self_check["passed"],
                "created_utc": self_check["created_utc"],
                "is_not_a_measurement": self_check["is_not_a_measurement"],
            },
            "cells": cells,
            "not_run_reason": None,
        }
    )
    return payload


# -------------------------------------------------------------------- scoring


def _identity_covariate(
    sequences: Mapping[str, str], args: argparse.Namespace
) -> dict[str, Any]:
    """Maximum identity to the searched corpus, per generation. A covariate only."""

    from src.capability.context import homology

    if args.identity_corpus is None:
        return {
            "run": False,
            "reason": "no --identity-corpus was named, so this covariate is reported NOT RUN "
            "rather than as an absence of homology",
            "ceiling": cg.CEILING["max_identity_is_a_covariate"],
        }
    tool = homology.prepare_diamond(args.diamond_tarball, args.diamond_checksum, args.work / "diamond")
    database = homology.build_database(
        tool,
        args.identity_corpus,
        args.identity_db,
        threads=args.diamond_threads,
        tmpdir=args.work / "diamond_tmp",
    )
    workspace = args.work / "identity"
    workspace.mkdir(parents=True, exist_ok=True)
    fasta = ci.write_fasta(workspace / "generations.fasta", sequences)
    output = workspace / "generations.tsv"
    command, _ = homology.run_diamond_blastp(
        tool,
        database,
        fasta,
        output,
        threads=args.diamond_threads,
        sensitivity="very-sensitive",
        evalue=1e-3,
        max_target_seqs=5,
    )
    best: dict[str, float] = {}
    for hit in homology.parse_hits(output):
        value = hit.identity_over_query
        if value > best.get(hit.query, 0.0):
            best[hit.query] = value
    return {
        "run": True,
        "corpus": str(args.identity_corpus),
        "corpus_records": int(database.source_records),
        "command": command,
        "max_identity_over_query": {name: best.get(name, 0.0) for name in sequences},
        "corpus_note": (
            "the searched corpus is what this host stages, which is not necessarily the "
            "arm's own pretraining set; what was searched is named here rather than "
            "implied"
        ),
        "ceiling": cg.CEILING["max_identity_is_a_covariate"],
    }


def _condition_rates(
    cells: Mapping[str, Any],
    *,
    unit: str,
    referents: Mapping[str, Sequence[str]],
    hits: Mapping[str, Sequence[Mapping[str, Any]]],
    names: Mapping[str, list[str]],
    admitted: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Per class and condition, the near-duplicate-grouped assignment rate."""

    rates: dict[str, dict[str, Any]] = {}
    for key, cell in cells.items():
        samples = cell["samples"]
        groups, grouping = cg.near_duplicate_group_ids(samples, unit=unit)
        block: dict[str, Any] = {"grouping": grouping, "per_class": {}}
        for class_key in admitted:
            hit_flags = cg.assigned(hits, names[key], referents[class_key])
            block["per_class"][class_key] = {
                "rate": cg.grouped_rate(hit_flags, groups),
                "ungrouped_rate": float(np.mean([1.0 if flag else 0.0 for flag in hit_flags])),
                "n": len(samples),
            }
        block["group_ids"] = [int(value) for value in groups]
        rates[key] = block
    return rates


def _protein_arm_report(
    *,
    arm_name: str,
    queue: Mapping[str, Any],
    anchors: Mapping[str, Any],
    generations: Mapping[str, Any],
    floors: Mapping[str, Any],
    args: argparse.Namespace,
    tool: Any,
    database: Any,
) -> dict[str, Any]:
    admitted = list(anchors.get("admitted_classes", ()))
    referents = {key: anchors["classes"][key]["referent"] for key in admitted}
    report: dict[str, Any] = {
        "arm": arm_name,
        "channel": cg.arm(arm_name).channel,
        "queue_digest": queue["digest"],
        "n_admitted_classes": len(admitted),
        "admitted_classes": admitted,
        "unmeasurable_classes": {
            key: block.get("unmeasurable_reasons")
            for key, block in anchors.get("classes", {}).items()
            if not block.get("admitted")
        },
        "instrument_anchors": {key: anchors["classes"][key] for key in admitted},
    }
    if not anchors.get("arm_scorable"):
        report["verdict"] = {"outcome": "not_scored"}
        report["not_scored_reason"] = anchors.get("shortfall_note") or anchors.get("not_run_reason")
        return report
    if generations.get("not_run_reason"):
        report["verdict"] = {"outcome": "not_scored"}
        report["not_scored_reason"] = generations["not_run_reason"]
        return report

    named: dict[str, str] = {}
    names: dict[str, list[str]] = {}
    sources: list[tuple[str, Mapping[str, Any]]] = [(arm_name, generations)]
    sources.extend((name, payload) for name, payload in floors.items())
    for source, payload in sources:
        for key, cell in payload["cells"].items():
            handle = f"{source}|{key}"
            names[handle] = []
            for index, sequence in enumerate(cell["samples"]):
                name = f"{source}#{key}#{index}".replace(" ", "_")
                names[handle].append(name)
                if sequence:
                    named[name] = sequence

    hits, scan = cg.annotate(
        named,
        tool=tool,
        database=database,
        workspace=args.work / f"score_{arm_name}",
        threads=args.hmmscan_threads,
        shards=args.hmmscan_shards,
        label=f"score_{arm_name}",
    )
    report["hmmscan"] = scan

    cells = {f"{arm_name}|{key}": cell for key, cell in generations["cells"].items()}
    for source, payload in floors.items():
        for key, cell in payload["cells"].items():
            cells[f"{source}|{key}"] = cell
    rates = _condition_rates(
        cells, unit="residues", referents=referents, hits=hits, names=names, admitted=admitted
    )

    def rate(source: str, class_key: str, condition: str, target: str) -> float:
        return rates[f"{source}|{class_key}|{condition}"]["per_class"][target]["rate"]

    requested = {key: rate(arm_name, key, "requested", key) for key in admitted}
    mismatched = {key: rate(arm_name, key, "mismatched", key) for key in admitted}
    floor_rates = {
        source: {
            key: rate(source, "__unconditioned__", "unconditioned_floor", key) for key in admitted
        }
        for source in floors
    }
    against_mismatch = {key: requested[key] - mismatched[key] for key in admitted}
    against_floor = {
        source: {key: requested[key] - values[key] for key in admitted}
        for source, values in floor_rates.items()
    }

    mismatch_block = cg.class_clustered_mean(
        against_mismatch, resamples=args.bootstrap, seed=args.bootstrap_seed
    )
    floor_blocks = {
        source: cg.class_clustered_mean(values, resamples=args.bootstrap, seed=args.bootstrap_seed + 1)
        for source, values in against_floor.items()
    }
    primary = floor_blocks.get(PRIMARY_FLOOR)
    if primary is None:
        raise RuntimeError(
            f"the declared floor {PRIMARY_FLOOR!r} produced no generations; clause 2 has "
            "no reference and the arm is not scored against a substitute"
        )

    report.update(
        {
            "per_class": {
                key: {
                    "label": next(
                        entry["label"]
                        for entry in queue["arms"][arm_name]["classes"]
                        if entry["key"] == key
                    ),
                    "p_requested": requested[key],
                    "p_mismatched": mismatched[key],
                    "p_floor": {source: values[key] for source, values in floor_rates.items()},
                    "requested_minus_mismatched": against_mismatch[key],
                    "requested_minus_floor": {
                        source: values[key] for source, values in against_floor.items()
                    },
                }
                for key in admitted
            },
            "requested_minus_mismatched": mismatch_block,
            "requested_minus_floor": floor_blocks,
            "primary_floor": PRIMARY_FLOOR,
            "secondary_floors_are_reported_not_gated": [
                source for source in floor_blocks if source != PRIMARY_FLOOR
            ],
            "verdict": cg.compound_verdict(
                against_mismatch=mismatch_block,
                against_floor=primary,
                per_class_contrast=against_mismatch,
            ),
            "composition": {
                key: cell["statistics"] for key, cell in generations["cells"].items()
            },
            "floor_composition": {
                source: {key: cell["statistics"] for key, cell in payload["cells"].items()}
                for source, payload in floors.items()
            },
            "not_scored_reason": None,
        }
    )
    if args.identity_corpus is not None:
        report["max_identity_to_corpus"] = _identity_covariate(named, args)
    else:
        report["max_identity_to_corpus"] = _identity_covariate({}, args)
    return report


def _text_arm_report(
    *, arm_name: str, generations: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """The text positive control: the estimand shape with instrument error removed.

    Appendix B rule 3 requires a text positive control where a real analogue
    exists, and one does: a decoder conditioned on a language tag and scored by
    **Unicode script identity**, a deterministic zero-parameter oracle. The anchor
    is arithmetic rather than empirical, because script detection on text of a
    known script is exact by construction -- and that is precisely why it is the
    right control for the shape. It shows the estimand returns a strong positive
    with instrument error removed, so a weak protein reading cannot be blamed on
    the estimand.
    """

    entries = cg.script_classes()
    keys = [entry.key for entry in entries]
    cells = generations["cells"]
    grouped: dict[str, tuple[np.ndarray, dict[str, Any]]] = {
        key: cg.near_duplicate_group_ids(cell["samples"], unit="characters")
        for key, cell in cells.items()
    }
    scripts = {
        key: [cg.assign_script(sample) for sample in cell["samples"]]
        for key, cell in cells.items()
    }

    def flags(cell_key: str, target: str) -> list[bool]:
        return [value == target for value in scripts[cell_key]]

    def rate(cell_key: str, target: str) -> float:
        return cg.grouped_rate(flags(cell_key, target), grouped[cell_key][0])

    requested = {key: rate(f"{key}|requested", key) for key in keys}
    mismatched = {key: rate(f"{key}|mismatched", key) for key in keys}
    floor = {key: rate("__unconditioned__|unconditioned_floor", key) for key in keys}
    against_mismatch = {key: requested[key] - mismatched[key] for key in keys}
    against_floor = {key: requested[key] - floor[key] for key in keys}

    screen = {}
    for index, key in enumerate(keys):
        screen[key] = cg.two_sample_rate_contrast(
            flags(f"{key}|requested", key),
            grouped[f"{key}|requested"][0],
            flags(f"{key}|mismatched", key),
            grouped[f"{key}|mismatched"][0],
            resamples=args.bootstrap,
            seed=args.bootstrap_seed + index,
        )
    n_attained = sum(1 for block in screen.values() if cg.lower_bound_positive(block))

    mismatch_block = cg.class_clustered_mean(
        against_mismatch, resamples=args.bootstrap, seed=args.bootstrap_seed
    )
    floor_block = cg.class_clustered_mean(
        against_floor, resamples=args.bootstrap, seed=args.bootstrap_seed + 1
    )
    return {
        "arm": arm_name,
        "channel": "script",
        "classes": [entry.record() for entry in entries],
        "oracle": {
            "name": "unicode_script_identity",
            "deterministic": True,
            "parameters": 0,
            "dominance": cg.SCRIPT_DOMINANCE,
            "anchor": (
                "arithmetic rather than empirical: script detection on text of a known "
                "script is exact by construction, which is why this control prices the "
                "estimand's SHAPE rather than an instrument"
            ),
            "latin_excluded_as_a_class": (
                "Latin is the script the prompt is written in and the default output of "
                "every one of these checkpoints, so 'assigned to Latin' is not a "
                "request-driven event. It is still counted in the denominator, so an "
                "English continuation cannot be assigned to Greek on two Greek letters"
            ),
        },
        "per_class": {
            key: {
                "p_requested": requested[key],
                "p_mismatched": mismatched[key],
                "p_unconditioned_floor": floor[key],
                "requested_minus_mismatched": against_mismatch[key],
                "requested_minus_floor": against_floor[key],
                "within_class_screen": screen[key],
            }
            for key in keys
        },
        "requested_minus_mismatched": mismatch_block,
        "requested_minus_floor": floor_block,
        "attainability_screen": {
            "classes_with_a_positive_lower_bound": n_attained,
            "required": cg.MINIMUM_CLASSES,
            "attained": n_attained >= cg.MINIMUM_CLASSES,
            "rule": (
                "base checkpoints, not instruction-tuned ones, so compliance with a "
                "language request is not assumed. If neither text arm reaches a "
                "requested-minus-mismatched contrast whose 95% lower bound exceeds 0 on "
                "at least eight classes, the control is reported UNATTAINABLE on the "
                "available text arms, the protein readings still stand, and the missing "
                "control is carried as a stated limitation rather than quietly dropped"
            ),
        },
        "verdict": cg.compound_verdict(
            against_mismatch=mismatch_block,
            against_floor=floor_block,
            per_class_contrast=against_mismatch,
        ),
        "statistics": {key: cell["statistics"] for key, cell in cells.items()},
    }


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    queue = cg.load_queue(args.queue)
    instruments = _read(args.instruments)
    tool = ci.prepare_hmmer(HMMER_TARBALL, args.work / "hmmer")
    database = ci.prepare_pfam(PFAM_ARCHIVE, PFAM_CHECKSUM, args.work / "pfam", tool=tool)

    def maybe(name: str) -> dict[str, Any] | None:
        path = args.generation_dir / f"generations_{name}.json"
        return _read(path) if path.is_file() else None

    floors = {
        name: payload
        for name in ("progen2-medium", "protgpt2")
        if (payload := maybe(name)) is not None
    }
    protein: dict[str, Any] = {}
    for name in ("zymctrl", "prollama"):
        anchor_path = args.anchor_dir / f"anchors_{name}.json"
        generations = maybe(name)
        if not anchor_path.is_file() or generations is None:
            protein[name] = {
                "arm": name,
                "verdict": {"outcome": "not_scored"},
                "not_scored_reason": (
                    "this arm's anchor artefact or its generations are absent; an arm "
                    "that did not run is reported as not scored and is never replaced "
                    "by another conditioned arm -- there are only two"
                ),
            }
            continue
        protein[name] = _protein_arm_report(
            arm_name=name,
            queue=queue,
            anchors=_read(anchor_path),
            generations=generations,
            floors=floors,
            args=args,
            tool=tool,
            database=database,
        )

    text: dict[str, Any] = {}
    for name in ("qwen2.5-0.5b", "llama-3.2-3b"):
        generations = maybe(name)
        if generations is None:
            continue
        text[name] = _text_arm_report(arm_name=name, generations=generations, args=args)
    control_attained = any(block["attainability_screen"]["attained"] for block in text.values())

    payload = _preamble("conditioned_generation_report")
    payload.update(
        {
            "queue_digest": queue["digest"],
            "instruments": {
                "primary_oracle": instruments.get("primary_oracle"),
                "hmmer_version": instruments.get("hmmer", {}).get("version"),
                "pfam_sha256": instruments.get("pfam", {}).get("source_sha256"),
                "pfam_threshold": instruments.get("pfam_threshold"),
                "clean_ec": instruments.get("clean_ec"),
                "structural_covariates": instruments.get("structural_covariates"),
            },
            "bootstrap": {
                "resamples": int(args.bootstrap),
                "seed": int(args.bootstrap_seed),
                "unit": "the class",
            },
            "protein_arms": protein,
            "text_positive_control": text,
            "text_positive_control_attained": control_attained,
            "text_positive_control_limitation": (
                None
                if control_attained
                else "the text positive control is UNATTAINABLE on the available text "
                "arms. The protein readings stand and this is carried as a stated "
                "limitation of this campaign rather than quietly dropped"
            ),
            "cross_arm_rates_are_descriptive": cg.CEILING["cross_arm_rates_are_descriptive"],
        }
    )
    return payload


# ------------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("instruments", "queue", "anchors", "self-check", "generate", "score"),
        help="the registration's operational sequence, in order; generation is cheap "
        "relative to the oracles and nothing about the ordering may be relaxed to save time",
    )
    parser.add_argument("--arm", default=None, choices=sorted(cg.ARMS), help="the arm an anchor or generation cell belongs to")
    parser.add_argument("--queue", type=Path, default=FROZEN_QUEUE, help="the frozen class queue; its digest is verified before it is used")
    parser.add_argument("--instruments", type=Path, default=None, help="an instruments.json written by --stage instruments")
    parser.add_argument("--anchor-dir", type=Path, default=None, help="directory holding anchors_<arm>.json")
    parser.add_argument("--self-check-dir", type=Path, default=None, help="directory holding generation_self_check_<arm>.json; --stage generate refuses an arm whose interface check is absent or failed")
    parser.add_argument("--generation-dir", type=Path, default=None, help="directory holding generations_<arm>.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK, help="where the built HMMER, the pressed Pfam and the scan tables go; never inside the repository")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--generations", type=int, default=cg.GENERATIONS_PER_CELL)
    parser.add_argument("--top-p", type=float, default=cg.TOP_P)
    parser.add_argument("--temperature", type=float, default=cg.TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=cg.MAX_NEW_TOKENS)
    parser.add_argument("--sampling-seed", type=int, default=cg.SAMPLING_SEED)
    parser.add_argument("--bootstrap", type=int, default=cg.BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=cg.BOOTSTRAP_SEED)
    parser.add_argument("--draw-seed", type=int, default=cg.DRAW_SEED)
    parser.add_argument("--classes", type=int, default=cg.CLASSES_PER_ARM)
    parser.add_argument("--reservoir", type=int, default=200_000, help="UniRef50 records held for the length-matched random anchor side")
    parser.add_argument("--hmmscan-threads", type=int, default=8)
    parser.add_argument("--hmmscan-shards", type=int, default=16)
    parser.add_argument("--identity-corpus", type=Path, default=None, help="FASTA searched for the max-identity COVARIATE; omitted means the covariate is reported NOT RUN, never as an absence of homology")
    parser.add_argument("--identity-db", type=Path, default=None)
    parser.add_argument("--diamond-tarball", type=Path, default=REPO / "external/tools/diamond-linux64-v2.1.24.tar.gz")
    parser.add_argument("--diamond-checksum", type=Path, default=REPO / "external/tools/diamond-linux64-v2.1.24.tar.gz.sha256")
    parser.add_argument("--diamond-threads", type=int, default=32)
    parser.add_argument("--esmfold", type=Path, default=ESMFOLD_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage in ("generate", "score"):
        cg.require_frozen_parameters(
            resamples=args.bootstrap,
            bootstrap_seed=args.bootstrap_seed,
            sampling_seed=args.sampling_seed,
            generations=args.generations,
            top_p=args.top_p,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    if args.stage in ("anchors", "self-check", "generate") and args.arm is None:
        raise SystemExit("--arm is required for --stage anchors and --stage generate")
    if args.stage == "generate" and args.self_check_dir is None:
        raise SystemExit("--self-check-dir is required for --stage generate: an arm is sampled only after its interface check has passed")
    if args.stage in ("anchors", "score") and args.instruments is None:
        raise SystemExit("--instruments is required: the oracle's availability is recorded before any generation is scored")
    if args.stage == "score":
        if args.generation_dir is None:
            raise SystemExit("--generation-dir is required for --stage score")
        if args.anchor_dir is None:
            args.anchor_dir = args.generation_dir
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage not in ("generate", "self-check"):
        # Only the oracle sub-stages need a working directory, and the pod that
        # runs `generate` stages no HMMER, no Pfam and no InterPro release, so
        # creating one there would fail on a path this stage never reads.
        args.work.mkdir(parents=True, exist_ok=True)

    if args.stage == "instruments":
        payload, name = run_instruments(args), "instruments.json"
    elif args.stage == "queue":
        payload, name = run_queue(args), "class_queue.json"
    elif args.stage == "anchors":
        payload, name = run_anchors(args), f"anchors_{args.arm}.json"
    elif args.stage == "self-check":
        payload, name = run_self_check(args), f"generation_self_check_{args.arm}.json"
    elif args.stage == "generate":
        payload, name = run_generate(args), f"generations_{args.arm}.json"
    else:
        payload, name = run_score(args), "conditioned_generation.json"

    destination = args.out / name
    write_json(destination, payload)
    print(f"wrote {destination}")
    if args.stage == "queue":
        print(f"queue digest: {payload['digest']}")
        for arm_name, block in payload["arms"].items():
            print(f"  {arm_name}: {len(block['classes'])} classes ({block['label_kind']})")
    if args.stage == "anchors":
        print(f"admitted classes: {payload.get('n_admitted')} of {len(payload.get('classes', {}))}")
    if args.stage == "self-check":
        print(f"{args.arm} self-check: {'PASS' if payload['passed'] else 'FAIL'}")
    if args.stage == "score":
        for arm_name, block in payload["protein_arms"].items():
            print(f"{arm_name}: {block['verdict']['outcome']}")
        for arm_name, block in payload["text_positive_control"].items():
            print(f"{arm_name} (text control): {block['verdict']['outcome']}")


if __name__ == "__main__":
    main()
