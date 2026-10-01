#!/usr/bin/env python3
"""Does in-context homologue conditioning exist at all in a frozen decoder?

EXP-R2-228. Every established positive in this literature -- PoET,
ProtMamba, E1, Protriever, ProFam -- comes from a model **trained** for
multi-sequence context. None of them answers whether a decoder that never saw a
homologue set during pretraining can use one, which is the question this panel
can ask and they cannot.

The hypothesis to defeat is in-context copying, not a control to include:
Kantroo, Wagner & Machta (arXiv:2504.17068) measure ``progen2-medium`` itself and
find the self-copy likelihood collapse occurs for *random* sequences, persists to
50% divergence and fires on 10-residue needles. ``src.capability.context.context_homologue``
carries the design that follows from that, the binding ceiling, and every frozen
constant; this file is the operational sequence.

Five sub-stages, in the registration's order:

``--stage cohort``      the protein pool, its DIAMOND all-against-all identity
                        bands, the passage pool, its BM25 bands, the local-overlap
                        screen, the near-duplicate groups and the frozen units --
                        pinned by a content digest every later sub-stage verifies.
``--stage plan``        one arm's token budget, its realised k distribution and
                        all four control constructions. Tokenizers only: **no
                        weights load here**, which is what makes "publish the
                        censuses before any model is loaded" executable.
``--stage self-check``  the arm's declared rendering, its scored span and a fixed
                        record's NLL at the campaign precision.
``--stage score``       the only GPU sub-stage, one condition per invocation, in
                        the frozen order: the two controls most likely to close
                        the line are measured before the condition the campaign
                        hopes for.
``--stage analyse``     the paired endpoints, the group bootstrap and the
                        three-clause compound.

CPU for ``cohort``, ``plan`` and ``analyse``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.capability.context import context_homologue as ch  # noqa: E402
from src.capability.context import homology  # noqa: E402
from src.capability.core.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    PANEL,
    Arm,
    arm_spec,
    load_arm,
    load_arm_spec,
    protein_cohort,
    text_cohort,
)
from src.capability.core.io import write_json  # noqa: E402

DEFAULT_OUT = REPO_ROOT / 'results/R5/context_homologue_cohort_20260826'
DEFAULT_WORK = REPO_ROOT.parent / "transfer_work" / "context_homologue"
DIAMOND_TARBALL = REPO_ROOT / "external/tools/diamond-linux64-v2.1.24.tar.gz"
DIAMOND_CHECKSUM = REPO_ROOT / "external/tools/diamond-linux64-v2.1.24.tar.gz.sha256"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _header(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": ch.SCHEMA_VERSION,
        "pre_registration": ch.PRE_REGISTRATION,
        "written_utc": _now(),
        "ceiling": ch.CEILING,
    }
    if extra:
        payload.update(extra)
    return payload


# ----------------------------------------------------------------- the cohort


def run_cohort(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from src.capability.core.arms import PANEL

    pool = protein_cohort(
        args.protein_pool,
        *ch.PROTEIN_BAND,
        seed=args.cohort_draw_seed,
        name="r228_protein_pool",
    )
    fasta = args.work / "protein_pool.fasta"
    homology.write_query_fasta(pool, fasta)
    tool = homology.prepare_diamond(args.diamond_tarball, args.diamond_checksum, args.work / "diamond")
    database = homology.build_database(
        tool,
        fasta,
        args.work / "protein_pool.dmnd",
        threads=args.diamond_threads,
        tmpdir=args.work / "diamond_tmp",
        rebuild=args.rebuild_db,
    )
    hits_tsv = args.work / "protein_pool_hits.tsv"
    command, log_tail = homology.run_diamond_blastp(
        tool,
        database,
        fasta,
        hits_tsv,
        threads=args.diamond_threads,
        sensitivity=ch.DIAMOND_SENSITIVITY,
        evalue=ch.DIAMOND_EVALUE,
        max_target_seqs=ch.DIAMOND_MAX_TARGET_SEQS,
    )
    best: dict[tuple[int, int], float] = {}
    for hit in homology.parse_hits(hits_tsv):
        if hit.query == hit.subject:
            continue
        key = (int(hit.query[1:]), int(hit.subject[1:]))
        value = hit.identity_over_query
        if value > best.get(key, -1.0):
            best[key] = value
    protein = ch.protein_cohort_units(pool.records, best)
    protein["sampling"] = pool.sampling
    protein["diamond"] = {
        "tool": tool.record(),
        "database": database.record(),
        "command": list(command),
        "log_tail": log_tail,
        "sensitivity": ch.DIAMOND_SENSITIVITY,
        "evalue": ch.DIAMOND_EVALUE,
        "max_target_seqs": ch.DIAMOND_MAX_TARGET_SEQS,
        "pairs": len(best),
    }

    documents = text_cohort(
        args.text_pool,
        ch.TEXT_DOCUMENT_MIN_CHARS,
        seed=args.cohort_draw_seed,
        name="r228_text_documents",
    )
    tokenizer = AutoTokenizer.from_pretrained(PANEL[ch.TEXT_ARMS[0]].path)
    passages, sources, carving = _carve_passages(tokenizer, documents.records)
    index = ch.Bm25Index(passages)
    text = ch.text_cohort_units(passages, sources, index, max_examined=args.text_examine)
    text["sampling"] = documents.sampling
    text["carving"] = carving

    payload = _header(
        {
            "draw": {
                "seeds": {
                    "cohort_draw_seed": int(args.cohort_draw_seed),
                    "campaign_draw_seed": ch.DRAW_SEED,
                    "bootstrap_seed": ch.BOOTSTRAP_SEED,
                },
                "protein_pool": int(args.protein_pool),
                "text_pool": int(args.text_pool),
                "protein_band_residues": list(ch.PROTEIN_BAND),
                "text_token_band": list(ch.TEXT_TOKEN_BAND),
                "identity_bands": [list(band) for band in ch.IDENTITY_BANDS],
                "text_bands": [list(band) for band in ch.TEXT_BANDS],
                "near_duplicate_identity": ch.NEAR_DUPLICATE_IDENTITY,
                "band_target_floor": ch.BAND_TARGET_FLOOR,
                "min_context_items": ch.MIN_CONTEXT_ITEMS,
                "high_local_overlap_lcs": ch.HIGH_LOCAL_OVERLAP_LCS,
                "text_high_overlap_lcs": ch.TEXT_HIGH_OVERLAP_LCS,
                "text_shingle_words": ch.TEXT_SHINGLE_WORDS,
                "local_overlap_kmer": ch.LOCAL_OVERLAP_KMER,
                "seed_note": (
                    "the campaign's own draws are seeded at "
                    f"{ch.DRAW_SEED}; the corpus permutation is the panel's one "
                    "declared --cohort-draw-seed, which the manifest pins to the "
                    "same value and which this record carries"
                ),
            },
            "arms": {"protein": list(ch.PROTEIN_ARMS), "text": list(ch.TEXT_ARMS)},
            "excluded_arms": ch.EXCLUDED_ARMS,
            "protein": protein,
            "text": text,
        }
    )
    payload["digest"] = ch.cohort_digest(payload)
    return payload


def _carve_passages(tokenizer, documents: list[str]) -> tuple[list[str], list[int], dict[str, Any]]:
    """Contiguous passages of 80-130 BPE tokens, never a prefix, one per document.

    A passage is kept only when re-encoding its decoded text reproduces the token
    run exactly. A run that does not round-trip would be scored on a different
    token grid from the one it was carved on, so it is dropped and counted rather
    than silently admitted.
    """

    low, high = ch.TEXT_TOKEN_BAND
    generator = np.random.default_rng(ch.DRAW_SEED)
    passages: list[str] = []
    sources: list[int] = []
    too_short = 0
    no_round_trip = 0
    for position, document in enumerate(documents):
        ids = tokenizer(document, return_tensors=None)["input_ids"]
        wanted = int(generator.integers(low, high + 1))
        if len(ids) < wanted + 1:
            too_short += 1
            continue
        start = int(generator.integers(1, len(ids) - wanted + 1))
        run = ids[start : start + wanted]
        text = tokenizer.decode(run)
        if tokenizer(text, return_tensors=None)["input_ids"] != run:
            no_round_trip += 1
            continue
        passages.append(text)
        sources.append(position)
    if not passages:
        raise RuntimeError("no passage survived carving")
    return (
        passages,
        sources,
        {
            "documents_examined": len(documents),
            "passages": len(passages),
            "dropped_too_short": too_short,
            "dropped_no_round_trip": no_round_trip,
            "token_band": list(ch.TEXT_TOKEN_BAND),
            "seed": ch.DRAW_SEED,
            "never_a_prefix": True,
        },
    )


# ------------------------------------------------------------------- the plan


def run_plan(args: argparse.Namespace) -> dict[str, Any]:
    cohort = ch.load_cohort(args.cohort)
    modality = ch.modality_of(args.arm)
    arm = ch.tokenizer_arm(args.arm)
    plan = ch.plan_units(arm, cohort, modality=modality)
    payload = _header(
        {
            "arm": args.arm,
            "modality": modality,
            "cohort_digest": cohort["digest"],
            "position_budget": ch.POSITION_BUDGET,
            "rendering": ch.rendering_check(arm, modality=modality),
            "conditions": {name: ch.CONDITION_PURPOSE[name] for name in ch.CONDITIONS},
            "weights_loaded": False,
            **plan,
        }
    )
    payload["digest"] = ch.plan_digest(payload)
    return payload


# ---------------------------------------------------------------- the census


def run_census(args: argparse.Namespace) -> dict[str, Any]:
    """The censuses the registration requires published before any model loads.

    A compact, durable extract of what the cohort and the four plans already
    hold: the digests every later sub-stage verifies, the per-band eligible and
    drawn counts with the reason each dropped stratum was dropped, the local
    overlap the primary analysis actually contains, and the realised k. It is
    committed as evidence because the artefacts it summarises are large and live
    outside version control, and a census that only exists beside the run it
    justifies cannot be checked afterwards.
    """

    cohort = ch.load_cohort(args.cohort)
    arms_block: dict[str, Any] = {}
    for arm in ch.ARMS:
        path = args.score_dir / f"plan_{arm}.json"
        if not path.is_file():
            arms_block[arm] = {"status": "no plan artefact", "path": str(path)}
            continue
        plan = ch.load_plan(path, cohort=cohort, arm=arm)
        arms_block[arm] = {
            "digest": plan["digest"],
            "modality": plan["modality"],
            "n_units": len(plan["units"]),
            "k_distribution": plan["k_distribution"],
            "context_fraction": plan["context_fraction"],
            "unrelated_token_gap": plan["unrelated_token_gap"],
            "shuffle_exact_items": plan["shuffle_exact_items"],
            "shuffle_trimmed_items": plan["shuffle_trimmed_items"],
            "shuffle_short_items": plan["shuffle_short_items"],
            "shuffle_trim_excess_tokens": plan["shuffle_trim_excess_tokens"],
            "tiled_filler_items": plan["tiled_filler_items"],
            "widened_length_matches": plan["widened_length_matches"],
            "refusals": plan["refusals"],
            "rendering": plan["rendering"],
            "condition_token_gap": _condition_gaps(plan),
        }
    return _header(
        {
            "cohort_digest": cohort["digest"],
            "draw": cohort["draw"],
            "excluded_arms": cohort["excluded_arms"],
            "protein": {
                "census": cohort["protein"]["census"],
                "sampling": cohort["protein"]["sampling"],
                "diamond": {
                    key: cohort["protein"]["diamond"][key]
                    for key in ("tool", "sensitivity", "evalue", "max_target_seqs", "pairs")
                },
                "near_duplicate_pairs": cohort["protein"]["near_duplicate_pairs"],
                "near_duplicate_groups": len(set(cohort["protein"]["groups"])),
                "bootstrap_unit": cohort["protein"]["bootstrap_unit"],
                "filler": _filler_record(cohort["protein"]["filler"]),
                "local_overlap": _overlap_census(cohort["protein"]["units"]),
            },
            "text": {
                "census": cohort["text"]["census"],
                "sampling": cohort["text"]["sampling"],
                "carving": cohort["text"]["carving"],
                "bootstrap_unit": cohort["text"]["bootstrap_unit"],
                "filler": _filler_record(cohort["text"]["filler"]),
                "local_overlap": _overlap_census(cohort["text"]["units"]),
            },
            "arms": arms_block,
        }
    )


def _filler_record(filler: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in filler.items() if key != "record"}


def _overlap_census(units: list[dict[str, Any]]) -> dict[str, Any]:
    """Longest common substring the primary contexts actually carry, per band."""

    buckets: dict[str, list[float]] = {}
    for unit in units:
        window = unit["partner_lcs"][: ch.MIN_CONTEXT_ITEMS]
        if not window:
            continue
        buckets.setdefault(f"{unit['band']}|{unit['stratum']}", []).append(float(max(window)))
    return {
        name: ch.summarise(np.array(values, dtype=np.float64))
        for name, values in sorted(buckets.items())
    }


def _condition_gaps(plan: dict[str, Any]) -> dict[str, Any]:
    """How far each control's context is from the homologue context it matches."""

    gaps: dict[str, Any] = {}
    for condition in ch.CONDITIONS:
        if condition in (ch.HOMOLOGUE, ch.NO_CONTEXT):
            continue
        values = np.array(
            [
                unit["context_tokens"][condition] - unit["context_tokens"][ch.HOMOLOGUE]
                for unit in plan["units"]
            ],
            dtype=np.float64,
        )
        gaps[condition] = ch.summarise(values)
    return gaps


# ------------------------------------------------------------- the self-check


def load_scorable_arm(name: str, *, device: str, dtype: str) -> Arm:
    """A weights-loaded handle that can score a packed row.

    Routes like stage 29 ``_open_scorer`` but returns an :class:`Arm`. Illegal
    dtypes are refused here rather than silently recast.
    """

    if name not in ch.ARMS:
        raise KeyError(f"unknown arm {name!r}; arms are {list(ch.ARMS)}")
    if name in ch.EXCLUDED_ARMS:
        raise ValueError(f"{name} is excluded: {ch.EXCLUDED_ARMS[name]}")
    packing = ch.packing_of(name)
    if name in PANEL:
        return load_arm(name, device=device, dtype=dtype)
    if name in {"progen2-large", "progen2-xlarge", "protgpt3-1.3b"}:
        return load_arm_spec(arm_spec(name), device=device, dtype=dtype)
    if name == "proteinglm-7b-clm":
        from src.capability.models import proteinglm as PGLM

        if dtype != "float32":
            raise ValueError(f"{name}: ProteinGLM scoring is float32-only; got {dtype!r}")
        loaded = PGLM.load_pretrained(arm_spec(name).path, device=device, dtype=dtype)
        return Arm(
            spec=arm_spec(name),
            model=loaded["model"],
            tokenizer=loaded["tokenizer"],
            device=device,
            dtype=dtype,
            serving_provenance={
                "derived": loaded.get("derived"),
                "max_length": loaded.get("max_length"),
            },
        )
    if name == "rita-xl":
        from src.capability.models import rita_fitness as RITA

        if dtype != "float32":
            raise ValueError(f"{name}: RITA scoring is float32-only; got {dtype!r}")
        scorer = RITA.RitaFitnessScorer(
            checkpoint=arm_spec(name).path,
            device=device,
            dtype=dtype,
            batch_size=1,
        )
        return Arm(
            spec=arm_spec(name),
            model=scorer.model,
            tokenizer=scorer.tokenizer,
            device=device,
            dtype=dtype,
            serving_provenance={"rita_scorer": scorer},
        )
    if packing == ch.PACKING_PROGEN3:
        from src.capability.models.progen3 import PROGEN3_DTYPES, load_progen3_as_arm

        if dtype not in PROGEN3_DTYPES:
            raise ValueError(
                f"{name}: ProGen3 has no kernel for {dtype!r}; supported "
                f"{sorted(PROGEN3_DTYPES)}"
            )
        return load_progen3_as_arm(name, device=device, dtype=dtype)
    if packing == ch.PACKING_GALACTICA:
        from src.capability.models import galactica_fitness as G

        if dtype != "float32":
            raise ValueError(f"{name}: Galactica scoring is float32-only; got {dtype!r}")
        loaded = G.load_galactica(name, device=device, dtype=dtype)
        handle = ch.tokenizer_arm(name)
        return Arm(
            spec=handle.spec,
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            device=device,
            dtype=dtype,
            serving_provenance={"joint_tokenisation": loaded.tokenisation},
        )
    if packing == ch.PACKING_INSTRUCT:
        from src.capability.models import instructprotein_fitness as IP

        loaded = IP.load_instructprotein(name, device=device, dtype=dtype)
        handle = ch.tokenizer_arm(name)
        return Arm(
            spec=handle.spec,
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            device=device,
            dtype=dtype,
            serving_provenance={"joint_tokenisation": loaded.tokenisation},
        )
    if packing == ch.PACKING_LLAMA:
        from src.capability.models import joint_lineage as L

        loaded = L.load_rung(name, device=device, dtype=dtype)
        handle = ch.tokenizer_arm(name)
        return Arm(
            spec=handle.spec,
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            device=device,
            dtype=dtype,
            serving_provenance={"joint_tokenisation": loaded.tokenisation},
        )
    raise KeyError(f"{name}: no scorable loader; packing is {packing!r}")


def run_self_check(args: argparse.Namespace) -> dict[str, Any]:
    modality = ch.modality_of(args.arm)
    arm = load_scorable_arm(args.arm, device=args.device, dtype=args.dtype)
    declared = ch.require_position_budget(arm.model.config, arm=args.arm)
    record = ch.self_check_record(modality)
    rendering = ch.rendering_check(arm, modality=modality)
    packed = ch._self_check_packed_row(arm, record, modality=modality)
    target = ch.item_ids(arm, record, modality=modality)
    span0, span1 = ch.target_span(arm, target, record=record)
    context_len = len(packed) - len(ch.row_prefix_ids(arm)) - len(target)
    start = len(ch.row_prefix_ids(arm)) + context_len + span0
    end = start + (span1 - span0)
    logits, scored_ids = _forward_rows(arm, [packed])
    nll = _target_nll(logits[:1], scored_ids[:1], start, end)
    return _header(
        {
            "arm": args.arm,
            "modality": modality,
            "device": args.device,
            "dtype": args.dtype,
            "declared_positions": declared,
            "position_budget": ch.POSITION_BUDGET,
            "rendering": rendering,
            "packing_assertions": rendering.get("packing_assertions", {}),
            "caveats": ch.CAVEATS.get(args.arm),
            "fixed_record_nll_nats_per_token": nll["nats_per_token"],
            "fixed_record_scored_tokens": nll["scored_tokens"],
            "fixed_record": record,
            "passed": True,
        }
    )


def _target_nll(logits: torch.Tensor, ids: torch.Tensor, start: int, end: int) -> dict[str, float]:
    """Mean next-token NLL over the target's own tokens, and nothing else.

    ``start`` is the first scored *token* position, so the prediction of it is
    read from column ``start - 1``. The context's own tokens are never scored:
    the whole contrast is what the context does to the target.
    """

    if start < 1:
        raise ValueError("the first target token has no preceding position to predict it")
    logprobs = F.log_softmax(logits[0, start - 1 : end - 1].float(), dim=-1)
    targets = ids[0, start:end]
    values = -logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return {
        "nats_per_token": float(values.mean()),
        "nll_sum": float(values.sum()),
        "scored_tokens": int(targets.numel()),
    }


# ------------------------------------------------------------------ the score


#: Packings whose scored NLL is not invariant to the batch it was scored in, with
#: the measured size of that dependence. ProGen3's ``SparseMoeBlock`` flattens
#: batch and position into one token axis and runs one gather-and-GEMM per expert
#: over the whole flattened batch, so the accumulation a token's output comes out
#: of depends on how many other tokens routed to its expert; the model has no
#: kernel above float16, so that reduction order reaches the result. Measured in
#: bfloat16 against singleton scoring of the same rows, the maximum absolute
#: difference is 0.02464 nats per token for ProGen3-3B at batch size two and
#: 0.02796 for ProGen3-112M at batch size eight. Eight identical 770-token rows
#: agree exactly with each other while sitting 0.02444 nats per token from the
#: same row scored alone, so this is batch extent rather than batch position, and
#: equal-length rows disagree too, so it is not padding. This stage's endpoint is
#: a paired difference of a few hundredths of a nat per token, so a batched
#: ProGen3 row is not a measurement of this arm and the score path refuses one.
#: ``archive/docs/R5/progen3-stage46-remeasurement.md`` holds the full measurement.
SINGLETON_ONLY_PACKINGS: dict[str, str] = {
    ch.PACKING_PROGEN3: (
        "its expert mixture reduces over the whole flattened batch and the model "
        "has no kernel above float16, so a batched row's scored NLL sits up to "
        "0.0246 (3B, batch two) to 0.0280 (112M, batch eight) nats per token "
        "from its singleton value -- larger than the paired difference this "
        "stage reports"
    )
}


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    packing = ch.packing_of(args.arm)
    if packing in SINGLETON_ONLY_PACKINGS and args.batch_size != 1:
        raise SystemExit(
            f"{args.arm}: --batch-size {args.batch_size} is refused because "
            f"{SINGLETON_ONLY_PACKINGS[packing]}. Score this arm with "
            "--batch-size 1"
        )
    cohort = ch.load_cohort(args.cohort)
    plan = ch.load_plan(args.plan, cohort=cohort, arm=args.arm)
    modality = plan["modality"]
    block = cohort[modality]
    records = block["records"]
    filler = block["filler"]["record"]
    arm = load_scorable_arm(args.arm, device=args.device, dtype=args.dtype)
    ch.require_position_budget(arm.model.config, arm=args.arm)

    rows: list[dict[str, Any]] = []
    drift: list[dict[str, Any]] = []
    units = plan["units"]
    for start_index in range(0, len(units), args.batch_size):
        chunk = units[start_index : start_index + args.batch_size]
        built = []
        for unit in chunk:
            row, span_start = ch.build_row(
                arm,
                unit,
                args.condition,
                records=records,
                filler=filler,
                modality=modality,
            )
            target_record = records[int(unit["target"])]
            target = ch.item_ids(arm, target_record, modality=modality)
            span0, span1 = ch.target_span(arm, target, record=target_record)
            expected = unit["context_tokens"][args.condition]
            observed = span_start - len(ch.row_prefix_ids(arm)) - span0
            if observed != expected:
                drift.append(
                    {"key": unit["key"], "planned": expected, "rebuilt": observed}
                )
            built.append((unit, row, span_start, span_start + (span1 - span0), observed))
        packed = [row for _, row, _, _, _ in built]
        logits, scored_ids = _forward_rows(arm, packed)
        for position, (unit, row, span_start, span_end, context_tokens) in enumerate(built):
            scored = _target_nll(
                logits[position : position + 1],
                scored_ids[position : position + 1],
                span_start,
                span_end,
            )
            rows.append(
                {
                    "key": unit["key"],
                    "band": unit["band"],
                    "stratum": unit["stratum"],
                    "group": unit["group"],
                    "k": unit["k"],
                    "max_lcs": unit["max_lcs"],
                    "shared_kmers": unit["shared_kmers"],
                    "context_tokens": context_tokens,
                    "row_tokens": len(row),
                    **scored,
                }
            )
    if drift:
        raise RuntimeError(
            f"{args.arm}/{args.condition}: {len(drift)} units rebuild to a different "
            f"context token count than the plan recorded, first {drift[0]}. The plan "
            "and the scoring code disagree about what was built, which is refused"
        )
    return _header(
        {
            "arm": args.arm,
            "condition": args.condition,
            "condition_purpose": ch.CONDITION_PURPOSE[args.condition],
            "modality": modality,
            "device": args.device,
            "dtype": args.dtype,
            # Recorded, not reconstructed. The batch size of the withdrawn
            # ProGen3 scores had to be recovered from a queue manifest.
            "batch_size": args.batch_size,
            "cohort_digest": cohort["digest"],
            "plan_digest": plan["digest"],
            "caveats": ch.CAVEATS.get(args.arm),
            "rows": rows,
            "n_units": len(rows),
        }
    )


def _forward_rows(arm: Arm, rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits and id tensor for packed rows. RITA is unpadded; ProGen3 needs MoE ids."""

    packing = ch.packing_of(arm.name)
    lengths = [len(row) for row in rows]
    if packing == ch.PACKING_RITA:
        if len(set(lengths)) != 1:
            raise ValueError(
                f"{arm.name}: RITA refuses padding and attention_mask; unequal "
                f"row lengths {sorted(set(lengths))} need batch_size=1"
            )
        ids = torch.tensor(rows, dtype=torch.long, device=arm.device)
        with torch.no_grad():
            logits = arm.model(input_ids=ids).logits
        return logits, ids
    if packing == ch.PACKING_PROGEN3:
        return _forward_progen3(arm, rows)
    width = max(lengths)
    if packing == ch.PACKING_F16:
        pad = arm.tokenizer.eos_token_id
    else:
        pad = arm.tokenizer.pad_token_id
        if pad is None:
            pad = arm.tokenizer.eos_token_id
    if pad is None:
        raise ValueError(f"{arm.name}: tokenizer has no pad or end-of-text token")
    ids = torch.full((len(rows), width), int(pad), dtype=torch.long, device=arm.device)
    mask = torch.zeros((len(rows), width), dtype=torch.long, device=arm.device)
    for position, row in enumerate(rows):
        ids[position, : len(row)] = torch.tensor(row, dtype=torch.long, device=arm.device)
        mask[position, : len(row)] = 1
    with torch.no_grad():
        logits = arm.model(input_ids=ids, attention_mask=mask).logits
    return logits, ids


def _forward_progen3(arm: Arm, rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed rows through the MoE, at the native independent-sequence ids.

    ``sequence_ids`` indexes a learned embedding that ``ProGen3Model.forward``
    adds to every token embedding, and the native ``ProGen3BatchPreparer``
    assigns zero to every token of an independent sequence and pads with zero.
    Assigning the batch-row index instead -- which this function did until
    2026-09-23 -- adds ``embed_seq_id.weight[i]`` to every token of row ``i``,
    so every row after the first carried an input the native interface never
    produces. That defect is the withdrawn ProGen3 Stage-46 measurement. Zero
    is the value, for content and padding alike, at every batch position.

    Building a native batch is not the same as being able to measure one; what
    this stage's score path may batch is declared in :func:`run_score`.
    """

    from src.capability.models.progen3 import forward

    pg = (arm.serving_provenance or {}).get("progen3")
    if pg is None:
        raise ValueError(
            f"{arm.name}: ProGen3 serving handle is missing; refusing naked "
            "model(input_ids=), which is a silent wrong answer on this MoE"
        )
    specials = ch._progen3_special_ids(arm.tokenizer)
    pad = specials["<pad>"]
    width = max(len(row) for row in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long, device=arm.device)
    position_ids = torch.zeros((len(rows), width), dtype=torch.long, device=arm.device)
    sequence_ids = torch.zeros((len(rows), width), dtype=torch.long, device=arm.device)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=arm.device)
        position_ids[index, : len(row)] = torch.arange(len(row), device=arm.device)
    batch = {
        "input_ids": ids,
        "position_ids": position_ids,
        "sequence_ids": sequence_ids,
    }
    with torch.no_grad():
        logits = forward(pg, batch).logits
    return logits, ids


# ---------------------------------------------------------------- the analysis


def run_analyse(args: argparse.Namespace) -> dict[str, Any]:
    cohort = ch.load_cohort(args.cohort)
    arms_block: dict[str, Any] = {}
    plans = sorted(args.score_dir.glob("plan_*.json"))
    for plan_path in plans:
        arm = plan_path.name[len("plan_") : -len(".json")]
        plan = ch.load_plan(plan_path, cohort=cohort, arm=arm)
        scored: dict[str, dict[str, dict[str, Any]]] = {}
        missing = []
        for condition in ch.CONDITIONS:
            path = args.score_dir / f"scores_{arm}_{condition}.json"
            if not path.is_file():
                missing.append(condition)
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["cohort_digest"] != cohort["digest"] or payload["plan_digest"] != plan["digest"]:
                raise RuntimeError(
                    f"{path} was scored against a different cohort or plan digest"
                )
            scored[condition] = {row["key"]: row for row in payload["rows"]}
        caveats = ch.CAVEATS.get(arm)
        if missing:
            arms_block[arm] = {
                "status": f"conditions not scored: {missing}",
                "caveats": caveats,
            }
            continue
        block = _analyse_arm(arm, plan, scored, resamples=args.bootstrap, seed=args.bootstrap_seed)
        block["caveats"] = caveats
        arms_block[arm] = block
    return _header(
        {
            "cohort_digest": cohort["digest"],
            "bootstrap": {"resamples": args.bootstrap, "seed": args.bootstrap_seed},
            "cohort_census": {
                "protein": cohort["protein"]["census"],
                "text": cohort["text"]["census"],
            },
            "arms": arms_block,
            "never_the_effect": ch.DIAGNOSTIC_NEVER_THE_EFFECT,
        }
    )


def _analyse_arm(
    arm: str,
    plan: dict[str, Any],
    scored: dict[str, dict[str, dict[str, Any]]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for unit in plan["units"]:
        key = unit["key"]
        if any(key not in scored[condition] for condition in ch.CONDITIONS):
            continue
        per_condition = {
            condition: scored[condition][key]["nats_per_token"] for condition in ch.CONDITIONS
        }
        token_counts = {
            condition: scored[condition][key]["scored_tokens"] for condition in ch.CONDITIONS
        }
        if len(set(token_counts.values())) != 1:
            raise RuntimeError(
                f"{arm}: unit {key} was scored on different token counts across "
                f"conditions ({token_counts}); the contrast is not paired"
            )
        statistics = ch.paired_statistics(per_condition)
        rows.append(
            {
                "key": key,
                "band": unit["band"],
                "stratum": unit["stratum"],
                "group": unit["group"],
                "k": unit["k"],
                "max_lcs": unit["max_lcs"],
                "shared_kmers": unit["shared_kmers"],
                "scored_tokens": token_counts[ch.HOMOLOGUE],
                **{condition: per_condition[condition] for condition in ch.CONDITIONS},
                **statistics,
            }
        )
    retained = [row for row in rows if row["stratum"] == "retained"]
    bands: dict[str, Any] = {}
    for band in sorted({row["band"] for row in rows}):
        bands[band] = {}
        for stratum in ch.STRATA:
            subset = [row for row in rows if row["band"] == band and row["stratum"] == stratum]
            if not subset:
                continue
            block = ch.endpoint_block(subset, resamples=resamples, seed=seed)
            if stratum == "retained":
                split = ch.terciles(subset)
                block["terciles"] = {
                    name: {
                        **ch.endpoint_block(
                            [subset[i] for i in indices], resamples=resamples, seed=seed
                        ),
                        "max_lcs_range": (
                            [
                                min(subset[i]["max_lcs"] for i in indices),
                                max(subset[i]["max_lcs"] for i in indices),
                            ]
                            if indices
                            else None
                        ),
                    }
                    for name, indices in split.items()
                }
            bands[band][stratum] = block
    pooled = ch.endpoint_block(retained, resamples=resamples, seed=seed)
    decisive = (
        bands.get(ch.DECISIVE_BAND, {})
        .get("retained", {})
        .get("terciles", {})
        .get(ch.TERCILES[0])
    )
    block = {
        "status": "scored",
        "modality": plan["modality"],
        "plan_digest": plan["digest"],
        "k_distribution": plan["k_distribution"],
        "context_fraction": plan["context_fraction"],
        "unrelated_token_gap": plan["unrelated_token_gap"],
        "refusals": plan["refusals"],
        "rendering": plan["rendering"],
        "n_rows": len(rows),
        "bands": bands,
        "pooled": pooled,
        "decisive_stratum": decisive,
        "occupancy_diagnostic": _occupancy(retained, resamples=resamples, seed=seed),
        "delta_nll_curve": _delta_curve(retained),
        "per_unit": rows,
    }
    block["verdict"] = ch.gate(block)
    return block


def _occupancy(rows, *, resamples: int, seed: int) -> dict[str, Any]:
    """The registration's failure branch: does mere occupancy move the target?

    Needs the k = 0 condition and is the only thing that reads it.
    """

    if not rows:
        return {"n_rows": 0}
    groups = [row["group"] for row in rows]
    occupancy = [row[ch.NO_CONTEXT] - row[ch.POSITION_ONLY] for row in rows]
    homologue = [row[ch.NO_CONTEXT] - row[ch.HOMOLOGUE] for row in rows]
    return {
        "n_rows": len(rows),
        "n_groups": len(set(groups)),
        "note": ch.DIAGNOSTIC_NEVER_THE_EFFECT,
        "no_context_minus_position_only": ch.group_bootstrap_mean(
            occupancy, groups, resamples=resamples, seed=seed
        ),
        "no_context_minus_homologue": ch.group_bootstrap_mean(
            homologue, groups, resamples=resamples, seed=seed
        ),
        "occupancy_share_of_the_homologue_move": (
            float(np.mean(occupancy) / np.mean(homologue))
            if abs(float(np.mean(homologue))) > 1e-12
            else None
        ),
    }


def _delta_curve(rows) -> dict[str, Any]:
    """The descriptive ΔNLL curve. Never differenced across arms (L23)."""

    curve: dict[str, Any] = {
        "unit": "nats per scored token",
        "note": (
            "descriptive only. Reported per arm beside that arm's own symbols per "
            "token and never differenced across arms: a per-token magnitude's "
            "cross-arm sign can reverse under a change of unit"
        ),
        "by_k": {},
        "by_band": {},
    }
    for key, attribute in (("by_k", "k"), ("by_band", "band")):
        buckets: dict[Any, list[float]] = {}
        for row in rows:
            buckets.setdefault(row[attribute], []).append(
                row["delta_nll_homologue_minus_unrelated"]
            )
        curve[key] = {
            str(name): {"n": len(values), "mean_nats_per_token": float(np.mean(values))}
            for name, values in sorted(buckets.items(), key=lambda item: str(item[0]))
        }
    return curve


# ------------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("cohort", "plan", "census", "self-check", "score", "analyse"),
        help="the registration's operational sequence, in order",
    )
    parser.add_argument("--arm", default=None, choices=sorted(ch.ARMS))
    parser.add_argument(
        "--condition",
        default=None,
        choices=list(ch.CONDITIONS),
        help="one condition per scoring invocation, in the frozen order: the two "
        "controls most likely to close the line are measured first",
    )
    parser.add_argument("--cohort", type=Path, default=DEFAULT_OUT / "cohort.json")
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--score-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("bfloat16", "float16", "float32"),
        help="float32 by default and declared as such: the endpoint is a paired "
        "difference of two log-likelihoods a few hundredths of a nat apart, every "
        "arm here is under 800M parameters, and a reduced-precision cast buys "
        "nothing a 143 GB card needs",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="the panel's one declared corpus-draw seed; the campaign manifest pins "
        "it to the registration's frozen value and the cohort digest covers it",
    )
    parser.add_argument("--protein-pool", type=int, default=ch.PROTEIN_POOL)
    parser.add_argument("--text-pool", type=int, default=ch.TEXT_POOL)
    parser.add_argument(
        "--text-examine",
        type=int,
        default=None,
        help="cap on how many passages the BM25 walk examines; the default walks "
        "until every band holds its floor",
    )
    parser.add_argument("--bootstrap", type=int, default=ch.BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=ch.BOOTSTRAP_SEED)
    parser.add_argument("--diamond-tarball", type=Path, default=DIAMOND_TARBALL)
    parser.add_argument("--diamond-checksum", type=Path, default=DIAMOND_CHECKSUM)
    parser.add_argument("--diamond-threads", type=int, default=32)
    parser.add_argument("--rebuild-db", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage in ("plan", "self-check", "score") and args.arm is None:
        raise SystemExit("--arm is required for --stage plan, self-check and score")
    if args.stage == "score":
        if args.condition is None:
            raise SystemExit("--condition is required for --stage score")
        if args.plan is None:
            args.plan = args.out / f"plan_{args.arm}.json"
    if args.stage in ("census", "analyse"):
        if args.score_dir is None:
            args.score_dir = args.out
    if args.stage == "analyse":
        ch.require_frozen_parameters(
            resamples=args.bootstrap,
            bootstrap_seed=args.bootstrap_seed,
            draw_seed=ch.DRAW_SEED,
            position_budget=ch.POSITION_BUDGET,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "cohort":
        args.work.mkdir(parents=True, exist_ok=True)

    if args.stage == "cohort":
        payload, name = run_cohort(args), "cohort.json"
    elif args.stage == "plan":
        payload, name = run_plan(args), f"plan_{args.arm}.json"
    elif args.stage == "census":
        payload, name = run_census(args), "cohort_census.json"
    elif args.stage == "self-check":
        payload, name = run_self_check(args), f"self_check_{args.arm}.json"
    elif args.stage == "score":
        payload, name = run_score(args), f"scores_{args.arm}_{args.condition}.json"
    else:
        payload, name = run_analyse(args), "context_homologue.json"

    destination = args.out / name
    write_json(destination, payload)
    print(f"wrote {destination}")
    if args.stage == "cohort":
        print(f"cohort digest: {payload['digest']}")
        for modality in ("protein", "text"):
            census = payload[modality]["census"]
            for band, strata in census.items():
                if not isinstance(strata, dict):
                    continue
                for stratum, block in strata.items():
                    print(
                        f"  {modality} {band} {stratum}: "
                        f"{block['eligible_targets']} eligible, drawn {block['drawn']}"
                    )
    if args.stage == "plan":
        print(f"plan digest: {payload['digest']}")
        print(f"  k: {payload['k_distribution']}")
        print(f"  refusals: {len(payload['refusals'])}")
    if args.stage == "census":
        print(f"cohort digest: {payload['cohort_digest']}")
        for arm, block in payload["arms"].items():
            print(f"  {arm}: {block.get('digest', block.get('status'))}")
    if args.stage == "self-check":
        print(f"{args.arm}: {payload['fixed_record_nll_nats_per_token']:.4f} nats/token")
    if args.stage == "analyse":
        for arm, block in payload["arms"].items():
            if block.get("status") != "scored":
                print(f"{arm}: {block.get('status')}")
                continue
            print(f"{arm}: {block['verdict']['outcome']}")


if __name__ == "__main__":
    main()
