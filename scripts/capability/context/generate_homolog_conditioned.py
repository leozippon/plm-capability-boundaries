#!/usr/bin/env python3
"""E11: does an external homologue in the context improve what a decoder generates?

Four conditions per target, built from the same retrieval artefact E09 scores
against and at the same item count and per-item token length: no context, a
matched-unrelated context, a close-homologue context and a remote-homologue
context. Every attempt carries its own copying statistics against the very items
it was prompted with, and yield is reported twice -- before and after copies are
excluded -- on one denominator: every attempt made.

The reason copying is the first endpoint rather than a caveat
============================================================

The obvious way for a close homologue in the prompt to "improve quality" is for
the decoder to copy it. A product that is a copy of its prompt is not a designed
protein, and a yield that rises because copies rose is not an improvement. So the
longest verbatim run and the k-mer containment against each context item are
measured here, for every attempt, and the alignment identity -- which only an
aligner can supply -- is filled in by the retrieval stage's ``annotate`` phase.
Until it is, the identity rule abstains and says so rather than passing.

Scope, stated rather than implied
=================================

Only arms whose context packing is the self-delimiting document stream are
admitted, because a conditioned prompt in those arms is exactly what the
checkpoint saw in pretraining: items concatenated, then the marker that starts the
next one. An arm with instruction slots, direction markers or a mask protocol
would need its own declaration of what a conditioned prompt means, and borrowing
another arm's is how a number gets produced behind a rendering no checkpoint was
trained on.

The sampling operating point is the unconditional generation campaign's own
(:data:`~src.capability.generation.unconditional_generation.POLICY`), so the
conditions differ from it in the prompt and nothing else. The attempt count is
this experiment's own and is **not** the campaign's 800, so these yields are not
poolable with the frozen generation results.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402

from scripts.capability.entrypoints import stage_path  # noqa: E402
from src.capability.context import context_homologue as ch  # noqa: E402
from src.capability.context import homology_context as H  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation import unconditional_generation as ug  # noqa: E402

EXPECT = "homolog_conditioned_generation.json"
PRODUCTS = "products.jsonl"

#: The conditions of this experiment, each naming a declared E09 condition so the
#: two experiments are built from one retrieval artefact and one set of edges.
GENERATION_CONDITIONS: dict[str, str] = {
    "no_context": H.NO_CONTEXT,
    "unrelated": H.UNRELATED,
    "close_homolog": "id_70_90",
    "remote_homolog": "id_30_50",
}

#: The homologue conditions a target must supply to enter the panel. The two
#: controls are required of every target, so the panel is complete-case by
#: construction and the four conditions rest on the same proteins.
REQUIRED_BINS: tuple[str, ...] = ("id_70_90", "id_30_50")

#: Arms whose conditioned prompt is a plain document stream. Anything else is
#: refused by name with the reason, rather than rendered by analogy.
ADMITTED_PACKINGS: tuple[str, ...] = (ch.PACKING_F16,)

#: The token that closes a *document* in each input format's pretraining stream,
#: for formats whose records are not self-delimiting at their start. Declared by
#: name and round-trip checked, never taken from ``tokenizer.eos_token_id``: see
#: :func:`terminator_ids` for the measured reason.
DOCUMENT_END: dict[str, str] = {"n_to_c_control": "<|eos|>"}


def runtime(device: str) -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "device": device,
        "peak_rss_bytes": usage.ru_maxrss * 1024,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "threads_requested": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def stage_module(filename: str):
    spec = importlib.util.spec_from_file_location(filename, stage_path(ROOT, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def terminator_ids(arm) -> dict[str, int]:
    """The ids that close a record and a document for this arm, round-trip checked.

    Resolved through the tokeniser and then *verified* by decoding back to the
    declared token string. The check is here because the obvious spelling is
    wrong in a way nothing downstream would reveal: ids 2 and 4 of ProGen2's
    vocabulary both look plausible, one decodes to ``<|eos|>`` and the other to
    the ``2`` direction marker, and appending the wrong one produces a prompt the
    checkpoint answers with an immediate terminator and no protein at all.
    """

    close = ug.close_token_for(arm.name)
    close_id = arm.tokenizer.convert_tokens_to_ids(close)
    if close_id is None or int(close_id) < 0 or arm.tokenizer.decode([int(close_id)]) != close:
        raise SystemExit(
            f"{arm.name}: the declared close token {close!r} does not round-trip through "
            f"the tokeniser (id {close_id!r}); a conditioned prompt built on it would "
            "not be the stream this checkpoint was trained on"
        )
    document = None
    expected = DOCUMENT_END.get(arm.spec.input_format)
    if expected is not None:
        candidate = arm.tokenizer.convert_tokens_to_ids(expected)
        if (
            candidate is None
            or int(candidate) < 0
            or arm.tokenizer.decode([int(candidate)]) != expected
        ):
            raise SystemExit(
                f"{arm.name}: the document-end token {expected!r} does not round-trip "
                f"through the tokeniser (id {candidate!r})"
            )
        document = int(candidate)
        # ``tokenizer.eos_token_id`` is deliberately not used: ProGen2's tokenizer
        # config declares '<|endoftext|>' (id 30) as its end-of-sequence token
        # while the stream it was pretrained on closes a record with '<|eos|>'
        # (id 2), and appending the former produces a prompt the checkpoint
        # answers with a terminator and no protein.
    return {
        "close": int(close_id),
        "document_end": document,
        "document_end_token": expected,
        "tokenizer_eos_token_id": (
            None if arm.tokenizer.eos_token_id is None else int(arm.tokenizer.eos_token_id)
        ),
    }


def generation_item_ids(arm, record: str, *, ids: dict[str, int]) -> list[int]:
    """One context item as a **complete** native record, which generation needs.

    The scoring packing deliberately leaves an item unterminated: the next item's
    own marker is the boundary, the scored span never includes a marker, and a
    terminator would only add a token. For generation the difference decides
    whether anything is generated at all, so the grammar was qualified against the
    checkpoint rather than assumed. Measured on ProGen2-medium, three relatives in
    the prompt and eight samples per form:

    =========================================  ==========================
    context item rendering                     attempts yielding >= 32 aa
    =========================================  ==========================
    ``1 SEQ`` (the scoring packing)            0 of 8
    ``1 SEQ 2``                                0 of 8
    ``1 SEQ 2 <|eos|>``                        8 of 8
    ``<|bos|> 1 SEQ 2 <|eos|>``                8 of 8
    =========================================  ==========================

    An unterminated or half-terminated record leaves the checkpoint mid-document,
    so it emits its own terminator instead of starting a new sequence. The
    complete record is therefore ``direction + residues + close + document end``,
    and the minimal form that works is used rather than the one that also adds a
    beginning-of-sequence token the item does not need.

    ``fasta_wrapped`` items are self-delimiting at their *start* -- each begins
    with the end-of-text token, which is the record boundary in ProtGPT2's own
    pretraining stream -- so nothing is appended and a terminator would double the
    separator. Any other format is refused rather than rendered by analogy.
    """

    rendered = ch.item_ids(arm, record, modality="protein")
    fmt = arm.spec.input_format
    if fmt == "n_to_c_control":
        if ids["document_end"] is None:
            raise SystemExit(f"{arm.name}: no document-end token to close a context record")
        return list(rendered) + [ids["close"], ids["document_end"]]
    if fmt == "fasta_wrapped":
        return list(rendered)
    raise SystemExit(
        f"{arm.name}: no declared complete-record grammar for input format {fmt!r}; "
        "a conditioned prompt cannot be rendered by analogy with another arm"
    )


def require_arm(arm: str) -> None:
    packing = ch.packing_of(arm)
    if packing not in ADMITTED_PACKINGS:
        raise SystemExit(
            f"{arm} packs context as {packing!r}; this stage admits only "
            f"{list(ADMITTED_PACKINGS)}, because a conditioned prompt in any other "
            "packing needs its own declaration of what the prompt means"
        )
    ug.require_admitted(arm)


def filtered_scores(logits, *, temperature, top_k, top_p):
    """The campaign's next-token filter, vectorised over a batch of rows.

    Statement for statement the same rule as
    :func:`~src.capability.generation.unconditional_generation._sample_next_id`
    -- temperature, then top-k, then nucleus with the shifted-mask convention --
    applied to a whole batch at once. The reference is a per-row function and
    calling it row by row costs one device synchronisation per row per step,
    which on a four-condition panel is the dominant cost of the whole stage.
    Agreement with the reference is not assumed: :func:`check_sampler` runs the
    reference against this filter at the start of every run and refuses to
    continue if a reference draw falls outside this filter's support.
    """

    import torch

    scores = logits / float(temperature)
    if top_k > 0:
        kept = min(int(top_k), int(scores.size(-1)))
        threshold = torch.topk(scores, kept, dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < threshold, float("-inf"))
    if 0.0 < top_p < 1.0:
        ranked, order = torch.sort(scores, descending=True, dim=-1)
        cumulative = torch.cumsum(torch.softmax(ranked, dim=-1), dim=-1)
        drop = cumulative > top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        ranked = ranked.masked_fill(drop, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(-1, order, ranked)
    return scores


def check_sampler(logits, *, draws: int = 512) -> dict:
    """Every draw of the campaign's own sampler must lie in this filter's support.

    A one-sided containment check, which is the direction that matters: a
    vectorised filter that admitted fewer tokens than the reference would change
    the distribution these products come from, and this run would be sampling
    from a rule no declaration describes.
    """

    import torch

    row = logits[0]
    support = torch.isfinite(
        filtered_scores(
            row,
            temperature=ug.POLICY["temperature"],
            top_k=ug.POLICY["top_k"],
            top_p=ug.POLICY["top_p"],
        )
    )
    drawn = set()
    for index in range(draws):
        torch.manual_seed(H.DRAW_SEED + index)
        drawn.add(
            int(
                ug._sample_next_id(
                    row,
                    temperature=ug.POLICY["temperature"],
                    top_k=ug.POLICY["top_k"],
                    top_p=ug.POLICY["top_p"],
                ).item()
            )
        )
    outside = sorted(token for token in drawn if not bool(support[token]))
    if outside:
        raise RuntimeError(
            f"the campaign's sampler drew tokens {outside} that this run's vectorised "
            "filter excludes; the two rules disagree and no product may be recorded"
        )
    return {
        "reference": "unconditional_generation._sample_next_id",
        "reference_draws": draws,
        "distinct_reference_tokens": len(drawn),
        "vectorised_support": int(support.sum()),
        "all_reference_draws_inside_support": True,
    }


def sample_batch(model, tokenizer, prompt_ids, *, count, seed, max_new_tokens, stops, batch):
    """Continuations of one fixed prompt, in batches, at the campaign's operating point.

    Decoded through ``forward`` with the model's own key-value cache rather than
    through ``generate``. Not a preference: ``generate`` in Transformers 4.57
    builds its cache from ``config.num_hidden_layers``, which ProGen2's config does
    not declare, so the call raises before sampling a token. Disabling the cache
    would make it run and cost a quadratic factor; feeding ``past_key_values``
    forward is exact and fast, and was verified against this checkpoint.

    The next-token rule itself is the campaign's own
    (:func:`~src.capability.generation.unconditional_generation._sample_next_id`),
    reused rather than restated so that these products and the unconditional ones
    differ in their prompt and in nothing else. It is a per-row rule, so it is
    called once per row of the batch; the cost is negligible beside the forward.
    """

    import torch

    outputs: list[str] = []
    index = 0
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    while len(outputs) < count:
        size = min(batch, count - len(outputs))
        torch.manual_seed(seed + index)
        rows: list[list[int]] = [[] for _ in range(size)]
        live = torch.ones(size, dtype=torch.bool, device=prompt.device)
        with torch.no_grad():
            step = model(input_ids=prompt.repeat(size, 1), use_cache=True)
            past = step.past_key_values
            logits = step.logits[:, -1]
            for _ in range(max_new_tokens):
                scores = filtered_scores(
                    logits,
                    temperature=ug.POLICY["temperature"],
                    top_k=ug.POLICY["top_k"],
                    top_p=ug.POLICY["top_p"],
                )
                drawn = torch.multinomial(torch.softmax(scores, dim=-1), 1)
                tokens = drawn[:, 0].tolist()
                still = live.tolist()
                for row, token in enumerate(tokens):
                    if not still[row]:
                        continue
                    rows[row].append(int(token))
                    if int(token) in stops:
                        live[row] = False
                if not bool(live.any()):
                    break
                step = model(input_ids=drawn, past_key_values=past, use_cache=True)
                past = step.past_key_values
                logits = step.logits[:, -1]
        outputs.extend(tokenizer.decode(row, skip_special_tokens=False) for row in rows)
        index += 1
    return outputs


def run(args: argparse.Namespace) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    contexts = json.loads(args.homologs.read_text(encoding="utf-8"))
    H.require_declaration(contexts, scope="context")
    require_arm(args.arm)

    scorer = stage_module("score_context_identity.py")
    stage = stage_module("context_homologue.py")
    arm = stage.load_scorable_arm(args.arm, device=args.device, dtype=args.dtype)
    ch.require_position_budget(arm.model.config, arm=args.arm)
    plans, support = scorer.build_plans(arm, contexts, budget=args.budget)

    statuses = {
        identifier: {name: record["status"] for name, record in plan["conditions"].items()}
        for identifier, plan in plans.items()
    }
    panel = H.balanced_targets(statuses, bins=REQUIRED_BINS)
    if args.target_limit:
        panel = panel[: args.target_limit]
    clusters = {plans[identifier]["cluster"] for identifier in panel}
    if not panel:
        raise RuntimeError("no target supplies both homologue conditions and both controls")

    offset = ch.content_offset(arm)
    if offset < 1:
        raise SystemExit(f"{args.arm}: no item-start marker to prompt a new sequence with")
    markers = terminator_ids(arm)
    stops = {markers["close"]} | (
        {markers["document_end"]} if markers["document_end"] is not None else set()
    )
    row_prefix = ch.row_prefix_ids(arm)

    args.out.mkdir(parents=True, exist_ok=True)
    products_path = args.out / PRODUCTS
    attempts: list[dict] = []
    sampler_check = None
    started = time.monotonic()
    for number, identifier in enumerate(panel, start=1):
        plan = plans[identifier]
        target = next(row for row in contexts["targets"] if row["target_id"] == identifier)
        start_marker = ch.item_ids(arm, target["wildtype"], modality="protein")[:offset]
        for label, condition in GENERATION_CONDITIONS.items():
            items = plan["items"][condition]
            context_ids: list[int] = []
            for item in items:
                context_ids.extend(generation_item_ids(arm, item["sequence"], ids=markers))
            prompt_ids = row_prefix + context_ids + list(start_marker)
            headroom = args.budget - len(prompt_ids)
            if headroom < H.MIN_PRODUCT_RESIDUES:
                raise RuntimeError(
                    f"{identifier} {label}: {headroom} positions left for a product"
                )
            max_new = min(ug.POLICY["max_new_tokens"], headroom)
            if sampler_check is None:
                with torch.no_grad():
                    probe = arm.model(
                        input_ids=torch.tensor([prompt_ids], device=arm.model.device),
                        use_cache=False,
                    ).logits[:, -1]
                sampler_check = check_sampler(probe)
            raws = sample_batch(
                arm.model,
                arm.tokenizer,
                prompt_ids,
                count=args.attempts,
                seed=H.DRAW_SEED + number,
                max_new_tokens=max_new,
                stops=stops,
                batch=args.batch_size,
            )
            context_sequences = [item["sequence"] for item in items]
            subjects = [item["subject"] for item in items]
            for index, raw in enumerate(raws):
                residues = ug.extract_residues(args.arm, raw)
                statistics = H.copy_statistics(residues, context_sequences)
                attempts.append(
                    {
                        "attempt_id": f"{args.arm}|{identifier}|{label}|{index:04d}",
                        "raw_continuation": raw,
                        "arm": args.arm,
                        "target_id": identifier,
                        "cluster": plan["cluster"],
                        "condition": label,
                        "declared_condition": condition,
                        "attempt_index": index,
                        "sequence": residues,
                        "residues": len(residues),
                        "stop_status": ug.stop_status(args.arm, raw, residues),
                        "prompt_tokens": len(prompt_ids),
                        "max_new_tokens": max_new,
                        "context_items": len(items),
                        "context_subjects": subjects,
                        "context_identities": plan["conditions"][condition]["identities"],
                        "copy_statistics": statistics,
                        "copy_verdict": H.copy_verdict(statistics),
                        "lcs_to_target_wildtype": (
                            H.max_lcs(residues, [target["wildtype"]])[0] if residues else 0
                        ),
                    }
                )
        print(
            f"{number}/{len(panel)} {identifier}: "
            f"{args.attempts * len(GENERATION_CONDITIONS)} attempts",
            flush=True,
        )

    # The structural selection is length-matched across conditions, so it can only
    # be made once every attempt exists. It is recorded on the product itself, so
    # the folding instrument needs no rule of its own.
    selected, selection = H.select_structure_products(
        attempts, conditions=list(GENERATION_CONDITIONS)
    )
    with products_path.open("w", encoding="utf-8") as handle:
        for row in attempts:
            row["structure_selected"] = row["attempt_id"] in selected
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    write_json(
        args.out / EXPECT,
        {
            "schema_version": H.SCHEMA_VERSION,
            "stage": "generate_homolog_conditioned",
            "status": "complete",
            "arm": args.arm,
            "device": args.device,
            "dtype": args.dtype,
            **H.declaration_digests(),
            "declaration": H.declaration(),
            "homologs_sha256": sha256_file(args.homologs),
            "corpus_identity": contexts["corpus_identity"],
            "caveat": ch.CAVEATS.get(args.arm),
            "conditions": GENERATION_CONDITIONS,
            "required_bins": list(REQUIRED_BINS),
            "record_grammar": {
                "input_format": arm.spec.input_format,
                "terminators": markers,
                "qualified": (
                    "the complete-record form was measured against this checkpoint; see "
                    "generation_item_ids"
                ),
            },
            "sampling": {
                "temperature": ug.POLICY["temperature"],
                "top_p": ug.POLICY["top_p"],
                "top_k": ug.POLICY["top_k"],
                "max_new_tokens": ug.POLICY["max_new_tokens"],
                "attempts_per_cell": args.attempts,
                "seed_rule": "DRAW_SEED + target index, then + batch index",
                "draw_seed": H.DRAW_SEED,
                "vectorised_filter_agreement": sampler_check,
                "not_poolable_with": (
                    "the frozen generation campaign's 800-attempt cells; the attempt "
                    "count, the prompt and the seeds all differ"
                ),
            },
            "support": {
                "targets": len(panel),
                "clusters": len(clusters),
                "group_floor": H.GROUP_FLOOR,
                "clears_group_floor": len(clusters) >= H.GROUP_FLOOR,
                "attempts": len(attempts),
                "attempts_per_condition": {
                    label: sum(1 for row in attempts if row["condition"] == label)
                    for label in GENERATION_CONDITIONS
                },
                "plan_support": support,
                "target_ids": panel,
            },
            "yields": yield_table(attempts),
            "structure_selection": selection,
            "products": PRODUCTS,
            "products_sha256": sha256_file(products_path),
            "annotation_pending": (
                "the alignment-identity copying rule and the corpus-novelty endpoint "
                "need an aligner: run retrieve_homology_context.py annotate on "
                f"{PRODUCTS} to fill them in"
            ),
            "structure_pending": (
                "predicted confidence needs the structure instrument: run "
                f"fold_conditioned_products.py on the selected products of {PRODUCTS}"
            ),
            "elapsed_seconds": time.monotonic() - started,
            "runtime": runtime(args.device),
        },
    )
    print(json.dumps(yield_table(attempts), indent=2), flush=True)


def yield_table(attempts: list[dict]) -> dict:
    """Yields on one denominator -- every attempt -- before and after copy exclusion."""

    out = {}
    for label in GENERATION_CONDITIONS:
        rows = [row for row in attempts if row["condition"] == label]
        if not rows:
            continue
        complete = [
            row
            for row in rows
            if row["stop_status"] == "native_terminal" and row["residues"] >= H.MIN_PRODUCT_RESIDUES
        ]
        copies = [row for row in complete if row["copy_verdict"]["is_copy"]]
        lengths = [row["residues"] for row in rows if row["residues"]]
        overlaps = [row["copy_statistics"]["max_lcs_to_context"] for row in rows]
        out[label] = {
            "attempts": len(rows),
            "native_terminal": sum(1 for row in rows if row["stop_status"] == "native_terminal"),
            "budget_censored": sum(1 for row in rows if row["stop_status"] == "budget_censored"),
            "empty_or_noncanonical": sum(
                1 for row in rows if row["stop_status"] == "empty_or_noncanonical"
            ),
            "complete_products": len(complete),
            "complete_yield": len(complete) / len(rows),
            "copies_among_complete": len(copies),
            "copy_rate_among_complete": (len(copies) / len(complete)) if complete else None,
            "non_copy_complete_products": len(complete) - len(copies),
            "non_copy_complete_yield": (len(complete) - len(copies)) / len(rows),
            "copy_rules_fired": {
                rule: sum(1 for row in rows if row["copy_verdict"]["rules"].get(rule) is True)
                for rule in ("long_verbatim_run", "kmer_containment")
            },
            "identity_rule_evaluated": any(
                row["copy_verdict"]["identity_rule_evaluated"] for row in rows
            ),
            "median_residues": float(np.median(lengths)) if lengths else None,
            "median_max_lcs_to_context": float(np.median(overlaps)) if overlaps else None,
            "max_max_lcs_to_context": int(max(overlaps)) if overlaps else None,
            "mean_kmer_containment": float(
                np.mean([row["copy_statistics"]["max_kmer_containment"] for row in rows])
            ),
            "denominator": "every attempt made in this cell",
        }
    return out


EXPECT_ANALYSE = "conditioned_generation_comparison.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def analyse(args: argparse.Namespace) -> None:
    """Join products, annotations and structures into the comparison E11 asks for.

    Four readings, each on a stated denominator: yield before and after copy
    exclusion, family recognition of the product, novelty against the corpus, and
    predicted confidence *within a length band at an equal count per condition*.
    The last qualification is load-bearing: predicted confidence rises with length
    and completeness, so a condition that generates longer products would win on it
    without the homology in its context carrying any information at all.
    """

    products = read_jsonl(args.products)
    if not products:
        raise SystemExit(f"{args.products} holds no product")
    annotations: dict[str, dict] = {}
    annotation_record = None
    if args.annotations is not None:
        annotation_record = json.loads(args.annotations.read_text(encoding="utf-8"))
        H.require_declaration(annotation_record, scope="context")
        for row in annotation_record["annotations"]:
            if row.get("attempt_id"):
                annotations[row["attempt_id"]] = row
    structures: dict[str, dict] = {}
    structure_record = None
    if args.structures is not None:
        structure_record = json.loads(args.structures.read_text(encoding="utf-8"))
        for row in read_jsonl(args.structures.parent / structure_record["records"]):
            structures[row["attempt_id"]] = row

    for row in products:
        annotation = annotations.get(row["attempt_id"])
        identity = None if annotation is None else annotation.get("context_alignment_identity")
        row["copy_verdict"] = H.copy_verdict(row["copy_statistics"], identity_percent=identity)
        row["prompt_family_recognised"] = (
            None if annotation is None else annotation["prompt_family_recognised"]
        )
        row["corpus_max_identity"] = (
            None if annotation is None else annotation.get("corpus_max_identity")
        )
        structure = structures.get(row["attempt_id"])
        row["plddt"] = None if structure is None else structure["plddt"]
        row["ptm"] = None if structure is None else structure["ptm"]

    conditions = sorted({row["condition"] for row in products})
    table = {}
    for condition in conditions:
        rows = [row for row in products if row["condition"] == condition]
        complete = [
            row
            for row in rows
            if row["stop_status"] == "native_terminal" and row["residues"] >= H.MIN_PRODUCT_RESIDUES
        ]
        copies = [row for row in complete if row["copy_verdict"]["is_copy"]]
        clean = [row for row in complete if not row["copy_verdict"]["is_copy"]]
        recognised = [row for row in clean if row["prompt_family_recognised"] is True]
        novelty = [
            row["corpus_max_identity"] for row in clean if row["corpus_max_identity"] is not None
        ]
        table[condition] = {
            "attempts": len(rows),
            "complete_products": len(complete),
            "complete_yield": len(complete) / len(rows),
            "copies": len(copies),
            "non_copy_complete_yield": len(clean) / len(rows),
            "copy_rules_fired": {
                rule: sum(1 for row in rows if row["copy_verdict"]["rules"].get(rule) is True)
                for rule in ("long_verbatim_run", "kmer_containment", "alignment_identity")
            },
            "identity_rule_evaluated": bool(annotations),
            "family_recognised_among_non_copies": len(recognised),
            "family_recognition_rate_non_copy": (len(recognised) / len(clean)) if clean else None,
            "family_recognition_rate_all_attempts": len(recognised) / len(rows),
            "corpus_max_identity_median": float(np.median(novelty)) if novelty else None,
            "median_residues_complete": (
                float(np.median([row["residues"] for row in complete])) if complete else None
            ),
            "denominator": "every attempt made in this cell",
        }

    bands = H.structure_comparison(products, conditions=conditions)

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out / EXPECT_ANALYSE,
        {
            "schema_version": H.SCHEMA_VERSION,
            "stage": "analyse",
            "status": "complete",
            **H.declaration_digests(),
            "declaration": H.declaration(),
            "products": str(args.products),
            "products_sha256": sha256_file(args.products),
            "annotations": None if args.annotations is None else str(args.annotations),
            "structures": None if args.structures is None else str(args.structures),
            "structure_instrument": None
            if structure_record is None
            else structure_record.get("instrument"),
            "corpus_identity": None
            if annotation_record is None
            else annotation_record.get("corpus_identity"),
            "conditions": conditions,
            "yields": table,
            "structure_by_length_band": bands,
            "structure_comparison_rule": H.STRUCTURE_COMPARISON_RULE,
            "pending": {
                "alignment_identity_copy_rule": args.annotations is None,
                "corpus_novelty": args.annotations is None,
                "predicted_structure": args.structures is None,
            },
            "limitations": list(H.LIMITATIONS),
            "runtime": runtime("cpu"),
        },
    )
    print(json.dumps(table, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["generate", "analyse"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arm")
    parser.add_argument("--homologs", type=Path)
    parser.add_argument("--attempts", type=int, default=128)
    parser.add_argument("--budget", type=int, default=H.POSITION_BUDGET)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--target-limit", type=int, default=0)
    parser.add_argument("--products", type=Path, help="analyse phase: the products JSONL")
    parser.add_argument("--annotations", type=Path, help="analyse phase: generation_annotation.json")
    parser.add_argument("--structures", type=Path, help="analyse phase: conditioned_structure.json")
    args = parser.parse_args()
    if args.budget != H.POSITION_BUDGET:
        raise SystemExit(f"the declaration fixes the position budget at {H.POSITION_BUDGET}")
    if args.phase == "generate":
        if args.arm is None or args.homologs is None:
            raise SystemExit("--arm and --homologs are required to generate")
        if args.attempts < 1:
            raise SystemExit("attempts must be positive")
        run(args)
    else:
        if args.products is None:
            raise SystemExit("--products is required for the analyse phase")
        analyse(args)


if __name__ == "__main__":
    main()
