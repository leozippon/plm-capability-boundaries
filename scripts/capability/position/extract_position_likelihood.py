#!/usr/bin/env python3
"""Retain the per-position log likelihood of one checkpoint over a DMS cohort.

No per-token likelihood array survives anywhere in this project: every stage that
formed one summed it in the expression that built it, and only the scalar was
archived. This stage is the producer for the array itself, under the arm's own
rendering and the arm's own reduction, so that a position-resolved reading is a
finer view of the published scalar rather than a second measurement of it.

For every assay it writes one compressed archive holding, for the wild type and
for each variant in the cohort's own order, the per-scored-token negative log
likelihood in nats, the packed token ids, the scored span, the per-token residue
counts and the residue offset -- the layout
``src.capability.extensions.responses.RetainedResponses`` and
``scripts.capability.position.analyse_position_terms.Retained`` already validate,
declared in ``source-recovery-requirements.json`` before a producer for it
existed. Two invariants are enforced at write time rather than checked later:

* the retained vector's own float32 reduction equals the scalar the arm's score
  path produced in the same forward, identically and not within a tolerance;
* for a causal arm, the terms upstream of a substitution are bit-identical
  between the two states, because the prefix that predicts them is untouched.

It also retains, for the wild type only, the arm's log probability over the
resolved single-residue alphabet at every position whose scored token covers one
residue. That is the one GPU product the contact-anticipation question needs, and
retaining it here keeps that question a pure CPU analysis afterwards.

Handoff. The frozen CPU pipeline reads a packing sidecar rather than an archive's
own ids, so ``archive-input.json`` is written in the shape that pipeline accepts
and the completion record names the two commands that consume it. The sidecar is
deliberately not written here: re-deriving it with
``analyse_residue_responses.py --mode packing`` loads no model and checks a fresh
packing against the counts this stage retained, which is a verification rather
than a copy.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.position.position_likelihood import (  # noqa: E402
    CAUSAL,
    MASKED,
    PREFIX_INVARIANT_RULE,
    REFUSED_ARMS,
    SCHEMA,
    TERM_SEMANTICS,
    TIER2_NATS,
    TIER2_REPEAT_MULTIPLE,
    PositionArchive,
    available_arms,
    default_dtype,
    load_position_scorer,
    paradigm_of,
    repeat_residual,
    residual_tier,
    upstream_invariance,
)
from src.capability.position.position_terms import (  # noqa: E402
    ALIGNMENT,
    alignment,
    blas_pinning,
    residue_bounds,
)

#: The completion record. Its presence is the campaign's only completion test, so
#: it is the last file this stage writes.
COMPLETION = "position_likelihood.json"

#: Packed-token ceiling. The campaign that declared this panel's scoring imposed a
#: 1024-position budget on every arm, and an arm scored past it would not be
#: comparable with any published number on the same cohort.
TOKEN_BUDGET = 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def assay_filename(assay: str) -> str:
    return "position_" + hashlib.sha256(assay.encode()).hexdigest()[:20] + ".npz"


def select_assays(cohort, args, site_assays) -> tuple[list[dict], list[dict]]:
    """The assays this cell scores, with every exclusion reported, not silent."""

    rows = {row["assay"]: row for row in cohort["assays"]}
    if len(rows) != len(cohort["assays"]):
        raise SystemExit("the cohort carries a duplicate assay identity")
    requested = list(args.assays) if args.assays else sorted(rows)
    missing = [name for name in requested if name not in rows]
    if missing:
        raise SystemExit(f"the cohort has no assay {missing!r}")
    selected, excluded = [], []
    for name in requested:
        row = rows[name]
        length = len(row["wildtype"])
        if args.structure_only and site_assays is not None and name not in site_assays:
            excluded.append({"assay": name, "reason": "no admitted structural mapping"})
            continue
        if length > args.max_residues:
            excluded.append(
                {"assay": name, "reason": f"{length} residues exceeds --max-residues"}
            )
            continue
        if length < args.min_residues:
            excluded.append(
                {"assay": name, "reason": f"{length} residues is below --min-residues"}
            )
            continue
        if set(row["wildtype"]) - set(AA20):
            excluded.append({"assay": name, "reason": "wild type carries a non-AA20 symbol"})
            continue
        selected.append(row)
    selected.sort(key=lambda row: -len(row["wildtype"]) * (1 + len(row["mutants"])))
    shard = [row for index, row in enumerate(selected) if index % args.shards == args.shard]
    return shard, excluded


def substitution_sites(mutant: str, wildtype: str) -> list[int]:
    sites = []
    for token in mutant.split(":"):
        before, after = token[0], token[-1]
        position = int(token[1:-1]) - 1
        if not 0 <= position < len(wildtype) or wildtype[position] != before or after not in AA20:
            raise SystemExit(f"{mutant}: {token} disagrees with its own wild type")
        sites.append(position)
    if len(set(sites)) != len(sites):
        raise SystemExit(f"{mutant}: repeats a substituted position")
    return sorted(sites)


def conditional_block(state, score, slots) -> dict:
    """Wild-type residue conditionals, where a scored token covers one residue.

    A residue qualifies only when its scored token carries one unambiguous id
    across every state of this assay, which is the condition under which a
    vocabulary column is the model's probability of that residue at that
    position. A byte-pair interface whose segmentation depends on a residue's
    neighbours fails it, and the block is then absent with the reason recorded
    instead of being filled with a column that means something else.
    """

    if not slots:
        return {
            "available": False,
            "reason": (
                "no unambiguous single-residue token alphabet: this arm's segmentation "
                "of a residue depends on its neighbours, so a per-residue conditional "
                "is undefined under its own rendering"
            ),
        }
    residues = [residue for residue in AA20 if residue in slots]
    if score.gathered is None:
        raise SystemExit("the wild-type state was scored without its conditional columns")
    starts, ends = residue_bounds(state.counts, state.offset)
    keep = [local for local in range(len(state.counts)) if ends[local] - starts[local] == 1]
    if not keep:
        return {"available": False, "reason": "no scored token covers exactly one residue"}
    return {
        "available": True,
        "residues": residues,
        "positions": np.asarray([int(starts[local]) for local in keep], dtype=np.int64),
        "logprobs": np.asarray(score.gathered[keep], dtype=np.float32),
        "resolved_residues": len(residues),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, help=f"one of {list(available_arms())}")
    parser.add_argument("--cohort", type=Path, required=True, help="the frozen DMS cohort JSON")
    parser.add_argument("--structures", type=Path, help="structural site table, for --structure-only")
    parser.add_argument("--structure-only", action="store_true",
                        help="score only assays with an admitted structural mapping")
    parser.add_argument("--assays", nargs="+", default=(), help="explicit assay identities")
    parser.add_argument("--max-residues", type=int, default=1000)
    parser.add_argument("--min-residues", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=TOKEN_BUDGET)
    parser.add_argument("--dtype", default="", help="default: the arm's own admitted precision")
    parser.add_argument("--mask-batch-size", type=int, default=16,
                        help="masked arms only: how many masked copies of one sequence share a "
                             "forward; a causal arm is always scored one row per forward, because "
                             "batch extent changes the fp32 reduction and the shared-prefix "
                             "invariant would stop being exact")
    parser.add_argument("--model-root", type=Path, help="checkpoint root, bidirectional arms only")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--no-conditionals", action="store_true",
                        help="skip the wild-type residue conditionals the E02 analysis reads")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not 0 <= args.shard < args.shards:
        parser.error("--shard must be inside --shards")
    if args.mask_batch_size < 1 or args.max_tokens < 2 or args.min_residues < 1:
        parser.error("--mask-batch-size, --max-tokens and --min-residues are positive")
    if args.structure_only and args.structures is None:
        parser.error("--structure-only needs --structures")
    if args.arm in REFUSED_ARMS:
        parser.error(f"{args.arm} is excluded from position-resolved work: {REFUSED_ARMS[args.arm]}")
    paradigm = paradigm_of(args.arm)
    if paradigm == MASKED and args.model_root is None:
        parser.error(f"{args.arm} is a bidirectional arm and needs --model-root")

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    blas = blas_pinning()

    cohort = read_json(args.cohort)
    site_assays = None
    if args.structures is not None:
        sites = read_json(args.structures)
        rows = sites["sites"] if isinstance(sites, dict) else sites
        site_assays = {row["assay_id"] for row in rows if row["status"] == "admitted"}
    selected, excluded = select_assays(cohort, args, site_assays)
    if not selected:
        raise SystemExit("this shard selected no assay; refusing to write a completion record")

    dtype = args.dtype or default_dtype(args.arm)
    scorer = load_position_scorer(
        args.arm, device=args.device, dtype=dtype, model_root=args.model_root
    )
    mask_batch = args.mask_batch_size if paradigm == MASKED else 1

    sources = {
        "cohort": {"path": str(args.cohort), "sha256": sha256_file(args.cohort)},
    }
    if args.structures is not None:
        sources["structures"] = {
            "path": str(args.structures), "sha256": sha256_file(args.structures)
        }
    code = [
        "scripts/capability/position/extract_position_likelihood.py",
        "src/capability/position/position_likelihood.py",
        "src/capability/position/position_terms.py",
        "src/capability/readouts/readout_extraction.py",
        "src/capability/context/context_homologue.py",
        "scripts/capability/stages/context_homologue.py",
    ]
    identity = {
        "schema": SCHEMA,
        "arm": args.arm,
        "paradigm": paradigm,
        "dtype": dtype,
        "rows_per_forward": 1,
        "mask_batch_size": mask_batch,
        "term_semantics": TERM_SEMANTICS[paradigm],
        "alignment_rule": ALIGNMENT,
        "prefix_invariant_rule": PREFIX_INVARIANT_RULE,
        "provenance": scorer.provenance,
        "blas": blas,
        "max_residues": args.max_residues,
        "max_tokens": args.max_tokens,
        "conditionals": not args.no_conditionals,
        "sources": sources,
        "code_sha256": {name: sha256_file(ROOT / name) for name in code},
    }

    archives = args.out / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    receipts, skipped = [], []
    for row in selected:
        assay = row["assay"]
        wildtype = row["wildtype"]
        sequences = [wildtype, *row["sequences"]]
        metadata = {
            "identity": identity,
            "assay": assay,
            "cluster": row["cluster"],
            "mutant_digest": row["mutant_digest"],
            "states": len(sequences),
        }
        path = archives / assay_filename(assay)
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if json.loads(str(saved["metadata"])) != metadata:
                    raise SystemExit(f"{assay}: an existing archive was written under a different identity")
            receipts.append({
                "assay": assay, "file": path.name, "sha256": sha256_file(path), "resumed": True,
                "cluster": row["cluster"], "residues": len(wildtype), "states": len(sequences),
            })
            print(json.dumps({"assay": assay, "status": "resumed"}), flush=True)
            continue

        states = [scorer.pack(sequence) for sequence in sequences]
        widest = max(len(state.ids) for state in states)
        if widest > args.max_tokens:
            skipped.append({"assay": assay, "reason": f"{widest} packed tokens exceeds --max-tokens"})
            print(json.dumps({"assay": assay, "status": "skipped-token-budget"}), flush=True)
            continue

        slots = {} if args.no_conditionals else scorer.residue_token_slots(states)
        gather = [slots[residue] for residue in AA20 if residue in slots] if slots else None
        if paradigm == MASKED:
            wild_score = scorer.score(states[:1], batch_size=mask_batch, gather_ids=gather)[0]
            repeat = scorer.score(states[:1], batch_size=mask_batch)[0]
            mutant_scores = scorer.score(states[1:], batch_size=mask_batch)
        else:
            wild_score = scorer.score(states[:1], gather_ids=gather)[0]
            # One extra forward of the row just scored, so this arm's own
            # nondeterminism is measured on the cohort being checked rather than
            # assumed. Costs one row in a hundred and twenty-nine.
            repeat = scorer.score(states[:1])[0]
            mutant_scores = scorer.score(states[1:])
        assay_repeat = repeat_residual(wild_score, repeat)

        archive = PositionArchive(assay=assay, paradigm=paradigm)
        archive.add_wildtype(states[0], wild_score)
        for mutant, state, score in zip(row["mutants"], states[1:], mutant_scores):
            archive.add_mutant(mutant, state, score)

        wild_sidecar = states[0].sidecar()
        aligned, misaligned, worst_upstream = 0, [], 0.0
        for index, mutant in enumerate(row["mutants"]):
            sites = substitution_sites(mutant, wildtype)
            if len(sites) != 1:
                continue
            verdict = alignment(wild_sidecar, states[index + 1].sidecar(), sites[0])
            if not verdict["aligned"]:
                misaligned.append({"mutation": mutant, "reason": verdict["reason"]})
                continue
            aligned += 1
            worst_upstream = max(
                worst_upstream,
                upstream_invariance(
                    states[0], states[index + 1], wild_score, mutant_scores[index], sites[0]
                ),
            )

        block = conditional_block(states[0], wild_score, slots)
        extras = {}
        if block["available"]:
            extras = {
                "wt_conditional_logprobs": block["logprobs"],
                "wt_conditional_positions": block["positions"],
                "wt_conditional_residues": np.asarray(block["residues"], dtype=object).astype("U"),
            }
        archive.write(path, metadata=metadata, extras=extras)
        receipts.append({
            "assay": assay,
            "file": path.name,
            "sha256": sha256_file(path),
            "resumed": False,
            "cluster": row["cluster"],
            "residues": len(wildtype),
            "states": len(states),
            "packed_tokens": widest,
            "scored_tokens": int(states[0].scored_tokens),
            "retention_max_abs_nats": archive.retention_max_abs_nats,
            "upstream_max_abs_nats": worst_upstream,
            "repeat_max_abs_nats": assay_repeat,
            "aligned_single_substitutions": aligned,
            "misaligned_single_substitutions": misaligned,
            "conditional": {key: value for key, value in block.items()
                            if key not in ("positions", "logprobs")},
        })
        print(json.dumps({"assay": assay, "states": len(states),
                          "retention": archive.retention_max_abs_nats,
                          "upstream": worst_upstream, "repeat": assay_repeat}), flush=True)

    # The prefix invariant is adjudicated once, at run level, against this arm's own
    # repeat maximum over the same cohort. Doing it per assay in scoring order would
    # make the verdict depend on which assay happened to be measured first, because
    # the repeat estimate only improves as assays accumulate; doing it here compares
    # a maximum over N assays against a maximum over the same N assays. A refusal
    # stops the run before the completion record is written, so the cell fails.
    measured = [item for item in receipts if "upstream_max_abs_nats" in item]
    repeat_max = max((float(item["repeat_max_abs_nats"]) for item in measured), default=0.0)
    invariant = {
        "rule": PREFIX_INVARIANT_RULE,
        "tier2_nats": TIER2_NATS,
        "tier2_repeat_multiple": TIER2_REPEAT_MULTIPLE,
        "repeat_max_abs_nats": repeat_max,
        "admitted_tolerance_nats": (
            0.0 if repeat_max == 0.0 else min(TIER2_NATS, TIER2_REPEAT_MULTIPLE * repeat_max)
        ),
        "upstream_max_abs_nats": max(
            (float(item["upstream_max_abs_nats"]) for item in measured), default=0.0
        ),
    }
    invariant["tier"] = residual_tier(invariant["upstream_max_abs_nats"], repeat_max)
    invariant["exactly_zero"] = bool(invariant["upstream_max_abs_nats"] == 0.0)
    refused = [
        {
            "assay": item["assay"],
            "upstream_max_abs_nats": float(item["upstream_max_abs_nats"]),
            "repeat_max_abs_nats": float(item["repeat_max_abs_nats"]),
        }
        for item in measured
        if residual_tier(float(item["upstream_max_abs_nats"]), repeat_max) == 3
    ]
    for item in measured:
        item["upstream_tier"] = residual_tier(float(item["upstream_max_abs_nats"]), repeat_max)
    if paradigm == CAUSAL and refused:
        worst = max(entry["upstream_max_abs_nats"] for entry in refused)
        raise SystemExit(
            f"{args.arm}: {len(refused)} assay(s) carry a prefix residual the pre-registered "
            f"rule refuses, worst {worst} nats against a measured repeat maximum of "
            f"{repeat_max} nats over {len(measured)} assays. The prefix is the same tokens in "
            "both states, so a residual this far above the arm's own reproducibility is a "
            "packing, alignment or layout property of the arm and not a measurement. "
            f"Refused assays: {[entry['assay'] for entry in refused][:5]}"
        )

    # Every archive present in this cell enters the handoff, resumed ones included:
    # a resumed cell that described only its fresh assays would hand the frozen CPU
    # pipeline a strictly smaller cohort than it actually holds.
    by_assay = {row["assay"]: row for row in selected}
    scored = [item for item in receipts if not item["resumed"]]
    handoff = {
        "schema": "position_response_archive_input_v1",
        "arm": args.arm,
        "assays": [
            {
                "assay": item["assay"],
                "cluster": by_assay[item["assay"]]["cluster"],
                "protein": item["assay"],
                "wildtype": by_assay[item["assay"]]["wildtype"],
                "mutants": by_assay[item["assay"]]["mutants"],
                "sequences": by_assay[item["assay"]]["sequences"],
                "archive": f"archives/{item['file']}",
            }
            for item in receipts
        ],
    }
    write_json(args.out / "archive-input.json", handoff)

    write_json(args.out / COMPLETION, {
        "status": "complete",
        "identity": identity,
        "prefix_invariant": invariant,
        "created_utc": _now(),
        "shard": args.shard,
        "shards": args.shards,
        "assays": receipts,
        "excluded_assays": excluded,
        "skipped_assays": skipped,
        "totals": {
            "assays_present": len(receipts),
            "assays_written": len(scored),
            "assays_resumed": len(receipts) - len(scored),
            "states": sum(int(item.get("states", 0)) for item in receipts),
            "retention_max_abs_nats": max(
                (float(item["retention_max_abs_nats"]) for item in scored), default=0.0
            ),
            "upstream_max_abs_nats": invariant["upstream_max_abs_nats"],
            "repeat_max_abs_nats": repeat_max,
            "upstream_tier": invariant["tier"],
            "conditionals_available": sum(
                1 for item in scored if item["conditional"]["available"]
            ),
        },
        "handoff": {
            "archive_input": "archive-input.json",
            "packing_export": (
                "python scripts/capability/extensions/analyse_residue_responses.py --mode packing "
                "--arm <arm> --input <out>/archive-input.json --out <new>/packing-export.json"
            ),
            "frozen_contact_analysis": (
                "python scripts/capability/extensions/analyse_residue_responses.py --mode contacts "
                "--input <new>/packing-export.json --structures <sites.json.gz> "
                "--coverage <coverage.json> --out <new>/contacts.json"
            ),
            "note": (
                "the packing sidecar is re-derived rather than copied, so a fresh "
                "tokenizer-only packing is checked against the counts retained here"
            ),
        },
        "torch": torch.__version__,
        "gpu": (torch.cuda.get_device_name(args.device)
                if str(args.device).startswith("cuda") and torch.cuda.is_available() else None),
    })


if __name__ == "__main__":
    main()
