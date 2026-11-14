#!/usr/bin/env python3
"""The likelihood instrument of the reward-provenance experiment: four declared families.

Declared in ``docs/rl-environment-provenance.md``. Four of that declaration's
twenty families read one instrument -- a model's own per-residue likelihood and
its own predictive entropy -- and this producer is that instrument:

``cross_nll``            each pool's sequences under the base model, and the base
                         pool's sequences under each policy, in nats per residue
                         token with the stop decision excluded.
``support_tail_share``   the share of a pool's sequences whose per-residue NLL
                         under the **base** model exceeds the base pool's own 0.99
                         quantile of the same quantity. This is the one that gates
                         a bias reading: without it an installed bias cannot be
                         distinguished from a reweighting inside the base's own
                         support, and the declaration forbids reading the share as
                         zero when it has not been measured.
``entropy``              each pool's own token-mean predictive entropy, per
                         sequence, under the model that produced it.
``held_out_natural_nll`` a natural-sequence reference set under the base and under
                         each policy.

The measurement convention is not re-implemented here. ``mean_nll`` over extracted
residues with the stop token excluded, float32 weights, temperature one, right
padding with an explicit mask, length-sorted batches and the batching gate that
aborts before writing anything all come from
``scripts/capability/generation/score_cohort_likelihood.py``, which is this
project's one statement of that convention and is imported by path. What this
producer adds is which sequences are scored under which weights, and the entropy,
which is a different quantity rather than a second spelling of the same one.

Three stages:

``natural``   draw the natural-sequence reference set at its declared seed.
``score``     score one model over one or more source pools, and the natural set.
``assemble``  turn the scored tables into the four families' columns and the
              support-tail shares, in the shape ``analyse_rl_provenance.py``
              consumes through ``--supplied`` and ``--support``.

The reference set is a *natural-sequence* reference and not a held-out set in the
training sense: the policy's declared pre-training mixture is UniRef90 and
GigaRef, this draws from Swiss-Prot, and this project's own non-identifiability
finding is that corpus exposure cannot be established from either. What the
endpoint measures is the change in fit to natural protein between the base and a
policy, on sequences neither was optimised against during this experiment. That
is a real cost and it is not a perplexity on unseen data; the name is kept for the
declaration's sake and the limitation travels with every number it produces.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.extensions import rl_environment_provenance as rp  # noqa: E402
from src.capability.extensions import rl_provenance_analysis as ra  # noqa: E402

#: The one statement of the cross-likelihood convention, imported by path because
#: it is an entry point rather than a package module. Duplicating its scoring
#: would be a second spelling of one measured quantity.
CONVENTION = REPO_ROOT / "scripts/capability/generation/score_cohort_likelihood.py"

#: The natural-sequence reference set, declared in full so it is reproducible
#: from this file alone.
NATURAL_POOL = "natural_reference"
NATURAL_SEED = 20261114
NATURAL_COUNT = 1024
#: Length support, chosen to overlap the pools' own so that a likelihood
#: difference is not a length difference. The pools are generated under a
#: 200-token budget and their realised lengths run from the low tens upward.
NATURAL_LENGTH_WINDOW = (60, 200)

DIRECTIONS = ("pool_under_base", "base_under_pool")


def convention():
    """The shared cross-likelihood producer, loaded as a module."""

    spec = importlib.util.spec_from_file_location("score_cohort_likelihood", CONVENTION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_for(pool: str, policies: Path | None) -> Path:
    if pool == rp.POOL_BASE:
        return rp.policy_checkpoint()
    if pool == rp.POOL_PUBLISHED_DPO:
        return rp.published_dpo_checkpoint()
    if policies is None:
        raise SystemExit(f"pool {pool!r} is a trained arm, so --policies is required")
    directory = Path(policies) / pool / "latest"
    if not (directory / "config.json").is_file():
        raise SystemExit(f"no trained checkpoint for arm {pool!r} at {directory}")
    return directory


def pool_sequences(pools: Path, pool: str) -> list[dict[str, Any]]:
    """One pool's per-stream sequences, from the closed-form statistics table.

    The attempt stays the unit and the stream stays the resampling stratum, so
    every attempt is carried with its stream even though identical sequences
    share one score.
    """

    path = Path(pools) / f"{pool}_closed_form_statistics.jsonl"
    if not path.is_file():
        raise SystemExit(f"no closed-form statistics for pool {pool!r} at {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append(row)
    return rows


def pool_strings(pools: Path, pool: str) -> dict[str, str]:
    """Digest to sequence for one pool, read from its own generation cells.

    The statistics table carries digests rather than sequences, because the
    sequences are already in the lane's own immutable ledgers and a second copy
    of them is a second source for one fact.
    """

    from src.capability.generation import unconditional_generation as ug

    out: dict[str, str] = {}
    for stream in range(rp.STREAMS):
        directory = Path(pools) / f"rlp_{pool}_stream{stream}"
        for row in ug.read_attempts(directory / ug.LEDGER_NAME):
            if row["sequence"]:
                out[row["sequence_sha256"]] = row["sequence"]
    if not out:
        raise SystemExit(f"pool {pool!r} carries no non-empty sequence under {pools}")
    return out


def natural(args: argparse.Namespace) -> int:
    """Draw the natural-sequence reference set at its declared seed."""

    from src.capability.core.arms import SWISSPROT_FASTA

    source = Path(args.fasta) if args.fasta else Path(SWISSPROT_FASTA)
    low, high = NATURAL_LENGTH_WINDOW
    alphabet = set(AA20)
    opener = gzip.open if source.name.endswith(".gz") else open
    admitted: list[str] = []
    with opener(source, "rt", encoding="utf-8", errors="strict") as handle:
        pieces: list[str] = []
        for line in handle:
            if line.startswith(">"):
                sequence = "".join(pieces)
                if low <= len(sequence) <= high and not set(sequence) - alphabet:
                    admitted.append(sequence)
                pieces = []
            else:
                pieces.append(line.strip())
        sequence = "".join(pieces)
        if low <= len(sequence) <= high and not set(sequence) - alphabet:
            admitted.append(sequence)
    if len(admitted) < NATURAL_COUNT:
        raise SystemExit(
            f"{source} yields {len(admitted)} admissible entries, below the declared "
            f"{NATURAL_COUNT}; the reference set is a declared size and is not shrunk "
            "to fit what a source happens to hold"
        )
    rng = np.random.default_rng(NATURAL_SEED)
    chosen = sorted(set(rng.choice(len(admitted), size=NATURAL_COUNT, replace=False).tolist()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{NATURAL_POOL}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in chosen:
            sequence = admitted[index]
            handle.write(json.dumps({
                "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "sequence": sequence, "length": len(sequence),
            }, sort_keys=True) + "\n")
    write_json(out / f"{NATURAL_POOL}_record.json", {
        "schema": rp.SCHEMA, "campaign": rp.CAMPAIGN, "stage": "natural",
        "source": str(source), "source_sha256_available": source.is_file(),
        "admissible_entries": len(admitted), "drawn": NATURAL_COUNT,
        "seed": NATURAL_SEED, "length_window": list(NATURAL_LENGTH_WINDOW),
        "alphabet": "the canonical twenty; an entry carrying anything else is not drawn",
        "sha256": sha256_file(path),
        "limitation": (
            "a natural-sequence reference, not a held-out set: the policy's declared "
            "pre-training mixture is UniRef90 and GigaRef, this draws from Swiss-Prot, "
            "and corpus exposure cannot be established from either. The endpoint is the "
            "change in fit to natural protein between base and policy, not a perplexity "
            "on unseen data"
        ),
        "written_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({"drawn": NATURAL_COUNT, "admissible": len(admitted),
                      "path": str(path)}, indent=1), flush=True)
    return 0


def score(args: argparse.Namespace) -> int:
    """Score one model's weights over one or more source pools, plus the natural set."""

    import torch

    module = convention()
    checkpoint = checkpoint_for(args.model, args.policies)
    scorer = module.load_scorer(args.model, args.device, args.dtype, checkpoint=checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sources: dict[str, dict[str, str]] = {}
    for pool in args.source:
        sources[pool] = pool_strings(args.pools, pool)
    if args.natural:
        natural_path = Path(args.natural)
        table = {}
        for line in natural_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                table[row["sequence_sha256"]] = row["sequence"]
        sources[NATURAL_POOL] = table

    first = next(iter(sources.values()))
    gate = module.batching_gate(scorer, sorted(first.values(), key=len), args.gate_threshold)
    write_json(out / f"{args.model}.gate.json", gate)
    print(f"{args.model}: batching gate passed, max diff "
          f"{gate['max_abs_diff_mean_nll_or_stop_nll']:.2e}", flush=True)

    prompt_length = len(scorer["prompt_ids"])
    for pool, table in sources.items():
        distinct = sorted(set(table.values()), key=len)
        scored = module.score_all(scorer, distinct, max_batch=args.max_batch,
                                  max_tokens=args.max_tokens)
        entropy = per_sequence_entropies(scorer, distinct, prompt_length=prompt_length,
                                         max_batch=args.max_batch,
                                         max_tokens=args.max_tokens)
        stem = f"{args.model}__on__{pool}"
        with (out / f"{stem}.jsonl").open("w", encoding="utf-8") as handle:
            for sequence in distinct:
                tokens, total, stop = scored[sequence]
                handle.write(json.dumps({
                    "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    "n_tokens": tokens, "sum_nll": total, "mean_nll": total / tokens,
                    "stop_nll": stop, "mean_entropy_nats": entropy[sequence],
                }, sort_keys=True) + "\n")
        write_json(out / f"{stem}.manifest.json", {
            "schema": rp.SCHEMA, "campaign": rp.CAMPAIGN, "stage": "score",
            "model": args.model, "model_checkpoint": str(checkpoint), "source": pool,
            "distinct_sequences": len(distinct), "dtype": scorer["dtype"],
            "temperature": 1.0, "prompt_ids": scorer["prompt_ids"],
            "close_token": scorer["close_token"], "close_id": scorer["close_id"],
            "convention": str(CONVENTION.relative_to(REPO_ROOT)),
            "convention_sha256": sha256_file(CONVENTION),
            "scored_tokens": "mean_nll over extracted residues only, stop token "
                             "excluded; stop_nll separate; entropy is the model's own "
                             "next-token entropy averaged over the same positions",
            "gate": gate,
            "written_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"{stem}: scored {len(distinct)} distinct sequences", flush=True)
    del scorer
    torch.cuda.empty_cache()
    return 0


def per_sequence_entropies(scorer: dict[str, Any], sequences: list[str], *,
                           prompt_length: int, max_batch: int,
                           max_tokens: int) -> dict[str, float]:
    """Each sequence's own token-mean predictive entropy under the loaded model.

    Batched exactly as the likelihood is -- length-sorted, right-padded, with the
    residue positions masked from true lengths -- so the two quantities are read
    off the same forward geometry and cannot disagree about which positions
    belong to a sequence.
    """

    import torch

    from src.capability.extensions import grpo

    model = scorer["model"]
    device = next(model.parameters()).device
    order = sorted(set(sequences), key=len)
    out: dict[str, float] = {}
    index = 0
    while index < len(order):
        stop = index + 1
        while (stop < len(order) and stop - index < max_batch
               and (stop - index + 1) * len(order[stop]) <= max_tokens):
            stop += 1
        batch = order[index:stop]
        encoded = [scorer["prompt_ids"] + [scorer["residue_ids"][residue]
                                           for residue in sequence]
                   for sequence in batch]
        width = max(len(item) for item in encoded)
        ids = torch.full((len(encoded), width), scorer["pad_id"], dtype=torch.long)
        mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for row, item in enumerate(encoded):
            ids[row, : len(item)] = torch.tensor(item)
            mask[row, : len(item)] = 1
        ids, mask = ids.to(device), mask.to(device)
        action = mask[:, prompt_length:].bool()
        values = grpo.per_sequence_entropy(model, ids, mask, action,
                                           prompt_length=prompt_length)
        for sequence, value in zip(batch, values):
            out[sequence] = float(value)
        index = stop
    return out


def _streams(pools: Path, pool: str, scores: dict[str, float]) -> dict[int, list[float]]:
    """One pool's scored values by stream, through the shared digest join.

    The join lives in the analysis module because the fold, the likelihood and
    the homology search all measure a distinct sequence and all have to be read
    back onto attempts; one implementation of that keeps the three from drifting.
    """

    try:
        return ra.streams_by_digest(pool_sequences(pools, pool), scores, pool=pool,
                                    quantity="likelihood")
    except ValueError as error:
        raise SystemExit(str(error)) from error


def read_scores(directory: Path, model: str, source: str, field: str) -> dict[str, float]:
    path = Path(directory) / f"{model}__on__{source}.jsonl"
    if not path.is_file():
        raise SystemExit(f"no scored table at {path}")
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row["sequence_sha256"])] = float(row[field])
    return out


def assemble(args: argparse.Namespace) -> int:
    """Form the four families' columns and the support-tail shares."""

    scores = Path(args.scores)
    pools = Path(args.pools)
    families: dict[str, Any] = {}

    # cross_nll: each pool under the base model, and the base pool under each policy.
    base_on = {pool: read_scores(scores, rp.POOL_BASE, pool, "mean_nll")
               for pool in rp.EVALUATED_POOLS}
    columns = []
    for pool in rp.CONTRAST_POOLS:
        columns.append({"key": f"{pool}|pool_under_base",
                        "left": _streams(pools, pool, base_on[pool]),
                        "right": _streams(pools, rp.POOL_BASE, base_on[rp.POOL_BASE])})
        own = read_scores(scores, pool, rp.POOL_BASE, "mean_nll")
        own_on_self = read_scores(scores, pool, pool, "mean_nll")
        columns.append({"key": f"{pool}|base_under_pool",
                        "left": _streams(pools, rp.POOL_BASE, own),
                        "right": _streams(pools, pool, own_on_self)})
    families[rp.FAMILY_CROSS_NLL] = ra.contrast_bands(columns, family=rp.FAMILY_CROSS_NLL)

    # entropy: each pool under its own model, against the base pool under the base.
    base_entropy = read_scores(scores, rp.POOL_BASE, rp.POOL_BASE, "mean_entropy_nats")
    columns = []
    for pool in rp.CONTRAST_POOLS:
        own = read_scores(scores, pool, pool, "mean_entropy_nats")
        columns.append({"key": pool, "left": _streams(pools, pool, own),
                        "right": _streams(pools, rp.POOL_BASE, base_entropy)})
    families[rp.FAMILY_ENTROPY] = ra.contrast_bands(columns, family=rp.FAMILY_ENTROPY)

    # held_out_natural_nll: the reference set under each policy, against the base.
    natural_base = read_scores(scores, rp.POOL_BASE, NATURAL_POOL, "mean_nll")
    reference = [path for path in (Path(args.natural),) if path.is_file()]
    if not reference:
        raise SystemExit("--natural must name the reference set this was scored over")
    digests = [json.loads(line)["sequence_sha256"]
               for line in reference[0].read_text(encoding="utf-8").splitlines()
               if line.strip()]
    # One stream: the reference set is a single declared draw, not a streamed pool,
    # so it is resampled as one stratum and that is recorded rather than faked into
    # three.
    columns = []
    for pool in rp.CONTRAST_POOLS:
        natural_pool = read_scores(scores, pool, NATURAL_POOL, "mean_nll")
        columns.append({
            "key": pool,
            "left": {0: [natural_pool[digest] for digest in digests]},
            "right": {0: [natural_base[digest] for digest in digests]},
        })
    families[rp.FAMILY_HELD_OUT_NLL] = ra.contrast_bands(
        columns, family=rp.FAMILY_HELD_OUT_NLL)
    families[rp.FAMILY_HELD_OUT_NLL]["resampling_note"] = (
        "the reference set is one declared draw of natural sequences, so it carries "
        "one resampling stratum rather than the three streams a generated pool has; "
        "the sequence is still the unit"
    )

    # support_tail_share: the base model's own tail, and each pool's share beyond it.
    base_values = np.concatenate([
        np.asarray(values, dtype=float)
        for values in _streams(pools, rp.POOL_BASE, base_on[rp.POOL_BASE]).values()
    ])
    support = {}
    share_columns = []
    for pool in rp.CONTRAST_POOLS:
        pool_values = np.concatenate([
            np.asarray(values, dtype=float)
            for values in _streams(pools, pool, base_on[pool]).values()
        ])
        support[pool] = ra.support_tail_share(pool_values, base_values)
        threshold = support[pool]["threshold_nats_per_residue"]
        share_columns.append({
            "key": pool,
            "left": {stream: [1.0 if value > threshold else 0.0 for value in values]
                     for stream, values in _streams(pools, pool, base_on[pool]).items()},
            "right": {stream: [1.0 if value > threshold else 0.0 for value in values]
                      for stream, values in
                      _streams(pools, rp.POOL_BASE, base_on[rp.POOL_BASE]).items()},
        })
    families[rp.FAMILY_SUPPORT_TAIL] = ra.contrast_bands(
        share_columns, family=rp.FAMILY_SUPPORT_TAIL)
    families[rp.FAMILY_SUPPORT_TAIL]["threshold_note"] = (
        "the threshold is the base pool's own 0.99 quantile of per-residue NLL under "
        "the base model, so the base column is 0.01 by construction and the contrast "
        "is each pool's excess over that null"
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "supplied_families.json", families)
    write_json(out / "support_tail_share.json", support)
    write_json(out / "likelihood_record.json", {
        "schema": rp.SCHEMA, "campaign": rp.CAMPAIGN, "stage": "assemble",
        "families": sorted(families),
        "convention": str(CONVENTION.relative_to(REPO_ROOT)),
        "convention_sha256": sha256_file(CONVENTION),
        "natural_reference": str(reference[0]),
        "written_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({family: {"resolved": record["resolved_family_size"],
                               "declared": record["declared_family_size"]}
                      for family, record in families.items()}
                     | {"support": {pool: round(value["share_above"], 5)
                                    for pool, value in support.items()}}, indent=1),
          flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    draw = sub.add_parser("natural")
    draw.add_argument("--out", type=Path, required=True)
    draw.add_argument("--fasta", type=Path, default=None)

    run = sub.add_parser("score")
    run.add_argument("--model", required=True, choices=list(rp.EVALUATED_POOLS))
    run.add_argument("--source", action="append", required=True,
                     choices=list(rp.EVALUATED_POOLS))
    run.add_argument("--pools", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--policies", type=Path, default=None)
    run.add_argument("--natural", type=Path, default=None)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--dtype", default="float32")
    run.add_argument("--max-batch", type=int, default=16)
    run.add_argument("--max-tokens", type=int, default=8192)
    run.add_argument("--gate-threshold", type=float, default=1e-4)

    build = sub.add_parser("assemble")
    build.add_argument("--scores", type=Path, required=True)
    build.add_argument("--pools", type=Path, required=True)
    build.add_argument("--natural", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    return {"natural": natural, "score": score, "assemble": assemble}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
