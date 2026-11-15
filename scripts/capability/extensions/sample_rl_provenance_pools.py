#!/usr/bin/env python3
"""Generate and measure the reward-provenance experiment's evaluation pools.

Declared in ``docs/rl-environment-provenance.md``. One invocation is one pool:
the untrained base, one trained arm's final checkpoint, or the released DPO
sibling. Every pool is drawn through the project's free-length unconditional
lane at its frozen operating point, in the declared number of streams at the
declared seeds, with the producing checkpoint recorded in each cell's ledger --
which is what makes the pools distinguishable at all, since all of them are the
same arm name with different weights.

Three stages, because they need different things:

``generate``  samples the declared streams and writes one lane cell per stream,
              then the pool's closed-form statistics and the environment reward,
              which need no structure. It needs the policy on a GPU.
``cohort``    emits the fold cohort the project's own structure runner consumes,
              one record per distinct non-empty sequence of the pool. It needs
              nothing but the ledgers.
``collect``   joins the folded structures back onto the pool, writes every
              reward under every declared reward function, and writes the
              per-stream value tables the analysis resamples. It needs the
              completed evidence tree.

The fold itself is not done here: it is ``run_structure_evidence.py``, unchanged,
which owns the evidence tree, the receipts and the replay verification. The
second declared realisation of the base pool is that runner at the declared
second base seed, into its own tree, because the seed is part of the evaluation
signature and two realisations must not share one tree.

Usage:
    python scripts/capability/extensions/sample_rl_provenance_pools.py generate \
        --pool base --out <run dir> --device cuda:0
    python ... cohort   --pool base --out <run dir>
    python ... collect  --pool base --out <run dir> \
        --structures <evidence tree>/index.jsonl [--structures-b <tree B>/index.jsonl]
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

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.extensions import rl_environment_provenance as rp  # noqa: E402
from src.capability.extensions import sequence_reward as sr  # noqa: E402
from src.capability.generation import objective_tuned_arms as ot  # noqa: E402
from src.capability.generation import structure_evidence as se  # noqa: E402
from src.capability.generation import unconditional_generation as ug  # noqa: E402

#: One directory per (pool, stream), named the way the lane names its cells so
#: that a reader of the tree can tell a generation cell from anything else.
CELL = "rlp_{pool}_stream{stream}"
POOL_RECORD = "pool.json"
STATISTICS = "closed_form_statistics.jsonl"
COHORT = "fold_cohort.jsonl"
VALUES = "pool_values.json"


def cell_dirs(out: Path, pool: str) -> list[Path]:
    return [Path(out) / CELL.format(pool=pool, stream=stream)
            for stream in range(rp.STREAMS)]


def read_pool(out: Path, pool: str) -> list[dict[str, Any]]:
    """Every attempt of every stream of one pool, with its stream recorded.

    The lane's ledger does not carry a stream label -- it carries the realised
    seed, which is what distinguishes the cells -- so the stream is attached here
    from the declared seed and refused if the ledger's seed is not one of them.
    """

    seeds = rp.pool_seeds(pool)
    rows: list[dict[str, Any]] = []
    for stream, directory in enumerate(cell_dirs(out, pool)):
        summary_path = directory / ug.EXPECT_GENERATE
        if not summary_path.is_file():
            raise SystemExit(f"{pool} stream {stream}: no generation record at {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary["seed"]) != seeds[stream]:
            raise SystemExit(
                f"{pool} stream {stream}: the ledger records seed {summary['seed']}, "
                f"not the declared {seeds[stream]}"
            )
        if summary["arm"] != rp.POLICY_ARM:
            raise SystemExit(f"{pool} stream {stream}: ledger names arm {summary['arm']!r}")
        for row in ug.read_attempts(directory / ug.LEDGER_NAME):
            rows.append({**row, "pool": pool, "stream": stream,
                         "checkpoint": summary["checkpoint"]})
    return rows


def checkpoint_for(pool: str, *, policies: Path | None) -> Path:
    """The weights one pool is drawn from, resolved rather than passed by hand."""

    if pool == rp.POOL_BASE:
        return rp.policy_checkpoint()
    if pool == rp.POOL_PUBLISHED_DPO:
        return rp.published_dpo_checkpoint()
    if policies is None:
        raise SystemExit(
            f"pool {pool!r} is a trained arm, so --policies must name the directory "
            "holding the arms' run directories"
        )
    directory = Path(policies) / pool / "latest"
    if not (directory / "config.json").is_file():
        raise SystemExit(f"no trained checkpoint for arm {pool!r} at {directory}")
    return directory


def generate(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_for(args.pool, policies=args.policies)
    model, tokenizer = ug.load_generator(rp.POLICY_ARM, device=args.device,
                                         checkpoint=checkpoint)
    prompt = ug.prompt_for(rp.POLICY_ARM)
    interface = ot.prompt_contract(rp.POLICY_ARM, tokenizer, prompt)
    ot.residue_alphabet_contract(rp.POLICY_ARM, tokenizer)
    # The declared completion budget, which is this experiment's and not the
    # lane's frozen 400: it is the budget the policies were trained under and the
    # one the calibration record measured the base distribution at.
    ug.POLICY["max_new_tokens"] = rp.MAX_NEW_TOKENS
    attempts = rp.pool_attempts(args.pool) if args.smoke is None else int(args.smoke)
    seeds = rp.pool_seeds(args.pool)
    cells = []
    for stream, directory in enumerate(cell_dirs(out, args.pool)):
        rows = ug.generate(rp.POLICY_ARM, n=attempts, seed=seeds[stream], model=model,
                           tokenizer=tokenizer, output_dir=directory,
                           checkpoint=checkpoint)
        cells.append({"stream": stream, "directory": str(directory), "seed": seeds[stream],
                      "attempts": len(rows),
                      "non_empty": sum(1 for row in rows if row["sequence"])})
        print(json.dumps(cells[-1]), flush=True)

    rows = read_pool(out, args.pool)
    statistics_path = out / f"{args.pool}_{STATISTICS}"
    with statistics_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            stats = sr.statistics(row["sequence"])
            handle.write(json.dumps({
                "id": row["id"], "stream": row["stream"],
                "sequence_sha256": row["sequence_sha256"],
                "stop_status": row["stop_status"],
                **stats.as_record(),
                "environment_reward": sr.reward(sr.REWARD_ENVIRONMENT, stats),
                "terms": {key: value for key, value in
                          sr.terms(stats, mean_ca_plddt=None).items()},
                "hits": sr.meets_declared_objective(stats),
            }, sort_keys=True) + "\n")
    record = {
        "schema": rp.SCHEMA, "campaign": rp.CAMPAIGN, "stage": "generate",
        "pool": args.pool, "arm": rp.POLICY_ARM, "checkpoint": str(checkpoint),
        "operating_point": {key: ug.POLICY[key] for key in
                            ("temperature", "top_p", "top_k", "max_new_tokens",
                             "batch_size", "repetition_penalty")},
        "declared_completion_budget": rp.MAX_NEW_TOKENS,
        "streams": cells, "interface": interface,
        "attempts_total": len(rows),
        "non_empty_total": sum(1 for row in rows if row["sequence"]),
        "distinct_non_empty": len({row["sequence"] for row in rows if row["sequence"]}),
        "closed_form_statistics": str(statistics_path),
        "closed_form_statistics_sha256": sha256_file(statistics_path),
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "declared_attempts_per_stream": rp.pool_attempts(args.pool),
        "is_scientific_measurement": all(
            cell["attempts"] == rp.pool_attempts(args.pool) for cell in cells
        ),
    }
    write_json(out / f"{args.pool}_{POOL_RECORD}", record)
    print(json.dumps({key: record[key] for key in
                      ("pool", "attempts_total", "non_empty_total", "distinct_non_empty",
                       "is_scientific_measurement")}, indent=1), flush=True)
    return 0


def cohort(args: argparse.Namespace) -> int:
    """The fold cohort of one pool: one record per distinct non-empty sequence.

    Distinct, because the project's folding instrument is content-addressed: two
    identical sequences share a fold and the attempt stays the sampling unit. The
    cohort id is the sequence digest for the same reason -- it is the key the
    evidence tree is organised by, and joining back on it is what ``collect``
    does.
    """

    out = Path(args.out)
    rows = read_pool(out, args.pool)
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        sequence = row["sequence"]
        if not sequence:
            continue
        digest = se.sequence_digest(sequence)
        seen.setdefault(digest, {"id": digest, "sequence": sequence,
                                 "sequence_sha256": digest, "pool": args.pool})
    path = out / f"{args.pool}_{COHORT}"
    with path.open("w", encoding="utf-8") as handle:
        for digest in sorted(seen):
            handle.write(json.dumps(seen[digest], sort_keys=True) + "\n")
    se.load_cohort(path)  # the runner's own reader, as the acceptance check
    print(json.dumps({"pool": args.pool, "cohort": str(path),
                      "distinct_sequences": len(seen),
                      "sha256": sha256_file(path)}, indent=1), flush=True)
    return 0


def read_structures(path: Path) -> dict[str, Any]:
    """Mean CA-pLDDT per sequence digest from a completed evidence index.

    The runner nests each row's measurement under ``structure``, carrying the
    status beside it, so a pending or failed row is absent from the table rather
    than read as a low confidence. The index's own evaluation signature is
    returned with the table: two realisations of the same pool differ exactly in
    that signature, and recording it is what shows the cross-fit used two folds
    and not one.
    """

    table: dict[str, float] = {}
    signatures: set[str] = set()
    statuses: dict[str, int] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        structure = json.loads(line).get("structure") or {}
        status = str(structure.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
        if status != "ok":
            continue
        value = structure.get("mean_ca_plddt")
        if value is None:
            continue
        signatures.add(str(structure.get("evaluation_signature")))
        table[str(structure["sequence_sha256"])] = float(value)
    if not table:
        raise SystemExit(f"{path} carries no readable mean CA-pLDDT")
    if len(signatures) != 1:
        raise SystemExit(
            f"{path} mixes {len(signatures)} evaluation signatures; one realisation is "
            "one signature and a mixed tree is not one realisation"
        )
    return {"index": str(path), "table": table, "sequences": len(table),
            "evaluation_signature": signatures.pop(), "status_counts": statuses}


def collect(args: argparse.Namespace) -> int:
    out = Path(args.out)
    rows = read_pool(out, args.pool)
    realisations = {"A": read_structures(args.structures)}
    if args.structures_b is not None:
        realisations["B"] = read_structures(args.structures_b)
        if (realisations["A"]["evaluation_signature"]
                == realisations["B"]["evaluation_signature"]):
            raise SystemExit(
                "the two realisations carry one evaluation signature, so they are one "
                "fold read twice; the cross-fit needs the declared second base seed, "
                "which the signature records"
            )
    elif args.pool in rp.DOUBLE_FOLDED_POOLS:
        raise SystemExit(
            f"pool {args.pool!r} is declared double-folded, so the best-of-k control's "
            "second realisation is required; pass --structures-b"
        )

    values: dict[str, dict[str, dict[str, list[float]]]] = {}
    missing = 0
    per_sequence = []
    for row in rows:
        sequence, stream = row["sequence"], str(row["stream"])
        stats = sr.statistics(sequence)
        entry: dict[str, Any] = {"id": row["id"], "stream": row["stream"],
                                 "sequence_sha256": row["sequence_sha256"],
                                 "stop_status": row["stop_status"], **stats.as_record()}
        foldable = sr.within_structure_support(stats)
        for name, realisation in realisations.items():
            plddt = (None if not foldable
                     else realisation["table"].get(row["sequence_sha256"]))
            if foldable and plddt is None:
                missing += 1
                continue
            entry[f"mean_ca_plddt_{name}"] = plddt
            for reward_name, value in sr.rewards(stats, mean_ca_plddt=plddt).items():
                entry[f"reward_{reward_name}_{name}"] = value
                values.setdefault(reward_name, {}).setdefault(name, {}).setdefault(
                    stream, []).append(value)
            if plddt is not None:
                values.setdefault("mean_ca_plddt", {}).setdefault(name, {}).setdefault(
                    stream, []).append(plddt)
        entry["hits"] = sr.meets_declared_objective(
            stats, mean_ca_plddt=entry.get("mean_ca_plddt_A")
        )
        # The hit rates are per-sequence indicators so that a rate is a pool mean
        # and resamples under exactly the machinery every other column does.
        # The structure-conditional hit lands under realisation A, the rest under
        # the closed-form table, because that is where each is measured.
        for key, value in entry["hits"].items():
            # Flat, and spelled exactly as the value tables spell it, so a
            # stratified family and a pooled one read one name for one quantity.
            entry[f"hit_{key}"] = 1.0 if value else 0.0
        for key, value in entry["hits"].items():
            realisation = "A" if key in ("plddt_criterion", "memory_conjunction") \
                else "closed_form"
            values.setdefault(f"hit_{key}", {}).setdefault(realisation, {}).setdefault(
                stream, []).append(1.0 if value else 0.0)
        for key in ("length", "low_complexity_fraction", "distinct_5mer_fraction",
                    "longest_single_residue_run", "composition_entropy_bits",
                    "net_charge_ph7"):
            values.setdefault(key, {}).setdefault("closed_form", {}).setdefault(
                stream, []).append(float(entry[key]))
        per_sequence.append(entry)
    if missing:
        raise SystemExit(
            f"{missing} attempts of pool {args.pool!r} are inside the declared "
            f"structural support {sr.STRUCTURE_SUPPORT} and have no folded structure "
            "in the evidence index; a pool is collected when its fold is complete, "
            "and a partial pool is not reported as a pool. Completions outside that "
            "support are not counted here: they are paid the empty-completion reward "
            "by declaration and were never folded"
        )

    joined = out / f"{args.pool}_per_sequence.jsonl"
    with joined.open("w", encoding="utf-8") as handle:
        for entry in per_sequence:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    record = {
        "schema": rp.SCHEMA, "campaign": rp.CAMPAIGN, "stage": "collect",
        "pool": args.pool,
        "realisations": {name: {key: value for key, value in realisation.items()
                                if key != "table"}
                         for name, realisation in realisations.items()},
        "attempts": len(rows),
        "per_sequence": str(joined),
        "per_sequence_sha256": sha256_file(joined),
        "values": {quantity: {realisation: {stream: len(items)
                                            for stream, items in streams.items()}
                              for realisation, streams in by_realisation.items()}
                   for quantity, by_realisation in values.items()},
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / f"{args.pool}_{VALUES}",
               {"record": record,
                "values": {quantity: {realisation: {stream: items
                                                    for stream, items in streams.items()}
                                      for realisation, streams in by_realisation.items()}
                           for quantity, by_realisation in values.items()}})
    write_json(out / f"{args.pool}_collect.json", record)
    print(json.dumps({key: record[key] for key in ("pool", "attempts")}
                     | {"quantities": sorted(values)}, indent=1), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("generate", "cohort", "collect"):
        stage = sub.add_parser(name)
        stage.add_argument("--pool", required=True, choices=list(rp.EVALUATED_POOLS))
        stage.add_argument("--out", type=Path, required=True)
        if name == "generate":
            stage.add_argument("--device", default="cuda:0")
            stage.add_argument("--policies", type=Path, default=None,
                               help="directory holding the trained arms' run directories")
            # Interface check only. A pool drawn at anything but its declared
            # attempt count is recorded is_scientific_measurement false, which is
            # the lane's own rule for a sample that is not the declared one.
            stage.add_argument("--smoke", type=int, default=None,
                               help="attempts per stream for an interface check; the "
                                    "record then claims no measurement")
        if name == "collect":
            stage.add_argument("--structures", type=Path, required=True,
                               help="evidence index of realisation A")
            stage.add_argument("--structures-b", type=Path, default=None,
                               help="evidence index of realisation B, required for a "
                                    "declared double-folded pool")
    args = parser.parse_args(argv)
    return {"generate": generate, "cohort": cohort, "collect": collect}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
