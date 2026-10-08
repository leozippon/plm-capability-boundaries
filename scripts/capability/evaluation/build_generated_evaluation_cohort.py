#!/usr/bin/env python3
"""Freeze the evaluation cohort for E14 and the fixed candidate pool for E17.

One cohort file serves both experiments and both instruments. It holds, under
one identifier each:

* the generated sequences E14 compares, drawn per arm from the two declared
  termination strata (:data:`src.capability.evaluation.generated_phenotype.STRATA`);
* one **whole** Swiss-Prot record per generated sequence, matched on length, which
  is the comparator the E12 work showed a length-matched *fragment* comparator
  flatters generation against;
* the fixed E17 candidate pool, frozen here, before any structure is predicted
  and before any likelihood is computed.

Freezing the pool in one artefact, with a digest, is what makes E17 a selection
experiment rather than a post-hoc story: every selector later ranks exactly these
members, and every method is compared at the same selected-set size on them.

This stage reads frozen ledgers and a FASTA. It loads no model and needs no GPU;
it accepts ``--device`` because the campaign queue injects it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.evaluation import generated_phenotype as gp  # noqa: E402

COMPLETION = "evaluation_cohort.json"
SCHEMA_VERSION = "d1_generated_evaluation_cohort_v1"

#: The E14 arms, declared before any evaluation. Chosen to cover every model
#: category the panel distinguishes -- pure-protein at residue and at subword
#: tokenisation, joint, and text-to-protein adapted -- rather than to maximise
#: any yield. Arms whose stratum support falls below the package's eight-unit
#: bootstrap floor stay in the roster and are recorded as unsupported, because
#: "this arm has no support in this stratum" is a finding about the arm.
E14_ARMS: tuple[str, ...] = (
    "protgpt2",
    "progen3-3b",
    "rita-xl",
    "proteinglm-7b-clm",
    "progen2-medium",
    "galactica-6.7b",
    "instructprotein",
    "prollama",
    "prollama-stage-1",
)

#: The E17 pool arms and the generation condition each pool is drawn from. Both
#: are panel members, so the generating model's own likelihood can be recomputed
#: through the one audited scoring door. ProtGPT2 is unconditioned and subword
#: tokenised; ZymCTRL is EC-conditioned and residue tokenised, and carries the
#: largest conditional yield this programme measured, so a selection gain on it
#: is being asked for on top of an already strong pool.
E17_POOL: dict[str, str] = {"protgpt2": "unconditioned", "zymctrl": "requested"}

#: The independence unit each pool resamples on. ZymCTRL's attempts are nested
#: in the EC class they were requested under, so the class is the unit; an
#: unconditioned pool has no such nesting and resamples on the attempt.
E17_POOL_UNIT: dict[str, str] = {"protgpt2": "id", "zymctrl": "class_key"}


def read_cells(cells: Path, *, conditions: tuple[str, ...]) -> list[dict[str, Any]]:
    directories = sorted(
        path
        for path in cells.iterdir()
        if path.is_dir() and path.name.rsplit("__", 1)[-1] in conditions
    )
    if not directories:
        raise SystemExit(f"{cells} holds no generation cell for conditions {conditions}")
    rows: list[dict[str, Any]] = []
    for directory in directories:
        ledger = directory / "attempts.jsonl"
        if not ledger.is_file():
            raise SystemExit(f"{ledger} is absent; the frozen cell is incomplete")
        with ledger.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                rows.append(
                    {
                        key: record.get(key)
                        for key in (
                            "id", "arm", "campaign", "condition", "class_key", "length",
                            "sequence", "sequence_sha256", "decoder_stop", "valid_aa20",
                            "exact_duplicate_group",
                        )
                    }
                )
    if not rows:
        raise SystemExit(f"{cells} yielded no attempt")
    return rows


def natural_population(
    fasta: Path, *, min_length: int, max_length: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Whole Swiss-Prot records of canonical composition, in the needed length band.

    A record carrying a non-canonical residue is dropped rather than repaired:
    the structure instrument refuses it, and a silently edited comparator would
    not be the natural record it is named as.
    """

    from src.capability.core.arms import iter_fasta

    population: list[dict[str, Any]] = []
    allowed = set(gp.AA20)
    # Swiss-Prot carries the same sequence under several accessions (isoforms,
    # species duplicates). The comparator is drawn without replacement so that no
    # natural record prices two generated sequences, and two accessions sharing a
    # sequence would defeat that at the level that matters, since every evaluator
    # here is a function of the sequence alone. The first accession wins.
    seen: set[str] = set()
    duplicates = 0
    for header, sequence in iter_fasta(fasta):
        length = len(sequence)
        if length < min_length or length > max_length:
            continue
        if set(sequence) - allowed:
            continue
        if sequence in seen:
            duplicates += 1
            continue
        seen.add(sequence)
        accession = header.split("|")[1] if header.count("|") >= 2 else header.split()[0]
        population.append(
            {
                "id": f"nat_{accession}",
                "accession": accession,
                "sequence": sequence,
                "length": length,
            }
        )
    if not population:
        raise SystemExit(f"{fasta} holds no whole canonical record in [{min_length}, {max_length}]")
    identifiers = {record["id"] for record in population}
    if len(identifiers) != len(population):
        raise SystemExit(f"{fasta} yields duplicate accessions; the comparator ids would collide")
    return population, {
        "source": str(fasta),
        "length_band": [int(min_length), int(max_length)],
        "n_records": len(population),
        "n_sequence_duplicates_dropped": int(duplicates),
        "kind": "whole canonical Swiss-Prot entries, deduplicated by sequence",
    }


def draw(rows: list[dict[str, Any]], *, cap: int, seed: int) -> list[dict[str, Any]]:
    """A seeded draw without replacement, stable in the attempt identifier."""

    ordered = sorted(rows, key=lambda row: str(row["id"]))
    if len(ordered) <= cap:
        return ordered
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(ordered), size=cap, replace=False)
    return [ordered[int(index)] for index in sorted(chosen)]


def draw_stratified(
    rows: list[dict[str, Any]], *, key: str, cap: int, seed: int
) -> list[dict[str, Any]]:
    """A seeded draw that keeps every value of ``key`` in proportion.

    Pooling a conditional model's attempts and then sampling them uniformly would
    let the draw silently re-weight its requested classes, so a later yield
    difference between methods could be a difference in which classes survived
    sampling.
    """

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    if len(rows) <= cap:
        return sorted(rows, key=lambda row: str(row["id"]))
    chosen: list[dict[str, Any]] = []
    for index, (value, members) in enumerate(sorted(groups.items())):
        share = max(1, int(round(cap * len(members) / len(rows))))
        chosen.extend(draw(members, cap=share, seed=seed + index))
    return sorted(chosen, key=lambda row: str(row["id"]))


def _target(row: dict[str, Any], *, role: str, stratum: str) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "sequence": str(row["sequence"]),
        "length": int(row["length"]),
        "arm": row["arm"],
        "campaign": row["campaign"],
        "condition": row["condition"],
        "class_key": row["class_key"],
        "decoder_stop": row["decoder_stop"],
        "exact_duplicate_group": row["exact_duplicate_group"],
        "role": role,
        "stratum": stratum,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    gp.require_fresh_out(args.out, COMPLETION)
    unconditioned = read_cells(args.cells, conditions=("unconditioned",))
    census = gp.termination_census(unconditioned)
    write_json(args.out / "termination_census.json", census)

    requested = read_cells(args.cells, conditions=("requested",))
    by_arm_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unconditioned + requested:
        by_arm_condition[(str(row["arm"]), str(row["condition"]))].append(row)

    # ---------------------------------------------------------------- E14 draw
    e14_targets: list[dict[str, Any]] = []
    e14_strata: dict[str, Any] = {}
    for index, arm in enumerate(args.arms):
        rows = by_arm_condition.get((arm, "unconditioned"), [])
        if not rows:
            raise SystemExit(f"no unconditioned cell carries arm {arm!r}")
        for offset, stratum in enumerate(sorted(gp.STRATA)):
            eligible = [row for row in rows if gp.classify_outcome(row) == stratum]
            floor = gp.bootstrap_unit_floor(len(eligible))
            record: dict[str, Any] = {
                "n_eligible": len(eligible),
                "unit_floor": floor,
                "band": dict(gp.STRATA[stratum]),
            }
            if floor["degenerate"]:
                record["drawn"] = 0
                record["status"] = "support_below_unit_floor"
                e14_strata[f"{arm}::{stratum}"] = record
                continue
            sample = draw(eligible, cap=args.per_stratum, seed=args.seed + 101 * index + offset)
            e14_targets.extend(_target(row, role="generated", stratum=stratum) for row in sample)
            record["drawn"] = len(sample)
            record["status"] = "drawn"
            # How much of this stratum an absolute-stability predictor could ever
            # reach, had this project one. Recorded per stratum so the reader can
            # check the stability component's irreducible limitation instead of
            # taking it on trust.
            low, high = gp.SMALL_DOMAIN_STABILITY_BAND
            in_band = sum(1 for row in sample if low <= int(row["length"]) <= high)
            record["small_domain_stability_band"] = {
                "band": [low, high],
                "n_in_band": in_band,
                "unit_floor": gp.bootstrap_unit_floor(in_band),
            }
            e14_strata[f"{arm}::{stratum}"] = record
    if not e14_targets:
        raise SystemExit("no arm supplied a stratum above the unit floor; nothing to evaluate")

    # ---------------------------------------------------------------- E17 pool
    pool: list[dict[str, Any]] = []
    e17_pools: dict[str, Any] = {}
    for index, (arm, condition) in enumerate(sorted(args.pool_arms.items())):
        rows = by_arm_condition.get((arm, condition), [])
        if not rows:
            raise SystemExit(f"no {condition!r} cell carries pool arm {arm!r}")
        eligible = [row for row in rows if gp.classify_outcome(row) == "native"]
        unit = E17_POOL_UNIT[arm]
        if unit == "class_key":
            sample = draw_stratified(
                eligible, key="class_key", cap=args.pool_cap, seed=args.seed + 7001 + index
            )
        else:
            sample = draw(eligible, cap=args.pool_cap, seed=args.seed + 7001 + index)
        units = {str(row.get(unit)) for row in sample}
        floor = gp.bootstrap_unit_floor(len(units))
        if floor["degenerate"]:
            raise SystemExit(
                f"pool arm {arm!r} resamples on {unit!r} and offers only {len(units)} "
                f"units: {floor['degenerate_reason']}"
            )
        pool.extend(_target(row, role="pool", stratum="native") for row in sample)
        e17_pools[arm] = {
            "condition": condition,
            "n_eligible": len(eligible),
            "n_pool": len(sample),
            "independence_unit": unit,
            "n_units": len(units),
            "unit_floor": floor,
        }

    # -------------------------------------------------- natural comparator draw
    lengths = [int(row["length"]) for row in e14_targets]
    population, population_report = natural_population(
        args.swissprot,
        min_length=min(lengths) - args.length_tolerance,
        max_length=max(lengths) + args.length_tolerance,
    )
    pairs, match_report = gp.match_natural_records(
        e14_targets, population, tolerance=args.length_tolerance, seed=args.seed + 31
    )

    # The cheap-feature background is fitted on whole natural records, on a
    # seeded draw that is disjoint from nothing in particular: it prices natural
    # composition, and the comparator draw is a sample of the same corpus.
    background_sample = draw(
        [{"id": record["id"], "sequence": record["sequence"]} for record in population],
        cap=args.background_sample,
        seed=args.seed + 53,
    )
    background = gp.residue_background(record["sequence"] for record in background_sample)
    write_json(
        args.out / "residue_background.json",
        {
            "schema_version": SCHEMA_VERSION,
            "background": background,
            "n_records": len(background_sample),
            "source": str(args.swissprot),
            "smoothing": "add-one over all twenty canonical residues",
            "role": (
                "the cheap external feature of E17: nats per residue of a sequence's "
                "composition under natural protein composition, which knows nothing "
                "about order, structure or family"
            ),
        },
    )

    # --------------------------------------------------------- the one cohort
    records: dict[str, dict[str, Any]] = {}

    def add(record: dict[str, Any]) -> None:
        existing = records.get(record["id"])
        if existing is None:
            records[record["id"]] = record
            return
        if existing["sequence"] != record["sequence"]:
            raise SystemExit(f"identifier {record['id']!r} names two different sequences")
        # One attempt can belong to both E14 and the E17 pool. It is folded once
        # and its roles are both recorded, so neither analysis silently loses it.
        existing["roles"] = sorted({*existing.get("roles", [existing["role"]]), record["role"]})

    for target in e14_targets:
        add({**target, "roles": [target["role"]]})
    for member in pool:
        add({**member, "roles": [member["role"]]})
    for pair in pairs:
        natural = pair["natural"]
        add(
            {
                "id": natural["id"],
                "sequence": natural["sequence"],
                "length": int(natural["length"]),
                "arm": None,
                "campaign": None,
                "condition": None,
                "class_key": None,
                "decoder_stop": None,
                "exact_duplicate_group": None,
                "role": "natural",
                "roles": ["natural"],
                "stratum": pair["target"]["stratum"],
                "accession": natural["accession"],
                "pair_id": pair["pair_id"],
            }
        )
    ordered = [records[key] for key in sorted(records)]
    for record in ordered:
        record["sequence_sha256"] = hashlib.sha256(record["sequence"].encode()).hexdigest()
    cohort_path = args.out / "evaluation_cohort.jsonl"
    cohort_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in ordered), encoding="utf-8"
    )

    write_json(
        args.out / "e14_declaration.json",
        {
            "schema_version": SCHEMA_VERSION,
            "question": (
                "do generated proteins show plausible structural, stability-related and "
                "function-related properties, against matched natural proteins?"
            ),
            "arms": list(args.arms),
            "strata": {name: dict(value) for name, value in gp.STRATA.items()},
            "per_arm_stratum": e14_strata,
            "n_generated": len(e14_targets),
            "pairs": [
                {
                    "pair_id": pair["pair_id"],
                    "generated_id": pair["target"]["id"],
                    "natural_id": pair["natural"]["id"],
                    "arm": pair["target"]["arm"],
                    "stratum": pair["target"]["stratum"],
                    "generated_length": pair["target"]["length"],
                    "natural_length": int(pair["natural"]["length"]),
                }
                for pair in pairs
            ],
            "match_quality": match_report,
            "natural_population": population_report,
            "stability_component": gp.STABILITY_UNAVAILABLE,
            "ceiling": gp.CEILING,
        },
    )
    write_json(
        args.out / "e17_declaration.json",
        {
            "schema_version": SCHEMA_VERSION,
            "question": (
                "can the predictive information models learn be used to select better "
                "generated proteins?"
            ),
            "pools": e17_pools,
            "pool_digest": gp.cohort_digest([record for record in ordered if "pool" in record["roles"]]),
            "n_pool": len(pool),
            "selection_fractions": list(gp.SELECTION_FRACTIONS),
            "random_baseline_keys": gp.RANDOM_BASELINE_KEYS,
            "selectors": {
                "random": "a seeded uniform key; the baseline that must be beaten",
                "likelihood": (
                    "the generating model's own mean negative log-likelihood per scored "
                    "token on its own product, recomputed under the arm's native rendering"
                ),
                "composition": (
                    "nats per residue of the sequence's composition under a Swiss-Prot "
                    "unigram background; the cheap external feature"
                ),
                "combined": "the within-pool average rank of likelihood and composition",
            },
            "independence": gp.declared_independence(
                ["random", "likelihood", "composition", "combined"],
                ["esmfold2_ca_plddt", "esmfold2_ptm", "pfam_complete_domain"],
            ),
            "validity_conditions": [
                "the pool is fixed by this artefact and its digest before any evaluation",
                "methods are compared at equal selected-set size, never at equal threshold",
                "the yield is reported as a curve over selection fraction, not at one point",
                "the repertoire of the selected set is reported beside every yield",
            ],
        },
    )

    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cohort_jsonl": str(cohort_path),
        "cohort_sha256": sha256_file(cohort_path),
        "cohort_digest": gp.cohort_digest(ordered),
        "n_records": len(ordered),
        "n_by_role": dict(Counter(role for item in ordered for role in item["roles"])),
        "n_by_stratum": dict(Counter(str(item["stratum"]) for item in ordered)),
        "length_summary": {
            "min": min(int(item["length"]) for item in ordered),
            "max": max(int(item["length"]) for item in ordered),
            "mean": float(np.mean([int(item["length"]) for item in ordered])),
        },
        "sources": {
            "cells": str(args.cells),
            "swissprot": str(args.swissprot),
            "n_unconditioned_attempts": len(unconditioned),
            "n_requested_attempts": len(requested),
            "n_natural_population": len(population),
        },
        "seed": int(args.seed),
        "termination_census": {
            key: census[key]
            for key in ("n_attempts", "n_cells", "budget_censored_fraction", "budget_censored_per_cell_mean")
        },
        "e14": {"n_generated": len(e14_targets), "n_pairs": len(pairs), "match_quality": match_report},
        "e17": {"n_pool": len(pool), "pools": e17_pools},
        "stability_component": gp.STABILITY_UNAVAILABLE,
    }
    write_json(args.out / COMPLETION, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        type=Path,
        default=REPO_ROOT / "results/R6/generation_replication_20260927/cells",
        help="the frozen generation replication cell directory",
    )
    parser.add_argument(
        "--swissprot",
        type=Path,
        default=None,
        help="whole-record Swiss-Prot FASTA; defaults to the SWISSPROT_FASTA location",
    )
    parser.add_argument("--arms", nargs="+", default=list(E14_ARMS))
    parser.add_argument("--per-stratum", type=int, default=150)
    parser.add_argument("--pool-cap", type=int, default=600)
    parser.add_argument("--length-tolerance", type=int, default=2)
    parser.add_argument("--background-sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--device", default="cpu", help="accepted because the campaign queue injects it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.per_stratum < 1 or args.pool_cap < 1:
        parser.error("--per-stratum and --pool-cap must be positive")
    if args.length_tolerance < 0:
        parser.error("--length-tolerance is non-negative")
    if args.swissprot is None:
        from src.capability.core.arms import SWISSPROT_FASTA

        args.swissprot = SWISSPROT_FASTA
    args.pool_arms = dict(E17_POOL)
    record = run(args)
    print(json.dumps({key: record[key] for key in ("status", "n_records", "cohort_digest")}, sort_keys=True))


if __name__ == "__main__":
    main()
