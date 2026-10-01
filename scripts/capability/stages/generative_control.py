#!/usr/bin/env python3
"""Matched statistical generators and the profile oracle for the generation gate.

Four phases, each resumable and none of which loads a model.

``pool``      One seeded pass over the staged UniRef50 FASTA, keeping a
              reservoir of canonical records per length stratum together with
              each stratum's population count. The pass reports its own residue
              and record totals, which the staged k-mer background counted
              independently, so the draw is checkable rather than asserted.

``build``     For every attempt of every admitted generation ledger, one
              sequence from each matched generator at that attempt's own length.
              Unsearchable attempts get an empty control and stay in the
              denominator. Writes the control ledger and the oracle shards.

``annotate``  The programme's own profile oracle -- HMMER against the staged
              Pfam-A at the release's gathering thresholds -- over every shard,
              retaining the sequence table and the domain table from one call.

``qualify``   The order profile of every cohort, and the pre-declared
              competence receipt for each matched generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.capability.generation import generative_control as gc  # noqa: E402
from src.capability.core.amino_acids import AA20  # noqa: E402

#: Length strata for the natural and fragment draws. The first four are the
#: generation-biology preregistration's own strata; the two above them exist
#: because a fragment donor must be at least as long as the attempt it is
#: matched to and the longest retained generation is 1,457 residues.
STRATA: tuple[tuple[str, int, int], ...] = (
    ("1_15", 1, 15),
    ("16_128", 16, 128),
    ("129_256", 129, 256),
    ("257_512", 257, 512),
    ("513_1024", 513, 1024),
    ("1025_2048", 1025, 2048),
    ("2049_up", 2049, 10 ** 9),
)

#: Reservoir size per stratum. 20,000 records per stratum is two orders of
#: magnitude above the largest per-stratum demand of one arm's 800 attempts, so
#: a donor is drawn with replacement from a pool no attempt exhausts.
RESERVOIR = 20_000

COHORTS: tuple[str, ...] = (
    "generated", "shuffle", "markov_0", "markov_2", "markov_4",
    "fragment", "hydropathy", "natural",
)

_CANONICAL = set(AA20)


def digest_seed(*parts: object) -> int:
    """A per-attempt, per-cohort seed that does not depend on iteration order."""

    payload = "|".join([str(gc.GATE_SEED), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stratum_of(length: int) -> str:
    for name, low, high in STRATA:
        if low <= length <= high:
            return name
    raise ValueError(f"length {length} falls in no stratum")


def stratum_bounds() -> dict[str, tuple[int, int]]:
    return {name: (low, high) for name, low, high in STRATA}


# --------------------------------------------------------------------------
# phase: pool
# --------------------------------------------------------------------------


def phase_pool(args: argparse.Namespace) -> dict:
    from src.capability.core import arms

    location = arms.corpus_location("uniref50", path=args.uniref50)
    rng = np.random.default_rng(digest_seed("uniref50_reservoir"))
    reservoirs: dict[str, list[str]] = {name: [] for name, _, _ in STRATA}
    seen: Counter = Counter()
    population: Counter = Counter()
    records = 0
    canonical_residues = 0
    all_residues = 0
    noncanonical_records = 0
    for _, sequence in arms.iter_fasta(location):
        records += 1
        all_residues += len(sequence)
        if not sequence or not set(sequence) <= _CANONICAL:
            noncanonical_records += 1
            # Twenty C-level counts rather than a Python scan of every residue:
            # the same number, and the difference across 17 billion residues is
            # the difference between a ten-minute pass and an hour-long one.
            canonical_residues += sum(sequence.count(residue) for residue in AA20)
            continue
        canonical_residues += len(sequence)
        name = stratum_of(len(sequence))
        population[name] += 1
        seen[name] += 1
        pool = reservoirs[name]
        if len(pool) < RESERVOIR:
            pool.append(sequence)
        else:
            position = int(rng.integers(0, seen[name]))
            if position < RESERVOIR:
                pool[position] = sequence
        if args.limit_records and records >= args.limit_records:
            break
    receipt = {
        "schema": "d1_generative_control_pool_v1",
        "corpus": "uniref50",
        "corpus_bytes": location.stat().st_size,
        "seed": gc.GATE_SEED,
        "reservoir_per_stratum": RESERVOIR,
        "records_streamed": records,
        "residues_streamed_all_symbols": all_residues,
        "residues_streamed_canonical_symbols": canonical_residues,
        "records_with_any_noncanonical_symbol": noncanonical_records,
        "canonical_record_population": dict(population),
        "reservoir_sizes": {name: len(pool) for name, pool in reservoirs.items()},
        "limit_records": args.limit_records,
        "draw_rule": ("reservoir sampling per length stratum over the whole corpus "
                      "under one declared seed; not a head-of-file prefix"),
        "canonical_residue_cross_check": (
            "the staged uniref50 k-mer background counted 17,277,105,157 canonical "
            "residues over 60,315,044 records in an independent pass"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reservoir.json").write_text(json.dumps(
        {"strata": {name: pool for name, pool in reservoirs.items()},
         "population": dict(population)}), encoding="utf-8")
    receipt["reservoir_sha256"] = sha256_file(args.out / "reservoir.json")
    (args.out / "pool_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


# --------------------------------------------------------------------------
# phase: build
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def discover_ledgers(args: argparse.Namespace) -> list[dict]:
    """The admitted generation ledgers, with the condition each one contributes.

    Every ledger is an already frozen artefact of a completed campaign. Nothing
    here regenerates a sequence, and nothing selects an attempt on its outcome:
    the conditioned arms are subsampled because their requested condition holds
    3,200 attempts against the 800 of an unconditioned arm, and the subsample is
    drawn from the class-stratified population under the gate seed with no
    reference to any profile hit.
    """

    ledgers: list[dict] = []
    for directory in sorted(Path(args.expansion_analysis).iterdir()):
        ledger = directory / "grouped_annotated_attempts.jsonl"
        if ledger.is_file():
            ledgers.append({"arm": directory.name, "condition": "unconditioned",
                            "campaign": "EXP-R2-246", "path": ledger,
                            "population": None, "subsample": None})
    if args.progen3_ledger:
        ledgers.append({"arm": "progen3-3b", "condition": "unconditioned",
                        "campaign": "EXP-R2-233", "path": Path(args.progen3_ledger),
                        "population": None, "subsample": None})
    if args.r232_attempts:
        ledgers.append({"arm": "zymctrl", "condition": "requested",
                        "campaign": "EXP-R2-232", "path": Path(args.r232_attempts),
                        "population": 3200, "subsample": args.conditioned_subsample})
        ledgers.append({"arm": "prollama", "condition": "requested",
                        "campaign": "EXP-R2-232", "path": Path(args.r232_attempts),
                        "population": 3200, "subsample": args.conditioned_subsample})
    return ledgers


def select_rows(spec: dict) -> tuple[list[dict], dict]:
    rows = read_jsonl(spec["path"])
    rows = [row for row in rows
            if row.get("role") in (None, "generation")
            and row.get("arm") == spec["arm"]
            and (spec["condition"] == "unconditioned"
                 or row.get("condition") == spec["condition"])]
    if spec["condition"] == "unconditioned":
        rows = [row for row in rows if row.get("condition") == "unconditioned"]
    accounting = {"population": len(rows), "selected": len(rows),
                  "inclusion_probability": 1.0,
                  "selection_rule": "every attempt of the frozen ledger"}
    if spec["subsample"] and len(rows) > spec["subsample"]:
        by_class: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_class[str(row.get("class_key"))].append(row)
        rng = np.random.default_rng(digest_seed("conditioned_subsample", spec["arm"]))
        per_class = spec["subsample"] // len(by_class)
        remainder = spec["subsample"] - per_class * len(by_class)
        chosen: list[dict] = []
        for index, key in enumerate(sorted(by_class)):
            pool = sorted(by_class[key], key=lambda row: row["id"])
            take = per_class + (1 if index < remainder else 0)
            take = min(take, len(pool))
            picks = rng.choice(len(pool), size=take, replace=False)
            chosen.extend(pool[int(position)] for position in sorted(picks))
        accounting = {
            "population": len(rows), "selected": len(chosen),
            "inclusion_probability": len(chosen) / len(rows),
            "selection_rule": ("class-stratified draw without replacement under the "
                               "gate seed; consults class identity and attempt id "
                               "only, never a profile hit, a reference distance or a "
                               "predicted score"),
            "classes": len(by_class),
        }
        rows = sorted(chosen, key=lambda row: row["id"])
    return rows, accounting


def control_record(cell: str, spec: dict, row: dict, *, samplers: dict,
                   composition: np.ndarray, donor, natural_pool: dict) -> dict:
    """One attempt's row: the parent, one sequence per matched generator, provenance.

    Reads the attempt's sequence, its length, its frozen near-duplicate group and
    its reference distance. It does not read the attempt's profile outcome, and
    none of the generators is a function of it: the model's own recognition is
    carried through as a recorded field so that the analysis can check the
    oracle replays, never as an input to which control the attempt receives.

    An attempt whose output no oracle could search gets an empty sequence in
    every cohort and stays in the row set. That is the honest denominator: the
    model produced nothing searchable, the matched control for nothing is
    nothing, and the pair is a non-recognition on both sides rather than a
    dropped attempt.
    """

    bounds = stratum_bounds()
    sequence = row["sequence"] if row.get("valid_aa20") and row.get("sequence") else ""
    searchable = bool(sequence) and set(sequence) <= _CANONICAL
    length = len(sequence) if searchable else 0
    record: dict = {
        "cell": cell, "arm": spec["arm"], "condition": spec["condition"],
        "campaign": spec["campaign"], "attempt_id": row["id"],
        "class_key": row.get("class_key"),
        "near_duplicate_group": row["near_duplicate_group"],
        "parent_length": length, "parent_searchable": searchable,
        "ledger_any_profile_hit": bool(row.get("any_profile_hit")),
        "ledger_pfam_families": row.get("pfam_families") or [],
        "reference_identity": row.get("reference_identity"),
        "reference_search_status": row.get("reference_search_status"),
        "stop_status": row.get("stop_status") or row.get("source_stop_reason"),
        "sequences": {}, "provenance": {},
    }
    if not searchable:
        record["sequences"] = {cohort: "" for cohort in COHORTS}
        record["provenance"]["reason"] = (
            "the attempt carries no canonical searchable sequence, so no matched "
            "control is defined; the attempt stays in every denominator as a "
            "non-recognition on both sides")
        return record
    record["sequences"]["generated"] = sequence
    record["sequences"]["shuffle"] = gc.composition_shuffle(
        np.random.default_rng(digest_seed(cell, row["id"], "shuffle")), sequence)
    for order in gc.MARKOV_ORDERS:
        record["sequences"][f"markov_{order}"] = samplers[order].emit(
            np.random.default_rng(digest_seed(cell, row["id"], f"markov_{order}")), length)
    fragment_rng = np.random.default_rng(digest_seed(cell, row["id"], "fragment"))
    donor_stratum, donor_record = donor(fragment_rng, length)
    record["sequences"]["fragment"] = gc.corpus_fragment(fragment_rng, donor_record, length)
    record["provenance"]["fragment_donor_stratum"] = donor_stratum
    record["provenance"]["fragment_donor_length"] = len(donor_record)
    record["provenance"]["fragment_donor_sha256"] = hashlib.sha256(
        donor_record.encode()).hexdigest()
    target = gc.mean_hydropathy(sequence)
    emitted, lam, reachable = gc.hydropathy_matched(
        np.random.default_rng(digest_seed(cell, row["id"], "hydropathy")),
        composition, target, length)
    record["sequences"]["hydropathy"] = emitted
    record["provenance"]["hydropathy_target_kd"] = target
    record["provenance"]["hydropathy_tilt"] = lam
    record["provenance"]["hydropathy_target_reachable"] = reachable
    natural_stratum = stratum_of(length)
    natural_rng = np.random.default_rng(digest_seed(cell, row["id"], "natural"))
    candidates = natural_pool[natural_stratum]
    record["sequences"]["natural"] = candidates[int(natural_rng.choice(len(candidates)))]
    record["provenance"]["natural_stratum"] = natural_stratum
    record["provenance"]["natural_band"] = list(bounds[natural_stratum])
    return record


def phase_build(args: argparse.Namespace) -> dict:
    pool = json.loads((Path(args.pool) / "reservoir.json").read_text(encoding="utf-8"))
    reservoirs = {name: pool["strata"][name] for name, _, _ in STRATA}
    population = pool["population"]
    reservoir_lengths = {name: np.array([len(record) for record in records], dtype=np.int64)
                         for name, records in reservoirs.items()}

    counts = gc.load_kmer_counts(Path(args.kmer_background),
                                 [1, *[order + 1 for order in gc.MARKOV_ORDERS]])
    composition = np.asarray(counts[1], dtype=float)
    composition = composition / composition.sum()
    samplers = {order: gc.markov_from_counts(counts, order) for order in gc.MARKOV_ORDERS}

    def donor(rng: np.random.Generator, length: int) -> tuple[str, str]:
        """A corpus record of at least ``length`` residues, drawn corpus-wide."""

        weights = []
        names = []
        for name, _, _ in STRATA:
            lengths = reservoir_lengths[name]
            if lengths.size == 0 or population.get(name, 0) == 0:
                continue
            share = float((lengths >= length).mean())
            if share <= 0:
                continue
            names.append(name)
            weights.append(population[name] * share)
        if not names:
            raise ValueError(f"no corpus record reaches {length} residues in the pool")
        probabilities = np.array(weights, dtype=float)
        probabilities /= probabilities.sum()
        name = names[int(rng.choice(len(names), p=probabilities))]
        eligible = np.flatnonzero(reservoir_lengths[name] >= length)
        return name, reservoirs[name][int(rng.choice(eligible))]

    workspace = Path(args.out)
    (workspace / "cells").mkdir(parents=True, exist_ok=True)
    (workspace / "shards").mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    oracle_records: list[tuple[str, str]] = []
    for spec in discover_ledgers(args):
        rows, accounting = select_rows(spec)
        if not rows:
            raise ValueError(f"{spec['arm']}/{spec['condition']} selected no attempt")
        cell = f"{spec['arm']}__{spec['condition']}"
        controls = [control_record(cell, spec, row, samplers=samplers,
                                   composition=composition, donor=donor,
                                   natural_pool=reservoirs)
                    for row in rows]
        cell_path = workspace / "cells" / f"{cell}.jsonl"
        cell_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n"
                                     for record in controls), encoding="utf-8")
        manifest.append({"cell": cell, "arm": spec["arm"], "condition": spec["condition"],
                         "campaign": spec["campaign"],
                         "ledger": str(spec["path"]),
                         "ledger_sha256": sha256_file(spec["path"]),
                         "attempts": len(controls),
                         "searchable_attempts": sum(1 for r in controls if r["parent_searchable"]),
                         "accounting": accounting,
                         "cell_sha256": sha256_file(cell_path)})
        for record in controls:
            for cohort in COHORTS:
                sequence = record["sequences"].get(cohort, "")
                if sequence:
                    oracle_records.append((f"{record['cell']}|{cohort}|{record['attempt_id']}",
                                           sequence))

    oracle_records.sort()
    shard_size = max(1, args.shard_size)
    index: list[dict] = []
    names: dict[str, str] = {}
    for position, (key, _) in enumerate(oracle_records):
        names[key] = f"q{position:08d}"
    for shard, start in enumerate(range(0, len(oracle_records), shard_size)):
        block = oracle_records[start:start + shard_size]
        path = workspace / "shards" / f"shard_{shard:05d}.fasta"
        path.write_text("".join(f">{names[key]}\n{sequence}\n" for key, sequence in block),
                        encoding="utf-8")
        index.append({"shard": shard, "path": str(path), "n": len(block),
                      "sha256": sha256_file(path)})
    (workspace / "query_names.json").write_text(
        json.dumps(names, sort_keys=True), encoding="utf-8")
    summary = {"schema": "d1_generative_control_build_v1", "seed": gc.GATE_SEED,
               "cohorts": list(COHORTS), "markov_orders": list(gc.MARKOV_ORDERS),
               "cells": manifest, "n_oracle_queries": len(oracle_records),
               "shards": index,
               "kmer_background": str(args.kmer_background),
               "pool_receipt_sha256": sha256_file(Path(args.pool) / "pool_receipt.json")}
    (workspace / "build_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"cells": len(manifest), "queries": len(oracle_records), "shards": len(index)}


# --------------------------------------------------------------------------
# phase: annotate
# --------------------------------------------------------------------------


def oracle_command(hmmscan: Path, pfam: Path, fasta: Path, tbl: Path, domtbl: Path,
                   threads: int) -> list[str]:
    """The programme's own oracle call, with the domain table also retained.

    Identical to :func:`src.capability.concept_injection.run_hmmscan` in tool,
    database, threshold and the absence of any masking option. The two additions
    are ``--domtblout``, which retains the profile-side alignment coordinates
    the complete-domain endpoint reads, and ``-o /dev/null``, which discards the
    human-readable report rather than buffering it. Neither changes which
    families a sequence carries, and the endpoint reader checks that every
    domain-table family is one the sequence table carries rather than assuming
    it. The reverse is not required and is counted instead: a family clearing
    GA1 on the full sequence with no single domain clearing GA2 appears on the
    sequence table alone, which is what HMMER does and not a disagreement.
    """

    return [str(hmmscan), "--tblout", str(tbl), "--domtblout", str(domtbl),
            "--noali", "--cut_ga", "--cpu", str(threads), "-o", os.devnull,
            str(pfam), str(fasta)]


def _scan(task: tuple[str, str, str, str, str, int]) -> dict:
    hmmscan, pfam, fasta, tbl, domtbl, threads = task
    command = oracle_command(Path(hmmscan), Path(pfam), Path(fasta), Path(tbl),
                             Path(domtbl), threads)
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        return {"fasta": fasta, "status": "failed", "returncode": completed.returncode,
                "stderr": completed.stderr[-2000:]}
    Path(str(tbl) + ".done").write_text(
        json.dumps({"command": command, "fasta_sha256": sha256_file(Path(fasta))}),
        encoding="utf-8")
    return {"fasta": fasta, "status": "ok"}


def phase_annotate(args: argparse.Namespace) -> dict:
    workspace = Path(args.workspace)
    manifest = json.loads((workspace / "build_manifest.json").read_text(encoding="utf-8"))
    oracle = workspace / "oracle"
    oracle.mkdir(parents=True, exist_ok=True)
    tasks = []
    reused = 0
    for shard in manifest["shards"]:
        fasta = Path(shard["path"])
        tbl = oracle / (fasta.stem + ".tbl")
        domtbl = oracle / (fasta.stem + ".domtbl")
        marker = Path(str(tbl) + ".done")
        if marker.is_file() and tbl.is_file() and domtbl.is_file():
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if saved.get("fasta_sha256") == shard["sha256"]:
                reused += 1
                continue
        tasks.append((str(args.hmmscan), str(args.pfam), str(fasta), str(tbl),
                      str(domtbl), args.threads))
    failures: list[dict] = []
    done = 0
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_scan, task) for task in tasks]
            for future in as_completed(futures):
                outcome = future.result()
                if outcome["status"] != "ok":
                    failures.append(outcome)
                else:
                    done += 1
                    if done % 16 == 0:
                        print(f"{done}/{len(tasks)} shards scanned", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} oracle shards failed: {failures[:3]}")
    return {"shards_total": len(manifest["shards"]), "shards_scanned": done,
            "shards_reused": reused}


# --------------------------------------------------------------------------
# phase: qualify
# --------------------------------------------------------------------------


def phase_qualify(args: argparse.Namespace) -> dict:
    workspace = Path(args.workspace)
    needed = sorted({order + 1 for order in gc.MARKOV_ORDERS}
                    | {order + 2 for order in gc.MARKOV_ORDERS})
    counts = gc.load_kmer_counts(Path(args.kmer_background), needed)
    cohort_sequences: dict[str, list[str]] = {cohort: [] for cohort in COHORTS}
    parents: list[str] = []
    shuffles: list[str] = []
    fragment_sources: list[str] = []
    fragment_controls: list[str] = []
    fragment_lengths: list[int] = []
    hydropathy_controls: list[str] = []
    hydropathy_targets: list[float] = []
    hydropathy_lengths: list[int] = []
    natural_controls: list[str] = []
    natural_strata: list[str] = []
    donor_index: dict[str, str] = {}
    pool = json.loads((Path(args.pool) / "reservoir.json").read_text(encoding="utf-8"))
    for name, _, _ in STRATA:
        for record in pool["strata"][name]:
            donor_index[hashlib.sha256(record.encode()).hexdigest()] = record

    for cell in sorted((workspace / "cells").glob("*.jsonl")):
        for record in read_jsonl(cell):
            if not record["parent_searchable"]:
                continue
            for cohort in COHORTS:
                cohort_sequences[cohort].append(record["sequences"][cohort])
            parents.append(record["sequences"]["generated"])
            shuffles.append(record["sequences"]["shuffle"])
            fragment_controls.append(record["sequences"]["fragment"])
            fragment_lengths.append(record["parent_length"])
            fragment_sources.append(
                donor_index[record["provenance"]["fragment_donor_sha256"]])
            hydropathy_controls.append(record["sequences"]["hydropathy"])
            hydropathy_targets.append(record["provenance"]["hydropathy_target_kd"])
            hydropathy_lengths.append(record["parent_length"])
            natural_controls.append(record["sequences"]["natural"])
            natural_strata.append(record["provenance"]["natural_stratum"])

    orders = sorted({order for order in gc.MARKOV_ORDERS}
                    | {order + 1 for order in gc.MARKOV_ORDERS})
    profiles: dict[str, dict[int, float]] = {cohort: {} for cohort in COHORTS}
    scored: dict[str, dict[int, int]] = {cohort: {} for cohort in COHORTS}
    for order in orders:
        if order + 1 not in counts:
            continue
        table = gc.conditional_table(counts, order)
        for cohort in COHORTS:
            value, n = gc.conditional_log_likelihood(cohort_sequences[cohort], table, order)
            profiles[cohort][order] = value
            scored[cohort][order] = n
        del table

    receipts = [gc.qualify_shuffle(parents, shuffles)]
    for order in gc.MARKOV_ORDERS:
        # The corpus reference is the fragment cohort, not the natural one: a
        # fragment is a corpus draw at the *attempt's own length*, so comparing
        # against it removes the length-distribution difference between whole
        # records and generated products from the order profile.
        receipts.append(gc.qualify_markov(profiles[f"markov_{order}"],
                                          profiles["fragment"], order))
    receipts.append(gc.qualify_fragment(fragment_controls, fragment_sources, fragment_lengths))
    receipts.append(gc.qualify_hydropathy(hydropathy_controls, hydropathy_targets,
                                          hydropathy_lengths))
    receipts.append(gc.qualify_natural(natural_controls, natural_strata, stratum_bounds()))

    result = {
        "schema": "d1_generative_control_qualification_v1",
        "seed": gc.GATE_SEED,
        "n_searchable_parents": len(parents),
        "order_profile_nats_per_residue": {cohort: {str(k): v for k, v in profile.items()}
                                           for cohort, profile in profiles.items()},
        "order_profile_scored_positions": {cohort: {str(k): v for k, v in counts.items()}
                                           for cohort, counts in scored.items()},
        "order_profile_note": (
            "mean log P(residue | previous k residues) in nats per residue under the "
            "staged UniRef50 conditional at order k, add-one smoothed for evaluation. "
            "The corpus reference for the competence conditions is the length-matched "
            "fragment cohort. Both corpus-drawn cohorts contribute ~1.5e-4 of the "
            "residues those counts were built from, which inflates their own profile "
            "slightly and therefore makes the declared-order condition harder to pass "
            "and the order-axis condition easier"),
        "order_reference_cohort": "fragment",
        "receipts": receipts,
        "qualified_cohorts": [receipt["cohort"] for receipt in receipts if receipt["qualified"]],
        "unqualified_cohorts": [receipt["cohort"] for receipt in receipts
                                if not receipt["qualified"]],
        "qualified_cohorts_under_the_scale_free_amendment": [
            receipt["cohort"] for receipt in receipts
            if receipt.get("qualified_under_the_scale_free_amendment", receipt["qualified"])],
        "amendment": (
            "the pre-declared order-axis condition uses an absolute 0.02-nat margin "
            "that the measured adjacent-order step of this corpus does not reach "
            "below order 4; the amendment asks instead whether the cohort captures a "
            "non-positive share of the reference's own gain one order up. The "
            "pre-declared set decides the primary verdict"),
    }
    (workspace / "qualification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"qualified": result["qualified_cohorts"],
            "unqualified": result["unqualified_cohorts"],
            "qualified_amended":
                result["qualified_cohorts_under_the_scale_free_amendment"]}


# --------------------------------------------------------------------------
# phase: profile ceiling
# --------------------------------------------------------------------------


def phase_ceiling(args: argparse.Namespace) -> dict:
    """Sequences emitted from the oracle's own profiles: the endpoint's ceiling.

    Not a matched control and never reported as one. It answers a different
    question: what recognition rate does a generator with direct access to the
    profile database reach, so that a model's rate can be read against an
    attainable maximum instead of against 1.0.
    """

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    stat = subprocess.run([str(args.hmmstat), str(args.pfam)],
                          capture_output=True, text=True, check=True)
    accessions = []
    for line in stat.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if len(fields) > 2:
            accessions.append(fields[2])
    rng = np.random.default_rng(digest_seed("profile_ceiling"))
    picks = sorted({accessions[int(index)] for index in
                    rng.choice(len(accessions), size=min(args.n_profiles, len(accessions)),
                               replace=False)})
    names = workspace / "ceiling_profiles.txt"
    names.write_text("\n".join(picks) + "\n", encoding="utf-8")
    fetched = workspace / "ceiling_profiles.hmm"
    subprocess.run([str(args.hmmfetch), "-o", str(fetched), "-f", str(args.pfam),
                    str(names)], capture_output=True, text=True, check=True)
    emitted = workspace / "ceiling.fasta"
    subprocess.run([str(args.hmmemit), "-N", "1", "--seed", str(gc.GATE_SEED % 2 ** 31),
                    "-o", str(emitted), str(fetched)], capture_output=True, text=True,
                   check=True)
    tbl = workspace / "ceiling.tbl"
    domtbl = workspace / "ceiling.domtbl"
    command = oracle_command(Path(args.hmmscan), Path(args.pfam), emitted, tbl, domtbl,
                             args.threads)
    subprocess.run(command, capture_output=True, text=True, check=True)
    from src.capability.core.protein_annotations import parse_hmmscan_table

    hits = parse_hmmscan_table(tbl)
    lengths = []
    queries = 0
    for line in emitted.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            queries += 1
            lengths.append(0)
        elif lengths:
            lengths[-1] += len(line.strip())
    recognised = sum(1 for name in _fasta_names(emitted) if name in hits)
    domains = gc.parse_domain_table(domtbl)
    complete = sum(1 for name in _fasta_names(emitted)
                   if (gc.best_profile_coverage(domains.get(name, [])) or 0.0)
                   >= gc.COMPLETE_DOMAIN_COVERAGE)
    lower, upper = gc.wilson_interval(recognised, queries)
    complete_lower, complete_upper = gc.wilson_interval(complete, queries)
    result = {
        "schema": "d1_generative_control_ceiling_v1",
        "role": "oracle_access_ceiling_not_a_matched_control",
        "n_profiles_requested": args.n_profiles, "n_sequences": queries,
        "any_family_rate": recognised / queries,
        "any_family_wilson95": [lower, upper],
        "complete_domain_rate": complete / queries,
        "complete_domain_wilson95": [complete_lower, complete_upper],
        "complete_domain_coverage_threshold": gc.COMPLETE_DOMAIN_COVERAGE,
        "median_length": float(np.median(lengths)) if lengths else None,
        "note": ("hmmemit draws from the profiles the oracle itself scores, so this "
                 "rate is the endpoint's attainable maximum and not a generator the "
                 "model is asked to beat; lengths are profile consensus lengths and "
                 "are not matched to any arm"),
    }
    (workspace / "ceiling.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _fasta_names(path: Path) -> list[str]:
    return [line[1:].split()[0] for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith(">")]


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="phase", required=True)

    pool = sub.add_parser("pool")
    pool.add_argument("--uniref50", type=Path)
    pool.add_argument("--out", type=Path, required=True)
    pool.add_argument("--limit-records", type=int, default=0,
                     help="stop after this many records; for interface smokes only")
    pool.set_defaults(run=phase_pool)

    build = sub.add_parser("build")
    build.add_argument("--pool", type=Path, required=True)
    build.add_argument("--kmer-background", type=Path, required=True)
    build.add_argument("--expansion-analysis", type=Path, required=True)
    build.add_argument("--progen3-ledger", type=Path)
    build.add_argument("--r232-attempts", type=Path)
    build.add_argument("--conditioned-subsample", type=int, default=800)
    build.add_argument("--shard-size", type=int, default=460)
    build.add_argument("--out", type=Path, required=True)
    build.set_defaults(run=phase_build)

    annotate = sub.add_parser("annotate")
    annotate.add_argument("--workspace", type=Path, required=True)
    annotate.add_argument("--hmmscan", type=Path, required=True)
    annotate.add_argument("--pfam", type=Path, required=True)
    annotate.add_argument("--workers", type=int, default=32)
    annotate.add_argument("--threads", type=int, default=1)
    annotate.set_defaults(run=phase_annotate)

    qualify = sub.add_parser("qualify")
    qualify.add_argument("--workspace", type=Path, required=True)
    qualify.add_argument("--pool", type=Path, required=True)
    qualify.add_argument("--kmer-background", type=Path, required=True)
    qualify.set_defaults(run=phase_qualify)

    ceiling = sub.add_parser("ceiling")
    ceiling.add_argument("--workspace", type=Path, required=True)
    ceiling.add_argument("--hmmscan", type=Path, required=True)
    ceiling.add_argument("--hmmstat", type=Path, required=True)
    ceiling.add_argument("--hmmfetch", type=Path, required=True)
    ceiling.add_argument("--hmmemit", type=Path, required=True)
    ceiling.add_argument("--pfam", type=Path, required=True)
    ceiling.add_argument("--n-profiles", type=int, default=1000)
    ceiling.add_argument("--threads", type=int, default=4)
    ceiling.set_defaults(run=phase_ceiling)

    args = parser.parse_args()
    print(json.dumps(args.run(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
