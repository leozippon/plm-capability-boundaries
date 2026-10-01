#!/usr/bin/env python3
"""How far a staged endpoint's targets are from anything the corpus retrieves.

The retrieval and memorization gate cannot test the remote-homology end because
both current cohorts are saturated with high-identity units: 49 of 64 stability
groups sit at or above 95% identity to a retrieved homolog, and no readout
cluster falls below 30%. Whether a newly staged endpoint changes that is a
measurement about its targets, not about its variant count, and it is the
measurement this stage makes.

Each query is searched against the local reference index, binned by the identity
of its best hit under the same definition the earlier disjointness certificate
used -- 100 * nident / qlen, the percent of the query identically matched -- and
grouped with the other queries under the frozen grouping contract's own identity
and coverage floors. The reported quantity is the number of independent groups
whose every member is remote, because a group is the unit a held-group inference
can use and a variant is not.

No detected alignment in a searched snapshot is not absence from any pretraining
corpus, and this stage reports a stratified identity distribution rather than
any certificate of exclusion.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import math
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

from src.capability.core.arms import env_path
from src.capability.core.family_grouping import Union, batch_align, encode
from src.capability.context.homology import STRATUM_NAMES, assign_stratum
from src.capability.core.io import write_json

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The grouping contract frozen for the MegaScale backgrounds, reused unchanged
#: so a group count here is commensurable with the 102 groups it produced there.
IDENTITY_FLOOR = 30.0
COVERAGE_FLOOR = 80.0
GAP_OPEN = 11.0
GAP_EXTEND = 1.0

#: The identity bands are the frozen declaration in src.capability.context.homology, not a
#: second copy of it, so a count here is commensurable with the retrieval gate's
#: 1/2/12/49 stability and 0/5/18/140 ranking tables. A query that retrieves
#: nothing bands at identity 0.0, which is what the gate does with it; the no-hit
#: count is reported separately because the band alone does not distinguish a
#: query with a distant hit from one with none.
BANDS = STRATUM_NAMES

#: The declared percentile-interval floor: a stratum below it carries no verdict
#: in either direction, which is why a group count and not a variant count is the
#: quantity this stage reports.
UNIT_FLOOR = 8


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    name = None
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(chunks)
            name = line[1:].strip()
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if name is not None:
        records[name] = "".join(chunks)
    return records


def band(identity: float | None) -> str:
    return assign_stratum(0.0 if identity is None else identity)


def wilson(successes: int, total: int) -> list[float]:
    """95% Wilson interval for a proportion, the sampled counts' own uncertainty."""

    if total == 0:
        return [float("nan"), float("nan")]
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def search(
    diamond: Path, index: Path, queries: Path, hits: Path, threads: int, max_targets: int
) -> None:
    command = [
        str(diamond),
        "blastp",
        "--db", str(index),
        "--query", str(queries),
        "--out", str(hits),
        "--very-sensitive",
        "--masking", "0",
        "--evalue", "1e-3",
        "--max-target-seqs", str(max_targets),
        "--threads", str(threads),
        "--outfmt", "6", "qseqid", "sseqid", "nident", "qlen", "pident", "length", "evalue",
    ]
    subprocess.run(command, check=True)


def best_identity(hits: Path, names: list[str]) -> dict[str, float | None]:
    best: dict[str, float | None] = {name: None for name in names}
    for line in hits.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        query, _, nident, qlen = fields[0], fields[1], int(fields[2]), int(fields[3])
        if query not in best or qlen == 0:
            continue
        identity = 100.0 * nident / qlen
        current = best[query]
        if current is None or identity > current:
            best[query] = identity
    return best


def group(records: dict[str, str]) -> dict[str, list[str]]:
    """Connected components under the frozen identity and coverage floors."""

    names = sorted(records)
    codes, lengths = encode([records[name] for name in names])
    pairs = np.array(
        [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))],
        dtype=np.int64,
    )
    union = Union(names)
    if len(pairs):
        statistics = batch_align(
            codes, lengths, pairs, gap_open=GAP_OPEN, gap_extend=GAP_EXTEND
        )
        accepted = statistics.edges(
            identity_floor=IDENTITY_FLOOR, coverage_floor=COVERAGE_FLOOR
        )
        for (left, right) in pairs[accepted]:
            union.join(names[left], names[right], "alignment")
    return union.components()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fasta",
        type=Path,
        action="append",
        required=True,
        help="query backgrounds; repeatable, one population per file named by its stem",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=env_path("TRANSFER_DIAMOND_DB",
                         REPO_ROOT / "data/homology_db/uniref50_full.dmnd"),
        help="reference DIAMOND index",
    )
    parser.add_argument("--diamond", type=Path, required=True, help="diamond binary")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="query this many backgrounds under --seed; 0 searches every one",
    )
    parser.add_argument("--seed", type=int, default=20260924, help="sampling seed")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--max-targets", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=127, help="drop longer queries")
    parser.add_argument("--out", type=Path, required=True, help="report JSON")
    args = parser.parse_args(argv)

    canonical = set("ACDEFGHIKLMNPQRSTVWY")
    populations: dict[str, dict[str, object]] = {}
    sampled: dict[str, str] = {}
    for path in args.fasta:
        records = read_fasta(path)
        too_long = sorted(n for n, s in records.items() if len(s) > args.max_length)
        records = {n: s for n, s in records.items() if len(s) <= args.max_length}
        noncanonical = sorted(n for n, s in records.items() if set(s) - canonical)
        records = {n: s for n, s in records.items() if not set(s) - canonical}
        population = len(records)
        names = sorted(records)
        if args.sample and args.sample < population:
            names = sorted(random.Random(args.seed).sample(names, args.sample))
        collision = set(names) & set(sampled)
        if collision:
            raise SystemExit(f"{path} repeats query names already read: {sorted(collision)[:5]}")
        sampled.update({name: records[name] for name in names})
        populations[path.stem] = {
            "fasta": str(path),
            "population_after_filters": population,
            "dropped_longer_than_max_length": too_long,
            "dropped_noncanonical_residues": noncanonical,
            "queries_searched": names,
        }

    work = args.out.parent
    work.mkdir(parents=True, exist_ok=True)
    queries = work / f"{args.out.stem}_queries.fasta"
    with queries.open("w", encoding="utf-8") as handle:
        for name in sorted(sampled):
            handle.write(f">{name}\n{sampled[name]}\n")

    hits = work / f"{args.out.stem}_hits.tsv"
    started = time.time()
    search(args.diamond, args.index, queries, hits, args.threads, args.max_targets)
    search_seconds = time.time() - started

    best = best_identity(hits, sorted(sampled))
    bands = {name: band(best[name]) for name in sampled}
    remote_bands = {STRATUM_NAMES[0], STRATUM_NAMES[1]}

    for label, entry in populations.items():
        names = list(entry.pop("queries_searched"))
        counts = {
            key: sum(1 for name in names if bands[name] == key) for key in BANDS
        }
        components = group({name: sampled[name] for name in names})
        group_bands: dict[str, set[str]] = defaultdict(set)
        for root, members in components.items():
            for member in members:
                group_bands[root].add(bands[member])
        remote_count = sum(counts[key] for key in remote_bands)
        no_hit = sum(1 for name in names if best[name] is None)
        entry.update(
            {
                "queries_searched": len(names),
                "sampling": {
                    "sample": args.sample,
                    "seed": args.seed,
                    "is_census": not args.sample
                    or args.sample >= int(entry["population_after_filters"]),
                },
                "queries_per_band": counts,
                "queries_with_no_reported_hit": no_hit,
                "remote_query_proportion": {
                    "successes": remote_count,
                    "total": len(names),
                    "wilson_95": wilson(remote_count, len(names)),
                },
                "grouping": {
                    "groups": len(components),
                    "singletons": sum(1 for m in components.values() if len(m) == 1),
                    "largest_group": max((len(m) for m in components.values()), default=0),
                    "groups_with_every_member_remote": sum(
                        1 for seen in group_bands.values() if seen <= remote_bands
                    ),
                    "groups_with_any_member_remote": sum(
                        1 for seen in group_bands.values() if seen & remote_bands
                    ),
                    "unit_floor": UNIT_FLOOR,
                    "remote_groups_clear_the_unit_floor": sum(
                        1 for seen in group_bands.values() if seen <= remote_bands
                    )
                    >= UNIT_FLOOR,
                },
            }
        )

    report = {
        "schema_version": "endpoint_remoteness_v1",
        "index": str(args.index),
        "search": {
            "settings": "diamond blastp --very-sensitive --masking 0 --evalue 1e-3",
            "max_target_seqs": args.max_targets,
            "threads": args.threads,
            "wall_seconds": round(search_seconds, 1),
            "queries": len(sampled),
            "hit_lines": sum(1 for _ in hits.open(encoding="utf-8")),
        },
        "identity_definition": "100 * nident / qlen over the best hit, the percent of the query identically matched",
        "grouping_contract": (
            f"exact Smith-Waterman, BLOSUM62, gap {GAP_OPEN} + k * {GAP_EXTEND}, "
            f"edge at >= {IDENTITY_FLOOR}% identity and >= {COVERAGE_FLOOR}% coverage of both"
        ),
        "populations": populations,
        "claim_boundary": (
            "A band is the identity of the best hit in the searched index under the "
            "declared settings. No detected alignment is not absence from a pretraining "
            "corpus, and a group here is a conservative held-group control rather than a "
            "certified remote-family holdout."
        ),
    }
    write_json(args.out, report)
    print(json.dumps(report["populations"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
