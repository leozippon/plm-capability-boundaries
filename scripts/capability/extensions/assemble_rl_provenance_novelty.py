#!/usr/bin/env python3
"""The homology instrument of the reward-provenance experiment: two declared families.

Declared in ``docs/rl-environment-provenance.md``. Two of that declaration's
twenty families read distance from the natural record, and they are separate
families because their units differ: mean maximum identity over the query is a
percentage, and the fraction below 30% identity is a share of attempts. A
maximum-statistic band in native units may not span the two.

The search itself is not done here. It is
``scripts/capability/evaluation/search_generated_novelty.py``, unchanged, with the
masking setting, the identity-over-query definition and the four frozen strata
that the campaign already fixed for exactly this question -- so these numbers sit
beside the released DPO pairs' without translation. This producer emits the cohort
that search consumes and reads its per-record output back onto the pools.

One cohort covers every pool, with one record per *distinct* sequence across all
of them. Identity to the corpus is a property of a sequence and not of the pool
that drew it, so searching one sequence twice would buy nothing; the search is
database-scan bound, so one pass over the union costs barely more than one pass
over any single pool.

Usage:
    python ... assemble_rl_provenance_novelty.py cohort --pools <pool dir> --out <dir>
    # then search_generated_novelty.py --cohort <dir>/novelty_cohort.jsonl ...
    python ... assemble_rl_provenance_novelty.py assemble \
        --pools <pool dir> --novelty <search out>/generated_novelty.jsonl --out <dir>
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

from src.capability.context.homology import STRATUM_EDGES, STRATUM_NAMES  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.extensions import rl_environment_provenance as rp  # noqa: E402
from src.capability.extensions import rl_provenance_analysis as ra  # noqa: E402

COHORT = "novelty_cohort.jsonl"

#: The role the search's default ``--roles`` already admits, so the cohort needs
#: no flag of its own.
ROLE = "generated"

#: The identity below which a sequence counts as novel, read off the frozen edges
#: rather than written as a literal.
NOVEL_EDGE = STRATUM_EDGES[1]
NOVEL_STRATUM = STRATUM_NAMES[0]


def pool_rows(pools: Path, pool: str) -> list[dict[str, Any]]:
    path = Path(pools) / f"{pool}_closed_form_statistics.jsonl"
    if not path.is_file():
        raise SystemExit(f"no closed-form statistics for pool {pool!r} at {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def present_pools(pools: Path) -> list[str]:
    return [pool for pool in rp.EVALUATED_POOLS
            if (Path(pools) / f"{pool}_closed_form_statistics.jsonl").is_file()]


def cohort(args: argparse.Namespace) -> int:
    from src.capability.generation import unconditional_generation as ug

    pools = present_pools(args.pools)
    if rp.POOL_BASE not in pools:
        raise SystemExit("the base pool is the control for both families and is required")
    seen: dict[str, dict[str, Any]] = {}
    per_pool: dict[str, int] = {}
    for pool in pools:
        count = 0
        for stream in range(rp.STREAMS):
            directory = Path(args.pools) / f"rlp_{pool}_stream{stream}"
            for row in ug.read_attempts(directory / ug.LEDGER_NAME):
                if not row["sequence"]:
                    continue
                count += 1
                digest = str(row["sequence_sha256"])
                if digest not in seen:
                    seen[digest] = {"id": digest, "sequence": row["sequence"],
                                    "length": len(row["sequence"]), "role": ROLE,
                                    "roles": [ROLE]}
        per_pool[pool] = count
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / COHORT
    with path.open("w", encoding="utf-8") as handle:
        for digest in sorted(seen):
            handle.write(json.dumps(seen[digest], sort_keys=True) + "\n")
    write_json(out / "novelty_cohort.json", {
        "schema": rp.SCHEMA, "campaign": rp.CAMPAIGN, "stage": "novelty-cohort",
        "pools": pools, "non_empty_attempts_per_pool": per_pool,
        "distinct_sequences": len(seen), "role": ROLE,
        "cohort": str(path), "sha256": sha256_file(path),
        "search": "scripts/capability/evaluation/search_generated_novelty.py at "
                  "--masking 0, identity over the query, the four frozen strata",
        "note": "one record per distinct sequence across every pool: identity to the "
                "corpus is a property of the sequence and not of the pool that drew it",
        "written_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({"pools": pools, "distinct_sequences": len(seen),
                      "cohort": str(path)}, indent=1), flush=True)
    return 0


def read_novelty(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("searched"):
            continue
        out[str(row["id"])] = row
    if not out:
        raise SystemExit(f"{path} carries no searched record")
    return out


def assemble(args: argparse.Namespace) -> int:
    annotations = read_novelty(args.novelty)
    identity = {digest: float(row["nearest_corpus_identity"])
                for digest, row in annotations.items()}
    below = {digest: (1.0 if float(row["nearest_corpus_identity"]) < NOVEL_EDGE else 0.0)
             for digest, row in annotations.items()}
    pools = present_pools(args.pools)
    families: dict[str, Any] = {}
    for family, scores, quantity in (
        (rp.FAMILY_IDENTITY_MEAN, identity, "identity over the query"),
        (rp.FAMILY_IDENTITY_FRACTION, below, "novel-stratum membership"),
    ):
        columns = []
        for pool in rp.CONTRAST_POOLS:
            if pool not in pools or rp.POOL_BASE not in pools:
                columns.append({"key": pool,
                                "skip_reason": f"pool {pool!r} has not been generated"})
                continue
            try:
                left = ra.streams_by_digest(pool_rows(args.pools, pool), scores,
                                            pool=pool, quantity=quantity)
                right = ra.streams_by_digest(pool_rows(args.pools, rp.POOL_BASE), scores,
                                             pool=rp.POOL_BASE, quantity=quantity)
            except ValueError as error:
                columns.append({"key": pool, "skip_reason": str(error)})
                continue
            columns.append({"key": pool, "left": left, "right": right})
        families[family] = ra.contrast_bands(columns, family=family)

    strata: dict[str, dict[str, float]] = {}
    for pool in pools:
        rows = pool_rows(args.pools, pool)
        counts = {name: 0 for name in STRATUM_NAMES}
        total = 0
        for row in rows:
            annotation = annotations.get(str(row["sequence_sha256"]))
            if annotation is None:
                continue
            counts[str(annotation["identity_stratum"])] += 1
            total += 1
        strata[pool] = {name: (counts[name] / total if total else 0.0)
                        for name in STRATUM_NAMES} | {"n": total}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "novelty_families.json", families)
    write_json(out / "novelty_strata.json", {
        "schema": rp.SCHEMA, "campaign": rp.CAMPAIGN, "stage": "novelty-assemble",
        "strata": strata, "stratum_names": list(STRATUM_NAMES),
        "stratum_edges": list(STRATUM_EDGES),
        "novel_stratum": NOVEL_STRATUM, "novel_edge": NOVEL_EDGE,
        "searched_sequences": len(annotations),
        "novelty_table": str(args.novelty),
        "novelty_table_sha256": sha256_file(args.novelty),
        "limitation": (
            "identity to a UniRef50 snapshot is similarity to a named snapshot and not "
            "pretraining exposure. The ProtGPT3 family's declared training mixture is "
            "UniRef90 and GigaRef, which the searched snapshot does not cover, so the "
            "measured identity is biased and the bias is named rather than corrected"
        ),
        "written_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({family: {"resolved": record["resolved_family_size"],
                               "declared": record["declared_family_size"]}
                      for family, record in families.items()}
                     | {"strata": {pool: round(values[NOVEL_STRATUM], 4)
                                   for pool, values in strata.items()}}, indent=1),
          flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    build = sub.add_parser("cohort")
    build.add_argument("--pools", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    read = sub.add_parser("assemble")
    read.add_argument("--pools", type=Path, required=True)
    read.add_argument("--novelty", type=Path, required=True)
    read.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"cohort": cohort, "assemble": assemble}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
