#!/usr/bin/env python3
"""Retrieve aligned homolog rows for the MegaScale WT backgrounds.

The retained disjointness hit table is a prior retrieval certificate: it caps at
100 hits per query and retains no aligned strings, so it cannot support a
coupling fit. This search asks for the aligned fields and a depth ceiling high
enough that the ceiling itself is reportable, using the same command builder,
field list and parser as every other search in this repository.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.capability.context.homology import (
    ALIGNMENT_FIELDS, DiamondDatabase, DiamondTool, run_diamond_blastp)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            sha.update(chunk)
    return sha.hexdigest()


def dbinfo(executable: Path, database: Path) -> tuple[int, int, str]:
    completed = subprocess.run([str(executable), "dbinfo", "--db", str(database)],
                               capture_output=True, text=True, check=True)
    values = {}
    for key in ("Sequences", "Letters"):
        for line in completed.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == key:
                values[key] = int(parts[1])
    if len(values) != 2:
        raise RuntimeError("cannot read Sequences/Letters from `diamond dbinfo`")
    return values["Sequences"], values["Letters"], completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diamond", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=160)
    parser.add_argument("--evalue", type=float, default=1e-3)
    parser.add_argument("--max-target-seqs", type=int, default=10000)
    parser.add_argument("--sensitivity", default="very-sensitive")
    parser.add_argument("--source-fasta", type=Path,
                        help="the FASTA this index was built from, when the host "
                             "retains it; whether it is on disk is then measured "
                             "rather than asserted")
    args = parser.parse_args()

    # Retention is a provenance fact about this host, so it is measured here and
    # a declared source that is absent is a failure rather than a false report.
    source_fasta_retained = args.source_fasta is not None and args.source_fasta.is_file()
    if args.source_fasta is not None and not source_fasta_retained:
        raise FileNotFoundError(
            f"{args.source_fasta}: declared as this index's retained source FASTA "
            "but not a file on disk; the manifest states retention as measured")

    hits = args.out_dir / "hits.tsv"
    if hits.exists():
        raise FileExistsError(hits)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    version = subprocess.run([str(args.diamond), "--version"], capture_output=True,
                             text=True, check=True).stdout.strip()
    tool = DiamondTool(executable=args.diamond, version=version,
                       binary_sha256=digest(args.diamond), tarball=args.diamond,
                       tarball_sha256=digest(args.diamond))
    sequences, letters, info = dbinfo(args.diamond, args.database)
    # source_records is set from the index itself, so no coverage fraction against
    # the source FASTA is claimed here even when that FASTA is retained: counting
    # its records is a separate pass over tens of gigabytes and nothing in this
    # search reads the fraction.
    database = DiamondDatabase(path=args.database, source_fasta=args.database,
                               source_records=sequences, sequences=sequences,
                               letters=letters, makedb_command=("not-retained",))

    entries = json.loads(args.catalogue.read_text())
    query = args.out_dir / "query.faa"
    query.write_text("".join(
        f">{row['WT_name']}\n{row['sequence']}\n" for row in
        sorted(entries, key=lambda row: row["WT_name"])))

    command, log = run_diamond_blastp(
        tool, database, query, hits, threads=args.threads,
        sensitivity=args.sensitivity, evalue=args.evalue,
        max_target_seqs=args.max_target_seqs, fields=ALIGNMENT_FIELDS)
    (args.out_dir / "search_manifest.json").write_text(json.dumps({
        "schema": "pairwise_homolog_search_v1",
        "tool": tool.record(),
        "database": {"path": str(args.database), "bytes": args.database.stat().st_size,
                     "sha256": digest(args.database), "indexed_sequences": sequences,
                     "indexed_letters": letters, "dbinfo": info.strip(),
                     "source_fasta_retained": source_fasta_retained,
                     "source_fasta": str(args.source_fasta) if source_fasta_retained else None,
                     "source_fasta_bytes": (args.source_fasta.stat().st_size
                                            if source_fasta_retained else None)},
        "query": {"path": str(query), "sha256": digest(query), "records": len(entries)},
        "fields": list(ALIGNMENT_FIELDS),
        "command": command,
        "log_tail": log,
        "hits": {"path": str(hits), "bytes": hits.stat().st_size, "sha256": digest(hits)},
    }, indent=1) + "\n")
    print("hits", hits.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
