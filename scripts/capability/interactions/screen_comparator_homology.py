#!/usr/bin/env python3
"""Can a natural comparator be built with the homology support a generated product has?

A generated protein has essentially no retrievable relatives: 73 of 600 ProLLaMA
products reach any hit against the staged UniRef50 snapshot and the median
profile holds one sequence, itself, against 600 of 600 and about 48 effective
sequences for length-matched Swiss-Prot entries. Any generated-versus-natural
difference in likelihood response is therefore confounded with sequence
retrievability, and a Swiss-Prot comparator cannot break the confound because
every Swiss-Prot entry is in UniRef50 by construction.

That argument is true of *Swiss-Prot* and not of natural protein sequence in
general. This stage screens candidate natural sources against the same reference
the generated products were screened against, and reports, per source, how much
of it falls in each declared identity band -- in particular the band this
project already named ``lt30_no_detectable_homology``, which is the band a
generated product mostly sits in. A source that can populate that band at the
lengths the generated set occupies is a comparator that closes the retrievability
channel instead of adjusting for it.

Four source doors, each carrying its own provenance:

``generated``
    the frozen generated-sequence JSONL, so the arm under test is screened in the
    same pass and under the same settings as its candidate comparators;
``swissprot``
    the curated corpus the first comparator came from, through the project's own
    eligible-population door;
``mgnify``
    the staged MGnify cDNA-display library, restricted to rows the deposit flags
    ``mgnify`` and not ``dm_design`` -- metagenome-assembled natural domains,
    which is the population the prior remote gate drew its own low-identity
    backgrounds from;
``fasta``
    any further labelled FASTA, for a UniRef90 probe or the prior remote gate's
    own backgrounds.

The banding is not new. ``context.homology.assign_stratum`` already defines
``lt30_no_detectable_homology``, ``id30_to_70_remote_homology``,
``id70_to_95_close_homology`` and ``ge95_near_duplicate``, and the prior remote
gate's own caveat travels with every number here: **an identity band is a
property of a search against one UniRef50 snapshot, and no detected alignment is
not absence from any model's training data.** This stage measures retrievability
against a named reference. It does not measure memorisation.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.context import homology, profiles  # noqa: E402
from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.interactions import generated_mutation as gm  # noqa: E402

COMPLETION = "comparator_homology_screen.json"
TABLE = "comparator-screen.json"
SCHEMA = "comparator_homology_screen_v1"

SENSITIVITY = "very-sensitive"
EVALUE = 1e-3

#: Enough targets to place a sequence in a band and to count admissible-coverage
#: relatives. The profile itself is built later, on the selected cohort only, by
#: ``measure_generated_profile_control.py`` at the full target depth.
SCREEN_TARGET_SEQS = 25

BANDS = (
    "lt30_no_detectable_homology",
    "id30_to_70_remote_homology",
    "id70_to_95_close_homology",
    "ge95_near_duplicate",
)

#: The band a generated product mostly occupies, and therefore the band a
#: retrievability-matched natural comparator has to populate.
TARGET_BAND = "lt30_no_detectable_homology"

CAVEAT = (
    "an identity band is a property of a search against one UniRef50 snapshot, and no "
    "detected alignment is not absence from any model's training data; this screen "
    "measures retrievability against a named reference and not memorisation"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _admissible(sequence: str, hits) -> int:
    """Relatives covering at least the profile coverage floor of the query."""

    length = len(sequence)
    best: dict[str, float] = {}
    for hit in hits:
        covered = 100.0 * (hit.qend - hit.qstart + 1) / length
        if covered < profiles.PROFILE_COVERAGE_FLOOR:
            continue
        best[hit.subject] = max(best.get(hit.subject, 0.0), hit.bitscore)
    return len(best)


def read_generated_source(path: Path, band: tuple[int, int]) -> list[tuple[str, str]]:
    low, high = band
    return [
        (row["id"], row["sequence"])
        for row in gm.read_generated(path)
        if low <= len(row["sequence"]) <= high
    ]


def read_swissprot_source(band: tuple[int, int]) -> list[tuple[str, str]]:
    from src.capability.core.arms import eligible_protein_population

    low, high = band
    records, _labels, _corpus = eligible_protein_population(low, high)
    return [
        (gm.natural_identity(sequence), sequence)
        for sequence in dict.fromkeys(records)
    ]


def read_mgnify_source(path: Path, band: tuple[int, int]) -> list[tuple[str, str]]:
    """Natural metagenome-assembled **backgrounds**, not their mutational variants.

    Two declarations, both the project's own, and both load-bearing. ``mgnify``
    true with ``dm_design`` false is the deposit's statement that a row is a
    metagenome-assembled natural domain rather than a design; nothing here infers
    naturalness from a sequence. And
    :func:`context.remote_homology.background_map` is the project's rule for which
    rows are backgrounds -- a name whose longest proper underscore-prefix in the
    name set is itself -- which matters more than it looks: these libraries are
    mutagenesis libraries, so most of the 1.86 million distinct natural sequences
    are single-substitution variants of a few thousand domains. Drawing sequences
    rather than backgrounds would fill a comparator arm with near-copies of each
    other and collapse its independent units.
    """

    from src.capability.context import remote_homology as rh

    low, high = band
    allowed = set(AA20)
    rows = rh.read_source(path)
    index = {name: position for position, name in enumerate(rh.SOURCE_COLUMNS)}
    natural = {
        row[index["name"]]: row
        for row in rows
        if str(row[index["mgnify"]]).lower() in ("true", "1", "1.0")
        and str(row[index["dm_design"]]).lower() not in ("true", "1", "1.0")
    }
    roots = rh.background_map(set(natural))
    # Being its own root is not sufficient: in the raw table almost every row is
    # its own root, because the shared prefix is not itself a row. What makes a
    # name a *background* is that a mutational series roots to it, and the prior
    # gate's own threshold for that series is MIN_VARIANTS.
    children: dict[str, int] = {}
    for name, root in roots.items():
        if root != name:
            children[root] = children.get(root, 0) + 1
    out: dict[str, str] = {}
    for name, row in natural.items():
        if roots[name] != name or children.get(name, 0) < rh.MIN_VARIANTS:
            continue
        sequence = str(row[index["aa_seq"]])
        if low <= len(sequence) <= high and not set(sequence) - allowed:
            out.setdefault(sequence, name)
    return [(f"mg_{name}", sequence) for sequence, name in sorted(out.items())]


def read_fasta_source(path: Path, band: tuple[int, int]) -> list[tuple[str, str]]:
    low, high = band
    allowed = set(AA20)
    out, name, buffer = [], None, []

    def flush() -> None:
        if name is None:
            return
        sequence = "".join(buffer)
        if low <= len(sequence) <= high and not set(sequence) - allowed:
            out.append((name, sequence))

    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            flush()
            name, buffer = line[1:].split()[0], []
        elif name is not None:
            buffer.append(line.strip().upper())
    flush()
    return out


def subsample(records: list[tuple[str, str]], *, limit: int, label: str):
    if limit <= 0 or len(records) <= limit:
        return records, len(records)
    order = gm._rng("comparator-screen", label, len(records)).permutation(len(records))
    return [records[int(index)] for index in sorted(order[:limit])], len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generated", type=Path, help="the frozen generated-sequence JSONL")
    parser.add_argument("--swissprot", action="store_true",
                        help="screen the curated corpus through the eligible-population door")
    parser.add_argument("--mgnify", type=Path, help="the staged MGnify stability CSV")
    parser.add_argument("--fasta", action="append", default=[], metavar="NAME=PATH",
                        help="a further labelled FASTA source, repeatable")
    parser.add_argument("--min-residues", type=int, default=55)
    parser.add_argument("--max-residues", type=int, default=85)
    parser.add_argument("--sample", type=int, default=20000,
                        help="seeded cap per source; 0 screens everything")
    parser.add_argument("--database", type=Path,
                        default=ROOT / "data/homology_db/uniref50_full.dmnd")
    parser.add_argument("--corpus-fasta", type=Path, default=ROOT / "data/uniref50/uniref50.fasta")
    parser.add_argument("--diamond-tarball", type=Path,
                        default=ROOT / "external/tools/diamond-linux64-v2.1.24.tar.gz")
    parser.add_argument("--diamond-dir", type=Path, default=ROOT / "external/tools/diamond")
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--max-target-seqs", type=int, default=SCREEN_TARGET_SEQS)
    parser.add_argument("--hits", type=Path, help="a previous run's DIAMOND output, to reuse")
    parser.add_argument("--device", default="cpu",
                        help="accepted because the campaign queue injects it; unused")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.min_residues < 1 or args.max_residues < args.min_residues:
        parser.error("--min-residues and --max-residues must bracket a positive band")
    band = (args.min_residues, args.max_residues)

    out = gm.prepare_output_directory(args.out, COMPLETION)
    sources: dict[str, list[tuple[str, str]]] = {}
    provenance: dict[str, dict] = {}
    if args.generated is not None:
        sources["generated"] = read_generated_source(args.generated, band)
        provenance["generated"] = {"path": str(args.generated),
                                   "sha256": sha256_file(args.generated),
                                   "kind": "model output"}
    if args.swissprot:
        sources["swissprot"] = read_swissprot_source(band)
        provenance["swissprot"] = {"kind": "curated natural",
                                   "door": "core.arms.eligible_protein_population"}
    if args.mgnify is not None:
        sources["mgnify_natural"] = read_mgnify_source(args.mgnify, band)
        provenance["mgnify_natural"] = {
            "path": str(args.mgnify), "sha256": sha256_file(args.mgnify),
            "kind": "metagenome-assembled natural",
            "selection": "deposit flags mgnify true and dm_design false",
        }
    for item in args.fasta:
        if "=" not in item:
            parser.error(f"--fasta takes NAME=PATH; got {item!r}")
        label, path = item.split("=", 1)
        if label in sources:
            parser.error(f"duplicate source label {label!r}")
        sources[label] = read_fasta_source(Path(path), band)
        provenance[label] = {"path": path, "sha256": sha256_file(path), "kind": "labelled fasta"}
    if not sources:
        parser.error("at least one source is required")

    query = out / "queries.faa"
    screened: dict[str, list[tuple[str, str]]] = {}
    counts: dict[str, int] = {}
    seen: set[str] = set()
    with query.open("w") as handle:
        for label, records in sources.items():
            drawn, available = subsample(records, limit=args.sample, label=label)
            counts[label] = available
            kept = []
            for name, sequence in drawn:
                # One query id per source and sequence, so a sequence shared by two
                # sources is screened once per source rather than silently merged.
                key = f"{label}|{hashlib.sha256(sequence.encode()).hexdigest()[:20]}"
                if key in seen:
                    continue
                seen.add(key)
                kept.append((key, sequence))
                handle.write(f">{key}\n{sequence}\n")
            screened[label] = kept

    tool = homology.prepare_diamond(args.diamond_tarball,
                                    Path(str(args.diamond_tarball) + ".sha256"),
                                    args.diamond_dir)
    sequences, letters = homology.database_counts(tool, args.database)
    database = homology.DiamondDatabase(
        path=args.database, source_fasta=args.corpus_fasta, source_records=sequences,
        sequences=sequences, letters=letters, makedb_command=("pre-staged",),
    )
    search = out / "hits.tsv"
    if args.hits is not None:
        search = Path(args.hits)
        command, log = ["reused", str(search)], "search reused from a previous run"
    else:
        command, log = homology.run_diamond_blastp(
            tool, database, query, search, threads=args.threads, sensitivity=SENSITIVITY,
            evalue=EVALUE, max_target_seqs=args.max_target_seqs,
        )
    hits = homology.parse_hits(search)
    by_query: dict[str, list] = {}
    for hit in hits:
        by_query.setdefault(hit.query, []).append(hit)

    rows, census = [], {}
    for label, records in screened.items():
        bands = {name: 0 for name in BANDS}
        admissible, identities = [], []
        for key, sequence in records:
            found = by_query.get(key, [])
            best = max((homology.potential_identity_over_query(h) for h in found), default=0.0)
            stratum = homology.assign_stratum(best) if found else TARGET_BAND
            if stratum not in bands:
                bands[stratum] = 0
            bands[stratum] += 1
            count = _admissible(sequence, found)
            admissible.append(count)
            identities.append(best)
            rows.append({
                "source": label, "query": key, "length": len(sequence),
                "sequence": sequence, "n_hits": len(found),
                "max_identity_over_query": best, "band": stratum,
                "admissible_relatives": count,
            })
        counted = np.asarray(admissible)
        census[label] = {
            "available_in_band": counts[label],
            "screened": len(records),
            "bands": bands,
            "share_in_target_band": (
                bands.get(TARGET_BAND, 0) / len(records) if records else None
            ),
            "with_no_admissible_relative": int((counted == 0).sum()) if records else 0,
            "share_with_no_admissible_relative": (
                float((counted == 0).mean()) if records else None
            ),
            "median_max_identity_over_query": (
                float(np.median(identities)) if identities else None
            ),
            "median_admissible_relatives": int(np.median(counted)) if records else None,
            "lengths": (
                {"min": int(min(len(s) for _k, s in records)),
                 "max": int(max(len(s) for _k, s in records))} if records else None
            ),
        }

    write_json(out / TABLE, {"schema": SCHEMA, "band": list(band), "sequences": rows})
    write_json(out / COMPLETION, {
        "status": "complete",
        "schema": SCHEMA,
        "created_utc": _now(),
        "question": (
            "can a natural source populate the identity band a generated product occupies, "
            "at the lengths the generated set occupies?"
        ),
        "band_residues": list(band),
        "target_band": TARGET_BAND,
        "bands": list(BANDS),
        "table": TABLE,
        "table_sha256": sha256_file(out / TABLE),
        "sources": provenance,
        "sample_cap": args.sample,
        "search": {
            "sensitivity": SENSITIVITY, "evalue": EVALUE,
            "max_target_seqs": args.max_target_seqs, "threads": args.threads,
            "queries": sum(len(v) for v in screened.values()), "hits": len(hits),
            "command": list(command), "log_tail": log, "output_sha256": sha256_file(search),
        },
        "tool": tool.record(),
        "database": database.record(),
        "coverage_floor": profiles.PROFILE_COVERAGE_FLOOR,
        "census": census,
        "caveat": CAVEAT,
        "code_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "scripts/capability/interactions/screen_comparator_homology.py",
                "src/capability/context/homology.py",
            )
        },
    })


if __name__ == "__main__":
    main()
