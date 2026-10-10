#!/usr/bin/env python3
"""E19 coverage: band every backbone by its best UniRef90 hit, on the host that holds the index.

A CPU stage, and one that cannot run in the pod: the 88 GB UniRef90 release lives
on the Compute host, and ``TRANSFER_DIAMOND_DIR`` / ``TRANSFER_DIAMOND_DB`` inside
the pod are dead declarations pointing at paths that do not exist there.

What the band is and is not. It is how close this backbone's nearest relative in
a published reference release is, which is the covariate the endpoint has to be
stratified by before a remote-generalization reading is admissible. It is **not**
pretraining exposure: no arm's training set is searched here and none is claimed
to be. The remote band is never pooled with the close band.

The index is adopted only after its own ``dbinfo`` counts reproduce the counts its
staging manifest recorded, which is the admission route this project already
declares for a staged corpus. An index that cannot be identified that way is
refused, because a band against an unidentifiable corpus cannot be interpreted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.context.homology import (  # noqa: E402
    DiamondDatabase,
    assign_stratum,
    database_counts,
    parse_hits,
    prepare_diamond,
    run_diamond_blastp,
)
from src.capability.context.homology_context import EVALUE, MAX_TARGET_SEQS, SENSITIVITY  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.forcing.backbone_cohort import load_cohort  # noqa: E402

COMPLETION = "forcing_coverage.json"
CORPUS = ROOT / "data/pairwise_assets/uniref90_2026_03"
HITS = "uniref90_hits.tsv"
QUERIES = "backbones.faa"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_self_hit(hit) -> bool:
    """Whether this alignment is the query sequence itself, present in the corpus.

    A reviewed Swiss-Prot entry is a member of a UniRef90 cluster, so a search of
    the panel against UniRef90 finds every backbone verbatim. That self-hit is the
    right answer to "is this backbone covered by the reference release" and the
    wrong answer to "how remote is its nearest relative", which is the quantity a
    remoteness stratum needs. Both are therefore reported, and this predicate is
    what separates them: an alignment covering the whole query, the whole subject,
    with every residue identical.
    """

    return hit.nident == hit.qlen and hit.slen == hit.qlen


def band_degeneracy(bands: dict[str, str], *, name: str) -> dict:
    """Whether a stratum has more than one level, and what it means if it does not.

    A stratum with one level is not a control that passed; it is a control that
    could not be applied. Saying so here keeps a reader from reading a single
    populated band as evidence that remoteness was held fixed.
    """

    from collections import Counter

    counts = Counter(bands.values())
    return {
        "stratum": name,
        "levels": len(counts),
        "counts": dict(sorted(counts.items())),
        "usable_as_a_stratum": len(counts) >= 2,
        "consequence": (
            None if len(counts) >= 2 else
            "one level only: every backbone falls in the same band, so this stratum "
            "cannot separate remote from close and no remoteness contrast is available "
            "from this panel. It is a named non-identifiable contrast, not a control "
            "that was satisfied"
        ),
    }


def release_clusters(manifest: dict) -> int:
    """The cluster count the release note states, used as the corpus's record count."""

    for line in str(manifest["release_note"]).splitlines():
        if "Number of clusters" in line:
            return int(line.split(":")[1].strip().replace(",", ""))
    raise ValueError("the staging manifest's release note states no cluster count")


def adopt(tool, index: Path, manifest_path: Path) -> tuple[DiamondDatabase, dict]:
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("schema") != "uniref90_corpus_staging_v1":
        raise SystemExit(f"{manifest_path} is not a UniRef90 staging manifest")
    sequences, letters = database_counts(tool, Path(index))
    declared = manifest["index"]
    if (sequences, letters) != (declared["indexed_sequences"], declared["indexed_letters"]):
        raise SystemExit(
            f"{index} reports {sequences} sequences and {letters} letters against the "
            f"manifest's {declared['indexed_sequences']} and {declared['indexed_letters']}; "
            "this index is not the one the manifest identified and is refused"
        )
    clusters = release_clusters(manifest)
    database = DiamondDatabase(
        path=Path(index),
        source_fasta=ROOT / manifest["payload"]["path"],
        source_records=clusters,
        sequences=sequences,
        letters=letters,
        makedb_command=tuple(declared["makedb_command"]),
    )
    return database, {
        "release": manifest["release_note"].split("Release:")[1].split(",")[0].strip(),
        "release_clusters": clusters,
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "adoption": (
            "the index's own dbinfo counts reproduce the counts its staging manifest "
            "recorded, which is how a staged corpus is identified in this project"
        ),
    }


def run(args: argparse.Namespace) -> None:
    cohort = load_cohort(args.cohort)
    args.out.mkdir(parents=True, exist_ok=True)
    queries = args.out / QUERIES
    queries.write_text(
        "".join(
            f">{accession}\n{cohort['sequences'][accession]}\n"
            for accession in cohort["accessions"]
        ),
        encoding="utf-8",
    )
    tool = prepare_diamond(args.diamond_tarball, args.diamond_checksum, args.diamond_dir)
    database, provenance = adopt(tool, args.database, args.manifest)
    command, log_tail = run_diamond_blastp(
        tool, database, queries, args.out / HITS,
        threads=args.threads, sensitivity=SENSITIVITY, evalue=EVALUE,
        max_target_seqs=args.max_target_seqs,
    )
    hits = parse_hits(args.out / HITS)
    best: dict[str, float] = {}
    subject: dict[str, str] = {}
    best_nonself: dict[str, float] = {}
    subject_nonself: dict[str, str] = {}
    for hit in hits:
        identity = hit.identity_over_query
        if identity > best.get(hit.query, -1.0):
            best[hit.query] = identity
            subject[hit.query] = hit.subject
        if is_self_hit(hit):
            continue
        if identity > best_nonself.get(hit.query, -1.0):
            best_nonself[hit.query] = identity
            subject_nonself[hit.query] = hit.subject
    bands = {
        accession: assign_stratum(best.get(accession, 0.0))
        for accession in cohort["accessions"]
    }
    nonself_bands = {
        accession: assign_stratum(best_nonself.get(accession, 0.0))
        for accession in cohort["accessions"]
    }
    write_json(args.out / COMPLETION, {
        "schema": "forcing_coverage_v1",
        "status": "complete",
        "created_utc": _now(),
        "experiment": "E19",
        "identity_band": bands,
        "nonself_identity_band": nonself_bands,
        "best_identity_percent": {
            accession: float(best.get(accession, 0.0)) for accession in cohort["accessions"]
        },
        "best_nonself_identity_percent": {
            accession: float(best_nonself.get(accession, 0.0))
            for accession in cohort["accessions"]
        },
        "best_subject": {
            accession: subject.get(accession) for accession in cohort["accessions"]
        },
        "best_nonself_subject": {
            accession: subject_nonself.get(accession) for accession in cohort["accessions"]
        },
        "band_counts": dict(Counter(bands.values())),
        "nonself_band_counts": dict(Counter(nonself_bands.values())),
        "degeneracy": {
            "identity_band": band_degeneracy(bands, name="identity_band"),
            "nonself_identity_band": band_degeneracy(
                nonself_bands, name="nonself_identity_band"
            ),
        },
        "which_band_stratifies": (
            "the endpoint is stratified on nonself_identity_band, because the "
            "self-inclusive band answers whether a backbone is in the reference "
            "release rather than how remote its nearest relative is. Whichever band "
            "is used, a stratum with one level is reported as a contrast this panel "
            "cannot make"
        ),
        "backbones": len(cohort["accessions"]),
        "backbones_without_a_hit": sum(
            1 for accession in cohort["accessions"] if accession not in best
        ),
        "backbones_without_a_nonself_hit": sum(
            1 for accession in cohort["accessions"] if accession not in best_nonself
        ),
        "identity_definition": (
            "percent of the QUERY identically matched (nident / qlen), not percent "
            "identity within the aligned region: a short perfect fragment is not a close "
            "relative of the whole backbone"
        ),
        "coverage_is_not_exposure": (
            "this bands distance to a published reference release. No arm's training "
            "corpus is searched and none is claimed to be; a remote band is not evidence "
            "of pretraining-remote generalization by itself"
        ),
        "search": {
            "command": list(command),
            "sensitivity": SENSITIVITY,
            "evalue": EVALUE,
            "max_target_seqs": int(args.max_target_seqs),
            "threads": int(args.threads),
            "log_tail": log_tail,
        },
        "corpus": database.record() | provenance,
        "diamond": tool.record(),
        "cohort": {"path": str(args.cohort), "sha256": sha256_file(args.cohort)},
    })
    print(json.dumps({
        "band_counts": dict(Counter(bands.values())),
        "nonself_band_counts": dict(Counter(nonself_bands.values())),
        "artefact": str(args.out / COMPLETION),
    }, indent=1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=CORPUS / "uniref90_full.dmnd")
    parser.add_argument("--manifest", type=Path, default=CORPUS / "uniref90_staging_manifest.json")
    parser.add_argument(
        "--diamond-tarball", type=Path,
        default=ROOT / "external/tools/diamond-linux64-v2.1.24.tar.gz",
    )
    parser.add_argument(
        "--diamond-checksum", type=Path,
        default=ROOT / "external/tools/diamond-linux64-v2.1.24.tar.gz.sha256",
    )
    parser.add_argument("--diamond-dir", type=Path, default=ROOT / "external/tools/diamond")
    parser.add_argument("--max-target-seqs", type=int, default=MAX_TARGET_SEQS)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
