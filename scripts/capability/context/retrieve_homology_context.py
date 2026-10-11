#!/usr/bin/env python3
"""Retrieve identity-binned homologous contexts for the anchor targets (CPU).

Four phases, all CPU and all on the host that holds the corpus indexes:

``search``
    One DIAMOND ``blastp`` of the anchor cohort's wild types against one named
    corpus, under the parameters frozen in
    :mod:`src.capability.context.homology_context`, writing the alignments
    themselves so that context items can be taken from the aligned segment.
``bin``
    Join the alignments back onto the targets, band every hit by identity over
    the target, retain per-bin candidates and a composition-matched donor pool,
    and record the distance of every target to the searched corpus. The product
    is the frozen input every GPU stage of E09 and E11 consumes.
``novelty``
    Search every generated product (E11) against the same corpus, in bounded,
    resumable, concurrently searchable chunks. Separate from ``annotate`` because
    the search is hours and the join is seconds, and because one invocation over
    the whole product set is not a bounded unit: it ran 55 minutes and 20
    CPU-hours without reaching its output phase.
``annotate``
    Join the products onto the prompt's own family -- which supplies the
    alignment-identity copying rule -- and onto the corpus hits, either from a
    search it runs itself or from the ones ``novelty`` already produced.

The corpus is never rebuilt here. An index staged against a published release is
adopted only after its own ``dbinfo`` counts match that release's record; an index
with no release record is adopted only after its counts match the FASTA beside it.
A corpus that cannot be identified either way is refused, because a band against
an unidentifiable corpus is the defect this experiment exists to replace.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.context import homology_context as H  # noqa: E402
from src.capability.context.homology import (  # noqa: E402
    ALIGNMENT_FIELDS,
    DiamondDatabase,
    build_database,
    database_counts,
    parse_hits,
    prepare_diamond,
    run_diamond_blastp,
    assign_stratum,
    truncated_alignment,
    truncation_raises_stratum,
)

#: Why the masking setting is part of this measurement rather than a flag, said
#: once and written into every artefact the corpus search produces.
CORPUS_MASKING_REASON = (
    "DIAMOND masks low-complexity and repetitive query regions by default, which "
    "truncates the alignment of exactly the records a novelty endpoint is about and "
    "biases identity downward, so a verbatim corpus member is binned as a distant "
    "relative. The measured case in src.capability.context.homology: a record "
    "byte-identical to UniRef50_Q3E8Z8 over all 732 residues was reported at pident "
    "100 over 607 aligned residues, i.e. identity over query 82.9, and 5 of 48 "
    "records were mis-binned that way, all in the same direction. The error runs in "
    "the flattering direction for a novelty claim: it makes the most retrievable "
    "sequences look the most novel. --masking 0 is therefore a correctness "
    "requirement, not a tuning choice"
)
from src.capability.core.io import sha256_file, write_json  # noqa: E402

#: Default anchor inputs. The cohort carries the wild-type sequences, the frozen
#: family cluster of every assay and the 128-variant draw; the admission record
#: names the 201 assays every arm actually scored.
COHORT = "results/R5/context_mutation_rescue_20260923/cohort.json"
ADMISSION = (
    "results/R5/local_context_20260923/20260923233257_e429ce7f31e4/"
    "lcgp_admission/local_context_measurement.json"
)

#: The prior remote-homology gate's own cohort. Its 305 metagenome-derived
#: backgrounds are searched in the same pass as the anchor targets, because the
#: one claim this experiment can settle at no extra cost is whether the groups
#: that gate called *remote* -- banded through a staged UniRef50 snapshot -- are
#: remote with respect to the corpus the arms that read positive on them were
#: actually trained on. One search, two panels, no second scan of a 62 GB index.
REMOTE_GATE = "results/R5/remote_homology_20260924/cohort.json"

#: The corpora this host can search, with what identifies each one. ``release`` is
#: the published release string; ``None`` means the snapshot has no recorded
#: release and may only be reported as an object, never as a named corpus.
CORPORA: dict[str, dict[str, object]] = {
    "uniref90_2026_03": {
        "database": "data/pairwise_assets/uniref90_2026_03/uniref90_full.dmnd",
        "source_fasta": "data/pairwise_assets/uniref90_2026_03/uniref90.fasta.gz",
        "manifest": "data/pairwise_assets/uniref90_2026_03/uniref90_staging_manifest.json",
    },
    "uniref50_local_snapshot": {
        "database": "data/homology_db/uniref50_full.dmnd",
        "source_fasta": "data/uniref50/uniref50.fasta",
        "manifest": None,
    },
}

EXPECT_SEARCH = "search.json"
EXPECT_BIN = "completion.json"
EXPECT_ANNOTATE = "generation_annotation.json"
ARTIFACT = "homology_context.json"
HITS = "corpus_hits.tsv"
QUERIES = "targets.faa"


def runtime() -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "device": "cpu",
        "cpu_count": os.cpu_count(),
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "peak_rss_bytes": max(usage.ru_maxrss, children.ru_maxrss) * 1024,
        "cpu_seconds": usage.ru_utime + usage.ru_stime + children.ru_utime + children.ru_stime,
        "threads_requested": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def tool_for(args: argparse.Namespace):
    return prepare_diamond(args.diamond_tarball, args.diamond_checksum, args.diamond_dir)


def resolve_corpus(args: argparse.Namespace, tool) -> tuple[DiamondDatabase, dict]:
    """Adopt one identified corpus index, or refuse it.

    Two admission routes, one per kind of evidence:

    * a staging manifest, which bound the downloaded bytes to a publisher digest
      and recorded the release note and the index's own counts. The index is
      adopted when ``dbinfo`` reproduces those counts, and the release string is
      carried into every artefact.
    * no manifest, in which case the FASTA beside the index is counted and both
      sequences and residues must agree (:func:`build_database` with
      ``rebuild=False``). The corpus is then pinned as an object -- bytes,
      records, residues -- and reported with ``release: null``.
    """

    if args.database is not None:
        if args.corpus_id is None or args.source_fasta is None:
            raise SystemExit("--database needs --corpus-id and --source-fasta")
        spec = {"database": args.database, "source_fasta": args.source_fasta, "manifest": None}
        name = args.corpus_id
        registered = False
    else:
        registered = True
        if args.corpus not in CORPORA:
            raise SystemExit(f"unknown corpus {args.corpus!r}; known: {sorted(CORPORA)}")
        spec = dict(CORPORA[args.corpus])
        name = args.corpus
    database = Path(spec["database"]) if Path(str(spec["database"])).is_absolute() else ROOT / str(spec["database"])
    source = Path(spec["source_fasta"]) if Path(str(spec["source_fasta"])).is_absolute() else ROOT / str(spec["source_fasta"])
    if not source.is_file():
        raise SystemExit(f"{source} does not exist; this host cannot search {name}")
    # A registered corpus is never built here: an absent 24-62 GB index is an
    # operational fact to report, not something to rebuild inside a measurement.
    # An explicitly named corpus may be indexed on the spot, which is how a small
    # ad-hoc corpus is searched.
    if registered and not database.is_file():
        raise SystemExit(
            f"{database} does not exist; this host cannot search the registered corpus "
            f"{name}, and a registered index is never rebuilt inside a measurement"
        )

    manifest_path = None if spec["manifest"] is None else ROOT / str(spec["manifest"])
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index = manifest["index"]
        sequences, letters = database_counts(tool, database)
        if (sequences, letters) != (index["indexed_sequences"], index["indexed_letters"]):
            raise SystemExit(
                f"{database} reports {sequences} sequences and {letters} letters; the "
                f"staging manifest for {name} records {index['indexed_sequences']} and "
                f"{index['indexed_letters']}. The index beside the manifest is not the "
                "index the manifest describes."
            )
        if args.verify_index_digest:
            observed = sha256_file(database)
            if observed != index["sha256"]:
                raise SystemExit(f"{database} sha256 {observed} != manifest {index['sha256']}")
        record = DiamondDatabase(
            path=database,
            source_fasta=source,
            source_records=int(index["indexed_sequences"]),
            sequences=int(sequences),
            letters=int(letters),
            makedb_command=tuple(str(part) for part in index["makedb_command"]),
        )
        identity = {
            "corpus_id": name,
            "release": manifest["release_note"].strip(),
            "release_note_url": manifest.get("release_note_url"),
            "payload_sha256": manifest["payload"]["digests"]["sha256"],
            "payload_declared_digest": manifest["payload"]["verified_against"],
            "index_sha256": index["sha256"],
            "index_sha256_reverified": bool(args.verify_index_digest),
            "staging_manifest": str(manifest_path.relative_to(ROOT)),
            "staging_manifest_sha256": sha256_file(manifest_path),
            "named_release": True,
        }
    else:
        record = build_database(
            tool,
            source,
            database,
            threads=args.threads,
            tmpdir=args.tmpdir,
            rebuild=False,
        )
        identity = {
            "corpus_id": name,
            "release": None,
            "release_note_url": None,
            "index_sha256_reverified": False,
            "named_release": False,
            "object_pin": {
                "source_fasta_bytes": source.stat().st_size,
                "source_fasta_records": record.source_records,
                "source_fasta_residues": record.letters,
            },
            "caveat": (
                "this snapshot carries no release identifier; it is pinned as an object "
                "(bytes, records, residues) and may not be reported as a named corpus"
            ),
        }
    return record, identity


def assay_payload(cohort: dict, targets: list[dict]) -> list[dict]:
    """The scored variant draw of every admitted assay, carried with the contexts.

    The GPU stage that consumes this artefact then needs nothing but the artefact
    and the weights: no ProteinGym tree, no second cohort file, no path that could
    resolve differently in a pod than on this host. Only the mutation strings and
    the measured labels travel; the mutated sequences are rebuilt from the wild
    type, and that rebuild is verified here against the frozen cohort's own
    sequences so the stage cannot score a sequence the cohort never held.
    """

    from src.capability.models.fitness import parse_mutant

    wanted = {target["target_id"] for target in targets}
    payload = []
    for row in cohort["assays"]:
        if row["wildtype_id"] not in wanted:
            continue
        wildtype = row["wildtype"]
        for mutant, sequence in zip(row["mutants"], row["sequences"]):
            rebuilt = list(wildtype)
            for wild, position, mutated in parse_mutant(mutant):
                if rebuilt[position - 1] != wild:
                    raise SystemExit(
                        f"{row['assay']} {mutant}: the wild type carries "
                        f"{rebuilt[position - 1]!r} at position {position}"
                    )
                rebuilt[position - 1] = mutated
            if "".join(rebuilt) != sequence:
                raise SystemExit(f"{row['assay']} {mutant}: rebuilt sequence differs from the cohort")
        payload.append(
            {
                "assay": row["assay"],
                "target_id": row["wildtype_id"],
                "cluster": int(row["cluster"]),
                "mutants": list(row["mutants"]),
                "measured": list(row["measured"]),
                "mutant_digest": row["mutant_digest"],
                "profile_scores": list(row["profile_scores"]),
            }
        )
    return sorted(payload, key=lambda row: row["assay"])


def load_targets(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """The anchor targets: one per distinct wild type of the admitted assays."""

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    admission = json.loads(args.admission.read_text(encoding="utf-8"))
    if admission.get("status") != "admitted":
        raise SystemExit(f"{args.admission} is not an admission receipt")
    admitted = sorted(admission["support"]["assay_ids"])
    rows = {row["assay"]: row for row in cohort["assays"]}
    missing = [assay for assay in admitted if assay not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} admitted assays are absent from the cohort: {missing[:5]}")
    grouped: dict[str, dict] = {}
    for assay in admitted:
        row = rows[assay]
        target = grouped.setdefault(
            row["wildtype_id"],
            {
                "target_id": row["wildtype_id"],
                "wildtype": row["wildtype"],
                "cluster": int(row["cluster"]),
                "assays": [],
            },
        )
        if target["wildtype"] != row["wildtype"] or target["cluster"] != int(row["cluster"]):
            raise SystemExit(f"{row['wildtype_id']} carries two wild types or two clusters")
        target["assays"].append(assay)
    targets = [grouped[key] for key in sorted(grouped)]
    for target in targets:
        target["assays"] = sorted(target["assays"])
        target["length"] = len(target["wildtype"])
    if args.limit:
        targets = targets[: args.limit]
    support = {
        "targets": len(targets),
        "assays": sum(len(target["assays"]) for target in targets),
        "clusters": len({target["cluster"] for target in targets}),
        "admitted_assays": len(admitted),
        "cohort": str(args.cohort.relative_to(ROOT)),
        "cohort_sha256": sha256_file(args.cohort),
        "admission": str(args.admission.relative_to(ROOT)),
        "admission_sha256": sha256_file(args.admission),
        "limited_to": args.limit or None,
    }
    return targets, support


def load_remote_gate(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """The prior gate's backgrounds, reconstructed from its own frozen cohort.

    The cohort stores every admitted variant's full sequence but not the
    background it came from. Each variant differs from its background at exactly
    one declared position, so the background is the per-position consensus of its
    variants -- and that reconstruction is *verified* rather than trusted: every
    variant must differ from the consensus at exactly its own declared position
    and carry exactly its own declared substituted residue. Anything else stops
    the run, because a mis-reconstructed background would be searched as a protein
    that was never measured.
    """

    if args.no_remote_gate:
        return [], {"included": False, "reason": "disabled with --no-remote-gate"}
    path = args.remote_gate_cohort
    if not path.is_file():
        raise SystemExit(f"{path} does not exist; pass --no-remote-gate to search only the anchor")
    cohort = json.loads(path.read_text(encoding="utf-8"))
    panel: list[dict] = []
    for background in cohort["backgrounds"]:
        sequences = [variant["sequence"] for variant in background["variants"]]
        length = int(background["length"])
        if {len(sequence) for sequence in sequences} != {length}:
            raise SystemExit(f"{background['name']}: variant lengths disagree with the background")
        wildtype = "".join(
            max(
                {sequence[index] for sequence in sequences},
                key=lambda residue, index=index: sum(
                    1 for sequence in sequences if sequence[index] == residue
                ),
            )
            for index in range(length)
        )
        for variant in background["variants"]:
            position = int(variant["position"])
            differing = [
                index for index in range(length) if variant["sequence"][index] != wildtype[index]
            ]
            if differing != [position - 1] or variant["sequence"][position - 1] != variant["mutant"]:
                raise SystemExit(
                    f"{background['name']}: variant at position {position} does not reconstruct "
                    f"against the consensus background (differs at {differing})"
                )
        panel.append(
            {
                "target_id": background["name"],
                "wildtype": wildtype,
                "length": length,
                "group": background["group"],
                "prior_band": background["band"],
                "prior_stratum": background["stratum"],
                "library": background["library"],
                "admitted_variants": int(background["admitted_variants"]),
            }
        )
    panel.sort(key=lambda row: row["target_id"])
    if args.limit:
        panel = panel[: args.limit]
    support = {
        "included": True,
        "cohort": str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path),
        "cohort_sha256": sha256_file(path),
        "backgrounds": len(panel),
        "groups": len({row["group"] for row in panel}),
        "prior_band_counts": {
            band: sum(1 for row in panel if row["prior_band"] == band)
            for band in sorted({row["prior_band"] for row in panel})
        },
        "prior_stratum_counts": {
            stratum: sum(1 for row in panel if row["prior_stratum"] == stratum)
            for stratum in sorted({row["prior_stratum"] for row in panel})
        },
        "reconstruction": "per-position consensus of the admitted variants, verified variant by variant",
        "limited_to": args.limit or None,
    }
    return panel, support


def write_fasta(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    identifiers = [row["target_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise SystemExit("two queries share an identifier; the hit join would be ambiguous")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f">{row['target_id']}\n{row['wildtype']}\n")


def search(args: argparse.Namespace) -> None:
    targets, support = load_targets(args)
    gate_panel, gate_support = load_remote_gate(args)
    tool = tool_for(args)
    database, identity = resolve_corpus(args, tool)
    args.out.mkdir(parents=True, exist_ok=True)
    queries = args.out / QUERIES
    write_fasta(targets + gate_panel, queries)
    started = time.monotonic()
    command, log_tail = run_diamond_blastp(
        tool,
        database,
        queries,
        args.out / HITS,
        threads=args.threads,
        sensitivity=H.SENSITIVITY,
        evalue=H.EVALUE,
        max_target_seqs=args.max_target_seqs,
        fields=ALIGNMENT_FIELDS,
    )
    hits_path = args.out / HITS
    n_hsps = sum(1 for line in hits_path.open(encoding="utf-8") if line.strip())
    if n_hsps < 1:
        raise RuntimeError(f"{command} produced no alignment at all; refusing to record a search")
    write_json(
        args.out / EXPECT_SEARCH,
        {
            "schema_version": H.SCHEMA_VERSION,
            "stage": "search",
            **H.declaration_digests(),
            "declaration": H.declaration(),
            "diamond": tool.record(),
            "database": database.record(),
            "corpus_identity": identity,
            "support": support,
            "remote_gate_support": gate_support,
            "queries": str(queries.relative_to(args.out)),
            "queries_sha256": sha256_file(queries),
            "hits": str(hits_path.relative_to(args.out)),
            "hits_sha256": sha256_file(hits_path),
            "hits_fields": list(ALIGNMENT_FIELDS),
            "n_hsps": n_hsps,
            "max_target_seqs": args.max_target_seqs,
            "command": list(command),
            "log_tail": log_tail,
            "elapsed_seconds": time.monotonic() - started,
            "runtime": runtime(),
        },
    )
    print(
        f"search: {len(targets)} anchor targets + {len(gate_panel)} prior-gate backgrounds, "
        f"{n_hsps} HSPs in {time.monotonic()-started:.1f}s",
        flush=True,
    )


def reband_remote_gate(panel: list[dict], hits: dict[str, list]) -> dict:
    """Re-band the prior gate's groups against this corpus, by its own rule.

    One variable changes: the corpus. The band edges, the identity definition and
    the group rule -- close only if every member bands at or above 70%, remote
    only if every member bands below it -- are the prior gate's own, imported from
    :mod:`~src.capability.context.remote_homology` rather than restated, so a
    migration between strata cannot be an artefact of a second banding
    convention.

    The prior gate applied no coverage requirement, so the like-for-like band is
    computed without one and reported as primary; this experiment's stricter
    covered band (:data:`~src.capability.context.homology_context.COVERAGE_FLOOR`)
    is reported beside it, and the two bracket the reading.
    """

    from src.capability.context import remote_homology as RH

    labels = {row["target_id"]: row["group"] for row in panel}
    prior_band = {row["target_id"]: row["prior_band"] for row in panel}
    prior_group = {}
    for row in panel:
        prior_group[row["group"]] = row["prior_stratum"]

    new_band: dict[str, str] = {}
    covered_band: dict[str, str] = {}
    rows = []
    for row in panel:
        found = hits[row["target_id"]]
        for hit in found:
            if hit.qlen != row["length"]:
                raise SystemExit(
                    f"{row['target_id']}: DIAMOND reports qlen {hit.qlen} for a "
                    f"{row['length']}-residue background"
                )
        plain = max((hit.identity_over_query for hit in found), default=0.0)
        covered = H.max_identity_over_query(found)
        new_band[row["target_id"]] = assign_stratum(plain)
        covered_band[row["target_id"]] = assign_stratum(covered)
        rows.append(
            {
                "background": row["target_id"],
                "group": row["group"],
                "length": row["length"],
                "n_hits": len(found),
                "prior_band": row["prior_band"],
                "prior_stratum": row["prior_stratum"],
                "max_identity_over_query": plain,
                "max_identity_over_query_covered": covered,
                "new_band": new_band[row["target_id"]],
                "new_band_covered": covered_band[row["target_id"]],
            }
        )

    assignment, report = RH.group_strata(labels, new_band)
    covered_assignment, covered_report = RH.group_strata(labels, covered_band)
    migration = {}
    for group, was in prior_group.items():
        now = assignment[group]
        migration[f"{was}->{now}"] = migration.get(f"{was}->{now}", 0) + 1
    moved = sorted(
        group for group, was in prior_group.items() if assignment[group] != was
    )
    remote_now_close = sorted(
        group
        for group, was in prior_group.items()
        if was == "remote" and assignment[group] in {"close", "mixed"}
    )
    return {
        "question": (
            "are the family groups the prior gate called remote -- banded through a "
            "staged UniRef50 snapshot -- still remote with respect to the corpus "
            "searched here?"
        ),
        "band_rule": "imported from remote_homology.group_strata; only the corpus changed",
        "prior_group_strata": {
            stratum: sum(1 for value in prior_group.values() if value == stratum)
            for stratum in sorted(set(prior_group.values()))
        },
        "new_group_strata": {
            "close": report["close_groups"],
            "remote": report["remote_groups"],
            "mixed": report["mixed_groups"],
        },
        "new_group_strata_covered": {
            "close": covered_report["close_groups"],
            "remote": covered_report["remote_groups"],
            "mixed": covered_report["mixed_groups"],
        },
        "group_migration_counts": migration,
        "groups_that_moved": moved,
        "prior_remote_groups_now_close_or_mixed": remote_now_close,
        "prior_remote_groups_now_close_or_mixed_fraction": (
            len(remote_now_close)
            / max(1, sum(1 for value in prior_group.values() if value == "remote"))
        ),
        "new_band_counts": {
            band: sum(1 for value in new_band.values() if value == band)
            for band in sorted(set(new_band.values()))
        },
        "prior_band_counts": {
            band: sum(1 for value in prior_band.values() if value == band)
            for band in sorted(set(prior_band.values()))
        },
        "group_report": report,
        "group_report_covered": covered_report,
        "backgrounds": rows,
        "does_not_license": (
            "a re-band changes which groups a stratum holds; it does not re-read any "
            "model increment. The prior gate's per-row predictions are not retained, "
            "so the stratified increments under this banding require refitting that "
            "gate from its retained extractions and are not computed here"
        ),
    }


def bin_hits(args: argparse.Namespace) -> None:
    """Band the alignments and build every target's candidate and donor sets."""

    record = json.loads((args.search / EXPECT_SEARCH).read_text(encoding="utf-8"))
    H.require_declaration(record, scope="search")
    hits_path = args.search / record["hits"]
    if sha256_file(hits_path) != record["hits_sha256"]:
        raise SystemExit(f"{hits_path} is not the file {EXPECT_SEARCH} describes")
    targets, support = load_targets(args)
    gate_panel, gate_support = load_remote_gate(args)
    if support["targets"] != record["support"]["targets"]:
        raise SystemExit(
            f"the search covered {record['support']['targets']} targets and this run "
            f"assembled {support['targets']}"
        )
    if gate_support.get("backgrounds", 0) != record["remote_gate_support"].get("backgrounds", 0):
        raise SystemExit("the prior-gate panel differs from the one the search covered")
    hits = parse_hits(hits_path, fields=ALIGNMENT_FIELDS)
    by_query: dict[str, list] = {target["target_id"]: [] for target in targets}
    gate_hits: dict[str, list] = {row["target_id"]: [] for row in gate_panel}
    for hit in hits:
        if hit.query in by_query:
            by_query[hit.query].append(hit)
        elif hit.query in gate_hits:
            gate_hits[hit.query].append(hit)
        else:
            raise SystemExit(f"alignment for unknown query {hit.query!r}")

    args.out.mkdir(parents=True, exist_ok=True)
    prepared: list[dict] = []
    for target in targets:
        found = by_query[target["target_id"]]
        for hit in found:
            if hit.qlen != target["length"]:
                raise SystemExit(
                    f"{target['target_id']}: DIAMOND reports qlen {hit.qlen} for a "
                    f"{target['length']}-residue target"
                )
        bins, refusals = H.bin_candidates(found, wildtype=target["wildtype"])
        best = H.max_identity_over_query(found)
        prepared.append(
            {
                **target,
                "n_hits": len(found),
                "hit_list_saturated": len(found) >= args.max_target_seqs,
                "max_identity_over_query": best,
                "corpus_stratum": assign_stratum(best),
                "verbatim_in_corpus": H.self_copy_candidate(found, wildtype=target["wildtype"])
                is not None,
                "self_copy": H.self_copy_candidate(found, wildtype=target["wildtype"]),
                "bins": bins,
                "hit_refusals": refusals,
            }
        )

    # Donors come from every other target's retrieved segments: they are corpus
    # records of established provenance, they are not relatives of this target
    # (different family cluster, absent from its own hit list), and the pool is
    # large enough for the composition match to bite.
    pool_by_cluster: dict[int, list[dict]] = {}
    for target in prepared:
        rows = [row for bucket in target["bins"].values() for row in bucket]
        pool_by_cluster.setdefault(target["cluster"], []).extend(rows)
    for target in prepared:
        candidates = [
            row
            for cluster, rows in pool_by_cluster.items()
            if cluster != target["cluster"]
            for row in rows
        ]
        own_subjects = {hit.subject for hit in by_query[target["target_id"]]}
        target["unrelated_pool"] = H.donor_pool(
            target["wildtype"], candidates=candidates, excluded_subjects=own_subjects
        )

    gate = reband_remote_gate(gate_panel, gate_hits) if gate_panel else None

    bin_support = {
        name: sum(1 for target in prepared if target["bins"][name]) for name in H.BIN_NAMES
    }
    bin_clusters = {
        name: len({target["cluster"] for target in prepared if target["bins"][name]})
        for name in H.BIN_NAMES
    }
    payload = {
        "schema_version": H.SCHEMA_VERSION,
        "stage": "bin",
        **H.declaration_digests(),
        "declaration": H.declaration(),
        "search": {
            "record": str((args.search / EXPECT_SEARCH).relative_to(ROOT))
            if str(args.search).startswith(str(ROOT))
            else str(args.search / EXPECT_SEARCH),
            "record_sha256": sha256_file(args.search / EXPECT_SEARCH),
            "hits_sha256": record["hits_sha256"],
            "n_hsps": record["n_hsps"],
            "max_target_seqs": record["max_target_seqs"],
        },
        "diamond": record["diamond"],
        "database": record["database"],
        "corpus_identity": record["corpus_identity"],
        "support": {
            **support,
            "candidate_support_by_bin": bin_support,
            "cluster_support_by_bin": bin_clusters,
            "targets_with_verbatim_record": sum(
                1 for target in prepared if target["verbatim_in_corpus"]
            ),
            "targets_with_empty_donor_pool": sum(
                1 for target in prepared if not target["unrelated_pool"]
            ),
            "corpus_stratum_counts": {
                name: sum(1 for target in prepared if target["corpus_stratum"] == name)
                for name in sorted({target["corpus_stratum"] for target in prepared})
            },
            "group_floor": H.GROUP_FLOOR,
            "bins_below_group_floor": [
                name for name, value in bin_clusters.items() if value < H.GROUP_FLOOR
            ],
        },
        "remote_gate": gate,
        "remote_gate_support": gate_support,
        "targets": prepared,
        "assays": assay_payload(json.loads(args.cohort.read_text(encoding="utf-8")), targets),
    }
    write_json(args.out / ARTIFACT, payload)
    write_json(
        args.out / EXPECT_BIN,
        {
            "schema_version": H.SCHEMA_VERSION,
            "stage": "bin",
            "status": "complete",
            **H.declaration_digests(),
            "artifact": ARTIFACT,
            "artifact_sha256": sha256_file(args.out / ARTIFACT),
            "artifact_bytes": (args.out / ARTIFACT).stat().st_size,
            "support": payload["support"],
            "runtime": runtime(),
        },
    )
    print(
        f"bin: {len(prepared)} targets; per-bin target support {bin_support}; "
        f"per-bin cluster support {bin_clusters}",
        flush=True,
    )


def product_annotation(
    row: Mapping[str, Any],
    *,
    index: int,
    family: Sequence[Mapping[str, Any]],
    corpus: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """One product's alignment annotation: family membership, copying, and novelty.

    Separated from the search so the join can be tested without an aligner,
    because the join is where it went wrong. Two rules, and they are not the
    same rule:

    * **family membership** needs an alignment to a sequence of the product's own
      target that is *of that target's family*, covering at least
      :data:`~src.capability.context.homology_context.FAMILY_COVERAGE_FLOOR` of
      the product. The matched-unrelated donors sit in the same search and are
      marked out of family, so they can never satisfy it.
    * **identity to the conditioning context** needs an alignment to a sequence
      the product's own prompt actually contained, matched on the corpus
      accession the product recorded. No coverage floor: a short, high-identity
      run reproduced from the prompt is exactly what the copy rule is looking
      for, and the coverage is reported beside the identity rather than used to
      discard it.

    ``None`` for the context identity means the aligner found no alignment at
    all, which is different from an alignment at zero identity and is why the
    field is not defaulted.

    Corpus novelty is a **third** question and shares nothing with the first two
    but the aligner. "This product is not a copy of its four conditioning
    homologues" and "this product is not a near-duplicate of anything in the
    reference" are different claims, and a product can satisfy the first and fail
    the second. The best corpus hit is therefore reported twice: once over every
    hit, and once over hits that are *not* one of the product's own context items,
    so retrieval of the prompt and retrieval of the corpus at large are never read
    as one number.
    """

    target = row["target_id"]
    in_family = [
        entry for entry in family if entry["target_id"] == target and entry["in_family"]
    ]
    own = [entry for entry in in_family if entry["coverage"] >= H.FAMILY_COVERAGE_FLOOR]
    cited = set(row.get("context_subjects", ()))
    context_entries = [
        entry
        for entry in family
        if entry["target_id"] == target and entry["subject"] in cited
    ]
    return {
        "product_index": index,
        "attempt_id": row.get("attempt_id"),
        "arm": row.get("arm"),
        "target_id": target,
        "condition": row.get("condition"),
        "residues": len(row["sequence"]),
        "context_items": len(cited),
        "prompt_family_recognised": bool(own),
        "prompt_family_identity": max(
            (entry["identity_over_query"] for entry in own), default=0.0
        ),
        "prompt_family_best_coverage": max(
            (entry["coverage"] for entry in in_family), default=0.0
        ),
        "context_alignment_identity": (
            max(entry["identity_over_query"] for entry in context_entries)
            if context_entries
            else None
        ),
        "context_alignment_coverage": (
            max(entry["coverage"] for entry in context_entries) if context_entries else None
        ),
        "context_items_aligned": len({entry["subject"] for entry in context_entries}),
        **corpus_annotation(corpus, cited=cited),
    }


def corpus_annotation(
    corpus: Sequence[Mapping[str, Any]], *, cited: set[str]
) -> dict[str, Any]:
    """The corpus-proximity fields of one product, with the prompt split out."""

    hits = list(corpus)
    outside = [hit for hit in hits if hit["subject"] not in cited]
    best = max(hits, key=lambda hit: hit["identity_over_query"], default=None)
    best_outside = max(outside, key=lambda hit: hit["identity_over_query"], default=None)
    return {
        "corpus_hits": len(hits),
        "corpus_max_identity": None if best is None else best["identity_over_query"],
        "corpus_best_subject": None if best is None else best["subject"],
        "corpus_best_coverage": None if best is None else best["coverage"],
        "corpus_best_is_own_context": None if best is None else best["subject"] in cited,
        "corpus_max_identity_outside_context": (
            None if best_outside is None else best_outside["identity_over_query"]
        ),
        "corpus_best_subject_outside_context": (
            None if best_outside is None else best_outside["subject"]
        ),
        "corpus_truncated_alignments": sum(1 for hit in hits if hit["truncated_alignment"]),
        "corpus_exact_full_length_hits": sum(1 for hit in hits if hit["exact_full_length"]),
    }


EXPECT_NOVELTY = "product_novelty_search.json"


def novelty(args: argparse.Namespace) -> None:
    """Search every generated product against the reference corpus, in chunks.

    **Why chunks, measured rather than assumed.** A single invocation over all
    6144 E11 products ran 55 minutes and 20 CPU-hours without reaching its output
    phase, against 10.7 minutes for a 64-query probe over the same index. The
    database scan *is* amortised across a query block, but it is not what
    dominates: DIAMOND iterates 114 reference blocks times 14 seed shapes, and the
    per-shape query-side work -- query seed array, hash join, alignment search --
    scales with query letters and is paid inside every one of those iterations. At
    96 times the query residues the query side swamps the reference side, so cost
    grows with query count rather than being flat in it.

    So the query set is split into chunks, each its own invocation with its own
    progress log and its own hits file. Three properties follow, and they are the
    point:

    * **bounded** -- a chunk is a unit whose cost is known from the previous one;
    * **resumable** -- a chunk whose hits file and digest sidecar are already
      present is skipped, so an interrupted search resumes instead of restarting;
    * **observable** -- each chunk streams DIAMOND's block-by-block progress to a
      file, which the single invocation could not do.

    Chunks may run concurrently. Physical reads were 7.7 GB over 55 minutes on a
    62 GB index, so the index is page-cached and the constraint is CPU, not I/O:
    several chunks at fewer threads each beat one chunk at many.

    The corpus is **not** reduced to a smaller reference to save time. Changing
    the reference changes the estimand, and this number exists to be set beside a
    measurement made on this one.
    """

    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    products = [
        json.loads(line)
        for line in args.products.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not products:
        raise SystemExit(f"{args.products} holds no generated product")
    if args.chunk_size < 1 or args.workers < 1:
        raise SystemExit("the chunk size and the worker count must be positive")
    tool = tool_for(args)
    database, corpus_identity = resolve_corpus(args, tool)
    args.out.mkdir(parents=True, exist_ok=True)
    chunk_dir = args.out / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic chunking on the attempt identifier, so a resumed run and a
    # fresh one cut the query set the same way.
    measurable = sorted(
        (
            row
            for row in products
            if len(str(row["sequence"])) >= H.MIN_PRODUCT_RESIDUES
        ),
        key=lambda row: str(row["attempt_id"]),
    )
    if not measurable:
        raise SystemExit(
            f"no product of {args.products} reaches {H.MIN_PRODUCT_RESIDUES} residues"
        )
    chunks: list[dict] = []
    for index in range(0, len(measurable), args.chunk_size):
        members = measurable[index : index + args.chunk_size]
        number = len(chunks)
        fasta = chunk_dir / f"chunk_{number:03d}.faa"
        fasta.write_text(
            "".join(
                f">{row['attempt_id']}\n{row['sequence']}\n" for row in members
            ),
            encoding="utf-8",
        )
        chunks.append(
            {
                "chunk": number,
                "queries": len(members),
                "residues": sum(len(str(row["sequence"])) for row in members),
                "query_fasta": str(fasta),
                "hits": str(chunk_dir / f"chunk_{number:03d}.tsv"),
                "progress_log": str(chunk_dir / f"chunk_{number:03d}.progress.log"),
                "query_sha256": sha256_file(fasta),
            }
        )

    def search_chunk(chunk: dict) -> dict:
        hits = Path(chunk["hits"])
        sidecar = hits.with_suffix(hits.suffix + ".sha256")
        if hits.is_file() and sidecar.is_file():
            recorded = sidecar.read_text(encoding="utf-8").split()[0]
            if recorded == sha256_file(hits):
                return {**chunk, "status": "reused", "seconds": 0.0, "command": None}
            raise SystemExit(
                f"{hits} does not match its digest sidecar; a partially written chunk "
                "is not a completed one, so remove it rather than resuming over it"
            )
        started = time.monotonic()
        command, log = run_diamond_blastp(
            tool,
            database,
            Path(chunk["query_fasta"]),
            hits,
            threads=args.threads,
            sensitivity=H.SENSITIVITY,
            evalue=H.EVALUE,
            max_target_seqs=args.corpus_max_target_seqs,
            fields=ALIGNMENT_FIELDS,
            progress_log=Path(chunk["progress_log"]),
        )
        digest = sha256_file(hits)
        sidecar.write_text(f"{digest}  {hits.name}\n", encoding="utf-8")
        return {
            **chunk,
            "status": "searched",
            "seconds": time.monotonic() - started,
            "command": list(command),
            "log_tail": log,
            "hits_sha256": digest,
            "hit_lines": sum(1 for _ in hits.open(encoding="utf-8")),
        }

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        completed = list(pool.map(search_chunk, chunks))
    elapsed = time.monotonic() - started

    combined = args.out / "corpus_hits_products.tsv"
    with combined.open("w", encoding="utf-8") as handle:
        for chunk in completed:
            handle.write(Path(chunk["hits"]).read_text(encoding="utf-8"))
    queries_with_a_hit = len(
        {line.split("\t", 1)[0] for line in combined.read_text(encoding="utf-8").splitlines() if line}
    )
    write_json(
        args.out / EXPECT_NOVELTY,
        {
            "schema_version": H.SCHEMA_VERSION,
            "stage": "novelty",
            "status": "complete",
            **H.declaration_digests(),
            "products": str(args.products),
            "products_sha256": sha256_file(args.products),
            "queries": len(measurable),
            "queries_below_min_residues": len(products) - len(measurable),
            "min_product_residues": H.MIN_PRODUCT_RESIDUES,
            "queries_with_a_hit": queries_with_a_hit,
            "hits": combined.name,
            "hits_sha256": sha256_file(combined),
            "corpus_identity": corpus_identity,
            "corpus_database": database.record(),
            "diamond": tool.record(),
            "search": {
                "masking": H.MASKING,
                "masking_reason": CORPUS_MASKING_REASON,
                "sensitivity": H.SENSITIVITY,
                "evalue": H.EVALUE,
                "max_target_seqs": args.corpus_max_target_seqs,
                "identity_definition": H.IDENTITY_DEFINITION,
                "chunk_size": args.chunk_size,
                "workers": args.workers,
                "threads_per_worker": args.threads,
                "chunks": len(chunks),
                "elapsed_seconds": elapsed,
                "self_match_note": (
                    "the queries are generated product sequences and the database is "
                    "the reference corpus, so no query is a record of the database and "
                    "no hit can be the query itself; a hit to one of a product's own "
                    "context items is a different thing and the annotate phase "
                    "separates it out rather than assuming it absent"
                ),
            },
            "chunk_records": [
                {key: value for key, value in chunk.items() if key != "log_tail"}
                for chunk in completed
            ],
        },
    )
    print(
        f"novelty: {len(measurable)} queries in {len(chunks)} chunks "
        f"({sum(1 for c in completed if c['status'] == 'reused')} reused), "
        f"{queries_with_a_hit} with a hit, {elapsed:.0f}s",
        flush=True,
    )


def annotate(args: argparse.Namespace) -> None:
    """Search generated products back against the corpus and the prompt family.

    Two questions one aligner answers at once: how much of a product is an
    alignable relative of the context it was generated under (the copying rule a
    substring statistic cannot settle), and how close the product is to anything
    the corpus holds (novelty). The prompt's own family is represented by the
    retained context items of that target, so family recognition here is
    "does the product align to the family it was prompted with", measured by the
    same aligner as everything else and explicitly not a profile-HMM oracle.

    **The two questions need two reference sets, and used to share one.** The
    reference held only the homologue bins and the wild type, labelled by a
    composite ``target__bin__index`` identifier, while a product records the
    corpus *accessions* of the items actually placed in its context. Those two
    identifier spaces never intersect, so the identity-to-context rule could not
    fire for any product under any condition, and the matched-unrelated
    condition's donors were not in the reference at all. The zero it reported was
    therefore a property of the join, not a measurement. Each reference record now
    carries its accession, and the unrelated donor pools are included and marked
    as outside the target's family so that family recognition is unaffected. The
    positive control that establishes the repaired rule can fire -- a verbatim
    context item, a 40% mutated one and a random sequence -- is recorded in
    ``tests/context/test_retrieve_homology_context.py``.
    """

    context = json.loads(args.homologs.read_text(encoding="utf-8"))
    H.require_declaration(context, scope="context")
    products = [json.loads(line) for line in args.products.read_text().splitlines() if line.strip()]
    if not products:
        raise SystemExit(f"{args.products} holds no generated product")
    tool = tool_for(args)
    args.out.mkdir(parents=True, exist_ok=True)

    # Reference database: every sequence any condition of any target can place in
    # a context, plus the wild type, each record carrying the target it belongs
    # to, the corpus accession a product cites, and whether it is of the target's
    # own family. The last distinction is why the unrelated donors can be in the
    # same search without becoming evidence of family recognition.
    family_fasta = args.out / "family_reference.faa"
    labels: dict[str, dict[str, Any]] = {}
    letters = 0
    with family_fasta.open("w", encoding="utf-8") as handle:
        for target in context["targets"]:
            buckets = [(name, bucket, True) for name, bucket in target["bins"].items()]
            buckets.append(("unrelated", target.get("unrelated_pool") or [], False))
            for name, bucket, in_family in buckets:
                for index, row in enumerate(bucket):
                    identifier = f"{target['target_id']}__{name}__{index}"
                    labels[identifier] = {
                        "target_id": target["target_id"],
                        "subject": row["subject"],
                        "in_family": in_family,
                    }
                    handle.write(f">{identifier}\n{row['sequence']}\n")
                    letters += len(row["sequence"])
            identifier = f"{target['target_id']}__wildtype"
            labels[identifier] = {
                "target_id": target["target_id"],
                "subject": identifier,
                "in_family": True,
            }
            handle.write(f">{identifier}\n{target['wildtype']}\n")
            letters += len(target["wildtype"])
    family_db = args.out / "family_reference.dmnd"
    import subprocess

    subprocess.run(
        [str(tool.executable), "makedb", "--in", str(family_fasta), "--db", str(family_db),
         "--threads", str(args.threads), "--quiet"],
        check=True,
        capture_output=True,
        text=True,
    )
    family_database = DiamondDatabase(
        path=family_db,
        source_fasta=family_fasta,
        source_records=len(labels),
        sequences=len(labels),
        letters=letters,
        makedb_command=("makedb", str(family_fasta)),
    )

    query = args.out / "products.faa"
    with query.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(products):
            sequence = row["sequence"]
            if len(sequence) >= H.MIN_PRODUCT_RESIDUES:
                handle.write(f">p{index:06d}\n{sequence}\n")
    family_hits_path = args.out / "family_hits.tsv"
    family_command, family_log = run_diamond_blastp(
        tool,
        family_database,
        query,
        family_hits_path,
        threads=args.threads,
        sensitivity=H.SENSITIVITY,
        evalue=H.EVALUE,
        max_target_seqs=100,
        fields=ALIGNMENT_FIELDS,
    )
    best_family: dict[str, list[dict]] = {}
    for hit in parse_hits(family_hits_path, fields=ALIGNMENT_FIELDS):
        label = labels[hit.subject]
        best_family.setdefault(hit.query, []).append(
            {
                "record": hit.subject,
                "subject": label["subject"],
                "target_id": label["target_id"],
                "in_family": label["in_family"],
                "identity_over_query": hit.identity_over_query,
                "coverage": H.hit_coverage(hit),
                "bitscore": hit.bitscore,
            }
        )

    corpus_rows: dict[str, list[dict]] = {}
    corpus_identity = None
    corpus_record = None
    corpus_search = None
    if args.corpus_hits is not None:
        # The search is the expensive half and the join is the cheap one, so a
        # hits file the novelty phase already produced is read rather than
        # recomputed. Its own completion record carries the literal per-chunk
        # commands, and they are copied into this artefact so the masking setting
        # travels with the number that depends on it.
        if not args.corpus_hits.is_file():
            raise SystemExit(f"{args.corpus_hits} does not exist")
        corpus_hits_path = args.corpus_hits
        if args.novelty_search is None:
            raise SystemExit(
                "--corpus-hits needs --novelty-search: a novelty number whose search "
                "command is not in the artefact cannot be checked from the artefact"
            )
        searched = json.loads(args.novelty_search.read_text(encoding="utf-8"))
        H.require_declaration(searched, scope="context")
        if searched.get("hits_sha256") != sha256_file(corpus_hits_path):
            raise SystemExit(
                f"{args.corpus_hits} does not match the digest its search recorded; "
                "the hits and the search record are from different runs"
            )
        corpus_identity = searched.get("corpus_identity")
        corpus_record = searched.get("corpus_database")
        corpus_search = {
            **searched["search"],
            "source": "novelty phase",
            "novelty_search": str(args.novelty_search),
            "novelty_search_sha256": sha256_file(args.novelty_search),
            "hits": searched.get("hits_sha256"),
            "commands": [
                chunk["command"]
                for chunk in searched.get("chunk_records", ())
                if chunk.get("command")
            ],
            "queries": searched.get("queries"),
            "queries_with_a_hit": searched.get("queries_with_a_hit"),
        }
    elif not args.skip_corpus:
        database, corpus_identity = resolve_corpus(args, tool)
        corpus_record = database.record()
        corpus_hits_path = args.out / "corpus_hits_products.tsv"
        corpus_command, corpus_log = run_diamond_blastp(
            tool,
            database,
            query,
            corpus_hits_path,
            threads=args.threads,
            sensitivity=H.SENSITIVITY,
            evalue=H.EVALUE,
            max_target_seqs=args.corpus_max_target_seqs,
            fields=ALIGNMENT_FIELDS,
            progress_log=args.out / "corpus_search_progress.log",
        )
        # The literal command, in the artefact. The masking setting is the
        # difference between this endpoint measuring novelty and measuring its
        # opposite, so a reader must be able to check it from the artefact alone
        # rather than from this file at some revision.
        corpus_search = {
            "command": list(corpus_command),
            "log_tail": corpus_log,
            "masking": H.MASKING,
            "masking_reason": CORPUS_MASKING_REASON,
            "sensitivity": H.SENSITIVITY,
            "evalue": H.EVALUE,
            "max_target_seqs": args.corpus_max_target_seqs,
            "threads": args.threads,
            "identity_definition": H.IDENTITY_DEFINITION,
            "self_match_note": (
                "the query set is generated product sequences and the database is the "
                "reference corpus, so no query is a record of the database and no hit "
                "can be the query itself; a hit to one of the product's own context "
                "items is a different thing and is separated out rather than assumed "
                "absent"
            ),
            "hits": sum(1 for _ in corpus_hits_path.open(encoding="utf-8")),
        }
    if corpus_search is not None:
        for hit in parse_hits(corpus_hits_path, fields=ALIGNMENT_FIELDS):
            corpus_rows.setdefault(hit.query, []).append(
                {
                    "subject": hit.subject,
                    "identity_over_query": hit.identity_over_query,
                    "pident": hit.pident,
                    "coverage": H.hit_coverage(hit),
                    "bitscore": hit.bitscore,
                    "truncated_alignment": bool(truncated_alignment(hit)),
                    "truncation_raises_stratum": bool(
                        truncation_raises_stratum(hit, hit.identity_over_query)
                    ),
                    # The self-match signature, counted rather than assumed
                    # absent: every residue of the query identically matched by a
                    # subject of the query's own length. A coverage screen on
                    # natural backbones found all 111 of its records here, because
                    # the search finds the sequence itself. These queries are
                    # generated and cannot be records of the database, so a hit
                    # with this signature is a verbatim corpus member rather than
                    # a self-match -- which makes the count the headline number of
                    # this endpoint, not a diagnostic to be filtered away.
                    "exact_full_length": bool(
                        hit.nident == hit.qlen and hit.slen == hit.qlen
                    ),
                }
            )
        corpus_search["truncated_alignments"] = sum(
            1 for hits in corpus_rows.values() for hit in hits if hit["truncated_alignment"]
        )
        corpus_search["exact_full_length_hits"] = sum(
            1 for hits in corpus_rows.values() for hit in hits if hit["exact_full_length"]
        )
        corpus_search["products_with_an_exact_full_length_hit"] = sum(
            1 for hits in corpus_rows.values() if any(hit["exact_full_length"] for hit in hits)
        )
        corpus_search["truncations_that_would_raise_a_stratum"] = sum(
            1
            for hits in corpus_rows.values()
            for hit in hits
            if hit["truncation_raises_stratum"]
        )

    # The family search and this phase's own corpus search share one query file
    # and so name their queries positionally; the novelty phase names its queries
    # by attempt identifier, because its chunking is by identifier and positions
    # would not survive it. The lookup key therefore follows the route the hits
    # came by rather than being assumed.
    by_attempt = args.corpus_hits is not None
    annotations = [
        product_annotation(
            row,
            index=index,
            family=best_family.get(f"p{index:06d}", []),
            corpus=corpus_rows.get(
                str(row["attempt_id"]) if by_attempt else f"p{index:06d}", ()
            ),
        )
        for index, row in enumerate(products)
    ]
    if by_attempt:
        matched = sum(1 for row in annotations if row["corpus_hits"])
        if matched < 1:
            raise SystemExit(
                f"{args.corpus_hits} holds hits for none of these products by attempt "
                "identifier; the hits and the product file are from different runs"
            )
    write_json(
        args.out / EXPECT_ANNOTATE,
        {
            "schema_version": H.SCHEMA_VERSION,
            "stage": "annotate",
            "status": "complete",
            **H.declaration_digests(),
            "products": str(args.products),
            "products_sha256": sha256_file(args.products),
            "homologs_sha256": sha256_file(args.homologs),
            "diamond": tool.record(),
            "family_reference": {
                "records": len(labels),
                "command": list(family_command),
                "log_tail": family_log,
                "family_coverage_floor": H.FAMILY_COVERAGE_FLOOR,
                "note": (
                    "family membership is a DIAMOND alignment to the prompt target's own "
                    "retained relatives, not a profile-HMM family call; it is not "
                    "comparable to the Pfam oracle used by the generation lane"
                ),
            },
            "corpus_identity": corpus_identity,
            "corpus_database": corpus_record,
            "corpus_search": corpus_search,
            "annotated_products": len(annotations),
            "products_below_min_residues": sum(
                1 for row in products if len(row["sequence"]) < H.MIN_PRODUCT_RESIDUES
            ),
            "annotations": annotations,
            "runtime": runtime(),
        },
    )
    print(f"annotate: {len(annotations)} products annotated", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["search", "bin", "annotate", "novelty"])
    parser.add_argument("--out", type=Path, required=True)
    # Accepted because every campaign runner injects it; this stage is CPU-only
    # and refuses a GPU rather than silently ignoring the request.
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cohort", type=Path, default=ROOT / COHORT)
    parser.add_argument("--admission", type=Path, default=ROOT / ADMISSION)
    parser.add_argument("--corpus", default="uniref90_2026_03")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--source-fasta", type=Path)
    parser.add_argument("--corpus-id")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--verify-index-digest", action="store_true")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument("--tmpdir", type=Path, default=ROOT / "data/homology_db/tmp")
    parser.add_argument("--max-target-seqs", type=int, default=H.MAX_TARGET_SEQS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--remote-gate-cohort", type=Path, default=ROOT / REMOTE_GATE)
    parser.add_argument("--no-remote-gate", action="store_true")
    parser.add_argument("--search", type=Path, help="bin phase: the search phase's --out")
    parser.add_argument("--homologs", type=Path, help="annotate phase: the binned artifact")
    parser.add_argument("--products", type=Path, help="annotate phase: generated products JSONL")
    parser.add_argument("--skip-corpus", action="store_true", help="annotate phase: family search only")
    parser.add_argument(
        "--corpus-hits",
        type=Path,
        help="annotate phase: a hits TSV the novelty phase already produced, read "
        "instead of searching the corpus again",
    )
    parser.add_argument(
        "--novelty-search",
        type=Path,
        help="annotate phase: the novelty phase's completion JSON, whose literal "
        "per-chunk commands are carried into this artefact",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=384, help="novelty phase: queries per invocation"
    )
    parser.add_argument(
        "--workers", type=int, default=6, help="novelty phase: chunks searched concurrently"
    )
    parser.add_argument(
        "--corpus-max-target-seqs",
        type=int,
        default=25,
        help="annotate phase: corpus hits retained per product before the best is taken",
    )
    parser.add_argument(
        "--diamond-tarball", type=Path, default=ROOT / "external/tools/diamond-linux64-v2.1.24.tar.gz"
    )
    parser.add_argument(
        "--diamond-checksum",
        type=Path,
        default=ROOT / "external/tools/diamond-linux64-v2.1.24.tar.gz.sha256",
    )
    parser.add_argument("--diamond-dir", type=Path, default=ROOT / "external/tools/diamond")
    args = parser.parse_args()
    if args.device != "cpu":
        raise SystemExit(
            f"this stage is CPU-only and was given --device {args.device!r}; DIAMOND "
            "retrieval runs on the host that holds the corpus indexes"
        )
    if args.phase == "search":
        search(args)
    elif args.phase == "bin":
        if args.search is None:
            raise SystemExit("--search is required for the bin phase")
        bin_hits(args)
    elif args.phase == "novelty":
        if args.products is None:
            raise SystemExit("--products is required for the novelty phase")
        novelty(args)
    else:
        if args.homologs is None or args.products is None:
            raise SystemExit("--homologs and --products are required for the annotate phase")
        annotate(args)


if __name__ == "__main__":
    main()
