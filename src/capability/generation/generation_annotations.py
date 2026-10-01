"""Immutable sequence-oracle sidecars for the generated-output supplement.

Retained R227 query coordinates recover coverage without another search. New
unconditional attempts use the same Pfam gathering thresholds and DIAMOND
configuration. Neither a no-hit result nor distance to this corpus proves
training disjointness or biochemical function.
"""

from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import conditioned_generation as cg
from ..core import protein_annotations as ci
from . import generation_evidence as ge
from ..context import homology
from ..core.io import sha256_file
from .near_duplicates import RESIDUE_SHINGLE, near_duplicate_groups
from .progen3_generation import POLICY as NATIVE_POLICY
from .unconditional_generation import ADMITTED

NATIVE_ANNOTATION_ARMS = frozenset(ADMITTED) | {"progen3-3b"}
MINIMUM_NONEMPTY_FOR_SEQUENCE_DIVERSITY = 8


def require_native_ledger(rows: list[dict]) -> str:
    """Admit one unconditioned 800-attempt ledger; keep ProGen3-3B and the expansion arms."""
    if len(rows) != NATIVE_POLICY["attempts"] or {row["source_sample_index"] for row in rows} != set(range(NATIVE_POLICY["attempts"])):
        raise ValueError("native annotation requires every declared generation attempt")
    arms = {row["arm"] for row in rows}
    if len(arms) != 1:
        raise ValueError("native annotation requires a single-arm ledger")
    arm = next(iter(arms))
    if arm not in NATIVE_ANNOTATION_ARMS:
        raise ValueError(
            f"native annotation admits only unconditioned generation arms, not {arm!r}"
        )
    if any(row.get("class_key") is not None or row.get("condition") != "unconditioned" for row in rows):
        raise ValueError("native annotation requires an unconditioned ledger")
    return arm


def native_alignment_fields(arm: str) -> tuple[str, ...]:
    """ProGen3-3B keeps the original columns; expansion arms keep identity and both coverages together."""
    if arm == "progen3-3b":
        return homology.DIAMOND_FIELDS
    return homology.ALIGNMENT_FIELDS


def unconditional_diversity_census(rows: list[dict]) -> dict[str, Any]:
    """Near-duplicate groups on one frozen unconditioned 800-attempt ledger.

    Empty decodes share one group. Zero Pfam hits do not become zero hit-groups.
    Fewer than eight nonempty sequences is a named unreadably-small protein set,
    not a diversity of zero. Groups are not biological diversity or function.
    """

    arm = require_native_ledger(rows)
    if any("any_profile_hit" not in row for row in rows):
        raise ValueError("diversity census requires any_profile_hit on every attempt")
    sequences = [row.get("sequence") or "" for row in rows]
    nonempty = [index for index, sequence in enumerate(sequences) if sequence]
    _, grouping = cg.near_duplicate_group_ids(sequences, unit="residues")
    unique_hashes = len({rows[index]["sequence_sha256"] for index in nonempty})
    if nonempty:
        _, filled = near_duplicate_groups([sequences[index] for index in nonempty], unit="residues")
        n_groups_nonempty = filled["n_groups"]
        n_singleton = filled["n_singleton_groups"]
        largest = filled["largest_group_size"]
    else:
        n_groups_nonempty = None
        n_singleton = None
        largest = None
    hit_indices = [index for index, row in enumerate(rows) if row.get("any_profile_hit") is True]
    unknown_hits = sum(1 for row in rows if row.get("any_profile_hit") is None)
    if not hit_indices:
        n_hit_groups = None
    else:
        hit_sequences = [sequences[index] for index in hit_indices]
        if not all(hit_sequences):
            raise ValueError("a profile hit cannot be an empty decode")
        _, hits = near_duplicate_groups(hit_sequences, unit="residues")
        n_hit_groups = hits["n_groups"]
    readable = len(nonempty) >= MINIMUM_NONEMPTY_FOR_SEQUENCE_DIVERSITY
    return {
        "arm": arm,
        "n_attempts": len(rows),
        "n_empty": int(grouping["n_empty_records"]),
        "n_nonempty": len(nonempty),
        "n_unique_nonempty_hashes": unique_hashes,
        "n_groups_nonempty": n_groups_nonempty,
        "n_singleton_groups_nonempty": n_singleton,
        "largest_nonempty_group_size": largest,
        "n_groups_including_empty": int(grouping["n_groups_including_empty"]),
        "n_any_profile_hits": len(hit_indices),
        "n_any_profile_groups": n_hit_groups,
        "n_any_profile_unknown": unknown_hits,
        "sequence_diversity_readable": readable,
        "unit": "residues",
        "shingle_length": RESIDUE_SHINGLE,
        "not_biological_diversity": True,
        "not_function": True,
    }


def read_attempts(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate attempt identifier")
    for row in rows:
        if ge.sequence_hash(row["sequence"]) != row["sequence_sha256"]:
            raise ValueError(f"attempt sequence hash mismatch: {row['id']}")
    return rows


def best_hits(table: Path, sequences: dict[str, str], *, fields: tuple[str, ...] = homology.DIAMOND_FIELDS) -> dict[str, homology.Hit]:
    best: dict[str, homology.Hit] = {}
    for hit in homology.parse_hits(table, fields=fields):
        if hit.query not in sequences or len(sequences[hit.query]) != hit.qlen:
            raise ValueError(f"alignment query/length mismatch: {hit.query}")
        if not 1 <= min(hit.qstart, hit.qend) <= max(hit.qstart, hit.qend) <= hit.qlen:
            raise ValueError(f"invalid alignment query coordinates: {hit.query}")
        previous = best.get(hit.query)
        # Identity over the full query is the original R227 covariate. Coverage
        # belongs to that same selected hit, not a different convenient HSP.
        rank = (hit.identity_over_query, hit.bitscore, -hit.evalue, hit.subject)
        old_rank = (previous.identity_over_query, previous.bitscore, -previous.evalue, previous.subject) if previous else None
        if old_rank is None or rank > old_rank:
            best[hit.query] = hit
    return best


def reference_fields(hit: homology.Hit | None, *, searched: bool, empty: bool = False) -> dict[str, Any]:
    target_coverage = None
    if hit and hit.sseq_gapped is not None:
        target_coverage = sum(residue != "-" for residue in hit.sseq_gapped) / hit.slen
        if not 0 < target_coverage <= 1:
            raise ValueError(f"invalid subject coverage: {hit.query}")
    return {
        "reference_search_status": "aligned_query_coverage_available" if hit else ("no_reported_alignment" if searched else ("empty_not_searched" if empty else "not_searched")),
        "reference_identity": hit.identity_over_query if hit else None,
        "reference_coverage": (abs(hit.qend - hit.qstart) + 1) / hit.qlen if hit else None,
        "reference_coverage_definition": "query_coordinate_span_fraction_of_full_query",
        "reference_aligned_region_identity_percent": hit.pident if hit else None,
        "reference_subject": hit.subject if hit else None,
        "reference_subject_length": hit.slen if hit else None,
        "reference_target_coverage": target_coverage,
        "reference_target_coverage_status": "aligned_subject_residue_fraction" if target_coverage is not None else "subject_coordinates_not_retained",
        "reference_hit_selection": "max_identity_over_query_then_bitscore_then_evalue_then_subject",
    }


def write_sidecar(rows: list[dict], output: Path, provenance: dict) -> dict:
    content = ge.jsonl_bytes(rows)
    ge.write_immutable(output / "sequence_annotations.jsonl", content)
    result = {"records": len(rows), "search_status": dict(Counter(row["reference_search_status"] for row in rows)),
              "provenance": provenance, "annotations_sha256": sha256_file(output / "sequence_annotations.jsonl")}
    ge.write_immutable(output / "annotation_manifest.json", (json.dumps(result, sort_keys=True, indent=2) + "\n").encode())
    return result


def recover_r227(attempts: Path, table: Path, query_fasta: Path, report: Path, output: Path) -> dict:
    ledger = read_attempts(attempts)
    queries = ge.read_fasta(query_fasta)
    best = best_hits(table, queries)
    old_report = json.loads(report.read_text())
    corpus_records = {arm: block["max_identity_to_corpus"] for arm, block in old_report["protein_arms"].items()}
    output_rows = []
    accounted_queries = set()
    for row in ledger:
        name = f"{row['arm']}#{row['source_key']}#{row['source_sample_index']}".replace(" ", "_")
        originally_searched = row["role"] in ("generation", "unconditioned_floor") and bool(row["sequence"])
        searched = originally_searched and name in queries
        if searched:
            if queries.get(name) != row["sequence"]:
                raise ValueError(f"retained search input differs from frozen attempt: {name}")
            accounted_queries.add(name)
        hit = best.get(name) if searched else None
        fields = reference_fields(hit, searched=searched, empty=not row["sequence"])
        fields["reference_coverage_status"] = "query_coordinates_retained" if searched else "not_applicable"
        if originally_searched and not searched:
            # The R227 work path was reused for each arm; the final arm's table
            # survives while earlier per-attempt identities survive in report.
            fields.update({"reference_identity": row["reference_identity"],
                           "reference_search_status": row["reference_search_status"],
                           "reference_coverage_status": "raw_alignment_not_retained"})
        if searched and row["reference_identity"] != fields["reference_identity"]:
            raise ValueError(f"retained hit disagrees with original identity covariate: {name}")
        output_rows.append({"id": row["id"], "sequence_sha256": row["sequence_sha256"], **fields})
    if accounted_queries != set(queries):
        raise ValueError("retained query FASTA contains unaccounted sequences")
    provenance = {
        "campaign": "EXP-R2-232", "new_inference": False,
        "inputs": {label: sha256_file(path) for label, path in {"attempts": attempts, "diamond_table": table, "diamond_query_fasta": query_fasta, "original_report": report}.items()},
        "corpus_and_commands": {arm: {key: block.get(key) for key in ("corpus", "corpus_records", "command", "corpus_note")} for arm, block in corpus_records.items()},
        "note": "Original report omitted coverage. Retained final-arm qstart/qend/qlen permit partial recovery; earlier-arm raw table was overwritten. Frozen attempts/subsets remain unchanged.",
    }
    return write_sidecar(output_rows, output, provenance)


def reference_only(attempts: Path, output: Path, *, diamond: Path, diamond_db: Path,
                   reference_metadata: Path, threads: int = 8) -> dict:
    """Fresh full-ledger R232 search; never replace an old identity/coverage pair."""
    rows = read_attempts(attempts)
    if len(rows) != 16400 or Counter(row["role"] for row in rows) != {"generation": 12800, "natural_reference": 3200, "unconditioned_floor": 400}:
        raise ValueError("R232 reference-only extension requires the complete frozen 16400-record ledger")
    if threads < 1:
        raise ValueError("reference search needs a positive thread count")
    paths = {"attempts": attempts, "diamond": diamond, "diamond_db": diamond_db,
             "reference_metadata": reference_metadata}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    configuration = {
        "campaign": "EXP-R2-232", "extension": "fresh_full_ledger_reference_search",
        "inputs": {label: sha256_file(path) for label, path in paths.items()},
        "reference_metadata": json.loads(reference_metadata.read_text()),
        "threads": threads, "sensitivity": "very-sensitive", "evalue": 1e-3,
        "max_target_seqs": 5, "masking": 0, "fields": list(homology.ALIGNMENT_FIELDS),
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "source_hashes": {"annotation_module": sha256_file(Path(__file__)),
                          "homology_module": sha256_file(Path(homology.__file__))},
        "unchanged": "original ledger, profiles, structure subset and primary endpoints",
    }
    config_path = output / "search_configuration.json"
    ge.write_immutable(config_path, (json.dumps(configuration, sort_keys=True, indent=2) + "\n").encode())
    manifest_path = output / "reference_fresh_manifest.json"
    sidecar = output / "reference_fresh_annotations.jsonl"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest["annotations_sha256"] != sha256_file(sidecar):
            raise ValueError("completed fresh reference annotation hash mismatch")
        return manifest
    sequences = {row["id"]: row["sequence"] for row in rows if row["sequence"] and row["valid_aa20"]}
    fasta = ci.write_fasta(output / "queries.fasta", sequences)
    table = output / "reference_hits.tsv"
    command, log_tail = homology.run_diamond_blastp(
        SimpleNamespace(executable=diamond), SimpleNamespace(path=diamond_db),
        fasta, table, threads=threads, sensitivity="very-sensitive", evalue=1e-3,
        max_target_seqs=5, fields=homology.ALIGNMENT_FIELDS,
    )
    best = best_hits(table, sequences, fields=homology.ALIGNMENT_FIELDS)
    annotations = []
    for row in rows:
        fields = reference_fields(best.get(row["id"]), searched=row["id"] in sequences, empty=not row["sequence"])
        annotations.append({"id": row["id"], "sequence_sha256": row["sequence_sha256"],
                            **{key: row[key] for key in ("reference_identity", "reference_coverage", "reference_search_status")},
                            **{key.replace("reference_", "reference_fresh_", 1): value for key, value in fields.items()}})
    ge.write_immutable(sidecar, ge.jsonl_bytes(annotations))
    manifest = {"campaign": "EXP-R2-232", "new_inference": True, "records": len(rows),
                "n_searched": len(sequences), "n_not_searched": len(rows) - len(sequences),
                "fresh_search_status": dict(Counter(row["reference_fresh_search_status"] for row in annotations)),
                "configuration_sha256": sha256_file(config_path),
                "annotations_sha256": sha256_file(sidecar), "query_fasta_sha256": sha256_file(fasta),
                "diamond_table_sha256": sha256_file(table), "diamond_command": command, "diamond_log_tail": log_tail,
                "note": "Fresh hit identity and coverage travel together. Old reported identities remain unchanged; same named searchable corpus does not certify full training disjointness."}
    ge.write_immutable(manifest_path, (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    return manifest


def annotate_native(attempts: Path, output: Path, *, hmmscan: Path, pfam_hmm: Path,
                    diamond: Path, diamond_db: Path, reference_metadata: Path,
                    threads: int = 8, shards: int = 4) -> dict:
    """Run already staged executables/databases; never install or build assets."""
    rows = read_attempts(attempts)
    arm = require_native_ledger(rows)
    fields = native_alignment_fields(arm)
    campaign = "EXP-R2-233" if arm == "progen3-3b" else "EXP-R2-246"
    paths = {"attempts": attempts, "hmmscan": hmmscan, "pfam_hmm": pfam_hmm,
             "diamond": diamond, "diamond_db": diamond_db, "reference_metadata": reference_metadata}
    paths.update({f"pfam_index{suffix}": Path(str(pfam_hmm) + suffix) for suffix in (".h3f", ".h3i", ".h3m", ".h3p")})
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    configuration = {"inputs": {label: sha256_file(path) for label, path in paths.items()},
                     "reference_metadata": json.loads(reference_metadata.read_text()),
                     "arm": arm, "campaign": campaign, "threads": threads, "shards": shards,
                     "pfam_threshold": cg.PFAM_THRESHOLD,
                     "diamond_sensitivity": "very-sensitive", "diamond_evalue": 1e-3,
                     "diamond_max_target_seqs": 5, "diamond_masking": 0,
                     "diamond_fields": list(fields),
                     "runner_sha256": sha256_file(Path(__file__))}
    ge.write_immutable(output / "search_configuration.json", (json.dumps(configuration, sort_keys=True, indent=2) + "\n").encode())
    manifest = output / "annotation_manifest.json"
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        if saved["annotations_sha256"] != sha256_file(output / "sequence_annotations.jsonl"):
            raise ValueError("completed annotation sidecar hash mismatch")
        return saved
    sequences = {row["id"]: row["sequence"] for row in rows if row["sequence"] and row["valid_aa20"]}
    hits, profile_receipt = cg.annotate(sequences, tool=SimpleNamespace(hmmscan=hmmscan),
                                      database=SimpleNamespace(path=pfam_hmm), workspace=output / "pfam",
                                      threads=threads, shards=shards, label="native_generation")
    fasta = ci.write_fasta(output / "queries.fasta", sequences)
    table = output / "reference_hits.tsv"
    command, log_tail = homology.run_diamond_blastp(SimpleNamespace(executable=diamond), SimpleNamespace(path=diamond_db),
                                                 fasta, table, threads=threads * shards,
                                                 sensitivity="very-sensitive", evalue=1e-3, max_target_seqs=5,
                                                 fields=fields)
    best = best_hits(table, sequences, fields=fields)
    annotations = []
    for row in rows:
        families = ge._families(hits, row["id"])
        annotations.append({"id": row["id"], "sequence_sha256": row["sequence_sha256"],
                            "pfam_families": families, "any_profile_hit": bool(families),
                            "target_profile_hit": None, "profile_hit_classes": None,
                            "profile_search_status": "searched" if row["id"] in sequences else "empty_or_noncanonical_not_searched",
                            **reference_fields(best.get(row["id"]), searched=row["id"] in sequences, empty=not row["sequence"])})
    return write_sidecar(annotations, output, {"campaign": campaign, "arm": arm, "new_inference": True,
                                              "configuration_sha256": sha256_file(output / "search_configuration.json"),
                                              "profile_receipt": profile_receipt, "diamond_command": command,
                                              "diamond_log_tail": log_tail, "diamond_table_sha256": sha256_file(table)})
