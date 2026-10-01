"""Auditable R227 attempts and score-blind structural cohorts for EXP-R2-232.

The old generation campaign is immutable. This module reads its retained records,
keeps every attempt, and selects the new structural measurements without looking
at a Pfam hit, sequence likelihood, or predicted structure. Natural references are
annotation-based controls, not newly measured functional proteins.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from . import conditioned_generation as cg
from ..core.amino_acids import AA20 as _AA20_ALPHABET
from ..core.protein_annotations import parse_hmmscan_table
from ..core.io import _atomic_write, sha256_file

SCHEMA_VERSION = "generation_biological_evidence_v1"
CAMPAIGN = "EXP-R2-232"
ARMS = ("zymctrl", "prollama")
FLOORS = ("progen2-medium", "protgpt2")
AA20 = frozenset(_AA20_ALPHABET)
POLICY = {
    "campaign": CAMPAIGN,
    "seed": 20260905,
    "min_residues": 16,
    "max_residues": 1024,
    "length_strata": [[16, 128], [129, 256], [257, 512], [513, 1024]],
    "generated_per_cell": 16,
    "pilot_natural_per_class": 2,
    "main_natural_per_class": 8,
    "selection": "seeded_uniform_within_length_stratum_no_score_selection",
    "allocation": "one_per_nonempty_stratum_then_largest_remainder_on_remaining_capacity",
    "simple_reference": "one_fixed_composition_shuffle_per_selected_parent",
    "natural_main_excludes": "all_exact_sequence_hashes_selected_for_natural_pilot",
    "evaluator_never_truncates": True,
    "floor_structure_inference": False,
}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def seeded_rng(seed: int, key: str) -> np.random.Generator:
    return np.random.default_rng(int(digest([seed, key])[:16], 16))


def identifier(*parts: Any) -> str:
    return "ge_" + digest(list(parts))[:24]


def read_fasta(path: Path) -> dict[str, str]:
    """Read exact FASTA sequences, rejecting duplicate ids and internal whitespace."""
    result: dict[str, str] = {}
    name: str | None = None
    pieces: list[str] = []
    for line in path.read_text().splitlines() + [">__END__"]:
        if line.startswith(">"):
            if name is not None:
                if name in result:
                    raise ValueError(f"duplicate FASTA id {name!r}: {path}")
                result[name] = "".join(pieces)
            name, pieces = line[1:], []
            if not name:
                raise ValueError(f"empty FASTA id: {path}")
        else:
            if name is None or any(character.isspace() for character in line):
                raise ValueError(f"invalid FASTA content: {path}")
            pieces.append(line)
    return result


def length_stratum(length: int, policy: Mapping[str, Any] = POLICY) -> str | None:
    for low, high in policy["length_strata"]:
        if low <= length <= high:
            return f"{low}_{high}"
    return None


def classify_support(sequence: str, policy: Mapping[str, Any] = POLICY) -> str:
    if not sequence:
        return "empty_sequence"
    if not set(sequence) <= AA20:
        return "noncanonical_residue"
    if len(sequence) < policy["min_residues"]:
        return "below_length_support"
    if len(sequence) > policy["max_residues"]:
        return "above_length_support"
    return "eligible"


def _source_record(label: str, path: Path, sources: dict[str, dict[str, Any]]) -> None:
    if label in sources:
        raise ValueError(f"duplicate source label {label}")
    sources[label] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _read_json(path: Path, label: str, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    _source_record(label, path, sources)
    return json.loads(path.read_text())


def _read_scan(directory: Path, label: str, sources: dict[str, dict[str, Any]]) -> dict:
    tables = sorted(directory.glob("*.tbl"))
    if not tables:
        raise FileNotFoundError(f"no retained HMMER tables: {directory}")
    hits: dict[str, list] = {}
    for table in tables:
        _source_record(f"r227_work/{label}/{table.name}", table, sources)
        for name, entries in parse_hmmscan_table(table).items():
            if name in hits:
                raise ValueError(f"query appears in multiple HMMER shards: {name}")
            hits[name] = entries
    return hits


def _families(hits: Mapping[str, list], name: str) -> list[str]:
    return sorted({entry["accession_unversioned"] for entry in hits.get(name, ())})


def _profile_fields(families: list[str], class_key: str | None, referents: Mapping) -> dict:
    profile_hits = sorted(
        key for key, wanted in referents.items() if set(families) & set(wanted)
    )
    return {
        "pfam_families": families,
        "any_profile_hit": bool(families),
        "target_profile_hit": class_key in profile_hits if class_key else None,
        "profile_hit_classes": profile_hits,
    }


def _row(sequence: str, *, arm: str, class_key: str | None, condition: str,
         role: str, source_label: str, source_sample_index: int, primary_class: bool,
         source_key: str) -> dict[str, Any]:
    # Source strings are preserved; unexpected unicode is rejected rather than cleaned.
    seqhash = sequence_hash(sequence)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": identifier(role, arm, source_key, source_sample_index),
        "sequence": sequence,
        "sequence_sha256": seqhash,
        "length": len(sequence),
        "valid_aa20": bool(sequence) and set(sequence) <= AA20,
        "support_status": classify_support(sequence),
        "arm": arm,
        "class_key": class_key,
        "condition": condition,
        "role": role,
        "primary_class": primary_class,
        "source_label": source_label,
        "source_key": source_key,
        "source_sample_index": source_sample_index,
        "near_duplicate_group": None,
        "reference_identity": None,
        "reference_coverage": None,
        "reference_search_status": "not_searched",
        "paired_id": None,
        "phase": None,
        "stratum": length_stratum(len(sequence)),
        "inclusion_probability": None,
        "selected_for_structure": False,
        "structure_exclusion_reason": None,
    }


def load_attempts(result_dir: Path, work_dir: Path, queue_path: Path) -> tuple[list[dict], dict]:
    """Recover every native generation, floor, and natural real-anchor record."""
    sources: dict[str, dict] = {}
    queue = _read_json(queue_path, "r227_evidence/class_queue.json", sources)
    if queue.get("digest") != cg.queue_digest(queue):
        raise ValueError("R227 queue digest mismatch")
    report = _read_json(result_dir / "conditioned_generation.json", "r227_results/report.json", sources)
    if report.get("queue_digest") != queue["digest"]:
        raise ValueError("R227 report and queue differ")
    rows: list[dict] = []
    cell_metadata: dict[str, Any] = {}
    anchors_by_arm: dict[str, Any] = {}
    scans: dict[str, Any] = {}
    for arm in ARMS:
        anchor = _read_json(result_dir / f"anchors_{arm}.json", f"r227_results/anchors_{arm}.json", sources)
        if anchor.get("queue_digest") != queue["digest"]:
            raise ValueError(f"R227 anchors and queue differ: {arm}")
        anchors_by_arm[arm] = anchor
        scans[arm] = _read_scan(work_dir / f"score_{arm}", f"score_{arm}", sources)
    for arm in (*ARMS, *FLOORS):
        source = f"r227_results/generations_{arm}.json"
        generations = _read_json(result_dir / f"generations_{arm}.json", source, sources)
        expected_queue_digest = queue["digest"] if arm in ARMS else None
        if generations.get("queue_digest") != expected_queue_digest:
            raise ValueError(f"R227 generations and queue differ: {arm}")
        if not generations.get("self_check", {}).get("passed"):
            raise ValueError(f"R227 source self-check did not pass: {arm}")
        if generations["sampling"].get("post_selection_filter") is not None:
            raise ValueError("source generations were postselected")
        oracle_arm = arm if arm in ARMS else ARMS[0]
        anchor = anchors_by_arm[oracle_arm]
        referents = {key: value["referent"] for key, value in anchor["classes"].items() if value.get("admitted")}
        identities = report["protein_arms"][oracle_arm]["max_identity_to_corpus"]["max_identity_over_query"]
        entries = {entry["key"]: entry for entry in queue["arms"].get(arm, {}).get("classes", ())}
        for cell_key, cell in sorted(generations["cells"].items()):
            sequence_list = cell["samples"]
            if len(sequence_list) != generations["sampling"]["generations_per_cell"]:
                raise ValueError(f"source sample count mismatch: {arm}/{cell_key}")
            statistics = cell["statistics"]
            if statistics["n"] != len(sequence_list) or statistics["n_empty"] != sum(not s for s in sequence_list):
                raise ValueError(f"source statistics inconsistent: {arm}/{cell_key}")
            target = cell["class_key"] if arm in ARMS else None
            if target is not None:
                entry = entries[target]
                expected = entry["label"] if cell["condition"] == "requested" else entry["mismatched_label"]
                if cell["label"] != expected:
                    raise ValueError(f"native label mismatch: {arm}/{cell_key}")
            groups, grouping = cg.near_duplicate_group_ids(sequence_list, unit="residues")
            cell_metadata[f"{arm}|{cell_key}"] = {
                "n_attempts": len(sequence_list), "statistics": statistics,
                "sampling": generations["sampling"], "grouping": grouping,
                "prompt": cell["prompt"], "seed": cell["seed"],
                "per_sample_termination_available": False,
            }
            for index, sequence in enumerate(sequence_list):
                row = _row(sequence, arm=arm, class_key=target, condition=cell["condition"],
                           role="generation" if arm in ARMS else "unconditioned_floor",
                           source_label=source, source_key=cell_key, source_sample_index=index,
                           primary_class=target in anchor["admitted_classes"])
                query = f"{arm}#{cell_key}#{index}".replace(" ", "_")
                families = _families(scans[oracle_arm], query)
                row.update(_profile_fields(families, target, referents))
                identity = identities.get(query)
                if sequence and identity is None:
                    raise ValueError(f"missing saved identity covariate: {query}")
                row.update({
                    "reference_identity": identity if identity else None,
                    "reference_search_status": "aligned_coverage_unavailable" if identity else ("no_reported_alignment" if sequence else "empty_not_searched"),
                    "near_duplicate_group": f"{arm}|{cell_key}|g{int(groups[index])}",
                    "native_prompt_class": cell.get("requested_class"),
                    "native_prompt_label": cell.get("label"),
                    "source_termination_observed": None,
                    "source_cell_n_terminated": statistics.get("n_terminated"),
                    "source_cell_n": len(sequence_list),
                    "source_max_new_tokens": generations["sampling"]["max_new_tokens"],
                    "source_per_sample_termination_note": "raw_continuations_not_persisted; do_not_infer_EOS_from_length",
                })
                rows.append(row)
    for arm in ARMS:
        directory = work_dir / f"anchors_{arm}"
        fasta_paths = sorted(directory.glob("*.fasta"))
        if not fasta_paths:
            raise FileNotFoundError(f"no retained anchor FASTA: {directory}")
        natural: dict[str, tuple[str, str]] = {}
        for path in fasta_paths:
            label = f"r227_work/anchors_{arm}/{path.name}"
            _source_record(label, path, sources)
            for name, sequence in read_fasta(path).items():
                if name in natural:
                    raise ValueError(f"duplicate anchor id across shards: {arm}/{name}")
                natural[name] = (sequence, label)
        anchor = anchors_by_arm[arm]
        hits = _read_scan(directory, f"anchors_{arm}", sources)
        referents = {key: value["referent"] for key, value in anchor["classes"].items() if value.get("admitted")}
        for class_key in sorted(anchor["classes"]):
            safe = class_key.replace(".", "_").replace("/", "_")
            names = sorted((name for name in natural if name.startswith(f"{safe}|real|")), key=lambda name: int(name.rsplit("|", 1)[1]))
            # On a class with no referent the old oracle stored n_real=0 after
            # passing empty annotation lists, although 100 real sequences were
            # still drawn and written. The sampling contract supplies this count.
            if len(names) != cg.ANCHOR_DRAW:
                raise ValueError(f"natural anchor count mismatch: {arm}/{class_key}")
            sequences = [natural[name][0] for name in names]
            groups, _ = cg.near_duplicate_group_ids(sequences, unit="residues")
            for position, name in enumerate(names):
                sequence, source = natural[name]
                row = _row(sequence, arm=arm, class_key=class_key, condition="natural",
                           role="natural_reference", source_label=source, source_key=name,
                           source_sample_index=int(name.rsplit("|", 1)[1]),
                           primary_class=class_key in anchor["admitted_classes"])
                row.update(_profile_fields(_families(hits, name), class_key, referents))
                row["near_duplicate_group"] = f"{arm}|{class_key}|natural|g{int(groups[position])}"
                row["biological_evidence"] = "native_class_annotation_and_Pfam_anchor; not_new_function_or_structure_measurement"
                rows.append(row)
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate cohort identifiers")
    return sorted(rows, key=lambda row: row["id"]), {
        "sources": sources, "r227_queue_digest": queue["digest"], "source_cells": cell_metadata,
        "admitted_classes": {arm: anchor["admitted_classes"] for arm, anchor in anchors_by_arm.items()},
        "source_checkpoint_facts_in_generation_records": True,
    }


def allocate_strata(counts: Mapping[str, int], budget: int) -> dict[str, int]:
    """Allocate without losing a nonempty stratum, then use Hamilton remainders."""
    nonempty = {key: count for key, count in sorted(counts.items()) if count > 0}
    if budget < len(nonempty):
        raise ValueError("sample budget cannot represent every nonempty stratum")
    budget = min(budget, sum(nonempty.values()))
    allocation = {key: 1 for key in nonempty}
    remaining = budget - len(nonempty)
    capacities = {key: value - 1 for key, value in nonempty.items()}
    capacity = sum(capacities.values())
    if remaining and capacity:
        quotas = {key: remaining * value / capacity for key, value in capacities.items()}
        for key, quota in quotas.items():
            allocation[key] += int(quota)
        left = budget - sum(allocation.values())
        ranking = sorted(nonempty, key=lambda key: (-(quotas[key] - int(quotas[key])), key))
        for key in ranking[:left]:
            allocation[key] += 1
    if sum(allocation.values()) != budget or any(allocation[key] > nonempty[key] for key in nonempty):
        raise AssertionError("invalid stratum allocation")
    return allocation


def _uniform(rows: list[dict], n: int, *, seed: int, key: str) -> list[dict]:
    ordered = sorted(rows, key=lambda row: row["id"])
    if not ordered:
        return []
    indices = seeded_rng(seed, key).permutation(len(ordered))[:n]
    return [ordered[int(index)] for index in indices]


def composition_shuffle(parent: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    permutation = seeded_rng(seed, f"shuffle|{parent['id']}").permutation(len(parent["sequence"]))
    sequence = "".join(parent["sequence"][int(index)] for index in permutation)
    if Counter(sequence) != Counter(parent["sequence"]):
        raise AssertionError("shuffle changed composition")
    row = dict(parent)
    row.update({
        "id": identifier("composition_shuffle", parent["id"], seed),
        "sequence": sequence, "sequence_sha256": sequence_hash(sequence),
        "role": "composition_shuffle", "paired_id": parent["id"],
        "parent_role": parent["role"], "parent_condition": parent["condition"],
        "parent_near_duplicate_group": parent["near_duplicate_group"],
        "near_duplicate_group": f"shuffle_exact|{sequence_hash(sequence)}",
        "shuffle_unchanged": sequence == parent["sequence"],
        "pfam_families": None, "target_profile_hit": None, "any_profile_hit": None,
        "profile_hit_classes": None, "reference_identity": None, "reference_coverage": None,
        "reference_search_status": "not_searched",
        "source_termination_observed": None,
        "biological_evidence": "constructed_composition_reference; not_experimentally_nonfolding_or_inactive",
    })
    for key in ("raw_continuation", "raw_continuation_character_length", "leading_canonical_residue_length",
                "source_budget_censored", "official_compiled_sequence", "official_compilation_valid",
                "source_stop_reason", "residue_prefix_is_whole_generated_sequence",
                "censored_fragment_included_without_claim_of_biological_completeness"):
        if key in row:
            row[key] = None
    return row


def select_cohorts(rows: list[dict], policy: Mapping[str, Any] = POLICY) -> tuple[list[dict], list[dict], dict]:
    """Select calibration separately, then main outputs with known inclusion weights."""
    if dict(policy) != POLICY:
        raise ValueError("EXP-R2-232 frozen selection policy differs")
    natural: dict[tuple, list[dict]] = defaultdict(list)
    generation: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if not row["primary_class"]:
            row["structure_exclusion_reason"] = "outside_original_Pfam_admitted_classes" if row["role"] != "unconditioned_floor" else "floor_reported_without_structure_sampling"
            continue
        if row["support_status"] != "eligible":
            row["structure_exclusion_reason"] = row["support_status"]
            continue
        if row["role"] == "natural_reference":
            natural[(row["arm"], row["class_key"])].append(row)
        elif row["role"] == "generation":
            generation[(row["arm"], row["class_key"], row["condition"])].append(row)
    pilot: list[dict] = []
    main: list[dict] = []
    selection: dict[str, Any] = {"natural": {}, "generation": {}}

    def mark(row: dict, phase: str, probability: float) -> None:
        row.update({"phase": phase, "inclusion_probability": probability, "selected_for_structure": True})
        (pilot if phase == "pilot" else main).append(dict(row))

    for key, pool in sorted(natural.items()):
        chosen = _uniform(pool, policy["pilot_natural_per_class"], seed=policy["seed"], key=f"pilot|{key}")
        for row in chosen:
            mark(row, "pilot", len(chosen) / len(pool))
        selection["natural"]["|".join(key)] = {
            "eligible_n": len(pool), "pilot_n": len(chosen), "pilot_inclusion_probability": len(chosen) / len(pool),
        }
    pilot_hashes = {row["sequence_sha256"] for row in pilot}
    for key, pool in sorted(natural.items()):
        candidates = [row for row in pool if row["sequence_sha256"] not in pilot_hashes]
        chosen = _uniform(candidates, policy["main_natural_per_class"], seed=policy["seed"], key=f"main_natural|{key}")
        for row in chosen:
            mark(row, "main", len(chosen) / len(candidates))
        selection["natural"]["|".join(key)].update({
            "main_eligible_after_global_pilot_hash_exclusion": len(candidates), "main_n": len(chosen),
            "main_inclusion_probability_conditional_on_pilot": len(chosen) / len(candidates) if candidates else None,
            "main_shortfall": max(0, policy["main_natural_per_class"] - len(chosen)),
        })
        for row in pool:
            if row["phase"] is None and row["sequence_sha256"] in pilot_hashes:
                row["structure_exclusion_reason"] = "same_exact_sequence_as_natural_pilot"
    for key, pool in sorted(generation.items()):
        strata: dict[str, list[dict]] = defaultdict(list)
        for row in pool:
            strata[row["stratum"]].append(row)
        allocation = allocate_strata({name: len(values) for name, values in strata.items()}, policy["generated_per_cell"])
        blocks: dict[str, Any] = {}
        for name, n in allocation.items():
            population = strata[name]
            chosen = _uniform(population, n, seed=policy["seed"], key=f"main_generated|{key}|{name}")
            for row in chosen:
                mark(row, "main", n / len(population))
            blocks[name] = {"N": len(population), "n": n, "inclusion_probability": n / len(population)}
        selection["generation"]["|".join(key)] = {"strata": blocks, "eligible_n": len(pool), "selected_n": sum(allocation.values()), "shortfall": max(0, policy["generated_per_cell"] - len(pool))}
    for row in rows:
        if not row["selected_for_structure"] and row["structure_exclusion_reason"] is None:
            row["structure_exclusion_reason"] = "eligible_not_sampled"
    for destination in (pilot, main):
        destination.extend(composition_shuffle(row, seed=policy["seed"]) for row in list(destination))
        destination.sort(key=lambda row: row["id"])
    main_natural_hashes = {row["sequence_sha256"] for row in main if row["role"] == "natural_reference"}
    if pilot_hashes & main_natural_hashes:
        raise AssertionError("pilot and main natural controls overlap exactly")
    return pilot, main, selection


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows).encode()


def write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite changed frozen artifact: {path}")
        return
    _atomic_write(path, payload)


def prepare(result_dir: Path, work_dir: Path, queue_path: Path, output_dir: Path,
            evidence_dir: Path) -> dict[str, Any]:
    rows, provenance = load_attempts(result_dir, work_dir, queue_path)
    pilot, main, selection = select_cohorts(rows)
    artifacts = {"attempts.jsonl": rows, "pilot_subset.jsonl": pilot, "main_subset.jsonl": main}
    outputs: dict[str, Any] = {}
    for filename, records in artifacts.items():
        payload = jsonl_bytes(records)
        outputs[filename] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "records": len(records)}
        write_immutable(output_dir / filename, payload)
    fields = ("id", "sequence_sha256", "length", "arm", "class_key", "condition", "role", "paired_id", "phase", "stratum", "inclusion_probability", "near_duplicate_group")
    selection_rows = [{key: row.get(key) for key in fields} for row in (*pilot, *main)]
    frozen_selection = jsonl_bytes(selection_rows)
    write_immutable(evidence_dir / "selected_records.jsonl", frozen_selection)
    counters = {
        "all_roles": dict(Counter(row["role"] for row in rows)),
        "primary_roles": dict(Counter(row["role"] for row in rows if row["primary_class"])),
        "support_status": dict(Counter(row["support_status"] for row in rows)),
        "pilot_roles": dict(Counter(row["role"] for row in pilot)),
        "main_roles": dict(Counter(row["role"] for row in main)),
        "pilot_unique_sequences": len({row["sequence_sha256"] for row in pilot}),
        "main_unique_sequences": len({row["sequence_sha256"] for row in main}),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION, "campaign": CAMPAIGN, "policy": POLICY,
        "policy_sha256": digest(POLICY), "inputs": provenance["sources"], "outputs": outputs,
        "r227_queue_digest": provenance["r227_queue_digest"],
        "admitted_classes": provenance["admitted_classes"], "census": counters,
        "selection": selection,
        "selected_records_sha256": hashlib.sha256(frozen_selection).hexdigest(),
        "selection_never_reads": ["target_profile_hit", "any_profile_hit", "reference_identity", "model_score", "predicted_structure"],
        "bounds": [
            "All original samples remain in attempts; structure support is AA20 and 16-1024 residues without cropping.",
            "R227 omitted per-sample EOS; aggregate termination is not a biological-validity criterion.",
            "Natural real anchors have native-class annotation, not newly established folding or activity.",
            "Composition shuffles are constructed references, not experimentally verified negatives.",
            "Old DIAMOND identity lacks alignment coverage; no reported alignment is not evolutionary novelty.",
            "Selection is a prospective structural extension of previously observed generations and classes.",
        ],
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    write_immutable(evidence_dir / "cohort_manifest.json", encoded)
    source_cells = (json.dumps(provenance["source_cells"], sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    write_immutable(output_dir / "source_cells.json", source_cells)
    return manifest
