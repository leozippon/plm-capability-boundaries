#!/usr/bin/env python3
"""CPU-only, retained-record export. No model/oracle imports or new measurements."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import resource
import shutil

R6 = "results/R6"
EVIDENCE = f"{R6}/generation_evidence_20260905"
REPLICATION = f"{R6}/generation_replication_20260927"
GATE = "archive/logs/R6/gate_generative_control/build"
AA20 = set("ACDEFGHIKLMNPQRSTVWY")
# Explicit allowlists: arbitrary launch/environment/command manifests are never copied.
CONFIG_KEYS = set("""campaign id seed campaign_seed sampling_seed arm model cell condition attempts
batch_size max_new_tokens effective_max_new_tokens effective_hf_max_new_tokens max_new_tokens_argument
min_new_tokens_argument temperature top_k top_p repetition_penalty use_cache kv_cache seed_rule
batch_seed_rule per_cell_seed_rule prompt native_task dtype add_special_tokens official_generator_added_terminal_tokens
residue_budget token_safety_cap combination checkpoint checkpoint_facts checkpoint_files fingerprint
files_sha256 model_files_sha256 upstream_generation_files policy sampling architectures context d_model
model_type n_layers rung tokenizer_class tokenizer_vocab_size vocab_size input_vocab_size output_vocab_size
pad_token_id dtype_observed dtype_requested joint_mode rendering symbol_unit code_sha256 module_sha256
runner_sha256 runner_module_sha256 cohort_module_sha256 strict_loader_sha256 torch torch_version
python transformers versions numpy model_id predictor_kind precision min_length max_length never_truncate
num_diffusion_samples num_loops num_sampling_steps sample_selection plddt_conversion schema_version
evaluation_signature gpu_name device_name snapshot digest run_digest manifest_sha256 oracle_sha256
threshold domain_coverage_threshold reservoir_sha256 attempts_sha256 source_sha256 source_ledger
revision commit_hash model_revision tokenizer_revision path protein_mode terminator token end_delimiter
label key requested_class class_key created_utc contract provenance generation_provenance pfam_receipt
oracle hmmer pfam hmmscan version source_sha256 hmmscan_sha256 tarball_sha256 n_profiles database engine
sampling_seed postselection post_selection_filter profiles_available n_requested n_searched n_empty
configuration model_path native_prompt campaigns cells classes attempts_per_class comparison_attempts
comparison_indices historical_selection_sha256 upstream_source_sha256 upstream_tokenizer_sha256
compatibility endpoint_config_sha256 endpoint_loading_facts tokenizer_sha256 training_stage component_order
stage_index generation_policy generation_precision tf32 inventory_sha256 native_manifest_sha256""".split())
HASH_MAPS = {"files_sha256", "checkpoint_files", "model_files_sha256", "upstream_generation_files", "oracle_sha256", "code_sha256", "upstream_source_sha256", "upstream_tokenizer_sha256"}
RAW_FIELDS = """source_key source_label source_sample_index class_key native_prompt_class native_prompt_label
near_duplicate_group exact_duplicate_group comparison_selected selected_for_structure primary_class
inclusion_probability phase stratum support_status structure_exclusion_reason paired_id valid_aa20
stop_status source_stop_reason source_budget_censored source_termination_observed
source_per_sample_termination_note source_max_new_tokens source_cell_n source_cell_n_terminated
 generated_tokens effective_max_new_tokens effective_eos_token_ids decoder_stop native_delimiter_observed
 official_compilation_valid empty_output empty_residue_prefix format_failure format_valid model_eos_observed
native_complete native_terminal_observed stop_reason residue_budget residue_budget_overshoot
biological_completeness residue_prefix_is_whole_generated_sequence leading_canonical_residue_length
reference_identity reference_coverage reference_search_status target_profile_hit profile_hit_classes
combination campaign_seed""".split()
PROFILE_FIELDS = ["pfam_status", "pfam_any_hit", "pfam_max_profile_coverage", "pfam_families", "pfam_complete_domain", "pfam_library_scope", "pfam_source_ref", "pfam_oracle_ref", "retained_any_profile_hit"]
ATTEMPT_FIELDS = ["attempt_key", "campaign", "cell", "arm", "condition", "role", "modality", "id", "id_origin", "sequence_sha256", "length", "length_unit", "empty", "truncated", "truncation_basis", "main_comparison_member", "source_ref", "source_sha256", "producer_ref", "model_revision_status", "annotation_ref", "historical_support_ref", "reclassification_ref", "reclassification_json", "folding_record_count"] + RAW_FIELDS + PROFILE_FIELDS
FOLD_FIELDS = ["folding_key", "attempt_key", "join_status", "id", "attempt_id", "paired_id", "role", "arm", "condition", "campaign", "cell", "sequence_sha256", "length", "source_ref", "source_sha256", "evaluation_signature", "hardware", "worker_refs", "status", "reason", "predictor", "mean_ca_plddt", "fraction_ca_plddt_ge70", "ptm", "mean_pae_angstrom", "predicted_confidence_event", "evaluator_truncated", "diffusion_sample_index", "diffusion_sample_splits", "files_sha256", "selected_for_structure", "inclusion_probability", "phase", "stratum", "near_duplicate_group"]
CONTROL_FIELDS = ["control_key", "attempt_key", "role", "id", "campaign", "cell", "arm", "condition", "class_key", "sequence_sha256", "length", "source_ref", "source_sha256", "fragment_provenance", "profile"]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def safe_config(value, key=""):
    if isinstance(value, dict):
        if key == "stage_index":
            return {k: v for k, v in value.items() if k in {"0", "1"} and v in {"stage1", "stage2"}}
        if key in HASH_MAPS:
            return {k: v for k, v in value.items() if isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)}
        return {k: safe_config(v, k) for k, v in value.items() if k in CONFIG_KEYS}
    if isinstance(value, list):
        return [safe_config(v, key) for v in value]
    return value


def sequence_hash(row, field="sequence"):
    sequence = row[field]
    digest = sha(sequence.encode())
    if row.get("sequence_sha256") not in (None, digest):
        raise ValueError(f"sequence SHA mismatch: {row.get('id')}")
    return digest


def exact_join(left, right, right_id="id", sequence_field="sequence"):
    if str(left["id"]) != str(right[right_id]) or left["sequence_sha256"] != sequence_hash(right, sequence_field):
        raise ValueError(f"ID/sequence mismatch: {left['id']}")


def truncation(row):
    """Budget censoring only, never inferred from residue length or format validity."""
    if row.get("source_budget_censored") is not None:
        return row["source_budget_censored"], "source_budget_censored"
    for key in ("decoder_stop", "stop_status", "source_stop_reason", "stop_reason"):
        value = row.get(key)
        if value in {"max_new_tokens", "budget_censored", "budget_censored_residue_continuation", "residue_budget", "token_safety_cap"}:
            return True, key + ":" + value
        if value in {"eos", "native_terminal"}:
            return False, key + ":" + value
    return None, "unknown; not inferred from length"


def profile(row, ref, retained_only=True):
    generated = row.get("profile", {}).get("generated")
    hit = generated.get("any_family") if generated is not None else row.get("any_profile_hit")
    result = dict(pfam_any_hit=hit, retained_any_profile_hit=hit,
                  pfam_max_profile_coverage=generated.get("best_profile_coverage") if generated else None,
                  pfam_families=generated.get("families") if generated else row.get("pfam_families"),
                  pfam_complete_domain=generated.get("complete_domain") if generated else None,
                  pfam_source_ref=ref, pfam_library_scope="all_Pfam" if generated is not None else "retained_library_scope_not_verified")
    sequence = row["sequence"]
    if not sequence or not set(sequence) <= AA20:
        result.update(pfam_status="not_scanned_empty_or_noncanonical", pfam_any_hit=None, pfam_max_profile_coverage=None)
    elif hit is None:
        result["pfam_status"] = "missing_or_incomplete"
    elif not retained_only or row.get("profile_search_status") == "searched":
        result["pfam_status"] = "scanned_hit" if hit else "scanned_no_hit"
    else:
        result["pfam_status"] = "retained_hit_scan_provenance_unverified" if hit else "retained_no_hit_scan_provenance_unverified"
    return result


def make_attempt(row, campaign, cell, ref, source_sha, modality="protein"):
    digest = sequence_hash(row)
    ident = str(row.get("id", row.get("attempt_index", row.get("source_sample_index"))))
    if ident == "None":
        raise ValueError(f"Missing attempt identity: {ref}")
    truncated, basis = truncation(row)
    out = {k: row.get(k) for k in RAW_FIELDS}
    out.update(attempt_key=compact([campaign, cell, ident]), campaign=campaign, cell=cell,
               arm=row["arm"], condition=row.get("condition", "unconditioned"), role=row.get("role", "generation"),
               modality=modality, id=ident, id_origin="retained" if "id" in row else "attempt_index" if "attempt_index" in row else "source_sample_index",
               sequence_sha256=digest, length=len(row["sequence"]), length_unit="characters" if modality == "text" else "residues",
               empty=not row["sequence"], truncated=truncated, truncation_basis=basis,
               main_comparison_member=bool(row.get("comparison_selected", False)), source_ref=ref,
               source_sha256=source_sha, model_revision_status="not_retained", folding_record_count=0)
    out.update(profile(row, ref))
    if modality == "text":
        out.update(pfam_status="not_applicable_text", pfam_any_hit=None, pfam_library_scope=None)
    return out


def csv_gzip(path, rows, fields):
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
            writer = csv.DictWriter(text, fieldnames=fields, extrasaction="raise", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                converted = {}
                for k in fields:
                    v = row.get(k)
                    converted[k] = "" if v is None else ("true" if v is True else "false" if v is False else compact(v) if isinstance(v, (dict, list)) else v)
                writer.writerow(converted)


class Export:
    def __init__(self, root: Path):
        self.root = root
        self.sources = {}
        self.configs = {}
        self.attempts = {}
        self.by_id = defaultdict(list)
        self.controls = []
        self.folds = []
        self.gaps = []
        self.aliases = []
        self.inventories = {}

    def source(self, path):
        path = Path(path)
        rel = str(path.relative_to(self.root)) if path.is_absolute() else str(path)
        if rel not in self.sources:
            p = self.root / rel
            with p.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            self.sources[rel] = {"sha256": digest, "bytes": p.stat().st_size, "uses": []}
        return rel

    def use(self, path, use):
        rel = self.source(path)
        if use not in self.sources[rel]["uses"]:
            self.sources[rel]["uses"].append(use)
        return rel

    def data(self, path, use="provenance"):
        rel = self.use(path, use)
        return json.loads((self.root / rel).read_text())

    def rows(self, path, use):
        rel = self.use(path, use)
        count = 0
        with (self.root / rel).open() as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    raise ValueError(f"Blank JSONL line {rel}:{n}")
                count += 1
                yield json.loads(line), f"{rel}#L{n}"
        self.sources[rel]["rows"] = count

    def paths(self, pattern, count=None):
        paths = sorted(self.root.glob(pattern))
        if not paths or (count is not None and len(paths) != count):
            raise ValueError(f"Inventory mismatch {pattern}: {len(paths)} != {count}")
        self.inventories[pattern] = [str(p.relative_to(self.root)) for p in paths]
        return paths

    def config(self, path):
        rel = self.source(path)
        if rel not in self.configs:
            self.configs[rel] = safe_config(self.data(path, "whitelisted_configuration"))
        return rel

    def add(self, raw, campaign, cell, ref, producer=None, modality="protein"):
        path = ref.split("#", 1)[0]
        out = make_attempt(raw, campaign, cell, ref, self.sources[path]["sha256"], modality)
        out["producer_ref"] = producer
        if out["attempt_key"] in self.attempts:
            raise ValueError(f"Duplicate attempt key: {out['attempt_key']}")
        self.attempts[out["attempt_key"]] = out
        self.by_id[out["id"]].append(out)
        return out

    def unique(self, ident, arm=None, condition=None, campaign=None):
        candidates = [r for r in self.by_id[str(ident)] if (arm is None or r["arm"] == arm) and (condition is None or r["condition"] == condition) and (campaign is None or r["campaign"] == campaign)]
        if len(candidates) != 1:
            raise ValueError(f"Expected unique attempt {ident}: {len(candidates)}")
        return candidates[0]

    def historical(self):
        for p in self.paths(f"{EVIDENCE}/analysis/esmfold2_20260920012718/*/grouped_annotated_attempts.jsonl", 17):
            for row, ref in self.rows(p, "historical_unconditional_attempts"):
                self.add(row, row["campaign"], row["arm"] + "__" + row["condition"], ref)
        p = self.root / f"{R6}/progen3_generation_20260905/attempts.jsonl"
        producer = self.config(p.with_name("run_manifest.json"))
        for row, ref in self.rows(p, "historical_progen3_attempts"):
            self.add(row, "EXP-R2-233", "progen3-3b__unconditioned", ref, producer)
        p = self.root / f"{EVIDENCE}/attempts.jsonl"
        for row, ref in self.rows(p, "historical_conditioned_attempts_and_reference_controls"):
            if row["role"] not in {"generation", "unconditioned_floor"}:
                self.controls.append(dict(control_key=ref, role=row["role"], id=row["id"], campaign="EXP-R2-232", cell=row["arm"] + "__" + row["condition"], arm=row["arm"], condition=row["condition"], class_key=row.get("class_key"), sequence_sha256=sequence_hash(row), length=len(row["sequence"]), source_ref=ref, source_sha256=self.sources[str(p.relative_to(self.root))]["sha256"]))
                continue
            self.add(row, "EXP-R2-232", row["arm"] + "__" + row["condition"], ref)
        self.config(f"{R6}/conditioned_generation_20260826/instruments.json")
        # The original producer JSONs bind each historical sample index, not a regenerated draw.
        historical = {(r["arm"], r["source_key"], r["source_sample_index"]): r for r in self.attempts.values() if r["campaign"] == "EXP-R2-232"}
        for p in self.paths(f"{R6}/conditioned_generation_20260826/generations_*.json", 6):
            data = self.data(p, "conditioned_producer_samples")
            rel = self.config(p)
            for key, cell in sorted(data["cells"].items()):
                self.configs[rel].setdefault("cells", {})[key] = safe_config(cell)
                pointer = key.replace("~", "~0").replace("/", "~1")
                for i, sequence in enumerate(cell["samples"]):
                    ref = f"{rel}#/cells/{pointer}/samples/{i}"
                    if data["modality"] == "protein":
                        out = historical[(data["arm"], key, i)]
                        if out["sequence_sha256"] != sha(sequence.encode()):
                            raise ValueError(f"Historical producer sequence mismatch: {ref}")
                        out["producer_ref"] = ref
                    else:
                        raw = dict(arm=data["arm"], sequence=sequence, source_sample_index=i, source_key=key, class_key=cell["class_key"], condition=cell["condition"], native_prompt_class=cell.get("requested_class"))
                        self.add(raw, data["campaign"] + "-text-controls", data["arm"] + "__" + key, ref, rel, "text")
        for p in self.paths(f"{R6}/progen3_generation_20260905/analysis/*/annotated_attempts.jsonl", 1):
            for row, ref in self.rows(p, "historical_progen3_annotations"):
                out = self.unique(row["id"], row["arm"], row["condition"], "EXP-R2-233")
                exact_join(out, row)
                out.update(profile(row, ref), annotation_ref=ref)
        self.gaps.append("Historical unconditional grouped ledgers retain stop_status but not decoder token traces/checkpoint revisions; corresponding raw ledgers were not located in the bounded loose-file inventory. Original grouped rows are retained without inferring EOS from length.")

    def replication(self):
        self.config(f"configs/generation_replication_manifest.json")
        for p in self.paths(f"{REPLICATION}/cells/*/attempts.jsonl", 40):
            producer = self.config(p.with_name("run_manifest.json"))
            summary = self.data(p.with_name("generation_replication.json"), "attempt_terminal_receipt")
            if summary["attempts_sha256"] != self.sources[self.source(p)]["sha256"]:
                raise ValueError(f"Replication receipt SHA mismatch: {p}")
            local = {}
            for row, ref in self.rows(p, "replication_attempts"):
                out = self.add(row, row["campaign"], summary["cell"], ref, producer)
                if row["condition"] == "requested":
                    group = next(g for g in self.configs[producer]["cell"]["classes"] if g["class_key"] == row["class_key"])
                    out.update(native_prompt_class=group["class_key"], native_prompt_label=group["label"])
                local[out["id"]] = out
            if len(local) != summary["attempts"]:
                raise ValueError(f"Replication attempt count mismatch: {p}")
            profile_dir = self.root / REPLICATION / "profiles" / p.parent.name
            oracle = self.config(profile_dir / "oracle_configuration.json")
            if self.configs[oracle]["attempts_sha256"] != summary["attempts_sha256"]:
                raise ValueError(f"Oracle attempt SHA mismatch: {p}")
            receipt = self.data(profile_dir / "generation_profiles.json", "profile_annotation_terminal_receipt")
            if receipt["annotated_attempts_sha256"] != self.sources[self.source(profile_dir / "annotated_attempts.jsonl")]["sha256"]:
                raise ValueError("Annotated attempt SHA mismatch")
            shard_manifest = self.data(profile_dir / "build_manifest.json", "profile_shard_inventory")
            for shard in shard_manifest["shards"]:
                stem = Path(shard["path"]).stem
                terminal = self.data(profile_dir / "oracle" / (stem + ".tbl.done"), "terminal_Pfam_receipt")
                if terminal.get("fasta_sha256") != shard["sha256"]:
                    raise ValueError("Incomplete replication oracle shard")
                for suffix in ("tbl", "domtbl"):
                    table = profile_dir / "oracle" / (stem + "." + suffix)
                    self.use(table, "verified_retained_profile_table")
                    if terminal.get(suffix + "_sha256") != self.sources[self.source(table)]["sha256"]:
                        raise ValueError("Replication oracle table SHA mismatch")
            seen = set()
            fragment_profiles = {}
            for row, ref in self.rows(profile_dir / "annotated_attempts.jsonl", "replication_profile_annotations"):
                out = local[row["id"]]
                exact_join(out, row)
                if row["id"] in seen:
                    raise ValueError("Duplicate annotation")
                seen.add(row["id"])
                out.update(profile(row, ref, retained_only=False), annotation_ref=ref, pfam_oracle_ref=oracle)
                fragment_profiles[row["id"]] = row["profile"]["fragment"]
            if seen != local.keys():
                raise ValueError("Missing replication annotation")
            seen = set()
            for row, ref in self.rows(profile_dir / "paired_fragments.jsonl", "paired_real_fragment_controls"):
                out = local[row["attempt_id"]]
                exact_join(out, row, "attempt_id", "generated")
                if row["attempt_id"] in seen:
                    raise ValueError("Duplicate paired fragment")
                seen.add(row["attempt_id"])
                self.controls.append(dict(control_key=ref, attempt_key=out["attempt_key"], role="paired_real_fragment", id=row["attempt_id"], campaign=out["campaign"], cell=out["cell"], arm=out["arm"], condition=out["condition"], class_key=out["class_key"], sequence_sha256=sha(row["fragment"].encode()), length=len(row["fragment"]), source_ref=ref, source_sha256=self.sources[ref.split("#")[0]]["sha256"], fragment_provenance=row["fragment_provenance"], profile=fragment_profiles[row["attempt_id"]]))
            if seen != local.keys():
                raise ValueError("Missing paired fragment")

    def support(self):
        manifest = self.data(f"{GATE}/build_manifest.json", "original_support_accounting")
        # The whitelist is structural here: these are frozen selection facts, not launch manifests.
        self.configs[f"{GATE}/build_manifest.json"] = {"cells": manifest["cells"], "seed": manifest["seed"]}
        for cell in manifest["cells"]:
            old_ledger = cell["ledger"]
            ledger = old_ledger.replace("results/transfer/generation_evidence/", EVIDENCE + "/").replace("results/transfer/progen3_generation_evidence/", f"{R6}/progen3_generation_20260905/")
            self.use(ledger, "original_support_ledger_hash_binding")
            if self.sources[ledger]["sha256"] != cell["ledger_sha256"]:
                raise ValueError(f"Original support ledger SHA mismatch: {ledger}")
            p = f"{GATE}/cells/{cell['cell']}.jsonl"
            if self.sources[self.source(p)]["sha256"] != cell["cell_sha256"]:
                raise ValueError(f"Support cell SHA mismatch: {p}")
            for row, ref in self.rows(p, "original_comparison_membership"):
                out = self.unique(row["attempt_id"], row["arm"], row["condition"], cell["campaign"])
                if out["sequence_sha256"] != sha(row["sequences"]["generated"].encode()):
                    raise ValueError("Historical support sequence mismatch")
                out.update(main_comparison_member=True, historical_support_ref=ref)
        self.gate_oracle(manifest)

    def gate_oracle(self, manifest):
        names = self.data(f"{GATE}/query_names.json", "exact_support_query_identifiers")
        reverse = {v: k for k, v in names.items()}
        hits = defaultdict(set)
        domains = defaultdict(set)
        coverage = {}
        hit_refs = defaultdict(list)
        for shard in manifest["shards"]:
            stem = Path(shard["path"]).stem
            fasta = f"{GATE}/shards/{stem}.fasta"
            self.use(fasta, "exact_sequence_oracle_query_binding")
            if self.sources[fasta]["sha256"] != shard["sha256"]:
                raise ValueError("Oracle query FASTA SHA mismatch")
            query = None
            residues = []
            def validate_query():
                if query is not None and "|generated|" in reverse[query]:
                    cell, _, ident = reverse[query].split("|", 2)
                    arm, condition = cell.split("__", 1)
                    attempt = self.unique(ident, arm, condition)
                    if sha("".join(residues).encode()) != attempt["sequence_sha256"]:
                        raise ValueError("Oracle query sequence mismatch")
            for line in (self.root / fasta).read_text().splitlines():
                if line.startswith(">"):
                    validate_query()
                    query, residues = line[1:].split()[0], []
                else:
                    residues.append(line.strip())
            validate_query()
            tbl = f"{GATE}/oracle/{stem}.tbl"
            dom = f"{GATE}/oracle/{stem}.domtbl"
            receipt = self.data(tbl + ".done", "terminal_Pfam_receipt")
            if receipt.get("fasta_sha256") != shard["sha256"] or "--cut_ga" not in receipt.get("command", []):
                raise ValueError(f"Invalid oracle receipt: {tbl}")
            for path, is_domain in [(tbl, False), (dom, True)]:
                self.use(path, "retained_Pfam_domain_coverage" if is_domain else "retained_Pfam_sequence_hits")
                with (self.root / path).open() as f:
                    for n, line in enumerate(f, 1):
                        if not line.strip() or line.startswith("#"):
                            continue
                        fields = line.split()
                        query = fields[3] if is_domain else fields[2]
                        if query not in reverse:
                            raise ValueError(f"Unknown Pfam query: {query}")
                        family = fields[1].split(".")[0]
                        if is_domain:
                            domains[query].add(family)
                            cov = (int(fields[16]) - int(fields[15]) + 1) / int(fields[2])
                            if not 0 <= cov <= 1:
                                raise ValueError("Invalid domain profile coverage")
                            coverage[query] = max(coverage.get(query, 0), cov)
                        else:
                            hits[query].add(family)
                        if "|generated|" in reverse[query]:
                            hit_refs[query].append(f"{path}#L{n}")
        for query, families in domains.items():
            if not families <= hits[query]:
                raise ValueError("Domain family absent from sequence table")
        for name, query in names.items():
            cell, role, ident = name.split("|", 2)
            if role != "generated":
                continue
            arm, condition = cell.split("__", 1)
            out = self.unique(ident, arm, condition)
            if not out.get("historical_support_ref"):
                raise ValueError("Oracle query has no exact-sequence support record")
            families = sorted(hits[query])
            out.update(pfam_any_hit=bool(families), pfam_families=families,
                       pfam_max_profile_coverage=coverage.get(query), pfam_status="scanned_hit" if families else "scanned_no_hit",
                       pfam_library_scope="all_Pfam", pfam_source_ref=hit_refs[query] or [f"{GATE}/query_names.json#/{name}"],
                       pfam_oracle_ref=f"{GATE}/build_manifest.json")

    def auxiliary(self):
        for p in self.paths(f"{R6}/generation_followup_20260923/generation/*/*/attempts.jsonl", 8):
            config = self.config(p.with_name("generation_failure_summary.json"))
            if p.parent.name.startswith("genhistory_"):
                arm = p.parent.name.removeprefix("genhistory_")
                for row, ref in self.rows(p, "historical_reclassification_not_new_attempt"):
                    out = self.unique(row["id"], arm)
                    exact_join(out, row)
                    out.update(reclassification_ref=ref, reclassification_json={k: row[k] for k in RAW_FIELDS if k in row})
                continue
            for row, ref in self.rows(p, "followup_new_attempts"):
                self.add(row, "followup:" + p.parent.parent.name, p.parent.name, ref, config)
        for p in self.paths(f"{R6}/component_swap_20260923/components/*/*/*/attempts.jsonl", 4):
            config = self.config(p.with_name("summary.json"))
            for intervention in sorted(p.parents[1].glob("*manifest.json")):
                self.config(intervention)
            for row, ref in self.rows(p, "component_swap_new_attempts"):
                self.add(row, "component-swap:" + p.parents[2].name, p.parents[1].name + "/" + p.parent.name, ref, config)
        first = None
        for p in [self.root / f"{R6}/instructprotein_family_probe_20260917/first_run/generations.json", self.root / f"{R6}/instructprotein_family_probe_20260917/generations.json"]:
            data = self.data(p, "family_probe_samples")
            rel = self.config(p)
            self.config(p.with_name("report.json"))
            digest = self.sources[rel]["sha256"]
            if first == digest:
                self.aliases.append({"path": rel, "reason": "byte-identical probe alias; not additional attempts"})
                continue
            first = digest
            campaign = "instructprotein-probe:" + ("first_run" if p.parent.name == "first_run" else "current_run")
            for key, cell in sorted(data["cells"].items()):
                self.configs[rel].setdefault("cells", {})[key] = safe_config(cell)
                for i, sequence in enumerate(cell["samples"]):
                    row = dict(sequence=sequence, arm=data["arm"], condition=key, class_key=data["cells"]["requested"]["key"], native_prompt_class=cell["key"], source_key=key, source_sample_index=i)
                    if "raw" in cell and data.get("terminator", {}).get("token"):
                        row["native_delimiter_observed"] = data["terminator"]["token"] in cell["raw"][i]
                    self.add(row, campaign, data["arm"] + "__" + key, f"{rel}#/cells/{key}/samples/{i}", rel)

    def folding(self):
        paths = self.paths(f"{EVIDENCE}/structure/esmfold2_*/*/predictions.jsonl", 20)
        paths += self.paths(f"{R6}/manuscript_structures_20261004_l20/predictions/index-000-of-001.jsonl", 1)
        controls = {(c.get("attempt_key"), c["sequence_sha256"]) for c in self.controls if c["role"] == "paired_real_fragment"}
        for p in paths:
            workers = defaultdict(list)
            hardware = defaultdict(set)
            worker_paths = sorted(p.parent.glob("worker-*.json"))
            if p.parent.name.startswith("s48_"):
                arm = p.parent.name.removeprefix("s48_")
                worker_paths = sorted((p.parent.parent / "shards").glob(f"s48_esmfold2_{arm}_shard*/worker-*.json"))
            for worker in worker_paths:
                config = self.data(worker, "folding_worker_receipt")
                rel = self.config(worker)
                workers[config["evaluation_signature"]].append(rel)
                hardware[config["evaluation_signature"]].add(config["gpu_name"])
            for row, ref in self.rows(p, "existing_folding_record"):
                structure = row["structure"]
                digest = sequence_hash(row)
                if structure["sequence_sha256"] != digest:
                    raise ValueError("Folding structure sequence mismatch")
                out = {k: row.get(k) for k in FOLD_FIELDS}
                out.update({k: structure.get(k) for k in FOLD_FIELDS if k in structure})
                ident = row.get("attempt_id", row["id"])
                out.update(folding_key=ref, source_ref=ref, source_sha256=self.sources[ref.split("#")[0]]["sha256"], attempt_id=ident, sequence_sha256=digest)
                sig = structure["evaluation_signature"]
                out.update(worker_refs=workers[sig], hardware=sorted(hardware[sig]))
                if not workers[sig]:
                    raise ValueError(f"No matching folding worker signature: {ref}")
                join_folding(out, self.by_id, controls)
                if out.get("attempt_key") and out["role"] == "generation":
                    self.attempts[out["attempt_key"]]["folding_record_count"] += 1
                self.folds.append(out)

    def finish(self, out):
        attempts = sorted(self.attempts.values(), key=lambda r: r["attempt_key"])
        canonical = [r for r in attempts if r["campaign"] in {"EXP-R2-246", "EXP-R2-233", "EXP-R2-232", "replicate_1", "replicate_2"}]
        counts = {"attempts": len(attempts), "canonical_attempts": len(canonical), "main_comparison_members": sum(r["main_comparison_member"] for r in attempts), "controls": len(self.controls), "folding_records": len(self.folds), "folding_status": dict(Counter(r["status"] for r in self.folds)), "folding_role": dict(Counter(r["role"] for r in self.folds)), "folding_join_status": dict(Counter(r["join_status"] for r in self.folds)), "pfam_status": dict(Counter(r["pfam_status"] for r in attempts)), "campaign_attempts": dict(Counter(r["campaign"] for r in attempts))}
        counts["historical_reclassifications"] = sum(bool(r.get("reclassification_ref")) for r in attempts)
        counts["folding_signatures"] = dict(Counter(r["evaluation_signature"] for r in self.folds))
        expected = {"canonical_attempts": 69200, "main_comparison_members": 48000, "attempts": 80512, "folding_records": 6796, "historical_reclassifications": 3200}
        for key, value in expected.items():
            if counts[key] != value:
                raise ValueError(f"Census mismatch: {key}={counts[key]}, expected {value}")
        csv_gzip(out / "attempts.csv.gz", attempts, ATTEMPT_FIELDS)
        csv_gzip(out / "folding.csv.gz", self.folds, FOLD_FIELDS)
        csv_gzip(out / "controls.csv.gz", self.controls, CONTROL_FIELDS)
        groups = defaultdict(Counter)
        for r in attempts:
            g = groups[(r["campaign"], r["cell"], r["arm"], r["condition"], r["class_key"], r["source_key"])]
            g.update(attempts=1, empty=int(r["empty"]), main_comparison_members=int(r["main_comparison_member"]), truncation_unknown=int(r["truncated"] is None), truncated=int(r["truncated"] is True), folding_records=r["folding_record_count"], pfam_coverage_present=int(r.get("pfam_max_profile_coverage") is not None))
        group_rows = [dict(zip(["campaign", "cell", "arm", "condition", "class_key", "source_key"], key), **value) for key, value in sorted(groups.items(), key=lambda item: compact(item[0]))]
        csv_gzip(out / "groups.csv.gz", group_rows, ["campaign", "cell", "arm", "condition", "class_key", "source_key", "attempts", "empty", "main_comparison_members", "truncation_unknown", "truncated", "folding_records", "pfam_coverage_present"])
        provenance = {"schema": "generation_details_v1", "exporter_sha256": sha(Path(__file__).read_bytes()), "sources": dict(sorted(self.sources.items())), "producer_and_evaluator_configurations": dict(sorted(self.configs.items())), "inventories": self.inventories, "aliases": self.aliases}
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        report = {"counts": counts, "gaps": self.gaps + ["Global completeness across arbitrary older experiments is not asserted; inventories list the exact bounded campaign files read.", "Missing checkpoint revisions remain explicit; producer file/configuration hashes, where retained, are in provenance.json.", "Retained-only Pfam booleans outside verified scan support do not establish missing profile coverage. No union-domain coverage, new thresholds, selection, bootstrap, scan, inference or folding was performed.", "Retired ESMFold-v1 records are excluded: superseded evaluator, incompatible calibration. Every retained ESMFold2 record in the enumerated 21 files is exported, including failures and controls.", "Text controls use character length, not protein residue length; their Pfam fields are not applicable.", "Probe reports retain aggregate Pfam counts, not per-attempt hit assignments; these are not imputed to individual records. Raw probe delimiters are observable, but decoder EOS/token traces are not reconstructed.", "Historical conditional per-sample termination is unavailable; truncation stays null, regardless of length.", "Historical reclassification rows enrich their exact original ID/sequence rather than contributing another attempt. Probe first/current runs remain distinct unless byte-identical."], "null_encoding": "blank CSV field", "boolean_encoding": "true/false", "profile_coverage_definition": "maximum over SINGLE retained domain instances of (hmm_to-hmm_from+1)/target_length; never union", "plddt_scale": "mean CA pLDDT 0–100 from structure summaries, never PDB 0–1 B-factor tokens"}
        (out / "coverage-and-gaps.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (out / "README.md").write_text(README)
        manifest = {"schema": "generation_details_export_v1", "counts": counts, "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())} for p in sorted(out.iterdir()) if p.is_file()}}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return counts


def join_folding(row, by_id, controls):
    candidates = [r for r in by_id.get(str(row["attempt_id"]), []) if r["arm"] == row["arm"] and r["condition"] == row["condition"] and (not row.get("campaign") or r["campaign"] == row["campaign"])]
    if row["role"] == "generation":
        if not candidates:
            row["join_status"] = "unmatched_generation_retained"
            row["attempt_key"] = None
            return
        if len(candidates) != 1 or candidates[0]["sequence_sha256"] != row["sequence_sha256"]:
            raise ValueError("Folding ID/sequence mismatch")
        row.update(attempt_key=candidates[0]["attempt_key"], join_status="exact_generation")
    elif row["role"] in {"paired_real_fragment", "real_fragment", "fragment"}:
        if len(candidates) != 1 or (candidates[0]["attempt_key"], row["sequence_sha256"]) not in controls:
            raise ValueError("Folding control role/sequence mismatch")
        row.update(attempt_key=candidates[0]["attempt_key"], join_status="exact_paired_control_not_generation")
    else:
        if candidates and any(r["sequence_sha256"] == row["sequence_sha256"] for r in candidates):
            raise ValueError("Bad folding role join: control claims generation identity")
        row.update(attempt_key=None, join_status="control_not_generation")


README = """# Retained generation details

This is a CPU-only export of the explicitly enumerated retained campaigns, not a new experiment or a claim of global repository completeness. It keeps every attempt in those ledgers, including empty, format-invalid, token-censored and non-comparison attempts. No ranking, new support selection, bootstrap, Pfam scan or folding was performed.

- `attempts.csv.gz`: one row per campaign/cell/original-ID attempt (80,512 total: 69,200 canonical, 10,000 text controls, 256 follow-up, 256 component swaps, 800 probe attempts). `main_comparison_member` reproduces the original 48,000 comparison members, never a first-800 or favorable-model selection.
- `folding.csv.gz`: all 6,796 existing ESMFold2 prediction records, including failures, shuffles and natural/paired-fragment controls. Join generated metrics using `attempt_key` **and role=generation**. Keep evaluation signatures and hardware separate; do not choose a favorable evaluator or pool controls into generated metrics. CA pLDDT is on the 0–100 summary scale. L20 sample splits remain separate from the historical H200 configurations.
- `controls.csv.gz`: original natural references and replication paired real fragments; these are not additional generation attempts.
- `groups.csv.gz`: census counts by the original campaign, cell, checkpoint arm, condition, class and source grouping.
- `provenance.json`: source file SHA256/size/usage inventory, original selection accounting and whitelisted producer/evaluator configurations. Source references are relative file paths plus 1-based `#L` JSONL lines or RFC6901 JSON pointers. Generated sequences and raw traces are not redistributed; exact sequence SHA256 binds every join. Configuration manifests are not copied wholesale.
- `coverage-and-gaps.json`: exact scope, counts and limitations; `manifest.json` binds all export files.

For replication requested cells, native prompt class/label comes from the bound producer manifest (the producer uses that class label for its requested prompt); original `class_key` is retained separately. Historical requested/mismatched prompt classes come directly from retained rows.

Blank CSV fields are null/unknown, not false. Boolean values are explicitly `true`/`false`; nested lists/maps are JSON. Gzip headers use mtime=0 and no embedded filename. Attempt keys are unambiguous JSON triples `[campaign,cell,id]`. Where producers supplied only a sample index, `id_origin` states that fact.

`truncated` means retained evidence of budget censoring, not biological incompleteness. Unknown termination is never inferred from residue length; decoder stop, native delimiters, official compilation validity and format flags remain independent. Follow-up historical reclassifications are linked separately and never counted twice.

`pfam_any_hit` is distinct from requested-target hit and actual prompt class. Maximum profile coverage is the maximum **single domain instance** `(hmm_to-hmm_from+1)/target_length`, not union coverage. A sequence-level GA1 hit can legitimately have null domain coverage when no domain clears GA2. Empty/noncanonical records are not negative scans; original denominator-convention booleans remain in `retained_any_profile_hit`. Historical support scans require terminal receipts and domain-family containment. Outside these and replication annotations, retained booleans have an explicit unverified scan-provenance status and coverage stays unknown. No new thresholds are introduced.

Run from the original repository with its retained inputs:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/lzp/miniconda3/envs/ct/bin/python scripts/capability/reporting/export_generation_details.py --root . --out /path/to/new-or-empty-directory
```

`--help` reads no data. The destination must be new or empty; existing exports are never overwritten. The exporter uses only the Python standard library, checks CPU memory/disk availability, logs that preflight to stdout, and refuses broken hashes, identity/sequence joins or incomplete inventories. Retired ESMFold-v1 is excluded because it is a superseded, differently calibrated evaluator. Missing upstream revisions and unavailable original traces are not reconstructed.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Original repository containing retained sources")
    parser.add_argument("--out", required=True, type=Path, help="New or empty output directory")
    args = parser.parse_args(argv)
    root, out = args.root.resolve(), args.out.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error("--out must be a NEW or EMPTY directory")
    disk = shutil.disk_usage(out.parent if out.parent.exists() else root)
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    preflight = {"cpu_only": True, "memory_available_bytes": available, "disk_free_bytes": disk.free, "minimum_memory_bytes": 2 * 1024**3, "minimum_disk_bytes": 1024**3}
    print(compact({"resource_preflight": preflight}), flush=True)
    if available < preflight["minimum_memory_bytes"] or disk.free < preflight["minimum_disk_bytes"]:
        raise RuntimeError("Insufficient CPU memory or output disk space")
    out.mkdir(parents=True, exist_ok=True)
    export = Export(root)
    for stage in (export.historical, export.replication, export.support, export.auxiliary, export.folding):
        print(compact({"stage": stage.__name__}), flush=True)
        stage()
    counts = export.finish(out)
    print(compact({"counts": counts, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)


if __name__ == "__main__":
    main()
