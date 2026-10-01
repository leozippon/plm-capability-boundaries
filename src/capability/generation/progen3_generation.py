"""One native unconditional ProGen3-3B generation point, EXP-R2-233.

Uses the released generator behind the repository's validated strict loader.
Every raw continuation survives, including the official compiler's refusals.
No label, protein prefix, reranking, or filtering is supplied to generation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import MethodType
from typing import Any, Mapping

import torch

from . import generation_evidence as ge
from ..core.io import sha256_file
from ..models import progen3 as _progen3
from ..models.progen3 import PROGEN3_SOURCE, load_progen3, self_check

CAMPAIGN = "EXP-R2-233"
POLICY = {
    "campaign": CAMPAIGN,
    "model": "progen3-3b",
    "native_task": "unconditional_N_to_C_CLM",
    "prompt": "1",
    "attempts": 800,
    "batch_size": 8,
    "seed": 20260905,
    "min_new_tokens_argument": 0,
    "max_new_tokens_argument": 400,
    "official_generator_added_terminal_tokens": 2,
    "effective_hf_max_new_tokens": 402,
    "temperature": 0.85,
    "top_p": 0.95,
    "top_k": 50,
    "top_k_note": "made_explicit_from_released_generator_GenerationConfig_default",
    "repetition_penalty": 1.0,
    "use_cache": True,
    "postselection": None,
    "structure_samples": 128,
    "length_strata": ge.POLICY["length_strata"],
    "structure_support": [16, 1024],
    "batch_seed_rule": "seed_plus_batch_index",
}


def install_generation_compatibility(model: Any) -> str:
    """Restore the removed HF cache-output accessor used by released ProGen3."""
    if hasattr(model, "_extract_past_from_model_output"):
        return "upstream_accessor_present"

    def extract_cache(self: Any, outputs: Any) -> tuple[str, Any]:
        cache = outputs.past_key_values
        if cache is None:
            raise RuntimeError("ProGen3 generation did not return its required KV cache")
        return "past_key_values", cache

    model._extract_past_from_model_output = MethodType(extract_cache, model)
    return "explicit_past_key_values_accessor_for_transformers_4_57"


def make_generator(pg: Any, policy: Mapping[str, Any] = POLICY) -> Any:
    from progen3.generator import ProGen3Generator  # noqa: PLC0415

    install_generation_compatibility(pg.model)
    # Native unconditional encoding is <bos>1; outer batches never exceed eight.
    generator = ProGen3Generator(
        pg.model,
        max_batch_tokens=policy["batch_size"] * (2 + policy["effective_hf_max_new_tokens"]),
        temperature=policy["temperature"], top_p=policy["top_p"],
    )
    generator.default_gen_config.top_k = policy["top_k"]
    generator.default_gen_config.repetition_penalty = policy["repetition_penalty"]
    generator.default_gen_config.bos_token_id = pg.config.bos_token_id
    generator.default_gen_config.output_logits = False
    return generator


@torch.no_grad()
def generation_self_check(pg: Any) -> dict[str, Any]:
    """Existing loader gate, cached-vs-full logits, and two short native samples."""
    scoring = self_check(pg)
    compatibility = install_generation_compatibility(pg.model)
    batch = pg.batch(["MALWMRLLPLLALLALWGP"])
    inputs = {key: batch[key][:, :10] for key in ("input_ids", "position_ids", "sequence_ids")}
    full = pg.model(**inputs, use_cache=False, return_dict=True).logits[:, -1, :].float()
    prefix = {key: value[:, :-1] for key, value in inputs.items()}
    cached = pg.model(**prefix, use_cache=True, return_dict=True)
    final = {key: value[:, -1:] for key, value in inputs.items()}
    incremental = pg.model(**final, past_key_values=cached.past_key_values, use_cache=True, return_dict=True).logits[:, -1, :].float()
    difference = float((full - incremental).abs().max().item())
    if not torch.isfinite(incremental).all() or difference > 0.125:
        raise RuntimeError(f"cached and full-prefix ProGen3 logits disagree: max abs {difference}")
    generator = make_generator(pg)
    torch.manual_seed(POLICY["seed"] + 1_000_000)
    samples = list(generator.generate("1", 2, 0, 24))
    if len(samples) != 2 or any(not re.match(r"[ACDEFGHIKLMNPQRSTVWY]+", item.generation) for item in samples):
        raise RuntimeError("native generation failed its two-sample residue-prefix interface check")
    return {
        "loader": scoring, "cache_max_absolute_logit_difference": difference,
        "cache_tolerance": 0.125, "compatibility": compatibility,
        "smoke_max_new_tokens_argument": 24,
        "smoke_samples": [item._asdict() for item in samples],
        "is_scientific_measurement": False, "passed": True,
    }


def generation_row(raw: str, compiled: str | None, index: int) -> dict[str, Any]:
    """Keep native compilation and the strict canonical prefix as separate facts."""
    prefix = re.match(r"[ACDEFGHIKLMNPQRSTVWY]*", raw).group(0)
    native_complete = compiled is not None
    if native_complete and compiled != prefix:
        raise ValueError("unconditional native compilation differs from the emitted residue run")
    sequence = compiled if compiled is not None else prefix
    if native_complete:
        reason = "native_terminal_eos"
    elif raw == prefix:
        reason = "budget_censored_residue_continuation"
    else:
        reason = "native_format_not_compilable"
    row = ge._row(sequence, arm=POLICY["model"], class_key=None, condition="unconditioned",
                  role="generation", source_label="native_progen3_raw_attempts",
                  source_key=CAMPAIGN, source_sample_index=index, primary_class=False)
    row.update({
        "campaign": CAMPAIGN, "raw_continuation": raw,
        "raw_continuation_character_length": len(raw),
        "leading_canonical_residue_length": len(prefix),
        "official_compiled_sequence": compiled, "official_compilation_valid": native_complete,
        "source_termination_observed": "<eos>" in raw, "source_stop_reason": reason,
        "source_budget_censored": reason == "budget_censored_residue_continuation",
        "source_max_new_tokens": POLICY["effective_hf_max_new_tokens"],
        "residue_prefix_is_whole_generated_sequence": native_complete,
        "censored_fragment_included_without_claim_of_biological_completeness": not native_complete,
        "native_prompt": "1", "native_prompt_class": None,
        "native_prompt_label": None, "pfam_families": None,
        "target_profile_hit": None, "any_profile_hit": None, "profile_hit_classes": None,
    })
    return row


def select_structure(rows: list[dict]) -> tuple[list[dict], dict]:
    strata: dict[str, list[dict]] = {}
    for row in rows:
        if row["support_status"] == "eligible":
            strata.setdefault(row["stratum"], []).append(row)
        else:
            row["structure_exclusion_reason"] = row["support_status"]
    allocation = ge.allocate_strata({key: len(value) for key, value in strata.items()}, POLICY["structure_samples"])
    selected: list[dict] = []
    blocks = {}
    for key, n in allocation.items():
        population = strata[key]
        chosen = ge._uniform(population, n, seed=POLICY["seed"], key=f"{CAMPAIGN}|structure|{key}")
        for row in chosen:
            row.update({"phase": "main", "inclusion_probability": n / len(population), "selected_for_structure": True})
            selected.append(dict(row))
        blocks[key] = {"N": len(population), "n": n, "inclusion_probability": n / len(population)}
    for row in rows:
        if not row["selected_for_structure"] and row["structure_exclusion_reason"] is None:
            row["structure_exclusion_reason"] = "eligible_not_sampled"
    selected.extend(ge.composition_shuffle(row, seed=POLICY["seed"]) for row in list(selected))
    return sorted(selected, key=lambda row: row["id"]), blocks


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def checkpoint_fingerprint(checkpoint: Path, source: Path) -> dict[str, Any]:
    files = [checkpoint / "config.json", *sorted(checkpoint.glob("*.safetensors"))]
    if len(files) < 2 or not files[0].is_file():
        raise FileNotFoundError(f"checkpoint weights/config incomplete: {checkpoint}")
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        files.append(index_path)
    return {
        "checkpoint_files": {path.name: sha256_file(path) for path in files},
        "upstream_generation_files": {
            name: sha256_file(source / "progen3" / name)
            for name in ("generator.py", "modeling.py", "batch_preparer.py", "tokenizer.json")
        },
        "runner_module_sha256": sha256_file(Path(__file__)),
        "cohort_module_sha256": sha256_file(Path(ge.__file__)),
        "strict_loader_sha256": sha256_file(Path(_progen3.__file__)),
    }


def run(checkpoint: Path, output_dir: Path, *, source: Path = PROGEN3_SOURCE,
        device: str = "cuda:0", interface_only: bool = False) -> dict[str, Any]:
    fingerprint = checkpoint_fingerprint(checkpoint, source)
    pg = load_progen3(checkpoint, source=source, device=device, dtype=torch.bfloat16)
    if not interface_only and (pg.config.num_hidden_layers, pg.config.hidden_size) != (24, 1280):
        raise ValueError("EXP-R2-233 scientific generation requires the declared ProGen3-3B")
    check = generation_self_check(pg)
    if interface_only:
        result = {"campaign": CAMPAIGN, "fingerprint": fingerprint, "interface_only": True, "self_check": check}
        ge.write_immutable(output_dir / "native_generation_interface.json", _json_bytes(result))
        return result
    run_record = {"campaign": CAMPAIGN, "policy": POLICY, "fingerprint": fingerprint,
                  "dtype": "bfloat16", "self_check": check,
                  "torch_version": torch.__version__}
    run_id = ge.digest({key: value for key, value in run_record.items() if key != "self_check"})
    run_record["run_digest"] = run_id
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("run_digest") != run_id:
            raise ValueError("resume native-generation configuration or fingerprint changed")
        # The live self-check has passed again. Preserve the initial receipt;
        # its final floating-point digits need not match across valid GPUs.
    else:
        ge.write_immutable(manifest_path, _json_bytes(run_record))
    generator = make_generator(pg)
    all_rows: list[dict] = []
    for start in range(0, POLICY["attempts"], POLICY["batch_size"]):
        end = min(start + POLICY["batch_size"], POLICY["attempts"])
        path = output_dir / "batches" / f"batch_{start:05d}_{end:05d}.json"
        if path.is_file():
            block = json.loads(path.read_text())
            if block.get("run_digest") != run_id or block.get("rows_digest") != ge.digest(block.get("rows")):
                raise ValueError(f"resume batch provenance/digest mismatch: {path}")
            batch_rows = block["rows"]
            if [row["source_sample_index"] for row in batch_rows] != list(range(start, end)):
                raise ValueError(f"resume batch attempt indices mismatch: {path}")
        else:
            torch.manual_seed(POLICY["seed"] + start // POLICY["batch_size"])
            samples = list(generator.generate(POLICY["prompt"], end - start,
                                             POLICY["min_new_tokens_argument"], POLICY["max_new_tokens_argument"]))
            if len(samples) != end - start:
                raise RuntimeError("native generator returned fewer attempts than requested")
            batch_rows = [generation_row(item.generation, item.sequence, start + offset) for offset, item in enumerate(samples)]
            block = {"run_digest": run_id, "start": start, "end": end,
                     "rows": batch_rows, "rows_digest": ge.digest(batch_rows)}
            ge.write_immutable(path, _json_bytes(block))
        all_rows.extend(batch_rows)
        print(f"{CAMPAIGN} attempts persisted: {end}/{POLICY['attempts']}", flush=True)
    groups, grouping = ge.cg.near_duplicate_group_ids([row["sequence"] for row in all_rows], unit="residues")
    for index, row in enumerate(all_rows):
        row["near_duplicate_group"] = f"{CAMPAIGN}|g{int(groups[index])}"
    subset, selection = select_structure(all_rows)
    ge.write_immutable(output_dir / "attempts.jsonl", ge.jsonl_bytes(all_rows))
    ge.write_immutable(output_dir / "main_subset.jsonl", ge.jsonl_bytes(subset))
    summary = {"campaign": CAMPAIGN, "run_digest": run_id, "attempts": len(all_rows),
               "official_compilation_valid": sum(row["official_compilation_valid"] for row in all_rows),
               "support_eligible": sum(row["support_status"] == "eligible" for row in all_rows),
               "structure_records_including_shuffles": len(subset), "structure_selection": selection,
               "near_duplicate_grouping": grouping,
               "attempts_sha256": sha256_file(output_dir / "attempts.jsonl"),
               "main_subset_sha256": sha256_file(output_dir / "main_subset.jsonl")}
    ge.write_immutable(output_dir / "generation_summary.json", _json_bytes(summary))
    return summary
