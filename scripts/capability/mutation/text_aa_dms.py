#!/usr/bin/env python3
"""Independent ProteinGym controller for the 13 text amino-acid string controls.

Phases: ``census`` (tokenizer-only CPU), ``probe``, ``score``, and ``analyse``.
``score`` runs the five labelled probe cases plus the longest-legal-input ×3
resource gate in-process, writes a probe artefact, then scores. A failed probe
refuses a scientific score. ``analyse`` is CPU.

Merged BPE is scored as a token-sum under a user-ordered amendment of
``text-aa-fp32-v1``. That sum is not the per-residue protein functional.

Queue runners inject ``--device`` and ``--out`` before the remaining arguments.
Census still runs tokenizer-only on CPU even when ``--device`` names a CUDA card.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.mutation.designed_referent import spearman  # noqa: E402
from src.capability.models.fitness import load_assay  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.core.precision_policy import FP32_PER_TARGET_ABS, TEXT_AA_FP32_V1  # noqa: E402
from src.capability.core.sequence_metadata import PROTEINGYM_ROOT
from src.capability.models.scale_comparison import paired_unit_delta, unit_mean_record  # noqa: E402
from src.capability.models.text_aa_cohort import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED_BASE,
    FAMILY_ADJACENT_PAIRS,
    IMPLEMENTED_PHASES,
    METRIC_INDEX,
    RESEARCH_ROLE,
    RESIDUE_ALIGNED_MODELS,
    STRING_LEVEL_BPE_MODELS,
    SUPPORT_INDEX,
    TEXT_AA_SEED_OFFSET,
    UNIMPLEMENTED_PHASES,
    census_models,
    load_text_aa_boundary_table,
    resolve_text_aa_checkpoint,
    text_aa_draw_seed,
    text_aa_model_names,
)
from src.capability.models.text_aa_fitness import (  # noqa: E402
    ENCODING_RULE_AMENDMENT,
    load_text_aa_scorer,
    resolve_text_aa_boundary,
    text_aa_encoding_record,
)

CENSUS_ARTEFACT = "text_aa_census.json"
PROBE_ARTEFACT = "text_aa_probe.json"
SCORE_ARTEFACT = "text_aa_score.json"
ANALYSE_ARTEFACT = "text_aa_analyse.json"
STANDARD_AA20 = AA20
PROBE_CASES = (
    "A",
    "standard_aa20",
    "first_admitted_wildtype",
    "first_admitted_mutant",
    "longest_legal_input",
)
RESOURCE_REPEATS = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} does not exist")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    name, raw = value.split("=", 1)
    if not name or not raw:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    return name, Path(raw)


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _load_native_dms_extension() -> Any:
    path = REPO_ROOT / "scripts/capability/mutation/native_dms_extension.py"
    spec = importlib.util.spec_from_file_location("native_dms_extension", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def freeze_request(
    *,
    lookup_path: Path,
    proteingym_dir: Path,
    wildtypes_path: Path,
    wildtypes_fasta_path: Path,
) -> dict[str, Any]:
    native = _load_native_dms_extension()
    lookup = _read_json(lookup_path)
    wildtypes = _read_json(wildtypes_path)
    frozen = native.freeze_lookup_cohort(
        lookup,
        proteingym_dir=proteingym_dir,
        wildtypes=wildtypes,
        wildtypes_path=wildtypes_path,
        wildtypes_fasta_path=wildtypes_fasta_path,
    )
    frozen["lookup_sha256"] = sha256_file(lookup_path)
    frozen["lookup_path"] = str(Path(lookup_path).resolve())
    frozen["proteingym_dir"] = str(Path(proteingym_dir).resolve())
    return frozen


def run_census(
    *,
    out: Path,
    lookup_path: Path,
    wildtypes_path: Path,
    wildtypes_fasta_path: Path,
    proteingym_dir: Path,
    models: Sequence[str],
    requested_device: str,
) -> dict[str, Any]:
    table = load_text_aa_boundary_table()
    names = list(models) if models else list(text_aa_model_names(table))
    request = freeze_request(
        lookup_path=lookup_path,
        proteingym_dir=proteingym_dir,
        wildtypes_path=wildtypes_path,
        wildtypes_fasta_path=wildtypes_fasta_path,
    )
    payload = census_models(names, request=request, table=table, repo_root=REPO_ROOT)
    payload["created_utc"] = _utc_now()
    payload["execution"] = {
        "phase": "census",
        "tokenizer_only": True,
        "weights_loaded": False,
        "cuda_invoked": False,
        "requested_device": str(requested_device),
        "executed_on": "cpu",
        "gpu_occupancy_is_not_a_scientific_exclusion": True,
    }
    payload["inputs"] = {
        "lookup_sha256": request["lookup_sha256"],
        "wildtypes_sha256": request["wildtypes_sha256"],
        "wildtypes_fasta_sha256": request["wildtypes_fasta_sha256"],
        "lookup_path": request["lookup_path"],
        "proteingym_dir": request["proteingym_dir"],
        "declared_assays": request["declared_assays"],
        "declared_clusters": request["declared_clusters"],
    }
    write_json(Path(out) / CENSUS_ARTEFACT, payload)
    return payload


def _require_census_model(census: Mapping[str, Any], name: str) -> dict[str, Any]:
    models = census.get("models")
    if not isinstance(models, Mapping) or name not in models:
        raise ValueError(f"census has no model {name!r}")
    row = models[name]
    if not isinstance(row, Mapping):
        raise ValueError(f"census model {name!r} is not a mapping")
    return dict(row)


def _recover_sequence(
    request: Mapping[str, Any], identity: Mapping[str, Any]
) -> str:
    assay = str(identity["assay"])
    role = str(identity["role"])
    by_assay = {str(row["assay"]): row for row in request["assays"]}
    if assay not in by_assay:
        raise ValueError(f"request has no assay {assay!r}")
    if role == "wildtype":
        sequence = str(by_assay[assay]["wildtype_sequence"])
    elif role == "mutant":
        index = identity.get("mutant_index")
        if index is None:
            raise ValueError(f"{assay}: mutant identity has no mutant_index")
        sequences = list(request["sequences"][assay])
        sequence = str(sequences[int(index)])
    else:
        raise ValueError(f"unknown sequence role {role!r}")
    expected = str(identity["sequence_sha256"])
    got = _sequence_sha256(sequence)
    if got != expected:
        raise ValueError(
            f"{assay} {role}: recovered sequence sha256 {got} disagrees with "
            f"census identity {expected}"
        )
    return sequence


def _first_admitted_sequences(
    census_model: Mapping[str, Any], request: Mapping[str, Any]
) -> tuple[str, str, str]:
    admitted = list(census_model.get("native_fixed") or [])
    if not admitted:
        raise ValueError(
            f"{census_model.get('name')}: no admitted assay, so the first-admitted "
            "probe cases cannot run"
        )
    assay = str(admitted[0])
    by_assay = {str(row["assay"]): row for row in request["assays"]}
    if assay not in by_assay:
        raise ValueError(f"first admitted assay {assay!r} is missing from the request")
    wildtype = str(by_assay[assay]["wildtype_sequence"])
    mutants = list(request["sequences"][assay])
    if not mutants:
        raise ValueError(f"{assay}: admitted assay has no selected mutant")
    return assay, wildtype, str(mutants[0])


def _account_probe_case(
    scorer: Any,
    *,
    label: str,
    sequence: str,
) -> dict[str, Any]:
    ids = scorer.encode(sequence)
    encoding = scorer.encoding_metadata(sequence)
    got = float(scorer.log_likelihood([sequence])[0])
    independent, n_target = scorer.independent_shifted_ce_sum(ids)
    if n_target != int(encoding["n_scored_tokens"]):
        raise RuntimeError(
            f"{label}: independent target count {n_target} disagrees with "
            f"n_scored_tokens {encoding['n_scored_tokens']}"
        )
    if n_target == 0:
        per_target = None
        passed = got == 0.0 and independent == 0.0
    else:
        per_target = abs(got - independent) / n_target
        passed = bool(per_target <= FP32_PER_TARGET_ABS)
    return {
        "label": label,
        "n_residues": encoding["n_residues"],
        "n_target_tokens": encoding["n_target_tokens"],
        "n_scored_tokens": encoding["n_scored_tokens"],
        "encoding_class": encoding["encoding_class"],
        "residues_per_token": encoding["residues_per_token"],
        "sequence_sha256": _sequence_sha256(sequence),
        "model_total": got,
        "independent_shifted_ce_total": independent,
        "per_target_abs": per_target,
        "ceiling": FP32_PER_TARGET_ABS,
        "passed": passed,
    }


def _scorer_cuda_device(scorer: Any) -> Any:
    model = getattr(scorer, "model", None)
    if model is None:
        return None
    device = getattr(model, "device", None)
    if device is not None and str(getattr(device, "type", "")) == "cuda":
        return device
    return None


def _resource_gate(scorer: Any, sequence: str) -> dict[str, Any]:
    torch = scorer.torch
    batch = [sequence] * RESOURCE_REPEATS
    cuda_device = _scorer_cuda_device(scorer)
    measured_cuda = cuda_device is not None
    if measured_cuda:
        torch.cuda.reset_peak_memory_stats(cuda_device)
        torch.cuda.synchronize(cuda_device)
    started = time.perf_counter()
    totals = np.asarray(scorer.log_likelihood(batch), dtype=np.float64)
    if measured_cuda:
        torch.cuda.synchronize(cuda_device)
    elapsed = time.perf_counter() - started
    if totals.shape != (RESOURCE_REPEATS,):
        raise RuntimeError(
            f"resource gate shape {totals.shape} != ({RESOURCE_REPEATS},)"
        )
    finite = bool(np.all(np.isfinite(totals)))
    record: dict[str, Any] = {
        "kind": "one_public_log_likelihood_call_longest_actual_input_x3",
        "n_sequences": RESOURCE_REPEATS,
        "sequence_sha256": _sequence_sha256(sequence),
        "n_residues": len(sequence),
        "n_input_tokens": int(scorer.token_lengths([sequence])[0]),
        "finite_output_count": int(np.isfinite(totals).sum()),
        "all_finite": finite,
        "elapsed_seconds": elapsed,
        "cuda_synchronized": measured_cuda,
        "cuda_memory_measured": measured_cuda,
        "log_likelihood_retained": False,
        "passed": finite,
    }
    del totals
    if measured_cuda:
        record["cuda_peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(cuda_device)
        )
        record["cuda_peak_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(cuda_device)
        )
    else:
        record["cuda_peak_allocated_bytes"] = None
        record["cuda_peak_reserved_bytes"] = None
        record["cuda_memory_note"] = (
            "CPU stub: CUDA memory unmeasured; values are not fabricated"
        )
    return record


def run_probe(
    *,
    out: Path,
    name: str,
    census: Mapping[str, Any],
    request: Mapping[str, Any],
    device: str,
    scorer: Any | None = None,
) -> dict[str, Any]:
    """Five labelled cases plus one longest-actual-input ×3 resource gate."""

    if str(census.get("protocol_id")) != TEXT_AA_FP32_V1:
        raise ValueError(
            f"probe accepts only {TEXT_AA_FP32_V1!r} census, "
            f"got {census.get('protocol_id')!r}"
        )
    census_model = _require_census_model(census, name)
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary(name, table)
    window = int(census_model["application_window_tokens"])
    owned = scorer is None
    if scorer is None:
        scorer = load_text_aa_scorer(
            resolve_text_aa_checkpoint(name),
            boundary=boundary,
            application_window_tokens=window,
            device=device,
            name=name,
        )
    failure: str | None = None
    cases: list[dict[str, Any]] = []
    resource: dict[str, Any] | None = None
    try:
        first_assay, first_wt, first_mutant = _first_admitted_sequences(
            census_model, request
        )
        longest = _recover_sequence(
            request, census_model["longest_probe_identity"]
        )
        labelled = {
            "A": "A",
            "standard_aa20": STANDARD_AA20,
            "first_admitted_wildtype": first_wt,
            "first_admitted_mutant": first_mutant,
            "longest_legal_input": longest,
        }
        for label in PROBE_CASES:
            cases.append(
                _account_probe_case(scorer, label=label, sequence=labelled[label])
            )
        resource = _resource_gate(scorer, longest)
        if not all(bool(row["passed"]) for row in cases):
            failure = "a labelled probe case failed the independent shifted-CE gate"
        elif not bool(resource["passed"]):
            failure = "the longest-legal-input resource gate was not all-finite"
        payload = {
            "schema": "text-aa-probe/v1",
            "protocol_id": TEXT_AA_FP32_V1,
            "research_role": RESEARCH_ROLE,
            "encoding_rule_amendment": ENCODING_RULE_AMENDMENT,
            "phase": "probe",
            "name": name,
            "first_admitted_assay": first_assay,
            "application_window_tokens": window,
            "probe_passed": failure is None,
            "failure": failure,
            "cases": cases,
            "resource_gate": resource,
            "created_utc": _utc_now(),
            "experiment_admitted": False,
        }
        write_json(Path(out) / PROBE_ARTEFACT, payload)
        return payload
    finally:
        if owned and scorer is not None and hasattr(scorer, "release"):
            scorer.release()


def _score_admitted_assay(
    scorer: Any,
    *,
    assay_row: Mapping[str, Any],
    request: Mapping[str, Any],
    proteingym_dir: Path,
) -> dict[str, Any]:
    name = str(assay_row["assay"])
    wildtype = str(assay_row["wildtype_sequence"])
    mutants = list(request["sequences"][name])
    loaded = load_assay(
        name,
        n=int(assay_row["n_variants"]),
        seed=int(assay_row["seed"]),
        directory=proteingym_dir,
    )
    if list(loaded.sequences) != mutants:
        raise ValueError(f"{name}: load_assay sequences disagree with the frozen request")
    if loaded.wildtype != wildtype:
        raise ValueError(f"{name}: load_assay wild type disagrees with the frozen request")
    sequences = [wildtype, *mutants]
    encodings = [
        text_aa_encoding_record(sequence, scorer.encode(sequence), scorer.boundary)
        for sequence in sequences
    ]
    totals = np.asarray(scorer.log_likelihood(sequences), dtype=np.float64)
    if totals.shape != (len(sequences),):
        raise RuntimeError(f"{name}: score shape {totals.shape} != ({len(sequences)},)")
    if not np.all(np.isfinite(totals)):
        return {
            "assay": name,
            "index": int(assay_row["index"]),
            "cluster": assay_row["cluster"],
            "wildtype_id": assay_row.get("wildtype_id"),
            "mutant_digest": assay_row["mutant_digest"],
            "csv_sha256": assay_row["csv_sha256"],
            "n_variants": int(assay_row["n_variants"]),
            "spearman": None,
            "instrument_failure": True,
            "reason": "a scored sequence returned a non-finite total",
            "encoding_class_wt": encodings[0]["encoding_class"],
            "n_residues": encodings[0]["n_residues"],
            "n_target_tokens_wt": encodings[0]["n_target_tokens"],
            "residues_per_token_wt": encodings[0]["residues_per_token"],
            "sequence_encodings": encodings,
        }
    deltas = totals[1:] - totals[0]
    rho = spearman(deltas, loaded.scores)
    classes = [str(item["encoding_class"]) for item in encodings]
    uniform = classes[0] if all(item == classes[0] for item in classes) else "mixed"
    return {
        "assay": name,
        "index": int(assay_row["index"]),
        "cluster": assay_row["cluster"],
        "wildtype_id": assay_row.get("wildtype_id"),
        "mutant_digest": assay_row["mutant_digest"],
        "csv_sha256": assay_row["csv_sha256"],
        "n_variants": int(assay_row["n_variants"]),
        "spearman": None if rho is None else float(rho),
        "instrument_failure": False,
        "encoding_class": uniform,
        "encoding_class_wt": encodings[0]["encoding_class"],
        "n_residues": encodings[0]["n_residues"],
        "n_target_tokens_wt": encodings[0]["n_target_tokens"],
        "residues_per_token_wt": encodings[0]["residues_per_token"],
        "n_one_token_per_residue": sum(
            1 for item in encodings if item["encoding_class"] == "one_token_per_residue"
        ),
        "n_merged_bpe": sum(
            1 for item in encodings if item["encoding_class"] == "merged_bpe"
        ),
        "sequence_encodings": encodings,
    }


def run_score(
    *,
    out: Path,
    name: str,
    census_path: Path,
    lookup_path: Path,
    wildtypes_path: Path,
    wildtypes_fasta_path: Path,
    proteingym_dir: Path,
    device: str,
    scorer: Any | None = None,
) -> dict[str, Any]:
    """Probe in-process, then score admitted assays. Refuse a score if probe fails."""

    census = _read_json(census_path)
    if str(census.get("protocol_id")) != TEXT_AA_FP32_V1:
        raise ValueError(
            f"score accepts only {TEXT_AA_FP32_V1!r} census, "
            f"got {census.get('protocol_id')!r}"
        )
    request = freeze_request(
        lookup_path=lookup_path,
        proteingym_dir=proteingym_dir,
        wildtypes_path=wildtypes_path,
        wildtypes_fasta_path=wildtypes_fasta_path,
    )
    census_model = _require_census_model(census, name)
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary(name, table)
    window = int(census_model["application_window_tokens"])
    owned = scorer is None
    local = scorer
    if local is None:
        local = load_text_aa_scorer(
            resolve_text_aa_checkpoint(name),
            boundary=boundary,
            application_window_tokens=window,
            device=device,
            name=name,
        )
    try:
        probe = run_probe(
            out=out,
            name=name,
            census=census,
            request=request,
            device=device,
            scorer=local,
        )
        if not probe["probe_passed"]:
            raise RuntimeError(
                f"{name}: probe failed ({probe.get('failure')}); "
                "refusing to emit a scientific score"
            )
        by_assay = {str(row["assay"]): row for row in request["assays"]}
        rows: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        instrument = False
        for assay_name in census_model["native_fixed"]:
            assay_row = by_assay[str(assay_name)]
            row = _score_admitted_assay(
                local,
                assay_row=assay_row,
                request=request,
                proteingym_dir=proteingym_dir,
            )
            if row.get("instrument_failure"):
                instrument = True
                skipped.append(
                    {
                        "assay": row["assay"],
                        "reason": "instrument_failure",
                        "detail": row.get("reason"),
                    }
                )
            rows.append(row)
        payload = {
            "schema": "text-aa-score/v1",
            "protocol_id": TEXT_AA_FP32_V1,
            "research_role": RESEARCH_ROLE,
            "encoding_rule_amendment": ENCODING_RULE_AMENDMENT,
            "functional_note": (
                "ByGPT5 is the residue-aligned subset; GPT-2, Qwen, Llama and "
                "DialoGPT are string-level BPE controls. Merged-BPE text scores "
                "are not the same functional as residue protein models."
            ),
            "phase": "score",
            "name": name,
            "created_utc": _utc_now(),
            "settings": {
                "dtype": "float32",
                "batch_size": 1,
                "scoring_stratum": local.scoring_stratum,
                "score": local.score_description,
                "application_window_tokens": window,
            },
            "facts": dict(getattr(local, "facts", {})),
            "probe": {
                "path": str((Path(out) / PROBE_ARTEFACT).resolve()),
                "sha256": sha256_file(Path(out) / PROBE_ARTEFACT),
                "probe_passed": True,
            },
            "census_sha256": sha256_file(census_path),
            "assays": rows,
            "skipped": skipped,
            "native_fixed": list(census_model["native_fixed"]),
            "instrument_failure": instrument,
            "scientific_admission": False,
            "experiment_admitted": False,
            "no_acquired_or_retrieval_bounded_verdict": True,
        }
        write_json(Path(out) / SCORE_ARTEFACT, payload)
        return payload
    finally:
        if owned and local is not None and hasattr(local, "release"):
            local.release()


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _complete_or_defined(
    values: dict[str, float | None],
    units: dict[str, str],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    missing = [name for name, value in values.items() if value is None]
    defined = {name: float(value) for name, value in values.items() if value is not None}
    defined_units = {name: units[name] for name in defined}
    defined_record = (
        None
        if not defined
        else unit_mean_record(defined, defined_units, resamples=resamples, seed=seed)
    )
    if missing:
        return {
            "point": None,
            "interval": None,
            "complete_fixed_set": False,
            "undefined_assays": missing,
            "n_undefined": len(missing),
            "defined_only": defined_record,
            "n_defined": len(defined),
        }
    record = unit_mean_record(defined, defined_units, resamples=resamples, seed=seed)
    record["complete_fixed_set"] = True
    record["defined_only"] = None
    return record


def _lookup_channels(lookup: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    lookup_raw: dict[str, float] = {}
    blosum_raw: dict[str, float] = {}
    units: dict[str, str] = {}
    for row in lookup["assays"]:
        assay = str(row["assay"])
        channels = row["spearman"]
        lookup_raw[assay] = float(channels["lookup"])
        blosum_raw[assay] = float(channels["blosum62"])
        units[assay] = str(row["cluster"])
    return lookup_raw, blosum_raw, units


def _per_assay_from_score(
    score: Mapping[str, Any],
    assays: Sequence[str],
) -> dict[str, float | None]:
    by_assay = {str(row["assay"]): row for row in score.get("assays") or []}
    values: dict[str, float | None] = {}
    for assay in assays:
        if assay not in by_assay:
            values[assay] = None
            continue
        row = by_assay[assay]
        if row.get("instrument_failure"):
            values[assay] = None
            continue
        values[assay] = _finite_or_none(row.get("spearman"))
    return values


def _model_channels(
    score: Mapping[str, Any],
    assays: Sequence[str],
    lookup_raw: Mapping[str, float],
    blosum_raw: Mapping[str, float],
) -> dict[str, dict[str, float | None]]:
    raw = _per_assay_from_score(score, assays)
    minus_lookup: dict[str, float | None] = {}
    minus_blosum: dict[str, float | None] = {}
    for assay, value in raw.items():
        if value is None:
            minus_lookup[assay] = None
            minus_blosum[assay] = None
            continue
        minus_lookup[assay] = value - float(lookup_raw[assay])
        minus_blosum[assay] = value - float(blosum_raw[assay])
    return {
        "raw": raw,
        "model_minus_lookup": minus_lookup,
        "model_minus_blosum62": minus_blosum,
    }


def _order_index(name: str, order: Sequence[str]) -> int:
    return list(order).index(name)


def run_analyse(
    *,
    out: Path,
    census_path: Path,
    lookup_path: Path,
    scores: Mapping[str, Path],
) -> dict[str, Any]:
    """CPU family-bootstrap of raw / MODEL−LOOKUP / MODEL−BLOSUM62. No residue verdicts."""

    census = _read_json(census_path)
    if str(census.get("protocol_id")) != TEXT_AA_FP32_V1:
        raise ValueError(
            f"analyse accepts only {TEXT_AA_FP32_V1!r} census, "
            f"got {census.get('protocol_id')!r}"
        )
    lookup = _read_json(lookup_path)
    table = load_text_aa_boundary_table()
    order = list(text_aa_model_names(table))
    requested = [str(name) for name in census.get("models_requested") or order]
    unknown = [name for name in requested if name not in order]
    if unknown:
        raise KeyError(f"analyse census names unknown text-AA models {unknown}")
    loaded: dict[str, dict[str, Any]] = {}
    for name in requested:
        if name not in scores:
            raise FileNotFoundError(f"analyse is missing a score for {name}")
        path = Path(scores[name])
        if not path.is_file():
            raise FileNotFoundError(f"analyse is missing a score for {name}: {path}")
        payload = _read_json(path)
        if str(payload.get("protocol_id")) != TEXT_AA_FP32_V1:
            raise ValueError(
                f"{name}: score protocol {payload.get('protocol_id')!r} is not "
                f"{TEXT_AA_FP32_V1!r}"
            )
        loaded[name] = payload
    lookup_raw, blosum_raw, units = _lookup_channels(lookup)
    support = census["support_sets"]
    native_fixed = {
        name: list(support["native_fixed"][name])
        for name in order
        if name in support.get("native_fixed", {})
    }
    native_records: dict[str, Any] = {}
    for name in requested:
        index = _order_index(name, order)
        assays = native_fixed.get(name) or []
        channels = _model_channels(loaded[name], assays, lookup_raw, blosum_raw)
        native_records[name] = {
            "support_index": index,
            "n_assays": len(assays),
            "encoding_role": (
                "residue-aligned subset"
                if name in RESIDUE_ALIGNED_MODELS
                else "string-level BPE control"
            ),
            "raw": _complete_or_defined(
                channels["raw"],
                {assay: units[assay] for assay in assays},
                resamples=BOOTSTRAP_RESAMPLES,
                seed=text_aa_draw_seed(
                    support_index=index,
                    metric_index=METRIC_INDEX["raw"],
                    contrast_index=index,
                ),
            ),
            "model_minus_lookup": _complete_or_defined(
                channels["model_minus_lookup"],
                {assay: units[assay] for assay in assays},
                resamples=BOOTSTRAP_RESAMPLES,
                seed=text_aa_draw_seed(
                    support_index=index,
                    metric_index=METRIC_INDEX["model_minus_lookup"],
                    contrast_index=index,
                ),
            ),
            "model_minus_blosum62": _complete_or_defined(
                channels["model_minus_blosum62"],
                {assay: units[assay] for assay in assays},
                resamples=BOOTSTRAP_RESAMPLES,
                seed=text_aa_draw_seed(
                    support_index=index,
                    metric_index=METRIC_INDEX["model_minus_blosum62"],
                    contrast_index=index,
                ),
            ),
        }

    def _support_block(
        *,
        members: Sequence[str],
        assays: Sequence[str] | None,
        complete: bool,
        support_index: int,
        pairs: Sequence[tuple[str, str]],
    ) -> dict[str, Any]:
        if not complete or assays is None:
            return {
                "complete": False,
                "assays": None,
                "n_assays": None,
                "per_model": None,
                "adjacent_delta": None,
            }
        assay_units = {assay: units[assay] for assay in assays}
        per_model: dict[str, Any] = {}
        raw_maps: dict[str, dict[str, float]] = {}
        for name in members:
            channels = _model_channels(loaded[name], assays, lookup_raw, blosum_raw)
            contrast = _order_index(name, order)
            per_model[name] = {
                "encoding_role": (
                    "residue-aligned subset"
                    if name in RESIDUE_ALIGNED_MODELS
                    else "string-level BPE control"
                ),
                "raw": _complete_or_defined(
                    channels["raw"],
                    assay_units,
                    resamples=BOOTSTRAP_RESAMPLES,
                    seed=text_aa_draw_seed(
                        support_index=support_index,
                        metric_index=METRIC_INDEX["raw"],
                        contrast_index=contrast,
                    ),
                ),
                "model_minus_lookup": _complete_or_defined(
                    channels["model_minus_lookup"],
                    assay_units,
                    resamples=BOOTSTRAP_RESAMPLES,
                    seed=text_aa_draw_seed(
                        support_index=support_index,
                        metric_index=METRIC_INDEX["model_minus_lookup"],
                        contrast_index=contrast,
                    ),
                ),
                "model_minus_blosum62": _complete_or_defined(
                    channels["model_minus_blosum62"],
                    assay_units,
                    resamples=BOOTSTRAP_RESAMPLES,
                    seed=text_aa_draw_seed(
                        support_index=support_index,
                        metric_index=METRIC_INDEX["model_minus_blosum62"],
                        contrast_index=contrast,
                    ),
                ),
            }
            defined_raw = {
                assay: float(value)
                for assay, value in channels["raw"].items()
                if value is not None
            }
            raw_maps[name] = defined_raw
        adjacent: dict[str, Any] = {}
        for pair_index, (smaller, larger) in enumerate(pairs):
            shared = sorted(set(raw_maps[smaller]) & set(raw_maps[larger]))
            if not shared:
                adjacent[f"{smaller}__{larger}"] = {
                    "point": None,
                    "interval": None,
                    "n_defined": 0,
                    "note": "no assay is defined on both sides",
                }
                continue
            adjacent[f"{smaller}__{larger}"] = paired_unit_delta(
                {assay: raw_maps[smaller][assay] for assay in shared},
                {assay: raw_maps[larger][assay] for assay in shared},
                {assay: units[assay] for assay in shared},
                resamples=BOOTSTRAP_RESAMPLES,
                seed=text_aa_draw_seed(
                    support_index=support_index,
                    metric_index=METRIC_INDEX["family_adjacent_delta"],
                    contrast_index=pair_index,
                ),
            )
            adjacent[f"{smaller}__{larger}"]["n_defined"] = len(shared)
            adjacent[f"{smaller}__{larger}"]["not_the_fixed_set"] = len(shared) != len(
                assays
            )
        return {
            "complete": True,
            "assays": list(assays),
            "n_assays": len(assays),
            "per_model": per_model,
            "adjacent_delta": adjacent,
        }

    common = support["common_all13"]
    families = {
        family: _support_block(
            members=list(table["families"][family]),
            assays=support["families"][family].get("assays"),
            complete=bool(support["families"][family].get("complete")),
            support_index=int(SUPPORT_INDEX[family]),
            pairs=FAMILY_ADJACENT_PAIRS[family],
        )
        for family in ("gpt2", "qwen2.5", "bygpt5")
    }
    payload = {
        "schema": "text-aa-analyse/v1",
        "protocol_id": TEXT_AA_FP32_V1,
        "research_role": RESEARCH_ROLE,
        "encoding_rule_amendment": ENCODING_RULE_AMENDMENT,
        "phase": "analyse",
        "created_utc": _utc_now(),
        "functional_note": {
            "residue_aligned_subset": list(RESIDUE_ALIGNED_MODELS),
            "string_level_bpe_controls": list(STRING_LEVEL_BPE_MODELS),
            "not_the_same_functional_as_residue_protein_models": True,
            "no_acquired_or_retrieval_bounded_verdict": True,
        },
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed_base": BOOTSTRAP_SEED_BASE,
            "offset": TEXT_AA_SEED_OFFSET,
            "formula": (
                "base + 2000 + 10000 * support_index + 100 * metric_index + "
                "contrast_index"
            ),
            "support_index": dict(SUPPORT_INDEX),
            "metric_index": dict(METRIC_INDEX),
            "endpoint_plus10_pair_offset_reused": False,
        },
        "inputs": {
            "census_sha256": sha256_file(census_path),
            "lookup_sha256": sha256_file(lookup_path),
            "scores": {name: sha256_file(path) for name, path in scores.items()},
        },
        "native": native_records,
        "common_all13": _support_block(
            members=order,
            assays=common.get("assays"),
            complete=bool(common.get("complete")),
            support_index=int(SUPPORT_INDEX["common_all13"]),
            pairs=(),
        ),
        "families": families,
        "experiment_admitted": False,
        "no_acquired_or_retrieval_bounded_verdict": True,
        "descriptive_not_causal": True,
    }
    write_json(Path(out) / ANALYSE_ARTEFACT, payload)
    return payload


def _require_request_paths(args: argparse.Namespace, *, phase: str) -> None:
    if args.lookup is None or args.wildtypes is None or args.wildtypes_fasta is None:
        raise ValueError(
            f"{phase} requires --lookup, --wildtypes, and --wildtypes-fasta"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=IMPLEMENTED_PHASES,
        help="flat positional phase so queue injection of --device/--out still parses",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cpu",
        help="accepted because campaign runners inject it; census does not use CUDA",
    )
    parser.add_argument(
        "--protocol",
        dest="protocol_id",
        default=TEXT_AA_FP32_V1,
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="repeatable; default is the full 13. A subset cannot claim common_all13",
    )
    parser.add_argument("--lookup", type=Path)
    parser.add_argument("--wildtypes", type=Path)
    parser.add_argument("--wildtypes-fasta", type=Path)
    parser.add_argument("--proteingym-dir", type=Path, default=None)
    parser.add_argument("--census", type=Path, help="probe/score/analyse: census JSON")
    parser.add_argument(
        "--score",
        action="append",
        default=[],
        help="analyse: NAME=PATH of a text_aa_score.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    if str(args.protocol_id) != TEXT_AA_FP32_V1:
        raise ValueError(
            f"text-AA controller accepts only {TEXT_AA_FP32_V1!r}, got {args.protocol_id!r}"
        )
    if UNIMPLEMENTED_PHASES:
        raise RuntimeError(
            f"UNIMPLEMENTED_PHASES must be empty after the 2026-09-20 amendment; "
            f"got {UNIMPLEMENTED_PHASES}"
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proteingym = Path(args.proteingym_dir) if args.proteingym_dir else PROTEINGYM_ROOT
    if args.phase == "census":
        _require_request_paths(args, phase="census")
        return run_census(
            out=out,
            lookup_path=Path(args.lookup),
            wildtypes_path=Path(args.wildtypes),
            wildtypes_fasta_path=Path(args.wildtypes_fasta),
            proteingym_dir=proteingym,
            models=list(args.model),
            requested_device=str(args.device),
        )
    if args.phase == "probe":
        if len(args.model) != 1:
            raise ValueError("probe requires exactly one --model")
        if args.census is None:
            raise ValueError("probe requires --census")
        _require_request_paths(args, phase="probe")
        request = freeze_request(
            lookup_path=Path(args.lookup),
            proteingym_dir=proteingym,
            wildtypes_path=Path(args.wildtypes),
            wildtypes_fasta_path=Path(args.wildtypes_fasta),
        )
        return run_probe(
            out=out,
            name=str(args.model[0]),
            census=_read_json(Path(args.census)),
            request=request,
            device=str(args.device),
        )
    if args.phase == "score":
        if len(args.model) != 1:
            raise ValueError("score requires exactly one --model")
        if args.census is None:
            raise ValueError("score requires --census")
        _require_request_paths(args, phase="score")
        return run_score(
            out=out,
            name=str(args.model[0]),
            census_path=Path(args.census),
            lookup_path=Path(args.lookup),
            wildtypes_path=Path(args.wildtypes),
            wildtypes_fasta_path=Path(args.wildtypes_fasta),
            proteingym_dir=proteingym,
            device=str(args.device),
        )
    if args.phase == "analyse":
        if args.census is None:
            raise ValueError("analyse requires --census")
        if args.lookup is None:
            raise ValueError("analyse requires --lookup")
        named = dict(_parse_named_path(item) for item in args.score)
        return run_analyse(
            out=out,
            census_path=Path(args.census),
            lookup_path=Path(args.lookup),
            scores=named,
        )
    raise ValueError(f"unknown phase {args.phase!r}")


if __name__ == "__main__":
    main()
