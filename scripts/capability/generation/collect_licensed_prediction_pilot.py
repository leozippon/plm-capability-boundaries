#!/usr/bin/env python3
"""Licensed prediction pilot: native seeded generation and response measurement.

This is one cell of the two-domain, two-arm, three-stream licensed pilot. Each
invocation samples unsteered native continuations of one domain's exact WT
conditioning prefix under one admitted protein arm, then measures the native
full-sum residue-span score of every eligible design and of its generated-suffix
single substitutions. It is a model-internal measurement: the native score is a
model quantity, not a biological conditional probability, and no label,
structural estimate or physics quantity is produced or inferred here.

Admitted arms are ``protgpt2`` and ``progen2-medium`` only. Both are loaded once,
in float32, through the admitted readout door so that sampling and scoring share
one handle. The generator loader in ``unconditional_generation`` is deliberately
not used: its admitted branch ignores a checkpoint override and defaults to
bfloat16, which is unusable for a score calibration. That shared loader is left
unmodified and the actually resolved handle is recorded instead.

Protocol schema (frozen by main; ``licensed_prediction_pilot_v1``)

``targets``
    Mapping ``domain -> {wt_sequence, wt_sha256, L, group}``. ``L`` is the exact
    WT length in residues and must equal ``len(wt_sequence)``; ``wt_sha256`` is
    the SHA-256 of the exact UTF-8 sequence bytes.
``models``
    The two admitted arms, either a list or a mapping keyed by arm name. The set
    must be exactly ``{protgpt2, progen2-medium}``.
``seed_streams``
    Exactly three integer stream seeds. ``--stream-index`` selects one by
    position. The intervals ``[seed, seed + hard_attempt_cap)`` must be pairwise
    disjoint and deliberately separated, never adjacent.
``sampling``
    ``temperature`` 1.0, ``top_p`` 1.0, ``top_k`` 0, ``max_new_tokens`` 400,
    ``max_eligible_attempts`` 32, ``hard_attempt_cap`` 256. Every value is
    required exactly; nothing is relaxed.
``native_contract``
    ``batch_size`` 1, ``budget`` 1024, ``dtype`` ``float32`` and a declared
    ``score`` string. Every required value is enforced.
``inputs``
    Mapping of named provenance inputs, each ``{path, sha256}``. ``path`` is
    resolved under ``--root``; a path that escapes the root is refused.
    ``native_calibration`` must be present and point at the frozen
    ``licensed_native_calibration_v1`` file. Every listed file is re-hashed and
    must match its declaration before any sampling happens. The calibration is
    pinned here and nowhere else: the protocol carries no duplicate top-level
    calibration entry.

The ``licensed_native_calibration_v1`` file the protocol pins only through
``inputs.native_calibration`` holds, for each admitted arm, the old-archive
``identity`` and ``manifest``, and per target an ``archive`` plus exactly three
``states``: state 0 is the exact WT and states 1 and 2 are frozen single
substitutions. The script selects the arm's target whose state 0 binds the
protocol WT SHA, requires ``tolerance_nats`` exactly 0.001, and re-scores the
three states before sampling. The cached ``packed_token_count``,
``scored_token_count``, ``pooled_token_count`` and full ``token_ids`` must match
the native packing exactly, each score must be finite and within the tolerance,
and the file's declared arm, batch size, budget and float32 dtype must match
``native_contract``. A mismatch of any kind refuses the run outright.

Outputs, all owned by a fresh ``--out`` directory and never resumed or
overwritten: ``attempts.jsonl`` (every attempt, eligible or rejected),
``candidate_reference_index.json`` (candidate SHA to attempt/stream identity and
references), ``grids/<candidate_sha>.npz`` plus ``grid_manifest.json`` (one grid
of the base design and its 20-response-per-site substitution ladder), and
``licensed_pilot_collection.json`` (the completion record with protocol, source,
checkpoint and input digests and the exact query and calibration counts). A
successful collection with zero eligible designs records completion with
``analysis_available`` false: that is an unavailable analysis, not a biological
negative. A failure writes an explicit failure record and raises; it never
writes a completion record.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.context import context_homologue as ch  # noqa: E402
from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.generation import generation_evidence as ge  # noqa: E402
from src.capability.generation.unconditional_generation import (  # noqa: E402
    _eos_id,
    _sample_next_id,
    require_admitted,
)
from src.capability.readouts.readout_extraction import load_readout_arm  # noqa: E402

SCHEMA = "licensed_prediction_pilot_v1"
CALIBRATION_SCHEMA = "licensed_native_calibration_v1"
COLLECTION_SCHEMA = "licensed_prediction_pilot_collection_v1"
ATTEMPT_SCHEMA = "licensed_prediction_pilot_attempt_v1"
INDEX_SCHEMA = "licensed_prediction_pilot_index_v1"
MANIFEST_SCHEMA = "licensed_prediction_pilot_grid_manifest_v1"
FAILURE_SCHEMA = "licensed_prediction_pilot_failure_v1"

#: The admitted native generation/scoring arms of this pilot.
ALLOWED_ARMS = ("protgpt2", "progen2-medium")
#: The two licensed domains, identified by their PDB token, and their exact WT lengths.
LICENSED_TARGETS: dict[str, int] = {"1UFM.pdb": 72, "3L1X.pdb": 69}
#: The licensed length band; a target outside it is refused rather than used generically.
LICENSED_BAND = (60, 80)
#: The exact WT prefix residues that condition generation and are never mutated.
PREFIX_RESIDUES = 8
#: Every value is settled before the run; a deviation is refused rather than logged.
SETTLED_SAMPLING: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 400,
    "max_eligible_attempts": 32,
    "hard_attempt_cap": 256,
}
SETTLED_NATIVE_CONTRACT: dict[str, Any] = {"batch_size": 1, "budget": 1024, "dtype": "float32"}
SETTLED_TOLERANCE_NATS = 0.001
STREAM_COUNT = 3
#: The seven output stop statuses of one attempt, kept as an explicit vocabulary.
ATTEMPT_STATUSES = (
    "design_length_stop",
    "early_eos",
    "noncanonical",
    "overshoot",
    "budget_exhausted",
)

ScoreFn = Callable[[str], float]
TokenRecordFn = Callable[[str], dict]
NextIdFn = Callable[[torch.Tensor], int]


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name!r} is refused")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_immutable_json(path: Path, value: Any) -> None:
    ge.write_immutable(Path(path), _json_bytes(value))


def _dtos() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare_output_dir(path: Path) -> Path:
    """A fresh, owned output directory. An existing non-empty directory is refused."""
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output path {path} exists and is not a directory")
        if any(path.iterdir()):
            raise ValueError(f"refusing to reuse the non-empty output directory {path}")
    else:
        path.mkdir(parents=True)
    return path


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_stage46(root: Path) -> Any:
    return _load_module(
        "licensed_pilot_stage46", root / "scripts/capability/stages/context_homologue.py"
    )


def _load_stability_singles(root: Path) -> Any:
    return _load_module(
        "licensed_pilot_stability_singles",
        root / "scripts/capability/stability/extract_stability_singles.py",
    )


def _code_paths(root: Path) -> list[Path]:
    return [
        root / "scripts/capability/generation/collect_licensed_prediction_pilot.py",
        root / "scripts/capability/stability/extract_stability_singles.py",
        root / "scripts/capability/stages/context_homologue.py",
        root / "src/capability/readouts/readout_extraction.py",
        root / "src/capability/context/context_homologue.py",
        root / "src/capability/generation/unconditional_generation.py",
        root / "src/capability/core/arms.py",
        root / "src/capability/core/io.py",
    ]


# --------------------------------------------------------------- protocol binding


def _require_str(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _sequence_sha(sequence: str) -> str:
    return sha256(sequence.encode("utf-8")).hexdigest()


def _require_aa20(sequence: Any, *, field: str) -> str:
    _require_str(sequence, field=field)
    offenders = sorted(set(sequence) - set(AA20))
    if offenders:
        raise ValueError(f"{field} contains non-AA20 characters {offenders}")
    return sequence


def _hamming(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("hamming distance needs equal-length sequences")
    return sum(a != b for a, b in zip(left, right, strict=True))


def validate_protocol(protocol: Any, arm: str, domain: str, stream_index: int) -> dict[str, Any]:
    """Validate the frozen protocol shape and bind it to one cell, without filesystem access."""
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a JSON object")
    if protocol.get("schema") != SCHEMA:
        raise ValueError(f"protocol schema must be {SCHEMA!r}")

    models = protocol.get("models")
    if not isinstance(models, list) or any(not isinstance(name, str) for name in models):
        raise ValueError("protocol.models must be a list of the two admitted arm names")
    if len(models) != len(ALLOWED_ARMS) or set(models) != set(ALLOWED_ARMS):
        raise ValueError(
            f"protocol.models must be exactly {list(ALLOWED_ARMS)}; got {models!r}"
        )
    arm_names = list(models)
    if arm not in arm_names:
        raise ValueError(f"arm {arm!r} is not declared by the protocol")

    sampling = protocol.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("protocol.sampling must be an object")
    for key, expected in SETTLED_SAMPLING.items():
        observed = sampling.get(key)
        if isinstance(expected, float):
            same = isinstance(observed, (int, float)) and not isinstance(observed, bool) and float(observed) == expected
        else:
            same = _safe_int_equal(observed, expected)
        if not same:
            raise ValueError(f"protocol.sampling.{key} must be {expected!r}, got {observed!r}")

    contract = protocol.get("native_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("protocol.native_contract must be an object")
    for key, expected in SETTLED_NATIVE_CONTRACT.items():
        if contract.get(key) != expected:
            raise ValueError(f"protocol.native_contract.{key} must be {expected!r}")
    _require_str(contract.get("score"), field="protocol.native_contract.score")

    seeds = protocol.get("seed_streams")
    if not isinstance(seeds, list) or len(seeds) != STREAM_COUNT:
        raise ValueError(f"protocol.seed_streams must list exactly {STREAM_COUNT} integer seeds")
    parsed_seeds = [_require_int(seed, field="protocol.seed_streams[]") for seed in seeds]
    if len(set(parsed_seeds)) != STREAM_COUNT:
        raise ValueError("protocol.seed_streams must be pairwise distinct")
    hard_cap = int(SETTLED_SAMPLING["hard_attempt_cap"])
    ordered = sorted(parsed_seeds)
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        if left + hard_cap > right:
            raise ValueError(
                "protocol.seed_streams must occupy pairwise disjoint, non-adjacent 10k ranges; "
                f"{left} and {right} overlap"
            )
    if not 0 <= stream_index < STREAM_COUNT:
        raise ValueError(f"stream index {stream_index} is outside 0..{STREAM_COUNT - 1}")
    stream_seed = parsed_seeds[stream_index]

    targets = protocol.get("targets")
    if not isinstance(targets, Mapping):
        raise ValueError("protocol.targets must be an object")
    matched: dict[str, Any] = {}
    for key, value in targets.items():
        if key not in LICENSED_TARGETS:
            raise ValueError(
                f"protocol.targets key {key!r} is not exactly one of the licensed domains "
                f"{sorted(LICENSED_TARGETS)}"
            )
        matched[key] = value
    if sorted(matched) != sorted(LICENSED_TARGETS):
        raise ValueError(
            f"protocol.targets must name exactly the licensed pair {sorted(LICENSED_TARGETS)}; "
            f"got {sorted(targets)}"
        )
    if domain not in targets:
        raise ValueError(f"protocol.targets has no entry for domain {domain!r}")
    entry = targets[domain]
    if not isinstance(entry, Mapping):
        raise ValueError(f"protocol.targets.{domain} must be an object")
    wt_sequence = _require_aa20(entry.get("wt_sequence"), field=f"protocol.targets.{domain}.wt_sequence")
    wt_sha = _require_str(entry.get("wt_sha256"), field=f"protocol.targets.{domain}.wt_sha256")
    if wt_sha != _sequence_sha(wt_sequence):
        raise ValueError(f"protocol.targets.{domain}.wt_sha256 does not bind the WT bytes")
    length = _require_int(entry.get("L"), field=f"protocol.targets.{domain}.L")
    if length != len(wt_sequence):
        raise ValueError(f"protocol.targets.{domain}.L must equal the WT length {len(wt_sequence)}")
    if length != LICENSED_TARGETS[domain]:
        raise ValueError(
            f"protocol.targets.{domain}.L must be the licensed length {LICENSED_TARGETS[domain]}"
        )
    if not LICENSED_BAND[0] <= length <= LICENSED_BAND[1]:
        raise ValueError(
            f"protocol.targets.{domain}.L={length} is outside the licensed band {LICENSED_BAND}"
        )
    if length <= PREFIX_RESIDUES:
        raise ValueError(
            f"protocol.targets.{domain}.L={length} leaves no generated suffix after the "
            f"{PREFIX_RESIDUES}-residue prefix"
        )
    group = _require_str(entry.get("group"), field=f"protocol.targets.{domain}.group")

    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping) or "native_calibration" not in inputs:
        raise ValueError("protocol.inputs must declare a native_calibration entry")
    for name, descriptor in inputs.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"protocol.inputs.{name} must be an object")
        _require_str(descriptor.get("path"), field=f"protocol.inputs.{name}.path")
        declared = _require_str(descriptor.get("sha256"), field=f"protocol.inputs.{name}.sha256")
        if len(declared) != 64 or any(c not in "0123456789abcdef" for c in declared):
            raise ValueError(f"protocol.inputs.{name}.sha256 must be a 64-character hex digest")

    return {
        "target": {
            "domain": domain,
            "wt_sequence": wt_sequence,
            "wt_sha256": wt_sha,
            "L": length,
            "group": group,
        },
        "arm": arm,
        "stream_index": stream_index,
        "stream_seed": stream_seed,
        "seeds": parsed_seeds,
        "sampling": {key: sampling[key] for key in SETTLED_SAMPLING},
        "native_contract": dict(contract),
    }


def _safe_int_equal(observed: Any, expected: int) -> bool:
    return (
        not isinstance(observed, bool)
        and isinstance(observed, int)
        and int(observed) == int(expected)
    )


def resolve_inputs(protocol: Mapping[str, Any], root: Path) -> dict[str, dict[str, str]]:
    """Re-hash every declared provenance input, refusing a path that escapes ``root``."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"input root {root} is not a directory")
    resolved: dict[str, dict[str, str]] = {}
    for name, descriptor in protocol["inputs"].items():
        declared = Path(descriptor["path"])
        candidate = declared if declared.is_absolute() else root / declared
        candidate = candidate.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"protocol.inputs.{name}.path escapes the declared root")
        if not candidate.is_file():
            raise ValueError(f"protocol.inputs.{name} is not a file: {candidate}")
        actual = sha256_file(candidate)
        if actual != descriptor["sha256"]:
            raise ValueError(
                f"protocol.inputs.{name} digest mismatch: declared {descriptor['sha256']}, "
                f"observed {actual}"
            )
        resolved[name] = {"path": descriptor["path"], "resolved": str(candidate), "sha256": actual}
    return resolved


# ------------------------------------------------------------- native calibration


def validate_native_calibration_payload(payload: Any) -> dict[str, Any]:
    """Validate the ``licensed_native_calibration_v1`` envelope and fixed tolerance."""
    if not isinstance(payload, Mapping) or payload.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(f"native calibration schema must be {CALIBRATION_SCHEMA!r}")
    tolerance = payload.get("tolerance_nats")
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or float(tolerance) != SETTLED_TOLERANCE_NATS
    ):
        raise ValueError(
            f"native calibration tolerance_nats must be the fixed {SETTLED_TOLERANCE_NATS}"
        )
    return dict(payload)


def load_native_calibration(path: Path) -> dict[str, Any]:
    """Read the frozen ``licensed_native_calibration_v1`` file, rejecting non-finite JSON."""
    payload = json.loads(Path(path).read_bytes().decode("utf-8"), parse_constant=_reject_constant)
    return validate_native_calibration_payload(payload)


def _calibration_state(state: Any, *, index: int, target: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError(f"native calibration state {index} must be an object")
    if _require_int(state.get("state"), field=f"native_calibration.state[{index}].state") != index:
        raise ValueError(f"native calibration state {index} carries the wrong state index")
    sequence = _require_aa20(state.get("sequence"), field=f"native_calibration.state[{index}].sequence")
    sequence_sha = _require_str(
        state.get("sequence_sha256"), field=f"native_calibration.state[{index}].sequence_sha256"
    )
    if sequence_sha != _sequence_sha(sequence):
        raise ValueError(f"native calibration state {index} sha256 does not bind its sequence")
    score = state.get("score_nats")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
        raise ValueError(f"native calibration state {index} cached score must be finite")
    token_ids = state.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids or any(
        isinstance(value, bool) or not isinstance(value, int) for value in token_ids
    ):
        raise ValueError(f"native calibration state {index} token_ids must be a non-empty int list")
    position = _require_int(state.get("position"), field=f"native_calibration.state[{index}].position")
    mutant = state.get("mutant")
    if index == 0:
        if mutant is not None or position != 0:
            raise ValueError("native calibration state 0 must be the unmutated WT")
        if sequence != target["wt_sequence"] or sequence_sha != target["wt_sha256"]:
            raise ValueError("native calibration state 0 is not the exact target WT")
    else:
        mutant = _require_str(mutant, field=f"native_calibration.state[{index}].mutant")
        if len(mutant) != 1 or mutant not in AA20:
            raise ValueError(f"native calibration state {index} mutant must be one AA20 residue")
        if _hamming(sequence, target["wt_sequence"]) != 1:
            raise ValueError(f"native calibration state {index} is not a single substitution of the WT")
        changed = next(
            offset
            for offset in range(len(sequence))
            if sequence[offset] != target["wt_sequence"][offset]
        )
        if position != changed + 1 or sequence[position - 1] != mutant:
            raise ValueError(f"native calibration state {index} position/mutant does not bind its sequence")
    return {
        "state": index,
        "sequence": sequence,
        "sequence_sha256": sequence_sha,
        "score_nats": float(score),
        "packed_token_count": _require_int(
            state.get("packed_token_count"), field=f"native_calibration.state[{index}].packed_token_count"
        ),
        "scored_token_count": _require_int(
            state.get("scored_token_count"), field=f"native_calibration.state[{index}].scored_token_count"
        ),
        "pooled_token_count": _require_int(
            state.get("pooled_token_count"), field=f"native_calibration.state[{index}].pooled_token_count"
        ),
        "token_ids": [int(value) for value in token_ids],
        "position": position,
        "mutant": mutant,
    }


def select_native_calibration(
    calibration: Mapping[str, Any],
    arm: str,
    target: Mapping[str, Any],
    native_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one arm's calibration block to the protocol target by WT bytes, not by name."""
    models = calibration.get("models")
    if not isinstance(models, Mapping) or arm not in models:
        raise ValueError(f"native calibration has no block for arm {arm!r}")
    block = models[arm]
    if not isinstance(block, Mapping):
        raise ValueError(f"native calibration block for {arm!r} must be an object")
    identity = block.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"native calibration for {arm!r} lacks an identity block")
    for key, expected in (
        ("arm", arm),
        ("batch_size", int(native_contract["batch_size"])),
        ("budget", int(native_contract["budget"])),
        ("dtype", native_contract["dtype"]),
    ):
        if identity.get(key) != expected:
            raise ValueError(
                f"native calibration identity {arm}.{key} is {identity.get(key)!r}, "
                f"expected {expected!r}"
            )
    _require_str(identity.get("likelihood_semantics"), field=f"native_calibration.{arm}.identity.likelihood_semantics")
    cohorts = block.get("targets")
    if not isinstance(cohorts, Mapping):
        raise ValueError(f"native calibration for {arm!r} lacks a targets mapping")
    matches = [
        (key, value)
        for key, value in cohorts.items()
        if isinstance(value, Mapping)
        and isinstance(value.get("states"), list)
        and value["states"]
        and isinstance(value["states"][0], Mapping)
        and value["states"][0].get("sequence_sha256") == target["wt_sha256"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"native calibration for {arm!r} binds {len(matches)} targets to the protocol WT SHA; "
            "exactly one is required"
        )
    calibration_target_key, value = matches[0]
    states = value["states"]
    if len(states) != 3:
        raise ValueError(f"native calibration {arm}/{calibration_target_key} must hold exactly 3 states")
    parsed_states = [
        _calibration_state(state, index=index, target=target) for index, state in enumerate(states)
    ]
    return {
        "tolerance_nats": float(calibration["tolerance_nats"]),
        "arm": arm,
        "calibration_target_key": str(calibration_target_key),
        "identity": dict(identity),
        "manifest": dict(block.get("manifest") or {}),
        "archive": dict(value.get("archive") or {}),
        "cohort_sha256": ((calibration.get("sources") or {}).get("cohort") or {}).get("sha256"),
        "states": parsed_states,
    }


def native_calibration_check(
    selection: Mapping[str, Any],
    *,
    score_fn: ScoreFn,
    token_record_fn: TokenRecordFn,
) -> dict[str, Any]:
    """Re-score the WT and two singles and match full native token layout and score."""
    tolerance = float(selection["tolerance_nats"])
    records = []
    for state in selection["states"]:
        sequence = state["sequence"]
        tokens = token_record_fn(sequence)
        ids = [int(value) for value in tokens["ids"]]
        packed = len(ids)
        scored = int(tokens["scored_span"][1]) - int(tokens["scored_span"][0])
        pooled = len(tokens["pooled_positions"])
        if ids != state["token_ids"]:
            raise ValueError(
                f"native calibration state {state['state']} token ids differ from the native packing"
            )
        if (packed, scored, pooled) != (
            state["packed_token_count"],
            state["scored_token_count"],
            state["pooled_token_count"],
        ):
            raise ValueError(
                f"native calibration state {state['state']} packed/scored/pooled counts "
                f"({packed}/{scored}/{pooled}) differ from the cached "
                f"({state['packed_token_count']}/{state['scored_token_count']}/"
                f"{state['pooled_token_count']})"
            )
        observed = float(score_fn(sequence))
        if not math.isfinite(observed):
            raise ValueError(f"native calibration state {state['state']} produced a non-finite score")
        delta = abs(observed - state["score_nats"])
        if delta > tolerance:
            raise ValueError(
                f"native calibration state {state['state']} disagrees with the cached value by "
                f"{delta!r} nats, beyond the fixed {tolerance} tolerance"
            )
        records.append(
            {
                "state": state["state"],
                "sequence_sha256": state["sequence_sha256"],
                "position": state["position"],
                "mutant": state["mutant"],
                "cached_nats": state["score_nats"],
                "observed_nats": observed,
                "abs_delta_nats": delta,
                "packed_token_count": packed,
                "scored_token_count": scored,
                "pooled_token_count": pooled,
            }
        )
    return {
        "tolerance_nats": tolerance,
        "arm": selection["arm"],
        "calibration_target_key": selection["calibration_target_key"],
        "identity": selection["identity"],
        "manifest": selection["manifest"],
        "archive": selection["archive"],
        "cohort_sha256": selection["cohort_sha256"],
        "entries": records,
    }



# ------------------------------------------------------------- native generation


def analyze_native_text(
    arm_name: str, raw_text: str, *, allow_lf_wrap: bool | None = None
) -> tuple[str, int, str, list[int]]:
    """Split raw native text into its residue content and its linewrap mapping.

    ProtGPT2's declared native linewrap is the only character removed, and only
    from the residue count and the canonical candidate; the raw text and the raw
    token ids are always retained. No punctuation, unknown residue or partial
    token is stripped, clipped or padded: any remaining non-AA20 character is
    reported so the attempt is rejected rather than silently cleaned.
    """
    if allow_lf_wrap is None:
        allow_lf_wrap = arm_name == "protgpt2"
    removed: list[int] = []
    kept: list[str] = []
    for index, character in enumerate(raw_text):
        if character == "\n" and allow_lf_wrap:
            removed.append(index)
        else:
            kept.append(character)
    cleaned = "".join(kept)
    offenders = "".join(sorted({character for character in cleaned if character not in AA20}))
    return cleaned, len(cleaned), offenders, removed


def conditioning_ids(arm: Any, spec: Any, prefix_residues: str) -> tuple[list[int], list[int]]:
    """The exact native prompt continuation for the WT prefix, verified not to merge."""
    tokenizer = arm.tokenizer
    prompt_ids = list(
        tokenizer(spec.prompt, add_special_tokens=spec.add_special_tokens, return_tensors=None)[
            "input_ids"
        ]
    )
    prefix_ids = list(
        tokenizer(prefix_residues, add_special_tokens=False, return_tensors=None)["input_ids"]
    )
    joined_ids = list(
        tokenizer(
            spec.prompt + prefix_residues,
            add_special_tokens=spec.add_special_tokens,
            return_tensors=None,
        )["input_ids"]
    )
    if not prefix_ids:
        raise ValueError("the WT conditioning prefix tokenised to nothing")
    if joined_ids != prompt_ids + prefix_ids:
        raise ValueError(
            "the WT conditioning prefix does not encode as an exact continuation of the "
            "native prompt; a BPE merge would move the conditioning boundary"
        )
    decoded = tokenizer.decode(
        prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if decoded != prefix_residues:
        raise ValueError("the conditioning prefix does not decode back to the declared residues")
    return joined_ids, prefix_ids


def _model_next_id(arm: Any, ids: torch.Tensor) -> int:
    with torch.no_grad():
        output = arm.model(input_ids=ids)
    logits = output.logits if hasattr(output, "logits") else output[0]
    sampled = _sample_next_id(logits[0, -1], temperature=1.0, top_k=0, top_p=1.0)
    return int(sampled.item())


def sample_attempt(
    *,
    arm_name: str,
    tokenizer: Any,
    prompt_ids: list[int],
    eos_id: int,
    prefix_residues: str,
    design_length: int,
    max_new_tokens: int,
    seed: int,
    attempt_index: int,
    next_id_fn: NextIdFn,
    target: str | None = None,
    stream_seed: int | None = None,
    stream_index: int | None = None,
    wt_sequence: str | None = None,
    device: str = "cpu",
    allow_lf_wrap: bool | None = None,
) -> dict[str, Any]:
    """One seeded native attempt, retained whatever its stop status."""
    target_suffix_aa = int(design_length) - len(prefix_residues)
    if target_suffix_aa < 1:
        raise ValueError("the design length leaves no generated suffix")
    torch.manual_seed(int(seed))
    ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    raw_ids: list[int] = []
    raw_text = ""
    cleaned = ""
    aa_count = 0
    removed: list[int] = []
    offenders = ""
    status = "budget_exhausted"
    for _ in range(int(max_new_tokens)):
        next_id = int(next_id_fn(ids))
        raw_ids.append(next_id)
        ids = torch.cat(
            [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
        )
        raw_text = tokenizer.decode(
            raw_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if next_id == int(eos_id):
            status = "early_eos"
            break
        cleaned, aa_count, offenders, removed = analyze_native_text(
            arm_name, raw_text, allow_lf_wrap=allow_lf_wrap
        )
        if offenders:
            status = "noncanonical"
            break
        if aa_count > target_suffix_aa:
            status = "overshoot"
            break
        if aa_count == target_suffix_aa:
            status = "design_length_stop"
            break
    if status not in ATTEMPT_STATUSES:  # pragma: no cover - the loop sets one of these
        raise AssertionError(f"unknown attempt status {status!r}")

    candidate = prefix_residues + cleaned if status == "design_length_stop" else None
    candidate_sha = _sequence_sha(candidate) if candidate is not None else None
    mutation_positions: list[int] | None = None
    hamming_to_wt: int | None = None
    if candidate is not None and wt_sequence is not None:
        if len(candidate) != len(wt_sequence):
            raise ValueError("an eligible candidate changed the total design length")
        mutation_positions = [
            index + 1 for index in range(len(candidate)) if candidate[index] != wt_sequence[index]
        ]
        hamming_to_wt = len(mutation_positions)
    return {
        "schema": ATTEMPT_SCHEMA,
        "target": target,
        "arm": arm_name,
        "stream_index": stream_index,
        "stream_seed": stream_seed,
        "attempt_index": int(attempt_index),
        "attempt_seed": int(seed),
        "status": status,
        "is_eligible": status == "design_length_stop",
        "stop_reason": status,
        "design_length": int(design_length),
        "max_new_tokens": int(max_new_tokens),
        "new_token_count": len(raw_ids),
        "raw_token_ids": raw_ids,
        "raw_text": raw_text,
        "raw_text_sha256": _sequence_sha_loose(raw_text),
        "new_aa_count": int(aa_count),
        "removed_lf_offsets": list(removed),
        "noncanonical_characters": offenders,
        "candidate_sequence": candidate,
        "candidate_sha256": candidate_sha,
        "candidate_length": len(candidate) if candidate is not None else None,
        "prefix_residues": prefix_residues,
        "prefix_exact": bool(candidate is not None and candidate.startswith(prefix_residues)),
        "duplicate_of": None,
        "mutation_positions_vs_wt": mutation_positions,
        "mutation_burden_vs_wt": hamming_to_wt,
        "novelty_vs_wt_hamming": hamming_to_wt,
    }


def _sequence_sha_loose(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------ native response grid


def candidate_grid(
    sequence: str,
    *,
    prefix_residues: str,
    design_length: int,
    token_record_fn: TokenRecordFn,
    score_fn: ScoreFn,
) -> dict[str, Any]:
    """Base native score plus the generated-suffix substitution ladder of one design.

    The prefix is conditioning only and is never mutated. Positions run from
    ``len(prefix) + 1`` to ``L`` inclusive; each carries the nineteen alternate
    canonical residues, with the current residue left implicit as the zero of
    that site's twenty responses. A mutant is scored from its own full sequence,
    never by adding a wild-type single-mutant delta to some other prediction.
    """
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("candidate sequence must be a non-empty string")
    offenders = sorted(set(sequence) - set(AA20))
    if offenders:
        raise ValueError(f"candidate sequence contains non-AA20 characters {offenders}")
    if len(sequence) != int(design_length):
        raise ValueError("candidate sequence length does not match the declared design length")
    if not sequence.startswith(prefix_residues):
        raise ValueError("candidate does not preserve the exact conditioning prefix")
    prefix_length = len(prefix_residues)
    positions = list(range(prefix_length + 1, int(design_length) + 1))
    if not positions:
        raise ValueError("the generated suffix has no mutable position")

    state_sequence = [sequence]
    state_position = [0]
    state_mutant_index = [-1]
    state_residue = ["-"]
    tokens = [token_record_fn(sequence)]
    scores = [_finite_score(score_fn(sequence), where="base")]
    for position in positions:
        current = sequence[position - 1]
        alternates = [residue for residue in AA20 if residue != current]
        if len(alternates) != len(AA20) - 1:
            raise AssertionError("the canonical alphabet must contain twenty residues")
        for alternate in alternates:
            mutant = sequence[: position - 1] + alternate + sequence[position:]
            if len(mutant) != len(sequence) or _hamming(mutant, sequence) != 1:
                raise ValueError("a generated substitution is not an exact single mutation")
            if mutant[position - 1] != alternate:
                raise AssertionError("the substituted residue was not written at its position")
            state_sequence.append(mutant)
            state_position.append(position)
            state_mutant_index.append(AA20.index(alternate))
            state_residue.append(alternate)
            tokens.append(token_record_fn(mutant))
            scores.append(_finite_score(score_fn(mutant), where=f"position {position}"))

    expected_states = 1 + (len(AA20) - 1) * len(positions)
    if len(scores) != expected_states:
        raise AssertionError("the substitution ladder did not produce every declared state")
    base_score = float(scores[0])
    mutant_scores = np.asarray(scores[1:], dtype=np.float64)
    deltas = mutant_scores - base_score
    if not np.isfinite(deltas).all():
        raise ValueError("a substitution delta is not finite")

    packed_counts = [len(record["ids"]) for record in tokens]
    scored_counts = [
        int(record["scored_span"][1]) - int(record["scored_span"][0]) for record in tokens
    ]
    pooled_counts = [len(record["pooled_positions"]) for record in tokens]
    token_ids: list[int] = []
    offsets = [0]
    for record in tokens:
        token_ids.extend(int(token) for token in record["ids"])
        offsets.append(len(token_ids))

    return {
        "sequence": sequence,
        "sequence_sha256": _sequence_sha(sequence),
        "prefix_residues": prefix_residues,
        "design_length": int(design_length),
        "n_positions": len(positions),
        "n_alternates": len(AA20) - 1,
        "state_count": expected_states,
        "forward_count": expected_states,
        "base_score": base_score,
        "mutant_scores": mutant_scores,
        "delta": deltas,
        "state_position": np.asarray(state_position, dtype=np.int64),
        "state_mutant_index": np.asarray(state_mutant_index, dtype=np.int64),
        "state_residue": np.asarray(state_residue, dtype="<U1"),
        "state_sequence_sha256": np.asarray(
            [_sequence_sha(item) for item in state_sequence], dtype="<U64"
        ),
        "packed_token_counts": np.asarray(packed_counts, dtype=np.int64),
        "scored_token_counts": np.asarray(scored_counts, dtype=np.int64),
        "pooled_token_counts": np.asarray(pooled_counts, dtype=np.int64),
        "token_ids": np.asarray(token_ids, dtype=np.int64),
        "token_offsets": np.asarray(offsets, dtype=np.int64),
    }


def _finite_score(value: Any, *, where: str) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"a non-finite native score was produced at {where}")
    return score


def _grid_arrays(grid: Mapping[str, Any], metadata: Any) -> dict[str, np.ndarray]:
    return {
        "base_score": np.asarray(grid["base_score"], dtype=np.float64),
        "mutant_scores": np.asarray(grid["mutant_scores"], dtype=np.float64),
        "delta": np.asarray(grid["delta"], dtype=np.float64),
        "state_position": np.asarray(grid["state_position"], dtype=np.int64),
        "state_mutant_index": np.asarray(grid["state_mutant_index"], dtype=np.int64),
        "state_residue": np.asarray(grid["state_residue"], dtype="<U1"),
        "state_sequence_sha256": np.asarray(grid["state_sequence_sha256"], dtype="<U64"),
        "packed_token_counts": np.asarray(grid["packed_token_counts"], dtype=np.int64),
        "scored_token_counts": np.asarray(grid["scored_token_counts"], dtype=np.int64),
        "pooled_token_counts": np.asarray(grid["pooled_token_counts"], dtype=np.int64),
        "token_ids": np.asarray(grid["token_ids"], dtype=np.int64),
        "token_offsets": np.asarray(grid["token_offsets"], dtype=np.int64),
        "state_count": np.asarray(int(grid["state_count"]), dtype=np.int64),
        "n_positions": np.asarray(int(grid["n_positions"]), dtype=np.int64),
        "n_alternates": np.asarray(int(grid["n_alternates"]), dtype=np.int64),
        "design_length": np.asarray(int(grid["design_length"]), dtype=np.int64),
        "prefix_length": np.asarray(len(str(grid["prefix_residues"])), dtype=np.int64),
        "candidate_sequence_sha256": np.asarray(grid["sequence_sha256"], dtype="<U64"),
        "metadata": np.asarray(
            json.dumps(metadata, allow_nan=False, sort_keys=True), dtype="<U4096"
        ),
    }


def write_immutable_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write one grid once. An existing file is refused rather than replaced."""
    path = Path(path)
    if path.exists():
        raise ValueError(f"refusing to overwrite the existing grid {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


# ------------------------------------------------------------------ orchestration


def _checkpoint_record(arm: Any) -> dict[str, Any]:
    checkpoint = Path(arm.spec.path)
    metadata: dict[str, str] = {}
    tensors: list[dict[str, Any]] = []
    if checkpoint.is_dir():
        for item in sorted(checkpoint.iterdir()):
            if not item.is_file():
                continue
            if item.suffix == ".json" or item.name in ("tokenizer.model", "tokenizer.json"):
                metadata[str(item.relative_to(checkpoint))] = sha256_file(item)
            elif item.suffix in (".safetensors", ".bin", ".pt"):
                tensors.append({"name": item.name, "bytes": item.stat().st_size})
    return {
        "arm": arm.name,
        "resolved_path": str(checkpoint.resolve()),
        "declared_dtype": str(arm.dtype),
        "attn_implementation": arm.attn_implementation,
        "metadata_sha256": metadata,
        "tensor_files": tensors,
        "weight_byte_proof": None,
    }


def require_checkpoint_binding(checkpoint: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    """Bind the actually resolved handle's metadata hashes and tensor sizes to the identity."""
    actual_metadata = checkpoint.get("metadata_sha256") or {}
    for name, digest in (identity.get("checkpoint_metadata_sha256") or {}).items():
        if actual_metadata.get(name) != digest:
            raise ValueError(
                f"checkpoint metadata {name!r} on the resolved handle does not match the "
                "calibration identity"
            )
    actual_tensors = {
        (entry["name"], entry["bytes"]) for entry in (checkpoint.get("tensor_files") or [])
    }
    for entry in identity.get("checkpoint_tensor_files") or []:
        if (entry["name"], entry["bytes"]) not in actual_tensors:
            raise ValueError(
                f"checkpoint tensor {entry['name']!r} ({entry['bytes']} bytes) on the resolved "
                "handle does not match the calibration identity"
            )


def _observed_parameter_dtypes(arm: Any) -> list[str]:
    return sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in arm.model.parameters()
            if parameter.is_floating_point()
        }
    )


def analysis_availability(unique_candidates: int) -> tuple[bool, str | None]:
    """A successful collection with no eligible design is unavailable, not a negative."""
    if unique_candidates > 0:
        return True, None
    return False, (
        "no eligible design-length candidate was retained; this is an unavailable analysis, "
        "not a biological negative"
    )


def group_eligible_candidates(
    attempts: list[dict[str, Any]],
    *,
    arm: str,
    domain: str,
    stream_seed: int,
    stream_index: int,
) -> list[dict[str, Any]]:
    """Collapse eligible attempts to their distinct candidate sequences, in first-seen order.

    The returned order is the grid evaluation order, so a duplicate sequence is
    forwarded exactly once per job even though every duplicate attempt stays in
    the ledger and in the denominator.
    """
    unique: list[dict[str, Any]] = []
    by_sha: dict[str, dict[str, Any]] = {}
    for row in attempts:
        if not row["is_eligible"]:
            continue
        candidate_sha = row["candidate_sha256"]
        record = by_sha.get(candidate_sha)
        if record is None:
            record = {
                "sha256": candidate_sha,
                "sequence": row["candidate_sequence"],
                "domain": domain,
                "arm": arm,
                "stream_seed": int(stream_seed),
                "stream_index": int(stream_index),
                "attempt_indexes": [],
                "repeat_count": 0,
                "mutation_burden_vs_wt": row["mutation_burden_vs_wt"],
                "novelty_vs_wt_hamming": row["novelty_vs_wt_hamming"],
            }
            by_sha[candidate_sha] = record
            unique.append(record)
        record["attempt_indexes"].append(row["attempt_index"])
        record["repeat_count"] += 1
    return unique


def run_job(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol)
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha = sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes.decode("utf-8"), parse_constant=_reject_constant)
    resolved = validate_protocol(protocol, args.arm, args.domain, int(args.stream_index))
    target = resolved["target"]
    sampling = resolved["sampling"]
    native_contract = resolved["native_contract"]

    inputs = resolve_inputs(protocol, Path(args.root))
    calibration_payload = load_native_calibration(Path(inputs["native_calibration"]["resolved"]))
    calibration_selection = select_native_calibration(
        calibration_payload, args.arm, target, native_contract
    )

    device = str(args.device)
    if not device.startswith("cuda"):
        raise SystemExit(
            "the native licensed collection requires a CUDA device; a CPU device would not "
            "measure the declared float32 kernel"
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available on this host")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    out = prepare_output_dir(Path(args.out))

    stage46 = _load_stage46(ROOT)
    singles = _load_stability_singles(ROOT)
    budget = int(SETTLED_NATIVE_CONTRACT["budget"])
    arm: Any = load_readout_arm(args.arm, stage46, device=device, dtype="float32")
    arm.model.eval()
    arm.model.requires_grad_(False)
    observed_dtypes = _observed_parameter_dtypes(arm)
    if observed_dtypes != ["float32"]:
        raise ValueError(
            f"{args.arm}: the loaded handle carries dtypes {observed_dtypes}; this pilot "
            "requires one float32 handle for both sampling and scoring"
        )
    ch.require_position_budget(arm.model.config, arm=args.arm)

    code_digests: dict[str, str] = {}
    for source in _code_paths(ROOT):
        if not source.is_file():
            raise ValueError(f"a shared kernel source is missing: {source}")
        code_digests[str(source.relative_to(ROOT))] = sha256_file(source)
    checkpoint = _checkpoint_record(arm)
    require_checkpoint_binding(checkpoint, calibration_selection["identity"])

    def score_fn(sequence: str) -> float:
        return float(singles.likelihood_score(arm, sequence, stage46, budget))

    def token_record_fn(sequence: str) -> dict:
        return singles.token_record(arm, sequence, budget)

    calibration = native_calibration_check(
        calibration_selection, score_fn=score_fn, token_record_fn=token_record_fn
    )

    spec = require_admitted(args.arm)
    eos_id = _eos_id(spec, arm.tokenizer)
    wt_prefix = target["wt_sequence"][:PREFIX_RESIDUES]
    prompt_ids, _prefix_ids = conditioning_ids(arm, spec, wt_prefix)
    design_length = int(target["L"])

    attempts: list[dict[str, Any]] = []
    eligible = 0
    attempt_index = 0
    while (
        eligible < int(sampling["max_eligible_attempts"])
        and attempt_index < int(sampling["hard_attempt_cap"])
    ):
        seed = int(resolved["stream_seed"]) + attempt_index
        row = sample_attempt(
            arm_name=args.arm,
            tokenizer=arm.tokenizer,
            prompt_ids=prompt_ids,
            eos_id=eos_id,
            prefix_residues=wt_prefix,
            design_length=design_length,
            max_new_tokens=int(sampling["max_new_tokens"]),
            seed=seed,
            attempt_index=attempt_index,
            next_id_fn=lambda ids, handle=arm: _model_next_id(handle, ids),
            target=target["domain"],
            stream_seed=int(resolved["stream_seed"]),
            stream_index=int(resolved["stream_index"]),
            wt_sequence=target["wt_sequence"],
            device=device,
        )
        attempts.append(row)
        attempt_index += 1
        if row["is_eligible"]:
            eligible += 1
    stop_reason = (
        "eligible_cap"
        if eligible >= int(sampling["max_eligible_attempts"])
        else "hard_attempt_cap"
    )

    first_seen: dict[str, int] = {}
    for row in attempts:
        if not row["is_eligible"]:
            continue
        candidate_sha = row["candidate_sha256"]
        if candidate_sha in first_seen:
            row["duplicate_of"] = first_seen[candidate_sha]
        else:
            first_seen[candidate_sha] = row["attempt_index"]

    ge.write_immutable(out / "attempts.jsonl", ge.jsonl_bytes(attempts))

    unique_candidates = group_eligible_candidates(
        attempts,
        arm=args.arm,
        domain=target["domain"],
        stream_seed=int(resolved["stream_seed"]),
        stream_index=int(resolved["stream_index"]),
    )

    grids_dir = out / "grids"
    manifest_entries: list[dict[str, Any]] = []
    total_forwards = 0
    for record in unique_candidates:
        candidate_sha = record["sha256"]
        grid = candidate_grid(
            record["sequence"],
            prefix_residues=wt_prefix,
            design_length=design_length,
            token_record_fn=token_record_fn,
            score_fn=score_fn,
        )
        filename = f"{candidate_sha}.npz"
        metadata = {
            "candidate_sha256": candidate_sha,
            "domain": target["domain"],
            "group": target["group"],
            "arm": args.arm,
            "stream_index": int(resolved["stream_index"]),
            "stream_seed": int(resolved["stream_seed"]),
            "design_length": design_length,
            "prefix_residues": wt_prefix,
            "n_positions": grid["n_positions"],
            "n_alternates": grid["n_alternates"],
            "state_count": grid["state_count"],
            "raw_text_sha256": next(
                row["raw_text_sha256"]
                for row in attempts
                if row["is_eligible"] and row["candidate_sha256"] == candidate_sha
            ),
        }
        write_immutable_npz(grids_dir / filename, _grid_arrays(grid, metadata))
        file_sha = sha256_file(grids_dir / filename)
        entry = {
            "sha256": candidate_sha,
            "file": f"grids/{filename}",
            "file_sha256": file_sha,
            "states": int(grid["state_count"]),
            "n_positions": int(grid["n_positions"]),
            "n_alternates": int(grid["n_alternates"]),
            "base_score": float(grid["base_score"]),
            "forwards": int(grid["forward_count"]),
            "repeat_count": record["repeat_count"],
            "attempt_indexes": list(record["attempt_indexes"]),
        }
        manifest_entries.append(entry)
        record["grid_file"] = entry["file"]
        total_forwards += int(grid["forward_count"])

    index_payload = {
        "schema": INDEX_SCHEMA,
        "target": {
            "domain": target["domain"],
            "wt_sha256": target["wt_sha256"],
            "L": design_length,
            "group": target["group"],
        },
        "references": {
            target["domain"]: {
                "wt_sha256": target["wt_sha256"],
                "wt_sequence_sha256": target["wt_sha256"],
                "L": design_length,
                "group": target["group"],
                "arm": args.arm,
                "stream_seed": int(resolved["stream_seed"]),
            }
        },
        "candidates": {
            record["sha256"]: {
                "attempt_indexes": list(record["attempt_indexes"]),
                "repeat_count": record["repeat_count"],
                "arm": args.arm,
                "domain": target["domain"],
                "stream_index": int(resolved["stream_index"]),
                "stream_seed": int(resolved["stream_seed"]),
                "grid_file": record.get("grid_file"),
                "mutation_burden_vs_wt": record["mutation_burden_vs_wt"],
            }
            for record in unique_candidates
        },
    }
    _write_immutable_json(out / "candidate_reference_index.json", index_payload)

    manifest_payload = {
        "schema": MANIFEST_SCHEMA,
        "arm": args.arm,
        "domain": target["domain"],
        "stream_index": int(resolved["stream_index"]),
        "stream_seed": int(resolved["stream_seed"]),
        "unique_candidates": len(manifest_entries),
        "total_forwards": total_forwards,
        "grids": manifest_entries,
    }
    _write_immutable_json(out / "grid_manifest.json", manifest_payload)

    rejection_counts: dict[str, int] = {}
    for row in attempts:
        if not row["is_eligible"]:
            rejection_counts[row["status"]] = rejection_counts.get(row["status"], 0) + 1
    analysis_available, unavailable_reason = analysis_availability(len(unique_candidates))
    completion = {
        "schema": COLLECTION_SCHEMA,
        "status": "complete",
        "analysis_available": analysis_available,
        "analysis_unavailable_reason": unavailable_reason,
        "target": {
            "domain": target["domain"],
            "wt_sha256": target["wt_sha256"],
            "L": design_length,
            "group": target["group"],
            "prefix_residues": wt_prefix,
            "prefix_length": PREFIX_RESIDUES,
        },
        "arm": args.arm,
        "stream_index": int(resolved["stream_index"]),
        "stream_seed": int(resolved["stream_seed"]),
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha},
        "source": {"script": str(Path(__file__).relative_to(ROOT)), "code_sha256": code_digests},
        "checkpoint": checkpoint,
        "inputs": inputs,
        "native_contract": resolved["native_contract"],
        "sampling": sampling,
        "calibration": calibration,
        "counts": {
            "attempts": len(attempts),
            "eligible_attempts": eligible,
            "rejected_attempts": len(attempts) - eligible,
            "rejection_by_status": rejection_counts,
            "unique_candidates": len(unique_candidates),
            "candidate_states": sum(entry["states"] for entry in manifest_entries),
            "candidate_forwards": total_forwards,
            "calibration_forwards": len(calibration["entries"]),
        },
        "caps": {
            "max_eligible_attempts": int(sampling["max_eligible_attempts"]),
            "hard_attempt_cap": int(sampling["hard_attempt_cap"]),
            "max_new_tokens": int(sampling["max_new_tokens"]),
        },
        "stop_reason": stop_reason,
        "eligibility": {
            "eligible_attempts": eligible,
            "status": "eligible_retained" if eligible else "no_eligible_attempts",
        },
        "artifacts": {
            "attempts": {
                "file": "attempts.jsonl",
                "sha256": sha256_file(out / "attempts.jsonl"),
                "records": len(attempts),
            },
            "index": {
                "file": "candidate_reference_index.json",
                "sha256": sha256_file(out / "candidate_reference_index.json"),
            },
            "grid_manifest": {
                "file": "grid_manifest.json",
                "sha256": sha256_file(out / "grid_manifest.json"),
            },
        },
        "producer": {
            "script": "scripts/capability/generation/collect_licensed_prediction_pilot.py",
            "invocation": {
                "arm": args.arm,
                "domain": args.domain,
                "stream_index": int(args.stream_index),
                "root": str(Path(args.root)),
            },
        },
        "updated_utc": _dtos(),
    }
    _write_immutable_json(out / "licensed_pilot_collection.json", completion)
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--stream-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    out = Path(args.out)
    try:
        run_job(args)
    except Exception as error:
        if out.exists():
            try:
                write_json(
                    out / "failure_licensed_pilot_collection.json",
                    {
                        "schema": FAILURE_SCHEMA,
                        "status": "failed",
                        "arm": args.arm,
                        "domain": args.domain,
                        "stream_index": int(args.stream_index),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "updated_utc": _dtos(),
                    },
                )
            except Exception:  # pragma: no cover - the original failure is re-raised
                pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())