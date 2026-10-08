"""ProLLaMA Stage 1 against Stage 2: does better prediction come with better generation?

The question
============

Stage 2 is Stage 1 plus further training, and it is one of only two checkpoints
in the whole panel whose single-substitution stability increment resolves
positive (+0.00862 kcal^2/mol^2). So it predicts mutation effects better. The
question this set exists to answer is whether that improvement shows up in what
the model *generates*: are Stage 2's products better proteins, by any property
that can be read off a sequence or a predicted structure, than Stage 1's?

Why this comparison is unusually clean, and what makes it fragile
=================================================================

The two checkpoints differ only by further training, so a difference between
them is attributable to that training rather than to architecture, tokeniser or
corpus. The whole inference therefore rests on **decoding symmetry**: the same
prompt, the same sampling configuration, the same seed rule and the same attempt
count. :func:`verify_symmetry` refuses a pair that differs in any of them, and it
is checked against the campaign manifest's own cell declarations rather than
asserted, because an asymmetry here would be invisible in the result.

The replication campaign already froze that symmetry: both cells are declared at
prompt ``Seq=<``, temperature 0.85, top-k 50, top-p 0.95, repetition penalty
1.0, 400 new tokens, batch 8, bfloat16, ``campaign_seed + batch_index``, 800
attempts, over the same two campaign seeds. This module therefore **freezes and
documents the retained sets** instead of generating new ones: regenerating would
produce a third decoding stream that is not comparable with the three the
manuscript's stage contrast already reads.

What this set is for
====================

Two later analyses consume it: single-mutation scanning and double-mutant
interaction information on generated sequences. Neither is implemented here. The
deliverable is a frozen, documented, stably keyed sequence set per stage, with
its decoding parameters and seeds recorded beside it, plus the predicted-structure
and sequence-property readouts this experiment needs in its own right.

What cannot be compared
=======================

Stage 2 carries an instruction interface (``[Generate by superfamily]``) that
Stage 1 does not have at all. A conditioned stage-1 counterpart therefore does
not exist and cannot be constructed; the conditional comparison is impossible by
construction, not merely unmeasured. And nothing here measures stability: pLDDT
and pTM are predicted *confidence*, no stability predictor is applied to a
generated sequence, and neither quantity is read as folding or function.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.amino_acids import AA20
from ..core.protein_properties import AA_CLASSES, CLASS_NAMES
from .generative_control import mean_hydropathy

_AA20: frozenset[str] = frozenset(AA20)

SCHEMA_VERSION = "d1_prollama_stage_pair_v1"

STAGES: dict[str, str] = {
    "stage_1": "prollama-stage-1",
    "stage_2": "prollama",
}

#: Every manifest field the two cells must agree on. Each one, if it differed,
#: would make the stage difference partly a decoding difference.
SYMMETRY_FIELDS: tuple[str, ...] = (
    "prompt",
    "condition",
    "attempts",
    "max_new_tokens",
    "effective_max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "batch_size",
    "dtype",
    "use_cache",
    "add_special_tokens",
    "seed_rule",
)

#: The residue band the frozen set is drawn from. The lower bound keeps a
#: sequence long enough for a mutation scan to have positions to scan; the upper
#: bound is a folding-cost bound and is declared, not tuned.
MIN_LENGTH: int = 40
MAX_LENGTH: int = 400

#: Sequences per stage per stream. Declared before any property was read.
PER_STREAM: int = 300

#: The selection seed. One value reproduces the whole frozen set.
SELECTION_SEED: int = 20261008

CEILING: dict[str, str] = {
    "conditional_comparison_is_impossible": (
        "Stage 2 has an instruction interface Stage 1 does not have, so a conditioned "
        "stage-1 counterpart cannot be constructed. The conditional arm is impossible "
        "by construction, not unmeasured"
    ),
    "confidence_is_not_stability": (
        "pLDDT and pTM are predicted confidence. No stability predictor and no "
        "stability measurement is applied to a generated sequence here, and predicted "
        "confidence is never reported as stability"
    ),
    "predicted_structure_is_not_function": (
        "a predicted structure cannot demonstrate folding, function or competence; it "
        "is a calibrated instrument reading on a sequence"
    ),
    "selection_is_outcome_blind": (
        "the frozen set is selected on identity, length and canonical-residue validity "
        "only, under one declared seed. No recognition, confidence or property value "
        "enters the selection"
    ),
    "streams_are_not_lineages": (
        "campaign seed streams resample from fixed checkpoints and are not independent "
        "training lineages; the paired difference is reported per stream and summarised "
        "at the stream as the unit"
    ),
}

#: Fields of an attempt record the selection is permitted to read. Enforced, so a
#: later edit cannot quietly make the frozen set outcome-dependent.
SELECTION_FIELDS: frozenset[str] = frozenset(
    {"id", "sequence", "sequence_sha256", "length", "valid_aa20"}
)


def verify_symmetry(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Refuse a stage pair whose declared decoding is not identical.

    ``cells`` maps a stage label to that stage's manifest cell declaration.
    """

    missing = sorted(set(STAGES) - set(cells))
    if missing:
        raise ValueError(f"no manifest cell was supplied for stages {missing}")
    observed: dict[str, dict[str, Any]] = {}
    for field in SYMMETRY_FIELDS:
        values = {}
        for stage, cell in cells.items():
            if field not in cell:
                raise ValueError(
                    f"{stage}: the manifest cell declares no {field!r}; decoding "
                    "symmetry cannot be verified and the comparison is refused"
                )
            values[stage] = cell[field]
        observed[field] = values
    asymmetric = {
        field: values
        for field, values in observed.items()
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1
    }
    if asymmetric:
        raise ValueError(
            "the two ProLLaMA stages are not declared at identical decoding, so a "
            "difference between them is partly a decoding difference: "
            + "; ".join(f"{field}={values}" for field, values in sorted(asymmetric.items()))
        )
    return {
        "fields": list(SYMMETRY_FIELDS),
        "declared": {field: values[next(iter(values))] for field, values in observed.items()},
        "symmetric": True,
        "arms": dict(sorted(STAGES.items())),
    }


def _selection_view(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in SELECTION_FIELDS if field in row}


def select_stream(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    stream: str,
    per_stream: int = PER_STREAM,
    min_length: int = MIN_LENGTH,
    max_length: int = MAX_LENGTH,
    seed: int = SELECTION_SEED,
) -> dict[str, Any]:
    """One stream's contribution to one stage's frozen set.

    Exact duplicates collapse to one representative before the draw: folding the
    same sequence twice adds nothing, and duplication is reported as its own
    quantity rather than inflating the set. The draw is a seeded permutation of
    the canonically sorted eligible identifiers.
    """

    if per_stream < 1:
        raise ValueError("a frozen set needs at least one sequence per stream")
    if not min_length <= max_length:
        raise ValueError("the length band is empty")
    views = [_selection_view(row) for row in rows]
    eligible: dict[str, dict[str, Any]] = {}
    rejected = Counter()
    for view in views:
        sequence = view.get("sequence") or ""
        if not sequence:
            rejected["empty_attempt"] += 1
            continue
        # Canonicality is measured here rather than read off the ledger's flag. This
        # function decides what enters a frozen artefact that three later analyses
        # consume, and a flag it merely trusted would put a non-residue character
        # into all three.
        declared = view.get("valid_aa20")
        canonical = set(sequence) <= _AA20
        if declared is not None and bool(declared) != canonical:
            raise ValueError(
                f"{view['id']}: the ledger declares valid_aa20={declared!r} and the "
                "sequence says otherwise"
            )
        if not canonical:
            rejected["non_canonical_residue"] += 1
            continue
        length = len(sequence)
        if not min_length <= length <= max_length:
            rejected["outside_length_band"] += 1
            continue
        digest = view.get("sequence_sha256") or hashlib.sha256(sequence.encode()).hexdigest()
        if digest in eligible:
            rejected["exact_duplicate_of_a_kept_sequence"] += 1
            continue
        eligible[digest] = {"id": view["id"], "sequence": sequence, "sequence_sha256": digest}
    keys = sorted(eligible)
    offset = int.from_bytes(hashlib.sha256(f"{stage}|{stream}".encode()).digest()[:4], "big")
    rng = np.random.default_rng((int(seed) + offset) % (2**31 - 1))
    order = rng.permutation(len(keys)) if keys else np.asarray([], dtype=int)
    taken = [eligible[keys[int(index)]] for index in order[:per_stream]]
    return {
        "stage": stage,
        "stream": stream,
        "arm": STAGES[stage],
        "n_attempts": len(rows),
        "n_eligible": len(keys),
        "n_selected": len(taken),
        "shortfall": max(0, per_stream - len(taken)),
        "shortfall_reason": (
            None
            if len(taken) >= per_stream
            else (
                f"{len(keys)} distinct canonical sequences inside "
                f"[{min_length}, {max_length}] residues cannot supply {per_stream}; the "
                "realised count is reported and the band is never widened afterwards"
            )
        ),
        "rejected": dict(sorted(rejected.items())),
        "selection_seed": int(seed),
        "length_band_residues": [int(min_length), int(max_length)],
        "records": taken,
    }


def sequence_properties(sequence: str) -> dict[str, Any]:
    """Sequence-level properties that need no model and no structure.

    These are descriptive covariates of a generated product -- composition
    entropy, the longest single-residue run, the charged and hydrophobic shares,
    mean Kyte-Doolittle hydropathy -- and not a stability prediction.
    """

    if not sequence:
        raise ValueError("a property needs a non-empty sequence")
    counts = Counter(sequence)
    total = len(sequence)
    shares = np.asarray([counts.get(residue, 0) / total for residue in AA20], dtype=float)
    nonzero = shares[shares > 0]
    longest = 1
    run = 1
    for index in range(1, total):
        run = run + 1 if sequence[index] == sequence[index - 1] else 1
        longest = max(longest, run)
    class_shares = {
        name: sum(counts.get(residue, 0) for residue in AA_CLASSES[name]) / total
        for name in CLASS_NAMES
    }
    return {
        "length": total,
        "composition_entropy_nats": float(-(nonzero * np.log(nonzero)).sum()),
        "longest_single_residue_run": int(longest),
        "mean_kyte_doolittle_hydropathy": float(mean_hydropathy(sequence)),
        "residue_class_shares": {name: float(value) for name, value in sorted(class_shares.items())},
        "distinct_residues": len(counts),
    }


def freeze(
    streams: Sequence[Mapping[str, Any]],
    *,
    symmetry: Mapping[str, Any],
    per_stream: int = PER_STREAM,
) -> dict[str, Any]:
    """Assemble the per-stage frozen sets into one keyed artefact.

    The key is the original attempt identifier, which is unique across the whole
    generation programme, so a downstream analysis can always join a result back
    to the attempt, the cell and the stream that produced it. ``stage`` and
    ``stream`` travel on every record so neither has to be re-derived.
    """

    by_stage: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
    per_stream_record: list[dict[str, Any]] = []
    for block in streams:
        stage = block["stage"]
        if stage not in by_stage:
            raise ValueError(f"unknown stage {stage!r}; declared: {sorted(STAGES)}")
        for record in block["records"]:
            by_stage[stage].append(
                {
                    "id": record["id"],
                    "stage": stage,
                    "arm": block["arm"],
                    "stream": block["stream"],
                    "sequence": record["sequence"],
                    "sequence_sha256": record["sequence_sha256"],
                    "length": len(record["sequence"]),
                    "properties": sequence_properties(record["sequence"]),
                }
            )
        per_stream_record.append({key: value for key, value in block.items() if key != "records"})
    empty = sorted(stage for stage, records in by_stage.items() if not records)
    if empty:
        raise ValueError(f"the frozen set is empty for stages {empty}")
    identifiers = [record["id"] for records in by_stage.values() for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("the frozen set carries duplicate attempt identifiers")
    return {
        "schema_version": SCHEMA_VERSION,
        "keying": {
            "primary_key": "id",
            "description": (
                "the original generation attempt identifier, unique across the whole "
                "generation programme; every record also carries stage, arm, stream and "
                "sequence_sha256"
            ),
            "stages": dict(sorted(STAGES.items())),
            "records_file": "stage_pair_sequences.jsonl",
            "structures_key": "id",
        },
        "decoding": dict(symmetry),
        "per_stream": per_stream_record,
        "per_stage": {
            stage: {
                "arm": STAGES[stage],
                "n_sequences": len(records),
                "n_streams": len({record["stream"] for record in records}),
                "length_residues": {
                    "mean": float(np.mean([record["length"] for record in records])),
                    "median": float(np.median([record["length"] for record in records])),
                    "min": int(min(record["length"] for record in records)),
                    "max": int(max(record["length"] for record in records)),
                },
            }
            for stage, records in sorted(by_stage.items())
        },
        "per_stream_target": int(per_stream),
        "n_sequences": len(identifiers),
        "ceiling": dict(CEILING),
        "records": [record for stage in sorted(by_stage) for record in by_stage[stage]],
    }


def paired_property_contrast(
    records: Sequence[Mapping[str, Any]], *, field: str
) -> dict[str, Any]:
    """Stage 2 minus Stage 1 on one per-sequence quantity, at the stream as the unit.

    Per-stream means first, then the difference, then an equal-weight summary over
    streams. The sequences are not paired one to one -- they are different
    samples from two checkpoints -- so nothing here pretends to a paired test at
    the sequence level.
    """

    values: dict[str, dict[str, list[float]]] = {}
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        values.setdefault(record["stream"], {}).setdefault(record["stage"], []).append(float(value))
    per_stream: dict[str, Any] = {}
    for stream, block in sorted(values.items()):
        if set(block) != set(STAGES):
            per_stream[stream] = {
                "difference": None,
                "reason": f"stream carries only {sorted(block)}",
            }
            continue
        means = {stage: float(np.mean(block[stage])) for stage in sorted(block)}
        per_stream[stream] = {
            "means": means,
            "n": {stage: len(block[stage]) for stage in sorted(block)},
            "difference": means["stage_2"] - means["stage_1"],
        }
    usable = [
        block["difference"] for block in per_stream.values() if block.get("difference") is not None
    ]
    return {
        "field": field,
        "per_stream": per_stream,
        "n_streams": len(usable),
        "mean_difference": float(np.mean(usable)) if usable else None,
        "direction_replicated": (
            bool(all(value > 0 for value in usable) or all(value < 0 for value in usable))
            if len(usable) >= 2
            else None
        ),
        "unit": "the campaign stream",
        "note": (
            "Stage 2 minus Stage 1. Sequences are independent samples from two "
            "checkpoints, not matched pairs"
        ),
    }


def normalise_ledger(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reduce either retained attempt-ledger shape to the fields selection may read.

    Two shapes exist. The replication campaign's ``annotated_attempts.jsonl``
    carries ``id``/``sequence`` directly. The 2026-09-05 generation-and-control
    build cells carry ``attempt_id`` and a ``sequences`` map of cohort to string,
    of which ``generated`` is the model's own product; the other entries of that
    map are the matched controls and are deliberately not read here. Normalising
    rather than branching at the call site keeps one definition of what a
    selectable attempt is.
    """

    normalised: list[dict[str, Any]] = []
    for row in rows:
        if "sequence" in row and "id" in row:
            sequence = row.get("sequence") or ""
            identifier = str(row["id"])
        elif "attempt_id" in row and isinstance(row.get("sequences"), Mapping):
            sequence = row["sequences"].get("generated") or ""
            identifier = str(row["attempt_id"])
        else:
            raise ValueError(
                f"a ledger record carries neither id/sequence nor attempt_id/sequences: "
                f"{sorted(row)[:8]}"
            )
        digest = row.get("sequence_sha256") or hashlib.sha256(sequence.encode()).hexdigest()
        # Canonicality is measured on the residues, not taken from a flag that one of
        # the two retained shapes does not carry: a ledger without the flag must not
        # be treated as if every attempt passed.
        canonical = bool(sequence) and set(sequence) <= _AA20
        declared = row.get("valid_aa20")
        if declared is not None and bool(declared) != canonical:
            raise ValueError(
                f"{identifier}: the ledger declares valid_aa20={declared!r} and the "
                "sequence says otherwise"
            )
        normalised.append(
            {
                "id": identifier,
                "sequence": sequence,
                "sequence_sha256": digest,
                "length": int(row.get("length") or len(sequence)),
                "valid_aa20": canonical,
            }
        )
    if not normalised:
        raise ValueError("the ledger carries no record")
    return normalised


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"{path} carries no record")
    return rows
