#!/usr/bin/env python3
"""Limited synthetic numerical diagnostic for Galactica-30B BF16 packing.

This is not an interface gate, not a DMS score, and not a change to protocol v1.
It reuses the native Galactica loader and renderer, then independently packs
fixed synthetic batches so width and precision can be recorded. Observations are
kept in full; nothing here writes PASS, acquired, or retrieval-bounded.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.models import galactica_fitness as G  # noqa: E402
from src.capability.core.arms import AA20  # noqa: E402
from src.capability.core.io import write_json  # noqa: E402

KIND = "numerical_diagnostic_not_interface_gate_or_dms_score"
OUTPUT_NAME = "galactica_numerics_diagnostic.json"
ALLOWED_ARM = "galactica-30b"
SHORT_SEQUENCE = "MKT"
LONG_SEQUENCE = AA20
SHORT_WIDTH = 5
LONG_WIDTH = 22
SHORT_TARGETS = 3
LONG_TARGETS = 20
REPETITIONS = 3
LOAD_DTYPE = "bfloat16"
PUBLIC_BATCH_SIZE = 2

CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "single_short",
        "sequences": (SHORT_SEQUENCE,),
        "width": SHORT_WIDTH,
        "forced_width": False,
        "public_scorer": True,
    },
    {
        "name": "single_long",
        "sequences": (LONG_SEQUENCE,),
        "width": LONG_WIDTH,
        "forced_width": False,
        "public_scorer": True,
    },
    {
        "name": "mixed",
        "sequences": (SHORT_SEQUENCE, LONG_SEQUENCE),
        "width": LONG_WIDTH,
        "forced_width": False,
        "public_scorer": True,
    },
    {
        "name": "single_short_padded",
        "sequences": (SHORT_SEQUENCE,),
        "width": LONG_WIDTH,
        "forced_width": True,
        "public_scorer": False,
    },
    {
        "name": "duplicate_short",
        "sequences": (SHORT_SEQUENCE, SHORT_SEQUENCE),
        "width": SHORT_WIDTH,
        "forced_width": False,
        "public_scorer": True,
    },
    {
        "name": "duplicate_short_padded",
        "sequences": (SHORT_SEQUENCE, SHORT_SEQUENCE),
        "width": LONG_WIDTH,
        "forced_width": True,
        "public_scorer": False,
    },
    {
        "name": "duplicate_long",
        "sequences": (LONG_SEQUENCE, LONG_SEQUENCE),
        "width": LONG_WIDTH,
        "forced_width": False,
        "public_scorer": True,
    },
)

# Row-wise contrasts. Denominator is residue targets (3 or 20), never pad width.
COMPARISONS: tuple[tuple[str, int, str, int, int], ...] = (
    ("mixed", 0, "single_short", 0, SHORT_TARGETS),
    ("mixed", 1, "single_long", 0, LONG_TARGETS),
    ("single_short_padded", 0, "single_short", 0, SHORT_TARGETS),
    ("duplicate_short", 0, "single_short", 0, SHORT_TARGETS),
    ("duplicate_short", 1, "single_short", 0, SHORT_TARGETS),
    ("duplicate_short_padded", 0, "single_short", 0, SHORT_TARGETS),
    ("duplicate_short_padded", 1, "single_short", 0, SHORT_TARGETS),
    ("duplicate_short_padded", 0, "duplicate_short", 0, SHORT_TARGETS),
    ("duplicate_short_padded", 1, "duplicate_short", 1, SHORT_TARGETS),
    ("mixed", 0, "single_short_padded", 0, SHORT_TARGETS),
    ("duplicate_short_padded", 0, "single_short_padded", 0, SHORT_TARGETS),
    ("duplicate_short_padded", 1, "single_short_padded", 0, SHORT_TARGETS),
    ("mixed", 0, "duplicate_short_padded", 0, SHORT_TARGETS),
    ("duplicate_long", 0, "single_long", 0, LONG_TARGETS),
    ("duplicate_long", 1, "single_long", 0, LONG_TARGETS),
)


@dataclass
class PackedBatch:
    """One independently packed synthetic batch at an explicit tensor width."""

    sequences: tuple[str, ...]
    width: int
    forced_width: bool
    ids: Any
    mask: Any
    labels: Any
    scored_positions: tuple[tuple[int, ...], ...]
    n_targets: tuple[int, ...]
    natural_widths: tuple[int, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any, *, what: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} is not finite: {value!r}")
    return number


def _require_arm(arm: str) -> str:
    if arm != ALLOWED_ARM:
        raise ValueError(f"this diagnostic accepts only {ALLOWED_ARM}, got {arm!r}")
    return arm


def _require_facts(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            "recorded facts must be a non-empty mapping of observed loader facts; "
            f"got {value!r}"
        )
    return dict(value)


def _runtime_versions(torch: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "cuda_compiled": getattr(torch.version, "cuda", None),
    }
    try:
        import transformers

        payload["transformers"] = transformers.__version__
    except ImportError:
        payload["transformers"] = None
    return payload


def _matmul_policy(torch: Any) -> dict[str, Any]:
    policy: dict[str, Any] = {}
    cuda_matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if cuda_matmul is not None and hasattr(cuda_matmul, "allow_tf32"):
        policy["cuda_matmul_allow_tf32"] = bool(cuda_matmul.allow_tf32)
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None and hasattr(cudnn, "allow_tf32"):
        policy["cudnn_allow_tf32"] = bool(cudnn.allow_tf32)
    getter = getattr(torch, "get_float32_matmul_precision", None)
    if getter is not None:
        policy["float32_matmul_precision"] = str(getter())
    return policy


def _set_fp32_exec_backends(torch: Any) -> None:
    cuda_matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if cuda_matmul is not None and hasattr(cuda_matmul, "allow_tf32"):
        cuda_matmul.allow_tf32 = False
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None and hasattr(cudnn, "allow_tf32"):
        cudnn.allow_tf32 = False
    setter = getattr(torch, "set_float32_matmul_precision", None)
    if setter is not None:
        setter("highest")


def _restore_backends(torch: Any, previous: Mapping[str, Any]) -> None:
    cuda_matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if (
        cuda_matmul is not None
        and hasattr(cuda_matmul, "allow_tf32")
        and "cuda_matmul_allow_tf32" in previous
    ):
        cuda_matmul.allow_tf32 = bool(previous["cuda_matmul_allow_tf32"])
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None and hasattr(cudnn, "allow_tf32") and "cudnn_allow_tf32" in previous:
        cudnn.allow_tf32 = bool(previous["cudnn_allow_tf32"])
    setter = getattr(torch, "set_float32_matmul_precision", None)
    if setter is not None and "float32_matmul_precision" in previous:
        setter(str(previous["float32_matmul_precision"]))


def _observed_float_dtypes(model: Any) -> dict[str, list[str]]:
    parameters = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
    )
    buffers = sorted(
        {
            str(buffer.dtype).removeprefix("torch.")
            for buffer in model.buffers()
            if buffer.is_floating_point()
        }
    )
    return {
        "parameter_float_dtypes": parameters,
        "buffer_float_dtypes": buffers,
    }


def _attention_implementation(model: Any) -> Any:
    config = getattr(model, "config", None)
    if config is None:
        return None
    return getattr(config, "_attn_implementation", None) or getattr(
        config, "attn_implementation", None
    )


def _on_cuda(torch: Any, device: Any) -> bool:
    text = str(device)
    return text.startswith("cuda") and bool(torch.cuda.is_available())


def require_natural_geometry(scorer: Any) -> dict[str, Any]:
    """Refuse a tokenizer whose MKT / AA20 renderings are not width 5 / 22."""

    short = scorer.render([SHORT_SEQUENCE])[0]
    long = scorer.render([LONG_SEQUENCE])[0]
    if len(short.token_ids) != SHORT_WIDTH:
        raise ValueError(
            f"MKT natural width is {len(short.token_ids)}, expected {SHORT_WIDTH}"
        )
    if len(long.token_ids) != LONG_WIDTH:
        raise ValueError(
            f"AA20 natural width is {len(long.token_ids)}, expected {LONG_WIDTH}"
        )
    if short.n_residues != SHORT_TARGETS or short.n_scored_tokens != SHORT_TARGETS:
        raise ValueError(
            f"MKT residue/target count is {short.n_residues}/{short.n_scored_tokens}, "
            f"expected {SHORT_TARGETS}"
        )
    if long.n_residues != LONG_TARGETS or long.n_scored_tokens != LONG_TARGETS:
        raise ValueError(
            f"AA20 residue/target count is {long.n_residues}/{long.n_scored_tokens}, "
            f"expected {LONG_TARGETS}"
        )
    return {
        "mkt_width": SHORT_WIDTH,
        "aa20_width": LONG_WIDTH,
        "mkt_targets": SHORT_TARGETS,
        "aa20_targets": LONG_TARGETS,
        "mkt_scored_positions": list(short.scored_positions),
        "aa20_scored_positions": list(long.scored_positions),
    }


def pack_batch(scorer: Any, sequences: Sequence[str], *, width: int, forced_width: bool) -> PackedBatch:
    """Right-pad with the checkpoint pad id; labels keep residue positions only."""

    torch = scorer.torch
    records = scorer.render(list(sequences))
    if not records:
        raise ValueError("nothing to pack")
    natural = tuple(len(record.token_ids) for record in records)
    natural_max = max(natural)
    if width < natural_max:
        raise ValueError(
            f"illegal pad width {width} is shorter than natural max {natural_max}"
        )
    if not forced_width and width != natural_max:
        raise ValueError(
            f"natural-width case requested tensor width {width}, but the batch "
            f"renders at {natural_max}"
        )
    ids = torch.full((len(records), width), scorer.pad_id, dtype=torch.long)
    mask = torch.zeros((len(records), width), dtype=torch.long)
    labels = torch.full((len(records), width), -100, dtype=torch.long)
    scored: list[tuple[int, ...]] = []
    n_targets: list[int] = []
    for row, record in enumerate(records):
        length = len(record.token_ids)
        token_ids = torch.tensor(record.token_ids, dtype=torch.long)
        ids[row, :length] = token_ids
        mask[row, :length] = 1
        positions = tuple(int(position) for position in record.scored_positions)
        if positions:
            labels[row, list(positions)] = ids[row, list(positions)]
        scored.append(positions)
        n_targets.append(int(record.n_residues))
        if n_targets[-1] != len(positions):
            raise ValueError(
                f"residue count {n_targets[-1]} disagrees with scored positions "
                f"{positions}"
            )
    return PackedBatch(
        sequences=tuple(str(sequence) for sequence in sequences),
        width=int(width),
        forced_width=bool(forced_width),
        ids=ids,
        mask=mask,
        labels=labels,
        scored_positions=tuple(scored),
        n_targets=tuple(n_targets),
        natural_widths=natural,
    )


def score_packed(scorer: Any, packed: PackedBatch) -> dict[str, Any]:
    """Independent FP32 shifted CE plus a native labels forward; no 0.02 filter."""

    torch = scorer.torch
    model = scorer.loaded.model
    device = model.device
    vocab = int(scorer.vocab_size)
    ids = packed.ids.to(device)
    mask = packed.mask.to(device)
    labels = packed.labels.to(device)
    expected = (*ids.shape, vocab)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        if tuple(logits.shape) != expected:
            raise RuntimeError(
                f"{scorer.name}: expected logits shape {expected}, got {tuple(logits.shape)}"
            )
        shift_logits = logits[:, :-1].contiguous().float()
        shift_labels = labels[:, 1:]
        per_token = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(len(packed.sequences), packed.width - 1)
        if not torch.isfinite(per_token).all():
            raise RuntimeError(f"{scorer.name}: independent CE produced a non-finite value")
        valid = shift_labels != -100
        nll = (per_token * valid).sum(1)
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            labels=labels,
        )
        native_loss = getattr(outputs, "loss", None)
        if native_loss is None:
            raise RuntimeError(
                f"{scorer.name}: native CausalLM forward(labels=...) returned no loss; "
                "the independent shifted-CE reference is not a substitute"
            )
        native_loss_value = _finite(native_loss, what="native loss")
        del logits, outputs
    rows: list[dict[str, Any]] = []
    per_token_cpu = per_token.detach().cpu()
    nll_cpu = nll.detach().cpu()
    ids_cpu = ids.detach().cpu().tolist()
    mask_cpu = mask.detach().cpu().tolist()
    for row, sequence in enumerate(packed.sequences):
        targets = int(packed.n_targets[row])
        if targets < 1:
            raise RuntimeError(f"{scorer.name}: row {row} has no scored residue targets")
        per_target = []
        for position in packed.scored_positions[row]:
            per_target.append(
                _finite(
                    -per_token_cpu[row, position - 1],
                    what=f"target log-prob at position {position}",
                )
            )
        if len(per_target) != targets:
            raise RuntimeError(
                f"{scorer.name}: row {row} has {len(per_target)} target log-probs, "
                f"expected {targets}"
            )
        total = _finite(math.fsum(per_target), what=f"row {row} total log-likelihood")
        rows.append(
            {
                "sequence": sequence,
                "input_ids": [int(token) for token in ids_cpu[row]],
                "attention_mask": [int(flag) for flag in mask_cpu[row]],
                "scored_positions": list(packed.scored_positions[row]),
                "n_residue_targets": targets,
                "natural_width": int(packed.natural_widths[row]),
                "per_target_log_prob": per_target,
                "total_log_likelihood": total,
            }
        )
    n_targets = int(sum(packed.n_targets))
    independent_nll_sum = _finite(nll_cpu.sum(), what="independent NLL sum")
    native_vs = abs(native_loss_value * n_targets - independent_nll_sum) / n_targets
    del ids, mask, labels, per_token, nll, per_token_cpu, nll_cpu
    return {
        "sequences": list(packed.sequences),
        "width": int(packed.width),
        "forced_width": bool(packed.forced_width),
        "n_residue_targets_batch": n_targets,
        "independent_nll_sum": independent_nll_sum,
        "native_loss": native_loss_value,
        "native_vs_independent_nats_per_target": _finite(
            native_vs, what="native vs independent per-target"
        ),
        "rows": rows,
    }


def _public_totals(scorer: Any, sequences: Sequence[str]) -> list[float]:
    totals = scorer.log_likelihood(list(sequences))
    return [
        _finite(value, what=f"public scorer total for {sequence!r}")
        for sequence, value in zip(sequences, totals)
    ]


def _attach_public(scorer: Any, case: Mapping[str, Any], observation: dict[str, Any]) -> None:
    if not case["public_scorer"]:
        observation["public_scorer"] = {
            "called": False,
            "reason": "no public forced-width API; padded width is diagnostic-only",
        }
        return
    public = _public_totals(scorer, case["sequences"])
    independent = [row["total_log_likelihood"] for row in observation["rows"]]
    per_target = []
    for row, (left, right) in enumerate(zip(public, independent)):
        targets = observation["rows"][row]["n_residue_targets"]
        per_target.append(_finite(abs(left - right) / targets, what="public vs independent"))
    observation["public_scorer"] = {
        "called": True,
        "total_log_likelihood": public,
        "minus_independent": [public[i] - independent[i] for i in range(len(public))],
        "abs_nats_per_residue_target": per_target,
    }


def _row_key(repetition: int, case: str, row: int) -> tuple[int, str, int]:
    return (int(repetition), str(case), int(row))


def _comparisons(repetitions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    for block in repetitions:
        repetition = int(block["repetition"])
        for name, observation in block["cases"].items():
            for row_index, row in enumerate(observation["rows"]):
                by_key[_row_key(repetition, name, row_index)] = row
    contrasts: list[dict[str, Any]] = []
    for left_case, left_row, right_case, right_row, targets in COMPARISONS:
        for block in repetitions:
            repetition = int(block["repetition"])
            left = by_key[_row_key(repetition, left_case, left_row)]
            right = by_key[_row_key(repetition, right_case, right_row)]
            if left["sequence"] != right["sequence"]:
                raise RuntimeError(
                    f"comparison {left_case}[{left_row}] vs {right_case}[{right_row}] "
                    f"joins {left['sequence']!r} to {right['sequence']!r}"
                )
            if left["n_residue_targets"] != targets or right["n_residue_targets"] != targets:
                raise RuntimeError(
                    f"comparison {left_case}[{left_row}] vs {right_case}[{right_row}] "
                    f"expected {targets} residue targets"
                )
            per_target = [
                _finite(a - b, what="per-target contrast")
                for a, b in zip(left["per_target_log_prob"], right["per_target_log_prob"])
            ]
            total_delta = _finite(
                left["total_log_likelihood"] - right["total_log_likelihood"],
                what="total contrast",
            )
            contrasts.append(
                {
                    "repetition": repetition,
                    "left_case": left_case,
                    "left_row": left_row,
                    "right_case": right_case,
                    "right_row": right_row,
                    "sequence": left["sequence"],
                    "n_residue_targets": targets,
                    "total_delta": total_delta,
                    "abs_nats_per_residue_target": _finite(
                        abs(total_delta) / targets, what="per-target mean abs"
                    ),
                    "per_target_delta": per_target,
                }
            )
    return contrasts


def _repetition_complete(block: Mapping[str, Any]) -> bool:
    cases = block.get("cases")
    return isinstance(cases, Mapping) and all(case["name"] in cases for case in CASES)


def _resource_fields(torch: Any, device: Any, elapsed_s: float, measured: bool) -> dict[str, Any]:
    payload = {
        "elapsed_s": _finite(elapsed_s, what="elapsed_s"),
        "memory_measured": bool(measured),
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    if measured:
        payload["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        payload["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    else:
        payload["note"] = "CPU diagnostic stub; GPU memory was not measured"
    return payload


def _begin_phase(payload: dict[str, Any], phase: str) -> dict[str, Any]:
    record = {
        "phase": phase,
        "incomplete": True,
        "in_progress": None,
        "repetitions": [],
    }
    payload["phases"][phase] = record
    return record


def run_phase(
    scorer: Any,
    *,
    record: dict[str, Any],
    setup: Callable[[], dict[str, Any]] | None = None,
) -> None:
    torch = scorer.torch
    device = scorer.loaded.model.device
    measured = _on_cuda(torch, device)
    started = time.perf_counter()
    if measured:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    if setup is not None:
        record["setup"] = setup()
    record["model_training"] = bool(scorer.loaded.model.training)
    record["attention_implementation"] = _attention_implementation(scorer.loaded.model)
    record["observed_float_dtypes"] = _observed_float_dtypes(scorer.loaded.model)
    record["matmul_policy"] = _matmul_policy(torch)
    for repetition in range(REPETITIONS):
        block: dict[str, Any] = {"repetition": repetition, "cases": {}}
        record["repetitions"].append(block)
        for case in CASES:
            record["in_progress"] = {
                "repetition": repetition,
                "case": case["name"],
            }
            packed = pack_batch(
                scorer,
                case["sequences"],
                width=int(case["width"]),
                forced_width=bool(case["forced_width"]),
            )
            observation = score_packed(scorer, packed)
            observation["case"] = case["name"]
            observation["repetition"] = repetition
            block["cases"][case["name"]] = observation
            _attach_public(scorer, case, observation)
            record["in_progress"] = None
        if _repetition_complete(block):
            block["comparisons"] = _comparisons([block])
    if measured:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    record["resources"] = _resource_fields(torch, device, elapsed, measured)
    if (
        len(record["repetitions"]) == REPETITIONS
        and all(_repetition_complete(block) for block in record["repetitions"])
    ):
        record["comparisons"] = _comparisons(record["repetitions"])
        record["incomplete"] = False


def promote_bf16_rounded_weights_to_fp32_execution(
    loaded: G.LoadedGalactica,
    torch: Any,
    *,
    previous_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """In-place float() of already-loaded BF16 weights. This is not a reload.

    The caller must snapshot ``previous_policy`` before this mutates backends.
    """

    before = _observed_float_dtypes(loaded.model)
    _set_fp32_exec_backends(torch)
    loaded.model.float()
    loaded.model.eval()
    after = _observed_float_dtypes(loaded.model)
    return {
        "kind": "bf16_rounded_weights_executed_in_fp32",
        "reloaded": False,
        "facts_rewritten": False,
        "parameter_buffer_dtypes_before": before,
        "parameter_buffer_dtypes_after": after,
        "matmul_policy_before": dict(previous_policy),
        "matmul_policy_after": _matmul_policy(torch),
    }


def _base_payload(*, arm: str, device: str) -> dict[str, Any]:
    return {
        "kind": KIND,
        "status": "failed",
        "no_dms_scores": True,
        "no_interface_admission": True,
        "arm": arm,
        "device_requested": device,
        "load_dtype": LOAD_DTYPE,
        "created_utc": _utc_now(),
        "code_sources": {
            "diagnose_galactica_numerics": str(Path(__file__).resolve()),
            "galactica_fitness": str(Path(G.__file__).resolve()),
        },
        "cases": [
            {
                "name": case["name"],
                "sequences": list(case["sequences"]),
                "width": case["width"],
                "forced_width": case["forced_width"],
            }
            for case in CASES
        ],
        "repetitions": REPETITIONS,
        "phases": {},
        "failure": None,
    }


def run_diagnostic(
    *,
    out: Path,
    device: str,
    arm: str = ALLOWED_ARM,
    scorer: Any | None = None,
    release_scorer: bool | None = None,
) -> dict[str, Any]:
    """Load BF16 once, record 7×3, promote in place, record 7×3 again."""

    payload = _base_payload(arm=arm, device=device)
    owned = scorer is None if release_scorer is None else bool(release_scorer)
    current_phase = "load"
    previous_policy: dict[str, Any] | None = None
    torch: Any = None
    try:
        arm = _require_arm(arm)
        payload["arm"] = arm
        if scorer is None:
            try:
                import torch as torch_module
            except ImportError as exc:
                raise RuntimeError(
                    "torch is required for this diagnostic and is not installed; "
                    "missing dependencies are not auto-installed"
                ) from exc
            torch = torch_module
            payload["runtime"] = _runtime_versions(torch)
            loaded = G.load_galactica(arm, device=device, dtype=LOAD_DTYPE)
            loaded.model.eval()
            scorer = G.GalacticaFitnessScorer(loaded, batch_size=PUBLIC_BATCH_SIZE)
        else:
            torch = scorer.torch
            payload["runtime"] = _runtime_versions(torch)
        payload["initial_bf16_facts"] = _require_facts(getattr(scorer, "facts", None))
        payload["geometry"] = require_natural_geometry(scorer)
        current_phase = "bf16"
        run_phase(scorer, record=_begin_phase(payload, "bf16"))
        current_phase = "fp32_exec"
        fp32_record = _begin_phase(payload, "fp32_exec")
        previous_policy = _matmul_policy(torch)

        def _promote() -> dict[str, Any]:
            record = promote_bf16_rounded_weights_to_fp32_execution(
                scorer.loaded, torch, previous_policy=previous_policy
            )
            payload["converted_parameter_buffer_dtypes"] = record[
                "parameter_buffer_dtypes_after"
            ]
            payload["fp32_exec_note"] = (
                "BF16-rounded weights executed in FP32; not a from-checkpoint "
                "float32 load, not an interface PASS, and not a DMS protocol pass"
            )
            return record

        run_phase(scorer, record=fp32_record, setup=_promote)
        payload["status"] = "complete"
        payload["created_utc"] = _utc_now()
    except Exception as exc:
        payload["status"] = "failed"
        in_progress = None
        current = payload["phases"].get(current_phase)
        if isinstance(current, Mapping):
            in_progress = current.get("in_progress")
        payload["failure"] = {
            "phase": current_phase,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "in_progress": in_progress,
        }
        payload["created_utc"] = _utc_now()
    finally:
        cleanup: list[dict[str, str]] = []
        if torch is not None and previous_policy is not None:
            try:
                _restore_backends(torch, previous_policy)
            except Exception as exc:
                cleanup.append(
                    {
                        "where": "restore_backends",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        if owned and scorer is not None:
            try:
                scorer.release()
            except Exception as exc:
                cleanup.append(
                    {
                        "where": "release",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        if cleanup:
            payload["cleanup"] = cleanup
            payload["status"] = "failed"
            if payload.get("failure") is None:
                payload["failure"] = {
                    "phase": "cleanup",
                    "exception_type": cleanup[0]["exception_type"],
                    "message": cleanup[0]["message"],
                    "in_progress": None,
                }
        payload["created_utc"] = _utc_now()
        _write_payload(out, payload)
    return payload


def _write_payload(out: Path, payload: Mapping[str, Any]) -> None:
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / OUTPUT_NAME, dict(payload))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm", default=ALLOWED_ARM)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    payload = run_diagnostic(
        out=Path(args.out), device=str(args.device), arm=str(args.arm)
    )
    if payload["status"] != "complete":
        raise SystemExit(1)
    return payload


if __name__ == "__main__":
    main()
