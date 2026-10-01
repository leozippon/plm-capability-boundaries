#!/usr/bin/env python3
"""ProteinGLM derived-budget FP32 interface qualification. Not a panel stage.

This controller asks whether the declared 7B ProteinGLM continuation path is
usable at FP32: native ``<gmask><sop><eos>`` prefix, residues 2..L, no trailing
EOS. A qualified artefact is not EXP-R2-225 PASS, not ProteinGym admission, and
not panel admission. The 225 runner and ``DECLARED_UNAVAILABLE`` are untouched.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import transformers

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.core.arms import (  # noqa: E402
    REPO,
    Arm,
    Cohort,
    arm_spec,
    load_arm_spec,
    output_logit_width,
    require_declared_shape,
    require_scoring_target_ids,
    scoring_target_alphabet,
    tokenize_batch,
)
from src.capability.core.budget import scored_tokens  # noqa: E402
from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.core.precision_policy import FP32_PER_TARGET_ABS, fp32_matmul_context  # noqa: E402
from src.capability.models import proteinglm as pglm  # noqa: E402

PROTOCOL_ID = "proteinglm-derived-budget-fp32-v1"
SCHEMA_VERSION = "r2_transfer_proteinglm_derived_budget_qualification_v1"
ARTEFACT_NAME = "proteinglm_derived_budget_qualification.json"
ARM_NAME = "proteinglm-7b-clm"
REQUIRED_DTYPE = "float32"
REQUIRED_LIVE_WIDTH = 128
REQUIRED_N_LAYER = 36
REQUIRED_D_MODEL = 4096
STATUS_QUALIFIED = "derived-budget-interface-qualified"
STATUS_FAILED = "failed"
DEFAULT_OUT = REPO / "results/R1/proteinglm_budget_qualification"
MAX_LEN = pglm.CONTEXT_LENGTH
LONGEST_RESIDUES = pglm.CONTEXT_LENGTH - pglm.PREFIX_LENGTH
RESOURCE_N_SEQUENCES = 3
RESOURCE_N_TARGETS = RESOURCE_N_SEQUENCES * (LONGEST_RESIDUES - 1)

AVGFP_N80 = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQ"
)
AVGFP_N80_SHA256 = (
    "3a6033d2eb88fa724c2aab48cb19d903201b1c2efefda1b82b8edb83a03ebbcf"
)

SOURCE_RELATIVE_PATHS = (
    Path("scripts/capability/mutation/proteinglm_budget_qualification.py"),
    Path("src/capability/models/proteinglm.py"),
    Path("src/capability/core/budget.py"),
    Path("src/capability/core/scoring.py"),
    Path("src/capability/core/arms.py"),
    Path("src/capability/core/precision_policy.py"),
    Path("src/capability/core/amino_acids.py"),
)


class QualificationFailed(Exception):
    """Published qualification failed after valid CLI arguments."""

    def __init__(self, path: Path, payload: Mapping[str, Any]):
        self.path = Path(path)
        self.payload = dict(payload)
        super().__init__(f"wrote failed artefact {self.path}")


@dataclass(frozen=True)
class NumericCase:
    name: str
    sequence: str

    @property
    def n_residues(self) -> int:
        return len(self.sequence)

    @property
    def n_targets(self) -> int:
        return self.n_residues - 1


@dataclass(frozen=True)
class IndependentReference:
    nll_nats: np.ndarray
    target_ids: np.ndarray
    target_positions: np.ndarray
    live_width: int
    logits_dtype: str
    logits_shape: tuple[int, ...]
    parameter_dtypes: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def longest_legal_sequence() -> str:
    n = LONGEST_RESIDUES
    return (AA20 * ((n + len(AA20) - 1) // len(AA20)))[:n]


def case_specifications() -> tuple[NumericCase, ...]:
    avgfp = NumericCase("avgfp_n80", AVGFP_N80)
    if len(avgfp.sequence) != 80:
        raise ValueError("avgfp_n80 must be 80 residues")
    longest = NumericCase("longest_legal_1021", longest_legal_sequence())
    if longest.n_residues != LONGEST_RESIDUES:
        raise ValueError("longest legal case must be 1021 residues")
    return (
        NumericCase("minimum_two_residues", "AC"),
        NumericCase("canonical_aa20", AA20),
        avgfp,
        longest,
    )


def source_identity() -> dict[str, Any]:
    files = []
    for relative in SOURCE_RELATIVE_PATHS:
        path = REPO_ROOT / relative
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
            }
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "files": files,
        "not_git_head": (
            "hashes are of the files this process imported; a neighbouring "
            "commit SHA is not substituted"
        ),
    }


def runtime_whitelist(device: str) -> dict[str, Any]:
    cuda = getattr(torch.version, "cuda", None)
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "cuda": None if cuda is None else str(cuda),
        "device": str(device),
    }


def sanitize_error(exc: BaseException) -> dict[str, str]:
    message = str(exc).replace("\n", " ").strip()
    if len(message) > 500:
        message = message[:500]
    return {"class": type(exc).__name__, "message": message}


def require_arm_name(name: str) -> None:
    if name != ARM_NAME:
        raise ValueError(f"this controller accepts only {ARM_NAME!r}, got {name!r}")


def require_dtype(dtype: str) -> None:
    if dtype != REQUIRED_DTYPE:
        raise ValueError(
            f"ProteinGLM derived-budget qualification is FP32-only; refused {dtype!r}"
        )


def require_cuda_device(device: str) -> None:
    if not str(device).startswith("cuda"):
        raise ValueError(
            "published derived-budget qualification loads the 7B checkpoint "
            f"only on CUDA; refused device {device!r}"
        )


def floating_parameter_dtypes(model: torch.nn.Module) -> tuple[str, ...]:
    observed = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
    )
    return tuple(observed)


def require_eval_fp32(arm: Arm) -> None:
    if arm.model.training:
        raise RuntimeError(f"{arm.name}: serving requires eval(); training is unsupported")
    dtypes = floating_parameter_dtypes(arm.model)
    if dtypes != ("float32",):
        raise ValueError(f"{arm.name}: declared dtype float32, observed {list(dtypes)}")


def require_load_evidence(arm: Arm) -> None:
    if arm.strict_load is None:
        raise ValueError(
            f"{arm.name}: missing strict_load from this load_arm_spec call; "
            "refusing to reconstruct a clean load"
        )
    provenance = arm.serving_provenance
    if not isinstance(provenance, Mapping):
        raise ValueError(
            f"{arm.name}: missing serving_provenance from this load_arm_spec call"
        )
    if "derived" not in provenance or "max_length" not in provenance:
        raise ValueError(
            f"{arm.name}: serving_provenance must contain derived and max_length "
            "from this load, not a later re-derive"
        )


def require_declared_7b_shape(arm: Arm) -> None:
    if (arm.spec.n_layer, arm.spec.d_model) != (REQUIRED_N_LAYER, REQUIRED_D_MODEL):
        raise ValueError(
            f"{arm.name}: declared {REQUIRED_N_LAYER}L/{REQUIRED_D_MODEL}d, "
            f"spec is {arm.spec.n_layer}L/{arm.spec.d_model}d; a tiny or "
            "mislabelled checkpoint cannot be the published 7B arm"
        )
    require_declared_shape(arm.spec, arm.model.config)
    pglm.require_serving_config(arm.model.config)
    seq_length = int(getattr(arm.model.config, "seq_length"))
    if seq_length != pglm.CONTEXT_LENGTH:
        raise ValueError(
            f"{arm.name}: scoring window is seq_length={pglm.CONTEXT_LENGTH}, "
            f"got {seq_length}; max_length is not the context"
        )


@contextmanager
def inference_precision(arm: Arm) -> Iterator[dict[str, Any]]:
    """inference_mode + no autocast around the shared fp32_matmul_context."""

    device_type = "cuda" if str(arm.device).startswith("cuda") else "cpu"
    with torch.inference_mode():
        with torch.autocast(device_type=device_type, enabled=False):
            with fp32_matmul_context(torch) as policy:
                yield policy


def public_scored_tokens(
    arm: Arm,
    strings: Sequence[str],
    *,
    max_len: int,
    batch_size: int,
    call_log: list[dict[str, Any]] | None = None,
):
    recorded = {
        "n_strings": len(list(strings)),
        "strings": list(strings),
        "max_len": int(max_len),
        "batch_size": int(batch_size),
    }
    if call_log is not None:
        call_log.append(recorded)
    return scored_tokens(
        arm,
        list(strings),
        max_len=max_len,
        batch_size=batch_size,
    )


def independent_shifted_ce(
    arm: Arm,
    ids: torch.Tensor,
    mask: torch.Tensor,
) -> IndependentReference:
    """Per-case shifted CE. Does not call sequence_target_mask or production gather."""

    if ids.ndim != 2 or mask.shape != ids.shape:
        raise ValueError("independent reference needs a single padded [B,S] pair")
    if int(ids.shape[0]) != 1:
        raise ValueError("independent reference is per-case; batch must be 1")
    device_ids = ids.to(arm.device)
    device_mask = mask.to(arm.device)
    prefix = device_ids[0, : pglm.PREFIX_LENGTH]
    expected = device_ids.new_tensor(pglm.PREFIX_IDS)
    if int(prefix.numel()) != pglm.PREFIX_LENGTH or not bool((prefix == expected).all()):
        raise ValueError("independent reference batch does not start with <gmask><sop><eos> ids")
    valid_len = int(device_mask[0].sum().item())
    n_residues = valid_len - pglm.PREFIX_LENGTH
    if n_residues < 2:
        raise ValueError("independent reference needs at least two residues")
    outputs = arm.model(input_ids=device_ids, attention_mask=device_mask)
    logits = outputs.logits
    if logits.ndim != 3:
        raise ValueError(f"{arm.name}: reference logits must be [B,S,V], got {tuple(logits.shape)}")
    batch, width, vocab = tuple(int(dim) for dim in logits.shape)
    if batch != 1 or width != int(device_ids.shape[1]):
        raise ValueError(
            f"{arm.name}: reference logits shape {tuple(logits.shape)} does not match ids"
        )
    live_width = int(vocab)
    if str(logits.dtype).removeprefix("torch.") != "float32":
        raise ValueError(
            f"{arm.name}: reference logits dtype must be float32, got {logits.dtype}"
        )
    parameter_dtypes = floating_parameter_dtypes(arm.model)
    shift_logits = logits[:, :-1, :].float()
    shift_targets = device_ids[:, 1:]
    token_ce = F.cross_entropy(
        shift_logits.reshape(-1, live_width),
        shift_targets.reshape(-1),
        reduction="none",
    ).reshape(1, width - 1)
    columns = torch.arange(width - 1, device=token_ce.device)
    valid = device_mask[:, 1:].bool() & device_mask[:, :-1].bool()
    keep = valid[0] & (columns >= pglm.PREFIX_LENGTH) & (columns <= valid_len - 2)
    selected = token_ce[0, keep]
    target_ids = shift_targets[0, keep]
    positions = columns[keep]
    if not bool(torch.isfinite(selected).all()):
        raise FloatingPointError(f"{arm.name}: non-finite independent reference NLL")
    if int(selected.numel()) != n_residues - 1:
        raise ValueError(
            f"{arm.name}: independent reference kept {int(selected.numel())} "
            f"targets, expected {n_residues - 1}"
        )
    if int(positions[0].item()) != pglm.PREFIX_LENGTH:
        raise ValueError("first residue must not be a target (keep starts at q=3)")
    if int(positions[-1].item()) != valid_len - 2:
        raise ValueError("last residue must be a target")
    return IndependentReference(
        nll_nats=selected.detach().double().cpu().numpy(),
        target_ids=target_ids.detach().cpu().numpy().astype(np.int64),
        target_positions=positions.detach().cpu().numpy().astype(np.int64),
        live_width=live_width,
        logits_dtype=str(logits.dtype).removeprefix("torch."),
        logits_shape=(batch, width, vocab),
        parameter_dtypes=parameter_dtypes,
    )


def _case_record(
    case: NumericCase,
    *,
    encoded_ids: Sequence[int],
    production_ids: np.ndarray,
    production_nll: np.ndarray,
    reference: IndependentReference,
) -> dict[str, Any]:
    if production_ids.shape != production_nll.shape:
        raise ValueError(f"{case.name}: production ids and NLL are misaligned")
    if int(production_ids.size) != case.n_targets:
        raise ValueError(
            f"{case.name}: production kept {int(production_ids.size)} targets, "
            f"expected {case.n_targets}"
        )
    if not np.array_equal(production_ids, reference.target_ids):
        raise ValueError(f"{case.name}: production and reference target ids differ")
    if int(reference.target_positions.size) != case.n_targets:
        raise ValueError(f"{case.name}: reference target count {reference.target_positions.size}")
    if not np.isfinite(production_nll).all():
        raise FloatingPointError(f"{case.name}: non-finite production NLL")
    delta = np.abs(production_nll - reference.nll_nats)
    max_abs = float(delta.max()) if delta.size else float("nan")
    if max_abs > FP32_PER_TARGET_ABS:
        raise ValueError(
            f"{case.name}: independent CE max abs {max_abs} exceeds "
            f"{FP32_PER_TARGET_ABS}"
        )
    if reference.live_width != REQUIRED_LIVE_WIDTH:
        raise ValueError(
            f"{case.name}: live logits width {reference.live_width} != "
            f"{REQUIRED_LIVE_WIDTH}; head/config width is not a substitute"
        )
    return {
        "name": case.name,
        "sequence": case.sequence,
        "L": case.n_residues,
        "n_targets": case.n_targets,
        "encoded_length": len(list(encoded_ids)),
        "target_positions": [int(value) for value in reference.target_positions.tolist()],
        "target_ids": [int(value) for value in production_ids.tolist()],
        "production_nll_nats": [float(value) for value in production_nll.tolist()],
        "reference_nll_nats": [float(value) for value in reference.nll_nats.tolist()],
        "max_abs": max_abs,
        "live_logits_width": reference.live_width,
        "logits_dtype": reference.logits_dtype,
        "logits_shape": list(reference.logits_shape),
        "parameter_dtypes": list(reference.parameter_dtypes),
    }


def score_numeric_cases(arm: Arm) -> dict[str, Any]:
    """Score the four frozen strings through production budget NLL plus independent CE."""

    require_eval_fp32(arm)
    cases = case_specifications()
    sequences = [case.sequence for case in cases]
    cohort = Cohort(
        name=PROTOCOL_ID,
        kind="protein",
        records=list(sequences),
        min_symbols=2,
        max_symbols=LONGEST_RESIDUES,
        metadata={"protocol_id": PROTOCOL_ID, "role": "derived-budget-numeric-cases"},
    )
    rendered = cohort.input_strings(arm)
    if len(rendered) != len(cases):
        raise ValueError("rendering the numeric cases returned the wrong count")
    alphabet = scoring_target_alphabet(arm.spec, getattr(arm.model, "config", None))
    production_calls: list[dict[str, Any]] = []
    with inference_precision(arm) as policy:
        scored = public_scored_tokens(
            arm,
            rendered,
            max_len=MAX_LEN,
            batch_size=1,
            call_log=production_calls,
        )
        if len(production_calls) != 1:
            raise ValueError("numeric cases must use one public scored_tokens call")
        if production_calls[0]["batch_size"] != 1 or production_calls[0]["max_len"] != MAX_LEN:
            raise ValueError("numeric production must be batch_size=1 and max_len=1024")
        if production_calls[0]["strings"] != list(rendered):
            raise ValueError("numeric production strings must be the four rendered cases")
        records = []
        live_widths = []
        for index, case in enumerate(cases):
            ids, mask = tokenize_batch(arm, [rendered[index]], MAX_LEN)
            encoded = [int(value) for value in ids[0, : int(mask[0].sum())].tolist()]
            reference = independent_shifted_ce(arm, ids, mask)
            selected = scored.sequence_index == index
            production_ids = scored.target_ids[selected]
            production_nll = scored.nll_nats[selected]
            require_scoring_target_ids(production_ids, alphabet, arm=arm.name)
            records.append(
                _case_record(
                    case,
                    encoded_ids=encoded,
                    production_ids=production_ids,
                    production_nll=production_nll,
                    reference=reference,
                )
            )
            live_widths.append(reference.live_width)
    head_width = output_logit_width(arm)
    return {
        "cases": records,
        "numeric_public_call_count": 1,
        "production_calls": [
            {
                "n_strings": item["n_strings"],
                "max_len": item["max_len"],
                "batch_size": item["batch_size"],
            }
            for item in production_calls
        ],
        "independent_ce_max_abs": max(float(record["max_abs"]) for record in records),
        "live_logits_width": int(live_widths[0]),
        "head_declared_width": {
            "size": int(head_width["size"]),
            "source": str(head_width["source"]),
            "not_live_logits_width": (
                "recorded for comparison; the gate is the actual logits [B,S,V] axis"
            ),
        },
        "matmul_policy": policy,
        "target_rule": pglm.TARGET_RULE,
        "input_format": pglm.INPUT_FORMAT,
        "prefix_ids": list(pglm.PREFIX_IDS),
        "max_len": MAX_LEN,
        "max_length_not_context": True,
    }


def capture_cuda_end_state(device: torch.device, started: float) -> dict[str, Any]:
    """Bounded post-call CUDA telemetry. Failures are unknown, never invented zeros."""

    readings: dict[str, Any] = {}
    try:
        torch.cuda.synchronize(device)
        readings["synchronize"] = "ok"
    except Exception as exc:
        readings["synchronize"] = {"status": "unknown", "error": sanitize_error(exc)}
    try:
        readings["duration_s"] = float(time.perf_counter() - started)
    except Exception as exc:
        readings["duration_s"] = {"status": "unknown", "error": sanitize_error(exc)}
    try:
        readings["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    except Exception as exc:
        readings["peak_allocated_bytes"] = {"status": "unknown", "error": sanitize_error(exc)}
    try:
        readings["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    except Exception as exc:
        readings["peak_reserved_bytes"] = {"status": "unknown", "error": sanitize_error(exc)}
    return readings


def cuda_end_state_is_complete(cuda_memory: Mapping[str, Any] | None) -> bool:
    if not isinstance(cuda_memory, Mapping):
        return False
    if cuda_memory.get("synchronize") != "ok":
        return False
    required = (
        "baseline_allocated_bytes",
        "baseline_reserved_bytes",
        "total_memory_bytes",
        "duration_s",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    for key in required:
        value = cuda_memory.get(key)
        if isinstance(value, bool) or type(value) not in (int, float):
            return False
        if not math.isfinite(float(value)):
            return False
    return True


def run_resource_gate(
    arm: Arm,
    longest_sequence: str,
    *,
    record_cuda: bool,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if progress is None:
        progress = {}
    progress["phase"] = "validate_input"
    progress["expected_n_sequences"] = RESOURCE_N_SEQUENCES
    progress["expected_n_targets"] = RESOURCE_N_TARGETS
    progress["longest_L"] = LONGEST_RESIDUES
    progress["batch_size"] = 1
    progress["max_len"] = MAX_LEN
    progress["resource_public_call_attempted"] = 0
    progress["resource_public_call_completed"] = 0
    progress["terminal_public_call_capture_count"] = 0
    progress.pop("resource_public_call_count", None)
    progress.pop("finite", None)

    if longest_sequence != longest_legal_sequence():
        raise ValueError(
            "resource gate requires the frozen 1021-residue longest legal input; "
            "short strings are not a substitute"
        )
    require_eval_fp32(arm)
    rendered = pglm.render_budget_sequence(longest_sequence)
    inputs = [rendered, rendered, rendered]
    progress["input_identity"] = {
        "n_strings": len(inputs),
        "max_len": MAX_LEN,
        "batch_size": 1,
    }
    production_calls: list[dict[str, Any]] = []
    cuda_block: dict[str, Any] | None = None
    started: float | None = None
    cuda_device: torch.device | None = None

    if record_cuda:
        require_cuda_device(arm.device)
        if not torch.cuda.is_available():
            raise RuntimeError("resource gate requires a live CUDA device")
        cuda_device = torch.device(arm.device)
        progress["phase"] = "cuda_reset"
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
        cuda_block = {}
        progress["cuda_memory"] = cuda_block
        cuda_block["baseline_allocated_bytes"] = int(torch.cuda.memory_allocated(cuda_device))
        cuda_block["baseline_reserved_bytes"] = int(torch.cuda.memory_reserved(cuda_device))
        cuda_block["total_memory_bytes"] = int(
            torch.cuda.get_device_properties(cuda_device).total_memory
        )
        started = time.perf_counter()

    with inference_precision(arm) as policy:
        progress["matmul_policy"] = {
            "requested": dict(policy["requested"]),
            "observed": dict(policy["observed"]),
        }
        progress["phase"] = "public_scored_tokens"
        progress["resource_public_call_attempted"] = 1
        try:
            scored = public_scored_tokens(
                arm,
                inputs,
                max_len=MAX_LEN,
                batch_size=1,
                call_log=production_calls,
            )
            progress["resource_public_call_completed"] = 1
        finally:
            if record_cuda and started is not None and cuda_device is not None and cuda_block is not None:
                progress["terminal_public_call_capture_count"] = 1
                try:
                    cuda_block.update(capture_cuda_end_state(cuda_device, started))
                except Exception as telemetry_exc:
                    progress["telemetry_error"] = sanitize_error(telemetry_exc)
        if record_cuda and not cuda_end_state_is_complete(cuda_block):
            raise ValueError(
                "qualified resource facts require complete CUDA telemetry; "
                "synchronize must be ok and required readings finite"
            )
        if len(production_calls) != 1:
            raise ValueError(
                "resource gate must be one public scored_tokens call, "
                f"got {len(production_calls)}"
            )
        if production_calls[0]["strings"] != inputs:
            raise ValueError("resource gate strings must be [longest, longest, longest]")
        if production_calls[0]["batch_size"] != 1 or production_calls[0]["max_len"] != MAX_LEN:
            raise ValueError("resource gate must be batch_size=1 and max_len=1024")
        n_targets = int(scored.target_ids.size)
        if n_targets != RESOURCE_N_TARGETS:
            raise ValueError(
                f"resource gate kept {n_targets} targets, expected {RESOURCE_N_TARGETS}"
            )
        if not np.isfinite(scored.nll_nats).all():
            raise FloatingPointError("resource gate produced non-finite NLL")
        progress["phase"] = "independent_reference"
        ids, mask = tokenize_batch(arm, [rendered], MAX_LEN)
        reference = independent_shifted_ce(arm, ids, mask)
        per_sequence = []
        for index in range(RESOURCE_N_SEQUENCES):
            selected = scored.sequence_index == index
            production_ids = scored.target_ids[selected]
            production_nll = scored.nll_nats[selected]
            if not np.array_equal(production_ids, reference.target_ids):
                raise ValueError(
                    f"resource sequence {index}: target ids differ from longest reference"
                )
            delta = np.abs(production_nll - reference.nll_nats)
            max_abs = float(delta.max())
            if max_abs > FP32_PER_TARGET_ABS:
                raise ValueError(
                    f"resource sequence {index}: independent CE max abs {max_abs} "
                    f"exceeds {FP32_PER_TARGET_ABS}"
                )
            per_sequence.append(
                {
                    "sequence_index": index,
                    "n_targets": int(production_ids.size),
                    "max_abs": max_abs,
                }
            )
    payload: dict[str, Any] = {
        "resource_public_call_count": 1,
        "n_sequences": RESOURCE_N_SEQUENCES,
        "n_targets": n_targets,
        "batch_size": 1,
        "max_len": MAX_LEN,
        "longest_L": LONGEST_RESIDUES,
        "finite": True,
        "per_sequence": per_sequence,
        "matmul_policy": policy,
        "independent_ce_max_abs": max(item["max_abs"] for item in per_sequence),
        "cuda_memory": cuda_block,
        "production_calls": [
            {
                "n_strings": item["n_strings"],
                "max_len": item["max_len"],
                "batch_size": item["batch_size"],
            }
            for item in production_calls
        ],
        "not_all_qualification_public_calls": (
            "this count is the resource-gate public scored_tokens call only"
        ),
        "expected_n_sequences": RESOURCE_N_SEQUENCES,
        "expected_n_targets": RESOURCE_N_TARGETS,
        "input_identity": progress["input_identity"],
        "resource_public_call_attempted": 1,
        "resource_public_call_completed": 1,
        "terminal_public_call_capture_count": progress["terminal_public_call_capture_count"],
        "phase": "complete",
    }
    progress.update(payload)
    return payload


def envelope(
    *,
    device: str,
    dtype: str,
    status: str,
) -> dict[str, Any]:
    if status not in (STATUS_QUALIFIED, STATUS_FAILED):
        raise ValueError(f"unknown status {status!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "arm": ARM_NAME,
        "device": device,
        "dtype": dtype,
        "status": status,
        "experiment_admitted": False,
        "not_panel_admission": True,
        "not_exp_r2_225": True,
        "not_proteingym": True,
        "executed_at_utc": utc_now(),
        "runtime": runtime_whitelist(device),
        "source_identity": source_identity(),
        "pass_means": (
            "the declared 7B derived-budget continuation interface is usable at "
            "FP32 on this device; not panel admission, not ProteinGym, not 225"
        ),
    }


def build_qualified_payload(
    arm: Arm,
    *,
    device: str,
    dtype: str,
    numeric: Mapping[str, Any],
    resource: Mapping[str, Any],
) -> dict[str, Any]:
    require_cuda_device(device)
    require_load_evidence(arm)
    require_declared_7b_shape(arm)
    require_eval_fp32(arm)
    if int(resource.get("resource_public_call_count", 0)) != 1:
        raise ValueError("qualified artefact requires resource_public_call_count=1")
    if resource.get("cuda_memory") is None:
        raise ValueError("qualified artefact requires CUDA resource facts; CPU tests cannot mint them")
    if not cuda_end_state_is_complete(resource.get("cuda_memory")):
        raise ValueError(
            "qualified artefact requires complete CUDA telemetry; "
            "unknown readings cannot mint a pass"
        )
    max_length = arm.serving_provenance["max_length"]
    payload = envelope(device=device, dtype=dtype, status=STATUS_QUALIFIED)
    payload.update(
        {
            "loaded": True,
            "shape": {
                "n_layer": int(arm.spec.n_layer),
                "d_model": int(arm.spec.d_model),
                "seq_length": int(arm.model.config.seq_length),
                "is_causal": bool(arm.model.config.is_causal),
                "rotary_embedding_2d": bool(arm.model.config.rotary_embedding_2d),
                "quantization_bit": int(arm.model.config.quantization_bit),
                "moe": bool(arm.model.config.moe),
            },
            "strict_load": dict(arm.strict_load),
            "serving_provenance": {
                "derived": arm.serving_provenance["derived"],
                "max_length": max_length,
            },
            "max_length_not_used_as_context": True,
            "live_output_width": int(numeric["live_logits_width"]),
            "input_format": pglm.INPUT_FORMAT,
            "target_rule": pglm.TARGET_RULE,
            "prefix_ids": list(pglm.PREFIX_IDS),
            "numeric": dict(numeric),
            "resource": dict(resource),
            "independent_ce_max_abs": max(
                float(numeric["independent_ce_max_abs"]),
                float(resource["independent_ce_max_abs"]),
            ),
            "training_unsupported": True,
            "generate_unsupported": True,
            "cache_reuse_unsupported": True,
            "gradient_checkpointing": "raises; ordinary .train() without checkpointing may still forward",
        }
    )
    return payload


def build_failed_payload(
    *,
    device: str,
    dtype: str,
    error: BaseException,
    loaded: bool,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = envelope(device=device, dtype=dtype, status=STATUS_FAILED)
    payload.update(
        {
            "loaded": bool(loaded),
            "error": sanitize_error(error),
        }
    )
    if extras:
        payload["partial"] = dict(extras)
    return payload


def artefact_path(directory: Path) -> Path:
    return Path(directory) / ARTEFACT_NAME


def artefact_entry_exists(path: Path) -> bool:
    """True for any directory entry, including a dangling symlink. Status is not read."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def refuse_existing_artefact(path: Path) -> None:
    destination = Path(path)
    if artefact_entry_exists(destination):
        raise FileExistsError(
            f"refusing to replace existing artefact {destination}; use a fresh --out"
        )


def publish_artefact(destination: Path, payload: Mapping[str, Any]) -> Path:
    """Write a complete sibling via shared write_json, then os.link without replace."""
    destination = Path(destination)
    refuse_existing_artefact(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.publish-{os.getpid()}-{time.time_ns()}"
    )
    try:
        write_json(temporary, dict(payload))
        os.link(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def write_artefact(directory: Path, payload: Mapping[str, Any]) -> Path:
    status = payload.get("status")
    if status not in (STATUS_QUALIFIED, STATUS_FAILED):
        raise ValueError(f"refusing to write artefact with status {status!r}")
    if payload.get("experiment_admitted") is not False:
        raise ValueError("experiment_admitted must be false")
    if payload.get("not_panel_admission") is not True:
        raise ValueError("not_panel_admission must be true")
    return publish_artefact(artefact_path(directory), payload)


def load_published_arm(*, device: str, dtype: str) -> Arm:
    require_arm_name(ARM_NAME)
    require_dtype(dtype)
    require_cuda_device(device)
    spec = arm_spec(ARM_NAME)
    arm = load_arm_spec(spec, device=device, dtype=dtype, strict=True)
    require_load_evidence(arm)
    require_declared_7b_shape(arm)
    require_eval_fp32(arm)
    return arm


def run_qualification(
    *,
    arm: str,
    device: str,
    dtype: str,
    out: Path,
    load_fn: Callable[..., Arm] | None = None,
) -> Path:
    require_arm_name(arm)
    require_dtype(dtype)
    destination_dir = Path(out)
    destination = artefact_path(destination_dir)
    refuse_existing_artefact(destination)
    destination_dir.mkdir(parents=True, exist_ok=True)
    loaded_arm: Arm | None = None
    extras: dict[str, Any] = {}
    resource_progress: dict[str, Any] = {}
    extras["resource"] = resource_progress
    try:
        require_cuda_device(device)
        loader = load_fn or load_published_arm
        loaded_arm = loader(device=device, dtype=dtype)
        if loaded_arm.strict_load is not None:
            extras["strict_load"] = dict(loaded_arm.strict_load)
        provenance = loaded_arm.serving_provenance
        if isinstance(provenance, Mapping):
            extras["serving_provenance"] = {
                key: provenance[key]
                for key in ("derived", "max_length")
                if key in provenance
            }
        numeric = score_numeric_cases(loaded_arm)
        extras["numeric"] = numeric
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        resource = run_resource_gate(
            loaded_arm,
            longest_legal_sequence(),
            record_cuda=True,
            progress=resource_progress,
        )
        payload = build_qualified_payload(
            loaded_arm,
            device=device,
            dtype=dtype,
            numeric=numeric,
            resource=resource,
        )
        return write_artefact(destination_dir, payload)
    except FileExistsError:
        raise
    except Exception as exc:
        payload = build_failed_payload(
            device=device,
            dtype=dtype,
            error=exc,
            loaded=loaded_arm is not None,
            extras=extras or None,
        )
        try:
            path = write_artefact(destination_dir, payload)
        except FileExistsError:
            raise
        raise QualificationFailed(path, payload) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--arm",
        required=True,
        choices=(ARM_NAME,),
        help="the one checkpoint this invocation qualifies",
    )
    parser.add_argument(
        "--dtype",
        default=REQUIRED_DTYPE,
        choices=(REQUIRED_DTYPE,),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        path = run_qualification(
            arm=args.arm,
            device=args.device,
            dtype=args.dtype,
            out=args.out,
        )
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    except QualificationFailed as exc:
        print(f"failed artefact {exc.path}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
