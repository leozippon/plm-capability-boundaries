"""Shared FP32 matmul / TF32 policy and protocol identifiers.

This module is the only copy of the requested execution policy. It does not
load weights, score ProteinGym assays, or mint a capability verdict. Protocol
identifiers live here so probe helpers and scoring doors cannot drift.

``NATIVE_DMS_V1`` and ``GALACTICA_FP32_V2`` keep their existing meanings and
numeric behaviour. ``TEXT_AA_FP32_V1`` identifies the text-AA string-control
core; the stage-20 CLI parser does not accept that id yet.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping

NATIVE_DMS_V1 = "native-dms-v1"
GALACTICA_FP32_V2 = "galactica-fp32-v2"
TEXT_AA_FP32_V1 = "text-aa-fp32-v1"

#: Per-target absolute ceiling for independent shifted-CE versus a public
#: FP32 scorer, in nats. Native DMS re-exports this same constant; the numeric
#: ceiling is not moved.
FP32_PER_TARGET_ABS = 1e-4

_POLICY_KEYS = (
    "cuda_matmul_allow_tf32",
    "cudnn_allow_tf32",
    "float32_matmul_precision",
)


def requested_fp32_matmul_policy() -> dict[str, Any]:
    """The policy v2 asks for. Values are not read from the current process."""

    return {
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }


def snapshot_matmul_policy(torch: Any) -> dict[str, Any]:
    """Read the three policy fields. Missing attributes fail; they are not omitted."""

    try:
        cuda_matmul = torch.backends.cuda.matmul
        cuda_tf32 = cuda_matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        precision = torch.get_float32_matmul_precision()
    except AttributeError as exc:
        raise ValueError(
            "matmul policy snapshot requires cuda.matmul.allow_tf32, "
            f"cudnn.allow_tf32, and get_float32_matmul_precision; {exc}"
        ) from exc
    return {
        "cuda_matmul_allow_tf32": bool(cuda_tf32),
        "cudnn_allow_tf32": bool(cudnn_tf32),
        "float32_matmul_precision": str(precision),
    }


def require_observed_policy(policy: Mapping[str, Any]) -> None:
    """Refuse anything other than TF32 off and float32 matmul precision highest."""

    if not isinstance(policy, Mapping):
        raise ValueError(f"matmul policy must be a mapping, got {type(policy)!r}")
    missing = [key for key in _POLICY_KEYS if key not in policy]
    if missing:
        raise ValueError(f"matmul policy is missing {missing}")
    cuda_tf32 = policy["cuda_matmul_allow_tf32"]
    cudnn_tf32 = policy["cudnn_allow_tf32"]
    precision = policy["float32_matmul_precision"]
    if type(cuda_tf32) is not bool or cuda_tf32:
        raise ValueError(
            "cuda_matmul_allow_tf32 must be False, "
            f"got {cuda_tf32!r}"
        )
    if type(cudnn_tf32) is not bool or cudnn_tf32:
        raise ValueError(
            "cudnn_allow_tf32 must be False, "
            f"got {cudnn_tf32!r}"
        )
    if precision != "highest":
        raise ValueError(
            "float32_matmul_precision must be 'highest', "
            f"got {precision!r}"
        )


def _apply_matmul_policy(torch: Any, policy: Mapping[str, Any]) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(policy["cuda_matmul_allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(policy["cudnn_allow_tf32"])
    torch.set_float32_matmul_precision(str(policy["float32_matmul_precision"]))


@contextmanager
def fp32_matmul_context(torch: Any) -> Iterator[dict[str, Any]]:
    """Apply the v2 policy, yield requested/observed, restore even after failure.

    The snapshot is taken before any mutation. A later body or apply failure
    still restores that snapshot. A normal exit re-checks the live policy
    before restore.
    """

    requested = requested_fp32_matmul_policy()
    before = snapshot_matmul_policy(torch)
    try:
        _apply_matmul_policy(torch, requested)
        observed = snapshot_matmul_policy(torch)
        require_observed_policy(observed)
        yield {
            "requested": dict(requested),
            "observed": dict(observed),
            "before": dict(before),
        }
        require_observed_policy(snapshot_matmul_policy(torch))
    finally:
        _apply_matmul_policy(torch, before)
