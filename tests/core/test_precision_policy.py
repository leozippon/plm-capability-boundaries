"""CPU contract for the shared Galactica FP32 matmul policy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.capability.core.precision_policy import (
    FP32_PER_TARGET_ABS,
    GALACTICA_FP32_V2,
    NATIVE_DMS_V1,
    TEXT_AA_FP32_V1,
    fp32_matmul_context,
    requested_fp32_matmul_policy,
    require_observed_policy,
    snapshot_matmul_policy,
)


class _Flag:
    def __init__(self, value: bool) -> None:
        self.allow_tf32 = value


class _Cudnn:
    def __init__(self, value: bool) -> None:
        self._value = value
        self.fail_on_false = False

    @property
    def allow_tf32(self) -> bool:
        return self._value

    @allow_tf32.setter
    def allow_tf32(self, value: bool) -> None:
        if self.fail_on_false and not value:
            self.fail_on_false = False
            raise RuntimeError("partial apply")
        self._value = bool(value)


def _fake_torch(*, cuda: bool = True, cudnn: bool = True, precision: str = "high"):
    state = {"precision": precision}
    torch_mod = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=_Flag(cuda)),
            cudnn=_Cudnn(cudnn),
        )
    )

    def getter() -> str:
        return str(state["precision"])

    def setter(value: str) -> None:
        state["precision"] = str(value)

    torch_mod.get_float32_matmul_precision = getter
    torch_mod.set_float32_matmul_precision = setter
    torch_mod._state = state
    return torch_mod


def test_requested_policy_is_tf32_off_and_highest():
    assert NATIVE_DMS_V1 == "native-dms-v1"
    assert GALACTICA_FP32_V2 == "galactica-fp32-v2"
    assert TEXT_AA_FP32_V1 == "text-aa-fp32-v1"
    assert FP32_PER_TARGET_ABS == 1e-4
    requested = requested_fp32_matmul_policy()
    require_observed_policy(requested)
    assert requested == {
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }


def test_require_observed_policy_refuses_missing_or_wrong_fields():
    with pytest.raises(ValueError, match="missing"):
        require_observed_policy({"cuda_matmul_allow_tf32": False})
    require_observed_policy({**requested_fp32_matmul_policy(), "before": {}})
    with pytest.raises(ValueError, match="cuda_matmul_allow_tf32"):
        require_observed_policy(
            {
                "cuda_matmul_allow_tf32": 0,
                "cudnn_allow_tf32": False,
                "float32_matmul_precision": "highest",
            }
        )
    with pytest.raises(ValueError, match="highest"):
        require_observed_policy(
            {
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "float32_matmul_precision": "high",
            }
        )


def test_snapshot_refuses_missing_backend_attributes():
    with pytest.raises(ValueError, match="snapshot"):
        snapshot_matmul_policy(SimpleNamespace(backends=SimpleNamespace()))


def test_fp32_matmul_context_restores_on_success_exception_and_partial_apply():
    torch_mod = _fake_torch()
    with fp32_matmul_context(torch_mod) as policy:
        require_observed_policy(policy["requested"])
        require_observed_policy(policy["observed"])
        assert torch_mod.backends.cuda.matmul.allow_tf32 is False
        assert torch_mod.backends.cudnn.allow_tf32 is False
        assert torch_mod.get_float32_matmul_precision() == "highest"
        assert policy["before"]["cuda_matmul_allow_tf32"] is True
        assert policy["before"]["float32_matmul_precision"] == "high"
    assert torch_mod.backends.cuda.matmul.allow_tf32 is True
    assert torch_mod.backends.cudnn.allow_tf32 is True
    assert torch_mod.get_float32_matmul_precision() == "high"

    torch_mod = _fake_torch()
    with pytest.raises(RuntimeError, match="body failed"):
        with fp32_matmul_context(torch_mod):
            raise RuntimeError("body failed")
    assert torch_mod.backends.cuda.matmul.allow_tf32 is True
    assert torch_mod.backends.cudnn.allow_tf32 is True
    assert torch_mod.get_float32_matmul_precision() == "high"

    torch_mod = _fake_torch()
    torch_mod.backends.cudnn.fail_on_false = True
    with pytest.raises(RuntimeError, match="partial apply"):
        with fp32_matmul_context(torch_mod):
            raise AssertionError("context must not yield after a partial apply")
    assert torch_mod.backends.cuda.matmul.allow_tf32 is True
    assert torch_mod.backends.cudnn.allow_tf32 is True
    assert torch_mod.get_float32_matmul_precision() == "high"
