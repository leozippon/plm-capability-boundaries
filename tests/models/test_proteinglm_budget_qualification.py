"""CPU tests for ProteinGLM derived-budget FP32 qualification. No 7B weights."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts/capability") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts/capability"))

from src.capability.models import proteinglm as pglm  # noqa: E402
from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.core.arms import STAGED_ARMS, Arm  # noqa: E402
from src.capability.core.budget import ScoredTokens  # noqa: E402
from src.capability.core.io import write_json as real_write_json  # noqa: E402
from src.capability.core.precision_policy import (  # noqa: E402
    FP32_PER_TARGET_ABS,
    snapshot_matmul_policy,
)

NAME = "proteinglm-7b-clm"
MODEL_BASE_DIR = Path(os.environ.get("MODEL_BASE_DIR", ""))
_FROZEN_SOURCE = MODEL_BASE_DIR / "models_R2/proteinglm-7b-clm"


from scripts.capability.entrypoints import stage_path

def _load_script(filename: str):
    path = stage_path(REPO_ROOT, filename)
    name = path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


Q = _load_script("proteinglm_budget_qualification.py")
QUAL225 = _load_script("second_stage_interface_qualification.py")


def _require_source() -> Path:
    candidates = (STAGED_ARMS[NAME].path, _FROZEN_SOURCE)
    for path in candidates:
        if (path / "modeling_proteinglm.py").is_file() and (
            path / "tokenizer.model"
        ).is_file():
            return path
    pytest.skip("ProteinGLM source files are not staged on this host")


def _tokenizer():
    return AutoTokenizer.from_pretrained(
        str(_require_source()), trust_remote_code=True, local_files_only=True
    )


def _tiny_arm(*, num_layers: int = 1, hidden: int = 32, seq_length: int = 1024) -> Arm:
    source = _require_source()
    model = pglm.tiny_causal_lm(
        source,
        num_layers=num_layers,
        hidden_size=hidden,
        seq_length=seq_length,
    )
    spec = replace(STAGED_ARMS[NAME], n_layer=num_layers, d_model=hidden)
    return Arm(
        spec=spec,
        model=model,
        tokenizer=_tokenizer(),
        device="cpu",
        dtype="float32",
        attn_implementation=None,
    )


def test_avgfp_literal_hash_and_four_case_lengths():
    cases = Q.case_specifications()
    assert [case.name for case in cases] == [
        "minimum_two_residues",
        "canonical_aa20",
        "avgfp_n80",
        "longest_legal_1021",
    ]
    assert cases[0].sequence == "AC" and cases[0].n_targets == 1
    assert cases[1].sequence == AA20 and cases[1].n_targets == 19
    assert cases[2].sequence == Q.AVGFP_N80
    assert hashlib.sha256(cases[2].sequence.encode("utf-8")).hexdigest() == Q.AVGFP_N80_SHA256
    assert cases[2].n_residues == 80 and cases[2].n_targets == 79
    assert cases[3].n_residues == 1021 and cases[3].n_targets == 1020
    assert cases[3].sequence == (AA20 * 52)[:1021]


def test_cli_rejects_other_arm_dtype_and_tiny_flag():
    parser = Q.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--arm", "rita-xl", "--out", "x"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--arm", NAME, "--dtype", "bfloat16", "--out", "x"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--arm", NAME, "--tiny", "--out", "x"])
    args = parser.parse_args(["--out", "/tmp/out", "--device", "cuda:0", "--arm", NAME])
    assert args.dtype == "float32"
    assert args.arm == NAME


def test_old_225_still_skips_proteinglm_without_loading():
    def _explode(*args, **kwargs):
        raise AssertionError("an unavailable checkpoint must not be loaded")

    payload = QUAL225.qualify_arm(
        NAME, device="cpu", dtype="float32", load_fn=_explode
    )
    assert payload["verdict"] == "UNAVAILABLE"
    assert payload["loaded"] is False
    assert "deepspeed" in payload["reason"]


def test_cpu_device_does_not_load_7b(monkeypatch, tmp_path):
    def _explode(**kwargs):
        raise AssertionError("CPU must not load the 7B checkpoint")

    monkeypatch.setattr(Q, "load_published_arm", _explode)
    with pytest.raises(Q.QualificationFailed) as caught:
        Q.run_qualification(arm=NAME, device="cpu", dtype="float32", out=tmp_path)
    artefact = tmp_path / Q.ARTEFACT_NAME
    payload = json.loads(artefact.read_text(encoding="utf-8"))
    assert payload["status"] == Q.STATUS_FAILED
    assert payload["experiment_admitted"] is False
    assert payload["not_panel_admission"] is True
    assert payload["loaded"] is False
    assert payload["error"]["class"] == "ValueError"
    assert caught.value.path == artefact


def test_missing_strict_and_provenance_are_refused():
    arm = _tiny_arm()
    with pytest.raises(ValueError, match="strict_load"):
        Q.require_load_evidence(arm)
    with pytest.raises(ValueError, match="36L/4096d"):
        Q.require_declared_7b_shape(arm)


def test_independent_ce_rejects_wrong_prefix():
    arm = _tiny_arm(seq_length=64)
    ids = torch.tensor([[0, 0, 0, 2, 20]], dtype=torch.long)
    mask = torch.ones_like(ids)
    with pytest.raises(ValueError, match="gmask"):
        Q.independent_shifted_ce(arm, ids, mask)


def test_numeric_production_is_one_scored_tokens_call_of_four_cases(monkeypatch):
    arm = _tiny_arm()
    calls = []
    real = Q.scored_tokens

    def spy(loaded, strings, *, max_len, batch_size):
        calls.append(
            {
                "strings": list(strings),
                "max_len": int(max_len),
                "batch_size": int(batch_size),
            }
        )
        return real(loaded, strings, max_len=max_len, batch_size=batch_size)

    monkeypatch.setattr(Q, "scored_tokens", spy)
    result = Q.score_numeric_cases(arm)
    assert len(calls) == 1
    assert calls[0]["batch_size"] == 1
    assert calls[0]["max_len"] == 1024
    assert len(calls[0]["strings"]) == 4
    cases = Q.case_specifications()
    expected = [pglm.render_budget_sequence(case.sequence) for case in cases]
    assert calls[0]["strings"] == expected
    assert result["numeric_public_call_count"] == 1
    assert result["production_calls"][0]["n_strings"] == 4
    assert result["production_calls"][0]["batch_size"] == 1


def test_resource_gate_is_one_public_call_of_three_longest(monkeypatch):
    arm = _tiny_arm()
    calls = []
    real = Q.scored_tokens

    def spy(loaded, strings, *, max_len, batch_size):
        calls.append(
            {
                "strings": list(strings),
                "max_len": int(max_len),
                "batch_size": int(batch_size),
            }
        )
        return real(loaded, strings, max_len=max_len, batch_size=batch_size)

    monkeypatch.setattr(Q, "scored_tokens", spy)
    longest = Q.longest_legal_sequence()
    rendered = pglm.render_budget_sequence(longest)
    result = Q.run_resource_gate(arm, longest, record_cuda=False)
    assert len(calls) == 1
    assert calls[0]["strings"] == [rendered, rendered, rendered]
    assert calls[0]["batch_size"] == 1
    assert calls[0]["max_len"] == 1024
    assert result["resource_public_call_count"] == 1
    assert result["n_sequences"] == 3
    assert result["n_targets"] == 3060
    assert result["cuda_memory"] is None


def test_resource_gate_rejects_short_substitute():
    arm = _tiny_arm(seq_length=64)
    with pytest.raises(ValueError, match="1021-residue"):
        Q.run_resource_gate(arm, "AC", record_cuda=False)


def test_tiny_production_matches_independent_ce_and_target_accounting():
    arm = _tiny_arm()
    result = Q.score_numeric_cases(arm)
    residues = pglm.residue_token_ids(arm.tokenizer)
    records = {item["name"]: item for item in result["cases"]}
    two = records["minimum_two_residues"]
    assert two["L"] == 2 and two["n_targets"] == 1
    assert two["target_positions"] == [3]
    assert two["target_ids"] == [residues["C"]]
    aa20 = records["canonical_aa20"]
    assert aa20["n_targets"] == 19
    assert aa20["target_positions"][0] == 3
    assert aa20["target_positions"][-1] == 3 + 19 - 1
    assert aa20["target_ids"][0] == residues["C"]
    assert aa20["target_ids"][-1] == residues["Y"]
    longest = records["longest_legal_1021"]
    assert longest["n_targets"] == 1020
    assert longest["target_positions"][0] == 3
    assert longest["target_positions"][-1] == 1022
    for item in result["cases"]:
        assert item["max_abs"] <= FP32_PER_TARGET_ABS
        assert item["live_logits_width"] == 128
        assert item["logits_dtype"] == "float32"
        assert item["parameter_dtypes"] == ["float32"]
        assert len(item["production_nll_nats"]) == item["n_targets"]
        assert all(np.isfinite(item["production_nll_nats"]))
        assert all(np.isfinite(item["reference_nll_nats"]))
    assert result["independent_ce_max_abs"] <= FP32_PER_TARGET_ABS


def test_nonfinite_production_fails(monkeypatch):
    arm = _tiny_arm()
    real = Q.scored_tokens

    def boom(loaded, strings, *, max_len, batch_size):
        scored = real(loaded, strings, max_len=max_len, batch_size=batch_size)
        nll = np.array(scored.nll_nats, copy=True)
        nll[0] = np.inf
        return ScoredTokens(
            target_ids=scored.target_ids,
            nll_nats=nll,
            sequence_index=scored.sequence_index,
        )

    monkeypatch.setattr(Q, "scored_tokens", boom)
    with pytest.raises(FloatingPointError, match="non-finite"):
        Q.score_numeric_cases(arm)


def test_reference_deviation_fails(monkeypatch):
    arm = _tiny_arm()
    real = Q.independent_shifted_ce

    def drift(*args, **kwargs):
        reference = real(*args, **kwargs)
        nll = np.array(reference.nll_nats, copy=True)
        nll[0] += 1.0
        return Q.IndependentReference(
            nll_nats=nll,
            target_ids=reference.target_ids,
            target_positions=reference.target_positions,
            live_width=reference.live_width,
            logits_dtype=reference.logits_dtype,
            logits_shape=reference.logits_shape,
            parameter_dtypes=reference.parameter_dtypes,
        )

    monkeypatch.setattr(Q, "independent_shifted_ce", drift)
    with pytest.raises(ValueError, match="independent CE max abs"):
        Q.score_numeric_cases(arm)


def test_tiny_numeric_success_cannot_mint_7b_verdict():
    arm = _tiny_arm()
    numeric = Q.score_numeric_cases(arm)
    resource = Q.run_resource_gate(arm, Q.longest_legal_sequence(), record_cuda=False)
    assert numeric["independent_ce_max_abs"] <= FP32_PER_TARGET_ABS
    with pytest.raises(ValueError, match="CUDA"):
        Q.build_qualified_payload(
            arm,
            device="cpu",
            dtype="float32",
            numeric=numeric,
            resource=resource,
        )
    with pytest.raises(ValueError, match="strict_load|36L/4096d|CUDA resource"):
        Q.build_qualified_payload(
            arm,
            device="cuda:0",
            dtype="float32",
            numeric=numeric,
            resource=resource,
        )


def _failed_payload() -> dict:
    return Q.build_failed_payload(
        device="cpu",
        dtype="float32",
        error=ValueError("diagnostic"),
        loaded=False,
    )


def test_existing_artefact_entries_keep_bytes_and_skip_loader(tmp_path, monkeypatch):
    def explode(**kwargs):
        raise AssertionError("existing artefact must not load a checkpoint")

    monkeypatch.setattr(Q, "load_published_arm", explode)
    kinds = {
        "failed": None,
        "qualified": (
            b'{"status": "derived-budget-interface-qualified", '
            b'"experiment_admitted": false}\n'
        ),
        "corrupt": b"{not-json",
    }
    for kind, blob in kinds.items():
        directory = tmp_path / kind
        directory.mkdir()
        path = directory / Q.ARTEFACT_NAME
        if kind == "failed":
            Q.write_artefact(directory, _failed_payload())
        else:
            path.write_bytes(blob)
        before = path.read_bytes()
        with pytest.raises(FileExistsError, match="fresh --out"):
            Q.run_qualification(
                arm=NAME, device="cuda:0", dtype="float32", out=directory
            )
        assert path.read_bytes() == before


def test_late_publication_collision_does_not_overwrite(tmp_path, monkeypatch):
    dest = tmp_path / Q.ARTEFACT_NAME
    sneaked = b"preexisting-canonical-entry"
    real_link = Q.os.link

    def collide(source, target):
        Path(target).write_bytes(sneaked)
        return real_link(source, target)

    monkeypatch.setattr(Q.os, "link", collide)
    with pytest.raises(FileExistsError):
        Q.write_artefact(tmp_path, _failed_payload())
    assert dest.read_bytes() == sneaked
    leftovers = [path.name for path in tmp_path.iterdir() if path.name != Q.ARTEFACT_NAME]
    assert leftovers == []


def test_write_failure_leaves_no_canonical_or_temp(tmp_path, monkeypatch):
    def boom_json(path, value):
        raise OSError("injected write failure")

    monkeypatch.setattr(Q, "write_json", boom_json)
    with pytest.raises(OSError, match="injected write failure"):
        Q.write_artefact(tmp_path, _failed_payload())
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr(Q, "write_json", real_write_json)

    def boom_link(source, target):
        raise OSError("injected link failure")

    monkeypatch.setattr(Q.os, "link", boom_link)
    with pytest.raises(OSError, match="injected link failure"):
        Q.write_artefact(tmp_path, _failed_payload())
    assert list(tmp_path.iterdir()) == []


class _FakeDeviceProps:
    total_memory = 999


def _install_fake_cuda(monkeypatch, *, peak_allocated="error"):
    """Inject CUDA counters. Values are test fakes, not GPU measurements."""

    monkeypatch.setattr(Q.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(Q.torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(Q.torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(Q.torch.cuda, "memory_allocated", lambda device=None: 111)
    monkeypatch.setattr(Q.torch.cuda, "memory_reserved", lambda device=None: 222)
    monkeypatch.setattr(
        Q.torch.cuda, "get_device_properties", lambda device=None: _FakeDeviceProps()
    )
    monkeypatch.setattr(Q.torch.cuda, "empty_cache", lambda: None)
    if peak_allocated == "error":

        def boom_peak(device=None):
            raise RuntimeError("injected telemetry failure")

        monkeypatch.setattr(Q.torch.cuda, "max_memory_allocated", boom_peak)
    else:
        monkeypatch.setattr(Q.torch.cuda, "max_memory_allocated", lambda device=None: 333)
    monkeypatch.setattr(Q.torch.cuda, "max_memory_reserved", lambda device=None: 444)


def test_resource_oom_keeps_progress_and_primary_cause(monkeypatch):
    arm = _tiny_arm()
    progress: dict = {}
    real = Q.scored_tokens

    def oom_on_resource(loaded, strings, *, max_len, batch_size):
        if len(list(strings)) == 3:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory: simulated")
        return real(loaded, strings, max_len=max_len, batch_size=batch_size)

    monkeypatch.setattr(Q, "scored_tokens", oom_on_resource)
    monkeypatch.setattr(Q, "require_cuda_device", lambda device: None)
    _install_fake_cuda(monkeypatch, peak_allocated="error")
    before = snapshot_matmul_policy(torch)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="simulated"):
        Q.run_resource_gate(
            arm,
            Q.longest_legal_sequence(),
            record_cuda=True,
            progress=progress,
        )
    after = snapshot_matmul_policy(torch)
    assert after == before
    assert progress["resource_public_call_attempted"] == 1
    assert progress["resource_public_call_completed"] == 0
    assert progress.get("resource_public_call_count") != 1
    assert progress.get("finite") is not True
    assert progress["expected_n_sequences"] == 3
    assert progress["expected_n_targets"] == 3060
    cuda_memory = progress["cuda_memory"]
    assert cuda_memory["baseline_allocated_bytes"] == 111
    assert cuda_memory["baseline_reserved_bytes"] == 222
    assert cuda_memory["total_memory_bytes"] == 999
    peak = cuda_memory["peak_allocated_bytes"]
    assert peak["status"] == "unknown"
    assert peak["error"]["class"] == "RuntimeError"
    assert "injected telemetry failure" in peak["error"]["message"]


def test_qualification_oom_keeps_numeric_and_does_not_hide_cause(monkeypatch, tmp_path):
    arm = _tiny_arm()
    real = Q.scored_tokens

    def oom_on_resource(loaded, strings, *, max_len, batch_size):
        if len(list(strings)) == 3:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory: simulated")
        return real(loaded, strings, max_len=max_len, batch_size=batch_size)

    def load_fn(*, device, dtype):
        return arm

    monkeypatch.setattr(Q, "scored_tokens", oom_on_resource)
    monkeypatch.setattr(Q, "require_cuda_device", lambda device: None)
    _install_fake_cuda(monkeypatch, peak_allocated="error")
    before = snapshot_matmul_policy(torch)
    with pytest.raises(Q.QualificationFailed) as caught:
        Q.run_qualification(
            arm=NAME,
            device="cuda:0",
            dtype="float32",
            out=tmp_path,
            load_fn=load_fn,
        )
    after = snapshot_matmul_policy(torch)
    assert after == before
    payload = json.loads(caught.value.path.read_text(encoding="utf-8"))
    assert payload["status"] == Q.STATUS_FAILED
    assert payload["error"]["class"] == "OutOfMemoryError"
    assert "simulated" in payload["error"]["message"]
    assert payload["error"]["class"] != "RuntimeError"
    numeric = payload["partial"]["numeric"]
    assert len(numeric["cases"]) == 4
    assert numeric["numeric_public_call_count"] == 1
    resource = payload["partial"]["resource"]
    assert resource["resource_public_call_attempted"] == 1
    assert resource["resource_public_call_completed"] == 0
    assert resource.get("resource_public_call_count") != 1
    assert resource.get("finite") is not True
    assert resource["cuda_memory"]["baseline_allocated_bytes"] == 111
    peak = resource["cuda_memory"]["peak_allocated_bytes"]
    assert peak["status"] == "unknown"
    assert peak["error"]["class"] == "RuntimeError"


def _a4_numeric_only_complete(cuda_memory: dict) -> bool:
    """a4 helper: six numeric fields only. Documents the ignored-sync defect."""
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
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
    return True


def test_a4_complete_helper_ignored_unknown_synchronize():
    payload = {
        "synchronize": {"status": "unknown", "error": {"class": "RuntimeError", "message": "x"}},
        "baseline_allocated_bytes": 1,
        "baseline_reserved_bytes": 2,
        "total_memory_bytes": 3,
        "duration_s": 0.1,
        "peak_allocated_bytes": 4,
        "peak_reserved_bytes": 5,
    }
    assert _a4_numeric_only_complete(payload) is True
    assert Q.cuda_end_state_is_complete(payload) is False


def test_terminal_synchronize_failure_cannot_qualify(monkeypatch):
    arm = _tiny_arm()
    progress: dict = {}
    syncs = {"n": 0}

    def sync(device=None):
        syncs["n"] += 1
        if syncs["n"] > 1:
            raise RuntimeError("injected terminal synchronize failure")

    monkeypatch.setattr(Q, "require_cuda_device", lambda device: None)
    _install_fake_cuda(monkeypatch, peak_allocated="ok")
    monkeypatch.setattr(Q.torch.cuda, "synchronize", sync)
    with pytest.raises(ValueError, match="synchronize must be ok"):
        Q.run_resource_gate(
            arm,
            Q.longest_legal_sequence(),
            record_cuda=True,
            progress=progress,
        )
    cuda_memory = progress["cuda_memory"]
    assert cuda_memory["synchronize"]["status"] == "unknown"
    assert cuda_memory["baseline_allocated_bytes"] == 111
    assert isinstance(cuda_memory["duration_s"], float)
    assert cuda_memory["peak_allocated_bytes"] == 333
    assert Q.cuda_end_state_is_complete(cuda_memory) is False
    assert progress.get("resource_public_call_count") != 1
    assert progress.get("finite") is not True
    assert progress["resource_public_call_completed"] == 1
    assert progress["terminal_public_call_capture_count"] == 1


def test_reference_failure_does_not_recapture(monkeypatch):
    arm = _tiny_arm()
    progress: dict = {}
    captures = {"n": 0}
    real_capture = Q.capture_cuda_end_state

    def counting_capture(device, started):
        captures["n"] += 1
        return real_capture(device, started)

    def boom_reference(*args, **kwargs):
        raise RuntimeError("injected independent reference failure")

    monkeypatch.setattr(Q, "require_cuda_device", lambda device: None)
    monkeypatch.setattr(Q, "capture_cuda_end_state", counting_capture)
    monkeypatch.setattr(Q, "independent_shifted_ce", boom_reference)
    _install_fake_cuda(monkeypatch, peak_allocated="ok")
    with pytest.raises(RuntimeError, match="injected independent reference failure"):
        Q.run_resource_gate(
            arm,
            Q.longest_legal_sequence(),
            record_cuda=True,
            progress=progress,
        )
    assert captures["n"] == 1
    assert progress["terminal_public_call_capture_count"] == 1
    assert progress["resource_public_call_completed"] == 1
    assert progress.get("resource_public_call_count") != 1


def test_precision_enter_failure_does_not_count_public_call(monkeypatch):
    arm = _tiny_arm()
    progress: dict = {}
    public = {"n": 0}
    captures = {"n": 0}
    real_public = Q.public_scored_tokens
    real_capture = Q.capture_cuda_end_state

    def counting_public(*args, **kwargs):
        public["n"] += 1
        return real_public(*args, **kwargs)

    def counting_capture(*args, **kwargs):
        captures["n"] += 1
        return real_capture(*args, **kwargs)

    def boom_precision(arm):
        @contextmanager
        def inner():
            raise RuntimeError("injected precision enter failure")
            yield {}

        return inner()

    monkeypatch.setattr(Q, "require_cuda_device", lambda device: None)
    monkeypatch.setattr(Q, "inference_precision", boom_precision)
    monkeypatch.setattr(Q, "public_scored_tokens", counting_public)
    monkeypatch.setattr(Q, "capture_cuda_end_state", counting_capture)
    _install_fake_cuda(monkeypatch, peak_allocated="ok")
    with pytest.raises(RuntimeError, match="injected precision enter failure"):
        Q.run_resource_gate(
            arm,
            Q.longest_legal_sequence(),
            record_cuda=True,
            progress=progress,
        )
    assert progress["resource_public_call_attempted"] == 0
    assert progress["resource_public_call_completed"] == 0
    assert progress["terminal_public_call_capture_count"] == 0
    assert public["n"] == 0
    assert captures["n"] == 0
    assert progress["cuda_memory"]["baseline_allocated_bytes"] == 111
