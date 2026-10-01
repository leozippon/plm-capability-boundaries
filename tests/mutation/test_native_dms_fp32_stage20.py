"""CPU contracts for stage-20 galactica-fp32-v2 scoring.

No real weights, no GPU, no network. Scorers reuse the Galactica CPU fixtures.
v1 sampling, default arms, and payload fields stay unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core import precision_policy as PP  # noqa: E402
from tests.models.test_galactica_fitness import (  # noqa: E402
    _RuleLogits,
    _residue_model,
    _scorer,
)


from scripts.capability.entrypoints import stage_path

def _load_stage(filename: str):
    path = stage_path(REPO_ROOT, filename)
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE20 = _load_stage("retrieval_bound.py")
_MISSING = object()


def _prepare_cell(tmp_path: Path, assays: tuple[str, ...] = ("assay_a", "assay_b")):
    gym = tmp_path / "gym"
    gym.mkdir(exist_ok=True)
    for name in assays:
        (gym / f"{name}.csv").write_text(
            "mutant,mutated_sequence,DMS_score,DMS_score_bin\nM1A,AKT,0.1,1\n",
            encoding="utf-8",
        )
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    catalogue = {
        "assay_to_wildtype": {name: f"q{index:05d}" for index, name in enumerate(assays)},
    }
    (out / "wildtypes.json").write_text(
        json.dumps(catalogue, indent=2) + "\n", encoding="utf-8"
    )
    return gym, out, catalogue


def _args(
    tmp_path: Path,
    *,
    protocol_id: object = STAGE20.GALACTICA_FP32_V2,
    dtype: str = "float32",
    arms: list[str] | None = None,
    stages: object = _MISSING,
    **kwargs: object,
) -> SimpleNamespace:
    gym, out, _catalogue = _prepare_cell(tmp_path)
    namespace: dict[str, object] = {
        "arms": ["galactica-1.3b"] if arms is None else arms,
        "assays": ["assay_a", "assay_b"],
        "variants": 1000,
        "seed": 20260807,
        "batch_size": 2,
        "dtype": dtype,
        "device": "cpu",
        "proteingym_dir": gym,
        "out": out,
    }
    if protocol_id is not _MISSING:
        namespace["protocol_id"] = protocol_id
    if stages is not _MISSING:
        namespace["stages"] = stages
    namespace.update(kwargs)
    return SimpleNamespace(**namespace)


def _install_assays(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_load_assay(name, *, n, seed, directory=None):
        calls.append({"name": name, "n": n, "seed": seed, "directory": directory})
        if name == "assay_b":
            sequence = "ACDEFGHIKLMNPQRSTVWY"
            return SimpleNamespace(
                name=name,
                mutants=["A1C", "A1D"],
                sequences=[sequence, "C" + sequence[1:]],
                scores=np.array([0.1, 0.9], dtype=np.float64),
                wildtype=sequence,
            )
        return SimpleNamespace(
            name=name,
            mutants=["M1A", "K3T"],
            sequences=["AKT", "AMT"],
            scores=np.array([0.2, 0.8], dtype=np.float64),
            wildtype="MKT",
        )

    monkeypatch.setattr(STAGE20, "load_assay", fake_load_assay)
    return calls


def _install_stub_scorer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dtype_observed: list[str] | None = None,
    fail_load: bool = False,
    fail_forward: bool = False,
    context: int = 64,
) -> tuple[dict[str, list], list[str]]:
    snapshots: dict[str, list] = {"load": [], "forward": []}
    released: list[str] = []

    def fake_load(arm, args):
        if fail_load:
            raise RuntimeError("load-failed")
        snapshots["load"].append(PP.snapshot_matmul_policy(torch))
        _tokenizer, model, _records = _residue_model(
            ["AKT", "AMT", "ACDEFGHIKLMNPQRSTVWY"]
        )

        class _Watch(_RuleLogits):
            def forward(self, input_ids, attention_mask=None, use_cache=None):
                snapshots["forward"].append(PP.snapshot_matmul_policy(torch))
                if fail_forward:
                    raise RuntimeError("forward-failed")
                return super().forward(
                    input_ids, attention_mask=attention_mask, use_cache=use_cache
                )

        watched = _Watch(
            int(model.vocab_size),
            start_id=model.start_id,
            end_id=model.end_id,
            pad_id=model.pad_id,
        )
        scorer = _scorer(watched, context=context, batch_size=args.batch_size)
        scorer.loaded.facts["dtype_observed"] = (
            ["float32"] if dtype_observed is None else list(dtype_observed)
        )
        scorer.loaded.facts["dtype_requested"] = args.dtype
        original_release = scorer.release

        def release() -> None:
            released.append(arm)
            original_release()

        scorer.release = release
        loader_record = {
            "checkpoint": "cpu-test-stub",
            "context": scorer.context,
            "input_format": "galactica declared protein rendering, context=None",
            "checkpoint_facts": scorer.loaded.facts,
            "scientific_role": scorer.loaded.facts["scientific_role"],
        }
        return scorer, scorer.context, loader_record

    monkeypatch.setattr(STAGE20, "_load_scorer", fake_load)
    return snapshots, released


def _score_path(args: SimpleNamespace) -> Path:
    return Path(args.out) / f"model_{args.arms[0]}.json"


def _dirty_tf32() -> dict[str, object]:
    before = PP.snapshot_matmul_policy(torch)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return before


def _restore_policy(before: dict[str, object]) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(before["cuda_matmul_allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(before["cudnn_allow_tf32"])
    torch.set_float32_matmul_precision(str(before["float32_matmul_precision"]))


@pytest.fixture
def high_policy():
    before = _dirty_tf32()
    try:
        yield PP.snapshot_matmul_policy(torch)
    finally:
        _restore_policy(before)


def test_protocol_constants_match_the_shared_module():
    assert STAGE20.NATIVE_DMS_V1 == PP.NATIVE_DMS_V1 == "native-dms-v1"
    assert STAGE20.GALACTICA_FP32_V2 == PP.GALACTICA_FP32_V2 == "galactica-fp32-v2"
    assert sorted(STAGE20.ARM_CORPUS) == ["progen2-medium", "progen3-112m", "protgpt2"]
    assert "galactica-1.3b" in STAGE20.SCOREABLE_ARMS
    assert "galactica-1.3b" not in STAGE20.ARM_CORPUS


def test_missing_protocol_id_is_v1_and_non_gal_or_bf16_v2_are_refused(tmp_path):
    bare = SimpleNamespace(arms=["protgpt2"], dtype="bfloat16")
    assert STAGE20._requested_protocol_id(bare) == STAGE20.NATIVE_DMS_V1
    assert STAGE20._require_protocol(bare) == STAGE20.NATIVE_DMS_V1

    for stages in (["analyse"], ["lookup", "score"]):
        with pytest.raises(ValueError, match="score-only"):
            STAGE20._require_protocol(_args(tmp_path, stages=stages))
    for protocol_id in (None, ""):
        with pytest.raises(ValueError, match="unknown scoring protocol"):
            STAGE20._require_protocol(_args(tmp_path, protocol_id=protocol_id))
    with pytest.raises(ValueError, match="at least one explicit Galactica"):
        STAGE20._require_protocol(_args(tmp_path, arms=[]))

    with pytest.raises(ValueError, match="float32"):
        STAGE20._require_protocol(_args(tmp_path, dtype="bfloat16", stages=["score"]))

    with pytest.raises(ValueError, match="explicit Galactica"):
        STAGE20._require_protocol(
            _args(tmp_path, arms=["rita-xl"], dtype="float32", stages=["score"])
        )
    with pytest.raises(ValueError, match="explicit Galactica"):
        STAGE20._require_protocol(
            _args(tmp_path, arms=["protgpt2"], dtype="float32", stages=["score"])
        )


def test_v2_load_and_forward_see_tf32_off_and_writes_verified_settings(
    tmp_path, monkeypatch
):
    args = _args(tmp_path, stages=["score"])
    calls = _install_assays(monkeypatch)
    snapshots, released = _install_stub_scorer(monkeypatch)
    before = _dirty_tf32()
    try:
        results = STAGE20.stage_score(args)
        after = PP.snapshot_matmul_policy(torch)
    finally:
        _restore_policy(before)

    assert after["cuda_matmul_allow_tf32"] is True
    assert after["cudnn_allow_tf32"] is True
    assert after["float32_matmul_precision"] == "high"
    assert snapshots["load"]
    assert snapshots["forward"]
    for snapshot in snapshots["load"] + snapshots["forward"]:
        PP.require_observed_policy(snapshot)
    assert released == ["galactica-1.3b"]

    path = _score_path(args)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == results["galactica-1.3b"]
    assert payload["settings"]["protocol_id"] == PP.GALACTICA_FP32_V2
    assert payload["settings"]["precision_policy"]["requested"] == (
        PP.requested_fp32_matmul_policy()
    )
    PP.require_observed_policy(payload["settings"]["precision_policy"]["observed"])
    assert set(payload["settings"]["precision_policy"]) == {"requested", "observed"}
    assert payload["loader"]["checkpoint_facts"]["dtype_observed"] == ["float32"]
    assert payload["loader"]["precision_policy_source"] == str(Path(PP.__file__).resolve())
    assert payload["settings"]["seed"] == 20260807
    assert payload["settings"]["dtype"] == "float32"
    assert [row["assay"] for row in payload["assays"]] == ["assay_a", "assay_b"]
    assert payload["assays"][0]["n_variants"] == 2
    assert len(payload["assays"][0]["csv_sha256"]) == 64
    assert "wildtypes_sha256" in payload["input_fingerprints"]
    assert [item["seed"] for item in calls] == [20260807, 20260808]
    assert [item["n"] for item in calls] == [1000, 1000]


def test_v1_does_not_apply_v2_policy_and_keeps_seed_index_including_skips(
    tmp_path, monkeypatch
):
    args = _args(tmp_path, protocol_id=_MISSING)
    assert not hasattr(args, "protocol_id")
    calls = _install_assays(monkeypatch)
    snapshots, released = _install_stub_scorer(monkeypatch, context=8)
    before = _dirty_tf32()
    try:
        results = STAGE20.stage_score(args)
    finally:
        _restore_policy(before)

    assert snapshots["load"]
    for snapshot in snapshots["load"] + snapshots["forward"]:
        assert snapshot["cuda_matmul_allow_tf32"] is True
        assert snapshot["float32_matmul_precision"] == "high"
    payload = results["galactica-1.3b"]
    assert "protocol_id" not in payload["settings"]
    assert "precision_policy" not in payload["settings"]
    assert "precision_policy_source" not in payload["loader"]
    assert [row["assay"] for row in payload["assays"]] == ["assay_a"]
    assert payload["skipped"][0]["assay"] == "assay_b"
    assert [item["seed"] for item in calls] == [20260807, 20260808]
    assert released == ["galactica-1.3b"]
    written = json.loads(_score_path(args).read_text(encoding="utf-8"))
    assert written["settings"]["seed"] == 20260807
    assert "protocol_id" not in written["settings"]


def test_v2_load_forward_and_restore_failures_release_and_write_nothing(
    tmp_path, monkeypatch, high_policy
):
    calls = _install_assays(monkeypatch)

    args = _args(tmp_path, stages=["score"])
    snapshots, released = _install_stub_scorer(monkeypatch, fail_load=True)
    with pytest.raises(RuntimeError, match="load-failed"):
        STAGE20.stage_score(args)
    assert released == []
    assert PP.snapshot_matmul_policy(torch) == high_policy
    assert not _score_path(args).exists()
    assert calls == []

    args = _args(tmp_path, stages=["score"])
    snapshots, released = _install_stub_scorer(monkeypatch, fail_forward=True)
    with pytest.raises(RuntimeError, match="forward-failed"):
        STAGE20.stage_score(args)
    assert snapshots["forward"]
    assert PP.snapshot_matmul_policy(torch) == high_policy
    PP.require_observed_policy(snapshots["forward"][0])
    assert released == ["galactica-1.3b"]
    assert not _score_path(args).exists()

    args = _args(tmp_path, stages=["score"])
    snapshots, released = _install_stub_scorer(
        monkeypatch, dtype_observed=["bfloat16"]
    )
    with pytest.raises(ValueError, match="dtype_observed"):
        STAGE20.stage_score(args)
    assert PP.snapshot_matmul_policy(torch) == high_policy
    assert released == ["galactica-1.3b"]
    assert not _score_path(args).exists()

    real_context = PP.fp32_matmul_context

    @contextmanager
    def restore_fails(torch_mod):
        ctx = real_context(torch_mod)
        value = ctx.__enter__()
        try:
            yield value
        finally:
            ctx.__exit__(None, None, None)
            raise RuntimeError("restore-failed")

    args = _args(tmp_path, stages=["score"])
    _install_stub_scorer(monkeypatch)
    monkeypatch.setattr(PP, "fp32_matmul_context", restore_fails)
    with pytest.raises(RuntimeError, match="restore-failed"):
        STAGE20.stage_score(args)
    assert PP.snapshot_matmul_policy(torch) == high_policy
    assert not _score_path(args).exists()
