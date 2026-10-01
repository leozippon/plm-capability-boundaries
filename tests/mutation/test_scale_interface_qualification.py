"""CPU-stub contracts for ProGen2 scale-interface qualification.

Nothing here loads released weights. The stage is an external baseline, not a
panel stage: a pass is interface availability, not a capability or knowledge
result.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts/capability") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts/capability"))

from src.capability.core import arms as A  # noqa: E402
from src.capability.core.arms import (  # noqa: E402
    LOADING_INFO_KEYS,
    PANEL,
    Arm,
    ArmSpec,
    load_arm,
    load_arm_spec,
    unpack_pretrained_loading_info,
    require_clean_loading_info,
)


from scripts.capability.entrypoints import stage_path

def _load_stage(filename: str):
    path = stage_path(REPO_ROOT, filename)
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load stage module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE = _load_stage("scale_interface_qualification.py")

PROGEN_TOKENS: tuple[str, ...] = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "1",
    "2",
    *"ABCDEFGHIKLMNOPQRSTUVWXYZ",
    "<|endoftext|>",
)


class _ResidueTokenizer:
    """One token per character, matching ProGen2's residue tokenizer shape."""

    pad_token = "<|endoftext|>"
    eos_token = "<|endoftext|>"
    unk_token_id = len(PROGEN_TOKENS) - 1

    def __init__(self) -> None:
        self._ids = {token: index for index, token in enumerate(PROGEN_TOKENS)}

    def __call__(self, text, return_tensors=None, **_kwargs):
        return {"input_ids": [self._ids[character] for character in text]}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids.get(token, self.unk_token_id)


class _Head:
    def __init__(self, width: int) -> None:
        self.out_features = width


class _StubLM(torch.nn.Module):
    """Deterministic logits: native marker peaks on the next token, wrong does not."""

    def __init__(self, width: int, *, native_marker: int, vocab_size=None) -> None:
        super().__init__()
        self.lm_head = _Head(width)
        self.width = width
        self.native_marker = native_marker
        self.config = (
            SimpleNamespace(vocab_size=vocab_size)
            if vocab_size is not None
            else SimpleNamespace()
        )

    def forward(self, input_ids, **_kwargs):
        batch, steps = input_ids.shape
        logits = torch.zeros(batch, steps, self.width)
        marker = int(input_ids[0, 0])
        if marker == self.native_marker:
            for index in range(steps - 1):
                logits[0, index, int(input_ids[0, index + 1])] = 8.0
        return SimpleNamespace(logits=logits)


def _stub_arm(name: str, width: int | None = None) -> Arm:
    spec = A.arm_spec(name)
    live_width = int(width if width is not None else STAGE.REQUIRED_LIVE_WIDTH[name])
    tokenizer = _ResidueTokenizer()
    vocab_size = 32 if name == "progen2-medium" else (51200 if name == "progen2-large" else None)
    model = _StubLM(
        live_width,
        native_marker=tokenizer._ids["1"],
        vocab_size=vocab_size,
    )
    # The stub stands in for a strict load, so it carries the counts
    # require_clean_loading_info returns for a clean loading-info report rather
    # than a hand-written zero block: what the artefact records has to be what a
    # check produced.
    return Arm(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        dtype="float32",
        strict_load=require_clean_loading_info(_empty_info(), arm=name),
    )


def _empty_info() -> dict[str, list]:
    return {key: [] for key in LOADING_INFO_KEYS}


def _spec(path: Path) -> ArmSpec:
    return ArmSpec(
        name="progen2-medium",
        path=path,
        path_variable="MODEL_ROOT",
        modality="protein",
        n_layer=27,
        d_model=1536,
        tokenisation="residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="swissprot",
        architecture="progen",
        pretraining_corpus="uniref90_bfd30",
    )


class _Cfg:
    def __init__(self, n_layer: int = 27, n_embd: int = 1536) -> None:
        self.n_layer = n_layer
        self.n_embd = n_embd
        self._attn_implementation = None


class _DummyModel(torch.nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, dtype=dtype))
        self.config = _Cfg()


class _DummyTokenizer:
    pad_token = None
    eos_token = "</s>"


# ------------------------------------------------------------- frozen surface


def test_stage_is_not_a_registered_panel_stage():
    from scripts.capability import panel_contract

    stage_file = STAGE.__file__
    assert isinstance(stage_file, str)
    assert Path(stage_file).name == "scale_interface_qualification.py"
    assert all(
        contract.entry_point != "scale_interface_qualification.py"
        for contract in panel_contract.STAGE_CONTRACTS.values()
    )
    assert "scale_interface_qualification" not in panel_contract.STAGE_CONTRACTS


def test_arms_are_the_frozen_set_and_are_not_new_panel_members():
    assert STAGE.SCALE_INTERFACE_ARMS == (
        "progen2-medium",
        "progen2-large",
        "progen2-xlarge",
    )
    assert "progen2-large" not in PANEL
    assert "progen2-xlarge" not in PANEL
    assert STAGE.NATIVE_MARKER == "1"
    assert STAGE.WRONG_MARKER == "2"
    assert len(STAGE.FIXED_SEQUENCE) == 74
    assert STAGE.sequence_digest(STAGE.FIXED_SEQUENCE) == STAGE.FIXED_SEQUENCE_SHA256


def test_cli_exposes_only_device_dtype_and_out():
    dests = {action.dest for action in STAGE.build_parser()._actions}
    assert dests == {"help", "device", "dtype", "out"}


def test_wrong_arm_order_or_set_is_refused():
    with pytest.raises(ValueError, match="fixed as"):
        STAGE.require_qualification_arms(
            ["progen2-large", "progen2-medium", "progen2-xlarge"]
        )
    with pytest.raises(ValueError, match="fixed as"):
        STAGE.require_qualification_arms(
            ["progen2-small", "progen2-medium", "progen2-large"]
        )


def test_a_mutated_sequence_is_refused():
    with pytest.raises(ValueError, match="frozen"):
        STAGE.require_fixed_sequence(STAGE.FIXED_SEQUENCE[:-1] + "A")


# ------------------------------------------------------------- marker alignment


def test_wrong_marker_keeps_residue_targets_and_does_not_reverse():
    tokenizer = _ResidueTokenizer()
    rendered = STAGE.aligned_marker_targets(
        tokenizer, sequence=STAGE.FIXED_SEQUENCE
    )
    assert rendered["native_marker_id"] != rendered["wrong_marker_id"]
    assert rendered["target_ids"] == rendered["native_token_ids"][1:]
    assert rendered["target_ids"] == rendered["wrong_token_ids"][1:]
    assert len(rendered["target_ids"]) == 74
    assert STAGE.native_input(STAGE.FIXED_SEQUENCE) == "1" + STAGE.FIXED_SEQUENCE
    assert STAGE.wrong_marker_input(STAGE.FIXED_SEQUENCE) == "2" + STAGE.FIXED_SEQUENCE
    reversed_ids = tokenizer("2" + STAGE.FIXED_SEQUENCE[::-1])["input_ids"][1:]
    assert reversed_ids != rendered["target_ids"]


def test_a_split_marker_or_shared_marker_id_is_refused():
    class _Split:
        def __call__(self, text, return_tensors=None, **_kwargs):
            if text in {"1", "2"}:
                return {"input_ids": [3, 3]}
            return {"input_ids": [3, 10, 11]}

    with pytest.raises(ValueError, match="not one"):
        STAGE.require_single_marker_token(_Split(), "1", label="native")

    class _Same:
        def __call__(self, text, return_tensors=None, **_kwargs):
            if text in {"1", "2"}:
                return {"input_ids": [7]}
            return {"input_ids": [7, 10, 11]}

    with pytest.raises(ValueError, match="same id"):
        STAGE.aligned_marker_targets(_Same(), sequence="MK")


# ------------------------------------------------------------- thresholds


def test_native_repeat_threshold_is_inclusive_at_one_e_minus_six():
    ids = [1, 2, 3]
    limit = STAGE.NATIVE_REPEAT_MAX_ABS
    STAGE.require_native_repeat([0.0, 0.2], [limit, 0.2], ids, ids)
    with pytest.raises(ValueError, match="repeat"):
        STAGE.require_native_repeat(
            [0.0, 0.2], [math.nextafter(limit, math.inf), 0.2], ids, ids
        )
    with pytest.raises(ValueError, match="disagree"):
        STAGE.require_native_repeat([0.1], [0.1], [1], [2])
    with pytest.raises(ValueError, match="non-finite"):
        STAGE.require_native_repeat([math.nan], [0.0], [1], [1])


def test_wrong_marker_cost_is_strictly_above_0_05():
    assert STAGE.require_wrong_marker_cost(0.05 + 1e-12) > 0.05
    with pytest.raises(ValueError, match="strictly"):
        STAGE.require_wrong_marker_cost(0.05)
    with pytest.raises(ValueError, match="strictly"):
        STAGE.require_wrong_marker_cost(0.049999)
    with pytest.raises(ValueError, match="strictly"):
        STAGE.require_wrong_marker_cost(0.0)


# ------------------------------------------------------------- width vs support


def test_live_width_and_scoring_support_are_not_interchangeable():
    STAGE.require_support_and_live_width("progen2-medium", 32, 32)
    STAGE.require_support_and_live_width("progen2-large", 32, 51200)
    STAGE.require_support_and_live_width("progen2-xlarge", 32, 32)
    with pytest.raises(ValueError, match="scoring-target support"):
        STAGE.require_support_and_live_width("progen2-large", 51200, 51200)
    with pytest.raises(ValueError, match="live output width"):
        STAGE.require_support_and_live_width("progen2-large", 32, 32)
    with pytest.raises(ValueError, match="live output width"):
        STAGE.require_support_and_live_width("progen2-medium", 32, 51200)
    with pytest.raises(ValueError, match="live output width"):
        STAGE.require_support_and_live_width("progen2-xlarge", 32, 51200)


def test_required_live_widths_match_the_released_progen2_heads():
    """The released heads, read off the checkpoints rather than off a config key.

    ``progen2-large`` emits 51200 columns; ``progen2-xlarge`` emits 32, which is
    what EXP-R2-068 recorded from a real forward pass. Freezing 51200 for xlarge
    would make the qualification unpassable on the checkpoint it names.
    """

    assert STAGE.REQUIRED_LIVE_WIDTH == {
        "progen2-medium": 32,
        "progen2-large": 51200,
        "progen2-xlarge": 32,
    }


def test_cropped_logits_are_refused_against_a_51200_live_width():
    logits = torch.zeros(1, 3, 32)
    with pytest.raises(ValueError, match="cropping is forbidden"):
        STAGE.require_uncropped_logits(logits, 51200, arm="progen2-large")
    STAGE.require_uncropped_logits(torch.zeros(1, 3, 51200), 51200, arm="progen2-large")


def test_full_width_nll_gathers_over_the_declared_last_dimension():
    logits = torch.zeros(1, 3, 8)
    logits[0, 0, 2] = 5.0
    logits[0, 1, 4] = 5.0
    ids = torch.tensor([[1, 2, 4]])
    nll = STAGE.full_width_target_nll(logits, ids)
    expected = -torch.log_softmax(logits[:, :-1, :].float(), dim=-1).gather(
        -1, ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    assert torch.allclose(nll, expected)
    assert nll.shape[-1] == 2


# ------------------------------------------------------------- strict loading


def test_strict_loading_info_refuses_each_nonempty_list():
    assert require_clean_loading_info(_empty_info(), arm="progen2-medium") == {
        key: 0 for key in LOADING_INFO_KEYS
    }
    for key in LOADING_INFO_KEYS:
        dirty = _empty_info()
        dirty[key] = ["x"]
        with pytest.raises(ValueError, match=key):
            require_clean_loading_info(dirty, arm="progen2-large")


def test_strict_loading_info_refuses_a_wrong_api_shape():
    model = object()
    with pytest.raises(TypeError, match="\\(model, loading_info\\)"):
        unpack_pretrained_loading_info(model)
    with pytest.raises(TypeError, match="length 1"):
        unpack_pretrained_loading_info((model,))
    with pytest.raises(TypeError, match="length 3"):
        unpack_pretrained_loading_info((model, _empty_info(), {}))
    with pytest.raises(TypeError, match="dict"):
        unpack_pretrained_loading_info((model, ["missing_keys"]))
    with pytest.raises(TypeError, match="missing"):
        unpack_pretrained_loading_info((model, {"missing_keys": []}))


def test_load_arm_spec_strict_defaults_to_false_and_load_arm_does_not_opt_in():
    params = inspect.signature(load_arm_spec).parameters
    assert params["strict"].default is False
    assert params["strict"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "strict" not in inspect.signature(load_arm).parameters


def test_default_load_arm_spec_does_not_request_loading_info(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    captured: dict[str, object] = {}

    def fake_from_pretrained(_path, **kwargs):
        captured.update(kwargs)
        return _DummyModel(torch.float32)

    monkeypatch.setattr(
        A.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: _Cfg()
    )
    monkeypatch.setattr(
        A.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(
        A.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _DummyTokenizer(),
    )
    arm = load_arm_spec(spec, device="cpu", dtype="float32")
    assert "output_loading_info" not in captured
    assert arm.dtype == "float32"
    assert arm.spec is spec
    assert arm.strict_load is None


def test_strict_load_arm_spec_requests_loading_info_and_refuses_a_bare_model(
    tmp_path, monkeypatch
):
    spec = _spec(tmp_path)
    captured: dict[str, object] = {}

    def fake_pair(_path, **kwargs):
        captured.update(kwargs)
        return _DummyModel(torch.float32), _empty_info()

    monkeypatch.setattr(
        A.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: _Cfg()
    )
    monkeypatch.setattr(A.AutoModelForCausalLM, "from_pretrained", fake_pair)
    monkeypatch.setattr(
        A.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _DummyTokenizer(),
    )
    arm = load_arm_spec(spec, device="cpu", dtype="float32", strict=True)
    assert captured["output_loading_info"] is True
    assert arm.name == "progen2-medium"
    assert arm.strict_load == {key: 0 for key in LOADING_INFO_KEYS}

    monkeypatch.setattr(
        A.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: _DummyModel(torch.float32),
    )
    with pytest.raises(TypeError, match="\\(model, loading_info\\)"):
        load_arm_spec(spec, device="cpu", dtype="float32", strict=True)


def test_load_arm_still_calls_load_arm_spec_without_strict(monkeypatch):
    seen: dict[str, object] = {}

    def fake(spec, device="cuda:0", dtype="bfloat16", attn_implementation=None, *, strict=False):
        seen["strict"] = strict
        seen["name"] = spec.name
        return SimpleNamespace()

    monkeypatch.setattr(A, "load_arm_spec", fake)
    load_arm("progen2-medium", device="cpu", dtype="float32")
    assert seen["strict"] is False
    assert seen["name"] == "progen2-medium"


# ------------------------------------------------------------- artefact write


def test_failure_does_not_leave_the_success_artefact(tmp_path):
    dest = tmp_path / STAGE.SUCCESS_ARTEFACT
    dest.write_text("stale", encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise RuntimeError("forced failure")

    with pytest.raises(RuntimeError, match="forced failure"):
        STAGE.run_qualification(
            device="cpu",
            dtype="float32",
            out=tmp_path,
            load_fn=lambda name, *, device, dtype: _stub_arm(name),
            score_fn=boom,
        )
    assert not dest.exists()


def test_a_second_arm_failure_stops_and_writes_nothing(tmp_path):
    seen: list[str] = []
    released: list[str | None] = []

    def load(name, *, device, dtype):
        seen.append(name)
        if name == "progen2-large":
            raise RuntimeError("stop at large")
        return _stub_arm(name)

    def release(arm):
        released.append(None if arm is None else arm.name)
        STAGE.release_arm(arm)

    with pytest.raises(RuntimeError, match="stop at large"):
        STAGE.run_qualification(
            device="cpu",
            dtype="float32",
            out=tmp_path,
            load_fn=load,
            release_fn=release,
        )
    assert seen == ["progen2-medium", "progen2-large"]
    assert released == ["progen2-medium", None]
    assert not (tmp_path / STAGE.SUCCESS_ARTEFACT).exists()


def test_write_success_artefact_refuses_a_non_pass_payload(tmp_path):
    with pytest.raises(ValueError, match="non-PASS"):
        STAGE.write_success_artefact(tmp_path, {"verdict": "FAIL"})
    assert not (tmp_path / STAGE.SUCCESS_ARTEFACT).exists()


def test_stubbed_ladder_writes_the_required_pass_artefact(tmp_path):
    path = STAGE.run_qualification(
        device="cpu",
        dtype="float32",
        out=tmp_path,
        load_fn=lambda name, *, device, dtype: _stub_arm(name),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert payload["schema_version"] == STAGE.SCHEMA_VERSION
    assert payload["fixed_sequence_sha256"] == STAGE.FIXED_SEQUENCE_SHA256
    assert payload["logits_not_cropped"] is True
    assert payload["not_panel_admission"] is True
    assert payload["descriptive_not_causal"] is True
    assert payload["no_knowledge"] is True
    assert "capability" in payload["pass_means"]
    assert list(payload["arms"]) == list(STAGE.SCALE_INTERFACE_ARMS)
    assert set(payload["per_arm"]) == set(STAGE.SCALE_INTERFACE_ARMS)
    for name, expected_width in STAGE.REQUIRED_LIVE_WIDTH.items():
        row = payload["per_arm"][name]
        assert row["verdict"] == "PASS"
        assert row["strict_load"] == {key: 0 for key in LOADING_INFO_KEYS}
        assert row["scoring_target_support"]["size"] == 32
        assert row["live_output_width"]["size"] == expected_width
        assert row["target_count"] == 74
        assert row["native_marker_id"] != row["wrong_marker_id"]
        assert row["wrong_marker_cost_nats_per_target"] > 0.05
        assert row["native_repeat_max_abs_diff"] <= 1e-6
        assert len(row["native_nll_run_1"]) == 74
        assert len(row["native_nll_run_2"]) == 74
        assert len(row["wrong_nll"]) == 74
        if name == "progen2-medium":
            assert row["scoring_target_support"]["source"] == A.SCORING_TARGET_ALPHABET_CONFIG
        else:
            assert (
                row["scoring_target_support"]["source"]
                == A.SCORING_TARGET_ALPHABET_DECLARED
            )


def test_qualify_loaded_arm_refuses_an_arm_that_did_not_load_strictly():
    arm = _stub_arm("progen2-medium")
    arm.strict_load = None
    with pytest.raises(ValueError, match="not loaded strictly"):
        STAGE.qualify_loaded_arm(arm)


def test_an_injected_loader_cannot_smuggle_a_non_strict_arm_into_the_artefact(tmp_path):
    def load(name, *, device, dtype):
        arm = _stub_arm(name)
        arm.strict_load = None
        return arm

    with pytest.raises(ValueError, match="not loaded strictly"):
        STAGE.run_qualification(
            device="cpu", dtype="float32", out=tmp_path, load_fn=load
        )
    assert not (tmp_path / STAGE.SUCCESS_ARTEFACT).exists()


def test_qualify_loaded_arm_refuses_a_width_support_swap():
    arm = _stub_arm("progen2-large", width=32)
    with pytest.raises(ValueError, match="live output width"):
        STAGE.qualify_loaded_arm(arm)
