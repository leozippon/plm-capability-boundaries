"""Stage-20 native ProteinGym door for progen2-small and progen2-base.

The gap is SCOREABLE_ARMS / corpus_record wiring, not a missing scorer.
These tests do not load weights, do not mint a cohort, and do not produce ρ.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.capability.core.arms import PANEL, UNIREF90_BFD30_INCOMPLETE_SEARCH, arm_spec
from src.capability.models.scale_comparison import STRATUM_N_TO_C

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _stage():
    path = REPO_ROOT / "scripts" / "capability" / "stages/retrieval_bound.py"
    spec = importlib.util.spec_from_file_location("_stage_20_gap", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GAP_ARMS = ("progen2-small", "progen2-base")
DEFAULT_ARMS = ["progen2-medium", "progen3-112m", "protgpt2"]


def test_gap_corpus_is_explicit_and_does_not_widen_the_default_run():
    stage = _stage()
    assert sorted(stage.ARM_CORPUS) == DEFAULT_ARMS
    assert sorted(stage.NATIVE_PROTEIN_GAP_CORPUS) == ["progen2-base", "progen2-small"]
    for name in GAP_ARMS:
        assert name in stage.SCOREABLE_ARMS
        assert name not in stage.ARM_CORPUS
        assert name in PANEL
    parser = stage.build_parser()
    assert parser.parse_args([]).arms == DEFAULT_ARMS
    assert parser.parse_args(["--arms", "progen2-small"]).arms == ["progen2-small"]
    assert parser.parse_args(["--arms", "progen2-base"]).arms == ["progen2-base"]


def test_unknown_arm_is_still_refused_by_parser_and_corpus_record():
    stage = _stage()
    parser = stage.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--arms", "not-a-registered-arm"])
    with pytest.raises(KeyError, match="unknown arm"):
        stage.corpus_record("not-a-registered-arm")
    for name in ("qwen2.5-7b", "gpt2"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--arms", name])
        with pytest.raises(KeyError):
            stage.corpus_record(name)
        assert name not in stage.SCOREABLE_ARMS


def test_small_reuses_the_uniref90_bfd_incomplete_search_bound():
    stage = _stage()
    record = stage.corpus_record("progen2-small")
    assert record["declared"] == arm_spec("progen2-small").pretraining_corpus
    assert record["declared"] == "uniref90_bfd30"
    assert record["identification"] == "lower bound on support"
    assert record["note"] == UNIREF90_BFD30_INCOMPLETE_SEARCH


def test_base_mixture_is_not_a_uniref90_bfd_retrieval_bound():
    stage = _stage()
    record = stage.corpus_record("progen2-base")
    assert record["declared"] == arm_spec("progen2-base").pretraining_corpus
    assert record["declared"] == "progen2_base_mixture"
    assert record["identification"] == (
        "external UniRef50 profile baseline, not a retrieval bound"
    )
    assert "not independently identified" in record["note"]
    assert "UniRef90+BFD30" in record["note"]
    assert "not a retrieval exclusion" in record["note"]
    assert "not an upper bound on retrieval" in record["note"]
    assert "residual bias is not signed" in record["note"]
    assert record["note"] != UNIREF90_BFD30_INCOMPLETE_SEARCH
    assert "lower bound on support" not in record["identification"]


def test_loader_record_context_comes_from_checkpoint_config_not_a_1024_literal(
    monkeypatch,
):
    stage = _stage()
    loaded: list[tuple[str, str]] = []

    class _DirectionMarkerTokenizer:
        """Just enough tokenizer for `rendering_marker_ids` on `n_to_c_control`.

        ProGen2 declares only ``<|pad|>``/``<|bos|>``/``<|eos|>`` special, so its
        direction marker is an ordinary vocabulary entry whose id has to be
        looked up and can come back as the unknown id. This stub answers that one
        lookup; what the test above it checks is the context record, not the id.
        """

        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            assert token == "1"
            return 3

    class _FakeArm:
        def __init__(self, name: str, n_positions: int) -> None:
            self.name = name
            self.spec = arm_spec(name)
            self.tokenizer = _DirectionMarkerTokenizer()
            self.model = SimpleNamespace(config=SimpleNamespace(n_positions=n_positions))

    def _load_arm(name, device="cpu", dtype="bfloat16"):
        loaded.append(("load_arm", name))
        n_positions = 2048 if name == "progen2-base" else 1024
        return _FakeArm(name, n_positions)

    def _load_arm_spec(spec, device="cpu", dtype="bfloat16"):
        loaded.append(("load_arm_spec", spec.name))
        raise AssertionError(f"{spec.name} is a panel member; load_arm is the door")

    monkeypatch.setattr(stage, "load_arm", _load_arm)
    monkeypatch.setattr(stage, "load_arm_spec", _load_arm_spec)
    args = argparse.Namespace(device="cpu", dtype="bfloat16", batch_size=16)
    expected = {"progen2-small": 1024, "progen2-base": 2048}
    for name, context in expected.items():
        scorer, reported, record = stage._load_scorer(name, args)
        assert type(scorer) is stage._ArmScorer
        assert scorer.scoring_stratum == STRATUM_N_TO_C
        assert reported == context
        assert record["context"] == context
        assert record["input_format"] == "n_to_c_control"
        assert record["checkpoint"] == str(arm_spec(name).path)
        if name == "progen2-base":
            assert record["context"] != 1024
    assert loaded == [
        ("load_arm", "progen2-small"),
        ("load_arm", "progen2-base"),
    ]
    source = Path(stage.__file__).read_text(encoding="utf-8")
    assert "self.context = config_context_length(self.arm.model.config)" in source
    assert "self.context = 1024" not in source
    assert 'record["context"] = 1024' not in source
