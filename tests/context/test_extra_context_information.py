"""Doors for the extra context-information campaign: ProGen3, RITA, joint wrapper.

Interface implemented is not an 8-block identification interval. These tests
load no released multi-billion weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts/capability") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts/capability"))

from src.capability.core.arms import (  # noqa: E402
    PANEL,
    PROGEN3_ARMS,
    STAGED_ARMS,
    STAGED_CANDIDATE_ARMS,
    STAGED_PROGEN3_ARMS,
    STAGED_SCALE_ARMS,
    STAGED_SECOND_STAGE_ARMS,
    Arm,
    arm_spec,
    load_arm,
    load_arm_spec,
    tokenize_batch,
)


from scripts.capability.entrypoints import stage_path

def _load_stage(filename: str):
    path = stage_path(REPO_ROOT, filename)
    name = path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _closed(**extra):
    args = dict(
        kind="protein",
        with_ec=False,
        allow_staged_scale_arms=False,
        allow_second_stage_arms=False,
        allow_candidate_arms=False,
        allow_progen3_arms=False,
        skip_truncation=False,
        dtype="bfloat16",
    )
    args.update(extra)
    return argparse.Namespace(**args)


def test_progen3_is_a_fourth_door_not_panel_or_staged_arms():
    assert STAGED_PROGEN3_ARMS == ("progen3-112m", "progen3-3b")
    for name in STAGED_PROGEN3_ARMS:
        assert name in PROGEN3_ARMS
        assert name not in PANEL
        assert name not in STAGED_ARMS
        assert name not in STAGED_SCALE_ARMS
        assert name not in STAGED_SECOND_STAGE_ARMS
        assert name not in STAGED_CANDIDATE_ARMS
        assert arm_spec(name) is PROGEN3_ARMS[name]
        with pytest.raises(KeyError):
            load_arm(name, device="cpu")


def test_load_arm_spec_refuses_progen3_architecture():
    with pytest.raises(ValueError, match="load_progen3"):
        load_arm_spec(PROGEN3_ARMS["progen3-112m"], device="cpu", dtype="bfloat16")


def test_tokenize_batch_refuses_progen3():
    spec = PROGEN3_ARMS["progen3-112m"]
    arm = Arm(spec=spec, model=object(), tokenizer=object(), device="cpu", dtype="float32")
    with pytest.raises(ValueError, match="ProGen3BatchPreparer"):
        tokenize_batch(arm, ["ACDE"], max_len=16)


def test_cohort_power_progen3_opt_in_and_truncation_guard():
    stage = _load_stage("cohort_power.py")
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms(["progen3-112m"], _closed())
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms(["progen3-112m"], _closed(allow_candidate_arms=True))
    with pytest.raises(ValueError, match="skip-truncation"):
        stage.validate_arms(
            ["progen3-112m"], _closed(allow_progen3_arms=True, skip_truncation=False)
        )
    stage.validate_arms(
        ["progen3-112m", "progen3-3b"],
        _closed(allow_progen3_arms=True, skip_truncation=True),
    )
    record = stage._progen3_record(["progen3-112m"], True)
    assert record["not_panel_admission"] is True
    assert record["scoring_directions"] == ["n_to_c"]
    assert record["measured_progen3_arms"] == ["progen3-112m"]
    contract = stage._cohort_power_stage_contract(["progen3-112m"])
    assert contract["not_panel_admission"] is True
    assert contract["measured"] == []
    assert "progen3-112m" in contract["measured_staged_arms"]


def test_rita_xl_is_admitted_on_native_encoding_not_skipped_for_empty_capabilities():
    stage = _load_stage("cohort_power.py")
    with pytest.raises(ValueError, match="skip-truncation"):
        stage.validate_arms(
            ["rita-xl"],
            _closed(allow_second_stage_arms=True, skip_truncation=False, dtype="float32"),
        )
    with pytest.raises(ValueError, match="float32"):
        stage.validate_arms(
            ["rita-xl"],
            _closed(allow_second_stage_arms=True, skip_truncation=True, dtype="bfloat16"),
        )
    stage.validate_arms(
        ["rita-xl"],
        _closed(allow_second_stage_arms=True, skip_truncation=True, dtype="float32"),
    )
    stage.validate_arms(
        ["proteinglm-7b-clm"],
        _closed(allow_second_stage_arms=True, skip_truncation=True, dtype="float32"),
    )


def test_joint_wrapper_resolves_declared_targets_and_refuses_unknown():
    stage = _load_stage("joint_context_information.py")
    args = argparse.Namespace(
        joint_target="galactica-1.3b",
        checkpoint=None,
        rendering=None,
        joint_mode="protein",
        arm_name=None,
    )
    checkpoint, rendering, arm_name = stage.resolve_checkpoint(args)
    assert checkpoint.name == "galactica-1.3b"
    assert rendering == "galactica"
    assert arm_name == "galactica-1.3b_protein"
    args.joint_target = "llama-2-7b"
    args.joint_mode = "text"
    _, rendering, arm_name = stage.resolve_checkpoint(args)
    assert rendering == "prollama"
    assert arm_name == "llama-2-7b_text"
    args.joint_target = None
    args.checkpoint = None
    args.rendering = None
    with pytest.raises(ValueError, match="joint-target"):
        stage.resolve_checkpoint(args)
