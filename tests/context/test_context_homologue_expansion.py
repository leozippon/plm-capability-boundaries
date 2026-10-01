"""Packing and inclusion contracts for the homologous-context expansion wave.

F16 behaviour is covered by ``test_context_homologue.py``. This file holds the
lists and the id-concatenation rules the expansion arms must keep, using pinned
ids or stubs when a checkpoint is not on this host.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.context import context_homologue as ch  # noqa: E402
from src.capability.mutation import designed_referent as D  # noqa: E402
from src.capability.core.arms import Arm, ArmSpec, MODEL_ROOT  # noqa: E402
from src.capability.models.joint_modes import resolve  # noqa: E402
from tests.core.test_joint_mode_qualification import (  # noqa: E402
    SEQUENCE,
    galactica_stub,
    instructprotein_stub,
    prollama_stub,
)
from tests.models.test_joint_prefix_declaration import (  # noqa: E402
    INSTRUCTPROTEIN_BLOCK_IDS,
    LLAMA_LINEAGE_BLOCK_IDS,
)


def _spec(name: str, *, input_format: str = "raw", tokenisation: str = "residue") -> ArmSpec:
    return ArmSpec(
        name=name,
        path=Path("."),
        path_variable="MODEL_ROOT",
        modality="protein",
        n_layer=1,
        d_model=1,
        tokenisation=tokenisation,
        input_format=input_format,
        evaluation_cohort_source="swissprot",
        architecture="test",
    )


def _arm(name: str, tokenizer, *, serving_provenance=None, **kwargs) -> Arm:
    return Arm(
        spec=_spec(name, **kwargs),
        model=None,
        tokenizer=tokenizer,
        device="cpu",
        dtype="none",
        serving_provenance=serving_provenance,
    )


class _MapTokenizer:
    """A residue tokenizer with an explicit special-id map."""

    def __init__(self, *, bos=None, eos=None, unk=0, extra=None):
        self.bos_token_id = bos
        self.eos_token_id = eos
        self.unk_token_id = unk
        self._extra = dict(extra or {})

    def convert_tokens_to_ids(self, token: str) -> int:
        if token in self._extra:
            return self._extra[token]
        return self.unk_token_id

    def get_vocab(self) -> dict[str, int]:
        vocab = dict(self._extra)
        for index, residue in enumerate("ACDEFGHIKLMNPQRSTVWY"):
            vocab[residue] = 10 + index
        if self.bos_token_id is not None:
            vocab.setdefault("<bos>", int(self.bos_token_id))
        if self.eos_token_id is not None:
            vocab.setdefault("<eos>", int(self.eos_token_id))
        return vocab

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        vocab = self.get_vocab()
        return {"input_ids": [vocab[symbol] for symbol in text]}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(value)) for value in ids)


def _load_stage():
    path = REPO_ROOT / "scripts" / "capability" / "stages/context_homologue.py"
    spec = importlib.util.spec_from_file_location("stage46_context_homologue", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_expansion_names_are_protein_and_lists_hold():
    assert ch.modality_of("gpt2-large") == "text"
    assert "zymctrl" in ch.EXCLUDED_ARMS
    assert "progen2-base" not in ch.EXCLUDED_ARMS
    assert "progen2-large" not in ch.EXCLUDED_ARMS
    assert "progen2-xlarge" not in ch.EXCLUDED_ARMS
    for name in ch.EXPANSION_PROTEIN_ARMS:
        assert ch.modality_of(name) == "protein", name
        assert name in ch.PROTEIN_ARMS
        assert name in ch.ARMS
        assert name in ch.CAVEATS
    assert ch.TEXT_ARMS == ("gpt2-large",)
    assert "zymctrl" not in ch.ARMS
    with pytest.raises(ValueError, match="excluded"):
        ch.modality_of("zymctrl")


def test_identification_tables_untouched():
    assert set(D.ARM_IDENTIFICATION) == {
        "protgpt2",
        "progen2-small",
        "progen2-base",
        "progen2-medium",
        "progen2-large",
        "progen2-xlarge",
        "proteinglm-7b-clm",
        "protgpt3-1.3b",
        "rita-xl",
        "progen3-112m",
        "progen3-3b",
    }
    assert set(D.JOINT_LINEAGE_IDENTIFICATION) == {
        "llama-2-7b",
        "prollama-stage-1",
        "prollama",
    }
    assert set(D.JOINT_RENDERING_IDENTIFICATION) == {
        "galactica-125m",
        "galactica-1.3b",
        "galactica-6.7b",
        "galactica-30b",
        "instructprotein",
    }


def test_protgpt3_bos_direction_offset_is_two():
    tokenizer = _MapTokenizer(bos=1, extra={"1": 5})
    arm = _arm("protgpt3-1.3b", tokenizer, input_format="bos_direction_seq")
    ids = ch.item_ids(arm, "ACDE", modality="protein")
    assert ids[:2] == [1, 5]
    assert ch.content_offset(arm) == 2
    start, end = ch.target_span(arm, ids, record="ACDE")
    assert start == 2
    assert end == len(ids)
    assert end - start == 4


def test_rita_document_stream_has_single_eos_separators():
    tokenizer = _MapTokenizer(eos=None, extra={"<EOS>": 2})
    arm = _arm("rita-xl", tokenizer, input_format="eos_bounded_seq")
    records = ["ACDE", "FGHI", "KLMN"]
    unit = {
        "key": "u",
        "target": 2,
        "conditions": {ch.HOMOLOGUE: [["pool", 0], ["pool", 1]]},
    }
    row, start = ch.build_row(
        arm, unit, ch.HOMOLOGUE, records=records, filler=records[0], modality="protein"
    )
    assert 2 not in list(zip(row, row[1:])) or all(
        not (left == 2 and right == 2) for left, right in zip(row, row[1:])
    )
    assert row[0] == 2
    assert row.count(2) == 4
    target = ch.item_ids(arm, records[2], modality="protein")
    span0, span1 = ch.target_span(arm, target, record=records[2])
    scored = row[start : start + (span1 - span0)]
    assert 2 not in scored
    ch.packing_assertions(arm, row)


def test_galactica_multi_block_has_no_heading_and_scores_target_delimiters():
    tokenizer = galactica_stub()
    tokenisation = resolve(tokenizer, "galactica")
    arm = _arm(
        "galactica-125m",
        tokenizer,
        input_format="galactica_amino_blocks",
        serving_provenance={"joint_tokenisation": tokenisation},
    )
    records = [SEQUENCE, SEQUENCE]
    unit = {"key": "u", "target": 1, "conditions": {ch.HOMOLOGUE: [["pool", 0]]}}
    row, start = ch.build_row(
        arm, unit, ch.HOMOLOGUE, records=records, filler=SEQUENCE, modality="protein"
    )
    decoded = ch._decode_packed_row(arm, row)
    assert "# " not in decoded
    assert "Instruction:" not in decoded
    target = ch.item_ids(arm, SEQUENCE, modality="protein")
    span0, span1 = ch.target_span(arm, target, record=SEQUENCE)
    assert start == len(ch.item_ids(arm, SEQUENCE, modality="protein")) + span0
    assert row[start : start + (span1 - span0)] == target[span0:span1]
    ch.packing_assertions(arm, row)


def test_instructprotein_leading_eos_once_and_no_instruction():
    tokenizer = instructprotein_stub()
    tokenisation = resolve(tokenizer, "instructprotein")
    arm = _arm(
        "instructprotein",
        tokenizer,
        input_format="instructprotein_protein_blocks",
        serving_provenance={"joint_tokenisation": tokenisation},
    )
    records = [SEQUENCE, SEQUENCE]
    unit = {"key": "u", "target": 1, "conditions": {ch.HOMOLOGUE: [["pool", 0]]}}
    row, start = ch.build_row(
        arm, unit, ch.HOMOLOGUE, records=records, filler=SEQUENCE, modality="protein"
    )
    decoded = ch._decode_packed_row(arm, row)
    assert "Instruction:" not in decoded
    assert row[0] == 2
    assert row.count(2) == 1
    assert row[1:] != INSTRUCTPROTEIN_BLOCK_IDS
    ch.packing_assertions(arm, row)
    target = ch.item_ids(arm, SEQUENCE, modality="protein")
    assert target[0] != 2
    span0, span1 = ch.target_span(arm, target, record=SEQUENCE)
    assert row[start : start + (span1 - span0)] == target[span0:span1]


def test_llama_leading_bos_once_and_no_superfamily():
    tokenizer = prollama_stub()
    tokenisation = resolve(tokenizer, "prollama")
    arm = _arm(
        "prollama",
        tokenizer,
        input_format="llama_seq_blocks",
        tokenisation="sentencepiece",
        serving_provenance={"joint_tokenisation": tokenisation},
    )
    records = [SEQUENCE, SEQUENCE]
    unit = {"key": "u", "target": 1, "conditions": {ch.HOMOLOGUE: [["pool", 0]]}}
    row, start = ch.build_row(
        arm, unit, ch.HOMOLOGUE, records=records, filler=SEQUENCE, modality="protein"
    )
    decoded = ch._decode_packed_row(arm, row)
    assert "Superfamily=" not in decoded
    bos = int(tokenizer.bos_id)
    assert row[0] == bos
    assert row.count(bos) == 1
    ch.packing_assertions(arm, row)
    target = ch.item_ids(arm, SEQUENCE, modality="protein")
    span0, span1 = ch.target_span(arm, target, record=SEQUENCE)
    assert row[start : start + (span1 - span0)] == target[span0:span1]
    assert LLAMA_LINEAGE_BLOCK_IDS[0] == 1


def test_build_row_target_ids_identical_across_conditions():
    tokenizer = _MapTokenizer(eos=None, extra={"<EOS>": 2})
    arm = _arm("rita-xl", tokenizer)
    records = ["ACDE", "FGHI", "KLMN"]
    unit = {
        "key": "u",
        "target": 0,
        "conditions": {
            ch.HOMOLOGUE: [["pool", 1]],
            ch.UNRELATED: [["pool", 2]],
            ch.NO_CONTEXT: [],
        },
    }
    spans = set()
    for condition in (ch.HOMOLOGUE, ch.UNRELATED, ch.NO_CONTEXT):
        row, start = ch.build_row(
            arm, unit, condition, records=records, filler=records[1], modality="protein"
        )
        target = ch.item_ids(arm, records[0], modality="protein")
        span0, span1 = ch.target_span(arm, target, record=records[0])
        spans.add(tuple(row[start : start + (span1 - span0)]))
    assert len(spans) == 1


def test_analyse_skips_arms_without_plans(tmp_path):
    stage = _load_stage()
    (tmp_path / "plan_protgpt3-1.3b.json").write_text("{}", encoding="utf-8")
    cohort = {
        "digest": "abc",
        "protein": {"census": {}},
        "text": {"census": {}},
    }
    plan = {
        "digest": "plan",
        "modality": "protein",
        "units": [],
        "k_distribution": {"n": 0},
        "context_fraction": {"n": 0},
        "unrelated_token_gap": {"n": 0},
        "refusals": [],
        "rendering": {},
    }
    args = argparse.Namespace(
        cohort=tmp_path / "cohort.json",
        score_dir=tmp_path,
        bootstrap=ch.BOOTSTRAP_RESAMPLES,
        bootstrap_seed=ch.BOOTSTRAP_SEED,
    )
    with (
        patch.object(stage.ch, "load_cohort", return_value=cohort),
        patch.object(stage.ch, "load_plan", return_value=plan),
    ):
        payload = stage.run_analyse(args)
    assert set(payload["arms"]) == {"protgpt3-1.3b"}
    assert payload["arms"]["protgpt3-1.3b"]["status"].startswith("conditions not scored")
    assert payload["arms"]["protgpt3-1.3b"]["caveats"] == ch.CAVEATS["protgpt3-1.3b"]
    assert "gpt2-large" not in payload["arms"]


@pytest.mark.skipif(
    not (MODEL_ROOT / "ProtGPT3-1.3B").is_dir(),
    reason="ProtGPT3 checkpoint is host-local",
)
def test_protgpt3_tokenizer_arm_when_checkpoint_exists():
    arm = ch.tokenizer_arm("protgpt3-1.3b")
    ids = ch.item_ids(arm, "ACDE", modality="protein")
    assert ch.content_offset(arm) == 2
    assert len(ids) == 6


@pytest.mark.skipif(
    not (MODEL_ROOT / "RITA_xl").is_dir(),
    reason="RITA checkpoint is host-local",
)
def test_rita_tokenizer_arm_when_checkpoint_exists():
    arm = ch.tokenizer_arm("rita-xl")
    row, _ = ch.build_row(
        arm,
        {"key": "u", "target": 1, "conditions": {ch.HOMOLOGUE: [["pool", 0]]}},
        ch.HOMOLOGUE,
        records=["ACDE", "FGHI"],
        filler="ACDE",
        modality="protein",
    )
    ch.packing_assertions(arm, row)


def test_rita_eos_uses_native_token_when_eos_token_id_is_unset():
    tokenizer = _MapTokenizer(eos=None, extra={"<EOS>": 2})
    arm = _arm("rita-xl", tokenizer, input_format="eos_bounded_seq")
    assert ch._rita_eos(arm) == 2


def test_resolve_plan_path_finds_the_queue_cell_directory(tmp_path):
    missing = tmp_path / "plan_progen2-base.json"
    actual = tmp_path / "s46x_plan_progen2-base" / "plan_progen2-base.json"
    actual.parent.mkdir()
    actual.write_text("{}", encoding="utf-8")
    assert ch.resolve_plan_path(missing, arm="progen2-base") == actual
