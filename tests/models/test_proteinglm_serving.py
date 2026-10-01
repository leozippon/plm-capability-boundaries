"""ProteinGLM budget serving: derived loader, continuation mask, tiny CPU forwards.

These tests load no released 7B weights. The 1.1277 / 2.8974 / 16.9930 nats/residue
probe remains a future 7B reproduction target, not a tiny-model result.
Equivalence checks below are in raw logit units at 1e-4 unless noted as NLL nats.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts/capability") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts/capability"))

from src.capability.models import proteinglm as pglm  # noqa: E402
from src.capability.core.arms import (  # noqa: E402
    PANEL,
    STAGED_ARMS,
    STAGED_CANDIDATE_ARMS,
    STAGED_SCALE_ARMS,
    STAGED_SECOND_STAGE_ARMS,
    Arm,
    Cohort,
    INPUT_FORMAT_GMASK_SOP_EOS,
    _ATTENTION_PATH,
    _DECOMPOSABLE,
    _ROTARY_DECODERS,
    load_arm,
    load_arm_spec,
    rendering_marker_ids,
    symbols_per_token,
    tokenize_batch,
)
from src.capability.core.budget import scored_tokens  # noqa: E402
from src.capability.core.scoring import sequence_target_mask, target_rule  # noqa: E402

NAME = "proteinglm-7b-clm"
LOGIT_ATOL = 1e-4
MODEL_BASE_DIR = Path(os.environ.get("MODEL_BASE_DIR", ""))
_FROZEN_SOURCE = MODEL_BASE_DIR / "models_R2/proteinglm-7b-clm"


from scripts.capability.entrypoints import stage_path

def _load_stage(filename: str):
    path = stage_path(REPO_ROOT, filename)
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def _tiny_arm(*, batch_layers: int = 2, hidden: int = 32, sdpa: bool = True) -> Arm:
    source = _require_source()
    model = pglm.tiny_causal_lm(
        source, use_pytorch_sdpa=sdpa, num_layers=batch_layers, hidden_size=hidden
    )
    spec = replace(STAGED_ARMS[NAME], n_layer=batch_layers, d_model=hidden)
    return Arm(
        spec=spec,
        model=model,
        tokenizer=_tokenizer(),
        device="cpu",
        dtype="float32",
        attn_implementation=None,
    )


# ------------------------------------------------------------------ doors


def test_protein_glm_stays_behind_the_second_stage_door_only():
    spec = STAGED_ARMS[NAME]
    assert spec.architecture == "proteinglm"
    assert spec.input_format == INPUT_FORMAT_GMASK_SOP_EOS
    assert spec.capabilities == frozenset({"budget"})
    assert NAME in STAGED_SECOND_STAGE_ARMS
    assert NAME not in PANEL
    assert NAME not in STAGED_SCALE_ARMS
    assert NAME not in STAGED_CANDIDATE_ARMS
    assert spec.architecture not in _ROTARY_DECODERS
    assert spec.architecture not in _DECOMPOSABLE
    assert spec.architecture not in _ATTENTION_PATH
    with pytest.raises(KeyError):
        load_arm(NAME, device="cpu")


def test_cohort_power_still_needs_the_second_stage_opt_in():
    stage = _load_stage("cohort_power.py")
    closed = argparse.Namespace(
        kind="protein",
        with_ec=False,
        allow_staged_scale_arms=False,
        allow_second_stage_arms=False,
        allow_candidate_arms=False,
    )
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms([NAME], closed)
    opened = argparse.Namespace(
        kind="protein",
        with_ec=False,
        allow_staged_scale_arms=False,
        allow_second_stage_arms=True,
        allow_candidate_arms=False,
    )
    stage.validate_arms([NAME], opened)


def test_protein_gym_and_designed_referent_doors_are_unchanged():
    retrieval = _load_stage("retrieval_bound.py")
    designed = _load_stage("designed_referent.py")
    assert NAME in retrieval.SCOREABLE_ARMS
    assert NAME not in retrieval.ARM_CORPUS
    assert NAME not in designed.DEFAULT_ARMS


# ------------------------------------------------------------------ derivation


def test_source_files_are_unmodified_and_derived_patches_are_exact():
    source = _require_source()
    before = {name: (source / name).read_bytes() for name in pglm.SOURCE_FILENAMES}
    package = pglm.derive_serving_package(source)
    after = {name: (source / name).read_bytes() for name in pglm.SOURCE_FILENAMES}
    assert before == after
    by_name = {item.name: item for item in package.files}
    assert by_name["configuration_proteinglm.py"].patch_kinds == ()
    assert by_name["quantization.py"].patch_kinds == ()
    assert by_name["modeling_proteinglm.py"].patch_kinds == (
        pglm.PATCH_DROP_DEEPSPEED,
        pglm.PATCH_INFERENCE_CHECKPOINT,
    )
    assert (package.path / "configuration_proteinglm.py").read_bytes() == before[
        "configuration_proteinglm.py"
    ]
    assert (package.path / "quantization.py").read_bytes() == before["quantization.py"]
    assert (package.path / "modeling_proteinglm.py").read_bytes() != before[
        "modeling_proteinglm.py"
    ]
    derived_modeling = (package.path / "modeling_proteinglm.py").read_text(encoding="utf-8")
    source_modeling = (source / "modeling_proteinglm.py").read_text(encoding="utf-8")
    assert "import torch, deepspeed" in source_modeling
    assert "import torch, deepspeed" not in derived_modeling
    assert "deepspeed.checkpointing" not in derived_modeling
    assert "gradient checkpointing is unsupported" in derived_modeling


def test_original_automodel_path_still_refuses_deepspeed():
    source = _require_source()
    with pytest.raises(ImportError, match="deepspeed"):
        AutoModelForCausalLM.from_pretrained(
            str(source), trust_remote_code=True, local_files_only=True
        )


def test_derived_tiny_model_constructs_without_deepspeed():
    model = pglm.tiny_causal_lm(_require_source(), num_layers=1, hidden_size=32)
    assert model.training is False
    assert model.config.use_pytorch_sdpa is True
    assert model.config.is_causal is True
    assert model.config.rotary_embedding_2d is False


def test_max_length_exists_and_is_not_rewritten():
    source = _require_source()
    config_cls, _, _ = pglm.load_derived_classes(source)
    config = config_cls.from_pretrained(str(source), local_files_only=True)
    assert hasattr(config, "max_length")
    original = config.max_length
    note = pglm.max_length_compatibility(config)
    assert note["filled_max_length_from_seq_length"] is False
    assert config.max_length == original
    assert int(config.seq_length) == 1024


def test_training_checkpoint_path_raises_instead_of_falling_back():
    source = _require_source()
    model = pglm.tiny_causal_lm(source, num_layers=1, hidden_size=32)
    _, model_cls, package = pglm.load_derived_classes(source)
    modeling = pglm._import_derived_module(package, "modeling_proteinglm")
    with pytest.raises(RuntimeError, match="inference-only"):
        modeling.get_checkpoint_fn()
    model.train()
    model.transformer.encoder.gradient_checkpointing = True
    ids = torch.tensor([[29, 32, 34, 2, 20]], dtype=torch.long)
    mask = torch.ones_like(ids)
    with torch.enable_grad():
        with pytest.raises(RuntimeError, match="inference-only"):
            model(input_ids=ids, attention_mask=mask)
    assert model_cls.__name__ == "ProteinGLMForCasualLM"


# ------------------------------------------------------------------ tokenizer / mask


def test_default_tokenizer_appends_eos_and_served_path_does_not():
    tokenizer = _tokenizer()
    residues = "ACDE"
    default_ids = tokenizer(residues, return_tensors=None)["input_ids"]
    assert default_ids[-1] == tokenizer.eos_token_id
    rendered = pglm.render_budget_sequence(residues)
    served = pglm.encode_budget_text(tokenizer, rendered, max_len=32)
    assert served[:3] == list(pglm.PREFIX_IDS)
    assert served[-1] != tokenizer.eos_token_id
    chat = tokenizer.apply_chat_template(
        residues, tokenize=True, add_special_tokens=False
    )
    if isinstance(chat[0], list):
        chat = chat[0]
    assert served == [int(value) for value in chat]


def test_budget_encode_rejects_illegal_short_wrong_prefix_and_over_window():
    tokenizer = _tokenizer()
    with pytest.raises(ValueError, match="AA20"):
        pglm.render_budget_sequence("AX")
    with pytest.raises(ValueError, match="two residues"):
        pglm.render_budget_sequence("A")
    with pytest.raises(ValueError, match="must start with"):
        pglm.encode_budget_text(tokenizer, "ACDE", max_len=32)
    too_long = "A" * (pglm.CONTEXT_LENGTH - pglm.PREFIX_LENGTH + 1)
    with pytest.raises(ValueError, match="seq_length"):
        pglm.render_budget_sequence(too_long)
    rendered = pglm.render_budget_sequence("ACDE")
    with pytest.raises(ValueError, match="max_len"):
        pglm.encode_budget_text(tokenizer, rendered, max_len=6)


def _tokenizer_only_arm() -> Arm:
    return Arm(
        spec=STAGED_ARMS[NAME],
        model=torch.nn.Identity(),
        tokenizer=_tokenizer(),
        device="cpu",
        dtype="float32",
        attn_implementation=None,
    )


def test_symbols_per_token_counts_prefix_plus_l_and_refuses_tail_eos():
    arm = _tokenizer_only_arm()
    residues = "ACDE"
    rendered = pglm.render_budget_sequence(residues)
    ids = pglm.encode_budget_text(arm.tokenizer, rendered, max_len=32)
    assert len(ids) == len(residues) + pglm.PREFIX_LENGTH
    assert ids[:3] == list(pglm.PREFIX_IDS)
    assert ids[-1] != arm.tokenizer.eos_token_id
    ratio = symbols_per_token(arm, [rendered], 32)
    assert ratio == pytest.approx(len(residues) / (len(residues) + pglm.PREFIX_LENGTH))


def test_symbols_per_token_refuses_silent_truncation():
    arm = _tokenizer_only_arm()
    rendered = pglm.render_budget_sequence("ACDE")
    with pytest.raises(ValueError, match="max_len"):
        symbols_per_token(arm, [rendered], 6)


def test_target_rule_and_mask_keep_residues_2_to_L_not_four_dropped_columns():
    assert target_rule(INPUT_FORMAT_GMASK_SOP_EOS) == pglm.TARGET_RULE
    # prefix + L=4 residues -> 7 tokens; keep columns = 6; q>=3 keeps 3 = L-1.
    ids = torch.tensor([[29, 32, 34, 2, 20, 10, 6]], dtype=torch.long)
    mask = torch.ones_like(ids)
    keep = sequence_target_mask(ids, mask, rule=pglm.TARGET_RULE)
    assert keep.shape == (1, 6)
    assert keep.tolist() == [[False, False, False, True, True, True]]
    assert int(keep.sum()) == 3


def test_wrong_prefix_is_rejected_by_the_mask():
    ids = torch.tensor([[0, 0, 0, 2, 20, 10, 6]], dtype=torch.long)
    mask = torch.ones_like(ids)
    with pytest.raises(ValueError, match="does not start with"):
        sequence_target_mask(ids, mask, rule=pglm.TARGET_RULE)


# ------------------------------------------------------------------ tiny forwards


def test_logits_are_batch_first_and_hidden_are_seq_first():
    arm = _tiny_arm(batch_layers=2, hidden=32)
    batch, seq = 3, 7
    ids = torch.zeros(batch, seq, dtype=torch.long)
    ids[:, :3] = torch.tensor(pglm.PREFIX_IDS)
    ids[:, 3:] = 2
    mask = torch.ones_like(ids)
    with torch.no_grad():
        out = arm.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    assert tuple(out.logits.shape) == (batch, seq, 128)
    hidden = out.hidden_states[-1]
    assert tuple(hidden.shape) == (seq, batch, 32)


def test_sdpa_matches_independent_reference_in_logit_units():
    source = _require_source()
    sdpa = pglm.tiny_causal_lm(source, use_pytorch_sdpa=True, num_layers=2, hidden_size=32)
    ref = pglm.tiny_causal_lm(source, use_pytorch_sdpa=False, num_layers=2, hidden_size=32)
    ref.load_state_dict(sdpa.state_dict())
    sdpa.eval()
    ref.eval()
    batch, seq = 2, 9
    ids = torch.zeros(batch, seq, dtype=torch.long)
    ids[:, :3] = torch.tensor(pglm.PREFIX_IDS)
    ids[0, 3:] = torch.tensor([2, 20, 10, 6, 8, 12])
    ids[1, 3:] = torch.tensor([5, 7, 9, 11, 13, 15])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        left = sdpa(input_ids=ids, attention_mask=mask).logits.float()
        right = ref(input_ids=ids, attention_mask=mask).logits.float()
    assert left.shape[0] != left.shape[1]
    assert torch.allclose(left, right, atol=LOGIT_ATOL, rtol=0)
    # Unit: raw logit, absolute tolerance 1e-4.


def test_last_residue_perturbation_does_not_change_earlier_logits():
    arm = _tiny_arm(batch_layers=1, hidden=32)
    ids = torch.tensor([[29, 32, 34, 2, 20, 10, 6]], dtype=torch.long)
    mask = torch.ones_like(ids)
    perturbed = ids.clone()
    perturbed[0, -1] = 8
    with torch.no_grad():
        clean = arm.model(input_ids=ids, attention_mask=mask).logits.float()
        moved = arm.model(input_ids=perturbed, attention_mask=mask).logits.float()
    assert torch.allclose(clean[:, :-1, :], moved[:, :-1, :], atol=LOGIT_ATOL, rtol=0)


def test_padded_mixed_lengths_match_unpadded_rows():
    arm = _tiny_arm(batch_layers=1, hidden=32)
    short = torch.tensor([[29, 32, 34, 2, 20, 10, 6]], dtype=torch.long)
    long = torch.tensor([[29, 32, 34, 2, 20, 10, 6, 8, 12]], dtype=torch.long)
    pad = arm.tokenizer.pad_token_id
    batched = torch.full((2, long.shape[1]), pad, dtype=torch.long)
    batched[0, : short.shape[1]] = short
    batched[1] = long
    batched_mask = torch.zeros_like(batched)
    batched_mask[0, : short.shape[1]] = 1
    batched_mask[1] = 1
    with torch.no_grad():
        short_logits = arm.model(
            input_ids=short, attention_mask=torch.ones_like(short)
        ).logits.float()
        long_logits = arm.model(
            input_ids=long, attention_mask=torch.ones_like(long)
        ).logits.float()
        batch_logits = arm.model(input_ids=batched, attention_mask=batched_mask).logits.float()
    assert torch.allclose(
        batch_logits[0, : short.shape[1], :], short_logits[0], atol=LOGIT_ATOL, rtol=0
    )
    assert torch.allclose(batch_logits[1], long_logits[0], atol=LOGIT_ATOL, rtol=0)


def test_scored_tokens_account_l_minus_one_and_skip_prefix():
    arm = _tiny_arm(batch_layers=1, hidden=32)
    residues = "ACDE"
    rendered = Cohort(
        name="stub", kind="protein", records=[residues], min_symbols=0, max_symbols=8
    ).input_strings(arm)
    assert rendered == [pglm.NATIVE_PREFIX + residues]
    ids, mask = tokenize_batch(arm, rendered, max_len=32)
    assert ids[0, :3].tolist() == list(pglm.PREFIX_IDS)
    assert ids[0, -1].item() != arm.tokenizer.eos_token_id
    assert rendering_marker_ids(arm) == pglm.PREFIX_IDS
    scored = scored_tokens(arm, rendered, max_len=32, batch_size=1)
    assert len(scored) == len(residues) - 1
    keep = sequence_target_mask(ids, mask, rule=target_rule(arm.spec.input_format))
    assert int(keep.sum()) == len(residues) - 1


def test_float32_is_required_and_eager_attention_is_refused():
    spec = STAGED_ARMS[NAME]
    with pytest.raises(ValueError, match="FP32-only"):
        load_arm_spec(spec, device="cpu", dtype="bfloat16")
    with pytest.raises(ValueError, match="inert"):
        load_arm_spec(spec, device="cpu", dtype="float32", attn_implementation="eager")
