"""CPU contract for the FP32 text-AA string-control scorer.

Real published weights are not loaded. Tokenisers may be read from local
``MODEL_BASE_DIR``. Tiny Transformers configs stand in for GPT-2, Qwen2,
and Llama. ByGPT5 uses its local tokenizer plus a float32 stub, and it is the one
family whose boundary declares no conditioning prefix at all, so the scorer's
zero-prefix path is checked on its real vocabulary without executing an
unbounded remote-code weight load.
"""

from __future__ import annotations

import gc
import os
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.precision_policy import FP32_PER_TARGET_ABS  # noqa: E402
from src.capability.models.scale_comparison import STRATUM_N_TO_C  # noqa: E402
from src.capability.models.text_aa_fitness import (  # noqa: E402
    ENCODING_MERGED_BPE,
    ENCODING_ONE_TOKEN_PER_RESIDUE,
    NO_CONDITIONING_KIND,
    TEXT_AA_FP32_V1,
    TextAABoundary,
    TextAAEncodingError,
    TextAAFitnessScorer,
    encode_text_aa,
    load_text_aa_scorer,
    load_text_aa_tokenizer,
    read_hard_context,
    resolve_text_aa_boundary,
)

PROBE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
MODEL_BASE_DIR = Path(os.environ.get("MODEL_BASE_DIR", ""))
GPT2_DIR = MODEL_BASE_DIR / "gpt2"
QWEN2_DIR = MODEL_BASE_DIR / "Qwen2.5-0.5B"
QWEN3_DIR = MODEL_BASE_DIR / "Qwen3-8B-Base"
LLAMA_DIR = MODEL_BASE_DIR / "Llama-3.2-3B"
BYGPT5_DIR = MODEL_BASE_DIR / "bygpt5-small-en"


def _gpt2_boundary() -> TextAABoundary:
    return TextAABoundary(
        name="gpt2",
        model_type="gpt2",
        tokenizer_class="GPT2TokenizerFast",
        conditioning_kind="bos_same_id_as_eos",
        conditioning_id=50256,
        conditioning_token="<|endoftext|>",
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=None,
        unk_token_id=50256,
        decoder_start_token_id=None,
        documented_context=1024,
        forbid_token_ids=(),
        trust_remote_code=False,
    )


def _qwen_boundary(*, name: str = "qwen2.5-0.5b", model_type: str = "qwen2") -> TextAABoundary:
    return TextAABoundary(
        name=name,
        model_type=model_type,
        tokenizer_class="Qwen2TokenizerFast",
        conditioning_kind="eos_document_separator",
        conditioning_id=151643,
        conditioning_token="<|endoftext|>",
        bos_token_id=None,
        eos_token_id=151643,
        pad_token_id=151643,
        unk_token_id=None,
        decoder_start_token_id=None,
        documented_context=32768,
        forbid_token_ids=(151644,),
        trust_remote_code=False,
    )


def _llama_boundary() -> TextAABoundary:
    return TextAABoundary(
        name="llama-3.2-3b",
        model_type="llama",
        tokenizer_class="PreTrainedTokenizerFast",
        conditioning_kind="true_bos",
        conditioning_id=128000,
        conditioning_token="<|begin_of_text|>",
        bos_token_id=128000,
        eos_token_id=128001,
        pad_token_id=None,
        unk_token_id=None,
        decoder_start_token_id=None,
        documented_context=131072,
        forbid_token_ids=(),
        trust_remote_code=False,
    )


def _bygpt5_boundary() -> TextAABoundary:
    return TextAABoundary(
        name="bygpt5-small-en",
        model_type="bygpt5",
        tokenizer_class="ByGPT5Tokenizer",
        conditioning_kind=NO_CONDITIONING_KIND,
        conditioning_id=None,
        conditioning_token=None,
        bos_token_id=None,
        eos_token_id=1,
        pad_token_id=0,
        unk_token_id=2,
        decoder_start_token_id=0,
        documented_context=None,
        forbid_token_ids=(),
        trust_remote_code=True,
    )


def _require_dir(path: Path) -> Path:
    if not path.is_dir():
        pytest.skip(f"local tokenizer directory is absent: {path}")
    return path


#: The text tokenizers that merge neighbouring residues into multi-residue
#: pieces, with the boundary each is declared under. GPT-2 reaches 22 scored
#: tokens for the 33-residue PROBE, Qwen2/Qwen3/Llama 21, so none of them can
#: carry the per-residue functional.
_MERGING_CASES = {
    "gpt2_tokenizer": _gpt2_boundary,
    "qwen2_tokenizer": _qwen_boundary,
    "qwen3_tokenizer": lambda: _qwen_boundary(
        name="qwen3-8b-base", model_type="qwen3"
    ),
    "llama_tokenizer": _llama_boundary,
}


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    path = _require_dir(GPT2_DIR)
    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )


@pytest.fixture(scope="module")
def qwen2_tokenizer():
    path = _require_dir(QWEN2_DIR)
    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    path = _require_dir(QWEN3_DIR)
    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )


@pytest.fixture(scope="module")
def llama_tokenizer():
    path = _require_dir(LLAMA_DIR)
    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )


@pytest.fixture(scope="module")
def bygpt5_tokenizer():
    path = _require_dir(BYGPT5_DIR)
    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True
    )


class _StubLM(nn.Module):
    def __init__(self, vocab_size: int, *, logits_fn=None) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.embed = nn.Embedding(vocab_size, 8)
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self._logits_fn = logits_fn
        self.forward_ids: list[torch.Tensor] = []
        self.forward_masks: list[torch.Tensor | None] = []
        self.logit_tensors: list[torch.Tensor] = []

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        assert use_cache is False
        self.forward_ids.append(input_ids.detach().cpu().clone())
        self.forward_masks.append(
            None if attention_mask is None else attention_mask.detach().cpu().clone()
        )
        if self._logits_fn is not None:
            logits = self._logits_fn(input_ids)
        else:
            batch, width = input_ids.shape
            position = torch.arange(width, dtype=torch.float32, device=input_ids.device)
            vocab = torch.arange(
                self.embed.num_embeddings, dtype=torch.float32, device=input_ids.device
            )
            logits = (0.15 * position.view(1, width, 1) + 0.02 * vocab.view(1, 1, -1)).expand(
                batch, width, self.embed.num_embeddings
            ).contiguous()
        self.logit_tensors.append(logits)
        return SimpleNamespace(logits=logits)


def _tiny_gpt2(tokenizer, *, n_positions: int = 64) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=32,
        n_layer=1,
        n_head=4,
        n_inner=64,
        bos_token_id=50256,
        eos_token_id=50256,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def _tiny_qwen2(tokenizer, *, max_position_embeddings: int = 64) -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max_position_embeddings,
        bos_token_id=151643,
        eos_token_id=151643,
        pad_token_id=151643,
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def _tiny_llama(tokenizer, *, max_position_embeddings: int = 64) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max_position_embeddings,
        bos_token_id=128000,
        eos_token_id=128001,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        attention_dropout=0.0,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _scorer(model, tokenizer, boundary, *, window: int = 64, hard=None):
    return TextAAFitnessScorer(
        model=model,
        tokenizer=tokenizer,
        boundary=boundary,
        application_window_tokens=window,
        facts={"hard_context": hard, "name": boundary.name},
    )


def _independent_shifted_ce_sum(model, ids: list[int]) -> tuple[float, int]:
    tokens = torch.tensor([ids], dtype=torch.long)
    mask = torch.ones_like(tokens)
    with torch.no_grad():
        logits = model(input_ids=tokens, attention_mask=mask, use_cache=False).logits
        assert logits.dtype == torch.float32
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
            reduction="sum",
        )
    n_target = len(ids) - 1
    return float((-loss).double().cpu()), n_target


def test_protocol_constant_and_stratum_are_declared():
    assert TEXT_AA_FP32_V1 == "text-aa-fp32-v1"
    assert TextAAFitnessScorer.scoring_stratum == STRATUM_N_TO_C
    assert "document-boundary" in TextAAFitnessScorer.score_description
    assert "PASS" not in TextAAFitnessScorer.score_description


def test_read_hard_context_ignores_generation_max_length():
    assert read_hard_context(SimpleNamespace(n_positions=1024, max_length=20)) == 1024
    assert read_hard_context(SimpleNamespace(max_position_embeddings=32768)) == 32768
    assert read_hard_context(SimpleNamespace(max_length=20, d_model=1472)) is None


def test_resolve_boundary_inheritance_and_refuse_remote_code_on_builtins():
    table = {
        "text_aa_checkpoints": {
            "gpt2": {
                "model_type": "gpt2",
                "tokenizer_class": "GPT2TokenizerFast",
                "eos_token_id": 50256,
                "bos_token_id": 50256,
                "unk_token_id": 50256,
                "pad_token_id": None,
                "context": 1024,
                "conditioning": {
                    "kind": "bos_same_id_as_eos",
                    "token": "<|endoftext|>",
                    "id": 50256,
                },
            },
            "gpt2-medium": {
                "inherits_boundary_from": "gpt2",
                "context": 1024,
            },
        }
    }
    resolved = resolve_text_aa_boundary("gpt2-medium", table)
    assert resolved.conditioning_id == 50256
    assert resolved.model_type == "gpt2"
    assert resolved.trust_remote_code is False
    with pytest.raises(ValueError, match="trust_remote_code"):
        TextAABoundary.from_mapping(
            {
                "model_type": "gpt2",
                "tokenizer_class": "GPT2TokenizerFast",
                "trust_remote_code": True,
                "eos_token_id": 50256,
                "conditioning": {"kind": "bos", "token": "x", "id": 1},
            },
            name="gpt2",
        )


def test_illegal_empty_and_non_aa20_are_refused_before_unk(gpt2_tokenizer):
    boundary = _gpt2_boundary()
    for illegal in ("", "MKTZ", "MKTJ", "mkt", "MKT A", "MKT\n", "MKT\t"):
        with pytest.raises(TextAAEncodingError):
            encode_text_aa(gpt2_tokenizer, illegal, boundary)
    with pytest.raises(TypeError, match="sequence must be a str"):
        encode_text_aa(gpt2_tokenizer, 123, boundary)  # type: ignore[arg-type]


class _CallBoom:
    def __init__(self, inner, *, error):
        self._inner = inner
        self._error = error
        self.all_special_ids = getattr(inner, "all_special_ids", [])

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, sequence, add_special_tokens=False):
        raise self._error

    def decode(self, ids, skip_special_tokens=False):
        return self._inner.decode(ids, skip_special_tokens=skip_special_tokens)


class _DecodeBoom:
    def __init__(self, inner, *, error):
        self._inner = inner
        self._error = error
        self.all_special_ids = getattr(inner, "all_special_ids", [])

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, sequence, add_special_tokens=False):
        return self._inner(sequence, add_special_tokens=add_special_tokens)

    def decode(self, ids, skip_special_tokens=False):
        raise self._error


class _MemoryIds:
    def __iter__(self):
        raise MemoryError("synthetic ids conversion")


class _BadOutput:
    def __init__(self, payload):
        self._payload = payload
        self.all_special_ids = []

    def __call__(self, sequence, add_special_tokens=False):
        return self._payload

    def decode(self, ids, skip_special_tokens=False):
        return "AAA"


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        (lambda tok: _CallBoom(tok, error=ValueError("internal call")), ValueError),
        (lambda tok: _CallBoom(tok, error=TypeError("internal call")), TypeError),
        (lambda tok: _DecodeBoom(tok, error=ValueError("internal decode")), ValueError),
        (lambda tok: _DecodeBoom(tok, error=TypeError("internal decode")), TypeError),
    ],
)
def test_tokenizer_internal_errors_are_not_encoding_errors(gpt2_tokenizer, factory, error):
    boundary = _gpt2_boundary()
    with pytest.raises(error) as caught:
        encode_text_aa(factory(gpt2_tokenizer), "AAA", boundary)
    assert not isinstance(caught.value, TextAAEncodingError)


def test_ids_conversion_memoryerror_is_not_wrapped(gpt2_tokenizer):
    boundary = _gpt2_boundary()
    tokenizer = _BadOutput({"input_ids": _MemoryIds()})
    tokenizer.all_special_ids = getattr(gpt2_tokenizer, "all_special_ids", [])
    with pytest.raises(MemoryError, match="synthetic ids conversion"):
        encode_text_aa(tokenizer, "AAA", boundary)


def test_missing_or_malformed_input_ids_are_fatal_tool_errors():
    boundary = _gpt2_boundary()
    with pytest.raises(ValueError, match="input_ids") as missing:
        encode_text_aa(_BadOutput({}), "AAA", boundary)
    assert not isinstance(missing.value, TextAAEncodingError)
    with pytest.raises(ValueError, match="batched tokenizer output") as batched:
        encode_text_aa(_BadOutput({"input_ids": [[1, 2]]}), "AAA", boundary)
    assert not isinstance(batched.value, TextAAEncodingError)


def test_merging_text_tokenizer_is_accepted_and_tagged_merged_bpe(request):
    """A merged scored span is accepted and tagged, not refused.

    User-ordered 2026-09-20: the token-sum is scored; the encoding class records
    that it is not one token per residue.
    """

    for fixture_name, factory in _MERGING_CASES.items():
        tokenizer = request.getfixturevalue(fixture_name)
        boundary = factory()
        targets = tokenizer(PROBE, add_special_tokens=False)["input_ids"]
        assert len(targets) < len(PROBE)
        ids = encode_text_aa(tokenizer, PROBE, boundary)
        assert ids[0] == boundary.conditioning_id
        assert ids[1:] == list(targets)
        scorer = _scorer(_StubLM(len(tokenizer)), tokenizer, boundary, window=128)
        meta = scorer.encoding_metadata(PROBE)
        assert meta["encoding_class"] == ENCODING_MERGED_BPE
        assert meta["n_residues"] == len(PROBE)
        assert meta["n_target_tokens"] == len(targets)
        assert meta["residues_per_token"] == pytest.approx(len(PROBE) / len(targets))
        assert meta["n_prefix_tokens"] == 1
        assert meta["n_scored_tokens"] == len(ids) - 1
        assert meta["n_prefix_tokens"] != len(ids) - len(PROBE)


def test_the_residue_rule_belongs_to_the_encoding_not_the_checkpoint(gpt2_tokenizer):
    """GPT-2 merges ``MM`` into one piece but still carries ``M`` one-to-one."""

    boundary = _gpt2_boundary()
    ids = encode_text_aa(gpt2_tokenizer, "M", boundary)
    assert ids == [50256, *gpt2_tokenizer("M", add_special_tokens=False)["input_ids"]]
    assert len(ids) == 2
    scorer = _scorer(_StubLM(len(gpt2_tokenizer)), gpt2_tokenizer, boundary, window=64)
    one = scorer.encoding_metadata("M")
    assert one["encoding_class"] == ENCODING_ONE_TOKEN_PER_RESIDUE
    assert one["n_prefix_tokens"] == 1
    merged = encode_text_aa(gpt2_tokenizer, "MM", boundary)
    assert len(merged) - 1 == 1
    meta = scorer.encoding_metadata("MM")
    assert meta["encoding_class"] == ENCODING_MERGED_BPE
    assert meta["n_residues"] == 2
    assert meta["n_target_tokens"] == 1
    assert meta["n_prefix_tokens"] == 1


def test_one_token_per_residue_is_still_tagged_and_merged_bpe_is_scored(
    bygpt5_tokenizer, gpt2_tokenizer
):
    boundary = _bygpt5_boundary()
    ids = encode_text_aa(bygpt5_tokenizer, PROBE, boundary)
    assert len(ids) == len(PROBE)
    assert ids == list(bygpt5_tokenizer(PROBE, add_special_tokens=False)["input_ids"])
    scorer = _scorer(
        _StubLM(len(bygpt5_tokenizer)), bygpt5_tokenizer, boundary, window=64
    )
    meta = scorer.encoding_metadata(PROBE)
    assert meta["encoding_class"] == ENCODING_ONE_TOKEN_PER_RESIDUE
    assert meta["n_prefix_tokens"] == 0
    assert meta["n_scored_tokens"] == len(PROBE) - 1
    assert np.isfinite(scorer.log_likelihood([PROBE])).all()
    merging = _scorer(
        _StubLM(len(gpt2_tokenizer)), gpt2_tokenizer, _gpt2_boundary(), window=64
    )
    assert merging.encoding_metadata(PROBE)["encoding_class"] == ENCODING_MERGED_BPE
    assert np.isfinite(merging.log_likelihood([PROBE])).all()
    assert merging.token_lengths([PROBE]) == [len(merging.encode(PROBE))]


def test_gpt2_encode_prepends_boundary_keeps_first_target_and_omits_eos(
    gpt2_tokenizer,
):
    boundary = _gpt2_boundary()
    ids = encode_text_aa(gpt2_tokenizer, "M", boundary)
    assert ids[0] == 50256
    targets = gpt2_tokenizer("M", add_special_tokens=False)["input_ids"]
    assert ids[1:] == list(targets)
    assert len(targets) == 1
    assert ids[-1] != 50256
    assert 50256 not in ids[1:]
    assert gpt2_tokenizer.decode(targets, skip_special_tokens=False) == "M"
    default = gpt2_tokenizer("M", add_special_tokens=True)["input_ids"]
    assert list(default) == list(targets)


def test_qwen_uses_eos_separator_not_chat_or_bos(
    qwen2_tokenizer, qwen3_tokenizer
):
    for tokenizer, name, model_type in (
        (qwen2_tokenizer, "qwen2.5-0.5b", "qwen2"),
        (qwen3_tokenizer, "qwen3-8b-base", "qwen3"),
    ):
        boundary = _qwen_boundary(name=name, model_type=model_type)
        assert tokenizer.bos_token_id is None
        ids = encode_text_aa(tokenizer, "M", boundary)
        assert ids[0] == 151643
        assert 151644 not in ids
        targets = tokenizer("M", add_special_tokens=False)["input_ids"]
        assert len(targets) == 1
        assert ids[-1] != 151643
        assert tokenizer.decode(targets, skip_special_tokens=False) == "M"


def test_llama_true_bos_is_concatenated_without_double_prefix(llama_tokenizer):
    boundary = _llama_boundary()
    ids = encode_text_aa(llama_tokenizer, "M", boundary)
    assert ids[0] == 128000
    targets = llama_tokenizer("M", add_special_tokens=False)["input_ids"]
    assert 128000 not in targets
    assert ids[1:] == list(targets)
    assert len(targets) == 1
    assert ids[-1] != 128001
    with_special = llama_tokenizer("M", add_special_tokens=True)["input_ids"]
    assert list(with_special) == ids


def test_bygpt5_byte_targets_are_scored_without_a_conditioning_prefix(bygpt5_tokenizer):
    """The declared rendering is the byte string itself, and it is declared.

    ``decoder_start_token_id`` 0 is ByGPT5Config(T5Config)'s own default carried
    into a decoder-only config: measured on the staged checkpoint over 200
    OpenWebText documents at the 384-token window on the byte targets both
    renderings score, the bare string assigns 0.88939 nats/token against 0.90865
    prefixed with id 0, so the prefix is not the rendering and is not prepended.
    The declaration is explicit rather than an absent field: the boundary table
    names ``none_declared``, and ``from_mapping`` refuses a boundary that carries
    no conditioning block at all.
    """

    boundary = _bygpt5_boundary()
    targets = list(bygpt5_tokenizer(PROBE, add_special_tokens=False)["input_ids"])
    ids = encode_text_aa(bygpt5_tokenizer, PROBE, boundary)
    assert ids == targets
    assert len(ids) == len(PROBE)
    assert 0 not in ids
    assert bygpt5_tokenizer.decode(ids, skip_special_tokens=False) == PROBE
    stub = _StubLM(len(bygpt5_tokenizer))
    scorer = _scorer(stub, bygpt5_tokenizer, boundary, window=64)
    meta = scorer.encoding_metadata(PROBE)
    assert meta["n_prefix_tokens"] == 0
    # Every input token but the first is a target, so with no prefix the scored
    # set starts at the sequence's second byte rather than at its first.
    assert meta["n_scored_tokens"] == len(PROBE) - 1
    assert meta["n_input_tokens"] == len(PROBE)
    assert meta["conditioning_id"] is None
    assert meta["conditioning_kind"] == NO_CONDITIONING_KIND
    scorer.log_likelihood([PROBE])
    # The mask is still built from position rather than from id == pad, which is
    # the fact this vocabulary makes worth checking: its pad id is 0.
    forwarded = stub.forward_ids[0]
    mask = stub.forward_masks[0]
    assert mask is not None
    assert int(forwarded[0, 0]) == targets[0]
    assert torch.equal(mask, torch.ones_like(forwarded))


def test_a_null_conditioning_id_is_only_the_declaration_that_none_is_prefixed():
    base = {
        "model_type": "bygpt5",
        "tokenizer_class": "ByGPT5Tokenizer",
        "eos_token_id": 1,
        "pad_token_id": 0,
    }
    with pytest.raises(ValueError, match="missing conditioning"):
        TextAABoundary.from_mapping(base, name="bygpt5-small-en")
    with pytest.raises(ValueError, match="conditioning.id is required"):
        TextAABoundary.from_mapping(
            {**base, "conditioning": {"kind": "decoder_start", "token": "<pad>", "id": None}},
            name="bygpt5-small-en",
        )
    with pytest.raises(ValueError, match="cannot also carry one"):
        TextAABoundary.from_mapping(
            {**base, "conditioning": {"kind": NO_CONDITIONING_KIND, "id": 0}},
            name="bygpt5-small-en",
        )
    resolved = TextAABoundary.from_mapping(
        {**base, "conditioning": {"kind": NO_CONDITIONING_KIND, "token": None, "id": None}},
        name="bygpt5-small-en",
    )
    assert resolved.conditioning_id is None
    assert resolved.conditioning_token is None
    assert resolved.decoder_start_token_id is None


def test_the_declared_start_token_is_not_required_to_be_the_scoring_prefix(
    bygpt5_tokenizer,
):
    """The removed requirement, on the real checkpoint and the tracked table.

    The loader used to refuse any boundary whose ``decoder_start_token_id``
    differed from its conditioning id, which forced a config default to be a
    training fact. It still checks that key against the loaded config; it no
    longer ties it to the rendering the scorer uses.
    """

    from src.capability.models.text_aa_cohort import load_text_aa_boundary_table

    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("bygpt5-small-en", table)
    assert boundary.decoder_start_token_id == 0
    assert boundary.conditioning_id is None
    assert boundary.conditioning_kind == NO_CONDITIONING_KIND
    assert resolve_text_aa_boundary("bygpt5-base-en", table).conditioning_id is None
    bundle = load_text_aa_tokenizer(
        BYGPT5_DIR, boundary=boundary, name="bygpt5-small-en"
    )
    assert bundle.boundary.conditioning_id is None
    assert bundle.tokenizer is not None


def test_missing_window_is_refused(gpt2_tokenizer):
    model = _StubLM(len(gpt2_tokenizer))
    with pytest.raises(TypeError):
        TextAAFitnessScorer(
            model=model,
            tokenizer=gpt2_tokenizer,
            boundary=_gpt2_boundary(),
            facts={"hard_context": 64},
        )
    with pytest.raises(ValueError, match="application_window_tokens"):
        _scorer(model, gpt2_tokenizer, _gpt2_boundary(), window=1)


def test_over_window_and_hard_context_are_refused(bygpt5_tokenizer):
    boundary = _bygpt5_boundary()
    stub = _StubLM(len(bygpt5_tokenizer))
    encoded = encode_text_aa(bygpt5_tokenizer, "MKT", boundary)
    scorer = _scorer(stub, bygpt5_tokenizer, boundary, window=len(encoded) - 1)
    assert scorer.token_lengths(["MKT"]) == [len(encoded)]
    with pytest.raises(ValueError, match="application_window_tokens"):
        scorer.log_likelihood(["MKT"])
    with pytest.raises(ValueError, match="hard context"):
        _scorer(
            stub,
            bygpt5_tokenizer,
            boundary,
            window=4096,
            hard=1024,
        )


def test_empty_list_mixed_length_order_and_repeats(bygpt5_tokenizer):
    boundary = _bygpt5_boundary()
    model = _StubLM(len(bygpt5_tokenizer))
    scorer = _scorer(model, bygpt5_tokenizer, boundary, window=64, hard=64)
    assert scorer.token_lengths([]) == []
    empty = scorer.log_likelihood([])
    assert empty.shape == (0,)
    assert empty.dtype == np.float64
    sequences = ["W", "MKT", "ACDEFGHIKLMNPQRSTVWY"]
    lengths = scorer.token_lengths(sequences)
    assert lengths[0] < lengths[1] < lengths[2]
    totals = scorer.log_likelihood(sequences)
    singles = np.array(
        [scorer.log_likelihood([sequence])[0] for sequence in sequences]
    )
    assert totals == pytest.approx(singles, abs=0)
    shuffled = list(reversed(sequences))
    reversed_totals = scorer.log_likelihood(shuffled)
    assert reversed_totals[0] == pytest.approx(totals[2], abs=0)
    assert reversed_totals[2] == pytest.approx(totals[0], abs=0)
    repeats = scorer.log_likelihood(["MKT", "MKT", "MKT"])
    assert repeats.shape == (3,)
    assert repeats[0] == pytest.approx(repeats[1], abs=1e-12)
    assert repeats[1] == pytest.approx(repeats[2], abs=1e-12)


def test_independent_shifted_ce_matches_public_api_and_keeps_first_target(
    bygpt5_tokenizer,
):
    boundary = _bygpt5_boundary()
    model = _StubLM(len(bygpt5_tokenizer))
    scorer = _scorer(model, bygpt5_tokenizer, boundary, window=64, hard=64)
    sequence = "MKTAYIAK"
    ids = scorer.encode(sequence)
    got = scorer.log_likelihood([sequence])[0]
    independent, n_target = _independent_shifted_ce_sum(model, ids)
    assert n_target == len(ids) - 1
    assert abs(got - independent) / n_target <= FP32_PER_TARGET_ABS
    tokens = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(
            input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False
        ).logits
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        skip_first = logp[:, 1:].gather(-1, tokens[:, 2:].unsqueeze(-1)).squeeze(-1)
        skip_total = float(skip_first.sum().double().cpu())
    assert abs(got - skip_total) / n_target > FP32_PER_TARGET_ABS
    meta = scorer.encoding_metadata(sequence)
    # ByGPT5's boundary declares no conditioning token, so the one token that is
    # context is the sequence's own first byte and every other byte is a target.
    assert meta["n_prefix_tokens"] == 0
    assert scorer.encode(sequence) == list(
        bygpt5_tokenizer(sequence, add_special_tokens=False)["input_ids"]
    )
    assert meta["n_scored_tokens"] == n_target
    assert meta["n_residues"] == len(sequence)
    assert meta["trailing_eos_scored"] is False
    assert meta["prefix_scored"] is False


def test_qwen_conditioning_equals_pad_still_attends_position_zero(qwen2_tokenizer):
    boundary = _qwen_boundary()
    stub = _StubLM(len(qwen2_tokenizer))
    scorer = _scorer(stub, qwen2_tokenizer, boundary, window=64)
    scorer.log_likelihood(["M"])
    ids = stub.forward_ids[0]
    mask = stub.forward_masks[0]
    assert int(ids[0, 0]) == 151643
    assert mask is not None
    assert int(mask[0, 0]) == 1
    assert torch.equal(mask, torch.ones_like(mask))
    dropped = (ids != 151643).to(dtype=torch.long)
    assert int(dropped[0, 0]) == 0


def test_llama_tiny_numeric_gate(llama_tokenizer):
    boundary = _llama_boundary()
    model = _tiny_llama(llama_tokenizer)
    scorer = _scorer(model, llama_tokenizer, boundary, window=64, hard=64)
    ids = scorer.encode("M")
    got = scorer.log_likelihood(["M"])[0]
    independent, n_target = _independent_shifted_ce_sum(model, ids)
    assert abs(got - independent) / n_target <= FP32_PER_TARGET_ABS


def test_qwen2_tiny_numeric_gate(qwen2_tokenizer):
    boundary = _qwen_boundary()
    model = _tiny_qwen2(qwen2_tokenizer)
    scorer = _scorer(model, qwen2_tokenizer, boundary, window=64, hard=64)
    ids = scorer.encode("M")
    got = scorer.log_likelihood(["M"])[0]
    independent, n_target = _independent_shifted_ce_sum(model, ids)
    assert abs(got - independent) / n_target <= FP32_PER_TARGET_ABS


def test_bf16_parameters_are_refused(gpt2_tokenizer):
    model = _tiny_gpt2(gpt2_tokenizer).to(torch.bfloat16)
    with pytest.raises(ValueError, match="float32"):
        _scorer(model, gpt2_tokenizer, _gpt2_boundary())


def test_declared_storage_dtype_reads_both_spellings_and_invents_nothing():
    from src.capability.models.text_aa_fitness import _declared_storage_dtype

    assert _declared_storage_dtype(SimpleNamespace(dtype=torch.bfloat16)) == "bfloat16"
    assert _declared_storage_dtype(SimpleNamespace(torch_dtype=torch.float16)) == "float16"
    assert _declared_storage_dtype(SimpleNamespace()) == "undeclared"
    assert (
        _declared_storage_dtype(SimpleNamespace(dtype=None, torch_dtype=None))
        == "undeclared"
    )


def test_stored_bf16_checkpoint_is_upcast_and_recorded_not_refused(
    gpt2_tokenizer, tmp_path
):
    """The protocol requires float32 execution, not float32 storage.

    ``load_text_aa_scorer`` asks for float32 from the checkpoint, which performs
    the upcast, so the only refusal that can fire is the loaded-parameter check.
    The checkpoint's own declared dtype is recorded as ``storage_torch_dtype``
    instead of being refused a second time on a claim the loader already acted on.
    """

    model = _tiny_gpt2(gpt2_tokenizer, n_positions=64)
    model.config.dtype = torch.bfloat16
    model = model.to(torch.bfloat16)
    checkpoint = tmp_path / "tiny-gpt2-bf16"
    checkpoint.mkdir()
    model.save_pretrained(checkpoint)
    gpt2_tokenizer.save_pretrained(checkpoint)
    with safe_open(checkpoint / "model.safetensors", framework="pt") as handle:
        assert handle.get_tensor("transformer.wte.weight").dtype == torch.bfloat16
    scorer = load_text_aa_scorer(
        checkpoint,
        boundary=_gpt2_boundary(),
        application_window_tokens=64,
        device="cpu",
        name="gpt2",
    )
    try:
        assert scorer.facts["storage_torch_dtype"] == "bfloat16"
        assert scorer.facts["dtype_observed"] == ["float32"]
        assert scorer.model.get_input_embeddings().weight.dtype == torch.float32
        got = scorer.log_likelihood(["M"])[0]
        independent, n_target = _independent_shifted_ce_sum(
            scorer.model, scorer.encode("M")
        )
        assert abs(got - independent) / n_target <= FP32_PER_TARGET_ABS
    finally:
        scorer.release()


def test_fp32_matmul_context_restores_caller_state(gpt2_tokenizer):
    model = _tiny_gpt2(gpt2_tokenizer)
    scorer = _scorer(model, gpt2_tokenizer, _gpt2_boundary(), window=64, hard=64)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    scorer.log_likelihood(["M"])
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.get_float32_matmul_precision() == "high"


def test_three_item_lifetime_releases_prior_logits(gpt2_tokenizer):
    boundary = _gpt2_boundary()
    model = _tiny_gpt2(gpt2_tokenizer)
    refs: list[weakref.ReferenceType[torch.Tensor]] = []
    original = model.forward

    def wrapped(*args, **kwargs):
        if refs:
            assert refs[-1]() is None
        outputs = original(*args, **kwargs)
        refs.append(weakref.ref(outputs.logits))
        return outputs

    model.forward = wrapped
    scorer = _scorer(model, gpt2_tokenizer, boundary, window=64, hard=64)
    sequences = ["W", "M", "K"]
    totals = scorer.log_likelihood(sequences)
    assert totals.shape == (3,)
    gc.collect()
    assert len(refs) == 3
    assert refs[0]() is None
    assert refs[1]() is None
    assert refs[2]() is None


def test_load_text_aa_scorer_from_saved_tiny_gpt2(gpt2_tokenizer, tmp_path):
    model = _tiny_gpt2(gpt2_tokenizer, n_positions=64)
    checkpoint = tmp_path / "tiny-gpt2"
    checkpoint.mkdir()
    model.save_pretrained(checkpoint)
    gpt2_tokenizer.save_pretrained(checkpoint)
    scorer = load_text_aa_scorer(
        checkpoint,
        boundary=_gpt2_boundary(),
        application_window_tokens=64,
        device="cpu",
        name="gpt2",
    )
    try:
        assert scorer.facts["protocol"] == TEXT_AA_FP32_V1
        assert scorer.facts["dtype_observed"] == ["float32"]
        assert scorer.facts["storage_torch_dtype"] == "float32"
        assert scorer.facts["hard_context"] == 64
        assert scorer.facts["experiment_admitted"] is False
        assert scorer.facts["production_batch_size"] == 1
        totals = scorer.log_likelihood(["M"])
        independent, n_target = _independent_shifted_ce_sum(scorer.model, scorer.encode("M"))
        assert abs(float(totals[0]) - independent) / n_target <= FP32_PER_TARGET_ABS
    finally:
        scorer.release()


def test_load_refuses_missing_path_and_window_beyond_hard_context(
    gpt2_tokenizer, tmp_path
):
    missing = tmp_path / "absent-checkpoint"
    with pytest.raises(FileNotFoundError):
        load_text_aa_scorer(
            missing,
            boundary=_gpt2_boundary(),
            application_window_tokens=64,
            device="cpu",
        )
    model = _tiny_gpt2(gpt2_tokenizer, n_positions=32)
    checkpoint = tmp_path / "tiny-gpt2-short"
    checkpoint.mkdir()
    model.save_pretrained(checkpoint)
    gpt2_tokenizer.save_pretrained(checkpoint)
    with pytest.raises(ValueError, match="hard context"):
        load_text_aa_scorer(
            checkpoint,
            boundary=_gpt2_boundary(),
            application_window_tokens=64,
            device="cpu",
            name="gpt2",
        )
