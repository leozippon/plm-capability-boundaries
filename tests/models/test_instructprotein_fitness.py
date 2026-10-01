"""CPU contract for the InstructProtein ProteinGym door's vocabulary guard.

No real weights, no GPU, no network. The fixture numbers are the staged
checkpoint's own measured shape -- 50287 real tokens over a 50304-row embedding
and head, leaving 17 unreachable rows -- so the padded-vocabulary case that was
refused on the first ProteinGym-supplement dispatch is the accepted case here.

The guard is written as two invariants rather than one equality: the model must
agree with its own config, and the tokenizer must fit inside the embedding. The
tail above the tokenizer is recorded as a measurement, not tolerated silently,
and :func:`src.capability.models.joint_modes.encode` is what makes it safe by refusing any
id the tokenizer's own declared size does not cover.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.models import instructprotein_fitness as IP  # noqa: E402
from src.capability.core.arms import LOADING_INFO_KEYS  # noqa: E402
from src.capability.models.joint_modes import resolve  # noqa: E402
from tests.core.test_joint_mode_qualification import instructprotein_stub  # noqa: E402

#: `vocab.json` carries 50265 entries and `added_tokens.json` 22 disjoint ones:
#: `<protein>`, `</protein>` and the twenty `ƤA`..`ƤY` residue tokens.
TOKENIZER_TOKENS = 50287

#: `config.json` declares 50304, and the embedding and head both carry that many
#: rows: 128 x 393, the vocabulary padded up to a multiple of 128.
EMBEDDING_ROWS = 50304


def _empty_info() -> dict[str, list[str]]:
    return {key: [] for key in LOADING_INFO_KEYS}


class _Tokenizer:
    """The staged rendering, with its own token count declared separately.

    The length is a constructor argument because it is the quantity the guard
    compares against the embedding width, and the two differ on this checkpoint.
    Id 1 is present so the config-declared pad id can be adopted, exactly as the
    released tokenizer's own vocabulary carries it.
    """

    def __init__(self, *, length: int) -> None:
        self._inner = instructprotein_stub()
        self._inner._vocab["<pad>"] = 1
        self._inner._inverse[1] = "<pad>"
        self.pad_token: str | None = None
        self._length = int(length)

    def __len__(self) -> int:
        return self._length

    def __call__(self, text, return_tensors=None):
        return self._inner(text, return_tensors=return_tensors)

    def convert_ids_to_tokens(self, index):
        return self._inner.convert_ids_to_tokens(index)

    def convert_tokens_to_ids(self, token):
        return self._inner.convert_tokens_to_ids(token)

    @property
    def pad_token_id(self):
        if self.pad_token is None:
            return None
        return self._inner.convert_tokens_to_ids(self.pad_token)


class _DummyLM(nn.Module):
    """An OPT-shaped module whose two vocabularies and config can disagree on demand."""

    def __init__(
        self,
        config: SimpleNamespace,
        *,
        input_vocab: int | None = None,
        output_vocab: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device("cpu")
        self.weight = nn.Parameter(torch.ones(1))
        declared = int(config.vocab_size)
        self._input = SimpleNamespace(
            num_embeddings=int(declared if input_vocab is None else input_vocab)
        )
        self._output = SimpleNamespace(
            out_features=int(declared if output_vocab is None else output_vocab)
        )

    def get_input_embeddings(self):
        return self._input

    def get_output_embeddings(self):
        return self._output

    def eval(self):
        return super().eval()


def _opt_config(**overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "model_type": "opt",
        "architectures": ["OPTForCausalLM"],
        "num_hidden_layers": 24,
        "hidden_size": 2048,
        "vocab_size": EMBEDDING_ROWS,
        "max_position_embeddings": 2048,
        "pad_token_id": 1,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokenizer_length: int = TOKENIZER_TOKENS,
    model: _DummyLM | None = None,
) -> IP.LoadedInstructProtein:
    checkpoint = tmp_path / "InstructProtein"
    checkpoint.mkdir(exist_ok=True)
    monkeypatch.setitem(IP.CHECKPOINTS, IP.INSTRUCTPROTEIN_ARM, checkpoint)
    tokenizer = _Tokenizer(length=tokenizer_length)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *_args, **_kwargs: tokenizer
    )
    built = _DummyLM(_opt_config()) if model is None else model
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *_args, **_kwargs: (built, _empty_info()),
    )
    return IP.load_instructprotein(IP.INSTRUCTPROTEIN_ARM, device="cpu", dtype="float32")


def test_a_padded_embedding_tail_is_accepted_and_reported(tmp_path, monkeypatch):
    loaded = _load(tmp_path, monkeypatch)
    facts = loaded.facts
    assert facts["vocab_size"] == EMBEDDING_ROWS
    assert facts["input_vocab_size"] == EMBEDDING_ROWS
    assert facts["output_vocab_size"] == EMBEDDING_ROWS
    assert facts["tokenizer_vocab_size"] == TOKENIZER_TOKENS
    assert facts["padded_embedding_rows"] == EMBEDDING_ROWS - TOKENIZER_TOKENS == 17
    note = facts["padded_embedding_note"]
    assert str(TOKENIZER_TOKENS) in note
    assert "unreachable" in note
    assert loaded.tokenisation.start_id is not None


def test_an_unpadded_tokenizer_reports_a_measured_zero_tail(tmp_path, monkeypatch):
    loaded = _load(tmp_path, monkeypatch, tokenizer_length=EMBEDDING_ROWS)
    assert loaded.facts["padded_embedding_rows"] == 0
    assert "0 embedding rows" in loaded.facts["padded_embedding_note"]


def test_a_model_that_disagrees_with_itself_is_still_refused(tmp_path, monkeypatch):
    for overrides in (
        {"input_vocab": EMBEDDING_ROWS - 1},
        {"output_vocab": EMBEDDING_ROWS + 1},
        {"input_vocab": TOKENIZER_TOKENS, "output_vocab": TOKENIZER_TOKENS},
    ):
        model = _DummyLM(_opt_config(), **overrides)
        with pytest.raises(ValueError, match="vocabulary sizes disagree"):
            _load(tmp_path, monkeypatch, model=model)


def test_a_tokenizer_wider_than_the_embedding_is_still_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="no embedding row"):
        _load(tmp_path, monkeypatch, tokenizer_length=EMBEDDING_ROWS + 1)


def test_an_id_above_the_tokenizers_own_size_is_refused_by_the_renderer():
    """The invariant that lets the padded tail be harmless, pinned on its own."""

    tokenizer = _Tokenizer(length=20)
    with pytest.raises(ValueError, match="declared size"):
        resolve(tokenizer, "instructprotein")
    tokenizer = _Tokenizer(length=TOKENIZER_TOKENS)
    record = resolve(tokenizer, "instructprotein").render("MK")
    assert record.token_ids
    assert max(record.token_ids) < TOKENIZER_TOKENS
