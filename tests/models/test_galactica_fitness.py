"""CPU contract for native Galactica ProteinGym scoring.

No real weights, no GPU, no network. Tokenisation uses the existing Galactica
stub that honours the declared split-marker escape. Scoring uses a tiny logit
table whose next-token sums can be checked by an independent predecessor formula.
"""

from __future__ import annotations

import importlib.util
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.models import galactica_fitness as G  # noqa: E402
from src.capability.core.arms import LOADING_INFO_KEYS  # noqa: E402
from src.capability.models.joint_modes import RESIDUE_UNIT, resolve  # noqa: E402
from src.capability.models.scale_comparison import STRATUM_N_TO_C  # noqa: E402
from tests.core.test_joint_mode_qualification import galactica_stub  # noqa: E402


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


def _empty_info() -> dict[str, list[str]]:
    return {key: [] for key in LOADING_INFO_KEYS}


class _PadTokenizer:
    """Galactica stub plus config-declared pad id 1 and a declared vocabulary length."""

    def __init__(self, vocab_size: int = 50000) -> None:
        self._inner = galactica_stub()
        self._inner._vocab["<pad>"] = G.DECLARED_PAD_TOKEN_ID
        self._inner._inverse[G.DECLARED_PAD_TOKEN_ID] = "<pad>"
        self.pad_token: str | None = None
        self._vocab_size = int(vocab_size)

    def __len__(self) -> int:
        return self._vocab_size

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


class _RuleLogits(nn.Module):
    """Logits vary with position and target id; START/END/pad columns stay huge."""

    def __init__(self, vocab_size: int, *, start_id: int, end_id: int, pad_id: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.start_id = int(start_id)
        self.end_id = int(end_id)
        self.pad_id = int(pad_id)
        self.device = torch.device("cpu")
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        assert use_cache is False
        batch, width = input_ids.shape
        position = torch.arange(width, dtype=torch.float32).view(1, width, 1)
        vocab = torch.arange(self.vocab_size, dtype=torch.float32).view(1, 1, -1)
        logits = (0.25 * position + 0.05 * vocab).expand(batch, width, self.vocab_size).contiguous()
        logits[..., self.end_id] = 50.0
        logits[..., self.start_id] = 40.0
        logits[..., self.pad_id] = 100.0
        return SimpleNamespace(logits=logits)


class _NanLogits(_RuleLogits):
    def forward(self, input_ids, attention_mask=None, use_cache=None):
        assert use_cache is False
        batch, width = input_ids.shape
        logits = torch.full((batch, width, self.vocab_size), float("nan"))
        return SimpleNamespace(logits=logits)


class _WrongShapeLogits(_RuleLogits):
    def forward(self, input_ids, attention_mask=None, use_cache=None):
        assert use_cache is False
        batch, width = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(batch, width, self.vocab_size - 1))


class _DummyLM(nn.Module):
    def __init__(
        self,
        dtype: torch.dtype,
        config: SimpleNamespace,
        *,
        input_vocab: int | None = None,
        output_vocab: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device("cpu")
        self.weight = nn.Parameter(torch.ones(1, dtype=dtype))
        vocab = int(config.vocab_size)
        self._input = SimpleNamespace(
            num_embeddings=int(vocab if input_vocab is None else input_vocab)
        )
        self._output = SimpleNamespace(
            out_features=int(vocab if output_vocab is None else output_vocab)
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
        "num_hidden_layers": 12,
        "hidden_size": 768,
        "vocab_size": 50000,
        "max_position_embeddings": 2048,
        "pad_token_id": G.DECLARED_PAD_TOKEN_ID,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _prepared_tokenizer(vocab_size: int = 50000) -> _PadTokenizer:
    tokenizer = _PadTokenizer(vocab_size=vocab_size)
    tokenizer.pad_token = "<pad>"
    return tokenizer


def _scorer(
    model: _RuleLogits, *, context: int = 64, batch_size: int = 2
) -> G.GalacticaFitnessScorer:
    # The stub's declared size must cover its own vocabulary, because the renderer
    # refuses an emitted id at or above len(tokenizer); the default is the released
    # tokenizer's own 50000. `model.vocab_size` is the logit-table width these tests
    # keep small, which is a different quantity.
    tokenizer = _prepared_tokenizer()
    tokenisation = resolve(tokenizer, "galactica")
    loaded = G.LoadedGalactica(
        name="galactica-1.3b",
        model=model,
        tokenizer=tokenizer,
        tokenisation=tokenisation,
        context=context,
        facts={
            "rung": "galactica-1.3b",
            "vocab_size": int(model.vocab_size),
            "scientific_role": "cpu test-stub metadata, not a GPU observation",
        },
    )
    return G.GalacticaFitnessScorer(loaded, batch_size=batch_size)


def _residue_model(sequences: list[str]) -> tuple[_PadTokenizer, _RuleLogits, list]:
    probe = _prepared_tokenizer()
    tokenisation = resolve(probe, "galactica")
    records = [tokenisation.render(sequence, context=None) for sequence in sequences]
    vocab = max(max(record.token_ids) for record in records) + 8
    tokenizer = _prepared_tokenizer(vocab)
    start_id = tokenizer.convert_tokens_to_ids("[START_AMINO]")
    end_id = tokenizer.convert_tokens_to_ids("[END_AMINO]")
    assert isinstance(start_id, int)
    assert isinstance(end_id, int)
    model = _RuleLogits(
        vocab, start_id=start_id, end_id=end_id, pad_id=G.DECLARED_PAD_TOKEN_ID
    )
    return tokenizer, model, records


def _predecessor_residue_sum(
    ids: list[int], scored: tuple[int, ...], logits: torch.Tensor
) -> float:
    """Sum log p(ids[p] | logits[p-1]) over scored targets, full-vocabulary softmax."""

    total = 0.0
    for position in scored:
        row = logits[position - 1].float()
        total += float(row[ids[position]] - torch.logsumexp(row, dim=0))
    return total


def _reference_logits(model: _RuleLogits, ids: list[int]) -> torch.Tensor:
    return model(torch.tensor([ids]), use_cache=False).logits[0]


def _previous_batch_alive(scorer: G.GalacticaFitnessScorer, sequences: list[str]):
    """CPU lifetime of full-vocab logits/logp at each later forward; not GPU qualification."""
    refs: dict[str, weakref.ReferenceType[torch.Tensor]] = {}
    entries: list[dict[str, bool]] = []
    model = scorer.loaded.model
    inner = model.forward
    orig_softmax = torch.log_softmax

    def forward(input_ids, attention_mask=None, use_cache=None):
        entries.append({key: ref() is not None for key, ref in refs.items()})
        output = inner(input_ids, attention_mask=attention_mask, use_cache=use_cache)
        refs["logits"] = weakref.ref(output.logits)
        return output

    def softmax(*args, **kwargs):
        values = orig_softmax(*args, **kwargs)
        refs["logp"] = weakref.ref(values)
        return values

    with patch.object(model, "forward", forward), patch.object(torch, "log_softmax", softmax):
        totals = scorer.log_likelihood(sequences)
    after = {key: ref() is not None for key, ref in refs.items()}
    return entries, after, totals


# ------------------------------------------------------------- declarations


def test_the_four_rungs_are_optional_stage20_arms_and_not_the_default_run():
    assert G.GALACTICA_RUNGS == (
        "galactica-125m",
        "galactica-1.3b",
        "galactica-6.7b",
        "galactica-30b",
    )
    for name in G.GALACTICA_RUNGS:
        assert G.CHECKPOINTS[name] == G.MODEL_ROOT / name
        assert name in STAGE20.SCOREABLE_ARMS
        record = STAGE20.corpus_record(name)
        assert record["declared"] == "galactica_scientific_corpus"
        assert record["identification"] == (
            "external UniRef50 profile baseline, not a retrieval bound"
        )
        assert "fitness verdict" in record["note"] or name == "galactica-125m"
    note_125m = STAGE20.corpus_record("galactica-125m")["note"]
    assert "small-scale reference" in note_125m
    assert "not a ProteinGym PASS or FAIL" in note_125m
    assert sorted(STAGE20.ARM_CORPUS) == ["progen2-medium", "progen3-112m", "protgpt2"]
    assert set(STAGE20.ARM_CORPUS).isdisjoint(G.GALACTICA_RUNGS)


def test_the_scorer_declares_the_residue_unit_and_n_to_c_stratum():
    assert G.GalacticaFitnessScorer.symbol_unit == RESIDUE_UNIT
    assert G.GalacticaFitnessScorer.scoring_stratum == STRATUM_N_TO_C
    assert "protein-content" in G.GalacticaFitnessScorer.score_description
    assert "full-vocabulary" in G.GalacticaFitnessScorer.score_description


def test_an_unknown_rung_is_refused_by_name():
    with pytest.raises(KeyError, match="unknown Galactica rung"):
        G.rung("galactica-70b")


# ------------------------------------------------------------- tokenisation / scoring


def test_declared_galactica_tokenisation_scores_residues_and_excludes_boundaries():
    tokenizer, model, records = _residue_model(["MK"])
    record = records[0]
    assert record.text.startswith("[START_AMINO]")
    assert not record.text.startswith("#")
    assert record.n_residues == record.n_scored_tokens == 2
    start_id = tokenizer.convert_tokens_to_ids("[START_AMINO]")
    end_id = tokenizer.convert_tokens_to_ids("[END_AMINO]")
    assert record.token_ids[0] == start_id
    assert record.token_ids[-1] == end_id
    assert start_id not in {record.token_ids[i] for i in record.scored_positions}
    assert end_id not in {record.token_ids[i] for i in record.scored_positions}

    scorer = _scorer(model, batch_size=1)
    ids = list(record.token_ids)
    logits = _reference_logits(model, ids)
    expected = _predecessor_residue_sum(ids, record.scored_positions, logits)
    got = scorer.log_likelihood(["MK"])
    assert got.shape == (1,)
    assert got[0] == pytest.approx(expected)
    assert abs(got[0] - expected / record.n_residues) > 0.1
    end_keep = tuple(record.scored_positions) + (len(record.token_ids) - 1,)
    with_end = _predecessor_residue_sum(ids, end_keep, logits)
    assert abs(with_end - expected) > 1.0
    accounting = scorer.residue_accounting()
    assert accounting["residues_per_scored_token"] == pytest.approx(1.0)


def test_previous_batch_full_vocab_outputs_die_before_the_next_forward():
    sequences = ["MK", "MKT", "MKTAA", "MKTA", "M"]
    _tokenizer, model, records = _residue_model(sequences)
    batch_size = 2
    assert len(sequences) == 2 * batch_size + 1
    scorer = _scorer(model, batch_size=batch_size)
    entries, after, totals = _previous_batch_alive(scorer, sequences)
    assert entries[0] == {}
    assert entries[1:] == [{"logits": False, "logp": False}] * 2
    assert after == {"logits": False, "logp": False}

    expected = np.array(
        [
            _predecessor_residue_sum(
                list(record.token_ids),
                record.scored_positions,
                _reference_logits(model, list(record.token_ids)),
            )
            for record in records
        ]
    )
    assert totals == pytest.approx(expected)
    accounting = scorer.residue_accounting()
    residues = sum(record.n_residues for record in records)
    scored = sum(record.n_scored_tokens for record in records)
    assert accounting["residues"] == residues == accounting["scored_tokens"] == scored
    assert accounting["residues_per_scored_token"] == pytest.approx(1.0)


def test_batched_and_single_sequence_sums_match_and_pad_is_not_scored():
    sequences = ["MK", "MKTAA"]
    tokenizer, model, records = _residue_model(sequences)
    del tokenizer
    scorer = _scorer(model, batch_size=2)
    batched = scorer.log_likelihood(sequences)
    single = np.array(
        [_scorer(model, batch_size=1).log_likelihood([sequence])[0] for sequence in sequences]
    )
    assert batched == pytest.approx(single)
    expected = []
    for record in records:
        ids = list(record.token_ids)
        logits = _reference_logits(model, ids)
        expected.append(_predecessor_residue_sum(ids, record.scored_positions, logits))
    assert batched == pytest.approx(np.array(expected))


def test_non_canonical_symbols_and_over_context_and_nonfinite_are_refused():
    tokenizer, model, records = _residue_model(["MK"])
    vocab = int(model.vocab_size)
    start_id = tokenizer.convert_tokens_to_ids("[START_AMINO]")
    end_id = tokenizer.convert_tokens_to_ids("[END_AMINO]")
    assert isinstance(start_id, int)
    assert isinstance(end_id, int)
    scorer = _scorer(model, context=64)
    with pytest.raises(ValueError, match="non-canonical"):
        scorer.log_likelihood(["MKTZ"])
    with pytest.raises(ValueError, match="non-canonical"):
        scorer.token_lengths(["M1K"])

    short = _scorer(model, context=3)
    lengths = short.token_lengths(["MK"])
    assert lengths[0] > 3
    with pytest.raises(ValueError, match="exceeds this checkpoint"):
        short.log_likelihood(["MK"])

    broken = _scorer(
        _NanLogits(vocab, start_id=start_id, end_id=end_id, pad_id=G.DECLARED_PAD_TOKEN_ID)
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        broken.log_likelihood(["MK"])
    alive = _scorer(model)
    alive.release()
    assert not hasattr(alive, "loaded")
    assert records[0].n_residues == 2


def test_a_failed_render_does_not_reuse_the_previous_batch():
    _tokenizer, model, _records = _residue_model(["MK", "MKTAA"])
    scorer = _scorer(model, batch_size=2)
    first = scorer.log_likelihood(["MK", "MKTAA"])
    assert first.shape == (2,)
    for _ in range(2):
        with pytest.raises(ValueError, match="non-canonical"):
            scorer.log_likelihood(["MKTZ"])
    again = scorer.log_likelihood(["MK", "MKTAA"])
    assert again == pytest.approx(first)
    later = scorer.log_likelihood(["MK"])
    assert later.shape == (1,)
    assert later[0] == pytest.approx(first[0])


# ------------------------------------------------------------- loader refusals


def test_strict_loader_refuses_loading_diagnostics_and_dtype_mismatch(tmp_path, monkeypatch):
    checkpoint = tmp_path / "galactica-1.3b"
    checkpoint.mkdir()
    monkeypatch.setitem(G.CHECKPOINTS, "galactica-1.3b", checkpoint)
    tokenizer = _PadTokenizer()

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )

    with pytest.raises(ValueError, match="unsupported inference dtype"):
        G.load_galactica("galactica-1.3b", device="cpu", dtype="float64")

    def fake_dirty(_path, **_kwargs):
        info = _empty_info()
        info["missing_keys"] = ["lm_head.weight"]
        return _DummyLM(torch.float32, _opt_config()), info

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_dirty)
    with pytest.raises(ValueError, match="missing_keys"):
        G.load_galactica("galactica-1.3b", device="cpu", dtype="float32")

    def fake_wrong_dtype(_path, **kwargs):
        assert kwargs["output_loading_info"] is True
        assert kwargs["local_files_only"] is True
        return _DummyLM(torch.float32, _opt_config()), _empty_info()

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", fake_wrong_dtype
    )
    with pytest.raises(ValueError, match="requested dtype"):
        G.load_galactica("galactica-1.3b", device="cpu", dtype="bfloat16")


def test_strict_loader_refuses_non_opt_and_vocabulary_mismatch(tmp_path, monkeypatch):
    checkpoint = tmp_path / "galactica-1.3b"
    checkpoint.mkdir()
    monkeypatch.setitem(G.CHECKPOINTS, "galactica-1.3b", checkpoint)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: _PadTokenizer(),
    )

    def fake_gpt2(_path, **_kwargs):
        return _DummyLM(torch.float32, _opt_config(model_type="gpt2")), _empty_info()

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_gpt2)
    with pytest.raises(ValueError, match="expected an OPT checkpoint, got 'gpt2'"):
        G.load_galactica("galactica-1.3b", device="cpu", dtype="float32")

    def fake_missing_type(_path, **_kwargs):
        config = _opt_config()
        del config.model_type
        return _DummyLM(torch.float32, config), _empty_info()

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", fake_missing_type
    )
    with pytest.raises(ValueError, match="expected an OPT checkpoint, got None"):
        G.load_galactica("galactica-1.3b", device="cpu", dtype="float32")

    def fake_input_mismatch(_path, **_kwargs):
        return _DummyLM(torch.float32, _opt_config(), input_vocab=49999), _empty_info()

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", fake_input_mismatch
    )
    with pytest.raises(ValueError, match="vocabulary sizes disagree"):
        G.load_galactica("galactica-1.3b", device="cpu", dtype="float32")

    def fake_output_mismatch(_path, **_kwargs):
        return _DummyLM(torch.float32, _opt_config(), output_vocab=49998), _empty_info()

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", fake_output_mismatch
    )
    with pytest.raises(ValueError, match="vocabulary sizes disagree"):
        G.load_galactica("galactica-1.3b", device="cpu", dtype="float32")


def test_scorer_facts_are_the_loaded_observation():
    _tokenizer, model, _records = _residue_model(["MK"])
    scorer = _scorer(model, batch_size=1)
    assert scorer.facts is scorer.loaded.facts
    marker = "cpu test-stub metadata, not a GPU observation"
    scorer.loaded.facts["scientific_role"] = marker
    assert scorer.facts["scientific_role"] is marker
    assert "vocab_size" in scorer.facts


def test_scorer_refuses_logits_that_do_not_match_the_loaded_vocabulary():
    _tokenizer, model, _records = _residue_model(["MK"])
    broken = _WrongShapeLogits(
        int(model.vocab_size),
        start_id=model.start_id,
        end_id=model.end_id,
        pad_id=model.pad_id,
    )
    scorer = _scorer(broken, batch_size=1)
    with pytest.raises(RuntimeError, match="expected logits shape"):
        scorer.log_likelihood(["MK"])


def test_strict_loader_adopts_pad_id_1_and_records_opt_facts(tmp_path, monkeypatch):
    checkpoint = tmp_path / "galactica-125m"
    checkpoint.mkdir()
    monkeypatch.setitem(G.CHECKPOINTS, "galactica-125m", checkpoint)
    tokenizer = _PadTokenizer()
    captured: dict[str, object] = {}

    def fake_tokenizer(path, **kwargs):
        captured["tokenizer_path"] = path
        captured["tokenizer_local"] = kwargs.get("local_files_only")
        return tokenizer

    def fake_model(_path, **kwargs):
        captured.update(kwargs)
        return _DummyLM(torch.float32, _opt_config()), _empty_info()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fake_tokenizer)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_model)
    loaded = G.load_galactica("galactica-125m", device="cpu", dtype="float32")
    assert tokenizer.pad_token_id == G.DECLARED_PAD_TOKEN_ID
    assert loaded.facts["pad_token_id"] == G.DECLARED_PAD_TOKEN_ID
    assert loaded.facts["model_type"] == "opt"
    assert loaded.facts["architectures"] == ["OPTForCausalLM"]
    assert loaded.facts["n_layers"] == 12
    assert loaded.facts["d_model"] == 768
    assert loaded.facts["vocab_size"] == 50000
    assert loaded.facts["input_vocab_size"] == 50000
    assert loaded.facts["output_vocab_size"] == 50000
    assert loaded.facts["tokenizer_vocab_size"] == 50000
    assert loaded.facts["context"] == 2048
    assert "PASS" not in loaded.facts["scientific_role"]
    assert "unidentified" in loaded.facts["scientific_role"]
    assert captured["local_files_only"] is True
    assert captured["output_loading_info"] is True
    assert captured["tokenizer_local"] is True
    assert len(tokenizer) == 50000


def test_a_missing_checkpoint_directory_is_refused_without_a_hub_fallback(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(G.CHECKPOINTS, "galactica-30b", tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="MODEL_ROOT"):
        G.load_galactica("galactica-30b", device="cpu", dtype="float32")
