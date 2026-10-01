"""CPU contract for native RITA ProteinGym scoring.

No real weights, no GPU, no network. Scoring uses a tiny logit table whose
shifted next-token sums match the author CausalLM cross-entropy. A staged
tokenizer, if present, may be read without loading weights.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.arms import (  # noqa: E402
    EOS_BOUNDED_BOUNDARY,
    INPUT_FORMAT_EOS_BOUNDED_SEQ,
    LOADING_INFO_KEYS,
    STAGED_ARMS,
    Arm,
    Cohort,
    eos_bounded_rendering,
    rendering_marker_ids,
)
from src.capability.core.budget import scored_tokens  # noqa: E402
from src.capability.core.measurement_support import scored_target_records  # noqa: E402
from src.capability.models.rita_fitness import (  # noqa: E402
    NATIVE_EOS_ID,
    NATIVE_PAD_ID,
    RitaFitnessScorer,
    VOCAB_SIZE,
    _encode_native,
    _verify_residue_tokenisation,
)
from src.capability.models.scale_comparison import STRATUM_N_TO_C  # noqa: E402
from src.capability.core.scoring import sequence_target_mask, target_rule  # noqa: E402

AA20_IDS = {
    "L": 3,
    "A": 4,
    "G": 5,
    "V": 6,
    "E": 7,
    "S": 8,
    "I": 9,
    "K": 10,
    "R": 11,
    "D": 12,
    "T": 13,
    "P": 14,
    "N": 15,
    "Q": 16,
    "F": 17,
    "Y": 18,
    "M": 19,
    "H": 20,
    "C": 21,
    "W": 22,
}


def _empty_info() -> dict[str, list[str]]:
    return {key: [] for key in LOADING_INFO_KEYS}


class _FakeRitaTokenizer:
    """Mirrors native RITA ids: <PAD>=1, <EOS>=2, AA20=3..22, empty special map.

    ``eos_token_id``, ``pad_token_id`` and ``unk_token_id`` are all ``None`` on
    the staged checkpoint's tokenizer, which is why its post-processor appends
    ``<EOS>`` of its own accord while the id has to be resolved by token string.
    """

    def __init__(self, *, eos_id: int = NATIVE_EOS_ID, append_eos: int | None = None) -> None:
        self._vocab = {"<unk>": 0, "<PAD>": NATIVE_PAD_ID, "<EOS>": int(eos_id), **AA20_IDS}
        self.eos_token_id = None
        self.pad_token_id = None
        self.unk_token_id = None
        self._append_eos = NATIVE_EOS_ID if append_eos is None else int(append_eos)
        self.backend_tokenizer = SimpleNamespace(
            post_processor=SimpleNamespace(eos_id=self._append_eos)
        )

    def __len__(self) -> int:
        return VOCAB_SIZE

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self._vocab.get(str(token), 0))

    def convert_ids_to_tokens(self, index: int) -> str:
        inverse = {value: key for key, value in self._vocab.items()}
        return inverse.get(int(index), "<unk>")

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        rendered = str(text)
        boundary: list[int] = []
        if rendered.startswith(EOS_BOUNDED_BOUNDARY):
            boundary = [self.convert_tokens_to_ids(EOS_BOUNDED_BOUNDARY)]
            rendered = rendered[len(EOS_BOUNDED_BOUNDARY) :]
        ids = boundary + [self.convert_tokens_to_ids(symbol) for symbol in rendered]
        if add_special_tokens:
            ids = ids + [self._append_eos]
        return ids


class _UnkRitaTokenizer(_FakeRitaTokenizer):
    """Every amino acid collapses to <unk>=0; special tokens stay native."""

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<EOS>":
            return NATIVE_EOS_ID
        if token == "<PAD>":
            return NATIVE_PAD_ID
        return 0


class _PermutingRitaTokenizer(_FakeRitaTokenizer):
    """The boundary is honest; the multi-residue encode is a permutation."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        rendered = str(text)
        if not rendered.startswith(EOS_BOUNDED_BOUNDARY):
            return super().encode(rendered, add_special_tokens=add_special_tokens)
        sequence = rendered[len(EOS_BOUNDED_BOUNDARY) :]
        if len(sequence) <= 1:
            return super().encode(rendered, add_special_tokens=add_special_tokens)
        ids = [self.convert_tokens_to_ids(EOS_BOUNDED_BOUNDARY)] + [
            self.convert_tokens_to_ids(symbol) for symbol in reversed(sequence)
        ]
        if add_special_tokens:
            ids = ids + [self._append_eos]
        return ids


class _RitaDummy(nn.Module):
    def __init__(
        self,
        dtype: torch.dtype,
        config: SimpleNamespace,
        *,
        logits_fn=None,
        vocab_rows: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device("cpu")
        vocab = int(vocab_rows if vocab_rows is not None else config.vocab_size)
        width = int(config.d_model)
        self.embed = nn.Embedding(vocab, width)
        self.lm_head = nn.Linear(width, vocab, bias=False)
        self._logits_fn = logits_fn
        self.forward_kwargs: list[dict[str, object]] = []
        self.to(dtype)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, **kwargs):
        self.forward_kwargs.append(kwargs)
        if self._logits_fn is not None:
            return SimpleNamespace(logits=self._logits_fn(input_ids))
        batch, width = input_ids.shape
        return SimpleNamespace(
            logits=torch.zeros(batch, width, int(self.config.vocab_size))
        )


def _rita_config(**overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "model_type": "rita",
        "architectures": ["RITAModelForCausalLM"],
        "num_layers": 24,
        "d_model": 2048,
        "vocab_size": VOCAB_SIZE,
        "max_seq_len": 1024,
        "eos_token_id": 50256,
        "torch_dtype": "float16",
        "pad_token_id": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _rule_logits(input_ids: torch.Tensor) -> torch.Tensor:
    batch, width = input_ids.shape
    logits = torch.zeros(batch, width, VOCAB_SIZE)
    logits[..., NATIVE_EOS_ID] = 50.0
    logits[..., AA20_IDS["M"]] = 40.0
    logits[..., NATIVE_PAD_ID] = 100.0
    return logits


def _nan_logits(input_ids: torch.Tensor) -> torch.Tensor:
    batch, width = input_ids.shape
    return torch.full((batch, width, VOCAB_SIZE), float("nan"))


def _wide_logits(input_ids: torch.Tensor) -> torch.Tensor:
    batch, width = input_ids.shape
    return torch.zeros(batch, width, 32)


def _scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: nn.Module,
    *,
    tokenizer: _FakeRitaTokenizer | None = None,
    batch_size: int = 2,
    dtype: str = "float32",
    captured: dict[str, object] | None = None,
) -> RitaFitnessScorer:
    checkpoint = tmp_path / "RITA_xl"
    checkpoint.mkdir(exist_ok=True)
    tok = tokenizer or _FakeRitaTokenizer()
    sink = captured if captured is not None else {}

    def fake_tokenizer(path, **kwargs):
        sink["tokenizer_path"] = path
        sink["tokenizer_kwargs"] = kwargs
        return tok

    def fake_model(path, **kwargs):
        sink["model_path"] = path
        sink["model_kwargs"] = kwargs
        return model, _empty_info()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fake_tokenizer)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_model)
    return RitaFitnessScorer(
        checkpoint=checkpoint,
        device="cpu",
        dtype=dtype,
        batch_size=batch_size,
    )


def test_the_scorer_declares_n_to_c_stratum_and_its_scored_span():
    assert RitaFitnessScorer.scoring_stratum == STRATUM_N_TO_C
    assert RitaFitnessScorer.score_description == (
        "document-boundary rendering; summed N-to-C next-token log likelihood "
        "over the sequence's residues"
    )
    assert "qualified" not in RitaFitnessScorer.score_description


def test_native_encoding_carries_both_boundaries_and_scores_the_residues(
    tmp_path, monkeypatch
):
    model = _RitaDummy(torch.float32, _rita_config(), logits_fn=_rule_logits)
    scorer = _scorer(tmp_path, monkeypatch, model, batch_size=1)
    assert scorer.token_lengths(["MKT"]) == [5]
    assert scorer.facts["native_eos_token_id"] == NATIVE_EOS_ID
    assert scorer.facts["native_document_boundary"] == "<EOS>"
    assert "are not scored" in scorer.facts["scoring_span"]
    assert scorer.facts["config_eos_token_id"] == 50256
    assert "ignored" in scorer.facts["config_eos_token_id_status"]
    assert scorer.facts["storage_torch_dtype"] == "float16"
    assert scorer.facts["dtype_requested"] == "float32"
    assert "not a qualified information arm" in scorer.facts["scientific_role"]
    assert "PASS" not in scorer.facts["scientific_role"]

    ids = [NATIVE_EOS_ID, AA20_IDS["M"], AA20_IDS["K"], AA20_IDS["T"], NATIVE_EOS_ID]
    logits = _rule_logits(torch.tensor([ids]))[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    shifted = logp[:-1].gather(-1, torch.tensor(ids[1:]).unsqueeze(-1)).squeeze(-1)
    # The scored targets are the residues: residue 1 is scored because the
    # boundary conditions it, and the tokenizer's terminal EOS is not content.
    expected = float(shifted[:-1].sum().double())
    got = scorer.log_likelihood(["MKT"])
    assert got.shape == (1,)
    assert got[0] == pytest.approx(expected)
    assert abs(float(shifted.sum().double()) - expected) > 1.0
    with_first = logp.gather(-1, torch.tensor(ids).unsqueeze(-1)).squeeze(-1).sum()
    assert abs(float(with_first.double()) - expected) > 1.0
    assert model.forward_kwargs
    assert all("attention_mask" not in kwargs for kwargs in model.forward_kwargs)


def test_batched_and_single_sequence_sums_match_author_shifted_ce(tmp_path, monkeypatch):
    model = _RitaDummy(torch.float32, _rita_config(), logits_fn=_rule_logits)
    sequences = ["MKT", "AAA"]
    batched = _scorer(tmp_path, monkeypatch, model, batch_size=2).log_likelihood(sequences)
    singles = np.array(
        [
            _scorer(tmp_path, monkeypatch, model, batch_size=1).log_likelihood([sequence])[0]
            for sequence in sequences
        ]
    )
    assert batched == pytest.approx(singles)
    expected = []
    for sequence in sequences:
        ids = _FakeRitaTokenizer().encode(eos_bounded_rendering(sequence))
        logits = _rule_logits(torch.tensor([ids]))[0]
        logp = torch.log_softmax(logits.float(), dim=-1)
        shifted = logp[:-1].gather(-1, torch.tensor(ids[1:]).unsqueeze(-1)).squeeze(-1)
        expected.append(float(shifted[:-1].sum().double()))
    assert batched == pytest.approx(np.array(expected))


def test_variable_length_illegal_aa_over_context_and_nonfinite_are_refused(
    tmp_path, monkeypatch
):
    model = _RitaDummy(
        torch.float32, _rita_config(max_seq_len=4), logits_fn=_rule_logits
    )
    scorer = _scorer(tmp_path, monkeypatch, model)
    with pytest.raises(ValueError, match="variable-length"):
        scorer.log_likelihood(["MK", "MKT"])
    for illegal in ("MKTZ", "X", "O", "U", "B", "mkt", ""):
        with pytest.raises(ValueError, match="non-canonical"):
            scorer.log_likelihood([illegal])
        with pytest.raises(ValueError, match="non-canonical"):
            scorer.token_lengths([illegal])

    lengths = scorer.token_lengths(["MKTA"])
    assert lengths == [6]
    with pytest.raises(ValueError, match="exceeds this checkpoint"):
        scorer.log_likelihood(["MKTA"])

    broken = _RitaDummy(
        torch.float32, _rita_config(), logits_fn=_nan_logits
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        _scorer(tmp_path, monkeypatch, broken).log_likelihood(["MK"])

    alive = _scorer(tmp_path, monkeypatch, model)
    alive.release()
    assert not hasattr(alive, "model")


def test_wrong_eos_mapping_and_output_shape_are_refused(tmp_path, monkeypatch):
    model = _RitaDummy(torch.float32, _rita_config(), logits_fn=_rule_logits)
    with pytest.raises(ValueError, match="native <EOS>"):
        _scorer(
            tmp_path,
            monkeypatch,
            model,
            tokenizer=_FakeRitaTokenizer(eos_id=50256),
        )
    with pytest.raises(ValueError, match="native encoding must be"):
        _scorer(
            tmp_path,
            monkeypatch,
            model,
            tokenizer=_FakeRitaTokenizer(append_eos=50256),
        )
    wide = _RitaDummy(torch.float32, _rita_config(), logits_fn=_wide_logits)
    with pytest.raises(ValueError, match="logits must have shape"):
        _scorer(tmp_path, monkeypatch, wide).log_likelihood(["MKT"])


def test_strict_loader_refuses_loading_diagnostics_and_non_float32(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "RITA_xl"
    checkpoint.mkdir()
    tokenizer = _FakeRitaTokenizer()
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )

    def fake_dirty(_path, **_kwargs):
        info = _empty_info()
        info["missing_keys"] = ["lm_head.weight"]
        return _RitaDummy(torch.float32, _rita_config()), info

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_dirty)
    with pytest.raises(ValueError, match="missing_keys"):
        RitaFitnessScorer(
            checkpoint=checkpoint, device="cpu", dtype="float32", batch_size=1
        )

    with pytest.raises(ValueError, match="accepts only float32"):
        RitaFitnessScorer(
            checkpoint=checkpoint, device="cpu", dtype="float16", batch_size=1
        )

    def fake_half(_path, **kwargs):
        assert kwargs["torch_dtype"] is torch.float32
        assert kwargs["trust_remote_code"] is True
        assert kwargs["local_files_only"] is True
        assert kwargs["output_loading_info"] is True
        return _RitaDummy(torch.float16, _rita_config()), _empty_info()

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_half)
    with pytest.raises(ValueError, match="requested inference dtype"):
        RitaFitnessScorer(
            checkpoint=checkpoint, device="cpu", dtype="float32", batch_size=1
        )


def test_strict_loader_records_rita_facts_without_adopting_config_eos(
    tmp_path, monkeypatch
):
    captured: dict[str, object] = {}
    model = _RitaDummy(torch.float32, _rita_config(), logits_fn=_rule_logits)
    scorer = _scorer(tmp_path, monkeypatch, model, captured=captured)
    kwargs = captured["model_kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["trust_remote_code"] is True
    assert kwargs["local_files_only"] is True
    assert kwargs["output_loading_info"] is True
    assert kwargs["torch_dtype"] is torch.float32
    assert kwargs["device_map"] == {"": "cpu"}
    tok_kwargs = captured["tokenizer_kwargs"]
    assert isinstance(tok_kwargs, dict)
    assert tok_kwargs["local_files_only"] is True
    assert scorer.context == 1024
    assert scorer.facts["model_type"] == "rita"
    assert scorer.facts["vocab_size"] == 26
    assert scorer.facts["n_layers"] == 24
    assert scorer.facts["d_model"] == 2048
    assert scorer.facts["embedding_rows"] == 26
    assert scorer.facts["lm_head_rows"] == 26
    assert scorer.facts["pad_used_for_scoring"] is False
    assert scorer.tokenizer.pad_token_id is None
    assert scorer.tokenizer.eos_token_id is None
    assert scorer.model.config.eos_token_id == 50256


def test_a_missing_checkpoint_directory_is_refused_without_a_hub_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="MODEL_ROOT"):
        RitaFitnessScorer(
            checkpoint=tmp_path / "absent",
            device="cpu",
            dtype="float32",
            batch_size=1,
        )


def test_all_residues_mapping_to_unk_are_refused_before_scoring():
    with pytest.raises(ValueError, match="3\\.\\.22"):
        _verify_residue_tokenisation(_UnkRitaTokenizer(), eos_id=NATIVE_EOS_ID)


def test_permuted_multiresidue_encoding_is_refused_before_scoring():
    tokenizer = _PermutingRitaTokenizer()
    with pytest.raises(ValueError, match="per-letter id list"):
        _verify_residue_tokenisation(tokenizer, eos_id=NATIVE_EOS_ID)
    mapping = {
        residue: tokenizer.convert_tokens_to_ids(residue)
        for residue in AA20_IDS
    }
    with pytest.raises(ValueError, match="per-letter id list"):
        _encode_native(
            tokenizer,
            eos_bounded_rendering("MKT"),
            eos_id=NATIVE_EOS_ID,
            residue_ids=mapping,
        )


def test_a_bare_residue_string_is_refused_rather_than_rendered():
    """The rendering is the arm's, and this door does not supply it silently.

    ``_encode_native`` used to take a bare residue string, which is exactly the
    input the checkpoint was not given; a caller that means to score a sequence
    renders it through :func:`eos_bounded_rendering` first.
    """

    tokenizer = _FakeRitaTokenizer()
    mapping = {residue: tokenizer.convert_tokens_to_ids(residue) for residue in AA20_IDS}
    with pytest.raises(ValueError, match="document boundary"):
        _encode_native(tokenizer, "MKT", eos_id=NATIVE_EOS_ID, residue_ids=mapping)
    assert _encode_native(
        tokenizer,
        eos_bounded_rendering("MKT"),
        eos_id=NATIVE_EOS_ID,
        residue_ids=mapping,
    ) == [2, 19, 10, 13, 2]


def test_staged_tokenizer_native_mkt_keeps_eos_without_loading_weights():
    from src.capability.core.arms import STAGED_ARMS

    path = STAGED_ARMS["rita-xl"].path
    if not path.exists():
        pytest.skip(
            "MODEL_ROOT is not set to the staged model root; "
            "RITA_xl is not missing, the path configuration is"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    mapping = _verify_residue_tokenisation(tokenizer, eos_id=NATIVE_EOS_ID)
    assert mapping["M"] == 19
    assert mapping["K"] == 10
    assert mapping["T"] == 13
    assert set(mapping.values()) == set(range(3, 23))
    assert tokenizer.encode("MKT") == [19, 10, 13, 2]
    assert tokenizer.encode("MKT", add_special_tokens=False) == [19, 10, 13]
    assert _encode_native(
        tokenizer,
        eos_bounded_rendering("MKT"),
        eos_id=NATIVE_EOS_ID,
        residue_ids=mapping,
    ) == [2, 19, 10, 13, 2]
    assert tokenizer.convert_tokens_to_ids("<EOS>") == 2
    assert tokenizer.convert_tokens_to_ids("<PAD>") == 1


def _rita_arm(tokenizer, model: nn.Module | None = None) -> Arm:
    """The staged RITA spec with a real tokenizer and no loaded weights."""

    return Arm(
        spec=STAGED_ARMS["rita-xl"],
        model=(
            model
            if model is not None
            else SimpleNamespace(config=SimpleNamespace(vocab_size=VOCAB_SIZE))
        ),
        tokenizer=tokenizer,
        device="cpu",
        dtype="float32",
    )


def test_rita_renders_the_document_boundary_and_declares_one_marker():
    """The rendered id sequence, pinned, with both boundaries and the marker id.

    The published post-processor is Sequence-then-``<EOS>``, so it prefixes
    nothing and the leading boundary has to be rendered; the trailing one is the
    tokenizer's own. Both are id 2, which is why the declared marker is a single
    id and why the scored targets are the residues.
    """

    tokenizer = _FakeRitaTokenizer()
    arm = _rita_arm(tokenizer)
    assert arm.spec.input_format == INPUT_FORMAT_EOS_BOUNDED_SEQ == "eos_bounded_seq"
    assert eos_bounded_rendering("MKT") == "<EOS>MKT"
    assert rendering_marker_ids(arm) == (NATIVE_EOS_ID,)
    assert target_rule(arm.spec.input_format) == "all_valid"
    cohort = Cohort(
        name="stub", kind="protein", records=["MKT", "AAA"], min_symbols=0, max_symbols=8
    )
    rendered = cohort.input_strings(arm)
    assert rendered == ["<EOS>MKT", "<EOS>AAA"]
    # The bare rendering this arm was scored under is gone, and it is not a
    # subsequence of the new one: the format carries a token it never sent.
    assert "MKT" not in rendered and "AAA" not in rendered
    assert tokenizer.encode(rendered[0], add_special_tokens=False) == [2, 19, 10, 13]
    assert tokenizer.encode(rendered[0]) == [2, 19, 10, 13, 2]


def test_the_declared_span_is_the_residues_and_its_size_is_unchanged():
    """One target per residue, on the gate's path and on the reference path.

    The span is resolved from the arm's own declarations -- ``target_rule`` and
    ``rendering_marker_ids`` -- rather than from a hand-written slice of the id
    list, and it holds the size the bare rendering scored: one target per
    residue, L. What moved is which token is scored at each position -- residue 1
    is scored here, conditioned on the boundary, and the terminal ``<EOS>`` that
    the bare rendering scored in its place is a marker and is not -- so the
    count is a like-for-like one and the membership is not.
    """

    tokenizer = _FakeRitaTokenizer()
    arm = _rita_arm(
        tokenizer, _RitaDummy(torch.float32, _rita_config(), logits_fn=_rule_logits)
    )
    records = ["MKT", "AAA"]
    cohort = Cohort(
        name="stub", kind="protein", records=records, min_symbols=0, max_symbols=8
    )
    rendered = cohort.input_strings(arm)
    scored = scored_tokens(arm, rendered, max_len=8, batch_size=1)
    counts, _ = scored_target_records(arm, rendered, max_len=8)
    residue_total = sum(len(record) for record in records)
    assert scored.target_ids.size == residue_total
    assert NATIVE_EOS_ID not in set(scored.target_ids.tolist())
    assert (counts == np.bincount(scored.target_ids, minlength=VOCAB_SIZE)).all()
    assert int(counts[NATIVE_EOS_ID]) == 0
    assert int(counts.sum()) == residue_total

    tensor = torch.tensor([_FakeRitaTokenizer().encode(rendered[0])])
    keep = sequence_target_mask(
        tensor,
        torch.ones_like(tensor),
        rule=target_rule(arm.spec.input_format),
        marker_token_ids=rendering_marker_ids(arm),
    )[0]
    assert keep.tolist() == [True, True, True, False]
    assert int(keep.sum()) == len(records[0])
