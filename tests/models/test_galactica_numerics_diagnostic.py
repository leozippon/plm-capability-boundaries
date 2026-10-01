"""CPU contract for the Galactica-30B synthetic numerical diagnostic.

Fixtures are tiny logit tables. They do not claim H200 numerical stability and
do not load real weights.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.models import galactica_fitness as G  # noqa: E402
from src.capability.core.arms import AA20  # noqa: E402
from src.capability.models.joint_modes import resolve  # noqa: E402
from tests.core.test_joint_mode_qualification import galactica_stub  # noqa: E402


def _load_script():
    path = REPO_ROOT / "scripts/capability/mutation/diagnose_galactica_numerics.py"
    spec = importlib.util.spec_from_file_location("diagnose_galactica_numerics", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DIAG = _load_script()


class _PadTokenizer:
    def __init__(self, vocab_size: int) -> None:
        self._inner = galactica_stub()
        self._inner._vocab["<pad>"] = G.DECLARED_PAD_TOKEN_ID
        self._inner._inverse[G.DECLARED_PAD_TOKEN_ID] = "<pad>"
        self.pad_token = "<pad>"
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
        return self._inner.convert_tokens_to_ids(self.pad_token)


class _WidthLogits(nn.Module):
    """Position/target logits that also see batch index and tensor width."""

    vocab_size: int

    def __init__(
        self,
        vocab_size: int,
        *,
        start_id: int,
        end_id: int,
        pad_id: int,
        dtype: torch.dtype = torch.bfloat16,
        drop_fp32_label_after: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.start_id = int(start_id)
        self.end_id = int(end_id)
        self.pad_id = int(pad_id)
        self.device = torch.device("cpu")
        self.drop_fp32_label_after = drop_fp32_label_after
        self._fp32_label_forwards = 0
        self.weight = nn.Parameter(torch.ones(1, dtype=dtype))

    def _logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, width = input_ids.shape
        position = torch.arange(width, dtype=torch.float32).view(1, width, 1)
        vocab = torch.arange(self.vocab_size, dtype=torch.float32).view(1, 1, -1)
        batch_idx = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1)
        logits = (
            0.25 * position
            + 0.05 * vocab
            + 0.01 * width * vocab
            + 0.001 * batch_idx * vocab
        ).expand(batch, width, self.vocab_size).contiguous()
        logits[..., self.end_id] = 50.0
        logits[..., self.start_id] = 40.0
        logits[..., self.pad_id] = 100.0
        return logits

    def forward(self, input_ids, attention_mask=None, use_cache=None, labels=None):
        assert use_cache is False
        logits = self._logits(input_ids)
        if labels is None:
            return SimpleNamespace(logits=logits)
        if self.weight.dtype == torch.float32:
            self._fp32_label_forwards += 1
            if (
                self.drop_fp32_label_after is not None
                and self._fp32_label_forwards >= int(self.drop_fp32_label_after)
            ):
                return SimpleNamespace(logits=logits)
        shift_logits = logits[:, :-1].float()
        shift_labels = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, self.vocab_size),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )
        return SimpleNamespace(logits=logits, loss=loss)


class _NanLogits(_WidthLogits):
    def forward(self, input_ids, attention_mask=None, use_cache=None, labels=None):
        assert use_cache is False
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.vocab_size), float("nan")
        )
        if labels is None:
            return SimpleNamespace(logits=logits)
        return SimpleNamespace(logits=logits, loss=torch.tensor(float("nan")))


class _NoLossLogits(_WidthLogits):
    def forward(self, input_ids, attention_mask=None, use_cache=None, labels=None):
        assert use_cache is False
        return SimpleNamespace(logits=self._logits(input_ids))


def _tokenizer_and_ids() -> tuple[_PadTokenizer, int, int, int]:
    probe = _PadTokenizer(64)
    tokenisation = resolve(probe, "galactica")
    records = [
        tokenisation.render(sequence, context=None)
        for sequence in (DIAG.SHORT_SEQUENCE, DIAG.LONG_SEQUENCE)
    ]
    vocab = max(max(record.token_ids) for record in records) + 8
    tokenizer = _PadTokenizer(vocab)
    start_id = tokenizer.convert_tokens_to_ids("[START_AMINO]")
    end_id = tokenizer.convert_tokens_to_ids("[END_AMINO]")
    assert isinstance(start_id, int)
    assert isinstance(end_id, int)
    return tokenizer, vocab, start_id, end_id


def _scorer(model: _WidthLogits, tokenizer: _PadTokenizer) -> G.GalacticaFitnessScorer:
    tokenisation = resolve(tokenizer, "galactica")
    loaded = G.LoadedGalactica(
        name="galactica-30b",
        model=model,
        tokenizer=tokenizer,
        tokenisation=tokenisation,
        context=64,
        facts={
            "rung": "galactica-30b",
            "vocab_size": int(model.vocab_size),
            "dtype_requested": "bfloat16",
            "dtype_observed": ["bfloat16"],
            "scientific_role": "cpu test-stub metadata, not a GPU observation",
        },
    )
    return G.GalacticaFitnessScorer(loaded, batch_size=DIAG.PUBLIC_BATCH_SIZE)


def _ready_scorer(
    model_cls: type[_WidthLogits] = _WidthLogits,
    **model_kwargs: object,
) -> G.GalacticaFitnessScorer:
    tokenizer, vocab, start_id, end_id = _tokenizer_and_ids()
    model = model_cls(
        vocab,
        start_id=start_id,
        end_id=end_id,
        pad_id=G.DECLARED_PAD_TOKEN_ID,
        **model_kwargs,  # type: ignore[arg-type]
    )
    return _scorer(model, tokenizer)


def test_the_seven_cases_and_target_denominators_are_fixed():
    assert DIAG.ALLOWED_ARM == "galactica-30b"
    assert DIAG.SHORT_SEQUENCE == "MKT"
    assert DIAG.LONG_SEQUENCE == AA20
    assert [case["name"] for case in DIAG.CASES] == [
        "single_short",
        "single_long",
        "mixed",
        "single_short_padded",
        "duplicate_short",
        "duplicate_short_padded",
        "duplicate_long",
    ]
    assert DIAG.REPETITIONS == 3
    assert {item[4] for item in DIAG.COMPARISONS} == {3, 20}
    assert ("mixed", 0, "single_short", 0, 3) in DIAG.COMPARISONS
    assert ("mixed", 1, "single_long", 0, 20) in DIAG.COMPARISONS
    assert [item for item in DIAG.COMPARISONS if item[0] == "duplicate_short"] == [
        ("duplicate_short", 0, "single_short", 0, 3),
        ("duplicate_short", 1, "single_short", 0, 3),
    ]
    assert (
        "mixed",
        0,
        "single_short_padded",
        0,
        3,
    ) in DIAG.COMPARISONS
    assert (
        "duplicate_short_padded",
        0,
        "single_short_padded",
        0,
        3,
    ) in DIAG.COMPARISONS
    assert (
        "duplicate_short_padded",
        1,
        "single_short_padded",
        0,
        3,
    ) in DIAG.COMPARISONS
    assert (
        "mixed",
        0,
        "duplicate_short_padded",
        0,
        3,
    ) in DIAG.COMPARISONS


def test_independent_ce_matches_public_scorer_on_natural_width_and_excludes_pad():
    scorer = _ready_scorer()
    geometry = DIAG.require_natural_geometry(scorer)
    assert geometry["mkt_width"] == 5
    assert geometry["aa20_width"] == 22
    packed = DIAG.pack_batch(scorer, ["MKT"], width=5, forced_width=False)
    assert packed.n_targets == (3,)
    observation = DIAG.score_packed(scorer, packed)
    public = float(scorer.log_likelihood(["MKT"])[0])
    assert observation["rows"][0]["n_residue_targets"] == 3
    assert observation["rows"][0]["total_log_likelihood"] == pytest.approx(public)
    padded = DIAG.pack_batch(scorer, ["MKT"], width=22, forced_width=True)
    assert padded.n_targets == (3,)
    assert padded.mask[0, 5:].tolist() == [0] * 17
    assert padded.labels[0, 5:].tolist() == [-100] * 17
    padded_obs = DIAG.score_packed(scorer, padded)
    assert padded_obs["n_residue_targets_batch"] == 3
    assert padded_obs["rows"][0]["n_residue_targets"] == 3
    assert padded_obs["rows"][0]["total_log_likelihood"] != pytest.approx(public)


def test_duplicate_rows_are_compared_separately_not_averaged(tmp_path: Path):
    scorer = _ready_scorer()
    payload = DIAG.run_diagnostic(
        out=tmp_path, device="cpu", scorer=scorer, release_scorer=False
    )
    assert payload["status"] == "complete"
    contrasts = payload["phases"]["bf16"]["comparisons"]
    row0 = [
        item
        for item in contrasts
        if item["left_case"] == "duplicate_short" and item["left_row"] == 0
    ]
    row1 = [
        item
        for item in contrasts
        if item["left_case"] == "duplicate_short" and item["left_row"] == 1
    ]
    assert len(row0) == DIAG.REPETITIONS
    assert len(row1) == DIAG.REPETITIONS
    for item in row0:
        assert item["n_residue_targets"] == 3
        assert item["abs_nats_per_residue_target"] == pytest.approx(0.0, abs=1e-6)
    for item in row1:
        assert item["n_residue_targets"] == 3
        assert item["abs_nats_per_residue_target"] > 1e-6
        assert item["abs_nats_per_residue_target"] != pytest.approx(row0[0]["abs_nats_per_residue_target"])
    mixed = [
        item
        for item in contrasts
        if item["left_case"] == "mixed" and item["repetition"] == 0
    ]
    assert mixed[0]["n_residue_targets"] == 3
    assert mixed[1]["n_residue_targets"] == 20
    padded_public = payload["phases"]["bf16"]["repetitions"][0]["cases"]["single_short_padded"]
    assert padded_public["public_scorer"]["called"] is False
    width22 = [
        (item["left_case"], item["left_row"], item["right_case"], item["right_row"])
        for item in contrasts
        if item["repetition"] == 0 and item["n_residue_targets"] == 3
    ]
    assert ("mixed", 0, "single_short_padded", 0) in width22
    assert ("duplicate_short_padded", 0, "single_short_padded", 0) in width22
    assert ("duplicate_short_padded", 1, "single_short_padded", 0) in width22
    assert ("mixed", 0, "duplicate_short_padded", 0) in width22


def test_bf16_to_fp32_keeps_loader_facts_and_does_not_reload(tmp_path: Path):
    scorer = _ready_scorer()
    facts_before = dict(scorer.facts)
    with torch.no_grad():
        scorer.loaded.model.weight.fill_(torch.tensor(1.25, dtype=torch.bfloat16))
    before = scorer.loaded.model.weight.detach().clone()
    expected = before.float().clone()
    assert before.dtype == torch.bfloat16
    assert not torch.equal(before, torch.ones_like(before))
    payload = DIAG.run_diagnostic(
        out=tmp_path, device="cpu", scorer=scorer, release_scorer=False
    )
    assert payload["status"] == "complete"
    assert payload["initial_bf16_facts"] == facts_before
    assert scorer.facts is scorer.loaded.facts
    assert scorer.facts["dtype_observed"] == ["bfloat16"]
    after = scorer.loaded.model.weight.detach()
    assert after.dtype == torch.float32
    assert torch.equal(after.cpu(), expected.cpu())
    assert payload["converted_parameter_buffer_dtypes"]["parameter_float_dtypes"] == [
        "float32"
    ]
    assert payload["phases"]["fp32_exec"]["setup"]["reloaded"] is False
    assert payload["phases"]["fp32_exec"]["setup"]["facts_rewritten"] is False
    assert payload["phases"]["fp32_exec"]["setup"]["kind"] == (
        "bf16_rounded_weights_executed_in_fp32"
    )
    assert "BF16-rounded weights executed in FP32" in payload["fp32_exec_note"]


def test_payload_has_no_pass_or_admission_semantics(tmp_path: Path):
    payload = DIAG.run_diagnostic(
        out=tmp_path, device="cpu", scorer=_ready_scorer(), release_scorer=False
    )
    dumped = json.loads((tmp_path / DIAG.OUTPUT_NAME).read_text(encoding="utf-8"))
    assert dumped["kind"] == DIAG.KIND
    assert dumped["status"] == "complete"
    assert dumped["no_dms_scores"] is True
    assert dumped["no_interface_admission"] is True
    assert dumped.get("probe_passed") is None
    assert "acquired" not in dumped
    assert "retrieval_bounded" not in dumped
    assert payload["arm"] == "galactica-30b"
    assert DIAG.__file__ is not None
    assert dumped["code_sources"]["diagnose_galactica_numerics"] == str(
        Path(DIAG.__file__).resolve()
    )
    assert dumped["code_sources"]["galactica_fitness"] == str(Path(G.__file__).resolve())


def test_illegal_width_missing_loss_and_nonfinite_fail(tmp_path: Path):
    scorer = _ready_scorer()
    with pytest.raises(ValueError, match="illegal pad width"):
        DIAG.pack_batch(scorer, ["MKT"], width=4, forced_width=True)

    missing = DIAG.run_diagnostic(
        out=tmp_path / "missing",
        device="cpu",
        scorer=_ready_scorer(model_cls=_NoLossLogits),
        release_scorer=False,
    )
    assert missing["status"] == "failed"
    assert missing["failure"]["phase"] == "bf16"
    assert "no loss" in missing["failure"]["message"]
    assert missing["no_interface_admission"] is True

    nonfinite = DIAG.run_diagnostic(
        out=tmp_path / "nonfinite",
        device="cpu",
        scorer=_ready_scorer(model_cls=_NanLogits),
        release_scorer=False,
    )
    assert nonfinite["status"] == "failed"
    assert nonfinite["failure"]["phase"] == "bf16"
    assert "non-finite" in nonfinite["failure"]["message"]


def test_second_phase_failure_keeps_completed_fp32_cases(tmp_path: Path):
    payload = DIAG.run_diagnostic(
        out=tmp_path,
        device="cpu",
        scorer=_ready_scorer(drop_fp32_label_after=3),
        release_scorer=False,
    )
    assert payload["status"] == "failed"
    assert payload["failure"]["phase"] == "fp32_exec"
    assert payload["failure"]["in_progress"] == {"repetition": 0, "case": "mixed"}
    assert payload["phases"]["bf16"]["incomplete"] is False
    fp32 = payload["phases"]["fp32_exec"]
    assert fp32["incomplete"] is True
    assert list(fp32["repetitions"][0]["cases"]) == ["single_short", "single_long"]
    assert "mixed" not in fp32["repetitions"][0]["cases"]
    assert "comparisons" not in fp32
    assert payload["no_dms_scores"] is True
    assert payload["no_interface_admission"] is True


def test_conversion_failure_restores_nondefault_matmul_policy(tmp_path: Path):
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        scorer = _ready_scorer()

        def boom(*_args: object, **_kwargs: object):
            raise RuntimeError("forced conversion failure")

        scorer.loaded.model.float = boom  # type: ignore[method-assign]
        payload = DIAG.run_diagnostic(
            out=tmp_path, device="cpu", scorer=scorer, release_scorer=False
        )
        assert payload["status"] == "failed"
        assert payload["failure"]["phase"] == "fp32_exec"
        assert "forced conversion failure" in payload["failure"]["message"]
        assert torch.get_float32_matmul_precision() == "high"
        assert payload["phases"]["bf16"]["incomplete"] is False
    finally:
        torch.set_float32_matmul_precision(previous)


def test_cleanup_exception_still_writes_failed_payload(tmp_path: Path):
    scorer = _ready_scorer()

    def boom() -> None:
        raise RuntimeError("release boom")

    scorer.release = boom  # type: ignore[method-assign]
    payload = DIAG.run_diagnostic(
        out=tmp_path, device="cpu", scorer=scorer, release_scorer=True
    )
    assert payload["status"] == "failed"
    assert payload["cleanup"][0]["where"] == "release"
    assert "release boom" in payload["cleanup"][0]["message"]
    dumped = json.loads((tmp_path / DIAG.OUTPUT_NAME).read_text(encoding="utf-8"))
    assert dumped["status"] == "failed"
    assert dumped["phases"]["bf16"]["incomplete"] is False


@pytest.mark.parametrize("failure_point", ["cuda_sync", "dtypes", "resources"])
def test_required_phase_evidence_failure_is_not_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
):
    def boom(*_args, **_kwargs):
        raise RuntimeError(f"synthetic {failure_point} failure")

    if failure_point == "cuda_sync":
        monkeypatch.setattr(DIAG, "_on_cuda", lambda *_args: True)
        monkeypatch.setattr(torch.cuda, "synchronize", boom)
    elif failure_point == "dtypes":
        monkeypatch.setattr(DIAG, "_observed_float_dtypes", boom)
    else:
        monkeypatch.setattr(DIAG, "_resource_fields", boom)
    payload = DIAG.run_diagnostic(
        out=tmp_path, device="cpu", scorer=_ready_scorer(), release_scorer=False
    )
    assert payload["status"] == "failed"
    assert payload["failure"]["phase"] == "bf16"
    assert f"synthetic {failure_point} failure" in payload["failure"]["message"]
    phase = payload["phases"]["bf16"]
    assert phase["incomplete"] is True
    assert "resources" not in phase
    if failure_point == "resources":
        assert len(phase["repetitions"]) == DIAG.REPETITIONS
        assert all(len(block["cases"]) == len(DIAG.CASES) for block in phase["repetitions"])
    else:
        assert phase["repetitions"] == []
    dumped = json.loads((tmp_path / DIAG.OUTPUT_NAME).read_text(encoding="utf-8"))
    assert dumped["status"] == "failed"
    assert dumped["no_interface_admission"] is True


def test_cli_accepts_out_and_device_and_refuses_other_arms(tmp_path: Path):
    args = DIAG._build_parser().parse_args(
        ["--out", str(tmp_path), "--device", "cpu"]
    )
    assert args.arm == "galactica-30b"
    assert args.device == "cpu"
    with pytest.raises(SystemExit):
        DIAG._build_parser().parse_args(
            ["--out", str(tmp_path), "--lookup", "ignored.json"]
        )
    payload = DIAG.run_diagnostic(
        out=tmp_path / "wrong-arm",
        device="cpu",
        arm="galactica-1.3b",
        scorer=_ready_scorer(),
        release_scorer=False,
    )
    assert payload["status"] == "failed"
    assert payload["failure"]["phase"] == "load"
    assert "galactica-30b" in payload["failure"]["message"]
    with pytest.raises(SystemExit) as exited:
        DIAG.main(
            ["--out", str(tmp_path / "cli-fail"), "--device", "cpu", "--arm", "galactica-125m"]
        )
    assert exited.value.code == 1
    written = json.loads(
        (tmp_path / "cli-fail" / DIAG.OUTPUT_NAME).read_text(encoding="utf-8")
    )
    assert written["status"] == "failed"
    assert written["no_interface_admission"] is True
