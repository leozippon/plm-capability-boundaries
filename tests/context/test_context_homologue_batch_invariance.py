"""What Stage 46 may batch, and what a batched row must equal.

The withdrawn ProGen3 Stage-46 scores came from a batched forward that changed
the model's input with batch position, and the stage's only numerical check was
a singleton self-check, which a batch-position defect cannot reach. The
condition these tests hold is the one that was missing: **the target NLL Stage 46
reports for a unit must be the NLL that unit has on its own**, whatever else was
scored beside it. Where a packing cannot deliver that, the score path must refuse
the batch rather than report a batch-dependent number.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.context import context_homologue as ch  # noqa: E402
from src.capability.models import progen3 as P3  # noqa: E402
from src.capability.core.arms import Arm, ArmSpec  # noqa: E402

#: The stage's own declared equality: two scores of one row must agree to this
#: many nats per scored token. Zero, because a batched and a singleton forward of
#: the same row on a batch-invariant model are the same arithmetic.
EXACT = 0.0

PROGEN3_ROWS = [[7, 11, 12, 13, 8, 9], [7, 14, 15, 8, 9], [7, 16, 17, 18, 19, 8, 9]]


def _load_stage():
    path = REPO_ROOT / "scripts" / "capability" / "stages/context_homologue.py"
    spec = importlib.util.spec_from_file_location("stage46_batch_invariance", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _arm(name: str, *, tokenizer=None, serving_provenance=None) -> Arm:
    spec = ArmSpec(
        name=name,
        path=Path("."),
        path_variable="MODEL_ROOT",
        modality="protein",
        n_layer=1,
        d_model=1,
        tokenisation="residue",
        input_format="raw",
        evaluation_cohort_source="swissprot",
        architecture="test",
    )
    return Arm(
        spec=spec,
        model=None,
        tokenizer=tokenizer,
        device="cpu",
        dtype="none",
        serving_provenance=serving_provenance,
    )


class _ProGen3Tokenizer:
    """Only the special-id map ``_progen3_special_ids`` reads."""

    def get_vocab(self) -> dict[str, int]:
        vocab = {"<bos>": 7, "1": 8, "2": 9, "<eos>": 10, "<pad>": 0}
        for index, residue in enumerate("ACDEFGHIKLMNPQRSTVWY"):
            vocab[residue] = 11 + index
        return vocab


def _progen3_batches(monkeypatch, rows):
    """The kwargs ``_forward_progen3`` hands the model for ``rows``."""

    stage = _load_stage()
    received: list[dict[str, torch.Tensor]] = []

    def forward(pg, batch):
        received.append(batch)
        return SimpleNamespace(logits=batch["input_ids"].float().unsqueeze(-1))

    monkeypatch.setattr(P3, "forward", forward)
    arm = _arm(
        "progen3-3b",
        tokenizer=_ProGen3Tokenizer(),
        serving_provenance={"progen3": object()},
    )
    stage._forward_progen3(arm, rows)
    return received[0]


# --------------------------------------------- the native sequence-id contract


def test_progen3_sequence_ids_are_the_native_zero_at_every_batch_position(monkeypatch):
    """The defect, stated as the condition it broke.

    ``sequence_ids`` is a learned-embedding index added to every token
    embedding, so the batch-row index that this path used to assign changed the
    model's input for every row after the first. Zero is the native value, for
    content and padding alike, at every batch position and every batch size.
    """

    for rows in (PROGEN3_ROWS, list(reversed(PROGEN3_ROWS)), PROGEN3_ROWS[:1]):
        batch = _progen3_batches(monkeypatch, rows)
        assert torch.equal(
            batch["sequence_ids"],
            torch.zeros((len(rows), max(len(row) for row in rows)), dtype=torch.long),
        )


def test_progen3_zero_sequence_ids_are_the_vendored_preparers_own_value():
    """Not a hardcoded expectation: the native preparer's value, read from it.

    Skipped where the third-party tree is not on this host; the contract is
    still the preparer's, not this file's.
    """

    source = P3.PROGEN3_SOURCE
    if not (source / "progen3").is_dir():
        pytest.skip(f"no vendored progen3 package under {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from progen3.batch_preparer import ProGen3BatchPreparer

    preparer = ProGen3BatchPreparer()
    encodings = [preparer.prepare_clm(record, False) for record in ("MKV", "MKVLAA")]
    for encoding in encodings:
        assert torch.equal(
            encoding["sequence_ids"], torch.zeros(len(encoding["input_ids"]))
        )
    padded = preparer.pad_encodings(encodings)
    assert torch.equal(padded["sequence_ids"], torch.zeros_like(padded["sequence_ids"]))


def test_progen3_row_kwargs_do_not_depend_on_batch_position_or_composition(monkeypatch):
    """A row's model input is a function of that row, and of nothing else."""

    alone = _progen3_batches(monkeypatch, [PROGEN3_ROWS[1]])
    beside_others = _progen3_batches(monkeypatch, PROGEN3_ROWS)
    reordered = _progen3_batches(monkeypatch, list(reversed(PROGEN3_ROWS)))
    width = len(PROGEN3_ROWS[1])
    for key in ("input_ids", "position_ids", "sequence_ids"):
        assert torch.equal(beside_others[key][1, :width], alone[key][0])
        assert torch.equal(reordered[key][1, :width], alone[key][0])
    pad = _ProGen3Tokenizer().get_vocab()["<pad>"]
    assert torch.equal(
        beside_others["input_ids"][1, width:],
        torch.full((beside_others["input_ids"].shape[1] - width,), pad, dtype=torch.long),
    )
    assert int(beside_others["position_ids"][1, width:].sum()) == 0


# ------------------------------------------------- what the score path may batch


def test_score_refuses_a_progen3_batch_before_any_weight_is_loaded(tmp_path):
    """The declaration that replaces the withdrawn batched production.

    It fires before ``load_scorable_arm``, so an operator learns this from the
    flag rather than from a loaded 3B model's first forward.
    """

    stage = _load_stage()
    args = argparse.Namespace(
        arm="progen3-3b",
        condition=ch.HOMOLOGUE,
        batch_size=8,
        cohort=tmp_path / "absent-cohort.json",
        plan=tmp_path / "absent-plan.json",
        device="cpu",
        dtype="bfloat16",
    )
    with pytest.raises(SystemExit, match="--batch-size 1"):
        stage.run_score(args)
    assert ch.PACKING_PROGEN3 in stage.SINGLETON_ONLY_PACKINGS
    args.batch_size = 1
    # Batch size one passes the declaration and stops at the absent cohort,
    # which is what proves the refusal is the batch size and not the arm.
    with pytest.raises((FileNotFoundError, OSError)):
        stage.run_score(args)


def test_non_singleton_packings_are_not_refused(tmp_path):
    stage = _load_stage()
    for arm in ("progen2-base", "galactica-1.3b", "prollama"):
        assert ch.packing_of(arm) not in stage.SINGLETON_ONLY_PACKINGS


# --------------------------------------- batched equals singleton where batched


class _AdditiveModel:
    """A batch-invariant stand-in: each position's logits depend on its own id.

    ``batch_dependent`` makes it depend on the batch's width instead, which is
    the shape of failure the equality below exists to catch.
    """

    def __init__(self, vocab: int, *, batch_dependent: bool = False):
        self.vocab = vocab
        self.batch_dependent = batch_dependent
        self.config = SimpleNamespace(max_position_embeddings=1024)

    def __call__(self, input_ids, attention_mask=None):
        base = input_ids.float().unsqueeze(-1) * torch.arange(
            1, self.vocab + 1, dtype=torch.float32
        )
        if self.batch_dependent:
            # Scaled, not shifted: a shift is invisible through log-softmax, so a
            # test built on one would pass against a genuinely batch-dependent
            # model. This is the padded width the neighbours in the batch set.
            base = base * (1.0 + 0.1 * float(input_ids.shape[1]))
        return SimpleNamespace(logits=base)


class _EosTokenizer:
    def __init__(self, eos: int):
        self.eos_token_id = eos
        self.pad_token_id = None


def _padded_path_gaps(stage, *, batch_dependent: bool) -> list[float]:
    """Max |batched - singleton| target NLL, in nats per scored token."""

    vocab = 24
    rows = [[3, 5, 7, 9, 11, 13], [4, 6, 8], [2, 12, 14, 16, 18]]
    arm = _arm("progen2-base", tokenizer=_EosTokenizer(vocab - 1))
    arm = Arm(
        spec=arm.spec,
        model=_AdditiveModel(vocab, batch_dependent=batch_dependent),
        tokenizer=arm.tokenizer,
        device="cpu",
        dtype="none",
        serving_provenance=None,
    )
    assert ch.packing_of(arm.name) == ch.PACKING_F16
    batched_logits, batched_ids = stage._forward_rows(arm, rows)
    gaps = []
    for position, row in enumerate(rows):
        batched = stage._target_nll(
            batched_logits[position : position + 1],
            batched_ids[position : position + 1],
            1,
            len(row),
        )
        single_logits, single_ids = stage._forward_rows(arm, [row])
        single = stage._target_nll(single_logits, single_ids, 1, len(row))
        assert batched["scored_tokens"] == single["scored_tokens"]
        gaps.append(abs(batched["nats_per_token"] - single["nats_per_token"]))
    return gaps


def test_a_batched_row_scores_what_it_scores_alone_on_the_padded_path():
    """The equality every batched Stage-46 packing owes the reader.

    Padding and a shorter neighbour must not move a row's own NLL. This is the
    check the ProGen3 path fails and therefore no longer takes.
    """

    stage = _load_stage()
    assert max(_padded_path_gaps(stage, batch_dependent=False)) <= EXACT


def test_the_equality_detects_a_batch_dependent_model():
    """Teeth: the same comparison on a model whose output moves with the batch."""

    stage = _load_stage()
    assert max(_padded_path_gaps(stage, batch_dependent=True)) > EXACT


# ------------------------------------------------------- with the real weights


@pytest.mark.skipif(
    not (P3.PROGEN3_CHECKPOINT / "config.json").is_file()
    or not (P3.PROGEN3_SOURCE / "progen3").is_dir()
    or not torch.cuda.is_available(),
    reason="needs the released ProGen3-112M checkpoint, the vendored package and a GPU",
)
def test_progen3_singleton_scores_do_not_depend_on_what_was_scored_before():
    """On the real MoE, the stage's singleton score is a property of the row.

    Scored one row at a time, as ``SINGLETON_ONLY_PACKINGS`` requires, the same
    row returns the same number whatever was scored before it. The batched
    counterpart of this equality is what that declaration refuses; the measured
    size of its violation is in ``archive/docs/R5/progen3-stage46-remeasurement.md``.
    """

    stage = _load_stage()
    arm = stage.load_scorable_arm("progen3-112m", device="cuda:0", dtype="bfloat16")
    records = ["MKVLAAGIVGL" * 6, "TTQSPAILSVS" * 9]
    rows = [ch._progen3_item_ids(arm, record) for record in records]
    spans = [ch.target_span(arm, row, record=record) for row, record in zip(rows, records)]

    def score(index: int) -> float:
        logits, ids = stage._forward_rows(arm, [rows[index]])
        start, end = spans[index]
        return stage._target_nll(logits, ids, start, end)["nats_per_token"]

    first = score(0)
    score(1)
    assert abs(score(0) - first) <= EXACT
