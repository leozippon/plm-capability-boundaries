"""E3's within-record token-shuffle control: what it must preserve, and when.

The control exists because neither eligibility criterion has ever been exercised
near its boundary -- the panel jumps from -4.08 nats on ``dialogpt-small`` to
+1.06 on ``progen2-base`` -- and its whole justification is one invariance: the
permutation must leave each record's *target* multiset exactly as it found it,
so that the unigram baseline every context-information figure is measured
against does not move at all. That invariance is proved here rather than
asserted, on both masking rules and with padding present, together with the
three properties that make a control artefact readable: the permutation is
reproducible from its seed alone, it is off unless it is asked for, and a run
under it cannot be mistaken for or overwrite a measurement.

The last test is the one that says the control does anything: a real GPT-2 on
real prose, scored twice on one cohort, whose clean cross-entropy rises towards
the unigram baseline while the baseline itself does not move by a bit.
"""

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.arms import (  # noqa: E402
    PANEL,
    Arm,
    Cohort,
    load_arm,
    target_shuffle_for,
    tokenize_batch,
)
from src.capability.core.budget import arm_power  # noqa: E402
from src.capability.core.scoring import (  # noqa: E402
    TOKEN_SHUFFLE_CONTROL,
    TargetTokenShuffle,
    sequence_target_mask,
)

SEED = 20260820
START_ID, END_ID = 90, 91


def _stage():
    """``cohort_power.py``, whose numeric name keeps it off the import path."""

    path = REPO_ROOT / "scripts/capability/stages/cohort_power.py"
    spec = importlib.util.spec_from_file_location("_stage_01", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plain_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Two ragged records, right-padded, in a plain rendering."""

    ids = torch.tensor(
        [
            [5, 6, 7, 8, 9, 10, 11, 12, 0, 0],
            [3, 4, 5, 4, 3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    return ids, mask


def _conditioned_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Two EC-conditioned renderings: tag, ``<sep>``, ``<start>``, content, ``<end>``."""

    ids = torch.tensor(
        [
            [11, 12, 13, 14, START_ID, 20, 21, 22, 23, 24, 25, END_ID, 0],
            [15, 16, 17, 18, START_ID, 30, 31, 32, 33, 34, 35, END_ID, 36],
        ],
        dtype=torch.long,
    )
    mask = torch.ones_like(ids)
    mask[0, -1] = 0
    return ids, mask


def _targets(ids: torch.Tensor, mask: torch.Tensor, shuffle: TargetTokenShuffle) -> list[list[int]]:
    """The scored target ids of each record, in scored order."""

    keep = sequence_target_mask(
        ids,
        mask,
        rule=shuffle.rule,
        start_token_id=shuffle.start_token_id,
        end_token_id=shuffle.end_token_id,
    )
    return [ids[row, 1:][keep[row]].tolist() for row in range(ids.shape[0])]


# ------------------------------------------------------- the load-bearing one


@pytest.mark.parametrize("conditioned", [False, True])
def test_the_per_record_target_multiset_is_exactly_invariant(conditioned: bool) -> None:
    """The whole justification for the control, on both masking rules.

    ``H_baseline`` is a unigram cross-entropy over exactly these tokens, so a
    permutation that moved one of them in or out of the scored span would move
    the baseline as well and the control would no longer isolate ``H_model``.
    Exact multiset equality is therefore the requirement, not approximate.
    """

    ids, mask = _conditioned_batch() if conditioned else _plain_batch()
    shuffle = TargetTokenShuffle(
        seed=SEED,
        rule="between_boundaries" if conditioned else "all_valid",
        start_token_id=START_ID if conditioned else None,
        end_token_id=END_ID if conditioned else None,
    )

    shuffled = shuffle.apply(ids, mask)
    before = _targets(ids, mask, shuffle)
    after = _targets(shuffled, mask, shuffle)

    assert [sorted(row) for row in before] == [sorted(row) for row in after]
    # And it is a shuffle rather than a no-op: at least one record's targets
    # come back in a different order.
    assert before != after


def test_padding_is_not_touched_and_the_scored_count_does_not_move() -> None:
    """A permutation that reached a pad position would change what is scored."""

    ids, mask = _plain_batch()
    shuffle = TargetTokenShuffle(seed=SEED, rule="all_valid")
    shuffled = shuffle.apply(ids, mask)

    pad = ~mask.bool()
    assert torch.equal(shuffled[pad], ids[pad])
    keep_before = sequence_target_mask(ids, mask, rule="all_valid")
    keep_after = sequence_target_mask(shuffled, mask, rule="all_valid")
    assert torch.equal(keep_before, keep_after)
    # The first token of a plain rendering is context and never a target, so it
    # is not in the permuted set either.
    assert torch.equal(shuffled[:, 0], ids[:, 0])


def test_a_declared_rendering_marker_is_never_a_scored_target() -> None:
    """A marker the rendering added is content for no rule, and a one-token
    prefix is already excluded by the position it occupies.

    Only the first token of a batch is context and never a target, so a
    rendering whose prefix is a single marker needs no help. A rendering that
    prefixes two does: its second token is a target, and it is the position the
    model predicts most confidently and a pooled unigram baseline prices as rare,
    which is context information the sequence does not supply.
    """

    ids = torch.tensor([[70, 71, 8, 9, 0, 0], [70, 71, 8, 9, 10, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.long)
    plain = sequence_target_mask(ids, mask, rule="all_valid")
    # Id 70 is the first token, so it is already outside the span.
    assert torch.equal(
        plain, sequence_target_mask(ids, mask, rule="all_valid", marker_token_ids=(70,))
    )
    # Id 71 is not, and excluding it leaves exactly the residues.
    excluded = sequence_target_mask(ids, mask, rule="all_valid", marker_token_ids=(70, 71))
    assert int(excluded.sum()) == int(plain.sum()) - ids.shape[0]
    assert not bool(excluded[:, 0].any())
    assert [ids[row, 1:][excluded[row]].tolist() for row in range(2)] == [
        [8, 9],
        [8, 9, 10],
    ]


# ------------------------------------------------------------- reproducibility


def test_one_seed_reproduces_the_permutation_and_another_seed_changes_it() -> None:
    ids, mask = _plain_batch()
    first = TargetTokenShuffle(seed=SEED, rule="all_valid").apply(ids, mask)
    again = TargetTokenShuffle(seed=SEED, rule="all_valid").apply(ids, mask)
    other = TargetTokenShuffle(seed=SEED + 1, rule="all_valid").apply(ids, mask)

    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def test_the_permutation_does_not_depend_on_batching_or_padding_width() -> None:
    """Keyed by the record's own unpadded ids, so a re-run at a different batch
    size reproduces the artefact rather than a neighbouring one."""

    ids, mask = _plain_batch()
    shuffle = TargetTokenShuffle(seed=SEED, rule="all_valid")
    batched = shuffle.apply(ids, mask)

    for row in range(ids.shape[0]):
        length = int(mask[row].sum())
        alone = shuffle.apply(ids[row : row + 1, :length], mask[row : row + 1, :length])
        assert torch.equal(alone[0], batched[row, :length])


# ------------------------------------------------------------ the EC boundary


def test_the_conditioned_span_is_unchanged_and_nothing_crosses_a_boundary() -> None:
    """ZymCTRL's EC prefix stays in context, in place, and out of the shuffle.

    The permuted set is the span strictly between ``<start>`` and ``<end>``,
    which is exactly the set ``between_boundaries`` scores. So the prompt, both
    markers and anything after ``<end>`` are identical afterwards, the same
    positions are scored, and no content token can leave the span.
    """

    ids, mask = _conditioned_batch()
    shuffle = TargetTokenShuffle(
        seed=SEED,
        rule="between_boundaries",
        start_token_id=START_ID,
        end_token_id=END_ID,
    )
    shuffled = shuffle.apply(ids, mask)

    start = int((ids[0] == START_ID).nonzero()[0])
    end = int((ids[0] == END_ID).nonzero()[0])
    # Prompt, <sep>, <start>, <end> and the tail beyond it, all untouched.
    assert torch.equal(shuffled[:, : start + 1], ids[:, : start + 1])
    assert torch.equal(shuffled[:, end:], ids[:, end:])
    # The scored positions are the same positions.
    keep_before = sequence_target_mask(
        ids, mask, rule="between_boundaries", start_token_id=START_ID, end_token_id=END_ID
    )
    keep_after = sequence_target_mask(
        shuffled, mask, rule="between_boundaries", start_token_id=START_ID, end_token_id=END_ID
    )
    assert torch.equal(keep_before, keep_after)
    assert sorted(shuffled[0, start + 1 : end].tolist()) == sorted(ids[0, start + 1 : end].tolist())
    assert shuffled[0, start + 1 : end].tolist() != ids[0, start + 1 : end].tolist()


def test_boundary_ids_under_the_plain_rule_are_refused() -> None:
    """The rule and the ids travel together or the mask is not the scored one."""

    ids, mask = _plain_batch()
    mismatched = TargetTokenShuffle(
        seed=SEED, rule="all_valid", start_token_id=START_ID, end_token_id=END_ID
    )
    with pytest.raises(ValueError, match="all_valid"):
        mismatched.apply(ids, mask)


# --------------------------------------------------------------- off by default


class _CharTokenizer:
    """One id per character; ``<start>``/``<end>`` resolve to the boundary ids."""

    pad_token_id = 0
    unk_token_id = 99

    def __call__(self, text: str, return_tensors=None) -> dict[str, list[int]]:
        ids: list[int] = []
        index = 0
        while index < len(text):
            if text.startswith("<start>", index):
                ids.append(START_ID)
                index += len("<start>")
            elif text.startswith("<end>", index):
                ids.append(END_ID)
                index += len("<end>")
            else:
                ids.append(ord(text[index]) % 64 + 1)
                index += 1
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<start>": START_ID, "<end>": END_ID}.get(token, self.unk_token_id)


def _tokenizer_only_arm(name: str) -> Arm:
    """An arm with no model: every path below stops at ``tokenize_batch``."""

    return Arm(
        spec=PANEL[name],
        model=None,
        tokenizer=_CharTokenizer(),
        device="cpu",
        dtype="float32",
    )


def test_an_arm_carries_no_shuffle_unless_one_is_attached() -> None:
    arm = _tokenizer_only_arm("gpt2")
    assert arm.target_token_shuffle is None

    texts = ["the quick brown fox jumps over the lazy dog", "a shorter record"]
    plain_ids, plain_mask = tokenize_batch(arm, texts, 64)
    control = replace(arm, target_token_shuffle=target_shuffle_for(arm, seed=SEED))
    shuffled_ids, shuffled_mask = tokenize_batch(control, texts, 64)

    assert torch.equal(plain_mask, shuffled_mask)
    assert not torch.equal(plain_ids, shuffled_ids)
    assert [sorted(row[1:]) for row in plain_ids.tolist()] == [
        sorted(row[1:]) for row in shuffled_ids.tolist()
    ]


def test_the_shuffle_runs_after_truncation_so_the_window_still_decides() -> None:
    """Permuting before truncation would change *which* tokens survive
    ``max_len``, which is the one thing the control may not do."""

    arm = _tokenizer_only_arm("gpt2")
    text = "abcdefghijklmnopqrstuvwxyz"
    plain_ids, _ = tokenize_batch(arm, [text], 10)
    control = replace(arm, target_token_shuffle=target_shuffle_for(arm, seed=SEED))
    shuffled_ids, _ = tokenize_batch(control, [text], 10)

    assert shuffled_ids.shape == plain_ids.shape
    assert sorted(shuffled_ids[0].tolist()) == sorted(plain_ids[0].tolist())


def test_an_ec_conditioned_arm_gets_the_boundary_rule_from_its_input_format() -> None:
    arm = _tokenizer_only_arm("zymctrl")
    shuffle = target_shuffle_for(arm, seed=SEED)

    assert shuffle.rule == "between_boundaries"
    assert (shuffle.start_token_id, shuffle.end_token_id) == (START_ID, END_ID)
    assert shuffle.record()["control"] == TOKEN_SHUFFLE_CONTROL


# ------------------------------------------------- the artefact identifies itself


def test_a_control_run_writes_a_different_artefact_under_a_different_name() -> None:
    """The cohort digest cannot separate the two runs: the shuffle happens in
    token space, after the cohort is frozen, so both runs share a cohort byte for
    byte and a control writing under the measurement's name would overwrite it."""

    stage = _stage()
    measurement = stage.artifact_names("swissprot", "0123456789abcdef", None)
    control = stage.artifact_names("swissprot", "0123456789abcdef", SEED)

    assert measurement[0] == "cohort_power_report"
    assert measurement[1] == "power_swissprot_0123456789ab"
    assert control[0] != measurement[0]
    assert control[1] != measurement[1]
    assert TOKEN_SHUFFLE_CONTROL in control[1] and str(SEED) in control[1]
    assert stage.negative_control_record(None) is None
    block = stage.negative_control_record(SEED)
    assert block["control"] == TOKEN_SHUFFLE_CONTROL and block["seed"] == SEED
    assert "not a measurement" in block["this_is_not_a_measurement"]


def test_a_control_run_refuses_to_carry_a_truncation_curve() -> None:
    """The curve is tokenised separately from the scored pass, so under the
    control it would describe the unshuffled rendering."""

    stage = _stage()
    args = Namespace(
        token_shuffle_control_seed=SEED,
        skip_truncation=False,
        truncation_contexts=[8, 64],
        max_len=384,
    )
    with pytest.raises(ValueError, match="--skip-truncation"):
        stage.validate_negative_control(args)

    stage.validate_negative_control(replace_namespace(args, skip_truncation=True))
    stage.validate_negative_control(replace_namespace(args, token_shuffle_control_seed=None))


def replace_namespace(args: Namespace, **changes) -> Namespace:
    return Namespace(**{**vars(args), **changes})


ZYMCTRL = PANEL["zymctrl"].path


@pytest.mark.skipif(not ZYMCTRL.is_dir(), reason="the ZymCTRL checkpoint is host-local")
def test_the_real_ec_rendering_keeps_its_prompt_and_permutes_only_its_residues() -> None:
    """The same boundary contract, on ZymCTRL's own tokenizer and rendering.

    Tokenizer only: nothing here needs a forward pass, and what the fixture above
    cannot check is that ``<start>`` and ``<end>`` really do resolve to single
    ids in this vocabulary and that the EC digits really are ordinary tokens
    inside the rendered string.
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ZYMCTRL), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    arm = Arm(
        spec=PANEL["zymctrl"],
        model=None,
        tokenizer=tokenizer,
        device="cpu",
        dtype="float32",
    )
    cohort = Cohort(
        "ec",
        "protein",
        ["MKTAYIAKQRQISFVKSHFSRQ", "MVLSPADKTNVKAAWGKVGAHAGEYG"],
        20,
        64,
        {"ec_labels": ["1.1.1.1", "2.7.1.1"]},
    )
    texts = cohort.input_strings(arm)
    shuffle = target_shuffle_for(arm, seed=SEED)
    plain, mask = tokenize_batch(arm, texts, 64)
    shuffled, shuffled_mask = tokenize_batch(
        replace(arm, target_token_shuffle=shuffle), texts, 64
    )

    assert shuffle.rule == "between_boundaries"
    assert torch.equal(mask, shuffled_mask)
    before = _targets(plain, mask, shuffle)
    after = _targets(shuffled, shuffled_mask, shuffle)
    assert [sorted(row) for row in before] == [sorted(row) for row in after]
    assert before != after
    for row in range(plain.shape[0]):
        start = int((plain[row] == shuffle.start_token_id).nonzero()[0])
        end = int((plain[row] == shuffle.end_token_id).nonzero()[0])
        # The EC tag, <sep> and <start> before the span; <end> and the padding
        # after it. Only the residues between them move.
        assert torch.equal(plain[row, : start + 1], shuffled[row, : start + 1])
        assert torch.equal(plain[row, end:], shuffled[row, end:])
        assert tokenizer.decode(shuffled[row, : start + 1].tolist()).startswith(
            texts[row].split("<sep>")[0][0]
        )


# ------------------------------------------------------ end to end, real model

GPT2 = PANEL["gpt2"].path

PROSE = [
    "The river rose steadily through the night, and by morning the fields on "
    "either side of the road had turned into a shallow lake. Farmers who had "
    "lived in the valley for forty years said they had never seen the water "
    "climb so quickly. The bridge at the edge of town held, but the approach "
    "road on the northern side was washed away, and the county engineer said "
    "it would be at least a week before traffic could cross again. Several "
    "families spent the night in the school hall, where volunteers had set out "
    "camp beds and made soup in the kitchen. By the middle of the afternoon "
    "the water had begun to fall again, leaving a thick layer of brown silt "
    "across the gardens and the lower rooms of a dozen houses.",
    "The library was built in eighteen ninety two on a plot of land given to "
    "the town by a retired shipping merchant who had learned to read late in "
    "life. Its reading room has a high ceiling of pale plaster and tall "
    "windows that face the square, and on a bright morning the light falls "
    "across the long oak tables in wide bars. For most of the last century the "
    "collection grew slowly, a few hundred volumes a year, until a bequest in "
    "the nineteen sixties paid for a new wing and a proper catalogue. The "
    "building was closed for two years for repairs to the roof, and when it "
    "reopened the shelves had been rearranged so that the local history "
    "collection sat beside the windows rather than in the basement.",
    "Most of the birds that pass through the estuary in autumn are on their "
    "way from the breeding grounds in the far north to the wintering grounds "
    "along the coast of west Africa. They arrive in waves, a few thousand at a "
    "time, and feed on the mudflats at low tide before moving on. The warden "
    "counts them from a hide at the eastern end of the marsh, walking out "
    "along the sea wall before dawn and returning when the tide has covered "
    "the last of the sand. Numbers have fallen over the past twenty years, "
    "though not evenly: some species are steady or slightly up, while others "
    "have declined by more than half, and nobody is certain how much of the "
    "change happens here rather than at the other end of the journey.",
    "The recipe is simple enough that it survives being written down badly. "
    "Warm the milk until it is just below a simmer, then take it off the heat "
    "and let it stand while you beat the eggs and the sugar together in a "
    "bowl. Pour the milk into the eggs slowly, stirring the whole time, and "
    "then return the mixture to the pan over a low flame. It will thicken "
    "after a few minutes, and the moment to stop is when it coats the back of "
    "a spoon and holds a clean line when you draw a finger through it. If it "
    "goes too far the eggs will scramble and the texture is lost, so the "
    "safest course is to take it off early and let the heat of the pan finish "
    "the work.",
]


@pytest.mark.skipif(not GPT2.is_dir(), reason="the gpt2 checkpoint is host-local")
def test_a_real_model_loses_its_context_and_its_baseline_does_not_move() -> None:
    """GPT-2 on real prose, scored twice on one cohort.

    The measurement and the control differ in exactly one input: the order of
    the scored tokens. So the unigram baseline and the scored-token count must be
    identical to the bit, and the clean cross-entropy must rise towards the
    baseline as the context the model was reading is taken away. The per-symbol
    expansion is checked loosely rather than exactly, because it is the one
    figure a permutation can move without moving the multiset: a byte-level BPE
    can split a multi-byte character across two tokens, and reordering them makes
    the decoder emit replacement characters. It is exact on the ASCII prose
    below and moved by 0.3% on real OpenWebText.

    What this does *not* show is that the shuffled arm has zero context
    information: shuffled prose is off GPT-2's training distribution, and the
    cross-entropy it pays there is not the cross-entropy of its own marginal.
    The direction is the claim; the level is a measurement of this control, and
    it depends on the baseline it is read against -- against the plug-in unigram
    of this four-record cohort the control reads about -2.2 nats/token, and
    against a held-out unigram on 300 OpenWebText records it reads -0.29.
    """

    arm = load_arm("gpt2", device="cpu", dtype="float32")
    cohort = Cohort("prose", "text", list(PROSE), 400, 0)
    control = replace(arm, target_token_shuffle=target_shuffle_for(arm, seed=SEED))
    common = {"max_len": 128, "batch_size": 2, "unigram_estimator": "plugin"}

    measured = arm_power(arm, cohort, **common)
    shuffled = arm_power(control, cohort, **common)

    assert shuffled["n_scored_tokens"] == measured["n_scored_tokens"]
    assert shuffled["n_distinct_scored_tokens"] == measured["n_distinct_scored_tokens"]
    assert (
        shuffled["unigram_entropy_on_cohort_nats"] == measured["unigram_entropy_on_cohort_nats"]
    )
    assert shuffled["symbols_per_token"] == pytest.approx(
        measured["symbols_per_token"], rel=5e-3
    )
    assert shuffled["clean_ce_nats"] > measured["clean_ce_nats"] + 1.0
    assert shuffled["context_information_nats"] < measured["context_information_nats"] - 1.0
    print(
        "\nunigram(plug-in) {baseline:.4f}  clean_ce {measured:.4f} -> {shuffled:.4f}  "
        "context_info {info_m:+.4f} -> {info_s:+.4f} nats/token".format(
            baseline=measured["unigram_entropy_on_cohort_nats"],
            measured=measured["clean_ce_nats"],
            shuffled=shuffled["clean_ce_nats"],
            info_m=measured["context_information_nats"],
            info_s=shuffled["context_information_nats"],
        )
    )
