"""What has to hold before a Potts-guided decoder may drive ProGen3.

Four things could make a ProGen3 arm look like a valid E20 cell while being one of
something else, and each has its own test here.

**The alphabet.** ProGen3's residue tokens are the twenty-six single uppercase
letters at ids 8..33 -- plain alphabetical order, including the six non-canonical
ones -- which is neither of the two orderings this repository uses. A guided arm
indexed in the wrong order is a permuted bias on every position, and no error
would surface. The decoder derives each candidate's residue code from the token's
*decoded text*, so the ordering never enters; that immunity is asserted rather
than assumed, together with the exclusion of the non-canonical letters, the
specials and the hundred span tokens.

**The layout.** ``prepare_clm`` renders ``<bos> 1 r(0) .. r(L-1) 2 <eos>`` and
generation is conditioned on the first two tokens, so Potts column ``t`` is decode
step ``t`` with no shift. The adapter derives both the prompt and the rendering
from the checkpoint's own preparer and refuses a disagreement; the arithmetic is
pinned here against the preparer directly.

**The call.** ``ProGen3ForCausalLM.forward`` requires ``position_ids`` and
``sequence_ids``, and a wrong ``sequence_ids`` is the withdrawn Stage-46
measurement rather than an exception. A recording stub checks that the adapter
continues positions from the cache and holds sequence ids at the native zero.

**Zero guidance.** The decisive one: at ``lambda = 0`` the guided loop must draw
what native sampling draws from the same seed, token for token. It is checked
against :func:`native_reference_draw`, a second implementation that shares the
declared sampling recipe and none of the layout, candidate-table or bias
machinery.

Everything runs against the real ProGen3 tokenizer and the real batch preparer --
which ship with the third-party package, not with the checkpoint -- and a
deterministic stub carrying ProGen3's forward signature. Nothing here loads a
checkpoint, so the tests skip only where the patched third-party tree is absent.
The real-weights counterpart of the decisive gate is a measurement rather than a
test: on ProGen3-3B, one L20, bfloat16, batch size one, 48 sequences at L=37, 52
and 72 were token-for-token identical between guidance at zero and
:func:`native_reference_draw`, and all 48 moved under a unit dose.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.context.family_potts import (  # noqa: E402
    FamilyModel, load_family_model, save_family_model)
from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.core.arms import Arm, arm_spec  # noqa: E402
from src.capability.decoding.guided_decoding import (  # noqa: E402
    CellIdentity, GuidanceStrength, GuidedDecoder, build_arm_bias, generate_cell,
    generation_point, native_layout, native_reference_draw, token_span_table)
from src.capability.models.progen3 import PROGEN3_SOURCE, ProGen3  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (PROGEN3_SOURCE / "progen3").is_dir(),
    reason="the patched third-party progen3 package is not present",
)

ARM = "progen3-3b"
LENGTH = 11
LOGIT_WIDTH = 134


# --------------------------------------------------------------------- fixtures


class _StubModel:
    """ProGen3's forward signature, with logits that are a function of the history.

    Deterministic in the committed token history alone, so the guided loop and the
    native reference see the same distribution at the same prefix and any
    difference between them is the thing under test. It records every call, so the
    adapter's ``position_ids`` and ``sequence_ids`` can be read back.
    """

    def __init__(self, width: int = LOGIT_WIDTH, seed: int = 11):
        self.width = int(width)
        self.seed = int(seed)
        self.config = SimpleNamespace(
            vocab_size=self.width, bos_token_id=1, eos_token_id=2, pad_token_id=0,
            max_position_embeddings=65536)
        self.calls: list[dict[str, list[list[int]]]] = []

    def _row(self, history: list[int]) -> torch.Tensor:
        payload = f"{self.seed}|" + ",".join(str(value) for value in history)
        digest = hashlib.sha256(payload.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        return torch.as_tensor(rng.normal(size=self.width), dtype=torch.float32)

    def __call__(self, input_ids=None, position_ids=None, sequence_ids=None,
                 past_key_values=None, use_cache=None, return_dict=True):
        self.calls.append({
            "input_ids": input_ids.tolist(),
            "position_ids": None if position_ids is None else position_ids.tolist(),
            "sequence_ids": None if sequence_ids is None else sequence_ids.tolist(),
        })
        rows = input_ids.tolist()
        histories = [list(row) for row in (past_key_values or [[] for _ in rows])]
        logits = torch.zeros((len(rows), len(rows[0]), self.width), dtype=torch.float32)
        for index, row in enumerate(rows):
            for step, token in enumerate(row):
                histories[index].append(int(token))
                logits[index, step] = self._row(histories[index])
        return SimpleNamespace(logits=logits, past_key_values=_StubCache(histories))


class _StubCache(list):
    """The committed histories, with the one accessor the adapter reads off a cache."""

    def get_seq_length(self, layer: int = 0) -> int:
        return len(self[0])


def _handle() -> ProGen3:
    if str(PROGEN3_SOURCE) not in sys.path:
        sys.path.insert(0, str(PROGEN3_SOURCE))
    from progen3.batch_preparer import ProGen3BatchPreparer  # noqa: PLC0415

    model = _StubModel()
    return ProGen3(model=model, config=model.config, preparer=ProGen3BatchPreparer(),
                   device=torch.device("cpu"), checkpoint=Path("/nonexistent"))


@pytest.fixture()
def arm():
    from src.capability.decoding.progen3_decoding import decoding_arm_from_handle

    return decoding_arm_from_handle(ARM, _handle(), dtype="bfloat16")


@pytest.fixture()
def family(tmp_path) -> FamilyModel:
    rng = np.random.default_rng(3)
    save_family_model(
        tmp_path / "refs", "SYN", wildtype="".join(rng.choice(list(AA20), size=LENGTH)),
        h=rng.normal(size=(LENGTH, 20)),
        pairs=rng.normal(size=(LENGTH * (LENGTH - 1) // 2, 20, 20)),
        profile_freq=rng.dirichlet(np.ones(20), size=LENGTH),
        profile_column_weight=rng.uniform(5, 50, size=LENGTH),
        qualified=True, qualification_failures=[], extra={})
    return load_family_model("SYN", tmp_path / "refs")


def _decoder(arm, family, guidance_arm: str, strength: GuidanceStrength) -> GuidedDecoder:
    layout = native_layout(arm, family.length)
    return GuidedDecoder(
        bias=build_arm_bias(guidance_arm, family), layout=layout,
        table=token_span_table(arm, layout, logit_width=LOGIT_WIDTH), strength=strength)


# ------------------------------------------------------------------ the alphabet


def test_the_candidate_table_is_the_twenty_canonical_residues_in_AA20_codes(arm, family):
    """Residue codes come from decoded text, so ProGen3's A..Z ids cannot permute the bias.

    The tokenizer's canonical ids are alphabetical with gaps (A=8, C=10, ... Y=32),
    which is neither ``ACDEFGHIKLMNPQRSTVWY`` nor ``ARNDCQEGHILKMFPSTWYV``. What the
    bias is indexed by is the AA20 position of the token's decoded character, so
    the two orderings never meet.
    """

    layout = native_layout(arm, family.length)
    table = token_span_table(arm, layout, logit_width=LOGIT_WIDTH)
    vocabulary = arm.tokenizer.get_vocab()
    assert len(vocabulary) == LOGIT_WIDTH
    assert table.size == 20
    assert table.texts == tuple(sorted(AA20))
    assert int(table.character_length.max()) == 1 and int(table.residue_length.max()) == 1
    # code <-> residue, taken through the table rather than through any ordering
    assert {text: int(code) for text, code in zip(table.texts, table.residues[:, 0])} == {
        residue: AA20.index(residue) for residue in AA20}
    # every id the table admits is that residue's own id in ProGen3's vocabulary
    assert [int(value) for value in table.ids] == [vocabulary[text] for text in table.texts]


def test_non_canonical_letters_specials_and_span_tokens_are_unsamplable(arm, family):
    """B, J, O, U, X, Z, the six specials and the hundred span tokens are outside the table."""

    layout = native_layout(arm, family.length)
    table = token_span_table(arm, layout, logit_width=LOGIT_WIDTH)
    admitted = set(int(value) for value in table.ids)
    vocabulary = arm.tokenizer.get_vocab()
    for name in ("B", "J", "O", "U", "X", "Z", "<pad>", "<bos>", "<eos>", "<mask>", "1", "2",
                 "<bos_glm>", "<eos_span>", "<span_0>", "<span_99>"):
        assert vocabulary[name] not in admitted, name
    assert len(admitted) == 20


# -------------------------------------------------------------------- the layout


def test_the_layout_is_bare_residue_slots_behind_the_two_token_prompt(arm, family):
    """No literal, one slot per residue, and the prompt is the released conditioning row."""

    layout = native_layout(arm, family.length)
    assert layout.prompt == "1"
    assert layout.literals == {}
    assert layout.characters == family.length
    assert layout.expected_text(family.wildtype) == family.wildtype
    assert [int(value) for value in layout.residues_before] == list(range(family.length + 1))
    point = generation_point(arm)
    assert point.add_special_tokens is True
    ids = arm.tokenizer(point.prompt, add_special_tokens=True, return_tensors=None)["input_ids"]
    assert ids == arm.serving_provenance["generation_prompt_ids"]


def test_potts_column_t_is_the_token_at_index_t_plus_two(arm):
    """The preparer's own arithmetic: two leading markers, two trailing, no shift.

    Taken against ``prepare_clm`` rather than against the adapter, so this is the
    checkpoint's statement about where a residue sits and not the adapter's.
    """

    handle = arm.serving_provenance["progen3"]
    sequence = "MKWVTFIS"
    prepared = handle.preparer.prepare_clm(sequence, False)
    tokens = [handle.tokenizer.id_to_token(int(value))
              for value in prepared["input_ids"].tolist()]
    assert tokens[:2] == ["<bos>", "1"]
    assert tokens[-2:] == ["2", "<eos>"]
    assert tokens[2:-2] == list(sequence)
    for index, residue in enumerate(sequence):
        assert tokens[index + 2] == residue
    # generation drops the two trailing markers, so the prompt is exactly <bos> 1
    assert prepared["metadata"]["prefix_length"] == len(tokens) - 2
    assert len(arm.serving_provenance["generation_prompt_ids"]) == 2


def test_a_rendering_that_stopped_describing_progen3_is_refused():
    """The ``n_to_c_control`` spec view is checked against the preparer, not trusted."""

    from src.capability.decoding import progen3_decoding as pd

    handle = _handle()
    arm = pd.decoding_arm_from_handle(ARM, handle, dtype="bfloat16")
    assert pd.require_rendering_agreement(arm, handle) == "1" + pd.RENDERING_PROBE
    drifted = replace(arm, spec=replace(arm.spec, input_format="raw"))
    with pytest.raises(RuntimeError, match="not be this checkpoint's"):
        pd.require_rendering_agreement(drifted, handle)


# ---------------------------------------------------------------------- the call


def test_the_adapter_supplies_continued_positions_and_native_sequence_ids(arm, family):
    """ProGen3's two required arguments, read back off the recording stub."""

    decoder = _decoder(arm, family, "native", GuidanceStrength(0.0, 0.0))
    identity = CellIdentity(ARM, "SYN", "native", decoder.strength, 0, 0, 500)
    generate_cell(arm=arm, decoder=decoder, identity=identity, count=1, batch_size=1,
                  score_fn=lambda sequence: -float(len(sequence)))
    calls = arm.model._pg.model.calls
    # the two-token prompt, then one call per committed residue bar the last: the
    # loop stops as soon as the layout is full rather than forwarding the final token
    assert len(calls) == family.length
    assert calls[0]["input_ids"] == [arm.serving_provenance["generation_prompt_ids"]]
    assert calls[0]["position_ids"] == [[0, 1]]
    for step, call in enumerate(calls[1:], start=2):
        assert call["position_ids"] == [[step]], step
    for call in calls:
        row = call["sequence_ids"][0]
        assert set(row) == {0} and len(row) == len(call["input_ids"][0])


def test_a_batched_forward_is_refused_rather_than_recorded_as_the_same_cell(arm, family):
    """ProGen3's logits move with the batch's extent, so only a singleton cell is the cell.

    Measured on the 3B: sampling the identical cell at batch four rather than batch
    one changed 18 of 48 sequences from the same seeds. The guided decoder declares
    that a draw does not depend on which batch it was taken in, so the arm must
    refuse the wider batch instead of producing a pool that is reproducible only
    together with its schedule.
    """

    decoder = _decoder(arm, family, "native", GuidanceStrength(0.0, 0.0))
    identity = CellIdentity(ARM, "SYN", "native", decoder.strength, 0, 0, 600)
    with pytest.raises(ValueError, match="refuses a 2-row forward"):
        generate_cell(arm=arm, decoder=decoder, identity=identity, count=2, batch_size=2,
                      score_fn=lambda sequence: -float(len(sequence)))


def test_the_tokenizer_shim_refuses_a_tensor_request(arm):
    """The decoder asks for plain id lists; a tensor request is a caller using the wrong surface."""

    with pytest.raises(ValueError, match="plain id lists"):
        arm.tokenizer("A", add_special_tokens=False, return_tensors="pt")


def test_a_non_progen3_arm_still_reads_the_campaign_admission_table():
    """``generation_point`` defaults to ADMITTED, so no other arm's declaration moves."""

    from src.capability.generation.unconditional_generation import require_admitted

    other = Arm(spec=arm_spec("progen2-medium"), model=None, tokenizer=None, device="cpu",
                dtype="float32")
    assert generation_point(other) == require_admitted("progen2-medium")
    with pytest.raises(ValueError, match="already complete"):
        require_admitted(ARM)


# ------------------------------------------------------------- the decisive gate


@pytest.mark.parametrize("seed", [700, 701, 702, 703])
def test_zero_guidance_reproduces_native_sampling_token_for_token(arm, family, seed):
    """The gate: an independent native sampler draws the guided-at-zero sequence exactly."""

    reference = native_reference_draw(arm=arm, length=family.length, seed=seed, device="cpu")
    decoder = _decoder(arm, family, "native", GuidanceStrength(0.0, 0.0))
    identity = CellIdentity(ARM, "SYN", "native", decoder.strength, 0, 0, seed)
    records = generate_cell(arm=arm, decoder=decoder, identity=identity, count=1, batch_size=1,
                            score_fn=lambda sequence: -float(len(sequence)))
    assert records[0]["seed"] == seed
    assert records[0]["token_ids"] == reference["token_ids"]
    assert records[0]["sequence"] == reference["sequence"]
    assert len(records[0]["sequence"]) == family.length


def test_every_arm_at_zero_strength_draws_the_native_reference_sequence(arm, family):
    """Not only ``native``: a guided arm at zero dose adds a literal zero, so all agree."""

    seed = 800
    reference = native_reference_draw(arm=arm, length=family.length, seed=seed, device="cpu")
    for guidance_arm in ("native", "potts_fields", "coupling", "full", "profile"):
        decoder = _decoder(arm, family, guidance_arm, GuidanceStrength(0.0, 0.0))
        identity = CellIdentity(ARM, "SYN", guidance_arm, decoder.strength, 0, 0, seed)
        records = generate_cell(arm=arm, decoder=decoder, identity=identity, count=1,
                                batch_size=1, score_fn=lambda sequence: -float(len(sequence)))
        assert records[0]["sequence"] == reference["sequence"], guidance_arm


def test_a_positive_dose_moves_the_draw_away_from_the_native_reference(arm, family):
    """The gate would be vacuous if the bias could not change a draw at all."""

    seed = 900
    reference = native_reference_draw(arm=arm, length=family.length, seed=seed, device="cpu")
    decoder = _decoder(arm, family, "full", GuidanceStrength(2.0, 2.0))
    identity = CellIdentity(ARM, "SYN", "full", decoder.strength, 0, 0, seed)
    records = generate_cell(arm=arm, decoder=decoder, identity=identity, count=1, batch_size=1,
                            score_fn=lambda sequence: -float(len(sequence)))
    assert records[0]["sequence"] != reference["sequence"]


def test_the_native_reference_refuses_a_multi_residue_tokeniser():
    """ProtGPT2's stream needs the layout rule, so the reference must not pretend otherwise."""

    class _BpeTokenizer:
        def get_vocab(self):
            return {"<|endoftext|>": 0, **{residue: 1 + index
                                           for index, residue in enumerate(AA20)}, "ACD": 21}

        def batch_decode(self, rows, skip_special_tokens=False,
                         clean_up_tokenization_spaces=False):
            names = {value: key for key, value in self.get_vocab().items()}
            return [names[int(row[0])] for row in rows]

    bpe = Arm(spec=arm_spec("protgpt2"), model=None, tokenizer=_BpeTokenizer(), device="cpu",
              dtype="float32")
    with pytest.raises(ValueError, match="multi-residue tokens"):
        native_reference_draw(arm=bpe, length=4, seed=1, device="cpu")
