"""Conditions the coupling-complement guided decoder must always satisfy.

These are the experiment's guarantees, not coverage. The decisive one is that at
``lambda = 0`` every arm draws the identical sequence from one seed, because that
is what makes the arms differ in the bias term and in nothing else.

Seed convention adopted here, and by the decoder: each sequence owns a
``torch.Generator`` seeded ``stream_seed + sequence_index`` and one
``multinomial`` call is made per row per step, so a sequence is a function of its
seed and the step distributions alone. It does **not** follow the
``torch.manual_seed(seed + batch_index)`` convention of the HF-generate paths,
whose draws move with batch size. Every comparison below nevertheless holds the
batch size fixed, so the invariants do not rest on that independence.

Everything runs on CPU against a synthetic family reference written through the
real exporter and a tiny tokenizer fixture, so the token-span arithmetic is
checked against hand-written loops over the model's own ``h`` and ``J`` rather
than against the implementation.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.context.family_potts import (  # noqa: E402
    FamilyModel, encode, load_family_model, potts_score, save_family_model)
from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.core.arms import Arm, arm_spec  # noqa: E402
from src.capability.decoding.guided_decoding import (  # noqa: E402
    ARM_CODES, FACTORIAL_ARMS, GUIDANCE_ARMS, ArmBias, CellIdentity, EmptyCandidateSet,
    GuidanceStrength, GuidedDecoder, bias_vector, build_arm_bias, component_bias_vectors,
    component_scale, generate_cell, native_layout, nominal_lambda, permutation_for,
    select_cross_domain_donor, sequence_bias_sum, sequence_component_sums, token_span_table)

PERMUTATION_SEED = 20261013
SHORT = 7
WRAPPED = 63  # longer than the 60-column FASTA wrap, so the layout carries a newline literal


# --------------------------------------------------------------------- fixtures


def _synthetic_model(directory: Path, domain: str, length: int, *, seed: int,
                     qualified: bool = True) -> FamilyModel:
    rng = np.random.default_rng(seed)
    wildtype = "".join(rng.choice(list(AA20), size=length))
    save_family_model(
        directory, domain, wildtype=wildtype, h=rng.normal(size=(length, 20)),
        pairs=rng.normal(size=(length * (length - 1) // 2, 20, 20)),
        profile_freq=rng.dirichlet(np.ones(20), size=length),
        profile_column_weight=rng.uniform(5, 50, size=length),
        qualified=qualified, qualification_failures=[] if qualified else ["not_converged"], extra={})
    return load_family_model(domain, directory, allow_unqualified=not qualified)


class _StubTokenizer:
    """A tiny tokenizer whose vocabulary is spelled out, with greedy longest-match encoding."""

    def __init__(self, tokens: list[str], eos_token: str):
        if eos_token not in tokens:
            raise ValueError("the fixture vocabulary must declare its end-of-text token")
        self._tokens = list(tokens)
        self._ids = {token: index for index, token in enumerate(self._tokens)}
        self.eos_token = eos_token
        self.eos_token_id = self._ids[eos_token]

    def get_vocab(self) -> dict[str, int]:
        return dict(self._ids)

    def batch_decode(self, rows, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return [self.decode(row) for row in rows]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self._tokens[int(value)] for value in ids)

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        ids, position = [], 0
        order = sorted(self._tokens, key=len, reverse=True)
        while position < len(text):
            for token in order:
                if text.startswith(token, position):
                    ids.append(self._ids[token])
                    position += len(token)
                    break
            else:
                raise ValueError(f"fixture tokenizer cannot encode {text[position:]!r}")
        return {"input_ids": ids}


class _StubModel:
    """Deterministic logits from the full token history, so arms see identical distributions."""

    def __init__(self, width: int, seed: int = 7):
        self.width = int(width)
        self.seed = int(seed)
        self.config = SimpleNamespace(vocab_size=self.width, n_positions=2048)

    def _logits(self, history: list[int]) -> np.ndarray:
        payload = f"{self.seed}|" + ",".join(str(value) for value in history)
        digest = hashlib.sha256(payload.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        return rng.normal(size=self.width)

    def __call__(self, input_ids=None, past_key_values=None, use_cache=True):
        rows = input_ids.tolist()
        histories = [list(row) for row in (past_key_values or [[] for _ in rows])]
        logits = torch.zeros((len(rows), len(rows[0]), self.width), dtype=torch.float32)
        for index, row in enumerate(rows):
            for step, token in enumerate(row):
                histories[index].append(int(token))
                logits[index, step] = torch.as_tensor(
                    self._logits(histories[index]), dtype=torch.float32)
        return SimpleNamespace(logits=logits, past_key_values=histories)


def _multi_residue_arm() -> Arm:
    """ProtGPT2's declared interface: FASTA wrapped at 60, end-of-text prompt, BPE spans."""

    tokens = [
        "<|endoftext|>", "\n", "\nAC", " ", "X",
        *AA20,
        "AC", "DEF", "MK", "WYV", "GG",
    ]
    tokenizer = _StubTokenizer(tokens, "<|endoftext|>")
    return Arm(spec=arm_spec("protgpt2"), model=_StubModel(len(tokens)), tokenizer=tokenizer,
               device="cpu", dtype="float32")


def _single_residue_arm() -> Arm:
    """ProGen2's declared interface: a direction marker then one token per residue."""

    tokens = ["<|pad|>", "<|bos|>", "<|eos|>", "1", "2", "X", *AA20]
    tokenizer = _StubTokenizer(tokens, "<|eos|>")
    return Arm(spec=arm_spec("progen2-medium"), model=_StubModel(len(tokens)), tokenizer=tokenizer,
               device="cpu", dtype="float32")


@pytest.fixture()
def short_model(tmp_path):
    return _synthetic_model(tmp_path / "refs", "SYN", SHORT, seed=0)


@pytest.fixture()
def wrapped_model(tmp_path):
    return _synthetic_model(tmp_path / "wrapped", "WRAP", WRAPPED, seed=1)


def _strength(arm_name: str, dose: float) -> GuidanceStrength:
    """The dose on whichever components the arm carries; zero on the ones it does not."""

    components = {"native": (0.0, 0.0), "potts_fields": (dose, 0.0), "profile": (dose, 0.0),
                  "full": (dose, dose)}
    return GuidanceStrength(*components.get(arm_name, (0.0, dose)))


def _decoder(arm: Arm, model: FamilyModel, arm_name: str, dose: float, **kwargs) -> GuidedDecoder:
    layout = native_layout(arm, model.length)
    table = token_span_table(arm, layout, logit_width=arm.model.width)
    bias = build_arm_bias(arm_name, model, **kwargs)
    return GuidedDecoder(bias=bias, layout=layout, table=table,
                         strength=_strength(arm_name, dose))


def _identity(model_arm: str, domain: str, arm_name: str, dose: float, index: int,
              stream_index: int, stream_seed: int) -> CellIdentity:
    return CellIdentity(model_arm, domain, arm_name, _strength(arm_name, dose), index,
                        stream_index, stream_seed)


def _score_fn(sequence: str) -> float:
    """Stands in for the caller's native full-sum scorer; the module holds no scorer of its own."""

    return -float(len(sequence))


# ------------------------------------------------------- the decomposition itself


def test_the_factorial_is_exact_inside_one_models_decomposition(short_model):
    """Additivity of the four corners: F - M == C - N and F - C == M - N.

    This establishes only that the arms are built from shared additive components; the
    identities hold for any arrays and say nothing about gauge. What "coupling" and "field"
    mean is conditional on the declared gauge, which the gauge test below documents.
    """

    native, fields, coupling, full = (build_arm_bias(arm, short_model) for arm in FACTORIAL_ARMS)
    assert FACTORIAL_ARMS == ("native", "potts_fields", "coupling", "full")
    for position in range(short_model.length):
        prefix = encode(short_model.wildtype[:position])
        n = bias_vector(native, position, prefix)
        m = bias_vector(fields, position, prefix)
        c = bias_vector(coupling, position, prefix)
        f = bias_vector(full, position, prefix)
        np.testing.assert_allclose(m, short_model.h[position], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(f, m + c, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(f - m, c - n, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(f - c, m - n, rtol=0.0, atol=1e-12)
        parts = component_bias_vectors(full, position, prefix)
        np.testing.assert_allclose(parts[0], m, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(parts[1], c, rtol=0.0, atol=0.0)


def test_the_profile_arm_is_the_frozen_log_profile_and_sits_outside_the_factorial(short_model):
    profile = build_arm_bias("profile", short_model)
    assert "profile" not in FACTORIAL_ARMS
    for position in range(short_model.length):
        prefix = encode(short_model.wildtype[:position])
        np.testing.assert_allclose(
            bias_vector(profile, position, prefix), short_model.log_profile[position],
            rtol=0.0, atol=0.0)
    assert np.isclose(np.exp(short_model.log_profile[0]).sum(), 1.0)


def test_coupling_bias_at_the_empty_prefix_is_exactly_zero(short_model):
    for arm in ("coupling", "permuted_coupling"):
        kwargs = {"permutation_seed": PERMUTATION_SEED} if arm == "permuted_coupling" else {}
        bias = build_arm_bias(arm, short_model, **kwargs)
        assert np.array_equal(bias_vector(bias, 0, []), np.zeros(20))


def test_native_bias_is_zero_at_every_position(short_model):
    native = build_arm_bias("native", short_model)
    for position in range(short_model.length):
        prefix = encode(short_model.wildtype[:position])
        assert np.array_equal(bias_vector(native, position, prefix), np.zeros(20))


def test_full_arm_guidance_score_is_the_potts_score(short_model):
    bias = build_arm_bias("full", short_model)
    assert sequence_bias_sum(bias, short_model.wildtype) == pytest.approx(
        potts_score(short_model, short_model.wildtype), abs=1e-9)


def test_a_permuted_alphabet_is_refused_rather_than_indexed(short_model):
    permuted = replace(short_model, alphabet="ARNDCQEGHILKMFPSTWYV")
    with pytest.raises(ValueError, match="alphabet"):
        build_arm_bias("coupling", permuted)


# ------------------------------------------------------------- the token spans


def _hand_token_bias(model: FamilyModel, component: str, prefix: str, residues: str) -> float:
    """The token-span bias written out with explicit loops over the model's own arrays."""

    prefix_codes = [AA20.index(residue) for residue in prefix]
    span = [AA20.index(residue) for residue in residues]
    start = len(prefix_codes)
    total = 0.0
    for offset, code in enumerate(span):
        position = start + offset
        if component in ("full", "potts_fields"):
            total += float(model.h[position, code])
        elif component == "profile":
            total += float(model.log_profile[position, code])
        for earlier, other in enumerate(prefix_codes):
            if component in ("coupling", "full"):
                total += float(model.J[position, earlier, code, other])
        for earlier in range(offset):
            if component in ("coupling", "full"):
                total += float(model.J[position, start + earlier, code, span[earlier]])
    return total


@pytest.mark.parametrize("component", ["profile", "coupling", "full"])
def test_multi_residue_token_bias_is_the_sum_of_its_residue_biases(short_model, component):
    arm = _multi_residue_arm()
    decoder = _decoder(arm, short_model, component, 1.0)
    prefix = "AC"
    prefix_codes = encode(prefix)
    total = decoder.token_bias(len(prefix), prefix_codes)
    row = int(np.flatnonzero(np.asarray(decoder.table.texts) == "DEF")[0])
    assert decoder.table.residue_length[row] == 3
    assert total[row] == pytest.approx(
        _hand_token_bias(short_model, component, prefix, "DEF"), abs=1e-9)


def test_a_token_carrying_a_newline_biases_only_its_residues(wrapped_model):
    """The FASTA linewrap consumes a character slot, no residue budget and no bias."""

    arm = _multi_residue_arm()
    decoder = _decoder(arm, wrapped_model, "coupling", 1.0)
    offset = int(np.flatnonzero(decoder.layout.template != -1)[0])
    assert offset == 60 and decoder.layout.literals[int(decoder.layout.template[offset])] == "\n"
    prefix = wrapped_model.wildtype[:60]
    total = decoder.token_bias(60, encode(prefix))
    newline = int(np.flatnonzero(np.asarray(decoder.table.texts) == "\n")[0])
    span = int(np.flatnonzero(np.asarray(decoder.table.texts) == "\nAC")[0])
    assert decoder.table.residue_length[newline] == 0
    assert decoder.table.residue_length[span] == 2
    assert total[newline] == pytest.approx(0.0, abs=0.0)
    assert total[span] == pytest.approx(
        _hand_token_bias(wrapped_model, "coupling", prefix, "AC"), abs=1e-9)
    admissible = decoder.admissible(offset)
    assert admissible[newline] and admissible[span]
    assert not admissible[int(np.flatnonzero(np.asarray(decoder.table.texts) == "AC")[0])]


def test_tokens_that_would_overshoot_the_residue_budget_are_suppressed(short_model):
    arm = _multi_residue_arm()
    decoder = _decoder(arm, short_model, "coupling", 1.0)
    admissible = decoder.admissible(SHORT - 2)
    texts = np.asarray(decoder.table.texts)
    assert admissible[int(np.flatnonzero(texts == "AC")[0])]
    assert not admissible[int(np.flatnonzero(texts == "DEF")[0])]
    # End-of-sequence, the padding token, a space and the non-canonical ``X`` never enter the
    # table at all, so no logit column behind them can ever be drawn. A layout with no literal
    # also keeps the FASTA linewrap out.
    assert set(texts) == set(AA20) | {"AC", "DEF", "MK", "WYV", "GG"}


# -------------------------------------------------------------- the decoding loop


@pytest.mark.parametrize("build_arm", [_single_residue_arm, _multi_residue_arm])
def test_lambda_zero_reproduces_native_sampling_for_every_arm(tmp_path, build_arm):
    """The central invariant: at lambda = 0 the six arms differ by a literal 0.0."""

    model = _synthetic_model(tmp_path / "refs", "SYN", SHORT, seed=0)
    donor = _synthetic_model(tmp_path / "refs", "DONOR", SHORT + 2, seed=3)
    pools = {}
    for arm_name in GUIDANCE_ARMS:
        kwargs = {}
        if arm_name == "permuted_coupling":
            kwargs["permutation_seed"] = PERMUTATION_SEED
        if arm_name == "cross_domain_coupling":
            kwargs["donor"] = donor
        arm = build_arm()
        decoder = _decoder(arm, model, arm_name, 0.0, **kwargs)
        assert decoder.strength.is_zero
        pools[arm_name] = generate_cell(
            arm=arm, decoder=decoder,
            identity=_identity("fixture", "SYN", arm_name, 0.0, 0, 0, 1000),
            count=6, batch_size=3, score_fn=_score_fn)
    reference = pools["native"]
    for arm_name, pool in pools.items():
        assert [row["sequence"] for row in pool] == [row["sequence"] for row in reference], arm_name
        assert [row["token_ids"] for row in pool] == [row["token_ids"] for row in reference], arm_name
        assert [row["step_entropy_nats"] for row in pool] == [
            row["step_entropy_nats"] for row in reference], arm_name


@pytest.mark.parametrize("build_arm", [_single_residue_arm, _multi_residue_arm])
def test_every_sequence_is_exactly_L_canonical_residues(short_model, build_arm):
    arm = build_arm()
    decoder = _decoder(arm, short_model, "coupling", 1.5)
    records = generate_cell(
        arm=arm, decoder=decoder,
        identity=_identity("fixture", "SYN", "coupling", 1.5, 1, 0, 2000),
        count=5, batch_size=2, score_fn=_score_fn)
    assert len(records) == 5
    for record in records:
        assert len(record["sequence"]) == SHORT == record["length"]
        assert set(record["sequence"]) <= set(AA20)
        assert record["token_steps"] == len(record["token_ids"]) == len(record["step_entropy_nats"])
        assert record["guidance_score"] == pytest.approx(
            sequence_bias_sum(decoder.bias, record["sequence"]), abs=1e-9)


def test_a_wrapped_layout_emits_exactly_L_residues_across_the_linewrap(wrapped_model):
    arm = _multi_residue_arm()
    decoder = _decoder(arm, wrapped_model, "full", 1.0)
    records = generate_cell(
        arm=arm, decoder=decoder,
        identity=_identity("fixture", "WRAP", "full", 1.0, 0, 0, 3000),
        count=2, batch_size=2, score_fn=_score_fn)
    for record in records:
        assert len(record["sequence"]) == WRAPPED
        assert set(record["sequence"]) <= set(AA20)
        assert arm.tokenizer.decode(record["token_ids"]).count("\n") == 1


def test_identical_seeds_repeat_and_distinct_streams_stay_disjoint(short_model):
    arm = _multi_residue_arm()
    decoder = _decoder(arm, short_model, "coupling", 2.0)

    def pool(stream_seed: int, stream_index: int):
        return generate_cell(
            arm=arm, decoder=decoder,
            identity=_identity("fixture", "SYN", "coupling", 2.0, 0, stream_index, stream_seed),
            count=4, batch_size=4, score_fn=_score_fn)

    first, repeat, other = pool(5000, 0), pool(5000, 0), pool(9000, 1)
    assert [row["sequence"] for row in first] == [row["sequence"] for row in repeat]
    assert {row["seed"] for row in first}.isdisjoint({row["seed"] for row in other})
    assert [row["sequence"] for row in first] != [row["sequence"] for row in other]


def test_batch_size_does_not_change_a_sequence(short_model):
    arm = _multi_residue_arm()
    decoder = _decoder(arm, short_model, "full", 1.0)
    identity = _identity("fixture", "SYN", "full", 1.0, 0, 0, 7000)
    one = generate_cell(arm=arm, decoder=decoder, identity=identity, count=4, batch_size=1,
                        score_fn=_score_fn)
    four = generate_cell(arm=arm, decoder=decoder, identity=identity, count=4, batch_size=4,
                         score_fn=_score_fn)
    assert [row["sequence"] for row in one] == [row["sequence"] for row in four]


# ------------------------------------------------------------ the two controls


def test_permuted_coupling_keeps_the_value_multiset_and_moves_the_assignment(short_model):
    coupling = build_arm_bias("coupling", short_model)
    permuted = build_arm_bias("permuted_coupling", short_model, permutation_seed=PERMUTATION_SEED)
    np.testing.assert_allclose(np.sort(permuted.C.ravel()), np.sort(coupling.C.ravel()),
                               rtol=0.0, atol=0.0)
    assert not np.array_equal(permuted.C, coupling.C)
    permutation = np.asarray(permuted.provenance["column_permutation"])
    assert sorted(permutation.tolist()) == list(range(short_model.length))
    assert not np.array_equal(permutation, np.arange(short_model.length))
    np.testing.assert_allclose(
        permuted.C, coupling.C[np.ix_(permutation, permutation)], rtol=0.0, atol=0.0)
    assert permutation_for("SYN", SHORT, PERMUTATION_SEED).tolist() == permutation.tolist()
    assert permutation_for("OTHER", SHORT, PERMUTATION_SEED).tolist() != permutation.tolist()


def test_the_donor_rule_takes_the_smallest_admissible_donor_then_the_lowest_id():
    lengths = {"TGT": 70, "AAA": 72, "BBB": 72, "CCC": 71, "DDD": 65}
    assert select_cross_domain_donor("TGT", lengths) == "CCC"
    assert select_cross_domain_donor("CCC", {"CCC": 71, "AAA": 72, "BBB": 72}) == "AAA"


def test_cross_domain_coupling_uses_the_donor_first_L_columns(tmp_path):
    target = _synthetic_model(tmp_path / "refs", "TGT", SHORT, seed=0)
    donor = _synthetic_model(tmp_path / "refs", "DON", SHORT + 3, seed=4)
    bias = build_arm_bias("cross_domain_coupling", target, donor=donor)
    np.testing.assert_allclose(bias.C, donor.J[:SHORT, :SHORT], rtol=0.0, atol=0.0)
    assert bias.provenance["donor_domain"] == "DON"
    assert bias.provenance["donor_columns_used"] == SHORT


# -------------------------------------------------------------- the refusals


def test_a_column_count_other_than_L_is_refused(short_model):
    arm = _multi_residue_arm()
    layout = native_layout(arm, SHORT + 1)
    table = token_span_table(arm, layout, logit_width=arm.model.width)
    with pytest.raises(ValueError, match="column-aligned"):
        GuidedDecoder(bias=build_arm_bias("coupling", short_model), layout=layout, table=table,
                      strength=GuidanceStrength(0.0, 1.0))


def test_an_unqualified_domain_is_refused(tmp_path):
    model = _synthetic_model(tmp_path / "refs", "BAD", SHORT, seed=2, qualified=False)
    with pytest.raises(ValueError, match="not a qualified"):
        build_arm_bias("coupling", model)
    good = _synthetic_model(tmp_path / "refs", "GOOD", SHORT, seed=0)
    with pytest.raises(ValueError, match="not a qualified"):
        build_arm_bias("cross_domain_coupling", good, donor=model)


def test_cross_domain_coupling_with_no_admissible_donor_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no admissible donor"):
        select_cross_domain_donor("TGT", {"TGT": 70, "AAA": 65, "BBB": 69})
    target = _synthetic_model(tmp_path / "refs", "TGT", SHORT, seed=0)
    short_donor = _synthetic_model(tmp_path / "refs", "DON", SHORT - 1, seed=5)
    with pytest.raises(ValueError, match="fewer than the target"):
        build_arm_bias("cross_domain_coupling", target, donor=short_donor)


def test_an_arm_is_refused_a_seed_or_donor_it_does_not_use(short_model):
    with pytest.raises(ValueError, match="takes no permutation seed"):
        build_arm_bias("coupling", short_model, permutation_seed=PERMUTATION_SEED)
    with pytest.raises(ValueError, match="requires a declared permutation seed"):
        build_arm_bias("permuted_coupling", short_model)
    with pytest.raises(ValueError, match="requires a donor"):
        build_arm_bias("cross_domain_coupling", short_model)


def test_an_empty_candidate_set_is_refused_rather_than_truncating_a_token(short_model):
    """A vocabulary with no single-residue token cannot finish an odd-length layout."""

    tokens = ["<|endoftext|>", "\n", "AC", "DE", "FG", "HI"]
    arm = Arm(spec=arm_spec("protgpt2"), model=_StubModel(len(tokens)),
              tokenizer=_StubTokenizer(tokens, "<|endoftext|>"), device="cpu", dtype="float32")
    decoder = _decoder(arm, short_model, "coupling", 1.0)
    with pytest.raises(EmptyCandidateSet, match="no vocabulary token matches"):
        generate_cell(arm=arm, decoder=decoder,
                      identity=_identity("fixture", "SYN", "coupling", 1.0, 0, 0, 11000),
                      count=1, batch_size=1, score_fn=_score_fn)


def test_an_unknown_arm_name_is_refused(short_model):
    with pytest.raises(ValueError, match="unknown guidance arm"):
        build_arm_bias("coupling_only", short_model)


def test_a_bias_whose_shape_does_not_match_the_length_is_refused():
    with pytest.raises(ValueError, match="expected"):
        ArmBias("coupling", "SYN", SHORT, np.zeros((SHORT + 1, 20)), None, {})


# ------------------------------------------------- the two independent strengths


def test_the_two_strengths_act_only_on_their_own_component(short_model):
    """One dose per component, and a component the arm lacks contributes exactly zero."""

    arm = _multi_residue_arm()
    layout = native_layout(arm, short_model.length)
    table = token_span_table(arm, layout, logit_width=arm.model.width)
    full = build_arm_bias("full", short_model)
    prefix = encode("AC")

    fields_only = GuidedDecoder(bias=full, layout=layout, table=table,
                                strength=GuidanceStrength(3.0, 0.0))
    coupling_only = GuidedDecoder(bias=full, layout=layout, table=table,
                                  strength=GuidanceStrength(0.0, 3.0))
    both = GuidedDecoder(bias=full, layout=layout, table=table,
                         strength=GuidanceStrength(3.0, 3.0))
    row = int(np.flatnonzero(np.asarray(table.texts) == "DEF")[0])

    expected_fields = _hand_token_bias(short_model, "potts_fields", "AC", "DEF")
    expected_coupling = _hand_token_bias(short_model, "coupling", "AC", "DEF")
    assert fields_only.token_bias(2, prefix)[row] == pytest.approx(3.0 * expected_fields, abs=1e-9)
    assert coupling_only.token_bias(2, prefix)[row] == pytest.approx(3.0 * expected_coupling, abs=1e-9)
    assert both.token_bias(2, prefix)[row] == pytest.approx(
        3.0 * (expected_fields + expected_coupling), abs=1e-9)
    # the inert component is exactly zero, not merely small
    assert np.array_equal(coupling_only.fields_terms(2) * 0.0,
                          np.zeros(table.size))
    for arm_name in ("potts_fields", "profile"):
        unguided_coupling = GuidedDecoder(
            bias=build_arm_bias(arm_name, short_model), layout=layout, table=table,
            strength=GuidanceStrength(3.0, 0.0)).coupling_terms(2, prefix)
        assert np.array_equal(unguided_coupling, np.zeros(table.size))
    native = build_arm_bias("native", short_model)
    zero = GuidedDecoder(bias=native, layout=layout, table=table,
                         strength=GuidanceStrength(0.0, 0.0))
    assert np.array_equal(zero.token_bias(2, prefix), np.zeros(table.size))


def test_a_dose_on_a_component_the_arm_lacks_is_refused(short_model):
    arm = _multi_residue_arm()
    layout = native_layout(arm, short_model.length)
    table = token_span_table(arm, layout, logit_width=arm.model.width)
    with pytest.raises(ValueError, match="carries no fields component"):
        GuidedDecoder(bias=build_arm_bias("coupling", short_model), layout=layout, table=table,
                      strength=GuidanceStrength(1.0, 1.0))
    for arm_name in ("potts_fields", "profile"):
        with pytest.raises(ValueError, match="carries no coupling component"):
            GuidedDecoder(bias=build_arm_bias(arm_name, short_model), layout=layout, table=table,
                          strength=GuidanceStrength(1.0, 1.0))
    with pytest.raises(ValueError, match="carries no"):
        GuidedDecoder(bias=build_arm_bias("native", short_model), layout=layout, table=table,
                      strength=GuidanceStrength(0.0, 1.0))


def test_a_negative_or_non_finite_strength_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        GuidanceStrength(-1.0, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        GuidanceStrength(0.0, float("inf"))


def test_the_nominal_lambda_rule(short_model):
    strength = GuidanceStrength(2.0, 5.0)
    assert nominal_lambda(build_arm_bias("full", short_model), strength) == 5.0
    for arm_name in ("potts_fields", "profile"):
        assert nominal_lambda(build_arm_bias(arm_name, short_model),
                              GuidanceStrength(2.0, 0.0)) == 2.0
    assert nominal_lambda(build_arm_bias("coupling", short_model),
                          GuidanceStrength(0.0, 5.0)) == 5.0
    assert nominal_lambda(build_arm_bias("native", short_model),
                          GuidanceStrength(0.0, 0.0)) == 0.0


def test_records_carry_both_strengths_and_both_component_scores(short_model):
    arm = _multi_residue_arm()
    decoder = _decoder(arm, short_model, "full", 1.5)
    records = generate_cell(
        arm=arm, decoder=decoder,
        identity=_identity("protgpt2", "SYN", "full", 1.5, 2, 1, 13000),
        count=2, batch_size=2, score_fn=_score_fn)
    for record in records:
        assert record["arm_code"] == ARM_CODES["full"] == "F"
        assert record["guidance_arm"] == "full"
        assert record["model"] == "protgpt2" and record["domain"] == "SYN"
        assert record["lambda_nominal"] == 1.5
        assert record["lambda_fields"] == 1.5 and record["lambda_coupling"] == 1.5
        assert record["attempt_budget"] == 1
        fields, coupling = sequence_component_sums(decoder.bias, record["sequence"])
        assert record["fields_score"] == pytest.approx(fields, abs=1e-9)
        assert record["coupling_score"] == pytest.approx(coupling, abs=1e-9)
        assert record["guidance_score"] == pytest.approx(fields + coupling, abs=1e-9)
        assert record["guidance_bias_applied"] == pytest.approx(1.5 * (fields + coupling), abs=1e-9)
        assert record["id"].endswith(f"seed{record['seed']}")


def test_the_native_arm_has_no_guidance_quantity(short_model):
    """None, not zero: the arm applies no external statistics at all."""

    arm = _single_residue_arm()
    decoder = _decoder(arm, short_model, "native", 0.0)
    records = generate_cell(
        arm=arm, decoder=decoder,
        identity=_identity("progen2-medium", "SYN", "native", 0.0, 0, 0, 15000),
        count=2, batch_size=2, score_fn=_score_fn)
    for record in records:
        assert record["guidance_score"] is None
        assert record["fields_score"] is None and record["coupling_score"] is None
        assert record["guidance_bias_applied"] == 0.0
        assert record["arm_code"] == "N" and record["lambda_nominal"] == 0.0


def test_a_fields_only_arm_reports_no_coupling_score(short_model):
    arm = _single_residue_arm()
    decoder = _decoder(arm, short_model, "profile", 1.0)
    record = generate_cell(
        arm=arm, decoder=decoder,
        identity=_identity("progen2-medium", "SYN", "profile", 1.0, 2, 0, 17000),
        count=1, batch_size=1, score_fn=_score_fn)[0]
    assert record["coupling_score"] is None
    assert record["fields_score"] == pytest.approx(record["guidance_score"], abs=0.0)


def test_the_arm_codes_are_the_declared_set_with_the_two_controls_distinct():
    assert ARM_CODES == {"native": "N", "potts_fields": "M", "coupling": "C", "full": "F",
                         "profile": "Pr", "permuted_coupling": "Cx",
                         "cross_domain_coupling": "Cd"}
    assert sorted(ARM_CODES) == sorted(GUIDANCE_ARMS)
    assert len(set(ARM_CODES.values())) == len(ARM_CODES)
    assert set(FACTORIAL_ARMS) < set(GUIDANCE_ARMS)


# ------------------------------------------------- the component-scale diagnostic


def test_component_scale_separates_the_three_components(short_model):
    """The asymmetry the two strengths exist for has to be readable from the artefact."""

    summary = component_scale(short_model, [short_model.wildtype])
    assert summary["domain"] == "SYN" and summary["alphabet"] == AA20
    components = summary["components"]
    assert sorted(components) == ["coupling", "potts_fields", "profile"]
    for name, block in components.items():
        assert block["steps"] == SHORT
        assert len(block["per_position_mean"]) == SHORT
        assert block["mean"] >= 0.0 and block["max"] >= block["median"]
        assert block["prefix_dependent"] == (name == "coupling")
    # the coupling vector is identically zero at position 0, so its spread is too
    assert components["coupling"]["per_position_mean"][0] == 0.0
    assert components["coupling"]["zero_steps"] >= 1
    assert components["potts_fields"]["zero_steps"] == 0


def test_component_scale_is_prefix_independent_for_the_two_fields_components(short_model):
    other = "".join(AA20[(index + 3) % 20] for index in range(SHORT))
    first = component_scale(short_model, [short_model.wildtype])
    second = component_scale(short_model, [other])
    for name in ("potts_fields", "profile"):
        assert first["components"][name]["mean"] == second["components"][name]["mean"]
        assert (first["components"][name]["per_position_mean"]
                == second["components"][name]["per_position_mean"])
    assert first["components"]["coupling"]["mean"] != second["components"]["coupling"]["mean"]


def test_component_scale_matches_a_hand_computed_spread(short_model):
    summary = component_scale(short_model, [short_model.wildtype])
    codes = encode(short_model.wildtype)
    expected = []
    for position in range(SHORT):
        vector = np.asarray([
            sum(float(short_model.J[position, earlier, residue, codes[earlier]])
                for earlier in range(position))
            for residue in range(20)])
        expected.append(float(vector.std()))
    np.testing.assert_allclose(
        summary["components"]["coupling"]["per_position_mean"], expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        summary["components"]["potts_fields"]["per_position_mean"],
        short_model.h.std(axis=1), rtol=0.0, atol=1e-12)


def test_component_scale_pools_several_sequences_and_refuses_a_wrong_length(short_model):
    pool = [short_model.wildtype, "".join(AA20[(i + 5) % 20] for i in range(SHORT))]
    summary = component_scale(short_model, pool)
    assert summary["sequences"] == 2
    assert summary["components"]["coupling"]["steps"] == 2 * SHORT
    with pytest.raises(ValueError, match="not"):
        component_scale(short_model, [short_model.wildtype[:-1]])
    with pytest.raises(ValueError, match="at least one sequence"):
        component_scale(short_model, [])
