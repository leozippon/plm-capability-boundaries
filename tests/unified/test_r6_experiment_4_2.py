"""Unit & reproduction-gate tests for Unified Experiment 4.2 (Eight-Domain Phase Space)."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.capability.context.family_potts import FamilyModel, load_family_model, potts_score
from src.capability.core.amino_acids import AA20
from src.capability.decoding import guided_decoding as gd
from src.capability.unified import r6_experiment_4_2 as exp42

# --------------------------------------------------------------------------- #
# Synthetic CPU fixtures for all 4 guided sampling engines
# --------------------------------------------------------------------------- #


def _synthetic_family_model(domain: str = "SYN1", length: int = 8) -> FamilyModel:
    rng = np.random.default_rng(12345)
    wt = "".join(AA20[i % 20] for i in range(length))
    h = rng.normal(0.0, 0.5, size=(length, 20))
    J = np.zeros((length, length, 20, 20), dtype=np.float64)
    for i in range(length):
        for j in range(i + 1, length):
            block = rng.normal(0.0, 0.25, size=(20, 20))
            J[i, j] = block
            J[j, i] = block.T
    freq = rng.dirichlet(np.ones(20), size=length)
    cw = np.full(length, 50.0, dtype=np.float64)
    p = freq * cw[:, None] + 0.5
    log_prof = np.log(p / p.sum(axis=1, keepdims=True))
    return FamilyModel(
        domain=domain,
        wildtype=wt,
        alphabet=AA20,
        h=h,
        J=J,
        profile_freq=freq,
        profile_column_weight=cw,
        log_profile=log_prof,
        column_to_wt=np.arange(length, dtype=np.int64),
        qualified=True,
        qualification_failures=(),
        sidecar={},
    )


class _DummyTokenizer:
    """Tiny vocabulary carrying BOS (0), single AA20 (1..20), and one 2-AA BPE token 'AC' (21)."""

    def __init__(self, include_bpe: bool = False) -> None:
        self._vocab: dict[str, int] = {"<bos>": 0}
        for i, aa in enumerate(AA20, start=1):
            self._vocab[aa] = i
        if include_bpe:
            self._vocab["AC"] = 21
        self._id_to_tok = {v: k for k, v in self._vocab.items()}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def batch_decode(self, seqs: list[list[int]], **_: Any) -> list[str]:
        return ["".join(self._id_to_tok[int(i)] for i in row) for row in seqs]

    def decode(self, ids: list[int], **_: Any) -> str:
        return "".join(self._id_to_tok[int(i)] for i in ids)

    def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
        return {"input_ids": [0]}


class _TinyDeterministicLM(torch.nn.Module):
    """Tiny deterministic causal/masked/ESM3-compatible mock model."""

    def __init__(self, vocab_size: int = 24, *, require_progen3_args: bool = False) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.require_progen3_args = require_progen3_args
        gen = torch.Generator().manual_seed(99)
        self.embed = torch.nn.Parameter(torch.randn(vocab_size, 16, generator=gen))
        self.proj = torch.nn.Parameter(torch.randn(16, vocab_size, generator=gen))

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        sequence_ids: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> Any:
        if self.require_progen3_args:
            assert position_ids is not None and sequence_ids is not None
            assert (sequence_ids == 0).all()
        full = input_ids if past_key_values is None else torch.cat([past_key_values, input_ids], dim=1)
        h = self.embed[full]  # (B, T, 16)
        cum = torch.cumsum(h, dim=1) / torch.arange(1, full.shape[1] + 1, dtype=h.dtype).view(1, -1, 1)
        logits = cum @ self.proj
        if past_key_values is not None:
            logits = logits[:, -input_ids.shape[1] :, :]
        return SimpleNamespace(logits=logits, past_key_values=full if use_cache else None)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_panel_manifest_and_sharding_invariants() -> None:
    assert len(exp42.SEALED_8_DOMAINS) == 8
    assert len(exp42.PHASE_SPACE_MODELS) == 23
    assert len(exp42.CACHED_LOCAL_PLM_MODELS) == 3
    assert len(exp42.NEW_H200_MODELS) == 20
    assert len(exp42.PHASE_SPACE_ARMS) == 5

    # Generation shards: 20 models across 6 GPUs -> disjoint & exhaustive
    gen_shards = exp42.plan_generation_shards(exp42.NEW_H200_MODELS, num_shards=6)
    assert len(gen_shards) == 6
    flat_gen = [m for shard in gen_shards for m in shard]
    assert len(flat_gen) == 20
    assert set(flat_gen) == set(exp42.NEW_H200_MODELS)
    assert all(len(s) >= 2 for s in gen_shards)

    # Fold shards: 20 * 8 * 5 = 800 cells across 6 GPUs -> disjoint, exhaustive, balanced
    all_cells = [
        (m, d, a)
        for m in exp42.NEW_H200_MODELS
        for d in exp42.SEALED_8_DOMAINS
        for a in exp42.PHASE_SPACE_ARM_KEYS
    ]
    fold_shards = exp42.plan_fold_shards(all_cells, num_shards=6)
    flat_fold = [c for shard in fold_shards for c in shard]
    assert len(flat_fold) == 800
    assert set(flat_fold) == set(all_cells)
    sizes = [len(s) for s in fold_shards]
    assert max(sizes) - min(sizes) <= 1


def test_guided_causal_decoding_engine_cache_and_bpe_parity() -> None:
    fm = _synthetic_family_model(length=8)
    tok = _DummyTokenizer(include_bpe=True)
    model = _TinyDeterministicLM(vocab_size=22)

    @dataclass
    class _FakeArm:
        name: str
        model: Any
        tokenizer: Any
        device: str
        generation_point: Any

    arm = _FakeArm(
        name="prollama",
        model=model,
        tokenizer=tok,
        device="cpu",
        generation_point=SimpleNamespace(prompt="Seq=<", add_special_tokens=True),
    )

    # Cached vs uncached exact parity across all 5 arms (including multi-residue 'AC' BPE token)
    for arm_spec in exp42.PHASE_SPACE_ARMS:
        rec_cached = exp42.sample_guided_causal_decoding(
            arm=arm,
            model_id="prollama",
            family_model=fm,
            arm_spec=arm_spec,
            count=6,
            batch_size=3,
            stream_seed=5000,
            use_cache=True,
            logit_width=22,
        )
        rec_uncached = exp42.sample_guided_causal_decoding(
            arm=arm,
            model_id="prollama",
            family_model=fm,
            arm_spec=arm_spec,
            count=6,
            batch_size=3,
            stream_seed=5000,
            use_cache=False,
            logit_width=22,
        )
        assert [r["sequence"] for r in rec_cached] == [r["sequence"] for r in rec_uncached]
        assert all(len(r["sequence"]) == 8 and set(r["sequence"]) <= set(AA20) for r in rec_cached)


def test_general_plm_causal_and_progen3_guided_engines() -> None:
    fm = _synthetic_family_model(length=7)
    model_gp = _TinyDeterministicLM(vocab_size=22)
    fake_adapter = SimpleNamespace(
        device="cpu",
        model=model_gp,
        bos_prefix_ids=[0],
        aa20_ids=list(range(1, 21)),
        _precision_ctx=lambda: nullcontext(),
        _init_causal_cache=lambda batch_size: None,
    )
    for arm_spec in exp42.PHASE_SPACE_ARMS:
        rows = exp42.sample_guided_general_plm_causal(
            adapter=fake_adapter,
            model_id="dayhoff-170m",
            family_model=fm,
            arm_spec=arm_spec,
            count=4,
            batch_size=2,
            stream_seed=7000,
        )
        assert len(rows) == 4
        assert all(len(r["sequence"]) == 7 and set(r["sequence"]) <= set(AA20) for r in rows)

    # Zero-strength (Pr with lambda=0) must reproduce N identically
    zero_pr = exp42.PhaseSpaceArmSpec("Pr0", "profile", 0.0, 0.0, "zero")
    r_n = exp42.sample_guided_general_plm_causal(
        adapter=fake_adapter, model_id="dayhoff-170m", family_model=fm,
        arm_spec=exp42.PHASE_SPACE_ARM_MAP["N"], count=4, batch_size=2, stream_seed=7100,
    )
    r_0 = exp42.sample_guided_general_plm_causal(
        adapter=fake_adapter, model_id="dayhoff-170m", family_model=fm,
        arm_spec=zero_pr, count=4, batch_size=2, stream_seed=7100,
    )
    assert [r["sequence"] for r in r_n] == [r["sequence"] for r in r_0]

    # ProGen3 singleton engine with position_ids & sequence_ids=0 assertions
    tok_map = {"<bos>": 0, "1": 21, **{aa: i + 1 for i, aa in enumerate(AA20)}}
    pg_handle = SimpleNamespace(
        device="cpu",
        config=SimpleNamespace(bos_token_id=0),
        tokenizer=SimpleNamespace(token_to_id=lambda s: tok_map[s]),
        model=_TinyDeterministicLM(vocab_size=22, require_progen3_args=True),
    )
    for arm_spec in exp42.PHASE_SPACE_ARMS:
        rows = exp42.sample_guided_progen3(
            pg_handle=pg_handle,
            model_id="progen3-219m",
            family_model=fm,
            arm_spec=arm_spec,
            count=3,
            batch_size=1,
            stream_seed=8000,
        )
        assert len(rows) == 3
        assert all(len(r["sequence"]) == 7 for r in rows)


def test_dplm_and_esm3_diffusion_guided_engines() -> None:
    fm = _synthetic_family_model(length=8)
    mask_id = 21
    bos_id = 0
    eos_id = 22
    lm = _TinyDeterministicLM(vocab_size=24)

    # DPLM adapter mock
    fake_dplm = SimpleNamespace(
        device="cpu",
        mask_id=mask_id,
        aa20_ids=list(range(1, 21)),
        _precision_ctx=lambda: nullcontext(),
        _encode_sequence=lambda seq: ([bos_id] + [1] * len(seq) + [eos_id], 1),
        _forward_logits=lambda ids: lm(ids, use_cache=False).logits,
    )
    for arm_spec in exp42.PHASE_SPACE_ARMS:
        rows = exp42.sample_guided_dplm(
            adapter=fake_dplm,
            model_id="dplm-150m",
            family_model=fm,
            arm_spec=arm_spec,
            count=4,
            batch_size=2,
            stream_seed=9000,
        )
        assert len(rows) == 4
        assert all(len(r["sequence"]) == 8 and set(r["sequence"]) <= set(AA20) for r in rows)

    # Verify zero-strength C at lambda=0 reproduces N identically in DPLM
    zero_c = exp42.PhaseSpaceArmSpec("C0", "coupling", 0.0, 0.0, "zero")
    d_n = exp42.sample_guided_dplm(
        adapter=fake_dplm, model_id="dplm-150m", family_model=fm,
        arm_spec=exp42.PHASE_SPACE_ARM_MAP["N"], count=4, batch_size=2, stream_seed=9100,
    )
    d_0 = exp42.sample_guided_dplm(
        adapter=fake_dplm, model_id="dplm-150m", family_model=fm,
        arm_spec=zero_c, count=4, batch_size=2, stream_seed=9100,
    )
    assert [r["sequence"] for r in d_n] == [r["sequence"] for r in d_0]

    # ESM3 handle mock
    fake_esm3 = SimpleNamespace(
        device=torch.device("cpu"),
        bos_id=bos_id,
        eos_id=eos_id,
        mask_id=mask_id,
        aa20_token_ids=tuple(range(1, 21)),
    )

    def _fake_esm3_forward(_handle: Any, *, sequence_tokens: torch.Tensor) -> Any:
        return SimpleNamespace(sequence_logits=lm(sequence_tokens, use_cache=False).logits)

    for arm_spec in exp42.PHASE_SPACE_ARMS:
        rows = exp42.sample_guided_esm3(
            handle=fake_esm3,
            model_id="esm3-1.4b",
            family_model=fm,
            arm_spec=arm_spec,
            count=4,
            batch_size=2,
            stream_seed=9500,
            forward_fn=_fake_esm3_forward,
        )
        assert len(rows) == 4
        assert all(len(r["sequence"]) == 8 and set(r["sequence"]) <= set(AA20) for r in rows)

    e_n = exp42.sample_guided_esm3(
        handle=fake_esm3, model_id="esm3-1.4b", family_model=fm,
        arm_spec=exp42.PHASE_SPACE_ARM_MAP["N"], count=4, batch_size=2, stream_seed=9600,
        forward_fn=_fake_esm3_forward,
    )
    e_0 = exp42.sample_guided_esm3(
        handle=fake_esm3, model_id="esm3-1.4b", family_model=fm,
        arm_spec=zero_c, count=4, batch_size=2, stream_seed=9600,
        forward_fn=_fake_esm3_forward,
    )
    assert [r["sequence"] for r in e_n] == [r["sequence"] for r in e_0]


def test_potts_energy_decomposition_and_shannon_entropy() -> None:
    fm = load_family_model("1E0L", exp42.DEFAULT_FAMILY_REF_DIR)
    full, fields, couplings = exp42.compute_potts_energies(fm, fm.wildtype)
    assert abs(full - (fields + couplings)) < 1e-12
    assert abs(full - potts_score(fm, fm.wildtype)) < 1e-12

    # Uniform 20-AA sequence has log2(20) bits; homopolymer has 0 bits.
    assert abs(exp42.compute_shannon_entropy_bits(AA20) - np.log2(20.0)) < 1e-12
    assert exp42.compute_shannon_entropy_bits("A" * 20) == 0.0

    # Verify Cx uses exact permutation_for(domain, L, 20261013)
    bias_cx = exp42.build_phase_space_bias(fm, exp42.PHASE_SPACE_ARM_MAP["Cx"])
    expected_perm = gd.permutation_for("1E0L", fm.length, 20261013)
    assert np.array_equal(np.asarray(bias_cx.provenance["column_permutation"]), expected_perm)
    assert np.array_equal(bias_cx.C, fm.J.astype(np.float64)[np.ix_(expected_perm, expected_perm)])


def test_real_e20_and_progen3_reproduction_gates_and_partial_assembly() -> None:
    e20_gate = exp42.verify_e20_confirmatory_reproduction()
    assert e20_gate["passed"] is True
    assert e20_gate["n_scalars_checked"] > 1000
    assert e20_gate["max_abs_diff"] < 1e-12

    pg3_gate = exp42.verify_progen3_robustness_reproduction()
    assert pg3_gate["passed"] is True
    assert pg3_gate["n_scalars_checked"] > 40
    assert pg3_gate["max_abs_diff"] < 1e-12

    # Partial assembly on the 5 cached local arms (P, H, progen2-medium, protgpt2, progen3-112m)
    summary = exp42.assemble_phase_space_summary(allow_partial=True)
    assert summary["schema_version"] == exp42.SCHEMA_VERSION
    assert set(summary["reference_arms"]) == {"P", "H"}
    assert summary["coverage"]["n_completed_plms"] == 3
    assert summary["coverage"]["n_pending_plms"] == 20
    assert summary["coverage"]["expected_new_h200_cells"] == 800
    assert summary["reproduction_gates"]["e20_confirmatory_20261014"]["max_abs_diff"] < 1e-12
    assert summary["reproduction_gates"]["progen3_robustness_20261015"]["max_abs_diff"] < 1e-12
    for cached_plm in ("progen2-medium", "protgpt2", "progen3-112m"):
        entry = summary["models"][cached_plm]
        assert "N" in entry["arms"] and "Pr" in entry["arms"] and "C" in entry["arms"]
        assert "C-Pr" in entry["contrasts"]["novel_lt20_tm"]
        assert "C-Cx" in entry["contrasts"]["whole_pool_tm"]

    # Non-partial assembly must fail fast while H200 cells are absent
    with pytest.raises(FileNotFoundError):
        exp42.assemble_phase_space_summary(allow_partial=False)
