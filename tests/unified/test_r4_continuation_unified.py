"""Unit & historical reproduction-gate tests for Unified Workstream R4-Continuation (31 Models)."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.capability.core.amino_acids import AA20
from src.capability.unified import r4_continuation_yardstick as r4c


class _TinyMockLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 26, *, require_pg3: bool = False) -> None:
        super().__init__()
        self.require_pg3 = require_pg3
        gen = torch.Generator().manual_seed(42)
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
        if self.require_pg3:
            assert position_ids is not None and sequence_ids is not None
            assert (sequence_ids == 0).all()
        full = input_ids if past_key_values is None else torch.cat([past_key_values, input_ids], dim=1)
        h = self.embed[full]
        cum = torch.cumsum(h, dim=1) / torch.arange(1, full.shape[1] + 1, dtype=h.dtype).view(1, -1, 1)
        logits = cum @ self.proj
        if past_key_values is not None:
            logits = logits[:, -input_ids.shape[1] :, :]
        return SimpleNamespace(logits=logits, past_key_values=full if use_cache else None)


def test_roster_panel_and_sharding_invariants() -> None:
    assert len(r4c.R4_SINGLE_CHAIN_MODEL_IDS) == 31
    assert len(r4c.R4_MULTISEQ_ICL_MODEL_IDS) == 4
    assert len(r4c.R4_CONTINUATION_MODELS) == 35
    assert sum(1 for m in r4c.R4_CONTINUATION_MODELS if m.is_text_control) == 2
    assert all(m.input_alignment_format == "unaligned_concat" for m in r4c.R4_CONTINUATION_MODELS)

    cells = r4c.required_target_condition_cells()
    assert len(cells) == 80
    assert len({t for t, _ in cells}) == 22

    # 3-GPU balanced sharding covers all 35 entries disjointly
    shards = r4c.plan_model_shards(r4c.R4_MODEL_IDS, num_shards=3)
    flat = [m for s in shards for m in s]
    assert len(flat) == 35
    assert set(flat) == set(r4c.R4_MODEL_IDS)

    # Multi-sequence ICL prompt grammars (unaligned, NO '-' gaps):
    # ProtGPT3-MSA: <|bos|> 1 <no_gap> + hom1 <s> hom2 <s>
    aa_to_id = {aa: i + 10 for i, aa in enumerate(AA20)}
    fake_pg3_msa = SimpleNamespace(
        bos_prefix_ids=[1, 2, 3],  # <|bos|>, 1, <no_gap>
        separator_ids=[4],  # <s>
        aa_to_id=aa_to_id,
        tokenizer=SimpleNamespace(eos_token_id=5, convert_tokens_to_ids=lambda s: {"2": 6, "<|eos|>": 5}.get(s, -1)),
    )
    prompt_pg3, stops_pg3, used_pg3 = r4c.build_causal_prompt_and_stops(
        model_id="protgpt3-msa|b8192",
        backend="multiseq_icl_causal",
        loaded=fake_pg3_msa,
        context_sequences=["ACD", "EFG"],
        budget=8192,
    )
    assert used_pg3 == 2
    assert prompt_pg3 == [1, 2, 3, aa_to_id["A"], aa_to_id["C"], aa_to_id["D"], 4, aa_to_id["E"], aa_to_id["F"], aa_to_id["G"], 4]
    assert stops_pg3 == {4, 5, 6}

    # Dayhoff-3B-MSA: @ + hom1 / hom2 /
    fake_dh_msa = SimpleNamespace(
        bos_prefix_ids=[7],  # @
        separator_ids=[8],  # /
        aa_to_id=aa_to_id,
        tokenizer=SimpleNamespace(eos_token_id=9, convert_tokens_to_ids=lambda s: {"*": 9}.get(s, -1)),
    )
    prompt_dh, stops_dh, used_dh = r4c.build_causal_prompt_and_stops(
        model_id="dayhoff-3b-msa|b4096",
        backend="multiseq_icl_causal",
        loaded=fake_dh_msa,
        context_sequences=["ACD", "EFG"],
        budget=4096,
    )
    assert used_dh == 2
    assert prompt_dh == [7, aa_to_id["A"], aa_to_id["C"], aa_to_id["D"], 8, aa_to_id["E"], aa_to_id["F"], aa_to_id["G"], 8]
    assert stops_dh == {8, 9}


def test_diffusion_and_causal_continuation_engines_preserve_context() -> None:
    lm = _TinyMockLM(vocab_size=26)
    aa_to_id = {aa: i + 1 for i, aa in enumerate(AA20)}
    bos_id, eos_id, mask_id = 0, 21, 22

    # 1. DPLM uniform_random continuation preserves fixed context prefix and emits exact L_gen AA20
    fake_dplm = SimpleNamespace(
        device="cpu",
        mask_id=mask_id,
        aa20_ids=list(range(1, 21)),
        aa_to_id=aa_to_id,
        tokenizer=SimpleNamespace(cls_token_id=bos_id, eos_token_id=eos_id),
        _precision_ctx=lambda: nullcontext(),
        _forward_logits=lambda ids: lm(ids, use_cache=False).logits,
    )
    seqs, prompt_toks, used = r4c.sample_dplm_continuations(
        adapter=fake_dplm,
        context_sequences=["ACDEFGHIKLMNPQRSTVWY" * 2, "MNPQRSTVWYACDEFGHIKL" * 2],
        target_length=40,
        count=4,
        batch_size=2,
        budget=256,
        seed=1234,
    )
    assert len(seqs) == 4 and used == 2 and prompt_toks > 80
    assert all(len(s) == 40 and set(s) <= set(AA20) for s in seqs)

    # 2. ESM3 cosine continuation preserves fixed context prefix and emits exact L_gen AA20
    fake_esm3 = SimpleNamespace(
        device=torch.device("cpu"),
        bos_id=bos_id,
        eos_id=eos_id,
        mask_id=mask_id,
        aa20_token_ids=tuple(range(1, 21)),
        aa_to_id=aa_to_id,
    )
    seqs_e3, _, used_e3 = r4c.sample_esm3_continuations(
        handle=fake_esm3,
        context_sequences=["ACDEFGHIKLMNPQRSTVWY" * 2],
        target_length=36,
        count=4,
        batch_size=2,
        budget=256,
        seed=5678,
        forward_fn=lambda _h, *, sequence_tokens: SimpleNamespace(
            sequence_logits=lm(sequence_tokens, use_cache=False).logits
        ),
    )
    assert len(seqs_e3) == 4 and used_e3 == 1
    assert all(len(s) == 36 and set(s) <= set(AA20) for s in seqs_e3)

    # 3. Causal cached vs uncached parity + ProGen3 position_ids/sequence_ids=0 check
    id_to_tok = {0: "<bos>", 21: "2", 22: "<eos>", **{i + 1: aa for i, aa in enumerate(AA20)}}
    fake_tok = SimpleNamespace(decode=lambda ids, **_: "".join(id_to_tok.get(int(i), "A") for i in ids))
    pg3_loaded = SimpleNamespace(
        device="cpu",
        model=_TinyMockLM(vocab_size=26, require_pg3=True),
        tokenizer=fake_tok,
    )
    out_pg3 = r4c.sample_causal_continuations(
        model_id="progen3-112m",
        backend="progen3_causal",
        loaded=pg3_loaded,
        prompt_ids=[0, 1, 2, 3],
        stop_ids={21, 22},
        count=3,
        max_new_tokens=16,
        batch_size=3,
        use_cache=True,
        seed=999,
    )
    assert len(out_pg3) == 3

    gen_loaded = SimpleNamespace(device="cpu", model=lm, tokenizer=fake_tok)
    c_true = r4c.sample_causal_continuations(
        model_id="dayhoff-170m", backend="general_plm_causal", loaded=gen_loaded,
        prompt_ids=[0, 1, 2], stop_ids={21, 22}, count=4, max_new_tokens=12,
        batch_size=2, use_cache=True, seed=321,
    )
    c_false = r4c.sample_causal_continuations(
        model_id="dayhoff-170m", backend="general_plm_causal", loaded=gen_loaded,
        prompt_ids=[0, 1, 2], stop_ids={21, 22}, count=4, max_new_tokens=12,
        batch_size=2, use_cache=False, seed=321,
    )
    assert c_true == c_false


def test_structural_tm_score_rigid_invariance_and_unequal_length() -> None:
    # Ideal alpha-helix backbone (L=50)
    t = np.arange(50, dtype=np.float64)
    ref = np.column_stack([2.3 * np.cos(t * 1.745), 2.3 * np.sin(t * 1.745), 1.5 * t])
    # Rigid rotation + translation
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    mob_rigid = ref @ rot.T + np.array([12.0, -7.5, 3.2])
    assert abs(r4c.structural_tm_score(mob_rigid, ref) - 1.0) < 1e-6

    # Unequal length with 12-residue N-terminal extension
    t_ext = np.arange(-12, 50, dtype=np.float64)
    mob_ext = np.column_stack([2.3 * np.cos(t_ext * 1.745), 2.3 * np.sin(t_ext * 1.745), 1.5 * t_ext]) @ rot.T
    assert r4c.structural_tm_score(mob_ext, ref) > 0.90

    # Random distant cloud has low TM-score
    rng = np.random.default_rng(7)
    rand_mob = rng.normal(0.0, 25.0, size=(55, 3))
    assert r4c.structural_tm_score(rand_mob, ref) < 0.35


def test_real_historical_e11_reproduction_gate_and_canonical_plans(tmp_path: Path) -> None:
    gate = r4c.verify_historical_e11_reproduction()
    assert gate["passed"] is True
    assert gate["n_scalars_checked"] >= 30
    assert gate["max_abs_diff"] < 1e-12

    # Non-partial assembly must fail fast when H200 pools are missing
    with pytest.raises(FileNotFoundError):
        r4c.assemble_r4_continuation_summary(
            work_dir=tmp_path, run_annotations=False, allow_partial=False
        )
