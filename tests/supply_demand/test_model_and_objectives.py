"""Tests for Stage-2 matched Transformer parameter counts, padding zero-leakage, and 3 objectives."""
from __future__ import annotations

import math

import pytest
import torch

from src.capability.supply_demand.model import (
    VariableLengthProteinTransformer,
    count_model_parameters,
)
from src.capability.supply_demand.objectives import (
    PackedSequenceStore,
    collate_variable_length_sequences,
    compute_objective_loss,
    prepare_objective_batch,
)
from src.capability.supply_demand.spec import (
    AA20,
    BOS_TOKEN_ID,
    LN2,
    LOG2_20,
    MASK_TOKEN_ID,
    OBJECTIVES,
    PAD_TOKEN_ID,
    SCALE_SPECS,
    VOCAB_AA,
)

STAGE1_EXACT_NON_EMB_PARAMS = {
    "0.5m": 492_608,
    "2m": 1_992_640,
    "8m": 7_992_128,
    "35m": 35_337_088,
    "150m": 148_671_360,
}


def test_exact_stage1_non_embedding_param_counts() -> None:
    """All 5 scales must match Stage-1 n_params_non_embedding identically."""
    for scale_id, expected in STAGE1_EXACT_NON_EMB_PARAMS.items():
        spec = SCALE_SPECS[scale_id]
        assert spec.exact_non_embedding_params == expected
        # Instantiate 0.5m, 2m, 8m (and check formula + meta device for 35m/150m for speed)
        if scale_id in ("0.5m", "2m", "8m"):
            model = VariableLengthProteinTransformer(spec, "ar", max_seq_len=64)
            counts = count_model_parameters(model, spec)
            assert counts["n_params_non_embedding"] == expected
        else:
            with torch.device("meta"):
                model = VariableLengthProteinTransformer(spec, "oa_uniform", max_seq_len=64)
                counts = count_model_parameters(model, spec)
                assert counts["n_params_non_embedding"] == expected


@pytest.mark.parametrize("objective", ["mlm_15", "oa_uniform"])
def test_bidirectional_padding_zero_leakage_and_zero_pad_grad(objective: str) -> None:
    """<PAD> tokens must NEVER affect real token logits or receive gradient in bidirectional modes."""
    torch.manual_seed(123)
    model = VariableLengthProteinTransformer("0.5m", objective, max_seq_len=64)
    model.eval()

    seq = torch.randint(0, VOCAB_AA, (1, 18), dtype=torch.long)
    # 1) Unpadded
    valid_18 = torch.ones((1, 18), dtype=torch.bool)
    out_unpadded = model(seq, valid_18)

    # 2) Padded to 32 with PAD_TOKEN_ID
    pad32 = torch.full((1, 32), PAD_TOKEN_ID, dtype=torch.long)
    pad32[:, :18] = seq
    valid_32 = torch.zeros((1, 32), dtype=torch.bool)
    valid_32[:, :18] = True
    out_pad32 = model(pad32, valid_32)[:, :18]

    # 3) Padded to 48 with adversarial non-PAD garbage in masked-out positions
    pad48 = torch.randint(0, VOCAB_AA, (1, 48), dtype=torch.long)
    pad48[:, :18] = seq
    valid_48 = torch.zeros((1, 48), dtype=torch.bool)
    valid_48[:, :18] = True
    out_pad48 = model(pad48, valid_48)[:, :18]

    assert torch.allclose(out_unpadded, out_pad32, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_unpadded, out_pad48, atol=1e-5, rtol=1e-5)

    # Verify zero gradient on PAD embedding row when loss is computed only on valid masked sites
    model.train()
    model.zero_grad(set_to_none=True)
    lengths = torch.tensor([18], dtype=torch.long)
    gen = torch.Generator().manual_seed(7)
    prep = prepare_objective_batch(objective, pad32, lengths, generator=gen)
    logits = model(prep.input_ids, prep.attention_mask)
    loss = compute_objective_loss(logits, prep)
    loss.backward()
    pad_grad = model.tok_emb.weight.grad[PAD_TOKEN_ID]
    assert torch.all(pad_grad == 0.0), f"Non-zero gradient leaked to <PAD> embedding: {pad_grad.abs().max()}"


def test_ar_padding_zero_leakage_and_strict_causality() -> None:
    """AR must have zero padding leakage and strict left-to-right causality."""
    torch.manual_seed(456)
    model = VariableLengthProteinTransformer("0.5m", "ar", max_seq_len=64)
    model.eval()

    seq = torch.randint(0, VOCAB_AA, (1, 20), dtype=torch.long)
    inp_unpad = torch.cat([torch.tensor([[BOS_TOKEN_ID]]), seq[:, :-1]], dim=1)
    out_unpad = model(inp_unpad, right_padded=True)

    inp_pad = torch.full((1, 40), PAD_TOKEN_ID, dtype=torch.long)
    inp_pad[:, :20] = inp_unpad
    out_pad = model(inp_pad, right_padded=True)[:, :20]
    assert torch.allclose(out_unpad, out_pad, atol=1e-5, rtol=1e-5)

    # Mutate input at index 10 -> outputs at indices 0..9 must be bit-identical
    inp_mut = inp_unpad.clone()
    inp_mut[0, 10] = (inp_mut[0, 10] + 1) % VOCAB_AA
    out_mut = model(inp_mut, right_padded=True)
    assert torch.allclose(out_unpad[:, :10], out_mut[:, :10], atol=1e-6)
    assert not torch.allclose(out_unpad[:, 10:], out_mut[:, 10:], atol=1e-5)


@pytest.mark.parametrize("objective", list(OBJECTIVES))
def test_objectives_pad_exclusion_and_exact_log2_20_uniform_baseline(objective: str) -> None:
    """All 3 objectives must exclude <PAD> from masks/loss and yield exact log2(20) at zero logits."""
    seqs = [
        "".join(AA20[(i + j) % 20] for j in range(12 + 5 * i))
        for i in range(8)
    ]
    tokens, lengths = collate_variable_length_sequences(seqs, pad_to_multiple=8)
    gen = torch.Generator().manual_seed(99)
    prep = prepare_objective_batch(objective, tokens, lengths, generator=gen)

    bsz, max_len = tokens.shape
    for i in range(bsz):
        L = int(lengths[i].item())
        assert not prep.loss_mask[i, L:].any(), f"{objective} masked a <PAD> position!"
        assert not prep.attention_mask[i, L:].any()
        assert prep.attention_mask[i, :L].all()
        n_masked = int(prep.loss_mask[i].sum().item())
        assert n_masked == int(prep.n_loss_per_seq[i].item())
        if objective == "ar":
            assert int(prep.input_ids[i, 0].item()) == BOS_TOKEN_ID
            assert n_masked == L
        elif objective == "mlm_15":
            assert n_masked == max(1, round(0.15 * L))
            assert (prep.input_ids[i][prep.loss_mask[i]] == MASK_TOKEN_ID).all()
        elif objective == "oa_uniform":
            assert 1 <= n_masked <= L
            assert (prep.input_ids[i][prep.loss_mask[i]] == MASK_TOKEN_ID).all()

    zero_logits = torch.zeros((bsz, max_len, VOCAB_AA), dtype=torch.float32)
    loss_nats = compute_objective_loss(zero_logits, prep)
    loss_bits = float(loss_nats.item()) / LN2
    assert math.isclose(loss_bits, LOG2_20, rel_tol=1e-6, abs_tol=1e-6)


def test_packed_sequence_store_roundtrip() -> None:
    seqs = ["ACDEFGHIKLMNPQRSTVWY", "MKTAYIAKQRQISFVKSHFSRQ", "G" * 50]
    store = PackedSequenceStore.from_sequences(seqs)
    assert store.n_sequences == 3
    assert store.total_tokens == sum(len(s) for s in seqs)
    for idx, s in enumerate(seqs):
        assert store.get_sequence(idx) == s
    toks, lens = store.collate_indices([2, 0], pad_to_multiple=8)
    assert toks.shape == (2, 56)
    assert lens.tolist() == [50, 20]
    assert (toks[1, 20:] == PAD_TOKEN_ID).all()
