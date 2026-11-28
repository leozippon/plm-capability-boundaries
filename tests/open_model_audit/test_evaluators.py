"""Unit tests for all 4 open-model evaluator families using deterministic synthetic PyTorch models."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.capability.open_model_audit.cohort import (
    AuditSequenceRecord,
    OpenAuditBundle,
    Paired3DRecord,
    atomic_write_json_with_sha256,
    atomic_write_jsonl,
    seq_sha256,
)
from src.capability.open_model_audit.evaluators import (
    CausalAROpenEvaluator,
    MDMOpenEvaluator,
    MaskedLMOpenEvaluator,
    ProteinMPNNOpenEvaluator,
    run_audit_for_model,
)
from src.capability.open_model_audit.metrics import AA20, DEPTH_BUCKET_ORDER, H0_BITS


class _DummyLM(nn.Module):
    """Tiny deterministic PyTorch LM returning logits of shape ``(B, T, V)``."""

    def __init__(self, vocab_size: int = 33, hidden: int = 16, bias_to_token: int | None = None) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        if bias_to_token is not None:
            with torch.no_grad():
                self.proj.bias[bias_to_token] = 4.0
        # Add large junk logit on non-AA special token 2 (<eos>) to verify AA20 stripping!
        with torch.no_grad():
            self.proj.bias[2] = 50.0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **_kwargs):
        del attention_mask
        logits = self.proj(self.emb(input_ids))
        return SimpleNamespace(logits=logits)


def _make_mini_bundle(tmp_path: Path) -> OpenAuditBundle:
    splits = tmp_path / "splits"
    splits.mkdir(parents=True)

    def _rec(rid: str, split: str, sub: str, seq: str, bucket: str | None = None) -> dict:
        return AuditSequenceRecord(
            record_id=rid,
            split=split,
            sub_stratum=sub,
            sequence=seq,
            seq_len=len(seq),
            seq_sha256=seq_sha256(seq),
            cluster_size=1 if bucket == "singleton" else 10,
            depth_bucket=bucket,
        ).to_dict()

    rng = np.random.default_rng(123)
    aa_list = list(AA20)

    def _rand_seq(L: int) -> str:
        return "".join(rng.choice(aa_list, size=L).tolist())

    seen_rows = []
    for b in DEPTH_BUCKET_ORDER:
        for i in range(4):
            seen_rows.append(_rec(f"seen_{b}_{i}", "seen_pretrain_buckets", b, _rand_seq(40 + i), bucket=b))
    seen_cath = [_rec(f"sc_{i}", "seen_cath_matched", "swissprot", _rand_seq(45)) for i in range(4)]
    hom = [_rec(f"hm_{i}", "unseen_homolog", "trembl_same_cath", _rand_seq(45)) for i in range(4)]
    fold = (
        [_rec(f"dn_{i}", "unseen_fold", "denovo_zero_hit", _rand_seq(42)) for i in range(4)]
        + [_rec(f"or_{i}", "unseen_fold", "orphan_cath_fold", _rand_seq(48)) for i in range(4)]
    )

    p3_rows = []
    coords_pack = {}
    for g_idx, grp in enumerate(("natural_cath_s40", "de_novo_zero_hit")):
        for i in range(3):
            s = _rand_seq(32)
            ck = f"{grp[:3]}_{i}"
            # Smooth helical-like non-degenerate 3D backbone coordinates
            t = np.arange(len(s), dtype=np.float32)
            ca = np.stack([3.8 * np.cos(t * 0.5), 3.8 * np.sin(t * 0.5), 1.5 * t], axis=-1)
            coords = np.stack([ca - 0.5, ca, ca + 0.5, ca + np.array([0.0, 0.4, 0.0], dtype=np.float32)], axis=1)
            coords_pack[ck] = coords.astype(np.float32)
            p3_rows.append(
                Paired3DRecord(
                    record_id=f"p3_{g_idx}_{i}",
                    domain_group=grp,
                    sequence=s,
                    seq_len=len(s),
                    seq_sha256=seq_sha256(s),
                    three_di_seq="D" * len(s),
                    coords_key=ck,
                    source_kind="synthetic",
                ).to_dict()
            )

    atomic_write_jsonl(splits / "seen_pretrain_buckets.jsonl", seen_rows)
    atomic_write_jsonl(splits / "seen_cath_matched.jsonl", seen_cath)
    atomic_write_jsonl(splits / "unseen_homolog.jsonl", hom)
    atomic_write_jsonl(splits / "unseen_fold.jsonl", fold)
    atomic_write_jsonl(splits / "paired_3d_panel.jsonl", p3_rows)
    np.savez_compressed(splits / "panel_3d_coords.npz", **coords_pack)

    pop = {b: {"n_clusters": 1000, "total_residues": 50_000} for b in DEPTH_BUCKET_ORDER}
    atomic_write_json_with_sha256(
        tmp_path / "bundle_manifest.json",
        {
            "schema_version": "open_model_audit_bundle_v1",
            "population_buckets_all": pop,
            "population_buckets_eligible": pop,
        },
    )
    return OpenAuditBundle(tmp_path)


def test_causal_ar_evaluator_uniform_and_bidirectional(tmp_path: Path):
    bundle = _make_mini_bundle(tmp_path)
    model = _DummyLM(vocab_size=33, hidden=16)  # zero weights -> uniform over AA20 despite huge EOS logit!
    ev = CausalAROpenEvaluator(
        model_id="dummy-progen2",
        model=model,
        aa20_ids=list(range(4, 24)),
        bos_prefix_ids=[1],
        eos_suffix_ids=[2],
        reverse_bos_prefix_ids=[2],
        reverse_eos_suffix_ids=[1],
        score_bidirectional=True,
        device="cpu",
        dtype="float32",
    )
    report = run_audit_for_model(ev, bundle, batch_size=4)
    ar_dn = report["sequence_splits"]["unseen_fold_denovo_zero_hit"]["ar_exact"]
    assert ar_dn["bits_per_res"] == pytest.approx(H0_BITS, abs=1e-4)
    assert "ar_c_to_n" in report["sequence_splits"]["unseen_fold_denovo_zero_hit"]
    assert "ar_bidirectional_avg" in report["sequence_splits"]["unseen_fold_denovo_zero_hit"]
    # Zero-info uniform model explains ~0 bits/res and ~0 bits/param (up to float32 cross_entropy rounding)
    ht = report["horvitz_thompson_all_uniref50"]["ar_exact"]
    assert abs(ht["corpus_weighted_delta_h_bits_per_res"]) < 1e-5
    assert abs(ht["B_total_bits"]) < 0.1
    assert abs(ht["c_app_bits_per_non_emb_param"]) < 1e-4


def test_masked_lm_and_mdm_evaluators_all_three_readouts(tmp_path: Path):
    bundle = _make_mini_bundle(tmp_path)
    model = _DummyLM(vocab_size=33, hidden=16)
    ev_mlm = MaskedLMOpenEvaluator(
        model_id="dummy-esm2",
        model=model,
        aa20_ids=list(range(4, 24)),
        mask_id=32,
        pad_id=1,
        cls_id=0,
        eos_id=2,
        device="cpu",
        dtype="float32",
    )
    report = run_audit_for_model(
        ev_mlm, bundle, batch_size=4, n_mlm15_draws=4, n_loo_probes=8, n_mdm_strata=8
    )
    for readout in ("mlm_15", "loo", "mdm_16"):
        assert readout in report["horvitz_thompson_all_uniref50"]
        assert readout in report["three_tier_eq3a_denovo_unseen_fold"]
        st = report["sequence_splits"]["unseen_fold_denovo_zero_hit"][readout]
        assert st["bits_per_res"] == pytest.approx(H0_BITS, abs=1e-4)
        assert st["delta_cal_bits"] == pytest.approx(0.0, abs=1e-4)
    assert len(report["sequence_splits"]["unseen_fold_denovo_zero_hit"]["mdm_16"]["stratum_curve"]) == 8

    # MDM evaluator uses mdm_16 as primary readout
    ev_mdm = MDMOpenEvaluator(
        model_id="dummy-mdm",
        model=model,
        aa20_ids=list(range(4, 24)),
        mask_id=32,
        pad_id=1,
        cls_id=0,
        eos_id=2,
        device="cpu",
        dtype="float32",
    )
    assert ev_mdm.family_kind == "mdm"
    assert ev_mdm.primary_readout == "mdm_16"


def test_proteinmpnn_evaluator_synthetic_adapter(tmp_path: Path):
    """Verify ProteinMPNNOpenEvaluator strips index-20 'X' logit and computes 3D panel metrics."""
    from src.capability.unified.proteinmpnn_adapter import ProteinMPNNModel

    bundle = _make_mini_bundle(tmp_path)
    model = ProteinMPNNModel(
        num_letters=21, node_features=32, edge_features=32, hidden_dim=32, num_encoder_layers=1, num_decoder_layers=1, k_neighbors=8
    ).eval()
    # Zero out W_out weights/biases except massive logit on index 20 ('X') -> must still yield exact H0_BITS!
    with torch.no_grad():
        model.W_out.weight.zero_()
        model.W_out.bias.zero_()
        model.W_out.bias[20] = 99.0

    class _FakeMPNNAdapter:
        arm = "proteinmpnn-dummy"
        device = "cpu"
        dtype = "float32"
        ckpt_noise_level = 0.2
        spec = SimpleNamespace(variant="vanilla", ca_only=False)

        def __init__(self, m):
            self.model = m

        def _encode_single_backbone(self, coords):
            L = coords.shape[0]
            X = torch.from_numpy(coords).unsqueeze(0).float()
            mask = torch.ones((1, L), dtype=torch.float32)
            res_idx = torch.arange(L, dtype=torch.long).unsqueeze(0)
            chain = torch.ones((1, L), dtype=torch.long)
            h_V, h_E, E_idx, h_EXV = self.model.encode_structure(X, mask, res_idx, chain)
            return h_V, h_E, E_idx, h_EXV, mask

    ev = ProteinMPNNOpenEvaluator(_FakeMPNNAdapter(model), n_permutations=2)
    report = run_audit_for_model(ev, bundle, n_loo_probes=4, seq_reference_bits=3.8)
    ov = report["paired_3d_panel"]["overall"]
    assert ov["ar_3d_perm_avg"]["bits_per_res"] == pytest.approx(H0_BITS, abs=1e-4)
    assert ov["loo_3d"]["bits_per_res"] == pytest.approx(H0_BITS, abs=1e-4)
    assert "natural_cath_s40" in report["paired_3d_panel"]["by_domain_group"]
    assert "de_novo_zero_hit" in report["paired_3d_panel"]["by_domain_group"]
