"""Unit and offline integration smoke tests for ESM3 adapter and R4/R5/R123 scripts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.capability.esm3_ext.run_esm3_r123_seqonly import compute_cycle_epistasis
from scripts.capability.esm3_ext.run_esm3_r4_multimodal import (
    paired_auroc,
    permute_structure_tokens_on_mask,
    project_template_onto_target,
    project_unrelated_onto_mask,
)
from src.capability.esm3_ext.esm3_adapter import (
    AA20_SET,
    _cosine_unmask_steps,
    verify_offline_assets,
)


def test_verify_offline_assets_fails_fast_on_incomplete_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing required assets"):
        verify_offline_assets(tmp_path)


@pytest.mark.parametrize(
    ("total", "steps"),
    [(1, 1), (5, 5), (40, 40), (152, 64), (300, 64), (10, 64)],
)
def test_cosine_unmask_schedule_exact_sum(total: int, steps: int) -> None:
    counts = _cosine_unmask_steps(total, steps)
    assert sum(counts) == total
    assert all(c >= 1 for c in counts)
    assert len(counts) <= min(total, steps)


def test_r4_template_projection_never_leaks_unaligned_and_preserves_mask() -> None:
    # Target length 8; only indices {1, 2, 5, 6} aligned to template indices {0, 1, 3, 4}
    tmpl_coords = np.arange(5 * 37 * 3, dtype=np.float32).reshape(5, 37, 3)
    tmpl_ss8 = "HHEEC"
    aligned = [(1, 0), (2, 1), (5, 3), (6, 4)]

    proj_c, proj_ss8, mask_a = project_template_onto_target(8, aligned, tmpl_coords, tmpl_ss8)
    assert mask_a == [1, 2, 5, 6]
    assert proj_ss8 == "_HH__EC_"
    for idx in (0, 3, 4, 7):
        assert np.isnan(proj_c[idx]).all(), f"Unaligned target position {idx} leaked coords"
    assert np.allclose(proj_c[1], tmpl_coords[0])
    assert np.allclose(proj_c[6], tmpl_coords[4])

    # Unrelated fold projected onto the identical mask A
    unrel_coords = np.ones((10, 37, 3), dtype=np.float32) * 7.0
    unrel_ss8 = "CCCCCCCCCC"
    u_c, u_ss8 = project_unrelated_onto_mask(8, mask_a, unrel_coords, unrel_ss8)
    assert u_ss8 == "_CC__CC_"
    for idx in (0, 3, 4, 7):
        assert np.isnan(u_c[idx]).all()
    for idx in mask_a:
        assert np.allclose(u_c[idx], 7.0)

    # Permuting structure tokens modifies ONLY positions in mask A (shifted +1 for BOS)
    toks = torch.tensor([4098, 4096, 10, 20, 4096, 4096, 30, 40, 4096, 4097], dtype=torch.long)
    perm = permute_structure_tokens_on_mask(toks, mask_a, seed=42)
    assert int(perm[0]) == 4098 and int(perm[-1]) == 4097
    for idx in (0, 3, 4, 7):
        assert int(perm[idx + 1]) == 4096
    assert sorted(int(perm[i + 1]) for i in mask_a) == [10, 20, 30, 40]
    assert [int(perm[i + 1]) for i in mask_a] != [10, 20, 30, 40]


def test_paired_auroc_and_cycle_epistasis() -> None:
    assert paired_auroc([1.0, 2.0, 3.0], [2.0, 2.0, 1.0]) == pytest.approx(0.5)
    assert paired_auroc([1.0, 1.5], [2.0, 2.5]) == pytest.approx(1.0)
    assert compute_cycle_epistasis(-100.0, -101.0, -102.0, -102.5) == pytest.approx(0.5)


def _staged_esm3_dir() -> Path | None:
    for p in (
        Path("/home/lzp/models/general_plm_20261120/esm3-sm-open-v1"),
        Path("/gpfs/jiaotongdamoxing/zhk_zip/models/esm3-sm-open-v1"),
    ):
        if (p / "data/weights/esm3_sm_open_v1.pth").is_file():
            return p
    return None


@pytest.mark.skipif(
    _staged_esm3_dir() is None or not torch.cuda.is_available(),
    reason="Requires staged offline esm3-sm-open-v1 checkpoint and CUDA GPU",
)
def test_esm3_offline_gpu_smoke() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    from src.capability.esm3_ext import (
        encode_structure_tokens,
        exact_pll,
        extract_hidden_states,
        generate_sequence_cosine,
        infill_window_cosine,
        load_esm3_offline,
    )

    handle = load_esm3_offline(
        _staged_esm3_dir(), device="cuda:0", dtype="bfloat16", eager_load_aux=True
    )
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG"
    pll = exact_pll(handle, seq, batch_size=16, positions=[0, 5, 10, 15, 20])
    assert np.isfinite(pll["mean_nll_per_residue_nats"])
    assert pll["scored_tokens"] == 5

    # Synthetic backbone coordinates (idealized alpha-like chain N, CA, C)
    L = len(seq)
    coords = np.full((L, 37, 3), np.nan, dtype=np.float32)
    for i in range(L):
        coords[i, 0] = [i * 3.8, 0.0, 0.0]
        coords[i, 1] = [i * 3.8 + 1.45, 0.2, 0.1]
        coords[i, 2] = [i * 3.8 + 2.5, -0.3, 0.2]
    coords_pad, struct_toks = encode_structure_tokens(handle, coords, reference_sequence=seq)
    assert struct_toks.shape == (L + 2,)
    assert coords_pad.shape == (L + 2, 37, 3)

    pll_3d = exact_pll(
        handle,
        seq,
        batch_size=8,
        positions=[5, 10],
        structure_tokens=struct_toks,
        structure_coords=coords_pad,
    )
    assert np.isfinite(pll_3d["mean_nll_per_residue_nats"])

    hs = extract_hidden_states(handle, [seq], batch_size=1)
    assert hs["middle_mean"].shape == (1, handle.d_model)
    assert hs["final_cls"].shape == (1, handle.d_model)

    inf = infill_window_cosine(handle, seq, start=10, extent=5, seed=123, num_steps=5)
    assert len(inf["sequence"]) == L and len(inf["window"]) == 5
    assert set(inf["sequence"]) <= AA20_SET

    gen = generate_sequence_cosine(handle, 32, seed=456, num_steps=8, temperature=0.8)
    assert len(gen["sequence"]) == 32
    assert set(gen["sequence"]) <= AA20_SET
