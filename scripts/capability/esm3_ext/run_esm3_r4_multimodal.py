#!/usr/bin/env python3
"""ESM3-1.4B R4 Multimodal Non-Parametric Context Access Stage.

Tests whether non-parametric 3D geometric context overcomes the <30% twilight-zone
sequence-context failure across identity strata (<20%, 20-30%, 30-50%, 50-70%, 70-90%).

Six paired conditions per target at identical alignment coverage mask A(T, H):
  1. no_context             : sequence-only (structure & SS8 tracks masked)
  2. homolog_3d             : aligned homolog 3D backbone coords + VQ-VAE tokens on A
  3. homolog_ss8_only       : aligned homolog SS8 track only on A
  4. unrelated_fold_3d      : length- & alpha/beta-matched unrelated CATH fold on A
  5. permuted_homolog_3d    : index-shuffled homolog VQ-VAE structure tokens on A
                              (coords masked so geometric attention cannot leak)
  6. homolog_seq_chainbreak : primary-sequence chainbreak ``homolog|target`` (no 3D)
     vs unrelated_seq_chainbreak : primary-sequence chainbreak ``unrelated|target``

Supports both Route A (ground-truth AFDB x CATH / CATH-S40 3D structures, primary)
and Route B (frozen EXP-R2-228 R4 cohort bridge with ESMFold2-predicted templates).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("TRANSFER_DATA_ROOT") or REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.esm3_ext.esm3_adapter import (  # noqa: E402
    AA20_SET,
    ESM3_MAX_SEQ_LEN,
    encode_ss8_tokens,
    encode_structure_tokens,
    exact_pll,
    load_backbone_from_structure_file,
    load_esm3_offline,
)

IDENTITY_BANDS: tuple[tuple[str, float, float], ...] = (
    ("id_lt_20", 0.0, 0.20),
    ("id_20_30", 0.20, 0.30),
    ("id_30_50", 0.30, 0.50),
    ("id_50_70", 0.50, 0.70),
    ("id_70_90", 0.70, 0.90),
)

CONDITIONS: tuple[str, ...] = (
    "no_context",
    "homolog_3d",
    "homolog_ss8_only",
    "unrelated_fold_3d",
    "permuted_homolog_3d",
    "homolog_seq_chainbreak",
    "unrelated_seq_chainbreak",
)


def _clean_aa20(seq: str) -> str | None:
    s = seq.strip().upper()
    if not s or set(s) - AA20_SET:
        return None
    return s


def align_target_to_template(target_seq: str, template_seq: str) -> dict[str, Any]:
    """Global pairwise alignment (BLOSUM62) returning aligned index pairs ``[(t_idx, h_idx), ...]``."""

    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5

    aln = next(iter(aligner.align(target_seq, template_seq)))
    t_blocks, h_blocks = aln.aligned
    pairs: list[tuple[int, int]] = []
    matches = 0
    for (ts, te), (hs, he) in zip(t_blocks, h_blocks):
        span = min(int(te - ts), int(he - hs))
        for k in range(span):
            ti, hi = int(ts + k), int(hs + k)
            pairs.append((ti, hi))
            if target_seq[ti] == template_seq[hi]:
                matches += 1
    denom = max(1, min(len(target_seq), len(template_seq)))
    identity = float(matches / denom)
    coverage = float(len(pairs) / max(1, len(target_seq)))
    return {
        "aligned_pairs": pairs,
        "identity": identity,
        "coverage": coverage,
        "matches": matches,
    }


def project_template_onto_target(
    target_length: int,
    aligned_pairs: Sequence[tuple[int, int]],
    template_coords: np.ndarray,
    template_ss8: str,
) -> tuple[np.ndarray, str, list[int]]:
    """Project template backbone coords and SS8 onto aligned target positions A(T, H).

    Unaligned target indices remain strictly NaN / '_' (never leaks target structure).
    """

    proj_coords = np.full((target_length, 37, 3), np.nan, dtype=np.float32)
    proj_ss8 = ["_"] * target_length
    aligned_target_indices: list[int] = []
    seen: set[int] = set()

    for ti, hi in aligned_pairs:
        if 0 <= ti < target_length and 0 <= hi < len(template_coords) and ti not in seen:
            seen.add(ti)
            proj_coords[ti] = template_coords[hi]
            if 0 <= hi < len(template_ss8):
                proj_ss8[ti] = template_ss8[hi]
            aligned_target_indices.append(int(ti))

    return proj_coords, "".join(proj_ss8), sorted(aligned_target_indices)


def project_unrelated_onto_mask(
    target_length: int,
    mask_indices: Sequence[int],
    unrelated_coords: np.ndarray,
    unrelated_ss8: str,
) -> tuple[np.ndarray, str]:
    """Project an unrelated fold template onto the exact same target coverage mask A."""

    proj_coords = np.full((target_length, 37, 3), np.nan, dtype=np.float32)
    proj_ss8 = ["_"] * target_length
    u_len = len(unrelated_coords)
    if u_len == 0 or not mask_indices:
        return proj_coords, "".join(proj_ss8)

    # Order-preserving proportional index map so 3D local geometry of unrelated fold stays valid
    n_mask = len(mask_indices)
    for k, ti in enumerate(mask_indices):
        ui = min(u_len - 1, int(round(k * (u_len - 1) / max(1, n_mask - 1)))) if n_mask > 1 else 0
        proj_coords[ti] = unrelated_coords[ui]
        if 0 <= ui < len(unrelated_ss8):
            proj_ss8[ti] = unrelated_ss8[ui]
    return proj_coords, "".join(proj_ss8)


def permute_structure_tokens_on_mask(
    struct_tokens: torch.Tensor,
    mask_indices: Sequence[int],
    *,
    seed: int,
) -> torch.Tensor:
    """Index-shuffle VQ-VAE structure tokens strictly within aligned mask A (offset +1 for BOS)."""

    out = struct_tokens.clone()
    if len(mask_indices) < 2:
        return out
    rng = np.random.default_rng(int(seed))
    idx_tok = [int(i) + 1 for i in mask_indices]
    vals = out[idx_tok].detach().cpu().numpy().copy()
    perm = rng.permutation(len(vals))
    # Guarantee non-identity permutation when >= 2 distinct tokens exist
    if len(set(vals.tolist())) > 1 and np.array_equal(vals[perm], vals):
        perm = np.roll(perm, 1)
    out[idx_tok] = torch.as_tensor(vals[perm], dtype=out.dtype, device=out.device)
    return out


def _ss_helix_sheet_fraction(ss8: str) -> tuple[float, float]:
    n = max(1, len(ss8))
    h = sum(1 for c in ss8 if c in {"H", "G", "I"}) / n
    e = sum(1 for c in ss8 if c in {"E", "B"}) / n
    return (h, e)


def _assign_band(identity: float) -> str | None:
    for name, lo, hi in IDENTITY_BANDS:
        if lo <= identity < hi:
            return name
    return None


def build_afdb_cath_route_a_cohort(
    *,
    alphafold_dir: Path,
    cath_tsv: Path,
    max_per_band: int = 150,
    min_len: int = 50,
    max_len: int = 400,
    seed: int = 20261120,
) -> list[dict[str, Any]]:
    """Build a leakage-free Route A ground-truth 3D pair cohort from AFDB x CATH superfamilies."""

    rng = np.random.default_rng(seed)
    sf_to_accs: dict[str, list[str]] = defaultdict(list)
    with cath_tsv.open("r", encoding="utf-8") as fh:
        header = fh.readline()
        _ = header
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sf_to_accs[parts[1]].append(parts[0])

    # Keep superfamilies with >= 2 local AFDB structures
    sf_files: dict[str, list[tuple[str, Path]]] = {}
    for sf, accs in sorted(sf_to_accs.items()):
        found: list[tuple[str, Path]] = []
        for acc in accs:
            pdb_gz = alphafold_dir / f"AF-{acc}-F1-model_v6.pdb.gz"
            cif_gz = alphafold_dir / f"AF-{acc}-F1-model_v6.cif.gz"
            if pdb_gz.is_file():
                found.append((acc, pdb_gz))
            elif cif_gz.is_file():
                found.append((acc, cif_gz))
        if len(found) >= 2:
            sf_files[sf] = found

    # Parse candidate structures up to a reasonable working pool
    parsed_by_sf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_parsed: list[dict[str, Any]] = []
    sf_keys = list(sf_files.keys())
    rng.shuffle(sf_keys)

    for sf in sf_keys:
        members = list(sf_files[sf])
        rng.shuffle(members)
        for acc, path in members[:8]:
            try:
                rec = load_backbone_from_structure_file(path)
            except Exception:
                continue
            seq = _clean_aa20(rec["sequence"])
            if seq is None or not (min_len <= len(seq) <= max_len):
                continue
            h_frac, e_frac = _ss_helix_sheet_fraction(rec["ss8"])
            entry = {
                "accession": acc,
                "superfamily": sf,
                "cath_class": sf.split(".")[0],
                "cath_topology": ".".join(sf.split(".")[:3]),
                "path": str(path),
                "sequence": seq,
                "length": len(seq),
                "ss8": rec["ss8"],
                "helix_frac": h_frac,
                "sheet_frac": e_frac,
            }
            parsed_by_sf[sf].append(entry)
            all_parsed.append(entry)
        if len(all_parsed) >= max(400, max_per_band * 12):
            break

    band_counts: dict[str, int] = {b[0]: 0 for b in IDENTITY_BANDS}
    units: list[dict[str, Any]] = []
    used_targets: set[str] = set()

    for sf, members in parsed_by_sf.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            if all(band_counts[b[0]] >= max_per_band for b in IDENTITY_BANDS):
                break
            tgt = members[i]
            if tgt["accession"] in used_targets:
                continue
            for j in range(len(members)):
                if i == j:
                    continue
                hom = members[j]
                if abs(tgt["length"] - hom["length"]) > 0.5 * max(tgt["length"], hom["length"]):
                    continue
                aln = align_target_to_template(tgt["sequence"], hom["sequence"])
                band = _assign_band(aln["identity"])
                if band is None or band_counts[band] >= max_per_band or aln["coverage"] < 0.5:
                    continue

                # Select length- & alpha/beta-matched unrelated CATH topology template
                best_unrel = None
                best_dist = float("inf")
                for cand in all_parsed:
                    if cand["cath_topology"] == tgt["cath_topology"]:
                        continue
                    len_diff = abs(cand["length"] - tgt["length"]) / max(1, tgt["length"])
                    ss_diff = abs(cand["helix_frac"] - tgt["helix_frac"]) + abs(
                        cand["sheet_frac"] - tgt["sheet_frac"]
                    )
                    dist = len_diff + ss_diff
                    if dist < best_dist:
                        best_dist = dist
                        best_unrel = cand
                if best_unrel is None:
                    continue

                used_targets.add(tgt["accession"])
                band_counts[band] += 1
                units.append(
                    {
                        "unit_id": f"r4_cath_{band}_{tgt['accession']}",
                        "route": "ground_truth_3d",
                        "band": band,
                        "group": sf,
                        "target_id": tgt["accession"],
                        "target_sequence": tgt["sequence"],
                        "target_structure_path": tgt["path"],
                        "homolog_id": hom["accession"],
                        "homolog_sequence": hom["sequence"],
                        "homolog_structure_path": hom["path"],
                        "unrelated_id": best_unrel["accession"],
                        "unrelated_sequence": best_unrel["sequence"],
                        "unrelated_structure_path": best_unrel["path"],
                        "pairwise_identity": aln["identity"],
                        "alignment_coverage": aln["coverage"],
                        "aligned_pairs": aln["aligned_pairs"],
                    }
                )
                break

    return units


def load_route_b_bridge_units(
    cohort_path: Path,
    structures_dir: Path | None,
    *,
    max_per_band: int | None = None,
) -> list[dict[str, Any]]:
    """Load frozen EXP-R2-228 R4 protein units as Route B (ESMFold2-predicted template bridge)."""

    raw = json.loads(cohort_path.read_text(encoding="utf-8"))
    records: list[str] = raw["protein"]["records"]
    raw_units: list[dict[str, Any]] = raw["protein"]["units"]

    # Build lookup of all retained units to pick length-matched unrelated sequences from distinct groups
    by_group: dict[int, list[int]] = defaultdict(list)
    for u in raw_units:
        if u.get("stratum") == "retained":
            by_group[int(u["group"])].append(int(u["target"]))
    all_targets = [
        (int(u["group"]), int(u["target"]), len(records[int(u["target"])]))
        for u in raw_units
        if u.get("stratum") == "retained"
    ]

    band_counts: dict[str, int] = defaultdict(int)
    units: list[dict[str, Any]] = []
    for u in raw_units:
        if u.get("stratum") != "retained":
            continue
        band = str(u["band"])
        if max_per_band is not None and band_counts[band] >= max_per_band:
            continue
        t_idx = int(u["target"])
        partners = [int(p) for p in u.get("partners", [])]
        if not partners:
            continue
        h_idx = partners[0]
        t_seq = _clean_aa20(records[t_idx])
        h_seq = _clean_aa20(records[h_idx])
        if t_seq is None or h_seq is None:
            continue

        # Nearest-length unrelated sequence from a different group
        grp = int(u["group"])
        u_idx = min(
            (x for x in all_targets if x[0] != grp),
            key=lambda x: abs(x[2] - len(t_seq)),
        )[1]
        u_seq = _clean_aa20(records[u_idx])
        if u_seq is None:
            continue

        def _find_struct(idx: int) -> str | None:
            if structures_dir is None:
                return None
            for ext in (".pdb", ".pdb.gz", ".cif", ".cif.gz"):
                for stem in (f"rec_{idx:05d}", f"{idx}", f"protein_{idx:05d}"):
                    cand = structures_dir / f"{stem}{ext}"
                    if cand.is_file():
                        return str(cand)
            return None

        aln = align_target_to_template(t_seq, h_seq)
        band_counts[band] += 1
        units.append(
            {
                "unit_id": str(u["key"]),
                "route": "esmfold2_bridge",
                "band": band,
                "group": f"grp_{grp}",
                "target_id": f"rec_{t_idx:05d}",
                "target_sequence": t_seq,
                "target_structure_path": _find_struct(t_idx),
                "homolog_id": f"rec_{h_idx:05d}",
                "homolog_sequence": h_seq,
                "homolog_structure_path": _find_struct(h_idx),
                "unrelated_id": f"rec_{u_idx:05d}",
                "unrelated_sequence": u_seq,
                "unrelated_structure_path": _find_struct(u_idx),
                "pairwise_identity": aln["identity"],
                "alignment_coverage": aln["coverage"],
                "aligned_pairs": aln["aligned_pairs"],
            }
        )
    return units


def paired_auroc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """Paired dimensionless AUROC where lower NLL = better (``pos_scores < neg_scores``)."""

    if len(pos_scores) != len(neg_scores) or not pos_scores:
        return float("nan")
    wins = 0.0
    for p, n in zip(pos_scores, neg_scores):
        if p < n:
            wins += 1.0
        elif p == n:
            wins += 0.5
    return float(wins / len(pos_scores))


def cluster_bootstrap_ci(
    groups: Sequence[str],
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 20261120,
) -> dict[str, float]:
    """Cluster-bootstrap 95% CI resampling independent family/superfamily groups."""

    if not values:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_groups": 0}
    by_g: dict[str, list[float]] = defaultdict(list)
    for g, v in zip(groups, values):
        if math.isfinite(v):
            by_g[str(g)].append(float(v))
    if not by_g:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_groups": 0}
    g_means = np.array([float(np.mean(vs)) for vs in by_g.values()], dtype=np.float64)
    n_g = len(g_means)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_g, size=(n_boot, n_g))
    boots = g_means[idx].mean(axis=1)
    return {
        "mean": float(g_means.mean()),
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
        "n_groups": n_g,
    }


def _select_eval_positions(length: int, max_sites: int) -> list[int]:
    if max_sites <= 0 or max_sites >= length:
        return list(range(length))
    # Deterministic evenly spaced sites across the target sequence
    return sorted({int(round(i * (length - 1) / max(1, max_sites - 1))) for i in range(max_sites)})


def _full_masked_nll(
    handle: Any,
    sequence: str,
    eval_positions: Sequence[int],
    *,
    allow_chainbreak: bool = False,
    mask_span: tuple[int, int] | None = None,
    **track_kwargs: Any,
) -> float:
    """1-pass full-target-masked conditional NLL (p=1.0, zero target-sequence leakage)."""

    import torch.nn.functional as F
    from src.capability.esm3_ext.esm3_adapter import forward_tracks, tokenize_sequence_ids

    true_ids = tokenize_sequence_ids(handle, sequence, allow_chainbreak=allow_chainbreak)
    masked = true_ids.clone()
    if mask_span is None:
        masked[1:-1] = handle.mask_id
    else:
        masked[1 + int(mask_span[0]) : 1 + int(mask_span[1])] = handle.mask_id
    logits = forward_tracks(handle, sequence_tokens=masked.unsqueeze(0), **track_kwargs).sequence_logits.float()[0]
    lp = F.log_softmax(logits, dim=-1)
    terms = [float(-lp[int(p) + 1, int(true_ids[int(p) + 1])]) for p in eval_positions]
    return float(sum(terms) / max(1, len(terms)))


def _score_chainbreak_condition(
    handle: Any,
    context_seq: str,
    target_seq: str,
    eval_positions_target: Sequence[int],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Score target span in ``<cls> context | target <eos>`` (all structure tracks masked)."""

    max_ctx = ESM3_MAX_SEQ_LEN - len(target_seq) - 1
    if max_ctx < 1:
        raise ValueError(f"Target length {len(target_seq)} leaves no room for chainbreak context")
    ctx = context_seq[:max_ctx]
    combined = f"{ctx}|{target_seq}"
    offset = len(ctx) + 1
    shifted_positions = [offset + int(p) for p in eval_positions_target]
    chain_id = torch.zeros((1, len(combined) + 2), dtype=torch.long, device=handle.device)
    chain_id[0, 1 + offset : -1] = 1
    res = exact_pll(
        handle,
        combined,
        batch_size=batch_size,
        positions=shifted_positions,
        allow_chainbreak=True,
        chain_id=chain_id,
    )
    res["full_masked_nll_nats"] = _full_masked_nll(
        handle,
        combined,
        shifted_positions,
        allow_chainbreak=True,
        mask_span=(offset, offset + len(target_seq)),
        chain_id=chain_id,
    )
    return res


def evaluate_unit(
    handle: Any,
    unit: dict[str, Any],
    *,
    batch_size: int,
    max_sites_per_target: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate all 6 paired conditions (7 runs incl. chainbreak controls) on one unit.

    Reports BOTH ``mean_nll_nats`` (exact leave-one-out PLL, p=1/L) and
    ``full_masked_nll_nats`` (p=1.0 target-masked conditional NLL, zero target-sequence
    leakage). Template VQ-VAE tokens are encoded on the INTACT template structure first
    before projecting onto aligned target mask A(T, H) so k-NN neighborhoods are never
    distorted by alignment gaps.
    """

    t_seq = unit["target_sequence"]
    h_seq = unit["homolog_sequence"]
    u_seq = unit["unrelated_sequence"]
    L = len(t_seq)
    eval_pos = _select_eval_positions(L, max_sites_per_target)

    # 1. no_context (sequence-only)
    res_no_ctx = exact_pll(handle, t_seq, batch_size=batch_size, positions=eval_pos)
    scores: dict[str, float] = {"no_context": float(res_no_ctx["mean_nll_per_residue_nats"])}
    aa20_scores: dict[str, float] = {
        "no_context": float(res_no_ctx["aa20_mean_nll_per_residue_nats"])
    }
    full_masked: dict[str, float] = {
        "no_context": _full_masked_nll(handle, t_seq, eval_pos)
    }

    # 2-5. Multimodal structure / SS8 conditions (when template structures exist)
    h_path = unit.get("homolog_structure_path")
    u_path = unit.get("unrelated_structure_path")
    has_3d = bool(h_path and Path(h_path).is_file() and u_path and Path(u_path).is_file())
    aligned_mask_len = 0

    if has_3d:
        h_rec = load_backbone_from_structure_file(h_path)
        u_rec = load_backbone_from_structure_file(u_path)
        aligned_pairs = unit.get("aligned_pairs") or align_target_to_template(
            t_seq, h_rec["sequence"]
        )["aligned_pairs"]

        proj_h_coords, proj_h_ss8, mask_A = project_template_onto_target(
            L, aligned_pairs, h_rec["coords_atom37"], h_rec["ss8"]
        )
        aligned_mask_len = len(mask_A)
        proj_u_coords, _ = project_unrelated_onto_mask(
            L, mask_A, u_rec["coords_atom37"], u_rec["ss8"]
        )

        # Encode INTACT templates via local VQ-VAE first, then project onto mask A
        _, h_intact_toks = encode_structure_tokens(
            handle, h_rec["coords_atom37"], reference_sequence=h_rec["sequence"]
        )
        _, u_intact_toks = encode_structure_tokens(
            handle, u_rec["coords_atom37"], reference_sequence=u_rec["sequence"]
        )

        h_struct_toks = torch.full(
            (L + 2,), handle.structure_mask_id, dtype=torch.long, device=handle.device
        )
        h_struct_toks[0] = handle.structure_bos_id
        h_struct_toks[-1] = handle.structure_eos_id
        for ti, hi in aligned_pairs:
            if 0 <= ti < L and 0 <= hi < len(h_rec["coords_atom37"]):
                h_struct_toks[ti + 1] = h_intact_toks[hi + 1]

        u_struct_toks = torch.full(
            (L + 2,), handle.structure_mask_id, dtype=torch.long, device=handle.device
        )
        u_struct_toks[0] = handle.structure_bos_id
        u_struct_toks[-1] = handle.structure_eos_id
        u_len = len(u_rec["coords_atom37"])
        for k_i, ti in enumerate(mask_A):
            ui = (
                min(u_len - 1, int(round(k_i * (u_len - 1) / max(1, len(mask_A) - 1))))
                if len(mask_A) > 1
                else 0
            )
            u_struct_toks[ti + 1] = u_intact_toks[ui + 1]

        h_coords_pad = torch.full((L + 2, 3, 3), float("nan"), dtype=torch.float32, device=handle.device)
        h_coords_pad[1:-1] = torch.as_tensor(proj_h_coords[:, :3, :], dtype=torch.float32, device=handle.device)
        u_coords_pad = torch.full((L + 2, 3, 3), float("nan"), dtype=torch.float32, device=handle.device)
        u_coords_pad[1:-1] = torch.as_tensor(proj_u_coords[:, :3, :], dtype=torch.float32, device=handle.device)

        h_ss8_toks = encode_ss8_tokens(handle, proj_h_ss8)
        perm_h_struct_toks = permute_structure_tokens_on_mask(
            h_struct_toks, mask_A, seed=seed
        )

        cond_kwargs = (
            ("homolog_3d", {"structure_tokens": h_struct_toks, "structure_coords": h_coords_pad}),
            ("homolog_ss8_only", {"ss8_tokens": h_ss8_toks}),
            ("unrelated_fold_3d", {"structure_tokens": u_struct_toks, "structure_coords": u_coords_pad}),
            ("permuted_homolog_3d", {"structure_tokens": perm_h_struct_toks}),
        )
        for cname, kw in cond_kwargs:
            r = exact_pll(handle, t_seq, batch_size=batch_size, positions=eval_pos, **kw)
            scores[cname] = float(r["mean_nll_per_residue_nats"])
            aa20_scores[cname] = float(r["aa20_mean_nll_per_residue_nats"])
            full_masked[cname] = _full_masked_nll(handle, t_seq, eval_pos, **kw)

    # 6. Sequence-only chainbreak controls in the identical 1.4B weights
    res_hseq = _score_chainbreak_condition(
        handle, h_seq, t_seq, eval_pos, batch_size=batch_size
    )
    res_useq = _score_chainbreak_condition(
        handle, u_seq, t_seq, eval_pos, batch_size=batch_size
    )
    scores["homolog_seq_chainbreak"] = float(res_hseq["mean_nll_per_residue_nats"])
    scores["unrelated_seq_chainbreak"] = float(res_useq["mean_nll_per_residue_nats"])
    aa20_scores["homolog_seq_chainbreak"] = float(res_hseq["aa20_mean_nll_per_residue_nats"])
    aa20_scores["unrelated_seq_chainbreak"] = float(res_useq["aa20_mean_nll_per_residue_nats"])
    full_masked["homolog_seq_chainbreak"] = float(res_hseq["full_masked_nll_nats"])
    full_masked["unrelated_seq_chainbreak"] = float(res_useq["full_masked_nll_nats"])

    base = scores["no_context"]
    frac_reduction = {
        c: float((base - v) / max(1e-12, abs(base)))
        for c, v in scores.items()
        if c != "no_context"
    }
    fm_base = full_masked["no_context"]
    fm_frac_reduction = {
        c: float((fm_base - v) / max(1e-12, abs(fm_base)))
        for c, v in full_masked.items()
        if c != "no_context"
    }

    return {
        "unit_id": unit["unit_id"],
        "route": unit.get("route", "unknown"),
        "band": unit["band"],
        "group": unit["group"],
        "target_id": unit["target_id"],
        "homolog_id": unit["homolog_id"],
        "unrelated_id": unit["unrelated_id"],
        "target_length": L,
        "scored_sites": len(eval_pos),
        "has_3d_templates": has_3d,
        "aligned_mask_residues": aligned_mask_len,
        "pairwise_identity": float(unit.get("pairwise_identity", float("nan"))),
        "mean_nll_nats": scores,
        "aa20_mean_nll_nats": aa20_scores,
        "full_masked_nll_nats": full_masked,
        "fractional_pll_reduction": frac_reduction,
        "full_masked_fractional_reduction": fm_frac_reduction,
    }


def summarize_results(rows: Sequence[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    """Compute per-band paired AUROC and cluster-bootstrapped fractional PLL reductions."""

    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_band[str(r["band"])].append(r)

    summary: dict[str, Any] = {}
    for band, band_rows in sorted(by_band.items()):
        entry: dict[str, Any] = {
            "n_units": len(band_rows),
            "n_groups": len({r["group"] for r in band_rows}),
            "n_with_3d": sum(1 for r in band_rows if r["has_3d_templates"]),
            "paired_auroc": {},
            "full_masked_paired_auroc": {},
            "fractional_pll_reduction": {},
            "full_masked_fractional_reduction": {},
            "mean_nll_nats": {},
            "full_masked_mean_nll_nats": {},
        }
        contrasts = [
            ("homolog_3d_vs_unrelated_fold_3d", "homolog_3d", "unrelated_fold_3d"),
            ("homolog_3d_vs_permuted_homolog_3d", "homolog_3d", "permuted_homolog_3d"),
            ("homolog_3d_vs_no_context", "homolog_3d", "no_context"),
            ("homolog_ss8_vs_unrelated_fold_3d", "homolog_ss8_only", "unrelated_fold_3d"),
            ("homolog_3d_vs_homolog_ss8_only", "homolog_3d", "homolog_ss8_only"),
            (
                "homolog_seq_chainbreak_vs_unrelated_seq_chainbreak",
                "homolog_seq_chainbreak",
                "unrelated_seq_chainbreak",
            ),
            ("homolog_3d_vs_homolog_seq_chainbreak", "homolog_3d", "homolog_seq_chainbreak"),
        ]
        for label, pos_c, neg_c in contrasts:
            for field_key, target_dict in (
                ("mean_nll_nats", entry["paired_auroc"]),
                ("full_masked_nll_nats", entry["full_masked_paired_auroc"]),
            ):
                sub = [
                    r
                    for r in band_rows
                    if field_key in r and pos_c in r[field_key] and neg_c in r[field_key]
                ]
                if not sub:
                    continue
                wins = [
                    1.0
                    if r[field_key][pos_c] < r[field_key][neg_c]
                    else (0.5 if r[field_key][pos_c] == r[field_key][neg_c] else 0.0)
                    for r in sub
                ]
                target_dict[label] = {
                    "point": paired_auroc(
                        [r[field_key][pos_c] for r in sub],
                        [r[field_key][neg_c] for r in sub],
                    ),
                    "cluster_bootstrap": cluster_bootstrap_ci(
                        [r["group"] for r in sub], wins, seed=seed
                    ),
                    "n_units": len(sub),
                }

        for cond in CONDITIONS:
            sub = [r for r in band_rows if cond in r["mean_nll_nats"]]
            if sub:
                entry["mean_nll_nats"][cond] = float(np.mean([r["mean_nll_nats"][cond] for r in sub]))
                if cond != "no_context":
                    entry["fractional_pll_reduction"][cond] = cluster_bootstrap_ci(
                        [r["group"] for r in sub],
                        [r["fractional_pll_reduction"][cond] for r in sub],
                        seed=seed,
                    )
            sub_fm = [r for r in band_rows if "full_masked_nll_nats" in r and cond in r["full_masked_nll_nats"]]
            if sub_fm:
                entry["full_masked_mean_nll_nats"][cond] = float(
                    np.mean([r["full_masked_nll_nats"][cond] for r in sub_fm])
                )
                if cond != "no_context":
                    entry["full_masked_fractional_reduction"][cond] = cluster_bootstrap_ci(
                        [r["group"] for r in sub_fm],
                        [r["full_masked_fractional_reduction"][cond] for r in sub_fm],
                        seed=seed,
                    )
        summary[band] = entry
    return summary


def _write_json_with_sha256(path: Path, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=None, help="Local esm3-sm-open-v1 dir")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument(
        "--route",
        choices=("ground_truth_3d", "esmfold2_bridge"),
        default="ground_truth_3d",
    )
    p.add_argument("--cohort", type=Path, default=None, help="Prepared Route A JSON or Route B cohort.json")
    p.add_argument("--alphafold-dir", type=Path, default=DATA_ROOT / "data/alphafold")
    p.add_argument("--cath-tsv", type=Path, default=DATA_ROOT / "data/interpro/cath_superfamily.tsv")
    p.add_argument("--structures-dir", type=Path, default=None, help="Predicted PDBs for Route B bridge")
    p.add_argument("--max-per-band", type=int, default=150)
    p.add_argument("--max-sites-per-target", type=int, default=0, help="0 = exact full-L PLL")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="Optional cap on units for smoke test")
    p.add_argument("--seed", type=int, default=20261120)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load or construct cohort
    if args.route == "ground_truth_3d":
        if args.cohort is not None and args.cohort.is_file():
            raw_c = json.loads(args.cohort.read_text(encoding="utf-8"))
            units = raw_c["units"] if isinstance(raw_c, dict) and "units" in raw_c else raw_c
        else:
            units = build_afdb_cath_route_a_cohort(
                alphafold_dir=args.alphafold_dir,
                cath_tsv=args.cath_tsv,
                max_per_band=args.max_per_band,
                seed=args.seed,
            )
            _write_json_with_sha256(
                args.out / "route_a_cohort_built.json",
                {"schema": "esm3_r4_route_a_cohort_v1", "n_units": len(units), "units": units},
            )
    else:
        cohort_path = args.cohort or (DATA_ROOT / "results/R5/context_homologue_20260826/cohort.json")
        units = load_route_b_bridge_units(
            cohort_path, args.structures_dir, max_per_band=args.max_per_band
        )

    # Shard and optional limit
    units = [u for i, u in enumerate(units) if i % args.num_shards == args.shard_index]
    if args.limit > 0:
        units = units[: args.limit]

    handle = load_esm3_offline(args.model, device=args.device, dtype=args.dtype)
    rows: list[dict[str, Any]] = []
    per_target_path = args.out / "per_target_scores.jsonl"
    with per_target_path.open("w", encoding="utf-8") as out_fh:
        for idx, unit in enumerate(units):
            rec = evaluate_unit(
                handle,
                unit,
                batch_size=args.batch_size,
                max_sites_per_target=args.max_sites_per_target,
                seed=args.seed + idx,
            )
            rows.append(rec)
            out_fh.write(json.dumps(rec, sort_keys=True) + "\n")
            out_fh.flush()

    payload = {
        "schema": "esm3_r4_multimodal_v1",
        "route": args.route,
        "model_facts": handle.facts,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_units_evaluated": len(rows),
        "elapsed_seconds": round(time.time() - t0, 2),
        "summary_by_band": summarize_results(rows, seed=args.seed),
    }
    _write_json_with_sha256(args.out / "esm3_r4_multimodal.json", payload)


if __name__ == "__main__":
    main()
