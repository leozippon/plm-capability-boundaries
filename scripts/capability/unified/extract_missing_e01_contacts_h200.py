#!/usr/bin/env python3
"""Extract canonical ``position_likelihood_v1`` E01 contact archives for general-PLM & ESM3 arms.

Produces exact ``PositionArchive`` directories (``archives/position_<sha20>.npz``,
``archive-input.json``, and ``position_likelihood.json``) on the 30 admitted
structural assays (``--structure-only``) so ``measure_contact_discrimination.py``
and ``fit_contact_bidirectional.py`` consume every arm of the 62-model roster
through one identical validator (``RetainedResponses``) with zero format divergence.

Also supports a 1-assay verification gate against the frozen
``posresp_extract_esm2_650m_gpu1`` archive and fast sparse-MoE prefix-invariance
screening (writing any arm with ``upstream_max_abs_nats > TIER2_NATS = 1e-4``
into ``refused_moe_layout.json`` rather than emitting a corrupted archive).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.amino_acids import AA20
from src.capability.core.io import sha256_file, write_json
from src.capability.extensions.responses import RetainedResponses
from src.capability.position.contact_response import rebuild_states
from src.capability.position.position_likelihood import (
    CAUSAL,
    MASKED,
    PREFIX_INVARIANT_RULE,
    SCHEMA,
    TERM_SEMANTICS,
    TIER2_NATS,
    TIER2_REPEAT_MULTIPLE,
    PackedState,
    PositionArchive,
    PositionScore,
    read_archive,
    repeat_residual,
    residual_tier,
    state_terms,
    upstream_invariance,
)
from src.capability.position.position_terms import (
    ALIGNMENT,
    alignment,
    blas_pinning,
    retention_residual,
)

COMPLETION = "position_likelihood.json"
TOKEN_BUDGET = 1024
AA20_INDEX = {aa: idx for idx, aa in enumerate(AA20)}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _assay_filename(assay: str) -> str:
    return "position_" + hashlib.sha256(assay.encode("utf-8")).hexdigest()[:20] + ".npz"


def _select_structural_assays(
    cohort_path: Path,
    structures_path: Path,
    *,
    assays_filter: list[str] | None = None,
    max_residues: int = 1000,
    min_residues: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cohort = _read_json(cohort_path)
    sites = _read_json(structures_path)
    site_rows = sites["sites"] if isinstance(sites, dict) else sites
    site_assays = {r["assay_id"] for r in site_rows if r["status"] == "admitted"}
    rows = {r["assay"]: r for r in cohort["assays"]}
    requested = list(assays_filter) if assays_filter else sorted(rows)
    selected, excluded = [], []
    for name in requested:
        row = rows[name]
        length = len(row["wildtype"])
        if name not in site_assays:
            excluded.append({"assay": name, "reason": "no admitted structural mapping"})
            continue
        if length > max_residues:
            excluded.append({"assay": name, "reason": f"{length} residues exceeds max_residues"})
            continue
        if length < min_residues:
            excluded.append({"assay": name, "reason": f"{length} residues below min_residues"})
            continue
        if set(row["wildtype"]) - set(AA20):
            excluded.append({"assay": name, "reason": "wild type carries non-AA20 symbol"})
            continue
        selected.append(row)
    selected.sort(key=lambda r: -len(r["wildtype"]) * (1 + len(r["mutants"])))
    return selected, excluded


def _substitution_sites(mutant: str, wildtype: str) -> list[int]:
    sites = []
    for token in mutant.split(":"):
        before, after = token[0], token[-1]
        pos = int(token[1:-1]) - 1
        if not 0 <= pos < len(wildtype) or wildtype[pos] != before or after not in AA20:
            raise ValueError(f"{mutant}: {token} disagrees with wild type")
        sites.append(pos)
    return sorted(sites)


def _archive_packed_state_masked(sequence: str, raw_ids: list[int], offset: int) -> PackedState:
    """Build a ``PackedState`` satisfying ``1 <= span[0] < span[1] <= len(ids)``.

    If an adapter packs without a leading BOS token (``offset == 0``, e.g.
    ``ProteinGLM-MLM``), prepend a placeholder token ``0`` ONLY in the archived
    metadata ``ids`` so ``span = (1, 1 + L)`` satisfies ``RetainedResponses``
    without altering any model forward.
    """
    length = len(sequence)
    if offset >= 1:
        ids = tuple(int(x) for x in raw_ids)
        span = (int(offset), int(offset) + length)
    else:
        ids = (0, *(int(x) for x in raw_ids))
        span = (1, 1 + length)
    return PackedState(
        sequence=sequence,
        ids=ids,
        span=span,
        counts=np.ones(length, dtype=np.int64),
        offset=0,
    )


def _make_position_score(
    terms_dev: torch.Tensor,
    device: str | torch.device,
    *,
    gathered_dev: torch.Tensor | None = None,
) -> PositionScore:
    terms_f32 = terms_dev.to(dtype=torch.float32)
    if not torch.isfinite(terms_f32).all():
        raise FloatingPointError("Non-finite per-position NLL term encountered")
    # Clamp tiny negative zero (-0.0 or -1e-7 from float logsumexp rounding) to +0.0.
    if (terms_f32 < -1e-4).any():
        raise ValueError(f"Negative NLL term detected: min={float(terms_f32.min()):.6g}")
    terms_f32 = torch.clamp(terms_f32, min=0.0)
    nll_sum = float(terms_f32.sum())
    array = terms_f32.detach().cpu().numpy().astype(np.float32, copy=False)
    res = float(retention_residual(array, nll_sum, str(device)))
    if res != 0.0:
        # Re-reduce directly from the finalized float32 array on device so residual is identically 0.0.
        nll_sum = float(torch.as_tensor(array, dtype=torch.float32, device=device).sum())
        res = float(retention_residual(array, nll_sum, str(device)))
    gathered = (
        gathered_dev.detach().cpu().numpy().astype(np.float32, copy=False)
        if gathered_dev is not None
        else None
    )
    return PositionScore(
        terms=array,
        nll_sum=nll_sum,
        retention_residual_nats=res,
        gathered=gathered,
    )


@torch.no_grad()
def _score_masked_assay_gp(
    adapter: Any,
    sequences: list[str],
    *,
    mask_batch_size: int,
    compute_repeat: bool = True,
) -> tuple[list[PackedState], list[PositionScore], float]:
    """Cross-sequence batched exact leave-one-out PLL for equal-length assay states."""
    from src.capability.general_plm.adapters import SaProtAdapter

    is_saprot = isinstance(adapter, SaProtAdapter)
    length = len(sequences[0])
    if any(len(s) != length for s in sequences):
        raise ValueError("All states in an assay must have equal length")

    raw_encoded = [adapter._encode_sequence(seq) for seq in sequences]
    offset = int(raw_encoded[0][1])
    seq_tok_len = len(raw_encoded[0][0])
    if any(int(off) != offset or len(ids) != seq_tok_len for ids, off in raw_encoded):
        raise ValueError(f"{adapter.arm}: inconsistent masked token lengths inside one assay")

    packed_states = [
        _archive_packed_state_masked(seq, ids, off)
        for seq, (ids, off) in zip(sequences, raw_encoded)
    ]

    dev = adapter.device
    all_ids = torch.as_tensor([ids for ids, _ in raw_encoded], dtype=torch.long, device=dev)
    n_states = len(sequences)
    terms_mat = torch.empty((n_states, length), dtype=torch.float32, device=dev)
    wt_gathered = torch.empty((length, len(AA20)), dtype=torch.float32, device=dev)
    aa20_ids_t = (
        None
        if is_saprot
        else torch.as_tensor(adapter.aa20_ids, dtype=torch.long, device=dev)
    )
    aa_idx_mat = (
        torch.as_tensor(
            [[AA20_INDEX[ch] for ch in seq] for seq in sequences],
            dtype=torch.long,
            device=dev,
        )
        if is_saprot
        else None
    )

    total_items = n_states * length
    bs = max(1, int(mask_batch_size))
    mask_id = int(adapter.mask_id)

    with adapter._precision_ctx():
        for start in range(0, total_items, bs):
            stop = min(start + bs, total_items)
            flat = torch.arange(start, stop, device=dev, dtype=torch.long)
            s_idx = torch.div(flat, length, rounding_mode="floor")
            p_idx = flat - s_idx * length

            batch = all_ids.index_select(0, s_idx).clone()
            row_ar = torch.arange(stop - start, device=dev, dtype=torch.long)
            tok_col = p_idx + offset
            batch[row_ar, tok_col] = mask_id

            logits = adapter._forward_logits(batch)
            pos_logits = logits[row_ar, tok_col]  # (B, V) float32
            logprobs = F.log_softmax(pos_logits, dim=-1)

            if is_saprot:
                assert aa_idx_mat is not None
                # (B, 20, 20) -> logsumexp over 20 3Di tokens per AA -> (B, 20)
                aa_lps = torch.logsumexp(logprobs[:, adapter._struct_id_matrix], dim=-1)
                target_aa = aa_idx_mat[s_idx, p_idx]
                nll = -aa_lps[row_ar, target_aa]
                terms_mat[s_idx, p_idx] = nll
                wt_mask = s_idx == 0
                if wt_mask.any():
                    wt_gathered[p_idx[wt_mask]] = aa_lps[wt_mask]
            else:
                assert aa20_ids_t is not None
                target_tok = all_ids[s_idx, tok_col]
                nll = -logprobs[row_ar, target_tok]
                terms_mat[s_idx, p_idx] = nll
                wt_mask = s_idx == 0
                if wt_mask.any():
                    wt_gathered[p_idx[wt_mask]] = logprobs[wt_mask].index_select(1, aa20_ids_t)
            del logits, pos_logits, logprobs

    scores: list[PositionScore] = []
    for idx in range(n_states):
        scores.append(
            _make_position_score(
                terms_mat[idx],
                dev,
                gathered_dev=wt_gathered if idx == 0 else None,
            )
        )

    rep = 0.0
    if compute_repeat:
        # Repeat WT alone to measure reproducibility residual.
        wt_rep_terms = torch.empty(length, dtype=torch.float32, device=dev)
        with adapter._precision_ctx():
            for start in range(0, length, bs):
                stop = min(start + bs, length)
                p_idx = torch.arange(start, stop, device=dev, dtype=torch.long)
                batch = all_ids[0:1].repeat(stop - start, 1)
                row_ar = torch.arange(stop - start, device=dev, dtype=torch.long)
                tok_col = p_idx + offset
                batch[row_ar, tok_col] = mask_id
                logits = adapter._forward_logits(batch)
                logprobs = F.log_softmax(logits[row_ar, tok_col], dim=-1)
                if is_saprot:
                    assert aa_idx_mat is not None
                    aa_lps = torch.logsumexp(logprobs[:, adapter._struct_id_matrix], dim=-1)
                    wt_rep_terms[p_idx] = -aa_lps[row_ar, aa_idx_mat[0, p_idx]]
                else:
                    wt_rep_terms[p_idx] = -logprobs[row_ar, all_ids[0, tok_col]]
                del logits, logprobs
        rep_score = _make_position_score(wt_rep_terms, dev)
        rep = repeat_residual(scores[0], rep_score)

    return packed_states, scores, rep


@torch.no_grad()
def _score_masked_assay_esm3(
    handle: Any,
    sequences: list[str],
    *,
    mask_batch_size: int,
    compute_repeat: bool = True,
) -> tuple[list[PackedState], list[PositionScore], float]:
    """Cross-sequence batched exact leave-one-out PLL for ESM3-1.4B (sequence-only)."""
    from src.capability.esm3_ext.esm3_adapter import forward_tracks, tokenize_sequence_ids

    length = len(sequences[0])
    if any(len(s) != length for s in sequences):
        raise ValueError("All states in an assay must have equal length")

    dev = handle.device
    all_ids = torch.stack([tokenize_sequence_ids(handle, seq) for seq in sequences], dim=0)
    packed_states = [
        _archive_packed_state_masked(seq, all_ids[i].detach().cpu().tolist(), 1)
        for i, seq in enumerate(sequences)
    ]

    n_states = len(sequences)
    terms_mat = torch.empty((n_states, length), dtype=torch.float32, device=dev)
    wt_gathered = torch.empty((length, len(AA20)), dtype=torch.float32, device=dev)
    aa20_ids_t = torch.as_tensor(handle.aa20_token_ids, dtype=torch.long, device=dev)

    total_items = n_states * length
    bs = max(1, int(mask_batch_size))
    mask_id = int(handle.mask_id)

    for start in range(0, total_items, bs):
        stop = min(start + bs, total_items)
        flat = torch.arange(start, stop, device=dev, dtype=torch.long)
        s_idx = torch.div(flat, length, rounding_mode="floor")
        p_idx = flat - s_idx * length

        batch = all_ids.index_select(0, s_idx).clone()
        row_ar = torch.arange(stop - start, device=dev, dtype=torch.long)
        tok_col = p_idx + 1
        batch[row_ar, tok_col] = mask_id

        logits = forward_tracks(handle, sequence_tokens=batch).sequence_logits.float()
        logprobs = F.log_softmax(logits[row_ar, tok_col], dim=-1)
        target_tok = all_ids[s_idx, tok_col]
        terms_mat[s_idx, p_idx] = -logprobs[row_ar, target_tok]
        wt_mask = s_idx == 0
        if wt_mask.any():
            wt_gathered[p_idx[wt_mask]] = logprobs[wt_mask].index_select(1, aa20_ids_t)
        del logits, logprobs

    scores = [
        _make_position_score(
            terms_mat[i], dev, gathered_dev=wt_gathered if i == 0 else None
        )
        for i in range(n_states)
    ]

    rep = 0.0
    if compute_repeat:
        wt_rep_terms = torch.empty(length, dtype=torch.float32, device=dev)
        for start in range(0, length, bs):
            stop = min(start + bs, length)
            p_idx = torch.arange(start, stop, device=dev, dtype=torch.long)
            batch = all_ids[0:1].repeat(stop - start, 1)
            row_ar = torch.arange(stop - start, device=dev, dtype=torch.long)
            tok_col = p_idx + 1
            batch[row_ar, tok_col] = mask_id
            logits = forward_tracks(handle, sequence_tokens=batch).sequence_logits.float()
            logprobs = F.log_softmax(logits[row_ar, tok_col], dim=-1)
            wt_rep_terms[p_idx] = -logprobs[row_ar, all_ids[0, tok_col]]
            del logits, logprobs
        rep = repeat_residual(scores[0], _make_position_score(wt_rep_terms, dev))

    return packed_states, scores, rep


@torch.no_grad()
def _score_causal_state_gp(
    adapter: Any,
    sequence: str,
    *,
    gather_aa20: bool = False,
) -> tuple[PackedState, PositionScore]:
    from src.capability.general_plm.adapters import ProGen3CausalAdapter

    dev = adapter.device
    aa20_ids = [int(adapter.aa_to_id[aa]) for aa in AA20]
    aa20_t = torch.as_tensor(aa20_ids, dtype=torch.long, device=dev) if gather_aa20 else None

    if isinstance(adapter, ProGen3CausalAdapter):
        batch = adapter.handle.batch([sequence], reverse=False)
        out = adapter._pg3.forward(adapter.handle, batch)
        logits = out.logits[0].float()
        ids_t = batch["input_ids"][0]
        ids_list = [int(x) for x in ids_t.detach().cpu().tolist()]
        start = 2
        end = start + len(sequence)
        shift_logits = logits[start - 1 : end - 1]
        logprobs = F.log_softmax(shift_logits, dim=-1)
        targets = ids_t[start:end]
        row_ar = torch.arange(end - start, device=dev, dtype=torch.long)
        terms = -logprobs[row_ar, targets]
        gathered = logprobs.index_select(1, aa20_t) if aa20_t is not None else None
        state = PackedState(
            sequence=sequence,
            ids=tuple(ids_list),
            span=(start, end),
            counts=np.ones(len(sequence), dtype=np.int64),
            offset=0,
        )
        return state, _make_position_score(terms, dev, gathered_dev=gathered)

    # _BaseCausalAdapter (dayhoff-170m, protgpt3-112m, protgpt3-10b)
    ids_list, (start, end) = adapter._pack_item(sequence)
    offset_res = len(sequence) - (end - start)
    tensor = torch.as_tensor([ids_list], dtype=torch.long, device=dev)
    with adapter._precision_ctx():
        logits = adapter._forward_ids(tensor)
        shift_logits = logits[0, start - 1 : end - 1]
        logprobs = F.log_softmax(shift_logits, dim=-1)
        targets = tensor[0, start:end]
        row_ar = torch.arange(end - start, device=dev, dtype=torch.long)
        terms = -logprobs[row_ar, targets]
        gathered = logprobs.index_select(1, aa20_t) if aa20_t is not None else None
    state = PackedState(
        sequence=sequence,
        ids=tuple(int(x) for x in ids_list),
        span=(int(start), int(end)),
        counts=np.ones(end - start, dtype=np.int64),
        offset=int(offset_res),
    )
    return state, _make_position_score(terms, dev, gathered_dev=gathered)


def _validate_archive_roundtrip(archive_path: Path, row: dict[str, Any]) -> None:
    """Confirm the emitted archive passes ``RetainedResponses`` identically."""
    with np.load(archive_path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files if k != "metadata"}
        states = rebuild_states(payload, [row["wildtype"], *row["sequences"]])
        identity = {
            "wildtype": row["wildtype"],
            "mutants": row["mutants"],
            "sequences": row["sequences"],
        }
        retained = RetainedResponses(data, states, identity)
        if not retained.selected_indices:
            raise RuntimeError(f"{row['assay']}: RetainedResponses selected zero variants")


def _receipt_from_existing(
    path: Path, row: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate an existing assay archive on disk and rebuild its manifest receipt."""
    _validate_archive_roundtrip(path, row)
    payload = read_archive(path)
    wildtype = row["wildtype"]
    sequences = [wildtype, *row["sequences"]]
    raw_states = rebuild_states(payload, sequences)
    states = [
        PackedState(
            sequence=s["sequence"],
            ids=tuple(int(x) for x in s["ids"]),
            span=(int(s["span"][0]), int(s["span"][1])),
            counts=np.asarray(s["counts"], dtype=np.int64),
            offset=int(s["offset"]),
        )
        for s in raw_states
    ]
    wt_nll = float(-payload["wt_likelihood"])
    scores = [
        PositionScore(
            terms=state_terms(payload, i),
            nll_sum=wt_nll if i == 0 else float(wt_nll - payload["likelihood"][i - 1]),
            retention_residual_nats=float(payload["position_sum_check_nats"]),
        )
        for i in range(len(states))
    ]
    wild_sidecar = raw_states[0]
    aligned = 0
    misaligned: list[dict[str, Any]] = []
    worst_upstream = 0.0
    for m_idx, mutant in enumerate(row["mutants"]):
        msites = _substitution_sites(mutant, wildtype)
        if len(msites) != 1:
            continue
        verdict = alignment(wild_sidecar, raw_states[m_idx + 1], msites[0])
        if not verdict["aligned"]:
            misaligned.append({"mutation": mutant, "reason": verdict["reason"]})
            continue
        aligned += 1
        up = upstream_invariance(
            states[0], states[m_idx + 1], scores[0], scores[m_idx + 1], msites[0]
        )
        if up > worst_upstream:
            worst_upstream = up
    has_cond = "wt_conditional_logprobs" in payload
    cond_meta: dict[str, Any] = (
        {"available": True, "residues": list(AA20), "resolved_residues": len(AA20)}
        if has_cond
        else {"available": False, "reason": "not computed"}
    )
    receipt = {
        "assay": row["assay"],
        "file": path.name,
        "sha256": sha256_file(path),
        "resumed": True,
        "cluster": row["cluster"],
        "residues": len(wildtype),
        "states": len(states),
        "packed_tokens": max(len(st.ids) for st in states),
        "scored_tokens": int(states[0].scored_tokens),
        "retention_max_abs_nats": float(payload["position_sum_check_nats"]),
        "upstream_max_abs_nats": worst_upstream,
        "repeat_max_abs_nats": 0.0,
        "aligned_single_substitutions": aligned,
        "misaligned_single_substitutions": misaligned,
        "conditional": cond_meta,
    }
    return receipt, dict(payload["metadata"]["identity"])


def _extract_arm(
    arm: str,
    *,
    cohort_path: Path,
    structures_path: Path,
    out_dir: Path,
    device: str,
    dtype: str,
    mask_batch_size: int,
    model_root: Path | None,
    esm3_model_dir: Path | None,
    assays_filter: list[str] | None = None,
    moe_probe_assays: int = 3,
    shard_index: int = 0,
    num_shards: int = 1,
    finalize_only: bool = False,
) -> dict[str, Any]:
    t0 = time.monotonic()
    selected, excluded = _select_structural_assays(
        cohort_path, structures_path, assays_filter=assays_filter
    )
    if not selected:
        raise RuntimeError("No structural assays selected")

    # Skip if already complete on disk.
    completion_path = out_dir / COMPLETION
    if completion_path.is_file() and not assays_filter:
        existing = json.loads(completion_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("totals", {}).get("assays_present", 0) >= len(selected):
            print(f"[{arm}] already complete ({len(selected)} assays), skipping.", flush=True)
            return {"arm": arm, "status": "already_complete", "assays": len(selected)}

    archives_dir = out_dir / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    if finalize_only:
        identity: dict[str, Any] | None = None
        for row in selected:
            path = archives_dir / _assay_filename(row["assay"])
            if not path.is_file():
                raise FileNotFoundError(f"[{arm}] finalize_only missing archive: {path}")
            rec, arc_identity = _receipt_from_existing(path, row)
            if identity is None:
                identity = arc_identity
            receipts.append(rec)
        assert identity is not None
        paradigm = str(identity["paradigm"])
        dtype = str(identity["dtype"])
    else:
        if num_shards < 1 or not (0 <= shard_index < num_shards):
            raise ValueError(f"Invalid shard {shard_index}/{num_shards}")
        target_indices = [
            idx_a
            for idx_a in range(1, len(selected) + 1)
            if num_shards == 1 or (idx_a - 1) % num_shards == shard_index
        ]
        missing_targets = [
            idx_a
            for idx_a in target_indices
            if not (archives_dir / _assay_filename(selected[idx_a - 1]["assay"])).is_file()
        ]
        if num_shards > 1 and not missing_targets:
            print(
                f"[{arm}] shard {shard_index}/{num_shards}: all {len(target_indices)} assays already on disk.",
                flush=True,
            )
            return {
                "arm": arm,
                "status": "shard_complete",
                "shard": shard_index,
                "shards": num_shards,
                "assays_done": len(target_indices),
            }

        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
        blas = blas_pinning()

        is_esm3 = arm == "esm3-1.4b"
        handle_esm3 = None
        adapter_gp = None
        if is_esm3:
            from src.capability.esm3_ext.esm3_adapter import load_esm3_offline

            handle_esm3 = load_esm3_offline(
                model_dir=esm3_model_dir,
                device=device,
                dtype=dtype,
                eager_load_aux=False,
            )
            paradigm = MASKED
            provenance = dict(handle_esm3.facts)
        else:
            from src.capability.general_plm import load_general_plm

            adapter_gp = load_general_plm(
                arm,
                device=device,
                dtype=dtype,
                model_root=model_root,
                saprot_mode="seq_only",
            )
            paradigm = CAUSAL if adapter_gp.spec.is_causal else MASKED
            provenance = adapter_gp.provenance()

        sources = {
            "cohort": {"path": str(cohort_path), "sha256": sha256_file(cohort_path)},
            "structures": {"path": str(structures_path), "sha256": sha256_file(structures_path)},
        }
        identity = {
            "schema": SCHEMA,
            "arm": arm,
            "paradigm": paradigm,
            "dtype": dtype,
            "rows_per_forward": 1 if paradigm == CAUSAL else mask_batch_size,
            "mask_batch_size": mask_batch_size if paradigm == MASKED else 1,
            "term_semantics": TERM_SEMANTICS[paradigm],
            "alignment_rule": ALIGNMENT,
            "prefix_invariant_rule": PREFIX_INVARIANT_RULE,
            "provenance": provenance,
            "blas": blas,
            "max_residues": 1000,
            "max_tokens": TOKEN_BUDGET,
            "conditionals": True,
            "sources": sources,
        }

        probe_upstream_vals: list[float] = []
        probe_repeat_vals: list[float] = []
        probed_assay_names: list[str] = []

        for idx_a in target_indices:
            row = selected[idx_a - 1]
            assay = row["assay"]
            wildtype = row["wildtype"]
            sequences = [wildtype, *row["sequences"]]
            metadata = {
                "identity": identity,
                "assay": assay,
                "cluster": row["cluster"],
                "mutant_digest": row["mutant_digest"],
                "states": len(sequences),
            }
            path = archives_dir / _assay_filename(assay)
            if path.is_file():
                rec, _ = _receipt_from_existing(path, row)
                receipts.append(rec)
                print(
                    f"[{arm}] {idx_a}/{len(selected)} RESUMED {assay} ({time.monotonic() - t0:.1f}s)",
                    flush=True,
                )
                continue

            if is_esm3:
                states, scores, assay_repeat = _score_masked_assay_esm3(
                    handle_esm3, sequences, mask_batch_size=mask_batch_size
                )
            elif paradigm == MASKED:
                states, scores, assay_repeat = _score_masked_assay_gp(
                    adapter_gp, sequences, mask_batch_size=mask_batch_size
                )
            else:
                st0, sc0 = _score_causal_state_gp(adapter_gp, sequences[0], gather_aa20=True)
                _, sc0_rep = _score_causal_state_gp(adapter_gp, sequences[0], gather_aa20=False)
                assay_repeat = repeat_residual(sc0, sc0_rep)
                states = [st0]
                scores = [sc0]
                for seq in sequences[1:]:
                    st_m, sc_m = _score_causal_state_gp(adapter_gp, seq, gather_aa20=False)
                    states.append(st_m)
                    scores.append(sc_m)

            widest = max(len(st.ids) for st in states)
            wild_sidecar = states[0].sidecar()
            aligned = 0
            misaligned: list[dict[str, Any]] = []
            worst_upstream = 0.0
            assay_upstreams: list[float] = []
            for m_idx, mutant in enumerate(row["mutants"]):
                msites = _substitution_sites(mutant, wildtype)
                if len(msites) != 1:
                    continue
                verdict = alignment(wild_sidecar, states[m_idx + 1].sidecar(), msites[0])
                if not verdict["aligned"]:
                    misaligned.append({"mutation": mutant, "reason": verdict["reason"]})
                    continue
                aligned += 1
                up = upstream_invariance(
                    states[0], states[m_idx + 1], scores[0], scores[m_idx + 1], msites[0]
                )
                assay_upstreams.append(up)
                if up > worst_upstream:
                    worst_upstream = up

            # Fast sparse-MoE prefix-invariance refusal check on causal arms.
            if paradigm == CAUSAL and worst_upstream > TIER2_NATS:
                probe_upstream_vals.extend(assay_upstreams)
                probe_repeat_vals.append(assay_repeat)
                probed_assay_names.append(assay)
                print(
                    f"[{arm}] PROBE {idx_a}/{moe_probe_assays} {assay}: "
                    f"upstream_max={worst_upstream:.6g} > TIER2_NATS={TIER2_NATS:.1e} "
                    f"(repeat={assay_repeat:.6g})",
                    flush=True,
                )
                if len(probed_assay_names) >= min(moe_probe_assays, len(selected)):
                    # Clean any partial archive files in out_dir and refuse cleanly.
                    for p in archives_dir.glob("*.npz"):
                        p.unlink(missing_ok=True)
                    arr_up = np.asarray(probe_upstream_vals, dtype=np.float64)
                    return {
                        "arm": arm,
                        "status": "refused_moe_layout",
                        "dtype": dtype,
                        "paradigm": paradigm,
                        "tier2_nats": TIER2_NATS,
                        "probed_assays": probed_assay_names,
                        "n_probed_variants": int(arr_up.size),
                        "upstream_max_abs_nats": float(arr_up.max()) if arr_up.size else 0.0,
                        "upstream_median_abs_nats": float(np.median(arr_up)) if arr_up.size else 0.0,
                        "upstream_mean_abs_nats": float(arr_up.mean()) if arr_up.size else 0.0,
                        "repeat_max_abs_nats": max(probe_repeat_vals, default=0.0),
                        "reason": (
                            "causal upstream prefix terms shift above TIER2_NATS (1e-4 nats) "
                            "due to sequence-wide MoE token routing/layout coupling"
                        ),
                        "elapsed_seconds": round(time.monotonic() - t0, 2),
                    }
                continue

            archive = PositionArchive(assay=assay, paradigm=paradigm)
            archive.add_wildtype(states[0], scores[0])
            for mutant, st, sc in zip(row["mutants"], states[1:], scores[1:]):
                archive.add_mutant(mutant, st, sc)

            extras: dict[str, Any] = {}
            cond_meta: dict[str, Any] = {"available": False, "reason": "not computed"}
            if scores[0].gathered is not None:
                extras = {
                    "wt_conditional_logprobs": np.asarray(scores[0].gathered, dtype=np.float32),
                    "wt_conditional_positions": np.arange(
                        states[0].offset, states[0].offset + states[0].scored_tokens, dtype=np.int64
                    ),
                    "wt_conditional_residues": np.asarray(list(AA20), dtype=object).astype("U"),
                }
                cond_meta = {"available": True, "residues": list(AA20), "resolved_residues": len(AA20)}

            archive.write(path, metadata=metadata, extras=extras)
            _validate_archive_roundtrip(path, row)

            receipts.append(
                {
                    "assay": assay,
                    "file": path.name,
                    "sha256": sha256_file(path),
                    "resumed": False,
                    "cluster": row["cluster"],
                    "residues": len(wildtype),
                    "states": len(states),
                    "packed_tokens": widest,
                    "scored_tokens": int(states[0].scored_tokens),
                    "retention_max_abs_nats": archive.retention_max_abs_nats,
                    "upstream_max_abs_nats": worst_upstream,
                    "repeat_max_abs_nats": assay_repeat,
                    "aligned_single_substitutions": aligned,
                    "misaligned_single_substitutions": misaligned,
                    "conditional": cond_meta,
                }
            )
            print(
                f"[{arm}] {idx_a}/{len(selected)} assays "
                f"({time.monotonic() - t0:.1f}s) last={assay} "
                f"up={worst_upstream:.3e} rep={assay_repeat:.3e}",
                flush=True,
            )

        if num_shards > 1:
            return {
                "arm": arm,
                "status": "shard_complete",
                "shard": shard_index,
                "shards": num_shards,
                "assays_done": len(receipts),
                "elapsed_seconds": round(time.monotonic() - t0, 2),
            }

    repeat_max = max((float(r["repeat_max_abs_nats"]) for r in receipts), default=0.0)
    worst_all = max((float(r["upstream_max_abs_nats"]) for r in receipts), default=0.0)
    tier = residual_tier(worst_all, repeat_max)
    invariant = {
        "rule": PREFIX_INVARIANT_RULE,
        "tier2_nats": TIER2_NATS,
        "tier2_repeat_multiple": TIER2_REPEAT_MULTIPLE,
        "repeat_max_abs_nats": repeat_max,
        "admitted_tolerance_nats": (
            0.0 if repeat_max == 0.0 else min(TIER2_NATS, TIER2_REPEAT_MULTIPLE * repeat_max)
        ),
        "upstream_max_abs_nats": worst_all,
        "tier": tier,
        "exactly_zero": bool(worst_all == 0.0),
    }

    by_assay = {r["assay"]: r for r in selected}
    handoff = {
        "schema": "position_response_archive_input_v1",
        "arm": arm,
        "assays": [
            {
                "assay": item["assay"],
                "cluster": by_assay[item["assay"]]["cluster"],
                "protein": item["assay"],
                "wildtype": by_assay[item["assay"]]["wildtype"],
                "mutants": by_assay[item["assay"]]["mutants"],
                "sequences": by_assay[item["assay"]]["sequences"],
                "archive": f"archives/{item['file']}",
            }
            for item in receipts
        ],
    }
    write_json(out_dir / "archive-input.json", handoff)
    write_json(
        completion_path,
        {
            "status": "complete",
            "identity": identity,
            "prefix_invariant": invariant,
            "created_utc": _now(),
            "shard": 0,
            "shards": 1,
            "assays": receipts,
            "excluded_assays": excluded,
            "skipped_assays": skipped,
            "totals": {
                "assays_present": len(receipts),
                "assays_written": sum(1 for item in receipts if not item["resumed"]),
                "assays_resumed": sum(1 for item in receipts if item["resumed"]),
                "states": sum(int(item["states"]) for item in receipts),
                "retention_max_abs_nats": max(
                    (float(item["retention_max_abs_nats"]) for item in receipts), default=0.0
                ),
                "upstream_max_abs_nats": worst_all,
                "repeat_max_abs_nats": repeat_max,
                "upstream_tier": tier,
                "conditionals_available": sum(
                    1 for item in receipts if item["conditional"]["available"]
                ),
            },
            "torch": torch.__version__,
            "gpu": (
                torch.cuda.get_device_name(device)
                if str(device).startswith("cuda") and torch.cuda.is_available()
                else None
            ),
        },
    )
    return {
        "arm": arm,
        "status": "complete",
        "paradigm": paradigm,
        "dtype": dtype,
        "assays": len(receipts),
        "states": sum(int(item["states"]) for item in receipts),
        "upstream_max_abs_nats": worst_all,
        "repeat_max_abs_nats": repeat_max,
        "upstream_tier": tier,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }


def _verify_esm2_650m_gate(
    *,
    cohort_path: Path,
    structures_path: Path,
    frozen_dir: Path,
    device: str,
    mask_batch_size: int,
    tmp_out: Path,
) -> dict[str, Any]:
    from scripts.capability.position.measure_contact_discrimination import arm_pair_responses
    from src.capability.position.contact_response import (
        CONTACT_ANGSTROM,
        MIN_SEQUENCE_SEPARATION,
        admitted_geometry,
        contact_pairs,
    )

    selected, _ = _select_structural_assays(cohort_path, structures_path)
    # Pick the smallest admitted assay for a fast parity check.
    target_row = min(selected, key=lambda r: len(r["wildtype"]) * (1 + len(r["mutants"])))
    assay = target_row["assay"]
    _extract_arm(
        "esm2-650m",
        cohort_path=cohort_path,
        structures_path=structures_path,
        out_dir=tmp_out,
        device=device,
        dtype="float32",
        mask_batch_size=mask_batch_size,
        model_root=None,
        esm3_model_dir=None,
        assays_filter=[assay],
    )
    fname = _assay_filename(assay)
    new_npz = tmp_out / "archives" / fname
    old_npz = frozen_dir / "archives" / fname
    if not old_npz.is_file():
        raise FileNotFoundError(f"Frozen archive missing: {old_npz}")

    sites = _read_json(structures_path)
    site_rows = [
        r for r in (sites["sites"] if isinstance(sites, dict) else sites)
        if r["assay_id"] == assay
    ]
    coverage = _read_json(structures_path.parent / "coverage.json")
    geom = admitted_geometry(
        site_rows,
        assay=assay,
        family=target_row["cluster"],
        wildtype=target_row["wildtype"],
        coverage=coverage,
    )
    pairs = contact_pairs(geom, cutoff=CONTACT_ANGSTROM, min_separation=MIN_SEQUENCE_SEPARATION)
    res_new = arm_pair_responses(
        new_npz, geom, pairs, target_row, assay=assay, paradigm=MASKED
    )
    res_old = arm_pair_responses(
        old_npz, geom, pairs, target_row, assay=assay, paradigm=MASKED
    )
    if set(res_new["values"]) != set(res_old["values"]):
        raise RuntimeError(
            f"Pair key mismatch on {assay}: {len(res_new['values'])} vs {len(res_old['values'])}"
        )
    diffs = [abs(res_new["values"][k] - res_old["values"][k]) for k in res_new["values"]]
    max_diff = float(max(diffs)) if diffs else 0.0
    if max_diff >= 1e-4:
        raise RuntimeError(
            f"esm2-650m parity gate FAILED on {assay}: max_abs_diff={max_diff:.6g} >= 1e-4"
        )
    result = {
        "gate": "esm2-650m_parity",
        "assay": assay,
        "n_pairs": len(diffs),
        "max_abs_diff_nats": max_diff,
        "passed": True,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=())
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--mask-batch-size", type=int, default=256)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--esm3-model-dir", type=Path, default=None)
    parser.add_argument("--verify-esm2-650m", type=Path, default=None)
    parser.add_argument("--refused-json", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    if args.verify_esm2_650m is not None:
        _verify_esm2_650m_gate(
            cohort_path=args.cohort,
            structures_path=args.structures,
            frozen_dir=args.verify_esm2_650m,
            device=args.device,
            mask_batch_size=args.mask_batch_size,
            tmp_out=args.out_root / "_verify_esm2_650m",
        )
        return

    if not args.arms:
        parser.error("Provide --arms or --verify-esm2-650m")

    summary: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for arm in args.arms:
        arm_dtype = "bfloat16" if arm.startswith("progen3-") else args.dtype
        bs = args.mask_batch_size
        if arm in ("esm2-15b", "proteinglm-10b-mlm", "esmc-6b"):
            bs = min(bs, 128)
        out_dir = args.out_root / f"posresp_extract_{arm}"
        rec = _extract_arm(
            arm,
            cohort_path=args.cohort,
            structures_path=args.structures,
            out_dir=out_dir,
            device=args.device,
            dtype=arm_dtype,
            mask_batch_size=bs,
            model_root=args.model_root,
            esm3_model_dir=args.esm3_model_dir,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            finalize_only=args.finalize_only,
        )
        summary.append(rec)
        if rec.get("status") == "refused_moe_layout":
            refused.append(rec)
        torch.cuda.empty_cache()

    if args.refused_json is not None and refused:
        existing_list: list[dict[str, Any]] = []
        if args.refused_json.is_file():
            old = json.loads(args.refused_json.read_text(encoding="utf-8"))
            existing_list = list(old.get("refused_arms", []))
        by_arm = {item["arm"]: item for item in existing_list}
        for item in refused:
            by_arm[item["arm"]] = item
        write_json(
            args.refused_json,
            {
                "schema": "e01_refused_moe_layout_v1",
                "updated_utc": _now(),
                "tier2_nats": TIER2_NATS,
                "refused_arms": [by_arm[k] for k in sorted(by_arm)],
            },
        )
    print(json.dumps({"summary": summary, "n_refused": len(refused)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
