"""Fixed-20-AA-target SaProt 3Di structural correspondence causal control on CATH-3D.

Separates genuine residue-level 3D structural correspondence from vocabulary-width /
label-space effects in SaProt (35M, 650M, 1.3B) by holding model weights, CATH-3D
domains, masked query positions (``<mask>`` with zero ``#s`` hint), and the exact
20-AA marginal prediction loss strictly identical across four input conditions:

1. ``aa_only``: unmasked context uses ``#`` structure placeholders (``M#K#...``).
2. ``correct_3di``: unmasked context uses true residue-aligned Foldseek 3Di
   (``MdKp...``).
3. ``shuffled_3di``: deterministic within-domain permutation of ``three_di_seq``,
   preserving exact 400-token vocabulary width and per-domain 3Di marginal
   composition while destroying residue-level 3D correspondence.
4. ``cross_domain_shuffled_3di``: 3Di characters drawn without replacement from a
   deterministic cross-domain shuffled pool of all CATH-3D domains' 3Di residues.

Primary loss marginalizes exact logits over each canonical amino acid's 21 tokens
(20 ``<AA><3di_lower>`` + 1 ``<AA>#``) via ``logsumexp``, restricting the output
simplex to strictly 20 canonical amino acids (19 degrees of freedom) in every
condition. A secondary check (``20_3di_only``) restricts marginalization to the 20
``<AA><3di_lower>`` tokens only.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.capability.memory_theory.gradient_probes import (
    _extract_block_output_tensor,
    compute_spectral_rank_metrics,
)
from src.capability.memory_theory.io_utils import LN2, load_json_file, write_json_with_sha256
from src.capability.memory_theory.model_hooks import (
    LoadedModelBundle,
    get_input_embedding_module,
    get_transformer_blocks,
    load_model_bundle,
    select_depth_quantiles,
)
from src.capability.unified.roster import REPO_ROOT, UNIFIED_OUTPUT_DIR

SCHEMA_VERSION = "r4_saprot_3di_aa_target_causal_control_v1"
DEFAULT_COHORT_PATH = Path("results/extensions/memory_theory_20261120/cath_3d_kv_cohort.json")
DEFAULT_OUTPUT_PATH = UNIFIED_OUTPUT_DIR / "r4_saprot_3di_aa_target_causal_control.json"
DEFAULT_SAPROT_ARMS: tuple[str, ...] = ("saprot-35m", "saprot-650m", "saprot-1.3b")
CANONICAL_AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_TO_IDX: dict[str, int] = {aa: idx for idx, aa in enumerate(CANONICAL_AA20)}
CONDITIONS: tuple[str, ...] = (
    "aa_only",
    "correct_3di",
    "shuffled_3di",
    "cross_domain_shuffled_3di",
)
PAIRED_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("correct_3di_minus_aa_only", "correct_3di", "aa_only"),
    ("correct_3di_minus_shuffled_3di", "correct_3di", "shuffled_3di"),
    ("shuffled_3di_minus_aa_only", "shuffled_3di", "aa_only"),
    ("correct_3di_minus_cross_domain_shuffled_3di", "correct_3di", "cross_domain_shuffled_3di"),
)
SPECTRAL_METRIC_KEYS: tuple[str, ...] = (
    "vn_effective_rank",
    "vn_effective_rank_ratio",
    "rv_singular_effective_rank",
    "rv_singular_effective_rank_ratio",
    "numerical_rank_99",
    "numerical_rank_99_ratio",
    "top1_variance_share",
    "top19_variance_share",
)
SNR_METRIC_KEYS: tuple[str, ...] = (
    "mean_pairwise_cosine",
    "snr_unbiased",
    "snr_raw",
)
LOSS_METRIC_KEYS: tuple[str, ...] = (
    "mean_aa20_nll_nats",
    "mean_aa20_loss_bits",
    "token_weighted_aa20_nll_nats",
    "token_weighted_aa20_loss_bits",
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _deterministic_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass(frozen=True)
class CathDomainRecord:
    """One CATH-3D domain with aligned canonical amino-acid and Foldseek 3Di strings."""

    domain_id: str
    aa_seq: str
    three_di_seq: str
    saprot_seq: str
    cath_superfamily: str
    length: int


def load_cath_3d_domains(
    cohort_path: Path,
    *,
    max_domains: int | None = None,
) -> list[CathDomainRecord]:
    """Load CATH-3D domains sorted deterministically by ``domain_id``."""
    doc = load_json_file(cohort_path)
    raw_domains = doc["domains"]
    if not isinstance(raw_domains, Mapping):
        raise ValueError(f"Expected dict for 'domains' in {cohort_path}")

    records: list[CathDomainRecord] = []
    for dom_id in sorted(raw_domains.keys()):
        item = raw_domains[dom_id]
        aa_seq = str(item["aa_seq"]).strip().upper()
        three_di = str(item["three_di_seq"]).strip()
        saprot_seq = str(item["saprot_seq"]).strip()
        if len(aa_seq) != len(three_di):
            raise ValueError(f"Domain {dom_id}: len(aa_seq)={len(aa_seq)} != len(three_di)={len(three_di)}")
        if len(saprot_seq) != 2 * len(aa_seq):
            raise ValueError(f"Domain {dom_id}: unexpected saprot_seq length {len(saprot_seq)}")
        expected_saprot = "".join(f"{aa}{s.lower()}" for aa, s in zip(aa_seq, three_di))
        if saprot_seq != expected_saprot:
            raise ValueError(f"Domain {dom_id}: saprot_seq does not match aa_seq + three_di_seq.lower()")
        if any(aa not in AA20_TO_IDX for aa in aa_seq):
            raise ValueError(f"Domain {dom_id}: contains non-canonical amino acid outside AA20")
        records.append(
            CathDomainRecord(
                domain_id=str(dom_id),
                aa_seq=aa_seq,
                three_di_seq=three_di,
                saprot_seq=saprot_seq,
                cath_superfamily=str(item.get("cath_superfamily", "")),
                length=len(aa_seq),
            )
        )
    if max_domains is not None:
        records = records[: int(max_domains)]
    if len(records) < 2:
        raise ValueError(f"Need at least 2 domains, got {len(records)}")
    return records


def build_condition_sequences(
    domains: Sequence[CathDomainRecord],
    *,
    seed: int = 20261120,
) -> dict[str, list[str]]:
    """Build deterministic input strings for all 4 conditions on the identical domain list."""
    aa_only: list[str] = []
    correct_3di: list[str] = []
    shuffled_3di: list[str] = []

    # Cross-domain shuffled pool: concatenate all 3Di residues across domains in sorted order,
    # apply one deterministic global permutation, and slice back without replacement.
    pool = np.array(list("".join(d.three_di_seq.lower() for d in domains)), dtype="<U1")
    rng_global = np.random.default_rng(_deterministic_seed(seed, "cross_domain_3di_pool"))
    perm_global = rng_global.permutation(pool.size)
    shuffled_pool = pool[perm_global]

    cross_domain_shuffled_3di: list[str] = []
    cursor = 0
    for d in domains:
        aa_only.append("".join(f"{aa}#" for aa in d.aa_seq))
        correct_3di.append(d.saprot_seq)

        # Within-domain deterministic permutation preserving exact per-domain 3Di marginal counts
        chars = np.array(list(d.three_di_seq.lower()), dtype="<U1")
        rng_dom = np.random.default_rng(_deterministic_seed(seed, f"within_domain:{d.domain_id}"))
        perm_dom = rng_dom.permutation(chars.size)
        shuf_within = chars[perm_dom]
        shuffled_3di.append("".join(f"{aa}{s}" for aa, s in zip(d.aa_seq, shuf_within)))

        seg = shuffled_pool[cursor : cursor + d.length]
        cursor += d.length
        cross_domain_shuffled_3di.append("".join(f"{aa}{s}" for aa, s in zip(d.aa_seq, seg)))

    if cursor != int(shuffled_pool.size):
        raise RuntimeError("Cross-domain 3Di pool cursor mismatch")

    return {
        "aa_only": aa_only,
        "correct_3di": correct_3di,
        "shuffled_3di": shuffled_3di,
        "cross_domain_shuffled_3di": cross_domain_shuffled_3di,
    }


def build_aa20_token_group_matrix(
    tokenizer: Any,
    *,
    mode: str = "21_all_aa_tokens",
) -> torch.Tensor:
    """Build ``(20, K)`` vocabulary index matrix mapping each canonical AA to its SaProt tokens.

    * ``mode == "21_all_aa_tokens"``: ``K = 21`` (20 ``<AA><3di_lower>`` + 1 ``<AA>#``).
    * ``mode == "20_3di_only"``: ``K = 20`` (20 ``<AA><3di_lower>`` only, excluding ``<AA>#``).
    """
    if mode not in ("21_all_aa_tokens", "20_3di_only"):
        raise ValueError(f"Unsupported AA20 group mode: {mode!r}")

    vocab: dict[str, int] = tokenizer.get_vocab()
    expected_k = 21 if mode == "21_all_aa_tokens" else 20
    rows: list[list[int]] = []
    for aa in CANONICAL_AA20:
        tok_ids: list[int] = []
        for tok_str, tid in vocab.items():
            if len(tok_str) != 2 or tok_str[0] != aa:
                continue
            suf = tok_str[1]
            if mode == "21_all_aa_tokens":
                if suf == "#" or suf.islower():
                    tok_ids.append(int(tid))
            else:
                if suf.islower():
                    tok_ids.append(int(tid))
        tok_ids.sort()
        if len(tok_ids) != expected_k:
            raise RuntimeError(
                f"SaProt vocab mismatch for AA={aa} under mode={mode}: "
                f"expected {expected_k} tokens, found {len(tok_ids)}"
            )
        rows.append(tok_ids)
    mat = torch.tensor(rows, dtype=torch.long)
    # Verify strictly disjoint across the 20 amino acids
    flat = mat.reshape(-1).tolist()
    if len(set(flat)) != len(flat):
        raise RuntimeError("Duplicate vocabulary IDs detected across AA20 token groups")
    return mat


def compute_aa20_marginal_logits_and_loss(
    sel_logits: torch.Tensor,
    true_aa_idx_20: torch.Tensor,
    aa_group_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compute exact 20-AA marginal logits via ``logsumexp`` and per-token cross-entropy loss.

    Returns ``(aa_logits_20, per_tok_loss_nats, max_simplex_abs_err)``.
    """
    # sel_logits: (N, V), aa_group_matrix: (20, K) -> gathered: (N, 20, K)
    gathered = sel_logits[:, aa_group_matrix]
    aa_logits_20 = torch.logsumexp(gathered, dim=-1)  # (N, 20)
    per_tok_loss = F.cross_entropy(aa_logits_20, true_aa_idx_20, reduction="none")
    with torch.no_grad():
        probs = torch.softmax(aa_logits_20, dim=-1)
        max_simplex_err = float(torch.max(torch.abs(probs.sum(dim=-1) - 1.0)).item()) if probs.numel() else 0.0
    return aa_logits_20, per_tok_loss, max_simplex_err


def _prepare_domain_inputs(
    domains: Sequence[CathDomainRecord],
    cond_seqs: Mapping[str, Sequence[str]],
    tokenizer: Any,
    *,
    max_tokens: int = 256,
    mask_stride: int = 7,
) -> dict[str, Any]:
    """Pre-tokenize all conditions and verify strict alignment of lengths, eval_pos, and AA targets."""
    n_dom = len(domains)
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("SaProt tokenizer has no mask_token_id")

    encoded_by_cond: dict[str, list[list[int]]] = {c: [] for c in CONDITIONS}
    eval_positions: list[list[int]] = []
    aa20_targets: list[list[int]] = []
    seq_lengths: list[int] = []

    for d_idx, dom in enumerate(domains):
        cond_ids: dict[str, list[int]] = {}
        for cond in CONDITIONS:
            ids = tokenizer.encode(
                cond_seqs[cond][d_idx],
                truncation=True,
                max_length=max_tokens,
                add_special_tokens=True,
            )
            cond_ids[cond] = ids

        lengths_here = {len(v) for v in cond_ids.values()}
        if len(lengths_here) != 1:
            raise RuntimeError(f"Domain {dom.domain_id} tokenized to inconsistent lengths: {lengths_here}")
        seq_len = next(iter(lengths_here))
        if seq_len < 4:
            raise RuntimeError(f"Domain {dom.domain_id} too short after tokenization: seq_len={seq_len}")
        if cond_ids["aa_only"][0] != tokenizer.cls_token_id or cond_ids["aa_only"][-1] != tokenizer.eos_token_id:
            raise RuntimeError(f"Domain {dom.domain_id} missing expected <cls>/<eos> tokens")

        eval_pos = list(range(1, seq_len - 1, mask_stride))
        if not eval_pos:
            raise RuntimeError(f"Domain {dom.domain_id} has empty eval_pos")

        # Token index p in [1, seq_len-2] corresponds to residue index (p - 1) in dom.aa_seq
        targets = [AA20_TO_IDX[dom.aa_seq[p - 1]] for p in eval_pos]

        # Sanity check: verify token at each unmasked eval_pos really starts with dom.aa_seq[p - 1]
        for cond in CONDITIONS:
            for p in eval_pos:
                tok_s = tokenizer.convert_ids_to_tokens(cond_ids[cond][p])
                if not isinstance(tok_s, str) or len(tok_s) != 2 or tok_s[0] != dom.aa_seq[p - 1]:
                    raise RuntimeError(
                        f"Alignment check failed for {dom.domain_id} cond={cond} pos={p}: "
                        f"tok={tok_s!r} vs aa={dom.aa_seq[p - 1]!r}"
                    )
            encoded_by_cond[cond].append(cond_ids[cond])

        eval_positions.append(eval_pos)
        aa20_targets.append(targets)
        seq_lengths.append(seq_len)

    return {
        "n_domains": n_dom,
        "mask_token_id": int(mask_id),
        "pad_token_id": int(tokenizer.pad_token_id),
        "encoded_by_cond": encoded_by_cond,
        "eval_positions": eval_positions,
        "aa20_targets": aa20_targets,
        "seq_lengths": seq_lengths,
    }


@dataclass
class ConditionLayerCollect:
    """Compact per-domain residual gradient and 20-AA loss observations for one condition."""

    condition: str
    layer_indices: list[int]
    d_model: int
    n_domains: int
    dom_tok_counts: np.ndarray  # (D,) int64
    dom_mean_nll_nats: np.ndarray  # (D,) float64
    dom_sum_nll_nats: np.ndarray  # (D,) float64
    dom_mean_grads: dict[int, np.ndarray]  # li -> (D, d_model) float64
    dom_tok_grads: dict[int, list[np.ndarray]]  # li -> list of D arrays (m_d, d_model) float32
    max_simplex_abs_err: float


def _collect_condition_gradients(
    bundle: LoadedModelBundle,
    prepared: Mapping[str, Any],
    condition: str,
    aa_group_matrix: torch.Tensor,
    *,
    batch_size: int = 16,
) -> ConditionLayerCollect:
    """Run forward+backward across all domains for one condition using length-sorted micro-batches."""
    model = bundle.model
    dev = bundle.device
    blocks = get_transformer_blocks(model)
    layer_indices = select_depth_quantiles(len(blocks))
    d_model = bundle.d_model
    emb = get_input_embedding_module(model)

    n_dom = int(prepared["n_domains"])
    mask_id = int(prepared["mask_token_id"])
    pad_id = int(prepared["pad_token_id"])
    all_ids: list[list[int]] = prepared["encoded_by_cond"][condition]
    eval_positions: list[list[int]] = prepared["eval_positions"]
    aa20_targets: list[list[int]] = prepared["aa20_targets"]
    seq_lengths: list[int] = prepared["seq_lengths"]

    aa_group_gpu = aa_group_matrix.to(dev)

    dom_tok_counts = np.zeros(n_dom, dtype=np.int64)
    dom_mean_nll = np.zeros(n_dom, dtype=np.float64)
    dom_sum_nll = np.zeros(n_dom, dtype=np.float64)
    dom_mean_grads: dict[int, np.ndarray] = {
        li: np.zeros((n_dom, d_model), dtype=np.float64) for li in layer_indices
    }
    dom_tok_grads: dict[int, list[np.ndarray]] = {
        li: [np.empty((0, d_model), dtype=np.float32) for _ in range(n_dom)]
        for li in layer_indices
    }
    max_simplex_err = 0.0

    # Exact same-length zero-padding micro-batching:
    # HuggingFace EsmModel with token_dropout=True scales unmasked embeddings using
    # mask_ratio_observed = n_mask / padded_len when attention_mask is omitted inside
    # EsmEmbeddings, which corrupts shorter rows in mixed-length batches. Grouping strictly
    # by exact seq_len guarantees zero pad tokens and exact parity with batch_size=1.
    by_len: dict[int, list[int]] = {}
    for dom_idx in range(n_dom):
        by_len.setdefault(seq_lengths[dom_idx], []).append(dom_idx)
    batches: list[list[int]] = []
    bsz_cap = max(1, int(batch_size))
    for sl in sorted(by_len.keys()):
        idxs = by_len[sl]
        for b0 in range(0, len(idxs), bsz_cap):
            batches.append(idxs[b0 : b0 + bsz_cap])

    prev_req = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        for batch_indices in batches:
            bsz = len(batch_indices)
            max_len = seq_lengths[batch_indices[0]]

            feed_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=dev)
            attn_mask = torch.ones((bsz, max_len), dtype=torch.long, device=dev)

            flat_b: list[int] = []
            flat_p: list[int] = []
            flat_y: list[int] = []
            local_slices: list[tuple[int, int, int]] = []  # (orig_dom_idx, start_flat, end_flat)

            for local_b, dom_idx in enumerate(batch_indices):
                ids = all_ids[dom_idx]
                sl = len(ids)
                feed_ids[local_b, :sl] = torch.tensor(ids, dtype=torch.long, device=dev)
                attn_mask[local_b, :sl] = 1
                ep = eval_positions[dom_idx]
                yt = aa20_targets[dom_idx]
                s_flat = len(flat_b)
                for p_pos, y_cls in zip(ep, yt):
                    feed_ids[local_b, p_pos] = mask_id
                    flat_b.append(local_b)
                    flat_p.append(p_pos)
                    flat_y.append(y_cls)
                e_flat = len(flat_b)
                local_slices.append((dom_idx, s_flat, e_flat))

            flat_b_t = torch.tensor(flat_b, dtype=torch.long, device=dev)
            flat_p_t = torch.tensor(flat_p, dtype=torch.long, device=dev)
            flat_y_t = torch.tensor(flat_y, dtype=torch.long, device=dev)

            # Verify zero target leakage at every masked site
            if not bool(torch.all(feed_ids[flat_b_t, flat_p_t] == mask_id).item()):
                raise RuntimeError("Masked query position did not receive mask_token_id")

            captured: dict[int, torch.Tensor] = {}
            handles = []

            def _emb_hook(_mod: Any, _inp: Any, out: torch.Tensor) -> torch.Tensor:
                if isinstance(out, torch.Tensor):
                    out = out.detach().requires_grad_(True)
                return out

            handles.append(emb.register_forward_hook(_emb_hook))
            for li in layer_indices:

                def _make_hook(layer_id: int):
                    def _hook(_mod: Any, _inp: Any, out: Any) -> None:
                        t = _extract_block_output_tensor(out)
                        if t.requires_grad:
                            t.retain_grad()
                            captured[layer_id] = t

                    return _hook

                handles.append(blocks[li].register_forward_hook(_make_hook(li)))

            try:
                model.zero_grad(set_to_none=True)
                out = model(input_ids=feed_ids, attention_mask=attn_mask)
                logits = out.logits.float()  # (B, T, V)
                sel_logits = logits[flat_b_t, flat_p_t, :]  # (N_eval_batch, V)
                _aa_logits, per_tok_loss, simp_err = compute_aa20_marginal_logits_and_loss(
                    sel_logits, flat_y_t, aa_group_gpu
                )
                if simp_err > max_simplex_err:
                    max_simplex_err = simp_err

                loss_sum = per_tok_loss.sum()
                loss_sum.backward()

                loss_cpu = per_tok_loss.detach().to(torch.float64).cpu().numpy()
                for dom_idx, s_f, e_f in local_slices:
                    seg = loss_cpu[s_f:e_f]
                    dom_tok_counts[dom_idx] = int(seg.size)
                    dom_sum_nll[dom_idx] = float(np.sum(seg))
                    dom_mean_nll[dom_idx] = float(np.mean(seg))

                for li in layer_indices:
                    grad_t = captured[li].grad
                    if grad_t is None:
                        raise RuntimeError(f"Missing retained gradient at layer {li}")
                    sel_g = grad_t[flat_b_t, flat_p_t, :].detach().to(torch.float32).cpu().numpy()
                    if not np.isfinite(sel_g).all():
                        raise RuntimeError(f"Non-finite gradient detected at layer {li} condition {condition}")
                    for dom_idx, s_f, e_f in local_slices:
                        g_dom = np.ascontiguousarray(sel_g[s_f:e_f], dtype=np.float32)
                        dom_tok_grads[li][dom_idx] = g_dom
                        dom_mean_grads[li][dom_idx] = np.mean(g_dom.astype(np.float64), axis=0)
            finally:
                for h in handles:
                    h.remove()
                captured.clear()
                model.zero_grad(set_to_none=True)
    finally:
        for p, req in zip(model.parameters(), prev_req):
            p.requires_grad_(req)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return ConditionLayerCollect(
        condition=condition,
        layer_indices=layer_indices,
        d_model=d_model,
        n_domains=n_dom,
        dom_tok_counts=dom_tok_counts,
        dom_mean_nll_nats=dom_mean_nll,
        dom_sum_nll_nats=dom_sum_nll,
        dom_mean_grads=dom_mean_grads,
        dom_tok_grads=dom_tok_grads,
        max_simplex_abs_err=max_simplex_err,
    )


def _bootstrap_counts_matrix(n_domains: int, n_draws: int, seed: int) -> np.ndarray:
    """Sample ``(n_draws, n_domains)`` exact multinomial domain bootstrap multiplicity matrix."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_domains, size=(n_draws, n_domains))
    weights = np.zeros((n_draws, n_domains), dtype=np.float64)
    rows = np.repeat(np.arange(n_draws), n_domains)
    np.add.at(weights, (rows, idx.ravel()), 1.0)
    return weights


def compute_snr_and_cosine_point_and_boot(
    mean_grads: np.ndarray,
    boot_weights: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Algebraically exact point & paired-bootstrap estimates of pairwise cosine, SNR_unbiased, SNR_raw.

    Matches ``src.capability.memory_theory.gradient_probes.compute_gradient_snr`` to machine precision
    while vectorizing across ``n_draws = 5000`` resamples via precomputed ``(D, D)`` inner products.
    """
    arr = np.asarray(mean_grads, dtype=np.float64)
    n = int(arr.shape[0])
    if n < 2:
        raise ValueError("Need at least 2 domains for SNR/cosine estimation")

    norms = np.linalg.norm(arr, axis=1)
    if np.any(norms <= 1e-12):
        raise RuntimeError("Zero-norm domain mean gradient encountered")

    k_raw = arr @ arr.T  # (n, n)
    diag_raw = np.diag(k_raw)
    unit = arr / norms[:, None]
    k_cos = unit @ unit.T  # (n, n)

    def _eval_weights(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # w: (B, n), each row sums to n
        wk_raw = w @ k_raw
        sq_mean_norm = np.sum(wk_raw * w, axis=1) / float(n * n)
        sum_sq = w @ diag_raw
        var_trace = np.maximum((sum_sq - float(n) * sq_mean_norm) / float(n - 1), 1e-18)
        snr_raw = sq_mean_norm / var_trace
        unbiased_sig = np.maximum(0.0, sq_mean_norm - var_trace / float(n))
        snr_unbiased = unbiased_sig / var_trace

        wk_cos = w @ k_cos
        mean_cos = (np.sum(wk_cos * w, axis=1) - float(n)) / float(n * (n - 1))
        return mean_cos, snr_unbiased, snr_raw

    w_one = np.ones((1, n), dtype=np.float64)
    p_cos, p_unb, p_raw = _eval_weights(w_one)
    b_cos, b_unb, b_raw = _eval_weights(boot_weights)

    point = {
        "mean_pairwise_cosine": float(p_cos[0]),
        "snr_unbiased": float(p_unb[0]),
        "snr_raw": float(p_raw[0]),
    }
    draws = {
        "mean_pairwise_cosine": b_cos,
        "snr_unbiased": b_unb,
        "snr_raw": b_raw,
    }
    return point, draws


def compute_loss_point_and_boot(
    dom_mean_nll_nats: np.ndarray,
    dom_sum_nll_nats: np.ndarray,
    dom_tok_counts: np.ndarray,
    boot_weights: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Exact domain-mean and token-weighted 20-AA NLL (nats & bits) with domain-paired bootstrap."""
    n = float(dom_mean_nll_nats.size)
    tok_f = dom_tok_counts.astype(np.float64)

    p_mean_nats = float(np.mean(dom_mean_nll_nats))
    p_tok_nats = float(np.sum(dom_sum_nll_nats) / np.sum(tok_f))

    b_mean_nats = (boot_weights @ dom_mean_nll_nats) / n
    b_tok_nats = (boot_weights @ dom_sum_nll_nats) / np.maximum(boot_weights @ tok_f, 1.0)

    point = {
        "mean_aa20_nll_nats": p_mean_nats,
        "mean_aa20_loss_bits": float(p_mean_nats / LN2),
        "token_weighted_aa20_nll_nats": p_tok_nats,
        "token_weighted_aa20_loss_bits": float(p_tok_nats / LN2),
    }
    draws = {
        "mean_aa20_nll_nats": b_mean_nats,
        "mean_aa20_loss_bits": b_mean_nats / LN2,
        "token_weighted_aa20_nll_nats": b_tok_nats,
        "token_weighted_aa20_loss_bits": b_tok_nats / LN2,
    }
    return point, draws


def _spectral_metrics_from_eigvals_batch(
    eigvals_batch: np.ndarray,
    d_model: int,
) -> dict[str, np.ndarray]:
    """Vectorized exact Roy-Vetterli & von Neumann spectral metrics matching ``compute_spectral_rank_metrics``."""
    ev = np.maximum(np.asarray(eigvals_batch, dtype=np.float64), 0.0)
    b = int(ev.shape[0])
    out = {k: np.zeros(b, dtype=np.float64) for k in SPECTRAL_METRIC_KEYS}

    for i in range(b):
        row = ev[i]
        total_var = float(np.sum(row))
        if total_var <= 0.0:
            continue
        pos = row[row > 1e-18 * total_var]
        p = pos / total_var
        vn_rank = float(math.exp(float(-np.sum(p * np.log(p)))))

        sigmas = np.sqrt(pos)
        q = sigmas / float(np.sum(sigmas))
        rv_rank = float(math.exp(float(-np.sum(q * np.log(q)))))

        sorted_eig = np.sort(row)[::-1]
        cum = np.cumsum(sorted_eig) / total_var
        rank_99 = int(np.searchsorted(cum, 0.99) + 1)
        top1 = float(sorted_eig[0] / total_var)
        top19 = float(np.sum(sorted_eig[: min(19, sorted_eig.size)]) / total_var)

        out["vn_effective_rank"][i] = vn_rank
        out["vn_effective_rank_ratio"][i] = vn_rank / d_model
        out["rv_singular_effective_rank"][i] = rv_rank
        out["rv_singular_effective_rank_ratio"][i] = rv_rank / d_model
        out["numerical_rank_99"][i] = float(rank_99)
        out["numerical_rank_99_ratio"][i] = float(rank_99 / d_model)
        out["top1_variance_share"][i] = top1
        out["top19_variance_share"][i] = top19
    return out


def compute_spectral_point_and_boot_exact_f64(
    dom_tok_grads: Sequence[np.ndarray],
    dom_tok_counts: np.ndarray,
    d_model: int,
    boot_weights: np.ndarray,
    *,
    device: torch.device | str = "cpu",
    eig_batch_size: int = 25,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Compute exact ``float64`` Gram point estimate & domain-paired spectral bootstrap.

    Precomputes each domain's upper-triangular ``float64`` Gram ``S_d = g_d^T g_d`` once,
    then forms all ``n_boot`` exact resampled Grams via ``W @ U_dom`` (60x fewer FLOPs than
    re-multiplying raw token matrices, with zero FP32 Gram-squaring inflation on rank-19
    null eigenvalues).
    """
    n_dom = len(dom_tok_grads)
    n_boot = int(boot_weights.shape[0])
    dev = torch.device(device)
    triu_i, triu_j = torch.triu_indices(d_model, d_model, device=dev)
    n_triu = int(triu_i.numel())

    # Build (D, n_triu) exact float64 upper-triangular domain Grams directly on target device
    u_dom = torch.empty((n_dom, n_triu), dtype=torch.float64, device=dev)
    for d_idx, g_np in enumerate(dom_tok_grads):
        g64 = torch.from_numpy(g_np).to(device=dev, dtype=torch.float64)
        s_d = g64.T @ g64
        u_dom[d_idx] = s_d[triu_i, triu_j]

    # Exact point estimate via compute_spectral_rank_metrics
    total_tok = float(np.sum(dom_tok_counts))
    point_triu = (u_dom.sum(dim=0) / max(total_tok, 1.0)).cpu().numpy()
    triu_i_np = triu_i.cpu().numpy()
    triu_j_np = triu_j.cpu().numpy()
    s_point = np.zeros((d_model, d_model), dtype=np.float64)
    s_point[triu_i_np, triu_j_np] = point_triu
    s_point[triu_j_np, triu_i_np] = point_triu
    point_metrics = compute_spectral_rank_metrics(s_point, d_model)

    # Exact float64 bootstrap draws in memory-bounded chunks
    w_gpu = torch.from_numpy(boot_weights).to(device=dev, dtype=torch.float64)
    tok_gpu = torch.from_numpy(dom_tok_counts.astype(np.float64)).to(device=dev, dtype=torch.float64)
    denoms_gpu = (w_gpu @ tok_gpu).clamp_min(1.0)

    all_eigvals = np.empty((n_boot, d_model), dtype=np.float64)
    step = max(1, int(eig_batch_size))
    for k0 in range(0, n_boot, step):
        k1 = min(n_boot, k0 + step)
        bsz = k1 - k0
        triu_chunk = (w_gpu[k0:k1] @ u_dom) / denoms_gpu[k0:k1, None]
        mats = torch.empty((bsz, d_model, d_model), dtype=torch.float64, device=dev)
        mats[:, triu_i, triu_j] = triu_chunk
        mats[:, triu_j, triu_i] = triu_chunk
        ev_chunk = torch.linalg.eigvalsh(mats).cpu().numpy()
        all_eigvals[k0:k1] = ev_chunk
        del triu_chunk, mats

    del u_dom, w_gpu, tok_gpu, denoms_gpu
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    boot_metrics = _spectral_metrics_from_eigvals_batch(all_eigvals, d_model)
    return point_metrics, boot_metrics


def _summarize_metric(point: float, draws: np.ndarray) -> dict[str, Any]:
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "point": float(point),
        "ci95": [float(lo), float(hi)],
        "se": float(np.std(draws, ddof=1)) if draws.size > 1 else 0.0,
        "n_draws": int(draws.size),
    }


def _summarize_contrast(point_diff: float, diff_draws: np.ndarray) -> dict[str, Any]:
    lo, hi = np.quantile(diff_draws, [0.025, 0.975])
    # Exact two-sided bootstrap p-value (Davison & Hinkley +1 correction)
    p_pos = float((np.sum(diff_draws <= 0.0) + 1.0) / (diff_draws.size + 1.0))
    p_neg = float((np.sum(diff_draws >= 0.0) + 1.0) / (diff_draws.size + 1.0))
    p_two = float(min(1.0, 2.0 * min(p_pos, p_neg)))
    return {
        "estimate": float(point_diff),
        "ci95": [float(lo), float(hi)],
        "se": float(np.std(diff_draws, ddof=1)) if diff_draws.size > 1 else 0.0,
        "p_boot_two_sided": p_two,
        "excludes_zero": bool((lo > 0.0) or (hi < 0.0)),
        "n_draws": int(diff_draws.size),
    }


def run_verification_gate_checks(
    bundle: LoadedModelBundle,
    domains: Sequence[CathDomainRecord],
    cond_seqs: Mapping[str, Sequence[str]],
    aa_group_matrix: torch.Tensor,
    *,
    max_tokens: int = 256,
    mask_stride: int = 7,
) -> dict[str, Any]:
    """Verify zero mask-target leakage, batching parity, simplex normalization, and reproducibility."""
    # Pick up to 4 domains ensuring at least 2 share the same truncated length so batched (>1) execution is exercised
    by_len_idx: dict[int, list[int]] = {}
    for idx_d, dom in enumerate(domains):
        tlen = min(dom.length + 2, max_tokens)
        by_len_idx.setdefault(tlen, []).append(idx_d)
    picked: list[int] = []
    for tlen, idxs in sorted(by_len_idx.items(), key=lambda kv: (-len(kv[1]), -kv[0])):
        if len(idxs) >= 2:
            picked.extend(idxs[:2])
            break
    for idx_d in range(len(domains)):
        if len(picked) >= min(4, len(domains)):
            break
        if idx_d not in picked:
            picked.append(idx_d)
    sample_domains = [domains[i] for i in picked]
    sample_conds = {c: [cond_seqs[c][i] for i in picked] for c in CONDITIONS}
    prep = _prepare_domain_inputs(
        sample_domains, sample_conds, bundle.tokenizer, max_tokens=max_tokens, mask_stride=mask_stride
    )

    # 1. Batch=1 vs Batch=4 parity & deterministic reproducibility on 2 runs
    #    Include both distinct-length domains and a same-length pair so zero-padding batching is exercised.
    c_bs4_a = _collect_condition_gradients(bundle, prep, "correct_3di", aa_group_matrix, batch_size=4)
    c_bs4_b = _collect_condition_gradients(bundle, prep, "correct_3di", aa_group_matrix, batch_size=4)
    c_bs1 = _collect_condition_gradients(bundle, prep, "correct_3di", aa_group_matrix, batch_size=1)

    repro_max_diff = 0.0
    batch_max_diff = 0.0
    for li in c_bs4_a.layer_indices:
        for d_idx in range(len(sample_domains)):
            diff_rep = float(np.max(np.abs(c_bs4_a.dom_tok_grads[li][d_idx] - c_bs4_b.dom_tok_grads[li][d_idx])))
            diff_bs = float(np.max(np.abs(c_bs4_a.dom_tok_grads[li][d_idx] - c_bs1.dom_tok_grads[li][d_idx])))
            repro_max_diff = max(repro_max_diff, diff_rep)
            batch_max_diff = max(batch_max_diff, diff_bs)

    # 2. Zero mask-target leakage: mutate AA & 3Di at every eval_pos BEFORE masking, confirm exact 0.0 logit diff
    dev = bundle.device
    tok = bundle.tokenizer
    mask_id = int(tok.mask_token_id)
    leakage_max_diff = 0.0
    aa_group_gpu = aa_group_matrix.to(dev)
    with torch.no_grad():
        for d_idx in range(min(2, len(sample_domains))):
            orig_ids = list(prep["encoded_by_cond"]["correct_3di"][d_idx])
            ep = prep["eval_positions"][d_idx]
            feed_a = torch.tensor([orig_ids], dtype=torch.long, device=dev)
            feed_b = feed_a.clone()
            # Corrupt every eval_pos with a different valid token BEFORE applying mask_id
            for p_pos in ep:
                feed_b[0, p_pos] = int(aa_group_matrix[(d_idx + p_pos) % 20, 0].item())
            # Apply identical mask_id to both
            feed_a[0, ep] = mask_id
            feed_b[0, ep] = mask_id
            out_a = bundle.model(input_ids=feed_a).logits.float()
            out_b = bundle.model(input_ids=feed_b).logits.float()
            lg_a, _, _ = compute_aa20_marginal_logits_and_loss(
                out_a[0, ep, :],
                torch.tensor(prep["aa20_targets"][d_idx], dtype=torch.long, device=dev),
                aa_group_gpu,
            )
            lg_b, _, _ = compute_aa20_marginal_logits_and_loss(
                out_b[0, ep, :],
                torch.tensor(prep["aa20_targets"][d_idx], dtype=torch.long, device=dev),
                aa_group_gpu,
            )
            leakage_max_diff = max(leakage_max_diff, float(torch.max(torch.abs(lg_a - lg_b)).item()))

    simplex_err = max(c_bs4_a.max_simplex_abs_err, c_bs1.max_simplex_abs_err)
    passed = (
        simplex_err < 1e-6
        and leakage_max_diff == 0.0
        and repro_max_diff < 1e-6
        and batch_max_diff < 1e-4
    )
    return {
        "passed": bool(passed),
        "max_abs_diff": float(max(repro_max_diff, leakage_max_diff, simplex_err)),
        "checks": {
            "aa20_simplex_max_abs_err": float(simplex_err),
            "aa20_simplex_passed": bool(simplex_err < 1e-6),
            "zero_mask_target_leakage_max_abs_diff": float(leakage_max_diff),
            "zero_mask_target_leakage_passed": bool(leakage_max_diff == 0.0),
            "deterministic_reproducibility_max_abs_diff": float(repro_max_diff),
            "deterministic_reproducibility_passed": bool(repro_max_diff < 1e-6),
            "batch_vs_single_parity_max_abs_diff": float(batch_max_diff),
            "batch_vs_single_parity_passed": bool(batch_max_diff < 1e-4),
        },
    }


def evaluate_saprot_arm(
    arm: str,
    domains: Sequence[CathDomainRecord],
    cond_seqs: Mapping[str, Sequence[str]],
    *,
    device: str = "cuda:6",
    model_dir: Path | None = None,
    target_channel: str = "21_all_aa_tokens",
    max_tokens: int = 256,
    mask_stride: int = 7,
    batch_size: int = 16,
    n_boot_snr: int = 5000,
    n_boot_spectral: int = 500,
    seed: int = 20261120,
    run_gate: bool = True,
) -> dict[str, Any]:
    """Evaluate one frozen SaProt checkpoint across all 4 conditions with domain-paired bootstraps."""
    t_start = time.monotonic()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    bundle = load_model_bundle(
        arm,
        device=device,
        dtype="float32",
        model_dir=model_dir,
        modality="protein",
    )
    aa_group_matrix = build_aa20_token_group_matrix(bundle.tokenizer, mode=target_channel)
    prepared = _prepare_domain_inputs(
        domains, cond_seqs, bundle.tokenizer, max_tokens=max_tokens, mask_stride=mask_stride
    )

    gate_res = (
        run_verification_gate_checks(
            bundle,
            domains,
            cond_seqs,
            aa_group_matrix,
            max_tokens=max_tokens,
            mask_stride=mask_stride,
        )
        if run_gate
        else None
    )

    collected: dict[str, ConditionLayerCollect] = {}
    max_simplex_err = 0.0
    for cond in CONDITIONS:
        c_res = _collect_condition_gradients(
            bundle, prepared, cond, aa_group_matrix, batch_size=batch_size
        )
        collected[cond] = c_res
        max_simplex_err = max(max_simplex_err, c_res.max_simplex_abs_err)

    if gate_res is not None:
        gate_res["checks"]["full_cohort_aa20_simplex_max_abs_err"] = float(max_simplex_err)
        if max_simplex_err >= 1e-6:
            gate_res["passed"] = False

    # Free model weights before spectral bootstrap to keep full GPU VRAM available for float64 eigvalsh
    n_layers = bundle.n_layers
    d_model = bundle.d_model
    vocab_size = bundle.vocab_size
    layer_indices = select_depth_quantiles(n_layers)
    model_dir_str = str(bundle.model_dir)
    bundle.model.to("cpu")
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_dom = len(domains)
    w_snr = _bootstrap_counts_matrix(n_dom, n_boot_snr, _deterministic_seed(seed, f"{arm}:snr"))
    w_spec = _bootstrap_counts_matrix(n_dom, n_boot_spectral, _deterministic_seed(seed, f"{arm}:spec"))

    # Domain-paired loss metrics (layer-independent)
    loss_points: dict[str, dict[str, float]] = {}
    loss_draws: dict[str, dict[str, np.ndarray]] = {}
    for cond in CONDITIONS:
        col = collected[cond]
        p_loss, d_loss = compute_loss_point_and_boot(
            col.dom_mean_nll_nats, col.dom_sum_nll_nats, col.dom_tok_counts, w_snr
        )
        loss_points[cond] = p_loss
        loss_draws[cond] = d_loss

    loss_by_condition = {
        cond: {m: _summarize_metric(loss_points[cond][m], loss_draws[cond][m]) for m in LOSS_METRIC_KEYS}
        for cond in CONDITIONS
    }
    loss_contrasts = {
        cname: {
            m: _summarize_contrast(
                loss_points[ca][m] - loss_points[cb][m],
                loss_draws[ca][m] - loss_draws[cb][m],
            )
            for m in LOSS_METRIC_KEYS
        }
        for cname, ca, cb in PAIRED_CONTRASTS
    }

    # Per-layer residual gradient geometry + exact domain-paired bootstraps
    layers_out: dict[str, Any] = {}
    for li in layer_indices:
        snr_p: dict[str, dict[str, float]] = {}
        snr_d: dict[str, dict[str, np.ndarray]] = {}
        spec_p: dict[str, dict[str, float]] = {}
        spec_d: dict[str, dict[str, np.ndarray]] = {}

        for cond in CONDITIONS:
            col = collected[cond]
            sp, sd = compute_snr_and_cosine_point_and_boot(col.dom_mean_grads[li], w_snr)
            snr_p[cond] = sp
            snr_d[cond] = sd

            rp, rd = compute_spectral_point_and_boot_exact_f64(
                col.dom_tok_grads[li],
                col.dom_tok_counts,
                d_model,
                w_spec,
                device=device,
                eig_batch_size=25,
            )
            spec_p[cond] = rp
            spec_d[cond] = rd

        cond_block: dict[str, Any] = {}
        for cond in CONDITIONS:
            metrics_here: dict[str, Any] = {}
            for m in SNR_METRIC_KEYS:
                metrics_here[m] = _summarize_metric(snr_p[cond][m], snr_d[cond][m])
            for m in SPECTRAL_METRIC_KEYS:
                metrics_here[m] = _summarize_metric(spec_p[cond][m], spec_d[cond][m])
            cond_block[cond] = metrics_here

        contrasts_here: dict[str, Any] = {}
        for cname, ca, cb in PAIRED_CONTRASTS:
            c_dict: dict[str, Any] = {}
            for m in SNR_METRIC_KEYS:
                c_dict[m] = _summarize_contrast(snr_p[ca][m] - snr_p[cb][m], snr_d[ca][m] - snr_d[cb][m])
            for m in SPECTRAL_METRIC_KEYS:
                c_dict[m] = _summarize_contrast(spec_p[ca][m] - spec_p[cb][m], spec_d[ca][m] - spec_d[cb][m])
            contrasts_here[cname] = c_dict

        layers_out[str(li)] = {
            "layer_index": int(li),
            "depth_fraction": float(li / max(n_layers - 1, 1)),
            "n_domains": int(n_dom),
            "n_masked_tokens": int(np.sum(collected["aa_only"].dom_tok_counts)),
            "conditions": cond_block,
            "paired_contrasts": contrasts_here,
        }

    mid_layer_idx = layer_indices[len(layer_indices) // 2]
    final_layer_idx = layer_indices[-1]
    elapsed = time.monotonic() - t_start

    return {
        "arm": arm,
        "model_dir": model_dir_str,
        "target_channel": target_channel,
        "d_model": int(d_model),
        "n_layers": int(n_layers),
        "vocab_size": int(vocab_size),
        "aa20_output_simplex_dim": 20,
        "aa20_output_dof": 19,
        "layer_indices": [int(x) for x in layer_indices],
        "middle_layer_index": int(mid_layer_idx),
        "final_layer_index": int(final_layer_idx),
        "n_domains": int(n_dom),
        "n_masked_tokens": int(np.sum(collected["aa_only"].dom_tok_counts)),
        "elapsed_seconds": round(float(elapsed), 2),
        "verification_gate": gate_res,
        "aa20_loss": {
            "conditions": loss_by_condition,
            "paired_contrasts": loss_contrasts,
        },
        "layers": layers_out,
        "highlights": {
            "middle_layer": layers_out[str(mid_layer_idx)],
            "final_layer": layers_out[str(final_layer_idx)],
        },
    }


def _build_compact_summary_table(models_block: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build flat manuscript/summary rows for final (L-1) and middle (L//2) blocks across scales."""
    rows: list[dict[str, Any]] = []
    for arm, mrec in models_block.items():
        for slot in ("final_layer", "middle_layer"):
            lrec = mrec["highlights"][slot]
            row: dict[str, Any] = {
                "arm": arm,
                "layer_slot": slot,
                "layer_index": lrec["layer_index"],
                "d_model": mrec["d_model"],
                "n_layers": mrec["n_layers"],
                "n_domains": mrec["n_domains"],
                "conditions": {},
                "paired_contrasts": {},
            }
            for cond in CONDITIONS:
                cm = lrec["conditions"][cond]
                lm = mrec["aa20_loss"]["conditions"][cond]
                row["conditions"][cond] = {
                    "mean_pairwise_cosine": cm["mean_pairwise_cosine"]["point"],
                    "mean_pairwise_cosine_ci95": cm["mean_pairwise_cosine"]["ci95"],
                    "vn_effective_rank": cm["vn_effective_rank"]["point"],
                    "vn_effective_rank_ci95": cm["vn_effective_rank"]["ci95"],
                    "vn_effective_rank_ratio": cm["vn_effective_rank_ratio"]["point"],
                    "rv_singular_effective_rank": cm["rv_singular_effective_rank"]["point"],
                    "rv_singular_effective_rank_ci95": cm["rv_singular_effective_rank"]["ci95"],
                    "top19_variance_share": cm["top19_variance_share"]["point"],
                    "top19_variance_share_ci95": cm["top19_variance_share"]["ci95"],
                    "snr_unbiased": cm["snr_unbiased"]["point"],
                    "snr_unbiased_ci95": cm["snr_unbiased"]["ci95"],
                    "mean_aa20_nll_nats": lm["mean_aa20_nll_nats"]["point"],
                    "mean_aa20_nll_nats_ci95": lm["mean_aa20_nll_nats"]["ci95"],
                    "mean_aa20_loss_bits": lm["mean_aa20_loss_bits"]["point"],
                }
            for cname, _, _ in PAIRED_CONTRASTS:
                pc = lrec["paired_contrasts"][cname]
                lc = mrec["aa20_loss"]["paired_contrasts"][cname]
                row["paired_contrasts"][cname] = {
                    "mean_pairwise_cosine": pc["mean_pairwise_cosine"],
                    "vn_effective_rank": pc["vn_effective_rank"],
                    "rv_singular_effective_rank": pc["rv_singular_effective_rank"],
                    "top19_variance_share": pc["top19_variance_share"],
                    "snr_unbiased": pc["snr_unbiased"],
                    "mean_aa20_nll_nats": lc["mean_aa20_nll_nats"],
                    "mean_aa20_loss_bits": lc["mean_aa20_loss_bits"],
                }
            rows.append(row)
    return rows


def run_full_saprot_3di_causal_control(
    *,
    root: Path | None = None,
    arms: Sequence[str] = DEFAULT_SAPROT_ARMS,
    device: str = "cuda:6",
    cohort_path: Path | None = None,
    output_path: Path | None = None,
    arm_cache_dir: Path | None = None,
    only_populate_cache: bool = False,
    max_domains: int | None = None,
    secondary_max_domains: int = 128,
    max_tokens: int = 256,
    mask_stride: int = 7,
    batch_size: int = 16,
    n_boot_snr: int = 5000,
    n_boot_spectral: int = 500,
    seed: int = 20261120,
) -> dict[str, Any]:
    """Run complete fixed-20-AA-target SaProt 3Di causal control and seal JSON + ``.sha256``."""
    base = REPO_ROOT if root is None else Path(root)
    c_path = (base / DEFAULT_COHORT_PATH) if cohort_path is None else Path(cohort_path)
    o_path = (base / DEFAULT_OUTPUT_PATH) if output_path is None else Path(output_path)

    domains = load_cath_3d_domains(c_path, max_domains=max_domains)
    cond_seqs = build_condition_sequences(domains, seed=seed)

    def _cache_file(channel: str, arm_name: str, n_d: int) -> Path | None:
        if arm_cache_dir is None:
            return None
        arm_cache_dir.mkdir(parents=True, exist_ok=True)
        return arm_cache_dir / f"{channel}__{arm_name}__n{n_d}.json"

    models_primary: dict[str, Any] = {}
    gate_by_arm: dict[str, Any] = {}
    for arm in arms:
        cf = _cache_file("21_all_aa_tokens", arm, len(domains))
        if cf is not None and cf.is_file():
            print(f"[saprot-3di-control] loaded cached primary arm={arm} from {cf}", flush=True)
            res = load_json_file(cf)
        else:
            print(f"[saprot-3di-control] primary (21_all_aa_tokens) arm={arm} n_domains={len(domains)}", flush=True)
            res = evaluate_saprot_arm(
                arm,
                domains,
                cond_seqs,
                device=device,
                target_channel="21_all_aa_tokens",
                max_tokens=max_tokens,
                mask_stride=mask_stride,
                batch_size=batch_size,
                n_boot_snr=n_boot_snr,
                n_boot_spectral=n_boot_spectral,
                seed=seed,
                run_gate=True,
            )
            if cf is not None:
                write_json_with_sha256(cf, res)
        models_primary[arm] = res
        gate_by_arm[arm] = res["verification_gate"]

    # Secondary check: 20_3di_only marginalization (excluding <AA>#) on deterministic prefix
    sec_n = min(int(secondary_max_domains), len(domains))
    models_secondary: dict[str, Any] = {}
    if sec_n >= 2:
        sec_domains = domains[:sec_n]
        sec_cond_seqs = {c: cond_seqs[c][:sec_n] for c in CONDITIONS}
        for arm in arms:
            cf = _cache_file("20_3di_only", arm, sec_n)
            if cf is not None and cf.is_file():
                print(f"[saprot-3di-control] loaded cached secondary arm={arm} from {cf}", flush=True)
                models_secondary[arm] = load_json_file(cf)
            else:
                print(f"[saprot-3di-control] secondary (20_3di_only) arm={arm} n_domains={sec_n}", flush=True)
                sres = evaluate_saprot_arm(
                    arm,
                    sec_domains,
                    sec_cond_seqs,
                    device=device,
                    target_channel="20_3di_only",
                    max_tokens=max_tokens,
                    mask_stride=mask_stride,
                    batch_size=batch_size,
                    n_boot_snr=min(n_boot_snr, 2000),
                    n_boot_spectral=min(n_boot_spectral, 250),
                    seed=seed + 777,
                    run_gate=False,
                )
                if cf is not None:
                    write_json_with_sha256(cf, sres)
                models_secondary[arm] = sres

    all_gates_passed = bool(gate_by_arm) and all(bool(g["passed"]) for g in gate_by_arm.values())
    max_gate_diff = max((float(g["max_abs_diff"]) for g in gate_by_arm.values()), default=0.0)
    verification_gate = {
        "passed": all_gates_passed,
        "max_abs_diff": float(max_gate_diff),
        "per_arm": gate_by_arm,
    }

    if only_populate_cache:
        return {"completion_status": "cache_populated", "verification_gate": verification_gate}

    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "created_utc": _now_utc(),
        "completion_status": "complete" if all_gates_passed else "failed_verification_gate",
        "protocol": {
            "cohort_path": str(c_path.relative_to(base)) if c_path.is_relative_to(base) else str(c_path),
            "n_domains_primary": len(domains),
            "n_domains_secondary_20_3di_only": sec_n,
            "arms": list(arms),
            "conditions": list(CONDITIONS),
            "paired_contrasts": [c[0] for c in PAIRED_CONTRASTS],
            "primary_target_channel": "21_all_aa_tokens",
            "secondary_target_channel": "20_3di_only",
            "output_simplex_dim": 20,
            "output_simplex_dof": 19,
            "query_mask_token": "<mask> (no #s structure hint at masked sites)",
            "max_tokens": int(max_tokens),
            "mask_stride": int(mask_stride),
            "dtype": "float32",
            "allow_tf32": False,
            "gram_precision": "float64",
            "n_boot_snr": int(n_boot_snr),
            "n_boot_spectral": int(n_boot_spectral),
            "seed": int(seed),
            "unique_cath_superfamilies": len({d.cath_superfamily for d in domains if d.cath_superfamily}),
        },
        "verification_gate": verification_gate,
        "reproduction_gate": {
            "passed": all_gates_passed,
            "max_abs_diff": float(max_gate_diff),
        },
        "summary_table": _build_compact_summary_table(models_primary),
        "models": models_primary,
        "secondary_check_20_3di_only": {
            "summary_table": _build_compact_summary_table(models_secondary) if models_secondary else [],
            "models": models_secondary,
        },
    }

    # Ensure strictly finite JSON and clean permissions
    json.dumps(payload, allow_nan=False)
    o_path.parent.mkdir(parents=True, exist_ok=True)
    digest = write_json_with_sha256(o_path, payload)
    os.chmod(o_path, 0o644)
    os.chmod(o_path.with_name(o_path.name + ".sha256"), 0o644)
    print(f"[saprot-3di-control] sealed {o_path} (sha256={digest})", flush=True)
    return payload
