#!/usr/bin/env python3
"""Build Stage-2 Knob-3 temperature-controlled semi-synthetic ProteinMPNN corpus bundle.

Samples sequences on real 3D backbones (50 <= L <= 256) using local ``ProteinMPNN``
at controlled sampling temperatures ``T in {0.1, 0.3, 0.7, 1.2}`` while recording exact
ground-truth conditional Shannon entropy ``H_mpnn_bits_per_res`` and sampled NLL for
every temperature arm. Output uses the identical ``Stage2Bundle`` format as Knob 2.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.supply_demand.corpus import (  # noqa: E402
    AA20_SET,
    BUNDLE_SCHEMA,
    AnnotatedSequence,
    _now_utc,
    atomic_write_json_with_sha256,
    save_packed_split_npz,
)
from src.capability.supply_demand.spec import LN2  # noqa: E402
from src.capability.unified.proteinmpnn_adapter import (  # noqa: E402
    MPNN_ALPHABET,
    MPNN_X_IDX,
    ProteinMPNNAdapter,
    _cat_neighbors_nodes,
    _order_mask_backward,
    _prepare_coords_array,
)
from src.capability.unified.wt_backbones_3d import parse_pdb_backbone  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Stage-2 Knob-3 ProteinMPNN temperature corpus.")
    p.add_argument("--data-root", type=Path, default=Path("/Data/lzp/plm-capability-boundaries"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mpnn-arm", type=str, default="proteinmpnn-v-020")
    p.add_argument("--mpnn-model-root", type=Path, default=None)
    p.add_argument("--temperatures", type=str, default="0.1,0.3,0.7,1.2")
    p.add_argument("--n-train-backbones", type=int, default=256)
    p.add_argument("--depth-per-backbone", type=int, default=64)
    p.add_argument("--n-unseen-backbones", type=int, default=64)
    p.add_argument("--n-homolog-eval", type=int, default=1024)
    p.add_argument("--n-unseen-eval", type=int, default=1024)
    p.add_argument("--min-len", type=int, default=50)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=20261128)
    return p.parse_args()


@torch.no_grad()
def sample_mpnn_with_exact_entropy(
    adapter: ProteinMPNNAdapter,
    coords: np.ndarray,
    *,
    n_samples: int,
    temperature: float,
    seed: int,
) -> tuple[list[str], float, float]:
    """Sample n_samples sequences at `temperature` and compute exact conditional entropy H_T (bits/res).

    Returns ``(sequences, mean_cond_entropy_bits_per_res, mean_sampled_nll_bits_per_res)``.
    """
    clean, valid = _prepare_coords_array(coords, ca_only=adapter.spec.ca_only)
    length = clean.shape[0]
    h_V_enc, h_E, E_idx, h_EXV, mask_t = adapter._encode_single_backbone(coords)
    B = int(n_samples)
    gen_cpu = torch.Generator(device="cpu").manual_seed(int(seed))

    randn = torch.randn((B, length), generator=gen_cpu).to(device=adapter.device, dtype=torch.float32)
    order_t = torch.argsort(torch.abs(randn), dim=-1)

    h_V_b = h_V_enc.expand(B, -1, -1)
    h_E_b = h_E.expand(B, -1, -1, -1)
    E_idx_b = E_idx.expand(B, -1, -1)
    mask_b = mask_t.expand(B, -1)
    h_EXV_b = h_EXV.expand(B, -1, -1, -1)

    order_mask_bw = _order_mask_backward(order_t, dtype=h_V_b.dtype)
    mask_attend = torch.gather(order_mask_bw, 2, E_idx_b).unsqueeze(-1)
    mask_1D = mask_b.view(B, length, 1, 1)
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1.0 - mask_attend)
    h_EXV_fw = mask_fw * h_EXV_b

    h_S = torch.zeros_like(h_V_b)
    S = torch.zeros((B, length), dtype=torch.long, device=adapter.device)
    h_V_stack = [h_V_b.clone()] + [
        torch.zeros_like(h_V_b) for _ in range(len(adapter.model.decoder_layers))
    ]

    sample_gen = torch.Generator(device=adapter.device).manual_seed(int(seed) + 17)
    entropy_sum = 0.0
    nll_sum = 0.0
    valid_count = 0

    with adapter._precision_ctx():
        for t_step in range(length):
            t = order_t[:, t_step]
            mask_t_step = torch.gather(mask_b, 1, t[:, None])
            E_idx_t = torch.gather(E_idx_b, 1, t[:, None, None].expand(-1, 1, E_idx_b.shape[-1]))
            h_E_t = torch.gather(
                h_E_b, 1, t[:, None, None, None].expand(-1, 1, h_E_b.shape[-2], h_E_b.shape[-1])
            )
            h_ES_t = _cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
            h_EXV_t = torch.gather(
                h_EXV_fw, 1, t[:, None, None, None].expand(-1, 1, h_EXV_fw.shape[-2], h_EXV_fw.shape[-1])
            )
            mask_bw_t = torch.gather(
                mask_bw, 1, t[:, None, None, None].expand(-1, 1, mask_bw.shape[-2], mask_bw.shape[-1])
            )
            for l_idx, layer in enumerate(adapter.model.decoder_layers):
                h_ESV_dec_t = _cat_neighbors_nodes(h_V_stack[l_idx], h_ES_t, E_idx_t)
                h_V_t = torch.gather(
                    h_V_stack[l_idx], 1, t[:, None, None].expand(-1, 1, h_V_stack[l_idx].shape[-1])
                )
                h_ESV_t = mask_bw_t * h_ESV_dec_t + h_EXV_t
                h_V_stack[l_idx + 1].scatter_(
                    1,
                    t[:, None, None].expand(-1, 1, h_V_b.shape[-1]),
                    layer(h_V_t, h_ESV_t, mask_V=mask_t_step),
                )
            h_V_last = torch.gather(
                h_V_stack[-1], 1, t[:, None, None].expand(-1, 1, h_V_stack[-1].shape[-1])
            )[:, 0]
            raw_logits = adapter.model.W_out(h_V_last).to(dtype=torch.float32)[:, :20]
            scaled_logits = raw_logits / float(temperature)
            log_probs_T = F.log_softmax(scaled_logits, dim=-1)
            probs_T = log_probs_T.exp()

            # Exact conditional Shannon entropy H(p_T(. | X, s_{<t})) in bits
            step_h_bits = -(probs_T * log_probs_T).sum(dim=-1) / LN2
            S_t = torch.multinomial(probs_T, 1, generator=sample_gen)
            chosen_nll_bits = -torch.gather(log_probs_T, 1, S_t).squeeze(-1) / LN2

            m_valid = mask_t_step.squeeze(-1) > 0.5
            entropy_sum += float(step_h_bits[m_valid].sum().item())
            nll_sum += float(chosen_nll_bits[m_valid].sum().item())
            valid_count += int(m_valid.sum().item())

            emb_t = adapter.model.W_s(S_t)
            h_S.scatter_(1, t[:, None, None].expand(-1, 1, emb_t.shape[-1]), emb_t)
            S.scatter_(1, t[:, None], S_t)

    S_cpu = S.cpu().numpy()
    seqs = ["".join(MPNN_ALPHABET[int(c)] for c in S_cpu[b]) for b in range(B)]
    mean_h = entropy_sum / float(max(1, valid_count))
    mean_nll = nll_sum / float(max(1, valid_count))
    return seqs, mean_h, mean_nll


def _load_fold_disjoint_backbones(
    data_root: Path,
    *,
    n_train: int,
    n_unseen: int,
    min_len: int,
    max_len: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load fold-disjoint 3D backbones (N,CA,C,O) from cath_3d_kv_cohort + AFDB."""
    kv_path = data_root / "results/extensions/memory_theory_20261120/cath_3d_kv_cohort.json"
    af_dir = data_root / "data/alphafold"
    kv = json.loads(kv_path.read_text(encoding="utf-8"))

    by_topo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for acc, rec in sorted(kv["domains"].items()):
        L = int(rec["length"])
        if not (min_len <= L <= max_len):
            continue
        pdb_path = af_dir / f"AF-{acc}-F1-model_v6.pdb.gz"
        if not pdb_path.is_file():
            pdb_path = af_dir / f"AF-{acc}-F1-model_v4.pdb.gz"
        if not pdb_path.is_file():
            continue
        by_topo[str(rec["cath_topology"])].append(
            {
                "backbone_id": acc,
                "cath_superfamily": str(rec["cath_superfamily"]),
                "cath_topology": str(rec["cath_topology"]),
                "pdb_path": pdb_path,
            }
        )

    rng = np.random.default_rng(int(seed))
    topos = sorted(by_topo.keys())
    rng.shuffle(topos)

    unseen_bbs: list[dict[str, Any]] = []
    train_bbs: list[dict[str, Any]] = []
    unseen_topos: set[str] = set()

    for t in topos:
        recs = list(by_topo[t])
        rng.shuffle(recs)
        if len(unseen_bbs) < n_unseen:
            unseen_topos.add(t)
            for r in recs:
                if len(unseen_bbs) < n_unseen:
                    unseen_bbs.append(r)
        else:
            for r in recs:
                if len(train_bbs) < n_train:
                    train_bbs.append(r)

    if len(unseen_bbs) < n_unseen or len(train_bbs) < n_train:
        raise RuntimeError(
            f"Insufficient fold-disjoint AFDB backbones: train={len(train_bbs)}/{n_train}, "
            f"unseen={len(unseen_bbs)}/{n_unseen}"
        )

    def _hydrate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in items:
            seq, coords, _ = parse_pdb_backbone(r["pdb_path"])
            if min_len <= len(seq) <= max_len and set(seq).issubset(AA20_SET):
                out.append({**r, "sequence": seq, "coords": coords})
        return out

    return _hydrate(train_bbs), _hydrate(unseen_bbs)


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()
    out_dir = args.out_dir.resolve()
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    temps = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]
    train_bbs, unseen_bbs = _load_fold_disjoint_backbones(
        args.data_root.resolve(),
        n_train=int(args.n_train_backbones),
        n_unseen=int(args.n_unseen_backbones),
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        seed=int(args.seed),
    )

    device = args.device if torch.cuda.is_available() else "cpu"
    adapter = ProteinMPNNAdapter(
        args.mpnn_arm,
        model_root=args.mpnn_model_root,
        device=device,
        dtype="float32",
    )

    hom_per_bb = max(1, (int(args.n_homolog_eval) + len(train_bbs) - 1) // len(train_bbs))
    unseen_per_bb = max(1, (int(args.n_unseen_eval) + len(unseen_bbs) - 1) // len(unseen_bbs))

    arms_manifest: dict[str, Any] = {}
    unseen_fold_default: list[AnnotatedSequence] | None = None

    for t_idx, temp in enumerate(temps):
        arm_id = f"T{str(temp).replace('.', 'p')}"
        train_recs: list[AnnotatedSequence] = []
        hom_recs: list[AnnotatedSequence] = []
        unseen_recs: list[AnnotatedSequence] = []
        h_tr_list: list[float] = []
        nll_tr_list: list[float] = []
        h_hom_list: list[float] = []
        h_unf_list: list[float] = []

        for b_idx, bb in enumerate(train_bbs):
            base_seed = int(args.seed) + t_idx * 1_000_003 + b_idx * 97
            seqs_tr, h_tr, nll_tr = sample_mpnn_with_exact_entropy(
                adapter,
                bb["coords"],
                n_samples=int(args.depth_per_backbone),
                temperature=temp,
                seed=base_seed,
            )
            h_tr_list.append(h_tr)
            nll_tr_list.append(nll_tr)
            for s_i, s in enumerate(seqs_tr):
                train_recs.append(
                    AnnotatedSequence(
                        seq_id=f"{arm_id}_tr_{bb['backbone_id']}_{s_i}",
                        sequence=s,
                        cath_h=bb["cath_superfamily"],
                        cath_t=bb["cath_topology"],
                    )
                )

            seqs_hm, h_hm, _ = sample_mpnn_with_exact_entropy(
                adapter,
                bb["coords"],
                n_samples=hom_per_bb,
                temperature=temp,
                seed=base_seed + 500_000,
            )
            h_hom_list.append(h_hm)
            for s_i, s in enumerate(seqs_hm):
                hom_recs.append(
                    AnnotatedSequence(
                        seq_id=f"{arm_id}_hm_{bb['backbone_id']}_{s_i}",
                        sequence=s,
                        cath_h=bb["cath_superfamily"],
                        cath_t=bb["cath_topology"],
                    )
                )

        for b_idx, bb in enumerate(unseen_bbs):
            base_seed = int(args.seed) + t_idx * 1_000_003 + b_idx * 97 + 900_000
            seqs_uf, h_uf, _ = sample_mpnn_with_exact_entropy(
                adapter,
                bb["coords"],
                n_samples=unseen_per_bb,
                temperature=temp,
                seed=base_seed,
            )
            h_unf_list.append(h_uf)
            for s_i, s in enumerate(seqs_uf):
                unseen_recs.append(
                    AnnotatedSequence(
                        seq_id=f"{arm_id}_uf_{bb['backbone_id']}_{s_i}",
                        sequence=s,
                        cath_h=bb["cath_superfamily"],
                        cath_t=bb["cath_topology"],
                    )
                )

        hom_recs = hom_recs[: int(args.n_homolog_eval)]
        unseen_recs = unseen_recs[: int(args.n_unseen_eval)]
        if unseen_fold_default is None:
            unseen_fold_default = unseen_recs

        tr_meta = save_packed_split_npz(splits_dir / f"train_{arm_id}.npz", train_recs)
        sub_meta = save_packed_split_npz(
            splits_dir / f"eval_train_sub_{arm_id}.npz", train_recs[: min(1024, len(train_recs))]
        )
        hom_meta = save_packed_split_npz(splits_dir / f"eval_unseen_homolog_{arm_id}.npz", hom_recs)
        unf_meta = save_packed_split_npz(splits_dir / f"eval_unseen_fold_{arm_id}.npz", unseen_recs)

        arms_manifest[arm_id] = {
            "arm_id": arm_id,
            "temperature": float(temp),
            "n_train_backbones": len(train_bbs),
            "depth_per_backbone": int(args.depth_per_backbone),
            "H_mpnn_bits_per_res": float(np.mean(h_tr_list)),
            "sampled_nll_bits_per_res": float(np.mean(nll_tr_list)),
            "H_mpnn_homolog_bits_per_res": float(np.mean(h_hom_list)),
            "H_mpnn_unseen_fold_bits_per_res": float(np.mean(h_unf_list)),
            "train": tr_meta,
            "eval_train_sub": sub_meta,
            "eval_unseen_homolog": hom_meta,
            "eval_unseen_fold": unf_meta,
        }
        print(
            f"[knob3] {arm_id} (T={temp}): H_mpnn={arms_manifest[arm_id]['H_mpnn_bits_per_res']:.4f} bits/res "
            f"({tr_meta['n_sequences']} seqs, {tr_meta['total_tokens']:,} tokens)",
            flush=True,
        )

    assert unseen_fold_default is not None
    default_uf_meta = save_packed_split_npz(splits_dir / "eval_unseen_fold.npz", unseen_fold_default)
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "knob": "knob3_semisynthetic_proteinmpnn",
        "mpnn_arm": args.mpnn_arm,
        "created_utc": _now_utc(),
        "seed": int(args.seed),
        "unseen_fold_topologies": sorted({bb["cath_topology"] for bb in unseen_bbs}),
        "fold_disjointness_verified": True,
        "splits": {"eval_unseen_fold": default_uf_meta},
        "arms": arms_manifest,
        "benchmarks": {},
        "build_wall_sec": round(time.perf_counter() - t0, 2),
    }
    atomic_write_json_with_sha256(out_dir / "bundle_manifest.json", manifest)
    print(f"[knob3] Complete at {out_dir} in {manifest['build_wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
