#!/usr/bin/env python3
"""R2 quantitative readout runner for general PLMs: Domainome abundance
(109,568 vars), MegaScale stability (25,856 vars), and added phenotypes
(14,487 rows), plus p=0 frozen hidden-state blocks and bidirectional Ridge
evaluation against Profile + Potts + GEMME.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("TRANSFER_DATA_ROOT") or ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.amino_acids import AA20
from src.capability.general_plm import list_arms, load_general_plm
from src.capability.general_plm.adapters import apply_mutation
from src.capability.general_plm.eval_utils import (
    atomic_write_json,
    cluster_bootstrap_mean,
    nested_group_ridge_eval,
    project_four_blocks,
    safe_spearman,
)

SCHEMA = "general_plm_r2_phenotypes_and_readouts_v1"
AA_INDEX = {aa: idx for idx, aa in enumerate(AA20)}


def _load_baseline_tables(potts_path: Path, gemme_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    potts_map: dict[str, float] = {}
    if potts_path.is_file():
        for row in json.loads(potts_path.read_text(encoding="utf-8"))["samples"]:
            if row.get("status") == "fitted" and row.get("potts_delta") is not None:
                potts_map[row["sample_id"]] = float(row["potts_delta"])
    gemme_map: dict[str, float] = {}
    if gemme_path.is_file():
        for row in json.loads(gemme_path.read_text(encoding="utf-8"))["samples"]:
            if row.get("status") == "scored" and row.get("gemme_score") is not None:
                gemme_map[row["sample_id"]] = float(row["gemme_score"])
    return potts_map, gemme_map


def _profile_delta_from_npz(npz_path: Path) -> dict[str, np.ndarray]:
    """Load per-background frequency tables if present."""
    if not npz_path.is_file():
        return {}
    out: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as data:
        for key in data.files:
            if key.endswith("__frequencies") or key.startswith("freq::"):
                name = key.replace("__frequencies", "").replace("freq::", "")
                out[name] = np.asarray(data[key], dtype=np.float64)
            elif data[key].ndim == 2 and data[key].shape[1] == 20:
                out[key] = np.asarray(data[key], dtype=np.float64)
    return out


def _compute_profile_log_odds(freqs: np.ndarray | None, pos0: int, wt: str, mut: str) -> float:
    if freqs is None or not (0 <= pos0 < freqs.shape[0]):
        return np.nan
    f = np.clip(freqs[pos0], 1e-6, None)
    return float(np.log(f[AA_INDEX[mut]]) - np.log(f[AA_INDEX[wt]]))


def _score_and_fit_lane(
    adapter,
    *,
    lane_name: str,
    backgrounds: list[dict[str, Any]],
    potts_map: dict[str, float],
    gemme_map: dict[str, float],
    profile_freqs: dict[str, np.ndarray],
    batch_size: int,
    readout_batch_size: int,
    extract_readouts: bool,
    out_dir: Path,
) -> dict[str, Any]:
    t0 = time.monotonic()
    all_scores: list[float] = []
    all_targets: list[float] = []
    all_groups: list[str] = []
    all_subunits: list[str] = []
    all_profile: list[float] = []
    all_potts: list[float] = []
    all_gemme: list[float] = []
    r_chunks: list[np.ndarray] = []
    per_bg_rhos: list[float] = []
    per_bg_groups: list[str] = []
    skipped: list[dict[str, Any]] = []

    pooling = "mean_last" if adapter.spec.is_causal else "mean_cls"

    for b_idx, bg in enumerate(backgrounds, 1):
        wt = bg["wildtype"]
        if len(wt) > adapter.spec.max_residues or (set(wt) - set(AA20)):
            skipped.append({"background": bg["name"], "reason": "length_or_alphabet"})
            continue
        muts = bg["mutations"]
        targets = np.asarray(bg["targets"], dtype=np.float64)
        sample_ids = bg["sample_ids"]
        scores = adapter.score_variants(wt, muts, batch_size=batch_size, multi_exact_pll=False)

        rho = safe_spearman(scores, targets)
        if rho is not None:
            per_bg_rhos.append(rho)
            per_bg_groups.append(str(bg["group"]))

        freqs = profile_freqs.get(bg["profile_key"])
        for mut, sid, sc, yv in zip(muts, sample_ids, scores, targets):
            pos0 = int(mut[1:-1]) - 1
            all_scores.append(float(sc))
            all_targets.append(float(yv))
            all_groups.append(str(bg["group"]))
            all_subunits.append(f"{bg['name']}:{pos0}")
            all_profile.append(_compute_profile_log_odds(freqs, pos0, mut[0], mut[-1]))
            all_potts.append(potts_map.get(sid, np.nan))
            all_gemme.append(gemme_map.get(sid, np.nan))

        if extract_readouts:
            mut_seqs = [apply_mutation(wt, m) for m in muts]
            wt_h = adapter.extract_hidden_states([wt], batch_size=1, pooling=pooling)
            mut_h = adapter.extract_hidden_states(mut_seqs, batch_size=readout_batch_size, pooling=pooling)
            r_chunks.append(project_four_blocks(mut_h - wt_h))

        if b_idx % 50 == 0 or b_idx == len(backgrounds):
            print(f"[R2-{lane_name}:{adapter.arm}] {b_idx}/{len(backgrounds)} backgrounds", flush=True)

    elapsed = time.monotonic() - t0
    score_arr = np.asarray(all_scores, dtype=np.float64)
    target_arr = np.asarray(all_targets, dtype=np.float64)
    group_arr = np.asarray(all_groups, dtype="U")
    sub_arr = np.asarray(all_subunits, dtype="U")
    prof_arr = np.asarray(all_profile, dtype=np.float64)
    potts_arr = np.asarray(all_potts, dtype=np.float64)
    gemme_arr = np.asarray(all_gemme, dtype=np.float64)

    save_kwargs: dict[str, Any] = {
        "score": score_arr,
        "target": target_arr,
        "group": group_arr,
        "subunit": sub_arr,
    }
    R_mat = None
    if extract_readouts and r_chunks:
        R_mat = np.concatenate(r_chunks, axis=0)
        save_kwargs["R"] = R_mat
    np.savez_compressed(out_dir / f"r2_{lane_name}_arrays_{adapter.arm}.npz", **save_kwargs)

    # Bidirectional Ridge hierarchy against explicit controls (complete-case rows).
    complete = np.isfinite(score_arr) & np.isfinite(target_arr) & np.isfinite(potts_arr) & np.isfinite(gemme_arr)
    has_prof = np.isfinite(prof_arr[complete]).all() if complete.any() else False
    ridge_summary = None
    if int(complete.sum()) >= 20 and len(set(group_arr[complete].tolist())) >= 2:
        blocks: dict[str, np.ndarray] = {
            "M": score_arr[complete],
            "POTTS": potts_arr[complete],
            "GEMME": gemme_arr[complete],
        }
        base_ctrl: tuple[str, ...] = ("POTTS", "GEMME")
        if has_prof:
            blocks["P"] = prof_arr[complete]
            base_ctrl = ("P", "POTTS", "GEMME")
        designs: dict[str, tuple[str, ...]] = {
            "controls": base_ctrl,
            "M": ("M",),
            "controls_M": (*base_ctrl, "M"),
        }
        if R_mat is not None:
            blocks["R"] = R_mat[complete]
            designs["R"] = ("R",)
            designs["controls_R"] = (*base_ctrl, "R")
            designs["controls_M_R"] = (*base_ctrl, "M", "R")
        ridge_summary = nested_group_ridge_eval(
            target_arr[complete], group_arr[complete], sub_arr[complete], blocks, designs
        )

    return {
        "lane": lane_name,
        "n_backgrounds": len(backgrounds) - len(skipped),
        "n_variants": int(score_arr.size),
        "n_complete_case_explicit_stats": int(complete.sum()),
        "skipped": skipped,
        "elapsed_seconds": elapsed,
        "zero_shot_spearman": cluster_bootstrap_mean(per_bg_rhos, per_bg_groups, seed=20261120),
        "bidirectional_ridge": ridge_summary,
    }


def _load_abundance(root: Path, max_bgs: int) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float], dict[str, np.ndarray]]:
    cohort = json.loads((root / "results/R1/external_confirmation_20260924/cohort.json").read_text(encoding="utf-8"))
    potts, gemme = _load_baseline_tables(
        root / "results/extensions/explicit_statistics_abundance_20261109/potts/scores.json",
        root / "results/extensions/explicit_statistics_abundance_20261109/gemme/scores.json",
    )
    freqs = _profile_delta_from_npz(root / "results/R1/external_confirmation_20260924/profile_features.npz")
    bgs = []
    domains = cohort["domains"][:max_bgs] if max_bgs > 0 else cohort["domains"]
    for dom in domains:
        wt = dom["wildtype"]
        muts = [f"{wt[v['position'] - 1]}{v['position']}{v['mutant']}" for v in dom["variants"]]
        bgs.append(
            {
                "name": dom["name"],
                "profile_key": dom["name"],
                "group": dom["group"],
                "wildtype": wt,
                "mutations": muts,
                "sample_ids": [f"{dom['name']}::{m}" for m in muts],
                "targets": [float(v["target"]) for v in dom["variants"]],
            }
        )
    return bgs, potts, gemme, freqs


def _load_stability(root: Path, max_bgs: int) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float], dict[str, np.ndarray]]:
    cohort = json.loads((root / "results/R4/gate_stability_20260924/cohort/cohort.json").read_text(encoding="utf-8"))
    potts, gemme = _load_baseline_tables(
        root / "results/extensions/explicit_statistics_pairwise_20261109/potts/scores.json",
        root / "results/extensions/explicit_statistics_pairwise_20261109/gemme/scores.json",
    )
    freqs = _profile_delta_from_npz(root / "results/R4/gate_stability_20260924/cohort/profile_features.npz")
    bgs = []
    rows = cohort["backgrounds"][:max_bgs] if max_bgs > 0 else cohort["backgrounds"]
    for bg in rows:
        wt = bg["wildtype"]
        muts = [f"{wt[v['position'] - 1]}{v['position']}{v['mutant']}" for v in bg["variants"]]
        bgs.append(
            {
                "name": bg["name"],
                "profile_key": bg["name"],
                "group": bg["group"],
                "wildtype": wt,
                "mutations": muts,
                "sample_ids": [f"{bg['name']}::{m}" for m in muts],
                "targets": [float(v["ddg"]) for v in bg["variants"]],
            }
        )
    return bgs, potts, gemme, freqs


def _load_phenotypes(root: Path, max_bgs: int) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float], dict[str, np.ndarray]]:
    cat_list = json.loads(
        (root / "results/extensions/phenotype_breadth_20261008/qual-3/profile-catalogue.json").read_text(encoding="utf-8")
    )
    seq_by_wt = {item["WT_name"]: item["sequence"] for item in cat_list}
    potts, gemme = _load_baseline_tables(
        root / "results/extensions/explicit_statistics_phenotype_20261109/potts/scores.json",
        root / "results/extensions/explicit_statistics_phenotype_20261109/gemme/scores.json",
    )
    freqs = _profile_delta_from_npz(
        root / "results/extensions/phenotype_breadth_20261008/profiles/profile_features.npz"
    )
    rows_dir = root / "results/extensions/phenotype_breadth_20261008/qual-3/rows"
    grouped: dict[str, dict[str, Any]] = {}
    for jf in sorted(rows_dir.glob("*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            bg_name = r["background"]
            wt_sha = str(r.get("wildtype_sha256") or r.get("wt_sha256") or "")
            wt_key = r.get("wt_name") or r.get("profile_key") or f"{r['cohort']}:{wt_sha[:16]}"
            wt_seq = r.get("wildtype") or seq_by_wt.get(wt_key)
            if not wt_seq:
                continue
            pos = int(r["position"])
            wt_aa = r.get("wt_aa") or wt_seq[pos - 1]
            mt_aa = r["mt_aa"]
            if not (1 <= pos <= len(wt_seq)) or wt_seq[pos - 1] != wt_aa:
                continue
            mut = f"{wt_aa}{pos}{mt_aa}"
            label_val = (
                float(r["oriented_label"])
                if "oriented_label" in r and r["oriented_label"] is not None
                else float(r["label"]) * float(r.get("direction", 1))
            )
            bg_key = f"{r['cohort']}::{bg_name}"
            entry = grouped.setdefault(
                bg_key,
                {
                    "name": bg_name,
                    "profile_key": wt_key,
                    "group": str(r.get("group") or r.get("unit") or r.get("cluster") or bg_key),
                    "wildtype": wt_seq,
                    "mutations": [],
                    "sample_ids": [],
                    "targets": [],
                },
            )
            entry["mutations"].append(mut)
            entry["sample_ids"].append(f"{bg_name}::{mut}")
            entry["targets"].append(label_val)
    bgs = [grouped[k] for k in sorted(grouped.keys())]
    if max_bgs > 0:
        bgs = bgs[:max_bgs]
    return bgs, potts, gemme, freqs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", required=True, choices=list_arms())
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--saprot-mode", default="seq_only", choices=("seq_only", "foldseek_3di"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--readout-batch-size", type=int, default=16)
    parser.add_argument("--lanes", nargs="+", default=["abundance", "stability", "phenotype"])
    parser.add_argument("--skip-readouts", action="store_true")
    parser.add_argument("--max-backgrounds", type=int, default=0)
    parser.add_argument("--expect-file", default="")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    adapter = load_general_plm(
        args.arm,
        device=args.device,
        dtype=args.dtype,
        model_root=args.model_root,
        saprot_mode=args.saprot_mode,
    )

    loaders = {
        "abundance": _load_abundance,
        "stability": _load_stability,
        "phenotype": _load_phenotypes,
    }
    lane_results: dict[str, Any] = {}
    for lane in args.lanes:
        if lane not in loaders:
            raise SystemExit(f"Unknown R2 lane {lane!r}; expected {list(loaders)}")
        bgs, potts, gemme, freqs = loaders[lane](DATA_ROOT, args.max_backgrounds)
        lane_results[lane] = _score_and_fit_lane(
            adapter,
            lane_name=lane,
            backgrounds=bgs,
            potts_map=potts,
            gemme_map=gemme,
            profile_freqs=freqs,
            batch_size=args.batch_size,
            readout_batch_size=args.readout_batch_size,
            extract_readouts=not args.skip_readouts,
            out_dir=args.out,
        )

    out_name = args.expect_file or f"r2_phenotypes_and_readouts_{args.arm}.json"
    atomic_write_json(
        args.out / out_name,
        {
            "schema": SCHEMA,
            "roster": "general_plm_20261120_exploratory",
            "arm": args.arm,
            "provenance": adapter.provenance(),
            "lanes": lane_results,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"[R2:{args.arm}] wrote {args.out / out_name}", flush=True)


if __name__ == "__main__":
    main()
