#!/usr/bin/env python3
"""R3 interaction readout runner for general PLMs:
1. Exact deterministic O(L) leave-one-out PLL on all 12,977 MegaScale
   double-mutant states (L=37-72), producing genuinely non-degenerate
   epistasis cycles eps_PLL = PLL(AB) - PLL(A) - PLL(B) + PLL(WT), plus
   p=0 frozen hidden-state cycles R_cycle = R(AB) - R(A) - R(B) + R(WT).
2. 30-assay structural E01 contact-propagation extraction and 0-based
   contact-discrimination AUC (free of the historical -1 indexing bug).
"""

from __future__ import annotations

import argparse
import gzip
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
from src.capability.general_plm.eval_utils import (
    atomic_write_json,
    cluster_bootstrap_mean,
    project_four_blocks,
    safe_spearman,
    sha256_file,
)
from src.capability.position.contact_response import (
    CONTACT_ANGSTROM,
    MIN_SEQUENCE_SEPARATION,
    admitted_geometry,
    contact_pairs,
)

SCHEMA = "general_plm_r3_pairwise_pll_and_contacts_v1"


def _auc_minus_half(score: np.ndarray, label: np.ndarray) -> float | None:
    """Rank-sum AUC minus 0.5 with exact mid-rank tie handling."""
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(label, dtype=bool)
    pos = int(y.sum())
    neg = int(y.size - pos)
    if pos == 0 or neg == 0 or not np.isfinite(s).all():
        return None
    order = np.argsort(s, kind="stable")
    ranks = np.empty(s.size, dtype=np.float64)
    sorted_s = s[order]
    start = 0
    for stop in range(1, s.size + 1):
        if stop == s.size or sorted_s[stop] != sorted_s[start]:
            ranks[order[start:stop]] = 0.5 * (start + stop + 1)
            start = stop
    return float(ranks[y].sum() - pos * (pos + 1) / 2.0) / (pos * neg) - 0.5


def run_pairwise_cycles(
    adapter,
    root: Path,
    *,
    plan_path: Path,
    batch_size: int,
    readout_batch_size: int,
    extract_readouts: bool,
    max_backgrounds: int,
    out_dir: Path,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    backgrounds = plan["backgrounds"]
    if max_backgrounds > 0:
        backgrounds = backgrounds[:max_backgrounds]

    # Optional measured labels if present on host (kept strictly separate from extraction).
    measured_lookup: dict[tuple[str, str, str, str], float] = {}
    measured_cohort_path = root / "results/R3/epistasis_20260812/cohort.json"
    if measured_cohort_path.is_file():
        mc = json.loads(measured_cohort_path.read_text(encoding="utf-8"))
        for bg in mc.get("backgrounds", ()):
            for cyc in bg.get("cycles", ()):
                seqs = cyc.get("sequences")
                if seqs and len(seqs) == 4 and "epsilon" in cyc:
                    measured_lookup[tuple(seqs)] = float(cyc["epsilon"])

    t0 = time.monotonic()
    pooling = "mean_last" if adapter.spec.is_causal else "mean_cls"
    cycle_eps_list: list[float] = []
    cycle_groups: list[str] = []
    cycle_seps: list[str] = []
    cycle_measured: list[float] = []
    r_cycle_chunks: list[np.ndarray] = []
    state_scores_by_bg: dict[str, np.ndarray] = {}
    total_states = 0

    for b_idx, bg in enumerate(backgrounds, 1):
        seqs: list[str] = bg["sequences"]
        n_states = len(seqs)
        total_states += n_states
        plls = np.empty(n_states, dtype=np.float64)
        for s_idx, seq in enumerate(seqs):
            plls[s_idx] = float(adapter.exact_pll(seq, batch_size=batch_size).pll)
        state_scores_by_bg[bg["name"]] = plls

        proj_states = None
        if extract_readouts:
            raw_h = adapter.extract_hidden_states(seqs, batch_size=readout_batch_size, pooling=pooling)
            proj_states = project_four_blocks(raw_h)  # (n_states, 1024)

        bg_r_cycles = []
        for cyc in bg["cycles"]:
            i_wt, i_a, i_b, i_ab = [int(x) for x in cyc["states"]]
            eps_pll = float(plls[i_ab] - plls[i_a] - plls[i_b] + plls[i_wt])
            cycle_eps_list.append(eps_pll)
            cycle_groups.append(str(bg["group"]))
            cycle_seps.append(str(cyc["separation"]))
            key4 = (seqs[i_wt], seqs[i_a], seqs[i_b], seqs[i_ab])
            cycle_measured.append(measured_lookup.get(key4, np.nan))
            if proj_states is not None:
                r_cyc = proj_states[i_ab] - proj_states[i_a] - proj_states[i_b] + proj_states[i_wt]
                bg_r_cycles.append(r_cyc)
        if bg_r_cycles:
            r_cycle_chunks.append(np.stack(bg_r_cycles, axis=0))

        if b_idx % 8 == 0 or b_idx == len(backgrounds):
            print(f"[R3-cycles:{adapter.arm}] {b_idx}/{len(backgrounds)} backgrounds ({total_states} states)", flush=True)

    elapsed = time.monotonic() - t0
    eps_arr = np.asarray(cycle_eps_list, dtype=np.float64)
    meas_arr = np.asarray(cycle_measured, dtype=np.float64)
    grp_arr = np.asarray(cycle_groups, dtype="U")
    sep_arr = np.asarray(cycle_seps, dtype="U")

    save_payload: dict[str, Any] = {
        "cycle_eps_pll": eps_arr,
        "cycle_group": grp_arr,
        "cycle_separation": sep_arr,
        **{f"state_pll__{k}": v for k, v in state_scores_by_bg.items()},
    }
    if r_cycle_chunks:
        save_payload["R_cycle"] = np.concatenate(r_cycle_chunks, axis=0)
    np.savez_compressed(out_dir / f"r3_pairwise_cycles_{adapter.arm}.npz", **save_payload)

    nonzero_frac = float(np.mean(np.abs(eps_arr) > 1e-8)) if eps_arr.size else 0.0
    by_sep: dict[str, Any] = {}
    for sep_label in sorted(set(cycle_seps)):
        mask = sep_arr == sep_label
        by_sep[sep_label] = {
            "n_cycles": int(mask.sum()),
            "mean_abs_eps_pll_nats": float(np.mean(np.abs(eps_arr[mask]))),
            "std_eps_pll_nats": float(np.std(eps_arr[mask])),
        }

    measured_rho = None
    valid_m = np.isfinite(meas_arr)
    if int(valid_m.sum()) >= 10:
        per_g_rhos = []
        per_g_ids = []
        for grp in sorted(set(grp_arr[valid_m].tolist())):
            m = valid_m & (grp_arr == grp)
            rho = safe_spearman(eps_arr[m], meas_arr[m])
            if rho is not None:
                per_g_rhos.append(rho)
                per_g_ids.append(grp)
        measured_rho = cluster_bootstrap_mean(per_g_rhos, per_g_ids, seed=20261123)

    return {
        "endpoint": "r3_megascale_pairwise_cycles_exact_pll",
        "plan_sha256": sha256_file(plan_path),
        "n_backgrounds": len(backgrounds),
        "n_distinct_states": int(total_states),
        "n_cycles": int(eps_arr.size),
        "nonzero_cycle_fraction": nonzero_frac,
        "mean_abs_eps_pll_nats": float(np.mean(np.abs(eps_arr))) if eps_arr.size else 0.0,
        "max_abs_eps_pll_nats": float(np.max(np.abs(eps_arr))) if eps_arr.size else 0.0,
        "by_separation": by_sep,
        "zero_shot_spearman_vs_measured_epsilon": measured_rho,
        "elapsed_seconds": elapsed,
    }


def run_structural_contacts(
    adapter,
    root: Path,
    *,
    batch_size: int,
    max_assays: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Extract per-position propagation on the 30 structural E01 assays and compute 0-based contact AUC."""
    sites_path = root / "results/extensions/mutation_structure_20261006/structure/annotation/sites.json.gz"
    coverage_path = root / "results/extensions/mutation_structure_20261006/structure/annotation/coverage.json"
    cohort_path = root / "results/R5/context_mutation_rescue_20260923/cohort.json"
    if not (sites_path.is_file() and coverage_path.is_file() and cohort_path.is_file()):
        return {"endpoint": "r3_structural_contact_propagation", "status": "skipped_missing_inputs"}

    coverage_doc = json.loads(coverage_path.read_text(encoding="utf-8"))
    admitted_assays = {
        row["assay"]
        for row in coverage_doc.get("assays", ())
        if row.get("status") == "admitted"
    }

    with gzip.open(sites_path, "rt", encoding="utf-8") as fh:
        sites_doc = json.load(fh)
    site_rows = sites_doc["sites"] if isinstance(sites_doc, dict) else sites_doc
    sites_by_assay: dict[str, list[dict[str, Any]]] = {}
    for row in site_rows:
        sites_by_assay.setdefault(row["assay_id"], []).append(row)

    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort_by_assay = {row["assay"]: row for row in cohort["assays"]}
    assay_names = sorted(admitted_assays & set(sites_by_assay) & set(cohort_by_assay))
    if max_assays > 0:
        assay_names = assay_names[:max_assays]

    t0 = time.monotonic()
    assay_records: list[dict[str, Any]] = []
    aucs: list[float] = []
    families: list[int] = []

    for a_idx, assay in enumerate(assay_names, 1):
        c_row = cohort_by_assay[assay]
        wt = c_row["wildtype"]
        if len(wt) > adapter.spec.max_residues or (set(wt) - set(AA20)):
            continue
        geom = admitted_geometry(
            sites_by_assay[assay],
            assay=assay,
            family=int(c_row["cluster"]),
            wildtype=wt,
            coverage=coverage_doc,
        )
        pairs_doc = contact_pairs(
            geom, cutoff=CONTACT_ANGSTROM, min_separation=MIN_SEQUENCE_SEPARATION
        )
        eligible_pairs: dict[tuple[int, int], float] = pairs_doc["distance"]
        contact_set: set[tuple[int, int]] = pairs_doc["contacts"]
        if not eligible_pairs:
            continue

        wt_vec = adapter.exact_pll(wt, batch_size=batch_size).terms  # length L (0-based!)
        pair_responses: dict[tuple[int, int], list[float]] = {}
        n_single_scored = 0

        for mut_label, mut_seq in zip(c_row["mutants"], c_row["sequences"]):
            if ":" in mut_label:
                continue  # Single-site sending position only
            pos_i = int(mut_label[1:-1]) - 1  # 0-based
            mut_vec = adapter.exact_pll(mut_seq, batch_size=batch_size).terms
            if mut_vec.shape != wt_vec.shape:
                continue
            delta = np.abs(mut_vec - wt_vec)
            n_single_scored += 1
            for pos_j in range(len(wt)):
                if abs(pos_j - pos_i) < MIN_SEQUENCE_SEPARATION:
                    continue
                if adapter.spec.is_causal and pos_j < pos_i:
                    continue  # Causal upstream is identically zero
                key = (min(pos_i, pos_j), max(pos_i, pos_j))
                if key in eligible_pairs:
                    pair_responses.setdefault(key, []).append(float(delta[pos_j]))

        scored_keys = sorted(pair_responses.keys())
        if len(scored_keys) < 10:
            continue
        # Strictly 0-based key lookup (zero off-by-one bug!).
        scores = np.asarray([float(np.mean(pair_responses[k])) for k in scored_keys], dtype=np.float64)
        labels = np.asarray([k in contact_set for k in scored_keys], dtype=bool)
        auc_m_half = _auc_minus_half(scores, labels)
        if auc_m_half is not None:
            aucs.append(auc_m_half)
            families.append(int(c_row["cluster"]))
        assay_records.append(
            {
                "assay": assay,
                "cluster": int(c_row["cluster"]),
                "length": len(wt),
                "n_single_mutants": n_single_scored,
                "n_pairs": len(scored_keys),
                "n_contacts": int(labels.sum()),
                "auc_minus_half": auc_m_half,
            }
        )
        print(
            f"[R3-contacts:{adapter.arm}] {a_idx}/{len(assay_names)} {assay} "
            f"auc-0.5={auc_m_half if auc_m_half is not None else float('nan'):+.4f}",
            flush=True,
        )

    return {
        "endpoint": "r3_structural_contact_propagation_e01",
        "indexing_convention": "zero_based_direct_key_no_minus_one_shift",
        "n_assays": len(assay_records),
        "elapsed_seconds": time.monotonic() - t0,
        "summary_auc_minus_half": cluster_bootstrap_mean(aucs, families, seed=20261124),
        "assays": assay_records,
    }


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
    parser.add_argument("--skip-cycles", action="store_true")
    parser.add_argument("--skip-contacts", action="store_true")
    parser.add_argument("--skip-readouts", action="store_true")
    parser.add_argument("--max-backgrounds", type=int, default=0)
    parser.add_argument("--max-contact-assays", type=int, default=0)
    parser.add_argument(
        "--plan",
        type=Path,
        default=DATA_ROOT / "archive/logs/R3/pairwise_epistasis_20260924/extraction_plan.json",
    )
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

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "roster": "general_plm_20261120_exploratory",
        "arm": args.arm,
        "provenance": adapter.provenance(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not args.skip_cycles:
        payload["pairwise_cycles"] = run_pairwise_cycles(
            adapter,
            DATA_ROOT,
            plan_path=args.plan,
            batch_size=args.batch_size,
            readout_batch_size=args.readout_batch_size,
            extract_readouts=not args.skip_readouts,
            max_backgrounds=args.max_backgrounds,
            out_dir=args.out,
        )
    if not args.skip_contacts:
        payload["structural_contacts"] = run_structural_contacts(
            adapter,
            DATA_ROOT,
            batch_size=args.batch_size,
            max_assays=args.max_contact_assays,
            out_dir=args.out,
        )

    out_name = args.expect_file or f"r3_pairwise_and_contacts_{args.arm}.json"
    atomic_write_json(args.out / out_name, payload)
    print(f"[R3:{args.arm}] wrote {args.out / out_name}", flush=True)


if __name__ == "__main__":
    main()
