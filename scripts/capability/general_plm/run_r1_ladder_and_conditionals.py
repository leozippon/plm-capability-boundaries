#!/usr/bin/env python3
"""R1 stage runner for general PLMs: 201-assay mutation ladder, matched
discrimination, 1-pass masked (or 20-string causal) family conditionals, and
p=0 frozen hidden-state probes.

Compatible with ``h200/run_external_baseline_h200.sh`` (accepts ``--device``
and ``--out`` first, writes an atomic JSON completion artifact).
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
from src.capability.extensions import family_conditionals as fc_mod
from src.capability.general_plm import list_arms, load_general_plm
from src.capability.general_plm.eval_utils import (
    atomic_write_json,
    cluster_bootstrap_mean,
    nested_group_ridge_eval,
    project_four_blocks,
    safe_spearman,
    sha256_file,
    standardized_rank,
)

SCHEMA = "general_plm_r1_ladder_and_conditionals_v1"
AA_INDEX = {aa: idx for idx, aa in enumerate(AA20)}


def _load_r1_ladder_inputs(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, float]]:
    """Join frozen 201-assay variant table with measured labels, Potts, and GEMME."""
    vt_path = root / "results/extensions/statistical_ladder_20261106/inputs/variant_table.json"
    cohort_path = root / "results/R5/context_mutation_rescue_20260923/cohort.json"
    potts_path = root / "results/extensions/statistical_ladder_20261106/potts-explicit-complete/scores.json"
    gemme_path = root / "results/extensions/statistical_ladder_20261106/gemme/scores.json"

    vt = json.loads(vt_path.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort_by_assay = {row["assay"]: row for row in cohort["assays"]}

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

    assays: dict[str, dict[str, Any]] = {}
    for sample in vt["samples"]:
        assay = sample["assay"]
        entry = assays.get(assay)
        if entry is None:
            c_row = cohort_by_assay[assay]
            mut_to_measured = {
                m: float(y) for m, y in zip(c_row["mutants"], c_row["measured"])
            }
            entry = {
                "assay": assay,
                "cluster": int(sample["cluster"]),
                "wildtype_id": sample["wildtype_id"],
                "wildtype": c_row["wildtype"],
                "mut_to_measured": mut_to_measured,
                "samples": [],
            }
            assays[assay] = entry
        entry["samples"].append(sample)
    return assays, potts_map, gemme_map


def run_mutation_ladder(
    adapter,
    root: Path,
    *,
    batch_size: int,
    multi_exact_pll: bool,
    extract_readouts: bool,
    readout_batch_size: int,
    max_assays: int,
    out_dir: Path,
) -> dict[str, Any]:
    assays, potts_map, gemme_map = _load_r1_ladder_inputs(root)
    assay_names = sorted(assays.keys())
    if max_assays > 0:
        assay_names = assay_names[:max_assays]

    per_assay_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    sample_scores: dict[str, float] = {}

    # Vectors for optional p=0 hidden-state Ridge probe
    probe_y: list[float] = []
    probe_groups: list[int] = []
    probe_subunits: list[str] = []
    probe_m: list[float] = []
    probe_p: list[float] = []
    probe_potts: list[float] = []
    probe_gemme: list[float] = []
    probe_r_chunks: list[np.ndarray] = []

    t0 = time.monotonic()
    for idx, assay_name in enumerate(assay_names, 1):
        entry = assays[assay_name]
        wt = entry["wildtype"]
        if len(wt) > adapter.spec.max_residues:
            skipped.append({"assay": assay_name, "reason": f"len {len(wt)} > {adapter.spec.max_residues}"})
            continue
        samples = entry["samples"]
        mutations = [s["mutation"] for s in samples]
        measured = np.asarray([entry["mut_to_measured"][m] for m in mutations], dtype=np.float64)
        profile = np.asarray([float(s["profile_delta"]) for s in samples], dtype=np.float64)

        scores = adapter.score_variants(
            wt, mutations, batch_size=batch_size, multi_exact_pll=multi_exact_pll
        )
        for s, sc in zip(samples, scores):
            sample_scores[s["sample_id"]] = float(sc)

        potts_vals = np.asarray([potts_map.get(s["sample_id"], np.nan) for s in samples], dtype=np.float64)
        gemme_vals = np.asarray([gemme_map.get(s["sample_id"], np.nan) for s in samples], dtype=np.float64)

        rho_m = safe_spearman(scores, measured)
        rho_p = safe_spearman(profile, measured)
        rho_mp = safe_spearman(scores, profile)
        rho_potts = safe_spearman(potts_vals, measured) if np.isfinite(potts_vals).all() else None
        rho_gemme = safe_spearman(gemme_vals, measured) if np.isfinite(gemme_vals).all() else None

        # Rank-MSE within assay (higher score = better fitness -> rank standardized)
        y_rank = standardized_rank(measured)
        m_rank = standardized_rank(scores)
        p_rank = standardized_rank(profile)
        rank_mse_m = float(np.mean((y_rank - m_rank) ** 2))
        rank_mse_p = float(np.mean((y_rank - p_rank) ** 2))

        per_assay_rows.append(
            {
                "assay": assay_name,
                "cluster": entry["cluster"],
                "wildtype_length": len(wt),
                "n_variants": len(samples),
                "n_multi": int(sum(1 for s in samples if int(s["n_substitutions"]) > 1)),
                "model_spearman": rho_m,
                "profile_spearman": rho_p,
                "potts_spearman": rho_potts,
                "gemme_spearman": rho_gemme,
                "model_minus_profile": None if (rho_m is None or rho_p is None) else float(rho_m - rho_p),
                "model_minus_potts": None if (rho_m is None or rho_potts is None) else float(rho_m - rho_potts),
                "model_minus_gemme": None if (rho_m is None or rho_gemme is None) else float(rho_m - rho_gemme),
                "model_profile_spearman": rho_mp,
                "rank_mse_model": rank_mse_m,
                "rank_mse_profile": rank_mse_p,
            }
        )

        if extract_readouts:
            from src.capability.general_plm.adapters import apply_mutation

            mut_seqs = [apply_mutation(wt, m) for m in mutations]
            pooling = "mean_last" if adapter.spec.is_causal else "mean_cls"
            wt_h = adapter.extract_hidden_states([wt], batch_size=1, pooling=pooling)
            mut_h = adapter.extract_hidden_states(mut_seqs, batch_size=readout_batch_size, pooling=pooling)
            delta_h = mut_h - wt_h  # (n_var, 4, H)
            proj = project_four_blocks(delta_h)
            probe_r_chunks.append(proj)
            probe_y.extend(y_rank.tolist())
            probe_groups.extend([entry["cluster"]] * len(samples))
            probe_subunits.extend([assay_name] * len(samples))
            probe_m.extend(m_rank.tolist())
            probe_p.extend(p_rank.tolist())
            probe_potts.extend(
                standardized_rank(potts_vals).tolist()
                if np.isfinite(potts_vals).all()
                else [np.nan] * len(samples)
            )
            probe_gemme.extend(
                standardized_rank(gemme_vals).tolist()
                if np.isfinite(gemme_vals).all()
                else [np.nan] * len(samples)
            )

        if idx % 25 == 0 or idx == len(assay_names):
            print(f"[R1-ladder:{adapter.arm}] {idx}/{len(assay_names)} assays", flush=True)

    elapsed = time.monotonic() - t0
    clusters = [r["cluster"] for r in per_assay_rows]
    summary = {}
    for metric in (
        "model_spearman",
        "profile_spearman",
        "potts_spearman",
        "gemme_spearman",
        "model_minus_profile",
        "model_minus_potts",
        "model_minus_gemme",
        "model_profile_spearman",
    ):
        vals = [r[metric] for r in per_assay_rows if r[metric] is not None]
        cl = [r["cluster"] for r in per_assay_rows if r[metric] is not None]
        summary[metric] = cluster_bootstrap_mean(vals, cl, seed=20261120)

    readout_eval = None
    if extract_readouts and probe_r_chunks:
        R_mat = np.concatenate(probe_r_chunks, axis=0)
        y_arr = np.asarray(probe_y, dtype=np.float64)
        g_arr = np.asarray(probe_groups)
        s_arr = np.asarray(probe_subunits)
        m_arr = np.asarray(probe_m, dtype=np.float64)
        p_arr = np.asarray(probe_p, dtype=np.float64)
        potts_arr = np.asarray(probe_potts, dtype=np.float64)
        gemme_arr = np.asarray(probe_gemme, dtype=np.float64)
        valid = np.isfinite(potts_arr) & np.isfinite(gemme_arr)
        np.savez_compressed(
            out_dir / f"r1_readout_blocks_{adapter.arm}.npz",
            R=R_mat,
            y_rank=y_arr,
            group=g_arr,
            assay=s_arr,
            M_rank=m_arr,
            P_rank=p_arr,
        )
        if int(valid.sum()) > 10 and len(set(g_arr[valid].tolist())) >= 2:
            blocks = {
                "P": p_arr[valid],
                "POTTS": potts_arr[valid],
                "GEMME": gemme_arr[valid],
                "M": m_arr[valid],
                "R": R_mat[valid],
            }
            designs = {
                "P": ("P",),
                "P_POTTS_GEMME": ("P", "POTTS", "GEMME"),
                "M": ("M",),
                "R": ("R",),
                "P_POTTS_GEMME_M": ("P", "POTTS", "GEMME", "M"),
                "P_POTTS_GEMME_R": ("P", "POTTS", "GEMME", "R"),
                "P_POTTS_GEMME_M_R": ("P", "POTTS", "GEMME", "M", "R"),
            }
            readout_eval = nested_group_ridge_eval(
                y_arr[valid], g_arr[valid], s_arr[valid], blocks, designs
            )

    np.savez_compressed(
        out_dir / f"r1_ladder_scores_{adapter.arm}.npz",
        sample_ids=np.asarray(list(sample_scores.keys()), dtype="U"),
        scores=np.asarray(list(sample_scores.values()), dtype=np.float64),
    )

    return {
        "endpoint": "r1_mutation_ladder_201",
        "multi_exact_pll": bool(multi_exact_pll),
        "n_assays": len(per_assay_rows),
        "n_families": len(set(clusters)),
        "n_variants": int(sum(r["n_variants"] for r in per_assay_rows)),
        "skipped": skipped,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "assays": per_assay_rows,
        "hidden_state_probe": readout_eval,
    }


def run_matched_discrimination(
    adapter, root: Path, *, batch_size: int, max_assays: int
) -> dict[str, Any]:
    cohort_path = (
        root / "results/R1/matched_discrimination_20261102/expanded/expanded_variants/cohort/cohort.json"
    )
    pairs_path = (
        root / "results/R1/matched_discrimination_20261102/expanded/expanded_variants/cohort/pairs.json"
    )
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    pairs_doc = json.loads(pairs_path.read_text(encoding="utf-8"))
    assay_rows = cohort["assays"][:max_assays] if max_assays > 0 else cohort["assays"]

    t0 = time.monotonic()
    score_table: dict[str, dict[str, float]] = {}
    skipped: list[dict[str, Any]] = []
    for row in assay_rows:
        wt = row["wildtype"]
        if len(wt) > adapter.spec.max_residues or (set(wt) - set(AA20)):
            skipped.append({"assay": row["assay"], "reason": "length_or_alphabet"})
            continue
        vals = adapter.score_variants(wt, row["mutants"], batch_size=batch_size, multi_exact_pll=False)
        score_table[row["assay"]] = {m: float(v) for m, v in zip(row["mutants"], vals)}

    tier_summaries: dict[str, Any] = {}
    for tier_name, pair_list in pairs_doc["tiers"].items():
        hits: list[float] = []
        clusters: list[int] = []
        for pair in pair_list:
            assay = pair["assay"]
            if assay not in score_table:
                continue
            sc = score_table[assay]
            mb, mw = pair["mutation_better"], pair["mutation_worse"]
            if mb not in sc or mw not in sc:
                continue
            diff = sc[mb] - sc[mw]
            hits.append(1.0 if diff > 0 else (0.5 if diff == 0 else 0.0))
            clusters.append(int(pair["cluster"]))
        tier_summaries[tier_name] = {
            "n_pairs": len(hits),
            "accuracy": cluster_bootstrap_mean(hits, clusters, seed=20261122),
        }

    return {
        "endpoint": "r1_matched_discrimination_expanded",
        "cohort_sha256": sha256_file(cohort_path),
        "n_assays_scored": len(score_table),
        "skipped": skipped,
        "elapsed_seconds": time.monotonic() - t0,
        "tiers": tier_summaries,
    }


def run_family_conditionals(
    adapter, root: Path, *, max_units: int
) -> dict[str, Any]:
    """1-pass masked conditional (for masked PLMs) or 20-string causal conditional on 4,075 contexts."""
    inputs_path = root / "results/extensions/family_conditionals_20261107/conditional-inputs.json"
    doc = json.loads(inputs_path.read_text(encoding="utf-8"))
    units = doc["units"]
    if max_units > 0:
        units = units[:max_units]

    t0 = time.monotonic()
    unit_records: list[dict[str, Any]] = []
    eval_obs: list[float] = []
    eval_minus_prof: list[float] = []
    eval_minus_markov: list[float] = []
    eval_unit_ids: list[str] = []

    for u_idx, unit in enumerate(units, 1):
        # Attach frozen training-only Profile and Markov baselines in memory using repo's exact helper.
        fc_mod._attach_baselines(unit)
        site_obs_means: list[float] = []
        site_prof_means: list[float] = []
        site_markov_means: list[float] = []
        n_contexts = 0

        for site in unit["sites"]:
            prof_logp = site["baseline"]["profile"]
            markov_logps = site["baseline"]["markov"]
            ctx_obs: list[float] = []
            ctx_prof: list[float] = []
            ctx_markov: list[float] = []
            for ctx, m_lp in zip(site["contexts"], markov_logps):
                native = ctx["native"]
                nat_idx = int(ctx["native_index"])
                obs_idx = AA_INDEX[ctx["observed"]]
                if len(native) > adapter.spec.max_residues:
                    continue
                log_q = adapter.native_site_conditional(native, nat_idx)
                ctx_obs.append(float(log_q[obs_idx]))
                ctx_prof.append(float(prof_logp[obs_idx]))
                ctx_markov.append(float(m_lp[obs_idx]))
                n_contexts += 1
            if ctx_obs:
                site_obs_means.append(float(np.mean(ctx_obs)))
                site_prof_means.append(float(np.mean(ctx_prof)))
                site_markov_means.append(float(np.mean(ctx_markov)))

        if site_obs_means:
            u_obs = float(np.mean(site_obs_means))
            u_prof = float(np.mean(site_prof_means))
            u_markov = float(np.mean(site_markov_means))
            rec = {
                "unit_id": unit["unit_id"],
                "partition": unit["partition"],
                "n_sites": len(site_obs_means),
                "n_contexts": n_contexts,
                "model_logp": u_obs,
                "profile_logp": u_prof,
                "markov_logp": u_markov,
                "minus_profile": u_obs - u_prof,
                "minus_markov": u_obs - u_markov,
            }
            unit_records.append(rec)
            if unit["partition"] == "evaluation":
                eval_obs.append(u_obs)
                eval_minus_prof.append(u_obs - u_prof)
                eval_minus_markov.append(u_obs - u_markov)
                eval_unit_ids.append(unit["unit_id"])

        if u_idx % 16 == 0 or u_idx == len(units):
            print(f"[R1-conditionals:{adapter.arm}] {u_idx}/{len(units)} units", flush=True)

    return {
        "endpoint": "r1_family_conditionals",
        "conditional_semantics": (
            "native_1pass_masked_conditional_over_aa20"
            if adapter.spec.is_masked
            else "causal_20string_score_induced_conditional"
        ),
        "inputs_sha256": sha256_file(inputs_path),
        "n_units_scored": len(unit_records),
        "n_eval_units": len(eval_unit_ids),
        "elapsed_seconds": time.monotonic() - t0,
        "evaluation_summary": {
            "model_logp": cluster_bootstrap_mean(eval_obs, eval_unit_ids, seed=20261107),
            "minus_profile": cluster_bootstrap_mean(eval_minus_prof, eval_unit_ids, seed=20261108),
            "minus_markov": cluster_bootstrap_mean(eval_minus_markov, eval_unit_ids, seed=20261109),
            "uniform_reference_nats": float(np.log(1.0 / 20.0)),
        },
        "units": unit_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", required=True, choices=list_arms())
    parser.add_argument("--dtype", default="auto", help="float32 (default) or explicit bf16 opt-in")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--saprot-mode", default="seq_only", choices=("seq_only", "foldseek_3di"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--readout-batch-size", type=int, default=8)
    parser.add_argument("--multi-exact-pll", action="store_true", help="Exact O(L) PLL on multi-mutants")
    parser.add_argument("--skip-ladder", action="store_true")
    parser.add_argument("--skip-matched", action="store_true")
    parser.add_argument("--skip-conditionals", action="store_true")
    parser.add_argument("--skip-readouts", action="store_true")
    parser.add_argument("--max-assays", type=int, default=0, help="0 = all frozen assays")
    parser.add_argument("--max-units", type=int, default=0, help="0 = all frozen conditional units")
    parser.add_argument("--expect-file", default="", help="Optional output JSON filename inside --out")
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
        "args": {
            "dtype": args.dtype,
            "saprot_mode": args.saprot_mode,
            "batch_size": args.batch_size,
            "multi_exact_pll": args.multi_exact_pll,
            "max_assays": args.max_assays,
            "max_units": args.max_units,
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not args.skip_ladder:
        payload["mutation_ladder"] = run_mutation_ladder(
            adapter,
            DATA_ROOT,
            batch_size=args.batch_size,
            multi_exact_pll=args.multi_exact_pll,
            extract_readouts=not args.skip_readouts,
            readout_batch_size=args.readout_batch_size,
            max_assays=args.max_assays,
            out_dir=args.out,
        )
    if not args.skip_matched:
        payload["matched_discrimination"] = run_matched_discrimination(
            adapter, DATA_ROOT, batch_size=args.batch_size, max_assays=args.max_assays
        )
    if not args.skip_conditionals:
        payload["family_conditionals"] = run_family_conditionals(
            adapter, DATA_ROOT, max_units=args.max_units
        )

    out_name = args.expect_file or f"r1_ladder_and_conditionals_{args.arm}.json"
    atomic_write_json(args.out / out_name, payload)
    print(f"[R1:{args.arm}] wrote {args.out / out_name}", flush=True)


if __name__ == "__main__":
    main()
