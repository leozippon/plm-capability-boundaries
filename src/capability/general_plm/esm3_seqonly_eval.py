"""CPU nested-CV evaluator for ESM3-1.4B sequence-only R1, R2, and R3 outputs.

Consumes the locally admitted GPU forward-pass outputs produced by
``scripts/capability/esm3_ext/run_esm3_r123_seqonly.py``:
  - ``r1_ladder_scores.jsonl`` (25,728 variants across 201 ProteinGym assays)
  - ``r1_family_conditionals.jsonl`` (4,328 contexts across 68 units; 64 evaluation)
  - ``r2_abundance_scores.jsonl`` (109,568 variants across 428 Domainome domains)
  - ``r2_stability_scores.jsonl`` (25,856 variants across 101 MegaScale backgrounds)
  - ``r2_phenotypes_scores.jsonl`` (14,970 rows across added phenotype cohorts)
  - ``readouts_p0.npz`` (101 wild-type ``p=0`` unmasked hidden-state summaries)
  - ``r3_cycle_epistasis.jsonl`` (8,192 MegaScale double-mutant exact-PLL cycles)

Produces apples-to-apples R1, R2, and R3 nested-CV bidirectional metrics matching
``scripts/capability/general_plm/run_r1_ladder_and_conditionals.py``,
``scripts/capability/general_plm/run_r2_phenotypes_and_readouts.py``, and
``src/capability/general_plm/r3_cycle_eval.py``.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.capability.core.amino_acids import AA20
from src.capability.extensions import family_conditionals as fc_mod
from src.capability.general_plm.eval_utils import (
    cluster_bootstrap_mean,
    nested_group_ridge_eval,
    project_four_blocks,
    safe_spearman,
    sha256_file,
    standardized_rank,
)
from src.capability.general_plm.r3_cycle_eval import (
    PairwiseCycleContext,
    evaluate_r3_pairwise_cycles_arrays,
    load_pairwise_cycle_context,
)

SCHEMA = "esm3_1_4b_seqonly_cpu_eval_v1"
AA_INDEX = {aa: idx for idx, aa in enumerate(AA20)}


def _paired_ridge_increments(
    ridge_res: dict[str, Any],
    base_name: str,
    aug_name: str,
    *,
    seed: int = 20261125,
) -> dict[str, Any]:
    """Paired cluster-bootstrap MSE reduction (base - aug) and Spearman increment (aug - base)."""
    groups = ridge_res["group_order"]
    mse_base = np.asarray(ridge_res["mean_group_mse_vector"][base_name], dtype=np.float64)
    mse_aug = np.asarray(ridge_res["mean_group_mse_vector"][aug_name], dtype=np.float64)
    rho_base = ridge_res["mean_group_spearman_vector"][base_name]
    rho_aug = ridge_res["mean_group_spearman_vector"][aug_name]

    mse_diff = mse_base - mse_aug
    rho_diffs: list[float] = []
    rho_groups: list[str] = []
    for g, rb, ra in zip(groups, rho_base, rho_aug):
        if rb is not None and ra is not None and np.isfinite(rb) and np.isfinite(ra):
            rho_diffs.append(float(ra - rb))
            rho_groups.append(g)

    return {
        "mse_reduction": cluster_bootstrap_mean(mse_diff, groups, seed=seed),
        "spearman_increment": cluster_bootstrap_mean(rho_diffs, rho_groups, seed=seed + 1),
    }


def _load_explicit_baseline_tables(
    potts_path: Path, gemme_path: Path
) -> tuple[dict[str, float], dict[str, float]]:
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


def _load_profile_freqs(npz_path: Path) -> dict[str, np.ndarray]:
    if not npz_path.is_file():
        return {}
    out: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as data:
        for key in data.files:
            if key.endswith("|frequencies"):
                out[key.removesuffix("|frequencies")] = np.asarray(data[key], dtype=np.float64)
            elif key.endswith("__frequencies") or key.startswith("freq::"):
                name = key.replace("__frequencies", "").replace("freq::", "")
                out[name] = np.asarray(data[key], dtype=np.float64)
            elif data[key].ndim == 2 and data[key].shape[1] == 20:
                out[key] = np.asarray(data[key], dtype=np.float64)
    return out


def _profile_log_odds(freqs: np.ndarray | None, pos0: int, wt_aa: str, mt_aa: str) -> float:
    if freqs is None or not (0 <= pos0 < freqs.shape[0]):
        return float("nan")
    f = np.clip(freqs[pos0], 1e-6, None)
    return float(np.log(f[AA_INDEX[mt_aa]]) - np.log(f[AA_INDEX[wt_aa]]))


def _eval_r1_ladder_and_matched(esm3_dir: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    scores_path = esm3_dir / "r1_ladder_scores.jsonl"
    vt_path = root / "results/extensions/statistical_ladder_20261106/inputs/variant_table.json"
    cohort_path = root / "results/R5/context_mutation_rescue_20260923/cohort.json"
    potts_path = root / "results/extensions/statistical_ladder_20261106/potts-explicit-complete/scores.json"
    gemme_path = root / "results/extensions/statistical_ladder_20261106/gemme/scores.json"

    score_by_sid: dict[str, float] = {}
    score_by_assay_mut: dict[tuple[str, str], float] = {}
    with scores_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            val = float(r["score_masked_marginal_additive"])
            score_by_sid[str(r["sample_id"])] = val
            score_by_assay_mut[(str(r["assay"]), str(r["mutation"]))] = val

    vt = json.loads(vt_path.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort_by_assay = {row["assay"]: row for row in cohort["assays"]}
    potts_map, gemme_map = _load_explicit_baseline_tables(potts_path, gemme_path)

    assays: dict[str, dict[str, Any]] = {}
    for sample in vt["samples"]:
        assay = sample["assay"]
        entry = assays.get(assay)
        if entry is None:
            c_row = cohort_by_assay[assay]
            entry = {
                "assay": assay,
                "cluster": int(sample["cluster"]),
                "wildtype": c_row["wildtype"],
                "mut_to_measured": {m: float(y) for m, y in zip(c_row["mutants"], c_row["measured"])},
                "samples": [],
            }
            assays[assay] = entry
        entry["samples"].append(sample)

    per_assay_rows: list[dict[str, Any]] = []
    probe_y: list[float] = []
    probe_groups: list[int] = []
    probe_subunits: list[str] = []
    probe_m: list[float] = []
    probe_p: list[float] = []
    probe_potts: list[float] = []
    probe_gemme: list[float] = []

    for assay_name in sorted(assays):
        entry = assays[assay_name]
        samples = entry["samples"]
        if not all(str(s["sample_id"]) in score_by_sid for s in samples):
            continue
        scores = np.asarray([score_by_sid[str(s["sample_id"])] for s in samples], dtype=np.float64)
        measured = np.asarray([entry["mut_to_measured"][s["mutation"]] for s in samples], dtype=np.float64)
        profile = np.asarray([float(s["profile_delta"]) for s in samples], dtype=np.float64)
        potts_vals = np.asarray([potts_map.get(s["sample_id"], np.nan) for s in samples], dtype=np.float64)
        gemme_vals = np.asarray([gemme_map.get(s["sample_id"], np.nan) for s in samples], dtype=np.float64)

        rho_m = safe_spearman(scores, measured)
        rho_p = safe_spearman(profile, measured)
        rho_mp = safe_spearman(scores, profile)
        rho_potts = safe_spearman(potts_vals, measured) if np.isfinite(potts_vals).all() else None
        rho_gemme = safe_spearman(gemme_vals, measured) if np.isfinite(gemme_vals).all() else None

        y_rank = standardized_rank(measured)
        m_rank = standardized_rank(scores)
        p_rank = standardized_rank(profile)

        per_assay_rows.append(
            {
                "assay": assay_name,
                "cluster": entry["cluster"],
                "n_variants": len(samples),
                "model_spearman": rho_m,
                "profile_spearman": rho_p,
                "potts_spearman": rho_potts,
                "gemme_spearman": rho_gemme,
                "model_minus_profile": None if (rho_m is None or rho_p is None) else float(rho_m - rho_p),
                "model_minus_potts": None if (rho_m is None or rho_potts is None) else float(rho_m - rho_potts),
                "model_minus_gemme": None if (rho_m is None or rho_gemme is None) else float(rho_m - rho_gemme),
                "model_profile_spearman": rho_mp,
            }
        )

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

    summary: dict[str, Any] = {}
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

    y_arr = np.asarray(probe_y, dtype=np.float64)
    g_arr = np.asarray(probe_groups)
    s_arr = np.asarray(probe_subunits)
    m_arr = np.asarray(probe_m, dtype=np.float64)
    p_arr = np.asarray(probe_p, dtype=np.float64)
    potts_arr = np.asarray(probe_potts, dtype=np.float64)
    gemme_arr = np.asarray(probe_gemme, dtype=np.float64)
    valid = np.isfinite(potts_arr) & np.isfinite(gemme_arr)

    ridge_blocks = {
        "P": p_arr[valid],
        "POTTS": potts_arr[valid],
        "GEMME": gemme_arr[valid],
        "M": m_arr[valid],
    }
    ridge_designs = {
        "P": ("P",),
        "P_POTTS_GEMME": ("P", "POTTS", "GEMME"),
        "M": ("M",),
        "P_POTTS_GEMME_M": ("P", "POTTS", "GEMME", "M"),
    }
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    ridge = nested_group_ridge_eval(y_arr[valid], g_arr[valid], s_arr[valid], ridge_blocks, ridge_designs)
    ridge["paired_increments"] = {
        "forward_M_given_PEG": _paired_ridge_increments(ridge, "P_POTTS_GEMME", "P_POTTS_GEMME_M", seed=20261130),
        "reverse_PEG_given_M": _paired_ridge_increments(ridge, "M", "P_POTTS_GEMME_M", seed=20261132),
    }

    ladder_out = {
        "endpoint": "r1_mutation_ladder_201",
        "n_assays": len(per_assay_rows),
        "n_families": len({r["cluster"] for r in per_assay_rows}),
        "n_variants": int(sum(r["n_variants"] for r in per_assay_rows)),
        "n_complete_case_explicit_stats": int(valid.sum()),
        "summary": summary,
        "bidirectional_ridge": {
            "n_rows": ridge["n_rows"],
            "n_groups": ridge["n_groups"],
            "summary": ridge["summary"],
            "paired_increments": ridge["paired_increments"],
        },
    }

    # Matched discrimination on the overlapping pairs present in r1_ladder_scores.jsonl
    pairs_path = (
        root / "results/R1/matched_discrimination_20261102/expanded/expanded_variants/cohort/pairs.json"
    )
    pairs_doc = json.loads(pairs_path.read_text(encoding="utf-8"))
    tiers_out: dict[str, Any] = {}
    for tier_name, pair_list in pairs_doc["tiers"].items():
        hits: list[float] = []
        clusters: list[int] = []
        for pair in pair_list:
            kb = (pair["assay"], pair["mutation_better"])
            kw = (pair["assay"], pair["mutation_worse"])
            if kb not in score_by_assay_mut or kw not in score_by_assay_mut:
                continue
            diff = score_by_assay_mut[kb] - score_by_assay_mut[kw]
            hits.append(1.0 if diff > 0 else (0.5 if diff == 0 else 0.0))
            clusters.append(int(pair["cluster"]))
        tiers_out[tier_name] = {
            "n_pairs": len(hits),
            "n_pairs_total_cohort": len(pair_list),
            "accuracy": cluster_bootstrap_mean(hits, clusters, seed=20261122),
        }

    matched_out = {
        "endpoint": "r1_matched_discrimination_overlap_from_r1_ladder",
        "coverage_note": (
            "run_esm3_r123_seqonly.py scored only the 25,728-variant R1 ladder table and not the "
            "separate expanded matched-discrimination cohort; accuracies reflect only the subset "
            "of matched pairs present in r1_ladder_scores.jsonl."
        ),
        "tiers": tiers_out,
    }
    return ladder_out, matched_out


def _eval_r1_family_conditionals(esm3_dir: Path, root: Path) -> dict[str, Any]:
    cond_path = esm3_dir / "r1_family_conditionals.jsonl"
    inputs_path = root / "results/extensions/family_conditionals_20261107/conditional-inputs.json"

    # Key: (unit_id, query_position, row_id, native_index) -> observed_logp_nats
    obs_lookup: dict[tuple[str, int, str, int], float] = {}
    with cond_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (str(r["unit_id"]), int(r["query_position"]), str(r["row_id"]), int(r["native_index"]))
            obs_lookup[key] = float(r["observed_logp_nats"])

    doc = json.loads(inputs_path.read_text(encoding="utf-8"))
    eval_obs: list[float] = []
    eval_minus_prof: list[float] = []
    eval_minus_markov: list[float] = []
    eval_unit_ids: list[str] = []

    # Collect per-site evaluation records for Q1..Q4 conservation quartiles (high->low profile entropy).
    site_records: list[dict[str, Any]] = []

    for unit in doc["units"]:
        fc_mod._attach_baselines(unit)
        uid = str(unit["unit_id"])
        site_obs_means: list[float] = []
        site_prof_means: list[float] = []
        site_markov_means: list[float] = []

        for site in unit["sites"]:
            qpos = int(site["query_position"])
            prof_logp = np.asarray(site["baseline"]["profile"], dtype=np.float64)
            prof_p = np.exp(prof_logp)
            prof_entropy = float(-np.sum(prof_p * prof_logp))
            markov_logps = site["baseline"]["markov"]

            ctx_obs: list[float] = []
            ctx_prof: list[float] = []
            ctx_markov: list[float] = []
            for ctx, m_lp in zip(site["contexts"], markov_logps):
                key = (uid, qpos, str(ctx["row_id"]), int(ctx["native_index"]))
                if key not in obs_lookup:
                    continue
                obs_idx = AA_INDEX[ctx["observed"]]
                ctx_obs.append(obs_lookup[key])
                ctx_prof.append(float(prof_logp[obs_idx]))
                ctx_markov.append(float(m_lp[obs_idx]))

            if ctx_obs:
                m_obs = float(np.mean(ctx_obs))
                m_prof = float(np.mean(ctx_prof))
                m_mk = float(np.mean(ctx_markov))
                site_obs_means.append(m_obs)
                site_prof_means.append(m_prof)
                site_markov_means.append(m_mk)
                if unit["partition"] == "evaluation":
                    site_records.append(
                        {
                            "unit_id": uid,
                            "query_position": qpos,
                            "profile_entropy_nats": prof_entropy,
                            "model_logp": m_obs,
                            "profile_logp": m_prof,
                            "markov_logp": m_mk,
                            "minus_profile": m_obs - m_prof,
                            "minus_markov": m_obs - m_mk,
                        }
                    )

        if site_obs_means and unit["partition"] == "evaluation":
            u_obs = float(np.mean(site_obs_means))
            u_prof = float(np.mean(site_prof_means))
            u_mk = float(np.mean(site_markov_means))
            eval_obs.append(u_obs)
            eval_minus_prof.append(u_obs - u_prof)
            eval_minus_markov.append(u_obs - u_mk)
            eval_unit_ids.append(uid)

    # Conservation quartiles Q1 (most variable / highest entropy) -> Q4 (most conserved / lowest entropy).
    quartiles_out: dict[str, Any] = {}
    if len(site_records) >= 4:
        ent = np.asarray([r["profile_entropy_nats"] for r in site_records], dtype=np.float64)
        # Sort descending by entropy so Q1 = highest entropy (variable), Q4 = lowest entropy (conserved).
        order = np.argsort(-ent, kind="mergesort")
        chunks = np.array_split(order, 4)
        for q_idx, idx_chunk in enumerate(chunks, 1):
            sub = [site_records[int(i)] for i in idx_chunk]
            uids = [r["unit_id"] for r in sub]
            quartiles_out[f"Q{q_idx}"] = {
                "n_sites": len(sub),
                "entropy_nats_range": [
                    float(min(r["profile_entropy_nats"] for r in sub)),
                    float(max(r["profile_entropy_nats"] for r in sub)),
                ],
                "model_logp": cluster_bootstrap_mean([r["model_logp"] for r in sub], uids, seed=20261140 + q_idx),
                "minus_profile": cluster_bootstrap_mean([r["minus_profile"] for r in sub], uids, seed=20261150 + q_idx),
                "minus_markov": cluster_bootstrap_mean([r["minus_markov"] for r in sub], uids, seed=20261160 + q_idx),
            }

    return {
        "endpoint": "r1_family_conditionals",
        "conditional_semantics": "native_1pass_masked_conditional_over_aa20",
        "n_eval_units": len(eval_unit_ids),
        "n_eval_sites": len(site_records),
        "evaluation_summary": {
            "model_logp": cluster_bootstrap_mean(eval_obs, eval_unit_ids, seed=20261107),
            "minus_profile": cluster_bootstrap_mean(eval_minus_prof, eval_unit_ids, seed=20261108),
            "minus_markov": cluster_bootstrap_mean(eval_minus_markov, eval_unit_ids, seed=20261109),
            "uniform_reference_nats": float(-math.log(20.0)),
        },
        "conservation_quartiles_var_to_cons": quartiles_out,
    }


def _eval_r2_lanes(esm3_dir: Path, root: Path) -> dict[str, Any]:
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    lanes_out: dict[str, Any] = {}

    # 1. Domainome abundance
    ab_scores: dict[str, float] = {}
    with (esm3_dir / "r2_abundance_scores.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                ab_scores[str(r["variant_id"])] = float(r["score_masked_marginal"])

    ab_cohort = json.loads(
        (root / "results/R1/external_confirmation_20260924/cohort.json").read_text(encoding="utf-8")
    )
    ab_potts, ab_gemme = _load_explicit_baseline_tables(
        root / "results/extensions/explicit_statistics_abundance_20261109/potts/scores.json",
        root / "results/extensions/explicit_statistics_abundance_20261109/gemme/scores.json",
    )
    ab_freqs = _load_profile_freqs(root / "results/R1/external_confirmation_20260924/profile_features.npz")

    ab_y, ab_m, ab_p, ab_e, ab_g, ab_grp, ab_sub = [], [], [], [], [], [], []
    ab_rhos, ab_rgrps = [], []
    for dom in ab_cohort["domains"]:
        name = str(dom["name"])
        wt = str(dom["wildtype"])
        grp = str(dom["group"])
        freqs = ab_freqs.get(name)
        d_sc, d_tg = [], []
        for idx, v in enumerate(dom["variants"]):
            vid = f"{name}:{idx}"
            if vid not in ab_scores:
                continue
            sc = ab_scores[vid]
            tg = float(v["target"])
            pos0 = int(v["position"]) - 1
            wt_aa = wt[pos0]
            mt_aa = str(v["mutant"])
            sid = f"{name}::{wt_aa}{pos0 + 1}{mt_aa}"
            d_sc.append(sc)
            d_tg.append(tg)
            ab_m.append(sc)
            ab_y.append(tg)
            ab_grp.append(grp)
            ab_sub.append(f"{name}:{pos0}")
            ab_p.append(_profile_log_odds(freqs, pos0, wt_aa, mt_aa))
            ab_e.append(ab_potts.get(sid, np.nan))
            ab_g.append(ab_gemme.get(sid, np.nan))
        rho = safe_spearman(d_sc, d_tg)
        if rho is not None:
            ab_rhos.append(rho)
            ab_rgrps.append(grp)

    ab_m_arr = np.asarray(ab_m, dtype=np.float64)
    ab_y_arr = np.asarray(ab_y, dtype=np.float64)
    ab_p_arr = np.asarray(ab_p, dtype=np.float64)
    ab_e_arr = np.asarray(ab_e, dtype=np.float64)
    ab_g_arr = np.asarray(ab_g, dtype=np.float64)
    ab_grp_arr = np.asarray(ab_grp, dtype="U")
    ab_sub_arr = np.asarray(ab_sub, dtype="U")
    ab_cc = np.isfinite(ab_m_arr) & np.isfinite(ab_y_arr) & np.isfinite(ab_e_arr) & np.isfinite(ab_g_arr)
    ab_has_p = bool(np.isfinite(ab_p_arr[ab_cc]).all()) if ab_cc.any() else False
    ab_blocks: dict[str, np.ndarray] = {"M": ab_m_arr[ab_cc], "POTTS": ab_e_arr[ab_cc], "GEMME": ab_g_arr[ab_cc]}
    ab_ctrl: tuple[str, ...] = ("POTTS", "GEMME")
    if ab_has_p:
        ab_blocks["P"] = ab_p_arr[ab_cc]
        ab_ctrl = ("P", "POTTS", "GEMME")
    ab_ridge = nested_group_ridge_eval(
        ab_y_arr[ab_cc],
        ab_grp_arr[ab_cc],
        ab_sub_arr[ab_cc],
        ab_blocks,
        {"controls": ab_ctrl, "M": ("M",), "controls_M": (*ab_ctrl, "M")},
    )
    ab_ridge_summary = {
        "n_rows": ab_ridge["n_rows"],
        "n_groups": ab_ridge["n_groups"],
        "control_blocks": list(ab_ctrl),
        "summary": ab_ridge["summary"],
        "paired_increments": {
            "forward_M_given_controls": _paired_ridge_increments(ab_ridge, "controls", "controls_M", seed=20261170),
            "reverse_controls_given_M": _paired_ridge_increments(ab_ridge, "M", "controls_M", seed=20261172),
        },
    }
    lanes_out["abundance"] = {
        "n_variants": int(ab_m_arr.size),
        "n_complete_case_explicit_stats": int(ab_cc.sum()),
        "zero_shot_spearman": cluster_bootstrap_mean(ab_rhos, ab_rgrps, seed=20261120),
        "bidirectional_ridge": ab_ridge_summary,
    }

    # 2. MegaScale stability (+ readouts_p0.npz wild-type embeddings)
    st_scores: dict[str, float] = {}
    with (esm3_dir / "r2_stability_scores.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                st_scores[str(r["variant_id"])] = float(r["score_masked_marginal"])

    readout_map: dict[str, np.ndarray] = {}
    readouts_npz = esm3_dir / "readouts_p0.npz"
    if readouts_npz.is_file():
        with np.load(readouts_npz, allow_pickle=False) as rz:
            labels = [str(x) for x in rz["labels"].tolist()]
            raw_4block = np.stack(
                [rz["middle_mean"], rz["middle_cls"], rz["final_mean"], rz["final_cls"]], axis=1
            )  # (101, 4, 1536)
            proj_wt = project_four_blocks(raw_4block)  # (101, 1024)
            readout_map = {lbl: proj_wt[i] for i, lbl in enumerate(labels)}

    st_cohort = json.loads(
        (root / "results/R4/gate_stability_20260924/cohort/cohort.json").read_text(encoding="utf-8")
    )
    st_potts, st_gemme = _load_explicit_baseline_tables(
        root / "results/extensions/explicit_statistics_pairwise_20261109/potts/scores.json",
        root / "results/extensions/explicit_statistics_pairwise_20261109/gemme/scores.json",
    )
    st_freqs = _load_profile_freqs(root / "results/R4/gate_stability_20260924/cohort/profile_features.npz")

    st_y, st_m, st_p, st_e, st_g, st_grp, st_sub = [], [], [], [], [], [], []
    st_r_rows: list[np.ndarray] = []
    st_rhos, st_rgrps = [], []
    for bg in st_cohort["backgrounds"]:
        name = str(bg["name"])
        wt = str(bg["wildtype"])
        grp = str(bg["group"])
        freqs = st_freqs.get(name)
        r_vec = readout_map.get(name)
        b_sc, b_tg = [], []
        for idx, v in enumerate(bg["variants"]):
            vid = f"{name}:{idx}"
            if vid not in st_scores:
                continue
            sc = st_scores[vid]
            tg = float(v["ddg"])
            pos0 = int(v["position"]) - 1
            wt_aa = wt[pos0]
            mt_aa = str(v["mutant"])
            sid = f"{name}::{wt_aa}{pos0 + 1}{mt_aa}"
            b_sc.append(sc)
            b_tg.append(tg)
            st_m.append(sc)
            st_y.append(tg)
            st_grp.append(grp)
            st_sub.append(f"{name}:{pos0}")
            st_p.append(_profile_log_odds(freqs, pos0, wt_aa, mt_aa))
            st_e.append(st_potts.get(sid, np.nan))
            st_g.append(st_gemme.get(sid, np.nan))
            if r_vec is not None:
                st_r_rows.append(r_vec)
        rho = safe_spearman(b_sc, b_tg)
        if rho is not None:
            st_rhos.append(rho)
            st_rgrps.append(grp)

    st_m_arr = np.asarray(st_m, dtype=np.float64)
    st_y_arr = np.asarray(st_y, dtype=np.float64)
    st_p_arr = np.asarray(st_p, dtype=np.float64)
    st_e_arr = np.asarray(st_e, dtype=np.float64)
    st_g_arr = np.asarray(st_g, dtype=np.float64)
    st_grp_arr = np.asarray(st_grp, dtype="U")
    st_sub_arr = np.asarray(st_sub, dtype="U")
    st_cc = np.isfinite(st_m_arr) & np.isfinite(st_y_arr) & np.isfinite(st_e_arr) & np.isfinite(st_g_arr)
    st_has_p = bool(np.isfinite(st_p_arr[st_cc]).all()) if st_cc.any() else False

    st_blocks: dict[str, np.ndarray] = {"M": st_m_arr[st_cc], "POTTS": st_e_arr[st_cc], "GEMME": st_g_arr[st_cc]}
    st_ctrl: tuple[str, ...] = ("POTTS", "GEMME")
    if st_has_p:
        st_blocks["P"] = st_p_arr[st_cc]
        st_ctrl = ("P", "POTTS", "GEMME")
    st_designs: dict[str, tuple[str, ...]] = {
        "controls": st_ctrl,
        "M": ("M",),
        "controls_M": (*st_ctrl, "M"),
    }
    has_r_wt = len(st_r_rows) == len(st_m) and len(st_r_rows) > 0
    if has_r_wt:
        r_mat = np.asarray(st_r_rows, dtype=np.float64)
        st_blocks["R"] = r_mat[st_cc]
        st_designs["R"] = ("R",)
        st_designs["controls_R"] = (*st_ctrl, "R")
        st_designs["controls_M_R"] = (*st_ctrl, "M", "R")

    st_ridge = nested_group_ridge_eval(
        st_y_arr[st_cc], st_grp_arr[st_cc], st_sub_arr[st_cc], st_blocks, st_designs
    )
    st_paired = {
        "forward_M_given_controls": _paired_ridge_increments(st_ridge, "controls", "controls_M", seed=20261180),
        "reverse_controls_given_M": _paired_ridge_increments(st_ridge, "M", "controls_M", seed=20261182),
    }
    if has_r_wt:
        st_paired["forward_R_wt_given_controls"] = _paired_ridge_increments(
            st_ridge, "controls", "controls_R", seed=20261184
        )
        st_paired["reverse_controls_given_R_wt"] = _paired_ridge_increments(
            st_ridge, "R", "controls_R", seed=20261186
        )
        st_paired["forward_M_R_wt_given_controls"] = _paired_ridge_increments(
            st_ridge, "controls", "controls_M_R", seed=20261188
        )

    lanes_out["stability"] = {
        "n_variants": int(st_m_arr.size),
        "n_complete_case_explicit_stats": int(st_cc.sum()),
        "zero_shot_spearman": cluster_bootstrap_mean(st_rhos, st_rgrps, seed=20261120),
        "readouts_granularity": (
            "wildtype_background_only_101_wts_broadcast_to_variants_not_per_variant_mutant_delta"
            if has_r_wt
            else "absent"
        ),
        "bidirectional_ridge": {
            "n_rows": st_ridge["n_rows"],
            "n_groups": st_ridge["n_groups"],
            "control_blocks": list(st_ctrl),
            "summary": st_ridge["summary"],
            "paired_increments": st_paired,
        },
    }

    # 3. Added phenotypes
    ph_scores: dict[str, float] = {}
    with (esm3_dir / "r2_phenotypes_scores.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                ph_scores[str(r["variant_id"])] = float(r["score_masked_marginal"])

    rows_dir = root / "results/extensions/phenotype_breadth_20261008/qual-3/rows"
    grouped_ph: dict[str, dict[str, Any]] = {}
    for jf in sorted(rows_dir.glob("*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rid = str(r["row_id"])
            if rid not in ph_scores:
                continue
            cohort_name = str(r["cohort"])
            bg_name = str(r["background"])
            grp = str(r.get("group") or r.get("unit") or r.get("cluster") or bg_name)
            label_val = (
                float(r["oriented_label"])
                if "oriented_label" in r and r["oriented_label"] is not None
                else float(r["label"]) * float(r.get("direction", 1))
            )
            key = f"{cohort_name}::{bg_name}"
            entry = grouped_ph.setdefault(
                key, {"cohort": cohort_name, "group": grp, "scores": [], "targets": []}
            )
            entry["scores"].append(ph_scores[rid])
            entry["targets"].append(label_val)

    ph_rhos, ph_rgrps = [], []
    by_cohort: dict[str, tuple[list[float], list[str]]] = {}
    n_ph_vars = 0
    for key in sorted(grouped_ph):
        entry = grouped_ph[key]
        n_ph_vars += len(entry["scores"])
        rho = safe_spearman(entry["scores"], entry["targets"])
        if rho is not None:
            ph_rhos.append(rho)
            ph_rgrps.append(entry["group"])
            c_rhos, c_grps = by_cohort.setdefault(entry["cohort"], ([], []))
            c_rhos.append(rho)
            c_grps.append(entry["group"])

    lanes_out["phenotype"] = {
        "n_variants": n_ph_vars,
        "n_backgrounds": len(grouped_ph),
        "zero_shot_spearman": cluster_bootstrap_mean(ph_rhos, ph_rgrps, seed=20261120),
        "by_cohort_zero_shot_spearman": {
            c: cluster_bootstrap_mean(rs, gs, seed=20261120) for c, (rs, gs) in sorted(by_cohort.items())
        },
        "bidirectional_ridge": None,
    }
    return lanes_out


def _eval_r3_cycles(
    esm3_dir: Path,
    root: Path,
    *,
    ctx: PairwiseCycleContext | None = None,
    bootstrap_draws: int = 2000,
) -> dict[str, Any]:
    cycles_path = esm3_dir / "r3_cycle_epistasis.jsonl"
    if ctx is None:
        ctx = load_pairwise_cycle_context(root)

    rows = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_keys = [
        (bg["name"], cyc["positions"], cyc["separation"])
        for bg in ctx.plan["backgrounds"]
        for cyc in bg["cycles"]
    ]
    observed_keys = [(r["background"], r["positions"], r["separation"]) for r in rows]
    if observed_keys != expected_keys:
        raise ValueError(
            f"{cycles_path}: cycle ordering/keys do not match frozen extraction_plan.json "
            f"(got {len(observed_keys)} rows vs expected {len(expected_keys)})"
        )

    eps_all = np.asarray([float(r["epsilon_pll_nats"]) for r in rows], dtype=np.float64)
    eval_doc = evaluate_r3_pairwise_cycles_arrays(
        eps_all,
        "esm3-1.4b",
        ctx=ctx,
        r_cycle_all=None,
        source_path=str(cycles_path),
        source_sha256=sha256_file(cycles_path),
        bootstrap_draws=bootstrap_draws,
    )
    by_sep: dict[str, Any] = {}
    for sep_label in sorted(set(ctx.raw_all_separations.tolist())):
        mask = ctx.raw_all_separations == sep_label
        by_sep[sep_label] = {
            "n_cycles": int(mask.sum()),
            "mean_abs_eps_pll_nats": float(np.mean(np.abs(eps_all[mask]))),
            "std_eps_pll_nats": float(np.std(eps_all[mask])),
        }
    return {
        "endpoint": "r3_megascale_pairwise_cycles_exact_pll",
        "n_backgrounds": len(ctx.plan["backgrounds"]),
        "n_cycles": int(eps_all.size),
        "nonzero_cycle_fraction": float(np.mean(np.abs(eps_all) > 1e-8)),
        "mean_abs_eps_pll_nats": float(np.mean(np.abs(eps_all))),
        "max_abs_eps_pll_nats": float(np.max(np.abs(eps_all))),
        "by_separation": by_sep,
        "zero_shot_spearman_vs_measured_epsilon": (
            eval_doc["zero_shot"]["all_64_groups_raw_epsilon"]["within_background_spearman"]
        ),
        "nested_cv_epistasis": eval_doc,
    }


def evaluate_esm3_seqonly_dir(
    esm3_dir: Path,
    root: Path,
    *,
    ctx: PairwiseCycleContext | None = None,
    bootstrap_draws: int = 2000,
) -> dict[str, Any]:
    """Run complete CPU nested-CV evaluation on an admitted ``esm3_1_4b_r123_b4`` directory."""
    meta_path = esm3_dir / "esm3_r123_seqonly.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"ESM3 seqonly metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    r1_ladder, r1_matched = _eval_r1_ladder_and_matched(esm3_dir, root)
    r1_cond = _eval_r1_family_conditionals(esm3_dir, root)
    r2_lanes = _eval_r2_lanes(esm3_dir, root)
    r3_cycles = _eval_r3_cycles(esm3_dir, root, ctx=ctx, bootstrap_draws=bootstrap_draws)

    return {
        "schema": SCHEMA,
        "arm": "esm3-1.4b",
        "family": "esm3",
        "paradigm": "multimodal_seqonly",
        "approx_params": int(meta.get("model_facts", {}).get("n_parameters", 1_401_735_748)),
        "source_dir": str(esm3_dir),
        "source_meta_sha256": sha256_file(meta_path),
        "dtype_realised": meta.get("model_facts", {}).get("dtype_active"),
        "r1_ladder_and_conditionals": {
            "mutation_ladder": r1_ladder,
            "matched_discrimination": r1_matched,
            "family_conditionals": r1_cond,
        },
        "r2_phenotypes_and_readouts": {
            "lanes": r2_lanes,
        },
        "r3_pairwise_and_contacts": {
            "pairwise_cycles": r3_cycles,
            "structural_contacts": None,
        },
        "b4_vqvae": meta.get("results", {}).get("b4_vqvae"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
