"""Reusable benchmark runner, reference leaderboard ranker, and sealed report writer.

Enforces strict decoupling between:
* ``mode="smoke"`` (``status="smoke_ok"``, ``evaluation_scope="contract_smoke_only"``):
  fast contract/shape verification on tiny fixtures; NEVER computes or attaches
  ``reference_comparison`` ranks or percentiles against sealed leaderboards.
* ``mode="full"`` (``evaluation_scope="formal_full_cohort"``):
  loads ONLY the exact real frozen evaluation cohorts under ``root`` (never
  falling back to smoke subsamples, hardcoded toy sequences, or synthetic
  coordinates), returning ``status="not_configured"`` with ``missing_paths``
  and NO ``reference_comparison`` whenever any required cohort asset is absent.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.capability.benchmark.protocol import (
    AA20_INDEX,
    AA20_SET,
    CapabilityNotSupportedError,
    ProteinBenchmarkAdapter,
    _synthetic_alpha_helix_coords,
    validate_adapter,
)
from src.capability.unified import r4_scoring_yardstick as r4_mod
from src.capability.unified.master_summary import FORBIDDEN_INFRA_RE
from src.capability.unified.roster import REPO_ROOT, UNIFIED_OUTPUT_DIR

ALL_SUITES: tuple[str, ...] = (
    "validate",
    "r1_dms",
    "r2_biophysical",
    "r3_epistasis",
    "r3_contacts",
    "r4_family_memory",
    "r4_generation",
    "r5_context",
)

# Canonical relative paths of the real frozen evaluation cohorts under `root`.
R1_VARIANT_TABLE_REL = Path("results/extensions/statistical_ladder_20261106/inputs/variant_table.json")
R1_COHORT_REL = Path("results/R5/context_mutation_rescue_20260923/cohort.json")
R2_MEGASCALE_COHORT_REL = Path("results/R4/gate_stability_20260924/cohort/cohort.json")
R3_EPISTASIS_PLAN_REL = Path("archive/logs/R3/pairwise_epistasis_20260924/extraction_plan.json")
R3_CONTACTS_SITES_REL = Path("results/extensions/mutation_structure_20261006/structure/annotation/sites.json.gz")
R4_FAMILY_COND_REL = Path("results/extensions/family_conditionals_20261107/conditional-inputs.json")
R4_GEN_BACKBONES_REL = Path("results/audit_20261011_h200/ladder_original/ladder_backbones.jsonl")
R5_CONTEXT_COHORT_REL = Path(r4_mod.DEFAULT_COHORT_REL)
WT_BACKBONES_3D_REL = UNIFIED_OUTPUT_DIR / "wt_backbones_3d"


@dataclass(frozen=True)
class ReferenceLeaderboard:
    """Reference distribution for one headline metric across the 62–84 unified models."""

    suite: str
    metric_name: str
    higher_is_better: bool
    source_file: str
    scores_by_arm: dict[str, float]


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_a = np.asarray(x, dtype=np.float64)
    y_a = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_a) & np.isfinite(y_a)
    if mask.sum() < 3:
        return float("nan")
    if np.std(x_a[mask]) == 0.0 or np.std(y_a[mask]) == 0.0:
        return 0.0
    val = float(spearmanr(x_a[mask], y_a[mask]).statistic)
    return val if math.isfinite(val) else 0.0


def _binary_auroc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return wins / float(len(pos) * len(neg))


def compute_rank_and_percentile(
    value: float,
    reference: ReferenceLeaderboard,
) -> dict[str, Any]:
    """Compute exact 1-based rank and percentile of ``value`` against the reference cohort."""
    vals = [v for v in reference.scores_by_arm.values() if math.isfinite(v)]
    n_ref = len(vals)
    if not math.isfinite(value) or n_ref == 0:
        return {
            "metric": reference.metric_name,
            "value": None if not math.isfinite(value) else float(value),
            "n_reference_models": n_ref,
            "rank": None,
            "percentile": None,
            "source_file": reference.source_file,
        }
    if reference.higher_is_better:
        better = sum(1 for v in vals if v > value)
        worse_or_eq = sum(1 for v in vals if v <= value)
        top_arm, top_val = max(reference.scores_by_arm.items(), key=lambda kv: kv[1])
    else:
        better = sum(1 for v in vals if v < value)
        worse_or_eq = sum(1 for v in vals if v >= value)
        top_arm, top_val = min(reference.scores_by_arm.items(), key=lambda kv: kv[1])
    return {
        "metric": reference.metric_name,
        "value": float(value),
        "higher_is_better": reference.higher_is_better,
        "n_reference_models": n_ref,
        "rank": int(better + 1),
        "percentile": round(100.0 * float(worse_or_eq) / float(n_ref), 2),
        "reference_median": float(np.median(vals)),
        "reference_top_arm": top_arm,
        "reference_top_value": float(top_val),
        "source_file": reference.source_file,
    }


def _extract_point_map(mapping: Any) -> dict[str, float]:
    if not isinstance(mapping, dict):
        return {}
    out: dict[str, float] = {}
    for arm, info in mapping.items():
        if isinstance(info, (int, float)) and math.isfinite(float(info)):
            out[str(arm)] = float(info)
        elif isinstance(info, dict):
            for key in ("point_nats", "point", "mean", "auc_minus_half", "M_alone_auc"):
                val = info.get(key)
                if val is not None and math.isfinite(float(val)):
                    out[str(arm)] = float(val)
                    break
    return out


def load_reference_leaderboards(unified_dir: Path | str | None = None) -> dict[str, ReferenceLeaderboard]:
    """Load sealed 62–84 model reference leaderboards (supporting both ``*_full.json`` and ``*_62/65.json``)."""
    udir = Path(unified_dir) if unified_dir is not None else (REPO_ROOT / UNIFIED_OUTPUT_DIR)
    boards: dict[str, ReferenceLeaderboard] = {}

    for r1_name in ("r1_proteingym_unified_full.json", "r1_proteingym_unified_62.json"):
        p_r1 = udir / r1_name
        if not p_r1.is_file():
            continue
        doc = json.loads(p_r1.read_text(encoding="utf-8"))
        l1 = doc.get("layer1_zero_shot", {})
        scores = _extract_point_map(l1.get("full_201_assays_primary", {}).get("arms", {}).get("columns", {}))
        if not scores:
            sup = l1.get("supports", {})
            block = sup.get("clusters_163") or (next(iter(sup.values())) if sup else {})
            scores = _extract_point_map(block.get("models", {}))
        if scores:
            boards["r1_dms"] = ReferenceLeaderboard("r1_dms", "mean_spearman", True, p_r1.name, scores)
            break

    for r2_name in ("r2_biophysical_unified_full.json", "r2_biophysical_unified_62.json"):
        p_r2 = udir / r2_name
        if not p_r2.is_file():
            continue
        doc = json.loads(p_r2.read_text(encoding="utf-8"))
        ds = doc.get("datasets", {})
        scores = _extract_point_map(
            ds.get("megascale_79_cc", {}).get("zero_shot", {}).get("arms_62", {}).get("columns", {})
        )
        if not scores:
            zs = ds.get("megascale", {}).get("zero_shot_spearman", {})
            sup = zs.get("cc_79") or (next(iter(zs.values())) if zs else {})
            for v in sup.values():
                if isinstance(v, dict) and "columns" in v:
                    scores = _extract_point_map(v["columns"])
                    if scores:
                        break
        if scores:
            boards["r2_biophysical"] = ReferenceLeaderboard("r2_biophysical", "megascale_spearman", True, p_r2.name, scores)
            break

    for r3_name in ("r3_epistasis_and_contacts_unified_full.json", "r3_epistasis_and_contacts_unified_62.json"):
        p_r3 = udir / r3_name
        if not p_r3.is_file():
            continue
        doc = json.loads(p_r3.read_text(encoding="utf-8"))
        epi = doc.get("epistasis", {})
        scores = _extract_point_map(epi.get("zero_shot_raw_cc_61", {}).get("arms", {}).get("columns", {}))
        if not scores:
            scores = _extract_point_map(epi.get("zero_shot_epistasis_62", {}).get("ZS_raw_cc_61", {}).get("arms", {}))
        if scores:
            boards["r3_epistasis"] = ReferenceLeaderboard("r3_epistasis", "epistasis_spearman", True, p_r3.name, scores)
            break

    c_scores: dict[str, float] = {}
    c_src: str | None = None
    p_c62 = udir / "contacts_unified_62.json"
    if p_c62.is_file():
        doc62 = json.loads(p_c62.read_text(encoding="utf-8"))
        for arm, info in doc62.get("per_arm", {}).items():
            val = info.get("strata", {}).get("all", {}).get("all_admitted", {}).get("M_alone_auc")
            if val is not None and math.isfinite(float(val)):
                c_scores[str(arm)] = float(val)
        if c_scores:
            c_src = p_c62.name
    p_cfull = udir / "contacts_unified_full.json"
    if p_cfull.is_file():
        docf = json.loads(p_cfull.read_text(encoding="utf-8"))
        new_minus_half = _extract_point_map(docf.get("depth_strata_new_arms", {}).get("all", {}).get("new_arms_auc_minus_half", {}))
        for arm, delta in new_minus_half.items():
            c_scores[arm] = float(0.5 + delta)
        if new_minus_half:
            c_src = p_cfull.name
    if c_scores and c_src is not None:
        boards["r3_contacts"] = ReferenceLeaderboard("r3_contacts", "contact_auroc", True, c_src, c_scores)

    for fm_name in ("family_memory_unified_full.json", "family_memory_unified_62.json"):
        p_fm = udir / fm_name
        if not p_fm.is_file():
            continue
        doc = json.loads(p_fm.read_text(encoding="utf-8"))
        ov = (
            doc.get("family_conditionals", {}).get("strata", {}).get("overall_64")
            or doc.get("family_conditionals_62", {}).get("simultaneous_bands_by_stratum", {}).get("overall_64")
            or {}
        )
        block = ov.get("model_absolute_logp", {})
        scores = _extract_point_map(block.get("per_arm") or block.get("columns") or {})
        if scores:
            boards["r4_family_memory"] = ReferenceLeaderboard("r4_family_memory", "mean_logp_nats", True, p_fm.name, scores)
            break

    r5_scores: dict[str, float] = {}
    r5_src: str | None = None
    p_r4_65 = udir / "r4_scoring_unified_65.json"
    if p_r4_65.is_file():
        doc65 = json.loads(p_r4_65.read_text(encoding="utf-8"))
        for arm, info in doc65.get("per_arm", {}).items():
            val = info.get("bands", {}).get("id_30_50", {}).get("auroc_hom_vs_unrelated", {}).get("mean")
            if val is not None and math.isfinite(float(val)):
                r5_scores[str(arm)] = float(val)
        if r5_scores:
            r5_src = p_r4_65.name
    p_r4_full = udir / "r4_scoring_unified_full.json"
    if p_r4_full.is_file():
        docf = json.loads(p_r4_full.read_text(encoding="utf-8"))
        hbu = _extract_point_map(
            docf.get("metrics_by_band", {}).get("id_30_50", {}).get("hom_beats_unrelated", {}).get("bands", {}).get("columns", {})
        )
        for arm, val in hbu.items():
            r5_scores.setdefault(arm, float(val))
        if hbu:
            r5_src = p_r4_full.name
    if r5_scores and r5_src is not None:
        boards["r5_context"] = ReferenceLeaderboard("r5_context", "auroc_hom_vs_unrelated_30_50", True, r5_src, r5_scores)

    return boards


# --------------------------------------------------------------------------- #
# Helpers: Strict Full-Cohort Missing-Asset Checks & 3D Backbone Lookup
# --------------------------------------------------------------------------- #


def _not_configured(reason: str, missing_paths: Sequence[Path | str]) -> dict[str, Any]:
    return {
        "status": "not_configured",
        "mode": "full",
        "evaluation_scope": "formal_full_cohort",
        "reason": reason,
        "missing_paths": [str(p) for p in missing_paths],
    }


def _check_required_paths(root: Path, rel_paths: Sequence[Path]) -> list[Path]:
    return [root / rp for rp in rel_paths if not (root / rp).exists()]


def _lookup_real_backbone_coords(seq: str, root: Path) -> np.ndarray | None:
    """Strictly look up cached WT 3D backbone in ``wt_backbones_3d`` (NEVER synthesizes)."""
    bb_dir = root / WT_BACKBONES_3D_REL
    if not bb_dir.is_dir():
        return None
    digest = hashlib.sha256(seq.encode("utf-8")).hexdigest()
    for cand in (bb_dir / digest[:2] / f"{digest}.npz", bb_dir / f"{digest}.npz"):
        if cand.is_file():
            d = np.load(cand, allow_pickle=True)
            coords = np.asarray(d["coords"], dtype=np.float32)
            if coords.shape[0] == len(seq):
                return coords
    return None


def _smoke_coords_if_needed(adapter: ProteinBenchmarkAdapter, seq: str) -> np.ndarray | None:
    """Only permitted inside ``mode='smoke'`` contract checks."""
    if not adapter.capabilities.requires_backbone_3d:
        return None
    return _synthetic_alpha_helix_coords(len(seq))


def _diff_to_mutation_string(wt: str, mut: str) -> str:
    if len(wt) != len(mut):
        raise ValueError(f"Sequence length mismatch ({len(wt)} vs {len(mut)})")
    toks = [f"{w}{i + 1}{m}" for i, (w, m) in enumerate(zip(wt, mut)) if w != m]
    return ":".join(toks)


# --------------------------------------------------------------------------- #
# Individual Suite Evaluators (Strict smoke vs full Decoupling)
# --------------------------------------------------------------------------- #


def _run_suite_r1_dms(
    adapter: ProteinBenchmarkAdapter,
    *,
    mode: str,
    root: Path,
    leaderboards: dict[str, ReferenceLeaderboard],
) -> dict[str, Any]:
    if not adapter.capabilities.score_mutations:
        return {"status": "not_supported", "reason": "Adapter capability score_mutations=False"}

    if mode == "smoke":
        wt = "MKTAYIAKQRQISFVKSHFSRQLE"
        muts = ["M1A", "K2E", "T3S", "A4G", "Y5F", "I6V"]
        ys = [-0.8, -0.4, 0.1, -0.2, 0.3, 0.2]
        coords = _smoke_coords_if_needed(adapter, wt)
        preds = np.asarray(adapter.score_mutations(wt, muts, coords=coords), dtype=np.float64)
        return {
            "status": "smoke_ok",
            "mode": "smoke",
            "evaluation_scope": "contract_smoke_only",
            "n_assays": 1,
            "mean_spearman": _safe_spearman(preds, np.asarray(ys, dtype=np.float64)),
        }

    # Formal full cohort: exact frozen ProteinGym variant_table.json + cohort.json
    missing = _check_required_paths(root, [R1_VARIANT_TABLE_REL, R1_COHORT_REL])
    if adapter.capabilities.requires_backbone_3d and not (root / WT_BACKBONES_3D_REL).is_dir():
        missing.append(root / WT_BACKBONES_3D_REL)
    if missing:
        return _not_configured("Required frozen ProteinGym R1 cohort files are missing", missing)

    vt = json.loads((root / R1_VARIANT_TABLE_REL).read_text(encoding="utf-8"))
    cohort = json.loads((root / R1_COHORT_REL).read_text(encoding="utf-8"))
    cohort_by_assay = {r["assay"]: r for r in cohort["assays"]}

    assay_muts: dict[str, dict[str, Any]] = {}
    for sample in vt["samples"]:
        aname = sample["assay"]
        if aname not in cohort_by_assay:
            continue
        entry = assay_muts.get(aname)
        if entry is None:
            crow = cohort_by_assay[aname]
            entry = {
                "cluster": int(sample["cluster"]),
                "wildtype": str(crow["wildtype"]).strip().upper(),
                "mut_to_y": {str(m).strip().upper(): float(y) for m, y in zip(crow["mutants"], crow["measured"])},
                "muts": [],
                "ys": [],
            }
            assay_muts[aname] = entry
        m = str(sample["mutant"]).strip().upper()
        if m in entry["mut_to_y"]:
            entry["muts"].append(m)
            entry["ys"].append(entry["mut_to_y"][m])

    by_cluster: dict[int, list[float]] = defaultdict(list)
    per_assay: dict[str, float] = {}
    missing_bbs: list[str] = []
    for aname, entry in sorted(assay_muts.items()):
        wt = entry["wildtype"]
        coords: np.ndarray | None = None
        if adapter.capabilities.requires_backbone_3d:
            coords = _lookup_real_backbone_coords(wt, root)
            if coords is None:
                missing_bbs.append(f"wt_backbone:{aname}")
                continue
        preds = np.asarray(adapter.score_mutations(wt, entry["muts"], coords=coords), dtype=np.float64)
        rho = _safe_spearman(preds, np.asarray(entry["ys"], dtype=np.float64))
        per_assay[aname] = rho
        if math.isfinite(rho):
            by_cluster[entry["cluster"]].append(rho)

    if missing_bbs:
        return _not_configured("Missing real WT 3D backbones for R1 ProteinGym assays", missing_bbs[:10])
    if not by_cluster:
        return _not_configured("No valid R1 ProteinGym assays could be evaluated", [root / R1_VARIANT_TABLE_REL])

    # Equal assays within family, equal families (matching r1_proteingym_unified_62.json)
    cluster_means = [float(np.mean(v)) for v in by_cluster.values()]
    mean_rho = float(np.mean(cluster_means))
    res: dict[str, Any] = {
        "status": "ok",
        "mode": "full",
        "evaluation_scope": "formal_full_cohort",
        "n_assays": len(per_assay),
        "n_clusters": len(by_cluster),
        "mean_spearman": mean_rho,
        "per_assay_spearman": per_assay,
    }
    if "r1_dms" in leaderboards:
        res["reference_comparison"] = compute_rank_and_percentile(mean_rho, leaderboards["r1_dms"])
    return res


def _run_suite_r2_biophysical(
    adapter: ProteinBenchmarkAdapter,
    *,
    mode: str,
    root: Path,
    leaderboards: dict[str, ReferenceLeaderboard],
) -> dict[str, Any]:
    if not (adapter.capabilities.score_mutations or adapter.capabilities.extract_representations_1024):
        return {
            "status": "not_supported",
            "reason": "Adapter requires score_mutations or extract_representations_1024",
        }

    if mode == "smoke":
        wt = "MKTAYIAKQRQISFVKSHFSRQLE"
        coords = _smoke_coords_if_needed(adapter, wt)
        res_s: dict[str, Any] = {
            "status": "smoke_ok",
            "mode": "smoke",
            "evaluation_scope": "contract_smoke_only",
            "n_domains": 1,
        }
        if adapter.capabilities.score_mutations:
            preds = np.asarray(
                adapter.score_mutations(wt, ["M1A", "K2E", "T3S", "A4G"], coords=coords), dtype=np.float64
            )
            res_s["smoke_spearman"] = _safe_spearman(preds, np.array([-0.5, -0.2, 0.1, 0.3]))
        if adapter.capabilities.extract_representations_1024:
            c_list = [coords] if coords is not None else None
            reps = adapter.extract_representations_1024([wt], coords_list=c_list)
            res_s["readout_1024_shape"] = list(reps.shape)
        return res_s

    # Formal full cohort: real frozen MegaScale stability cohort (101 backgrounds)
    missing = _check_required_paths(root, [R2_MEGASCALE_COHORT_REL])
    if adapter.capabilities.requires_backbone_3d and not (root / WT_BACKBONES_3D_REL).is_dir():
        missing.append(root / WT_BACKBONES_3D_REL)
    if missing:
        return _not_configured("Required frozen MegaScale R2 cohort file is missing", missing)

    ms_cohort = json.loads((root / R2_MEGASCALE_COHORT_REL).read_text(encoding="utf-8"))
    bgs = ms_cohort.get("backgrounds", [])
    if not bgs:
        return _not_configured("MegaScale R2 cohort contains no backgrounds", [root / R2_MEGASCALE_COHORT_REL])

    missing_bbs: list[str] = []
    wt_seqs: list[str] = []
    wt_coords: list[np.ndarray | None] = []
    rhos_by_group: dict[str, list[float]] = defaultdict(list)
    per_bg: dict[str, float] = {}

    for bg in bgs:
        wt = str(bg["wildtype"]).strip().upper()
        coords: np.ndarray | None = None
        if adapter.capabilities.requires_backbone_3d:
            coords = _lookup_real_backbone_coords(wt, root)
            if coords is None:
                missing_bbs.append(f"wt_backbone:{bg['name']}")
                continue
        wt_seqs.append(wt)
        wt_coords.append(coords)
        if adapter.capabilities.score_mutations:
            muts = [f"{wt[int(v['position']) - 1]}{int(v['position'])}{v['mutant']}" for v in bg["variants"]]
            targets = np.asarray([float(v["ddg"]) for v in bg["variants"]], dtype=np.float64)
            preds = np.asarray(adapter.score_mutations(wt, muts, coords=coords), dtype=np.float64)
            rho = _safe_spearman(preds, targets)
            per_bg[str(bg["name"])] = rho
            if math.isfinite(rho):
                rhos_by_group[str(bg["group"])].append(rho)

    if missing_bbs:
        return _not_configured("Missing real WT 3D backbones for MegaScale R2 domains", missing_bbs[:10])

    res: dict[str, Any] = {
        "status": "ok",
        "mode": "full",
        "evaluation_scope": "formal_full_cohort",
        "n_domains": len(bgs),
        "n_groups": len(rhos_by_group),
    }
    if adapter.capabilities.score_mutations:
        mean_rho = float(np.mean([float(np.mean(v)) for v in rhos_by_group.values()])) if rhos_by_group else float("nan")
        res["megascale_spearman"] = mean_rho
        res["per_background_spearman"] = per_bg
        if "r2_biophysical" in leaderboards:
            res["reference_comparison"] = compute_rank_and_percentile(mean_rho, leaderboards["r2_biophysical"])
    if adapter.capabilities.extract_representations_1024 and wt_seqs:
        c_arg = wt_coords if adapter.capabilities.requires_backbone_3d else None
        reps = adapter.extract_representations_1024(wt_seqs, coords_list=c_arg)
        res["readout_1024_shape"] = list(reps.shape)
        res["r_mid_norm_mean"] = float(np.linalg.norm(reps[:, :512], axis=1).mean())
        res["r_last_norm_mean"] = float(np.linalg.norm(reps[:, 512:], axis=1).mean())
    return res


def _run_suite_r3_epistasis(
    adapter: ProteinBenchmarkAdapter,
    *,
    mode: str,
    root: Path,
    leaderboards: dict[str, ReferenceLeaderboard],
) -> dict[str, Any]:
    if not adapter.capabilities.score_mutations:
        return {"status": "not_supported", "reason": "Adapter capability score_mutations=False"}

    if mode == "smoke":
        wt = "MKTAYIAKQRQISFVKSHFSRQLE"
        coords = _smoke_coords_if_needed(adapter, wt)
        cycles = [("M1A", "K2E"), ("T3S", "A4G"), ("Y5F", "I6V")]
        mut_list = [x for m1, m2 in cycles for x in (m1, m2, f"{m1}:{m2}")]
        sc = np.asarray(adapter.score_mutations(wt, mut_list, coords=coords), dtype=np.float64)
        pred_epi = [float(sc[3 * i + 2] - sc[3 * i] - sc[3 * i + 1]) for i in range(len(cycles))]
        return {
            "status": "smoke_ok",
            "mode": "smoke",
            "evaluation_scope": "contract_smoke_only",
            "n_cycles": len(cycles),
            "mean_abs_predicted_epistasis": float(np.mean(np.abs(pred_epi))),
        }

    # Formal full cohort: real frozen MegaScale double-mutant cycles (extraction_plan.json)
    missing = _check_required_paths(root, [R3_EPISTASIS_PLAN_REL])
    if adapter.capabilities.requires_backbone_3d and not (root / WT_BACKBONES_3D_REL).is_dir():
        missing.append(root / WT_BACKBONES_3D_REL)
    if missing:
        return _not_configured("Required frozen R3 epistasis extraction_plan.json is missing", missing)

    plan = json.loads((root / R3_EPISTASIS_PLAN_REL).read_text(encoding="utf-8"))
    bgs = plan.get("backgrounds", [])
    if not bgs:
        return _not_configured("R3 epistasis extraction_plan.json contains no backgrounds", [root / R3_EPISTASIS_PLAN_REL])

    missing_bbs: list[str] = []
    rhos_by_group: dict[str, list[float]] = defaultdict(list)
    total_cycles = 0
    has_measured_targets = False

    for bg in bgs:
        seqs: list[str] = bg["sequences"]
        wt = str(seqs[0]).strip().upper()
        coords: np.ndarray | None = None
        if adapter.capabilities.requires_backbone_3d:
            coords = _lookup_real_backbone_coords(wt, root)
            if coords is None:
                missing_bbs.append(f"wt_backbone:{bg['name']}")
                continue

        # Score every non-WT state relative to WT once via score_mutations
        mut_strs = [_diff_to_mutation_string(wt, str(s).strip().upper()) for s in seqs[1:]]
        state_scores = np.zeros(len(seqs), dtype=np.float64)
        if mut_strs:
            state_scores[1:] = np.asarray(adapter.score_mutations(wt, mut_strs, coords=coords), dtype=np.float64)

        pred_eps: list[float] = []
        obs_eps: list[float] = []
        for cyc in bg.get("cycles", []):
            s_wt, s_a, s_b, s_ab = [int(i) for i in cyc["states"]]
            pred_eps.append(float(state_scores[s_ab] - state_scores[s_a] - state_scores[s_b] + state_scores[s_wt]))
            obs_val = cyc.get("epistasis", cyc.get("measured"))
            if obs_val is not None:
                has_measured_targets = True
                obs_eps.append(float(obs_val))
        total_cycles += len(pred_eps)
        if len(obs_eps) == len(pred_eps) and len(obs_eps) >= 3:
            rho = _safe_spearman(np.asarray(pred_eps), np.asarray(obs_eps))
            if math.isfinite(rho):
                rhos_by_group[str(bg.get("group", bg["name"]))].append(rho)

    if missing_bbs:
        return _not_configured("Missing real WT 3D backbones for R3 epistasis backgrounds", missing_bbs[:10])
    if not has_measured_targets or not rhos_by_group:
        return _not_configured(
            "R3 epistasis cohort lacks measured cycle epistasis targets for formal Spearman evaluation",
            [root / R3_EPISTASIS_PLAN_REL],
        )

    mean_rho = float(np.mean([float(np.mean(v)) for v in rhos_by_group.values()]))
    res: dict[str, Any] = {
        "status": "ok",
        "mode": "full",
        "evaluation_scope": "formal_full_cohort",
        "n_backgrounds": len(bgs),
        "n_groups": len(rhos_by_group),
        "n_cycles": total_cycles,
        "epistasis_spearman": mean_rho,
    }
    if "r3_epistasis" in leaderboards:
        res["reference_comparison"] = compute_rank_and_percentile(mean_rho, leaderboards["r3_epistasis"])
    return res


def _run_suite_r3_contacts(
    adapter: ProteinBenchmarkAdapter,
    *,
    mode: str,
    root: Path,
    leaderboards: dict[str, ReferenceLeaderboard],
) -> dict[str, Any]:
    if not adapter.capabilities.site_conditional_logprobs:
        return {"status": "not_supported", "reason": "Adapter capability site_conditional_logprobs=False"}

    if mode == "smoke":
        seq = "MKTAYIAKQRQISFVK"
        L = len(seq)
        coords = _smoke_coords_if_needed(adapter, seq)
        base = np.asarray(adapter.site_conditional_logprobs(seq, coords=coords), dtype=np.float64)
        pert_seq = ("A" if seq[0] != "A" else "G") + seq[1:]
        pert = np.asarray(adapter.site_conditional_logprobs(pert_seq, coords=coords), dtype=np.float64)
        return {
            "status": "smoke_ok",
            "mode": "smoke",
            "evaluation_scope": "contract_smoke_only",
            "length": L,
            "smoke_perturbation_norm": float(np.linalg.norm(pert - base)),
        }

    # Formal full cohort: real admitted structural E01 references with real 3D coordinates (NEVER synthetic!)
    missing = _check_required_paths(root, [R1_COHORT_REL, R3_CONTACTS_SITES_REL, WT_BACKBONES_3D_REL])
    if missing:
        return _not_configured("Required frozen E01 structural contacts files or wt_backbones_3d are missing", missing)

    cohort = json.loads((root / R1_COHORT_REL).read_text(encoding="utf-8"))
    with gzip.open(root / R3_CONTACTS_SITES_REL, "rt", encoding="utf-8") as fh:
        sites_doc = json.load(fh)
    site_rows = sites_doc["sites"] if isinstance(sites_doc, dict) else sites_doc
    admitted_ids = {str(r["assay_id"]) for r in site_rows if r.get("status") == "admitted"}
    rows_by_assay = {str(r["assay"]): r for r in cohort["assays"]}
    selected = [rows_by_assay[a] for a in sorted(admitted_ids & set(rows_by_assay))]
    if not selected:
        return _not_configured("No admitted structural E01 assays found", [root / R3_CONTACTS_SITES_REL])

    aurocs: list[float] = []
    missing_bbs: list[str] = []
    for row in selected:
        seq = str(row["wildtype"]).strip().upper()
        coords = _lookup_real_backbone_coords(seq, root)
        if coords is None:
            missing_bbs.append(f"wt_backbone:{row['assay']}")
            continue
        L = len(seq)
        c_arg = coords if adapter.capabilities.requires_backbone_3d else None
        base = np.asarray(adapter.site_conditional_logprobs(seq, coords=c_arg), dtype=np.float64)
        raw = np.zeros((L, L), dtype=np.float64)
        for i in range(L):
            alt_aa = "A" if seq[i] != "A" else "G"
            pert_seq = seq[:i] + alt_aa + seq[i + 1 :]
            pert = np.asarray(adapter.site_conditional_logprobs(pert_seq, coords=c_arg), dtype=np.float64)
            raw[i] = np.linalg.norm(pert - base, axis=-1)
        np.fill_diagonal(raw, 0.0)
        sym = 0.5 * (raw + raw.T)
        ca = coords[:, 1, :]
        dist = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
        pos_s, neg_s = [], []
        for i in range(L):
            for j in range(i + 6, L):
                if dist[i, j] < 8.0:
                    pos_s.append(float(sym[i, j]))
                else:
                    neg_s.append(float(sym[i, j]))
        if pos_s and neg_s:
            aurocs.append(_binary_auroc(pos_s, neg_s))

    if missing_bbs:
        return _not_configured("Missing real WT 3D backbones for E01 contact assays", missing_bbs[:10])
    if not aurocs:
        return _not_configured("No valid contact pairs found across admitted E01 assays", [root / R3_CONTACTS_SITES_REL])

    mean_auc = float(np.mean(aurocs))
    res: dict[str, Any] = {
        "status": "ok",
        "mode": "full",
        "evaluation_scope": "formal_full_cohort",
        "n_assays": len(aurocs),
        "contact_auroc": mean_auc,
    }
    if "r3_contacts" in leaderboards:
        res["reference_comparison"] = compute_rank_and_percentile(mean_auc, leaderboards["r3_contacts"])
    return res


def _run_suite_r4_family_memory(
    adapter: ProteinBenchmarkAdapter,
    *,
    mode: str,
    root: Path,
    leaderboards: dict[str, ReferenceLeaderboard],
) -> dict[str, Any]:
    if not adapter.capabilities.site_conditional_logprobs:
        return {"status": "not_supported", "reason": "Adapter capability site_conditional_logprobs=False"}

    if mode == "smoke":
        seq = "MKTAYIAKQRQISFVK"
        coords = _smoke_coords_if_needed(adapter, seq)
        lp = np.asarray(adapter.site_conditional_logprobs(seq, coords=coords), dtype=np.float64)
        mean_lp = float(np.mean([lp[p, AA20_INDEX[ch]] for p, ch in enumerate(seq)]))
        return {
            "status": "smoke_ok",
            "mode": "smoke",
            "evaluation_scope": "contract_smoke_only",
            "n_sequences": 1,
            "mean_logp_nats": mean_lp,
        }

    # Formal full cohort: exact 64 evaluation families from conditional-inputs.json
    if adapter.capabilities.requires_backbone_3d:
        return {
            "status": "not_supported",
            "mode": "full",
            "evaluation_scope": "formal_full_cohort",
            "reason": "Unaligned homolog contexts in family_conditionals lack per-homolog 3D backbones",
        }
    missing = _check_required_paths(root, [R4_FAMILY_COND_REL])
    if missing:
        return _not_configured("Required frozen family_conditionals conditional-inputs.json is missing", missing)

    doc = json.loads((root / R4_FAMILY_COND_REL).read_text(encoding="utf-8"))
    eval_units = [u for u in doc.get("units", []) if u.get("partition") == "evaluation"]
    if not eval_units:
        return _not_configured("No partition=='evaluation' units in conditional-inputs.json", [root / R4_FAMILY_COND_REL])

    unit_means: list[float] = []
    for unit in eval_units:
        site_means: list[float] = []
        for site in unit.get("sites", []):
            ctx_lps: list[float] = []
            for ctx in site.get("contexts", []):
                native = str(ctx["native"]).strip().upper()
                idx = int(ctx["native_index"])
                obs = str(ctx["observed"]).strip().upper()
                lp = np.asarray(adapter.site_conditional_logprobs(native, positions=[idx]), dtype=np.float64)
                ctx_lps.append(float(lp[0, AA20_INDEX[obs]]))
            if ctx_lps:
                site_means.append(float(np.mean(ctx_lps)))
        if site_means:
            unit_means.append(float(np.mean(site_means)))

    mean_logp = float(np.mean(unit_means))
    res: dict[str, Any] = {
        "status": "ok",
        "mode": "full",
        "evaluation_scope": "formal_full_cohort",
        "n_evaluation_families": len(unit_means),
        "mean_logp_nats": mean_logp,
        "mean_nll_nats": -mean_logp,
    }
    if "r4_family_memory" in leaderboards:
        res["reference_comparison"] = compute_rank_and_percentile(mean_logp, leaderboards["r4_family_memory"])
    return res


def _run_suite_r4_generation(
    adapter: ProteinBenchmarkAdapter,
    *,
    mode: str,
    root: Path,
    seed: int,
) -> dict[str, Any]:
    caps = adapter.capabilities
    if not any([caps.generate_unconditional, caps.infill_local, caps.sample_family_guided, caps.continue_from_prefix]):
        return {"status": "not_supported", "reason": "Adapter declares no generation capabilities"}

    if mode == "smoke":
        template = "MKTAYIAKQRQISFVK"
        coords = _smoke_coords_if_needed(adapter, template)
        out_s: dict[str, Any] = {
            "status": "smoke_ok",
            "mode": "smoke",
            "evaluation_scope": "contract_smoke_only",
            "subtasks": {},
        }
        if caps.generate_unconditional:
            seqs = adapter.generate_unconditional([16, 20], seed=seed)
            out_s["subtasks"]["unconditional"] = {
                "n_generated": len(seqs),
                "valid_aa20_fraction": float(np.mean([set(s) <= AA20_SET for s in seqs])),
            }
        if caps.infill_local:
            inf = adapter.infill_local(template, [5, 6, 7, 8], coords=coords, seed=seed)
            out_s["subtasks"]["infill_local"] = {
                "scaffold_preserved": bool(inf[:5] == template[:5] and inf[9:] == template[9:]),
                "valid_aa20": bool(set(inf) <= AA20_SET),
            }
        if caps.sample_family_guided:
            bias = np.zeros((len(template), 20), dtype=np.float32)
            bias[:, AA20_INDEX["A"]] = 2.0
            gdd = adapter.sample_family_guided(
                len(template), n_samples=1, profile_log_bias=bias, guidance_scale=1.0, coords=coords, seed=seed
            )
            out_s["subtasks"]["family_guided"] = {"guided_valid": bool(set(gdd[0]) <= AA20_SET)}
        if caps.continue_from_prefix:
            cont = adapter.continue_from_prefix([template[:6]], target_length=10, coords=coords, seed=seed)
            out_s["subtasks"]["continue_from_prefix"] = {"length": len(cont), "valid_aa20": bool(set(cont) <= AA20_SET)}
        return out_s

    # Formal full cohort: real frozen ladder_backbones.jsonl scaffolds
    missing = _check_required_paths(root, [R4_GEN_BACKBONES_REL])
    if caps.requires_backbone_3d and not (root / WT_BACKBONES_3D_REL).is_dir():
        missing.append(root / WT_BACKBONES_3D_REL)
    if missing:
        return _not_configured("Required frozen R4 generation ladder_backbones.jsonl is missing", missing)

    scaffolds: list[str] = []
    with (root / R4_GEN_BACKBONES_REL).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                seq = str(rec.get("sequence") or rec.get("wildtype") or "").strip().upper()
                if seq and set(seq) <= AA20_SET:
                    scaffolds.append(seq)
    if not scaffolds:
        return _not_configured("No valid scaffolds found in ladder_backbones.jsonl", [root / R4_GEN_BACKBONES_REL])

    out_f: dict[str, Any] = {
        "status": "ok",
        "mode": "full",
        "evaluation_scope": "formal_full_cohort",
        "n_scaffolds": len(scaffolds),
        "subtasks": {},
    }
    if caps.generate_unconditional:
        lens = [min(len(s), adapter.metadata.max_positions) for s in scaffolds]
        seqs = adapter.generate_unconditional(lens, seed=seed)
        out_f["subtasks"]["unconditional"] = {
            "n_generated": len(seqs),
            "valid_aa20_fraction": float(np.mean([set(s) <= AA20_SET for s in seqs])),
            "mean_unique_aa": float(np.mean([len(set(s)) for s in seqs])),
        }
    if caps.infill_local:
        preserved = []
        valid_flags = []
        for idx, scaf in enumerate(scaffolds):
            coords = _lookup_real_backbone_coords(scaf, root) if caps.requires_backbone_3d else None
            if caps.requires_backbone_3d and coords is None:
                return _not_configured("Missing real WT 3D backbones for R4 generation scaffolds", [root / WT_BACKBONES_3D_REL])
            start = max(1, len(scaf) // 3)
            extent = min(10, len(scaf) - start - 1)
            inf = adapter.infill_local(scaf, list(range(start, start + extent)), coords=coords, seed=seed + idx)
            preserved.append(inf[:start] == scaf[:start] and inf[start + extent :] == scaf[start + extent :])
            valid_flags.append(set(inf) <= AA20_SET)
        out_f["subtasks"]["infill_local"] = {
            "n_scaffolds": len(scaffolds),
            "scaffold_preserved_fraction": float(np.mean(preserved)),
            "valid_aa20_fraction": float(np.mean(valid_flags)),
        }
    if caps.continue_from_prefix:
        valid_flags = []
        for idx, scaf in enumerate(scaffolds):
            coords = _lookup_real_backbone_coords(scaf, root) if caps.requires_backbone_3d else None
            if caps.requires_backbone_3d and coords is None:
                return _not_configured("Missing real WT 3D backbones for R4 continuation scaffolds", [root / WT_BACKBONES_3D_REL])
            half = max(1, len(scaf) // 2)
            cont = adapter.continue_from_prefix([scaf[:half]], target_length=len(scaf) - half, coords=coords, seed=seed + idx)
            valid_flags.append(set(cont) <= AA20_SET)
        out_f["subtasks"]["continue_from_prefix"] = {
            "n_scaffolds": len(scaffolds),
            "valid_aa20_fraction": float(np.mean(valid_flags)),
        }
    return out_f


def _run_suite_r5_context(
    adapter: ProteinBenchmarkAdapter,
    *,
    mode: str,
    root: Path,
    leaderboards: dict[str, ReferenceLeaderboard],
) -> dict[str, Any]:
    if not adapter.capabilities.score_sequence_nll:
        return {"status": "not_supported", "reason": "Adapter capability score_sequence_nll=False"}

    if mode == "smoke":
        units = [
            ("MKTAYIAKQRQISFVK", "MKTAYIAKQRQISFVR", "GGGGPPPPWWWWCCCC"),
            ("AEAAAKEAAAKEAAAK", "AEAAAKDAAAKEAAAR", "YYYYFFFFHHHHDDDD"),
        ]
        hom_g, unr_g = [], []
        for target, hom, unrel in units:
            c_t = _smoke_coords_if_needed(adapter, target)
            c_h = [_smoke_coords_if_needed(adapter, hom)] if adapter.capabilities.requires_backbone_3d else []
            c_u = [_smoke_coords_if_needed(adapter, unrel)] if adapter.capabilities.requires_backbone_3d else []
            z = float(adapter.score_sequence_nll(target, coords=c_t)["nll_per_residue"])
            h = float(adapter.score_sequence_nll(target, context_items=[hom], coords=c_t, context_coords=[c for c in c_h if c is not None])["nll_per_residue"])
            u = float(adapter.score_sequence_nll(target, context_items=[unrel], coords=c_t, context_coords=[c for c in c_u if c is not None])["nll_per_residue"])
            hom_g.append(z - h)
            unr_g.append(z - u)
        return {
            "status": "smoke_ok",
            "mode": "smoke",
            "evaluation_scope": "contract_smoke_only",
            "n_units": len(units),
            "smoke_auroc_hom_vs_unrelated": _binary_auroc(hom_g, unr_g),
        }

    # Formal full cohort: exact 800-unit homologous-context cohort
    missing = _check_required_paths(root, [R5_CONTEXT_COHORT_REL])
    if adapter.capabilities.requires_backbone_3d and not (root / WT_BACKBONES_3D_REL).is_dir():
        missing.append(root / WT_BACKBONES_3D_REL)
    if missing:
        return _not_configured("Required frozen R5 homologous-context cohort.json is missing", missing)

    cohort = json.loads((root / R5_CONTEXT_COHORT_REL).read_text(encoding="utf-8"))
    prot = cohort.get("protein", {})
    records: list[str] = prot.get("records", [])
    related_map = prot.get("related", {})
    filler_raw = prot.get("filler", 0)
    filler_idx = int(filler_raw) if isinstance(filler_raw, int) else int(filler_raw.get("index", 0))
    filler_seq = (
        records[filler_idx]
        if isinstance(filler_raw, int)
        else str(filler_raw.get("record") or filler_raw.get("sequence") or records[filler_idx])
    )
    retained = [u for u in prot.get("units", []) if u.get("stratum") == "retained"]
    if not retained:
        return _not_configured("R5 context cohort contains no retained units", [root / R5_CONTEXT_COHORT_REL])

    hom_gains_30_50: list[float] = []
    unr_gains_30_50: list[float] = []
    missing_bbs: list[str] = []
    for unit in retained:
        band = str(unit.get("band") or unit.get("identity_band") or "")
        if band and band != "id_30_50":
            continue
        target_seq, cond_inputs = r4_mod.build_unit_condition_inputs(
            unit, records, related_map, filler_idx=filler_idx, filler_seq=filler_seq
        )
        c_tgt = _lookup_real_backbone_coords(target_seq, root) if adapter.capabilities.requires_backbone_3d else None
        if adapter.capabilities.requires_backbone_3d and c_tgt is None:
            missing_bbs.append(f"wt_backbone:{unit.get('unit_id', 'unknown')}")
            continue
        nll_zero = float(adapter.score_sequence_nll(target_seq, coords=c_tgt)["nll_per_residue"])
        hom_items = cond_inputs.get("homologue", [])
        unr_items = cond_inputs.get("unrelated", [])
        c_hom = [c for c in (_lookup_real_backbone_coords(s, root) for s in hom_items) if c is not None] if adapter.capabilities.requires_backbone_3d else []
        c_unr = [c for c in (_lookup_real_backbone_coords(s, root) for s in unr_items) if c is not None] if adapter.capabilities.requires_backbone_3d else []
        nll_hom = float(adapter.score_sequence_nll(target_seq, context_items=hom_items, coords=c_tgt, context_coords=c_hom)["nll_per_residue"])
        nll_unr = float(adapter.score_sequence_nll(target_seq, context_items=unr_items, coords=c_tgt, context_coords=c_unr)["nll_per_residue"])
        hom_gains_30_50.append(nll_zero - nll_hom)
        unr_gains_30_50.append(nll_zero - nll_unr)

    if missing_bbs:
        return _not_configured("Missing real WT 3D backbones for R5 context units", missing_bbs[:10])
    if not hom_gains_30_50:
        return _not_configured("No id_30_50 retained units found in R5 context cohort", [root / R5_CONTEXT_COHORT_REL])

    auroc = _binary_auroc(hom_gains_30_50, unr_gains_30_50)
    res: dict[str, Any] = {
        "status": "ok",
        "mode": "full",
        "evaluation_scope": "formal_full_cohort",
        "n_units_id_30_50": len(hom_gains_30_50),
        "auroc_hom_vs_unrelated_30_50": auroc,
        "mean_hom_nll_reduction": float(np.mean(hom_gains_30_50)),
        "mean_unrel_nll_reduction": float(np.mean(unr_gains_30_50)),
    }
    if "r5_context" in leaderboards:
        res["reference_comparison"] = compute_rank_and_percentile(auroc, leaderboards["r5_context"])
    return res


# --------------------------------------------------------------------------- #
# Top-Level Benchmark Orchestrator & Report Writer
# --------------------------------------------------------------------------- #


def _render_markdown_summary(report: dict[str, Any]) -> str:
    meta = report["metadata"]
    lines = [
        f"# Capability Benchmark Summary: {meta['display_name']} (`{meta['model_id']}`)",
        "",
    ]
    if report["mode"] == "smoke":
        lines.extend(
            [
                "> **Contract / Smoke Validation Only** — Smoke mode verifies adapter shapes/invariants on tiny fixtures and never computes formal leaderboard ranks or percentiles.",
                "",
            ]
        )
    lines.extend(
        [
            f"- **Paradigm / Modality**: `{meta['paradigm']}` / `{meta['modality']}`",
            f"- **Conditioning / Alignment**: `{meta['conditioning_regime']}` / `{meta['input_alignment_format']}`",
            f"- **Scale / Max Positions**: `{meta['n_params_millions']:.2f}M` params / `{meta['max_positions']}` positions",
            f"- **Mode / Generated UTC**: `{report['mode']}` / `{report['generated_utc']}`",
            "",
            "| Suite | Status | Primary Metric | Value | Ref Rank / Percentile |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for suite_name, res in report["suites"].items():
        status = res.get("status", "unknown")
        ref = res.get("reference_comparison")
        if ref and ref.get("rank") is not None:
            metric = ref["metric"]
            val = f"{ref['value']:.4f}"
            rank_s = f"#{ref['rank']} / {ref['n_reference_models']} ({ref['percentile']:.1f}%)"
        else:
            metric = "-"
            val = "-"
            rank_s = res.get("reason", "-") if status not in ("ok", "smoke_ok") else "-"
        lines.append(f"| `{suite_name}` | `{status}` | `{metric}` | {val} | {rank_s} |")
    lines.append("")
    return "\n".join(lines)


def run_benchmark(
    adapter: ProteinBenchmarkAdapter,
    *,
    suites: Sequence[str] = ALL_SUITES,
    mode: str = "smoke",
    out_dir: Path | str | None = None,
    unified_dir: Path | str | None = None,
    root: Path | str | None = None,
    seed: int = 20261124,
) -> dict[str, Any]:
    """Run selected benchmark suites, rank against sealed unified JSONs (full mode only), and write report + sidecar."""
    if mode not in ("smoke", "full"):
        raise ValueError(f"Unsupported mode {mode!r}; expected 'smoke' or 'full'")
    selected = list(ALL_SUITES) if ("all" in suites) else list(suites)
    unknown = [s for s in selected if s not in ALL_SUITES]
    if unknown:
        raise ValueError(f"Unknown suites {unknown}; expected subset of {list(ALL_SUITES)}")

    t0 = time.perf_counter()
    eff_root = Path(root) if root is not None else REPO_ROOT
    # Never load or attach formal reference leaderboards in smoke mode!
    leaderboards = load_reference_leaderboards(unified_dir=unified_dir) if mode == "full" else {}
    suite_results: dict[str, Any] = {}

    for suite in selected:
        try:
            if suite == "validate":
                suite_results[suite] = {"status": "ok", **validate_adapter(adapter, seed=seed)}
            elif suite == "r1_dms":
                suite_results[suite] = _run_suite_r1_dms(adapter, mode=mode, root=eff_root, leaderboards=leaderboards)
            elif suite == "r2_biophysical":
                suite_results[suite] = _run_suite_r2_biophysical(adapter, mode=mode, root=eff_root, leaderboards=leaderboards)
            elif suite == "r3_epistasis":
                suite_results[suite] = _run_suite_r3_epistasis(adapter, mode=mode, root=eff_root, leaderboards=leaderboards)
            elif suite == "r3_contacts":
                suite_results[suite] = _run_suite_r3_contacts(adapter, mode=mode, root=eff_root, leaderboards=leaderboards)
            elif suite == "r4_family_memory":
                suite_results[suite] = _run_suite_r4_family_memory(adapter, mode=mode, root=eff_root, leaderboards=leaderboards)
            elif suite == "r4_generation":
                suite_results[suite] = _run_suite_r4_generation(adapter, mode=mode, root=eff_root, seed=seed)
            elif suite == "r5_context":
                suite_results[suite] = _run_suite_r5_context(adapter, mode=mode, root=eff_root, leaderboards=leaderboards)
        except CapabilityNotSupportedError as exc:
            suite_results[suite] = {"status": "not_supported", "reason": str(exc)}

    report: dict[str, Any] = {
        "schema": "protein_capability_benchmark_report_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "evaluation_scope": "contract_smoke_only" if mode == "smoke" else "formal_full_cohort",
        "seed": int(seed),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "metadata": adapter.metadata.to_dict(),
        "capabilities": adapter.capabilities.to_dict(),
        "suites": suite_results,
    }

    if out_dir is not None:
        odir = Path(out_dir)
        odir.mkdir(parents=True, exist_ok=True)
        slug = adapter.metadata.model_id.replace("/", "_").replace("|", "_").replace(" ", "_")
        json_path = odir / f"{slug}_benchmark_report.json"
        sha_path = odir / f"{slug}_benchmark_report.json.sha256"
        md_path = odir / f"{slug}_benchmark_summary.md"

        serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if FORBIDDEN_INFRA_RE.search(serialized):
            raise ValueError("Refusing to write benchmark report containing forbidden infrastructure identifier")
        json_path.write_text(serialized, encoding="utf-8")
        digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
        sha_path.write_text(f"{digest}  {json_path.name}\n", encoding="utf-8")
        md_path.write_text(_render_markdown_summary(report), encoding="utf-8")
        report["artifacts"] = {
            "report_json": str(json_path),
            "report_sha256": digest,
            "summary_md": str(md_path),
        }
    return report
