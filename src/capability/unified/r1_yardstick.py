"""Unified 62-model R1 ProteinGym single-yardstick evaluation library.

Unifies all 62 single-sequence arms (33 pre-registered causal/joint/text arms
from ``pairwise_epistasis.ROSTER`` plus 29 expanded single-sequence general-PLM
arms from ``general_plm_20261120``) onto one identical 25,728-variant /
201-assay / 163-family-cluster ProteinGym yardstick (and the 24,832-variant /
194-assay / 156-cluster GEMME-nondegenerate subset), with:

* Layer 1: Zero-shot within-assay Spearman for all 62 models plus explicit
  Profile, Potts, and GEMME baselines, with 10,000-draw paired cluster bootstrap
  CIs and 62-arm simultaneous maxT contrasts.
* Layer 2 Core Bidirectional Increments: Family-grouped 5x4 nested Ridge CV
  over the exact historical ``SEEDS = (20260923, 20260924, 20260925)`` for both
  ``primary_scalar_baselines`` (``P_1col``, ``PEG_3col``) and
  ``profile_block_baselines`` (``P_14col``, ``PEG_16col``), reusing all 33
  pre-registered cached cells bit-for-bit.
* Layer 2 ``+R`` Representation Bridge: Exact nested-CV ``R`` / ``+R`` /
  ``+(M+R)`` increments for all 28 expanded arms with cached 1024-d R1 readouts,
  paired with the 33 pre-registered arms' cached ``R`` / ``B_R - B`` summary.
* Supplementary 6-rung ``C0`` ladder (744-contrast unified 62-arm family) with
  verbatim historical 33-arm 396-contrast embedding.
* Unified 62-model explicit pair-level ablation (``T_open``, ``T_bal_P``,
  ``T_tol_P``, ``T_bal_PEG``, ``T_tol_PEG``, ``T_tol_bal_PEG``, and reverse
  ``T_bal_M``, ``T_tol_M``, ``T_tol_bal_M``) plus deduplicated 37-arm expanded
  ``T_match`` tier summary.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from scripts.capability.extensions import fit_statistical_ladder as fsl
from src.capability.core.statistics import bootstrap_unit_floor
from src.capability.extensions import bidirectional, overlap, responses
from src.capability.extensions import statistical_ladder as sl
from src.capability.general_plm.eval_utils import atomic_write_json, sha256_file
from src.capability.mutation import matched_discrimination as md
from src.capability.readouts.readout_analysis import ALPHAS, row_weights
from src.capability.unified.roster import (
    ARM_METADATA,
    PREREGISTERED_33_ARMS,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_OUTPUT_DIR,
    resolve_all_gp_r1_readouts,
    resolve_all_gp_r1_scores,
    resolve_gp_r1_json,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LADDER_FULL_DIR = Path("results/extensions/statistical_ladder_20261106/full")
LADDER_SCORES_DIR = Path("results/extensions/statistical_ladder_20261106/accepted-scores")
PROGRESSION_DIR = Path("results/extensions/information_progression_20261006")
READOUT_PANEL_DIR = Path("results/R2/readout_panel_summary_20260923")
EXPLICIT_PAIR_REF_DIR = Path("results/R1/explicit_pair_ablation_20261114")
EXPANDED_MD_ROWS = Path(
    "results/R1/matched_discrimination_20261102/expanded/expanded_variants/analysis/rows.json"
)
MUTATION_SAMPLES_CSV = Path("manuscript/data/prediction-details/mutation-samples.csv.gz")
PROTEINGYM_DMS_CSV = Path("data/proteingym_raw/DMS_substitutions.csv")
FUNCTIONAL_CATEGORIES: tuple[str, ...] = (
    "Activity",
    "Binding",
    "Expression",
    "OrganismalFitness",
    "Stability",
)
ESM3_R1_READOUT_REL = Path("r2_readout_archives/r1_R_esm3-1.4b.npz")

BOOTSTRAP_DRAWS = sl.BOOTSTRAP_DRAWS  # 10,000
BOOTSTRAP_SEED = sl.BOOTSTRAP_SEED  # 20261106
PAIR_BOOTSTRAP_SEED = 20261114
PAIR_BALANCE_SEED = 20261114
PAIR_TOLERANCES: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.80)
PAIR_PRIMARY_TOLERANCE = 0.20
PAIR_RANDOM_SEPARATION = float(2.0 * np.sqrt(3.0) / 3.0)
PAIR_STATISTICS: tuple[str, ...] = ("profile", "potts", "gemme")
PAIR_CANONICAL_ORDER = (
    "assay",
    "wildtype_residue",
    "mutant_residue",
    "position_a",
    "position_b",
)
PAIR_TIER_RULES = {
    "T_open": "the pair rule and the phenotype gap alone; the reference tier",
    "T_tol_P": "Profile matched at the declared tolerance",
    "T_tol_PEG": "Profile, Potts and GEMME each matched at the declared tolerance",
    "T_bal_P": "the Profile's vote cancelled exactly",
    "T_bal_PEG": "the Profile's, Potts's and GEMME's votes each cancelled exactly",
    "T_tol_bal_PEG": "all three votes cancelled exactly inside T_tol_PEG; the decisive forward tier",
    "T_tol_M": "the arm's own likelihood rank matched at the declared tolerance",
    "T_bal_M": "the arm's own likelihood vote cancelled exactly",
    "T_tol_bal_M": "the arm's own vote cancelled exactly inside T_tol_M",
}

# Exact 7 GEMME-degenerate assays (all 0.0 across 3 official runs; 896 rows, 7 clusters).
GEMME_REFUSED_ASSAYS: frozenset[str] = frozenset(
    {
        "A0A247D711_LISMN_Stadelmann_2021",
        "DN7A_SACS2_Tsuboyama_2023_1JIC",
        "ISDH_STAAW_Tsuboyama_2023_2LHR",
        "PPM1D_HUMAN_Miller_2022",
        "SBI_STAAM_Tsuboyama_2023_2JVG",
        "SCIN_STAAR_Tsuboyama_2023_2QFF",
        "VG08_BPP22_Tsuboyama_2023_2GP8",
    }
)

# Canonical 62-arm order split into 33 pre-registered and 29 expanded single-sequence arms.
EXPANDED_29_SINGLE_ARMS: tuple[str, ...] = tuple(
    arm for arm in UNIFIED_62_SINGLE_SEQUENCE_ARMS if ARM_METADATA[arm].origin_cohort == "expanded_32"
)
EXPANDED_28_READOUT_ARMS: tuple[str, ...] = tuple(
    arm for arm in EXPANDED_29_SINGLE_ARMS if ARM_METADATA[arm].has_r1_raw_readout_1024
)

# Designs fitted per expanded arm on the [B, M, P, L, E, G] superset.
EXPANDED_NON_R_DESIGNS: tuple[str, ...] = (
    "M",
    "MP1",
    "MPEG3",
    "MP",
    "MPE",
    "MPG",
    "MPEG",
    "BMPL",
    "BMPLEG",
)
# Extra scalar-baseline joint designs fitted for the 33 pre-registered arms.
PREREG_SCALAR_DESIGNS: tuple[str, ...] = ("MP1", "MPEG3")
# Shared model-independent scalar-baseline controls.
SHARED_SCALAR_DESIGNS: tuple[str, ...] = ("P1", "PEG3")
# Cached model-independent block controls from statistical_ladder_20261106 / information_progression.
CACHED_CONTROL_DESIGNS_LADDER: tuple[str, ...] = ("P", "PE", "PG", "PEG", "BPLEG")
CACHED_CONTROL_DESIGNS_PROG: tuple[str, ...] = ("BPL",)
CACHED_MODEL_DESIGNS_LADDER: tuple[str, ...] = ("MPE", "MPG", "MPEG", "BMPLEG")
CACHED_MODEL_DESIGNS_PROG: tuple[str, ...] = ("M", "MP", "BMPL")
# 6 representation-bridge designs fitted on [M, P, E, G, R] for the 28 expanded arms.
R_BRIDGE_DESIGNS: tuple[str, ...] = (
    "R",
    "P1_R",
    "P1_M_R",
    "PEG3_R",
    "PEG3_M_R",
    "PEG16_M_R",
)


def write_sha256_sidecar(path: Path) -> str:
    """Write ``<filename>.sha256`` next to ``path`` and return its hex digest."""
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def _arm_meta_dict(arm_id: str) -> dict[str, Any]:
    meta = ARM_METADATA[arm_id]
    return {
        "arm_id": meta.arm_id,
        "display_name": meta.display_name,
        "family": meta.family,
        "paradigm": meta.paradigm,
        "modality": meta.modality,
        "architecture": meta.architecture,
        "params_m": meta.params_m,
        "vocab_type": meta.vocab_type,
        "has_r1_raw_readout_1024": meta.has_r1_raw_readout_1024,
        "origin_cohort": meta.origin_cohort,
    }


# --------------------------------------------------------------------------- #
# Exact multi-design nested-CV Ridge solver (verified < 1e-12 vs overlap.py)
# --------------------------------------------------------------------------- #


def _weighted_moments_and_cov(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute float64 training standardization, weighted covariance, and RHS once."""
    x = torch.as_tensor(x_train, device="cpu", dtype=torch.float64)
    y = torch.as_tensor(y_train, device="cpu", dtype=torch.float64)
    w = torch.as_tensor(weights, device="cpu", dtype=torch.float64)
    w = w / w.sum()
    mean = (w[:, None] * x).sum(0)
    xc = x - mean
    scale = (w[:, None] * xc.square()).sum(0).sqrt()
    scale = torch.where(scale < 1e-12, torch.ones_like(scale), scale)
    z = xc / scale
    ym = (w * y).sum()
    cov = z.T @ (w[:, None] * z)
    rhs = z.T @ (w * (y - ym))
    return mean, scale, ym, cov, rhs


def _predict_from_cov_slice(
    cov_full: torch.Tensor,
    rhs_full: torch.Tensor,
    mean_full: torch.Tensor,
    scale_full: torch.Tensor,
    ym: torch.Tensor,
    x_eval_full: torch.Tensor,
    cols: torch.Tensor,
    alphas: Sequence[float],
) -> np.ndarray:
    """Solve weighted ridge on a column slice of precomputed ``(cov_full, rhs_full)``."""
    cov = cov_full.index_select(0, cols).index_select(1, cols)
    rhs = rhs_full.index_select(0, cols)
    mean = mean_full.index_select(0, cols)
    scale = scale_full.index_select(0, cols)
    x_eval = x_eval_full.index_select(1, cols)
    alpha_t = torch.tensor(alphas, device="cpu", dtype=torch.float64)
    values, vectors = torch.linalg.eigh(cov)
    beta = vectors @ ((vectors.T @ rhs)[:, None] / (values.clamp_min(0)[:, None] + alpha_t))
    return (((x_eval - mean) / scale) @ beta + ym).cpu().numpy()


def projected_predict_batch(
    x_full: np.ndarray,
    design_cols: Mapping[str, Sequence[int]],
    measured: np.ndarray,
    aids: np.ndarray,
    families: np.ndarray,
    signature: Sequence[Any],
    *,
    alphas: Sequence[float] = ALPHAS,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]]]:
    """Exact multi-design equivalent of ``overlap.projected_predict``.

    Builds the weighted training standardization and full Gram matrix once per
    inner/outer fold on ``x_full``, then slices each design's columns for exact
    eigendecomposition ridge prediction. Matches ``overlap.projected_predict``
    to float64 machine precision (``< 1e-12``) while cutting covariance build
    cost by the number of shared designs.
    """
    x_full = np.asarray(x_full, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    aids = np.asarray(aids)
    families = np.asarray(families)
    if x_full.ndim != 2 or not len(x_full):
        raise ValueError("x_full must be a non-empty 2D matrix")
    if any(len(a) != len(x_full) for a in (measured, aids, families)):
        raise ValueError("unaligned fitting arrays")
    if not np.isfinite(x_full).all() or not np.isfinite(measured).all():
        raise ValueError("nonfinite fitting arrays")

    row_weights(aids, families)
    target = overlap.rerank(measured, aids)
    col_tensors = {
        name: torch.as_tensor(list(cols), dtype=torch.long) for name, cols in design_cols.items()
    }
    predictions = {name: np.full(len(x_full), np.nan, dtype=np.float64) for name in design_cols}
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in design_cols}
    folds = overlap.project_folds(signature, families)

    for fold, held, training, validations in folds:
        train = np.flatnonzero(np.isin(families, training))
        test = np.flatnonzero(np.isin(families, held))
        losses_by_design = {name: np.zeros(len(alphas), dtype=np.float64) for name in design_cols}

        for validation in validations:
            valid = np.flatnonzero(np.isin(families, validation))
            fit = train[~np.isin(families[train], validation)]
            w_fit = row_weights(aids[fit], families[fit])
            w_val = row_weights(aids[valid], families[valid])
            mean_f, scale_f, ym_f, cov_f, rhs_f = _weighted_moments_and_cov(
                x_full[fit], target[fit], w_fit
            )
            x_val_t = torch.as_tensor(x_full[valid], device="cpu", dtype=torch.float64)
            y_val = target[valid, None]
            for name, cols in col_tensors.items():
                pred_val = _predict_from_cov_slice(
                    cov_f, rhs_f, mean_f, scale_f, ym_f, x_val_t, cols, alphas
                )
                loss = (w_val[:, None] * (pred_val - y_val) ** 2).sum(axis=0)
                losses_by_design[name] += len(validation) * loss

        w_train = row_weights(aids[train], families[train])
        mean_o, scale_o, ym_o, cov_o, rhs_o = _weighted_moments_and_cov(
            x_full[train], target[train], w_train
        )
        x_test_t = torch.as_tensor(x_full[test], device="cpu", dtype=torch.float64)

        for name, cols in col_tensors.items():
            losses = losses_by_design[name] / len(training)
            best = len(losses) - 1 - int(np.argmin(losses[::-1]))
            alpha = float(alphas[best])
            pred_test = _predict_from_cov_slice(
                cov_o, rhs_o, mean_o, scale_o, ym_o, x_test_t, cols, [alpha]
            )[:, 0]
            predictions[name][test] = pred_test
            records[name].append(
                {
                    "fold": fold,
                    "held_families": held,
                    "training_families": training,
                    "alpha": alpha,
                }
            )

    for name, pred in predictions.items():
        if not np.isfinite(pred).all():
            raise ValueError(f"{name}: incomplete OOF predictions")
    return predictions, records


# --------------------------------------------------------------------------- #
# Support & Score Loading + Reproduction Gate
# --------------------------------------------------------------------------- #


@dataclass
class R1UnifiedContext:
    """Authoritative 25,728-row R1 support, blocks, and 62-arm aligned scores."""

    root: Path
    samples: list[dict[str, Any]]
    sample_ids: np.ndarray
    measured: np.ndarray
    target: np.ndarray
    aids: np.ndarray
    families: np.ndarray
    assay_categories: dict[str, str]
    assay_categories_sha256: str
    signatures: dict[int, Any]
    blocks: dict[str, np.ndarray]
    prepared_shared: dict[str, np.ndarray]
    raw_baselines: dict[str, np.ndarray]
    raw_model_scores: dict[str, np.ndarray]
    gp_score_provenance: dict[str, dict[str, Any]]
    gp_readout_paths: dict[str, Path]
    gp_readout_dtypes: dict[str, str | None]
    r_bridge_arms: tuple[str, ...]
    support_sha256: str
    provenance: dict[str, Any]


def load_assay_functional_categories(
    root: Path, assay_names: Sequence[str]
) -> tuple[dict[str, str], str]:
    """Load official ProteinGym ``coarse_selection_type`` for every retained R1 assay."""
    csv_path = root / PROTEINGYM_DMS_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing ProteinGym metadata CSV: {csv_path}")
    df = pd.read_csv(csv_path, usecols=["DMS_id", "coarse_selection_type"])
    lookup = dict(
        zip(df["DMS_id"].astype(str), df["coarse_selection_type"].astype(str), strict=True)
    )
    missing = sorted(set(assay_names) - set(lookup))
    if missing:
        raise ValueError(f"Assays missing from {csv_path}: {missing[:5]}")
    mapping = {str(a): lookup[str(a)] for a in assay_names}
    unknown_cats = sorted(set(mapping.values()) - set(FUNCTIONAL_CATEGORIES))
    if unknown_cats:
        raise ValueError(f"Unexpected coarse_selection_type values: {unknown_cats}")
    return mapping, sha256_file(csv_path)


def load_gp_r1_score_vector(
    arm_id: str,
    path: Path,
    sample_ids: Sequence[str],
) -> np.ndarray:
    """Load one expanded arm's 25,728-row R1 score vector strictly aligned to ``sample_ids``."""
    expected = [str(s) for s in sample_ids]
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as d:
            sids = [str(x) for x in d["sample_ids"]]
            scores = np.asarray(d["scores"], dtype=np.float64)
        if sids == expected:
            out = scores.copy()
        else:
            if len(set(sids)) != len(sids) or set(sids) != set(expected):
                raise ValueError(f"{arm_id}: sample_id set mismatch in {path}")
            pos = {sid: idx for idx, sid in enumerate(sids)}
            out = np.asarray([scores[pos[sid]] for sid in expected], dtype=np.float64)
    elif path.suffix == ".jsonl":
        by_sid: dict[str, float] = {}
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = str(row["sample_id"])
                val = row.get("score_masked_marginal_additive")
                if val is None:
                    val = row.get("score_exact_pll_delta")
                if val is None:
                    raise ValueError(f"{arm_id}: missing score for {sid} in {path}")
                by_sid[sid] = float(val)
        if set(by_sid) != set(expected):
            raise ValueError(
                f"{arm_id}: jsonl sample_id mismatch in {path}: "
                f"missing={len(set(expected) - set(by_sid))}, extra={len(set(by_sid) - set(expected))}"
            )
        out = np.asarray([by_sid[sid] for sid in expected], dtype=np.float64)
    else:
        raise ValueError(f"{arm_id}: unsupported R1 score file suffix {path.suffix!r}")

    if out.shape != (len(expected),) or not np.isfinite(out).all():
        raise ValueError(f"{arm_id}: non-finite or misaligned R1 scores from {path}")
    return out


def load_r1_unified_context(
    root: Path | str = REPO_ROOT,
    *,
    out_dir: Path | str | None = None,
) -> R1UnifiedContext:
    """Load the frozen 25,728-row R1 support, Potts/GEMME baselines, and all 62 arms."""
    base = Path(root).resolve()
    data = fsl.load_support(base, base / PROGRESSION_DIR)
    sample_ids = np.asarray([str(row["sample_id"]) for row in data["samples"]])
    aids = np.asarray(data["aids"])
    families = np.asarray(data["families"])
    measured = np.asarray(data["measured"], dtype=np.float64)
    target = overlap.rerank(measured, aids)
    assay_categories, dms_csv_sha = load_assay_functional_categories(
        base, sorted({str(a) for a in aids})
    )

    scores_root = base / LADDER_SCORES_DIR
    potts_table = fsl.read_baseline_table(scores_root, "potts", sl.POTTS_COLUMN)
    gemme_table = fsl.read_baseline_table(scores_root, "gemme", sl.GEMME_COLUMN)
    joined_potts = sl.join_scores(sample_ids, potts_table, columns=(sl.POTTS_COLUMN,))
    joined_gemme = sl.join_scores(sample_ids, gemme_table, columns=(sl.GEMME_COLUMN,))
    if sl.PROFILE_COLUMN in potts_table["columns"]:
        prof_check = sl.join_scores(sample_ids, potts_table, columns=(sl.PROFILE_COLUMN,))[
            sl.PROFILE_COLUMN
        ]
        if float(np.max(np.abs(prof_check - data["raw_profile"]))) > fsl.TOLERANCE:
            raise ValueError("Potts table profile_delta disagrees with frozen cohort raw_profile")

    blocks = dict(data["blocks"])
    blocks["E"] = joined_potts[sl.POTTS_COLUMN][:, None]
    blocks["G"] = joined_gemme[sl.GEMME_COLUMN][:, None]

    prepared_shared = {
        name: sl.prepare_block(name, blocks[name], aids) for name in ("B", "P", "L", "E", "G")
    }
    prepared_shared["P1"] = prepared_shared["P"][:, :1].copy()

    raw_baselines = {
        "profile": np.asarray(data["raw_profile"], dtype=np.float64),
        "potts": np.asarray(joined_potts[sl.POTTS_COLUMN], dtype=np.float64),
        "gemme": np.asarray(joined_gemme[sl.GEMME_COLUMN], dtype=np.float64),
    }

    gp_scores_resolved = resolve_all_gp_r1_scores(root=base)
    gp_readouts_resolved = resolve_all_gp_r1_readouts(root=base)

    missing_gp = sorted(set(EXPANDED_29_SINGLE_ARMS) - set(gp_scores_resolved))
    if missing_gp:
        raise FileNotFoundError(f"Missing GP R1 score artifacts for: {missing_gp}")
    missing_ro = sorted(set(EXPANDED_28_READOUT_ARMS) - set(gp_readouts_resolved))
    if missing_ro:
        raise FileNotFoundError(f"Missing GP R1 readout artifacts for: {missing_ro}")

    raw_model_scores: dict[str, np.ndarray] = {}
    gp_score_provenance: dict[str, dict[str, Any]] = {}
    for arm_id in UNIFIED_62_SINGLE_SEQUENCE_ARMS:
        if arm_id in PREREGISTERED_33_ARMS:
            raw_model_scores[arm_id] = np.asarray(data["features"][arm_id], dtype=np.float64)
        else:
            art = gp_scores_resolved[arm_id]
            vec = load_gp_r1_score_vector(arm_id, art.path, sample_ids)
            raw_model_scores[arm_id] = vec
            gp_score_provenance[arm_id] = {
                "path": str(art.path.relative_to(base)),
                "sha256": sha256_file(art.path),
                "dtype_realised": art.dtype_realised,
            }

    # Sign-direction sanity check across all 29 expanded protein/multimodal arms:
    # higher score must mean higher measured fitness (mean within-assay Spearman > 0).
    for arm_id in EXPANDED_29_SINGLE_ARMS:
        mean_rho = _equal_family_zero_shot_vector(
            raw_model_scores[arm_id], target, aids, families
        )[1].mean()
        if not (mean_rho > 0.10):
            raise ValueError(
                f"{arm_id}: unexpected sign/alignment on ProteinGym R1 (mean Spearman={mean_rho:.4f})"
            )

    gp_readout_paths = {a: gp_readouts_resolved[a].path for a in EXPANDED_28_READOUT_ARMS}
    gp_readout_dtypes = {
        a: gp_readouts_resolved[a].dtype_realised for a in EXPANDED_28_READOUT_ARMS
    }

    # Automatically include esm3-1.4b (29/29 expanded +R bridge) when r1_R_esm3-1.4b.npz exists.
    esm3_candidates = []
    if out_dir is not None:
        esm3_candidates.append(Path(out_dir).resolve() / ESM3_R1_READOUT_REL)
    esm3_candidates.append(base / UNIFIED_OUTPUT_DIR / ESM3_R1_READOUT_REL)
    for cand in esm3_candidates:
        if cand.is_file():
            gp_readout_paths["esm3-1.4b"] = cand
            esm3_ro_dtype = gp_score_provenance["esm3-1.4b"]["dtype_realised"]
            with np.load(cand, allow_pickle=False) as d_ro:
                if "dtype_realised" in d_ro.files:
                    esm3_ro_dtype = str(d_ro["dtype_realised"].item())
            gp_readout_dtypes["esm3-1.4b"] = esm3_ro_dtype
            break
    r_bridge_arms = tuple(a for a in EXPANDED_29_SINGLE_ARMS if a in gp_readout_paths)

    return R1UnifiedContext(
        root=base,
        samples=data["samples"],
        sample_ids=sample_ids,
        measured=measured,
        target=target,
        aids=aids,
        families=families,
        assay_categories=assay_categories,
        assay_categories_sha256=dms_csv_sha,
        signatures=data["signatures"],
        blocks=blocks,
        prepared_shared=prepared_shared,
        raw_baselines=raw_baselines,
        raw_model_scores=raw_model_scores,
        gp_score_provenance=gp_score_provenance,
        gp_readout_paths=gp_readout_paths,
        gp_readout_dtypes=gp_readout_dtypes,
        r_bridge_arms=r_bridge_arms,
        support_sha256=data["provenance"]["full_support_sha256"],
        provenance=dict(
            data["provenance"],
            potts_scores_sha256=potts_table["sha256"],
            gemme_scores_sha256=gemme_table["sha256"],
            proteingym_dms_csv_sha256=dms_csv_sha,
        ),
    )


def verify_reproduction_gate(
    ctx: R1UnifiedContext,
    *,
    models: Sequence[str] = ("progen2-small", "gpt2"),
    seed: int = 20260923,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Verify bit-exact / ``< 1e-12`` reproduction of cached pre-registered R1 cells.

    Refits ``M``, ``MP``, ``MPEG``, ``BMPLEG`` (plus shared ``P``, ``PEG``,
    ``BPLEG``) via BOTH ``sl.fit_design`` and ``projected_predict_batch`` and
    verifies exact agreement against ``results/extensions/statistical_ladder_20261106/full/cells/``
    and ``results/extensions/information_progression_20261006/full/``.
    """
    ladder_cells = ctx.root / LADDER_FULL_DIR / "cells"
    prog_full = ctx.root / PROGRESSION_DIR / "full"
    records: dict[str, Any] = {}
    worst = 0.0

    for model in models:
        prepared_m = dict(
            ctx.prepared_shared,
            M=sl.prepare_block("M", ctx.raw_model_scores[model][:, None], ctx.aids),
        )
        # Load cached ground-truth predictions
        prog_pred = sl.read_predictions(
            prog_full / f"{model}-seed{seed}.npz",
            ("M", "MP", "BPL", "BMPL"),
            sample_ids=ctx.sample_ids,
            target=ctx.target,
        )
        ladder_pred = sl.read_predictions(
            ladder_cells / f"{model}-seed{seed}-fitted.npz",
            ("P", "PEG", "MPEG", "BPLEG", "BMPLEG"),
            sample_ids=ctx.sample_ids,
            target=ctx.target,
        )
        cached = {**prog_pred, **ladder_pred}

        # 1. Direct sl.fit_design check on M, MPEG
        direct_diffs: dict[str, float] = {}
        for dname in ("M", "MPEG"):
            pred, _, _ = sl.fit_design(
                prepared_m, ctx.measured, ctx.aids, ctx.families, ctx.signatures[seed], dname
            )
            diff = float(np.max(np.abs(pred - cached[dname])))
            direct_diffs[dname] = diff
            worst = max(worst, diff)

        # 2. Batched superset solver check on M, MP, MPEG, BMPL, BMPLEG
        x_sup, col_map = _build_non_r_superset(prepared_m)
        batch_designs = {k: col_map[k] for k in ("M", "MP", "MPEG", "BMPL", "BMPLEG")}
        batch_preds, _ = projected_predict_batch(
            x_sup, batch_designs, ctx.measured, ctx.aids, ctx.families, ctx.signatures[seed]
        )
        batch_diffs: dict[str, float] = {}
        for dname, pred in batch_preds.items():
            diff = float(np.max(np.abs(pred - cached[dname])))
            batch_diffs[dname] = diff
            worst = max(worst, diff)

        records[model] = {
            "seed": seed,
            "direct_sl_fit_max_abs_diff": direct_diffs,
            "batched_solver_max_abs_diff": batch_diffs,
        }

    if worst > tolerance:
        raise ValueError(
            f"Reproduction gate FAILED: max_abs_diff={worst:g} exceeds tolerance {tolerance:g}"
        )
    return {
        "status": "passed",
        "tolerance": tolerance,
        "max_abs_diff": worst,
        "models_checked": list(models),
        "seed": seed,
        "details": records,
    }


def _build_non_r_superset(
    prepared: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Assemble ``[B, M, P, L, E, G]`` (572 cols) and exact index map for every R1 design."""
    blocks = ("B", "M", "P", "L", "E", "G")
    slices: dict[str, list[int]] = {}
    cursor = 0
    parts: list[np.ndarray] = []
    for name in blocks:
        arr = prepared[name]
        width = arr.shape[1]
        slices[name] = list(range(cursor, cursor + width))
        parts.append(arr)
        cursor += width
    x_sup = np.column_stack(parts)
    p1 = [slices["P"][0]]
    col_map: dict[str, list[int]] = {
        "M": slices["M"],
        "P1": p1,
        "MP1": slices["M"] + p1,
        "PEG3": p1 + slices["E"] + slices["G"],
        "MPEG3": slices["M"] + p1 + slices["E"] + slices["G"],
        "P": slices["P"],
        "PE": slices["P"] + slices["E"],
        "PG": slices["P"] + slices["G"],
        "PEG": slices["P"] + slices["E"] + slices["G"],
        "MP": slices["M"] + slices["P"],
        "MPE": slices["M"] + slices["P"] + slices["E"],
        "MPG": slices["M"] + slices["P"] + slices["G"],
        "MPEG": slices["M"] + slices["P"] + slices["E"] + slices["G"],
        "BPL": slices["B"] + slices["P"] + slices["L"],
        "BMPL": slices["B"] + slices["M"] + slices["P"] + slices["L"],
        "BPLEG": slices["B"] + slices["P"] + slices["L"] + slices["E"] + slices["G"],
        "BMPLEG": slices["B"] + slices["M"] + slices["P"] + slices["L"] + slices["E"] + slices["G"],
    }
    return x_sup, col_map


# --------------------------------------------------------------------------- #
# Layer 1: Unified 62-Model Zero-Shot + Profile / Potts / GEMME
# --------------------------------------------------------------------------- #


def _equal_family_zero_shot_vector(
    scores: np.ndarray,
    target: np.ndarray,
    aids: np.ndarray,
    families: np.ndarray,
    *,
    exclude_assays: frozenset[str] = frozenset(),
) -> tuple[list[str], np.ndarray, list[str]]:
    """Per-family mean within-assay Spearman vector (equal assays within family)."""
    assays = sorted({str(a) for a in aids if str(a) not in exclude_assays})
    assay_family: dict[str, str] = {}
    per_assay_val: dict[str, float] = {}
    undefined: list[str] = []
    for assay in assays:
        idx = aids == assay
        assay_family[assay] = str(families[idx][0])
        rho = sl.correlation(rankdata(scores[idx]), target[idx])
        if rho is None or not np.isfinite(rho):
            undefined.append(assay)
            per_assay_val[assay] = float("nan")
        else:
            per_assay_val[assay] = float(rho)

    groups: dict[str, list[str]] = {}
    for assay in assays:
        groups.setdefault(assay_family[assay], []).append(assay)
    fam_order = sorted(groups)
    vec = np.asarray(
        [float(np.mean([per_assay_val[a] for a in groups[fam]])) for fam in fam_order],
        dtype=np.float64,
    )
    return fam_order, vec, undefined


def _summarize_contrast_matrix(
    matrix: np.ndarray,
    labels: Sequence[str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run paired native + standardized simultaneous bands on a (groups, K) matrix."""
    mat = np.asarray(matrix, dtype=np.float64)
    native = bidirectional.paired_bands(mat, draws=draws, seed=seed)
    _, pw = sl.pointwise_interval(mat, draws=draws, seed=seed)
    std = responses.simultaneous_bands(mat, draws=draws, seed=seed)
    nat_verdicts = bidirectional.verdict(np.asarray(native["interval"], dtype=np.float64))
    std_verdicts = bidirectional.verdict(np.asarray(std["interval"], dtype=np.float64))
    rows = {}
    for idx, label in enumerate(labels):
        rows[label] = {
            "point": float(native["point"][idx]),
            "pointwise_95": [float(pw[0, idx]), float(pw[1, idx])],
            "simultaneous_native_95": [
                float(native["interval"][idx][0]),
                float(native["interval"][idx][1]),
            ],
            "verdict_native": nat_verdicts[idx],
            "simultaneous_standardized_95": [
                float(std["interval"][idx][0]),
                float(std["interval"][idx][1]),
            ],
            "verdict_standardized": std_verdicts[idx],
        }
    return {
        "n_families": int(mat.shape[0]),
        "family_size": int(mat.shape[1]),
        "draws": int(draws),
        "seed": int(seed),
        "native_critical_value": float(native["critical_value"]),
        "standardized_critical_value": float(std["critical_value"]),
        "verdict_counts_native": bidirectional.counts(nat_verdicts),
        "verdict_counts_standardized": bidirectional.counts(std_verdicts),
        "per_model": rows,
    }


def _compute_layer1_for_subset(
    ctx: R1UnifiedContext,
    *,
    exclude_assays: frozenset[str],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Compute Layer 1 zero-shot baselines, 62 models, and paired contrasts on an assay subset."""
    kept_mask = np.asarray([str(a) not in exclude_assays for a in ctx.aids], dtype=bool)
    n_rows = int(kept_mask.sum())
    n_assays = len({str(a) for a in ctx.aids[kept_mask]})

    fam_order_ref: list[str] | None = None
    model_vecs: dict[str, np.ndarray] = {}
    for arm_id in UNIFIED_62_SINGLE_SEQUENCE_ARMS:
        fams, vec, undef = _equal_family_zero_shot_vector(
            ctx.raw_model_scores[arm_id],
            ctx.target,
            ctx.aids,
            ctx.families,
            exclude_assays=exclude_assays,
        )
        if undef:
            raise ValueError(f"{arm_id}: undefined zero-shot on {undef}")
        if fam_order_ref is None:
            fam_order_ref = fams
        elif fams != fam_order_ref:
            raise ValueError("Family order mismatch in Layer 1")
        model_vecs[arm_id] = vec

    assert fam_order_ref is not None
    n_fams = len(fam_order_ref)

    baseline_vecs: dict[str, np.ndarray] = {}
    baseline_entries: dict[str, Any] = {}
    for bname in ("profile", "potts", "gemme"):
        fams, vec, undef = _equal_family_zero_shot_vector(
            ctx.raw_baselines[bname],
            ctx.target,
            ctx.aids,
            ctx.families,
            exclude_assays=exclude_assays,
        )
        assert fams == fam_order_ref
        if undef:
            zero_filled = np.where(np.isfinite(vec), vec, 0.0)
            _, pw_zf = sl.pointwise_interval(zero_filled[:, None], draws=draws, seed=seed)
            baseline_entries[bname] = {
                "status": "not_estimable_on_refused_assays",
                "undefined_assays": undef,
                "n_undefined_assays": len(undef),
                "diagnostic_zero_filled_point": float(zero_filled.mean()),
                "diagnostic_zero_filled_pointwise_95": [
                    float(pw_zf[0, 0]),
                    float(pw_zf[1, 0]),
                ],
                "note": (
                    "GEMME outputs identically 0.0 across all 3 official runs on these "
                    "wild-type assays (refused_degenerate); exact GEMME zero-shot is "
                    "reported on the GEMME-nondegenerate subset."
                ),
            }
        else:
            baseline_vecs[bname] = vec
            _, pw = sl.pointwise_interval(vec[:, None], draws=draws, seed=seed)
            baseline_entries[bname] = {
                "status": "complete",
                "point": float(vec.mean()),
                "pointwise_95": [float(pw[0, 0]), float(pw[1, 0])],
                "n_families": n_fams,
                "n_assays": n_assays,
            }

    model_mat = np.column_stack([model_vecs[a] for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS])
    _, model_pw = sl.pointwise_interval(model_mat, draws=draws, seed=seed)
    models_out = {
        arm_id: {
            "point": float(model_mat[:, idx].mean()),
            "pointwise_95": [float(model_pw[0, idx]), float(model_pw[1, idx])],
        }
        for idx, arm_id in enumerate(UNIFIED_62_SINGLE_SEQUENCE_ARMS)
    }

    contrasts_out = {
        f"M_minus_{bname}": _summarize_contrast_matrix(
            model_mat - bvec[:, None], UNIFIED_62_SINGLE_SEQUENCE_ARMS, draws=draws, seed=seed
        )
        for bname, bvec in baseline_vecs.items()
    }
    return {
        "n_families": n_fams,
        "n_assays": n_assays,
        "n_rows": n_rows,
        "baselines": baseline_entries,
        "models": models_out,
        "paired_contrasts": contrasts_out,
    }


def compute_layer1_zero_shot(
    ctx: R1UnifiedContext,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Layer 1: 62-model zero-shot Spearman + explicit Profile/Potts/GEMME + category breakdown."""
    all_assays = frozenset(str(a) for a in ctx.aids)
    out: dict[str, Any] = {
        "aggregation": "equal assays within family, equal families",
        "draws": draws,
        "seed": seed,
        "gemme_refused_assays": sorted(GEMME_REFUSED_ASSAYS),
        "functional_categories": list(FUNCTIONAL_CATEGORIES),
        "functional_category_source": str(PROTEINGYM_DMS_CSV),
        "functional_category_source_sha256": ctx.assay_categories_sha256,
        "supports": {},
        "by_functional_category": {},
    }

    support_specs = (
        ("clusters_163", frozenset()),
        ("clusters_156_gemme_nondegenerate", GEMME_REFUSED_ASSAYS),
    )
    for support_key, excl in support_specs:
        out["supports"][support_key] = _compute_layer1_for_subset(
            ctx, exclude_assays=excl, draws=draws, seed=seed
        )
        cat_map: dict[str, Any] = {}
        for cat in FUNCTIONAL_CATEGORIES:
            cat_assays = frozenset(
                a for a in all_assays if a not in excl and ctx.assay_categories[a] == cat
            )
            cat_excl = all_assays - cat_assays
            cat_map[cat] = _compute_layer1_for_subset(
                ctx, exclude_assays=cat_excl, draws=draws, seed=seed
            )
        out["by_functional_category"][support_key] = cat_map
    return out


# --------------------------------------------------------------------------- #
# Layer 2 & Supplementary C0 Fitting + Aggregation (Parallel Worker Functions)
# --------------------------------------------------------------------------- #


def _cache_file(cache_dir: Path, kind: str, arm_id: str, seed: int) -> Path:
    return cache_dir / f"{kind}__{arm_id}__seed{seed}.npz"


def _assay_metrics_for_predictions(
    predictions: Mapping[str, np.ndarray],
    target: np.ndarray,
    aids: np.ndarray,
    families: np.ndarray,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-assay Spearman and rank MSE for every design in ``predictions``."""
    assay_names = sorted({str(a) for a in aids})
    out: dict[str, dict[str, dict[str, float]]] = {}
    for dname, pred in predictions.items():
        sp_map: dict[str, float] = {}
        mse_map: dict[str, float] = {}
        for assay in assay_names:
            idx = aids == assay
            p_sub = pred[idx]
            t_sub = target[idx]
            rho = sl.correlation(rankdata(p_sub), t_sub)
            if rho is None or not np.isfinite(rho):
                raise ValueError(f"{dname}/{assay}: undefined per-assay Spearman")
            sp_map[assay] = float(rho)
            mse_map[assay] = float(np.mean((p_sub - t_sub) ** 2))
        out[dname] = {"spearman": sp_map, "rank_mse": mse_map}
    return out


# Worker globals for fork-free clean multiprocessing on Linux
_WORKER_STATE: dict[str, Any] = {}


def _init_non_r_worker(state: dict[str, Any]) -> None:
    os.environ["OMP_NUM_THREADS"] = str(state["threads"])
    os.environ["MKL_NUM_THREADS"] = str(state["threads"])
    os.environ["OPENBLAS_NUM_THREADS"] = str(state["threads"])
    torch.set_num_threads(int(state["threads"]))
    _WORKER_STATE.clear()
    _WORKER_STATE.update(state)


def _fit_non_r_cell_worker(task: tuple[str, int, tuple[str, ...]]) -> tuple[str, int, str]:
    """Fit requested non-R designs for one ``(arm_id, seed)`` and write cached ``.npz``."""
    arm_id, seed, design_names = task
    st = _WORKER_STATE
    cache_path = _cache_file(Path(st["cache_dir"]), "non_r", arm_id, seed)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as d:
            if set(design_names) <= set(d.files):
                return arm_id, seed, "cached"

    prepared = dict(
        st["prepared_shared"],
        M=sl.prepare_block("M", st["raw_scores"][arm_id][:, None], st["aids"]),
    )
    if set(design_names) <= {"MP1", "MPEG3"}:
        # Tiny 4-column superset [M, P1, E, G] for the 33 pre-registered arms (~0.3s!)
        x_sup = np.column_stack(
            [prepared["M"], prepared["P1"], prepared["E"], prepared["G"]]
        )
        col_map = {"MP1": [0, 1], "MPEG3": [0, 1, 2, 3]}
    else:
        x_sup, col_map = _build_non_r_superset(prepared)
    subset = {name: col_map[name] for name in design_names}
    preds, _ = projected_predict_batch(
        x_sup,
        subset,
        st["measured"],
        st["aids"],
        st["families"],
        st["signatures"][seed],
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez_compressed(tmp_path, **preds)
    os.replace(tmp_path, cache_path)
    return arm_id, seed, "fitted"


def _fit_r_bridge_cell_worker(task: tuple[str, int, str]) -> tuple[str, int, str]:
    """Fit all 6 ``R`` representation-bridge designs for one ``(arm_id, seed)``."""
    arm_id, seed, readout_npz_str = task
    st = _WORKER_STATE
    cache_path = _cache_file(Path(st["cache_dir"]), "r_bridge", arm_id, seed)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as d:
            if set(R_BRIDGE_DESIGNS) <= set(d.files):
                return arm_id, seed, "cached"

    with np.load(readout_npz_str, allow_pickle=False) as ro:
        r_mat = np.asarray(ro["R"], dtype=np.float64)
        sid_key = "sample_ids" if "sample_ids" in ro.files else ("sample_id" if "sample_id" in ro.files else None)
        if sid_key is not None:
            ro_sids = [str(x) for x in ro[sid_key]]
            exp_sids = [str(x) for x in st["sample_ids"]]
            if ro_sids != exp_sids:
                if len(set(ro_sids)) != len(ro_sids) or set(ro_sids) != set(exp_sids):
                    raise ValueError(f"{arm_id}: readout sample_id set mismatch in {readout_npz_str}")
                pos = {sid: idx for idx, sid in enumerate(ro_sids)}
                r_mat = r_mat[[pos[sid] for sid in exp_sids]]
        if "assay" in ro.files and not np.array_equal(ro["assay"], st["aids"]):
            raise ValueError(f"{arm_id}: readout assay order disagrees with frozen R1 support")
        if "y_rank" in ro.files and float(np.max(np.abs(ro["y_rank"] - st["target"]))) > 1e-10:
            raise ValueError(f"{arm_id}: readout y_rank disagrees with frozen R1 target")
        if r_mat.shape != (len(st["aids"]), 1024) or not np.isfinite(r_mat).all():
            raise ValueError(f"{arm_id}: invalid 1024-d R matrix shape={r_mat.shape}")

    m_col = sl.prepare_block("M", st["raw_scores"][arm_id][:, None], st["aids"])
    p_blk = st["prepared_shared"]["P"]  # 14 cols, col 0 is reranked P1
    e_col = st["prepared_shared"]["E"]  # 1 col
    g_col = st["prepared_shared"]["G"]  # 1 col
    # Superset: [M (0), P (1..14), E (15), G (16), R (17..1040)] -> width 1041
    x_sup = np.column_stack([m_col, p_blk, e_col, g_col, r_mat])
    r_idx = list(range(17, 17 + 1024))
    col_map = {
        "R": r_idx,
        "P1_R": [1] + r_idx,
        "P1_M_R": [0, 1] + r_idx,
        "PEG3_R": [1, 15, 16] + r_idx,
        "PEG3_M_R": [0, 1, 15, 16] + r_idx,
        "PEG16_M_R": [0] + list(range(1, 17)) + r_idx,
    }
    preds, _ = projected_predict_batch(
        x_sup,
        col_map,
        st["measured"],
        st["aids"],
        st["families"],
        st["signatures"][seed],
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez_compressed(tmp_path, **preds)
    os.replace(tmp_path, cache_path)
    return arm_id, seed, "fitted"


def fit_all_r1_cells(
    ctx: R1UnifiedContext,
    cache_dir: Path,
    *,
    workers: int = 4,
    threads: int = 2,
    include_r_bridge: bool = True,
) -> None:
    """Fit and cache all new R1 cells (shared scalar baselines, 62 non-R, 28 +R)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)

    # 1. Shared model-independent scalar controls (P1, PEG3) for each of the 3 seeds.
    x_shared = np.column_stack(
        [ctx.prepared_shared["P1"], ctx.prepared_shared["E"], ctx.prepared_shared["G"]]
    )
    shared_cols = {"P1": [0], "PEG3": [0, 1, 2]}
    for seed in sl.SEEDS:
        cpath = _cache_file(cache_dir, "shared_scalar", "shared", seed)
        if cpath.is_file():
            continue
        preds, _ = projected_predict_batch(
            x_shared, shared_cols, ctx.measured, ctx.aids, ctx.families, ctx.signatures[seed]
        )
        tmp = cpath.with_name(cpath.stem + ".tmp.npz")
        np.savez_compressed(tmp, **preds)
        os.replace(tmp, cpath)

    worker_state = {
        "threads": threads,
        "cache_dir": str(cache_dir),
        "prepared_shared": ctx.prepared_shared,
        "raw_scores": ctx.raw_model_scores,
        "sample_ids": ctx.sample_ids,
        "measured": ctx.measured,
        "target": ctx.target,
        "aids": ctx.aids,
        "families": ctx.families,
        "signatures": ctx.signatures,
    }

    # 2. Non-R designs: 33 pre-registered (MP1, MPEG3) + 29 expanded (9 designs) x 3 seeds.
    non_r_tasks: list[tuple[str, int, tuple[str, ...]]] = []
    for arm_id in PREREGISTERED_33_ARMS:
        for seed in sl.SEEDS:
            non_r_tasks.append((arm_id, seed, PREREG_SCALAR_DESIGNS))
    for arm_id in EXPANDED_29_SINGLE_ARMS:
        for seed in sl.SEEDS:
            non_r_tasks.append((arm_id, seed, EXPANDED_NON_R_DESIGNS))

    n_workers = max(1, min(int(workers), 4))
    mp_ctx = multiprocessing.get_context("spawn")
    if n_workers == 1:
        _init_non_r_worker(worker_state)
        for task in non_r_tasks:
            _fit_non_r_cell_worker(task)
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_ctx,
            initializer=_init_non_r_worker,
            initargs=(worker_state,),
        ) as pool:
            futs = [pool.submit(_fit_non_r_cell_worker, task) for task in non_r_tasks]
            done = 0
            for fut in as_completed(futs):
                fut.result()
                done += 1
                if done % 30 == 0 or done == len(futs):
                    print(f"[R1-unified:non-R] {done}/{len(futs)} cells ready", flush=True)

    if not include_r_bridge:
        return

    # 3. 1024-d R bridge designs: 28 (or 29 w/ esm3-1.4b) expanded arms x 3 seeds.
    r_tasks = [
        (arm_id, seed, str(ctx.gp_readout_paths[arm_id]))
        for arm_id in ctx.r_bridge_arms
        for seed in sl.SEEDS
    ]
    if n_workers == 1:
        _init_non_r_worker(worker_state)
        for task in r_tasks:
            _fit_r_bridge_cell_worker(task)
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_ctx,
            initializer=_init_non_r_worker,
            initargs=(worker_state,),
        ) as pool:
            futs = [pool.submit(_fit_r_bridge_cell_worker, task) for task in r_tasks]
            done = 0
            for fut in as_completed(futs):
                fut.result()
                done += 1
                if done % 12 == 0 or done == len(futs):
                    print(f"[R1-unified:+R] {done}/{len(futs)} cells ready", flush=True)


def _load_per_assay_metrics_table(
    ctx: R1UnifiedContext,
    cache_dir: Path,
    *,
    include_r_bridge: bool = True,
) -> tuple[
    dict[tuple[str, int, str], dict[str, dict[str, float]]],
    dict[str, str],
    list[str],
]:
    """Assemble exact per-assay ``(spearman, rank_mse)`` for all 62 arms x 3 seeds x all designs."""
    ladder_cells = ctx.root / LADDER_FULL_DIR / "cells"
    prog_full = ctx.root / PROGRESSION_DIR / "full"

    assay_names = sorted({str(a) for a in ctx.aids})
    assay_family = {
        assay: str(ctx.families[ctx.aids == assay][0]) for assay in assay_names
    }

    # Shared model-independent controls per seed: P1, PEG3 (fitted) + P, PE, PG, PEG, BPL, BPLEG (cached)
    anchor_model = PREREGISTERED_33_ARMS[0]
    shared_metrics: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
    for seed in sl.SEEDS:
        preds_shared: dict[str, np.ndarray] = {}
        with np.load(_cache_file(cache_dir, "shared_scalar", "shared", seed), allow_pickle=False) as d:
            for k in SHARED_SCALAR_DESIGNS:
                preds_shared[k] = np.asarray(d[k], dtype=np.float64)
        preds_shared.update(
            sl.read_predictions(
                ladder_cells / f"{anchor_model}-seed{seed}-fitted.npz",
                CACHED_CONTROL_DESIGNS_LADDER,
                sample_ids=ctx.sample_ids,
                target=ctx.target,
            )
        )
        preds_shared.update(
            sl.read_predictions(
                prog_full / f"{anchor_model}-seed{seed}.npz",
                CACHED_CONTROL_DESIGNS_PROG,
                sample_ids=ctx.sample_ids,
                target=ctx.target,
            )
        )
        shared_metrics[seed] = _assay_metrics_for_predictions(
            preds_shared, ctx.target, ctx.aids, ctx.families
        )

    # Per-(arm, seed, design) -> {metric: {assay: float}}
    metrics_by_cell: dict[tuple[str, int, str], dict[str, dict[str, float]]] = {}
    for arm_id in UNIFIED_62_SINGLE_SEQUENCE_ARMS:
        is_prereg = arm_id in PREREGISTERED_33_ARMS
        for seed in sl.SEEDS:
            # Attach shared controls
            for dname, mdict in shared_metrics[seed].items():
                metrics_by_cell[(arm_id, seed, dname)] = mdict

            preds_model: dict[str, np.ndarray] = {}
            if is_prereg:
                preds_model.update(
                    sl.read_predictions(
                        prog_full / f"{arm_id}-seed{seed}.npz",
                        CACHED_MODEL_DESIGNS_PROG,
                        sample_ids=ctx.sample_ids,
                        target=ctx.target,
                    )
                )
                preds_model.update(
                    sl.read_predictions(
                        ladder_cells / f"{arm_id}-seed{seed}-fitted.npz",
                        CACHED_MODEL_DESIGNS_LADDER,
                        sample_ids=ctx.sample_ids,
                        target=ctx.target,
                    )
                )
                with np.load(_cache_file(cache_dir, "non_r", arm_id, seed), allow_pickle=False) as d:
                    for k in PREREG_SCALAR_DESIGNS:
                        preds_model[k] = np.asarray(d[k], dtype=np.float64)
            else:
                with np.load(_cache_file(cache_dir, "non_r", arm_id, seed), allow_pickle=False) as d:
                    for k in EXPANDED_NON_R_DESIGNS:
                        preds_model[k] = np.asarray(d[k], dtype=np.float64)
                if include_r_bridge and arm_id in ctx.r_bridge_arms:
                    with np.load(
                        _cache_file(cache_dir, "r_bridge", arm_id, seed), allow_pickle=False
                    ) as d:
                        for k in R_BRIDGE_DESIGNS:
                            preds_model[k] = np.asarray(d[k], dtype=np.float64)

            cell_m = _assay_metrics_for_predictions(
                preds_model, ctx.target, ctx.aids, ctx.families
            )
            for dname, mdict in cell_m.items():
                metrics_by_cell[(arm_id, seed, dname)] = mdict

    return metrics_by_cell, assay_family, assay_names


def _family_matrix_for_designs(
    metrics_by_cell: Mapping[tuple[str, int, str], Mapping[str, Mapping[str, float]]],
    assay_family: Mapping[str, str],
    models: Sequence[str],
    designs: Sequence[str],
    metric: str,
    *,
    exclude_assays: frozenset[str] = frozenset(),
) -> tuple[list[str], dict[tuple[str, str], np.ndarray]]:
    """Aggregate (seed -> assay -> equal family) vector for each ``(model, design)``."""
    groups: dict[str, list[str]] = {}
    for assay, fam in sorted(assay_family.items()):
        if assay in exclude_assays:
            continue
        groups.setdefault(fam, []).append(assay)
    fam_order = sorted(groups)
    out: dict[tuple[str, str], np.ndarray] = {}
    for model in models:
        for dname in designs:
            vals = np.empty(len(fam_order), dtype=np.float64)
            for r_idx, fam in enumerate(fam_order):
                per_assay = [
                    float(
                        np.mean(
                            [metrics_by_cell[(model, s, dname)][metric][a] for s in sl.SEEDS]
                        )
                    )
                    for a in groups[fam]
                ]
                vals[r_idx] = float(np.mean(per_assay))
            out[(model, dname)] = vals
    return fam_order, out


def _evaluate_bidirectional_family(
    metrics_by_cell: Mapping[tuple[str, int, str], Mapping[str, Mapping[str, float]]],
    assay_family: Mapping[str, str],
    models: Sequence[str],
    control_sets: Mapping[str, Mapping[str, str]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate a bidirectional contrast family across both metrics and both 163/156 supports."""
    all_designs = sorted(
        {
            d
            for spec in control_sets.values()
            for d in (spec["control"], spec["model"], spec["joint"])
        }
    )
    support_specs = (
        ("clusters_163", frozenset(), 163, 201, 25728),
        ("clusters_156_gemme_nondegenerate", GEMME_REFUSED_ASSAYS, 156, 194, 24832),
    )
    results: dict[str, Any] = {}
    for support_key, excl, exp_fams, exp_assays, exp_rows in support_specs:
        sup_out: dict[str, Any] = {
            "n_families": exp_fams,
            "n_assays": exp_assays,
            "n_rows": exp_rows,
            "metrics": {},
        }
        for metric in sl.METRICS:
            sign = 1.0 if metric == "spearman" else -1.0
            fams, fam_vecs = _family_matrix_for_designs(
                metrics_by_cell,
                assay_family,
                models,
                all_designs,
                metric,
                exclude_assays=excl,
            )
            assert len(fams) == exp_fams

            # Contrast matrix: models x control_sets x 2 directions
            labels: list[tuple[str, str, str]] = []
            cols: list[np.ndarray] = []
            for model in models:
                for cset_name, spec in control_sets.items():
                    joint_v = fam_vecs[(model, spec["joint"])]
                    ctrl_v = fam_vecs[(model, spec["control"])]
                    mod_v = fam_vecs[(model, spec["model"])]
                    labels.append((model, cset_name, "model_given_controls"))
                    cols.append(sign * (joint_v - ctrl_v))
                    labels.append((model, cset_name, "controls_given_model"))
                    cols.append(sign * (joint_v - mod_v))
            cmat = np.column_stack(cols)
            native = bidirectional.paired_bands(cmat, draws=draws, seed=seed)
            _, pw = sl.pointwise_interval(cmat, draws=draws, seed=seed)
            std = responses.simultaneous_bands(cmat, draws=draws, seed=seed)
            nat_v = bidirectional.verdict(np.asarray(native["interval"], dtype=np.float64))
            std_v = bidirectional.verdict(np.asarray(std["interval"], dtype=np.float64))

            hypotheses = []
            for idx, (model, cset_name, direction) in enumerate(labels):
                hypotheses.append(
                    {
                        "model": model,
                        "arm": cset_name,
                        "direction": direction,
                        "metric": metric,
                        "point": float(native["point"][idx]),
                        "pointwise_95": [float(pw[0, idx]), float(pw[1, idx])],
                        "simultaneous_native_95": [
                            float(native["interval"][idx][0]),
                            float(native["interval"][idx][1]),
                        ],
                        "verdict_native": nat_v[idx],
                        "simultaneous_standardized_95": [
                            float(std["interval"][idx][0]),
                            float(std["interval"][idx][1]),
                        ],
                        "verdict_standardized": std_v[idx],
                    }
                )

            # Absolute design means + pointwise 95% CIs
            abs_keys = [(m, d) for m in models for d in all_designs]
            abs_mat = np.column_stack([fam_vecs[k] for k in abs_keys])
            _, abs_pw = sl.pointwise_interval(abs_mat, draws=draws, seed=seed)
            absolute = {
                f"{m}|{d}": {
                    "point": float(abs_mat[:, i].mean()),
                    "pointwise_95": [float(abs_pw[0, i]), float(abs_pw[1, i])],
                }
                for i, (m, d) in enumerate(abs_keys)
            }

            sup_out["metrics"][metric] = {
                "hypothesis_count": len(labels),
                "native": {
                    "method": native["method"],
                    "draws": int(native["draws"]),
                    "seed": int(native["seed"]),
                    "confidence": float(native["confidence"]),
                    "family_size": int(native["family_size"]),
                    "joint_critical_max_absolute_deviation": float(native["critical_value"]),
                    "verdict_counts": bidirectional.counts(nat_v),
                },
                "standardized_maxT": {
                    "method": std["method"],
                    "draws": int(std["draws"]),
                    "seed": int(std["seed"]),
                    "family_size": int(std["family_size"]),
                    "joint_critical_standardized": float(std["critical_value"]),
                    "verdict_counts": bidirectional.counts(std_v),
                },
                "hypotheses": hypotheses,
                "absolute": absolute,
            }
        results[support_key] = sup_out
    return results


def _compute_layer2_by_functional_category(
    ctx: R1UnifiedContext,
    metrics_by_cell: Mapping[tuple[str, int, str], Mapping[str, Mapping[str, float]]],
    assay_family: Mapping[str, str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute Layer 2 ``+M`` (62 arms) and ``+R`` / ``+(M+R)`` (28/29 arms) per selection type."""
    all_assays = frozenset(assay_family.keys())
    r_arms = ctx.r_bridge_arms
    m_designs = ["M", "P1", "MP1", "PEG3", "MPEG3", "P", "MP", "PEG", "MPEG"]
    r_designs = list(R_BRIDGE_DESIGNS) + ["M", "P1", "MP1", "PEG3", "MPEG3", "PEG", "MPEG"]

    m_contrasts = (
        ("M_given_P1", "MP1", "P1"),
        ("P1_given_M", "MP1", "M"),
        ("M_given_PEG3", "MPEG3", "PEG3"),
        ("PEG3_given_M", "MPEG3", "M"),
        ("M_given_P14", "MP", "P"),
        ("M_given_PEG16", "MPEG", "PEG"),
    )
    r_contrasts = (
        ("R_given_P1", "P1_R", "P1"),
        ("MR_given_P1", "P1_M_R", "P1"),
        ("R_given_P1_M", "P1_M_R", "MP1"),
        ("R_given_PEG3", "PEG3_R", "PEG3"),
        ("MR_given_PEG3", "PEG3_M_R", "PEG3"),
        ("R_given_PEG3_M", "PEG3_M_R", "MPEG3"),
        ("MR_given_PEG16", "PEG16_M_R", "PEG"),
        ("R_given_PEG16_M", "PEG16_M_R", "MPEG"),
        ("R_minus_M", "R", "M"),
    )

    out: dict[str, Any] = {}
    for support_key, base_excl in (
        ("clusters_163", frozenset()),
        ("clusters_156_gemme_nondegenerate", GEMME_REFUSED_ASSAYS),
    ):
        cat_out: dict[str, Any] = {}
        for cat in FUNCTIONAL_CATEGORIES:
            cat_assays = frozenset(
                a for a in all_assays if a not in base_excl and ctx.assay_categories[a] == cat
            )
            cat_excl = all_assays - cat_assays
            n_rows = int(np.isin(ctx.aids, list(cat_assays)).sum())
            n_fams = len({assay_family[a] for a in cat_assays})

            m_metrics: dict[str, Any] = {}
            r_metrics: dict[str, Any] = {}
            for metric in sl.METRICS:
                sign = 1.0 if metric == "spearman" else -1.0

                # 62-model scalar M increments
                _, m_vecs = _family_matrix_for_designs(
                    metrics_by_cell,
                    assay_family,
                    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
                    m_designs,
                    metric,
                    exclude_assays=cat_excl,
                )
                m_c_out = {}
                for cname, aug, base_d in m_contrasts:
                    cmat = np.column_stack(
                        [
                            sign * (m_vecs[(m, aug)] - m_vecs[(m, base_d)])
                            for m in UNIFIED_62_SINGLE_SEQUENCE_ARMS
                        ]
                    )
                    m_c_out[cname] = _summarize_contrast_matrix(
                        cmat, UNIFIED_62_SINGLE_SEQUENCE_ARMS, draws=draws, seed=seed
                    )
                abs_m_keys = [(m, d) for m in UNIFIED_62_SINGLE_SEQUENCE_ARMS for d in ("P1", "PEG3", "M", "MP1", "MPEG3")]
                abs_m_mat = np.column_stack([m_vecs[k] for k in abs_m_keys])
                _, abs_m_pw = sl.pointwise_interval(abs_m_mat, draws=draws, seed=seed)
                m_metrics[metric] = {
                    "contrasts": m_c_out,
                    "absolute": {
                        f"{m}|{d}": {
                            "point": float(abs_m_mat[:, i].mean()),
                            "pointwise_95": [float(abs_m_pw[0, i]), float(abs_m_pw[1, i])],
                        }
                        for i, (m, d) in enumerate(abs_m_keys)
                    },
                }

                # 28/29-model +R / +(M+R) increments
                _, r_vecs = _family_matrix_for_designs(
                    metrics_by_cell,
                    assay_family,
                    r_arms,
                    r_designs,
                    metric,
                    exclude_assays=cat_excl,
                )
                r_c_out = {}
                for cname, aug, base_d in r_contrasts:
                    cmat = np.column_stack(
                        [sign * (r_vecs[(m, aug)] - r_vecs[(m, base_d)]) for m in r_arms]
                    )
                    r_c_out[cname] = _summarize_contrast_matrix(
                        cmat, r_arms, draws=draws, seed=seed
                    )
                abs_r_keys = [(m, d) for m in r_arms for d in R_BRIDGE_DESIGNS]
                abs_r_mat = np.column_stack([r_vecs[k] for k in abs_r_keys])
                _, abs_r_pw = sl.pointwise_interval(abs_r_mat, draws=draws, seed=seed)
                r_metrics[metric] = {
                    "contrasts": r_c_out,
                    "absolute": {
                        f"{m}|{d}": {
                            "point": float(abs_r_mat[:, i].mean()),
                            "pointwise_95": [float(abs_r_pw[0, i]), float(abs_r_pw[1, i])],
                        }
                        for i, (m, d) in enumerate(abs_r_keys)
                    },
                }

            cat_out[cat] = {
                "n_families": n_fams,
                "n_assays": len(cat_assays),
                "n_rows": n_rows,
                "n_models_M": len(UNIFIED_62_SINGLE_SEQUENCE_ARMS),
                "n_models_R": len(r_arms),
                "scalar_M_increments": m_metrics,
                "representation_R_increments": r_metrics,
            }
        out[support_key] = cat_out
    return out


def compute_layer2_core_bidirectional(
    ctx: R1UnifiedContext,
    metrics_by_cell: Mapping[tuple[str, int, str], Mapping[str, Mapping[str, float]]],
    assay_family: Mapping[str, str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Layer 2 Core Bidirectional Increments (248-contrast families + functional breakdown)."""
    primary_scalar_sets = {
        "P_1col": {"control": "P1", "model": "M", "joint": "MP1"},
        "PEG_3col": {"control": "PEG3", "model": "M", "joint": "MPEG3"},
    }
    profile_block_sets = {
        "P_14col": {"control": "P", "model": "M", "joint": "MP"},
        "PEG_16col": {"control": "PEG", "model": "M", "joint": "MPEG"},
    }
    return {
        "seeds": list(sl.SEEDS),
        "n_models": len(UNIFIED_62_SINGLE_SEQUENCE_ARMS),
        "models": list(UNIFIED_62_SINGLE_SEQUENCE_ARMS),
        "contrast_sign": "joint minus reduced for spearman; reduced minus joint for rank_mse",
        "support_restriction_note": (
            "All nested-CV Ridge models are fitted on the full 25,728-row / 163-cluster support "
            "(preserving bit-exact reuse of the 33 pre-registered cached cells). "
            "clusters_156_gemme_nondegenerate restricts held-out evaluation to the 194 assays / "
            "156 family clusters where GEMME is non-degenerate (see "
            "explicit_pair_ablation_20261114/gemme-refusal-probe: training-set refit shift <= 0.0005)."
        ),
        "primary_scalar_baselines": {
            "description": (
                "Literal single-column P_1col (reranked P[:, 0], matching zero-shot P=0.361) "
                "and 3-column PEG_3col ([P_1col, E, G])."
            ),
            "control_sets": primary_scalar_sets,
            "supports": _evaluate_bidirectional_family(
                metrics_by_cell,
                assay_family,
                UNIFIED_62_SINGLE_SEQUENCE_ARMS,
                primary_scalar_sets,
                draws=draws,
                seed=seed,
            ),
        },
        "profile_block_baselines": {
            "description": (
                "Historical 14-column P_14col block and 16-column PEG_16col block (P_14col + E + G), "
                "reusing all 33 pre-registered cached cells bit-for-bit."
            ),
            "control_sets": profile_block_sets,
            "supports": _evaluate_bidirectional_family(
                metrics_by_cell,
                assay_family,
                UNIFIED_62_SINGLE_SEQUENCE_ARMS,
                profile_block_sets,
                draws=draws,
                seed=seed,
            ),
        },
        "by_functional_category": _compute_layer2_by_functional_category(
            ctx, metrics_by_cell, assay_family, draws=draws, seed=seed
        ),
    }


def _load_prereg_r_progression_context(root: Path) -> dict[str, Any]:
    """Attach the 33 pre-registered arms' cached ``R`` and ``B_R - B`` R1/R2 progression summary."""
    tsv_path = root / READOUT_PANEL_DIR / "panel_increments.tsv"
    if not tsv_path.is_file():
        return {"status": "missing", "source": str(READOUT_PANEL_DIR / "panel_increments.tsv")}
    df = pd.read_csv(tsv_path, sep="\t")
    anchor = df.loc[df["panel"] == "anchor201"].copy()
    arms_out: dict[str, Any] = {}
    for arm_id in PREREGISTERED_33_ARMS:
        sub = anchor.loc[anchor["arm"] == arm_id]
        if sub.empty:
            continue
        rec: dict[str, Any] = {}
        for metric_name, grp in sub.groupby("metric"):
            rec[str(metric_name)] = {
                "mean_point_3seeds": float(grp["point"].mean()),
                "per_seed": {
                    str(int(r.seed)): {
                        "point": float(r.point),
                        "lower": float(r.lower),
                        "upper": float(r.upper),
                        "interval_state": str(r.interval_state),
                    }
                    for r in grp.itertuples(index=False)
                },
            }
        arms_out[arm_id] = rec
    return {
        "source": str(READOUT_PANEL_DIR / "panel_increments.tsv"),
        "source_sha256": sha256_file(tsv_path),
        "panel": "anchor201 (201 assays, 163 family clusters)",
        "n_arms": len(arms_out),
        "note": (
            "Cached 33-arm pre-registered frozen-representation readout increments on the identical "
            "201-assay / 163-family support: R_minus_raw_M_spearman = R(1024-d) vs raw M; "
            "delta_spearman & delta_rank_mse = B_R (B+M+P+R) minus B (B+M+P)."
        ),
        "arms": arms_out,
    }


def compute_layer2_representation_bridge(
    ctx: R1UnifiedContext,
    metrics_by_cell: Mapping[tuple[str, int, str], Mapping[str, Mapping[str, float]]],
    assay_family: Mapping[str, str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Layer 2 ``+R`` Bridge to Result 2 for all 28 (or 29 w/ esm3-1.4b) expanded arms."""
    r_arms = ctx.r_bridge_arms
    r_designs = list(R_BRIDGE_DESIGNS) + ["M", "P1", "MP1", "PEG3", "MPEG3", "PEG", "MPEG"]
    contrast_defs = (
        ("R_given_P1", "P1_R", "P1"),
        ("MR_given_P1", "P1_M_R", "P1"),
        ("R_given_P1_M", "P1_M_R", "MP1"),
        ("R_given_PEG3", "PEG3_R", "PEG3"),
        ("MR_given_PEG3", "PEG3_M_R", "PEG3"),
        ("R_given_PEG3_M", "PEG3_M_R", "MPEG3"),
        ("MR_given_PEG16", "PEG16_M_R", "PEG"),
        ("R_given_PEG16_M", "PEG16_M_R", "MPEG"),
        ("R_minus_M", "R", "M"),
    )
    support_specs = (
        ("clusters_163", frozenset(), 163, 201, 25728),
        ("clusters_156_gemme_nondegenerate", GEMME_REFUSED_ASSAYS, 156, 194, 24832),
    )
    supports_out: dict[str, Any] = {}
    for support_key, excl, exp_fams, exp_assays, exp_rows in support_specs:
        sup_rec: dict[str, Any] = {
            "n_families": exp_fams,
            "n_assays": exp_assays,
            "n_rows": exp_rows,
            "metrics": {},
        }
        for metric in sl.METRICS:
            sign = 1.0 if metric == "spearman" else -1.0
            fams, fam_vecs = _family_matrix_for_designs(
                metrics_by_cell,
                assay_family,
                r_arms,
                r_designs,
                metric,
                exclude_assays=excl,
            )
            assert len(fams) == exp_fams

            # Absolute design performance for all expanded arms with R
            abs_keys = [(m, d) for m in r_arms for d in R_BRIDGE_DESIGNS]
            abs_mat = np.column_stack([fam_vecs[k] for k in abs_keys])
            _, abs_pw = sl.pointwise_interval(abs_mat, draws=draws, seed=seed)
            absolute = {
                f"{m}|{d}": {
                    "point": float(abs_mat[:, i].mean()),
                    "pointwise_95": [float(abs_pw[0, i]), float(abs_pw[1, i])],
                }
                for i, (m, d) in enumerate(abs_keys)
            }

            # Per-contrast-type simultaneous families + one joint family
            per_contrast: dict[str, Any] = {}
            all_cols: list[np.ndarray] = []
            all_labels: list[str] = []
            for cname, aug, base_d in contrast_defs:
                cols = [sign * (fam_vecs[(m, aug)] - fam_vecs[(m, base_d)]) for m in r_arms]
                cmat = np.column_stack(cols)
                per_contrast[cname] = _summarize_contrast_matrix(
                    cmat, r_arms, draws=draws, seed=seed
                )
                for m, col in zip(r_arms, cols):
                    all_cols.append(col)
                    all_labels.append(f"{m}|{cname}")

            joint_family = _summarize_contrast_matrix(
                np.column_stack(all_cols), all_labels, draws=draws, seed=seed
            )
            sup_rec["metrics"][metric] = {
                "absolute": absolute,
                "contrasts": per_contrast,
                "joint_family_all_contrasts": {
                    "family_size": joint_family["family_size"],
                    "native_critical_value": joint_family["native_critical_value"],
                    "standardized_critical_value": joint_family["standardized_critical_value"],
                    "verdict_counts_native": joint_family["verdict_counts_native"],
                    "verdict_counts_standardized": joint_family["verdict_counts_standardized"],
                },
            }
        supports_out[support_key] = sup_rec

    return {
        "n_expanded_arms_with_R": len(r_arms),
        "expanded_arms_with_R": list(r_arms),
        "includes_esm3_1_4b": "esm3-1.4b" in r_arms,
        "designs": list(R_BRIDGE_DESIGNS),
        "seeds": list(sl.SEEDS),
        "readout_provenance": {
            arm_id: {
                "path": str(ctx.gp_readout_paths[arm_id].relative_to(ctx.root)),
                "dtype_realised": ctx.gp_readout_dtypes[arm_id],
            }
            for arm_id in r_arms
        },
        "expanded_28_arms": supports_out,
        "preregistered_33_arms_cached_progression": _load_prereg_r_progression_context(ctx.root),
    }


def compute_supplementary_c0_ladder(
    ctx: R1UnifiedContext,
    metrics_by_cell: Mapping[tuple[str, int, str], Mapping[str, Mapping[str, float]]],
    assay_family: Mapping[str, str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Supplementary 6-rung C0 ladder (744-contrast 62-arm family) + historical 33-arm 396 band."""
    hist_path = ctx.root / LADDER_FULL_DIR / "contrasts.json"
    historical_396 = json.loads(hist_path.read_text(encoding="utf-8"))
    return {
        "schema": "r1_supplementary_c0_ladder_62_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "support_sha256": ctx.support_sha256,
        "n_models": len(UNIFIED_62_SINGLE_SEQUENCE_ARMS),
        "models": list(UNIFIED_62_SINGLE_SEQUENCE_ARMS),
        "arm_metadata": {a: _arm_meta_dict(a) for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS},
        "control_sets": {k: dict(v) for k, v in sl.CONTROL_SETS.items()},
        "seeds": list(sl.SEEDS),
        "hypotheses_per_metric": len(UNIFIED_62_SINGLE_SEQUENCE_ARMS) * len(sl.CONTROL_SETS) * 2,
        "contrast_sign": "joint minus reduced for spearman; reduced minus joint for rank_mse",
        "unified_62_arm_744_band": _evaluate_bidirectional_family(
            metrics_by_cell,
            assay_family,
            UNIFIED_62_SINGLE_SEQUENCE_ARMS,
            sl.CONTROL_SETS,
            draws=draws,
            seed=seed,
        ),
        "historical_preregistered_33_arm_396_band": {
            "source": str(LADDER_FULL_DIR / "contrasts.json"),
            "source_sha256": sha256_file(hist_path),
            "note": "Embedded verbatim from frozen 33-arm pre-registered statistical ladder; untouched.",
            "payload": historical_396,
        },
    }


# --------------------------------------------------------------------------- #
# Unified 62-Model Explicit Pair-Level Ablation + 37-Arm Expanded T_match
# --------------------------------------------------------------------------- #


def _pair_tier_label(name: str, tolerance: float | None = None) -> str:
    return name if tolerance is None else f"{name}({tolerance:g})"


def _pair_orient(pairs: pd.DataFrame) -> pd.DataFrame:
    frame = pairs.copy()
    left = frame["z_a"].to_numpy(dtype=np.float64)
    right = frame["z_b"].to_numpy(dtype=np.float64)
    if (left == right).any():
        raise ValueError("candidate pair has no phenotype gap")
    better_is_a = left > right
    frame["better_sample"] = np.where(better_is_a, frame["sample_a"], frame["sample_b"])
    frame["worse_sample"] = np.where(better_is_a, frame["sample_b"], frame["sample_a"])
    return frame


def _pair_attach_covariate(
    pairs: pd.DataFrame, name: str, ranks: Mapping[str, float]
) -> pd.DataFrame:
    frame = pairs.copy()
    values = md.direction_values(frame, ranks)
    better = frame["better_sample"].map(ranks).to_numpy(dtype=np.float64)
    worse = frame["worse_sample"].map(ranks).to_numpy(dtype=np.float64)
    frame[f"{name}_distance"] = np.abs(better - worse)
    frame[f"{name}_direction"] = values
    frame[f"{name}_vote"] = np.where(values > 0.5, 1, np.where(values < 0.5, -1, 0)).astype(np.int8)
    return frame


def _pair_tolerance_matched(
    candidates: pd.DataFrame, names: Sequence[str], tolerance: float
) -> pd.DataFrame:
    mask = np.ones(len(candidates), dtype=bool)
    for name in names:
        mask &= candidates[f"{name}_distance"].to_numpy(dtype=np.float64) <= tolerance
    return candidates.loc[mask].reset_index(drop=True)


def _pair_signature(names: Sequence[str], signs: Sequence[int]) -> str:
    return ",".join(f"{name}{'+' if sign > 0 else '-'}" for name, sign in zip(names, signs))


def _pair_balance_votes(
    pairs: pd.DataFrame, names: Sequence[str], *, seed: int = PAIR_BALANCE_SEED
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exact k-vote sign balance matching ``explicit_pair_ablation_20261114`` byte-for-byte."""
    if len(pairs) == 0:
        return pairs, {"balanced_votes": list(names), "status": "no_pairs", "retained_pairs": 0}
    ordered = pairs.sort_values(list(PAIR_CANONICAL_ORDER), kind="mergesort").reset_index(drop=True)
    votes = np.stack(
        [ordered[f"{name}_vote"].to_numpy(dtype=np.int64) for name in names], axis=1
    )
    neutral = votes == 0
    all_neutral = neutral.all(axis=1)
    partial = neutral.any(axis=1) & ~all_neutral
    cells: dict[tuple[int, ...], list[int]] = {}
    for pos in range(len(ordered)):
        if all_neutral[pos] or partial[pos]:
            continue
        cells.setdefault(tuple(int(v) for v in votes[pos]), []).append(pos)
    rng = np.random.default_rng(seed)
    keep: list[int] = [int(pos) for pos in np.flatnonzero(all_neutral)]
    handled: set[tuple[int, ...]] = set()
    for signs in sorted(cells):
        if signs in handled:
            continue
        mirror = tuple(-v for v in signs)
        handled.update((signs, mirror))
        size = min(len(cells.get(signs, [])), len(cells.get(mirror, [])))
        for key in (signs, mirror):
            rows = cells.get(key, [])
            order = rng.permutation(len(rows))
            keep.extend(rows[int(i)] for i in sorted(order[:size]))
    retained = ordered.iloc[sorted(keep)].reset_index(drop=True)
    record: dict[str, Any] = {
        "balanced_votes": list(names),
        "candidates": int(len(ordered)),
        "all_neutral_retained": int(all_neutral.sum()),
        "partially_neutral_excluded": int(partial.sum()),
        "retained_pairs": int(len(retained)),
        "seed": int(seed),
    }
    if len(retained) == 0:
        record["status"] = "no_pairs_after_balance"
        record["certified_direction_accuracy"] = None
        return retained, record
    cert: dict[str, float] = {}
    for name in names:
        acc = float(retained[f"{name}_direction"].to_numpy(dtype=np.float64).mean())
        if acc != md.NULL_ACCURACY:
            raise ValueError(f"Tier failed exact balance on {name}: acc={acc!r}")
        cert[name] = acc
    record["status"] = "balanced"
    record["certified_direction_accuracy"] = cert
    return retained, record


def _pair_pooled_draws(
    resampler: md.ClusterResampler, blocks: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    sums, counts = blocks
    drawn_counts = counts[resampler.index].sum(axis=1)
    drawn_sums = sums[resampler.index].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(drawn_counts > 0, drawn_sums / drawn_counts, np.nan)


def _pair_family_bands(
    columns: Mapping[str, Sequence[float]],
    clusters: Sequence[Any],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = PAIR_BOOTSTRAP_SEED,
    verdict_from: str = "simultaneous",
    resampler: md.ClusterResampler | None = None,
) -> dict[str, Any]:
    """Pooled pair fraction with native simultaneous max-deviation band over ``columns``."""
    labels = list(columns)
    labelled = np.asarray([int(v) for v in clusters])
    units = sorted(set(labelled.tolist()))
    floor = bootstrap_unit_floor(len(units))
    vectors = {label: np.asarray(columns[label], dtype=np.float64) for label in labels}

    def _summary(label: str) -> dict[str, Any]:
        vec = vectors[label]
        return {
            "pooled": None if vec.size == 0 else float(vec.mean()),
            "cluster_mean": None
            if vec.size == 0
            else float(np.mean([vec[labelled == u].mean() for u in units])),
            "n_ties": int(np.count_nonzero(vec == md.NULL_ACCURACY)),
        }

    rec: dict[str, Any] = {
        "family_size": len(labels),
        "n_pairs": int(labelled.size),
        "n_clusters": len(units),
        "verdict_from": verdict_from,
        "null": md.NULL_ACCURACY,
        "margin": md.PRACTICAL_MARGIN,
        **floor,
        "status": "complete",
        "band": None,
        "columns": {},
    }
    if floor["degenerate"]:
        rec["status"] = "not_estimable"
        rec["columns"] = {
            lbl: _summary(lbl) | {"pointwise": None, "simultaneous": None, "verdict": md.level_verdict(None)}
            for lbl in labels
        }
        return rec

    if resampler is None:
        resampler = md.ClusterResampler(units, draws=draws, seed=seed)
    blocks = {lbl: resampler.blocks(vectors[lbl], labelled) for lbl in labels}
    pointwise = {lbl: resampler.interval(blocks[lbl], confidence=bidirectional.CONFIDENCE) for lbl in labels}
    if any(pointwise[lbl]["interval"] is None for lbl in labels):
        rec["status"] = "not_estimable"
        rec["columns"] = {
            lbl: _summary(lbl) | {"pointwise": None, "simultaneous": None, "verdict": md.level_verdict(None)}
            for lbl in labels
        }
        return rec

    points = np.asarray([pointwise[lbl]["point"] for lbl in labels], dtype=np.float64)
    deviation = np.column_stack([_pair_pooled_draws(resampler, blocks[lbl]) for lbl in labels]) - points
    finite = np.isfinite(deviation).all(axis=1)
    critical = float(np.quantile(np.max(np.abs(deviation[finite]), axis=1), bidirectional.CONFIDENCE))
    rec["band"] = {
        "method": bidirectional.NATIVE_METHOD,
        "draws": int(resampler.draws),
        "finite_draws": int(finite.sum()),
        "seed": int(resampler.seed),
        "confidence": float(bidirectional.CONFIDENCE),
        "groups": len(units),
        "family_size": len(labels),
        "critical_value": critical,
    }
    verdict_tally = {"supported": 0, "bounded_negative": 0, "unresolved": 0, "withheld": 0}
    for pos, lbl in enumerate(labels):
        pt = float(points[pos])
        sim = [pt - critical, pt + critical]
        pw = [float(v) for v in pointwise[lbl]["interval"]]
        chosen = sim if verdict_from == "simultaneous" else pw
        vd = md.level_verdict(chosen)
        verdict_tally[vd["verdict"]] += 1
        rec["columns"][lbl] = _summary(lbl) | {
            "pointwise": pw,
            "simultaneous": sim,
            "verdict": vd,
        }
    rec["verdict_counts"] = verdict_tally
    return rec


def _collect_expanded_tmatch_37(root: Path) -> dict[str, Any]:
    """Assemble deduplicated 37-arm expanded ``T_match`` tier summary (S2_expanded + GP R1)."""
    # Map expanded_variants manifest names -> canonical unified arm_ids
    ev_name_map = {
        "ProtGPT2": "protgpt2",
        "RITA_xl": "rita-xl",
        "esm2_t33_650M_UR50D": "esm2-650m",
        "progen2-small": "progen2-small",
        "progen2-medium": "progen2-medium",
        "progen2-large": "progen2-large",
        "progen2-xlarge": "progen2-xlarge",
        "progen3-112m": "progen3-112m",
        "progen3-3b": "progen3-3b",
    }
    entries: dict[str, dict[str, Any]] = {}

    ev_path = root / EXPANDED_MD_ROWS
    if ev_path.is_file():
        ev_doc = json.loads(ev_path.read_text(encoding="utf-8"))
        for raw_name, arm_doc in ev_doc.get("arms", {}).items():
            arm_id = ev_name_map.get(raw_name)
            if arm_id is None:
                continue
            tiers = {}
            for tname in ("T_open", "T_match", "T_balanced", "T_double_balanced"):
                if tname in arm_doc.get("tiers", {}):
                    ev_t = arm_doc["tiers"][tname]["evaluation"]
                    tiers[tname] = {
                        "accuracy": float(ev_t["accuracy"]),
                        "interval_95": ev_t.get("interval"),
                        "n_pairs": int(ev_t["n_pairs"]),
                        "n_clusters": int(ev_t["n_units"]),
                        "verdict": ev_t.get("verdict"),
                    }
            entries[arm_id] = {
                "source": str(EXPANDED_MD_ROWS),
                "stratum": "S2_expanded_evaluation_clusters",
                "tiers": tiers,
            }

    # Add 28 GP arms + esm3-1.4b (keep esm2-650m's GP full-cohort entry alongside S2_expanded note)
    for arm_id in EXPANDED_29_SINGLE_ARMS:
        if arm_id == "esm3-1.4b":
            art = resolve_gp_r1_json(arm_id, root=root)
            if art.sibling_json and art.sibling_json.is_file():
                doc = json.loads(art.sibling_json.read_text(encoding="utf-8"))
                md_tiers = (
                    doc.get("r1_ladder_and_conditionals", {})
                    .get("matched_discrimination", {})
                    .get("tiers", {})
                )
                tiers = {}
                for tname in ("T_open", "T_match"):
                    if tname in md_tiers:
                        acc = md_tiers[tname]["accuracy"]
                        tiers[tname] = {
                            "accuracy": float(acc["mean"]),
                            "interval_95": [float(acc["ci_low"]), float(acc["ci_high"])],
                            "n_pairs": int(md_tiers[tname]["n_pairs"]),
                            "n_clusters": int(acc["n_clusters"]),
                        }
                entries[arm_id] = {
                    "source": str(art.sibling_json.relative_to(root)),
                    "stratum": "S2_expanded_all_clusters",
                    "tiers": tiers,
                }
        else:
            art = resolve_gp_r1_json(arm_id, root=root)
            doc = json.loads(art.path.read_text(encoding="utf-8"))
            md_tiers = (doc.get("matched_discrimination") or {}).get("tiers", {})
            tiers = {}
            for tname in ("T_open", "T_match"):
                if tname in md_tiers:
                    acc = md_tiers[tname]["accuracy"]
                    tiers[tname] = {
                        "accuracy": float(acc["mean"]),
                        "interval_95": [float(acc["ci_low"]), float(acc["ci_high"])],
                        "n_pairs": int(md_tiers[tname]["n_pairs"]),
                        "n_clusters": int(acc["n_clusters"]),
                    }
            rec = {
                "source": str(art.path.relative_to(root)),
                "stratum": "S2_expanded_all_clusters",
                "tiers": tiers,
            }
            if arm_id in entries:
                # Preserve S2_expanded evaluation-stratum record for esm2-650m as companion field
                rec["s2_expanded_eval_anchor"] = entries[arm_id]
            entries[arm_id] = rec

    ordered_ids = [a for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS if a in entries]
    return {
        "n_arms": len(ordered_ids),
        "arms": {a: entries[a] for a in ordered_ids},
        "note": (
            "Deduplicated expanded matched-discrimination T_match summary across the 8 "
            "S2_expanded autoregressive arms + 29 general-PLM arms (esm2-650m appears in both "
            "and carries its S2_expanded evaluation anchor under s2_expanded_eval_anchor). "
            "S2_expanded evaluates evaluation clusters only, whereas general-PLM R1 evaluates all "
            "retained S2_expanded clusters; both strata are explicitly tagged."
        ),
    }


def compute_adversarial_pairs_62(
    ctx: R1UnifiedContext,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = PAIR_BOOTSTRAP_SEED,
    tolerances: Sequence[float] = PAIR_TOLERANCES,
    verify_ref: bool = True,
) -> dict[str, Any]:
    """Unified 62-model explicit pair-level bidirectional ablation on the 24,832-row support."""
    samples_df = pd.read_csv(ctx.root / MUTATION_SAMPLES_CSV)
    support = md.common_support(samples_df)

    # Attach official Profile, Potts, GEMME onto support by sample_id
    sid_to_idx = {str(sid): idx for idx, sid in enumerate(ctx.sample_ids)}
    sup_sids = support["sample_id"].astype(str).tolist()
    if set(sup_sids) != set(sid_to_idx):
        raise ValueError("mutation-samples.csv.gz common_support disagrees with frozen R1 sample_ids")
    order = np.asarray([sid_to_idx[sid] for sid in sup_sids], dtype=np.int64)
    support = support.copy()
    support["profile"] = support["profile_score"].to_numpy(dtype=np.float64)
    support["potts"] = ctx.raw_baselines["potts"][order]
    support["gemme"] = ctx.raw_baselines["gemme"][order]

    # Refuse the exact 7 degenerate GEMME assays whole-assay before building rank scale
    refused_mask = support["unit_id"].astype(str).isin(GEMME_REFUSED_ASSAYS)
    if int(refused_mask.sum()) != 896:
        raise ValueError(f"Expected 896 refused GEMME rows, found {int(refused_mask.sum())}")
    support = support.loc[~refused_mask].reset_index(drop=True)
    if len(support) != 24832:
        raise ValueError(f"Expected 24,832 retained rows, got {len(support)}")

    for stat in PAIR_STATISTICS:
        spread = support.groupby("unit_id")[stat].transform(lambda b: b.std(ddof=0))
        if (spread.to_numpy(dtype=np.float64) == 0.0).any():
            raise ValueError(f"Degenerate {stat} block survived refusal")
        support[f"rank_{stat}"] = support.groupby("unit_id")[stat].transform(
            lambda b: md.standardized_rank(b.to_numpy())
        )

    stat_ranks = {
        stat: dict(
            zip(support["sample_id"].astype(str), support[f"rank_{stat}"].to_numpy(dtype=np.float64), strict=True)
        )
        for stat in PAIR_STATISTICS
    }

    # Within-assay standardized ranks for all 62 models on the 24,832-row support
    kept_order = np.asarray(
        [sid_to_idx[sid] for sid in support["sample_id"].astype(str)], dtype=np.int64
    )
    sup_aids = support["unit_id"].astype(str).to_numpy()
    sup_sids_kept = support["sample_id"].astype(str).tolist()
    model_ranks: dict[str, dict[str, float]] = {}
    for arm_id in UNIFIED_62_SINGLE_SEQUENCE_ARMS:
        vals = ctx.raw_model_scores[arm_id][kept_order]
        ranked = overlap.rerank(vals, sup_aids)
        model_ranks[arm_id] = dict(zip(sup_sids_kept, ranked.tolist(), strict=True))

    # Build candidate pairs (1,596 gap-satisfying pairs across 135 clusters)
    singles = md.single_substitutions(support)
    all_candidates = md.candidate_pairs(singles)
    gap_candidates = all_candidates.loc[all_candidates["phenotype_gap"] >= md.PHENOTYPE_GAP].copy()
    gap_candidates = _pair_orient(gap_candidates)
    for stat in PAIR_STATISTICS:
        gap_candidates = _pair_attach_covariate(gap_candidates, stat, stat_ranks[stat])

    if len(gap_candidates) != 1596:
        raise ValueError(f"Expected 1,596 gap-satisfying candidate pairs, got {len(gap_candidates)}")

    # Forward tiers (shared across all 62 models)
    open_tier = md.greedy_disjoint(gap_candidates, seed=md.SPLIT_SEED)
    fwd_tier_frames: dict[str, tuple[str, float | None, pd.DataFrame, dict[str, Any]]] = {}
    fwd_tier_frames["T_open"] = (
        "T_open",
        None,
        open_tier,
        {"candidates_offered": len(gap_candidates), "retained_after_disjointness": len(open_tier)},
    )
    for tname, votes in (("T_bal_P", ("profile",)), ("T_bal_PEG", PAIR_STATISTICS)):
        frame, rec = _pair_balance_votes(open_tier, votes, seed=PAIR_BALANCE_SEED)
        fwd_tier_frames[tname] = (tname, None, frame, rec)
    for tol in tolerances:
        for tname, names in (("T_tol_P", ("profile",)), ("T_tol_PEG", PAIR_STATISTICS)):
            sub = _pair_tolerance_matched(gap_candidates, names, tol)
            frame = md.greedy_disjoint(sub, seed=md.SPLIT_SEED) if len(sub) else sub
            lbl = _pair_tier_label(tname, tol)
            fwd_tier_frames[lbl] = (
                tname,
                float(tol),
                frame,
                {
                    "matched_quantities": list(names),
                    "candidates_matched": len(sub),
                    "retained_after_disjointness": len(frame),
                },
            )
        peg_frame = fwd_tier_frames[_pair_tier_label("T_tol_PEG", tol)][2]
        conj_frame, conj_rec = _pair_balance_votes(
            peg_frame, PAIR_STATISTICS, seed=PAIR_BALANCE_SEED
        )
        lbl = _pair_tier_label("T_tol_bal_PEG", tol)
        fwd_tier_frames[lbl] = ("T_tol_bal_PEG", float(tol), conj_frame, conj_rec)

    forward_out: dict[str, Any] = {}
    for lbl, (base_name, tol, frame, constr) in fwd_tier_frames.items():
        clusters = frame["group_id"].tolist() if len(frame) else []
        units = sorted({int(c) for c in clusters})
        resampler = (
            md.ClusterResampler(units, draws=draws, seed=seed)
            if not bootstrap_unit_floor(len(units))["degenerate"]
            else None
        )
        check_cols = {
            stat: md.direction_values(frame, stat_ranks[stat]) if len(frame) else np.array([])
            for stat in PAIR_STATISTICS
        }
        arm_cols = {
            arm_id: md.direction_values(frame, model_ranks[arm_id]) if len(frame) else np.array([])
            for arm_id in UNIFIED_62_SINGLE_SEQUENCE_ARMS
        }
        forward_out[lbl] = {
            "tier": lbl,
            "rule": PAIR_TIER_RULES[base_name],
            "tolerance": tol,
            "counts": md.tier_counts(frame) if len(frame) else {"pairs": 0, "variants": 0, "assays": 0, "clusters": 0},
            "construction": constr,
            "construction_check": _pair_family_bands(
                check_cols, clusters, draws=draws, seed=seed, verdict_from="pointwise", resampler=resampler
            ),
            "family_62_arms": _pair_family_bands(
                arm_cols, clusters, draws=draws, seed=seed, verdict_from="simultaneous", resampler=resampler
            ),
        }

    # Reverse tiers (arm-specific) across all 62 arms: T_bal_M, T_tol_M(tol), T_tol_bal_M(tol)
    reverse_out: dict[str, Any] = {}
    for arm_id in UNIFIED_62_SINGLE_SEQUENCE_ARMS:
        ranks_m = model_ranks[arm_id]
        cand_m = _pair_attach_covariate(gap_candidates, "model", ranks_m)
        open_m = _pair_attach_covariate(open_tier, "model", ranks_m)
        arm_tiers: dict[str, Any] = {}

        # Primary + full sensitivity reverse tiers
        rev_specs: list[tuple[str, float | None, pd.DataFrame, dict[str, Any]]] = []
        bal_m, bal_rec = _pair_balance_votes(open_m, ("model",), seed=PAIR_BALANCE_SEED)
        rev_specs.append(("T_bal_M", None, bal_m, bal_rec))
        for tol in tolerances:
            sub = _pair_tolerance_matched(cand_m, ("model",), tol)
            tol_frame = md.greedy_disjoint(sub, seed=md.SPLIT_SEED) if len(sub) else sub
            rev_specs.append(
                (
                    "T_tol_M",
                    float(tol),
                    tol_frame,
                    {"candidates_matched": len(sub), "retained_after_disjointness": len(tol_frame)},
                )
            )
            conj_frame, conj_rec = _pair_balance_votes(
                tol_frame, ("model",), seed=PAIR_BALANCE_SEED
            )
            rev_specs.append(("T_tol_bal_M", float(tol), conj_frame, conj_rec))

        for base_name, tol, frame, constr in rev_specs:
            lbl = _pair_tier_label(base_name, tol)
            clusters = frame["group_id"].tolist() if len(frame) else []
            units = sorted({int(c) for c in clusters})
            resampler = (
                md.ClusterResampler(units, draws=draws, seed=seed)
                if not bootstrap_unit_floor(len(units))["degenerate"]
                else None
            )
            stat_cols = {
                stat: md.direction_values(frame, stat_ranks[stat]) if len(frame) else np.array([])
                for stat in PAIR_STATISTICS
            }
            model_col = {
                "model": md.direction_values(frame, ranks_m) if len(frame) else np.array([])
            }
            arm_tiers[lbl] = {
                "tier": lbl,
                "rule": PAIR_TIER_RULES[base_name],
                "tolerance": tol,
                "counts": md.tier_counts(frame) if len(frame) else {"pairs": 0, "variants": 0, "assays": 0, "clusters": 0},
                "construction": constr,
                "model_construction_check": _pair_family_bands(
                    model_col, clusters, draws=draws, seed=seed, verdict_from="pointwise", resampler=resampler
                ),
                "family_3_statistics": _pair_family_bands(
                    stat_cols, clusters, draws=draws, seed=seed, verdict_from="simultaneous", resampler=resampler
                ),
            }
        reverse_out[arm_id] = arm_tiers

    # Exact reproduction verification against frozen 33-arm explicit_pair_ablation_20261114
    ref_check: dict[str, Any] = {"status": "skipped"}
    if verify_ref and (ctx.root / EXPLICIT_PAIR_REF_DIR / "forward.json").is_file():
        ref_fwd = json.loads((ctx.root / EXPLICIT_PAIR_REF_DIR / "forward.json").read_text())
        ref_rev = json.loads((ctx.root / EXPLICIT_PAIR_REF_DIR / "reverse.json").read_text())
        max_fwd_diff = 0.0
        for tlbl in ("T_open", "T_bal_P", "T_tol_P(0.2)", "T_bal_PEG", "T_tol_PEG(0.2)", "T_tol_bal_PEG(0.2)"):
            for arm_id in PREREGISTERED_33_ARMS:
                ref_pt = ref_fwd["tiers"][tlbl]["family"]["columns"][arm_id]["pooled"]
                cur_pt = forward_out[tlbl]["family_62_arms"]["columns"][arm_id]["pooled"]
                max_fwd_diff = max(max_fwd_diff, abs(float(cur_pt) - float(ref_pt)))
        max_rev_diff = 0.0
        for arm_id in PREREGISTERED_33_ARMS:
            for stat in PAIR_STATISTICS:
                ref_pt = ref_rev["arms"][arm_id]["T_bal_M"]["family"]["columns"][stat]["pooled"]
                cur_pt = reverse_out[arm_id]["T_bal_M"]["family_3_statistics"]["columns"][stat]["pooled"]
                max_rev_diff = max(max_rev_diff, abs(float(cur_pt) - float(ref_pt)))
        if max(max_fwd_diff, max_rev_diff) > 1e-12:
            raise ValueError(
                f"Adversarial pair reproduction mismatch vs 20261114 reference: "
                f"fwd={max_fwd_diff:g}, rev={max_rev_diff:g}"
            )
        ref_check = {
            "status": "passed",
            "reference_dir": str(EXPLICIT_PAIR_REF_DIR),
            "max_forward_pooled_diff_33_arms": max_fwd_diff,
            "max_reverse_T_bal_M_pooled_diff_33_arms": max_rev_diff,
        }

    return {
        "schema": "r1_adversarial_pairs_62_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "support": {
            "rows_offered": 25728,
            "rows_retained": 24832,
            "assays_retained": 194,
            "clusters_retained": 156,
            "gemme_refused_assays": sorted(GEMME_REFUSED_ASSAYS),
            "single_substitutions": int(len(singles)),
            "pair_rule_candidates": int(len(all_candidates)),
            "phenotype_gap_candidates": int(len(gap_candidates)),
            "phenotype_gap_clusters": int(gap_candidates["group_id"].nunique()),
        },
        "constants": {
            "phenotype_gap": md.PHENOTYPE_GAP,
            "primary_tolerance": PAIR_PRIMARY_TOLERANCE,
            "tolerance_ladder": list(tolerances),
            "random_pair_separation": PAIR_RANDOM_SEPARATION,
            "greedy_seed": md.SPLIT_SEED,
            "balance_seed": PAIR_BALANCE_SEED,
            "bootstrap_draws": draws,
            "bootstrap_seed": seed,
            "null_accuracy": md.NULL_ACCURACY,
            "practical_margin": md.PRACTICAL_MARGIN,
        },
        "reproduction_vs_33_arm_reference": ref_check,
        "n_models": len(UNIFIED_62_SINGLE_SEQUENCE_ARMS),
        "models": list(UNIFIED_62_SINGLE_SEQUENCE_ARMS),
        "arm_metadata": {a: _arm_meta_dict(a) for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS},
        "forward_tiers": forward_out,
        "reverse_tiers": reverse_out,
        "expanded_t_match_37_arms": _collect_expanded_tmatch_37(ctx.root),
    }


# --------------------------------------------------------------------------- #
# End-to-End Orchestrator
# --------------------------------------------------------------------------- #


def run_r1_unified_yardstick(
    *,
    root: Path | str = REPO_ROOT,
    out_dir: Path | str | None = None,
    workers: int = 4,
    threads: int = 2,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Run the full unified 62-model R1 ProteinGym yardstick and write all 3 JSONs + .sha256."""
    base = Path(root).resolve()
    target_dir = Path(out_dir).resolve() if out_dir is not None else (base / UNIFIED_OUTPUT_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = target_dir / "cells/r1"

    print("[R1-unified] Loading 25,728-row R1 support + all 62 arms...", flush=True)
    ctx = load_r1_unified_context(base, out_dir=target_dir)

    print("[R1-unified] Running reproduction gate on cached pre-registered cells...", flush=True)
    repro = verify_reproduction_gate(ctx)
    print(f"[R1-unified] Reproduction gate PASSED (max_abs_diff={repro['max_abs_diff']:.3e})", flush=True)

    print("[R1-unified] Computing Layer 1 zero-shot (62 arms + Profile/Potts/GEMME + 5 categories)...", flush=True)
    layer1 = compute_layer1_zero_shot(ctx, draws=draws, seed=BOOTSTRAP_SEED)

    print(
        f"[R1-unified] Fitting/loading R1 nested-CV cells (workers={workers}, threads={threads}, "
        f"r_bridge_arms={len(ctx.r_bridge_arms)})...",
        flush=True,
    )
    fit_all_r1_cells(ctx, cache_dir, workers=workers, threads=threads, include_r_bridge=True)

    print("[R1-unified] Computing per-assay metrics across all 62 arms x 3 seeds...", flush=True)
    metrics_by_cell, assay_family, _ = _load_per_assay_metrics_table(
        ctx, cache_dir, include_r_bridge=True
    )

    print("[R1-unified] Computing Layer 2 core bidirectional increments + 5 functional categories...", flush=True)
    layer2_core = compute_layer2_core_bidirectional(
        ctx, metrics_by_cell, assay_family, draws=draws, seed=BOOTSTRAP_SEED
    )

    print("[R1-unified] Computing Layer 2 +R representation bridge (28 expanded + 33 cached)...", flush=True)
    layer2_r = compute_layer2_representation_bridge(
        ctx, metrics_by_cell, assay_family, draws=draws, seed=BOOTSTRAP_SEED
    )

    main_payload = {
        "schema": "r1_proteingym_unified_62_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "support_sha256": ctx.support_sha256,
        "provenance": ctx.provenance,
        "gp_score_provenance": ctx.gp_score_provenance,
        "reproduction_gate": repro,
        "n_models": len(UNIFIED_62_SINGLE_SEQUENCE_ARMS),
        "models": list(UNIFIED_62_SINGLE_SEQUENCE_ARMS),
        "arm_metadata": {a: _arm_meta_dict(a) for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS},
        "layer1_zero_shot": layer1,
        "layer2_core_bidirectional": layer2_core,
        "layer2_representation_bridge": layer2_r,
    }
    p_main = target_dir / "r1_proteingym_unified_62.json"
    atomic_write_json(p_main, main_payload)
    sha_main = write_sha256_sidecar(p_main)
    print(f"[R1-unified] Wrote {p_main} (sha256={sha_main[:12]}...)", flush=True)

    print("[R1-unified] Computing Supplementary 6-rung C0 ladder (744-contrast family)...", flush=True)
    c0_payload = compute_supplementary_c0_ladder(
        ctx, metrics_by_cell, assay_family, draws=draws, seed=BOOTSTRAP_SEED
    )
    p_c0 = target_dir / "r1_supplementary_c0_ladder_62.json"
    atomic_write_json(p_c0, c0_payload)
    sha_c0 = write_sha256_sidecar(p_c0)
    print(f"[R1-unified] Wrote {p_c0} (sha256={sha_c0[:12]}...)", flush=True)

    print("[R1-unified] Computing Unified 62-model Adversarial Pairs (forward + reverse)...", flush=True)
    pairs_payload = compute_adversarial_pairs_62(ctx, draws=draws, seed=PAIR_BOOTSTRAP_SEED)
    p_pairs = target_dir / "r1_adversarial_pairs_62.json"
    atomic_write_json(p_pairs, pairs_payload)
    sha_pairs = write_sha256_sidecar(p_pairs)
    print(f"[R1-unified] Wrote {p_pairs} (sha256={sha_pairs[:12]}...)", flush=True)

    return {
        "outputs": {
            "r1_proteingym_unified_62.json": {"path": str(p_main), "sha256": sha_main},
            "r1_supplementary_c0_ladder_62.json": {"path": str(p_c0), "sha256": sha_c0},
            "r1_adversarial_pairs_62.json": {"path": str(p_pairs), "sha256": sha_pairs},
        },
        "reproduction_gate": repro,
        "pairs_reproduction": pairs_payload["reproduction_vs_33_arm_reference"],
    }
