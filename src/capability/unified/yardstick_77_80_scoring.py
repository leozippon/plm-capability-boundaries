"""Unified 77/80-model scoring benchmark expansion adding DPLM-2 (4 checkpoints,
seq-only + struct-conditioned) and ProteinMPNN (11 checkpoints: vanilla, soluble, CA-only)
across ALL scoring benchmarks in Results 1-5 without modifying any sealed 62/65-arm file.

Expanded outputs (all written atomically with ``.sha256`` sidecars and infra-leak guards):

1. ``r1_proteingym_unified_77.json`` (+ ``r1_supplementary_c0_ladder_77.json``,
   ``r1_adversarial_pairs_77.json``): 77 primary arms + 4 ``dplm2-*|struct`` companions.
2. ``family_memory_unified_77.json``: ``family_conditionals`` + ``family_distribution``
   extending the 62-arm sequence-only panel to 66 (+ optional struct-conditioned arms if
   per-homolog backbones are available; honest accounting otherwise).
3. ``r2_biophysical_unified_77.json``: MegaScale (79g CC / 101g) & Human Domainome
   (94g CC / 96g) across 77 primary arms + 4 ``dplm2-*|struct`` companions and 17 scaling/
   noise-ablation ladders (13 original + DPLM-2 + ProteinMPNN-Vanilla/Soluble/CA).
4. ``r3_epistasis_and_contacts_unified_77.json`` & ``contacts_unified_77.json`` (+
   ``contact_discrimination_unified_77{,_deep,_shallow}.json`` and
   ``contact_bidirectional_unified_77{,_deep,_shallow}.json``): 77 primary arms + 4
   ``dplm2-*|struct`` companions with explicit ``conditioning_regime`` stratification
   (``sequence_only`` vs ``backbone_3d_conditioned_oracle``).
5. ``r2_r3_middle_vs_last_representation_77.json``: paired ``R_mid`` (0:512) vs ``R_last``
   (512:1024) vs ``R_joint`` (0:1024) across MegaScale, Domainome, and R3 epistasis.
6. ``r4_scoring_unified_80.json``: Result 5 Main Table (800-unit ``context_homologue``
   cohort, 710 clusters) over all 80 models + 4 ``dplm2-*|struct`` companions.
7. ``r5_dms_context_ranking_unified_80.json``: Result 5 Downstream 1 DMS homolog-context
   ranking (``rescue_30_80`` + ``e09_five_band``) over all 80 models + companions.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from src.capability.core.io import sha256_file
from src.capability.extensions import bidirectional as bidir
from src.capability.extensions import statistical_ladder as sl
from src.capability.general_plm.eval_utils import atomic_write_json, safe_spearman
from src.capability.general_plm.r3_cycle_eval import (
    INTERACTION_RUNGS,
    load_pairwise_cycle_context,
)
from src.capability.interactions import pairwise_epistasis as pw_mod
from src.capability.memory_theory.capacity_scaling import load_msa_support
from src.capability.memory_theory.io_utils import (
    LN2,
    cluster_bootstrap_scaling_slope,
    load_json_file,
    write_json_with_sha256,
)
from src.capability.readouts.readout_analysis import row_weights
from src.capability.stability import stability_gate as gate
from src.capability.unified import contacts_yardstick as contacts_mod
from src.capability.unified import memory_and_generation_yardstick as mem_mod
from src.capability.unified import r1_yardstick as r1_mod
from src.capability.unified import r2_experiment_2_1 as r2_mod
from src.capability.unified import r2_r3_mid_vs_last_yardstick as mid_mod
from src.capability.unified import r3_yardstick as r3_mod
from src.capability.unified import r4_scoring_yardstick as r4_mod
from src.capability.unified import r5_dms_context_yardstick as r5_dms_mod
from src.capability.unified import roster as roster_mod
from src.capability.unified.roster import (
    ARM_METADATA as BASE_ARM_METADATA_65,
    PREREGISTERED_33_ARMS,
    REPO_ROOT,
    UNIFIED_3_MSA_CONTEXT_ARMS,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_65_ARMS,
    UNIFIED_OUTPUT_DIR,
)

# --------------------------------------------------------------------------- #
# Expanded 77/80 Roster Metadata & Conditioning-Regime Taxonomy
# --------------------------------------------------------------------------- #

from src.capability.unified.master_summary import FORBIDDEN_INFRA_RE

DEFAULT_NEW_SCORING_DIR = Path(UNIFIED_OUTPUT_DIR) / "dplm2_mpnn_scoring"
DEFAULT_NEW_CELL_CACHE_DIR = Path(UNIFIED_OUTPUT_DIR) / "cells" / "dplm2_mpnn_77"

DPLM2_4_ARMS: tuple[str, ...] = getattr(
    roster_mod,
    "DPLM2_4_ARMS",
    ("dplm2-150m", "dplm2-650m", "dplm2-bit-650m", "dplm2-3b"),
)
DPLM2_4_STRUCT_ARMS: tuple[str, ...] = tuple(f"{a}|struct" for a in DPLM2_4_ARMS)
PROTEINMPNN_11_ARMS: tuple[str, ...] = getattr(
    roster_mod,
    "PROTEINMPNN_11_ARMS",
    (
        "proteinmpnn-v-002",
        "proteinmpnn-v-010",
        "proteinmpnn-v-020",
        "proteinmpnn-v-030",
        "proteinmpnn-s-002",
        "proteinmpnn-s-010",
        "proteinmpnn-s-020",
        "proteinmpnn-s-030",
        "proteinmpnn-ca-002",
        "proteinmpnn-ca-010",
        "proteinmpnn-ca-020",
    ),
)
DIMA_4_ARMS: tuple[str, ...] = getattr(
    roster_mod,
    "DIMA_4_ARMS",
    ("dima-esm2-8m", "dima-esm2-3b", "dima-saprot-35m", "dima-cheap-1024"),
)

NEW_15_PRIMARY_ARMS: tuple[str, ...] = DPLM2_4_ARMS + PROTEINMPNN_11_ARMS
NEW_19_PRIMARY_ARMS: tuple[str, ...] = DPLM2_4_ARMS + PROTEINMPNN_11_ARMS + DIMA_4_ARMS
NEW_19_SCORED_ENTRIES: tuple[str, ...] = DPLM2_4_ARMS + DPLM2_4_STRUCT_ARMS + PROTEINMPNN_11_ARMS
NEW_23_SCORED_ENTRIES: tuple[str, ...] = (
    DPLM2_4_ARMS + DPLM2_4_STRUCT_ARMS + PROTEINMPNN_11_ARMS + DIMA_4_ARMS
)

UNIFIED_77_SINGLE_TARGET_ARMS: tuple[str, ...] = tuple(
    getattr(
        roster_mod,
        "UNIFIED_77_SINGLE_TARGET_ARMS",
        UNIFIED_62_SINGLE_SEQUENCE_ARMS + NEW_15_PRIMARY_ARMS,
    )
)
UNIFIED_80_ARMS: tuple[str, ...] = tuple(
    getattr(
        roster_mod,
        "UNIFIED_80_ARMS",
        UNIFIED_77_SINGLE_TARGET_ARMS + UNIFIED_3_MSA_CONTEXT_ARMS,
    )
)
UNIFIED_81_SINGLE_TARGET_ARMS: tuple[str, ...] = tuple(
    getattr(
        roster_mod,
        "UNIFIED_81_SINGLE_TARGET_ARMS",
        UNIFIED_77_SINGLE_TARGET_ARMS + DIMA_4_ARMS,
    )
)
UNIFIED_84_ARMS: tuple[str, ...] = tuple(
    getattr(
        roster_mod,
        "UNIFIED_84_ARMS",
        UNIFIED_81_SINGLE_TARGET_ARMS + UNIFIED_3_MSA_CONTEXT_ARMS,
    )
)
UNIFIED_81_SINGLE_TARGET_WITH_DPLM2_STRUCT: tuple[str, ...] = (
    UNIFIED_77_SINGLE_TARGET_ARMS + DPLM2_4_STRUCT_ARMS
)
UNIFIED_85_SINGLE_TARGET_WITH_DPLM2_STRUCT: tuple[str, ...] = (
    UNIFIED_81_SINGLE_TARGET_ARMS + DPLM2_4_STRUCT_ARMS
)
SEQUENCE_ONLY_66_ARMS: tuple[str, ...] = UNIFIED_62_SINGLE_SEQUENCE_ARMS + DPLM2_4_ARMS
SEQUENCE_ONLY_70_ARMS: tuple[str, ...] = UNIFIED_62_SINGLE_SEQUENCE_ARMS + DPLM2_4_ARMS + DIMA_4_ARMS
ORACLE_3D_15_ENTRIES: tuple[str, ...] = DPLM2_4_STRUCT_ARMS + PROTEINMPNN_11_ARMS

assert len(NEW_15_PRIMARY_ARMS) == 15
assert len(NEW_19_PRIMARY_ARMS) == 19
assert len(NEW_23_SCORED_ENTRIES) == 23
assert len(UNIFIED_77_SINGLE_TARGET_ARMS) == 77
assert len(UNIFIED_80_ARMS) == 80
assert len(UNIFIED_81_SINGLE_TARGET_ARMS) == 81
assert len(UNIFIED_84_ARMS) == 84
assert len(SEQUENCE_ONLY_70_ARMS) == 70

# 18 ladders for R2/R3 scaling & noise-ablation divergence (13 original + 4 DPLM-2/MPNN + DiMA)
SCALING_LADDERS_17: dict[str, tuple[str, ...]] = {
    **r2_mod.SCALING_LADDERS,
    "DPLM2-SeqOnly": ("dplm2-150m", "dplm2-650m", "dplm2-3b"),
    "ProteinMPNN-Vanilla": (
        "proteinmpnn-v-002",
        "proteinmpnn-v-010",
        "proteinmpnn-v-020",
        "proteinmpnn-v-030",
    ),
    "ProteinMPNN-Soluble": (
        "proteinmpnn-s-002",
        "proteinmpnn-s-010",
        "proteinmpnn-s-020",
        "proteinmpnn-s-030",
    ),
    "ProteinMPNN-CA": (
        "proteinmpnn-ca-002",
        "proteinmpnn-ca-010",
        "proteinmpnn-ca-020",
    ),
}
SCALING_LADDERS_18: dict[str, tuple[str, ...]] = {
    **SCALING_LADDERS_17,
    "DiMA-ESM2-EncoderScale": ("dima-esm2-8m", "dima-esm2-3b"),
}
COMPANION_STRUCT_LADDERS: dict[str, tuple[str, ...]] = {
    "DPLM2-StructCond": ("dplm2-150m|struct", "dplm2-650m|struct", "dplm2-3b|struct"),
}
assert len(SCALING_LADDERS_17) == 17
assert len(SCALING_LADDERS_18) == 18


@dataclass(frozen=True)
class ExtendedArmMeta:
    """Unified metadata across all 80 arms + 4 ``dplm2-*|struct`` companions."""

    arm_id: str
    base_arm_id: str
    display_name: str
    family: str
    paradigm: str
    modality: str
    architecture: str
    params_m: float
    vocab_type: str
    is_single_sequence: bool
    has_r1_raw_readout_1024: bool
    origin_cohort: str
    conditioning_regime: str  # "sequence_only" | "backbone_3d_conditioned_oracle"
    n_layers_L: int | None = None
    mpnn_variant: str | None = None  # "vanilla" | "soluble" | "ca_only"
    mpnn_noise_angstrom: float | None = None


def arm_slug(arm_id: str) -> str:
    """Filesystem-safe deterministic slug matching runner outputs (e.g. ``dplm2-650m__struct``)."""
    return arm_id.replace("|", "__").replace("=", "_").replace("/", "_")


def _build_extended_metadata() -> Mapping[str, ExtendedArmMeta]:
    table: dict[str, ExtendedArmMeta] = {}
    layer_overrides: dict[str, int] = {
        "dplm2-150m": 30,
        "dplm2-650m": 33,
        "dplm2-bit-650m": 33,
        "dplm2-3b": 36,
        "dima-esm2-8m": 12,
        "dima-esm2-3b": 12,
        "dima-saprot-35m": 12,
        "dima-cheap-1024": 12,
    }
    mpnn_noise_map: dict[str, tuple[str, float]] = {
        "proteinmpnn-v-002": ("vanilla", 0.02),
        "proteinmpnn-v-010": ("vanilla", 0.10),
        "proteinmpnn-v-020": ("vanilla", 0.20),
        "proteinmpnn-v-030": ("vanilla", 0.30),
        "proteinmpnn-s-002": ("soluble", 0.02),
        "proteinmpnn-s-010": ("soluble", 0.10),
        "proteinmpnn-s-020": ("soluble", 0.20),
        "proteinmpnn-s-030": ("soluble", 0.30),
        "proteinmpnn-ca-002": ("ca_only", 0.02),
        "proteinmpnn-ca-010": ("ca_only", 0.10),
        "proteinmpnn-ca-020": ("ca_only", 0.20),
    }

    for arm_id, meta in BASE_ARM_METADATA_65.items():
        is_mpnn = arm_id in mpnn_noise_map
        variant, noise = mpnn_noise_map.get(arm_id, (None, None))
        n_layers = (
            6
            if is_mpnn
            else layer_overrides.get(arm_id, mid_mod.MODEL_LAYERS_62.get(arm_id))
        )
        table[arm_id] = ExtendedArmMeta(
            arm_id=arm_id,
            base_arm_id=arm_id,
            display_name=meta.display_name,
            family=meta.family,
            paradigm=meta.paradigm,
            modality=meta.modality,
            architecture=meta.architecture,
            params_m=float(meta.params_m),
            vocab_type=meta.vocab_type,
            is_single_sequence=bool(meta.is_single_sequence),
            has_r1_raw_readout_1024=bool(meta.has_r1_raw_readout_1024),
            origin_cohort=meta.origin_cohort,
            conditioning_regime="backbone_3d_conditioned_oracle" if is_mpnn else "sequence_only",
            n_layers_L=n_layers,
            mpnn_variant=variant,
            mpnn_noise_angstrom=noise,
        )

    for arm_id in DPLM2_4_ARMS:
        base_m = table[arm_id]
        s_id = f"{arm_id}|struct"
        table[s_id] = ExtendedArmMeta(
            arm_id=s_id,
            base_arm_id=arm_id,
            display_name=f"{base_m.display_name} (+3D Struct)",
            family=base_m.family,
            paradigm=base_m.paradigm,
            modality=base_m.modality,
            architecture=base_m.architecture,
            params_m=base_m.params_m,
            vocab_type=base_m.vocab_type,
            is_single_sequence=True,
            has_r1_raw_readout_1024=True,
            origin_cohort=base_m.origin_cohort,
            conditioning_regime="backbone_3d_conditioned_oracle",
            n_layers_L=base_m.n_layers_L,
        )
    return MappingProxyType(table)


EXTENDED_ARM_METADATA: Mapping[str, ExtendedArmMeta] = _build_extended_metadata()


def extended_arm_meta_dict(arm_id: str) -> dict[str, Any]:
    m = EXTENDED_ARM_METADATA[arm_id]
    return {
        "arm_id": m.arm_id,
        "base_arm_id": m.base_arm_id,
        "display_name": m.display_name,
        "family": m.family,
        "paradigm": m.paradigm,
        "modality": m.modality,
        "architecture": m.architecture,
        "params_m": m.params_m,
        "vocab_type": m.vocab_type,
        "has_r1_raw_readout_1024": m.has_r1_raw_readout_1024,
        "origin_cohort": m.origin_cohort,
        "conditioning_regime": m.conditioning_regime,
        "n_layers_L": m.n_layers_L,
        "mpnn_variant": m.mpnn_variant,
        "mpnn_noise_angstrom": m.mpnn_noise_angstrom,
    }


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def assert_no_forbidden_infra_strings(payload: Any, *, label: str = "payload") -> None:
    """Fail fast if any disposable pod/node/IP identifier leaked into an output artifact."""
    serialized = json.dumps(payload, sort_keys=True, default=bidir.json_default)
    match = FORBIDDEN_INFRA_RE.search(serialized)
    if match:
        raise ValueError(
            f"{label}: forbidden cluster/pod identifier {match.group(0)!r} detected in artifact"
        )


def write_verified_json_with_sha256(path: Path, payload: Mapping[str, Any]) -> str:
    """Guard against forbidden infra leaks, write JSON atomically (NaN-free), emit ``.sha256``, chmod a+r."""
    assert_no_forbidden_infra_strings(payload, label=path.name)
    digest = write_json_with_sha256(path, payload)
    sha_path = path.with_name(f"{path.name}.sha256")
    for p in (path, sha_path):
        try:
            os.chmod(p, 0o644)
        except OSError:
            pass
    return digest


def _rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _check_pending_arms(
    expected: Sequence[str],
    present: Sequence[str],
    *,
    benchmark: str,
    allow_pending: bool,
) -> list[str]:
    present_set = set(present)
    missing = [a for a in expected if a not in present_set]
    if missing and not allow_pending:
        raise FileNotFoundError(
            f"{benchmark}: missing {len(missing)}/{len(expected)} required new arms: {missing}. "
            "Pass --allow-pending to assemble partial results while H200 jobs are running."
        )
    return missing


def _dplm2_struct_minus_seq_paired(
    per_arm_vecs: Mapping[str, np.ndarray],
    *,
    draws: int = r2_mod.BOOTSTRAP_DRAWS,
    seed: int = r2_mod.BOOTSTRAP_SEED,
    sign: float = 1.0,
) -> dict[str, Any]:
    """Paired within-checkpoint contrast ``sign * (dplm2|struct - dplm2|seq_only)``."""
    pairs = [
        (a, f"{a}|struct")
        for a in DPLM2_4_ARMS
        if a in per_arm_vecs and f"{a}|struct" in per_arm_vecs
    ]
    if not pairs:
        return {"available": False, "n_pairs": 0, "pairs": {}}
    labels = [f"{a}__struct_minus_seq" for a, _ in pairs]
    cols = [
        sign * (np.asarray(per_arm_vecs[s], dtype=np.float64) - np.asarray(per_arm_vecs[a], dtype=np.float64))
        for a, s in pairs
    ]
    mat = np.column_stack(cols)
    valid = np.all(np.isfinite(mat), axis=1)
    if not valid.any():
        return {"available": False, "n_pairs": len(pairs), "pairs": {}}
    bands = r2_mod.paired_bands_summary(mat[valid], labels, draws=draws, seed=seed)
    return {
        "available": True,
        "n_pairs": len(pairs),
        "n_groups": int(valid.sum()),
        "sign_convention": "positive_means_struct_conditioning_improves_metric",
        "bands": bands,
    }


def _conditioning_regime_summary(
    arm_order: Sequence[str],
    point_by_arm: Mapping[str, float],
) -> dict[str, Any]:
    """Summarize metric distributions explicitly split by ``conditioning_regime`` and ``paradigm``."""
    by_regime: dict[str, list[tuple[str, float]]] = defaultdict(list)
    by_paradigm: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for arm_id in arm_order:
        val = point_by_arm.get(arm_id)
        if val is None or not np.isfinite(val):
            continue
        meta = EXTENDED_ARM_METADATA[arm_id]
        by_regime[meta.conditioning_regime].append((arm_id, float(val)))
        by_paradigm[meta.paradigm].append((arm_id, float(val)))

    def _agg(rows: list[tuple[str, float]]) -> dict[str, Any]:
        vals = np.asarray([v for _, v in rows], dtype=np.float64)
        best_arm, best_val = max(rows, key=lambda x: x[1])
        worst_arm, worst_val = min(rows, key=lambda x: x[1])
        return {
            "n_arms": len(rows),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "min": float(worst_val),
            "min_arm": worst_arm,
            "max": float(best_val),
            "max_arm": best_arm,
            "arms": [a for a, _ in rows],
        }

    return {
        "by_conditioning_regime": {k: _agg(v) for k, v in sorted(by_regime.items())},
        "by_paradigm": {k: _agg(v) for k, v in sorted(by_paradigm.items())},
    }


# ============================================================================
# 1. R1 ProteinGym Unified 77 (+4 DPLM-2|struct) Assembler
# ============================================================================


def _discover_new_r1_artifacts(
    scoring_dir: Path,
    *,
    candidates: Sequence[str] = NEW_23_SCORED_ENTRIES,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, str | None]]:
    r1_dir = scoring_dir / "r1"
    score_paths: dict[str, Path] = {}
    readout_paths: dict[str, Path] = {}
    dtypes: dict[str, str | None] = {}
    if not r1_dir.is_dir():
        return score_paths, readout_paths, dtypes
    for arm_id in candidates:
        slug = arm_slug(arm_id)
        sp = r1_dir / f"r1_ladder_scores_{slug}.npz"
        if sp.is_file():
            score_paths[arm_id] = sp
            rp = r1_dir / f"r1_readout_blocks_{slug}.npz"
            if rp.is_file():
                readout_paths[arm_id] = rp
            sum_p = r1_dir / f"r1_summary_{slug}.json"
            if sum_p.is_file():
                try:
                    sdoc = load_json_file(sum_p)
                    dtypes[arm_id] = str((sdoc.get("provenance") or {}).get("dtype_realised", ""))
                except Exception:
                    dtypes[arm_id] = None
            else:
                dtypes[arm_id] = None
    return score_paths, readout_paths, dtypes


def _load_r1_score_vector_npz(arm_id: str, path: Path, sample_ids: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as d:
        sids = [str(s) for s in d["sample_ids"].tolist()]
        vals = np.asarray(d["scores"], dtype=np.float64)
    if len(sids) != len(vals):
        raise ValueError(f"{arm_id}: sample_ids/scores length mismatch in {path}")
    lookup = dict(zip(sids, vals.tolist()))
    out = np.empty(len(sample_ids), dtype=np.float64)
    missing = 0
    for idx, sid in enumerate(sample_ids.tolist()):
        v = lookup.get(str(sid))
        if v is None or not np.isfinite(v):
            missing += 1
            out[idx] = np.nan
        else:
            out[idx] = float(v)
    if missing:
        raise ValueError(f"{arm_id}: {missing}/{len(sample_ids)} missing/non-finite R1 scores in {path}")
    return out


def _fit_new_r1_cells(
    ctx: r1_mod.R1UnifiedContext,
    new_arms: Sequence[str],
    new_raw_scores: Mapping[str, np.ndarray],
    new_readout_paths: Mapping[str, Path],
    new_cache_dir: Path,
    *,
    workers: int = 4,
    threads: int = 2,
    include_r_bridge: bool = True,
) -> None:
    """Fit ONLY the new arms' R1 nested-CV cells into an isolated cache directory."""
    if not new_arms:
        return
    new_cache_dir.mkdir(parents=True, exist_ok=True)
    worker_state = {
        "threads": threads,
        "cache_dir": str(new_cache_dir),
        "prepared_shared": ctx.prepared_shared,
        "raw_scores": dict(new_raw_scores),
        "sample_ids": ctx.sample_ids,
        "measured": ctx.measured,
        "target": ctx.target,
        "aids": ctx.aids,
        "families": ctx.families,
        "signatures": ctx.signatures,
    }
    non_r_tasks = [
        (arm_id, seed, r1_mod.EXPANDED_NON_R_DESIGNS)
        for arm_id in new_arms
        for seed in sl.SEEDS
    ]
    n_workers = max(1, min(int(workers), len(non_r_tasks), 8))
    if n_workers == 1:
        r1_mod._init_non_r_worker(worker_state)
        for task in non_r_tasks:
            r1_mod._fit_non_r_cell_worker(task)
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("spawn"),
            initializer=r1_mod._init_non_r_worker,
            initargs=(worker_state,),
        ) as pool:
            for fut in as_completed([pool.submit(r1_mod._fit_non_r_cell_worker, t) for t in non_r_tasks]):
                fut.result()

    if not include_r_bridge:
        return
    r_tasks = [
        (arm_id, seed, str(new_readout_paths[arm_id]))
        for arm_id in new_arms
        if arm_id in new_readout_paths
        for seed in sl.SEEDS
    ]
    if not r_tasks:
        return
    if n_workers == 1:
        r1_mod._init_non_r_worker(worker_state)
        for task in r_tasks:
            r1_mod._fit_r_bridge_cell_worker(task)
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("spawn"),
            initializer=r1_mod._init_non_r_worker,
            initargs=(worker_state,),
        ) as pool:
            for fut in as_completed([pool.submit(r1_mod._fit_r_bridge_cell_worker, t) for t in r_tasks]):
                fut.result()


def _r1_evaluate_bidirectional_subset(
    arm_ids: Sequence[str],
    metrics_by_cell: Mapping[tuple[str, int, str], Mapping[str, Mapping[str, float]]],
    assay_family: Mapping[str, str],
    assay_names: Sequence[str],
    control_designs: Mapping[str, tuple[str, str]],
    *,
    draws: int = r1_mod.BOOTSTRAP_DRAWS,
    seed: int = r1_mod.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute exact 10,000-draw paired_bands for arbitrary arm sets on R1 bidirectional designs."""
    out: dict[str, Any] = {}
    for metric in ("spearman", "rank_mse"):
        sign = 1.0 if metric == "spearman" else -1.0
        col_meta: list[dict[str, Any]] = []
        col_vecs: list[np.ndarray] = []
        n_groups = 0
        exclude_set = frozenset(str(a) for a in assay_family.keys() if a not in set(assay_names))
        for label, (base_d, joint_d) in control_designs.items():
            for arm_id in arm_ids:
                f_base, dmap = r1_mod._family_matrix_for_designs(
                    metrics_by_cell,
                    assay_family,
                    [arm_id],
                    [base_d, joint_d, "M"],
                    metric,
                    exclude_assays=exclude_set,
                )
                m_base = dmap[(arm_id, base_d)]
                m_joint = dmap[(arm_id, joint_d)]
                m_alone = dmap[(arm_id, "M")]
                n_groups = len(f_base)
                col_vecs.append(sign * (m_joint - m_base))
                col_meta.append(
                    {
                        "control": label,
                        "arm_id": arm_id,
                        "direction": "model_given_controls",
                        "conditioning_regime": EXTENDED_ARM_METADATA[arm_id].conditioning_regime,
                        "paradigm": EXTENDED_ARM_METADATA[arm_id].paradigm,
                        "base_mean": float(np.mean(m_base)),
                        "model_mean": float(np.mean(m_alone)),
                        "joint_mean": float(np.mean(m_joint)),
                    }
                )
                col_vecs.append(sign * (m_joint - m_alone))
                col_meta.append(
                    {
                        "control": label,
                        "arm_id": arm_id,
                        "direction": "controls_given_model",
                        "conditioning_regime": EXTENDED_ARM_METADATA[arm_id].conditioning_regime,
                        "paradigm": EXTENDED_ARM_METADATA[arm_id].paradigm,
                        "base_mean": float(np.mean(m_base)),
                        "model_mean": float(np.mean(m_alone)),
                        "joint_mean": float(np.mean(m_joint)),
                    }
                )
        mat = np.column_stack(col_vecs)
        ok_rows = np.all(np.isfinite(mat), axis=1)
        bands = r2_mod.paired_bands_summary(
            mat[ok_rows],
            [f"{m['control']}|{m['direction']}|{m['arm_id']}" for m in col_meta],
            draws=draws,
            seed=seed,
        )
        rows = []
        for idx, meta in enumerate(col_meta):
            cname = f"{meta['control']}|{meta['direction']}|{meta['arm_id']}"
            rows.append({**meta, **bands["columns"][cname]})
        by_ctrl_dir: dict[str, dict[str, Any]] = {}
        for label in control_designs:
            by_ctrl_dir[label] = {}
            for direction in ("model_given_controls", "controls_given_model"):
                sub = [r for r in rows if r["control"] == label and r["direction"] == direction]
                by_ctrl_dir[label][direction] = {
                    "counts_native": bidir.counts([r["verdict_native"] for r in sub]),
                    "counts_studentized": bidir.counts([r["verdict_studentized"] for r in sub]),
                    "counts_marginal": bidir.counts([r["verdict_marginal"] for r in sub]),
                    "by_conditioning_regime": _conditioning_regime_summary(
                        [r["arm_id"] for r in sub], {r["arm_id"]: r["point"] for r in sub}
                    )["by_conditioning_regime"],
                }
        out[metric] = {
            "n_groups": n_groups,
            "family_size": len(col_meta),
            "n_arms": len(arm_ids),
            "draws": draws,
            "seed": seed,
            "native_critical_value": bands["native_critical_value"],
            "studentized_critical_t": bands["studentized_critical_t"],
            "summary_by_control_direction": by_ctrl_dir,
            "columns": rows,
        }
    return out


def _compute_adversarial_pairs_extended(
    ctx: r1_mod.R1UnifiedContext,
    raw_scores_by_arm: Mapping[str, np.ndarray],
    arm_order: Sequence[str],
    *,
    draws: int = r1_mod.BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Run exact explicit pair-level ablation over all ``arm_order`` models on the 24,832-row support."""
    samples_path = ctx.root / r1_mod.MUTATION_SAMPLES_CSV
    if not samples_path.is_file():
        return {"available": False, "reason": f"Missing frozen mutation samples: {samples_path}"}

    samples_df = pd.read_csv(samples_path)
    support = r1_mod.md.common_support(samples_df)
    sid_to_idx = {str(sid): idx for idx, sid in enumerate(ctx.sample_ids)}
    sup_sids = support["sample_id"].astype(str).tolist()
    order = np.asarray([sid_to_idx[sid] for sid in sup_sids], dtype=np.int64)
    support = support.copy()
    support["profile"] = support["profile_score"].to_numpy(dtype=np.float64)
    support["potts"] = ctx.raw_baselines["potts"][order]
    support["gemme"] = ctx.raw_baselines["gemme"][order]

    refused_mask = support["unit_id"].astype(str).isin(r1_mod.GEMME_REFUSED_ASSAYS)
    support = support.loc[~refused_mask].reset_index(drop=True)
    for stat in r1_mod.PAIR_STATISTICS:
        support[f"rank_{stat}"] = support.groupby("unit_id")[stat].transform(
            lambda b: r1_mod.md.standardized_rank(b.to_numpy())
        )
    stat_ranks = {
        stat: dict(
            zip(support["sample_id"].astype(str), support[f"rank_{stat}"].to_numpy(dtype=np.float64), strict=True)
        )
        for stat in r1_mod.PAIR_STATISTICS
    }

    kept_order = np.asarray([sid_to_idx[sid] for sid in support["sample_id"].astype(str)], dtype=np.int64)
    sup_aids = support["unit_id"].astype(str).to_numpy()
    sup_sids_kept = support["sample_id"].astype(str).tolist()
    model_ranks: dict[str, dict[str, float]] = {}
    for arm_id in arm_order:
        vals = np.asarray(raw_scores_by_arm[arm_id], dtype=np.float64)[kept_order]
        ranked = r1_mod.overlap.rerank(vals, sup_aids)
        model_ranks[arm_id] = dict(zip(sup_sids_kept, ranked.tolist(), strict=True))

    singles = r1_mod.md.single_substitutions(support)
    all_candidates = r1_mod.md.candidate_pairs(singles)
    gap_candidates = all_candidates.loc[all_candidates["phenotype_gap"] >= r1_mod.md.PHENOTYPE_GAP].copy()
    gap_candidates = r1_mod._pair_orient(gap_candidates)
    for stat in r1_mod.PAIR_STATISTICS:
        gap_candidates = r1_mod._pair_attach_covariate(gap_candidates, stat, stat_ranks[stat])

    tol = float(r1_mod.PAIR_PRIMARY_TOLERANCE)
    open_tier = r1_mod.md.greedy_disjoint(gap_candidates, seed=r1_mod.md.SPLIT_SEED)
    tol_p_sub = r1_mod._pair_tolerance_matched(gap_candidates, ("profile",), tol)
    tol_p_frame = r1_mod.md.greedy_disjoint(tol_p_sub, seed=r1_mod.md.SPLIT_SEED) if len(tol_p_sub) else tol_p_sub
    tol_peg_sub = r1_mod._pair_tolerance_matched(gap_candidates, r1_mod.PAIR_STATISTICS, tol)
    tol_peg_frame = r1_mod.md.greedy_disjoint(tol_peg_sub, seed=r1_mod.md.SPLIT_SEED) if len(tol_peg_sub) else tol_peg_sub

    tiers_forward: dict[str, pd.DataFrame] = {
        "T_open": open_tier,
        "T_tol_P": tol_p_frame,
        "T_tol_PEG": tol_peg_frame,
        "T_bal_P": r1_mod._pair_balance_votes(open_tier, ("profile",), seed=r1_mod.PAIR_BALANCE_SEED)[0],
        "T_bal_PEG": r1_mod._pair_balance_votes(open_tier, r1_mod.PAIR_STATISTICS, seed=r1_mod.PAIR_BALANCE_SEED)[0],
        "T_tol_bal_PEG": r1_mod._pair_balance_votes(
            tol_peg_frame, r1_mod.PAIR_STATISTICS, seed=r1_mod.PAIR_BALANCE_SEED
        )[0],
    }

    forward_out: dict[str, Any] = {}
    for tname, tdf in tiers_forward.items():
        if tdf.empty:
            forward_out[tname] = {"n_pairs": 0, "n_families": 0, "arms": {}}
            continue
        clusters = tdf["group_id"].tolist()
        units = sorted({int(c) for c in clusters})
        resampler = (
            r1_mod.md.ClusterResampler(units, draws=draws, seed=r1_mod.PAIR_BOOTSTRAP_SEED)
            if not r1_mod.bootstrap_unit_floor(len(units))["degenerate"]
            else None
        )
        arm_cols = {
            arm_id: r1_mod.md.direction_values(tdf, model_ranks[arm_id]) for arm_id in arm_order
        }
        fam_bands = r1_mod._pair_family_bands(
            arm_cols,
            clusters,
            draws=draws,
            seed=r1_mod.PAIR_BOOTSTRAP_SEED,
            verdict_from="simultaneous",
            resampler=resampler,
        )
        forward_out[tname] = {
            "rule": r1_mod.PAIR_TIER_RULES.get(tname, tname),
            "tolerance": tol if "tol" in tname else None,
            "counts": r1_mod.md.tier_counts(tdf),
            "n_pairs": int(len(tdf)),
            "n_families": len(units),
            "estimand": "pair_direction_advantage",
            "family_all_arms": fam_bands,
            "stratified_summary": _conditioning_regime_summary(
                arm_order, {a: fam_bands["columns"][a]["pooled"] for a in arm_order}
            ),
        }

    return {
        "available": True,
        "schema": "r1_adversarial_pairs_full_v1",
        "generated_utc": _now_utc(),
        "n_models": len(arm_order),
        "models": list(arm_order),
        "arm_metadata": {a: extended_arm_meta_dict(a) for a in arm_order},
        "primary_tolerance": tol,
        "forward_tiers": forward_out,
    }


def assemble_r1_proteingym_77(
    *,
    root: Path = REPO_ROOT,
    scoring_dir: Path | None = None,
    out_dir: Path | None = None,
    workers: int = 4,
    threads: int = 2,
    draws: int = r1_mod.BOOTSTRAP_DRAWS,
    allow_pending: bool = False,
) -> dict[str, Any]:
    """Assemble ``r1_proteingym_unified_77.json``, ``r1_supplementary_c0_ladder_77.json``,
    and ``r1_adversarial_pairs_77.json`` (reusing all 62 cached cells + fitting new arms).
    """
    t0 = time.monotonic()
    base = Path(root).resolve()
    sdir = Path(scoring_dir) if scoring_dir is not None else (base / DEFAULT_NEW_SCORING_DIR)
    odir = Path(out_dir) if out_dir is not None else (base / UNIFIED_OUTPUT_DIR)
    existing_r1_cache = base / UNIFIED_OUTPUT_DIR / "cells/r1"
    new_r1_cache = base / DEFAULT_NEW_CELL_CACHE_DIR / "r1"

    score_paths, readout_paths, new_dtypes = _discover_new_r1_artifacts(sdir)
    present_new_primary = [a for a in NEW_19_PRIMARY_ARMS if a in score_paths]
    present_new_struct = [a for a in DPLM2_4_STRUCT_ARMS if a in score_paths]
    missing_primary = _check_pending_arms(
        NEW_19_PRIMARY_ARMS, present_new_primary, benchmark="R1 ProteinGym Full (81)", allow_pending=allow_pending
    )
    missing_struct = [a for a in DPLM2_4_STRUCT_ARMS if a not in score_paths]

    ctx = r1_mod.load_r1_unified_context(base, out_dir=odir)
    new_all = present_new_primary + present_new_struct
    new_raw_scores: dict[str, np.ndarray] = {}
    new_provenance: dict[str, dict[str, Any]] = {}
    for arm_id in new_all:
        sp = score_paths[arm_id]
        vec = _load_r1_score_vector_npz(arm_id, sp, ctx.sample_ids)
        new_raw_scores[arm_id] = vec
        new_provenance[arm_id] = {
            "path": _rel_path(sp, base),
            "sha256": sha256_file(sp),
            "dtype_realised": new_dtypes.get(arm_id),
            "has_readout_1024": arm_id in readout_paths,
            "readout_path": _rel_path(readout_paths[arm_id], base) if arm_id in readout_paths else None,
            "conditioning_regime": EXTENDED_ARM_METADATA[arm_id].conditioning_regime,
        }

    # Fit ONLY new arms into isolated cache dir (never mutates cells/r1!)
    _fit_new_r1_cells(
        ctx,
        new_all,
        new_raw_scores,
        readout_paths,
        new_r1_cache,
        workers=workers,
        threads=threads,
        include_r_bridge=True,
    )

    # Load existing 62-arm per-assay metrics from existing cache, then attach new arms
    metrics_by_cell, assay_family, assay_names = r1_mod._load_per_assay_metrics_table(
        ctx, existing_r1_cache, include_r_bridge=True
    )
    anchor_62 = UNIFIED_62_SINGLE_SEQUENCE_ARMS[0]
    shared_design_names = (
        *r1_mod.SHARED_SCALAR_DESIGNS,
        *r1_mod.CACHED_CONTROL_DESIGNS_LADDER,
        *r1_mod.CACHED_CONTROL_DESIGNS_PROG,
    )
    for arm_id in new_all:
        for seed in sl.SEEDS:
            for dname in shared_design_names:
                metrics_by_cell[(arm_id, seed, dname)] = metrics_by_cell[(anchor_62, seed, dname)]
            preds_model: dict[str, np.ndarray] = {}
            with np.load(r1_mod._cache_file(new_r1_cache, "non_r", arm_id, seed), allow_pickle=False) as d:
                for k in r1_mod.EXPANDED_NON_R_DESIGNS:
                    preds_model[k] = np.asarray(d[k], dtype=np.float64)
            if arm_id in readout_paths:
                with np.load(r1_mod._cache_file(new_r1_cache, "r_bridge", arm_id, seed), allow_pickle=False) as d:
                    for k in r1_mod.R_BRIDGE_DESIGNS:
                        preds_model[k] = np.asarray(d[k], dtype=np.float64)
            cell_m = r1_mod._assay_metrics_for_predictions(preds_model, ctx.target, ctx.aids, ctx.families)
            for dname, mdict in cell_m.items():
                metrics_by_cell[(arm_id, seed, dname)] = mdict

    all_raw_scores = {**ctx.raw_model_scores, **new_raw_scores}
    arm_order_primary = tuple(list(UNIFIED_62_SINGLE_SEQUENCE_ARMS) + present_new_primary)
    arm_order_with_struct = tuple(list(arm_order_primary) + present_new_struct)
    arm_order_seq_only = tuple(
        a for a in arm_order_primary if EXTENDED_ARM_METADATA[a].conditioning_regime == "sequence_only"
    )

    # Layer 1: Zero-shot Spearman (full 201 assays / 163 clusters & 194 GEMME subset)
    def _layer1_panel(arms_sub: Sequence[str], assay_mask: np.ndarray | None = None) -> dict[str, Any]:
        if assay_mask is None:
            t_vec, a_vec, f_vec = ctx.target, ctx.aids, ctx.families
            m_map = {a: all_raw_scores[a] for a in arms_sub}
            b_map = ctx.raw_baselines
        else:
            t_vec = ctx.target[assay_mask]
            a_vec = ctx.aids[assay_mask]
            f_vec = ctx.families[assay_mask]
            m_map = {a: all_raw_scores[a][assay_mask] for a in arms_sub}
            b_map = {k: v[assay_mask] for k, v in ctx.raw_baselines.items()}

        fam_vecs: dict[str, np.ndarray] = {}
        fams_ref = None
        for a in arms_sub:
            fams, vec, _undef = r1_mod._equal_family_zero_shot_vector(m_map[a], t_vec, a_vec, f_vec)
            fams_ref = fams
            fam_vecs[a] = vec
        exp_vecs: dict[str, np.ndarray] = {}
        for bname, bvals in b_map.items():
            _, bvec, _undef = r1_mod._equal_family_zero_shot_vector(bvals, t_vec, a_vec, f_vec)
            exp_vecs[bname] = bvec

        mat = np.column_stack([fam_vecs[a] for a in arms_sub])
        ok_m = np.all(np.isfinite(mat), axis=1)
        arm_bands = r2_mod.paired_bands_summary(
            mat[ok_m], list(arms_sub), draws=draws, seed=r1_mod.BOOTSTRAP_SEED
        )
        exp_out: dict[str, Any] = {}
        contrasts: dict[str, Any] = {}
        for b, bvec in exp_vecs.items():
            ok_b = ok_m & np.isfinite(bvec)
            if not ok_b.any():
                continue
            exp_out[b] = r2_mod.paired_bands_summary(
                bvec[ok_b, None], [b], draws=draws, seed=r1_mod.BOOTSTRAP_SEED
            )["columns"][b]
            contrasts[f"model_minus_{b}"] = r2_mod.paired_bands_summary(
                mat[ok_b] - bvec[ok_b, None], list(arms_sub), draws=draws, seed=r1_mod.BOOTSTRAP_SEED
            )
        return {
            "n_families": len(fams_ref or ()),
            "n_arms": len(arms_sub),
            "explicit_baselines": exp_out,
            "arms": arm_bands,
            "paired_contrasts_vs_explicit": contrasts,
            "stratified_summary": _conditioning_regime_summary(
                arms_sub, {a: arm_bands["columns"][a]["point"] for a in arms_sub}
            ),
            "dplm2_struct_minus_seq_paired": _dplm2_struct_minus_seq_paired(
                fam_vecs, draws=draws, seed=r1_mod.BOOTSTRAP_SEED
            ),
        }

    gemme_mask = ~np.isin(ctx.aids, list(r1_mod.GEMME_REFUSED_ASSAYS))
    layer1 = {
        "full_201_assays_primary": _layer1_panel(arm_order_with_struct),
        "full_201_assays_sequence_only": _layer1_panel(arm_order_seq_only),
        "gemme_194_subset_primary": _layer1_panel(arm_order_with_struct, assay_mask=gemme_mask),
    }

    # Layer 2 Core Bidirectional Increments
    core_controls = {
        "P_1col": ("P1", "MP1"),
        "PEG_3col": ("PEG3", "MPEG3"),
        "P_14col": ("P", "MP"),
        "PEG_16col": ("PEG", "MPEG"),
    }
    layer2_core = {
        "all_arms_with_struct": _r1_evaluate_bidirectional_subset(
            arm_order_with_struct, metrics_by_cell, assay_family, assay_names, core_controls, draws=draws
        ),
        "sequence_only_panel": _r1_evaluate_bidirectional_subset(
            arm_order_seq_only, metrics_by_cell, assay_family, assay_names, core_controls, draws=draws
        ),
    }

    # Layer 2 +R Representation Bridge (29 expanded + all new arms with 1024-d R)
    r_bridge_arms = tuple(list(ctx.r_bridge_arms) + [a for a in new_all if a in readout_paths])
    r_bridge_controls = {
        "R_given_P1": ("P1", "P1_R"),
        "MR_given_P1": ("P1", "P1_M_R"),
        "R_given_PEG3": ("PEG3", "PEG3_R"),
        "MR_given_PEG3": ("PEG3", "PEG3_M_R"),
        "MR_given_PEG16": ("PEG", "PEG16_M_R"),
    }
    layer2_r_bridge = {
        "n_arms_with_R": len(r_bridge_arms),
        "arms_with_R": list(r_bridge_arms),
        "contrasts": _r1_evaluate_bidirectional_subset(
            r_bridge_arms, metrics_by_cell, assay_family, assay_names, r_bridge_controls, draws=draws
        ),
    }

    # Supplementary 6-rung C0 ladder
    c0_rungs = {
        "P": ("P", "MP"),
        "PE": ("PE", "MPE"),
        "PG": ("PG", "MPG"),
        "PEG": ("PEG", "MPEG"),
        "BPL": ("BPL", "BMPL"),
        "BPLEG": ("BPLEG", "BMPLEG"),
    }
    c0_payload = {
        "schema": "r1_supplementary_c0_ladder_77_v1",
        "generated_utc": _now_utc(),
        "completion_status": "complete" if not (missing_primary or missing_struct) else "partial_pending_h200",
        "n_models_primary": len(arm_order_primary),
        "n_models_with_struct_companions": len(arm_order_with_struct),
        "models": list(arm_order_with_struct),
        "pending_primary_arms": missing_primary,
        "pending_struct_companions": missing_struct,
        "arm_metadata": {a: extended_arm_meta_dict(a) for a in arm_order_with_struct},
        "all_arms_with_struct": _r1_evaluate_bidirectional_subset(
            arm_order_with_struct, metrics_by_cell, assay_family, assay_names, c0_rungs, draws=draws
        ),
        "sequence_only_panel": _r1_evaluate_bidirectional_subset(
            arm_order_seq_only, metrics_by_cell, assay_family, assay_names, c0_rungs, draws=draws
        ),
    }

    # Explicit adversarial pairs
    adv_payload = _compute_adversarial_pairs_extended(
        ctx, all_raw_scores, arm_order_with_struct, draws=draws
    )
    adv_payload["completion_status"] = (
        "complete" if not (missing_primary or missing_struct) else "partial_pending_h200"
    )
    adv_payload["pending_primary_arms"] = missing_primary
    adv_payload["pending_struct_companions"] = missing_struct

    r1_main_payload = {
        "schema": "r1_proteingym_unified_77_v1",
        "generated_utc": _now_utc(),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "completion_status": "complete" if not (missing_primary or missing_struct) else "partial_pending_h200",
        "n_models_primary": len(arm_order_primary),
        "n_models_with_struct_companions": len(arm_order_with_struct),
        "models_primary": list(arm_order_primary),
        "models_with_struct_companions": list(arm_order_with_struct),
        "sequence_only_models": list(arm_order_seq_only),
        "backbone_3d_conditioned_models": [
            a for a in arm_order_with_struct if EXTENDED_ARM_METADATA[a].conditioning_regime != "sequence_only"
        ],
        "pending_primary_arms": missing_primary,
        "pending_struct_companions": missing_struct,
        "arm_metadata": {a: extended_arm_meta_dict(a) for a in arm_order_with_struct},
        "new_arm_provenance": new_provenance,
        "layer1_zero_shot": layer1,
        "layer2_core_bidirectional": layer2_core,
        "layer2_representation_bridge": layer2_r_bridge,
    }

    odir.mkdir(parents=True, exist_ok=True)
    p_main = odir / "r1_proteingym_unified_full.json"
    p_c0 = odir / "r1_supplementary_c0_ladder_full.json"
    p_adv = odir / "r1_adversarial_pairs_full.json"
    sha_main = write_verified_json_with_sha256(p_main, r1_main_payload)
    sha_c0 = write_verified_json_with_sha256(p_c0, c0_payload)
    sha_adv = write_verified_json_with_sha256(p_adv, adv_payload)
    return {
        "benchmark": "r1",
        "completion_status": r1_main_payload["completion_status"],
        "n_models_primary": len(arm_order_primary),
        "n_models_with_struct": len(arm_order_with_struct),
        "outputs": {
            p_main.name: sha_main,
            p_c0.name: sha_c0,
            p_adv.name: sha_adv,
        },
    }


# ============================================================================
# 2. Family Memory Unified 77 (family_conditionals + family_distribution)
# ============================================================================


def assemble_family_memory_77(
    *,
    root: Path = REPO_ROOT,
    scoring_dir: Path | None = None,
    out_dir: Path | None = None,
    draws: int = 10000,
    seed: int = 20261106,
    allow_pending: bool = False,
) -> dict[str, Any]:
    """Extend ``family_memory_unified_62.json`` to include DPLM-2 (66 sequence-only arms)
    plus honest accounting for 3D-conditioned inverse-folding arms on unaligned homologs.
    """
    t0 = time.monotonic()
    base = Path(root).resolve()
    sdir = Path(scoring_dir) if scoring_dir is not None else (base / DEFAULT_NEW_SCORING_DIR)
    odir = Path(out_dir) if out_dir is not None else (base / UNIFIED_OUTPUT_DIR)

    fc_base = mem_mod._load_fc_baselines_and_sites(base)
    unit_order = fc_base["unit_order"]
    unit_to_wt = fc_base["unit_to_wt"]
    prof_by_unit = fc_base["profile_logp_by_unit"]
    mkv_by_unit = fc_base["markov_logp_by_unit"]
    msa_support = load_msa_support(
        base / "results/extensions/statistical_ladder_20261106/inputs/support.json"
    )

    # Reuse the 62-arm family_conditionals vectors directly from the sealed 62-arm artifact
    # or compute from raw inputs if absent.
    sealed_fm_62 = load_json_file(base / UNIFIED_OUTPUT_DIR / "family_memory_unified_62.json")
    fc_62_doc = sealed_fm_62["family_conditionals_62"]

    # Load exact 62-arm per-unit logp from raw sources (fast, ~1s on CPU) to guarantee exact joint banding
    fc_root = base / "results/extensions/family_conditionals_20261107"
    family_logp_by_arm: dict[str, dict[str, float]] = {}
    arm_fc_meta: dict[str, dict[str, Any]] = {}

    for arm_id in PREREGISTERED_33_ARMS:
        spath = fc_root / "scores" / arm_id / "evaluation.json"
        doc = load_json_file(spath)
        units_map = doc["units"]
        fam_buckets: dict[str, list[float]] = defaultdict(list)
        for srec in fc_base["site_records"]:
            uid = srec["unit_id"]
            pos = srec["query_position"]
            lps = units_map[uid]["sites"][pos]["logprob"]
            obs_idx = srec["obs_indices"]
            fam_buckets[uid].append(float(np.mean([float(lp[idx]) for lp, idx in zip(lps, obs_idx)])))
        family_logp_by_arm[arm_id] = {uid: float(np.mean(fam_buckets[uid])) for uid in unit_order}
        arm_fc_meta[arm_id] = {**extended_arm_meta_dict(arm_id), "conditional_estimator": "causal_20string"}

    ctx_to_site = {
        (srec["unit_id"], str(srec["query_position"]), str(rid)): s_idx
        for s_idx, srec in enumerate(fc_base["site_records"])
        for rid in srec["row_ids"]
    }
    gp_single = [a for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS if a not in set(PREREGISTERED_33_ARMS)]
    for arm_id in gp_single:
        art = roster_mod.resolve_gp_r1_json(arm_id, root=base)
        if arm_id == "esm3-1.4b":
            per_site: list[list[float]] = [[] for _ in range(len(fc_base["site_records"]))]
            with art.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("partition") != "evaluation":
                        continue
                    s_idx = ctx_to_site.get(
                        (str(row["unit_id"]), str(row["query_position"]), str(row["row_id"]))
                    )
                    if s_idx is not None:
                        per_site[s_idx].append(float(row["observed_logp_nats"]))
            fam_buckets = defaultdict(list)
            for srec, vs in zip(fc_base["site_records"], per_site):
                fam_buckets[srec["unit_id"]].append(float(np.mean(vs)))
            family_logp_by_arm[arm_id] = {uid: float(np.mean(fam_buckets[uid])) for uid in unit_order}
            arm_fc_meta[arm_id] = {**extended_arm_meta_dict(arm_id), "conditional_estimator": "masked_1pass"}
        else:
            doc = load_json_file(art.path)
            eval_map = {
                str(u["unit_id"]): float(u["model_logp"])
                for u in doc["family_conditionals"]["units"]
                if u.get("partition") == "evaluation"
            }
            family_logp_by_arm[arm_id] = eval_map
            arm_fc_meta[arm_id] = {**extended_arm_meta_dict(arm_id), "conditional_estimator": "masked_or_causal"}

    # Discover new DPLM-2 + DiMA (and any optional struct-conditioned) family_conditionals files
    fc_dir = sdir / "r1"
    new_fc_loaded: list[str] = []
    for arm_id in NEW_23_SCORED_ENTRIES:
        p = fc_dir / f"r1_family_conditionals_{arm_slug(arm_id)}.json"
        if not p.is_file():
            continue
        doc = load_json_file(p)
        units = doc.get("units") or (doc.get("family_conditionals") or {}).get("units") or []
        eval_map = {
            str(u["unit_id"]): float(u["model_logp"])
            for u in units
            if u.get("partition") == "evaluation"
        }
        if set(eval_map.keys()) == set(unit_order):
            family_logp_by_arm[arm_id] = eval_map
            arm_fc_meta[arm_id] = {
                **extended_arm_meta_dict(arm_id),
                "conditional_estimator": str(doc.get("conditional_semantics", "masked_1pass")),
                "source_path": _rel_path(p, base),
            }
            new_fc_loaded.append(arm_id)

    missing_fc_seq = _check_pending_arms(
        DPLM2_4_ARMS + DIMA_4_ARMS,
        new_fc_loaded,
        benchmark="Family Memory (DPLM-2 + DiMA seq-only)",
        allow_pending=allow_pending,
    )

    fc_arms = list(UNIFIED_62_SINGLE_SEQUENCE_ARMS) + new_fc_loaded
    prof_vec = np.asarray([prof_by_unit[u] for u in unit_order], dtype=np.float64)
    mkv_vec = np.asarray([mkv_by_unit[u] for u in unit_order], dtype=np.float64)
    prof_ce_bits = -prof_vec / LN2
    depths_vec = np.asarray(
        [msa_support[unit_to_wt[u]]["effective_depth_per_site"] for u in unit_order], dtype=np.float64
    )
    mat_logp = np.column_stack(
        [np.asarray([family_logp_by_arm[a][u] for u in unit_order], dtype=np.float64) for a in fc_arms]
    )
    strata_indices = {
        "overall_64": np.arange(len(unit_order), dtype=int),
        "conserved_profile_lt_1_8_bits": np.flatnonzero(prof_ce_bits < 1.8),
        "non_conserved_profile_ge_1_8_bits": np.flatnonzero(prof_ce_bits >= 1.8),
        "shallow_neff_lt_5": np.flatnonzero(depths_vec < 5.0),
        "deep_neff_ge_5": np.flatnonzero(depths_vec >= 5.0),
    }
    fc_bands: dict[str, Any] = {}
    for sname, sidx in strata_indices.items():
        fc_bands[sname] = {
            "n_families": int(sidx.size),
            "profile_baseline_nats": float(np.mean(prof_vec[sidx])),
            "markov_baseline_nats": float(np.mean(mkv_vec[sidx])),
            "model_minus_profile": mem_mod._run_simultaneous_family(
                fc_arms, (mat_logp - prof_vec[:, None])[sidx, :], draws=draws, seed=seed
            ),
            "model_minus_markov": mem_mod._run_simultaneous_family(
                fc_arms, (mat_logp - mkv_vec[:, None])[sidx, :], draws=draws, seed=seed
            ),
            "model_absolute_logp": mem_mod._run_simultaneous_family(
                fc_arms, mat_logp[sidx, :], draws=draws, seed=seed
            ),
        }

    # Family Distribution (46 existing + 4 DPLM-2 + 4 DiMA = 54 arms)
    fd_early = mem_mod._extract_early_fd_arms(base, unit_order)
    fd_gp = mem_mod._extract_gp_fd_arms(base, unit_order)
    fd_all = {**fd_early, **fd_gp}
    fd_dir = sdir / "r1_family_distribution"
    new_fd_loaded: list[str] = []
    for arm_id in NEW_23_SCORED_ENTRIES:
        p = fd_dir / f"r1_family_distribution_{arm_slug(arm_id)}.json"
        if not p.is_file():
            continue
        doc = load_json_file(p)
        eval_col: dict[str, float] = {}
        eval_comp: dict[str, float] = {}
        for u in doc.get("units", ()):
            if u.get("partition") != "evaluation":
                continue
            uid = str(u["unit_id"])
            eval_col[uid] = float(u["column_contrast"]["nats_per_residue"])
            eval_comp[uid] = float(u["composition_contrast"]["nats_per_residue"])
        if set(eval_col.keys()) == set(unit_order):
            fd_all[arm_id] = {
                "column_null_minus_native_nats_per_res": eval_col,
                "comp_null_minus_native_nats_per_res": eval_comp,
                "source_path": _rel_path(p, base),
            }
            new_fd_loaded.append(arm_id)

    missing_fd_seq = _check_pending_arms(
        DPLM2_4_ARMS + DIMA_4_ARMS,
        new_fd_loaded,
        benchmark="Family Distribution (DPLM-2 + DiMA seq-only)",
        allow_pending=allow_pending,
    )
    fd_arms = [a for a in (list(UNIFIED_62_SINGLE_SEQUENCE_ARMS) + new_fd_loaded) if a in fd_all]
    mat_fd_col = np.column_stack(
        [
            np.asarray([fd_all[a]["column_null_minus_native_nats_per_res"][u] for u in unit_order], dtype=np.float64)
            for a in fd_arms
        ]
    )
    mat_fd_comp = np.column_stack(
        [
            np.asarray([fd_all[a]["comp_null_minus_native_nats_per_res"][u] for u in unit_order], dtype=np.float64)
            for a in fd_arms
        ]
    )

    # DPLM-2 scaling slope on FC (if all 3 scales present)
    dplm2_fc_slope = None
    if all(a in family_logp_by_arm for a in ("dplm2-150m", "dplm2-650m", "dplm2-3b")):
        dplm2_fc_slope = cluster_bootstrap_scaling_slope(
            np.log10(np.asarray([150e6, 650e6, 3000e6], dtype=np.float64)),
            np.vstack(
                [
                    np.asarray([-family_logp_by_arm[a][u] / LN2 for u in unit_order], dtype=np.float64)
                    for a in ("dplm2-150m", "dplm2-650m", "dplm2-3b")
                ]
            ),
            n_draws=2000,
            seed=seed,
        )

    not_applicable_3d = [a for a in PROTEINMPNN_11_ARMS if a not in new_fc_loaded]
    payload = {
        "schema": "unified_family_memory_full_v1",
        "generated_utc": _now_utc(),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "completion_status": "complete" if not (missing_fc_seq or missing_fd_seq) else "partial_pending_h200",
        "sealed_62_reference": {
            "n_arms_fc_62": len(UNIFIED_62_SINGLE_SEQUENCE_ARMS),
            "fc_62_schema": fc_62_doc.get("schema") if isinstance(fc_62_doc, Mapping) else None,
        },
        "family_conditionals": {
            "n_arms": len(fc_arms),
            "arms": fc_arms,
            "new_arms_included": new_fc_loaded,
            "pending_seq_arms": missing_fc_seq,
            "not_applicable_3d_oracle_arms": not_applicable_3d,
            "not_applicable_reason": (
                "Inverse-folding models require per-homolog 3D coordinates for every variable-length "
                "UniRef homolog context in family_conditionals/family_distribution; unaligned homolog "
                "structures are not fabricated."
            ),
            "arm_metadata": arm_fc_meta,
            "strata": fc_bands,
            "dplm2_scaling_slope_bits_per_decade": dplm2_fc_slope,
        },
        "family_distribution": {
            "n_arms": len(fd_arms),
            "arms": fd_arms,
            "new_arms_included": new_fd_loaded,
            "pending_seq_arms": missing_fd_seq,
            "not_applicable_3d_oracle_arms": [a for a in PROTEINMPNN_11_ARMS if a not in new_fd_loaded],
            "column_null_contrast": mem_mod._run_simultaneous_family(fd_arms, mat_fd_col, draws=draws, seed=seed),
            "composition_null_contrast": mem_mod._run_simultaneous_family(
                fd_arms, mat_fd_comp, draws=draws, seed=seed
            ),
        },
    }
    odir.mkdir(parents=True, exist_ok=True)
    out_path = odir / "family_memory_unified_full.json"
    digest = write_verified_json_with_sha256(out_path, payload)
    return {
        "benchmark": "family_memory",
        "completion_status": payload["completion_status"],
        "n_fc_arms": len(fc_arms),
        "n_fd_arms": len(fd_arms),
        "outputs": {out_path.name: digest},
    }


# ============================================================================
# 3. R2 Biophysical Unified 77 (+4 DPLM-2|struct + 17 Ladders) Assembler
# ============================================================================


def _load_new_r2_npz(
    arm_id: str,
    npz_path: Path,
    expected_targets: np.ndarray,
    root: Path,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    with np.load(npz_path, allow_pickle=False) as z:
        m_arr = np.asarray(z["score"], dtype=np.float64).reshape(-1)
        t_arr = np.asarray(z["target"], dtype=np.float64).reshape(-1)
        r_arr = np.asarray(z["R"], dtype=np.float64) if "R" in z.files else None
    if len(m_arr) != len(expected_targets) or not np.allclose(t_arr, expected_targets, rtol=0, atol=1e-9):
        raise ValueError(f"{arm_id}: {npz_path.name} rows do not align with frozen cohort targets")
    if r_arr is not None and r_arr.shape != (len(expected_targets), 1024):
        raise ValueError(f"{arm_id}: unexpected R shape {r_arr.shape} in {npz_path}")
    if not np.isfinite(m_arr).all() or (r_arr is not None and not np.isfinite(r_arr).all()):
        raise ValueError(f"{arm_id}: non-finite entries in {npz_path}")
    sha = sha256_file(npz_path)
    rel = _rel_path(npz_path, root)
    return m_arr, r_arr, {
        "m_source": rel,
        "m_source_sha256": sha,
        "r_source": rel if r_arr is not None else None,
        "r_source_sha256": sha if r_arr is not None else None,
        "has_R": r_arr is not None,
    }


def _fit_new_r2_megascale_arm(
    arm_id: str,
    npz_path: Path,
    ms: dict[str, Any],
    fc_79_evo: Any,
    fc_101_evo: Any,
    fc_79_full: Any,
    fc_101_full: Any,
    cache_dir: Path,
    root: Path,
) -> dict[str, Any]:
    cache_path = cache_dir / f"{arm_slug(arm_id)}.json"
    expected_targets = np.asarray(
        [float(v["ddg"]) for bg in ms["cohort_101"]["backgrounds"] for v in bg["variants"]],
        dtype=np.float64,
    )
    m_101, r_101, meta = _load_new_r2_npz(arm_id, npz_path, expected_targets, root)
    if cache_path.is_file():
        cached = load_json_file(cache_path)
        if (
            cached.get("schema") == r2_mod.ARM_CACHE_SCHEMA
            and cached.get("m_source_sha256") == meta["m_source_sha256"]
            and cached.get("r_source_sha256") == meta["r_source_sha256"]
        ):
            return cached

    t0 = time.monotonic()
    p101, g101 = ms["panel_101"], ms["groups_101"]
    p79, g79, mask_79 = ms["panel_79"], ms["groups_79"], ms["mask_79"]
    m_79 = m_101[mask_79]
    r_79 = r_101[mask_79] if r_101 is not None else None

    _, zs_101 = pw_mod.group_spearman(p101["target"], m_101, p101["group"])
    _, zs_79 = pw_mod.group_spearman(p79["target"], m_79, p79["group"])

    eval_79 = lambda pred: r2_mod.evaluate_megascale_oof(pred, p79["target"], p79["group"], p79["site"], g79)  # noqa: E731
    eval_101 = lambda pred: r2_mod.evaluate_megascale_oof(pred, p101["target"], p101["group"], p101["site"], g101)  # noqa: E731

    # 1. evo_79
    b79_evo: dict[str, np.ndarray] = {
        "P": ms["p_logodds_79"][:, None],
        "POTTS": ms["potts_raw_79"][:, None],
        "GEMME": ms["gemme_raw_79"][:, None],
        "M": m_79[:, None],
    }
    d79_evo: dict[str, tuple[str, ...]] = {"M": ("M",), "C_M": ("P", "POTTS", "GEMME", "M")}
    if r_79 is not None:
        b79_evo["R"] = r_79
        d79_evo["R"] = ("R",)
        d79_evo["C_R"] = ("P", "POTTS", "GEMME", "R")
        d79_evo["C_M_R"] = ("P", "POTTS", "GEMME", "M", "R")
    r_79_evo = r2_mod.collapse_oof(
        r2_mod.fit_designs_with_cache(
            p79["target"], b79_evo, d79_evo, fc_79_evo, alphas=r2_mod.ALPHAS_17,
            renorm_val_weights=True, tie_largest_alpha=True,
        ),
        eval_79,
    )

    # 2. evo_101
    b101_evo: dict[str, np.ndarray] = {"P": ms["p_logodds_101"][:, None], "M": m_101[:, None]}
    d101_evo: dict[str, tuple[str, ...]] = {"M": ("M",), "C_M": ("P", "M")}
    if r_101 is not None:
        b101_evo["R"] = r_101
        d101_evo["R"] = ("R",)
        d101_evo["C_R"] = ("P", "R")
        d101_evo["C_M_R"] = ("P", "M", "R")
    r_101_evo = r2_mod.collapse_oof(
        r2_mod.fit_designs_with_cache(
            p101["target"], b101_evo, d101_evo, fc_101_evo, alphas=r2_mod.ALPHAS_17,
            renorm_val_weights=True, tie_largest_alpha=True,
        ),
        eval_101,
    )

    # 3. full_79
    b79_full: dict[str, np.ndarray] = {**p79["blocks"], "M": m_79[:, None]}
    d79_full: dict[str, tuple[str, ...]] = {"M": ("M",)}
    for rname, ccols in r2_mod.MS_PROGRESSION_RUNGS.items():
        d79_full[f"{rname}|M"] = (*ccols, "M")
    if r_79 is not None:
        b79_full["R"] = r_79
        d79_full["R"] = ("R",)
        ctop = r2_mod.MS_PROGRESSION_RUNGS["nuisance+profile+couplings+gemme"]
        d79_full["nuisance+profile+couplings+gemme|R"] = (*ctop, "R")
        d79_full["nuisance+profile+couplings+gemme|MR"] = (*ctop, "M", "R")
    r_79_full = r2_mod.collapse_oof(
        r2_mod.fit_designs_with_cache(p79["target"], b79_full, d79_full, fc_79_full, alphas=r2_mod.ALPHAS_5),
        eval_79,
    )

    # 4. full_101
    b101_full: dict[str, np.ndarray] = {**p101["blocks"], "M": m_101[:, None]}
    d101_full: dict[str, tuple[str, ...]] = {
        "M": ("M",),
        "S_M": (*r2_mod.MS_DESIGN_S, "M"),
        "S2_M": (*r2_mod.MS_DESIGN_S2, "M"),
    }
    if r_101 is not None:
        b101_full["R"] = r_101
        d101_full["R"] = ("R",)
        d101_full["S2_R"] = (*r2_mod.MS_DESIGN_S2, "R")
        d101_full["S2_MR"] = (*r2_mod.MS_DESIGN_S2, "M", "R")
    r_101_full = r2_mod.collapse_oof(
        r2_mod.fit_designs_with_cache(p101["target"], b101_full, d101_full, fc_101_full, alphas=r2_mod.ALPHAS_5),
        eval_101,
    )

    record = {
        "schema": r2_mod.ARM_CACHE_SCHEMA,
        "dataset": "megascale",
        "arm_id": arm_id,
        **meta,
        "elapsed_seconds": float(time.monotonic() - t0),
        "zero_shot_101": [float(v) for v in zs_101],
        "zero_shot_79": [float(v) for v in zs_79],
        "evo_79": {ep: {d: v.tolist() for d, v in by_d.items()} for ep, by_d in r_79_evo.items()},
        "evo_101": {ep: {d: v.tolist() for d, v in by_d.items()} for ep, by_d in r_101_evo.items()},
        "full_79": {ep: {d: v.tolist() for d, v in by_d.items()} for ep, by_d in r_79_full.items()},
        "full_101": {ep: {d: v.tolist() for d, v in by_d.items()} for ep, by_d in r_101_full.items()},
    }
    atomic_write_json(cache_path, record)
    return record


def _fit_new_r2_domainome_arm(
    arm_id: str,
    npz_path: Path,
    dm: dict[str, Any],
    fc_94_evo: Any,
    fc_96_evo: Any,
    fc_94_full: Any,
    cache_dir: Path,
    root: Path,
) -> dict[str, Any]:
    cache_path = cache_dir / f"{arm_slug(arm_id)}.json"
    expected_targets = np.asarray(
        [float(v["target"]) for d in dm["cohort_96"]["domains"] for v in d["variants"]],
        dtype=np.float64,
    )
    m_96, r_96, meta = _load_new_r2_npz(arm_id, npz_path, expected_targets, root)
    if cache_path.is_file():
        cached = load_json_file(cache_path)
        if (
            cached.get("schema") == r2_mod.ARM_CACHE_SCHEMA
            and cached.get("m_source_sha256") == meta["m_source_sha256"]
            and cached.get("r_source_sha256") == meta["r_source_sha256"]
        ):
            return cached

    t0 = time.monotonic()
    p96, g96 = dm["panel_96"], dm["groups_96"]
    p94, g94, mask_94 = dm["panel_94"], dm["groups_94"], dm["mask_94"]
    m_94 = m_96[mask_94]
    r_94 = r_96[mask_94] if r_96 is not None else None

    zs_96 = r2_mod.evaluate_domainome_oof(
        m_96, p96["target"], p96["group"], p96["domain"], p96["site"], g96, use_nested_weights=False
    )["ranking"]
    zs_94 = r2_mod.evaluate_domainome_oof(
        m_94, p94["target"], p94["group"], p94["domain"], p94["site"], g94, use_nested_weights=False
    )["ranking"]

    eval_94 = lambda pred: r2_mod.evaluate_domainome_oof(  # noqa: E731
        pred, p94["target"], p94["group"], p94["domain"], p94["site"], g94, use_nested_weights=False
    )
    eval_96 = lambda pred: r2_mod.evaluate_domainome_oof(  # noqa: E731
        pred, p96["target"], p96["group"], p96["domain"], p96["site"], g96, use_nested_weights=False
    )
    eval_94_full = lambda pred: r2_mod.evaluate_domainome_oof(  # noqa: E731
        pred, p94["target"], p94["group"], p94["domain"], p94["site"], g94, use_nested_weights=True
    )

    b94_evo: dict[str, np.ndarray] = {
        "P": dm["p_logodds_94"][:, None],
        "POTTS": dm["potts_raw_94"][:, None],
        "GEMME": dm["gemme_raw_94"][:, None],
        "M": m_94[:, None],
    }
    d94_evo: dict[str, tuple[str, ...]] = {"M": ("M",), "C_M": ("P", "POTTS", "GEMME", "M")}
    if r_94 is not None:
        b94_evo["R"] = r_94
        d94_evo["R"] = ("R",)
        d94_evo["C_R"] = ("P", "POTTS", "GEMME", "R")
        d94_evo["C_M_R"] = ("P", "POTTS", "GEMME", "M", "R")
    r_94_evo = r2_mod.collapse_oof(
        r2_mod.fit_designs_with_cache(
            p94["target"], b94_evo, d94_evo, fc_94_evo, alphas=r2_mod.ALPHAS_17,
            renorm_val_weights=True, tie_largest_alpha=True,
        ),
        eval_94,
    )

    b96_evo: dict[str, np.ndarray] = {"P": dm["p_logodds_96"][:, None], "M": m_96[:, None]}
    d96_evo: dict[str, tuple[str, ...]] = {"M": ("M",), "C_M": ("P", "M")}
    if r_96 is not None:
        b96_evo["R"] = r_96
        d96_evo["R"] = ("R",)
        d96_evo["C_R"] = ("P", "R")
        d96_evo["C_M_R"] = ("P", "M", "R")
    r_96_evo = r2_mod.collapse_oof(
        r2_mod.fit_designs_with_cache(
            p96["target"], b96_evo, d96_evo, fc_96_evo, alphas=r2_mod.ALPHAS_17,
            renorm_val_weights=True, tie_largest_alpha=True,
        ),
        eval_96,
    )

    b94_full: dict[str, np.ndarray] = {**p94["blocks"], "M": m_94[:, None]}
    d94_full: dict[str, tuple[str, ...]] = {"M": ("M",)}
    for rname, ccols in r2_mod.DM_PROGRESSION_RUNGS.items():
        d94_full[f"{rname}|M"] = (*ccols, "M")
    if r_94 is not None:
        b94_full["R"] = r_94
        d94_full["R"] = ("R",)
        ctop = r2_mod.DM_PROGRESSION_RUNGS["nuisance+profile+couplings+gemme"]
        d94_full["nuisance+profile+couplings+gemme|R"] = (*ctop, "R")
        d94_full["nuisance+profile+couplings+gemme|MR"] = (*ctop, "M", "R")
    r_94_full = r2_mod.collapse_oof(
        r2_mod.fit_designs_with_cache(p94["target"], b94_full, d94_full, fc_94_full, alphas=r2_mod.ALPHAS_5),
        eval_94_full,
    )

    record = {
        "schema": r2_mod.ARM_CACHE_SCHEMA,
        "dataset": "domainome",
        "arm_id": arm_id,
        **meta,
        "elapsed_seconds": float(time.monotonic() - t0),
        "zero_shot_96": [float(v) for v in zs_96],
        "zero_shot_94": [float(v) for v in zs_94],
        "evo_94": {ep: {d: v.tolist() for d, v in by_d.items()} for ep, by_d in r_94_evo.items()},
        "evo_96": {ep: {d: v.tolist() for d, v in by_d.items()} for ep, by_d in r_96_evo.items()},
        "full_94": {ep: {d: v.tolist() for d, v in by_d.items()} for ep, by_d in r_94_full.items()},
    }
    atomic_write_json(cache_path, record)
    return record


def _r2_ladder_summaries_17(
    ms_records: Mapping[str, Mapping[str, Any]],
    dm_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build 17-ladder (+ DPLM-2 struct companion) zero-shot and CV ranking/error divergence tables."""
    all_ladders = {**SCALING_LADDERS_18, **COMPANION_STRUCT_LADDERS}
    out: dict[str, Any] = {}
    for lname, arms in all_ladders.items():
        rows = []
        for a in arms:
            meta = EXTENDED_ARM_METADATA[a]
            ms_rec = ms_records.get(a)
            dm_rec = dm_records.get(a)
            if ms_rec is None and dm_rec is None:
                continue
            rows.append(
                {
                    "arm_id": a,
                    "params_m": meta.params_m,
                    "mpnn_noise_angstrom": meta.mpnn_noise_angstrom,
                    "conditioning_regime": meta.conditioning_regime,
                    "megascale_79_zs_spearman": float(np.mean(ms_rec["zero_shot_79"])) if ms_rec else None,
                    "megascale_101_zs_spearman": float(np.mean(ms_rec["zero_shot_101"])) if ms_rec else None,
                    "domainome_94_zs_spearman": float(np.mean(dm_rec["zero_shot_94"])) if dm_rec else None,
                    "megascale_79_evo_C_M_cal_mse": (
                        float(np.mean(ms_rec["evo_79"]["calibrated_error"]["C_M"])) if ms_rec else None
                    ),
                    "megascale_79_evo_C_M_trans_mse": (
                        float(np.mean(ms_rec["evo_79"]["transfer_error"]["C_M"])) if ms_rec else None
                    ),
                    "domainome_94_evo_C_M_cal_mse": (
                        float(np.mean(dm_rec["evo_94"]["calibrated_error"]["C_M"])) if dm_rec else None
                    ),
                    "domainome_94_evo_C_M_trans_mse": (
                        float(np.mean(dm_rec["evo_94"]["transfer_error"]["C_M"])) if dm_rec else None
                    ),
                }
            )
        out[lname] = {
            "declared_arms": list(arms),
            "n_available": len(rows),
            "axis": "noise_angstrom" if lname.startswith("ProteinMPNN-") else "log10_params_m",
            "rungs": rows,
        }
    return out


def _set_worker_threads(threads: int) -> None:
    import torch

    torch.set_num_threads(max(1, int(threads)))


_SPAWN_R2_STATE: dict[str, Any] = {}


def _init_r2_spawn_worker(
    base_str: str,
    r2_dir_str: str,
    new_cdir_str: str,
    threads: int,
) -> None:
    _set_worker_threads(threads)
    base = Path(base_str)
    ms = r2_mod.load_megascale_supports(root=base)
    dm = r2_mod.load_domainome_supports(root=base)
    ms_w_101 = row_weights(ms["panel_101"]["site"], ms["panel_101"]["group"])
    ms_w_79 = row_weights(ms["panel_79"]["site"], ms["panel_79"]["group"])
    dm_w_96 = row_weights(dm["panel_96"]["site"], dm["panel_96"]["group"])
    dm_w_94 = row_weights(dm["panel_94"]["site"], dm["panel_94"]["group"])
    _SPAWN_R2_STATE.clear()
    _SPAWN_R2_STATE.update(
        {
            "base": base,
            "r2_dir": Path(r2_dir_str),
            "new_cdir": Path(new_cdir_str),
            "ms": ms,
            "dm": dm,
            "fc_ms_79_evo": r2_mod.build_fold_cache(groups=ms["panel_79"]["group"], weights=ms_w_79),
            "fc_ms_101_evo": r2_mod.build_fold_cache(groups=ms["panel_101"]["group"], weights=ms_w_101),
            "fc_ms_79_full": r2_mod.build_fold_cache(
                groups=ms["panel_79"]["group"],
                weights=ms_w_79,
                g_builder=lambda held, inner: gate.nuisance_response(
                    ms["panel_79"]["states"], ms["panel_79"]["pair_states"], np.asarray(held), inner, device="cpu"
                )[0],
            ),
            "fc_ms_101_full": r2_mod.build_fold_cache(
                groups=ms["panel_101"]["group"],
                weights=ms_w_101,
                g_builder=lambda held, inner: gate.nuisance_response(
                    ms["panel_101"]["states"], ms["panel_101"]["pair_states"], np.asarray(held), inner, device="cpu"
                )[0],
            ),
            "fc_dm_94_evo": r2_mod.build_fold_cache(groups=dm["panel_94"]["group"], weights=dm_w_94),
            "fc_dm_96_evo": r2_mod.build_fold_cache(groups=dm["panel_96"]["group"], weights=dm_w_96),
            "fc_dm_94_full": r2_mod.build_fold_cache(
                groups=dm["panel_94"]["group"],
                weights=dm["panel_94"]["weights"],
                g_builder=lambda held, inner: r2_mod.abundance_mod.nuisance_response(
                    dm["panel_94"],
                    r2_mod.DM_FIRST_STAGE,
                    dm["panel_94"]["blocks"],
                    np.asarray(held),
                    inner,
                    device="cpu",
                )[0],
            ),
        }
    )


def _r2_spawn_worker(arm_id: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    st = _SPAWN_R2_STATE
    slug = arm_slug(arm_id)
    ms_rec = _fit_new_r2_megascale_arm(
        arm_id,
        st["r2_dir"] / f"r2_stability_arrays_{slug}.npz",
        st["ms"],
        st["fc_ms_79_evo"],
        st["fc_ms_101_evo"],
        st["fc_ms_79_full"],
        st["fc_ms_101_full"],
        st["new_cdir"] / "megascale",
        st["base"],
    )
    dm_rec = _fit_new_r2_domainome_arm(
        arm_id,
        st["r2_dir"] / f"r2_abundance_arrays_{slug}.npz",
        st["dm"],
        st["fc_dm_94_evo"],
        st["fc_dm_96_evo"],
        st["fc_dm_94_full"],
        st["new_cdir"] / "domainome",
        st["base"],
    )
    return arm_id, ms_rec, dm_rec


def assemble_r2_biophysical_77(
    *,
    root: Path = REPO_ROOT,
    scoring_dir: Path | None = None,
    out_dir: Path | None = None,
    workers: int = 8,
    threads: int = 4,
    draws: int = r2_mod.BOOTSTRAP_DRAWS,
    allow_pending: bool = False,
) -> dict[str, Any]:
    """Assemble ``r2_biophysical_unified_77.json`` reusing all 62 cached R2 cells + fitting new arms."""
    t0 = time.monotonic()
    base = Path(root).resolve()
    sdir = Path(scoring_dir) if scoring_dir is not None else (base / DEFAULT_NEW_SCORING_DIR)
    odir = Path(out_dir) if out_dir is not None else (base / UNIFIED_OUTPUT_DIR)
    r2_dir = sdir / "r2"
    old_cdir = base / UNIFIED_OUTPUT_DIR / "cells/r2_exp21"
    new_cdir = base / DEFAULT_NEW_CELL_CACHE_DIR / "r2_exp21"
    (new_cdir / "megascale").mkdir(parents=True, exist_ok=True)
    (new_cdir / "domainome").mkdir(parents=True, exist_ok=True)

    ms_new_paths = {
        a: r2_dir / f"r2_stability_arrays_{arm_slug(a)}.npz"
        for a in NEW_23_SCORED_ENTRIES
        if (r2_dir / f"r2_stability_arrays_{arm_slug(a)}.npz").is_file()
    }
    dm_new_paths = {
        a: r2_dir / f"r2_abundance_arrays_{arm_slug(a)}.npz"
        for a in NEW_23_SCORED_ENTRIES
        if (r2_dir / f"r2_abundance_arrays_{arm_slug(a)}.npz").is_file()
    }
    present_primary = [a for a in NEW_19_PRIMARY_ARMS if a in ms_new_paths and a in dm_new_paths]
    present_struct = [a for a in DPLM2_4_STRUCT_ARMS if a in ms_new_paths and a in dm_new_paths]
    missing_primary = _check_pending_arms(
        NEW_19_PRIMARY_ARMS, present_primary, benchmark="R2 Biophysical Full (81)", allow_pending=allow_pending
    )
    missing_struct = [a for a in DPLM2_4_STRUCT_ARMS if a not in present_struct]

    # Load frozen supports & build fold caches once
    ms = r2_mod.load_megascale_supports(root=base)
    dm = r2_mod.load_domainome_supports(root=base)

    ms_w_101 = row_weights(ms["panel_101"]["site"], ms["panel_101"]["group"])
    ms_w_79 = row_weights(ms["panel_79"]["site"], ms["panel_79"]["group"])
    fc_ms_79_evo = r2_mod.build_fold_cache(groups=ms["panel_79"]["group"], weights=ms_w_79)
    fc_ms_101_evo = r2_mod.build_fold_cache(groups=ms["panel_101"]["group"], weights=ms_w_101)
    fc_ms_79_full = r2_mod.build_fold_cache(
        groups=ms["panel_79"]["group"],
        weights=ms_w_79,
        g_builder=lambda held, inner: gate.nuisance_response(
            ms["panel_79"]["states"], ms["panel_79"]["pair_states"], np.asarray(held), inner, device="cpu"
        )[0],
    )
    fc_ms_101_full = r2_mod.build_fold_cache(
        groups=ms["panel_101"]["group"],
        weights=ms_w_101,
        g_builder=lambda held, inner: gate.nuisance_response(
            ms["panel_101"]["states"], ms["panel_101"]["pair_states"], np.asarray(held), inner, device="cpu"
        )[0],
    )

    dm_w_96 = row_weights(dm["panel_96"]["site"], dm["panel_96"]["group"])
    dm_w_94 = row_weights(dm["panel_94"]["site"], dm["panel_94"]["group"])
    fc_dm_94_evo = r2_mod.build_fold_cache(groups=dm["panel_94"]["group"], weights=dm_w_94)
    fc_dm_96_evo = r2_mod.build_fold_cache(groups=dm["panel_96"]["group"], weights=dm_w_96)
    fc_dm_94_full = r2_mod.build_fold_cache(
        groups=dm["panel_94"]["group"],
        weights=dm["panel_94"]["weights"],
        g_builder=lambda held, inner: r2_mod.abundance_mod.nuisance_response(
            dm["panel_94"],
            r2_mod.DM_FIRST_STAGE,
            dm["panel_94"]["blocks"],
            np.asarray(held),
            inner,
            device="cpu",
        )[0],
    )

    # Reuse all 62 existing cached cells bit-for-bit
    ms_records: dict[str, dict[str, Any]] = {
        a: load_json_file(old_cdir / "megascale" / f"{a}.json") for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
    }
    dm_records: dict[str, dict[str, Any]] = {
        a: load_json_file(old_cdir / "domainome" / f"{a}.json") for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
    }

    # Fit ONLY new arms into isolated cache dir (spawn-parallel across CPU cores; never fork post-OpenMP)
    todo_arms = present_primary + present_struct
    if workers <= 1 or len(todo_arms) <= 1:
        _set_worker_threads(int(threads))
        for idx, arm_id in enumerate(todo_arms, 1):
            ms_records[arm_id] = _fit_new_r2_megascale_arm(
                arm_id, ms_new_paths[arm_id], ms, fc_ms_79_evo, fc_ms_101_evo, fc_ms_79_full, fc_ms_101_full,
                new_cdir / "megascale", base,
            )
            dm_records[arm_id] = _fit_new_r2_domainome_arm(
                arm_id, dm_new_paths[arm_id], dm, fc_dm_94_evo, fc_dm_96_evo, fc_dm_94_full,
                new_cdir / "domainome", base,
            )
            print(f"[r2-full] {idx}/{len(todo_arms)} {arm_id} ({time.monotonic() - t0:.1f}s)", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=min(int(workers), len(todo_arms)),
            mp_context=mp.get_context("spawn"),
            initializer=_init_r2_spawn_worker,
            initargs=(str(base), str(r2_dir), str(new_cdir), int(threads)),
        ) as pool:
            futs = [pool.submit(_r2_spawn_worker, arm_id) for arm_id in todo_arms]
            for idx, fut in enumerate(as_completed(futs), 1):
                arm_id, ms_rec, dm_rec = fut.result()
                ms_records[arm_id] = ms_rec
                dm_records[arm_id] = dm_rec
                print(f"[r2-full] {idx}/{len(todo_arms)} {arm_id} ({time.monotonic() - t0:.1f}s)", flush=True)

    # Baseline C vectors
    eval_ms_79 = lambda p: r2_mod.evaluate_megascale_oof(p, ms["panel_79"]["target"], ms["panel_79"]["group"], ms["panel_79"]["site"], ms["groups_79"])  # noqa: E731
    eval_ms_101 = lambda p: r2_mod.evaluate_megascale_oof(p, ms["panel_101"]["target"], ms["panel_101"]["group"], ms["panel_101"]["site"], ms["groups_101"])  # noqa: E731
    eval_dm_94 = lambda p: r2_mod.evaluate_domainome_oof(  # noqa: E731
        p, dm["panel_94"]["target"], dm["panel_94"]["group"], dm["panel_94"]["domain"], dm["panel_94"]["site"], dm["groups_94"], use_nested_weights=False
    )

    ms_base_79_evo = r2_mod.collapse_oof(
        r2_mod.fit_designs_with_cache(
            ms["panel_79"]["target"],
            {"P": ms["p_logodds_79"][:, None], "POTTS": ms["potts_raw_79"][:, None], "GEMME": ms["gemme_raw_79"][:, None]},
            {"C": ("P", "POTTS", "GEMME")}, fc_ms_79_evo, alphas=r2_mod.ALPHAS_17, renorm_val_weights=True, tie_largest_alpha=True,
        ),
        eval_ms_79,
    )
    ms_base_101_evo = r2_mod.collapse_oof(
        r2_mod.fit_designs_with_cache(
            ms["panel_101"]["target"], {"P": ms["p_logodds_101"][:, None]}, {"C": ("P",)},
            fc_ms_101_evo, alphas=r2_mod.ALPHAS_17, renorm_val_weights=True, tie_largest_alpha=True,
        ),
        eval_ms_101,
    )
    dm_base_94_evo = r2_mod.collapse_oof(
        r2_mod.fit_designs_with_cache(
            dm["panel_94"]["target"],
            {"P": dm["p_logodds_94"][:, None], "POTTS": dm["potts_raw_94"][:, None], "GEMME": dm["gemme_raw_94"][:, None]},
            {"C": ("P", "POTTS", "GEMME")}, fc_dm_94_evo, alphas=r2_mod.ALPHAS_17, renorm_val_weights=True, tie_largest_alpha=True,
        ),
        eval_dm_94,
    )

    arm_order_primary = tuple(list(UNIFIED_62_SINGLE_SEQUENCE_ARMS) + present_primary)
    arm_order_all = tuple(list(arm_order_primary) + present_struct)
    arms_with_R_ms = tuple(a for a in arm_order_all if ms_records[a].get("has_R"))
    arms_with_R_dm = tuple(a for a in arm_order_all if dm_records[a].get("has_R"))

    def _to_np_designs(recs: Mapping[str, Mapping[str, Any]], key: str) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        return {
            a: {ep: {d: np.asarray(v, dtype=np.float64) for d, v in by_d.items()} for ep, by_d in recs[a][key].items()}
            for a in arm_order_all
        }

    # Explicit zero-shot vectors
    _, ms_zs_p_79 = pw_mod.group_spearman(ms["panel_79"]["target"], ms["p_logodds_79"], ms["panel_79"]["group"])
    _, ms_zs_e_79 = pw_mod.group_spearman(ms["panel_79"]["target"], ms["potts_raw_79"], ms["panel_79"]["group"])
    _, ms_zs_g_79 = pw_mod.group_spearman(ms["panel_79"]["target"], ms["gemme_raw_79"], ms["panel_79"]["group"])
    dm_zs_p_94 = eval_dm_94(dm["p_logodds_94"])["ranking"]
    dm_zs_e_94 = eval_dm_94(dm["potts_raw_94"])["ranking"]
    dm_zs_g_94 = eval_dm_94(dm["gemme_raw_94"])["ranking"]

    ms_zs_79_map = {a: np.asarray(ms_records[a]["zero_shot_79"], dtype=np.float64) for a in arm_order_all}
    dm_zs_94_map = {a: np.asarray(dm_records[a]["zero_shot_94"], dtype=np.float64) for a in arm_order_all}

    payload = {
        "schema": "r2_biophysical_unified_77_v1",
        "generated_utc": _now_utc(),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "completion_status": "complete" if not (missing_primary or missing_struct) else "partial_pending_h200",
        "n_arms_primary": len(arm_order_primary),
        "n_arms_with_struct_companions": len(arm_order_all),
        "arms_primary": list(arm_order_primary),
        "arms_with_struct_companions": list(arm_order_all),
        "pending_primary_arms": missing_primary,
        "pending_struct_companions": missing_struct,
        "arm_metadata": {a: extended_arm_meta_dict(a) for a in arm_order_all},
        "datasets": {
            "megascale_79_cc": {
                "zero_shot": r2_mod._build_zero_shot_block(
                    ms_zs_79_map,
                    {"profile": ms_zs_p_79, "potts": ms_zs_e_79, "gemme": ms_zs_g_79},
                    arm_order_all,
                ),
                "primary_evolutionary_control": r2_mod._build_six_design_block(
                    {ep: ms_base_79_evo[ep]["C"] for ep in r2_mod.ENDPOINTS},
                    _to_np_designs(ms_records, "evo_79"),
                    arm_order_all,
                    arms_with_R_ms,
                ),
                "dplm2_struct_minus_seq_zs": _dplm2_struct_minus_seq_paired(ms_zs_79_map, draws=draws),
                "stratified_zs_summary": _conditioning_regime_summary(
                    arm_order_all, {a: float(np.mean(ms_zs_79_map[a])) for a in arm_order_all}
                ),
            },
            "megascale_101_full": {
                "primary_evolutionary_control": r2_mod._build_six_design_block(
                    {ep: ms_base_101_evo[ep]["C"] for ep in r2_mod.ENDPOINTS},
                    _to_np_designs(ms_records, "evo_101"),
                    arm_order_all,
                    arms_with_R_ms,
                ),
            },
            "domainome_94_cc": {
                "zero_shot": r2_mod._build_zero_shot_block(
                    dm_zs_94_map,
                    {"profile": dm_zs_p_94, "potts": dm_zs_e_94, "gemme": dm_zs_g_94},
                    arm_order_all,
                ),
                "primary_evolutionary_control": r2_mod._build_six_design_block(
                    {ep: dm_base_94_evo[ep]["C"] for ep in r2_mod.ENDPOINTS},
                    _to_np_designs(dm_records, "evo_94"),
                    arm_order_all,
                    arms_with_R_dm,
                ),
                "dplm2_struct_minus_seq_zs": _dplm2_struct_minus_seq_paired(dm_zs_94_map, draws=draws),
                "stratified_zs_summary": _conditioning_regime_summary(
                    arm_order_all, {a: float(np.mean(dm_zs_94_map[a])) for a in arm_order_all}
                ),
            },
        },
        "scaling_and_ablation_ladders_18": _r2_ladder_summaries_17(ms_records, dm_records),
    }

    odir.mkdir(parents=True, exist_ok=True)
    out_path = odir / "r2_biophysical_unified_full.json"
    digest = write_verified_json_with_sha256(out_path, payload)
    return {
        "benchmark": "r2",
        "completion_status": payload["completion_status"],
        "n_arms_primary": len(arm_order_primary),
        "n_arms_with_struct": len(arm_order_all),
        "outputs": {out_path.name: digest},
    }


# ============================================================================
# 4. R3 Epistasis (77+4) & E01 Contacts (77+4, Strictly Tagged 3D Oracle vs Seq-Only)
# ============================================================================


def _fit_new_r3_epistasis_arm(
    arm_id: str,
    npz_path: Path,
    ctx: Any,
    rung_caches: dict[str, Any],
    oof_targets: dict[int, np.ndarray],
    u_c_metrics: dict[str, Any],
    yfit_map: dict[int, Any],
    cache_dir: Path,
) -> dict[str, Any]:
    cache_path = cache_dir / f"{arm_slug(arm_id)}.json"
    npz_sha = sha256_file(npz_path)
    if cache_path.is_file():
        cached = load_json_file(cache_path)
        if cached.get("source_sha256") == npz_sha and cached.get("support_sha256") == ctx.support_sha256:
            return cached

    t0 = time.monotonic()
    with np.load(npz_path, allow_pickle=False) as d:
        eps_all = np.asarray(d["cycle_eps_pll"], dtype=np.float64)
        r_all = np.asarray(d["R_cycle"], dtype=np.float64) if "R_cycle" in d.files else None
    idx_cc = ctx.kept_global_indices
    if len(eps_all) <= int(np.max(idx_cc)):
        raise ValueError(f"{arm_id}: cycle_eps_pll length {len(eps_all)} <= max kept index {int(np.max(idx_cc))}")
    m_cc = eps_all[idx_cc][:, None]
    r_cc = r_all[idx_cc] if r_all is not None else None
    if not np.isfinite(m_cc).all() or (r_cc is not None and not np.isfinite(r_cc).all()):
        raise ValueError(f"{arm_id}: non-finite R3 cycle features in {npz_path}")

    zs_raw = r3_mod._zero_shot_per_group_spearman(
        m_cc[:, 0], ctx.panel["epsilon"], ctx.panel["group"], ctx.group_order
    )
    res_target = np.mean([oof_targets[s] for s in pw_mod.SPLIT_SEEDS], axis=0)
    zs_res = r3_mod._zero_shot_per_group_spearman(
        m_cc[:, 0], res_target, ctx.panel["group"], ctx.group_order
    )

    fitted = r3_mod._r3_fit_arm_with_caches(
        ctx, rung_caches, oof_targets, u_c_metrics, yfit_map, m_cc, r_cc
    )
    record = {
        "arm_id": arm_id,
        "source_sha256": npz_sha,
        "support_sha256": ctx.support_sha256,
        "groups": list(ctx.group_order),
        "zs_cc_raw_spearman": [float(v) for v in zs_raw],
        "zs_cc_res_spearman": [float(v) for v in zs_res],
        "M_cycle": fitted["M_cycle"],
        "R_cycle": fitted["R_cycle"],
        "MR_cycle_top_rung": fitted["MR_cycle_top_rung"],
        "seconds": float(time.monotonic() - t0),
    }
    atomic_write_json(cache_path, record)
    return record


_SPAWN_R3_STATE: dict[str, Any] = {}


def _init_r3_spawn_worker(
    base_str: str,
    r3_dir_str: str,
    new_cdir_str: str,
    threads: int,
) -> None:
    _set_worker_threads(threads)
    base = Path(base_str)
    ctx = load_pairwise_cycle_context(base)
    rung_caches, oof_targets, u_c_metrics = r3_mod._r3_build_rung_caches(
        ctx, ctx.panel["blocks"], rungs=r3_mod.R3_RUNGS
    )
    yfit_map = r3_mod._load_yfit_map(ctx)
    _SPAWN_R3_STATE.clear()
    _SPAWN_R3_STATE.update(
        {
            "r3_dir": Path(r3_dir_str),
            "new_cdir": Path(new_cdir_str),
            "ctx": ctx,
            "rung_caches": rung_caches,
            "oof_targets": oof_targets,
            "u_c_metrics": u_c_metrics,
            "yfit_map": yfit_map,
        }
    )


def _r3_spawn_worker(arm_id: str) -> tuple[str, dict[str, Any]]:
    st = _SPAWN_R3_STATE
    rec = _fit_new_r3_epistasis_arm(
        arm_id,
        st["r3_dir"] / f"r3_pairwise_cycles_{arm_slug(arm_id)}.npz",
        st["ctx"],
        st["rung_caches"],
        st["oof_targets"],
        st["u_c_metrics"],
        st["yfit_map"],
        st["new_cdir"],
    )
    return arm_id, rec


def assemble_r3_epistasis_77(
    *,
    root: Path = REPO_ROOT,
    scoring_dir: Path | None = None,
    out_dir: Path | None = None,
    workers: int = 8,
    threads: int = 4,
    allow_pending: bool = False,
) -> dict[str, Any]:
    """Assemble ``r3_epistasis_and_contacts_unified_77.json`` reusing all 62 cached R3 cells."""
    t0 = time.monotonic()
    base = Path(root).resolve()
    sdir = Path(scoring_dir) if scoring_dir is not None else (base / DEFAULT_NEW_SCORING_DIR)
    odir = Path(out_dir) if out_dir is not None else (base / UNIFIED_OUTPUT_DIR)
    r3_dir = sdir / "r3"
    old_cdir = base / UNIFIED_OUTPUT_DIR / "cells/r2_r3/r3_epistasis"
    new_cdir = base / DEFAULT_NEW_CELL_CACHE_DIR / "r3_epistasis"
    new_cdir.mkdir(parents=True, exist_ok=True)

    new_paths = {
        a: r3_dir / f"r3_pairwise_cycles_{arm_slug(a)}.npz"
        for a in NEW_23_SCORED_ENTRIES
        if (r3_dir / f"r3_pairwise_cycles_{arm_slug(a)}.npz").is_file()
    }
    present_primary = [a for a in NEW_19_PRIMARY_ARMS if a in new_paths]
    present_struct = [a for a in DPLM2_4_STRUCT_ARMS if a in new_paths]
    missing_primary = _check_pending_arms(
        NEW_19_PRIMARY_ARMS, present_primary, benchmark="R3 Epistasis Full (81)", allow_pending=allow_pending
    )
    missing_struct = [a for a in DPLM2_4_STRUCT_ARMS if a not in present_struct]

    ctx = load_pairwise_cycle_context(base)
    rung_caches, oof_targets, u_c_metrics = r3_mod._r3_build_rung_caches(
        ctx, ctx.panel["blocks"], rungs=r3_mod.R3_RUNGS
    )
    yfit_map = r3_mod._load_yfit_map(ctx)

    loaded: dict[str, dict[str, Any]] = {
        a: load_json_file(old_cdir / f"{a}.json") for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS
    }
    todo_r3 = present_primary + present_struct
    if workers <= 1 or len(todo_r3) <= 1:
        _set_worker_threads(int(threads))
        for idx, arm_id in enumerate(todo_r3, 1):
            loaded[arm_id] = _fit_new_r3_epistasis_arm(
                arm_id, new_paths[arm_id], ctx, rung_caches, oof_targets, u_c_metrics, yfit_map, new_cdir
            )
            print(f"[r3-full] {idx}/{len(todo_r3)} {arm_id} ({time.monotonic() - t0:.1f}s)", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=min(int(workers), len(todo_r3)),
            mp_context=mp.get_context("spawn"),
            initializer=_init_r3_spawn_worker,
            initargs=(str(base), str(r3_dir), str(new_cdir), int(threads)),
        ) as pool:
            futs = [pool.submit(_r3_spawn_worker, arm_id) for arm_id in todo_r3]
            for idx, fut in enumerate(as_completed(futs), 1):
                arm_id, rec = fut.result()
                loaded[arm_id] = rec
                print(f"[r3-full] {idx}/{len(todo_r3)} {arm_id} ({time.monotonic() - t0:.1f}s)", flush=True)

    arm_order_primary = tuple(list(UNIFIED_62_SINGLE_SEQUENCE_ARMS) + present_primary)
    arm_order_all = tuple(list(arm_order_primary) + present_struct)
    arms_with_R = tuple(a for a in arm_order_all if loaded[a].get("R_cycle") is not None)

    # Zero-shot raw & residualized
    zs_raw_map = {a: np.asarray(loaded[a]["zs_cc_raw_spearman"], dtype=np.float64) for a in arm_order_all}
    zs_res_map = {a: np.asarray(loaded[a]["zs_cc_res_spearman"], dtype=np.float64) for a in arm_order_all}
    potts_zs_raw = r3_mod._zero_shot_per_group_spearman(
        ctx.potts_cycle_raw, ctx.panel["epsilon"], ctx.panel["group"], ctx.group_order
    )

    def _safe_band_dict(vec_map: Mapping[str, np.ndarray], arms_sub: Sequence[str]) -> dict[str, Any]:
        mat = np.column_stack([vec_map[a] for a in arms_sub])
        valid = np.all(np.isfinite(mat), axis=1)
        return r2_mod.paired_bands_summary(mat[valid], list(arms_sub), seed=bidir.SEED)

    # Top-rung MR_cycle contrasts over nuisance+profile+couplings
    mr_top: dict[str, Any] = {}
    for metric_key, to_loss in (("mse", 1.0), ("spearman", -1.0)):
        contrasts_m: dict[str, Any] = {}
        for label, red_k, joint_k in (
            ("MR_given_controls", "U_C", "U_C_MR"),
            ("R_given_controls", "U_C", "U_C_R"),
            ("M_given_controls", "U_C", "U_C_M"),
            ("R_given_controls_and_M", "U_C_M", "U_C_MR"),
            ("controls_given_R", "R", "U_C_R"),
        ):
            cols = [
                to_loss
                * (
                    np.asarray(loaded[a]["MR_cycle_top_rung"][metric_key][red_k], dtype=np.float64)
                    - np.asarray(loaded[a]["MR_cycle_top_rung"][metric_key][joint_k], dtype=np.float64)
                )
                for a in arms_with_R
            ]
            mat = np.column_stack(cols)
            valid = np.all(np.isfinite(mat), axis=1)
            bands = r2_mod.paired_bands_summary(mat[valid], list(arms_with_R), seed=bidir.SEED)
            contrasts_m[label] = {
                "reduced": red_k,
                "joint": joint_k,
                "groups": int(valid.sum()),
                "bands": bands,
                "stratified_summary": _conditioning_regime_summary(
                    arms_with_R, {a: bands["columns"][a]["point"] for a in arms_with_R}
                ),
            }
        mr_top[metric_key] = contrasts_m

    # Attach contacts summary if contacts_unified_full.json already exists
    contacts_full_path = odir / "contacts_unified_full.json"
    contacts_ref = (
        {"path": contacts_full_path.name, "sha256": sha256_file(contacts_full_path)}
        if contacts_full_path.is_file()
        else None
    )

    payload = {
        "schema": "r3_epistasis_and_contacts_unified_full_v1",
        "generated_utc": _now_utc(),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "completion_status": "complete" if not (missing_primary or missing_struct) else "partial_pending_h200",
        "n_arms_primary": len(arm_order_primary),
        "n_arms_with_struct_companions": len(arm_order_all),
        "arms_primary": list(arm_order_primary),
        "arms_with_struct_companions": list(arm_order_all),
        "pending_primary_arms": missing_primary,
        "pending_struct_companions": missing_struct,
        "arm_metadata": {a: extended_arm_meta_dict(a) for a in arm_order_all},
        "support": {
            "name": "complete_case_indel_excluded",
            "support_sha256": ctx.support_sha256,
            "n_groups": len(ctx.group_order),
            "n_cycles": int(len(ctx.panel["epsilon"])),
            "rungs": {k: list(INTERACTION_RUNGS[k]) for k in r3_mod.R3_RUNGS},
        },
        "epistasis": {
            "zero_shot_raw_cc_61": {
                "potts_mean": float(np.nanmean(potts_zs_raw)),
                "arms": _safe_band_dict(zs_raw_map, arm_order_all),
                "dplm2_struct_minus_seq": _dplm2_struct_minus_seq_paired(zs_raw_map, seed=bidir.SEED),
                "stratified_summary": _conditioning_regime_summary(
                    arm_order_all, {a: float(np.nanmean(zs_raw_map[a])) for a in arm_order_all}
                ),
            },
            "zero_shot_res_cc_61": {
                "arms": _safe_band_dict(zs_res_map, arm_order_all),
                "dplm2_struct_minus_seq": _dplm2_struct_minus_seq_paired(zs_res_map, seed=bidir.SEED),
            },
            "MR_cycle_top_rung_over_nuisance_profile_couplings": mr_top,
        },
        "contacts_unified_full_ref": contacts_ref,
    }
    odir.mkdir(parents=True, exist_ok=True)
    out_path = odir / "r3_epistasis_and_contacts_unified_full.json"
    digest = write_verified_json_with_sha256(out_path, payload)
    return {
        "benchmark": "r3_epistasis",
        "completion_status": payload["completion_status"],
        "n_arms_primary": len(arm_order_primary),
        "n_arms_with_struct": len(arm_order_all),
        "outputs": {out_path.name: digest},
    }


def _discover_extended_e01_extractions(
    base: Path,
    scoring_dir: Path,
) -> tuple[dict[str, contacts_mod.DiscoveredExtraction], dict[str, dict[str, Any]]]:
    """Discover all 62 legacy/GP E01 extractions PLUS new DPLM-2/ProteinMPNN extractions."""
    discovered, refused_moe = contacts_mod.discover_e01_extractions(root=base)
    e01_dir = scoring_dir / "e01_contacts"
    if not e01_dir.is_dir():
        return discovered, refused_moe

    slug_to_arm = {arm_slug(a): a for a in NEW_23_SCORED_ENTRIES}
    for child in sorted(e01_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        index_path = child / "position_likelihood.json"
        if not index_path.is_file():
            continue
        index = load_json_file(index_path)
        if index.get("status") not in (None, "complete"):
            continue
        raw_arm = str((index.get("identity") or {}).get("arm") or child.name)
        raw_clean = re.sub(r"^posresp[a-z0-9]*_extract_", "", raw_arm)
        arm_id = raw_clean if raw_clean in EXTENDED_ARM_METADATA else slug_to_arm.get(raw_clean)
        if arm_id is None:
            continue
        paradigm = str((index.get("identity") or {}).get("paradigm", "masked_position"))
        tol, tier, refusal_reason, caveat = contacts_mod._parse_upstream_tolerance(index, paradigm)
        if refusal_reason is not None:
            refused_moe[arm_id] = {
                "arm": arm_id,
                "status": "refused_moe_layout",
                "source": _rel_path(index_path, base),
                "reason": refusal_reason,
            }
            continue
        dtype_val = (index.get("identity") or {}).get("dtype")
        files_map = {str(r["assay"]): dict(r) for r in index.get("assays", ())}
        discovered[arm_id] = contacts_mod.DiscoveredExtraction(
            arm_id=arm_id,
            directory=child,
            paradigm=paradigm,
            dtype_realised=None if dtype_val is None else str(dtype_val),
            upstream_tolerance_nats=tol,
            prefix_tier=tier,
            sub_tier2_caveat=caveat,
            index_sha256=sha256_file(index_path),
            files=files_map,
            source_dir_name=child.name,
        )
    return discovered, refused_moe


def assemble_e01_contacts_77(
    *,
    root: Path = REPO_ROOT,
    scoring_dir: Path | None = None,
    out_dir: Path | None = None,
    workers: int = 8,
    allow_pending: bool = False,
) -> dict[str, Any]:
    """Assemble ``contacts_unified_77.json`` across All / Deep / Shallow strata with explicit
    ``sequence_only`` vs ``backbone_3d_conditioned_oracle`` tagging and separation.
    """
    t0 = time.monotonic()
    base = Path(root).resolve()
    sdir = Path(scoring_dir) if scoring_dir is not None else (base / DEFAULT_NEW_SCORING_DIR)
    odir = Path(out_dir) if out_dir is not None else (base / UNIFIED_OUTPUT_DIR)

    discovered, refused_moe = _discover_extended_e01_extractions(base, sdir)
    present_primary = [a for a in NEW_19_PRIMARY_ARMS if a in discovered]
    present_struct = [a for a in DPLM2_4_STRUCT_ARMS if a in discovered]
    missing_primary = _check_pending_arms(
        NEW_19_PRIMARY_ARMS, present_primary, benchmark="E01 Contacts Full (81)", allow_pending=allow_pending
    )
    missing_struct = [a for a in DPLM2_4_STRUCT_ARMS if a not in discovered]

    # Reuse sealed 62-arm contacts_unified_62.json for the 62 existing arms' per-stratum numbers,
    # and run exact 0-based measure_contact_discrimination + fit_contact_bidirectional on newly
    # extracted arms (plus full combined All/Deep/Shallow evaluation when new extractions exist).
    sealed_62 = load_json_file(base / UNIFIED_OUTPUT_DIR / "contacts_unified_62.json")

    # Compute exact 0-based per-reference AUC-0.5 for every discovered new arm via contacts_mod.arm_pair_responses_validated
    from src.capability.position.contact_response import (
        CONTACT_ANGSTROM,
        MIN_SEQUENCE_SEPARATION,
        admitted_geometry,
        contact_pairs,
    )

    import gzip

    with gzip.open(base / contacts_mod.DEFAULT_STRUCTURES_PATH, "rt", encoding="utf-8") as fh:
        sites_doc = json.load(fh)
    site_rows = sites_doc["sites"] if isinstance(sites_doc, dict) else sites_doc
    sites_by_assay: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in site_rows:
        sites_by_assay[str(row["assay_id"])].append(row)
    coverage_doc = load_json_file(base / contacts_mod.DEFAULT_COVERAGE_PATH)
    cohort_doc = load_json_file(base / contacts_mod.DEFAULT_COHORT_PATH)
    cohort_by_assay = {str(r["assay"]): r for r in cohort_doc["assays"]}

    # Depth lookup from sealed 62-arm discrimination artifact
    disc_62_all = load_json_file(base / UNIFIED_OUTPUT_DIR / "contact_discrimination_unified_62.json")
    depth_by_assay: dict[str, float] = {
        str(k): float(v) for k, v in (disc_62_all.get("depth") or {}).get("realised", {}).items()
    }
    admitted_assays = sorted(depth_by_assay.keys())

    # Precompute geometry & contact pairs once for the 28 admitted references
    geom_cache: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
    for assay in admitted_assays:
        crow = cohort_by_assay[assay]
        geom = admitted_geometry(
            sites_by_assay[assay],
            assay=assay,
            family=crow["cluster"],
            wildtype=crow["wildtype"],
            coverage=coverage_doc,
        )
        pairs = contact_pairs(geom, cutoff=CONTACT_ANGSTROM, min_separation=MIN_SEQUENCE_SEPARATION)
        geom_cache[assay] = (geom, pairs, crow)

    def _score_one_e01_arm(arm_id: str) -> dict[str, Any]:
        ext = discovered[arm_id]
        per_assay_auc: dict[str, float] = {}
        pair_covs: list[float] = []
        for assay in admitted_assays:
            frec = ext.files.get(assay)
            if frec is None:
                continue
            npz_path = ext.directory / "archives" / Path(str(frec["file"])).name
            if not npz_path.is_file():
                continue
            geom, pairs, crow = geom_cache[assay]
            res = contacts_mod.arm_pair_responses_validated(
                npz_path,
                geom,
                pairs,
                crow,
                assay=assay,
                paradigm=ext.paradigm,
                upstream_tolerance=ext.upstream_tolerance_nats,
            )
            vals = res.get("values") or {}
            if not vals:
                continue
            total_eligible = len(pairs["distance"])
            pair_covs.append(len(vals) / max(1, total_eligible))
            keys = sorted(vals.keys())
            s = np.asarray([vals[k] for k in keys], dtype=np.float64)
            y = np.asarray([k in pairs["contacts"] for k in keys], dtype=bool)
            pos = int(y.sum())
            neg = int(y.size - pos)
            if pos > 0 and neg > 0 and np.isfinite(s).all():
                from scipy.stats import rankdata

                r = rankdata(s, method="average")
                auc_m_half = float((r[y].sum() - pos * (pos + 1) / 2.0) / (pos * neg) - 0.5)
                per_assay_auc[assay] = auc_m_half
        return {
            "arm_id": arm_id,
            "conditioning_regime": EXTENDED_ARM_METADATA[arm_id].conditioning_regime,
            "paradigm": EXTENDED_ARM_METADATA[arm_id].paradigm,
            "n_references_scored": len(per_assay_auc),
            "mean_pair_coverage": float(np.mean(pair_covs)) if pair_covs else 0.0,
            "per_assay_auc_minus_half": per_assay_auc,
        }

    new_arm_e01: dict[str, dict[str, Any]] = {}
    for arm_id in present_primary + present_struct:
        new_arm_e01[arm_id] = _score_one_e01_arm(arm_id)

    # Summarize per-stratum (All / Deep / Shallow) AUC-0.5 for new arms + stratified comparison vs sealed 62
    strata_summary: dict[str, Any] = {}
    for stratum in contacts_mod.DEPTH_STRATA:
        skey = str(stratum["key"])
        min_n = float(stratum["min_neff"])
        max_n = stratum["max_neff"]
        s_assays = [
            a
            for a in admitted_assays
            if depth_by_assay[a] >= min_n and (max_n is None or depth_by_assay[a] < float(max_n))
        ]
        per_new: dict[str, Any] = {}
        new_vecs: dict[str, np.ndarray] = {}
        for arm_id, rec in new_arm_e01.items():
            aucs = rec["per_assay_auc_minus_half"]
            if all(a in aucs for a in s_assays) and s_assays:
                vec = np.asarray([aucs[a] for a in s_assays], dtype=np.float64)
                new_vecs[arm_id] = vec
                b = r2_mod.paired_bands_summary(vec[:, None], [arm_id], seed=bidir.SEED)["columns"][arm_id]
                per_new[arm_id] = {
                    **b,
                    "conditioning_regime": rec["conditioning_regime"],
                    "paradigm": rec["paradigm"],
                    "n_references": len(s_assays),
                    "mean_pair_coverage": rec["mean_pair_coverage"],
                }
        strata_summary[skey] = {
            "label": stratum["label"],
            "n_references": len(s_assays),
            "assays": s_assays,
            "new_arms_auc_minus_half": per_new,
            "dplm2_struct_minus_seq_paired": _dplm2_struct_minus_seq_paired(new_vecs, seed=bidir.SEED),
            "stratified_new_arms_summary": _conditioning_regime_summary(
                list(per_new.keys()), {a: per_new[a]["point"] for a in per_new}
            ),
        }

    all_arms_included = list(UNIFIED_62_SINGLE_SEQUENCE_ARMS) + present_primary + present_struct
    payload = {
        "schema": "contacts_unified_full_v1",
        "generated_utc": _now_utc(),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "completion_status": "complete" if not (missing_primary or missing_struct) else "partial_pending_h200",
        "oracle_conditioning_policy": (
            "ProteinMPNN (11 checkpoints) and DPLM-2|struct (4 checkpoints) receive 3D wild-type "
            "backbone coordinates/tokens as direct input and are explicitly tagged "
            "conditioning_regime='backbone_3d_conditioned_oracle'. All sequence-only emergence claims "
            "are strictly restricted to conditioning_regime='sequence_only' (62 existing + 4 DPLM-2 seq-only + 4 DiMA = 70 arms)."
        ),
        "n_arms_primary": len(UNIFIED_62_SINGLE_SEQUENCE_ARMS) + len(present_primary),
        "n_arms_with_struct_companions": len(all_arms_included),
        "sequence_only_arms": [
            a for a in all_arms_included if EXTENDED_ARM_METADATA[a].conditioning_regime == "sequence_only"
        ],
        "backbone_3d_conditioned_oracle_arms": [
            a for a in all_arms_included if EXTENDED_ARM_METADATA[a].conditioning_regime != "sequence_only"
        ],
        "pending_primary_arms": missing_primary,
        "pending_struct_companions": missing_struct,
        "arm_metadata": {a: extended_arm_meta_dict(a) for a in all_arms_included},
        "sealed_62_reference": {
            "path": "contacts_unified_62.json",
            "sha256": sha256_file(base / UNIFIED_OUTPUT_DIR / "contacts_unified_62.json"),
            "explicit_couplings_summary": sealed_62.get("explicit_couplings_summary"),
        },
        "depth_strata_new_arms": strata_summary,
        "refused_moe_layout": refused_moe,
    }
    odir.mkdir(parents=True, exist_ok=True)
    out_path = odir / "contacts_unified_full.json"
    digest = write_verified_json_with_sha256(out_path, payload)
    return {
        "benchmark": "e01_contacts",
        "completion_status": payload["completion_status"],
        "n_new_arms_scored": len(new_arm_e01),
        "outputs": {out_path.name: digest},
    }


# ============================================================================
# 5. R2/R3 Middle-vs-Last Representation (77 + 4 DPLM-2|struct) Assembler
# ============================================================================


def _fit_new_mid_last_arm_megascale(
    arm_id: str,
    npz_path: Path,
    ms: dict[str, Any],
    fc_79_evo: Any,
    cache_dir: Path,
    root: Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    cache_path = cache_dir / f"{arm_slug(arm_id)}.json"
    expected_targets = np.asarray(
        [float(v["ddg"]) for bg in ms["cohort_101"]["backgrounds"] for v in bg["variants"]],
        dtype=np.float64,
    )
    m_101, r_101, meta = _load_new_r2_npz(arm_id, npz_path, expected_targets, root)
    if r_101 is None:
        raise ValueError(f"{arm_id}: missing 1024-d R matrix in {npz_path}")
    if cache_path.is_file():
        cached = load_json_file(cache_path)
        if cached.get("r_source_sha256") == meta["r_source_sha256"]:
            return cached

    p79, g79, mask_79 = ms["panel_79"], ms["groups_79"], ms["mask_79"]
    m_79 = m_101[mask_79]
    r_79 = r_101[mask_79]
    eval_79 = lambda pred: r2_mod.evaluate_megascale_oof(pred, p79["target"], p79["group"], p79["site"], g79)  # noqa: E731
    blocks = {
        "P": ms["p_logodds_79"][:, None],
        "POTTS": ms["potts_raw_79"][:, None],
        "GEMME": ms["gemme_raw_79"][:, None],
        "M": m_79[:, None],
        "R_mid": r_79[:, :512],
        "R_last": r_79[:, 512:1024],
    }
    designs = {
        "C_M": ("P", "POTTS", "GEMME", "M"),
        "R_mid": ("R_mid",),
        "R_last": ("R_last",),
        "R_joint": ("R_mid", "R_last"),
        "C_M_R_mid": ("P", "POTTS", "GEMME", "M", "R_mid"),
        "C_M_R_last": ("P", "POTTS", "GEMME", "M", "R_last"),
        "C_M_R_joint": ("P", "POTTS", "GEMME", "M", "R_mid", "R_last"),
    }
    res = r2_mod.collapse_oof(
        r2_mod.fit_designs_with_cache(
            p79["target"], blocks, designs, fc_79_evo, alphas=r2_mod.ALPHAS_17,
            renorm_val_weights=True, tie_largest_alpha=True, device=device,
        ),
        eval_79,
    )
    record = {
        "schema": mid_mod.CELL_CACHE_SCHEMA,
        "dataset": "megascale",
        "arm_id": arm_id,
        "r_source_sha256": meta["r_source_sha256"],
        "evo_79": {ep: {d: v.tolist() for d, v in by_d.items()} for ep, by_d in res.items()},
    }
    atomic_write_json(cache_path, record)
    return record


def _fit_new_mid_last_arm_domainome(
    arm_id: str,
    npz_path: Path,
    dm: dict[str, Any],
    fc_94_evo: Any,
    cache_dir: Path,
    root: Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    cache_path = cache_dir / f"{arm_slug(arm_id)}.json"
    expected_targets = np.asarray(
        [float(v["target"]) for d in dm["cohort_96"]["domains"] for v in d["variants"]],
        dtype=np.float64,
    )
    m_96, r_96, meta = _load_new_r2_npz(arm_id, npz_path, expected_targets, root)
    if r_96 is None:
        raise ValueError(f"{arm_id}: missing 1024-d R matrix in {npz_path}")
    if cache_path.is_file():
        cached = load_json_file(cache_path)
        if cached.get("r_source_sha256") == meta["r_source_sha256"]:
            return cached

    p94, g94, mask_94 = dm["panel_94"], dm["groups_94"], dm["mask_94"]
    m_94 = m_96[mask_94]
    r_94 = r_96[mask_94]
    eval_94 = lambda pred: r2_mod.evaluate_domainome_oof(  # noqa: E731
        pred, p94["target"], p94["group"], p94["domain"], p94["site"], g94, use_nested_weights=False
    )
    blocks = {
        "P": dm["p_logodds_94"][:, None],
        "POTTS": dm["potts_raw_94"][:, None],
        "GEMME": dm["gemme_raw_94"][:, None],
        "M": m_94[:, None],
        "R_mid": r_94[:, :512],
        "R_last": r_94[:, 512:1024],
    }
    designs = {
        "C_M": ("P", "POTTS", "GEMME", "M"),
        "R_mid": ("R_mid",),
        "R_last": ("R_last",),
        "R_joint": ("R_mid", "R_last"),
        "C_M_R_mid": ("P", "POTTS", "GEMME", "M", "R_mid"),
        "C_M_R_last": ("P", "POTTS", "GEMME", "M", "R_last"),
        "C_M_R_joint": ("P", "POTTS", "GEMME", "M", "R_mid", "R_last"),
    }
    res = r2_mod.collapse_oof(
        r2_mod.fit_designs_with_cache(
            p94["target"], blocks, designs, fc_94_evo, alphas=r2_mod.ALPHAS_17,
            renorm_val_weights=True, tie_largest_alpha=True, device=device,
        ),
        eval_94,
    )
    record = {
        "schema": mid_mod.CELL_CACHE_SCHEMA,
        "dataset": "domainome",
        "arm_id": arm_id,
        "r_source_sha256": meta["r_source_sha256"],
        "evo_94": {ep: {d: v.tolist() for d, v in by_d.items()} for ep, by_d in res.items()},
    }
    atomic_write_json(cache_path, record)
    return record


def _fit_new_mid_last_arm_r3(
    arm_id: str,
    npz_path: Path,
    ctx: Any,
    rung_caches: dict[str, Any],
    oof_targets: dict[int, np.ndarray],
    u_c_metrics: dict[str, Any],
    yfit_map: dict[int, Any],
    cache_dir: Path,
) -> dict[str, Any]:
    cache_path = cache_dir / f"{arm_slug(arm_id)}.json"
    npz_sha = sha256_file(npz_path)
    if cache_path.is_file():
        cached = load_json_file(cache_path)
        if cached.get("source_sha256") == npz_sha:
            return cached

    with np.load(npz_path, allow_pickle=False) as d:
        eps_all = np.asarray(d["cycle_eps_pll"], dtype=np.float64)
        r_all = np.asarray(d["R_cycle"], dtype=np.float64)
    idx_cc = ctx.kept_global_indices
    m_cc = eps_all[idx_cc][:, None]
    r_cc = r_all[idx_cc]

    fit_mid = r3_mod._r3_fit_arm_with_caches(ctx, rung_caches, oof_targets, u_c_metrics, yfit_map, m_cc, r_cc[:, :512])
    fit_last = r3_mod._r3_fit_arm_with_caches(ctx, rung_caches, oof_targets, u_c_metrics, yfit_map, m_cc, r_cc[:, 512:1024])
    fit_joint = r3_mod._r3_fit_arm_with_caches(ctx, rung_caches, oof_targets, u_c_metrics, yfit_map, m_cc, r_cc)
    record = {
        "schema": mid_mod.CELL_CACHE_SCHEMA,
        "dataset": "r3_epistasis",
        "arm_id": arm_id,
        "source_sha256": npz_sha,
        "R_mid": fit_mid["MR_cycle_top_rung"],
        "R_last": fit_last["MR_cycle_top_rung"],
        "R_joint": fit_joint["MR_cycle_top_rung"],
    }
    atomic_write_json(cache_path, record)
    return record


_SPAWN_MID_STATE: dict[str, Any] = {}


def _init_mid_spawn_worker(
    base_str: str,
    r2_dir_str: str,
    r3_dir_str: str,
    new_cdir_str: str,
    threads: int,
) -> None:
    _set_worker_threads(threads)
    base = Path(base_str)
    ms = r2_mod.load_megascale_supports(root=base)
    dm = r2_mod.load_domainome_supports(root=base)
    fc_ms_79 = r2_mod.build_fold_cache(
        groups=ms["panel_79"]["group"], weights=row_weights(ms["panel_79"]["site"], ms["panel_79"]["group"])
    )
    fc_dm_94 = r2_mod.build_fold_cache(
        groups=dm["panel_94"]["group"], weights=row_weights(dm["panel_94"]["site"], dm["panel_94"]["group"])
    )
    r3_ctx = load_pairwise_cycle_context(base)
    r3_caches, r3_oof, r3_uc = r3_mod._r3_build_rung_caches(
        r3_ctx, r3_ctx.panel["blocks"], rungs=r3_mod.R3_RUNGS
    )
    r3_yfit = r3_mod._load_yfit_map(r3_ctx)
    _SPAWN_MID_STATE.clear()
    _SPAWN_MID_STATE.update(
        {
            "base": base,
            "r2_dir": Path(r2_dir_str),
            "r3_dir": Path(r3_dir_str),
            "new_cdir": Path(new_cdir_str),
            "ms": ms,
            "dm": dm,
            "fc_ms_79": fc_ms_79,
            "fc_dm_94": fc_dm_94,
            "r3_ctx": r3_ctx,
            "r3_caches": r3_caches,
            "r3_oof": r3_oof,
            "r3_uc": r3_uc,
            "r3_yfit": r3_yfit,
        }
    )


def _mid_last_spawn_worker(arm_id: str) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    st = _SPAWN_MID_STATE
    slug = arm_slug(arm_id)
    ms_rec = _fit_new_mid_last_arm_megascale(
        arm_id, st["r2_dir"] / f"r2_stability_arrays_{slug}.npz", st["ms"], st["fc_ms_79"], st["new_cdir"] / "megascale", st["base"]
    )
    dm_rec = _fit_new_mid_last_arm_domainome(
        arm_id, st["r2_dir"] / f"r2_abundance_arrays_{slug}.npz", st["dm"], st["fc_dm_94"], st["new_cdir"] / "domainome", st["base"]
    )
    r3_rec = _fit_new_mid_last_arm_r3(
        arm_id,
        st["r3_dir"] / f"r3_pairwise_cycles_{slug}.npz",
        st["r3_ctx"],
        st["r3_caches"],
        st["r3_oof"],
        st["r3_uc"],
        st["r3_yfit"],
        st["new_cdir"] / "r3_epistasis",
    )
    return arm_id, ms_rec, dm_rec, r3_rec


def assemble_r2_r3_mid_vs_last_77(
    *,
    root: Path = REPO_ROOT,
    scoring_dir: Path | None = None,
    out_dir: Path | None = None,
    workers: int = 8,
    threads: int = 4,
    draws: int = mid_mod.BOOTSTRAP_DRAWS,
    allow_pending: bool = False,
) -> dict[str, Any]:
    """Assemble ``r2_r3_middle_vs_last_representation_77.json`` reusing 62-arm mid/last cells."""
    t0 = time.monotonic()
    base = Path(root).resolve()
    sdir = Path(scoring_dir) if scoring_dir is not None else (base / DEFAULT_NEW_SCORING_DIR)
    odir = Path(out_dir) if out_dir is not None else (base / UNIFIED_OUTPUT_DIR)
    old_cdir = base / UNIFIED_OUTPUT_DIR / "cells/r2_r3_mid_vs_last"
    new_cdir = base / DEFAULT_NEW_CELL_CACHE_DIR / "r2_r3_mid_vs_last"
    for sub in ("megascale", "domainome", "r3_epistasis"):
        (new_cdir / sub).mkdir(parents=True, exist_ok=True)

    r2_dir, r3_dir = sdir / "r2", sdir / "r3"
    present_new = [
        a
        for a in NEW_23_SCORED_ENTRIES
        if (r2_dir / f"r2_stability_arrays_{arm_slug(a)}.npz").is_file()
        and (r2_dir / f"r2_abundance_arrays_{arm_slug(a)}.npz").is_file()
        and (r3_dir / f"r3_pairwise_cycles_{arm_slug(a)}.npz").is_file()
    ]
    present_primary = [a for a in NEW_19_PRIMARY_ARMS if a in present_new]
    present_struct = [a for a in DPLM2_4_STRUCT_ARMS if a in present_new]
    missing_primary = _check_pending_arms(
        NEW_19_PRIMARY_ARMS, present_primary, benchmark="R2/R3 Mid-vs-Last Full (81)", allow_pending=allow_pending
    )
    missing_struct = [a for a in DPLM2_4_STRUCT_ARMS if a not in present_struct]

    ms = r2_mod.load_megascale_supports(root=base)
    dm = r2_mod.load_domainome_supports(root=base)
    fc_ms_79 = r2_mod.build_fold_cache(
        groups=ms["panel_79"]["group"], weights=row_weights(ms["panel_79"]["site"], ms["panel_79"]["group"])
    )
    fc_dm_94 = r2_mod.build_fold_cache(
        groups=dm["panel_94"]["group"], weights=row_weights(dm["panel_94"]["site"], dm["panel_94"]["group"])
    )
    r3_ctx = load_pairwise_cycle_context(base)
    r3_caches, r3_oof, r3_uc = r3_mod._r3_build_rung_caches(
        r3_ctx, r3_ctx.panel["blocks"], rungs=r3_mod.R3_RUNGS
    )
    r3_yfit = r3_mod._load_yfit_map(r3_ctx)

    # Load existing 62-arm mid/last cells if available
    ms_recs: dict[str, dict[str, Any]] = {}
    dm_recs: dict[str, dict[str, Any]] = {}
    r3_recs: dict[str, dict[str, Any]] = {}
    missing_62_base: list[str] = []
    for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS:
        p_ms = old_cdir / "megascale" / f"{a}.json"
        p_dm = old_cdir / "domainome" / f"{a}.json"
        p_r3 = old_cdir / "r3_epistasis" / f"{a}.json"
        if p_ms.is_file() and p_dm.is_file() and p_r3.is_file():
            ms_recs[a] = load_json_file(p_ms)
            dm_recs[a] = load_json_file(p_dm)
            r3_recs[a] = load_json_file(p_r3)
        else:
            missing_62_base.append(a)
    if missing_62_base and not allow_pending:
        raise FileNotFoundError(
            f"R2/R3 Mid-vs-Last 77: {len(missing_62_base)} base 62-arm cells still pending in {old_cdir}"
        )

    if workers <= 1 or len(present_new) <= 1:
        _set_worker_threads(int(threads))
        for idx, a in enumerate(present_new, 1):
            ms_recs[a] = _fit_new_mid_last_arm_megascale(
                a, r2_dir / f"r2_stability_arrays_{arm_slug(a)}.npz", ms, fc_ms_79, new_cdir / "megascale", base
            )
            dm_recs[a] = _fit_new_mid_last_arm_domainome(
                a, r2_dir / f"r2_abundance_arrays_{arm_slug(a)}.npz", dm, fc_dm_94, new_cdir / "domainome", base
            )
            r3_recs[a] = _fit_new_mid_last_arm_r3(
                a, r3_dir / f"r3_pairwise_cycles_{arm_slug(a)}.npz", r3_ctx, r3_caches, r3_oof, r3_uc, r3_yfit,
                new_cdir / "r3_epistasis",
            )
            print(f"[mid-last-full] {idx}/{len(present_new)} {a} ({time.monotonic() - t0:.1f}s)", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=min(int(workers), len(present_new)),
            mp_context=mp.get_context("spawn"),
            initializer=_init_mid_spawn_worker,
            initargs=(str(base), str(r2_dir), str(r3_dir), str(new_cdir), int(threads)),
        ) as pool:
            futs = [pool.submit(_mid_last_spawn_worker, a) for a in present_new]
            for idx, fut in enumerate(as_completed(futs), 1):
                a, ms_rec, dm_rec, r3_rec = fut.result()
                ms_recs[a] = ms_rec
                dm_recs[a] = dm_rec
                r3_recs[a] = r3_rec
                print(f"[mid-last-full] {idx}/{len(present_new)} {a} ({time.monotonic() - t0:.1f}s)", flush=True)

    avail_arms = [a for a in (list(UNIFIED_62_SINGLE_SEQUENCE_ARMS) + present_new) if a in ms_recs]

    def _summarize_r2_mid_last(recs: Mapping[str, Mapping[str, Any]], key: str) -> dict[str, Any]:
        if not avail_arms:
            return {"n_arms": 0}
        out_ep: dict[str, Any] = {}
        for ep in r2_mod.ENDPOINTS:
            sign = 1.0 if ep == "ranking" else -1.0
            diff_alone: dict[str, np.ndarray] = {}
            diff_cm: dict[str, np.ndarray] = {}
            for a in avail_arms:
                d = recs[a][key][ep]
                diff_alone[a] = sign * (np.asarray(d["R_mid"], dtype=np.float64) - np.asarray(d["R_last"], dtype=np.float64))
                diff_cm[a] = sign * (
                    np.asarray(d["C_M_R_mid"], dtype=np.float64) - np.asarray(d["C_M_R_last"], dtype=np.float64)
                )
            b_alone = r2_mod._paired_bands_defined(r2_mod._stack_cols(diff_alone, avail_arms), avail_arms)
            b_cm = r2_mod._paired_bands_defined(r2_mod._stack_cols(diff_cm, avail_arms), avail_arms)
            out_ep[ep] = {
                "R_mid_minus_R_last_alone": b_alone,
                "R_mid_minus_R_last_given_CM": b_cm,
                "stratified_given_CM": _conditioning_regime_summary(
                    avail_arms, {a: b_cm["columns"][a]["point"] for a in avail_arms}
                ),
            }
        return {"n_arms": len(avail_arms), "endpoints": out_ep}

    def _summarize_r3_mid_last() -> dict[str, Any]:
        if not avail_arms:
            return {"n_arms": 0}
        out_m: dict[str, Any] = {}
        for mk, sign in (("mse", -1.0), ("spearman", 1.0)):
            diff_alone: dict[str, np.ndarray] = {}
            diff_cm: dict[str, np.ndarray] = {}
            for a in avail_arms:
                rec = r3_recs[a]
                # Support both @exp-architect's 62-arm schema ("variants" -> "mid"/"last") and 23-arm schema ("R_mid"/"R_last")
                if "R_mid" in rec and mk in rec["R_mid"]:
                    a_mid = np.asarray(rec["R_mid"][mk]["R"], dtype=np.float64)
                    a_last = np.asarray(rec["R_last"][mk]["R"], dtype=np.float64)
                    v_mid = np.asarray(rec["R_mid"][mk]["U_C_MR"], dtype=np.float64)
                    v_last = np.asarray(rec["R_last"][mk]["U_C_MR"], dtype=np.float64)
                elif "variants" in rec:
                    a_mid = np.asarray(rec["variants"]["mid"][f"R_alone_{mk}"], dtype=np.float64)
                    a_last = np.asarray(rec["variants"]["last"][f"R_alone_{mk}"], dtype=np.float64)
                    v_mid = np.asarray(rec["variants"]["mid"][f"U_C_MR_top_{mk}"], dtype=np.float64)
                    v_last = np.asarray(rec["variants"]["last"][f"U_C_MR_top_{mk}"], dtype=np.float64)
                else:
                    block = rec.get("top_rung") or rec
                    a_mid = np.asarray(block[mk]["R_mid"], dtype=np.float64)
                    a_last = np.asarray(block[mk]["R_last"], dtype=np.float64)
                    v_mid = np.asarray(block[mk]["U_C_M_R_mid"], dtype=np.float64)
                    v_last = np.asarray(block[mk]["U_C_M_R_last"], dtype=np.float64)
                diff_alone[a] = sign * (a_mid - a_last)
                diff_cm[a] = sign * (v_mid - v_last)
            b_alone = r2_mod._paired_bands_defined(r2_mod._stack_cols(diff_alone, avail_arms), avail_arms)
            b = r2_mod._paired_bands_defined(r2_mod._stack_cols(diff_cm, avail_arms), avail_arms)
            out_m[mk] = {
                "R_mid_minus_R_last_alone": b_alone,
                "R_mid_minus_R_last_given_CM": b,
                "stratified_alone": _conditioning_regime_summary(
                    avail_arms, {a: b_alone["columns"][a]["point"] for a in avail_arms}
                ),
                "stratified_given_CM": _conditioning_regime_summary(
                    avail_arms, {a: b["columns"][a]["point"] for a in avail_arms}
                ),
            }
        return {"n_arms": len(avail_arms), "metrics": out_m}

    payload = {
        "schema": "r2_r3_middle_vs_last_representation_full_v1",
        "generated_utc": _now_utc(),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "completion_status": (
            "complete" if not (missing_primary or missing_struct or missing_62_base) else "partial_pending"
        ),
        "n_arms_available": len(avail_arms),
        "arms": avail_arms,
        "pending_base_62_arms": missing_62_base,
        "pending_primary_arms": missing_primary,
        "pending_struct_companions": missing_struct,
        "arm_metadata": {a: extended_arm_meta_dict(a) for a in avail_arms},
        "sign_convention": "positive (> 0) always means R_mid beats R_last (lower MSE or higher Spearman)",
        "megascale_evo_79": _summarize_r2_mid_last(ms_recs, "evo_79"),
        "domainome_evo_94": _summarize_r2_mid_last(dm_recs, "evo_94"),
        "r3_epistasis_top_rung": _summarize_r3_mid_last(),
    }
    odir.mkdir(parents=True, exist_ok=True)
    out_path = odir / "r2_r3_middle_vs_last_representation_full.json"
    digest = write_verified_json_with_sha256(out_path, payload)
    return {
        "benchmark": "mid_vs_last",
        "completion_status": payload["completion_status"],
        "n_arms_available": len(avail_arms),
        "outputs": {out_path.name: digest},
    }


# ============================================================================
# 6. Result 5 Main Table — Unified 80-Model R4 Homologous Context Scoring
# ============================================================================


def _load_new_r4_main_arm(arm_id: str, path: Path, root: Path) -> dict[str, Any]:
    doc = load_json_file(path)
    if doc.get("schema") != r4_mod.SCHEMA_ARM_RUNNER:
        raise ValueError(f"{path}: unexpected R4 schema {doc.get('schema')!r}")
    units_raw = doc.get("units") or []
    if len(units_raw) != 800:
        raise ValueError(f"{path}: expected 800 retained units, got {len(units_raw)}")
    primary_units: list[dict[str, Any]] = []
    for u in units_raw:
        conds = u["conditions"]
        if "pll_8fold" in u:
            p8 = u["pll_8fold"]
            primary_units.append(
                {
                    "key": str(u["key"]),
                    "band": str(u["band"]),
                    "stratum": "retained",
                    "group": int(u["group"]),
                    "k_homologue": int(conds["homologue"]["k_items"]),
                    "nll_homologue": float(conds["homologue"]["pll_8fold"]["nll_per_residue"]),
                    "nll_unrelated": float(conds["unrelated"]["pll_8fold"]["nll_per_residue"]),
                    "nll_mono_shuffled": float(conds["mono_shuffled"]["pll_8fold"]["nll_per_residue"]),
                    "nll_position_only": float(conds["position_only"]["pll_8fold"]["nll_per_residue"]),
                    "nll_no_context": float(conds["no_context"]["pll_8fold"]["nll_per_residue"]),
                    "hom_beats_unrelated": float(p8["hom_beats_unrelated"]),
                    "hom_beats_mono_shuffled": float(p8["hom_beats_mono_shuffled"]),
                    "fractional_reduction_vs_unrelated": p8.get("fractional_reduction_vs_unrelated"),
                    "fractional_reduction_vs_shuffled": p8.get("fractional_reduction_vs_shuffled"),
                }
            )
        else:
            primary_units.append(
                {
                    "key": str(u["key"]),
                    "band": str(u["band"]),
                    "stratum": "retained",
                    "group": int(u["group"]),
                    "k_homologue": int(conds["homologue"]["k_items"]),
                    "nll_homologue": float(conds["homologue"]["nll_per_residue"]),
                    "nll_unrelated": float(conds["unrelated"]["nll_per_residue"]),
                    "nll_mono_shuffled": float(conds["mono_shuffled"]["nll_per_residue"]),
                    "nll_position_only": float(conds["position_only"]["nll_per_residue"]),
                    "nll_no_context": float(conds["no_context"]["nll_per_residue"]),
                    "hom_beats_unrelated": float(u["hom_beats_unrelated"]),
                    "hom_beats_mono_shuffled": float(u["hom_beats_mono_shuffled"]),
                    "fractional_reduction_vs_unrelated": u.get("fractional_reduction_vs_unrelated"),
                    "fractional_reduction_vs_shuffled": u.get("fractional_reduction_vs_shuffled"),
                }
            )
    meta = EXTENDED_ARM_METADATA[arm_id]
    return {
        "arm_id": arm_id,
        "source_group": "dplm2_mpnn_80_expansion",
        "context_paradigm": (
            "inverse_folding_2chain_spatial_offset"
            if meta.paradigm == "inverse_folding_gnn"
            else "single_chain_bidirectional_unaligned"
        ),
        "conditioning_regime": meta.conditioning_regime,
        "input_alignment_format": (
            "two_chain_spatial_offset_3d"
            if meta.paradigm == "inverse_folding_gnn"
            else "unaligned_concat"
        ),
        "primary_estimand": str(doc.get("primary_estimand", "pll_8fold")),
        "budget_tokens": int(doc.get("budget_tokens", 1024)),
        "aligned_msa": False,
        "dtype_realised": str((doc.get("provenance") or {}).get("dtype_realised", "unknown")),
        "source_path": _rel_path(path, root),
        "source_sha256": sha256_file(path),
        "units": primary_units,
    }


def assemble_r4_scoring_unified_80(
    *,
    root: Path = REPO_ROOT,
    scoring_dir: Path | None = None,
    out_dir: Path | None = None,
    draws: int = 10000,
    seed: int = 20261106,
    allow_pending: bool = False,
) -> dict[str, Any]:
    """Assemble ``r4_scoring_unified_80.json`` (65 existing + 15 new + 4 ``dplm2|struct``)."""
    t0 = time.monotonic()
    base = Path(root).resolve()
    sdir = Path(scoring_dir) if scoring_dir is not None else (base / DEFAULT_NEW_SCORING_DIR)
    odir = Path(out_dir) if out_dir is not None else (base / UNIFIED_OUTPUT_DIR)
    r4_work_65 = base / r4_mod.DEFAULT_WORK_SUBDIR
    r4_new_dir = sdir / "r4_main"

    # Load all 65 existing arms via r4_scoring_yardstick
    early_records: dict[str, tuple[Mapping[str, Any], str]] = {}
    for rel in (
        "results/R5/context_homologue_20260826/context_homologue.json",
        "results/R5/context_homologue_expansion_20260919/context_homologue.json",
        "results/R5/context_homologue_progen3_20260924/context_homologue.json",
    ):
        fpath = base / rel
        if fpath.is_file():
            doc = load_json_file(fpath)
            for arm_id, rec in (doc.get("arms") or {}).items():
                if arm_id in r4_mod.CACHED_EARLY_PROTEIN_ARMS:
                    early_records[arm_id] = (rec, rel)

    loaded: dict[str, dict[str, Any]] = {}
    for arm_id in r4_mod.CACHED_EARLY_PROTEIN_ARMS:
        loaded[arm_id] = r4_mod._load_early_protein_arm(arm_id, early_records)
    for arm_id in r4_mod.CACHED_GP_SINGLE_ARMS:
        loaded[arm_id] = r4_mod._load_gp_cached_arm(arm_id, base, context_paradigm="single_chain_causal")
    for arm_id in r4_mod.CACHED_MSA_ARMS:
        loaded[arm_id] = r4_mod._load_gp_cached_arm(arm_id, base, context_paradigm="msa_aligned")
    for arm_id in r4_mod.NEW_SCORE_ARMS:
        rec = r4_mod._load_new_r4_arm(arm_id, r4_work_65, base)
        if rec is not None:
            loaded[arm_id] = rec

    # Load all new DPLM-2 + ProteinMPNN + DiMA arms
    present_new: list[str] = []
    for arm_id in NEW_23_SCORED_ENTRIES:
        p = r4_new_dir / f"r4_unaligned_{arm_slug(arm_id)}.json"
        if p.is_file():
            loaded[arm_id] = _load_new_r4_main_arm(arm_id, p, base)
            present_new.append(arm_id)

    present_primary = [a for a in NEW_19_PRIMARY_ARMS if a in present_new]
    present_struct = [a for a in DPLM2_4_STRUCT_ARMS if a in present_new]
    missing_primary = _check_pending_arms(
        NEW_19_PRIMARY_ARMS, present_primary, benchmark="R4 Main Table Full (84)", allow_pending=allow_pending
    )
    missing_struct = [a for a in DPLM2_4_STRUCT_ARMS if a not in present_struct]

    arms_order = [a for a in (*UNIFIED_65_ARMS, *present_primary, *present_struct) if a in loaded]

    # Band pooled & per-band hom_beats_unrelated, hom_beats_mono_shuffled, and NLL reduction
    metrics_out: dict[str, Any] = {}
    for band_label, band_filter in (("all_800", None), *((b, b) for b in r4_mod.IDENTITY_BANDS)):
        band_res: dict[str, Any] = {}
        for mkey, null_shift in (
            ("hom_beats_unrelated", 0.5),
            ("hom_beats_mono_shuffled", 0.5),
            ("delta_nll_unrel_minus_hom", 0.0),
        ):
            if mkey == "delta_nll_unrel_minus_hom":
                # Build cluster matrix of (nll_unrelated - nll_homologue) directly
                g_sets = [
                    {int(u["group"]) for u in loaded[a]["units"] if band_filter is None or u["band"] == band_filter}
                    for a in arms_order
                ]
                common_g = sorted(set.intersection(*g_sets))
                mat = np.empty((len(common_g), len(arms_order)), dtype=np.float64)
                for j, a in enumerate(arms_order):
                    by_g: dict[int, list[float]] = defaultdict(list)
                    for u in loaded[a]["units"]:
                        if band_filter is not None and u["band"] != band_filter:
                            continue
                        by_g[int(u["group"])].append(float(u["nll_unrelated"] - u["nll_homologue"]))
                    for i, g in enumerate(common_g):
                        mat[i, j] = float(np.mean(by_g[g]))
            else:
                common_g, mat = r4_mod._cluster_matrix(
                    arms_order, loaded, band=band_filter, metric_key=mkey, null_shift=null_shift
                )
            bands = r2_mod.paired_bands_summary(mat, arms_order, draws=draws, seed=seed)
            band_res[mkey] = {
                "n_clusters": len(common_g),
                "null_shift": null_shift,
                "bands": bands,
                "stratified_summary": _conditioning_regime_summary(
                    arms_order, {a: bands["columns"][a]["point"] for a in arms_order}
                ),
            }
        metrics_out[band_label] = band_res

    per_arm_rows: list[dict[str, Any]] = []
    for a in arms_order:
        rec = loaded[a]
        all_hbu = metrics_out["all_800"]["hom_beats_unrelated"]["bands"]["columns"][a]
        all_hbs = metrics_out["all_800"]["hom_beats_mono_shuffled"]["bands"]["columns"][a]
        all_dnll = metrics_out["all_800"]["delta_nll_unrel_minus_hom"]["bands"]["columns"][a]
        per_arm_rows.append(
            {
                **extended_arm_meta_dict(a),
                "context_paradigm": rec.get("context_paradigm"),
                "input_alignment_format": (
                    "explicit_column_aligned"
                    if a == "msa-transformer-100m"
                    else (
                        "two_chain_spatial_offset_3d"
                        if EXTENDED_ARM_METADATA[a].paradigm == "inverse_folding_gnn"
                        else rec.get("input_alignment_format", "unaligned_concat")
                    )
                ),
                "primary_estimand": rec.get("primary_estimand"),
                "budget_tokens": rec.get("budget_tokens", 1024),
                "dtype_realised": rec.get("dtype_realised"),
                "source_path": rec.get("source_path"),
                "hom_beats_unrelated_rate": float(all_hbu["point"] + 0.5),
                "hom_beats_unrelated_excess_ci95_studentized": all_hbu["studentized_maxT_95"],
                "hom_beats_unrelated_verdict_studentized": all_hbu["verdict_studentized"],
                "hom_beats_mono_shuffled_rate": float(all_hbs["point"] + 0.5),
                "delta_nll_unrel_minus_hom_nats": float(all_dnll["point"]),
                "delta_nll_ci95_studentized": all_dnll["studentized_maxT_95"],
            }
        )

    payload = {
        "schema": "unified_r4_scoring_full_v1",
        "generated_utc": _now_utc(),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "completion_status": "complete" if not (missing_primary or missing_struct) else "partial_pending_h200",
        "n_arms_primary": len(UNIFIED_65_ARMS) + len(present_primary),
        "n_arms_total_rows": len(arms_order),
        "arms_order": arms_order,
        "pending_primary_arms": missing_primary,
        "pending_struct_companions": missing_struct,
        "arm_metadata": {a: extended_arm_meta_dict(a) for a in arms_order},
        "horizontal_table": per_arm_rows,
        "metrics_by_band": metrics_out,
    }
    odir.mkdir(parents=True, exist_ok=True)
    out_path = odir / "r4_scoring_unified_full.json"
    digest = write_verified_json_with_sha256(out_path, payload)
    return {
        "benchmark": "r4_main",
        "completion_status": payload["completion_status"],
        "n_arms_total_rows": len(arms_order),
        "outputs": {out_path.name: digest},
    }


# ============================================================================
# 7. Result 5 Downstream 1 — Unified 80-Model DMS Homolog-Context Ranking
# ============================================================================


def assemble_r5_dms_context_unified_80(
    *,
    root: Path = REPO_ROOT,
    scoring_dir: Path | None = None,
    out_dir: Path | None = None,
    draws: int = 10000,
    seed: int = 20261106,
    allow_pending: bool = False,
) -> dict[str, Any]:
    """Assemble ``r5_dms_context_ranking_unified_80.json`` extending the 68-entry DMS context
    yardstick with all 15 new models (+ 4 ``dplm2|struct`` companions = 87 entries total).
    """
    t0 = time.monotonic()
    base = Path(root).resolve()
    sdir = Path(scoring_dir) if scoring_dir is not None else (base / DEFAULT_NEW_SCORING_DIR)
    odir = Path(out_dir) if out_dir is not None else (base / UNIFIED_OUTPUT_DIR)
    new_scores_dir = sdir / "r5_dms"

    rescue_cohort = load_json_file(base / r5_dms_mod.RESCUE_COHORT_REL)
    e09_cohort = load_json_file(base / r5_dms_mod.E09_COHORT_REL)

    # Load all available base 68 entries (from r5_dms_context_scores + cached legacy paths)
    base_1a: dict[str, dict[str, Any]] = {}
    base_1b: dict[str, dict[str, Any]] = {}
    for spec in r5_dms_mod.DMS_CONTEXT_SPECS_ALL:
        doc1a, _ = r5_dms_mod._resolve_subtest_score_doc(spec, "rescue_30_80", base)
        if doc1a is not None:
            base_1a[spec.entry_key] = doc1a
        doc1b, _ = r5_dms_mod._resolve_subtest_score_doc(spec, "e09_five_band", base)
        if doc1b is not None:
            base_1b[spec.entry_key] = doc1b

    # Load new 23 DPLM-2 / ProteinMPNN / DiMA entries
    new_1a: dict[str, dict[str, Any]] = {}
    new_1b: dict[str, dict[str, Any]] = {}
    for arm_id in NEW_23_SCORED_ENTRIES:
        slug = arm_slug(arm_id)
        p1a = new_scores_dir / "rescue_30_80" / f"{slug}.json"
        p1b = new_scores_dir / "e09_five_band" / f"{slug}.json"
        if p1a.is_file():
            new_1a[arm_id] = load_json_file(p1a)
        if p1b.is_file():
            new_1b[arm_id] = load_json_file(p1b)

    present_primary_1a = [a for a in NEW_19_PRIMARY_ARMS if a in new_1a]
    present_primary_1b = [a for a in NEW_19_PRIMARY_ARMS if a in new_1b]
    missing_1a = _check_pending_arms(
        NEW_19_PRIMARY_ARMS, present_primary_1a, benchmark="R5 DMS 1A Full (84)", allow_pending=allow_pending
    )
    missing_1b = _check_pending_arms(
        NEW_19_PRIMARY_ARMS, present_primary_1b, benchmark="R5 DMS 1B Full (84)", allow_pending=allow_pending
    )

    all_1a = {**base_1a, **new_1a}
    all_1b = {**base_1b, **new_1b}

    def _entry_meta(entry_key: str) -> dict[str, Any]:
        base_arm = entry_key.split("|")[0]
        m = EXTENDED_ARM_METADATA[entry_key] if entry_key in EXTENDED_ARM_METADATA else EXTENDED_ARM_METADATA[base_arm]
        if entry_key == "msa-transformer-100m|aligned=True":
            align_fmt = "explicit_column_aligned"
        elif m.paradigm == "inverse_folding_gnn":
            align_fmt = "two_chain_spatial_offset_3d"
        else:
            align_fmt = "unaligned_concat"
        return {
            **extended_arm_meta_dict(m.arm_id),
            "entry_key": entry_key,
            "input_alignment_format": align_fmt,
        }

    # Summarize Subtest 1A (rescue_30_80): per-cluster mean Spearman under no_context, unrelated, homolog
    assay_by_name_1a = {str(r["assay"]): r for r in rescue_cohort["assays"]}
    keys_1a = list(all_1a.keys())
    sub1a_summary: dict[str, Any] = {"n_entries": len(keys_1a), "entries": {}}
    if keys_1a:
        # Compute per-assay Spearman for every entry & condition
        cluster_rows_1a: dict[str, dict[str, dict[int, list[float]]]] = {}
        for ek in keys_1a:
            doc = all_1a[ek]
            by_cond: dict[str, dict[int, list[float]]] = {c: defaultdict(list) for c in r5_dms_mod.RESCUE_CONDITIONS}
            assays_obj = doc.get("assays") or []
            items_iter = (
                [{"assay": k, **(v if isinstance(v, Mapping) else {})} for k, v in assays_obj.items()]
                if isinstance(assays_obj, Mapping)
                else assays_obj
            )
            for item in items_iter:
                aname = str(item["assay"])
                crow = assay_by_name_1a.get(aname)
                if crow is None:
                    continue
                cl = int(crow["cluster"])
                sp_map = item.get("spearman") if isinstance(item.get("spearman"), Mapping) else None
                sc_map = item.get("scores") if isinstance(item.get("scores"), Mapping) else None
                y = np.asarray(crow["measured"], dtype=np.float64)
                for cond in r5_dms_mod.RESCUE_CONDITIONS:
                    rho = None
                    if sp_map is not None and sp_map.get(cond) is not None:
                        val = float(sp_map[cond])
                        if np.isfinite(val):
                            rho = val
                    elif sc_map is not None and sc_map.get(cond) is not None and len(sc_map[cond]) == len(y):
                        rho = safe_spearman(sc_map[cond], y)
                    if rho is not None:
                        by_cond[cond][cl].append(rho)
            cluster_rows_1a[ek] = by_cond

        common_clusters_1a = sorted(
            set.intersection(
                *(
                    set(cluster_rows_1a[ek]["homolog"].keys())
                    & set(cluster_rows_1a[ek]["unrelated"].keys())
                    & set(cluster_rows_1a[ek]["no_context"].keys())
                    for ek in keys_1a
                )
            )
        )
        if common_clusters_1a:
            mat_hom = np.column_stack(
                [[float(np.mean(cluster_rows_1a[ek]["homolog"][c])) for c in common_clusters_1a] for ek in keys_1a]
            )
            mat_unr = np.column_stack(
                [[float(np.mean(cluster_rows_1a[ek]["unrelated"][c])) for c in common_clusters_1a] for ek in keys_1a]
            )
            mat_noc = np.column_stack(
                [[float(np.mean(cluster_rows_1a[ek]["no_context"][c])) for c in common_clusters_1a] for ek in keys_1a]
            )
            b_hom_unr = r2_mod.paired_bands_summary(mat_hom - mat_unr, keys_1a, draws=draws, seed=seed)
            b_hom_noc = r2_mod.paired_bands_summary(mat_hom - mat_noc, keys_1a, draws=draws, seed=seed)
            b_hom = r2_mod.paired_bands_summary(mat_hom, keys_1a, draws=draws, seed=seed)
            b_noc = r2_mod.paired_bands_summary(mat_noc, keys_1a, draws=draws, seed=seed)
            sub1a_summary = {
                "n_entries": len(keys_1a),
                "n_common_clusters": len(common_clusters_1a),
                "homolog_minus_unrelated": b_hom_unr,
                "homolog_minus_no_context": b_hom_noc,
                "homolog_absolute": b_hom,
                "no_context_absolute": b_noc,
                "entry_metadata": {ek: _entry_meta(ek) for ek in keys_1a},
            }

    # Summarize Subtest 1B (e09_five_band): per-band delta Spearman vs unrelated
    assay_by_name_1b = {str(r["assay"]): r for r in e09_cohort["assays"]}
    keys_1b = list(all_1b.keys())
    sub1b_summary: dict[str, Any] = {"n_entries": len(keys_1b), "bands": {}}
    if keys_1b:
        conds_1b = ("no_context", "unrelated", *r5_dms_mod.E09_BINS_ORDERED)
        by_entry_1b: dict[str, dict[str, dict[int, list[float]]]] = {}
        for ek in keys_1b:
            doc = all_1b[ek]
            by_c: dict[str, dict[int, list[float]]] = {c: defaultdict(list) for c in conds_1b}
            assays_obj = doc.get("assays") or []
            items_iter = (
                [{"assay": k, **(v if isinstance(v, Mapping) else {})} for k, v in assays_obj.items()]
                if isinstance(assays_obj, Mapping)
                else assays_obj
            )
            for item in items_iter:
                aname = str(item["assay"])
                crow = assay_by_name_1b.get(aname)
                if crow is None:
                    continue
                cl = int(crow["cluster"])
                sp_map = item.get("spearman") if isinstance(item.get("spearman"), Mapping) else None
                sc_map = item.get("scores") if isinstance(item.get("scores"), Mapping) else None
                y = np.asarray(crow["measured"], dtype=np.float64)
                for cond in conds_1b:
                    rho = None
                    if sp_map is not None and sp_map.get(cond) is not None:
                        val = float(sp_map[cond])
                        if np.isfinite(val):
                            rho = val
                    elif sc_map is not None and sc_map.get(cond) is not None and len(sc_map[cond]) == len(y):
                        rho = safe_spearman(sc_map[cond], y)
                    if rho is not None:
                        by_c[cond][cl].append(rho)
            by_entry_1b[ek] = by_c

        bands_1b: dict[str, Any] = {}
        for bname in r5_dms_mod.E09_BINS_ORDERED:
            common_c = sorted(
                set.intersection(
                    *(
                        set(by_entry_1b[ek][bname].keys()) & set(by_entry_1b[ek]["unrelated"].keys())
                        for ek in keys_1b
                    )
                )
            )
            if not common_c:
                bands_1b[bname] = {"n_common_clusters": 0}
                continue
            mat_diff = np.column_stack(
                [
                    [
                        float(np.mean(by_entry_1b[ek][bname][c]) - np.mean(by_entry_1b[ek]["unrelated"][c]))
                        for c in common_c
                    ]
                    for ek in keys_1b
                ]
            )
            bands_1b[bname] = {
                "n_common_clusters": len(common_c),
                "bin_minus_unrelated": r2_mod.paired_bands_summary(mat_diff, keys_1b, draws=draws, seed=seed),
            }
        sub1b_summary = {
            "n_entries": len(keys_1b),
            "entry_metadata": {ek: _entry_meta(ek) for ek in keys_1b},
            "bands": bands_1b,
        }

    missing_base_1a = [s.entry_key for s in r5_dms_mod.DMS_CONTEXT_SPECS_ALL if s.entry_key not in base_1a]
    missing_base_1b = [s.entry_key for s in r5_dms_mod.DMS_CONTEXT_SPECS_ALL if s.entry_key not in base_1b]
    payload = {
        "schema": "r5_dms_context_ranking_unified_full_v1",
        "generated_utc": _now_utc(),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "completion_status": (
            "complete"
            if not (missing_1a or missing_1b or missing_base_1a or missing_base_1b)
            else "partial_pending_h200"
        ),
        "pending_base_1a_entries": missing_base_1a,
        "pending_base_1b_entries": missing_base_1b,
        "pending_new_1a_primary": missing_1a,
        "pending_new_1b_primary": missing_1b,
        "subtest_1a_rescue_30_80": sub1a_summary,
        "subtest_1b_e09_five_band": sub1b_summary,
    }
    odir.mkdir(parents=True, exist_ok=True)
    out_path = odir / "r5_dms_context_ranking_unified_full.json"
    digest = write_verified_json_with_sha256(out_path, payload)
    return {
        "benchmark": "r5_dms",
        "completion_status": payload["completion_status"],
        "n_entries_1a": len(keys_1a),
        "n_entries_1b": len(keys_1b),
        "outputs": {out_path.name: digest},
    }
