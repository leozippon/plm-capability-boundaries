"""Experiment 4.1: Dual-Axis Full-Panel Unconditional Generation, Decoding Artifact & Anchored Controls.

Unifies all 29 unconditional generative checkpoints (32 arm-mode rows including BOTH
DPLM ``uniform_random`` and ``max_confidence``) + 2 controls (``natural_fragments``
and ``markov_order4``) = 34 rows on two orthogonal axes:
  - Axis 1 (Pure CPU): UniRef50 nearest-neighbor identity (DIAMOND ``--masking 0``,
    query-normalized ``100 * nident / qlen``), Pfam-A complete-domain recognition
    (pinned ``hmmscan --cut_ga``, model coverage ``>= 0.80``), Shannon AA entropy H,
    distinct 5-mer fraction, and pool-wide 5-mer near-duplicate deduplication.
  - Axis 2 (ESMFold2 3D Structure): Per-sequence ``mean_ca_plddt`` and ``ptm`` joined
    strictly with Axis 1, stratified by length (``50-128``, ``129-256``, ``257-400`` aa)
    and strictly gated for true novel natural-entropy foldability
    (``UniRef50 < 30% AND H >= 3.0 bits``; sensitivity ``H >= 3.2 bits`` & non-duplicate).

Assembles the four-stage narrative artifact ``r5_experiment_4_1_unified_generation.json``
(+ ``.sha256``), including Stage 4 anchored dual controls (18-arm R5a infill/prefix
ladder + ESM3 R5c conditioning + ZymCTRL EC / ProLLaMA superfamily label recall).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import tarfile
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.capability.core.amino_acids import AA20
from src.capability.core.protein_annotations import parse_hmmscan_table
from src.capability.generation.family_oracle import (
    HMMSCAN_RELATIVE,
    PFAM_RELATIVE,
    Oracle,
    scan_command,
)
from src.capability.generation.generative_control import (
    COMPLETE_DOMAIN_COVERAGE,
    best_profile_coverage,
    parse_domain_table,
)
from src.capability.generation.near_duplicates import (
    NEAR_DUPLICATE_CONTAINMENT,
    near_duplicate_groups,
)
from src.capability.memory_theory.io_utils import load_json_file, write_json_with_sha256
from src.capability.unified.roster import (
    ARM_METADATA,
    REPO_ROOT,
    UNIFIED_OUTPUT_DIR,
    resolve_gp_r5_json,
)

AA20_SET = frozenset(AA20)
EXP4_1_POOLS_REL = Path(UNIFIED_OUTPUT_DIR) / "exp4_1_pools"
EXP4_1_CACHE_REL = Path(UNIFIED_OUTPUT_DIR) / "exp4_1_cache"
ORACLE_ROOT_REL = Path("external/tools/r227-oracle")
DIAMOND_BIN_REL = Path("external/tools/diamond/diamond")
UNIREF50_DB_REL = Path("data/homology_db/uniref50_full.dmnd")

LENGTH_BINS: tuple[tuple[str, int, int], ...] = (
    ("len_50_128", 50, 128),
    ("len_129_256", 129, 256),
    ("len_257_400", 257, 400),
)


@dataclass(frozen=True)
class PoolEntrySpec:
    """Declarative specification for one row of the 34-row Experiment 4.1 table."""

    entry_key: str
    arm_id: str
    source_kind: str  # exp_r2_246 | progen3_3b | new_h200_pool | gp_r5 | esm3_r5 | control
    paradigm_group: str
    generation_regime: str
    diffusion_mode: str | None = None
    is_control: bool = False
    is_primary_mode: bool = True
    params_m: float | None = None


POOL_SPECS_34: tuple[PoolEntrySpec, ...] = (
    # Controls (2)
    PoolEntrySpec("natural_fragments", "natural_fragments", "control", "control_natural", "natural_reference", is_control=True),
    PoolEntrySpec("markov_order4", "markov_order4", "control", "control_markov4", "markov_order4_null", is_control=True),
    # Causal protein (17)
    PoolEntrySpec("progen2-small", "progen2-small", "exp_r2_246", "causal_protein", "free_length_causal"),
    PoolEntrySpec("progen2-base", "progen2-base", "exp_r2_246", "causal_protein", "free_length_causal"),
    PoolEntrySpec("progen2-medium", "progen2-medium", "exp_r2_246", "causal_protein", "free_length_causal"),
    PoolEntrySpec("progen2-large", "progen2-large", "exp_r2_246", "causal_protein", "free_length_causal"),
    PoolEntrySpec("progen2-xlarge", "progen2-xlarge", "exp_r2_246", "causal_protein", "free_length_causal"),
    PoolEntrySpec("progen3-112m", "progen3-112m", "exp_r2_246", "causal_protein", "free_length_causal"),
    PoolEntrySpec("progen3-219m", "progen3-219m", "new_h200_pool", "causal_protein", "free_length_causal"),
    PoolEntrySpec("progen3-339m", "progen3-339m", "new_h200_pool", "causal_protein", "free_length_causal"),
    PoolEntrySpec("progen3-762m", "progen3-762m", "new_h200_pool", "causal_protein", "free_length_causal"),
    PoolEntrySpec("progen3-1b", "progen3-1b", "new_h200_pool", "causal_protein", "free_length_causal"),
    PoolEntrySpec("progen3-3b", "progen3-3b", "progen3_3b", "causal_protein", "free_length_causal"),
    PoolEntrySpec("protgpt2", "protgpt2", "exp_r2_246", "causal_protein", "free_length_causal"),
    PoolEntrySpec("protgpt3-112m", "protgpt3-112m", "gp_r5", "causal_protein", "free_length_causal"),
    PoolEntrySpec("protgpt3-1.3b", "protgpt3-1.3b", "exp_r2_246", "causal_protein", "free_length_causal"),
    PoolEntrySpec("protgpt3-10b", "protgpt3-10b", "gp_r5", "causal_protein", "free_length_causal"),
    PoolEntrySpec("proteinglm-7b-clm", "proteinglm-7b-clm", "exp_r2_246", "causal_protein", "free_length_causal"),
    PoolEntrySpec("rita-xl", "rita-xl", "exp_r2_246", "causal_protein", "free_length_causal"),
    # Hybrid SSM (1)
    PoolEntrySpec("dayhoff-170m", "dayhoff-170m", "gp_r5", "hybrid_ssm", "free_length_causal"),
    # Joint / Adapted (7)
    PoolEntrySpec("galactica-125m", "galactica-125m", "exp_r2_246", "joint_or_adapted", "free_length_causal"),
    PoolEntrySpec("galactica-1.3b", "galactica-1.3b", "exp_r2_246", "joint_or_adapted", "free_length_causal"),
    PoolEntrySpec("galactica-6.7b", "galactica-6.7b", "exp_r2_246", "joint_or_adapted", "free_length_causal"),
    PoolEntrySpec("galactica-30b", "galactica-30b", "exp_r2_246", "joint_or_adapted", "free_length_causal"),
    PoolEntrySpec("instructprotein", "instructprotein", "exp_r2_246", "joint_or_adapted", "free_length_causal"),
    PoolEntrySpec("prollama-stage-1", "prollama-stage-1", "exp_r2_246", "joint_or_adapted", "free_length_causal"),
    PoolEntrySpec("prollama", "prollama", "exp_r2_246", "joint_or_adapted", "free_length_causal"),
    # Discrete Diffusion (3 x 2 modes = 6)
    PoolEntrySpec("dplm-150m|uniform_random", "dplm-150m", "gp_r5", "discrete_diffusion", "length_matched_diffusion", diffusion_mode="uniform_random", is_primary_mode=True),
    PoolEntrySpec("dplm-650m|uniform_random", "dplm-650m", "gp_r5", "discrete_diffusion", "length_matched_diffusion", diffusion_mode="uniform_random", is_primary_mode=True),
    PoolEntrySpec("dplm-3b|uniform_random", "dplm-3b", "gp_r5", "discrete_diffusion", "length_matched_diffusion", diffusion_mode="uniform_random", is_primary_mode=True),
    PoolEntrySpec("dplm-150m|max_confidence", "dplm-150m", "gp_r5", "discrete_diffusion_maxconf", "length_matched_diffusion_maxconf", diffusion_mode="max_confidence", is_primary_mode=False),
    PoolEntrySpec("dplm-650m|max_confidence", "dplm-650m", "gp_r5", "discrete_diffusion_maxconf", "length_matched_diffusion_maxconf", diffusion_mode="max_confidence", is_primary_mode=False),
    PoolEntrySpec("dplm-3b|max_confidence", "dplm-3b", "gp_r5", "discrete_diffusion_maxconf", "length_matched_diffusion_maxconf", diffusion_mode="max_confidence", is_primary_mode=False),
    # Multimodal (1)
    PoolEntrySpec("esm3-1.4b", "esm3-1.4b", "esm3_r5", "multimodal_3d", "length_matched_diffusion"),
)


def seq_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def is_canonical_aa20(sequence: str) -> bool:
    return bool(sequence) and (set(sequence) <= AA20_SET)


def sequence_shannon_entropy_bits(sequence: str) -> float:
    """Per-sequence Shannon AA entropy H in bits (0.0 for empty or homopolymer)."""
    if not sequence:
        return 0.0
    counts = Counter(sequence)
    n = float(len(sequence))
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def sequence_distinct_5mer_fraction(sequence: str) -> float:
    """Fraction of distinct 5-mers in a sequence (0.0 if length < 5)."""
    n_k = len(sequence) - 4
    if n_k <= 0:
        return 0.0
    kmers = {sequence[i : i + 5] for i in range(n_k)}
    return float(len(kmers) / n_k)


def classify_identity_band(identity_pct: float) -> str:
    if identity_pct < 30.0:
        return "lt30_novel"
    if identity_pct <= 70.0:
        return "30_to_70_homolog"
    return "gt70_near_copy"


def bootstrap_mean_ci95(
    values: Sequence[float] | np.ndarray,
    *,
    draws: int = 10000,
    seed: int = 20261124,
) -> dict[str, Any]:
    """Deterministic percentile bootstrap 95% CI for a 1D sample mean."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": None, "ci95": None}
    mean_val = float(np.mean(arr))
    if n == 1:
        return {"n": 1, "mean": mean_val, "ci95": [mean_val, mean_val]}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(draws, n))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"n": n, "mean": mean_val, "ci95": [float(lo), float(hi)]}


def _resolve_repo_path(repo_root: Path, raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (repo_root / p)


def _rel_repo_path(repo_root: Path, path: Path) -> str:
    resolved = _resolve_repo_path(repo_root, path)
    return str(resolved.relative_to(repo_root)) if repo_root in resolved.parents else str(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _find_pool_file(pools_dir: Path, patterns: Sequence[str]) -> Path | None:
    for pat in patterns:
        p = pools_dir / pat
        if p.is_file():
            return p
    return None


def _parse_fold_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize folded structure fields from any EXP-R2-246 / GP R5 / new H200 JSONL."""
    plddt = (
        row.get("mean_ca_plddt")
        if row.get("mean_ca_plddt") is not None
        else (row.get("plddt") if row.get("plddt") is not None else row.get("mean_plddt"))
    )
    ptm = row.get("ptm") if row.get("ptm") is not None else row.get("mean_ptm")
    if plddt is None:
        return None
    plddt_f = float(plddt)
    if not math.isfinite(plddt_f):
        return None
    ptm_f = float(ptm) if ptm is not None and math.isfinite(float(ptm)) else None
    return {
        "mean_ca_plddt": plddt_f,
        "ptm": ptm_f,
        "fold_status": str(row.get("status", "ok")),
    }


def load_entry_records(
    spec: PoolEntrySpec,
    repo_root: Path = REPO_ROOT,
    *,
    require_structures: bool = True,
) -> dict[str, Any] | None:
    """Load full unconditional pool + joined per-sequence ESMFold2 metrics for one entry.

    Returns ``None`` if a newly scheduled H200 pool file has not landed yet and
    ``require_structures=False``.
    """
    repo_root = Path(repo_root)
    pools_dir = repo_root / EXP4_1_POOLS_REL
    rows: list[dict[str, Any]] = []
    source_paths: list[str] = []

    if spec.source_kind == "exp_r2_246":
        arm_dir = repo_root / "results/R6/generation_evidence_20260905/analysis/esmfold2_20260920012718" / spec.arm_id
        att_path = arm_dir / "grouped_annotated_attempts.jsonl"
        pair_path = arm_dir / "pair_sufficient_statistics.csv"
        source_paths.extend([str(att_path.relative_to(repo_root)), str(pair_path.relative_to(repo_root))])

        struct_by_id: dict[str, dict[str, Any]] = {}
        struct_by_sha: dict[str, dict[str, Any]] = {}
        with pair_path.open("r", encoding="utf-8") as fh:
            for prow in csv.DictReader(fh):
                if prow.get("status") != "ok":
                    continue
                parsed = _parse_fold_row(
                    {"mean_ca_plddt": prow.get("mean_ca_plddt"), "ptm": prow.get("ptm"), "status": "ok"}
                )
                if parsed is not None:
                    struct_by_id[str(prow["id"])] = parsed
                    if prow.get("sequence_sha256"):
                        struct_by_sha[str(prow["sequence_sha256"])] = parsed

        with att_path.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if not line.strip():
                    continue
                r = json.loads(line)
                seq = str(r.get("sequence") or "").strip()
                sha = seq_sha256(seq) if seq else ""
                rid = str(r.get("id") or f"{spec.arm_id}_{idx:04d}")
                st = struct_by_id.get(rid) or (struct_by_sha.get(sha) if r.get("selected_for_structure") else None)
                stop_st = str(r.get("stop_status") or "")
                valid = bool(r.get("valid_aa20")) and is_canonical_aa20(seq)
                supported = (str(r.get("support_status")) in ("eligible", "supported")) and valid and len(seq) >= 16
                rows.append(
                    {
                        "seq_id": rid,
                        "sequence": seq,
                        "sequence_sha256": sha,
                        "length": len(seq),
                        "valid_aa20": valid,
                        "supported": supported,
                        "budget_censored": ("cap" in stop_st.lower() or "budget" in stop_st.lower() or len(seq) >= 400),
                        "cached_any_profile": r.get("any_profile_hit"),
                        "structure": st,
                    }
                )

    elif spec.source_kind == "progen3_3b":
        att_path = repo_root / "results/R6/progen3_generation_20260905/analysis/20260904232436_00129607e3c7/annotated_attempts.jsonl"
        tar_path = repo_root / "results/R6/generation_evidence_20260905/retention/20260904232436_00129607e3c7/r233-structure-metrics.tar.gz"
        source_paths.extend([str(att_path.relative_to(repo_root)), str(tar_path.relative_to(repo_root))])

        struct_by_id = {}
        struct_by_sha = {}
        with tarfile.open(tar_path, "r:gz") as tf:
            fobj = tf.extractfile("predictions.jsonl")
            assert fobj is not None
            for raw_line in fobj:
                d = json.loads(raw_line)
                if d.get("role") != "generation" or d.get("arm") != "progen3-3b":
                    continue
                st_raw = d.get("structure") or {}
                if st_raw.get("status") != "ok":
                    continue
                parsed = _parse_fold_row(st_raw)
                if parsed is not None:
                    struct_by_id[str(d["id"])] = parsed
                    if d.get("sequence_sha256"):
                        struct_by_sha[str(d["sequence_sha256"])] = parsed

        with att_path.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if not line.strip():
                    continue
                r = json.loads(line)
                seq = str(r.get("sequence") or "").strip()
                sha = seq_sha256(seq) if seq else ""
                rid = str(r.get("id") or f"progen3-3b_{idx:04d}")
                st = struct_by_id.get(rid) or (struct_by_sha.get(sha) if r.get("selected_for_structure") else None)
                valid = bool(r.get("valid_aa20")) and is_canonical_aa20(seq)
                supported = (str(r.get("support_status")) in ("eligible", "supported")) and valid and len(seq) >= 16
                rows.append(
                    {
                        "seq_id": rid,
                        "sequence": seq,
                        "sequence_sha256": sha,
                        "length": len(seq),
                        "valid_aa20": valid,
                        "supported": supported,
                        "budget_censored": bool(r.get("source_budget_censored")) or len(seq) >= 400,
                        "cached_any_profile": r.get("any_profile_hit"),
                        "structure": st,
                    }
                )

    elif spec.source_kind == "gp_r5":
        dmode = spec.diffusion_mode or "max_confidence"
        art = resolve_gp_r5_json(spec.arm_id, diffusion_mode=dmode, root=repo_root)
        draws_path = art.path.parent / f"r5_draws_{spec.arm_id}.jsonl"
        gs = load_json_file(repo_root / "results/extensions/general_plm_20261120/general_plm_summary.json")
        comp = gs.get("companion_artifacts", {})
        suffix_map = {
            ("protgpt3-112m", None): "r5_struct_protgpt3_112m",
            ("protgpt3-10b", None): "r5_struct_protgpt3_10b",
            ("dayhoff-170m", None): "r5_struct_dayhoff_170m",
            ("dplm-150m", "uniform_random"): "r5_struct_dplm_150m_uniform",
            ("dplm-650m", "uniform_random"): "r5_struct_dplm_650m_uniform",
            ("dplm-3b", "uniform_random"): "r5_struct_dplm_3b_uniform",
            ("dplm-150m", "max_confidence"): "r5_struct_dplm_150m_maxconf",
            ("dplm-650m", "max_confidence"): "r5_struct_dplm_650m_maxconf",
            ("dplm-3b", "max_confidence"): "r5_struct_dplm_3b_maxconf",
        }
        slabel = suffix_map[(spec.arm_id, spec.diffusion_mode)]
        eval_json = _resolve_repo_path(repo_root, comp[f"r5_structure_eval::{slabel}"]["source"])
        folded_path = eval_json.parent / "r5_folded_records.jsonl"
        source_paths.extend(
            [
                _rel_repo_path(repo_root, draws_path),
                _rel_repo_path(repo_root, folded_path),
            ]
        )

        struct_by_sha: dict[str, dict[str, Any]] = {}
        with folded_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("rung") is not None:
                    continue
                parsed = _parse_fold_row(d)
                if parsed is not None:
                    seq_f = str(d.get("sequence") or "").strip()
                    sha_f = str(d.get("sequence_sha256") or seq_sha256(seq_f))
                    struct_by_sha[sha_f] = parsed

        with draws_path.open("r", encoding="utf-8") as fh:
            idx = 0
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("rung") is not None:
                    continue
                seq = str(r.get("sequence") or r.get("generated_sequence") or "").strip()
                sha = seq_sha256(seq) if seq else ""
                valid = is_canonical_aa20(seq)
                supported = valid and len(seq) >= 16
                rows.append(
                    {
                        "seq_id": str(r.get("record_id") or r.get("id") or f"{spec.entry_key}_{idx:04d}"),
                        "sequence": seq,
                        "sequence_sha256": sha,
                        "length": len(seq),
                        "valid_aa20": valid,
                        "supported": supported,
                        "budget_censored": (spec.generation_regime == "free_length_causal" and len(seq) >= 400),
                        "cached_any_profile": None,
                        "structure": struct_by_sha.get(sha),
                    }
                )
                idx += 1

    elif spec.source_kind == "esm3_r5":
        gs = load_json_file(repo_root / "results/extensions/general_plm_20261120/general_plm_summary.json")
        eval_json = _resolve_repo_path(
            repo_root, gs["companion_artifacts"]["r5_structure_eval::r5_struct_esm3_1_4b"]["source"]
        )
        folded_path = eval_json.parent / "r5_folded_records.jsonl"
        source_paths.append(_rel_repo_path(repo_root, folded_path))

        seen_ids: set[str] = set()
        with folded_path.open("r", encoding="utf-8") as fh:
            idx = 0
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("rung") is not None or d.get("condition") is not None:
                    continue
                seq = str(d.get("sequence") or "").strip()
                sha = seq_sha256(seq) if seq else ""
                rid = str(d.get("record_id") or f"esm3_denovo_{idx:04d}")
                seen_ids.add(rid)
                rows.append(
                    {
                        "seq_id": rid,
                        "sequence": seq,
                        "sequence_sha256": sha,
                        "length": len(seq),
                        "valid_aa20": is_canonical_aa20(seq),
                        "supported": is_canonical_aa20(seq) and len(seq) >= 16,
                        "budget_censored": False,
                        "cached_any_profile": None,
                        "structure": _parse_fold_row(d),
                    }
                )
                idx += 1

        # Automatically merge any supplemental ESM3 unconditional folded file if staged in exp4_1_pools
        for extra in sorted(pools_dir.glob("esm3*esmfold2*.jsonl")):
            source_paths.append(str(extra.relative_to(repo_root)))
            with extra.open("r", encoding="utf-8") as fh:
                for e_idx, line in enumerate(fh):
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    rid = str(d.get("seq_id") or d.get("record_id") or f"esm3_extra_{e_idx:04d}")
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)
                    seq = str(d.get("sequence") or "").strip()
                    sha = seq_sha256(seq) if seq else ""
                    rows.append(
                        {
                            "seq_id": rid,
                            "sequence": seq,
                            "sequence_sha256": sha,
                            "length": len(seq),
                            "valid_aa20": is_canonical_aa20(seq),
                            "supported": is_canonical_aa20(seq) and len(seq) >= 16,
                            "budget_censored": False,
                            "cached_any_profile": None,
                            "structure": _parse_fold_row(d),
                        }
                    )

    elif spec.source_kind in ("new_h200_pool", "control"):
        stem_us = spec.arm_id.replace("-", "_")
        stem_hy = spec.arm_id
        pool_path = _find_pool_file(
            pools_dir,
            (
                f"{stem_us}_pool_800.jsonl",
                f"{stem_hy}_pool_800.jsonl",
                f"{stem_us}_pool.jsonl",
            ),
        )
        if pool_path is None:
            if require_structures:
                raise FileNotFoundError(f"Missing pool file for {spec.arm_id} under {pools_dir}")
            return None
        source_paths.append(str(pool_path.relative_to(repo_root)))

        fold_path = _find_pool_file(
            pools_dir,
            (
                f"{stem_us}_esmfold2_400.jsonl",
                f"{stem_hy}_esmfold2_400.jsonl",
                f"{stem_us}_fold_400_esmfold2.jsonl",
                f"{stem_us}_esmfold2.jsonl",
            ),
        )
        if fold_path is None and require_structures:
            raise FileNotFoundError(f"Missing ESMFold2 file for {spec.arm_id} under {pools_dir}")

        struct_by_id = {}
        struct_by_sha = {}
        if fold_path is not None:
            source_paths.append(str(fold_path.relative_to(repo_root)))
            with fold_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    parsed = _parse_fold_row(d)
                    if parsed is None:
                        continue
                    sid = str(d.get("seq_id") or d.get("id") or d.get("record_id") or "")
                    if sid:
                        struct_by_id[sid] = parsed
                    seq_f = str(d.get("sequence") or "").strip()
                    if seq_f:
                        struct_by_sha[seq_sha256(seq_f)] = parsed

        with pool_path.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if not line.strip():
                    continue
                r = json.loads(line)
                seq = str(r.get("sequence") or "").strip()
                sha = seq_sha256(seq) if seq else ""
                rid = str(r.get("seq_id") or r.get("id") or f"{stem_us}_{idx:04d}")
                valid = is_canonical_aa20(seq)
                supported = valid and len(seq) >= 16
                st = struct_by_id.get(rid) or (struct_by_sha.get(sha) if sha else None)
                budget_cens = (
                    (not bool(r.get("terminated_eos")))
                    if "terminated_eos" in r
                    else (spec.source_kind == "new_h200_pool" and len(seq) >= 400)
                )
                rows.append(
                    {
                        "seq_id": rid,
                        "sequence": seq,
                        "sequence_sha256": sha,
                        "length": len(seq),
                        "valid_aa20": valid,
                        "supported": supported,
                        "budget_censored": budget_cens,
                        "cached_any_profile": None,
                        "structure": st,
                    }
                )
    else:
        raise ValueError(f"Unknown source_kind: {spec.source_kind}")

    return {
        "spec": spec,
        "rows": rows,
        "source_paths": source_paths,
    }


# ---------------------------------------------------------------------------
# Axis 1 Incremental Annotation Caches (DIAMOND UniRef50 & Pinned Pfam hmmscan)
# ---------------------------------------------------------------------------


def _seed_diamond_cache(repo_root: Path, cache: dict[str, dict[str, Any]]) -> int:
    """Seed exact SHA-256 DIAMOND UniRef50 hits from generation_novelty_20261008."""
    ev_path = repo_root / "results/R6/generation_evaluation_20261008/evaluation_cohort.jsonl"
    nov_path = repo_root / "results/R6/generation_novelty_20261008/generated_novelty.jsonl"
    if not (ev_path.is_file() and nov_path.is_file()):
        return 0
    id_to_sha: dict[str, str] = {}
    with ev_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            seq = str(r.get("sequence") or "").strip()
            if is_canonical_aa20(seq):
                id_to_sha[str(r["id"])] = seq_sha256(seq)
    added = 0
    with nov_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            sha = id_to_sha.get(str(r["id"]))
            if sha and sha not in cache:
                ident = float(r["nearest_corpus_identity"])
                cache[sha] = {
                    "identity_over_query": ident,
                    "subject": r.get("nearest_corpus_subject"),
                    "evalue": r.get("nearest_corpus_evalue"),
                    "band": classify_identity_band(ident),
                    "source": "generation_novelty_20261008",
                }
                added += 1
    return added


def ensure_diamond_annotations(
    unique_seqs: Mapping[str, str],
    repo_root: Path = REPO_ROOT,
    *,
    threads: int = 64,
    block_size: float = 8.0,
    sensitivity: str = "very-sensitive",
    evalue: float = 1e-3,
    max_target_seqs: int = 25,
) -> dict[str, dict[str, Any]]:
    """Run DIAMOND blastp (--masking 0) against uniref50_full.dmnd for all uncached SHA-256s."""
    import time

    cache_path = repo_root / EXP4_1_CACHE_REL / "diamond_uniref50_by_sha256.json"
    cache: dict[str, dict[str, Any]] = load_json_file(cache_path) if cache_path.is_file() else {}
    if not cache:
        seeded = _seed_diamond_cache(repo_root, cache)
        print(f"[diamond] seeded {seeded} cached hits from generation_novelty_20261008", flush=True)

    missing = {sha: seq for sha, seq in unique_seqs.items() if sha not in cache and is_canonical_aa20(seq) and len(seq) >= 10}
    if not missing:
        if not cache_path.is_file():
            _atomic_write_json(cache_path, cache)
        return cache

    diamond_bin = repo_root / DIAMOND_BIN_REL
    db_path = repo_root / UNIREF50_DB_REL
    if not diamond_bin.is_file() or not db_path.is_file():
        raise FileNotFoundError(f"DIAMOND binary or DB missing: {diamond_bin}, {db_path}")

    t0 = time.monotonic()
    print(
        f"[diamond] querying {len(missing)} unique sequences against {db_path.name} "
        f"(--{sensitivity} --masking 0 --threads {threads} --block-size {block_size})...",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="exp4_1_diamond_") as tmpdir:
        tmp = Path(tmpdir)
        q_fa = tmp / "queries.faa"
        hits_tsv = tmp / "hits.tsv"
        with q_fa.open("w", encoding="utf-8") as fh:
            for sha, seq in sorted(missing.items()):
                fh.write(f">{sha}\n{seq}\n")

        cmd = [
            str(diamond_bin),
            "blastp",
            "--query",
            str(q_fa),
            "--db",
            str(db_path),
            "--out",
            str(hits_tsv),
            "--outfmt",
            "6",
            "qseqid",
            "sseqid",
            "nident",
            "qlen",
            "evalue",
            "--masking",
            "0",
            f"--{sensitivity}",
            "--evalue",
            str(evalue),
            "--max-target-seqs",
            str(max_target_seqs),
            "--block-size",
            str(block_size),
            "--threads",
            str(threads),
            "--quiet",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"DIAMOND failed ({res.returncode}): {res.stderr[-2000:]}")

        best: dict[str, tuple[float, str, float]] = {}
        if hits_tsv.is_file():
            with hits_tsv.open("r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 5:
                        continue
                    qsha, subj, nident_s, qlen_s, eval_s = parts[:5]
                    qlen = float(qlen_s)
                    if qlen <= 0:
                        continue
                    ident = 100.0 * float(nident_s) / qlen
                    ev = float(eval_s)
                    prev = best.get(qsha)
                    if prev is None or ident > prev[0]:
                        best[qsha] = (ident, subj, ev)

        for sha in missing:
            hit = best.get(sha)
            if hit is None:
                cache[sha] = {
                    "identity_over_query": 0.0,
                    "subject": None,
                    "evalue": None,
                    "band": "lt30_novel",
                    "source": "diamond_uniref50_full_masking0",
                }
            else:
                ident, subj, ev = hit
                cache[sha] = {
                    "identity_over_query": float(ident),
                    "subject": subj,
                    "evalue": float(ev),
                    "band": classify_identity_band(ident),
                    "source": "diamond_uniref50_full_masking0",
                }

    _atomic_write_json(cache_path, cache)
    print(
        f"[diamond] finished {len(missing)} sequences in {time.monotonic() - t0:.1f}s "
        f"(cache total={len(cache)})",
        flush=True,
    )
    return cache


def _seed_pfam_cache(
    repo_root: Path,
    entries_loaded: Sequence[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
) -> int:
    """Seed exact pinned-oracle Pfam negatives (EXP-R2-246/progen3-3b) & objective_tuned complete_domain flags."""
    added = 0
    # 1. Exact complete_domain flags from objective_tuned_endpoints_20261111/recognition/recognition_flags.jsonl
    ot_rec = repo_root / "results/extensions/objective_tuned_endpoints_20261111/recognition/recognition_flags.jsonl"
    if ot_rec.is_file():
        with ot_rec.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                sha = str(r.get("sequence_sha256") or "")
                if sha and sha not in cache:
                    fams = list(r.get("families") or [])
                    cov = float(r.get("best_profile_coverage") or 0.0)
                    cache[sha] = {
                        "any_family": bool(r.get("any_family")),
                        "complete_domain": bool(r.get("complete_domain")),
                        "best_profile_coverage": cov,
                        "families": sorted(set(fams)),
                        "source": "objective_tuned_recognition_flags",
                    }
                    added += 1

    # 2. Exact zero-hit negatives from EXP-R2-246 & progen3-3b (same pinned hmmscan --cut_ga oracle!)
    for entry in entries_loaded:
        for row in entry["rows"]:
            sha = row["sequence_sha256"]
            if not sha or sha in cache:
                continue
            if row.get("cached_any_profile") is False:
                cache[sha] = {
                    "any_family": False,
                    "complete_domain": False,
                    "best_profile_coverage": 0.0,
                    "families": [],
                    "source": "pinned_exp_r2_246_no_hit",
                }
                added += 1
    return added


def ensure_pfam_annotations(
    unique_seqs: Mapping[str, str],
    entries_loaded: Sequence[dict[str, Any]],
    repo_root: Path = REPO_ROOT,
    *,
    workers: int = 32,
    threads_per_worker: int = 2,
) -> dict[str, dict[str, Any]]:
    """Run pinned hmmscan (--cut_ga) against Pfam-A.hmm for all uncached unique SHA-256s."""
    import time

    cache_path = repo_root / EXP4_1_CACHE_REL / "pfam_hmmscan_by_sha256.json"
    cache: dict[str, dict[str, Any]] = load_json_file(cache_path) if cache_path.is_file() else {}
    seeded = _seed_pfam_cache(repo_root, entries_loaded, cache)
    if seeded:
        print(f"[hmmscan] seeded {seeded} pinned Pfam records (cache={len(cache)})", flush=True)

    missing = sorted(
        (sha, seq)
        for sha, seq in unique_seqs.items()
        if sha not in cache and is_canonical_aa20(seq) and len(seq) >= 16
    )
    if not missing:
        _atomic_write_json(cache_path, cache)
        return cache

    oracle_root = repo_root / ORACLE_ROOT_REL
    hmmscan_bin = oracle_root / HMMSCAN_RELATIVE
    pfam_db = oracle_root / PFAM_RELATIVE
    if not hmmscan_bin.is_file() or not pfam_db.is_file():
        raise FileNotFoundError(f"Pinned Pfam oracle missing under {oracle_root}")

    oracle = Oracle(hmmscan=hmmscan_bin, pfam=pfam_db, digests={}, root=oracle_root)
    n_shards = min(max(workers * 2, 1), len(missing))
    shards: list[list[tuple[str, str]]] = [[] for _ in range(n_shards)]
    for i, item in enumerate(missing):
        shards[i % n_shards].append(item)

    t0 = time.monotonic()
    print(
        f"[hmmscan] scanning {len(missing)} unique sequences across {n_shards} shards "
        f"({workers} workers x {threads_per_worker} threads)...",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="exp4_1_hmmscan_") as tmpdir:
        tmp = Path(tmpdir)

        def _run_one(shard_idx: int) -> dict[str, dict[str, Any]]:
            items = shards[shard_idx]
            fa = tmp / f"shard_{shard_idx:03d}.faa"
            tbl = tmp / f"shard_{shard_idx:03d}.tbl"
            domtbl = tmp / f"shard_{shard_idx:03d}.domtbl"
            with fa.open("w", encoding="utf-8") as fh:
                for sha, seq in items:
                    fh.write(f">{sha}\n{seq}\n")
            cmd = scan_command(oracle, fa, tbl, domtbl, threads=threads_per_worker)
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"hmmscan shard {shard_idx} failed: {res.stderr[-2000:]}")
            tbl_hits = parse_hmmscan_table(tbl)
            dom_hits = parse_domain_table(domtbl)
            local_out: dict[str, dict[str, Any]] = {}
            for sha, _ in items:
                fams = sorted({e["accession_unversioned"] for e in tbl_hits.get(sha, [])})
                dom_entries = dom_hits.get(sha, [])
                best_cov = best_profile_coverage(dom_entries) if dom_entries else 0.0
                best_cov_f = float(best_cov or 0.0)
                any_fam = bool(fams)
                complete_dom = bool(any_fam and best_cov_f >= COMPLETE_DOMAIN_COVERAGE)
                local_out[sha] = {
                    "any_family": any_fam,
                    "complete_domain": complete_dom,
                    "best_profile_coverage": best_cov_f,
                    "families": fams,
                    "source": "hmmscan_pfam_a_cut_ga",
                }
            return local_out

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for shard_res in ex.map(_run_one, range(len(shards))):
                cache.update(shard_res)
                done += 1
                if done % 16 == 0 or done == len(shards):
                    print(
                        f"[hmmscan] shard {done}/{len(shards)} complete ({time.monotonic() - t0:.1f}s)",
                        flush=True,
                    )

    _atomic_write_json(cache_path, cache)
    return cache


# ---------------------------------------------------------------------------
# Entry-Level Aggregation (Axis 1 + Axis 2 Joined + Gated + Length-Stratified)
# ---------------------------------------------------------------------------


def _summarize_subset_structure(
    rows: Sequence[dict[str, Any]],
    *,
    draws: int = 10000,
    seed: int = 20261124,
) -> dict[str, Any]:
    """Compute bootstrap 95% CIs on pLDDT, pLDDT>=70, pTM, Pfam, entropy, and UniRef50 id."""
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "mean_plddt": {"n": 0, "mean": None, "ci95": None},
            "foldable_rate_plddt_ge_70": {"n": 0, "mean": None, "ci95": None},
            "mean_ptm": {"n": 0, "mean": None, "ci95": None},
            "pfam_any_rate": None,
            "pfam_complete_domain_rate": None,
            "mean_entropy_bits": None,
            "mean_distinct_5mer": None,
            "mean_uniref50_identity": None,
        }
    plddts = np.asarray([r["mean_ca_plddt"] for r in rows], dtype=np.float64)
    ge70 = (plddts >= 70.0).astype(np.float64)
    ptms = np.asarray([r["ptm"] for r in rows if r["ptm"] is not None], dtype=np.float64)
    return {
        "n": n,
        "mean_plddt": bootstrap_mean_ci95(plddts, draws=draws, seed=seed),
        "foldable_rate_plddt_ge_70": bootstrap_mean_ci95(ge70, draws=draws, seed=seed + 1),
        "mean_ptm": bootstrap_mean_ci95(ptms, draws=draws, seed=seed + 2),
        "pfam_any_rate": float(np.mean([1.0 if r["pfam_any_family"] else 0.0 for r in rows])),
        "pfam_complete_domain_rate": float(np.mean([1.0 if r["pfam_complete_domain"] else 0.0 for r in rows])),
        "mean_entropy_bits": float(np.mean([r["entropy_bits"] for r in rows])),
        "mean_distinct_5mer": float(np.mean([r["distinct_5mer_fraction"] for r in rows])),
        "mean_uniref50_identity": float(np.mean([r["uniref50_identity"] for r in rows])),
    }


def _aggregate_one_entry(
    loaded: dict[str, Any],
    diamond_cache: Mapping[str, dict[str, Any]],
    pfam_cache: Mapping[str, dict[str, Any]],
    rep27_summary: Mapping[str, Any],
    *,
    draws: int = 10000,
    seed: int = 20261124,
) -> dict[str, Any]:
    spec: PoolEntrySpec = loaded["spec"]
    raw_rows: list[dict[str, Any]] = loaded["rows"]
    n_pool = len(raw_rows)

    # Pool-wide 5-mer near-duplicate clustering on valid AA20 sequences (len >= 5)
    nd_idx = [i for i, r in enumerate(raw_rows) if r["valid_aa20"] and len(r["sequence"]) >= 5]
    nd_group_of: dict[int, int] = {}
    is_rep_of: dict[int, bool] = {}
    if nd_idx:
        groups_arr, nd_summary = near_duplicate_groups(
            [raw_rows[i]["sequence"] for i in nd_idx],
            unit="residues",
            containment=NEAR_DUPLICATE_CONTAINMENT,
        )
        seen_groups: set[int] = set()
        for pos, orig_i in enumerate(nd_idx):
            g = int(groups_arr[pos])
            nd_group_of[orig_i] = g
            if g not in seen_groups:
                seen_groups.add(g)
                is_rep_of[orig_i] = True
            else:
                is_rep_of[orig_i] = False
        unique_group_rate = float(nd_summary["n_groups"] / len(nd_idx))
        largest_group_share = float(nd_summary["largest_group_share"])
    else:
        nd_summary = {"n_records": 0, "n_groups": 0, "largest_group_share": 0.0}
        unique_group_rate = 0.0
        largest_group_share = 0.0

    enriched: list[dict[str, Any]] = []
    for i, r in enumerate(raw_rows):
        seq = r["sequence"]
        sha = r["sequence_sha256"]
        dm = diamond_cache.get(sha)
        pf = pfam_cache.get(sha)
        ident = float(dm["identity_over_query"]) if dm is not None else 0.0
        pf_any = bool(pf["any_family"]) if pf is not None else False
        pf_comp = bool(pf["complete_domain"]) if pf is not None else False
        pf_cov = float(pf["best_profile_coverage"]) if pf is not None else 0.0
        pf_fams = list(pf["families"]) if pf is not None else []
        ent = sequence_shannon_entropy_bits(seq) if r["valid_aa20"] else 0.0
        d5 = sequence_distinct_5mer_fraction(seq) if r["valid_aa20"] else 0.0
        st = r["structure"]
        enriched.append(
            {
                "seq_id": r["seq_id"],
                "sequence_sha256": sha,
                "length": int(r["length"]),
                "valid_aa20": bool(r["valid_aa20"]),
                "supported": bool(r["supported"]),
                "budget_censored": bool(r["budget_censored"]),
                "entropy_bits": ent,
                "distinct_5mer_fraction": d5,
                "near_duplicate_group": nd_group_of.get(i),
                "is_non_duplicate_rep": bool(is_rep_of.get(i, False)),
                "uniref50_identity": ident,
                "uniref50_band": classify_identity_band(ident),
                "pfam_any_family": pf_any,
                "pfam_complete_domain": pf_comp,
                "pfam_best_coverage": pf_cov,
                "pfam_families": pf_fams,
                "has_structure": st is not None,
                "mean_ca_plddt": st["mean_ca_plddt"] if st is not None else None,
                "ptm": st["ptm"] if st is not None else None,
            }
        )

    # Axis 1 Full Pool Summary (all N attempts, denominator = N_pool for rates, plus supported subset)
    supported_rows = [r for r in enriched if r["supported"]]
    n_sup = len(supported_rows)
    sup_rate = float(n_sup / n_pool) if n_pool > 0 else 0.0
    valid_rate = float(sum(1 for r in enriched if r["valid_aa20"]) / n_pool) if n_pool > 0 else 0.0
    cap_rate = float(sum(1 for r in enriched if r["budget_censored"]) / n_pool) if n_pool > 0 else 0.0

    # Pfam rates over ALL attempts (conservative all-attempt yield) AND over supported seqs
    pfam_any_all = float(sum(1 for r in enriched if r["pfam_any_family"]) / n_pool) if n_pool else 0.0
    pfam_comp_all = float(sum(1 for r in enriched if r["pfam_complete_domain"]) / n_pool) if n_pool else 0.0
    pfam_any_sup = float(sum(1 for r in supported_rows if r["pfam_any_family"]) / n_sup) if n_sup else 0.0
    pfam_comp_sup = float(sum(1 for r in supported_rows if r["pfam_complete_domain"]) / n_sup) if n_sup else 0.0
    distinct_pfams = len({f for r in enriched for f in r["pfam_families"]})

    # UniRef50 identity stratification over supported sequences
    id_vals_sup = [r["uniref50_identity"] for r in supported_rows]
    frac_lt30 = float(sum(1 for v in id_vals_sup if v < 30.0) / n_sup) if n_sup else 0.0
    frac_30_70 = float(sum(1 for v in id_vals_sup if 30.0 <= v <= 70.0) / n_sup) if n_sup else 0.0
    frac_gt70 = float(sum(1 for v in id_vals_sup if v > 70.0) / n_sup) if n_sup else 0.0
    frac_ge95 = float(sum(1 for v in id_vals_sup if v >= 95.0) / n_sup) if n_sup else 0.0

    # Axis 2 Folded Subset & Strict Gating
    folded_rows = [r for r in enriched if r["has_structure"] and r["valid_aa20"]]
    n_folded = len(folded_rows)

    gated_30 = [r for r in folded_rows if r["uniref50_identity"] < 30.0 and r["entropy_bits"] >= 3.0]
    gated_32_nd = [
        r
        for r in folded_rows
        if r["uniref50_identity"] < 30.0 and r["entropy_bits"] >= 3.2 and r["is_non_duplicate_rep"]
    ]
    low_h_collapse = [r for r in folded_rows if r["entropy_bits"] < 3.0]
    homolog_ge30 = [r for r in folded_rows if r["uniref50_identity"] >= 30.0]

    # Joint unconditional gated yield of pool: P(supported) * P(<30% & H>=3.0 & pLDDT>=70 | folded)
    if n_folded > 0:
        joint_ind_30 = np.asarray(
            [
                sup_rate
                if (r["uniref50_identity"] < 30.0 and r["entropy_bits"] >= 3.0 and r["mean_ca_plddt"] >= 70.0)
                else 0.0
                for r in folded_rows
            ],
            dtype=np.float64,
        )
        joint_ind_32 = np.asarray(
            [
                sup_rate
                if (
                    r["uniref50_identity"] < 30.0
                    and r["entropy_bits"] >= 3.2
                    and r["is_non_duplicate_rep"]
                    and r["mean_ca_plddt"] >= 70.0
                )
                else 0.0
                for r in folded_rows
            ],
            dtype=np.float64,
        )
        gated_yield_30 = bootstrap_mean_ci95(joint_ind_30, draws=draws, seed=seed + 10)
        gated_yield_32 = bootstrap_mean_ci95(joint_ind_32, draws=draws, seed=seed + 11)
    else:
        gated_yield_30 = {"n": 0, "mean": 0.0, "ci95": [0.0, 0.0]}
        gated_yield_32 = {"n": 0, "mean": 0.0, "ci95": [0.0, 0.0]}

    # Length-stratified Axis 2 (both all-folded and strictly gated <30% & H>=3.0)
    by_length: dict[str, Any] = {}
    for bname, lo_l, hi_l in LENGTH_BINS:
        sub_all = [r for r in folded_rows if lo_l <= r["length"] <= hi_l]
        sub_gated = [r for r in gated_30 if lo_l <= r["length"] <= hi_l]
        by_length[bname] = {
            "all_folded": _summarize_subset_structure(sub_all, draws=min(draws, 2000), seed=seed + 20),
            "gated_lt30_h30": _summarize_subset_structure(sub_gated, draws=min(draws, 2000), seed=seed + 30),
        }

    meta_header: dict[str, Any] = {}
    if spec.arm_id in ARM_METADATA:
        meta_header = asdict(ARM_METADATA[spec.arm_id])

    return {
        "entry_key": spec.entry_key,
        "arm_id": spec.arm_id,
        "paradigm_group": spec.paradigm_group,
        "generation_regime": spec.generation_regime,
        "diffusion_mode": spec.diffusion_mode,
        "is_control": spec.is_control,
        "is_primary_mode": spec.is_primary_mode,
        "params_m": meta_header.get("params_m") if meta_header else spec.params_m,
        "family": meta_header.get("family"),
        "source_paths": loaded["source_paths"],
        "axis1_full_pool": {
            "n_pool": n_pool,
            "n_supported": n_sup,
            "valid_aa20_rate": valid_rate,
            "supported_rate": sup_rate,
            "budget_censored_rate": cap_rate,
            "length_min": int(min(r["length"] for r in supported_rows)) if n_sup else 0,
            "length_median": float(np.median([r["length"] for r in supported_rows])) if n_sup else 0.0,
            "length_max": int(max(r["length"] for r in supported_rows)) if n_sup else 0,
            "mean_entropy_bits": float(np.mean([r["entropy_bits"] for r in supported_rows])) if n_sup else 0.0,
            "frac_entropy_ge_3_0": float(np.mean([1.0 if r["entropy_bits"] >= 3.0 else 0.0 for r in supported_rows])) if n_sup else 0.0,
            "mean_distinct_5mer_fraction": float(np.mean([r["distinct_5mer_fraction"] for r in supported_rows])) if n_sup else 0.0,
            "near_duplicate_unique_group_rate": unique_group_rate,
            "near_duplicate_largest_group_share": largest_group_share,
            "near_duplicate_n_groups": int(nd_summary["n_groups"]),
            "pfam_any_family_rate_all": pfam_any_all,
            "pfam_complete_domain_rate_all": pfam_comp_all,
            "pfam_any_family_rate_supported": pfam_any_sup,
            "pfam_complete_domain_rate_supported": pfam_comp_sup,
            "n_distinct_pfam_families": distinct_pfams,
            "uniref50_mean_identity": float(np.mean(id_vals_sup)) if n_sup else 0.0,
            "uniref50_median_identity": float(np.median(id_vals_sup)) if n_sup else 0.0,
            "uniref50_frac_lt30_novel": frac_lt30,
            "uniref50_frac_30_to_70_homolog": frac_30_70,
            "uniref50_frac_gt70_near_copy": frac_gt70,
            "uniref50_frac_ge95_recitation": frac_ge95,
            "rep27_multi_stream_stability": rep27_summary.get(spec.arm_id),
        },
        "axis2_structure_joined": {
            "n_folded": n_folded,
            "all_folded": _summarize_subset_structure(folded_rows, draws=draws, seed=seed),
            "gated_lt30_h_ge_3_0": {
                **_summarize_subset_structure(gated_30, draws=draws, seed=seed + 5),
                "fraction_of_folded": float(len(gated_30) / n_folded) if n_folded else 0.0,
                "gated_yield_of_pool": gated_yield_30,
            },
            "gated_sensitivity_lt30_h_ge_3_2_nondup": {
                **_summarize_subset_structure(gated_32_nd, draws=draws, seed=seed + 6),
                "fraction_of_folded": float(len(gated_32_nd) / n_folded) if n_folded else 0.0,
                "gated_yield_of_pool": gated_yield_32,
            },
            "rejected_low_entropy_lt_3_0": {
                "n": len(low_h_collapse),
                "fraction_of_folded": float(len(low_h_collapse) / n_folded) if n_folded else 0.0,
                "mean_plddt": float(np.mean([r["mean_ca_plddt"] for r in low_h_collapse])) if low_h_collapse else None,
            },
            "homolog_ge30_subset": {
                "n": len(homolog_ge30),
                "fraction_of_folded": float(len(homolog_ge30) / n_folded) if n_folded else 0.0,
                "mean_plddt": float(np.mean([r["mean_ca_plddt"] for r in homolog_ge30])) if homolog_ge30 else None,
                "foldable_rate_plddt_ge_70": float(np.mean([1.0 if r["mean_ca_plddt"] >= 70.0 else 0.0 for r in homolog_ge30])) if homolog_ge30 else None,
            },
            "by_length_bin": by_length,
        },
    }


def _load_rep27_stability(repo_root: Path) -> dict[str, Any]:
    """Extract multi-stream Pfam recognition stability from generation_replication_20260927."""
    rep_path = repo_root / "results/R6/generation_replication_20260927/assessment_final/generation_assessment.json"
    if not rep_path.is_file():
        return {}
    doc = load_json_file(rep_path)
    out: dict[str, Any] = {}
    reports = doc.get("reports")
    if isinstance(reports, dict):
        for arm_id, rec in reports.items():
            if isinstance(rec, dict):
                out[arm_id] = {
                    "status": rec.get("status"),
                    "replicate_rates": rec.get("replicate_rates") or rec.get("any_profile_rates"),
                }
    elif isinstance(reports, list):
        for rec in reports:
            if isinstance(rec, dict) and rec.get("arm"):
                out[str(rec["arm"])] = {
                    "status": rec.get("status"),
                    "replicate_rates": rec.get("replicate_rates") or rec.get("any_profile_rates"),
                }
    # Supplement from profiles/replicate_{1,2}__<arm>__unconditioned/generation_profiles.json
    prof_root = repo_root / "results/R6/generation_replication_20260927/profiles"
    if prof_root.is_dir():
        by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pdir in sorted(prof_root.iterdir()):
            if not pdir.is_dir() or not pdir.name.endswith("__unconditioned"):
                continue
            gp_f = pdir / "generation_profiles.json"
            if not gp_f.is_file():
                continue
            parts = pdir.name.split("__")
            if len(parts) != 3:
                continue
            rep_name, arm_id, _ = parts
            gp_doc = load_json_file(gp_f)
            by_arm[arm_id].append(
                {
                    "replicate": rep_name,
                    "n_attempts": gp_doc.get("n_attempts"),
                    "any_profile_rate": gp_doc.get("any_profile_rate"),
                }
            )
        for arm_id, reps in by_arm.items():
            rates = [float(r["any_profile_rate"]) for r in reps if r["any_profile_rate"] is not None]
            out[arm_id] = {
                "n_replicates": len(reps),
                "replicates": reps,
                "mean_any_profile_rate": float(np.mean(rates)) if rates else None,
                "range_any_profile_rate": float(max(rates) - min(rates)) if len(rates) >= 2 else 0.0,
            }
    return out


def _build_stage4_anchored_controls(repo_root: Path) -> dict[str, Any]:
    """Assemble Stage 4: 18-arm R5a scaffold-anchoring ladder + ESM3 R5c + ZymCTRL/ProLLaMA label recall."""
    # 4a. R5a progressive scaffold anchoring (12 GP/ESM3 + 6 original ladder_scoped + composition-shuffle)
    gs = load_json_file(repo_root / "results/extensions/general_plm_20261120/general_plm_summary.json")
    comp = gs.get("companion_artifacts", {})

    r5a_gp_arms: dict[str, Any] = {}
    esm3_r5c_conditioned: dict[str, Any] = {}
    for key, meta in sorted(comp.items()):
        if not key.startswith("r5_structure_eval::"):
            continue
        label = key.split("::", 1)[1]
        src_p = _resolve_repo_path(repo_root, meta["source"])
        doc = load_json_file(src_p)
        rungs_out: dict[str, Any] = {}
        for lk, lv in sorted((doc.get("r5a_ladder") or {}).items()):
            rung = lk.split("|")[-1]
            rungs_out[rung] = {
                "condition": lv.get("condition"),
                "n_items": (lv.get("mean_plddt") or {}).get("n_items") if isinstance(lv.get("mean_plddt"), dict) else lv.get("n_folded"),
                "mean_plddt": (lv.get("mean_plddt") or {}).get("mean") if isinstance(lv.get("mean_plddt"), dict) else lv.get("mean_plddt"),
                "mean_ptm": (lv.get("mean_ptm") or {}).get("mean") if isinstance(lv.get("mean_ptm"), dict) else lv.get("mean_ptm"),
                "mean_tm_score_to_parent": (
                    (lv.get("mean_tm_score") or lv.get("mean_tm_score_to_parent") or {}).get("mean")
                    if isinstance(lv.get("mean_tm_score") or lv.get("mean_tm_score_to_parent"), dict)
                    else (lv.get("mean_tm_score") or lv.get("mean_tm_score_to_parent"))
                ),
            }
        r5a_gp_arms[label] = {
            "rungs": rungs_out,
            "n_rungs": len(rungs_out),
            "source_path": _rel_repo_path(repo_root, src_p),
        }
        if label == "r5_struct_esm3_1_4b" and doc.get("r5c_conditioned"):
            esm3_r5c_conditioned = doc["r5c_conditioned"]

    # Original 6-arm + composition-shuffle ladder from ladder_scoped_20261011
    ld_path = repo_root / "results/R6/ladder_scoped_20261011/primary/ladder_divergence.json"
    ld = load_json_file(ld_path)
    r5a_orig: dict[str, Any] = {}
    for _, cell in sorted((ld.get("cells") or {}).items()):
        prod = str(cell.get("product_arm", ""))
        sc = str(cell.get("scoring_arm", ""))
        if prod == "composition-shuffle":
            if sc != "dplm-650m":
                continue
        elif prod != sc:
            continue
        rung = str(cell.get("rung", ""))
        dists = cell.get("distributions") or {}
        r5a_orig.setdefault(
            prod,
            {"condition": cell.get("condition"), "rungs": {}, "source_path": str(ld_path.relative_to(repo_root))},
        )
        r5a_orig[prod]["rungs"][rung] = {
            "n_observations": cell.get("n_observations"),
            "mean_ca_plddt": (dists.get("mean_ca_plddt") or {}).get("mean"),
            "mean_tm_score_to_parent": (dists.get("tm_score_to_parent") or {}).get("mean"),
        }

    # 4b. Explicit Class Label Recall (ZymCTRL 14 EC classes + ProLLaMA 15 InterPro superfamilies)
    cg_path = repo_root / "results/R6/conditioned_generation_20260826/conditioned_generation.json"
    cg = load_json_file(cg_path)
    explicit_label: dict[str, Any] = {}
    for arm in ("zymctrl", "prollama"):
        pa = (cg.get("protein_arms") or {}).get(arm) or {}
        per_class = pa.get("per_class") or {}
        p_req = [float(c["p_requested"]) for c in per_class.values() if c.get("p_requested") is not None]
        p_mis = [float(c["p_mismatched"]) for c in per_class.values() if c.get("p_mismatched") is not None]
        p_flr_p2 = [
            float((c.get("p_floor") or {}).get("progen2-medium"))
            for c in per_class.values()
            if (c.get("p_floor") or {}).get("progen2-medium") is not None
        ]
        p_flr_pg2 = [
            float((c.get("p_floor") or {}).get("protgpt2"))
            for c in per_class.values()
            if (c.get("p_floor") or {}).get("protgpt2") is not None
        ]
        explicit_label[arm] = {
            "channel": pa.get("channel"),
            "n_admitted_classes": int(pa.get("n_admitted_classes") or len(per_class)),
            "primary_floor_arm": pa.get("primary_floor"),
            "mean_p_requested": float(np.mean(p_req)) if p_req else None,
            "median_p_requested": float(np.median(p_req)) if p_req else None,
            "max_p_requested": float(np.max(p_req)) if p_req else None,
            "n_classes_gt_0_10": int(sum(1 for v in p_req if v > 0.10)),
            "mean_p_mismatched": float(np.mean(p_mis)) if p_mis else None,
            "max_p_mismatched": float(np.max(p_mis)) if p_mis else None,
            "mean_p_floor_progen2_medium": float(np.mean(p_flr_p2)) if p_flr_p2 else None,
            "mean_p_floor_protgpt2": float(np.mean(p_flr_pg2)) if p_flr_pg2 else None,
            "per_class_summary": {
                ck: {
                    "label": cv.get("label"),
                    "p_requested": cv.get("p_requested"),
                    "p_mismatched": cv.get("p_mismatched"),
                    "p_floor": cv.get("p_floor"),
                }
                for ck, cv in sorted(per_class.items())
            },
            "source_path": str(cg_path.relative_to(repo_root)),
        }

    return {
        "progressive_scaffold_anchoring_r5a": {
            "n_gp_and_esm3_evals": len(r5a_gp_arms),
            "n_original_ladder_entries": len(r5a_orig),
            "gp_and_esm3_arms": r5a_gp_arms,
            "original_ladder_scoped_arms": r5a_orig,
            "esm3_r5c_multimodal_conditioning": esm3_r5c_conditioned,
        },
        "explicit_class_label_recall": explicit_label,
    }


def annotate_available_pools(
    repo_root: Path = REPO_ROOT,
    *,
    diamond_threads: int = 16,
    hmmscan_workers: int = 8,
    hmmscan_threads_per_worker: int = 2,
) -> dict[str, Any]:
    """Run incremental DIAMOND + Pfam hmmscan on all currently available pools (zero GPU)."""
    repo_root = Path(repo_root)
    loaded_list: list[dict[str, Any]] = []
    missing_entries: list[str] = []
    for spec in POOL_SPECS_34:
        loaded = load_entry_records(spec, repo_root, require_structures=False)
        if loaded is None:
            missing_entries.append(spec.entry_key)
        else:
            loaded_list.append(loaded)

    unique_seqs: dict[str, str] = {}
    for entry in loaded_list:
        for row in entry["rows"]:
            if row["valid_aa20"] and row["sequence_sha256"]:
                unique_seqs[row["sequence_sha256"]] = row["sequence"]

    dm_cache = ensure_diamond_annotations(unique_seqs, repo_root, threads=diamond_threads)
    pf_cache = ensure_pfam_annotations(
        unique_seqs,
        loaded_list,
        repo_root,
        workers=hmmscan_workers,
        threads_per_worker=hmmscan_threads_per_worker,
    )
    return {
        "n_entries_loaded": len(loaded_list),
        "missing_entries": missing_entries,
        "n_unique_valid_sequences": len(unique_seqs),
        "diamond_cache_size": len(dm_cache),
        "pfam_cache_size": len(pf_cache),
    }


def build_experiment_4_1_master(
    repo_root: Path = REPO_ROOT,
    *,
    draws: int = 10000,
    seed: int = 20261124,
    diamond_threads: int = 16,
    hmmscan_workers: int = 8,
    hmmscan_threads_per_worker: int = 2,
) -> dict[str, Any]:
    """Run any remaining Axis 1 annotations and assemble the complete 4-stage Experiment 4.1 JSON."""
    repo_root = Path(repo_root)
    loaded_list: list[dict[str, Any]] = [
        load_entry_records(spec, repo_root, require_structures=True)  # type: ignore[misc]
        for spec in POOL_SPECS_34
    ]
    assert len(loaded_list) == 34

    unique_seqs: dict[str, str] = {}
    for entry in loaded_list:
        for row in entry["rows"]:
            if row["valid_aa20"] and row["sequence_sha256"]:
                unique_seqs[row["sequence_sha256"]] = row["sequence"]

    dm_cache = ensure_diamond_annotations(unique_seqs, repo_root, threads=diamond_threads)
    pf_cache = ensure_pfam_annotations(
        unique_seqs,
        loaded_list,
        repo_root,
        workers=hmmscan_workers,
        threads_per_worker=hmmscan_threads_per_worker,
    )
    rep27_summary = _load_rep27_stability(repo_root)

    entries_by_key: dict[str, Any] = {}
    for idx, loaded in enumerate(loaded_list):
        rec = _aggregate_one_entry(
            loaded,
            dm_cache,
            pf_cache,
            rep27_summary,
            draws=draws,
            seed=seed + idx * 100,
        )
        entries_by_key[rec["entry_key"]] = rec

    # Stage 1: Evolutionary template baseline census (sorted by complete_domain rate desc)
    stage1_table = sorted(
        [
            {
                "entry_key": e["entry_key"],
                "arm_id": e["arm_id"],
                "paradigm_group": e["paradigm_group"],
                "is_control": e["is_control"],
                "is_primary_mode": e["is_primary_mode"],
                "n_pool": e["axis1_full_pool"]["n_pool"],
                "supported_rate": e["axis1_full_pool"]["supported_rate"],
                "pfam_any_family_rate_all": e["axis1_full_pool"]["pfam_any_family_rate_all"],
                "pfam_complete_domain_rate_all": e["axis1_full_pool"]["pfam_complete_domain_rate_all"],
                "n_distinct_pfam_families": e["axis1_full_pool"]["n_distinct_pfam_families"],
                "uniref50_frac_lt30_novel": e["axis1_full_pool"]["uniref50_frac_lt30_novel"],
                "uniref50_frac_30_to_70_homolog": e["axis1_full_pool"]["uniref50_frac_30_to_70_homolog"],
                "uniref50_frac_gt70_near_copy": e["axis1_full_pool"]["uniref50_frac_gt70_near_copy"],
                "uniref50_frac_ge95_recitation": e["axis1_full_pool"]["uniref50_frac_ge95_recitation"],
                "mean_entropy_bits": e["axis1_full_pool"]["mean_entropy_bits"],
                "mean_distinct_5mer_fraction": e["axis1_full_pool"]["mean_distinct_5mer_fraction"],
                "near_duplicate_unique_group_rate": e["axis1_full_pool"]["near_duplicate_unique_group_rate"],
            }
            for e in entries_by_key.values()
        ],
        key=lambda r: -(r["pfam_complete_domain_rate_all"] or 0.0),
    )

    # Stage 2: DPLM decoding artifact ablation (max_confidence vs uniform_random)
    stage2_pairs = []
    for base in ("dplm-150m", "dplm-650m", "dplm-3b"):
        u = entries_by_key[f"{base}|uniform_random"]
        m = entries_by_key[f"{base}|max_confidence"]
        stage2_pairs.append(
            {
                "arm_id": base,
                "params_m": u["params_m"],
                "max_confidence": {
                    "mean_entropy_bits": m["axis1_full_pool"]["mean_entropy_bits"],
                    "frac_entropy_ge_3_0": m["axis1_full_pool"]["frac_entropy_ge_3_0"],
                    "mean_distinct_5mer_fraction": m["axis1_full_pool"]["mean_distinct_5mer_fraction"],
                    "near_duplicate_unique_group_rate": m["axis1_full_pool"]["near_duplicate_unique_group_rate"],
                    "pfam_any_family_rate_all": m["axis1_full_pool"]["pfam_any_family_rate_all"],
                    "pfam_complete_domain_rate_all": m["axis1_full_pool"]["pfam_complete_domain_rate_all"],
                    "unfiltered_mean_plddt": m["axis2_structure_joined"]["all_folded"]["mean_plddt"],
                    "unfiltered_foldable_ge70": m["axis2_structure_joined"]["all_folded"]["foldable_rate_plddt_ge_70"],
                    "gated_lt30_h30_n": m["axis2_structure_joined"]["gated_lt30_h_ge_3_0"]["n"],
                    "gated_lt30_h30_yield_of_pool": m["axis2_structure_joined"]["gated_lt30_h_ge_3_0"]["gated_yield_of_pool"],
                    "gated_lt30_h30_foldable_ge70": m["axis2_structure_joined"]["gated_lt30_h_ge_3_0"]["foldable_rate_plddt_ge_70"],
                },
                "uniform_random": {
                    "mean_entropy_bits": u["axis1_full_pool"]["mean_entropy_bits"],
                    "frac_entropy_ge_3_0": u["axis1_full_pool"]["frac_entropy_ge_3_0"],
                    "mean_distinct_5mer_fraction": u["axis1_full_pool"]["mean_distinct_5mer_fraction"],
                    "near_duplicate_unique_group_rate": u["axis1_full_pool"]["near_duplicate_unique_group_rate"],
                    "pfam_any_family_rate_all": u["axis1_full_pool"]["pfam_any_family_rate_all"],
                    "pfam_complete_domain_rate_all": u["axis1_full_pool"]["pfam_complete_domain_rate_all"],
                    "unfiltered_mean_plddt": u["axis2_structure_joined"]["all_folded"]["mean_plddt"],
                    "unfiltered_foldable_ge70": u["axis2_structure_joined"]["all_folded"]["foldable_rate_plddt_ge_70"],
                    "gated_lt30_h30_n": u["axis2_structure_joined"]["gated_lt30_h_ge_3_0"]["n"],
                    "gated_lt30_h30_yield_of_pool": u["axis2_structure_joined"]["gated_lt30_h_ge_3_0"]["gated_yield_of_pool"],
                    "gated_lt30_h30_foldable_ge70": u["axis2_structure_joined"]["gated_lt30_h_ge_3_0"]["foldable_rate_plddt_ge_70"],
                },
            }
        )

    # Stage 3: Cross-paradigm novel foldability ladder (ranked by gated_yield_of_pool desc)
    stage3_ranked = sorted(
        [
            {
                "entry_key": e["entry_key"],
                "arm_id": e["arm_id"],
                "paradigm_group": e["paradigm_group"],
                "generation_regime": e["generation_regime"],
                "params_m": e["params_m"],
                "is_control": e["is_control"],
                "is_primary_mode": e["is_primary_mode"],
                "n_folded": e["axis2_structure_joined"]["n_folded"],
                "unfiltered_mean_plddt": e["axis2_structure_joined"]["all_folded"]["mean_plddt"],
                "unfiltered_foldable_ge70": e["axis2_structure_joined"]["all_folded"]["foldable_rate_plddt_ge_70"],
                "unfiltered_mean_ptm": e["axis2_structure_joined"]["all_folded"]["mean_ptm"],
                "uniref50_frac_lt30_novel": e["axis1_full_pool"]["uniref50_frac_lt30_novel"],
                "mean_entropy_bits": e["axis1_full_pool"]["mean_entropy_bits"],
                "pfam_complete_domain_rate_all": e["axis1_full_pool"]["pfam_complete_domain_rate_all"],
                "gated_lt30_h30_n": e["axis2_structure_joined"]["gated_lt30_h_ge_3_0"]["n"],
                "gated_lt30_h30_yield_of_pool": e["axis2_structure_joined"]["gated_lt30_h_ge_3_0"]["gated_yield_of_pool"],
                "gated_lt30_h30_conditional_foldable_ge70": e["axis2_structure_joined"]["gated_lt30_h_ge_3_0"]["foldable_rate_plddt_ge_70"],
                "gated_lt30_h30_mean_plddt": e["axis2_structure_joined"]["gated_lt30_h_ge_3_0"]["mean_plddt"],
                "gated_lt30_h30_mean_ptm": e["axis2_structure_joined"]["gated_lt30_h_ge_3_0"]["mean_ptm"],
                "gated_sensitivity_h32_nondup_yield": e["axis2_structure_joined"]["gated_sensitivity_lt30_h_ge_3_2_nondup"]["gated_yield_of_pool"],
            }
            for e in entries_by_key.values()
        ],
        key=lambda r: -(
            r["gated_lt30_h30_yield_of_pool"]["mean"]
            if r["gated_lt30_h30_yield_of_pool"]["mean"] is not None
            else -1.0
        ),
    )

    stage4 = _build_stage4_anchored_controls(repo_root)

    return {
        "schema": "unified_experiment_4_1_generation_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "n_unconditional_checkpoints": 29,
            "n_mode_entries_including_dplm_ablation": 32,
            "n_controls": 2,
            "n_total_rows": len(POOL_SPECS_34),
            "bootstrap_draws": int(draws),
            "bootstrap_seed": int(seed),
            "strict_novel_gate": "valid_aa20 AND uniref50_identity_over_query < 30% AND shannon_entropy_bits >= 3.0",
            "sensitivity_gate": "valid_aa20 AND uniref50_identity_over_query < 30% AND shannon_entropy_bits >= 3.2 AND 5mer_non_duplicate_rep",
            "pfam_complete_domain_coverage_threshold": COMPLETE_DOMAIN_COVERAGE,
            "near_duplicate_containment_threshold": NEAR_DUPLICATE_CONTAINMENT,
            "diamond_masking": 0,
        },
        "stage1_evolutionary_template_baseline": {
            "summary_note": "Full 29-checkpoint (32 mode-entries) + 2-control Pfam complete-domain, UniRef50 novelty, complexity & 5-mer deduplication census.",
            "ranked_by_pfam_complete_domain": stage1_table,
        },
        "stage2_decoding_artifact_ablation": {
            "summary_note": "Direct comparison of DPLM greedy max_confidence homopolymer collapse vs uniform_random absorbing diffusion across 150M / 650M / 3B.",
            "dplm_pairs": stage2_pairs,
        },
        "stage3_cross_paradigm_novel_foldability_ladder": {
            "summary_note": "Full 34-row ranked table separating homolog-copying and homopolymer-collapsing arms from true novel natural-entropy de novo foldability.",
            "ranked_by_gated_novel_foldable_yield": stage3_ranked,
        },
        "stage4_anchored_vs_unanchored_dual_controls": stage4,
        "entries_full_34": entries_by_key,
    }


def write_experiment_4_1_master(
    repo_root: Path = REPO_ROOT,
    out_dir: Path | None = None,
    *,
    draws: int = 10000,
    seed: int = 20261124,
    diamond_threads: int = 16,
    hmmscan_workers: int = 8,
    hmmscan_threads_per_worker: int = 2,
) -> dict[str, Any]:
    """Assemble and atomically write ``r5_experiment_4_1_unified_generation.json`` + ``.sha256``."""
    repo_root = Path(repo_root)
    target_dir = Path(out_dir) if out_dir is not None else (repo_root / UNIFIED_OUTPUT_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    doc = build_experiment_4_1_master(
        repo_root,
        draws=draws,
        seed=seed,
        diamond_threads=diamond_threads,
        hmmscan_workers=hmmscan_workers,
        hmmscan_threads_per_worker=hmmscan_threads_per_worker,
    )
    out_path = target_dir / "r5_experiment_4_1_unified_generation.json"
    digest = write_json_with_sha256(out_path, doc)
    return {"path": str(out_path), "sha256": digest, "doc": doc}
