"""Unified 30-model / 33-pool Experiment 4.3: Native Self-Likelihood Ranking,
Repetitiveness Trap, and Adverse Selection Inside Novel (<20% / <30%) Candidate Pools.

Expands the frozen E17 native-likelihood diagnostic (``protgpt2`` and ``zymctrl``)
to all 30 generative checkpoints (33 pools = 29 unconditional checkpoints [32 pools
including both ``uniform`` and ``maxconf`` DPLM decoding modes] + conditioned
``zymctrl``), plus a high-power ~2,500-sequence companion verification from
``objective_tuned_base_likelihood_20261114``.

Evaluates every pool across five declared candidate regimes:
  1. ``strictly_novel_lt20_natural_entropy`` (UniRef50 < 20% AND H >= 3.0 bits) — primary
  2. ``novel_lt30_natural_entropy``          (UniRef50 < 30% AND H >= 3.0 bits) — Exp 4.1 aligned
  3. ``novel_lt20_unfiltered``               (UniRef50 < 20%, no H gate)
  4. ``novel_lt30_unfiltered``               (UniRef50 < 30%, no H gate)
  5. ``whole_pool``                          (all UniRef50 strata)

For every pool & regime, computes (with 10,000-draw unit-bootstrap 95% CIs):
  - Spearman rank correlations between high likelihood (``-mean_nll`` / ``+mean_pll``)
    and ``pLDDT``, ``pTM``, ``distinct_5mer``, and ``homopolymer_frac``;
  - Truncation selection contrasts (``selected_mean - pool_mean`` and
    ``oracle_headroom_share``) at top 10%, 25%, 50%, and ``top_k = 10`` on
    ``pLDDT``, ``pTM``, and ``distinct_5mer``, classifying adverse vs. neutral vs.
    positive selection;
  - Frisch-Waugh-Lovell ``column_span`` control ladder via
    :mod:`src.capability.evaluation.likelihood_ranking`
    (``unadjusted -> length -> length_composition -> length_composition_novelty_family``).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing as mp
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from src.capability.core.io import sha256_file
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS, bootstrap_unit_floor
from src.capability.evaluation import generated_phenotype as gp
from src.capability.evaluation import likelihood_ranking as lr
from src.capability.core.protein_annotations import parse_hmmscan_table
from src.capability.generation.family_oracle import (
    HMMSCAN_RELATIVE,
    PFAM_RELATIVE,
    Oracle,
    scan_command,
)
from src.capability.general_plm.eval_utils import atomic_write_json
from src.capability.unified.roster import ARM_METADATA, REPO_ROOT, UNIFIED_OUTPUT_DIR

SCHEMA = "r6_experiment_4_3_native_likelihood_unified_v1"

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261011
FWL_BOOTSTRAP_DRAWS = 2000
E17_REPRODUCTION_TOLERANCE = 1e-12

NATURAL_ENTROPY_MIN_BITS = 3.0
SELECTION_BUDGETS: tuple[tuple[str, str, float | int], ...] = (
    ("top_10pct", "fraction", 0.10),
    ("top_25pct", "fraction", 0.25),
    ("top_50pct", "fraction", 0.50),
    ("top_k10", "count", 10),
)

REGIMES: tuple[tuple[str, float | None, float | None], ...] = (
    ("strictly_novel_lt20_natural_entropy", 20.0, NATURAL_ENTROPY_MIN_BITS),
    ("novel_lt30_natural_entropy", 30.0, NATURAL_ENTROPY_MIN_BITS),
    ("novel_lt20_unfiltered", 20.0, None),
    ("novel_lt30_unfiltered", 30.0, None),
    ("whole_pool", None, None),
)

AA20_SET = frozenset(lr.AA20)

# Relative paths to frozen immutable inputs.
EXP4_3_WORK_REL = Path(UNIFIED_OUTPUT_DIR) / "exp4_3_self_likelihoods"
EXP4_1_CACHE_REL = Path(UNIFIED_OUTPUT_DIR) / "exp4_1_cache"
EXP4_1_POOLS_REL = Path(UNIFIED_OUTPUT_DIR) / "exp4_1_pools"

DIAMOND_BIN_REL = Path("external/tools/diamond/diamond")
UNIREF50_DB_REL = Path("data/homology_db/uniref50_full.dmnd")
ORACLE_ROOT_REL = Path("external/tools/r227-oracle")

E17_FROZEN_RANKING_REL = Path("results/R6/e17_column_span_20261012/likelihood_ranking.json")
E17_COHORT_DIR_REL = Path("results/R6/generation_evaluation_20261008")
E17_NOVELTY_REL = Path("results/R6/generation_novelty_20261008/generated_novelty.jsonl")
E17_RECOG_DIR_REL = Path("results/audit_20261012/e17-column-span-recovery")
E17_STRUCT_DIRS_REL: tuple[Path, ...] = (
    Path("results/audit_20261012/e17-recovery/geneval_e14_fold_shard0_gpu2"),
    Path("results/audit_20261012/e17-recovery/geneval_e14_fold_shard1_gpu3"),
)
E17_LIK_DIRS_REL: dict[str, Path] = {
    "protgpt2": Path(
        "results/external_baseline/20261008091454_70777906ad8c/geneval_e17_likelihood_protgpt2_gpu2"
    ),
    "zymctrl": Path(
        "results/external_baseline/20261008091454_70777906ad8c/geneval_e17_likelihood_zymctrl_gpu3"
    ),
}

# High-power companion paths (objective_tuned_base_likelihood_20261114)
OT_NN_TSV_REL = Path(
    "results/external_baseline_analysis/length_adjusted_contrasts_20261110/"
    "eighteen_shard/natural_neighbour/per_structure_nearest_neighbour.tsv"
)
OT_COHORT_REL = Path(
    "results/external_baseline_analysis/length_adjusted_contrasts_20261110/input/fold_cohort.jsonl"
)
OT_LIK_DIR_REL = Path(
    "results/extensions/objective_tuned_endpoints_20261112/selector/own_model_likelihood"
)
OT_RECOG_REL = Path(
    "results/extensions/objective_tuned_endpoints_20261111/recognition/recognition_flags.jsonl"
)

# Canonical 33-pool ordering (30 unique generative checkpoints = 29 unconditional + zymctrl).
POOL_METADATA: dict[str, dict[str, Any]] = {
    # 17 EXP-R2-246 / E17 pre-registered causal/joint models (+ progen3-3b + zymctrl = 19)
    "protgpt2": {"arm_id": "protgpt2", "paradigm": "protein_causal", "family": "ProtGPT2", "source_kind": "e17_frozen", "conditioned": False},
    "zymctrl": {"arm_id": "zymctrl", "paradigm": "protein_causal_conditioned", "family": "ZymCTRL", "source_kind": "e17_frozen", "conditioned": True},
    "progen2-small": {"arm_id": "progen2-small", "paradigm": "protein_causal", "family": "ProGen2", "source_kind": "gpu_scored", "conditioned": False},
    "progen2-base": {"arm_id": "progen2-base", "paradigm": "protein_causal", "family": "ProGen2", "source_kind": "gpu_scored", "conditioned": False},
    "progen2-medium": {"arm_id": "progen2-medium", "paradigm": "protein_causal", "family": "ProGen2", "source_kind": "gpu_scored", "conditioned": False},
    "progen2-large": {"arm_id": "progen2-large", "paradigm": "protein_causal", "family": "ProGen2", "source_kind": "gpu_scored", "conditioned": False},
    "progen2-xlarge": {"arm_id": "progen2-xlarge", "paradigm": "protein_causal", "family": "ProGen2", "source_kind": "gpu_scored", "conditioned": False},
    "progen3-112m": {"arm_id": "progen3-112m", "paradigm": "protein_causal", "family": "ProGen3", "source_kind": "gpu_scored", "conditioned": False},
    "progen3-219m": {"arm_id": "progen3-219m", "paradigm": "protein_causal", "family": "ProGen3", "source_kind": "gpu_scored", "conditioned": False},
    "progen3-339m": {"arm_id": "progen3-339m", "paradigm": "protein_causal", "family": "ProGen3", "source_kind": "gpu_scored", "conditioned": False},
    "progen3-762m": {"arm_id": "progen3-762m", "paradigm": "protein_causal", "family": "ProGen3", "source_kind": "gpu_scored", "conditioned": False},
    "progen3-1b": {"arm_id": "progen3-1b", "paradigm": "protein_causal", "family": "ProGen3", "source_kind": "gpu_scored", "conditioned": False},
    "progen3-3b": {"arm_id": "progen3-3b", "paradigm": "protein_causal", "family": "ProGen3", "source_kind": "gpu_scored", "conditioned": False},
    "protgpt3-112m": {"arm_id": "protgpt3-112m", "paradigm": "protein_causal", "family": "ProtGPT3", "source_kind": "gpu_scored", "conditioned": False},
    "protgpt3-1.3b": {"arm_id": "protgpt3-1.3b", "paradigm": "protein_causal", "family": "ProtGPT3", "source_kind": "gpu_scored", "conditioned": False},
    "protgpt3-10b": {"arm_id": "protgpt3-10b", "paradigm": "protein_causal", "family": "ProtGPT3", "source_kind": "gpu_scored", "conditioned": False},
    "proteinglm-7b-clm": {"arm_id": "proteinglm-7b-clm", "paradigm": "protein_causal", "family": "ProteinGLM", "source_kind": "gpu_scored", "conditioned": False},
    "rita-xl": {"arm_id": "rita-xl", "paradigm": "protein_causal", "family": "RITA", "source_kind": "gpu_scored", "conditioned": False},
    "dayhoff-170m": {"arm_id": "dayhoff-170m", "paradigm": "protein_hybrid_ssm", "family": "Dayhoff", "source_kind": "gpu_scored", "conditioned": False},
    "instructprotein": {"arm_id": "instructprotein", "paradigm": "joint_protein_text", "family": "InstructProtein", "source_kind": "gpu_scored", "conditioned": False},
    "prollama-stage-1": {"arm_id": "prollama-stage-1", "paradigm": "joint_protein_text", "family": "ProLLaMA", "source_kind": "gpu_scored", "conditioned": False},
    "prollama": {"arm_id": "prollama", "paradigm": "joint_protein_text", "family": "ProLLaMA", "source_kind": "gpu_scored", "conditioned": False},
    "galactica-125m": {"arm_id": "galactica-125m", "paradigm": "joint_protein_text", "family": "Galactica", "source_kind": "gpu_scored", "conditioned": False},
    "galactica-1.3b": {"arm_id": "galactica-1.3b", "paradigm": "joint_protein_text", "family": "Galactica", "source_kind": "gpu_scored", "conditioned": False},
    "galactica-6.7b": {"arm_id": "galactica-6.7b", "paradigm": "joint_protein_text", "family": "Galactica", "source_kind": "gpu_scored", "conditioned": False},
    "galactica-30b": {"arm_id": "galactica-30b", "paradigm": "joint_protein_text", "family": "Galactica", "source_kind": "gpu_scored", "conditioned": False},
    "dplm-150m__uniform": {"arm_id": "dplm-150m", "paradigm": "protein_diffusion", "family": "DPLM", "source_kind": "gpu_scored", "conditioned": False, "decoding_mode": "uniform_random"},
    "dplm-150m__maxconf": {"arm_id": "dplm-150m", "paradigm": "protein_diffusion", "family": "DPLM", "source_kind": "gpu_scored", "conditioned": False, "decoding_mode": "max_confidence"},
    "dplm-650m__uniform": {"arm_id": "dplm-650m", "paradigm": "protein_diffusion", "family": "DPLM", "source_kind": "gpu_scored", "conditioned": False, "decoding_mode": "uniform_random"},
    "dplm-650m__maxconf": {"arm_id": "dplm-650m", "paradigm": "protein_diffusion", "family": "DPLM", "source_kind": "gpu_scored", "conditioned": False, "decoding_mode": "max_confidence"},
    "dplm-3b__uniform": {"arm_id": "dplm-3b", "paradigm": "protein_diffusion", "family": "DPLM", "source_kind": "gpu_scored", "conditioned": False, "decoding_mode": "uniform_random"},
    "dplm-3b__maxconf": {"arm_id": "dplm-3b", "paradigm": "protein_diffusion", "family": "DPLM", "source_kind": "gpu_scored", "conditioned": False, "decoding_mode": "max_confidence"},
    "esm3-1.4b": {"arm_id": "esm3-1.4b", "paradigm": "protein_multimodal_diffusion", "family": "ESM3", "source_kind": "gpu_scored", "conditioned": False},
}
PRIMARY_33_POOLS: tuple[str, ...] = tuple(POOL_METADATA.keys())


# --------------------------------------------------------------------------- #
# Sequence complexity & repetitiveness features
# --------------------------------------------------------------------------- #


def seq_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def is_canonical_aa20(sequence: str) -> bool:
    return bool(sequence) and all(ch in AA20_SET for ch in sequence)


def distinct_kmer_fraction(sequence: str, k: int = 5) -> float:
    """Fraction of overlapping k-mers that are distinct (1.0 = zero repetition)."""
    n = len(sequence)
    if n < k:
        return 1.0
    total = n - k + 1
    return float(len({sequence[i : i + k] for i in range(total)}) / total)


def sequence_complexity_features(sequence: str) -> dict[str, float | int]:
    """Compute exact Shannon entropy (bits), longest homopolymer run/frac, and distinct 5-mer frac."""
    seq = sequence.strip().upper()
    length = len(seq)
    if length == 0:
        raise ValueError("Empty sequence")
    comp = lr.aa20_composition(seq)
    entropy = lr.composition_entropy_bits(comp)
    best_run = int(gp.longest_homopolymer_run(seq))
    homo_frac = float(best_run) / float(length)
    return {
        "length": int(length),
        "entropy_bits": float(entropy),
        "homopolymer_frac": float(homo_frac),
        "longest_run": int(best_run),
        "distinct_5mer": distinct_kmer_fraction(seq, 5),
    }


# --------------------------------------------------------------------------- #
# E17 frozen loader & bit-exact (< 1e-12) reproduction gate
# --------------------------------------------------------------------------- #


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_e17_frozen_pools(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Load exact frozen E17 cohort, structure, novelty, Pfam recognition, and self-likelihoods."""
    cohort_rows = _read_jsonl(root / E17_COHORT_DIR_REL / "evaluation_cohort.jsonl")
    e17_decl = json.loads((root / E17_COHORT_DIR_REL / "e17_declaration.json").read_text(encoding="utf-8"))

    # Structure shards
    structures: dict[str, dict[str, Any]] = {}
    for rel in E17_STRUCT_DIRS_REL:
        for idx_path in sorted((root / rel).glob("index-*-of-*.jsonl")):
            for row in _read_jsonl(idx_path):
                st = row.get("structure") or {}
                structures[str(row["id"])] = {
                    "status": st.get("status"),
                    "plddt": st.get("mean_ca_plddt"),
                    "ptm": st.get("ptm"),
                }

    # Pfam recognition
    recog: dict[str, list[str]] = {}
    for row in _read_jsonl(root / E17_RECOG_DIR_REL / "family_recognition.jsonl"):
        fams = row.get("pfam_families") or []
        recog[str(row["id"])] = sorted(str(f) for f in fams)

    # UniRef50 novelty
    novelty: dict[str, float] = {}
    for row in _read_jsonl(root / E17_NOVELTY_REL):
        novelty[str(row["id"])] = float(row["nearest_corpus_identity"])

    pools_out: dict[str, list[dict[str, Any]]] = {}
    for arm, lik_rel in E17_LIK_DIRS_REL.items():
        lik_rows = _read_jsonl(root / lik_rel / "generated_likelihood.jsonl")
        lik_map = {str(r["id"]): r for r in lik_rows}
        pool_rows = [
            r
            for r in cohort_rows
            if r.get("arm") == arm and str(r["id"]) in lik_map
        ]
        if len(pool_rows) != len(lik_map):
            raise ValueError(f"E17 {arm}: cohort/likelihood size mismatch {len(pool_rows)} vs {len(lik_map)}")

        records: list[dict[str, Any]] = []
        for r in pool_rows:
            rid = str(r["id"])
            st = structures.get(rid) or {}
            if st.get("status") != "ok" or st.get("plddt") is None or st.get("ptm") is None:
                continue
            lk = lik_map[rid]
            seq = str(r["sequence"]).strip().upper()
            records.append(
                {
                    "pool_id": arm,
                    "arm_id": arm,
                    "seq_id": rid,
                    "sequence": seq,
                    "sequence_sha256": str(r.get("sequence_sha256") or seq_sha256(seq)),
                    "unit_id": str(r["class_key"]) if arm == "zymctrl" else rid,
                    "class_key": r.get("class_key"),
                    "plddt": float(st["plddt"]),
                    "ptm": float(st["ptm"]),
                    "mean_nll_per_scored_token": float(lk["mean_nll_per_token_nats"]),
                    "mean_nll_per_residue": float(lk["mean_nll_per_residue_nats"]),
                    "nll_sum_nats": float(lk["nll_sum_nats"]),
                    "scored_tokens": int(lk["scored_tokens"]),
                    "uniref50_identity": float(novelty[rid]),
                    "pfam_families": list(recog.get(rid, [])),
                }
            )
        pools_out[arm] = records
    return {"pools": pools_out, "e17_declaration": e17_decl}


def run_e17_reproduction_gate(
    e17_data: dict[str, Any],
    *,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Verify < 1e-12 reproduction of frozen E17 ``column_span`` ``likelihood_ranking.json``
    on ``protgpt2`` and ``zymctrl`` across both endpoints, all 3 FWL rungs, and raw selection.
    """
    frozen_path = root / E17_FROZEN_RANKING_REL
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    max_diff = 0.0
    checked_fields = 0
    arm_details: dict[str, Any] = {}

    for arm in ("protgpt2", "zymctrl"):
        rows = e17_data["pools"][arm]
        nll = np.asarray([r["mean_nll_per_scored_token"] for r in rows], dtype=np.float64)
        units = [r["unit_id"] for r in rows]
        feats = [
            lr.make_row_features(
                r["sequence"],
                len(r["sequence"]),
                nearest_corpus_identity=r["uniref50_identity"],
                families=r["pfam_families"],
            )
            for r in rows
        ]
        vocab = lr.family_vocabulary(feats)
        arm_max = 0.0

        for ep_name, field in (("mean_ca_plddt", "plddt"), ("ptm", "ptm")):
            y = np.asarray([r[field] for r in rows], dtype=np.float64)
            f_ep = frozen["arms"][arm]["endpoints"][ep_name]

            # 1. Whole-pool Spearman
            wp = lr.whole_pool_spearman(nll, y)["spearman"]
            f_wp = f_ep["whole_pool_spearman"]["spearman"]
            diff = abs(float(wp) - float(f_wp))
            arm_max = max(arm_max, diff)
            checked_fields += 1

            # 2. All 3 FWL column_span rungs
            for rung in lr.RUNGS:
                rung_vocab = vocab if rung.uses_family else ()
                design, _ = lr.build_design(feats, rung, rung_vocab)
                ps = lr.partial_spearman(
                    nll, y, design, units=units, rank_policy=lr.RANK_POLICY_COLUMN_SPAN
                )["spearman"]
                f_ps = f_ep["control_ladder"][rung.name]["partial_spearman"]["spearman"]
                if ps is None or f_ps is None:
                    if ps != f_ps:
                        raise RuntimeError(f"E17 gate status mismatch on {arm}/{ep_name}/{rung.name}")
                else:
                    arm_max = max(arm_max, abs(float(ps) - float(f_ps)))
                    checked_fields += 1

            # 3. Raw selection at all E17 fractions
            for frac in lr.SELECTION_FRACTIONS:
                key = f"{frac:.2f}"
                sel = lr.raw_selection_block(nll, y, fraction=frac)
                f_sel = f_ep["selection"][key]["raw"]
                for m_k in ("selected_mean", "pool_expected_random", "oracle_bound", "gain_over_random"):
                    arm_max = max(arm_max, abs(float(sel[m_k]) - float(f_sel[m_k])))
                    checked_fields += 1

        max_diff = max(max_diff, arm_max)
        arm_details[arm] = {"n_pool": len(rows), "max_abs_diff": arm_max}

    if max_diff >= E17_REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"E17 reproduction gate FAILED: max_abs_diff={max_diff:g} >= {E17_REPRODUCTION_TOLERANCE:g}"
        )
    return {
        "passed": True,
        "tolerance": E17_REPRODUCTION_TOLERANCE,
        "max_abs_diff": max_diff,
        "checked_scalars": checked_fields,
        "frozen_artifact_path": str(E17_FROZEN_RANKING_REL),
        "frozen_artifact_sha256": sha256_file(frozen_path),
        "arms": arm_details,
    }


# --------------------------------------------------------------------------- #
# Axis 1 Annotation Pipeline (DIAMOND UniRef50 + pinned Pfam-A hmmscan)
# --------------------------------------------------------------------------- #


def _seed_diamond_from_repo(root: Path, cache: dict[str, dict[str, Any]]) -> int:
    added = 0
    # 1. Read-only reuse of dev-memory-theory's exp4_1_cache if present
    exp41_dm = root / EXP4_1_CACHE_REL / "diamond_uniref50_by_sha256.json"
    if exp41_dm.is_file():
        try:
            d = json.loads(exp41_dm.read_text(encoding="utf-8"))
            for sha, rec in d.items():
                if sha not in cache and "identity_over_query" in rec:
                    cache[sha] = {
                        "identity_over_query": float(rec["identity_over_query"]),
                        "subject": rec.get("subject"),
                        "evalue": rec.get("evalue"),
                        "source": f"exp4_1_cache:{rec.get('source', 'diamond')}",
                    }
                    added += 1
        except Exception:
            pass

    # 2. Frozen generation_novelty_20261008
    ev_p = root / E17_COHORT_DIR_REL / "evaluation_cohort.jsonl"
    nov_p = root / E17_NOVELTY_REL
    if ev_p.is_file() and nov_p.is_file():
        id_to_sha = {}
        for r in _read_jsonl(ev_p):
            seq = str(r.get("sequence") or "").strip().upper()
            if seq:
                id_to_sha[str(r["id"])] = str(r.get("sequence_sha256") or seq_sha256(seq))
        for r in _read_jsonl(nov_p):
            sha = id_to_sha.get(str(r["id"]))
            if sha and sha not in cache:
                cache[sha] = {
                    "identity_over_query": float(r["nearest_corpus_identity"]),
                    "subject": r.get("nearest_corpus_subject"),
                    "evalue": r.get("nearest_corpus_evalue"),
                    "source": "generation_novelty_20261008",
                }
                added += 1

    # 3. Frozen objective_tuned per_structure_nearest_neighbour.tsv
    ot_nn = root / OT_NN_TSV_REL
    if ot_nn.is_file():
        with ot_nn.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                sha = row.get("sequence_sha256")
                val = row.get("identity_over_query")
                if sha and val not in (None, "") and sha not in cache:
                    cache[sha] = {
                        "identity_over_query": float(val),
                        "subject": None,
                        "evalue": float(row["evalue"]) if row.get("evalue") not in (None, "") else None,
                        "source": "objective_tuned_nearest_neighbour_20261110",
                    }
                    added += 1
    return added


def ensure_diamond_uniref50(
    unique_seqs: Mapping[str, str],
    *,
    root: Path = REPO_ROOT,
    threads: int = 32,
    block_size: float = 8.0,
) -> dict[str, dict[str, Any]]:
    """Query uncached unique sequences against uniref50_full.dmnd (--masking 0 --very-sensitive)."""
    ann_dir = root / EXP4_3_WORK_REL / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    cache_path = ann_dir / "diamond_uniref50_by_sha256.json"
    cache: dict[str, dict[str, Any]] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    )
    seeded = _seed_diamond_from_repo(root, cache)
    if seeded:
        print(f"[exp4.3-diamond] seeded {seeded} cached UniRef50 hits (total={len(cache)})", flush=True)

    missing = {
        sha: seq
        for sha, seq in unique_seqs.items()
        if sha not in cache and is_canonical_aa20(seq) and len(seq) >= 10
    }
    if not missing:
        atomic_write_json(cache_path, cache)
        return cache

    diamond_bin = root / DIAMOND_BIN_REL
    db_path = root / UNIREF50_DB_REL
    if not diamond_bin.is_file() or not db_path.is_file():
        raise FileNotFoundError(f"DIAMOND binary or DB missing: {diamond_bin}, {db_path}")

    t0 = time.monotonic()
    print(
        f"[exp4.3-diamond] running blastp on {len(missing)} unique sequences "
        f"(--very-sensitive --masking 0 --threads {threads})...",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="exp4_3_diamond_") as tmpdir:
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
            "--very-sensitive",
            "--evalue",
            "1e-3",
            "--max-target-seqs",
            "25",
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
                    "source": "diamond_uniref50_full_masking0",
                }
            else:
                ident, subj, ev = hit
                cache[sha] = {
                    "identity_over_query": float(ident),
                    "subject": subj,
                    "evalue": float(ev),
                    "source": "diamond_uniref50_full_masking0",
                }

    atomic_write_json(cache_path, cache)
    print(
        f"[exp4.3-diamond] finished {len(missing)} queries in {time.monotonic() - t0:.1f}s "
        f"(cache={len(cache)})",
        flush=True,
    )
    return cache


def _run_hmmscan_shard(args: tuple[str, str, str, list[tuple[str, str]], int]) -> dict[str, dict[str, Any]]:
    oracle_root_s, hmmscan_s, pfam_s, shard_items, cpu_threads = args
    oracle = Oracle(
        hmmscan=Path(hmmscan_s),
        pfam=Path(pfam_s),
        digests={},
        root=Path(oracle_root_s),
    )
    out: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="exp4_3_pfam_") as tmpdir:
        tmp = Path(tmpdir)
        fa = tmp / "shard.faa"
        tbl = tmp / "shard.tbl"
        domtbl = tmp / "shard.domtbl"
        with fa.open("w", encoding="utf-8") as fh:
            for sha, seq in shard_items:
                fh.write(f">{sha}\n{seq}\n")
        cmd = scan_command(oracle, fa, tbl, domtbl, threads=cpu_threads)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"hmmscan shard failed: {res.stderr[-2000:]}")
        tbl_hits = parse_hmmscan_table(tbl)
        for sha, _ in shard_items:
            fams = sorted({str(e["accession_unversioned"]) for e in tbl_hits.get(sha, [])})
            out[sha] = {
                "any_family": bool(fams),
                "families": fams,
                "source": "pinned_pfam_hmmscan_cut_ga",
            }
    return out


def _seed_pfam_from_repo(root: Path, cache: dict[str, dict[str, Any]]) -> int:
    added = 0
    # 1. Read-only reuse of dev-memory-theory's exp4_1_cache if present
    exp41_pf = root / EXP4_1_CACHE_REL / "pfam_hmmscan_by_sha256.json"
    if exp41_pf.is_file():
        try:
            d = json.loads(exp41_pf.read_text(encoding="utf-8"))
            for sha, rec in d.items():
                if sha not in cache:
                    fams = sorted(set(str(f) for f in (rec.get("families") or [])))
                    cache[sha] = {
                        "any_family": bool(fams),
                        "families": fams,
                        "source": f"exp4_1_cache:{rec.get('source', 'pfam')}",
                    }
                    added += 1
        except Exception:
            pass

    # 2. Frozen E17 family_recognition.jsonl
    rec_p = root / E17_RECOG_DIR_REL / "family_recognition.jsonl"
    if rec_p.is_file():
        for r in _read_jsonl(rec_p):
            sha = str(r.get("sequence_sha256") or "")
            if sha and sha not in cache:
                fams = sorted(set(str(f) for f in (r.get("pfam_families") or [])))
                cache[sha] = {
                    "any_family": bool(fams),
                    "families": fams,
                    "source": "e17_family_recognition",
                }
                added += 1

    # 3. Frozen objective_tuned recognition_flags.jsonl
    ot_p = root / OT_RECOG_REL
    if ot_p.is_file():
        for r in _read_jsonl(ot_p):
            sha = str(r.get("sequence_sha256") or "")
            if sha and sha not in cache:
                fams = sorted(set(str(f) for f in (r.get("families") or [])))
                cache[sha] = {
                    "any_family": bool(fams),
                    "families": fams,
                    "source": "objective_tuned_recognition_flags",
                }
                added += 1
    return added


def ensure_pfam_annotations(
    unique_seqs: Mapping[str, str],
    *,
    root: Path = REPO_ROOT,
    workers: int = 16,
    threads_per_worker: int = 2,
) -> dict[str, dict[str, Any]]:
    """Run pinned Pfam-A hmmscan (--cut_ga) for all uncached unique SHA-256s."""
    ann_dir = root / EXP4_3_WORK_REL / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    cache_path = ann_dir / "pfam_hmmscan_by_sha256.json"
    cache: dict[str, dict[str, Any]] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    )
    seeded = _seed_pfam_from_repo(root, cache)
    if seeded:
        print(f"[exp4.3-pfam] seeded {seeded} cached Pfam records (total={len(cache)})", flush=True)

    missing = sorted(
        (sha, seq)
        for sha, seq in unique_seqs.items()
        if sha not in cache and is_canonical_aa20(seq) and len(seq) >= 16
    )
    if not missing:
        atomic_write_json(cache_path, cache)
        return cache

    oracle_root = root / ORACLE_ROOT_REL
    hmmscan_bin = oracle_root / HMMSCAN_RELATIVE
    pfam_db = oracle_root / PFAM_RELATIVE
    if not hmmscan_bin.is_file() or not pfam_db.is_file():
        raise FileNotFoundError(f"Pinned Pfam oracle missing under {oracle_root}")

    n_workers = max(1, min(int(workers), 16))
    n_shards = min(max(n_workers * 2, 1), len(missing))
    shards: list[list[tuple[str, str]]] = [[] for _ in range(n_shards)]
    for i, item in enumerate(missing):
        shards[i % n_shards].append(item)

    t0 = time.monotonic()
    print(
        f"[exp4.3-pfam] scanning {len(missing)} unique sequences across {n_shards} shards "
        f"({n_workers} workers x {threads_per_worker} threads)...",
        flush=True,
    )
    tasks = [
        (str(oracle_root), str(hmmscan_bin), str(pfam_db), shard, int(threads_per_worker))
        for shard in shards
    ]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        for shard_out in pool.map(_run_hmmscan_shard, tasks):
            cache.update(shard_out)

    atomic_write_json(cache_path, cache)
    print(
        f"[exp4.3-pfam] finished {len(missing)} sequences in {time.monotonic() - t0:.1f}s "
        f"(cache={len(cache)})",
        flush=True,
    )
    return cache


# --------------------------------------------------------------------------- #
# Vectorized 10,000-draw unit bootstrap for Spearman & Truncation Selection
# --------------------------------------------------------------------------- #


def _spearman_1d(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    denom = float(np.sqrt(np.dot(dx, dx) * np.dot(dy, dy)))
    if denom <= 0.0:
        return None
    return float(np.clip(np.dot(dx, dy) / denom, -1.0, 1.0))


def _select_count(n: int, mode: str, value: float | int) -> int:
    if mode == "fraction":
        return int(gp.selection_count(n, float(value)))
    return max(1, min(int(value), n))


def _unit_resample_draw_indices(
    units: Sequence[str],
    *,
    n_draws: int,
    seed: int,
) -> tuple[list[np.ndarray] | np.ndarray, bool, int]:
    """Generate unit-bootstrap row-index arrays.

    Fast path: when every row is its own unit (all unconditional pools), returns a
    2-D integer array ``(n_draws, n_rows)`` for ultra-fast vectorized statistics.
    Clustered path (``zymctrl`` by ``class_key``): returns a list of 1-D index arrays
    preserving full cluster multiplicity.
    """
    n_rows = len(units)
    unique_units = sorted(set(units))
    n_units = len(unique_units)
    rng = np.random.default_rng(seed)
    if n_units == n_rows:
        return rng.integers(0, n_rows, size=(n_draws, n_rows), dtype=np.int32), True, n_units
    by_unit = lr.index_by_unit(units)
    u_arr = np.asarray(unique_units, dtype=object)
    draws: list[np.ndarray] = []
    for _ in range(n_draws):
        chosen = rng.choice(u_arr, size=n_units, replace=True)
        draws.append(lr.indices_for_unit_draw(by_unit, chosen))
    return draws, False, n_units


def _bootstrap_spearman_and_selection(
    nll: np.ndarray,
    endpoints: Mapping[str, np.ndarray],
    units: Sequence[str],
    *,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute exact point estimates + 10,000-draw unit-bootstrap 95% CIs for:
    - Spearman(high_likelihood = -nll, endpoint) for every endpoint
    - Truncation selection contrasts at top 10%, 25%, 50%, and top_k=10 for plddt, ptm, distinct_5mer
    """
    n_rows = int(nll.size)
    high_lik = -nll
    n_units = len(set(units))
    floor = bootstrap_unit_floor(n_units, minimum_units=MINIMUM_BOOTSTRAP_UNITS)

    spearman_out: dict[str, Any] = {}
    for name, arr in endpoints.items():
        pt = _spearman_1d(high_lik, arr)
        spearman_out[name] = {
            "point": pt,
            "ci95": None,
            "status": "measured" if pt is not None else "unavailable",
            "verdict": "unresolved",
        }

    sel_targets = ("plddt", "ptm", "distinct_5mer")
    selection_out: dict[str, dict[str, Any]] = {t: {} for t in sel_targets}

    if n_rows > 0:
        order_nll = np.argsort(nll, kind="stable")
        for b_name, b_mode, b_val in SELECTION_BUDGETS:
            k_sel = _select_count(n_rows, b_mode, b_val)
            chosen = order_nll[:k_sel]
            for t in sel_targets:
                y = endpoints[t]
                pool_mean = float(y.mean())
                sel_mean = float(y[chosen].mean())
                gain = sel_mean - pool_mean
                oracle = np.argsort(-y, kind="stable")[:k_sel]
                oracle_mean = float(y[oracle].mean())
                oracle_gap = oracle_mean - pool_mean
                headroom = float(gain / oracle_gap) if oracle_gap > 1e-12 else None
                selection_out[t][b_name] = {
                    "n_pool": n_rows,
                    "n_selected": int(k_sel),
                    "pool_mean": pool_mean,
                    "selected_mean": sel_mean,
                    "gain_over_random": float(gain),
                    "gain_ci95": None,
                    "oracle_bound": oracle_mean,
                    "oracle_gap": float(oracle_gap),
                    "oracle_headroom_share": headroom,
                    "headroom_ci95": None,
                    "point_adverse": bool(gain < 0.0),
                    "adverse_class": "below_unit_floor" if floor["degenerate"] else "neutral_unresolved",
                }

    if floor["degenerate"] or n_rows < MINIMUM_BOOTSTRAP_UNITS:
        return {
            "n_rows": n_rows,
            "n_units": n_units,
            "unit_floor": floor,
            "spearman": spearman_out,
            "truncation_selection": selection_out,
        }

    draws, is_row_level, _ = _unit_resample_draw_indices(units, n_draws=n_draws, seed=seed)

    # 1. Spearman bootstrap (vectorized in batches for row-level pools -> ~0.08s for 10,000 draws!)
    if is_row_level:
        idx_2d = draws  # shape (n_draws, n_rows)
        assert isinstance(idx_2d, np.ndarray)
        batch_sz = 1000
        sp_samples: dict[str, list[np.ndarray]] = {name: [] for name in endpoints}
        gain_samples: dict[str, dict[str, list[np.ndarray]]] = {
            t: {b_name: [] for b_name, _, _ in SELECTION_BUDGETS} for t in sel_targets
        }
        headroom_samples: dict[str, dict[str, list[np.ndarray]]] = {
            t: {b_name: [] for b_name, _, _ in SELECTION_BUDGETS} for t in sel_targets
        }

        for start in range(0, n_draws, batch_sz):
            b_idx = idx_2d[start : start + batch_sz]  # (B, N)
            B = b_idx.shape[0]
            nll_b = nll[b_idx]  # (B, N)
            hl_b = -nll_b

            # Ranks of high_likelihood across axis=1
            r_hl = rankdata(hl_b, method="average", axis=1)
            r_hl = r_hl - r_hl.mean(axis=1, keepdims=True)
            norm_hl = np.sqrt(np.sum(r_hl * r_hl, axis=1))

            for name, arr in endpoints.items():
                y_b = arr[b_idx]
                r_y = rankdata(y_b, method="average", axis=1)
                r_y = r_y - r_y.mean(axis=1, keepdims=True)
                denom = norm_hl * np.sqrt(np.sum(r_y * r_y, axis=1))
                with np.errstate(invalid="ignore", divide="ignore"):
                    corrs = np.where(denom > 0.0, np.sum(r_hl * r_y, axis=1) / denom, np.nan)
                sp_samples[name].append(corrs)

            # Truncation selection re-selected inside every replicate
            ord_nll = np.argsort(nll_b, axis=1, kind="stable")
            row_arange = np.arange(B)[:, None]
            for t in sel_targets:
                y_b = endpoints[t][b_idx]
                pool_m = y_b.mean(axis=1)
                y_sorted_desc = np.sort(y_b, axis=1)[:, ::-1]
                y_by_nll = y_b[row_arange, ord_nll]
                for b_name, b_mode, b_val in SELECTION_BUDGETS:
                    k_sel = _select_count(n_rows, b_mode, b_val)
                    sel_m = y_by_nll[:, :k_sel].mean(axis=1)
                    gain_b = sel_m - pool_m
                    ora_gap = y_sorted_desc[:, :k_sel].mean(axis=1) - pool_m
                    with np.errstate(invalid="ignore", divide="ignore"):
                        hr_b = np.where(ora_gap > 1e-12, gain_b / ora_gap, np.nan)
                    gain_samples[t][b_name].append(gain_b)
                    headroom_samples[t][b_name].append(hr_b)

        for name in endpoints:
            vals = np.concatenate(sp_samples[name])
            finite = vals[np.isfinite(vals)]
            if finite.size >= int(0.95 * n_draws):
                lo, hi = float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))
                spearman_out[name]["ci95"] = [lo, hi]
                spearman_out[name]["verdict"] = (
                    "positive" if lo > 0.0 else ("negative" if hi < 0.0 else "unresolved")
                )

        for t in sel_targets:
            for b_name, _, _ in SELECTION_BUDGETS:
                g_vals = np.concatenate(gain_samples[t][b_name])
                lo, hi = float(np.percentile(g_vals, 2.5)), float(np.percentile(g_vals, 97.5))
                rec = selection_out[t][b_name]
                rec["gain_ci95"] = [lo, hi]
                hr_vals = np.concatenate(headroom_samples[t][b_name])
                hr_fin = hr_vals[np.isfinite(hr_vals)]
                if hr_fin.size >= int(0.95 * n_draws):
                    rec["headroom_ci95"] = [
                        float(np.percentile(hr_fin, 2.5)),
                        float(np.percentile(hr_fin, 97.5)),
                    ]
                if hi < 0.0:
                    rec["adverse_class"] = "adverse_ci_negative"
                elif lo > 0.0:
                    rec["adverse_class"] = "positive_ci_positive"
                elif rec["gain_over_random"] < 0.0:
                    rec["adverse_class"] = "point_adverse_unresolved"
                else:
                    rec["adverse_class"] = "neutral_unresolved"
    else:
        # Clustered path (zymctrl by class_key)
        assert isinstance(draws, list)
        sp_c: dict[str, list[float]] = {name: [] for name in endpoints}
        g_c: dict[str, dict[str, list[float]]] = {
            t: {b_name: [] for b_name, _, _ in SELECTION_BUDGETS} for t in sel_targets
        }
        hr_c: dict[str, dict[str, list[float]]] = {
            t: {b_name: [] for b_name, _, _ in SELECTION_BUDGETS} for t in sel_targets
        }
        for idx in draws:
            nll_i = nll[idx]
            hl_i = -nll_i
            for name, arr in endpoints.items():
                v = _spearman_1d(hl_i, arr[idx])
                if v is not None:
                    sp_c[name].append(v)
            ord_i = np.argsort(nll_i, kind="stable")
            m_n = int(idx.size)
            for t in sel_targets:
                y_i = endpoints[t][idx]
                pm = float(y_i.mean())
                y_desc = np.sort(y_i)[::-1]
                for b_name, b_mode, b_val in SELECTION_BUDGETS:
                    k_sel = _select_count(m_n, b_mode, b_val)
                    gain_i = float(y_i[ord_i[:k_sel]].mean() - pm)
                    g_c[t][b_name].append(gain_i)
                    og = float(y_desc[:k_sel].mean() - pm)
                    if og > 1e-12:
                        hr_c[t][b_name].append(gain_i / og)

        for name in endpoints:
            if len(sp_c[name]) >= int(0.95 * n_draws):
                lo, hi = float(np.percentile(sp_c[name], 2.5)), float(np.percentile(sp_c[name], 97.5))
                spearman_out[name]["ci95"] = [lo, hi]
                spearman_out[name]["verdict"] = (
                    "positive" if lo > 0.0 else ("negative" if hi < 0.0 else "unresolved")
                )

        for t in sel_targets:
            for b_name, _, _ in SELECTION_BUDGETS:
                g_vals = g_c[t][b_name]
                lo, hi = float(np.percentile(g_vals, 2.5)), float(np.percentile(g_vals, 97.5))
                rec = selection_out[t][b_name]
                rec["gain_ci95"] = [lo, hi]
                if len(hr_c[t][b_name]) >= int(0.95 * n_draws):
                    rec["headroom_ci95"] = [
                        float(np.percentile(hr_c[t][b_name], 2.5)),
                        float(np.percentile(hr_c[t][b_name], 97.5)),
                    ]
                if hi < 0.0:
                    rec["adverse_class"] = "adverse_ci_negative"
                elif lo > 0.0:
                    rec["adverse_class"] = "positive_ci_positive"
                elif rec["gain_over_random"] < 0.0:
                    rec["adverse_class"] = "point_adverse_unresolved"
                else:
                    rec["adverse_class"] = "neutral_unresolved"

    return {
        "n_rows": n_rows,
        "n_units": n_units,
        "unit_floor": floor,
        "spearman": spearman_out,
        "truncation_selection": selection_out,
    }


def _fwl_column_span_ladder(
    rows: list[dict[str, Any]],
    nll: np.ndarray,
    *,
    n_bootstrap: int = FWL_BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run Frisch-Waugh-Lovell ``column_span`` control ladder on ``plddt`` and ``ptm``."""
    units = [r["unit_id"] for r in rows]
    feats = [
        lr.make_row_features(
            r["sequence"],
            len(r["sequence"]),
            nearest_corpus_identity=r["uniref50_identity"],
            families=r["pfam_families"],
        )
        for r in rows
    ]
    vocab = lr.family_vocabulary(feats)
    out: dict[str, Any] = {}
    for ep_name, field in (("plddt", "plddt"), ("ptm", "ptm")):
        y = np.asarray([r[field] for r in rows], dtype=np.float64)
        ep_out: dict[str, Any] = {
            "unadjusted": lr.whole_pool_spearman(nll, y),
            "rungs": {},
        }
        for rung in lr.RUNGS:
            if rung.uses_family and not lr.has_family_variation(vocab):
                # Still evaluate novelty-only fallback note? No: follow exact likelihood_ranking refusal!
                ep_out["rungs"][rung.name] = {
                    "status": "unavailable",
                    "reason": "family_categorical_has_no_variation",
                    "spearman": None,
                    "ci95": None,
                }
                continue
            rung_vocab = vocab if rung.uses_family else ()
            design, _ = lr.build_design(feats, rung, rung_vocab)
            pt = lr.partial_spearman(
                nll, y, design, units=units, rank_policy=lr.RANK_POLICY_COLUMN_SPAN
            )
            boot = None
            if pt["status"] == "measured" and n_bootstrap > 0:
                boot = lr.cluster_bootstrap_association(
                    nll,
                    y,
                    design,
                    units,
                    seed=seed,
                    n_bootstrap=n_bootstrap,
                    rank_policy=lr.RANK_POLICY_COLUMN_SPAN,
                )
            ci95 = boot["adjusted_ci95"] if boot and boot.get("status") == "measured" else None
            verdict = "unresolved"
            if ci95 is not None:
                verdict = "positive" if ci95[0] > 0.0 else ("negative" if ci95[1] < 0.0 else "unresolved")
            ep_out["rungs"][rung.name] = {
                "status": pt["status"],
                "reason": pt["reason"],
                "spearman": pt["spearman"],
                "n_rows": pt["n_rows"],
                "n_columns": pt["n_columns"],
                "rank": pt["rank"],
                "residual_df": pt["residual_df"],
                "ci95": ci95,
                "verdict": verdict,
                "n_valid_draws": boot["n_valid_draws"] if boot else 0,
            }
        out[ep_name] = ep_out
    return out


def evaluate_pool_across_regimes(
    pool_id: str,
    rows: list[dict[str, Any]],
    *,
    n_draws: int = BOOTSTRAP_DRAWS,
    fwl_draws: int = FWL_BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate one pool across all five regimes (Spearman, selection contrasts, FWL ladder)."""
    regimes_out: dict[str, Any] = {}
    n_total = len(rows)

    for reg_name, max_ident, min_ent in REGIMES:
        sub = [
            r
            for r in rows
            if (max_ident is None or r["uniref50_identity"] < max_ident)
            and (min_ent is None or r["entropy_bits"] >= min_ent)
        ]
        n_sub = len(sub)
        if n_sub == 0:
            regimes_out[reg_name] = {
                "n_rows": 0,
                "fraction_of_pool": 0.0,
                "n_units": 0,
                "status": "empty",
            }
            continue

        nll = np.asarray([r["mean_nll_per_scored_token"] for r in sub], dtype=np.float64)
        nll_res = np.asarray([r["mean_nll_per_residue"] for r in sub], dtype=np.float64)
        units = [r["unit_id"] for r in sub]
        endpoints = {
            "plddt": np.asarray([r["plddt"] for r in sub], dtype=np.float64),
            "ptm": np.asarray([r["ptm"] for r in sub], dtype=np.float64),
            "distinct_5mer": np.asarray([r["distinct_5mer"] for r in sub], dtype=np.float64),
            "homopolymer_frac": np.asarray([r["homopolymer_frac"] for r in sub], dtype=np.float64),
            "entropy_bits": np.asarray([r["entropy_bits"] for r in sub], dtype=np.float64),
            "length": np.asarray([r["length"] for r in sub], dtype=np.float64),
        }

        primary_stats = _bootstrap_spearman_and_selection(
            nll, endpoints, units, n_draws=n_draws, seed=seed
        )
        # Per-residue sensitivity (exact Spearman point estimate)
        per_res_sens = {
            "spearman_neg_nll_per_residue_vs_plddt": _spearman_1d(-nll_res, endpoints["plddt"]),
            "spearman_neg_nll_per_residue_vs_ptm": _spearman_1d(-nll_res, endpoints["ptm"]),
            "spearman_neg_nll_per_residue_vs_distinct_5mer": _spearman_1d(-nll_res, endpoints["distinct_5mer"]),
        }

        # Run FWL column-span control ladder on whole_pool & natural-entropy novel regimes
        run_fwl_boot = fwl_draws if reg_name in (
            "strictly_novel_lt20_natural_entropy",
            "novel_lt30_natural_entropy",
            "whole_pool",
        ) else 0
        fwl = (
            _fwl_column_span_ladder(sub, nll, n_bootstrap=run_fwl_boot, seed=seed)
            if n_sub >= MINIMUM_BOOTSTRAP_UNITS
            else None
        )

        regimes_out[reg_name] = {
            "status": "measured" if not primary_stats["unit_floor"]["degenerate"] else "below_unit_floor",
            "n_rows": n_sub,
            "fraction_of_pool": float(n_sub / n_total) if n_total else 0.0,
            "n_units": primary_stats["n_units"],
            "unit_floor": primary_stats["unit_floor"],
            "pool_means": {
                "plddt": float(endpoints["plddt"].mean()),
                "ptm": float(endpoints["ptm"].mean()),
                "distinct_5mer": float(endpoints["distinct_5mer"].mean()),
                "homopolymer_frac": float(endpoints["homopolymer_frac"].mean()),
                "entropy_bits": float(endpoints["entropy_bits"].mean()),
                "uniref50_identity": float(np.mean([r["uniref50_identity"] for r in sub])),
                "pfam_hit_rate": float(np.mean([1.0 if r["pfam_families"] else 0.0 for r in sub])),
                "mean_nll_per_scored_token": float(nll.mean()),
                "mean_nll_per_residue": float(nll_res.mean()),
            },
            "spearman_high_likelihood": primary_stats["spearman"],
            "per_residue_nll_sensitivity": per_res_sens,
            "truncation_selection": primary_stats["truncation_selection"],
            "fwl_column_span_ladder": fwl,
        }

    meta = POOL_METADATA.get(pool_id, {})
    arm_id = meta.get("arm_id", pool_id)
    arm_meta = ARM_METADATA.get(arm_id)
    return {
        "pool_id": pool_id,
        "arm_id": arm_id,
        "display_name": arm_meta.display_name if arm_meta else arm_id,
        "paradigm": meta.get("paradigm", arm_meta.paradigm if arm_meta else "unknown"),
        "family": meta.get("family", arm_meta.family if arm_meta else "unknown"),
        "params_m": arm_meta.params_m if arm_meta else None,
        "conditioned": bool(meta.get("conditioned", False)),
        "decoding_mode": meta.get("decoding_mode"),
        "n_pool_total": n_total,
        "regimes": regimes_out,
    }


def _eval_pool_worker(args: tuple[str, list[dict[str, Any]], int, int, int]) -> dict[str, Any]:
    pool_id, rows, n_draws, fwl_draws, seed = args
    return evaluate_pool_across_regimes(
        pool_id, rows, n_draws=n_draws, fwl_draws=fwl_draws, seed=seed
    )


# --------------------------------------------------------------------------- #
# Loading GPU-scored pools & high-power objective_tuned companion
# --------------------------------------------------------------------------- #


def _extract_score_fields(score_row: dict[str, Any]) -> tuple[float, float]:
    """Normalize causal NLL or bidirectional PLL score records into (nll_token, nll_residue)."""
    nll_tok = (
        score_row.get("mean_nll_per_scored_token")
        if score_row.get("mean_nll_per_scored_token") is not None
        else score_row.get("mean_nll_per_token_nats")
    )
    nll_res = (
        score_row.get("mean_nll_per_residue")
        if score_row.get("mean_nll_per_residue") is not None
        else score_row.get("mean_nll_per_residue_nats")
    )
    if nll_res is None and score_row.get("mean_pll_per_residue") is not None:
        nll_res = -float(score_row["mean_pll_per_residue"])
    if nll_tok is None:
        nll_tok = nll_res
    if nll_res is None:
        nll_res = nll_tok
    if nll_tok is None or nll_res is None:
        raise ValueError(f"Score record missing NLL/PLL fields: {sorted(score_row.keys())}")
    return float(nll_tok), float(nll_res)


def load_all_33_pools(
    e17_data: dict[str, Any],
    diamond_cache: Mapping[str, dict[str, Any]],
    pfam_cache: Mapping[str, dict[str, Any]],
    *,
    root: Path = REPO_ROOT,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Assemble all 33 primary pools (2 frozen E17 + 31 GPU-scored) plus s48_protgpt2 sensitivity."""
    work_dir = root / EXP4_3_WORK_REL
    inputs_dir = work_dir / "inputs"
    scores_dir = work_dir / "scores"

    pools: dict[str, list[dict[str, Any]]] = {}
    score_receipts: dict[str, Any] = {}

    for pool_id in PRIMARY_33_POOLS:
        meta = POOL_METADATA[pool_id]
        if meta["source_kind"] == "e17_frozen":
            enriched: list[dict[str, Any]] = []
            for r in e17_data["pools"][pool_id]:
                c = sequence_complexity_features(r["sequence"])
                enriched.append({**r, **c})
            pools[pool_id] = enriched
            score_receipts[pool_id] = {"source": "e17_frozen", "n_rows": len(enriched)}
            continue

        in_path = inputs_dir / f"{pool_id}.jsonl"
        sc_path = scores_dir / f"{pool_id}.jsonl"
        if not in_path.is_file():
            raise FileNotFoundError(f"Missing Exp 4.3 input pool: {in_path}")
        if not sc_path.is_file():
            raise FileNotFoundError(
                f"Missing GPU self-likelihood score file: {sc_path} (waiting on @ops-staging-esm3)"
            )

        in_rows = _read_jsonl(in_path)
        sc_rows = _read_jsonl(sc_path)
        sc_by_id = {str(r.get("seq_id") or r.get("id")): r for r in sc_rows}
        if len(sc_by_id) != len(in_rows):
            raise ValueError(f"{pool_id}: score count {len(sc_by_id)} != input count {len(in_rows)}")

        joined: list[dict[str, Any]] = []
        for r in in_rows:
            sid = str(r["seq_id"])
            srow = sc_by_id.get(sid)
            if srow is None:
                raise ValueError(f"{pool_id}: missing score for seq_id={sid}")
            if srow.get("sequence_sha256") and srow["sequence_sha256"] != r["sequence_sha256"]:
                raise ValueError(f"{pool_id}/{sid}: sequence_sha256 mismatch between input and score")
            nll_tok, nll_res = _extract_score_fields(srow)
            if not (math.isfinite(nll_tok) and math.isfinite(nll_res)):
                raise ValueError(f"{pool_id}/{sid}: non-finite self-likelihood")

            seq = str(r["sequence"]).strip().upper()
            sha = str(r["sequence_sha256"])
            dm = diamond_cache.get(sha)
            if dm is None:
                raise KeyError(f"{pool_id}/{sid}: missing DIAMOND UniRef50 annotation for sha={sha}")
            pf = pfam_cache.get(sha) or {"families": []}
            comp = sequence_complexity_features(seq)
            joined.append(
                {
                    "pool_id": pool_id,
                    "arm_id": meta["arm_id"],
                    "seq_id": sid,
                    "sequence": seq,
                    "sequence_sha256": sha,
                    "unit_id": sid,
                    "plddt": float(r["plddt"]),
                    "ptm": float(r["ptm"]),
                    "mean_nll_per_scored_token": nll_tok,
                    "mean_nll_per_residue": nll_res,
                    "uniref50_identity": float(dm["identity_over_query"]),
                    "pfam_families": list(pf.get("families") or []),
                    **comp,
                }
            )
        pools[pool_id] = joined
        score_receipts[pool_id] = {
            "source": str(sc_path.relative_to(root)),
            "sha256": sha256_file(sc_path),
            "n_rows": len(joined),
        }

    return pools, score_receipts


def evaluate_objective_tuned_companion(
    *,
    root: Path = REPO_ROOT,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """High-power (~2,500 folded sequences/arm) companion verification from
    ``objective_tuned_base_likelihood_20261114`` (``protgpt3-{112m,1.3b,10b}``, ``progen2-small``).
    """
    nn_path = root / OT_NN_TSV_REL
    cohort_path = root / OT_COHORT_REL
    lik_dir = root / OT_LIK_DIR_REL
    if not (nn_path.is_file() and cohort_path.is_file() and lik_dir.is_dir()):
        return {"status": "unavailable", "reason": "companion files missing"}

    seq_by_sha: dict[str, str] = {}
    with cohort_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            seq_by_sha[str(r["id"])] = str(r["sequence"]).strip().upper()

    pfam_by_sha: dict[str, list[str]] = {}
    ot_rec = root / OT_RECOG_REL
    if ot_rec.is_file():
        for r in _read_jsonl(ot_rec):
            sha = str(r.get("sequence_sha256") or "")
            if sha:
                pfam_by_sha[sha] = sorted(set(str(f) for f in (r.get("families") or [])))

    base_arms = ("protgpt3-112m", "protgpt3-1.3b", "protgpt3-10b", "progen2-small-hugohrban")
    lik_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in base_arms:
        p = lik_dir / f"{arm}__on__{arm}.jsonl"
        lik_by_arm[arm] = {str(r["sequence_sha256"]): r for r in _read_jsonl(p)}

    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in base_arms}
    with nn_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            arm = row["arm"]
            if arm not in rows_by_arm:
                continue
            sha = row["sequence_sha256"]
            lk = lik_by_arm[arm].get(sha)
            seq = seq_by_sha.get(sha)
            if lk is None or seq is None:
                continue
            comp = sequence_complexity_features(seq)
            rows_by_arm[arm].append(
                {
                    "pool_id": arm,
                    "arm_id": "progen2-small" if arm == "progen2-small-hugohrban" else arm,
                    "seq_id": sha,
                    "sequence": seq,
                    "sequence_sha256": sha,
                    "unit_id": sha,
                    "plddt": float(row["plddt"]),
                    "ptm": float(row["ptm"]),
                    "mean_nll_per_scored_token": float(lk["mean_nll"]),
                    "mean_nll_per_residue": float(lk["sum_nll"]) / max(1, len(seq)),
                    "uniref50_identity": float(row["identity_over_query"]),
                    "pfam_families": pfam_by_sha.get(sha, []),
                    **comp,
                }
            )

    arms_out = {
        arm: evaluate_pool_across_regimes(
            arm, rows_by_arm[arm], n_draws=n_draws, fwl_draws=0, seed=seed
        )
        for arm in base_arms
    }
    return {
        "status": "complete",
        "source_nn_tsv": str(OT_NN_TSV_REL),
        "source_likelihood_dir": str(OT_LIK_DIR_REL),
        "arms": arms_out,
    }


def _aggregate_cross_model_summary(
    pools_eval: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate cross-model adverse-selection & Spearman counts per regime and budget."""
    summary: dict[str, Any] = {}
    for reg_name, _, _ in REGIMES:
        sp_counts: dict[str, Counter[str]] = {
            ep: Counter() for ep in ("plddt", "ptm", "distinct_5mer", "homopolymer_frac")
        }
        sp_point_neg: dict[str, int] = {ep: 0 for ep in sp_counts}
        sp_evaluable = 0

        sel_summary: dict[str, dict[str, dict[str, Any]]] = {
            t: {
                b_name: {
                    "n_evaluable": 0,
                    "n_point_adverse": 0,
                    "n_ci_adverse": 0,
                    "n_ci_positive": 0,
                    "n_neutral": 0,
                    "point_adverse_pools": [],
                    "ci_adverse_pools": [],
                    "ci_positive_pools": [],
                    "median_gain_over_random": None,
                }
                for b_name, _, _ in SELECTION_BUDGETS
            }
            for t in ("plddt", "ptm", "distinct_5mer")
        }
        gain_lists: dict[str, dict[str, list[float]]] = {
            t: {b_name: [] for b_name, _, _ in SELECTION_BUDGETS}
            for t in ("plddt", "ptm", "distinct_5mer")
        }

        for pool_id, prec in pools_eval.items():
            reg = prec["regimes"][reg_name]
            if reg.get("status") != "measured":
                continue
            sp_evaluable += 1
            for ep in sp_counts:
                srec = reg["spearman_high_likelihood"][ep]
                sp_counts[ep][srec["verdict"]] += 1
                if srec["point"] is not None and srec["point"] < 0.0:
                    sp_point_neg[ep] += 1

            for t in ("plddt", "ptm", "distinct_5mer"):
                for b_name, _, _ in SELECTION_BUDGETS:
                    s = reg["truncation_selection"][t][b_name]
                    slot = sel_summary[t][b_name]
                    slot["n_evaluable"] += 1
                    gain_lists[t][b_name].append(float(s["gain_over_random"]))
                    if s["point_adverse"]:
                        slot["n_point_adverse"] += 1
                        slot["point_adverse_pools"].append(pool_id)
                    cls = s["adverse_class"]
                    if cls == "adverse_ci_negative":
                        slot["n_ci_adverse"] += 1
                        slot["ci_adverse_pools"].append(pool_id)
                    elif cls == "positive_ci_positive":
                        slot["n_ci_positive"] += 1
                        slot["ci_positive_pools"].append(pool_id)
                    else:
                        slot["n_neutral"] += 1

        for t in ("plddt", "ptm", "distinct_5mer"):
            for b_name, _, _ in SELECTION_BUDGETS:
                vals = gain_lists[t][b_name]
                if vals:
                    sel_summary[t][b_name]["median_gain_over_random"] = float(np.median(vals))

        summary[reg_name] = {
            "n_pools_evaluable": sp_evaluable,
            "n_pools_total": len(pools_eval),
            "spearman_counts": {
                ep: {
                    "ci_positive": int(sp_counts[ep]["positive"]),
                    "ci_negative": int(sp_counts[ep]["negative"]),
                    "unresolved": int(sp_counts[ep]["unresolved"]),
                    "point_negative": int(sp_point_neg[ep]),
                }
                for ep in sp_counts
            },
            "truncation_selection_counts": sel_summary,
        }
    return summary


# --------------------------------------------------------------------------- #
# Main orchestrator
# --------------------------------------------------------------------------- #


def collect_unique_input_sequences(root: Path = REPO_ROOT) -> dict[str, str]:
    """Collect all unique (sha256 -> sequence) pairs across inputs/*.jsonl."""
    inputs_dir = root / EXP4_3_WORK_REL / "inputs"
    unique: dict[str, str] = {}
    for p in sorted(inputs_dir.glob("*.jsonl")):
        for r in _read_jsonl(p):
            seq = str(r["sequence"]).strip().upper()
            sha = str(r.get("sequence_sha256") or seq_sha256(seq))
            unique[sha] = seq
    return unique


def build_unified_r6_experiment_4_3(
    *,
    root: Path = REPO_ROOT,
    out_dir: Path | None = None,
    diamond_threads: int = 32,
    pfam_workers: int = 16,
    eval_workers: int = 12,
    annotate_only: bool = False,
) -> dict[str, Any]:
    """Execute the full unified 30-model / 33-pool Experiment 4.3 pipeline."""
    t_start = time.monotonic()
    target_dir = Path(out_dir) if out_dir is not None else (root / UNIFIED_OUTPUT_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. E17 < 1e-12 reproduction gate
    print("[exp4.3] loading frozen E17 pools & running < 1e-12 reproduction gate...", flush=True)
    e17_data = load_e17_frozen_pools(root=root)
    e17_gate = run_e17_reproduction_gate(e17_data, root=root)
    print(
        f"[exp4.3] E17 reproduction gate PASSED: max_abs_diff={e17_gate['max_abs_diff']:.2e} "
        f"({e17_gate['checked_scalars']} scalars)",
        flush=True,
    )

    # 2. Ensure DIAMOND UniRef50 & pinned Pfam annotations for all unique input sequences
    unique_seqs = collect_unique_input_sequences(root=root)
    print(f"[exp4.3] ensuring DIAMOND & Pfam annotations for {len(unique_seqs)} unique sequences...", flush=True)
    diamond_cache = ensure_diamond_uniref50(unique_seqs, root=root, threads=diamond_threads)
    pfam_cache = ensure_pfam_annotations(unique_seqs, root=root, workers=pfam_workers)
    if annotate_only:
        return {
            "status": "annotations_ready",
            "e17_reproduction_gate": e17_gate,
            "n_unique_input_sequences": len(unique_seqs),
            "n_diamond_cached": len(diamond_cache),
            "n_pfam_cached": len(pfam_cache),
            "elapsed_seconds": float(time.monotonic() - t_start),
        }

    # 3. Join all 33 primary pools with GPU self-likelihoods + annotations
    print("[exp4.3] joining all 33 pools with native self-likelihoods & Axis 1 annotations...", flush=True)
    pools_data, score_receipts = load_all_33_pools(
        e17_data, diamond_cache, pfam_cache, root=root
    )

    # 4. Evaluate all 33 pools across all 5 regimes in parallel (spawn pool)
    n_workers = max(1, min(int(eval_workers), 16))
    print(
        f"[exp4.3] evaluating 33 pools x 5 regimes (10,000 bootstrap draws, workers={n_workers})...",
        flush=True,
    )
    tasks = [
        (pool_id, pools_data[pool_id], BOOTSTRAP_DRAWS, FWL_BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
        for pool_id in PRIMARY_33_POOLS
    ]
    pools_eval: dict[str, dict[str, Any]] = {}
    if n_workers == 1:
        for t in tasks:
            rec = _eval_pool_worker(t)
            pools_eval[rec["pool_id"]] = rec
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            for rec in pool.map(_eval_pool_worker, tasks):
                pools_eval[rec["pool_id"]] = rec

    cross_summary = _aggregate_cross_model_summary(pools_eval)

    # 5. Companion high-power objective_tuned_base_likelihood_20261114 verification (~2,500 seqs/arm)
    print("[exp4.3] running high-power objective_tuned_base_likelihood_20261114 companion...", flush=True)
    ot_companion = evaluate_objective_tuned_companion(root=root)

    # Optional s48_protgpt2 (N=128) sensitivity if scored alongside E17 protgpt2 (N=600)
    s48_protgpt2_sens = None
    s48_sc = root / EXP4_3_WORK_REL / "scores" / "protgpt2.jsonl"
    s48_in = root / EXP4_3_WORK_REL / "inputs" / "protgpt2.jsonl"
    if s48_sc.is_file() and s48_in.is_file():
        in_r = _read_jsonl(s48_in)
        sc_by_id = {str(r.get("seq_id") or r.get("id")): r for r in _read_jsonl(s48_sc)}
        if len(sc_by_id) == len(in_r):
            s48_rows = []
            for r in in_r:
                sid = str(r["seq_id"])
                nll_tok, nll_res = _extract_score_fields(sc_by_id[sid])
                seq = str(r["sequence"]).strip().upper()
                sha = str(r["sequence_sha256"])
                s48_rows.append(
                    {
                        "pool_id": "protgpt2__s48_128",
                        "arm_id": "protgpt2",
                        "seq_id": sid,
                        "sequence": seq,
                        "sequence_sha256": sha,
                        "unit_id": sid,
                        "plddt": float(r["plddt"]),
                        "ptm": float(r["ptm"]),
                        "mean_nll_per_scored_token": nll_tok,
                        "mean_nll_per_residue": nll_res,
                        "uniref50_identity": float(diamond_cache[sha]["identity_over_query"]),
                        "pfam_families": list((pfam_cache.get(sha) or {}).get("families") or []),
                        **sequence_complexity_features(seq),
                    }
                )
            s48_protgpt2_sens = evaluate_pool_across_regimes(
                "protgpt2__s48_128", s48_rows, n_draws=BOOTSTRAP_DRAWS, fwl_draws=0, seed=BOOTSTRAP_SEED
            )

    unique_checkpoints = sorted({POOL_METADATA[p]["arm_id"] for p in PRIMARY_33_POOLS})
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.monotonic() - t_start),
        "n_unique_checkpoints": len(unique_checkpoints),
        "n_primary_pools": len(PRIMARY_33_POOLS),
        "primary_pool_order": list(PRIMARY_33_POOLS),
        "unique_checkpoints": unique_checkpoints,
        "bootstrap_config": {
            "n_draws_spearman_and_selection": BOOTSTRAP_DRAWS,
            "n_draws_fwl_column_span": FWL_BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "minimum_bootstrap_units": MINIMUM_BOOTSTRAP_UNITS,
            "natural_entropy_min_bits": NATURAL_ENTROPY_MIN_BITS,
        },
        "e17_reproduction_gate": e17_gate,
        "cross_model_summary": cross_summary,
        "pools": pools_eval,
        "companion_verifications": {
            "objective_tuned_base_likelihood_20261114": ot_companion,
            "protgpt2_exp_r2_246_128_sensitivity": s48_protgpt2_sens,
        },
        "score_receipts": score_receipts,
    }

    out_path = target_dir / "r6_experiment_4_3_native_likelihood_unified.json"
    atomic_write_json(out_path, payload)
    digest = sha256_file(out_path)
    sidecar = out_path.with_name(out_path.name + ".sha256")
    sidecar.write_text(f"{digest}  {out_path.name}\n", encoding="utf-8")
    print(
        f"[exp4.3] WROTE {out_path} (sha256={digest[:16]}...) in {payload['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return payload
