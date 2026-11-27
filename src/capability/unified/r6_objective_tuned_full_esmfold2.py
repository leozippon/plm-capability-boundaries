"""Unified full-cohort Result 6 ESMFold2 evaluation (68/68 shards + 2,000/2,000 long-cap).

Re-runs every endpoint and confounding audit of the ``objective_tuned_20261109``
campaign and its ``long_cap_termination_20261110`` 768-residue companion with
**100.0% ESMFold2 structure coverage (ZERO sampling, ZERO departures)**:

* Main cohort: 48,000 attempts / 47,788 non-empty / 47,778 unique sequences
  across all 60 generation cells (8 arms = 4 matched pairs, 3 independent
  streams per arm), 100% UniRef50 identity, 100% Pfam HMMER 3.4 recognition,
  100% own- & base-model likelihood, 100% ESMFold2 (32 diffusion samples,
  ``seed=20260905``, ``seed_rule="sequence_digest"``).
* Companion cohort (``long_cap_termination_20261110``): 2,000/2,000 sequences
  (1,000 ``protgpt3-10b`` + 1,000 ``protgpt3-10b-dpo``), including all 321/321
  ``base-capped`` ``L=768`` sequences (zero subsampling).
"""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
import time
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.capability.context import homology as hm
from src.capability.core.io import sha256_file
from src.capability.extensions import bidirectional as bd
from src.capability.forcing.backbone_cohort import distinct_shingle_fraction
from src.capability.general_plm.eval_utils import atomic_write_json
from src.capability.generation import generative_control as gc_mod
from src.capability.generation import length_matching as lm
from src.capability.generation import objective_tuned_analysis as ota
from src.capability.generation import objective_tuned_arms as ot
from src.capability.unified.roster import REPO_ROOT, UNIFIED_OUTPUT_DIR

# Add scripts/capability/generation to path so we can import the frozen analysis modules directly.
_GEN_SCRIPTS = REPO_ROOT / "scripts/capability/generation"
if str(_GEN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_GEN_SCRIPTS))

import analyse_cross_likelihood as acl  # noqa: E402
import analyse_objective_tuned as aot  # noqa: E402

SCHEMA = "r6_objective_tuned_full_esmfold2_unified_v1"
EXPECTED_SIGNATURE = "9dc43daab2888ea6db0746f9e6567c34e52b379048066d0218a9d920951a8660"

PAIRS_SHORT: tuple[tuple[str, str, str], ...] = (
    ("112m", "protgpt3-112m", "protgpt3-112m-dpo"),
    ("1.3b", "protgpt3-1.3b", "protgpt3-1.3b-dpo"),
    ("10b", "protgpt3-10b", "protgpt3-10b-dpo"),
    ("mix7", "progen2-small-hugohrban", "progen2-small-mix7"),
)
LENGTH_STRATA: tuple[tuple[str, int, int], ...] = (
    ("16-128", 16, 128),
    ("129-256", 129, 256),
    ("257-512", 257, 512),
)
CAP_400 = 400
ZLIB_DEGENERATE_THRESHOLD = 0.4361
SEVEN_FAMILIES: tuple[str, ...] = tuple(
    sorted(set(re.findall(r"PF\d{5}", ot.OBJECTIVE_PFAM_SEVEN_FAMILIES)))
)


def _clean_json(obj: Any) -> Any:
    """Recursively convert NaN/Inf floats and numpy scalars to JSON-safe Python values."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if not math.isfinite(val) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    return obj


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _ci95(values: np.ndarray | list[float]) -> list[float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [None, None]
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return [float(lo), float(hi)]


# --------------------------------------------------------------------------- #
# Pfam recognition merger (12,640 frozen + 35,138 remainder = 47,778)
# --------------------------------------------------------------------------- #


def merge_pfam_recognition_68of68(
    root: Path = REPO_ROOT,
    cache_dir: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Merge frozen 18-shard Pfam JSONL (12,640) + remainder (35,138) into 47,778-row files."""
    cdir = cache_dir if cache_dir is not None else (root / UNIFIED_OUTPUT_DIR / "r6_objective_tuned_full_cache")
    p18 = root / "results/extensions/objective_tuned_endpoints_20261111/recognition/eighteen_shard/family_recognition.jsonl"
    prem = cdir / "pfam_remainder/family_recognition.jsonl"
    fc_path = root / "results/external_baseline_analysis/length_adjusted_contrasts_20261110/input/fold_cohort.jsonl"

    by_id: dict[str, dict[str, Any]] = {}
    for src in (p18, prem):
        for row in _read_jsonl(src):
            rid = str(row["sequence_sha256"] or row["id"])
            if rid in by_id:
                raise RuntimeError(f"Duplicate Pfam digest across 18-shard and remainder: {rid}")
            by_id[rid] = row

    cohort_ids = [str(r["id"]) for r in _read_jsonl(fc_path)]
    missing = [cid for cid in cohort_ids if cid not in by_id]
    if missing:
        raise RuntimeError(f"Merged Pfam recognition missing {len(missing)} cohort digests (e.g. {missing[:3]})")
    if len(by_id) != len(cohort_ids):
        raise RuntimeError(f"Merged Pfam size {len(by_id)} != cohort size {len(cohort_ids)}")

    full_jsonl = cdir / "family_recognition_68of68.jsonl"
    flags_jsonl = cdir / "recognition_flags_68of68.jsonl"
    ordered_rows = [by_id[cid] for cid in sorted(cohort_ids)]
    full_jsonl.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in ordered_rows),
        encoding="utf-8",
    )
    flags_rows = [
        {
            "sequence_sha256": str(r["sequence_sha256"]),
            "any_family": bool(r["any_profile_hit"]),
            "complete_domain": bool(r["complete_domain"]),
            "pfam_families": list(r.get("pfam_families") or []),
            "best_profile_coverage": r.get("best_profile_coverage"),
        }
        for r in ordered_rows
    ]
    flags_jsonl.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in flags_rows),
        encoding="utf-8",
    )
    meta = {
        "n_from_frozen_18of68": sum(1 for _ in _read_jsonl(p18)),
        "n_from_remainder_20261124": sum(1 for _ in _read_jsonl(prem)),
        "n_merged_total": len(ordered_rows),
        "n_any_family": sum(1 for r in flags_rows if r["any_family"]),
        "n_complete_domain": sum(1 for r in flags_rows if r["complete_domain"]),
        "family_recognition_68of68_sha256": sha256_file(full_jsonl),
        "recognition_flags_68of68_sha256": sha256_file(flags_jsonl),
    }
    return full_jsonl, flags_jsonl, meta


# --------------------------------------------------------------------------- #
# Gate 1: Exact Reproduction Against Frozen 18/68 & 1,748/2,000 Tables
# --------------------------------------------------------------------------- #


def verify_r6_reproduction_gate(
    full_main_tsv: Path,
    full_long_cap_tsv: Path,
    *,
    root: Path = REPO_ROOT,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Assert 100% exact parity on all 12,640 frozen 18/68 rows and all 1,748 frozen long_cap rows."""
    frozen_18 = pd.read_csv(
        root / "results/external_baseline_analysis/length_adjusted_contrasts_20261110/eighteen_shard/checkpoint_structures_18of68.tsv",
        sep="\t",
        comment="#",
    )
    full_68 = pd.read_csv(full_main_tsv, sep="\t", comment="#")
    sub_18 = full_68[full_68["shard_index_of_68"] < 18].copy()

    if len(full_68) != 47778:
        raise AssertionError(f"Expected 47,778 main rows, got {len(full_68)}")
    if len(sub_18) != len(frozen_18):
        raise AssertionError(f"18-shard subset count mismatch: {len(sub_18)} != {len(frozen_18)}")
    merged = frozen_18.merge(sub_18, on=["arm", "sequence_sha256"], suffixes=("_frozen", "_full"), validate="one_to_one")
    if len(merged) != len(frozen_18):
        raise AssertionError(f"18-shard join lost rows: {len(merged)} != {len(frozen_18)}")
    plddt_diff_18 = float(np.max(np.abs(merged["mean_ca_plddt_0_100_frozen"] - merged["mean_ca_plddt_0_100_full"])))
    ptm_diff_18 = float(np.max(np.abs(merged["ptm_0_1_frozen"] - merged["ptm_0_1_full"])))
    if not (merged["length_residues_frozen"] == merged["length_residues_full"]).all():
        raise AssertionError("18-shard length mismatch")
    if not (full_68["evaluation_signature"] == EXPECTED_SIGNATURE).all():
        raise AssertionError("Non-matching evaluation_signature in checkpoint_structures_68of68.tsv")

    frozen_lc = pd.read_csv(
        root / "results/long_cap_termination_20261110/analysis/per_structure_joined.tsv",
        sep="\t",
    )
    full_lc = pd.read_csv(full_long_cap_tsv, sep="\t")
    if len(full_lc) != 2000:
        raise AssertionError(f"Expected 2,000 long_cap rows, got {len(full_lc)}")
    m_lc = frozen_lc.merge(full_lc, on=["arm", "attempt_index", "sequence_sha256"], suffixes=("_frozen", "_full"), validate="one_to_one")
    if len(m_lc) != len(frozen_lc):
        raise AssertionError(f"long_cap join lost rows: {len(m_lc)} != {len(frozen_lc)}")
    plddt_diff_lc = float(np.max(np.abs(m_lc["plddt_frozen"] - m_lc["plddt_full"])))
    ptm_diff_lc = float(np.max(np.abs(m_lc["ptm_frozen"] - m_lc["ptm_full"])))
    if not (full_lc["evaluation_signature"] == EXPECTED_SIGNATURE).all():
        raise AssertionError("Non-matching evaluation_signature in long_cap_structures_2000of2000.tsv")

    max_diff = max(plddt_diff_18, ptm_diff_18, plddt_diff_lc, ptm_diff_lc)
    if max_diff >= tolerance:
        raise AssertionError(f"R6 reproduction gate failed: max_diff={max_diff:.3e} >= {tolerance:.1e}")

    return {
        "passed": True,
        "tolerance": tolerance,
        "main_68_total_rows": int(len(full_68)),
        "main_18_subset_rows": int(len(sub_18)),
        "main_18_max_plddt_diff": plddt_diff_18,
        "main_18_max_ptm_diff": ptm_diff_18,
        "long_cap_total_rows": int(len(full_lc)),
        "long_cap_existing_rows": int(len(m_lc)),
        "long_cap_newly_folded_rows": int(len(full_lc) - len(m_lc)),
        "long_cap_max_plddt_diff": plddt_diff_lc,
        "long_cap_max_ptm_diff": ptm_diff_lc,
        "single_evaluation_signature": EXPECTED_SIGNATURE,
    }


# --------------------------------------------------------------------------- #
# Full-Cohort Zero-Departure Campaign Table & Per-Structure DataFrame
# --------------------------------------------------------------------------- #


def _build_full_campaign_table(
    root: Path,
    main_tsv: Path,
    recognition_flags_jsonl: Path,
    cache_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, Path]:
    """Build zero-departure 48,000-attempt 3-stream campaign table + 47,778-row unique DataFrame."""
    campaign_root = root / "results/external_baseline/20261110013345_599d8983b45a"
    campaign = aot.read_campaign(campaign_root)

    # Load 47,778 structures
    fold_df = pd.read_csv(main_tsv, sep="\t", comment="#").rename(
        columns={"length_residues": "L", "mean_ca_plddt_0_100": "plddt", "ptm_0_1": "ptm"}
    )
    structure_for_aot: dict[str, dict[str, Any]] = {}
    ptm_by_sha: dict[str, float] = {}
    for row in fold_df.itertuples(index=False):
        sha = str(row.sequence_sha256)
        structure_for_aot[sha] = {
            "plddt": float(row.plddt),
            "status": "ok",
            "role": None,
            "paired_id": None,
        }
        ptm_by_sha[sha] = float(row.ptm)

    # Load 47,778 UniRef50 identities
    id_path = root / "results/external_baseline_analysis/objective_tuned_identity_20261114/remainder/identity_annotations_union.jsonl"
    novelty_for_aot: dict[str, dict[str, Any]] = {}
    id_by_sha: dict[str, float] = {}
    for rec in _read_jsonl(id_path):
        sha = str(rec["id"])
        ident = float(rec["nearest_corpus_identity"])
        id_by_sha[sha] = ident
        novelty_for_aot[sha] = {
            "identity": ident,
            "stratum": hm.assign_stratum(ident),
            "had_any_hit": bool(rec.get("had_any_hit", ident > 0.0)),
        }

    # Assemble through frozen aot.assemble for 100% parity (keeps empty attempts, digest-ordered keys, exact counts)
    table = aot.assemble(campaign, structure_for_aot, novelty_for_aot)

    # Load 47,778 Pfam recognition flags and attach ptm + pfam_families to every row
    recog_by_sha: dict[str, dict[str, Any]] = {}
    for rec in _read_jsonl(recognition_flags_jsonl):
        recog_by_sha[str(rec["sequence_sha256"])] = {
            "any_family": bool(rec["any_family"]),
            "complete_domain": bool(rec["complete_domain"]),
            "pfam_families": list(rec.get("pfam_families") or []),
        }

    total_attempts = 0
    total_non_empty = 0
    total_folded = 0
    total_searched = 0
    for pool in table["pools"].values():
        total_attempts += int(pool["counts"]["attempts"])
        total_non_empty += int(pool["counts"]["non_empty"])
        total_folded += int(pool["counts"]["folded"])
        total_searched += int(pool["counts"]["searched"])
        for row in pool["rows"]:
            if row["empty"]:
                row["ptm"] = None
                row["any_family"] = False
                row["complete_domain"] = False
                row["pfam_families"] = []
                continue
            sha = row["digest"]
            if sha not in ptm_by_sha:
                raise RuntimeError(f"Missing ESMFold2 structure for {pool['arm']} s{pool['stream']} {sha}")
            if sha not in recog_by_sha:
                raise RuntimeError(f"Missing Pfam recognition for {sha}")
            row["ptm"] = ptm_by_sha[sha]
            row["any_family"] = recog_by_sha[sha]["any_family"]
            row["complete_domain"] = recog_by_sha[sha]["complete_domain"]
            row["pfam_families"] = recog_by_sha[sha]["pfam_families"]

    if total_folded != total_non_empty or total_searched != total_non_empty:
        raise RuntimeError(
            f"Coverage departure detected: folded={total_folded}, searched={total_searched}, non_empty={total_non_empty}"
        )

    # Materialize 47,778-row selector_score_68of68.jsonl across all 8 arms for aot.selector_competence
    own_dir = root / "results/extensions/objective_tuned_endpoints_20261112/selector/own_model_likelihood"
    selector_jsonl = cache_dir / "selector_score_68of68.jsonl"
    sel_lines: list[str] = []
    for arm in sorted({a for _, b, t in PAIRS_SHORT for a in (b, t)}):
        scores = ota.read_own_likelihood(own_dir / f"{arm}__on__{arm}.jsonl")
        for sha, (tok_n, total_nll) in sorted(scores.items()):
            mean_nll = float(total_nll) / float(tok_n)
            sel_lines.append(
                json.dumps(
                    {
                        "arm": arm,
                        "sequence_sha256": sha,
                        "n_tokens": int(tok_n),
                        "mean_nll_per_residue_nats": mean_nll,
                        "score": -mean_nll,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    selector_jsonl.write_text("".join(sel_lines), encoding="utf-8")

    table["campaign"] = ot.CAMPAIGN
    table["campaign_root"] = str(campaign_root)
    table["totals"] = {
        "attempts": total_attempts,
        "non_empty": total_non_empty,
        "unique_sequences": len(structure_for_aot),
        "empty_or_noncanonical": total_attempts - total_non_empty,
        "fold_coverage": 1.0,
        "identity_coverage": 1.0,
        "pfam_coverage": 1.0,
        "selector_coverage": 1.0,
    }
    table["departures"] = (
        "NONE (0 departures: all 48,000 attempts / 47,788 non-empty / 47,778 unique sequences, "
        "exact 3 streams, exact stop_status, 100% ESMFold2, 100% Pfam, 100% UniRef50, 100% own-model likelihood)"
    )

    # Build 47,778-row unique-structure DataFrame with sequence composition metrics.
    fc_seqs = {str(r["id"]): str(r["sequence"]) for r in _read_jsonl(root / "results/external_baseline_analysis/length_adjusted_contrasts_20261110/input/fold_cohort.jsonl")}

    def _comp(s: str) -> tuple[float, float, int, float, float, float]:
        c = Counter(s)
        n = len(s)
        p = np.array(list(c.values()), dtype=np.float64) / n
        max_run = max((len(list(g)) for _, g in itertools.groupby(s)), default=0)
        k4 = len({s[i : i + 4] for i in range(n - 3)}) / max(n - 3, 1)
        k5 = distinct_shingle_fraction(s)
        return (
            max(c.values()) / n,
            float(-(p * np.log2(p)).sum()),
            int(max_run),
            len(zlib.compress(s.encode("utf-8"), 9)) / n,
            k4,
            k5,
        )

    seqs_col = [fc_seqs[sha] for sha in fold_df["sequence_sha256"]]
    comp_arr = np.array([_comp(s) for s in seqs_col], dtype=np.float64)
    fold_df["top_frac"] = comp_arr[:, 0]
    fold_df["entropy_bits"] = comp_arr[:, 1]
    fold_df["max_run"] = comp_arr[:, 2]
    fold_df["zlib_ratio"] = comp_arr[:, 3]
    fold_df["distinct_4mer_frac"] = comp_arr[:, 4]
    fold_df["distinct_5mer_frac"] = comp_arr[:, 5]
    fold_df["capped"] = fold_df["L"] == CAP_400
    fold_df["identity"] = fold_df["sequence_sha256"].map(id_by_sha)
    fold_df["stratum"] = fold_df["identity"].map(hm.assign_stratum)
    return table, fold_df, selector_jsonl


PTM_MARGIN_CRITERION = 0.50
PTM_MARGIN_FLOOR = 0.02
PTM_MARGIN_FRACTION = 0.25
PTM_MARGIN_RULE = (
    "max(PTM_MARGIN_FLOOR, PTM_MARGIN_FRACTION * max(0, 0.50 - base in-stratum mean)), "
    "in native ESMFold2 pTM units on the 0-1 scale, computed from the base pool alone."
)


def ptm_margin_for(base_mean: float) -> dict[str, float | str]:
    """Scale-calibrated margin on the 0-1 ESMFold2 pTM scale (never 0-100 pLDDT points)."""
    value = float(base_mean)
    if not math.isfinite(value):
        raise ValueError("a pTM margin needs a finite base in-stratum mean")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"base_mean={value!r} outside [0, 1] ESMFold2 pTM scale")
    headroom = max(0.0, PTM_MARGIN_CRITERION - value)
    return {
        "margin": max(PTM_MARGIN_FLOOR, PTM_MARGIN_FRACTION * headroom),
        "headroom_to_criterion": headroom,
        "base_in_stratum_mean": value,
        "criterion": float(PTM_MARGIN_CRITERION),
        "floor": float(PTM_MARGIN_FLOOR),
        "fraction": float(PTM_MARGIN_FRACTION),
        "rule": PTM_MARGIN_RULE,
    }


def _swap_table_metric_to_ptm(table: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy table with ``ptm`` in the ``plddt`` slot so exact 3-stream simultaneous bands run on pTM."""
    pools = {}
    for k, pool in table["pools"].items():
        pools[k] = {
            **pool,
            "rows": [{**r, "plddt": r["ptm"]} for r in pool["rows"]],
        }
    return {**table, "pools": pools}


def _stratified_ptm_contrast(
    table_ptm: dict[str, Any],
    pairs: tuple[ot.ObjectiveTunedPair, ...],
    family: str,
    *,
    shift: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Novelty-stratified 3-stream simultaneous contrast on the 0-1 ESMFold2 pTM scale.

    Uses the exact 3-stream simultaneous bootstrap bands ``aot.bands(columns, family)`` while
    calibrating margins strictly on the 0-1 pTM scale (``PTM_MARGIN_CRITERION=0.50``,
    ``PTM_MARGIN_FLOOR=0.02``, ``PTM_MARGIN_FRACTION=0.25``) rather than 0-100 pLDDT constants.
    """
    columns = []
    for pair in pairs:
        base, tuned = aot.require_pair(table_ptm, pair)
        for stratum in ota.STRATA:
            columns.append(
                {
                    "key": f"{pair.tuned}|{stratum}",
                    "base": aot.plddt_by_stratum(base, stratum),
                    "tuned": aot.plddt_by_stratum(tuned, stratum),
                }
            )
    band = aot.bands(columns, family)
    decisions: dict[str, Any] = {}
    for pair in pairs:
        base, tuned = aot.pool_of(table_ptm, pair.base), aot.pool_of(table_ptm, pair.tuned)
        per_stratum: dict[str, Any] = {}
        for stratum in ota.STRATA:
            entry = band["columns"][f"{pair.tuned}|{stratum}"]
            if entry["status"] != "resolved":
                per_stratum[stratum] = {
                    "finding": ota.UNRESOLVED,
                    "reason": entry["reason"],
                    "margin": None,
                    "streams": None,
                }
                continue
            margin = ptm_margin_for(entry["base_mean"])
            left, right = aot.plddt_by_stratum(base, stratum), aot.plddt_by_stratum(tuned, stratum)
            streams: dict[str, Any] = {}
            for stream in sorted(set(left) & set(right)):
                difference = float(np.mean(right[stream]) - np.mean(left[stream]))
                streams[str(stream)] = {
                    "difference": difference,
                    "exceeds_margin": bool(difference > float(margin["margin"])),
                    "units": {"base": len(left[stream]), "tuned": len(right[stream])},
                }
            per_stratum[stratum] = {
                **ota.margin_finding(entry["interval"], float(margin["margin"])),
                "margin": margin,
                "streams": streams,
            }
        split = None if shift is None else shift["shift_share"].get(pair.tuned)
        moves = (
            None
            if shift is None
            else {
                ota.IDENTITY_MEAN_FAMILY: shift["mean_identity"]["columns"][pair.tuned],
                ota.IDENTITY_FRACTION_FAMILY: shift["fraction_below_30"]["columns"][pair.tuned],
            }
        )
        decisions[pair.tuned] = {
            "per_stratum": per_stratum,
            "pair_decision": aot.pair_decision(per_stratum, split=split, moves=moves),
        }
    return {
        "bands": band,
        "per_pair": decisions,
        "endpoint": "mean ESMFold2 pTM, tuned minus base, within stratum",
        "units": "ESMFold2 pTM on the 0-1 scale",
        "margin_calibration": {
            "criterion": float(PTM_MARGIN_CRITERION),
            "floor": float(PTM_MARGIN_FLOOR),
            "fraction": float(PTM_MARGIN_FRACTION),
            "rule": PTM_MARGIN_RULE,
        },
        "decision_rule": (
            "Same 3-stream simultaneous bootstrap family and conjunction rule as pLDDT, calibrated "
            "strictly on the 0-1 ESMFold2 pTM scale (criterion=0.50, floor=0.02, fraction=0.25)."
        ),
    }


def _compact_contrast_rows(band: dict[str, Any], decisions: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = []
    for key, col in sorted(band["columns"].items()):
        pair, stratum = key.split("|")
        base_u = {int(s): int(v) for s, v in (col.get("units", {}).get("base") or {}).items()}
        tuned_u = {int(s): int(v) for s, v in (col.get("units", {}).get("tuned") or {}).items()}
        b_min = min(base_u.values()) if base_u else None
        t_min = min(tuned_u.values()) if tuned_u else None
        interval = col.get("interval") or [None, None]
        cell = None if decisions is None else decisions[pair]["per_stratum"][stratum]
        rows.append(
            {
                "pair": pair,
                "stratum": stratum,
                "status": col["status"],
                "n_base": sum(base_u.values()) if base_u else 0,
                "n_tuned": sum(tuned_u.values()) if tuned_u else 0,
                "base_min_stream": b_min,
                "tuned_min_stream": t_min,
                "thin_under_30": bool((b_min is not None and b_min < 30) or (t_min is not None and t_min < 30)),
                "base_mean": col.get("base_mean"),
                "tuned_mean": col.get("tuned_mean"),
                "diff": col.get("point"),
                "band_lo": interval[0],
                "band_hi": interval[1],
                "band_half_width": (
                    None if interval[0] is None or interval[1] is None else 0.5 * (interval[1] - interval[0])
                ),
                "critical_value": band.get("critical_value"),
                "margin": None if cell is None else (cell.get("margin") or {}).get("margin"),
                "finding": None if cell is None else cell["finding"],
                "zero_verdict": col.get("zero_verdict"),
                "reason": col.get("reason") or (None if cell is None else cell.get("reason")),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Section (a) supplemental: Termination-Split Identity-Stratified Contrasts
# --------------------------------------------------------------------------- #


def _run_termination_split_identity_contrasts(df: pd.DataFrame, *, draws: int = 10000, seed: int = 20261114) -> dict[str, Any]:
    """Exact ``identity_stratified_contrast.py`` logic on full 47,778 structures (pooled/finished/capped)."""
    rng = np.random.default_rng(seed)
    order = list(hm.STRATUM_NAMES)
    thin_floor = 30
    rows: list[dict[str, Any]] = []
    devs: dict[tuple[str, str, str], np.ndarray] = {}

    def _boot_diff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Chunked to keep peak RAM small even at N~6000 x B=10000.
        out = np.empty(draws, dtype=np.float64)
        chunk = 1000
        for start in range(0, draws, chunk):
            stop = min(draws, start + chunk)
            k = stop - start
            ix = rng.integers(0, len(x), size=(k, len(x)))
            iy = rng.integers(0, len(y), size=(k, len(y)))
            out[start:stop] = x[ix].mean(axis=1) - y[iy].mean(axis=1)
        return out

    for pair, b_arm, t_arm in PAIRS_SHORT:
        for stratum in order:
            gs = df[df["stratum"] == stratum]
            for term in ("pooled", "finished", "capped"):
                g = gs if term == "pooled" else (gs[~gs["capped"]] if term == "finished" else gs[gs["capped"]])
                xb = g[g["arm"] == b_arm]
                xt = g[g["arm"] == t_arm]
                rec: dict[str, Any] = {
                    "pair": pair,
                    "stratum": stratum,
                    "termination": term,
                    "n_base": int(len(xb)),
                    "n_tuned": int(len(xt)),
                }
                if len(xb) < 2 or len(xt) < 2:
                    rec["status"] = "not estimable (an arm has <2 rows)"
                    rec["reading"] = "not estimable"
                    rows.append(rec)
                    continue
                yb, yt = xb["plddt"].to_numpy(dtype=np.float64), xt["plddt"].to_numpy(dtype=np.float64)
                diff = float(yt.mean() - yb.mean())
                bs = _boot_diff(yt, yb)
                margin = max(2.0, 0.25 * max(0.0, 70.0 - float(yb.mean())))
                is_thin = bool(min(len(xb), len(xt)) < thin_floor)
                rec.update(
                    status="estimable",
                    base_mean=float(yb.mean()),
                    tuned_mean=float(yt.mean()),
                    diff=diff,
                    marg_lo=float(np.percentile(bs, 2.5)),
                    marg_hi=float(np.percentile(bs, 97.5)),
                    margin=margin,
                    thin=is_thin,
                    ptm_base_mean=float(xb["ptm"].mean()),
                    ptm_tuned_mean=float(xt["ptm"].mean()),
                    ptm_diff=float(xt["ptm"].mean() - xb["ptm"].mean()),
                    mean_identity_base=float(xb["identity"].mean()),
                    mean_identity_tuned=float(xt["identity"].mean()),
                )
                devs[(pair, stratum, term)] = bs - diff
                rows.append(rec)

    fam_a = [k for k in devs if k[2] == "pooled"]
    thin_lookup = {(r["pair"], r["stratum"], r["termination"]): r.get("thin", True) for r in rows}
    fam_b = [k for k in devs if k[2] != "pooled" and not thin_lookup[k]]
    hw_a = float(np.quantile(np.max(np.abs(np.column_stack([devs[k] for k in fam_a])), axis=1), 0.95))
    hw_b = float(np.quantile(np.max(np.abs(np.column_stack([devs[k] for k in fam_b])), axis=1), 0.95))

    for rec in rows:
        k = (rec["pair"], rec["stratum"], rec["termination"])
        if rec.get("status") != "estimable":
            rec["family"] = "none"
            rec["band_half_width"] = None
            continue
        if k in fam_a:
            rec["family"] = "A_declared_pairs_x_strata"
            hw = hw_a
        elif k in fam_b:
            rec["family"] = f"B_split_cells_both_arms_ge_{thin_floor}"
            hw = hw_b
        else:
            rec["family"] = "none"
            rec["band_half_width"] = None
            rec["reading"] = "thin: no simultaneous reading"
            continue
        rec["band_half_width"] = hw
        lo, hi = rec["diff"] - hw, rec["diff"] + hw
        rec["sim_lo"] = lo
        rec["sim_hi"] = hi
        m = rec["margin"]
        if lo > m:
            rec["reading"] = "exceeds margin (positive)"
        elif hi < -m:
            rec["reading"] = "beyond margin (negative)"
        elif lo > -m and hi < m:
            rec["reading"] = "bounded inside margin"
        else:
            rec["reading"] = "unresolved"

    # Secondary identity shift by termination
    id_shift_rows = []
    for pair, b_arm, t_arm in PAIRS_SHORT:
        for term in ("pooled", "finished", "capped"):
            g = df if term == "pooled" else (df[~df["capped"]] if term == "finished" else df[df["capped"]])
            xb = g[g["arm"] == b_arm]["identity"].to_numpy(dtype=np.float64)
            xt = g[g["arm"] == t_arm]["identity"].to_numpy(dtype=np.float64)
            bi = _boot_diff(xt, xb)
            bf = _boot_diff((xt < 30.0).astype(np.float64), (xb < 30.0).astype(np.float64))
            id_shift_rows.append(
                {
                    "pair": pair,
                    "termination": term,
                    "n_base": int(len(xb)),
                    "n_tuned": int(len(xt)),
                    "mean_identity_base": float(xb.mean()),
                    "mean_identity_tuned": float(xt.mean()),
                    "mean_identity_shift": float(xt.mean() - xb.mean()),
                    "shift_ci": _ci95(bi),
                    "frac_below30_base": float((xb < 30.0).mean()),
                    "frac_below30_tuned": float((xt < 30.0).mean()),
                    "frac_below30_shift": float((xt < 30.0).mean() - (xb < 30.0).mean()),
                    "frac_below30_ci": _ci95(bf),
                }
            )

    return {
        "draws": draws,
        "seed": seed,
        "family_A_columns": len(fam_a),
        "family_A_half_width_plddt": hw_a,
        "family_B_columns": len(fam_b),
        "family_B_half_width_plddt": hw_b,
        "cells": rows,
        "identity_shift_by_termination": id_shift_rows,
    }


def _secondary_identity_and_recognition_summary(
    table: dict[str, Any],
    pair: ot.ObjectiveTunedPair,
    *,
    draws: int = 2000,
    seed: int = 20261124,
) -> dict[str, Any]:
    """Descriptive 3-stream identity shift & Pfam recognition for the secondary pair (not in primary 3-pair family)."""
    base, tuned = aot.require_pair(table, pair)
    rng = np.random.default_rng(seed)

    def _stream_diff(getter) -> dict[str, Any]:
        b_s = {s: np.asarray(getter(pool), dtype=np.float64) for s, pool in sorted(base.items())}
        t_s = {s: np.asarray(getter(pool), dtype=np.float64) for s, pool in sorted(tuned.items())}
        b_mean = float(np.mean([v.mean() for v in b_s.values()]))
        t_mean = float(np.mean([v.mean() for v in t_s.values()]))
        bs = []
        for _ in range(draws):
            bm = float(np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in b_s.values()]))
            tm = float(np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in t_s.values()]))
            bs.append(tm - bm)
        return {
            "base_mean": b_mean,
            "tuned_mean": t_mean,
            "diff": t_mean - b_mean,
            "ci95": _ci95(bs),
            "per_stream_diff": {str(s): float(t_s[s].mean() - b_s[s].mean()) for s in sorted(base)},
        }

    return {
        "pair": pair.tuned,
        "note": "Descriptive 3-stream-equal summary (secondary pair is outside the primary 3-pair declared identity/recognition families).",
        "mean_identity": _stream_diff(lambda p: [r["identity"] for r in p["rows"] if r["identity"] is not None]),
        "fraction_below_30": _stream_diff(
            lambda p: [float(r["identity"] is not None and r["identity"] < ota.NOVEL_IDENTITY_EDGE) for r in p["rows"]]
        ),
        "any_family": _stream_diff(lambda p: [float(r["any_family"]) for r in p["rows"]]),
        "complete_domain": _stream_diff(lambda p: [float(r["complete_domain"]) for r in p["rows"]]),
    }


# --------------------------------------------------------------------------- #
# Section (b): ProGen2-small-mix7 7-Family Attribution on Full 68/68 Cohort
# --------------------------------------------------------------------------- #


def _run_progen2_mix7_attribution(
    table: dict[str, Any],
    counts: Any,
    secondary_full: dict[str, Any],
) -> dict[str, Any]:
    """Remove all 7 fine-tuning Pfam families from both ProGen2 arms and recompute 3-stream contrast."""
    pair = ota.SECONDARY_PAIRS[0]
    arms = set(pair.arms)
    seven_set = set(SEVEN_FAMILIES)
    pools_corr: dict[Any, dict[str, Any]] = {}
    exclusion_rows: list[dict[str, Any]] = []
    for (arm, stream), pool in sorted(table["pools"].items()):
        if arm not in arms:
            pools_corr[(arm, stream)] = pool
            continue
        kept = [r for r in pool["rows"] if r["empty"] or not (set(r["pfam_families"]) & seven_set)]
        pools_corr[(arm, stream)] = {**pool, "rows": kept}
        kept_keys = {r["key"] for r in kept if not r["empty"]}
        for stratum in (*ota.STRATA, "all_strata"):
            members = [
                r
                for r in pool["rows"]
                if not r["empty"] and (stratum == "all_strata" or r["stratum"] == stratum)
            ]
            left = sum(1 for r in members if r["key"] in kept_keys)
            exclusion_rows.append(
                {
                    "arm": arm,
                    "stream": stream,
                    "stratum": stratum,
                    "units_before": len(members),
                    "units_in_seven_families": len(members) - left,
                    "units_left": left,
                    "below_floor_after_removal": left < ota.MIN_UNITS_PER_POOL,
                    "thin_under_30_after_removal": left < 30,
                }
            )
    corrected_table = {**table, "pools": pools_corr}
    secondary_corr = aot.stratified_contrast(corrected_table, ota.SECONDARY_PAIRS, ota.SECONDARY_FAMILY)
    secondary_ctrl = aot.explicit_statistics_control(
        table, ota.SECONDARY_PAIRS, counts, secondary_full, ota.SECONDARY_CONTROL_FAMILIES
    )
    return {
        "seven_finetuning_families": list(SEVEN_FAMILIES),
        "exclusion_counts": exclusion_rows,
        "full_contrast": _compact_contrast_rows(secondary_full["bands"], secondary_full["per_pair"]),
        "out_of_seven_families_contrast": _compact_contrast_rows(
            secondary_corr["bands"], secondary_corr["per_pair"]
        ),
        "secondary_explicit_statistics_control": secondary_ctrl,
    }


# --------------------------------------------------------------------------- #
# Section (c): Likelihood-Repetitiveness & Support Containment (47,778 rows)
# --------------------------------------------------------------------------- #


def _run_likelihood_repetitiveness_and_containment(root: Path, df: pd.DataFrame) -> dict[str, Any]:
    """Exact ``run_likelihood_repetitiveness.py`` and ``run_support_containment.py`` on full cohort."""
    own_dir = root / "results/extensions/objective_tuned_endpoints_20261112/selector/own_model_likelihood"
    cross_dir = root / "results/external_baseline_analysis/objective_tuned_cross_likelihood_20261114"
    df_by_sha = df.set_index("sequence_sha256")
    edges = ((16, 128), (129, 256), (257, 512), (513, 10**9))
    min_n = 30
    draws = 2000
    seed = 20261114

    def _sp(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            return float("nan")
        return float(np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1])

    def _sp_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
        pt = _sp(x, y)
        bs = [_sp(x[idx], y[idx]) for idx in (rng.integers(0, x.size, x.size) for _ in range(draws))]
        return {"n": int(x.size), "spearman": pt, "ci": _ci95(bs)}

    def _outside(base: np.ndarray, tuned: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
        low, high = float(base.min()), float(base.max())
        below, above = tuned < low, tuned > high
        flags = (below | above).astype(np.float64)
        bs = [flags[rng.integers(0, flags.size, flags.size)].mean() for _ in range(draws)]
        ci = _ci95(bs)
        reading = "inside" if (ci[1] is not None and ci[1] <= 0.01) else ("outside" if (ci[0] is not None and ci[0] > 0.05) else "partly_outside")
        return {
            "n_tuned": int(tuned.size),
            "n_base": int(base.size),
            "range": [low, high],
            "n_below_range": int(below.sum()),
            "n_above_range": int(above.sum()),
            "fraction": float(flags.mean()),
            "ci": ci,
            "reading": reading,
        }

    rep_out: dict[str, Any] = {}
    cont_out: dict[str, Any] = {}
    for pair in (*ota.PRIMARY_PAIRS, *ota.SECONDARY_PAIRS):
        b, t = pair.base, pair.tuned
        rng_r = np.random.default_rng(seed)
        tb = ota.read_own_likelihood(own_dir / f"{b}__on__{b}.jsonl")
        tt = ota.read_own_likelihood(own_dir / f"{t}__on__{t}.jsonl")
        bd_ids, td_ids = sorted(tb), sorted(tt)
        b_len = df_by_sha.loc[bd_ids, "L"].to_numpy(dtype=np.float64)
        b_nll = np.array([tb[d][1] / tb[d][0] for d in bd_ids], dtype=np.float64)
        b_d5 = df_by_sha.loc[bd_ids, "distinct_5mer_frac"].to_numpy(dtype=np.float64)
        b_run = df_by_sha.loc[bd_ids, "max_run"].to_numpy(dtype=np.float64)
        t_len = df_by_sha.loc[td_ids, "L"].to_numpy(dtype=np.float64)
        t_nll = np.array([tt[d][1] / tt[d][0] for d in td_ids], dtype=np.float64)
        t_d5 = df_by_sha.loc[td_ids, "distinct_5mer_frac"].to_numpy(dtype=np.float64)
        t_run = df_by_sha.loc[td_ids, "max_run"].to_numpy(dtype=np.float64)

        within_l = {}
        for lo, hi in edges:
            m = (b_len >= lo) & (b_len <= hi)
            if int(m.sum()) >= min_n:
                within_l[f"{lo}_{hi}"] = {
                    "distinct5": _sp_ci(b_nll[m], b_d5[m], rng_r),
                    "longest_run": _sp_ci(b_nll[m], b_run[m], rng_r),
                }
        order = sorted(range(len(bd_ids)), key=lambda i: (-b_d5[i], bd_ids[i]))
        top_q = order[: len(bd_ids) // 4]
        rep_out[b] = {
            "n": len(bd_ids),
            "tuned": t,
            "overall": {
                "distinct5": _sp_ci(b_nll, b_d5, rng_r),
                "longest_run": _sp_ci(b_nll, b_run, rng_r),
            },
            "within_length_stratum": within_l,
            "context_means": {
                "base": {
                    "distinct5": float(b_d5.mean()),
                    "longest_run": float(b_run.mean()),
                    "mean_nll": float(b_nll.mean()),
                    "mean_length": float(b_len.mean()),
                    "n": len(bd_ids),
                },
                "tuned": {
                    "distinct5": float(t_d5.mean()),
                    "longest_run": float(t_run.mean()),
                    "mean_nll": float(t_nll.mean()),
                    "mean_length": float(t_len.mean()),
                    "n": len(td_ids),
                },
                "base_top_quarter_by_distinct5": {
                    "distinct5": float(b_d5[top_q].mean()),
                    "longest_run": float(b_run[top_q].mean()),
                    "mean_nll": float(b_nll[top_q].mean()),
                    "mean_length": float(b_len[top_q].mean()),
                    "n": len(top_q),
                },
            },
        }

        # Support containment from cross-likelihood files
        cb_map = ota.read_own_likelihood(cross_dir / f"{b}__on__{b}.jsonl")
        ct_map = ota.read_own_likelihood(cross_dir / f"{b}__on__{t}.jsonl")
        xb = np.array([cb_map[d][1] / cb_map[d][0] for d in bd_ids], dtype=np.float64)
        xt = np.array([ct_map[d][1] / ct_map[d][0] for d in td_ids], dtype=np.float64)
        rng_c = np.random.default_rng(seed)
        strata_c = {}
        for lo, hi in edges:
            mb = (b_len >= lo) & (b_len <= hi)
            mt = (t_len >= lo) & (t_len <= hi)
            if min(int(mb.sum()), int(mt.sum())) >= min_n:
                strata_c[f"{lo}_{hi}"] = {
                    "n": [int(mt.sum()), int(mb.sum())],
                    "mixture": acl.contrast(xt[mt], t_len[mt], xb[mb], b_len[mb], rng_c),
                    "entropy_change_h200": acl.contrast(t_nll[mt], t_len[mt], b_nll[mb], b_len[mb], rng_c),
                    "outside_within_stratum": _outside(xb[mb], xt[mt], rng_c),
                }
        cont_out[t] = {
            "n_base": len(bd_ids),
            "n_tuned": len(td_ids),
            "host_agreement_max_abs_diff": float(np.max(np.abs(b_nll - xb))),
            "pooled": {
                "raw_difference_of_means": float(xt.mean() - xb.mean()),
                "mixture": acl.contrast(xt, t_len, xb, b_len, rng_c),
                "entropy_change_h200": acl.contrast(t_nll, t_len, b_nll, b_len, rng_c),
                "outside_whole_pool": _outside(xb, xt, rng_c),
            },
            "strata": strata_c,
        }
    return {"likelihood_repetitiveness": rep_out, "support_containment": cont_out}


# --------------------------------------------------------------------------- #
# Section (d): Full 68/68 Length & Cap-Degeneracy Audit + Full 2,000 Long-Cap
# --------------------------------------------------------------------------- #


def _nn_match_fast(lb: np.ndarray, yb: np.ndarray, lt: np.ndarray, yt: np.ndarray, cal: int = 5) -> tuple[float, int]:
    """Exact caliper-5 nearest-length match (identical to ``length_adjusted_contrasts_20261110/analyse.py``)."""
    o = np.argsort(lb)
    ls = lb[o]
    ys = yb[o]
    idx = np.searchsorted(ls, lt)
    left = np.clip(idx - 1, 0, len(ls) - 1)
    right = np.clip(idx, 0, len(ls) - 1)
    dl = np.abs(ls[left] - lt)
    dr = np.abs(ls[right] - lt)
    best_pos = np.where(dl <= dr, left, right)
    best_val = ls[best_pos]
    ok = np.abs(best_val - lt) <= cal
    if not ok.any():
        return float("nan"), 0
    # Average all base ties at best_val via prefix sums
    uniq_l, first_idx, counts = np.unique(ls, return_index=True, return_counts=True)
    csum = np.concatenate([[0.0], np.cumsum(ys)])
    uniq_mean = (csum[first_idx + counts] - csum[first_idx]) / counts
    pos_u = np.searchsorted(uniq_l, best_val[ok])
    diffs = yt[ok] - uniq_mean[pos_u]
    return float(diffs.mean()), int(ok.sum())


def _run_length_and_cap_audit_68of68(df: pd.DataFrame, *, draws: int = 2000) -> dict[str, Any]:
    """Full 68/68-shard length adjustment, cap split, composition mediation, and identity-stratum length matching."""
    rng = np.random.default_rng(20261110)

    def _slope(L: np.ndarray, y: np.ndarray) -> float:
        x = L - L.mean()
        denom = float((x * x).sum())
        return float((x * (y - y.mean())).sum() / denom) if denom > 0 else float("nan")

    def _basis(L: np.ndarray, kn: np.ndarray, flex: bool) -> np.ndarray:
        x = L / 100.0
        if not flex:
            return x[:, None]
        cols = [x, x**2, x**3] + [np.maximum(x - k, 0.0) ** 3 for k in kn]
        return np.column_stack(cols)

    def _reg(Lb: np.ndarray, yb: np.ndarray, Lt: np.ndarray, yt: np.ndarray, kn: np.ndarray, flex: bool) -> float:
        L = np.concatenate([Lb, Lt])
        y = np.concatenate([yb, yt])
        a = np.concatenate([np.zeros(len(Lb)), np.ones(len(Lt))])
        X = np.column_stack([np.ones(len(L)), a, _basis(L, kn, flex)])
        return float(np.linalg.lstsq(X, y, rcond=None)[0][1])

    # 1. Slopes in 257-512 + contrasts across 3 length strata (raw, lin, spline, bin16, nn5)
    slopes_257_512 = []
    contrasts_by_stratum = []
    for ep in ("plddt", "ptm"):
        for pair, b_arm, t_arm in PAIRS_SHORT:
            for role, arm in (("base", b_arm), ("tuned", t_arm)):
                s = df[(df["arm"] == arm) & (df["L"] >= 257) & (df["L"] <= 512)]
                L = s["L"].to_numpy(dtype=np.float64)
                y = s[ep].to_numpy(dtype=np.float64)
                pt = _slope(L, y) * 100.0
                bs = [_slope(L[i], y[i]) * 100.0 for i in (rng.integers(0, len(L), len(L)) for _ in range(draws))]
                slopes_257_512.append(
                    {
                        "endpoint": ep,
                        "pair": pair,
                        "role": role,
                        "arm": arm,
                        "n": int(len(L)),
                        "per100": pt,
                        "ci": _ci95(bs),
                        "frac_at_400": float((L == CAP_400).mean()),
                        "mean_L": float(L.mean()),
                    }
                )
        for sname, lo, hi in LENGTH_STRATA:
            for pair, b_arm, t_arm in PAIRS_SHORT:
                sb = df[(df["arm"] == b_arm) & (df["L"] >= lo) & (df["L"] <= hi)]
                st = df[(df["arm"] == t_arm) & (df["L"] >= lo) & (df["L"] <= hi)]
                Lb, yb = sb["L"].to_numpy(dtype=np.float64), sb[ep].to_numpy(dtype=np.float64)
                Lt, yt = st["L"].to_numpy(dtype=np.float64), st[ep].to_numpy(dtype=np.float64)
                rec: dict[str, Any] = {
                    "endpoint": ep,
                    "stratum": sname,
                    "pair": pair,
                    "n_base": int(len(Lb)),
                    "n_tuned": int(len(Lt)),
                }
                if len(Lb) < 2 or len(Lt) < 2:
                    contrasts_by_stratum.append(rec)
                    continue
                kn = np.percentile(np.concatenate([Lb, Lt]) / 100.0, [25, 50, 75])
                pt_raw = float(yt.mean() - yb.mean())
                pt_lin = _reg(Lb, yb, Lt, yt, kn, False)
                pt_flex = _reg(Lb, yb, Lt, yt, kn, True)
                pt_bin, bin_t, bin_b = lm.bin_match(Lb, yb, Lt, yt, width=16)
                pt_nn, nn_t = _nn_match_fast(Lb, yb, Lt, yt, cal=5)
                bs_raw, bs_lin, bs_flex, bs_bin, bs_nn = [], [], [], [], []
                for _ in range(draws):
                    ib = rng.integers(0, len(Lb), len(Lb))
                    it = rng.integers(0, len(Lt), len(Lt))
                    bs_raw.append(float(yt[it].mean() - yb[ib].mean()))
                    bs_lin.append(_reg(Lb[ib], yb[ib], Lt[it], yt[it], kn, False))
                    bs_flex.append(_reg(Lb[ib], yb[ib], Lt[it], yt[it], kn, True))
                    bs_bin.append(lm.bin_match(Lb[ib], yb[ib], Lt[it], yt[it], width=16)[0])
                    bs_nn.append(_nn_match_fast(Lb[ib], yb[ib], Lt[it], yt[it], cal=5)[0])
                rec.update(
                    gap_L_base_minus_tuned=float(Lb.mean() - Lt.mean()),
                    frac400_base=float((Lb == CAP_400).mean()),
                    frac400_tuned=float((Lt == CAP_400).mean()),
                    bin_tuned_kept=int(bin_t),
                    bin_base_kept=int(bin_b),
                    nn_tuned_kept=int(nn_t),
                    raw=pt_raw,
                    raw_ci=_ci95(bs_raw),
                    lin=pt_lin,
                    lin_ci=_ci95(bs_lin),
                    flex=pt_flex,
                    flex_ci=_ci95(bs_flex),
                    bin16=float(pt_bin),
                    bin16_ci=_ci95(bs_bin),
                    nn5=float(pt_nn),
                    nn5_ci=_ci95(bs_nn),
                )
                contrasts_by_stratum.append(rec)

    # 2. Pooled uncapped (L < 400) contrasts + capped (L == 400) vs uncapped split & composition mediation
    rng2 = np.random.default_rng(20261112)
    uncapped_pooled = []
    cap_split = []
    capped_mediation = []
    comp_cols = ["top_frac", "entropy_bits", "max_run", "zlib_ratio", "distinct_4mer_frac"]
    u_df = df[~df["capped"]]
    c_df = df[df["capped"]]
    s257 = df[(df["L"] >= 257) & (df["L"] <= 512)]

    for ep in ("plddt", "ptm"):
        for pair, b_arm, t_arm in PAIRS_SHORT:
            # Uncapped pooled
            sb, st = u_df[u_df["arm"] == b_arm], u_df[u_df["arm"] == t_arm]
            Lb, yb = sb["L"].to_numpy(dtype=np.float64), sb[ep].to_numpy(dtype=np.float64)
            Lt, yt = st["L"].to_numpy(dtype=np.float64), st[ep].to_numpy(dtype=np.float64)
            kn = np.percentile(np.concatenate([Lb, Lt]) / 100.0, [20, 40, 60, 80])
            pt_bin, kt, kb = lm.bin_match(Lb, yb, Lt, yt, width=16)
            pt_vals = (float(yt.mean() - yb.mean()), _reg(Lb, yb, Lt, yt, kn, False), _reg(Lb, yb, Lt, yt, kn, True), float(pt_bin))
            bs_u = np.array(
                [
                    (
                        float(yt[it].mean() - yb[ib].mean()),
                        _reg(Lb[ib], yb[ib], Lt[it], yt[it], kn, False),
                        _reg(Lb[ib], yb[ib], Lt[it], yt[it], kn, True),
                        lm.bin_match(Lb[ib], yb[ib], Lt[it], yt[it], width=16)[0],
                    )
                    for ib, it in ((rng2.integers(0, len(Lb), len(Lb)), rng2.integers(0, len(Lt), len(Lt))) for _ in range(draws))
                ]
            )
            uncapped_pooled.append(
                {
                    "endpoint": ep,
                    "pair": pair,
                    "n_base": int(len(Lb)),
                    "n_tuned": int(len(Lt)),
                    "gap_L_base_minus_tuned": float(Lb.mean() - Lt.mean()),
                    "bin_tuned_kept": int(kt),
                    "bin_base_kept": int(kb),
                    "raw": pt_vals[0],
                    "raw_ci": _ci95(bs_u[:, 0]),
                    "lin": pt_vals[1],
                    "lin_ci": _ci95(bs_u[:, 1]),
                    "spline": pt_vals[2],
                    "spline_ci": _ci95(bs_u[:, 2]),
                    "bin16": pt_vals[3],
                    "bin16_ci": _ci95(bs_u[:, 3]),
                }
            )

            # Cap split inside 257-512
            for lab, mask_fn in (("L==400 (budget cap)", lambda z: z["L"] == CAP_400), ("L<400 (ended before cap)", lambda z: z["L"] < CAP_400)):
                yb_s = s257[(s257["arm"] == b_arm) & mask_fn(s257)][ep].to_numpy(dtype=np.float64)
                yt_s = s257[(s257["arm"] == t_arm) & mask_fn(s257)][ep].to_numpy(dtype=np.float64)
                bs_s = [float(yt_s[rng2.integers(0, len(yt_s), len(yt_s))].mean() - yb_s[rng2.integers(0, len(yb_s), len(yb_s))].mean()) for _ in range(draws)]
                cap_split.append(
                    {
                        "endpoint": ep,
                        "pair": pair,
                        "subset": lab,
                        "n_base": int(len(yb_s)),
                        "n_tuned": int(len(yt_s)),
                        "diff": float(yt_s.mean() - yb_s.mean()),
                        "ci": _ci95(bs_s),
                    }
                )

            # Capped (L == 400) composition mediation
            cdf = c_df[c_df["arm"].isin([b_arm, t_arm])].reset_index(drop=True)
            y_c = cdf[ep].to_numpy(dtype=np.float64)
            a_c = (cdf["arm"] == t_arm).to_numpy(dtype=np.float64)
            cov_mat = np.column_stack(
                [
                    (np.log(cdf[m].to_numpy(dtype=np.float64)) if m == "max_run" else cdf[m].to_numpy(dtype=np.float64))
                    for m in comp_cols
                ]
            )
            cov_mat = (cov_mat - cov_mat.mean(axis=0)) / cov_mat.std(axis=0)
            X0 = np.column_stack([np.ones(len(cdf)), a_c])
            X1 = np.column_stack([np.ones(len(cdf)), a_c, cov_mat])
            b0 = float(np.linalg.lstsq(X0, y_c, rcond=None)[0][1])
            beta1 = np.linalg.lstsq(X1, y_c, rcond=None)[0]
            b1 = float(beta1[1])
            r2 = float(1.0 - ((y_c - X1 @ beta1) ** 2).sum() / ((y_c - y_c.mean()) ** 2).sum())
            bs_m = []
            for _ in range(draws):
                idx = rng2.integers(0, len(cdf), len(cdf))
                c_b0 = float(np.linalg.lstsq(X0[idx], y_c[idx], rcond=None)[0][1])
                c_b1 = float(np.linalg.lstsq(X1[idx], y_c[idx], rcond=None)[0][1])
                bs_m.append((c_b0, c_b1))
            bs_m_arr = np.asarray(bs_m)
            capped_mediation.append(
                {
                    "endpoint": ep,
                    "pair": pair,
                    "n_base": int((cdf["arm"] == b_arm).sum()),
                    "n_tuned": int((cdf["arm"] == t_arm).sum()),
                    "arm_only": b0,
                    "arm_only_ci": _ci95(bs_m_arr[:, 0]),
                    "arm_plus_composition": b1,
                    "arm_plus_composition_ci": _ci95(bs_m_arr[:, 1]),
                    "r2_composition_plus_arm": r2,
                }
            )

    # 3. Length-matched contrasts inside ALL 4 identity strata (pooled/finished/capped): KS, median gap, share
    id_length_matched = []
    for pair, b_arm, t_arm in PAIRS_SHORT:
        b_df = df[df["arm"] == b_arm]
        t_df = df[df["arm"] == t_arm]
        for stratum in hm.STRATUM_NAMES:
            sb0, st0 = b_df[b_df["stratum"] == stratum], t_df[t_df["stratum"] == stratum]
            for term, sb, st in (
                ("pooled", sb0, st0),
                ("finished", sb0[~sb0["capped"]], st0[~st0["capped"]]),
                ("capped", sb0[sb0["capped"]], st0[st0["capped"]]),
            ):
                both_ok = len(sb) >= 2 and len(st) >= 2
                raw = float(st["plddt"].mean() - sb["plddt"].mean()) if len(sb) and len(st) else None
                ks = float(stats.ks_2samp(sb["L"], st["L"]).statistic) if both_ok else None
                med_gap = float(st["L"].median() - sb["L"].median()) if both_ok else None
                matched_16 = None
                one_length = (
                    len(sb) > 0 and len(st) > 0 and sb["L"].nunique() == 1 and st["L"].nunique() == 1
                    and int(sb["L"].iloc[0]) == int(st["L"].iloc[0])
                )
                if not one_length and min(len(sb), len(st)) >= lm.MIN_ROWS:
                    v, _, _ = lm.bin_match(
                        sb["L"].to_numpy(dtype=np.float64),
                        sb["plddt"].to_numpy(dtype=np.float64),
                        st["L"].to_numpy(dtype=np.float64),
                        st["plddt"].to_numpy(dtype=np.float64),
                        width=16,
                    )
                    matched_16 = float(v)
                id_length_matched.append(
                    {
                        "pair": pair,
                        "stratum": stratum,
                        "termination": term,
                        "n_base": int(len(sb)),
                        "n_tuned": int(len(st)),
                        "base_median_L": float(sb["L"].median()) if len(sb) else None,
                        "tuned_median_L": float(st["L"].median()) if len(st) else None,
                        "median_gap": med_gap,
                        "ks": ks,
                        "material": bool(lm.is_material(med_gap, ks)) if both_ok else None,
                        "raw_plddt_diff": raw,
                        "matched_16bin_plddt": matched_16,
                        "length_attributable_share": lm.attributable_share(raw, matched_16),
                    }
                )

    return {
        "slopes_257_512": slopes_257_512,
        "contrasts_by_length_stratum": contrasts_by_stratum,
        "uncapped_pooled_contrasts": uncapped_pooled,
        "cap_split_257_512": cap_split,
        "capped_L400_composition_mediation": capped_mediation,
        "identity_stratum_length_matched": id_length_matched,
    }


def _run_long_cap_companion_2000of2000(root: Path, full_lc_tsv: Path, *, draws: int = 2000) -> dict[str, Any]:
    """Exact ``analyse_companion.py`` re-run on ALL 2,000/2,000 folded long-cap sequences (0 subsampling)."""
    rng = np.random.default_rng(20261111)
    lc_dir = root / "results/long_cap_termination_20261110"
    t = pd.read_csv(full_lc_tsv, sep="\t")
    assert len(t) == 2000 and (t["status"] == "ok").all()
    seq_map = {str(r["id"]): str(r["sequence"]) for r in _read_jsonl(lc_dir / "fold/fold_cohort.jsonl")}
    # Also attach any sequences from attempts.jsonl (fold_cohort.jsonl has the 1,748; attempts.jsonl has all 2,000).
    for arm in ("protgpt3-10b", "protgpt3-10b-dpo"):
        for r in _read_jsonl(lc_dir / "out" / arm / "attempts.jsonl"):
            seq_map[str(r["sequence_sha256"])] = str(r["sequence"])

    nov = pd.DataFrame(_read_jsonl(lc_dir / "novelty/search/generated_novelty.jsonl"))
    nov = nov.rename(columns={"nearest_corpus_identity": "identity"})[["id", "identity", "had_any_hit"]]
    nov["identity"] = nov["identity"].fillna(0.0)
    t = t.merge(nov, left_on="sequence_sha256", right_on="id", how="left", validate="one_to_one")
    t["L"] = t["length"].astype(float)
    t["tuned"] = (t["arm_tag"] == "tuned").astype(float)
    t["capped"] = t["stop_status"] == "budget_censored"
    t["stratum_id"] = t["identity"].map(hm.assign_stratum)

    z9, d3 = [], []
    for sha in t["sequence_sha256"]:
        s = seq_map[sha]
        z9.append(len(zlib.compress(s.encode("utf-8"), 9)) / len(s))
        d3.append(len({s[i : i + 3] for i in range(len(s) - 2)}) / max(1, len(s) - 2))
    t["zlib9"] = z9
    t["d3"] = d3
    t["degenerate"] = t["zlib9"] < ZLIB_DEGENERATE_THRESHOLD

    order = list(hm.STRATUM_NAMES)

    def _basis(L: np.ndarray, kn: np.ndarray, flex: bool) -> np.ndarray:
        cols = [L / 100.0]
        if flex:
            cols += [(L / 100.0) ** 2, (L / 100.0) ** 3] + [np.maximum(L / 100.0 - k, 0.0) ** 3 for k in kn]
        return np.column_stack(cols)

    def _reg(d: pd.DataFrame, ep: str, kn: np.ndarray, flex: bool, extra: tuple[str, ...] = (), strata: bool = False) -> float:
        X_list = [np.ones(len(d)), d["tuned"].to_numpy(), _basis(d["L"].to_numpy(), kn, flex)]
        if strata:
            for sname in order[1:]:
                X_list.append((d["stratum_id"] == sname).to_numpy(dtype=np.float64)[:, None])
        for e in extra:
            vals = d[e].to_numpy(dtype=np.float64)
            X_list.append(((vals - vals.mean()) / vals.std())[:, None])
        X = np.column_stack(X_list)
        return float(np.linalg.lstsq(X, d[ep].to_numpy(dtype=np.float64), rcond=None)[0][1])

    def _bm(d: pd.DataFrame, ep: str) -> float:
        b = d[d["tuned"] == 0]
        tu = d[d["tuned"] == 1]
        v, _, _ = lm.bin_match(b["L"].to_numpy(), b[ep].to_numpy(), tu["L"].to_numpy(), tu[ep].to_numpy(), width=16)
        return float(v)

    def _boot_df(d: pd.DataFrame, fn) -> list[Any]:
        b = d[d["tuned"] == 0].reset_index(drop=True)
        tu = d[d["tuned"] == 1].reset_index(drop=True)
        out_b = []
        for _ in range(draws):
            ib = rng.integers(0, len(b), len(b))
            it = rng.integers(0, len(tu), len(tu))
            out_b.append(fn(pd.concat([b.iloc[ib], tu.iloc[it]], ignore_index=True)))
        return out_b

    nat = t[~t["capped"]]
    prim = nat[(nat["L"] >= 16) & (nat["L"] <= 512)]
    cap = t[t["capped"]]

    # 1. Population counts & levels
    pop_levels = []
    for pop, g in t.groupby("population"):
        pop_levels.append(
            {
                "population": str(pop),
                "n": int(len(g)),
                "plddt_mean": float(g["plddt"].mean()),
                "plddt_sd": float(g["plddt"].std()),
                "ptm_mean": float(g["ptm"].mean()),
                "identity_mean": float(g["identity"].mean()),
                "frac_degenerate": float(g["degenerate"].mean()),
                "zlib9_median": float(g["zlib9"].median()),
                "L_mean": float(g["L"].mean()),
            }
        )

    # 2. Primary natural 16-512 contrasts
    primary_nat = []
    for ep in ("plddt", "ptm"):
        for name, lo, hi in (*LENGTH_STRATA, ("16-512 pooled", 16, 512)):
            d = prim[(prim["L"] >= lo) & (prim["L"] <= hi)]
            kn = np.percentile(d["L"] / 100.0, [25, 50, 75])

            def _est(x: pd.DataFrame, ep: str = ep, kn: np.ndarray = kn) -> dict[str, float]:
                return {
                    "raw": float(x[x["tuned"] == 1][ep].mean() - x[x["tuned"] == 0][ep].mean()),
                    "linear": _reg(x, ep, kn, False),
                    "spline": _reg(x, ep, kn, True),
                    "bin16": _bm(x, ep),
                }

            pt = _est(d)
            bs_list = _boot_df(d, _est)
            primary_nat.append(
                {
                    "endpoint": ep,
                    "stratum": name,
                    "n_base": int((d["tuned"] == 0).sum()),
                    "n_tuned": int((d["tuned"] == 1).sum()),
                    "base_mean": float(d[d["tuned"] == 0][ep].mean()),
                    "tuned_mean": float(d[d["tuned"] == 1][ep].mean()),
                    **{k: pt[k] for k in pt},
                    **{f"{k}_ci": _ci95([b[k] for b in bs_list]) for k in pt},
                }
            )

    # 3. Capped-at-768 contrasts (NOW 100% FOLDED: 321/321 base-capped + all tuned-capped!)
    capped_contrasts = []
    for ep in ("plddt", "ptm"):
        for label, g_a, g_b in (
            ("tuned capped - base capped (full 2000)", cap[cap["tuned"] == 1], cap[cap["tuned"] == 0]),
            ("tuned capped - base capped (non-degenerate)", cap[(cap["tuned"] == 1) & (~cap["degenerate"])], cap[(cap["tuned"] == 0) & (~cap["degenerate"])]),
            ("base capped - base natural", t[t["population"] == "base-capped"], t[t["population"] == "base-natural"]),
            ("tuned capped - tuned natural", t[t["population"] == "tuned-capped"], t[t["population"] == "tuned-natural"]),
        ):
            xa, xb = g_a[ep].to_numpy(dtype=np.float64), g_b[ep].to_numpy(dtype=np.float64)
            bs = [float(xa[rng.integers(0, len(xa), len(xa))].mean() - xb[rng.integers(0, len(xb), len(xb))].mean()) for _ in range(draws)]
            capped_contrasts.append(
                {
                    "contrast": label,
                    "endpoint": ep,
                    "n_a": int(len(xa)),
                    "n_b": int(len(xb)),
                    "mean_a": float(xa.mean()),
                    "mean_b": float(xb.mean()),
                    "diff": float(xa.mean() - xb.mean()),
                    "ci": _ci95(bs),
                }
            )

    # 4. Degeneracy split
    deg_split = []
    for pop in ("base-capped", "tuned-capped", "base-natural", "tuned-natural"):
        g = t[t["population"] == pop]
        for deg in (True, False):
            h = g[g["degenerate"] == deg]
            if len(h):
                vals = h["plddt"].to_numpy(dtype=np.float64)
                bs = [float(vals[rng.integers(0, len(vals), len(vals))].mean()) for _ in range(draws)]
                deg_split.append(
                    {
                        "population": pop,
                        "degenerate": deg,
                        "n": int(len(h)),
                        "plddt_mean": float(vals.mean()),
                        "plddt_ci": _ci95(bs),
                        "ptm_mean": float(h["ptm"].mean()),
                        "identity_mean": float(h["identity"].mean()),
                    }
                )

    return {
        "n_total": int(len(t)),
        "n_folded_ok": int((t["status"] == "ok").sum()),
        "n_newly_folded_base_capped": int((t["source_batch"] == "completed_r6_20261124").sum()),
        "population_levels": pop_levels,
        "primary_natural_16_512": primary_nat,
        "capped_768_contrasts": capped_contrasts,
        "degeneracy_split": deg_split,
    }


# --------------------------------------------------------------------------- #
# Section (e): Full 68/68-Shard 5-mer Tie & Length-Matched Share Audit
# --------------------------------------------------------------------------- #


def _run_five_mer_ties_and_length_68of68(
    table: dict[str, Any],
    counts: Any,
    primary_contrast: dict[str, Any],
    secondary_contrast: dict[str, Any],
) -> dict[str, Any]:
    """Exact ``run_ties_and_length.py`` logic on full 68/68 3-stream pools."""
    rule = ota.filter_rules(counts)[0]
    assert rule.name == "low_complexity_residues"
    codes = {name: idx for idx, name in enumerate(ota.STRATA)}
    matchers = {
        "bin": lambda lb, yb, lt, yt: lm.bin_match(lb, yb, lt, yt, width=16),
        "nn": lambda lb, yb, lt, yt: _nn_match_fast(lb, yb, lt, yt, cal=5),
    }

    def _repr_z(rows: list[dict[str, Any]], keys: set[str], cut: float) -> dict[str, float | None]:
        group = [r for r in rows if r["score"] == cut]
        drawn = [r for r in group if r["key"] in keys]
        out: dict[str, float | None] = {}
        for field in ("length", "plddt"):
            vals = np.asarray([r[field] for r in group], dtype=np.float64)
            chosen = np.asarray([r[field] for r in drawn], dtype=np.float64)
            n, size = len(drawn), len(group)
            spread = float(vals.var(ddof=1)) if size > 1 else 0.0
            if n < 2 or n == size or spread == 0.0:
                out[f"z_{field}_retained_in_group"] = None
            else:
                out[f"z_{field}_retained_in_group"] = float(
                    (chosen.mean() - vals.mean()) / math.sqrt(spread / n * (1.0 - n / size))
                )
        return out

    tie_rows: list[dict[str, Any]] = []
    share_records: dict[str, Any] = {}
    gain_rows: list[dict[str, Any]] = []
    sets = (
        ("primary", ota.PRIMARY_PAIRS, ota.CONTROL_FAMILIES[rule.name], primary_contrast),
        ("secondary", ota.SECONDARY_PAIRS, ota.SECONDARY_CONTROL_FAMILIES[rule.name], secondary_contrast),
    )

    def _stream_equal(
        matcher,
        parent: dict[int, list[dict[str, Any]]],
        other: dict[int, list[dict[str, Any]]],
        stratum: str,
    ) -> tuple[float, int]:
        diffs, kept_cnt = [], 0
        for s in sorted(parent):
            lb = np.asarray([r["length"] for r in parent[s] if r["stratum"] == stratum], dtype=np.float64)
            yb = np.asarray([r["plddt"] for r in parent[s] if r["stratum"] == stratum], dtype=np.float64)
            lt = np.asarray([r["length"] for r in other[s] if r["stratum"] == stratum], dtype=np.float64)
            yt = np.asarray([r["plddt"] for r in other[s] if r["stratum"] == stratum], dtype=np.float64)
            res = matcher(lb, yb, lt, yt)
            diffs.append(float(res[0]))
            kept_cnt += int(res[1])
        return float(np.mean(diffs)), kept_cnt

    for label, pairs, family, contrast in sets:
        columns: dict[str, list[dict[str, Any]]] = {"raw": [], "bin": [], "nn": []}
        points: dict[tuple[str, str], dict[str, Any]] = {}
        for index, pair in enumerate(pairs):
            base, tuned = aot.require_pair(table, pair)
            scored: dict[int, list[dict[str, Any]]] = {}
            retained: dict[int, set[str]] = {}
            for stream in sorted(base):
                scored[stream] = ota.score_pool(
                    [r for r in base[stream]["rows"] if r["plddt"] is not None], rule
                )["scored"]
                for k in (2, 4, 8):
                    ret = ota.rank_and_retain(scored[stream], rule, fraction=1.0 / k)
                    keys = set(ret.pop("retained_keys"))
                    tie_rows.append(
                        {
                            "arm": pair.base,
                            "stream": stream,
                            "k": k,
                            "n_scored": ret["n_scored"],
                            "n_retained": ret["n_retained"],
                            "cut_score": ret["cut_score"],
                            "n_tied_at_cut": ret["n_tied_at_cut"],
                            "tie_group_share_of_pool": ret["n_tied_at_cut"] / ret["n_scored"],
                            "n_retained_at_cut": ret["n_retained_at_cut"],
                            "retained_decided_by_key_share": ret["n_retained_at_cut"] / ret["n_retained"],
                            **_repr_z(scored[stream], keys, ret["cut_score"]),
                        }
                    )
                    if k == 4:
                        retained[stream] = keys

            kept_rows = {s: [r for r in scored[s] if r["key"] in retained[s]] for s in scored}
            tuned_rows = {s: [r for r in tuned[s]["rows"] if r["plddt"] is not None] for s in tuned}
            cells: dict[str, dict[str, Any]] = {}
            needed: set[str] = set()
            for stratum in ota.STRATA:
                key = f"{pair.tuned}|{stratum}"
                cell = {
                    "key": key,
                    "primary": contrast["bands"]["columns"].get(key),
                    "base_pool": aot.scored_by_stratum(scored, stratum),
                    "filtered_pool": aot.scored_by_stratum(scored, stratum, retained),
                    "tuned_pool": aot.plddt_by_stratum(tuned, stratum),
                }
                cell["short"] = {
                    side: ota.pool_shortfall(cell[f"{side}_pool"])
                    for side in ("base", "filtered", "tuned")
                }
                cells[stratum] = cell
                identified = (
                    ota.primary_gain_disposition(cell["primary"])["gain"] == ota.GAIN_IDENTIFIED
                )
                if identified and not any(cell["short"].values()):
                    needed.add(stratum)

            draws_base: dict[int, dict[str, Any]] = {}
            matched_draws: dict[int, dict[str, dict[str, np.ndarray]]] = {}
            for pos, stream in enumerate(sorted(scored)):
                order = ota._by_score(scored[stream], rule.retain)
                L_arr = np.asarray([scored[stream][i]["length"] for i in order], dtype=np.float64)
                P_arr = np.asarray([scored[stream][i]["plddt"] for i in order], dtype=np.float64)
                B_arr = np.asarray([codes[scored[stream][i]["stratum"]] for i in order])
                store = matched_draws[stream] = {
                    st: {mname: np.full(bd.DRAWS, np.nan) for mname in matchers}
                    for st in needed
                }

                def _readout(start, stop, blocks, top, *, L_arr=L_arr, P_arr=P_arr, B_arr=B_arr, store=store):
                    for st in needed:
                        if st not in blocks:
                            continue
                        c_code = codes[st]
                        for off in range(stop - start):
                            drawn = blocks[st][off]
                            kept = top[off][B_arr[top[off]] == c_code]
                            for mname, fn in matchers.items():
                                store[st][mname][start + off] = fn(
                                    L_arr[drawn], P_arr[drawn], L_arr[kept], P_arr[kept]
                                )[0]

                draws_base[stream] = ota.nested_base_draws(
                    scored[stream],
                    retain=rule.retain,
                    family=family,
                    column=index,
                    stream_index=pos,
                    readout=_readout if needed else None,
                )

            for stratum in ota.STRATA:
                cell = cells[stratum]
                key = cell["key"]
                column: dict[str, Any] = {
                    "key": key,
                    "primary": cell["primary"],
                    "tuned": cell["tuned_pool"],
                    "units": {
                        side: ota.pool_units(cell[f"{side}_pool"])
                        for side in ("base", "filtered", "tuned")
                    },
                    "retained": sum(len(v) for v in cell["filtered_pool"].values()),
                }
                if any(cell["short"].values()):
                    column["skip_reason"] = (
                        f"below the resampling floor on stream(s) {cell['short']}; "
                        + ota.ADMISSIBILITY_RULE
                    )
                else:
                    column.update(
                        base_realised=ota.pool_mean(cell["base_pool"]),
                        filtered_realised=ota.pool_mean(cell["filtered_pool"]),
                        tuned_realised=ota.pool_mean(cell["tuned_pool"]),
                    )
                columns["raw"].append(
                    {
                        **column,
                        **{
                            field: {s: draws_base[s][stratum][field] for s in draws_base}
                            for field in ("base", "filtered", "valid")
                        },
                    }
                )
                if stratum not in needed:
                    for mname in matchers:
                        columns[mname].append(column)
                    continue
                for mname, fn in matchers.items():
                    diff_m, kept_m = _stream_equal(fn, scored, kept_rows, stratum)
                    gain_m, gain_kept = _stream_equal(fn, scored, tuned_rows, stratum)
                    columns[mname].append(
                        {
                            **column,
                            "base": {s: draws_base[s][stratum]["base"] for s in draws_base},
                            "filtered": {
                                s: draws_base[s][stratum]["base"] + matched_draws[s][stratum][mname]
                                for s in draws_base
                            },
                            "valid": {
                                s: draws_base[s][stratum]["valid"]
                                & np.isfinite(matched_draws[s][stratum][mname])
                                for s in draws_base
                            },
                            "filtered_realised": column["base_realised"] + diff_m,
                            "matched_retained": kept_m,
                        }
                    )
                    points[(key, mname)] = {
                        "filter_minus_base_matched": diff_m,
                        "n_tuned_matched": gain_kept,
                        "tuned_minus_base_matched": gain_m,
                        "tuned_minus_base_raw": column["tuned_realised"] - column["base_realised"],
                    }

        bands_out = {
            est_name: ota.reproduced_fraction_bands(columns[est_name], family=family)
            for est_name in ("raw", "bin", "nn")
        }
        share_records[label] = {"family": family, "bands": bands_out}
        for mname in matchers:
            for key, cell in bands_out[mname]["columns"].items():
                if (key, mname) not in points:
                    continue
                pt = points[(key, mname)]
                pair_name, stratum = key.split("|")
                gain_rows.append(
                    {
                        "pairs": label,
                        "pair": pair_name,
                        "stratum": stratum,
                        "estimator": mname,
                        "tuned_minus_base_raw": pt["tuned_minus_base_raw"],
                        "tuned_minus_base_matched": pt["tuned_minus_base_matched"],
                        "n_tuned_matched": pt["n_tuned_matched"],
                        "filter_minus_base_matched": pt["filter_minus_base_matched"],
                        "share_filter_matched_over_raw_gain": cell.get("point"),
                        "share_both_sides_matched": (
                            None
                            if pt["tuned_minus_base_matched"] == 0
                            else float(pt["filter_minus_base_matched"] / pt["tuned_minus_base_matched"])
                        ),
                    }
                )
    return {"ties": tie_rows, "length_matched_shares": share_records, "length_matched_gain": gain_rows}


# --------------------------------------------------------------------------- #
# Section (f): Side-by-Side 18/68 Sampled vs. 68/68 Full Comparison Table
# --------------------------------------------------------------------------- #


def _build_18_vs_68_comparison(
    root: Path,
    primary_68_rows: list[dict[str, Any]],
    secondary_68_rows: list[dict[str, Any]],
    lc_2000: dict[str, Any],
) -> dict[str, Any]:
    """Compare frozen 18/68 sampled (12,640) vs. complete 68/68 full cohort (47,778) & 1,748 vs. 2,000 long_cap."""
    p18 = json.loads(
        (root / "results/extensions/objective_tuned_endpoints_20261112/objective_tuned_endpoints.json").read_text(encoding="utf-8")
    )
    rows_18 = (
        _compact_contrast_rows(p18["primary"]["bands"], p18["primary"]["per_pair"])
        + _compact_contrast_rows(p18["secondary_pair"]["bands"], p18["secondary_pair"]["per_pair"])
    )
    map_18 = {(r["pair"], r["stratum"]): r for r in rows_18}
    rows_68 = primary_68_rows + secondary_68_rows

    main_comp = []
    for r68 in rows_68:
        r18 = map_18.get((r68["pair"], r68["stratum"]))
        hw18 = None if r18 is None else r18["band_half_width"]
        hw68 = r68["band_half_width"]
        main_comp.append(
            {
                "pair": r68["pair"],
                "stratum": r68["stratum"],
                "n_base_18": None if r18 is None else r18["n_base"],
                "n_tuned_18": None if r18 is None else r18["n_tuned"],
                "thin_18": None if r18 is None else r18["thin_under_30"],
                "diff_18": None if r18 is None else r18["diff"],
                "hw_18": hw18,
                "finding_18": None if r18 is None else r18["finding"],
                "n_base_68": r68["n_base"],
                "n_tuned_68": r68["n_tuned"],
                "thin_68": r68["thin_under_30"],
                "diff_68": r68["diff"],
                "hw_68": hw68,
                "finding_68": r68["finding"],
                "diff_shift_68_minus_18": (
                    None if (r18 is None or r18["diff"] is None or r68["diff"] is None) else float(r68["diff"] - r18["diff"])
                ),
                "half_width_ratio_68_over_18": (
                    None if (hw18 is None or hw68 is None or hw18 == 0) else float(hw68 / hw18)
                ),
                "resolved_from_thin": bool((r18 is not None and r18["thin_under_30"]) and not r68["thin_under_30"]),
            }
        )

    # Compare frozen 1,748 vs full 2,000 long_cap capped contrasts
    lc1748 = pd.read_csv(root / "results/long_cap_termination_20261110/analysis/capped_contrasts.tsv", sep="\t")
    lc_comp = []
    for row in lc1748.itertuples(index=False):
        if row.contrast == "tuned capped - base capped":
            target_label = "tuned capped - base capped (full 2000)"
        elif row.contrast == "tuned capped - base capped, non-degenerate rows only":
            target_label = "tuned capped - base capped (non-degenerate)"
        else:
            target_label = str(row.contrast)
        match = next(
            (x for x in lc_2000["capped_768_contrasts"] if x["contrast"] == target_label and x["endpoint"] == row.endpoint),
            None,
        )
        lc_comp.append(
            {
                "contrast": str(row.contrast),
                "endpoint": str(row.endpoint),
                "n_base_1748": int(row.n_base),
                "n_tuned_1748": int(row.n_tuned),
                "diff_1748": float(row.diff),
                "ci_1748": [float(row.lo), float(row.hi)],
                "n_a_2000": None if match is None else match["n_a"],
                "n_b_2000": None if match is None else match["n_b"],
                "diff_2000": None if match is None else match["diff"],
                "ci_2000": None if match is None else match["ci"],
            }
        )

    return {"main_18of68_vs_68of68": main_comp, "long_cap_1748_vs_2000": lc_comp}


# --------------------------------------------------------------------------- #
# Top-Level Unified Runner
# --------------------------------------------------------------------------- #


def run_r6_full_unified(
    *,
    root: Path = REPO_ROOT,
    out_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Run complete 68/68 + 2,000/2,000 Result 6 evaluation and write canonical JSON + SHA256."""
    t0 = time.monotonic()
    out_root = Path(out_dir) if out_dir is not None else (root / UNIFIED_OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)
    cdir = Path(cache_dir) if cache_dir is not None else (out_root / "r6_objective_tuned_full_cache")

    main_tsv = cdir / "checkpoint_structures_68of68.tsv"
    long_cap_tsv = cdir / "long_cap_structures_2000of2000.tsv"

    # Gate 1: exact reproduction against frozen 18/68 & 1,748/2,000
    repro_gate = verify_r6_reproduction_gate(main_tsv, long_cap_tsv, root=root)
    print(f"[R6-full] Gate 1 passed: {repro_gate}", flush=True)

    # Merge Pfam recognition (12,640 + 35,138 = 47,778)
    _, flags_jsonl, pfam_meta = merge_pfam_recognition_68of68(root=root, cache_dir=cdir)
    print(f"[R6-full] Pfam 68/68 merged: {pfam_meta['n_merged_total']} rows", flush=True)

    # Build zero-departure 48,000-attempt / 47,788-non-empty 3-stream campaign table + 47,778 DataFrame
    table, df, selector_jsonl = _build_full_campaign_table(root, main_tsv, flags_jsonl, cdir)
    table_ptm = _swap_table_metric_to_ptm(table)
    counts = gc_mod.load_kmer_counts(root / "data/kmer_background/uniref50_high_order", aot.KMER_ORDERS)

    # (a) Primary & Secondary Novelty-Stratified Contrasts (pLDDT & pTM, 3-stream simultaneous + termination split)
    shift_primary = aot.identity_shift(table, ota.PRIMARY_PAIRS)
    secondary_id_recog = _secondary_identity_and_recognition_summary(table, ota.SECONDARY_PAIRS[0])
    primary_plddt = aot.stratified_contrast(table, ota.PRIMARY_PAIRS, ota.PRIMARY_FAMILY, shift=shift_primary)
    secondary_plddt = aot.stratified_contrast(table, ota.SECONDARY_PAIRS, ota.SECONDARY_FAMILY)
    primary_ptm = _stratified_ptm_contrast(table_ptm, ota.PRIMARY_PAIRS, ota.PRIMARY_FAMILY, shift=shift_primary)
    secondary_ptm = _stratified_ptm_contrast(table_ptm, ota.SECONDARY_PAIRS, ota.SECONDARY_FAMILY)
    term_split = _run_termination_split_identity_contrasts(df)

    primary_plddt_rows = _compact_contrast_rows(primary_plddt["bands"], primary_plddt["per_pair"])
    secondary_plddt_rows = _compact_contrast_rows(secondary_plddt["bands"], secondary_plddt["per_pair"])
    primary_ptm_rows = _compact_contrast_rows(primary_ptm["bands"], primary_ptm["per_pair"])
    secondary_ptm_rows = _compact_contrast_rows(secondary_ptm["bands"], secondary_ptm["per_pair"])
    print("[R6-full] (a) primary & secondary novelty-stratified contrasts done", flush=True)

    # (b) Identity shift & Pfam family recognition + ProGen2-small-mix7 7-family attribution
    recog_primary = aot.recognition(table, ota.PRIMARY_PAIRS, flags_jsonl)
    mix7_attr = _run_progen2_mix7_attribution(table, counts, secondary_plddt)
    print("[R6-full] (b) identity shift, 47,778-seq Pfam recognition & mix7 attribution done", flush=True)

    # (c) Explicit-statistics controls + Base-likelihood controls (k=2,4,8) + Repetitiveness + Containment + Selector
    own_dir = root / "results/extensions/objective_tuned_endpoints_20261112/selector/own_model_likelihood"
    base_arms = sorted({pair.base for pair in (*ota.PRIMARY_PAIRS, *ota.SECONDARY_PAIRS)})
    base_likelihood = ota.merge_likelihood(
        {arm: ota.read_own_likelihood(own_dir / f"{arm}__on__{arm}.jsonl") for arm in base_arms}
    )
    selection_curve: dict[str, Any] = {}
    for frac in (0.5, 0.25, 0.125):
        k_val = round(1.0 / frac)
        selection_curve[f"k{k_val}"] = {
            "primary_statistical": aot.explicit_statistics_control(
                table, ota.PRIMARY_PAIRS, counts, primary_plddt, ota.CONTROL_FAMILIES, None, frac
            ),
            "primary_base_likelihood": aot.explicit_statistics_control(
                table, ota.PRIMARY_PAIRS, counts, primary_plddt, ota.LIKELIHOOD_CONTROL_FAMILIES, base_likelihood, frac
            ),
            "secondary_statistical": aot.explicit_statistics_control(
                table, ota.SECONDARY_PAIRS, counts, secondary_plddt, ota.SECONDARY_CONTROL_FAMILIES, None, frac
            ),
            "secondary_base_likelihood": aot.explicit_statistics_control(
                table, ota.SECONDARY_PAIRS, counts, secondary_plddt, ota.SECONDARY_LIKELIHOOD_CONTROL_FAMILIES, base_likelihood, frac
            ),
        }
    selector_all_arms = aot.selector_competence(table, selector_jsonl)
    rep_and_cont = _run_likelihood_repetitiveness_and_containment(root, df)
    print("[R6-full] (c) filter controls (k=2,4,8), repetitiveness, containment & selector done", flush=True)

    # (d) Length & cap-degeneracy audit (68/68) + full 2,000/2,000 768-residue companion
    length_audit = _run_length_and_cap_audit_68of68(df)
    long_cap_2000 = _run_long_cap_companion_2000of2000(root, long_cap_tsv)
    print("[R6-full] (d) length/cap audit + full 2,000/2,000 long_cap done", flush=True)

    # (e) 5-mer ties & length-matched nested-bootstrap share audit
    five_mer_audit = _run_five_mer_ties_and_length_68of68(table, counts, primary_plddt, secondary_plddt)
    print("[R6-full] (e) 5-mer ties & length-matched share audit done", flush=True)

    # (f) Side-by-side 18/68 sampled vs 68/68 full comparison
    comp_18_vs_68 = _build_18_vs_68_comparison(root, primary_plddt_rows, secondary_plddt_rows, long_cap_2000)

    payload = _clean_json(
        {
            "schema": SCHEMA,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "campaign": ot.CAMPAIGN,
            "coverage_declaration": {
                "main_cohort": table["totals"],
                "departures": table["departures"],
                "pfam_recognition_68of68": pfam_meta,
                "long_cap_companion": {
                    "n_total": long_cap_2000["n_total"],
                    "n_folded_ok": long_cap_2000["n_folded_ok"],
                    "n_newly_folded_base_capped": long_cap_2000["n_newly_folded_base_capped"],
                },
                "inputs_sha256": {
                    "checkpoint_structures_68of68.tsv": sha256_file(main_tsv),
                    "long_cap_structures_2000of2000.tsv": sha256_file(long_cap_tsv),
                },
            },
            "reproduction_gate": repro_gate,
            "a_novelty_stratified_contrasts": {
                "primary_plddt_3stream_rows": primary_plddt_rows,
                "secondary_plddt_3stream_rows": secondary_plddt_rows,
                "primary_ptm_3stream_rows": primary_ptm_rows,
                "secondary_ptm_3stream_rows": secondary_ptm_rows,
                "primary_plddt_full": primary_plddt,
                "secondary_plddt_full": secondary_plddt,
                "primary_ptm_full": primary_ptm,
                "secondary_ptm_full": secondary_ptm,
                "termination_split_identity_contrasts": term_split,
            },
            "b_identity_shift_and_pfam_recognition": {
                "identity_shift_primary_3stream": shift_primary,
                "pfam_recognition_primary_3stream": recog_primary,
                "secondary_pair_identity_and_recognition_descriptive": secondary_id_recog,
                "progen2_mix7_seven_family_attribution": mix7_attr,
            },
            "c_filter_controls_and_selector_competence": {
                "selection_curve_k2_k4_k8": selection_curve,
                "selector_competence_all_arms": selector_all_arms,
                **rep_and_cont,
            },
            "d_length_and_cap_confounding_audit": {
                "main_68of68_length_and_cap": length_audit,
                "long_cap_768_companion_2000of2000": long_cap_2000,
            },
            "e_five_mer_ties_and_length_matched_shares": five_mer_audit,
            "f_sampled_18of68_vs_full_68of68_comparison": comp_18_vs_68,
            "elapsed_seconds": round(time.monotonic() - t0, 2),
        }
    )

    out_path = out_root / "r6_objective_tuned_full_esmfold2_unified.json"
    atomic_write_json(out_path, payload)
    digest = sha256_file(out_path)
    (out_root / "r6_objective_tuned_full_esmfold2_unified.json.sha256").write_text(
        f"{digest}  {out_path.name}\n", encoding="utf-8"
    )
    print(f"[R6-full] wrote {out_path} (sha256={digest}, elapsed={payload['elapsed_seconds']}s)", flush=True)
    return payload
