"""Unified 62-model R3 contact discrimination & bidirectional ablation yardstick.

Discovers all canonical ``position_likelihood_v1`` E01 contact extraction
archives across legacy (``results/external_baseline/20261008*``,
``20261010*``) and newly extracted general-PLM / ESM3 directories
(``results/extensions/unified_yardstick_20261124/e01_contact_extractions/``),
maps every extraction deterministically to its canonical ``arm_id`` in
``UNIFIED_62_SINGLE_SEQUENCE_ARMS``, and runs the exact 0-based
``measure_contact_discrimination`` and ``fit_contact_bidirectional``
estimators across three alignment-depth strata:

* ``all``: all admitted references ($N = 28$, 41,086 pairs, 2,227 contacts)
* ``deep``: $N_{\\mathrm{eff}}/L \\ge 5$ ($N = 13$ admitted references)
* ``shallow``: $N_{\\mathrm{eff}}/L < 5$ ($N = 15$ admitted references)

Every stratum reports both:
* **Full-alphabet complete-case panel** (``pair_coverage >= 0.50``, preserving
  all 28 / 13 / 15 admitted references and avoiding thin-BPE band inflation).
* **All-admitted-arms panel** (every arm carrying at least
  ``MINIMUM_BOOTSTRAP_UNITS = 8`` references, including thin BPE text/joint
  tokenizers).

All 62/62 single-sequence arms are explicitly partitioned into disjoint
accounting buckets (``admitted_full_alphabet``, ``admitted_thin_bpe``,
``refused_below_8_refs``, ``refused_moe_prefix_layout``, ``not_yet_extracted``).

Finally, ``contacts_unified_62.json`` consolidates per-arm tables, paradigm
summaries, scaling ladders, and exact two-sample paired-family bootstrap
Deep-vs-Shallow MSA contrasts (``shallow_minus_deep`` and
``deep_minus_shallow``) for ``auc_minus_half``, forward
``+M | (geom + prof + couplings)``, and reverse
``+couplings | (geom + prof + M)``.
"""

from __future__ import annotations

import collections
import concurrent.futures
import gzip
import hashlib
import json
import multiprocessing
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from src.capability.core.io import sha256_file
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS
from src.capability.extensions import bidirectional as bidir
from src.capability.extensions import phenotype_breadth as breadth
from src.capability.extensions.responses import RetainedResponses
from src.capability.position.contact_response import (
    CONTACT_ANGSTROM,
    MIN_SEQUENCE_SEPARATION,
    rebuild_states,
    receiver_census,
    structural_rows,
)
from src.capability.position.position_likelihood import TIER2_NATS, read_archive
from src.capability.readouts.readout_analysis import ALPHAS
from src.capability.unified.roster import (
    ARM_METADATA,
    PARADIGM_ORDER,
    REPO_ROOT,
    UNIFIED_62_SINGLE_SEQUENCE_ARMS,
    UNIFIED_OUTPUT_DIR,
    dtype_rank,
    get_arm_metadata,
    sort_arms,
)

# Reuse exact 0-based estimators and constants from scripts/capability/position/.
_POSITION_SCRIPTS_DIR = REPO_ROOT / "scripts/capability/position"
if str(_POSITION_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_POSITION_SCRIPTS_DIR))

import fit_contact_bidirectional as _fcb  # noqa: E402
import measure_contact_discrimination as _mcd  # noqa: E402

#: Default frozen structural, cohort, and Potts-fit inputs (identical to 2026-11-20 refit).
DEFAULT_STRUCTURES_PATH = Path(
    "results/extensions/mutation_structure_20261006/structure/annotation/sites.json.gz"
)
DEFAULT_COVERAGE_PATH = Path(
    "results/extensions/mutation_structure_20261006/structure/annotation/coverage.json"
)
DEFAULT_COHORT_PATH = Path("results/R5/context_mutation_rescue_20260923/cohort.json")
DEFAULT_PARAMS_ROOTS: tuple[Path, ...] = (
    Path("archive/logs/R3/pairwise_deepen_20260924/baseline/work"),
    Path("archive/logs/R3/pairwise_homologs_20260924/baseline/work"),
)
DEFAULT_LEGACY_EXTRACTION_ROOTS: tuple[Path, ...] = (
    Path("results/external_baseline/20261008021550_7aab55b6c322"),
    Path("results/external_baseline/20261010114303_6ca55a7efd49"),
    Path("results/external_baseline/20261010121000_74fb06ef3717"),
    Path("results/external_baseline/20261010140713_5eb424817b6a"),
)
DEFAULT_NEW_EXTRACTION_ROOT = UNIFIED_OUTPUT_DIR / "e01_contact_extractions"

#: Historical declared / measured E01 layout refusals inside UNIFIED_62_SINGLE_SEQUENCE_ARMS.
DECLARED_E01_REFUSALS: Mapping[str, dict[str, Any]] = {
    "progen3-3b": {
        "arm": "progen3-3b",
        "status": "refused_moe_layout",
        "source": "declared_e01_refusal",
        "reason": (
            "declared E01 refusal on a measured wild-type layout discrepancy of "
            "0.53165 nats mean absolute error"
        ),
    },
    "protgpt3-1.3b": {
        "arm": "protgpt3-1.3b",
        "status": "refused_moe_layout",
        "source": "measured_e01_prefix_refusal",
        "reason": (
            "measured E01 prefix-invariant refusal: bit-reproducible forward "
            "(repeat_max=0.0) with upstream prefix residual on 155/162 assays "
            "(worst 2.0981e-5 nats)"
        ),
    },
}

#: Three canonical alignment-depth strata.
DEPTH_STRATA: tuple[dict[str, Any], ...] = (
    {"key": "all", "label": "All admitted references", "min_neff": 0.0, "max_neff": None, "suffix": ""},
    {"key": "deep", "label": "Deep MSA (Neff/L >= 5)", "min_neff": 5.0, "max_neff": None, "suffix": "_deep"},
    {"key": "shallow", "label": "Shallow MSA (Neff/L < 5)", "min_neff": 0.0, "max_neff": 5.0, "suffix": "_shallow"},
)

#: Scaling ladders grouped by family for consolidated reporting.
SCALING_LADDERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ESM2", ("esm2-8m", "esm2-35m", "esm2-150m", "esm2-650m", "esm2-3b", "esm2-15b")),
    ("ESMC", ("esmc-300m", "esmc-600m", "esmc-6b")),
    ("DPLM", ("dplm-150m", "dplm-650m", "dplm-3b")),
    ("ProteinGLM-MLM", ("proteinglm-1b-mlm", "proteinglm-3b-mlm", "proteinglm-10b-mlm")),
    ("SaProt", ("saprot-35m", "saprot-650m", "saprot-1.3b")),
    ("AMPLIFY", ("amplify-120m", "amplify-350m")),
    ("Prime", ("prime-690m",)),
    ("ESM3", ("esm3-1.4b",)),
    ("Dayhoff", ("dayhoff-170m",)),
    ("ProGen2", ("progen2-small", "progen2-base", "progen2-medium", "progen2-large", "progen2-xlarge")),
    ("ProGen3", ("progen3-112m", "progen3-219m", "progen3-339m", "progen3-762m", "progen3-1b", "progen3-3b")),
    ("ProtGPT3", ("protgpt3-112m", "protgpt3-1.3b", "protgpt3-10b")),
    ("ProtGPT2", ("protgpt2",)),
    ("RITA", ("rita-xl",)),
    ("ProteinGLM-CLM", ("proteinglm-7b-clm",)),
    ("ProLLaMA", ("prollama-stage-1", "prollama")),
    ("Galactica", ("galactica-125m", "galactica-1.3b", "galactica-6.7b", "galactica-30b")),
    ("InstructProtein", ("instructprotein",)),
    ("ByGPT5", ("bygpt5-small-en", "bygpt5-base-en", "bygpt5-medium-en")),
    ("GPT2", ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")),
    ("DialoGPT", ("dialogpt-small",)),
    ("Llama", ("llama-3.2-3b", "llama-2-7b")),
    ("Qwen", ("qwen2.5-0.5b", "qwen2.5-0.5b-instruct", "qwen2.5-7b", "qwen2.5-32b", "qwen3-8b-base")),
)

FULL_ALPHABET_MIN_PAIR_FRACTION = float(_mcd.FULL_ALPHABET_MIN_PAIR_FRACTION)
RUNGS = _fcb.RUNGS
STRATA = _fcb.STRATA
PARADIGM_SIDES = _mcd.PARADIGM_SIDES

_SLUG_TO_ARM_ID: dict[str, str] = {
    re.sub(r"[^a-z0-9]+", "_", arm_id.lower()).strip("_"): arm_id
    for arm_id in UNIFIED_62_SINGLE_SEQUENCE_ARMS
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json_with_sha256(path: Path, payload: Any) -> str:
    """Write formatted JSON atomically and emit a ``sha256sum -c`` compatible ``.sha256`` sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=bidir.json_default)
            handle.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    digest = sha256_file(path)
    sha_path = path.with_name(f"{path.name}.sha256")
    sha_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def canonicalize_arm_id(raw_name: str) -> str:
    """Deterministically normalize a directory name or raw identifier to a canonical 62-arm ID."""
    candidate = raw_name.strip()
    if candidate in ARM_METADATA and ARM_METADATA[candidate].is_single_sequence:
        return candidate
    stripped = re.sub(r"^posresp[a-z0-9]*_extract_", "", candidate)
    stripped = re.sub(r"_gpu\d+$", "", stripped)
    if stripped in ARM_METADATA and ARM_METADATA[stripped].is_single_sequence:
        return stripped
    slug = re.sub(r"[^a-z0-9]+", "_", stripped.lower()).strip("_")
    if slug in _SLUG_TO_ARM_ID:
        return _SLUG_TO_ARM_ID[slug]
    raise ValueError(
        f"Cannot map E01 extraction identifier {raw_name!r} to UNIFIED_62_SINGLE_SEQUENCE_ARMS"
    )


@dataclass(frozen=True)
class DiscoveredExtraction:
    """One validated ``position_likelihood_v1`` extraction directory bound to a canonical arm_id."""

    arm_id: str
    directory: Path
    paradigm: str
    dtype_realised: str | None
    upstream_tolerance_nats: float
    prefix_tier: int | None
    sub_tier2_caveat: str | None
    index_sha256: str
    files: Mapping[str, Mapping[str, Any]]
    source_dir_name: str


def _parse_upstream_tolerance(
    index: Mapping[str, Any], paradigm: str
) -> tuple[float, int | None, str | None, str | None]:
    """Validate prefix-invariant receipt; return ``(tolerance, tier, refusal_reason, caveat)``.

    Any causal arm with ``upstream_max_abs_nats > TIER2_NATS`` (1e-4 nats) is refused.
    Causal MoE/hybrid arms with bit-identical repeats (``repeat_max == 0``) whose
    upstream shift stays strictly within ``TIER2_NATS`` (e.g. ``<= 1.62e-5`` nats)
    are admitted with an explicit ``sub_tier2_caveat`` rather than silently dropped.
    """
    inv = index.get("prefix_invariant")
    if not isinstance(inv, Mapping):
        return 0.0, None, None, None
    tier_raw = inv.get("tier")
    tier = int(tier_raw) if tier_raw is not None else None
    if paradigm == "causal_next_token":
        worst = float(inv.get("upstream_max_abs_nats", 0.0))
        admitted = float(inv.get("admitted_tolerance_nats", 0.0))
        repeat_max = float(inv.get("repeat_max_abs_nats", 0.0))
        if not np.isfinite(worst) or worst < 0.0 or worst > TIER2_NATS + 1e-12:
            return (
                0.0,
                tier,
                (
                    f"causal upstream_max_abs_nats={worst} exceeds TIER2_NATS={TIER2_NATS:.1e} "
                    f"(tier={tier})"
                ),
                None,
            )
        if tier == 2:
            tol = min(TIER2_NATS, max(admitted, worst))
            return tol, tier, None, None
        if tier is not None and tier >= 3:
            caveat = (
                f"sub-TIER2 MoE/hybrid causal prefix shift: upstream_max_abs_nats={worst:.3e} "
                f"<= TIER2_NATS={TIER2_NATS:.1e} with repeat_max_abs_nats={repeat_max:.3e} "
                f"(strict repeat-multiple tier={tier})"
            )
            return min(TIER2_NATS, worst + 1e-12), tier, None, caveat
    return 0.0, tier, None, None


def load_refused_moe_records(
    search_roots: Sequence[Path],
) -> dict[str, dict[str, Any]]:
    """Load declared historical refusals plus any ``refused_moe_layout.json`` receipts."""
    refused: dict[str, dict[str, Any]] = {
        arm_id: dict(record) for arm_id, record in DECLARED_E01_REFUSALS.items()
    }
    for root in search_roots:
        rec_path = root / "refused_moe_layout.json"
        if not rec_path.is_file():
            continue
        doc = json.loads(rec_path.read_text(encoding="utf-8"))
        for item in doc.get("refused_arms", []):
            arm_id = canonicalize_arm_id(str(item["arm"]))
            refused[arm_id] = {
                **dict(item),
                "arm": arm_id,
                "status": str(item.get("status", "refused_moe_layout")),
                "source": str(rec_path),
            }
    return refused


def discover_e01_extractions(
    *,
    root: Path | None = None,
    extraction_roots: Sequence[Path] | None = None,
) -> tuple[dict[str, DiscoveredExtraction], dict[str, dict[str, Any]]]:
    """Discover and deduplicate canonical ``position_likelihood_v1`` extraction directories.

    Returns ``(extractions_by_arm_id, refused_moe_by_arm_id)``.
    Deduplication follows the unified roster precedence rule: ``float32`` beats
    ``bfloat16``; ties prefer the later search root (newest extraction).
    Directories starting with ``_`` (e.g. ``_verify_esm2_650m``) are skipped.
    """
    base = REPO_ROOT if root is None else Path(root)
    if extraction_roots is None:
        roots = [base / rel for rel in (*DEFAULT_LEGACY_EXTRACTION_ROOTS, DEFAULT_NEW_EXTRACTION_ROOT)]
    else:
        roots = [Path(p) for p in extraction_roots]

    refused_moe = load_refused_moe_records(roots)
    discovered: dict[str, DiscoveredExtraction] = {}

    for scan_root in roots:
        if not scan_root.is_dir():
            continue
        for child in sorted(scan_root.iterdir()):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            index_path = child / "position_likelihood.json"
            if not index_path.is_file():
                continue
            index = json.loads(index_path.read_text(encoding="utf-8"))
            status = index.get("status")
            if status is not None and status != "complete":
                continue
            identity = index.get("identity") or {}
            raw_arm = str(identity.get("arm") or child.name)
            arm_id = canonicalize_arm_id(raw_arm)
            paradigm = str(identity["paradigm"])
            if paradigm not in PARADIGM_SIDES:
                raise ValueError(f"{child}: unknown paradigm {paradigm!r}")

            tol, tier, refusal_reason, caveat = _parse_upstream_tolerance(index, paradigm)
            if refusal_reason is not None:
                refused_moe[arm_id] = {
                    "arm": arm_id,
                    "status": "refused_moe_layout",
                    "source": str(index_path),
                    "reason": refusal_reason,
                }
                continue

            dtype_val = identity.get("dtype") or (identity.get("provenance") or {}).get("dtype")
            dtype_str = None if dtype_val is None else str(dtype_val)
            cand = DiscoveredExtraction(
                arm_id=arm_id,
                directory=child,
                paradigm=paradigm,
                dtype_realised=dtype_str,
                upstream_tolerance_nats=tol,
                prefix_tier=tier,
                sub_tier2_caveat=caveat,
                index_sha256=sha256_file(index_path),
                files={str(entry["assay"]): entry for entry in index.get("assays", [])},
                source_dir_name=child.name,
            )
            prev = discovered.get(arm_id)
            if prev is None or dtype_rank(cand.dtype_realised) >= dtype_rank(prev.dtype_realised):
                discovered[arm_id] = cand
                if arm_id in refused_moe and arm_id != "progen3-3b":
                    refused_moe.pop(arm_id, None)

    return discovered, refused_moe


def arm_pair_responses_validated(
    archive: Path,
    geometry: Mapping[int, Any],
    pairs: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    assay: str,
    paradigm: str,
    upstream_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Exact ``measure_contact_discrimination.arm_pair_responses`` with tier-2 tolerance support."""
    sides = PARADIGM_SIDES[paradigm]
    payload = read_archive(archive)
    states = rebuild_states(payload, [row["wildtype"], *row["sequences"]])
    identity = {
        "wildtype": row["wildtype"],
        "mutants": row["mutants"],
        "sequences": row["sequences"],
    }
    with np.load(archive, allow_pickle=False) as data:
        retained = RetainedResponses(data, states, identity)
        frozen = {index: retained.response(index) for index in retained.selected_indices}
    collected: dict[tuple[int, int], list[float]] = {}
    for index, response in frozen.items():
        census = receiver_census(
            payload,
            index,
            site=int(response["i"]),
            paradigm=paradigm,
            upstream_tolerance=upstream_tolerance,
        )
        for entry in structural_rows(
            census,
            geometry,
            pairs,
            assay=assay,
            family=row["cluster"],
            mutation=row["mutants"][index],
        ):
            if entry["direction"] not in sides:
                continue
            key = (min(int(entry["i"]), int(entry["j"])), max(int(entry["i"]), int(entry["j"])))
            collected.setdefault(key, []).append(float(entry["absolute_response"]))
    return {
        "values": {key: float(np.mean(values)) for key, values in collected.items()},
        "admitted_variants": len(frozen),
    }


def _extract_one_arm_all_refs(
    ext: DiscoveredExtraction,
    admitted_refs: Mapping[str, Mapping[str, Any]],
    blocks_all: Mapping[str, Mapping[str, Any]],
    cohort_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, dict[str, Any]], dict[str, dict[str, int]], dict[str, float | None]]:
    """Extract per-reference pair arrays, coverage counts, and AUC-0.5 for one arm across all refs."""
    arm_models: dict[str, dict[str, Any]] = {}
    arm_cov: dict[str, dict[str, int]] = {}
    arm_auc: dict[str, float | None] = {}

    for assay, record in sorted(admitted_refs.items()):
        entry = ext.files.get(assay)
        archive = (
            None if entry is None else ext.directory / "archives" / Path(str(entry["file"])).name
        )
        if archive is None or not archive.exists():
            arm_auc[assay] = None
            continue
        measured = arm_pair_responses_validated(
            archive,
            record["geometry"],
            record["pairs"],
            cohort_rows[assay],
            assay=assay,
            paradigm=ext.paradigm,
            upstream_tolerance=ext.upstream_tolerance_nats,
        )
        keys = blocks_all[assay]["keys"]
        label_bool = blocks_all[assay]["label"] > 0.5
        seen = [pos for pos, key in enumerate(keys) if key in measured["values"]]
        arm_cov[assay] = {
            "pairs": len(seen),
            "admitted_variants": int(measured["admitted_variants"]),
        }
        if len(seen) < 2:
            arm_auc[assay] = None
            continue
        scores = np.asarray([measured["values"][keys[pos]] for pos in seen], dtype=np.float64)
        auc_val = _mcd.auc_minus_half(scores, label_bool[seen])
        arm_auc[assay] = auc_val
        if auc_val is not None:
            arm_models[assay] = {
                "rows": np.asarray(seen, dtype=np.int64),
                "column": scores[:, None],
            }
    return ext.arm_id, arm_models, arm_cov, arm_auc


def _support_signature(pieces: Sequence[tuple[str, np.ndarray]]) -> str:
    """Content digest identifying the exact ``(assay, rows)`` pair support of an arm."""
    h = hashlib.sha256()
    for assay, rows in pieces:
        h.update(assay.encode("utf-8"))
        h.update(b":")
        h.update(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
        h.update(b";")
    return h.hexdigest()


_ALPHAS_ARR = np.asarray(ALPHAS, dtype=np.float64)


def _fast_auc_minus_half(score: np.ndarray, label_bool: np.ndarray) -> float:
    """Vectorized mid-rank AUC - 0.5 matching ``measure_contact_discrimination.auc_minus_half``."""
    positives = int(label_bool.sum())
    negatives = int(len(label_bool) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("group carries one contact class only; AUC is undefined")
    ranks = rankdata(score, method="average")
    return float(ranks[label_bool].sum() - positives * (positives + 1) / 2.0) / (
        positives * negatives
    ) - 0.5


def _ridge_predict_np(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_test: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """Exact float64 weighted Ridge matching ``readout_analysis.ridge_predict``."""
    w = weights / weights.sum()
    mean = (w[:, None] * x_train).sum(axis=0)
    xc = x_train - mean
    scale = np.sqrt((w[:, None] * (xc * xc)).sum(axis=0))
    scale = np.where(scale < 1e-12, 1.0, scale)
    z = xc / scale
    ym = float((w * y_train).sum())
    covariance = z.T @ (w[:, None] * z)
    rhs = z.T @ (w * (y_train - ym))
    values, vectors = np.linalg.eigh(covariance)
    beta = vectors @ ((vectors.T @ rhs)[:, None] / (np.maximum(values, 0.0)[:, None] + alphas))
    return ((x_test - mean) / scale) @ beta + ym


def _fast_row_weights(assays: np.ndarray, families: np.ndarray) -> np.ndarray:
    """Vectorized exact equivalent of ``readout_analysis.row_weights``."""
    uniq_a, inv_a, counts_a = np.unique(assays, return_inverse=True, return_counts=True)
    # Map each unique assay to its family (one-to-one on this cohort; verify consistency).
    first_idx = np.empty(len(uniq_a), dtype=np.int64)
    first_idx[inv_a] = np.arange(len(assays), dtype=np.int64)
    assay_fams = families[first_idx]
    if not np.all(assay_fams[inv_a] == families):
        raise ValueError("assay assigned to multiple families")
    uniq_f, inv_f, fam_assay_counts = np.unique(assay_fams, return_inverse=True, return_counts=True)
    n_fams = float(len(uniq_f))
    per_assay_w = 1.0 / (n_fams * fam_assay_counts[inv_f].astype(np.float64) * counts_a.astype(np.float64))
    return per_assay_w[inv_a]


def _build_fold_plan(state: Mapping[str, Any]) -> dict[str, Any]:
    """Precompute outer/inner index arrays, fold weights, and group masks once per support."""
    groups = state["groups"]
    backgrounds = state["backgrounds"]
    label = state["label"]
    weights = state["weights"]
    order = state["order"]

    group_masks: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float]] = []
    for group in order:
        idx = np.flatnonzero(groups == group)
        w_g = weights[idx]
        group_masks.append((group, idx, label[idx] > 0.5, w_g, float(w_g.sum())))

    seed_plans: list[list[tuple[np.ndarray, np.ndarray, np.ndarray, int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]]]] = []
    for seed in breadth.SPLIT_SEEDS:
        outer_list = []
        for _fold, held, training, inner in state["signatures"][seed]:
            train = np.flatnonzero(np.isin(groups, training))
            test = np.flatnonzero(np.isin(groups, held))
            w_train = _fast_row_weights(backgrounds[train], groups[train])
            inner_list = []
            for validation in inner:
                valid = np.flatnonzero(np.isin(groups, validation))
                fit = train[~np.isin(groups[train], validation)]
                w_fit = _fast_row_weights(backgrounds[fit], groups[fit])
                w_valid = _fast_row_weights(backgrounds[valid], groups[valid])
                inner_list.append((fit, valid, w_fit, w_valid, len(validation)))
            outer_list.append((train, test, w_train, len(training), inner_list))
        seed_plans.append(outer_list)

    return {
        "label": label,
        "group_masks": group_masks,
        "seed_plans": seed_plans,
    }


def _fit_design_seed_avg_fast(
    plan: Mapping[str, Any], matrix: np.ndarray
) -> dict[str, dict[str, float]]:
    """Exact seed-averaged per-group statistics using precomputed fold masks and weights."""
    x = np.asarray(matrix, dtype=np.float64)
    label = plan["label"]
    n_rows = len(label)
    group_masks = plan["group_masks"]
    n_seeds = float(len(plan["seed_plans"]))

    rank_sum: dict[str, float] = {g: 0.0 for g, *_ in group_masks}
    sq_sum: dict[str, float] = {g: 0.0 for g, *_ in group_masks}
    sqc_sum: dict[str, float] = {g: 0.0 for g, *_ in group_masks}

    for outer_list in plan["seed_plans"]:
        prediction = np.empty(n_rows, dtype=np.float64)
        for train, test, w_train, n_training, inner_list in outer_list:
            losses = np.zeros(len(_ALPHAS_ARR), dtype=np.float64)
            for fit, valid, w_fit, w_valid, n_val in inner_list:
                pred_v = _ridge_predict_np(x[fit], label[fit], w_fit, x[valid], _ALPHAS_ARR)
                diff = pred_v - label[valid, None]
                losses += n_val * (w_valid[:, None] * (diff * diff)).sum(axis=0)
            losses /= n_training
            best = len(losses) - 1 - int(np.argmin(losses[::-1]))
            alpha_arr = _ALPHAS_ARR[best : best + 1]
            prediction[test] = _ridge_predict_np(
                x[train], label[train], w_train, x[test], alpha_arr
            )[:, 0]

        clipped = np.clip(prediction, 0.0, 1.0)
        for group, idx, lbl_bool, w_g, denom in group_masks:
            p_g = prediction[idx]
            y_g = label[idx]
            rank_sum[group] += _fast_auc_minus_half(p_g, lbl_bool) + 0.5
            d_raw = p_g - y_g
            d_clp = clipped[idx] - y_g
            sq_sum[group] += float((w_g * (d_raw * d_raw)).sum() / denom)
            sqc_sum[group] += float((w_g * (d_clp * d_clp)).sum() / denom)

    return {
        "contact_ranking": {g: rank_sum[g] / n_seeds for g, *_ in group_masks},
        "contact_squared_error": {g: sq_sum[g] / n_seeds for g, *_ in group_masks},
        "contact_squared_error_clipped": {g: sqc_sum[g] / n_seeds for g, *_ in group_masks},
    }


def _fit_arm_worker(
    arm_id: str,
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    base_designs: Mapping[str, np.ndarray],
    need_uc: bool,
) -> tuple[str, dict[tuple[str, str | None], dict[str, dict[str, float]]]]:
    """Worker computing ``M``, ``U_C_M`` (all 3 rungs), and optionally ``U_C`` for one arm."""
    out: dict[tuple[str, str | None], dict[str, dict[str, float]]] = {}
    out[("M", None)] = _fit_design_seed_avg_fast(plan, state["column"])
    for rung in RUNGS:
        base_x = base_designs[rung]
        if need_uc:
            out[("U_C", rung)] = _fit_design_seed_avg_fast(plan, base_x)
        out[("U_C_M", rung)] = _fit_design_seed_avg_fast(
            plan, np.column_stack([base_x, state["column"]])
        )
    return arm_id, out


def evaluate_bidirectional_panel_fast(
    panel_arms: Sequence[Mapping[str, Any]],
    blocks: Mapping[str, Mapping[str, Any]],
    model: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    support: Mapping[str, Mapping[str, Any]],
    *,
    min_units: int = MINIMUM_BOOTSTRAP_UNITS,
    workers: int = 1,
) -> dict[str, Any]:
    """Exact ``fit_contact_bidirectional.evaluate_panel`` with bit-identical ``U_C`` memoization.

    Because every arm sharing the same pair support (e.g. all full-alphabet
    bidirectional models, or all full-alphabet causal models with the same
    tokenizer) has identical ``label``, ``groups``, ``backgrounds``, ``control``,
    ``weights`` and ``signatures``, ``U_C`` depends strictly on the support
    signature and rung. Memoizing ``U_C`` avoids ~40% redundant ridge solves
    while producing bit-identical outputs, and returning per-group increment
    tables enables exact downstream Deep-vs-Shallow two-sample bootstrap
    contrasts without refitting.
    """
    complete = sorted(set.intersection(*[set(arm["references"]) for arm in panel_arms]))
    if len(complete) < min_units:
        raise ValueError(
            f"{len(complete)} references carry every admitted arm, below the "
            f"{min_units}-unit floor; no increment is fitted"
        )

    states: dict[str, dict[str, Any]] = {}
    sig_by_arm: dict[str, str] = {}
    first_arm_for_sig: dict[str, str] = {}

    for arm in panel_arms:
        arm_id = str(arm["arm"])
        pieces = [(assay, model[arm_id][assay]["rows"]) for assay in complete if assay in model[arm_id]]
        label = np.concatenate([blocks[assay]["label"][rows] for assay, rows in pieces])
        groups = np.concatenate(
            [np.full(len(rows), support[assay]["family"]) for assay, rows in pieces]
        ).astype(str)
        backgrounds = np.concatenate([np.full(len(rows), assay) for assay, rows in pieces])
        control = {
            name: np.concatenate([blocks[assay][name][rows] for assay, rows in pieces])
            for name in ("geom", "prof", "couplings")
        }
        column = np.concatenate([model[arm_id][assay]["column"] for assay, _ in pieces])
        order = sorted(set(groups.tolist()))
        if len(order) < min_units:
            raise ValueError(
                f"{arm_id}: {len(order)} families on its own pair support, below the "
                f"{min_units}-unit floor"
            )
        state = {
            "references": [assay for assay, _ in pieces],
            "label": label,
            "groups": groups,
            "backgrounds": backgrounds,
            "control": control,
            "column": column,
            "weights": _fast_row_weights(backgrounds, groups),
            "signatures": {
                seed: breadth.outer_signature(groups, seed) for seed in breadth.SPLIT_SEEDS
            },
            "order": order,
            "pairs": int(len(label)),
            "contacts": int(label.sum()),
        }
        states[arm_id] = state
        sig = _support_signature(pieces)
        sig_by_arm[arm_id] = sig
        first_arm_for_sig.setdefault(sig, arm_id)

    columns = [
        {"rung": rung, "arm": str(arm["arm"]), "direction": direction}
        for rung in RUNGS
        for arm in panel_arms
        for direction in bidir.DIRECTIONS
    ]
    declared = 2 * len(RUNGS) * len(panel_arms)
    if len(columns) != declared:
        raise ValueError("the realised family is not the declared product of directions, rungs and arms")

    cache: dict[tuple[str, str | None, str], dict[str, dict[str, float]]] = {}
    uc_by_sig: dict[tuple[str, str], dict[str, dict[str, float]]] = {}

    plan_by_sig: dict[str, dict[str, Any]] = {}
    base_designs_by_sig: dict[str, dict[str, np.ndarray]] = {}
    for sig, rep_arm in first_arm_for_sig.items():
        rep_state = states[rep_arm]
        plan_by_sig[sig] = _build_fold_plan(rep_state)
        base_designs_by_sig[sig] = {
            rung: breadth.design(rep_state["control"], names) for rung, names in RUNGS.items()
        }

    max_workers = max(1, min(int(workers), 4, len(panel_arms)))
    tasks = [
        (
            str(arm["arm"]),
            states[str(arm["arm"])],
            plan_by_sig[sig_by_arm[str(arm["arm"])]],
            base_designs_by_sig[sig_by_arm[str(arm["arm"])]],
            first_arm_for_sig[sig_by_arm[str(arm["arm"])]] == str(arm["arm"]),
        )
        for arm in panel_arms
    ]

    if max_workers == 1:
        results = [_fit_arm_worker(*t) for t in tasks]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_fit_arm_worker, *t) for t in tasks]
            results = [fut.result() for fut in futures]

    for arm_id, arm_res in results:
        sig = sig_by_arm[arm_id]
        cache[("M", None, arm_id)] = arm_res[("M", None)]
        for rung in RUNGS:
            cache[("U_C_M", rung, arm_id)] = arm_res[("U_C_M", rung)]
            if ("U_C", rung) in arm_res:
                uc_by_sig[(sig, rung)] = arm_res[("U_C", rung)]

    for arm in panel_arms:
        arm_id = str(arm["arm"])
        sig = sig_by_arm[arm_id]
        for rung in RUNGS:
            cache[("U_C", rung, arm_id)] = uc_by_sig[(sig, rung)]

    shared = sorted(set.intersection(*[set(state["order"]) for state in states.values()]))
    if len(shared) < min_units:
        raise ValueError(
            f"{len(shared)} families are carried by every admitted arm, below the "
            f"{min_units}-unit floor; no family is banded"
        )

    payload_strata: dict[str, Any] = {}
    per_group_increments: dict[str, dict[str, dict[str, list[float]]]] = {}

    for stratum, declaration in STRATA.items():
        sign = float(declaration["to_loss"])
        matrix_cols: list[list[float]] = []
        for column in columns:
            reduced = (
                ("U_C", column["rung"], column["arm"])
                if column["direction"] == "model_given_controls"
                else ("M", None, column["arm"])
            )
            joint = ("U_C_M", column["rung"], column["arm"])
            matrix_cols.append(
                [
                    sign * cache[reduced][stratum][group] - sign * cache[joint][stratum][group]
                    for group in shared
                ]
            )
        matrix = np.asarray(matrix_cols, dtype=np.float64).T
        bands = _mcd.paired_bands_with_marginal(matrix)
        verdicts = bidir.verdict(bands["interval"])
        marginal_verdicts = bidir.verdict(bands["marginal_interval"])

        control_performance = {
            rung: {
                str(arm["arm"]): float(
                    np.mean([cache[("U_C", rung, str(arm["arm"]))][stratum][g] for g in shared])
                )
                for arm in panel_arms
            }
            for rung in RUNGS
        }
        joint_performance = {
            rung: {
                str(arm["arm"]): float(
                    np.mean([cache[("U_C_M", rung, str(arm["arm"]))][stratum][g] for g in shared])
                )
                for arm in panel_arms
            }
            for rung in RUNGS
        }
        points = []
        for position, column in enumerate(columns):
            points.append(
                {
                    **column,
                    "point": bands["point"][position],
                    "interval": bands["interval"][position],
                    "verdict": verdicts[position],
                    "marginal_interval": bands["marginal_interval"][position],
                    "marginal_verdict": marginal_verdicts[position],
                }
            )

        # Pure couplings reverse increment: couplings_given_geom_prof_model
        coup_rows = [
            [
                sign * cache[("U_C_M", "nuisance+profile", str(arm["arm"]))][stratum][group]
                - sign
                * cache[("U_C_M", "nuisance+profile+couplings", str(arm["arm"]))][stratum][group]
                for group in shared
            ]
            for arm in panel_arms
        ]
        coup_matrix = np.asarray(coup_rows, dtype=np.float64).T
        coup_bands = _mcd.paired_bands_with_marginal(coup_matrix)
        coup_verdicts = bidir.verdict(coup_bands["interval"])
        coup_marginal_verdicts = bidir.verdict(coup_bands["marginal_interval"])
        coup_columns = []
        for position, arm in enumerate(panel_arms):
            arm_name = str(arm["arm"])
            coup_columns.append(
                {
                    "arm": arm_name,
                    "contrast": "couplings_given_geom_prof_model",
                    "reduced_design": "geom+prof+M",
                    "joint_design": "geom+prof+couplings+M",
                    "reduced_performance": joint_performance["nuisance+profile"][arm_name],
                    "joint_performance": joint_performance["nuisance+profile+couplings"][arm_name],
                    "point": coup_bands["point"][position],
                    "interval": coup_bands["interval"][position],
                    "verdict": coup_verdicts[position],
                    "marginal_interval": coup_bands["marginal_interval"][position],
                    "marginal_verdict": coup_marginal_verdicts[position],
                }
            )

        # Retain exact per-group vectors for top-rung forward and pure couplings reverse
        # so Deep-vs-Shallow contrasts can resample groups cleanly with zero refitting.
        fwd_top_by_arm: dict[str, list[float]] = {}
        rev_top_by_arm: dict[str, list[float]] = {}
        coup_rev_by_arm: dict[str, list[float]] = {}
        for position, arm in enumerate(panel_arms):
            arm_name = str(arm["arm"])
            fwd_top_by_arm[arm_name] = [
                float(
                    sign * cache[("U_C", "nuisance+profile+couplings", arm_name)][stratum][g]
                    - sign * cache[("U_C_M", "nuisance+profile+couplings", arm_name)][stratum][g]
                )
                for g in shared
            ]
            rev_top_by_arm[arm_name] = [
                float(
                    sign * cache[("M", None, arm_name)][stratum][g]
                    - sign * cache[("U_C_M", "nuisance+profile+couplings", arm_name)][stratum][g]
                )
                for g in shared
            ]
            coup_rev_by_arm[arm_name] = [float(v) for v in coup_rows[position]]

        per_group_increments[stratum] = {
            "forward_model_given_gpc": fwd_top_by_arm,
            "reverse_controls_given_model": rev_top_by_arm,
            "reverse_couplings_given_gpM": coup_rev_by_arm,
        }

        payload_strata[stratum] = {
            "units": declaration["units"],
            "to_loss": sign,
            "native": {
                key: value
                for key, value in bands.items()
                if key not in ("point", "interval", "marginal_interval")
            },
            "columns": points,
            "control_performance": control_performance,
            "joint_performance": joint_performance,
            "model_alone_performance": {
                str(arm["arm"]): float(
                    np.mean([cache[("M", None, str(arm["arm"]))][stratum][g] for g in shared])
                )
                for arm in panel_arms
            },
            "states": bidir.counts(verdicts),
            "marginal_states": bidir.counts(marginal_verdicts),
            "pure_couplings_increment": {
                "contrast": "couplings_given_geom_prof_model",
                "definition": "J(geom+prof+couplings+M) - J(geom+prof+M) oriented to loss reduction",
                "native": {
                    key: value
                    for key, value in coup_bands.items()
                    if key not in ("point", "interval", "marginal_interval")
                },
                "columns": coup_columns,
                "states": bidir.counts(coup_verdicts),
                "marginal_states": bidir.counts(coup_marginal_verdicts),
            },
        }

    return {
        "complete": complete,
        "shared": shared,
        "declared": declared,
        "states": states,
        "strata": payload_strata,
        "per_group_increments": per_group_increments,
    }


def two_sample_depth_contrast_bands(
    shallow_values: np.ndarray,
    deep_values: np.ndarray,
    column_names: Sequence[str],
    *,
    draws: int = bidir.DRAWS,
    seed: int = bidir.SEED,
    confidence: float = bidir.CONFIDENCE,
) -> dict[str, Any]:
    """Independent two-sample group-bootstrap simultaneous + marginal bands for Shallow vs Deep.

    ``shallow_values`` has shape ``(n_shallow_groups, k_cols)`` and ``deep_values``
    has shape ``(n_deep_groups, k_cols)``. Because Deep ($N_{\\mathrm{eff}}/L \\ge 5$)
    and Shallow ($N_{\\mathrm{eff}}/L < 5$) are disjoint biological family subsets,
    each bootstrap draw resamples the Shallow families and Deep families
    independently, keeping all ``k_cols`` paired within each stratum so the
    simultaneous max-absolute-centered-deviation critical value accounts for full
    cross-arm correlation.
    """
    sh = np.asarray(shallow_values, dtype=np.float64)
    dp = np.asarray(deep_values, dtype=np.float64)
    if sh.ndim != 2 or dp.ndim != 2 or sh.shape[1] != dp.shape[1] or sh.shape[1] != len(column_names):
        raise ValueError("shallow_values and deep_values must be 2D with matching column count")
    if len(sh) < 2 or len(dp) < 2:
        raise ValueError("two-sample depth contrast needs at least two groups per stratum")
    if not (np.isfinite(sh).all() and np.isfinite(dp).all()):
        raise ValueError("non-finite values in two-sample depth contrast matrix")

    sh_mean = sh.mean(axis=0)
    dp_mean = dp.mean(axis=0)
    diff_sh_minus_dp = sh_mean - dp_mean

    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    for start in range(0, draws, bidir.DRAW_CHUNK):
        stop = min(start + bidir.DRAW_CHUNK, draws)
        chunk = stop - start
        idx_sh = rng.integers(0, len(sh), size=(chunk, len(sh)))
        idx_dp = rng.integers(0, len(dp), size=(chunk, len(dp)))
        blocks.append(sh[idx_sh].mean(axis=1) - dp[idx_dp].mean(axis=1))
    boot = np.concatenate(blocks, axis=0)

    centered = boot - diff_sh_minus_dp
    critical = float(np.quantile(np.max(np.abs(centered), axis=1), confidence))
    alpha = 0.5 * (1.0 - confidence)
    marg_lo_sd = np.quantile(boot, alpha, axis=0)
    marg_hi_sd = np.quantile(boot, 1.0 - alpha, axis=0)

    sim_sd = np.column_stack([diff_sh_minus_dp - critical, diff_sh_minus_dp + critical])
    marg_sd = np.column_stack([marg_lo_sd, marg_hi_sd])
    verdicts_sd = bidir.verdict(sim_sd)
    marg_verdicts_sd = bidir.verdict(marg_sd)

    diff_dp_minus_sh = -diff_sh_minus_dp
    sim_ds = np.column_stack([diff_dp_minus_sh - critical, diff_dp_minus_sh + critical])
    marg_ds = np.column_stack([-marg_hi_sd, -marg_lo_sd])
    verdicts_ds = bidir.verdict(sim_ds)
    marg_verdicts_ds = bidir.verdict(marg_ds)

    rows = []
    for idx, name in enumerate(column_names):
        rows.append(
            {
                "column": name,
                "shallow_mean": float(sh_mean[idx]),
                "deep_mean": float(dp_mean[idx]),
                "shallow_minus_deep": {
                    "point": float(diff_sh_minus_dp[idx]),
                    "interval": [float(sim_sd[idx, 0]), float(sim_sd[idx, 1])],
                    "verdict": verdicts_sd[idx],
                    "marginal_interval": [float(marg_sd[idx, 0]), float(marg_sd[idx, 1])],
                    "marginal_verdict": marg_verdicts_sd[idx],
                },
                "deep_minus_shallow": {
                    "point": float(diff_dp_minus_sh[idx]),
                    "interval": [float(sim_ds[idx, 0]), float(sim_ds[idx, 1])],
                    "verdict": verdicts_ds[idx],
                    "marginal_interval": [float(marg_ds[idx, 0]), float(marg_ds[idx, 1])],
                    "marginal_verdict": marg_verdicts_ds[idx],
                },
            }
        )

    return {
        "method": (
            "independent two-sample paired-column family bootstrap maximum absolute "
            "centered deviation in native units; Shallow and Deep resampled independently"
        ),
        "draws": int(draws),
        "seed": int(seed),
        "confidence": float(confidence),
        "n_shallow_groups": int(len(sh)),
        "n_deep_groups": int(len(dp)),
        "family_size": int(len(column_names)),
        "critical_value": critical,
        "states_shallow_minus_deep": bidir.counts(verdicts_sd),
        "marginal_states_shallow_minus_deep": bidir.counts(marg_verdicts_sd),
        "states_deep_minus_shallow": bidir.counts(verdicts_ds),
        "marginal_states_deep_minus_shallow": bidir.counts(marg_verdicts_ds),
        "columns": rows,
    }


def _arm_meta_dict(arm_id: str) -> dict[str, Any]:
    meta = get_arm_metadata(arm_id)
    return {
        "arm_id": meta.arm_id,
        "display_name": meta.display_name,
        "family": meta.family,
        "paradigm": meta.paradigm,
        "modality": meta.modality,
        "architecture": meta.architecture,
        "params_m": meta.params_m,
        "vocab_type": meta.vocab_type,
        "origin_cohort": meta.origin_cohort,
    }


def _summarise_discrimination_columns(
    table: Mapping[str, Mapping[str, float | None]],
    selected_columns: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str], dict[str, list[float]]]:
    names = [str(col["readout"]) for col in selected_columns]
    complete_refs = [
        assay for assay in sorted(table) if all(table[assay][name] is not None for name in names)
    ]
    out_columns: list[dict[str, Any]] = []
    per_ref_vectors: dict[str, list[float]] = {}
    if len(complete_refs) >= MINIMUM_BOOTSTRAP_UNITS:
        matrix = np.asarray(
            [[float(table[assay][name]) for name in names] for assay in complete_refs],
            dtype=np.float64,
        )
        bands = _mcd.paired_bands_with_marginal(matrix)
        verdicts = bidir.verdict(bands["interval"])
        marginal_verdicts = bidir.verdict(bands["marginal_interval"])
        for position, column in enumerate(selected_columns):
            out_columns.append(
                {
                    **dict(column),
                    "point": bands["point"][position],
                    "interval": bands["interval"][position],
                    "verdict": verdicts[position],
                    "marginal_interval": bands["marginal_interval"][position],
                    "marginal_verdict": marginal_verdicts[position],
                }
            )
            per_ref_vectors[str(column["readout"])] = [float(v) for v in matrix[:, position]]
        record_band = {
            key: value
            for key, value in bands.items()
            if key not in ("point", "interval", "marginal_interval")
        }
        record_band.update(
            {
                "units": "rank-sum AUC minus one half, dimensionless",
                "resampling_unit": "biological family; one reference per family here",
                "references": complete_refs,
            }
        )
    else:
        record_band = {
            "refused": (
                f"{len(complete_refs)} references carry every column against the "
                f"{MINIMUM_BOOTSTRAP_UNITS}-unit resampling floor"
            )
        }
        for column in selected_columns:
            out_columns.append(
                {
                    **dict(column),
                    "point": None,
                    "interval": None,
                    "verdict": "unresolved",
                    "marginal_interval": None,
                    "marginal_verdict": "unresolved",
                }
            )
    return out_columns, record_band, complete_refs, per_ref_vectors


def _build_roster_accounting(
    discovered: Mapping[str, DiscoveredExtraction],
    refused_moe: Mapping[str, Mapping[str, Any]],
    admitted_all_arms: Sequence[str],
    full_alpha_arms: Sequence[str],
    refused_below_floor: Sequence[Mapping[str, Any]],
    pair_fraction_by_arm: Mapping[str, float],
) -> dict[str, Any]:
    """Exact disjoint 62/62 single-sequence roster accounting."""
    fa_set = set(full_alpha_arms)
    adm_set = set(admitted_all_arms)
    thin_set = adm_set - fa_set
    low_ref_map = {str(r["arm"]): dict(r) for r in refused_below_floor}
    low_set = set(low_ref_map)
    moe_set = set(refused_moe) - adm_set - low_set
    accounted = fa_set | thin_set | low_set | moe_set
    not_yet = sorted(set(UNIFIED_62_SINGLE_SEQUENCE_ARMS) - accounted)

    assert len(fa_set) + len(thin_set) + len(low_set) + len(moe_set) + len(not_yet) == 62
    assert (fa_set | thin_set | low_set | moe_set | set(not_yet)) == set(
        UNIFIED_62_SINGLE_SEQUENCE_ARMS
    )

    return {
        "n_roster_total": len(UNIFIED_62_SINGLE_SEQUENCE_ARMS),
        "n_admitted_full_alphabet": len(fa_set),
        "n_admitted_thin_bpe": len(thin_set),
        "n_refused_below_8_refs": len(low_set),
        "n_refused_moe_prefix_layout": len(moe_set),
        "n_not_yet_extracted": len(not_yet),
        "n_accounted_total": len(fa_set) + len(thin_set) + len(low_set) + len(moe_set) + len(not_yet),
        "admitted_full_alphabet": list(sort_arms(fa_set)),
        "admitted_thin_bpe": [
            {
                "arm_id": arm_id,
                "pair_coverage_fraction": float(pair_fraction_by_arm.get(arm_id, 0.0)),
                **_arm_meta_dict(arm_id),
            }
            for arm_id in sort_arms(thin_set)
        ],
        "refused_below_8_refs": [low_ref_map[arm_id] for arm_id in sort_arms(low_set)],
        "refused_moe_prefix_layout": [dict(refused_moe[arm_id]) for arm_id in sort_arms(moe_set)],
        "not_yet_extracted": list(sort_arms(not_yet)),
        "discovered_count": len(discovered),
    }


def run_contacts_unified_62(
    *,
    root: Path | None = None,
    out_dir: Path | None = None,
    extraction_roots: Sequence[Path] | None = None,
    workers: int = 4,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Execute unified 62-model R3 contact discrimination + bidirectional ablation across All/Deep/Shallow."""
    base = REPO_ROOT if root is None else Path(root)
    target_dir = (base / UNIFIED_OUTPUT_DIR) if out_dir is None else Path(out_dir)
    max_workers = max(1, min(int(workers), 4))

    structures_path = base / DEFAULT_STRUCTURES_PATH
    coverage_path = base / DEFAULT_COVERAGE_PATH
    cohort_path = base / DEFAULT_COHORT_PATH
    params_roots = [base / rel for rel in DEFAULT_PARAMS_ROOTS]

    with gzip.open(structures_path, "rt", encoding="utf-8") as handle:
        sites = json.load(handle)
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    cohort_rows = {
        row["assay"]: row
        for row in json.loads(cohort_path.read_text(encoding="utf-8"))["assays"]
    }
    by_assay: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in sites:
        by_assay[row["assay_id"]].append(row)

    admitted_refs, refused_refs = _mcd.reference_rows(
        by_assay,
        coverage,
        params_roots,
        cutoff=CONTACT_ANGSTROM,
        separation=MIN_SEQUENCE_SEPARATION,
    )
    if not admitted_refs:
        raise RuntimeError("No reference carries both an admitted structure and fitted Potts params")

    # Precompute 0-based pair blocks, MSA conservation, and realised depth once across all 28 refs.
    blocks_all: dict[str, dict[str, Any]] = {}
    depth_all: dict[str, float] = {}
    explicit_auc_all: dict[str, dict[str, float | None]] = {}
    ref_meta_all: dict[str, dict[str, Any]] = {}
    for assay, record in sorted(admitted_refs.items()):
        work = Path(record["params"]).parent
        conservation = _fcb.column_conservation(work / "full.fa", work / "full.weights")
        if len(conservation) != record["length"]:
            raise RuntimeError(
                f"{assay}: alignment has {len(conservation)} columns vs WT length {record['length']}"
            )
        depth = _mcd.effective_depth_per_site(work, record["length"])
        depth_all[assay] = depth
        keys, built = _fcb.pair_blocks(record, conservation)
        blocks_all[assay] = {"keys": keys, **built}
        _, label_bool, potts, gap = _mcd.pair_coupling_scores(record)
        explicit_auc_all[assay] = {
            "couplings_apc_frobenius": _mcd.auc_minus_half(potts, label_bool),
            "negative_sequence_separation": _mcd.auc_minus_half(gap, label_bool),
        }
        ref_meta_all[assay] = {
            "family": record["family"],
            "entry": record["entry"],
            "length": int(record["length"]),
            "pairs": len(keys),
            "contacts": int(built["label"].sum()),
            "admitted_coordinates": len(record["geometry"]),
            "params": record["params"],
            "params_sha256": record["params_sha256"],
            "effective_depth_per_site": depth,
            "msa_sha256": sha256_file(work / "full.fa"),
            "weights_sha256": sha256_file(work / "full.weights"),
        }

    discovered, refused_moe = discover_e01_extractions(
        root=base, extraction_roots=extraction_roots
    )
    if not discovered:
        raise RuntimeError("No position_likelihood_v1 E01 extraction directories discovered")

    ordered_arm_ids = list(sort_arms(discovered.keys()))
    ordered_exts = [discovered[arm_id] for arm_id in ordered_arm_ids]

    # Extract per-pair propagation responses ONCE per arm across all 28 references.
    # Use fork ProcessPoolExecutor on Linux so Python archive validation scales across workers.
    model_all: dict[str, dict[str, dict[str, Any]]] = {}
    cov_all: dict[str, dict[str, dict[str, int]]] = {}
    auc_all: dict[str, dict[str, float | None]] = {}

    if max_workers == 1 or len(ordered_exts) == 1:
        extracted = [
            _extract_one_arm_all_refs(ext, admitted_refs, blocks_all, cohort_rows)
            for ext in ordered_exts
        ]
    else:
        mp_ctx = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(max_workers, len(ordered_exts)), mp_context=mp_ctx
        ) as pool:
            futs = [
                pool.submit(_extract_one_arm_all_refs, ext, admitted_refs, blocks_all, cohort_rows)
                for ext in ordered_exts
            ]
            extracted = [fut.result() for fut in futs]

    for arm_id, arm_models, arm_cov, arm_auc in extracted:
        model_all[arm_id] = arm_models
        cov_all[arm_id] = arm_cov
        auc_all[arm_id] = arm_auc

    sources_block = {
        "cohort": {"path": str(DEFAULT_COHORT_PATH), "sha256": sha256_file(cohort_path)},
        "structures": {"path": str(DEFAULT_STRUCTURES_PATH), "sha256": sha256_file(structures_path)},
        "coverage": {"path": str(DEFAULT_COVERAGE_PATH), "sha256": sha256_file(coverage_path)},
        "params_roots": [str(rel) for rel in DEFAULT_PARAMS_ROOTS],
        "extractions": [
            {
                "arm": ext.arm_id,
                "directory": str(ext.directory),
                "source_dir_name": ext.source_dir_name,
                "index_sha256": ext.index_sha256,
                "paradigm": ext.paradigm,
                "dtype_realised": ext.dtype_realised,
                "prefix_tier": ext.prefix_tier,
                "upstream_tolerance_nats": ext.upstream_tolerance_nats,
                "sub_tier2_caveat": ext.sub_tier2_caveat,
                "roster_metadata": _arm_meta_dict(ext.arm_id),
            }
            for ext in ordered_exts
        ],
        "sub_tier2_prefix_caveats": [
            {
                "arm_id": ext.arm_id,
                "upstream_tolerance_nats": ext.upstream_tolerance_nats,
                "caveat": ext.sub_tier2_caveat,
            }
            for ext in ordered_exts
            if ext.sub_tier2_caveat is not None
        ],
    }

    disc_payloads: dict[str, dict[str, Any]] = {}
    bidir_payloads: dict[str, dict[str, Any]] = {}
    # Store per-group vectors for exact Deep-vs-Shallow two-sample bootstrap contrasts.
    depth_vectors: dict[str, dict[str, Any]] = {}
    artifact_digests: dict[str, dict[str, str]] = {}

    for stratum_cfg in DEPTH_STRATA:
        skey = str(stratum_cfg["key"])
        min_neff = float(stratum_cfg["min_neff"])
        max_neff = stratum_cfg["max_neff"]
        suffix = str(stratum_cfg["suffix"])

        kept_assays: list[str] = []
        below_floor: dict[str, float] = {}
        above_ceiling: dict[str, float] = {}
        for assay in sorted(admitted_refs):
            d = depth_all[assay]
            if d < min_neff:
                below_floor[assay] = d
                continue
            if max_neff is not None and d >= float(max_neff):
                above_ceiling[assay] = d
                continue
            kept_assays.append(assay)

        support_s: dict[str, dict[str, Any]] = {}
        blocks_s: dict[str, dict[str, Any]] = {}
        table_s: dict[str, dict[str, float | None]] = {}
        for assay in kept_assays:
            blocks_s[assay] = blocks_all[assay]
            covered_here = {
                arm_id: cov_all[arm_id][assay]
                for arm_id in ordered_arm_ids
                if assay in cov_all[arm_id]
            }
            support_s[assay] = {**ref_meta_all[assay], "arm_coverage": covered_here}
            row_auc: dict[str, float | None] = dict(explicit_auc_all[assay])
            for arm_id in ordered_arm_ids:
                row_auc[f"propagation:{arm_id}"] = auc_all[arm_id].get(assay)
            table_s[assay] = row_auc

        total_eligible_pairs = sum(entry["pairs"] for entry in support_s.values())
        pair_fraction_by_arm = {
            arm_id: (
                sum(
                    support_s[assay]["arm_coverage"].get(arm_id, {}).get("pairs", 0)
                    for assay in kept_assays
                )
                / total_eligible_pairs
                if total_eligible_pairs
                else 0.0
            )
            for arm_id in ordered_arm_ids
        }

        # ---- 1. Contact discrimination (exact measure_contact_discrimination.py logic) ----
        base_cols: list[dict[str, Any]] = [
            {"readout": "couplings_apc_frobenius", "kind": "explicit"},
            {"readout": "negative_sequence_separation", "kind": "design"},
        ]
        for ext in ordered_exts:
            base_cols.append(
                {
                    "readout": f"propagation:{ext.arm_id}",
                    "arm_id": ext.arm_id,
                    "kind": "model",
                    "paradigm": ext.paradigm,
                    "roster_paradigm": get_arm_metadata(ext.arm_id).paradigm,
                    "family": get_arm_metadata(ext.arm_id).family,
                    "params_m": get_arm_metadata(ext.arm_id).params_m,
                }
            )

        coverage_by_col = {
            col["readout"]: sum(1 for assay in table_s if table_s[assay][col["readout"]] is not None)
            for col in base_cols
        }
        admitted_cols, refused_cols = [], []
        for col in base_cols:
            cov_n = coverage_by_col[col["readout"]]
            if cov_n >= MINIMUM_BOOTSTRAP_UNITS:
                admitted_cols.append(col)
            else:
                refused_cols.append(
                    {
                        **col,
                        "references_covered": cov_n,
                        "reason": (
                            f"{cov_n} references carry this readout against the "
                            f"{MINIMUM_BOOTSTRAP_UNITS}-unit resampling floor"
                        ),
                    }
                )

        disc_cols, disc_band, disc_complete, disc_vecs_all = _summarise_discrimination_columns(
            table_s, admitted_cols
        )

        fa_disc_cols_in, thin_bpe_disc = [], []
        for col in admitted_cols:
            if col["kind"] != "model":
                fa_disc_cols_in.append(col)
                continue
            aid = str(col["arm_id"])
            frac = pair_fraction_by_arm[aid]
            if frac >= FULL_ALPHABET_MIN_PAIR_FRACTION:
                fa_disc_cols_in.append(col)
            else:
                thin_bpe_disc.append({"arm": aid, "pair_coverage_fraction": frac})

        fa_disc_cols, fa_disc_band, fa_disc_complete, disc_vecs_fa = _summarise_discrimination_columns(
            table_s, fa_disc_cols_in
        )

        disc_doc = {
            "schema": "contact_discrimination_v1",
            "experiment": (
                "Unified 62-model R3 contact localisation by explicit couplings and E01 propagation"
            ),
            "question": (
                "whether a residue pair can be identified as a three-dimensional contact "
                "from the fitted Potts coupling norm, from a model's per-pair propagation "
                "response, or from sequence separation alone"
            ),
            "created_utc": _now_utc(),
            "depth_stratum": skey,
            "contact_definition": {
                "cutoff_angstrom": float(CONTACT_ANGSTROM),
                "min_separation": int(MIN_SEQUENCE_SEPARATION),
                "rule": (
                    "C-beta, or C-alpha for glycine, strictly below 8.0 A; "
                    "0-based residue indices matching structure_sites() and contact_pairs()"
                ),
            },
            "statistic": {
                "per_reference": "rank-sum AUC minus one half, mid-ranks on ties",
                "pair_reduction": _mcd.PAIR_REDUCTION,
                "chance_reading": _mcd.CHANCE_READING,
                "gauge": "zero-sum, through src.capability.context.potts_gauge.to_zero_sum",
                "causal_sides": PARADIGM_SIDES,
            },
            "columns": disc_cols,
            "columns_refused": refused_cols,
            "column_coverage": coverage_by_col,
            "native": disc_band,
            "full_alphabet_secondary": {
                "rule": (
                    f"admitted non-model columns plus model arms covering at least "
                    f"{FULL_ALPHABET_MIN_PAIR_FRACTION:.2f} of eligible residue pairs across "
                    "admitted references, banded on their own complete-case intersection"
                ),
                "min_pair_fraction": FULL_ALPHABET_MIN_PAIR_FRACTION,
                "pair_coverage_fraction_by_arm": pair_fraction_by_arm,
                "thin_bpe_arms_excluded": thin_bpe_disc,
                "references_complete": fa_disc_complete,
                "columns": fa_disc_cols,
                "native": fa_disc_band,
            },
            "depth": {
                "min_neff_per_site": min_neff,
                "max_neff_per_site": None if max_neff is None else float(max_neff),
                "source": "the full.weights file each selected.params fit consumed",
                "references_below_floor": below_floor,
                "references_at_or_above_ceiling": above_ceiling,
                "realised": {assay: depth_all[assay] for assay in disc_complete},
                "realised_full_alphabet": {assay: depth_all[assay] for assay in fa_disc_complete},
            },
            "families": sorted({str(admitted_refs[a]["family"]) for a in kept_assays}),
            "references_admitted": len(kept_assays),
            "references_complete": disc_complete,
            "references_refused": refused_refs,
            "support": support_s,
            "per_reference": table_s,
            "sources": sources_block,
        }
        disc_payloads[skey] = disc_doc

        # ---- 2. Bidirectional ablation (exact fit_contact_bidirectional.py logic) ----
        admitted_bidir_arms: list[dict[str, Any]] = []
        refused_bidir_arms: list[dict[str, Any]] = []
        model_s: dict[str, dict[str, dict[str, Any]]] = {}
        for ext in ordered_exts:
            aid = ext.arm_id
            carried = sorted(a for a in kept_assays if a in model_all[aid])
            model_s[aid] = {a: model_all[aid][a] for a in carried}
            if len(carried) >= MINIMUM_BOOTSTRAP_UNITS:
                admitted_bidir_arms.append(
                    {"arm": aid, "paradigm": ext.paradigm, "references": carried}
                )
            else:
                refused_bidir_arms.append(
                    {
                        "arm": aid,
                        "paradigm": ext.paradigm,
                        "references_covered": len(carried),
                        "reason": (
                            f"{len(carried)} references against the "
                            f"{MINIMUM_BOOTSTRAP_UNITS}-unit floor; named rather than estimated"
                        ),
                    }
                )

        if not admitted_bidir_arms:
            raise RuntimeError(f"Stratum {skey}: no arm clears the {MINIMUM_BOOTSTRAP_UNITS}-unit floor")

        primary_b = evaluate_bidirectional_panel_fast(
            admitted_bidir_arms, blocks_s, model_s, support_s, workers=max_workers
        )

        fa_bidir_arms = [
            arm
            for arm in admitted_bidir_arms
            if pair_fraction_by_arm[arm["arm"]] >= FULL_ALPHABET_MIN_PAIR_FRACTION
        ]
        thin_bpe_bidir = [
            {"arm": arm["arm"], "pair_coverage_fraction": pair_fraction_by_arm[arm["arm"]]}
            for arm in admitted_bidir_arms
            if pair_fraction_by_arm[arm["arm"]] < FULL_ALPHABET_MIN_PAIR_FRACTION
        ]

        if not fa_bidir_arms:
            fa_secondary_b: dict[str, Any] = {
                "min_pair_fraction": FULL_ALPHABET_MIN_PAIR_FRACTION,
                "pair_coverage_fraction_by_arm": pair_fraction_by_arm,
                "thin_bpe_arms_excluded": thin_bpe_bidir,
                "refused": "no admitted arm cleared the full-alphabet pair coverage threshold",
            }
            fa_panel_eval = None
        elif len(fa_bidir_arms) == len(admitted_bidir_arms):
            fa_panel_eval = primary_b
            fa_secondary_b = {
                "min_pair_fraction": FULL_ALPHABET_MIN_PAIR_FRACTION,
                "pair_coverage_fraction_by_arm": pair_fraction_by_arm,
                "thin_bpe_arms_excluded": thin_bpe_bidir,
                "identical_to_primary": True,
                "family": {
                    "size_per_stratum": primary_b["declared"],
                    "arms": [arm["arm"] for arm in fa_bidir_arms],
                },
                "references_offered": primary_b["complete"],
                "banded_families": primary_b["shared"],
                "realised_depth": {a: depth_all[a] for a in primary_b["complete"]},
                "strata": primary_b["strata"],
            }
        else:
            fa_panel_eval = evaluate_bidirectional_panel_fast(
                fa_bidir_arms, blocks_s, model_s, support_s, workers=max_workers
            )
            fa_secondary_b = {
                "rule": (
                    f"admitted arms covering at least {FULL_ALPHABET_MIN_PAIR_FRACTION:.2f} of "
                    "eligible residue pairs across admitted references, fitted and banded on "
                    "their own complete-case reference intersection"
                ),
                "min_pair_fraction": FULL_ALPHABET_MIN_PAIR_FRACTION,
                "pair_coverage_fraction_by_arm": pair_fraction_by_arm,
                "thin_bpe_arms_excluded": thin_bpe_bidir,
                "family": {
                    "size_per_stratum": fa_panel_eval["declared"],
                    "arms": [arm["arm"] for arm in fa_bidir_arms],
                },
                "references_offered": fa_panel_eval["complete"],
                "banded_families": fa_panel_eval["shared"],
                "realised_depth": {a: depth_all[a] for a in fa_panel_eval["complete"]},
                "per_arm": {
                    name: {
                        "references": st["references"],
                        "pairs": st["pairs"],
                        "contacts": st["contacts"],
                        "families": len(st["order"]),
                    }
                    for name, st in fa_panel_eval["states"].items()
                },
                "strata": fa_panel_eval["strata"],
            }

        accounting_s = _build_roster_accounting(
            discovered=discovered,
            refused_moe=refused_moe,
            admitted_all_arms=[arm["arm"] for arm in admitted_bidir_arms],
            full_alpha_arms=[arm["arm"] for arm in fa_bidir_arms],
            refused_below_floor=refused_bidir_arms,
            pair_fraction_by_arm=pair_fraction_by_arm,
        )

        bidir_doc = {
            "schema": "contact_bidirectional_v1",
            "experiment": (
                "Unified 62-model R3 bidirectional ablation: E01 pair propagation against explicit couplings"
            ),
            "estimand": (
                "held-out prediction of whether a residue pair is in three-dimensional "
                "contact, on the pair population the contact definition admits"
            ),
            "created_utc": _now_utc(),
            "depth_stratum": skey,
            "contact_definition": {
                "cutoff_angstrom": float(CONTACT_ANGSTROM),
                "min_separation": int(MIN_SEQUENCE_SEPARATION),
            },
            "family": {
                "size_per_stratum": primary_b["declared"],
                "rule": "2 directions x rungs x admitted arms, banded as one family per stratum",
                "directions": list(bidir.DIRECTIONS),
                "designs": {d: list(pair) for d, pair in bidir.PRIMARY_DESIGNS.items()},
                "rungs": {r: list(names) for r, names in RUNGS.items()},
                "arms": [arm["arm"] for arm in admitted_bidir_arms],
            },
            "resampling_unit": "biological family; one reference per family on this cohort",
            "fitter": {
                "source": "src/capability/extensions/phenotype_breadth.py",
                "outer_folds": breadth.OUTER_FOLDS,
                "inner_folds": breadth.INNER_FOLDS,
                "split_seeds": list(breadth.SPLIT_SEEDS),
                "weighting": (
                    "family-equal, reference-equal within family, pair-equal within "
                    "reference, through phenotype_breadth.row_weights"
                ),
                "seed_rule": "three split seeds averaged within group before any resample",
            },
            "excluded_nuisance": _fcb.EXCLUDED_NUISANCE,
            "strata": primary_b["strata"],
            "full_alphabet_secondary": fa_secondary_b,
            "roster_accounting": accounting_s,
            "depth": {
                "min_neff_per_site": min_neff,
                "max_neff_per_site": None if max_neff is None else float(max_neff),
                "source": "the full.weights file each selected.params fit consumed",
                "references_below_floor": below_floor,
                "references_at_or_above_ceiling": above_ceiling,
                "realised": {a: depth_all[a] for a in primary_b["complete"]},
            },
            "support": {
                "rule": (
                    "each arm's three designs share that arm's own pair support; the "
                    "bootstrap axis is the families every admitted arm carries"
                ),
                "references_offered": primary_b["complete"],
                "references_admitted": sorted(blocks_s),
                "banded_families": primary_b["shared"],
                "per_arm": {
                    name: {
                        "references": st["references"],
                        "pairs": st["pairs"],
                        "contacts": st["contacts"],
                        "families": len(st["order"]),
                    }
                    for name, st in primary_b["states"].items()
                },
                "per_reference": support_s,
                "references_refused": refused_refs,
            },
            "arms_refused": refused_bidir_arms,
            "arms_refused_moe_layout": accounting_s["refused_moe_prefix_layout"],
            "sources": sources_block,
        }
        bidir_payloads[skey] = bidir_doc

        depth_vectors[skey] = {
            "disc_all_admitted": disc_vecs_all,
            "disc_full_alphabet": disc_vecs_fa,
            "bidir_all_admitted": primary_b["per_group_increments"],
            "bidir_full_alphabet": (
                None if fa_panel_eval is None else fa_panel_eval["per_group_increments"]
            ),
            "fa_arms": [arm["arm"] for arm in fa_bidir_arms],
            "all_arms": [arm["arm"] for arm in admitted_bidir_arms],
        }

        if write_outputs:
            disc_path = target_dir / f"contact_discrimination_unified_62{suffix}.json"
            bidir_path = target_dir / f"contact_bidirectional_unified_62{suffix}.json"
            disc_sha = atomic_write_json_with_sha256(disc_path, disc_doc)
            bidir_sha = atomic_write_json_with_sha256(bidir_path, bidir_doc)
            artifact_digests[skey] = {
                "discrimination_path": str(disc_path.name),
                "discrimination_sha256": disc_sha,
                "bidirectional_path": str(bidir_path.name),
                "bidirectional_sha256": bidir_sha,
            }

    # ---- 3. Deep-vs-Shallow MSA Contrasts (Full-Alphabet primary + All-Admitted secondary) ----
    depth_contrasts: dict[str, Any] = {}
    for panel_key, disc_k, bidir_k, arms_k in (
        ("full_alphabet", "disc_full_alphabet", "bidir_full_alphabet", "fa_arms"),
        ("all_admitted", "disc_all_admitted", "bidir_all_admitted", "all_arms"),
    ):
        sh_disc = depth_vectors["shallow"][disc_k]
        dp_disc = depth_vectors["deep"][disc_k]
        sh_bidir = depth_vectors["shallow"][bidir_k]
        dp_bidir = depth_vectors["deep"][bidir_k]
        if not sh_disc or not dp_disc or sh_bidir is None or dp_bidir is None:
            continue

        # Discrimination columns common to both deep and shallow
        disc_col_names = [
            name
            for name in ("couplings_apc_frobenius", "negative_sequence_separation")
            if name in sh_disc and name in dp_disc
        ]
        shared_arms = list(
            sort_arms(
                set(depth_vectors["shallow"][arms_k]) & set(depth_vectors["deep"][arms_k])
            )
        )
        for arm_id in shared_arms:
            col_name = f"propagation:{arm_id}"
            if col_name in sh_disc and col_name in dp_disc:
                disc_col_names.append(col_name)

        sh_disc_mat = np.column_stack([sh_disc[c] for c in disc_col_names])
        dp_disc_mat = np.column_stack([dp_disc[c] for c in disc_col_names])
        panel_out: dict[str, Any] = {
            "arms": shared_arms,
            "auc_minus_half": two_sample_depth_contrast_bands(
                sh_disc_mat, dp_disc_mat, disc_col_names
            ),
        }

        for metric_stratum in ("contact_ranking", "contact_squared_error_clipped"):
            panel_out[metric_stratum] = {}
            for contrast_key in (
                "forward_model_given_gpc",
                "reverse_controls_given_model",
                "reverse_couplings_given_gpM",
            ):
                sh_map = sh_bidir[metric_stratum][contrast_key]
                dp_map = dp_bidir[metric_stratum][contrast_key]
                sh_mat = np.column_stack([sh_map[a] for a in shared_arms])
                dp_mat = np.column_stack([dp_map[a] for a in shared_arms])
                panel_out[metric_stratum][contrast_key] = two_sample_depth_contrast_bands(
                    sh_mat, dp_mat, shared_arms
                )

        depth_contrasts[panel_key] = panel_out

    # ---- 4. Consolidated per-arm summary table, scaling ladders, & paradigm aggregates ----
    master_accounting = bidir_payloads["all"]["roster_accounting"]
    per_arm_summary: dict[str, dict[str, Any]] = {}

    def _extract_panel_arm_metrics(stratum_key: str, arm_id: str) -> dict[str, Any]:
        disc = disc_payloads[stratum_key]
        bdoc = bidir_payloads[stratum_key]
        out: dict[str, Any] = {}
        for panel_label, disc_cols_list, b_strata in (
            (
                "full_alphabet",
                disc["full_alphabet_secondary"]["columns"],
                bdoc["full_alphabet_secondary"].get("strata"),
            ),
            (
                "all_admitted",
                disc["columns"],
                bdoc["strata"],
            ),
        ):
            d_col = next(
                (c for c in disc_cols_list if c.get("arm_id") == arm_id),
                None,
            )
            if d_col is None or b_strata is None:
                continue
            rank_s = b_strata["contact_ranking"]
            brier_c = b_strata["contact_squared_error_clipped"]
            fwd_col = next(
                (
                    c
                    for c in rank_s["columns"]
                    if c["arm"] == arm_id
                    and c["rung"] == "nuisance+profile+couplings"
                    and c["direction"] == "model_given_controls"
                ),
                None,
            )
            rev_col = next(
                (
                    c
                    for c in rank_s["columns"]
                    if c["arm"] == arm_id
                    and c["rung"] == "nuisance+profile+couplings"
                    and c["direction"] == "controls_given_model"
                ),
                None,
            )
            coup_col = next(
                (
                    c
                    for c in rank_s["pure_couplings_increment"]["columns"]
                    if c["arm"] == arm_id
                ),
                None,
            )
            brier_fwd = next(
                (
                    c
                    for c in brier_c["columns"]
                    if c["arm"] == arm_id
                    and c["rung"] == "nuisance+profile+couplings"
                    and c["direction"] == "model_given_controls"
                ),
                None,
            )
            brier_coup = next(
                (
                    c
                    for c in brier_c["pure_couplings_increment"]["columns"]
                    if c["arm"] == arm_id
                ),
                None,
            )
            out[panel_label] = {
                "auc_minus_half": d_col["point"],
                "auc_minus_half_sim95": d_col["interval"],
                "auc_minus_half_marg95": d_col["marginal_interval"],
                "auc_minus_half_verdict": d_col["verdict"],
                "M_alone_auc": rank_s["model_alone_performance"].get(arm_id),
                "U_gpc_auc": rank_s["control_performance"]["nuisance+profile+couplings"].get(arm_id),
                "J_gpc_M_auc": rank_s["joint_performance"]["nuisance+profile+couplings"].get(arm_id),
                "fwd_M_given_gpc_auc": None if fwd_col is None else fwd_col["point"],
                "fwd_M_given_gpc_sim95": None if fwd_col is None else fwd_col["interval"],
                "fwd_M_given_gpc_marg95": None if fwd_col is None else fwd_col["marginal_interval"],
                "fwd_M_given_gpc_verdict": None if fwd_col is None else fwd_col["verdict"],
                "fwd_M_given_gpc_marg_verdict": None if fwd_col is None else fwd_col["marginal_verdict"],
                "rev_gpc_given_M_auc": None if rev_col is None else rev_col["point"],
                "rev_gpc_given_M_sim95": None if rev_col is None else rev_col["interval"],
                "rev_gpc_given_M_marg95": None if rev_col is None else rev_col["marginal_interval"],
                "rev_gpc_given_M_verdict": None if rev_col is None else rev_col["verdict"],
                "rev_coup_given_gpM_auc": None if coup_col is None else coup_col["point"],
                "rev_coup_given_gpM_sim95": None if coup_col is None else coup_col["interval"],
                "rev_coup_given_gpM_marg95": None if coup_col is None else coup_col["marginal_interval"],
                "rev_coup_given_gpM_verdict": None if coup_col is None else coup_col["verdict"],
                "rev_coup_given_gpM_marg_verdict": (
                    None if coup_col is None else coup_col["marginal_verdict"]
                ),
                "fwd_M_given_gpc_brier_clipped": None if brier_fwd is None else brier_fwd["point"],
                "rev_coup_given_gpM_brier_clipped": (
                    None if brier_coup is None else brier_coup["point"]
                ),
            }
        return out

    fa_set_all = set(master_accounting["admitted_full_alphabet"])
    thin_set_all = {item["arm_id"] for item in master_accounting["admitted_thin_bpe"]}
    low_set_all = {item["arm"] for item in master_accounting["refused_below_8_refs"]}
    moe_set_all = {item["arm"] for item in master_accounting["refused_moe_prefix_layout"]}

    for arm_id in UNIFIED_62_SINGLE_SEQUENCE_ARMS:
        if arm_id in fa_set_all:
            bucket = "admitted_full_alphabet"
        elif arm_id in thin_set_all:
            bucket = "admitted_thin_bpe"
        elif arm_id in low_set_all:
            bucket = "refused_below_8_refs"
        elif arm_id in moe_set_all:
            bucket = "refused_moe_prefix_layout"
        else:
            bucket = "not_yet_extracted"

        entry: dict[str, Any] = {
            **_arm_meta_dict(arm_id),
            "status_bucket": bucket,
            "pair_coverage_fraction_all": (
                bidir_payloads["all"]["full_alphabet_secondary"]["pair_coverage_fraction_by_arm"].get(
                    arm_id
                )
            ),
            "strata": {
                skey: _extract_panel_arm_metrics(skey, arm_id)
                for skey in ("all", "deep", "shallow")
            },
        }
        if bucket == "refused_moe_prefix_layout":
            entry["refusal"] = dict(refused_moe[arm_id])
        per_arm_summary[arm_id] = entry

    # Scaling ladders view
    ladders_out: dict[str, list[dict[str, Any]]] = {}
    for family_name, member_ids in SCALING_LADDERS:
        ladder_rows = []
        for aid in member_ids:
            arm_entry = per_arm_summary[aid]
            pref_panel = "full_alphabet" if arm_entry["status_bucket"] == "admitted_full_alphabet" else "all_admitted"
            row = {
                "arm_id": aid,
                "params_m": arm_entry["params_m"],
                "paradigm": arm_entry["paradigm"],
                "status_bucket": arm_entry["status_bucket"],
                "panel_used": pref_panel,
            }
            for skey in ("all", "deep", "shallow"):
                m = arm_entry["strata"][skey].get(pref_panel)
                if m is not None:
                    row[f"{skey}_auc_minus_half"] = m["auc_minus_half"]
                    row[f"{skey}_fwd_M_given_gpc"] = m["fwd_M_given_gpc_auc"]
                    row[f"{skey}_rev_coup_given_gpM"] = m["rev_coup_given_gpM_auc"]
            if (
                row.get("shallow_fwd_M_given_gpc") is not None
                and row.get("deep_fwd_M_given_gpc") is not None
            ):
                row["shallow_minus_deep_fwd_M_given_gpc"] = (
                    row["shallow_fwd_M_given_gpc"] - row["deep_fwd_M_given_gpc"]
                )
                row["deep_minus_shallow_rev_coup_given_gpM"] = (
                    row["deep_rev_coup_given_gpM"] - row["shallow_rev_coup_given_gpM"]
                )
            ladder_rows.append(row)
        ladders_out[family_name] = ladder_rows

    # Paradigm aggregates (on full_alphabet where available, else all_admitted)
    paradigm_summary: dict[str, dict[str, Any]] = {}
    for paradigm in PARADIGM_ORDER:
        if paradigm == "multiseq_msa":
            continue
        members = [a for a in UNIFIED_62_SINGLE_SEQUENCE_ARMS if get_arm_metadata(a).paradigm == paradigm]
        admitted_fa = [a for a in members if per_arm_summary[a]["status_bucket"] == "admitted_full_alphabet"]
        admitted_any = [
            a
            for a in members
            if per_arm_summary[a]["status_bucket"] in ("admitted_full_alphabet", "admitted_thin_bpe")
        ]
        p_stats: dict[str, Any] = {
            "n_roster": len(members),
            "n_admitted_full_alphabet": len(admitted_fa),
            "n_admitted_any": len(admitted_any),
        }
        for skey in ("all", "deep", "shallow"):
            vals_auc = []
            vals_fwd = []
            vals_coup = []
            for a in admitted_any:
                panel = "full_alphabet" if a in admitted_fa else "all_admitted"
                m = per_arm_summary[a]["strata"][skey].get(panel)
                if m is not None and m["auc_minus_half"] is not None:
                    vals_auc.append(float(m["auc_minus_half"]))
                    vals_fwd.append(float(m["fwd_M_given_gpc_auc"]))
                    vals_coup.append(float(m["rev_coup_given_gpM_auc"]))
            if vals_auc:
                p_stats[skey] = {
                    "n_arms": len(vals_auc),
                    "auc_minus_half_mean": float(np.mean(vals_auc)),
                    "auc_minus_half_max": float(np.max(vals_auc)),
                    "fwd_M_given_gpc_mean": float(np.mean(vals_fwd)),
                    "fwd_M_given_gpc_max": float(np.max(vals_fwd)),
                    "rev_coup_given_gpM_mean": float(np.mean(vals_coup)),
                }
        paradigm_summary[paradigm] = p_stats

    # Explicit Potts coupling baseline summary across all three depth strata
    explicit_couplings_summary: dict[str, Any] = {}
    for skey in ("all", "deep", "shallow"):
        disc_fa = disc_payloads[skey]["full_alphabet_secondary"]
        coup_col = next(c for c in disc_fa["columns"] if c["readout"] == "couplings_apc_frobenius")
        sep_col = next(c for c in disc_fa["columns"] if c["readout"] == "negative_sequence_separation")
        explicit_couplings_summary[skey] = {
            "n_references_full_alphabet": len(disc_fa["references_complete"]),
            "n_references_all_admitted": len(disc_payloads[skey]["references_complete"]),
            "couplings_apc_frobenius_auc_minus_half": coup_col["point"],
            "couplings_apc_frobenius_sim95": coup_col["interval"],
            "couplings_apc_frobenius_marg95": coup_col["marginal_interval"],
            "couplings_apc_frobenius_verdict": coup_col["verdict"],
            "negative_sequence_separation_auc_minus_half": sep_col["point"],
        }

    master_doc = {
        "schema": "contacts_unified_62_v1",
        "experiment": (
            "Unified 62-model R3 contact discrimination, bidirectional ablation, and "
            "Deep-vs-Shallow MSA depth-contrast yardstick"
        ),
        "readout_mechanism": "perturbation_propagation_e01",
        "created_utc": _now_utc(),
        "contact_definition": {
            "cutoff_angstrom": float(CONTACT_ANGSTROM),
            "min_separation": int(MIN_SEQUENCE_SEPARATION),
            "indexing": "0-based wild-type residue indices (0 <= i < j < L)",
        },
        "depth_strata": {
            cfg["key"]: {
                "label": cfg["label"],
                "min_neff_per_site": cfg["min_neff"],
                "max_neff_per_site": cfg["max_neff"],
                "n_admitted_references": disc_payloads[cfg["key"]]["references_admitted"],
                "n_complete_full_alphabet": len(
                    disc_payloads[cfg["key"]]["full_alphabet_secondary"]["references_complete"]
                ),
                "n_complete_all_admitted": len(disc_payloads[cfg["key"]]["references_complete"]),
            }
            for cfg in DEPTH_STRATA
        },
        "roster_accounting": master_accounting,
        "explicit_couplings_summary": explicit_couplings_summary,
        "depth_contrasts_shallow_vs_deep": depth_contrasts,
        "scaling_ladders": ladders_out,
        "paradigm_summary": paradigm_summary,
        "per_arm": per_arm_summary,
        "artifacts": artifact_digests,
        "sources": sources_block,
    }

    if write_outputs:
        master_path = target_dir / "contacts_unified_62.json"
        master_sha = atomic_write_json_with_sha256(master_path, master_doc)
        master_doc["master_sha256"] = master_sha

    return {
        "master": master_doc,
        "discrimination": disc_payloads,
        "bidirectional": bidir_payloads,
    }
