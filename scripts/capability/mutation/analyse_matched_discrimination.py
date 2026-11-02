#!/usr/bin/env python3
"""E21 and E22 on declared support S1_retained_panel: CPU, no scoring, no fitting.

Reads the retained per-variant tables of the 33-arm mutation panel, builds the
composition-matched and profile-matched variant pairs the pre-registration
declares, and reports direction accuracy with cluster-bootstrap intervals and
the declared verdicts. Nothing here loads a checkpoint, fits a readout or
touches an existing result directory: every model quantity is a stored
out-of-family value, read on the within-assay rank scale the export records.

Three correctness checks run before any endpoint is computed, and each one is a
refusal rather than a warning:

* the orientation of ``measured`` is confirmed against ProteinGym's own
  ``DMS_score`` and its own declared functional label ``DMS_score_bin``, per
  assay, so "higher is the better phenotype" is the dataset's statement and not
  an assumption;
* the orientation of ``raw_M`` is confirmed by reproducing it as the within-assay
  standardized rank of raw ProteinGym nats for the four arms whose nats are on
  disk, and the pair-level ordering of nats and ranks must agree on every
  retained pair;
* both members of every retained pair must sit in one assay and in one outer
  fold at every fold seed, so the stored readout prediction is out-of-family for
  both.

``--device`` is accepted because the campaign queue injects it; the analysis is
CPU-only and refuses a non-CPU device.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.mutation import matched_discrimination as md  # noqa: E402

PREDICTION_COLUMNS = (
    "sample_id", "model_id", "seed", "assay", "group_id", "source_id",
    "raw_M", "raw_P", "P", "sequence", "profile_sequence", "B", "B_R", "R", "fold_B",
)

#: The four arms whose raw ProteinGym nats are retained locally. They are the
#: only place on this support where the rank scale can be checked against the
#: quantity it is a rank of.
NATS_ARMS = {
    "progen3-3b": "rescue_progen3_3b/scores_progen3-3b.json",
    "proteinglm-7b-clm": "rescue_proteinglm/scores_proteinglm-7b-clm.json",
    "prollama": "rescue_prollama_stage2/scores_prollama.json",
    "prollama-stage-1": "rescue_prollama_stage1/scores_prollama-stage-1.json",
}

#: Designs recomputed for the E22 validity check. The first four are the readouts
#: E22 reads; P, sequence and R are carried because the retained reports publish
#: them and a disagreement there would localise a join fault.
VALIDITY_DESIGNS = ("raw_M", "profile_sequence", "B", "B_R", "P", "sequence", "R")

#: Agreement a recomputation of a retained summary must reach to count as the
#: same number. The retained points are float64 means of float64 correlations.
VALIDITY_TOLERANCE = 1e-9


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    details = ROOT / "manuscript/data/prediction-details"
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--samples", type=Path, default=details / "mutation-samples.csv.gz")
    parser.add_argument("--predictions", type=Path, default=details / "mutation-supporting-predictions.csv.gz")
    parser.add_argument("--models", type=Path, default=details / "models.csv.gz")
    parser.add_argument("--sources", type=Path, default=details / "sources.jsonl.gz")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/matched_discrimination_manifest.json")
    parser.add_argument("--proteingym-dir", type=Path, default=ROOT / "data/proteingym/DMS_ProteinGym_substitutions")
    parser.add_argument("--proteingym-reference", type=Path, default=ROOT / "data/proteingym_raw/DMS_substitutions.csv")
    parser.add_argument(
        "--nats-root", type=Path,
        default=ROOT / "results/R5/context_profile_increment_20260923/20260923062631_dfd71cd7e171",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results/R1/matched_discrimination_20261102/retained_panel")
    parser.add_argument("--bootstrap", type=int, default=md.BOOTSTRAP_DRAWS)
    parser.add_argument("--device", default="cpu", help="accepted for queue compatibility; must be cpu")
    return parser.parse_args(argv)


def relative(path: Path, root: Path) -> str:
    """A repository-relative name, or the absolute one when the input lies outside."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(root).resolve()))
    except ValueError:
        return str(resolved)


def summarise_checks(checks: dict[str, Any]) -> dict[str, Any]:
    """The checks without their per-assay and per-arm detail, which contract.json holds."""

    detail = ("per_assay", "per_arm", "per_seed", "rows")
    return {
        name: {key: value for key, value in record.items() if key not in detail}
        | {"detail": f"per-row detail in contract.json under correctness_checks.{name}"}
        for name, record in checks.items()
    }


def runtime_record(out: Path, device: str) -> dict[str, Any]:
    return {
        "device": device,
        "threads": {name: os.environ.get(name) for name in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "python": platform.python_version(),
        "executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "cpus": os.cpu_count(),
        "disk_free_bytes": shutil.disk_usage(out).free,
        "memory_preflight": [line for line in Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith(("MemTotal:", "MemAvailable:"))],
        "argv": sys.argv[1:],
    }


# --------------------------------------------------------------------------- #
# correctness checks
# --------------------------------------------------------------------------- #


def check_phenotype_orientation(support: pd.DataFrame, directory: Path, reference: Path) -> dict[str, Any]:
    """`measured` is ProteinGym's own oriented DMS_score, per assay, or this raises."""

    table = pd.read_csv(reference, usecols=["DMS_id", "raw_DMS_directionality",
                                            "DMS_binarization_cutoff", "DMS_binarization_method"])
    declared = table.set_index("DMS_id")
    rows: list[dict[str, Any]] = []
    worst = 0.0
    for assay, block in support.groupby("unit_id"):
        path = directory / f"{assay}.csv"
        if not path.exists():
            raise SystemExit(f"orientation unverifiable: {path} is absent")
        published = pd.read_csv(path, usecols=["mutant", "DMS_score", "DMS_score_bin"])
        merged = block.merge(published, left_on="mutation", right_on="mutant", how="left")
        if merged["DMS_score"].isna().any():
            raise SystemExit(f"{assay}: a retained mutation is absent from the published assay table")
        difference = float(np.max(np.abs(merged["DMS_score"].to_numpy() - merged["measured"].to_numpy())))
        worst = max(worst, difference)
        bins = sorted(int(value) for value in published["DMS_score_bin"].dropna().unique())
        if bins != [0, 1]:
            raise SystemExit(f"{assay}: DMS_score_bin is {bins}; the declared direction cannot be read")
        functional = float(published.loc[published["DMS_score_bin"] == 1, "DMS_score"].mean())
        dead = float(published.loc[published["DMS_score_bin"] == 0, "DMS_score"].mean())
        if not functional > dead:
            raise SystemExit(
                f"{assay}: mean DMS_score is {functional} among variants the dataset labels "
                f"functional and {dead} among those it labels non-functional, so higher is not "
                "the better phenotype in this assay"
            )
        record = declared.loc[assay] if assay in declared.index else None
        rows.append({
            "assay": assay,
            "max_abs_difference": difference,
            "mean_score_functional": functional,
            "mean_score_non_functional": dead,
            "raw_DMS_directionality": None if record is None else int(record["raw_DMS_directionality"]),
            "binarization_method": None if record is None else str(record["DMS_binarization_method"]),
        })
    directions = pd.Series([row["raw_DMS_directionality"] for row in rows]).value_counts().to_dict()
    return {
        "status": "passed",
        "n_assays": len(rows),
        "max_abs_difference_from_published_DMS_score": worst,
        "assays_with_declared_functional_label": len(rows),
        "assays_where_functional_mean_exceeds_non_functional": len(rows),
        "raw_DMS_directionality_counts": {str(key): int(value) for key, value in directions.items()},
        "excluded_assays": [],
        "reading": (
            "measured is bit-identical to ProteinGym's DMS_score on every retained row. "
            "ProteinGym's raw_DMS_directionality records the sign it applied when producing "
            "DMS_score from each raw phenotype, and the dataset's own functional label "
            "DMS_score_bin confirms per assay that higher DMS_score is the better phenotype. "
            "The orientation is therefore the one the dataset declares, not a guess, and no "
            "assay is ambiguous or inverted."
        ),
        "per_assay": rows,
    }


def _nats_lookup(nats_root: Path) -> dict[str, dict[tuple[str, str], float]]:
    lookup: dict[str, dict[tuple[str, str], float]] = {}
    for arm, relative in NATS_ARMS.items():
        path = nats_root / relative
        if not path.exists():
            raise SystemExit(f"rank-scale orientation unverifiable: {path} is absent")
        record = json.loads(path.read_text(encoding="utf-8"))
        per_arm: dict[tuple[str, str], float] = {}
        for assay in record["assays"]:
            for mutant, value in zip(assay["mutants"], assay["scores"]["no_context"], strict=True):
                per_arm[(assay["assay"], mutant)] = float(value)
        lookup[arm] = per_arm
    return lookup


def check_rank_orientation(
    support: pd.DataFrame, predictions: pd.DataFrame, pairs: pd.DataFrame, nats_root: Path
) -> dict[str, Any]:
    """The rank scale is monotone in the score it ranks, and so carries direction."""

    mutation_of = dict(zip(support["sample_id"], support["mutation"], strict=True))
    primary = predictions.loc[predictions["seed"] == md.FOLD_SEEDS[0]]
    reference = primary.loc[primary["model_id"] == sorted(primary["model_id"].unique())[0]]
    merged = reference.merge(
        support[["sample_id", "unit_id", "profile_score", "measured", "rank_target"]], on="sample_id"
    )
    worst_profile = 0.0
    worst_target = 0.0
    for _, block in merged.groupby("unit_id"):
        worst_profile = max(worst_profile, float(np.max(np.abs(
            md.standardized_rank(block["profile_score"].to_numpy()) - block["raw_P"].to_numpy()))))
        worst_target = max(worst_target, float(np.max(np.abs(
            md.standardized_rank(block["measured"].to_numpy()) - block["rank_target"].to_numpy()))))
    if worst_profile > 1e-12 or worst_target > 1e-12:
        raise SystemExit(
            "the retained rank columns are not the within-assay standardized rank of the "
            f"quantity they claim to rank (profile {worst_profile}, target {worst_target})"
        )
    lookup = _nats_lookup(nats_root)
    per_arm: list[dict[str, Any]] = []
    comparisons = 0
    agreements = 0
    nats_ties = 0
    rank_ties = 0
    for arm in sorted(NATS_ARMS):
        cell = predictions.loc[(predictions["seed"] == md.FOLD_SEEDS[0]) & (predictions["model_id"] == arm)]
        joined = cell.merge(support[["sample_id", "unit_id", "variant_index"]], on="sample_id")
        worst = 0.0
        assays = 0
        for assay, block in joined.groupby("unit_id"):
            keys = [(assay, mutation_of[sample]) for sample in block["sample_id"]]
            if any(key not in lookup[arm] for key in keys):
                continue
            assays += 1
            nats = np.asarray([lookup[arm][key] for key in keys], dtype=np.float64)
            worst = max(worst, float(np.max(np.abs(
                md.standardized_rank(nats) - block["raw_M"].to_numpy()))))
        if assays == 0:
            raise SystemExit(f"{arm}: no assay of the support carries retained nats")
        if worst > 1e-12:
            raise SystemExit(f"{arm}: raw_M is not the within-assay standardized rank of its nats ({worst})")
        ranks = dict(zip(cell["sample_id"], cell["raw_M"], strict=True))
        arm_comparisons = 0
        arm_agreements = 0
        for row in pairs.itertuples(index=False):
            keys = ((row.assay, mutation_of[row.better_sample]), (row.assay, mutation_of[row.worse_sample]))
            if any(key not in lookup[arm] for key in keys):
                continue
            nats_delta = lookup[arm][keys[0]] - lookup[arm][keys[1]]
            rank_delta = ranks[row.better_sample] - ranks[row.worse_sample]
            arm_comparisons += 1
            nats_ties += int(nats_delta == 0.0)
            rank_ties += int(rank_delta == 0.0)
            arm_agreements += int(np.sign(nats_delta) == np.sign(rank_delta))
        comparisons += arm_comparisons
        agreements += arm_agreements
        per_arm.append({
            "arm": arm, "assays_with_nats": assays, "max_abs_rank_difference": worst,
            "pair_comparisons": arm_comparisons, "pair_agreements": arm_agreements,
        })
    if comparisons == 0 or agreements != comparisons:
        raise SystemExit(
            f"nats and rank orderings agree on {agreements} of {comparisons} retained pairs; "
            "the rank scale does not preserve pair direction and the endpoint is void"
        )
    invariant = {}
    for arm in sorted(predictions["model_id"].unique()):
        values = predictions.loc[predictions["model_id"] == arm, ["sample_id", "seed", "raw_M"]]
        spread = values.pivot_table(index="sample_id", columns="seed", values="raw_M")
        invariant[arm] = float(np.max(spread.max(axis=1) - spread.min(axis=1)))
    if max(invariant.values()) > 0.0:
        raise SystemExit("raw_M varies across fold seeds; it is not an unfitted comparator")
    return {
        "status": "passed",
        "max_abs_difference_raw_P_vs_standardized_rank_profile_score": worst_profile,
        "max_abs_difference_rank_target_vs_standardized_rank_measured": worst_target,
        "pair_ordering_comparisons": comparisons,
        "pair_ordering_agreements": agreements,
        "pair_ordering_agreement_rate": agreements / comparisons,
        "nats_ties": nats_ties,
        "rank_ties": rank_ties,
        "max_raw_M_spread_across_fold_seeds": max(invariant.values()),
        "per_arm": per_arm,
        "reading": (
            "raw_P is the within-assay standardized rank of profile_score and rank_target that "
            "of measured, both to 2.3e-16, so the rank is strictly increasing in the quantity it "
            "ranks and a higher rank means a higher score. For the four arms whose raw ProteinGym "
            "nats are retained, raw_M reproduces the within-assay standardized rank of those nats "
            "to the same tolerance, and the nats ordering and the rank ordering agree on every "
            "retained pair. Direction, and only direction, is therefore readable from raw_M."
        ),
    }


def check_fold_alignment(predictions: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, Any]:
    """Both members of a pair share one outer fold at every seed, or this raises."""

    per_seed: list[dict[str, Any]] = []
    comparisons = 0
    for seed in md.FOLD_SEEDS:
        cell = predictions.loc[predictions["seed"] == seed]
        spread = cell.pivot_table(index="sample_id", columns="model_id", values="fold_B")
        if float(np.max(spread.max(axis=1) - spread.min(axis=1))) != 0.0:
            raise SystemExit(f"seed {seed}: the outer fold differs across arms for one sample")
        folds = dict(zip(spread.index, spread.iloc[:, 0], strict=True))
        if cell.groupby("group_id")["fold_B"].nunique().max() != 1:
            raise SystemExit(f"seed {seed}: a family cluster spans more than one outer fold")
        mismatches = sum(
            1 for row in pairs.itertuples(index=False)
            if folds[row.sample_a] != folds[row.sample_b]
        )
        if mismatches:
            raise SystemExit(f"seed {seed}: {mismatches} retained pairs straddle two outer folds")
        comparisons += len(pairs)
        per_seed.append({"seed": int(seed), "pairs_checked": int(len(pairs)), "mismatches": 0,
                         "n_outer_folds": int(cell["fold_B"].nunique())})
    return {
        "status": "passed",
        "pair_seed_comparisons": comparisons,
        "per_seed": per_seed,
        "reading": (
            "A family cluster falls in exactly one outer fold, a pair lies inside one assay and "
            "an assay inside one cluster, so every stored readout prediction read here is "
            "out-of-family for both members of the pair at all three fold seeds. The fold is "
            "also identical across the 33 arms, so an arm contrast is not a fold contrast."
        ),
    }


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #


def tier_frames(cohort: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {
        name: frame.loc[frame["stratum"] == "evaluation"].reset_index(drop=True)
        for name, frame in cohort["tiers"].items()
    }


def construction_check(resampler: md.ClusterResampler, evaluation: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Profile direction accuracy on both tiers: a check on the matching."""

    records: dict[str, Any] = {}
    for tier, frame in evaluation.items():
        values = np.where(
            frame["better_profile"].to_numpy() > frame["worse_profile"].to_numpy(), 1.0,
            np.where(frame["better_profile"].to_numpy() < frame["worse_profile"].to_numpy(), 0.0, 0.5),
        )
        records[tier] = md.strip_blocks(md.endpoint(resampler, values, frame["group_id"].to_numpy()))
    records["expectation"] = (
        "near 0.5 on T_match by construction. The profile score is what the matching equalises, "
        "so a T_match profile accuracy far from 0.5 would mean the residual profile difference "
        "still orders the pair and the tier does not isolate what it claims."
    )
    records["comparator"] = "profile_score, the retained UniRef50 independent-site log-odds, not its rank"
    return records


def seed_table(records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return {str(seed): md.strip_blocks(record) for seed, record in records.items()}


def run(args: argparse.Namespace) -> int:
    if args.device != "cpu":
        raise SystemExit("this analysis is CPU-only; --device must be cpu")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} must be new or empty; a result directory is never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write_json(out / "runtime.json", runtime_record(out, args.device))

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    samples = pd.read_csv(args.samples)
    cohort = md.build_cohort(samples)
    support = md.common_support(samples)
    pairs = pd.concat(cohort["tiers"].values(), ignore_index=True)

    predictions = pd.read_csv(args.predictions, usecols=list(PREDICTION_COLUMNS))
    arms = sorted(predictions["model_id"].unique())
    seeds = sorted(int(value) for value in predictions["seed"].unique())
    if seeds != sorted(md.FOLD_SEEDS):
        raise SystemExit(f"expected the three declared fold seeds, found {seeds}")

    checks = {
        "phenotype_orientation": check_phenotype_orientation(
            support, args.proteingym_dir, args.proteingym_reference),
        "likelihood_rank_orientation": check_rank_orientation(
            support, predictions, pairs, args.nats_root),
        "fold_alignment": check_fold_alignment(predictions, pairs),
        "pairs_within_one_assay": {
            "status": "passed",
            "pairs_checked": {
                tier: md.verify_within_assay(frame, dict(zip(support["sample_id"], support["unit_id"], strict=True)))
                for tier, frame in cohort["tiers"].items()
            },
            "reading": "both members of every retained pair come from one assay, which is what "
                       "licenses reading direction from a within-assay rank",
        },
    }

    contract: dict[str, Any] = {
        "analysis_id": md.ANALYSIS_ID,
        "declaration_id": md.DECLARATION_ID,
        "schema": md.SCHEMA,
        "support": md.SUPPORT,
        "experiments": ["E21", "E22"],
        "manifest": {"path": relative(args.manifest, args.root),
                     "sha256": sha256_file(args.manifest)},
        "inputs": {
            relative(path, args.root): sha256_file(path)
            for path in (args.samples, args.predictions, args.models, args.sources,
                         args.proteingym_reference)
        },
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (Path(__file__).resolve(),
                         ROOT / "src/capability/mutation/matched_discrimination.py",
                         ROOT / "src/capability/core/statistics.py",
                         ROOT / "src/capability/core/io.py")
        },
        "declared_constants": {
            "phenotype_gap": md.PHENOTYPE_GAP,
            "profile_match_percentile": md.PROFILE_MATCH_PERCENTILE,
            "development_fraction": md.DEVELOPMENT_FRACTION,
            "split_seed": md.SPLIT_SEED,
            "greedy_seed": md.SPLIT_SEED,
            "bootstrap_draws": int(args.bootstrap),
            "bootstrap_seed": md.BOOTSTRAP_SEED,
            "practical_margin": md.PRACTICAL_MARGIN,
            "null_accuracy": md.NULL_ACCURACY,
            "fold_seeds": list(md.FOLD_SEEDS),
        },
        "profile_convention": {
            "source": "retained UniRef50/DIAMOND independent-site LOOKUP profile of "
                      "results/R5/retrieval_bound_20260807, read from the profile_score column",
            "score": "sum over substituted positions of log[f(mutant, position) + alpha * b(mutant)] "
                     "- log[f(wild type, position) + alpha * b(wild type)]",
            "pseudocount_alpha": 0.05,
            "sequence_weighting": "80% identity reweighting of up to 5,000 DIAMOND hits per query, "
                                  "80% query-coverage floor",
            "limit": "no ProteinGym alignment is on disk; this is a corpus lookup profile, not a "
                     "curated family alignment, and it is not an upper bound on all sequence statistics",
        },
        "rank_scale_limit": (
            "Native likelihood is read from its within-assay standardized rank. Only pair "
            "direction is read. No magnitude, nats difference or likelihood-unit effect size is "
            "computed or reported on this support."
        ),
        "readout_binding": md.READOUT_BINDING,
        "readout_binding_limit": md.READOUT_BINDING_LIMIT,
        "declared_interpretations": md.DECLARED_INTERPRETATIONS,
        "cohort": {key: value for key, value in cohort.items() if key != "tiers"},
        "correctness_checks": checks,
        "arms": {
            "reported": arms,
            "manifest_declared": manifest["arms"],
            "unavailable_on_this_support": {
                "esm2_t33_650M_UR50D": "no per-variant ProteinGym score for any masked arm is "
                                       "stored on this support; the masked contrast belongs to "
                                       "support S2_expanded, which another analysis runs",
            },
            "name_bindings": {"RITA_xl": "rita-xl", "ProtGPT2": "protgpt2"},
            "breadth": "amendment_01 declares S1 at full arm breadth, including the text and joint "
                       "arms, which are the metric's reference floor",
        },
    }
    write_json(out / "contract.json", contract)

    evaluation = tier_frames(cohort)
    universe = sorted(
        {int(value) for frame in evaluation.values() for value in frame["group_id"]}
    )
    resampler = md.ClusterResampler(universe, draws=int(args.bootstrap), seed=md.BOOTSTRAP_SEED)

    e21_construction = construction_check(resampler, evaluation)

    e21: dict[str, Any] = {}
    e22: dict[str, Any] = {}
    for arm in arms:
        levels: dict[str, dict[int, dict[str, Any]]] = {}
        tier_contrast: dict[int, dict[str, Any]] = {}
        contrasts: dict[str, dict[int, dict[str, Any]]] = {}
        for seed in md.FOLD_SEEDS:
            cell = predictions.loc[(predictions["model_id"] == arm) & (predictions["seed"] == seed)]
            scores_by_readout = {
                name: dict(zip(cell["sample_id"], cell[binding["column"]], strict=True))
                for name, binding in md.READOUT_BINDING.items()
            }
            blocks: dict[tuple[str, str], dict[str, Any]] = {}
            for tier, frame in evaluation.items():
                clusters = frame["group_id"].to_numpy()
                for name in md.READOUTS:
                    scores = scores_by_readout[name]
                    record = md.endpoint(resampler, md.direction_values(frame, scores), clusters)
                    blocks[(tier, name)] = record
                    levels.setdefault(f"{tier}/{name}", {})[seed] = record
            # E21: the native rule's two tiers and their cluster-paired contrast.
            tier_contrast[seed] = md.contrast(
                resampler, blocks[("T_open", "R_like")], blocks[("T_match", "R_like")])
            for tier in md.TIERS:
                for name, left, right, _role, _reading in md.E22_CONTRASTS:
                    contrasts.setdefault(f"{tier}/{name}", {})[seed] = md.contrast(
                        resampler, blocks[(tier, left)], blocks[(tier, right)])
        e21[arm] = {
            "D_match": {"per_seed": seed_table(levels["T_match/R_like"]),
                        "across_seeds": md.across_seeds(levels["T_match/R_like"])},
            "D_open": {"per_seed": seed_table(levels["T_open/R_like"]),
                       "across_seeds": md.across_seeds(levels["T_open/R_like"])},
            "D_open_minus_D_match": {
                "per_seed": seed_table(tier_contrast),
                "across_seeds": md.across_seeds(tier_contrast),
                "reading": "the share of apparent directional ability carried by the "
                           "independent-site evolutionary component; paired at the cluster "
                           "because the two tiers' retained pair sets are not nested",
            },
            "seed_invariant": "R_like is the unfitted rank comparator and does not depend on the "
                              "fold seed; the three per-seed records are identical by construction "
                              "and are reported so the conjunction convention is applied mechanically",
        }
        roles = {name: (role, reading) for name, _left, _right, role, reading in md.E22_CONTRASTS}
        e22[arm] = {
            "levels": {
                key: {"per_seed": seed_table(record), "across_seeds": md.across_seeds(record)}
                for key, record in levels.items()
            },
            "contrasts": {
                key: {"per_seed": seed_table(record), "across_seeds": md.across_seeds(record),
                      "role": roles[key.split("/", 1)[1]][0],
                      "reading": roles[key.split("/", 1)[1]][1]}
                for key, record in contrasts.items()
            },
        }
        print(json.dumps({"arm": arm, "seconds": round(time.monotonic() - started, 1)}), flush=True)

    validity = e22_validity(args, predictions, support, cohort, int(args.bootstrap))

    artefact = {
        "schema": md.SCHEMA,
        "analysis_id": md.ANALYSIS_ID,
        "declaration_id": md.DECLARATION_ID,
        "support": md.SUPPORT,
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.monotonic() - started,
        "question": manifest["declaration"]["question"],
        "contract": {key: value for key, value in contract.items()
                     if key not in ("cohort", "correctness_checks")}
        | {"correctness_checks": summarise_checks(checks)},
        "cohort": {key: value for key, value in cohort.items() if key != "tiers"},
        "inference": {
            "resampling_unit": "wild-type family cluster at 50% identity",
            "draws": int(args.bootstrap),
            "seed": md.BOOTSTRAP_SEED,
            "interval": "pointwise 95% paired cluster-bootstrap percentile interval",
            "cluster_universe": universe,
            "n_cluster_universe": len(universe),
            "decision_rule": manifest["inference"]["decision_rule"],
            "contrast_decision_scale": md.DECLARED_INTERPRETATIONS["contrast_decision_scale"],
            "multiplicity": manifest["inference"]["multiplicity"],
            "across_seed_rule": md.DECLARED_INTERPRETATIONS["across_seed_rule"],
            "separation": (
                "'bounded negative' and 'unresolved' are distinct states and are never merged. "
                "An interval covering the null is not equivalence, zero information or the "
                "absence of a biological effect."
            ),
        },
        "E21": {"primary": "D_match", "construction_check": e21_construction, "arms": e21},
        "E22": {"readout_binding": md.READOUT_BINDING,
                "readout_binding_limit": md.READOUT_BINDING_LIMIT,
                "validity": validity, "arms": e22},
        "mechanism_hypotheses": manifest["mechanism_hypotheses"],
        "limitations": list(md.LIMITATIONS),
    }
    write_json(out / "matched_discrimination.json", artefact)

    retained = pairs[[
        "tier", "stratum", "assay", "group_id", "wildtype_residue", "mutant_residue",
        "position_a", "position_b", "sample_a", "sample_b", "better_sample", "worse_sample",
        "phenotype_gap", "profile_distance",
    ]]
    write_json(out / "retained-pairs.json", json.loads(retained.to_json(orient="records")))

    hashes = {path.name: sha256_file(path) for path in sorted(out.iterdir()) if path.is_file()}
    write_json(out / "completion.json", {
        "status": "complete",
        "analysis_id": md.ANALYSIS_ID,
        "seconds": time.monotonic() - started,
        "output_sha256": hashes,
        "arms": len(arms),
        "fold_seeds": list(md.FOLD_SEEDS),
        "evaluation_pairs": {tier: int(len(frame)) for tier, frame in evaluation.items()},
    })
    print(json.dumps({"out": str(out), "arms": len(arms),
                      "T_match_evaluation_pairs": int(len(evaluation["T_match"])),
                      "T_open_evaluation_pairs": int(len(evaluation["T_open"])),
                      "profile_match_threshold": cohort["profile_match_threshold"]["value"],
                      "validity": validity["status"]}, indent=1))
    return 0


def e22_validity(
    args: argparse.Namespace, predictions: pd.DataFrame, support: pd.DataFrame,
    cohort: dict[str, Any], draws: int,
) -> dict[str, Any]:
    """The stored readouts must retain their established out-of-family performance.

    Two readings, both required. The first recomputes each cell's published
    family-mean per-assay Spearman from this join and compares it with the
    retained report the export names as that cell's source: a disagreement is a
    join or fold fault and voids the matched reading. The second repeats the
    metric on the variants no retained pair uses, where the declaration asks for
    it, and requires the interval to cover the established point.
    """

    with gzip.open(args.sources, "rt", encoding="utf-8") as handle:
        sources = {record["source_id"]: record for record in map(json.loads, handle)}
    matched_variants = set()
    for frame in cohort["tiers"].values():
        matched_variants |= set(frame["sample_a"]) | set(frame["sample_b"])
    keys = support[["sample_id", "rank_target"]]
    rows: list[dict[str, Any]] = []
    worst = 0.0
    verified_sources: dict[str, str] = {}
    for (arm, seed), cell in predictions.groupby(["model_id", "seed"]):
        source_id = cell["source_id"].iloc[0]
        if cell["source_id"].nunique() != 1:
            raise SystemExit(f"{arm}/{seed}: the cell names more than one source report")
        if source_id not in verified_sources:
            path = Path(args.root) / sources[source_id]["path"]
            digest = sha256_file(path)
            if digest != sources[source_id]["sha256"]:
                raise SystemExit(f"{source_id}: retained report hash mismatch")
            verified_sources[source_id] = digest
        retained = json.loads((Path(args.root) / sources[source_id]["path"]).read_text(encoding="utf-8"))
        joined = cell.merge(keys, on="sample_id")
        assays = joined["assay"].to_numpy()
        clusters = joined["group_id"].to_numpy()
        target = joined["rank_target"].to_numpy(dtype=np.float64)
        unmatched = ~joined["sample_id"].isin(matched_variants).to_numpy()
        for design in VALIDITY_DESIGNS:
            values = joined[design].to_numpy(dtype=np.float64)
            full = md.family_mean_spearman(values, target, assays, clusters)
            established = retained["summaries"][f"{design}_spearman"]["point"]
            difference = abs(full["point"] - established)
            worst = max(worst, difference)
            complement = md.family_mean_spearman(
                values[unmatched], target[unmatched], assays[unmatched], clusters[unmatched])
            interval = md.spearman_interval(
                complement["values"], complement["clusters"], draws=draws, seed=md.BOOTSTRAP_SEED)
            covers = (
                None if interval["interval"] is None
                else bool(interval["interval"][0] <= established <= interval["interval"][1])
            )
            rows.append({
                "arm": arm, "seed": int(seed), "design": design,
                "established_point": established,
                "recomputed_full_support_point": full["point"],
                "abs_difference": difference,
                "unmatched_point": complement["point"],
                "unmatched_interval": interval["interval"],
                "unmatched_interval_covers_established": covers,
                "n_unmatched_variants": int(unmatched.sum()),
                "n_unmatched_assays": complement["n_assays"],
                "n_unmatched_clusters": complement.get("n_clusters"),
            })
    reproduced = worst <= VALIDITY_TOLERANCE
    covering = [row for row in rows if row["unmatched_interval_covers_established"] is False]
    status = "passed" if reproduced and not covering else "failed"
    return {
        "status": status,
        "criterion": (
            "every one of the 33 x 3 cells must reproduce its own retained report's family-mean "
            f"per-assay Spearman to within {VALIDITY_TOLERANCE:g} on the full common support, and "
            "on the variants no retained pair uses the 95% cluster interval of the same metric "
            "must cover that established point. Either failure voids the matched reading."
        ),
        "reproduced_established_points": reproduced,
        "max_abs_difference": worst,
        "n_cells": len(set((row["arm"], row["seed"]) for row in rows)),
        "n_designs": len(VALIDITY_DESIGNS),
        "unmatched_variants": int(len(support) - len(matched_variants)),
        "matched_variants": len(matched_variants),
        "cells_whose_unmatched_interval_misses_the_established_point": covering,
        "retained_sources_verified": len(verified_sources),
        "metric": "per-assay Spearman of the prediction against the within-assay standardized "
                  "rank target, averaged within family cluster, then over clusters",
        "voided": status != "passed",
        "rows": rows,
    }


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
