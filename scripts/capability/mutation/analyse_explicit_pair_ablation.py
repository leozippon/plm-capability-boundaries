#!/usr/bin/env python3
"""Pair-level bidirectional ablation against Profile, Potts and GEMME: CPU, no scoring.

Executes `docs/explicit-pair-ablation.md` on declared support S1_retained_panel.
Reads the retained 33-arm panel's stored per-variant likelihood ranks and the R1
statistical ladder's Profile, Potts and GEMME scores, builds the declared matched
and exactly balanced tiers in both directions, and reports direction accuracy
with the shared bidirectional simultaneous band and the declared verdicts.

Nothing here loads a checkpoint, fits a readout or writes into an existing result
directory. Four refusals run before any endpoint:

* the two upstream baseline documents are admitted only as complete official runs
  through the statistical ladder's own contract, and each must cover the support
  exactly, with losses enumerated by wild-type reference;
* the panel's stored profile rank column must reproduce the within-assay
  standardized rank of `profile_score`, which is the identity that licenses
  reading the three statistics on one scale;
* every arm must carry a finite likelihood on every support row, and that
  likelihood must not vary across the three fold seeds, or it is not the unfitted
  comparator the reading claims;
* both members of every retained pair must sit in one assay, which is what
  licenses reading direction from a within-assay rank.

`--device` is accepted because the campaign queue injects it; the analysis is
CPU-only and refuses a non-CPU device.
"""
from __future__ import annotations

import argparse
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
from src.capability.mutation import explicit_pair_ablation as epa  # noqa: E402
from src.capability.mutation import matched_discrimination as md  # noqa: E402

PREDICTION_COLUMNS = ("sample_id", "model_id", "seed", "raw_M", "raw_P")

#: The nine arms E21's freshly scored cohorts declare, so the coverage statement
#: names what this support cannot reach rather than leaving it implicit.
DECLARED_S2_ARMS = (
    "progen2-small", "progen2-medium", "progen2-large", "progen2-xlarge",
    "progen3-112m", "progen3-3b", "rita-xl", "protgpt2", "esm2-650m",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    details = ROOT / "manuscript/data/prediction-details"
    ladder = ROOT / "results/extensions/statistical_ladder_20261106/accepted-scores"
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--samples", type=Path, default=details / "mutation-samples.csv.gz")
    parser.add_argument(
        "--predictions", type=Path, default=details / "mutation-supporting-predictions.csv.gz"
    )
    parser.add_argument("--potts", type=Path, default=ladder / "potts/scores.json")
    parser.add_argument("--gemme", type=Path, default=ladder / "gemme/scores.json")
    parser.add_argument(
        "--published",
        type=Path,
        default=ROOT / "results/R1/matched_discrimination_20261102/retained_panel"
                      "/matched_discrimination.json",
        help="the published profile-only tier, quoted as the external anchor",
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/R1/explicit_pair_ablation_20261114"
    )
    parser.add_argument("--bootstrap", type=int, default=epa.BOOTSTRAP_DRAWS)
    parser.add_argument("--device", default="cpu", help="accepted for queue compatibility; must be cpu")
    return parser.parse_args(argv)


def relative(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(root).resolve()))
    except ValueError:
        return str(resolved)


def runtime_record(out: Path, device: str) -> dict[str, Any]:
    return {
        "device": device,
        "started_utc": datetime.now(timezone.utc).isoformat(),
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
# refusals
# --------------------------------------------------------------------------- #


def check_profile_scale(support: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    """The panel's stored `raw_P` is the within-assay standardized rank of the profile."""

    seed = sorted(int(value) for value in predictions["seed"].unique())[0]
    arm = sorted(predictions["model_id"].unique())[0]
    cell = predictions.loc[(predictions["seed"] == seed) & (predictions["model_id"] == arm)]
    merged = cell.merge(support[["sample_id", "unit_id", "rank_profile"]], on="sample_id")
    if len(merged) != len(support):
        raise SystemExit("the prediction table does not cover the support row for row")
    worst = float(np.max(np.abs(merged["rank_profile"].to_numpy() - merged["raw_P"].to_numpy())))
    if worst > 1e-9:
        raise SystemExit(
            f"the computed profile rank differs from the panel's stored raw_P by {worst}; the "
            "three statistics are not on the scale the declaration names"
        )
    return {
        "status": "passed",
        "max_abs_difference_rank_profile_vs_raw_P": worst,
        "checked_on": {"arm": arm, "seed": seed, "rows": int(len(merged))},
        "reading": (
            "the profile's within-assay standardized rank reproduces the panel's own raw_P column, "
            "so Potts and GEMME put on the same scale are on the scale the panel's raw_M already "
            "carries and one tolerance governs all four quantities"
        ),
    }


def check_likelihood(support: pd.DataFrame, predictions: pd.DataFrame) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Every arm carries a finite, seed-invariant likelihood rank on every support row."""

    identifiers = list(support["sample_id"])
    expected = set(identifiers)
    seeds = sorted(int(value) for value in predictions["seed"].unique())
    arms = sorted(predictions["model_id"].unique())
    spreads: dict[str, float] = {}
    ranks: dict[str, dict[str, float]] = {}
    for arm in arms:
        block = predictions.loc[predictions["model_id"] == arm]
        per_seed = {}
        for seed in seeds:
            cell = block.loc[block["seed"] == seed]
            offered = list(cell["sample_id"])
            if len(set(offered)) != len(offered):
                raise SystemExit(f"{arm} at seed {seed} carries duplicate sample_id values")
            uncovered = sorted(expected - set(offered))
            if uncovered:
                raise SystemExit(
                    f"{arm} at seed {seed} does not cover {len(uncovered)} support rows, first "
                    f"{uncovered[0]!r}; no likelihood is imputed"
                )
            values = cell.set_index("sample_id")["raw_M"].reindex(identifiers).to_numpy(np.float64)
            if not np.isfinite(values).all():
                raise SystemExit(f"{arm} at seed {seed} carries a non-finite likelihood rank")
            per_seed[seed] = values
        stacked = np.stack([per_seed[seed] for seed in seeds])
        spread = float(np.max(stacked.max(axis=0) - stacked.min(axis=0)))
        if spread > 0.0:
            raise SystemExit(
                f"{arm}'s likelihood rank varies by {spread} across the fold seeds; it is not the "
                "unfitted comparator this reading claims"
            )
        spreads[arm] = spread
        ranks[arm] = dict(zip(identifiers, per_seed[seeds[0]], strict=True))
    record = {
        "status": "passed",
        "n_arms": len(arms),
        "arms": arms,
        "seeds": seeds,
        "rows_per_arm_and_seed": len(identifiers),
        "max_rank_spread_across_fold_seeds": max(spreads.values()),
        "reading": (
            "the stored likelihood rank is identical at all three fold seeds for every arm, so it "
            "is the unfitted within-assay rank comparator and one read of it serves the reading"
        ),
    }
    return record, ranks


def arm_coverage(arms: list[str]) -> dict[str, Any]:
    """Which declared arms this support reaches, and why it does not reach the rest."""

    return {
        "covered": arms,
        "n_covered": len(arms),
        "panel": "the retained 33-arm mutation panel; every arm with a stored per-variant likelihood",
        "declared_elsewhere_not_covered": {
            arm: (
                "no per-variant masked score is stored on this support; the masked scoring rule "
                "belongs to E21's freshly scored cohort S2_expanded, which carries no Potts or "
                "GEMME score for its variants"
            )
            for arm in DECLARED_S2_ARMS
            if arm not in arms
        },
        "manuscript_eight": {
            arm: arm in arms
            for arm in DECLARED_S2_ARMS
            if arm != "esm2-650m"
        },
        "reason_this_support": (
            "Potts and GEMME exist only on the frozen 25,728-variant R1 common support. E21's "
            "expanded cohort scored 23,416 variant sequences chosen from the full ProteinGym "
            "single-substitution space, for which no Potts fit and no GEMME run exists, so its "
            "3,352-pair doubly balanced tier cannot be given a Potts- or GEMME-controlled tier "
            "without new inference"
        ),
    }


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #


def strip(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "pairs"}


def construction_check(frame: pd.DataFrame, resampler: md.ClusterResampler) -> dict[str, Any]:
    """Each statistic's own direction accuracy on a forward tier.

    Near 0.5 is what a matched tier claims and exactly 0.5 is what a balanced
    tier claims. The reference rules sit outside every verdict family, so their
    reading is the pointwise interval, exactly as E21 reports its own.
    """

    return epa.family_bands(
        {name: frame[f"{name}_direction"].to_numpy(np.float64) for name in epa.STATISTICS},
        frame["group_id"].to_numpy(),
        resampler=resampler,
        verdict_from="pointwise",
        role="construction_check",
    )


def published_anchor(path: Path) -> dict[str, Any]:
    """The published profile-only tier, quoted rather than recomputed."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    arms = document["E21"]["arms"]
    return {
        "read_from": str(path),
        "tier": "T_match of matched_discrimination_20261102, evaluation stratum",
        "stratum_note": (
            "the published tier is read on E21's evaluation clusters because its threshold was "
            "fixed on development clusters; this reading applies no split, so the two pair sets "
            "differ and the numbers are an anchor rather than a matched comparison"
        ),
        "points": {
            arm: record["D_match"]["across_seeds"]["points"]["20260923"]
            for arm, record in arms.items()
        },
        "verdicts": {
            arm: record["D_match"]["across_seeds"]["verdict"] for arm, record in arms.items()
        },
    }


def run(args: argparse.Namespace) -> int:
    if args.device != "cpu":
        raise SystemExit("this analysis is CPU-only; --device must be cpu")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} must be new or empty; a result directory is never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write_json(out / "runtime.json", runtime_record(out, args.device))

    samples = pd.read_csv(args.samples)
    support = md.common_support(samples)
    baselines = {
        name: epa.read_baseline(name, path)
        for name, path in (("potts", args.potts), ("gemme", args.gemme))
    }
    offered = support
    support, refusals = epa.attach_statistics(
        support, {name: record["table"] for name, record in baselines.items()}
    )

    predictions = pd.read_csv(args.predictions, usecols=list(PREDICTION_COLUMNS))
    checks: dict[str, Any] = {"profile_scale": check_profile_scale(support, predictions)}
    likelihood, ranks = check_likelihood(support, predictions)
    checks["likelihood_rank"] = likelihood
    arms = likelihood["arms"]

    singles = md.single_substitutions(support)
    candidates = md.candidate_pairs(singles)
    gap_satisfying = candidates.loc[candidates["phenotype_gap"] >= md.PHENOTYPE_GAP]
    cohort = epa.orient(gap_satisfying)
    for name in epa.STATISTICS:
        cohort = epa.attach_covariate(cohort, name, epa.rank_lookup(support, name))
    assay_of = dict(zip(support["sample_id"], support["unit_id"], strict=True))
    checks["pairs_within_one_assay"] = {
        "status": "passed",
        "pairs_checked": md.verify_within_assay(cohort, assay_of),
        "reading": "both members of every candidate pair come from one assay, which is what "
                   "licenses reading direction from a within-assay rank",
    }

    contract: dict[str, Any] = {
        "schema": epa.SCHEMA,
        "analysis_id": epa.ANALYSIS_ID,
        "declaration_id": epa.DECLARATION_ID,
        "support": epa.SUPPORT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "declared_constants": epa.declared_constants(),
        "inputs": {
            relative(path, args.root): sha256_file(path)
            for path in (args.samples, args.predictions, args.potts, args.gemme, args.published)
        },
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "src/capability/mutation/explicit_pair_ablation.py",
                ROOT / "src/capability/mutation/matched_discrimination.py",
                ROOT / "src/capability/extensions/bidirectional.py",
                ROOT / "src/capability/extensions/statistical_ladder.py",
                ROOT / "src/capability/core/statistics.py",
                ROOT / "docs/explicit-pair-ablation.md",
            )
        },
        "baseline_admission": {
            name: {
                "admission": record["admission"],
                "read_from": record["table"]["path"],
                "sha256": record["table"]["sha256"],
                "metadata_sha256": record["table"]["metadata_sha256"],
            }
            for name, record in baselines.items()
        },
        "refusals": refusals,
        "support_counts": {
            "variants_offered": int(len(offered)),
            "assays_offered": int(offered["unit_id"].nunique()),
            "clusters_offered": int(offered["group_id"].nunique()),
            "variants": int(len(support)),
            "assays": int(support["unit_id"].nunique()),
            "clusters": int(support["group_id"].nunique()),
            "references": int(support["protein_id"].nunique()),
            "single_substitution_variants": int(len(singles)),
        },
        "rank_scale": {name: epa.rank_distribution(support, name) for name in epa.STATISTICS},
        "candidates": {
            "pair_rule_pairs": int(len(candidates)),
            "phenotype_gap_pairs": int(len(cohort)),
            "phenotype_gap_clusters": int(cohort["group_id"].nunique()),
            "achieved_distances": epa.distance_summary(cohort, epa.STATISTICS),
        },
        "arms": arm_coverage(arms),
        "correctness_checks": checks,
        "limitations": list(epa.LIMITATIONS),
    }
    write_json(out / "contract.json", contract)

    # ---------------------------------------------------------------- forward
    forward = epa.forward_tiers(cohort)
    forward_record: dict[str, Any] = {}
    for label, tier in forward.items():
        frame = tier["pairs"]
        record = strip(tier)
        if len(frame) == 0:
            record["construction_check"] = {"status": "no_pairs"}
            record["family"] = {"status": "not_estimable", "reason": "the tier holds no pair"}
        else:
            clusters = frame["group_id"].to_numpy()
            resampler = md.ClusterResampler(
                sorted({int(value) for value in clusters}),
                draws=args.bootstrap,
                seed=epa.BOOTSTRAP_SEED,
            )
            record["construction_check"] = construction_check(frame, resampler)
            record["family"] = epa.family_bands(
                {arm: md.direction_values(frame, ranks[arm]) for arm in arms},
                clusters,
                resampler=resampler,
            )
        forward_record[label] = record
    write_json(out / "forward.json", {
        "schema": epa.SCHEMA, "analysis_id": epa.ANALYSIS_ID, "direction": "forward",
        "estimand": "each arm's direction accuracy on pairs the explicit statistics cannot order",
        "family": "one family per tier, 33 arms wide; tiers are never pooled",
        "published_anchor": published_anchor(args.published),
        "tiers": forward_record,
    })

    # ---------------------------------------------------------------- reverse
    open_tier = forward["T_open"]["pairs"]
    reverse_record: dict[str, Any] = {}
    for arm in arms:
        with_model = epa.attach_covariate(cohort, epa.MODEL, ranks[arm])
        arm_open = epa.attach_covariate(open_tier, epa.MODEL, ranks[arm])
        tiers = epa.reverse_tiers(with_model, arm_open)
        per_arm: dict[str, Any] = {}
        for label, tier in tiers.items():
            frame = tier["pairs"]
            record = strip(tier)
            if len(frame) == 0:
                record["family"] = {"status": "not_estimable", "reason": "the tier holds no pair"}
                record["model_construction_check"] = {"status": "no_pairs"}
            else:
                clusters = frame["group_id"].to_numpy()
                # One resampler per tier, shared by the family and its construction
                # check, so the statistics and the controlled model are read on
                # identical draws.
                resampler = md.ClusterResampler(
                    sorted({int(value) for value in clusters}),
                    draws=args.bootstrap,
                    seed=epa.BOOTSTRAP_SEED,
                )
                record["family"] = epa.family_bands(
                    {name: frame[f"{name}_direction"].to_numpy(np.float64)
                     for name in epa.STATISTICS},
                    clusters,
                    resampler=resampler,
                )
                record["model_construction_check"] = epa.family_bands(
                    {epa.MODEL: frame[f"{epa.MODEL}_direction"].to_numpy(np.float64)},
                    clusters,
                    resampler=resampler,
                    verdict_from="pointwise",
                    role="construction_check",
                )
            per_arm[label] = record
        reverse_record[arm] = per_arm
    write_json(out / "reverse.json", {
        "schema": epa.SCHEMA, "analysis_id": epa.ANALYSIS_ID, "direction": "reverse",
        "estimand": "each statistic's direction accuracy on pairs the arm's likelihood cannot order",
        "family": "one family per tier and arm, three statistics wide; the pair set is arm-specific",
        "arms": reverse_record,
    })

    write_json(out / "completion.json", {
        "schema": epa.SCHEMA, "analysis_id": epa.ANALYSIS_ID, "status": "complete",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.monotonic() - started,
        "forward_tiers": sorted(forward_record),
        "reverse_tiers": sorted(next(iter(reverse_record.values()))) if reverse_record else [],
        "arms": len(arms),
        "files": {
            name: sha256_file(out / name)
            for name in ("contract.json", "forward.json", "reverse.json")
        },
    })
    print(f"wrote {out} in {time.monotonic() - started:.1f}s over {len(arms)} arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
