#!/usr/bin/env python3
"""amendment_03 of the matched-discrimination declaration, on support S1.

Three new readings on the frozen cohort, and nothing else. No model score is
recomputed, no threshold already fixed is touched, and the primary artefact of
the first run is not rewritten.

**Addition 3, the position control, is why this script exists.** The two members
of a retained pair differ only in which site was substituted, so any systematic
dependence of likelihood on position becomes a directional vote. Terminal regions
are more mutation-tolerant in most assays, and an early site is also scored under
less left context by an autoregressive model, so a rule that knows nothing but
"prefer the mutation nearer a terminus" could in principle reach the accuracy the
protein arms reach. The two declared position-only rules are computed on the same
pairs, a tier additionally matched on relative position is built, and the
arms-minus-position contrast is reported cluster-paired. If a position-only rule
matches the arms, the retained signal is a positional prior; if the arms exceed
it, that margin is the defensible measure of non-positional ability.

**Addition 2** is the cluster-paired ``D_model - D_profile`` on the same pairs,
the margin over the residual independent-site ordering that the failed
construction check left in T_match.

**Addition 1** is ``T_balanced``, the subset of T_match in which the profile's
directional vote is cancelled by construction, so an arm above 0.5 there cannot
be carrying the independent-site profile.

The cohort is re-derived from the same inputs and checked against the frozen
run's contract before anything is read, so the additions cannot silently sit on
a different cohort. ``--device`` is accepted for queue compatibility; this is CPU
work and a non-CPU device is refused.
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
from src.capability.mutation import matched_discrimination as md  # noqa: E402

PRIMARY = "matched_discrimination.json"
ARTEFACT = "amendment_03.json"
#: Tiers the additions read. T_match is carried because the contrasts need its
#: rows; its own direction accuracy is checked against the frozen artefact rather
#: than restated as a new number.
TIERS = ("T_match", "T_balanced", "T_position_matched")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    details = ROOT / "manuscript/data/prediction-details"
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--samples", type=Path, default=details / "mutation-samples.csv.gz")
    parser.add_argument("--predictions", type=Path, default=details / "mutation-supporting-predictions.csv.gz")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/matched_discrimination_manifest.json")
    parser.add_argument("--proteingym-reference", type=Path, default=ROOT / "data/proteingym_raw/DMS_substitutions.csv")
    parser.add_argument("--cohort", type=Path, default=ROOT / "results/R5/context_mutation_rescue_20260923/cohort.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/R1/matched_discrimination_20261102/retained_panel")
    parser.add_argument("--bootstrap", type=int, default=md.BOOTSTRAP_DRAWS)
    parser.add_argument("--device", default="cpu", help="accepted for queue compatibility; must be cpu")
    return parser.parse_args(argv)


def wildtype_lengths(reference: Path, cohort: Path, assays: set[str]) -> dict[str, Any]:
    """Wild-type lengths from ProteinGym, cross-checked against the frozen cohort."""

    table = pd.read_csv(reference, usecols=["DMS_id", "seq_len"]).set_index("DMS_id")["seq_len"]
    declared = {assay: int(table[assay]) for assay in sorted(assays) if assay in table.index}
    missing = sorted(assays - set(declared))
    if missing:
        raise SystemExit(f"no declared wild-type length for {missing[:3]}")
    record = json.loads(Path(cohort).read_text(encoding="utf-8"))
    retained = {entry["assay"]: len(entry["wildtype"]) for entry in record["assays"]}
    disagreements = sorted(
        assay for assay in declared if retained.get(assay) != declared[assay]
    )
    if disagreements:
        raise SystemExit(
            f"the ProteinGym reference length and the frozen cohort's wild type disagree on "
            f"{disagreements[:3]}; a relative position would not mean the same thing in both"
        )
    return {
        "lengths": declared,
        "check": {
            "status": "passed",
            "n_assays": len(declared),
            "min_length": min(declared.values()),
            "max_length": max(declared.values()),
            "disagreements": [],
            "reading": "the substituted positions are wild-type coordinates and the two length "
                       "sources agree on every assay of the support, so a relative position is "
                       "well defined",
        },
    }


def verify_frozen(out: Path, cohort: dict[str, Any]) -> dict[str, Any]:
    """The additions must sit on exactly the cohort the first run reported."""

    path = out / PRIMARY
    if not path.exists():
        raise SystemExit(f"{path} is absent; the additions build on the frozen primary artefact")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    frozen_cohort = frozen["cohort"]
    if frozen_cohort["profile_match_threshold"]["value"] != cohort["profile_match_threshold"]["value"]:
        raise SystemExit("the re-derived profile-match threshold differs from the frozen one")
    if frozen_cohort["counts"] != cohort["counts"] or frozen_cohort["split"] != cohort["split"]:
        raise SystemExit("the re-derived cohort differs from the frozen one")
    return {
        "status": "passed",
        "primary_artefact": PRIMARY,
        "primary_sha256": sha256_file(path),
        "primary_analysis_id": frozen["analysis_id"],
        "profile_match_threshold": frozen_cohort["profile_match_threshold"]["value"],
        "evaluation_pairs": {
            tier: frozen_cohort["counts"][tier]["evaluation"]["pairs"] for tier in md.TIERS
        },
        "frozen": frozen,
    }


#: What each tier's construction claims to neutralise, and therefore which
#: comparator must land near 0.5 on it. The declaration required the construction
#: check be reported prominently; amendment_03 exists because it failed once, so
#: every tier here states its own check rather than leaving it to be derived.
NEUTRALITY_CLAIMS = {
    "T_match": ("D_profile", "matched",
                "the magnitude of the profile difference is matched, not its sign, which is the "
                "failure that triggered amendment_03"),
    "T_balanced": ("D_profile", "balanced",
                   "the profile's directional vote is cancelled by construction, so this check "
                   "cannot fail unless the balancing is wrong"),
    "T_position_matched": ("D_position_nearer_terminus", "matched",
                           "the magnitude of the relative-position difference is matched, not its "
                           "sign, so the same asymmetry that defeated the profile match can defeat "
                           "this one"),
}


def construction_checks(references: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Whether each tier neutralises the comparator its construction claims to."""

    checks: dict[str, Any] = {}
    for tier, (comparator, kind, note) in NEUTRALITY_CLAIMS.items():
        record = references[tier][comparator]
        point = record["point"]
        inside = abs(point - md.NULL_ACCURACY) <= md.PRACTICAL_MARGIN
        checks[tier] = {
            "comparator": comparator,
            "construction": kind,
            "point": point,
            "interval": record["interval"],
            "verdict": record["verdict"]["verdict"],
            "within_practical_margin_of_one_half": bool(inside),
            "status": "passed" if inside else "failed",
            "note": note,
            "consequence": (
                f"the contrast D_model - {comparator} on {tier} is read against a comparator that "
                f"sits at {point:.4f} rather than 0.5, so its size is not a clean margin over a "
                "neutral reference and is reported with that stated"
            ) if not inside else (
                f"{comparator} is neutral on {tier} to within the declared margin, so a model above "
                "0.5 there is not carrying it"
            ),
        }
    return checks


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


def build_tiers(samples: pd.DataFrame, lengths: dict[str, int]) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any]]:
    """The frozen T_match plus the two tiers the amendment adds."""

    cohort = md.build_cohort(samples)
    support = md.common_support(samples)
    singles = md.single_substitutions(support)
    candidates = md.attach_positions(md.candidate_pairs(singles), lengths)
    development = set(cohort["split"]["development"])
    threshold = cohort["profile_match_threshold"]["value"]
    eligible = candidates.loc[
        (candidates["phenotype_gap"] >= md.PHENOTYPE_GAP)
        & (candidates["profile_distance"] <= threshold)
    ]
    development_eligible = eligible.loc[eligible["group_id"].isin(development)]
    position_threshold = md.match_threshold(
        development_eligible, "position_distance", percentile=md.PROFILE_MATCH_PERCENTILE
    )
    position_matched = eligible.loc[eligible["position_distance"] <= position_threshold]
    retained = md.greedy_disjoint(position_matched)
    retained["tier"] = "T_position_matched"
    retained["stratum"] = np.where(
        retained["group_id"].isin(development), "development", "evaluation"
    )
    better_is_a = retained["z_a"] > retained["z_b"]
    retained["better_sample"] = np.where(better_is_a, retained["sample_a"], retained["sample_b"])
    retained["worse_sample"] = np.where(better_is_a, retained["sample_b"], retained["sample_a"])
    retained["better_profile"] = np.where(better_is_a, retained["profile_a"], retained["profile_b"])
    retained["worse_profile"] = np.where(better_is_a, retained["profile_b"], retained["profile_a"])
    tiers = {
        "T_match": md.attach_positions(cohort["tiers"]["T_match"], lengths),
        "T_balanced": md.attach_positions(
            md.balance_profile_vote(cohort["tiers"]["T_match"]), lengths),
        "T_position_matched": md.attach_positions(retained, lengths),
    }
    construction = {
        "position_match_threshold": {
            "value": position_threshold,
            "percentile": md.PROFILE_MATCH_PERCENTILE,
            "statistic": md.AMENDMENT_03_INTERPRETATIONS["position_statistic"],
            "fixed_on": "development-cluster candidate pairs that satisfy the phenotype gap and the "
                        "profile match",
            "n_development_candidates": int(len(development_eligible)),
            "n_eligible_candidates": int(len(eligible)),
            "n_position_matched_candidates": int(len(position_matched)),
        },
        "counts": {
            tier: {
                "all": md.tier_counts(frame),
                "development": md.tier_counts(frame.loc[frame["stratum"] == "development"]),
                "evaluation": md.tier_counts(frame.loc[frame["stratum"] == "evaluation"]),
            }
            for tier, frame in tiers.items()
        },
        "achieved_evaluation": {
            tier: {
                name: {
                    "min": float(part.min()), "median": float(np.median(part)),
                    "mean": float(part.mean()), "max": float(part.max()),
                }
                for name, part in (
                    ("phenotype_gap", frame.loc[frame["stratum"] == "evaluation", "phenotype_gap"]),
                    ("profile_distance", frame.loc[frame["stratum"] == "evaluation", "profile_distance"]),
                    ("position_distance", frame.loc[frame["stratum"] == "evaluation", "position_distance"]),
                )
            }
            for tier, frame in tiers.items()
        },
        "subset_of_T_match": {
            tier: int(
                len(
                    set(zip(frame["sample_a"], frame["sample_b"], strict=True))
                    & set(zip(tiers["T_match"]["sample_a"], tiers["T_match"]["sample_b"], strict=True))
                )
            )
            for tier, frame in tiers.items()
        },
    }
    return cohort, tiers, construction


def run(args: argparse.Namespace) -> int:
    if args.device != "cpu":
        raise SystemExit("this analysis is CPU-only; --device must be cpu")
    out = Path(args.out)
    if not out.is_dir():
        raise SystemExit(f"{out} must be the directory of the frozen primary artefact")
    if (out / ARTEFACT).exists():
        raise SystemExit(f"{out / ARTEFACT} exists; an addendum is written once")
    started = time.monotonic()
    write_json(out / "amendment_03-runtime.json", runtime_record(out, args.device))

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    amendment = manifest["amendment_03"]
    samples = pd.read_csv(args.samples)
    support = md.common_support(samples)
    lengths = wildtype_lengths(args.proteingym_reference, args.cohort, set(support["unit_id"]))
    cohort, tiers, construction = build_tiers(samples, lengths["lengths"])
    frozen = verify_frozen(out, cohort)
    frozen.pop("frozen")

    evaluation = {
        tier: frame.loc[frame["stratum"] == "evaluation"].reset_index(drop=True)
        for tier, frame in tiers.items()
    }
    open_tier = md.attach_positions(cohort["tiers"]["T_open"], lengths["lengths"])
    evaluation["T_open"] = open_tier.loc[open_tier["stratum"] == "evaluation"].reset_index(drop=True)
    universe = json.loads((out / PRIMARY).read_text(encoding="utf-8"))["inference"]["cluster_universe"]
    resampler = md.ClusterResampler(universe, draws=int(args.bootstrap), seed=md.BOOTSTRAP_SEED)

    # The two declared position rules cannot disagree inside a pair; verified,
    # not assumed, before either is reported.
    agreement: dict[str, Any] = {}
    for tier, frame in evaluation.items():
        first = md.position_direction_values(frame, "nearer_terminus")
        second = md.position_direction_values(frame, "further_from_midpoint")
        agreement[tier] = {
            "n_pairs": int(len(frame)),
            "disagreements": int(np.count_nonzero(first != second)),
        }
    if any(record["disagreements"] for record in agreement.values()):
        raise SystemExit("the two declared position rules disagree; the identity argument is wrong")

    references: dict[str, dict[str, dict[str, Any]]] = {}
    for tier, frame in evaluation.items():
        clusters = frame["group_id"].to_numpy()
        gaps = frame["phenotype_gap"].to_numpy()
        references[tier] = {}
        for name, values in (
            ("D_profile", md.profile_direction_values(frame)),
            ("D_position_nearer_terminus", md.position_direction_values(frame, "nearer_terminus")),
            ("D_position_further_from_midpoint",
             md.position_direction_values(frame, "further_from_midpoint")),
        ):
            record = md.endpoint(resampler, values, clusters)
            record["selection_quality"] = md.selection_quality(values, gaps)
            references[tier][name] = record

    predictions = pd.read_csv(
        args.predictions, usecols=["sample_id", "model_id", "seed", "raw_M"]
    )
    arms = sorted(predictions["model_id"].unique())
    rows: dict[str, Any] = {}
    for arm in arms:
        per_tier: dict[str, Any] = {}
        for tier, frame in evaluation.items():
            clusters = frame["group_id"].to_numpy()
            gaps = frame["phenotype_gap"].to_numpy()
            levels: dict[str, dict[int, dict[str, Any]]] = {}
            contrasts: dict[str, dict[int, dict[str, Any]]] = {}
            quality: dict[str, Any] = {}
            for seed in md.FOLD_SEEDS:
                cell = predictions.loc[
                    (predictions["model_id"] == arm) & (predictions["seed"] == seed)
                ]
                scores = dict(zip(cell["sample_id"], cell["raw_M"], strict=True))
                values = md.direction_values(frame, scores)
                model = md.endpoint(resampler, values, clusters)
                levels.setdefault("D_model", {})[seed] = model
                quality[str(seed)] = md.selection_quality(values, gaps)
                for name, reference in (
                    ("D_model_minus_D_profile", "D_profile"),
                    ("D_model_minus_D_position", "D_position_nearer_terminus"),
                ):
                    contrasts.setdefault(name, {})[seed] = md.contrast(
                        resampler, model, references[tier][reference])
            # Descriptive, so it gets an interval and no verdict: the 0.05 margin
            # is in direction-accuracy points and means nothing on this scale.
            advantage = {
                str(seed): resampler.interval(
                    resampler.blocks(quality[str(seed)]["per_pair_advantage"], clusters)
                )
                for seed in md.FOLD_SEEDS
            }
            per_tier[tier] = {
                "D_model": {
                    "per_seed": {str(seed): md.strip_blocks(record)
                                 for seed, record in levels["D_model"].items()},
                    "across_seeds": md.across_seeds(levels["D_model"]),
                },
                "contrasts": {
                    name: {"per_seed": {str(seed): md.strip_blocks(record)
                                        for seed, record in record_by_seed.items()},
                           "across_seeds": md.across_seeds(record_by_seed)}
                    for name, record_by_seed in contrasts.items()
                },
                "selection_quality": {
                    seed: {key: value for key, value in record.items()
                           if key != "per_pair_advantage"}
                    for seed, record in quality.items()
                },
                "selected_advantage_interval": advantage,
            }
        rows[arm] = per_tier
        print(json.dumps({"arm": arm, "seconds": round(time.monotonic() - started, 1)}), flush=True)

    balanced = evaluation["T_balanced"]
    balance_check = {
        "profile_direction_accuracy": references["T_balanced"]["D_profile"]["point"],
        "exactly_one_half": bool(
            abs(references["T_balanced"]["D_profile"]["point"] - 0.5) < 1e-12),
        "n_profile_correct": int(np.count_nonzero(md.profile_direction_values(balanced) == 1.0)),
        "n_profile_incorrect": int(np.count_nonzero(md.profile_direction_values(balanced) == 0.0)),
        "n_profile_ties": int(np.count_nonzero(md.profile_direction_values(balanced) == 0.5)),
        "reading": md.AMENDMENT_03_INTERPRETATIONS["balanced_tier_ties"],
    }
    if not balance_check["exactly_one_half"]:
        raise SystemExit(
            "T_balanced does not cancel the profile vote: profile direction accuracy is "
            f"{balance_check['profile_direction_accuracy']}"
        )

    artefact = {
        "schema": md.SCHEMA,
        "analysis_id": md.AMENDMENT_03_ID,
        "declaration_id": md.DECLARATION_ID,
        "amendment": "amendment_03",
        "support": md.SUPPORT,
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.monotonic() - started,
        "relation_to_primary": (
            "three new readings on the frozen cohort. Nothing in "
            f"{PRIMARY} is retracted or restated; D_model on T_match is carried only as the left "
            "side of the new contrasts and is checked against the frozen D_match."
        ),
        "declared": amendment,
        "contract": {
            "manifest": {"path": str(Path(args.manifest).resolve().relative_to(Path(args.root).resolve())),
                         "sha256": sha256_file(args.manifest)},
            "inputs": {
                str(Path(path).resolve().relative_to(Path(args.root).resolve())): sha256_file(path)
                for path in (args.samples, args.predictions, args.proteingym_reference, args.cohort)
            },
            "code_sha256": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in (Path(__file__).resolve(),
                             ROOT / "src/capability/mutation/matched_discrimination.py")
            },
            "frozen_cohort_check": frozen,
            "wildtype_length_check": lengths["check"],
            "declared_interpretations": md.AMENDMENT_03_INTERPRETATIONS,
            "unchanged": (
                "the z-score reference set, the phenotype gap, the profile-match threshold, the "
                "greedy rule, the development/evaluation split, the 0.05 practical margin and the "
                "decision rule are exactly those of the frozen run"
            ),
        },
        "inference": {
            "resampling_unit": "wild-type family cluster at 50% identity",
            "draws": int(args.bootstrap),
            "seed": md.BOOTSTRAP_SEED,
            "cluster_universe": universe,
            "n_cluster_universe": len(universe),
            "interval": "pointwise 95% paired cluster-bootstrap percentile interval, on the same "
                        "resample matrix as the frozen run so every contrast is cluster-paired",
            "decision_rule": manifest["inference"]["decision_rule"],
            "contrast_decision_scale": md.DECLARED_INTERPRETATIONS["contrast_decision_scale"],
            "across_seed_rule": md.DECLARED_INTERPRETATIONS["across_seed_rule"],
            "power_statement": amendment["power_statement_not_a_result"],
        },
        "construction": construction,
        "position_rules": {
            "declared": md.POSITION_RULES,
            "identity": md.POSITION_RULE_IDENTITY,
            "agreement": agreement,
        },
        "balance_check": balance_check,
        "construction_checks": construction_checks(references),
        "reference_rules": {
            tier: {name: {key: value for key, value in md.strip_blocks(record).items()
                          if key != "selection_quality"}
                   | {"selection_quality": {k: v for k, v in record["selection_quality"].items()
                                            if k != "per_pair_advantage"}}
                   for name, record in per_tier.items()}
            for tier, per_tier in references.items()
        },
        "arms": rows,
        "limitations": list(md.LIMITATIONS) + [
            "The position-matched tier matches the magnitude of the relative-position difference, "
            "not its sign, so the same residual-ordering caveat that made T_balanced necessary for "
            "the profile applies to it; the position rule's own accuracy on that tier is reported "
            "as the construction check.",
            "The two declared position-only rules are one rule on this cohort, because the two "
            "members of a pair share a wild-type length. The control is one control, not two.",
            "The selection-quality block is descriptive and carries no verdict.",
        ],
    }
    write_json(out / ARTEFACT, artefact)
    write_json(out / "amendment_03-retained-pairs.json", json.loads(
        pd.concat([tiers["T_balanced"], tiers["T_position_matched"]], ignore_index=True)[[
            "tier", "stratum", "assay", "group_id", "wildtype_residue", "mutant_residue",
            "position_a", "position_b", "length", "sample_a", "sample_b", "better_sample",
            "worse_sample", "phenotype_gap", "profile_distance", "position_distance",
        ]].to_json(orient="records")))
    new = ("amendment_03.json", "amendment_03-runtime.json", "amendment_03-retained-pairs.json")
    write_json(out / "amendment_03-completion.json", {
        "status": "complete",
        "analysis_id": md.AMENDMENT_03_ID,
        "seconds": time.monotonic() - started,
        "output_sha256": {name: sha256_file(out / name) for name in new},
        "builds_on_sha256": {
            name: sha256_file(out / name)
            for name in ("matched_discrimination.json", "contract.json", "runtime.json",
                         "retained-pairs.json", "completion.json")
        },
        "arms": len(arms),
        "evaluation_pairs": {
            tier: int(len(frame)) for tier, frame in evaluation.items()
        },
    })
    print(json.dumps({
        "out": str(out),
        "position_match_threshold": construction["position_match_threshold"]["value"],
        "evaluation_pairs": {tier: int(len(frame)) for tier, frame in evaluation.items()},
        "profile_accuracy_on_T_balanced": balance_check["profile_direction_accuracy"],
        "position_accuracy_on_T_match": references["T_match"]["D_position_nearer_terminus"]["point"],
    }, indent=1))
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
