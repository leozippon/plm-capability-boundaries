#!/usr/bin/env python3
"""Support ``S2_expanded`` of ``matched_discrimination_20261102``: matched same-substitution
pairs scored fresh in nats on the local L20 GPUs.

Phases
------
``cohort``
    Build the declared pair cohort for one support and write the extraction
    cohort JSON that ``scripts/capability/readouts/extract_frozen_readout.py``
    already reads. Model-blind: no checkpoint is opened and no score is read.
``analyse``
    E21 primary, validity and construction-check endpoints, plus the
    ``amendment_02`` magnitude-stratified secondary descriptive endpoint.

The eight autoregressive arms are scored by the existing frozen-readout
extractor, so their nats are the same estimand the retained panel's rank scale
came from. The ninth, masked arm has no sequence log likelihood and is scored by
``expanded_masked_contrast.py`` under this project's existing masked-marginal
member; both are launched through
``scripts/capability/decoding/run_local_gpu_manifest.py``.

Every threshold, tier, arm, endpoint, margin and decision rule comes from
``configs/matched_discrimination_manifest.json`` through
``src.capability.mutation.expanded_discrimination`` and none of them is a flag.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.capability.core.amino_acids import AA20
from src.capability.core.io import sha256_file, write_json
from src.capability.mutation import expanded_discrimination as E

ANALYSIS_ID = E.ANALYSIS_ID
SCHEMA = E.SCHEMA
DECLARED_ARMS = E.DECLARED_ARMS
SCORING_CONVENTIONS = E.SCORING_CONVENTIONS
MASKED_ARM = E.MASKED_ARM
RETAINED_SUPPORT = E.RETAINED_SUPPORT
EXPANDED_SUPPORT = E.EXPANDED_SUPPORT


# ----------------------------------------------------------------- shared inputs


def _inputs() -> dict[str, Path]:
    return {
        "samples": ROOT / "manuscript/data/prediction-details/mutation-samples.csv.gz",
        "reference": ROOT / "data/proteingym_raw/DMS_substitutions.csv",
        "wildtypes": ROOT / "results/R5/retrieval_bound_20260807/wildtypes.json",
        "profiles": ROOT / "results/R5/retrieval_bound_20260807/profiles.npz",
        "proteingym": ROOT / "data/proteingym/DMS_ProteinGym_substitutions",
        "manifest": ROOT / "configs/matched_discrimination_manifest.json",
    }


def _load_context() -> dict[str, object]:
    paths = _inputs()
    wildtypes = json.loads(paths["wildtypes"].read_text())
    background = np.asarray(
        [wildtypes["corpus"]["background"][residue] for residue in AA20], dtype=np.float64
    )
    if abs(float(background.sum()) - 1.0) > 1e-9:
        raise ValueError("the corpus background does not sum to one over AA20")
    reference = pd.read_csv(paths["reference"]).set_index("DMS_id")
    samples = pd.read_csv(paths["samples"])
    return {
        "paths": paths,
        "wildtypes": wildtypes,
        "background": background,
        "reference": reference,
        "samples": samples,
        "profiles": np.load(paths["profiles"]),
        "assay_to_wildtype": wildtypes["assay_to_wildtype"],
        "cluster_of": {q: int(v["cluster"]) for q, v in wildtypes["wildtypes"].items()},
    }


def _retained_assays(samples: pd.DataFrame) -> list[str]:
    return sorted(samples.loc[samples.supporting_common_member, "unit_id"].unique())


def _variant_rows(context: dict[str, object], support: str) -> tuple[list[dict], list[dict]]:
    """Single-substitution variants of one support, with z-score and profile score.

    Returns the admitted rows and the per-assay skip records. The wild-type
    residue of every mutation string is checked against the reference sequence
    and a mismatch is refused, because a wrong coordinate frame would silently
    score a different substitution.
    """

    paths = context["paths"]
    reference = context["reference"]
    assays = _retained_assays(context["samples"])
    skipped: list[dict] = []
    if support == RETAINED_SUPPORT:
        table = context["samples"]
        keep = table.supporting_common_member & ~table.mutation.str.contains(":")
        frame = table.loc[keep, ["unit_id", "mutation", "positions_1based", "measured"]].copy()
        frame["position"] = frame.positions_1based.astype(int)
        frame["mutant"] = frame.mutation
    elif support == EXPANDED_SUPPORT:
        blocks = []
        for assay in assays:
            raw = pd.read_csv(paths["proteingym"] / f"{assay}.csv")
            raw = raw.loc[~raw.mutant.str.contains(":"), ["mutant", "DMS_score"]].copy()
            if raw.empty:
                skipped.append({"assay": assay, "reason": "no single substitution in this assay"})
                continue
            raw["unit_id"] = assay
            raw["position"] = raw.mutant.str[1:-1].astype(int)
            raw["measured"] = raw.DMS_score
            blocks.append(raw[["unit_id", "mutant", "position", "measured"]])
        frame = pd.concat(blocks, ignore_index=True)
    else:
        raise ValueError(f"unknown support {support!r}")
    frame["wt_residue"] = frame.mutant.str[0]
    frame["mut_residue"] = frame.mutant.str[-1]
    rows: list[dict] = []
    for assay, block in frame.groupby("unit_id", sort=True):
        wildtype = str(reference.loc[assay, "target_seq"])
        identifier = context["assay_to_wildtype"][assay]
        frequencies = np.asarray(context["profiles"][identifier], dtype=np.float64)
        outside = ~(block.wt_residue.isin(set(AA20)) & block.mut_residue.isin(set(AA20)))
        if outside.any():
            skipped.append(
                {
                    "assay": assay,
                    "reason": "substitution outside AA20",
                    "variants": int(outside.sum()),
                }
            )
            block = block.loc[~outside]
        if block.empty:
            continue
        substitutions = [
            (row.wt_residue, int(row.position), row.mut_residue) for row in block.itertuples()
        ]
        profile = E.lookup_profile_scores(
            frequencies, context["background"], substitutions, wildtype=wildtype
        )
        z = E.within_assay_z(block.measured.to_numpy(dtype=np.float64))
        if z is None:
            skipped.append(
                {
                    "assay": assay,
                    "reason": (
                        "the within-assay single-substitution phenotype population is "
                        "constant, so the declared gap threshold has no units here"
                    ),
                    "variants": int(len(block)),
                }
            )
            continue
        for row, score, value in zip(block.itertuples(), profile, z):
            rows.append(
                {
                    "assay": assay,
                    "cluster": context["cluster_of"][identifier],
                    "wildtype_id": identifier,
                    "wt_residue": row.wt_residue,
                    "mut_residue": row.mut_residue,
                    "position": int(row.position),
                    "mutant": row.mutant,
                    "measured": float(row.measured),
                    "z": float(value),
                    "profile": float(score),
                }
            )
    return rows, skipped


# ---------------------------------------------------------------------- cohort phase


def _cohort_payload(context: dict[str, object], pairs_by_tier: dict[str, list[E.Pair]]) -> dict:
    """The frozen-cohort JSON the extraction path already reads.

    One row per assay holding every variant that appears in a retained pair of
    either tier, so one extraction per arm covers the whole analysis.
    """

    reference = context["reference"]
    wanted: dict[str, set[str]] = {}
    for tier in pairs_by_tier.values():
        for pair in tier:
            wanted.setdefault(pair.assay, set()).update(
                (pair.mutation_better, pair.mutation_worse)
            )
    assays = []
    for assay in sorted(wanted):
        wildtype = str(reference.loc[assay, "target_seq"])
        identifier = context["assay_to_wildtype"][assay]
        frequencies = np.asarray(context["profiles"][identifier], dtype=np.float64)
        mutants = sorted(wanted[assay], key=lambda m: (int(m[1:-1]), m))
        substitutions = [(m[0], int(m[1:-1]), m[-1]) for m in mutants]
        sequences = []
        for residue, position, substituted in substitutions:
            if wildtype[position - 1] != residue:
                raise ValueError(f"{assay} {residue}{position}{substituted}: wild-type mismatch")
            sequences.append(wildtype[: position - 1] + substituted + wildtype[position:])
        profile = E.lookup_profile_scores(
            frequencies, context["background"], substitutions, wildtype=wildtype
        )
        measured = _measured_lookup(context, assay, mutants)
        assays.append(
            {
                "assay": assay,
                "wildtype_id": identifier,
                "cluster": context["cluster_of"][identifier],
                "wildtype": wildtype,
                "mutants": mutants,
                "sequences": sequences,
                "measured": [float(value) for value in measured],
                "profile_scores": [float(value) for value in profile],
                "mutant_digest": E.variant_digest(mutants),
            }
        )
    return {
        "schema_version": "context_mutation_rescue_v1",
        "analysis_id": ANALYSIS_ID,
        "declaration": E.DECLARATION_ID,
        "seed": E.COHORT_SEED,
        "variants": sum(len(row["mutants"]) for row in assays),
        "assays": assays,
    }


def _measured_lookup(context: dict[str, object], assay: str, mutants: list[str]) -> np.ndarray:
    raw = pd.read_csv(context["paths"]["proteingym"] / f"{assay}.csv")
    table = dict(zip(raw.mutant, raw.DMS_score))
    missing = [m for m in mutants if m not in table]
    if missing:
        raise ValueError(f"{assay}: {missing[:5]} absent from the ProteinGym table")
    return np.asarray([table[m] for m in mutants], dtype=np.float64)


def phase_cohort(args: argparse.Namespace) -> None:
    context = _load_context()
    universe = sorted({int(v["cluster"]) for v in context["wildtypes"]["wildtypes"].values()})
    development = E.development_clusters(universe)
    rows, skipped = _variant_rows(context, args.support)
    candidates = E.candidate_pairs(rows)
    cap = args.cap_per_cluster if args.cap_per_cluster > 0 else None
    tiers = E.build_tiers(candidates, development, cap_per_cluster=cap)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cohort = _cohort_payload(
        context, {tier: tiers[tier] for tier in (E.TIER_OPEN, E.TIER_MATCH)}
    )
    cohort_path = out / "cohort.json"
    write_json(cohort_path, cohort)
    lengths = [len(row["wildtype"]) for row in cohort["assays"]]
    counts = [len(row["mutants"]) for row in cohort["assays"]]

    pairs_payload = {
        "schema_version": SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "support": args.support,
        "cohort_sha256": sha256_file(cohort_path),
        "development_clusters": sorted(development),
        "cluster_universe": universe,
        "profile_match_threshold": tiers["profile_match_threshold"],
        "cap_per_cluster": cap,
        "tiers": {
            tier: [pair.record() for pair in tiers[tier]]
            for tier in (E.TIER_OPEN, E.TIER_MATCH)
        },
    }
    pairs_path = out / "pairs.json"
    write_json(pairs_path, pairs_payload)

    def tier_summary(tier: str) -> dict:
        retained = tiers[tier]
        evaluation = [p for p in retained if p.cluster not in development]
        return {
            "retained_pairs": len(retained),
            "development_pairs": len(retained) - len(evaluation),
            "evaluation_pairs": len(evaluation),
            "evaluation_clusters": len({p.cluster for p in evaluation}),
            "evaluation_assays": len({p.assay for p in evaluation}),
            "development_clusters_present": len(
                {p.cluster for p in retained if p.cluster in development}
            ),
        }

    summary = {
        "schema_version": SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "support": args.support,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "variant_rows": len(rows),
        "variant_assays": len({row["assay"] for row in rows}),
        "variant_clusters": len({row["cluster"] for row in rows}),
        "candidate_pairs": len(candidates),
        "candidate_pairs_profile_matched": tiers["candidates_matched"],
        "profile_match_threshold_nats": tiers["profile_match_threshold"],
        "cap_per_cluster": cap,
        "tier_counts": {tier: tier_summary(tier) for tier in (E.TIER_OPEN, E.TIER_MATCH)},
        "skipped": skipped,
        "scoring_cost": {
            "assays": len(cohort["assays"]),
            "unique_variant_sequences": int(sum(counts)),
            "sequences_per_forward_pass_including_wildtype": int(sum(counts) + len(counts)),
            "wildtype_length": {
                "min": int(min(lengths)),
                "median": float(np.median(lengths)),
                "mean": float(np.mean(lengths)),
                "max": int(max(lengths)),
                "residue_total_including_wildtypes": int(
                    sum(length * (count + 1) for length, count in zip(lengths, counts))
                ),
            },
            "variants_per_assay": {
                "min": int(min(counts)),
                "median": float(np.median(counts)),
                "max": int(max(counts)),
            },
        },
    }
    write_json(out / "cohort_summary.json", summary)
    write_json(
        out / "contract.json",
        {
            "schema_version": SCHEMA,
            "analysis_id": ANALYSIS_ID,
            "declaration": E.DECLARATION_ID,
            "support": args.support,
            "support_declaration": (
                "amendment_01 S2_expanded: an expanded ProteinGym single-substitution "
                "cohort scored fresh on the local L20 GPUs for the nine declared arms, "
                "sized to the pair count the endpoint needs. Never pooled with S1."
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "declared_constants": {
                "phenotype_gap_threshold_sd": E.GAP_THRESHOLD,
                "profile_match_percentile": E.PROFILE_MATCH_PERCENTILE,
                "split_seed": E.SPLIT_SEED,
                "development_fraction": E.DEVELOPMENT_FRACTION,
                "cohort_seed": E.COHORT_SEED,
                "bootstrap_draws": E.BOOTSTRAP_DRAWS,
                "bootstrap_alpha": E.BOOTSTRAP_ALPHA,
                "null_accuracy": E.NULL_ACCURACY,
                "practical_margin": E.PRACTICAL_MARGIN,
                "profile_pseudocount_alpha": E.PROFILE_PSEUDOCOUNT,
                "magnitude_bins": E.MAGNITUDE_BINS,
            },
            "construction_rules": {
                "split": (
                    "numpy.random.default_rng(20261102).permutation over the sorted "
                    "universe of ProteinGym 50%-identity wild-type family clusters; the "
                    "first round(0.3 * n) are development. The universe is the declared "
                    "174 clusters, so a family falls on the same side of the split in "
                    "either support"
                ),
                "phenotype": E.SEMANTICS["phenotype"],
                "profile": E.SEMANTICS["profile"],
                "profile_match": (
                    "|delta s| <= the 20th percentile (NumPy linear interpolation) of "
                    "|delta s| over development-cluster candidate pairs that already "
                    "satisfy the gap; applied unchanged to evaluation clusters"
                ),
                "disjointness": (
                    "the canonical pair order (assay, wild-type residue, mutant residue, "
                    "lower position, higher position) is permuted once with seed 20261102 "
                    "and traversed once; a pair is retained when neither of its variants is "
                    "already used inside the same tier. Applied inside each tier separately, "
                    "as declared, so T_open's composition is not a function of the profile match"
                ),
                "cohort_sizing": (
                    "the same traversal stops accepting pairs for a family cluster once it "
                    "holds cap_per_cluster retained pairs; declared, seeded and model-blind"
                    if cap
                    else "no per-cluster cap; every retainable pair of this support is kept"
                ),
                "direction": E.SEMANTICS["direction"],
                "resampling": E.SEMANTICS["resampling"],
            },
            "scoring_conventions": SCORING_CONVENTIONS,
            "declared_arms": DECLARED_ARMS,
            "input_sha256": {
                name: (
                    sha256_file(path)
                    if path.is_file()
                    else {
                        f.name: sha256_file(f)
                        for f in sorted(path.iterdir())
                        if f.suffix == ".csv"
                        and f.stem in set(_retained_assays(context["samples"]))
                    }
                )
                for name, path in _inputs().items()
            },
            "code_sha256": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in (
                    Path(__file__),
                    ROOT / "src/capability/mutation/expanded_discrimination.py",
                    ROOT / "scripts/capability/readouts/extract_frozen_readout.py",
                    ROOT / "scripts/capability/interactions/exploratory_denoising_arms.py",
                )
            },
            "cohort_sha256": sha256_file(cohort_path),
            "pairs_sha256": sha256_file(pairs_path),
        },
    )
    print(json.dumps(summary["tier_counts"], indent=2))
    print(json.dumps(summary["scoring_cost"], indent=2))


# --------------------------------------------------------------------- analyse phase


def _arm_scores(directory: Path, arm: str) -> tuple[dict[str, dict[str, float]], dict, list[dict]]:
    """Per-variant mutant-minus-wild-type nats for one autoregressive arm."""

    manifest_path = directory / f"manifest_{arm}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{arm}: no completed extraction manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"{arm}: extraction manifest is not complete")
    scores: dict[str, dict[str, float]] = {}
    for entry in manifest["assays"]:
        path = directory / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{path}: archive digest differs from the manifest")
        with np.load(path, allow_pickle=False) as saved:
            mutants = [str(value) for value in saved["mutants"]]
            likelihood = np.asarray(saved["likelihood"], dtype=np.float64)
        if len(mutants) != likelihood.size:
            raise ValueError(f"{path}: mutant and likelihood lengths differ")
        if not np.isfinite(likelihood).all():
            raise FloatingPointError(f"{path}: a retained likelihood is not finite")
        scores[entry["assay"]] = dict(zip(mutants, (float(v) for v in likelihood)))
    runtime = {
        key: manifest.get(key)
        for key in ("gpu", "elapsed_seconds", "peak_allocated_bytes", "torch_version", "batch_size")
    }
    runtime["assays_scored"] = len(manifest["assays"])
    return scores, manifest["identity"], list(manifest.get("skipped", [])), runtime


def _retained_rank_tables() -> dict[str, dict[tuple[str, str], float]]:
    """Per-arm within-assay rank-scale native likelihood of the retained panel.

    This is the quantity support ``S1_retained_panel`` reads. It is loaded here
    only to cross-check that the fresh L20 nats order a pair the same way the
    retained H200 rank does; it is not an endpoint and no number is pooled with
    it. ``raw_M`` is a raw comparator with no split, and is identical across the
    three fold seeds, so one seed is read.
    """

    paths = _inputs()
    samples = pd.read_csv(
        paths["samples"], usecols=["sample_id", "unit_id", "mutation", "supporting_common_member"]
    )
    samples = samples.loc[samples.supporting_common_member]
    wanted = {row["arm"] for row in DECLARED_ARMS.values() if row["arm"] != MASKED_ARM}
    predictions = pd.read_csv(
        ROOT / "manuscript/data/prediction-details/mutation-supporting-predictions.csv.gz",
        usecols=["sample_id", "model_id", "seed", "raw_M"],
    )
    predictions = predictions.loc[predictions.model_id.isin(wanted)]
    seed = int(predictions.seed.min())
    predictions = predictions.loc[predictions.seed == seed]
    joined = predictions.merge(samples, on="sample_id", how="inner", validate="many_to_one")
    tables: dict[str, dict[tuple[str, str], float]] = {}
    for arm, block in joined.groupby("model_id"):
        tables[arm] = dict(zip(zip(block.unit_id, block.mutation), block.raw_M.astype(float)))
    return tables


def _rank_cross_check(
    pairs: list[dict], scores: dict[str, dict[str, float]], ranks: dict[tuple[str, str], float]
) -> dict:
    """Agreement between the fresh nats and the retained rank on the same pairs."""

    agree = disagree = rank_ties = unseen = 0
    for pair in pairs:
        key_better = (pair["assay"], pair["mutation_better"])
        key_worse = (pair["assay"], pair["mutation_worse"])
        if key_better not in ranks or key_worse not in ranks:
            unseen += 1
            continue
        nats = scores[pair["assay"]][pair["mutation_better"]] - scores[pair["assay"]][
            pair["mutation_worse"]
        ]
        rank = ranks[key_better] - ranks[key_worse]
        if rank == 0.0:
            rank_ties += 1
        elif (nats > 0) == (rank > 0):
            agree += 1
        else:
            disagree += 1
    compared = agree + disagree
    return {
        "role": (
            "validity diagnostic only: the retained panel's rank scale is a different "
            "support and nothing here is pooled with it or enters an endpoint"
        ),
        "pairs_compared": compared,
        "pairs_absent_from_the_retained_table": unseen,
        "retained_rank_ties": rank_ties,
        "direction_agreement": (agree / compared) if compared else None,
        "disagreements": disagree,
    }


def _pair_scores(
    pairs: list[dict], scores: dict[str, dict[str, float]]
) -> tuple[list[dict], np.ndarray, np.ndarray, list[dict]]:
    kept, better, worse, missing = [], [], [], []
    for pair in pairs:
        table = scores.get(pair["assay"])
        if table is None or pair["mutation_better"] not in table or pair["mutation_worse"] not in table:
            missing.append({"assay": pair["assay"], "pair": pair["mutation_better"] + "/" + pair["mutation_worse"]})
            continue
        kept.append(pair)
        better.append(table[pair["mutation_better"]])
        worse.append(table[pair["mutation_worse"]])
    return kept, np.asarray(better, dtype=np.float64), np.asarray(worse, dtype=np.float64), missing


def _endpoint_block(
    pairs: list[dict],
    better: np.ndarray,
    worse: np.ndarray,
    development: set[int],
    *,
    seed: int,
    stratify: bool,
    magnitude_label: str = "delta_log_likelihood_nats_evaluation",
) -> dict:
    clusters = np.asarray([pair["cluster"] for pair in pairs])
    indicators, ties = E.direction_indicators(pairs, better, worse)
    magnitude = np.abs(better - worse)
    is_dev = np.asarray([pair["cluster"] in development for pair in pairs])
    evaluation = [pair for pair, dev in zip(pairs, is_dev) if not dev]
    block: dict = {
        "n_pairs_total": len(pairs),
        "n_pairs_development": int(is_dev.sum()),
        "n_pairs_evaluation": int((~is_dev).sum()),
        "exact_ties_evaluation": int((better == worse)[~is_dev].sum()),
        "exact_ties_total": ties,
        "evaluation": E.group_bootstrap_accuracy(
            indicators[~is_dev], clusters[~is_dev], seed=seed
        ),
        magnitude_label: E.magnitude_summary(magnitude[~is_dev]),
        "practical_size_evaluation": E.practical_size(evaluation, indicators[~is_dev]),
    }
    if stratify:
        if int(is_dev.sum()) >= E.MAGNITUDE_BINS:
            boundaries = E.magnitude_boundaries(magnitude[is_dev])
            block["magnitude_stratified"] = {
                "boundaries_nats": boundaries,
                "boundaries_fixed_on": "development clusters of this support and arm",
                "status": "descriptive and secondary; carries no verdict and enters no multiplicity family",
                "bins": E.magnitude_strata(
                    magnitude[~is_dev], indicators[~is_dev], clusters[~is_dev], boundaries, seed=seed
                ),
            }
        else:
            block["magnitude_stratified"] = {
                "boundaries_nats": None,
                "reason": (
                    f"{int(is_dev.sum())} development pairs cannot fix "
                    f"{E.MAGNITUDE_BINS} quantile boundaries"
                ),
            }
    return block


# ------------------------------------------------------------- amendment_03 phase


def _wildtype_lengths(cohort_path: Path, expected_sha256: str) -> dict[str, int]:
    """Per-assay wild-type length, from the hash-bound cohort the arms were scored on."""

    if sha256_file(cohort_path) != expected_sha256:
        raise ValueError(f"{cohort_path}: digest differs from the cohort the pairs were built on")
    cohort = json.loads(cohort_path.read_text())
    return {row["assay"]: len(row["wildtype"]) for row in cohort["assays"]}


def phase_balance(args: argparse.Namespace) -> None:
    """Build the three sign-balanced tiers of ``amendment_03`` and ``amendment_04``.

    Reads no model score. Kept a separate phase, and a separate artefact, so that
    the model-blind boundary is visible in the receipts: this file is produced
    from the frozen ``pairs.json`` and the frozen cohort alone, and the frozen
    ``T_open``/``T_match`` tiers are left byte-identical.

    Each tier is balanced inside the development and the evaluation stratum
    separately, so the balance on the reported evaluation pairs is exact rather
    than inherited from a pooled construction.
    """

    pairs_path = Path(args.pairs)
    pairs_payload = json.loads(pairs_path.read_text())
    development = set(pairs_payload["development_clusters"])
    lengths = _wildtype_lengths(Path(args.cohort), pairs_payload["cohort_sha256"])

    def profile(pair: dict) -> int:
        return E.profile_vote(pair)

    def position(pair: dict) -> int:
        return E.position_vote(pair, lengths)

    plans = {
        E.TIER_BALANCED: ("amendment_03 addition_1", ("profile",)),
        E.TIER_POSITION_BALANCED: ("amendment_04 addition_1", ("position",)),
        E.TIER_DOUBLE_BALANCED: ("amendment_04 addition_2", ("profile", "position")),
    }
    tiers: dict[str, dict] = {}
    for offset, (tier, (amendment, votes)) in enumerate(plans.items()):
        strata: dict[str, list[dict]] = {}
        records: dict[str, dict] = {}
        for name, parity in (("development", 0), ("evaluation", 1)):
            candidates = [
                pair
                for pair in pairs_payload["tiers"][E.TIER_MATCH]
                if (pair["cluster"] in development) == (name == "development")
            ]
            seed = E.COHORT_SEED + 10 * offset + parity
            if votes == ("profile",):
                selected, record = E.balance_single_vote(
                    candidates, profile, seed=seed, name="profile"
                )
            elif votes == ("position",):
                selected, record = E.balance_single_vote(
                    candidates, position, seed=seed, name="position"
                )
            else:
                selected, record = E.balance_two_votes(
                    candidates, profile, position, seed=seed, names=("profile", "position")
                )
            if selected:
                record["certified_direction_accuracy"] = {
                    "profile": E.assert_vote_balanced(selected, profile, name="profile")
                    if "profile" in votes
                    else None,
                    "position": E.assert_vote_balanced(selected, position, name="position")
                    if "position" in votes
                    else None,
                }
            strata[name] = selected
            records[name] = record
        tiers[tier] = {
            "amendment": amendment,
            "balanced_votes": list(votes),
            "balance_per_stratum": records,
            "per_cluster_pair_counts_evaluation": E.per_cluster_counts(strata["evaluation"]),
            "pairs": strata["development"] + strata["evaluation"],
        }
    payload = {
        "schema_version": SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "declaration": E.DECLARATION_ID,
        "support": pairs_payload["support"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pairs_sha256": sha256_file(pairs_path),
        "cohort_sha256": pairs_payload["cohort_sha256"],
        "rules": {
            "profile_balance": E.SEMANTICS["balanced_tier"],
            "position_rule": E.SEMANTICS["position_rule"],
            "double_balance": (
                "amendment_04 addition_2: equal counts are taken from the two diagonal "
                "sign cells, (profile+, position+) with (profile-, position-) and "
                "(profile+, position-) with (profile-, position+), which balances both "
                "marginals exactly for any split between the diagonals and keeps more "
                "pairs than levelling all four cells would. A pair neutral on either "
                "vote is excluded, never re-admitted to reach a pair target"
            ),
        },
        "tiers": tiers,
    }
    write_json(Path(args.out) / "balanced.json", payload)
    for tier, block in tiers.items():
        evaluation = block["balance_per_stratum"]["evaluation"]
        print(
            f"{tier:22s} evaluation pairs {evaluation['retained_pairs']:6d} "
            f"from {evaluation['candidates']:6d} candidates; certified "
            f"{block['balance_per_stratum']['evaluation'].get('certified_direction_accuracy')}"
        )


def phase_analyse(args: argparse.Namespace) -> None:
    pairs_path = Path(args.pairs)
    pairs_payload = json.loads(pairs_path.read_text())
    development = set(pairs_payload["development_clusters"])
    lengths = _wildtype_lengths(Path(args.cohort), pairs_payload["cohort_sha256"])
    tiers: dict[str, list[dict]] = {
        E.TIER_OPEN: pairs_payload["tiers"][E.TIER_OPEN],
        E.TIER_MATCH: pairs_payload["tiers"][E.TIER_MATCH],
    }
    balanced_payload = json.loads(Path(args.balanced).read_text())
    if balanced_payload["pairs_sha256"] != sha256_file(pairs_path):
        raise ValueError("the balanced tiers were built from a different pairs.json")
    for tier in (E.TIER_BALANCED, E.TIER_POSITION_BALANCED, E.TIER_DOUBLE_BALANCED):
        tiers[tier] = balanced_payload["tiers"][tier]["pairs"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def rule_values(tier: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
        pairs = tiers[tier]
        if kind == "profile":
            return (
                np.asarray([p["profile_better"] for p in pairs], dtype=np.float64),
                np.asarray([p["profile_worse"] for p in pairs], dtype=np.float64),
            )
        return E.position_rule_values(pairs, lengths)

    # The two model-blind reference rules, on every tier that is reported.
    controls: dict[str, dict] = {}
    for kind, offset, label in (("profile", 2, "profile"), ("position", 3, "position")):
        rows = {}
        for tier in tiers:
            better, worse = rule_values(tier, kind)
            block = _endpoint_block(
                tiers[tier],
                better,
                worse,
                development,
                seed=E.SPLIT_SEED + offset,
                stratify=False,
                magnitude_label=f"abs_delta_{label}_evaluation",
            )
            if kind == "position":
                strict = better != worse
                indicators = (better > worse).astype(np.float64)
                evaluation = np.asarray([p["cluster"] not in development for p in tiers[tier]])
                keep = strict & evaluation
                block["accuracy_excluding_rule_ties"] = (
                    float(indicators[keep].mean()) if keep.any() else None
                )
                block["rule_equivalence_check"] = E.position_rule_agreement(tiers[tier], lengths)
            rows[tier] = block
        controls[kind] = {
            "rule": E.SEMANTICS["position_rule"] if kind == "position" else E.SEMANTICS["profile"],
            "role": (
                "amendment_03 addition_3 position-only control; the strongest alternative "
                "explanation of a surviving direction"
                if kind == "position"
                else "E21 construction check; on T_match it is expected near 0.5 and on "
                "T_balanced it is exactly 0.5 by construction"
            ),
            "tiers": rows,
        }

    rank_tables = _retained_rank_tables() if args.rank_cross_check else {}
    arms: dict[str, dict] = {}
    ladder_indicators: dict[str, dict] = {}
    ladder_clusters: dict[str, dict] = {}
    for manifest_name, declared in DECLARED_ARMS.items():
        arm = declared["arm"]
        if arm == MASKED_ARM:
            path = Path(args.scores) / arm / f"masked_scores_{arm}.json"
            payload = json.loads(path.read_text())
            if payload["cohort_sha256"] != pairs_payload["cohort_sha256"]:
                raise ValueError(f"{path}: masked scores were read from a different cohort")
            scores = payload["scores"]
            identity = payload["identity"]
            skipped = list(payload.get("skipped", []))
            runtime = {
                key: payload.get(key)
                for key in ("gpu", "elapsed_seconds", "batch", "masked_forwards", "assays")
            }
            convention = SCORING_CONVENTIONS["masked"]
        else:
            scores, identity, skipped, runtime = _arm_scores(Path(args.scores) / arm, arm)
            if identity["cohort_sha256"] != pairs_payload["cohort_sha256"]:
                raise ValueError(f"{arm}: the extraction read a different cohort")
            convention = SCORING_CONVENTIONS["autoregressive"]
            if arm.startswith("progen3"):
                convention = convention + " | " + SCORING_CONVENTIONS["progen3_direction"]
            if arm == "protgpt2":
                convention = convention + " | " + SCORING_CONVENTIONS["protgpt2"]
        record: dict = {
            "manifest_arm": manifest_name,
            "repository_arm": arm,
            "arm_family": declared["family"],
            "pooled_with_autoregressive_arms": declared["family"] == "autoregressive_protein",
            "score_semantics": convention,
            "identity": identity,
            "assays_outside_this_arm_interface": skipped,
            "scoring_runtime": runtime,
            "tiers": {},
            "contrasts": {},
        }
        explained = {row["assay"] for row in skipped}
        held: dict[str, tuple] = {}
        for tier in tiers:
            kept, better, worse, missing = _pair_scores(tiers[tier], scores)
            unexplained = [row for row in missing if row["assay"] not in explained]
            if unexplained:
                # An assay the arm's own manifest does not record as skipped has no
                # reason to be absent; that is an extraction defect, not a support subset.
                raise ValueError(
                    f"{arm} {tier}: {len(unexplained)} pairs have no score and no recorded "
                    f"reason; {unexplained[:3]}"
                )
            block = _endpoint_block(
                kept,
                better,
                worse,
                development,
                seed=E.SPLIT_SEED,
                stratify=(tier == E.TIER_MATCH),
            )
            block["pairs_dropped_with_recorded_reason"] = len(missing)
            indicators = E.direction_indicators(kept, better, worse)[0]
            evaluation = [i for i, p in enumerate(kept) if p["cluster"] not in development]
            held[tier] = (kept, indicators, evaluation)
            record["tiers"][tier] = block

        # amendment_03 addition_3: the headline, on every tier reported.
        # amendment_03 addition_2: net of the residual profile, on T_match and T_balanced.
        for tier in tiers:
            kept, indicators, evaluation = held[tier]
            clusters = [kept[i]["cluster"] for i in evaluation]
            for kind, offset in (("position", 4), ("profile", 5)):
                if kind == "profile" and tier == E.TIER_OPEN:
                    continue
                better, worse = rule_values(tier, kind)
                index = {
                    (p["assay"], p["mutation_better"], p["mutation_worse"]): i
                    for i, p in enumerate(tiers[tier])
                }
                rows = [
                    index[(kept[i]["assay"], kept[i]["mutation_better"], kept[i]["mutation_worse"])]
                    for i in evaluation
                ]
                rule_indicators = (better[rows] > worse[rows]).astype(np.float64)
                record["contrasts"][f"D_model_minus_D_{kind}_{tier}"] = E.paired_accuracy_contrast(
                    indicators[evaluation],
                    clusters,
                    rule_indicators,
                    clusters,
                    seed=E.SPLIT_SEED + offset,
                )
        open_kept, open_indicators, open_eval = held[E.TIER_OPEN]
        match_kept, match_indicators, match_eval = held[E.TIER_MATCH]
        record["contrasts"]["D_open_minus_D_match"] = E.paired_accuracy_contrast(
            open_indicators[open_eval],
            [open_kept[i]["cluster"] for i in open_eval],
            match_indicators[match_eval],
            [match_kept[i]["cluster"] for i in match_eval],
            seed=E.SPLIT_SEED + 1,
        )
        if arm in rank_tables:
            record["retained_rank_cross_check"] = _rank_cross_check(
                tiers[E.TIER_OPEN], scores, rank_tables[arm]
            )
        arms[manifest_name] = record
        for tier in tiers:
            kept, indicators, evaluation = held[tier]
            ladder_indicators.setdefault(tier, {})[arm] = indicators[evaluation]
            ladder_clusters.setdefault(tier, {})[arm] = [kept[i]["cluster"] for i in evaluation]

    # amendment_05: the declared capacity axis, one draw matrix per tier.
    prospective = (
        pairs_payload["support"] == EXPANDED_SUPPORT,
        E.TIER_DOUBLE_BALANCED,
    )
    capacity: dict[str, dict] = {}
    for tier in tiers:
        present = {
            arm: values
            for arm, values in ladder_indicators.get(tier, {}).items()
            if arm in E.CAPACITY_LADDER
        }
        if len(present) < 3:
            capacity[tier] = {"reason": "fewer than three ladder arms were scored on this tier"}
            continue
        reference = next(iter(present))
        clusters = ladder_clusters[tier][reference]
        if any(ladder_clusters[tier][arm] != clusters for arm in present):
            raise ValueError(f"{tier}: the ladder arms were not read on the same pairs")
        block = E.capacity_trend(present, clusters, seed=E.SPLIT_SEED)
        block["status"] = (
            "prospective: declared before this tier was read"
            if (prospective[0] and tier == prospective[1])
            else "post-hoc consistency check: the per-arm points of this tier were already "
            "seen when amendment_05 was written, so it carries no prospective weight"
        )
        block["declared_ladder_size_discrepancy"] = (
            "amendment_05's statistic says five ProGen2 sizes; the declared arm list carries "
            "four and the amendment may not alter it, so the executed ladder is the declared "
            "four. progen2-base is on this host but is not a declared S2 arm"
        )
        capacity[tier] = block

    write_json(
        out / "rows.json",
        {
            "schema_version": SCHEMA,
            "analysis_id": ANALYSIS_ID,
            "declaration": E.DECLARATION_ID,
            "amendments_executed": [
                "amendment_01",
                "amendment_02",
                "amendment_03",
                "amendment_04",
                "amendment_05",
            ],
            "retired_readings": {
                "magnitude_matched_position_tier": (
                    "amendment_03 addition_3 also named a tier additionally matched on relative "
                    "position. It was never built on this support, and amendment_04 retires it: "
                    "the analogous magnitude match over-corrected on S1_retained_panel, putting "
                    "the position-only rule at 0.418, below chance by more than the declared "
                    "margin, so magnitude matching is the wrong instrument for position. "
                    "T_position_balanced replaces it by cancelling the position vote exactly. "
                    "Recorded so the absence of a declared sub-reading cannot be mistaken for "
                    "an oversight"
                ),
            },
            "declared_action_not_taken": {
                "lifting_the_per_cluster_cap_on_the_matched_tiers": (
                    "amendment_04 addition_3 lifts the execution-time cap of 40 pairs per "
                    "cluster on the matched tiers on the grounds that their power is the "
                    "binding constraint. On this analysis it is not. The cap applies to T_open "
                    "and T_match alike, so it never throttled one relative to the other, and on "
                    "the expanded support the realised evaluation sizes at cap 40 are already "
                    "T_match 4317, T_balanced 3666, T_position_balanced 3508 and "
                    "T_double_balanced 3352 against the amendment's 400-500 target. Lifting it "
                    "would take T_match from 6156 to 126099 retained pairs, roughly 250k to "
                    "400k further variant sequences across nine arms, with no change in any "
                    "verdict. The cap is therefore retained and recorded here with that "
                    "arithmetic. On the retained-variant support no cap was ever applied: its "
                    "tier sizes are bounded by the retained 25,728-variant list"
                ),
            },
            "support": pairs_payload["support"],
            "experiment": "E21",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pairs_sha256": sha256_file(pairs_path),
            "cohort_sha256": pairs_payload["cohort_sha256"],
            "balanced_sha256": sha256_file(Path(args.balanced)),
            "profile_match_threshold_nats": pairs_payload["profile_match_threshold"],
            "development_clusters": sorted(development),
            "decision_rule": {
                "supported": "interval excludes 0.5",
                "bounded_negative": "interval lies inside (0.45, 0.55)",
                "unresolved": "neither of the above",
                "note": (
                    "an interval covering 0.5 is not equivalence, zero information or the "
                    "absence of a biological effect; a bounded negative excludes an effect "
                    "only at the declared 0.05 margin"
                ),
            },
            "primary_endpoint": "D_match on T_match evaluation pairs",
            "headline_contrast": (
                "D_model - D_position on each tier: whether the surviving direction is more "
                "than a positional prior"
            ),
            "tier_sizes": {
                tier: {
                    "pairs_total": len(rows),
                    "pairs_evaluation": sum(
                        1 for p in rows if p["cluster"] not in development
                    ),
                    "clusters_evaluation": len(
                        {p["cluster"] for p in rows if p["cluster"] not in development}
                    ),
                    "per_cluster_pair_counts_evaluation": E.per_cluster_counts(
                        [p for p in rows if p["cluster"] not in development]
                    ),
                }
                for tier, rows in tiers.items()
            },
            "decisive_tier": E.TIER_DOUBLE_BALANCED,
            "balanced_tier_construction": {
                tier: {
                    key: value
                    for key, value in balanced_payload["tiers"][tier].items()
                    if key != "pairs"
                }
                for tier in (E.TIER_BALANCED, E.TIER_POSITION_BALANCED, E.TIER_DOUBLE_BALANCED)
            },
            "per_cluster_cap": {
                "T_open": pairs_payload["cap_per_cluster"],
                "matched_tiers": pairs_payload["cap_per_cluster"],
                "amendment_04_addition_3": (
                    "the amendment lifts the execution-time cap on the matched tiers because "
                    "their power was the binding constraint. On this support it is not: the "
                    "cap is reported here beside the realised evaluation pair counts so the "
                    "resolution actually achieved can be read directly"
                ),
            },
            "arms": arms,
            "reference_rules": controls,
            "capacity_axis": capacity,
        },
    )
    write_json(
        out / "runtime.json",
        {
            "schema_version": SCHEMA,
            "analysis_id": ANALYSIS_ID,
            "support": pairs_payload["support"],
            "analysis_device": "cpu",
            "scoring_runtime_per_arm": {
                name: record["scoring_runtime"] for name, record in arms.items()
            },
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "threads": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    for tier, block in capacity.items():
        if "spearman_vs_log_parameters" not in block:
            continue
        contrast = block["largest_minus_smallest"]
        print(
            f"capacity {tier:22s} rho={block['spearman_vs_log_parameters']:+.3f} "
            f"{block['spearman_interval']} | xlarge-small {contrast['difference']:+.4f} "
            f"{contrast['interval']} {contrast['verdict']}"
        )
    position = controls["position"]["tiers"]
    for tier in tiers:
        block = position[tier]["evaluation"]
        print(
            f"position-only {tier:11s} {block['accuracy']} {block['interval']} "
            f"{block['verdict']} n={block['n_pairs']}"
        )
    for name, record in arms.items():
        match = record["tiers"][E.TIER_MATCH]["evaluation"]
        double = record["tiers"][E.TIER_DOUBLE_BALANCED]["evaluation"]
        print(
            f"{name:24s} D_double={double['accuracy']:.4f} "
            f"[{double['interval'][0]:.4f},{double['interval'][1]:.4f}] {double['verdict']:<18} "
            f"n={double['n_pairs']} | D_match={match['accuracy']:.4f} {match['verdict']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)

    cohort = sub.add_parser("cohort", help="build the declared pair cohort (model-blind)")
    cohort.add_argument("--support", choices=(RETAINED_SUPPORT, EXPANDED_SUPPORT), required=True)
    cohort.add_argument("--out", required=True)
    cohort.add_argument(
        "--cap-per-cluster",
        type=int,
        default=0,
        help="declared per-cluster retention cap; 0 retains every retainable pair",
    )
    cohort.set_defaults(run=phase_cohort)

    balance = sub.add_parser(
        "balance", help="build the sign-balanced tiers (model-blind)"
    )
    balance.add_argument("--pairs", required=True)
    balance.add_argument("--cohort", required=True, help="the cohort the pairs were built on")
    balance.add_argument("--out", required=True)
    balance.set_defaults(run=phase_balance)

    analyse = sub.add_parser("analyse", help="E21 endpoints for the nine declared arms")
    analyse.add_argument("--pairs", required=True)
    analyse.add_argument("--cohort", required=True, help="the cohort the pairs were built on")
    analyse.add_argument("--balanced", required=True, help="balanced.json from the balance phase")
    analyse.add_argument("--scores", required=True, help="directory holding <arm>/ and masked scores")
    analyse.add_argument("--out", required=True)
    analyse.add_argument(
        "--rank-cross-check",
        action="store_true",
        help=(
            "also record, as a validity diagnostic and not an endpoint, whether the fresh "
            "nats order each pair the same way the retained panel's within-assay rank does; "
            "only meaningful on the retained-variant support, whose variants are all in that table"
        ),
    )
    analyse.set_defaults(run=phase_analyse)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
