"""E21/E22 matched-discrimination cohort, endpoints and decision rule.

The tests fall in three groups: the invariants the declared construction must
satisfy on any input, the refusals that have to fire when an input cannot
support the endpoint, and one regression on the real retained sample table that
pins the pre-registered cohort so a later edit cannot move it in silence.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.capability.mutation import matched_discrimination as md  # noqa: E402

SAMPLES = ROOT / "manuscript/data/prediction-details/mutation-samples.csv.gz"

#: The construction frozen by configs/matched_discrimination_manifest.json, as
#: first computed before any model score was read. Any change to the pair rule,
#: the gap, the split or the greedy order moves these numbers.
FROZEN = {
    "profile_match_threshold": 0.3519594896367262,
    "candidates": {"pair_rule_pairs": 5013, "phenotype_gap_pairs": 1670,
                   "development_gap_pairs": 549, "profile_matched_pairs": 319},
    "split": {"n_clusters": 163, "n_development": 49, "n_evaluation": 114},
    "T_open": {"development": 396, "evaluation": 882},
    "T_match": {"development": 96, "evaluation": 190},
}


def synthetic_samples(
    *, assays=3, positions=(10, 20, 30, 40), measured=None, profile=None, group_of=None
) -> pd.DataFrame:
    """A common-support-shaped table whose pairs the pair rule admits."""

    rows = []
    for index in range(assays):
        assay = f"ASSAY_{index}"
        group = index if group_of is None else group_of(index)
        for order, position in enumerate(positions):
            value = (order if measured is None else measured[order])
            rows.append({
                "sample_id": f"s{index}_{position}",
                "unit_id": assay,
                "group_id": group,
                "family_id": group,
                "mutation": f"A{position}V",
                "positions_1based": str(position),
                "measured": float(value),
                "profile_score": float(order if profile is None else profile[order]),
                "rank_target": 0.0,
                "supporting_common_member": True,
                "variant_index": order,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the pair rule and the tiers
# --------------------------------------------------------------------------- #


def test_candidate_pairs_match_identity_length_and_composition_by_construction():
    pairs = md.candidate_pairs(md.single_substitutions(md.common_support(synthetic_samples())))
    assert len(pairs) == 3 * 6  # three assays, four variants, every unordered pair
    for row in pairs.itertuples(index=False):
        assert row.wildtype_residue == "A" and row.mutant_residue == "V"
        assert row.position_a != row.position_b
    assert pairs.groupby("assay")["group_id"].nunique().max() == 1


def test_pair_rule_never_crosses_an_assay_or_a_substitution_identity():
    table = synthetic_samples(assays=2)
    table.loc[table["sample_id"] == "s0_20", "mutation"] = "A20L"
    pairs = md.candidate_pairs(md.single_substitutions(md.common_support(table)))
    members = set(pairs["sample_a"]) | set(pairs["sample_b"])
    assert "s0_20" not in members  # the only A->L variant of its assay has no partner
    assert set(pairs["assay"]) == {"ASSAY_0", "ASSAY_1"}


def test_retained_pairs_satisfy_the_declared_gap_and_the_profile_match():
    cohort = md.build_cohort(synthetic_samples(
        assays=12, measured=(0.0, 1.0, 8.0, 9.0), profile=(0.0, 0.05, 2.0, 2.1)))
    threshold = cohort["profile_match_threshold"]["value"]
    for name, frame in cohort["tiers"].items():
        assert (frame["phenotype_gap"] >= md.PHENOTYPE_GAP).all()
        if name == "T_match":
            assert (frame["profile_distance"] <= threshold).all()


def test_each_variant_appears_at_most_once_inside_a_tier():
    cohort = md.build_cohort(synthetic_samples(assays=8, measured=(0.0, 1.0, 8.0, 9.0)))
    for frame in cohort["tiers"].values():
        members = list(frame["sample_a"]) + list(frame["sample_b"])
        assert len(members) == len(set(members))


def test_the_two_tiers_are_not_assumed_nested():
    cohort = md.build_cohort(synthetic_samples(
        assays=8, measured=(0.0, 1.0, 8.0, 9.0), profile=(0.0, 0.05, 2.0, 2.1)))
    overlap = cohort["tier_overlap"]
    assert overlap["T_match_pairs_also_in_T_open"] <= overlap["T_match_pairs"]


# --------------------------------------------------------------------------- #
# the split and the threshold
# --------------------------------------------------------------------------- #


def test_cluster_split_is_deterministic_disjoint_and_exhaustive():
    clusters = list(range(50))
    first = md.cluster_split(clusters)
    second = md.cluster_split(clusters)
    assert first == second
    assert set(first["development"]) & set(first["evaluation"]) == set()
    assert sorted(first["development"] + first["evaluation"]) == clusters
    assert first["n_development"] == 15
    assert md.cluster_split(clusters, seed=1)["development"] != first["development"]


def test_profile_match_threshold_reads_development_candidates_only():
    candidates = pd.DataFrame({"profile_distance": np.arange(100, dtype=float),
                               "group_id": [0] * 100})
    assert md.profile_match_threshold(candidates) == pytest.approx(19.8)
    # Adding evaluation-cluster rows cannot move it: the function sees only what
    # the caller already restricted to development clusters.
    assert md.profile_match_threshold(candidates.iloc[:50]) != pytest.approx(19.8)


def test_profile_match_threshold_refuses_an_empty_development_set():
    with pytest.raises(ValueError, match="no development-cluster candidate"):
        md.profile_match_threshold(pd.DataFrame({"profile_distance": []}))


def odd_group_candidates() -> pd.DataFrame:
    """Three variants of one substitution identity: the greedy rule must choose."""

    table = synthetic_samples(assays=1, positions=(10, 20, 30), measured=(0.0, 5.0, 10.0))
    candidates = md.candidate_pairs(md.single_substitutions(md.common_support(table)))
    return candidates.loc[candidates["phenotype_gap"] >= md.PHENOTYPE_GAP]


def test_greedy_pairing_is_deterministic_under_its_seed():
    candidates = odd_group_candidates()
    assert len(candidates) == 3
    first = md.greedy_disjoint(candidates)
    second = md.greedy_disjoint(candidates.iloc[::-1])
    assert len(first) == 1  # three variants admit one disjoint pair
    assert list(first["sample_a"]) == list(second["sample_a"])
    assert list(first["sample_b"]) == list(second["sample_b"])


def test_which_pair_survives_depends_on_the_declared_seed_and_on_nothing_else():
    candidates = odd_group_candidates()
    chosen = {
        tuple(md.greedy_disjoint(candidates, seed=seed).iloc[0][["sample_a", "sample_b"]])
        for seed in range(12)
    }
    assert len(chosen) > 1
    # No column the rule reads carries a model score or a phenotype value.
    assert set(md.greedy_disjoint(candidates).columns) >= {
        "assay", "wildtype_residue", "mutant_residue", "position_a", "position_b"}


# --------------------------------------------------------------------------- #
# direction, ties and the resampler
# --------------------------------------------------------------------------- #


def pair_frame(better, worse, clusters):
    return pd.DataFrame({"better_sample": better, "worse_sample": worse, "group_id": clusters})


def test_direction_values_score_above_below_and_the_declared_tie_rule():
    frame = pair_frame(["a", "c", "e"], ["b", "d", "f"], [1, 1, 2])
    scores = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0, "e": 0.5, "f": 0.5}
    assert list(md.direction_values(frame, scores)) == [1.0, 0.0, 0.5]


def test_direction_values_refuse_a_member_without_a_score():
    frame = pair_frame(["a"], ["b"], [1])
    with pytest.raises(ValueError, match="no finite score"):
        md.direction_values(frame, {"a": 1.0})


def test_endpoint_point_estimate_tie_rate_and_pair_count():
    clusters = list(range(10))
    frame = pair_frame([f"g{i}" for i in clusters], [f"b{i}" for i in clusters], clusters)
    scores = {f"g{i}": 1.0 for i in clusters} | {f"b{i}": 0.0 for i in clusters}
    scores["g0"], scores["b0"] = 0.0, 0.0
    record = md.endpoint(md.ClusterResampler(clusters, draws=200), md.direction_values(frame, scores), clusters)
    assert record["point"] == pytest.approx(0.95)
    assert record["n_pairs"] == 10 and record["n_ties"] == 1
    assert record["tie_rate"] == pytest.approx(0.1)


def test_resampler_is_reproducible_and_pairs_two_row_sets_at_the_cluster():
    clusters = list(range(12))
    resampler = md.ClusterResampler(clusters, draws=300)
    left = md.endpoint(resampler, np.ones(12), clusters)
    right = md.endpoint(resampler, np.zeros(12), clusters)
    difference = md.contrast(resampler, left, right)
    assert difference["point"] == pytest.approx(1.0)
    assert difference["interval"] == [pytest.approx(1.0), pytest.approx(1.0)]
    repeated = md.ClusterResampler(clusters, draws=300)
    assert np.array_equal(resampler.index, repeated.index)


def test_resampler_withholds_an_interval_below_the_unit_floor():
    clusters = [0, 1, 2]
    record = md.endpoint(md.ClusterResampler(clusters, draws=50), np.array([1.0, 0.0, 1.0]), clusters)
    assert record["interval"] is None
    assert record["degenerate"] and "floor" in record["interval_reason"]
    assert record["verdict"]["verdict"] == "withheld"


def test_a_contrast_over_two_different_pair_sets_uses_one_cluster_draw():
    # Four clusters in the open tier, two of which also carry a matched pair.
    open_clusters = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    match_clusters = [0, 1, 2, 3, 4, 5, 6, 7]
    resampler = md.ClusterResampler(open_clusters, draws=400)
    wide = md.endpoint(resampler, np.ones(len(open_clusters)), open_clusters)
    narrow = md.endpoint(resampler, np.zeros(len(match_clusters)), match_clusters)
    record = md.contrast(resampler, wide, narrow)
    assert record["point"] == pytest.approx(1.0)
    assert record["n_units"] == len(open_clusters)
    assert narrow["n_units"] == len(match_clusters)


def test_within_assay_verification_refuses_a_pair_that_spans_two_assays():
    frame = pd.DataFrame({"sample_a": ["a"], "sample_b": ["b"], "assay": ["A"]})
    assert md.verify_within_assay(frame, {"a": "A", "b": "A"}) == 1
    with pytest.raises(ValueError, match="spans two assays"):
        md.verify_within_assay(frame, {"a": "A", "b": "B"})


# --------------------------------------------------------------------------- #
# the decision rule
# --------------------------------------------------------------------------- #


def test_level_verdicts_follow_the_declared_rule():
    assert md.level_verdict([0.52, 0.60])["verdict"] == "supported"
    assert md.level_verdict([0.30, 0.45])["verdict"] == "supported"
    assert md.level_verdict([0.47, 0.53])["verdict"] == "bounded_negative"
    assert md.level_verdict([0.44, 0.56])["verdict"] == "unresolved"
    assert md.level_verdict([0.48, 0.70])["verdict"] == "unresolved"
    assert md.level_verdict(None)["verdict"] == "withheld"


def test_bounded_negative_and_unresolved_are_never_merged():
    bounded = md.level_verdict([0.46, 0.54])
    unresolved = md.level_verdict([0.40, 0.54])
    assert bounded["verdict"] != unresolved["verdict"]
    assert bounded["within_practical_margin"] is True
    assert unresolved["within_practical_margin"] is False
    # An interval covering the null is not equivalence: only the bounded case
    # carries the margin claim, and neither carries an absence of effect.
    assert bounded["excludes_null"] is False and unresolved["excludes_null"] is False


def test_an_interval_can_exclude_the_null_and_still_sit_inside_the_margin():
    record = md.level_verdict([0.455, 0.49])
    assert record["verdict"] == "supported"
    assert record["within_practical_margin"] is True
    assert record["direction"] == "below"


def test_contrast_verdicts_use_the_same_margin_against_zero():
    assert md.contrast_verdict([0.01, 0.20])["verdict"] == "supported"
    assert md.contrast_verdict([-0.03, 0.04])["verdict"] == "bounded_negative"
    assert md.contrast_verdict([-0.20, 0.04])["verdict"] == "unresolved"
    assert md.contrast_verdict([0.01, 0.20])["margin"] == md.PRACTICAL_MARGIN


def test_across_seed_conjunction_requires_every_seed():
    def record(interval, level=True):
        verdict = md.level_verdict(interval) if level else md.contrast_verdict(interval)
        return {"point": float(np.mean(interval)), "interval": list(interval), "verdict": verdict}

    supported = {seed: record([0.60, 0.70]) for seed in md.FOLD_SEEDS}
    assert md.across_seeds(supported)["verdict"] == "supported"
    assert md.across_seeds(supported)["direction"] == "above"
    bounded = {seed: record([0.47, 0.53]) for seed in md.FOLD_SEEDS}
    assert md.across_seeds(bounded)["verdict"] == "bounded_negative"
    mixed = dict(zip(md.FOLD_SEEDS, [record([0.60, 0.70]), record([0.47, 0.53]), record([0.60, 0.70])]))
    assert md.across_seeds(mixed)["verdict"] == "unresolved"
    opposed = dict(zip(md.FOLD_SEEDS, [record([0.60, 0.70]), record([0.30, 0.40]), record([0.60, 0.70])]))
    assert md.across_seeds(opposed)["verdict"] == "unresolved"


# --------------------------------------------------------------------------- #
# refusals on inputs that cannot support the endpoint
# --------------------------------------------------------------------------- #


def test_common_support_refuses_a_missing_column():
    with pytest.raises(ValueError, match="sample table lacks"):
        md.common_support(pd.DataFrame({"sample_id": ["a"]}))


def test_common_support_refuses_a_constant_assay():
    table = synthetic_samples(assays=1, measured=(1.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="zero measured variance"):
        md.common_support(table)


def test_common_support_refuses_an_assay_that_spans_two_clusters():
    table = synthetic_samples(assays=1)
    table.loc[0, ["group_id", "family_id"]] = 99
    with pytest.raises(ValueError, match="more than one family cluster"):
        md.common_support(table)


def test_single_substitutions_refuse_a_position_that_disagrees_with_the_mutation():
    table = synthetic_samples(assays=1)
    table.loc[0, "positions_1based"] = "999"
    with pytest.raises(ValueError, match="disagrees with positions_1based"):
        md.single_substitutions(md.common_support(table))


def test_family_mean_spearman_averages_by_cluster_and_drops_a_constant_assay():
    predictions = np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    target = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    assays = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"])
    clusters = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2])
    record = md.family_mean_spearman(predictions, target, assays, clusters)
    assert record["n_assays"] == 2 and record["excluded_assays"] == 1
    assert record["point"] == pytest.approx(0.0)  # cluster 1 averages +1 and -1


# --------------------------------------------------------------------------- #
# the frozen cohort on the real retained table
# --------------------------------------------------------------------------- #


def test_the_declared_cohort_on_the_retained_table_is_unchanged():
    cohort = md.build_cohort(pd.read_csv(SAMPLES))
    assert cohort["profile_match_threshold"]["value"] == pytest.approx(
        FROZEN["profile_match_threshold"], abs=1e-15)
    for key, value in FROZEN["candidates"].items():
        assert cohort["candidates"][key] == value
    for key, value in FROZEN["split"].items():
        assert cohort["split"][key] == value
    for tier in md.TIERS:
        for stratum, value in FROZEN[tier].items():
            assert cohort["counts"][tier][stratum]["pairs"] == value
    assert cohort["support"] == {
        "variants": 25728, "assays": 201, "clusters": 163,
        "single_substitution_variants": 20568, "single_substitution_assays": 195,
        "single_substitution_clusters": 160,
    }


# --------------------------------------------------------------------------- #
# amendment_03: position control, sign-balanced tier, practical size
# --------------------------------------------------------------------------- #


def position_pairs() -> pd.DataFrame:
    """A pair frame with the columns the position machinery needs."""

    return pd.DataFrame({
        "assay": ["A", "A", "A"],
        "group_id": [1, 1, 2],
        "stratum": ["evaluation"] * 3,
        "wildtype_residue": ["A"] * 3,
        "mutant_residue": ["V"] * 3,
        "sample_a": ["a1", "a2", "a3"],
        "sample_b": ["b1", "b2", "b3"],
        "position_a": [2, 50, 10],
        "position_b": [50, 98, 91],
        "better_sample": ["a1", "b2", "a3"],
        "worse_sample": ["b1", "a2", "b3"],
        "better_profile": [1.0, 0.0, 0.5],
        "worse_profile": [0.0, 1.0, 0.5],
        "phenotype_gap": [1.0, 2.0, 4.0],
        "profile_distance": [1.0, 1.0, 0.0],
    })


def test_relative_midpoint_distance_is_zero_at_the_midpoint_and_one_at_a_terminus():
    assert md.relative_midpoint_distance([1, 100], [100, 100]).tolist() == [1.0, 1.0]
    assert md.relative_midpoint_distance([50, 51], [100, 100])[0] == pytest.approx(1 / 99)
    assert md.relative_midpoint_distance([3], [5]) == pytest.approx(0.0)


def test_relative_position_refuses_a_site_outside_its_wild_type():
    with pytest.raises(ValueError, match="outside its wild type"):
        md.relative_midpoint_distance([101], [100])
    with pytest.raises(ValueError, match="outside its wild type"):
        md.relative_midpoint_distance([0], [100])


def test_the_two_declared_position_rules_order_every_pair_identically():
    # Both members of a pair share one wild-type length, so terminal distance and
    # midpoint distance are affine images of each other inside the pair.
    frame = md.attach_positions(position_pairs(), {"A": 100})
    first = md.position_direction_values(frame, "nearer_terminus")
    second = md.position_direction_values(frame, "further_from_midpoint")
    assert first.tolist() == second.tolist()
    # Site 2 against site 50: the better member is at the terminus, so correct.
    assert first[0] == 1.0
    # Site 50 against site 98: the better member (b2, site 98) is nearer a terminus.
    assert first[1] == 1.0
    # Sites 10 and 91 of a 100-residue wild type are both nine residues from the
    # nearer terminus, so neither is preferred and the declared tie rule applies.
    assert first[2] == 0.5


def test_position_rules_read_no_phenotype_profile_or_model_column():
    frame = md.attach_positions(position_pairs(), {"A": 100})
    values = md.position_direction_values(frame, "nearer_terminus")
    altered = frame.copy()
    altered["phenotype_gap"] = 9.9
    altered["better_profile"] = -5.0
    assert md.position_direction_values(altered, "nearer_terminus").tolist() == values.tolist()
    with pytest.raises(ValueError, match="not a declared position rule"):
        md.position_direction_values(frame, "prefer_the_model")


def test_attach_positions_works_on_a_candidate_frame_without_a_better_member():
    candidates = md.candidate_pairs(md.single_substitutions(md.common_support(synthetic_samples())))
    frame = md.attach_positions(candidates, {f"ASSAY_{i}": 100 for i in range(3)})
    assert "position_distance" in frame.columns
    assert "better_midpoint_distance" not in frame.columns
    assert (frame["position_distance"] >= 0).all()


def test_attach_positions_refuses_an_assay_without_a_length():
    with pytest.raises(ValueError, match="no wild-type length"):
        md.attach_positions(position_pairs(), {"B": 100})


def test_balanced_tier_cancels_the_profile_vote_exactly():
    frame = pd.DataFrame({
        "assay": ["A"] * 7,
        "group_id": [1, 1, 2, 2, 3, 3, 4],
        "stratum": ["evaluation"] * 7,
        "wildtype_residue": ["A"] * 7,
        "mutant_residue": ["V"] * 7,
        "position_a": [1, 2, 3, 4, 5, 6, 7],
        "position_b": [11, 12, 13, 14, 15, 16, 17],
        "sample_a": [f"a{i}" for i in range(7)],
        "sample_b": [f"b{i}" for i in range(7)],
        "better_sample": [f"a{i}" for i in range(7)],
        "worse_sample": [f"b{i}" for i in range(7)],
        # four profile-correct, two profile-incorrect, one tie
        "better_profile": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5],
        "worse_profile": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5],
        "phenotype_gap": [1.5] * 7,
        "profile_distance": [1.0] * 6 + [0.0],
    })
    balanced = md.balance_profile_vote(frame)
    values = md.profile_direction_values(balanced)
    assert float(values.mean()) == pytest.approx(0.5)
    assert int(np.count_nonzero(values == 1.0)) == int(np.count_nonzero(values == 0.0)) == 2
    assert int(np.count_nonzero(values == 0.5)) == 1  # every tie is retained
    assert len(balanced) == 5
    # A subset of its input, variant-disjointness inherited, and deterministic.
    assert set(balanced["sample_a"]) <= set(frame["sample_a"])
    assert list(md.balance_profile_vote(frame)["sample_a"]) == list(balanced["sample_a"])
    assert (balanced["tier"] == "T_balanced").all()


def test_balanced_tier_balances_inside_each_stratum():
    frame = pd.DataFrame({
        "assay": ["A"] * 6,
        "group_id": [1, 1, 1, 2, 2, 2],
        "stratum": ["development"] * 3 + ["evaluation"] * 3,
        "wildtype_residue": ["A"] * 6,
        "mutant_residue": ["V"] * 6,
        "position_a": [1, 2, 3, 4, 5, 6],
        "position_b": [11, 12, 13, 14, 15, 16],
        "sample_a": [f"a{i}" for i in range(6)],
        "sample_b": [f"b{i}" for i in range(6)],
        "better_sample": [f"a{i}" for i in range(6)],
        "worse_sample": [f"b{i}" for i in range(6)],
        "better_profile": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        "worse_profile": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        "phenotype_gap": [1.5] * 6,
        "profile_distance": [1.0] * 6,
    })
    balanced = md.balance_profile_vote(frame)
    for stratum in ("development", "evaluation"):
        part = balanced.loc[balanced["stratum"] == stratum]
        assert float(md.profile_direction_values(part).mean()) == pytest.approx(0.5)
        assert len(part) == 2


def test_match_threshold_is_generic_over_the_matched_statistic():
    frame = pd.DataFrame({"position_distance": np.arange(100, dtype=float),
                          "profile_distance": np.arange(100, dtype=float) * 2})
    assert md.match_threshold(frame, "position_distance") == pytest.approx(19.8)
    assert md.match_threshold(frame, "profile_distance") == pytest.approx(39.6)
    assert md.profile_match_threshold(frame) == pytest.approx(39.6)


def test_selection_quality_is_bounded_by_chance_and_the_oracle():
    gaps = np.array([1.0, 2.0, 4.0, 1.0])
    perfect = md.selection_quality(np.ones(4), gaps)
    assert perfect["gap_capture"] == pytest.approx(1.0)
    assert perfect["selected_advantage_sd"] == pytest.approx(perfect["selected_advantage_oracle_sd"])
    assert perfect["share_of_chance_to_oracle"] == pytest.approx(1.0)
    chance = md.selection_quality(np.full(4, 0.5), gaps)
    assert chance["gap_capture"] == pytest.approx(0.5)
    assert chance["selected_advantage_sd"] == pytest.approx(0.0)
    assert chance["share_of_chance_to_oracle"] == pytest.approx(0.0)
    # Accuracy and gap capture are different quantities: being right on the small
    # gaps and wrong on the large one is above chance on one and below on the other.
    skewed = md.selection_quality(np.array([1.0, 1.0, 0.0, 1.0]), gaps)
    assert skewed["accuracy"] == pytest.approx(0.75)
    assert skewed["gap_capture"] == pytest.approx(0.5)
    assert skewed["status"].startswith("descriptive")


def test_selection_quality_refuses_misaligned_or_empty_input():
    with pytest.raises(ValueError, match="align and be non-empty"):
        md.selection_quality(np.ones(3), np.ones(2))
    with pytest.raises(ValueError, match="align and be non-empty"):
        md.selection_quality(np.array([]), np.array([]))


def test_the_position_rules_agree_on_the_real_retained_pairs():
    samples = pd.read_csv(SAMPLES)
    cohort = md.build_cohort(samples)
    reference = pd.read_csv(ROOT / "data/proteingym_raw/DMS_substitutions.csv",
                            usecols=["DMS_id", "seq_len"]).set_index("DMS_id")["seq_len"]
    lengths = {assay: int(reference[assay]) for assay in cohort["tiers"]["T_open"]["assay"].unique()}
    frame = md.attach_positions(cohort["tiers"]["T_open"], lengths)
    first = md.position_direction_values(frame, "nearer_terminus")
    second = md.position_direction_values(frame, "further_from_midpoint")
    assert np.array_equal(first, second)
    assert ((frame["better_midpoint_distance"] >= 0) & (frame["better_midpoint_distance"] <= 1)).all()


def test_every_retained_pair_sits_in_one_assay_and_one_cluster():
    samples = pd.read_csv(SAMPLES)
    support = md.common_support(samples)
    cohort = md.build_cohort(samples)
    assay_of = dict(zip(support["sample_id"], support["unit_id"], strict=True))
    cluster_of = dict(zip(support["sample_id"], support["group_id"], strict=True))
    for frame in cohort["tiers"].values():
        assert md.verify_within_assay(frame, assay_of) == len(frame)
        for row in frame.itertuples(index=False):
            assert cluster_of[row.sample_a] == cluster_of[row.sample_b] == row.group_id
