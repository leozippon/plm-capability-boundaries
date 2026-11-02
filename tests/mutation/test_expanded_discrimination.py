"""Pair construction, thresholds and endpoints of support ``S2_expanded``.

CPU only: no checkpoint, no GPU, no likelihood is computed here. What is tested
is the part of ``matched_discrimination_20261102`` that must hold whatever the
models turn out to do -- the pair rule's matching guarantee, the thresholds
being fixed on development clusters alone, variant disjointness, the declared
decision rule at its boundaries, and the refusals that keep a wrong coordinate
frame or a constant assay from producing a number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.mutation import expanded_discrimination as E  # noqa: E402

MANIFEST = json.loads((REPO_ROOT / "configs/matched_discrimination_manifest.json").read_text())


def row(assay, position, z, profile, *, cluster=0, wt="A", mut="G"):
    return {
        "assay": assay,
        "cluster": cluster,
        "wildtype_id": "q00000",
        "wt_residue": wt,
        "mut_residue": mut,
        "position": position,
        "z": z,
        "profile": profile,
    }


# ------------------------------------------------- the declaration is the source


def test_declared_constants_match_the_committed_manifest():
    cohort = MANIFEST["cohort"]
    inference = MANIFEST["inference"]
    assert E.GAP_THRESHOLD == cohort["phenotype_gap_threshold"]["value"]
    assert E.SPLIT_SEED == cohort["development_evaluation_split"]["seed"]
    assert [E.DEVELOPMENT_FRACTION, 1 - E.DEVELOPMENT_FRACTION] == pytest.approx(
        cohort["development_evaluation_split"]["fractions"]
    )
    assert E.BOOTSTRAP_DRAWS == inference["draws"]
    assert E.PRACTICAL_MARGIN == inference["practical_margin"]["value"]
    assert E.DECLARATION_ID == MANIFEST["declaration"]["id"]


def test_the_nine_declared_arms_are_exactly_the_manifest_arms():
    declared = set(MANIFEST["arms"]["autoregressive_protein"]) | set(
        MANIFEST["arms"]["different_scoring_rule_contrast"]
    )
    assert set(E.DECLARED_ARMS) == declared
    assert len(E.DECLARED_ARMS) == 9
    masked = [name for name, row in E.DECLARED_ARMS.items() if row["arm"] == E.MASKED_ARM]
    assert masked == [E.MASKED_MANIFEST_ARM]
    assert E.DECLARED_ARMS[E.MASKED_MANIFEST_ARM]["family"] == "different_scoring_rule_contrast"


# ----------------------------------------------------------------- the pair rule


def test_a_retained_pair_matches_length_and_composition_exactly():
    """The scientific guarantee the cohort is built on, checked on real sequences."""

    wildtype = "MAKLVAGSTAYQW"
    pairs = E.candidate_pairs(
        [row("assay", 2, 2.0, 0.0), row("assay", 10, 0.0, 0.1)]
    )
    assert len(pairs) == 1
    pair = pairs[0]
    built = []
    for position in (pair.position_better, pair.position_worse):
        assert wildtype[position - 1] == pair.wt_residue
        built.append(
            wildtype[: position - 1] + pair.mut_residue + wildtype[position:]
        )
    assert len(built[0]) == len(built[1]) == len(wildtype)
    assert sorted(built[0]) == sorted(built[1])
    assert built[0] != built[1]


def test_the_gap_threshold_is_inclusive_and_refuses_below_it():
    assert len(E.candidate_pairs([row("a", 1, 1.0, 0.0), row("a", 2, 0.0, 0.0)])) == 1
    assert E.candidate_pairs([row("a", 1, 0.999, 0.0), row("a", 2, 0.0, 0.0)]) == []


def test_pairs_need_the_same_assay_and_the_same_substituted_residues():
    rows = [
        row("a", 1, 2.0, 0.0),
        row("b", 2, -2.0, 0.0),
        row("a", 3, -2.0, 0.0, mut="V"),
        row("a", 4, -2.0, 0.0, wt="C"),
    ]
    assert E.candidate_pairs(rows) == []


def test_the_better_member_is_the_higher_phenotype_whichever_order_it_arrives_in():
    for first, second in ((1, 2), (2, 1)):
        pairs = E.candidate_pairs(
            [row("a", first, 3.0, 0.5), row("a", second, 0.0, -0.5)]
        )
        assert len(pairs) == 1
        assert pairs[0].position_better == first
        assert pairs[0].z_better > pairs[0].z_worse
        assert pairs[0].gap == pytest.approx(3.0)
        assert pairs[0].profile_distance == pytest.approx(1.0)


def test_every_pair_of_a_group_of_three_is_a_candidate():
    rows = [row("a", 1, 2.0, 0.0), row("a", 2, 0.0, 0.0), row("a", 3, -2.0, 0.0)]
    assert len(E.candidate_pairs(rows)) == 3


# ------------------------------------------------------------ phenotype refusals


def test_a_constant_assay_has_no_z_score_and_is_reported_rather_than_raised():
    assert E.within_assay_z([1.5]) is None
    assert E.within_assay_z([2.0, 2.0, 2.0]) is None
    assert E.within_assay_z([0.0, 2.0]) == pytest.approx([-1.0, 1.0])


def test_a_non_finite_phenotype_is_refused():
    with pytest.raises(ValueError):
        E.within_assay_z([0.0, float("nan"), 1.0])


# --------------------------------------------------------------- profile channel


def test_the_lookup_score_is_the_declared_log_odds():
    frequencies = np.zeros((3, len(AA20)), dtype=np.float64)
    frequencies[1, AA20.index("A")] = 0.4
    frequencies[1, AA20.index("G")] = 0.1
    background = np.full(len(AA20), 1.0 / len(AA20))
    scores = E.lookup_profile_scores(
        frequencies, background, [("A", 2, "G")], wildtype="MAK"
    )
    alpha = E.PROFILE_PSEUDOCOUNT / len(AA20)
    assert scores[0] == pytest.approx(np.log(0.1 + alpha) - np.log(0.4 + alpha))


def test_a_wild_type_residue_mismatch_is_refused_rather_than_scored():
    frequencies = np.full((3, len(AA20)), 0.05)
    background = np.full(len(AA20), 1.0 / len(AA20))
    with pytest.raises(ValueError, match="wild type carries"):
        E.lookup_profile_scores(
            frequencies, background, [("C", 2, "G")], wildtype="MAK"
        )
    with pytest.raises(ValueError, match="outside the wild type"):
        E.lookup_profile_scores(
            frequencies, background, [("K", 9, "G")], wildtype="MAK"
        )


def test_a_substitution_outside_the_twenty_residues_is_refused():
    frequencies = np.full((3, len(AA20)), 0.05)
    background = np.full(len(AA20), 1.0 / len(AA20))
    with pytest.raises(ValueError, match="outside AA20"):
        E.lookup_profile_scores(
            frequencies, background, [("A", 2, "X")], wildtype="MAK"
        )


# ------------------------------------------------------------------- the split


def test_the_split_is_a_deterministic_partition_of_the_declared_universe():
    universe = list(range(174))
    development = E.development_clusters(universe)
    assert development == E.development_clusters(reversed(universe))
    assert len(development) == 52
    assert development <= set(universe)


def test_the_assignment_depends_on_the_universe_not_on_the_support():
    """Why both supports must pass the declared universe rather than their own clusters.

    A permutation split is a function of the list it permutes, so passing only
    the clusters a support happens to contain would put the same family on
    different sides in the two supports. The caller therefore passes the declared
    174-cluster universe; this test pins that the shortcut really would differ.
    """

    universe = list(range(174))
    full = E.development_clusters(universe)
    subset = E.development_clusters([c for c in universe if c % 2 == 0])
    assert {c for c in universe if c % 2 == 0 and c in full} != subset


def test_a_split_that_would_empty_one_side_is_refused():
    with pytest.raises(ValueError):
        E.development_clusters([7])
    with pytest.raises(ValueError):
        E.development_clusters(range(10), fraction=0.01)


# ------------------------------------------------- thresholds and tier assembly


def test_the_profile_threshold_reads_development_clusters_only():
    """Moving an evaluation pair's profile distance must not move the threshold."""

    development = frozenset({0})

    def build(evaluation_distance):
        return E.candidate_pairs(
            [row("dev", 1, 2.0, 0.0, cluster=0), row("dev", 2, 0.0, 1.0, cluster=0)]
            + [row("dev", 3, 2.0, 0.0, cluster=0), row("dev", 4, 0.0, 3.0, cluster=0)]
            + [
                row("ev", 1, 2.0, 0.0, cluster=1),
                row("ev", 2, 0.0, evaluation_distance, cluster=1),
            ]
        )

    threshold = E.profile_match_threshold(build(99.0), development)
    assert threshold == E.profile_match_threshold(build(0.0), development)
    # The four development members form four gap-qualifying pairs, with
    # |delta s| of 1, 1, 3 and 3.
    assert threshold == pytest.approx(
        np.percentile([1.0, 1.0, 3.0, 3.0], E.PROFILE_MATCH_PERCENTILE)
    )


def test_the_threshold_is_undefined_without_a_development_candidate():
    candidates = E.candidate_pairs([row("ev", 1, 2.0, 0.0, cluster=1), row("ev", 2, 0.0, 1.0, cluster=1)])
    with pytest.raises(ValueError):
        E.profile_match_threshold(candidates, frozenset({0}))


def test_the_matched_tier_is_a_subset_selected_by_the_threshold():
    rows = []
    for index in range(12):
        rows.append(row(f"a{index}", 1, 2.0, 0.0, cluster=index))
        rows.append(row(f"a{index}", 2, 0.0, float(index), cluster=index))
    candidates = E.candidate_pairs(rows)
    development = frozenset(range(6))
    tiers = E.build_tiers(candidates, development)
    threshold = tiers["profile_match_threshold"]
    assert {p.key for p in tiers[E.TIER_MATCH]} <= {p.key for p in tiers[E.TIER_OPEN]}
    assert all(p.profile_distance <= threshold for p in tiers[E.TIER_MATCH])
    assert all(
        p.profile_distance > threshold
        for p in candidates
        if p.key not in {q.key for q in tiers[E.TIER_MATCH]}
    )


# ------------------------------------------------------------- disjointness rule


def test_a_variant_appears_at_most_once_inside_a_tier():
    rows = [row("a", position, float(position), 0.0) for position in range(1, 7)]
    candidates = E.candidate_pairs(rows)
    retained = E.retain_disjoint(candidates)
    seen = []
    for pair in retained:
        seen.extend([(pair.assay, pair.mutation_better), (pair.assay, pair.mutation_worse)])
    assert len(seen) == len(set(seen))
    assert len(retained) == 3


def test_the_same_variant_may_serve_one_pair_in_each_tier():
    rows = []
    for index in range(10):
        rows.append(row(f"a{index}", 1, 2.0, 0.0, cluster=index))
        rows.append(row(f"a{index}", 2, 0.0, 0.0, cluster=index))
    tiers = E.build_tiers(E.candidate_pairs(rows), frozenset(range(5)))
    overlap = {p.key for p in tiers[E.TIER_OPEN]} & {p.key for p in tiers[E.TIER_MATCH]}
    assert overlap


def test_retention_is_reproducible_and_capped_per_cluster():
    rows = [row("a", position, float(position % 3) * 2.0, 0.0) for position in range(1, 21)]
    candidates = E.candidate_pairs(rows)
    first = E.retain_disjoint(candidates)
    assert [p.key for p in first] == [p.key for p in E.retain_disjoint(candidates)]
    capped = E.retain_disjoint(candidates, cap_per_cluster=2)
    assert len(capped) == 2
    assert len(capped) < len(first)


def test_duplicate_candidate_pairs_are_refused():
    pair = E.candidate_pairs([row("a", 1, 2.0, 0.0), row("a", 2, 0.0, 0.0)])[0]
    with pytest.raises(ValueError, match="unique"):
        E.retain_disjoint([pair, pair])


# ------------------------------------------------------------------- direction


def test_an_exact_tie_counts_against_the_rule_and_is_counted():
    pairs = E.candidate_pairs([row("a", 1, 2.0, 0.0), row("a", 2, 0.0, 0.0)])
    indicators, ties = E.direction_indicators(pairs, [1.0], [1.0])
    assert indicators.tolist() == [0.0]
    assert ties == 1
    indicators, ties = E.direction_indicators(pairs, [2.0], [1.0])
    assert indicators.tolist() == [1.0]
    assert ties == 0


def test_a_non_finite_pair_score_is_refused():
    pairs = E.candidate_pairs([row("a", 1, 2.0, 0.0), row("a", 2, 0.0, 0.0)])
    with pytest.raises(ValueError):
        E.direction_indicators(pairs, [float("inf")], [0.0])
    with pytest.raises(ValueError):
        E.direction_indicators(pairs, [1.0, 2.0], [0.0, 1.0])


# --------------------------------------------------------------- decision rule


@pytest.mark.parametrize(
    "interval,expected",
    [
        ((0.55, 0.62), E.VERDICT_SUPPORTED),
        ((0.30, 0.49), E.VERDICT_SUPPORTED),
        ((0.46, 0.54), E.VERDICT_BOUNDED_NEGATIVE),
        ((0.451, 0.549), E.VERDICT_BOUNDED_NEGATIVE),
        ((0.45, 0.55), E.VERDICT_UNRESOLVED),
        ((0.40, 0.60), E.VERDICT_UNRESOLVED),
        ((0.44, 0.52), E.VERDICT_UNRESOLVED),
        ((0.48, 0.58), E.VERDICT_UNRESOLVED),
    ],
)
def test_the_declared_decision_rule_at_its_boundaries(interval, expected):
    assert E.verdict_of(*interval) == expected


def test_bounded_negative_and_unresolved_are_never_the_same_state():
    assert E.VERDICT_BOUNDED_NEGATIVE != E.VERDICT_UNRESOLVED
    assert E.verdict_of(0.49, 0.51) == E.VERDICT_BOUNDED_NEGATIVE
    assert E.verdict_of(0.20, 0.80) == E.VERDICT_UNRESOLVED


# ------------------------------------------------------------------- inference


def test_the_point_estimate_is_the_pooled_pair_fraction_not_a_cluster_mean():
    indicators = [1.0] * 10 + [0.0] * 2
    clusters = [0] * 10 + list(range(1, 3))
    record = E.group_bootstrap_accuracy(indicators, clusters, draws=200, seed=1)
    assert record["accuracy"] == pytest.approx(10 / 12)
    assert record["n_pairs"] == 12


def test_a_stratum_below_the_unit_floor_publishes_no_interval():
    record = E.group_bootstrap_accuracy([1.0, 0.0, 1.0], [0, 1, 2], draws=50, seed=1)
    assert record["degenerate"] is True
    assert record["interval"] is None
    assert record["verdict"] == E.VERDICT_DEGENERATE
    assert record["degenerate_reason"]
    empty = E.group_bootstrap_accuracy([], [], draws=50, seed=1)
    assert empty["interval"] is None and empty["n_pairs"] == 0


def test_a_perfectly_ordered_cohort_is_supported_and_a_coin_flip_is_not():
    clusters = [index // 4 for index in range(80)]
    ordered = E.group_bootstrap_accuracy([1.0] * 80, clusters, draws=500, seed=3)
    assert ordered["accuracy"] == 1.0
    assert ordered["verdict"] == E.VERDICT_SUPPORTED
    balanced = [1.0 if index % 2 else 0.0 for index in range(80)]
    coin = E.group_bootstrap_accuracy(balanced, clusters, draws=2000, seed=3)
    assert coin["accuracy"] == pytest.approx(0.5)
    assert coin["interval"][0] <= 0.5 <= coin["interval"][1]
    assert coin["verdict"] in (E.VERDICT_BOUNDED_NEGATIVE, E.VERDICT_UNRESOLVED)


def test_the_interval_is_reproducible_under_the_declared_seed():
    clusters = [index // 3 for index in range(60)]
    values = [float(index % 2) for index in range(60)]
    first = E.group_bootstrap_accuracy(values, clusters, draws=300, seed=11)
    second = E.group_bootstrap_accuracy(values, clusters, draws=300, seed=11)
    assert first["interval"] == second["interval"]
    other = E.group_bootstrap_accuracy(values, clusters, draws=300, seed=12)
    assert other["interval"] != first["interval"]


def test_the_tier_contrast_uses_only_clusters_present_in_both_tiers():
    open_clusters = list(range(12)) + [99]
    open_values = [1.0] * 12 + [0.0]
    match_clusters = list(range(12))
    match_values = [0.0] * 12
    record = E.paired_accuracy_contrast(
        open_values, open_clusters, match_values, match_clusters, draws=200, seed=5
    )
    assert record["n_pairs_left"] == 12
    assert record["left_accuracy_on_common_clusters"] == pytest.approx(1.0)
    assert record["right_accuracy_on_common_clusters"] == pytest.approx(0.0)
    assert record["difference"] == pytest.approx(1.0)
    assert record["excludes_zero"] is True


def test_the_tier_contrast_is_refused_below_the_unit_floor():
    record = E.paired_accuracy_contrast([1.0], [0], [0.0], [0], draws=50, seed=5)
    assert record["degenerate"] is True
    assert record["interval"] is None and record["difference"] is None


# ------------------------------------------------- magnitude-stratified reading


def test_quintile_boundaries_come_from_development_pairs_and_are_monotone():
    boundaries = E.magnitude_boundaries(list(range(100)))
    assert len(boundaries) == E.MAGNITUDE_BINS - 1
    assert boundaries == sorted(boundaries)
    with pytest.raises(ValueError):
        E.magnitude_boundaries([0.1, 0.2])


def test_every_evaluation_pair_lands_in_exactly_one_magnitude_bin():
    magnitudes = np.linspace(0.0, 10.0, 60)
    clusters = [index // 3 for index in range(60)]
    indicators = [float(index % 2) for index in range(60)]
    boundaries = E.magnitude_boundaries(magnitudes)
    strata = E.magnitude_strata(magnitudes, indicators, clusters, boundaries, draws=100, seed=7)
    assert len(strata) == E.MAGNITUDE_BINS
    assert sum(block["n_pairs"] for block in strata) == 60
    assert [block["bin_index"] for block in strata] == list(range(E.MAGNITUDE_BINS))
    assert strata[0]["lower_nats"] is None and strata[-1]["upper_nats"] is None


# ------------------------------------------------------- amendment_03 additions


def test_amendment_03_is_the_declaration_this_module_executes():
    amendment = MANIFEST["amendment_03"]
    assert "T_balanced" == amendment["addition_1_signed_balance"]["tier"] == E.TIER_BALANCED
    assert "position" in amendment["addition_3_position_control"]["endpoint"]
    # The amendment moves no threshold, margin or decision rule.
    assert E.PRACTICAL_MARGIN == MANIFEST["inference"]["practical_margin"]["value"]
    assert E.NULL_ACCURACY == 0.5


def test_amendment_04_declares_the_two_new_tiers_this_module_builds():
    amendment = MANIFEST["amendment_04"]
    assert amendment["addition_1_sign_balanced_position"]["tier"] == E.TIER_POSITION_BALANCED
    assert amendment["addition_2_double_balanced"]["tier"] == E.TIER_DOUBLE_BALANCED
    assert set(E.PRACTICAL_SHARE_DEFINITIONS) == {
        "accuracy_displacement_share",
        "phenotype_gap_capture_share",
    }
    assert "full mean pair gap" in E.ORACLE_CONVENTION


def test_the_two_position_spellings_are_one_rule_on_any_shared_length():
    """The geometric identity the amendment's two rules collapse to."""

    lengths = {"a": 11}
    pairs = []
    for better, worse in ((2, 6), (6, 2), (1, 5), (11, 4), (3, 9)):
        pairs.append(
            {
                "assay": "a",
                "cluster": 0,
                "position_better": better,
                "position_worse": worse,
                "profile_better": 0.0,
                "profile_worse": 0.0,
                "gap": 1.0,
            }
        )
    check = E.position_rule_agreement(pairs, lengths)
    assert check["orderings_disagreeing"] == 0
    terminus = E.position_rule_values(pairs, lengths, variant=E.POSITION_RULE_TERMINUS)
    midpoint = E.position_rule_values(pairs, lengths, variant=E.POSITION_RULE_MIDPOINT)
    assert np.sign(terminus[0] - terminus[1]).tolist() == np.sign(midpoint[0] - midpoint[1]).tolist()


def test_the_position_rule_prefers_the_member_nearer_a_terminus():
    lengths = {"a": 101}
    pair = {
        "assay": "a",
        "cluster": 0,
        "position_better": 3,
        "position_worse": 50,
        "profile_better": 0.0,
        "profile_worse": 0.0,
        "gap": 1.0,
    }
    better, worse = E.position_rule_values([pair], lengths)
    assert better[0] > worse[0]
    mirrored = dict(pair, position_better=3, position_worse=99)
    better, worse = E.position_rule_values([mirrored], lengths)
    assert better[0] == worse[0]  # exact mirror positions tie, and a tie counts zero


def test_the_position_rule_refuses_a_position_outside_the_wild_type():
    pair = {
        "assay": "a",
        "cluster": 0,
        "position_better": 500,
        "position_worse": 2,
        "profile_better": 0.0,
        "profile_worse": 0.0,
        "gap": 1.0,
    }
    with pytest.raises(ValueError, match="outside"):
        E.position_rule_values([pair], {"a": 100})



PROFILE = E.profile_vote
LENGTHS = {f"a{i}": 101 for i in range(400)}


def position(pair):
    return E.position_vote(pair, LENGTHS)


def voted_pair(index, profile_vote, position_vote, cluster=0):
    """A pair with the two votes set independently and nothing else varying."""

    delta = 1.0 if profile_vote > 0 else (-1.0 if profile_vote < 0 else 0.0)
    # Position 3 is nearer a terminus than position 50; swapping the two members
    # flips the position vote, and equal distances make it neutral.
    if position_vote > 0:
        better, worse = 3, 50
    elif position_vote < 0:
        better, worse = 50, 3
    else:
        better, worse = 3, 99
    return {
        "assay": f"a{index}",
        "cluster": cluster,
        "position_better": better,
        "position_worse": worse,
        "profile_better": delta,
        "profile_worse": 0.0,
        "gap": 1.5,
        "mutation_better": f"A{better}G",
        "mutation_worse": f"A{worse}G",
    }


def balanced_pair(index, vote, cluster=0):
    return voted_pair(index, vote, +1, cluster=cluster)


def test_the_balanced_tier_cancels_the_profile_vote_exactly():
    pairs = [balanced_pair(i, +1) for i in range(9)] + [balanced_pair(100 + i, -1) for i in range(4)]
    selected, record = E.balance_single_vote(pairs, PROFILE, name="profile")
    assert record["profile_concordant_available"] == 9
    assert record["profile_discordant_available"] == 4
    assert record["retained_per_side"] == 4
    assert len(selected) == 8
    assert E.assert_vote_balanced(selected, PROFILE, name="profile") == 0.5


def test_the_position_balanced_tier_cancels_the_position_vote_exactly():
    pairs = [voted_pair(i, +1, +1) for i in range(7)] + [voted_pair(50 + i, +1, -1) for i in range(3)]
    selected, record = E.balance_single_vote(pairs, position, name="position")
    assert record["position_concordant_available"] == 7
    assert record["position_discordant_available"] == 3
    assert len(selected) == 6
    assert E.assert_vote_balanced(selected, position, name="position") == 0.5
    # The profile is deliberately left uncontrolled on this tier.
    assert all(PROFILE(pair) > 0 for pair in selected)


def test_the_doubly_balanced_tier_cancels_both_votes_at_once():
    pairs = (
        [voted_pair(i, +1, +1) for i in range(10)]
        + [voted_pair(100 + i, -1, -1) for i in range(6)]
        + [voted_pair(200 + i, +1, -1) for i in range(5)]
        + [voted_pair(300 + i, -1, +1) for i in range(8)]
    )
    selected, record = E.balance_two_votes(
        pairs, PROFILE, position, names=("profile", "position")
    )
    assert record["retained_per_diagonal"] == [6, 5]
    assert record["retained_pairs"] == 22
    assert len(selected) == 22
    assert E.assert_vote_balanced(selected, PROFILE, name="profile") == 0.5
    assert E.assert_vote_balanced(selected, position, name="position") == 0.5


def test_the_diagonal_construction_keeps_more_than_levelling_four_cells():
    """Why the diagonals rather than the smallest cell: both are exactly balanced."""

    pairs = (
        [voted_pair(i, +1, +1) for i in range(20)]
        + [voted_pair(100 + i, -1, -1) for i in range(20)]
        + [voted_pair(200 + i, +1, -1) for i in range(2)]
        + [voted_pair(300 + i, -1, +1) for i in range(2)]
    )
    selected, record = E.balance_two_votes(
        pairs, PROFILE, position, names=("profile", "position")
    )
    assert record["retained_pairs"] == 44
    assert record["retained_pairs"] > 4 * 2
    assert E.assert_vote_balanced(selected, PROFILE, name="profile") == 0.5
    assert E.assert_vote_balanced(selected, position, name="position") == 0.5


def test_a_pair_neutral_on_either_vote_is_excluded_from_the_double_balance():
    pairs = (
        [voted_pair(i, +1, +1) for i in range(3)]
        + [voted_pair(100 + i, -1, -1) for i in range(3)]
        + [voted_pair(200 + i, 0, +1) for i in range(4)]
        + [voted_pair(300 + i, +1, 0) for i in range(2)]
    )
    selected, record = E.balance_two_votes(
        pairs, PROFILE, position, names=("profile", "position")
    )
    assert record["neutral_on_either_vote_excluded"] == 6
    assert len(selected) == 6


def test_profile_neutral_pairs_are_excluded_rather_than_kept():
    """Keeping them would pull the tier's profile accuracy below 0.5, not to it."""

    pairs = (
        [balanced_pair(i, +1) for i in range(3)]
        + [balanced_pair(10 + i, -1) for i in range(3)]
        + [balanced_pair(20 + i, 0) for i in range(4)]
    )
    selected, record = E.balance_single_vote(pairs, PROFILE, name="profile")
    assert record["profile_neutral_excluded"] == 4
    assert len(selected) == 6
    assert E.assert_vote_balanced(selected, PROFILE, name="profile") == 0.5
    with pytest.raises(ValueError, match="not exactly"):
        E.assert_vote_balanced(pairs, PROFILE, name="profile")


def test_an_unbalanced_tier_is_refused_rather_than_reported():
    pairs = [balanced_pair(i, +1) for i in range(3)] + [balanced_pair(10, -1)]
    with pytest.raises(ValueError, match="not exactly"):
        E.assert_vote_balanced(pairs, PROFILE, name="profile")
    with pytest.raises(ValueError):
        E.assert_vote_balanced([], PROFILE, name="profile")


def test_the_balanced_selection_is_seeded_and_reproducible():
    pairs = [balanced_pair(i, +1) for i in range(20)] + [
        balanced_pair(100 + i, -1) for i in range(5)
    ]
    first, _ = E.balance_single_vote(pairs, PROFILE, seed=7, name="profile")
    second, _ = E.balance_single_vote(pairs, PROFILE, seed=7, name="profile")
    other, _ = E.balance_single_vote(pairs, PROFILE, seed=8, name="profile")
    keys = lambda rows: [row["assay"] for row in rows]  # noqa: E731
    assert keys(first) == keys(second)
    assert keys(first) != keys(other)


def test_the_per_cluster_count_record_exposes_dominance():
    pairs = [balanced_pair(i, +1, cluster=0) for i in range(80)] + [
        balanced_pair(100 + i, +1, cluster=i + 1) for i in range(10)
    ]
    record = E.per_cluster_counts(pairs)
    assert record["clusters"] == 11
    assert record["pairs"] == 90
    assert record["max"] == 80
    assert record["min"] == 1
    assert record["largest_cluster_share"] == pytest.approx(80 / 90)
    assert E.per_cluster_counts([]) == {"clusters": 0}


def test_a_rule_versus_rule_contrast_is_paired_pair_by_pair():
    clusters = [index // 4 for index in range(60)]
    model = [1.0] * 60
    rule = [1.0 if index % 2 else 0.0 for index in range(60)]
    record = E.paired_accuracy_contrast(model, clusters, rule, clusters, draws=500, seed=9)
    assert record["n_pairs_left"] == record["n_pairs_right"] == 60
    assert record["difference"] == pytest.approx(0.5)
    assert record["excludes_zero"] is True
    assert record["left_accuracy_on_common_clusters"] == pytest.approx(1.0)


def test_a_rule_that_only_matches_the_model_leaves_no_margin():
    clusters = [index // 4 for index in range(60)]
    shared = [1.0 if index % 3 else 0.0 for index in range(60)]
    record = E.paired_accuracy_contrast(shared, clusters, shared, clusters, draws=500, seed=9)
    assert record["difference"] == pytest.approx(0.0)
    assert record["interval"] == [pytest.approx(0.0), pytest.approx(0.0)]
    assert record["excludes_zero"] is False


def test_the_two_practical_shares_are_separate_quantities():
    pairs = [balanced_pair(i, +1) for i in range(10)]
    perfect = E.practical_size(pairs, [1.0] * 10)
    assert perfect["accuracy_displacement_share"] == pytest.approx(1.0)
    assert perfect["phenotype_gap_capture_share"] == pytest.approx(1.0)
    assert perfect["selected_member_advantage_sd"] == pytest.approx(1.5)
    assert perfect["oracle_member_advantage_sd"] == pytest.approx(1.5)
    coin = E.practical_size(pairs, [1.0] * 5 + [0.0] * 5)
    assert coin["accuracy_displacement_share"] == pytest.approx(0.0)
    assert coin["phenotype_gap_capture_share"] == pytest.approx(0.0)
    assert E.practical_size([], [])["n_pairs"] == 0


def test_the_two_shares_diverge_when_errors_fall_on_the_wide_gap_pairs():
    """Equal accuracy, different realised phenotype: the reason both are reported."""

    narrow = [dict(balanced_pair(i, +1), gap=0.5) for i in range(5)]
    wide = [dict(balanced_pair(10 + i, +1), gap=4.0) for i in range(5)]
    pairs = narrow + wide
    errs_on_wide = E.practical_size(pairs, [1.0] * 5 + [0.0] * 5)
    errs_on_narrow = E.practical_size(pairs, [0.0] * 5 + [1.0] * 5)
    assert errs_on_wide["accuracy_displacement_share"] == pytest.approx(
        errs_on_narrow["accuracy_displacement_share"]
    )
    assert errs_on_wide["phenotype_gap_capture_share"] < 0
    assert errs_on_narrow["phenotype_gap_capture_share"] > 0


# --------------------------------------------- amendment_05 capacity axis


def test_the_executed_ladder_is_the_declared_arm_list_not_five_sizes():
    """The amendment says five sizes; the arm list it may not alter carries four."""

    declared = [
        arm for arm in MANIFEST["arms"]["autoregressive_protein"] if arm.startswith("progen2")
    ]
    assert list(E.CAPACITY_LADDER) == declared
    assert len(E.CAPACITY_LADDER) == 4
    assert "progen2-base" not in E.CAPACITY_LADDER
    assert set(E.CAPACITY_PARAMETERS) == set(E.CAPACITY_LADDER)
    counts = [E.CAPACITY_PARAMETERS[arm] for arm in E.CAPACITY_LADDER]
    assert counts == sorted(counts)
    assert E.PRACTICAL_MARGIN == MANIFEST["amendment_05"]["declared_margin_for_the_trend"]["value"]
    assert any("no scale/window contrast" in limit for limit in E.CAPACITY_LIMITS)


def ladder_indicators(accuracies, n_per_cluster=4, clusters=20):
    """One indicator vector per ladder arm, on one shared pair set.

    Correct answers are spread evenly over the pair set rather than packed at the
    front, so that two arms differing by a few pairs differ across many clusters,
    the way real arms do.
    """

    labels = [index // n_per_cluster for index in range(clusters * n_per_cluster)]
    total = len(labels)
    out = {}
    for arm, accuracy in zip(E.CAPACITY_LADDER, accuracies):
        correct = round(accuracy * total)
        chosen = set(np.linspace(0, total - 1, correct).round().astype(int).tolist())
        assert len(chosen) == correct
        out[arm] = [1.0 if index in chosen else 0.0 for index in range(total)]
    return out, labels


def test_a_falling_ladder_gives_a_negative_trend_and_a_resolved_contrast():
    indicators, labels = ladder_indicators([0.70, 0.65, 0.60, 0.50])
    record = E.capacity_trend(indicators, labels, draws=500, seed=3)
    assert record["spearman_vs_log_parameters"] == pytest.approx(-1.0)
    assert record["spearman_interval"][1] < 0.0
    assert record["spearman_excludes_zero"] is True
    contrast = record["largest_minus_smallest"]
    assert contrast["arms"] == ["progen2-xlarge", "progen2-small"]
    assert contrast["difference"] == pytest.approx(-0.20, abs=1e-9)
    assert contrast["verdict"] == E.VERDICT_SUPPORTED


def test_a_rising_ladder_gives_a_positive_trend():
    indicators, labels = ladder_indicators([0.50, 0.60, 0.65, 0.70])
    record = E.capacity_trend(indicators, labels, draws=500, seed=3)
    assert record["spearman_vs_log_parameters"] == pytest.approx(1.0)
    assert record["largest_minus_smallest"]["difference"] > 0


def test_a_perfectly_flat_ladder_withholds_the_coefficient_but_keeps_the_contrast():
    """An undefined rank trend must not be reported as a zero one."""

    indicators, labels = ladder_indicators([0.60, 0.60, 0.60, 0.60])
    record = E.capacity_trend(indicators, labels, draws=2000, seed=3)
    assert record["spearman_interval"] is None
    assert record["spearman_interval_withheld"]
    assert record["undefined_spearman_draws"] == 2000
    contrast = record["largest_minus_smallest"]
    assert contrast["difference"] == pytest.approx(0.0)
    assert contrast["verdict"] == E.VERDICT_BOUNDED_NEGATIVE
    assert contrast["interval"] == [pytest.approx(0.0), pytest.approx(0.0)]


def test_a_shallow_but_ordered_ladder_still_publishes_a_coefficient_interval():
    indicators, labels = ladder_indicators([0.60, 0.575, 0.55, 0.525], clusters=40)
    record = E.capacity_trend(indicators, labels, draws=500, seed=3)
    assert record["spearman_interval"] is not None
    assert record["largest_minus_smallest"]["verdict"] in (
        E.VERDICT_BOUNDED_NEGATIVE,
        E.VERDICT_UNRESOLVED,
        E.VERDICT_SUPPORTED,
    )


def test_the_trend_records_its_own_granularity_and_limits():
    indicators, labels = ladder_indicators([0.70, 0.65, 0.60, 0.50])
    record = E.capacity_trend(indicators, labels, draws=200, seed=3)
    assert record["spearman_distinct_values_without_ties"] == 11
    assert record["limits"] == list(E.CAPACITY_LIMITS)
    assert record["parameter_note"]


def test_the_trend_refuses_arms_read_on_different_pair_sets():
    indicators, labels = ladder_indicators([0.70, 0.65, 0.60, 0.50])
    indicators["progen2-xlarge"] = indicators["progen2-xlarge"][:-1]
    with pytest.raises(ValueError, match="same pairs"):
        E.capacity_trend(indicators, labels, draws=50, seed=3)


def test_the_trend_refuses_a_ladder_shorter_than_three_arms():
    indicators, labels = ladder_indicators([0.70, 0.65, 0.60, 0.50])
    pruned = {arm: indicators[arm] for arm in list(E.CAPACITY_LADDER)[:2]}
    with pytest.raises(ValueError, match="at least three"):
        E.capacity_trend(pruned, labels, draws=50, seed=3)


def test_the_trend_publishes_no_interval_below_the_cluster_floor():
    indicators, labels = ladder_indicators([0.70, 0.65, 0.60, 0.50], clusters=3)
    record = E.capacity_trend(indicators, labels, draws=50, seed=3)
    assert record["degenerate"] is True
    assert record["spearman_interval"] is None
    assert record["largest_minus_smallest"] is None


@pytest.mark.parametrize(
    "interval,expected",
    [
        ((0.01, 0.09), E.VERDICT_SUPPORTED),
        ((-0.20, -0.06), E.VERDICT_SUPPORTED),
        ((-0.04, 0.04), E.VERDICT_BOUNDED_NEGATIVE),
        ((-0.05, 0.05), E.VERDICT_UNRESOLVED),
        ((-0.12, 0.03), E.VERDICT_UNRESOLVED),
    ],
)
def test_the_difference_scale_decision_rule_keeps_the_two_states_apart(interval, expected):
    assert E.difference_verdict(*interval) == expected


def test_the_magnitude_distribution_is_reported_in_nats():
    summary = E.magnitude_summary([0.0, 0.05, 0.5, 1.0, 2.0])
    assert summary["n"] == 5
    assert summary["max_nats"] == 2.0
    assert summary["fraction_below_0.1_nats"] == pytest.approx(0.4)
    assert summary["fraction_below_1_nat"] == pytest.approx(0.6)
    assert set(summary["percentiles_nats"]) >= {"p50", "p95"}
    assert E.magnitude_summary([]) == {"n": 0}
