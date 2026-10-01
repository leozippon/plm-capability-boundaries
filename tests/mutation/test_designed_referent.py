"""The corpus-disjoint referent estimand: its grammar, its baselines, its rule.

Three failures here would produce numbers that look entirely normal, and each has
a test that targets it rather than the happy path.

*A variant that is not a substitution.* ``dataset2`` spells insertions,
deletions, wild-type rows and no-op tokens (``T38T``) in the same column as real
substitutions. Admitting one silently changes what the estimand is measured over,
and a no-op token would enter the free baselines as a substitution of a residue
for itself.

*The superseded k-mer background.* The corrected and the superseded counts are
both on disk. As a distribution they differ by 0.00028 total variation, so a
fragment baseline built on the wrong one reads plausibly -- while one 4-mer's
frequency is off by 42%. The refusal is by total, not by path.

*The pre-registered rule read as a disjunction.* A positive needs the arm to beat
**every** baseline, and a control failure must dominate whatever the design side
did. Both are asserted directly, including the case where the design side passes
and the control does not, which is the one that would otherwise be reported as a
discovery.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from src.capability.mutation import designed_referent as D
from src.capability.context.kmer_background import ALPHABET, KmerBackground, kmer_index


# ------------------------------------------------------------ variant grammar


@pytest.mark.parametrize(
    "mut_type",
    ["wt", "ins12A", "del12", "T38T", "E16E:A43A", "D28A:T32T", "", "12A", "A12"],
)
def test_ineligible_variants_are_refused(mut_type):
    assert D.eligible_substitutions(mut_type) is None


def test_eligible_variants_parse_to_the_shared_grammar():
    assert D.eligible_substitutions("A12G") == (("A", 12, "G"),)
    assert D.eligible_substitutions("A12G:C40W") == (("A", 12, "G"), ("C", 40, "W"))


def test_apply_substitutions_reconstructs_and_checks_the_wild_type():
    wild = "ACDEFGHIK"
    assert D.apply_substitutions(wild, (("C", 2, "W"), ("K", 9, "A"))) == "AWDEFGHIA"
    with pytest.raises(ValueError, match="mutation string says"):
        D.apply_substitutions(wild, (("W", 2, "A"),))
    with pytest.raises(ValueError, match="outside a wild type"):
        D.apply_substitutions(wild, (("A", 40, "G"),))


# ---------------------------------------------------------- the background gate


def _background(totals: dict[int, int], *, zero_a_kmer: bool = False) -> KmerBackground:
    counts = {}
    for k, total in totals.items():
        width = len(ALPHABET) ** k
        vector = np.full(width, total // width, dtype=np.int64)
        vector[0] += total - int(vector.sum())
        if zero_a_kmer:
            vector[1] += vector[2]
            vector[2] = 0
        counts[k] = vector
    return KmerBackground(
        counts=counts,
        residues=17_277_105_157,
        records=60_315_044,
        source=Path("uniref50.fasta"),
        source_bytes=1,
        wall_seconds=1.0,
    )


def test_the_corrected_background_is_accepted():
    record = D.require_corrected_background(_background(dict(D.CORRECTED_KMER_TOTALS)))
    assert record["totals"] == dict(D.CORRECTED_KMER_TOTALS)


def test_the_superseded_background_is_refused_by_its_total():
    with pytest.raises(ValueError, match="superseded line-local"):
        D.require_corrected_background(_background(dict(D.SUPERSEDED_KMER_TOTALS)))


def test_a_background_with_an_unobserved_kmer_is_refused():
    with pytest.raises(ValueError, match="observes"):
        D.require_corrected_background(
            _background(dict(D.CORRECTED_KMER_TOTALS), zero_a_kmer=True)
        )


# ------------------------------------------------------- the fragment baseline


def test_the_conditional_normalises_over_every_context():
    table = D.conditional_log_probabilities(
        _background(dict(D.CORRECTED_KMER_TOTALS)), 3
    )
    rows = np.exp(table).reshape(len(ALPHABET) ** 2, len(ALPHABET))
    assert np.allclose(rows.sum(axis=1), 1.0)


def test_a_uniform_background_scores_every_sequence_of_one_length_alike():
    table = D.conditional_log_probabilities(
        _background({3: len(ALPHABET) ** 3 * 1000}), 3
    )
    sequences = ["ACDEFGHIKL", "LKIHGFEDCA", "AAAAAAAAAA"]
    scores = D.fragment_log_likelihood(sequences, table, 3)
    assert np.allclose(scores, 8 * np.log(1.0 / len(ALPHABET)))


def test_the_fragment_score_is_the_sum_of_its_own_conditionals():
    background = _background(dict(D.CORRECTED_KMER_TOTALS))
    # Make one context strongly prefer one residue, so a hand-summed expectation
    # is not the uniform answer by accident.
    counts = background.counts[3]
    width = len(ALPHABET)
    context = ALPHABET.index("A") * width + ALPHABET.index("C")
    counts[context * width + ALPHABET.index("D")] += 10**9
    table = D.conditional_log_probabilities(background, 3)
    sequence = "ACDACD"
    expected = sum(
        table[
            ALPHABET.index(sequence[i]) * width**2
            + ALPHABET.index(sequence[i + 1]) * width
            + ALPHABET.index(sequence[i + 2])
        ]
        for i in range(len(sequence) - 2)
    )
    assert D.fragment_log_likelihood([sequence], table, 3)[0] == pytest.approx(expected)


def test_encoding_refuses_ragged_and_non_canonical_input():
    with pytest.raises(ValueError, match="same length"):
        D.encode_sequences(["ACDE", "ACD"])
    with pytest.raises(ValueError, match="canonical alphabet"):
        D.encode_sequences(["ACDX"])


# ------------------------------------------------------------------ statistics


def test_spearman_returns_none_rather_than_a_nan_on_a_constant_channel():
    assert D.spearman([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]) is None
    assert D.spearman([1.0, 2.0], [1.0, 2.0]) is None
    value = D.spearman([1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0])
    assert value == pytest.approx(
        float(stats.spearmanr([1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]).statistic)
    )


def test_the_unit_bootstrap_refuses_below_the_package_floor():
    record = D.unit_bootstrap([0.1] * 4, list("abcd"), resamples=100, seed=1)
    assert record["degenerate"] is True
    assert record["interval"] is None
    record = D.unit_bootstrap(
        [0.1] * 16, [f"u{i}" for i in range(16)], resamples=200, seed=1
    )
    assert record["degenerate"] is False
    assert record["interval"] is not None
    assert record["n_units"] == 16


def test_a_contrast_reports_the_wild_types_only_one_channel_could_be_read_on():
    units = {f"w{i}": f"u{i % 8}" for i in range(16)}
    model = {f"w{i}": 0.5 for i in range(16)}
    baseline = {f"w{i}": 0.1 for i in range(15)}
    record = D.channel_comparison(model, baseline, units, resamples=200, seed=3)
    assert record["n_wildtypes"] == 15
    assert record["n_wildtypes_dropped"] == 1
    assert record["beats_baseline"] is True


# ------------------------------------------------------------ the frozen rule


def _gates(**overrides: bool) -> dict[str, bool]:
    gates = {name: True for name in D.BASELINES}
    gates.update(overrides)
    return gates


def test_a_positive_needs_every_baseline_on_both_sides():
    assert D.arm_verdict(_gates(), _gates())["verdict"] == "positive"


def test_one_failed_baseline_on_the_designs_is_a_clean_negative():
    verdict = D.arm_verdict(_gates(fragment_markov_k4=False), _gates())
    assert verdict["verdict"] == "negative"
    assert verdict["design_failures"] == ["fragment_markov_k4"]


def test_a_failing_control_dominates_a_passing_design_side():
    verdict = D.arm_verdict(_gates(), _gates(blosum62=False))
    assert verdict["verdict"] == "uninterpretable_instrument_bound"
    assert verdict["beats_every_baseline_on_designs"] is True
    assert verdict["control_failures"] == ["blosum62"]


def test_the_verdict_refuses_a_partial_baseline_set():
    partial = {name: True for name in D.BASELINES[:-1]}
    with pytest.raises(ValueError, match="needs every baseline"):
        D.arm_verdict(partial, _gates())


# ------------------------------------------------------------------- the cohort


def _wildtype(name: str, kind: str, series: str, cluster: str, **overrides) -> D.WildType:
    fields = {
        "name": name,
        "kind": kind,
        "series": series,
        "cluster": cluster,
        "zero_hit": kind == "design",
        "series_zero_hit": kind == "design",
        "sequence": "ACDEFGHIKLMNPQRSTVWY",
        "mutants": ("A1C", "C2D", "D3E", "E4F"),
        "phenotype": np.array([0.5, -0.5, 1.5, -1.5]),
        "replicates": (1, 1, 2, 1),
    }
    fields.update(overrides)
    return D.WildType(**fields)


def test_a_wild_types_unit_is_its_series_or_its_cluster():
    assert _wildtype("d", "design", "HHH/rd1", "HHH").unit == "design:HHH/rd1"
    assert _wildtype("n", "natural", "-", "15").unit == "natural:15"


def test_the_cohort_round_trips_through_its_artefact(tmp_path):
    referent = D.Referent(
        wildtypes=(
            _wildtype("d0", "design", "HHH/rd1", "HHH"),
            _wildtype("n0", "natural", "-", "15"),
        ),
        provenance={"sampling": {"mode": "census"}},
    )
    path = tmp_path / "cohort.json"
    D.save_referent(referent, path)
    reloaded = D.load_referent(path)
    assert [wt.record() for wt in reloaded.wildtypes] == [
        wt.record() for wt in referent.wildtypes
    ]
    assert reloaded.provenance == referent.provenance


def test_a_cohort_artefact_of_the_wrong_schema_is_refused(tmp_path):
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps({"schema_version": "other", "wildtypes": []}))
    with pytest.raises(ValueError, match="not r2_designed_referent"):
        D.load_referent(path)


def test_the_design_side_excludes_the_designs_that_hit_the_corpus():
    referent = D.Referent(
        wildtypes=(
            _wildtype("clean", "design", "HHH/rd1", "HHH"),
            _wildtype(
                "hit", "design", "EEHEE/rd3", "EEHEE", zero_hit=False, series_zero_hit=False
            ),
            _wildtype("nat", "natural", "-", "15"),
        ),
        provenance={},
    )
    assert [wt.name for wt in referent.side("design")] == ["clean"]
    assert [wt.name for wt in referent.side("design", zero_hit_only=False)] == [
        "clean",
        "hit",
    ]
    assert [wt.name for wt in referent.side("natural")] == ["nat"]


def test_cohort_counts_track_the_variant_floor():
    wildtypes = (
        _wildtype("d0", "design", "HHH/rd1", "HHH"),
        _wildtype("d1", "design", "HHH/rd2", "HHH", mutants=("A1C",), phenotype=np.array([0.1]), replicates=(1,)),
        _wildtype("n0", "natural", "-", "15"),
    )
    assert D.cohort_counts(wildtypes)["design_series_scored"] == 2
    counts = D.cohort_counts(wildtypes, min_variants=4)
    assert counts["design_series_scored"] == 1
    assert counts["designs_zero_hit_scored"] == 1
    assert counts["naturals_scored"] == 1


def test_the_post_hoc_length_bands_are_taken_from_the_design_lengths():
    designs = [
        _wildtype(f"d{index}", "design", "s", "c", sequence="A" * length)
        for index, length in enumerate([40, 42, 44, 50, 60])
    ]
    assert D.design_length_bands(designs) == ((40, 44), (40, 50), (40, 60))
    with pytest.raises(ValueError, match="no designs"):
        D.design_length_bands([])


def test_a_wild_type_whose_arrays_disagree_is_refused():
    with pytest.raises(ValueError, match="disagree in length"):
        _wildtype("d", "design", "s", "c", phenotype=np.array([0.1]))
    with pytest.raises(ValueError, match="non-finite phenotype"):
        _wildtype("d", "design", "s", "c", phenotype=np.array([0.1, np.nan, 0.2, 0.3]))


# ------------------------------------------------------------------ baselines


def test_every_declared_baseline_is_actually_computed():
    background = _background(dict(D.CORRECTED_KMER_TOTALS))
    residues = np.full(20, 1.0 / 20.0)
    scores = D.baseline_scores(
        _wildtype("d", "design", "HHH/rd1", "HHH"),
        residue_background=residues,
        log_conditional={k: D.conditional_log_probabilities(background, k) for k in (3, 4)},
    )
    assert list(scores) == list(D.BASELINES)
    assert all(value.shape == (4,) for value in scores.values())
    # BLOSUM62 of A->C, C->D, D->E, E->F, read off the matrix rather than restated.
    assert scores["blosum62"].tolist() == [0.0, -3.0, 2.0, -3.0]


def test_the_fragment_baselines_see_the_mutation():
    background = _background(dict(D.CORRECTED_KMER_TOTALS))
    counts = background.counts[3]
    width = len(ALPHABET)
    counts[
        (ALPHABET.index("A") * width + ALPHABET.index("C")) * width + ALPHABET.index("D")
    ] += 10**9
    tables = {k: D.conditional_log_probabilities(background, k) for k in (3, 4)}
    scores = D.baseline_scores(
        _wildtype("d", "design", "HHH/rd1", "HHH"),
        residue_background=np.full(20, 1.0 / 20.0),
        log_conditional=tables,
    )
    # Only E4F leaves the enriched ACD window intact; A1C, C2D and D3E all
    # destroy it, so the fragment channel has to separate the fourth variant from
    # the other three rather than scoring on composition alone.
    values = scores["fragment_markov_k3"]
    assert values[3] > values[:3].max()


# ------------------------------------------------- the design/natural interaction
#
# The three ways this statistic goes wrong silently. It can be built from two
# separately resampled means, which gives an interval that is not an interval on
# the difference. It can drift away from the unit definition the unadjusted
# estimator uses once covariates are added, so the two answer different questions
# while looking like a sensitivity. And its four-way reading can be written so
# that underpower reports as refutation, which is the one failure this entry
# exists to avoid.


def test_every_scored_design_series_has_a_declared_family():
    for group, family in D.DESIGN_FAMILY_OF_GROUP.items():
        assert D.design_family(f"{group}/rd1") == family
        assert family in D.DESIGN_FAMILIES
    with pytest.raises(ValueError, match="no declared design family"):
        D.design_family("NOVEL/rd1")


def test_the_interaction_is_the_difference_of_unit_mean_averages():
    record = D.interaction_bootstrap(
        [1.0, 3.0, 10.0], ["u1", "u1", "u2"],
        [0.0, 2.0], ["v1", "v2"],
        resamples=64, seed=1,
    )
    # Unit means are (2.0, 10.0) and (0.0, 2.0): 6.0 - 1.0.
    assert record["point"] == pytest.approx(5.0)
    assert record["design_side"] == pytest.approx(6.0)
    assert record["natural_side"] == pytest.approx(1.0)


def test_the_interaction_is_refused_when_either_side_is_below_the_floor():
    many = ([1.0] * 12, [f"u{index}" for index in range(12)])
    few = ([0.0] * 4, [f"v{index}" for index in range(4)])
    for design, natural in ((many, few), (few, many)):
        record = D.interaction_bootstrap(*design, *natural, resamples=64, seed=1)
        assert record["degenerate"] is True
        assert record["interval"] is None
        assert record["point"] is not None
        assert D.interaction_outcome(record) == "unresolved"


def test_the_interval_is_taken_on_the_difference_and_not_on_two_means():
    # Both sides are the same numbers in the same order, so a joint draw that
    # recomputed the difference inside the replicate straddles zero. The failure
    # this catches is an interval assembled from two separately resampled means,
    # which on identical sides would be centred on zero but far too narrow only
    # if the draws were shared -- and far too wide if they were not. What must
    # hold is that the point is exactly zero and the interval contains it.
    values = list(np.linspace(-1.0, 1.0, 16))
    units = [f"u{index}" for index in range(16)]
    record = D.interaction_bootstrap(values, units, values, units, resamples=512, seed=7)
    assert record["point"] == pytest.approx(0.0)
    assert record["interval"][0] < 0.0 < record["interval"][1]
    assert record["excludes_zero"] is False


@pytest.mark.parametrize(
    "interval,point,expected",
    [
        ([-0.40, -0.20], -0.30, "confirms"),
        ([-0.12, -0.02], -0.07, "attenuated"),
        ([-0.10, 0.05], -0.02, "refutes"),
        ([0.10, 0.30], 0.20, "refutes"),
        ([-0.45, 0.05], -0.20, "unresolved"),
        (None, -0.30, "unresolved"),
    ],
)
def test_the_four_way_reading_separates_underpower_from_refutation(interval, point, expected):
    assert D.interaction_outcome({"interval": interval, "point": point}) == expected


def test_a_wide_interval_around_a_large_point_is_never_a_refutation():
    # The failure mode this programme keeps catching: an underpowered interval
    # read as absence. A point at the declared magnitude with an interval that
    # covers zero must come back unresolved, not refuted.
    assert (
        D.interaction_outcome({"interval": [-1.0, 0.3], "point": -0.5}) == "unresolved"
    )


def test_the_adjusted_interaction_reduces_to_the_unadjusted_one_without_covariates():
    values = [1.0, 3.0, 10.0, 0.0, 2.0]
    units = ["d1", "d1", "d2", "n1", "n2"]
    designed = [True, True, True, False, False]
    plain = D.interaction_bootstrap(
        values[:3], units[:3], values[3:], units[3:], resamples=32, seed=3
    )
    adjusted = D.adjusted_interaction_bootstrap(
        values, units, designed, np.zeros((5, 0)), resamples=32, seed=3
    )
    assert adjusted["point"] == pytest.approx(plain["point"])
    assert adjusted["n_design_units"] == 2
    assert adjusted["n_natural_units"] == 2


def test_the_adjusted_interaction_removes_a_covariate_that_explains_the_gap():
    # Designs and naturals differ only through the covariate: y = 2 * x, with the
    # designs sitting at high x. The unadjusted difference is large and the
    # adjusted coefficient is zero, which is exactly what a length adjustment has
    # to be able to do before it can be trusted to report a residual.
    x = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                  14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0])
    designed = list(x >= 14.0)
    values = (2.0 * x).tolist()
    units = [f"{'d' if flag else 'n'}{index}" for index, flag in enumerate(designed)]
    adjusted = D.adjusted_interaction_bootstrap(
        values, units, designed, x[:, None], resamples=64, seed=5
    )
    assert adjusted["point"] == pytest.approx(0.0, abs=1e-9)


def test_the_adjusted_interaction_refuses_a_unit_that_mixes_the_two_sides():
    with pytest.raises(ValueError, match="mixes designs and naturals"):
        D.adjusted_interaction_bootstrap(
            [1.0, 2.0], ["u1", "u1"], [True, False], np.zeros((2, 0)),
            resamples=8, seed=1,
        )


def test_balance_is_reported_rather_than_assumed_from_a_restriction():
    left = np.array([[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]])
    close = left + 0.05
    far = left + 10.0
    assert D.standardised_mean_differences(left, close, ["a", "b"])["balanced"] is True
    report = D.standardised_mean_differences(left, far, ["a", "b"])
    assert report["balanced"] is False
    assert report["max_abs_smd"] > 0.25


# ----------------------------------------- the higher-order fragment channel
#
# The surviving claim is a margin over 3-mer and 4-mer conditionals, so the
# channel that tests it has to be stronger *and* has to remain a model. Three
# failures here would produce a channel that looks stronger and is not: a
# smoothing scheme whose conditionals do not sum to one is not a likelihood and
# its perplexity is meaningless; a scheme that fails to back off assigns zero
# probability to an unseen k-mer and reports minus infinity as a score; and a
# channel evaluated only where the corpus has seen everything is a lookup whose
# sparsity is invisible. Each has a test.


def _toy_counts(records, ks, chunk_bytes=64):
    """Count a toy corpus through the production counter, not a reimplementation."""

    from tempfile import TemporaryDirectory

    from src.capability.context.kmer_background import count_kmers

    with TemporaryDirectory() as work:
        path = Path(work) / "toy.fasta"
        with path.open("w", encoding="ascii") as handle:
            for index, record in enumerate(records):
                handle.write(f">r{index}\n")
                for start in range(0, len(record), 60):
                    handle.write(record[start : start + 60] + "\n")
        return count_kmers(path, ks=ks, chunk_bytes=chunk_bytes).counts


def _toy_corpus(n_records=120, seed=11):
    rng = np.random.default_rng(seed)
    return [
        "".join(rng.choice(list(ALPHABET), size=int(rng.integers(45, 200))))
        for _ in range(n_records)
    ]


@pytest.mark.parametrize("scheme", D.FRAGMENT_SMOOTHING)
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_every_declared_smoothing_scheme_is_a_proper_distribution(scheme, order):
    counts = _toy_counts(_toy_corpus(), (1, 2, 3, 4))
    model = D.InterpolatedFragmentModel(counts, 4, scheme)
    contexts = np.random.default_rng(0).integers(0, len(ALPHABET) ** (order - 1), size=64)
    total = np.zeros(contexts.size)
    for symbol in range(len(ALPHABET)):
        total += np.exp(model.log_probability(contexts, np.full(contexts.size, symbol), order))
    assert np.allclose(total, 1.0, atol=1e-12)


@pytest.mark.parametrize("scheme", D.FRAGMENT_SMOOTHING)
def test_an_unseen_context_backs_off_rather_than_returning_minus_infinity(scheme):
    # Two order-4 contexts the corpus never saw, sharing their last two residues.
    # Both must fall back to the identical lower-order state, so the top-order
    # table cannot be contributing anything -- which is what "backed off" means,
    # and is the invariant that holds whichever lower-order distribution the
    # scheme backs off *to*.
    counts = _toy_counts(_toy_corpus(), (1, 2, 3, 4))
    width = len(ALPHABET)
    totals = counts[4].reshape(-1, width).sum(axis=1)
    unseen = np.flatnonzero(totals == 0)
    assert unseen.size, "the toy corpus is too dense to exercise the backoff"
    by_suffix: dict[int, list[int]] = {}
    for context in unseen.tolist():
        by_suffix.setdefault(context % (width**2), []).append(context)
    pairs = [group for group in by_suffix.values() if len(group) >= 2]
    assert pairs, "no two unseen contexts share a suffix in this toy corpus"
    left, right = pairs[0][0], pairs[0][1]
    model = D.InterpolatedFragmentModel(counts, 4, scheme)
    symbols = np.arange(width)
    first = model.log_probability(np.full(width, left), symbols, 4)
    second = model.log_probability(np.full(width, right), symbols, 4)
    assert np.isfinite(first).all()
    assert np.allclose(first, second)


@pytest.mark.parametrize("scheme", D.FRAGMENT_SMOOTHING)
def test_the_channel_scores_the_leading_residues_the_k_mer_baseline_drops(scheme):
    counts = _toy_counts(_toy_corpus(), (1, 2, 3, 4))
    model = D.InterpolatedFragmentModel(counts, 4, scheme)
    sequences = ["".join(np.random.default_rng(3).choice(list(ALPHABET), size=30))]
    record = model.evaluate(sequences, 4)
    # Every residue carries an emission term, and only the first three fall short
    # of the full order. fragment_log_likelihood would have scored 27.
    assert record["positions"] == 30
    assert record["positions_at_full_order"] == 27
    assert record["log_likelihood"].shape == (1,)
    assert np.isfinite(record["log_likelihood"]).all()


def test_the_channel_reports_the_sparsity_it_was_read_under():
    counts = _toy_counts(_toy_corpus(), (1, 2, 3, 4))
    counts[4][kmer_index("WWWW")] = 0
    model = D.InterpolatedFragmentModel(counts, 4, "kneser_ney")
    record = model.evaluate(["WWWWWWWW"], 4)
    assert record["unseen_kmer_fraction"] == 1.0
    assert np.isfinite(record["log_likelihood"]).all()
    seen = model.evaluate(_toy_corpus(4, seed=11), 4)
    assert seen["unseen_kmer_fraction"] == 0.0


def test_an_undeclared_smoothing_scheme_and_a_missing_order_are_refused():
    counts = _toy_counts(_toy_corpus(20), (1, 2, 3))
    with pytest.raises(ValueError, match="not one of"):
        D.fragment_channel_name(5, "add_one")
    with pytest.raises(ValueError, match="not one of"):
        D.InterpolatedFragmentModel(counts, 3, "add_one")
    with pytest.raises(ValueError, match="missing"):
        D.InterpolatedFragmentModel(counts, 4, "kneser_ney")
    assert D.fragment_channel_name(5, "kneser_ney") == "fragment_interp_k5_kneser_ney"


def test_a_higher_order_channel_fits_the_corpus_it_was_counted_on_better():
    # Plug-in perplexity falls with the order by construction, which is exactly
    # why admissibility is decided on held-out sequence and not on this. Asserted
    # so that a channel which failed to use its context would be caught.
    records = _toy_corpus(60, seed=7)
    counts = _toy_counts(records, (1, 2, 3, 4))
    model = D.InterpolatedFragmentModel(counts, 4, "kneser_ney")
    curve = [model.cross_entropy(records, order)["cross_entropy_nats"] for order in (1, 2, 3, 4)]
    assert curve == sorted(curve, reverse=True)


# ----------------------------------- the pre-registered rules of EXP-R2-196
#
# Two decision rules decide what the higher-order channel is allowed to say, and
# both live in the stage. The admissibility rule reads a turning point in
# held-out cross-entropy; the verdict rule reads the surviving claim against the
# admissible orders. Each is asserted against its own truth table here, including
# the case that would otherwise be reported as a discovery -- a channel that beats
# the model on the designs *and* on the natural control, which is a statement
# about the channel and not about the referent.


def _stage():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts/capability/stages/designed_referent.py"
    spec = importlib.util.spec_from_file_location("_stage_29_fragment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _curves(values):
    return {
        scheme: {
            draw: {str(order): {"cross_entropy_nats": value} for order, value in values.items()}
            for draw in ("a", "b")
        }
        for scheme in D.FRAGMENT_SMOOTHING
    }


def test_admissibility_stops_at_the_order_where_held_out_cross_entropy_turns():
    stage = _stage()
    falling = stage._supported_orders(_curves({1: 3.0, 2: 2.8, 3: 2.7, 4: 2.6}), 4)
    assert falling["highest_supported_order"] == 4
    assert falling["turned_at"] is None
    turned = stage._supported_orders(_curves({1: 3.0, 2: 2.8, 3: 2.7, 4: 2.75}), 4)
    assert turned["highest_supported_order"] == 3
    assert turned["turned_at"] == 4


def test_one_disagreeing_draw_or_scheme_stops_admissibility():
    stage = _stage()
    curves = _curves({1: 3.0, 2: 2.8, 3: 2.7, 4: 2.6})
    curves["kneser_ney"]["b"]["4"]["cross_entropy_nats"] = 2.71
    record = stage._supported_orders(curves, 4)
    assert record["highest_supported_order"] == 3
    assert record["turned_at"] == 4


def _arm_rows(design_intervals, control_intervals=None):
    control_intervals = control_intervals or {}
    def side(intervals):
        return {
            key: {"interval": value, "point": 0.0 if value is None else sum(value) / 2}
            for key, value in intervals.items()
        }
    return {
        D.ORIGIN_ARM: {
            "designs": side(design_intervals),
            "control": side(
                {key: control_intervals.get(key, [0.1, 0.3]) for key in design_intervals}
            ),
        }
    }


def _keys(orders):
    return [
        D.fragment_channel_name(order, scheme)
        for order in orders
        for scheme in D.FRAGMENT_SMOOTHING
    ]


def test_the_surviving_half_stands_only_when_every_admissible_channel_is_beaten():
    stage = _stage()
    rows = _arm_rows({key: [0.05, 0.15] for key in _keys([3, 4, 5])})
    verdict = stage._fragment_verdict(rows, arm=D.ORIGIN_ARM, admissible=[3, 4, 5])
    assert verdict["verdict"] == "stands"
    assert verdict["stronger_order_was_available"] is True
    assert not verdict["channels_that_beat_the_model"]


def test_an_interval_covering_zero_weakens_rather_than_overturns():
    stage = _stage()
    intervals = {key: [0.05, 0.15] for key in _keys([3, 4, 5])}
    intervals[D.fragment_channel_name(5, "kneser_ney")] = [-0.02, 0.09]
    verdict = stage._fragment_verdict(_arm_rows(intervals), arm=D.ORIGIN_ARM, admissible=[3, 4, 5])
    assert verdict["verdict"] == "weakened"
    assert verdict["channels_without_a_demonstrated_margin"] == [
        D.fragment_channel_name(5, "kneser_ney")
    ]


def test_a_channel_that_beats_the_model_on_both_sides_is_about_the_channel():
    stage = _stage()
    intervals = {key: [0.05, 0.15] for key in _keys([3, 4, 5])}
    beaten = D.fragment_channel_name(5, "kneser_ney")
    intervals[beaten] = [-0.20, -0.05]
    both = stage._fragment_verdict(
        _arm_rows(intervals, {beaten: [-0.18, -0.03]}), arm=D.ORIGIN_ARM, admissible=[3, 4, 5]
    )
    assert both["verdict"] == "overturned"
    assert "about the channel" in both["reading"]
    designs_only = stage._fragment_verdict(
        _arm_rows(intervals, {beaten: [0.10, 0.30]}), arm=D.ORIGIN_ARM, admissible=[3, 4, 5]
    )
    assert designs_only["verdict"] == "overturned"
    assert "about the referent" in designs_only["reading"]


def test_no_admissible_order_beyond_f12s_own_is_unresolved_rather_than_a_pass():
    stage = _stage()
    verdict = stage._fragment_verdict(_arm_rows({}), arm=D.ORIGIN_ARM, admissible=[])
    assert verdict["verdict"] == "unresolved"
    assert verdict["stronger_order_was_available"] is False


def test_the_turning_point_rule_finds_the_order_a_known_corpus_supports():
    # The positive control for the admissibility rule, on a corpus whose true
    # order is known by construction: sequences built from five-residue motifs,
    # so a four-residue context is the longest one that predicts anything and
    # order five is where the estimate stops being supported. Held-out
    # cross-entropy turns exactly there under both schemes and on both draws --
    # while the plug-in cross-entropy on the counted records keeps falling, which
    # is why admissibility is not decided on it.
    stage = _stage()
    rng = np.random.default_rng(4)
    alphabet = list(ALPHABET)
    motifs = ["".join(rng.choice(alphabet, size=5)) for _ in range(12)]

    def corpus(count, length):
        records = []
        for _ in range(count):
            parts: list[str] = []
            while sum(map(len, parts)) < length:
                if rng.random() < 0.5:
                    parts.append("".join(rng.choice(motifs)))
                else:
                    parts.append("".join(rng.choice(alphabet, size=int(rng.integers(3, 9)))))
            records.append("".join(parts))
        return records

    train, held = corpus(1200, 1000), corpus(80, 1000)
    counts = _toy_counts(train, (1, 2, 3, 4, 5, 6), chunk_bytes=1 << 20)
    curves = stage._held_out_curves(counts, {"a": held[:40], "b": held[40:]}, max_order=6)
    record = stage._supported_orders(curves, 6)
    assert record["highest_supported_order"] == 4
    assert record["turned_at"] == 5

    plug_in = D.InterpolatedFragmentModel(counts, 6, "kneser_ney")
    curve = [plug_in.cross_entropy(train[:60], order)["cross_entropy_nats"] for order in range(1, 7)]
    assert curve == sorted(curve, reverse=True)
