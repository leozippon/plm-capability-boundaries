"""The decoding sweep's declaration, its length controls, and what it refuses.

The tests that matter here are the ones that would let a false positive through:
a grid that silently lost a configuration, a best-of-k curve whose blocks cross
a class boundary, a fixed-compute table that compares two different budgets, a
length control that is satisfied by an unsupported extrapolation, and a
degeneracy check that passes a set which has collapsed onto one sequence.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.capability.decoding import decoding_sweep as ds


# --------------------------------------------------------------- the grid


def test_the_grid_is_a_factorial_with_both_historical_operating_points():
    keys = set(ds.CONFIG_KEYS)
    assert len(keys) == len(ds.GRID) == 20
    for temperature in ds.CORE_TEMPERATURES:
        for top_p, top_k in ds.CORE_TRUNCATIONS:
            assert ds._key(temperature, top_p, top_k, "eos400") in keys
    # Both points the programme's existing generation results were drawn at are
    # inside the sweep, so a configuration can be compared with them rather than
    # with a re-derivation of them.
    assert ds.REFERENCE_KEY in keys
    assert ds.HISTORICAL_UNCONDITIONED_KEY in keys
    assert ds.config(ds.REFERENCE_KEY).temperature == 1.0
    assert ds.config(ds.REFERENCE_KEY).top_p == 0.95
    assert ds.config(ds.REFERENCE_KEY).top_k == 0
    # The truncation corners are reachable: nucleus off, and k-truncation alone.
    assert any(row.top_p == 1.00 and row.top_k == 0 for row in ds.GRID)
    assert any(row.top_k == 50 for row in ds.GRID)
    # Termination moves in both directions and is suppressible.
    caps = {row.max_new_tokens for row in ds.GRID}
    assert {128, 400, 600} <= caps
    assert any(row.min_new_tokens > 0 for row in ds.GRID)
    assert any(not row.allow_terminator for row in ds.GRID)


def test_the_deep_budget_configurations_are_declared_and_balanced_across_shards():
    for key in ds.DEEP_BUDGET_KEYS:
        assert ds.config(key).draws_per_cluster == ds.DRAWS_PER_CLUSTER_DEEP
    left, right = ds.config_shard(0, 2), ds.config_shard(1, 2)
    assert set(row.key for row in left) | set(row.key for row in right) == set(ds.CONFIG_KEYS)
    assert not set(row.key for row in left) & set(row.key for row in right)
    # Round-robin over a cost-sorted grid: the expensive cells cannot pile onto
    # one card and leave the other idle.
    assert sum(row.draws for row in left) == sum(row.draws for row in right)


def test_an_unknown_configuration_or_shard_is_refused():
    with pytest.raises(KeyError):
        ds.config("t9.99_p0.95_k0_eos400")
    with pytest.raises(ValueError):
        ds.config_shard(2, 2)


def test_k_values_only_admit_whole_block_structures():
    assert ds.k_values(8) == (1, 2, 4, 8)
    assert ds.k_values(32) == (1, 2, 4, 8, 16, 32)
    with pytest.raises(ValueError):
        ds.k_values(0)


# ------------------------------------------------------- seeds and identity


def test_cell_seeds_are_derived_reproducible_and_distinct_per_cell():
    first = ds.cell_seed(arm_name="zymctrl", config_key=ds.REFERENCE_KEY, cluster="3.6.1.27")
    assert first == ds.cell_seed(
        arm_name="zymctrl", config_key=ds.REFERENCE_KEY, cluster="3.6.1.27"
    )
    # Two configurations sharing a prompt must not share a sample, or the two
    # become dependent in a way the cluster bootstrap does not model.
    assert first != ds.cell_seed(
        arm_name="zymctrl", config_key="t0.60_p0.95_k0_eos400", cluster="3.6.1.27"
    )
    assert first != ds.cell_seed(
        arm_name="prollama", config_key=ds.REFERENCE_KEY, cluster="3.6.1.27"
    )
    seeds = {
        ds.cell_seed(arm_name="protgpt2", config_key=row.key, cluster=cluster)
        for row in ds.GRID
        for cluster, _ in ds.clusters_for("protgpt2", None)
    }
    assert len(seeds) == len(ds.GRID) * ds.CLUSTERS_PER_CONFIG


def test_an_unconditioned_arm_has_the_same_number_of_units_as_a_conditioned_one():
    blocks = ds.clusters_for("protgpt2", None)
    assert len(blocks) == ds.CLUSTERS_PER_CONFIG
    assert all(label is None for _, label in blocks)
    with pytest.raises(ValueError):
        ds.clusters_for("zymctrl", None)


# -------------------------------------------------------------- the band


def test_the_evaluation_band_refuses_run_ons_and_fragments_with_a_reason():
    low, high = ds.EVALUATION_BAND
    assert ds.in_band("A" * low)
    assert ds.in_band("A" * high)
    assert not ds.in_band("A" * (low - 1))
    assert not ds.in_band("A" * (high + 1))
    assert ds.out_of_band_reason("") == "empty_product"
    assert ds.out_of_band_reason("A" * (low - 1)) == "below_band"
    assert ds.out_of_band_reason("A" * (high + 1)) == "above_band"
    assert ds.out_of_band_reason("AXA" + "A" * low) == "noncanonical_residues"
    assert ds.out_of_band_reason("A" * low) is None


# ------------------------------------------------------------- the census


def _row(**overrides):
    row = {
        "id": "dc_0",
        "cluster": "c0",
        "draw_index": 0,
        "sequence": "M" + "A" * 199,
        "length": 200,
        "n_new_tokens": 200,
        "terminated_natively": True,
        "model_logprob_per_token": -1.0,
        "model_logprob_total": -200.0,
    }
    row.update(overrides)
    row["length"] = len(row["sequence"])
    return row


def test_the_census_reports_termination_and_both_length_distributions():
    rows = [
        _row(id="a", sequence="M" + "A" * 199, n_new_tokens=200, terminated_natively=True),
        _row(id="b", sequence="M" + "A" * 419, n_new_tokens=420, terminated_natively=False),
        _row(id="c", sequence="", n_new_tokens=1, terminated_natively=True),
    ]
    census = ds.config_census(rows)
    assert census["n_draws"] == 3
    assert census["n_in_band"] == 1
    assert census["termination"]["n_terminated_natively"] == 2
    assert census["termination"]["n_budget_censored"] == 1
    assert census["generation_tokens"]["total"] == 621
    # The out-of-band products stay in the denominator and are named, so a
    # configuration that merely ran on cannot look like one that produced little.
    assert census["band_eligibility"] == {"above_band": 1, "empty_product": 1, "in_band": 1}
    assert census["length_distribution_all_products"]["max"] == 420
    assert census["length_distribution_in_band"]["n"] == 1

    with pytest.raises(ValueError):
        ds.config_census([])


# -------------------------------------------------------- degeneracy checks


def test_a_collapsed_set_is_flagged_and_its_gain_is_not_a_gain():
    rng = np.random.default_rng(7)
    diverse = [
        "".join(rng.choice(list("ARNDCQEGHILKMFPSTWYV"), size=120)) for _ in range(40)
    ]
    collapsed = ["M" + "AAAAAAAAAAGG" * 10] * 40
    reference = ds.size_matched_reference(diverse, size=20, seed=3)
    good = ds.degeneracy_verdict(ds.set_profile(diverse[:20]), reference, gain=1.0)
    bad = ds.degeneracy_verdict(ds.set_profile(collapsed[:20]), reference, gain=1.0)
    assert not good["degenerate"]
    assert bad["degenerate"]
    assert bad["gain_is_not_a_gain"]
    # Each failure mode is named separately: duplication, repertoire, complexity.
    assert {"duplicate_fraction", "mean_pairwise_kmer_distance"} <= set(bad["axes_flagged"])
    assert ds.set_profile(collapsed)["duplicate_fraction"] == pytest.approx(1.0 - 1 / 40)


def test_corpus_identity_enters_the_profile_and_flags_retrieval():
    rng = np.random.default_rng(11)
    sequences = ["".join(rng.choice(list("ARNDCQEGHILKMFPSTWYV"), size=120)) for _ in range(40)]
    novel = ds.set_profile(sequences[:20], corpus_identity=[5.0] * 20)
    retrieved = ds.set_profile(sequences[:20], corpus_identity=[99.0] * 20)
    assert novel["fraction_near_duplicate_of_corpus"] == 0.0
    assert retrieved["fraction_near_duplicate_of_corpus"] == 1.0
    reference = ds.size_matched_reference(sequences, size=20, seed=5, corpus_identity=[5.0] * 40)
    verdict = ds.degeneracy_verdict(retrieved, reference, gain=2.0)
    assert "fraction_near_duplicate_of_corpus" in verdict["axes_flagged"]
    assert verdict["gain_is_not_a_gain"]
    with pytest.raises(ValueError):
        ds.set_profile(sequences[:3], corpus_identity=[5.0, 5.0])


# ------------------------------------------------------------- selection


def _cell(draws_per_cluster, clusters=8, *, confidence=None, score=None, in_band_mask=None):
    rows = []
    for cluster in range(clusters):
        for index in range(draws_per_cluster):
            position = cluster * draws_per_cluster + index
            good = True if in_band_mask is None else in_band_mask(cluster, index)
            rows.append(
                {
                    "id": f"dc_{position:04d}",
                    "cluster": f"c{cluster}",
                    "draw_index": index,
                    "sequence": ("M" + "A" * 99) if good else "AA",
                    "length": 100 if good else 2,
                    "n_new_tokens": 100,
                    "model_logprob_per_token": (
                        -1.0 - index if score is None else score(cluster, index)
                    ),
                    "mean_ca_plddt": (
                        50.0 + index if confidence is None else confidence(cluster, index)
                    ),
                }
            )
    return rows


def test_best_of_k_blocks_stay_inside_a_cluster():
    # One cluster is uniformly excellent and the rest are poor. If blocks were
    # pooled across clusters, best-of-k would spend every block on that cluster
    # and the curve would read as a decoding gain; within-cluster blocks cannot.
    rows = _cell(8, clusters=8, confidence=lambda c, i: 95.0 if c == 0 else 40.0,
                 score=lambda c, i: 10.0 if c == 0 else -float(i))
    curve = ds.selection_curve(
        rows,
        selector="model_logprob_per_token",
        field="mean_ca_plddt",
        k_values=(1, 8),
        draws_per_cluster=8,
    )
    assert [point["n_blocks"] for point in curve] == [64, 8]
    # Eight blocks, one per cluster, so exactly one of them lands on the good
    # cluster and the mean is pulled only 1/8 of the way towards it.
    assert curve[1]["mean"] == pytest.approx((95.0 + 7 * 40.0) / 8)
    assert curve[0]["mean"] == pytest.approx((8 * 95.0 + 56 * 40.0) / 64)


def test_best_of_k_charges_each_selector_the_compute_it_spends():
    rows = _cell(8)
    cheap = ds.selection_curve(
        rows, selector="model_logprob_per_token", field="mean_ca_plddt",
        k_values=(1, 4), draws_per_cluster=8,
    )
    oracle = ds.selection_curve(
        rows, selector="esmfold2_confidence_oracle", field="mean_ca_plddt",
        k_values=(1, 4), draws_per_cluster=8,
    )
    # Generation tokens scale with k for both; folds scale with k only for the
    # oracle, which is what makes it an upper bound rather than a strategy.
    assert [point["generation_tokens_per_block"] for point in cheap] == [100.0, 400.0]
    assert [point["folds_per_block"] for point in cheap] == [1.0, 1.0]
    assert [point["folds_per_block"] for point in oracle] == [1.0, 4.0]
    # The oracle selects the best by construction; the realistic selector here
    # prefers the lowest-confidence draw, so it must do worse.
    assert oracle[1]["mean"] > cheap[1]["mean"]


def test_a_block_with_no_in_band_product_is_counted_not_dropped():
    rows = _cell(8, clusters=8, in_band_mask=lambda c, i: c != 0)
    curve = ds.selection_curve(
        rows, selector="model_logprob_per_token", field="mean_ca_plddt",
        k_values=(8,), draws_per_cluster=8,
    )
    point = curve[0]
    assert point["n_blocks"] == 8
    assert point["n_blocks_with_candidate"] == 7
    assert point["n_blocks_empty"] == 1
    assert point["block_yield"] == pytest.approx(7 / 8)
    # The empty block still spent its tokens.
    assert point["generation_tokens_per_block"] == pytest.approx(800.0)


def test_selection_refuses_a_block_structure_that_does_not_exist():
    rows = _cell(8)
    with pytest.raises(ValueError):
        ds.selection_curve(
            rows, selector="model_logprob_per_token", field="mean_ca_plddt",
            k_values=(3,), draws_per_cluster=8,
        )
    with pytest.raises(ValueError):
        ds.selection_curve(
            rows, selector="model_logprob_per_token", field="mean_ca_plddt",
            k_values=(1,), draws_per_cluster=4,
        )
    with pytest.raises(KeyError):
        ds.selection_curve(
            rows, selector="coin_flip", field="mean_ca_plddt",
            k_values=(1,), draws_per_cluster=8,
        )


def test_matched_compute_refuses_to_compare_two_different_budgets():
    curves = {
        "cheap": ds.selection_curve(
            _cell(8), selector="model_logprob_per_token", field="mean_ca_plddt",
            k_values=(1, 2, 4, 8), draws_per_cluster=8,
        ),
        "shallow": ds.selection_curve(
            _cell(2), selector="model_logprob_per_token", field="mean_ca_plddt",
            k_values=(1, 2), draws_per_cluster=2,
        ),
    }
    table = ds.matched_compute_table(curves, budgets=[100.0, 800.0])
    at_one = next(row for row in table if row["generation_tokens_per_selected_candidate"] == 100.0)
    at_eight = next(row for row in table if row["generation_tokens_per_selected_candidate"] == 800.0)
    assert {entry["configuration"] for entry in at_one["entries"]} == {"cheap", "shallow"}
    assert all(entry["k"] == 1 for entry in at_one["entries"])
    # The shallow configuration's grid cannot reach eight hundred tokens per kept
    # candidate, so it is listed as absent rather than compared at its own cost.
    assert at_eight["configurations_absent"] == ["shallow"]
    assert [entry["k"] for entry in at_eight["entries"]] == [8]


# ------------------------------------------------------- the natural band


def _folded(lengths, values, clusters=None, field="mean_ca_plddt"):
    rows = []
    for index, (length, value) in enumerate(zip(lengths, values)):
        rows.append(
            {
                "id": f"x_{index:04d}",
                "cluster": f"c{index % 8}" if clusters is None else clusters[index],
                "length": int(length),
                field: float(value),
                "sequence": "M" + "A" * (int(length) - 1),
            }
        )
    return rows


def test_the_standardised_band_removes_a_pure_length_advantage():
    # Natural confidence rises with length. One configuration is uniformly long
    # and one uniformly short, and neither differs from natural at its own
    # length: the raw means differ, the standardised gaps do not.
    rng = np.random.default_rng(3)
    natural_lengths = list(range(60, 400, 2))
    natural = _folded(
        natural_lengths,
        [0.2 * length + rng.normal(0, 1.0) for length in natural_lengths],
    )
    long_lengths = [300 + 2 * (index % 20) for index in range(80)]
    short_lengths = [100 + 2 * (index % 20) for index in range(80)]
    long_config = _folded(long_lengths, [0.2 * length for length in long_lengths])
    short_config = _folded(short_lengths, [0.2 * length for length in short_lengths])

    assert np.mean([row["mean_ca_plddt"] for row in long_config]) > np.mean(
        [row["mean_ca_plddt"] for row in short_config]
    )
    left = ds.natural_standardised_band(
        long_config, natural, field="mean_ca_plddt", seed=1, n_bootstrap=200
    )
    right = ds.natural_standardised_band(
        short_config, natural, field="mean_ca_plddt", seed=1, n_bootstrap=200
    )
    assert left["resolved"] and right["resolved"]
    assert left["gap"] == pytest.approx(0.0, abs=1.5)
    assert right["gap"] == pytest.approx(0.0, abs=1.5)
    assert left["gap_interval"][0] < 0.0 < left["gap_interval"][1]


def test_a_configuration_without_natural_support_is_named_not_forced():
    natural = _folded(list(range(60, 120, 2)), [50.0] * 30)
    # Every product sits in a stratum the natural pool does not populate.
    stranded = _folded([380] * 40, [90.0] * 40)
    band = ds.natural_standardised_band(
        stranded, natural, field="mean_ca_plddt", seed=1, n_bootstrap=100
    )
    assert band["resolved"] is False
    assert band["gap"] is None
    assert band["unsupported_length_mass"] == pytest.approx(1.0)
    assert "not identifiable" in band["unresolved_reason"]


def test_the_paired_comparator_declares_itself_unresolved_when_it_cannot_match():
    natural = _folded(list(range(60, 400, 2)), [50.0] * 170)
    # Sixty-four candidates all at one length: the folded natural pool holds only
    # a handful of records inside the tolerance, so a without-replacement pairing
    # cannot be formed and the contrast must say so.
    concentrated = _folded([201] * 128, [90.0] * 128)
    record = ds.natural_band_contrast(
        concentrated, natural, field="mean_ca_plddt", seed=1, n_bootstrap=200
    )
    assert record["resolved"] is False
    assert record["contrast"] is None
    assert record["match_rate"] < ds.MIN_PAIRED_MATCH_RATE

    spread = _folded(list(range(80, 336, 4)), [90.0] * 64)
    good = ds.natural_band_contrast(
        spread, natural, field="mean_ca_plddt", seed=1, n_bootstrap=200
    )
    assert good["resolved"]
    assert good["contrast"]["difference"] == pytest.approx(40.0)


def test_capping_keeps_the_cluster_structure():
    rows = _folded(list(range(100, 164)), [50.0] * 64)
    sample = ds.capped_by_cluster(rows, cap=16, seed=2)
    assert len(sample) == 16
    assert len({row["cluster"] for row in sample}) == 8
    with pytest.raises(ValueError):
        ds.capped_by_cluster(rows, cap=0, seed=2)


# --------------------------------------------------- simultaneous inference


def test_the_simultaneous_bound_is_about_the_best_configuration_not_each_one():
    # Twenty configurations, all genuinely below the band by five units with
    # noise. Marginal intervals will individually exclude zero; the simultaneous
    # bound on the maximum is the statement a panel-wide claim needs, and it is
    # looser than any single marginal upper end.
    rng = np.random.default_rng(13)
    per_config = {
        f"cfg{index:02d}": {
            f"c{cluster}": -5.0 + rng.normal(0, 1.5) for cluster in range(16)
        }
        for index in range(20)
    }
    joint = ds.joint_cluster_bootstrap(per_config, seed=4, n_bootstrap=1000)
    assert joint["n_clusters"] == 16
    bound = joint["simultaneous"]["max_over_configurations"]
    assert bound["upper_bound"] < 0.0
    marginals = [block["interval"][1] for block in joint["marginal"].values()]
    assert bound["upper_bound"] >= max(marginals) - 1e-9
    assert all(block["resolved"] for block in joint["marginal"].values())


def test_a_configuration_below_the_unit_floor_is_excluded_and_recorded():
    per_config = {
        "rich": {f"c{index}": -1.0 for index in range(16)},
        "thin": {"c0": 9.0, "c1": 9.0},
    }
    joint = ds.joint_cluster_bootstrap(per_config, seed=4, n_bootstrap=200)
    assert joint["marginal"]["thin"]["resolved"] is False
    assert joint["marginal"]["thin"]["interval"] is None
    assert joint["simultaneous"]["excluded_configurations"] == ["thin"]
    # The excluded configuration's inflated values must not reach the bound.
    assert joint["simultaneous"]["max_over_configurations"]["upper_bound"] < 0.0


# ------------------------------------------------- length standardisation


def test_length_standardisation_equalises_a_length_only_advantage():
    rng = np.random.default_rng(17)
    long_lengths = [200 + 4 * (index % 40) for index in range(160)]
    short_lengths = [60 + 4 * (index % 40) for index in range(160)]

    def value(length):
        return 0.25 * length + rng.normal(0, 0.5)

    rows = {
        "long": _folded(long_lengths, [value(length) for length in long_lengths]),
        "short": _folded(short_lengths, [value(length) for length in short_lengths]),
    }
    report = ds.length_standardised(rows, field="mean_ca_plddt", seed=9, n_bootstrap=200)
    long_block = report["configurations"]["long"]
    short_block = report["configurations"]["short"]
    assert long_block["raw_mean"] > short_block["raw_mean"] + 20.0
    # Neither configuration covers the pooled length range, so neither has a
    # standardised mean: an unsupported extrapolation is refused, not filled.
    assert long_block["standardised_mean"] is None
    assert short_block["standardised_mean"] is None
    assert long_block["empty_bins"] and short_block["empty_bins"]


def test_length_standardisation_resolves_when_both_cover_the_range():
    rng = np.random.default_rng(19)
    lengths = [60 + 4 * (index % 80) for index in range(320)]
    rows = {
        "flat": _folded(lengths, [0.25 * length + rng.normal(0, 0.5) for length in lengths]),
        "better": _folded(lengths, [0.25 * length + 10.0 + rng.normal(0, 0.5) for length in lengths]),
    }
    report = ds.length_standardised(rows, field="mean_ca_plddt", seed=9, n_bootstrap=300)
    flat = report["configurations"]["flat"]
    better = report["configurations"]["better"]
    assert flat["resolved"] and better["resolved"]
    assert better["standardised_mean"] - flat["standardised_mean"] == pytest.approx(10.0, abs=1.0)
    assert flat["empty_bins"] == [] and better["empty_bins"] == []


# ------------------------------------------------- configuration-level means


def test_a_configuration_mean_declares_its_unit_and_both_readings():
    # Real between-cluster variation: each cluster sits at its own level and
    # varies little inside itself, which is what a frozen class does.
    rng = np.random.default_rng(23)
    lengths, values, clusters = [], [], []
    for cluster in range(8):
        level = 40.0 + 6.0 * cluster
        for index in range(8):
            lengths.append(100 + cluster * 8 + index)
            values.append(level + rng.normal(0, 0.3))
            clusters.append(f"c{cluster}")
    rows = _folded(lengths, values, clusters=clusters)
    block = ds.cluster_level_mean(rows, field="mean_ca_plddt")
    assert block["unit"] == "cluster"
    assert block["n_clusters"] == 8
    assert block["n_draws"] == 64
    assert block["resolved"]
    assert block["cluster_level"]["n"] == 8
    assert block["candidate_level"]["n"] == 64
    # With genuine between-cluster spread the cluster-level interval is the wider,
    # conservative one. Both are published, each with its own n, so a reader
    # cannot mistake the narrow candidate-level reading for a statement about the
    # arm. Which is wider depends on where the variance sits, and that is exactly
    # why the unit is declared rather than left implicit.
    span = lambda block: block["interval"][1] - block["interval"][0]  # noqa: E731
    assert span(block["cluster_level"]) > span(block["candidate_level"])

    thin = ds.cluster_level_mean(
        _folded([120, 130, 140], [50.0, 55.0, 60.0], clusters=["c0", "c1", "c2"]),
        field="mean_ca_plddt",
    )
    assert thin["resolved"] is False
    assert thin["unit_floor"]["degenerate"]


def test_lifting_a_structure_row_keeps_the_pairwise_summary_and_refuses_a_failure():
    ok = {
        "structure": {
            "status": "ok",
            "mean_ca_plddt": 61.5,
            "fraction_ca_plddt_ge70": 0.3,
            "ptm": 0.4,
            "mean_pae_angstrom": 18.0,
            "predicted_confidence_event": False,
            "object_directory": "objects/abc",
            "diffusion_sample_index": 7,
            "evaluation_signature": "sig",
        }
    }
    lifted = ds.lift_structure(ok)
    assert lifted["neg_mean_pae_angstrom"] == -18.0
    assert lifted["predicted_confidence_event"] == 0.0
    # The object directory travels with the row, so the retained PAE matrix and
    # per-residue pLDDT stay reachable rather than being reduced away.
    assert lifted["structure_object_directory"] == "objects/abc"
    assert ds.lift_structure({"structure": {"status": "error"}}) is None
    assert ds.lift_structure({}) is None
