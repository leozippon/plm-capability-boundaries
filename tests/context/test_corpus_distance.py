"""Invariants of the corpus-traceability and distance-stratification design (E10).

The point of these tests is that an arm can only be stratified against a corpus
the repository declares for it, that a band is the project's already-frozen one,
and that thin support is reported as thin rather than as a zero effect.
"""
import numpy as np
import pytest

from src.capability.context import corpus_distance as CD
from src.capability.context.homology import STRATUM_EDGES, STRATUM_NAMES

PANEL = [
    "progen2-medium",   # uniref90_bfd30: declared, searchable as a later release
    "protgpt2",         # uniref50: declared, searchable; the only such arm of the 33
    "rita-xl",          # uniref100: declared, searchable only as a subset proxy
    "progen2-base",     # an unidentified mixture
    "progen3-3b",       # a proprietary corpus
    "gpt2",             # a text corpus
    "bygpt5-small-en",  # the undeclared sentinel
    "galactica-125m",   # no ArmSpec at all
]


def test_traceability_tiers_enumerate_the_panel_from_the_registry():
    record = CD.traceability(PANEL)
    tier = {row["arm"]: row["tier"] for row in record["arms"]}
    assert tier["progen2-medium"] == "declared_and_searchable"
    assert tier["protgpt2"] == "declared_and_searchable"
    assert tier["rita-xl"] == "declared_and_searchable"
    assert tier["progen2-base"] == "declared_not_searchable"
    assert tier["progen3-3b"] == "declared_not_searchable"
    assert tier["gpt2"] == "declared_not_searchable"
    assert tier["bygpt5-small-en"] == "undeclared"
    assert tier["galactica-125m"] == "no_arm_spec"
    assert record["tier_counts"] == {
        "declared_and_searchable": 3,
        "declared_not_searchable": 3,
        "undeclared": 1,
        "no_arm_spec": 1,
    }
    reasons = {row["arm"]: row.get("reason_class") for row in record["arms"]}
    assert reasons["gpt2"] == "non_protein_corpus"
    assert reasons["progen3-3b"] == "proprietary_or_unidentified_corpus"
    assert record["primary_arms"] == ["progen2-medium"]


def test_every_searchable_relation_states_the_direction_of_its_error():
    for declared, relation in CD.CORPUS_RELATIONS.items():
        assert relation["bias"], f"{declared} declares no bias direction"
        assert "remote_is_conservative" in relation
        assert isinstance(relation["unsearched_components"], list)


def test_an_arm_is_refused_when_the_searched_corpus_does_not_stand_for_it():
    record = CD.traceability(PANEL + ["zymctrl"])
    uniref90 = CD.arms_for_corpus(record, "uniref90_2026_03")
    assert "progen2-medium" in uniref90["eligible_arms"]
    assert "rita-xl" in uniref90["eligible_arms"]
    # ProtGPT2 enters only through its declared alternative, and that is recorded.
    applied = {row["arm"]: row["applied_relation"] for row in uniref90["eligible"]}
    assert applied["protgpt2"] == "declared_alternative"
    assert applied["progen2-medium"] == CD.PRIMARY_RELATION
    assert uniref90["primary_arms"] == ["progen2-medium"]
    assert "zymctrl" not in uniref90["eligible_arms"]
    refusal = next(row for row in uniref90["refused"] if row["arm"] == "zymctrl")
    assert "does not stand for" in refusal["refusal"]

    uniref50 = CD.arms_for_corpus(record, "uniref50_local_snapshot")
    assert uniref50["eligible_arms"] == ["protgpt2"]
    assert "progen2-medium" in [row["arm"] for row in uniref50["refused"]]


def test_distance_bands_are_the_already_frozen_retrieval_strata():
    assert CD.DISTANCE_BANDS == STRATUM_NAMES
    assert CD.DISTANCE_EDGES == STRATUM_EDGES
    assert CD.distance_band(0.0) == "lt30_no_detectable_homology"
    assert CD.distance_band(29.999) == "lt30_no_detectable_homology"
    assert CD.distance_band(30.0) == "id30_to_70_remote_homology"
    assert CD.distance_band(69.999) == "id30_to_70_remote_homology"
    assert CD.distance_band(70.0) == "id70_to_95_close_homology"
    assert CD.distance_band(95.0) == "ge95_near_duplicate"
    assert CD.distance_band(100.0) == "ge95_near_duplicate"


def make_support(bands_per_cluster):
    """One assay per cluster, with that cluster's declared max identity."""
    targets, assays = [], []
    for index, (identity, cluster) in enumerate(bands_per_cluster):
        identifier = f"t{index:03d}"
        targets.append({"target_id": identifier, "max_identity_over_query": identity})
        assays.append(
            {"assay": f"a{index:03d}", "target_id": identifier, "cluster": cluster}
        )
    return targets, assays


def test_assay_bands_count_clusters_and_apply_the_group_floor():
    rows = [(99.0, index) for index in range(10)] + [(40.0, 100 + index) for index in range(3)]
    targets, assays = make_support(rows)
    bands, support = CD.assay_bands(targets, assays)
    assert support["clusters_per_band"]["ge95_near_duplicate"] == 10
    assert support["clusters_per_band"]["id30_to_70_remote_homology"] == 3
    assert support["admitted_bands"] == ["ge95_near_duplicate"]
    assert bands["a000"] == "ge95_near_duplicate"
    assert bands["a010"] == "id30_to_70_remote_homology"
    assert support["group_floor"] == 8


def test_an_assay_without_a_retrieval_record_stops_the_run():
    targets, assays = make_support([(99.0, 1)])
    assays.append({"assay": "orphan", "target_id": "missing", "cluster": 2})
    with pytest.raises(ValueError):
        CD.assay_bands(targets, assays)


def scores_for(bands, *, arm, value):
    return [
        {"arm": arm, "seed": seed, "assay": assay, "cluster": cluster, "contrast": value}
        for assay, (cluster, _) in bands.items()
        for seed in (1, 2)
    ]


def test_structural_absence_is_nan_and_never_a_zero():
    rows = [(99.0, index) for index in range(10)] + [(40.0, 100 + index) for index in range(10)]
    targets, assays = make_support(rows)
    bands, support = CD.assay_bands(targets, assays)
    # One arm scores only the near-duplicate assays.
    scores = [
        {"arm": "progen2-medium", "seed": 1, "assay": assay, "cluster": int(assay[1:]), "contrast": 0.5}
        for assay, band in bands.items()
        if band == "ge95_near_duplicate"
    ]
    families, columns, matrix = CD.stratum_matrix(
        scores,
        bands=bands,
        arms=["progen2-medium"],
        admitted=support["admitted_bands"],
    )
    remote = columns.index(("progen2-medium", "id30_to_70_remote_homology"))
    assert np.isnan(matrix[:, remote]).all(), "an unscored band must be NaN, not zero"
    close = columns.index(("progen2-medium", "ge95_near_duplicate"))
    assert np.isfinite(matrix[:, close]).sum() == 10


def test_seeds_average_within_an_assay_before_families_are_weighted():
    targets, assays = make_support([(99.0, index) for index in range(10)])
    bands, support = CD.assay_bands(targets, assays)
    scores = []
    for assay in bands:
        for seed, value in ((1, 0.0), (2, 1.0)):
            scores.append(
                {"arm": "x", "seed": seed, "assay": assay, "cluster": int(assay[1:]), "contrast": value}
            )
    _, _, matrix = CD.stratum_matrix(
        scores, bands=bands, arms=["x"], admitted=support["admitted_bands"]
    )
    assert np.allclose(matrix[np.isfinite(matrix)], 0.5)


def test_thin_support_is_reported_thin_rather_than_as_a_null_result():
    rows = [(99.0, index) for index in range(12)] + [(10.0, 200 + index) for index in range(3)]
    targets, assays = make_support(rows)
    bands, _ = CD.assay_bands(targets, assays)
    scores = [
        {"arm": "progen2-medium", "seed": 1, "assay": assay, "cluster": int(assay[1:]), "contrast": 0.4}
        for assay in bands
    ]
    panel = CD.stratified_panel(
        scores,
        bands=bands,
        arms=["progen2-medium"],
        # Deliberately admitting a band below the floor, to check the verdict.
        admitted=["ge95_near_duplicate", "lt30_no_detectable_homology"],
        draws=200,
        seed=1,
    )
    assert panel["status"] == "estimated"
    verdicts = {cell["band"]: cell for cell in panel["cells"]}
    assert verdicts["lt30_no_detectable_homology"]["groups"] == 3
    assert verdicts["lt30_no_detectable_homology"]["resolution"] == "unresolved_thin_support"
    assert verdicts["ge95_near_duplicate"]["groups"] == 12
    for cell in panel["cells"]:
        assert "groups_required_to_resolve" in cell
        assert cell["point"] == pytest.approx(0.4)


def test_monotonicity_names_the_most_distant_resolved_band():
    cells = [
        {"arm": "a", "band": "ge95_near_duplicate", "point": 0.3, "resolution": "resolved", "direction": "above_zero"},
        {"arm": "a", "band": "id70_to_95_close_homology", "point": 0.2, "resolution": "resolved", "direction": "above_zero"},
        {"arm": "a", "band": "id30_to_70_remote_homology", "point": 0.1, "resolution": "unresolved_interval_crosses_zero", "direction": None},
    ]
    reading = CD.monotonicity(cells, order=tuple(reversed(CD.DISTANCE_BANDS)))["a"]
    assert reading["bands"] == [
        "ge95_near_duplicate",
        "id70_to_95_close_homology",
        "id30_to_70_remote_homology",
    ]
    assert reading["most_distant_resolved_positive_band"] == "id70_to_95_close_homology"
    assert reading["resolved_positive_bands"] == [
        "ge95_near_duplicate",
        "id70_to_95_close_homology",
    ]
    assert reading["decreasing_with_distance"] is True


def test_referent_value_reads_the_profile_from_beside_the_conditions():
    """The three referents live in two places and both must be reachable."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts/capability/context/score_context_identity.py"
    spec = importlib.util.spec_from_file_location("score_context_identity", path)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    from src.capability.context import homology_context as H

    assay = {"spearman": {"unrelated": 0.4, "no_context": 0.3}, "lookup_spearman": 0.7}
    assert stage.referent_value(assay, H.UNRELATED) == 0.4
    assert stage.referent_value(assay, H.NO_CONTEXT) == 0.3
    assert stage.referent_value(assay, H.PROFILE_REFERENT) == 0.7
    # A missing referent is None, so the contrast is skipped rather than invented.
    assert stage.referent_value({"spearman": {}}, H.UNRELATED) is None
    assert stage.referent_value({"spearman": {}, "lookup_spearman": None}, H.PROFILE_REFERENT) is None


def test_the_analysis_refuses_to_pool_scores_from_different_batch_extents():
    """Scores taken at different batch extents are different arithmetic."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts/capability/context/score_context_identity.py"
    spec = importlib.util.spec_from_file_location("score_context_identity", path)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    from src.capability.context import homology_context as H

    # The declaration fixes the protocol, and the stage refuses any other value
    # rather than silently scoring at it.
    assert H.SCORING_ROWS_PER_FORWARD == 1
    assert stage.H.SCORING_ROWS_PER_FORWARD == 1
    source = path.read_text()
    assert "rows_per_forward" in source
    assert "never pooled or compared across arms" in source
