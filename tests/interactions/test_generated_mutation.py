"""Conditions that must hold for every generated-protein mutation landscape.

The properties tested here are the ones a wrong answer would hide: that a cohort
is reproducible from its inputs alone, that a label written by the builder parses
back under the extractor's own rule, that a matched design is actually matched,
that a missing input is a refusal rather than a weaker number, and that the
four-state interaction is the definition and not an approximation of it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.capability.interactions import generated_mutation as gm

ROOT = Path(__file__).resolve().parents[2]


def _extractor():
    path = ROOT / "scripts/capability/position/extract_position_likelihood.py"
    spec = importlib.util.spec_from_file_location("_extract_position_likelihood", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(identity: str, sequence: str, *, stage: str = "stage_1",
            stream: str = "replicate_1", entropy: float = 2.5, run: int = 3) -> dict:
    return {
        "arm": "prollama-stage-1",
        "id": identity,
        "length": len(sequence),
        "sequence": sequence,
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "stage": stage,
        "stream": stream,
        "properties": {
            "composition_entropy_nats": entropy,
            "longest_single_residue_run": run,
            "length": len(sequence),
        },
    }


def _write(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "generated.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    return path


SEQ_A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHS"
SEQ_B = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFELRRQACGRTRTLLDAEAVFWQPVETGSGHWVKLE"


# ------------------------------------------------------------ stage plumbing


def test_output_directory_accepts_an_empty_existing_directory(tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    assert gm.prepare_output_directory(target, "done.json") == target


def test_output_directory_refuses_a_completed_cell(tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    (target / "done.json").write_text("{}")
    with pytest.raises(SystemExit, match="already present"):
        gm.prepare_output_directory(target, "done.json")


def test_output_directory_refuses_prior_content(tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    (target / "archives").mkdir()
    with pytest.raises(SystemExit, match="already holds"):
        gm.prepare_output_directory(target, "done.json")


# ---------------------------------------------------------- the generated set


def test_generated_set_is_reverified_against_its_own_digest(tmp_path):
    row = _record("ge_a", SEQ_A)
    row["sequence_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="own sha256"):
        gm.read_generated(_write(tmp_path, [row]))


def test_generated_set_refuses_a_length_disagreement(tmp_path):
    row = _record("ge_a", SEQ_A)
    row["length"] = len(SEQ_A) + 1
    with pytest.raises(ValueError, match="length disagrees with its own field"):
        gm.read_generated(_write(tmp_path, [row]))


def test_generated_set_refuses_a_non_canonical_symbol(tmp_path):
    row = _record("ge_a", SEQ_A[:-1] + "X")
    with pytest.raises(ValueError, match="non-AA20"):
        gm.read_generated(_write(tmp_path, [row]))


def test_generated_set_refuses_a_duplicate_identity(tmp_path):
    rows = [_record("ge_a", SEQ_A), _record("ge_a", SEQ_B)]
    with pytest.raises(ValueError, match="duplicate identity"):
        gm.read_generated(_write(tmp_path, rows))


def test_subsample_is_reproducible_and_balanced_over_streams():
    records = []
    for stage_index, stage in enumerate(("stage_1", "stage_2")):
        for stream_index, stream in enumerate(("a", "b", "c")):
            for offset in range(10):
                number = 100 * stage_index + 10 * stream_index + offset
                records.append(
                    _record(
                        f"ge_{number:03d}",
                        SEQ_A[: 40 + offset],
                        stage=stage,
                        stream=stream,
                    )
                )
    first = gm.stratified_subsample(records, per_stage=6)
    assert [row["id"] for row in first] == [
        row["id"] for row in gm.stratified_subsample(records, per_stage=6)
    ]
    assert len(first) == 12
    for stage in ("stage_1", "stage_2"):
        streams = [row["stream"] for row in first if row["stage"] == stage]
        assert sorted(streams) == sorted(["a", "a", "b", "b", "c", "c"])


def test_subsample_refuses_more_than_a_stream_holds():
    records = [_record("ge_1", SEQ_A), _record("ge_2", SEQ_B)]
    with pytest.raises(ValueError, match="were requested"):
        gm.stratified_subsample(records, per_stage=9)


def test_degeneracy_stratum_follows_the_declared_rule():
    assert gm.degenerate(_record("ge_a", SEQ_A, entropy=0.5, run=2))
    assert gm.degenerate(_record("ge_a", SEQ_A, entropy=2.5, run=40))
    assert not gm.degenerate(_record("ge_a", SEQ_A, entropy=2.5, run=3))


# ------------------------------------------------------------- the matching


def test_length_matching_is_one_to_one_inside_the_caliper():
    generated = [_record(f"ge_{index}", SEQ_A[: 60 + index]) for index in range(4)]
    pool = [SEQ_B[: 60 + index] for index in range(4)] + [SEQ_B[:61], SEQ_B[:62]]
    matched = gm.match_natural(generated, pool)
    assert matched["balance"]["pairs"] == 4
    assert matched["balance"]["unmatched_generated"] == 0
    assert len({row["pool_position"] for row in matched["matched"]}) == 4
    for row in matched["matched"]:
        caliper = max(gm.LENGTH_CALIPER_RESIDUES,
                      int(round(gm.LENGTH_CALIPER_FRACTION * row["length"])))
        assert abs(row["length_delta"]) <= caliper


def test_a_byte_identical_pool_entry_cannot_become_two_comparators():
    # Swiss-Prot is non-redundant per entry and not per sequence: one protein can
    # appear many times. Drawing records rather than distinct sequences gave two
    # generated products the same comparator under one content identity, which is
    # the cohort defect this test exists to make impossible.
    twin = SEQ_B[:70]
    generated = [_record("ge_1", SEQ_A[:70]), _record("ge_2", SEQ_A[:70])]
    pool = [twin, twin, twin, SEQ_B[:69] + "W"]
    matched = gm.match_natural(generated, pool)
    sequences = [row["sequence"] for row in matched["matched"]]
    identities = [row["id"] for row in matched["matched"]]
    assert len(set(sequences)) == len(sequences)
    assert len(set(identities)) == len(identities)
    assert matched["balance"]["pool_records"] == 4
    assert matched["balance"]["pool_distinct_sequences"] == 2
    assert matched["balance"]["pool_duplicate_records_removed"] == 2


def test_duplicate_assay_identities_are_refused_at_the_producer():
    rows = [
        gm.cohort_assay(assay="x", wildtype=SEQ_A, mutants=["A1G"],
                        sequences=[gm.apply_substitutions(SEQ_A, {0: "G"})], cluster="g"),
        gm.cohort_assay(assay="x", wildtype=SEQ_A, mutants=["A1W"],
                        sequences=[gm.apply_substitutions(SEQ_A, {0: "W"})], cluster="g"),
    ]
    gm.require_unique_assays(rows[:1])
    with pytest.raises(ValueError, match="not unique"):
        gm.require_unique_assays(rows)


def test_a_built_singles_cohort_has_unique_identities_against_a_redundant_pool(tmp_path):
    # End to end through the builder's own singles path, with a pool deliberately
    # made as redundant as the real corpus is.
    records = [
        _record(f"ge_{index:02d}", SEQ_A[: 60 + index], stage=stage, stream=stream)
        for stage in ("stage_1", "stage_2")
        for stream in ("a", "b")
        for index in range(3)
    ]
    pool = []
    for index in range(3):
        pool.extend([SEQ_B[: 60 + index]] * 5)
    matched = gm.match_natural(records[:3], pool)
    assays = []
    for row in records[:3]:
        partner = next(p for p in matched["matched"] if p["matched_to"] == row["id"])
        for identity, sequence in ((row["id"], row["sequence"]),
                                   (partner["id"], partner["sequence"])):
            scan = gm.scan_mutations(sequence, sites=2, subs=1)
            assays.append(gm.cohort_assay(
                assay=identity, wildtype=sequence,
                mutants=[item["label"] for item in scan],
                sequences=[item["sequence"] for item in scan], cluster="g"))
    gm.require_unique_assays(assays)
    assert len({row["assay"] for row in assays}) == len(assays)


def test_length_matching_reports_a_generated_product_it_cannot_match():
    generated = [_record("ge_long", SEQ_A), _record("ge_short", SEQ_A[:45])]
    matched = gm.match_natural(generated, [SEQ_A[:45]])
    assert [row["id"] for row in matched["unmatched"]] == ["ge_long"]
    assert matched["balance"]["pairs"] == 1


def test_independence_groups_join_near_duplicates():
    names, record = gm.independence_groups([SEQ_A, SEQ_A, SEQ_B])
    assert names[0] == names[1]
    assert names[2] != names[0]
    assert record["shingle"] == 5


# ------------------------------------------------------------- the mutations


def test_scan_labels_parse_back_under_the_extractor_rule():
    extractor = _extractor()
    scan = gm.scan_mutations(SEQ_A, sites=6, subs=2)
    assert len(scan) == 12
    assert len({item["label"] for item in scan}) == 12
    for item in scan:
        assert extractor.substitution_sites(item["label"], SEQ_A) == [item["site"]]
        assert item["sequence"] != SEQ_A
        assert sum(1 for a, b in zip(SEQ_A, item["sequence"]) if a != b) == 1


def test_scan_is_a_function_of_the_sequence_and_the_declared_seed():
    assert gm.scan_mutations(SEQ_A, sites=4, subs=1) == gm.scan_mutations(SEQ_A, sites=4, subs=1)
    assert gm.scan_mutations(SEQ_A, sites=4, subs=1) != gm.scan_mutations(SEQ_B, sites=4, subs=1)


def test_pair_states_share_one_substitution_per_position():
    pairs = [
        {"i": 3, "j": 20, "contact": True, "distance": 6.0},
        {"i": 3, "j": 40, "contact": False, "distance": 11.0},
    ]
    built = gm.pair_mutations(SEQ_A, pairs)
    assert len(built["mutants"]) == len(set(built["mutants"]))
    first, second = built["cycles"]
    assert first["single_low"] == second["single_low"]
    states = dict(zip(built["mutants"], built["sequences"]))
    for cycle in built["cycles"]:
        low, high = cycle["i"], cycle["j"]
        double = states[cycle["double"]]
        assert double[low] == states[cycle["single_low"]][low]
        assert double[high] == states[cycle["single_high"]][high]
        assert sum(1 for a, b in zip(SEQ_A, double) if a != b) == 2


def test_state_label_round_trips_a_double():
    extractor = _extractor()
    double = gm.apply_substitutions(SEQ_A, {4: "W", 30: "W"})
    label = gm.state_label(SEQ_A, double)
    assert label.count(":") == 1
    assert extractor.substitution_sites(label, SEQ_A) == [4, 30]


# --------------------------------------------------------- predicted structure


def _structure(tmp_path: Path, length: int, *, distance_key="cb_distance_angstrom",
               confidence=True) -> Path:
    rng = np.random.default_rng(7)
    xyz = np.cumsum(rng.normal(size=(length, 3)), axis=0) * 2.0
    distance = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    payload = {distance_key: distance.astype(np.float32)}
    if confidence:
        payload["plddt"] = np.linspace(40.0, 95.0, length).astype(np.float32)
    path = tmp_path / "structure.npz"
    np.savez_compressed(path, **payload)
    return path


def test_structure_refuses_an_archive_without_a_distance_matrix(tmp_path):
    path = _structure(tmp_path, len(SEQ_A), distance_key="something_else")
    with pytest.raises(ValueError, match="no CB-CB distance matrix"):
        gm.read_predicted_structure(path, sequence=SEQ_A)


def test_structure_refuses_an_unknown_confidence_rather_than_assuming_one(tmp_path):
    path = _structure(tmp_path, len(SEQ_A), confidence=False)
    with pytest.raises(ValueError, match="no confidence array"):
        gm.read_predicted_structure(path, sequence=SEQ_A)
    read = gm.read_predicted_structure(path, sequence=SEQ_A, confidence_key="none")
    assert read["confidence_key"] is None
    assert read["admitted_positions"] == len(SEQ_A)


def test_structure_refuses_a_matrix_of_the_wrong_shape(tmp_path):
    path = _structure(tmp_path, len(SEQ_A) - 1)
    with pytest.raises(ValueError, match="is not"):
        gm.read_predicted_structure(path, sequence=SEQ_A)


def test_confidence_floor_removes_positions_from_the_pair_population(tmp_path):
    path = _structure(tmp_path, len(SEQ_A))
    low = gm.read_predicted_structure(path, sequence=SEQ_A, confidence_floor=0.0)
    high = gm.read_predicted_structure(path, sequence=SEQ_A, confidence_floor=90.0)
    assert high["admitted_positions"] < low["admitted_positions"] == len(SEQ_A)


def test_eligible_pairs_honour_the_floor_and_the_strict_cutoff():
    distance = np.full((20, 20), 20.0)
    np.fill_diagonal(distance, 0.0)
    distance[0, 10] = distance[10, 0] = 8.0
    distance[1, 11] = distance[11, 1] = 7.999
    structure = {"distance": distance, "admitted": np.ones(20, dtype=bool)}
    pairs = gm.eligible_pairs(structure)
    assert all(row["separation"] >= gm.MIN_SEQUENCE_SEPARATION for row in pairs)
    found = {(row["i"], row["j"]): row["contact"] for row in pairs}
    assert found[(0, 10)] is False
    assert found[(1, 11)] is True


def test_eligible_pairs_refuse_a_non_finite_distance():
    distance = np.full((20, 20), 20.0)
    distance[0, 10] = distance[10, 0] = np.nan
    structure = {"distance": distance, "admitted": np.ones(20, dtype=bool)}
    with pytest.raises(ValueError, match="not a non-contact"):
        gm.eligible_pairs(structure)


def test_pair_design_matches_separation_and_drops_what_it_cannot_match():
    pairs = [
        {"i": 0, "j": 10, "separation": 10, "stratum": "9-16", "distance": 5.0, "contact": True},
        {"i": 2, "j": 13, "separation": 11, "stratum": "9-16", "distance": 20.0, "contact": False},
        {"i": 4, "j": 54, "separation": 50, "stratum": "33-64", "distance": 5.0, "contact": True},
    ]
    design = gm.matched_pair_design(pairs, SEQ_A, per_stratum=2)
    assert design["balance"]["contacts"] == design["balance"]["controls"] == 1
    assert abs(design["balance"]["separation_imbalance_residues"]) <= gm.SEPARATION_CALIPER
    assert [row["j"] for row in design["dropped_contacts"]] == [54]


def test_pair_design_never_matches_across_the_caliper():
    pairs = [
        {"i": 0, "j": 10, "separation": 10, "stratum": "9-16", "distance": 5.0, "contact": True},
        {"i": 2, "j": 18, "separation": 16, "stratum": "9-16", "distance": 20.0, "contact": False},
    ]
    design = gm.matched_pair_design(pairs, SEQ_A, per_stratum=2)
    assert design["pairs"] == []
    assert len(design["dropped_contacts"]) == 1


# ------------------------------------------------------------- the estimators


def test_four_state_interaction_is_the_definition():
    cycle = {"single_low": "A1G", "single_high": "L9V", "double": "A1G:L9V"}
    likelihoods = {"A1G": -1.5, "L9V": 0.25, "A1G:L9V": -0.75}
    assert gm.four_state_interaction(likelihoods, cycle) == pytest.approx(-0.75 + 1.5 - 0.25)
    assert gm.additive_prediction(likelihoods, cycle) == pytest.approx(-1.25)


def test_paired_contrast_uses_the_group_and_ignores_a_half_pair():
    rows = [
        {"group": "g1", "origin": "generated", "sequence_id": "a", "value": 3.0},
        {"group": "g1", "origin": "natural", "sequence_id": "b", "value": 1.0},
        {"group": "g2", "origin": "generated", "sequence_id": "c", "value": 9.0},
        {"group": "g2", "origin": "natural", "sequence_id": "d", "value": 5.0},
        {"group": "g3", "origin": "generated", "sequence_id": "e", "value": 100.0},
    ]
    record = gm.paired_origin_contrast(rows, value="value", draws=50)
    assert record["paired_groups"] == 2
    assert record["difference"]["point"] == pytest.approx(3.0)
    assert record["generated"]["groups"] == 3


def test_paired_contrast_averages_rows_inside_a_sequence_before_the_group():
    rows = [
        {"group": "g1", "origin": "generated", "sequence_id": "a", "value": 1.0},
        {"group": "g1", "origin": "generated", "sequence_id": "a", "value": 3.0},
        {"group": "g1", "origin": "generated", "sequence_id": "b", "value": 10.0},
        {"group": "g1", "origin": "natural", "sequence_id": "c", "value": 0.0},
    ]
    record = gm.paired_origin_contrast(rows, value="value", draws=50)
    assert record["difference"]["point"] == pytest.approx(6.0)


def test_precision_record_reports_an_unresolved_interval_rather_than_a_zero():
    record = gm.precision_record({"interval": [-1.0, 1.0], "groups": 12})
    assert record["half_width"] == pytest.approx(1.0)
    assert record["minimum_detectable_80"] > record["half_width"]
    assert gm.precision_record({"interval": None, "undefined": "too few"})["half_width"] is None


def test_a_retokenised_mutation_has_no_position_resolved_reading():
    # Same token count, different boundaries: a shape check alone would pass and
    # then compare the likelihood of different residues.
    wild = {"sequence": "A" * 8, "ids": [9, 1, 2, 3, 4], "span": [1, 5],
            "counts": [2, 2, 2, 2], "offset": 0}
    mutant = {"sequence": "A" * 3 + "G" + "A" * 4, "ids": [9, 1, 7, 8, 4],
              "span": [1, 5], "counts": [2, 1, 3, 2], "offset": 0}
    census, reason = gm.mutation_receivers(
        {}, states=[wild, mutant], paradigm="causal_next_token", index=0, site=3
    )
    assert census is None
    assert reason


def test_an_alignment_disagreement_with_the_producer_is_refused(tmp_path):
    extraction = gm.Extraction(
        arm="prollama",
        paradigm="causal_next_token",
        root=tmp_path,
        files={"a": "a.npz"},
        reasons={},
        misaligned={"a": frozenset({"A1G"})},
        completion={},
    )
    extraction.agrees_on_alignment("a", ["A1G"])
    extraction.agrees_on_alignment("unrecorded", ["anything"])
    with pytest.raises(ValueError, match="disagree with the producing"):
        extraction.agrees_on_alignment("a", ["A1G", "L9V"])


def test_covered_separates_a_recorded_absence_from_a_defect(tmp_path):
    extraction = gm.Extraction(
        arm="progen2-small",
        paradigm="causal_next_token",
        root=tmp_path,
        files={"a": "a.npz"},
        reasons={"b": "1200 residues exceeds --max-residues"},
        misaligned={},
        completion={},
    )
    present, absent = extraction.covered(["a", "b", "c"])
    assert present == ["a"]
    assert [row["assay"] for row in absent] == ["b", "c"]
    assert "without a recorded reason" in dict(
        (row["assay"], row["reason"]) for row in absent
    )["c"]


def test_cohort_assay_refuses_an_inconsistent_row():
    with pytest.raises(ValueError, match="one sequence per mutant"):
        gm.cohort_assay(assay="x", wildtype=SEQ_A, mutants=["A1G"], sequences=[], cluster="g")
    with pytest.raises(ValueError, match="duplicate mutant"):
        gm.cohort_assay(assay="x", wildtype=SEQ_A, mutants=["A1G", "A1G"],
                        sequences=[SEQ_A, SEQ_A], cluster="g")
    with pytest.raises(ValueError, match="non-AA20"):
        gm.cohort_assay(assay="x", wildtype="XXXX", mutants=["X1G"], sequences=["G" * 4],
                        cluster="g")
