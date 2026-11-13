"""The two inputs the 2026-11-12 endpoints run restores: stream labels and a selector score.

Both producers live in
``scripts/capability/generation/objective_tuned_endpoints_20261112/``. Neither
computes a statistic: one splits the 2026-11-11 driver's own pooled table into
the three declared streams, the other reads the arms' own likelihoods into the
shape ``analyse_objective_tuned.selector_competence`` declares. So what has to
hold is that the split is a partition the module can resample, that it refuses a
label it cannot trust rather than pooling around it, and that the selector rows
actually *join* -- an unmatched digest makes an arm ``unresolved`` inside the
module instead of failing, which is the one failure mode that would pass silently.

The selector's direction is pinned here too, in both directions: a likelihood
that orders the pool exactly as the endpoint does must capture the whole oracle
headroom, and one that orders it backwards must capture a negative share.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS = REPO_ROOT / "scripts/capability/generation"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyse_objective_tuned as aot  # noqa: E402

from src.capability.generation import objective_tuned_analysis as ota  # noqa: E402
from src.capability.generation import objective_tuned_arms as ot  # noqa: E402
from src.capability.generation.structure_evidence import sequence_digest  # noqa: E402

PRODUCERS = SCRIPTS / "objective_tuned_endpoints_20261112"
ARMS = sorted({arm for pair in ot.pairs_in_lane(ot.LANE_FREE_LENGTH) for arm in pair.arms})


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PRODUCERS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


streams = _load("run_stream_endpoints")
selector = _load("build_selector_scores")


# ------------------------------------------------------------------- the split

def pooled_table(arm: str, digests: list[str]) -> dict:
    """The shape the 2026-11-11 driver hands on: one pool per arm, stream zero."""

    rows = [{"key": f"{arm}|s0|{index}", "cell": "eighteen_shard_18of68", "digest": digest,
             "sequence": "A" * (10 + index), "length": 10 + index,
             "stop_status": "budget_censored" if index % 4 == 0 else "native_terminal",
             "empty": False, "valid_aa20": True, "plddt": 50.0 + index,
             "identity": 10.0, "stratum": ota.NOVEL_STRATUM}
            for index, digest in enumerate(digests)]
    return {"pools": {(arm, 0): {"arm": arm, "stream": 0, "role": "base", "pair": "pair",
                                 "cells": ["eighteen_shard_18of68"], "seeds": [],
                                 "checkpoints": [], "rows": rows,
                                 "counts": {"attempts": len(rows)}}},
            "join_rule": "x", "denominator_rule": "y"}


def labels_for(arm: str, digests: list[str], *, streams_cycle=(0, 1, 2)) -> dict:
    return {digest: {"sequence_sha256": digest, "arm": arm,
                     "stream": str(streams_cycle[index % len(streams_cycle)]),
                     "replicate": "0",
                     "cell_label": f"ot_{arm}_s{streams_cycle[index % len(streams_cycle)]}_r0"}
            for index, digest in enumerate(digests)}


@pytest.fixture
def digests() -> list[str]:
    return [sequence_digest(f"SEQ{index}") for index in range(9)]


def test_the_split_is_a_partition_the_module_can_resample(digests):
    arm = ARMS[0]
    table = streams.split_streams(pooled_table(arm, digests), labels_for(arm, digests))
    assert sorted(table["pools"]) == [(arm, 0), (arm, 1), (arm, 2)]
    assert sum(len(pool["rows"]) for pool in table["pools"].values()) == len(digests)
    keys = [row["key"] for pool in table["pools"].values() for row in pool["rows"]]
    assert len(set(keys)) == len(keys)
    assert {digest for pool in table["pools"].values() for digest in
            {row["digest"] for row in pool["rows"]}} == set(digests)
    for (_, stream), pool in table["pools"].items():
        assert pool["stream"] == stream
        assert pool["counts"]["attempts"] == pool["counts"]["folded"] == len(pool["rows"])
        assert pool["counts"]["budget_censored"] == sum(
            1 for row in pool["rows"] if row["stop_status"] == "budget_censored")
        assert pool["cells"] == [f"ot_{arm}_s{stream}_r0"]
    # The module reads the split as three streams, which is what the declared
    # stream-equal pool mean and the per-(arm, stream) floor are stated over.
    pools = aot.pool_of(table, arm)
    assert sorted(pools) == [0, 1, 2]
    assert sorted(aot.plddt_by_stratum(pools, ota.NOVEL_STRATUM)) == [0, 1, 2]


def test_an_unlabelled_structure_refuses_rather_than_being_pooled(digests):
    arm = ARMS[0]
    labels = labels_for(arm, digests)
    labels.pop(digests[3])
    with pytest.raises(SystemExit, match="carries no recovered stream label"):
        streams.split_streams(pooled_table(arm, digests), labels)


def test_a_label_naming_another_arm_refuses(digests):
    arm, other = ARMS[0], ARMS[1]
    labels = labels_for(arm, digests)
    labels[digests[2]] = {**labels[digests[2]], "arm": other}
    with pytest.raises(SystemExit, match="the recovered label says"):
        streams.split_streams(pooled_table(arm, digests), labels)


def test_a_cell_label_that_does_not_spell_its_own_stream_refuses(digests):
    arm = ARMS[0]
    labels = labels_for(arm, digests)
    labels[digests[1]] = {**labels[digests[1]], "cell_label": f"ot_{arm}_s0_r0"}
    with pytest.raises(SystemExit, match="does not spell arm"):
        streams.split_streams(pooled_table(arm, digests), labels)


def test_an_arm_missing_a_declared_stream_refuses(digests):
    arm = ARMS[0]
    with pytest.raises(SystemExit, match="against the declared"):
        streams.split_streams(pooled_table(arm, digests),
                              labels_for(arm, digests, streams_cycle=(0, 1)))


# ---------------------------------------------------------------- the selector

def selector_table(arm: str, plddt: list[float]) -> dict:
    rows = [{"key": f"{arm}|s0|{index}", "digest": sequence_digest(f"S{index}"),
             "plddt": value} for index, value in enumerate(plddt)]
    return {"pools": {(arm, 0): {"arm": arm, "rows": rows}}}


def write_selector(path: Path, arm: str, rows: list[tuple[str, float]]) -> Path:
    path.write_text("".join(json.dumps({"arm": arm, "sequence_sha256": digest, "score": score})
                            + "\n" for digest, score in rows), encoding="utf-8")
    return path


def test_a_selector_that_orders_the_pool_as_the_endpoint_does_captures_all_of_it(tmp_path):
    arm = ARMS[0]
    plddt = [float(index) for index in range(20)]
    table = selector_table(arm, plddt)
    path = write_selector(tmp_path / "s.jsonl", arm,
                          [(row["digest"], row["plddt"])
                           for pool in table["pools"].values() for row in pool["rows"]])
    entry = aot.selector_competence(table, path)["per_arm"][arm]
    assert "status" not in entry["captured"]
    assert entry["captured"]["captured_fraction"] == pytest.approx(1.0)
    assert entry["captured"]["k"] == ota.SELECTION_BUDGET


def test_a_selector_that_orders_the_pool_backwards_captures_a_negative_share(tmp_path):
    arm = ARMS[0]
    table = selector_table(arm, [float(index) for index in range(20)])
    path = write_selector(tmp_path / "s.jsonl", arm,
                          [(row["digest"], -row["plddt"])
                           for pool in table["pools"].values() for row in pool["rows"]])
    captured = aot.selector_competence(table, path)["per_arm"][arm]["captured"]
    assert captured["captured_fraction"] < 0.0


def test_a_score_that_misses_one_folded_attempt_leaves_the_arm_unresolved(tmp_path):
    arm = ARMS[0]
    table = selector_table(arm, [float(index) for index in range(20)])
    rows = [(row["digest"], row["plddt"])
            for pool in table["pools"].values() for row in pool["rows"]][:-1]
    path = write_selector(tmp_path / "s.jsonl", arm, rows)
    captured = aot.selector_competence(table, path)["per_arm"][arm]["captured"]
    assert captured["status"] == ota.UNRESOLVED


# ------------------------------------------------------- the selector's builder

def write_sources(root: Path, *, tokens: dict[str, int] | None = None,
                  drop: str | None = None) -> dict[str, Path]:
    """A two-structure folded sample per declared arm, with its own-model scores."""

    tokens = tokens or {}
    likelihood = root / "likelihood"
    likelihood.mkdir(parents=True, exist_ok=True)
    cohort, neighbour = [], ["arm\tsequence_sha256"]
    for arm in ARMS:
        rows = []
        for index in range(2):
            sequence = f"MKV{arm}{index}".replace("-", "").replace(".", "")
            digest = sequence_digest(sequence)
            cohort.append(json.dumps({"id": digest, "sequence": sequence,
                                      "length": len(sequence)}))
            neighbour.append(f"{arm}\t{digest}")
            if drop == arm and index == 1:
                continue
            count = tokens.get(f"{arm}:{index}", len(sequence))
            rows.append(json.dumps({"sequence_sha256": digest, "n_tokens": count,
                                    "sum_nll": 2.0 * count, "mean_nll": 2.0}))
        (likelihood / f"{arm}__on__{arm}.jsonl").write_text("\n".join(rows) + "\n",
                                                            encoding="utf-8")
    (root / "cohort.jsonl").write_text("\n".join(cohort) + "\n", encoding="utf-8")
    (root / "neighbour.tsv").write_text("\n".join(neighbour) + "\n", encoding="utf-8")
    return {"likelihood": likelihood, "cohort": root / "cohort.jsonl",
            "neighbour": root / "neighbour.tsv"}


def run_builder(root: Path, paths: dict[str, Path]) -> dict:
    out = root / "out"
    sys.argv = ["build_selector_scores.py", "--likelihood", str(paths["likelihood"]),
                "--neighbour", str(paths["neighbour"]), "--cohort", str(paths["cohort"]),
                "--out", str(out)]
    selector.main()
    return json.loads((out / "selector_score.json").read_text(encoding="utf-8"))


def test_the_builder_writes_rows_that_join_under_the_modules_own_digest_rule(tmp_path):
    paths = write_sources(tmp_path)
    record = run_builder(tmp_path, paths)
    assert record["n_rows"] == 2 * len(ARMS)
    rows = [json.loads(line) for line in
            (tmp_path / "out/selector_score.jsonl").read_text(encoding="utf-8").splitlines()]
    # The score is -mean_nll, so a lower negative log-likelihood ranks higher.
    assert all(row["score"] == pytest.approx(-row["mean_nll_per_residue_nats"]) for row in rows)
    for row in rows:
        assert aot.digest_of(row, paths["likelihood"], "selector row") == row["sequence_sha256"]


def test_the_builder_refuses_a_folded_attempt_with_no_score(tmp_path):
    paths = write_sources(tmp_path, drop=ARMS[2])
    with pytest.raises(SystemExit, match="carry no score"):
        run_builder(tmp_path, paths)


def test_the_builder_refuses_a_score_computed_over_the_wrong_length(tmp_path):
    paths = write_sources(tmp_path, tokens={f"{ARMS[0]}:0": 3})
    with pytest.raises(SystemExit, match="not the sequence's length"):
        run_builder(tmp_path, paths)


def test_the_builder_refuses_a_mean_that_is_not_the_sum_over_the_count(tmp_path):
    paths = write_sources(tmp_path)
    path = paths["likelihood"] / f"{ARMS[0]}__on__{ARMS[0]}.jsonl"
    first, rest = path.read_text(encoding="utf-8").split("\n", 1)
    row = json.loads(first)
    row["mean_nll"] = row["mean_nll"] + 1.0
    path.write_text(json.dumps(row) + "\n" + rest, encoding="utf-8")
    with pytest.raises(SystemExit, match="is not sum_nll/n_tokens"):
        run_builder(tmp_path, paths)


def test_the_builder_refuses_a_duplicate_digest_in_one_arms_scores(tmp_path):
    paths = write_sources(tmp_path)
    path = paths["likelihood"] / f"{ARMS[0]}__on__{ARMS[0]}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines + [lines[0]]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="appears twice"):
        run_builder(tmp_path, paths)


# ------------------------------------- the 2026-11-12 amendment and attribution

attribution = _load("run_progen2_attribution")


def test_the_seven_finetuning_families_come_from_the_arm_declaration():
    assert len(attribution.SEVEN) == 7
    assert all(family in ot.OBJECTIVE_PFAM_SEVEN_FAMILIES for family in attribution.SEVEN)


def test_the_amendments_control_families_are_declared_sized_and_seeded_apart():
    families = set(ota.SECONDARY_CONTROL_FAMILIES.values())
    assert len(ota.SECONDARY_CONTROL_FAMILIES) == len(ota.FILTER_RULE_NAMES)
    assert not families & set(ota.CONTROL_FAMILIES.values())
    for family in families:
        assert ota.FAMILY_SIZES[family] == len(ota.SECONDARY_PAIRS) * len(ota.STRATA)
        assert family in ota.FAMILY_SEED_OFFSETS
    seeds = [ota.FAMILY_SEED_OFFSETS[family] for family in families]
    assert len(set(seeds)) == len(seeds)
    assert not set(seeds) & {ota.FAMILY_SEED_OFFSETS[other] for other in
                             ota.CONTROL_FAMILIES.values()}
    assert "2026-11-12" in ota.SECONDARY_CONTROL_AMENDMENT


def test_the_declared_primary_control_family_still_refuses_a_four_column_realisation():
    """The amendment was necessary: the old declaration does not absorb this pair."""

    keys = [f"progen2-small-mix7|{stratum}" for stratum in ota.STRATA]
    with pytest.raises(ValueError, match="declares 12 columns and realises 4"):
        ota.require_declared_family(keys, ota.CONTROL_FAMILIES["low_complexity_residues"])
    ota.require_declared_family(keys, ota.SECONDARY_CONTROL_FAMILIES["low_complexity_residues"])


def test_a_band_on_the_amendments_family_is_the_same_construction_on_four_columns():
    import numpy as np

    rng = np.random.default_rng(12)
    family = ota.SECONDARY_CONTROL_FAMILIES["low_complexity_residues"]
    columns = []
    for index, stratum in enumerate(ota.STRATA):
        members = [{"key": f"m{position:03d}", "plddt": float(60 + 5 * rng.standard_normal()),
                    "stratum": stratum, "score": float(rng.random())}
                   for position in range(40)]
        draws = ota.nested_base_draws(members, retain="highest", family=family,
                                      column=index, stream_index=0, draws=200)
        mean = float(np.mean([row["plddt"] for row in members]))
        columns.append({
            "key": f"progen2-small-mix7|{stratum}",
            "primary": {"status": "resolved", "zero_verdict": "positive", "point": 6.0,
                        "interval": [4.0, 8.0]},
            "base": {0: draws[stratum]["base"]}, "filtered": {0: draws[stratum]["filtered"]},
            "valid": {0: draws[stratum]["valid"]},
            "tuned": {0: [row["plddt"] + 6.0 for row in members]},
            "base_realised": mean, "filtered_realised": mean + 3.0,
            "tuned_realised": mean + 6.0,
        })
    record = ota.reproduced_fraction_bands(columns, family=family, draws=200)
    assert record["declared_family_size"] == ota.SECONDARY_FAMILY_SIZE
    assert record["control_status"] == "readable"
    assert record["resolved_family_size"] == ota.SECONDARY_FAMILY_SIZE
    assert record["method"] == ota.NATIVE_BAND_METHOD


def test_a_family_set_that_does_not_name_all_three_rules_refuses():
    with pytest.raises(SystemExit, match="must name all three declared rules"):
        aot.explicit_statistics_control({"pools": {}}, ota.SECONDARY_PAIRS, {},
                                        {"bands": {"columns": {}}},
                                        {"low_complexity_residues": "x"})


def _annotated(arm: str, digests: list[str], families: dict[int, list[str]]) -> dict:
    return {digest: families.get(index, [])
            for index, digest in enumerate(digests)}


def test_removal_drops_exactly_the_in_seven_rows_and_leaves_other_arms_alone(digests):
    pair = ota.SECONDARY_PAIRS[0]
    outside = ota.PRIMARY_PAIRS[0].base
    assert outside not in pair.arms
    table = pooled_table(pair.tuned, digests)
    table["pools"][(outside, 0)] = pooled_table(outside, digests)["pools"][(outside, 0)]
    inside = attribution.SEVEN[0]
    annotations = _annotated(pair.tuned, digests,
                             {1: [inside], 4: ["PF99999", inside], 6: ["PF99999"]})
    corrected = attribution.remove_in_seven(table, set(pair.arms), annotations)
    kept = {row["digest"] for row in corrected["pools"][(pair.tuned, 0)]["rows"]}
    assert kept == {digest for index, digest in enumerate(digests) if index not in (1, 4)}
    # An arm outside the pair is passed through untouched, rows and all.
    assert (corrected["pools"][(outside, 0)]["rows"]
            is table["pools"][(outside, 0)]["rows"])


def test_removal_refuses_a_row_the_oracle_never_annotated(digests):
    pair = ota.SECONDARY_PAIRS[0]
    table = pooled_table(pair.tuned, digests)
    annotations = _annotated(pair.tuned, digests, {})
    annotations.pop(digests[2])
    with pytest.raises(SystemExit, match="carries no oracle annotation"):
        attribution.remove_in_seven(table, set(pair.arms), annotations)


def test_an_unsearched_annotation_refuses_rather_than_counting_as_no_family(tmp_path):
    path = tmp_path / "family_recognition.jsonl"
    path.write_text(json.dumps({"arm": ARMS[0], "sequence_sha256": sequence_digest("A"),
                                "pfam_families": [], "profile_search_status": "skipped"}) + "\n",
                    encoding="utf-8")
    with pytest.raises(SystemExit, match="is not measured"):
        attribution.read_annotations(path, {ARMS[0]})
