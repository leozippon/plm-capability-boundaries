"""Conditions the frozen-evaluator requalification must always satisfy.

These are negative invariants over small synthetic fixtures, not a unit test of
the frozen science: they pin the *boundary* the requalification exists to defend.
The measured outcomes on every excluded row are poisoned with ``NaN`` sentinels
wherever the fixture permits, so a test that reads one fails loudly rather than
passing by accident. The head weights here are explicitly synthetic and are
labelled as such; nothing in this file claims a laboratory or model result.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest

from src.capability.core.io import sha256_file
from src.capability.evaluation import domain_stability as ds

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts/capability/evaluation/requalify_frozen_domain_stability.py"
_spec = importlib.util.spec_from_file_location("requalify_frozen_domain_stability", _SCRIPT)
assert _spec and _spec.loader
rq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rq)

#: The same twelve reviewed targets the protocol withholds, used here only as
#: synthetic fixture labels.
TARGETS = [
    "1E0L.pdb",
    "1F0M.pdb",
    "1IFY.pdb",
    "1K1V.pdb",
    "1UFM.pdb",
    "1V1C.pdb",
    "2K1B.pdb",
    "2KWH.pdb",
    "2LGW.pdb",
    "2LVN.pdb",
    "3DKM.pdb",
    "3L1X.pdb",
]
#: A stand-in for the historical row digest 678e8b...; the fixture's current rows
#: hash to something else, which is the point of the distinction being recorded.
HISTORICAL = "678e8b918bb1c16f419400b4b8f1744156bf34940de549f3c37c33f4607541a3"
POISON = float("nan")


def _seq(name: str, length: int) -> str:
    generator = random.Random(name)
    return "".join(generator.choice(ds.AA20) for _ in range(length))


def _digest(sequence: str) -> str:
    return hashlib.blake2b(sequence.encode("utf-8"), digest_size=16).hexdigest()


def _row(identifier, sequence, length, *, split, dataset, measured, group=None,
         wild_type_name=None, is_wild_type=None, table="fit", role=None):
    row = {
        "id": identifier,
        "sequence": sequence,
        "length": int(length),
        "measured_delta_g": measured,
        "split": split,
        "dataset": dataset,
        "table": table,
    }
    if group is not None:
        row["group"] = group
    if wild_type_name is not None:
        row["wild_type_name"] = wild_type_name
        row["is_wild_type"] = bool(is_wild_type)
    if role is not None:
        row["role"] = role
    return row


def _build_root(root: Path, *, width: int = 4, draws: int = 200, mutate_rows=None, mutate_cohort=None,
                omit_index_ids=(), omit_row_ids=()):
    """Write a miniature but structurally faithful frozen-evaluator tree.

    ``omit_index_ids`` / ``omit_row_ids`` build the partial-current-table cases: an
    apply row left out of the saved index (the permitted extra-application case) or
    an index id with no current row (which must refuse).
    """

    root = Path(root)
    fit_dir = root / "results/external_baseline/20261008101801_33265fa60eac/stabdg_fit_and_validate_cpu"
    embed_dir = root / "results/external_baseline/20261008101801_33265fa60eac/stabdg_embed_esm2_gpu2"
    rows_path = root / "results/R6/domain_stability_20261008/domain_stability_rows.jsonl"
    cohort_path = root / "results/R4/gate_stability_20260924/cohort/cohort.json"
    roles_path = root / "data/r6_constraint_response_20261013/diagnostic_protocol.json"
    for directory in (fit_dir, embed_dir, rows_path.parent, cohort_path.parent, roles_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    tags: dict[str, str] = {}
    mg_sequences: list[str] = []

    for split in rq.USED_SPLITS:
        for index in range(2):
            name = f"mg-{split}-{index}"
            length = 60 + index
            sequence = _seq(name, length)
            mg_sequences.append(sequence)
            identifier = "mg_" + _digest(sequence)
            rows.append(_row(identifier, sequence, length, split=split,
                             dataset="synthetic_mgnify", measured=0.31, group=identifier,
                             table="fit"))
            tags[identifier] = "mg"

    # Collision cluster: the wild type repeats an MG training sequence, so its
    # derived suffix must be cut by the exact fit-sequence rule. Its second member
    # is distinct and must remain eligible.
    collision = mg_sequences[0]
    for offset, sequence in enumerate((collision, _seq("collision-distinct", 63))):
        identifier = "ms_" + _digest(sequence)
        rows.append(_row(identifier, sequence, len(sequence), split="transfer",
                         dataset="synthetic_megascale", measured=0.42, group="200",
                         wild_type_name="COL.pdb" if offset == 0 else "COL.pdb_A1V",
                         is_wild_type=offset == 0, table="transfer"))
        tags[identifier] = "collision" if offset == 0 else "eligible"

    cohort = []
    for index, target in enumerate(TARGETS):
        length = 60 + (index % 8)
        wild = _seq(f"wt-{target}", length)
        variants = [_seq(f"var-{target}-{variant}", length) for variant in range(3)]
        for offset, sequence in enumerate([wild, *variants]):
            identifier = "ms_" + _digest(sequence)
            rows.append(_row(identifier, sequence, length, split="transfer",
                             dataset="synthetic_megascale", measured=0.17,
                             group=str(index + 1),
                             wild_type_name=target if offset == 0 else f"{target}_A{offset}V",
                             is_wild_type=offset == 0, table="transfer"))
            tags[identifier] = "target_wt" if offset == 0 else "target_variant"
        cohort.append({"name": target, "cluster": str(index + 1), "group": f"nat-{index:03d}",
                       "length": length, "sequences": [wild, *variants]})

    for cluster in range(10):
        group = str(100 + cluster)
        length = 60 + cluster
        for offset, sequence in enumerate(
            [_seq(f"elig-{cluster}-{member}", length) for member in range(3)]
        ):
            identifier = "ms_" + _digest(sequence)
            rows.append(_row(identifier, sequence, length, split="transfer",
                             dataset="synthetic_megascale",
                             measured=0.5 + 0.05 * cluster + 0.001 * offset, group=group,
                             wild_type_name=f"ELIG{cluster}.pdb" if offset == 0
                             else f"ELIG{cluster}.pdb_A{offset}V",
                             is_wild_type=offset == 0, table="transfer"))
            tags[identifier] = "eligible"

    for offset, length in enumerate((40, 85)):
        sequence = _seq(f"out-of-band-{offset}", length)
        identifier = "ms_" + _digest(sequence)
        rows.append(_row(identifier, sequence, length, split="transfer",
                         dataset="synthetic_megascale", measured=0.5, group="150",
                         wild_type_name="OUT.pdb" if offset == 0 else "OUT.pdb_A1V",
                         is_wild_type=offset == 0, table="transfer"))
        tags[identifier] = "out_of_band"

    for index in range(2):
        identifier = f"gen-{index:04d}"
        sequence = _seq(identifier, 60 + 3 * index)
        rows.append(_row(identifier, sequence, len(sequence), split="apply",
                         dataset="synthetic_cohort", measured=None, group=identifier,
                         table="apply", role="generated"))
        tags[identifier] = "apply"

    # Poison every excluded row's outcome with a NaN sentinel. Nothing may read it.
    for row in rows:
        if tags[row["id"]] in {"target_wt", "target_variant", "collision", "out_of_band"}:
            row["measured_delta_g"] = POISON

    cohort = [background for background in cohort]
    if mutate_cohort is not None:
        cohort = mutate_cohort(cohort)
    if mutate_rows is not None:
        mutate_rows(rows, tags)

    omit_index = set(omit_index_ids)
    index_source = list(rows)
    if omit_row_ids:
        rows = [row for row in rows if row["id"] not in set(omit_row_ids)]
    # An unindexed apply row is an added application record whose measurement must
    # never be touched; poison it to prove the reader leaves it alone.
    for row in rows:
        if row["id"] in omit_index and row["split"] == "apply":
            row["measured_delta_g"] = POISON
    index_rows = [row for row in index_source if row["id"] not in omit_index]

    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )

    generator = np.random.default_rng(11)
    vectors = generator.normal(0.0, 1.0, (len(index_rows), width)).astype(np.float16)
    array_path = embed_dir / "embeddings.npy"
    np.save(array_path, vectors)
    index_path = embed_dir / "embedding_index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "d1_stability_embeddings_v1",
                "ids": [row["id"] for row in index_rows],
                "splits": [row["split"] for row in index_rows],
                "lengths": [row["length"] for row in index_rows],
            }
        ),
        encoding="utf-8",
    )
    metadata_path = embed_dir / "stability_embeddings.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "d1_stability_embeddings_v1",
                "status": "complete",
                "n_rows": len(index_rows),
                "width": width,
                "dtype_stored": "float16",
                "dtype_compute": "bfloat16",
                "pooling": "mean over residue positions of the last hidden state, excluding special tokens",
                "pooler_instantiated": False,
                "rows_sha256": HISTORICAL,
                "embeddings_sha256": sha256_file(array_path),
                "model": "synthetic/esm2-test",
            }
        ),
        encoding="utf-8",
    )

    plm_head = fit_dir / "model_plm.npz"
    baseline_head = fit_dir / "model_baseline.npz"
    np.savez(
        plm_head,
        coefficients=generator.normal(0.0, 1.0, len(ds.BASELINE_FEATURES) + width),
        intercept=np.asarray([0.25]),
        mean=np.zeros(len(ds.BASELINE_FEATURES) + width),
        scale=np.ones(len(ds.BASELINE_FEATURES) + width),
    )
    np.savez(
        baseline_head,
        coefficients=generator.normal(0.0, 1.0, len(ds.BASELINE_FEATURES)),
        intercept=np.asarray([-0.1]),
        mean=np.zeros(len(ds.BASELINE_FEATURES)),
        scale=np.ones(len(ds.BASELINE_FEATURES)),
    )
    fit_report = fit_dir / "domain_stability_fit.json"
    fit_report.write_text(
        json.dumps({"schema_version": "synthetic", "status": "complete", "licensed_band": [60, 80]}),
        encoding="utf-8",
    )

    cohort_path.write_text(json.dumps({"backgrounds": cohort}), encoding="utf-8")
    roles_path.write_text(
        json.dumps({"roles": {"development": TARGETS[:4], "evaluation": TARGETS[4:]}}),
        encoding="utf-8",
    )

    pins = {
        "fit_report_sha256": sha256_file(fit_report),
        "plm_head_sha256": sha256_file(plm_head),
        "baseline_head_sha256": sha256_file(baseline_head),
        "embedding_metadata_sha256": sha256_file(metadata_path),
        "embeddings_sha256": sha256_file(array_path),
        "reviewed_cohort_sha256": sha256_file(cohort_path),
        "reviewed_roles_protocol_sha256": sha256_file(roles_path),
    }
    protocol_path = root / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "schema": rq.PROTOCOL_SCHEMA,
                "frozen_date": "2026-10-13",
                "purpose": "synthetic fixture",
                "source_version": {
                    "historical_rows_sha256": HISTORICAL,
                    "current_rows_sha256": sha256_file(rows_path),
                    "historical_rows_recovered": False,
                    "cause_of_difference": "synthetic",
                    "original_fit_and_validation_report_unchanged": True,
                },
                "pins": pins,
                "preparation": {
                    "outcomes_used": False,
                    "reviewed_targets": TARGETS,
                    "eligible_population": "synthetic",
                },
                "validation": {
                    "quantity": "synthetic",
                    "predictors": "synthetic",
                    "features": "synthetic",
                    "groups": "synthetic",
                    "seed": 20261008,
                    "bootstrap_draws": draws,
                    "G1": "synthetic G1",
                    "G2": "synthetic G2",
                    "joint_rule": "synthetic joint rule",
                    "implementation": "domain_stability.transfer_contrast and evaluate_gate",
                },
                "limits": ["synthetic fixture; no laboratory claim"],
            }
        ),
        encoding="utf-8",
    )

    eligible_ids = [identifier for identifier, tag in tags.items() if tag == "eligible"]
    return {
        "root": root,
        "protocol": protocol_path,
        "paths": rq.input_paths(root),
        "rows": rows,
        "tags": tags,
        "eligible_ids": eligible_ids,
        "plm_head_sha256": pins["plm_head_sha256"],
        "width": width,
        "draws": draws,
    }


def _prepare(fx, out):
    return rq.prepare(fx["protocol"], fx["root"], out)


def _prepare_and_evaluate(fx, tmp_path, device="cpu"):
    prepared = tmp_path / "prep"
    _prepare(fx, prepared)
    out = tmp_path / "eval"
    digest = sha256_file(prepared / rq.PREPARATION_COMPLETION)
    record = rq.evaluate(fx["protocol"], fx["root"], prepared, digest, device, out)
    return prepared, digest, record


# --------------------------------------------------- the exclusion boundary


def test_prepare_freezes_the_allowlist_without_touching_excluded_outcomes(tmp_path):
    fx = _build_root(tmp_path / "root")
    preparation = _prepare(fx, tmp_path / "prep")

    assert preparation["status"] == "prepared"
    assert preparation["outcomes_used"] is False
    assert preparation["no_measured_outcomes_read"] is True
    assert set(preparation["targets"]["excluded_groups"]) == {str(i + 1) for i in range(12)}

    lines = (tmp_path / "prep" / rq.ELIGIBLE_ROWS).read_text(encoding="utf-8").splitlines()
    keep = {json.loads(line)["id"] for line in lines}
    assert all("measured_delta_g" not in json.loads(line) for line in lines)
    assert keep == set(fx["eligible_ids"])
    assert preparation["eligible"]["n"] == len(fx["eligible_ids"])
    for identifier, tag in fx["tags"].items():
        if tag in {"target_wt", "target_variant", "collision", "out_of_band"}:
            assert identifier not in keep
    for identifier in keep:
        assert identifier[3:] == _digest(
            next(row["sequence"] for row in fx["rows"] if row["id"] == identifier)
        )


def test_exact_fit_sequence_overlap_is_cut_before_selection(tmp_path):
    fx = _build_root(tmp_path / "root")
    preparation = _prepare(fx, tmp_path / "prep")
    collision = [identifier for identifier, tag in fx["tags"].items() if tag == "collision"]
    assert len(collision) == 1
    assert preparation["accounting"]["transfer_fit_sequence_overlap"] == 1
    assert collision[0] not in {json.loads(line)["id"] for line in
                                (tmp_path / "prep" / rq.ELIGIBLE_ROWS).read_text().splitlines()}


def test_a_missing_or_inconsistent_target_link_refuses(tmp_path):
    def drop_one(backgrounds):
        return [b for b in backgrounds if b["name"] != "2LVN.pdb"]

    fx = _build_root(tmp_path / "root", mutate_cohort=drop_one)
    with pytest.raises(rq.Refusal, match="2LVN.pdb"):
        _prepare(fx, tmp_path / "prep")

    def break_cluster(backgrounds):
        for background in backgrounds:
            if background["name"] == "1UFM.pdb":
                background["cluster"] = "999"
        return backgrounds

    fx = _build_root(tmp_path / "root2", mutate_cohort=break_cluster)
    with pytest.raises(rq.Refusal, match="inconsistent"):
        _prepare(fx, tmp_path / "prep2")


def test_removing_the_wild_type_row_refuses_rather_than_assuming_absence(tmp_path):
    def drop_wild(rows, tags):
        rows[:] = [row for row in rows if tags[row["id"]] != "target_wt"]

    fx = _build_root(tmp_path / "root", mutate_rows=drop_wild)
    with pytest.raises(rq.Refusal):
        _prepare(fx, tmp_path / "prep")


# ------------------------------------------------------- table/index identity


def test_duplicate_and_wrong_identifiers_are_refused(tmp_path):
    def duplicate(rows, tags):
        rows.append(dict(rows[0]))

    with pytest.raises(rq.Refusal, match="duplicate"):
        _prepare(_build_root(tmp_path / "a", mutate_rows=duplicate), tmp_path / "pa")

    def corrupt(rows, tags):
        for row in rows:
            if tags[row["id"]] == "eligible":
                row["id"] = row["id"][:-1] + ("0" if row["id"][-1] != "0" else "1")
                break

    with pytest.raises(rq.Refusal, match="blake2b16"):
        _prepare(_build_root(tmp_path / "b", mutate_rows=corrupt), tmp_path / "pb")


def test_index_length_and_split_mismatches_are_refused(tmp_path):
    for mutate, expected in (("lengths", "length"), ("splits", "split")):
        fx = _build_root(tmp_path / f"root-{mutate}")
        index_path = fx["paths"]["embedding_index"]
        index = json.loads(index_path.read_text(encoding="utf-8"))
        position = next(i for i, row in enumerate(fx["rows"]) if row["split"] == "transfer")
        if mutate == "lengths":
            index["lengths"][position] += 1
        else:
            index["splits"][position] = "train"
        index_path.write_text(json.dumps(index), encoding="utf-8")
        with pytest.raises(rq.Refusal, match=expected):
            _prepare(fx, tmp_path / f"out-{mutate}")


def test_extra_apply_row_without_embedding_is_disclosed_and_never_used(tmp_path):
    """The permitted extra-application case: counted and listed, never eligible."""

    fx = _build_root(tmp_path / "root", omit_index_ids={"gen-0000"})
    prepared = tmp_path / "prep"
    preparation = _prepare(fx, prepared)
    coverage = preparation["coverage"]
    assert coverage["unindexed_apply_count"] == 1
    assert coverage["unindexed_apply_ids"] == ["gen-0000"]
    assert coverage["n_current_rows"] == coverage["n_index_rows"] + 1
    assert preparation["outcomes_used"] is False
    keep = {
        json.loads(line)["id"]
        for line in (prepared / rq.ELIGIBLE_ROWS).read_text(encoding="utf-8").splitlines()
    }
    assert "gen-0000" not in keep

    # The poisoned measurement on the extra apply row must not disturb the gate run.
    digest = sha256_file(prepared / rq.PREPARATION_COMPLETION)
    record = rq.evaluate(fx["protocol"], fx["root"], prepared, digest, "cpu", tmp_path / "eval")
    assert record["status"] == "complete"
    assert record["eligibility"]["excluded_outcomes_never_read"] is True


def test_an_unindexed_transfer_row_refuses(tmp_path):
    victim = _build_root(tmp_path / "root")["eligible_ids"][0]
    fx = _build_root(tmp_path / "root2", omit_index_ids={victim})
    with pytest.raises(rq.Refusal, match="not an explicit apply-table application record"):
        _prepare(fx, tmp_path / "prep")


def test_an_indexed_row_missing_from_the_table_refuses(tmp_path):
    fx = _build_root(tmp_path / "root", omit_row_ids={"gen-0000"})
    with pytest.raises(rq.Refusal, match="no current row"):
        _prepare(fx, tmp_path / "prep")


def test_a_non_canonical_sequence_is_refused(tmp_path):
    def poison_sequence(rows, tags):
        for row in rows:
            if tags[row["id"]] == "eligible":
                row["sequence"] = "X" + row["sequence"][1:]
                break

    with pytest.raises(rq.Refusal, match="non-canonical"):
        _prepare(_build_root(tmp_path / "root", mutate_rows=poison_sequence), tmp_path / "prep")


# --------------------------------------------------------------- the reader


def test_evaluate_never_reads_excluded_outcomes_and_requires_finite_eligible(tmp_path):
    fx = _build_root(tmp_path / "root")
    _, _, record = _prepare_and_evaluate(fx, tmp_path)
    assert record["status"] == "complete"
    assert record["eligibility"]["excluded_outcomes_never_read"] is True
    assert isinstance(record["gates_passed"], bool)

    def nan_an_eligible(rows, tags):
        for row in rows:
            if tags[row["id"]] == "eligible":
                row["measured_delta_g"] = POISON
                break

    fx = _build_root(tmp_path / "root2", mutate_rows=nan_an_eligible)
    prepared = tmp_path / "prep2"
    _prepare(fx, prepared)
    digest = sha256_file(prepared / rq.PREPARATION_COMPLETION)
    with pytest.raises(rq.Refusal, match="non-finite measured value"):
        rq.evaluate(fx["protocol"], fx["root"], prepared, digest, "cpu", tmp_path / "eval2")


def test_evaluate_refuses_a_stale_preparation_or_changed_input(tmp_path):
    fx = _build_root(tmp_path / "root")
    prepared = tmp_path / "prep"
    _prepare(fx, prepared)
    digest = sha256_file(prepared / rq.PREPARATION_COMPLETION)

    with pytest.raises(rq.Refusal, match="preparation checksum mismatch"):
        rq.evaluate(fx["protocol"], fx["root"], prepared, "0" * 64, "cpu", tmp_path / "e1")

    fx["paths"]["rows"].write_bytes(fx["paths"]["rows"].read_bytes() + b"\n")
    with pytest.raises(rq.Refusal, match="digest"):
        rq.evaluate(fx["protocol"], fx["root"], prepared, digest, "cpu", tmp_path / "e2")

    # The index has no producer-time protocol pin, so a change after preparation is
    # caught against the preparation's own input record.
    fx2 = _build_root(tmp_path / "root2")
    prepared2 = tmp_path / "prep2"
    _prepare(fx2, prepared2)
    digest2 = sha256_file(prepared2 / rq.PREPARATION_COMPLETION)
    index_path = fx2["paths"]["embedding_index"]
    index_path.write_bytes(index_path.read_bytes() + b"\n")
    with pytest.raises(rq.Refusal, match="changed since preparation"):
        rq.evaluate(fx2["protocol"], fx2["root"], prepared2, digest2, "cpu", tmp_path / "e3")


def test_tampering_with_the_frozen_allowlist_is_refused(tmp_path):
    fx = _build_root(tmp_path / "root")
    prepared = tmp_path / "prep"
    _prepare(fx, prepared)
    digest = sha256_file(prepared / rq.PREPARATION_COMPLETION)
    eligible_path = prepared / rq.ELIGIBLE_ROWS
    records = [json.loads(line) for line in eligible_path.read_text().splitlines()]
    records[0]["length"] = int(records[0]["length"]) + 1
    eligible_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    with pytest.raises(rq.Refusal, match="eligible rows file digest"):
        rq.evaluate(fx["protocol"], fx["root"], prepared, digest, "cpu", tmp_path / "eval")


def test_evaluate_rejects_a_gpu_and_hides_cuda(tmp_path):
    fx = _build_root(tmp_path / "root")
    with pytest.raises(rq.Refusal, match="CPU only"):
        rq.evaluate(fx["protocol"], fx["root"], tmp_path / "prep", "0" * 64, "cuda:0", tmp_path / "e")


def test_prepare_refuses_to_overwrite_an_existing_output(tmp_path):
    fx = _build_root(tmp_path / "root")
    out = tmp_path / "prep"
    _prepare(fx, out)
    with pytest.raises(SystemExit):
        _prepare(fx, out)


# ------------------------------------------------------------ hashes and heads


def test_historical_and_current_digests_are_recorded_and_not_relabelled(tmp_path):
    fx = _build_root(tmp_path / "root")
    preparation = _prepare(fx, tmp_path / "prep")
    source = preparation["source_version"]
    assert source["historical_rows_sha256_pinned"] == HISTORICAL
    assert source["historical_rows_recovered"] is False
    assert source["current_rows_sha256_observed"] != source["historical_rows_sha256_pinned"]
    assert source["current_equals_historical"] is False

    real = rq.load_protocol(
        REPO_ROOT / "data/r6_licensed_prediction_pilot_20261013/evaluator-requalification-protocol.json"
    )
    assert real["source_version"]["historical_rows_sha256"] != real["source_version"]["current_rows_sha256"]
    assert len(real["preparation"]["reviewed_targets"]) == 12


def test_feature_order_is_composition_then_embedding(tmp_path):
    vectors = np.zeros((1, 3), dtype=np.float16)
    vectors[0, 0] = 2.0
    block = rq.feature_block(["ACDE"], vectors)
    assert block.shape == (1, len(ds.BASELINE_FEATURES) + 3)
    assert np.allclose(block[0, : len(ds.BASELINE_FEATURES)], ds.composition_features("ACDE"))
    assert block[0, len(ds.BASELINE_FEATURES)] == 2.0
    assert ds.BASELINE_FEATURES[-2:] == ("length", "log_length")

    width = len(ds.BASELINE_FEATURES) + 3
    model = {
        "coefficients": np.concatenate([np.zeros(len(ds.BASELINE_FEATURES)), [1.0, 0.0, 0.0]]),
        "mean": np.zeros(width),
        "scale": np.ones(width),
        "intercept": 0.0,
    }
    assert ds.predict_with(model, block)[0] == pytest.approx(2.0)


def test_frozen_head_bytes_are_unchanged_by_evaluation(tmp_path):
    fx = _build_root(tmp_path / "root")
    _prepare_and_evaluate(fx, tmp_path)
    assert sha256_file(fx["paths"]["plm_head"]) == fx["plm_head_sha256"]


def test_qualify_reuses_the_existing_gate_helpers():
    generator = np.random.default_rng(7)
    truth, good, noise, baseline, groups = [], [], [], [], []
    for cluster in range(40):
        offset = generator.normal(0.0, 2.0)
        for _ in range(10):
            value = offset + generator.normal(0.0, 1.0)
            groups.append(f"g{cluster}")
            truth.append(value)
            good.append(value + generator.normal(0.0, 0.3))
            noise.append(generator.normal(0.0, 1.0))
            baseline.append(generator.normal(0.0, 1.0))
    truth, good, noise, baseline = map(np.asarray, (truth, good, noise, baseline))
    _, gate = rq.qualify(truth, good, baseline, groups, seed=1, draws=500)
    assert gate["passed"] is True
    _, failed = rq.qualify(truth, noise, baseline, groups, seed=1, draws=500)
    assert failed["passed"] is False
    assert "G1 failed" in failed["reason"]


def test_an_unresolvable_group_bootstrap_is_a_failure_not_a_pass():
    truth = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    predictions = np.asarray([0.2, 0.1, 0.4, 0.3, 0.6, 0.5])
    _, gate = rq.qualify(
        truth, predictions, predictions[::-1], ["a", "a", "b", "b", "c", "c"], seed=1, draws=200
    )
    assert gate["passed"] is False
    assert "not resolvable" in gate["reason"]


# ------------------------------------------------------------------ the run cell


def test_run_seals_the_preparation_digest_and_writes_one_pointer(tmp_path):
    fx = _build_root(tmp_path / "root")
    record = rq.run_requalification(fx["protocol"], fx["root"], "cpu", tmp_path / "run")
    prepared = tmp_path / "run" / "preparation" / rq.PREPARATION_COMPLETION
    qualification = tmp_path / "run" / "qualification" / rq.EVALUATION_COMPLETION
    assert record["status"] == "complete"
    assert record["preparation_sha256"] == sha256_file(prepared)
    assert record["qualification_sha256"] == sha256_file(qualification)
    assert record["no_outcomes_read_by_run"] is True
    assert isinstance(record["gates_passed"], bool)
    # The pointer must not carry a copy of the qualification's gate numbers.
    assert "contrast" not in record and "gates" not in record
    assert json.loads(qualification.read_text())["gates_passed"] == record["gates_passed"]


def test_run_stops_when_prepare_refuses(tmp_path):
    def drop_one(backgrounds):
        return [b for b in backgrounds if b["name"] != "1UFM.pdb"]

    fx = _build_root(tmp_path / "root", mutate_cohort=drop_one)
    with pytest.raises(rq.Refusal, match="1UFM.pdb"):
        rq.run_requalification(fx["protocol"], fx["root"], "cpu", tmp_path / "run")
    assert not (tmp_path / "run" / "qualification" / rq.EVALUATION_COMPLETION).exists()


# ------------------------------------------------------------- real inputs


def _real_inputs_present() -> bool:
    embed_dir = REPO_ROOT / rq.EMBED_DIR_REL
    return (embed_dir / "embeddings.npy").is_file() and (embed_dir / "embedding_index.json").is_file()


@pytest.mark.skipif(
    not _real_inputs_present(),
    reason="the full frozen embedding array/index are absent locally; run on the H200 parent",
)
def test_real_inputs_metadata_only_prepare(tmp_path):
    """Optional real-input preparation. Reads no measured outcome and no model."""

    protocol = (
        REPO_ROOT / "data/r6_licensed_prediction_pilot_20261013/evaluator-requalification-protocol.json"
    )
    record = rq.prepare(protocol, REPO_ROOT, tmp_path / "real-prep")
    assert record["outcomes_used"] is False
    assert record["no_measured_outcomes_read"] is True
    assert record["eligible"]["n"] > 0