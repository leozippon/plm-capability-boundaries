"""Joining E20 cell outputs: what must hold and what is refused."""

from __future__ import annotations

import json
from hashlib import sha256

import pytest

from scripts.capability.generation import merge_e20_cells as m
from src.capability.generation import structure_evidence as se

WILD = {f"D{i}": "ACDEFGHIKL" for i in range(12)}


def _row(i, domain="D0", arm="N", lam=0.0):
    return {"schema_version": "e20_generated_sequence_v1", "id": i, "domain": domain, "model": "x", "arm": arm,
            "lambda": lam, "seed": 1, "sequence": "ACDEFGHIKM", "native_loglik": -1.0,
            "guidance_score": None if arm == "N" else 0.5, "mean_step_entropy": 1.0, "attempt_budget": 1}


def _cell(root, name, seed, rows, arm_code="N", fields=0.0, domain="D0", protocol="p", batch=2):
    cell = root / name
    cell.mkdir(parents=True)
    body = "".join(json.dumps(r) + "\n" for r in rows)
    (cell / "sequences.jsonl").write_text(body)
    (cell / "guided_decoding_pool.json").write_text(json.dumps({
        "status": "complete", "sampling": {"batch_size": batch}, "native_contract": {"budget": 1024}, "component_scale": {"pool": {"h": 1.0}}, "protocol": {"sha256": protocol},
        "artifacts": {"sequences": {"file": "sequences.jsonl", "sha256": sha256(body.encode()).hexdigest()}},
        "cell": {"model_arm": "x", "domain": domain, "arm_code": arm_code, "lambda_fields": fields,
                 "lambda_coupling": 0.0, "stream_seed": seed, "sequences_per_cell": 1, "attempt_budget": 1}}))
    return cell


def test_records_join_and_refusals(tmp_path):
    a = _cell(tmp_path, "a", 1, [_row("a1")])
    b = _cell(tmp_path, "b", 2, [_row("b1", arm="M", lam=0.5)], arm_code="M", fields=0.5)
    summary = m.merge_records([a, b], tmp_path / "out", WILD, {f"D{i}": "development" for i in range(12)}, "development")
    assert summary == {"cells": 2, "records": 2}
    with pytest.raises(ValueError, match="stream seed"):
        m.merge_records([a, _cell(tmp_path, "c", 1, [_row("c1")], arm_code="M", fields=0.5)], tmp_path / "o2", WILD, None, None)
    # a second protocol (an extension run) is accepted and both digests are recorded; a different recipe is not
    e = _cell(tmp_path, "e", 4, [_row("e1", arm="M", lam=0.25)], arm_code="M", fields=0.25, protocol="q")
    m.merge_records([a, e], tmp_path / "o3", WILD, None, None)
    assert json.loads((tmp_path / "o3" / m.RECORDS_COMPLETION).read_text())["protocol_sha256"] == ["p", "q"]
    with pytest.raises(ValueError, match="different sampling recipes"):
        m.merge_records([a, _cell(tmp_path, "d", 3, [_row("d1")], fields=0.25, batch=9)], tmp_path / "o6", WILD, None, None)
    with pytest.raises(ValueError, match="expected 3"):
        m.merge_records([tmp_path], tmp_path / "o7", WILD, None, None, expect_cells=3)
    with pytest.raises(ValueError, match="not a evaluation domain"):
        m.merge_records([a], tmp_path / "o4", WILD, {"D0": "development"}, "evaluation")
    (a / "sequences.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="differs"):
        m.merge_records([a], tmp_path / "o5", WILD, None, None)


def _shard(root, name, index, shas):
    shard = root / name
    contract = {"k": 1}
    (shard).mkdir(parents=True)
    (shard / f"worker-{index:03d}-of-002.json").write_text(json.dumps(
        {"contract": contract, "evaluation_signature": se.contract_signature(contract)}))
    for sha in shas:
        (shard / "objects" / sha).mkdir(parents=True)
        (shard / "objects" / sha / "result.json").write_text(json.dumps({"status": "ok"}))
    (shard / "structure_evidence.json").write_text(json.dumps({"status_counts": {"ok": len(shas)}}))
    return shard


def test_shards_join_and_refuse_overlap(tmp_path):
    s0, s1 = _shard(tmp_path, "s0", 0, ["aa"]), _shard(tmp_path, "s1", 1, ["bb", "cc"])
    assert m.merge_shards([s0, s1], tmp_path / "t")["objects"] == 3
    assert se.tree_signature(tmp_path / "t") == se.contract_signature({"k": 1})
    with pytest.raises(ValueError, match="two shards"):
        m.merge_shards([s0, _shard(tmp_path, "s2", 1, ["aa"])], tmp_path / "t2")


def test_parent_directory_discovery_and_strength_pair(tmp_path):
    parent = tmp_path / "run"
    _cell(parent, "a", 1, [_row("a1")])
    _cell(parent, "b", 2, [_row("b1", arm="M", lam=0.5)], arm_code="M", fields=0.5)
    _cell(parent, "c", 3, [_row("c1", arm="M", lam=0.25)], arm_code="M", fields=0.25)
    (parent / "e20dev_cohort_cpu").mkdir()      # not cells: no completion record
    (parent / "e20dev_fold_shard0_gpu0" / "objects").mkdir(parents=True)
    (parent / "e20_merged_records.json").write_text("{}")
    assert [c.name for c in m.discover_cells([parent])] == ["a", "b", "c"]
    summary = m.merge_records([parent], tmp_path / "out", WILD, None, None)
    assert summary == {"cells": 3, "records": 3}   # two strengths of arm M validate together
    rows = [json.loads(x) for x in (tmp_path / "out" / "records.jsonl").read_text().splitlines()]
    assert {(r["arm"], r["lambda_fields"], r["lambda_coupling"]) for r in rows} == {("N", 0.0, 0.0), ("M", 0.5, 0.0), ("M", 0.25, 0.0)}
    assert len(json.loads((tmp_path / "out" / "component_scale.json").read_text())) == 3
    # the explicit-list form still works, and evaluation refuses several strengths of one arm
    assert m.merge_records([parent / "a", parent / "b"], tmp_path / "o2", WILD, None, None)["cells"] == 2
    pair = lambda f, n=1: {"lambda_fields": f, "lambda_coupling": 0.0, "pool_size": n}
    declared = [{"model": "x", "arm": "N", "frozen_strength": pair(0.0)},
                {"model": "x", "arm": "M", "frozen_strength": pair(0.5)}]
    with pytest.raises(ValueError, match="no declared pool_size"):  # 0.25 was not pre-declared
        m.merge_records([parent], tmp_path / "o3", WILD, {"D0": "evaluation"}, "evaluation", strengths=declared)
    declared[1]["dose_response_strengths"] = [pair(0.25)]
    assert m.merge_records([parent], tmp_path / "o4", WILD, {"D0": "evaluation"}, "evaluation", strengths=declared)["cells"] == 3
    # depth is each cell's own declaration: 2 for the dose arm, 1 elsewhere is fine, a mismatch is refused
    declared[1]["dose_response_strengths"] = [pair(0.25, 2)]
    with pytest.raises(ValueError, match="declaration names a depth of 2"):
        m.merge_records([parent], tmp_path / "o5", WILD, {"D0": "evaluation"}, "evaluation", strengths=declared)
    deep = _cell(tmp_path / "deep", "c", 9, [_row("c1", arm="M", lam=0.25), _row("c2", arm="M", lam=0.25)],
                 arm_code="M", fields=0.25)
    (deep / "guided_decoding_pool.json").write_text(
        (deep / "guided_decoding_pool.json").read_text().replace('"sequences_per_cell": 1', '"sequences_per_cell": 2'))
    assert m.merge_records([parent / "a", parent / "b", deep], tmp_path / "o6", WILD, {"D0": "evaluation"},
                           "evaluation", strengths=declared)["records"] == 4
    del declared[1]["dose_response_strengths"][0]["pool_size"]
    with pytest.raises(ValueError, match="no declared pool_size"):
        m.merge_records([parent / "a", parent / "b", deep], tmp_path / "o7", WILD, {"D0": "evaluation"},
                        "evaluation", strengths=declared)
    # without a declaration (development) one shared depth is still required
    with pytest.raises(ValueError, match="different sampling recipes"):
        m.merge_records([parent / "a", deep], tmp_path / "o8", WILD, None, None)
    with pytest.raises(ValueError, match="no cell"):
        m.discover_cells([tmp_path / "out"])
