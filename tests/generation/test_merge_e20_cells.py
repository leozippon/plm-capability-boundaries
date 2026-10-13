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


def _cell(root, name, seed, rows, arm_code="N", fields=0.0, domain="D0", protocol="p"):
    cell = root / name
    cell.mkdir(parents=True)
    body = "".join(json.dumps(r) + "\n" for r in rows)
    (cell / "sequences.jsonl").write_text(body)
    (cell / "guided_decoding_pool.json").write_text(json.dumps({
        "status": "complete", "protocol": {"sha256": protocol},
        "artifacts": {"sequences": {"file": "sequences.jsonl", "sha256": sha256(body.encode()).hexdigest()}},
        "cell": {"model_arm": "x", "domain": domain, "arm_code": arm_code, "lambda_fields": fields,
                 "lambda_coupling": 0.0, "stream_seed": seed}}))
    return cell


def test_records_join_and_refusals(tmp_path):
    a = _cell(tmp_path, "a", 1, [_row("a1")])
    b = _cell(tmp_path, "b", 2, [_row("b1", arm="M", lam=0.5)], arm_code="M", fields=0.5)
    summary = m.merge_records([a, b], tmp_path / "out", WILD, {f"D{i}": "development" for i in range(12)}, "development")
    assert summary == {"cells": 2, "records": 2}
    with pytest.raises(ValueError, match="stream seed"):
        m.merge_records([a, _cell(tmp_path, "c", 1, [_row("c1")], arm_code="M", fields=0.5)], tmp_path / "o2", WILD, None, None)
    with pytest.raises(ValueError, match="protocol digests"):
        m.merge_records([a, _cell(tmp_path, "d", 3, [_row("d1")], protocol="q", fields=0.25)], tmp_path / "o3", WILD, None, None)
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
