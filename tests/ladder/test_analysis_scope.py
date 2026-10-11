"""Producer file declarations, not incidental sequence overlap, define ladders."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from src.capability.core.io import sha256_file
from src.capability.ladder import design


@pytest.fixture
def scope_inputs(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/capability/ladder/analyse_ladder_divergence.py"
    spec = importlib.util.spec_from_file_location("ladder_scope_analysis", script)
    assert spec is not None and spec.loader is not None
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    backbones = tmp_path / "backbones.jsonl"
    design.write_jsonl(backbones, [{"backbone_id": f"b{b}"} for b in range(8)])
    reference = "composition-shuffle"
    arms = ("protgpt2", "zymctrl", reference)
    paths, pools, structure = [], {}, []
    for arm_index, arm in enumerate(arms):
        rows = []
        for b in range(8):
            for extent in design.WINDOW_EXTENTS:
                for draw in range(4):
                    prefix = "A" if extent == 1 and arm_index < 2 else "CDE"[arm_index]
                    sequence = prefix + "A" * (b + 1) + "C" * extent + "D" * (draw + 1)
                    identifier = f"{arm}|b{b}|k{extent}|{draw}"
                    rows.append({
                        "id": identifier, "arm": arm, "condition": design.arm(arm).condition,
                        "backbone_id": f"b{b}", "rung": f"k{extent}", "draw": draw,
                        "status": "filled", "sequence": sequence,
                        "sequence_sha256": design.sequence_digest(sequence),
                        "parent_sequence_sha256": "parent", "composition_distance_to_parent": draw / 10,
                    })
                    structure.append({"id": identifier, "fold_status": "ok", "length": len(sequence),
                                      "tm_score_to_parent": [.2, .4, .8, .6][draw]})
        path = tmp_path / f"{arm}.variants.jsonl"
        design.write_jsonl(path, rows)
        paths.append(path)
        pools[arm] = rows
    likelihood_paths, receipt_paths = [], []
    for arm, own_path in zip(arms[:2], paths[:2]):
        jobs = {row["sequence_sha256"]: {
            "sequence_sha256": row["sequence_sha256"], "scored_by": arm,
            "mean_nll_per_token_nats": -row["draw"],
        } for row in pools[arm] + pools[reference]}
        records = tmp_path / f"{arm}.ladder_likelihood.jsonl"
        design.write_jsonl(records, list(jobs.values()))
        receipt = records.with_suffix(".json")
        receipt.write_text(json.dumps({
            "schema_version": "ladder_likelihood_v1", "arm": arm,
            "records": "ladder_likelihood.jsonl", "records_sha256": sha256_file(records),
            "n_sequences": len(jobs),
            "backbones": {"path": str(backbones), "sha256": sha256_file(backbones)},
            "variants": {str(path): sha256_file(path) for path in (own_path, paths[2])},
        }), encoding="utf-8")
        likelihood_paths.append(records)
        receipt_paths.append(receipt)
    structure_path = tmp_path / "structure.jsonl"
    design.write_jsonl(structure_path, structure)
    args = argparse.Namespace(out=tmp_path / "out", backbones=backbones, variants=paths,
                              likelihood=likelihood_paths, structure=[structure_path], seed=1, draws=40)
    return stage, args, receipt_paths, reference


def test_shared_sequence_does_not_expand_declared_scope(scope_inputs):
    stage, args, _, reference = scope_inputs
    payload = stage.run(args)
    expected = {f"{scorer}|{product}|{design.arm(product).condition}"
                for scorer in ("protgpt2", "zymctrl") for product in (scorer, reference)}
    assert set(payload["ladders"]) == expected
    assert len(payload["cells"]) == 24
    assert all(ladder["ladder_complete"] for ladder in payload["ladders"].values())
    assert len(payload["scientific_scope"]["omitted_crosspairs"]) == 2
    assert set(payload["scientific_scope"]["expected_ladders"]) == expected
    assert payload["schema_version"] == "ladder_divergence_v2"


def test_explicit_subset_keeps_only_selected_declared_ladders(scope_inputs):
    stage, args, _, reference = scope_inputs
    args.variants = [args.variants[2]]
    payload = stage.run(args)
    assert set(payload["ladders"]) == {
        f"{scorer}|{reference}|{design.arm(reference).condition}" for scorer in ("protgpt2", "zymctrl")
    }
    assert all(len(binding["unselected_declared_variant_sources"]) == 1
               for binding in payload["scientific_scope"]["likelihood_producers"].values())


def test_single_scorer_subset_does_not_require_global_arms(scope_inputs):
    stage, args, _, reference = scope_inputs
    args.likelihood = args.likelihood[:1]
    args.variants = [args.variants[0], args.variants[2]]
    payload = stage.run(args)
    assert set(payload["ladders"]) == {
        f"protgpt2|{product}|{design.arm(product).condition}" for product in ("protgpt2", reference)
    }


@pytest.mark.parametrize("mismatch", ["variants", "records", "scorer", "backbones", "schema", "missing"])
def test_contradictory_or_missing_producer_binding_is_refused(scope_inputs, mismatch):
    stage, args, receipts, _ = scope_inputs
    path = receipts[0]
    receipt = json.loads(path.read_text())
    if mismatch == "variants":
        receipt["variants"][str(args.variants[0])] = "0" * 64
    elif mismatch == "records":
        receipt["records_sha256"] = "0" * 64
    elif mismatch == "scorer":
        rows = design.read_jsonl(args.likelihood[0])
        rows[0]["scored_by"] = "zymctrl"
        receipt["records_sha256"] = design.write_jsonl(args.likelihood[0], rows)
    elif mismatch == "backbones":
        receipt["backbones"]["sha256"] = "0" * 64
    elif mismatch == "schema":
        receipt["schema_version"] = "unknown"
    if mismatch == "missing":
        path.unlink()
    else:
        path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(SystemExit):
        stage.run(args)


def test_relocated_sources_and_explicit_receipts_bind_by_hash(scope_inputs):
    stage, args, receipts, _ = scope_inputs
    for path in receipts:
        receipt = json.loads(path.read_text())
        receipt["variants"] = {"/remote/" + Path(name).name: digest for name, digest in receipt["variants"].items()}
        receipt["backbones"]["path"] = "/remote/backbones.jsonl"
        path.write_text(json.dumps(receipt), encoding="utf-8")
    relocated = []
    for index, path in enumerate(receipts):
        new = path.with_name(f"retained-{index}.json")
        path.rename(new)
        relocated.append(new)
    args.likelihood_receipts = relocated
    assert len(stage.run(args)["ladders"]) == 4


def test_declared_zero_join_ladder_stays_in_family(scope_inputs):
    stage, args, _, _ = scope_inputs
    structure = design.read_jsonl(args.structure[0])
    for row in structure:
        if row["id"].startswith("zymctrl|"):
            row["fold_status"] = "missing"
    design.write_jsonl(args.structure[0], structure)
    payload = stage.run(args)
    key = f"zymctrl|zymctrl|{design.CONDITION_CAUSAL_PREFIX}"
    missing = [f"k{extent}" for extent in design.WINDOW_EXTENTS]
    assert payload["ladders"][key]["missing_rungs"] == missing
    assert payload["ladders"][key]["rungs_present"] == []
    assert not payload["simultaneous_band"]["available"]
    assert set(payload["simultaneous_band"]["declared_family"]) == {
        f"{ladder}|{rung}" for ladder in payload["ladders"] for rung in missing
    }
