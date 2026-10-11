"""Synthetic analysis inputs: completeness is distinct from computability/inference."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from src.capability.ladder import design


def _analyse(tmp_path, mode):
    path = Path(__file__).resolve().parents[2] / "scripts/capability/ladder/analyse_ladder_divergence.py"
    spec = importlib.util.spec_from_file_location("ladder_analysis_inference", path)
    assert spec is not None and spec.loader is not None
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    backbones, variants, likelihood, structure = [], [], [], []
    for cluster in range(8):
        backbone = f"b{cluster}"
        backbones.append({"backbone_id": backbone})
        for extent in design.WINDOW_EXTENTS:
            if mode == "missing_middle" and extent == 5:
                continue
            for draw in range(4):
                identifier = f"{backbone}|k{extent}|{draw}"
                variant = {
                    "id": identifier, "backbone_id": backbone, "draw": draw,
                    "status": "filled", "sequence": "ACDE", "sequence_sha256": identifier,
                    "parent_sequence_sha256": backbone, "arm": "protgpt2",
                    "condition": design.CONDITION_CAUSAL_PREFIX, "rung": f"k{extent}",
                    "composition_distance_to_parent": draw / 10,
                }
                if mode == "missing_composition":
                    del variant["composition_distance_to_parent"]
                elif mode == "null_composition":
                    variant["composition_distance_to_parent"] = None
                variants.append(variant)
                likelihood.append({"scored_by": "protgpt2", "sequence_sha256": identifier,
                                   "mean_nll_per_token_nats": (
                                       0 if mode == "one_informative" and cluster > 0 else -draw
                                   )})
                # Real within-backbone variation; exact covariate collinearity in x.
                structure.append({"id": identifier, "fold_status": "ok", "length": 4,
                                  "tm_score_to_parent": 1 if mode == "constant_primary" else
                                  [0.2, 0.4, 0.8, 0.6][draw]})
                if mode == "one_informative":
                    structure[-1]["tm_score_to_parent"] = (
                        1 if cluster > 0 else draw if extent <= 5 else 3 - draw
                    )
    if mode == "zero_joined_ladder":
        for row in list(variants):
            absent = {**row, "arm": "zymctrl", "id": "absent|" + row["id"],
                      "sequence_sha256": "absent|" + row["sequence_sha256"]}
            variants.append(absent)
            structure.append({"id": absent["id"], "fold_status": "missing"})
    paths = {}
    for name, rows in (("backbones", backbones), ("variants", variants),
                       ("likelihood", likelihood), ("structure", structure)):
        paths[name] = tmp_path / f"{name}.jsonl"
        design.write_jsonl(paths[name], rows)
    payload = stage.run(argparse.Namespace(
        out=tmp_path / "out", backbones=paths["backbones"],
        variants=[paths["variants"]], likelihood=[paths["likelihood"]],
        structure=[paths["structure"]], seed=1, draws=100 if mode == "one_informative" else 40,
    ))
    # No serializer-only repair is needed for these numerical failures.
    assert payload["non_finite_fields"] == []
    json.dumps(payload, allow_nan=False)
    return payload, next(iter(payload["ladders"].values()))


def test_absent_middle_rung_is_not_a_complete_ladder(tmp_path):
    payload, ladder = _analyse(tmp_path, "missing_middle")
    assert ladder["missing_rungs"] == ["k5"]
    assert not ladder["ladder_complete"] and not ladder["inferential_available"]
    assert "divergence" not in ladder
    assert not payload["simultaneous_band"]["available"]
    assert ladder["adjacent_rung_contrasts"][1]["reason"] == "missing adjacent rung"


def test_constant_primary_is_complete_but_inference_unavailable(tmp_path):
    payload, ladder = _analyse(tmp_path, "constant_primary")
    assert ladder["ladder_complete"] and not ladder["inferential_available"]
    assert ladder["divergence"]["loses_significance"] is None
    assert not ladder["divergence"]["simultaneous_available"]
    for cell in payload["cells"].values():
        raw = cell["correlations"][design.PRIMARY_STRUCTURE_READOUT]["raw"]
        assert raw["point"] is None and raw["point_reason"]


@pytest.mark.parametrize("mode", ["collinear_composition", "missing_composition", "null_composition"])
def test_undefined_adjustment_does_not_invalidate_raw_primary(tmp_path, mode):
    payload, ladder = _analyse(tmp_path, mode)
    assert ladder["ladder_complete"] and ladder["inferential_available"]
    for cell in payload["cells"].values():
        primary = cell["correlations"][design.PRIMARY_STRUCTURE_READOUT]
        assert primary["raw"]["point"] == pytest.approx(.8)
        assert primary["raw"]["interval"] is not None
        adjusted = primary["composition_adjusted"]
        assert adjusted["point"] is None and adjusted["interval_reason"]
        if mode == "collinear_composition":
            assert "collinear" in adjusted["diagnostics"]["reason"]
        else:
            assert "composition" in adjusted["point_reason"]


def test_a_zero_joined_supplied_ladder_remains_in_the_declared_family(tmp_path):
    payload, _ = _analyse(tmp_path, "zero_joined_ladder")
    condition = design.CONDITION_CAUSAL_PREFIX
    joined_key = f"protgpt2|protgpt2|{condition}"
    absent_key = f"protgpt2|zymctrl|{condition}"
    assert set(payload["ladders"]) == {joined_key, absent_key}
    assert payload["ladders"][joined_key]["ladder_complete"]
    absent = payload["ladders"][absent_key]
    declared_rungs = [f"k{extent}" for extent in design.WINDOW_EXTENTS]
    assert absent["status"] == "incomplete"
    assert absent["rungs_present"] == [] and absent["missing_rungs"] == declared_rungs
    assert not absent["inferential_available"]
    band = payload["simultaneous_band"]
    assert not band["available"]
    assert set(band["declared_family"]) == {
        f"{key}|{rung}" for key in (joined_key, absent_key) for rung in declared_rungs
    }
    assert all(f"{absent_key}|{rung}" in band["excluded_cells"] for rung in declared_rungs)


def test_unsupported_primary_floor_cannot_issue_a_crossing_interval(tmp_path):
    payload, ladder = _analyse(tmp_path, "one_informative")
    assert ladder["ladder_complete"] and not ladder["inferential_available"]
    assert [row["point"] for row in ladder["ladder"]] == [1, 1, 1, -1, -1, -1]
    for cell in payload["cells"].values():
        raw = cell["correlations"][design.PRIMARY_STRUCTURE_READOUT]["raw"]
        assert raw["n_informative_units"] == 1 and raw["interval"] is None
        assert "informative independent backbones" in raw["interval_reason"]
    crossing = ladder["continuous_crossing"]
    assert crossing["crossing_extent"] == pytest.approx(6.564197879454706)
    assert crossing["n_draws_with_crossing_in_range"] >= 2
    assert crossing["fraction_without_crossing"] > .05
    assert crossing["interval"] is None and not crossing["inferential_available"]
    assert crossing["interval_reason"] == "required primary inference unavailable"
    assert "conditional" in crossing["interval_interpretation"]
