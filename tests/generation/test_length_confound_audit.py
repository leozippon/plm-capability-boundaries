"""The length audit: the sixteen-bin estimator, the checks that license it, and its refusals.

The estimator is the one the campaign retains for its length-matched readings, so what
has to hold here is that it computes what a hand calculation gives, that a comparison of
recomputed against retained values cannot pass on nothing or on a near miss, and that
the producers stop on an input that is not the retained one. The shape analysis is
checked on a corpus built to be U-shaped, where its minimum and its rise are known, and
on a monotone fall, where the same contrast must come out negative.

The producers live in ``scripts/capability/generation/length_confound_audit_20261114/``.
The reproduction of the retained values themselves is not a test: it is a check the
producer performs on every run and records in its result tree, and it fails the run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.generation import length_matching as lm  # noqa: E402

PRODUCERS = REPO_ROOT / "scripts/capability/generation/length_confound_audit_20261114"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PRODUCERS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matched = _load("length_matched_contrasts")
shape = _load("length_dependence_of_confidence")
register = _load("length_register_retained_comparisons")


# ------------------------------------------------------------------- estimator


def test_the_sixteen_bin_estimator_matches_a_hand_calculation():
    # bins of floor(L / 16): base {0: [0, 10], 1: [20], 2: [40]}, tuned {0: [5], 1: [25], 6: [99]}
    estimate, tuned_kept, base_kept = lm.bin_match([10, 12, 30, 33], [0, 10, 20, 40],
                                                   [11, 31, 100], [5, 25, 99])
    # bin 0: 5 - 5 = 0; bin 1: 25 - 20 = 5; bin 6 has no base row and is dropped.
    assert estimate == pytest.approx(2.5)
    assert (tuned_kept, base_kept) == (2, 3)


def test_each_bin_is_weighted_by_its_tuned_count_not_equally():
    # tuned bin 0 holds three rows at a difference of 20, tuned bin 2 one row at 10
    estimate, _, _ = lm.bin_match([5, 45], [10, 50], [1, 2, 3, 40], [30, 30, 30, 60])
    assert estimate == pytest.approx((3 * 20 + 1 * 10) / 4)       # 17.5; an equal-bin mean would be 15


def test_with_no_shared_bin_the_estimate_is_nan_and_is_never_imputed():
    estimate, tuned_kept, base_kept = lm.bin_match([100], [50.0], [10], [60.0])
    assert np.isnan(estimate) and (tuned_kept, base_kept) == (0, 0)


def test_the_length_summary_counts_sequences_below_fifty():
    summary = lm.length_summary([40, 49, 50, 100])
    assert summary["n"] == 4 and summary["min"] == 40 and summary["below_50"] == 2
    assert summary["median"] == pytest.approx(49.5)
    assert (summary["q1"], summary["q3"]) == (pytest.approx(46.75), pytest.approx(62.5))
    assert lm.length_summary([])["n"] == 0


def test_materiality_is_a_median_gap_or_a_distribution_gap_and_the_share_needs_a_real_gap():
    assert lm.is_material(25.0, 0.05) and lm.is_material(-20.0, 0.0) and lm.is_material(0.0, 0.2)
    assert not lm.is_material(19.9, 0.19)
    assert lm.attributable_share(10.0, 6.0) == pytest.approx(0.4)
    assert lm.attributable_share(-5.0, -7.0) == pytest.approx(-0.4)     # matching enlarged the gap
    assert lm.attributable_share(0.5, 0.2) is None                      # below one point
    assert lm.attributable_share(10.0, float("nan")) is None


# ------------------------------------------------------------ the check itself


def test_the_reproduction_check_cannot_pass_on_nothing_a_near_miss_or_the_wrong_count():
    good = [{"abs_difference": 3e-5}, {"abs_difference": 9e-5}]
    assert lm.check_values(good, 1e-4, expected=2)["passed"]
    assert not lm.check_values(good + [{"abs_difference": 2e-4}], 1e-4)["passed"]
    assert not lm.check_values([], 1e-4)["passed"]
    assert not lm.check_values(good, 1e-4, expected=3)["passed"]
    assert not lm.check_values([{"abs_difference": float("nan")}], 1e-4)["passed"]
    with pytest.raises(SystemExit, match="does not match the retained"):
        register.require({}, "toy", [{"abs_difference": 1.0}], 1e-6)


def test_a_perturbed_retained_value_fails_the_producers_reproduction_check():
    base = pd.DataFrame({"L": [10, 12, 30, 33], "plddt": [0.0, 10.0, 20.0, 40.0]})
    tuned = pd.DataFrame({"L": [11, 31, 100], "plddt": [5.0, 25.0, 99.0]})
    cells = lambda frame: iter([("length_stratum", "112m", "16-128", base, tuned)])  # noqa: E731
    retained = {("length_stratum", "112m", "16-128"): {"matched": {"16-bin": 2.5}, "table": "toy"}}
    rows = matched.reproduction_rows("toy", None, cells, retained)
    assert lm.check_values(rows, matched.ESTIMATOR_TOLERANCE, expected=1)["passed"]
    retained[("length_stratum", "112m", "16-128")]["matched"]["16-bin"] = 2.5 + 2e-4
    rows = matched.reproduction_rows("toy", None, cells, retained)
    assert not lm.check_values(rows, matched.ESTIMATOR_TOLERANCE, expected=1)["passed"]


def test_a_retained_cell_whose_count_differs_fails_the_set_definition_check():
    base = pd.DataFrame({"L": [10, 12], "plddt": [1.0, 3.0]})
    tuned = pd.DataFrame({"L": [11], "plddt": [4.0]})
    cells = lambda frame: iter([("identity", "112m", "x pooled", base, tuned)])  # noqa: E731
    retained = {("identity", "112m", "x pooled"): {"raw": 2.0, "n": (2, 2), "table": "toy", "matched": {}}}
    rows = matched.set_check_rows("objective_tuned", None, cells, retained)
    assert rows[0]["abs_difference"] == pytest.approx(0.0) and not rows[0]["counts_equal"]


def test_a_pinned_retained_table_with_the_wrong_digest_stops_the_producer(tmp_path, monkeypatch):
    stray = tmp_path / "fold.tsv"
    stray.write_text("not the retained table\n", encoding="utf-8")
    monkeypatch.setattr(matched, "PINNED", {"fold_table": (stray.name, "0" * 64)})
    monkeypatch.setattr(sys, "argv", ["length_matched_contrasts", "--repo", str(tmp_path),
                                      "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="is not the retained"):
        matched.main()
    assert not (tmp_path / "out").exists()


def test_a_substituted_fold_index_is_refused_against_the_digest_the_evaluation_record_carries(tmp_path):
    record = tmp_path / lm.EVALUATION_RECORD
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"structure_receipts": [
        {"directory": "/run/geneval_e14_fold_shard0_gpu2", "summary": {"index_sha256": "0" * 64}},
        {"directory": "/run/geneval_e14_fold_shard1_gpu3", "summary": {"index_sha256": "1" * 64}}]}),
        encoding="utf-8")
    for relative in lm.EVALUATION_FOLD_INDEXES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="the retained evaluation record carries"):
        lm.read_evaluation_folds(tmp_path)


# ------------------------------------------------------------------- the shape


def _corpus(values_of_length, n: int = 2000, noise: float = 6.0, seed: int = 7):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(30, 400, n)
    return lengths, values_of_length(lengths) + rng.normal(0.0, noise, n)


def test_on_a_u_shaped_corpus_the_minimum_is_found_in_the_trough_and_the_rise_is_positive():
    u = lambda length: 52.0 + 0.0008 * (np.minimum(length, 260) - 75.0) ** 2  # noqa: E731
    lengths, values = _corpus(u)
    out = shape.analyse_corpus("u", "synthetic", lengths, values, np.random.default_rng(1))
    assert out["minimum_bin"] in {"60-69", "70-79", "80-89"}
    assert out["rise_200_299_minus_trough"]["ci95"][0] > 10.0


def test_on_a_monotone_fall_the_same_contrast_is_negative():
    fall = lambda length: 70.0 - 0.05 * length  # noqa: E731
    lengths, values = _corpus(fall)
    out = shape.analyse_corpus("fall", "synthetic", lengths, values, np.random.default_rng(1))
    assert out["rise_200_299_minus_trough"]["ci95"][1] < 0.0
