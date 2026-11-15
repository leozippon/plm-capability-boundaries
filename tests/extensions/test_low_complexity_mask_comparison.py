"""The comparison of the two low-complexity statistics, on sequences small enough to count by hand.

Both statistics are the repository's own functions, so what has to hold here is the
producer's arithmetic around them: that the counts it recovers are the counts the
functions imply, that a sequence at exactly the limit is flagged by integer
arithmetic and not moved by a float comparison, that the kappa and the matched-
prevalence cut are what a hand calculation gives, and that a quantile retention
inside a tie group is reported as such. The refusals are tested too: an empty
sequence has neither statistic, and a retained input whose digest is not the pinned
one stops the producer before any number is read.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PRODUCER = REPO_ROOT / "scripts/capability/extensions/compare_low_complexity_masks.py"
spec = importlib.util.spec_from_file_location("compare_low_complexity_masks", PRODUCER)
assert spec is not None and spec.loader is not None
masks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(masks)

# Twelve alanines, then fifty-six residues from nineteen letters read cyclically, so that
# every later window holds twelve distinct residues. The windows starting at 0..5 hold at
# least seven alanines and are at most 1.95 bits; the window starting at 6 holds six and is
# 2.29 bits. The labelled union is residues 0..16, seventeen of sixty-eight: exactly 0.25.
CYCLE = "CDEFGHIKLMNPQRSTVWY"
AT_THE_LIMIT = "A" * 12 + (CYCLE * 3)[:56]


def frame(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_a_homopolymer_has_every_residue_labelled_and_one_distinct_5mer():
    row = masks.measure("A" * 12)
    assert (row["labelled_residues"], row["kmer_positions"], row["distinct_5mers"]) == (12, 8, 1)
    assert row["s1"] == 1.0 and row["d5"] == 1 / 8


def test_the_boundary_sequence_sits_at_exactly_the_limit_and_is_flagged():
    row = masks.measure(AT_THE_LIMIT)
    assert (row["labelled_residues"], row["length"]) == (17, 68)
    assert row["s1"] == 0.25
    derived = masks.derive(pd.DataFrame([row]))
    assert bool(derived["flag_s1"].iloc[0]) and bool(derived["s1_exactly_limit"].iloc[0])


def test_below_five_residues_d5_is_zero_by_convention_so_the_repeat_share_is_one():
    row = masks.measure("ACD")
    derived = masks.derive(pd.DataFrame([row]))
    assert row["d5"] == 0.0 and row["kmer_positions"] == 0
    assert float(derived["s2"].iloc[0]) == 1.0 and bool(derived["flag_s2"].iloc[0])


def test_the_confusion_and_kappa_match_a_hand_count():
    a = [1, 1, 0, 0, 1, 0, 0, 0]
    b = [1, 0, 0, 0, 0, 0, 0, 1]
    table = masks.confusion(frame(flag_s1=np.array(a, bool), flag_s2=np.array(b, bool)))
    assert (table["both"], table["s1_only"], table["s2_only"], table["neither"]) == (1, 2, 1, 4)
    assert table["agreement"] == pytest.approx(5 / 8)
    # agreement 0.625, chance 3/8 * 2/8 + 5/8 * 6/8 = 0.5625, kappa = 0.0625 / 0.4375
    assert table["kappa"] == pytest.approx(1 / 7)


def test_matching_prevalence_raises_the_s1_cut_until_the_two_flag_the_same_share():
    s1 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    flag_s2 = np.array([False] * 8 + [True, True])           # two of ten: 0.2
    out = masks.matched_prevalence(frame(s1=s1, flag_s2=flag_s2))
    assert out["s1_cut"] == pytest.approx(0.8)
    assert out["prevalence_s1_at_cut"] == pytest.approx(0.2)
    assert out["both"] == 2 and out["s1_only"] == 0 and out["s2_only"] == 0


def test_a_tie_group_larger_than_the_retained_quarter_is_reported_as_swallowing_it():
    d5 = [1.0] * 6 + [0.9, 0.8]
    pool = pd.DataFrame({"source": "OT_folded", "arm": "protgpt3-112m", "d5": d5,
                         "sequence_sha256": [f"{i:02d}" for i in range(8)],
                         "no_repeated_5mer": [x == 1.0 for x in d5],
                         "flag_s1": [True, False] * 4})
    arm = masks.top_quarter_reading(pool)["arms"]["protgpt3-112m"]
    assert arm["n_retained"] == 2 and arm["tie_group_exceeds_retained"]
    assert arm["minimum_retained_d5"] == 1.0
    assert arm["no_repeated_5mer_share"] == pytest.approx(6 / 8)


def test_an_empty_sequence_is_refused():
    with pytest.raises(SystemExit, match="empty sequence"):
        masks.measure("")


def test_a_pinned_input_with_the_wrong_digest_stops_the_producer(tmp_path, monkeypatch):
    stray = tmp_path / "evaluation_cohort.jsonl"
    stray.write_text("not the retained cohort\n", encoding="utf-8")
    monkeypatch.setattr(masks, "PINNED", {"evaluation_cohort": (stray.name, "0" * 64)})
    monkeypatch.setattr(sys, "argv", ["compare_low_complexity_masks", "--repo", str(tmp_path),
                                      "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="is not the retained"):
        masks.main()
    assert not (tmp_path / "out").exists()
