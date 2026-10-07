"""Small invariant and negative-path tests for descriptive frozen-panel analysis."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.capability.extensions.phenotype_consistency import (
    METRICS, SEEDS, analyse, correlations, release_means, validate_cells,
    frozen_category_labels, release_families, within_release_categories,
)

ROOT = Path(__file__).resolve().parents[2]


def test_cell_validation_rejects_missing_duplicates_and_nonfinite():
    frame = pd.DataFrame([{"arm": "a", "split_seed": s, "estimate": 1., "ci_low": 0., "ci_high": 2.} for s in SEEDS])
    validate_cells(frame, ["a"], True)
    for invalid in (frame.iloc[:2], pd.concat([frame, frame.iloc[:1]]), frame.assign(arm="wrong"), frame.assign(estimate=np.nan)):
        with pytest.raises(ValueError):
            validate_cells(invalid, ["a"], True)


def test_correlations_preserve_ties_and_no_inference():
    frame = pd.DataFrame({m: [1., 1., 3., 4.] for m in METRICS})
    records = correlations(frame, "test")
    assert len(records) == 6
    assert all(np.isclose(r["spearman"], 1) and np.isclose(r["pearson"], 1) for r in records)
    assert all("pvalue" not in r and "ci" not in r for r in records)
    assert correlations(frame.assign(proteingym_rank=0), "constant")[0]["spearman"] is None
    with pytest.raises(ValueError):
        correlations(frame.assign(proteingym_rank=np.nan), "missing")


def test_release_means_weight_checkpoints_not_counts_of_other_releases():
    joined = pd.DataFrame({"arm": ["a", "b", "c"], **{m: [0., 4., 10.] for m in METRICS}})
    releases = release_means(joined, {"one": ("no", ("a", "b")), "two": ("yes", ("c",))})
    assert releases.proteingym_rank.tolist() == [2., 10.]


def test_real_frozen_panel():
    joined, releases, result = analyse(ROOT)
    assert len(joined) == 33 and len(releases) == 16
    assert np.isfinite(joined[list(METRICS)].to_numpy(dtype=float)).all()
    assert len(result["pairwise"]) == 12
    assert len(result["leave_one_release_out"]) == 96
    assert len(result["positives"]["proteingym_rank"]) == 14
    assert result["positives"]["proteingym_rank"] == result["positives"]["abundance_rank"]
    assert result["biological_bootstrap"]["status"] == "not_computed"
    declaration = ROOT / "scripts/capability/reporting/audit_followup_support.py"
    families = release_families(declaration)
    labels = frozen_category_labels(declaration)
    within = within_release_categories(releases, families, labels)
    assert len(within) == 12 and all(r["n"] == 8 for r in within)
    assert set().union(*(set(r["release_families"]) for r in within)) == set(families)
    assert {r["category_label"] for r in within} == set(labels.values())
    tied = releases.copy()
    for metric in METRICS:
        tied[metric] = [1., 1., 3., 4.] * 4
    assert all(np.isclose(r["spearman"], 1) for r in within_release_categories(tied, families, labels))
    assert all(r["spearman"] is None for r in within_release_categories(tied.assign(proteingym_rank=0), families, labels) if r["x"] == "proteingym_rank")
    for invalid in (releases.iloc[:-1], pd.concat([releases, releases.iloc[:1]]), releases.assign(release_category="yes")):
        with pytest.raises(ValueError):
            within_release_categories(invalid, families, labels)
