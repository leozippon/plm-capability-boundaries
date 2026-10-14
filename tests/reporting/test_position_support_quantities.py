"""Primary position support extrema retain verified fit-leaf provenance."""
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.capability.reporting import followup_quantities as followup
from src.capability.core.evidence_ledger import Artifacts, Ledger, LedgerError, digest, resolve

PRIMARY = 20260923


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture
def sources(tmp_path):
    payload = {"arms": ["zeta", "beta", "alpha"], "fold_seeds": [PRIMARY, PRIMARY + 1],
               "source_sha256": {}}
    paths = {}
    for arm, counts in (("zeta", (90, 8, 5)), ("beta", (12, 7, 5)), ("alpha", (12, 7, 5))):
        for seed in payload["fold_seeds"]:
            path = f"sources/{arm}_fold{seed}.json"
            fit = {"arm": arm, "fold_seed": seed, "support_rule": "label-blind aligned substitutions",
                   **dict(zip(("variants", "assays", "families"), counts, strict=True))}
            if seed != PRIMARY:
                fit.update(variants=999, assays=999, families=999)
            write_json(tmp_path / path, fit)
            payload["source_sha256"][path] = digest(tmp_path / path)
            paths[arm, seed] = path
    return tmp_path, payload, paths


def run_helper(sources):
    root, payload, _ = sources
    write_json(root / followup.POSITION / "position_simultaneous.json", payload)
    ledger = Ledger(Artifacts(root))
    followup._position_support_counts(ledger, followup._emitter(ledger))
    return ledger


def test_six_extrema_bind_verified_primary_fit_leaves(sources):
    ledger = run_helper(sources)
    root, _, paths = sources
    expected = {"variants": (12, 90), "assays": (7, 8), "families": (5, 5)}
    units = {"variants": "variants", "assays": "assays", "families": "wild-type families"}
    assert len(ledger.quantities) == len({q.id for q in ledger.quantities}) == 6
    for quantity in ledger.quantities:
        field, bound = quantity.id.split("/")[-2:]
        assert quantity.id == f"position_terms/primary_support/{field}/{bound}"
        assert quantity.value == expected[field][bound == "max"]
        # Ties choose the lexicographically first arm, even for the maximum.
        arm = "zeta" if bound == "max" and field != "families" else "alpha"
        assert quantity.source_path == paths[arm, PRIMARY]
        assert quantity.source_sha256 == digest(root / quantity.source_path)
        assert quantity.source_pointer == (field,)
        assert resolve(ledger.artifacts.json(quantity.source_path), quantity.source_pointer) == quantity.value
        assert quantity.seed_set == (PRIMARY,)
        assert quantity.unit == units[field]
        assert quantity.kind == "support_count"
        assert quantity.level == "likelihood"
        assert quantity.interval is None and quantity.resampling_draws is None
    description = ledger.supports[ledger.quantities[0].support_id].description
    assert "range across checkpoint-specific aligned supports; not a common cohort" in description
    assert "3 declared arms" in description and str(PRIMARY) in description
    assert "label-blind aligned substitutions" in description


def test_reject_digest_mismatch(sources):
    _, payload, paths = sources
    payload["source_sha256"][paths["alpha", PRIMARY]] = "0" * 64
    with pytest.raises(LedgerError, match="digest mismatch"):
        run_helper(sources)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "extra", "duplicate declaration"])
def test_reject_inexact_declared_roster(sources, failure):
    root, payload, paths = sources
    if failure == "missing":
        del payload["source_sha256"][paths["alpha", PRIMARY]]
    elif failure == "duplicate declaration":
        payload["arms"].append("alpha")
    else:
        path = f"sources/additional_fold{PRIMARY}.json"
        fit = json.loads((root / paths["alpha", PRIMARY]).read_text())
        if failure == "extra":
            fit["arm"] = "undeclared"
        write_json(root / path, fit)
        payload["source_sha256"][path] = digest(root / path)
    with pytest.raises(LedgerError, match="roster|unexpected or duplicate"):
        run_helper(sources)


def test_primary_filename_cannot_override_payload_seed(sources):
    root, payload, paths = sources
    path = paths["alpha", PRIMARY]
    fit = json.loads((root / path).read_text())
    fit["fold_seed"] = PRIMARY + 1
    write_json(root / path, fit)
    payload["source_sha256"][path] = digest(root / path)
    with pytest.raises(LedgerError, match="primary seed mismatch"):
        run_helper(sources)
