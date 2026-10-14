"""Framework metadata comes from verified pinned maps, without fitting."""
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.capability.reporting import followup_quantities as followup
from src.capability.core.evidence_ledger import Artifacts, Ledger, LedgerError, digest, resolve

MANIFEST = "manuscript/assets/selection.json"


@pytest.fixture
def sources(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("retained biological source")
    # Deliberately unsorted domain positions: numeric ordering, not list ordering.
    residue_map = [{"wt_position": position, "coordinate_id": coordinate,
                    "author_id": str(coordinate + 100)}
                   for position, coordinate in ((10, 19), (2, 11), (1, 10))]
    payload = {
        "schema": "framework-pinned-structures-v1",
        "inputs": {"source": {"path": "source.txt", "sha256": digest(source)}},
        "structures": {name: {"residue_map": residue_map, "model": 2, "models": 7}
                       for name in ("ubc9", "dvl1", "pci")},
        "records": {"dvl1": {"selected": {"positions": [2]}}},
    }
    return tmp_path, payload


def run_helper(sources):
    root, payload = sources
    path = root / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    ledger = Ledger(Artifacts(root))
    followup._framework_mapping_quantities(ledger, followup._emitter(ledger))
    return ledger


def test_counts_and_literal_metadata_pointers(sources):
    ledger = run_helper(sources)
    quantities = {quantity.id: quantity for quantity in ledger.quantities}
    assert len(quantities) == len(ledger.quantities) == 15
    for quantity in quantities.values():
        assert quantity.source_path == MANIFEST
        assert quantity.source_sha256 == digest(sources[0] / MANIFEST)
        assert quantity.kind == "support_count" and quantity.level == "measurement"
        assert quantity.interval is None and quantity.resampling_draws is None
        leaf = resolve(ledger.artifacts.json(MANIFEST), quantity.source_pointer)
        if quantity.id.endswith("/mapped_residue_count"):
            assert quantity.value == len(leaf) == 3
            assert quantity.unit == "residues"
            assert quantity.source_pointer[-1] == "residue_map"
            assert "len(residue_map)" in quantity.claim
        else:
            assert quantity.value == float(leaf)
    first = quantities["framework_mapping/ubc9/coordinate_id_first"]
    assert first.value == 10
    assert first.source_pointer == ("structures", "ubc9", "residue_map", "[2]", "coordinate_id")
    assert quantities["framework_mapping/ubc9/coordinate_id_end"].value == 19
    assert quantities["framework_mapping/dvl1/selected_domain_position"].source_pointer == (
        "records", "dvl1", "selected", "positions", "[0]")
    assert quantities["framework_mapping/dvl1/selected_coordinate_id"].value == 11
    assert quantities["framework_mapping/pci/author_id_first"].value == 110
    assert quantities["framework_mapping/pci/author_id_end"].value == 119
    assert quantities["framework_mapping/pci/model"].value == 2
    assert quantities["framework_mapping/pci/models"].value == 7


def test_refuses_source_hash_mismatch(sources):
    root, _ = sources
    (root / "source.txt").write_text("changed source")
    with pytest.raises(LedgerError, match="source digest mismatch"):
        run_helper(sources)


def test_refuses_unmapped_selected_site(sources):
    _, payload = sources
    payload["records"]["dvl1"]["selected"]["positions"] = [3]
    with pytest.raises(LedgerError, match="not uniquely mapped"):
        run_helper(sources)
