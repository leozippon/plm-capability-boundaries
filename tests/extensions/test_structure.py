"""Admission, missingness and independent all-residue coordinate checks."""
from pathlib import Path
import copy
import json

import numpy as np
import pytest

from src.capability.extensions.structure import (
    _candidates, annotate_cohort, annotate_residues, exact_mapping, load_anchor,
    mutation_positions, mutation_site_rows, site_table,
)
from src.capability.interactions.contact_enrichment import Chain, Residue, load_structure

ROOT = Path(__file__).resolve().parents[2]
RETAINED = ROOT / 'archive/logs/shared/gate_structure_contact_20260924/structures/1UFM.cif.gz'


def test_exact_admission():
    assert exact_mapping('ACD', 'ACD')['kind'] == 'full_entity'
    assert exact_mapping('ACD', 'MACDE')['mapping'] == {1: 2, 2: 3, 3: 4}
    assert exact_mapping('ACD', 'ACE')['status'] == 'no_exact_full_wt_match'
    assert exact_mapping('ACD', 'ACDACD')['status'] == 'ambiguous_exact_mapping'
    assert exact_mapping('ACD', 'AD')['mapping'] == {}
    with pytest.raises(ValueError):
        exact_mapping('', 'ACD')


def test_missing_coordinates_strict_cb_and_degree():
    # WT 1 is at exactly 8 A from WT 4: no contact. WT 5 Gly CA is
    # within 8 A. WT 2 missing coordinates; WT 3 non-Gly CA-only is missing CB.
    residues = {}
    for i, aa, atom, x in [(1, 'A', 'CB', 0), (3, 'A', 'CA', 0),
                            (4, 'A', 'CB', 8), (5, 'G', 'CA', 7)]:
        residues[i] = Residue(i, str(i), 'GLY' if aa == 'G' else 'ALA', aa,
                              [atom], ['C'], [(x, 0., 0.)])
    chain = Chain('A', 'A', '1', 1, residues)
    rows = annotate_residues('AAAAG', chain, {i: i for i in range(1, 6)})
    assert rows[0]['contact_degree'] == 1
    assert rows[3]['contact_degree'] == 0
    assert rows[1]['rsa'] is None and rows[1]['contact_degree'] is None
    assert not rows[1]['coordinate_present']
    assert rows[2]['coordinate_present'] and rows[2]['contact_degree'] is None
    assert rows[2]['contact_coordinate'] is None
    assert rows[4]['contact_coordinate'] == [7., 0., 0.]
    assert all(r['secondary_structure'] is None for r in rows)
    with pytest.raises(ValueError, match='complete WT'):
        annotate_residues('AAAAG', chain, {1: 1})
    with pytest.raises(ValueError, match='identity'):
        annotate_residues('CAAAG', chain, {i: i for i in range(1, 6)})


def test_anchor_projection_and_join(tmp_path):
    path = tmp_path / 'cohort.json'
    def save(label):
        path.write_text(json.dumps({'assays': [{'assay': 'test', 'cluster': 2,
            'wildtype': 'ACD', 'mutants': ['A1V:C2A'], 'measured': [label]}]}))
    save(10)
    first = load_anchor(path)
    save(-100)
    second = load_anchor(path)
    assert first['assays'] == second['assays']
    assert 'measured' not in first['assays'][0]
    receipt = annotate_cohort(first, [])
    assert receipt['coverage']['assays_admitted'] == 0
    assert len(site_table(first, receipt)) == 3
    joined = mutation_site_rows(first, receipt)
    assert [r['wt_position'] for r in joined] == [1, 2]
    assert all(r['contact_degree'] is None for r in joined)
    with pytest.raises(ValueError, match='unknown assay'):
        load_anchor(path, assay_ids=['absent'])
    with pytest.raises(ValueError, match='hash mismatch'):
        site_table(second, receipt)
    with pytest.raises(ValueError):
        mutation_positions('D1A', 'ACD')


def test_real_retained_coordinates_and_outcome_blind_choice():
    if not RETAINED.exists():
        pytest.skip('retained experimental coordinate archive unavailable')
    structure = load_structure(RETAINED)
    entity = next(iter(structure['entities']))
    wt = structure['entities'][entity]
    anchor = {'cohort_sha256': 'test', 'assays': [
        {'assay': 'coordinate-control', 'cluster': 1, 'wildtype': wt, 'mutants': []}]}
    receipt = annotate_cohort(anchor, [RETAINED])
    assert receipt['coverage']['assays_admitted'] == 1
    saved = receipt['assays'][0]
    chain = structure['chains'][(saved['model'], saved['chain'])]
    sites = saved['residues']
    assert len(sites) == len(wt)
    # Independent brute-force strict CB/Gly-CA calculation over all WT sites.
    for row in sites:
        residue = chain.residues.get(row['entity_position'])
        if residue is None:
            assert row['rsa'] is None and row['contact_degree'] is None
            continue
        vector = residue.named('CA' if row['residue'] == 'G' else 'CB')
        if vector is None:
            assert row['contact_degree'] is None
            continue
        assert np.array_equal(vector, row['contact_coordinate'])
        degree = 0
        for other in sites:
            second = chain.residues.get(other['entity_position'])
            if second is None or abs(row['wt_position'] - other['wt_position']) <= 2:
                continue
            target = second.named('CA' if other['residue'] == 'G' else 'CB')
            degree += target is not None and np.linalg.norm(vector - target) < 8
        assert row['contact_degree'] == degree
        assert np.isfinite(row['rsa']) and row['rsa'] >= 0
    poisoned = copy.deepcopy(anchor)
    poisoned['assays'][0]['measured'] = [1e10]
    assert annotate_cohort(poisoned, [RETAINED]) == receipt
    assert len(saved['source_sha256']) == 64
    predicted = copy.deepcopy(structure)
    predicted['method'] = 'THEORETICAL MODEL'
    assert _candidates(wt, predicted)[0] == []
    damaged = copy.deepcopy(structure)
    for chain in damaged['chains'].values():
        position = next(iter(chain.residues))
        chain.residues[position].residue = 'X'
    assert _candidates(wt, damaged)[0] == []
