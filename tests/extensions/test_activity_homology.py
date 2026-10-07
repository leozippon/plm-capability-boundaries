"""Activity adapter invariants; shared alignment backend has its own tests."""
import json
from pathlib import Path

import pytest

from scripts.capability.extensions.qualify_activity_homology import load_activity, export_units
from src.capability.extensions.phenotype_homology import dump, seqhash, sha

ROOT = Path(__file__).resolve().parents[2]


def fixture_inputs(tmp_path):
    wt, mutant = 'A' * 40, 'C' + 'A' * 39
    h, mh = seqhash(wt), seqhash(mutant)
    groups = [dict(wt_sha256=h, assay_partitions=['a', 'b'], assays=2)]
    states = [dict(sequence=wt, sequence_sha256=h, length=40, roles=['WT'], assays=['a', 'b']),
              dict(sequence=mutant, sequence_sha256=mh, length=40, roles=['mutant'], assays=['a'])]
    readiness = [dict(assay=a, wt_sha256=h, WT_consistent=True, label_ready=False, physical_MSE_ready=False) for a in ('a', 'b')]
    for name, data in [('candidate-exactWT-groups.json', dict(groups=groups)), ('state-inventory.json', states), ('assay-readiness.json', readiness)]:
        dump(tmp_path / name, data)
    receipt = dict(schema='activity_identity_qualification_v1', qualified='identity only', blocked='labels',
        counts=dict(unique_exact_WT_proteins=1, downloaded_activity_files=2, unique_sequence_states=2,
                    label_ready_files=0, physical_MSE_ready_files=0),
        output_sha256={p.name: sha(p) for p in tmp_path.iterdir()})
    dump(tmp_path / 'receipt.json', receipt)
    return h, mh


def test_WT_only_extraction_no_binding_adjudication_required(tmp_path):
    h, mh = fixture_inputs(tmp_path)
    groups, chains, states, readiness, manifest = load_activity(tmp_path)
    assert set(chains) == {h}
    assert mh not in chains and len(states) == 2
    assert len(readiness) == 2 and manifest['counts']['label_ready_files'] == 0


def test_receipt_hash_mismatch_fails_before_alignment(tmp_path):
    fixture_inputs(tmp_path)
    (tmp_path / 'state-inventory.json').write_text('[]')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_activity(tmp_path)


def test_internal_sequence_hash_mismatch_even_when_file_rebound(tmp_path):
    fixture_inputs(tmp_path)
    states = json.loads((tmp_path / 'state-inventory.json').read_text())
    states[0]['sequence'] = 'D' * 40
    dump(tmp_path / 'state-inventory.json', states)
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    receipt['output_sha256']['state-inventory.json'] = sha(tmp_path / 'state-inventory.json')
    dump(tmp_path / 'receipt.json', receipt)
    with pytest.raises(ValueError, match='state hash'):
        load_activity(tmp_path)


def unit_inputs():
    chains = {'wt1': 'A' * 40, 'wt2': 'C' * 40, 'wt3': 'D' * 40, 'short': 'E' * 20}
    groups: list[dict] = [dict(wt_sha256=h, assay_partitions=assays, assays=len(assays)) for h, assays in
              [('wt1', ['a', 'b']), ('wt2', ['c']), ('wt3', ['d']), ('short', ['e'])]]
    states: list[dict] = [dict(sequence_sha256=h, roles=['WT'], assays=g['assay_partitions']) for h, g in zip(chains, groups)]
    states.append(dict(sequence_sha256='mutant', roles=['mutant'], assays=['a', 'c']))
    readiness = {a: dict(label_ready=False, physical_MSE_ready=False) for g in groups for a in g['assay_partitions']}
    return groups, chains, states, readiness


def test_full_membership_anchor_cascade_and_no_zero_family_claim():
    inputs = unit_inputs()
    membership = {'wt1': 'family', 'wt2': 'family', 'wt3': 'other', 'short': 'other'}
    hits = [dict(kind='anchor', query='wt1', admitted=True, family_overlap=True)]
    result = export_units(*inputs, membership, hits)
    assert result['preexclusion_WT_membership'] == membership
    assert result['candidate_WT_membership'] == {}
    assert result['counts']['full_components'] == 2  # Not zero just because all are excluded.
    assert result['counts']['direct_anchor_WTs'] == 1
    assert result['counts']['anchor_excluded_WTs'] == 2
    assert result['counts']['short_unresolved_components'] == 1
    assert result['counts']['remaining_sequence_states'] == 0
    units = {u['wt_sha256']: u for u in result['WT_units']}
    assert units['wt2']['cascade_only_anchor_excluded']
    assert units['wt3']['component_short_WT_unresolved']
    assert len(result['assay_membership']) == 5 and len(result['state_membership']) == 5
    assert result['labels_promoted'] is False and result['counts']['promoted_labels'] == 0


def test_lower_30_overlap_is_reported_not_excluded_or_claimed_negative():
    inputs = unit_inputs()
    membership = {h: h for h in inputs[1]}
    hits = [dict(kind='anchor', query='wt1', admitted=False, family_overlap=True)]
    result = export_units(*inputs, membership, hits)
    assert set(result['candidate_WT_membership']) == {'wt1', 'wt2', 'wt3'}
    assert result['counts']['remaining_assays'] == 4  # a/b are one WT, not independent proteins.
    assert result['counts']['remaining_WTs'] == 3
    assert result['counts']['remaining_sequence_states'] == 4
    assert result['counts']['lower_30_query_overlap_WTs'] == 1
    assert result['counts']['remaining_label_ready_assays'] == 0
    assert 'not absence of homology' in result['interpretation']


def test_frozen_actual_roster_has_only_97_WTs_and_1536_states():
    mapping = ROOT / 'results/extensions/phenotype_followups_20261007/activity/final-20261007'
    if not mapping.exists():
        pytest.skip('local frozen activity data not present')
    groups, chains, states, readiness, manifest = load_activity(mapping)
    assert len(groups) == len(chains) == 97
    assert len(states) == 1536 and len(readiness) == 130
    assert all(seqhash(sequence) == h for h, sequence in chains.items())
    assert len(manifest['output_sha256']) == 8
