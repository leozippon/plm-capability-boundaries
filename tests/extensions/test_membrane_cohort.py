"""Exact state identity, dependency/missingness and source-parser failure tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.capability.extensions.membrane_cohort import (
    DISCOVERY, anchor_matches, bind_state, dependencies, intersections,
    number, parse_change, prepare, sgca_coding_wt, sgca_prefix, sha, table,
)

ROOT = Path(__file__).resolve().parents[2]


def test_exact_identity_and_wrong_WT():
    state, status = bind_state('MACK', 'p.Ala2Gly')
    assert status == 'validated'
    assert state is not None
    assert state['mutation'] == 'A2G'
    assert state['mutated_sequence'] == 'MGCK'
    assert state['wt_sha256'] == sha('MACK')
    assert state['mutant_sha256'] == sha('MGCK')
    assert bind_state('MDCK', 'p.Ala2Gly')[1] == 'WT_residue_or_numbering_mismatch'
    assert bind_state('MACK', 'p.Ala9Gly')[0] is None
    assert bind_state(None, 'p.Ala2Gly')[1].startswith('source_full_protein_WT_unavailable')
    with pytest.raises(ValueError, match='noncanonical'):
        bind_state('MAXK', 'p.Ala2Gly')


@pytest.mark.parametrize('hgvs,status', [('p.Ala2Ter', 'stop'), ('p.Ala2*', 'stop'),
    ('p.Ala2=', 'synonymous'), ('p.Ala2Ala', 'synonymous'),
    ('p.Met1Leu', 'start_loss_or_initiator_substitution'),
    ('p.Ala2Gly;Cys3Arg', 'not_single_protein_substitution'),
    ('p.Ala2del', 'not_single_protein_substitution'), ('p.=', 'not_single_protein_substitution')])
def test_missense_vs_stop_startloss_and_multi(hgvs, status):
    assert bind_state('MACK', hgvs)[1] == status
    assert bind_state('MACK', 'p.Ala2Gly', consequence='Start-loss')[0] is None


def test_no_imputation_or_uncensoring():
    assert number('NA') == (None, 'missing')
    assert number('inf') == (None, 'nonfinite')
    assert number('>0.2') == (None, 'censored_bound')
    assert number('0') == (0, 'finite')
    with pytest.raises(ValueError, match='invalid measured'):
        number('broken')
    assert parse_change('p.Xaa2Gly') is None


def test_duplicate_codons_ties_missing_and_intersection():
    rows = [dict(channel='a', state_id='wt1:mt1', accepted=True, score=1, nucleotide='c.4A>C'),
            dict(channel='a', state_id='wt1:mt1', accepted=True, score=1, nucleotide='c.5A>C'),
            dict(channel='a', state_id='wt1:mt2', accepted=False, score=None, nucleotide='c.6A>C'),
            dict(channel='b', state_id='wt1:mt1', accepted=True, score=1),
            dict(channel='b', state_id='wt2:mt1', accepted=True, score=1)]
    dependencies(rows[:3])
    assert rows[0]['dependency_group_size'] == rows[1]['dependency_group_size'] == 2
    assert rows[0]['nucleotide'] != rows[1]['nucleotide']
    assert rows[0]['score'] == rows[1]['score'] == 1
    result = intersections(rows, ['a', 'b'])
    assert result[0]['state_ids'] == ['wt1:mt1']
    assert result[0]['n_states'] == 1
    assert len(rows) == 5


def test_source_parse_guard(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text('a,b\n1,2,3\n')
    with pytest.raises(ValueError, match='width'):
        table(path)
    path.write_text('a,a\n1,2\n')
    with pytest.raises(ValueError, match='header'):
        table(path)
    rows, qc = sgca_prefix(ROOT / DISCOVERY / 'catalog_candidates/sgca_scores.csv')
    assert len(rows) == 3483
    assert qc['malformed_alleles_rows'] == 3483
    assert all('MLS3' in r and 'sigma' in r and 'hgvs_nt' in r for r in rows)
    assert all('alleles' not in r and 'alphamissense' not in r and '3low_reads' not in r for r in rows)
    assert rows[0]['source_consequence'] == 'Start-loss'


def sgca_real_sources():
    meta = json.loads((ROOT / DISCOVERY / 'catalog_candidates/sgca_metadata.json').read_text())
    rows, _ = sgca_prefix(ROOT / DISCOVERY / 'catalog_candidates/sgca_scores.csv')
    return meta, rows


def test_sgca_source_anchored_translation_without_terminal_stop():
    meta, rows = sgca_real_sources()
    wt, qc = sgca_coding_wt(meta, rows)
    assert len(wt) == 387 and wt[0] == 'M' and '*' not in wt
    assert qc['nucleotide_positions_covered'] == 1161
    assert qc['codons_covered'] == 387
    assert qc['crosscheck_conflicts'] == 0
    assert not qc['terminal_stop_required']
    assert qc['source_consequences'] == dict(Missense=2482, Synonymous=897, Nonsense=95, **{'Start-loss': 9})
    duplicated_wt, _ = sgca_coding_wt(meta, rows + [dict(rows[10])])
    assert duplicated_wt == wt


@pytest.mark.parametrize('field,value,error', [
    ('WT_nt', 'G', 'nucleotide/site/codon/reference'),
    ('hgvs_nt', 'c.1G>C', 'nucleotide/site/codon/reference'),
    ('codon', '2', 'nucleotide/site/codon/reference'),
    ('WT_AA', 'A', 'translated amino acid'),
    ('Variant_AA', 'A', 'translated amino acid'),
    ('hgvs_pro', 'p.Met2Leu', 'protein HGVS/consequence'),
    ('sigma', '-1', 'invalid measured sigma'),
])
def test_sgca_annotation_conflicts_fail(field, value, error):
    meta, rows = sgca_real_sources()
    rows[0][field] = value
    with pytest.raises(ValueError, match=error):
        sgca_coding_wt(meta, rows)


def test_sgca_wrong_frame_and_incomplete_coverage_fail():
    meta, rows = sgca_real_sources()
    dna = meta['targetGenes'][0]['targetSequence']['sequence']
    meta['targetGenes'][0]['targetSequence']['sequence'] = dna[1:] + dna[0]
    with pytest.raises(ValueError, match='coding target|conflict'):
        sgca_coding_wt(meta, rows)
    meta['targetGenes'][0]['targetSequence']['sequence'] = dna
    with pytest.raises(ValueError, match='incomplete coding SNV coverage'):
        sgca_coding_wt(meta, rows[1:])


def test_anchor_exact_and_containment_not_homology():
    result = anchor_matches('MACK', {'exact': 'MACK', 'long': 'AMACKT', 'short': 'ACK', 'different': 'MACR'})
    assert result['exact'] == ['exact']
    assert result['anchor_contains_panel'] == ['long']
    assert result['panel_contains_anchor'] == ['short']
    assert result['homology_exclusion'] == 'NOT_ESTABLISHED'


def test_real_source_end_to_end(tmp_path):
    summary = prepare(ROOT, tmp_path)
    assert summary['validated_WT_proteins'] == 3
    assert summary['unique_candidate_proteins'] == 3
    for letter in 'abcde':
        counts = summary['channels'][f'F9_{letter}-1']
        assert counts['rows'] == 9682
        expected = {'a': 8525, 'b': 8527, 'c': 8528, 'd': 8528, 'e': 8528}[letter]
        assert counts['accepted_rows'] == counts['accepted_states'] == expected
        assert counts['finite_identity_validated_scores'] == 8528
    assert summary['channels']['RHO_method1']['rows'] == 7244
    assert summary['channels']['RHO_method2']['rows'] == 7244
    assert summary['channels']['RHO_method1']['accepted_states'] == 6341
    assert summary['channels']['RHO_method2']['accepted_states'] == 6592
    assert summary['channels']['RHO_method1']['discordant_both_method_rows'] == 410
    assert summary['channels']['RHO_method2']['discordant_both_method_rows'] == 410
    assert summary['channels']['SGCA_surface']['accepted_rows'] == 2482
    assert summary['channels']['SGCA_surface']['accepted_states'] == 2257
    assert summary['SGCA_pre_gate_candidates'] == dict(finite_missense_records=2491, distinct_HGVS=2263, duplicate_HGVS_extra_rows=228)
    proteins = json.loads((tmp_path / 'canonical-WT-manifest.json').read_text())
    assert proteins['RHO']['length'] == 348
    assert proteins['F9']['length'] == 461
    assert proteins['SGCA']['length'] == 387
    assert proteins['SGCA']['wt'].startswith('M')
    assert proteins['SGCA']['identity_provenance']['coding_nt'] == 1161
    assert proteins['SGCA']['full_FLAG_construct_context'].startswith('UNKNOWN')
    intersections_ = json.loads((tmp_path / 'aligned-channel-intersections.json').read_text())
    assert len(intersections_['F9']) == 26
    assert all(item['n_states'] <= 8528 for item in intersections_['F9'])
    assert next(item for item in intersections_['F9'] if item['channels'] == ['F9_c-1', 'F9_d-1', 'F9_e-1'])['n_states'] == 8528
    rows = [json.loads(line) for line in (tmp_path / 'channel-rows.jsonl').read_text().splitlines()]
    assert len(rows) == 5 * 9682 + 2 * 7244 + 3483
    assert all(r['model_score_coverage'].startswith('UNKNOWN') for r in rows)
    assert all(r['state_id'] for r in rows if r['accepted'])
    assert all(r['quality']['source_method_discordance'] is None for r in rows
               if r['protein'] == 'RHO' and r['original']['consequence'] != 'missense')
    assert not any('composite_score' in r['original'] or 'ClinVar' in str(r['original']) for r in rows)
    states = [json.loads(line) for line in (tmp_path / 'canonical-mutant-states.jsonl').read_text().splitlines()]
    assert len({r['state_id'] for r in states}) == len(states)
    sgca_rows = [r for r in rows if r['protein'] == 'SGCA']
    accepted_sgca = [r for r in sgca_rows if r['accepted']]
    assert len(accepted_sgca) - len({r['state_id'] for r in accepted_sgca}) == 225
    assert all(r['quality']['tail_parse'] == 'quarantined' for r in accepted_sgca)
    assert all(not r['accepted'] for r in sgca_rows if r['original']['codon'] == '1')
    groups = {}
    for row in accepted_sgca:
        groups.setdefault(row['state_id'], []).append(row)
    assert any(len({r['score'] for r in group}) > 1 for group in groups.values() if len(group) > 1)
    assert all(r['dependency_group_size'] == len(groups[r['state_id']]) for r in accepted_sgca)
    assert all(r['original']['source_consequence'] == 'Missense' for r in accepted_sgca)
    assert len([s for s in states if s['protein'] != 'SGCA']) == 15333
    for state in states:
        wt = proteins[state['protein']]['wt']
        assert sum(a != b for a, b in zip(wt, state['mutated_sequence'])) == 1
        assert sha(state['mutated_sequence']) == state['mutant_sha256']
