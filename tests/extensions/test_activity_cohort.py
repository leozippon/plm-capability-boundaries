"""Identity, repeated assay partitions and uninterpreted-score boundaries."""
import hashlib
import json

import pytest

from src.capability.extensions.activity_cohort import (
    exact_groups, fetch_immutable, overlap, qualify_rows, reverse_substitution,
    seqhash, source_mapping, trace_calb_table, validate_release_receipts,
)


def row(mutant='C2A', sequence='AAD', score='1'):
    return dict(mutant=mutant, mutated_sequence=sequence, fitness_score=score)


def test_exact_reversal_statehash_and_ties():
    rows, report = qualify_rows('not_a_protein_identity', [row(), row('D3E','ACE'), row('A1A','ACD','nan')])
    assert report['WT_consistent']
    assert report['identity_rows'] == 2
    assert report['identity_substitution_rows'] == 1
    assert report['tied_score_groups'] == 1
    assert report['missing_label_rows'] == 1
    assert rows[0]['wt_sha256'] == hashlib.sha256(b'ACD').hexdigest()
    assert rows[0]['mutant_sha256'] == seqhash('AAD')
    assert rows[2]['identity_ready'] and not rows[2]['admissible_identity_single']
    assert not report['label_ready'] and not report['physical_MSE_ready']
    assert rows[0]['fitness_score_raw'] == '1'


@pytest.mark.parametrize('mutant,sequence', [('C2A','ACD'), ('C0A','AAD'), ('C4A','AAD'), ('C2A:D3E','AAE'), ('C2A','AXD'), ('C2A','aaD')])
def test_invalid_substitution_rejected(mutant, sequence):
    with pytest.raises(ValueError):
        reverse_substitution(mutant, sequence)


def test_inconsistent_background_or_length_blocks_entire_file():
    for bad in [row('D3E','VCE'), row('D3E','ACEE'), row('C2A','ACD')]:
        rows, report = qualify_rows('ambiguous', [row(), bad])
        assert not report['WT_consistent']
        assert report['identity_rows'] == 0
        assert all(not r['identity_ready'] for r in rows)


def test_duplicate_condition_kept_not_independent_and_missing_scale():
    rows, a = qualify_rows('condition1', [row(), row()])
    _, b = qualify_rows('condition2', [row()])
    groups = exact_groups([a,b])
    assert len(rows) == 2 and rows[0]['same_assay_state_rows'] == 2
    assert a['distinct_single_variants'] == 1
    discordant, qc = qualify_rows('condition1', [row(score='1'), row(score='2')])
    assert qc['discordant_same_assay_variant_groups'] == 1
    assert all(r['same_assay_state_discordant_labels'] for r in discordant)
    assert len(groups) == 1 and groups[0]['assays'] == 2
    assert groups[0]['verified_distinct_conditions'] is None
    assert not groups[0]['final_homology_family']
    mapping = source_mapping('condition1', [{'mutant_file_id':'condition1','doi':'doi.org/10.1000/example'}],
                             [{'dataset_id':'condition1','wt_sequence':'ACD','category':'activity'}])
    assert mapping['source_doi_candidates'] == ['10.1000/example']
    assert mapping['unit'] is None and mapping['direction'] is None
    uncertain = source_mapping('x', [{'mutant_file_id':'x','doi':'?10.1000/example'}], [])
    assert uncertain['source_doi_candidates'] == []
    assert uncertain['source_doi_raw'] == ['?10.1000/example']


def test_identity_survives_missing_label_without_rank_readiness():
    rows, report = qualify_rows('a', [row(score=''),row('D3E','ACE','inf')])
    assert report['identity_rows'] == 2
    assert report['finite_single_labels'] == 0
    assert not report['descriptive_nonconstant_single_support']
    assert all(not r['finite_label'] for r in rows)


def test_actual_anchor_exact_and_containment_distinguish_membership():
    anchors = [dict(assay='oldonly',sequence='ACD',retained201=False),
               dict(assay='retained',sequence='MACD',retained201=True)]
    matches = overlap('ACD', anchors)
    assert matches[0]['relation'] == 'exact' and not matches[0]['retained201']
    assert matches[1]['relation'] == 'activity_contained_in_anchor' and matches[1]['retained201']


def test_targeted_units_need_all_row_original_table_match(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    paper = source/'5A71-source-paper.xml'
    paper.write_text('''<article><article-id pub-id-type="doi">10.1038/s41467-019-11155-3</article-id>
    <table-wrap id="Tab1"><table><thead><tr><th>Substrate</th><th>Entry</th><th>Enzymes</th><th>Mutations</th><th>kcat (s−1)</th></tr></thead>
    <tbody><tr><td>1</td><td>1</td><td>WT</td><td>–</td><td>0.025 ± 0.009</td></tr>
    <tr><td></td><td>2</td><td>QW1</td><td>C2A</td><td>0.028 ± 0.007</td></tr></tbody></table></table-wrap></article>''')
    receipt = dict(status='ready',sha256=hashlib.sha256(paper.read_bytes()).hexdigest(), origin_url='https://public.example/paper.xml')
    (source/'5A71-europepmc-fulltext-receipt.json').write_text(json.dumps(receipt))
    rows, _ = qualify_rows('5A71_kcat',[row('A1A','ACD','0.025'),row(score='0.028')])
    trace = trace_calb_table(tmp_path,'5A71_kcat',rows)
    assert trace is not None
    assert trace['unit'] == 's^-1' and len(trace['rows']) == 2
    assert not trace['full_experimental_WT_construct_verified']
    assert trace_calb_table(tmp_path,'unknown_kcat',rows) is None
    rows[1]['source_score'] = 0.029
    with pytest.raises(ValueError,match='score row mismatch'):
        trace_calb_table(tmp_path,'5A71_kcat',rows)
    paper.write_text(paper.read_text().replace('s−1','mM'))
    with pytest.raises(ValueError,match='receipt/hash mismatch'):
        trace_calb_table(tmp_path,'5A71_kcat',rows)


def test_offline_release_requires_complete_unique_source_inventory():
    records = [dict(source_path=f'single_mutant/activity/a{i}.csv', status='ready') for i in range(130)]
    validate_release_receipts(records)
    for bad in [records[:-1], [*records[:-1], records[0]],
                [*records[:-1], dict(source_path='../outside.csv', status='ready')]]:
        with pytest.raises(ValueError, match='receipt coverage'):
            validate_release_receipts(bad)


def test_cached_raw_is_verified_never_overwritten(tmp_path):
    p = tmp_path/'raw.csv'
    payload = b'public bytes'
    p.write_bytes(payload)
    oid = hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
    receipt = fetch_immutable('https://huggingface.co/public/not-requested', p, len(payload), oid)
    assert receipt['status'] == 'ready' and receipt['network_bytes'] == 0
    bad = fetch_immutable('https://huggingface.co/public/not-requested', p, len(payload)+1, oid)
    assert bad['status'] == 'failed'
    assert p.read_bytes() == payload
