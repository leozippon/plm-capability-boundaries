"""Exact CIF numbering, conservative identity gates and bounded-acquisition tests."""
import gzip
import io
import json
from pathlib import Path

import pytest

from src.capability.extensions import binding_mapping as mapping

ROOT = Path(__file__).resolve().parents[2]

CIF = '''data_1ABC
_entry.id 1ABC
loop_
_entity.id
_entity.pdbx_mutation
1 ?
2 engineered
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code
_entity_poly.pdbx_seq_one_letter_code_can
1 polypeptide(L) ACD ACD
2 polypeptide(L) GG GG
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 ALA
1 2 CYS
1 3 ASP
2 1 GLY
2 2 GLY
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.auth_seq_num
_pdbx_poly_seq_scheme.pdb_strand_id
_pdbx_poly_seq_scheme.pdb_ins_code
X 1 1 ALA -1 99 A .
X 1 2 CYS 10 100 A a
X 1 3 ASP 11 101 A .
Y 2 1 GLY 1 1 B .
Y 2 2 GLY 2 2 B .
loop_
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.label_comp_id
_atom_site.auth_comp_id
X A 10 2 a CYS CYS
#
'''


def row(complex_id='1ABC_A_B', chain='A', position=10, insertion='a', wt='C', mt='A'):
    return dict(source_row_id='s:1', complex_id=complex_id, condition_group='condition', mutation_group='mutation',
                mutations=[dict(chain=chain, author_position=position,insertion_code=insertion,wt_aa=wt,mt_aa=mt)])


def test_exact_author_insertion_and_full_partner_sequences():
    reference = mapping.parse_reference(CIF, '1ABC')
    result = mapping.map_row(row(),reference)
    assert result['status'] == 'validated_operational_PDB_reference'
    assert result['sequence_index_1based'] == 2
    assert result['wt_sequence'] == 'ACD'
    assert result['mutant_sequence'] == 'AAD'
    assert {c['sequence'] for c in result['chain_states']} == {'ACD','GG'}
    assert result['wt_sha256'] == mapping.seqhash('ACD')
    assert result['chain_states'][1]['pdb_metadata']['metadata']['pdbx_mutation'] == 'engineered'
    assert result['construct_status'] == 'UNKNOWN_CONSTRUCT'
    assert result['experimental_construct_identity'].startswith('unverified')
    assert 'unverified' in result['model_score_coverage']


@pytest.mark.parametrize('mutation,expected', [(dict(position=100),'author_site_unmapped'),
                                              (dict(insertion=''),'author_site_unmapped'),
                                              (dict(wt='A'),'declared_WT_mismatch_or_ambiguous_site_monomer')])
def test_no_guessed_numbering_or_forced_wt(mutation,expected):
    result = mapping.map_row(row(**mutation),mapping.parse_reference(CIF,'1ABC'))
    assert result['status'] == 'blocked'
    assert expected in result['blockers']
    assert 'mutant_sha256' not in result


def test_unknown_partner_and_ambiguous_atom_mapping_block_states():
    unknown = CIF.replace('2 2 GLY','2 2 UNK')
    result = mapping.map_row(row(),mapping.parse_reference(unknown,'1ABC'))
    assert result['status'] == 'blocked'
    assert any('partner_chain_B' in b for b in result['blockers'])
    assert 'mutant_sha256' not in result
    ambiguous = CIF.replace('X A 10 2 a CYS CYS','X A 10 3 a CYS CYS')
    result = mapping.map_row(row(),mapping.parse_reference(ambiguous,'1ABC'))
    assert 'author_site_ambiguous' in result['blockers']


def test_identity_and_sequence_disagreement_fail_or_block():
    with pytest.raises(ValueError,match='identity mismatch'):
        mapping.parse_reference(CIF,'9XYZ')
    result = mapping.map_row(row(),mapping.parse_reference(CIF.replace('ACD ACD','ACD ADD'),'1ABC'))
    assert result['status'] == 'blocked'


def test_shared_partner_connects_different_mutated_proteins():
    reference = mapping.parse_reference(CIF,'1ABC')
    a = mapping.map_row(row(),reference)
    b = mapping.map_row(row(complex_id='2ABC_A_B'),reference)
    # A different mutated-chain sequence still shares the full partner.
    b['chain_states'][0]['wt_sha256'] = mapping.seqhash('AAA')
    b['variant_key'] = 'other'
    membership, coverage = mapping.components([a,b])
    assert len(set(membership.values())) == 1
    assert coverage[0]['complexes'] == 2
    assert coverage[0]['label_records'] == 2
    assert coverage[0]['distinct_single_variants'] == 2


def test_budget_failure_and_per_file_limit_receipted(tmp_path,monkeypatch):
    budget = mapping.Budget(total=3)
    budget.consume(3)
    with pytest.raises(ValueError,match='budget exhausted'):
        budget.consume(1)
    assert budget.used == 3
    payload = gzip.compress(CIF.encode())
    monkeypatch.setattr(mapping.urllib.request,'urlopen',lambda *a,**k: io.BytesIO(payload))
    monkeypatch.setattr(mapping,'MAX_FILE',10)
    result = mapping.acquire('1ABC',tmp_path,{},mapping.Budget())
    assert result['status'] == 'failed'
    assert 'per-file compressed limit' in result['error']
    assert result['download_bytes'] == 10
    assert not list(tmp_path.iterdir())


def test_inflight_reservations_cannot_overshoot_total_budget():
    budget = mapping.Budget(total=10)
    a = budget.reserve(8)
    b = budget.reserve(8)
    assert (a,b) == (8,2)
    with pytest.raises(ValueError,match='budget exhausted'):
        budget.reserve(1)
    budget.settle(a,5)
    budget.settle(b,2)
    assert budget.used == 7 and budget.reserved == 0
    assert budget.reserve(10) == 3


def test_public_payload_cache_and_failure_receipts(tmp_path,monkeypatch):
    payload = gzip.compress(CIF.encode())
    monkeypatch.setattr(mapping.urllib.request,'urlopen',lambda *a,**k: io.BytesIO(payload))
    result = mapping.acquire('1ABC',tmp_path,{},mapping.Budget())
    assert result['status'] == 'ready'
    assert result['download_bytes'] == len(payload)
    assert result['reference']['chains']['A']['sequence'] == 'ACD'
    cached = mapping.acquire('1ABC',tmp_path,{},mapping.Budget())
    assert cached['route'] == 'public_cache'
    assert cached['download_bytes'] == 0
    def fail(*args,**kwargs):
        raise TimeoutError('test timeout')
    monkeypatch.setattr(mapping.urllib.request,'urlopen',fail)
    failed = mapping.acquire('2ABC',tmp_path,{},mapping.Budget())
    assert failed['status'] == 'failed'
    assert 'TimeoutError' in failed['error']


def test_actual_1cse_author_mapping_from_cached_public_payload():
    path = ROOT/'data/phenotype_followups_20261007/binding-structures/1CSE.cif.gz'
    assert path.exists(), 'run bounded mapping preparation to stage the actual public test fixture'
    reference = mapping.parse_reference(mapping.read_payload(path),'1CSE')
    with (ROOT/'results/extensions/phenotype_followups_20261007/binding/admission.jsonl').open() as handle:
        first = json.loads(next(handle))
    result = mapping.map_row(first,reference)
    assert result['status'] == 'validated_operational_PDB_reference'
    assert result['wt_sequence'][result['sequence_index_1based']-1] == 'L'
    assert result['sequence_index_1based'] == 45
    assert len(result['wt_sequence']) == 70
    assert len(result['chain_states']) == 2
    assert {len(c['sequence']) for c in result['chain_states']} == {274,70}


def admitted_source_rows():
    phase1 = ROOT/'results/extensions/phenotype_followups_20261007/binding'
    return [r for line in (phase1/'admission.jsonl').read_text().splitlines()
            if (r := json.loads(line))['admitted']]


@pytest.mark.parametrize('complex_id,n', [('3S9D_A_B',156), ('1KBH_A_B',9),
                                         ('1A22_A_B',209), ('1BP3_A_B',52)])
def test_source_backed_background_conflicts_block_all_label_states(complex_id,n):
    rows = [r for r in admitted_source_rows() if r['complex_id'] == complex_id]
    assert len(rows) == n
    pdb = complex_id.split('_')[0]
    reference = mapping.parse_reference(mapping.read_payload(
        ROOT/f'data/phenotype_followups_20261007/binding-structures/{pdb}.cif.gz'),pdb)
    for r in rows:
        operational = mapping.map_row(r,reference,{'conflicts': []})
        assert operational['status'] == 'validated_operational_PDB_reference'
        blocked = mapping.map_row(r,reference)
        assert blocked['status'] == blocked['construct_status'] == 'KNOWN_INCOMPATIBLE'
        assert blocked['chain_states'] == []
        assert 'variant_key' not in blocked and 'mutant_sha256' not in blocked
        assert all(b.startswith('known_construct_conflict:') for b in blocked['blockers'])
        assert mapping.scoring_states([blocked]) == {}
        assert mapping.components([blocked]) == ({},[])


def test_frozen_adjudication_reviews_every_distinct_admitted_note():
    phase1 = ROOT/'results/extensions/phenotype_followups_20261007/binding'
    table = mapping.load_adjudication()
    mapping.validate_adjudication(ROOT,admitted_source_rows(),phase1,table)
    assert table['distinct_notes'] == 110
    table['review_groups'][0]['note_indexes'].pop()
    with pytest.raises(ValueError,match='review incomplete or changed'):
        mapping.validate_adjudication(ROOT,admitted_source_rows(),phase1,table)


def test_omitted_mutant_substitutions_are_row_scoped_not_blanket_pdb_exclusion():
    rows = [r for r in admitted_source_rows() if r['complex_id'] == '1BJ1_HL_VW']
    table = mapping.load_adjudication()
    assert sum(bool(mapping.construct_conflicts(r,table)) for r in rows) == 5
    assert len(rows) == 10
    engineered = next(r for r in admitted_source_rows() if r['complex_id'] == '3EQY_A_C')
    assert 'mutant' in engineered['original']['Notes']
    assert mapping.construct_conflicts(engineered,table) == []


def test_scoring_and_components_defensively_exclude_conflicting_construct_status():
    good = mapping.map_row(row(),mapping.parse_reference(CIF,'1ABC'))
    conflict = dict(good, construct_status='KNOWN_INCOMPATIBLE')
    assert mapping.qualified_rows([conflict,good]) == [good]
    assert mapping.scoring_states([conflict]) == {}
    assert mapping.components([conflict]) == ({},[])


def test_cache_only_never_uses_network(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('cache-only attempted a download')
    monkeypatch.setattr(mapping.urllib.request,'urlopen',forbidden)
    missing = mapping.acquire('1ABC',tmp_path,{},mapping.Budget(),cache_only=True)
    assert missing['status'] == 'failed'
    assert 'cache-only coordinate missing' in missing['error']
    assert missing['download_bytes'] == 0
    (tmp_path/'1ABC.cif.gz').write_bytes(gzip.compress(CIF.encode()))
    cached = mapping.acquire('1ABC',tmp_path,{},mapping.Budget(),cache_only=True)
    assert cached['status'] == 'ready' and cached['download_bytes'] == 0
