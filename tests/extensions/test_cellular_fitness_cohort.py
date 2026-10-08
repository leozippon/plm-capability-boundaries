"""Protein identity, source-bound CDS, label semantics and provenance checks."""
import hashlib
import json

import pytest
import io

from src.capability.extensions.cellular_fitness_cohort import (
    Acquisition, digest, duplicate_diagnostics, label_columns, map_coding_snv,
    parse_protein, refseq_reference, validate_row,
)

REF = dict(accession='NM_123.4', protein='MAE', cds='ATGGCTGAATAA')


def test_exact_protein_HGVS_identity_and_negative_WT():
    state, reason = validate_row(dict(hgvs_pro='p.Ala2Val'), REF)
    assert reason == 'validated_single_missense'
    assert state is not None
    assert state['mutation'] == 'A2V'
    assert state['mutant_sequence'] == 'MVE'
    assert state['mutant_sha256'] == hashlib.sha256(b'MVE').hexdigest()
    assert validate_row(dict(hgvs_pro='p.Cys2Val'), REF)[1] == 'protein_WT_mismatch'
    assert validate_row(dict(hgvs_pro='p.Ala20Val'), REF)[1] == 'protein_position_out_of_range'
    assert validate_row(dict(hgvs_pro='p.Ala2Val'), None)[1] == 'missing_source_bound_full_protein_WT'


@pytest.mark.parametrize('hgvs,reason', [('p.Met1Val','start_loss'), ('p.Ala2Ter','stop'),
    ('p.A2*','stop'), ('p.Ala2=','synonymous_or_WT'), ('p.Ala2Ala','synonymous_or_WT'),
    ('p.Ala2del','indel_or_frameshift'), ('p.[Ala2Val;Glu3Gly]','unsupported_or_multi_protein_HGVS'),
    ('p.?','missing_protein_annotation')])
def test_declared_protein_only_support(hgvs, reason):
    assert parse_protein(hgvs) == (None, reason)


def test_missing_mapping_cannot_guess_frame_or_isoform():
    assert map_coding_snv('NM_123.4:c.5C>T', None)[1] == 'missing_explicit_CDS_reference'
    assert map_coding_snv('NM_123.5:c.5C>T', REF)[1] == 'transcript_version_mismatch'
    assert map_coding_snv('NM_123.4:c.5A>T', REF)[1] == 'DNA_WT_mismatch'
    assert map_coding_snv('NC_123:g.5C>T', REF)[1] == 'missing_explicit_coding_mapping'
    assert map_coding_snv('NM_123.4:c.5+1C>T', REF)[1] == 'noncoding_or_splice_only'
    assert map_coding_snv('NM_123.4:c.*5C>T', REF)[1] == 'noncoding_or_splice_only'
    assert map_coding_snv('NM_123.4:c.1A>G', REF)[1] == 'start_loss'
    state, reason = validate_row(dict(hgvs_pro='NA', hgvs_nt='NM_123.4:c.5C>T'), REF)
    assert state is not None
    assert reason == 'validated_single_missense' and state['mutation'] == 'A2V'
    assert validate_row(dict(hgvs_pro='p.Ala2Val', hgvs_splice='c.5+1C>T'), REF)[1] == 'noncoding_splice_or_non_SNV_transcript_annotation'


def test_ddx3x_transcript_coordinate_field_is_not_splice_only():
    reference = dict(REF, protein_accession='NP_123.4')
    row = dict(hgvs_nt='g.29C>T', hgvs_splice='NM_123.4:c.5C>T', hgvs_pro='NP_123.4:p.Ala2Val')
    assert validate_row(row, reference)[1] == 'validated_single_missense'
    row['hgvs_pro'] = 'NP_123.5:p.Ala2Val'
    assert validate_row(row, reference)[1] == 'protein_accession_version_mismatch'
    row['hgvs_pro'] = 'NA'
    assert validate_row(row, reference)[1] == 'validated_single_missense'


def test_codon_duplicates_are_dependencies_not_averages_and_RNA_is_diagnostic():
    rows = [dict(wt_sha256='x', mutation='A2V', score_set='s', endpoint='score',
                 hgvs_nt=nt, label=label, rna_diagnostics=rna)
            for nt,label,rna in [('c.4G>T',-1,{}), ('c.5C>T',-3,{'rna_score':'-2'})]]
    result = duplicate_diagnostics(rows)
    assert result[0]['label_discordance'] and result[0]['nucleotide_variants'] == 2
    assert result[0]['label_range'] == 2 and result[0]['rna_diagnostic_present']
    assert 'no silent averaging' in result[0]['policy']
    assert [r['label'] for r in rows] == [-1,-3]


def test_classifier_rejection_and_measured_endpoint_selection():
    meta = dict(title='DDX3X SGE scores for clinical use', methodText='Random Forest posterior-probability',
                datasetColumns=dict(scoreColumns=['score','clinical_prediction']))
    assert label_columns(meta, 'ddx3x') == []
    assert label_columns(dict(methodText='combined log fold-change', datasetColumns=dict(scoreColumns=['score','D7_combined_LFC','cLFCd7_BH_FDR','prediction'])), 'ddx3x') == ['score','D7_combined_LFC']
    assert label_columns(dict(datasetColumns=dict(scoreColumns=['score','processed_LFC_continuous','clinical_score'])), 'bap1') == ['processed_LFC_continuous']
    assert label_columns(dict(datasetColumns=dict(scoreColumns=['score','rna_score_d6','score_no_DAB'])), 'vhl') == ['score','score_no_DAB']


def reference_text(codon_start='1', accession='NM_123.4'):
    return f'''<GBSet><GBSeq><GBSeq_accession-version>{accession}</GBSeq_accession-version>
    <GBSeq_sequence>CCCATGGCTGAATAA</GBSeq_sequence><GBSeq_feature-table><GBFeature>
    <GBFeature_key>CDS</GBFeature_key><GBFeature_location>4..15</GBFeature_location><GBFeature_quals>
    <GBQualifier><GBQualifier_name>codon_start</GBQualifier_name><GBQualifier_value>{codon_start}</GBQualifier_value></GBQualifier>
    <GBQualifier><GBQualifier_name>translation</GBQualifier_name><GBQualifier_value>MAE</GBQualifier_value></GBQualifier>
    <GBQualifier><GBQualifier_name>protein_id</GBQualifier_name><GBQualifier_value>NP_123.4</GBQualifier_value></GBQualifier>
    </GBFeature_quals></GBFeature></GBSeq_feature-table></GBSeq></GBSet>''' 


def test_refseq_requires_exact_version_and_explicit_validated_CDS():
    assert refseq_reference(reference_text(), 'NM_123.4')['protein'] == 'MAE'
    with pytest.raises(ValueError, match='version'):
        refseq_reference(reference_text(), 'NM_123.5')
    with pytest.raises(ValueError, match='frame'):
        refseq_reference(reference_text('2'), 'NM_123.4')
    with pytest.raises(ValueError, match='translation'):
        refseq_reference(reference_text().replace('>MAE<','>MVE<'), 'NM_123.4')


def test_source_hash_and_failed_request_receipt(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('public source unavailable')
    monkeypatch.setattr('urllib.request.urlopen', fail)
    fetch = Acquisition(tmp_path/'cache', tmp_path/'out')
    assert fetch.get('https://example.org/public', 'payload', 'CC0') is None
    receipt = json.loads((tmp_path/'out/acquisition.json').read_text())
    assert receipt['responses'] == 0 and receipt['receipts'][0]['failure'].endswith('public source unavailable')
    assert digest(b'source') == hashlib.sha256(b'source').hexdigest()


def test_source_hash_success_and_response_count(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status, url, headers = 200, 'https://example.org/public', {}
    monkeypatch.setattr('urllib.request.urlopen', lambda *args,**kwargs: Response(b'source'))
    fetch = Acquisition(tmp_path/'cache', tmp_path/'out')
    assert fetch.get('https://example.org/public','payload','CC0') == b'source'
    receipt = json.loads((tmp_path/'out/acquisition.json').read_text())
    assert receipt['responses'] == 1 and receipt['transferred_bytes'] == 6
    assert receipt['receipts'][0]['sha256'] == digest((tmp_path/'cache/payload').read_bytes())
