"""Source-bound cellular-fitness preparation only: no inference or fitting.

A score row is not a protein state. Clinical classifiers are not measured labels,
and a transcript accession alone is not a guessed coding frame. RNA is retained
as a labelled diagnostic, never as an unmarked protein-only predictor.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import xml.etree.ElementTree as ET
from src.capability.core.amino_acids import AA20

# Cohort-local HGVS names; Bio is not installed in the validated ct runtime.
HGVS_AA: dict[str, str] = dict(zip('Ala Cys Asp Glu Phe Gly His Ile Lys Leu Met Asn Pro Gln Arg Ser Thr Val Trp Tyr'.split(), AA20))
# NCBI standard genetic code, Base1/Base2/Base3 each in TCAG order.
CODONS = {a+b+c: aa for (a,b,c), aa in zip(
    ((a,b,c) for a in 'TCAG' for b in 'TCAG' for c in 'TCAG'),
    'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG')}


def translate(dna: str) -> str:
    if len(dna)%3 or set(dna)-set('ACGT'):
        raise ValueError('noncanonical or incomplete coding sequence')
    return ''.join(CODONS[dna[i:i+3]] for i in range(0,len(dna),3))

GENES = 'bap1 ddx3x bard1 ctcf palb2 rad51c rad51d xrcc2 sbds sfpq vhl tinf2'.split()
MAX_BYTES = 150 * 1024**2


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


class Acquisition:
    """Serial bounded transfers, receipt after every request including failures."""
    def __init__(self, cache: Path, out: Path, prior_receipts=None):
        self.cache, self.out = cache, out
        self.prior = prior_receipts or []
        self.used = 0
        self.receipts = []
        self.cache.mkdir(parents=True, exist_ok=True)

    def get(self, url: str, name: str, license_info=None, maximum=12*1024**2):
        receipt: dict = dict(origin_api_url=url, path=name, as_of='2026-10-07',
                       retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                       license=license_info, response_count=0, bytes=0)
        try:
            previous = next((r for r in self.prior if r.get('origin_api_url') == url and r.get('sha256')), None)
            if previous and (self.cache/name).exists():
                payload = (self.cache/name).read_bytes()
                if digest(payload) != previous['sha256']:
                    raise ValueError('cached source hash mismatch')
                receipt.update(status='reused_hash_verified', sha256=previous['sha256'], bytes=len(payload), original_receipt=previous)
                return payload
            if self.used >= MAX_BYTES:
                raise ValueError('total transfer budget exhausted')
            limit = min(maximum, MAX_BYTES-self.used)
            request = urllib.request.Request(url, headers={'User-Agent': 'cellular-fitness-public-qualification/1.0'})
            with urllib.request.urlopen(request, timeout=45) as response:
                receipt.update(http_status=response.status, final_url=response.url,
                               etag=response.headers.get('ETag'), last_modified=response.headers.get('Last-Modified'))
                pieces = []
                while True:
                    block = response.read(min(65536, limit-receipt['bytes']))
                    if not block:
                        break
                    self.used += len(block)
                    receipt['bytes'] += len(block)
                    if receipt['bytes'] >= limit:
                        raise ValueError('bounded response reached budget; conservatively rejected')
                    pieces.append(block)
            payload = b''.join(pieces)
            receipt.update(sha256=digest(payload), response_count=1, status='acquired')
            (self.cache/name).write_bytes(payload)
            return payload
        except Exception as error:
            receipt.update(status='failed', failure=f'{type(error).__name__}: {error}')
            return None
        finally:
            self.receipts.append(receipt)
            dump(self.out/'acquisition.json', dict(max_bytes=MAX_BYTES, transferred_bytes=self.used,
                 request_count=len(self.receipts), responses=sum(r['response_count'] for r in self.receipts), receipts=self.receipts))


def refseq_reference(text: str, accession: str) -> dict:
    """Use versioned RefSeq feature CDS, codon_start and deposited translation."""
    records = ET.fromstring(text).findall('GBSeq')
    if len(records) != 1 or records[0].findtext('GBSeq_accession-version') != accession:
        raise ValueError('RefSeq version mismatch')
    record = records[0]
    cds = [f for f in record.findall('./GBSeq_feature-table/GBFeature') if f.findtext('GBFeature_key') == 'CDS']
    if len(cds) != 1:
        raise ValueError('requires exactly one explicit CDS')
    feature = cds[0]
    qualifiers = {q.findtext('GBQualifier_name'):q.findtext('GBQualifier_value') for q in feature.findall('./GBFeature_quals/GBQualifier')}
    if qualifiers.get('codon_start') != '1' or qualifiers.get('transl_table','1') != '1':
        raise ValueError('unsupported explicit CDS frame/table')
    location = feature.findtext('GBFeature_location','')
    match = re.fullmatch(r'(\d+)\.\.(\d+)', location)
    if not match:
        raise ValueError('partial or compound CDS unsupported, cannot guess')
    start,end = map(int, match.groups())
    transcript = record.findtext('GBSeq_sequence','').upper()
    if not 1 <= start <= end <= len(transcript):
        raise ValueError('CDS coordinates out of range')
    dna = transcript[start-1:end]
    protein = qualifiers.get('translation','')
    if not protein or translate(dna) != protein+'*':
        raise ValueError('CDS translation mismatch')
    if set(protein)-set(AA20):
        raise ValueError('incomplete canonical CDS')
    return dict(accession=accession, protein=protein, cds=dna, protein_accession=qualifiers.get('protein_id'),
                identity_binding='exact versioned source target transcript + deposited CDS feature and translation')


def parse_protein(hgvs: str) -> tuple[dict | None, str]:
    text = hgvs.split(':')[-1]
    match = re.fullmatch(r'p\.\(?([A-Z][a-z]{2}|[A-Z])(\d+)([A-Z][a-z]{2}|[A-Z]|\*|=)\)?', text)
    if not match:
        if 'del' in text or 'ins' in text or 'dup' in text or 'fs' in text:
            return None, 'indel_or_frameshift'
        if hgvs in ('', 'NA', 'p.?'):
            return None, 'missing_protein_annotation'
        return None, 'unsupported_or_multi_protein_HGVS'
    ref, pos, alt = match.groups()
    ref = HGVS_AA.get(ref, ref)
    alt = HGVS_AA.get(alt, alt)
    pos = int(pos)
    if alt in ('Ter', '*') or ref in ('Ter', '*'):
        return None, 'stop'
    if ref not in AA20 or alt not in AA20+'=':
        return None, 'noncanonical_residue'
    if alt == ref or alt == '=':
        return None, 'synonymous_or_WT'
    if pos == 1 and ref == 'M':
        return None, 'start_loss'
    return dict(wt_aa=ref, position=pos, mt_aa=alt, mutation=f'{ref}{pos}{alt}'), 'single_missense'


def map_coding_snv(hgvs: str, reference: dict | None) -> tuple[dict | None, str]:
    """Only exact transcript c.SNV, explicit deposited CDS; no genomic guesses."""
    if any(x in hgvs for x in ('del', 'ins', 'dup')):
        return None, 'indel_or_frameshift'
    match = re.fullmatch(r'([^:]+):c\.(\d+)([ACGT])>([ACGT])', hgvs)
    if not match:
        if ':c.' in hgvs and re.search(r':c\.(?:\*|-|\d+[+-])', hgvs):
            return None, 'noncoding_or_splice_only'
        return None, 'missing_explicit_coding_mapping'
    if not reference or 'cds' not in reference:
        return None, 'missing_explicit_CDS_reference'
    accession, position, wt, alt = match.groups()
    if accession != reference['accession']:
        return None, 'transcript_version_mismatch'
    index = int(position)-1
    dna = reference['cds']
    if not 0 <= index < len(dna):
        return None, 'coding_position_out_of_range'
    if dna[index] != wt:
        return None, 'DNA_WT_mismatch'
    start = index//3*3
    codon = dna[start:start+3]
    changed = codon[:index%3]+alt+codon[index%3+1:]
    refaa, altaa = translate(codon), translate(changed)
    return parse_protein(f'p.{refaa}{index//3+1}{altaa}')


def validate_row(row: dict, reference: dict | None) -> tuple[dict | None, str]:
    # DDX3X exon exports use hgvs_splice for ALL transcript-coordinate variants,
    # including ordinary coding SNVs. This is a coordinate field, not an RNA label.
    transcript_hgvs = row.get('hgvs_splice', 'NA')
    if transcript_hgvs not in ('NA', '', None) and not re.fullmatch(r'[^:]+:c\.\d+[ACGT]>[ACGT]', transcript_hgvs):
        return None, 'noncoding_splice_or_non_SNV_transcript_annotation'
    if row.get('variant_qc_flag', '') not in ('', 'NA', 'PASS', 'pass', 'False', '0'):
        return None, 'source_QC_flag_requires_adjudication'
    parsed, reason = parse_protein(row.get('hgvs_pro', 'NA'))
    if reason == 'missing_protein_annotation':
        nt = transcript_hgvs if transcript_hgvs not in ('NA', '', None) else row.get('hgvs_nt', '')
        if reference and reference.get('source_n_reference_is_exact_CDS') and re.fullmatch(r'n\.\d+[ACGT]>[ACGT]', nt):
            nt = reference['accession']+':c.'+nt[2:]
        parsed, reason = map_coding_snv(nt, reference)
    if not parsed:
        return None, reason
    if not reference or not reference.get('protein'):
        return None, 'missing_source_bound_full_protein_WT'
    protein_hgvs = row.get('hgvs_pro','')
    if ':' in protein_hgvs and protein_hgvs.split(':')[0] != reference.get('protein_accession'):
        return None, 'protein_accession_version_mismatch'
    protein = reference['protein']
    pos = parsed['position']
    if not 1 <= pos <= len(protein):
        return None, 'protein_position_out_of_range'
    if protein[pos-1] != parsed['wt_aa']:
        return None, 'protein_WT_mismatch'
    mutant = protein[:pos-1]+parsed['mt_aa']+protein[pos:]
    return dict(**parsed, wt_sha256=digest(protein.encode()), mutant_sha256=digest(mutant.encode()),
                mutant_sequence=mutant), 'validated_single_missense'


def label_columns(metadata: dict, gene: str) -> list[str]:
    """Only these source-specific measured fitness endpoints, never classifiers."""
    columns = metadata.get('datasetColumns', {}).get('scoreColumns', [])
    if gene == 'ddx3x':
        method = metadata.get('methodText', '').lower()
        if 'random forest' in method or 'posterior-probability' in method or 'clinical use' in metadata.get('title','').lower():
            return []
        # Primary experimental exon maps explicitly define score as cLFC-trend;
        # BH/FDR and Z statistics are QC, not the phenotype itself.
        return [c for c in columns if re.fullmatch(r'D(?:7|11|15|21)_combined_LFC', c) or c == 'score' and ('clfc-trend is given' in method or 'combined log fold' in method)]
    if gene == 'bap1':
        return [c for c in columns if c.startswith('processed_LFC_')]
    if gene == 'rad51c':
        return [c for c in columns if c.startswith('processed_adj_lfc_')]
    if gene == 'vhl':
        return [c for c in columns if c in ('score', 'score_no_DAB', 'score_HAP1-HIF1A-KO')]
    return ['score'] if 'score' in columns else []


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def duplicate_diagnostics(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in records:
        groups[(row['wt_sha256'], row['mutation'], row['score_set'], row['endpoint'])].append(row)
    return [dict(wt_sha256=k[0], mutation=k[1], score_set=k[2], endpoint=k[3], records=len(v),
                 nucleotide_variants=len({r['hgvs_nt'] for r in v}),
                 label_range=max(r['label'] for r in v)-min(r['label'] for r in v),
                 label_discordance=len({r['label'] for r in v})>1,
                 rna_diagnostic_present=any(r['rna_diagnostics'] for r in v),
                 policy='dependency group; no silent averaging; RNA sensitivity is label-assisted')
            for k,v in sorted(groups.items()) if len({r['hgvs_nt'] for r in v})>1]


def prepare(root: Path, cache: Path, out: Path, acquire: bool = True) -> dict:
    discovery = root/'results/extensions/phenotype_followups_20261007/discovery'
    registry = json.loads((discovery/'candidate_registry.json').read_text())
    original_receipts = json.loads((discovery/'consolidated_acquisition_manifest.json').read_text())['receipts']
    entries = {c['id']: c for c in registry['candidates']}
    prior_path = root/'results/extensions/phenotype_followups_20261007/cellular-fitness/acquisition.json'
    prior = json.loads(prior_path.read_text()).get('receipts', []) if prior_path.exists() and prior_path.parent != out else []
    fetch = Acquisition(cache, out, prior)
    references, sets, gates = {}, [], []
    for gene in GENES:
        candidate = entries['mavedb_'+gene]
        meta_path = discovery/'catalog_candidates'/f'{gene}_metadata.json'
        payload = meta_path.read_bytes()
        receipt = next(r for r in original_receipts if r.get('path') == f'catalog_candidates/{gene}_metadata.json')
        if digest(payload) != receipt['sha256']:
            raise ValueError(f'discovery metadata hash mismatch: {gene}')
        metadata = json.loads(payload)
        (cache/f'{gene}_metadata.json').write_bytes(payload)
        targets = metadata.get('targetGenes', [])
        target = targets[0] if len(targets)==1 else {}
        accession = target.get('targetAccession', {}).get('accession')
        if gene == 'tinf2' and 'NM_001099274.3' in metadata.get('experiment',{}).get('shortDescription',''):
            accession = 'NM_001099274.3'  # Explicit source-declared RefSeq, not a chosen isoform.
        reference = None
        ref_error = None
        if accession and accession.startswith('NM_'):
            name = f'{gene}_reference.xml'
            url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id='+accession+'&rettype=gb&retmode=xml'
            data = fetch.get(url, name, 'NCBI public RefSeq; source attribution', maximum=1024**2) if acquire else (cache/name).read_bytes() if (cache/name).exists() else None
            if data:
                try:
                    reference = refseq_reference(data.decode(), accession)
                except Exception as error:
                    ref_error = str(error)
        elif accession and accession.startswith('ENST'):
            try:
                base, version = accession.rsplit('.',1)
                name = f'{gene}_reference_lookup.json'
                url = f'https://rest.ensembl.org/lookup/id/{base}?expand=1;content-type=application/json'
                data = fetch.get(url,name,'Ensembl public reference; source attribution',maximum=1024**2) if acquire else (cache/name).read_bytes() if (cache/name).exists() else None
                if not data:
                    raise ValueError('exact transcript lookup unavailable')
                lookup = json.loads(data)
                if lookup.get('id') != base or str(lookup.get('version')) != version:
                    raise ValueError('Ensembl source transcript version mismatch; no current-isoform substitution')
                assembly = target.get('targetAccession',{}).get('assembly')
                if assembly and lookup.get('assembly_name') != assembly:
                    raise ValueError('source assembly mismatch')
                sequences = {}
                for kind in ('protein','cds'):
                    name = f'{gene}_reference_{kind}.json'
                    url = f'https://rest.ensembl.org/sequence/id/{base}?type={kind};content-type=application/json'
                    data = fetch.get(url,name,'Ensembl public reference; source attribution',maximum=1024**2) if acquire else (cache/name).read_bytes() if (cache/name).exists() else None
                    if not data:
                        raise ValueError('source-bound protein/CDS unavailable')
                    obj = json.loads(data)
                    if obj.get('query') != base or not obj.get('seq'):
                        raise ValueError('source sequence query binding missing')
                    sequences[kind] = obj['seq']
                protein,dna = sequences['protein'], sequences['cds']
                if set(protein)-set(AA20) or translate(dna) not in (protein,protein+'*'):
                    raise ValueError('explicit Ensembl CDS/protein translation mismatch')
                translation = lookup.get('Translation',{})
                reference = dict(accession=accession, protein=protein, cds=dna, assembly=lookup.get('assembly_name'),
                    protein_accession=translation.get('id','')+'.'+str(translation.get('version','')),
                    identity_binding='exact source version checked by Ensembl transcript lookup; explicit type=cds and type=protein agree')
            except Exception as error:
                ref_error = str(error)
        if reference:
            if gene == 'tinf2':
                source_dna = target.get('targetSequence',{}).get('sequence')
                reference['source_n_reference_is_exact_CDS'] = source_dna == reference.get('cds')
            references[gene] = reference
        gates.append(dict(gene=gene.upper(), target_binding=target, reference_error=ref_error,
                          identity_ready=bool(reference), missing_mapping_gate=None if reference else 'source-bound full WT/CDS acquisition or annotation required',
                          source_provenance='linked_primary_study' if candidate['release']['source_publications'] else 'unknown_primary_study_not_excluded_for_recency',
                          original_metadata_receipt=receipt, selected_score_set=metadata['urn']))
        source_sets = [metadata]
        if gene == 'ddx3x':
            source_sets = []
            for urn in metadata.get('metaAnalyzesScoreSetUrns', []):
                name = urn.split(':')[-1]
                url = 'https://api.mavedb.org/api/v1/score-sets/'+urn
                data = fetch.get(url, name+'_metadata.json', candidate['access']['license']) if acquire else (cache/(name+'_metadata.json')).read_bytes() if (cache/(name+'_metadata.json')).exists() else None
                if data:
                    exon = json.loads(data)
                    # Explicit meta-analysis link identifies original exon maps, but labels still require methods.
                    if label_columns(exon, gene):
                        source_sets.append(exon)
            gates[-1]['classifier_rejected'] = True
            gates[-1]['original_exon_maps_measured'] = len(source_sets)
        for source in source_sets:
            urn = source['urn']
            name = urn.split(':')[-1]+'_scores.csv'
            url = 'https://api.mavedb.org/api/v1/score-sets/'+urn+'/scores/'
            local = discovery/'catalog_candidates'/f'{gene}_scores.csv'
            reuse = gene != 'ddx3x' and local.exists()
            if reuse:
                data = local.read_bytes()
                old = next(r for r in original_receipts if r.get('path') == f'catalog_candidates/{gene}_scores.csv')
                if digest(data) != old['sha256']:
                    raise ValueError('discovery score hash mismatch')
                (cache/name).write_bytes(data)
                fetch.receipts.append(dict(**old, status='reused_verified_discovery', response_count=0, license=candidate['access']['license']))
            else:
                data = fetch.get(url, name, source.get('license')) if acquire else (cache/name).read_bytes() if (cache/name).exists() else None
            count_data = None
            if source.get('datasetColumns',{}).get('countColumns'):
                count_name = urn.split(':')[-1]+'_counts.csv'
                count_url = 'https://api.mavedb.org/api/v1/score-sets/'+urn+'/counts/'
                count_data = fetch.get(count_url,count_name,source.get('license')) if acquire else (cache/count_name).read_bytes() if (cache/count_name).exists() else None
            sets.append((gene, source, data, reference, count_data))
    dump(out/'reference-bindings.json', references)
    observations, states, summaries, rejected = [], {}, [], []
    for gene, metadata, payload, reference, count_payload in sets:
        reasons = Counter()
        records = list(csv.DictReader(io.StringIO(payload.decode()))) if payload else []
        count_records = list(csv.DictReader(io.StringIO(count_payload.decode()))) if count_payload else []
        count_index = {r['accession']: r for r in count_records}
        if len(count_index) != len(count_records):
            raise ValueError('nonunique source count record accession')
        endpoints = label_columns(metadata, gene)
        validated_records, valid_states, finite_records = 0, set(), 0
        for row in records:
            state, reason = validate_row(row, reference)
            reasons[reason] += 1
            if not state:
                rejected.append(dict(gene=gene.upper(), score_set=metadata['urn'], record=row.get('accession'), hgvs_nt=row.get('hgvs_nt'), reason=reason))
                continue
            validated_records += 1
            valid_states.add(state['mutation'])
            labels = [(endpoint, finite(row.get(endpoint))) for endpoint in endpoints]
            labels = [(name, value) for name, value in labels if value is not None]
            if labels:
                finite_records += 1
            protein = reference['protein']
            for sequence, role in ((protein, 'WT'), (state['mutant_sequence'], 'single_missense')):
                h = digest(sequence.encode())
                states[h] = dict(sequence_sha256=h, sequence=sequence, gene=gene.upper(), role=role,
                                 source_transcript=reference['accession'], likelihood_coverage='unverified_until_exact_state_join')
            count_row = count_index.get(row.get('accession'),{})
            if count_row and any(count_row.get(k) != row.get(k) for k in ('hgvs_nt','hgvs_splice','hgvs_pro')):
                raise ValueError('source score/count HGVS join mismatch')
            replicate_counts = {k:v for k,v in count_row.items() if k in metadata.get('datasetColumns',{}).get('countColumns',[])}
            for endpoint, label in labels:
                retained = {k:v for k,v in row.items() if k in ('standard_error','95_ci_upper','95_ci_lower','variant_qc_flag') or k.startswith(('SE_','tHDR_', 'rLD2_', 'cLFC')) or k.endswith('BH_FDR')}
                observations.append(dict(gene=gene.upper(), score_set=metadata['urn'], record=row.get('accession'),
                    hgvs_nt=row.get('hgvs_nt',''), hgvs_pro=row.get('hgvs_pro',''),
                    mutation=state['mutation'], wt_sha256=state['wt_sha256'], mutant_sha256=state['mutant_sha256'],
                    endpoint=endpoint, label=label, uncertainty_QC_replicate_columns=retained, source_replicate_counts=replicate_counts,
                    rna_diagnostics={k:v for k,v in row.items() if 'rna' in k.lower()},
                    rna_use='diagnostic / explicitly label-assisted sensitivity only'))
        summaries.append(dict(gene=gene.upper(), score_set=metadata['urn'], version=metadata.get('modificationDate'),
            published=metadata.get('publishedDate'), source_registered_records=metadata.get('numVariants'), downloaded_records=len(records),
            validated_variant_records=validated_records, validated_single_states=len(valid_states), finite_measured_variant_records=finite_records,
            finite_measured_labels=sum(r['score_set']==metadata['urn'] for r in observations), reasons=dict(reasons),
            identity_ready=validated_records>0, label_ready=finite_records>0, sourcequality_ready=False,
            sourcequality_gate='replicate/QC/condition adjudication pending; primary-study provenance separate',
            endpoints=endpoints, exported_columns=list(records[0]) if records else [],
            method=metadata.get('methodText'), experiment_method=metadata.get('experiment',{}).get('methodText'),
            raw_replicates_independent_count=2 if gene=='vhl' else None,
            endpoint_count=len(endpoints), condition_count=3 if gene=='vhl' else 1,
            condition_count_scope='reported cell/background conditions; timepoint endpoints and exon tiles are not independent families',
            acquired_count_records=len(count_records), replicate_count_columns=metadata.get('datasetColumns',{}).get('countColumns',[]),
            license=metadata.get('license'), original_exon_source_links=metadata.get('primaryPublicationIdentifiers',[])))
    for name, data in [('observations.jsonl', observations), ('rejected-records.jsonl', rejected)]:
        with (out/name).open('w') as handle:
            for row in data:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
    dump(out/'state-inventory.json', dict(states=list(states.values()), policy='validated protein states only; no state yield for unmapped variants; no model inference'))
    dump(out/'candidate-gates.json', gates)
    dump(out/'score-set-support.json', summaries)
    dump(out/'codon-dependency-diagnostics.json', duplicate_diagnostics(observations))
    anchor = root/'data/proteingym_raw/DMS_substitutions.csv'
    anchors = list(csv.DictReader(anchor.open(newline='')))
    overlap = [dict(gene=g.upper(), wt_sha256=digest(r['protein'].encode()),
                    matches=[dict(assay=a['DMS_id'], relation='exact' if r['protein']==a['target_seq'] else 'candidate_contained_in_anchor' if r['protein'] in a['target_seq'] else 'anchor_contained_in_candidate')
                             for a in anchors if r['protein'] in a['target_seq'] or a['target_seq'] in r['protein']]) for g,r in references.items()]
    dump(out/'anchor-overlap.json', dict(canonical_reference=str(anchor.relative_to(root)), sha256=digest(anchor.read_bytes()), proteins=overlap,
        scope='exact/containment only; old217 registry, retained201 membership and homology remain separate gates'))
    counts = dict(candidate_genes=len(GENES), selected_registered_records=sum(entries['mavedb_'+g]['counts']['registered_variant_records'] for g in GENES),
        acquired_score_sets=sum(s['downloaded_records']>0 for s in summaries), downloaded_variant_records=sum(s['downloaded_records'] for s in summaries),
        source_bound_full_WT_genes=len(references), validated_proteins=len({r['wt_sha256'] for r in observations}),
        validated_variant_records=sum(s['validated_variant_records'] for s in summaries),
        finite_measured_variant_records=sum(s['finite_measured_variant_records'] for s in summaries),
        distinct_labelled_single_states=len({(r['wt_sha256'],r['mutation']) for r in observations}),
        native_state_inventory=len(states), finite_measured_label_records=len(observations),
        labelled_genes=len({r['gene'] for r in observations}), sourcequality_ready_genes=0,
        candidate_known_primary_studies=len({p['doi'] for g in GENES for p in entries['mavedb_'+g]['release']['source_publications'] if p.get('doi')}),
        labelled_known_primary_studies=len({p['doi'] for g in GENES if g.upper() in {r['gene'] for r in observations} for p in entries['mavedb_'+g]['release']['source_publications'] if p.get('doi')}),
        candidate_unknown_primary_study_genes=sum(not entries['mavedb_'+g]['release']['source_publications'] for g in GENES),
        labelled_unknown_primary_study_genes=sum(not entries['mavedb_'+g]['release']['source_publications'] for g in GENES if g.upper() in {r['gene'] for r in observations}),
        final_independent_families=None, transfer_bytes=fetch.used,
        codon_dependency_groups=len(duplicate_diagnostics(observations)))
    dump(out/'receipt.json', dict(counts=counts, code_sha256=digest(Path(__file__).read_bytes()),
        dependencies=dict(known_family_links=[['RAD51C','RAD51D','XRCC2']], additional_homologs='DDX3Y/DDX helicases, CTCFL, NONO/PSPC1 and BRCA1/BARD1 domain relationships require established homology closure; not new independent-family assignments',
                          shared_study_link=['RAD51D','XRCC2'], final_homology_groups='pending; no new arbitrary threshold algorithm'),
        likelihood_coverage='unverified until exact join', main_text_eligibility=False,
        scientific_boundary='Endogenous growth integrates protein and RNA effects; a coding missense label is not necessarily a protein-only mechanism.',
        metric_plan='Within assay/condition intracellular-fitness rank with ties and assay-normalized MSE; training-only calibration, withheld-family folds, mutation identity/chemistry/profile/context controls; no arbitrary absolute physical error. Keep same-AA SNVs together and retain discordance; RNA is a named label-assisted sensitivity.',
        no_execution='No fits, model inference, GPU, H200 or model downloads'))
    return counts
