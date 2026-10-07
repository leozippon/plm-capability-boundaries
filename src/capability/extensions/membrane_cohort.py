"""Exact identity and row-preserving measurement qualification; no inference or fits."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..core.amino_acids import AA20
from ..generation.structure_inputs import THREE_TO_ONE

DISCOVERY = Path('results/extensions/phenotype_followups_20261007/discovery')
MISSING = {'', 'NA', 'NaN', 'nan', 'null'}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_rows(path: Path, rows) -> None:
    with path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')


def table(path: Path, required=(), delimiter=',') -> list[dict]:
    """Never silently accept extra/missing cells or repeated headers."""
    with path.open(newline='') as handle:
        reader = csv.reader(handle, delimiter=delimiter, strict=True)
        header = next(reader)
        if len(set(header)) != len(header) or not set(required) <= set(header):
            raise ValueError(f'{path.name}: invalid header')
        rows = []
        for number, cells in enumerate(reader, 2):
            if len(cells) != len(header):
                raise ValueError(f'{path.name}:{number}: source row width mismatch')
            rows.append(dict(zip(header, cells)))
    return rows


def number(value: str) -> tuple[float | None, str]:
    if value in MISSING:
        return None, 'missing'
    if value.startswith(('<', '>')):
        return None, 'censored_bound'
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f'invalid measured value: {value!r}') from error
    return (parsed, 'finite') if math.isfinite(parsed) else (None, 'nonfinite')


def parse_change(hgvs: str) -> tuple[str, int, str] | None:
    match = re.fullmatch(r'p\.([A-Z][a-z]{2})([1-9][0-9]*)([A-Z][a-z]{2}|\*|=)', hgvs)
    if not match:
        return None
    before, pos, after = match.groups()
    wt = THREE_TO_ONE.get(before.upper())
    mt = THREE_TO_ONE.get(after.upper()) if after not in ('Ter', '*', '=') else after
    if wt is None or wt not in AA20 or mt is None:
        return None
    return wt, int(pos), mt


def bind_state(wt: str | None, hgvs: str, *, consequence: str = '') -> tuple[dict | None, str]:
    change = parse_change(hgvs)
    if change is None:
        return None, 'not_single_protein_substitution'
    before, pos, after = change
    if after in ('Ter', '*'):
        return None, 'stop'
    if after == '=' or before == after:
        return None, 'synonymous'
    if consequence.lower() == 'start-loss' or pos == 1:
        return None, 'start_loss_or_initiator_substitution'
    if wt is None:
        return None, 'source_full_protein_WT_unavailable_no_DNA_frame_guess'
    if not wt or any(aa not in AA20 for aa in wt):
        raise ValueError('noncanonical source WT')
    if not 1 <= pos <= len(wt) or wt[pos - 1] != before:
        return None, 'WT_residue_or_numbering_mismatch'
    mutant = wt[:pos - 1] + after + wt[pos:]
    return dict(mutation=f'{before}{pos}{after}', position=pos, wt_aa=before,
                mutant_aa=after, wt_sha256=sha(wt), mutant_sha256=sha(mutant),
                state_id=sha(wt) + ':' + sha(mutant), mutated_sequence=mutant), 'validated'


def dependencies(rows: list[dict]) -> None:
    """Retain multiple nucleotide observations and ties, never average them."""
    grouped = defaultdict(list)
    for row in rows:
        if row['state_id']:
            grouped[row['state_id']].append(row)
    for group in grouped.values():
        for row in group:
            row['dependency_group_size'] = len(group)
            row['dependency_group'] = row['state_id']
            row['dependency_kind'] = 'same_protein_state_not_independent' if len(group) > 1 else 'single_observation'


def intersections(rows: list[dict], channels: list[str]) -> list[dict]:
    """Same full-WT AND mutant hashes, not variant strings or normalized ranks."""
    sets = {channel: {r['state_id'] for r in rows if r['channel'] == channel and r['accepted']} for channel in channels}
    return [dict(channels=list(combo), state_ids=sorted(set.intersection(*(sets[c] for c in combo))),
                 n_states=len(set.intersection(*(sets[c] for c in combo))))
            for size in range(2, len(channels) + 1) for combo in itertools.combinations(channels, size)]


def anchor_matches(wt: str, anchors: dict[str, str]) -> dict:
    return dict(exact=[key for key, seq in anchors.items() if seq == wt],
                anchor_contains_panel=[key for key, seq in anchors.items() if seq != wt and wt in seq],
                panel_contains_anchor=[key for key, seq in anchors.items() if seq != wt and seq in wt],
                homology_exclusion='NOT_ESTABLISHED')


def sgca_prefix(path: Path) -> tuple[list[dict], dict]:
    """Quarantine tail from embedded alleles onward, even when widths match."""
    lines = path.read_text().splitlines()
    header = next(csv.reader([lines[0]], strict=True))
    boundary = header.index('alleles')
    kept = ['accession', 'hgvs_nt', 'hgvs_splice', 'hgvs_pro', 'score', 'Variant ID',
            'site', 'gene', 'codon', 'WT_nt', 'Variant_nt', 'site variant', 'WT_AA',
            'Variant_AA', 'codon variant', 'protein change', 'splicing-region variant',
            'Block', 'MLS1', 'MLS2', 'MLS3', 'sigma', 'confidence_3.0syn', 'locus', 'variant_key']
    if boundary != 27 or not set(kept) <= set(header[:boundary]):
        raise ValueError('SGCA prefix schema changed; no safe salvage')
    rows, malformed = [], 0
    widths = Counter()
    for line_number, line in enumerate(lines[1:], 2):
        # Current prefix is entirely unquoted; reject unproven prefix changes.
        pieces = line.split(',', boundary)
        if len(pieces) != boundary + 1 or any('"' in value for value in pieces[:boundary]):
            raise ValueError(f'SGCA line {line_number}: unsafe core prefix')
        core = dict(zip(header[:boundary], pieces[:boundary]))
        cells = next(csv.reader([line], strict=True))
        widths[len(cells)] += 1
        try:
            alleles = json.loads(cells[boundary])
            good = isinstance(alleles, list) and len(alleles) == 2 and all(isinstance(x, str) for x in alleles)
        except (ValueError, IndexError):
            good = False
        malformed += not good
        row = {key: core[key] for key in kept}
        row['source_consequence'] = core['classification']  # variant consequence only, not a clinical predictor
        row['source_line_sha256'] = sha(line)
        rows.append(row)
    return rows, dict(header_columns=len(header), observed_row_widths=dict(widths),
                      malformed_alleles_rows=malformed, trusted_prefix_columns=kept,
                      quarantined_tail='alleles and all following fields, including read counts and predictors',
                      gate='verified_unquoted_measurement_prefix_admitted; tail_not_recovered')


# NCBI standard genetic code (table 1), in T/C/A/G codon order.
STANDARD_CODE = dict(zip((''.join(c) for c in itertools.product('TCAG', repeat=3)),
    'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'))


def sgca_coding_wt(meta: dict, records: list[dict]) -> tuple[str, dict]:
    """Validate the declared coding target against every annotated SNV; no ORF search."""
    target = meta['targetGenes'][0]
    sequence = target['targetSequence']
    if (target['name'] != 'SGCA' or target['category'] != 'protein_coding'
            or sequence['sequenceType'] != 'dna'
            or sequence['taxonomy']['code'] != 9606
            or 'all possible coding single-nucleotide variants' not in meta['methodText']):
        raise ValueError('SGCA declared human coding target required')
    dna = sequence['sequence']
    if not dna or len(dna) % 3 or any(nt not in 'ACGT' for nt in dna):
        raise ValueError('SGCA invalid coding frame/DNA')
    wt = ''.join(STANDARD_CODE[dna[i:i + 3]] for i in range(0, len(dna), 3))
    if wt[0] != 'M' or any(aa not in AA20 for aa in wt):
        raise ValueError('SGCA coding target must start Met with no stops; terminal stop not required')
    observed = set()
    consequence_counts = Counter()
    for i, row in enumerate(records, 1):
        match = re.fullmatch(r'c\.([1-9][0-9]*)([ACGT])>([ACGT])', row['hgvs_nt'])
        if not match:
            raise ValueError(f'SGCA row {i}: invalid nucleotide HGVS')
        site_text, before, after = match.groups()
        site = int(site_text)
        codon = (site - 1) // 3 + 1
        if (not 1 <= site <= len(dna) or before == after
                or row['site'] != str(site) or row['codon'] != str(codon)
                or row['WT_nt'] != before or row['Variant_nt'] != after
                or dna[site - 1] != before or row['gene'] != 'SGCA'):
            raise ValueError(f'SGCA row {i}: nucleotide/site/codon/reference conflict')
        triplet = dna[3 * (codon - 1):3 * codon]
        offset = (site - 1) % 3
        mutant_aa = STANDARD_CODE[triplet[:offset] + after + triplet[offset + 1:]]
        before_aa = wt[codon - 1]
        if row['WT_AA'] != before_aa or row['Variant_AA'] != mutant_aa:
            raise ValueError(f'SGCA row {i}: translated amino acid conflict')
        expected = ('Start-loss' if codon == 1 else 'Nonsense' if mutant_aa == '*'
                    else 'Synonymous' if before_aa == mutant_aa else 'Missense')
        change = parse_change(row['hgvs_pro'])
        expected_after = '=' if before_aa == mutant_aa else 'Ter' if mutant_aa == '*' else mutant_aa
        if change != (before_aa, codon, expected_after) or row['source_consequence'] != expected:
            raise ValueError(f'SGCA row {i}: protein HGVS/consequence conflict')
        # Repeated observations stay as rows; coverage is an identity set, not an averaging gate.
        observed.add((site, after))
        consequence_counts[expected] += 1
        for column in ('score', 'MLS1', 'MLS2', 'MLS3', 'sigma'):
            value, status = number(row[column])
            if status != 'finite' or (column == 'sigma' and value is not None and value < 0):
                raise ValueError(f'SGCA row {i}: invalid measured {column}')
    expected_snvs = {(i, nt) for i, ref in enumerate(dna, 1) for nt in 'ACGT' if nt != ref}
    if observed != expected_snvs:
        raise ValueError('SGCA incomplete coding SNV coverage')
    return wt, dict(genetic_code='NCBI standard table 1', coding_frame='declared coding target, first nucleotide; no ORF search',
                   dna_sha256=sha(dna), coding_nt=len(dna), protein_aa=len(wt),
                   nucleotide_positions_covered=len(dna), codons_covered=len(wt),
                   source_SNV_rows=len(records), source_consequences=dict(consequence_counts),
                   crosscheck_conflicts=0, terminal_stop_required=False,
                   crosschecks=['complete three-alternative SNV coverage at every coding nucleotide',
                                'HGVS nucleotide/reference/site/codon', 'translated WT and mutant amino acids',
                                'protein HGVS and source consequence', 'finite score/MLS1/MLS2/MLS3 and nonnegative finite sigma'])


def prepare(root: Path, out: Path) -> dict:
    discovery = root / DISCOVERY
    receipts = json.loads((discovery / 'consolidated_acquisition_manifest.json').read_text())['receipts']
    # Required discovery documents are hashed as input evidence, never changed.
    sources = []
    for name in ('evidence-report.md', 'candidate_registry.json', 'downloaded_label_inspection.json', 'consolidated_acquisition_manifest.json'):
        path = discovery / name
        sources.append(dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), role='discovery_evidence'))

    def source(name: str, license: str, version: str) -> dict:
        path = discovery / name
        matching = [r for r in receipts if r.get('path') == name and 'sha256' in r]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if len(matching) != 1 or matching[0]['sha256'] != digest:
            raise ValueError(f'source receipt mismatch: {name}')
        record = dict(matching[0], license=license, version=version)
        sources.append(record)
        return record

    all_rows, channels, proteins, state_manifest = [], [], {}, {}

    def add(protein, channel, original, hgvs, score_col, se_col, src, wt, quality,
            source_qc_reason='', consequence='', identity=None):
        state, reason = bind_state(wt, hgvs, consequence=consequence) if identity is None else identity
        score, score_status = number(original[score_col])
        se, se_status = number(original[se_col])
        if se is not None and se < 0:
            raise ValueError('negative source uncertainty')
        reasons = [] if state else [reason]
        if score_status != 'finite':
            reasons.append('score_' + score_status)
        if se_status != 'finite':
            reasons.append('uncertainty_' + se_status)
        if source_qc_reason:
            reasons.append(source_qc_reason)
        state_id = state['state_id'] if state else None
        row = dict(protein=protein, channel=channel, source_row_id=f'{channel}:{quality["source_row_number"]}',
                   original=original, hgvs_pro=hgvs, state_id=state_id,
                   mutation=state['mutation'] if state else None, identity_status=reason,
                   score=score, score_status=score_status, uncertainty=se, uncertainty_status=se_status,
                   accepted=not reasons, exclusions=reasons, quality=quality,
                   source_sha256=src['sha256'], source_url=src['url'],
                   source_version=src['version'], license=src['license'],
                   censoring='bound' if score_status == 'censored_bound' else 'not_reported_in_export',
                   dependency_group_size=0, model_score_coverage='UNKNOWN_until_native_state_context_join',
                   model_context_coverage='UNKNOWN_until_native_state_context_join')
        all_rows.append(row)
        if state:
            state_manifest[state_id] = dict(protein=protein, **state)
        return row

    # F9 protein sequence is explicit metadata; no translation/reconstruction.
    f9 = None
    for letter in 'abcde':
        suffix = f'{letter}-1'
        metadata_name = f'multistep/{suffix}_metadata.json'
        meta = json.loads((discovery / metadata_name).read_text())
        source(metadata_name, 'CC0', meta['modificationDate'])
        target = meta['targetGenes'][0]['targetSequence']
        if target['sequenceType'] != 'protein' or len(target['sequence']) != 461:
            raise ValueError('F9 full protein WT gate')
        wt = target['sequence']
        if f9 is not None and wt != f9:
            raise ValueError('F9 channels have different protein WT')
        f9 = wt
        src = source(f'multistep/{suffix}_scores.csv', 'CC0', meta['urn'] + '@' + meta['modificationDate'])
        channel = 'F9_' + suffix
        records = table(discovery / src['path'], ['hgvs_pro', 'score', 'SE_score', 'N_replicates'])
        channels.append(dict(channel=channel, protein='F9', endpoint=meta['abstractText'], units='assay_normalized_dimensionless',
                             direction='higher antibody-detected tethered secretion or Gla PTM signal',
                             normalization='WT barcode median 1; lowest 5th-percentile missense median 0; weighted FACS bins',
                             quality_definition=meta['methodText'], assay_context=meta['experiment']['methodText'],
                             RNA_context='DNA barcode sequencing; no endogenous RNA phenotype',
                             uncertainty='SE_score; SD/sqrt(N_replicates)', source=src))
        for i, record in enumerate(records, 1):
            n, status = number(record['N_replicates'])
            if status != 'finite' or n is None or n < 0 or int(n) != n:
                raise ValueError('F9 invalid N_replicates')
            quality = dict(source_row_number=i, N_replicates=int(n),
                           tile_biological_replicates={k: v for k, v in record.items() if k.startswith('Score_Tile')},
                           below_frequency_threshold='not independently available; source filtering applied')
            add('F9', channel, record, record['hgvs_pro'], 'score', 'SE_score', src, wt, quality,
                'source_fewer_than_two_replicates' if n < 2 else '')
    assert f9 is not None
    proteins['F9'] = dict(wt=f9, wt_sha256=sha(f9), length=len(f9), source='multistep/*_metadata.json targetSequence protein',
                          input_context='full native precursor; engineered linker/tag/CD28 fusion is assay context, not native sequence')

    # Exact source residue map is contiguous and covers the complete 348-aa RHO.
    wt_src = source('rho_wt_aa.tsv', 'CC BY 4.0 Zenodo; GitHub lacks LICENSE', 'discovery snapshot 2026-10-07')
    residues = table(discovery / 'rho_wt_aa.tsv', ['wt_aa', 'pos'], delimiter='\t')
    if [int(r['pos']) for r in residues] != list(range(1, 349)) or any(r['wt_aa'] not in AA20 for r in residues):
        raise ValueError('RHO incomplete or noncanonical WT map')
    rho = ''.join(r['wt_aa'] for r in residues)
    src = source('rho_supplementary_table1.csv', 'CC BY 4.0 Zenodo; GitHub lacks LICENSE', 'discovery snapshot 2026-10-07')
    records = table(discovery / src['path'], ['protein', 'HGVSP', 'MEE_mean', 'MEE_se', 'Octant_mean', 'Octant_se'])
    for label, prefix, endpoint, replicates in (
            ('RHO_method1', 'MEE', 'surface-antibody FACS steady-state surface abundance; N-terminal residues 1-10 epitope', 2),
            ('RHO_method2', 'Octant', 'membrane-proximity protease/transcription reporter; RNA barcode expression, not endogenous RHO RNA abundance', 8)):
        channels.append(dict(channel=label, protein='RHO', endpoint=endpoint, units='assay_normalized_dimensionless',
                             direction='higher surface-antibody or membrane-proximity reporter signal',
                             normalization='NBGLMM log2 effect transformed to nonsense 0 and WT/synonymous 1 scale',
                             study_replicates=replicates, per_variant_replicates='not reported in export',
                             quality_definition='author Method 1 keeps flag OK and excludes positions 2-10 before export; no new effect-based filtering',
                             assay_context='HEK293T; Method 1 lentiviral integration; Method 2 RHO transcription-factor fusion landing-pad integration',
                             RNA_context='Method 2 RNA barcode is reporter readout, not native RHO RNA phenotype',
                             uncertainty=prefix + '_se', source=src))
        for i, record in enumerate(records, 1):
            hgvs = record['HGVSP']
            identity = bind_state(rho, hgvs)
            if identity[0] and identity[0]['mutation'] != record['protein']:
                raise ValueError('RHO author protein/HGVS mismatch')
            both = bool(record['MEE_mean'] and record['Octant_mean']) and record['consequence'] == 'missense'
            original = {k: record[k] for k in ('protein', 'consequence', 'HGVSP', 'MEE_mean', 'MEE_se', 'MEE_adj.p', 'Octant_mean', 'Octant_se', 'Octant_adj.p')}
            # Composite values are never used as endpoints or a pathogenicity composite.
            quality = dict(source_row_number=i, study_replicates=replicates, per_variant_replicates=None,
                           source_method_discordance=(not bool(record['composite_score_homog'])) if both else None,
                           discordance_definition='both method scores exist but author composite_score_homog is absent; author gate I2<93 AND absolute difference<0.5',
                           source_method1_epitope_excluded=bool(identity[0] and 2 <= identity[0]['position'] <= 10))
            add('RHO', label, original, hgvs, prefix + '_mean', prefix + '_se', src, rho, quality, identity=identity)
    proteins['RHO'] = dict(wt=rho, wt_sha256=sha(rho), length=len(rho), source=wt_src,
                           input_context='full native RHO; reporter fusion/protease context kept separate')

    meta = json.loads((discovery / 'catalog_candidates/sgca_metadata.json').read_text())
    sgca_meta_src = source('catalog_candidates/sgca_metadata.json', 'CC BY 4.0', meta['modificationDate'])
    src = source('catalog_candidates/sgca_scores.csv', 'CC BY 4.0', meta['urn'] + '@' + meta['modificationDate'])
    records, parse_qc = sgca_prefix(discovery / src['path'])
    dump(out / 'sgca-source-parse-qc.json', parse_qc)
    sgca, coding_qc = sgca_coding_wt(meta, records)
    dump(out / 'sgca-coding-validation.json', coding_qc)
    proteins['SGCA'] = dict(wt=sgca, wt_sha256=sha(sgca), length=len(sgca), source=sgca_meta_src,
                            identity_provenance=coding_qc,
                            input_context='source-defined human SGCA coding target; standard-code translation anchored by all source SNV annotations',
                            full_FLAG_construct_context='UNKNOWN; tag/insertion/linker sequence not supplied; separate from native target identity')
    channels.append(dict(channel='SGCA_surface', protein='SGCA', endpoint='extracellular FLAG surface staining in HEK-BDG cells expressing beta/delta/gamma-sarcoglycan',
                         units='assay-normalized scale; exact score normalization undocumented in supplied metadata',
                         direction='higher surface localization; not clinical pathogenicity',
                         uncertainty='sigma (not established as SE)', replicates=['MLS1', 'MLS2', 'MLS3'],
                         assay_context=meta['methodText'], RNA_context='construct SNVs; no native RNA channel',
                         quality_definition='confidence_3.0syn retained descriptively, not used to select effect magnitude',
                         limitations=['sigma is not established as SE', 'exact normalization incompletely documented',
                                      'full FLAG construct context unknown', 'manuscript in preparation',
                                      'malformed alleles and following predictors/read counts quarantined; not recovered'], source=src))
    for i, record in enumerate(records, 1):
        add('SGCA', 'SGCA_surface', record, record['hgvs_pro'], 'score', 'sigma', src, sgca,
            dict(source_row_number=i, MLS={key: record[key] for key in ('MLS1', 'MLS2', 'MLS3')},
                 confidence=record['confidence_3.0syn'], tail_parse='quarantined'),
            consequence=record['source_consequence'])

    sgca_groups = Counter(r['hgvs_pro'] for r in all_rows if r['protein'] == 'SGCA')
    for row in all_rows:
        if row['protein'] == 'SGCA':
            row['candidate_HGVS_dependency_group_size'] = sgca_groups[row['hgvs_pro']]
            row['candidate_HGVS_dependency_status'] = 'validated_protein_state; dependent SNVs retained separately' if row['state_id'] else 'excluded_nonmissense_or_initiator; SNVs retained separately'
    for channel in channels:
        subset = [r for r in all_rows if r['channel'] == channel['channel']]
        dependencies(subset)
    write_rows(out / 'channel-rows.jsonl', all_rows)
    write_rows(out / 'mapping-exclusions.jsonl', (r for r in all_rows if not r['accepted']))
    write_rows(out / 'canonical-mutant-states.jsonl', sorted(state_manifest.values(), key=lambda r: (r['protein'], r['mutation'])))
    dump(out / 'canonical-WT-manifest.json', proteins)
    dump(out / 'channel-manifest.json', channels)
    pairs = {protein: intersections(all_rows, [c['channel'] for c in channels if c['protein'] == protein]) for protein in proteins}
    dump(out / 'aligned-channel-intersections.json', pairs)

    # Compare against the actual local canonical 217 assay WT catalogue, not gene names.
    from ..models.fitness import wildtype_of
    anchor_dir = root / 'data/proteingym/DMS_ProteinGym_substitutions'
    anchor_files = sorted(anchor_dir.glob('*.csv'))
    if len(anchor_files) != 217:
        raise ValueError('expected canonical 217-assay anchor catalogue')
    anchors = {p.stem: wildtype_of(p.stem, anchor_dir) for p in anchor_files}
    dump(out / 'anchor-WT-manifest.json', dict(source=str(anchor_dir.relative_to(root)),
         qualification='existing wildtype_of first-row revert; canonical catalogue identity only, not native prediction coverage',
         WTs={key: dict(wt=wt, wt_sha256=sha(wt)) for key, wt in anchors.items()}))
    dump(out / 'anchor-comparison.json', {protein: anchor_matches(info['wt'], anchors) if info['wt'] else {'gate': 'WT_blocked'} for protein, info in proteins.items()})
    dump(out / 'source-manifest.json', sources)
    summary: dict[str, Any] = dict(unique_candidate_proteins=3, validated_WT_proteins=3, independent_families='UNKNOWN; at most 3 proteins, channels do not increase this',
                   model_score_coverage='UNKNOWN_until_native_state_context_join', channels={})
    for channel in channels:
        subset = [r for r in all_rows if r['channel'] == channel['channel']]
        accepted = [r for r in subset if r['accepted']]
        summary['channels'][channel['channel']] = dict(rows=len(subset), accepted_rows=len(accepted),
             accepted_states=len({r['state_id'] for r in accepted}), identity_validated_rows=sum(r['state_id'] is not None for r in subset),
             finite_scores=sum(r['score_status'] == 'finite' for r in subset),
             finite_identity_validated_scores=sum(r['state_id'] is not None and r['score_status'] == 'finite' for r in subset),
             discordant_both_method_rows=sum(bool(r['quality'].get('source_method_discordance')) for r in subset),
             exclusions=dict(Counter(reason for r in subset for reason in r['exclusions'])),
             tied_score_groups=sum(count > 1 for count in Counter(r['score'] for r in accepted).values()))
    candidate_sgca = []
    for record in records:
        change = parse_change(record['hgvs_pro'])
        if change is not None and change[2] in AA20 and change[0] != change[2] and number(record['score'])[0] is not None:
            candidate_sgca.append(record)
    summary['SGCA_pre_gate_candidates'] = dict(finite_missense_records=len(candidate_sgca), distinct_HGVS=len({r['hgvs_pro'] for r in candidate_sgca}),
                                              duplicate_HGVS_extra_rows=len(candidate_sgca)-len({r['hgvs_pro'] for r in candidate_sgca}))
    dump(out / 'qc-summary.json', summary)
    return summary
