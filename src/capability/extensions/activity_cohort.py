"""Public activity identity qualification only; source scores are not physical labels."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit, urlunsplit
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AA = frozenset('ACDEFGHIKLMNPQRSTVWY')
SUBSTITUTION = re.compile(r'([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])')
REVISION = '08a27d07b5764af58f99d26734d096ebe2a5f9cf'
BASE = 'https://huggingface.co/datasets/AI4Protein/VenusMutHub'
MAX_TOTAL = 250 * 1024**2
MAX_FILE = 8 * 1024**2
AS_OF = '2026-10-07'
SOURCE_HASHES = {
    'mutant__doi.csv': '899e958f659d2d7d55083969a1524594f032c11cea8e22013ac50b2a7afa4f1d',
    'mutant__dataset_summary.csv': 'ba893bcd7a39d6a5d367b7df89aa134a804473538dd7c15e66e01574cb07a2d0',
}


def seqhash(sequence: str) -> str:
    if not sequence or not set(sequence) <= AA:
        raise ValueError('empty or noncanonical sequence')
    return hashlib.sha256(sequence.encode('ascii')).hexdigest()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def jsonlines(path: Path, rows: list[dict]) -> None:
    with path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')


def reverse_substitution(mutant: str, sequence: str) -> tuple[str, dict]:
    """The row asserts a full mutant state; reverse exactly its one annotated site."""
    match = SUBSTITUTION.fullmatch(mutant)
    if not match:
        raise ValueError('not_one_canonical_substitution')
    wt, raw_position, mt = match.groups()
    position = int(raw_position)
    seqhash(sequence)
    if position > len(sequence):
        raise ValueError('position_outside_sequence')
    if sequence[position-1] != mt:
        raise ValueError('mutant_residue_mismatch')
    reference = sequence[:position-1] + wt + sequence[position:]
    if len(reference) != len(sequence):
        raise ValueError('length_mismatch')
    return reference, dict(wt_aa=wt, position=position, mt_aa=mt, identity_substitution=wt == mt)


def qualify_rows(assay: str, rows: list[dict[str, str]]) -> tuple[list[dict], dict]:
    records = []
    references = set()
    for ordinal, source in enumerate(rows, 1):
        record: dict[str, Any] = dict(assay=assay, source_record=ordinal,
            source_row_id=f'venus_activity:{assay}:{ordinal}', original=source,
            fitness_score_raw=source.get('fitness_score'), score_semantics='uninterpreted_source_score',
            identity_ready=False, finite_label=False, blockers=[], wt_sha256=None,
            mutant_sha256=None, variant_key=None, label_readiness='blocked_untraced_endpoint_scale_direction')
        try:
            if set(source) != {'mutant', 'mutated_sequence', 'fitness_score'} or any(v is None for v in source.values()):
                raise ValueError('CSV_schema_or_width_mismatch')
            reference, mutation = reverse_substitution(source['mutant'], source['mutated_sequence'])
            references.add(reference)
            record.update(wt_sequence=reference, wt_sha256=seqhash(reference),
                          mutant_sha256=seqhash(source['mutated_sequence']), mutation=mutation,
                          variant_key=seqhash(reference)+':'+seqhash(source['mutated_sequence']))
            record['identity_ready'] = True
        except ValueError as error:
            record['blockers'].append(str(error))
        try:
            score = float(source.get('fitness_score', ''))
            if not math.isfinite(score):
                raise ValueError('nonfinite_label')
            record.update(source_score=score, finite_label=True)
        except (ValueError, TypeError):
            record['blockers'].append('missing_malformed_or_nonfinite_label')
        records.append(record)
    # An invalid row can hide a competing background: conservatively block the
    # whole file, not just the offending row, rather than choosing a majority WT.
    consistent = bool(records) and len(references) == 1 and all(r['identity_ready'] for r in records)
    for record in records:
        if not consistent:
            record['identity_ready'] = False
            record['blockers'].append('file_WT_not_fully_consistent')
        record['admissible_identity_single'] = record['identity_ready'] and not record['mutation']['identity_substitution'] if record.get('mutation') else False
    variants = Counter(r['variant_key'] for r in records if r['identity_ready'])
    score_counts = Counter(r['source_score'] for r in records if r['admissible_identity_single'] and r['finite_label'])
    state_scores = defaultdict(set)
    for record in records:
        if record['admissible_identity_single'] and record['finite_label']:
            state_scores[record['variant_key']].add(record['source_score'])
    for record in records:
        record['same_assay_state_rows'] = variants.get(record['variant_key'], 0)
        record['source_score_tie_rows'] = score_counts.get(record.get('source_score'), 0)
        record['same_assay_state_discordant_labels'] = len(state_scores.get(record['variant_key'], set())) > 1
    accepted = [r for r in records if r['admissible_identity_single']]
    report = dict(assay=assay, source_rows=len(rows), identity_rows=len(accepted),
                  WT_consistent=consistent, unique_candidate_WTs=len(references),
                  wt_sha256=seqhash(next(iter(references))) if consistent else None,
                  WT_length=len(next(iter(references))) if consistent else None,
                  identity_substitution_rows=sum(r.get('mutation', {}).get('identity_substitution', False) for r in records),
                  finite_single_labels=sum(r['finite_label'] for r in accepted),
                  distinct_single_variants=len({r['variant_key'] for r in accepted}),
                  duplicate_single_state_groups=sum(v > 1 for k,v in variants.items() if any(r['variant_key'] == k for r in accepted)),
                  distinct_source_scores=len(score_counts), tied_score_groups=sum(v > 1 for v in score_counts.values()),
                  discordant_same_assay_variant_groups=sum(len(v) > 1 for v in state_scores.values()),
                  missing_label_rows=sum(not r['finite_label'] for r in records),
                  label_ready=False, physical_MSE_ready=False,
                  condition_identity='unresolved; file ID is an assay partition, not verified experimental condition',
                  descriptive_nonconstant_single_support=len({r['variant_key'] for r in accepted if r['finite_label']}) >= 5 and len(score_counts) > 1)
    return records, report


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['mutant', 'mutated_sequence', 'fitness_score']:
            raise ValueError('activity CSV header mismatch')
        return list(reader)


def fetch_immutable(url: str, target: Path, expected_size: int | None = None,
                    git_oid: str | None = None, timeout: int = 25) -> dict:
    """No auth/config reads; bounded public GET, cached bytes never overwritten."""
    started = time.monotonic()
    receipt: dict[str, Any] = dict(origin_url=url, path=str(target),
        revision=REVISION if REVISION in url and 'huggingface.co/' in url else None,
        as_of=AS_OF, status='failed', timeout_seconds=timeout, max_bytes=MAX_FILE,
        route='public_URL_urllib_no_auth', network_bytes=0)
    try:
        if target.exists():
            data = target.read_bytes()
            receipt['route'] = 'immutable_local_cache_verified_against_pinned_tree'
        else:
            request = urllib.request.Request(url, headers={'User-Agent': 'activity-cohort-qualification/1.0'})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                resolved = urlsplit(response.geturl())
                # Public servers can redirect to temporary signed object URLs;
                # retain the stable origin and redirect host/path, not signatures.
                receipt['resolved_host_path'] = urlunsplit((resolved.scheme, resolved.netloc, resolved.path, '', ''))
                data = response.read((expected_size if expected_size is not None else MAX_FILE) + 1)
                receipt['network_bytes'] = len(data)
                if time.monotonic() - started > 2*timeout:
                    raise TimeoutError('per-request wall bound exceeded')
        if len(data) > MAX_FILE or (expected_size is not None and len(data) != expected_size):
            raise ValueError('payload_size_disagrees_with_bound_or_pinned_tree')
        if git_oid and hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest() != git_oid:
            raise ValueError('payload_disagrees_with_pinned_git_blob_oid')
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(data)
        receipt.update(status='ready', bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                       git_blob_oid=git_oid, verified_at_utc=datetime.now(timezone.utc).isoformat())
        if receipt['network_bytes']:
            receipt['retrieved_at_utc'] = receipt['verified_at_utc']
    except Exception as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
    receipt['wall_seconds'] = time.monotonic()-started
    return receipt


def acquire_all(data: Path, out: Path, workers: int = 4) -> list[dict]:
    if not 1 <= workers <= 4:
        raise ValueError('workers must be 1–4')
    tree_path = data/'source/activity-tree.json'
    url = f'https://huggingface.co/api/datasets/AI4Protein/VenusMutHub/tree/{REVISION}/single_mutant/activity?limit=1000'
    tree_receipt = fetch_immutable(url, tree_path)
    dump(out/'tree-acquisition.json', tree_receipt)
    if tree_receipt['status'] != 'ready':
        raise ValueError('pinned activity tree acquisition failed')
    tree = json.loads(tree_path.read_text())
    files = [r for r in tree if r['type'] == 'file' and r['path'].endswith('.csv')]
    if len(files) != 130 or len({r['path'] for r in files}) != 130:
        raise ValueError('pinned tree does not establish exactly 130 unique activity CSV files')
    if any(not r['path'].startswith('single_mutant/activity/') or '/' in r['path'].removeprefix('single_mutant/activity/') for r in files):
        raise ValueError('unexpected source tree path')
    if sum(r['size']+1 for r in files) + MAX_FILE > MAX_TOTAL or any(not 0 < r['size'] <= MAX_FILE for r in files):
        raise ValueError('bounded acquisition exceeds 250 MiB total')
    def get(item: dict) -> dict:
        receipt = fetch_immutable(f'{BASE}/resolve/{REVISION}/{item["path"]}',
            data/'raw'/Path(item['path']).name, item['size'], item['oid'])
        receipt['source_path'] = item['path']
        return receipt
    with ThreadPoolExecutor(max_workers=workers) as pool:
        receipts = list(pool.map(get, sorted(files, key=lambda r:r['path'])))
    dump(out/'acquisition.json', dict(as_of=AS_OF, revision=REVISION, files=receipts,
         expected_files=len(files), expected_payload_bytes=sum(r['size'] for r in files),
         total_network_bytes=sum(r['network_bytes'] for r in receipts)+tree_receipt['network_bytes'],
         cap_bytes=MAX_TOTAL, max_workers=workers, failed_files=sum(r['status'] != 'ready' for r in receipts)))
    return receipts


def source_mapping(assay: str, doi_rows: list[dict], summary_rows: list[dict]) -> dict:
    matches = [r for r in doi_rows if r['mutant_file_id'] == assay]
    summaries = [r for r in summary_rows if r['dataset_id'] == assay and r['category'] == 'activity']
    raw = [r['doi'] for r in matches]
    # Preserve author '?' prefixes as uncertainty; do not silently repair a DOI.
    normalized = []
    for value in raw:
        match = re.search(r'10\.\d{4,9}/[^\s]+', value)
        if match and '?' not in value:
            normalized.append(match.group())
    return dict(source_doi_raw=raw, source_doi_candidates=sorted(set(normalized)),
                source_mapping_status='author_map_exact_ID_unverified_original_study' if len(matches) == 1 and len(normalized) == 1 else 'missing_ambiguous_or_author_uncertain',
                author_summary_WTs=[r['wt_sequence'] for r in summaries],
                author_summary_status='one_exact_dataset_ID' if len(summaries) == 1 else 'missing_or_ambiguous',
                measurement_basis=None, unit=None, direction=None, normalization=None,
                source_unresolved=['original study-to-variant/WT match', 'endpoint and substrate/condition',
                    'unit and transformation/normalization', 'direction', 'replicate/reliability/censoring'],
                provenance_evidence='pinned mutant/doi.csv and mutant/dataset_summary.csv exact dataset-ID joins; filenames are not protein identity')


def trace_calb_table(data: Path, assay: str, rows: list[dict]) -> dict | None:
    """A bounded targeted source review, not a filename-based unit assignment.

    Only the two exact release files whose every exported row matches the
    same substrate block in original Table 1 receive a unit trace. Complete
    construct identity and experimental reliability remain unqualified.
    """
    columns = {'5A71_kcat': (4, 'kcat (s−1)', 'kcat', 's^-1'),
               '5A71_kcatkm': (6, 'kcat/Km(s−1M−1)', 'kcat_over_Km', 's^-1 M^-1')}
    if assay not in columns:
        return None
    path = data/'source/5A71-source-paper.xml'
    receipt_path = data/'source/5A71-europepmc-fulltext-receipt.json'
    if not path.exists() or not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt['status'] != 'ready' or digest(path) != receipt['sha256']:
        raise ValueError('targeted source paper receipt/hash mismatch')
    document = ET.fromstring(path.read_bytes())
    if not any(e.get('pub-id-type') == 'doi' and e.text == '10.1038/s41467-019-11155-3' for e in document.iter('article-id')):
        raise ValueError('targeted source paper DOI mismatch')
    table = next(t for t in document.iter('table-wrap') if t.get('id') == 'Tab1')
    column, header, endpoint, unit = columns[assay]
    headers = [''.join(e.itertext()).strip() for e in table.findall('./table/thead/tr/th')]
    if headers[column] != header:
        raise ValueError('targeted source unit/header mismatch')
    originals = {}
    for tr in table.findall('./table/tbody/tr'):
        cells = [''.join(e.itertext()).strip() for e in tr.findall('td')]
        if int(cells[1]) > 13:
            break  # Later blocks use other substrates and must not be pooled.
        mutation = cells[3]
        if mutation == '–' or SUBSTITUTION.fullmatch(mutation):
            raw_mean, raw_uncertainty = cells[column].split('±')
            originals['WT' if mutation == '–' else mutation] = dict(table_entry=cells[1],
                mean=float(raw_mean.strip()), uncertainty_raw=raw_uncertainty.strip())
    matches = []
    for row in rows:
        key = 'WT' if row.get('mutation', {}).get('identity_substitution') else row['original']['mutant']
        original = originals.get(key)
        if not original or not row['finite_label'] or row['source_score'] != original['mean']:
            raise ValueError('targeted source Table 1 and released score row mismatch')
        matches.append(dict(source_row_id=row['source_row_id'], **original))
    if {m['table_entry'] for m in matches} != {v['table_entry'] for v in originals.values()}:
        raise ValueError('targeted source does not match all original single/WT table entries')
    return dict(status='all_export_rows_exactly_match_original_table_means', source_doi='10.1038/s41467-019-11155-3',
        source_url=receipt['origin_url'], source_sha256=receipt['sha256'], table='Table 1 / Tab1',
        table_header=headers[column], unit=unit, endpoint=endpoint,
        direction='higher numerical turnover' if endpoint == 'kcat' else 'higher numerical catalytic efficiency',
        measurement_basis='purified-enzyme kinetic parameters, original Table 1 substrate block 1',
        substrate='article substrate 1 (p-nitrophenyl benzoate)', normalization='no transform: exact table means',
        rows=matches, full_experimental_WT_construct_verified=False,
        reliability='original uncertainty strings retained; replicate independence and uncertainty definition not yet qualified')


def exact_groups(reports: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for report in reports:
        if report['WT_consistent']:
            groups[report['wt_sha256']].append(report['assay'])
    return [dict(candidate_group='exactWT:'+h, wt_sha256=h, assay_partitions=sorted(v),
                 assays=len(v), verified_distinct_conditions=None, final_homology_family=False)
            for h,v in sorted(groups.items())]


def freeze_anchors(root: Path, data: Path) -> dict:
    cohort_path = root/'archive/logs/R2/readout_20260923/cohort.json'
    support_path = root/'archive/logs/R1/position_terms_20260926/alignment_v2/anchor_panel_support.json'
    cohort_bytes = cohort_path.read_bytes()
    cohort = json.loads(cohort_bytes)
    support = json.loads(support_path.read_text())
    if hashlib.sha256(cohort_bytes).hexdigest() != support['cohort_sha256']:
        raise ValueError('actual frozen cohort and 201 support digest disagree')
    assays = cohort['assays']
    members = set(support['assays'])
    if len(assays) != 217 or len(members) != 201 or not members <= {a['assay'] for a in assays}:
        raise ValueError('frozen anchor membership mismatch')
    snapshot = dict(cohort_path=str(cohort_path.relative_to(root)), cohort_sha256=hashlib.sha256(cohort_bytes).hexdigest(),
        support_path=str(support_path.relative_to(root)), support_sha256=digest(support_path),
        scope='actual frozen assay wildtype states; old217 and retained201 distinguished',
        rows=[dict(assay=a['assay'], sequence=a['wildtype'], wt_sha256=seqhash(a['wildtype']),
                   retained201=a['assay'] in members, frozen_cluster=a['cluster']) for a in assays])
    target = data/'source/frozen-anchor-WTs.json'
    if target.exists():
        if json.loads(target.read_text()) != snapshot:
            raise ValueError('existing anchor snapshot differs; refusing overwrite')
    else:
        dump(target, snapshot)
    return snapshot


def overlap(sequence: str, anchors: list[dict]) -> list[dict]:
    return [dict(anchor_assay=a['assay'], retained201=a['retained201'],
                 relation='exact' if sequence == a['sequence'] else 'activity_contained_in_anchor' if sequence in a['sequence'] else 'anchor_contained_in_activity')
            for a in anchors if sequence in a['sequence'] or a['sequence'] in sequence]


def validate_release_receipts(receipts: list[dict]) -> None:
    paths = [r['source_path'] for r in receipts]
    if (len(paths) != 130 or len(set(paths)) != 130
            or any(Path(p).parent.as_posix() != 'single_mutant/activity' or not p.endswith('.csv') for p in paths)
            or any(r['status'] not in ('ready', 'failed') for r in receipts)):
        raise ValueError('incomplete, duplicate or invalid pinned activity receipt coverage')


def prepare(root: Path, data: Path, out: Path, acquire: bool = True, workers: int = 4) -> dict:
    receipts = acquire_all(data, out, workers) if acquire else json.loads((data/'source/acquisition-frozen.json').read_text())['files']
    validate_release_receipts(receipts)
    doi_path, summary_path = data/'source/mutant__doi.csv', data/'source/mutant__dataset_summary.csv'
    for source in [doi_path, summary_path]:
        if digest(source) != SOURCE_HASHES[source.name]:
            raise ValueError('author source metadata disagrees with pinned release hash')
    with doi_path.open(newline='') as handle:
        doi_rows = list(csv.DictReader(handle))
    with summary_path.open(newline='') as handle:
        summary_rows = list(csv.DictReader(handle))
    anchors = freeze_anchors(root, data)
    records, reports, states = [], [], {}
    for receipt in receipts:
        assay = Path(receipt['source_path']).stem
        if receipt['status'] != 'ready':
            reports.append(dict(assay=assay, acquisition_status='failed', acquisition_error=receipt['error'], WT_consistent=False, identity_rows=0, label_ready=False, physical_MSE_ready=False))
            continue
        path = Path(receipt['path'])
        if digest(path) != receipt['sha256']:
            raise ValueError('acquired source changed before preparation')
        rows, report = qualify_rows(assay, read_rows(path))
        provenance = source_mapping(assay, doi_rows, summary_rows)
        trace = trace_calb_table(data, assay, rows)
        provenance['targeted_source_trace'] = trace
        if trace:
            provenance.update(unit=trace['unit'], direction=trace['direction'], normalization=trace['normalization'],
                              measurement_basis=trace['measurement_basis'])
            provenance['source_unresolved'] = ['complete original experimental WT construct verification',
                'replicate independence and uncertainty definition', 'condition/protocol completeness and censoring audit']
            report['label_readiness'] = 'endpoint_units_traced_full_qualification_pending'
        report.update(provenance, acquisition_status='ready', source_sha256=receipt['sha256'], origin_url=receipt['origin_url'])
        if report['WT_consistent']:
            wt = rows[0]['wt_sequence']
            report['author_summary_matches_reconstructed_WT'] = provenance['author_summary_WTs'] == [wt]
            if not report['author_summary_matches_reconstructed_WT']:
                # Export sequence-validated identities, but do not claim publication mapping.
                report['source_unresolved'].append('author summary WT differs/missing')
            report['anchor_matches'] = overlap(wt, anchors['rows'])
        else:
            report['anchor_matches'] = []
            report['author_summary_matches_reconstructed_WT'] = False
        for row in rows:
            if trace:
                row.update(score_semantics=trace['endpoint'], unit=trace['unit'],
                           direction=trace['direction'], label_readiness='endpoint_units_traced_full_qualification_pending')
            row.update(source_sha256=receipt['sha256'], source_url=receipt['origin_url'], revision=REVISION,
                       source_doi_candidates=provenance['source_doi_candidates'],
                       candidate_group='exactWT:'+row['wt_sha256'] if row['identity_ready'] else None,
                       condition_key='unverified_file_partition:'+assay)
            if row['identity_ready']:
                for seq, role in [(row['wt_sequence'],'WT'),(row['original']['mutated_sequence'],'mutant' if not row['mutation']['identity_substitution'] else 'WT')]:
                    h = seqhash(seq)
                    state = states.setdefault(h, dict(sequence_sha256=h, sequence=seq, length=len(seq), roles=set(), assays=set()))
                    state['roles'].add(role)
                    state['assays'].add(assay)
        records.extend(rows)
        reports.append(report)
    identity = [r for r in records if r['admissible_identity_single']]
    groups = exact_groups(reports)
    cross_assay = defaultdict(set)
    for row in identity:
        cross_assay[row['variant_key']].add(row['assay'])
    counts = dict(expected_activity_files=130, downloaded_activity_files=sum(r['status']=='ready' for r in receipts),
        failed_activity_files=sum(r['status']!='ready' for r in receipts), source_rows=len(records),
        fully_consistent_WT_files=sum(r['WT_consistent'] for r in reports),
        admissible_single_identity_rows=len(identity), finite_single_label_rows=sum(r['finite_label'] for r in identity),
        identity_substitution_rows=sum(r.get('mutation',{}).get('identity_substitution',False) for r in records),
        unique_exact_WT_proteins=len(groups), unique_WT_mutant_variant_states=len(cross_assay),
        unique_sequence_states=len(states), WT_groups_repeated_across_assay_partitions=sum(g['assays']>1 for g in groups),
        same_variant_states_in_multiple_assays=sum(len(a)>1 for a in cross_assay.values()),
        exact_author_DOI_mapped_files=sum(r.get('source_mapping_status')=='author_map_exact_ID_unverified_original_study' for r in reports),
        author_summary_WT_matching_files=sum(r.get('author_summary_matches_reconstructed_WT',False) for r in reports),
        files_with_missing_labels=sum(r.get('missing_label_rows',0)>0 for r in reports),
        files_with_tied_single_scores=sum(r.get('tied_score_groups',0)>0 for r in reports),
        files_with_discordant_same_variant_scores=sum(r.get('discordant_same_assay_variant_groups',0)>0 for r in reports),
        endpoint_unit_traced_files=sum(r.get('targeted_source_trace') is not None for r in reports),
        descriptive_nonconstant_single_support_files=sum(r.get('descriptive_nonconstant_single_support',False) for r in reports),
        WT_exact_overlap_old217=len({r['wt_sha256'] for r in reports if any(m['relation']=='exact' for m in r.get('anchor_matches',[]))}),
        WT_exact_or_containment_overlap_old217=len({r['wt_sha256'] for r in reports if r.get('anchor_matches')}),
        WT_exact_or_containment_overlap_retained201=len({r['wt_sha256'] for r in reports if any(m['retained201'] for m in r.get('anchor_matches',[]))}),
        label_ready_files=0, physical_MSE_ready_files=0, final_independent_homology_families=None)
    jsonlines(out/'row-registry.jsonl', records)
    jsonlines(out/'admissible-identity-registry.jsonl', identity)
    dump(out/'assay-readiness.json', reports)
    dump(out/'candidate-exactWT-groups.json', dict(groups=groups, scope='Exact WT states only; repeated assays are not independent proteins or confirmed different conditions. Homology pending.'))
    dump(out/'state-inventory.json', [dict(s, roles=sorted(s['roles']), assays=sorted(s['assays'])) for _,s in sorted(states.items())])
    dump(out/'repeated-variant-states.json', [dict(variant_key=k, assays=sorted(v)) for k,v in sorted(cross_assay.items()) if len(v)>1])
    dump(out/'anchor-overlap.json', dict(anchor_snapshot_sha256=digest(data/'source/frozen-anchor-WTs.json'),
        frozen_old217_assays=217, frozen_retained201_assays=201,
        WT_states=[dict(wt_sha256=g['wt_sha256'], matches=next(r['anchor_matches'] for r in reports if r.get('wt_sha256')==g['wt_sha256'])) for g in groups],
        homology='pending; absence of exact/containment overlap is not independence'))
    dump(out/'inference-blockers.json', dict(promotion=False, inference_run=False, fits_run=False,
        blockers=['original study/construct and source-row mapping verification', 'endpoint/unit/direction/normalization per assay',
            'replicate/reliability/censoring and condition provenance', 'homology families and source-study independence',
            'native checkpoint/interface/exact-state coverage join and qualified controls'],
        score_policy='fitness_score remains raw; uninterpreted source-score except explicitly all-row table-matched targeted endpoint traces. No pooling Km/kcat/efficiency/activity or physical MSE claim',
        missing_source_resolution=[dict(assay=r['assay'], unresolved=r.get('source_unresolved',['acquisition_failed'])) for r in reports]))
    outputs = ('row-registry.jsonl', 'admissible-identity-registry.jsonl', 'assay-readiness.json',
               'candidate-exactWT-groups.json', 'state-inventory.json', 'repeated-variant-states.json',
               'anchor-overlap.json', 'inference-blockers.json')
    dump(out/'receipt.json', dict(schema='activity_identity_qualification_v1', as_of=AS_OF, revision=REVISION, counts=counts,
        output_sha256={name: digest(out/name) for name in outputs},
        acquisition_receipts_sha256=hashlib.sha256(json.dumps(receipts, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        code_hashes={str(p.relative_to(root)):digest(p) for p in [Path(__file__),root/'scripts/capability/extensions/prepare_activity_cohort.py']},
        source_hashes={p.name:digest(p) for p in [doi_path,summary_path,data/'source/frozen-anchor-WTs.json']},
        targeted_source_hashes={p.name:digest(p) for p in [data/'source/5A71-source-paper.xml',data/'source/5A71-europepmc-fulltext-receipt.json'] if p.exists()},
        qualified='row-preserving canonical single-substitution identities and within-file exact WT consistency only',
        blocked='biochemical label interpretation, final homology independence, inference and fits'))
    return counts
