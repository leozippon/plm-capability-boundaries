"""Exact, conservative PDB-reference mapping; no experimental construct equivalence claim."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from typing import Any

from .binding_cohort import AA, digest, dump
from ..interactions.contact_enrichment import read_cif, THREE_TO_ONE

MAX_FILE = 25 * 1024**2
MAX_TOTAL = 500 * 1024**2
MAX_EXPANDED = 150 * 1024**2
ADJUDICATION = Path(__file__).with_name('binding-construct-adjudication.json')


def load_adjudication() -> dict:
    return json.loads(ADJUDICATION.read_text())


def validate_adjudication(root: Path, rows: list[dict], phase1: Path, table: dict) -> None:
    """Fail closed if frozen source, admitted-note review or evidence changes."""
    if (digest(root/'data/skempi2/skempi_v2.csv') != table['source_sha256']
            or digest(phase1/'admission.jsonl') != table['admission_sha256']):
        raise ValueError('construct adjudication source/admission binding mismatch')
    notes = sorted({r['original']['Notes'] for r in rows})
    inventory = hashlib.sha256(json.dumps(notes, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()
    indexes = [i for g in table['review_groups'] for i in g['note_indexes']]
    if (len(notes) != table['distinct_notes'] or inventory != table['sorted_notes_sha256']
            or sorted(indexes) != list(range(len(notes)))):
        raise ValueError('construct adjudication distinct Notes review incomplete or changed')
    source = list(csv.DictReader((root/'data/skempi2/skempi_v2.csv').open(newline=''), delimiter=';'))
    for r in rows:
        if source[r['source_record']-1] != r['original']:
            raise ValueError('construct adjudication row/source provenance mismatch')
    originals = {r['source_row_id']: r for r in rows}
    for conflict in table['conflicts']:
        for evidence in conflict['evidence']:
            r = originals[evidence['source_row_id']]
            if (r['complex_id'] not in conflict['complex_ids']
                    or hashlib.sha256(r['original']['Notes'].encode()).hexdigest() != evidence['note_sha256']):
                raise ValueError('construct adjudication evidence mismatch')


def construct_conflicts(row: dict, table: dict) -> list[str]:
    note_hash = hashlib.sha256(row.get('original', {}).get('Notes', '').encode()).hexdigest()
    return [c['reason'] for c in table['conflicts'] if row['complex_id'] in c['complex_ids']
            and ('note_sha256' not in c or c['note_sha256'] == note_hash)]


def qualified_rows(rows: list[dict]) -> list[dict]:
    """Operational references without known conflicts; UNKNOWN is not verified."""
    return [r for r in rows if r['status'] == 'validated_operational_PDB_reference'
            and r.get('construct_status') != 'KNOWN_INCOMPATIBLE']


def scoring_states(rows: list[dict]) -> dict:
    states = {}
    for r in qualified_rows(rows):
        for c in r['chain_states']:
            states[c['wt_sha256']] = dict(sequence=c['sequence'], sequence_sha256=c['wt_sha256'])
        states[r['mutant_sha256']] = dict(sequence=r['mutant_sequence'], sequence_sha256=r['mutant_sha256'])
    return states


def seqhash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode('ascii')).hexdigest()


def category(data: dict, name: str) -> list[dict[str, str]]:
    fields = data.get(name, {})
    if not fields:
        return []
    lengths = {len(v) for v in fields.values()}
    if len(lengths) != 1:
        raise ValueError(f'inconsistent CIF category {name}')
    return [dict(zip(fields, values)) for values in zip(*fields.values())]


def blank(value: str) -> str:
    return '' if value in ('.', '?', '') else value


def parse_reference(text: str, pdb_id: str) -> dict:
    data = read_cif(text)
    entry = category(data, 'entry')
    if len(entry) != 1 or entry[0]['id'].upper() != pdb_id.upper():
        raise ValueError('CIF entry identity mismatch')
    entities = {r['id']: r for r in category(data, 'entity')}
    poly = {r['entity_id']: r for r in category(data, 'entity_poly')}
    sequence_rows = defaultdict(list)
    for r in category(data, 'entity_poly_seq'):
        sequence_rows[r['entity_id']].append(r)
    sequences = {}
    for entity, p in poly.items():
        rows = sequence_rows[entity]
        numbers = [int(r['num']) for r in rows]
        valid = (p.get('type') == 'polypeptide(L)' and len(set(numbers)) == len(numbers)
                 and sorted(numbers) == list(range(1, len(rows)+1)) and bool(rows))
        ordered = sorted(rows, key=lambda r: int(r['num']))
        # Do not convert MSE or unknown monomers to a guessed canonical residue.
        residues = [THREE_TO_ONE.get(r['mon_id'], '?') if r['mon_id'] not in ('MSE','SEC','PYL') else '?' for r in ordered]
        sequence = ''.join(residues)
        canonical = ''.join(p.get('pdbx_seq_one_letter_code_can', '').split())
        raw = ''.join(p.get('pdbx_seq_one_letter_code', '').split())
        valid = valid and set(sequence) <= set(AA) and sequence == canonical == raw
        sequences[entity] = dict(sequence=sequence if valid else None, canonical_complete=valid,
                                 metadata=entities.get(entity, {}), polymer_metadata=p,
                                 reason=None if valid else 'noncanonical_incomplete_or_entity_sequence_disagreement')
    chain_rows = defaultdict(list)
    for r in category(data, 'pdbx_poly_seq_scheme'):
        chain_rows[r['pdb_strand_id']].append(r)
    atom_rows = category(data, 'atom_site')
    chains = {}
    for auth_chain, rows in chain_rows.items():
        identities = {(r['asym_id'], r['entity_id']) for r in rows}
        if len(identities) != 1:
            chains[auth_chain] = dict(status='ambiguous_auth_chain', sequence=None)
            continue
        asym, entity = next(iter(identities))
        info = sequences.get(entity, {})
        sequence = info.get('sequence')
        mapping = defaultdict(set)
        monomers = defaultdict(set)
        for r in rows:
            # pdb_seq_num is the coordinate-file author number, not the historical
            # original author number in auth_seq_num. Atom auth_seq_id cross-checks it.
            number = blank(r.get('pdb_seq_num', '?'))
            if number and number.lstrip('-').isdigit():
                site = (int(number), blank(r.get('pdb_ins_code', '?')))
                mapping[site].add(int(r['seq_id']))
                monomers[site].add(r['mon_id'])
        for r in atom_rows:
            if r.get('label_asym_id') != asym or r.get('auth_asym_id') != auth_chain:
                continue
            number, index = r.get('auth_seq_id', '?'), r.get('label_seq_id', '?')
            if number.lstrip('-').isdigit() and index.isdigit():
                site = (int(number), blank(r.get('pdbx_PDB_ins_code', '?')))
                mapping[site].add(int(index))
                monomers[site].add(r.get('label_comp_id', '?'))
                monomers[site].add(r.get('auth_comp_id', '?'))
        chains[auth_chain] = dict(status='canonical_complete' if sequence else 'noncanonical_or_incomplete',
                                 sequence=sequence, entity_id=entity, label_asym_id=asym,
                                 mapping=dict(mapping), monomers=dict(monomers), metadata=info)
    return dict(pdb_id=pdb_id, chains=chains,
                reference_scope='PDB full deposited entity sequence; operational reference, not measured whole-construct WT identity',
                construct_records={k: category(data, k) for k in ('struct_ref', 'struct_ref_seq', 'struct_ref_seq_dif')})


def map_row(row: dict, reference: dict, adjudication: dict | None = None) -> dict:
    complex_id = row['complex_id']
    _, side1, side2 = complex_id.split('_')
    mutation = row['mutations'][0]
    chains = reference['chains']
    result: dict[str, Any] = dict(source_row_id=row['source_row_id'], complex_id=complex_id,
                  condition_group=row['condition_group'], mutation_group=row['mutation_group'],
                  status='blocked', blockers=[], chain_states=[],
                  experimental_construct_identity='unverified; deposited PDB may be engineered/truncated/differ from measured construct',
                  construct_status='UNKNOWN_CONSTRUCT',
                  model_score_coverage='unverified_no_exact_checkpoint_interface_state_manifest_join')
    conflicts = construct_conflicts(row, load_adjudication() if adjudication is None else adjudication)
    if conflicts:
        result.update(status='KNOWN_INCOMPATIBLE', construct_status='KNOWN_INCOMPATIBLE',
                      experimental_construct_identity='known incompatible deposited reference; measured background recovery required',
                      model_score_coverage='blocked_known_construct_conflict',
                      blockers=['known_construct_conflict:'+reason for reason in conflicts])
        return result
    for name in side1 + side2:
        chain = chains.get(name)
        if not chain or not chain.get('sequence'):
            result['blockers'].append(f'partner_chain_{name}:missing_ambiguous_or_noncanonical_complete_sequence')
        else:
            result['chain_states'].append(dict(auth_chain=name, entity_id=chain['entity_id'],
                  label_asym_id=chain['label_asym_id'], role='partner1' if name in side1 else 'partner2',
                  sequence=chain['sequence'], wt_sha256=seqhash(chain['sequence']),
                  pdb_metadata=chain['metadata']))
    chain = chains.get(mutation['chain'])
    if not chain or not chain.get('sequence'):
        result['blockers'].append('mutated_chain_sequence_unavailable')
        return result
    site = (mutation['author_position'], mutation['insertion_code'])
    indexes = chain['mapping'].get(site, set())
    if len(indexes) != 1:
        result['blockers'].append('author_site_unmapped' if not indexes else 'author_site_ambiguous')
        return result
    index = next(iter(indexes))
    sequence = chain['sequence']
    if not 1 <= index <= len(sequence):
        result['blockers'].append('entity_poly_index_out_of_range')
        return result
    actual = sequence[index-1]
    monomers = chain['monomers'].get(site, set())
    observed = {THREE_TO_ONE.get(m, '?') if m not in ('MSE','SEC','PYL') else '?' for m in monomers}
    if actual != mutation['wt_aa'] or observed != {actual}:
        result['blockers'].append('declared_WT_mismatch_or_ambiguous_site_monomer')
        return result
    result['sequence_index_1based'] = index
    if result['blockers']:
        return result
    mutant = sequence[:index-1] + mutation['mt_aa'] + sequence[index:]
    result.update(status='validated_operational_PDB_reference', wt_sequence=sequence, mutant_sequence=mutant,
                  wt_sha256=seqhash(sequence), mutant_sha256=seqhash(mutant), mutated_auth_chain=mutation['chain'],
                  variant_key=f'{seqhash(sequence)}:{index}:{mutation["wt_aa"]}:{mutation["mt_aa"]}',
                  partner_context=[s['wt_sha256'] for s in result['chain_states']])
    return result


class Budget:
    def __init__(self, total: int = MAX_TOTAL):
        self.limit = total
        self.used = 0
        self.reserved = 0
        self.lock = threading.Lock()

    def reserve(self, maximum: int) -> int:
        """Reserve before reading so concurrent transfers cannot overshoot."""
        with self.lock:
            amount = min(maximum, self.limit-self.used-self.reserved)
            if amount <= 0:
                raise ValueError('total compressed download budget exhausted')
            self.reserved += amount
            return amount

    def settle(self, reservation: int, received: int) -> None:
        if not 0 <= received <= reservation:
            raise ValueError('invalid download reservation settlement')
        with self.lock:
            self.reserved -= reservation
            self.used += received

    def consume(self, size: int) -> None:
        reservation = self.reserve(size)
        if reservation != size:
            self.settle(reservation, 0)
            raise ValueError('total compressed download budget exhausted')
        self.settle(reservation, size)


def read_payload(path: Path) -> str:
    if path.stat().st_size > MAX_FILE:
        raise ValueError('cached/local payload exceeds per-file bound')
    raw = path.read_bytes()
    if path.suffix == '.gz':
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as handle:
            raw = handle.read(MAX_EXPANDED+1)
    if len(raw) > MAX_EXPANDED:
        raise ValueError('expanded payload safety bound exceeded')
    return raw.decode('utf-8')


def acquire(pdb_id: str, cache: Path, local: dict[str, Path], budget: Budget, timeout: int = 30,
            cache_only: bool = False) -> dict:
    started = time.monotonic()
    url = f'https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz'
    target = cache / f'{pdb_id.upper()}.cif.gz'
    receipt: dict[str, Any] = dict(pdb_id=pdb_id, url=url, status='failed', download_bytes=0, timeout_seconds=timeout)
    try:
        existing = local.get(pdb_id.upper())
        path = existing if existing is not None else target
        if path.exists():
            receipt['route'] = 'existing_local_coordinate' if existing is not None else 'public_cache'
        else:
            if cache_only:
                raise FileNotFoundError('cache-only coordinate missing; network acquisition prohibited')
            receipt['route'] = 'RCSB_public_gzip_CIF'
            chunks = []
            with urllib.request.urlopen(url, timeout=timeout) as response:
                while True:
                    remaining = MAX_FILE-receipt['download_bytes']
                    if remaining <= 0:
                        raise ValueError('per-file compressed limit reached before verified EOF')
                    reserved = budget.reserve(min(64*1024, remaining))
                    try:
                        chunk = response.read(reserved)
                    except Exception:
                        # A failed socket read may have consumed bytes internally.
                        # Charge the whole reservation rather than undercounting.
                        budget.settle(reserved, reserved)
                        receipt['uncertain_failed_read_charged_bytes'] = reserved
                        raise
                    budget.settle(reserved, len(chunk))
                    if not chunk:
                        break
                    receipt['download_bytes'] += len(chunk)
                    if time.monotonic()-started > 2*timeout:
                        raise TimeoutError('per-file wall-time bound exceeded')
                    chunks.append(chunk)
            payload = b''.join(chunks)
            # Validate entry before committing an immutable cached public payload.
            with gzip.GzipFile(fileobj=io.BytesIO(payload)) as handle:
                expanded = handle.read(MAX_EXPANDED+1)
            if len(expanded) > MAX_EXPANDED:
                raise ValueError('expanded payload safety bound exceeded')
            text = expanded.decode('utf-8')
            parse_reference(text, pdb_id)
            with target.open('xb') as handle:
                handle.write(payload)
        text = read_payload(path)
        reference = parse_reference(text, pdb_id)
        receipt.update(status='ready', path=str(path), sha256=digest(path), payload_bytes=path.stat().st_size,
                       reference=reference)
    except Exception as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
    receipt['wall_seconds'] = time.monotonic()-started
    return receipt


def local_coordinates(root: Path, ids: set[str], cache: Path) -> dict[str, Path]:
    found = {}
    # Existing CIFs are usable for exact numbering; legacy PDB coordinates alone
    # cannot establish full entity sequence/author scheme and are not substituted.
    for base in (root/'data', root/'results'):
        for parent, dirs, names in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ('.git','__pycache__') and Path(parent)/d != cache]
            for name in sorted(names):
                upper = name.upper()
                stem = upper.removesuffix('.GZ').removesuffix('.CIF')
                if stem in ids and (upper.endswith('.CIF') or upper.endswith('.CIF.GZ')):
                    found.setdefault(stem, Path(parent)/name)
    return found


def components(rows: list[dict]) -> tuple[dict[str, str], list[dict]]:
    rows = qualified_rows(rows)
    parent = {r['complex_id']: r['complex_id'] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    owner = {}
    for row in rows:
        for chain in row['chain_states']:
            h = chain['wt_sha256']
            if h in owner:
                a, b = find(row['complex_id']), find(owner[h])
                parent[max(a,b)] = min(a,b)
            owner[h] = row['complex_id']
    membership = {c: find(c) for c in parent}
    groups = defaultdict(list)
    for r in rows:
        groups[membership[r['complex_id']]].append(r)
    coverage = [dict(component=k, complexes=len({r['complex_id'] for r in v}), label_records=len(v),
                     distinct_single_variants=len({r['variant_key'] for r in v}),
                     conditions=len({r['condition_group'] for r in v}),
                     exact_sequence_proteins=len({c['wt_sha256'] for r in v for c in r['chain_states']})) for k,v in sorted(groups.items())]
    return membership, coverage


def prepare_mapping(root: Path, phase1: Path, out: Path, cache: Path, workers: int = 4,
                    cache_only: bool = False) -> dict:
    if not 1 <= workers <= 4:
        raise ValueError('concurrency must be 1–4')
    receipt = json.loads((phase1/'receipt.json').read_text())
    if digest(root/'data/skempi2/skempi_v2.csv') != receipt['source_sha256']:
        raise ValueError('phase-one source binding mismatch')
    rows = [json.loads(line) for line in (phase1/'admission.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['admitted']]
    adjudication = load_adjudication()
    validate_adjudication(root, rows, phase1, adjudication)
    recovery = json.loads((phase1/'sequence-recovery.json').read_text())
    if {r['complex_id'] for r in rows} != {r['complex_id'] for r in recovery}:
        raise ValueError('phase-one recovery/admission mismatch')
    ids = {r['pdb_id'].upper() for r in rows}
    local = local_coordinates(root, ids, cache)
    cache.mkdir(parents=True, exist_ok=True)
    budget = Budget()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        payloads = list(pool.map(lambda pdb: acquire(pdb, cache, local, budget, cache_only=cache_only), sorted(ids)))
    references = {p['pdb_id']: p['reference'] for p in payloads if p['status'] == 'ready'}
    dump(out/'acquisition.json', dict(max_compressed_total=MAX_TOTAL, max_per_file=MAX_FILE,
                                    concurrency=workers, total_network_bytes=budget.used,
                                    local_coordinate_hits=len(local), records=[{k:v for k,v in p.items() if k != 'reference'} for p in payloads]))
    mapped = []
    deposited_good = 0
    conflicts_previously_mapped = Counter()
    for row in rows:
        reference = references.get(row['pdb_id'].upper())
        if reference:
            operational = map_row(row, reference, {'conflicts': []})
            deposited_good += operational['status'] == 'validated_operational_PDB_reference'
            result = map_row(row, reference, adjudication)
            if operational['status'] == 'validated_operational_PDB_reference' and result['status'] == 'KNOWN_INCOMPATIBLE':
                conflicts_previously_mapped[row['complex_id']] += 1
        else:
            result = map_row(row, {'chains': {}}, adjudication)
            if result['status'] != 'KNOWN_INCOMPATIBLE':
                result['blockers'] = ['coordinate_acquisition_failed']
        mapped.append(result)
    with (out/'row-mapping.jsonl').open('w') as handle:
        for row in mapped:
            handle.write(json.dumps(row, sort_keys=True)+'\n')
    good = qualified_rows(mapped)
    membership, coverage = components(good)
    dump(out/'exact-components.json', dict(membership=membership, coverage=coverage,
         scope='Connected components of exact full-chain sequence sharing on either partner, NOT final independent families. Homology and study dependence pending.'))
    states = scoring_states(good)
    dump(out/'states.json', dict(states=[states[k] for k in sorted(states)],
         validation='canonical full PDB entity sequence and exact declared WT at mapped site; known construct conflicts excluded; UNKNOWN_CONSTRUCT operational reference only',
         model_coverage='unverified; native checkpoint/interface and exact state hash manifest join still required'))
    dump(out/'blocked-tasks.json', [dict(source_row_id=r['source_row_id'], complex_id=r['complex_id'], blockers=r['blockers']) for r in mapped if r['blockers']])
    anchor = root/'data/proteingym_raw/DMS_substitutions.csv'
    anchor_rows = list(csv.DictReader(anchor.open(newline='')))
    anchors = [(r['DMS_id'],r['target_seq']) for r in anchor_rows]
    overlap = []
    for h, state in sorted(states.items()):
        # Compare WT/partner chains, not mutants, to the original public anchor reference.
        if not any(c['wt_sha256'] == h for r in good for c in r['chain_states']):
            continue
        s = state['sequence']
        matches = [dict(anchor_assay=name, relation='exact' if s == a else 'binding_chain_contained_in_anchor' if s in a else 'anchor_contained_in_binding_chain')
                   for name,a in anchors if s in a or a in s]
        overlap.append(dict(sequence_sha256=h, matches=matches, status='anchor_overlap' if matches else 'no_exact_or_containment_match_homology_pending'))
    dump(out/'anchor-overlap.json', dict(reference=str(anchor.relative_to(root)), sha256=digest(anchor),
         scope='ProteinGym original assay reference; exact/containment only, no remote-homology negative claim',
         chains=overlap, independent_candidate_gate='Homology, original retained anchor membership, study overlap and measured construct equivalence still pending.'))
    backgrounds = defaultdict(list)
    originals = {r['source_row_id']:r for r in rows}
    for r in good:
        backgrounds[(r['complex_id'],r['mutated_auth_chain'],r['wt_sha256'],r['condition_group'])].append(r)
    rank = []
    for k,v in sorted(backgrounds.items()):
        labels = [originals[r['source_row_id']]['ln_kd_ratio'] for r in v]
        n = len({r['variant_key'] for r in v})
        rank.append(dict(complex_id=k[0], auth_chain=k[1], wt_sha256=k[2], condition_group=k[3],
                         label_records=len(v), distinct_single_variants=n, nonconstant_labels=len(set(labels))>1,
                         provisional_rank_label_support=n>=5 and len(set(labels))>1,
                         condition_equivalence_and_reliability='pending; 5 distinct variants is descriptive support screen, not promotion rule'))
    dump(out/'ranking-support.json', rank)
    counts = dict(admitted_phase1_labels=len(rows), requested_pdb_ids=len(ids), ready_pdb_ids=len(references),
                  failed_pdb_ids=len(ids)-len(references), validated_operational_labels=len(good), blocked_labels=len(mapped)-len(good),
                  blockers=dict(Counter(b for r in mapped for b in r['blockers'])),
                  operational_complexes=len(membership), exact_identity_components=len(coverage),
                  unique_complete_chain_sequences=len(overlap), unique_pdb_auth_chains=len({(r['complex_id'].split('_')[0],c['auth_chain']) for r in good for c in r['chain_states']}),
                  different_single_variants=len({r['variant_key'] for r in good}), state_sequences=len(states),
                  repeated_variant_label_records=len(good)-len({r['variant_key'] for r in good}),
                  condition_groups=len({r['condition_group'] for r in good}),
                  backgrounds_with_provisional_ranking_support=sum(r['provisional_rank_label_support'] for r in rank),
                  anchor_overlap_chain_sequences=sum(bool(r['matches']) for r in overlap),
                  candidate_without_exact_containment_overlap=sum(not r['matches'] for r in overlap),
                  final_independent_families=None, experimental_constructs_verified=0)
    counts.update(deposited_reference_validated_labels_before_adjudication=deposited_good,
                  known_incompatible_labels=sum(r['status'] == 'KNOWN_INCOMPATIBLE' for r in mapped),
                  previously_mapped_conflicts_by_complex=dict(sorted(conflicts_previously_mapped.items())))
    dump(out/'readiness-gate.json', dict(adjudication_applied=True,
         adjudication_sha256=digest(ADJUDICATION), distinct_source_notes_reviewed=adjudication['distinct_notes'],
         qualified_for_homology=True, ready_for_scoring_or_fits=False,
         qualified_scope='Operational PDB references excluding known construct conflicts; measured identity still UNKNOWN_CONSTRUCT.',
         known_conflict_recovery='Actual measured full background/state sequences required; no guessed residue repairs.',
         cache_only=cache_only, network_bytes=budget.used,
         qualified_outputs=['row-mapping.jsonl', 'states.json', 'exact-components.json', 'ranking-support.json'],
         before_labels=deposited_good, after_labels=len(good),
         blocked_reasons=counts['blockers']))
    dump(out/'receipt.json', dict(counts=counts, source_sha256=receipt['source_sha256'],
         adjudication_gate=dict(status='complete', qualified_row_status='validated_operational_PDB_reference',
              qualified_source_row_ids=sorted(r['source_row_id'] for r in good),
              known_incompatible_source_row_ids=sorted(r['source_row_id'] for r in mapped if r['construct_status'] == 'KNOWN_INCOMPATIBLE'),
              unknown_source_row_ids=sorted(r['source_row_id'] for r in good if r['construct_status'] == 'UNKNOWN_CONSTRUCT'),
              admission=dict(path=os.path.relpath(phase1/'admission.jsonl', out), sha256=digest(phase1/'admission.jsonl')),
              adjudication_sha256=digest(ADJUDICATION), measured_construct_equivalence='pending'),
         adjudication=dict(path=str(ADJUDICATION.relative_to(root)), sha256=digest(ADJUDICATION), applied=True),
         execution=dict(cache_only=cache_only, network_bytes=budget.used),
         inputs={p.name:digest(p) for p in (phase1/'admission.jsonl',phase1/'sequence-recovery.json',phase1/'receipt.json')},
         code_sha256={p.name:digest(p) for p in (Path(__file__),root/'scripts/capability/extensions/map_binding_cohort.py')},
         gates_pending=['measured full-construct WT identity/engineering', 'homology and study overlap grouping',
                        'final retained anchor membership overlap', 'reliability and baseline qualification', 'exact native model-score manifest join'],
         promotion='No fits, inference or main-text promotion'))
    return counts
