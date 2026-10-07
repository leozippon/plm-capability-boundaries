"""Outcome-blind first-stage SKEMPI admission; no sequence or independence claims."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

AA = 'ACDEFGHIKLMNPQRSTVWY'
MUTATION = re.compile(rf'([{AA}])([A-Za-z0-9])(-?\d+)([a-zA-Z]?)([{AA}])')
NUMBER = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?')
MISSING = {'', 'n.b', 'n.b.', 'unf'}  # Source non-binding/unfolded markers are not points.
R_KCAL = 1.9872041e-3
STATUS = 'labels_ready_sequence_mapping_pending'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(values: Any) -> str:
    return hashlib.sha256(json.dumps(values, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()


def point(raw: str, parsed: str, unit: str = 'M') -> tuple[float | None, str]:
    """Validate original strings before using parsed values; malformed values fail."""
    if unit != 'M':
        raise ValueError('paired affinity units must be the source molar units')
    raw, parsed = raw.strip(), parsed.strip()
    if raw in MISSING:
        if parsed:
            raise ValueError('missing/non-binding affinity has a parsed point')
        return None, 'missing_nonbinding_or_unfolded'
    prefix = raw[0] if raw and raw[0] in '<>~≤≥≈' else ''
    value_string = raw[1:].strip() if prefix else raw
    if not NUMBER.fullmatch(value_string):
        raise ValueError(f'malformed original affinity {raw!r}')
    original = float(value_string)
    if parsed:
        if not NUMBER.fullmatch(parsed):
            raise ValueError(f'malformed parsed affinity {parsed!r}')
        if not math.isclose(float(parsed), original, rel_tol=1e-10, abs_tol=0):
            raise ValueError('original/parsed affinity mismatch')
    elif not prefix:
        raise ValueError('point affinity missing parsed value')
    if prefix:
        return None, 'censored_bound' if prefix in '<>≤≥' else 'approximate_not_point'
    if not math.isfinite(original) or original <= 0:
        return None, 'nonpositive_or_nonfinite'
    return original, 'point'


def temperature(raw: str) -> tuple[float | None, str]:
    raw = raw.strip()
    if not raw:
        return None, 'missing'
    assumed = raw.endswith('(assumed)')
    number = raw.removesuffix('(assumed)')
    if not NUMBER.fullmatch(number):
        raise ValueError(f'malformed temperature {raw!r}')
    value = float(number)
    if assumed:
        return None, 'source_assumed_not_measurement'
    if not math.isfinite(value) or not 250 <= value <= 350:
        return None, 'outside_250_350_K'
    return value, 'source_reported_K_not_independently_verified'


def parse_mutations(raw: str, complex_id: str) -> list[dict[str, Any]]:
    if not re.fullmatch(r'[A-Za-z0-9]{4}_[A-Za-z0-9]+_[A-Za-z0-9]+', complex_id):
        raise ValueError(f'malformed complex {complex_id!r}')
    _, side1, side2 = complex_id.split('_')
    if set(side1) & set(side2) or len(set(side1 + side2)) != len(side1 + side2):
        raise ValueError('complex chain sides overlap or repeat')
    mutations = []
    for token in raw.split(','):
        match = MUTATION.fullmatch(token)
        if not match:
            raise ValueError(f'malformed PDB mutation {token!r}')
        wt, chain, position, insertion, mt = match.groups()
        if chain not in side1 + side2:
            raise ValueError('mutation chain not in complex')
        mutations.append(dict(wt_aa=wt, chain=chain, author_position=int(position),
                              insertion_code=insertion, mt_aa=mt))
    return mutations


def admit(source: Path) -> list[dict[str, Any]]:
    records = []
    with source.open(newline='') as handle:
        reader = csv.DictReader(handle, delimiter=';')
        required = {'#Pdb', 'Mutation(s)_PDB', 'Mutation(s)_cleaned', 'Reference', 'Method',
                    'Temperature', 'Affinity_mut (M)', 'Affinity_wt (M)',
                    'Affinity_mut_parsed', 'Affinity_wt_parsed', 'Protein 1', 'Protein 2', 'Notes'}
        if len(reader.fieldnames or []) != 29 or not required.issubset(reader.fieldnames or []):
            raise ValueError('SKEMPI schema mismatch')
        for ordinal, row in enumerate(reader, 1):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'CSV width mismatch at record {ordinal}')
            try:
                mutations = parse_mutations(row['Mutation(s)_PDB'], row['#Pdb'])
                cleaned = row['Mutation(s)_cleaned'].split(',')
                if len(cleaned) != len(mutations):
                    raise ValueError('cleaned/PDB mutation count mismatch')
                for token, mutation in zip(cleaned, mutations):
                    match = MUTATION.fullmatch(token)
                    if not match or (match[1], match[2], match[5]) != (mutation['wt_aa'], mutation['chain'], mutation['mt_aa']):
                        raise ValueError('cleaned/PDB substitution identity mismatch')
                mut, mut_status = point(row['Affinity_mut (M)'], row['Affinity_mut_parsed'])
                wt, wt_status = point(row['Affinity_wt (M)'], row['Affinity_wt_parsed'])
                temp, temp_status = temperature(row['Temperature'])
            except ValueError as error:
                raise ValueError(f'source record {ordinal}: {error}') from error
            reasons = []
            sites = [(m['chain'], m['author_position'], m['insertion_code']) for m in mutations]
            if len(set(sites)) != len(sites):
                reasons.append('source_integrity_repeated_mutation_site')
            if len(mutations) != 1:
                reasons.append('multiple_substitutions')
            if any(m['wt_aa'] == m['mt_aa'] for m in mutations):
                reasons.append('identity_substitution')
            if mut_status != 'point':
                reasons.append('mut_' + mut_status)
            if wt_status != 'point':
                reasons.append('wt_' + wt_status)
            accepted = not reasons
            log_ratio = math.log(mut) - math.log(wt) if accepted and mut and wt else None
            condition = [row[k] for k in ('#Pdb', 'Reference', 'Method', 'Temperature', 'Notes', 'Affinity_wt (M)')]
            record = dict(source_record=ordinal, source_row_id=f'skempi2:{ordinal:06d}',
                          complex_id=row['#Pdb'], pdb_id=row['#Pdb'].split('_')[0],
                          mutations=mutations, single_substitution=len(mutations) == 1,
                          mutation_group=key([row['#Pdb'], row['Mutation(s)_PDB']]),
                          condition_group=key(condition), duplicate_group=key(row),
                          admitted=accepted, exclusion_reasons=reasons,
                          affinity_mut_status=mut_status, affinity_wt_status=wt_status,
                          ln_kd_ratio=log_ratio, temperature_K=temp, temperature_status=temp_status,
                          ddg_kcal_mol=R_KCAL * temp * log_ratio if temp is not None and log_ratio is not None else None,
                          mapping_status=STATUS if accepted else 'excluded', likelihood_coverage='UNKNOWN_before_sequence_join',
                          original=row)
            records.append(record)
    duplicates = Counter(r['duplicate_group'] for r in records)
    for record in records:
        record['duplicate_group_size'] = duplicates[record['duplicate_group']]
    return records


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError('empty support table')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(root: Path, out: Path) -> dict[str, Any]:
    source = root / 'data/skempi2/skempi_v2.csv'
    registry_path = root / 'data/dataset_registry.json'
    prior_path = root / 'data/endpoint_qualification_20260924.json'
    registry = json.loads(registry_path.read_text())
    provenance = registry['datasets']['skempi2']
    source_hash = digest(source)
    expected = next(x['sha256'] for x in provenance['inventory']['files'] if x['path'] == source.name)
    if source_hash != expected:
        raise ValueError('source hash disagrees with dataset registry')
    records = admit(source)
    accepted = [r for r in records if r['admitted']]
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'admission.jsonl').open('w') as handle:
        for r in records:
            handle.write(json.dumps(r, sort_keys=True, allow_nan=False) + '\n')
    counts = dict(source_records=len(records), single_substitution_records=sum(r['single_substitution'] for r in records),
                  source_complexes=len({r['complex_id'] for r in records}), admitted_records=len(accepted),
                  admitted_complexes=len({r['complex_id'] for r in accepted}),
                  admitted_studies=len({r['original']['Reference'] for r in accepted}),
                  admitted_mutation_groups=len({r['mutation_group'] for r in accepted}),
                  admitted_condition_groups=len({r['condition_group'] for r in accepted}),
                  measured_temperature_ddg_records=sum(r['ddg_kcal_mol'] is not None for r in accepted),
                  excluded_records=len(records)-len(accepted),
                  exclusion_reasons=dict(Counter(reason for r in records for reason in r['exclusion_reasons'])),
                  temperature_status=dict(Counter(r['temperature_status'] for r in accepted)),
                  exact_duplicate_groups=sum(v > 1 for v in Counter(r['duplicate_group'] for r in accepted).values()),
                  records_in_exact_duplicate_groups=sum(r['duplicate_group_size'] > 1 for r in accepted),
                  independent_families=None, validated_complete_chains=0)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        o = r['original']
        groups[(o['Reference'], r['complex_id'], r['mutation_group'], r['condition_group'])].append(r)
    table(out / 'support-qc.csv', [dict(study=k[0], complex_id=k[1], mutation_group=k[2], condition_group=k[3],
                                      records=len(v), admitted=sum(x['admitted'] for x in v),
                                      measured_temperature_ddg=sum(x['ddg_kcal_mol'] is not None for x in v),
                                      exclusion_reasons=json.dumps(dict(Counter(z for x in v for z in x['exclusion_reasons'])), sort_keys=True))
                                 for k, v in sorted(groups.items())])
    table(out / 'exclusions.csv', [dict(source_row_id=r['source_row_id'], complex_id=r['complex_id'],
                                      reasons=';'.join(r['exclusion_reasons'])) for r in records if not r['admitted']])
    recovery = []
    graph = []
    for complex_id in sorted({r['complex_id'] for r in accepted}):
        pdb, side1, side2 = complex_id.split('_')
        rows = [r for r in accepted if r['complex_id'] == complex_id]
        chains = sorted({r['mutations'][0]['chain'] for r in rows})
        recovery.append(dict(complex_id=complex_id, pdb_id=pdb, partner1_chains=side1, partner2_chains=side2,
                             mutated_chains=chains, required_author_sites=sorted({r['original']['Mutation(s)_PDB'] for r in rows}),
                             required_states=['complete experimental WT chain sequences for both partners',
                                              'author numbering/insertion-code to sequence index mapping',
                                              'WT residue identity at every admitted site', 'exact one-substitution mutant states',
                                              'construct boundaries, missing residues, engineered background and assembly'],
                             metadata_blockers=['buffer/pH/salt/construct conditions not structured in CSV; recover study metadata',
                                                'uncertainty and replicate independence not supplied per entry'],
                             status=STATUS, likelihood_coverage='UNKNOWN_before_sequence_join'))
        graph.append(dict(complex_id=complex_id, known_pdb_chain_nodes=[f'{pdb}:{c}' for c in side1+side2],
                          named_partner_candidates=sorted({r['original'][field] for r in rows for field in ('Protein 1', 'Protein 2')}),
                          mutated_chain_nodes=[f'{pdb}:{c}' for c in chains]))
    dump(out / 'sequence-recovery.json', recovery)
    dump(out / 'grouping-plan.json', dict(nodes=graph, status='pending_complete_sequence_join',
         rule='Connected components across complexes sharing mutated chains OR partner chains OR homologous chains on either side. Cross-PDB names are candidate links, not validated identities. Resolve study/construct aliases and sequence homology before outer held-component splitting; keep duplicate/condition/mutation groups together.',
         homology_thresholds='Declare before fitting after inspecting sequence/construct coverage; do not equate PDB count with family count.',
         sampling='joint resampling of finalized biological components; rows and fold seeds are not replicates'))
    dump(out / 'source-overlap-inventory.json', dict(status='pending_sequence_homology_and_reference_join',
         studies=sorted({r['original']['Reference'] for r in accepted}), complexes=sorted({r['complex_id'] for r in accepted}),
         anchor_sources={k: {f: v.get(f) for f in ('source_name', 'source_url', 'release', 'retrieved')} for k, v in registry['datasets'].items() if 'proteingym' in k.lower()},
         required_checks=['join anchor assay primary references and construct identity', 'exact WT sequence, containment and homology against anchor and other phenotype cohorts', 'shared publications/partners/families block independence'],
         disjoint=None, novel_families=None))
    dump(out / 'qualification-plan.json', dict(status=STATUS, evidence_level='candidate independent task cohort; independence unqualified',
         primary_target='ln(Kd_mut/Kd_wt), dimensionless; same source M units cancel; positive weakens binding',
         secondary_target='R*T*ln(Kd_mut/Kd_wt) kcal/mol only source-reported actual 250–350 K; never 298(assumed)',
         condition_key_limit='reference/method/temperature/notes/WT affinity is a provisional background proxy, not proof of matched biochemical conditions',
         duplicates='retain all row IDs; no silent averaging, no independence claim; adjudicate replicates and uncertainty before fitting',
         reliability='No per-entry uncertainties. Between-reference/condition spreads are heterogeneity diagnostics, not calibrated noise floors.',
         baselines=['substitution identity and local chemistry', 'training-only qualified sequence profiles/local context',
                    'assay method/temperature and recovered conditions', 'verified interface burial/contact/partner context, not inferred from location labels'],
         metrics=['held-component log-affinity-ratio prediction MSE on qualified common endpoint; condition calibration training-only',
                  'within-experimental-background Spearman only with predeclared sufficient distinct variants and nonconstant target',
                  'temperature-qualified physical ddG MSE separately; no rank-MSE called physical error and no heterogeneous-unit pooling'],
         gates=['complete sequence and WT mapping', 'family/partner leakage graph and anchor overlap adjudication',
                'measurement/condition reliability', 'qualified held-family baseline and uncertainty-supported biological support',
                'join existing likelihood coverage by exact checkpoint/interface/state sequence before requesting genuinely missing scores'],
         main_text='Not promoted. Reasonable unit counts alone do not pass gates; no automatic eight-family rule.',
         forbidden=['model inference', 'new fits', 'frozen result changes']))
    dump(out / 'receipt.json', dict(schema='binding_preparation_v1', counts=counts, source_sha256=source_hash,
         source=source.relative_to(root).as_posix(), source_provenance=provenance,
         registry_sha256=digest(registry_path), prior_qualification_sha256=digest(prior_path),
         prior_definition=json.loads(prior_path.read_text())['endpoints']['skempi2'],
         departure_from_prior='First-stage log ratio accepts single point substitutions without requiring known temperature, rejects original-string bounds/approximations; physical ddG excludes source-assumed temperature. Historical parsed-affinity counts are not this support.',
         prepared_files=sorted(p.name for p in out.iterdir() if p.is_file())))
    return counts
