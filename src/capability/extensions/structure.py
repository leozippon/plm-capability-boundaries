"""Experimental, exact-sequence structural adapter for mutation anchors.

No outcome, prediction, selected contact pair or fitted coefficient is consumed.
Positions are 1-based WT/entity positions. Degrees count all mapped WT residues
with observed CB (Gly CA), strict distance <8 A and sequence separation >2.
They are observed-coordinate lower bounds, not completed native degrees. RSA is
recomputed with the existing deterministic Shrake-Rupley implementation using
all observed heavy atoms of the isolated selected chain (including entity flanks).
Crystal partners, ligands, missing atoms and biological assembly are not included;
fragment degree is truncated to the WT while RSA can include entity flanks.
Secondary structure is deliberately unavailable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil

import numpy as np

from ..interactions.contact_enrichment import (
    file_digest, load_structure, relative_accessibility, residue_accessibility,
)

SCHEMA = 'mutation-structure-v1'
CHOICE_RULE = ('most mapped coordinate residues, then full entity before fragment, '
               'then source sha256, entity id, lowest model number, label chain id')
EXPERIMENTAL_METHODS = frozenset({
    'X-RAY DIFFRACTION', 'SOLUTION NMR', 'SOLID-STATE NMR',
    'ELECTRON MICROSCOPY', 'ELECTRON CRYSTALLOGRAPHY', 'NEUTRON DIFFRACTION',
    'FIBER DIFFRACTION', 'POWDER DIFFRACTION',
})
CAVEATS = [
    'Retained structures were acquired for a selected pair cohort, not a representative mutation anchor sample.',
    'Degree counts only mapped WT residues with observed CB (Gly CA); absent coordinates or CB are missing, never zero.',
    'Observed degree is a lower bound when neighbors are missing; entity flanks outside WT are excluded from degree.',
    'RSA uses isolated-chain observed heavy atoms including entity flanks; partners, ligands, assembly and missing atoms are excluded.',
    'One deterministically selected chain/model is used; conformational ensemble uncertainty is not estimated.',
    'RSA is un-clipped Tien-2013-normalized Shrake-Rupley with 256 sphere points; incomplete atoms can bias exposure.',
    'No validated secondary structure tool is used; secondary structure is unavailable.',
]


def mutation_positions(mutation: str, wildtype: str) -> list[int]:
    """Validate substitutions against WT, retaining every site of a multi-mutant."""
    positions = []
    for token in mutation.split(':'):
        match = re.fullmatch(r'([A-Z])(\d+)([A-Z])', token)
        if match is None:
            raise ValueError(f'invalid substitution: {mutation}')
        before, text, after = match.groups()
        position = int(text)
        if not 1 <= position <= len(wildtype) or wildtype[position - 1] != before:
            raise ValueError(f'mutation disagrees with WT: {mutation}')
        if position in positions or before == after:
            raise ValueError(f'duplicate site or non-substitution: {mutation}')
        positions.append(position)
    return positions


def load_anchor(cohort_path, *, assay_ids=None) -> dict:
    """Project the replay cohort's ``assays`` onto label-blind identifiers only.

    Pass admission['support']['assay_ids'] to select the existing 201-assay anchor.
    None selects all assays; no support is inferred from measurements.
    """
    path = Path(cohort_path)
    cohort = json.loads(path.read_text())
    selected = None if assay_ids is None else set(assay_ids)
    rows = []
    seen = set()
    for assay in cohort['assays']:
        name = assay['assay']
        if name in seen:
            raise ValueError(f'duplicate assay: {name}')
        seen.add(name)
        if selected is not None and name not in selected:
            continue
        wt = assay['wildtype']
        if not wt or any(residue not in 'ACDEFGHIKLMNPQRSTVWY' for residue in wt):
            raise ValueError(f'noncanonical WT: {name}')
        mutants = list(assay['mutants'])
        if len(set(mutants)) != len(mutants):
            raise ValueError(f'duplicate mutant: {name}')
        for mutation in mutants:
            mutation_positions(mutation, wt)
        rows.append(dict(assay=name, cluster=assay['cluster'], wildtype=wt, mutants=mutants))
    if selected is not None and selected - seen:
        raise ValueError(f'unknown assay ids: {sorted(selected - seen)}')
    return {'cohort_path': str(path), 'cohort_sha256': file_digest(path)[0],
            'assays': sorted(rows, key=lambda row: row['assay'])}


def exact_mapping(wildtype: str, entity: str) -> dict:
    """Require one exact occurrence of the full WT, never a local near match."""
    if not wildtype:
        raise ValueError('empty WT')
    starts = [i for i in range(len(entity) - len(wildtype) + 1)
              if entity.startswith(wildtype, i)]
    if len(starts) != 1:
        return {'status': 'ambiguous_exact_mapping' if starts else 'no_exact_full_wt_match',
                'mapping': {}}
    offset = starts[0]
    return {'status': 'admitted', 'kind': 'full_entity' if wildtype == entity else 'full_wt_fragment',
            'mapping': {i: i + offset for i in range(1, len(wildtype) + 1)}}


def _candidates(wt, structure):
    if structure['method'].upper() not in EXPERIMENTAL_METHODS:
        return [], [{'reason': 'nonexperimental_or_unknown_method'}]
    candidates, exclusions = [], []
    for entity, sequence in sorted(structure['entities'].items()):
        matched = exact_mapping(wt, sequence)
        if matched['status'] != 'admitted':
            exclusions.append({'entity_id': entity, 'reason': matched['status']})
            continue
        chains = [chain for chain in structure['chains'].values() if chain.entity_id == entity]
        if not chains:
            exclusions.append({'entity_id': entity, 'reason': 'no_coordinate_chain'})
        for chain in chains:
            mapping = matched['mapping']
            bad = [i for i, j in mapping.items() if j in chain.residues
                   and chain.residues[j].residue != wt[i - 1]]
            if bad:
                exclusions.append({'entity_id': entity, 'chain': chain.label_asym_id,
                                   'model': chain.model, 'reason': 'coordinate_identity_mismatch',
                                   'wt_positions': bad})
                continue
            if any(not np.isfinite(residue.array()).all() for residue in chain.residues.values()):
                exclusions.append({'entity_id': entity, 'chain': chain.label_asym_id,
                                   'model': chain.model, 'reason': 'nonfinite_coordinates'})
                continue
            observed = sum(j in chain.residues for j in mapping.values())
            candidates.append({'chain': chain, 'matched': matched, 'observed': observed})
    return candidates, exclusions


def annotate_residues(wildtype: str, chain, mapping: dict[int, int]) -> list[dict]:
    """Annotate every WT position, never selected contact pairs or mutant sites."""
    if set(mapping) != set(range(1, len(wildtype) + 1)) or any(
            mapping[i] != mapping[1] + i - 1 for i in mapping) or mapping[1] < 1:
        raise ValueError('map must cover the complete WT contiguously with 1-based positions')
    if any(j in chain.residues and chain.residues[j].residue != wildtype[i - 1]
           for i, j in mapping.items()):
        raise ValueError('mapped coordinate identity disagrees with WT')
    vectors = {}
    for i, j in mapping.items():
        residue = chain.residues.get(j)
        if residue is not None:
            vector = residue.named('CA' if wildtype[i - 1] == 'G' else 'CB')
            if vector is not None:
                vectors[i] = vector
    wanted = [j for j in mapping.values() if j in chain.residues]
    areas = residue_accessibility(chain, wanted) if wanted else {}
    rows = []
    for i, j in sorted(mapping.items()):
        residue = chain.residues.get(j)
        degree = None
        if i in vectors:
            degree = sum(abs(i - k) > 2 and float(np.linalg.norm(vectors[i] - v)) < 8.0
                         for k, v in vectors.items())
        rows.append({'wt_position': i, 'entity_position': j, 'residue': wildtype[i - 1],
                     'coordinate_residue': None if residue is None else residue.residue,
                     'auth_seq_id': None if residue is None else residue.auth_seq_id,
                     'coordinate_present': residue is not None,
                     'contact_atom_present': i in vectors,
                     'contact_coordinate': vectors[i].tolist() if i in vectors else None,
                     'contact_atom': 'CA' if wildtype[i - 1] == 'G' else 'CB',
                     'contact_degree': degree,
                     'rsa': None if residue is None else relative_accessibility(areas[j], wildtype[i - 1]),
                     'secondary_structure': None})
    return rows


def annotate_cohort(anchor: dict, structure_paths) -> dict:
    """Return annotation plus exhaustive assay/source/entity exclusion audit.

    Sources are parsed once. No coordinates are copied into the receipt. Selection
    compares every admitted chain/model, independent of mutant counts and labels.
    """
    sources, loaded = [], []
    for path in sorted({Path(p).resolve() for p in structure_paths}):
        sha, size = file_digest(path)
        source = {'path': str(path), 'sha256': sha, 'bytes': size}
        try:
            structure = load_structure(path)
        except (ValueError, KeyError, OSError) as error:
            source.update(status='parse_failure', error=f'{type(error).__name__}: {error}')
            structure = None
        else:
            source.update(status='parsed', method=structure['method'], entry_id=structure['entry_id'],
                          entity_lengths={k: len(v) for k, v in structure['entities'].items()},
                          coordinate_chains=len(structure['chains']))
        sources.append(source)
        loaded.append((source, structure))
    assays, audit = [], []
    for assay in anchor['assays']:
        candidates = []
        for source, structure in loaded:
            found, excluded = _candidates(assay['wildtype'], structure) if structure is not None else (
                [], [{'reason': 'source_parse_failure'}])
            audit.append({'assay': assay['assay'], 'source_sha256': source['sha256'],
                          'candidate_chains': len(found), 'exclusions': excluded})
            for candidate in found:
                candidate.update(source=source, structure=structure)
                candidates.append(candidate)
        if not candidates:
            assays.append({'assay': assay['assay'], 'cluster': assay['cluster'],
                           'status': 'excluded_no_admissible_structure', 'residues': []})
            continue
        candidates.sort(key=lambda c: (-c['observed'], c['matched']['kind'] != 'full_entity',
                          c['source']['sha256'], c['chain'].entity_id, c['chain'].model,
                          c['chain'].label_asym_id))
        chosen = candidates[0]
        chain = chosen['chain']
        assays.append({'assay': assay['assay'], 'cluster': assay['cluster'], 'status': 'admitted',
                       'source_sha256': chosen['source']['sha256'], 'source_path': chosen['source']['path'],
                       'entry_id': chosen['structure']['entry_id'], 'method': chosen['structure']['method'],
                       'entity_id': chain.entity_id, 'entity_length': len(chosen['structure']['entities'][chain.entity_id]),
                       'chain': chain.label_asym_id, 'auth_chain': chain.auth_asym_id, 'model': chain.model,
                       'mapping_kind': chosen['matched']['kind'], 'candidate_chains': len(candidates),
                       'residues': annotate_residues(assay['wildtype'], chain, chosen['matched']['mapping'])})
    admitted = [row for row in assays if row['status'] == 'admitted']
    sites = [site for row in admitted for site in row['residues']]
    admitted_ids = {row['assay'] for row in admitted}
    return {'schema': SCHEMA, 'cohort_sha256': anchor['cohort_sha256'],
            'choice_rule': CHOICE_RULE, 'caveats': CAVEATS, 'sources': sources,
            'definitions': {'contact': 'CB (Gly CA), strict <8 angstrom, |WT seqdist|>2, all mapped WT residues',
                            'rsa': 'isolated full observed chain, Shrake-Rupley 256 points, probe 1.4 A, Tien 2013 maxima'},
            'coverage': {'assays_total': len(assays), 'assays_admitted': len(admitted),
                         'families_total': len({r['cluster'] for r in assays}),
                         'families_admitted': len({r['cluster'] for r in admitted}),
                         'variants_total': sum(len(r['mutants']) for r in anchor['assays']),
                         'variants_in_admitted_assays': sum(len(r['mutants']) for r in anchor['assays'] if r['assay'] in admitted_ids),
                         'wt_sites_total': sum(len(r['wildtype']) for r in anchor['assays']),
                         'wt_sites_mapped': len(sites), 'coordinate_sites': sum(r['coordinate_present'] for r in sites),
                         'degree_sites': sum(r['contact_degree'] is not None for r in sites),
                         'rsa_sites': sum(r['rsa'] is not None for r in sites)},
            'assays': assays, 'audit': audit}


def site_table(anchor: dict, annotation: dict) -> list[dict]:
    """Flat all-WT-site table, including excluded proteins and missing coordinates.

    Fields: assay_id, cluster, wildtype, wildtype_sha256, wt_position (1-based),
    residue, entity_position, coordinate_present, contact_coordinate ([x,y,z] A
    or None), contact_atom (CB/Gly CA), contact_atom_present, contact_degree, rsa,
    source_sha256/source_path, entry_id, entity_id, chain/auth_chain, model,
    mapping_kind, auth_seq_id, status. The entire map is these per-position rows.
    """
    import hashlib
    if anchor['cohort_sha256'] != annotation['cohort_sha256']:
        raise ValueError('anchor/annotation cohort hash mismatch')
    mapped = {row['assay']: row for row in annotation['assays']}
    if len(mapped) != len(annotation['assays']) or set(mapped) != {r['assay'] for r in anchor['assays']}:
        raise ValueError('anchor/annotation assay mismatch')
    rows = []
    for assay in anchor['assays']:
        saved = mapped[assay['assay']]
        sites = {r['wt_position']: r for r in saved['residues']}
        wt = assay['wildtype']
        provenance = {key: saved.get(key) for key in ('source_sha256', 'source_path', 'entry_id',
                      'entity_id', 'chain', 'auth_chain', 'model', 'mapping_kind')}
        for position, residue in enumerate(wt, 1):
            site = sites.get(position, {'wt_position': position, 'entity_position': None,
                'residue': residue, 'coordinate_residue': None, 'auth_seq_id': None,
                'coordinate_present': False, 'contact_atom_present': False,
                'contact_coordinate': None, 'contact_atom': 'CA' if residue == 'G' else 'CB',
                'contact_degree': None, 'rsa': None, 'secondary_structure': None})
            rows.append({'assay_id': assay['assay'], 'cluster': assay['cluster'],
                         'wildtype': wt, 'wildtype_sha256': hashlib.sha256(wt.encode()).hexdigest(),
                         'status': saved['status'], **provenance, **site})
    return rows


def mutation_site_rows(anchor: dict, annotation: dict) -> list[dict]:
    """Long-form mutation/site join; excluded assays yield explicit missing rows.

    Each multi-mutant yields one row per site, NOT one row per response pair.
    A fitting consumer must decide its own preregistered aggregation/support rule.
    """
    if anchor['cohort_sha256'] != annotation['cohort_sha256']:
        raise ValueError('anchor/annotation cohort hash mismatch')
    mapped = {row['assay']: row for row in annotation['assays']}
    if len(mapped) != len(annotation['assays']) or set(mapped) != {r['assay'] for r in anchor['assays']}:
        raise ValueError('anchor/annotation assay mismatch')
    rows = []
    for assay in anchor['assays']:
        saved = mapped[assay['assay']]
        sites = {r['wt_position']: r for r in saved['residues']}
        for index, mutation in enumerate(assay['mutants']):
            for position in mutation_positions(mutation, assay['wildtype']):
                site = sites.get(position, {'wt_position': position, 'entity_position': None,
                    'residue': assay['wildtype'][position - 1], 'coordinate_present': False,
                    'contact_atom_present': False, 'contact_degree': None, 'rsa': None,
                    'secondary_structure': None})
                rows.append({'assay': assay['assay'], 'cluster': assay['cluster'],
                             'variant_index': index, 'mutation': mutation, 'status': saved['status'],
                             'source_sha256': saved.get('source_sha256'), **site})
    return rows


def main():
    """CPU-only receipt producer; caller supplies existing anchor admission IDs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--structures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--extra-structure', type=Path, action='append', default=[],
                        help='Additional existing local experimental mmCIF source; repeatable')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('output must be new or empty')
    args.out.mkdir(parents=True, exist_ok=True)
    threads = int(os.environ.get('OMP_NUM_THREADS', '2'))
    if not 1 <= threads <= 4:
        raise ValueError('CPU annotation requires 1-4 threads')
    if shutil.disk_usage(args.out).free < 1_000_000_000:
        raise ValueError('insufficient disk headroom')
    def save(name, value):
        (args.out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    save('runtime.json', {'device': 'cpu', 'python': platform.python_version(), 'threads': threads,
         'disk_free_bytes': shutil.disk_usage(args.out).free, 'memory_preflight': Path('/proc/meminfo').read_text()})
    admission = json.loads(args.admission.read_text())
    anchor = load_anchor(args.cohort, assay_ids=admission['support']['assay_ids'])
    if anchor['cohort_sha256'] != admission['cohort_sha256']:
        raise ValueError('admission/cohort hash mismatch')
    paths = sorted(args.structures.glob('*.cif*')) + args.extra_structure
    if not paths:
        raise ValueError('no retained coordinates')
    receipt = annotate_cohort(anchor, paths)
    receipt['anchor_admission'] = {'path': str(args.admission.resolve()), 'sha256': file_digest(args.admission)[0]}
    receipt['code_sha256'] = {str(Path(__file__).resolve()): file_digest(__file__)[0],
        'contact_enrichment.py': file_digest(Path(__file__).parents[1] / 'interactions/contact_enrichment.py')[0]}
    save('coverage.json', receipt)
    import gzip
    with gzip.open(args.out / 'sites.json.gz', 'wt') as handle:
        json.dump(site_table(anchor, receipt), handle, allow_nan=False)
        handle.write('\n')
    variants = mutation_site_rows(anchor, receipt)
    variant_coverage = {'mutation_sites_total': len(variants),
                        'mutation_sites_with_coordinates': sum(r['coordinate_present'] for r in variants),
                        'mutation_sites_with_degree': sum(r['contact_degree'] is not None for r in variants),
                        'mutation_sites_with_rsa': sum(r['rsa'] is not None for r in variants)}
    save('variant-coverage.json', variant_coverage)
    save('completion.json', {'status': 'complete' if receipt['coverage']['assays_admitted'] else 'zero_intersection',
                            'coverage': receipt['coverage'], 'coverage_sha256': file_digest(args.out / 'coverage.json')[0]})
    print(json.dumps(receipt['coverage']), flush=True)


if __name__ == '__main__':
    main()
