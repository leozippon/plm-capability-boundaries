#!/usr/bin/env python3
"""Exhaustive label-blind activity WT homology screen; no label promotion.

Only exact WT groups are queries. Mutants remain in the complete downstream
unit map, never in the alignment roster. No binding adjudication is consulted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.capability.extensions.phenotype_homology import (
    ExactAligner, MIN_ALIGNED_RESIDUES, align_task, canonical, dump, load_sources,
    pair_tasks, seqhash, sha, task_plan)
from src.capability.core.family_grouping import Union


def load_activity(mapping: Path) -> tuple[list[dict], dict[str, str], list[dict], dict[str, dict], dict]:
    receipt = json.loads((mapping / 'receipt.json').read_text())
    if receipt.get('schema') != 'activity_identity_qualification_v1':
        raise ValueError('activity identity receipt required')
    bound = receipt.get('output_sha256', {})
    required = {'candidate-exactWT-groups.json', 'state-inventory.json', 'assay-readiness.json'}
    if not required <= set(bound):
        raise ValueError('receipt must hash-bind WT groups, states and readiness')
    for name, expected in bound.items():
        if Path(name).name != name or sha(mapping / name) != expected:
            raise ValueError(f'activity input hash mismatch: {name}')
    groups = json.loads((mapping / 'candidate-exactWT-groups.json').read_text())['groups']
    states = json.loads((mapping / 'state-inventory.json').read_text())
    readiness = json.loads((mapping / 'assay-readiness.json').read_text())
    by_state = {}
    for state in states:
        h, sequence = state['sequence_sha256'], state['sequence']
        canonical(sequence)
        if seqhash(sequence) != h or state['length'] != len(sequence) or h in by_state:
            raise ValueError('state hash/length mismatch or duplicate state')
        by_state[h] = state
    chains, assay_wt = {}, {}
    for group in groups:
        h = group['wt_sha256']
        if h in chains or h not in by_state or 'WT' not in by_state[h]['roles']:
            raise ValueError('WT group must identify one explicit WT inventory state')
        assays = group['assay_partitions']
        if not assays or len(assays) != len(set(assays)) or group['assays'] != len(assays):
            raise ValueError('WT group assay census mismatch')
        if not set(assays) <= set(by_state[h]['assays']):
            raise ValueError('WT inventory/group assay mismatch')
        chains[h] = by_state[h]['sequence']
        for assay in assays:
            if assay in assay_wt:
                raise ValueError('assay assigned to multiple WT groups')
            assay_wt[assay] = h
    if {h for h, s in by_state.items() if 'WT' in s['roles']} != set(chains):
        raise ValueError('WT group roster omits inventory WT')
    if any(not s['assays'] or not set(s['assays']) <= set(assay_wt) for s in states):
        raise ValueError('state has orphan assay')
    by_assay = {r['assay']: r for r in readiness}
    if len(by_assay) != len(readiness) or set(by_assay) != set(assay_wt):
        raise ValueError('readiness assay roster mismatch')
    if any(r['wt_sha256'] != assay_wt[a] or not r['WT_consistent'] for a, r in by_assay.items()):
        raise ValueError('readiness WT identity mismatch')
    counts = dict(unique_exact_WT_proteins=len(chains), downloaded_activity_files=len(assay_wt),
                  unique_sequence_states=len(states), label_ready_files=sum(r['label_ready'] for r in readiness),
                  physical_MSE_ready_files=sum(r['physical_MSE_ready'] for r in readiness))
    if any(receipt.get('counts', {}).get(k) != v for k, v in counts.items()):
        raise ValueError('activity receipt census mismatch')
    manifest = dict(receipt_sha256=sha(mapping / 'receipt.json'), output_sha256=bound, counts=counts,
                    roster='explicit exact WT groups only; mutant-only states never align',
                    prior_qualification=receipt['qualified'], prior_blocked=receipt['blocked'])
    return groups, dict(sorted(chains.items())), states, by_assay, manifest


def export_units(groups, chains, states, readiness, membership, hits) -> dict:
    assay_wt = {a: g['wt_sha256'] for g in groups for a in g['assay_partitions']}
    direct = {r['query'] for r in hits if r['kind'] == 'anchor' and r['admitted']}
    overlap = {r['query'] for r in hits if r['kind'] == 'anchor' and r['family_overlap']}
    excluded = {membership[h] for h in direct}
    short = {membership[h] for h, s in chains.items() if len(s) < MIN_ALIGNED_RESIDUES}
    candidate = {h: g for h, g in membership.items() if g not in excluded | short}
    candidate_assays = {a for a, h in assay_wt.items() if h in candidate}
    units = []
    for h in chains:
        assays = sorted(a for a, wt in assay_wt.items() if wt == h)
        units.append(dict(wt_sha256=h, component=membership[h], assay_partitions=assays,
            sequence_states=sorted(s['sequence_sha256'] for s in states if set(s['assays']) & set(assays)),
            direct_anchor_hit=h in direct, cascade_only_anchor_excluded=h not in direct and membership[h] in excluded,
            lower_30_query_family_overlap=h in overlap, short_WT_unresolved=len(chains[h]) < MIN_ALIGNED_RESIDUES,
            component_short_WT_unresolved=membership[h] in short, candidate=h in candidate))
    components = []
    for component in sorted(set(membership.values())):
        wts = {h for h, g in membership.items() if g == component}
        components.append(dict(component=component, WT_members=sorted(wts),
            direct_anchor_WTs=sorted(wts & direct), cascade_only_WTs=sorted(wts-direct) if component in excluded else [],
            anchor_excluded=component in excluded, short_WT_unresolved=component in short,
            candidate=component not in excluded | short))
    assay_map = {a: dict(wt_sha256=h, component=membership[h], candidate=h in candidate,
        label_ready=readiness[a]['label_ready'], physical_MSE_ready=readiness[a]['physical_MSE_ready'],
        labels_promoted=False) for a, h in sorted(assay_wt.items())}
    state_map = {s['sequence_sha256']: dict(roles=s['roles'], assays=s['assays'],
        WT_groups=sorted({assay_wt[a] for a in s['assays']}),
        components=sorted({membership[assay_wt[a]] for a in s['assays']}),
        candidate_assays=sorted(set(s['assays']) & candidate_assays)) for s in states}
    remaining_states = sum(bool(s['candidate_assays']) for s in state_map.values())
    return dict(scope='operational sequence-homology components, not proved evolutionary or study independence',
        preexclusion_WT_membership=membership, candidate_WT_membership=candidate,
        WT_units=units, components=components, assay_membership=assay_map, state_membership=state_map,
        labels_promoted=False, study_dependency='pending; repeated assays of the same WT are not independent',
        interpretation='absence of a 50/80 anchor is not absence of homology; lower 30/80 overlap is report-only',
        counts=dict(full_WTs=len(chains), full_assays=len(assay_map), full_sequence_states=len(states),
            full_components=len(components), direct_anchor_WTs=len(direct),
            anchor_excluded_WTs=sum(g in excluded for g in membership.values()),
            short_unresolved_components=len(short), lower_30_query_overlap_WTs=len(overlap),
            remaining_WTs=len(candidate), remaining_assays=len(candidate_assays),
            remaining_sequence_states=remaining_states, remaining_components=len(set(candidate.values())),
            full_label_ready_assays=sum(r['label_ready'] for r in readiness.values()),
            remaining_label_ready_assays=sum(readiness[a]['label_ready'] for a in candidate_assays),
            remaining_physical_MSE_ready_assays=sum(readiness[a]['physical_MSE_ready'] for a in candidate_assays),
            promoted_labels=0))


def full_screen(aligner, chains, sources, out, workers):
    union, counts, hits = Union(chains), Counter(), []
    with (out / 'pair-evidence.jsonl').open('x') as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = iter(pair_tasks(chains, sources))
        while chunk := list(itertools.islice(tasks, 256)):
            for r in pool.map(lambda t: align_task(aligner, t), chunk):
                kind = r['kind']
                counts[kind + '_pairs'] += 1
                if r['admitted']:
                    counts[kind + '_admitted_links'] += 1
                    counts[kind + ('_exact_containment_links' if r['relation'] else '_homolog_links')] += 1
                    if kind == 'family':
                        union.join(r['query'], r['subject'], r['relation'] or '30/80/80')
                if r['minimum_suppressed']:
                    counts[kind + '_minimum_suppressed'] += 1
                if kind == 'anchor':
                    if r['family_overlap']:
                        counts['anchor_lower_30_query_overlap_pairs'] += 1
                    if r['family_overlap_minimum_suppressed']:
                        counts['anchor_lower_30_minimum_suppressed'] += 1
                if r['admitted'] or r['minimum_suppressed'] or (kind == 'anchor' and
                        (r['family_overlap'] or r['family_overlap_minimum_suppressed'])):
                    if kind == 'anchor':
                        r['source'] = {k: v for k, v in sources[int(r['subject'])].items() if k != 'sequence'}
                    handle.write(json.dumps(r, sort_keys=True) + '\n')
                    hits.append(r)
    plan = task_plan(chains, sources)
    if any(counts[k] != plan[k] for k in ('family_pairs', 'anchor_pairs')):
        raise ValueError('incomplete exhaustive screen; no candidate export')
    return {h: union.find(h) for h in chains}, hits, dict(counts)


def resources():
    return dict(executable=sys.executable, python=sys.version, disk_free_bytes=shutil.disk_usage(ROOT).free,
        memory=[s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith(('MemTotal:', 'MemAvailable:'))],
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        thread_environment={k: os.environ[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
        execution='CPU only; no inference/GPU/H200/network/fits')


def main():
    base = ROOT / 'results/extensions/phenotype_followups_20261007/activity'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', type=Path, default=base / 'final-20261007')
    parser.add_argument('--out', type=Path, default=base / 'homology')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('refusing to overwrite frozen homology output')
    args.out.mkdir(parents=True)
    before, started = resources(), time.monotonic()
    dump(args.out / 'resource-receipt.json', dict(status='running', before=before))
    try:
        groups, chains, states, readiness, inputs = load_activity(args.mapping)
        sources, source_manifest = load_sources(ROOT)
        code_paths = [Path(__file__), ROOT / 'src/capability/extensions/phenotype_homology.py',
            ROOT / 'src/capability/extensions/phenotype_homology.c', ROOT / 'src/capability/core/family_grouping.py',
            ROOT / 'src/capability/core/amino_acids.py']
        code = {str(p.relative_to(ROOT)): sha(p) for p in code_paths}
        frozen = dict(schema='activity_WT_homology_contract_v1', label_blind=True, inputs=inputs,
            development_sources=source_manifest, code_sha256=code,
            algorithm='unchanged phenotype_homology ExactAligner/align_task: exact SW canonical BLOSUM62, affine 11+1*k, frozen ties; identity identical/all columns; coverage coordinate span/full length',
            family_rule=dict(identity_percent=30, both_coverage_percent=80, min_paired_residues=30, exact_or_either_containment='admit any length'),
            anchor_rule=dict(identity_percent=50, query_coverage_percent=80, min_paired_residues=30, subject_coverage='unconstrained', exact_or_either_containment='admit any length'),
            lower_family_overlap='30 identity/80 query coverage and >=30 paired residues, or exact/containment; report-only',
            exclusion='exclude entire WT component if any admitted anchor hit or WT length <30; preserve full map and direct/cascade attribution',
            resource_contract=dict(workers=args.workers, threads_per_alignment=1, numerical_threads=1, execution='CPU only'),
            study_dependency='pending; repeated assays same WT not independent',
            promotion='none; source/endpoint/units/normalization/reliability/baseline and state-interface qualification remain separate')
        dump(args.out / 'contract.json', frozen)
        dump(args.out / 'input-plan.json', task_plan(chains, sources))
        dump(args.out / 'development-sources.json', sources)
        aligner = ExactAligner(args.out / 'runtime')
        aligner.prepare([*chains.values(), *(s['sequence'] for s in sources)])
        membership, hits, counts = full_screen(aligner, chains, sources, args.out, args.workers)
        # Recheck every frozen scientific input/code before exporting a completed candidate map.
        if load_activity(args.mapping)[4] != inputs or load_sources(ROOT)[1] != source_manifest or any(sha(ROOT / p) != h for p, h in code.items()):
            raise ValueError('frozen inputs/source/code changed during alignment')
        units = export_units(groups, chains, states, readiness, membership, hits)
        dump(args.out / 'unit-map.json', units)
        overlap = {}
        for source in ('ProteinGym', 'Tsuboyama'):
            rs = [r for r in hits if r['kind'] == 'anchor' and r['source']['source'] == source]
            overlap[source] = dict(admitted_query_WTs=sorted({r['query'] for r in rs if r['admitted']}),
                admitted_source_proteins=sorted({r['source']['protein'] for r in rs if r['admitted']}),
                lower_30_query_overlap_WTs=sorted({r['query'] for r in rs if r['family_overlap']}),
                lower_30_query_overlap_components=sorted({membership[r['query']] for r in rs if r['family_overlap']}),
                lower_30_source_proteins=sorted({r['source']['protein'] for r in rs if r['family_overlap']}))
        dump(args.out / 'anchor-overlap.json', overlap)
        result = dict(status='exhaustive_WT_alignment_complete_labels_not_promoted', counts=counts,
            unit_counts=units['counts'], backend=aligner.provenance, code_sha256=code,
            contract_sha256=sha(args.out / 'contract.json'), wall_seconds=time.monotonic()-started,
            output_sha256={name: sha(args.out / name) for name in ('contract.json', 'input-plan.json', 'development-sources.json', 'pair-evidence.jsonl', 'unit-map.json', 'anchor-overlap.json')})
        dump(args.out / 'receipt.json', result)
    except Exception as error:
        dump(args.out / 'resource-receipt.json', dict(status='failed', before=before, after=resources(),
            wall_seconds=time.monotonic()-started, error=f'{type(error).__name__}: {error}'))
        raise
    dump(args.out / 'resource-receipt.json', dict(status='complete', before=before, after=resources(), wall_seconds=time.monotonic()-started))
    print(json.dumps({k: result[k] for k in ('status', 'counts', 'unit_counts', 'wall_seconds')}, indent=2))


if __name__ == '__main__':
    main()
