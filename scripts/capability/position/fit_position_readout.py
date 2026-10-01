#!/usr/bin/env python3
"""Decompose R1 likelihood prediction above frozen strong local controls.

No checkpoint forward: retained scored-token vectors and matching archive rows
are validated and partitioned, then refitted on exact same-support controls.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/capability'))
from scripts.capability.position.analyse_position_terms import Packer, load_anchor, alignment_table, sha
from scripts.capability.context.analyse_local_context_gate import build_blocks, crossed, CODE_FILES
from src.capability.core.io import write_json
from src.capability.position.position_readout import evaluate
from src.capability.position.paired_position import attach_paired
from src.capability.position.position_terms import ALIGNMENT, SPAN_ALIGNMENT
from src.capability.readouts.readout_extraction import load_readout_arm
from src.capability.readouts.readout_analysis import sequence_features
from src.capability.context.profile_increment import standardized_rank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True)
    p.add_argument('--wave', required=True, type=Path)
    p.add_argument('--support', required=True, type=Path)
    p.add_argument('--cohort', required=True, type=Path)
    p.add_argument('--profile-store', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--partition', choices=('strict','span'), default='strict')
    p.add_argument('--paired-terms', type=Path)
    p.add_argument('--numerical-sensitivity', action='store_true')
    p.add_argument('--fold-seeds', nargs='+', type=int, default=[20260923,20260924,20260925])
    args = p.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    support = json.loads(args.support.read_bytes())
    cohort = json.loads(args.cohort.read_bytes())
    source_sha = {str(args.cohort): sha(args.cohort), str(args.support): sha(args.support)}
    if support['cohort_sha256'] != source_sha[str(args.cohort)]:
        raise ValueError('support cohort digest mismatch')
    archive_manifest = Path(support['archived_manifests'][args.arm])
    wave_manifest = args.wave / f'manifest_{args.arm}.json'
    for path in (archive_manifest, wave_manifest):
        source_sha[str(path)] = sha(path)
        manifest = json.loads(path.read_bytes())
        if manifest['identity']['cohort_sha256'] != source_sha[str(args.cohort)]:
            raise ValueError('manifest cohort digest mismatch')
        if manifest['status'] != 'complete' or manifest['identity']['arm'] != args.arm:
            raise ValueError('incomplete or wrong-arm manifest')
        mapping = {row['assay']: row for row in manifest['assays']}
        if set(support['assays']) - set(mapping):
            raise ValueError('manifest misses declared assay support')
        for assay in support['assays']:
            item = mapping[assay]; file = path.parent / item['file']
            digest = sha(file)
            if digest != item['sha256']:
                raise ValueError(f'{assay}: source archive hash mismatch')
            source_sha[str(file)] = digest
    packer = Packer(load_readout_arm(args.arm, None, device=None), partition_mode=args.partition)
    loaded = load_anchor(args.wave, archive_manifest.parent, args.arm, args.cohort,
                         set(support['assays']), [], packer, workers=args.workers)
    if max(map(abs, loaded['t1'])) != 0.0:
        raise ValueError('same-forward retention did not reconstruct exactly')
    if any(error > bound for error, bound in loaded['closure']):
        raise ValueError('partition closure exceeds declared arithmetic bound')
    paired = None
    if args.paired_terms is not None:
        if args.arm != 'progen3-3b' or args.partition != 'strict':
            raise ValueError('paired scoring is declared only for strict ProGen3-3B')
        paired = attach_paired(loaded,cohort,packer,args.paired_terms,
                               cohort_sha=source_sha[str(args.cohort)],
                               support_sha=source_sha[str(args.support)])
    by_assay = {row['assay']:row for row in cohort['assays']}
    rows, audits, archive_delta = [], [], []
    for row in loaded['rows']:
        c = by_assay[row['assay']]
        entries = row['entries']
        if len(entries) != len(c['mutants']):
            raise ValueError('cohort and retained row counts differ')
        single = np.array([e['substitutions']==1 for e in entries])
        keep = np.array([e['alignment']['aligned'] and e['own_scored'] for e in entries]) & single
        measured = np.asarray([e['measured'] for e in entries])
        if not np.array_equal(measured, c['measured']):
            raise ValueError('retained labels differ from cohort')
        archive_delta.extend(e['full']-e['archived_full'] for e in entries)
        ranks = standardized_rank(measured[single]) if single.any() else np.empty(0)
        retained = keep[single]
        audits.append(dict(assay=row['assay'], cluster=row['cluster'], singles=int(single.sum()),
                           retained=int(keep.sum()), fitted=bool(keep.sum()>=3),
                           own_residue_width_max=max((e['own_residue_width'] for e,k in zip(entries,keep) if k),default=0),
                           own_residue_width_mean=float(np.mean([e['own_residue_width'] for e,k in zip(entries,keep) if k])) if keep.any() else None,
                           retained_mutants=[m for m,k in zip(c['mutants'],keep) if k],
                           retained_target_rank_mean=float(ranks[retained].mean()) if retained.any() else None,
                           excluded_target_rank_mean=float(ranks[~retained].mean()) if (~retained).any() else None,
                           alignment=alignment_table([e for e in entries if e['substitutions']==1],
                                                     rule=SPAN_ALIGNMENT if args.partition=='span' else ALIGNMENT)))
        if keep.sum()<3:
            continue
        # Build controls before restriction: sequence-derived features depend only
        # on that row's sequence, never on labels or which other variants survive.
        item = dict(assay=row['assay'], cluster=row['cluster'], mutants=c['mutants'],
                    measured=measured, P=np.asarray(c['profile_scores']),
                    S=sequence_features(c['wildtype'],c['mutants']), keep=keep)
        for key in ('own','downstream','full','upstream'):
            item[key]=np.asarray([e[key] for e in entries], dtype=np.float64)
        if paired is not None:
            item['historical_full']=np.asarray([e['archived_full'] for e in entries],dtype=np.float64)
        rows.append(item)
    store, store_sha = crossed.load_profile_store(args.profile_store, cohort)
    build_blocks(rows, cohort, store, ['wall','rf3'])
    for row in rows:
        keep=row.pop('keep')
        row['mutants']=[m for m,k in zip(row['mutants'],keep) if k]
        for key in ('measured','P','S','P_block','wall','rf3','own','downstream','full','upstream'):
            row[key]=row[key][keep]
        if paired is not None:
            row['historical_full']=row['historical_full'][keep]
    args.out.mkdir(parents=True, exist_ok=True)
    upstream = np.concatenate([r['upstream'] for r in rows])
    identity = dict(schema='position_controlled_v1', arm=args.arm, partition=args.partition,paired_measurement=paired, numerical_sensitivity=args.numerical_sensitivity, source_sha256=source_sha,
                    profile_store_sha256=store_sha, support=audits,
                    support_rule='single substitutions with scored mutation and token-grid alignment, at least three retained rows per assay; no phenotype-dependent selection; conditional estimand with no minimum coverage extrapolation',
                    retention_worst_nats=max(map(abs,loaded['t1'])),
                    historical_delta_max_abs_nats=float(np.max(np.abs(archive_delta))),
                    historical_delta_nonzero=int(np.count_nonzero(archive_delta)),
                    closure_max_nats=paired['closure_max_nats'] if paired is not None else max(e for e,b in loaded['closure']),
                    aligned_upstream_max_abs_nats=float(np.max(np.abs(upstream))),
                    analysis_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [*(ROOT/name for name in CODE_FILES),Path(__file__),ROOT/'scripts/capability/position/analyse_position_terms.py',ROOT/'src/capability/position/position_terms.py',ROOT/'src/capability/position/position_readout.py',ROOT/'src/capability/position/paired_position.py']},
                    runtime=dict(torch=torch.__version__, numpy=np.__version__, device=args.device))
    for seed in args.fold_seeds:
        print(json.dumps(dict(arm=args.arm,fold_seed=seed,status='fitting')),flush=True)
        report,predictions=evaluate(rows,device=args.device,fold_seed=seed,
                                    numerical_sensitivity=args.numerical_sensitivity)
        if args.partition == 'span':
            report['estimand'] = 'family-held predictive rank increment on single substitutions admitting an exact unchanged-prefix / mutation-associated token span / rejoined-suffix partition'
            report['interpretation'] = 'own denotes the entire mutation-associated retokenized span, which may contain different token counts between states; downstream denotes the rejoined identical residue/token suffix'
            identity['support_rule'] = 'single substitutions with scored mutation in both states, identical unscored conditioning, and an exact common-prefix/rejoined-suffix partition, at least three retained rows per assay'
        stem=f'position_controlled_{args.arm}_fold{seed}'
        arrays=args.out/f'{stem}.npz'
        np.savez_compressed(arrays,**predictions)
        write_json(args.out/f'{stem}.json',dict(identity,**report,prediction_sha256=sha(arrays)))
    print(json.dumps(dict(arm=args.arm,status='complete')),flush=True)

if __name__=='__main__':
    main()
