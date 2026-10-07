#!/usr/bin/env python3
"""Execute bounded-CPU retrospective fixed-OOF endpoint stratification."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def runtime():
    return dict(device='cpu', threads=2, memory=Path('/proc/meminfo').read_text(),
                disk_free_bytes=shutil.disk_usage(ROOT).free,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})


def run(root, out):
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    import src.capability.extensions.phenotype_strata as p
    if out.exists():
        raise ValueError('never overwrite prior output')
    if shutil.disk_usage(root).free < 500_000_000:
        raise ValueError('insufficient disk headroom')
    out.mkdir(parents=True)
    started = time.monotonic()
    save(out/'runtime-start.json', runtime())
    declaration = dict(schema='phenotype_strata_fixed_oof_v1', evidence='explicitly retrospective existing-support sensitivity',
        primary='within-assay Spearman(BMPL fitted OOF) minus Spearman(BPL fitted OOF)',
        aggregation='seed average within assay, equal assay within category/family, equal supported families within category',
        multiplicity='one simultaneous Spearman family: all 33 x 5 supported model/category contrasts',
        bootstrap_design='10000 paired draws across the same 163 original biological family IDs jointly for every contrast',
        secondary_rank_target_mse='not computed; no physical phenotype error claim',
        limitations=['Coarse metadata labels, not scientifically pure task definitions or independently adjudicated endpoints.',
            'All predictions share the fixed mixed training population; no endpoint-specific refits, new blocks, rank-trained new fits, or independent held-out confirmation.',
            'Original progression ridge predictions target standardized ranks; BPL/BMPL are fitted OOF predictions, not raw likelihood or input features.',
            'Baseline qualified on the full anchor, not independently qualified in each category; per-class baseline performance is descriptive positive-control context.',
            'Noise, replicate and condition reliability are unknown; no new noise exclusions or replicate inference.',
            'Seeds averaged, not independent replicates; checkpoints fixed, never bootstrapped; intervals omit training/tuning uncertainty.'])
    save(out/'declaration.json', declaration)
    with threadpool_limits(limits=2):
        contract, samples, scores, hashes = p.load_production(root)
        export_path = root/'manuscript/data/prediction-details/mutation-samples.csv.gz'
        hashes[str(export_path.relative_to(root))] = p.digest(export_path)
        grouping_audit = p.audit_export_groups(samples, pd.read_csv(export_path))
        metadata_path = root/'data/proteingym_raw/DMS_substitutions.csv'
        hashes[str(metadata_path.relative_to(root))] = p.digest(metadata_path)
        raw = pd.read_csv(metadata_path)
        if raw.DMS_id.duplicated().any():
            raise ValueError('duplicate assay metadata')
        fields = ['DMS_id','coarse_selection_type','selection_assay','selection_type','raw_DMS_phenotype_name','raw_DMS_directionality','UniProt_ID','ProteinGym_version']
        metadata = pd.DataFrame(raw[fields]).rename(columns={'DMS_id':'assay','coarse_selection_type':'category'})
        support = pd.DataFrame(samples)
        support['strict_single_substitution'] = support.mutation.map(lambda m: bool(re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY][1-9][0-9]*[ACDEFGHIKLMNPQRSTVWY]', m)))
        coverage = support.groupby(['assay','cluster'], as_index=False).agg(rows=('sample_id', 'size'))
        coverage = coverage.merge(metadata, on='assay', how='left', validate='one_to_one')
        if coverage.category.isna().any() or set(coverage.category) != set(p.CATEGORIES):
            raise ValueError('missing or unsupported category metadata')
        coverage.to_csv(out/'assay-coverage-metadata.csv', index=False)
        excluded = metadata.loc[~metadata.assay.isin(coverage.assay)].copy()
        excluded['reason'] = 'outside exact production 201-assay all-model OOF support; metadata registry is not score coverage'
        excluded.to_csv(out/'metadata-outside-production.csv', index=False)
        support = support.merge(coverage[['assay','category']], on='assay', validate='many_to_one', sort=False)
        support.to_csv(out/'raw-sample-intersection.csv', index=False)
        pd.DataFrame(scores).to_csv(out/'assay-seed-metrics.csv', index=False)
        families = sorted(set(support.cluster))
        keys, matrices, averaged = p.family_matrix(scores, coverage, contract['models'], families)
        averaged.to_csv(out/'assay-seed-average.csv', index=False)
        labels = [f'{a}/{c}' for a,c in keys]
        pd.DataFrame(matrices['contrast'], index=pd.Index(families, name='original_family_id'), columns=pd.Index(labels)).to_csv(out/'model-group-contrast-matrix.csv')
        np.savez_compressed(out/'family-matrices.npz', original_family_id=np.asarray(families), arm=np.asarray([a for a,c in keys]), category=np.asarray([c for a,c in keys]), **matrices)
        stats, boot = p.shared_bootstrap(matrices['contrast'])
        np.savez_compressed(out/'shared-bootstrap-estimates.npz', estimates=boot)
        results = []
        for i,(arm,category) in enumerate(keys):
            results.append(dict(arm=arm, category=category, supported_families=stats['supported_families'][i],
                point=stats['point'][i], se=stats['se'][i], pointwise_low=stats['pointwise_interval'][i][0], pointwise_high=stats['pointwise_interval'][i][1],
                simultaneous_low=stats['simultaneous_interval'][i][0], simultaneous_high=stats['simultaneous_interval'][i][1],
                baseline_spearman=float(np.nanmean(matrices['baseline_spearman'][:,i])), augmented_spearman=float(np.nanmean(matrices['augmented_spearman'][:,i]))))
        result_frame = pd.DataFrame(results)
        result_frame.to_csv(out/'model-category-statistics.csv', index=False)
        summaries, family_coverage = [], []
        for category in p.CATEGORIES:
            sub = coverage.loc[coverage.category==category]; cells=result_frame.loc[result_frame.category==category]
            present = sorted(set(sub.cluster)); absent = sorted(set(families)-set(present))
            for family in families:
                local=sub.loc[sub.cluster==family]
                family_coverage.append(dict(category=category, original_family_id=family, present=bool(len(local)), assays=len(local), rows=int(local.rows.sum())))
            summaries.append(dict(category=category, rows=int(sub.rows.sum()), assays=len(sub), proteins=int(sub.UniProt_ID.nunique()), families=len(present),
                missing_original_families=absent, assay_rows_range=[int(sub.rows.min()),int(sub.rows.max())],
                increment_range=[float(cells.point.min()),float(cells.point.max())],
                baseline_spearman_range=[float(cells.baseline_spearman.min()),float(cells.baseline_spearman.max())],
                positive_point=int((cells.point>0).sum()), positive_pointwise=int((cells.pointwise_low>0).sum()), positive_simultaneous=int((cells.simultaneous_low>0).sum()),
                negative_simultaneous=int((cells.simultaneous_high<0).sum())))
        pd.DataFrame(family_coverage).to_csv(out/'family-category-coverage.csv', index=False)
        for relative in ['src/capability/extensions/phenotype_strata.py','scripts/capability/extensions/analyse_phenotype_strata.py','tests/extensions/test_phenotype_strata.py']:
            hashes[relative]=p.digest(root/relative)
        save(out/'source-hashes.json', hashes)
        save(out/'statistics.json', dict(**declaration, status='complete', exact_support=dict(rows=len(samples), assays=len(coverage), biological_families=len(families), models=33, seeds=list(p.SEEDS), archives=99),
             strict_single_substitution_rows=int(support.strict_single_substitution.sum()), mixed_substitution_support=True,
             grouping_audit=grouping_audit,
             metadata_registry_assays=len(metadata), metadata_not_scored_assays=len(excluded), exclusions='No new row exclusions; exact production support only',
             categories=summaries, bootstrap=stats, full_model_category_statistics=results,
             numerical_helper='Existing simultaneous_bands requires all-finite matrices, unsuitable for structural category absence; local ratio-mean helper preserves joint biological draws.'))
    save(out/'runtime-end.json', dict(**runtime(), elapsed_seconds=time.monotonic()-started))
    save(out/'completion.json', dict(status='complete', output_sha256={str(path.relative_to(out)):p.digest(path) for path in sorted(out.iterdir()) if path.is_file()}))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, default=ROOT/'results/extensions/phenotype_followups_20261007/strata')
    args=parser.parse_args()
    run(args.root, args.out)
