#!/usr/bin/env python3
"""CPU-only descriptive RHO remeasurement; read existing outputs, never fit/infer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
COHORT = 'results/R5/context_mutation_rescue_20260923/cohort.json'
ADMISSION = 'results/R5/local_context_20260923/20260923233257_e429ce7f31e4/lcgp_admission/local_context_measurement.json'
SOURCE_REGISTRY = 'results/R2/readout_expansion_20260923/complete/admission/source_registry.json'
DEFAULT_MEMBRANE = 'results/extensions/phenotype_followups_20261007/membrane/source-anchored-sgca'
DEFAULT_PRODUCTION = 'results/extensions/information_progression_20261006'
DEFAULT_OUT = 'results/extensions/phenotype_followups_20261007/remeasurement'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_rows(path, rows):
    with path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')


def resources():
    return {'device': 'CPU', 'threads': 2, 'python': platform.python_version(),
            'memory': Path('/proc/meminfo').read_text(), 'disk_free_bytes': shutil.disk_usage(ROOT).free,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'cpu_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            'environment': {k: os.environ.get(k) for k in
                            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')},
            'no_H200': True, 'no_fits': True, 'no_inference': True}


def analyse(root, membrane, production, out):
    import numpy as np
    from scipy.stats import rankdata
    from src.capability.extensions import phenotype_remeasurement as r
    from src.capability.extensions.overlap import project_folds
    from src.capability.context.local_context import fold_membership

    bindings = {}

    def bind(path, expected=None):
        digest = r.file_sha(path)
        if expected is not None:
            r.require(digest == expected, f'producer hash mismatch: {path}')
        bindings[str(path.relative_to(root))] = digest
        return digest

    receipt = r.read_json(membrane / 'verification-receipt.json')
    r.require(receipt['status'] == 'passed', 'accepted membrane QC required')
    product = r.read_json(membrane / 'output-hashes.json')
    bind(membrane / 'output-hashes.json')
    required = {'channel-rows.jsonl', 'canonical-mutant-states.jsonl', 'canonical-WT-manifest.json',
                'source-manifest.json', 'channel-manifest.json', 'qc-summary.json',
                'anchor-comparison.json', 'verification-receipt.json'}
    r.require(required <= set(product['outputs']), 'membrane output bindings incomplete')
    for path, expected in product['code'].items():
        bind(root / path, expected)
    for path, expected in product['outputs'].items():
        r.require(Path(path).name == path, 'unsafe membrane output binding')
        bind(membrane / path, expected)
    sources = r.read_json(membrane / 'source-manifest.json')
    for source in sources:
        path = r.measurement_source_path(root, source['path'])
        bind(path, source['sha256'])
    source_hashes = {s['sha256'] for s in sources}
    rows = r.read_rows(membrane / 'channel-rows.jsonl')
    r.require(all(row['source_sha256'] in source_hashes for row in rows if row['protein'] == 'RHO'),
              'measurement source hash not in verified source manifest')
    external = r.read_json(membrane / 'canonical-WT-manifest.json')['RHO']
    r.require(r.text_sha(external['wt']) == external['wt_sha256'], 'external WT hash mismatch')

    for path in ('contract.json', 'completion.json', 'original-memberships.json',
                 'full/samples.json', 'full/projected-memberships.json'):
        bind(production / path)
    contract = r.read_json(production / 'contract.json')
    completion = r.read_json(production / 'completion.json')
    r.require(contract['mode'] == 'production' and completion['status'] == 'complete'
              and completion['tests_passed'], 'production not complete/verified')
    r.require(len(contract['models']) == 33 and len(set(contract['models'])) == 33
              and contract['seeds'] == [20260923, 20260924, 20260925], 'require all 33 models / three original seeds')
    for path, expected in contract['code_sha256'].items():
        bind(root / path, expected)
    provenance = contract['provenance']
    cohort_sha = bind(root / COHORT, provenance['cohort_sha256'])
    bind(root / ADMISSION, provenance['admission_sha256'])
    bind(root / SOURCE_REGISTRY, provenance['source_registry_sha256'])
    admission = r.read_json(root / ADMISSION)
    r.require(admission['cohort_sha256'] == cohort_sha, 'admission cohort mismatch')
    cohort = r.read_json(root / COHORT)
    assays = {a['assay']: a for a in cohort['assays']}
    r.require(len(assays) == len(cohort['assays']), 'duplicate original assay')
    samples = r.read_json(production / 'full/samples.json')
    r.require(r.identity_sha(samples) == provenance['full_support_sha256'], 'sample support hash mismatch')
    rebuilt = []
    targets = []
    for aid in sorted(admission['support']['assay_ids']):
        assay = assays[aid]
        measured = np.asarray(assay['measured'], float)
        r.require(len(measured) == len(assay['mutants']) and np.isfinite(measured).all(), 'old label support mismatch')
        ranks = rankdata(measured)
        scaled = (ranks - ranks.mean()) / ranks.std() if ranks.std() else np.zeros(len(ranks))
        targets.extend(scaled)
        for j, mutation in enumerate(assay['mutants']):
            rebuilt.append({'assay': aid, 'cluster': assay['cluster'], 'mutation': mutation,
                            'variant_index': j, 'original_index': len(rebuilt),
                            'sample_id': r.identity_sha([cohort_sha, aid, mutation])})
    r.require(samples == rebuilt and len(samples) == 25728, 'original sample/mutation/order identity mismatch')
    expected_target = np.asarray(targets, float)
    original = [s for s in samples if s['assay'] == r.ASSAY]
    r.require(len(original) == 128, 'expected original 128 RHO states')
    assay = assays[r.ASSAY]
    r.require(len(assay['mutants']) == len(assay['sequences']) == 128, 'original sequences missing')
    for sample, saved_sequence in zip(original, assay['sequences']):
        r.require(r.state(assay['wildtype'], sample['mutation'])['mutated_sequence'] == saved_sequence,
                  'original saved full mutant sequence mismatch')
    r.require(len({s['cluster'] for s in original}) == 1, 'RHO must have one original family')
    family = original[0]['cluster']
    registry, matched, coverage = r.join_measurements(assay['wildtype'], external['wt'], original,
                                                     r.read_rows(membrane / 'canonical-mutant-states.jsonl'), rows)
    write_rows(out / 'original-state-coverage.jsonl', registry)
    write_rows(out / 'paired-method-row-registry.jsonl', matched)
    save(out / 'coverage.json', {'original_states': 128, 'primary_BOTH_method_states': len(matched),
                               'per_method': coverage, 'primary_denominator': 'identical BOTH-method accepted intersection with original states'})
    indices = np.asarray([s['original_index'] for s in matched])
    measured = {c: [s['methods'][c]['row']['score'] for s in matched] for c in r.CHANNELS}
    old_labels = [assay['measured'][s['variant_index']] for s in matched]
    measurement = {'matched_states': len(matched), 'method1': 'surface antibody', 'method2': 'membrane proximity',
                   'method_rank_correlation': r.rho(measured[r.CHANNELS[0]], measured[r.CHANNELS[1]]),
                   'old_label_rank_relation': {c: r.rho(old_labels, measured[c]) for c in r.CHANNELS},
                   'author_method_discordant_matched_states': sum(
                       bool(s['methods'][r.CHANNELS[0]]['row']['quality']['source_method_discordance']) for s in matched),
                   'uncertainty': 'original method-specific SE retained in paired row registry, descriptive only',
                   'source_quality': '2025 public RHO supplementary normalized method scores; method-specific replication reported separately, not independent proteins; per-variant counts not supplied; CC BY 4.0 Zenodo, GitHub lacks LICENSE',
                   'source_replication': r.measurement_replication(r.read_json(membrane / 'channel-manifest.json')),
                   'genotype_vs_protein_state': 'single amino-acid substitutions of full native RHO joined by WT and mutant sequence hashes; exact protein state, not DNA genotype or engineered reporter construct identity'}
    save(out / 'measurement-relations.json', measurement)

    original_memberships = r.read_json(production / 'original-memberships.json')
    projected = r.read_json(production / 'full/projected-memberships.json')
    families = [s['cluster'] for s in samples]
    # Bind original readout producer files and independently compare historical splits.
    sources = provenance['rank_feature_sources']
    r.require({(s['arm'], s['seed']) for s in sources} == {
        (model, seed) for model in contract['models'] for seed in contract['seeds']}, 'original source cells incomplete')
    historical_registry = r.read_json(root / SOURCE_REGISTRY)['files']
    source_bundle = root / 'results/R2/readout_expansion_20260923/complete'
    representatives = {}
    for source in sources:
        path = root / source['path']
        registered = historical_registry[str(path.relative_to(source_bundle))]['sha256']
        r.require(source['sha256'] == registered, 'original source registry hash disagreement')
        bind(path, registered)
        if source['seed'] not in representatives:
            report = r.read_json(path)
            memberships = [fold_membership(records) for records in report['folds'].values()]
            normal = json.loads(json.dumps(memberships[0]))
            r.require(all(m == memberships[0] for m in memberships), 'historical original design fold disagreement')
            r.require(normal == original_memberships[str(source['seed'])], 'historical family fold signature mismatch')
            representatives[source['seed']] = source['path']
    for seed in contract['seeds']:
        expected = json.loads(json.dumps(project_folds(original_memberships[str(seed)], families)))
        r.require(expected == projected[str(seed)], 'original-to-full projected memberships mismatch')
    cells = []
    prediction_bindings = []
    completed = {(c['arm'], c['seed']): c for c in completion['completed_cells'] if c['panel'] == 'full'}
    r.require(set(completed) == {(m, s) for m in contract['models'] for s in contract['seeds']}, 'production full cells incomplete')
    sample_ids = np.asarray([s['sample_id'] for s in samples])
    for model in sorted(contract['models']):
        for seed in contract['seeds']:
            stem = production / 'full' / f'{model}-seed{seed}'
            archive = stem.with_name(stem.name + '.npz')
            checks_path = stem.with_name(stem.name + '-checks.json')
            bind(checks_path)
            checks = r.read_json(checks_path)
            verified = checks['verification']
            r.require(verified['identity_verified'] and verified['folds_verified'], 'unverified OOF cell')
            entry = completed[(model, seed)]
            r.require(entry['identity_verified'] and entry['folds_verified'] and
                      entry['prediction_archive_sha256'] == verified['prediction_archive_sha256'], 'completion binding mismatch')
            archive_sha = bind(archive, verified['prediction_archive_sha256'])
            vectors = {}
            design_bindings = {}
            with np.load(archive, allow_pickle=False) as saved:
                r.require(np.array_equal(saved['original_index'], np.arange(len(samples))) and
                          np.array_equal(saved['sample_id'], sample_ids), 'prediction sample identity mismatch')
                r.require(np.array_equal(saved['target'], expected_target), 'original label-scale target mismatch')
                for design in ('BPL', 'BMPL'):
                    check = checks['designs'][design]
                    vector = saved[design]
                    r.require(vector.shape == (len(samples),) and np.isfinite(vector).all(), 'invalid prediction vector')
                    r.require(r.array_sha(vector) == check['prediction_sha256'] and
                              check['support_sha256'] == r.identity_sha(samples) and
                              check['target_sha256'] == r.array_sha(expected_target) and
                              check['membership_sha256'] == r.identity_sha(projected[str(seed)]) and
                              check['rows'] == len(samples) and check['design'] == list(design), 'prediction/design/support binding mismatch')
                    fold_path = stem.with_name(stem.name + f'-{design}-folds.json')
                    fold_sha = bind(fold_path)
                    held_fold = r.verify_fold_records(r.read_json(fold_path), projected[str(seed)], family)
                    vectors[design] = vector[indices].copy()
                    design_bindings[design] = {'producer_checks': check, 'fold_file_sha256': fold_sha,
                                               'held_family': family, 'held_fold': held_fold,
                                               'matched_prediction_sha256': r.array_sha(vectors[design])}
            contrast = r.paired_contrasts(vectors['BPL'], vectors['BMPL'], measured[r.CHANNELS[0]], measured[r.CHANNELS[1]])
            cells.append({'model': model, 'seed': seed, 'matched_states': len(matched), **contrast})
            prediction_bindings.append({'model': model, 'seed': seed, 'archive_sha256': archive_sha,
                                        'checks_sha256': r.file_sha(checks_path), 'designs': design_bindings,
                                        'matched_ordered_sample_ids': [s['sample_id'] for s in matched]})
    save(out / 'prediction-bindings.json', prediction_bindings)
    save(out / 'per-model-seed-method-contrasts.json', cells)
    summary = r.descriptive_summary(cells)
    save(out / 'summary.json', summary)
    save(out / 'source-bindings.json', {'verified_file_sha256': bindings, 'historical_fold_representatives': representatives,
                                       'original_cohort_sha256': cohort_sha, 'RHO_WT_sha256': external['wt_sha256']})
    save(out / 'analysis-contract.json', {'analysis': 'fixed-readout external RHO remeasurement sensitivity',
                                        'evaluation': 'average-tie Spearman within identical matched states; predictions unchanged',
                                        'contrasts': 'BMPL minus BPL; method2 contrast minus method1 contrast; arithmetic mean over original seeds',
                                        'limitations': r.LIMITATIONS, 'model_count': 33, 'seed_count': 3,
                                        'protein_count': 1, 'family_count': 1, 'optional_M': 'not evaluated'})
    lines = ['# Fixed-readout RHO remeasurement sensitivity', '',
             f"Original RHO states: 128; BOTH-method primary matched states: {len(matched)}. All 33 models and three historical seeds (99 prediction cells; 198 method comparisons) are retained.", '',
             f"Measured method Spearman: {measurement['method_rank_correlation']:.6f}. Old-label relations: method1 {measurement['old_label_rank_relation']['RHO_method1']:.6f}; method2 {measurement['old_label_rank_relation']['RHO_method2']:.6f}.", '',
             'Per-method original coverage is reported separately in coverage.json; all correlations below use the same BOTH-method denominator. Measurement SE, quality flags, source rows and source hashes are retained without noise resampling.', '',
             '| Model | Mean seed Δρ method1 | Mean seed Δρ method2 | Method2 − method1 Δρ |',
             '|---|---:|---:|---:|']
    for row in summary['models']:
        means = row['mean_seed_methods']
        lines.append(f"| {row['model']} | {means['RHO_method1']['difference_BMPL_minus_BPL']:.6f} | {means['RHO_method2']['difference_BMPL_minus_BPL']:.6f} | {row['mean_seed_method_contrast_difference_method2_minus_method1']:.6f} |")
    lines += ['', 'All positive and negative signals are shown without winner selection. Seed means are descriptive, not iid checkpoint tests.', '', *r.LIMITATIONS]
    (out / 'evidence-report.md').write_text('\n'.join(lines) + '\n')
    return {'original_states': 128, 'primary_matched_states': len(matched), 'prediction_cells': len(cells),
            'method_comparisons': 2 * len(cells), 'coverage': coverage, 'measurement': measurement,
            'sign_counts_mean_seed': summary['sign_counts_mean_seed']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--membrane', type=Path)
    parser.add_argument('--production', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    from src.capability.extensions import phenotype_remeasurement as r
    root = args.root.resolve()
    out = r.project_path(root, args.out or DEFAULT_OUT)
    membrane = r.project_path(root, args.membrane or DEFAULT_MEMBRANE)
    production = r.project_path(root, args.production or DEFAULT_PRODUCTION)
    r.require(membrane.is_relative_to(root) and production.is_relative_to(root),
              'prepared input directories must be within the project root')
    if out.exists():
        parser.error('never overwrite previous outputs; select a new output directory')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if shutil.disk_usage(root).free < 1_000_000_000:
        parser.error('insufficient disk headroom')
    out.mkdir(parents=True)
    start = time.monotonic()
    save(out / 'resource-start.json', resources())
    try:
        result = subprocess.run([sys.executable, '-m', 'pytest', str(root / 'tests/extensions/test_phenotype_remeasurement.py'), '-q'],
                                cwd=root, capture_output=True, text=True, check=False)
        (out / 'test-results.txt').write_text(result.stdout + result.stderr)
        r.require(result.returncode == 0, 'remeasurement tests failed')
        save(out / 'test-receipt.json', {'status': 'passed', 'command': 'ct python -m pytest tests/extensions/test_phenotype_remeasurement.py -q',
                                         'exit_code': result.returncode, 'output_sha256': r.file_sha(out / 'test-results.txt')})
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            counts = analyse(root, membrane, production, out)
        implementation = {p: r.file_sha(root / p) for p in (
            'src/capability/extensions/phenotype_remeasurement.py',
            'scripts/capability/extensions/analyse_rho_remeasurement.py',
            'tests/extensions/test_phenotype_remeasurement.py')}
        save(out / 'resource-end.json', resources())
        output_hashes = {p.name: r.file_sha(p) for p in sorted(out.iterdir()) if p.is_file()}
        save(out / 'completion.json', {'status': 'complete_descriptive_fixed_readout_remeasurement',
                                       'seconds': time.monotonic() - start, 'counts_and_numbers': counts,
                                       'implementation_sha256': implementation, 'output_sha256': output_hashes,
                                       'tests_passed': True, 'no_fits_or_inference': True})
        print(json.dumps(counts, indent=2, allow_nan=False))
    except Exception as error:
        save(out / 'resource-end.json', resources())
        save(out / 'completion.json', {'status': 'blocked', 'error': f'{type(error).__name__}: {error}',
                                       'seconds': time.monotonic() - start, 'no_fits_or_inference': True})
        raise


if __name__ == '__main__':
    main()
