#!/usr/bin/env python3
"""Independent CPU fits from retained scalar features; never model/token inference.

Default output is results/extensions/information_progression_20261006/.
--smoke fits only one arm/seed on the 2067-row structural panel and estimates
CPU costs; it does not fit the full panel. Full fitting requires --authorize-full.
The CLI runs its extension tests before any fitting. Never resumes/overwrites.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    """Atomic JSON publication, including final receipts; NaNs are forbidden."""
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def archive_path(stem):
    """Append extension without treating dots in model IDs as file suffixes."""
    return stem.with_name(stem.name + '.npz')


def runtime(threads):
    import numpy as np
    import scipy
    import torch
    return dict(device='cpu', threads=threads, python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, torch=str(torch.__version__),
                memory=Path('/proc/meminfo').read_text(), disk_free_bytes=shutil.disk_usage(ROOT).free,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime +
                            resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                blas_environment={k: os.environ.get(k) for k in
                                  ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})


def load_inputs(root, sites_path):
    """Use the existing raw_M and profile/window loaders, not fitted M/P_M."""
    import numpy as np
    from scripts.capability.reporting import export_prediction_details as d
    from scripts.capability.reporting.replay_mutation_predictions import retained_features
    from scripts.capability.context import analyse_crossed_controls as crossed
    from scripts.capability.context import analyse_local_context_gate as gate
    from src.capability.extensions import overlap as o
    from src.capability.extensions import progression as p
    from src.capability.readouts.readout_analysis import sequence_features

    admission = json.loads((root / d.LOCAL).read_text())
    if digest(root / d.MUTATION) != admission['cohort_sha256']:
        raise ValueError('cohort hash mismatch')
    cohort = json.loads((root / d.MUTATION).read_text())
    mapping = {a['assay']: a for a in cohort['assays']}
    if len(mapping) != len(cohort['assays']):
        raise ValueError('duplicate cohort assays')
    assays = [mapping[a] for a in sorted(admission['support']['assay_ids'])]
    samples = []
    for assay in assays:
        for j, mutation in enumerate(assay['mutants']):
            samples.append(dict(assay=assay['assay'], cluster=assay['cluster'], mutation=mutation,
                                variant_index=j, original_index=len(samples),
                                sample_id=d.sample_id(admission['cohort_sha256'], assay['assay'], mutation)))
    aids = np.asarray([r['assay'] for r in samples]); families = np.asarray([r['cluster'] for r in samples])
    y = np.concatenate([a['measured'] for a in assays])
    p.validate_samples(samples, y, aids, families)
    if (len(assays), len(set(families)), len(samples)) != (201, 163, 25728):
        raise ValueError('full admitted support must be 201 assays / 163 families / 25728 rows')
    features, signatures, sources, source_registry = retained_features(root, admission, assays)
    if len(features) != 33 or set(signatures) != set(p.SEEDS):
        raise ValueError('require original 33 arms and three historical seeds')
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', a) for a in features):
        raise ValueError('unsafe arm identifier')
    baseline_rows = [dict(assay=a['assay'], cluster=a['cluster'], mutants=a['mutants'],
                          measured=a['measured'], P=a['profile_scores'],
                          S=sequence_features(a['wildtype'], a['mutants'])) for a in assays]
    store, profile_hashes = crossed.load_profile_store(root/'results/R5/retrieval_bound_20260807', cohort)
    try:
        deviation = gate.build_blocks(baseline_rows, cohort, store, ['wall'])
    finally:
        store['stored'].close()
    # Translate historical sequence S and window wall exactly once at this boundary.
    blocks = {name: np.concatenate([a[key] for a in baseline_rows])
              for name, key in (('B', 'S'), ('P', 'P_block'), ('L', 'wall'))}
    with (gzip.open(sites_path, 'rt') if sites_path.suffix == '.gz' else sites_path.open()) as handle:
        sites = json.load(handle)
    if not isinstance(sites, list):
        raise ValueError('sites must be a flat JSON array')
    structural_hashes = {}
    for site in sites:
        if site.get('status') == 'admitted':
            path, sha = site['source_path'], site['source_sha256']
            if structural_hashes.setdefault(path, sha) != sha:
                raise ValueError('conflicting structural source hashes')
    for path, sha in structural_hashes.items():
        if digest(path) != sha:
            raise ValueError('structural source hash mismatch')
    coverage_path = sites_path.parent/'coverage.json'; coverage_sha = None
    if any(s.get('status') == 'admitted' and not s.get('method') for s in sites):
        coverage = json.loads(coverage_path.read_text()); coverage_sha = digest(coverage_path)
        source_map = {s['sha256']: s for s in coverage['sources']}
        for site in sites:
            if site.get('status') == 'admitted' and not site.get('method'):
                source = source_map[site['source_sha256']]
                if source['path'] != site['source_path']:
                    raise ValueError('structural lineage mismatch')
                site['method'] = source['method']
    selected, exclusions = o.select_rows(assays, sites)
    if (len(selected), len({r['assay'] for r in selected}), len({r['cluster'] for r in selected})) != (2067, 30, 30):
        raise ValueError('structural support must be 2067 strict-single rows / 30 assays / 30 families')
    if any('Tsuboyama' not in r['assay'] for r in selected):
        raise ValueError('expected Tsuboyama structural support')
    index = np.asarray([r['original_index'] for r in selected])
    structural_samples = [samples[i] for i in index]
    structural_blocks = {b: x[index].copy() for b, x in blocks.items()}
    structural_blocks['S'] = o.structural_features(selected)
    panels: dict[str, Any] = dict(full=dict(samples=samples, y=y, aids=aids, families=families, blocks=blocks,
                            index=np.arange(len(samples))),
                  structural=dict(samples=structural_samples, y=y[index], aids=aids[index],
                                  families=families[index], blocks=structural_blocks, index=index))
    for panel in panels.values():
        p.validate_samples(panel['samples'], panel['y'], panel['aids'], panel['families'])
        # Unsupported projected folds raise explicitly; never regenerate partitions.
        panel['projected'] = {s: o.project_folds(signatures[s], panel['families']) for s in p.SEEDS}
    provenance = dict(cohort_sha256=admission['cohort_sha256'],
                      admission_sha256=digest(root/d.LOCAL), source_registry_sha256=source_registry,
                      rank_feature_sources=sources, profile_sources=profile_hashes,
                      max_profile_deviation=deviation, sites_sha256=digest(sites_path),
                      paired_coverage_sha256=coverage_sha, structural_source_sha256=structural_hashes,
                      structural_exclusions=exclusions, full_support_sha256=p.identity_digest(samples))
    return panels, features, signatures, provenance


def verify_cell(path, predictions, records, checks, data, projected):
    """Read back saved sample/target/OOF identities and every realized partition."""
    import numpy as np
    from src.capability.extensions import overlap as o
    from src.capability.extensions import progression as p
    from src.capability.context.local_context import fold_membership
    membership = [(f, tuple(h), tuple(t), tuple(tuple(v) for v in inn)) for f, h, t, inn in projected]
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(predictions) | {'original_index', 'target', 'sample_id'}:
            raise ValueError('OOF archive key disagreement')
        if not np.array_equal(saved['original_index'], data['index']) or not np.array_equal(
                saved['sample_id'], [s['sample_id'] for s in data['samples']]):
            raise ValueError('saved sample identity disagreement')
        if not np.array_equal(saved['target'], o.rerank(data['y'], data['aids'])):
            raise ValueError('saved target disagreement')
        for name, pred in predictions.items():
            if not np.array_equal(saved[name], pred) or p.array_digest(saved[name]) != checks[name]['prediction_sha256']:
                raise ValueError('saved OOF disagreement')
            if fold_membership(records[name]) != membership:
                raise ValueError('saved fold membership disagreement')
    return dict(identity_verified=True, folds_verified=True, prediction_archive_sha256=digest(path))


def run(root, sites, out, *, smoke=False, bootstrap=2000, threads=2, test_evidence=None):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from src.capability.extensions import progression as p
    from src.capability.extensions import overlap as o
    from src.capability.readouts.readout_analysis import ALPHAS

    if not 2 <= threads <= 4:
        raise ValueError('require 2-4 threads')
    if not test_evidence or test_evidence.get('status') != 'passed':
        raise ValueError('successful extension tests required before fitting')
    if bootstrap != 2000:
        raise ValueError('frozen production bootstrap is 2000 draws')
    if out.exists():
        raise ValueError('never overwrite prior outputs; choose a new output root')
    if shutil.disk_usage(out.parent if out.parent.exists() else root).free < 2_000_000_000:
        raise ValueError('insufficient disk headroom')
    torch.set_num_threads(threads)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    save(out/'runtime-start.json', runtime(threads))
    save(out/'tests.json', test_evidence)
    # Include all feature/fitting dependencies, not just extension source bytes.
    names = ['scripts/capability/extensions/fit_information_progression.py',
             'src/capability/extensions/progression.py', 'src/capability/extensions/overlap.py',
             'src/capability/extensions/structure.py',
             'scripts/capability/reporting/replay_mutation_predictions.py',
             'scripts/capability/reporting/export_prediction_details.py',
             'scripts/capability/context/analyse_crossed_controls.py',
             'scripts/capability/context/analyse_local_context_gate.py',
             'scripts/capability/stages/retrieval_bound.py',
             'src/capability/readouts/readout_analysis.py', 'src/capability/context/local_context.py',
             'src/capability/context/crossed_controls.py', 'src/capability/context/profile_increment.py',
             'src/capability/context/profiles.py', 'src/capability/core/amino_acids.py',
             'src/capability/core/protein_properties.py', 'src/capability/models/fitness.py',
             'tests/extensions/test_progression.py']
    hashes = {name: digest(ROOT/name) for name in names}
    with threadpool_limits(limits=threads):
        panels, features, signatures, provenance = load_inputs(root, sites)
        contract = dict(analysis='information_progression_20261006', mode='smoke' if smoke else 'production',
                        provenance=provenance, code_sha256=hashes, column_inventory=p.column_inventory(),
                        registry={panel: p.registry(panel) for panel in panels}, limitations=p.LIMITATIONS,
                        seeds=list(p.SEEDS), models=sorted(features), bootstrap=bootstrap,
                        alphas=list(ALPHAS), outer_folds=5, inner_folds=4,
                        OOF_reuse=False, target='within-panel within-assay standardized ranks',
                        preprocessing='subset first; rerank raw_M and P0; preserve all other historical columns',
                        uncertainty='shared family draws, independent within-panel multiplicity families',
                        panel_support={name: dict(rows=len(data['y']), assays=len(set(data['aids'])),
                                                families=len(set(data['families'])),
                                                support_sha256=p.identity_digest(data['samples']))
                                       for name, data in panels.items()})
        save(out/'contract.json', contract); save(out/'original-memberships.json', signatures)
        for name, data in panels.items():
            folder = out/name; folder.mkdir()
            save(folder/'samples.json', data['samples'])
            save(folder/'projected-memberships.json', data['projected'])
            prepared = p.prepare_blocks({**data['blocks'], 'M': features[sorted(features)[0]][data['index'], None]},
                                        data['aids'], name)
            audits = {b: p.census(x) for b, x in prepared.items() if b != 'M'}
            audits['M_by_arm'] = {a: p.census(o.rerank(x[data['index']], data['aids'])[:, None])
                                  for a, x in sorted(features.items())}
            save(folder/'label-blind-column-census.json', audits)
        completed = []; all_scores = {name: [] for name in panels}
        selected_panels = ['structural'] if smoke else list(panels)
        selected_seeds = p.SEEDS[:1] if smoke else p.SEEDS
        selected_arms = sorted(features)[:1] if smoke else sorted(features)
        expected = {(panel, arm, seed) for panel in selected_panels for seed in selected_seeds
                    for arm in selected_arms}
        timings = {}
        for panel in selected_panels:
            data = panels[panel]
            for seed in selected_seeds:
                for arm in selected_arms:
                    cell_started = time.monotonic()
                    blocks = {**data['blocks'], 'M': features[arm][data['index'], None]}
                    pred, records, checks = p.fit_cell(blocks, data['y'], data['aids'], data['families'],
                                                     signatures[seed], panel, data['samples'])
                    stem = out/panel/f'{arm}-seed{seed}'
                    for name, record in records.items():
                        save(stem.with_name(stem.name + '-' + name + '-folds.json'), record)
                    np.savez_compressed(archive_path(stem), **pred, original_index=data['index'],
                                        target=o.rerank(data['y'], data['aids']),
                                        sample_id=np.asarray([s['sample_id'] for s in data['samples']]))
                    # Verify persisted folds, not only in-memory records.
                    saved_records = {name: json.loads(stem.with_name(stem.name+'-'+name+'-folds.json').read_text())
                                     for name in records}
                    verification = verify_cell(archive_path(stem), pred, saved_records, checks,
                                               data, data['projected'][seed])
                    scores = p.assay_metrics(pred, data['y'], data['aids'], data['families'], arm, seed, panel)
                    all_scores[panel].extend(scores)
                    seconds = time.monotonic() - cell_started
                    save(stem.with_name(stem.name+'-checks.json'), dict(designs=checks, verification=verification,
                                                                     seconds=seconds))
                    completed.append(dict(panel=panel, arm=arm, seed=seed, seconds=seconds, **verification))
                    timings[panel] = seconds
                    save(out/'progress.json', dict(completed=completed, expected_cells=len(expected),
                                                   elapsed_seconds=time.monotonic()-started))
                    print(json.dumps(completed[-1]), flush=True)
        if {(c['panel'], c['arm'], c['seed']) for c in completed} != expected:
            raise ValueError('completed cell identities disagree with requested scope')
        for panel in selected_panels:
            save(out/panel/'assay-metrics.json', all_scores[panel])
            if not smoke:
                save(out/panel/'contrasts.json', p.joint_bootstrap(all_scores[panel], panel,
                     bootstrap=bootstrap, expected_arms=sorted(features)))
        if smoke:
            # Row-linear extrapolation deliberately overcharges small designs and
            # eigensolve overhead; neither an actual full benchmark nor a guarantee.
            full_ratio = len(panels['full']['y']) / len(panels['structural']['y'])
            full_seconds = timings['structural'] * full_ratio * len(p.DESIGNS['full']) / len(p.DESIGNS['structural']) * 99
            structural_seconds = timings['structural'] * 99
            save(out/'cpu-estimate.json', dict(measured_structural_cell_seconds=timings['structural'],
                 smoke_rows=2067, smoke_designs=11, smoke_arm=selected_arms[0], threads=threads,
                 full_panel_fit_seconds_row_scaled=full_seconds,
                 structural_panel_fit_seconds=structural_seconds,
                 both_panels_hours_range=[(full_seconds+structural_seconds)/3600/2,
                                         (full_seconds+structural_seconds)/3600*2],
                 total_nested_fits=99*(8+11), total_ridge_calls=99*(8+11)*25,
                 approximate_design_array_bytes=25728*576*8,
                 note='rough 0.5x-2x wall estimate from real structural fits, not full-panel benchmark; excludes loading/bootstrap and concurrent-host variation; streaming designs, no giant cache'))
    if any(digest(ROOT/name) != sha for name, sha in hashes.items()):
        raise ValueError('code dependency changed during fit')
    save(out/'runtime-end.json', runtime(threads))
    output_hashes = {str(path.relative_to(out)): digest(path) for path in sorted(out.rglob('*')) if path.is_file()}
    save(out/'completion.json', dict(status='smoke_complete' if smoke else 'complete',
                                    tests_passed=True, completed_cells=completed,
                                    expected_cells=len(expected), output_sha256=output_hashes,
                                    elapsed_seconds=time.monotonic()-started,
                                    full_panel_fitted=not smoke))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--sites', type=Path, default=ROOT/'results/extensions/mutation_structure_20261006/structure/annotation/sites.json.gz')
    parser.add_argument('--out', type=Path, default=ROOT/'results/extensions/information_progression_20261006')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--smoke', action='store_true')
    modes.add_argument('--authorize-full', action='store_true', help='explicit authorization for both complete panels')
    parser.add_argument('--threads', type=int, choices=(2, 3, 4), default=2)
    args = parser.parse_args()
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[variable] = str(args.threads)
    # Successful tests are mandatory evidence for the atomic completion receipt.
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(ROOT/'tests/extensions/test_progression.py')],
                          cwd=ROOT, capture_output=True, text=True)
    print(test.stdout, end='', flush=True)
    if test.returncode:
        print(test.stderr, file=sys.stderr)
        raise SystemExit('progression tests failed; no fits launched')
    evidence = dict(status='passed', command='pytest -q tests/extensions/test_progression.py',
                    stdout=test.stdout, test_sha256=digest(ROOT/'tests/extensions/test_progression.py'))
    run(args.root.resolve(), args.sites.resolve(), args.out.resolve(), smoke=args.smoke,
        threads=args.threads, test_evidence=evidence)


if __name__ == '__main__':
    main()
