#!/usr/bin/env python3
"""R1 statistical-baseline ladder: reuse the frozen fits, add only what is new.

Every design is the historical nested ridge on the original three seeded
five-outer/four-inner family partitions over the complete 25,728-row support.
``M``, ``MP`` (=P+M), ``BPL`` (=C0) and ``BMPL`` (=C0+M) are read back from
``results/extensions/information_progression_20261006`` and admitted only after
their archive digest, sample identities, target, realized folds and per-assay
metrics agree with that frozen run. Everything else is fitted here: the new
``P`` arm, the Potts/GEMME additions and the C0+E+G controls.

The declared 396-hypothesis family (6 control sets x 33 models x 2 directions)
per metric needs both new scalars. ``--stage profile_only`` is therefore a
declared partial pilot: P/M/P+M and the original C0 control only, always
published as partial, never as the complete ladder. It also refits ``M`` and
``MP`` for every model it covers and requires bit-exact reproduction of the
archives, which is the reuse gate the full stage demands via ``--reuse-pilot``.

The CLI runs its extension tests before any fitting. Output directories are
never reused or overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.capability.extensions.fit_information_progression import (  # noqa: E402
    archive_path, digest, runtime, save)

SOURCE = ROOT/'results/extensions/information_progression_20261006'
SCORES = ROOT/'results/extensions/statistical_ladder_20261106'
PROFILE_STORE = 'results/R5/retrieval_bound_20260807'
TOLERANCE = 1e-12
MISSING_SCORES = 'the full ladder needs the complete Potts and GEMME score tables'
MISSING_PILOT = 'the full ladder needs a passing --reuse-pilot receipt'
CODE_DEPENDENCIES = (
    'scripts/capability/extensions/fit_statistical_ladder.py',
    'src/capability/extensions/statistical_ladder.py',
    'src/capability/extensions/bidirectional.py',
    'src/capability/extensions/responses.py',
    'src/capability/extensions/progression.py',
    'src/capability/extensions/overlap.py',
    'scripts/capability/position/position_simultaneous.py',
    'scripts/capability/reporting/replay_mutation_predictions.py',
    'scripts/capability/reporting/export_prediction_details.py',
    'scripts/capability/context/analyse_crossed_controls.py',
    'scripts/capability/context/analyse_local_context_gate.py',
    'scripts/capability/stages/retrieval_bound.py',
    'src/capability/readouts/readout_analysis.py',
    'src/capability/context/local_context.py',
    'src/capability/context/crossed_controls.py',
    'src/capability/context/profile_increment.py',
    'src/capability/context/profiles.py',
    'src/capability/core/amino_acids.py',
    'src/capability/core/protein_properties.py',
    'src/capability/models/fitness.py',
    'tests/extensions/test_statistical_ladder.py',
)


def load_json(path):
    """Read one required JSON artifact; missing or malformed is never a fallback."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'{path} is missing')
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f'{path} is not valid JSON') from error


def load_support(root, source):
    """Rebuild the frozen full panel and bind every piece to its declared source."""
    import numpy as np
    from scripts.capability.reporting import export_prediction_details as d
    from scripts.capability.reporting.replay_mutation_predictions import retained_features
    from scripts.capability.context import analyse_crossed_controls as crossed
    from scripts.capability.context import analyse_local_context_gate as gate
    from src.capability.extensions import progression as p
    from src.capability.readouts.readout_analysis import sequence_features

    contract = load_json(source/'contract.json')
    completion = load_json(source/'completion.json')
    if completion.get('status') != 'complete' or completion.get('full_panel_fitted') is not True:
        raise ValueError('the frozen source run is not a complete full-panel fit')
    admission = load_json(root/d.LOCAL)
    if digest(root/d.MUTATION) != admission['cohort_sha256']:
        raise ValueError('cohort hash mismatch')
    cohort = load_json(root/d.MUTATION)
    mapping = {assay['assay']: assay for assay in cohort['assays']}
    if len(mapping) != len(cohort['assays']):
        raise ValueError('duplicate cohort assays')
    assays = [mapping[name] for name in sorted(admission['support']['assay_ids'])]
    samples = []
    for assay in assays:
        for index, mutation in enumerate(assay['mutants']):
            samples.append(dict(assay=assay['assay'], cluster=assay['cluster'], mutation=mutation,
                                variant_index=index, original_index=len(samples),
                                sample_id=d.sample_id(admission['cohort_sha256'], assay['assay'], mutation)))
    aids = np.asarray([row['assay'] for row in samples])
    families = np.asarray([row['cluster'] for row in samples])
    measured = np.concatenate([assay['measured'] for assay in assays])
    p.validate_samples(samples, measured, aids, families)
    shape = (len(assays), len(set(families)), len(samples), len({a['wildtype'] for a in assays}))
    if shape != (201, 163, 25728, 174):
        raise ValueError('full admitted support must be 201 assays / 163 families / '
                         f'25728 rows / 174 wildtypes, not {shape}')
    if load_json(source/'full/samples.json') != samples:
        raise ValueError('rebuilt sample identities disagree with the frozen source')
    features, signatures, sources, registry = retained_features(root, admission, assays)
    if len(features) != 33 or set(signatures) != set(p.SEEDS):
        raise ValueError('require the original 33 arms and three historical seeds')
    if sorted(features) != sorted(contract['models']) or list(contract['seeds']) != list(p.SEEDS):
        raise ValueError('model roster or seeds disagree with the frozen source contract')
    rows = [dict(assay=assay['assay'], cluster=assay['cluster'], mutants=assay['mutants'],
                 measured=assay['measured'], P=assay['profile_scores'],
                 S=sequence_features(assay['wildtype'], assay['mutants'])) for assay in assays]
    store, profile_hashes = crossed.load_profile_store(root/Path(PROFILE_STORE), cohort)
    try:
        deviation = gate.build_blocks(rows, cohort, store, ['wall'])
    finally:
        store['stored'].close()
    blocks = {name: np.concatenate([row[key] for row in rows])
              for name, key in (('B', 'S'), ('P', 'P_block'), ('L', 'wall'))}
    support_sha256 = p.identity_digest(samples)
    if support_sha256 != contract['panel_support']['full']['support_sha256']:
        raise ValueError('rebuilt support digest disagrees with the frozen source contract')
    provenance = dict(cohort_sha256=admission['cohort_sha256'],
                      admission_sha256=digest(root/d.LOCAL), source_registry_sha256=registry,
                      rank_feature_sources=sources, profile_sources=profile_hashes,
                      max_profile_deviation=deviation, full_support_sha256=support_sha256,
                      source_contract_sha256=digest(source/'contract.json'),
                      source_completion_sha256=digest(source/'completion.json'),
                      source_samples_sha256=digest(source/'full/samples.json'))
    return dict(samples=samples, measured=measured, aids=aids, families=families, blocks=blocks,
                raw_profile=np.concatenate([row['P'] for row in rows]), features=features,
                signatures=signatures, contract=contract, provenance=provenance,
                wildtypes=len({assay['wildtype'] for assay in assays}))


def read_baseline_table(directory, name, column):
    """Admit one upstream baseline document, its arrays and its completion."""
    import numpy as np
    from src.capability.extensions import statistical_ladder as sl

    folder = Path(directory)/name
    document = folder/'scores.json'
    if not document.is_file():
        raise ValueError(f'{document} is missing; the ladder reads the upstream baseline '
                         'document, never a bare array dump')
    table = sl.read_score_table(document, metadata_required=True)
    if column not in table['columns']:
        raise ValueError(f'{document} carries no {column}; numeric columns are '
                         f'{sorted(table["columns"])}')
    # A finite value is not evidence of a finished baseline: unconverged
    # backgrounds and partly scored wild types are refused here, by contract.
    table['completeness'] = sl.require_complete_baseline(name, table)
    arrays = folder/'scores.npz'
    if arrays.is_file():
        stored = sl.read_score_table(arrays)
        if stored['sample_id'] != table['sample_id']:
            raise ValueError(f'{arrays}: sample order disagrees with {document}')
        for other in sorted(set(stored['columns']) & set(table['columns'])):
            document_values = np.asarray(table['columns'][other], float)
            array_values = np.asarray(stored['columns'][other], float)
            if (array_values.shape != document_values.shape
                    or not np.allclose(array_values, document_values, rtol=0, atol=TOLERANCE,
                                       equal_nan=True)):
                raise ValueError(f'{arrays}: {other} disagrees with {document}')
        table['arrays'] = str(arrays)
        table['arrays_sha256'] = stored['sha256']
    return table


def verify_upstream_support(directory, sample_ids):
    """Confirm the baselines were scored on exactly the frozen support order."""
    path = Path(directory)/'inputs'/'variant_table.json'
    if not path.is_file():
        return None
    document = load_json(path)
    rows = document.get('samples') if isinstance(document, dict) else None
    if not isinstance(rows, list) or [str(row.get('sample_id')) for row in rows] != list(sample_ids):
        raise ValueError('the upstream variant table does not list the frozen support in order')
    return dict(path=str(path), sha256=digest(path), schema=document.get('schema'),
                samples=len(rows), cohort_sha256=document.get('cohort_sha256'))


def source_metrics(source):
    """The frozen source's per-assay metrics, keyed for recomputation checks."""
    rows = load_json(source/'full/assay-metrics.json')
    index = {(row['arm'], row['seed'], row['assay']): row for row in rows}
    if len(index) != len(rows):
        raise ValueError('frozen source assay metrics carry duplicate cells')
    return index


def verify_reuse(source, designs, *, model, seed, data, membership, index):
    """Admit archived predictions only after digest, identity, fold and metric checks."""
    from src.capability.extensions import statistical_ladder as sl

    stem = source/'full'/f'{model}-seed{seed}'
    archive = Path(str(stem) + '.npz')
    checks = load_json(Path(str(stem) + '-checks.json'))
    if checks.get('verification', {}).get('prediction_archive_sha256') != digest(archive):
        raise ValueError(f'{archive.name}: archived digest disagrees with its receipt')
    predictions = sl.read_predictions(archive, designs, sample_ids=data['sample_ids'], target=data['target'])
    folds = {name: load_json(Path(str(stem) + f'-{name}-folds.json')) for name in designs}
    summary = sl.verify_checks(checks, predictions, designs=designs,
                              support_sha256=data['support_sha256'], membership=membership,
                              target=data['target'], fold_records=folds)
    worst = 0.0
    rows = sl.assay_rows(predictions, data['measured'], data['aids'], data['families'],
                         model=model, seed=seed, designs=designs)
    for row in rows:
        previous = index[(model, seed, row['assay'])]
        for metric in sl.METRICS:
            for name in designs:
                value = previous[metric][name]
                if value is None or row[metric][name] is None:
                    raise ValueError('frozen source metric is undefined; reuse is refused')
                worst = max(worst, abs(row[metric][name] - value))
    if worst > TOLERANCE:
        raise ValueError(f'{model}/{seed}: recomputed metrics differ by {worst:g} from the source')
    return predictions, folds, dict(summary=summary,
                                    archive_sha256=digest(archive),
                                    max_recomputed_metric_difference=worst)


def reproduce(source, data, prepared, designs, models, seeds):
    """Refit the declared reused designs and require exact archive reproduction."""
    import numpy as np
    from src.capability.extensions import statistical_ladder as sl

    records = {}
    for model in models:
        prepared_model = dict(prepared, M=sl.prepare_block('M', data['features'][model][:, None], data['aids']))
        cells = {}
        for seed in seeds:
            started = time.monotonic()
            stem = source/'full'/f'{model}-seed{seed}'
            archive = Path(str(stem) + '.npz')
            checks = load_json(Path(str(stem) + '-checks.json'))
            expected = sl.read_predictions(archive, designs, sample_ids=data['sample_ids'], target=data['target'])
            archived = {name: load_json(Path(str(stem) + f'-{name}-folds.json')) for name in designs}
            cell = {}
            for name in designs:
                prediction, folds, design_sha256 = sl.fit_design(
                    prepared_model, data['measured'], data['aids'], data['families'],
                    data['signatures'][seed], name)
                record = dict(
                    max_absolute_difference=float(np.max(np.abs(prediction - expected[name]))),
                    prediction_identical=bool(np.array_equal(prediction, expected[name])),
                    design_digest_matched=design_sha256 == checks['designs'][name]['design_sha256'],
                    alpha_reproduced=[fold['alpha'] for fold in folds] == [fold['alpha'] for fold in archived[name]],
                    fold_membership_matched=sl.realized_membership(folds) == sl.realized_membership(archived[name]))
                if (record['max_absolute_difference'] != 0. or not record['prediction_identical']
                        or not record['design_digest_matched'] or not record['alpha_reproduced']
                        or not record['fold_membership_matched']):
                    raise ValueError(f'{model}/{seed}/{name}: the pilot did not reproduce the archived fit')
                cell[name] = record
            cell['seconds'] = time.monotonic() - started
            cells[str(seed)] = cell
            print(json.dumps(dict(pilot_model=model, seed=seed, seconds=cell['seconds'])), flush=True)
        records[model] = cells
    return records


def publish_cell(out, model, seed, predictions, folds, design_sha256, grid):
    """Persist one fitted cell and read it back through the same verifier."""
    import numpy as np
    from src.capability.extensions import statistical_ladder as sl

    stem = out/'cells'/f'{model}-seed{seed}'
    archive = archive_path(Path(str(stem) + '-fitted'))
    np.savez_compressed(archive, **predictions, original_index=grid['index'],
                        target=grid['target'], sample_id=grid['sample_ids'])
    fold_paths = {}
    for name, records in folds.items():
        fold_paths[name] = Path(str(stem) + f'-{name}-folds.json')
        save(fold_paths[name], records)
    checks = dict(designs={name: dict(design=list(sl.DESIGNS[name]), design_sha256=design_sha256[name],
                                     target_sha256=grid['target_sha256'],
                                     support_sha256=grid['support_sha256'],
                                     membership_sha256=grid['membership_sha256'],
                                     rows=len(grid['target']),
                                     prediction_sha256=sl.array_digest(predictions[name]))
                            for name in predictions},
                  verification=dict(prediction_archive_sha256=digest(archive)))
    save(Path(str(stem) + '-checks.json'), checks)
    stored = sl.read_predictions(archive, predictions, sample_ids=grid['sample_ids'], target=grid['target'])
    reloaded = load_json(Path(str(stem) + '-checks.json'))
    return sl.verify_checks(reloaded, stored, designs=tuple(predictions),
                            support_sha256=grid['support_sha256'], membership=grid['membership'],
                            target=grid['target'],
                            fold_records={name: load_json(path)
                                          for name, path in fold_paths.items()})


def declare_scope(source, *, stage, scores, pilot, models, seeds, threads,
                  potts_column, gemme_column) -> tuple[list, tuple, dict]:
    """Resolve the run scope, admit the baselines and gate reuse before any fit.

    ``scores`` is the upstream output root; the two admitted tables come back in
    ``preflight['tables']`` so the full stage parses each document exactly once.
    """
    from src.capability.extensions import statistical_ladder as sl

    # Argument refusals first: they need no artifact at all.
    if stage == 'profile_only' and scores is not None:
        raise ValueError('profile_only is a declared partial pilot; it never reads Potts/GEMME')
    if stage == 'full':
        if scores is None:
            raise ValueError(MISSING_SCORES)
        if pilot is None:
            raise ValueError(MISSING_PILOT)
    contract = load_json(source/'contract.json')
    contract_sha256 = digest(source/'contract.json')
    completion_sha256 = digest(source/'completion.json')
    roster = list(contract['models'])
    if models is None:
        models = sorted(roster)
    else:
        models = sorted(set(models))
        unknown = [model for model in models if model not in roster]
        if unknown:
            raise ValueError(f'models outside the frozen roster: {unknown}')
    seeds = tuple(sl.SEEDS if seeds is None else seeds)
    if not seeds or set(seeds) - set(sl.SEEDS):
        raise ValueError('seeds must be drawn from the original three')
    if stage == 'profile_only':
        return models, seeds, dict(tables={}, pilot=None)
    if scores is None or pilot is None:  # refused above; kept local so the reader sees it
        raise ValueError(MISSING_SCORES if scores is None else MISSING_PILOT)
    if len(models) != 33 or list(seeds) != list(sl.SEEDS):
        raise ValueError('the full ladder is declared for all 33 models and three seeds')
    # Admit the upstream baselines before the reuse gate: their completion
    # contract is what the operator cannot repair by re-running a pilot.
    tables = {name: read_baseline_table(scores, name, column)
              for name, column in (('potts', potts_column), ('gemme', gemme_column))}
    gate = sl.validate_pilot(load_json(pilot), source_contract_sha256=contract_sha256,
                             source_completion_sha256=completion_sha256,
                             current_script_sha256=digest(Path(__file__)),
                             current_module_sha256=digest(Path(sl.__file__)),
                             models=models, threads=threads)
    return models, seeds, dict(tables=tables,
                               pilot=dict(path=str(pilot), sha256=digest(pilot), gate=gate))


def run(root, out, *, stage, source=SOURCE, scores=None, pilot=None, models=None,
        seeds=None, threads=2, bootstrap=None, test_evidence=None,
        potts_column=None, gemme_column=None) -> None:
    """Fit the declared stage on the frozen support; never a silent fallback."""
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from src.capability.extensions import overlap
    from src.capability.extensions import statistical_ladder as sl

    if stage not in sl.STAGE_CONTROL_SETS:
        raise ValueError(f'unknown stage {stage!r}')
    if threads not in (1, 2):
        raise ValueError('this ladder declares one or two CPU threads')
    if bootstrap is None:
        bootstrap = sl.BOOTSTRAP_DRAWS
    if bootstrap != sl.BOOTSTRAP_DRAWS:
        raise ValueError(f'the declared joint family uses {sl.BOOTSTRAP_DRAWS} draws')
    if not test_evidence or test_evidence.get('status') != 'passed':
        raise ValueError('successful extension tests required before fitting')
    if out.exists():
        raise ValueError('never overwrite prior outputs; choose a new output root')
    potts_column = potts_column or sl.POTTS_COLUMN
    gemme_column = gemme_column or sl.GEMME_COLUMN
    # Resolve and gate the declared scope before the expensive support rebuild.
    models, seeds, preflight = declare_scope(source, stage=stage, scores=scores, pilot=pilot,
                                            models=models, seeds=seeds, threads=threads,
                                            potts_column=potts_column, gemme_column=gemme_column)
    if shutil.disk_usage(out.parent if out.parent.exists() else root).free < 2_000_000_000:
        raise ValueError('insufficient disk headroom')
    torch.set_num_threads(threads)
    out.mkdir(parents=True, exist_ok=False)
    (out/'cells').mkdir()
    started = time.monotonic()
    save(out/'runtime-start.json', runtime(threads))
    save(out/'tests.json', test_evidence)
    hashes = {name: digest(root/name) for name in CODE_DEPENDENCIES}
    control_sets = sl.STAGE_CONTROL_SETS[stage]
    sets = {name: dict(sl.CONTROL_SETS[name]) for name in control_sets}
    designs = sl.stage_designs(stage)
    reused = sl.reused_designs(stage)
    fitted = tuple(name for name in designs if name not in reused)
    shared_designs = tuple(name for name in fitted if 'M' not in sl.DESIGNS[name])
    tables = preflight['tables']
    with threadpool_limits(limits=threads):
        data = load_support(root, source)
        data['sample_ids'] = np.asarray([row['sample_id'] for row in data['samples']])
        data['target'] = overlap.rerank(data['measured'], data['aids'])
        data['index'] = np.arange(len(data['target']))
        data['support_sha256'] = data['provenance']['full_support_sha256']
        data['target_sha256'] = sl.array_digest(data['target'])
        raw = {'profile': data['raw_profile']}
        if stage == 'full':
            joined = {}
            for name, column in (('potts', potts_column), ('gemme', gemme_column)):
                joined.update(sl.join_scores(data['sample_ids'], tables[name], columns=(column,)))
            if sl.PROFILE_COLUMN in tables['potts']['columns']:
                joined[sl.PROFILE_COLUMN] = sl.join_scores(
                    data['sample_ids'], tables['potts'], columns=(sl.PROFILE_COLUMN,))[sl.PROFILE_COLUMN]
                difference = float(np.max(np.abs(joined[sl.PROFILE_COLUMN] - data['raw_profile'])))
                if difference > TOLERANCE:
                    raise ValueError('the upstream profile delta disagrees with the retained cohort '
                                     f'profile score by {difference:g}')
            data['blocks']['E'] = joined[potts_column][:, None]
            data['blocks']['G'] = joined[gemme_column][:, None]
            raw['potts_delta'] = joined[potts_column]
            raw['gemme_score'] = joined[gemme_column]
            save(out/'scores.json', dict(
                directory=str(scores),
                upstream_support=verify_upstream_support(scores, data['sample_ids']),
                tables={name: dict(path=table['path'], sha256=table['sha256'],
                                   arrays=table.get('arrays'),
                                   arrays_sha256=table.get('arrays_sha256'),
                                   metadata_sha256=table['metadata_sha256'],
                                   column=column, samples=len(table['sample_id']),
                                   completeness=table['completeness'],
                                   ranges=dict(minimum=float(joined[column].min()),
                                               maximum=float(joined[column].max()),
                                               mean=float(joined[column].mean())))
                        for name, table, column in ((name, tables[name], column)
                                                    for name, column in (('potts', potts_column),
                                                                         ('gemme', gemme_column)))},
                join='unique sample_id, complete over the frozen support, reordered to the frozen row order',
                gemme_aggregation='mean of the declared three official runs per wild type, '\
                                  'then ranked within assay exactly like M; run spread is tool '\
                                  'stochasticity, not biological replication',
                profile_delta_max_absolute_difference=(
                    float(np.max(np.abs(joined[sl.PROFILE_COLUMN] - data['raw_profile'])))
                    if sl.PROFILE_COLUMN in joined else None)))
        memberships = {seed: sl.projected_membership(data['signatures'][seed], data['families'])
                       for seed in seeds}
        data['membership_sha256'] = {seed: sl.identity_digest(memberships[seed]) for seed in seeds}
        prepared = {name: sl.prepare_block(name, data['blocks'][name], data['aids'])
                    for name in ('B', 'P', 'L', 'E', 'G') if name in data['blocks']}
        save(out/'contract.json', dict(
            analysis='statistical_baseline_ladder_20261106', stage=stage,
            mode='pilot_partial' if stage == 'profile_only' else 'declared_full',
            ladder_complete=stage == 'full', scope=dict(models=list(models), seeds=list(seeds),
                control_sets=list(control_sets), designs=dict(fitted=list(fitted), reused=list(reused),
                shared_across_models=list(shared_designs))),
            support=dict(rows=len(data['target']), assays=len(set(data['aids'])),
                         families=len(set(data['families'])), wildtypes=data['wildtypes'],
                         support_sha256=data['support_sha256']),
            registry=sl.registry(), metrics=list(sl.METRICS),
            hypothesis_family=dict(declared=396,
                                   realized=len(models)*len(control_sets)*len(sl.DIRECTIONS),
                                   note='6 control sets x 33 models x 2 directions per metric; a partial '
                                        'stage realizes a smaller pilot family and is never reported as declared'),
            bootstrap=dict(draws=bootstrap, seed=sl.BOOTSTRAP_SEED, unit='biological family', paired=True,
                           method='shared family draws; 95th percentile of the joint maximum absolute '
                                  'centered bootstrap deviation in native units, separately per metric'),
            sensitivity=dict(method='shared-family standardized maximum statistic (fixed group SE)',
                             role='declared sensitivity only; the native-unit family is primary'),
            aggregation='mean the three seeds within assay, then equal assays within family, then equal families',
            preprocessing='within-assay standardized rank of M, P column 0, E and G on the frozen support; '
                          'all other historical columns unchanged',
            target='within-assay standardized rank of the measured phenotype',
            folds='the original three seeded five-outer/four-inner family partitions, projected unchanged',
            reuse='archived M, MP, BPL and BMPL held-out predictions admitted only after digest, identity, '
                  'fold and metric verification; the reproduction pilot is required by the full stage',
            uncertainty='conditional on fitted held-out predictions; no refitting or training-history draws',
            limitations=sl.LIMITATIONS, code_sha256=hashes, provenance=data['provenance'],
            reuse_pilot=preflight['pilot']))
        index = source_metrics(source)
        grid = {seed: dict(index=data['index'], target=data['target'], sample_ids=data['sample_ids'],
                           support_sha256=data['support_sha256'],
                           target_sha256=data['target_sha256'],
                           membership_sha256=data['membership_sha256'][seed],
                           membership=memberships[seed]) for seed in seeds}
        remaining = [(model, seed) for model in models for seed in seeds]
        reused_predictions = {}
        reuse_records = []
        for model, seed in remaining:
            predictions, folds, record = verify_reuse(source, reused, model=model, seed=seed,
                                                      data=data, membership=memberships[seed],
                                                      index=index)
            reused_predictions[(model, seed)] = predictions
            reuse_records.append(dict(model=model, seed=seed, designs=list(reused), **record))
        save(out/'reuse-verification.json', dict(designs=list(reused), cells=reuse_records,
             status='passed', max_recomputed_metric_difference=max(
                 cell['max_recomputed_metric_difference'] for cell in reuse_records)))
        if stage == 'profile_only':
            reproduction = reproduce(source, data, prepared,
                                     tuple(name for name in sl.REPRODUCTION_DESIGNS if name in reused),
                                     models, seeds)
            save(out/'reuse-pilot.json', dict(
                schema=sl.PILOT_SCHEMA, status='passed', threads=threads,
                source_contract_sha256=data['provenance']['source_contract_sha256'],
                source_completion_sha256=data['provenance']['source_completion_sha256'],
                script_sha256=hashes['scripts/capability/extensions/fit_statistical_ladder.py'],
                module_sha256=hashes['src/capability/extensions/statistical_ladder.py'],
                designs=[name for name in sl.REPRODUCTION_DESIGNS if name in reused],
                models=list(models), seeds=list(seeds), reproduction=reproduction))
        rows = []
        completed = []
        for seed in seeds:
            shared = {}
            for name in shared_designs:
                shared[name] = sl.fit_design(prepared, data['measured'], data['aids'],
                                             data['families'], data['signatures'][seed], name)
            for model in models:
                cell_started = time.monotonic()
                model_prepared = dict(prepared, M=sl.prepare_block('M', data['features'][model][:, None],
                                                                  data['aids']))
                predictions = dict(reused_predictions[(model, seed)])
                folds = {}
                design_sha256 = {}
                for name in fitted:
                    if 'M' in sl.DESIGNS[name]:
                        prediction, records, design_sha256[name] = sl.fit_design(
                            model_prepared, data['measured'], data['aids'], data['families'],
                            data['signatures'][seed], name)
                        folds[name] = records
                    else:
                        prediction, records, design_sha256[name] = shared[name]
                        folds[name] = records
                    predictions[name] = prediction
                verification = publish_cell(out, model, seed, {name: predictions[name] for name in fitted},
                                            folds, design_sha256, grid[seed])
                rows.extend(sl.assay_rows(predictions, data['measured'], data['aids'], data['families'],
                                          model=model, seed=seed, designs=designs))
                completed.append(dict(model=model, seed=seed, designs=list(fitted),
                                      seconds=time.monotonic() - cell_started,
                                      archive=verification['designs']))
                save(out/'progress.json', dict(completed=completed, expected_cells=len(models)*len(seeds),
                                               elapsed_seconds=time.monotonic() - started))
                print(json.dumps(completed[-1]), flush=True)
        if len(completed) != len(models)*len(seeds):
            raise ValueError('completed cells disagree with the requested scope')
        save(out/'assay-metrics.json', rows)
        summary = {}
        for metric in sl.METRICS:
            ladder = sl.ladder_matrix(rows, metric, control_sets=sets, models=models, seeds=seeds)
            absolute = sl.absolute_matrix(rows, metric, designs=designs, models=models, seeds=seeds)
            if ladder['undefined']:
                summary[metric] = dict(status='not_estimable', metric=metric,
                                       undefined_cells=len(ladder['undefined']),
                                       reason='undefined per-assay correlations; no outcome-driven exclusions')
                continue
            inference = sl.infer(ladder['matrix'], ladder['labels'], metric=metric, bootstrap=bootstrap)
            _, absolute_pointwise = sl.pointwise_interval(absolute['matrix'], draws=bootstrap,
                                                          seed=sl.BOOTSTRAP_SEED)
            keys = [f'{model}|{name}' for model, name in absolute['labels']]
            inference['verdict_counts'] = sl.verdict_counts(inference['hypotheses'])
            inference['absolute'] = dict(
                point=dict(zip(keys, absolute['matrix'].mean(axis=0).tolist())),
                pointwise=dict(zip(keys, absolute_pointwise.T.tolist())))
            summary[metric] = inference
            np.savez_compressed(out/f'aggregate-{metric}.npz',
                                contrasts=ladder['matrix'], absolute=absolute['matrix'],
                                families=np.asarray(ladder['families']),
                                contrast_labels=np.asarray([json.dumps(list(label))
                                                            for label in ladder['labels']]),
                                design_labels=np.asarray([json.dumps(list(label))
                                                          for label in absolute['labels']]))
        save(out/'contrasts.json', dict(
            analysis='statistical_baseline_ladder_20261106', stage=stage,
            ladder_complete=stage == 'full', control_sets=list(control_sets), metrics=list(sl.METRICS),
            directions=dict(sl.DIRECTIONS),
            contrast_sign='joint minus reduced arm for Spearman; reduced minus joint for rank MSE',
            hypotheses_per_metric=len(models)*len(control_sets)*len(sl.DIRECTIONS),
            declared_hypotheses_per_metric=396,
            aggregation='mean seeds within assay, equal assays within family, equal families',
            bootstrap=dict(draws=bootstrap, seed=sl.BOOTSTRAP_SEED), results=summary))
        save(out/'zero-shot.json', sl.zero_shot(raw, data['measured'], data['aids'], data['families'],
                                                draws=bootstrap))
    if any(digest(root/name) != sha for name, sha in hashes.items()):
        raise ValueError('a code dependency changed during the run')
    save(out/'runtime-end.json', runtime(threads))
    output_hashes = {str(path.relative_to(out)): digest(path) for path in sorted(out.rglob('*')) if path.is_file()}
    save(out/'completion.json', dict(
        status='complete' if stage == 'full' else 'partial_profile_only',
        ladder_complete=stage == 'full', stage=stage, tests_passed=True,
        models=list(models), seeds=list(seeds), control_sets=list(control_sets),
        completed_cells=len(completed),
        hypotheses_per_metric=len(models)*len(control_sets)*len(sl.DIRECTIONS),
        declared_hypotheses_per_metric=396,
        note=('complete declared ladder' if stage == 'full' else
              'partial pilot: only P/M/P+M and the original C0 control were computed; '
              'this receipt is not the complete ladder'),
        output_sha256=output_hashes, elapsed_seconds=time.monotonic() - started))
    print((out/'completion.json').read_text())


def main():
    from src.capability.extensions import statistical_ladder as sl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('profile_only', 'full'), required=True)
    parser.add_argument('--scores', type=Path,
                        help='statistical_ladder_20261106 output root holding potts/ and gemme/ tables')
    parser.add_argument('--potts-column', default=sl.POTTS_COLUMN)
    parser.add_argument('--gemme-column', default=sl.GEMME_COLUMN)
    parser.add_argument('--reuse-pilot', type=Path, help='reuse-pilot.json from a profile_only run')
    parser.add_argument('--models', nargs='+', help='subset for a declared partial run')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--threads', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[variable] = str(args.threads)
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q',
                           str(ROOT/'tests/extensions/test_statistical_ladder.py')],
                          cwd=ROOT, capture_output=True, text=True)
    print(test.stdout, end='', flush=True)
    if test.returncode:
        print(test.stderr, file=sys.stderr)
        raise SystemExit('statistical-ladder tests failed; no fits launched')
    evidence = dict(status='passed', command='pytest -q tests/extensions/test_statistical_ladder.py',
                    stdout=test.stdout,
                    test_sha256=digest(ROOT/'tests/extensions/test_statistical_ladder.py'))
    run(args.root.resolve(), args.out.resolve(), stage=args.stage, source=args.source.resolve(),
        scores=args.scores.resolve() if args.scores else None,
        pilot=args.reuse_pilot.resolve() if args.reuse_pilot else None,
        models=args.models, seeds=args.seeds, threads=args.threads, test_evidence=evidence,
        potts_column=args.potts_column, gemme_column=args.gemme_column)


if __name__ == '__main__':
    main()