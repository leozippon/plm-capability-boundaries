#!/usr/bin/env python3
"""CPU reproduction of historical abundance/stability primary phenotype fits.

No extraction, model loading, requalification, remote purge, or source overwrite.
--extraction names a root containing <arm>/manifest_<arm>.json and its NPZs.
--baseline-only recovers the original baseline without model likelihoods; its
augmented predictions remain blank and explicitly unavailable, not zero-filled.
All three historical seeds are retained, including explicit failed sample rows.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

# Must precede NumPy/Torch imports in the executable process.
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '4'

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEEDS = (20260923, 20260924, 20260925)
ATOL, RTOL = 1e-9, 1e-8
NAMESPACE = 'prediction_details_replay_20261005/L20reproduction'
FIELDS = (
    'endpoint', 'arm', 'sample_id', 'row_index', 'unit', 'family', 'variant_index',
    'state_index', 'mutation', 'mutation_position', 'wildtype_residue', 'mutant_residue',
    'phenotype', 'phenotype_units', 'seed', 'heldout_fold', 'heldout',
    'baseline_name', 'baseline_prediction', 'augmented_name', 'augmented_prediction',
    'status', 'provenance', 'cohort_sha256', 'feature_manifest_sha256',
)


class ReplayError(ValueError):
    """An input/support/fold mismatch; predictions must not be presented as valid."""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise ReplayError(f'Missing required input: {path}')
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def input_paths(root: Path, endpoint: str, arm: str) -> dict[str, Path]:
    if endpoint == 'abundance':
        base = root / 'results/R1/external_confirmation_20260924'
        return {'cohort': base / 'cohort.json', 'profiles': base / 'profile_features.npz',
                'controls': base / 'controls_qualification.json',
                'reference': base / 'fits' / f'fit_{arm}.json'}
    base = root / 'results/R4/gate_stability_20260924'
    return {'cohort': base / 'cohort/cohort.json',
            'profiles': base / 'cohort/profile_features.npz',
            'controls': base / 'controls/controls_qualification.json',
            'reference': root / 'results/R4/stability_likelihood_replay_20260927' / f'{arm}.json'}


def sample_id(cohort_sha256: str, unit: str, mutation: str) -> str:
    """Shared retained-export contract; independent of seed, arm and row indices."""
    payload = json.dumps([cohort_sha256, unit, mutation], ensure_ascii=False,
                         sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def registry(cohort: dict, endpoint: str, cohort_sha256: str) -> list[dict]:
    """Exact source unit/variant order, with zero-based variant indices (not IDs guessed from scores)."""
    units = cohort['domains' if endpoint == 'abundance' else 'backgrounds']
    records, names, sample_ids = [], set(), set()
    for unit in units:
        name = unit['name']
        if name in names:
            raise ReplayError(f'Duplicate unit: {name}')
        names.add(name)
        for index, variant in enumerate(unit['variants']):
            position = variant['position']
            wild = unit['wildtype']
            sequence = variant['sequence']
            if (not 1 <= position <= len(wild) or len(sequence) != len(wild)
                    or [i + 1 for i, (a, b) in enumerate(zip(wild, sequence)) if a != b] != [position]
                    or sequence[position - 1] != variant['mutant']):
                raise ReplayError(f'{name}/{index}: inconsistent single substitution')
            target = float(variant['target' if endpoint == 'abundance' else 'ddg'])
            if not math.isfinite(target):
                raise ReplayError(f'{name}/{index}: nonfinite phenotype')
            mutation = f'{wild[position - 1]}{position}{variant["mutant"]}'
            identifier = sample_id(cohort_sha256, name, mutation)
            if identifier in sample_ids:
                raise ReplayError(f'{name}/{index}: duplicated source mutation/sample ID')
            sample_ids.add(identifier)
            records.append({'sample_id': identifier,
                            'mutation': mutation, 'row_index': len(records),
                            'unit': name, 'family': unit['group'], 'variant_index': index,
                            'state_index': variant.get('state', index + 1),
                            'mutation_position': position, 'wildtype_residue': wild[position - 1],
                            'mutant_residue': variant['mutant'], 'phenotype': target,
                            'phenotype_units': 'normalized_fitness' if endpoint == 'abundance' else 'kcal/mol'})
    return records


def validate_reference(endpoint: str, arm: str, records: list[dict], controls: dict,
                       reference: dict, hashes: dict) -> tuple[str, tuple[str, ...]]:
    """Freeze the original baseline, never rerun its qualification rule."""
    schema = 'nested_gate_control_qualification_v1' if endpoint == 'abundance' else 'stability_control_qualification_v1'
    if controls['schema'] != schema:
        raise ReplayError('Unexpected control schema')
    for value, expected, label in [
            (controls['cohort_sha256'], hashes['cohort'], 'controls cohort'),
            (controls['profile_sha256'], hashes['profiles'], 'controls profiles'),
            (reference['cohort_sha256'], hashes['cohort'], 'reference cohort'),
            (reference['arm'], arm, 'reference arm')]:
        if value != expected:
            raise ReplayError(f'{label} differs from original reference')
    qualified = tuple(controls['qualified_control_set'])
    expected_controls = ('ident', 'geom', 'comp', 'chem', 'prof', 'prof2', 'G') if endpoint == 'abundance' else ('ident', 'geom', 'chem', 'G')
    if qualified != expected_controls or controls['split_seeds'] != list(SEEDS):
        raise ReplayError('Original control set/split seeds changed')
    if controls['rows'] != len(records):
        raise ReplayError('Original support row count changed')
    if endpoint == 'abundance':
        if reference['schema'] != 'nested_gate_fit_v1':
            raise ReplayError('Unexpected abundance fit schema')
        for key, actual in [('profile_sha256', hashes['profiles']),
                            ('controls_sha256', hashes['controls'])]:
            if reference[key] != actual:
                raise ReplayError(f'Original {key} changed')
        if (reference['qualified_control_set'] != list(qualified)
                or reference['split_seeds'] != list(SEEDS) or reference['rows'] != len(records)):
            raise ReplayError('Original abundance fit support/controls/seeds changed')
        baseline = reference['matched_baseline']['primary']
        if baseline not in ('S', 'S_T'):
            raise ReplayError('Unsupported historical primary baseline')
    else:
        baseline = reference['baseline']
        if baseline != 'S' or set(reference['seeds']) != set(map(str, SEEDS)):
            raise ReplayError('Stability must use original S primary and all three seeds')
        expected_groups = sorted({r['family'] for r in records})
        for previous in reference['seeds'].values():
            if previous['groups'] != expected_groups:
                raise ReplayError('Stability reference group support/order differs')
            if any(len(previous[key]) != len(expected_groups) for key in ('baseline_mse', 'augmented_mse')):
                raise ReplayError('Stability reference group MSE support differs')
    return baseline, qualified + (('T',) if baseline == 'S_T' else ())


def canonical_plan(root: Path, endpoint: str, cohort_sha256: str, units: list[dict]) -> tuple[dict, str]:
    from scripts.capability.stability.extract_stability_singles import plan_digest
    path = (root / 'results/R1/external_confirmation_20260924/extraction_plan.json'
            if endpoint == 'abundance' else root / 'results/R4/gate_stability_20260924/cohort/extraction_plan.json')
    plan = read_json(path)
    if plan['schema'] != 'stability_singles_extraction_v1' or plan['cohort_sha256'] != cohort_sha256:
        raise ReplayError('Canonical extraction plan schema/cohort mismatch')
    rows = plan['backgrounds']
    if [r['name'] for r in rows] != [u['name'] for u in units]:
        raise ReplayError('Canonical extraction plan unit order mismatch')
    for row, unit in zip(rows, units):
        if row['sequences'] != [unit['wildtype'], *[v['sequence'] for v in unit['variants']]]:
            raise ReplayError('Canonical extraction plan state sequences mismatch')
    return plan, plan_digest(plan)


def tokenizer_records(arm: str, units: list[dict]) -> tuple[dict, dict]:
    """Native tokenizer-only packing; device=None never loads model weights."""
    from src.capability.interactions.pairwise_epistasis import ARM_DTYPE
    from src.capability.readouts.readout_extraction import load_readout_arm
    from scripts.capability.stability.extract_stability_singles import token_record
    handle = load_readout_arm(arm, None, device=None, dtype=ARM_DTYPE.get(arm, 'float32'))
    if handle.model is not None:
        raise ReplayError('Tokenizer-only reconstruction unexpectedly loaded a model')
    metadata = {str(p.relative_to(handle.spec.path)): digest(p)
                for p in sorted(handle.spec.path.rglob('*')) if p.is_file()
                and (p.suffix in ('.json', '.model', '.txt') or p.name in ('vocab', 'merges'))}
    records = {u['name']: [token_record(handle, sequence, 1024)
                           for sequence in [u['wildtype'], *[v['sequence'] for v in u['variants']]]]
               for u in units}
    return records, {'method': 'native token_record; tokenizer only; no model weights or inference',
                     'checkpoint_metadata_sha256': metadata}


def token_blocks(units: list[dict], records: dict) -> dict:
    import numpy as np
    from src.capability.stability.stability_gate import tokenisation_block
    tokens = []
    for unit in units:
        states = records[unit['name']]
        pooled = np.asarray([len(s['pooled_positions']) for s in states])
        ids = [s['ids'] for s in states]
        for i, variant in enumerate(unit['variants']):
            tokens.append(tokenisation_block(pooled, ids, 0, variant.get('state', i + 1), len(unit['wildtype'])))
    return {'T': np.asarray(tokens, dtype=float)}


def load_likelihood_blocks(directory: Path, units: list[dict], manifest: dict) -> dict:
    """Exactly the original M and T transformations, without constructing unused R."""
    import numpy as np
    from src.capability.stability.stability_gate import tokenisation_block
    by_name = {r['background']: r for r in manifest['backgrounds']}
    likelihood, tokens = [], []
    for unit in units:
        with np.load(directory / by_name[unit['name']]['file'], allow_pickle=False) as data:
            scores, pooled = data['likelihood'], data['pooled_token_counts']
            offsets, ids = data['token_offsets'], data['token_ids']
            token_ids = [ids[offsets[i]:offsets[i + 1]].tolist() for i in range(len(scores))]
            for state in data['variant_states']:
                likelihood.append(float(scores[state] - scores[0]))
                tokens.append(tokenisation_block(pooled, token_ids, 0, int(state), len(unit['wildtype'])))
    return {'M': np.asarray(likelihood, dtype=float)[:, None], 'T': np.asarray(tokens, dtype=float)}


def validate_features(directory: Path, arm: str, cohort_sha256: str,
                      units: list[dict], *, plan: dict | None = None,
                      expected_plan_sha256: str | None = None) -> tuple[dict, dict]:
    """Check every supplied archive before fitting; never search another cohort/wave."""
    import numpy as np
    from src.capability.interactions.pairwise_epistasis import ARM_DTYPE
    path = directory / f'manifest_{arm}.json'
    manifest = read_json(path)
    identity = manifest['identity']
    likelihood_only = identity['schema'] == 'stability_singles_likelihood_extraction_v1'
    if (manifest['status'] != 'complete' or identity['arm'] != arm
            or identity['schema'] not in ('stability_singles_extraction_v1', 'stability_singles_likelihood_extraction_v1')
            or identity['cohort_sha256'] != cohort_sha256):
        raise ReplayError('Extraction schema/status/arm/cohort mismatch')
    if (identity['batch_size'] != 1 or identity['budget'] != 1024
            or identity['dtype'] != ARM_DTYPE.get(arm, 'float32')):
        raise ReplayError('Extraction must retain singleton batch, budget 1024 and original ARM_DTYPE')
    if expected_plan_sha256 is not None and identity.get('plan_sha256') != expected_plan_sha256:
        raise ReplayError('Extraction canonical plan digest mismatch')
    if likelihood_only and (plan is None or expected_plan_sha256 is None):
        raise ReplayError('Likelihood-only extraction requires the original canonical plan')
    expected_tokens = tokenizer_records(arm, units)[0] if likelihood_only else None
    by_name = {row['background']: row for row in manifest['backgrounds']}
    if len(by_name) != len(manifest['backgrounds']) or set(by_name) != {u['name'] for u in units}:
        raise ReplayError('Extraction unit support differs; no partial or substitute cohort allowed')
    hashes = {'manifest': digest(path), 'archives': {}}
    for unit in units:
        item = by_name[unit['name']]
        archive = (directory / item['file']).resolve()
        if not archive.is_relative_to(directory.resolve()):
            raise ReplayError('Extraction archive escapes the supplied directory')
        if not archive.is_file():
            raise ReplayError(f'Missing archive for {unit["name"]}: {archive}; supply complete original-plan extraction')
        actual = digest(archive)
        if actual != item['sha256']:
            raise ReplayError(f'{unit["name"]}: archive checksum mismatch')
        hashes['archives'][item['file']] = actual
        with np.load(archive, allow_pickle=False) as data:
            states = np.asarray([v.get('state', i + 1) for i, v in enumerate(unit['variants'])])
            positions = np.asarray([v['position'] for v in unit['variants']])
            if (not np.array_equal(data['variant_states'], states)
                    or not np.array_equal(data['variant_positions'], positions)
                    or not np.array_equal(states, np.arange(1, len(states) + 1))):
                raise ReplayError(f'{unit["name"]}: extraction variant state/position order changed')
            n = len(states) + 1
            for key in ('likelihood', 'pooled_token_counts'):
                if data[key].shape != (n,) or not np.isfinite(data[key]).all():
                    raise ReplayError(f'{unit["name"]}: invalid {key} state support')
            if likelihood_only:
                expected_sequences = np.asarray([hashlib.sha256(s.encode()).hexdigest()
                    for s in [unit['wildtype'], *[v['sequence'] for v in unit['variants']]]], dtype='<U64')
                if (data['state_sequence_sha256'].dtype != np.dtype('<U64')
                        or not np.array_equal(data['state_sequence_sha256'], expected_sequences)):
                    raise ReplayError(f'{unit["name"]}: canonical state sequence hashes mismatch')
            offsets, ids = data['token_offsets'], data['token_ids']
            if (offsets.shape != (n + 1,) or offsets.dtype.kind not in 'iu'
                    or ids.ndim != 1 or ids.dtype.kind not in 'iu'
                    or offsets[0] != 0 or offsets[-1] != len(ids)
                    or np.any(np.diff(offsets) <= 0) or np.any(np.diff(offsets) > 1024)
                    or np.any(data['pooled_token_counts'] <= 0)):
                raise ReplayError(f'{unit["name"]}: invalid token support/offsets')
            if expected_tokens is not None:
                for i, expected in enumerate(expected_tokens[unit['name']]):
                    if (ids[offsets[i]:offsets[i + 1]].tolist() != expected['ids']
                            or data['pooled_token_counts'][i] != len(expected['pooled_positions'])):
                        raise ReplayError(f'{unit["name"]}: canonical token content mismatch at state {i}')
    return manifest, hashes


def validate_panel(panel: dict, records: list[dict], blocks: dict) -> None:
    import numpy as np
    expected = {'target': [r['phenotype'] for r in records],
                'group': [r['family'] for r in records],
                'site': [f'{r["unit"]}:{r["mutation_position"]}' for r in records]}
    for key, values in expected.items():
        if not np.array_equal(panel[key], np.asarray(values)):
            raise ReplayError(f'Panel {key} row order differs from the source registry')
    for name, block in blocks.items():
        if block.ndim != 2 or len(block) != len(records) or not np.isfinite(block).all():
            raise ReplayError(f'{name}: missing/nonfinite/misaligned features')


@contextmanager
def capture_partitions(module):
    """Observe actual family_folds calls without changing the historical fit algorithm."""
    original, calls = module.family_folds, []

    def observed(groups, splits, seed):
        result = original(groups, splits, seed)
        # The original returns a list; preserve it exactly, including order.
        calls.append({'seed': int(seed), 'splits': int(splits),
                      'groups': sorted(set(groups.tolist())),
                      'held_groups': [list(fold) for fold in result]})
        return result

    module.family_folds = observed
    try:
        yield calls
    finally:
        module.family_folds = original


def heldout_join(groups, folds: list[dict]):
    import numpy as np
    universe = set(np.asarray(groups).tolist())
    assigned = {}
    for index, fold in enumerate(folds):
        held = fold['held_groups']
        if fold['fold'] != index or len(set(held)) != len(held):
            raise ReplayError('Invalid outer fold numbering/duplicate held groups')
        if fold.get('purged_training_groups'):
            raise ReplayError('Remote purge is not the historical primary comparison')
        if set(fold['training_groups']) != universe - set(held):
            raise ReplayError('Outer training support differs from the unpurged complement')
        for group in held:
            if group not in universe or group in assigned:
                raise ReplayError('Held-out group appears outside support or in multiple folds')
            assigned[group] = index
    if set(assigned) != universe:
        raise ReplayError('Incomplete held-out group coverage')
    return np.asarray([assigned[group] for group in groups], dtype=int)


def compare_numeric(label: str, actual, expected) -> dict:
    """No relaxed tolerances: numerical disagreement remains usable but flagged."""
    import numpy as np
    if actual is None or expected is None:
        agrees = actual is None and expected is None
        return {'metric': label, 'actual': actual, 'expected': expected, 'agrees': agrees}
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ReplayError(f'{label}: metric support/nonfinite reference mismatch')
    return {'metric': label, 'actual': a.tolist(), 'expected': b.tolist(),
            'agrees': bool(np.allclose(a, b, atol=ATOL, rtol=RTOL)),
            'max_absolute_difference': float(np.max(np.abs(a - b))) if a.size else 0.0}


def validate_fit(result: dict, reference: dict, endpoint: str, seed: int,
                 baseline: str, panel: dict, partitions: list[dict], *, baseline_only: bool = False) -> tuple[object, list[dict]]:
    import numpy as np
    folds = result['folds']
    assigned = heldout_join(panel['group'], folds)
    if (len(folds) != 5 or len(partitions) != 6 or partitions[0]['splits'] != 5
            or partitions[0]['seed'] != seed):
        raise ReplayError('Historical 5-outer/4-inner partition flow changed')
    for index, fold in enumerate(folds):
        if (partitions[0]['held_groups'][index] != fold['held_groups']
                or partitions[index + 1]['splits'] != 4
                or partitions[index + 1]['seed'] != seed + 100 + index
                or partitions[index + 1]['groups'] != fold['training_groups']):
            raise ReplayError('Actual inner/outer fitting partition mismatch')
    previous = reference['per_seed' if endpoint == 'abundance' else 'seeds'][str(seed)]
    historical_folds = previous.get('folds')
    held = ([f['held_groups'] for f in historical_folds] if historical_folds is not None
            else previous['held_groups_per_fold'])
    if [f['held_groups'] for f in folds] != held:
        raise ReplayError('Held groups/order differs from historical fit')
    comparisons = []
    for index, fold in enumerate(folds):
        old = historical_folds[index] if historical_folds is not None else previous
        if historical_folds is not None and (old['training_groups'] != fold['training_groups']
                                             or old.get('purged_training_groups')):
            raise ReplayError('Historical training groups differ')
        for name in ((baseline,) if baseline_only else (baseline, baseline + '_M')):
            pred = np.asarray(result['predictions'][name])
            if pred.shape != panel['target'].shape or not np.isfinite(pred).all():
                raise ReplayError(f'{name}: incomplete/nonfinite prediction support')
            if fold['dimensions'][name] != old['dimensions'][name]:
                raise ReplayError(f'{name}: historical design width differs')
            alpha = old['alpha'][name] if historical_folds is not None else old['alpha'][index][name]
            comparisons.append(compare_numeric(f'outer_{index}.alpha.{name}', fold['alpha'][name], alpha))
        if endpoint == 'abundance':
            actual_n, old_n = result['nuisance'][index], previous['nuisance'][index]
            for key in ('held_groups', 'first_stage_columns', 'first_stage_width', 'held_out_rows'):
                if actual_n[key] != old_n[key]:
                    raise ReplayError(f'Nuisance {key} differs')
            for key in ('selected_alpha', 'held_out_weighted_r2', 'response_range'):
                comparisons.append(compare_numeric(f'outer_{index}.nuisance.{key}', actual_n[key], old_n[key]))
    return assigned, comparisons


def summary_metrics(module, panel: dict, result: dict, baseline: str, endpoint: str,
                    *, baseline_only: bool = False) -> dict:
    predictions = result['predictions']
    arguments = [panel['target'], predictions[baseline], panel['group']]
    if endpoint == 'abundance':
        arguments.append(panel['domain'])
    arguments.append(panel['site'])
    groups, base = module.group_errors(*arguments)
    if baseline_only:
        out = {'groups': groups.tolist(), 'baseline_mse': base.tolist(),
               'primary_baseline_mse': module.interval(base)}
        if endpoint == 'abundance':
            out['primary_baseline_within_unit_spearman'] = module.raw_spearman(panel, predictions[baseline])
        return out
    arguments[1] = predictions[baseline + '_M']
    other_groups, augmented = module.group_errors(*arguments)
    if groups.tolist() != other_groups.tolist():
        raise ReplayError('Baseline/augmented metric group support differs')
    out = {'groups': groups.tolist(), 'baseline_mse': base.tolist(), 'augmented_mse': augmented.tolist(),
           'primary_likelihood': module.paired_increment(panel, predictions, baseline + '_M', baseline),
           'primary_baseline_mse': module.interval(base),
           'primary_likelihood_spearman': module.spearman_increment(panel, predictions, baseline + '_M', baseline)}
    if endpoint == 'abundance':
        out['primary_baseline_within_unit_spearman'] = module.raw_spearman(panel, predictions[baseline])
    return out


def compare_metrics(actual: dict, previous: dict, endpoint: str, *, baseline_only: bool = False) -> list[dict]:
    if endpoint == 'stability':
        if actual['groups'] != previous['groups']:
            raise ReplayError('Group MSE labels/order differs from H200 reference')
        return [compare_numeric(key, actual[key], previous[key]) for key in
                (('baseline_mse',) if baseline_only else ('baseline_mse', 'augmented_mse'))]
    checks = []
    for key in (('primary_baseline_mse', 'primary_baseline_within_unit_spearman') if baseline_only else
                ('primary_likelihood', 'primary_baseline_mse',
                 'primary_likelihood_spearman', 'primary_baseline_within_unit_spearman')):
        for metric in ('point', 'baseline_mse', 'augmented_mse', 'interval'):
            if metric in previous[key]:
                checks.append(compare_numeric(f'{key}.{metric}', actual[key][metric], previous[key][metric]))
        for metric in ('groups', 'undefined_groups', 'evaluated_groups'):
            if metric in previous[key] and actual[key][metric] != previous[key][metric]:
                raise ReplayError(f'{key}.{metric}: metric support differs')
    return checks


@contextmanager
def csv_stream(path: Path):
    # filename='' suppresses source filename; mtime=0 makes identical data byte-identical.
    with path.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as text:
            writer = csv.DictWriter(text, fieldnames=FIELDS, lineterminator='\n')
            writer.writeheader()
            yield writer


def serialize_seed(writer, records: list[dict], *, endpoint: str, arm: str, seed: int,
                   baseline: str, status: str, cohort_sha256: str, feature_sha256: str = '',
                   result: dict | None = None, heldout=None, baseline_only: bool = False) -> None:
    import numpy as np
    if result is not None:
        if heldout is None or len(heldout) != len(records):
            raise ReplayError('Missing actual-fit held-out join')
        for name in ((baseline,) if baseline_only else (baseline, baseline + '_M')):
            values = np.asarray(result['predictions'][name])
            if values.shape != (len(records),) or not np.isfinite(values).all():
                raise ReplayError('Baseline/augmented prediction support differs')
    fold_numbers = [] if heldout is None else heldout
    for index, row in enumerate(records):
        values = dict(row, endpoint=endpoint, arm=arm, seed=seed,
                      heldout_fold='' if result is None else int(fold_numbers[index]),
                      heldout='' if result is None else True, baseline_name=baseline,
                      augmented_name=baseline + '_M' if baseline else '', status=status,
                      provenance=NAMESPACE, cohort_sha256=cohort_sha256,
                      feature_manifest_sha256=feature_sha256,
                      baseline_prediction='' if result is None else float(result['predictions'][baseline][index]),
                      augmented_prediction='' if result is None or baseline_only else float(result['predictions'][baseline + '_M'][index]))
        writer.writerow(values)


def runtime_record() -> dict:
    import numpy as np
    import sklearn
    import torch
    from threadpoolctl import threadpool_info
    torch.set_num_threads(4)
    pools = threadpool_info()
    if any(p['num_threads'] != 4 for p in pools if p['user_api'] in ('blas', 'openmp')):
        raise ReplayError('Loaded BLAS/OpenMP pool is not pinned to four threads; start a fresh CLI process')
    return {'location': 'localhost L20 server', 'fit_device': 'cpu', 'model_inference': False,
            'python': platform.python_version(), 'numpy': np.__version__,
            'torch': torch.__version__, 'sklearn': sklearn.__version__,
            'platform': {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine()},
            'threads': 4, 'threadpools': [{k: p[k] for k in ('user_api', 'internal_api', 'num_threads', 'version')} for p in pools],
            'resources': {'cpus': os.cpu_count(), 'free_disk_bytes': shutil.disk_usage(ROOT).free,
                          'meminfo_kib': {line.split(':')[0]: int(line.split()[1])
                              for line in Path('/proc/meminfo').read_text().splitlines()
                              if line.split(':')[0] in ('MemTotal', 'MemAvailable', 'SwapFree')}}, 
            'difference_from_original': 'Local CPU fit; supplied feature hashes and extraction settings may differ from H200. Current relocated code is not asserted frozen-equivalent.'}


def code_hashes(endpoint: str) -> dict:
    paths = [Path(__file__), ROOT / 'src/capability/readouts/readout_analysis.py',
             ROOT / 'src/capability/interactions/pairwise_epistasis.py',
             ROOT / 'src/capability/stability/stability_gate.py',
             ROOT / 'src/capability/mutation/external_confirmation.py',
             ROOT / 'src/capability/readouts/readout_extraction.py',
             ROOT / 'src/capability/context/context_homologue.py',
             ROOT / 'scripts/capability/stability/extract_stability_singles.py',
             ROOT / ('scripts/capability/gates/fit_nested_singles.py' if endpoint == 'abundance'
                     else 'scripts/capability/stability/fit_stability_singles.py')]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def replay_arm(root: Path, out: Path, extraction: Path | None, endpoint: str, arm: str,
               runtime: dict, *, check_only: bool = False, baseline_only: bool = False,
               fit_cache: dict | None = None) -> dict:
    started = time.monotonic()
    paths = input_paths(root, endpoint, arm)
    report = {'schema': 'phenotype_prediction_replay_v1', 'namespace': NAMESPACE,
              'endpoint': endpoint, 'arm': arm, 'runtime': runtime, 'code_sha256': code_hashes(endpoint),
              'atol': ATOL, 'rtol': RTOL, 'seeds': {}, 'original_artifacts_unchanged': None,
              'baseline_only': baseline_only,
              'augmented_status': 'unavailable_not_requested_likelihood_extraction' if baseline_only else 'pending'}
    records, baseline, hashes, feature_hashes = [], '', {}, {}
    csv_path = out / f'{arm}.csv.gz'
    try:
        cohort = read_json(paths['cohort'])
        schema = 'external_confirmation_cohort_v1' if endpoint == 'abundance' else 'stability_singles_cohort_v1'
        if cohort['schema'] != schema:
            raise ReplayError('Wrong source cohort schema')
        hashes['cohort'] = digest(paths['cohort'])
        records = registry(cohort, endpoint, hashes['cohort'])
        report['rows_per_seed'] = len(records)
        controls, reference = read_json(paths['controls']), read_json(paths['reference'])
        hashes = {key: digest(path) for key, path in paths.items()}
        report['original_input_sha256'] = hashes
        baseline, columns = validate_reference(endpoint, arm, records, controls, reference, hashes)
        report.update(baseline=baseline, designs={baseline: list(columns), baseline + '_M': [*columns, 'M']})
        report['original_reference_hashes'] = {k: v for k, v in reference.items() if k.endswith('_sha256')}
        units = cohort['domains' if endpoint == 'abundance' else 'backgrounds']
        if baseline_only:
            report['designs'] = {baseline: list(columns)}
            model_blocks = {}
            if baseline == 'S_T':
                token_records, token_provenance = tokenizer_records(arm, units)
                model_blocks = token_blocks(units, token_records)
                report['tokenizer_provenance'] = token_provenance
            report['feature_sha256'] = {'source': 'original cohort and profiles; native tokenizer only when primary is S_T'}
        else:
            if extraction is None:
                raise ReplayError('Likelihood replay requires --extraction; use --baseline-only for baseline recovery')
            directory = extraction / arm
            plan, expected_plan_sha256 = canonical_plan(root, endpoint, hashes['cohort'], units)
            manifest, feature_hashes = validate_features(directory, arm, hashes['cohort'], units,
                plan=plan, expected_plan_sha256=expected_plan_sha256)
            report['feature_sha256'] = feature_hashes
            report['extraction_settings'] = {k: manifest['identity'].get(k) for k in
                                             ('dtype', 'batch_size', 'budget', 'plan_sha256', 'precision_gate',
                                              'code_sha256', 'checkpoint_metadata_sha256')}
            report['feature_matches_original_manifest'] = (
                feature_hashes['manifest'] == reference['extraction_manifest_sha256']
                if 'extraction_manifest_sha256' in reference else None)
            model_blocks = load_likelihood_blocks(directory, units, manifest)
        if check_only:
            report['status'] = 'inputs_validated_not_fitted'
        else:
            from src.capability.stability import stability_gate as stability
            if endpoint == 'abundance':
                from src.capability.mutation import external_confirmation as abundance_module
                module = abundance_module
            else:
                module = stability
            if (tuple(module.SPLIT_SEEDS) != SEEDS or tuple(module.ALPHAS) != (.01, .1, 1., 10., 100.)
                    or module.OUTER_SPLITS != 5 or module.INNER_SPLITS != 4):
                raise ReplayError('Current fitting constants differ from the historical recipe')
            profiles, _ = stability.load_profiles(paths['profiles'], {u['name']: u['wildtype'] for u in units})
            if endpoint == 'abundance':
                from src.capability.mutation import external_confirmation as abundance_module
                panel = abundance_module.build_panel(units, profiles)
            else:
                panel = stability.build_panel(cohort, profiles)
            validate_panel(panel, records, model_blocks)
            if endpoint == 'abundance':
                identity = abundance_module.row_identity(panel['group'], panel['site'], panel['target'])
                if identity != controls['row_identity_sha256'] or identity != reference['row_identity_sha256']:
                    raise ReplayError('Original abundance row identity changed')
            blocks = dict(panel['blocks'], **model_blocks)
            designs = {baseline: columns}
            if not baseline_only:
                designs[baseline + '_M'] = (*columns, 'M')
            # Baselines depend only on cohort/profile/control hashes, declared columns,
            # seed and (for S_T) exact T bytes, never the arm or likelihood block.
            import numpy as np
            cache_identity = (endpoint, hashes['cohort'], hashes['profiles'], hashes['controls'], columns,
                digest(paths['controls']), hashlib.sha256(np.ascontiguousarray(model_blocks['T']).tobytes()).hexdigest()
                if baseline == 'S_T' else None)
            with csv_stream(csv_path) as writer:
                for seed in SEEDS:
                    result, partitions = None, []
                    try:
                        kwargs = {'first_stage': tuple(c for c in controls['qualified_control_set'] if c != 'G')} if endpoint == 'abundance' else {'purge': None}
                        cache_key = (*cache_identity, seed)
                        reused_from = None
                        if baseline_only and fit_cache is not None and cache_key in fit_cache:
                            result, partitions, reused_from = fit_cache[cache_key]
                        else:
                            with capture_partitions(module) as partitions:
                                result = module.fold_predictions(panel, blocks, designs, seed=seed, device='cpu', **kwargs)
                            if baseline_only and fit_cache is not None:
                                fit_cache[cache_key] = (result, partitions, arm)
                        heldout, comparisons = validate_fit(result, reference, endpoint, seed, baseline, panel, partitions,
                                                           baseline_only=baseline_only)
                        summary = summary_metrics(module, panel, result, baseline, endpoint, baseline_only=baseline_only)
                        previous = reference['per_seed' if endpoint == 'abundance' else 'seeds'][str(seed)]
                        comparisons += compare_metrics(summary, previous, endpoint, baseline_only=baseline_only)
                        status = 'reproduced_within_tolerance' if all(c['agrees'] for c in comparisons) else 'reproduced_metric_mismatch'
                        if baseline_only:
                            status = 'baseline_' + status + '_augmented_unavailable'
                        report['seeds'][str(seed)] = {'status': status, 'folds': result['folds'],
                            'actual_partitions': partitions, 'nuisance': result['nuisance'],
                            'summary': summary, 'comparison': comparisons, 'fit_reused_from_arm': reused_from} 
                        serialize_seed(writer, records, endpoint=endpoint, arm=arm, seed=seed, baseline=baseline,
                                       status=status, cohort_sha256=hashes['cohort'], feature_sha256=feature_hashes.get('manifest', ''),
                                       result=result, heldout=heldout, baseline_only=baseline_only)
                    except Exception as error:
                        report['seeds'][str(seed)] = failure_record(error)
                        if result is not None:
                            report['seeds'][str(seed)].update(folds=result['folds'],
                                actual_partitions=partitions, nuisance=result['nuisance'])
                        serialize_seed(writer, records, endpoint=endpoint, arm=arm, seed=seed, baseline=baseline,
                                       status='failed', cohort_sha256=hashes['cohort'], feature_sha256=feature_hashes.get('manifest', ''))
            statuses = [r['status'] for r in report['seeds'].values()]
            report['status'] = ('failed' if 'failed' in statuses else 'reproduced_metric_mismatch'
                                if any('metric_mismatch' in s for s in statuses) else 'reproduced_within_tolerance')
            if not baseline_only:
                report['augmented_status'] = report['status']
    except Exception as error:
        report.update(failure_record(error))
        report['seeds'] = {str(seed): failure_record(error) for seed in SEEDS}
        if not check_only and not csv_path.exists():
            with csv_stream(csv_path) as writer:
                for seed in SEEDS:
                    serialize_seed(writer, records, endpoint=endpoint, arm=arm, seed=seed, baseline=baseline,
                                   status='failed', cohort_sha256=hashes.get('cohort', ''))
    finally:
        changed = [key for key, old in hashes.items() if not paths[key].is_file() or digest(paths[key]) != old]
        report['original_artifacts_unchanged'] = not changed if hashes else None
        if changed:
            report.update(status='failed', error='Original input changed during replay', changed_inputs=changed)
            if csv_path.exists():
                invalid = csv_path.with_name(f'{arm}.invalid.csv.gz')
                csv_path.rename(invalid)
                report['invalid_prediction_csv'] = {'file': invalid.name, 'sha256': digest(invalid),
                                                    'status': 'invalid_original_inputs_changed_do_not_plot'}
                with csv_stream(csv_path) as writer:
                    for seed in SEEDS:
                        serialize_seed(writer, records, endpoint=endpoint, arm=arm, seed=seed,
                                       baseline=baseline, status='failed', cohort_sha256=hashes.get('cohort', ''))
        if csv_path.exists():
            report['prediction_csv'] = {'file': csv_path.name, 'sha256': digest(csv_path)}
        report['seconds'] = time.monotonic() - started
        write_json(out / f'{arm}.json', report)
    return report


def failure_record(error: Exception) -> dict:
    # Do not persist arbitrary exception text from libraries/manifests (can contain remote names).
    message = str(error) if isinstance(error, ReplayError) else f'{type(error).__name__}: input or fitting failure; inspect supplied schema/arrays and current fitting API'
    return {'status': 'failed', 'error': message,
            'action': 'Supply complete per-background NPZ extraction from the unchanged cohort plan under --extraction/<arm> for augmented recovery, or request --baseline-only for original baseline recovery; rerun into a NEW output directory. No inference or zero-likelihood fallback performed.'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', choices=('abundance', 'stability'), required=True)
    arms = parser.add_mutually_exclusive_group()
    arms.add_argument('--arm', action='append', help='Repeat for multiple arms; default: frozen 33-arm roster')
    arms.add_argument('--arms', nargs='+')
    parser.add_argument('--extraction', type=Path, help='Root containing <arm>/manifest_<arm>.json; required unless --baseline-only')
    parser.add_argument('--baseline-only', action='store_true', help='Recover original S/S_T; augmented predictions explicitly unavailable')
    parser.add_argument('--out', type=Path, required=True, help='NEW directory inside the L20reproduction namespace')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--check-inputs-only', action='store_true', help='Receipts only; no fitting or prediction claims')
    args = parser.parse_args(argv)
    if not args.baseline_only and args.extraction is None:
        parser.error('--extraction is required unless --baseline-only is set')
    from src.capability.interactions.pairwise_epistasis import ROSTER
    chosen = args.arm or args.arms or list(ROSTER)
    if len(set(chosen)) != len(chosen) or any(arm not in ROSTER for arm in chosen):
        parser.error('Arms must be unique members of the frozen 33-arm roster')
    root, out = args.root.resolve(), args.out.resolve()
    allowed = root / 'results/shared' / NAMESPACE
    if not out.is_relative_to(allowed):
        parser.error(f'--out must stay in the reproduction namespace: {allowed}')
    if out.exists():
        parser.error('Refuse existing output directory; original and earlier reproductions are never overwritten')
    out.mkdir(parents=True)
    try:
        runtime = runtime_record()
    except Exception as error:
        write_json(out / 'run.json', failure_record(error))
        return 2
    declaration = {'namespace': NAMESPACE, 'endpoint': args.endpoint, 'arms': chosen, 'seeds': list(SEEDS),
                   'runtime': runtime, 'check_inputs_only': args.check_inputs_only,
                   'baseline_only': args.baseline_only,
                   'augmented_status': 'unavailable_not_requested_likelihood_extraction' if args.baseline_only else 'pending',
                   'prediction_status': 'not original artifacts; raw phenotype fits, not likelihood scores',
                   'fitting': 'unchanged primary functions; five outer/four inner folds; alpha .01,.1,1,10,100',
                   'comparison': {'atol': ATOL, 'rtol': RTOL, 'numerical_mismatch': 'retained and flagged',
                                  'schema_fold_support_mismatch': 'fail closed; blank prediction pairs'}}
    write_json(out / 'run.json', declaration)
    fit_cache = {}
    reports = [replay_arm(root, out, args.extraction.resolve() if args.extraction is not None else None,
                          args.endpoint, arm, runtime, check_only=args.check_inputs_only,
                          baseline_only=args.baseline_only, fit_cache=fit_cache) for arm in chosen]
    declaration['arm_status'] = {r['arm']: r['status'] for r in reports}
    declaration['exit_code'] = 2 if any(r['status'] == 'failed' for r in reports) else 1 if any(
        r['status'] == 'reproduced_metric_mismatch' for r in reports) else 0
    write_json(out / 'run.json', declaration)
    print(json.dumps({'endpoint': args.endpoint, 'out': str(out), 'arm_status': declaration['arm_status']}))
    return declaration['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
