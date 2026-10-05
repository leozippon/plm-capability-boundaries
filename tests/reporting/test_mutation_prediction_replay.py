"""Focused invariants for the separately identified CPU mutation replay."""
import copy
import json
from unittest.mock import patch

import numpy as np
import pytest

from scripts.capability.reporting import export_prediction_details as details
from scripts.capability.reporting import replay_mutation_predictions as replay
from src.capability.context import local_context
from src.capability.context.profile_increment import standardized_rank
from src.capability.readouts.readout_analysis import nested_predict


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def retained_fixture(root):
    assays = [{'assay': f'a{i}', 'cluster': f'g{i}', 'mutants': ['A1C', 'A1D', 'A1E']}
              for i in range(4)]
    bundle = root / 'results/R2/readout_expansion_20260923/complete'
    manifest = bundle / 'results/external_baseline/run/manifest_toy.json'
    save(manifest, {'declared': 'extraction lineage; no fitted prediction'})
    manifest_sha = details.digest(manifest)
    folds = [dict(fold=0, held_families=['g0', 'g1'], training_families=['g2', 'g3'],
                  inner_folds=[{'validation_families': ['g2']}, {'validation_families': ['g3']}]),
             dict(fold=1, held_families=['g2', 'g3'], training_families=['g0', 'g1'],
                  inner_folds=[{'validation_families': ['g0']}, {'validation_families': ['g1']}])]
    registry = {str(manifest.relative_to(bundle)): {'sha256': manifest_sha}}
    paths = []
    for seed in replay.SEEDS:
        report = dict(status='complete', support={'definition': 'common'}, n_assays=201,
                      n_families=163, n_variants=25728, arm='toy', fold_seed=seed,
                      source_sha256={'cohort.json': 'cohort-hash',
                                     '/frozen/results/external_baseline/run/manifest_toy.json': manifest_sha},
                      predictions=[dict(assay=a['assay'], mutants=a['mutants'],
                                        raw_M=standardized_rank(np.array([1., 1., 4.])).tolist(),
                                        M=[999., -999., 17.], B=[999., 999., 999.]) for a in assays],
                      folds={'M': copy.deepcopy(folds), 'B': copy.deepcopy(folds)})
        path = root / details.READOUT / f'run/readout_toy_common_fold{seed}.json'
        save(path, report)
        registry[str(path.relative_to(bundle))] = {'sha256': details.digest(path)}
        paths.append(path)
    # Old, overlapping 40-assay report is not an admitted source, even without a registry entry.
    save(root / details.READOUT / 'old/readout_toy_common.json',
         dict(status='complete', support={'definition': 'common'}, n_assays=40,
              n_families=40, n_variants=120, predictions=[{'raw_M': [999.]}]))
    registry_path = bundle / 'admission/source_registry.json'
    save(registry_path, {'files': registry})
    admission = {'cohort_sha256': 'cohort-hash', 'cells': [f'toy/{s}' for s in replay.SEEDS]}
    return assays, admission, paths, registry_path


def test_raw_feature_not_fitted_prediction_and_exact_support(tmp_path):
    assays, admission, _, registry = retained_fixture(tmp_path)
    features, folds, sources, sha = replay.retained_features(tmp_path, admission, assays)
    assert np.array_equal(features['toy'], np.tile(standardized_rank(np.array([1., 1., 4.])), 4))
    assert len(sources) == 3 and sha == details.digest(registry)
    assert set(folds) == set(replay.SEEDS)
    assert all(s['feature'] == 'predictions[].raw_M' for s in sources)


@pytest.mark.parametrize('defect', ['inner_membership', 'mutation_order', 'hash', 'raw_rank'])
def test_material_source_mismatches_fail_closed(tmp_path, defect):
    assays, admission, paths, registry_path = retained_fixture(tmp_path)
    report = json.loads(paths[0].read_text())
    if defect == 'inner_membership':
        # Same valid outer partition, different valid inner partition ordering.
        report['folds']['B'][0]['inner_folds'].reverse()
    elif defect == 'mutation_order':
        report['predictions'][0]['mutants'].reverse()
    elif defect == 'raw_rank':
        report['predictions'][0]['raw_M'] = [1., 2., 3.]
    else:
        report['arm'] = 'changed'
    save(paths[0], report)
    if defect != 'hash':
        registry = json.loads(registry_path.read_text())
        bundle = registry_path.parent.parent
        registry['files'][str(paths[0].relative_to(bundle))]['sha256'] = details.digest(paths[0])
        save(registry_path, registry)
    with pytest.raises(ValueError, match={'inner_membership': 'source design fold disagreement',
                                         'mutation_order': 'mutation row order mismatch',
                                         'hash': 'registry report hash mismatch',
                                         'raw_rank': 'rank semantics mismatch'}[defect]):
        replay.retained_features(tmp_path, admission, assays)


def test_paired_metrics_match_original_evaluator_and_family_weighting():
    rows = []
    for i in range(8):
        rows.append(dict(assay=f'a{i}', cluster=f'g{i//2}' if i < 4 else f'g{i}',
                         mutants=['A1C', 'A1D', 'A1E', 'A1F'], measured=[0., 0., 2., 4.],
                         P=[1., 2., 3., 4.], M=[2., 1., 4., 3.],
                         S=np.zeros((4, 2)), P_block=np.zeros((4, 1)),
                         R=np.zeros((4, 1)), wall=np.zeros((4, 2))))
    y = np.concatenate([r['measured'] for r in rows])
    indices = {r['assay']: np.arange(4*i, 4*i+4) for i, r in enumerate(rows)}
    baseline = np.tile([0., 0., 1., 2.], len(rows))
    augmented = np.tile([0., 1., 2., 3.], len(rows))
    # Undefined correlation must be excluded per metric, not silently replaced with zero.
    baseline[indices['a0']] = 0.
    augmented[indices['a3']] = 0.
    with patch.object(local_context, 'nested_predict', side_effect=lambda x, *a, **kw:
                      (baseline if x.shape[1] == 5 else augmented, [])):
        original, _ = local_context.evaluate_local_context(
            rows, local_blocks=['wall'], additions={'': (), '+M': ('M',)}, bootstrap=20)
    points = replay.paired_points(rows, y, indices, baseline, augmented)
    assert len(points) == 6
    for key, value in points.items():
        assert value == original['summaries'][key]['point']
    assert replay.point_metrics([{'cluster': 'a', 'v': 0.}, {'cluster': 'a', 'v': 2.},
                                 {'cluster': 'b', 'v': 5.}, {'cluster': 'c', 'v': None}], ['v']) == {'v': 3.}
    assert replay.point_metrics([{'cluster': 'a', 'v': None}], ['v']) == {'v': None}


def test_original_design_and_nested_fit_are_deterministic_on_cpu():
    import torch
    torch.set_num_threads(4)
    rng = np.random.default_rng(42)
    n = 60
    blocks = {'S': rng.normal(size=(n, 2)), 'P': rng.normal(size=(n, 1)),
              'wall': rng.normal(size=(n, 2)), 'M': rng.normal(size=(n, 1))}
    designs = local_context.assemble_designs(blocks, {replay.BASE: ('S', 'P', 'wall')},
                                            {'': (), '+M': ('M',)})
    assert np.array_equal(designs[replay.AUGMENTED][:, :-1], designs[replay.BASE])
    assert np.array_equal(designs[replay.AUGMENTED][:, -1], blocks['M'][:, 0])
    aids = np.repeat([f'a{i}' for i in range(10)], 6)
    groups = np.repeat([f'g{i}' for i in range(10)], 6)
    y = rng.normal(size=n)
    pred, folds = nested_predict(designs[replay.BASE], y, aids, groups, seed=replay.SEEDS[0])
    same, same_folds = nested_predict(designs[replay.BASE], y, aids, groups, seed=replay.SEEDS[0])
    _, augmented_folds = nested_predict(designs[replay.AUGMENTED], y, aids, groups, seed=replay.SEEDS[0])
    assert np.array_equal(pred, same) and folds == same_folds
    assert local_context.fold_membership(folds) == local_context.fold_membership(augmented_folds)
    mapping = details.outer_mapping(folds, set(groups), held='held_families', training='training_families')
    assert all(g in folds[mapping[g]]['held_families'] and g not in folds[mapping[g]]['training_families']
               for g in groups)


def test_summary_tolerance_does_not_hide_discrepancies_or_nulls():
    comparison = replay.compare_points({'ok': 1. + 1e-10, 'bad': 1.1, 'null': None,
                                        'missing': None},
                                       {'ok': {'point': 1.}, 'bad': {'point': 1.},
                                        'null': {'point': None}, 'missing': {'point': 0.}})
    assert comparison['ok']['consistent'] and comparison['null']['consistent']
    assert not comparison['bad']['consistent'] and not comparison['missing']['consistent']
    assert comparison['bad']['delta'] == pytest.approx(.1)
