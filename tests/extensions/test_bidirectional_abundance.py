"""The abundance three-arm declaration, its two paired families and its reproduction gate.

These tests are data-independent: they exercise the declared designs, the two
readings, the common-support rule and the pilot gate on synthetic records, so they
run before the archived per-arm extraction is restored and without pretending that
it is.
"""
import numpy as np
import pytest

from scripts.capability.extensions.bidirectional_ablation_20261106 import r2_abundance as module
from src.capability.extensions import bidirectional

SEEDS = module.SEEDS
QUALIFIED = ('ident', 'geom', 'comp', 'chem', 'prof', 'prof2', 'G')
#: The archived per-fold penalty record, exactly as ``fit_<arm>.json`` holds it:
#: S, S2 and R designs per arm, and no pure likelihood-only design.
ARCHIVE_ALPHA = {'S': .01, 'S_M': .01, 'S_R': .01, 'S_T': .1, 'S_T_M': 1., 'S_T_R': .01,
                 'S2': .01, 'S2_M': .01, 'S2_R': .01, 'S2_T': .01, 'S2_T_M': .01, 'S2_T_R': .01}
ROLES_BY_NAME = {'C': 'S_T', 'M': 'M', 'CM': 'S_T_M'}


# --------------------------------------------------------------------------- #
# The three arms.
# --------------------------------------------------------------------------- #

def test_pure_likelihood_arm_is_one_column_and_never_carries_controls():
    designs, roles = module.fold_designs('S_T', QUALIFIED)
    assert roles == ROLES_BY_NAME
    assert list(designs['M']) == ['M']
    assert designs[roles['C']] == QUALIFIED + ('T',)
    assert designs[roles['CM']] == QUALIFIED + ('T', 'M')
    # G is a training-fold response and belongs to the controls; T is the arm's own
    # tokenisation nuisance and also belongs to the controls. Neither reaches M.
    assert 'G' in designs[roles['C']] and 'G' in designs[roles['CM']]
    assert 'G' not in designs['M'] and 'T' not in designs['M']

    designs, roles = module.fold_designs('S', QUALIFIED)
    assert set(designs) == {'S', 'S_M', 'M'}
    assert designs[roles['C']] == QUALIFIED
    assert designs[roles['CM']] == QUALIFIED + ('M',)
    assert list(designs['M']) == ['M']


def test_control_columns_follow_the_archived_qualification_only():
    assert module.control_columns('S', QUALIFIED) == QUALIFIED
    assert module.control_columns('S_T', QUALIFIED) == QUALIFIED + ('T',)
    with pytest.raises(ValueError, match='unknown archived baseline'):
        module.control_columns('S2_T', QUALIFIED)
    with pytest.raises(ValueError, match='already carries'):
        module.control_columns('S_T', QUALIFIED + ('T',))


def test_archived_baseline_is_read_and_never_requalified():
    assert module.archived_baseline({'matched_baseline': {'primary': 'S_T',
                                                          'secondary': 'S2_T'}}, 'arm') == 'S_T'
    assert module.archived_baseline({'matched_baseline': {'primary': 'S'}}, 'arm') == 'S'
    with pytest.raises(module.replay.ReplayError, match='never re-qualifies'):
        module.archived_baseline({'matched_baseline': {'primary': 'S2_T'}}, 'arm')
    with pytest.raises(module.replay.ReplayError):
        module.archived_baseline({}, 'arm')


def test_missing_archive_is_refused_rather_than_regenerated(tmp_path):
    with pytest.raises(module.replay.ReplayError, match='never regenerated'):
        module.locate_extraction(tmp_path, 'gpt2')
    (tmp_path / 'ec-gpt2').mkdir()
    (tmp_path / 'ec-gpt2' / 'manifest_gpt2.json').write_text('{}')
    assert module.locate_extraction(tmp_path, 'gpt2') == tmp_path / 'ec-gpt2'


# --------------------------------------------------------------------------- #
# What a reproduced fold must match, and what the archive cannot tell us.
# --------------------------------------------------------------------------- #

def archive_reference(seed=SEEDS[0], folds=5):
    return {'per_seed': {str(seed): {
        'alpha': [dict(ARCHIVE_ALPHA) for _ in range(folds)],
        'dimensions': {'S_T': 528, 'S_T_M': 529},
        'nuisance': [{'selected_alpha': 1.} for _ in range(folds)]}}}


def test_archived_expectations_cover_only_the_reused_arms():
    assert 'M' not in ARCHIVE_ALPHA
    expectations = module.archived_expectations(archive_reference(), SEEDS[0], ROLES_BY_NAME)
    assert set(expectations) == set(module.REUSED) == {'C', 'CM'}
    assert 'M' not in expectations
    assert [entry['alpha'] for entry in expectations['CM']] == [1.] * 5
    assert [entry['dimensions'] for entry in expectations['CM']] == [529] * 5
    assert [entry['alpha'] for entry in expectations['C']] == [.1] * 5


def test_a_changed_reused_penalty_or_width_is_refused():
    reference, roles = archive_reference(), ROLES_BY_NAME
    expectations = module.archived_expectations(reference, SEEDS[0], roles)
    folds = [{'fold': index, 'alpha': {'C': .1, 'M': .01, 'CM': 1.},
              'dimensions': {'C': 528, 'M': 1, 'CM': 529}} for index in range(5)]
    # The pure likelihood arm keeps its own penalty and is never compared to an
    # archived one, because the archive has none.
    module.reproduced_expectations(folds, expectations, arm='gpt2', seed=SEEDS[0])

    changed = [dict(fold) for fold in folds]
    changed[2] = dict(changed[2], alpha=dict(changed[2]['alpha'], CM=.1))
    with pytest.raises(module.replay.ReplayError, match='CM penalty differs'):
        module.reproduced_expectations(changed, expectations, arm='gpt2', seed=SEEDS[0])
    widths = [dict(fold) for fold in folds]
    widths[1] = dict(widths[1], dimensions=dict(widths[1]['dimensions'], C=520))
    with pytest.raises(module.replay.ReplayError, match='C design width differs'):
        module.reproduced_expectations(widths, expectations, arm='gpt2', seed=SEEDS[0])
    with pytest.raises(module.replay.ReplayError, match='outer folds'):
        module.reproduced_expectations(folds[:4], expectations, arm='gpt2', seed=SEEDS[0])


# --------------------------------------------------------------------------- #
# The paired reductions of both readings.
# --------------------------------------------------------------------------- #

def test_pairing_reads_both_directions_on_the_same_joint_arm():
    assert list(module.DESIGNS) == list(bidirectional.DIRECTIONS)
    assert module.DESIGNS == {'model_given_controls': ('C', 'CM'),
                              'controls_given_model': ('M', 'CM')}
    errors = {'C': np.array([4., 5., 6.]), 'M': np.array([7., 8., 9.]),
              'CM': np.array([1., 1., 1.])}
    reductions = bidirectional.directional_reductions(errors, module.DESIGNS)
    assert np.array_equal(reductions[:, 0], errors['C'] - errors['CM'])
    assert np.array_equal(reductions[:, 1], errors['M'] - errors['CM'])
    # Both columns share the joint arm, so their difference is the difference of
    # the two reduced arms on the same rows.
    assert np.allclose(reductions[:, 0] - reductions[:, 1], errors['C'] - errors['M'],
                       rtol=0, atol=1e-12)


def record(arm, values, *, status='complete', seeds=SEEDS, groups=('g1', 'g2', 'g3')):
    return {'arm': arm, 'status': status, 'groups': list(groups),
            'seeds': {str(seed): values(seed) for seed in seeds}}


def readings(mse=None, rank=None):
    def values(seed):
        return {'mse': mse or {'C': [1., 1., 1.], 'M': [1., 1., 1.], 'CM': [1., 1., 1.]},
                'rank': rank or {'C': [0., 0., 0.], 'M': [0., 0., 0.], 'CM': [0., 0., 0.]}}
    return values


def test_seeds_are_averaged_inside_a_group_before_groups_enter_equally():
    def values(seed):
        return {'mse': {'C': {'20260923': [1., 2., 3.], '20260924': [3., 2., 1.],
                              '20260925': [2., 2., 2.]}[str(seed)],
                        'M': [0., 0., 0.], 'CM': [0., 0., 0.]},
                'rank': {'C': [0., 0., 0.], 'M': [0., 0., 0.], 'CM': [0., 0., 0.]}}

    records = [record('a', values), record('b', values)]
    family = module.panel_from_records(records, arms=['a', 'b'])['metrics']['squared_error']
    assert family['contrasts'].shape == (3, 4)
    assert family['columns'] == ['a|model_given_controls', 'a|controls_given_model',
                                 'b|model_given_controls', 'b|controls_given_model']
    # Every seed means 2 in every group, so the group-equal mean is 2 as well and
    # the model column is a flat zero.
    assert np.allclose(family['contrasts'][:, 0], 2., rtol=0, atol=1e-12)
    assert np.allclose(family['contrasts'][:, 1], 0., rtol=0, atol=1e-12)
    assert family['undefined_groups'] == []
    # Reversing the seed order cannot change an average taken inside the group.
    reordered = module.panel_from_records(records, arms=['a', 'b'],
                                          seeds=SEEDS[::-1])['metrics']['squared_error']
    assert np.array_equal(reordered['contrasts'], family['contrasts'])
    assert family['per_seed_points']['a|model_given_controls'] == {str(seed): 2. for seed in SEEDS}
    assert family['absolute'].shape == (3, 6)
    assert np.allclose(family['absolute'].mean(axis=0), [2., 0., 0., 2., 0., 0.],
                       rtol=0, atol=1e-12)


def test_ranking_is_a_score_while_the_squared_error_is_a_loss():
    def values(seed):
        return {'mse': {'C': [0.3] * 3, 'M': [0.5] * 3, 'CM': [0.1] * 3},
                'rank': {'C': [0.3] * 3, 'M': [0.2] * 3, 'CM': [0.4] * 3}}

    assert module.METRICS['squared_error']['to_loss'] == 1.
    assert module.METRICS['ranking']['to_loss'] == -1.
    assert 'lower' in module.METRICS['squared_error']['direction']
    assert 'raises' in module.METRICS['ranking']['direction']
    metrics = module.panel_from_records([record('a', values)], arms=['a'])['metrics']
    error = metrics['squared_error']['contrasts']
    rank = metrics['ranking']['contrasts']
    # Squared error: the joint arm is better because its error is lower.
    assert np.allclose(error[:, 0], 0.2, rtol=0, atol=1e-12)
    assert np.allclose(error[:, 1], 0.4, rtol=0, atol=1e-12)
    # Ranking: the joint arm is better because its correlation is higher, so the
    # reduction is joint minus reduced and both directions are positive here.
    assert np.allclose(rank[:, 0], 0.1, rtol=0, atol=1e-12)
    assert np.allclose(rank[:, 1], 0.2, rtol=0, atol=1e-12)
    # The absolute scores are the arms' own readings, never sign-flipped.
    assert np.allclose(metrics['ranking']['absolute'].mean(axis=0), [0.3, 0.2, 0.4],
                       rtol=0, atol=1e-12)


def test_an_undefined_rank_group_is_dropped_from_every_arm():
    groups = ('g1', 'g2', 'g3', 'g4')

    def values(seed):
        # The likelihood-only arm has no defined rank correlation in g3 at one seed.
        rank = {'C': [.1, .1, .1, .1], 'M': [.2, .2, None, .2], 'CM': [.3, .3, .3, .3]}
        return {'mse': {'C': [1., 1., 1., 1.], 'M': [1., 1., 1., 1.], 'CM': [1., 1., 1., 1.]},
                'rank': rank if seed == SEEDS[0] else dict(rank, M=[.2, .2, .2, .2])}

    family = module.panel_from_records([record('a', values, groups=groups)], arms=['a'])['metrics']
    assert family['squared_error']['groups'] == list(groups)
    assert family['squared_error']['undefined_groups'] == []
    # One reading cannot silently keep a group the other arm does not define.
    assert family['ranking']['groups'] == ['g1', 'g2', 'g4']
    assert family['ranking']['undefined_groups'] == ['g3']
    assert family['ranking']['contrasts'].shape == (3, 2)

    def undefined(seed):
        return {'mse': {'C': [1., 1., 1.], 'M': [1., 1., 1.], 'CM': [1., 1., 1.]},
                'rank': {'C': [None, None, None], 'M': [.2, .2, .2], 'CM': [.3, .3, .3]}}

    with pytest.raises(ValueError, match='at least two'):
        module.panel_from_records([record('a', undefined)], arms=['a'])


def test_the_declared_family_has_sixty_six_columns_per_metric():
    def values(seed):
        return {'mse': {'C': [0.3, 0.2, 0.1], 'M': [0.4, 0.4, 0.4], 'CM': [0.1, 0.1, 0.1]},
                'rank': {'C': [.20, .20, .20], 'M': [.10, .10, .10], 'CM': [.25, .25, .25]}}

    arms = [f'arm-{index:02d}' for index in range(len(module.ROSTER))]
    family = module.panel_from_records([record(arm, values) for arm in arms], arms=arms)
    assert list(family['metrics']) == list(module.METRICS)
    for name, reading in family['metrics'].items():
        assert reading['contrasts'].shape == (3, bidirectional.FAMILY_SIZE), name
        assert len(reading['columns']) == len(set(reading['columns'])) == bidirectional.FAMILY_SIZE
        verdicts = module.summarise_verdicts(
            reading['columns'], bidirectional.verdict(
                bidirectional.paired_bands(reading['contrasts'], draws=500)['interval']))
        for direction in bidirectional.DIRECTIONS:
            assert sum(verdicts[direction]['counts'].values()) == len(module.ROSTER)
            assert sorted(verdicts[direction]['positive'] + verdicts[direction]['negative']
                          + verdicts[direction]['unresolved']) == arms


def test_verdicts_are_unresolved_when_the_band_spans_zero():
    groups = tuple(f'g{index}' for index in range(8))
    alternating = [0.01, -0.01] * 4
    swing = [0.01, -0.01] * 4

    def values(seed):
        # Both directions swing around zero group by group, so the point estimate is
        # zero while the resampled deviations are not.
        return {'mse': {'C': [value + step for value, step in zip(alternating, swing)],
                        'M': [value - step for value, step in zip(alternating, swing)],
                        'CM': alternating},
                'rank': {'C': [0.] * 8, 'M': [0.] * 8, 'CM': [0.] * 8}}

    family = module.panel_from_records(
        [record('flat', values, groups=groups),
         record('pos', readings(mse={'C': [1.] * 8, 'M': [2.] * 8, 'CM': [0.5] * 8},
                                rank={'C': [.1] * 8, 'M': [.05] * 8, 'CM': [.2] * 8}),
                groups=groups)], arms=['flat', 'pos'])
    error = family['metrics']['squared_error']
    bands = bidirectional.paired_bands(error['contrasts'], draws=2000)
    verdicts = module.summarise_verdicts(error['columns'], bidirectional.verdict(bands['interval']))
    for direction in bidirectional.DIRECTIONS:
        assert verdicts[direction]['unresolved'] == ['flat']
        assert verdicts[direction]['positive'] == ['pos']
        assert verdicts[direction]['counts'] == {'positive': 1, 'negative': 0, 'unresolved': 1}
    # A family whose resampled deviations all vanish has a zero-width band, and zero
    # is still not an exclusion, so such a column stays unresolved.
    rank = family['metrics']['ranking']
    rank_bands = bidirectional.paired_bands(rank['contrasts'], draws=500)
    rank_verdicts = module.summarise_verdicts(rank['columns'],
                                             bidirectional.verdict(rank_bands['interval']))
    assert rank_verdicts['model_given_controls']['unresolved'] == ['flat']


def test_an_incomplete_family_is_never_reduced():
    arms = ['a', 'b']
    with pytest.raises(ValueError, match='did not complete'):
        module.panel_from_records([record('a', readings()),
                                   record('b', readings(), status='failed')], arms=arms)
    with pytest.raises(ValueError, match='exactly one record per arm'):
        module.panel_from_records([record('a', readings())], arms=arms)
    with pytest.raises(ValueError, match='no fit record'):
        module.panel_from_records([record('a', readings()),
                                   record('b', readings(), seeds=SEEDS[:2])], arms=arms)


# --------------------------------------------------------------------------- #
# The reproduction gate.
# --------------------------------------------------------------------------- #

def fit_record(arm='gpt2', baseline='S_T', difference=1e-12, alpha=.1, response=1.):
    """A record shaped like this stage's own output for one arm."""
    folds = [{'fold': fold,
              'alpha': {'C': alpha, 'M': .01, 'CM': 1.},
              'alpha_expected': {'C': alpha, 'CM': 1.},
              'dimensions': {'C': 528, 'M': 1, 'CM': 529},
              'dimensions_expected': {'C': 528, 'CM': 529}} for fold in range(5)]
    nuisance = [{'fold': fold, 'selected_alpha': response, 'archived_selected_alpha': 1.}
                for fold in range(5)]
    return {'schema': module.FIT_SCHEMA, 'pilot': False, 'arm': arm, 'status': 'complete',
            'baseline': baseline,
            'hashes': {'extraction_manifest': 'a' * 64, 'archives_digest': 'b' * 64,
                       'reference': 'c' * 64},
            'sample_identity': {'row_identity_sha256': 'd' * 64},
            'seeds': {str(seed): {'folds': folds, 'disagreements': 0, 'nuisance': nuisance,
                                  'mse': {'C': [1., 1.], 'M': [1., 1.], 'CM': [1., 1.]},
                                  'rank': {'C': [.1, .1], 'M': [.2, .2], 'CM': [.3, .3]},
                                  'archived_metric_max_absolute_difference': {'all': difference}}
                      for seed in SEEDS},
            'groups': ['g1', 'g2']}


def pilot_record(source, **overrides):
    arm = source['arm']
    record = {'schema': module.PILOT_SCHEMA, 'pilot': True, 'status': 'complete',
              'tolerance': module.PILOT_TOLERANCE, 'atol': module.ATOL, 'rtol': module.RTOL,
              'contract_sha256': 'e' * 64, 'code_sha256': {'script': 'f' * 64},
              'shared_hashes': {'cohort': '1' * 64},
              'sample_identity': source['sample_identity'],
              'reused_designs': list(module.REUSED), 'seeds': [str(seed) for seed in SEEDS],
              'outer_folds': 5,
              'arms': {arm: {'baseline': source['baseline'], 'hashes': source['hashes'],
                             'sample_identity': source['sample_identity'],
                             'seeds': {seed: {
                                 'folds': [{key: fold[key] for key in ('fold', 'alpha', 'alpha_expected')}
                                           for fold in entry['folds']],
                                 'response': {
                                     'selected_alpha': [fold['selected_alpha']
                                                        for fold in entry['nuisance']],
                                     'archived_selected_alpha': [
                                         fold['archived_selected_alpha']
                                         for fold in entry['nuisance']]},
                                 'disagreements': entry['disagreements'],
                                 'archived_metric_max_absolute_difference':
                                     entry['archived_metric_max_absolute_difference']}
                                 for seed, entry in source['seeds'].items()}}}}
    record.update(overrides)
    return record


def gate(pilot, records, **overrides):
    arguments = {'contract_sha256': 'e' * 64, 'code': {'script': 'f' * 64},
                 'shared': {'cohort': '1' * 64},
                 'sample_identity': records[0]['sample_identity'], 'records': records}
    arguments.update(overrides)
    return module.require_pilot(pilot, **arguments)


def test_pilot_gate_accepts_a_three_seed_reproduction():
    fitted = fit_record()
    outcome = gate(pilot_record(fitted), [fitted])
    assert outcome == {'arms': ['gpt2'], 'seeds': 3, 'outer_folds': 5,
                       'designs': ['C', 'CM'], 'worst_archived_metric_difference': 1e-12}
    # The pure likelihood arm records its own penalty but has no archived one.
    pilot = pilot_record(fitted)
    for entry in pilot['arms']['gpt2']['seeds'].values():
        assert 'M' in entry['folds'][0]['alpha']
        assert 'M' not in entry['folds'][0]['alpha_expected']
    assert gate(pilot, [fitted])['designs'] == ['C', 'CM']


def test_pilot_gate_refuses_an_ordinary_fit_record():
    fitted = fit_record()
    with pytest.raises(ValueError, match='not a pilot'):
        gate(fitted, [fitted])
    with pytest.raises(ValueError, match='a pilot is a record object'):
        gate(None, [fitted])
    with pytest.raises(ValueError, match='not complete'):
        gate(pilot_record(fitted, status='running'), [fitted])
    with pytest.raises(ValueError, match='schema'):
        gate(pilot_record(fitted, schema='other'), [fitted])


def test_pilot_gate_refuses_a_missed_penalty_or_metric_reproduction():
    fitted = fit_record()
    missed = pilot_record(fitted)
    missed['arms']['gpt2']['seeds'][str(SEEDS[1])]['folds'][2]['alpha']['CM'] = 10.
    with pytest.raises(ValueError, match='archived CM penalty'):
        gate(missed, [fitted])
    absent = pilot_record(fitted)
    del absent['arms']['gpt2']['seeds'][str(SEEDS[0])]['folds'][0]['alpha']['M']
    with pytest.raises(ValueError, match='lacks the M penalty'):
        gate(absent, [fitted])
    drifted = pilot_record(fit_record(difference=2e-7))
    with pytest.raises(ValueError, match='above'):
        gate(drifted, [fit_record()])
    response = pilot_record(fit_record(response=10.))
    with pytest.raises(ValueError, match='response penalty'):
        gate(response, [fit_record(response=10.)])
    short = pilot_record(fitted)
    del short['arms']['gpt2']['seeds'][str(SEEDS[2])]
    with pytest.raises(ValueError, match='all split seeds'):
        gate(short, [fitted])
    truncated = pilot_record(fitted)
    truncated['arms']['gpt2']['seeds'][str(SEEDS[0])]['folds'].pop()
    with pytest.raises(ValueError, match='outer folds'):
        gate(truncated, [fitted])


def test_pilot_gate_never_loosens_the_declared_tolerances():
    fitted = fit_record()
    with pytest.raises(ValueError, match='tolerance differs'):
        gate(pilot_record(fitted, tolerance=1e-5), [fitted])
    with pytest.raises(ValueError, match='never loosened'):
        gate(pilot_record(fitted), [fitted], tolerance=1e-5)
    with pytest.raises(ValueError, match='atol/rtol'):
        gate(pilot_record(fitted, atol=1e-6), [fitted])
    with pytest.raises(ValueError, match='designs'):
        gate(pilot_record(fitted, reused_designs=['C']), [fitted])


def test_pilot_gate_binds_the_contract_code_inputs_and_sample_identity():
    fitted = fit_record()
    with pytest.raises(ValueError, match='current contract'):
        gate(pilot_record(fitted), [fitted], contract_sha256='0' * 64)
    with pytest.raises(ValueError, match='current code'):
        gate(pilot_record(fitted), [fitted], code={'script': '0' * 64})
    with pytest.raises(ValueError, match='frozen original extraction inputs'):
        gate(pilot_record(fitted), [fitted], shared={'cohort': '0' * 64})
    with pytest.raises(ValueError, match='sample identity'):
        gate(pilot_record(fitted), [fitted], sample_identity={'row_identity_sha256': '0' * 64})
    other = fit_record(arm='gpt2')
    other['hashes'] = dict(other['hashes'], archives_digest='9' * 64)
    with pytest.raises(ValueError, match='not bound to the extraction'):
        gate(pilot_record(fitted), [other])
    with pytest.raises(ValueError, match='no completed fit record'):
        gate(pilot_record(fitted), [fit_record(arm='gpt2-xl')])
