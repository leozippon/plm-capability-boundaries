"""Native simultaneous bands, two-direction pairing and the reused-arm pilot gate."""
import json

import numpy as np
import pytest

from scripts.capability.interactions.run_residual_panel import simultaneous_bands
from src.capability.extensions import bidirectional

SEEDS = (20260923, 20260924, 20260925)


def heterogeneous_family(groups=20):
    """One tiny-effect column beside one large-spread column, on fixed draws."""

    rng = np.random.default_rng(7)
    small = 0.01 + 0.0005 * rng.standard_normal(groups)
    large = 5.0 + 3.0 * rng.standard_normal(groups)
    return np.column_stack([small, large])


def test_native_band_is_a_centered_deviation_in_the_contrast_units():
    values = heterogeneous_family()
    bands = bidirectional.paired_bands(values, draws=4000)
    assert bands['method'] == bidirectional.NATIVE_METHOD
    assert bands['family_size'] == 2 and bands['groups'] == 20
    assert np.allclose(bands['point'], values.mean(axis=0), rtol=0, atol=0)
    interval = np.asarray(bands['interval'])
    # One additive critical value in native units, not one width per column.
    width = interval[:, 1] - interval[:, 0]
    assert np.allclose(width, 2 * bands['critical_value'], rtol=1e-12, atol=0)
    assert np.allclose(width / 2, bands['critical_value'], rtol=1e-12, atol=0)
    # The declared family is at least as wide as the widest single-column deviation,
    # and it is set by the large-spread column, not by the tiny-effect one.
    assert bands['critical_value'] > 1.


def test_native_and_standardized_families_disagree_on_heterogeneous_spread():
    values = heterogeneous_family()
    native = bidirectional.paired_bands(values, draws=4000)
    standardized = simultaneous_bands(values, draws=4000, seed=bidirectional.SEED)
    native_interval = np.asarray(native['interval'])
    standardized_interval = np.asarray(standardized['interval'])
    # The studentised family rescales each column by its own group standard error.
    assert standardized['critical_value'] > 1
    assert not np.allclose(native_interval, standardized_interval)
    # The tiny-effect column is resolved by the studentised family and not by the
    # declared native one, which is exactly why the two are not interchangeable.
    assert native_interval[0, 0] < 0 < native_interval[0, 1]
    assert standardized_interval[0, 0] > 0
    widths = standardized_interval[:, 1] - standardized_interval[:, 0]
    assert not np.allclose(widths[0], widths[1])


def test_equal_spread_columns_share_the_same_draws_under_both_statistics():
    groups = 25
    base = np.random.default_rng(11).standard_normal(groups)
    values = np.column_stack([base, base + 3.0, base - 7.0, base + 0.5])
    native = bidirectional.paired_bands(values, draws=2000)
    standardized = simultaneous_bands(values, draws=2000, seed=bidirectional.SEED)
    se = values.std(axis=0, ddof=1) / np.sqrt(groups)
    assert np.allclose(se, se[0], rtol=1e-12, atol=0)
    # Equal spread reduces the studentised maximum to the native one over that spread,
    # so the two agree only because both read the identical resamples.
    assert np.isclose(standardized['critical_value'] * se[0], native['critical_value'],
                      rtol=1e-9, atol=0)
    assert np.allclose(np.asarray(native['interval']), np.asarray(standardized['interval']),
                       rtol=1e-9, atol=0)


def test_directional_pairing_keeps_both_directions_on_one_joint_arm():
    groups = 8
    errors = {'U_C': np.arange(groups, dtype=float) + 4,
              'M': np.arange(groups, dtype=float) + 1,
              'U_C_M': np.arange(groups, dtype=float),
              'U_M': np.arange(groups, dtype=float) + 0.5}
    reductions = bidirectional.directional_reductions(errors, bidirectional.PRIMARY_DESIGNS)
    assert reductions.shape == (groups, 2)
    assert np.array_equal(reductions[:, 0], errors['U_C'] - errors['U_C_M'])
    assert np.array_equal(reductions[:, 1], errors['M'] - errors['U_C_M'])
    # The difference of the two directions is the difference of the two reduced
    # arms, because both are paired on the same joint arm and the same rows.
    assert np.allclose(reductions[:, 0] - reductions[:, 1], errors['U_C'] - errors['M'],
                       rtol=0, atol=1e-15)
    bands = bidirectional.paired_bands(reductions, draws=1000)
    assert np.isclose(bands['point'][0] - bands['point'][1],
                      float(np.mean(errors['U_C'] - errors['M'])), rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match='unknown direction'):
        bidirectional.directional_reductions(errors, {'x': ('M', 'U_C_M')})


def test_stacked_reductions_keep_the_algebra_on_arms_and_bootstrap_draws():
    names = ('U', 'M', 'U_M', 'U_C', 'U_C_M')
    rng = np.random.default_rng(3)
    table = rng.standard_normal((2, len(names), 6))
    reductions = bidirectional.stacked_reductions(table, bidirectional.PRIMARY_DESIGNS, names)
    assert reductions.shape == (2, 2, 6)
    assert np.array_equal(reductions[:, 0], table[:, 3] - table[:, 4])
    assert np.array_equal(reductions[:, 1], table[:, 1] - table[:, 4])
    assert np.allclose(reductions[:, 0] - reductions[:, 1], table[:, 3] - table[:, 1],
                       rtol=0, atol=1e-15)
    # A draw stack reduces through the same definition, so a bootstrap family and
    # the point family cannot disagree about which arm is the joint one.
    draws = bidirectional.stacked_reductions(np.stack([table, table + 1.0], axis=0),
                                            bidirectional.PRIMARY_DESIGNS, names)
    assert draws.shape == (2, 2, 2, 6)
    assert np.allclose(draws[0], reductions, rtol=0, atol=0)
    assert np.allclose(draws[1], reductions, rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match='design axis'):
        bidirectional.stacked_reductions(table, bidirectional.PRIMARY_DESIGNS, names[:3])


def test_reverse_direction_is_the_pure_likelihood_readout_not_the_conditioned_arm():
    assert bidirectional.PRIMARY_DESIGNS['controls_given_model'] == ('M', 'U_C_M')
    assert bidirectional.SENSITIVITY_DESIGNS['controls_given_model_U_M_conditioned'] == ('U_M', 'U_C_M')
    errors = {'U_C': np.array([1., 2.]), 'M': np.array([3., 3.]), 'U_C_M': np.array([1., 1.]),
              'U_M': np.array([1.5, 1.5])}
    primary = bidirectional.directional_reductions(
        errors, {'controls_given_model': bidirectional.PRIMARY_DESIGNS['controls_given_model']})
    sensitivity = bidirectional.directional_reductions(errors, bidirectional.SENSITIVITY_DESIGNS)
    assert np.array_equal(primary[:, 0], errors['M'] - errors['U_C_M'])
    assert np.array_equal(sensitivity[:, 0], errors['U_M'] - errors['U_C_M'])
    assert not np.allclose(primary, sensitivity)
    # Both reused designs are archived arms, so no new fit is introduced here.
    assert set(bidirectional.PRIMARY_DESIGNS) == {'model_given_controls', 'controls_given_model'}


def test_verdict_and_counts_use_the_simultaneous_interval_alone():
    interval = [[1e-9, 0.1], [-0.2, -1e-9], [-0.1, 0.1], [0.0, 0.2]]
    verdicts = bidirectional.verdict(interval)
    assert verdicts == ['positive', 'negative', 'unresolved', 'unresolved']
    assert bidirectional.counts(verdicts) == {'positive': 1, 'negative': 1, 'unresolved': 2}
    with pytest.raises(ValueError):
        bidirectional.verdict([[0.1, 0.2, 0.3]])


def pilot_record(**overrides):
    seeds = {}
    for seed in SEEDS:
        folds = [{'fold': fold,
                  'alpha': {'U_C': .1, 'U_C_M': 1.},
                  'alpha_expected': {'U_C': .1, 'U_C_M': 1.}} for fold in range(5)]
        seeds[str(seed)] = {'per_group_mse_max_absolute_difference': {'U_C': 1e-9, 'U_C_M': 2e-8},
                            'folds': folds}
    record = {'schema': bidirectional.PILOT_SCHEMA, 'arm': 'arm', 'status': 'complete',
              'pilot': True, 'tolerance': bidirectional.PILOT_TOLERANCE,
              'contract_sha256': 'c' * 64, 'script_sha256': 'a' * 64,
              'old_source_sha256': 'b' * 64, 'original_manifest_sha256': 'd' * 64,
              'reused_designs': ['U_C', 'U_C_M'], 'seeds': seeds,
              'max_abs_group_mse': 2e-8}
    record.update(overrides)
    return record


def require(record, **overrides):
    arguments = {'contract_sha256': 'c' * 64, 'manifest_sha256': 'd' * 64,
                 'split_seeds': SEEDS, 'outer_folds': 5, 'designs': ('U_C', 'U_C_M')}
    arguments.update(overrides)
    return bidirectional.require_pilot(record, **arguments)


def test_record_json_round_trip_keeps_the_pilot_flag_boolean():
    record = pilot_record()
    rendered = json.dumps(record, allow_nan=False, default=bidirectional.json_default)
    restored = json.loads(rendered)
    # A converter that tests `int` before `bool` writes 1 and makes every reader
    # that compares against True reject a genuine pilot.
    assert restored['pilot'] is True
    assert '"pilot": true' in rendered
    assert bidirectional.json_default(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert bidirectional.json_default(np.bool_(True)) is True
    assert bidirectional.json_default(np.float64(2.5)) == 2.5
    with pytest.raises(TypeError, match='not JSON serializable'):
        bidirectional.json_default(object())


def test_pilot_gate_accepts_a_complete_reproduction():
    outcome = require(pilot_record())
    assert outcome['seeds'] == 3 and outcome['outer_folds'] == 5
    assert outcome['worst_per_group_mse_difference'] == 2e-8


def test_pilot_gate_refuses_an_ordinary_fit_record():
    with pytest.raises(ValueError, match='not a pilot'):
        require(pilot_record(pilot=False))
    ordinary = {'arm': 'arm', 'status': 'complete', 'pilot': False,
                'seeds': {str(seed): {'per_group_mse': {}, 'folds': []} for seed in SEEDS}}
    with pytest.raises(ValueError, match='not a pilot'):
        require(ordinary)


def test_pilot_gate_refuses_an_incomplete_seed_set():
    record = pilot_record()
    del record['seeds'][str(SEEDS[2])]
    with pytest.raises(ValueError, match='not all of'):
        require(record)
    record = pilot_record()
    del record['seeds'][str(SEEDS[0])]['folds'][4]
    with pytest.raises(ValueError, match='outer folds'):
        require(record)


def test_pilot_gate_refuses_a_missed_alpha_or_error_reproduction():
    record = pilot_record()
    record['seeds'][str(SEEDS[1])]['folds'][2]['alpha']['U_C_M'] = 10.
    with pytest.raises(ValueError, match='archived U_C_M alpha'):
        require(record)
    record = pilot_record()
    record['seeds'][str(SEEDS[0])]['per_group_mse_max_absolute_difference']['U_C'] = 1.5e-7
    with pytest.raises(ValueError, match='above'):
        require(record)


def test_pilot_schema_is_a_stable_id_and_the_written_label_stays_readable():
    # The label is an identity, not a version counter: the format writes one stable
    # name, and the single label already written stays readable beside it.
    assert bidirectional.PILOT_SCHEMA == 'r3_bidirectional_pilot'
    assert bidirectional.PILOT_SCHEMAS == ('r3_bidirectional_pilot', 'r3_bidirectional_pilot_v2')
    assert require(pilot_record(schema='r3_bidirectional_pilot_v2'))['seeds'] == 3
    # Nothing is stripped back to a version-free stem, so a label outside the two
    # known ones is refused rather than guessed at.
    for unknown in ('r3_bidirectional_pilot_v1', 'r3_bidirectional_pilot_v3'):
        with pytest.raises(ValueError, match='schema'):
            require(pilot_record(schema=unknown))


def test_pilot_gate_never_loosens_the_tolerance():
    record = pilot_record(tolerance=1e-5)
    with pytest.raises(ValueError, match='tolerance differs'):
        require(record)
    with pytest.raises(ValueError, match='never loosened'):
        require(pilot_record(), tolerance=1e-5)


def test_pilot_gate_binds_the_contract_and_the_original_manifest():
    with pytest.raises(ValueError, match='current contract'):
        require(pilot_record(), contract_sha256='e' * 64)
    with pytest.raises(ValueError, match='original extraction manifest'):
        require(pilot_record(), manifest_sha256='e' * 64)
    with pytest.raises(ValueError, match='schema'):
        require(pilot_record(schema='other'))
    with pytest.raises(ValueError, match='not complete'):
        require(pilot_record(status='running'))
    with pytest.raises(ValueError, match='digest'):
        require(pilot_record(script_sha256='short'))
    with pytest.raises(ValueError, match='designs'):
        require(pilot_record(reused_designs=['U_C']))
