"""Declared uncertainty and reproduction rules shared by the bidirectional ablation.

The ablation compares one joint readout with two reduced readouts on identical
held-out groups: the joint arm against the traditional controls (``model given
controls``) and the joint arm against the model readout alone (``controls given
model``). Every direction is a paired per-group reduction, positive when the joint
arm lowers the error, and the two directions are read as one family of 66
contrasts across 2 directions and 33 arms.

The run contract states that family as the joint maximum absolute *centered
bootstrap deviation* in the endpoint's native units: a draw counts by how far a
resampled column mean moves from its point estimate, and that largest movement,
not a studentised multiple, sets the band half-width. That is a different
statement from the project's fixed-standard-error ``simultaneous_bands`` helper,
which divides each deviation by the column's group standard error. On a family
whose columns carry unequal spread the two give different widths, so the
studentised family is retained as an explicitly labelled sensitivity rather than
substituted for the declared primary.

Everything here is a definition, not an analysis: the bootstrap statistic, the
verdict rule that turns a simultaneous interval into positive, negative or
unresolved, the two-direction pairing, and the pilot gate that a reused arm must
pass before any summary may be written.

A schema label is an identity, not a version counter: this format writes one
stable name and a reader accepts that name plus the labels already written for the
same declaration. Nothing is stripped back to a version-free stem, so an unknown
label is refused rather than guessed at.
"""
from __future__ import annotations

import re

import numpy as np

#: One frozen bootstrap family per endpoint: 2 directions x 33 arms.
FAMILY_SIZE = 66
DRAWS = 10000
SEED = 20261106
CONFIDENCE = .95
DRAW_CHUNK = 100

#: The declared primary family: largest centered deviation in native units.
NATIVE_METHOD = ('paired group bootstrap maximum absolute centered deviation in native units; '
                 'no SE division; prediction-conditional')
#: The retained studentised family, as its own home states it.
STANDARDIZED_METHOD = 'paired group bootstrap maximum absolute centered statistic / fixed group SE'

#: Direction names, when a reduction is positive because the joint arm lowers error.
DIRECTIONS = ('model_given_controls', 'controls_given_model')

#: Primary contrasts as (reduced arm, joint arm). ``G`` is part of the controls, and
#: ``T`` is a model-specific tokenizer nuisance carried by the full control, so
#: ``U_C`` is not a pure traditional-statistics or model-invariant control. The
#: reverse direction asks what the controls add to the pure scalar likelihood
#: readout, not what they add to a control-conditioned likelihood readout.
PRIMARY_DESIGNS = {'model_given_controls': ('U_C', 'U_C_M'),
                   'controls_given_model': ('M', 'U_C_M')}

#: The earlier reverse contrast, conditioned on the nuisance-only arm. It answers a
#: different question and is never reported as the primary reverse direction.
SENSITIVITY_DESIGNS = {'controls_given_model_U_M_conditioned': ('U_M', 'U_C_M')}

#: Largest admissible pilot disagreement, in the endpoint's native squared-error
#: units. A reproduced arm matches the archived per-group errors to this tolerance
#: or it is not reused at all; no weaker tolerance is accepted downstream.
PILOT_TOLERANCE = 1e-7
#: The stable label of this pilot declaration.
PILOT_SCHEMA = 'r3_bidirectional_pilot'
#: Every pilot label this reader knows: the stable id and the one already written.
#: Any other label is refused rather than stripped back to a version-free stem.
PILOT_SCHEMAS = (PILOT_SCHEMA, 'r3_bidirectional_pilot_v2')
_HEX64 = re.compile(r'^[0-9a-f]{64}$')


def json_default(value):
    """JSON hook for the array and NumPy types an ablation record carries.

    Python booleans and tuples are left to :mod:`json`, which renders them as
    ``true`` and as a list. Only NumPy types need a hook, and a NumPy boolean must
    reach that renderer as a boolean: ``bool`` is a subclass of ``int``, so a
    converter that tests ``int`` first silently writes ``1`` where the record
    declares a flag, and every reader that compares against ``True`` then fails.
    """

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f'{type(value).__name__} is not JSON serializable')


def resample_means(values, *, draws=DRAWS, seed=SEED):
    """Row-resampled column means, in chunks of :data:`DRAW_CHUNK` draws.

    The draw order is the one the project's fixed-standard-error helper uses, so
    the declared native family and the retained studentised sensitivity read the
    same resamples and differ only in the statistic they form from them.
    """

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError('a bootstrap family needs a groups-by-columns matrix with at least two groups')
    rng = np.random.default_rng(seed)
    blocks = []
    for start in range(0, draws, DRAW_CHUNK):
        stop = min(start + DRAW_CHUNK, draws)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        blocks.append(values[indices].mean(axis=1))
    return np.concatenate(blocks)


def paired_bands(values, *, draws=DRAWS, seed=SEED, confidence=CONFIDENCE) -> dict:
    """Simultaneous bands from the largest centered bootstrap deviation.

    ``values`` is groups by contrasts, one column per paired reduction. Each
    column's point estimate is its group-equal mean, and the critical value is the
    ``confidence`` quantile of the largest absolute centered deviation over the
    whole family in one draw. Every column then carries ``point +- critical`` in
    the same native units as the contrast, which is what the run contract
    declares; no standard error is estimated and none is divided by.
    """

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2 or values.shape[1] < 1:
        raise ValueError('a simultaneous family is a groups-by-columns matrix with at least two groups')
    if not np.isfinite(values).all():
        raise ValueError('a simultaneous family is finite in every group and column')
    if not 0 < confidence < 1:
        raise ValueError('a simultaneous confidence level lies strictly between zero and one')
    points = values.mean(axis=0)
    deviation = resample_means(values, draws=draws, seed=seed) - points
    critical = float(np.quantile(np.max(np.abs(deviation), axis=1), confidence))
    return {'method': NATIVE_METHOD, 'draws': draws, 'seed': seed,
            'groups': int(values.shape[0]), 'family_size': int(values.shape[1]),
            'confidence': confidence, 'critical_value': critical,
            'point': points.tolist(),
            'interval': np.column_stack([points - critical, points + critical]).tolist(),
            'conditional_on_fitted_predictions': True}


def _paired(values, index, designs):
    """Reduced-minus-joint columns from a table whose design axis is ``-2``."""

    columns = []
    for direction, pair in designs.items():
        if direction not in DIRECTIONS and direction not in SENSITIVITY_DESIGNS:
            raise ValueError(f'unknown direction {direction!r}')
        reduced, joint = pair
        columns.append(np.take(values, index[reduced], axis=-2)
                       - np.take(values, index[joint], axis=-2))
    return np.stack(columns, axis=-2)


def stacked_reductions(values, designs, names) -> np.ndarray:
    """Paired reductions of a (..., designs, groups) table, one column per direction.

    Both directions of one family are formed here so the pairing cannot drift:
    on the same rows the two columns share the same joint error, hence their
    difference is exactly the difference between the two reduced arms. The table
    may carry any leading axes, which is what lets a bootstrap draw stack of
    ``(draws, arms, designs, groups)`` be reduced without a second implementation.
    """

    values = np.asarray(values, dtype=float)
    names = list(names)
    if values.ndim < 2 or values.shape[-2] != len(names):
        raise ValueError('a stacked reduction needs a design axis matching the named designs')
    return _paired(values, {name: position for position, name in enumerate(names)}, designs)


def directional_reductions(errors: dict, designs: dict) -> np.ndarray:
    """Paired reductions of named per-group error vectors, as groups by direction."""

    names = list(errors)
    arrays = [np.asarray(errors[name], dtype=float) for name in names]
    if any(array.ndim != 1 for array in arrays) or len({len(array) for array in arrays}) != 1:
        raise ValueError('paired reductions need aligned one-dimensional error vectors')
    stacked = stacked_reductions(np.stack(arrays, axis=-2), designs, names)
    return np.moveaxis(stacked, -2, -1)


def verdict(interval) -> list[str]:
    """Per-column verdict from a simultaneous interval that contains zero or not."""

    interval = np.asarray(interval, dtype=float)
    if interval.ndim != 2 or interval.shape[1] != 2 or not np.isfinite(interval).all():
        raise ValueError('a verdict needs a finite lower/upper interval per column')
    return ['positive' if low > 0 else 'negative' if high < 0 else 'unresolved'
            for low, high in interval]


def counts(verdicts) -> dict[str, int]:
    """Verdict tally over a declared family."""

    values = list(verdicts)
    unknown = sorted(set(values) - {'positive', 'negative', 'unresolved'})
    if unknown:
        raise ValueError(f'unknown verdict values {unknown}')
    return {name: values.count(name) for name in ('positive', 'negative', 'unresolved')}


def require_pilot(pilot, *, contract_sha256: str, manifest_sha256: str, split_seeds,
                  outer_folds: int, designs, tolerance: float = PILOT_TOLERANCE) -> dict:
    """Refuse to reuse an arm unless a real pilot reproduced it on every seed.

    The gate is the whole point of the reuse: new fits are pooled with archived
    per-group errors only after a pilot reproduced the archived alpha choices and
    per-group errors on all three split seeds. An ordinary fit record is not a
    pilot and cannot be promoted to one, an incomplete seed set is refused, and
    no tolerance looser than :data:`PILOT_TOLERANCE` is accepted.
    """

    if not isinstance(pilot, dict):
        raise ValueError('a pilot is a record object')
    if pilot.get('pilot') is not True:
        raise ValueError('refusing a record that is not a pilot: the reuse gate needs an '
                         'independent pilot run, not a fit record or a re-read of one')
    if pilot.get('status') != 'complete':
        raise ValueError(f"pilot status is {pilot.get('status')!r}, not complete")
    if pilot.get('schema') not in PILOT_SCHEMAS:
        raise ValueError(f"pilot schema is {pilot.get('schema')!r}, not {PILOT_SCHEMAS}")
    if pilot.get('tolerance') != PILOT_TOLERANCE:
        raise ValueError('pilot tolerance differs from the declared reproduction tolerance')
    if tolerance > PILOT_TOLERANCE:
        raise ValueError('the pilot gate tolerance is never loosened beyond the declared value')
    if pilot.get('contract_sha256') != contract_sha256:
        raise ValueError('pilot was not run against the current contract')
    for key in ('script_sha256', 'old_source_sha256'):
        if not isinstance(pilot.get(key), str) or not _HEX64.match(pilot[key]):
            raise ValueError(f'pilot {key} is not a recorded digest')
    if pilot.get('original_manifest_sha256') != manifest_sha256:
        raise ValueError('pilot is not bound to the frozen original extraction manifest')
    if list(pilot.get('reused_designs', [])) != list(designs):
        raise ValueError('pilot did not reproduce the designs being reused')
    seeds = pilot.get('seeds')
    if not isinstance(seeds, dict):
        raise ValueError('pilot carries no per-seed records')
    expected = [str(seed) for seed in split_seeds]
    if sorted(seeds) != sorted(expected):
        raise ValueError(f'pilot seeds are {sorted(seeds)}, not all of {sorted(expected)}')
    worst = 0.
    for seed in expected:
        record = seeds[seed]
        folds = record.get('folds')
        if not isinstance(folds, list) or [fold.get('fold') for fold in folds] != list(range(outer_folds)):
            raise ValueError(f'pilot seed {seed} does not cover outer folds 0..{outer_folds - 1}')
        differences = record.get('per_group_mse_max_absolute_difference')
        if not isinstance(differences, dict) or sorted(differences) != sorted(designs):
            raise ValueError(f'pilot seed {seed} records no per-design reproduction differences')
        for fold in folds:
            for name in designs:
                alpha = fold.get('alpha', {}).get(name)
                reproduced = fold.get('alpha_expected', {}).get(name)
                if alpha is None or reproduced is None:
                    raise ValueError(f'pilot seed {seed} fold {fold.get("fold")} lacks {name} alpha')
                if alpha != reproduced:
                    raise ValueError(f'pilot seed {seed} did not reproduce the archived {name} alpha')
        for name in designs:
            difference = float(differences[name])
            if not np.isfinite(difference) or difference > tolerance:
                raise ValueError(f'pilot seed {seed} {name} per-group MSE differs by {difference:g}, '
                                 f'above {tolerance:g}')
            worst = max(worst, difference)
    return {'worst_per_group_mse_difference': worst, 'seeds': len(expected),
            'outer_folds': outer_folds, 'designs': list(designs)}
