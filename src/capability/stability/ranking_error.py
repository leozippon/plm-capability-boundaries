"""Ranking versus quantitative error on the single-mutant stability endpoint.

The stability gate answers one question per arm: does a frozen checkpoint's
likelihood difference ``M_mut - M_WT`` carry information about the measured
stability change ``ddG`` beyond controls that predict it competently on their
own.  It answers it on two metrics that are routinely read as if they were the
same answer, and they are not:

*Error* is a quantitative fit.  The likelihood difference enters a ridge
regression whose target is ``ddG`` in kcal/mol, the penalty is tuned on inner
held-out families and the increment is the reduction in group-equal held-out
mean squared error.  The nat-to-kcal/mol conversion is therefore learned, once,
on the training families and transferred to families the fit never saw.  That is
a real and demanding question, but it is *two* questions at once: does the
likelihood order the variants of a family correctly, and does one global slope
convert nats into kcal/mol for every family alike.  A likelihood increment is not
in kcal/mol and nothing forces its scale to be shared across folds of different
length, fold class and dynamic range, so an arm can fail this while ordering
every family correctly.

*Ranking* is the within-family rank correlation increment.  It is invariant to
any per-family monotone transform, so it isolates the ordering question and
discards the scale question entirely.

Comparing the two as they stand therefore compares "does the model carry
quantitative information" against "is a nat a kcal/mol", and the second term is
not a property of the model.  :func:`family_calibrated` supplies the missing
control: before the error is taken, each held-out prediction is recalibrated
inside its own family by an affine map fitted on the *other* sites of that
family, so baseline and augmented predictions are granted exactly the same
per-family scale and offset.  The resulting increment asks the quantitative
question with the scale question answered, and it is the honest counterpart of
the within-family rank increment because both now read only within-family
information.

The three readings are kept separate and reported together:

``transfer_error``
    The frozen convention.  One global slope, no per-family calibration.  This
    is the number the panel of 2026-09-24 published and it is recomputed here so
    the recovery can be checked against it.
``within_family_calibrated_error``
    The same predictions, the same control set and the same group-equal error,
    after the declared per-family affine recalibration.  A quantitative reading
    that a family-specific scale cannot explain away.
``within_family_ranking``
    The frozen within-background Spearman increment, taken through
    :func:`~..interactions.pairwise_epistasis.group_spearman` exactly as the
    gate's own ``spearman_increment`` does, so the recomputation and the frozen
    record are the same statistic.

Nothing here resamples on its own.  Per-arm intervals come from the gate's group
bootstrap and the across-arm family comes from the project's single
maximum-statistic helper; the family is the resampling unit throughout, because
a wild-type measurement is shared by every variant of its background.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..interactions.pairwise_epistasis import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, group_errors, group_spearman, interval)
from ..readouts.readout_analysis import row_weights

#: Within-family folds the calibrated error control cross-fits over.  Four
#: mirrors the gate's own inner partition; every cohort background carries at
#: least 34 mutated sites, so the smallest fold still holds eight of them.
CALIBRATION_FOLDS = 4

#: Seed of the site-to-fold assignment.  The assignment is a stable hash of the
#: seed, the family label and the site label, so it reads no measurement, no
#: confidence width and no model score and cannot be tuned to an outcome.
CALIBRATION_SEED = 20261008

#: Smallest weighted standard deviation of a prediction, in the prediction's own
#: units, at which an affine recalibration has a defined slope.  Below it the
#: calibration collapses to the weighted mean of the fitting rows, which is the
#: correct limit of the affine family rather than a fallback, and every such
#: collapse is counted in the returned record.
DEGENERATE_PREDICTION_SCALE = 1e-12

#: Draws every across-arm family is bootstrapped with.  Ten thousand is the count
#: the project's residual and stability replay panels already publish under, so a
#: recomputed band is comparable with the frozen one.
PANEL_DRAWS = 10000


def site_folds(group, sites, *, folds: int = CALIBRATION_FOLDS,
               seed: int = CALIBRATION_SEED) -> dict:
    """Balanced, label-blind partition of one family's mutated sites.

    The sites are ordered by a stable hash of the seed, the family and the site
    label -- the same construction the cohort's own variant draw uses -- and then
    dealt round robin, so each fold receives within one site of an equal share
    whatever the family's size.  A modulo of the hash would leave fold sizes
    multinomial and admit an empty fold; dealing an ordering cannot.
    """

    ordered = sorted(set(np.asarray(sites).tolist()),
                     key=lambda site: hashlib.sha256(
                         f'{seed}:{group}:{site}'.encode()).digest())
    return {site: index % folds for index, site in enumerate(ordered)}


def _weighted_affine(x: np.ndarray, y: np.ndarray,
                     weights: np.ndarray) -> tuple[float, float, bool]:
    """Site-weighted least squares ``a + b x``, with a flat slope declared."""

    share = weights / weights.sum()
    x_centre = float((share * x).sum())
    y_centre = float((share * y).sum())
    deviation = x - x_centre
    variance = float((share * deviation * deviation).sum())
    if variance <= DEGENERATE_PREDICTION_SCALE ** 2:
        return y_centre, 0.0, True
    slope = float((share * deviation * (y - y_centre)).sum()) / variance
    return y_centre - slope * x_centre, slope, False


def family_calibrated(prediction, target, groups, sites, *,
                      folds: int = CALIBRATION_FOLDS,
                      seed: int = CALIBRATION_SEED) -> tuple[np.ndarray, dict]:
    """Within-family, site-cross-fitted affine recalibration of a prediction.

    Inside each family the mutated sites are partitioned by :func:`site_folds`.
    For every part, an affine map is fitted by site-weighted least squares on the
    family's *other* parts and applied to this one, so no site's own measurement
    enters its own calibration.  The weighting is the gate's own -- sites equal
    inside a family -- so the calibration optimises the same loss the error is
    later read on.

    This is deliberately *not* a transfer estimate.  It uses the held-out
    family's own labels, at other sites, to fix that family's scale and offset.
    The question it answers is "does this quantity carry quantitative information
    about ddG inside a family whose scale is known", which is the error
    counterpart of the within-family rank correlation and is reported under that
    name.  Baseline and augmented predictions receive the identical treatment, so
    the paired increment remains a statement about the added column alone.
    """

    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    groups = np.asarray(groups)
    sites = np.asarray(sites)
    if not (len(prediction) == len(target) == len(groups) == len(sites)):
        raise ValueError('calibration needs aligned prediction, target, group and site arrays')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('calibration refuses a nonfinite prediction or target')
    if folds < 2:
        raise ValueError('an affine calibration needs at least two within-family folds')
    calibrated = np.full(len(prediction), np.nan)
    slopes: list[float] = []
    degenerate = 0
    for group in sorted(set(groups.tolist())):
        rows = np.flatnonzero(groups == group)
        assignment = site_folds(group, sites[rows], folds=folds, seed=seed)
        if len(assignment) < folds:
            raise ValueError(
                f'{group}: {len(assignment)} mutated sites cannot support {folds} '
                'within-family calibration folds; the support is not evaluable under '
                'this control and no weaker partition is substituted')
        fold_of = np.asarray([assignment[site] for site in sites[rows]])
        for fold in range(folds):
            fit = rows[fold_of != fold]
            evaluate = rows[fold_of == fold]
            if not len(fit) or not len(evaluate):
                raise ValueError(f'{group}: within-family calibration fold {fold} '
                                 'has an empty side')
            weights = row_weights(sites[fit], np.zeros(len(fit), dtype=int))
            intercept, slope, flat = _weighted_affine(
                prediction[fit], target[fit], weights)
            calibrated[evaluate] = intercept + slope * prediction[evaluate]
            slopes.append(slope)
            degenerate += int(flat)
    if not np.isfinite(calibrated).all():
        raise ValueError('incomplete within-family calibration')
    return calibrated, {
        'folds': int(folds),
        'seed': int(seed),
        'assignment': 'stable sha256 of seed, family and site label, dealt round robin',
        'weighting': 'sites equal inside the fitting part, variants equal inside a site',
        'calibrations': len(slopes),
        'degenerate_slope_calibrations': int(degenerate),
        'slope_range': [float(min(slopes)), float(max(slopes))],
        'semantics': ('affine recalibration fitted on the other sites of the same family; '
                      'a within-family reading, not a transfer estimate'),
    }


def jsonable(values) -> list:
    """Finite values as floats, undefined ones as null.

    The project's writer rejects ``NaN`` outright, for the reason the writer
    states: a non-finite value that reaches an artefact is how an undefined
    family becomes a plotted zero.  An undefined per-family statistic is a fact
    about that family and is stored as ``null`` so a reader cannot miss it.
    """

    return [None if value is None or not np.isfinite(value) else float(value)
            for value in values]


def vector(values) -> np.ndarray:
    """A stored contrast column back as a float array, nulls reopened as NaN."""

    return np.asarray([np.nan if value is None else float(value) for value in values],
                      dtype=float)


def mse_contrast(target, baseline, augmented, groups, sites) -> tuple[list, np.ndarray]:
    """Per-family group-equal squared-error reduction, in squared kcal/mol."""

    labels, base = group_errors(np.asarray(target, dtype=float), np.asarray(baseline, dtype=float),
                                np.asarray(groups), np.asarray(sites))
    _, better = group_errors(np.asarray(target, dtype=float), np.asarray(augmented, dtype=float),
                             np.asarray(groups), np.asarray(sites))
    return [str(label) for label in labels], base - better


def calibrated_mse_contrast(target, baseline, augmented, groups, sites, *,
                            folds: int = CALIBRATION_FOLDS,
                            seed: int = CALIBRATION_SEED) -> tuple[list, np.ndarray, dict]:
    """The same reduction after the declared per-family affine recalibration."""

    base_calibrated, base_record = family_calibrated(
        baseline, target, groups, sites, folds=folds, seed=seed)
    better_calibrated, better_record = family_calibrated(
        augmented, target, groups, sites, folds=folds, seed=seed)
    labels, values = mse_contrast(target, base_calibrated, better_calibrated, groups, sites)
    return labels, values, {'baseline': base_record, 'augmented': better_record}


def rank_contrast(target, baseline, augmented, groups) -> tuple[list, np.ndarray, list]:
    """Per-family within-background Spearman increment, undefined families named.

    The correlation is the gate's own :func:`group_spearman`, which is what the
    frozen panel's ``spearman_increment`` reads, so this recomputation and the
    frozen record are the same statistic rather than two defensible ones.  A
    family whose prediction has no rank ordering under either design yields no
    value; it is returned as ``nan`` and named, never deleted and never imputed
    as zero, because deleting it would make the family size depend on the arm.
    """

    target = np.asarray(target, dtype=float)
    groups = np.asarray(groups)
    labels, base = group_spearman(target, np.asarray(baseline, dtype=float), groups)
    _, better = group_spearman(target, np.asarray(augmented, dtype=float), groups)
    undefined = [str(label) for label, low, high in zip(labels, base, better)
                 if low is None or high is None]
    values = np.asarray([np.nan if low is None or high is None else float(high) - float(low)
                         for low, high in zip(base, better)])
    return [str(label) for label in labels], values, undefined


def arm_contrasts(panel: dict, predictions: dict, *, channels: dict,
                  error_control: str = 'S', rank_control: str = 'S2',
                  folds: int = CALIBRATION_FOLDS,
                  seed: int = CALIBRATION_SEED) -> dict:
    """Every declared per-family contrast vector for one arm, on one split seed.

    ``predictions`` holds the held-out prediction vector of each design.
    ``channels`` maps a measurement-channel name to its target vector; the
    combined channel is the gate's published endpoint and the two protease
    channels are the robustness reading.  Error contrasts are read over the
    squared-error-qualified control set and rank contrasts over the
    rank-qualified one, which is the rule the frozen gate licenses; the rank
    contrast is additionally read over the error control set so that one pair of
    numbers differs in nothing but the metric.
    """

    groups, sites = panel['group'], panel['site']
    out: dict[str, dict] = {}
    for channel, target in channels.items():
        labels, transfer = mse_contrast(
            target, predictions[error_control], predictions[f'{error_control}_M'], groups, sites)
        _, calibrated, calibration = calibrated_mse_contrast(
            target, predictions[error_control], predictions[f'{error_control}_M'],
            groups, sites, folds=folds, seed=seed)
        rank_labels, licensed_rank, licensed_undefined = rank_contrast(
            target, predictions[rank_control], predictions[f'{rank_control}_M'], groups)
        _, paired_rank, paired_undefined = rank_contrast(
            target, predictions[error_control], predictions[f'{error_control}_M'], groups)
        if rank_labels != labels:
            raise ValueError('family label order differs between the two metrics')
        _, baseline_mse = group_errors(np.asarray(target, dtype=float),
                                       np.asarray(predictions[error_control], dtype=float),
                                       np.asarray(groups), np.asarray(sites))
        _, baseline_rank = group_spearman(np.asarray(target, dtype=float),
                                          np.asarray(predictions[rank_control], dtype=float),
                                          np.asarray(groups))
        out[channel] = {
            'groups': labels,
            'transfer_error': jsonable(transfer),
            'within_family_calibrated_error': jsonable(calibrated),
            'within_family_ranking': jsonable(licensed_rank),
            'within_family_ranking_on_error_controls': jsonable(paired_rank),
            'baseline_transfer_error': jsonable(baseline_mse),
            'baseline_within_family_ranking': [None if value is None else float(value)
                                               for value in baseline_rank],
            'undefined_ranking_groups': {rank_control: licensed_undefined,
                                         error_control: paired_undefined},
            'calibration': calibration,
        }
    return out


#: Per-family contrast names, with the metric each is read on and the control set
#: it is licensed over.  ``within_family_ranking_on_error_controls`` is the
#: strictly paired comparison: identical predictions, identical control set, only
#: the metric differs.
CONTRASTS: dict[str, dict[str, str]] = {
    'transfer_error': {
        'control': 'S',
        'metric': 'group-equal mean squared error reduction, squared kcal/mol',
        'role': 'frozen convention; one global nat-to-kcal/mol slope transferred across families',
    },
    'within_family_calibrated_error': {
        'control': 'S',
        'metric': 'group-equal mean squared error reduction after per-family affine '
                  'recalibration, squared kcal/mol',
        'role': 'qualified quantitative reading; the per-family scale is granted to both sides',
    },
    'within_family_ranking': {
        'control': 'S2',
        'metric': 'within-background Spearman increment, dimensionless',
        'role': 'licensed ranking reading over the rank-qualified control set',
    },
    'within_family_ranking_on_error_controls': {
        'control': 'S',
        'metric': 'within-background Spearman increment, dimensionless',
        'role': 'strictly paired metric comparison against within_family_calibrated_error',
    },
}


def simultaneous_family(matrix, columns, *, draws: int = PANEL_DRAWS) -> dict:
    """The project's maximum-statistic band over one declared contrast family.

    The helper is imported rather than restated: the frozen stability replay
    panel, the pairwise residual panel and the position-term follow-up all
    publish under this exact construction, so a recomputed band is comparable
    with the frozen one.  Rows are families -- the resampling unit, because every
    variant of a background shares that background's single wild-type
    measurement -- and columns are the prespecified contrasts of the family.
    """

    from scripts.capability.interactions.run_residual_panel import simultaneous_bands

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(columns):
        raise ValueError('a contrast family needs one column per declared contrast')
    if not np.isfinite(matrix).all():
        raise ValueError('a contrast family with a nonfinite cell is not evaluable; '
                         'the undefined families are reported rather than dropped')
    bands = simultaneous_bands(matrix, draws=draws)
    verdicts = {}
    for index, column in enumerate(columns):
        low, high = bands['interval'][index]
        verdicts[column] = ('above_zero' if low > 0
                            else 'below_zero' if high < 0 else 'unresolved')
    return {'columns': list(columns), 'bands': bands, 'verdicts': verdicts,
            'resolved_above_zero': [c for c, v in verdicts.items() if v == 'above_zero'],
            'resolved_below_zero': [c for c, v in verdicts.items() if v == 'below_zero']}


def marginal_interval(values, *, draws: int = BOOTSTRAP_DRAWS,
                      seed: int = BOOTSTRAP_SEED) -> dict:
    """The gate's own per-arm family bootstrap, labelled as marginal.

    Published beside the simultaneous band so a reader can see the cost of the
    33-arm family, and never instead of it.  The frozen panel's own multiplicity
    note records that reading these as if they were simultaneous is how the
    stability positives were over-read once already.
    """

    record = interval(values, draws=draws, seed=seed)
    record['multiplicity'] = 'marginal for one arm and one contrast; not adjusted for the family'
    return record


def ranking_versus_error(families: dict, arms) -> dict:
    """How widespread each reading is, and where the two disagree, arm by arm.

    The comparison is deliberately a count of resolved arms and a cross
    tabulation rather than a pooled difference: the two metrics carry different
    units, so a single pooled contrast between them would have no interpretation.
    Each metric keeps its own simultaneous family and the question "is ranking
    more widespread than quantitative prediction" is answered by how many arms
    each family resolves above zero and by which arms resolve on one reading
    only.
    """

    arms = list(arms)
    table = {}
    for arm in arms:
        table[arm] = {name: families[name]['verdicts'][arm] for name in families}
    counts = {name: {'above_zero': len(families[name]['resolved_above_zero']),
                     'below_zero': len(families[name]['resolved_below_zero']),
                     'unresolved': len(arms) - len(families[name]['resolved_above_zero'])
                     - len(families[name]['resolved_below_zero'])}
              for name in families}
    paired = ('within_family_calibrated_error', 'within_family_ranking_on_error_controls')
    cross = {}
    if all(name in families for name in paired):
        error, rank = paired
        for label, predicate in (
                ('ranking_only', lambda row: row[rank] == 'above_zero'
                 and row[error] != 'above_zero'),
                ('error_only', lambda row: row[error] == 'above_zero'
                 and row[rank] != 'above_zero'),
                ('both', lambda row: row[error] == 'above_zero' and row[rank] == 'above_zero'),
                ('neither', lambda row: row[error] != 'above_zero'
                 and row[rank] != 'above_zero')):
            cross[label] = [arm for arm in arms if predicate(table[arm])]
    return {
        'arms': arms,
        'per_arm_verdicts': table,
        'resolved_counts': counts,
        'paired_metric_comparison': {
            'contrasts': list(paired),
            'basis': 'identical held-out predictions and identical control set; only the '
                     'metric differs, so a difference in how many arms resolve is a '
                     'difference between ordering and quantitative information',
            'cross_tabulation': cross,
        },
        'multiplicity': 'each contrast is its own 33-column maximum-statistic family over '
                        'the same 101 resampled families; the families are not pooled across '
                        'metrics, because the metrics carry different units',
    }


__all__ = [
    'CALIBRATION_FOLDS', 'CALIBRATION_SEED', 'CONTRASTS', 'DEGENERATE_PREDICTION_SCALE',
    'PANEL_DRAWS', 'arm_contrasts', 'calibrated_mse_contrast', 'family_calibrated',
    'jsonable', 'marginal_interval', 'mse_contrast', 'rank_contrast', 'ranking_versus_error',
    'simultaneous_family', 'site_folds', 'vector',
]
