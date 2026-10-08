"""Ranking versus quantitative error on the single-mutant stability endpoint.

The stability gate asks, per arm, whether a frozen checkpoint's likelihood
difference ``M_mut - M_WT`` carries information about the measured stability
change ``ddG`` beyond controls that predict it competently on their own. It
answers on two metrics that are routinely read as one answer, over two control
sets that give different answers, and the combination has been over-read in both
directions. This module fixes the grid: **two baselines times three readings**,
every cell computed the same way, every cell published.

Two things vary, and they vary for different reasons.

**The baseline varies in what it already knows.** The matched baseline ``S`` --
directed substitution identity, mutated-position geometry, local chemistry and
the nonlinear-additive global response -- carries no evolutionary statistic at
all. The profile-inclusive baseline ``S2`` adds amino-acid composition and the
mutation-local independent-site alignment profile. Only the second answers the
question a reviewer actually asks of a protein language model: does it add
information *beyond evolutionary statistics*. A gain over ``S`` can be nothing
more than a rediscovery of the alignment column, which is why the profile
baseline is primary here and the matched baseline is reported beside it rather
than instead of it.

This is a **nested increment**, not a comparison of correlations. Both sides of
every contrast are fitted ridge predictors on identical rows and identical nested
family partitions; the augmented design is the baseline design plus one column.
So the likelihood must improve a predictor *that already contains the profile*.
The weaker statement -- that the likelihood correlates with ``ddG`` about as well
as a profile lookup does -- would compare two separate correlations and is not
computed anywhere here, because two quantities can each correlate with a target
while carrying the same information.

**The metric varies in what it demands of the scale.** A likelihood difference is
in nats and ``ddG`` is in kcal/mol. The transfer error learns one global
nat-to-kcal/mol slope on the training families and applies it to families the fit
never saw, which asks two questions at once: does the model order a family's
variants, and is one slope right for every family. Nothing makes the second true,
so an arm can order every family correctly and still fail. The rank increment,
invariant to any per-family monotone transform, isolates ordering and discards
scale entirely. Comparing those two as they stand compares "does the model carry
quantitative information" against "is a nat a kcal/mol", and the second term is
not a property of the model.

:func:`family_calibrated` supplies the missing middle. Before the error is taken,
each held-out prediction is recalibrated inside its own family by an affine map
fitted on that family's *other* sites, so baseline and augmented receive exactly
the same per-family scale and offset. The resulting increment asks the
quantitative question with the scale question already answered, and it is the
honest counterpart of the rank increment because both then read only
within-family information. It also stabilises a reading that is otherwise
erratic: the profile block transfers rank well and level badly, so the
profile-inclusive baseline's uncalibrated error swings between split seeds, and
per-family recalibration removes precisely that swing.

The three readings are therefore:

``transfer_error``
    Group-equal held-out mean squared error reduction, squared kcal/mol, one
    global slope. Over the matched baseline this is the convention the
    2026-09-24 panel published, recomputed so the recovery can be checked.
``calibrated_error``
    The same reduction after the declared per-family affine recalibration. The
    qualified quantitative reading.
``ranking``
    Within-background Spearman increment, taken through the gate's own
    :func:`~..interactions.pairwise_epistasis.group_spearman` so that the
    recomputation and the published record are the same statistic.

Nothing here resamples on its own. Per-arm intervals come from the gate's group
bootstrap and the across-arm family from the project's one maximum-statistic
helper; the family group is the resampling unit throughout, because every variant
of a background shares that background's single wild-type measurement.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..interactions.pairwise_epistasis import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, group_errors, group_spearman, interval)
from ..readouts.readout_analysis import row_weights

#: Within-family folds the calibrated error control cross-fits over. Four
#: mirrors the gate's own inner partition; every cohort background carries at
#: least 34 mutated sites, so the smallest fold still holds eight of them.
CALIBRATION_FOLDS = 4

#: Seed of the site-to-fold assignment. The assignment is a stable hash of the
#: seed, the family label and the site label, so it reads no measurement, no
#: confidence width and no model score and cannot be tuned to an outcome.
CALIBRATION_SEED = 20261008

#: Smallest weighted standard deviation of a prediction, in the prediction's own
#: units, at which an affine recalibration has a defined slope. Below it the
#: calibration collapses to the weighted mean of the fitting rows, which is the
#: correct limit of the affine family rather than a fallback, and every such
#: collapse is counted in the returned record.
DEGENERATE_PREDICTION_SCALE = 1e-12

#: Draws every across-arm family is bootstrapped with. Ten thousand is the count
#: the project's residual and stability replay panels already publish under, so a
#: recomputed band is comparable with the frozen one.
PANEL_DRAWS = 10000


# --------------------------------------------------------------------------- #
# The contrast grid: two baselines, three readings, every cell published.
# --------------------------------------------------------------------------- #

#: The two control sets, and which one is primary. ``profile`` is primary for
#: every reading because it is the only one that asks whether a model adds
#: information beyond evolutionary statistics; ``matched`` is published beside it
#: because the difference between the two readings is itself the finding.
BASELINES: dict[str, dict[str, str]] = {
    'profile': {
        'design': 'S2',
        'role': 'primary',
        'blocks': 'ident + geom + comp + chem + prof2 + G',
        'carries': 'mutation-local independent-site alignment profile and composition',
        'question': 'does the likelihood add information beyond evolutionary statistics',
    },
    'matched': {
        'design': 'S',
        'role': 'companion',
        'blocks': 'ident + geom + chem + G',
        'carries': 'no evolutionary statistic',
        'question': 'does the likelihood add information beyond substitution identity, '
                    'position geometry and local chemistry',
    },
}

#: The three readings, and what each one demands of the likelihood's scale.
READINGS: dict[str, dict[str, str]] = {
    'ranking': {
        'metric': 'within-background Spearman increment, dimensionless',
        'scale': 'invariant to any per-family monotone transform; ordering only',
    },
    'calibrated_error': {
        'metric': 'group-equal mean squared error reduction after per-family affine '
                  'recalibration, squared kcal/mol',
        'scale': 'per-family scale and offset granted to both sides, fitted on the '
                 'family\'s other sites; quantitative information within a family',
    },
    'transfer_error': {
        'metric': 'group-equal mean squared error reduction, squared kcal/mol',
        'scale': 'one global nat-to-kcal/mol slope learned on training families and '
                 'transferred; quantitative information across families',
    },
}


def _contrasts() -> dict[str, dict[str, str]]:
    """The grid, generated so a cell cannot be added on one axis only."""

    grid = {}
    for baseline, about in BASELINES.items():
        for reading, metric in READINGS.items():
            grid[f'{baseline}_{reading}'] = {
                'baseline': baseline,
                'design': about['design'],
                'baseline_role': about['role'],
                'baseline_blocks': about['blocks'],
                'reading': reading,
                'increment': 'nested: the baseline design plus the likelihood column M',
                **metric,
            }
    return grid


#: Every published per-family contrast, keyed ``<baseline>_<reading>``.
CONTRASTS: dict[str, dict[str, str]] = _contrasts()

#: The one strictly paired comparison that answers "is ranking more widespread
#: than quantitative prediction": identical held-out predictions, identical
#: profile-inclusive baseline, only the metric differs.
PRIMARY_PAIR = ('profile_calibrated_error', 'profile_ranking')

#: The same pair over the matched baseline, published beside it because the gap
#: between the two pairs is what the alignment profile is worth.
COMPANION_PAIR = ('matched_calibrated_error', 'matched_ranking')

#: Frozen 2026-09-24 panel contrasts that each recomputed cell reproduces. The
#: calibrated readings are new controls and have no frozen counterpart.
FROZEN_EQUIVALENT: dict[str, str] = {
    'matched_transfer_error': 'primary_likelihood',
    'matched_ranking': 'primary_likelihood_spearman',
    'profile_transfer_error': 'secondary_likelihood',
    'profile_ranking': 'secondary_likelihood_spearman',
}


# --------------------------------------------------------------------------- #
# Serialisation of per-family vectors.
# --------------------------------------------------------------------------- #

def jsonable(values) -> list:
    """Finite values as floats, undefined ones as null.

    The project's writer rejects ``NaN`` outright, for the reason the writer
    states: a non-finite value that reaches an artefact is how an undefined
    family becomes a plotted zero. An undefined per-family statistic is a fact
    about that family and is stored as ``null`` so a reader cannot miss it.
    """

    return [None if value is None or not np.isfinite(value) else float(value)
            for value in values]


def vector(values) -> np.ndarray:
    """A stored contrast column back as a float array, nulls reopened as NaN."""

    return np.asarray([np.nan if value is None else float(value) for value in values],
                      dtype=float)


# --------------------------------------------------------------------------- #
# The per-family affine recalibration.
# --------------------------------------------------------------------------- #

def site_folds(group, sites, *, folds: int = CALIBRATION_FOLDS,
               seed: int = CALIBRATION_SEED) -> dict:
    """Balanced, label-blind partition of one family's mutated sites.

    The sites are ordered by a stable hash of the seed, the family and the site
    label -- the same construction the cohort's own variant draw uses -- and then
    dealt round robin, so each fold receives within one site of an equal share
    whatever the family's size. A modulo of the hash would leave fold sizes
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
    enters its own calibration. The weighting is the gate's own -- sites equal
    inside a family -- so the calibration optimises the same loss the error is
    later read on.

    This is deliberately *not* a transfer estimate. It uses the held-out
    family's own labels, at other sites, to fix that family's scale and offset.
    The question it answers is "does this quantity carry quantitative information
    about ddG inside a family whose scale is known", which is the error
    counterpart of the within-family rank correlation and is reported under that
    name. Baseline and augmented predictions receive the identical treatment, so
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


# --------------------------------------------------------------------------- #
# The three readings, as per-family vectors.
# --------------------------------------------------------------------------- #

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
    _, base_level = group_errors(np.asarray(target, dtype=float), base_calibrated,
                                 np.asarray(groups), np.asarray(sites))
    return labels, values, {'baseline': base_record, 'augmented': better_record,
                            'baseline_calibrated_mse': jsonable(base_level)}


def rank_contrast(target, baseline, augmented, groups) -> tuple[list, np.ndarray, list]:
    """Per-family within-background Spearman increment, undefined families named.

    The correlation is the gate's own :func:`group_spearman`, which is what the
    frozen panel's ``spearman_increment`` reads, so this recomputation and the
    frozen record are the same statistic rather than two defensible ones. A
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
                  folds: int = CALIBRATION_FOLDS,
                  seed: int = CALIBRATION_SEED) -> dict:
    """The whole contrast grid for one arm, on one split seed, per channel.

    ``predictions`` holds the held-out prediction vector of each fitted design;
    both baseline designs and both augmented designs must be present.
    ``channels`` maps a measurement-channel name to its target vector: the
    combined channel is the gate's published endpoint and the two protease
    channels are the robustness reading.

    Every cell of the grid is read from the same predictions, so the comparison
    between two cells differs in exactly one declared respect -- the baseline, or
    the metric, never both at once unless the reader asks for it.
    """

    groups, sites = panel['group'], panel['site']
    missing = [design for about in BASELINES.values()
               for design in (about['design'], f"{about['design']}_M")
               if design not in predictions]
    if missing:
        raise ValueError(f'the contrast grid needs every baseline and augmented design; '
                         f'missing {missing}')
    out: dict[str, dict] = {}
    for channel, target in channels.items():
        record: dict = {'calibration': {}, 'undefined_ranking_groups': {},
                        'baseline_transfer_error': {}, 'baseline_calibrated_error': {},
                        'baseline_within_family_ranking': {}}
        labels: list | None = None
        for baseline, about in BASELINES.items():
            design = about['design']
            base, augmented = predictions[design], predictions[f'{design}_M']
            names, transfer = mse_contrast(target, base, augmented, groups, sites)
            _, calibrated, calibration = calibrated_mse_contrast(
                target, base, augmented, groups, sites, folds=folds, seed=seed)
            rank_names, ranking, undefined = rank_contrast(target, base, augmented, groups)
            if labels is None:
                labels = names
            if names != labels or rank_names != labels:
                raise ValueError('family label order differs between readings')
            record[f'{baseline}_transfer_error'] = jsonable(transfer)
            record[f'{baseline}_calibrated_error'] = jsonable(calibrated)
            record[f'{baseline}_ranking'] = jsonable(ranking)
            record['undefined_ranking_groups'][baseline] = undefined
            record['baseline_calibrated_error'][baseline] = calibration.pop(
                'baseline_calibrated_mse')
            record['calibration'][baseline] = calibration
            _, level = group_errors(np.asarray(target, dtype=float),
                                    np.asarray(base, dtype=float),
                                    np.asarray(groups), np.asarray(sites))
            _, rank_level = group_spearman(np.asarray(target, dtype=float),
                                           np.asarray(base, dtype=float),
                                           np.asarray(groups))
            record['baseline_transfer_error'][baseline] = jsonable(level)
            record['baseline_within_family_ranking'][baseline] = [
                None if value is None else float(value) for value in rank_level]
        record['groups'] = labels
        out[channel] = record
    return out


# --------------------------------------------------------------------------- #
# Inference.
# --------------------------------------------------------------------------- #

def simultaneous_family(matrix, columns, *, draws: int = PANEL_DRAWS) -> dict:
    """The project's maximum-statistic band over one declared contrast family.

    The helper is imported rather than restated: the frozen stability replay
    panel, the pairwise residual panel and the position-term follow-up all
    publish under this exact construction, so a recomputed band is comparable
    with the frozen one. Rows are families -- the resampling unit, because every
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
    33-arm family, and never instead of it. The frozen panel's own multiplicity
    note records that reading these as if they were simultaneous is how the
    stability positives were over-read once already.
    """

    record = interval(values, draws=draws, seed=seed)
    record['multiplicity'] = 'marginal for one arm and one contrast; not adjusted for the family'
    return record


def _counts(family: dict, arms: list) -> dict:
    above = family.get('resolved_above_zero', [])
    below = family.get('resolved_below_zero', [])
    return {'above_zero': len(above), 'below_zero': len(below),
            'unresolved': len(arms) - len(above) - len(below),
            'arms_above_zero': list(above), 'arms_below_zero': list(below)}


def _cross(table: dict, arms: list, error: str, rank: str) -> dict:
    tests = {
        'ranking_only': lambda row: row[rank] == 'above_zero' and row[error] != 'above_zero',
        'error_only': lambda row: row[error] == 'above_zero' and row[rank] != 'above_zero',
        'both': lambda row: row[error] == 'above_zero' and row[rank] == 'above_zero',
        'neither': lambda row: row[error] != 'above_zero' and row[rank] != 'above_zero',
    }
    return {label: [arm for arm in arms if test(table[arm])] for label, test in tests.items()}


def ranking_versus_error(families: dict, arms) -> dict:
    """How widespread each reading is, where the two metrics disagree, and what
    the alignment profile is worth.

    The comparison is a count of resolved arms and a cross tabulation rather than
    a pooled difference: the readings carry different units, so a single pooled
    contrast between a rank correlation and a squared kcal/mol would have no
    interpretation, and that caveat travels with the counts wherever they go.

    The primary answer is read over the profile-inclusive baseline, because that
    is the baseline under which a resolved increment means the model added
    something an alignment profile did not already supply. The matched baseline's
    answer is reported beside it, and the difference between the two is reported
    as its own quantity: arms that resolve only without the profile are arms
    whose apparent contribution the profile already contains.
    """

    arms = list(arms)
    available = {name: family for name, family in families.items()
                 if family.get('status') == 'complete'}
    table = {arm: {name: family['verdicts'][arm] for name, family in available.items()}
             for arm in arms}
    counts = {name: _counts(family, arms) for name, family in available.items()}
    pairs = {}
    for label, (error, rank) in (('primary_profile_baseline', PRIMARY_PAIR),
                                 ('companion_matched_baseline', COMPANION_PAIR)):
        if error in available and rank in available:
            pairs[label] = {
                'contrasts': [error, rank],
                'baseline': CONTRASTS[rank]['baseline'],
                'baseline_blocks': CONTRASTS[rank]['baseline_blocks'],
                'basis': 'identical held-out predictions and identical baseline; only the '
                         'metric differs, so a difference in how many arms resolve is a '
                         'difference between ordering and quantitative information',
                'resolved_ranking': counts[rank]['above_zero'],
                'resolved_calibrated_error': counts[error]['above_zero'],
                'ranking_more_widespread': bool(
                    counts[rank]['above_zero'] > counts[error]['above_zero']),
                'cross_tabulation': _cross(table, arms, error, rank),
            }
    profile_cost = {}
    for reading in READINGS:
        profile, matched = f'profile_{reading}', f'matched_{reading}'
        if profile in available and matched in available:
            profile_cost[reading] = {
                'resolved_with_profile_in_baseline': counts[profile]['above_zero'],
                'resolved_without_profile_in_baseline': counts[matched]['above_zero'],
                'arms_resolving_only_without_profile': sorted(
                    set(counts[matched]['arms_above_zero'])
                    - set(counts[profile]['arms_above_zero'])),
                'arms_resolving_with_profile': counts[profile]['arms_above_zero'],
                'reading': 'an arm that resolves only without the profile in the baseline '
                           'contributed information the alignment profile already carries',
            }
    return {
        'arms': arms,
        'primary_baseline': 'profile',
        'per_arm_verdicts': table,
        'resolved_counts': counts,
        'paired_metric_comparison': pairs,
        'what_the_profile_is_worth': profile_cost,
        'multiplicity': 'each contrast is its own 33-column maximum-statistic family over '
                        'the same 101 resampled family groups, with split seeds averaged '
                        'inside the family before resampling; the families are not pooled '
                        'across baselines or across metrics, because a rank correlation and '
                        'a squared kcal/mol are not commensurable and no single band covers '
                        'a comparison between them',
    }


__all__ = [
    'BASELINES', 'CALIBRATION_FOLDS', 'CALIBRATION_SEED', 'COMPANION_PAIR', 'CONTRASTS',
    'DEGENERATE_PREDICTION_SCALE', 'FROZEN_EQUIVALENT', 'PANEL_DRAWS', 'PRIMARY_PAIR',
    'READINGS', 'arm_contrasts', 'calibrated_mse_contrast', 'family_calibrated', 'jsonable',
    'marginal_interval', 'mse_contrast', 'rank_contrast', 'ranking_versus_error',
    'simultaneous_family', 'site_folds', 'vector',
]
