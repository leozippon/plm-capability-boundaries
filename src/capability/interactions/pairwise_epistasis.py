"""Measured double-mutant nonadditivity: frozen roster, label-free extraction plan,
declared feature blocks and the nested held-group comparison.

The endpoint is the wild-type-centred cycle epsilon = y_AB - y_A - y_B + y_WT on the
combined MegaScale ``dG_ML`` scale in kcal/mol, frozen by
``archive/docs/R3/pairwise-baseline-readiness.md``. This module carries no experimental
label on the extraction path: :func:`extraction_plan` emits sequences and cycle
state indices only, so a model forward pass cannot see a measurement.

:func:`measured_singles_correction` is a separate label-side residualizer. On each
outer family fold it fits a response of epsilon to the measured additive
y_A + y_B - y_WT from training groups only, and subtracts that prediction from
the held-out epsilon. It reads no model quantity, and it has no branch that
refits the curve on the held-out families.

Weighting, folds and the ridge recipe are imported from the admitted Readout
analysis rather than restated: :func:`~.readout_analysis.row_weights` gives equal
group weight with equal site-pair weight inside a group,
:func:`~.readout_analysis.family_folds` gives the seeded round-robin held-group
split, and :func:`~.readout_analysis.ridge_predict` gives the weighted-mean-squared-error
ridge with an unpenalised intercept and training-only feature scaling.

The independent unit is the site pair, not the cycle: 99.98% of admitted cycles
sit on site pairs the source study selected as coupling tables, and cycles inside
one site pair share both single-mutant measurements. Every weight, resample and
reported count here is at the group or site-pair level for that reason.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from scipy.stats import rankdata

from ..core.amino_acids import AA20
from ..stability.pairwise_stability import COHORT_SCHEMA, separation_stratum, validate_cycle
from ..context.profile_increment import correlation
from ..context.profiles import cluster_bootstrap
from ..readouts.readout_analysis import family_folds, ridge_predict, row_weights

#: The pre-declared panel. Every unconditioned checkpoint the Readout expansion
#: extracted, named exactly as its extraction arm. ZymCTRL is excluded because it
#: is a biologically conditioned interface on its own 40-assay support and has no
#: unconditional substitute. The roster is fixed before any pairwise scoring and
#: is not a function of any Readout or pairwise outcome.
ROSTER: tuple[str, ...] = (
    'bygpt5-base-en', 'bygpt5-medium-en', 'bygpt5-small-en', 'dialogpt-small',
    'galactica-1.3b', 'galactica-125m', 'galactica-30b', 'galactica-6.7b',
    'gpt2', 'gpt2-large', 'gpt2-medium', 'gpt2-xl', 'instructprotein',
    'llama-2-7b', 'llama-3.2-3b', 'progen2-base', 'progen2-large',
    'progen2-medium', 'progen2-small', 'progen2-xlarge', 'progen3-112m',
    'progen3-3b', 'prollama', 'prollama-stage-1', 'proteinglm-7b-clm',
    'protgpt2', 'protgpt3-1.3b', 'qwen2.5-0.5b', 'qwen2.5-0.5b-instruct',
    'qwen2.5-32b', 'qwen2.5-7b', 'qwen3-8b-base', 'rita-xl',
)

#: Interface stratum of each roster arm, reported separately because a
#: representation cycle can be nonzero purely through segmentation.
TOKENISATION_STRATUM: dict[str, str] = {
    **{arm: 'amino_acid' for arm in (
        'progen2-base', 'progen2-large', 'progen2-medium', 'progen2-small',
        'progen2-xlarge', 'progen3-112m', 'progen3-3b', 'proteinglm-7b-clm',
        'rita-xl', 'prollama', 'prollama-stage-1')},
    **{arm: 'byte' for arm in ('bygpt5-base-en', 'bygpt5-medium-en', 'bygpt5-small-en')},
    **{arm: 'bpe' for arm in (
        'dialogpt-small', 'galactica-1.3b', 'galactica-125m', 'galactica-30b',
        'galactica-6.7b', 'gpt2', 'gpt2-large', 'gpt2-medium', 'gpt2-xl',
        'instructprotein', 'llama-2-7b', 'llama-3.2-3b', 'protgpt2',
        'protgpt3-1.3b', 'qwen2.5-0.5b', 'qwen2.5-0.5b-instruct', 'qwen2.5-32b',
        'qwen2.5-7b', 'qwen3-8b-base')},
}

#: ProGen3 has no kernel above float16 and its expert mixture reduces over the
#: flattened batch, so it is extracted in bfloat16 exactly as the admitted Readout
#: production did. Every other arm is float32.
ARM_DTYPE: dict[str, str] = {'progen3-3b': 'bfloat16', 'progen3-112m': 'bfloat16'}

#: Singleton extraction for the whole panel. At batch size one the production
#: forward and the repeat-check forward are the same single-row computation, so
#: the 0.001 gates are inapplicable rather than passed; they are retained to
#: record batch-composition invariance trivially and to measure run-to-run
#: reproducibility of the same single-row computation.
PRODUCTION_BATCH_SIZE = 1

#: Four frozen feature blocks in the declared order, and the Gaussian projection
#: the Readout recipe declares: entries of standard deviation 1/sqrt(256), seeds
#: 20260923 through 20260926 consumed in that order, depending only on the width
#: and the seed and never on a label or an observed vector.
FEATURE_BLOCKS = ('middle_mean', 'middle_last', 'final_mean', 'final_last')
PROJECTION_SEED = 20260923
PROJECTION_DIM = 256


def require_projected_width(width: int, *, arm: str, source: str) -> int:
    """Refuse a projected representation that is not the admitted four-block design.

    The admitted design pools four summaries across two depths, so its width is
    fixed at ``len(FEATURE_BLOCKS) * PROJECTION_DIM``. A narrower array is not a
    narrower version of that design -- one depth's summaries are a differently
    constructed measurement, at different capacity -- so a stage handed one must
    refuse rather than fit it and report the result beside the published number.

    ``fit_stability_singles.load_arm`` has always refused this. The pairwise stage
    did not, so the same substitution failed loudly on three gates and silently on
    the fourth; the silent direction is the one that produces a plausible wrong
    number, which is why this lives here and is called from both.
    """
    expected = len(FEATURE_BLOCKS) * PROJECTION_DIM
    if int(width) != expected:
        raise ValueError(
            f'{arm}: {source} carries a {int(width)}-coordinate projected representation and this '
            f'design is {expected} coordinates, {len(FEATURE_BLOCKS)} pooled summaries at '
            f'{PROJECTION_DIM} each. A different width is a different design at a different '
            'capacity, not a replay of this one')
    return expected

#: Outer held-group split seeds and the inner seed rule, both unchanged from the
#: admitted Readout recipe; only the group universe changes.
SPLIT_SEEDS = (20260923, 20260924, 20260925)
OUTER_SPLITS = 5
INNER_SPLITS = 4
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260923

_CODE = {residue: index for index, residue in enumerate(AA20)}


def projection_matrices(width: int, *, dim: int = PROJECTION_DIM,
                        seed: int = PROJECTION_SEED) -> list[np.ndarray]:
    """One fixed Gaussian matrix per feature block, as the Readout recipe declares."""

    if width < 1 or dim < 1:
        raise ValueError('projection needs a positive width and target dimension')
    return [np.random.default_rng(seed + index).normal(
        0, 1 / np.sqrt(dim), size=(width, dim)).astype(np.float32)
        for index in range(len(FEATURE_BLOCKS))]


def cycle_states(wildtype: str, sequences: list[str]) -> tuple[str, str, str, str]:
    """Canonical state order: wild type, lower-site single, higher-site single, double.

    The cohort builder already emits ``[wt, a, b, double]`` with ``a`` carrying the
    lower-index substitution, but the order is re-derived here rather than trusted,
    because every first-order feature and every tokenisation descriptor below is
    defined against this ordering and a silent swap would be unreadable downstream.
    """

    wild, first, second, double = sequences
    if wild != wildtype:
        raise ValueError('cycle wild-type state is not the background wild type')
    low, high = validate_cycle(wild, first, second, double)
    if low > high:
        first, second = second, first
        low, high = high, low
    if first[low] == wild[low] or second[high] == wild[high]:
        raise ValueError('canonical single states do not carry their own substitution')
    return wild, first, second, double


def extraction_plan(cohort: dict, cohort_sha256: str) -> dict:
    """Label-free extraction plan: sequences and cycle state indices only.

    No measured stability, cycle epsilon or row-level measurement metadata is
    copied. This is what reaches a GPU, so the extraction cannot be influenced by
    an outcome even in principle.
    """

    if cohort.get('schema') != COHORT_SCHEMA:
        raise ValueError('unexpected cohort schema')
    if not isinstance(cohort_sha256, str) or len(cohort_sha256) != 64:
        raise ValueError('extraction plan must be bound to the cohort digest')
    backgrounds = []
    for row in cohort['backgrounds']:
        wildtype = row['cycles'][0]['sequences'][0]
        order = [wildtype] + sorted(set(row['sequences']) - {wildtype})
        index = {sequence: position for position, sequence in enumerate(order)}
        if set(order) != set(row['sequences']) or len(order) != len(row['sequences']):
            raise ValueError(f"{row['name']}: sequence set does not contain its wild type once")
        cycles = []
        for cycle in row['cycles']:
            states = cycle_states(wildtype, cycle['sequences'])
            positions = sorted(cycle['positions'])
            if separation_stratum(*positions) != cycle['separation']:
                raise ValueError(f"{row['name']}: separation stratum disagrees with its positions")
            cycles.append({'positions': positions, 'separation': cycle['separation'],
                           'states': [index[state] for state in states]})
        backgrounds.append({'name': row['name'], 'group': row['group'], 'length': row['length'],
                            'wildtype': wildtype, 'sequences': order, 'cycles': cycles})
    sequences = {s for row in backgrounds for s in row['sequences']}
    return {'schema': 'pairwise_epistasis_extraction_v1', 'cohort_sha256': cohort_sha256,
            'roster': list(ROSTER), 'projection': {'seed': PROJECTION_SEED, 'dim': PROJECTION_DIM,
                                                   'blocks': list(FEATURE_BLOCKS)},
            'backgrounds': backgrounds,
            'summary': {'backgrounds': len(backgrounds),
                        'groups': len({row['group'] for row in backgrounds}),
                        'cycles': sum(len(row['cycles']) for row in backgrounds),
                        'distinct_sequences': len(sequences),
                        'residues': sum(len(s) for s in sequences),
                        'length_range': [min(map(len, sequences)), max(map(len, sequences))],
                        'site_pairs': len({(row['name'], tuple(c['positions']))
                                           for row in backgrounds for c in row['cycles']})}}


def plan_digest(plan: dict) -> str:
    """Content digest over the extraction plan's scientific fields."""

    import json
    return hashlib.sha256(json.dumps(
        {k: plan[k] for k in ('schema', 'cohort_sha256', 'roster', 'projection', 'backgrounds')},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Declared feature blocks. Every block below is a function of sequences, frozen
# sequence-derived quantities or training-group labels only. No held-out group
# label reaches any of them.
# --------------------------------------------------------------------------- #

#: Ordered coordinate names of the cycle-level sequence/profile control C.
C_BLOCKS = ('ident', 'geom', 'comp', 'local', 'prof')

#: Mutation-local independent-site profile summaries, in order. The final
#: coordinate is the availability indicator: a background with no qualifying
#: homolog row carries zeros in the preceding coordinates and a zero here, which
#: is a declared absence rather than an imputed profile.
PROFILE_FEATURE_ORDER = (
    'log_frequency_wild_low', 'log_frequency_wild_high',
    'log_frequency_mutant_low', 'log_frequency_mutant_high',
    'log_odds_low', 'log_odds_high', 'log_odds_sum',
    'column_entropy_low_nats', 'column_entropy_high_nats',
    'column_weight_low', 'column_weight_high',
    'wildtype_log10_neff', 'wildtype_supported_column_fraction', 'profile_available',
)

#: Tokenisation interface descriptors, in order. A representation cycle can be
#: nonzero purely through segmentation, so these enter the matched baseline.
TOKENISATION_FEATURE_ORDER = (
    'pooled_tokens_wild', 'pooled_tokens_low', 'pooled_tokens_high', 'pooled_tokens_double',
    'pooled_token_count_cycle',
    'common_prefix_tokens_low', 'common_prefix_tokens_high', 'common_prefix_tokens_double',
    'common_suffix_tokens_low', 'common_suffix_tokens_high', 'common_suffix_tokens_double',
    'segmentation_changed_tokens_low', 'segmentation_changed_tokens_high',
    'segmentation_changed_tokens_double',
    'pooled_tokens_per_residue_wild', 'pooled_tokens_per_residue_double',
)

#: First-stage state descriptors for the nonlinear-additive nuisance G.
G_STATE_BLOCKS = ('count', 'ident', 'comp', 'geom', 'prof')

PROFILE_PSEUDO_FREQUENCY = 1e-6


def _codes(sequence: str) -> list[int]:
    try:
        return [_CODE[residue] for residue in sequence]
    except KeyError as error:
        raise ValueError(f'noncanonical residue {error.args[0]!r} in a cohort sequence') from error


def _dipeptide_counts(sequence: str) -> np.ndarray:
    counts = np.zeros(len(AA20) * len(AA20))
    codes = _codes(sequence)
    for left, right in zip(codes[:-1], codes[1:]):
        counts[left * len(AA20) + right] += 1.0
    return counts / max(len(sequence) - 1, 1)


def _composition(sequence: str) -> np.ndarray:
    return np.array([sequence.count(residue) for residue in AA20], dtype=float) / len(sequence)


def profile_summary(profile, wildtype: str, sites: tuple[int, int],
                    mutants: tuple[str, str]) -> np.ndarray:
    """Independent-site profile summaries at the cycle's two mutated sites.

    ``profile`` is a :class:`~.profiles.Profile` fitted on the frozen homolog rows
    of this background, or ``None`` when the background retrieved no qualifying
    homolog. An absent profile yields zeros and a zero availability indicator; it
    is never replaced by a pooled or nearest-background profile.
    """

    values = np.zeros(len(PROFILE_FEATURE_ORDER))
    if profile is None:
        return values
    frequencies = np.clip(profile.frequencies, PROFILE_PSEUDO_FREQUENCY, None)
    entropy = -(frequencies * np.log(frequencies)).sum(axis=1)
    low, high = sites
    wild = [np.log(frequencies[site, _CODE[wildtype[site]]]) for site in (low, high)]
    mutant = [np.log(frequencies[site, _CODE[residue]]) for site, residue in zip((low, high), mutants)]
    values[:2] = wild
    values[2:4] = mutant
    values[4] = mutant[0] - wild[0]
    values[5] = mutant[1] - wild[1]
    values[6] = values[4] + values[5]
    values[7:9] = [entropy[low], entropy[high]]
    values[9:11] = [profile.column_weight[low], profile.column_weight[high]]
    values[11] = profile.log10_neff
    values[12] = float((profile.column_weight > 0).sum()) / profile.length
    values[13] = 1.0
    return values


def cycle_control_features(wildtype: str, states: tuple[str, str, str, str],
                           positions: tuple[int, int], separation: str, profile) -> np.ndarray:
    """The sequence/profile control C for one cycle, in the declared block order."""

    wild, low_state, high_state, double = states
    for state in states:
        _codes(state)
    low, high = positions[0] - 1, positions[1] - 1
    identity = np.zeros((len(AA20), len(AA20)))
    for site, state in ((low, low_state), (high, high_state)):
        identity[_CODE[wildtype[site]], _CODE[state[site]]] += 1.0
    length = len(wildtype)
    relative = np.array([(low + 1) / length, (high + 1) / length])
    geometry = np.array([
        relative[0], relative[1], relative.mean(), relative.std(),
        (high - low) / length, np.log10(high - low),
        float(separation == '1-2'), float(separation == '3-9'), float(separation == '10+'),
        length / 1024,
    ])
    composition = np.concatenate([_composition(wild), _composition(double)])
    local = _dipeptide_counts(double) - _dipeptide_counts(wild)
    prof = profile_summary(profile, wildtype, (low, high), (low_state[low], high_state[high]))
    return np.concatenate([identity.ravel(), geometry, composition, local, prof])


def state_features(wildtype: str, sequence: str, profile) -> np.ndarray:
    """First-stage descriptors of one absolute stability state, for the nuisance G."""

    sites = [index for index in range(len(wildtype)) if sequence[index] != wildtype[index]]
    identity = np.zeros((len(AA20), len(AA20)))
    for site in sites:
        identity[_CODE[wildtype[site]], _CODE[sequence[site]]] += 1.0
    length = len(wildtype)
    relative = np.array([(site + 1) / length for site in sites]) if sites else np.zeros(1)
    geometry = np.array([relative.mean(), relative.std(), relative.min(), relative.max(),
                         length / 1024]) if sites else np.array([0., 0., 0., 0., length / 1024])
    prof = np.zeros(7)
    if profile is not None:
        frequencies = np.clip(profile.frequencies, PROFILE_PSEUDO_FREQUENCY, None)
        entropy = -(frequencies * np.log(frequencies)).sum(axis=1)
        if sites:
            prof[0] = sum(float(np.log(frequencies[s, _CODE[sequence[s]]])
                                - np.log(frequencies[s, _CODE[wildtype[s]]])) for s in sites)
            prof[1] = float(np.mean([entropy[s] for s in sites]))
            prof[2] = float(np.mean([profile.column_weight[s] for s in sites]))
        supported = profile.column_weight > 0
        prof[3] = float(entropy[supported].mean()) if supported.any() else 0.0
        prof[4] = profile.log10_neff
        prof[5] = float(supported.sum()) / profile.length
        prof[6] = 1.0
    return np.concatenate([[float(len(sites))], identity.ravel(),
                           _composition(wildtype), _composition(sequence), geometry, prof])


def tokenisation_features(pooled_counts: np.ndarray, token_ids: list[list[int]],
                          states: tuple[int, int, int, int], length: int) -> np.ndarray:
    """Per-state token counts, their cycle, position shifts and segmentation changes.

    ``common_prefix`` and ``common_suffix`` are token counts shared with the wild
    type at each end, so a substitution that shifts every later token registers
    here. ``segmentation_changed_tokens`` counts the tokens of both states that
    lie outside that shared prefix and suffix.
    """

    wild, low, high, double = states
    reference = token_ids[wild]

    def shared(other: list[int]) -> tuple[int, int, int]:
        prefix = 0
        while prefix < min(len(reference), len(other)) and reference[prefix] == other[prefix]:
            prefix += 1
        suffix = 0
        while (suffix < min(len(reference), len(other)) - prefix
               and reference[-1 - suffix] == other[-1 - suffix]):
            suffix += 1
        return prefix, suffix, len(reference) + len(other) - 2 * (prefix + suffix)

    parts = [shared(token_ids[index]) for index in (low, high, double)]
    counts = [float(pooled_counts[index]) for index in (wild, low, high, double)]
    return np.array(counts + [counts[3] - counts[1] - counts[2] + counts[0]]
                    + [float(part[0]) for part in parts]
                    + [float(part[1]) for part in parts]
                    + [float(part[2]) for part in parts]
                    + [counts[0] / length, counts[3] / length])


def cycle_contrast(values: np.ndarray, states) -> np.ndarray:
    """``x_AB - x_A - x_B + x_WT`` over the leading axis of per-state quantities."""

    wild, low, high, double = (np.asarray(states)[:, i] for i in range(4))
    return values[double] - values[low] - values[high] + values[wild]


# --------------------------------------------------------------------------- #
# The nonlinear-additive nuisance G, and the nested held-group comparison.
# --------------------------------------------------------------------------- #

ALPHAS = (.01, .1, 1., 10., 100.)


def _select_alpha(losses: np.ndarray) -> int:
    """Argmin with exact ties broken toward stronger regularisation."""

    return len(losses) - 1 - int(np.argmin(losses[::-1]))


def nuisance_cycle(states: dict, cycle_states: np.ndarray, held: np.ndarray,
                   inner_partition: list, *, device='cpu') -> tuple[np.ndarray, dict]:
    """The nonlinear-additive control G for one outer fold.

    A first-stage ridge predictor of absolute WT/single/double stability is fitted
    on the outer training groups only, its regularisation selected on the same
    inner held-group partition the outer ridge tuning uses. Training-state
    predictions are cross-fitted on that partition, an isotonic calibration is
    fitted on those cross-fitted predictions, and the calibration is then applied
    to all four states of every cycle. The resulting calibrated cycle lets a
    global monotone response induce apparent interaction without any fitted pair
    term. No held-out group label enters the first stage, the regularisation
    choice or the calibration.
    """

    from sklearn.isotonic import IsotonicRegression

    x, y = states['features'], states['y']
    groups, weights = states['group'], states['weight']
    train = np.flatnonzero(~np.isin(groups, held))
    test = np.flatnonzero(np.isin(groups, held))
    if not len(train) or not len(test):
        raise ValueError('nuisance fold has an empty side')
    losses = np.zeros(len(ALPHAS))
    crossfit = np.full((len(train), len(ALPHAS)), np.nan)
    position = {index: order for order, index in enumerate(train)}
    for validation in inner_partition:
        fit = train[~np.isin(groups[train], validation)]
        valid = train[np.isin(groups[train], validation)]
        if not len(fit) or not len(valid):
            raise ValueError('nuisance inner fold has an empty side')
        prediction = ridge_predict(x[fit], y[fit], weights[fit], x[valid], ALPHAS, device)
        crossfit[[position[index] for index in valid]] = prediction
        share = weights[valid] / weights[valid].sum()
        losses += len(validation) * (share[:, None] * (prediction - y[valid][:, None]) ** 2).sum(0)
    if not np.isfinite(crossfit).all():
        raise ValueError('incomplete cross-fitted first-stage predictions')
    index = _select_alpha(losses / len(set(groups[train])))
    alpha = ALPHAS[index]
    fitted = np.empty(len(y))
    fitted[train] = crossfit[:, index]
    fitted[test] = ridge_predict(x[train], y[train], weights[train], x[test], [alpha], device)[:, 0]
    calibration = IsotonicRegression(increasing=True, out_of_bounds='clip')
    calibration.fit(crossfit[:, index], y[train], sample_weight=weights[train])
    wild, low, high, double = (cycle_states[:, i] for i in range(4))
    additive = fitted[low] + fitted[high] - fitted[wild]
    calibrated = {name: calibration.predict(value) for name, value in
                  (('wild', fitted[wild]), ('low', fitted[low]),
                   ('high', fitted[high]), ('additive', additive))}
    cycle = (calibrated['additive'] - calibrated['low']
             - calibrated['high'] + calibrated['wild'])
    block = np.column_stack([cycle, additive, fitted[wild], fitted[low], fitted[high]])
    share = weights[test] / weights[test].sum()
    centre = float((share * y[test]).sum())
    residual = float((share * (y[test] - fitted[test]) ** 2).sum())
    variance = float((share * (y[test] - centre) ** 2).sum())
    diagnostics = {
        'selected_alpha': alpha,
        'uncalibrated_additive_cycle_max_absolute': float(np.max(np.abs(
            additive - fitted[low] - fitted[high] + fitted[wild]))),
        'held_out_states': int(len(test)),
        'held_out_weighted_r2': None if variance <= 0 else 1.0 - residual / variance,
        'held_out_spearman': correlation(rankdata(fitted[test]), rankdata(y[test])),
        'calibration_response_range_kcal_mol': float(
            calibration.predict(np.array([crossfit[:, index].min(), crossfit[:, index].max()]))
            @ np.array([-1.0, 1.0])),
        'unused_state_count': int(len(y) - len(train) - len(test)),
    }
    return block, diagnostics


# --------------------------------------------------------------------------- #
# Family-holdout response of epsilon to the measured single-mutant additive.
# --------------------------------------------------------------------------- #

#: Bin count of the label instrument's in-sample global response. A cross-fit
#: uses the same count on training groups only.
MEASURED_SINGLES_BINS = 20
MEASURED_SINGLES_METHODS = ('isotonic', 'quantile_bins')


def _one_dimensional(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    if not np.isfinite(array).all():
        raise ValueError(f'{name} contains a nonfinite value')
    return array


def _group_labels(groups) -> np.ndarray:
    array = np.asarray(groups)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError('groups must be a non-empty one-dimensional array')
    return array


def _quantile_bin_predict(train_x: np.ndarray, train_y: np.ndarray,
                          query_x: np.ndarray, bins: int) -> tuple[np.ndarray, dict]:
    """Twenty-bin mean of epsilon on the additive, edges fit on the training rows.

    Bin edges and bin means are the label instrument's rule: quantile edges and
    the unweighted mean of epsilon inside each bin. A query outside the training
    range, or one that lands in a bin no training row occupies, takes the nearest
    occupied bin's mean rather than zero.
    """

    if bins < 2:
        raise ValueError('quantile-bin response needs at least two bins')
    if len(train_x) < bins:
        raise ValueError(f'{bins} quantile bins need at least {bins} training cycles')
    edges = np.quantile(train_x, np.linspace(0.0, 1.0, bins + 1))

    def assign(values: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(edges, values, side='right') - 1, 0, bins - 1)

    train_index = assign(train_x)
    means = np.full(bins, np.nan)
    for bin_index in range(bins):
        selected = train_index == bin_index
        if selected.any():
            means[bin_index] = float(train_y[selected].mean())
    occupied = np.flatnonzero(np.isfinite(means))
    if occupied.size == 0:
        raise ValueError('quantile-bin response has no occupied bin')
    empty = int(np.isnan(means).sum())
    for bin_index in np.flatnonzero(np.isnan(means)):
        nearest = int(occupied[np.argmin(np.abs(occupied - bin_index))])
        means[bin_index] = means[nearest]
    return means[assign(query_x)], {'bins': bins, 'empty_bins_filled': empty}


def _isotonic_predict(train_x: np.ndarray, train_y: np.ndarray,
                      query_x: np.ndarray) -> tuple[np.ndarray, dict]:
    """Monotone regression of epsilon on the additive, direction chosen on training rows.

    A constant training response has no direction. It predicts that constant, which
    is the isotonic solution, without asking a rank correlation to choose a slope.
    """

    import warnings

    from sklearn.isotonic import IsotonicRegression

    if float(np.std(train_y)) < 1e-12 or float(np.std(train_x)) < 1e-12:
        return np.full(len(query_x), float(train_y.mean())), {
            'increasing': None, 'constant_training_response': True}
    model = IsotonicRegression(increasing='auto', out_of_bounds='clip')
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', message='Confidence interval of the Spearman', module='sklearn.isotonic')
        model.fit(train_x, train_y)
    return model.predict(query_x), {
        'increasing': bool(model.increasing_), 'constant_training_response': False}


def measured_singles_predict(train_additive, train_epsilon, query_additive, *,
                             method: str = 'isotonic',
                             bins: int = MEASURED_SINGLES_BINS) -> tuple[np.ndarray, dict]:
    """Predict epsilon from the measured additive using training rows only.

    ``method='quantile_bins'`` is the label instrument's bin mean.
    ``method='isotonic'`` is a monotone map whose direction is selected from the
    training rows. Query rows contribute no label.
    """

    if method not in MEASURED_SINGLES_METHODS:
        raise ValueError(f'unknown measured-singles method {method}')
    train_x = _one_dimensional(train_additive, 'training additive')
    train_y = _one_dimensional(train_epsilon, 'training epsilon')
    query_x = _one_dimensional(query_additive, 'query additive')
    if len(train_x) != len(train_y) or len(train_x) == 0 or len(query_x) == 0:
        raise ValueError('measured-singles response needs aligned non-empty training and query rows')
    if method == 'quantile_bins':
        prediction, info = _quantile_bin_predict(train_x, train_y, query_x, bins)
    else:
        prediction, info = _isotonic_predict(train_x, train_y, query_x)
    info['method'] = method
    return np.asarray(prediction, dtype=float), info


def _require_fold_partition(groups: np.ndarray, folds) -> list[list]:
    """Refuse anything other than a partition of the observed groups into non-empty folds."""

    if not folds:
        raise ValueError('measured-singles correction needs the outer held-group partition')
    present = set(np.unique(groups).tolist())
    normalised, seen = [], []
    for index, held in enumerate(folds):
        labels = [label.item() if isinstance(label, np.generic) else label for label in held]
        if not labels:
            raise ValueError(f'outer fold {index} holds out no groups')
        train = np.flatnonzero(~np.isin(groups, labels))
        test = np.flatnonzero(np.isin(groups, labels))
        if len(train) == 0 or len(test) == 0:
            raise ValueError(f'outer fold {index} has an empty side')
        normalised.append(labels)
        seen.extend(labels)
    if len(seen) != len(set(seen)):
        raise ValueError('outer folds overlap, so one group would be scored by two curves')
    if set(seen) != present:
        missing = sorted(present - set(seen))
        extra = sorted(set(seen) - present)
        raise ValueError(
            'outer folds are not a partition of the groups; '
            f'missing {missing}, extra {extra}')
    return normalised


def measured_singles_correction(epsilon, additive, groups, folds, *,
                                method: str = 'isotonic',
                                bins: int = MEASURED_SINGLES_BINS) -> dict:
    """Subtract a family-holdout response of epsilon to the measured additive.

    ``folds`` is the outer held-group partition. Each fold's curve is fit on the
    complementary groups only and applied to the held-out cycles. Held-out family
    labels never enter the curve. The function does not refit on the pooled
    sample when a fold is empty or unrecognised; those inputs raise.
    """

    target = _one_dimensional(epsilon, 'epsilon')
    regressor = _one_dimensional(additive, 'measured additive')
    labels = _group_labels(groups)
    if not (len(target) == len(regressor) == len(labels)):
        raise ValueError('epsilon, measured additive and groups differ in length')
    partition = _require_fold_partition(labels, folds)
    prediction = np.full(len(target), np.nan)
    records = []
    for index, held in enumerate(partition):
        train = np.flatnonzero(~np.isin(labels, held))
        test = np.flatnonzero(np.isin(labels, held))
        fitted, info = measured_singles_predict(
            regressor[train], target[train], regressor[test], method=method, bins=bins)
        prediction[test] = fitted
        records.append({'fold': index, 'held_groups': held,
                        'training_cycles': int(len(train)), 'held_out_cycles': int(len(test)),
                        **info})
    if not np.isfinite(prediction).all():
        raise ValueError('measured-singles correction left a cycle without a prediction')
    return {'method': method, 'prediction': prediction, 'residual': target - prediction,
            'folds': records}


def require_recorded_family_folds(groups, recorded_held, *, n_splits: int, seed: int) -> dict:
    """Return the recorded held-group partition after it reproduces ``family_folds``.

    A mismatch raises. The seeded partition is not written in place of the
    recorded one, and this function fits nothing.
    """

    reproduced = family_folds(groups, n_splits, seed)
    recorded = [[label.item() if isinstance(label, np.generic) else label for label in held]
                for held in recorded_held]
    if reproduced != recorded:
        raise ValueError(
            f'family fold assignment at seed {seed} does not reproduce the recorded held groups; '
            'the in-sample curve is not used in its place')
    return {'seed': int(seed), 'held_groups': recorded,
            'fold_identity_sha256': fold_identity([{'held_groups': held} for held in recorded])}


#: Nested control sets over the declared blocks. ``C`` is the sequence/profile
#: baseline, ``G`` the nonlinear-additive nuisance, ``T`` the tokenisation
#: interface descriptors and ``Q`` the fitted pairwise sequence contrast. The
#: two first-order sets add the constituent single-mutant model quantities to the
#: matched baseline, so the interaction contrast is scored against a control that
#: already carries richer single-mutant information.
CONTROL_SETS: dict[str, tuple[str, ...]] = {
    'C': C_BLOCKS,
    'C_G': (*C_BLOCKS, 'G'),
    'C_G_T': (*C_BLOCKS, 'G', 'T'),
    'C_G_T_Q': (*C_BLOCKS, 'G', 'T', 'Q'),
    'C_G_T_M1': (*C_BLOCKS, 'G', 'T', 'M1'),
    'C_G_T_R1': (*C_BLOCKS, 'G', 'T', 'R1'),
}

#: Model additions evaluated over a control set. Likelihood and representation
#: are added separately, never only jointly.
ADDITIONS: dict[str, tuple[str, ...]] = {'': (), 'M': ('M',), 'R': ('R',), 'MR': ('M', 'R')}

#: Which control set carries which additions on which support. ``all`` is the
#: 64-group support; ``q`` is the 45-group support on which baseline Q is
#: admissible. Q is never compared against anything on unmatched support.
DESIGN_PLAN: dict[str, dict[str, tuple[str, ...]]] = {
    'all': {'C': ('', 'M', 'R', 'MR'), 'C_G': ('', 'M', 'R', 'MR'),
            'C_G_T': ('', 'M', 'R', 'MR'), 'C_G_T_M1': ('', 'M'), 'C_G_T_R1': ('', 'R')},
    'q': {'C_G_T': ('', 'M', 'R', 'MR'), 'C_G_T_Q': ('', 'M', 'R', 'MR')},
}


def design_names(support: str) -> list[str]:
    return [f'{control}+{addition}' if addition else control
            for control, additions in DESIGN_PLAN[support].items() for addition in additions]


def design_blocks(name: str) -> tuple[str, ...]:
    control, _, addition = name.partition('+')
    return (*CONTROL_SETS[control], *ADDITIONS[addition])


def nested_compare(panel: dict, support: str, *, seed: int, device='cpu') -> dict:
    """Held-group predictions for every design on one support and one split seed.

    Every design sees the identical rows, the identical outer and inner group
    partitions and the identical label budget; only its declared column blocks
    differ. Feature centring and scaling are fitted on training rows with the
    training weights inside the ridge recipe, so no held-out group label reaches
    feature scaling, regularisation selection or the nuisance calibration.
    """

    groups = panel['group']
    weights = row_weights(panel['site_pair'], groups)
    names = design_names(support)
    predictions = {name: np.full(len(groups), np.nan) for name in names}
    records, nuisance = [], []
    for outer, held in enumerate(family_folds(groups, OUTER_SPLITS, seed)):
        train = np.flatnonzero(~np.isin(groups, held))
        test = np.flatnonzero(np.isin(groups, held))
        inner_partition = family_folds(groups[train], INNER_SPLITS, seed + 100 + outer)
        block, diagnostics = nuisance_cycle(panel['states'], panel['cycle_states'],
                                            np.asarray(held), inner_partition, device=device)
        diagnostics.update(fold=outer, held_groups=list(held))
        nuisance.append(diagnostics)
        blocks = dict(panel['blocks'], G=block)
        fold = {'fold': outer, 'held_groups': list(held),
                'training_groups': sorted(set(groups[train])), 'alpha': {}, 'dimensions': {}}
        for name in names:
            x = np.column_stack([blocks[key] for key in design_blocks(name)])
            if not np.isfinite(x).all():
                raise ValueError(f'{name}: nonfinite design matrix')
            losses = np.zeros(len(ALPHAS))
            for validation in inner_partition:
                fit = train[~np.isin(groups[train], validation)]
                valid = train[np.isin(groups[train], validation)]
                predicted = ridge_predict(x[fit], panel['epsilon'][fit], weights[fit],
                                          x[valid], ALPHAS, device)
                losses += len(validation) * (
                    weights[valid][:, None] * (predicted - panel['epsilon'][valid][:, None]) ** 2).sum(0)
            alpha = ALPHAS[_select_alpha(losses / len(set(groups[train])))]
            predictions[name][test] = ridge_predict(
                x[train], panel['epsilon'][train], weights[train], x[test], [alpha], device)[:, 0]
            fold['alpha'][name] = alpha
            fold['dimensions'][name] = int(x.shape[1])
        records.append(fold)
    for name, value in predictions.items():
        if not np.isfinite(value).all():
            raise ValueError(f'{name}: incomplete held-out predictions')
    predictions['ADDITIVE_NULL'] = np.zeros(len(groups))
    return {'predictions': predictions, 'folds': records, 'nuisance': nuisance,
            'row_weights': weights}


def group_errors(epsilon: np.ndarray, prediction: np.ndarray, groups: np.ndarray,
                 site_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-group mean squared error in squared kcal/mol, site pairs weighted equally.

    Cycles inside one site pair share both single-mutant measurements, so the site
    pair carries the weight inside a group and the cycle count never enters as a
    sample size.
    """

    labels = sorted(set(groups))
    values = np.empty(len(labels))
    for index, label in enumerate(labels):
        rows = np.flatnonzero(groups == label)
        pairs = sorted(set(site_pairs[rows]))
        values[index] = float(np.mean([
            np.mean((epsilon[rows][site_pairs[rows] == pair]
                     - prediction[rows][site_pairs[rows] == pair]) ** 2) for pair in pairs]))
    return np.asarray(labels), values


def group_spearman(epsilon: np.ndarray, prediction: np.ndarray,
                   groups: np.ndarray) -> tuple[list, list]:
    """Within-background rank correlation per group; ``None`` where undefined.

    A constant prediction has no rank ordering, so the additive null's Spearman is
    undefined rather than zero, and every undefined group is counted.
    """

    labels, values = sorted(set(groups)), []
    for label in labels:
        rows = np.flatnonzero(groups == label)
        values.append(None if len(rows) < 3 or np.std(prediction[rows]) < 1e-12
                      else correlation(rankdata(prediction[rows]), rankdata(epsilon[rows])))
    return labels, values


def row_identity(groups, site_pairs, epsilon) -> str:
    """Digest of the evaluated rows: held group, site pair and target, in order.

    Several comparisons prove that a refit used identical rows by comparing this
    digest, so it must not depend on anything but the rows. Each target is
    rendered through ``float`` before ``repr`` rather than left as a NumPy scalar:
    ``repr(np.float64(0.5))`` is ``0.5`` under NumPy 1.26 and ``np.float64(0.5)``
    under NumPy 2, so a NumPy-scalar rendering would hash identical data to
    different values across interpreters and make the check unreliable in both
    directions. ``float`` reproduces the NumPy 1.26 rendering exactly, so the
    digests already recorded under the pinned runtime are unchanged.
    """

    if not (len(groups) == len(site_pairs) == len(epsilon)):
        raise ValueError('row identity needs aligned group, site-pair and target arrays')
    return hashlib.sha256('\n'.join(
        f'{group}|{pair}|{float(target)!r}'
        for group, pair, target in zip(groups, site_pairs, epsilon)).encode()).hexdigest()


def fold_identity(folds) -> str:
    """Digest of the held-group partition, one declaration for every fit that compares it.

    Two fits are the same fit only if they held the same groups out in the same
    folds, and the global-context gate refuses to proceed when its digest differs
    from the admitted pairwise fit's. A comparison like that is only as reliable
    as the two sides being computed the same way, and until now the expression
    lived twice: once in ``fit_pairwise_epistasis.py``, which records the digest,
    and once in ``fit_global_context_gate.py``, which checks it. The expression is
    preserved exactly, so every digest already recorded under it reproduces.

    ``held_groups`` carries plain group labels. A NumPy integer would make
    ``json.dumps`` raise rather than hash differently, which is the safe direction
    of the rendering hazard ``row_identity`` documents, so no coercion is applied
    here and a caller passing NumPy labels is refused rather than silently served.
    """

    return hashlib.sha256(json.dumps(
        [fold['held_groups'] for fold in folds], sort_keys=True).encode()).hexdigest()


def kish_effective_site_pairs(groups, site_pairs) -> float:
    """Effective site-pair count under the weighting the estimator actually applies.

    Groups are weighted equally and site pairs equally inside a group, which is
    what :func:`~.readout_analysis.row_weights` realises, so the effective count
    is the Kish count of those site-pair weights. It is deliberately not the
    count under a cycle-share weighting, which is a different convention and a
    different number on the same support.
    """

    weights = []
    groups, site_pairs = np.asarray(groups), np.asarray(site_pairs)
    labels = sorted(set(groups.tolist()))
    for label in labels:
        pairs = set(site_pairs[groups == label].tolist())
        weights.extend([1 / (len(labels) * len(pairs))] * len(pairs))
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / (weights ** 2).sum())


def interval(values, *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Group-bootstrap percentile interval on the group-equal mean.

    Resampling whole groups keeps every shared wild-type and single-mutant
    measurement's signed contribution to its own cycles intact; cycle-level
    resampling would break exactly that dependence. The interval conditions on
    the fitted cross-validation predictions and omits training and split variation.
    """

    values = list(values)
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return {'point': None, 'interval': None, 'groups': 0,
                'undefined_groups': len(values), 'excludes_zero': None,
                'undefined': 'no group carries a defined value for this statistic'}
    record = cluster_bootstrap(finite, list(range(len(finite))), resamples=draws, seed=seed)
    return {'point': record['point'], 'interval': record['interval'],
            'groups': len(finite), 'undefined_groups': len(values) - len(finite),
            'excludes_zero': record['excludes_zero']}


#: Largest admissible disagreement, in kcal/mol, between a panel's own cycle
#: target and the same cycle recomputed from its four absolute-stability states.
#: The two are the same arithmetic in a different order, so a larger gap means
#: the state rows are not the ones the cycle indexes.
CYCLE_IDENTITY_TOLERANCE = 1e-9

#: The support name carried by a fit that admits every frozen cycle, and the one
#: carried by a fit whose support is restricted by the declared indel exclusion.
SUPPORT_ALL = 'all'
SUPPORT_INDEL_EXCLUDED = 'indel-excluded'


def panel_state_keys(plan: dict, groups: set) -> list[tuple[str, str]]:
    """``(background, sequence)`` of every absolute-stability state row, in the
    order the panel assembly appends them."""

    return [(background['name'], sequence)
            for background in plan['backgrounds'] if background['group'] in groups
            for sequence in background['sequences']]


def indel_excluded_support(panel: dict, plan: dict, declaration: dict,
                           groups: set) -> tuple[dict, dict]:
    """The panel restricted to the cycles that survive the declared indel exclusion.

    An insertion or deletion construct carries an ``aa_seq`` truncated to the wild
    type's length in the pinned measurement files, so such a row is length-matched
    to the substitution states and entered the frozen cohort as though it were a
    substitution. The declaration names every cohort state that rests on one and
    gives the median of its remaining accepted rows, or ``None`` where no row
    remains.

    Restriction happens before any fit sees the panel: a state with no remaining
    row is dropped together with every cycle that indexes it, a state whose
    remaining rows give a different median carries that value, and every retained
    cycle's target is recomputed from its four corrected states. The returned
    panel therefore carries neither the contaminated rows nor the cycles that rest
    on them on the fitting side or the evaluation side.

    The restriction keeps the frozen extraction plan and its archived scores: the
    retained states are a subset of the frozen ones, so no sequence enters that no
    archive covers. It draws no replacement cycle for a dropped one, so three
    backgrounds retain fewer cycles than the cohort's per-background cap.
    """

    if declaration.get('schema') != 'pairwise_indel_exclusion_v1':
        raise ValueError('unexpected indel-exclusion declaration schema')
    if 'states' not in declaration:
        raise ValueError('the indel-exclusion declaration must be the state-level shape '
                         "(key 'states'); found keys " + str(sorted(declaration)))
    keys = panel_state_keys(plan, groups)
    y = panel['states']['y']
    if len(keys) != len(y):
        raise ValueError('state rows are not aligned to the plan')
    cycle_states_index = panel['cycle_states']
    wild, low, high, double = (cycle_states_index[:, index] for index in range(4))
    moved = float(np.max(np.abs(
        y[double] - y[low] - y[high] + y[wild] - panel['epsilon'])))
    if moved > CYCLE_IDENTITY_TOLERANCE:
        raise ValueError(f'panel cycles disagree with their own states by {moved} kcal/mol')

    absent, corrected = set(), {}
    for state in declaration['states']:
        key = (state['background'], state['sequence'])
        value = state['value_without_indel_rows_kcal_mol']
        if value is None:
            absent.add(key)
        else:
            corrected[key] = float(value)
    corrected_y = y.copy()
    keep_state = np.ones(len(keys), dtype=bool)
    for index, key in enumerate(keys):
        if key in absent:
            keep_state[index] = False
        elif key in corrected:
            corrected_y[index] = corrected[key]
    if len(absent) and not (~keep_state).any():
        raise ValueError('the declaration names absent states that the panel does not carry')
    order = np.cumsum(keep_state) - 1
    keep_row = keep_state[cycle_states_index].all(axis=1)
    epsilon = corrected_y[double] - corrected_y[low] - corrected_y[high] + corrected_y[wild]

    rows = len(keep_row)
    carried, dropped_blocks = {}, []
    for name, block in panel['blocks'].items():
        if len(block) == rows:
            carried[name] = block[keep_row]
        else:
            dropped_blocks.append(name)
    filtered = {key: panel[key][keep_row] for key in
                ('group', 'site_pair', 'background', 'separation', 'cycle_index')}
    filtered['epsilon'] = epsilon[keep_row]
    filtered['blocks'] = carried
    filtered['cycle_states'] = order[cycle_states_index[keep_row]]
    filtered['independent_site_max_absolute'] = panel['independent_site_max_absolute']
    filtered['states'] = {
        'features': panel['states']['features'][keep_state],
        'y': corrected_y[keep_state],
        'group': panel['states']['group'][keep_state],
        'background': panel['states']['background'][keep_state],
    }
    filtered['states']['weight'] = row_weights(filtered['states']['background'],
                                               filtered['states']['group'])
    shifted = np.flatnonzero(keep_row & (epsilon != panel['epsilon']))
    accounting = {
        'declaration_states': len(declaration['states']),
        'states_removed': int((~keep_state).sum()),
        'states_corrected': sum(1 for key in keys if key in corrected),
        'cycles_dropped': int((~keep_row).sum()),
        'cycles_with_a_moved_target': int(len(shifted)),
        'largest_target_move_kcal_mol': float(
            np.max(np.abs(epsilon[shifted] - panel['epsilon'][shifted]))) if len(shifted) else 0.0,
        'site_pairs_dropped': sorted(set(panel['site_pair'][~keep_row])
                                     - set(panel['site_pair'][keep_row])),
        'groups_dropped': sorted(set(panel['group'][~keep_row]) - set(panel['group'][keep_row])),
        'blocks_not_carried': dropped_blocks,
        'retained': {'cycles': int(keep_row.sum()),
                     'groups': len(set(filtered['group'])),
                     'site_pairs': len(set(filtered['site_pair'])),
                     'states': int(keep_state.sum())},
    }
    return filtered, accounting
