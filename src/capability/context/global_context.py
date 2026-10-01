"""Long-range but additive sequence context: the gate's declared control block,
its bounded-receptive-field comparator and the nested held-group comparison that
re-estimates the model increments over them.

The gate this module implements asks one question about the flagship pairwise
result in ``archive/docs/R3/pairwise-epistasis-results.md``: can the residual model-side
information on the wild-type-centred cycle be explained by longer-range but still
primarily additive sequence context, rather than by pairwise interaction? The
existing controls cannot answer it. The sequence/profile control C carries
mutation identities, positions, whole-sequence composition, a whole-sequence
dipeptide difference and mutation-local profile summaries; the nuisance control G
applies one global monotone response to an additive prediction. Neither
represents a per-position effect that depends on distant sequence content.

Two blocks are declared here, both built from per-site vectors summed over the
cycle's two mutated sites:

``W``  the bounded-receptive-field comparator. Every coordinate reads only
       residues within :data:`BOUNDED_RADIUS` of its own mutated site.
``X``  the global-context block. It carries ``W`` as its leading coordinates and
       adds the sequence content outside the mutation-local window at four
       declared ranges, the full-background rung of the window ladder, the
       distant composition, the size of each outside set and the distances to
       both termini.

Additive by construction, and this is the condition that makes the gate
meaningful. :func:`site_context_features` receives one site, that site's own
substituted residue and the **wild-type** background. It cannot read the other
mutated site's index or its substitution, so no coordinate of either block is a
function of both sites. :func:`cycle_context_features` is the plain sum of the
two per-site vectors, so a linear readout of either block equals a sum of two
per-site terms with no cross-site term, at any range. Computing the context on
the wild-type background rather than on the double mutant is what buys that: a
context read off the double mutant would carry the other site's substitution and
would be a cross-site term.

The comparison keeps every convention of the flagship fit rather than restating
it: the weights, folds, ridge recipe, penalty grid, tie rule, nonlinear-additive
nuisance, group-equal squared error and group bootstrap are all imported from
:mod:`~.pairwise_epistasis` and :mod:`~.readout_analysis`. The independent unit
is the site pair, never the cycle count.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..interactions import pairwise_epistasis as flagship
from ..core.amino_acids import AA20
from ..core.protein_properties import PROPERTY_BASIS
from ..interactions.pairwise_epistasis import (
    ALPHAS, C_BLOCKS, INNER_SPLITS, OUTER_SPLITS, group_errors, group_spearman,
    interval, nuisance_cycle)
from ..readouts.readout_analysis import family_folds, ridge_predict, row_weights

#: Residue-level property scales, taken from the declared three-property basis
#: :data:`~..core.protein_properties.PROPERTY_BASIS` (Kyte-Doolittle hydropathy, formal
#: charge at pH 7, side-chain van der Waals volume in cubic angstroms) rather
#: than re-tabulated here. Three axes, declared before this gate was fitted;
#: a fourth chosen after seeing a result would be selection.
SCALES: tuple[str, ...] = ('hydropathy', 'charge', 'volume')

#: Window radii in residues for the sets *inside* the mutation-local window,
#: measured from the mutated site itself and excluding it.
INSIDE_RADII: tuple[int, ...] = (1, 2, 5)

#: The bounded receptive field of the comparator block ``W``, in residues. Every
#: ``W`` coordinate reads only residues at sequence distance at most this from
#: its own mutated site.
BOUNDED_RADIUS: int = max(INSIDE_RADII)

#: Radii whose *complements* give the distant sets: residues at sequence
#: distance strictly greater than the radius from the mutated site. The ladder
#: runs from just outside the local window to 20 residues away, against cohort
#: lengths of 37 to 72 residues.
OUTSIDE_RADII: tuple[int, ...] = (2, 5, 10, 20)

#: Amino-acid composition is reported for one inside set and one outside set,
#: at the radius that separates the mutation-local window from the background.
COMPOSITION_RADIUS: int = 5


def _set_names() -> tuple[tuple[str, ...], tuple[str, ...]]:
    bounded = tuple(f'in{radius}' for radius in INSIDE_RADII)
    distant = ('in_all',) + tuple(f'out{radius}' for radius in OUTSIDE_RADII)
    return bounded, distant


def _feature_order() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Coordinate names of ``X``, and of ``W`` as its leading coordinates."""

    bounded_sets, distant_sets = _set_names()
    bounded = (
        tuple(f'mean_{scale}_{name}' for name in bounded_sets for scale in SCALES)
        + tuple(f'delta_{scale}_x_mean_{scale}_{name}'
                for name in bounded_sets for scale in SCALES)
        + tuple(f'composition_in{COMPOSITION_RADIUS}_{residue}' for residue in AA20))
    distant = (
        tuple(f'mean_{scale}_{name}' for name in distant_sets for scale in SCALES)
        + tuple(f'delta_{scale}_x_mean_{scale}_{name}'
                for name in distant_sets for scale in SCALES)
        + tuple(f'composition_out{COMPOSITION_RADIUS}_{residue}' for residue in AA20)
        + tuple(f'size_fraction_out{radius}' for radius in OUTSIDE_RADII)
        + ('residues_to_n_terminus', 'residues_to_c_terminus', 'residues_to_nearest_terminus'))
    return bounded + distant, bounded


#: Ordered coordinate names of the global-context block ``X`` and of the
#: bounded-receptive-field comparator ``W``. ``W`` is exactly the leading
#: coordinates of ``X``, so the column space of ``W`` is contained in that of
#: ``X`` and the two context steps of the ladder are genuinely nested.
GLOBAL_FEATURE_ORDER, BOUNDED_FEATURE_ORDER = _feature_order()

#: Positions of the ``W`` coordinates inside an ``X`` row.
BOUNDED_INDEX: tuple[int, ...] = tuple(range(len(BOUNDED_FEATURE_ORDER)))

_CODE = {residue: index for index, residue in enumerate(AA20)}


def _scale_matrix(sequence: str) -> np.ndarray:
    """``(len(SCALES), len(sequence))`` residue-level property values."""

    try:
        return np.asarray([[PROPERTY_BASIS[scale][residue] for residue in sequence]
                           for scale in SCALES], dtype=float)
    except KeyError as error:
        raise ValueError(f'noncanonical residue {error.args[0]!r} in a cohort sequence') from error


def site_context_features(wildtype: str, site: int, mutant: str) -> np.ndarray:
    """Per-site context vector in :data:`GLOBAL_FEATURE_ORDER`.

    Every coordinate is a function of the wild-type background, this one site and
    this one site's substituted residue. The signature cannot express a
    dependence on another mutated site, and the context is read off the wild-type
    background rather than a mutant state, so no coordinate returned here carries
    a second substitution.
    """

    length = len(wildtype)
    if not 0 <= site < length:
        raise ValueError(f'site {site} is outside a background of {length} residues')
    if mutant not in _CODE or wildtype[site] not in _CODE:
        raise ValueError('noncanonical residue at a mutated site')
    if mutant == wildtype[site]:
        raise ValueError('a mutated site must carry a substitution')

    scales = _scale_matrix(wildtype)
    delta = np.asarray([PROPERTY_BASIS[scale][mutant] - PROPERTY_BASIS[scale][wildtype[site]]
                        for scale in SCALES], dtype=float)
    offset = np.abs(np.arange(length) - site)
    bounded_masks = [(offset > 0) & (offset <= radius) for radius in INSIDE_RADII]
    distant_masks = [offset > 0] + [offset > radius for radius in OUTSIDE_RADII]

    def means(masks: list[np.ndarray]) -> np.ndarray:
        """Set means per scale; an empty set contributes zeros, and the outside
        size fractions below carry its emptiness rather than leaving it implied."""

        table = np.zeros((len(masks), len(SCALES)))
        for index, mask in enumerate(masks):
            if mask.any():
                table[index] = scales[:, mask].mean(axis=1)
        return table

    def composition(mask: np.ndarray) -> np.ndarray:
        codes = np.asarray([_CODE[residue] for residue in wildtype])
        return np.bincount(codes[mask], minlength=len(AA20)) / length

    bounded_means = means(bounded_masks)
    distant_means = means(distant_masks)
    local = offset <= COMPOSITION_RADIUS
    values = np.concatenate([
        bounded_means.ravel(),
        (bounded_means * delta[None, :]).ravel(),
        composition((offset > 0) & local),
        distant_means.ravel(),
        (distant_means * delta[None, :]).ravel(),
        composition(~local),
        np.asarray([float(mask.sum()) / length for mask in distant_masks[1:]]),
        np.asarray([float(site), float(length - 1 - site), float(min(site, length - 1 - site))]),
    ])
    if values.shape != (len(GLOBAL_FEATURE_ORDER),):
        raise ValueError('site context vector does not match the declared feature order')
    return values


def cycle_context_features(wildtype: str, states: tuple[str, str, str, str],
                           positions: tuple[int, int]) -> np.ndarray:
    """The global-context block ``X`` for one cycle: the sum of the two per-site
    vectors, with no term that reads both sites.

    ``states`` is the canonical quartet :func:`~.pairwise_epistasis.cycle_states`
    emits and ``positions`` its two one-based mutated positions, exactly as
    :func:`~.pairwise_epistasis.cycle_control_features` receives them.
    """

    wild, low_state, high_state, double = states
    if wild != wildtype:
        raise ValueError('cycle wild-type state is not the background wild type')
    low, high = positions[0] - 1, positions[1] - 1
    if not low < high:
        raise ValueError('cycle positions must be ordered and distinct')
    if double[low] != low_state[low] or double[high] != high_state[high]:
        raise ValueError('double state does not carry both single substitutions')
    return (site_context_features(wildtype, low, low_state[low])
            + site_context_features(wildtype, high, high_state[high]))


def context_blocks(wildtype: str, states: tuple[str, str, str, str],
                   positions: tuple[int, int]) -> dict[str, np.ndarray]:
    """Both declared blocks for one cycle. ``W`` is a subset of ``X``'s columns."""

    values = cycle_context_features(wildtype, states, positions)
    return {'X': values, 'W': values[list(BOUNDED_INDEX)]}


#: What was declared before any of this gate's fits ran. The digest of this
#: record is stated in ``archive/docs/R5/gate-global-context.md`` and is written into
#: every fit record, so a reader can check that the block fitted is the block
#: declared.
#:
#: **These bytes are frozen.** :func:`declaration_digest` hashes this dict, so
#: every character inside it is part of a content-addressed identity that was
#: fixed at 2026-09-24T05:15Z and written into 33 fit records. Renaming a module,
#: dropping a schema-version suffix or any other mechanical edit must leave this
#: dict untouched, even when the names it quotes no longer exist: the record
#: names what was declared then, not what the tree looks like now. Two such
#: edits did reach it once and were recovered from a frozen pre-rename package.
#: ``scale_source`` below therefore still names the pre-rename module path; the
#: scales are read today from :data:`~..core.protein_properties.PROPERTY_BASIS`,
#: which is the same table under its current name.
BLOCK_DECLARATION: dict = {
    'schema': 'global_context_gate_declaration_v1',
    'question': ('can residual model-side information on the wild-type-centred cycle be '
                 'explained by longer-range but still primarily additive sequence context, '
                 'rather than by pairwise interaction'),
    'scales': {scale: dict(PROPERTY_BASIS[scale]) for scale in SCALES},
    'scale_source': ('src.transfer.concept_lens.PROPERTY_BASIS: Kyte-Doolittle hydropathy, '
                     'formal side-chain charge at pH 7, side-chain van der Waals volume in '
                     'cubic angstroms'),
    'inside_radii_residues': list(INSIDE_RADII),
    'outside_radii_residues': list(OUTSIDE_RADII),
    'bounded_radius_residues': BOUNDED_RADIUS,
    'composition_radius_residues': COMPOSITION_RADIUS,
    'blocks': {
        'W': {'width': len(BOUNDED_FEATURE_ORDER), 'features': list(BOUNDED_FEATURE_ORDER),
              'receptive_field': (f'residues at sequence distance at most {BOUNDED_RADIUS} '
                                  'from the coordinate\'s own mutated site')},
        'X': {'width': len(GLOBAL_FEATURE_ORDER), 'features': list(GLOBAL_FEATURE_ORDER),
              'receptive_field': 'the whole wild-type background, per mutated site'},
    },
    'additive_by_construction': (
        'every coordinate of both blocks is a per-site function of the wild-type background, '
        'one mutated site and that site\'s own substituted residue, summed over the cycle\'s '
        'two mutated sites; no coordinate is a function of both sites, so a linear readout of '
        'either block is a sum of two per-site terms with no cross-site interaction term at '
        'any range'),
    'context_state': ('the wild-type background, never a mutant state, so one site\'s context '
                      'cannot carry the other site\'s substitution'),
}


def declaration_digest() -> str:
    """Content digest over :data:`BLOCK_DECLARATION`."""

    return hashlib.sha256(json.dumps(
        BLOCK_DECLARATION, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# The gate ladder. Every control set below extends the flagship's own matched
# baseline C+G+T, so each increment composes with the numbers that baseline
# already carries.
# --------------------------------------------------------------------------- #

#: ``CGT`` is the flagship matched baseline, ``CGTW`` adds the bounded-receptive
#: -field comparator and ``CGTX`` the global-context block; the three ``_M1``
#: sets add the two constituent first-order likelihood differences to each.
GATE_CONTROL_SETS: dict[str, tuple[str, ...]] = {
    'CGT': (*C_BLOCKS, 'G', 'T'),
    'CGTW': (*C_BLOCKS, 'G', 'T', 'W'),
    'CGTX': (*C_BLOCKS, 'G', 'T', 'X'),
    'CGT_M1': (*C_BLOCKS, 'G', 'T', 'M1'),
    'CGTW_M1': (*C_BLOCKS, 'G', 'T', 'W', 'M1'),
    'CGTX_M1': (*C_BLOCKS, 'G', 'T', 'X', 'M1'),
}

#: Only the likelihood interaction contrast is added over these controls. The
#: projected representation contrast resolved above zero for no arm of 33 in the
#: flagship fit and is outside this gate's scope.
GATE_ADDITIONS: dict[str, tuple[str, ...]] = {'': (), 'M': ('M',)}

GATE_DESIGNS: tuple[str, ...] = tuple(
    f'{control}+{addition}' if addition else control
    for control in GATE_CONTROL_SETS for addition in GATE_ADDITIONS)

#: Declared contrasts, in three families: what each context block adds on its
#: own, the first-order likelihood gain over each context baseline, and the
#: likelihood interaction gain over each context baseline with and without the
#: first-order control in it.
GATE_CONTRASTS: tuple[tuple[str, str], ...] = (
    ('CGT', 'CGTW'), ('CGT', 'CGTX'), ('CGTW', 'CGTX'),
    ('CGT', 'CGT_M1'), ('CGTW', 'CGTW_M1'), ('CGTX', 'CGTX_M1'),
    ('CGT', 'CGT+M'), ('CGTW', 'CGTW+M'), ('CGTX', 'CGTX+M'),
    ('CGT_M1', 'CGT_M1+M'), ('CGTW_M1', 'CGTW_M1+M'), ('CGTX_M1', 'CGTX_M1+M'),
)

#: Control sets whose held-group reconstruction of the model's own first-order
#: likelihood differences is reported as a diagnostic on the model-approximation
#: estimand, which is not the phenotype estimand the increments above measure.
RECONSTRUCTION_CONTROLS: tuple[str, ...] = ('CGT', 'CGTW', 'CGTX')

#: Every block this gate reads, excluding the nuisance ``G``, which is refitted
#: inside each outer fold rather than carried on the panel. The admitted fit
#: script also assembles ``Q``, ``R`` and ``R1``; ``Q`` is only defined on the
#: backgrounds carrying a fitted pairwise model, so it is shorter than the panel
#: on the 64-group support and no gate design reads it.

STRATA: tuple[str, ...] = ('1-2', '3-9', '10+')


def design_blocks(name: str) -> tuple[str, ...]:
    control, _, addition = name.partition('+')
    return (*GATE_CONTROL_SETS[control], *GATE_ADDITIONS[addition])


GATE_BLOCKS: tuple[str, ...] = tuple(sorted(
    {block for name in GATE_DESIGNS for block in design_blocks(name) if block != 'G'}
    | {'M1'}))


def _tuned_prediction(x: np.ndarray, y: np.ndarray, weights: np.ndarray, groups: np.ndarray,
                      train: np.ndarray, test: np.ndarray, inner_partition: list,
                      device) -> tuple[np.ndarray, float]:
    """The admitted ridge recipe on one outer fold: penalty selected on the inner
    held-group partition, ties broken toward stronger regularisation, feature
    centring and scaling fitted on training rows inside
    :func:`~.readout_analysis.ridge_predict`. No held-out row reaches the
    selection or the scaling."""

    losses = np.zeros(len(ALPHAS))
    for validation in inner_partition:
        fit = train[~np.isin(groups[train], validation)]
        valid = train[np.isin(groups[train], validation)]
        if not len(fit) or not len(valid):
            raise ValueError('inner fold has an empty side')
        predicted = ridge_predict(x[fit], y[fit], weights[fit], x[valid], ALPHAS, device)
        losses += len(validation) * (
            weights[valid][:, None] * (predicted - y[valid][:, None]) ** 2).sum(0)
    alpha = ALPHAS[flagship._select_alpha(losses / len(set(groups[train])))]
    return ridge_predict(x[train], y[train], weights[train], x[test], [alpha], device)[:, 0], alpha


def gate_compare(panel: dict, *, seed: int, device='cpu') -> dict:
    """Held-group predictions for every gate design on one support and one seed.

    The rows, the outer and inner held-group partitions, the row weights, the
    nonlinear-additive nuisance and the ridge recipe are the flagship fit's, so a
    design this gate shares with that fit reproduces its held-out predictions
    exactly and every increment reported here composes with the ones it reports.
    """

    groups = panel['group']
    weights = row_weights(panel['site_pair'], groups)
    predictions = {name: np.full(len(groups), np.nan) for name in GATE_DESIGNS}
    reconstruction = {f'{control}|m1_{index}': np.full(len(groups), np.nan)
                      for control in RECONSTRUCTION_CONTROLS for index in (0, 1)}
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
        fold = {'fold': outer, 'held_groups': list(held), 'alpha': {}, 'dimensions': {}}
        for name in GATE_DESIGNS:
            x = np.column_stack([blocks[key] for key in design_blocks(name)])
            if not np.isfinite(x).all():
                raise ValueError(f'{name}: nonfinite design matrix')
            predictions[name][test], alpha = _tuned_prediction(
                x, panel['epsilon'], weights, groups, train, test, inner_partition, device)
            fold['alpha'][name] = alpha
            fold['dimensions'][name] = int(x.shape[1])
        for control in RECONSTRUCTION_CONTROLS:
            x = np.column_stack([blocks[key] for key in GATE_CONTROL_SETS[control]])
            for index in (0, 1):
                target = panel['blocks']['M1'][:, index]
                value, alpha = _tuned_prediction(x, target, weights, groups, train, test,
                                                 inner_partition, device)
                reconstruction[f'{control}|m1_{index}'][test] = value
                fold['alpha'][f'reconstruct_{control}|m1_{index}'] = alpha
        records.append(fold)
    for name, value in {**predictions, **reconstruction}.items():
        if not np.isfinite(value).all():
            raise ValueError(f'{name}: incomplete held-out predictions')
    return {'predictions': predictions, 'reconstruction': reconstruction,
            'folds': records, 'nuisance': nuisance, 'row_weights': weights}


def _pair_equal_mean(values: np.ndarray, site_pairs: np.ndarray) -> float:
    return float(np.mean([values[site_pairs == pair].mean() for pair in sorted(set(site_pairs))]))


def group_r2(target: np.ndarray, prediction: np.ndarray, groups: np.ndarray,
             site_pairs: np.ndarray) -> tuple[np.ndarray, list]:
    """Per-group held-out weighted R2, site pairs weighted equally inside a group.

    The denominator is the same site-pair-equal squared error about the group's
    own site-pair-equal mean, so this is the fraction of within-group target
    spread the held-group prediction accounts for. A group with no spread carries
    ``None`` rather than a manufactured value.
    """

    labels, residual = group_errors(target, prediction, groups, site_pairs)
    values: list = []
    for label, error in zip(labels, residual):
        rows = np.flatnonzero(groups == label)
        centre = _pair_equal_mean(target[rows], site_pairs[rows])
        _, spread = group_errors(target[rows], np.full(len(rows), centre),
                                 groups[rows], site_pairs[rows])
        values.append(None if spread[0] <= 0 else float(1.0 - error / spread[0]))
    return labels, values


def evaluate_gate(panel: dict, outcome: dict, *, draws: int, seed: int) -> dict:
    """Group-equal squared error, the declared paired increments with their
    separation strata, and the first-order reconstruction diagnostic."""

    epsilon, groups, pairs = panel['epsilon'], panel['group'], panel['site_pair']
    per_group_mse, per_group_spearman, designs = {}, {}, {}
    for name, prediction in outcome['predictions'].items():
        labels, values = group_errors(epsilon, prediction, groups, pairs)
        per_group_mse[name] = dict(zip(labels.tolist(), values.tolist()))
        _, spearman = group_spearman(epsilon, prediction, groups)
        per_group_spearman[name] = spearman
        designs[name] = {
            'group_equal_mse_kcal2': interval(values, draws=draws, seed=seed),
            'within_background_spearman': interval(spearman, draws=draws, seed=seed),
        }
    labels = sorted(set(groups))
    increments = {}
    for base, augmented in GATE_CONTRASTS:
        difference = [per_group_mse[base][label] - per_group_mse[augmented][label]
                      for label in labels]
        record = {'mse_reduction_kcal2': interval(difference, draws=draws, seed=seed)}
        left, right = per_group_spearman[base], per_group_spearman[augmented]
        paired = [None if a is None or b is None else b - a for a, b in zip(left, right)]
        record['spearman_increment'] = interval(paired, draws=draws, seed=seed)
        record['strata'] = {}
        for stratum in STRATA:
            rows = np.flatnonzero(panel['separation'] == stratum)
            if not len(rows):
                continue
            stratum_labels, base_values = group_errors(
                epsilon[rows], outcome['predictions'][base][rows], groups[rows], pairs[rows])
            _, augmented_values = group_errors(
                epsilon[rows], outcome['predictions'][augmented][rows], groups[rows], pairs[rows])
            record['strata'][stratum] = {
                'groups': len(stratum_labels), 'cycles': int(len(rows)),
                'site_pairs': len(set(pairs[rows])),
                'mse_reduction_kcal2': interval(base_values - augmented_values,
                                                draws=draws, seed=seed)}
        increments[f'{augmented}|{base}'] = record
    reconstruction = {}
    for key, prediction in outcome['reconstruction'].items():
        control, _, target_name = key.partition('|')
        target = panel['blocks']['M1'][:, int(target_name.rsplit('_', 1)[1])]
        _, values = group_r2(target, prediction, groups, pairs)
        _, ranks = group_spearman(target, prediction, groups)
        reconstruction[key] = {
            'group_equal_weighted_r2': interval(values, draws=draws, seed=seed),
            'within_background_spearman': interval(ranks, draws=draws, seed=seed),
            'target': ('the model first-order likelihood difference of the lower-site single '
                       'state, in nats' if target_name.endswith('0') else
                       'the model first-order likelihood difference of the higher-site single '
                       'state, in nats'),
            'estimand': ('model-score approximation on held groups, not phenotype prediction'),
        }
    return {'designs': designs, 'increments': increments, 'reconstruction': reconstruction,
            'per_group_mse': per_group_mse, 'per_group_spearman': per_group_spearman}


# --------------------------------------------------------------------------- #
# The insertion/deletion-construct exclusion, applied as a declared filter over
# the frozen cohort rather than as a new cohort.
# --------------------------------------------------------------------------- #

#: Largest admissible disagreement, in kcal/mol, between a panel's own cycle and
#: the same cycle recomputed from its four absolute-stability states. The two are
#: the same arithmetic in a different order, so anything above this means the
#: state rows were not the ones the cycle indexes.
CYCLE_IDENTITY_TOLERANCE = 1e-9


def state_keys(plan: dict, groups: set) -> list[tuple[str, str]]:
    """``(background, sequence)`` of every absolute-stability state row, in the
    order the admitted fit script appends them."""

    return [(background['name'], sequence)
            for background in plan['backgrounds'] if background['group'] in groups
            for sequence in background['sequences']]


def apply_indel_exclusion(panel: dict, plan: dict, declaration: dict,
                          groups: set) -> tuple[dict, dict]:
    """A filtered copy of a panel with insertion/deletion constructs excluded.

    An indel construct's sequence is truncated to the wild type's length in the
    pinned files, so it entered the frozen cohort as though it were a
    substitution. The frozen cohort is not re-built here — several gates
    reference its digest as their declared support — so the declared exclusion is
    applied to the assembled panel instead: a state whose rows are all indel
    constructs is removed and every cycle that uses it is dropped, a state whose
    remaining rows give a different median carries that value, and every cycle's
    target is recomputed from the four corrected states.

    The returned panel has different rows from the frozen support and therefore a
    different row identity and different folds where a whole group changes; it is
    a declared sensitivity beside the unfiltered fit, never a replacement for it.
    """

    if declaration.get('schema') != 'pairwise_indel_exclusion_v1':
        raise ValueError('unexpected indel-exclusion declaration schema')
    if 'states' not in declaration:
        raise ValueError('the indel-exclusion declaration must be the state-level shape '
                         "(key 'states'); found keys " + str(sorted(declaration)))
    keys = state_keys(plan, groups)
    y = panel['states']['y']
    if len(keys) != len(y):
        raise ValueError('state rows are not aligned to the plan')
    cycle_states = panel['cycle_states']
    wild, low, high, double = (cycle_states[:, index] for index in range(4))
    recomputed = y[double] - y[low] - y[high] + y[wild]
    moved = float(np.max(np.abs(recomputed - panel['epsilon'])))
    if moved > CYCLE_IDENTITY_TOLERANCE:
        raise ValueError(f'panel cycles disagree with their own states by {moved} kcal/mol')

    absent, corrected = set(), {}
    for state in declaration['states']:
        key = (state['background'], state['sequence'])
        if state['value_without_indel_rows_kcal_mol'] is None:
            absent.add(key)
        else:
            corrected[key] = float(state['value_without_indel_rows_kcal_mol'])
    corrected_y = y.copy()
    keep_state = np.ones(len(keys), dtype=bool)
    for index, key in enumerate(keys):
        if key in absent:
            keep_state[index] = False
        elif key in corrected:
            corrected_y[index] = corrected[key]
    order = np.cumsum(keep_state) - 1
    keep_row = keep_state[cycle_states].all(axis=1)
    epsilon = (corrected_y[double] - corrected_y[low]
               - corrected_y[high] + corrected_y[wild])

    filtered = {key: value[keep_row] for key, value in panel.items()
                if key in ('group', 'site_pair', 'background', 'separation')}
    filtered['epsilon'] = epsilon[keep_row]
    for name in GATE_BLOCKS:
        if len(panel['blocks'][name]) != len(keep_row):
            raise ValueError(f'{name}: block rows are not aligned to the panel rows')
    filtered['blocks'] = {name: panel['blocks'][name][keep_row] for name in GATE_BLOCKS}
    filtered['cycle_states'] = order[cycle_states[keep_row]]
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
        'by_stratum': {stratum: {
            'cycles_dropped': int((~keep_row & (panel['separation'] == stratum)).sum()),
            'cycles_with_a_moved_target': int(
                len(np.intersect1d(shifted,
                                   np.flatnonzero(panel['separation'] == stratum)))),
        } for stratum in STRATA},
        'blocks_not_carried': sorted(set(panel['blocks']) - set(GATE_BLOCKS)),
        'retained': {'cycles': int(keep_row.sum()),
                     'groups': len(set(filtered['group'])),
                     'site_pairs': len(set(filtered['site_pair'])),
                     'states': int(keep_state.sum())},
    }
    return filtered, accounting
