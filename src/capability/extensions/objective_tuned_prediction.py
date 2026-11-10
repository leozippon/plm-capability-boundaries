"""The prediction side-effect of objective tuning, declared before any tuned arm is scored.

``docs/objective-alignment.md`` (third amendment of 2026-11-10) is the registration; this
module is its executable declaration and states nothing the amendment does not. The
question is whether optimising a design proxy costs, keeps or adds predictive information
*relative to explicit evolutionary statistics* on the R1 mutation-effect cohort. It is the
one place the objective-alignment experiment tests the claim's first half, that the memory
carries function-discriminative information, instead of its second half.

The estimand
------------
Per admitted pair, on one matched support, against the control set Profile + Potts + GEMME
(the published ladder's ``P+E+G``), two increments in within-assay Spearman rho:

``forward``  the arm's gain over the controls alone, rho(M+controls) - rho(controls);
``reverse``  the controls' gain over the arm alone, rho(M+controls) - rho(M).

The reported quantity is the tuned-minus-base *difference* in each. Both directions are
needed because a forward increment alone cannot separate shared information from model
inferiority; an overlap reading needs both small.

Nothing is refitted here. The ladder's own fitter, folds, aggregation (seeds within assay,
assays within family, equal families) and ``ladder_matrix`` are reused, so a tuned arm and
the published rows it sits beside differ only in the scalar ``M``.

The family
----------
Admitted pairs crossed with the two directions, read under one simultaneous maximum
absolute centered-deviation band in native correlation units, the construction
:func:`bidirectional.paired_bands` already provides. Here the matrix is genuinely
groups-by-columns (base and tuned score the same variants), so no substitute estimand is
needed, and the resampling unit is the biological family. The family is separate from the
primary pLDDT family and from the published ladder's 396-hypothesis family and is never
merged with either; no tuned arm joins the published ladder and no published verdict is
recomputed. A base sibling that is already a ladder arm contributes its archived scalar.

Support
-------
Complete-case support is defined per stratum. This endpoint has one stratum, the
within-assay ranking, so every arm and every control column is fitted on the identical row
set, and the band refuses a matrix whose columns were fitted on different ones.

The frozen admission list
-------------------------
Which pairs are scored is fixed in ``configs/objective_tuned_prediction.json`` before the
first score and is bound to the measured cost estimate that justified it by digest. The
union of admitted and not-measured pairs must equal the declared pairs exactly, so a pair
cannot be dropped silently once numbers arrive, and a pair left out takes no part in the
family or its band.

The narrowing reading (fourth amendment)
----------------------------------------
Each interval is reported in native correlation units and as a fraction of the control set's
own performance on the identical support. Narrowing is a bounded negative with that stated
size, never a non-rejection; an interval containing zero that is too wide to exclude a change
as large as the controls' own performance is unresolved. A reverse-only rise is its own,
strongest, form of narrowing. There are five readings in all.

Instrument bounds
-----------------
The coupling fitter spreads about 1.07e-2 per variant from run to run on a deep background,
and a fit stopped at its iteration cap moves a score about four times that, which is why a
non-converged background is booked incomplete. Those bounds are in score units; the
declaration carries their measured propagation into correlation units, and a column whose
point estimate is smaller than that floor is never reported as resolved.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import bidirectional, explicit_statistics_lanes as lanes, statistical_ladder as ladder
from .objective_tuned_bounds import CAPPED_FIT_FACTOR, COUPLING_RUN_SPREAD
from ..core import statistics
from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS, bootstrap_unit_floor
from ..generation import objective_tuned_arms as tuned_arms

ROOT = Path(__file__).resolve().parents[3]
DECLARATION_PATH = ROOT / 'configs/objective_tuned_prediction.json'

ENDPOINT = 'objective_tuned_prediction_20261110'
COHORT = 'r1'
CONTROL_SET = 'P+E+G'
METRIC = 'spearman'
#: The one stratum: within-assay ranking. The amendment's estimand is a rank correlation
#: and names no other cell, so there is nothing for support to be matched across.
STRATUM = 'ranking'

FORWARD, REVERSE = 'forward', 'reverse'
#: The amendment's two directions, bound to the ladder's own direction names.
DIRECTIONS = {FORWARD: 'model_given_controls', REVERSE: 'controls_given_model'}

#: The control blocks of ``P+E+G`` by role, so the set is named by what it holds and not
#: by letters that mean other things in other lanes.
CONTROL_ROLES = ('profile', 'couplings', 'gemme')

STATES = statistics.FINDING_STATES
SUPPORTED, BOUNDED_NEGATIVE, UNRESOLVED, NON_IDENTIFIABLE = STATES
#: A pair left out is not a finding state; it takes no part in the family.
NOT_MEASURED = 'not_measured'

#: The four finding states as this endpoint reads them. They describe the sign of a
#: tuned-minus-base *difference*, not of an increment.
DISPOSITION = {
    SUPPORTED: 'the simultaneous band lies above zero and the point clears the instrument floor',
    BOUNDED_NEGATIVE: ('a resolved absence at a stated size: the band lies below zero and the '
                       'point clears the floor, or the band contains zero (or the point is '
                       'under the floor) and still admits less than the controls\' own '
                       'performance, reported as that fraction'),
    UNRESOLVED: ('the band contains zero (or the point is under the floor) and is wide enough '
                 'to admit a change as large as the controls\' own performance; never '
                 'equivalence'),
    NON_IDENTIFIABLE: ('declared and unused: the single scalar block and the complete controls '
                       'make every column identifiable, so a column that cannot be estimated '
                       'raises instead of being booked'),
}

#: The five readings of a pair (fourth amendment). Narrowing is a bounded negative with a
#: stated size, never a non-rejection: an interval that contains zero but is too wide to
#: exclude a change as large as the controls' own performance is ``unresolved``.
ADDED_INFORMATION = 'added_information'
NARROWING_WITH_COST = 'narrowing_with_cost'
NARROWING_BOUNDED = 'narrowing_bounded'
NARROWING_LOSS_OF_OVERLAP = 'narrowing_with_loss_of_overlap'
READING_UNRESOLVED = 'unresolved'
READINGS = {
    ADDED_INFORMATION: ('the forward difference rose beyond the band: predictive information '
                        'the statistics do not contain, which is evidence against the central '
                        'claim and is reported as such'),
    NARROWING_WITH_COST: ('the forward difference fell beyond the band: narrowing with a '
                          'measured price in evolutionary-memory accuracy'),
    NARROWING_BOUNDED: ('neither difference is resolved and both intervals lie wholly within '
                        'the stated fraction of the controls\' own performance, which is '
                        'reported as measured'),
    NARROWING_LOSS_OF_OVERLAP: ('the reverse difference rose while the forward one is bounded: '
                                'the controls explain more of what the tuned arm fails to hold '
                                'than they did of the base, so the memory moved off the explicit '
                                'statistics without adding anything beyond them; the strongest '
                                'form of narrowing and never reported as the bounded case'),
    READING_UNRESOLVED: ('an interval contains zero but is too wide to exclude a change as large '
                         'as the controls\' own performance, or the reverse difference fell '
                         'with the forward one bounded, a cell the amendment declares no '
                         'reading for; reported as unresolved'),
}

BOUNDS = (
    'R1 measures mutation-effect prediction, which is not design. A side-effect on prediction, '
    'in either direction, constrains what the tuning did to the memory; it is not itself '
    'evidence that a model can or cannot design.',
    'No equivalence margin is declared. A bounded negative is stated as the fraction of the '
    'controls\' own performance its interval still admits, measured and never compared with a '
    'threshold chosen afterwards; the only reference is the controls\' performance itself.',
    'Increments are predictive contrasts under the ladder\'s ridge and folds, conditional on '
    'fitted held-out predictions; they are not exclusive biological knowledge.',
)


def load_declaration(path: Path = DECLARATION_PATH) -> dict[str, Any]:
    """Read the frozen declaration and bind it to the two measurements it rests on by digest."""

    declaration = json.loads(Path(path).read_text())
    for key in ('cost_estimate', 'instrument_floor'):
        evidence = ROOT / declaration[key]['path']
        if not evidence.is_file():
            raise RuntimeError(f'the {key} {declaration[key]["path"]} the list was frozen on is missing')
        if hashlib.sha256(evidence.read_bytes()).hexdigest() != declaration[key]['sha256']:
            raise RuntimeError(f'the {key} no longer matches the digest the list was frozen '
                               'against; the declaration is not decided from it')
    return declaration


DECLARATION = load_declaration()
ADMITTED: tuple[str, ...] = tuple(DECLARATION['admitted'])
NOT_MEASURED_REASONS: dict[str, str] = dict(DECLARATION['not_measured'])
INSTRUMENT_FLOOR: float = float(DECLARATION['instrument_floor']['correlation'])


def admitted_pairs() -> tuple[tuned_arms.ObjectiveTunedPair, ...]:
    return tuple(tuned_arms.pair_for(name) for name in ADMITTED)


def family_size() -> int:
    """Admitted pairs crossed with the two directions."""

    return len(ADMITTED) * len(DIRECTIONS)


def labels(pairs: Sequence[tuned_arms.ObjectiveTunedPair] | None = None) -> list[tuple[str, str]]:
    """Column labels, (tuned arm, direction), in declared order."""

    return [(p.tuned, direction) for p in (admitted_pairs() if pairs is None else pairs)
            for direction in DIRECTIONS]


def reuses_archive(base: str) -> bool:
    """A base that is already a published ladder arm contributes its archived scalar."""

    return base in lanes.PANEL_ARMS


# ----------------------------------------------------------------- support


def support_digest(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256('\n'.join(map(str, sample_ids)).encode()).hexdigest()


def matched_support(available: Mapping[str, np.ndarray], controls: np.ndarray) -> dict[str, Any]:
    """The one row set every arm and every control column is fitted on.

    ``available`` maps each arm to a boolean per-row mask of finite scores and ``controls``
    is the per-row mask where Profile, Potts and GEMME are all finite. The intersection is
    the stratum's complete case, with what each arm lost enumerated by name.
    """

    controls = np.asarray(controls, bool)
    masks = {arm: np.asarray(mask, bool) for arm, mask in available.items()}
    if not masks or any(mask.shape != controls.shape for mask in masks.values()):
        raise ValueError('every arm needs a per-row availability mask over the same rows')
    keep = controls.copy()
    for mask in masks.values():
        keep &= mask
    if not keep.any():
        raise ValueError('the matched support is empty')
    return {'mask': keep, 'rows': int(keep.sum()),
            'lost': {arm: int((~mask).sum()) for arm, mask in masks.items()},
            'lost_to_controls': int((~controls).sum())}


# --------------------------------------------------------------- estimand


def difference_matrix(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Families by (tuned arm, direction): tuned increment minus base increment.

    ``rows`` are the ladder's per-assay metric rows, one per (arm, seed, assay) for every
    base and tuned arm, so the ladder's own ``ladder_matrix`` forms the increments and the
    seed, assay and family aggregation is the published one. An incomplete or undefined cell
    raises: the estimand has no partial support to fall back on.
    """

    pairs = admitted_pairs()
    models = [name for p in pairs for name in p.arms]
    increments = ladder.ladder_matrix(list(rows), METRIC, models=models,
                                      control_sets={CONTROL_SET: ladder.CONTROL_SETS[CONTROL_SET]})
    if not np.isfinite(increments['matrix']).all():
        raise ValueError('undefined per-assay correlations; no outcome-driven exclusions')
    column = {label: i for i, label in enumerate(increments['labels'])}
    out = [increments['matrix'][:, column[(p.tuned, CONTROL_SET, name)]]
           - increments['matrix'][:, column[(p.base, CONTROL_SET, name)]]
           for p in pairs for name in DIRECTIONS.values()]
    return {'families': increments['families'], 'matrix': np.column_stack(out),
            'labels': labels(pairs)}


# -------------------------------------------------------------------- band


def column_state(point: float, interval: Sequence[float], controls: float,
                 floor: float | None = None) -> dict[str, Any]:
    """One column's finding state, with the forward-bound fractions it rests on.

    ``controls`` is the control set's own family-mean rho on the identical support. The
    interval is reported in native units and as a fraction of it; ``extent`` is the larger
    absolute end of that fraction. Nothing under the instrument floor is resolved, and an
    interval containing zero is a bounded negative only if ``extent`` is below one, the
    controls' own performance being the one reference the amendment names.
    """

    if controls <= 0:
        raise ValueError('the controls must perform above zero for a fraction of them to mean anything')
    floor = INSTRUMENT_FLOOR if floor is None else floor
    low, high = interval
    extent = max(abs(low), abs(high)) / controls
    rises = low > 0 and point >= floor
    falls = high < 0 and point <= -floor
    state = SUPPORTED if rises else BOUNDED_NEGATIVE if falls or extent < 1 else UNRESOLVED
    return {'state': state, 'rises': rises, 'falls': falls, 'extent_fraction': extent,
            'interval_fraction': [low / controls, high / controls]}


def reading(forward: Mapping[str, Any], reverse: Mapping[str, Any]) -> str:
    """The amendment's interpretation of one pair, from its two columns."""

    if forward['rises']:
        return ADDED_INFORMATION
    if forward['falls']:
        return NARROWING_WITH_COST
    if forward['state'] != BOUNDED_NEGATIVE:
        return READING_UNRESOLVED
    if reverse['rises']:
        return NARROWING_LOSS_OF_OVERLAP
    if reverse['state'] == BOUNDED_NEGATIVE and not reverse['falls']:
        return NARROWING_BOUNDED
    return READING_UNRESOLVED


def band(matrix: np.ndarray, column_labels: Sequence[tuple[str, str]],
         supports: Mapping[str, str], *, controls_performance: float) -> dict[str, Any]:
    """Simultaneous native-unit band and dispositions over the declared family.

    ``supports`` maps every arm, and the key ``'controls'``, to the digest of the rows it
    was fitted on. The band refuses unless they are all one digest, because a nested
    increment computed on shifting rows means nothing. ``controls_performance`` is the
    control set's family-mean rho on that same support.
    """

    matrix = np.asarray(matrix, float)
    expected = labels()
    if list(column_labels) != expected or matrix.ndim != 2 or matrix.shape[1] != len(expected):
        raise ValueError(f'the family is the {len(expected)} declared (arm, direction) columns '
                         'in declared order')
    arms = {name for p in admitted_pairs() for name in p.arms}
    if set(supports) != arms | {'controls'}:
        raise ValueError('a support digest is required for every arm and for the controls')
    if len(set(supports.values())) != 1:
        raise ValueError('unmatched support: the arms and controls were not fitted on the same '
                         'rows, so the increments are not comparable')
    floor = bootstrap_unit_floor(len(matrix))
    if floor['degenerate']:
        return {'status': UNRESOLVED, 'reason': floor['degenerate_reason'],
                'groups': len(matrix), 'family_size': len(expected)}
    native = bidirectional.paired_bands(matrix)
    columns = [{'arm': arm, 'direction': direction, 'point': point, 'interval': interval,
                **column_state(point, interval, controls_performance)}
               for (arm, direction), point, interval
               in zip(column_labels, native['point'], native['interval'])]
    by = {(c['arm'], c['direction']): c for c in columns}
    pairs = [{'base': p.base, 'tuned': p.tuned, 'secondary': p.secondary,
              'forward': by[(p.tuned, FORWARD)]['state'],
              'reverse': by[(p.tuned, REVERSE)]['state'],
              'reading': reading(by[(p.tuned, FORWARD)], by[(p.tuned, REVERSE)])}
             for p in admitted_pairs()]
    return {'status': 'complete', 'method': native['method'], 'draws': native['draws'],
            'seed': native['seed'], 'groups': native['groups'],
            'family_size': native['family_size'], 'confidence': native['confidence'],
            'resolution': native['critical_value'], 'instrument_floor': INSTRUMENT_FLOOR,
            'controls_performance': controls_performance, 'columns': columns, 'pairs': pairs}


def contract() -> dict[str, Any]:
    """Everything this endpoint declares, for the run record."""

    return {
        'endpoint': ENDPOINT, 'declared_in': 'docs/objective-alignment.md, third and fourth amendments',
        'cohort': COHORT, 'control_set': CONTROL_SET, 'control_roles': list(CONTROL_ROLES),
        'metric': METRIC, 'stratum': STRATUM, 'directions': dict(DIRECTIONS),
        'estimand': 'tuned-minus-base difference of the forward and reverse rank-correlation '
                    'increments against Profile + Potts + GEMME',
        'unit': 'biological family; seeds within assay, assays within family, families equal',
        'family': {'admitted_pairs': len(ADMITTED), 'directions': len(DIRECTIONS),
                   'size': family_size(),
                   'separate_from': ['the primary pLDDT family', 'the published ladder family']},
        'admitted': list(ADMITTED), 'not_measured': dict(NOT_MEASURED_REASONS),
        'reused_archived_bases': [p.base for p in admitted_pairs() if reuses_archive(p.base)],
        'bootstrap': {'draws': bidirectional.DRAWS, 'seed': bidirectional.SEED,
                      'minimum_units': MINIMUM_BOOTSTRAP_UNITS},
        'states': list(STATES), 'disposition': DISPOSITION, 'readings': READINGS,
        'instrument_bounds': {'coupling_run_spread': COUPLING_RUN_SPREAD,
                              'capped_fit_factor': CAPPED_FIT_FACTOR,
                              'correlation_floor': INSTRUMENT_FLOOR,
                              'floor_basis': DECLARATION['instrument_floor']['basis']},
        'archived_reuse_rule': lanes.ARCHIVED_REUSE_RULE, 'bounds': list(BOUNDS),
    }


def _check_declarations() -> None:
    """Assert the declarations against each other at import."""

    if tuple(DIRECTIONS.values()) != tuple(ladder.DIRECTIONS) or \
            tuple(DIRECTIONS.values()) != bidirectional.DIRECTIONS:
        raise RuntimeError('the two directions must be the ladder\'s, in the ladder\'s order')
    if METRIC not in ladder.METRICS or CONTROL_SET not in ladder.CONTROL_SETS:
        raise RuntimeError('the metric and control set must be ones the ladder declares')
    letters = {letter for role in CONTROL_ROLES for letter in lanes.VARIANT_EFFECT_ROLES[role]}
    declared = ladder.CONTROL_SETS[CONTROL_SET]
    if set(ladder.DESIGNS[declared['control']]) != letters or declared['model'] != 'M' \
            or set(ladder.DESIGNS[declared['joint']]) != letters | {'M'}:
        raise RuntimeError('the control set no longer holds exactly Profile, Potts and GEMME')
    if STATES != ('supported', 'bounded_negative', 'unresolved', 'non_identifiable') or \
            set(DISPOSITION) != set(STATES) or NOT_MEASURED in STATES:
        raise RuntimeError('the disposition must map onto exactly the four finding states')
    declared_pairs = {p.tuned for p in tuned_arms.PAIRS}
    if len(ADMITTED) != len(set(ADMITTED)) or set(ADMITTED) & set(NOT_MEASURED_REASONS) or \
            set(ADMITTED) | set(NOT_MEASURED_REASONS) != declared_pairs:
        raise RuntimeError('admitted and not-measured pairs must partition the declared pairs')
    if not ADMITTED or not all(NOT_MEASURED_REASONS.values()):
        raise RuntimeError('the family needs a pair, and a pair left out needs its reason')
    if any(p.tuned in lanes.PANEL_ARMS for p in admitted_pairs()):
        raise RuntimeError('a tuned arm may not join the published ladder\'s roster')
    if not 0 < INSTRUMENT_FLOOR < 1:
        raise RuntimeError('the correlation-unit floor must be a measured value in (0, 1)')
    if family_size() != len(labels()):
        raise RuntimeError('the family is the admitted pairs crossed with the two directions')


_check_declarations()
