"""Retrospective measured-label selection over frozen held-out stability predictions.

The question is narrower than "does a model predict stability well". It is whether
the model's held-out likelihood difference, added to a baseline that already
carries evolutionary information, finds the measured stabilising members of a
*known* wild-type library better than that baseline does -- on a benign cohort
whose membership was reviewed and frozen before any outcome was read.

Six things are fixed here rather than left to a run.

**Orientation.** The endpoint is ``ddG = y_mut - y_WT`` on the combined MegaScale
``dG_ML`` scale in kcal/mol, and ``dG_ML`` is an unfolding free energy, so a
**positive measured ddG is stabilising**. Every selector ranks by *descending*
held-out prediction, which is a fitted ddG-scale prediction; the descriptive
oracle ranks by descending *measured* ddG. A sign inversion therefore reads as a
sign flip rather than as a plausible gain.

**Selection is always inside one wild type.** Every wild type is its own frozen
candidate library, so the random expectation a selector is scored against is
that library's own measured mean. Raw scores are never pooled across wild types
or across models: the only quantities that leave a wild type are its selection
gain and the paired gain between two selectors over the identical library.

**Seeds are not independent units.** The three split seeds are three readings of
one fitted held-out prediction set, so they are averaged *inside* each wild type
before any resampling. Treating them as units would narrow every interval by
about the square root of three.

**The bootstrap unit is the family group**, and the draw is one common set of
group indices shared by every selector, arm and budget. Outcomes are fixed
(group-level selection results conditional on the trained models) and are not
resampled inside a draw, so no retraining uncertainty is claimed. A group drawn
more than once contributes its weight once per draw and is never deduplicated.

**The contrast is primary and secondary, not a grid.** Primary is ``S2_M`` minus
``S2``: the likelihood column over the profile-inclusive baseline, which is the
only baseline that answers whether the model added something an alignment
profile did not already supply. Secondary is ``S_M`` minus ``S`` over the
matched baseline, published beside it. ``S_T`` is the tokenisation-descriptor
control and is not an information arm. A native absolute log-likelihood and the
delta log-likelihood differ by a per-wild-type constant, so within a fixed wild
type they induce the identical ranking and are not two experiments.

**Support gates the interval.** Below
:data:`~..core.statistics.MINIMUM_BOOTSTRAP_UNITS` family groups the result is
descriptive with an explicit reason and no interval, because a percentile
interval over so few atoms under-covers and can come out narrower than one over
hundreds of units.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..core.io import sha256_file
from ..core.statistics import MINIMUM_BOOTSTRAP_UNITS, bootstrap_unit_floor
from ..evaluation.generated_phenotype import selection_count
from ..interactions.pairwise_epistasis import BOOTSTRAP_SEED, ROSTER

#: Repository root, used to resolve the relative paths a frozen allowlist names.
ROOT = Path(__file__).resolve().parents[3]

SCHEMA = 'measured_selection_analysis_v1'
ALLOWLIST_SCHEMA = 'measured_selection_allowlist_v1'
ROLES_SCHEMA = 'measured_selection_roles_v1'

#: The roles the intended comparison is frozen over. The mapping from each role
#: to a concrete roster arm is supplied by the CLI and hashed, so which
#: checkpoint plays "stage 2" is declared rather than inferred from a name.
REQUIRED_ROLES = ('prollama_stage1', 'prollama_stage2', 'protgpt2', 'progen2_xlarge')

#: Selection budgets, as fractions of a wild type's own candidate library. The
#: whole-library budget is deliberately absent: it carries no selection.
FRACTIONS: tuple[float, ...] = (0.02, 0.05, 0.1, 0.25, 0.5)

#: The held-out design columns a selector may rank by. ``S_T`` is retained only
#: so a restricted run can prove it was not used as a selector.
SELECTION_DESIGNS = ('S', 'S2', 'S_M', 'S2_M')

PRIMARY = {
    'name': 'profile_likelihood_selection',
    'augmented': 'S2_M',
    'baseline': 'S2',
    'role': 'primary',
    'question': ('does the likelihood difference find stabilising variants beyond a '
                 'baseline that already carries the mutation-local alignment profile'),
}
SECONDARY = {
    'name': 'matched_likelihood_selection',
    'augmented': 'S_M',
    'baseline': 'S',
    'role': 'secondary',
    'question': ('does the likelihood difference find stabilising variants beyond the '
                 'matched baseline, which carries no evolutionary statistic'),
}
EXCLUDED = {
    'S_T': ('tokenisation-descriptor control, not an information arm; it exists to '
            're-derive the matched-baseline rule, not to select'),
}

LABEL_ORIENTATION = {
    'endpoint': 'ddG = y_mut - y_WT on the combined MegaScale dG_ML scale, kcal/mol',
    'sign_convention': 'dG_ML is an unfolding free energy, so a positive measured ddG is '
                       'stabilising',
    'selector_direction': 'rank by descending held-out prediction; the prediction is on the '
                          'fitted ddG scale, so a larger value is read as more stabilising',
    'oracle_direction': 'rank by descending measured ddG; a descriptive best-k ceiling, not '
                        'a usable selector',
    'random_expectation': 'the whole-library mean measured ddG of the same wild type',
    'native_likelihood_note': 'an absolute native log-likelihood and the delta '
                              'log-likelihood differ by a per-wild-type constant, so '
                              'within a fixed wild type they induce the identical ranking '
                              'and are not two experiments',
}


# --------------------------------------------------------------------------- #
# Cohort flattening and the positional join onto the retained predictions.
# --------------------------------------------------------------------------- #

def flatten_cohort(cohort: dict) -> dict:
    """Every variant as one row, in the frozen panel order.

    The order is the order :func:`~.stability_gate.build_panel` appends, which is
    the order the held-out prediction archives were written in. ``sample_id`` is
    ``background:position`` and is **not unique**: several substitutions at one
    position share it. The distinguishing key is the triple
    ``(background, position, state)``, where ``state`` is the variant's own index
    into its background's sequence list, and it is that state key -- never
    ``sample_id`` -- that breaks a prediction tie.
    """

    backgrounds, groups, positions, states, measured, mutants = [], [], [], [], [], []
    for row in cohort['backgrounds']:
        name = row['name']
        for variant in row['variants']:
            backgrounds.append(name)
            groups.append(row['group'])
            positions.append(int(variant['position']))
            states.append(int(variant['state']))
            mutants.append(str(variant['mutant']))
            measured.append(float(variant['ddg']))
    values = np.asarray(measured, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('the cohort carries a nonfinite measured ddG')
    backgrounds = np.asarray(backgrounds)
    groups = np.asarray(groups)
    positions = np.asarray(positions, dtype=int)
    states = np.asarray(states, dtype=int)
    state_keys = [f'{b}:{p}:{s}' for b, p, s in zip(backgrounds, positions, states)]
    if len(set(state_keys)) != len(state_keys):
        raise ValueError('the cohort state keys are not unique')
    return {
        'backgrounds': backgrounds,
        'groups': groups,
        'positions': positions,
        'states': states,
        'mutants': np.asarray(mutants),
        'measured': values,
        'state_keys': state_keys,
        'sample_ids': [f'{b}:{p}' for b, p in zip(backgrounds, positions)],
    }


def verify_oof_join(flattened: dict, sample_id, group, target, *, arm: str) -> None:
    """Refuse any held-out archive that is not on the frozen cohort rows.

    The comparison is positional: row ``i`` of the archive must be row ``i`` of
    the flattened cohort, with the same ``background:position`` sample id, the
    same family group and the same measured ddG. A dict keyed by ``sample_id``
    would silently collapse repeated substitutions at one position, so no such
    key is ever built; a reordered or substituted archive fails on the first
    disagreeing row.
    """

    n = len(flattened['measured'])
    sample_id = np.asarray(sample_id).astype(str)
    group = np.asarray(group).astype(str)
    target = np.asarray(target, dtype=float)
    if not (len(sample_id) == len(group) == len(target) == n):
        raise ValueError(f'{arm}: the held-out archive has {len(sample_id)} rows against the '
                         f"cohort's {n}")
    if not np.array_equal(sample_id, np.asarray(flattened['sample_ids'])):
        bad = int(np.flatnonzero(sample_id != np.asarray(flattened['sample_ids']))[0])
        raise ValueError(f'{arm}: row {bad} carries sample id {sample_id[bad]!r} against the '
                         f"cohort's {flattened['sample_ids'][bad]!r}; the archive is not in "
                         'the frozen row order')
    if not np.array_equal(group, flattened['groups']):
        bad = int(np.flatnonzero(group != flattened['groups'])[0])
        raise ValueError(f'{arm}: row {bad} carries group {group[bad]!r} against the '
                         f"cohort's {flattened['groups'][bad]!r}")
    if not np.allclose(target, flattened['measured'], rtol=0.0, atol=1e-9):
        bad = int(np.flatnonzero(~np.isclose(target, flattened['measured'], rtol=0.0,
                                             atol=1e-9))[0])
        raise ValueError(f'{arm}: row {bad} carries target {target[bad]!r} against the '
                         f"cohort's measured {flattened['measured'][bad]!r}")


# --------------------------------------------------------------------------- #
# Within-wild-type selection, deterministic ties, oracle ceiling.
# --------------------------------------------------------------------------- #

def _require_aligned(measured, prediction, background, state_keys) -> None:
    if not (len(measured) == len(prediction) == len(background) == len(state_keys)):
        raise ValueError('measured, prediction, background and state keys must align')
    if not np.isfinite(np.asarray(prediction, dtype=float)).all():
        raise ValueError('a held-out prediction vector carries a nonfinite value')


def rank_order(score, state_keys) -> np.ndarray:
    """Descending-score order with an outcome-blind state-key tie break.

    Equal scores are ordered by ``background:position:state``, which names no
    measurement and no confidence width, so a tie cannot be resolved towards a
    larger measured effect.
    """

    score = np.asarray(score, dtype=float)
    if len(score) != len(state_keys):
        raise ValueError('a rank order needs one state key per score')
    return np.asarray(sorted(range(len(score)),
                             key=lambda index: (-float(score[index]), state_keys[index])),
                      dtype=int)


def oracle_order(measured) -> np.ndarray:
    """Descending-measured order; the descriptive best-k, never a selector."""

    measured = np.asarray(measured, dtype=float)
    return np.asarray(sorted(range(len(measured)), key=lambda index: -float(measured[index])),
                      dtype=int)


def library_gains(measured, prediction, background, state_keys, fraction: float) -> dict:
    """Per-wild-type selection gain and descriptive oracle gain at one budget.

    Selection is strictly inside a wild type: the candidate library is that
    wild type's own measured variants, the budget is ``max(1, ceil(fraction*n))``
    of that library, and the random expectation is the library's own mean. The
    selector ranks by descending prediction and the oracle by descending measured
    ddG, both with the state-key tie break.
    """

    measured = np.asarray(measured, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    background = np.asarray(background)
    _require_aligned(measured, prediction, background, state_keys)
    labels = sorted(set(background.tolist()))
    columns = {key: [] for key in ('k', 'n_library', 'library_mean', 'selected_mean', 'gain',
                                   'oracle_mean', 'oracle_gain')}
    shares = []
    for label in labels:
        rows = np.flatnonzero(background == label)
        k = selection_count(int(len(rows)), fraction)
        keys = [state_keys[index] for index in rows]
        chosen = rows[rank_order(prediction[rows], keys)[:k]]
        best = rows[oracle_order(measured[rows])[:k]]
        library = float(measured[rows].mean())
        selected = float(measured[chosen].mean())
        oracle = float(measured[best].mean())
        gain = selected - library
        oracle_gain = oracle - library
        columns['k'].append(k)
        columns['n_library'].append(int(len(rows)))
        columns['library_mean'].append(library)
        columns['selected_mean'].append(selected)
        columns['gain'].append(gain)
        columns['oracle_mean'].append(oracle)
        columns['oracle_gain'].append(oracle_gain)
        shares.append(gain / oracle_gain if oracle_gain > 0 else None)
    numeric = {key: np.asarray(value, dtype=float) for key, value in columns.items()}
    return {'backgrounds': labels, 'oracle_share': shares, **numeric}


def oracle_ceiling(measured, background, state_keys, fraction: float) -> dict:
    """Per-wild-type descriptive best-k gain, with no selector involved."""

    measured = np.asarray(measured, dtype=float)
    background = np.asarray(background)
    if not (len(measured) == len(background) == len(state_keys)):
        raise ValueError('measured, background and state keys must align')
    labels = sorted(set(background.tolist()))
    gains, means, counts, budgets = [], [], [], []
    for label in labels:
        rows = np.flatnonzero(background == label)
        k = selection_count(int(len(rows)), fraction)
        best = rows[oracle_order(measured[rows])[:k]]
        library = float(measured[rows].mean())
        gains.append(float(measured[best].mean()) - library)
        means.append(library)
        counts.append(int(len(rows)))
        budgets.append(k)
    return {'backgrounds': labels, 'oracle_gain': np.asarray(gains, dtype=float),
            'library_mean': np.asarray(means, dtype=float),
            'n_library': np.asarray(counts, dtype=int),
            'k': np.asarray(budgets, dtype=int)}


def average_seeds(per_seed) -> np.ndarray:
    """Split seeds averaged inside each wild type before any resampling."""

    stacked = np.asarray(per_seed, dtype=float)
    if stacked.ndim != 2 or stacked.shape[0] < 1:
        raise ValueError('seed averaging needs at least one seed on a shared wild-type axis')
    return stacked.mean(axis=0)


def aggregate_groups(values, wt_labels, wt_groups) -> tuple[list, np.ndarray]:
    """One value per family group, equally weighting the wild types inside it.

    ``values`` may be one- or two-dimensional over the wild-type axis. A group's
    value is the mean over its own wild types and is not weighted by how many
    wild types the group happens to hold, so a group with two libraries does not
    dominate one with a single library.
    """

    values = np.asarray(values, dtype=float)
    labels = np.asarray(wt_labels)
    groups = np.asarray(wt_groups)
    if values.shape[0] != len(labels) or len(labels) != len(groups):
        raise ValueError('the wild-type values, labels and groups must align')
    order = sorted(set(groups.tolist()))
    aggregated = np.stack([values[groups == group].mean(axis=0) for group in order])
    return order, aggregated


# --------------------------------------------------------------------------- #
# Common-draw group bootstrap and the two published inference readings.
# --------------------------------------------------------------------------- #

def resample_group_means(matrix, indices) -> np.ndarray:
    """Mean of a matrix over drawn group indices, repeated groups kept in full.

    A group named twice in one draw contributes twice. There is no deduplication
    by group identity: collapsing a repeat would turn the group bootstrap back
    into a row bootstrap and down-weight the repeated family.
    """

    matrix = np.asarray(matrix, dtype=float)
    indices = np.asarray(indices, dtype=int)
    if matrix.ndim != 2 or indices.ndim != 2:
        raise ValueError('the matrix and the group index draw must be two-dimensional')
    if indices.size and (indices.min() < 0 or indices.max() >= matrix.shape[0]):
        raise ValueError('a group index draw names a group outside the support')
    return matrix[indices].mean(axis=1)


def joint_group_bootstrap(matrix, *, draws: int = 10000, seed: int = BOOTSTRAP_SEED) -> dict:
    """One common group draw for every column, so the columns stay paired.

    Rows are family groups and columns are selectors, arms and budgets. The same
    group indices resample every column, which keeps a paired model-minus-baseline
    gain paired inside the bootstrap rather than only at the point estimate.
    Outcomes are fixed before the draw: selection is not redone per replicate, so
    the interval conditions on the trained models and claims no retraining
    variation.
    """

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError('a group bootstrap needs a finite group-by-column matrix')
    if not np.isfinite(matrix).all():
        raise ValueError('a nonfinite group-level value is not evaluable')
    if draws < 1:
        raise ValueError('a bootstrap needs at least one draw')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, matrix.shape[0], size=(int(draws), matrix.shape[0]))
    return {
        'draws_matrix': resample_group_means(matrix, indices),
        'point': matrix.mean(axis=0),
        'standard_error': matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0]),
        'n_groups': int(matrix.shape[0]),
        'n_columns': int(matrix.shape[1]),
        'n_draws': int(draws),
        'seed': int(seed),
    }


def percentile_interval(draws) -> list:
    """Two-sided 95% percentile interval of one draw column."""

    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 1 or draws.size < 2:
        raise ValueError('a percentile interval needs one draw column')
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def simultaneous_band(draws_matrix, point, standard_error, *, confidence: float = 0.95) -> dict:
    """Maximum-absolute-centred-statistic band over one declared column family.

    The construction is the project's own: the statistic is studentised by the
    fixed group standard error, the critical value is the ``confidence`` quantile
    of the maximum absolute deviation across the family's columns, and the band
    is the point estimate plus and minus that critical value times each column's
    own fixed standard error. It is simultaneous over the declared family only;
    it is not a pooled statement across the primary and secondary families.
    """

    draws_matrix = np.asarray(draws_matrix, dtype=float)
    point = np.asarray(point, dtype=float)
    standard_error = np.asarray(standard_error, dtype=float)
    if draws_matrix.ndim != 2 or draws_matrix.shape[1] != point.size:
        raise ValueError('a simultaneous band needs one draw column per point')
    varying = standard_error > 0
    if varying.any():
        deviations = (draws_matrix[:, varying] - point[varying]) / standard_error[varying]
        critical = float(np.quantile(np.max(np.abs(deviations), axis=1), confidence))
    else:
        critical = 0.0
    intervals = np.column_stack([point - critical * standard_error,
                                 point + critical * standard_error])
    return {
        'method': ('paired group bootstrap maximum absolute centered statistic / fixed '
                   'group SE'),
        'critical_value': critical,
        'confidence': float(confidence),
        'interval': intervals.tolist(),
        'conditional_on_fitted_predictions': True,
    }


def verdict(interval) -> str:
    """Resolved direction of one interval, with the straddle case named."""

    low, high = float(interval[0]), float(interval[1])
    if low > 0:
        return 'above_zero'
    if high < 0:
        return 'below_zero'
    return 'unresolved'


def support_floor(n_groups: int) -> dict:
    """Whether this group count may publish a percentile interval at all."""

    return bootstrap_unit_floor(int(n_groups), minimum_units=MINIMUM_BOOTSTRAP_UNITS)


# --------------------------------------------------------------------------- #
# Frozen-allowlist and role-sheet validation. The allowlist is a supplied review
# protocol; no metadata field in it is treated as an automated safety verdict.
# --------------------------------------------------------------------------- #

def validate_allowlist(raw: dict, cohort: dict, *, root=ROOT) -> dict:
    """Validate a reviewed benign-background allowlist against the frozen cohort.

    The allowlist is the eligibility protocol: an explicit, reviewed list of
    wild-type backgrounds, each naming its group, organism, function, construct
    match and the hashed source and provenance files the review read. Nothing
    here certifies that a background is safe -- a metadata field cannot, and no
    field is read as if it did. What is checked is that the list is explicit,
    nonempty, duplicate-free, that every named background exists in the frozen
    cohort under the group the review assigned, that the construct match is
    asserted true with a stated method, and that every named file still hashes to
    what the review recorded.
    """

    if not isinstance(raw, dict) or raw.get('schema') != ALLOWLIST_SCHEMA:
        raise ValueError(f'the allowlist must declare schema {ALLOWLIST_SCHEMA!r}')
    review = raw.get('review')
    if not isinstance(review, dict) or not str(review.get('protocol', '')).strip():
        raise ValueError('the allowlist must carry a review block with a nonempty protocol')
    backgrounds = raw.get('backgrounds')
    if not isinstance(backgrounds, list) or not backgrounds:
        raise ValueError('the allowlist must name at least one reviewed background')
    group_of = {row['name']: row['group'] for row in cohort['backgrounds']}
    entries, seen = [], set()
    for item in backgrounds:
        if not isinstance(item, dict):
            raise ValueError('every allowlist background must be an object')
        name = str(item.get('name', '')).strip()
        if not name:
            raise ValueError('an allowlist background has an empty name')
        if name in seen:
            raise ValueError(f'{name}: duplicated allowlist background')
        seen.add(name)
        if name not in group_of:
            raise ValueError(f'{name}: not a background of the frozen cohort')
        group = str(item.get('group', '')).strip()
        if not group:
            raise ValueError(f'{name}: an allowlist background must name its group')
        if group != group_of[name]:
            raise ValueError(f'{name}: the review assigns group {group!r} against the '
                             f"cohort's {group_of[name]!r}")
        for field in ('protein', 'organism', 'function'):
            if not str(item.get(field, '')).strip():
                raise ValueError(f'{name}: an allowlist background must name its {field}')
        match = item.get('construct_match')
        if not isinstance(match, dict) or match.get('matched') is not True:
            raise ValueError(f'{name}: construct_match must assert a matched construct')
        if not str(match.get('method', '')).strip():
            raise ValueError(f'{name}: construct_match must state its method')
        references = {}
        for kind in ('sources', 'provenance'):
            refs = item.get(kind)
            if not isinstance(refs, list) or not refs:
                raise ValueError(f'{name}: the allowlist must carry hashed {kind} paths')
            resolved = []
            for ref in refs:
                if not isinstance(ref, dict) or not str(ref.get('path', '')).strip():
                    raise ValueError(f'{name}: every {kind} entry needs a path')
                digest = str(ref.get('sha256', '')).strip().lower()
                if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                    raise ValueError(f'{name}: {kind} path {ref["path"]!r} needs a sha256')
                path = Path(ref['path'])
                if not path.is_absolute():
                    path = Path(root) / path
                if not path.is_file():
                    raise ValueError(f'{name}: {kind} path {path} does not exist')
                actual = sha256_file(path)
                if actual != digest:
                    raise ValueError(f'{name}: {kind} path {path} hashes to {actual}, not the '
                                     f'reviewed {digest}')
                resolved.append({'path': str(ref['path']), 'resolved_path': str(path),
                                 'sha256': actual})
            references[kind] = resolved
        review_ref = item.get('construct_review') or {}
        review_path = Path(review_ref.get('path', ''))
        if not review_path.is_absolute():
            review_path = Path(root) / review_path
        if not review_path.is_file() or sha256_file(review_path) != review_ref.get('sha256'):
            raise ValueError(f'{name}: construct review file is absent or its hash differs')
        evidence = json.loads(review_path.read_bytes())
        wildtype = next(row['wildtype'] for row in cohort['backgrounds'] if row['name'] == name)
        if (evidence.get('schema') != 'measured_construct_review_v1'
                or evidence.get('name') != name
                or evidence.get('reference_construct_sha256')
                != hashlib.sha256(wildtype.encode()).hexdigest()
                or evidence.get('source') not in item['sources']
                or evidence.get('match_rule') != 'exact_or_contiguous_full_construct'):
            raise ValueError(f'{name}: construct review does not bind the source and wild type')
        for kind, id_field in (('canonical', 'entity_id'), ('resolved', 'label_asym_id')):
            part = evidence.get(kind) or {}
            bounds = part.get('match_range_1based_inclusive') or []
            if (not part.get(id_field) or len(bounds) != 2
                    or not all(isinstance(value, int) for value in bounds)
                    or bounds[0] < 1 or bounds[1] - bounds[0] + 1 != len(wildtype)
                    or (kind == 'resolved' and part.get('model') != 1)):
                raise ValueError(f'{name}: incomplete {kind} construct review')
        entries.append({
            'name': name, 'group': group,
            'protein': str(item['protein']).strip(),
            'organism': str(item['organism']).strip(),
            'function': str(item['function']).strip(),
            'construct_match': {'matched': True, 'method': str(match['method']).strip()},
            'sources': references['sources'],
            'provenance': references['provenance'],
            'construct_review': {**review_ref, 'record': evidence},
        })
    return {
        'schema': ALLOWLIST_SCHEMA,
        'review': review,
        'backgrounds': entries,
        'background_count': len(entries),
        'review_is_supplied_protocol': True,
        'safety_not_inferred_from_metadata': (
            'eligibility is the supplied review protocol; no field in this file is read as '
            'an automated safety verdict, and permitting a background here is not a claim '
            'that it is safe to deploy or optimise against'),
    }


def load_allowlist(path, cohort: dict, *, root=ROOT) -> dict:
    """Read and validate a frozen allowlist, recording its own digest."""

    path = Path(path)
    record = validate_allowlist(json.loads(path.read_bytes()), cohort, root=root)
    record.update(path=str(path), sha256=sha256_file(path))
    return record


def validate_roles(raw: dict, *, roster=ROSTER) -> dict:
    """Validate the frozen role-to-arm sheet the comparison is declared over.

    The intended arms are named by role rather than by a name pattern, and the
    mapping is supplied and hashed on the CLI. Every required role must be
    present, no other role may be added silently, and each role must resolve to a
    distinct arm on the frozen roster.
    """

    if not isinstance(raw, dict) or raw.get('schema') != ROLES_SCHEMA:
        raise ValueError(f'the role sheet must declare schema {ROLES_SCHEMA!r}')
    roles = raw.get('roles')
    if not isinstance(roles, dict):
        raise ValueError('the role sheet must carry a roles object')
    missing = sorted(set(REQUIRED_ROLES) - set(roles))
    extra = sorted(set(roles) - set(REQUIRED_ROLES))
    if missing or extra:
        raise ValueError(f'the role sheet must declare exactly {list(REQUIRED_ROLES)}; '
                         f'missing {missing}, unexpected {extra}')
    mapping = {role: str(roles[role]) for role in REQUIRED_ROLES}
    arms = list(mapping.values())
    if len(set(arms)) != len(arms):
        raise ValueError(f'the role sheet maps two roles onto one arm: {arms}')
    unknown = [arm for arm in arms if arm not in roster]
    if unknown:
        raise ValueError(f'not on the frozen roster: {unknown}')
    protocol = raw.get('protocol')
    if not isinstance(protocol, dict):
        raise ValueError('the role sheet must freeze the final panel protocol hashes')
    for field in ('completion_sha256', 'panel_sha256', 'cohort_sha256', 'allowlist_sha256'):
        digest = protocol.get(field)
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError(f'the role sheet protocol needs {field}')
    return {'schema': ROLES_SCHEMA, 'roles': mapping, 'arms': arms,
            'rationale': raw.get('rationale'), 'protocol': protocol}


def load_roles(path, *, roster=ROSTER) -> dict:
    """Read and validate a frozen role sheet, recording its own digest."""

    path = Path(path)
    record = validate_roles(json.loads(path.read_bytes()), roster=roster)
    record.update(path=str(path), sha256=sha256_file(path))
    return record


def jsonable(values) -> list:
    """Finite values as floats, undefined ones as null, for a strict JSON writer."""

    return [None if value is None or not np.isfinite(value) else float(value)
            for value in values]


__all__ = [
    'ALLOWLIST_SCHEMA', 'EXCLUDED', 'FRACTIONS', 'LABEL_ORIENTATION', 'PRIMARY',
    'REQUIRED_ROLES', 'ROLES_SCHEMA', 'ROOT', 'SCHEMA', 'SECONDARY', 'SELECTION_DESIGNS',
    'aggregate_groups', 'average_seeds', 'flatten_cohort', 'joint_group_bootstrap',
    'jsonable', 'library_gains', 'load_allowlist', 'load_roles', 'oracle_ceiling',
    'oracle_order', 'percentile_interval', 'rank_order', 'resample_group_means',
    'simultaneous_band', 'support_floor', 'validate_allowlist', 'validate_roles', 'verdict',
    'verify_oof_join',
]
