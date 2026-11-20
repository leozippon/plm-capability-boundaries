#!/usr/bin/env python3
"""The bidirectional ablation of contact information: propagation against couplings.

`measure_contact_discrimination.py` reads each quantity on its own axis. Side-by-side
columns cannot separate "the model carries contact information the couplings lack"
from "both carry the same information and the coupling column is merely noisier",
because neither column is ever conditioned on the other. This stage fits the
quantity the rest of the programme reports instead: controls alone, model alone, and
both jointly, on identical rows, giving a forward increment -- the model beyond the
explicit statistics -- and a reverse increment -- the explicit statistics beyond the
model.

**The control set contains sequence separation.** Contact frequency falls with
separation, so separation is a nuisance term inside ``U_C`` rather than a comparator
beside it, which is how every lane handles its own geometry block and is what makes
the forward increment mean "beyond what geometry alone gives you". Relative
accessibility is deliberately absent: it is derived from the same structure the
contact labels come from, and a nuisance rung built out of the endpoint's own
measurement apparatus is not a nuisance rung.

**Nothing here is a new estimator.** The folds, the ridge penalty choice, the row
weighting and the split seeds are :mod:`...extensions.phenotype_breadth`'s, the
designs and directions are :mod:`...extensions.bidirectional`'s, and the band is its
``paired_bands``. The only new things are the blocks a residue *pair* carries and the
two per-group statistics a binary endpoint admits.

**Two strata, never pooled.** ``contact_ranking`` is the per-group area under the
curve of the held-out prediction against the contact label, which is the quantity a
contact readout is conventionally judged on. ``contact_squared_error`` is the
per-group weighted Brier score of the same predictions. They are two readings of one
set of predictions and each is banded as its own family.

**Alignment depth is read from the fit that produced the coupling column**, not from
another fit of the same backbone. ``--min-neff-per-site`` restricts the references to
those whose own alignment reaches a stated effective depth per site, so the increment
can be read in the regime where a coupling model is expected to work rather than only
on the cohort average. The realised depth of every reference is recorded either way.

**One row support per column, never a shifting one.** A causal arm carries only the
pairs downstream of the substitution, and a byte-pair alphabet loses receivers whose
token spans several residues, so which pairs carry a model value differs by arm. The
lane rule is that an increment is never computed on shifting rows, so each arm's three
designs -- ``U_C``, ``M`` and ``U_C_M`` -- are fitted on exactly the pairs that arm
carries, with the control set refitted on those same rows. Rows differ *between* arms
and are identical *within* every contrast, which is the thing the rule protects. No
model value is ever imputed: a pair an arm does not carry is absent from that arm's
support rather than filled with a column summary.

**The arm support floor.** A column enters only if its arm carries at least
``MINIMUM_BOOTSTRAP_UNITS`` references. An arm whose byte-pair segmentation leaves
almost no single-residue receiver cannot be compared with a full-alphabet arm, and
admitting it would cost every other column the references it lacks.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json  # noqa: E402
from src.capability.core.statistics import MINIMUM_BOOTSTRAP_UNITS  # noqa: E402
from src.capability.extensions import bidirectional as bidir  # noqa: E402
from src.capability.extensions import phenotype_breadth as breadth  # noqa: E402
from src.capability.position.contact_response import (  # noqa: E402
    CONTACT_ANGSTROM,
    MIN_SEQUENCE_SEPARATION,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_contact_discrimination import (  # noqa: E402
    FULL_ALPHABET_MIN_PAIR_FRACTION,
    PARADIGM_SIDES,
    arm_pair_responses,
    auc_minus_half,
    effective_depth_per_site,
    paired_bands_with_marginal,
    reference_rows,
)

#: The rung hierarchy, as the declaration fixes it. Separation lives in ``geom`` and
#: therefore inside every control set; see the module docstring.
RUNGS = {
    'nuisance': ('geom',),
    'nuisance+profile': ('geom', 'prof'),
    'nuisance+profile+couplings': ('geom', 'prof', 'couplings'),
}

#: The per-group statistics a binary endpoint admits, and the sign that turns each
#: into a loss. A reduction is positive when the joint design lowers the loss.
#: ``contact_squared_error_clipped`` clips linear Ridge predictions to ``[0, 1]`` so
#: single-protein heavy-tail overshoot (e.g. raw-nats propagation extremes) cannot
#: corrupt squared-error evaluation, while ``contact_squared_error`` retains raw
#: unclipped Ridge Brier for backwards compatibility.
STRATA = {
    'contact_ranking': {
        'units': 'area under the curve of the held-out prediction against the contact label; '
                 'family-equal, pair-equal within reference',
        'to_loss': -1.,
    },
    'contact_squared_error': {
        'units': 'weighted Brier score of the held-out raw Ridge prediction; family-equal, '
                 'pair-equal within reference',
        'to_loss': 1.,
    },
    'contact_squared_error_clipped': {
        'units': 'weighted Brier score of the held-out Ridge prediction clipped to [0, 1]; '
                 'family-equal, pair-equal within reference',
        'to_loss': 1.,
    },
}

#: Why relative accessibility is not a control here, declared rather than left out
#: silently.
EXCLUDED_NUISANCE = (
    'relative accessibility and every other coordinate-derived receiver property is '
    'excluded from the control set. It is computed from the same experimental structure '
    'the contact labels come from, so admitting it would build the nuisance rung out of '
    'the endpoint\'s own measurement apparatus; no other lane does that')


def column_conservation(msa: Path, weights: Path) -> np.ndarray:
    """Weighted per-column Shannon entropy of one reference's own alignment, in nats.

    The alignment is the one its coupling model was fitted on and its columns are that
    fit's focus columns, so a column index here is the same position as a coupling
    index. Gaps are their own symbol: a column that is mostly gap is not a conserved
    column and must not be scored as one.
    """

    sequences, current = [], []
    for line in msa.read_text(encoding='ascii').splitlines():
        if line.startswith('>'):
            if current:
                sequences.append(''.join(current))
            current = []
        elif line:
            current.append(line.strip())
    if current:
        sequences.append(''.join(current))
    weight = np.array([float(value) for value in weights.read_text().split()], dtype=np.float64)
    if len(weight) != len(sequences):
        raise ValueError('the alignment and its weight file disagree on the sequence count')
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError('the alignment is not rectangular; its columns are not positions')
    table = np.frombuffer(''.join(sequences).encode('ascii'), dtype=np.uint8)
    table = table.reshape(len(sequences), lengths.pop())
    entropy = np.zeros(table.shape[1])
    total = weight.sum()
    for position in range(table.shape[1]):
        symbols, inverse = np.unique(table[:, position], return_inverse=True)
        mass = np.bincount(inverse, weights=weight, minlength=len(symbols)) / total
        mass = mass[mass > 0]
        entropy[position] = float(-(mass * np.log(mass)).sum())
    return entropy


def pair_blocks(record, conservation: np.ndarray) -> tuple[list, dict]:
    """The pair-level control blocks of one reference, and the pair keys they index.

    Pair keys from :func:`structure_sites` / :func:`contact_pairs` are already 0-based
    wild-type residue indices (``0 <= i < j < L``), matching both ``conservation`` and
    ``record['couplings']``. Bounds are checked explicitly so an out-of-range or
    negative index can never silently wrap.
    """

    keys = sorted(record['pairs']['distance'])
    length = int(record['length'])
    conservation = np.asarray(conservation, dtype=np.float64)
    if conservation.shape != (length,):
        raise ValueError(
            f'{record.get("assay", "<unknown>")}: conservation shape {conservation.shape} '
            f'disagrees with wild-type length {length}')
    couplings_matrix = np.asarray(record['couplings'], dtype=np.float64)
    if couplings_matrix.shape != (length, length):
        raise ValueError(
            f'{record.get("assay", "<unknown>")}: coupling matrix shape {couplings_matrix.shape} '
            f'disagrees with wild-type length {length}')
    low_idx = np.array([key[0] for key in keys], dtype=np.int64)
    high_idx = np.array([key[1] for key in keys], dtype=np.int64)
    if len(keys) and not (
        (low_idx >= 0).all() and (low_idx < high_idx).all() and (high_idx < length).all()
    ):
        raise ValueError(
            f'{record.get("assay", "<unknown>")}: pair keys must satisfy 0 <= low < high < {length}')
    low = low_idx.astype(np.float64)
    high = high_idx.astype(np.float64)
    separation = high - low
    geom = np.column_stack([
        separation, np.log(separation), 1. / separation,
        low / length, high / length, separation / length,
        np.full(len(keys), float(length)),
    ])
    first = conservation[low_idx]
    second = conservation[high_idx]
    prof = np.column_stack([
        np.minimum(first, second), np.maximum(first, second), first + second, first * second,
    ])
    couplings = np.array([[couplings_matrix[int(i), int(j)]] for i, j in keys], dtype=np.float64)
    label = np.array([record['pairs']['distance'][key] < record['pairs']['cutoff_angstrom']
                      for key in keys], dtype=np.float64)
    return keys, {'geom': geom, 'prof': prof, 'couplings': couplings, 'label': label}


def group_statistics(prediction: np.ndarray, label: np.ndarray, weights: np.ndarray,
                     groups: np.ndarray) -> dict[str, dict]:
    """Per-group AUC, raw weighted Brier and [0, 1]-clipped Brier of one out-of-fold prediction."""

    order = sorted(set(groups.tolist()))
    clipped = np.clip(prediction, 0.0, 1.0)
    ranking, squared, squared_clipped = {}, {}, {}
    for group in order:
        mask = groups == group
        area = auc_minus_half(prediction[mask], label[mask] > .5)
        if area is None:
            raise ValueError(f'group {group!r} carries one contact class only; its area under the '
                             'curve is undefined and it cannot enter the support')
        ranking[group] = area + .5
        inside = weights[mask]
        denom = inside.sum()
        squared[group] = float((inside * (prediction[mask] - label[mask]) ** 2).sum() / denom)
        squared_clipped[group] = float((inside * (clipped[mask] - label[mask]) ** 2).sum() / denom)
    return {
        'contact_ranking': ranking,
        'contact_squared_error': squared,
        'contact_squared_error_clipped': squared_clipped,
    }


def evaluate_panel(panel_arms: list[dict], blocks: dict[str, dict],
                   model: dict[str, dict[str, dict]], support: dict[str, dict],
                   *, min_units: int = MINIMUM_BOOTSTRAP_UNITS) -> dict:
    """Fit and band the bidirectional panel plus pure couplings reverse increment."""

    complete = sorted(set.intersection(*[set(arm['references']) for arm in panel_arms]))
    if len(complete) < min_units:
        raise ValueError(f'{len(complete)} references carry every admitted arm, below the '
                         f'{min_units}-unit floor; no increment is fitted')

    def assemble(arm: dict) -> dict:
        """Rows, blocks, weights and fold signatures of one arm's own pair support."""
        pieces = [(assay, model[arm['arm']][assay]['rows'])
                  for assay in complete if assay in model[arm['arm']]]
        label = np.concatenate([blocks[assay]['label'][rows] for assay, rows in pieces])
        groups = np.concatenate([np.full(len(rows), support[assay]['family'])
                                 for assay, rows in pieces]).astype(str)
        backgrounds = np.concatenate([np.full(len(rows), assay) for assay, rows in pieces])
        control = {name: np.concatenate([blocks[assay][name][rows] for assay, rows in pieces])
                   for name in ('geom', 'prof', 'couplings')}
        column = np.concatenate([model[arm['arm']][assay]['column'] for assay, _ in pieces])
        return {'references': [assay for assay, _ in pieces], 'label': label, 'groups': groups,
                'backgrounds': backgrounds, 'control': control, 'column': column,
                'weights': breadth.row_weights(backgrounds, groups),
                'signatures': {seed: breadth.outer_signature(groups, seed)
                               for seed in breadth.SPLIT_SEEDS},
                'order': sorted(set(groups.tolist())), 'pairs': int(len(label)),
                'contacts': int(label.sum())}

    def fit(state: dict, matrix: np.ndarray) -> dict[str, dict]:
        """Seed-averaged per-group statistics of one design on one row support."""
        gathered: dict[str, dict] = {name: collections.defaultdict(list) for name in STRATA}
        for seed in breadth.SPLIT_SEEDS:
            prediction, _ = breadth.fold_predictions(
                matrix, state['label'], state['backgrounds'], state['groups'],
                state['signatures'][seed])
            for name, values in group_statistics(prediction, state['label'], state['weights'],
                                                 state['groups']).items():
                for group, value in values.items():
                    gathered[name][group].append(value)
        return {name: {group: float(np.mean(values)) for group, values in table.items()}
                for name, table in gathered.items()}

    columns = [{'rung': rung, 'arm': arm['arm'], 'direction': direction}
               for rung in RUNGS for arm in panel_arms for direction in bidir.DIRECTIONS]
    declared = 2 * len(RUNGS) * len(panel_arms)
    if len(columns) != declared:
        raise ValueError('the realised family is not the declared product of directions, rungs '
                         'and admitted arms')

    states, cache = {}, {}
    for arm in panel_arms:
        state = assemble(arm)
        if len(state['order']) < min_units:
            raise ValueError(f'{arm["arm"]}: {len(state["order"])} families on its own pair '
                             f'support, below the {min_units}-unit floor')
        states[arm['arm']] = state
        cache[('M', None, arm['arm'])] = fit(state, state['column'])
        for rung, names in RUNGS.items():
            cache[('U_C', rung, arm['arm'])] = fit(state, breadth.design(state['control'], names))
            cache[('U_C_M', rung, arm['arm'])] = fit(
                state, np.column_stack([breadth.design(state['control'], names), state['column']]))

    # The family's resampling axis is the families every admitted arm carries, so one
    # bootstrap draw moves the same units in every column.
    shared = sorted(set.intersection(*[set(state['order']) for state in states.values()]))
    if len(shared) < min_units:
        raise ValueError(f'{len(shared)} families are carried by every admitted arm, below the '
                         f'{min_units}-unit floor; no family is banded')

    payload_strata = {}
    for stratum, declaration in STRATA.items():
        sign = declaration['to_loss']
        matrix, points = [], []
        for column in columns:
            reduced = (('U_C', column['rung'], column['arm'])
                       if column['direction'] == 'model_given_controls'
                       else ('M', None, column['arm']))
            joint = ('U_C_M', column['rung'], column['arm'])
            matrix.append([sign * cache[reduced][stratum][group]
                           - sign * cache[joint][stratum][group] for group in shared])
        matrix = np.asarray(matrix).T
        bands = paired_bands_with_marginal(matrix)
        verdicts = bidir.verdict(bands['interval'])
        marginal_verdicts = bidir.verdict(bands['marginal_interval'])
        control_performance = {
            rung: {arm['arm']: float(np.mean([cache[('U_C', rung, arm['arm'])][stratum][group]
                                              for group in shared]))
                   for arm in panel_arms} for rung in RUNGS}
        joint_performance = {
            rung: {arm['arm']: float(np.mean([cache[('U_C_M', rung, arm['arm'])][stratum][group]
                                              for group in shared]))
                   for arm in panel_arms} for rung in RUNGS}
        for position, column in enumerate(columns):
            points.append({
                **column,
                'point': bands['point'][position],
                'interval': bands['interval'][position],
                'verdict': verdicts[position],
                'marginal_interval': bands['marginal_interval'][position],
                'marginal_verdict': marginal_verdicts[position],
            })

        # Explicit pure couplings reverse increment:
        # couplings_given_geom_prof_model = J(geom+prof+couplings+M) - J(geom+prof+M)
        # (oriented via `sign` so positive always means adding couplings improves the stratum).
        coup_matrix = np.asarray([
            [
                sign * cache[('U_C_M', 'nuisance+profile', arm['arm'])][stratum][group]
                - sign * cache[('U_C_M', 'nuisance+profile+couplings', arm['arm'])][stratum][group]
                for group in shared
            ]
            for arm in panel_arms
        ]).T
        coup_bands = paired_bands_with_marginal(coup_matrix)
        coup_verdicts = bidir.verdict(coup_bands['interval'])
        coup_marginal_verdicts = bidir.verdict(coup_bands['marginal_interval'])
        coup_columns = []
        for position, arm in enumerate(panel_arms):
            arm_name = arm['arm']
            coup_columns.append({
                'arm': arm_name,
                'contrast': 'couplings_given_geom_prof_model',
                'reduced_design': 'geom+prof+M',
                'joint_design': 'geom+prof+couplings+M',
                'reduced_performance': joint_performance['nuisance+profile'][arm_name],
                'joint_performance': joint_performance['nuisance+profile+couplings'][arm_name],
                'point': coup_bands['point'][position],
                'interval': coup_bands['interval'][position],
                'verdict': coup_verdicts[position],
                'marginal_interval': coup_bands['marginal_interval'][position],
                'marginal_verdict': coup_marginal_verdicts[position],
            })

        payload_strata[stratum] = {
            'units': declaration['units'], 'to_loss': sign,
            'native': {key: value for key, value in bands.items()
                       if key not in ('point', 'interval', 'marginal_interval')},
            'columns': points,
            'control_performance': control_performance,
            'joint_performance': joint_performance,
            'model_alone_performance': {
                arm['arm']: float(np.mean([cache[('M', None, arm['arm'])][stratum][group]
                                           for group in shared])) for arm in panel_arms},
            'states': bidir.counts(verdicts),
            'marginal_states': bidir.counts(marginal_verdicts),
            'pure_couplings_increment': {
                'contrast': 'couplings_given_geom_prof_model',
                'definition': 'J(geom+prof+couplings+M) - J(geom+prof+M) oriented to loss reduction',
                'native': {key: value for key, value in coup_bands.items()
                           if key not in ('point', 'interval', 'marginal_interval')},
                'columns': coup_columns,
                'states': bidir.counts(coup_verdicts),
                'marginal_states': bidir.counts(coup_marginal_verdicts),
            },
        }

    return {
        'complete': complete,
        'shared': shared,
        'declared': declared,
        'states': states,
        'strata': payload_strata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--structures', type=Path, required=True)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--params-root', type=Path, action='append', required=True)
    parser.add_argument('--extraction', type=Path, action='append', required=True)
    parser.add_argument('--contact-angstrom', type=float, default=CONTACT_ANGSTROM)
    parser.add_argument('--min-separation', type=int, default=MIN_SEQUENCE_SEPARATION)
    parser.add_argument('--min-neff-per-site', type=float, default=0.,
                        help='keep only references whose own fitted alignment reaches this '
                             'effective depth per site (inclusive); 0 keeps every reference')
    parser.add_argument('--max-neff-per-site', type=float, default=None,
                        help='if set, keep only references strictly below this effective depth '
                             'per site (e.g. 5.0 for the shallow stratum)')
    parser.add_argument('--omitted-arm', action='append', default=[],
                        help='an arm the panel does not carry and why, as "name=reason"; '
                             'recorded so a locally-bounded panel names its own omissions')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.max_neff_per_site is not None and args.max_neff_per_site <= args.min_neff_per_site:
        raise SystemExit('--max-neff-per-site must exceed --min-neff-per-site')

    with gzip.open(args.structures, 'rt', encoding='utf-8') as handle:
        sites = json.load(handle)
    coverage = json.loads(args.coverage.read_text(encoding='utf-8'))
    cohort_rows = {row['assay']: row
                   for row in json.loads(args.cohort.read_text(encoding='utf-8'))['assays']}
    by_assay = collections.defaultdict(list)
    for row in sites:
        by_assay[row['assay_id']].append(row)

    admitted, refused_references = reference_rows(
        by_assay, coverage, args.params_root,
        cutoff=args.contact_angstrom, separation=args.min_separation)

    arms = []
    for directory in args.extraction:
        index = json.loads((directory / 'position_likelihood.json').read_text(encoding='utf-8'))
        paradigm = index['identity']['paradigm']
        if paradigm not in PARADIGM_SIDES:
            raise SystemExit(f'{directory.name}: unknown paradigm {paradigm!r}')
        arms.append({'arm': directory.name, 'directory': directory, 'paradigm': paradigm,
                     'files': {entry['assay']: entry for entry in index['assays']},
                     'index_sha256': sha256_file(directory / 'position_likelihood.json')})

    # ---- rows, blocks and per-arm model columns, per reference ----
    blocks: dict[str, dict] = {}
    shallow: dict[str, float] = {}
    excluded_above_ceiling: dict[str, float] = {}
    model: dict[str, dict[str, dict]] = {arm['arm']: {} for arm in arms}
    support: dict[str, dict] = {}
    for assay, record in sorted(admitted.items()):
        work = Path(record['params']).parent
        conservation = column_conservation(work / 'full.fa', work / 'full.weights')
        if len(conservation) != record['length']:
            raise SystemExit(f'{assay}: the alignment has {len(conservation)} columns against a '
                             f'{record["length"]}-residue wild type; columns are not positions')
        depth = effective_depth_per_site(work, record['length'])
        if depth < args.min_neff_per_site:
            shallow[assay] = depth
            continue
        if args.max_neff_per_site is not None and depth >= args.max_neff_per_site:
            excluded_above_ceiling[assay] = depth
            continue
        keys, built = pair_blocks(record, conservation)
        blocks[assay] = {'keys': keys, **built}
        covered = {}
        for arm in arms:
            entry = arm['files'].get(assay)
            archive = (None if entry is None
                       else arm['directory'] / 'archives' / Path(entry['file']).name)
            if archive is None or not archive.exists():
                continue
            measured = arm_pair_responses(archive, record['geometry'], record['pairs'],
                                          cohort_rows[assay], assay=assay,
                                          paradigm=arm['paradigm'])
            seen = [position for position, key in enumerate(keys) if key in measured['values']]
            if len(seen) < 2 or len(set(built['label'][seen].tolist())) < 2:
                continue
            model[arm['arm']][assay] = {
                'rows': np.asarray(seen, dtype=np.int64),
                'column': np.asarray([[measured['values'][keys[position]]]
                                      for position in seen], dtype=np.float64)}
            covered[arm['arm']] = {'pairs': len(seen),
                                   'admitted_variants': measured['admitted_variants']}
        support[assay] = {
            'family': record['family'], 'entry': record['entry'], 'length': record['length'],
            'pairs': len(keys), 'contacts': int(built['label'].sum()),
            'params': record['params'], 'params_sha256': record['params_sha256'],
            'effective_depth_per_site': depth,
            'msa_sha256': sha256_file(work / 'full.fa'),
            'weights_sha256': sha256_file(work / 'full.weights'),
            'arm_coverage': covered,
        }

    admitted_arms, refused_arms = [], []
    for arm in arms:
        carried = sorted(model[arm['arm']])
        if len(carried) >= MINIMUM_BOOTSTRAP_UNITS:
            admitted_arms.append({**arm, 'references': carried})
        else:
            refused_arms.append({'arm': arm['arm'], 'paradigm': arm['paradigm'],
                                 'references_covered': len(carried), 'reason': (
                                     f'{len(carried)} references against the '
                                     f'{MINIMUM_BOOTSTRAP_UNITS}-unit floor; the arm is named '
                                     'rather than estimated and does not thin the support')})
    if not admitted_arms:
        raise SystemExit('no arm clears the support floor; there is no family to fit')

    try:
        primary = evaluate_panel(admitted_arms, blocks, model, support)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    complete = primary['complete']
    shared = primary['shared']
    declared = primary['declared']
    states = primary['states']
    payload_strata = primary['strata']

    # ---- full-alphabet complete-case secondary summary ----
    total_eligible_pairs = sum(entry['pairs'] for entry in support.values())
    pair_fraction_by_arm = {
        arm['arm']: (
            sum(support[assay]['arm_coverage'].get(arm['arm'], {}).get('pairs', 0)
                for assay in support) / total_eligible_pairs
            if total_eligible_pairs else 0.0
        )
        for arm in arms
    }
    full_alpha_arms, thin_bpe_excluded = [], []
    for arm in admitted_arms:
        fraction = pair_fraction_by_arm.get(arm['arm'], 0.0)
        if fraction >= FULL_ALPHABET_MIN_PAIR_FRACTION:
            full_alpha_arms.append(arm)
        else:
            thin_bpe_excluded.append({'arm': arm['arm'], 'pair_coverage_fraction': fraction})

    if not full_alpha_arms:
        full_alphabet_secondary = {
            'rule': (f'admitted arms covering at least {FULL_ALPHABET_MIN_PAIR_FRACTION:.2f} of '
                     'eligible residue pairs across admitted references'),
            'min_pair_fraction': FULL_ALPHABET_MIN_PAIR_FRACTION,
            'pair_coverage_fraction_by_arm': pair_fraction_by_arm,
            'thin_bpe_arms_excluded': thin_bpe_excluded,
            'refused': 'no admitted arm cleared the full-alphabet pair coverage threshold',
        }
    elif len(full_alpha_arms) == len(admitted_arms):
        full_alphabet_secondary = {
            'rule': (f'admitted arms covering at least {FULL_ALPHABET_MIN_PAIR_FRACTION:.2f} of '
                     'eligible residue pairs across admitted references'),
            'min_pair_fraction': FULL_ALPHABET_MIN_PAIR_FRACTION,
            'pair_coverage_fraction_by_arm': pair_fraction_by_arm,
            'thin_bpe_arms_excluded': thin_bpe_excluded,
            'identical_to_primary': True,
        }
    else:
        secondary = evaluate_panel(full_alpha_arms, blocks, model, support)
        full_alphabet_secondary = {
            'rule': (f'admitted arms covering at least {FULL_ALPHABET_MIN_PAIR_FRACTION:.2f} of '
                     'eligible residue pairs across admitted references, fitted and banded on '
                     'their own larger complete-case reference intersection'),
            'min_pair_fraction': FULL_ALPHABET_MIN_PAIR_FRACTION,
            'pair_coverage_fraction_by_arm': pair_fraction_by_arm,
            'thin_bpe_arms_excluded': thin_bpe_excluded,
            'family': {
                'size_per_stratum': secondary['declared'],
                'arms': [arm['arm'] for arm in full_alpha_arms],
            },
            'references_offered': secondary['complete'],
            'banded_families': secondary['shared'],
            'realised_depth': {assay: support[assay]['effective_depth_per_site']
                               for assay in secondary['complete']},
            'per_arm': {
                name: {'references': state['references'], 'pairs': state['pairs'],
                       'contacts': state['contacts'], 'families': len(state['order'])}
                for name, state in secondary['states'].items()
            },
            'strata': secondary['strata'],
        }

    write_json(args.out, {
        'schema': 'contact_bidirectional_v1',
        'experiment': 'R2/R3 bidirectional ablation: pair propagation against explicit couplings',
        'estimand': ('held-out prediction of whether a residue pair is in three-dimensional '
                     'contact, on the pair population the contact definition admits'),
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'contact_definition': {'cutoff_angstrom': float(args.contact_angstrom),
                               'min_separation': int(args.min_separation)},
        'family': {'size_per_stratum': declared,
                   'rule': '2 directions x rungs x admitted arms, banded as one family per stratum',
                   'directions': list(bidir.DIRECTIONS),
                   'designs': {direction: list(pair)
                               for direction, pair in bidir.PRIMARY_DESIGNS.items()},
                   'rungs': {rung: list(names) for rung, names in RUNGS.items()},
                   'arms': [arm['arm'] for arm in admitted_arms]},
        'resampling_unit': 'biological family; one reference per family on this cohort',
        'fitter': {'source': 'src/capability/extensions/phenotype_breadth.py',
                   'outer_folds': breadth.OUTER_FOLDS, 'inner_folds': breadth.INNER_FOLDS,
                   'split_seeds': list(breadth.SPLIT_SEEDS),
                   'weighting': 'family-equal, reference-equal within family, pair-equal within '
                                'reference, through phenotype_breadth.row_weights',
                   'seed_rule': 'the three split seeds are averaged within group before any '
                                'resample, as the lanes do'},
        'excluded_nuisance': EXCLUDED_NUISANCE,
        'strata': payload_strata,
        'full_alphabet_secondary': full_alphabet_secondary,
        'depth': {'min_neff_per_site': float(args.min_neff_per_site),
                  'max_neff_per_site': (None if args.max_neff_per_site is None
                                        else float(args.max_neff_per_site)),
                  'source': 'the full.weights file each selected.params fit consumed',
                  'references_below_floor': shallow,
                  'references_at_or_above_ceiling': excluded_above_ceiling,
                  'realised': {assay: support[assay]['effective_depth_per_site']
                               for assay in complete}},
        'support': {'rule': ("each arm's three designs share that arm's own pair support; the "
                             'bootstrap axis is the families every admitted arm carries'),
                    'references_offered': complete, 'references_admitted': sorted(blocks),
                    'banded_families': shared,
                    'per_arm': {name: {'references': state['references'], 'pairs': state['pairs'],
                                       'contacts': state['contacts'],
                                       'families': len(state['order'])}
                                for name, state in states.items()},
                    'per_reference': support, 'references_refused': refused_references},
        'arms_refused': refused_arms,
        'arms_omitted': [{'arm': item.split('=', 1)[0], 'reason': item.split('=', 1)[1]}
                         for item in args.omitted_arm],
        'sources': {
            'cohort': {'path': str(args.cohort), 'sha256': sha256_file(args.cohort)},
            'structures': {'path': str(args.structures), 'sha256': sha256_file(args.structures)},
            'coverage': {'path': str(args.coverage), 'sha256': sha256_file(args.coverage)},
            'params_roots': [str(root) for root in args.params_root],
            'extractions': [{'arm': arm['arm'], 'directory': str(arm['directory']),
                             'index_sha256': arm['index_sha256'],
                             'paradigm': arm['paradigm']} for arm in arms],
        },
    })


if __name__ == '__main__':
    main()
