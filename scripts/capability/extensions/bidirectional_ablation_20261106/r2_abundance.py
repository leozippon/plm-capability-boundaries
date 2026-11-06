#!/usr/bin/env python3
"""Replay the frozen abundance three-arm fits and form the declared paired family.

The three arms on identical rows, folds and tuning are the joint readout ``CM``,
the frozen traditional controls ``C`` and the pure likelihood readout ``M``.
``C`` is the archived qualified control set -- ``ident``, ``geom``, ``comp``,
``chem``, ``prof``, ``prof2`` and the training-fold response ``G`` -- plus, for
the 23 of 33 arms whose archived primary baseline is ``S_T``, that arm's own
tokenisation block ``T``. ``T`` is a model-specific tokenisation nuisance, so it
is carried by the controls and never folded into ``M``: the reverse direction
asks what the controls add to the pure scalar likelihood difference, not what
they add to a control-conditioned likelihood readout.

Nothing here re-runs model inference, re-qualifies a control, re-ranks the target
or rewrites ``normalized_fitness``. The scalar likelihood difference and the
tokenisation block are read from the archived extraction's own arrays exactly as
the original fit read them, and every arm is first reproduced against the
archived ``fit_<arm>.json`` record at the declared ``atol=1e-9``/``rtol=1e-8``:
a disagreement fails that arm instead of being summarised. The original recipe
is used unchanged -- five outer groups, four inner splits, inner seed
``seed + 100 + outer``, penalties ``.01 .1 1 10 100`` with ties broken toward
stronger regularisation, float64 weighted ridge -- including the accepted
limitation that ``G`` is cross-fitted on the outer training partition and then
reused inside the inner tuning of that same outer fold.

The primary family is the joint maximum absolute *centered bootstrap deviation*
in the endpoint's native units: 66 columns = 2 directions x 33 arms, read
separately for each declared metric. Two metrics are read from the identical
out-of-fold predictions on the identical groups and weights: the group-equal mean
squared error in squared-normalised-fitness units, and the grouped within-domain
Spearman rank correlation, because a rank improvement and a level improvement are
different claims and the rank reading is the one this endpoint's original
conclusion rests on. A squared error is a loss and a rank correlation is a score,
so the family is stated as ``MSE(reduced) - MSE(joint)`` and
``rho(joint) - rho(reduced)``: positive means the joint arm is better in both.
The project's fixed-standard-error studentised maxT family over the same draws is
retained as an explicitly labelled sensitivity for each metric, because the two
give different bands and different positive counts.

The archived extraction is hashed file by file against its manifest before any
fit, so a compacted re-export or a partially restored pool cannot stand in for
the original per-domain archives: this stage has no compact-format path, and a
missing score, a changed support or an archived number that does not reproduce
fails the arm rather than being substituted.

Entry points: fitting arms (``--arms``), an explicit pilot (``--pilot``, one
model on all three split seeds, bound to the contract and to the archived
inputs), a cheap input-availability check (``--check-inputs``) and the summary
(``--summarize``), which refuses to write a completed panel unless a real pilot
reproduced the archived fits for every arm it covers and every requested arm has
its own reproduced fit record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Must precede NumPy/Torch imports in the executable process: one BLAS thread per
# worker keeps the float64 eigendecomposition reproducible at its declared setting.
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.capability.interactions.run_residual_panel import simultaneous_bands
from scripts.capability.reporting import replay_phenotype_predictions as replay
from src.capability.core.io import sha256_file
from src.capability.extensions import bidirectional
from src.capability.interactions.pairwise_epistasis import ROSTER
from src.capability.mutation import external_confirmation as abundance
from src.capability.stability import stability_gate

DEFAULT_ROOT = ROOT
DEFAULT_EXTRACTION = ROOT / 'results/external_baseline/20260924085613_5f8d32cf0368'
DEFAULT_OUT = ROOT / 'results/extensions/bidirectional_ablation_20261106/r2/abundance-v2'
#: The earlier abundance attempt; never written by this stage.
HISTORICAL_OUT = ROOT / 'results/extensions/bidirectional_ablation_20261106/r2/abundance'
#: Relative to ``--root``: the frozen cohort, profiles, controls and archives plan.
SOURCE = 'results/R1/external_confirmation_20260924'
#: Directory names accepted under ``--extraction-root``, in priority order.
EXTRACTION_LAYOUTS = ('ec-{arm}', '{arm}')

SEEDS = tuple(abundance.SPLIT_SEEDS)
ROLES = ('C', 'M', 'CM')
#: Direction name -> (reduced arm, joint arm). Both directions are paired on the
#: same joint arm and the same held groups.
DESIGNS = {'model_given_controls': ('C', 'CM'), 'controls_given_model': ('M', 'CM')}
#: The endpoint readings. Both are formed from the identical out-of-fold
#: predictions, on the identical held groups and the identical weights; they differ
#: only in the per-group statistic and in its sign convention, because a squared
#: error is better when it is smaller and a rank correlation when it is larger.
METRICS = {
    'squared_error': {
        'field': 'mse', 'to_loss': 1.,
        'units': ('squared normalised fitness; group-equal mean squared error with domains, '
                  'then sites, then variants weighted equally inside a group'),
        'direction': 'reduced minus joint: positive when the joint arm lowers the error'},
    'ranking': {
        'field': 'rank', 'to_loss': -1.,
        'units': ('dimensionless within-domain Spearman rho, averaged over the domains of a '
                  'group; groups equal'),
        'direction': 'joint minus reduced: positive when the joint arm raises the rank correlation'},
}
ATOL, RTOL = replay.ATOL, replay.RTOL
PILOT_SCHEMA = 'r2_abundance_pilot_v3'
PILOT_TOLERANCE = bidirectional.PILOT_TOLERANCE
FIT_SCHEMA = 'r2_abundance_fit_v3'
CONTRACT_SCHEMA = 'bidirectional_abundance_contract_v3'
PANEL_SCHEMA = 'bidirectional_abundance_v3'
#: Fully fixed support of the archived endpoint, asserted rather than assumed.
SUPPORT = {'rows': 109568, 'groups': 96, 'domains': 428}
#: The archived arms whose penalties, widths and metrics a pilot must reproduce.
#: The archive fitted S/S2/R designs per arm and never a pure likelihood-only arm,
#: so ``M`` has no archived counterpart and one is never invented for it.
REUSED = ('C', 'CM')
HEX64 = re.compile(r'^[0-9a-f]{64}$')

if tuple(DESIGNS) != bidirectional.DIRECTIONS:
    raise RuntimeError('the paired family must read the two declared directions in order')

OUT = DEFAULT_OUT
STATE = {}


# --------------------------------------------------------------------------- #
# Small shared helpers.
# --------------------------------------------------------------------------- #

def write(path, value) -> None:
    """Atomic JSON with no NaN, so a partial record is never left in place."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def map_digest(mapping: dict) -> str:
    """Content digest of a name -> digest map, independent of insertion order."""

    return digest_text('\n'.join(f'{key}:{mapping[key]}' for key in sorted(mapping)))


def code_hashes() -> dict:
    """Every module whose bytes can move a reproduced fit or a declared band."""

    import scripts.capability.interactions.run_residual_panel as residual
    import scripts.capability.stability.extract_stability_singles as singles
    from src.capability.interactions import pairwise_epistasis
    from src.capability.readouts import readout_analysis
    paths = [Path(__file__), Path(replay.__file__), Path(abundance.__file__),
             Path(stability_gate.__file__), Path(readout_analysis.__file__),
             Path(pairwise_epistasis.__file__), Path(bidirectional.__file__),
             Path(residual.__file__), Path(singles.__file__)]
    return {str(path.resolve().relative_to(ROOT)): sha256_file(path) for path in paths}


def resource_record(workers) -> dict:
    """Refuse an unpinned BLAS pool and record what the run actually had."""

    from threadpoolctl import threadpool_info
    pools = [{key: pool[key] for key in ('user_api', 'internal_api', 'num_threads', 'version')}
             for pool in threadpool_info() if pool['user_api'] in ('blas', 'openmp')]
    if any(pool['num_threads'] != 1 for pool in pools):
        raise ValueError('a loaded BLAS/OpenMP pool is not pinned to one thread; start a fresh CLI process')
    return {'workers': workers, 'blas_threads_per_worker': 1, 'gpu': False,
            'fit_device': 'cpu', 'model_inference': False,
            'python': platform.python_version(), 'numpy': np.__version__,
            'torch': torch.__version__,
            'cpus': os.cpu_count(),
            'free_disk_bytes': shutil.disk_usage(ROOT).free,
            'threadpools': pools,
            'launcher_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


# --------------------------------------------------------------------------- #
# The three arms, as a declaration over the frozen control set.
# --------------------------------------------------------------------------- #

def archived_baseline(reference: dict, arm: str) -> str:
    """The archived primary baseline, read and never re-derived by requalification."""

    baseline = reference.get('matched_baseline', {}).get('primary')
    if baseline not in ('S', 'S_T'):
        raise replay.ReplayError(f'{arm}: archived primary baseline is {baseline!r}; only the frozen S or S_T '
                                 'baseline is reused and this stage never re-qualifies the tokenisation block')
    return baseline


def control_columns(baseline: str, qualified) -> tuple[str, ...]:
    """The ``C`` columns: the frozen qualified set, plus this arm's own ``T``.

    ``T`` enters only when the archived arm qualified it, and it enters the
    controls. ``G`` is already part of the qualified set and stays there: it is a
    training-fold derived nonlinear response, not a model quantity.
    """

    qualified = tuple(qualified)
    if baseline not in ('S', 'S_T'):
        raise ValueError(f'unknown archived baseline {baseline!r}')
    if 'T' in qualified:
        raise ValueError('the frozen qualified control set already carries the tokenisation block')
    return qualified + (('T',) if baseline == 'S_T' else ())


def fold_designs(baseline: str, qualified) -> tuple[dict, dict]:
    """The three fitted designs, and the role name each archived design plays.

    ``M`` is one column: the arm's own scalar likelihood difference against the
    wild type. No control block and no tokenisation block reaches it, so the
    reverse direction reads the pure model readout.
    """

    columns = control_columns(baseline, qualified)
    designs = {baseline: columns, baseline + '_M': (*columns, 'M'), 'M': ('M',)}
    roles = {'C': baseline, 'M': 'M', 'CM': baseline + '_M'}
    if len(set(designs)) != len(ROLES):
        raise ValueError(f'design names collide for baseline {baseline!r}: {sorted(designs)}')
    if list(designs['M']) != ['M']:
        raise ValueError('the pure likelihood arm is exactly one column: the scalar likelihood difference')
    return designs, roles


def archived_expectations(reference: dict, seed: int, roles: dict) -> dict:
    """The archived penalties and design widths of the reused arms, per outer fold.

    Only ``C`` and ``CM`` are reused from the archive, which fitted ``S``, ``S2``
    and ``R`` designs per arm and never a pure likelihood-only design. ``M``
    therefore has no archived penalty or width, and none is invented: it runs on
    the same folds, the same penalty grid and the same support, and its own
    selected penalty is reported as the new fit's.
    """

    previous = reference['per_seed'][str(seed)]
    historical = previous.get('folds')
    expectations = {}
    for role in REUSED:
        design = roles[role]
        if historical is not None:
            expectations[role] = [{'alpha': fold['alpha'][design],
                                   'dimensions': fold['dimensions'][design]}
                                  for fold in historical]
        else:
            expectations[role] = [{'alpha': previous['alpha'][index][design],
                                   'dimensions': previous['dimensions'][design]}
                                  for index in range(len(previous['alpha']))]
    return expectations


def archived_nuisance_alpha(reference: dict, seed: int, fold: int):
    return reference['per_seed'][str(seed)]['nuisance'][fold]['selected_alpha']


def reproduced_expectations(folds: list[dict], expectations: dict, *, arm: str, seed: int) -> None:
    """Refuse a fold whose reproduced penalty or width differs from the archived one.

    Only the reused arms are checked. The pure likelihood arm has no archived
    number to be checked against, so nothing is compared for it; loosening the
    check for the reused arms would instead hide a changed fit.
    """

    for role in REUSED:
        expected = expectations[role]
        if len(expected) != len(folds):
            raise replay.ReplayError(f'{arm} seed {seed}: archived {role} covers {len(expected)} '
                                     f'outer folds, the fit covers {len(folds)}')
        for fold, want in zip(folds, expected):
            if fold['alpha'][role] != want['alpha']:
                raise replay.ReplayError(f'{arm} seed {seed} fold {fold["fold"]}: reproduced {role} '
                                         'penalty differs from the archived one')
            if fold['dimensions'][role] != want['dimensions']:
                raise replay.ReplayError(f'{arm} seed {seed} fold {fold["fold"]}: reproduced {role} '
                                         'design width differs from the archived one')


def group_ranks(panel: dict, prediction) -> list:
    """Per-group within-domain Spearman, from the endpoint's own grouped statistic.

    This calls the same private helper the archived ``summary_metrics`` reads
    through ``raw_spearman`` and ``spearman_increment``; the ranked reading is not
    reimplemented here. An undefined group stays undefined (``None``) so the family
    can hold every arm to one common support instead of silently dropping it from
    one arm only.
    """

    groups = sorted(set(panel['group'].tolist()))
    values = abundance._domain_spearman(panel, np.asarray(prediction, dtype=float))
    if len(values) != len(groups):
        raise ValueError(f'the grouped rank statistic covers {len(values)} groups, not the '
                         f'{len(groups)} groups of the panel')
    return [None if value is None or not np.isfinite(value) else float(value) for value in values]


# --------------------------------------------------------------------------- #
# Frozen inputs: one shared panel, then one arm at a time.
# --------------------------------------------------------------------------- #

def shared_paths(root) -> dict:
    base = Path(root) / SOURCE
    return {'cohort': base / 'cohort.json', 'profiles': base / 'profile_features.npz',
            'controls': base / 'controls_qualification.json',
            'plan': base / 'extraction_plan.json'}


def shared_hashes() -> dict:
    """Digests of every frozen input this stage reads, including all 33 fit records."""

    state = initialise()
    root = Path(state['root'])
    hashes = dict(state['hashes'])
    hashes['reference_fits'] = {}
    for arm in sorted(ROSTER):
        path = replay.input_paths(root, 'abundance', arm)['reference']
        if not path.is_file():
            raise replay.ReplayError(f'missing archived fit record {path}')
        hashes['reference_fits'][arm] = replay.digest(path)
    return hashes


def initialise(root=None, extraction=None) -> dict:
    """Load the frozen cohort, profiles and control panel once per process."""

    root = str(Path(root or STATE.get('root') or DEFAULT_ROOT).resolve())
    extraction = str(Path(extraction or STATE.get('extraction') or DEFAULT_EXTRACTION).resolve())
    if STATE.get('panel') is not None and STATE['root'] == root and STATE['extraction'] == extraction:
        return STATE
    STATE.clear()
    torch.set_num_threads(1)
    paths = shared_paths(root)
    hashes = {key: replay.digest(path) for key, path in paths.items()}
    cohort = replay.read_json(paths['cohort'])
    if cohort['schema'] != 'external_confirmation_cohort_v1':
        raise replay.ReplayError('wrong source cohort schema')
    units = cohort['domains']
    if len(units) != SUPPORT['domains']:
        raise replay.ReplayError(f'cohort holds {len(units)} domains, not the frozen {SUPPORT["domains"]}')
    records = replay.registry(cohort, 'abundance', hashes['cohort'])
    if len(records) != SUPPORT['rows']:
        raise replay.ReplayError(f'cohort holds {len(records)} variants, not the frozen {SUPPORT["rows"]}')
    controls = replay.read_json(paths['controls'])
    profiles, _ = stability_gate.load_profiles(paths['profiles'], {u['name']: u['wildtype'] for u in units})
    panel = abundance.build_panel(units, profiles)
    groups = sorted(set(panel['group'].tolist()))
    if len(groups) != SUPPORT['groups']:
        raise replay.ReplayError(f'panel holds {len(groups)} groups, not the frozen {SUPPORT["groups"]}')
    identity = abundance.row_identity(panel['group'], panel['site'], panel['target'])
    if identity != controls['row_identity_sha256']:
        raise replay.ReplayError('panel row identity differs from the frozen control qualification')
    STATE.update(root=root, extraction=extraction, paths={k: str(v) for k, v in paths.items()},
                 hashes=hashes, cohort=cohort, controls=controls, units=units, records=records,
                 panel=panel, groups=groups, identity=identity,
                 sample_identity={'row_identity_sha256': identity,
                                  'cohort_sha256': hashes['cohort'],
                                  'controls_sha256': hashes['controls'],
                                  'profile_sha256': hashes['profiles'],
                                  'plan_sha256': hashes['plan'],
                                  'rows': len(records), 'groups': len(groups),
                                  'domains': len(units)})
    return STATE


def locate_extraction(extraction_root, arm: str) -> Path:
    """The arm's restored archive directory, or an explicit refusal to invent one."""

    for layout in EXTRACTION_LAYOUTS:
        directory = Path(extraction_root) / layout.format(arm=arm)
        if (directory / f'manifest_{arm}.json').is_file():
            return directory
    raise replay.ReplayError(
        f'{arm}: no extraction directory with manifest_{arm}.json under {extraction_root} '
        f'(looked for {", ".join(layout.format(arm=arm) for layout in EXTRACTION_LAYOUTS)}); '
        'the original per-arm archive must be restored from the frozen plan, never regenerated '
        'or replaced by a mismatched substitute')


def prepare(arm: str) -> dict:
    """Validate one arm's archived inputs and return its three design declarations."""

    state = initialise()
    root = Path(state['root'])
    reference_path = replay.input_paths(root, 'abundance', arm)['reference']
    reference = replay.read_json(reference_path)
    hashes = {**state['hashes'], 'reference': replay.digest(reference_path)}
    baseline, columns = replay.validate_reference('abundance', arm, state['records'],
                                                  state['controls'], reference, hashes)
    derived = control_columns(baseline, state['controls']['qualified_control_set'])
    if tuple(columns) != derived:
        raise replay.ReplayError(f'{arm}: archived control columns {columns} are not the frozen '
                                 f'qualified set plus its own tokenisation block')
    directory = locate_extraction(state['extraction'], arm)
    plan, plan_sha256 = replay.canonical_plan(root, 'abundance', hashes['cohort'], state['units'])
    manifest, features = replay.validate_features(directory, arm, hashes['cohort'], state['units'],
                                                  plan=plan, expected_plan_sha256=plan_sha256)
    blocks = replay.load_likelihood_blocks(directory, state['units'], manifest)
    replay.validate_panel(state['panel'], state['records'], blocks)
    if set(blocks) != {'M', 'T'} or blocks['M'].shape != (SUPPORT['rows'], 1):
        raise replay.ReplayError(f'{arm}: the archived extraction does not carry one scalar likelihood '
                                 'difference per variant')
    designs, roles = fold_designs(baseline, state['controls']['qualified_control_set'])
    return {'arm': arm, 'reference': reference, 'baseline': baseline, 'columns': list(columns),
            'designs': {name: list(cols) for name, cols in designs.items()}, 'roles': roles,
            'blocks': blocks, 'directory': str(directory),
            'first_stage': tuple(block for block in state['controls']['qualified_control_set'] if block != 'G'),
            'features': features, 'plan_sha256': plan_sha256}


def fit_seed(prepared: dict, seed: int) -> dict:
    """One seed's three fits, with the two reused arms reproduced against the archive."""

    state = initialise()
    panel, roles, reference = state['panel'], prepared['roles'], prepared['reference']
    designs = {name: tuple(columns) for name, columns in prepared['designs'].items()}
    with replay.capture_partitions(abundance) as partitions:
        result = abundance.fold_predictions(panel, prepared['blocks'], designs, seed=seed,
                                            first_stage=prepared['first_stage'], device='cpu')
    heldout, comparisons = replay.validate_fit(result, reference, 'abundance', seed,
                                               prepared['baseline'], panel, partitions)
    summary = replay.summary_metrics(abundance, panel, result, prepared['baseline'], 'abundance')
    comparisons += replay.compare_metrics(summary, reference['per_seed'][str(seed)], 'abundance')
    disagreements = [record for record in comparisons if not record['agrees']]
    if disagreements:
        raise replay.ReplayError(f"{prepared['arm']} seed {seed}: {len(disagreements)} archived metrics "
                                 f"disagree, worst {max(record['metric'] for record in disagreements)}")
    predictions = result['predictions']
    arguments = [panel['target'], None, panel['group'], panel['domain'], panel['site']]
    measured = {'mse': {}, 'rank': {}}
    for role in ROLES:
        arguments[1] = predictions[roles[role]]
        labels, errors = abundance.group_errors(*arguments)
        if labels.tolist() != state['groups']:
            raise replay.ReplayError(f"{prepared['arm']} seed {seed}: {role} group support differs")
        measured['mse'][role] = errors.tolist()
        # Same out-of-fold predictions, same groups, same nesting; only the per-group
        # statistic differs. An undefined rank correlation stays undefined.
        measured['rank'][role] = group_ranks(panel, predictions[roles[role]])
    expectations = archived_expectations(reference, seed, roles)
    folds = [{'fold': fold['fold'], 'held_groups': fold['held_groups'],
              'training_groups': fold['training_groups'],
              'alpha': {role: fold['alpha'][roles[role]] for role in ROLES},
              'alpha_expected': {role: expectations[role][fold['fold']]['alpha'] for role in REUSED},
              'dimensions': {role: fold['dimensions'][roles[role]] for role in ROLES},
              'dimensions_expected': {role: expectations[role][fold['fold']]['dimensions']
                                      for role in REUSED}}
             for fold in result['folds']]
    reproduced_expectations(folds, expectations, arm=prepared['arm'], seed=seed)
    nuisance = [dict(record, archived_selected_alpha=archived_nuisance_alpha(reference, seed, index))
                for index, record in enumerate(result['nuisance'])]
    if any(record['selected_alpha'] != record['archived_selected_alpha'] for record in nuisance):
        raise replay.ReplayError(f"{prepared['arm']} seed {seed}: the reproduced response penalty "
                                 'differs from the archived one')
    archived = [record for record in comparisons
                if record['metric'].startswith(('primary_', 'outer_'))]
    return {'record': {
                'seed': seed, 'folds': folds, 'nuisance': nuisance,
                'mse': measured['mse'], 'rank': measured['rank'],
                'summary': summary, 'comparisons': comparisons,
                'archived_metric_max_absolute_difference': {
                    **{f'C|{record["metric"]}': record.get('max_absolute_difference')
                       for record in archived},
                    'all': max((record.get('max_absolute_difference') or 0.)
                               for record in comparisons)},
                'disagreements': len(disagreements)},
            'predictions': {role: predictions[roles[role]] for role in ROLES},
            'heldout': heldout}


def run_arm(arm: str, contract_sha256: str) -> dict:
    """Fit or reuse one arm; a failure is recorded, never silently dropped."""

    started = time.monotonic()
    try:
        state = initialise()
        existing = reusable(arm, contract_sha256)
        if existing is not None:
            return existing
        prepared = prepare(arm)
        per_seed, predictions, folds_by_seed = {}, {}, {}
        for seed in SEEDS:
            outcome = fit_seed(prepared, seed)
            entry = dict(outcome['record'])
            entry['heldout_folds'] = outcome['heldout'].tolist()
            per_seed[str(seed)] = entry
            for role, values in outcome['predictions'].items():
                predictions[f'{seed}|{role}'] = values
            folds_by_seed[str(seed)] = outcome['heldout']
        oof = OUT / 'oof' / f'{arm}.npz'
        oof.parent.mkdir(parents=True, exist_ok=True)
        panel = state['panel']
        np.savez_compressed(oof, target=panel['target'], group=panel['group'],
                            domain=panel['domain'], site=panel['site'],
                            sample_id=np.asarray([row['sample_id'] for row in state['records']]),
                            folds=np.stack([folds_by_seed[str(seed)] for seed in SEEDS]),
                            **predictions)
        record = {
            'schema': FIT_SCHEMA, 'pilot': False, 'arm': arm, 'status': 'complete',
            'contract_sha256': contract_sha256, 'code_sha256': code_hashes(),
            'seconds': time.monotonic() - started,
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'baseline': prepared['baseline'], 'roles': prepared['roles'],
            'designs': prepared['designs'],
            'qualified_control_set': list(state['controls']['qualified_control_set']),
            'control_columns': prepared['columns'],
            'support': {'rows': int(len(panel['target'])), 'groups': len(state['groups']),
                        'domains': int(len(set(panel['domain'].tolist())))},
            'groups': state['groups'],
            'sample_identity': dict(state['sample_identity']),
            'hashes': {'cohort': state['hashes']['cohort'],
                       'controls': state['hashes']['controls'],
                       'profiles': state['hashes']['profiles'],
                       'plan': state['hashes']['plan'],
                       'reference': replay.digest(replay.input_paths(
                           Path(state['root']), 'abundance', arm)['reference']),
                       'extraction_manifest': prepared['features']['manifest'],
                       'extraction_directory': prepared['directory'],
                       'archives': prepared['features']['archives'],
                       'archives_digest': map_digest(prepared['features']['archives']),
                       'archives_count': len(prepared['features']['archives'])},
            'seeds': per_seed,
            'oof': {'file': oof.name, 'sha256': replay.digest(oof)},
            'validation': {'atol': ATOL, 'rtol': RTOL,
                           'archived_metric_max_absolute_difference':
                               max(entry['archived_metric_max_absolute_difference']['all']
                                   for entry in per_seed.values()),
                           'disagreements': 0},
        }
        write(OUT / 'fits' / f'{arm}.json', record)
        print(json.dumps({'finished': arm, 'seconds': record['seconds']}), flush=True)
        return record
    except Exception as error:  # noqa: BLE001 - the failure is the record
        failure = {'schema': FIT_SCHEMA, 'pilot': False, 'arm': arm, 'status': 'failed',
                   'contract_sha256': contract_sha256,
                   'seconds': time.monotonic() - started, 'error': message(error),
                   'action': 'restore the archived extraction for this arm and rerun into a new '
                             '--out; no substitute scores, no implicit refit of a different design'}
        try:
            write(OUT / 'fits' / f'{arm}.json', failure)
        except Exception:  # noqa: BLE001 - a recorded failure is best effort, exit code carries it
            pass
        print(json.dumps({'failed': arm, 'error': failure['error']}), flush=True)
        return failure


def message(error: Exception) -> str:
    """A failure line that never carries library text or remote paths."""

    if isinstance(error, replay.ReplayError):
        return str(error)
    return (f'{type(error).__name__}: input or fitting failure; inspect the supplied extraction '
            'schema, arrays and the archived fit record')


def reusable(arm: str, contract_sha256: str):
    """A stored record for the same contract, verified against its OOF digest."""

    path = OUT / 'fits' / f'{arm}.json'
    if not path.is_file():
        return None
    record = replay.read_json(path)
    if record.get('schema') != FIT_SCHEMA or record.get('arm') != arm:
        raise ValueError(f'{path} is not a {FIT_SCHEMA} record of {arm}')
    if record.get('contract_sha256') != contract_sha256:
        raise ValueError(f'{path} was written under a different contract; rerun into a new --out '
                         'instead of mixing declarations')
    if record.get('status') != 'complete':
        return None
    if record.get('support') != dict(SUPPORT):
        raise ValueError(f'{arm}: the stored fit covers {record.get("support")}, not the frozen support')
    if record['hashes'].get('archives_count') != SUPPORT['domains']:
        raise ValueError(f'{arm}: the stored fit hashed {record["hashes"].get("archives_count")} '
                         f'archives, not the {SUPPORT["domains"]} domains of the frozen cohort')
    oof = OUT / 'oof' / f"{arm}.npz"
    if not oof.is_file() or replay.digest(oof) != record['oof']['sha256']:
        raise ValueError(f'{arm}: the OOF archive is missing or differs from its recorded digest')
    return record


def worker_init(out, root, extraction) -> None:
    """Per-worker result directory and shared panel, explicit for any start method."""

    global OUT
    OUT = Path(out)
    initialise(root, extraction)


# --------------------------------------------------------------------------- #
# The declared contract and the reproduction gate.
# --------------------------------------------------------------------------- #

def contract() -> dict:
    """One declaration of support, arms, seeds, family and units; arms-independent.

    The arms an invocation asks for live in ``run.json`` so that a one-model pilot
    and the full 33-arm run bind to the same frozen contract.
    """

    state = initialise()
    return {
        'schema': CONTRACT_SCHEMA,
        'support': dict(SUPPORT, seeds=[str(seed) for seed in SEEDS]),
        'designs': {'C': list(state['controls']['qualified_control_set']) + ['T when archived as S_T'],
                    'M': ['M'], 'CM': list(state['controls']['qualified_control_set'])
                    + ['T when archived as S_T', 'M']},
        'archived_counterparts': {'C': 'reproduced exactly (penalty, width, response, metrics)',
                                  'CM': 'reproduced exactly (penalty, width, response, metrics)',
                                  'M': 'new fit: the archive never fitted a pure likelihood-only '
                                       'arm, so no archived penalty or width exists and none is '
                                       'invented; the same folds, penalty grid and support are used'},
        'roles': {'C': 'frozen qualified controls plus this arm\'s own qualified tokenisation '
                       'block; G is a training-fold derived nonlinear response and belongs here',
                  'M': 'one column: this arm\'s scalar likelihood difference against the wild type',
                  'CM': 'C and M, on identical rows, folds and tuning'},
        'T': ('model-specific tokenisation nuisance: part of C when the archived arm qualified it, '
              'never part of M'),
        'seeds': [str(seed) for seed in SEEDS],
        'recipe': {'outer_splits': abundance.OUTER_SPLITS, 'inner_splits': abundance.INNER_SPLITS,
                   'inner_seed': 'seed + 100 + outer', 'alphas': list(abundance.ALPHAS),
                   'tie_break': 'stronger regularisation',
                   'estimator': 'float64 weighted ridge, unpenalised intercept, training-row scaling',
                   'weighting': METRICS['squared_error']['units']},
        'metrics': {name: {'units': specification['units'],
                           'direction': specification['direction']}
                    for name, specification in METRICS.items()},
        'bootstrap': {'draws': bidirectional.DRAWS, 'seed': bidirectional.SEED,
                      'confidence': bidirectional.CONFIDENCE, 'method': bidirectional.NATIVE_METHOD,
                      'units': 'native', 'family': f'{bidirectional.FAMILY_SIZE} contrasts = '
                      '2 directions x 33 arms, read separately for every metric',
                      'unit': '96 families; the three split seeds are averaged within a group '
                              'before the 96 groups enter equally',
                      'support': 'a group enters a metric only when all three arms and all three '
                                 'split seeds define it there; undefined groups are reported'},
        'sensitivity': {'method': bidirectional.STANDARDIZED_METHOD,
                        'role': 'fixed-standard-error studentised maxT over the same draws; '
                                'retained sensitivity only, because its bands and its positive '
                                'counts differ from the declared native family'},
        'contrasts': {'model_given_controls': 'MSE(C) - MSE(CM), or rho(CM) - rho(C) for the ranked '
                                             'reading; both are paired on CM',
                      'controls_given_model': 'MSE(M) - MSE(CM), or rho(CM) - rho(M)',
                      'verdict': 'positive when the simultaneous interval excludes zero above; '
                                 'negative when it excludes zero below; otherwise unresolved'},
        'gate': {'schema': PILOT_SCHEMA, 'tolerance': PILOT_TOLERANCE, 'atol': ATOL, 'rtol': RTOL,
                 'requirement': 'a three-seed pilot reproduces the archived folds, penalties, '
                                'design widths, response diagnostics and grouped metrics of its '
                                'reused arms C and CM before any summary is written; an ordinary '
                                'fit record cannot stand in for it, and the pure likelihood arm has '
                                'no archived counterpart to reproduce'},
        'historical_limitation': 'the archived fits cross-fit G on the outer training partition '
                                 'and reuse it inside that fold\'s inner tuning; reproduced as '
                                 'accepted, not silently changed',
        'source': {'root_relative': SOURCE,
                   'extraction_layouts': list(EXTRACTION_LAYOUTS),
                   'hashes': shared_hashes(),
                   'code_sha256': code_hashes()},
        'not_reproduced': 'the archived R-block (representation) cells: R is not one of the three '
                          'arms, so those archived numbers stay unverified by this stage',
        'resource': {'workers': 2, 'blas_threads_per_worker': 1, 'gpu': False},
    }


def contract_sha256(declaration: dict) -> str:
    return digest_text(json.dumps(declaration, sort_keys=True, separators=(',', ':')))


def ensure_contract() -> dict:
    """Write the frozen contract once, then refuse any drift in it."""

    declaration = contract()
    digest = contract_sha256(declaration)
    path = OUT / 'contract.json'
    if path.is_file():
        stored = replay.read_json(path)
        if contract_sha256(stored) != digest:
            raise ValueError(f'{path} differs from the current declaration; the frozen inputs, '
                             'the recipe or the code moved. Rerun into a new --out')
    else:
        write(path, declaration)
    return {'declaration': declaration, 'sha256': digest}


def require_pilot(pilot, *, contract_sha256: str, code: dict, shared: dict, sample_identity: dict,
                  records, split_seeds=SEEDS, outer_folds=abundance.OUTER_SPLITS,
                  designs=REUSED, tolerance: float = PILOT_TOLERANCE) -> dict:
    """Refuse a summary unless a real pilot reproduced the archived arm behaviour.

    The pilot must be its own recorded invocation, on all three split seeds, bound
    to the frozen contract, to the code that produced it, to the shared inputs and
    to the arm's archived extraction. Every reused penalty must equal the archived
    one exactly, and every archived grouped metric must agree within the declared
    ``atol``/``rtol`` and inside the shared ``1e-7`` ceiling. The archived
    per-group error vectors are not stored, so their agreement is read through the
    archived grouped intervals, which are functions of those vectors.
    """

    if not isinstance(pilot, dict):
        raise ValueError('a pilot is a record object')
    if pilot.get('pilot') is not True:
        raise ValueError('refusing a record that is not a pilot: the reuse gate needs an explicit '
                         'three-seed pilot invocation, not a fit record')
    if pilot.get('status') != 'complete':
        raise ValueError(f"pilot status is {pilot.get('status')!r}, not complete")
    if pilot.get('schema') != PILOT_SCHEMA:
        raise ValueError(f"pilot schema is {pilot.get('schema')!r}, not {PILOT_SCHEMA}")
    if pilot.get('tolerance') != PILOT_TOLERANCE:
        raise ValueError('pilot tolerance differs from the declared reproduction tolerance')
    if tolerance > PILOT_TOLERANCE:
        raise ValueError('the pilot gate tolerance is never loosened beyond the declared value')
    if (pilot.get('atol'), pilot.get('rtol')) != (ATOL, RTOL):
        raise ValueError('pilot comparison tolerances differ from the declared atol/rtol')
    if pilot.get('contract_sha256') != contract_sha256:
        raise ValueError('pilot was not run against the current contract')
    if pilot.get('code_sha256') != code:
        raise ValueError('pilot was not run by the current code')
    if pilot.get('shared_hashes') != shared:
        raise ValueError('pilot is not bound to the frozen original extraction inputs')
    if pilot.get('sample_identity') != sample_identity:
        raise ValueError('pilot sample identity differs from the frozen support')
    if list(pilot.get('reused_designs', [])) != list(designs):
        raise ValueError('pilot did not reproduce the designs being reused')
    if [str(seed) for seed in split_seeds] != [str(seed) for seed in pilot.get('seeds', [])]:
        raise ValueError(f"pilot seeds are {pilot.get('seeds')}, not all of "
                         f"{[str(seed) for seed in split_seeds]}")
    if pilot.get('outer_folds') != outer_folds:
        raise ValueError(f'pilot covers {pilot.get("outer_folds")} outer folds, not {outer_folds}')
    fitted = {record['arm']: record for record in records}
    arms = pilot.get('arms')
    if not isinstance(arms, dict) or not arms:
        raise ValueError('pilot carries no per-arm records')
    worst = 0.
    for arm, sample in arms.items():
        if arm not in fitted:
            raise ValueError(f'pilot arm {arm} has no completed fit record to summarise')
        record = fitted[arm]
        for key in ('extraction_manifest', 'archives_digest', 'reference'):
            if sample.get('hashes', {}).get(key) != record['hashes'][key]:
                raise ValueError(f'pilot arm {arm} is not bound to the extraction being summarised '
                                 f'({key} differs)')
        if sample.get('sample_identity') != record['sample_identity']:
            raise ValueError(f'pilot arm {arm} sample identity differs from the fitted record')
        seeds = sample.get('seeds')
        if not isinstance(seeds, dict) or [str(s) for s in split_seeds] != sorted(seeds):
            raise ValueError(f'pilot arm {arm} does not cover all split seeds')
        for seed in (str(s) for s in split_seeds):
            entry = seeds[seed]
            folds = entry.get('folds')
            if not isinstance(folds, list) or [fold.get('fold') for fold in folds] != list(range(outer_folds)):
                raise ValueError(f'pilot arm {arm} seed {seed} does not cover outer folds '
                                 f'0..{outer_folds - 1}')
            if entry.get('disagreements'):
                raise ValueError(f'pilot arm {arm} seed {seed} records '
                                 f'{entry["disagreements"]} archived disagreements')
            response = entry.get('response')
            if not isinstance(response, dict):
                raise ValueError(f'pilot arm {arm} seed {seed} records no response diagnostics')
            alphas = response.get('selected_alpha')
            expected_alphas = response.get('archived_selected_alpha')
            if (not isinstance(alphas, list) or not isinstance(expected_alphas, list)
                    or len(alphas) != outer_folds or len(expected_alphas) != outer_folds):
                raise ValueError(f'pilot arm {arm} seed {seed} does not record the response penalty '
                                 f'of all {outer_folds} outer folds')
            if alphas != expected_alphas:
                raise ValueError(f'pilot arm {arm} seed {seed} did not reproduce the archived '
                                 'response penalty')
            for fold in folds:
                for role in ROLES:
                    if fold.get('alpha', {}).get(role) is None:
                        raise ValueError(f'pilot arm {arm} seed {seed} fold {fold.get("fold")} '
                                         f'lacks the {role} penalty')
                for name in designs:
                    alpha = fold.get('alpha', {}).get(name)
                    expected = fold.get('alpha_expected', {}).get(name)
                    if alpha is None or expected is None:
                        raise ValueError(f'pilot arm {arm} seed {seed} fold {fold.get("fold")} '
                                         f'lacks {name} penalty')
                    if alpha != expected:
                        raise ValueError(f'pilot arm {arm} seed {seed} did not reproduce the '
                                         f'archived {name} penalty')
            differences = entry.get('archived_metric_max_absolute_difference')
            if not isinstance(differences, dict):
                raise ValueError(f'pilot arm {arm} seed {seed} records no archived comparison')
            value = differences.get('all')
            if value is None or not np.isfinite(value) or value > tolerance:
                raise ValueError(f'pilot arm {arm} seed {seed} archived metrics differ by '
                                 f'{value}, above {tolerance:g}')
            worst = max(worst, float(value))
    return {'arms': sorted(arms), 'seeds': len(split_seeds), 'outer_folds': outer_folds,
            'designs': list(designs), 'worst_archived_metric_difference': worst}


# --------------------------------------------------------------------------- #
# The declared family.
# --------------------------------------------------------------------------- #

def pointwise_interval(values, *, draws=bidirectional.DRAWS, seed=bidirectional.SEED) -> list:
    """Pointwise percentile interval from the same group draws as the family."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('a pointwise interval needs a finite groups-by-columns matrix')
    rng = np.random.default_rng(seed)
    blocks = []
    for start in range(0, draws, bidirectional.DRAW_CHUNK):
        size = min(bidirectional.DRAW_CHUNK, draws - start)
        blocks.append(values[rng.integers(0, len(values), size=(size, len(values)))].mean(axis=1))
    return np.quantile(np.concatenate(blocks), [.025, .975], axis=0).T.tolist()


def metric_values(record: dict, field: str, seeds) -> dict:
    """One arm's per-seed group values of one reading, with ``None`` kept undefined."""

    values = {}
    for role in ROLES:
        rows = []
        for seed in seeds:
            branch = record['seeds'][str(seed)].get(field)
            if not isinstance(branch, dict) or role not in branch:
                raise ValueError(f"{record['arm']} seed {seed} carries no {field} values for {role}")
            rows.append(np.asarray([np.nan if value is None else float(value)
                                    for value in branch[role]], dtype=float))
        values[role] = rows
    return values


def metric_family(records, name: str, groups, *, seeds=SEEDS) -> dict:
    """Both directions of one reading, on the groups every arm and seed defines.

    A group enters only when all three arms and all three split seeds carry a
    defined value for it, so a paired reduction never compares one arm on a support
    another arm does not share. Undefined groups are reported rather than dropped
    from one arm alone.
    """

    field, to_loss = METRICS[name]['field'], METRICS[name]['to_loss']
    per_arm = {record['arm']: metric_values(record, field, seeds) for record in records}
    mask = np.ones(len(groups), dtype=bool)
    for values in per_arm.values():
        for role in ROLES:
            for row in values[role]:
                if len(row) != len(groups):
                    raise ValueError(f'{name}: group values do not match the held-group support')
                mask &= np.isfinite(row)
    support = [label for label, keep in zip(groups, mask) if keep]
    if len(support) < 2:
        raise ValueError(f'{name}: only {len(support)} groups carry a defined value for every arm and '
                         'split seed; a paired family needs at least two')
    columns, cells, absolute, absolute_columns, per_seed_points = [], [], [], [], {}
    for arm, values in per_arm.items():
        means = {role: np.mean([row[mask] for row in values[role]], axis=0) for role in ROLES}
        for role in ROLES:
            absolute.append(means[role])
            absolute_columns.append(arm + '|' + role)
        # A squared error is better when smaller and a rank correlation when larger,
        # so one sign carries both readings into the same declared direction.
        cells.append(bidirectional.directional_reductions(
            {role: to_loss * means[role] for role in ROLES}, DESIGNS))
        columns.extend(arm + '|' + direction for direction in DESIGNS)
        for index, direction in enumerate(DESIGNS):
            per_seed_points[arm + '|' + direction] = {
                str(seed): float(np.mean(bidirectional.directional_reductions(
                    {role: to_loss * values[role][position][mask] for role in ROLES},
                    DESIGNS)[:, index]))
                for position, seed in enumerate(seeds)}
    return {'groups': support,
            'undefined_groups': [label for label, keep in zip(groups, mask) if not keep],
            'columns': columns, 'contrasts': np.column_stack(cells),
            'absolute': np.column_stack(absolute), 'absolute_columns': absolute_columns,
            'per_seed_points': per_seed_points}


def panel_from_records(records, *, seeds=SEEDS, arms=None) -> dict:
    """Both directions of every declared reading, seeds averaged inside a group first.

    Each arm contributes its group-equal per-group statistic, averaged over the three
    split seeds inside the group; the groups then enter the reduction equally. The
    seed is never an independent unit and no variant or domain count reaches it.
    The fits, targets and weights are untouched: the two readings are two statistics
    of the same out-of-fold predictions.
    """

    records = sorted(records, key=lambda record: record['arm'])
    names = [record['arm'] for record in records]
    expected = sorted(arms) if arms is not None else sorted(ROSTER)
    if names != expected:
        raise ValueError(f'the paired family needs exactly one record per arm; found {len(names)} '
                         f'of {len(expected)}')
    failed = [record['arm'] for record in records if record.get('status') != 'complete']
    if failed:
        raise ValueError(f'{failed} did not complete; a partially reproduced family is not reduced')
    groups = records[0]['groups']
    if any(record['groups'] != groups for record in records):
        raise ValueError('arms do not share one held-group support')
    for record in records:
        missing = [seed for seed in seeds if str(seed) not in record['seeds']]
        if missing:
            raise ValueError(f'{record["arm"]} has no fit record for split seeds {missing}')
    return {'groups': list(groups),
            'metrics': {name: metric_family(records, name, groups, seeds=seeds)
                        for name in METRICS}}


def summarise_verdicts(columns, verdicts) -> dict:
    """Verdict per direction, with the arms that fall on each side."""

    out = {}
    for direction in bidirectional.DIRECTIONS:
        cells = [(label.rsplit('|', 1)[0], value) for label, value in zip(columns, verdicts)
                 if label.rsplit('|', 1)[1] == direction]
        out[direction] = {name: [arm for arm, value in cells if value == name]
                          for name in ('positive', 'negative', 'unresolved')}
        out[direction]['counts'] = bidirectional.counts(value for _, value in cells)
    return out


def aggregate(records, *, gate: dict, resource: dict) -> dict:
    """Reduce every declared reading, write it, and declare what it is conditional on."""

    OUT.mkdir(parents=True, exist_ok=True)
    family = panel_from_records(records)
    maximum = max(record['validation']['archived_metric_max_absolute_difference']
                  for record in records)
    arrays = {'groups': np.asarray(family['groups'])}
    panels, methods = {}, {}
    for name, specification in METRICS.items():
        reading = family['metrics'][name]
        matrix = reading['contrasts']
        if matrix.shape[1] != bidirectional.FAMILY_SIZE:
            raise ValueError(f'{name}: the declared family is {bidirectional.FAMILY_SIZE} columns; '
                             f'this reduction has {matrix.shape[1]}')
        primary = bidirectional.paired_bands(matrix)
        verdicts = summarise_verdicts(reading['columns'], bidirectional.verdict(primary['interval']))
        standardized = simultaneous_bands(matrix, draws=bidirectional.DRAWS,
                                          seed=bidirectional.SEED)
        panels[name] = {
            'units': specification['units'], 'direction': specification['direction'],
            'groups': reading['groups'], 'undefined_groups': reading['undefined_groups'],
            'columns': reading['columns'],
            'simultaneous': dict(primary, units=specification['units']),
            'standardized_maxT_sensitivity': dict(
                standardized,
                units='multiples of the fixed column standard error; retained sensitivity only'),
            'pointwise_interval': pointwise_interval(matrix),
            'per_seed_points': reading['per_seed_points'],
            'verdicts': verdicts,
            'absolute': {'columns': reading['absolute_columns'],
                         'points': reading['absolute'].mean(axis=0).tolist(),
                         'pointwise_interval': pointwise_interval(reading['absolute'])}}
        methods[name] = {'units': specification['units'], 'direction': specification['direction'],
                         'family': f'{bidirectional.FAMILY_SIZE} contrasts = 2 directions x 33 arms',
                         'groups': len(reading['groups']),
                         'undefined_groups': len(reading['undefined_groups'])}
        arrays[name + '|contrasts'] = matrix
        arrays[name + '|absolute'] = reading['absolute']
        arrays[name + '|groups'] = np.asarray(reading['groups'])
    np.savez_compressed(OUT / 'group_contrasts.npz', **arrays)
    panels['conditional'] = {
        'on': 'fitted held-out predictions; no bootstrap retraining',
        'control_set': 'C carries the frozen qualified controls and, for the 23 '
                       'arms archived as S_T, that arm\'s own tokenisation block T',
        'support': 'a group enters a reading only when all three arms and all three split '
                   'seeds define it there; undefined groups are listed per reading',
        'tokenisation_in_controls': {
            record['arm']: record['baseline'] == 'S_T' for record in records}}
    declaration = {'schema': PANEL_SCHEMA, 'arms': [record['arm'] for record in records],
                   'groups': family['groups'],
                   'contract_sha256': ensure_contract()['sha256'],
                   'gate': gate, 'resource': resource,
                   'method': {'primary': bidirectional.NATIVE_METHOD,
                              'sensitivity': bidirectional.STANDARDIZED_METHOD,
                              'family': f'{bidirectional.FAMILY_SIZE} contrasts = 2 directions x '
                                        '33 arms, separately per metric',
                              'unit': '96 families; split seeds averaged inside a group first',
                              'metrics': methods,
                              'archived_metric_max_absolute_difference': maximum,
                              'atol': ATOL, 'rtol': RTOL},
                   'panel': panels}
    write(OUT / 'panel.json', declaration)
    write(OUT / 'completion.json', {
        'status': 'complete', 'endpoint': 'abundance', 'models': len(records),
        'seeds': len(SEEDS), 'gpu_used': False, 'model_inference': False,
        'panel_sha256': replay.digest(OUT / 'panel.json'),
        'contrasts_sha256': replay.digest(OUT / 'group_contrasts.npz'),
        'primary_method': bidirectional.NATIVE_METHOD,
        'sensitivity_method': bidirectional.STANDARDIZED_METHOD,
        'pilot_worst_archived_metric_difference': gate['worst_archived_metric_difference'],
        'archived_metric_max_absolute_difference': maximum,
        'metrics': list(METRICS),
        'positive_counts': {name: {direction: panels[name]['verdicts'][direction]['counts']['positive']
                                   for direction in bidirectional.DIRECTIONS}
                            for name in METRICS},
        'unresolved_counts': {name: {direction:
                                     panels[name]['verdicts'][direction]['counts']['unresolved']
                                     for direction in bidirectional.DIRECTIONS}
                              for name in METRICS},
        'groups_evaluated': {name: len(panels[name]['groups']) for name in METRICS},
        'undefined_groups': {name: panels[name]['undefined_groups'] for name in METRICS}})
    return declaration


def pilot_record(fitted: list[dict], contract_sha256: str) -> dict:
    """The gate record of an explicit pilot invocation."""

    identity = initialise()['sample_identity']
    arms, status = {}, 'complete'
    for record in fitted:
        if record.get('status') != 'complete':
            status = 'failed'
            continue
        arms[record['arm']] = {
            'baseline': record['baseline'], 'hashes': record['hashes'],
            'sample_identity': record['sample_identity'],
            'seeds': {seed: {'folds': [{key: fold[key]
                                        for key in ('fold', 'alpha', 'alpha_expected')}
                                       for fold in entry['folds']],
                             'response': {
                                 'selected_alpha': [fold['selected_alpha']
                                                    for fold in entry['nuisance']],
                                 'archived_selected_alpha': [fold['archived_selected_alpha']
                                                             for fold in entry['nuisance']]},
                             'disagreements': entry['disagreements'],
                             'archived_metric_max_absolute_difference':
                                 entry['archived_metric_max_absolute_difference']}
                      for seed, entry in record['seeds'].items()}}
    return {'schema': PILOT_SCHEMA, 'pilot': True, 'status': status,
            'tolerance': PILOT_TOLERANCE, 'atol': ATOL, 'rtol': RTOL,
            'contract_sha256': contract_sha256, 'code_sha256': code_hashes(),
            'shared_hashes': shared_hashes(), 'sample_identity': identity,
            'reused_designs': list(REUSED), 'seeds': [str(seed) for seed in SEEDS],
            'outer_folds': abundance.OUTER_SPLITS,
            'arms': arms,
            'failed': [record['arm'] for record in fitted if record.get('status') != 'complete']}


# --------------------------------------------------------------------------- #
# Entry points.
# --------------------------------------------------------------------------- #

def resolve_out(path) -> Path:
    out = Path(path).resolve()
    allowed = (ROOT / 'results/extensions/bidirectional_ablation_20261106').resolve()
    if not out.is_relative_to(allowed):
        raise SystemExit(f'--out must stay inside {allowed}')
    if out == HISTORICAL_OUT.resolve() or HISTORICAL_OUT.resolve().is_relative_to(out) \
            or out.is_relative_to(HISTORICAL_OUT.resolve()):
        raise SystemExit('refusing to write the historical abundance directory; use a new --out')
    out.mkdir(parents=True, exist_ok=True)
    return out


def check_inputs(arms) -> dict:
    """Cheap availability of the frozen inputs; archive bytes are hashed at fit time."""

    state = initialise()
    report = {'shared': {key: value for key, value in state['hashes'].items()},
              'support': dict(SUPPORT), 'arms': {}}
    for arm in arms:
        try:
            directory = locate_extraction(state['extraction'], arm)
        except replay.ReplayError as error:
            report['arms'][arm] = {'status': 'missing', 'detail': message(error)}
            continue
        manifest_path = directory / f'manifest_{arm}.json'
        try:
            manifest = replay.read_json(manifest_path)
            identity = manifest['identity']
            report['arms'][arm] = {
                'status': 'present' if manifest.get('status') == 'complete' else 'incomplete',
                'directory': str(directory),
                'manifest_sha256': replay.digest(manifest_path),
                'schema': identity.get('schema'), 'arm': identity.get('arm'),
                'cohort_sha256': identity.get('cohort_sha256'),
                'backgrounds': len(manifest.get('backgrounds', [])),
                'archive_file': 'hashed at fit time'}
        except Exception as error:  # noqa: BLE001 - the report is the record
            report['arms'][arm] = {'status': 'unreadable', 'detail': message(error)}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or '').splitlines()[0])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--extraction-root', type=Path, default=DEFAULT_EXTRACTION,
                        help='root holding ec-<arm>/manifest_<arm>.json from the frozen plan')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--arms', nargs='*')
    parser.add_argument('--pilot', action='store_true',
                        help='fit the named arms and record the three-seed reproduction gate')
    parser.add_argument('--summarize', action='store_true',
                        help='reduce the declared family from completed fit records')
    parser.add_argument('--check-inputs', action='store_true',
                        help='report which arms are restored; no fitting, no claims')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(argv)
    modes = sum((args.pilot, args.summarize, args.check_inputs))
    if modes > 1:
        parser.error('choose one of --pilot, --summarize or --check-inputs')
    if args.workers < 1:
        parser.error('--workers must be at least one')
    global OUT
    ROOT_PATH = Path(args.root).resolve()
    extraction = Path(args.extraction_root).resolve()
    arms = sorted(args.arms or ROSTER)
    if len(set(arms)) != len(arms) or any(arm not in ROSTER for arm in arms):
        parser.error('arms must be unique members of the frozen 33-arm roster')
    if args.pilot and not args.arms:
        parser.error('--pilot needs --arms: a pilot is an explicit small set of arms')
    OUT = DEFAULT_OUT if args.out == DEFAULT_OUT else resolve_out(args.out)
    if args.check_inputs:
        initialise(ROOT_PATH, extraction)
        report = check_inputs(arms)
        print(json.dumps({'arms': {arm: entry['status'] for arm, entry in report['arms'].items()},
                          'present': sum(entry['status'] == 'present' for entry in report['arms'].values())},
                          indent=1))
        return 0 if all(entry['status'] == 'present' for entry in report['arms'].values()) else 2
    initialise(ROOT_PATH, extraction)
    OUT.mkdir(parents=True, exist_ok=True)
    declared = ensure_contract()
    resource = resource_record(args.workers)
    write(OUT / 'run.json', {'contract_sha256': declared['sha256'], 'arms': arms,
                             'workers': args.workers, 'pilot': args.pilot,
                             'summarize': args.summarize,
                             'root': str(ROOT_PATH), 'extraction_root': str(extraction),
                             'resource': resource})
    if args.summarize:
        pilot_path = OUT / 'pilot.json'
        if not pilot_path.is_file():
            raise SystemExit(f'summarise needs a recorded pilot at {pilot_path}; run --pilot first')
        records = []
        for arm in arms:
            path = OUT / 'fits' / f'{arm}.json'
            if not path.is_file():
                raise SystemExit(f'{arm}: no fit record under {OUT}; summarise never fits implicitly')
            try:
                record = reusable(arm, declared['sha256'])
            except ValueError as error:
                raise SystemExit(f'refusing to summarise: {error}') from None
            if record is None:
                raise SystemExit(f'{arm}: fit record is not complete; summarise refuses a partial family')
            records.append(record)
        try:
            gate = require_pilot(replay.read_json(pilot_path), contract_sha256=declared['sha256'],
                                 code=code_hashes(), shared=shared_hashes(),
                                 sample_identity=initialise()['sample_identity'], records=records)
        except ValueError as error:
            raise SystemExit(f'refusing to summarise: {error}') from None
        declaration = aggregate(records, gate=gate, resource=resource)
        print(json.dumps({'status': declaration['schema'], 'models': len(records),
                          'positive': declaration['panel']['verdicts']}, indent=1))
        return 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init,
                             initargs=(str(OUT), str(ROOT_PATH), str(extraction))) as pool:
        fitted = list(pool.map(run_arm, arms, [declared['sha256']] * len(arms)))
    failed = [record['arm'] for record in fitted if record.get('status') != 'complete']
    if args.pilot:
        gate_record = pilot_record([reusable(arm, declared['sha256']) or record
                                    for arm, record in zip(arms, fitted)], declared['sha256'])
        write(OUT / 'pilot.json', gate_record)
    print(json.dumps({'complete': [record['arm'] for record in fitted
                                   if record.get('status') == 'complete'],
                      'failed': failed}, indent=1))
    return 2 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
