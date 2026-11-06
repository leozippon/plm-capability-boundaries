#!/usr/bin/env python3
"""Matched R3 bidirectional ablation using archived scores, no model inference.

Three arms are compared on one residualized target: ``U_C`` is the full control
(traditional sequence, position, composition and profile blocks, the training-fold
nonlinear additive nuisance ``G``, and the model-specific tokenizer descriptors
``T``), ``M`` is the pure scalar likelihood readout, and ``U_C_M`` is their union.
``G`` belongs to the controls, and ``T`` is a nuisance of the model's own
tokenization, so ``U_C`` is a full control rather than a pure traditional-statistics
or model-invariant ``C``. The declared primary directions are therefore
``MSE(U_C) - MSE(U_C_M)`` and ``MSE(M) - MSE(U_C_M)``: what the likelihood adds to
the controls, and what the controls add to the pure likelihood readout. The earlier
reverse contrast conditioned on ``U_M`` is retained beside them as an explicitly
separate sensitivity, because conditioning the likelihood readout on ``G`` and
``T`` first answers a different question.

The archived ``C_G_T`` and ``C_G_T+M`` per-group errors are reused only after an
independent CPU pilot reproduced, on all three split seeds, the archived alpha
choice of both arms and their per-group errors within 1e-7. Summarizing refuses to
run without such a pilot: a fit record is not a pilot, and no tolerance is loosened.
"""
from pathlib import Path
import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.capability.interactions import fit_pairwise_epistasis as fit
from scripts.capability.interactions import refit_likelihood_on_residual as oldfit
from scripts.capability.interactions.run_residual_panel import simultaneous_bands
from src.capability.extensions import bidirectional
from src.capability.interactions import pairwise_epistasis as mod
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DESIGNS = {'U': ('G', 'T'), 'U_M': ('G', 'T', 'M'), 'M': ('M',),
           'U_C': (*mod.C_BLOCKS, 'G', 'T'), 'U_C_M': (*mod.C_BLOCKS, 'G', 'T', 'M')}
OLD_NAMES = {'U_C': 'C_G_T', 'U_C_M': 'C_G_T+M'}
DRAWS = bidirectional.DRAWS
SEED = bidirectional.SEED
RELATIVE_OUT = Path('results/extensions/bidirectional_ablation_20261106/r3-v2')
RELATIVE_OLD = Path('results/R3/pairwise_residual_full_panel_indel_excluded_20260928')
RELATIVE_EXCLUSION = Path('results/R3/pairwise_indel_exclusion_20260928/pairwise_indel_exclusion.json')
RELATIVE_INPUTS = {
    'plan': Path('archive/logs/R3/pairwise_epistasis_20260924/extraction_plan.json'),
    'cohort': Path('archive/logs/R3/pairwise_cohort_20260924/cohort.json'),
    'baseline': Path('archive/logs/R3/pairwise_homologs_20260924/baseline/baseline_q.json'),
    'profiles': Path('archive/logs/R3/pairwise_epistasis_20260924/profile_features.npz'),
    'published': Path('archive/logs/R3/pairwise_epistasis_20260924/panel/panel.json'),
    'exclusion': RELATIVE_EXCLUSION}


def configure(root=None, out=None):
    """Resolve the repository root, the read-only archive and the result directory.

    Every frozen input and every product lives under the repository root, so a moved
    checkout relocates them together; ``--out`` selects a fresh result directory
    rather than rewriting a frozen one.
    """

    global ROOT, OUT, OLD, EXCLUSION, DECL, INPUTS
    ROOT = Path(root or ROOT).resolve()
    OUT = Path(out).resolve() if out else ROOT / RELATIVE_OUT
    OLD = ROOT / RELATIVE_OLD
    EXCLUSION = ROOT / RELATIVE_EXCLUSION
    DECL = json.loads((OLD / 'declaration.json').read_text())
    INPUTS = {key: ROOT / path for key, path in RELATIVE_INPUTS.items()}


configure()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False,
                              default=bidirectional.json_default) + '\n')
    tmp.replace(path)


def arrhash(a):
    a = np.asarray(a)
    h = hashlib.sha256(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def common_hashes(p):
    return {**{'states_' + k: arrhash(v) for k, v in p['states'].items()},
            **{k: arrhash(p[k]) for k in ('group', 'site_pair', 'background', 'cycle_index',
                                          'epsilon', 'cycle_states')},
            **{'C_' + k: arrhash(p['blocks'][k]) for k in mod.C_BLOCKS}}


def load(arm, preparation=False):
    plan = json.loads(INPUTS['plan'].read_text())
    cohort = json.loads(INPUTS['cohort'].read_text())
    profiles, _ = fit.load_profiles(INPUTS['profiles'], plan)
    if preparation:
        directory = ROOT / 'archive/logs/R3/pairwise_epistasis_20260924/local_extraction' / arm
    else:
        directory = ROOT / 'results/pairwise_epistasis_20260924/extraction' / arm
    if not preparation and fit.digest(directory / f'manifest_{arm}.json') != DECL['extractions'][arm]['manifest_sha256']:
        raise ValueError('changed manifest ' + arm)
    data, manifest = fit.load_arm(directory, arm, plan)
    p = fit.build_panel(plan, cohort, profiles, data, json.loads(INPUTS['baseline'].read_text()),
                        set(plan['supports']['all_groups']['groups']))
    p, accounting = mod.indel_excluded_support(p, plan, json.loads(EXCLUSION.read_text()),
                                               set(plan['supports']['all_groups']['groups']))
    counts = {'groups': len(set(p['group'])), 'cycles': len(p['group']),
              'site_pairs': len(set(p['site_pair']))}
    if counts != DECL['support_counts']:
        raise ValueError(f'realised support {counts} differs from the declaration {DECL["support_counts"]}')
    states = p['states']['y'][p['cycle_states']]
    additive = states[:, 1] + states[:, 2] - states[:, 0]
    if not np.allclose(states[:, 3] - states[:, 1] - states[:, 2] + states[:, 0], p['epsilon']):
        raise ValueError('the panel target is not the realized four-state cycle')
    return p, additive


def freeze():
    """Declare the frozen inputs, designs, family and reuse rules of this output."""

    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {key: fit.digest(path) for key, path in INPUTS.items()}
    if hashes != DECL['input_sha256']:
        raise ValueError('frozen input digest differs from the archived declaration')
    contract = {
        'schema': 'r3_bidirectional_v2',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'arms': DECL['arms'], 'support': DECL['support_counts'], 'split_seeds': DECL['split_seeds'],
        'designs': DESIGNS,
        'primary_designs': ['U_C', 'M', 'U_C_M'],
        'primary_contrasts': {direction: f'MSE({reduced}) - MSE({joint})'
                              for direction, (reduced, joint)
                              in bidirectional.PRIMARY_DESIGNS.items()},
        'additional_sensitivities': ['U', 'U_M'],
        'sensitivity_contrasts': {direction: f'MSE({reduced}) - MSE({joint})'
                                  for direction, (reduced, joint)
                                  in bidirectional.SENSITIVITY_DESIGNS.items()},
        'U': ('G is a training-fold nonlinear additive nuisance and belongs with the controls; '
              'T is a model-specific tokenizer nuisance, so U_C is a full control, not a pure '
              'traditional-statistics or model-invariant C'),
        'target': DECL['target'] + '; the archived measured-single-assisted residual, not a raw '
                                  'zero-shot target',
        'folds': {'outer': mod.OUTER_SPLITS, 'inner': mod.INNER_SPLITS}, 'alphas': mod.ALPHAS,
        'aggregation': ('average split errors within group, then equally weight 64 groups; '
                        'equal site-pair weights inside group'),
        'uncertainty': {'draws': DRAWS, 'seed': SEED, 'confidence': .95,
                        'simultaneous_family': bidirectional.FAMILY_SIZE,
                        'method': bidirectional.NATIVE_METHOD, 'units': 'kcal^2/mol^2',
                        'sensitivity': bidirectional.STANDARDIZED_METHOD,
                        'sensitivity_role': ('fixed-standard-error maxT over the same draws, retained '
                                             'as a sensitivity; its bands and positive counts differ '
                                             'from the declared native family')},
        'pilot': {'policy': ('one completed pilot reproducing every archived alpha and per-group MSE '
                             'of both reused designs on all three split seeds is a mandatory '
                             'precondition of summarize'),
                  'tolerance': bidirectional.PILOT_TOLERANCE,
                  'manifest_binding': ('every pilot and every fit record binds the original frozen '
                                       'extraction manifest digest of its arm')},
        'reuse': ('archived U_C and U_C_M per-group errors; new U/U_M/M out-of-fold predictions saved'),
        'interpretation': ('conditional predictive value, not a decomposition of biological '
                           'information; M-only supplemental shares the same training-residualized target'),
        'input_sha256': hashes, 'old_declaration_sha256': fit.digest(OLD / 'declaration.json'),
        'script_sha256': fit.digest(Path(__file__)), 'module_sha256': fit.digest(Path(mod.__file__)),
        'historical_fitter_sha256': DECL['fitter_sha256'],
        'historical_module_sha256': DECL['module_sha256'],
        'code_equivalence': ('October release source paths changed; the pilot must reproduce all '
                             'original alpha choices and per-group errors within 1e-7 before the '
                             'archived errors are pooled'),
        'preparation': ('G and target shared arrays built from verified same-plan local GPT2 support; '
                        'model scores unused in preparation, and every fit must bind the original '
                        'frozen extraction manifest'),
        'resource_policy': {'device': 'cpu', 'torch_threads': 1, 'blas_threads': 1, 'maximum_workers': 2}}
    if (OUT / 'contract.json').exists():
        previous = json.loads((OUT / 'contract.json').read_text())
        for key in ('input_sha256', 'old_declaration_sha256', 'script_sha256'):
            if previous[key] != contract[key]:
                raise ValueError(f'{key} changed after the contract was frozen; a new output '
                                 'directory is required rather than a rewritten contract')
    else:
        write(OUT / 'contract.json', contract)


def prepare():
    """Cache the shared G blocks and residual targets once per seed and outer fold."""

    start = time.monotonic()
    p, additive = load('gpt2', preparation=True)
    groups = p['group']
    write(OUT / 'shared-inputs.json', common_hashes(p))
    for seed in mod.SPLIT_SEEDS:
        for outer, held in enumerate(mod.family_folds(groups, mod.OUTER_SPLITS, seed)):
            dest = OUT / 'cache' / f'{seed}-{outer}.npz'
            if dest.exists():
                continue
            train = np.flatnonzero(~np.isin(groups, held))
            test = np.flatnonzero(np.isin(groups, held))
            inner = mod.family_folds(groups[train], mod.INNER_SPLITS, seed + 100 + outer)
            arrays = {}
            meta = {'seed': seed, 'outer': outer, 'held_groups': held, 'inner': []}
            for i, validation in enumerate(inner):
                fitting = train[~np.isin(groups[train], validation)]
                valid = train[np.isin(groups[train], validation)]
                calibration = mod.family_folds(groups[fitting], mod.INNER_SPLITS,
                                               seed + 1000 + outer * 10 + i)
                excluded = np.asarray(list(held) + list(validation))
                g, diagnostic = mod.nuisance_cycle(p['states'], p['cycle_states'], excluded,
                                                   calibration, device='cpu')
                yf, yv = oldfit.residual_targets(p['epsilon'], additive, fitting, valid)
                arrays.update({f'G_{i}': g, f'fit_{i}': fitting, f'valid_{i}': valid,
                               f'yfit_{i}': yf, f'yvalid_{i}': yv})
                meta['inner'].append({'validation_groups': validation,
                                      'calibration_folds': calibration, 'G_diagnostic': diagnostic})
            g, diagnostic = mod.nuisance_cycle(p['states'], p['cycle_states'], np.asarray(held),
                                               inner, device='cpu')
            yt, yq = oldfit.residual_targets(p['epsilon'], additive, train, test)
            arrays.update(G=g, train=train, test=test, ytrain=yt, ytest=yq)
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, **arrays)
            meta.update(G_diagnostic=diagnostic, arrays_sha256=fit.digest(dest),
                        common_inputs_sha256=fit.digest(OUT / 'shared-inputs.json'))
            write(dest.with_suffix('.json'), meta)
            print(json.dumps({'event': 'prepared', 'seed': seed, 'outer': outer,
                              'elapsed_sec': time.monotonic() - start}), flush=True)
    write(OUT / 'prepare-completion.json', {
        'status': 'complete', 'elapsed_sec': time.monotonic() - start,
        'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'cache_sha256': {x.name: fit.digest(x) for x in sorted((OUT / 'cache').glob('*'))}})


def one(arm, pilot=False):
    """Fit the new designs for one arm, or reproduce both reused designs as the pilot."""

    start = time.monotonic()
    dest = (OUT / 'pilot' if pilot else OUT / 'fits') / f'{arm}.json'
    if dest.exists():
        record = json.loads(dest.read_text())
        if pilot and record.get('pilot') is not True:
            raise ValueError(f'{dest} exists and is not a pilot record; an ordinary fit cannot be '
                             'promoted, so a real pilot run is required')
        return record
    p, additive = load(arm)
    if common_hashes(p) != json.loads((OUT / 'shared-inputs.json').read_text()):
        raise ValueError('model-dependent shared nuisance input ' + arm)
    groups = p['group']
    weights = mod.row_weights(p['site_pair'], groups)
    old = json.loads((OLD / f'residual_{arm}.json').read_text())
    if old['groups_order'] != sorted(set(groups)):
        raise ValueError('archived group order differs from the realised support')
    if old['declaration_sha256'] != fit.digest(OLD / 'declaration.json'):
        raise ValueError('archived residual source is not bound to the declaration')
    names = ['U', 'U_M', 'M'] + (list(OLD_NAMES) if pilot else [])
    rows = {}
    archives = {'group': groups, 'site_pair': p['site_pair'], 'background': p['background'],
                'cycle_index': p['cycle_index']}
    maxdiff = 0.
    differences = {}
    for seed in mod.SPLIT_SEEDS:
        preds = {name: np.full(len(groups), np.nan) for name in names}
        target = np.full(len(groups), np.nan)
        records = []
        for outer, held in enumerate(mod.family_folds(groups, mod.OUTER_SPLITS, seed)):
            cache = OUT / 'cache' / f'{seed}-{outer}.npz'
            meta = json.loads(cache.with_suffix('.json').read_text())
            if meta['arrays_sha256'] != fit.digest(cache):
                raise ValueError(f'cached fold arrays changed: {cache}')
            if not held == meta['held_groups'] == old['seeds'][str(seed)]['folds'][outer]['held_groups']:
                raise ValueError(f'outer fold {outer} membership differs from the archived fit')
            with np.load(cache, allow_pickle=False) as a:
                losses = {name: np.zeros(len(mod.ALPHAS)) for name in names}
                for i, inner in enumerate(meta['inner']):
                    fitting, valid = a[f'fit_{i}'], a[f'valid_{i}']
                    blocks = dict(p['blocks'], G=a[f'G_{i}'])
                    if np.isin(groups[fitting], held + inner['validation_groups']).any():
                        raise ValueError('an outer held group enters an inner training partition')
                    for name in names:
                        x = np.column_stack([blocks[b] for b in DESIGNS[name]])
                        prediction = mod.ridge_predict(x[fitting], a[f'yfit_{i}'], weights[fitting],
                                                       x[valid], mod.ALPHAS, 'cpu')
                        losses[name] += len(inner['validation_groups']) * (
                            weights[valid][:, None] * (prediction - a[f'yvalid_{i}'][:, None]) ** 2).sum(0)
                train, test = a['train'], a['test']
                target[test] = a['ytest']
                blocks = dict(p['blocks'], G=a['G'])
                record = {'fold': outer, 'held_groups': held, 'alpha': {},
                          'inner_losses': {n: v.tolist() for n, v in losses.items()}}
                for name in names:
                    alpha = mod.ALPHAS[mod._select_alpha(losses[name])]
                    x = np.column_stack([blocks[b] for b in DESIGNS[name]])
                    preds[name][test] = mod.ridge_predict(x[train], a['ytrain'], weights[train],
                                                          x[test], [alpha], 'cpu')[:, 0]
                    record['alpha'][name] = alpha
                    if name in OLD_NAMES:
                        expected = old['seeds'][str(seed)]['folds'][outer]['alpha'][OLD_NAMES[name]]
                        record.setdefault('alpha_expected', {})[name] = expected
                        if alpha != expected:
                            raise ValueError(f'pilot alpha mismatch {arm} {seed} {outer} {name}: '
                                             f'{alpha} != {expected}')
                records.append(record)
        if not np.isfinite(target).all() or not all(np.isfinite(v).all() for v in preds.values()):
            raise ValueError(f'{arm} {seed}: incomplete out-of-fold predictions')
        errors = {n: mod.group_errors(target, v, groups, p['site_pair'])[1] for n, v in preds.items()}
        reproduced = errors.copy()
        for n, oldn in OLD_NAMES.items():
            olderr = np.asarray(old['seeds'][str(seed)]['per_group_mse'][oldn])
            if n in errors:
                discrepancy = float(np.max(np.abs(errors[n] - olderr)))
                if discrepancy > bidirectional.PILOT_TOLERANCE:
                    raise ValueError(f'pilot old MSE disagreement {arm} {seed} {n}: {discrepancy:g}')
                differences.setdefault(str(seed), {})[n] = discrepancy
                maxdiff = max(maxdiff, discrepancy)
            errors[n] = olderr
        null = mod.group_errors(target, np.zeros(len(groups)), groups, p['site_pair'])[1]
        if np.max(np.abs(null - old['seeds'][str(seed)]['per_group_mse']['ADDITIVE_NULL'])) >= 1e-12:
            raise ValueError(f'{arm} {seed}: the additive null does not reproduce the archived errors')
        errors['ADDITIVE_NULL'] = null
        rows[str(seed)] = {'per_group_mse': {name: np.asarray(value, dtype=float).tolist()
                                            for name, value in errors.items()},
                           'folds': records}
        if pilot:
            rows[str(seed)] = {'per_group_mse': {name: np.asarray(value, dtype=float).tolist()
                                                for name, value in reproduced.items()},
                               'folds': records,
                               'per_group_mse_max_absolute_difference': differences[str(seed)]}
        archives[f'{seed}__target'] = target
        for name, value in preds.items():
            archives[f'{seed}__{name}'] = value
    base = OUT / 'pilot' if pilot else OUT
    outnpz = base / 'oof' / f'{arm}.npz'
    outnpz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outnpz, **archives)
    report = {'arm': arm, 'status': 'complete', 'pilot': pilot,
              'contract_sha256': fit.digest(OUT / 'contract.json'),
              'script_sha256': fit.digest(Path(__file__)),
              'old_source_sha256': fit.digest(OLD / f'residual_{arm}.json'),
              'old_source': str((OLD / f'residual_{arm}.json').relative_to(ROOT)),
              'original_manifest_sha256': DECL['extractions'][arm]['manifest_sha256'],
              'shared_inputs_sha256': fit.digest(OUT / 'shared-inputs.json'),
              'oof_sha256': fit.digest(outnpz), 'groups_order': old['groups_order'],
              'seeds': rows, 'elapsed_sec': time.monotonic() - start,
              'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if pilot:
        report.update({'schema': bidirectional.PILOT_SCHEMA,
                       'tolerance': bidirectional.PILOT_TOLERANCE,
                       'reused_designs': list(OLD_NAMES),
                       'split_seeds': list(mod.SPLIT_SEEDS), 'outer_folds': mod.OUTER_SPLITS,
                       'max_abs_group_mse': maxdiff})
    else:
        report['old_reproduction_max_abs_group_mse'] = None
    write(dest, report)
    print(json.dumps({'event': 'piloted' if pilot else 'fitted', 'arm': arm,
                      'elapsed_sec': report['elapsed_sec'], 'pilot_max_error': maxdiff}), flush=True)


def require_pilots():
    """Every pilot record on disk, each passing the declared reproduction gate."""

    directory = OUT / 'pilot'
    paths = sorted(directory.glob('*.json')) if directory.exists() else []
    if not paths:
        raise ValueError(f'{directory} holds no pilot record; summarize needs a completed pilot that '
                         'reproduced both reused designs on all three split seeds within '
                         f'{bidirectional.PILOT_TOLERANCE:g}')
    contract = fit.digest(OUT / 'contract.json')
    passed = {}
    for path in paths:
        record = json.loads(path.read_text())
        arm = record.get('arm')
        if arm not in DECL['arms']:
            raise ValueError(f'{path} names an arm outside the frozen roster')
        outcome = bidirectional.require_pilot(
            record, contract_sha256=contract,
            manifest_sha256=DECL['extractions'][arm]['manifest_sha256'],
            split_seeds=mod.SPLIT_SEEDS, outer_folds=mod.OUTER_SPLITS,
            designs=tuple(OLD_NAMES))
        oof = OUT / 'pilot' / 'oof' / f'{arm}.npz'
        if not oof.exists() or fit.digest(oof) != record['oof_sha256']:
            raise ValueError(f'{path} does not bind its pilot out-of-fold archive')
        if record['old_source_sha256'] != fit.digest(OLD / f'residual_{arm}.json'):
            raise ValueError(f'{path} is not bound to the archived source it names')
        passed[arm] = {'record_sha256': fit.digest(path), **outcome}
    return passed


def summarize():
    """Write the declared primary family, its sensitivity and the completion record."""

    arms = DECL['arms']
    contract = fit.digest(OUT / 'contract.json')
    pilots = require_pilots()
    all_errors, perseed, receipts = [], [], {}
    for arm in arms:
        path = OUT / 'fits' / f'{arm}.json'
        record = json.loads(path.read_text())
        receipts[arm] = fit.digest(path)
        if record.get('status') != 'complete' or record.get('pilot') is not False:
            raise ValueError(f'{arm}: the production fit record is not a completed non-pilot fit')
        if record['contract_sha256'] != contract:
            raise ValueError(f'{arm}: fit recorded under a different contract')
        if record['oof_sha256'] != fit.digest(OUT / 'oof' / f'{arm}.npz'):
            raise ValueError(f'{arm}: out-of-fold archive differs from its recorded digest')
        if record['shared_inputs_sha256'] != fit.digest(OUT / 'shared-inputs.json'):
            raise ValueError(f'{arm}: fit used different shared nuisance inputs')
        if record['original_manifest_sha256'] != DECL['extractions'][arm]['manifest_sha256']:
            raise ValueError(f'{arm}: fit is not bound to the original extraction manifest')
        if record['old_source_sha256'] != fit.digest(OLD / f'residual_{arm}.json'):
            raise ValueError(f'{arm}: fit is not bound to the archived residual source')
        names = ('U', 'M', 'U_M', 'U_C', 'U_C_M', 'ADDITIVE_NULL')
        seeds = [record['seeds'].get(str(seed)) for seed in mod.SPLIT_SEEDS]
        if any(seed is None for seed in seeds):
            raise ValueError(f'{arm}: the fit does not cover all split seeds')
        e = np.asarray([[seed['per_group_mse'][name] for name in names] for seed in seeds])
        all_errors.append(e.mean(0))
        perseed.append(e)
    errors = np.asarray(all_errors)
    perseed = np.asarray(perseed)
    names = ('U', 'M', 'U_M', 'U_C', 'U_C_M', 'ADDITIVE_NULL')
    groups = errors.shape[-1]
    if groups != DECL['support_counts']['groups']:
        raise ValueError('aggregated group count differs from the declared support')
    results = summarize_errors(errors, perseed, arms, names, DECL['target'])
    results.update({'contract_sha256': contract, 'pilot': pilots, 'source_sha256': receipts})
    write(OUT / 'results.json', results)
    write(OUT / 'completion.json', {
        'status': 'complete', 'models': len(arms), 'seeds': len(mod.SPLIT_SEEDS),
        'primary_designs': ['U_C', 'M', 'U_C_M'], 'new_oof_designs': ['U', 'U_M', 'M'],
        'reused_designs': list(OLD_NAMES), 'groups': groups,
        'primary_method': bidirectional.NATIVE_METHOD,
        'sensitivity_method': bidirectional.STANDARDIZED_METHOD,
        'pilot_arms': sorted(pilots), 'verdict_counts': results['verdict_counts'],
        'results_sha256': fit.digest(OUT / 'results.json')})


def summarize_errors(errors, perseed, arms, names, target):
    """The declared family from seed-mean and per-seed per-group errors.

    Kept separate from the record reading so the arithmetic of both directions, of
    the native family, of its studentised sensitivity and of the absolute arms can
    be exercised without any archived bytes.
    """

    errors = np.asarray(errors, dtype=float)
    perseed = np.asarray(perseed, dtype=float)
    groups = errors.shape[-1]
    if errors.shape[:2] != (len(arms), len(names)) or perseed.shape[:3] != (
            len(arms), len(mod.SPLIT_SEEDS), len(names)) or perseed.shape[-1] != groups:
        raise ValueError('error tables do not match the arms, designs, seeds and groups')
    # Direction definitions come from the shared module: reduced minus joint per group.
    primary = bidirectional.stacked_reductions(errors, bidirectional.PRIMARY_DESIGNS, names)
    sensitivity = bidirectional.stacked_reductions(errors, bidirectional.SENSITIVITY_DESIGNS, names)
    columns = [f'{arm}|{direction}' for arm in arms for direction in bidirectional.DIRECTIONS]
    primary_matrix = primary.reshape(len(arms) * len(bidirectional.DIRECTIONS), groups).T
    native = bidirectional.paired_bands(primary_matrix)
    standardized = simultaneous_bands(primary_matrix, draws=DRAWS, seed=SEED)
    verdicts = bidirectional.verdict(native['interval'])

    sensitivity_matrix = sensitivity[:, 0, :].T
    native_sensitivity = bidirectional.paired_bands(sensitivity_matrix)
    standardized_sensitivity = simultaneous_bands(sensitivity_matrix, draws=DRAWS, seed=SEED)
    sensitivity_verdicts = bidirectional.verdict(native_sensitivity['interval'])

    rng = np.random.default_rng(SEED)
    index = rng.integers(0, groups, (DRAWS, groups))
    bootstrap = np.empty((DRAWS,) + errors.shape[:2])
    bootdelta = np.empty((DRAWS, len(arms), len(bidirectional.DIRECTIONS)))
    bootsens = np.empty((DRAWS, len(arms)))
    for start in range(0, DRAWS, bidirectional.DRAW_CHUNK):
        stop = min(start + bidirectional.DRAW_CHUNK, DRAWS)
        # Group-equal means per draw; the singleton axis keeps the shared reduction
        # definition, so a draw cannot pair different arms than the point estimate.
        boot = errors[:, :, index[start:stop]].mean(-1).transpose(2, 0, 1)[..., None]
        bootstrap[start:stop] = boot[..., 0]
        bootdelta[start:stop] = bidirectional.stacked_reductions(
            boot, bidirectional.PRIMARY_DESIGNS, names)[..., 0]
        bootsens[start:stop] = bidirectional.stacked_reductions(
            boot, bidirectional.SENSITIVITY_DESIGNS, names)[:, :, 0, 0]
    absolute_ci = np.quantile(bootstrap, [.025, .975], axis=0)
    delta_ci = np.quantile(bootdelta, [.025, .975], axis=0)
    sensitivity_ci = np.quantile(bootsens, [.025, .975], axis=0)

    records = {}
    for a, arm in enumerate(arms):
        record = {'absolute_mse_kcal2': {
            name: {'point': float(errors[a, d].mean()), 'pointwise95_ci': absolute_ci[:, a, d].tolist()}
            for d, name in enumerate(names)}, 'contrasts': {}}
        for d, direction in enumerate(bidirectional.DIRECTIONS):
            reduced, joint = bidirectional.PRIMARY_DESIGNS[direction]
            record['contrasts'][direction] = {
                'reduced': reduced, 'joint': joint, 'units': 'kcal^2/mol^2',
                'mse_reduction': float(primary[a, d].mean()),
                'simultaneous95_ci': native['interval'][a * len(bidirectional.DIRECTIONS) + d],
                'standardized_maxT_sensitivity95_ci': standardized['interval'][
                    a * len(bidirectional.DIRECTIONS) + d],
                'pointwise95_ci': delta_ci[:, a, d].tolist(),
                'per_seed_points': (perseed[a, :, names.index(reduced), :]
                                    - perseed[a, :, names.index(joint), :]).mean(-1).tolist(),
                'verdict': verdicts[a * len(bidirectional.DIRECTIONS) + d],
                'positive': bool(native['interval'][a * len(bidirectional.DIRECTIONS) + d][0] > 0),
                'negative': bool(native['interval'][a * len(bidirectional.DIRECTIONS) + d][1] < 0)}
        direction = next(iter(bidirectional.SENSITIVITY_DESIGNS))
        reduced, joint = bidirectional.SENSITIVITY_DESIGNS[direction]
        record['sensitivity'] = {direction: {
            'reduced': reduced, 'joint': joint, 'units': 'kcal^2/mol^2',
            'role': ('separate sensitivity: the reverse contrast conditioned on the nuisance-only '
                     'arm, not the declared primary controls-given-model direction'),
            'mse_reduction': float(sensitivity[a, 0].mean()),
            'simultaneous95_ci': native_sensitivity['interval'][a],
            'standardized_maxT_sensitivity95_ci': standardized_sensitivity['interval'][a],
            'pointwise95_ci': sensitivity_ci[:, a].tolist(),
            'per_seed_points': (perseed[a, :, names.index(reduced), :]
                                - perseed[a, :, names.index(joint), :]).mean(-1).tolist(),
            'verdict': sensitivity_verdicts[a]}}
        records[arm] = record
    return {
        'schema': 'r3_bidirectional_summary_v2', 'columns': columns, 'target': target,
        'arms': records,
        'primary': {'designs': {'U_C': 'full controls: C blocks + G + T',
                               'M': 'pure scalar likelihood readout',
                               'U_C_M': 'U_C plus M'},
                    'contrasts': {direction: f'MSE({reduced}) - MSE({joint})'
                                  for direction, (reduced, joint)
                                  in bidirectional.PRIMARY_DESIGNS.items()}},
        'simultaneous_bootstrap': dict(native, units='kcal^2/mol^2'),
        'standardized_maxT_sensitivity': dict(
            standardized, units='multiples of the fixed column standard error; sensitivity only'),
        'sensitivity_controls_given_model_U_M_conditioned': {
            'columns': list(arms), 'simultaneous': dict(native_sensitivity, units='kcal^2/mol^2'),
            'standardized_maxT_sensitivity': dict(
                standardized_sensitivity,
                units='multiples of the fixed column standard error; sensitivity only')},
        'verdict_counts': {direction: bidirectional.counts(
            [verdicts[a * len(bidirectional.DIRECTIONS) + d] for a in range(len(arms))])
            for d, direction in enumerate(bidirectional.DIRECTIONS)},
        'sensitivity_verdict_counts': bidirectional.counts(sensitivity_verdicts)}


def main():
    parser = argparse.ArgumentParser(description='R3 matched bidirectional ablation, CPU only')
    parser.add_argument('mode', choices=['freeze', 'prepare', 'fit', 'pilot', 'summarize'])
    parser.add_argument('--arm')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--pilot', action='store_true',
                        help='deprecated synonym of the pilot mode; never an ordinary fit')
    args = parser.parse_args()
    configure(args.root, args.out)
    if args.mode == 'freeze':
        freeze()
        return
    contract = json.loads((OUT / 'contract.json').read_text())
    if contract['script_sha256'] != fit.digest(Path(__file__)):
        raise ValueError('the script changed after the contract was frozen; re-freeze into a new '
                         'output directory instead of relaxing the check')
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'pilot' or (args.mode == 'fit' and args.pilot):
        if not args.arm:
            raise ValueError('a pilot names one arm')
        one(args.arm, pilot=True)
    elif args.mode == 'fit':
        if not args.arm:
            raise ValueError('a fit names one arm')
        one(args.arm, pilot=False)
    else:
        summarize()


if __name__ == '__main__':
    main()