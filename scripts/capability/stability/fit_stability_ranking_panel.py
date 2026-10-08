#!/usr/bin/env python3
"""Recompute the 33-arm stability likelihood panel: ranking against error.

The question is whether a frozen checkpoint's likelihood difference improves the
*ordering* of single-substitution stability changes inside a protein family more
widely than it improves the *quantitative* prediction of those changes, and
whether any gain survives being read on each protease channel separately.

Inputs are the frozen gate's own bytes -- the 25,856-variant cohort over 101
family groups, the control qualification frozen before any model quantity was
read, the mutation-local profiles and the published panel -- plus the native
likelihood archives produced by ``extract_stability_singles.py
--likelihood-only``.  Nothing is refitted from a different cohort and nothing is
reconstructed from a stored summary: every number here comes from held-out
predictions this stage computes.

Per arm and per split seed, five designs are fitted on the identical rows and the
identical nested family partitions, differing only in their declared columns:
the squared-error-qualified control set ``S``, that set plus the arm's own
tokenisation descriptors ``S_T``, ``S`` plus the likelihood difference ``M``, the
rank-qualified set ``S2``, and ``S2`` plus ``M``.  ``S_T`` exists to re-derive
the gate's matched-baseline rule rather than assume it: the published panel
selected ``S`` for all 33 arms, and an arm whose tokenisation descriptors now
qualify would no longer be comparable with that record, so the run is refused
instead of quietly changing baseline.

Four per-family contrasts are read from those predictions.  Two are errors, in
squared kcal/mol: the frozen transfer reading, in which one global
nat-to-kcal/mol slope is learned on the training families, and the qualified
within-family reading, in which both predictions are first recalibrated inside
each family by an affine map fitted on that family's other sites.  Two are
rankings, dimensionless: the licensed within-background Spearman increment over
``S2`` and the same increment over ``S``, which differs from the calibrated
error contrast in nothing but the metric.  Each contrast is then published as its
own 33-column maximum-statistic family over the same 101 resampled families; the
marginal per-arm interval is published beside it and never in place of it.

The protease channels are read from the same held-out predictions against the
cohort's own trypsin and chymotrypsin endpoints, as a declared 99-column family
per metric, together with the two channels' own agreement -- because a difference
between channels smaller than the channels disagree with each other is a
statement about the assay, not about the model.

A missing arm is a failure, not a smaller panel.  The published support is 33
arms and the panel is written only when all 33 are present and accepted; the
2026-09-24 panel registered 33 arms while 32 of its per-arm fits had already
ceased to exist, and a stage that silently shrinks its panel is how that becomes
invisible.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json
from src.capability.extensions.stability_followups import (
    COHORT_SHA256, EXPECTED, control_sets, verify_scalar_archive)
from src.capability.interactions.pairwise_epistasis import (
    OUTER_SPLITS, ROSTER, SPLIT_SEEDS, TOKENISATION_STRATUM)
from src.capability.readouts.readout_analysis import family_folds
from src.capability.stability import gate_inputs, ranking_error as contrast
from src.capability.stability.stability_gate import (
    ENDPOINT, QUALIFICATION_SEEDS, TOKENISATION_FEATURE_ORDER, build_panel, fold_predictions,
    group_errors, load_profiles, paired_increment, qualify, tokenisation_block)

SCHEMA = 'stability_ranking_error_panel_v1'
CHANNEL_SCHEMA = 'stability_channel_robustness_panel_v1'
LIKELIHOOD_SCHEMA = 'stability_singles_likelihood_extraction_v1'

#: Written last, and only over the complete 33-arm support.
COMPLETION = 'completion.json'

#: Written instead of :data:`COMPLETION` by an arm-restricted interface check, so
#: a restricted run can never satisfy a production cell's expected artefact.
SMOKE_COMPLETION = 'smoke_fit.json'

#: The frozen gate inputs this stage reads, resolved under ``--gate-dir`` by
#: :mod:`~src.capability.stability.gate_inputs`, which states why the layout is
#: resolved rather than assumed.  The extraction plan is not among them: this
#: stage reads scores, never the plan that produced them.
GATE_INPUTS = ('cohort.json', 'controls_qualification.json', 'profile_features.npz',
               'panel.json')

#: Largest tolerated disagreement between a recomputed and a frozen group-equal
#: baseline mean squared error.  The baseline carries no arm-specific column, so
#: the two are the same arithmetic on the same bytes and differ only by
#: float accumulation order.
BASELINE_TOLERANCE_KCAL2_MOL2 = 1e-9

#: Measurement channels, and the cohort field carrying each one's endpoint.
CHANNELS = {'combined': 'ddg', 'trypsin': 'ddg_trypsin', 'chymotrypsin': 'ddg_chymotrypsin'}

_STATE: dict = {}


# --------------------------------------------------------------------------- #
# Input resolution and support verification.
# --------------------------------------------------------------------------- #

def resolve_gate_inputs(gate_dir: Path) -> dict[str, Path]:
    """Locate each frozen gate input this stage reads, exactly once."""

    return gate_inputs.resolve(gate_dir, GATE_INPUTS)


def locate_extractions(roots: list[Path], arms: list[str]) -> dict[str, Path]:
    """One likelihood-only extraction directory per arm, from one tree walk.

    A root may be a cell's own output directory, a campaign run directory or the
    parent of several, so manifests are looked for at the root and one or two
    levels below it.  The walk happens once and only the manifests of the
    requested arms are opened, because the same tree holds the campaign
    manifests of every other lane and reading them all would be both slow and
    pointless.

    A candidate is admitted only on the likelihood-only archive schema and the
    frozen cohort digest, which is what separates this recomputation from the
    2026-09-24 representation extractions that live in the same tree under the
    same filenames.  Two surviving candidates for one arm are a refusal, not a
    choice: the two could differ, and picking one silently is the failure mode
    this whole stage exists to prevent.
    """

    wanted = {f'manifest_{arm}.json': arm for arm in arms}
    found: dict[str, set[Path]] = {arm: set() for arm in arms}
    for root in roots:
        for pattern in ('manifest_*.json', '*/manifest_*.json', '*/*/manifest_*.json'):
            for path in root.glob(pattern):
                arm = wanted.get(path.name)
                if arm is None:
                    continue
                try:
                    identity = json.loads(path.read_text())['identity']
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                if (identity.get('schema') == LIKELIHOOD_SCHEMA
                        and identity.get('arm') == arm
                        and identity.get('cohort_sha256') == COHORT_SHA256):
                    found[arm].add(path.resolve().parent)
    located, refused = {}, {}
    for arm in arms:
        if len(found[arm]) == 1:
            located[arm] = next(iter(found[arm]))
        else:
            refused[arm] = sorted(str(path) for path in found[arm])
    if refused:
        raise SystemExit(
            f'{len(refused)} of {len(arms)} arms do not resolve to exactly one '
            'likelihood-only extraction directory under '
            f'{[str(root) for root in roots]}: {json.dumps(refused, sort_keys=True)}')
    return located


def discover_extraction(roots: list[Path], arm: str) -> Path:
    """One arm's extraction directory; the single-arm form of the walk above."""

    return locate_extractions(roots, [arm])[arm]


def channel_targets(cohort: dict) -> dict[str, np.ndarray]:
    """One endpoint vector per channel, in the panel's own row order.

    The three endpoints are already frozen in the cohort, formed by the same
    two-estimate difference against the same wild-type row under the same
    substitution, censoring and three-width quality rules.  Reading them here
    rather than re-deriving them from the source tables is what keeps the channel
    comparison on exactly the rows the combined endpoint was published on.
    """

    targets = {name: [] for name in CHANNELS}
    for background in cohort['backgrounds']:
        for variant in background['variants']:
            for name, field in CHANNELS.items():
                targets[name].append(float(variant[field]))
    vectors = {}
    for name, values in targets.items():
        vector = np.asarray(values, dtype=float)
        if not np.isfinite(vector).all():
            raise SystemExit(f'{name}: the frozen cohort carries a nonfinite endpoint')
        vectors[name] = vector
    return vectors


def tokenisation_blocks(directory: Path, arm: str, cohort: dict) -> np.ndarray:
    """The arm's own segmentation descriptors, read from the scalar archives.

    The likelihood-only archive retains the packed token identities and pooled
    token counts of every state, which is exactly what the gate's tokenisation
    block consumes, so the matched-baseline rule can be re-derived without a
    second forward pass.
    """

    manifest = json.loads((directory / f'manifest_{arm}.json').read_text())
    by_name = {row['background']: row for row in manifest['backgrounds']}
    rows = []
    for background in cohort['backgrounds']:
        record = by_name[background['name']]
        with np.load(directory / record['file'], allow_pickle=False) as data:
            offsets = data['token_offsets']
            identifiers = data['token_ids']
            pooled = data['pooled_token_counts']
            token_ids = [identifiers[offsets[index]:offsets[index + 1]].tolist()
                         for index in range(len(offsets) - 1)]
            for state in data['variant_states']:
                rows.append(tokenisation_block(pooled, token_ids, 0, int(state),
                                               background['length']))
    block = np.asarray(rows, dtype=float)
    if block.shape[1] != len(TOKENISATION_FEATURE_ORDER):
        raise SystemExit(f'{arm}: tokenisation descriptor width changed')
    return block


# --------------------------------------------------------------------------- #
# Per-arm fit. One worker process holds the panel; only contrast vectors return.
# --------------------------------------------------------------------------- #

def _initialise(paths: dict, device: str, threads: int, draws: int,
                calibration_folds: int, calibration_seed: int) -> None:
    torch.set_num_threads(max(threads, 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cohort = json.loads(Path(paths['cohort.json']).read_bytes())
    controls = json.loads(Path(paths['controls_qualification.json']).read_bytes())
    sets, derivation = control_sets(controls)
    profiles, _ = load_profiles(Path(paths['profile_features.npz']),
                               {row['name']: row['wildtype'] for row in cohort['backgrounds']})
    panel = build_panel(cohort, profiles)
    _STATE.update(cohort=cohort, controls=controls, sets=sets, derivation=derivation,
                  panel=panel, device=device, draws=draws,
                  channels=channel_targets(cohort),
                  calibration_folds=calibration_folds, calibration_seed=calibration_seed)


def fit_arm(task: tuple[str, str]) -> dict:
    """Held-out predictions and every declared per-family contrast for one arm."""

    arm, directory = task
    began = time.monotonic()
    panel, sets = _STATE['panel'], _STATE['sets']
    cohort = _STATE['cohort']
    scalar, manifest = verify_scalar_archive(Path(directory), arm, cohort, COHORT_SHA256)
    if len(scalar) != len(panel['target']):
        raise ValueError(f'{arm}: {len(scalar)} scored variants against '
                         f"{len(panel['target'])} cohort variants")
    blocks = dict(panel['blocks'], M=scalar,
                  T=tokenisation_blocks(Path(directory), arm, cohort))
    designs = {'S': tuple(sets['S']), 'S_T': (*sets['S'], 'T'), 'S_M': (*sets['S'], 'M'),
               'S2': tuple(sets['S2']), 'S2_M': (*sets['S2'], 'M')}
    seeds, baselines = {}, {}
    for seed in SPLIT_SEEDS:
        outcome = fold_predictions(panel, blocks, designs, seed=seed,
                                   device=_STATE['device'])
        predictions = outcome['predictions']
        seeds[str(seed)] = {
            'contrasts': contrast.arm_contrasts(
                panel, predictions, channels=_STATE['channels'],
                folds=_STATE['calibration_folds'], seed=_STATE['calibration_seed']),
            'tokenisation_increment_kcal2_mol2': paired_increment(
                panel, predictions, 'S_T', 'S')['point'],
            'alpha': [record['alpha'] for record in outcome['folds']],
            'dimensions': outcome['folds'][0]['dimensions'],
            'held_groups': [record['held_groups'] for record in outcome['folds']],
        }
        for name in ('S', 'S2'):
            _, errors = group_errors(panel['target'], predictions[name],
                                     panel['group'], panel['site'])
            baselines.setdefault(str(seed), {})[name] = float(errors.mean())
    verdict = qualify({seed: seeds[str(seed)]['tokenisation_increment_kcal2_mol2']
                       for seed in QUALIFICATION_SEEDS})
    if verdict['qualified']:
        raise ValueError(
            f'{arm}: the tokenisation descriptors now qualify into this arm\'s matched '
            'baseline, while the published panel selected the control set alone for all '
            '33 arms; the recomputed increments would not be comparable with that record '
            f'and no baseline is substituted silently. Verdict: {verdict}')
    return {'arm': arm, 'extraction_directory': str(directory),
            'extraction_manifest_sha256': sha256_file(Path(directory) / f'manifest_{arm}.json'),
            'dtype': manifest['identity']['dtype'],
            'checkpoint_path': manifest['identity']['checkpoint_path'],
            'repeat_likelihood_nats_max': max(row['repeat_likelihood_nats']
                                              for row in manifest['backgrounds']),
            'tokenisation_stratum': TOKENISATION_STRATUM[arm],
            'tokenisation_verdict': verdict, 'matched_baseline': 'S',
            'baseline_mse_kcal2_mol2': baselines, 'seeds': seeds,
            'seconds': time.monotonic() - began}


# --------------------------------------------------------------------------- #
# Panel assembly.
# --------------------------------------------------------------------------- #

def seed_mean(record: dict, channel: str, name: str) -> np.ndarray:
    """One per-family vector per arm, split seeds averaged before any resampling.

    Averaging inside the family is the project's convention and it is not
    cosmetic: three split seeds over the same 101 families are three readings of
    one quantity, not three independent units, and treating them as units would
    narrow every interval by about a factor of the square root of three.
    """

    return np.mean([contrast.vector(record['seeds'][str(seed)]['contrasts'][channel][name])
                    for seed in SPLIT_SEEDS], axis=0)


def contrast_family(records: list[dict], channel: str, name: str, *, draws: int) -> dict:
    """One declared contrast as a 33-column maximum-statistic family."""

    arms = [record['arm'] for record in records]
    matrix = np.column_stack([seed_mean(record, channel, name) for record in records])
    undefined = {record['arm']: {
        seed: record['seeds'][seed]['contrasts'][channel]['undefined_ranking_groups']
        for seed in record['seeds']
        if any(record['seeds'][seed]['contrasts'][channel]['undefined_ranking_groups'].values())}
        for record in records}
    undefined = {arm: value for arm, value in undefined.items() if value}
    if not np.isfinite(matrix).all():
        return {'status': 'blocked_undefined_families', 'columns': arms,
                'undefined_families': undefined,
                'rule': 'a family without a defined statistic is reported, never deleted '
                        'and never imputed as zero, because deleting it would make the '
                        'family size depend on the arm'}
    family = contrast.simultaneous_family(matrix, arms, draws=draws)
    family.update(status='complete', **contrast.CONTRASTS[name],
                  undefined_families=undefined,
                  marginal={arm: contrast.marginal_interval(matrix[:, index])
                            for index, arm in enumerate(arms)},
                  per_seed_point={
                      arm: {str(seed): float(np.mean(
                          records[index]['seeds'][str(seed)]['contrasts'][channel][name]))
                          for seed in SPLIT_SEEDS}
                      for index, arm in enumerate(arms)})
    return family


def channel_family(records: list[dict], name: str, *, draws: int) -> dict:
    """One contrast read on both proteases and on their difference, per arm.

    Three columns per arm -- trypsin, chymotrypsin and the paired difference --
    make one declared family of 99 contrasts.  The difference column is the
    robustness reading: a gain that appears on one protease only is a property of
    that measurement channel, and the paired column is what can resolve it.
    """

    arms = [record['arm'] for record in records]
    columns, vectors = [], []
    for record in records:
        trypsin = seed_mean(record, 'trypsin', name)
        chymotrypsin = seed_mean(record, 'chymotrypsin', name)
        columns.extend([f"{record['arm']}:trypsin", f"{record['arm']}:chymotrypsin",
                        f"{record['arm']}:trypsin-minus-chymotrypsin"])
        vectors.extend([trypsin, chymotrypsin, trypsin - chymotrypsin])
    matrix = np.column_stack(vectors)
    if not np.isfinite(matrix).all():
        return {'status': 'blocked_undefined_families', 'columns': columns}
    family = contrast.simultaneous_family(matrix, columns, draws=draws)
    family.update(status='complete', arms=arms, **contrast.CONTRASTS[name])
    return family


def channel_agreement(cohort: dict, panel: dict, targets: dict) -> dict:
    """What the two proteases agree on, per family, before any model is read.

    Every accepted row of this cohort had to pass a 0.5 kcal/mol confidence-width
    rule on the combined, trypsin and chymotrypsin fits alike, so each of the
    25,856 variants carries both channels by construction; the qualification
    below states that rather than assuming it.  The agreement is the floor the
    between-channel contrasts have to be read against: a difference between
    channels smaller than the channels' own disagreement is a statement about the
    assay.
    """

    from scipy.stats import spearmanr

    groups = panel['group']
    trypsin, chymotrypsin = targets['trypsin'], targets['chymotrypsin']
    per_family = []
    for group in sorted(set(groups.tolist())):
        rows = np.flatnonzero(groups == group)
        per_family.append({
            'group': group, 'variants': int(len(rows)),
            'channel_spearman': float(spearmanr(trypsin[rows], chymotrypsin[rows])[0]),
            'mean_squared_channel_difference_kcal2_mol2':
                float(np.mean((trypsin[rows] - chymotrypsin[rows]) ** 2))})
    return {
        'qualification': {
            'wild_types_with_both_channels': len(cohort['backgrounds']),
            'variants_with_both_channels': int(len(trypsin)),
            'variants_total': int(len(panel['target'])),
            'rule': cohort['row_accounting']['indel_rule'],
            'width_rule': 'each of the combined, trypsin and chymotrypsin 95% confidence '
                          'widths inside [0, 0.5] kcal/mol, applied before the support was '
                          'drawn, so no variant of this cohort lacks either channel',
            'rows_after_trypsin_width_rule':
                cohort['row_accounting']['rows_after_deltaG_t_95CI_rule'],
            'rows_after_chymotrypsin_width_rule':
                cohort['row_accounting']['rows_after_deltaG_c_95CI_rule'],
            'variants_at_channel_fit_bound':
                cohort['row_accounting']['accepted_rows_at_channel_fit_bound'],
        },
        'per_family': per_family,
        'mean_family_channel_spearman': float(np.mean(
            [row['channel_spearman'] for row in per_family])),
        'mean_family_squared_channel_difference_kcal2_mol2': float(np.mean(
            [row['mean_squared_channel_difference_kcal2_mol2'] for row in per_family])),
        'interpretation': 'correlated channels of one proteolysis assay, not independent '
                          'replication; a nonsignificant between-channel difference is not '
                          'evidence of equivalence',
    }


def frozen_comparison(published: dict, records: list[dict]) -> dict:
    """Recomputed against published, arm by arm, whichever way it comes out.

    Two frozen quantities are directly comparable: the published squared-error
    increment over the squared-error-qualified control set, which this stage
    recomputes as ``transfer_error``, and the published within-background
    Spearman increment over the rank-qualified set, recomputed as
    ``within_family_ranking``.  The published baselines carry no arm-specific
    column and are compared as a hard gate, because a baseline that differs means
    the recomputation is not on the frozen support at all.
    """

    arms_reported = published.get('arms_reported', [])
    frozen_panel = published.get('panel', {})
    pairs = {'transfer_error': 'primary_likelihood',
             'within_family_ranking': 'secondary_likelihood_spearman'}
    increments: dict[str, dict] = {}
    for name, key in pairs.items():
        frozen = frozen_panel.get(key, {}).get('arms', {})
        rows = {}
        for record in records:
            arm = record['arm']
            if arm not in frozen:
                continue
            recomputed = float(np.mean(seed_mean(record, 'combined', name)))
            rows[arm] = {'recomputed_seed_mean': recomputed,
                         'published_seed_mean': frozen[arm]['seed_mean'],
                         'difference': recomputed - frozen[arm]['seed_mean'],
                         'published_marginal_verdict': frozen[arm].get('verdict')}
        increments[name] = {
            'published_contrast': key,
            'published_metric': frozen_panel.get(key, {}).get('metric'),
            'arms': rows,
            'max_absolute_difference': (max(abs(row['difference']) for row in rows.values())
                                        if rows else None),
            'published_marginal_resolved_above_zero':
                frozen_panel.get(key, {}).get('resolved_above_zero'),
        }
    baselines = {}
    for seed in SPLIT_SEEDS:
        frozen = published.get('baseline_identity', {}).get(str(seed), {})
        for name, role in (('S', 'primary'), ('S2', 'secondary')):
            if role not in frozen:
                continue
            recomputed = {record['baseline_mse_kcal2_mol2'][str(seed)][name]
                          for record in records}
            spread = max(recomputed) - min(recomputed)
            baselines[f'{seed}|{name}'] = {
                'recomputed_mse_kcal2_mol2': float(max(recomputed)),
                'recomputed_spread_across_arms': float(spread),
                'published_mse_kcal2_mol2': frozen[role]['mse_kcal2_mol2'],
                'difference': float(max(recomputed) - frozen[role]['mse_kcal2_mol2'])}
    offending = {key: row for key, row in baselines.items()
                 if abs(row['difference']) > BASELINE_TOLERANCE_KCAL2_MOL2
                 or row['recomputed_spread_across_arms'] > BASELINE_TOLERANCE_KCAL2_MOL2}
    if offending:
        raise SystemExit('the recomputed control baselines do not reproduce the published '
                         f'ones on the frozen support: {json.dumps(offending)}')
    return {
        'published_panel_schema': published.get('schema'),
        'published_arms_reported': len(arms_reported),
        'published_arms_missing': published.get('arms_missing'),
        'published_tokenisation_arms_in_baseline': {
            str(seed): published.get('baseline_identity', {}).get(
                str(seed), {}).get('arms_with_tokenisation_in_baseline')
            for seed in SPLIT_SEEDS},
        'baseline_gate': {'tolerance_kcal2_mol2': BASELINE_TOLERANCE_KCAL2_MOL2,
                          'cells': baselines},
        'increments': increments,
        'reading': ('the baselines are a gate and the increments are a finding; a published '
                    'increment that does not reproduce is reported as it comes out, in '
                    'either direction'),
    }


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate-dir', type=Path, action='append', required=True,
                        help='a candidate frozen stability gate directory; repeatable, '
                             'because two directory layouts of that measurement are in '
                             'use and candidates that do not exist are passed over')
    parser.add_argument('--extraction-root', type=Path, action='append', required=True,
                        help='a directory holding, or containing, the likelihood-only '
                             'extraction directories; repeatable')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--draws', type=int, default=contrast.PANEL_DRAWS)
    parser.add_argument('--calibration-folds', type=int, default=contrast.CALIBRATION_FOLDS)
    parser.add_argument('--calibration-seed', type=int, default=contrast.CALIBRATION_SEED)
    parser.add_argument('--arms', default='',
                        help='interface check only: a comma-separated subset. A restricted '
                             'run publishes per-arm fits and no panel, and writes '
                             f'{SMOKE_COMPLETION} rather than {COMPLETION}')
    args = parser.parse_args()

    arms = [item.strip() for item in args.arms.split(',') if item.strip()] or list(ROSTER)
    unknown = [arm for arm in arms if arm not in ROSTER]
    if unknown:
        raise SystemExit(f'not on the frozen roster: {unknown}')
    restricted = len(arms) != len(ROSTER)
    torch.set_num_threads(max(args.threads, 1))
    began = time.monotonic()

    paths = resolve_gate_inputs(args.gate_dir)
    digests = {name: sha256_file(path) for name, path in paths.items()}
    if digests['cohort.json'] != COHORT_SHA256:
        raise SystemExit(f"the resolved cohort digest {digests['cohort.json']} is not the "
                         f'frozen {COHORT_SHA256}')
    cohort = json.loads(paths['cohort.json'].read_bytes())
    if cohort.get('schema') != 'stability_singles_cohort_v1':
        raise SystemExit('unexpected cohort schema')
    published = json.loads(paths['panel.json'].read_bytes())
    controls = json.loads(paths['controls_qualification.json'].read_bytes())
    if controls.get('cohort_sha256') != COHORT_SHA256:
        raise SystemExit('the control qualification was frozen against a different cohort')
    sets, derivation = control_sets(controls)

    # Resolved before anything expensive is built. A missing arm is the failure
    # this stage exists to make loud, and it must not wait behind a fifteen-second
    # panel assembly to be reported.
    located = locate_extractions(args.extraction_root, arms)

    profiles, _ = load_profiles(paths['profile_features.npz'],
                                {row['name']: row['wildtype'] for row in cohort['backgrounds']})
    panel = build_panel(cohort, profiles)
    support = {'rows': int(len(panel['target'])), 'groups': int(len(set(panel['group']))),
               'sites': int(len(set(panel['site'])))}
    if support != EXPECTED:
        raise SystemExit(f'the exact single-mutant support changed: {support} against {EXPECTED}')
    folds = {str(seed): family_folds(panel['group'], OUTER_SPLITS, seed)
             for seed in SPLIT_SEEDS}
    targets = channel_targets(cohort)

    args.out.mkdir(parents=True, exist_ok=True)
    fits = args.out / 'fits'
    fits.mkdir(exist_ok=True)

    pending = [(arm, str(located[arm])) for arm in arms
               if not (fits / f'fit_{arm}.json').exists()]
    initargs = ({name: str(path) for name, path in paths.items()}, args.device,
                args.threads, args.draws, args.calibration_folds, args.calibration_seed)
    if pending:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(pending)),
                                     initializer=_initialise, initargs=initargs) as pool:
                for record in pool.map(fit_arm, pending):
                    write_json(fits / f"fit_{record['arm']}.json", record)
                    print(f"[fit] {record['arm']} {record['seconds']:.1f}s", flush=True)
        else:
            _initialise(*initargs)
            for task in pending:
                record = fit_arm(task)
                write_json(fits / f"fit_{record['arm']}.json", record)
                print(f"[fit] {record['arm']} {record['seconds']:.1f}s", flush=True)

    records = []
    for arm in arms:
        path = fits / f'fit_{arm}.json'
        if not path.exists():
            raise SystemExit(f'{arm}: no per-arm fit was produced')
        records.append(json.loads(path.read_text()))

    identity = {
        'schema': SCHEMA,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'endpoint': ENDPOINT,
        'support': {**support, 'split_seeds': list(SPLIT_SEEDS),
                    'qualified_control_set': list(sets['S']),
                    'rank_qualified_control_set': list(sets['S2']),
                    'secondary_control_derivation': derivation,
                    'cohort_sha256': digests['cohort.json'],
                    'controls_sha256': digests['controls_qualification.json'],
                    'profile_sha256': digests['profile_features.npz'],
                    'published_panel_sha256': digests['panel.json'],
                    'endpoint_sha256': cohort['endpoint_sha256'],
                    'outer_fold_held_groups': folds},
        'calibration': {'folds': args.calibration_folds, 'seed': args.calibration_seed,
                        'semantics': contrast.family_calibrated.__doc__.strip()},
        'contrast_definitions': contrast.CONTRASTS,
        'arms': arms,
        'extraction': {record['arm']: {
            'directory': record['extraction_directory'],
            'manifest_sha256': record['extraction_manifest_sha256'],
            'dtype': record['dtype'], 'checkpoint_path': record['checkpoint_path'],
            'repeat_likelihood_nats_max': record['repeat_likelihood_nats_max']}
            for record in records},
        'code_sha256': {str(path.relative_to(ROOT)): sha256_file(path) for path in (
            Path(__file__), ROOT / 'src/capability/stability/ranking_error.py',
            ROOT / 'src/capability/stability/stability_gate.py',
            ROOT / 'src/capability/extensions/stability_followups.py',
            ROOT / 'scripts/capability/interactions/run_residual_panel.py')},
    }

    if restricted:
        write_json(args.out / SMOKE_COMPLETION, {
            **identity, 'status': 'interface_check_restricted_arms',
            'arms_requested': arms, 'arms_in_published_support': len(ROSTER),
            'panel': 'not written; the published support is 33 arms',
            'per_arm_point_estimates': {record['arm']: {
                channel: {name: float(np.mean(seed_mean(record, channel, name)))
                          for name in contrast.CONTRASTS}
                for channel in CHANNELS} for record in records},
            'frozen_comparison': frozen_comparison(published, records),
            'elapsed_seconds': time.monotonic() - began})
        print(json.dumps({'status': 'interface_check_restricted_arms', 'arms': arms},
                         indent=1), flush=True)
        return

    families = {name: contrast_family(records, 'combined', name, draws=args.draws)
                for name in contrast.CONTRASTS}
    blocked = [name for name, family in families.items() if family['status'] != 'complete']
    comparison = frozen_comparison(published, records)
    panel_record = {
        **identity,
        'families': families,
        'ranking_versus_error': contrast.ranking_versus_error(
            {name: family for name, family in families.items() if family['status'] == 'complete'},
            arms),
        'frozen_comparison': comparison,
        'blocked_contrasts': blocked,
        'multiplicity': (
            'each contrast is one 33-column maximum-statistic family over the same 101 '
            'resampled family groups, with split seeds averaged inside the family before '
            'resampling; the bands condition on the fitted cross-validation predictions '
            'and omit training and split variation; families are not pooled across '
            'metrics, because the metrics carry different units; the marginal per-arm '
            'interval beside each column is not a simultaneous statement'),
    }
    write_json(args.out / 'panel.json', panel_record)

    channel_record = {
        'schema': CHANNEL_SCHEMA,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'support': identity['support'],
        'arms': arms,
        'channels': ['trypsin', 'chymotrypsin'],
        'mode': ('the combined-endpoint held-out predictions re-read against each channel '
                 'endpoint on identical rows and identical family partitions; not '
                 'channel-specific refits, so the two channels are two readings of one fit'),
        'agreement': channel_agreement(cohort, panel, targets),
        'families': {name: channel_family(records, name, draws=args.draws)
                     for name in contrast.CONTRASTS},
        'interpretation': (
            'a gain that resolves on one protease only is a property of that measurement '
            'channel rather than of folding stability; the paired difference column is the '
            'reading that can resolve it, and it must be read against the channels\' own '
            'disagreement reported in "agreement"'),
        'multiplicity': ('one declared 99-column family per contrast: 33 arms times '
                         'trypsin, chymotrypsin and their paired difference'),
    }
    write_json(args.out / 'channels.json', channel_record)

    write_json(args.out / COMPLETION, {
        'schema': 'stability_ranking_error_completion_v1',
        'status': 'complete' if not blocked else 'complete_with_blocked_contrasts',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'arms': arms,
        'blocked_contrasts': blocked,
        'artifacts': {name: sha256_file(args.out / name)
                      for name in ('panel.json', 'channels.json')},
        'per_arm_fits': {record['arm']: sha256_file(fits / f"fit_{record['arm']}.json")
                         for record in records},
        'resolved_above_zero': {name: family.get('resolved_above_zero')
                                for name, family in families.items()},
        'frozen_increment_max_absolute_difference': {
            name: row['max_absolute_difference']
            for name, row in comparison['increments'].items()},
        'elapsed_seconds': time.monotonic() - began,
    })
    print(json.dumps({
        'arms': len(arms), 'blocked_contrasts': blocked,
        'resolved_above_zero': {name: family.get('resolved_above_zero')
                                for name, family in families.items()},
        'frozen_increment_max_absolute_difference': {
            name: row['max_absolute_difference']
            for name, row in comparison['increments'].items()},
        'seconds': round(time.monotonic() - began, 1)}, indent=1), flush=True)


if __name__ == '__main__':
    main()
