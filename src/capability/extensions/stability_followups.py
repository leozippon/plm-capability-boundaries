"""CPU-only, exact-support stability follow-ups; never recover ranks from losses.

S2 licenses ranking and S licenses MSE. Fits are new extension fits unless original
OOF provenance has actually been recovered. G is fold-dependent, not a static matrix.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from ..core.io import sha256_file
from ..stability import stability_gate as gate
from ..readouts.readout_analysis import family_folds

EXPECTED = {'rows': 25856, 'sites': 5664, 'groups': 101}
COHORT_SHA256 = '1e42c3cc1d11d479fcbcce23ffcc37b5799fcf7fc79c65780b1a521fd953e9de'


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def binding(path: Path, root: Path) -> dict:
    return {'path': str(path.relative_to(root)), 'sha256': sha256_file(path)}


def registry(cohort: dict, samples: pd.DataFrame) -> pd.DataFrame:
    """Require exported order and exact identities, not merely equal row counts."""
    rows = []
    for background_index, background in enumerate(cohort['backgrounds']):
        gate.validate_plan_background(background)
        for variant_index, variant in enumerate(background['variants']):
            if variant['sequence'] != background['sequences'][variant['state']]:
                raise ValueError('variant sequence differs from declared exact state')
            rows.append({'unit_id': background['name'], 'group_id': background['group'],
                         'variant_index': variant_index, 'state': variant['state'],
                         'positions_1based': variant['position'], 'sequence': variant['sequence'],
                         'wildtype': background['wildtype'],
                         'source_pointer': f'/backgrounds/{background_index}/variants/{variant_index}',
                         'ddg': variant['ddg'], 'ddg_trypsin': variant['ddg_trypsin'],
                         'ddg_chymotrypsin': variant['ddg_chymotrypsin'],
                         'wildtype_combined_kcal_mol': background['wildtype_combined_kcal_mol']})
    expected = pd.DataFrame(rows)
    if len(expected) != len(samples) or samples.sample_id.duplicated().any():
        raise ValueError('missing or duplicate sample identities')
    for column in ('unit_id', 'group_id', 'variant_index', 'state', 'positions_1based', 'source_pointer'):
        if not np.array_equal(expected[column].to_numpy(), samples[column].to_numpy()):
            raise ValueError(f'ordered sample alignment differs: {column}')
    for column in ('ddg', 'ddg_trypsin', 'ddg_chymotrypsin', 'wildtype_combined_kcal_mol'):
        if not np.allclose(expected[column], samples[column], rtol=0, atol=1e-12):
            raise ValueError(f'exported label differs: {column}')
    expected.insert(0, 'sample_id', samples.sample_id.to_numpy())
    expected['site'] = expected.unit_id + ':' + expected.positions_1based.astype(str)
    return expected


def control_sets(controls: dict) -> tuple[dict, dict]:
    secondary, evidence = gate.secondary_control_set(controls)
    primary = tuple(controls['qualified_control_set'])
    if primary != ('ident', 'geom', 'chem', 'G') or set(secondary) != set(primary) | {'comp', 'prof2'}:
        raise ValueError('frozen dual control qualification changed')
    return {'S': primary, 'S2': secondary}, evidence


def recorded_partitions(groups, historical: dict) -> dict:
    """Verify recorded outer memberships; disclose regenerated inner memberships."""
    result = {}
    for seed in gate.SPLIT_SEEDS:
        frozen = (historical['primary'][str(seed)]['nuisance'] if 'primary' in historical
                  else historical['seeds'][str(seed)]['folds'])
        generated = family_folds(groups, gate.OUTER_SPLITS, seed)
        if len(frozen) != len(generated):
            raise ValueError('outer partition length changed')
        folds = []
        for outer, (original, held) in enumerate(zip(frozen, generated)):
            if set(original['held_groups']) != set(held):
                raise ValueError('recorded outer partition differs from current recipe')
            train = np.asarray(groups)[~np.isin(groups, held)]
            inner = family_folds(train, gate.INNER_SPLITS, seed + 100 + outer)
            folds.append({'fold': outer, 'held_groups': original['held_groups'],
                          'training_groups': sorted(set(train)),
                          'inner_held_groups': [list(part) for part in inner],
                          'inner_provenance': 'regenerated historical recipe; original serialized inner memberships unavailable'})
        result[str(seed)] = folds
    return result


def channel_panel(panel: dict, values: pd.DataFrame, channel: str) -> dict:
    """Replace both delta targets and absolute WT/mutant labels for channel G."""
    if channel not in ('combined', 'trypsin', 'chymotrypsin'):
        raise ValueError('unknown measurement channel')
    output = dict(panel)
    target = values[f'{channel}_mutant'].to_numpy() - values[f'{channel}_wt'].to_numpy()
    states = dict(panel['states'])
    y = np.full(len(states['y']), np.nan)
    for index, (wild, mutant) in enumerate(panel['pair_states']):
        wt = values.iloc[index][f'{channel}_wt']
        if np.isfinite(y[wild]) and y[wild] != wt:
            raise ValueError('inconsistent repeated WT label')
        y[wild], y[mutant] = wt, values.iloc[index][f'{channel}_mutant']
    if not np.isfinite(y).all() or not np.isfinite(target).all():
        raise ValueError('missing absolute channel/WT state labels')
    states['y'], output['states'], output['target'] = y, states, target
    return output


def join_channels(reg: pd.DataFrame, aggregated: pd.DataFrame) -> pd.DataFrame:
    """One exact sequence-state join; aggregation uses frozen QC and median."""
    if aggregated.duplicated(['WT_name', 'aa_seq']).any():
        raise ValueError('duplicate aggregated state')
    indexed = aggregated.set_index(['WT_name', 'aa_seq'])
    output = pd.DataFrame(reg[['sample_id']]).copy()
    for channel, column in [('combined', 'combined'), *gate.CHANNELS.items()]:
        for role, sequence in [('mutant', 'sequence'), ('wt', 'wildtype')]:
            keys = pd.MultiIndex.from_arrays([reg.unit_id, reg[sequence]])
            values = indexed[column].reindex(keys).to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f'missing {channel} {role} absolute state')
            output[f'{channel}_{role}'] = values
        label = 'ddg' if channel == 'combined' else f'ddg_{channel}'
        delta = output[f'{channel}_mutant'] - output[f'{channel}_wt']
        if not np.allclose(delta, reg[label], atol=1e-12, rtol=0):
            raise ValueError(f'channel delta disagrees with retained label: {channel}')
    return output


def validate_predictions(ids, predictions, expected_ids) -> np.ndarray:
    if not np.array_equal(np.asarray(ids), np.asarray(expected_ids)):
        raise ValueError('prediction sample order/coverage differs')
    predictions = np.asarray(predictions, float)
    if predictions.shape != (len(expected_ids),) or not np.isfinite(predictions).all():
        raise ValueError('missing/nonfinite held-out predictions')
    return predictions


def rank_target(target, groups) -> np.ndarray:
    """Average ties, within-background standardized ranks; not a raw-score proxy."""
    output = np.empty(len(target))
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        rank = rankdata(np.asarray(target)[rows], method='average')
        scale = rank.std()
        if scale == 0:
            raise ValueError(f'nonestimable rank target: {group}')
        output[rows] = (rank - rank.mean()) / scale
    return output


def rank_contrasts(target, baseline, augmented, groups) -> tuple[list, np.ndarray, list]:
    """Within-background Spearman; nonestimable groups reported, never dropped."""
    values, missing = [], []
    names = np.unique(groups).tolist()
    for name in names:
        rows = np.flatnonzero(groups == name)
        y, b, a = (np.asarray(v)[rows] for v in (target, baseline, augmented))
        if min(len(np.unique(y)), len(np.unique(b)), len(np.unique(a))) < 2:
            values.append(np.nan)
            missing.append(name)
        else:
            values.append(float(cast(Any, spearmanr(y, a))[0]) - float(cast(Any, spearmanr(y, b))[0]))
    return names, np.asarray(values), missing


def ranking_family(target, groups, ids, predictions: dict, arms: list, control: str) -> dict:
    """33 arms, shared 10,000 draws, seed-average contrasts within family."""
    from scripts.capability.interactions.run_residual_panel import simultaneous_bands
    from ..interactions.pairwise_epistasis import ROSTER
    if len(arms) != 33 or set(arms) != set(ROSTER) or set(predictions) != set(arms):
        raise ValueError('full declared 33-arm family is required; no partial-panel inference')
    if control not in ('S', 'S2'):
        raise ValueError('ranking control must be S or S2')
    matrix, nonestimable = [], {}
    for arm in arms:
        if set(predictions[arm]) != set(gate.SPLIT_SEEDS):
            raise ValueError('missing split predictions')
        splits = []
        for seed in gate.SPLIT_SEEDS:
            row = predictions[arm][seed]
            base = validate_predictions(row['sample_id'], row['baseline'], ids)
            aug = validate_predictions(row['sample_id'], row['augmented'], ids)
            _, contrast, missing = rank_contrasts(target, base, aug, groups)
            nonestimable[f'{arm}/{seed}'] = missing
            splits.append(contrast)
        matrix.append(np.mean(splits, axis=0))
    if any(nonestimable.values()):
        return {'status': 'blocked_nonestimable_backgrounds', 'nonestimable': nonestimable,
                'rule': 'no outcome-based background deletion or zero imputation'}
    return {'status': 'complete', 'control': control,
            'role': 'primary_rank_qualified' if control == 'S2' else 'same_prediction_MSE_comparison_sensitivity',
            'arms': arms, 'bands': simultaneous_bands(np.asarray(matrix).T, draws=10000)}


def verify_scalar_archive(directory: Path, arm: str, cohort: dict, cohort_hash: str) -> tuple[np.ndarray, dict]:
    """Native mutant-minus-WT scalar only; no token arrays or new inference."""
    path = directory / f'manifest_{arm}.json'
    manifest = json.loads(path.read_text())
    identity = manifest['identity']
    if (manifest['status'] != 'complete' or identity['arm'] != arm
            or identity['cohort_sha256'] != cohort_hash
            or identity['schema'] != 'stability_singles_likelihood_extraction_v1'):
        raise ValueError('scalar extraction identity mismatch')
    by_name = {row['background']: row for row in manifest['backgrounds']}
    if len(by_name) != len(cohort['backgrounds']):
        raise ValueError('scalar background support differs')
    scores = []
    for background in cohort['backgrounds']:
        entry = by_name[background['name']]
        archive = directory / entry['file']
        if sha256_file(archive) != entry['sha256']:
            raise ValueError('scalar archive hash mismatch')
        with np.load(archive, allow_pickle=False) as data:
            expected = [hashlib.sha256(s.encode()).hexdigest() for s in background['sequences']]
            if not np.array_equal(data['state_sequence_sha256'], expected):
                raise ValueError('scalar exact sequence-state alignment mismatch')
            states = np.asarray([v['state'] for v in background['variants']])
            if not np.array_equal(data['variant_states'], states):
                raise ValueError('scalar variant state order differs')
            if not np.array_equal(data['variant_positions'], [v['position'] for v in background['variants']]):
                raise ValueError('scalar mutation positions differ')
            likelihood = data['likelihood']
            if likelihood.shape != (len(expected),) or not np.isfinite(likelihood).all():
                raise ValueError('missing native scalar states')
            scores.extend((likelihood[states].astype(float) - float(likelihood[0])).tolist())
    return np.asarray(scores)[:, None], manifest


def fit_extension(panel: dict, sets: dict, partitions: dict, *, scalar=None, objective='mse') -> dict:
    """New CPU nested fits on recorded outer splits, not asserted recovered originals.

    Optional rank-target objective remains a separate sensitivity. G continues to
    calibrate absolute measurement labels using training groups only.
    """
    working = dict(panel)
    if objective == 'rank':
        working['target'] = rank_target(panel['target'], panel['group'])
    elif objective != 'mse':
        raise ValueError('unknown objective')
    blocks = dict(panel['blocks'])
    designs = dict(sets)
    if scalar is not None:
        if scalar.shape != (len(panel['target']), 1) or not np.isfinite(scalar).all():
            raise ValueError('incomplete scalar support')
        blocks['M'] = scalar
        designs.update({f'{key}_M': tuple(value) + ('M',) for key, value in sets.items()})
    output = {}
    for seed in gate.SPLIT_SEEDS:
        # Refuse partition drift before fitting anything.
        actual = family_folds(panel['group'], gate.OUTER_SPLITS, seed)
        if [set(x) for x in actual] != [set(x['held_groups']) for x in partitions[str(seed)]]:
            raise ValueError('recorded outer split drift')
        output[str(seed)] = gate.fold_predictions(working, blocks, designs, seed=seed, device='cpu')
    return output


def channel_reevaluation(reg: pd.DataFrame, panel: dict, families: dict, arms: list) -> dict:
    """Combined-trained OOF reevaluation: joint model/channel/difference family.

    S MSE and S2 ranking are distinct 99-contrast families. These are not channel
    trained fits; correlated channels and post-outcome model choice do not create
    independent replication or evidence of equivalence.
    """
    from scripts.capability.interactions.run_residual_panel import simultaneous_bands
    from ..interactions.pairwise_epistasis import ROSTER
    if set(arms) != set(ROSTER):
        raise ValueError('full 33-arm channel comparison is required')
    output = {}
    for control, metric in [('S', 'mse'), ('S2', 'spearman')]:
        matrix, columns, missing = [], [], {}
        for arm in arms:
            seeds = []
            for seed in gate.SPLIT_SEEDS:
                record = families[control][arm][seed]
                baseline = validate_predictions(record['sample_id'], record['baseline'], reg.sample_id)
                augmented = validate_predictions(record['sample_id'], record['augmented'], reg.sample_id)
                channels = []
                for channel in ('trypsin', 'chymotrypsin'):
                    target = reg[f'ddg_{channel}'].to_numpy()
                    if metric == 'mse':
                        _, base = gate.group_errors(target, baseline, panel['group'], panel['site'])
                        _, better = gate.group_errors(target, augmented, panel['group'], panel['site'])
                        contrast = base - better
                    else:
                        _, contrast, absent = rank_contrasts(target, baseline, augmented, panel['group'])
                        if absent:
                            missing[f'{arm}/{seed}/{channel}'] = absent
                    channels.append(contrast)
                seeds.append(np.column_stack([*channels, channels[0] - channels[1]]))
            values = np.mean(seeds, axis=0)
            matrix.extend(values.T)
            columns.extend([f'{arm}:trypsin', f'{arm}:chymotrypsin', f'{arm}:trypsin-minus-chymotrypsin'])
        output[control] = ({'status': 'blocked_nonestimable_backgrounds', 'nonestimable': missing} if missing
                           else {'status': 'complete', 'metric': metric, 'columns': columns,
                                 'bands': simultaneous_bands(np.asarray(matrix).T, draws=10000)})
    return {'mode': 'combined-trained OOF metric reevaluation, not channel-specific fits',
            'interpretation': 'not independent replication; nonsignificant differences do not establish equivalence',
            'families': output}


def matched_channel_family(reg: pd.DataFrame, panel: dict, predictions: dict, arms: list,
                           *, control: str, metric: str) -> dict:
    """Channel-trained paired increments, combined reference and channel difference.

    Four contrasts per arm form one declared 132-contrast family. Inputs must be
    separately channel-trained OOF, not combined predictions with changed labels.
    """
    from scripts.capability.interactions.run_residual_panel import simultaneous_bands
    from ..interactions.pairwise_epistasis import ROSTER
    if set(arms) != set(ROSTER) or set(predictions) != {'combined', 'trypsin', 'chymotrypsin'}:
        raise ValueError('complete 33-arm three-channel fits are required')
    if (control, metric) not in (('S', 'mse'), ('S2', 'spearman')):
        raise ValueError('use separately metric-qualified channel controls')
    columns, matrix, missing = [], [], {}
    for arm in arms:
        seeds = []
        for seed in gate.SPLIT_SEEDS:
            channels = []
            for channel in ('combined', 'trypsin', 'chymotrypsin'):
                record = predictions[channel][control][arm][seed]
                base = validate_predictions(record['sample_id'], record['baseline'], reg.sample_id)
                aug = validate_predictions(record['sample_id'], record['augmented'], reg.sample_id)
                target = reg['ddg' if channel == 'combined' else f'ddg_{channel}'].to_numpy()
                if metric == 'mse':
                    _, baseline = gate.group_errors(target, base, panel['group'], panel['site'])
                    _, augmented = gate.group_errors(target, aug, panel['group'], panel['site'])
                    contrast = baseline - augmented
                else:
                    _, contrast, absent = rank_contrasts(target, base, aug, panel['group'])
                    if absent:
                        missing[f'{channel}/{arm}/{seed}'] = absent
                channels.append(contrast)
            seeds.append(np.column_stack([*channels, channels[1] - channels[2]]))
        matrix.extend(np.mean(seeds, axis=0).T)
        columns.extend([f'{arm}:{name}' for name in ('combined', 'trypsin', 'chymotrypsin', 'trypsin-minus-chymotrypsin')])
    if missing:
        return {'status': 'blocked_nonestimable_backgrounds', 'nonestimable': missing}
    return {'status': 'complete', 'mode': 'channel-matched nested fits', 'control': control,
            'metric': metric, 'columns': columns,
            'bands': simultaneous_bands(np.asarray(matrix).T, draws=10000),
            'interpretation': 'correlated channels; selected frozen positives not independent confirmatory tests; no equivalence claim'}


def channel_qualification(panel: dict, values: pd.DataFrame, partitions: dict) -> dict:
    """New full CPU ladder; MSE and ranking eligibility are separate per channel."""
    output = {}
    for channel in ('combined', 'trypsin', 'chymotrypsin'):
        current = channel_panel(panel, values, channel)
        kept: list[str] = list(gate.BASE_BLOCKS)
        ladder = []
        for candidate in gate.CANDIDATE_BLOCKS:
            design = {'base': tuple(kept), 'candidate': (*kept, candidate)}
            outcome = fit_extension(current, design, partitions)
            mse, ranks = {}, {}
            for seed in gate.SPLIT_SEEDS:
                pred = outcome[str(seed)]['predictions']
                mse[seed] = gate.paired_increment(current, pred, 'candidate', 'base')['point']
                ranks[seed] = gate.spearman_increment(current, pred, 'candidate', 'base')
            verdict = gate.qualify(mse)
            ladder.append({'candidate': candidate, **verdict,
                           'spearman_increment': {str(s): ranks[s] for s in ranks}})
            if verdict['qualified']:
                kept.append(candidate)
        controls = {'qualified_control_set': kept, 'ladder': ladder,
                    'candidate_order': list(gate.CANDIDATE_BLOCKS), 'base_blocks': list(gate.BASE_BLOCKS)}
        secondary, evidence = gate.secondary_control_set(controls)
        output[channel] = {**controls, 'rank_qualified_control_set': list(secondary),
                           'rank_derivation': evidence,
                           'provenance': 'new local CPU channel qualification; not historical frozen qualification'}
    return output
