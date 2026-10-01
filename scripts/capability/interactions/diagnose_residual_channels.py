#!/usr/bin/env python3
"""Post-outcome channel diagnostic for the frozen adjusted pairwise target.

Neither construction supplies independent repeats of the combined residual.
Channel-specific correction changes the estimand; common correction introduces
shared dependence. The existing target and model-panel decisions never change.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.capability.stability import qualify_megascale_label_instrument as instrument
from scripts.capability.interactions.refit_likelihood_on_residual import residual_targets
from src.capability.interactions import pairwise_epistasis as model
from src.capability.core.io import sha256_file


def agreement(values, groups, pairs):
    """Original channel moments; the fitted task's nested group/pair weights."""
    vectors = []
    for group in sorted(set(groups)):
        selected = groups == group
        vectors.append(np.mean([
            instrument._moments(*values[selected & (pairs == pair)].T)
            for pair in sorted(set(pairs[selected]))], axis=0))
    return instrument._bootstrap(vectors, [], 2000, 20260923)


def validated_channel_sources(root, cohort):
    """Only ordered row shards are raw channels; other sources built the cohort."""
    pinned = {row['path']: row['sha256'] for row in cohort['source_files']}
    ordered = cohort['source_row_order']
    if len(pinned) != len(cohort['source_files']) or len(ordered) != len(set(ordered)):
        raise ValueError('duplicate source provenance')
    parts = []
    for relative in ordered:
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or path.suffix != '.parquet' or relative not in pinned:
            raise ValueError('ordered raw shard is not bound by cohort provenance')
        resolved = root/path
        if sha256_file(resolved) != pinned[relative]:
            raise ValueError('raw parquet differs from the frozen cohort source digest')
        parts.append(resolved)
    if not parts:
        raise ValueError('no ordered channel sources')
    return parts


def reconstruct_channels(frame, backgrounds):
    """Recover exact state rows and the combined and two channel cycle labels."""
    rows, groups, pairs = [], [], []
    for background in backgrounds:
        states = {}
        for sequence, record in background['measurements'].items():
            indices = [r['source_row'] for r in record['rows']]
            selected = frame.iloc[indices]
            if (not selected.WT_name.eq(background['name']).all()
                    or not selected.aa_seq.eq(sequence).all()
                    or selected.name.tolist() != [r['name'] for r in record['rows']]):
                raise ValueError('retained source-row identity differs')
            combined = pd.to_numeric(selected.dG_ML).to_numpy(float)
            if not np.isclose(np.median(combined), record['value'], rtol=0, atol=1e-12):
                raise ValueError('combined state value does not reproduce')
            states[sequence] = np.array([record['value'], selected.deltaG_t.median(), selected.deltaG_c.median()])
        for cycle in background['cycles']:
            y = np.array([states[s] for s in cycle['sequences']])
            additive = y[1]+y[2]-y[0]
            epsilon = y[3]-additive
            if not np.isclose(epsilon[0], cycle['epsilon'], rtol=0, atol=1e-12):
                raise ValueError('combined cycle does not reproduce')
            rows.append([additive, epsilon])
            groups.append(background['group'])
            pairs.append(background['name']+':'+str(cycle['positions']))
    values, groups, pairs = np.asarray(rows), np.asarray(groups), np.asarray(pairs)
    return values, groups, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('refuse existing diagnostic output')
    args.out.mkdir(parents=True)
    root = args.root
    cohort_path = root/'data/pairwise_epistasis/cohort_20260924_8133463e.json'
    declaration_path = root/'results/R3/pairwise_residual_full_panel_20260927/declaration.json'
    declaration = json.loads(declaration_path.read_text())
    if sha256_file(cohort_path) != declaration['input_sha256']['cohort']:
        raise ValueError('cohort differs from completed full-panel run')
    protocol = {
        'timing': 'post-outcome diagnostic declared after all33 model results were inspected',
        'target_unchanged': True, 'new_admission_threshold': None,
        'support': declaration['support_counts'], 'split_seeds': declaration['split_seeds'],
        'channel_specific': 'each channel epsilon minus its own outer-training isotonic response to its own additive',
        'shared_correction': 'both channel epsilon values minus the exact combined-target outer-training curve',
        'common_function_own_input': 'one combined-label outer-training curve evaluated separately at each held channel additive',
        'weighting': 'equal groups; equal site pairs within group; equal cycles within pair',
        'bootstrap': 'original moment decomposition,2000 paired group draws; fixed fitted correction',
        'limitation': 'No independent repeat of combined nonlinear target; channel mean is not assumed equal to combined residual.',
        'cohort_sha256': sha256_file(cohort_path),
        'full_panel_declaration_sha256': sha256_file(declaration_path),
        'script_sha256': sha256_file(Path(__file__)),
    }
    (args.out/'declaration.json').write_text(json.dumps(protocol, indent=2)+'\n')
    cohort = json.loads(cohort_path.read_text())
    parts = validated_channel_sources(root, cohort)
    frame = pd.concat([pd.read_parquet(p, columns=['name','WT_name','aa_seq','dG_ML','deltaG_t','deltaG_c'])
                       for p in parts], ignore_index=True)
    values, groups, pairs = reconstruct_channels(frame, cohort['backgrounds'])
    if values.shape != (8192,2,3) or len(set(groups)) != 64 or len(set(pairs)) != 217:
        raise ValueError('diagnostic support differs')
    if not np.isfinite(values).all():
        raise ValueError('nonfinite retained channel value')
    additive, epsilon = values[:,0,:], values[:,1,:]
    weights = model.row_weights(pairs, groups)
    report = {'protocol': protocol, 'pinned_input': {str(p.relative_to(root)): sha256_file(p) for p in parts},
              'raw_channels': agreement(epsilon[:,1:], groups, pairs), 'seeds': {}}
    fitted_receipt = json.loads((root/'results/R3/pairwise_residual_full_panel_20260927/residual_progen3-3b.json').read_text())
    for seed in declaration['split_seeds']:
        residual = np.full_like(epsilon, np.nan)
        own_input = np.full((len(epsilon),2), np.nan)
        for held in model.family_folds(groups, model.OUTER_SPLITS, seed):
            train = np.flatnonzero(~np.isin(groups, held))
            test = np.flatnonzero(np.isin(groups, held))
            for channel in range(3):
                _, residual[test,channel] = residual_targets(epsilon[:,channel], additive[:,channel], train, test)
            for channel in (1,2):
                curve, _ = model.measured_singles_predict(additive[train,0], epsilon[train,0], additive[test,channel])
                own_input[test,channel-1] = epsilon[test,channel]-curve
        common_curve = epsilon[:,0]-residual[:,0]
        shared = epsilon[:,1:]-common_curve[:,None]
        # The common correction cancels exactly in the channel difference.
        if not np.allclose(shared[:,0]-shared[:,1], epsilon[:,1]-epsilon[:,2], atol=1e-12, rtol=0):
            raise ValueError('shared correction changed channel discordance')
        def weighted(x): return float(np.sum(weights*x)/np.sum(weights))
        if not np.isclose(weighted(residual[:,0]**2), fitted_receipt['seeds'][str(seed)]['residual_group_equal_mse_kcal2']['point'], rtol=0, atol=1e-12):
            raise ValueError('diagnostic combined target differs from the completed model fitting target')
        report['seeds'][str(seed)] = {
            'channel_specific': agreement(residual[:,1:], groups, pairs),
            'shared_correction': agreement(shared, groups, pairs),
            'common_function_own_input': agreement(own_input, groups, pairs),
            'combined_target_mse_kcal2': weighted(residual[:,0]**2),
            'channel_specific_mean_minus_combined_rms_kcal': np.sqrt(weighted((residual[:,1:].mean(1)-residual[:,0])**2)),
            'shared_corrected_mean_minus_combined_rms_kcal': np.sqrt(weighted((shared.mean(1)-residual[:,0])**2)),
            'common_function_own_input_mean_minus_combined_rms_kcal': np.sqrt(weighted((own_input.mean(1)-residual[:,0])**2)),
        }
    (args.out/'diagnostic.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('RESIDUAL_CHANNEL_DIAGNOSTIC_EXIT=0', flush=True)


if __name__ == '__main__':
    main()
