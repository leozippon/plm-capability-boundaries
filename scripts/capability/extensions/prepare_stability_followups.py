#!/usr/bin/env python3
"""Prepare exact stability CPU inputs, audit recovery, and run explicitly new fits.

No inference is implemented. See the generated procedure.json for blocked and
available operations. Outputs must remain in the stability extension directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import src.capability.extensions.stability_followups as follow
from src.capability.stability import stability_gate as gate
from src.capability.interactions.pairwise_epistasis import ROSTER
from src.capability.core.io import sha256_file

OUT = ROOT / 'results/extensions/phenotype_followups_20261007/stability'
FROZEN = ROOT / 'results/R4/gate_stability_20260924'
REPLAY = ROOT / 'results/shared/prediction_details_replay_20261005/L20reproduction'


def resources():
    return {'cpus': os.cpu_count(), 'disk_free_bytes': shutil.disk_usage(ROOT).free,
            'threads': torch.get_num_threads(), 'python': sys.version,
            'memory': [line.strip() for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith(('MemAvailable:', 'MemTotal:'))]}


def input_paths():
    return {'cohort': FROZEN / 'cohort/cohort.json',
             'profiles': FROZEN / 'cohort/profile_features.npz',
             'controls': FROZEN / 'controls/controls_qualification.json',
             'samples': ROOT / 'manuscript/data/prediction-details/stability-samples.csv.gz'}


def inputs():
    paths = input_paths()
    cohort = json.loads(paths['cohort'].read_text())
    if sha256_file(paths['cohort']) != follow.COHORT_SHA256:
        raise ValueError('frozen cohort bytes changed')
    samples = pd.read_csv(paths['samples'])
    if set(samples.cohort_sha256) != {follow.COHORT_SHA256}:
        raise ValueError('sample export cohort binding changed')
    reg = follow.registry(cohort, samples)
    counts = {'rows': len(reg), 'sites': reg.site.nunique(), 'groups': reg.group_id.nunique()}
    if counts != follow.EXPECTED or reg.unit_id.nunique() != 101:
        raise ValueError(f'exact single-mutant support changed: {counts}')
    controls = json.loads(paths['controls'].read_text())
    if controls['cohort_sha256'] != follow.COHORT_SHA256:
        raise ValueError('control cohort binding changed')
    profiles, _ = gate.load_profiles(paths['profiles'], {b['name']: b['wildtype'] for b in cohort['backgrounds']})
    panel = gate.build_panel(cohort, profiles)
    return paths, cohort, reg, panel, controls


def inspect_predictions(path, receipt, reg, partitions, paired):
    if sha256_file(path) != receipt['prediction_csv']['sha256']:
        raise ValueError('replay CSV digest mismatch')
    frame = pd.read_csv(path)
    if set(frame.seed) != set(gate.SPLIT_SEEDS):
        raise ValueError('replay split coverage changed')
    for seed in gate.SPLIT_SEEDS:
        rows = frame[frame.seed == seed]
        follow.validate_predictions(rows.sample_id, rows.baseline_prediction, reg.sample_id)
        if paired:
            follow.validate_predictions(rows.sample_id, rows.augmented_prediction, reg.sample_id)
        if not np.allclose(rows.phenotype, reg.ddg, rtol=0, atol=1e-12):
            raise ValueError('replay labels differ')
        if not rows.heldout.all():
            raise ValueError('in-sample predictions mixed into replay')
        membership = {g: part['fold'] for part in partitions[str(seed)] for g in part['held_groups']}
        if not np.array_equal(rows.heldout_fold, reg.group_id.map(membership)):
            raise ValueError('replay recorded held-out split differs')
    return {'rows': len(frame), 'exact_sample_label_outer_fold_alignment': True}


def prepare(out):
    paths, cohort, reg, panel, controls = inputs()
    sets, qualification = follow.control_sets(controls)
    historical_paths = {arm: ROOT / 'results/R4/stability_likelihood_replay_20260927' / f'{arm}.json' for arm in ROSTER}
    historical = {arm: json.loads(path.read_text()) for arm, path in historical_paths.items()}
    partitions = follow.recorded_partitions(panel['group'], historical[ROSTER[0]])
    for arm, fit in historical.items():
        if fit['baseline'] != 'S':
            raise ValueError('arm-specific tokenizer baseline needs exact additional feature binding; not silently omitted')
        if fit['cohort_sha256'] != follow.COHORT_SHA256:
            raise ValueError(f'{arm}: frozen fit input hashes differ')
        if follow.recorded_partitions(panel['group'], fit) != partitions:
            raise ValueError('model outer folds differ')
    reg.to_csv(out / 'sample-registry.csv.gz', index=False)
    follow.dump(out / 'partitions.json', partitions)
    # G is a source binding, deliberately not materialized using all labels.
    arrays = {f'block_{key}': value for key, value in panel['blocks'].items()}
    arrays.update(sample_id=reg.sample_id.to_numpy(str), target=panel['target'], group=panel['group'],
                  site=panel['site'], pair_states=panel['pair_states'],
                  state_features=panel['states']['features'], state_y=panel['states']['y'],
                  state_group=panel['states']['group'], state_weight=panel['states']['weight'])
    np.savez_compressed(out / 'design-inputs.npz', **arrays)
    data_dir = ROOT / 'data/megascale_tsuboyama2023/dataset2/data'
    frame = gate.load_source_frame(data_dir)
    mask, accounting = gate.accept_rows(frame)
    values = gate.aggregate_states(frame, mask, statistic='median')
    joined = follow.join_channels(reg, values)
    joined.to_csv(out / 'channel-absolute-labels.csv.gz', index=False)
    for channel in ('combined', 'trypsin', 'chymotrypsin'):
        follow.channel_panel(panel, joined, channel)
    reliability = []
    for group in np.unique(panel['group']):
        rows = reg.group_id == group
        a, b = reg.loc[rows, 'ddg_trypsin'], reg.loc[rows, 'ddg_chymotrypsin']
        reliability.append({'group': group, 'rows': int(rows.sum()),
                            'channel_spearman': float(cast(Any, spearmanr(a, b))[0]),
                            'mean_squared_channel_difference': float(np.mean((a - b) ** 2))})
    follow.dump(out / 'channel-reliability.json', {
        'support': follow.EXPECTED, 'absolute_and_delta_joins_verified': True,
        'aggregation': 'frozen substitution/censor/three-width QC; exact state median',
        'accounting': accounting, 'groups': reliability,
        'mean_family_channel_spearman': float(np.mean([x['channel_spearman'] for x in reliability])),
        'interpretation': 'correlated measurement channels, not independent replication; no equivalence claim'})
    models = []
    for arm in ROSTER:
        baseline_path = REPLAY / 'baseline-only-20261005/stability' / f'{arm}.json'
        if not baseline_path.exists():
            baseline_path = REPLAY / 'baseline-only-main-session/stability-remaining' / f'{arm}.json'
        baseline = json.loads(baseline_path.read_text())
        if (baseline['original_input_sha256']['controls'] != sha256_file(paths['controls'])
                or baseline['original_input_sha256']['profiles'] != sha256_file(paths['profiles'])):
            raise ValueError('replay frozen control/profile binding differs')
        base_csv = baseline_path.parent / baseline['prediction_csv']['file']
        base_check = inspect_predictions(base_csv, baseline, reg, partitions, False)
        local = REPLAY / 'likelihood-recovery/stability' / arm
        pair_path = REPLAY / 'likelihood-recovery/paired/stability' / arm / f'{arm}.json'
        record = {'arm': arm, 'historical_fit': follow.binding(historical_paths[arm], ROOT),
                  'historical_matched_baseline': historical[arm]['baseline'],
                  'baseline_only_receipt': follow.binding(baseline_path, ROOT),
                  'baseline_only_status': baseline['status'], 'baseline_only_check': base_check,
                  'original_scalar_status': 'unavailable_locally', 'original_OOF_status': 'unavailable_locally',
                  'required_state_registry': 'sample-registry.csv.gz', 'required_states': 25957,
                  'required_score': 'native per-state scalar likelihood (nats), mutant minus same-background WT',
                  'token_arrays_required': False,
                  'native_interface_checkpoint_binding': 'recover original extraction manifest identified by historical replay receipt; checkpoint weights/code equivalence not established by tensor file sizes',
                  'local_replay_code_bindings_not_asserted_original': baseline['code_sha256'],
                  'original_extraction_manifest_sha256': baseline['original_reference_hashes']['extraction_manifest_sha256']}
        if pair_path.exists():
            scalar, manifest = follow.verify_scalar_archive(local, arm, cohort, follow.COHORT_SHA256)
            paired = json.loads(pair_path.read_text())
            csv = pair_path.parent / paired['prediction_csv']['file']
            record.update(local_scalar_manifest=follow.binding(local / f'manifest_{arm}.json', ROOT),
                          local_scalar_identity=manifest['identity'], scalar_rows=len(scalar),
                          local_pair_receipt=follow.binding(pair_path, ROOT),
                          local_pair_check=inspect_predictions(csv, paired, reg, partitions, True),
                          local_pair_status=paired['status'],
                          feature_matches_original_manifest=paired['feature_matches_original_manifest'],
                          discrepancy_comparisons={s: row['comparison'] for s, row in paired['seeds'].items()},
                          extension_score_provenance='blocked: original identity mismatch unresolved; no independently verified tensor hashes/interface equivalence',
                          accepted_original=False)
        else:
            record.update(local_scalar_manifest=None, local_pair_receipt=None, accepted_original=False)
        models.append(record)
    missing = [m['arm'] for m in models if m['local_scalar_manifest'] is None]
    follow.dump(out / 'model-recovery-manifest.json', {
        'models': models, 'missing_local_scalar_models': missing, 'missing_count': len(missing),
        'local_unaccepted_count': 33 - len(missing), 'original_scalar_unavailable_count': 33,
        'baseline_only_count': len(models), 'likelihood_recovery': 'paused',
        'full_33_ranking': 'blocked; group MSE cannot reconstruct Spearman',
        'five_arm_exploratory_metrics': 'not run: no independently validated separate native-score provenance; S-only paired OOF does not supply S2',
        'frozen_MSE_positives_not_covered': ['progen3-3b', 'prollama']})
    support = {
        'schema': 'stability_followup_preparation_v1', 'counts': follow.EXPECTED,
        'sources': {k: follow.binding(p, ROOT) for k, p in paths.items()},
        'parquet_sources': [follow.binding(p, ROOT) for p in sorted(data_dir.glob('*.parquet'))],
        'control_sets': sets, 'secondary_control_derivation': qualification,
        'G': {'implementation': follow.binding(Path(gate.__file__), ROOT),
              'recipe': 'outer training states only; inner ridge crossfit then training isotonic calibration; absolute WT and mutant labels channel-specific',
              'static_matrix': False},
        'historical_replay_receipts': {a: follow.binding(p, ROOT) for a, p in historical_paths.items()},
        'historical': {'S_pointwise_all_seed_rank_positives': 13, 'S_rank_licensed': False,
                       'S2_pointwise_all_seed_rank_positives': ['progen3-3b'], 'simultaneous': False},
        'code': {str(p.relative_to(ROOT)): sha256_file(p) for p in [Path(__file__), ROOT / 'src/capability/extensions/stability_followups.py']}}
    follow.dump(out / 'procedure.json', {
        'CPU_interpreter': '/home/lzp/miniconda3/envs/ct/bin/python',
        'ranking': 'Use ranking_family with exact per-seed MSE-trained paired OOF, all 33 arms. Primary S2 and S same-prediction MSE-versus-ranking sensitivity are separate 33-arm families. T matching must follow each historical arm baseline, not be dropped. Average seeds inside family then 10000 shared family bootstrap draws. Nonestimable backgrounds block inference, never silently delete.',
        'optional_rank_fit': 'fit --objective rank is separately named new nested-fit sensitivity, not same-MSE-trained ranking. Preserve S/S2. Inner memberships regenerated from historical recipe; outer memberships verified recorded.',
        'channel_fit': 'qualify-channels runs independent new CPU full control ladders for combined/trypsin/chymotrypsin; fit --channel refits delta and channel absolute/WT G labels on identical rows/folds. Declare model x channel paired increments and channel difference before fitting with shared family draws.',
        'combined_OOF_channel_reevaluation': 'Evaluate existing combined-trained OOF against each channel target; label metric reevaluation, not channel-specific fitting. Same family bootstrap contrasts should include the paired channel difference. A nonsignificant difference is not equivalence.',
        'available_commands': ['prepare', 'calibration-pilot', 'qualify-channels', 'fit --objective mse|rank --channel combined|trypsin|chymotrypsin', 'rank-summary --prediction-manifest verified-full33.json', 'matched-summary --prediction-manifest verified-three-channel-full33.json'],
        'matched_summary_manifest_schema': 'map combined/trypsin/chymotrypsin to the OOF manifest schema below; S-MSE and S2-rank each declare a separate 132-contrast family (33 arms times combined/trypsin/chymotrypsin/paired channel difference)',
        'OOF_manifest_schema': {'cohort_sha256': follow.COHORT_SHA256, 'fit_objective': 'mse',
            'arms': {'ARM': {'path': 'absolute path to NPZ with sample_id and SEED|S, SEED|S_M, SEED|S2, SEED|S2_M',
                             'sha256': 'digest', 'receipt': {'path': 'absolute receipt path', 'sha256': 'digest'}}}},
        'channel_qualification_policy': 'channel-specific fit requires --channel-qualification from qualify-channels; no transfer of combined baseline qualification. Matched-summary requires identical qualified S and S2 block sets across all compared channels (channel-specific G parameters allowed); otherwise paired channel contrasts are blocked pending common-baseline qualification.',
        'scalar_provenance_gate': 'fit with scalars currently accepts only separate verified local provenance with exact checkpoint tensor SHA256s checked on disk, native-interface validation receipt and explicit mismatch acknowledgment. Existing five receipts do not satisfy this. Original OOF can be evaluated directly after recovery.',
        'no_inference': True, 'full_panel_status': 'blocked', 'five_arm_metrics_status': 'blocked'})
    support['prepared_artifacts'] = {p.name: sha256_file(p) for p in out.iterdir()
                                     if p.is_file() and p.name not in ('support.json', 'verification.json', 'diagnostics.json') and 'execution' not in p.name}
    follow.dump(out / 'support.json', support)
    print(json.dumps({'prepared': str(out), 'counts': follow.EXPECTED, 'missing_scalars': len(missing),
                      'unaccepted_local': 33 - len(missing)}), flush=True)


def qualified_channel_sets(artifact, channel):
    """Read only the exact, hash-bound qualification for this channel and labels."""
    if not isinstance(artifact, dict) or not artifact.get('path') or not artifact.get('sha256'):
        raise ValueError('missing channel qualification artifact binding')
    path = Path(artifact['path'])
    if sha256_file(path) != artifact['sha256']:
        raise ValueError('channel qualification artifact hash differs')
    qualified = json.loads(path.read_text())[channel]
    if (qualified.get('cohort_sha256') != follow.COHORT_SHA256
            or qualified.get('channel_labels_sha256') != sha256_file(OUT / 'channel-absolute-labels.csv.gz')):
        raise ValueError('channel qualification input binding differs')
    return {'S': tuple(qualified['qualified_control_set']),
            'S2': tuple(qualified['rank_qualified_control_set'])}


def load_OOF(declaration, channel, partitions):
    if declaration.get('cohort_sha256') != follow.COHORT_SHA256 or declaration.get('fit_objective') != 'mse':
        raise ValueError('exact-support MSE-trained OOF required')
    if set(declaration['arms']) != set(ROSTER):
        raise ValueError('all 33 arms required')
    families = {'S': {}, 'S2': {}}
    for arm in ROSTER:
        item = declaration['arms'][arm]
        path, receipt_path = Path(item['path']), Path(item['receipt']['path'])
        if sha256_file(path) != item['sha256'] or sha256_file(receipt_path) != item['receipt']['sha256']:
            raise ValueError('OOF provenance hash differs')
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('status') not in ('accepted_original_OOF', 'new_CPU_extension_not_frozen_recovery'):
            raise ValueError('unaccepted/mismatched OOF is not a verified ranking input')
        if (receipt.get('cohort_sha256') != follow.COHORT_SHA256 or receipt.get('arm') != arm
                or receipt.get('prediction_sha256') != item['sha256']
                or receipt.get('objective') != 'mse' or receipt.get('channel') != channel):
            raise ValueError('OOF receipt input/objective/endpoint identity differs')
        if channel == 'combined':
            controls = json.loads((FROZEN / 'controls/controls_qualification.json').read_text())
            expected, _ = follow.control_sets(controls)
        else:
            expected = qualified_channel_sets(receipt.get('channel_qualification'), channel)
        if receipt.get('control_sets') != {key: list(value) for key, value in expected.items()}:
            raise ValueError(f'{channel} S/S2 control qualification differs')
        if receipt.get('status') == 'new_CPU_extension_not_frozen_recovery' and not receipt.get('scalar_provenance'):
            raise ValueError('baseline-only OOF cannot provide model increments')
        for seed in gate.SPLIT_SEEDS:
            original = partitions[str(seed)]
            actual = receipt['folds'][str(seed)]
            if [set(x['held_groups']) for x in actual] != [set(x['held_groups']) for x in original]:
                raise ValueError('OOF outer fold identity differs')
        with np.load(path, allow_pickle=False) as data:
            for control in families:
                families[control][arm] = {seed: {'sample_id': data['sample_id'],
                    'baseline': data[f'{seed}|{control}'], 'augmented': data[f'{seed}|{control}_M']}
                    for seed in gate.SPLIT_SEEDS}
    return families


def require_common_channel_controls(declaration):
    """A channel contrast must not also change the qualified baseline features."""
    sets = [json.loads(Path(declaration[channel]['arms'][ROSTER[0]]['receipt']['path']).read_text())['control_sets']
            for channel in ('combined', 'trypsin', 'chymotrypsin')]
    if any(value != sets[0] for value in sets[1:]):
        raise ValueError('paired channel contrast blocked: common-baseline qualification required; '
                         'identical S/S2 block sets across channels required (channel-specific G parameters allowed)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'calibration-pilot', 'qualify-channels', 'fit', 'rank-summary', 'matched-summary'])
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--channel', choices=['combined', 'trypsin', 'chymotrypsin'], default='combined')
    parser.add_argument('--objective', choices=['mse', 'rank'], default='mse')
    parser.add_argument('--scalar-directory', type=Path)
    parser.add_argument('--arm', choices=list(ROSTER))
    parser.add_argument('--provenance', type=Path)
    parser.add_argument('--prediction-manifest', type=Path)
    parser.add_argument('--channel-qualification', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if out != OUT and OUT not in out.parents:
        raise ValueError('outputs must be within owned stability extension scope')
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    log = out / f'{args.operation}-execution.json'
    if log.exists():
        raise ValueError('refuse overwriting prior execution record; use a new child directory')
    record = {'operation': args.operation, 'started_resources': resources(), 'status': 'running', 'inference': False}
    follow.dump(log, record)
    try:
        if args.operation == 'qualify-channels':
            code_paths = [Path(__file__), Path(cast(str, follow.__file__)), Path(cast(str, gate.__file__)),
                          ROOT / 'src/capability/readouts/readout_analysis.py',
                          ROOT / 'src/capability/core/io.py',
                          ROOT / 'scripts/capability/interactions/run_residual_panel.py']
            source_paths = {**input_paths(), **{name: OUT / name for name in
                            ('support.json', 'partitions.json', 'channel-absolute-labels.csv.gz')}}
            record.update(code_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in code_paths},
                          input_bindings={key: {'path': str(p.resolve()), 'sha256': sha256_file(p)}
                                          for key, p in source_paths.items()})
            # Persist before loading panels or starting the long CPU ladder.
            follow.dump(log, record)
        if args.operation == 'prepare':
            prepare(out)
        else:
            _, cohort, reg, panel, controls = inputs()
            support = json.loads((OUT / 'support.json').read_text())
            for name in ('partitions.json', 'channel-absolute-labels.csv.gz'):
                if sha256_file(OUT / name) != support['prepared_artifacts'][name]:
                    raise ValueError('prepared label/split bytes changed')
            partitions = json.loads((OUT / 'partitions.json').read_text())
            labels = pd.read_csv(OUT / 'channel-absolute-labels.csv.gz')
            if not np.array_equal(labels.sample_id, reg.sample_id):
                raise ValueError('prepared channel label order changed')
            if args.operation == 'calibration-pilot':
                seed = gate.SPLIT_SEEDS[0]
                part = partitions[str(seed)][0]
                pilot = {}
                for channel in ('combined', 'trypsin', 'chymotrypsin'):
                    current = follow.channel_panel(panel, labels, channel)
                    block, diagnostic = gate.nuisance_response(current['states'], current['pair_states'],
                                                              np.asarray(part['held_groups']), part['inner_held_groups'])
                    pilot[channel] = {'diagnostics': diagnostic, 'shape': list(block.shape),
                                      'finite': bool(np.isfinite(block).all())}
                follow.dump(out / 'calibration-pilot.json', {'seed': seed, 'outer_fold': 0,
                            'channels': pilot, 'qualification': 'not performed; one-fold calibration pilot only',
                            'provenance': 'new CPU preparation; not original recovery'})
            elif args.operation == 'qualify-channels':
                qualification = follow.channel_qualification(panel, labels, partitions)
                for row in qualification.values():
                    row.update(cohort_sha256=follow.COHORT_SHA256,
                               channel_labels_sha256=record['input_bindings']['channel-absolute-labels.csv.gz']['sha256'])
                follow.dump(out / 'channel-qualification.json', qualification)
            elif args.operation == 'rank-summary':
                if not args.prediction_manifest:
                    raise ValueError('full-panel verified OOF manifest required; no reconstruction from group MSE')
                declaration = json.loads(args.prediction_manifest.read_text())
                families = load_OOF(declaration, 'combined', partitions)
                for control, predictions in families.items():
                    report = follow.ranking_family(panel['target'], panel['group'], reg.sample_id, predictions, list(ROSTER), control)
                    follow.dump(out / f'{control}-ranking.json', report)
                from scripts.capability.interactions.run_residual_panel import simultaneous_bands
                matrix = []
                for arm in ROSTER:
                    splits = []
                    for seed in gate.SPLIT_SEEDS:
                        row = families['S'][arm][seed]
                        _, base = gate.group_errors(panel['target'], row['baseline'], panel['group'], panel['site'])
                        _, aug = gate.group_errors(panel['target'], row['augmented'], panel['group'], panel['site'])
                        splits.append(base - aug)
                    matrix.append(np.mean(splits, axis=0))
                follow.dump(out / 'S-same-OOF-MSE.json', {'arms': list(ROSTER),
                    'role': 'MSE companion on the exact same S predictions; separate from S2 primary ranking family',
                    'bands': simultaneous_bands(np.asarray(matrix).T, draws=10000)})
                follow.dump(out / 'combined-trained-channel-reevaluation.json',
                            follow.channel_reevaluation(reg, panel, families, list(ROSTER)))
            elif args.operation == 'matched-summary':
                if not args.prediction_manifest:
                    raise ValueError('full three-channel verified OOF manifest required')
                declaration = json.loads(args.prediction_manifest.read_text())
                families = {channel: load_OOF(declaration[channel], channel, partitions)
                            for channel in ('combined', 'trypsin', 'chymotrypsin')}
                require_common_channel_controls(declaration)
                for control, metric in [('S', 'mse'), ('S2', 'spearman')]:
                    follow.dump(out / f'{control}-channel-matched.json',
                        follow.matched_channel_family(reg, panel, families, list(ROSTER), control=control, metric=metric))
            else:
                sets, _ = follow.control_sets(controls)
                qualification_binding = None
                if args.channel != 'combined':
                    if not args.channel_qualification:
                        raise ValueError('channel-matched fits require new channel-specific qualification')
                    qualification_binding = {'path': str(args.channel_qualification.resolve()),
                                             'sha256': sha256_file(args.channel_qualification)}
                    sets = qualified_channel_sets(qualification_binding, args.channel)
                scalar = None
                if args.scalar_directory:
                    if not args.arm or not args.provenance:
                        raise ValueError('scalar fit requires arm and verified provenance')
                    proof = json.loads(args.provenance.read_text())
                    scalar, manifest = follow.verify_scalar_archive(args.scalar_directory, args.arm, cohort, follow.COHORT_SHA256)
                    digest = sha256_file(args.scalar_directory / f'manifest_{args.arm}.json')
                    if proof.get('manifest_sha256') != digest or proof.get('arm') != args.arm:
                        raise ValueError('provenance does not bind native scalar manifest')
                    if proof.get('status') != 'verified_local_extension' or not proof.get('acknowledged_not_frozen_recovery'):
                        raise ValueError('only separately verified local extension scoring is allowed here')
                    tensors = proof.get('checkpoint_tensor_sha256', {})
                    expected_tensors = {record['name'] for record in manifest['identity']['checkpoint_tensor_files']}
                    if not tensors or set(tensors) != expected_tensors:
                        raise ValueError('missing verified complete checkpoint tensor hashes')
                    for name, expected in tensors.items():
                        if sha256_file(Path(manifest['identity']['checkpoint_path']) / name) != expected:
                            raise ValueError('checkpoint tensor digest differs')
                    if not proof.get('native_interface_validation_receipt'):
                        raise ValueError('missing independent native interface validation receipt')
                    receipt_path = Path(proof['native_interface_validation_receipt']['path'])
                    if sha256_file(receipt_path) != proof['native_interface_validation_receipt']['sha256']:
                        raise ValueError('native interface validation receipt digest differs')
                    interface = json.loads(receipt_path.read_text())
                    if interface.get('arm') != args.arm or interface.get('status') != 'validated' or interface.get('manifest_sha256') != digest:
                        raise ValueError('native interface validation is not a validated record for these scores')
                current = follow.channel_panel(panel, labels, args.channel)
                outcome = follow.fit_extension(current, sets, partitions, scalar=scalar, objective=args.objective)
                arrays = {'sample_id': reg.sample_id.to_numpy(str)}
                for seed, row in outcome.items():
                    arrays.update({f'{seed}|{key}': value for key, value in row['predictions'].items()})
                np.savez_compressed(out / 'new-extension-OOF.npz', **arrays)
                follow.dump(out / 'fit-receipt.json', {'channel': args.channel, 'objective': args.objective,
                    'control_sets': sets, 'channel_qualification': qualification_binding,
                    'status': 'new_CPU_extension_not_frozen_recovery',
                    'arm': args.arm, 'cohort_sha256': follow.COHORT_SHA256,
                    'prediction_sha256': sha256_file(out / 'new-extension-OOF.npz'),
                    'inner_provenance': 'regenerated historical recipe, not verified original serialized memberships',
                    'scalar_provenance': {'path': str(args.provenance), 'sha256': sha256_file(args.provenance)} if args.provenance else None,
                    'folds': {seed: row['folds'] for seed, row in outcome.items()},
                    'nuisance': {seed: row['nuisance'] for seed, row in outcome.items()}})
        record['status'] = 'complete'
    except Exception as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        record.update(seconds=time.monotonic() - started, finished_resources=resources())
        follow.dump(log, record)


if __name__ == '__main__':
    main()
