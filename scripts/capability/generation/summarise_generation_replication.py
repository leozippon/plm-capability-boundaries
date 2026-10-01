#!/usr/bin/env python3
"""Compare three fixed-checkpoint sampling campaigns without pooling seed units."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import t
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.capability.generation.generation_replication import sha256, write_json
from src.capability.generation.generative_control import wilson_interval


def campaign_interval(values, *, campaigns=3):
    values = np.asarray(values, dtype=float)
    if campaigns not in (2, 3) or values.shape != (campaigns,) or not np.isfinite(values).all():
        raise ValueError(f'exactly {campaigns} finite campaign estimates required')
    mean = float(values.mean())
    sd = 0.0 if np.ptp(values) == 0 else float(values.std(ddof=1))
    # Python scalars, not numpy ones: a numpy comparison below would yield a
    # numpy.bool_, which json.dumps cannot serialise and which no writer in this
    # repository coerces. Every field here has to survive serialisation.
    half_width = float(t.ppf(.975, campaigns - 1)) * sd / float(np.sqrt(campaigns))
    interval = None if sd == 0 else [mean - half_width, mean + half_width]
    direction = 'positive' if (values > 0).all() else 'negative' if (values < 0).all() else 'not_replicated'
    return {'values': values.tolist(), 'mean': mean, 'range': [float(values.min()), float(values.max())],
            'ci95': interval, 'n_campaigns': campaigns, 'degrees_of_freedom': campaigns - 1,
            'standard_deviation': sd, 'direction': direction,
            'resolved_across_campaigns': interval is not None and (interval[0] > 0 or interval[1] < 0),
            'interval_status': 'student_t_small_sample_approximation' if interval is not None else 'no_observed_campaign_variation_not_zero_uncertainty'}


def historical_lengths(historical, build, cell_directory):
    expected = next(v for k, v in historical['input_sha256'].items() if k.endswith('/build_manifest.json'))
    if sha256(build) != expected:
        raise ValueError('historical build manifest digest mismatch')
    result, sources = {}, {str(build): expected}
    for spec in json.loads(build.read_text())['cells']:
        path = cell_directory / (spec['cell'] + '.jsonl')
        if sha256(path) != spec['cell_sha256']:
            raise ValueError(f'historical cell digest mismatch: {spec["cell"]}')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 800 or len({r['attempt_id'] for r in rows}) != 800:
            raise ValueError('historical length census is not 800 unique attempts')
        lengths = [r['parent_length'] for r in rows]
        if any(r['parent_length'] != len(r['sequences']['generated']) for r in rows):
            raise ValueError('historical sequence length mismatch')
        result[spec['cell']] = {'mean': float(np.mean(lengths)), 'median': float(np.median(lengths)),
            'q25': float(np.quantile(lengths, .25)), 'q75': float(np.quantile(lengths, .75)),
            'min': int(min(lengths)), 'max': int(max(lengths)), 'n_attempts': len(rows)}
        sources[str(path)] = spec['cell_sha256']
    return result, sources


def summarise(manifest, historical, profiles, old_lengths):
    old = {r['cell']: r for r in historical['cells']}
    cells = []
    for spec in manifest['cells']:
        cell = spec['cell']
        source = old[cell]
        reports = [profiles[(cell, c['id'])] for c in manifest['campaigns']]
        if any(r['comparison']['n_attempts'] != 800 for r in reports) or source['n_attempts'] != 800:
            raise ValueError('campaign support mismatch')
        endpoints = {}
        for name in ['any_family', 'complete_domain']:
            retained = source['endpoints'][name]
            first = {'model_rate': retained['model_rate'],
                     'fragment_rate': retained['controls']['fragment']['control_rate'],
                     'model_minus_fragment': retained['controls']['fragment']['difference']}
            endpoints[name] = {metric: campaign_interval([first[metric], *[r['comparison']['endpoints'][name][metric] for r in reports]])
                               for metric in first}
            endpoints[name]['counts'] = {'model': [retained['model_successes'], *[r['comparison']['endpoints'][name]['model_successes'] for r in reports]],
                                         'fragment': [retained['controls']['fragment']['control_successes'], *[r['comparison']['endpoints'][name]['fragment_successes'] for r in reports]],
                                         'denominators': [800, 800, 800]}
            endpoints[name]['within_campaign_model_wilson95'] = [
                list(wilson_interval(k, 800)) for k in endpoints[name]['counts']['model']]
        lengths = [old_lengths[cell], *[r['comparison']['length_residues'] for r in reports]]
        stops = {stop: campaign_interval([r['comparison']['decoder_stop_counts'].get(stop, 0) / 800
                                         for r in reports], campaigns=2)
                 for stop in ['eos', 'max_new_tokens']}
        for r in reports:
            if sum(r['comparison']['decoder_stop_counts'].values()) != 800 or set(r['comparison']['decoder_stop_counts']) - {'eos', 'max_new_tokens'}:
                raise ValueError('decoder termination census is incomplete or unknown')
        cells.append({'cell': cell, 'arm': spec['arm'], 'condition': spec['condition'], 'endpoints': endpoints,
                      'new_campaign_diagnostics': [{'campaign': r['campaign'], 'comparison': r['comparison'], 'all_attempts': r['all_attempts']} for r in reports],
                      'length_residues': {'campaign_descriptive': lengths,
                          'campaign_mean': campaign_interval([r['mean'] for r in lengths])},
                      'decoder_stop_rates_two_new_campaigns': stops,
                      'historical_decoder_stop': 'unknown_without_exact_token_traces'})
    by_cell = {c['cell']: c for c in cells}
    stage = {}
    for endpoint in ['any_family', 'complete_domain']:
        stage[endpoint] = {}
        for metric in ['model_rate', 'model_minus_fragment']:
            one = by_cell['prollama-stage-1__unconditioned']['endpoints'][endpoint][metric]['values']
            two = by_cell['prollama__unconditioned']['endpoints'][endpoint][metric]['values']
            stage[endpoint][metric] = campaign_interval(np.asarray(two) - np.asarray(one))
    return {'cells': cells, 'stage2_minus_stage1_unconditional': stage,
            'units': 'recognition probability; contrasts are probability differences',
            'interval_scope': 'three generation streams at fixed checkpoints and declared decoding; fragment draws vary by campaign; not training-lineage uncertainty',
            'multiplicity': 'descriptive census; no confirmatory familywise superiority claim',
            'within_campaign_interval_scope': 'marginal model-rate Wilson intervals condition on each stream and use a working independent-attempt assumption; they do not represent between-campaign or training-lineage uncertainty',
            'decoder_stop_campaign_order': [c['id'] for c in manifest['campaigns']],
            'campaign_order': ['historical', *[c['id'] for c in manifest['campaigns']]]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=(Path(__file__).resolve().parents[3] / 'configs/generation_replication_manifest.json'))
    p.add_argument('--historical', type=Path, required=True)
    p.add_argument('--profiles', type=Path, required=True)
    p.add_argument('--historical-build', type=Path, required=True)
    p.add_argument('--historical-cells', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    expected_hist = next(v for k, v in manifest['historical_inputs'].items() if k.endswith('/gate_endpoints.json'))
    if sha256(a.historical) != expected_hist:
        raise ValueError('historical endpoint digest differs from declaration')
    profiles = {}
    sources = {}
    for spec in manifest['cells']:
        for campaign in manifest['campaigns']:
            path = a.profiles / f'{campaign["id"]}__{spec["cell"]}' / 'generation_profiles.json'
            report = json.loads(path.read_text())
            if report['cell'] != spec['cell'] or report['campaign'] != campaign['id']:
                raise ValueError('profile report identity mismatch')
            if report['configuration']['manifest_sha256'] != sha256(a.manifest):
                raise ValueError('profile report declaration mismatch')
            profiles[(spec['cell'], campaign['id'])] = report
            sources[str(path)] = sha256(path)
    historical = json.loads(a.historical.read_text())
    lengths, length_sources = historical_lengths(historical, a.historical_build, a.historical_cells)
    result = summarise(manifest, historical, profiles, lengths)
    result['manifest_sha256'] = sha256(a.manifest)
    result['code_sha256'] = sha256(Path(__file__).resolve().parents[3] / 'CODE_CONTENT_SHA256SUMS')
    result['source_sha256'] = {str(a.historical): expected_hist, **sources, **length_sources}
    write_json(a.out / 'generation_campaign_summary.json', result)
    lines = ['# Independent generation campaigns', '', 'Rates use all 800 comparison attempts in each campaign. Three-stream Student t intervals are exploratory small-sample approximations, not independent-training uncertainty. Every campaign value and range is retained in JSON; no observed variation does not establish zero uncertainty. Decoder termination has only two traced campaigns. Within-campaign Wilson intervals use a separate working independent-attempt assumption.', '', '| Cell | Any-family counts (three campaigns) | Complete-domain counts | Complete-domain minus fragment mean [95% campaign interval] |', '|---|---|---|---|']
    for cell in result['cells']:
        ep = cell['endpoints']
        contrast = ep['complete_domain']['model_minus_fragment']
        interval = contrast['ci95']
        ci = 'unestimated (no observed variation)' if interval is None else f'[{interval[0]:+.4f}, {interval[1]:+.4f}]'
        lines.append(f"| {cell['cell']} | {ep['any_family']['counts']['model']} / 800 | {ep['complete_domain']['counts']['model']} / 800 | {contrast['mean']:+.4f} {ci} |")
    lines.extend(['', 'Stage 2 minus Stage 1, unconditional matched decoding:'])
    for endpoint, metrics in result['stage2_minus_stage1_unconditional'].items():
        value = metrics['model_rate']
        lines.append(f"- {endpoint}: campaign differences {value['values']}; mean {value['mean']:+.4f}, 95% campaign interval {value['ci95']}; direction {value['direction']}.")
    (a.out / 'generation_campaign_summary.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'cells': len(result['cells']), 'campaigns_per_cell': 3, 'passed': True}))


if __name__ == '__main__':
    main()
