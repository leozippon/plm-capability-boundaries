#!/usr/bin/env python3
"""Assemble an explicit fixed Potts source map using the frozen snapshot's scorer.

The map names every assigned checkpoint once. An independently produced q00059
final may replace only that final, preserving its frozen CV selection and original
failed files. No historical execution manifest or checkpoint is rewritten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'cannot read required JSON: {path}') from error


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_files(files):
    for path, expected in files.items():
        if digest(path) != expected:
            raise ValueError(f'changed source: {path}')


def recover_final(original, directory, variants, B):
    """Derive a scored record without modifying or relabelling the old producer."""
    from src.capability.stability.pairwise_stability import read_plmc

    directory = Path(directory)
    preparation, receipt = load(directory/'preparation.json'), load(directory/'receipt.json')
    if (original['wildtype_id'] != 'q00059' or original['status'] != 'final_not_converged_after_caps'
            or original['selected_lambda_e'] != 80 or original['lambda_h'] != .01
            or original['final']['status_text'] != 'ROUNDING_ERROR'
            or len(original['sweep']) != 5 or not all(e['converged'] for e in original['sweep'])):
        raise ValueError('not the declared q00059 final-only replacement')
    final = receipt.get('true_final_event') or {}
    if (receipt.get('scientific_success') is not True or receipt.get('original_bytes_unchanged') is not True
            or receipt.get('process_returncode') != 0 or final.get('accepted') != '1'
            or final.get('unquantized') != '1' or int(final.get('return_code', -1)) not in (0, 2)
            or not math.isfinite(float(final.get('ratio', 'nan')))
            or not 0 <= float(final['ratio']) <= .001
            or not 0 <= int(final.get('cumulative', -1)) <= 5000
            or receipt.get('history_resets') not in (0, 1)):
        raise ValueError('independent final did not pass its scientific convergence gate')
    if (preparation['reused_cv'] != original['sweep'] or preparation['selected_lambda_e'] != 80
            or preparation['lambda_h'] != .01 or preparation['total_outer_attempt_cap'] != 5000
            or preparation['producer'] != receipt['producer'] or receipt['producer']['testing']):
        raise ValueError('recovery changed selection or producer identity')
    check_files(preparation['original_files'])
    check_files({directory/k: v for k, v in preparation['inputs'].items()})
    check_files({directory/k: v for k, v in receipt['artifacts'].items()})
    build = Path(preparation['build_dir'])
    check_files({build/'plmc-recovery': receipt['producer']['binary_sha256']})
    check_files({build/k: v for k, v in receipt['producer']['patched_sources'].items()})
    params = directory/'final.params'
    if not {'final.params', 'final.double'} <= receipt['artifacts'].keys():
        raise ValueError('independent final artifacts missing')
    record = copy.deepcopy(original)
    record.pop('execution_manifest_sha256', None)
    record.pop('failure', None)
    record['status'] = B.STATUS_FITTED
    record['plmc'] = receipt['producer']
    record['params'] = dict(path=str(params), sha256=digest(params), bytes=params.stat().st_size)
    record['final'] = dict(status='converged_independent_producer', status_text=receipt['status'],
                           converged=True, max_iter=5000, iterations=int(final['cumulative']),
                           relative_gradient=float(final['ratio']), history_resets=receipt['history_resets'],
                           receipt_sha256=digest(directory/'receipt.json'))
    record['final_seconds'] = receipt['elapsed_seconds']
    wildtype = B.read_fasta(directory/'full.fa')[0][1]
    scorer = B.potts_scorer(read_plmc(params), wildtype)
    record['sign_check'] = B.shuffle_sign_check(scorer)
    record['sample_scores'] = {r['sample_id']: scorer.delta(B.mutant_codes(wildtype, r['substitutions']))
                               for r in variants}
    record['score_variation'] = dict(variants=len(variants), distinct_values=len(
        {round(v, 9) for v in record['sample_scores'].values()}))
    return record


def require_fixed_sources(mapping, partition, support):
    """A duplicate or unassigned checkpoint can never fill an assigned missing source."""
    if (set(mapping['wildtypes']) != set(support) or len(support) != 174
            or set(partition['source_role']) != set(support)):
        raise ValueError('source map must cover the fixed 174 wild types exactly')
    export = Path(partition['out'])
    for name, entry in mapping['wildtypes'].items():
        role = partition['source_role'][name]
        expected = export/partition['trees'][role]/'backgrounds'/f'{name}.json'
        if Path(entry['checkpoint']) != expected or entry['role'] != role:
            raise ValueError(f'{name}: not the assigned fixed source')
        if 'recovery' in entry and name != 'q00059':
            raise ValueError('only the predeclared q00059 replacement is authorized')


def publish_historical_completion(document):
    """Add the historical combined count without relabelling split failures.

    The current exporter separately counts final and CV non-convergence. The
    ladder's older contract requires their sum under one field; a recorded value
    that disagrees with that sum is an error, never a value to overwrite.
    """
    counts = document['counts']
    derived = (counts['backgrounds_final_not_converged']
               + counts['backgrounds_cv_not_converged'])
    recorded = counts.get('backgrounds_not_converged')
    if recorded not in (None, derived):
        raise ValueError('historical and split non-convergence counts disagree')
    counts['backgrounds_not_converged'] = derived
    return counts


def assemble(snapshot, source_map, out):
    snapshot, source_map, out = map(lambda p: Path(p).resolve(), (snapshot, source_map, out))
    if out.exists():
        raise ValueError('assembly never overwrites existing evidence')
    sys.path.insert(0, str(snapshot))
    R = importlib.import_module('scripts.capability.baselines.r1_msa_baselines')
    if R.__file__ is None or Path(R.__file__).resolve() != snapshot/'scripts/capability/baselines/r1_msa_baselines.py':
        raise ValueError('use a fresh process with the declared frozen snapshot')
    B = R.B
    mapping = load(source_map)
    partition_path = Path(mapping['partition'])
    if digest(partition_path) != mapping['partition_sha256']:
        raise ValueError('partition changed')
    partition = load(partition_path)
    export = Path(partition['out'])
    inputs = B.verify_inputs(export, root=snapshot)
    table = load(export/'inputs/variant_table.json')['samples']
    support = load(export/'inputs/support.json')['wildtypes']
    require_fixed_sources(mapping, partition, support)
    if len(table) != 25728:
        raise ValueError('variant table must hold the frozen 25728 samples')
    resources: dict = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                     disk_free=shutil.disk_usage(out.parent).free,
                     available_memory=R.available_memory(), affinity_cpus=len(os.sched_getaffinity(0)))
    if resources['disk_free'] < 2 << 30 or resources['available_memory'] < 4 << 30:
        raise ValueError('insufficient assembly headroom')
    records, sources, executions = {}, {}, {}
    by_wildtype = {n: [r for r in table if r['wildtype_id'] == n] for n in support}
    for name, entry in sorted(mapping['wildtypes'].items()):
        role = entry['role']
        path = Path(entry['checkpoint'])
        if digest(path) != entry['sha256']:
            raise ValueError(f'{name}: assigned checkpoint changed')
        execution_path = path.parent.parent/'execution_manifest.json'
        execution = load(execution_path)
        if role not in executions:
            B.verify_execution(execution, inputs=inputs,
                code_paths={k: snapshot/v for k, v in R.CODE.items()}, plmc_record=R.plmc_record())
            executions[role] = execution
        record = load(path)
        if (record['wildtype_id'] != name or record['execution_manifest_sha256'] != digest(execution_path)
                or digest(execution_path) != partition['execution_manifest_sha256']):
            raise ValueError(f'{name}: execution identity mismatch')
        check_files({record['params']['path']: record['params']['sha256']})
        provenance = dict(entry, execution_manifest_sha256=digest(execution_path))
        if 'recovery' in entry:
            directory = Path(entry['recovery'])
            if str(path) not in load(directory/'preparation.json')['original_files']:
                raise ValueError('recovery is not bound to the assigned original failure')
            record = recover_final(record, directory, by_wildtype[name], B)
            provenance.update(recovery_receipt=str(directory/'receipt.json'),
                              recovery_receipt_sha256=digest(directory/'receipt.json'),
                              preparation_sha256=digest(directory/'preparation.json'),
                              producer=record['plmc'], params=record['params'])
        if (record['status'] not in B.COMPLETE or record.get('final', {}).get('converged') is not True
                or ('recovery' not in entry and (record['final'].get('status') != 'converged'
                    or record['final'].get('status_text') != 'Minimization success'))
                or any(e.get('converged') is not True for e in record.get('sweep', []))):
            raise ValueError(f'{name}: failed/unconverged/pending fixed source')
        scores = record['sample_scores']
        if (set(scores) != {r['sample_id'] for r in by_wildtype[name]}
                or not all(math.isfinite(v) for v in scores.values())):
            raise ValueError(f'{name}: incomplete or nonfinite sample scores')
        records[name], sources[name] = record, provenance
    protocol = copy.deepcopy(next(iter(executions.values()))['optimization'])
    protocol['independent_final_producers'] = {n: e['producer'] for n, e in sources.items() if 'producer' in e}
    document, arrays = B.assemble_scores(table, records, wildtype_ids=sorted(support), protocol=protocol,
        inputs=dict(inputs_manifest_sha256=inputs['manifest']['sha256'],
                    variant_table_sha256=inputs['derived']['variant_table']['sha256'],
                    support_sha256=inputs['derived']['support']['sha256'],
                    source={k: v['sha256'] for k, v in inputs['source'].items()}))
    counts = publish_historical_completion(document)
    if not counts['complete'] or counts['backgrounds_fitted'] != 174:
        raise ValueError('174-source admission failed')
    document['source_map'] = dict(path=str(source_map), sha256=digest(source_map), wildtypes=sources)
    out.mkdir(parents=True, exist_ok=False)
    R.save(out/'admission.json', dict(resources=resources, counts=document['counts'],
                                    partition_sha256=digest(partition_path), sources=sources))
    R.save(out/'scores.json', document)
    R.np.savez_compressed(out/'scores.npz', **arrays)
    R.save(out/'completion.json', dict(status='complete', files={n: digest(out/n) for n in
                                      ('scores.json', 'scores.npz', 'admission.json')}))
    print(json.dumps(dict(out=str(out), counts=document['counts'])), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'source-map', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    assemble(a.snapshot, a.source_map, a.out)


if __name__ == '__main__':
    main()
