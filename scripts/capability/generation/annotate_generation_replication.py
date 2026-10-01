#!/usr/bin/env python3
"""Apply the fixed profile oracle and paired fragment control to every new attempt."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.capability.generation import generative_control as gc
from src.capability.generation import generation_evidence as ge
from src.capability.generation.generation_replication import sha256, write_json


def fragment_for(row, pool):
    length = len(row['sequence'])
    if not length:
        return '', {'reason': 'empty_attempt_retained'}
    material = f"generation_replication|{row['campaign_seed']}|{row['id']}|fragment"
    seed = int.from_bytes(hashlib.blake2b(material.encode(), digest_size=8).digest(), 'big')
    rng = np.random.default_rng(seed)
    names, weights, eligible = [], [], {}
    for name, records in pool['strata'].items():
        candidates = [i for i, r in enumerate(records) if len(r) >= length]
        if candidates and pool['population'][name]:
            names.append(name)
            eligible[name] = candidates
            weights.append(pool['population'][name] * len(candidates) / len(records))
    if not names:
        raise ValueError(f'no fragment donor for length {length}')
    probabilities = np.asarray(weights, dtype=float)
    probabilities /= probabilities.sum()
    name = names[int(rng.choice(len(names), p=probabilities))]
    donor = pool['strata'][name][int(rng.choice(eligible[name]))]
    fragment = gc.corpus_fragment(rng, donor, length)
    if len(fragment) != length or fragment not in donor:
        raise ValueError('fragment length/contiguity failure')
    return fragment, {'seed': seed, 'donor_stratum': name, 'donor_length': len(donor),
                      'donor_sha256': hashlib.sha256(donor.encode()).hexdigest()}


def scan(task):
    hmmscan, pfam, fasta, out, expected_hash = task
    tbl = out.with_suffix('.tbl')
    dom = out.with_suffix('.domtbl')
    marker = Path(str(tbl) + '.done')
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['fasta_sha256'] != expected_hash or saved['tbl_sha256'] != sha256(tbl) or saved['domtbl_sha256'] != sha256(dom):
            raise ValueError(f'oracle resume digest mismatch: {marker}')
        return
    command = [str(hmmscan), '--tblout', str(tbl), '--domtblout', str(dom), '--noali',
               '--cut_ga', '--cpu', '1', '-o', '/dev/null', str(pfam), str(fasta)]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'hmmscan failed {result.returncode}: {result.stderr[-2000:]}')
    write_json(marker, {'command': command, 'fasta_sha256': expected_hash,
                       'tbl_sha256': sha256(tbl), 'domtbl_sha256': sha256(dom),
                       'elapsed_seconds': time.monotonic() - started})


def descriptive(rows):
    lengths = np.asarray([r['length'] for r in rows], dtype=float)
    endpoints = {}
    for endpoint in ['any_family', 'complete_domain']:
        m = sum(r['profile']['generated'][endpoint] for r in rows)
        c = sum(r['profile']['fragment'][endpoint] for r in rows)
        endpoints[endpoint] = {'model_successes': m, 'fragment_successes': c, 'n_attempts': len(rows),
                               'model_rate': m / len(rows), 'fragment_rate': c / len(rows),
                               'model_minus_fragment': (m - c) / len(rows)}
    strata = []
    for low, high in [(0, 15), (16, 128), (129, 256), (257, 512), (513, None)]:
        selected = [r for r in rows if r['length'] >= low and (high is None or r['length'] <= high)]
        strata.append({'length_bounds_residues': [low, high], 'n_attempts': len(selected),
                       'model_any_family_successes': sum(r['profile']['generated']['any_family'] for r in selected),
                       'model_complete_domain_successes': sum(r['profile']['generated']['complete_domain'] for r in selected)})
    return {'n_attempts': len(rows), 'endpoints': endpoints,
            'length_residues': {'mean': float(lengths.mean()), 'median': float(np.median(lengths)),
                                'q25': float(np.quantile(lengths, .25)), 'q75': float(np.quantile(lengths, .75)),
                                'min': int(lengths.min()), 'max': int(lengths.max())},
            'decoder_stop_counts': dict(Counter(r['decoder_stop'] for r in rows)),
            'native_delimiter_count': sum(r['native_delimiter_observed'] for r in rows),
            'length_strata': strata}


def run(args):
    manifest = json.loads(args.manifest.read_text())
    oracle_files = [args.hmmscan, *sorted(args.pfam.parent.glob(args.pfam.name + '*'))]
    expected = {Path(p).name: value for p, value in manifest['oracle_inputs'].items()}
    hashes = {p.name: sha256(p) for p in oracle_files}
    if hashes != expected:
        raise ValueError('oracle executable/database bytes differ from predeclared release')
    expected_pool = next(v for k, v in manifest['historical_inputs'].items() if k.endswith('/reservoir.json'))
    if sha256(args.pool) != expected_pool:
        raise ValueError('fragment reservoir differs from frozen historical pool')
    pool = json.loads(args.pool.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    config = {'manifest_sha256': sha256(args.manifest), 'attempts_sha256': sha256(args.attempts),
              'code_sha256': sha256(Path(__file__).resolve().parents[3] / 'CODE_CONTENT_SHA256SUMS'),
              'oracle_sha256': hashes, 'reservoir_sha256': expected_pool,
              'domain_coverage_threshold': .8, 'threshold': '--cut_ga', 'threads_per_shard': 1}
    write_json(args.out / 'oracle_configuration.json', config)
    rows = [json.loads(line) for line in args.attempts.read_text().splitlines()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate attempt identifiers')
    cell_name = rows[0]['source_label']
    cell = next(c for c in manifest['cells'] if c['cell'] == cell_name)
    if len(rows) != cell['attempts'] or any(r['source_label'] != cell_name for r in rows):
        raise ValueError('attempt census differs from manifest')
    pairs = []
    sequences = {}
    for row in rows:
        fragment, provenance = fragment_for(row, pool)
        record = {'attempt_id': row['id'], 'generated': row['sequence'], 'fragment': fragment,
                  'fragment_provenance': provenance}
        pairs.append(record)
        for sequence in [row['sequence'], fragment]:
            if sequence:
                key = hashlib.sha256(sequence.encode()).hexdigest()
                sequences[key] = sequence
    ge.write_immutable(args.out / 'paired_fragments.jsonl', ge.jsonl_bytes(pairs))
    (args.out / 'shards').mkdir(exist_ok=True)
    (args.out / 'oracle').mkdir(exist_ok=True)
    names = {key: f'q{i:08d}' for i, key in enumerate(sorted(sequences))}
    write_json(args.out / 'query_names.json', names)
    shards = []
    keys = sorted(sequences)
    for i, start in enumerate(range(0, len(keys), args.shard_size)):
        selected = keys[start:start + args.shard_size]
        path = args.out / 'shards' / f'shard_{i:05d}.fasta'
        ge.write_immutable(path, ''.join(f'>{names[k]}\n{sequences[k]}\n' for k in selected).encode())
        shards.append({'path': str(path), 'sha256': sha256(path), 'n': len(selected)})
    write_json(args.out / 'build_manifest.json', {'shards': shards})
    tasks = [(args.hmmscan, args.pfam, Path(s['path']), args.out / 'oracle' / Path(s['path']).stem, s['sha256']) for s in shards]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(scan, tasks))
    oracle = gc.collect_oracle(args.out)
    for row, pair in zip(rows, pairs):
        row['profile'] = {}
        for cohort in ['generated', 'fragment']:
            sequence = pair[cohort]
            info = oracle.get(hashlib.sha256(sequence.encode()).hexdigest(), {}) if sequence else {}
            families = info.get('families', [])
            coverage = info.get('best_profile_coverage')
            row['profile'][cohort] = {'any_family': bool(families), 'complete_domain': bool(families) and coverage is not None and coverage >= .8,
                                      'families': families, 'best_profile_coverage': coverage}
    ge.write_immutable(args.out / 'annotated_attempts.jsonl', ge.jsonl_bytes(rows))
    comparison = [r for r in rows if r['comparison_selected']]
    if len(comparison) != 800:
        raise ValueError('comparison support is not exactly 800 attempts')
    report = {'cell': cell_name, 'campaign': rows[0]['campaign'], 'comparison': descriptive(comparison),
              'all_attempts': descriptive(rows), 'configuration': config,
              'annotated_attempts_sha256': sha256(args.out / 'annotated_attempts.jsonl')}
    write_json(args.out / 'generation_profiles.json', report)
    print(json.dumps(report, sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=(Path(__file__).resolve().parents[3] / 'configs/generation_replication_manifest.json'))
    p.add_argument('--attempts', type=Path, required=True)
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--hmmscan', type=Path, required=True)
    p.add_argument('--pfam', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--shard-size', type=int, default=200)
    p.add_argument('--device', default='cpu')
    run(p.parse_args())


if __name__ == '__main__':
    main()
