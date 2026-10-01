#!/usr/bin/env python3
"""Vary the readout class on the frozen readout states of one admitted panel cell.

One invocation is one (arm, panel, split seed) cell of the admitted readout panel.
It reloads that cell's retained full-width mutant-minus-wild-type states, refits
the admitted compressed linear class to reproduce the admitted predictions exactly,
then fits the declared stronger classes on identical rows, supports, cluster folds,
weights, targets and resampling contract. It performs no model inference.

The panel, the assay support, the cohort and the admitted comparison report are
all read from the frozen expansion roster, which is bound by SHA-256 so a cell
cannot silently be fitted against a different panel definition.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.capability.readouts.readout_class_sweep import (BOOTSTRAP_DRAWS, CLASSES, DEPTH_CLASSES,
                                              PROJECTION_SEED, REPRODUCTION_CLASS,
                                              evaluate_class_sweep, load_panel,
                                              verify_against_admitted)

#: The admitted full-available-panel roster this sweep is bound to. Its digest is
#: the one the passed final admission receipt records for `expected.json`.
ROSTER_SHA256 = '1bf26a6ae8d71e3c0f790f12c92da7d3dd4a9add7be6f7ef807836297e12dd0c'
CODE_FILES = ('scripts/capability/readouts/analyse_readout_class_sweep.py',
              'src/capability/readouts/readout_class_sweep.py',
              'src/capability/readouts/readout_analysis.py',
              'src/capability/context/profile_increment.py',
              'src/capability/context/profiles.py')


def digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_cell(roster_path: Path, arm: str, panel: str, seed: int):
    """Bind the roster, then return this cell's cohort, manifest, support and baseline."""
    observed = digest(roster_path)
    if observed != ROSTER_SHA256:
        raise ValueError(f'roster digest {observed} is not the admitted roster {ROSTER_SHA256}')
    roster = json.loads(Path(roster_path).read_text())
    if panel not in roster['panels']:
        raise ValueError(f'panel {panel} is not in the admitted roster')
    declared = roster['panels'][panel]
    if arm not in declared['arms']:
        raise ValueError(f'{arm} is not an arm of panel {panel}')
    if seed not in declared['seeds']:
        raise ValueError(f'seed {seed} is not a declared split seed of panel {panel}')
    manifests = {e['arm']: e['manifest'] for e in roster['extractions']}
    if arm not in manifests:
        raise ValueError(f'{arm} has no extraction manifest in the roster')
    matching = [a for a in roster['analyses']
                if a['arm'] == arm and a['panel'] == panel and a['seed'] == seed]
    if len(matching) != 1:
        raise ValueError(f'expected exactly one admitted analysis for {arm}/{panel}/{seed}')
    return (Path(roster['cohort']['path']), Path(manifests[arm]), list(declared['assay_ids']),
            Path(matching[0]['report']), roster['cohort']['sha256'],
            roster['panel_counts'][panel])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roster', required=True, type=Path,
                        help='Frozen readout expansion roster, bound by SHA-256')
    parser.add_argument('--arm', required=True)
    parser.add_argument('--panel', required=True)
    parser.add_argument('--fold-seed', required=True, type=int)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=64)
    parser.add_argument('--bootstrap', type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument('--seed', type=int, default=PROJECTION_SEED,
                        help='Fixed projection, bootstrap and permutation seed')
    parser.add_argument('--classes', default='all',
                        help='Comma-separated declared class names, or all')
    parser.add_argument('--depth-resolution', action='store_true',
                        help='Also fit the four depth-and-pooling blocks separately')
    parser.add_argument('--shuffle-control', action='store_true',
                        help='Also fit the fixed within-assay label permutation per class')
    args = parser.parse_args()
    if args.device != 'cpu':
        raise ValueError('this sweep is a fitting workload and runs on CPU only')
    if args.threads < 1:
        raise ValueError('positive thread count required')
    torch.set_num_threads(args.threads)

    specs = dict(CLASSES) if args.classes == 'all' else {
        name: CLASSES[name] for name in args.classes.split(',')}
    if REPRODUCTION_CLASS not in specs:
        raise ValueError(f'{REPRODUCTION_CLASS} is required: it binds the cell to its baseline')
    if args.depth_resolution:
        specs.update(DEPTH_CLASSES)
    shuffle = tuple(specs) if args.shuffle_control else ()

    cohort_path, manifest_path, assay_ids, baseline_path, cohort_sha, counts = resolve_cell(
        args.roster, args.arm, args.panel, args.fold_seed)
    began = time.monotonic()
    rows, provenance = load_panel(cohort_path, manifest_path, args.arm, assay_ids)
    if provenance['source_sha256'][str(cohort_path)] != cohort_sha:
        raise ValueError('cohort digest differs from the roster')
    for key, expected in (('n_assays', 'assays'), ('n_clusters', 'clusters'), ('n_variants', 'variants')):
        if provenance[key] != counts[expected]:
            raise ValueError(f'{key} {provenance[key]} differs from the roster count {counts[expected]}')
    print(json.dumps(dict(arm=args.arm, panel=args.panel, seed=args.fold_seed,
                          assays=provenance['n_assays'], clusters=provenance['n_clusters'],
                          variants=provenance['n_variants'], width=provenance['hidden_width'],
                          classes=sorted(specs), threads=args.threads)), flush=True)
    report = evaluate_class_sweep(
        rows, class_specs=specs, fold_seed=args.fold_seed, device=args.device,
        bootstrap=args.bootstrap, bootstrap_seed=args.seed, shuffle_classes=shuffle,
        progress=lambda name: print(json.dumps(dict(arm=args.arm, panel=args.panel,
                                                    seed=args.fold_seed, fitting=name,
                                                    elapsed_seconds=round(time.monotonic() - began, 1))),
                                    flush=True))
    admitted = json.loads(baseline_path.read_text())
    report['baseline_agreement'] = verify_against_admitted(admitted, provenance, report)
    report['baseline_agreement']['admitted_report'] = dict(path=str(baseline_path),
                                                           sha256=digest(baseline_path))
    report.update(schema_version='d1_readout_class_sweep_v1', status='complete', arm=args.arm,
                  panel=args.panel, stratum=next(e['stratum'] for e in
                                                 json.loads(Path(args.roster).read_text())['extractions']
                                                 if e['arm'] == args.arm),
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  roster=dict(path=str(args.roster), sha256=ROSTER_SHA256),
                  support=dict(panel=args.panel, assay_ids=provenance['assay_ids'],
                               n_assays=provenance['n_assays'], n_clusters=provenance['n_clusters'],
                               n_variants=provenance['n_variants']),
                  extraction=dict(manifest=str(manifest_path),
                                  identity=provenance['extraction_identity'],
                                  block_indices=provenance['extraction_block_indices'],
                                  hidden_width=provenance['hidden_width']),
                  source_sha256=provenance['source_sha256'],
                  depth_resolution=bool(args.depth_resolution),
                  shuffle_control=bool(args.shuffle_control),
                  code_sha256={name: digest(ROOT / name) for name in CODE_FILES},
                  runtime=dict(torch=torch.__version__, numpy=np.__version__,
                               device=args.device, threads=args.threads,
                               projection_blas_threads=provenance['projection_blas_threads']),
                  elapsed_seconds=round(time.monotonic() - began, 1))
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / 'readout_class_sweep.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(destination)
    headline = {name: report['summaries'][f'{name}/delta_spearman']['point']
                for name in specs if f'{name}/delta_spearman' in report['summaries']}
    print(json.dumps(dict(arm=args.arm, panel=args.panel, seed=args.fold_seed,
                          delta_spearman=headline,
                          baseline_max_deviation=report['baseline_agreement']['max_absolute_deviation'],
                          elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
