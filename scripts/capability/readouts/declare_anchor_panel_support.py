#!/usr/bin/env python3
"""Recover the frozen anchor panel's assay support from the archives themselves.

The published readout panel names its primary support as the 201 assays, 163
wild-type identity clusters and 25,728 variants covered by all 33 unconditioned
arms. That support is a property of the archived extractions rather than a list
anyone wrote down: an arm's own eligible set is whatever fits the 1024-position
budget under its own packing, so the panel is the intersection over the roster.

This entry point rebuilds it that way and refuses to publish a support whose
counts differ from the declared ones, so a later analysis cannot quietly read a
different panel. It also records each arm's own native count, because an arm
whose native support is wider than the panel is scored on more assays than the
panel reads and the difference is a fact about the arm, not about the panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.capability.core.io import write_json
from src.capability.interactions.pairwise_epistasis import ROSTER

SCHEMA = 'anchor_panel_support_v1'
DECLARED = {'assays': 201, 'clusters': 163, 'variants': 25728, 'arms': 33}


def production_manifests(roots: list[Path]) -> dict[str, dict]:
    """The widest complete non-smoke readout manifest of each roster arm."""

    best: dict[str, dict] = {}
    for root in roots:
        for path in Path(root).rglob('manifest_*.json'):
            try:
                payload = json.loads(path.read_bytes())
            except (ValueError, OSError):
                continue
            identity = payload.get('identity', {})
            if identity.get('schema_version') != 'frozen_readout_v1':
                continue
            if payload.get('status') != 'complete' or identity.get('smoke_variants'):
                continue
            if identity.get('assay_limit'):
                continue
            arm = identity.get('arm')
            if arm not in set(ROSTER):
                continue
            row = {'path': str(path), 'assays': {r['assay'] for r in payload['assays']},
                   'cohort_sha256': identity['cohort_sha256'],
                   'batch_size': identity['batch_size'], 'dtype': identity['dtype']}
            prior = best.get(arm)
            if prior is None or len(row['assays']) > len(prior['assays']):
                best[arm] = row
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', required=True, type=Path, action='append')
    parser.add_argument('--cohort', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()

    cohort_bytes = args.cohort.read_bytes()
    cohort = json.loads(cohort_bytes)
    rows = production_manifests(list(args.archive_root))
    if sorted(rows) != sorted(ROSTER):
        missing = sorted(set(ROSTER) - set(rows))
        raise SystemExit(f'the archive does not cover the roster; missing {missing}')
    digests = {row['cohort_sha256'] for row in rows.values()}
    if digests != {hashlib.sha256(cohort_bytes).hexdigest()}:
        raise SystemExit('archived extractions disagree with the supplied cohort digest')
    support = set.intersection(*(row['assays'] for row in rows.values()))
    by_assay = {row['assay']: row for row in cohort['assays']}
    if not support <= set(by_assay):
        raise SystemExit('the intersected support contains assays absent from the cohort')
    measured = {'assays': len(support),
                'clusters': len({by_assay[a]['cluster'] for a in support}),
                'variants': sum(len(by_assay[a]['mutants']) for a in support),
                'arms': len(rows)}
    if measured != DECLARED:
        raise SystemExit(f'recovered support {measured} differs from the declared {DECLARED}; '
                         'a different panel is not substituted for the frozen one')
    single = sum(1 for a in support for m in by_assay[a]['mutants'] if ':' not in m)
    record = {'schema': SCHEMA, 'declared': DECLARED, 'measured': measured,
              'cohort_sha256': hashlib.sha256(cohort_bytes).hexdigest(),
              'assays': sorted(support),
              'single_substitution_variants': single,
              'multi_substitution_variants': measured['variants'] - single,
              'native_assay_counts': {arm: len(row['assays']) for arm, row in sorted(rows.items())},
              'archived_batch_size': {arm: row['batch_size'] for arm, row in sorted(rows.items())},
              'archived_dtype': {arm: row['dtype'] for arm, row in sorted(rows.items())},
              'archived_manifests': {arm: row['path'] for arm, row in sorted(rows.items())},
              'created_utc': datetime.now(timezone.utc).isoformat()}
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'anchor_panel_support.json', record)
    print(json.dumps({'measured': measured, 'single': single,
                      'multi': measured['variants'] - single}), flush=True)


if __name__ == '__main__':
    main()
