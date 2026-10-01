#!/usr/bin/env python3
"""Measurement-only qualification of the single-mutant stability endpoint.

Builds ddG = y_mut - y_WT separately from the trypsin and the chymotrypsin
channel of the pinned Tsuboyama 2023 dataset2 bytes on one identical accepted
support, then reports between-channel agreement, the measurement-noise floor in
kcal/mol, the support geometry and the censoring and range-restriction
accounting. The endpoint is a two-estimate difference and carries its own
qualification: nothing in the four-state cycle endpoint's admission transfers.

No model score, likelihood or representation enters any quantity here, so the
admission decision cannot be tuned to a model outcome. The two proteases are two
assay channels over shared sequences, a shared library and a shared stability
inference, not independent ground truth; no oracle ceiling is constructed by
adding independent error variances.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capability.core.io import sha256_file, write_json
from src.capability.stability.stability_gate import (
    ENDPOINT, QC_SOURCE, QC_WIDTH_KCAL_MOL, accept_rows, aggregate_states,
    build_endpoint, endpoint_digest, load_source_frame)


def verify_pinned_bytes(data_dir: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_bytes())
    expected = {Path(f['path']).name: f['sha256'] for f in manifest['verification']['files']}
    measured = {p.name: sha256_file(p) for p in sorted(data_dir.glob('*.parquet'))}
    if set(measured) != set(expected):
        raise ValueError(f'staged parquet set {sorted(measured)} differs from manifest')
    wrong = {n: measured[n] for n in measured if measured[n] != expected[n]}
    if wrong:
        raise ValueError(f'staged bytes differ from the pinned manifest digests: {wrong}')
    return {'corpus_id': manifest['corpus_id'], 'config': manifest['config'],
            'revision': manifest['canonical_source']['revision'], 'files': measured}


def convention_check(frame: pd.DataFrame, mask: np.ndarray, endpoint: pd.DataFrame) -> dict:
    """Read the sign convention off the source's own ddG_ML column."""

    reported = pd.to_numeric(frame.loc[mask, 'ddG_ML'], errors='coerce')
    rows = frame.loc[mask, ['WT_name', 'aa_seq']].copy()
    rows['reported'] = reported.to_numpy()
    keyed = endpoint.set_index(['WT_name', 'sequence']).combined
    rows['ours'] = [keyed.get((n, s), np.nan) for n, s in
                    zip(rows.WT_name.to_numpy(), rows.aa_seq.to_numpy())]
    both = rows[np.isfinite(rows.reported) & np.isfinite(rows.ours)]
    return {
        'rows_compared': int(len(both)),
        'rms_residual_same_sign_kcal_mol': float(np.sqrt(np.mean((both.ours - both.reported) ** 2))),
        'rms_residual_opposite_sign_kcal_mol': float(np.sqrt(np.mean((both.ours + both.reported) ** 2))),
        'mean_reported_ddG_ML_kcal_mol': float(both.reported.mean()),
        'convention': ('dG_ML is an unfolding free energy: larger is more stable, so a '
                       'destabilising substitution gives ddG < 0'),
    }


def _moments(t: np.ndarray, c: np.ndarray) -> np.ndarray:
    d = t - c
    return np.array([t.sum(), c.sum(), (t * t).sum(), (c * c).sum(), (t * c).sum(),
                     (d * d).sum(), np.abs(d).sum()]) / t.size


def _derive(v: np.ndarray) -> dict:
    et, ec, ett, ecc, etc, ed2, eabsd = v
    var_t = max(float(ett - et * et), 0.0)
    var_c = max(float(ecc - ec * ec), 0.0)
    cov = float(etc - et * ec)
    bias = float(et - ec)
    var_d = max(float(ed2 - bias * bias), 0.0)
    shared = float(np.sqrt(max(cov, 0.0)))
    discordance = float(np.sqrt(var_d / 2.0))
    return {
        'mean_ddg_trypsin_kcal_mol': float(et),
        'mean_ddg_chymotrypsin_kcal_mol': float(ec),
        'sd_ddg_trypsin_kcal_mol': float(np.sqrt(var_t)),
        'sd_ddg_chymotrypsin_kcal_mol': float(np.sqrt(var_c)),
        'pearson_r': float(cov / np.sqrt(var_t * var_c)) if var_t > 0 and var_c > 0 else None,
        'shared_component_sd_kcal_mol': shared,
        'channel_difference_mean_kcal_mol': bias,
        'channel_difference_rms_kcal_mol': float(np.sqrt(ed2)),
        'per_channel_discordance_sd_kcal_mol': discordance,
        'shared_to_discordance_ratio': float(shared / discordance) if discordance > 0 else None,
    }


def _unit_vectors(endpoint: pd.DataFrame, unit: str) -> tuple[list[np.ndarray], list[float], dict]:
    """Nested moment vectors: variants equal inside a site, sites equal inside a
    background, backgrounds equal inside the resampling unit.

    Every variant of a background is differenced against that background's single
    wild-type measurement, so resampling whole backgrounds or whole groups keeps
    that shared measurement's signed contribution to all of its variants intact.
    Variant-level resampling would break exactly that reuse.
    """

    per_background, sizes, ranks = {}, {}, {}
    for name, part in endpoint.groupby('WT_name', sort=True):
        site_vectors = [
            _moments(site.trypsin.to_numpy(float), site.chymotrypsin.to_numpy(float))
            for _, site in part.groupby('position', sort=True)]
        per_background[name] = np.mean(site_vectors, axis=0)
        sizes[name] = len(part)
        t, c = part.trypsin.to_numpy(float), part.chymotrypsin.to_numpy(float)
        if len(part) >= 20 and t.std() > 0 and c.std() > 0:
            ranks[name] = float(np.corrcoef(rankdata(t), rankdata(c))[0, 1])
    key_of = endpoint.groupby('WT_name', sort=True)[unit].first().to_dict()
    members: dict[str, list[str]] = {}
    for name in sorted(per_background):
        members.setdefault(str(key_of[name]), []).append(name)
    keys = sorted(members)
    vectors = [np.mean([per_background[n] for n in members[k]], axis=0) for k in keys]
    counts = np.array([sum(sizes[n] for n in members[k]) for k in keys], float)
    rank_values = [float(np.mean([ranks[n] for n in members[k] if n in ranks]))
                   for k in keys if any(n in ranks for n in members[k])]
    support = {'unit': unit, 'units': len(keys),
               'effective_units_kish': float(counts.sum() ** 2 / (counts ** 2).sum()),
               'units_with_within_background_spearman': len(rank_values)}
    return vectors, rank_values, support


def _bootstrap(vectors, ranks, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    stack = np.array(vectors)
    index = rng.integers(0, len(stack), size=(draws, len(stack)))
    samples = [_derive(stack[row].mean(axis=0)) for row in index]
    point = _derive(stack.mean(axis=0))
    report = {}
    for key in point:
        column = np.array([s[key] for s in samples if s[key] is not None], float)
        column = column[np.isfinite(column)]
        report[key] = {'point': point[key],
                       'ci95': [float(np.percentile(column, 2.5)),
                                float(np.percentile(column, 97.5))] if column.size else None}
    if ranks:
        array = np.array(ranks, float)
        resampled = array[rng.integers(0, array.size, size=(draws, array.size))].mean(axis=1)
        report['within_background_spearman'] = {
            'point': float(array.mean()),
            'ci95': [float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))]}
    return report


def summarise(endpoint: pd.DataFrame, *, draws: int, seed: int) -> dict:
    out = {}
    for label, part in (('all', endpoint),
                        ('natural', endpoint[endpoint.kind == 'natural'])):
        entry = {
            'variants': int(len(part)),
            'backgrounds': int(part.WT_name.nunique()),
            'family_groups': int(part.group.nunique()),
            'source_clusters': int(part.cluster.nunique()),
            'sites': int(len(part.groupby(['WT_name', 'position']))),
            'sd_combined_ddg_kcal_mol': float(part.combined.std()),
            'mean_combined_ddg_kcal_mol': float(part.combined.mean()),
        }
        for unit in ('group', 'WT_name'):
            vectors, ranks, support = _unit_vectors(part, unit)
            entry['background' if unit == 'WT_name' else 'family_group'] = {
                **support, 'agreement': _bootstrap(vectors, ranks, draws, seed)}
        out[label] = entry
    return out


def replicate_sensitivity(frame: pd.DataFrame, mask: np.ndarray, catalogue: dict,
                          *, draws: int, seed: int) -> dict:
    """Single-row-in-stable-source-order sensitivity against median aggregation."""

    values = aggregate_states(frame, mask, statistic='first')
    endpoint, _ = build_endpoint(values, catalogue)
    natural = endpoint[endpoint.kind == 'natural']
    vectors, ranks, support = _unit_vectors(natural, 'group')
    return {'aggregation': 'first row in stable source order', **support,
            'agreement': _bootstrap(vectors, ranks, draws, seed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=ROOT / 'data/megascale_tsuboyama2023/dataset2/data')
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'data/megascale_tsuboyama2023/manifest.json')
    parser.add_argument('--catalogue', type=Path,
                        default=ROOT / 'results/shared/megascale_disjointness_20260813/query_index.json')
    parser.add_argument('--groups', type=Path,
                        default=ROOT / 'data/pairwise_assets/megascale_family_groups_20260924.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260923)
    args = parser.parse_args()

    pinned = verify_pinned_bytes(args.data_dir, args.manifest)
    frame = load_source_frame(args.data_dir)
    grouping = json.loads(args.groups.read_bytes())
    if grouping.get('status') != 'admitted':
        raise SystemExit('the family grouping map is not admitted')
    assignments = grouping['assignments']

    wt = frame[frame.mut_type == 'wt'].groupby('WT_name').aa_seq.agg(['nunique', 'first'])
    if int((wt['nunique'] > 1).sum()):
        raise SystemExit('a background carries more than one distinct wild-type sequence')
    clusters = frame.groupby('WT_name').WT_cluster.agg(['nunique', 'first'])
    catalogue = {}
    for row in json.loads(args.catalogue.read_bytes()):
        name = row['WT_name']
        if name not in wt.index or name not in assignments:
            continue
        if row['sequence'] != wt.loc[name, 'first']:
            raise SystemExit(f'catalogue sequence for {name} differs from its wild-type row')
        catalogue[name] = {'kind': row['kind'], 'sequence': row['sequence'],
                           'cluster': str(clusters.loc[name, 'first']),
                           'group': assignments[name]}

    mask, accounting = accept_rows(frame)
    values = aggregate_states(frame, mask, statistic='median')
    endpoint, support = build_endpoint(values, catalogue)
    natural = endpoint[endpoint.kind == 'natural']
    records = [{'background': r.WT_name, 'sequence': r.sequence, 'position': int(r.position),
                'ddg': float(r.combined)} for r in natural.itertuples()]

    report = {
        'schema': 'stability_single_endpoint_qualification_v1',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'endpoint': ENDPOINT,
        'endpoint_sha256': endpoint_digest(records),
        'qc_source': QC_SOURCE,
        'qc_width_kcal_mol': QC_WIDTH_KCAL_MOL,
        'admission_rule': ('finite numeric dG_ML; each of the combined, trypsin and '
                           'chymotrypsin 95% confidence widths in [0, 0.5] kcal/mol; '
                           'substitution constructs only'),
        'aggregation': 'median over accepted rows per exact (WT_name, aa_seq), identical rows per channel',
        'pinned_input': pinned,
        'row_accounting': accounting,
        'support_accounting': support,
        'convention_check': convention_check(frame, mask, endpoint),
        'agreement': summarise(endpoint, draws=args.bootstrap, seed=args.seed),
        'single_row_sensitivity': replicate_sensitivity(
            frame, mask, catalogue, draws=args.bootstrap, seed=args.seed),
        'range_restriction': (
            'accepted rows carry dG_ML inside the assay range and both channel widths at '
            'most 0.5 kcal/mol, so variants whose mutant state falls below the stability '
            'floor or is otherwise unresolved are absent; the admitted support represents '
            'resolvable stability states and is selected against strongly destabilised mutants'),
        'bootstrap': {'draws': args.bootstrap, 'seed': args.seed,
                      'unit': 'family group, and background as a secondary unit'},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'endpoint_qualification.json', report)
    natural.to_parquet(args.out / 'endpoint_natural.parquet', index=False)
    print(json.dumps({k: report[k] for k in
                      ('endpoint', 'endpoint_sha256', 'row_accounting', 'support_accounting',
                       'convention_check')}, indent=1)[:4000])
    print(json.dumps(report['agreement']['natural']['family_group'], indent=1))


if __name__ == '__main__':
    main()
