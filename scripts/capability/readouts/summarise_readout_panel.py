#!/usr/bin/env python3
"""Summarise the admitted readout panel by interface stratum.

Reads only artifacts a passed final admission receipt binds, plus the deterministic
rendering that receipt licensed. It performs no fitting, resampling, threshold
selection or model ranking: every number it writes is copied from an admitted
artifact or is a maximum/minimum over admitted per-assay values.

Two quantities the renderer does not expose are required to report the panel
honestly. The first is each arm's observed production repeat-check maximum, in
nats and relative L2, together with whether that check can carry evidence at all:
at batch size one the production scoring forward and the repeat check's reference
forward in scripts/capability/readouts/extract_frozen_readout.py are the same single-row
computation over the same string, so an exactly-zero drift is true by construction
and certifies nothing about numerical accuracy. The second is the resampling unit
behind every interval, read from the admitted reports and required to be identical
within each panel, because intervals on different units are not comparable.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

#: Fixed by the protocol for every production assay of every arm, in nats and in
#: relative L2 of the mutation vector. A roster declaring anything else is not the
#: experiment this tool summarises.
DECLARED_GATE = 0.001
#: Fixed resampling contract: 2,000 paired draws over wild-type clusters at
#: percentile 95%, seeded once for the whole experiment.
DECLARED_RESAMPLES = 2000
DECLARED_ALPHA = 0.05
DECLARED_BOOTSTRAP_SEED = 20260923
#: Endpoints carried into the stratum tables. The primary endpoint is the paired
#: increment of baseline-plus-representation over the matched supervised baseline.
ENDPOINTS = ('R_minus_raw_M_spearman', 'delta_spearman', 'delta_rank_mse')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(test, message):
    if not test:
        raise ValueError(message)


def load_inputs(admission_path, estimates_path, provenance_path):
    """Bind the receipt, the rendering and the reports to one another, or fail."""
    admission = json.loads(Path(admission_path).read_text())
    need(admission.get('status') == 'admitted',
         f'Final admission is required before reading outcomes; status is {admission.get("status")!r}')
    provenance = json.loads(Path(provenance_path).read_text())
    need(provenance.get('admission_sha256') == digest(admission_path),
         'Rendering provenance does not bind this admission receipt')
    need(provenance.get('expected_sha256') == admission['expected_sha256'],
         'Rendering provenance and receipt bind different expected rosters')
    need(provenance.get('original_test') is False,
         'The original five-arm renderer test cannot summarise the expanded panel')
    need(Path(estimates_path).name in provenance.get('artifacts', {})
         and provenance['artifacts'][Path(estimates_path).name] == digest(estimates_path),
         'Estimates file is not the rendering this provenance records')
    rows = list(csv.DictReader(Path(estimates_path).open(newline='')))
    need(rows, 'Estimates file is empty')
    return admission, provenance, rows


def precision_rows(admission):
    """Per-arm observed repeat-check maxima and whether the check can carry evidence."""
    out = []
    for arm in sorted(admission['receipts']):
        identity = admission['interfaces'][arm]['identity']
        artifacts = admission['receipts'][arm]['artifacts']
        need(identity['max_score_drift'] == DECLARED_GATE
             and identity['max_feature_drift'] == DECLARED_GATE,
             f'{arm} declares a gate other than {DECLARED_GATE}')
        score = max(abs(a['score_delta_nats']) for a in artifacts.values())
        feature = max(abs(a['mutation_relative_l2']) for a in artifacts.values())
        need(score <= DECLARED_GATE and feature <= DECLARED_GATE,
             f'{arm} retains an admitted assay above its declared gate')
        widths = {a['feature_width'] for a in artifacts.values()}
        need(len(widths) == 1, f'{arm} hidden width varies across assays')
        batch = identity['batch_size']
        informative = batch > 1
        need(informative or (score == 0.0 and feature == 0.0),
             f'{arm} runs at batch size one yet reports nonzero repeat drift')
        out.append(dict(arm=arm, stratum=admission['interfaces'][arm]['stratum'], batch_size=batch,
                        dtype=identity['dtype'], hidden_width=widths.pop(), assays_checked=len(artifacts),
                        observed_max_score_drift_nats=repr(score),
                        observed_max_feature_relative_l2=repr(feature),
                        repeat_check_informative=str(informative).lower()))
    return out


def panel_units(provenance):
    """Resampling unit, unit count and floor per panel, required constant within a panel."""
    units = {}
    for key, record in sorted(provenance['report_files'].items()):
        arm, panel, seed = key.split('/')
        need(digest(record['path']) == record['sha256'],
             f'Report {record["path"]} changed after rendering')
        report = json.loads(Path(record['path']).read_text())
        need(report['bootstrap_seed'] == DECLARED_BOOTSTRAP_SEED,
             f'{key} uses an undeclared bootstrap seed')
        for name, summary in report['summaries'].items():
            if summary.get('interval') is None:
                continue
            need(summary['resamples'] == DECLARED_RESAMPLES and summary['alpha'] == DECLARED_ALPHA,
                 f'{key}/{name} uses an undeclared resampling contract')
            observed = (summary['unit'], summary['n_units'], summary['minimum_units'])
            need(units.setdefault(panel, observed) == observed,
                 f'{panel} mixes resampling units across reports')
    return units


def increment_rows(rows, units):
    """Long-form endpoint table, one row per arm, panel, seed and endpoint."""
    out = []
    for row in rows:
        if row['metric'] not in ENDPOINTS:
            continue
        unit, n_units, minimum = units[row['panel']]
        need(int(row['clusters']) == n_units,
             f'{row["arm"]}/{row["panel"]} cluster count disagrees with its resampling unit count')
        out.append(dict(stratum=row['stratum'], arm=row['arm'], panel=row['panel'], seed=row['seed'],
                        assays=row['assays'], resampling_unit=unit, resampling_units=n_units,
                        minimum_units=minimum, metric=row['metric'], point=row['point'],
                        lower=row['lower'], upper=row['upper'], interval_state=row['interval_state'],
                        excluded_assays=row['excluded_assays']))
    need(out, 'No declared endpoint rows found in the rendering')
    return out


def tally(increments):
    """Count interval directions per stratum, panel, seed and endpoint. Counts are
    descriptive: checkpoints share lineages, the panel is not a random sample of
    models, and no multiplicity adjustment is applied."""
    counts = defaultdict(lambda: defaultdict(int))
    for row in increments:
        key = '/'.join((row['stratum'], row['panel'], str(row['seed']), row['metric']))
        counts[key][row['interval_state']] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(counts.items())}


def write_tsv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--admission', type=Path, required=True, help='Passed final admission receipt')
    parser.add_argument('--estimates', type=Path, required=True, help='readout_estimates.csv from the renderer')
    parser.add_argument('--provenance', type=Path, required=True, help='render_provenance.json from the renderer')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    admission, provenance, rows = load_inputs(args.admission, args.estimates, args.provenance)
    arms = {row['arm'] for row in rows}
    need(arms == set(admission['receipts']), 'Rendered arms differ from the admitted extraction arms')
    precision = precision_rows(admission)
    units = panel_units(provenance)
    increments = increment_rows(rows, units)
    args.out.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out / 'extraction_precision.tsv', precision)
    write_tsv(args.out / 'panel_increments.tsv', increments)
    summary = dict(admission_sha256=digest(args.admission), expected_sha256=admission['expected_sha256'],
                   arms=len(precision), panels={panel: dict(unit=unit, units=count, minimum_units=minimum)
                                                for panel, (unit, count, minimum) in sorted(units.items())},
                   repeat_check_uninformative_arms=sorted(r['arm'] for r in precision
                                                          if r['repeat_check_informative'] == 'false'),
                   interval_direction_counts=tally(increments))
    (args.out / 'panel_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(arms=len(precision), panels=len(units), endpoint_rows=len(increments),
                          output=str(args.out))))


if __name__ == '__main__':
    main()
