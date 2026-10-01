"""Binding, gate and resampling-unit invariants of the readout panel summary.

The summary tool is the only place where an arm's observed repeat-check maxima
reach a report, so its refusals matter more than its tables: an unadmitted
receipt, a rendering it does not bind, a gate other than the declared 0.001, an
assay above that gate, a nonzero drift at batch size one, or two resampling units
inside one panel must all stop the run rather than produce a plausible table.
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/capability/readouts/summarise_readout_panel.py'
_spec = importlib.util.spec_from_file_location('summarise_readout_panel', SCRIPT)
summarise = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarise)

CSV_FIELDS = ('arm', 'stratum', 'panel', 'seed', 'assays', 'clusters', 'variants', 'metric',
              'point', 'lower', 'upper', 'point_sign', 'interval_state', 'excluded_assays')
ARMS = {'batched-arm': dict(stratum='native_sequence', batch_size=8, drift=(4e-4, 5e-4)),
        'singleton-arm': dict(stratum='literal_text_AA', batch_size=1, drift=(0.0, 0.0))}


def summary_interval(point, lower, upper, units):
    return dict(point=point, resamples=2000, alpha=0.05, unit='wild-type family at 50% identity',
                n_units=units, minimum_units=8, interval=[lower, upper], excluded_assays=0)


class PanelSummaryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.units = 12

    def build(self, arms=None, gate=summarise.DECLARED_GATE, report_units=None, bootstrap_seed=20260923,
              status='admitted', bind_admission=True):
        """Write one internally consistent fixture, overriding exactly one field per test."""
        arms = arms or ARMS
        admission = dict(status=status, expected_sha256='e' * 64, receipts={}, interfaces={})
        for arm, spec in arms.items():
            score, feature = spec['drift']
            admission['receipts'][arm] = dict(artifacts={
                f'assay_{i}': dict(sha256='a' * 64, score_delta_nats=score if i else 0.0,
                                   mutation_relative_l2=feature if i else 0.0, feature_width=768)
                for i in range(2)})
            admission['interfaces'][arm] = dict(stratum=spec['stratum'], identity=dict(
                batch_size=spec['batch_size'], dtype='float32', max_score_drift=gate,
                max_feature_drift=gate))
        admission_path = self.root / 'final_admission.json'
        admission_path.write_text(json.dumps(admission, indent=2))

        rows, report_files = [], {}
        for index, arm in enumerate(arms):
            units = (report_units or {}).get(arm, self.units)
            summaries = {'R_minus_raw_M_spearman': summary_interval(0.1, 0.05, 0.15, units),
                         'delta_spearman': summary_interval(-0.02, -0.03, -0.01, units),
                         'delta_rank_mse': summary_interval(0.01, -0.02, 0.04, units)}
            report = dict(bootstrap_seed=bootstrap_seed, summaries=summaries)
            path = self.root / f'report_{index}.json'
            path.write_text(json.dumps(report, indent=2))
            report_files[f'{arm}/anchor/20260923'] = dict(
                path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for metric, summary in summaries.items():
                rows.append(dict(arm=arm, stratum=arms[arm]['stratum'], panel='anchor', seed='20260923',
                                 assays='20', clusters=str(units), variants='400', metric=metric,
                                 point=repr(summary['point']), lower=repr(summary['interval'][0]),
                                 upper=repr(summary['interval'][1]), point_sign='positive',
                                 interval_state='positive' if summary['interval'][0] > 0
                                 else 'negative' if summary['interval'][1] < 0 else 'crosses/touches zero',
                                 excluded_assays='0'))
        estimates = self.root / 'readout_estimates.csv'
        estimates.write_text(','.join(CSV_FIELDS) + '\n'
                             + '\n'.join(','.join(row[f] for f in CSV_FIELDS) for row in rows) + '\n')
        provenance = self.root / 'render_provenance.json'
        provenance.write_text(json.dumps(dict(
            admission_sha256=hashlib.sha256(admission_path.read_bytes()).hexdigest() if bind_admission
            else 'f' * 64,
            expected_sha256='e' * 64, original_test=False,
            artifacts={estimates.name: hashlib.sha256(estimates.read_bytes()).hexdigest()},
            report_files=report_files), indent=2))
        return admission_path, estimates, provenance

    def run_tool(self, paths, out=None):
        admission, estimates, provenance = paths
        out = out or self.root / 'out'
        completed = subprocess.run([sys.executable, str(SCRIPT), '--admission', str(admission),
                                    '--estimates', str(estimates), '--provenance', str(provenance),
                                    '--out', str(out)], capture_output=True, text=True)
        return completed, out

    def test_reports_observed_maxima_and_marks_singleton_checks_uninformative(self):
        completed, out = self.run_tool(self.build())
        self.assertEqual(completed.returncode, 0, completed.stderr)
        precision = {row.split('\t')[0]: row.split('\t')
                     for row in (out / 'extraction_precision.tsv').read_text().splitlines()[1:]}
        header = (out / 'extraction_precision.tsv').read_text().splitlines()[0].split('\t')
        informative = header.index('repeat_check_informative')
        score = header.index('observed_max_score_drift_nats')
        self.assertEqual(precision['batched-arm'][informative], 'true')
        self.assertEqual(precision['singleton-arm'][informative], 'false')
        # Full precision is retained, so a reader cannot mistake 4e-4 for the gate.
        self.assertEqual(float(precision['batched-arm'][score]), 4e-4)
        self.assertEqual(float(precision['singleton-arm'][score]), 0.0)
        summary = json.loads((out / 'panel_summary.json').read_text())
        self.assertEqual(summary['repeat_check_uninformative_arms'], ['singleton-arm'])
        self.assertEqual(summary['panels']['anchor'],
                         dict(unit='wild-type family at 50% identity', units=12, minimum_units=8))
        self.assertEqual(summary['interval_direction_counts']['native_sequence/anchor/20260923/delta_spearman'],
                         {'negative': 1})
        self.assertEqual(len((out / 'panel_increments.tsv').read_text().splitlines()), 7)

    def test_refuses_receipt_that_has_not_passed_final_admission(self):
        for status in ('extraction_admitted', 'rejected'):
            completed, _ = self.run_tool(self.build(status=status))
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('Final admission is required', completed.stderr)

    def test_refuses_rendering_it_does_not_bind(self):
        completed, _ = self.run_tool(self.build(bind_admission=False))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('does not bind this admission receipt', completed.stderr)

    def test_refuses_undeclared_gate_and_an_assay_above_the_gate(self):
        completed, _ = self.run_tool(self.build(gate=0.01))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('declares a gate other than', completed.stderr)
        over = {'batched-arm': dict(ARMS['batched-arm'], drift=(0.0011, 5e-4))}
        completed, _ = self.run_tool(self.build(arms=over))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('above its declared gate', completed.stderr)

    def test_refuses_nonzero_repeat_drift_at_batch_size_one(self):
        impossible = {'singleton-arm': dict(ARMS['singleton-arm'], drift=(1e-6, 0.0))}
        completed, _ = self.run_tool(self.build(arms=impossible))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('batch size one yet reports nonzero repeat drift', completed.stderr)

    def test_refuses_mixed_resampling_units_and_undeclared_bootstrap_seed(self):
        completed, _ = self.run_tool(self.build(report_units={'singleton-arm': 9}))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('mixes resampling units', completed.stderr)
        completed, _ = self.run_tool(self.build(bootstrap_seed=1))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('undeclared bootstrap seed', completed.stderr)


if __name__ == '__main__':
    unittest.main()
