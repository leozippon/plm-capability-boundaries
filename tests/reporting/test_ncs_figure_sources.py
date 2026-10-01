"""Prevent repaired uncertainty and changed supports being silently mixed in artwork."""
import csv
import hashlib
import json

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / 'scripts/capability/reporting/figure_data/ncs'


def rows(name):
    if not (FIGURES / name).is_file():
        pytest.skip('local manuscript source tables are not distributed with the code repository')
    with (FIGURES / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def test_full_residual_panel_keeps_simultaneous_family_and_baseline_provenance():
    panel = rows('residual-full-panel-source.csv')
    assert len(panel) == len({r['arm'] for r in panel}) == 33
    for row in panel:
        assert (int(row['groups']), int(row['cycles']), int(row['site_pairs'])) == (64, 8192, 217)
        assert int(row['simultaneous_family']) == 33
        assert row['source_path'].startswith('results/R3/pairwise_residual_full_panel_20260927/')
        baseline = ROOT / row['baseline_source_path']
        if not baseline.is_file():
            pytest.skip('original experiment receipts are not distributed with the manuscript')
        assert hashlib.sha256(baseline.read_bytes()).hexdigest() == row['baseline_source_sha256']


def test_stability_panel_uses_matched_simultaneous_inference_and_exact_support():
    table = rows('stability-panel-source.csv')
    assert len(table) == len({r['arm'] for r in table}) == 33
    assert {(int(r['groups']), int(r['variants']), int(r['sites'])) for r in table} == {(101, 25856, 5664)}
    assert {(int(r['simultaneous_family']), int(r['bootstrap_resamples'])) for r in table} == {(33, 10000)}
    assert {r['arm'] for r in table if float(r['ci_low']) > 0} == {'progen3-3b', 'prollama'}
    for row in table:
        baseline = ROOT / row['baseline_source_path']
        if not baseline.is_file():
            pytest.skip('original experiment receipts are not distributed with the manuscript')
        assert hashlib.sha256(baseline.read_bytes()).hexdigest() == row['baseline_source_sha256']


def test_stability_display_uses_exact_model_support_not_broad_qualification():
    point = next(r for r in rows('endpoint-boundaries-source.csv') if r['endpoint'] == 'single_substitution_stability')
    assert point['source_path'].endswith('exact_stability_qualification.json')
    selected = [r for r in rows('stability-support-source.csv') if r['population'] == 'exact_model_cohort']
    assert {(int(r['variants']), int(r['sites']), int(r['backgrounds']), int(r['groups'])) for r in selected} == {(25856, 5664, 101, 101)}


def test_nested_sensitivity_is_nine_selected_cells_with_current_receipts():
    repaired = [r for r in rows('nested-residual-source.csv') if r['target'] == 'nested_adjusted']
    assert len(repaired) == 9
    assert {r['arm'] for r in repaired} == {'protgpt2', 'progen3-3b', 'prollama'}
    assert len({(r['arm'], r['split_seed']) for r in repaired}) == 9
    for row in repaired:
        assert float(row['ci_low']) <= 0 <= float(row['ci_high'])
        assert row['source_path'].startswith('results/R3/pairwise_residual_nested_20260927/')
        source = ROOT / row['source_path']
        if not source.is_file():
            pytest.skip('original experiment receipts are not distributed with the manuscript')
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row['source_sha256']


def test_contact_artwork_uses_repaired_receipt_without_moving_point_estimates():
    table = rows('contact-sensitivity-source.csv')
    historical = {(r['definition'], r['endpoint'], r['weighting']): r for r in table if r['version'] == 'historical'}
    corrected = [r for r in table if r['version'] == 'corrected']
    # Thirty pooled-definition cells are drawn/retained here; fourteen method-stratum
    # cells remain in the original receipt, making 44 primary-support cells in total.
    assert len(corrected) == 30
    for row in corrected:
        old = historical[(row['definition'], row['endpoint'], row['weighting'])]
        assert row['estimate'] == old['estimate']
        assert float(row['ci_low']) <= 0 <= float(row['ci_high'])
        assert row['source_path'] == 'results/R3/contact_bootstrap_corrected_20260927/epsilon_enrichment.json'
        source = ROOT / row['source_path']
        if not source.is_file():
            pytest.skip('original experiment receipts are not distributed with the manuscript')
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row['source_sha256']
        if row['weighting'] == 'site_pair_equal':
            assert (row['ci_low'], row['ci_high']) == (old['ci_low'], old['ci_high'])


def test_context_comparators_use_identical_biological_support():
    table = rows('context-ranking-source.csv')
    assert len(table) == 8
    assert {(r['n_assays'], r['n_units']) for r in table} == {('164', '134')}
    endpoints = ('delta_spearman', 'homolog_minus_no_context_spearman')
    assert {r['arm'] for r in table if r['endpoint'] == endpoints[0]} == {
        r['arm'] for r in table if r['endpoint'] == endpoints[1]}
    assert len({r['source_sha256'] for r in table}) == 1


def test_all_44_primary_contact_intervals_include_zero():
    receipt = ROOT / 'results/R3/contact_bootstrap_corrected_20260927/epsilon_enrichment.json'
    if not receipt.is_file():
        pytest.skip('original corrected contact receipt is not installed')
    data = json.loads(receipt.read_text())

    def contrasts(value):
        if isinstance(value, dict):
            if 'difference' in value and 'interval' in value:
                yield value
            for child in value.values():
                yield from contrasts(child)
        elif isinstance(value, list):
            for child in value:
                yield from contrasts(child)

    estimates = list(contrasts(data['supports'][data['primary_support']]))
    assert len(estimates) == 44
    assert all(v['interval'][0] <= 0 <= v['interval'][1] for v in estimates)
