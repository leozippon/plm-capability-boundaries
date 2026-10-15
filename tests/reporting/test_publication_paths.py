"""Submission filenames never change the historical figure/source identifiers."""
import importlib.util
from pathlib import Path

import pytest

from scripts.capability.reporting import build_derived_numbers as builder
from scripts.capability.reporting import publication_paths as paths
from src.capability.core.evidence_ledger import Artifacts, digest

ROOT = Path(__file__).resolve().parents[2]


def test_pdf_names_and_current_asset_locations():
    assert len(paths.PDF_NAMES) == len(set(paths.PDF_NAMES.values())) == 10
    assert paths.figure_pdf('Fig1_framework') == Path('manuscript/submission/figures/figure-1-framework.pdf')
    assert paths.figure_pdf('matched-generators').name == 'supplementary-figure-4-matched-generators.pdf'
    assert paths.FRAMEWORK_SOURCE == Path('manuscript/assets/framework-original.png')
    assert paths.STRUCTURE_SELECTION == Path('manuscript/assets/selection.json')
    with pytest.raises(KeyError):
        paths.figure_pdf('unknown-figure')


def test_pdf_contract_keeps_main_and_supplement_table_identifiers(tmp_path, monkeypatch):
    docs = tmp_path / paths.SUBMISSION
    docs.mkdir(parents=True)
    (docs / 'main.tex').write_text(r'\includegraphics{figures/figure-2-mutation.pdf}')
    (docs / 'supplementary-information.tex').write_text(
        r'\includegraphics{figures/supplementary-figure-1-native-comparators.pdf}'
        r'\includegraphics{figures/unknown.pdf}')
    source = tmp_path / 'retained.json'
    source.write_text('{}')
    tables = tmp_path / 'scripts/capability/reporting/figure_data'
    for directory, identifier in [('main', 'Fig2_mutation'), ('ncs', 'native-comparators')]:
        folder = tables / directory
        folder.mkdir(parents=True)
        (folder / f'{identifier}-plotted.csv').write_text(
            f'source_path,source_sha256\nretained.json,{digest(source)}\n')
    monkeypatch.setattr(builder, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(builder, 'FIGURE_DIR', tables)
    contract = builder.figure_data_contract(Artifacts(tmp_path), {'counts': {'quantities': 0}})
    figures = contract['figures']
    assert isinstance(figures, dict)
    main = figures['figures/figure-2-mutation.pdf']
    supplement = figures['figures/supplementary-figure-1-native-comparators.pdf']
    assert main['bound'] and supplement['bound']
    assert main['input_tables'][0].endswith('/main/Fig2_mutation-plotted.csv')
    assert supplement['input_tables'][0].endswith('/ncs/native-comparators-plotted.csv')
    assert contract['figures_without_a_complete_binding'] == ['figures/unknown.pdf']


def test_packaged_stage_restores_retained_json_names_and_shared_helper(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('packaged_figures', ROOT / 'manuscript/code/run_figures.py')
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    package = tmp_path / 'package'
    code, data, assets = [package / name for name in ('code', 'data', 'assets')]
    for directory in (code, data, assets):
        directory.mkdir(parents=True)
    for name in runner.SOURCES:
        source = ROOT / 'scripts/capability/reporting' / name
        assert (ROOT / 'manuscript/code' / name).read_bytes() == source.read_bytes()
        (code / name).write_bytes(source.read_bytes())
    for name in ('checkpoint-name-key.csv', 'clean-interaction-source.csv',
                 'figure-2-mutation-plotted.csv', 'position-simultaneous.json',
                 'generation-campaign-summary.json', 'figure-validation.json',
                 'framework-validation.json', 'source-provenance.json'):
        (data / name).write_text(name)
    for name in ('inset-manifest.json', 'dvl1-abundance.png', '1ufm-stability.png', '1ufm-pair.png'):
        (assets / name).write_text(name)
    monkeypatch.setattr(runner, 'CODE', code)
    monkeypatch.setattr(runner, 'PACKAGE', package)
    workspace = tmp_path / 'workspace'
    reporting = runner.stage(workspace)
    assert (reporting / 'publication_paths.py').is_file()
    assert (workspace / paths.PDF_DIRECTORY).is_dir()
    assert not (workspace / 'manuscript/figures').exists()
    main = reporting / 'figure_data/main'
    for original, staged in [('position-simultaneous.json', 'position_simultaneous.json'),
                             ('generation-campaign-summary.json', 'generation_campaign_summary.json'),
                             ('figure-validation.json', 'figure_validation.json'),
                             ('framework-validation.json', 'framework_validation.json')]:
        assert (main / staged).read_text() == original
    assert (main / 'clean-interaction-source.csv').is_file()
