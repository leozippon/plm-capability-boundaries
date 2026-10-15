"""Current render receipts certify real outputs, not stale layouts or inferred type."""
from pathlib import Path
import json
import shutil
import sys
import subprocess

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.capability.reporting import render_figures as renderer
from scripts.capability.reporting import validate_manuscript as validator
from scripts.capability.reporting import publication_paths


@pytest.fixture
def isolated_renderer(tmp_path, monkeypatch):
    data = tmp_path / 'scripts/capability/reporting/figure_data/main'
    data.mkdir(parents=True)
    retained = renderer.DATA
    manifest = json.loads((retained / 'figure_validation.json').read_text())
    expected = {}
    for path, sha in manifest['source_sha256'].items():
        if path.endswith('.csv'):
            shutil.copyfile(ROOT / path, data / Path(path).name)
            expected[str((data / Path(path).name).relative_to(tmp_path))] = sha
    bundle = data.parent / 'structure-insets'
    shutil.copytree(retained.parent / 'structure-insets', bundle)
    manifest_path = bundle / 'manifest.json'
    bundle_manifest = json.loads(manifest_path.read_text())
    bundle_manifest['source_selection']['path'] = publication_paths.STRUCTURE_SELECTION.as_posix()
    manifest_path.write_text(json.dumps(bundle_manifest))
    for source in [Path(renderer.__file__), renderer.palette.PATH, renderer.checkpoint_labels.PATH, publication_paths.PATH]:
        shutil.copyfile(source, data.parent.parent / source.name)
    (data / 'figure_validation.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(renderer, 'REPO', tmp_path)
    monkeypatch.setattr(renderer, 'DATA', data)
    monkeypatch.setattr(renderer, 'SUPPLEMENT_DATA', data.parent / 'ncs')
    monkeypatch.setattr(renderer, 'OUT', tmp_path / publication_paths.PDF_DIRECTORY)
    for name in ['HASHES', 'EXTRACTION_HASHES']:
        monkeypatch.setattr(renderer, name, {})
    for name in ['CHECKS', 'CONTRACTS']:
        monkeypatch.setattr(renderer, name, [])
    monkeypatch.setattr(renderer, 'EXPECTED_HASHES', expected)
    return tmp_path, data


def test_full_render_uses_public_tables_without_results_and_routes_supplements(isolated_renderer):
    root, data = isolated_renderer
    before = {name: (data / name).read_bytes() for name in ['clean-interaction-source.csv', 'position-support-source.csv']}
    renderer.main_figures()
    renderer.supplementary_figures()
    assert not (root / 'results').exists()
    assert len(renderer.CHECKS) == 8
    for check in renderer.CHECKS:
        supplement = not check['figure'].startswith('Fig')
        for kind in ['pdf', 'svg', 'plotted_source']:
            path = root / check[f'{kind}_path']
            assert renderer.digest(path) == check[f'{kind}_sha256']
            if kind == 'pdf':
                assert path == root / publication_paths.figure_pdf(check['figure'])
            else:
                assert path.parent == (data.parent / 'ncs' if supplement else data)
                content = path.read_bytes()
                assert b'\r' not in content
                assert all(line == line.rstrip() for line in content.splitlines())
        assert check['panel_labels']
        relative = Path(check['plotted_source_path'])
        pd.testing.assert_frame_equal(pd.read_csv(root / relative), pd.read_csv(ROOT / relative))
    assert {name: (data / name).read_bytes() for name in before} == before
    assert not (data / 'native-comparators.svg').exists()


def test_isolated_public_cli_copies_shared_helper_and_binds_auxiliary_outputs(isolated_renderer):
    root, data = isolated_renderer
    retained = json.loads((data / 'figure_validation.json').read_text())
    inset_manifest = str((data.parent / 'structure-insets/manifest.json').relative_to(root))
    retained.setdefault('extraction_source_sha256', {})[inset_manifest] = '0' * 64
    (data / 'figure_validation.json').write_text(json.dumps(retained))
    subprocess.run([sys.executable, str(data.parent.parent / 'render_figures.py')],
                   cwd=root, check=True, capture_output=True, text=True)
    report = json.loads((data / 'figure_validation.json').read_text())
    assert not (root / 'results').exists()
    assert not (root / 'data').exists()
    assert not (root / publication_paths.ASSETS).exists()
    assert not (root / 'manuscript/figures').exists()
    assert 'scripts/capability/reporting/publication_paths.py' in report['renderer_sha256']
    assert 'scripts/capability/reporting/checkpoint_labels.py' in report['renderer_sha256']
    assert len(report['auxiliary_outputs_sha256']) == 2
    for path, sha in report['auxiliary_outputs_sha256'].items():
        assert renderer.digest(root / path) == sha
    insets = [path for path in report['source_sha256'] if '/structure-insets/' in path]
    assert len(insets) == 4
    assert not set(insets) & set(report['extraction_source_sha256'])
    assert report['source_sha256'][inset_manifest] == renderer.digest(root / inset_manifest)


def test_structure_bundle_missing_or_changed_is_refused(isolated_renderer):
    _, data = isolated_renderer
    directory = data.parent / 'structure-insets'
    manifest_path = directory / 'manifest.json'
    original = manifest_path.read_bytes()
    manifest_path.unlink()
    with pytest.raises(ValueError, match='Missing structure-inset bundle'):
        renderer.structure_insets()
    manifest = json.loads(original)
    changed = json.loads(original)
    changed['images']['1ufm-pair']['sha256'] = '0' * 64
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='Structure-inset digest mismatch'):
        renderer.structure_insets()
    manifest_path.write_bytes(original)
    image = directory / manifest['images']['1ufm-pair']['file']
    content = image.read_bytes()
    image.unlink()
    with pytest.raises(ValueError, match='Missing structure-inset image'):
        renderer.structure_insets()
    image.write_bytes(content + b'tampered')
    with pytest.raises(ValueError, match='Structure-inset digest mismatch'):
        renderer.structure_insets()


def test_changed_csv_is_refused_before_render(isolated_renderer):
    _, data = isolated_renderer
    path = data / 'clean-interaction-source.csv'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Source-table digest mismatch'):
        renderer.clean_interactions()


def test_explicit_position_refresh_refuses_changed_fit_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, 'REPO', tmp_path)
    panel = tmp_path / 'results/R1/position_terms_20260926/position_simultaneous.json'
    panel.parent.mkdir(parents=True)
    fit = tmp_path / 'fit.json'
    fit.write_text('{}')
    panel.write_text(json.dumps({'source_sha256': {'fit.json': '0' * 64}, 'arms': []}))
    with pytest.raises(ValueError, match='Position fit digest mismatch'):
        renderer.position_support(refresh=True)


def receipt_fixture(root) -> tuple[Path, dict]:
    source = root / 'input.csv'
    source.write_text('value\n1\n')
    script = root / 'render.py'
    script.write_text('# fixture renderer\n')
    check = dict(figure='fixture', width_inches=5.14, height_inches=2,
                 minimum_font_points=7, panel_letter_points=8,
                 panel_labels=[{'svg_id': 'panel-a', 'text': 'a'}],
                 all_retained_intervals_inside_axes=True, all_text_inside_canvas=True)
    for kind in ['pdf', 'svg', 'plotted_source']:
        output = root / (kind + '.fixture')
        output.write_text(kind)
        check[f'{kind}_path'] = output.name
        check[f'{kind}_sha256'] = validator.digest(output)
    payload = dict(schema='figure_render_validation_v1', checks=[check],
                   source_sha256={source.name: validator.digest(source)},
                   renderer_sha256={script.name: validator.digest(script)})
    path = root / 'receipt.json'
    path.write_text(json.dumps(payload))
    return path, payload


def framework_receipt_fixture(root):
    path, report = receipt_fixture(root)
    check = report['checks'][0]
    report['schema'] = 'framework-assembly-validation-v1'
    report['renderer_path'], report['renderer_sha256'] = next(iter(report['renderer_sha256'].items()))
    check.update(figure='Fig1_framework', scope='framework', inferential_intervals=False,
                 all_text_inside_panel=True, text_text_overlaps=0,
                 text_extents=[{'text': letter, 'font_points': 8} for letter in 'abcdef'])
    del check['panel_labels']
    del check['all_retained_intervals_inside_axes']
    for relative in [publication_paths.FRAMEWORK_SOURCE.as_posix(), publication_paths.STRUCTURE_SELECTION.as_posix()]:
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('fixture source')
        report['source_sha256'][relative] = validator.digest(source)
    pdf = root / publication_paths.figure_pdf('Fig1_framework')
    pdf.parent.mkdir(parents=True)
    pdf.write_text('fixture PDF')
    check.update(pdf_path=str(pdf.relative_to(root)), pdf_sha256=validator.digest(pdf))
    svg = root / check['svg_path']
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg">' + ''.join(
        f'<g id="text-{letter}"><text style="font-size: 8px; font-weight: 700">{letter}</text></g>'
        for letter in 'abcdef') + '</svg>')
    check['svg_sha256'] = validator.digest(svg)
    preview = root / 'preview.png'
    preview.write_text('fixture preview')
    check.update(png_path=preview.name, png_sha256=validator.digest(preview))
    path.write_text(json.dumps(report))
    return path, report


def test_receipt_refuses_changed_inputs_outputs_and_missing_framework(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, 'REPO', tmp_path)
    path, payload = receipt_fixture(tmp_path)
    failures = []
    assert validator.load_render_manifest(path, failures)
    assert failures == []
    files = ['input.csv', 'render.py', 'pdf.fixture', 'svg.fixture', 'plotted_source.fixture']
    for name in files:
        output = tmp_path / name
        original = output.read_bytes()
        output.write_bytes(original + b'changed')
        failures = []
        validator.load_render_manifest(path, failures)
        assert failures, name
        output.write_bytes(original)
    failures = []
    assert validator.load_render_manifest(tmp_path / 'framework_validation.json', failures) == []
    assert 'Missing render receipt' in failures[0]
    del payload['checks'][0]['all_text_inside_canvas']
    path.write_text(json.dumps(payload))
    failures = []
    validator.load_render_manifest(path, failures)
    assert any('Unverified all_text_inside_canvas' in failure for failure in failures)
    payload['renderer_sha256'] = '0' * 64
    path.write_text(json.dumps(payload))
    failures = []
    assert validator.load_render_manifest(path, failures) == []
    assert any('path-to-digest map' in failure for failure in failures)

    path, report = framework_receipt_fixture(tmp_path)
    failures = []
    normalized = validator.load_render_manifest(path, failures)
    assert failures == []
    assert [panel['text'] for panel in normalized[0]['panel_labels']] == list('abcdef')
    assert normalized[0]['interval_check_status'] == 'not_applicable_conceptual_framework'
    assert 'all_retained_intervals_inside_axes' not in normalized[0]
    assert json.loads(path.read_text()) == report  # Normalization never rewrites a receipt.
    reference = tmp_path / publication_paths.FRAMEWORK_SOURCE
    reference.write_text('authorized new reference still needs a new receipt')
    failures = []
    validator.load_render_manifest(path, failures)
    assert any('source_sha256 input: manuscript/assets/framework-original.png' in failure for failure in failures)
    report['checks'][0]['inferential_intervals'] = True
    path.write_text(json.dumps(report))
    failures = []
    assert validator.load_render_manifest(path, failures) == []
    assert any('no inferential intervals' in failure for failure in failures)


def test_placed_7_8_typography_is_measured_with_explicit_panel_ids():
    content = '''<svg xmlns="http://www.w3.org/2000/svg" width="370.08pt" height="144pt">
    <g id="panel-a"><text style="font-size: 8px; font-weight: 700" x="1" y="10" transform="rotate(0 1 10)">a</text></g>
    <text style="font-size: 7px" x="30" y="40" transform="rotate(0 30 40)">a</text>
    </svg>'''
    check = {'panel_labels': [{'svg_id': 'panel-a', 'text': 'a'}]}
    scale = validator.placed_scale(r'width=\textwidth', 370.08, 144)
    assert validator.svg_typography(content, check, scale) == [8, 7]
    # Heatmap/image transforms cannot scale live text and must not trip type checks.
    image = '<image transform="matrix(8 0 0 8 1 1)" />'
    assert validator.svg_typography(content.replace('</svg>', image + '</svg>'), check, scale) == [8, 7]
    with pytest.raises(ValueError, match='Unmeasured SVG text transform'):
        validator.svg_typography(content.replace('<g id="panel-a">', '<g id="panel-a" transform="scale(.5)">'), check, scale)
    assert validator.layout_faults(content, 370.08, 144)[1] == []
    mono = content.replace('font-size: 7px', "font-size: 7px; font-family: 'DejaVu Sans Mono'")
    assert validator.layout_faults(mono, 370.08, 144)[1] == []
    unavailable = content.replace('font-size: 7px', "font-size: 7px; font-family: 'Unavailable fixture font'")
    with pytest.raises(ValueError):
        validator.layout_faults(unavailable, 370.08, 144)
    for invalid in [content.replace('7px', '6px'), content.replace('7px', '9px'),
                    content.replace('8px', '7px'), content.replace('font-weight: 700', 'font-weight: 400')]:
        with pytest.raises(ValueError):
            validator.svg_typography(invalid, check, scale)
    with pytest.raises(ValueError, match='target at insertion'):
        validator.svg_typography(content, check, validator.placed_scale(r'width=.5\textwidth', 370.08, 144))
    assert validator.placed_scale(r'width=\textwidth,height=.1\textheight,keepaspectratio', 370.08, 144) == pytest.approx(.1 * validator.TEXT_HEIGHT_PT / 144)
    for option in [r'width=50pt', r'width=\textwidth,scale=.5', r'width=\textwidth,height=.1\textheight']:
        with pytest.raises(ValueError):
            validator.placed_scale(option, 370.08, 144)


def test_new_deliberate_section_structure_and_seven_results():
    text = r'\section{Results}' + r'\subsection{Result}' * 7
    text += r'\section{Discussion}\section{Limitations}\section{Methods}\subsection{Method}'
    sections = validator.arguments(text, 'section')
    subsections = {'Results': 7, 'Discussion': 0, 'Limitations': 0, 'Methods': 1}
    assert validator.section_faults(sections, subsections) == []
    assert validator.section_faults([s for s in sections if s != 'Limitations'], subsections)
    # The count is exact in both directions: the generative-guidance section is
    # the seventh, and neither six nor eight titles may pass.
    assert validator.section_faults(sections, dict(subsections, Results=6))
    assert validator.section_faults(sections, dict(subsections, Results=8))
    # Concrete support counts remain retained data, not specimen uncertainty.
    table = pd.DataFrame({'arm': ['x'], 'assays': [3], 'source_path': ['fit.json'],
                          'source_sha256': ['0' * 64], 'source_pointer': ['assays']})
    renderer.validate_source_rows(table, 1, ['assays'])
    with pytest.raises(ValueError, match='concrete support'):
        renderer.validate_source_rows(table.assign(assays=3.5), 1, ['assays'])
