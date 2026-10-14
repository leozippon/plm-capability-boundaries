"""The author raster is preserved, never certified as live summary typography."""
import base64
import json
import sys
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

from scripts.capability.reporting import render_framework as assembler
from scripts.capability.reporting import render_framework_structures as structures
from scripts.capability.reporting import validate_manuscript as validator
from scripts.capability.reporting import publication_paths


@pytest.fixture
def composite(tmp_path, monkeypatch):
    out = tmp_path / 'scripts/capability/reporting/figure_data/main'
    out.mkdir(parents=True)
    assets = tmp_path / publication_paths.ASSETS
    assets.mkdir(parents=True)
    supplied = tmp_path / publication_paths.FRAMEWORK_SOURCE
    Image.new('RGB', (512, 288), '#345678').save(supplied)
    (assets / 'selection.json').write_text(json.dumps({'schema': 'framework-pinned-structures-v1'}))
    palette = tmp_path / 'palette.py'
    palette.write_text('# fixture palette\n')
    # A renderer digest is still required; the immutable production script need
    # not be copied because the main() path is bound through its own __file__.
    script = tmp_path / 'render.py'
    script.write_text('# fixture renderer\n')
    monkeypatch.setattr(assembler, '__file__', str(script))
    for key, value in [('ROOT', tmp_path), ('OUT', out), ('ASSETS', assets),
                       ('REFERENCE', supplied)]:
        monkeypatch.setattr(assembler, key, value)
    monkeypatch.setattr(assembler.palette, 'PATH', palette)
    helper = tmp_path / 'publication_paths.py'
    helper.write_text('# fixture publication paths\n')
    monkeypatch.setattr(publication_paths, 'PATH', helper)
    texts = ['Mutation ranking\n14/33 positive at all 3 splits\nBeyond local/profile controls',
             'Likelihood terms\n6 site + 6 downstream gains\nNo joint-over-full gain',
             'Phenotype transfer\nAbundance ranking: 14/33\nStability: 2/33 simultaneous',
             'Adjusted interactions\n0/33 positive; 1 negative\nClean two-site comparisons',
             'Context / generalization\n2/4 beat both references\nRemote transfer not established',
             'Complete-domain generation\n3 positive / 13 negative / 4 unresolved\n20 cells; 3 streams; exploratory t']
    monkeypatch.setattr(assembler, 'summary_rows', lambda: ([dict(panel=p, displayed_value=t) for p, t in zip('abcdef', texts)], []))
    monkeypatch.setattr(sys, 'argv', ['render_framework.py'])
    monkeypatch.setattr(validator, 'REPO', tmp_path)
    before = supplied.read_bytes()
    assembler.main()
    assert before == supplied.read_bytes()
    receipt = out / 'framework_validation.json'
    return tmp_path, receipt, json.loads(receipt.read_text())


def test_native_embedding_identity_and_honest_summary_scope(composite):
    root, receipt, report = composite
    check = report['checks'][0]
    assert check['pdf_path'] == publication_paths.figure_pdf('Fig1_framework').as_posix()
    assert check['raster']['native_dimensions'] == [512, 288]
    assert check['raster']['effective_dpi'] == pytest.approx(512 / assembler.WIDTH)
    assert check['raster']['native_below_450dpi'] is True
    assert check['preview_dpi'] == 450
    assert check['svg_text_editable'] == 'generated summary only'
    assert check['not_certified_scope'] == assembler.LIMITATION
    assembler.verify_embedding(root / check['svg_path'], assembler.REFERENCE)
    failures = []
    checks = validator.load_render_manifest(receipt, failures)
    assert failures == []
    assert checks[0]['interval_check_status'] == 'not_applicable_summary_counts'
    content = (root / check['svg_path']).read_text()
    sizes = validator.svg_typography(content, checks[0], 1)
    assert sizes.count(8) == 6 and sizes.count(7) == 18
    width, height = 72 * check['width_inches'], 72 * check['height_inches']
    assert validator.layout_faults(content, width, height) == (24, [])
    for kind in ['svg', 'plotted_source']:
        raw = (root / check[kind + '_path']).read_bytes()
        assert b'\r' not in raw
        assert all(line == line.rstrip() for line in raw.splitlines())


@pytest.mark.parametrize('change', ['dpi', 'dimensions', 'scope', 'pixel_identity', 'placement', 'editable', 'selection', 'intervals'])
def test_composite_receipt_refuses_false_metadata(composite, change):
    _, path, report = composite
    check = report['checks'][0]
    if change == 'dpi':
        check['raster']['effective_dpi'] = 450
    elif change == 'dimensions':
        check['raster']['native_dimensions'] = [2300, 1200]
    elif change == 'scope':
        check['not_certified_scope'] = 'all text certified'
    elif change == 'pixel_identity':
        check['raster']['pixel_identity'] = False
    elif change == 'placement':
        check['raster']['placement_points'][0] = 1
    elif change == 'editable':
        check['svg_text_editable'] = True
    elif change == 'selection':
        check['separate_structure_selection_path'] = 'manuscript/figures/framework-assets/selection.json'
    else:
        check['inferential_intervals'] = True
    path.write_text(json.dumps(report))
    failures = []
    assert validator.load_render_manifest(path, failures) == []
    assert failures


def test_changed_embedding_refused_even_with_refreshed_output_digest(composite):
    root, path, report = composite
    check = report['checks'][0]
    svg = root / check['svg_path']
    tree = ET.fromstring(svg.read_text())
    image = next(n for n in tree.iter() if n.tag.rsplit('}', 1)[-1] == 'image')
    import io
    other = io.BytesIO()
    Image.new('RGB', (512, 288), '#abcdef').save(other, format='PNG')
    image.set('{http://www.w3.org/1999/xlink}href', 'data:image/png;base64,' + base64.b64encode(other.getvalue()).decode())
    svg.write_text(ET.tostring(tree, encoding='unicode'))
    check['svg_sha256'] = assembler.sha(svg)
    path.write_text(json.dumps(report))
    failures = []
    assert validator.load_render_manifest(path, failures) == []
    assert any('original PNG bytes' in failure for failure in failures)


def test_changed_author_source_requires_deliberate_rerender(composite):
    _, path, _ = composite
    Image.new('RGBA', (512, 288), '#abcdef').save(assembler.REFERENCE)
    failures = []
    assert validator.load_render_manifest(path, failures) == []
    assert failures
    assembler.main()
    failures = []
    assert validator.load_render_manifest(path, failures)
    assert not failures


def test_optional_auxiliary_outputs_are_digest_bound(composite):
    root, path, report = composite
    auxiliary = root / publication_paths.CHECKPOINT_KEY
    auxiliary.write_text('checkpoint name key\n')
    report['auxiliary_outputs_sha256'] = {publication_paths.CHECKPOINT_KEY.as_posix(): assembler.sha(auxiliary)}
    path.write_text(json.dumps(report))
    failures = []
    assert validator.load_render_manifest(path, failures)
    assert not failures
    auxiliary.write_text('changed key\n')
    validator.load_render_manifest(path, failures)
    assert any('auxiliary_outputs_sha256' in failure for failure in failures)


def test_separate_structure_output_location():
    assert structures.ASSETS == structures.ROOT / publication_paths.ASSETS


def test_retained_summary_counts_use_checkpoint_split_support():
    ledger = assembler.ROOT / 'results/shared/manuscript_evidence_20261004/derived-numbers.csv'
    if not ledger.exists():
        pytest.skip('Retained local research artifacts are not distributed')
    rows, sources = assembler.summary_rows()
    assert [r['count'] for r in rows] == [14, '[6, 6, 0]', 14, '[2, 33]', '0;1;33', 2, '3;13;4;20;3']
    assert all(r['source_pointer'] and r['source_sha256'] and r['upstream_sources'] for r in rows)
    assert sources
