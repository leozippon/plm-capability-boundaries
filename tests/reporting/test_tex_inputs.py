"""Local input files participate in reference and freshness checks."""
import pytest

from scripts.capability.reporting.validate_manuscript import arguments, float_placement_faults, tex_inputs


def test_extensionless_nested_inputs_expose_labels_and_dependencies(tmp_path):
    (tmp_path / 'notes.tex').write_text(r'\input{key.tex}')
    key = tmp_path / 'key.tex'
    key.write_text(r'\label{tab:checkpoint-key}')
    text = r'\ref{tab:checkpoint-key}\input{notes}'
    dependencies = tex_inputs(text, tmp_path)
    assert [path.name for path, _ in dependencies] == ['notes.tex', 'key.tex']
    expanded = text + ''.join(content for _, content in dependencies)
    assert set(arguments(expanded, 'ref')) <= set(arguments(expanded, 'label'))
    assert key in [path for path, _ in dependencies]


def test_repeated_input_does_not_hide_duplicate_labels(tmp_path):
    (tmp_path / 'key.tex').write_text(r'\label{tab:key}')
    dependencies = tex_inputs(r'\input{key}\input{key.tex}', tmp_path)
    labels = arguments(''.join(content for _, content in dependencies), 'label')
    assert labels == ['tab:key', 'tab:key']
    assert tex_inputs('% \\input{missing}\n', tmp_path) == []


def test_float_placement_accepts_flexible_and_default_options():
    for placement in ('[!htbp]', '[tbp]', '[hp]', ''):
        for environment in ('figure', 'table', 'figure*', 'table*'):
            assert float_placement_faults('\\begin{' + environment + '}' + placement) == []
    assert float_placement_faults('% \\begin{figure}[!t]\n') == []


def test_float_placement_refuses_text_only_and_float_only_options():
    for text in (r'\begin{figure}[!t]', r'\begin{table}[p]', r'\begin{figure}[H]'):
        assert len(float_placement_faults(text)) == 1


def test_float_placement_checks_expanded_local_inputs(tmp_path):
    (tmp_path / 'key.tex').write_text(r'\begin{table}[p]')
    text = r'\input{key}'
    expanded = text + ''.join(content for _, content in tex_inputs(text, tmp_path))
    assert float_placement_faults(expanded)


def test_invalid_input_refuses_instead_of_being_ignored(tmp_path):
    with pytest.raises(FileNotFoundError):
        tex_inputs(r'\input{missing}', tmp_path)
    (tmp_path / 'cycle.tex').write_text(r'\input{cycle}')
    with pytest.raises(ValueError, match='Cyclic'):
        tex_inputs(r'\input{cycle}', tmp_path)
    with pytest.raises(ValueError, match='Nonlocal'):
        tex_inputs(r'\input{../outside}', tmp_path)
