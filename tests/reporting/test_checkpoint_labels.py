"""Plot aliases, public name keys and verdict strips use one retained source."""
from pathlib import Path
import csv
import io
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.capability.reporting import checkpoint_labels as labels
from scripts.capability.reporting import render_figures as renderer


def test_alias_coverage_uniqueness_and_condition_identity():
    assert set(labels.CHECKPOINTS) == set(renderer.ORDER) | {'zymctrl'}
    aliases = [labels.label(arm) for arm in labels.CHECKPOINTS]
    assert len(aliases) == len(set(aliases)) == 34
    assert max(map(len, aliases)) <= 11
    assert labels.label('prollama') != labels.label('prollama-stage-1')
    assert labels.label('qwen2.5-0.5b') != labels.label('qwen2.5-0.5b-instruct')
    assert labels.label('prollama__requested') == labels.label('prollama') + '†'
    assert labels.label('zymctrl__requested') == 'ZymCTRL†'
    with pytest.raises(KeyError):
        labels.label('unknown-checkpoint')
    for path in renderer.DATA.glob('*source.csv'):
        table = pd.read_csv(path)
        if 'arm' in table:
            assert set(table.arm.dropna()) - {'profile_sampler'} <= set(labels.CHECKPOINTS)


def test_generated_keys_are_consistent_with_single_mapping(tmp_path):
    csv_path, tex_path = labels.write_key(tmp_path)
    rows = list(csv.DictReader(io.StringIO(csv_path.read_text())))
    assert [(r['checkpoint_id'], r['short_label'], r['full_name']) for r in rows] == [
        (arm, short, full) for arm, (short, full) in labels.CHECKPOINTS.items()]
    assert tex_path == tmp_path / 'manuscript/submission/checkpoint-name-key.tex'
    tex = tex_path.read_text()
    assert '\\begin{table}[!htbp]' in tex
    assert '\\label{tab:checkpoint-key}' in tex
    assert 'requested-condition' in tex and 'unconditioned' in tex
    for _, (short, full) in labels.CHECKPOINTS.items():
        assert f'{short} & {full} \\\\' in tex
    for path, expected in [(csv_path, labels.key_csv()), (tex_path, labels.key_tex())]:
        assert path.read_text() == expected
        assert b'\r' not in path.read_bytes()
        assert all(line == line.rstrip() for line in path.read_bytes().splitlines())
    assert (ROOT / csv_path.relative_to(tmp_path)).read_text() == labels.key_csv()
    assert (ROOT / tex_path.relative_to(tmp_path)).read_text() == labels.key_tex()


def test_verdict_strip_counts_retained_intervals_without_inference():
    table = pd.read_csv(renderer.DATA / 'generation-streams-replot.csv')
    selected = table[table.endpoint == 'complete_domain']
    assert len(selected) == 20
    assert renderer.generation_verdict_counts(selected) == (3, 13, 4)
    boundaries = pd.DataFrame({'ci_low': [0, -1, 1], 'ci_high': [1, 0, 2]})
    assert renderer.generation_verdict_counts(boundaries) == (1, 0, 2)
