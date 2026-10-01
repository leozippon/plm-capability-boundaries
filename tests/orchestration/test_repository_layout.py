"""Operational names resolve uniquely; historical recipes remain recoverable."""
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.capability.campaigns import ARCHIVE, records

ROOT = Path(__file__).resolve().parents[2]


def resolve(snapshot, name):
    return subprocess.run([sys.executable, str(ROOT / 'scripts/capability/entrypoints.py'), '--root', str(snapshot), name],
        text=True, capture_output=True)


def test_grouped_and_frozen_legacy_entrypoints_are_unambiguous(tmp_path):
    grouped = tmp_path / 'scripts/capability/stages'
    grouped.mkdir(parents=True)
    stage = grouped / 'cohort_power.py'
    stage.write_text('pass\n')
    assert resolve(tmp_path, '01_cohort_power.py').stdout.strip() == str(stage)
    assert resolve(tmp_path, 'cohort_power.py').stdout.strip() == str(stage)
    (grouped.parent / '01_cohort_power.py').write_text('pass\n')
    assert resolve(tmp_path, '01_cohort_power.py').returncode != 0
    stage.unlink()
    assert resolve(tmp_path, '01_cohort_power.py').stdout.strip().endswith('/01_cohort_power.py')
    assert resolve(tmp_path, 'cohort_power.py').stdout.strip().endswith('/01_cohort_power.py')
    assert resolve(tmp_path, '../escape.py').returncode != 0
    assert resolve(tmp_path, 'missing.py').returncode != 0


def test_historical_manifest_extract_is_exact_and_never_overwrites(tmp_path):
    if not ARCHIVE.is_file():
        pytest.skip("the historical recipe archive is host-local under h200/ and is not distributed")
    archive = records()
    assert len(archive) == 354
    name = 'campaign_s48_unconditional_generation.tsv'
    output = tmp_path / name
    command = [sys.executable, str(ROOT / 'scripts/capability/campaigns.py'), 'extract', name, '--out', str(output)]
    subprocess.run(command, check=True, capture_output=True)
    assert output.read_text() == archive[name]['content']
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert set(archive[name]) == {'original_path', 'content'}
