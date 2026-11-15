"""Operational names resolve uniquely; historical recipes remain recoverable, and dated analysis entries locate
the repository from their own path."""
import ast
import json
import os
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


# Dated analysis entries run their analysis at import, so this test never executes their bodies: it copies them into an
# isolated checkout and executes only the bootstrap prefix, everything before the first data read. The stubbed
# first-party modules exist only to show which checkout an import resolves to; they compute nothing, and no data is
# read or written. `analyse.py` is not covered: it imports no first-party module, and its caller-relative table paths
# select the twelve- and eighteen-shard result directories; its bytes are also the frozen producer recorded in
# results/extensions/objective_tuned_five_mer_rule_20261114/five_mer_rule.json.
BOOTSTRAP_ENTRIES = {
    'long_cap_termination_20261110/analyse_companion.py': ('ROOT', 'H'),
    'long_cap_termination_20261110/capped_identity_adjusted.py': ('ROOT', 'H'),
    'long_cap_termination_20261110/mid_natural_contrast.py': ('ROOT', 'H'),
    'long_cap_termination_20261110/stage_primary_support.py': ('ROOT', 'SRC', 'OUT'),
    'length_adjusted_contrasts_20261110/identity_shift_share.py': ('ROOT', 'D'),
    'length_adjusted_contrasts_20261110/identity_family_a_without_thin_cells.py': ('ROOT', 'D'),
    'length_adjusted_contrasts_20261110/identity_stratified_contrast.py': ('ROOT', 'D'),
}

STUB_MODULES = {
    'src/__init__.py': '',
    'src/capability/__init__.py': '',
    'src/capability/context/__init__.py': '',
    'src/capability/context/homology.py': "STRATUM_NAMES = ('[0,30)', '[30,70)', '[70,95)', '[95,100]')\n",
    'src/capability/generation/__init__.py': '',
    'src/capability/generation/objective_tuned_analysis.py': 'def shift_share(base, tuned):\n    return {}\n',
}

BOOTSTRAP_PROBE = '''
import json, sys
script, names = sys.argv[1], sys.argv[2:]
lines = open(script).read().splitlines(keepends=True)
cut = next((i for i, line in enumerate(lines)
            if any(token in line for token in ('read_csv(', 'open(', 'glob.glob('))), None)
if cut is None:
    raise SystemExit('no data-read boundary in ' + script)
namespace = {'__file__': script, '__name__': 'bootstrap'}
exec(compile(''.join(lines[:cut]), script, 'exec'), namespace)
print(json.dumps({'roots': {name: str(namespace[name]) for name in names},
                  'first_party': sorted(getattr(module, '__file__', '') for name, module in sys.modules.items()
                                        if name.split('.')[0] == 'src' and name != 'src')}))
'''


def _stubs(root):
    for name, body in STUB_MODULES.items():
        stub = root / name
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text(body)


def _checkout(tmp_path, first_party):
    checkout = tmp_path / 'checkout'
    for entry in BOOTSTRAP_ENTRIES:
        target = checkout / 'scripts/capability/generation' / entry
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / 'scripts/capability/generation' / entry).read_text())
    if first_party:
        _stubs(checkout)
    return checkout


def _caller(tmp_path, decoy):
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir(parents=True)
    if decoy:
        _stubs(elsewhere)
    return elsewhere


def _bootstrap(script, cwd, names, isolated=False):
    env = {k: v for k, v in os.environ.items() if k not in {'PYTHONPATH', 'RESDIR'}}
    command = [sys.executable] + (['-I'] if isolated else []) + ['-c', BOOTSTRAP_PROBE, str(script), *names]
    return subprocess.run(command, cwd=str(cwd), env=env, text=True, capture_output=True)


@pytest.mark.parametrize('entry,names', sorted(BOOTSTRAP_ENTRIES.items()))
def test_dated_analysis_entries_derive_their_root_from_their_own_path(tmp_path, entry, names):
    source = (ROOT / 'scripts/capability/generation' / entry).read_text()
    code = ''.join(source.splitlines(keepends=True)[ast.parse(source).body[0].end_lineno:])  # docstring names the host interpreter
    assert '/Data/lzp' not in code and '/home/' not in code

    # A caller's working directory carries its own first-party package, so an entry that does not insert its own root
    # first reads the caller's tree, or one named by an absolute path, instead of the checkout it sits in.
    checkout = _checkout(tmp_path / 'configured', first_party=True)
    run = _bootstrap(checkout / 'scripts/capability/generation' / entry, _caller(tmp_path / 'configured', decoy=True), names)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    resolved = checkout.resolve()
    for name in names:
        assert Path(report['roots'][name]).resolve().is_relative_to(resolved), (name, report['roots'][name])
    for module_file in report['first_party']:
        assert Path(module_file).resolve().is_relative_to(resolved), module_file

    # Without its own first-party package the entry fails on that import rather than reaching another tree.
    if not report['first_party']:
        return
    bare = _checkout(tmp_path / 'bare', first_party=False)
    missing = _bootstrap(bare / 'scripts/capability/generation' / entry, _caller(tmp_path / 'bare', decoy=False), names, isolated=True)
    assert missing.returncode != 0
    assert "No module named 'src'" in missing.stderr, missing.stderr
