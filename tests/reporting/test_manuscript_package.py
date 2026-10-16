"""The manuscript package carries every module its entry points import.

A packaged entry point that imports a module the package does not hold fails on
import and reproduces nothing, which is how two ledger source mappings went
missing: ``build_derived_numbers.build`` imports its emitters inside the
function, so nobody maintaining the package by hand saw them appear.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.capability.reporting import build_manuscript_package as packager

ANALYSIS = 'code/analysis/scripts/capability'


def staged_package() -> Path:
    if not (packager.PACKAGE / 'code/analysis').is_dir():
        pytest.skip('the manuscript package is not staged in this checkout')
    return packager.PACKAGE


def test_package_carries_every_module_the_entry_points_import():
    package = staged_package()
    derived, _ = packager.derived_code()
    assert packager.missing_modules(derived, package) == []


def test_closure_reaches_modules_imported_inside_a_function_and_beside_a_script():
    derived, _ = packager.derived_code()
    for emitter in ('potts_construction_quantities', 'progen3_robustness_quantities',
                    'coupling_guidance_quantities', 'phenotype_quantities',
                    'propagation_quantities', 'current_manuscript_quantities',
                    'followup_quantities'):
        assert f'{ANALYSIS}/reporting/{emitter}.py' in derived
    assert f'{ANALYSIS}/generation/analyse_selection_factorial.py' in derived


def test_a_module_the_package_lacks_or_has_moved_on_from_is_named(tmp_path):
    derived, _ = packager.derived_code()
    for path, source in derived.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((packager.REPO_ROOT / source).read_bytes())
    absent = f'{ANALYSIS}/reporting/progen3_robustness_quantities.py'
    stale = f'{ANALYSIS}/reporting/build_derived_numbers.py'
    (tmp_path / absent).unlink()
    (tmp_path / stale).write_text('# a copy the repository has moved on from\n')

    faults = packager.missing_modules(derived, tmp_path)
    assert sorted(fault.split(':')[0] for fault in faults) == sorted([absent, stale])
    assert 'not packaged' in next(fault for fault in faults if fault.startswith(absent))
    assert 'differs from' in next(fault for fault in faults if fault.startswith(stale))


def test_library_dependency_surface_is_a_recorded_decision():
    _, library = packager.derived_code()
    assert library == set(packager.LIBRARY_CARRIED) | set(packager.LIBRARY_REFERENCED)


def test_closure_stops_at_the_library_it_does_not_carry(tmp_path, monkeypatch):
    reporting = tmp_path / 'scripts/capability/reporting'
    reporting.mkdir(parents=True)
    library = tmp_path / 'src/capability/core'
    library.mkdir(parents=True)
    (library / 'evidence_ledger.py').write_text('from ..core.io import read\n')
    (library / 'io.py').write_text('read = None\n')
    (reporting / 'helper.py').write_text('def emit(): pass\n')
    (reporting / 'sibling.py').write_text('records = ()\n')
    (reporting / 'entry.py').write_text(
        'import json\n'
        'import sibling\n'
        'from src.capability.core.evidence_ledger import Ledger\n'
        'def build():\n'
        '    from scripts.capability.reporting.helper import emit\n'
        '    return emit, sibling.records, Ledger\n')
    monkeypatch.setattr(packager, 'REPO_ROOT', tmp_path)

    scripts, surface = packager.closure(('scripts/capability/reporting/entry.py',))
    assert scripts == {'scripts/capability/reporting/entry.py',
                       'scripts/capability/reporting/helper.py',
                       'scripts/capability/reporting/sibling.py'}
    assert surface == {'src/capability/core/evidence_ledger.py'}


def test_adapter_stages_only_modules_the_package_carries():
    package = staged_package()
    derived, _ = packager.derived_code()
    for name in packager.adapter_modules(package):
        flat = f'code/{name}'
        assert flat in derived or flat in packager.PACKAGE_ONLY_CODE
        assert (package / flat).is_file()
